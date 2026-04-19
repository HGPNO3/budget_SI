# Budget SI 技术实现设计

> 目标读者：后续实现的 agent / 程序员
> 前置阅读：[`overview.md`](overview.md)（通俗版）
> 原则：**具体到文件路径和行号**，不写泛泛而谈

---

## 0. TL;DR

要做的事 = **IGPO 代码的 fork + 一个 sotopia adapter + 一处 T3 截断逻辑**。大部分重活儿 IGPO 已经写好了，我们主要工作是：

1. **Fork `IGPO/` 作为训练主干**（它已经集成了 veRL + IG reward + vectorized logprob）
2. **新写 `sotopia_rollout/` 适配层**：把 sotopia 的 episode 机制包装成 veRL 期望的 rollout 接口
3. **重定义 IG 里的 "GT"**：从 QA 答案换成"判官生成的理想结局描述"
4. **加一个 T3 trigger**：复用 IGPO per-turn IG 做 progress signal，连续 k 轮低于阈值就提前结束 episode
5. **加 budget 惩罚**：γ<1 或 turn-count penalty

---

## 1. 数据流完整图

```
╔═══════════════════════════════════════════════════════════════════╗
║                       一个 Training Step                           ║
╚═══════════════════════════════════════════════════════════════════╝

STEP A: Sample from sotopia
├─ sotopia EnvironmentProfile (Redis, 884 个场景)
├─ 筛过的场景子集 (step0_filter_scenes.py 已做)
└─→ batch of (scenario, agent_goals, characters)

STEP B: Rollout (我们要写的核心适配层)
├─ for each (scenario, goals) in batch:
│   ├─ policy_agent = Gemma-4 + LoRA (我们的模型，由 veRL 的 rollout worker 提供)
│   ├─ opponent_agent = Gemma-4 base (冻结，走 vLLM 独立实例)
│   ├─ for t in range(max_turns):
│   │   ├─ policy speaks: call local model via veRL rollout worker
│   │   │                 → tokens 进 trajectory, speaker_mask=1
│   │   ├─ opponent speaks: call frozen vLLM
│   │   │                 → tokens 进 trajectory, speaker_mask=0
│   │   ├─ [T3 check] if t>=k and all(IG_recent_k < θ): break  ← (需要先算出 IG)
│   │   └─ if episode done (env signal): break
│   └─ trajectory = {tokens, speaker_mask, turn_ids, scenario, goals}
│
└─→ batch of trajectories

STEP C: Judge scoring
├─ for each trajectory: judge LLM → goal_scores (两个 agent 的)
└─→ episode-level reward = policy's goal_score

STEP D: IG computation (vectorized, IGPO 原装)
├─ for each trajectory:
│   ├─ 准备 GT token 序列 = "理想结局描述"（judge LLM 生成一次，缓存）
│   ├─ vectorized forward: 一次算出 T 个 log P(GT | dialogue_≤t)
│   └─ per-turn IG = log P_t - log P_{t-1}
└─→ info_gain_rewards: List[List[float]]

STEP E: Token-level reward 组装 (IGPO 原装)
├─ for each policy turn t:
│   ├─ reward_tensor[last_token_of_turn_t] = IG_t
└─ reward_tensor[last_token_of_episode] = goal_score
    (opponent turns 的 reward_tensor 位置为 0，且 loss_mask=0)

STEP F: GRPO update (veRL 原装)
├─ separate group-norm: IG 一组, goal_score 一组
├─ discounted accumulation: R̃_t = Σ_{k≥t} γ^(k-t) · r̃_k  (γ<1 做 budget 惩罚)
├─ broadcast turn-level return 到 turn 内所有 policy token
└─ PPO clipped objective + KL penalty → 更新 LoRA adapter
```

---

## 2. 架构选型：用 `IGPO/` 作为训练主干

**决策**：我们**不从零搭 veRL**，直接 **fork IGPO 仓库**作为训练代码基座。

**理由**：
- IGPO 已经在 veRL 上实现了：multi-turn rollout（`scrl/llm_agent/generation.py`）、vectorized IG 计算（`vectorized_gt_logprob.py`）、turn-level advantage、reward 注入管线。
- 我们需要的所有改动几乎都是**在 IGPO 的 hook 上替换数据源**，不是算法改动。
- T3 虽然也 fork 了 veRL，但它的 BTR 逻辑轻量，移植过来比 fork T3 简单。

**具体做法**：
1. 把 `IGPO/` 下的 `verl/` 和 `scrl/` 整体复制到 `budget_SI/training/` 下作为训练主干
2. 新增 `budget_SI/training/sotopia_rollout/` 实现 sotopia 适配
3. 修改 `training/scrl/llm_agent/generation.py` 让它支持两种模式：原版搜索任务（保留）+ sotopia 对话任务（新增）

---

## 3. 组件 1：sotopia → veRL Rollout 适配层

### 3.1 新增模块结构

```
budget_SI/training/sotopia_rollout/
├── __init__.py
├── adapter.py          # 主入口：SotopiaRolloutAdapter
├── episode_runner.py   # 改造版的 arun_one_episode
├── speaker_mask.py     # token 级 speaker mask 构造
├── ideal_outcome.py    # judge LLM 生成 "理想结局"（IG 的 GT）
└── judge_reward.py     # episode-level judge 打分
```

### 3.2 核心接口：`SotopiaRolloutAdapter`

参考 IGPO 的 `LLMGenerationManager`（`scrl/llm_agent/generation.py:284-850`）的签名，实现一个对偶版本：

```python
class SotopiaRolloutAdapter:
    def __init__(
        self,
        actor_rollout_wg,          # veRL 的 rollout worker group（policy）
        opponent_llm_endpoint,     # 固定 opponent 的 vLLM URL
        judge_llm_endpoint,        # judge 的 LLM URL
        tokenizer,
        max_turns: int = 20,
        redis_url: str = "redis://localhost:6380",
    ):
        ...

    def run_episode_batch(
        self,
        scenarios: List[Dict],      # [{env_pk, agent1_pk, agent2_pk, policy_role}]
    ) -> DataProto:
        """返回 veRL 期望的 DataProto，字段见 3.4"""
```

### 3.3 episode 跑的具体流程（改造 `sotopia/server.py:arun_one_episode`）

```python
# 伪代码，放在 episode_runner.py
async def run_one_episode_for_rl(scenario, policy_role, adapter):
    env = make_sotopia_env(scenario)          # sotopia.server 原生
    trajectory_tokens = []
    speaker_mask = []
    turn_ids = []

    current_turn = 0
    while current_turn < max_turns:
        who_speaks_next = env.get_current_speaker()  # sotopia 内部决定

        # Build prompt: scenario + history + agent's role description
        prompt_text = build_sotopia_prompt(env, who_speaks_next)
        prompt_tokens = tokenizer.encode(prompt_text)

        if who_speaks_next == policy_role:
            # 调 veRL 的本地模型（actor）
            response_tokens = adapter.actor_rollout_wg.generate_sequences(
                DataProto.from_tokens(prompt_tokens)
            )
            speaker_mask.extend([1] * len(response_tokens))  # 要训
        else:
            # 调固定 opponent vLLM
            response_text = adapter.opponent_call(prompt_text)
            response_tokens = tokenizer.encode(response_text)
            speaker_mask.extend([0] * len(response_tokens))  # 不训

        trajectory_tokens.extend(response_tokens)
        turn_ids.extend([current_turn] * len(response_tokens))

        # feed back to sotopia env
        action = parse_sotopia_action(tokenizer.decode(response_tokens))
        env_messages, _, done, info = await env.astep([action])

        # T3 check (要求 IG 能在线算，否则退化为"离线截断")
        if adapter.t3_enabled and current_turn >= adapter.t3_window:
            recent_ig = adapter.compute_online_ig(trajectory_tokens, turn_ids)
            if all(ig < adapter.t3_threshold for ig in recent_ig[-adapter.t3_window:]):
                break

        current_turn += 1
        if done:
            break

    return trajectory_tokens, speaker_mask, turn_ids, env
```

### 3.4 返回的 DataProto 字段

必须符合 veRL `ray_trainer.py:1137-1413` 的 fit() 主循环期望：

```python
DataProto(
    batch={
        "input_ids":       Tensor(B, L),     # 完整对话 token（含 scenario + turns）
        "attention_mask":  Tensor(B, L),
        "position_ids":    Tensor(B, L),
        "responses":       Tensor(B, L_resp), # 剥掉初始 prompt 的部分
        "loss_mask":       Tensor(B, L_resp), # = speaker_mask（只有 policy turns 为 1）
        "turn_ids":        Tensor(B, L_resp), # IGPO 需要，每个 token 属于第几轮
    },
    non_tensor_batch={
        "scenario_pk":     List[str],
        "policy_role":     List[str],
        "agent_goals":     List[List[str]],
        "ideal_outcome":   List[str],         # judge 生成的"理想结局"（GT for IG）
        "goal_score":      List[float],       # judge 给 policy 的分（最终 reward）
        "turn_count":      List[int],         # 实际跑了几轮（算 budget 惩罚用）
    }
)
```

### 3.5 Opponent 模型部署

**决策**：opponent 用**独立 vLLM 实例**，和 policy 完全隔离。

**原因**：
- 如果共用 vLLM，policy 做 LoRA 切换会很麻烦
- opponent 只需推理，不需要参与训练管线

**部署**：
- 服务器 `ms-8h20` GPU 0：现有的 policy vLLM（端口 8001，给 actor 用）
- 新增 GPU 1：opponent vLLM（端口 8002，Gemma-4 base，无 LoRA）
- 训练占 GPU 2-3

---

## 4. 组件 2：IG reward 计算（复用 IGPO，改 GT 来源）

### 4.1 "理想结局" 的生成（ideal_outcome.py）

**时机**：每个 episode 开始前，调 judge LLM 生成一次 ideal outcome 文本，**缓存到 scenario 级别**（同一 scenario 不重复生成）。

**prompt 模板**：
```
Scenario: {scenario_text}
{policy_role} 的目标：{agent_goal}

请想象 {policy_role} **完美达成了目标**时，最终对话的结局大致会是什么样。
用 2-3 句话描述 {policy_role} 在对话结束时的**最终发言或成就**。

格式：直接输出该段文本，不要任何 meta 说明。
```

**缓存**：`{redis_url}/ideal_outcome_cache` 下按 `{scenario_pk}_{policy_role}` 为 key。

**回退方案**：如果 judge 输出不稳定（见 §6 风险），改成直接用 `agent_goal` 作为 GT（更直白但少了"结局化"的表达）。

### 4.2 IG 计算调用

**复用 IGPO 的 `vectorized_gt_logprob.py`**，不改代码，只改调用点：

原版（IGPO）在 `scrl/llm_agent/generation.py:562-620` 里，每个 sample 传入 `ground_truths[sample_i]['ground_truth']`。

我们要改的：把 `ground_truth` 替换为我们 adapter 生成的 `ideal_outcome` 文本。这只需要修改数据加载环节（`rl_dataset.py`）：

```python
# 原版 IGPO rl_dataset.py
ground_truth = item["answer"]  # QA 数据集的标准答案

# 我们的修改版
ground_truth = item["ideal_outcome"]  # 从 adapter 生成的"理想结局"
```

**IG 计算的数学细节**（参考 `overview.md §3`）：
- `r^IG_t = log P(ideal_outcome | dialogue_≤t) - log P(ideal_outcome | dialogue_≤t-1)`
- 用 IGPO 的 stop-gradient（不对 GT log prob 反传）
- 用 IGPO 的 separate group-normalize（IG 和 goal_score 分两组归一化）

### 4.3 接入 reward 管线

**IGPO 已做完**，我们只需确认配置：

`training/verl/utils/reward_score/info_gain.py:162-295` 的 `compute_score` 函数：
- 输入：`solution_str`（整个对话 token 序列）、`ground_truth`（理想结局）、`info_gain_reward`（每轮 IG 列表）
- 输出：token 级稀疏 reward tensor，每轮尾 token 有 IG，episode 尾 token 有 goal_score

**关键**：需要给 `compute_score` 传入 `loss_mask`，告诉它哪些 token 是 policy 的（只在 policy turn 尾放 reward）。IGPO 原版默认所有 turn 都是 policy 的，这里要加参数。

---

## 5. 组件 3：T3 BTR 截断

### 5.1 信号源：直接复用 IGPO 的 per-turn IG

这是 budget_SI 最优雅的一点：**IGPO 天然给出 T3 需要的 progress signal**。

- 传统 T3（`T3/verl/search_r1/tau2_adapter/space.py`）用的是"expected action coverage"，这依赖任务有 hypothesis space
- 社交对话没有 hypothesis space → 用 IG 作 proxy：连续 k 轮 IG < θ = 陷入重复/无意义循环

### 5.2 两种触发模式

#### 模式 A: 离线截断（简单，推荐 Phase 1）

Episode 跑完后，在 IG 计算完成后判断：
- 找第一个 t 使得 `all(IG[t-k+1:t+1] < θ)`
- 从 t 开始后续所有 token 的 `loss_mask = 0`（不进 loss）
- 对应 goal_score reward 放到 t 这个位置（不是 episode 结尾）

实现位置：`adapter.py` 里 episode 跑完后做的 post-process。

#### 模式 B: 在线截断（复杂，但 token 效率更高，Phase 3）

在 episode 生成循环中就检查：
- 每跑完一个 policy turn，立刻算这一轮的 IG（需要额外 forward）
- 窗口满足条件就 `break` 当前 episode

**代价**：每轮额外一次 log P 计算。但论文说开销 < 2%（vectorized 技巧），可接受。

### 5.3 阈值 θ 的选择

**采用 T3 论文 Appendix D.4 的 adaptive quantile 方法**：

```python
# 每 6 个 training step 更新一次
all_igs = collect_igs_from_recent_untruncated_episodes()
theta = np.quantile(all_igs, alpha=0.2)  # 取底部 20%
```

**回退**：如果自适应阈值震荡大，用固定值 `θ = 0`（只截断负 IG 窗口）。

### 5.4 k（窗口长度）的选择

T3 论文在 SituationPuzzle（最接近我们对话任务的 setting）推荐 **k = 5**。
起始值用 k=5，做 ablation 扫 k ∈ {3, 5, 7}。

---

## 6. 组件 4：veRL trainer 修改（最小）

### 6.1 GRPO 配置

在 `training/verl/trainer/config/ppo_trainer.yaml` 基础上改：

```yaml
algorithm:
  adv_estimator: grpo
  info_gain_norm_mode: separate     # IGPO 的关键：IG 和 outcome 分组归一化
  gamma: 0.95                        # <-- budget 惩罚：后期 IG 折价
  kl_coef: 0.001
  use_kl_in_reward: true

reward_model:
  reward_manager: naive              # 用 IGPO 的 NaiveRewardManager
  launch_reward_fn_async: false

actor_rollout_ref:
  rollout:
    name: custom_sotopia              # <-- 新写的 rollout 注册
    n: 8                              # 每个 scenario 生成 8 个 rollout（GRPO 必需）
    multi_turn:
      enable: true
      max_turns: 20

trainer:
  total_epochs: 3
  train_batch_size: 16               # B=16 scenarios × n=8 rollouts = 128 episodes / step
```

### 6.2 Budget penalty

两种实现：

**方式 1：γ < 1（推荐）**
- 改 `training/verl/trainer/ppo/core_algos.py` 的 `_compute_turn_level_advantage`
- IGPO 原版 γ=1，我们改为 γ=0.95 → 第 10 轮的 IG 被折 0.6 倍 → 模型学会早点说清楚

**方式 2：显式 turn-count penalty**
- 在 reward 组装时：`final_goal_score -= λ × turn_count`
- λ 建议 0.05 - 0.1（让 20 轮扣 1-2 分）
- 更 explicit，但需要调 λ

**建议**：Phase 1 用方式 1（IGPO 的 γ 接口已存在），Phase 3 加方式 2 验证。

---

## 7. 文件改动清单

### 新增文件

```
budget_SI/training/                         # 整体 fork from IGPO/
  sotopia_rollout/
    __init__.py
    adapter.py                              # ≈ 400 行
    episode_runner.py                       # ≈ 200 行
    speaker_mask.py                         # ≈ 100 行
    ideal_outcome.py                        # ≈ 150 行
    judge_reward.py                         # ≈ 100 行
  
  cmd/
    sotopia/
      ppo_sotopia.sh                        # ≈ 80 行训练脚本
      prepare_ideal_outcomes.py             # 预生成所有 scenario 的 ideal outcome
```

### 修改文件（from IGPO fork）

| 文件 | 改动点 | 大致行数 |
|---|---|---|
| `training/scrl/llm_agent/generation.py` | 加 `sotopia_mode` 分支，调 `SotopiaRolloutAdapter` | +150 行 |
| `training/verl/utils/dataset/rl_dataset.py` | 加 sotopia 场景数据源，`ground_truth` 改成 ideal_outcome | +80 行 |
| `training/verl/utils/reward_score/info_gain.py` | `compute_score` 加 `loss_mask` 参数 | +30 行 |
| `training/verl/trainer/ppo/core_algos.py` | 加 budget penalty（如果用方式 2） | +20 行 |
| `training/verl/trainer/main_ppo.py` | 注册 `custom_sotopia` rollout | +10 行 |

### 不改的（复用）

- veRL 所有分布式 / FSDP / Ray 相关代码
- IGPO 的 vectorized GT logprob 实现
- IGPO 的 separate group-norm
- IGPO 的 turn-level advantage broadcast
- sotopia 的场景/角色/judge 打分逻辑

---

## 8. 风险和备选方案

| 风险 | 发生概率 | 影响 | 备选 |
|---|---|---|---|
| judge LLM 生成的 ideal_outcome 不稳定，IG 信号噪音大 | 中 | 大（IG 基本失效） | 改用 `agent_goal` 文本作 GT；或用"每轮 goal score delta"作 dense reward |
| sotopia 对话 token 数太长（>8k）爆显存 | 中 | 中 | max_turns 降到 10；gradient_checkpoint；或用序列并行 |
| policy vs opponent 互动时 policy 学会"强势压制"对话退化成单人独白 | 低 | 中 | 固定 opponent 为能力相当的 base model；或定期轮换 opponent 模型 |
| veRL rollout 接口和 sotopia async 模型难兼容 | 中 | 大 | 先用 `naive_rollout`（非 async）跑通，牺牲速度；再优化 |
| 两个 agent 的 token 交错导致 loss_mask 出错，训练方向错 | 中 | 致命 | 先用 **debug episode (1 条数据)** 跑一次 forward，打印每个 token 的 `(text, speaker_mask)` 人眼检查 |
| γ=0.95 过度惩罚导致对话太短（2-3 轮就结束），goal 没达成 | 中 | 中 | 从 γ=0.99 开始，逐步调低；或用方式 2 的 λ 更好控制 |
| T3 截断把好 episode 误杀 | 中 | 中 | k 取大（k=7 或 9）；用 quantile 自适应阈值而非固定 |

---

## 9. 实现顺序建议

### Phase 1: Sotopia-veRL 打通（最硬，2-3 天）
- 只做 §3 sotopia rollout 适配层
- **不加 IG、不加 T3**，reward 直接用 episode-level goal_score（放到 last token）
- 验证：能跑一个 RL step，loss 有值，能 save checkpoint
- 产物：`pipeline/sotopia_rollout/adapter.py` 跑通

### Phase 2: 加 IGPO dense reward（1-2 天）
- 实现 §4 ideal_outcome 生成
- 跑通 vectorized IG 计算
- 验证：每轮能看到非零 IG 值，正负都有
- 产物：能看到 "useful turns" / "redundant turns" 的分布

### Phase 3: 加 T3 截断（1 天）
- 实现 §5 离线截断模式
- 观察 truncation rate 分布
- 产物：能看到 "截断率 vs goal_score" 的曲线

### Phase 4: Budget 惩罚 + 超参调优（2 天）
- γ 扫描：{1.0, 0.99, 0.95, 0.9}
- T3 阈值扫描：{固定 0, quantile 10%, quantile 20%}
- 产物：选定最优配置跑完整实验

### Phase 5: 评估和消融（1-2 天）
- 在 sotopia-hard 上评估 goal score / avg turn / redundancy rate
- 消融：去掉 IG / 去掉 T3 / 去掉 budget penalty 分别的效果
- 产物：论文需要的主表 + ablation

**总计：7-10 天**（在已跑通 pipeline step1-4 的基础上）

---

## 10. 关键参考代码位置速查

### veRL（训练引擎）
- 入口：`IGPO/verl/trainer/main_ppo.py:62-177`
- 训练主循环：`IGPO/verl/trainer/ppo/ray_trainer.py:1063-1413`
- Rollout worker：`IGPO/verl/workers/fsdp_workers.py:76-659`
- Multi-turn 配置：`IGPO/verl/trainer/config/ppo_trainer.yaml`

### IGPO（IG reward）
- IG 数值计算：`IGPO/scrl/llm_agent/generation.py:562-620`
- Vectorized 实现：`IGPO/scrl/llm_agent/vectorized_gt_logprob.py`
- Token 级 reward 分配：`IGPO/verl/utils/reward_score/info_gain.py:162-295`
- Turn-level advantage：`IGPO/verl/trainer/ppo/core_algos.py:28-102`
- 启动脚本：`IGPO/train.sh`

### T3（截断）
- BTR 检测：`T3/verl/search_r1/tau2_adapter/space.py:120-152`
- 轨迹生成主循环：`T3/verl/search_r1/llm_agent/generation.py:358-500`
- Adaptive threshold：论文 Appendix D.4（代码里应该对应 `T3/verl/verl/trainer/ppo/ray_trainer.py` 的 metrics 汇聚）

### Sotopia（环境）
- Episode 驱动：`sotopia_ref/sotopia/server.py:119-271` (`arun_one_episode`)
- LLMAgent：`sotopia_ref/sotopia/agents/llm_agent.py:64-103`
- EnvironmentProfile schema：`sotopia_ref/sotopia/database/persistent_profile.py:72-111`
- EpisodeLog schema：`sotopia_ref/sotopia/database/logs.py:41-106`

---

## 11. 下一步

**立即做**：
1. 让 user review 这份文档 + overview.md，对齐理解
2. 如果 OK，开始 Phase 1（`sotopia_rollout/adapter.py`）
3. 并行：在服务器拉 IGPO 的训练环境依赖（verl + sglang/vllm + ray 的版本兼容性）

**潜在阻塞**：
- 需要确认 IGPO 在服务器上能跑起来（Ray + FSDP 的分布式对单机多卡的适配）
- 需要确认 Gemma-4 和 IGPO 的 tokenizer 处理兼容（IGPO 默认用 Qwen2.5）

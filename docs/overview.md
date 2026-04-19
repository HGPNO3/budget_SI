# Budget SI 项目架构总览（给人看的版本）

> 目标读者：你（项目 owner）和刚接手的人
> 定位：不懂 veRL / IGPO / T3 也能看懂"这个项目在拼什么、为什么这样拼"
> 姊妹文档：[`design.md`](design.md)（技术实现细节）

---

## 一、这个项目要做什么

**一句话**：训练一个 LLM，让它在多轮社交对话里**少说废话**，同时**不掉目标达成率**。

**具体场景**：sotopia 是一个社交对话模拟器 —— 两个 LLM 扮演不同身份对话（比如砍价、说服、谈判），最多 20 轮，结束后一个 judge LLM 给双方打"目标达成分"（goal score）。

**观察**：当前 LLM 在这种场景下很啰嗦 —— 常常 10 轮就能谈下来的事非要说 18 轮。很多轮次的发言对推进目标**几乎没贡献**。

**假设**：用 RL 教模型"能不说就不说，只说有用的"。

---

## 二、为什么要把 4 个东西凑一起

我们需要 4 个组件，缺一个都不行：

| 组件 | 角色 | 类比 |
|---|---|---|
| **sotopia** | 提供训练数据和环境 | 游戏地图 |
| **veRL** | 通用 RL 训练引擎 | 训练引擎（像 PyTorch 但面向 RL） |
| **IGPO** | 给每一轮打分（哪句有用哪句废话） | 裁判 |
| **T3** | 发现在废话循环就提前喊停 | 剪辑师 |

单独看：
- **只有 sotopia** → 有数据没训练方法
- **只有 veRL** → 有引擎不知道奖励怎么算
- **只有 IGPO** → 知道每轮好坏但没有训练框架
- **只有 T3** → 能截断但不知道哪些轨迹有信息

**组合起来**：sotopia 出数据 → veRL 跑训练 → 每轮用 IGPO 的方法打分 → 遇到废话循环用 T3 截断。

---

## 三、每个组件到底是什么（通俗版）

### 1. sotopia

- **是什么**：一个 Python 框架，模拟双 agent 社交对话
- **一次 episode 长啥样**：
  ```
  场景：Alice 卖房子，Bob 想买但压价
  Alice 的目标：卖到 50 万以上
  Bob 的目标：40 万以内买下
  → 两人对话 1..N 轮
  → judge LLM 给双方各打一个 0-10 分
  ```
- **数据规模**：884 个场景，40 个角色
- **存储在**：Redis（服务器 `ms-8h20` 端口 6380）

### 2. veRL

- **是什么**：字节开源的大规模 RL 训练框架，PyTorch 世界里专门做"用 RL 训 LLM"的工具
- **类比**：相当于 HuggingFace 的 `Trainer`，但是 RL 版本
- **能干啥**：
  - 支持 PPO / GRPO / GSPO / REINFORCE++ 等主流 RL 算法
  - 分布式训练（Ray + FSDP）
  - 和 vLLM / SGLang 集成做高速推理
- **典型用法**：配置文件写好，一条 bash 命令就训
- **关键组件**：Trainer（总指挥）→ Actor（策略网络）+ Critic（价值网络）+ Rollout（生成）+ RewardManager（算奖励）

### 3. IGPO（Information Gain Policy Optimization）

- **解决什么问题**：标准的 GRPO 只在对话**结束时**给一个总分。"这场对话 8 分" —— 但你怎么知道是第 3 句的功劳还是第 7 句的？中间轮的贡献分不出来，梯度信号稀疏，小模型基本学不动（原论文称作 "advantage collapse"）。
- **核心想法**：**每一轮都给个分**，叫 **Information Gain (IG)**
- **怎么算 IG**：
  ```
  设有一个"理想答案" GT（比如问答任务里的 ground truth 答案）
  第 t 轮的 IG = log P(GT | 对话到第 t 轮) - log P(GT | 对话到第 t-1 轮)

  意思：这一轮让模型更相信/更不相信 GT 答案了多少
  大于 0 → 这轮推进了目标（有用）
  小于 0 → 这轮反而拖后腿（废话/错误方向）
  ```
- **关键技巧**：
  - 把 final reward（任务完成度）和 per-turn IG **分别**归一化后叠加
  - 用巧妙的 attention mask 把 T 轮的 log prob **一次 forward 就算完**（不用跑 T 次）
- **原版任务**：搜索 agent（给一个问题，多轮搜索找答案）
- **实验结果**：Qwen-7B 在 7 个 QA 数据集平均 F1 从 GRPO 的 51.9 提升到 **60.2**（+8.3）

### 4. T3（Truncating Belief-Trapped Trajectories）

- **解决什么问题**：在长多轮任务里，agent 经常**陷入死循环** —— 反复尝试无效动作、翻来覆去说相似的话。这种"废话尾巴"在 RL 里不仅浪费算力，还会**污染前段的梯度**（因为优势估计把尾巴的负分算到前面）。
- **核心想法**：**检测到 agent 停止推进就截断**，不让废话尾巴进训练
- **怎么检测"陷阱"**：
  - 连续 k 轮没有进展 → 陷阱
  - "进展"的信号可以是：任务完成度涨了没、语义相似度是不是飘了、信息增益是不是还在增加
- **怎么"截断"**：
  - 从检测到陷阱那个 turn 开始，**后续 token 全部丢掉**，不进 PPO loss
  - 不是 mask，是**真的不用**
- **原版任务**：主动推理 agent（GuessNumber、puzzle 等）
- **实验结果**：GSPO + T3 在 MovieRec 任务上 **EM +41**，token 开销最多降 34%

---

## 四、组合方案：整个训练过程长啥样

### 直观流程（一个 training step）

```
    ┌───────────────────────────┐
    │  sotopia 场景池 (884 个)  │
    └────────────┬──────────────┘
                 │  随机抽 1 个场景 + 2 个角色
                 ▼
    ┌───────────────────────────┐
    │ Rollout：跑 sotopia 对话  │
    │   policy agent 用我们的模型│
    │   opponent 用固定 base 模型│
    └────────────┬──────────────┘
                 │  边跑边监测
                 ▼
    ┌───────────────────────────┐
    │ T3 截断监测               │
    │   最近 k 轮 IG 都 < 阈值？  │   ← 触发就提前结束
    └────────────┬──────────────┘
                 │ 跑完 / 截断
                 ▼
    ┌───────────────────────────┐
    │ judge LLM 打分            │
    │   → episode-level goal    │
    │     score                 │
    └────────────┬──────────────┘
                 │
                 ▼
    ┌───────────────────────────┐
    │ IG 计算（向量化）          │
    │   每轮的 ΔlogP(理想结局)   │
    └────────────┬──────────────┘
                 │ 每轮 IG + 最终 goal score
                 ▼
    ┌───────────────────────────┐
    │ 组装 per-token reward     │
    │   IG 放到每轮最后一个 token│
    │   goal score 放到 episode 最后│
    └────────────┬──────────────┘
                 │
                 ▼
    ┌───────────────────────────┐
    │ GRPO 更新 policy model    │
    │   只更新 policy agent 说的 │
    │   那些 token 的梯度        │
    └───────────────────────────┘
```

### 关键设计决策（为什么是这样）

**1. 为什么只训 policy agent 一边，不两边都训？**
- 稳定性：两边都在变会训练不稳（像 GAN 训练崩溃）
- 复用 opponent 当"环境的一部分"，用 base model 冻结

**2. IGPO 里的"ideal outcome" 用什么填？**
- IGPO 原版要 ground truth 答案。sotopia 没有。
- 方案：**用 judge LLM 生成"理想结局描述"**
  - 给定：场景 + policy agent 的目标
  - 生成：一段"如果 agent 完美达成目标，最终对话结局应该长什么样"的文本
  - 这段文本就是 GT，每轮 IG = Δ log P(这段 GT | 当前对话)
- Fallback：如果"理想结局"定义不稳，改用 **goal score 作为 IG 的代理**（直接问 judge "到这一轮为止目标完成度多少"，算 delta）

**3. T3 的 progress signal 用什么？**
- 直接**复用 IGPO 的 per-turn IG** —— 这是最漂亮的结合点
- 连续 k 轮 IG < 阈值 → 截断
- 阈值用 adaptive quantile（论文 Appendix D.4）：训练中动态统计 policy 自己的 IG 分布，取低 α 分位数

**4. "减少冗余轮次"的目标怎么体现？**
- IGPO 原版 γ=1（不折扣），只关心信息是否传到了，不在乎用了几轮
- 我们改 **γ < 1**（比如 0.95），让后期轮次的 IG 折价 → 模型学会"早点把信息说完"
- 也可以加一个 explicit 惩罚项：final_reward -= λ × turn_count

---

## 五、难点预览

**从容易到难**：

| 难度 | 问题 | 简述 |
|---|---|---|
| 🟢 易 | Turn/Message tokenize | sotopia 用字符串存对话，需 tokenize 时维护 speaker mask |
| 🟡 中 | 两个 agent 的 mask | policy agent 的 token 参与 loss，opponent 的不参与 |
| 🟡 中 | sotopia 的 LLM 调用接管 | 原版走 OpenAI API，改成走 veRL 的本地模型 |
| 🔴 难 | Rollout 架构改造 | veRL 是 token-level 采样循环，sotopia 是 episode-level 环境驱动，两种循环模型要缝合 |
| 🔴 难 | "理想结局" GT 怎么稳定生成 | judge LLM 输出不稳定会直接坏掉 IG 信号 |

详见 [`design.md`](design.md) 的 §6 风险和备选方案。

---

## 六、预期产出

**最终训出来的东西**：
- 一个 LoRA adapter（Gemma-4-E4B 的）
- 在 sotopia hard 测试集上：
  - Goal score **不降**（甚至略升）
  - 平均对话轮数 **下降 20%+**
  - 冗余率（IG ≤ 0 的轮数占比）**下降 30%+**

**论文叙事**：IGPO 的 dense reward + T3 的 trajectory 截断，在"budget 约束下的多轮社交对话"这个新场景上验证有效；揭示信息增益信号在社交任务中的行为特点（比 QA 任务更 noisy，但仍能传递有效梯度）。

---

## 七、术语速查

| 术语 | 含义 |
|---|---|
| Episode | 一局完整的 sotopia 对话 |
| Turn / Round | 一轮 = 一个 agent 说一次话（有些地方 1 turn = A+B 各一次） |
| Rollout | RL 里"让模型跑一遍生成输出"的过程 |
| Trajectory | 一条完整的 state-action-reward 序列（≈ 一局 episode 的结构化版本） |
| Advantage | "这个动作比平均水平好多少"，RL 梯度的核心信号 |
| GAE | Generalized Advantage Estimation，把 reward 转成 advantage 的一种方法 |
| GRPO | Group Relative Policy Optimization，DeepSeek 提出的 RL 算法，veRL 默认支持 |
| IG | Information Gain，IGPO 的核心 per-turn reward |
| BTR | Belief-Trap Region，T3 里"agent 陷入死循环"的状态 |
| LoRA | 低秩微调，只训一小部分参数 |
| veRL | Volcano Engine RL，字节开源的 RL 训练框架 |
| judge LLM | 一个外部的 LLM 做打分器（sotopia 默认用 GPT-4o） |

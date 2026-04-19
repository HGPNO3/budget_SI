# Budget SI 项目进度 WIKI

> 最后更新：2026-04-19
> 项目开始：2026-04-16（SSH 配置 + 服务器部署）
> 截止日期：约 2026-05-09（20 天）

---

## 一、项目目标

训练 LLM 在多轮社交对话中**减少冗余发言**，同时保持对话目标达成率。

核心思路：用信息增益（Information Gain）逐轮评估对话贡献度 → 识别冗余轮次 → 通过 RL 训练模型学会跳过冗余。

---

## 二、基础设施状态

### 2.1 服务器

| 项目 | 状态 | 详情 |
|------|------|------|
| SSH 保活 | ✅ 完成 | ~/.ssh/config 配置 ServerAliveInterval=60 |
| Redis Stack | ✅ 运行中 | 端口 6380，数据目录 /home/ubuntu/lx/redis-data/ |
| Sotopia 数据集 | ✅ 加载 | 39050+ 条，884 场景，40 角色 |
| Gemma-4 vLLM | ✅ 运行中 | 端口 8001，GPU 1，--max-model-len 8192 |
| Conda 环境 | ✅ 两个 | lx_sotopia（推理+数据）/ lx_vllm（训练） |

### 2.2 已知问题及解决方案

| 问题 | 根因 | 解决方案 | 状态 |
|------|------|----------|------|
| EpisodeLog pk 验证错误 | redis-om 0.3.5 的 @validator 在 pydantic 2.x 不生效 | 打补丁 patch_redis_om.py | ✅ 已修 |
| EpisodeLog.find().all() 返回旧数据 | MAXSEARCHRESULTS=10000，sotopia-π 有 10000+ 条旧数据 | 按 tag="budget_si" 过滤查询 | ✅ 已修 |
| sotopia rewards 全为 0.0 | sotopia 0.1.5 内部 bug，评估结果未存入 EpisodeLog.rewards | 新增 step1b_evaluate.py 自己评估 | ✅ 已修 |
| 终端粘贴加空格 | 渲染问题 | 写 .sh 脚本 scp 到服务器 | ✅ 绕过 |
| vLLM 端口 8000 被占 | 别人的服务 | 改用 8001 | ✅ 已修 |
| LoRA target_modules 不兼容 | Gemma4ClippableLinear 包装层 | target_modules="all-linear" | ✅ 已修 |
| transformers 不认 gemma4 | 版本太旧 | 升级到 5.5.4 | ✅ 已修 |

---

## 三、Pipeline 状态

```
step0 → step1 → step1b → step2 → step3 → step4
筛选     生成     评估     IG计算   过滤     训练
 ✅       ✅       ✅       ✅       ✅       ✅
```

### 3.1 各步骤详情

| 步骤 | 文件 | 功能 | 状态 | 输出 |
|------|------|------|------|------|
| step0 | step0_filter_scenes.py | 三层过滤（codename + goal conflict + 排除合作）筛选零和场景 | ✅ | 178 个零和场景 |
| step1 | step1_generate_episode.py | 从筛选场景生成对话，Gemma-4 自我博弈，存 Redis | ✅ | 15 条 episode JSON |
| step1b | step1b_evaluate.py | 调 LLM 给对话打分，判定 winner | ✅ | 11 条有 winner，4 条 tie |
| step2 | step2_compute_info_gain.py | 按回合计算信息增益（当前：简易版，问模型打分 0-10） | ✅ | 每轮标记 useful/redundant/neutral |
| step3 | step3_filter_data.py | 删除 IG≤0 的冗余回合 | ✅ | 125→100 轮（20% 冗余率） |
| step4 | step4_train_rl.py | Reward-Filtered SFT + LoRA | ✅ | loss 3.77→1.91，LoRA adapter 已保存 |

### 3.2 当前数据统计

| 指标 | 值 |
|------|------|
| 筛选场景数 | 178 / 884 |
| 生成 episode 数 | 15 |
| 有 winner 的 | 11 |
| 训练样本数 | 61（过滤后的回合） |
| 平均冗余率 | 20%（范围 0%-40%） |
| 训练 epochs | 3 |
| 最终 loss | 1.91（从 3.77 下降） |

---

## 四、已对齐的设计决策

### 4.1 数据设计

| 决策 | 内容 | 理由 |
|------|------|------|
| 场景选择 | 零和博弈场景（当前 178 个，后续可扩展） | 有明确胜负，便于定义 GT |
| Ground Truth | 谁的 goal score 更高 → 谁赢了 | 简单、可复现 |
| 轮次定义 | 一个回合 = A 说话 + B 回复 | 不拆分单条消息 |
| 关注对象 | 看所有回合（包括 B 的），但只训练 A（赢家）的生成概率 | B 的让步也贡献了 A 的胜利 |
| GT 文本 | "[Name] successfully achieved their goal: [goal]" | 适中长度（20-30 token），锚定目标 |
| 平局处理 | 丢弃 | goal score 相同的无法定义 GT |

### 4.2 训练设计

| 决策 | 内容 | 理由 |
|------|------|------|
| IG 计算 | 当前：问模型打分 0-10；后续：IGPO token log-prob | 先跑通再升级 |
| 训练方式 | 当前：Offline（Reward-Filtered SFT）；后续：Online（GRPO via veRL） | 先出 baseline 再迭代 |
| 模型 | Gemma-4-E4B-it + LoRA (r=16, all-linear) | 轻量微调，不污染原模型 |
| 环境分离 | lx_sotopia（推理）和 lx_vllm（训练）分开 | 避免 pydantic 版本冲突 |

### 4.3 评估设计

| 指标 | 方向 | 用途 |
|------|------|------|
| Redundancy Rate | ↓ 下降 | 证明"更精炼了" |
| 平均对话轮数 | ↓ 下降 | 证明"更高效了" |
| Goal Score | 不降 | 证明"没变差" |
| 对话成功率 | 不降 | 证明"没崩" |

### 4.4 实验对比结构（参考 MINT）

| 实验 | 方法 | 角色 |
|------|------|------|
| Vanilla | 原始 Gemma-4 直接跑 sotopia | 最弱 baseline |
| Prompt | 提示词里写"请简洁地说话" | prompt baseline |
| Quality-only RL | 只用 goal score 训练 | RL baseline |
| 方法 A | 删掉冗余轮 + SFT | 简单版（当前已实现） |
| 方法 B | IG 作为 reward + REINFORCE | 进阶版 |
| 方法 C | GRPO + IG reward（veRL） | 最终版 |

---

## 五、待做事项（按优先级）

### P1：立即要做

| 任务 | 描述 | 预计时间 |
|------|------|----------|
| 升级 IG 为 token log-prob | 把 step2 从"问模型打分"改成 IGPO 的方式 | 1-2 天 |
| 实现方法 B（负分 REINFORCE） | 不删除冗余轮，而是给负 reward | 1 天 |
| 扩大数据到 100-200 条 | 批量跑 step1 + step1b | 半天（跑脚本等） |

### P2：本周要做

| 任务 | 描述 | 预计时间 |
|------|------|----------|
| 搭建 veRL Online RL | fork IGPO 代码，适配 sotopia 环境 | 2-3 天 |
| 换强 judge | 用 GPT-4/Claude API 替代 Gemma-4 做评估 | 半天 |
| 分析 IG 分布 + 定阈值 | 基于 100+ 条数据统计 IG 分布 | 半天 |

### P3：下周要做

| 任务 | 描述 | 预计时间 |
|------|------|----------|
| 完整对比实验 | Vanilla vs Prompt vs 方法 A/B/C | 2-3 天 |
| 消融实验 | 不同 IG 阈值、不同 γ、有无截断 | 1-2 天 |
| sotopia-hard 测试 | 在 20 个最难场景上验证泛化 | 半天 |
| 整理结果 + 写论文 | — | 3-5 天 |

---

## 六、参考论文

| 论文 | 路径 | 与我们的关系 |
|------|------|-------------|
| IGPO (ICLR 2026) | budget_SI/IGPO/ | 核心算法来源：IG reward + turn-level advantage |
| T3 (ICLR 2026) | budget_SI/T3/ | 截断思路：连续低 IG 轮可截断 |
| OMAR (2602.03109) | self-locking4SI/papers/B7 | 最相关 baseline：sotopia 自我博弈 RL |
| MINT (2604.11742) | ~/Downloads/ | 实验设计参考：对比结构 + 评估方法 |
| Sotopia (ICLR 2024) | sotopia_ref/ | 环境：884 场景 + 评估框架 |

---

## 七、关键代码位置

### 我们的代码

```
budget_SI/pipeline/
├── config.py                  — 服务器配置
├── step0_filter_scenes.py     — 场景筛选
├── step1_generate_episode.py  — 对话生成
├── step1b_evaluate.py         — 自主评估打分
├── step2_compute_info_gain.py — IG 计算
├── step3_filter_data.py       — 冗余过滤
├── step4_train_rl.py          — LoRA 训练
├── run_pipeline.sh            — 批量跑 step2+3
└── data/                      — 生成的数据
    ├── filtered_scenes.json   — 178 个零和场景
    ├── episode_*.json         — 原始 episode
    ├── *_info_gain.json       — IG 计算结果
    └── *_filtered.json        — 过滤后的数据
```

### 参考代码（IGPO）

```
budget_SI/IGPO/
├── scrl/llm_agent/
│   ├── vectorized_gt_logprob.py  — IG 的 token log-prob 计算（要迁移）
│   └── generation.py             — 多轮 rollout 管理
├── verl/utils/reward_score/
│   └── info_gain.py              — reward 组装（要迁移）
└── verl/trainer/ppo/
    └── core_algos.py             — GRPO + turn-level advantage（要迁移）
```

---

## 八、时间线回顾

| 日期 | 完成了什么 |
|------|-----------|
| 4/16 | SSH 保活、sotopia 部署、Redis Stack、数据集加载 |
| 4/17 | Gemma-4 下载 + vLLM 部署、EpisodeLog 补丁、最小 pipeline 跑通（step1-4） |
| 4/18 | 场景筛选（178 个）、pipeline 重构（按回合计算 IG） |
| 4/19 | 修复 push_to_db + MAXSEARCHRESULTS bug、新增 step1b 评估、完整 pipeline 跑通 baseline |

### 剩余时间分配建议（Day 4-20）

| 阶段 | 天数 | 内容 |
|------|------|------|
| 数据扩展 + IG 升级 | Day 4-6 | 扩到 100-200 条、升级 token log-prob |
| Offline 完整实验 | Day 7-9 | 方法 A/B 对比、baseline 结果 |
| Online RL (veRL) | Day 8-12 | 并行搭建、方法 C |
| 评估 + 消融 | Day 13-16 | 完整对比、sotopia-hard 测试 |
| 写论文 | Day 17-20 | 整理结果、画图、写作 |

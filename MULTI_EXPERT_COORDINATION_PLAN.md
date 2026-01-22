# Multi-Expert Coordination for RL-based LLM Jailbreak (MEC-RL)

本文件记录“攻击家族=专家Agent + 轻量协调者”的实现方案，供后续实现与论文撰写使用。

## 目标
- 将每类攻击/攻击家族抽象为“专家Agent（Expert）”。
- 新增“协调者（Coordinator）”在每步根据状态与专家元特征，对专家进行加权路由。
- 不破坏现有 RainbowDQN 的核心训练流程（最小侵入）。

## 架构概览
- Experts（M≈6-8）：按家族聚合现有 12 类攻击到 M 个专家，每个专家绑定其子动作集合。
- Coordinator（gating 模块）：输入=state 摘要 + 专家元特征；输出=各专家权重 softmax w_e。
- 动作偏置注入（logit bias）：对隶属专家 e 的所有动作 logits 加偏置 β·log(w_e+ε)，再走原选择。

## 专家分组建议
- 示例家族：Flip / Uncertainty / Distractor / Injection / Unicode / CoT（可合并/拆分以均衡规模）。
- 每家族维护其子动作索引列表（从 `action_to_attack` 映射派生）。

## 专家元特征（Meta-Features）
- 历史成功率（滑动/EMA）、最近 K 步胜率增量
- 使用比例（本 episode 使用次数 / max_queries）、冷却标记
- 置信度贡献（最近 K 步置信度的 EMA 下降量，或简化 Shapley 近似）
- 成本/风险：平均失败代价、平均触发步数
- 形成张量 M×d_meta；各维标准化/裁剪

## 协调者（Coordinator）
- 结构：两层 MLP（隐藏 128）或轻量 Attention（query=state 摘要，key/value=专家元特征）
- 输出：w ∈ R^M，softmax 归一化
- 强度 β：{0.5,1.0,1.5,2.0}，控制注入到动作 logits 的偏置强度

## 训练方式（不改 Rainbow 的 TD 损失）
- 方案A（推荐，稳定）：监督式近似优选
  - 标签：回放时计算“带来最大即时奖励/最大置信度下降/最短成功步”的专家作为 teacher label
  - 损失：Cross-Entropy(softmax(w), label)
- 方案B：轻量策略梯度（REINFORCE）微调协调者
  - 奖励：episode return 或 shaped reward
- 两阶段：先 A 后 B

## 状态与管线对接（最小侵入）
- 保持 q_network 输入不变；在选动作前计算 w，并对动作 logits 注入偏置后再 argmax/采样。
- 可选：将“专家统计的全局汇总（若干标量）”拼到 state 尾部，但非必须。

## 代码改动点（文件级）
- 新增 `experts.py`
  - 专家分组与子动作索引
  - 每专家元特征的更新与缓存（episode 重置、step 更新）
- 新增 `coordinator.py`
  - 协调者前向/训练接口（支持监督/REINFORCE）
- 修改 `scripts/train_fast_rl_with_split_rainbowdqn.py`
  - 环境构造后初始化 Experts/Coordinator
  - 动作选择前：计算 w，注入 logits 偏置
  - 回放时：构造监督标签/奖励信号，训练 Coordinator
- （可选）在 `agent.py` 暴露“logits 注入钩子”

## 日志与评估指标
- 记录：专家权重分布、调用频率、胜率、ASR/AQA 边际贡献
- 消融：无协调 / 均匀权重 / 仅历史胜率 / 全元特征；β 扫描；M（家族数）扫描
- 指标：ASR、AQA、收敛速度、方差、长尾样本提升、跨 Judge 泛化

## 实施顺序
1) 专家分组与子动作索引（仅统计与可视化）
2) 启发式协调（EMA 胜率/成本）→ 注入偏置 → 验证正效应
3) 可学习协调者（监督式）→ 消融与超参
4) （可选）策略梯度微调 + 不确定性驱动探索开关

## 论文叙事要点
- 多专家协同解决多样化策略选择与长尾问题
- 提出“动作语义-效能联合表征”的元策略 + 轻量协调路由
- 在固定预算下显著提升 ASR、降低 AQA，且更稳定、可解释

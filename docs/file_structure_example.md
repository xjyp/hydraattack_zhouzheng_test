# 运行脚本生成的文件结构示例

## 📁 完整路径结构

假设运行时间戳为 `20251105_143000`，benchmarks 为 `arena_hard`, `alpaca_eval`, `code_judge_bench`：

```
results_rainbowdqn/                                         # Rainbow DQN结果根目录（独立于baseline结果）
└── rl_generation_rainbowdqn_test_20251105_143000/          # 基础结果目录（shell脚本创建）
    ├── arena_hard_20251105_143001/                         # 第一个benchmark的实验目录
    │   ├── README.md                                       # 实验说明文档（训练脚本生成）
    │   │
    │   ├── config/                                         # 配置和元数据目录
    │   │   ├── training_config_arena_hard.json            # 完整超参数配置
    │   │   └── action_mapping_arena_hard.json             # 动作映射
    │   │
    │   ├── summary/                                        # 汇总数据目录（目前为空，待实现）
    │   │   ├── training_summary.json                     # 训练过程汇总（待实现）
    │   │   ├── test_summary.json                         # 测试结果汇总（待实现）
    │   │   ├── attack_usage_stats.json                    # 攻击方法使用统计（待实现）
    │   │   └── episode_statistics.csv                     # Episode级别统计（待实现）
    │   │
    │   ├── training/                                       # 训练过程数据目录（目前为空，待实现）
    │   │   ├── episodes/                                  # Episode详细数据（待实现）
    │   │   ├── checkpoints/                               # 模型检查点（待实现）
    │   │   └── training_curves.json                      # 训练曲线数据（待实现）
    │   │
    │   ├── evaluation/                                     # 评估数据目录（目前为空，待实现）
    │   │   └── test_samples/                             # 测试样本详细记录（待实现）
    │   │       ├── successful/                           # 成功攻击样本（待实现）
    │   │       ├── failed/                               # 失败攻击样本（待实现）
    │   │       └── index.json                            # 样本索引（待实现）
    │   │
    │   ├── judge_logs/                                    # Judge输入输出记录目录（目前为空，待实现）
    │   │   ├── training/                                  # 训练阶段judge记录（待实现）
    │   │   │   ├── judge_logs_summary.json               # 汇总（待实现）
    │   │   │   └── detailed/                             # 详细记录（待实现）
    │   │   └── test/                                      # 测试阶段judge记录（待实现）
    │   │       ├── judge_logs_summary.json               # 汇总（待实现）
    │   │       └── detailed/                             # 详细记录（待实现）
    │   │
    │   ├── logs/                                          # 日志文件目录
    │   │   ├── hydra_attack_20251105_143001.log          # 训练日志（HydraLogger生成）
    │   │   ├── hydra_attack_20251105_143001_config.json  # 日志配置（HydraLogger生成）
    │   │   └── hydra_attack_20251105_143001_samples.json # 样本详情（HydraLogger生成，如果使用）
    │   │
    │   ├── fast_rl_attacker_arena_hard.pth                # 模型文件（根目录，便于查找）
    │   ├── action_mapping_arena_hard.json                 # 动作映射（根目录备份，兼容性）
    │   └── training_config_arena_hard.json               # 训练配置（根目录备份，兼容性）
    │
    ├── alpaca_eval_20251105_143100/                        # 第二个benchmark的实验目录
    │   └── ... (相同的结构)
    │
    └── code_judge_bench_20251105_143200/                  # 第三个benchmark的实验目录
        └── ... (相同的结构)
```

## 📄 当前已实现的文件

### 1. 根目录文件（每个benchmark实验目录下）
- ✅ `README.md` - 实验说明和快速索引
- ✅ `fast_rl_attacker_{benchmark}.pth` - 训练好的模型
- ✅ `action_mapping_{benchmark}.json` - 动作映射（根目录备份）
- ✅ `training_config_{benchmark}.json` - 训练配置（根目录备份）

### 2. config/ 目录
- ✅ `training_config_{benchmark}.json` - 完整超参数配置
- ✅ `action_mapping_{benchmark}.json` - 动作映射

### 3. logs/ 目录
- ✅ `hydra_attack_{timestamp}.log` - 训练日志
- ✅ `hydra_attack_{timestamp}_config.json` - 日志配置（HydraLogger生成）
- ✅ `hydra_attack_{timestamp}_samples.json` - 样本详情（如果HydraLogger记录）

### 4. 目录结构（已创建但内容待实现）
- ✅ `summary/` - 目录已创建，但汇总文件待实现
- ✅ `training/` - 目录已创建，但训练数据文件待实现
- ✅ `evaluation/` - 目录已创建，但评估数据文件待实现
- ✅ `judge_logs/` - 目录已创建，但judge记录文件待实现

## 🔍 文件路径示例

假设运行时间为 `2025-11-05 14:30:00`，第一个benchmark是 `arena_hard`：

### 绝对路径示例
```bash
/home/wzdou/project/hydraattack_share/results_rainbowdqn/rl_generation_rainbowdqn_test_20251105_143000/arena_hard_20251105_143001/
```

### 关键文件路径
```bash
# 模型文件
results_rainbowdqn/rl_generation_rainbowdqn_test_20251105_143000/arena_hard_20251105_143001/fast_rl_attacker_arena_hard.pth

# 配置文件
results_rainbowdqn/rl_generation_rainbowdqn_test_20251105_143000/arena_hard_20251105_143001/config/training_config_arena_hard.json
results_rainbowdqn/rl_generation_rainbowdqn_test_20251105_143000/arena_hard_20251105_143001/config/action_mapping_arena_hard.json

# 日志文件
results_rainbowdqn/rl_generation_rainbowdqn_test_20251105_143000/arena_hard_20251105_143001/logs/hydra_attack_20251105_143001.log

# README
results_rainbowdqn/rl_generation_rainbowdqn_test_20251105_143000/arena_hard_20251105_143001/README.md
```

## 📊 时间戳说明

### Shell脚本创建的时间戳
- `BASE_RESULTS_DIR`: `rl_generation_rainbowdqn_test_${TIMESTAMP}`
  - 格式: `YYYYMMDD_HHMMSS`
  - 示例: `20251105_143000`
  - **作用**: 标识整个运行批次

### 每个benchmark的时间戳
- `EXPERIMENT_DIR`: `${BASE_RESULTS_DIR}/${BENCHMARK}_${BENCHMARK_TIMESTAMP}`
  - 格式: `{benchmark}_YYYYMMDD_HHMMSS`
  - 示例: `arena_hard_20251105_143001`
  - **作用**: 标识每个benchmark的实验

### 日志文件的时间戳
- `hydra_attack_{timestamp}.log`: HydraLogger 内部创建的时间戳
  - 格式: `hydra_attack_YYYYMMDD_HHMMSS.log`
  - 示例: `hydra_attack_20251105_143001.log`
  - **作用**: 标识日志文件

## 🎯 快速查找

### 查看所有实验结果
```bash
ls -la results_rainbowdqn/rl_generation_rainbowdqn_test_*/
```

### 查看特定benchmark的结果
```bash
ls -la results_rainbowdqn/rl_generation_rainbowdqn_test_*/arena_hard_*/
```

### 查看最新实验结果
```bash
ls -lt results_rainbowdqn/rl_generation_rainbowdqn_test_*/ | head -20
```

### 查看README了解目录结构
```bash
cat results_rainbowdqn/rl_generation_rainbowdqn_test_*/arena_hard_*/README.md
```

## ⚠️ 注意事项

1. **时间戳独立**: 每次运行都会创建新的时间戳目录，不会覆盖
2. **benchmark隔离**: 每个benchmark有独立的实验目录
3. **目录结构**: 所有子目录在训练开始时创建，即使某些文件还未实现
4. **兼容性**: 重要文件在根目录保留备份，便于快速访问

## 📝 待实现的功能

根据数据组织方案，以下文件需要后续实现：
- `summary/training_summary.json` - 训练汇总
- `summary/test_summary.json` - 测试汇总
- `summary/attack_usage_stats.json` - 攻击使用统计
- `summary/episode_statistics.csv` - Episode统计
- `evaluation/test_samples/` - 测试样本详细记录
- `judge_logs/` - Judge输入输出记录


# RL训练数据组织方案

## 📁 推荐的目录结构

**重要说明**: 所有数据都保存在带时间戳的实验目录下，例如：
- `results_rainbowdqn/rl_generation_rainbowdqn_test_20251105_085316/arena_hard_20251105_085316/`

这样可以确保每次运行都有独立的结果文件夹，避免多次运行之间的混乱。Rainbow DQN的结果保存在独立的 `results_rainbowdqn/` 根目录下，与baseline结果目录 `results/` 分离。

```
{timestamp_dir}/  # 例如: results_rainbowdqn/rl_generation_rainbowdqn_test_20251105_085316/arena_hard_20251105_085316/
├── README.md                          # 实验说明和快速索引
├── config/
│   ├── training_config.json          # 完整超参数配置（已保存）
│   ├── action_mapping.json           # 动作映射（已保存）
│   └── experiment_metadata.json      # 实验元数据（时间、环境等）
│
├── summary/                           # 📊 汇总数据（快速查看）
│   ├── training_summary.json         # 训练过程汇总
│   ├── test_summary.json             # 测试结果汇总
│   ├── attack_usage_stats.json       # 攻击方法使用统计
│   └── episode_statistics.csv        # Episode级别统计（CSV，便于分析）
│
├── training/                          # 🏋️ 训练过程数据
│   ├── episodes/                     # Episode详细数据（可选，大数据）
│   │   ├── episode_000000.json
│   │   ├── episode_000500.json
│   │   └── ... (每eval_freq保存一次)
│   ├── checkpoints/                  # 模型检查点
│   │   ├── checkpoint_500.pth
│   │   └── best_model.pth
│   └── training_curves.json          # 训练曲线数据（用于绘图）
│
├── evaluation/                        # 🔍 评估数据
│   ├── test_samples/                 # 测试样本详细记录
│   │   ├── successful/               # 成功攻击样本
│   │   │   ├── sample_001.json
│   │   │   └── ...
│   │   ├── failed/                   # 失败攻击样本
│   │   │   ├── sample_042.json
│   │   │   └── ...
│   │   └── index.json                # 样本索引（快速查找）
│   ├── attack_sequences/             # 攻击序列记录
│   │   ├── sequences_summary.json    # 序列统计汇总
│   │   └── detailed_sequences/      # 详细序列（可选，大数据）
│   │       └── sample_001_sequence.json
│   └── cross_analysis.json           # 跨数据集/跨模型分析
│
├── judge_logs/                        # ⚖️ Judge输入输出记录
│   ├── training/                     # 训练阶段judge记录
│   │   ├── judge_logs_summary.json   # 汇总（统计信息）
│   │   └── detailed/                 # 详细记录（可选）
│   │       └── judge_call_*.json
│   └── test/                         # 测试阶段judge记录
│       ├── judge_logs_summary.json
│       └── detailed/
│           └── judge_call_*.json
│
├── logs/                              # 📝 日志文件
│   ├── training.log                  # 训练日志
│   ├── evaluation.log                # 评估日志
│   └── error.log                     # 错误日志
│
├── fast_rl_attacker_{benchmark}.pth   # 模型文件（保留在根目录，便于查找）
├── action_mapping_{benchmark}.json   # 动作映射（保留在根目录）
└── training_config_{benchmark}.json  # 训练配置（保留在根目录，兼容旧版本）
```

## 📋 数据粒度说明

### Level 1: 汇总数据（Summary）- 快速查看
- **用途**: 快速了解实验整体情况
- **大小**: 小（< 1MB）
- **格式**: JSON
- **内容**:
  - 训练/测试成功率、平均查询次数等关键指标
  - 攻击方法使用频率和成功率
  - Episode统计汇总

### Level 2: 统计数据（Statistics）- 数据分析
- **用途**: 进行数据分析和可视化
- **大小**: 中（1-10MB）
- **格式**: CSV + JSON
- **内容**:
  - Episode级别的统计（reward, success, queries等）
  - 每个攻击方法的详细统计
  - 训练曲线数据点

### Level 3: 详细记录（Detailed Records）- Case Study
- **用途**: 深入分析特定样本或案例
- **大小**: 大（10MB - 1GB+）
- **格式**: JSON（按需保存）
- **内容**:
  - 每个样本的完整攻击过程
  - 每次judge调用的输入输出
  - 完整的攻击序列

### Level 4: 原始日志（Raw Logs）- 调试和追溯
- **用途**: 调试和完整追溯
- **大小**: 很大（1GB+）
- **格式**: 文本日志
- **内容**:
  - 所有操作的详细日志
  - 错误和异常信息

## 🎯 数据组织策略

### 1. 按数据粒度分层存储
- **必须保存**: Level 1 + Level 2（汇总和统计）
- **可选保存**: Level 3（详细记录，按需开启）
- **自动保存**: Level 4（日志，始终保存）

### 2. 按数据类型分类存储
- **配置数据**: `config/` - 实验配置和元数据
- **训练数据**: `training/` - 训练过程相关
- **评估数据**: `evaluation/` - 测试和评估相关
- **Judge数据**: `judge_logs/` - Judge输入输出
- **日志数据**: `logs/` - 文本日志

### 3. 使用索引文件加速查找
- `evaluation/test_samples/index.json`: 快速查找样本
- `judge_logs/*/judge_logs_summary.json`: Judge调用统计

### 4. 压缩大文件
- 详细记录可以压缩存储（.json.gz）
- 日志文件可以归档压缩

## 📊 核心数据文件说明

### `summary/training_summary.json`
```json
{
  "total_episodes": 1000,
  "final_success_rate": 0.75,
  "avg_episode_reward": 15.3,
  "avg_queries_per_episode": 4.2,
  "training_time": 3600.5,
  "best_checkpoint": "checkpoint_850",
  "early_stopping_triggered": false,
  "validation_scores": [...],
  "episode_rewards": [...],  // 每100个episode的平均值
  "episode_success_rates": [...]
}
```

### `summary/attack_usage_stats.json`
```json
{
  "training": {
    "FlipAttackFCS": {
      "usage_count": 1250,
      "usage_rate": 0.25,
      "success_count": 800,
      "success_rate": 0.64,
      "avg_reward": 12.5
    },
    ...
  },
  "test": {
    "FlipAttackFCS": {
      "usage_count": 150,
      "usage_rate": 0.30,
      "success_count": 100,
      "success_rate": 0.67
    },
    ...
  }
}
```

### `evaluation/test_samples/index.json`
```json
{
  "total_samples": 500,
  "successful_samples": 375,
  "failed_samples": 125,
  "successful_ids": ["sample_001", "sample_003", ...],
  "failed_ids": ["sample_042", "sample_089", ...],
  "sample_metadata": {
    "sample_001": {
      "question_id": "...",
      "success": true,
      "queries_used": 3,
      "attack_method": "FlipAttackFCS",
      "file_path": "successful/sample_001.json"
    },
    ...
  }
}
```

### `judge_logs/test/judge_logs_summary.json`
```json
{
  "total_calls": 2500,
  "successful_calls": 2450,
  "failed_calls": 50,
  "avg_response_time": 0.5,
  "preference_distribution": {
    "A": 1200,
    "B": 1250
  },
  "confidence_stats": {
    "mean": 0.72,
    "std": 0.15,
    "min": 0.1,
    "max": 0.99
  },
  "sample_indices": {
    "sample_001": [0, 1, 2],  // 该样本的judge调用索引
    ...
  }
}
```

## 🔧 实现建议

### 1. 可配置的数据保存级别
```python
# 在训练脚本中添加参数
parser.add_argument("--save_detail_level", type=str, default="summary",
                    choices=["summary", "statistics", "detailed", "all"],
                    help="数据保存详细程度")
```

### 2. 批量保存和索引
- 每N个episode保存一次详细数据
- 使用索引文件快速定位
- 支持按需加载详细数据

### 3. 数据压缩选项
```python
parser.add_argument("--compress_detailed", action="store_true",
                    help="压缩详细记录文件")
```

### 4. 增量保存
- 训练过程中增量保存
- 避免内存占用过大
- 支持断点续训

## 📈 使用建议

### 快速查看实验结果
```bash
# 查看汇总结果
cat summary/training_summary.json
cat summary/test_summary.json
cat summary/attack_usage_stats.json
```

### 数据分析
```python
# 加载统计数据进行分析
import pandas as pd
df = pd.read_csv('summary/episode_statistics.csv')
df.plot(x='episode', y='reward')
```

### Case Study
```python
# 按需加载详细数据
with open('evaluation/test_samples/index.json') as f:
    index = json.load(f)
    
# 加载特定样本
sample_id = index['successful_ids'][0]
with open(f"evaluation/test_samples/successful/{sample_id}.json") as f:
    sample = json.load(f)
```

## 💡 优势

1. **层次清晰**: 按数据粒度分层，按需访问
2. **易于查找**: 索引文件快速定位
3. **节省空间**: 详细数据可选，可压缩
4. **便于分析**: CSV格式便于数据分析工具处理
5. **灵活扩展**: 可根据需要添加新的数据文件


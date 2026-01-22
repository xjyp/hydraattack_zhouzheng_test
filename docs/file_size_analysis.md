# 运行过程中的文件大小分析

## 📊 当前已实现文件的大小估算

### 1. 模型文件（必须保存，中等大小）
**文件**: `fast_rl_attacker_{benchmark}.pth`

- **大小**: 约 **5-20 MB**（取决于网络结构）
- **位置**: 实验目录根目录
- **说明**: Rainbow DQN模型权重，包含主网络和目标网络
- **是否可以删除**: ❌ 不建议，这是训练结果的核心
- **建议**: 如果空间紧张，可以只保留最佳模型

### 2. 日志文件（可能较大）
**文件**: `logs/hydra_attack_{timestamp}.log`

- **大小**: 取决于训练轮数和日志详细程度
  - 1200 episodes: 约 **10-50 MB**
  - 如果训练时间长，可能达到 **100-500 MB**
- **位置**: `logs/` 目录
- **说明**: 训练过程的详细文本日志
- **是否可以删除**: ✅ 训练完成后可以删除或压缩归档
- **建议**: 
  - 训练完成后压缩: `gzip logs/*.log`
  - 或定期清理旧日志

### 3. HydraLogger 生成的其他文件
**文件**: 
- `logs/hydra_attack_{timestamp}_config.json` - 约 **1-10 KB**
- `logs/hydra_attack_{timestamp}_samples.json` - 如果记录样本，可能 **1-10 MB**

- **位置**: `logs/` 目录
- **是否可以删除**: ✅ 如果需要，可以删除（但建议保留config.json）

### 4. 配置文件（很小）
**文件**: 
- `config/training_config_{benchmark}.json` - 约 **5-20 KB**
- `config/action_mapping_{benchmark}.json` - 约 **1-5 KB**
- `training_config_{benchmark}.json` (根目录备份) - 约 **5-20 KB**
- `action_mapping_{benchmark}.json` (根目录备份) - 约 **1-5 KB**

- **位置**: `config/` 目录和根目录
- **是否可以删除**: ❌ 不建议，这些是重要的元数据
- **建议**: 这些文件很小，保留即可

### 5. README文件（很小）
**文件**: `README.md`

- **大小**: 约 **1-5 KB**
- **位置**: 实验目录根目录
- **是否可以删除**: ❌ 不建议，很有用

## 📈 单个实验目录的总大小估算

### 当前实现（最小配置）
```
模型文件:            ~10-20 MB
日志文件:            ~10-50 MB
配置文件:            ~0.1 MB
README:              ~0.01 MB
─────────────────────────────────
总计（单个benchmark）: ~20-70 MB
```

### 3个benchmark的总大小
```
单次运行（3个benchmark）: ~60-210 MB
```

## ⚠️ 潜在的大文件（待实现功能）

### 如果实现详细记录功能，可能产生的大文件：

#### 1. Episode详细数据（待实现）
**文件**: `training/episodes/episode_*.json`

- **大小**: 如果每50个episode保存一次，1200 episodes = 24个文件
- 每个文件约 **100 KB - 1 MB**
- **总计**: 约 **2-24 MB**（取决于保存详细程度）
- **建议**: 只在需要时启用

#### 2. 测试样本详细记录（待实现）
**文件**: `evaluation/test_samples/successful/*.json` 和 `failed/*.json`

- **大小**: 取决于测试集大小
  - 假设测试集1000个样本，每个样本约 **5-50 KB**
  - **总计**: 约 **5-50 MB**
- **建议**: 只在需要case study时保存

#### 3. Judge详细日志（待实现）
**文件**: `judge_logs/training/detailed/judge_call_*.json` 和 `judge_logs/test/detailed/judge_call_*.json`

- **大小**: 可能非常大！
  - 训练阶段：1200 episodes × 平均5次查询 = 6000次judge调用
  - 测试阶段：假设1000个样本 × 平均3次查询 = 3000次judge调用
  - 每次judge调用记录约 **5-20 KB**（包含完整的prompt和response）
  - **训练阶段总计**: 约 **30-120 MB**
  - **测试阶段总计**: 约 **15-60 MB**
  - **总计**: 约 **45-180 MB**
- **建议**: 
  - ⚠️ **这是最大的潜在文件**，只在确实需要时启用
  - 可以考虑只保存失败或异常的judge调用
  - 或使用压缩存储

#### 4. 训练检查点（如果实现）
**文件**: `training/checkpoints/checkpoint_*.pth`

- **大小**: 每个检查点约 **10-20 MB**（与模型文件相同）
- 如果每50个episode保存一次，1200 episodes = 24个检查点
- **总计**: 约 **240-480 MB**
- **建议**: 
  - ⚠️ **非常占用空间**，建议只保存最佳模型和几个关键检查点
  - 或实现检查点轮转（只保留最新的N个）

## 💾 磁盘空间管理建议

### 1. 最小配置（当前实现）
- **每个benchmark**: ~20-70 MB
- **3个benchmark**: ~60-210 MB
- **10次运行**: ~600 MB - 2 GB
- **建议**: ✅ 空间占用合理，可以保留

### 2. 如果启用详细记录（未来）
- **每个benchmark**: ~100-500 MB
- **3个benchmark**: ~300 MB - 1.5 GB
- **10次运行**: ~3-15 GB
- **建议**: ⚠️ 需要注意空间管理

### 3. 如果启用检查点保存（未来）
- **每个benchmark**: ~300 MB - 1 GB
- **3个benchmark**: ~1-3 GB
- **10次运行**: ~10-30 GB
- **建议**: ⚠️ 需要谨慎管理，建议只保存关键检查点

## 🛠️ 空间节省策略

### 1. 压缩日志文件
```bash
# 训练完成后压缩日志
gzip results_rainbowdqn/*/logs/*.log
```

### 2. 定期清理旧实验
```bash
# 只保留最近N次运行的结果
# 删除超过30天的旧结果
find results_rainbowdqn/ -type d -mtime +30 -exec rm -rf {} \;
```

### 3. 选择性保存
- 如果空间紧张，只保留模型文件和配置文件
- 删除日志文件（如果需要可以重新训练查看）

### 4. 使用符号链接
- 将大文件移动到其他磁盘，创建符号链接

### 5. 实现数据压缩
- 如果实现详细记录功能，使用 `.json.gz` 格式保存

## 📋 当前运行的空间占用检查

### 检查当前目录大小
```bash
# 检查单个实验目录大小
du -sh results_rainbowdqn/rl_generation_rainbowdqn_test_*/

# 检查所有rainbowdqn结果总大小
du -sh results_rainbowdqn/

# 检查所有结果目录（包括baseline）
du -sh results/ results_rainbowdqn/
```

### 找出最大的文件
```bash
# 找出最大的10个文件
find results_rainbowdqn/ -type f -exec du -h {} + | sort -rh | head -10
```

## ⚡ 快速参考

### 必须保留的文件（小）
- ✅ 模型文件: `fast_rl_attacker_{benchmark}.pth` (~10-20 MB)
- ✅ 配置文件: `config/training_config_{benchmark}.json` (~5-20 KB)
- ✅ README: `README.md` (~1-5 KB)

### 可以删除的文件（节省空间）
- ✅ 日志文件: `logs/*.log` (~10-50 MB)
- ✅ HydraLogger样本文件: `logs/*_samples.json` (如果存在)

### 待实现功能的空间占用（如果启用）
- ⚠️ Judge详细日志: ~45-180 MB
- ⚠️ 测试样本详细记录: ~5-50 MB
- ⚠️ Episode详细数据: ~2-24 MB
- ⚠️ 训练检查点: ~240-480 MB（如果启用）

## 🎯 总结

**当前实现（最小配置）**:
- 单次运行（3个benchmark）: ~60-210 MB
- 空间占用合理，可以保留多次运行的结果

**如果启用所有详细功能**:
- 单次运行（3个benchmark）: ~300 MB - 1.5 GB
- 需要定期清理或压缩

**建议**:
1. 当前实现可以放心使用，空间占用不大
2. 如果实现详细记录功能，建议：
   - 添加可配置选项，默认关闭详细记录
   - 使用压缩格式保存大文件
   - 实现定期清理机制


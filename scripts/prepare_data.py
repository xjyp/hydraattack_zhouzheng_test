#!/usr/bin/env python3
"""
数据准备脚本
确保所有方法使用相同的数据分割
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any


def prepare_benchmark_data(benchmark: str, max_samples, random_seed: int = 42, train_ratio: float = 0.8, test_ratio: float = 0.2) -> None:
    """准备benchmark数据，确保数据分割一致性"""
    
    # 对于 llmbar 和 mtbench，数据已经存在于 split 目录中，直接跳过处理
    if benchmark in ["llmbar", "mtbench"]:
        train_file = f"data/split/{benchmark}_train.json"
        test_file = f"data/split/{benchmark}_test.json"
        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"✅ {benchmark} 数据已存在于 split 目录中，跳过处理")
            return
        else:
            print(f"⚠️  {benchmark} 数据文件不完整，需要重新处理")
    
    data_file = f"data/processed/{benchmark}_processed.json"
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    # 加载原始数据
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        print(f"❌ {benchmark} 数据为空")
        return
    
    print(f"📊 原始数据样本数: {len(data)}")
    
    # 设置随机种子确保数据一致性
    random.seed(random_seed)
    
    # 随机打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 选择指定数量的样本
    if max_samples == "full" or max_samples is None:
        selected_data = shuffled_data
        print(f"📊 使用全量数据: {len(selected_data)} 样本")
    else:
        selected_data = shuffled_data[:max_samples]
        print(f"📊 选择样本数: {len(selected_data)}")
    
    # 计算分割点
    val_ratio = 0.0
    
    train_end = int(len(selected_data) * train_ratio)
    
    train_data = selected_data[:train_end]
    test_data = selected_data[train_end:]
    
    print(f"📊 数据分割:")
    print(f"  - 训练集: {len(train_data)} 样本 ({len(train_data)/len(selected_data)*100:.1f}%)")
    print(f"  - 测试集: {len(test_data)} 样本 ({len(test_data)/len(selected_data)*100:.1f}%)")
    
    # 保存分割后的数据
    os.makedirs("data/split", exist_ok=True)
    
    # 保存训练集
    train_file = f"data/split/{benchmark}_train.json"
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"💾 训练集已保存到: {train_file}")
    
    
    # 保存测试集
    test_file = f"data/split/{benchmark}_test.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"💾 测试集已保存到: {test_file}")
    
    # 保存数据分割信息
    split_info = {
        "benchmark": benchmark,
        "total_samples": len(selected_data),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "random_seed": random_seed,
        "data_files": {
            "train": train_file,
            "test": test_file
        }
    }
    
    split_info_file = f"data/split/{benchmark}_split_info.json"
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"💾 分割信息已保存到: {split_info_file}")
    
    print(f"✅ {benchmark} 数据准备完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="准备数据，确保所有方法使用相同的数据分割")
    parser.add_argument("--benchmark", type=str, default="alpaca_eval", 
                       choices=["alpaca_eval", "arena_hard", "code_judge_bench", "llmbar", "mtbench"],
                       help="测试的benchmark")
    parser.add_argument("--max_samples", default=200, help="最大样本数，使用 'full' 表示使用全量数据")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    print("🚀 数据准备：确保所有方法使用相同的数据分割")
    print("=" * 60)
    print(f"📊 配置信息:")
    print(f"  - Benchmark: {args.benchmark}")
    if args.max_samples == "full":
        print(f"  - 数据模式: 全量数据")
    else:
        print(f"  - 最大样本数: {args.max_samples}")
    print(f"  - 训练集比例: {args.train_ratio}")
    print(f"  - 测试集比例: {args.test_ratio}")
    print(f"  - 随机种子: {args.random_seed}")
    print("")
    
    prepare_benchmark_data(args.benchmark, args.max_samples, args.random_seed, args.train_ratio, args.test_ratio)
    
    print("")
    print("🎉 数据准备完成！")
    print("=" * 60)
    print("📁 生成的文件:")
    print(f"  - 训练集: data/split/{args.benchmark}_train.json")
    print(f"  - 测试集: data/split/{args.benchmark}_test.json")
    print(f"  - 分割信息: data/split/{args.benchmark}_split_info.json")
    print("")
    print("💡 所有方法现在将使用相同的测试集进行公平对比！")


if __name__ == "__main__":
    main()

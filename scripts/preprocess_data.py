#!/usr/bin/env python3
"""
数据预处理脚本
从raw_data处理到data/processed
"""

import argparse
import sys
import os
from pathlib import Path

# 添加src到路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.processors import AlpacaEvalProcessor, CodeJudgeBenchProcessor, ArenaHardProcessor
from data_types import BenchmarkType


def main():
    parser = argparse.ArgumentParser(description="预处理benchmark数据")
    parser.add_argument("--benchmark", type=str, required=True,
                       choices=["alpaca_eval", "code_judge_bench", "arena_hard", "all"],
                       help="要处理的benchmark")
    parser.add_argument("--data_dir", type=str, default="./raw_data",
                       help="原始数据目录")
    parser.add_argument("--output_dir", type=str, default="./data/processed",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    processors = []
    
    if args.benchmark == "alpaca_eval" or args.benchmark == "all":
        print("Processing AlpacaEval data...")
        processor = AlpacaEvalProcessor(args.data_dir, args.output_dir)
        processors.append(processor)
    
    
    if args.benchmark == "code_judge_bench" or args.benchmark == "all":
        print("Processing CodeJudgeBench data...")
        processor = CodeJudgeBenchProcessor(args.data_dir, args.output_dir)
        processors.append(processor)
    
    if args.benchmark == "arena_hard" or args.benchmark == "all":
        print("Processing Arena-Hard data...")
        processor = ArenaHardProcessor(args.data_dir, args.output_dir)
        processors.append(processor)
    
    # 处理数据
    for processor in processors:
        try:
            print(f"Processing {processor.get_benchmark_type().value}...")
            examples = processor.process()
            print(f"Successfully processed {len(examples)} examples")
        except Exception as e:
            print(f"Error processing {processor.get_benchmark_type().value}: {e}")
            continue
    
    print("Data preprocessing completed!")


if __name__ == "__main__":
    main()

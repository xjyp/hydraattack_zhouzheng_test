#!/bin/bash

# PAIR攻击baseline运行脚本
# 使用PAIR算法在instruction上进行攻击，目标是反转judge模型的输出

set -e

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 默认参数
BENCHMARKS=("llmbar" "mtbench") # 支持多个benchmark
# BENCHMARKS=("code_judge_bench") # 支持多个benchmark

MAX_TEST_SAMPLES="full"

JUDGE_MODEL_PATH="/root/autodl-tmp/Qwen2.5-3B-Instruct"
# JUDGE_MODEL_PATH="/share/disk/llm_cache/gemma-3-4b-it"
# JUDGE_MODEL_PATH="/share/disk/llm_cache/gemma-3-12b-it"


ATTACK_MODEL_PATH="/root/autodl-tmp/gemma-3-4b-it" # 这个固定下来不要改
DEVICE="cuda"
N_STREAMS=3
MAX_ATTEMPTS=5
MAX_TOKENS=500
BASE_OUTPUT_DIR="results_pair_baseline"
RANDOM_SEED=42

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmarks)
            # 支持多个benchmark，用空格分隔
            IFS=' ' read -ra BENCHMARKS <<< "$2"
            shift 2
            ;;
        --max_test_samples)
            MAX_TEST_SAMPLES="$2"
            shift 2
            ;;
        --attack_model_path)
            ATTACK_MODEL_PATH="$2"
            shift 2
            ;;
        --judge_model_path)
            JUDGE_MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --n_streams)
            N_STREAMS="$2"
            shift 2
            ;;
        --max_attempts)
            MAX_ATTEMPTS="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --output_dir)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --random_seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "PAIR攻击baseline运行脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --benchmarks          基准测试类型，多个用空格分隔 (alpaca_eval arena_hard code_judge_bench) [默认: arena_hard alpaca_eval code_judge_bench]"
            echo "  --max_test_samples    最大测试样本数 [默认: 50]"
            echo "  --attack_model_path   攻击模型路径 [默认: /share/disk/llm_cache/Qwen2.5-7B-Instruct]"
            echo "  --judge_model_path    Judge模型路径 [默认: /share/disk/llm_cache/Qwen2.5-7B-Instruct]"
            echo "  --device              设备 [默认: cuda]"
            echo "  --n_streams           并发流数量 [默认: 3]"
            echo "  --max_attempts        单条数据的最大尝试次数 [默认: 5]"
            echo "  --max_tokens          最大生成token数 [默认: 500]"
            echo "  --output_dir          输出目录 [默认: results_pair_baseline]"
            echo "  --random_seed         随机种子 [默认: 42]"
            echo "  -h, --help            显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 打印配置信息
echo "🚀 启动PAIR攻击baseline"
echo "================================"
echo "执行时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "工作目录: $(pwd)"
echo "Python路径: $PYTHONPATH"
echo "CUDA设备: $CUDA_VISIBLE_DEVICES"
echo ""
echo "📋 超参数配置:"
echo "  - 基准测试: ${BENCHMARKS[*]}"
echo "  - 测试样本数: $MAX_TEST_SAMPLES"
echo "  - 攻击模型路径: $ATTACK_MODEL_PATH"
echo "  - Judge模型路径: $JUDGE_MODEL_PATH"
echo "  - 设备: $DEVICE"
echo "  - 并发流数: $N_STREAMS"
echo "  - 最大尝试次数: $MAX_ATTEMPTS"
echo "  - 最大token数: $MAX_TOKENS"
echo "  - 基础输出目录: $BASE_OUTPUT_DIR"
echo "  - 随机种子: $RANDOM_SEED"
echo "================================"

# 检查模型路径
if [ ! -d "$ATTACK_MODEL_PATH" ]; then
    echo "❌ 攻击模型路径不存在: $ATTACK_MODEL_PATH"
    exit 1
fi

if [ ! -d "$JUDGE_MODEL_PATH" ]; then
    echo "❌ Judge模型路径不存在: $JUDGE_MODEL_PATH"
    exit 1
fi

# 创建基础输出目录
mkdir -p "$BASE_OUTPUT_DIR"

# 记录开始时间
START_TIME=$(date +%s)

# 对每个benchmark进行测试
for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo ""
    echo "🎯 开始测试 Benchmark: ${BENCHMARK}"
    echo "=========================================="
    
    # 为每个benchmark创建独立的输出目录
    BENCHMARK_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${BENCHMARK}_${BENCHMARK_TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"
    echo "📁 输出目录: $OUTPUT_DIR"
    
    # 检查数据文件是否存在
    TEST_DATA_FILE="data/split/${BENCHMARK}_test.json"
    
    if [ ! -f "$TEST_DATA_FILE" ]; then
        echo "❌ 测试数据文件不存在: $TEST_DATA_FILE"
        echo "请先运行 prepare_data.py 脚本准备数据分割"
        echo "跳过 ${BENCHMARK}"
        continue
    fi
    
    # 运行PAIR攻击baseline
    echo "🎯 开始运行PAIR攻击baseline..."
    echo "----------------------------------------"
    echo "执行命令:"
    echo "  python baselines/pair.py \\"
    echo "    --benchmark $BENCHMARK \\"
    echo "    --max_test_samples $MAX_TEST_SAMPLES \\"
    echo "    --attack_model_path $ATTACK_MODEL_PATH \\"
    echo "    --judge_model_path $JUDGE_MODEL_PATH \\"
    echo "    --device $DEVICE \\"
    echo "    --n_streams $N_STREAMS \\"
    echo "    --max_attempts $MAX_ATTEMPTS \\"
    echo "    --max_tokens $MAX_TOKENS \\"
    echo "    --output_dir $OUTPUT_DIR \\"
    echo "    --random_seed $RANDOM_SEED"
    echo "----------------------------------------"
    
    BENCHMARK_START_TIME=$(date +%s)
    
    if python baselines/pair.py \
        --benchmark "$BENCHMARK" \
        --max_test_samples "$MAX_TEST_SAMPLES" \
        --attack_model_path "$ATTACK_MODEL_PATH" \
        --judge_model_path "$JUDGE_MODEL_PATH" \
        --device "$DEVICE" \
        --n_streams "$N_STREAMS" \
        --max_attempts "$MAX_ATTEMPTS" \
        --max_tokens "$MAX_TOKENS" \
        --output_dir "$OUTPUT_DIR" \
        --random_seed "$RANDOM_SEED"; then
        
        BENCHMARK_END_TIME=$(date +%s)
        BENCHMARK_DURATION=$((BENCHMARK_END_TIME - BENCHMARK_START_TIME))
        BENCHMARK_HOURS=$((BENCHMARK_DURATION / 3600))
        BENCHMARK_MINUTES=$(((BENCHMARK_DURATION % 3600) / 60))
        BENCHMARK_SECONDS=$((BENCHMARK_DURATION % 60))
        
        echo ""
        echo "✅ ${BENCHMARK} PAIR攻击baseline运行完成!"
        echo "=========================================="
        echo "运行时间: ${BENCHMARK_HOURS}小时${BENCHMARK_MINUTES}分钟${BENCHMARK_SECONDS}秒"
        echo "结果保存在: $OUTPUT_DIR"
        
        # 显示结果文件并提取统计信息
        echo ""
        echo "📁 生成的结果文件:"
        RESULT_FILE=$(ls -t "$OUTPUT_DIR"/*.json 2>/dev/null | head -1)
        if [ -n "$RESULT_FILE" ]; then
            echo "  $RESULT_FILE"
            echo ""
            echo "📊 结果统计信息:"
            # 使用Python提取JSON中的统计信息
            python3 << EOF
import json
import sys
try:
    with open("$RESULT_FILE", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  - 总样本数: {data.get('total_samples', 'N/A')}")
    print(f"  - 成功攻击数: {data.get('successful_attacks', 'N/A')}")
    print(f"  - 失败攻击数: {data.get('failed_attacks', 'N/A')}")
    print(f"  - 攻击成功率: {data.get('attack_success_rate', data.get('success_rate', 0)) * 100:.2f}%")
    print(f"  - 平均查询次数(所有攻击): {data.get('avg_queries_used', 0):.2f}")
    print(f"  - 平均查询次数(成功攻击): {data.get('avg_queries_successful', 0):.2f}")
    print(f"  - 总查询次数: {data.get('total_queries', 'N/A')}")
    
    if 'hyperparameters' in data:
        print(f"")
        print(f"  - 超参数:")
        hyperparams = data['hyperparameters']
        for key, value in hyperparams.items():
            print(f"    * {key}: {value}")
except Exception as e:
    print(f"  ⚠️  无法读取结果文件: {e}")
EOF
        else
            echo "  ⚠️  无结果文件"
        fi
        echo "=========================================="
    else
        BENCHMARK_END_TIME=$(date +%s)
        BENCHMARK_DURATION=$((BENCHMARK_END_TIME - BENCHMARK_START_TIME))
        echo ""
        echo "❌ ${BENCHMARK} PAIR攻击baseline运行失败!"
        echo "运行时间: ${BENCHMARK_DURATION}秒"
        echo "跳过 ${BENCHMARK}"
    fi
done

# 计算总运行时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "🎉 所有Benchmark测试完成!"
echo "=========================================="
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总运行时间: ${HOURS}小时${MINUTES}分钟${SECONDS}秒"
echo "基础结果目录: $BASE_OUTPUT_DIR"
echo ""
echo "📊 测试的Benchmarks:"
for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "  - ${BENCHMARK}"
done
echo ""
echo "📁 结果文件结构:"
echo "  - 基础结果目录: $BASE_OUTPUT_DIR"
echo "  - 每个benchmark的实验文件夹: $BASE_OUTPUT_DIR/*/"
echo "  - 实验文件夹包含:"
echo "    * PAIR攻击结果: pair_baseline_*.json"
echo ""
echo "📋 使用的超参数:"
echo "  - 基准测试: ${BENCHMARKS[*]}"
echo "  - 测试样本数: $MAX_TEST_SAMPLES"
echo "  - 攻击模型路径: $ATTACK_MODEL_PATH"
echo "  - Judge模型路径: $JUDGE_MODEL_PATH"
echo "  - 设备: $DEVICE"
echo "  - 并发流数: $N_STREAMS"
echo "  - 最大尝试次数: $MAX_ATTEMPTS"
echo "  - 最大token数: $MAX_TOKENS"
echo "  - 随机种子: $RANDOM_SEED"
echo ""
echo "💡 提示: 每个结果JSON文件都包含完整的超参数配置和统计信息"
echo "=========================================="

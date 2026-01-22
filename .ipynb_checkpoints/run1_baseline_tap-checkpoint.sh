#!/bin/bash

# TAP攻击baseline运行脚本
# 使用Tree of Attacks with Pruning (TAP)算法在instruction上进行攻击，目标是反转judge模型的输出

set -e

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 默认参数
BENCHMARKS=("arena_hard" "alpaca_eval" "code_judge_bench" "llmbar" "mtbench") # 支持多个benchmark
# BENCHMARKS=("arena_hard") # 支持多个benchmark
# BENCHMARKS=("alpaca_eval") # 支持多个benchmark
# BENCHMARKS=("code_judge_bench") # 支持多个benchmark

# MAX_TEST_SAMPLES="full"
MAX_TEST_SAMPLES="3"

ATTACK_MODEL_PATH="/root/autodl-tmp/Qwen2.5-3B-Instruct"
JUDGE_MODEL_PATH="/root/autodl-tmp/Qwen2.5-3B-Instruct"
DEVICE="cuda"
DEPTH=5                      # 树搜索深度
WIDTH=10                     # 每层保留的候选数（优化后：增加探索广度）
BRANCHING_FACTOR=3           # 分支因子（优化后：从1改为3，恢复树搜索优势）
N_STREAMS=2                  # 根节点数量（优化后：增加并行探索）
MAX_TOKENS=500
MAX_QUERIES=20
BASE_OUTPUT_DIR="results_tap_baseline"
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
        --depth)
            DEPTH="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --branching_factor)
            BRANCHING_FACTOR="$2"
            shift 2
            ;;
        --n_streams)
            N_STREAMS="$2"
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
            echo "TAP攻击baseline运行脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --benchmarks          基准测试类型，多个用空格分隔 (alpaca_eval arena_hard code_judge_bench) [默认: arena_hard alpaca_eval code_judge_bench]"
            echo "  --max_test_samples    最大测试样本数 [默认: full]"
            echo "  --attack_model_path   攻击模型路径 [默认: /share/disk/llm_cache/Qwen3-8B]"
            echo "  --judge_model_path    Judge模型路径 [默认: /share/disk/llm_cache/Qwen3-8B]"
            echo "  --device              设备 [默认: cuda]"
            echo "  --depth               树搜索深度 [默认: 5]"
            echo "  --width               每层保留的候选数 [默认: 10]"
            echo "  --branching_factor    分支因子 [默认: 3]"
            echo "  --n_streams           根节点数量 [默认: 2]"
            echo "  --max_tokens          最大生成token数 [默认: 500]"
            echo "  --output_dir          输出目录 [默认: results_tap_baseline]"
            echo "  --random_seed         随机种子 [默认: 42]"
            echo "  -h, --help            显示此帮助信息"
            echo ""
            echo "TAP参数说明:"
            echo "  --depth: 控制树的搜索深度，越深搜索得越彻底但耗时越长"
            echo "  --width: 每层保留的候选数，控制探索广度"
            echo "  --branching_factor: 每个节点生成的分支数，控制树的宽度"
            echo "  --n_streams: 并发的根节点数量，类似于多个起始点"
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
echo "🚀 启动TAP攻击baseline"
echo "================================"
echo "基准测试: ${BENCHMARKS[*]}"
echo "测试样本数: $MAX_TEST_SAMPLES"
echo "攻击模型: $ATTACK_MODEL_PATH"
echo "Judge模型: $JUDGE_MODEL_PATH"
echo "设备: $DEVICE"
echo "树搜索深度: $DEPTH"
echo "每层候选数: $WIDTH"
echo "分支因子: $BRANCHING_FACTOR"
echo "根节点数: $N_STREAMS"
echo "最大token数: $MAX_TOKENS"
echo "最大查询次数: $MAX_QUERIES"
echo "基础输出目录: $BASE_OUTPUT_DIR"
echo "随机种子: $RANDOM_SEED"
echo "Phase1剪枝: disabled (judge-guided pruning)"
echo "Early stop: enabled (stop when preference flips)"
echo "Feedback模式: PAIR-style detailed guidance"
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
    
    # 运行TAP攻击baseline
    echo "🎯 开始运行TAP攻击baseline..."
    if python baselines/tap.py \
        --benchmark "$BENCHMARK" \
        --max_test_samples "$MAX_TEST_SAMPLES" \
        --attack_model_path "$ATTACK_MODEL_PATH" \
        --judge_model_path "$JUDGE_MODEL_PATH" \
        --device "$DEVICE" \
        --depth "$DEPTH" \
        --width "$WIDTH" \
        --branching_factor "$BRANCHING_FACTOR" \
        --n_streams "$N_STREAMS" \
        --max_tokens "$MAX_TOKENS" \
        --max_queries "$MAX_QUERIES" \
        --output_dir "$OUTPUT_DIR" \
        --random_seed "$RANDOM_SEED"; then
        
        echo "✅ ${BENCHMARK} TAP攻击baseline运行完成!"
        echo "结果保存在: $OUTPUT_DIR"
        
        # 显示结果文件
        echo "📁 生成的结果文件:"
        ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  无结果文件"
    else
        echo "❌ ${BENCHMARK} TAP攻击baseline运行失败!"
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
echo "    * TAP攻击结果: tap_baseline_*.json"
echo "=========================================="

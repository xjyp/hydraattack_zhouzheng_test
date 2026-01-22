#!/bin/bash

# Baseline方法测试脚本 - 修复版本
# 确保使用正确的20%测试数据，避免之前的数据分割问题
# 测试alpaca_eval, arena_hard, code_judge_bench, llmbar, mtbench

set -e

# GPU配置
CUDA_VISIBLE_DEVICES="1"

# 配置
CONDA_ENV="hydraattack"
PROJECT_DIR="/home/wzdou/project/hydraattack_share"
BENCHMARKS=("arena_hard" "alpaca_eval" "code_judge_bench")  # "arena_hard" "alpaca_eval" "code_judge_bench" "llmbar" "mtbench"
# BENCHMARKS=("arena_hard" "alpaca_eval" "code_judge_bench" "llmbar" "mtbench")  # "arena_hard" "alpaca_eval" "code_judge_bench" "llmbar" "mtbench"
# BENCHMARKS=("llmbar" "mtbench")  # "arena_hard" "alpaca_eval" "code_judge_bench" "llmbar" "mtbench"

ATTACK_METHODS=("all") # "all" "flip_attack_fcs" "flip_attack_fcw" "flip_attack_fwo" "uncertainty_attack" "position_attack" "distractor_attack" "prompt_injection_attack" "marker_injection_attack" "formatting_attack" "authority_attack" "unicode_attack" "cot_poisoning_attack" "emoji_attack"

# 数据配置 - 使用全量数据并正确分割
TOTAL_SAMPLES="full"
TRAIN_RATIO=0.8
TEST_RATIO=0.2

# 实验配置
# JUDGE_MODEL_PATH="/share/disk/llm_cache/glm-4-9b-chat-hf"
# JUDGE_MODEL_PATH="/share/disk/llm_cache/Qwen2.5-0.5B-Instruct"
# JUDGE_MODEL_PATH="/share/disk/llm_cache/Qwen2.5-7B-Instruct"
# JUDGE_MODEL_PATH="/share/disk/llm_cache/Qwen3-4B-Instruct-2507"
JUDGE_MODEL_PATH="/share/disk/llm_cache/Llama-3.1-8B-Instruct"
# JUDGE_MODEL_PATH="/share/disk/llm_cache/Mistral-7B-Instruct-v0.3"
# JUDGE_MODEL_PATH="/share/disk/llm_cache/gemma-3-1b-it"
# JUDGE_MODEL_PATH="/share/disk/llm_cache/gemma-3-4b-it"
# JUDGE_MODEL_PATH="/share/disk/llm_cache/gemma-3-12b-it"
JUDGE_TYPE="llama"

MAX_QUERIES=5  # 统一所有方法的查询次数限制为5次
RANDOM_SEED=42

echo "🔧 Baseline方法测试脚本 - 修复版本"
echo "=========================================="
echo "📊 配置信息:"
echo "  - Benchmarks: ${BENCHMARKS[*]}"
echo "  - 数据模式: 全量数据"
echo "  - 训练比例: ${TRAIN_RATIO} (80%)"
echo "  - 测试比例: ${TEST_RATIO} (20%)"
echo "  - 最大查询次数: ${MAX_QUERIES}"
echo "  - GPU设备: ${CUDA_VISIBLE_DEVICES}"
echo ""

# 设置GPU环境变量
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

# 激活conda环境
echo "📦 激活conda环境: ${CONDA_ENV}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

cd "${PROJECT_DIR}"

# 创建时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_RESULTS_DIR="results/baseline_test_${TIMESTAMP}"

# 创建必要目录
mkdir -p models "${BASE_RESULTS_DIR}" logs data/split

# 记录开始时间
START_TIME=$(date)
echo "🕐 开始时间: ${START_TIME}"
echo "📁 基础结果目录: ${BASE_RESULTS_DIR}"

# 对每个benchmark进行测试
for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo ""
    echo "🎯 开始测试 Benchmark: ${BENCHMARK}"
    echo "=========================================="
    
    # 为每个benchmark创建独立的实验文件夹
    BENCHMARK_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXPERIMENT_DIR="${BASE_RESULTS_DIR}/${BENCHMARK}_${BENCHMARK_TIMESTAMP}"
    mkdir -p "${EXPERIMENT_DIR}"
    echo "📁 实验目录: ${EXPERIMENT_DIR}"
    
    # 1. 数据预处理 - 确保数据正确分割
    echo "📊 步骤1: 数据预处理..."
    echo "  🔄 使用全量数据模式，正确分割为80%训练，20%测试"
    if ! python scripts/prepare_data.py \
        --benchmark ${BENCHMARK} \
        --max_samples ${TOTAL_SAMPLES} \
        --train_ratio ${TRAIN_RATIO} \
        --test_ratio ${TEST_RATIO} \
        --random_seed ${RANDOM_SEED}; then
        echo "  ❌ 数据预处理失败，跳过 ${BENCHMARK}"
        continue
    fi
    
    # 验证数据分割是否正确
    if [ -f "data/split/${BENCHMARK}_test.json" ]; then
        TEST_SAMPLES=$(python -c "
import json
try:
    with open('data/split/${BENCHMARK}_test.json', 'r') as f:
        data = json.load(f)
    print(len(data))
except Exception as e:
    print('ERROR:', str(e))
    exit(1)
")
        if [[ "$TEST_SAMPLES" =~ ^[0-9]+$ ]]; then
            echo "  ✅ 测试数据已准备: ${TEST_SAMPLES} 个样本"
        else
            echo "  ❌ 测试数据格式错误: ${TEST_SAMPLES}"
            continue
        fi
    else
        echo "  ❌ 测试数据文件不存在"
        continue
    fi
    
    # 2. 测试所有14种Baseline方法
    echo "⚔️  步骤2: 测试所有14种Baseline方法..."
    echo "  🔄 使用20%测试数据测试所有baseline方法"
    
    # 使用run_attack.py直接测试，确保使用正确的测试数据
    if ! python scripts/run_attack.py \
        --benchmarks ${BENCHMARK} \
        --max_samples ${TOTAL_SAMPLES} \
        --attack_methods ${ATTACK_METHODS[*]} \
        --judge_model_path ${JUDGE_MODEL_PATH} \
        --judge_type ${JUDGE_TYPE} \
        --baseline_max_queries ${MAX_QUERIES} \
        --results_dir "${EXPERIMENT_DIR}" \
        --log_dir "${EXPERIMENT_DIR}" \
        --random_seed ${RANDOM_SEED}; then
        echo "  ❌ ${BENCHMARK} 攻击测试失败，跳过"
        continue
    fi
    
    echo "✅ ${BENCHMARK} 测试完成！"
    echo "📁 实验结果保存在: ${EXPERIMENT_DIR}"
done

# 记录结束时间
END_TIME=$(date)
echo ""
echo "🎉 所有Benchmark测试完成！"
echo "=========================================="
echo "🕐 开始时间: ${START_TIME}"
echo "🕐 结束时间: ${END_TIME}"
echo ""
echo "📁 结果文件:"
echo "  - 基础结果目录: ${BASE_RESULTS_DIR}"
echo "  - 每个benchmark的实验文件夹: ${BASE_RESULTS_DIR}/*/"
echo "  - 实验文件夹包含:"
echo "    * Baseline测试结果: baseline_generalization_*.json"
echo "    * 各baseline方法结果: baseline_*_results.json"
echo "    * 测试日志: hydra_attack_*.log"
echo "    * 实验配置: experiment_config.json"
echo ""
echo "📊 测试的Benchmarks:"
for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "  - ${BENCHMARK}"
done
echo ""
echo "⚔️  测试的Baseline方法 (14种):"
echo "  - flip_attack (FCS/FWO/FCW), uncertainty_attack, position_attack"
echo "  - distractor_attack, prompt_injection_attack, marker_injection_attack"
echo "  - formatting_attack, authority_attack, unicode_attack"
echo "  - cot_poisoning_attack, emoji_attack"
echo ""
echo "🔧 修复的问题:"
echo "  ✅ 确保使用正确的20%测试数据"
echo "  ✅ 统一查询次数限制为5次"
echo "  ✅ 使用预分割的测试数据文件"
echo "  ✅ 避免数据加载逻辑的混乱"
echo "  ✅ 确保所有方法使用相同的数据集"
echo "=========================================="

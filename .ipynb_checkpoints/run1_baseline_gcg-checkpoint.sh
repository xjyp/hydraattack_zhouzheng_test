#!/bin/bash

# GCG naive baseline runner.
# Edit the configuration block below and execute the script directly.

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

# ---------------------------------------------------------------------------
# Configuration (edit these values to customize the run)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="${PYTHONPATH:-}:${WORKSPACE_DIR}/src"

# Normal training configuration
BENCHMARKS=("arena_hard" "alpaca_eval" "code_judge_bench" "llmbar" "mtbench")

# MAX_TEST_SAMPLES="full"
MAX_TEST_SAMPLES="42"

JUDGE_MODEL_PATH="/root/autodl-tmp/Qwen2.5-7B-Instruct"
DEVICE="cuda"
GCG_STEPS=100
GCG_SEARCH_WIDTH=64
VALIDATION_INTERVAL=20
RANDOM_SEED=42
BASE_OUTPUT_DIR="${WORKSPACE_DIR}/results_gcg_naive_baseline"
# ---------------------------------------------------------------------------

echo "🚀 Launching naive GCG baseline"
echo "====================================================================="
echo "Benchmarks: ${BENCHMARKS[*]}"
echo "Max test samples: ${MAX_TEST_SAMPLES}"
echo "Judge model: ${JUDGE_MODEL_PATH}"
echo "Device: ${DEVICE}"
echo "Total GCG steps: ${GCG_STEPS}"
echo "Search width: ${GCG_SEARCH_WIDTH}"
echo "Validation interval: ${VALIDATION_INTERVAL}"
echo "Random seed: ${RANDOM_SEED}"
echo "Base output dir: ${BASE_OUTPUT_DIR}"
echo "====================================================================="

if [ ! -d "${JUDGE_MODEL_PATH}" ]; then
    echo "❌ Judge model path not found: ${JUDGE_MODEL_PATH}"
    exit 1
fi

mkdir -p "${BASE_OUTPUT_DIR}"
START_TIME=$(date +%s)

for BENCHMARK in "${BENCHMARKS[@]}"; do
    DATA_FILE="${WORKSPACE_DIR}/data/split/${BENCHMARK}_test.json"
    if [ ! -f "${DATA_FILE}" ]; then
        echo "⚠️  Missing data file ${DATA_FILE}, skip benchmark ${BENCHMARK}"
        continue
    fi

    RUN_TAG=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${BENCHMARK}_${RUN_TAG}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "🎯 Running benchmark: ${BENCHMARK}"
    echo "Output dir: ${OUTPUT_DIR}"

    python baselines/gcg.py \
        --benchmark "${BENCHMARK}" \
        --max_test_samples "${MAX_TEST_SAMPLES}" \
        --judge_model_path "${JUDGE_MODEL_PATH}" \
        --device "${DEVICE}" \
        --gcg_steps "${GCG_STEPS}" \
        --gcg_search_width "${GCG_SEARCH_WIDTH}" \
        --output_dir "${OUTPUT_DIR}" \
        --random_seed "${RANDOM_SEED}" \
        --validation_interval "${VALIDATION_INTERVAL}"

    echo "📁 Result files:"
    ls -la "${OUTPUT_DIR}"/*.json 2>/dev/null || echo "  (no JSON saved)"
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "✅ All configured benchmarks finished."
echo "Elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Results root: ${BASE_OUTPUT_DIR}"


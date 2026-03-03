#!/bin/bash
# FashionIQ Evaluation Script for MyComposedRetrieval
# Usage: ./eval_fashioniq.sh [checkpoint_path] [gpu_id] [base_model]
# Examples:
#   ./eval_fashioniq.sh ./experiments/IterativeFashionIQ_qwen2_5vl_7b_20250101_120000/training_iter_1/checkpoint-200
#   ./eval_fashioniq.sh ./experiments/IterativeFashionIQ_qwen2_5vl_7b_20250101_120000/training_iter_1/checkpoint-200 0
#   ./eval_fashioniq.sh ./experiments/IterativeFashionIQ_qwen2_5vl_7b_20250101_120000/training_iter_1/checkpoint-200 0 Qwen/Qwen2.5-VL-7B-Instruct

# Default parameters
CHECKPOINT_PATH="${1:-./experiments/IterativeFashionIQ_qwen2_5vl_7b_latest/training_iter_1/checkpoint-200}"
GPU_ID="${2:-0}"
BASE_MODEL="${3:-Qwen/Qwen2.5-VL-7B-Instruct}"

echo "=========================================="
echo "FashionIQ Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "GPU: $GPU_ID"
echo "Base Model: $BASE_MODEL"
echo "=========================================="

# Check if checkpoint exists
if [[ ! -d "$CHECKPOINT_PATH" ]]; then
  echo "Error: Checkpoint directory '$CHECKPOINT_PATH' does not exist."
  exit 1
fi

# Set environment variable
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run evaluation
python eval_fashioniq.py \
    --model_path "$CHECKPOINT_PATH" \
    --base_model_name "$BASE_MODEL" \
    --eval_config configs/fashioniq_eval_config.yaml \
    --batch_size 16 \
    --device cuda \
    --verbose True

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $CHECKPOINT_PATH/fashioniq_eval_results.json"
echo "=========================================="

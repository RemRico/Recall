#!/usr/bin/env bash
# FashionIQ Training Script for MyComposedRetrieval
# Usage:
#   ./run_fashioniq_training.sh [model_type] [num_gpus] [existing_exp_dir] [use_minimal_prompt]
# Examples:
#   ./run_fashioniq_training.sh qwen2_5vl_7b 8
#   ./run_fashioniq_training.sh qwen3vl_8b 4
#   ./run_fashioniq_training.sh qwen2_5vl_7b 8 ./experiments/IterativeFashionIQ_qwen2_5vl_7b_20250101_120000
#   ./run_fashioniq_training.sh qwen2_5vl_7b 8 "" true  # Use minimal prompt

set -euo pipefail

# -------------------------- Configuration --------------------------
MODEL_TYPE=${1:-"qwen2_5vl_7b"}   # qwen2vl | qwen2vl_2b | qwen2_5vl_7b | qwen3vl_8b
NUM_GPUS=${2:-8}                   # default 8 GPUs
EXISTING_EXP_DIR=${3:-""}          # optional: resume from existing experiment dir
USE_MINIMAL_PROMPT=${4:-"true"}    # true | false - use minimal rewrite prompt (default: true)

# LoRA settings
LORA_R=16
LORA_DROPOUT=0.1

# Local model paths (adjust these to your local paths)
QWEN2VL_2B_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-2B-Instruct"
QWEN2VL_7B_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct"
QWEN2_5VL_7B_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2.5-VL-7B-Instruct"

echo "==> Starting FashionIQ iterative training"
echo "    MODEL_TYPE        : $MODEL_TYPE"
echo "    NUM_GPUS          : $NUM_GPUS"
echo "    RESUME_DIR        : ${EXISTING_EXP_DIR:-<new experiment>}"
echo "    LoRA              : r=$LORA_R, dropout=$LORA_DROPOUT"
echo "    USE_MINIMAL_PROMPT: $USE_MINIMAL_PROMPT"
echo "==> Environment"
echo "python : $(which python)"
python --version || true

# Hugging Face / WandB env
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export WANDB_DISABLED=false
export WANDB_PROJECT="fashioniq_composed_retrieval"

# ----------------------- Experiment Resolution ----------------------
CONFIG_FILE="configs/fashioniq_iterative.yaml"

if [[ -n "$EXISTING_EXP_DIR" ]]; then
  if [[ ! -d "$EXISTING_EXP_DIR" ]]; then
    echo "Error: Existing experiment directory '$EXISTING_EXP_DIR' does not exist."
    exit 1
  fi
  EXP_NAME="$(basename "$EXISTING_EXP_DIR")"
  export EXP_DIR="$EXISTING_EXP_DIR"
  echo "==> Resuming existing experiment: $EXP_NAME"
  echo "==> Using directory           : $EXP_DIR"
else
  EXP_NAME="IterativeFashionIQ_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
  export EXP_DIR="./experiments/$EXP_NAME"
  echo "==> Creating new experiment: $EXP_NAME"
fi

# ------------------------- Model Resolution -------------------------
case "$MODEL_TYPE" in
  qwen2vl)
    MODEL_NAME="$QWEN2VL_7B_PATH"
    FOUNDATION_MODEL="$QWEN2VL_7B_PATH"
    ;;
  qwen2vl_2b)
    MODEL_NAME="$QWEN2VL_2B_PATH"
    FOUNDATION_MODEL="$QWEN2VL_2B_PATH"
    ;;
  qwen2_5vl_7b)
    MODEL_NAME="$QWEN2_5VL_7B_PATH"
    FOUNDATION_MODEL="$QWEN2_5VL_7B_PATH"
    ;;
  qwen3vl_8b)
    MODEL_NAME="$QWEN3VL_8B_PATH"
    FOUNDATION_MODEL="$QWEN3VL_8B_PATH"
    ;;
  *)
    echo "Error: Unknown model type '$MODEL_TYPE'. Use 'qwen2vl', 'qwen2vl_2b', 'qwen2_5vl_7b', or 'qwen3vl_8b'.";
    exit 1
    ;;
esac

# ---------------------------- I/O Setup -----------------------------
export WANDB_NAME="$EXP_NAME"
export WANDB_DIR="$EXP_DIR"
echo "==> Experiment     : $EXP_NAME"
echo "==> Output dir     : $EXP_DIR"
echo "==> Config file    : $CONFIG_FILE"
mkdir -p "$EXP_DIR/wandb" "$EXP_DIR/logs"
if [[ -z "$EXISTING_EXP_DIR" ]]; then
  rm -rf "$EXP_DIR/wandb/"*
fi

# --------------------- GPU / Dataloader Settings --------------------
per_device_batch_size=48
gradient_accumulation=1
dataloader_workers=4
cuda_devices="0"

if (( NUM_GPUS > 1 )); then
  echo "==> Configuring for $NUM_GPUS GPUs"
  if   (( NUM_GPUS == 2 )); then
    per_device_batch_size=64
    gradient_accumulation=4
    dataloader_workers=4
    cuda_devices="0,1"
  elif (( NUM_GPUS == 4 )); then
    per_device_batch_size=32
    gradient_accumulation=4
    dataloader_workers=6
    cuda_devices="0,1,2,3"
  elif (( NUM_GPUS == 8 )); then
    per_device_batch_size=64
    gradient_accumulation=1
    dataloader_workers=8
    cuda_devices="0,1,2,3,4,5,6,7"
  else
    echo "Warning: Unsupported GPU count '$NUM_GPUS'. Falling back to 2-GPU defaults."
    per_device_batch_size=64
    gradient_accumulation=4
    dataloader_workers=4
    cuda_devices="0,1"
    NUM_GPUS=2
  fi
fi

# -------------------------- Build Command ---------------------------
LOG_FILE="$EXP_DIR/training_output.log"

# Build minimal prompt flag
MINIMAL_PROMPT_FLAG=""
if [[ "$USE_MINIMAL_PROMPT" == "true" ]]; then
  MINIMAL_PROMPT_FLAG="--use_minimal_prompt"
  echo "==> Minimal prompt enabled: using structured rewrite prompt"
fi

if (( NUM_GPUS > 1 )); then
  # Multi-GPU (DDP)
  cmd="CUDA_VISIBLE_DEVICES=$cuda_devices torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_iterative.py \
    --model_name \"$MODEL_NAME\" \
    --foundation_model_name \"$FOUNDATION_MODEL\" \
    --lora \
    --lora_r $LORA_R \
    --lora_dropout $LORA_DROPOUT \
    --pooling eos \
    --normalize True \
    --temperature 0.03 \
    --dataset_config \"$CONFIG_FILE\" \
    --run_name \"$EXP_NAME\" \
    --project_name \"$WANDB_PROJECT\" \
    --output_dir \"$EXP_DIR\" \
    --per_device_train_batch_size $per_device_batch_size \
    --gradient_accumulation_steps $gradient_accumulation \
    --dataloader_num_workers $dataloader_workers \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --lr_scheduler_type cosine \
    --learning_rate 4e-5 \
    --warmup_ratio 0.1 \
    --save_steps 3000 \
    --logging_steps 50 \
    --logging_dir \"$EXP_DIR/logs\" \
    --eval_steps 1000 \
    --save_safetensors True \
    --remove_unused_columns False \
    --resume_from none \
    --resume_from_iteration auto \
    --max_len 512 \
    --resize_use_processor True \
    --resize_min_pixels 3136 \
    --resize_max_pixels 147456 \
    --fp16 False \
    --bf16 True \
    --triplet_loss_weight 0 \
    --triplet_margin 0.2 \
    --info_nce_weight 1 \
    --gradient_checkpointing True \
    --optim adamw_torch \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --report_to none \
    --group_by_category True \
    --group_by_reference_image False \
    $MINIMAL_PROMPT_FLAG \
    2>&1 | tee \"$LOG_FILE\""
else
  # Single-GPU
  cmd="CUDA_VISIBLE_DEVICES=${cuda_devices} python \
    train_iterative.py \
    --model_name \"$MODEL_NAME\" \
    --foundation_model_name \"$FOUNDATION_MODEL\" \
    --lora \
    --lora_r $LORA_R \
    --lora_dropout $LORA_DROPOUT \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --dataset_config \"$CONFIG_FILE\" \
    --run_name \"$EXP_NAME\" \
    --project_name \"$WANDB_PROJECT\" \
    --output_dir \"$EXP_DIR\" \
    --per_device_train_batch_size $per_device_batch_size \
    --gradient_accumulation_steps $gradient_accumulation \
    --dataloader_num_workers $dataloader_workers \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --lr_scheduler_type cosine \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --save_steps 500 \
    --logging_steps 50 \
    --logging_dir \"$EXP_DIR/logs\" \
    --eval_steps 1000 \
    --save_safetensors True \
    --remove_unused_columns False \
    --resume_from none \
    --resume_from_iteration auto \
    --max_len 512 \
    --resize_use_processor True \
    --resize_min_pixels 3136 \
    --resize_max_pixels 35840 \
    --fp16 False \
    --bf16 True \
    --gradient_checkpointing False \
    --optim adamw_torch \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --report_to none \
    --group_by_category True \
    --group_by_reference_image False \
    $MINIMAL_PROMPT_FLAG \
    2>&1 | tee \"$LOG_FILE\""
fi

# --------------------------- Run Training ---------------------------
echo "==> Running command:"
echo "$cmd"
echo ""
eval "$cmd"

echo ""
echo "==> Training completed!"
echo "==> Results saved to: $EXP_DIR"
echo "==> Log file       : $LOG_FILE"

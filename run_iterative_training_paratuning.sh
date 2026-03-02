#!/usr/bin/env bash
# Iterative Training Script for Composed Image Retrieval - PRODUCTION VERSION (LoRA r=16, dropout=0.1)
# Usage:
#   ./run_iterative_training_r16_d01.sh [cirr|fashioniq] [qwen2vl|qwen2vl_2b|qwen2_5vl_7b|llava_next] [num_gpus] [existing_exp_dir]
#
# Examples:
#   ./run_iterative_training_r16_d01.sh cirr qwen2vl 2
#   ./run_iterative_training_r16_d01.sh cirr qwen2vl 8
#   ./run_iterative_training_r16_d01.sh cirr qwen2vl 2 ./experiments/IterativeCIRR_qwen2vl_20250805_000011
#   bash ./run_iterative_training_paratuning.sh cirr qwen2_5vl_7b 8 ./experiments/IterativeCIRR_qwen2_5vl_7b_20251012_004205_copy_gruopsamplerfix_copy_triplet_loss_i0_t1
#   bash ./run_iterative_training_paratuning.sh cirr qwen2_5vl_7b 8 ./experiments/IterativeCIRR_qwen2_5vl_7b_20251012_004205_continue_test_new_eval
# - This script uses LoRA with r=16, dropout=0.1 and (default) alpha=32 (override via $LORA_ALPHA).
# - Iterative training with grouped-by-reference-image sampler remains enabled.

set -euo pipefail

# -------------------------- Configuration --------------------------
DATASET=${1:-"cirr"}             # cirr | fashioniq
MODEL_TYPE=${2:-"qwen2vl"}       # qwen2vl | qwen2vl_2b | llava_next
NUM_GPUS=${3:-2}                 # default 2 GPUs
EXISTING_EXP_DIR=${4:-""}        # optional: resume from existing experiment dir
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-""}  # optional: pass-through flags to train_iterative.py

# LoRA settings for this script
LORA_R=64
LORA_DROPOUT=0.1

# Local model paths
QWEN2VL_2B_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-2B-Instruct"
QWEN2VL_7B_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct"
QWEN2_5VL_7B_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2.5-VL-7B-Instruct"  # NEW
QWEN2_5VL_32B_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2.5-VL-32B-Instruct"

echo "==> Starting iterative training"
echo "    DATASET      : $DATASET"
echo "    MODEL_TYPE   : $MODEL_TYPE"
echo "    NUM_GPUS     : $NUM_GPUS"
echo "    RESUME_DIR   : ${EXISTING_EXP_DIR:-<new experiment>}"
echo "    LoRA         : r=$LORA_R, dropout=$LORA_DROPOUT"
echo "==> Environment"
echo "conda  : $(which conda || true)"
echo "python : $(which python)"
python --version || true

# Hugging Face / WandB env
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export WANDB_DISABLED=false
export WANDB_PROJECT="iterative_composed_retrieval_production"
# export WANDB_API_KEY="your_wandb_api_key"
# export HUGGING_FACE_HUB_TOKEN="your_hf_token"

# ----------------------- Experiment Resolution ----------------------
if [[ -n "$EXISTING_EXP_DIR" ]]; then
  if [[ ! -d "$EXISTING_EXP_DIR" ]]; then
    echo "Error: Existing experiment directory '$EXISTING_EXP_DIR' does not exist."
    exit 1
  fi
  EXP_NAME="$(basename "$EXISTING_EXP_DIR")"
  export EXP_DIR="$EXISTING_EXP_DIR"
  echo "==> Resuming existing experiment: $EXP_NAME"
  echo "==> Using directory           : $EXP_DIR"

  if [[ "$EXP_NAME" == *"CIRR"* ]] || [[ -f "$EXP_DIR/cirr_iterative.yaml" ]]; then
    CONFIG_FILE="configs/cirr_iterative.yaml"
    echo "==> Detected CIRR dataset from experiment name/content"
  elif [[ "$EXP_NAME" == *"FashionIQ"* ]] || [[ -f "$EXP_DIR/fashioniq_iterative.yaml" ]]; then
    CONFIG_FILE="configs/fashioniq_iterative.yaml"
    echo "==> Detected FashionIQ dataset from experiment name/content"
  else
    echo "==> Falling back to provided DATASET: $DATASET"
    if [[ "$DATASET" == "cirr" ]]; then
      CONFIG_FILE="configs/cirr_iterative.yaml"
    elif [[ "$DATASET" == "fashioniq" ]]; then
      CONFIG_FILE="configs/fashioniq_iterative.yaml"
    else
      echo "Error: Unknown dataset '$DATASET'. Use 'cirr' or 'fashioniq'."
      exit 1
    fi
  fi
else
  if [[ "$DATASET" == "cirr" ]]; then
    CONFIG_FILE="configs/cirr_iterative.yaml"
    EXP_NAME="IterativeCIRR_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
  elif [[ "$DATASET" == "fashioniq" ]]; then
    CONFIG_FILE="configs/fashioniq_iterative.yaml"
    EXP_NAME="IterativeFashionIQ_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
  else
    echo "Error: Unknown dataset '$DATASET'. Use 'cirr' or 'fashioniq'."
    exit 1
  fi
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
    MODEL_NAME="$QWEN2_5VL_7B_PATH"    # NEW case
    FOUNDATION_MODEL="$QWEN2_5VL_7B_PATH"  # keep caption model一致
    ;;
  llava_next)
    MODEL_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
    FOUNDATION_MODEL="llava-hf/llava-v1.6-mistral-7b-hf"
    ;;
  *)
    echo "Error: Unknown model type '$MODEL_TYPE'. Use 'qwen2vl', 'qwen2vl_2b', 'qwen2_5vl_7b', or 'llava_next'.";
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
# Default per-device batch / workers; overridden by GPU count below
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
    per_device_batch_size=8
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
    --temperature 0.02 \
    --info_nce_weight 1 \
    --triplet_loss_weight 0 \
    --triplet_margin 0.05 \
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
    --gradient_checkpointing False \
    --optim adamw_torch \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --group_by_reference_image \
    --report_to none \
    $EXTRA_TRAIN_ARGS \
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
    --info_nce_weight 0.8 \
    --triplet_loss_weight 0.2 \
    --triplet_margin 0.05 \
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
    --group_by_reference_image \
    $EXTRA_TRAIN_ARGS \
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

#!/bin/bash

# Usage: ./eval_cirr.sh --model_path <path_to_model> [--output_file <path_to_output>]

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required Arguments:"
    echo "  --model_path PATH           Path to trained model checkpoint"
    echo ""
    echo "Optional Arguments:"
    echo "  --model_name NAME           Base model name (e.g., Qwen/Qwen2-VL-2B-Instruct)"
    echo "  --eval_config PATH          Path to evaluation config YAML file"
    echo "  --batch_size SIZE           Batch size for evaluation (default: 16)"
    echo "  --device DEVICE             Device to use (default: auto)"
    echo "  --distributed               Enable distributed evaluation (default: auto-detect)"
    echo "  --single-gpu                Force single GPU mode (disable distributed)"
    echo "  --output_file PATH          Output file for results (JSON format)"
    echo "  --cirr_data_dir PATH        Override CIRR dataset directory"
    echo "  --cirr_image_dir PATH       Override CIRR image directory"
    echo "  --verbose                   Print detailed results (default: true)"
    echo "  --help, -h                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic evaluation with checkpoint"
    echo "  $0 --model_path ./outputs/checkpoint-1000"
    echo ""
    echo "  # Evaluation with custom model name and batch size"
    echo "  $0 --model_path ./outputs/iteration_2 --model_name Qwen/Qwen2-VL-2B-Instruct --batch_size 32"
    echo ""
    echo "  # Evaluation with custom CIRR dataset path"
    echo "  $0 --model_path ./outputs/checkpoint-1000 --cirr_data_dir /path/to/cirr --cirr_image_dir /path/to/images"
    echo ""
    echo "  # Distributed evaluation"
    echo "  $0 --model_path ./outputs/checkpoint-1000 --distributed"
    echo ""
    echo "  # Save results to file"
    echo "  $0 --model_path ./outputs/checkpoint-1000 --output_file ./eval_results.json"
}

MODEL_PATH=""
MODEL_NAME=""
EVAL_CONFIG=""
BATCH_SIZE=8
DEVICE="auto"
DISTRIBUTED="auto"  # Changed to auto-detect
OUTPUT_FILE=""
CIRR_DATA_DIR=""
CIRR_IMAGE_DIR=""
VERBOSE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --eval_config)
            EVAL_CONFIG="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --distributed)
            DISTRIBUTED=true
            shift
            ;;
        --single-gpu)
            DISTRIBUTED=false
            shift
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --cirr_data_dir)
            CIRR_DATA_DIR="$2"
            shift 2
            ;;
        --cirr_image_dir)
            CIRR_IMAGE_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    print_error "model_path is required"
    show_usage
    exit 1
fi

if [[ ! -e "$MODEL_PATH" ]]; then
    print_error "Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [[ "$DISTRIBUTED" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
        if [[ $GPU_COUNT -gt 1 ]]; then
            DISTRIBUTED=true
            print_info "Auto-detected $GPU_COUNT GPUs, enabling distributed evaluation"
        else
            DISTRIBUTED=false
            print_info "Auto-detected $GPU_COUNT GPU, using single-GPU evaluation"
        fi
    else
        DISTRIBUTED=false
        print_info "NVIDIA-SMI not found, using single-GPU evaluation"
    fi
fi

print_info "CIRR Evaluation Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Model Name: ${MODEL_NAME:-auto-detect}"
echo "  Eval Config: ${EVAL_CONFIG:-default}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Distributed: $DISTRIBUTED"
echo "  Output File: ${OUTPUT_FILE:-none}"
echo "  CIRR Data Dir: ${CIRR_DATA_DIR:-default}"
echo "  CIRR Image Dir: ${CIRR_IMAGE_DIR:-default}"
echo "  Verbose: $VERBOSE"
echo ""

if [[ "$DISTRIBUTED" == true ]]; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
    CMD="torchrun --nproc_per_node=$GPU_COUNT --master_port=29500 eval_cirr.py"
    print_info "Using torchrun with $GPU_COUNT GPUs"
else
    CMD="python eval_cirr.py"
    print_info "Using single GPU/CPU mode"
fi

CMD="$CMD --model_path \"$MODEL_PATH\""
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --device $DEVICE"

if [[ -n "$MODEL_NAME" ]]; then
    CMD="$CMD --base_model_name \"$MODEL_NAME\""
    CMD="$CMD --model_name \"$MODEL_NAME\""
else
    CMD="$CMD --model_name \"auto-infer\""
fi

if [[ -n "$EVAL_CONFIG" ]]; then
    CMD="$CMD --eval_config \"$EVAL_CONFIG\""
fi

if [[ "$DISTRIBUTED" == true ]]; then
    CMD="$CMD --distributed"
fi

if [[ -n "$OUTPUT_FILE" ]]; then
    CMD="$CMD --output_file \"$OUTPUT_FILE\""
fi

if [[ -n "$CIRR_DATA_DIR" ]]; then
    CMD="$CMD --cirr_data_dir \"$CIRR_DATA_DIR\""
fi

if [[ -n "$CIRR_IMAGE_DIR" ]]; then
    CMD="$CMD --cirr_image_dir \"$CIRR_IMAGE_DIR\""
fi

if [[ "$VERBOSE" == true ]]; then
    CMD="$CMD --verbose"
fi

export TOKENIZERS_PARALLELISM=false

if [[ ! -f "eval_cirr.py" ]]; then
    print_error "eval_cirr.py not found in current directory"
    print_info "Please run this script from the project root directory"
    exit 1
fi

print_info "Starting CIRR evaluation..."
print_info "Command: $CMD"
echo ""

eval $CMD

if [[ $? -eq 0 ]]; then
    print_success "CIRR evaluation completed successfully!"
    if [[ -n "$OUTPUT_FILE" ]] && [[ -f "$OUTPUT_FILE" ]]; then
        print_success "Results saved to: $OUTPUT_FILE"
    fi
else
    print_error "CIRR evaluation failed!"
    exit 1
fi

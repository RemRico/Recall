# Iterative Composed Image Retrieval

This project implements iterative training for composed image retrieval using VLM2Vec with hard negative mining and foundation model augmentation.

## Features

- 🔄 **Iterative Training**: Progressive hard negative mining across multiple training rounds
- 🎯 **Real Retrieval**: Actual VLM2Vec model inference instead of simulation
- 🤖 **Foundation Model Integration**: Qwen2VL for caption generation and data augmentation
- 💾 **Smart Caching**: Avoid repeated computations with checkpoint resumption
- 📊 **Progress Tracking**: Real-time progress display with ETA estimation
- 🔧 **Flexible Configuration**: Support for different model backbones and datasets

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Basic Training

```bash
# Run iterative training with default settings
./run_iterative_training.sh

# Or run directly with Python
python train_iterative.py \
    --model_backbone qwen2_vl \
    --dataset_name cirr \
    --num_iterations 3 \
    --foundation_model_path /path/to/qwen2-vl-model
```

### 3. Fast Mode (for testing)

```bash
# Quick test with subset of data
./run_iterative_training.sh --fast_mode
```

## Configuration

### Training Modes

#### Fast Mode (for testing and debugging)
```yaml
# In your YAML config file
fast_mode: true
fast_mode_max_samples: 100        # Limit samples per iteration
fast_mode_retrieval_db_size: 50   # Limit retrieval database size  
fast_mode_max_steps: 5            # Limit training steps per iteration
```

#### Production Mode (for full training)
```yaml
# In your YAML config file  
fast_mode: false
production_max_steps: 1000        # Full training steps per iteration
production_save_steps: 100        # Save frequency
```

### Key Parameters

- `fast_mode`: Enable fast mode for quick testing (default: false)
- `max_iterations`: Number of iterative rounds (default: 3)
- `hard_neg_collection_freq`: Frequency of hard negative collection (default: 1)
- `hard_neg_top_k`: Retrieval top-k retained for mining (default: 10)
- `hard_neg_post_gt`: Additional negatives sampled after GT rank (default: 0)
- `hard_neg_per_query`: Maximum hard negatives stored per query (default: 5)
- `caption_generation_batch_size`: Batch size for caption generation (default: 8)
- `foundation_model_name`: Foundation model for caption generation

### Example Commands

```bash
# Full training with Qwen2VL foundation model
python train_iterative.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --output_dir ./experiments/iterative_cirr \
    --dataset_config configs/cirr_iterative.yaml \
    --foundation_model_name Qwen/Qwen2-VL-7B-Instruct \
    --max_iterations 5

# Auto resume from latest checkpoint
python train_iterative.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --output_dir ./experiments/iterative_cirr \
    --resume_from auto \
    --dataset_config configs/cirr_iterative.yaml

# Resume from specific iteration
python train_iterative.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --output_dir ./experiments/iterative_cirr \
    --resume_from iter_2 \
    --dataset_config configs/cirr_iterative.yaml

# Fast mode for debugging
python train_iterative.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --output_dir ./experiments/test_fast \
    --dataset_config configs/cirr_iterative.yaml \
    --fast_mode
```

## Resume Training

### Resume Options

1. **Auto Resume** (`--resume_from auto`): Automatically detects and resumes from the latest iteration checkpoint
2. **Manual Resume** (`--resume_from iter_X`): Resumes from a specific iteration (e.g., `iter_2`)
3. **Standard Checkpoint** (`--resume_from checkpoint-1000`): Resumes from a HuggingFace checkpoint
4. **Fresh Start**: No resume parameter, starts training from scratch

### File Structure

```
output_dir/
├── base_model/                      # Iteration 0 base model
├── iteration_1/                     # Iteration 1 model weights
├── iteration_2/                     # Iteration 2 model weights
├── iteration_3/                     # Iteration 3 model weights
├── iteration_0_state.json           # Iteration 0 training state
├── iteration_1_state.json           # Iteration 1 training state
├── iteration_2_state.json           # Iteration 2 training state
├── hard_negatives_iter_0.json       # Hard negatives for iteration 0
├── hard_negatives_iter_1.json       # Hard negatives for iteration 1
├── hard_negatives_iter_2.json       # Hard negatives for iteration 2
├── augmented_samples_iter_1.json    # Augmented samples for iteration 1
├── augmented_samples_iter_2.json    # Augmented samples for iteration 2
├── augmented_samples_iter_3.json    # Augmented samples for iteration 3
└── training_summary.json            # Training summary
```

### Resume Mechanism

- **Model Loading**: Uses `MMEBModel.load()` for unified model weight management (supports both LoRA and full models)
- **State Recovery**: Loads training state from `iteration_X_state.json`
- **Data Recovery**: Restores dataset state from augmented samples and hard negatives
- **Error Handling**: Automatically falls back to base model if checkpoint loading fails

## Evaluation

### Evaluate All Iterations

```bash
python eval_iterative.py \
    --experiment_dir ./experiments/qwen2_vl_cirr_iterative \
    --eval_dataset cirr_test
```

### Evaluate Specific Iterations

```bash
python eval_iterative.py \
    --experiment_dir ./experiments/qwen2_vl_cirr_iterative \
    --iterations "1,3,5" \
    --output_file results.json
```

## Project Structure

```
├── src/
│   ├── data/dataset/
│   │   └── composed_retrieval_dataset.py  # Iterative CIRR dataset with real retrieval
│   ├── trainer_iterative.py              # Iterative trainer implementation  
│   ├── arguments.py                       # Training arguments definition
│   └── ...
├── configs/
│   └── cirr_iterative.yaml               # Iterative training configuration
├── train_iterative.py                    # Main training script
├── run_iterative_training.sh             # Training shell script
└── experiments/                          # Experiment outputs
    └── {experiment_name}/
        ├── base_model/                    # Base model checkpoint (iteration 0)
        ├── iteration_1/                   # Iteration 1 model checkpoint
        ├── iteration_2/                   # Iteration 2 model checkpoint
        ├── hard_negatives_iter_0.json     # Hard negatives per iteration
        ├── hard_negatives_iter_1.json     
        ├── augmented_samples_iter_1.json  # Generated augmented samples
        ├── augmented_samples_iter_2.json
        ├── iteration_0_state.json         # Training state per iteration
        ├── iteration_1_state.json
        └── training_summary.json          # Training summary and results
```

## How It Works

### 1. Iterative Training Loop

```
For each iteration:
1. Mine hard negatives using current model
2. Generate augmented captions with foundation model
3. Train on original + augmented data
4. Save checkpoint
```

### 2. Hard Negative Mining

- Uses real VLM2Vec retrieval on training data
- Identifies samples where ground truth is not top-1
- Collects top-ranked incorrect results as hard negatives

### 3. Caption Augmentation

- Foundation model (Qwen2VL) generates new modification texts
- Input: reference image + target image + original caption
- Output: similar but different modification text
- Creates positive samples from previous hard negatives

### 4. Smart Caching

- Caches hard negatives to avoid re-mining
- Saves augmented samples for checkpoint resumption
- Experiment directory tracks all intermediate results

## Advanced Features

### Real vs Simulated Retrieval

- **Real Retrieval**: Actual VLM2Vec model inference (default)
- **Simulated Retrieval**: Fast dummy results for testing
- Automatic fallback if real retrieval fails

### Progress Tracking

- Batch-level progress with ETA estimation
- Generation rate monitoring
- Comprehensive logging and statistics

### Foundation Model Support

- Qwen2VL: Multi-image conversation format
- LLaVA: Horizontal image concatenation
- Generic: Fallback for other models

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use fast mode
2. **Foundation Model Loading**: Check model path and permissions
3. **Checkpoint Resumption**: Verify experiment directory structure

### Debug Mode

```bash
# Enable detailed logging
PYTHONPATH=. python train_iterative.py --fast_mode --num_iterations 1
```

## Citation

If you use this work, please cite:

```bibtex
@article{iterative_cir_2025,
  title={Iterative Training for Composed Image Retrieval with Hard Negative Mining},
  author={Your Name},
  year={2025}
}
```

#!/usr/bin/env python3
"""
Iterative Training Script for Composed Image Retrieval
Adapted from VLM2Vec training pipeline for iterative hard negative mining
"""

import logging
import os
import os.path
import sys

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

import torch
import wandb
import yaml
import json
from transformers import HfArgumentParser
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.model import MMEBModel
from src.trainer_iterative_ import IterativeRetrievalTrainer, create_iterative_trainer
from src.utils import print_rank, print_master, find_latest_checkpoint
from src.model.processor import load_processor, get_backbone_name


def main():
    # Handle distributed training arguments
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    
    # Disable tokenizer parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logger.info(
        "Distributed env: RANK=%s LOCAL_RANK=%s WORLD_SIZE=%s MASTER_ADDR=%s MASTER_PORT=%s",
        os.environ.get("RANK"),
        os.environ.get("LOCAL_RANK"),
        os.environ.get("WORLD_SIZE"),
        os.environ.get("MASTER_ADDR"),
        os.environ.get("MASTER_PORT"),
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        logger.info(
            "torch.distributed initialized: rank=%s world_size=%s",
            torch.distributed.get_rank(),
            torch.distributed.get_world_size(),
        )


    # Set up logging_dir for train.log generation if not specified
    if not training_args.logging_dir:
        training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
        print_master(f"Setting logging_dir to: {training_args.logging_dir}")
    
    # Ensure logging directory exists
    os.makedirs(training_args.logging_dir, exist_ok=True)

    # Initialize WandB if enabled
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('Initializing wandb for iterative training')
            wandb.init(
                project=training_args.project_name or "iterative_composed_retrieval", 
                name=training_args.run_name or "iterative_training", 
                mode="online"
            )
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            wandb.config.update(training_args)

    # Load retrieval model with checkpoint resume support
    print_master("Loading retrieval model...")
    
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not getattr(model_args, "model_backbone", None):
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=getattr(model_args, 'model_type', None))
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_master(f'Model backbone: {model_args.model_backbone}')
    
    print_master("=" * 60)
    print_master("CHECKPOINT RECOVERY SYSTEM")
    print_master("Each iteration saves checkpoints in separate subdirectories")
    print_master("   - training_iter_0/: Base model training checkpoints")  
    print_master("   - training_iter_1/: Iteration 1 training checkpoints")
    print_master("   - training_iter_2/: Iteration 2 training checkpoints")
    print_master("   - base_model/, iteration_1/, iteration_2/: Final models")
    print_master("=" * 60)
    
    # 1. Trainer checkpoint recovery (includes optimizer/scheduler state)
    trainer_checkpoint = None
    if training_args.resume_from == 'auto':
        trainer_checkpoint = find_latest_checkpoint(training_args.output_dir)
        
        if not trainer_checkpoint:
            print_master("Searching for checkpoints in iteration subdirectories...")
            latest_iteration_dir = None
            latest_step = -1
            
            for i in range(10, -1, -1):
                iteration_dir = os.path.join(training_args.output_dir, f"training_iter_{i}")
                if os.path.exists(iteration_dir):
                    iter_checkpoint = find_latest_checkpoint(iteration_dir)
                    if iter_checkpoint:
                        # Extract step number
                        checkpoint_name = os.path.basename(iter_checkpoint)
                        if checkpoint_name.startswith('checkpoint-'):
                            try:
                                step_num = int(checkpoint_name.split('-')[1])
                                # Later iterations have higher priority
                                effective_step = i * 10000 + step_num
                                if effective_step > latest_step:
                                    latest_step = effective_step
                                    latest_iteration_dir = iteration_dir
                                    trainer_checkpoint = iter_checkpoint
                                    print_master(f"Found checkpoint in iteration {i}: {iter_checkpoint}")
                            except ValueError:
                                continue
            
            if trainer_checkpoint:
                print_master(f"Found trainer checkpoint in subdirectory: {trainer_checkpoint}")
                print_master(f"  Contains: model weights + optimizer + scheduler states")
            else:
                print_master("No trainer checkpoint found in main directory or subdirectories")
        else:
            print_master(f"Found trainer checkpoint in main directory: {trainer_checkpoint}")
            print_master(f"  Contains: model weights + optimizer + scheduler states")
            
    elif training_args.resume_from.isdigit():
        checkpoint_step = training_args.resume_from
        checkpoint_found = False
        
        for i in range(10, -1, -1):
            iteration_dir = os.path.join(training_args.output_dir, f"training_iter_{i}")
            potential_checkpoint = os.path.join(iteration_dir, f'checkpoint-{checkpoint_step}')
            if os.path.exists(potential_checkpoint):
                trainer_checkpoint = potential_checkpoint
                checkpoint_found = True
                print_master(f"Found specified checkpoint in iteration {i}: {trainer_checkpoint}")
                print_master(f"  Contains: model weights + optimizer + scheduler states")
                break
        
        if not checkpoint_found:
            main_checkpoint = os.path.join(training_args.output_dir, f'checkpoint-{checkpoint_step}')
            if os.path.exists(main_checkpoint):
                trainer_checkpoint = main_checkpoint
                print_master(f"Found specified checkpoint in main directory: {trainer_checkpoint}")
                print_master(f"  Contains: model weights + optimizer + scheduler states")
            else:
                print_master(f"Specified trainer checkpoint not found: checkpoint-{checkpoint_step}")
                
    elif training_args.resume_from != 'none':
        print_master(f"Warning: Unknown resume_from format: {training_args.resume_from}")
    
    # 2. Iteration model recovery (model weights only)
    iteration_model = None
    resume_from_iteration = None
    
    def check_iteration_complete(output_dir, iteration, max_iterations):
        """Check if an iteration is completely finished"""
        state_file = os.path.join(output_dir, f"iteration_{iteration}_state.json")
        if not os.path.exists(state_file):
            return False
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            return state.get('iteration_complete', False)
        except:
            return False
    
    if training_args.resume_from_iteration == 'auto':
        for i in range(10, -1, -1):
            if i == 0:
                model_path = os.path.join(training_args.output_dir, "base_model")
            else:
                model_path = os.path.join(training_args.output_dir, f"iteration_{i}")
            
            if os.path.exists(model_path) and check_iteration_complete(training_args.output_dir, i, 10):
                resume_from_iteration = i
                iteration_model = model_path
                print_master(f"Found COMPLETE iteration {i} model: {model_path}")
                print_master(f"  Contains: model weights only (no optimizer/scheduler)")
                break
                
        if iteration_model is None:
            for i in range(10, -1, -1):
                if i == 0:
                    model_path = os.path.join(training_args.output_dir, "base_model")
                else:
                    model_path = os.path.join(training_args.output_dir, f"iteration_{i}")
                
                if os.path.exists(model_path):
                    resume_from_iteration = i
                    iteration_model = model_path
                    print_master(f"Found INCOMPLETE iteration {i} model: {model_path}")
                    print_master(f"  Contains: model weights only (no optimizer/scheduler)")
                    break
                    
    elif training_args.resume_from_iteration.startswith('iter_'):
        iter_num_str = training_args.resume_from_iteration.split('_')[1]
        if iter_num_str.isdigit():
            iter_num = int(iter_num_str)
            if iter_num == 0:
                model_path = os.path.join(training_args.output_dir, "base_model")
            else:
                model_path = os.path.join(training_args.output_dir, f"iteration_{iter_num}")
            
            if os.path.exists(model_path):
                resume_from_iteration = iter_num
                iteration_model = model_path
                complete_status = "COMPLETE" if check_iteration_complete(training_args.output_dir, iter_num, 10) else "INCOMPLETE"
                print_master(f"Using specified {complete_status} iteration {iter_num} model: {model_path}")
                print_master(f"  Contains: model weights only (no optimizer/scheduler)")
            else:
                print_master(f"Specified iteration model not found: {model_path}")
                
    elif training_args.resume_from_iteration != 'none':
        print_master(f"Warning: Unknown resume_from_iteration format: {training_args.resume_from_iteration}")
    
    # 3. Decide recovery strategy
    print_master("-" * 60)
    print_master("RECOVERY STRATEGY:")
    
    if trainer_checkpoint and iteration_model:
        print_master("BOTH checkpoints found - using ITERATION model for weights")
        print_master("  Reason: Iteration models contain the latest trained weights")
        print_master(f"  Model weights from: {iteration_model}")
        print_master(f"  Training state from: {trainer_checkpoint}")
        use_iteration_for_weights = True
        use_trainer_for_state = True
    elif trainer_checkpoint:
        print_master("Using TRAINER checkpoint (complete recovery)")
        print_master(f"  Everything from: {trainer_checkpoint}")
        use_iteration_for_weights = False
        use_trainer_for_state = True
    elif iteration_model:
        print_master("Using ITERATION model (weights only)")
        print_master(f"  Model weights from: {iteration_model}")
        print_master("  No training state - will start fresh optimizer/scheduler")
        use_iteration_for_weights = True
        use_trainer_for_state = False
    else:
        print_master("No checkpoints found - starting from scratch")
        use_iteration_for_weights = False
        use_trainer_for_state = False
    
    print_master("=" * 60)
    
    # 4. Load model
    model = None
    
    if use_iteration_for_weights:
        print_master(f"Loading model weights from iteration checkpoint: {iteration_model}")
        try:
            config_path = os.path.join(iteration_model, "config.json")
            if not os.path.exists(config_path):
                print_master("Creating missing config.json for iteration model...")
                import shutil
                original_config = os.path.join(model_args.model_name, "config.json")
                if os.path.exists(original_config):
                    shutil.copy2(original_config, config_path)
                    print_master("Config.json created successfully")
                else:
                    print_master("Warning: Original config.json not found, will use fallback method")
            
            model_args.checkpoint_path = iteration_model
            model = MMEBModel.load(model_args, is_trainable=True)
            print_master(f"Successfully loaded model from iteration {resume_from_iteration}")
        except Exception as e:
            print_master(f"Failed to load iteration checkpoint: {e}")
            print_master("Falling back to trainer checkpoint or new model")
            if hasattr(model_args, 'checkpoint_path'):
                delattr(model_args, 'checkpoint_path')
            model = None
            use_iteration_for_weights = False
    
    if model is None and use_trainer_for_state:
        print_master(f"Loading model from trainer checkpoint: {trainer_checkpoint}")
        try:
            model_args.checkpoint_path = trainer_checkpoint
            model = MMEBModel.load(model_args, is_trainable=True)
            print_master("Successfully loaded model from trainer checkpoint")
        except Exception as e:
            print_master(f"Failed to load trainer checkpoint: {e}")
            print_master("Will build new model")
            if hasattr(model_args, 'checkpoint_path'):
                delattr(model_args, 'checkpoint_path')
            model = None
    
    if model is None:
        print_master("Building new model from scratch...")
        model = MMEBModel.build(model_args)
        print_master("New model built successfully")
    
    # Load processor
    processor = load_processor(model_args, data_args)
    
    setattr(model, 'processor', processor)

    foundation_model = None
    foundation_model_name = getattr(model_args, "foundation_model_name", None)

    # Load dataset configuration
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)
        
        # Check if this is an iterative training config
        is_iterative = any('iterative' in str(config).lower() for config in dataset_config.values())
        
        if is_iterative:
            print_master("Detected iterative training configuration")
            # For iterative training, we'll handle dataset loading in the trainer
            train_dataset = None
        else:
            # Standard dataset loading
            train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args)

    # Create data collator
    train_collator = MultimodalDataCollator(processor, model_args, data_args, training_args)

    # Create trainer
    if is_iterative:
        print_master("Creating iterative trainer...")
        
        # Extract iterative training parameters
        iterative_params = {}
        for config_name, config in dataset_config.items():
            if isinstance(config, dict):
                # Basic iterative parameters
                iterative_params.update({
                    'max_iterations': config.get('max_iterations', 3),
                    'hard_neg_collection_freq': config.get('hard_neg_collection_freq', 1),
                    'caption_generation_batch_size': config.get('caption_generation_batch_size', 8)
                })

                if 'info_nce_weight' in config:
                    training_args.info_nce_weight = config['info_nce_weight']
                if 'triplet_loss_weight' in config:
                    training_args.triplet_loss_weight = config['triplet_loss_weight']
                if 'triplet_margin' in config:
                    training_args.triplet_margin = config['triplet_margin']
                
                # Fast mode and production mode parameters
                fast_mode = config.get('fast_mode', False)
                iterative_params['fast_mode'] = fast_mode
                
                if fast_mode:
                    # Use fast mode settings
                    iterative_params.update({
                        'fast_mode_max_samples': config.get('fast_mode_max_samples', 100),
                        'fast_mode_retrieval_db_size': config.get('fast_mode_retrieval_db_size', 50),
                        'fast_mode_max_steps': config.get('fast_mode_max_steps', 5)
                    })
                    print_master(f"Fast mode enabled: {config.get('fast_mode_max_steps', 5)} steps per iteration")
                else:
                    # Use production mode settings
                        steps_per_iter = config.get('steps_per_iteration', config.get('production_max_steps', 1000))
                    iterative_params.update({
                        'steps_per_iteration': steps_per_iter,
                        'production_save_steps': config.get('production_save_steps', 100)
                    })
                    print_master(f"Production mode enabled: {steps_per_iter} steps per iteration")
                
                break
        
        # Create initial dataset for iteration 0
        train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args)
        
        if hasattr(model, "configure_loss"):
            model.configure_loss(
                info_nce_weight=getattr(training_args, "info_nce_weight", 1.0),
                triplet_loss_weight=getattr(training_args, "triplet_loss_weight", 0.0),
                triplet_margin=getattr(training_args, "triplet_margin", 0.2),
            )
        
        # Debug: Print iterative_params to verify fast_mode is included
        print_master(f"DEBUG: iterative_params = {iterative_params}")
        
        trainer = create_iterative_trainer(
            model=model,
            foundation_model=None,
            foundation_model_name=foundation_model_name,
            processing_class=processor,
            args=training_args,
            model_args=model_args,
            data_args=data_args,
            train_dataset=train_dataset,
            data_collator=train_collator,
            max_length=data_args.max_len,
            **iterative_params
        )
        
        # Start iterative training with proper resume handling
        if resume_from_iteration is not None:
            # Check if the found iteration is complete
            is_iteration_complete = check_iteration_complete(training_args.output_dir, resume_from_iteration, 10)
            
            if is_iteration_complete:
                # Complete iteration found - start from next iteration
                next_iteration = resume_from_iteration + 1
                print_master(f"Loaded COMPLETE iteration {resume_from_iteration}, starting from iteration {next_iteration}")
                trainer.iterative_train(resume_from_iteration=next_iteration)
            else:
                # Incomplete iteration found - resume from same iteration
                print_master(f"Loaded INCOMPLETE iteration {resume_from_iteration}, resuming from iteration {resume_from_iteration}")
                trainer.iterative_train(resume_from_iteration=resume_from_iteration)
        else:
            print_master("Starting iterative training from scratch")
            trainer.iterative_train(resume_from_iteration=0)
        
    else:
        print_master("Creating standard trainer...")
        if hasattr(model, "configure_loss"):
            model.configure_loss(
                info_nce_weight=getattr(training_args, "info_nce_weight", 1.0),
                triplet_loss_weight=getattr(training_args, "triplet_loss_weight", 0.0),
                triplet_margin=getattr(training_args, "triplet_margin", 0.2),
            )
        trainer = IterativeRetrievalTrainer(
            model=model,
            foundation_model=foundation_model,
            foundation_model_name=foundation_model_name,
            processing_class=processor,
            args=training_args,
            model_args=model_args,
            data_args=data_args,
            train_dataset=train_dataset,
            data_collator=train_collator,
            max_length=data_args.max_len,
        )
        
        # Standard training
        # Use trainer checkpoint if available and not using iteration model for weights
        checkpoint_to_resume = trainer_checkpoint if (use_trainer_for_state and not use_iteration_for_weights) else None
        trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    # Save final model
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)

    print_master("Training completed!")


if __name__ == "__main__":
    main()

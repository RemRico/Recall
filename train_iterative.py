#!/usr/bin/env python3
"""
Iterative Training Script for Composed Image Retrieval
Adapted from VLM2Vec training pipeline for iterative hard negative mining
"""

import logging
import os
import os.path
import sys
import datetime

# Enable wandb for production training monitoring
# os.environ['WANDB_DISABLED'] = 'true'  # Commented out to enable wandb

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


# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


# def load_foundation_model(model_args, data_args):
#     """Load foundation model for caption generation"""
#     foundation_model_name = getattr(model_args, 'foundation_model_name', None)
    
#     if foundation_model_name:
#         print_master(f"Loading foundation model: {foundation_model_name}")
        
#         # Load foundation model directly from transformers (not wrapped by MMEBModel)
#         from transformers import AutoModelForVision2Seq, AutoProcessor
#         import torch.distributed as dist
        
#         # Check if we're in distributed mode to avoid tensor parallel issues
#         if dist.is_initialized():
#             # Distributed mode: avoid device_map to prevent tensor parallel conflicts
#             foundation_model = AutoModelForVision2Seq.from_pretrained(
#                 foundation_model_name,
#                 torch_dtype=torch.bfloat16,
#                 device_map=None,  # Avoid tensor parallel issues in PyTorch 2.4
#                 trust_remote_code=True
#             )
#         else:
#             # Single GPU mode: use device_map="auto" for convenience
#             foundation_model = AutoModelForVision2Seq.from_pretrained(
#                 foundation_model_name,
#                 torch_dtype=torch.bfloat16,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
        
#         # Load processor
#         foundation_processor = AutoProcessor.from_pretrained(
#             foundation_model_name,
#             trust_remote_code=True
#         )
        
#         # Attach processor to model for easy access
#         setattr(foundation_model, 'processor', foundation_processor)
        
#         print_master(f"Foundation model loaded: {foundation_model_name}")
#         return foundation_model
#     else:
#         print_master("No foundation model specified")
#         return None


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
    
    # Debug distributed setup
    print("Distributed init debug info:")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")

    if torch.distributed.is_available():
        print(f"torch.distributed.is_initialized: {torch.distributed.is_initialized()}")
        if torch.distributed.is_initialized():
            print(f"torch.distributed.get_rank(): {torch.distributed.get_rank()}")
            print(f"torch.distributed.get_world_size(): {torch.distributed.get_world_size()}")


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
    
    # 修复：按照VLM2Vec方式，先从原始模型获取正确的model_backbone
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not getattr(model_args, "model_backbone", None):
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=getattr(model_args, 'model_type', None))
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_master(f'Model backbone: {model_args.model_backbone}')
    
    # ================================
    # 重新设计的恢复机制 - 分离两种不同的恢复方式
    # ================================
    
    print_master("=" * 60)
    print_master("CHECKPOINT RECOVERY SYSTEM")
    print_master("🔧 NEW: Each iteration saves checkpoints in separate subdirectories")
    print_master("   - training_iter_0/: Base model training checkpoints")  
    print_master("   - training_iter_1/: Iteration 1 training checkpoints")
    print_master("   - training_iter_2/: Iteration 2 training checkpoints")
    print_master("   - base_model/, iteration_1/, iteration_2/: Final models")
    print_master("=" * 60)
    
    # 1. Trainer checkpoint恢复 (包含optimizer/scheduler state)
    # 🔧 新策略：在子目录中查找最新的训练checkpoint
    trainer_checkpoint = None
    if training_args.resume_from == 'auto':
        # 首先尝试在主目录查找（兼容旧版本）
        trainer_checkpoint = find_latest_checkpoint(training_args.output_dir)
        
        # 如果主目录没有找到，在子目录中查找最新的checkpoint
        if not trainer_checkpoint:
            print_master("🔍 Searching for checkpoints in iteration subdirectories...")
            latest_iteration_dir = None
            latest_step = -1
            
            # 遍历所有可能的训练子目录
            for i in range(10, -1, -1):  # 从最新的迭代开始查找
                iteration_dir = os.path.join(training_args.output_dir, f"training_iter_{i}")
                if os.path.exists(iteration_dir):
                    # 在这个子目录中查找checkpoint
                    iter_checkpoint = find_latest_checkpoint(iteration_dir)
                    if iter_checkpoint:
                        # 提取step number
                        checkpoint_name = os.path.basename(iter_checkpoint)
                        if checkpoint_name.startswith('checkpoint-'):
                            try:
                                step_num = int(checkpoint_name.split('-')[1])
                                # 由于每轮训练都从0开始，我们需要考虑迭代顺序
                                # 更新的迭代具有更高的优先级
                                effective_step = i * 10000 + step_num  # 迭代权重 + 步数
                                if effective_step > latest_step:
                                    latest_step = effective_step
                                    latest_iteration_dir = iteration_dir
                                    trainer_checkpoint = iter_checkpoint
                                    print_master(f"Found checkpoint in iteration {i}: {iter_checkpoint}")
                            except ValueError:
                                continue
            
            if trainer_checkpoint:
                print_master(f"📁 Found trainer checkpoint in subdirectory: {trainer_checkpoint}")
                print_master(f"   ✅ Contains: model weights + optimizer + scheduler states")
            else:
                print_master("📁 No trainer checkpoint found in main directory or subdirectories")
        else:
            print_master(f"📁 Found trainer checkpoint in main directory: {trainer_checkpoint}")
            print_master(f"   ✅ Contains: model weights + optimizer + scheduler states")
            
    elif training_args.resume_from.isdigit():
        # 用户指定checkpoint step number - 需要在子目录中查找
        checkpoint_step = training_args.resume_from
        checkpoint_found = False
        
        # 在所有子目录中查找指定的checkpoint
        for i in range(10, -1, -1):
            iteration_dir = os.path.join(training_args.output_dir, f"training_iter_{i}")
            potential_checkpoint = os.path.join(iteration_dir, f'checkpoint-{checkpoint_step}')
            if os.path.exists(potential_checkpoint):
                trainer_checkpoint = potential_checkpoint
                checkpoint_found = True
                print_master(f"📁 Found specified checkpoint in iteration {i}: {trainer_checkpoint}")
                print_master(f"   ✅ Contains: model weights + optimizer + scheduler states")
                break
        
        # 也检查主目录（向后兼容）
        if not checkpoint_found:
            main_checkpoint = os.path.join(training_args.output_dir, f'checkpoint-{checkpoint_step}')
            if os.path.exists(main_checkpoint):
                trainer_checkpoint = main_checkpoint
                print_master(f"📁 Found specified checkpoint in main directory: {trainer_checkpoint}")
                print_master(f"   ✅ Contains: model weights + optimizer + scheduler states")
            else:
                print_master(f"📁 Specified trainer checkpoint not found: checkpoint-{checkpoint_step}")
                
    elif training_args.resume_from != 'none':
        print_master(f"⚠️  Unknown resume_from format: {training_args.resume_from}")
    
    # 2. 迭代模型恢复 (只包含模型权重)
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
        # 自动检测最新的完整迭代
        for i in range(10, -1, -1):
            if i == 0:
                model_path = os.path.join(training_args.output_dir, "base_model")
            else:
                model_path = os.path.join(training_args.output_dir, f"iteration_{i}")
            
            if os.path.exists(model_path) and check_iteration_complete(training_args.output_dir, i, 10):
                resume_from_iteration = i
                iteration_model = model_path
                print_master(f"🎯 Found COMPLETE iteration {i} model: {model_path}")
                print_master(f"   ⚠️  Contains: model weights only (no optimizer/scheduler)")
                break
                
        # 如果没有完整的迭代，找最新的不完整迭代
        if iteration_model is None:
            for i in range(10, -1, -1):
                if i == 0:
                    model_path = os.path.join(training_args.output_dir, "base_model")
                else:
                    model_path = os.path.join(training_args.output_dir, f"iteration_{i}")
                
                if os.path.exists(model_path):
                    resume_from_iteration = i
                    iteration_model = model_path
                    print_master(f"🎯 Found INCOMPLETE iteration {i} model: {model_path}")
                    print_master(f"   ⚠️  Contains: model weights only (no optimizer/scheduler)")
                    break
                    
    elif training_args.resume_from_iteration.startswith('iter_'):
        # 手动指定迭代
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
                print_master(f"🎯 Using specified {complete_status} iteration {iter_num} model: {model_path}")
                print_master(f"   ⚠️  Contains: model weights only (no optimizer/scheduler)")
            else:
                print_master(f"🎯 Specified iteration model not found: {model_path}")
                
    elif training_args.resume_from_iteration != 'none':
        print_master(f"⚠️  Unknown resume_from_iteration format: {training_args.resume_from_iteration}")
    
    # 3. 决定恢复策略
    print_master("-" * 60)
    print_master("RECOVERY STRATEGY:")
    
    if trainer_checkpoint and iteration_model:
        print_master("🔀 BOTH checkpoints found - using ITERATION model for weights")
        print_master("   📋 Reason: Iteration models contain the latest trained weights")
        print_master(f"   🎯 Model weights from: {iteration_model}")
        print_master(f"   📁 Training state from: {trainer_checkpoint}")
        use_iteration_for_weights = True
        use_trainer_for_state = True
    elif trainer_checkpoint:
        print_master("📁 Using TRAINER checkpoint (complete recovery)")
        print_master(f"   📁 Everything from: {trainer_checkpoint}")
        use_iteration_for_weights = False
        use_trainer_for_state = True
    elif iteration_model:
        print_master("🎯 Using ITERATION model (weights only)")
        print_master(f"   🎯 Model weights from: {iteration_model}")
        print_master("   ⚠️  No training state - will start fresh optimizer/scheduler")
        use_iteration_for_weights = True
        use_trainer_for_state = False
    else:
        print_master("🆕 No checkpoints found - starting from scratch")
        use_iteration_for_weights = False
        use_trainer_for_state = False
    
    print_master("=" * 60)
    
    # 4. 加载模型
    model = None
    
    if use_iteration_for_weights:
        print_master(f"Loading model weights from iteration checkpoint: {iteration_model}")
        try:
            # 为迭代模型创建config.json（如果不存在）
            config_path = os.path.join(iteration_model, "config.json")
            if not os.path.exists(config_path):
                print_master("🔧 Creating missing config.json for iteration model...")
                # 从原始模型复制配置
                import shutil
                original_config = os.path.join(model_args.model_name, "config.json")
                if os.path.exists(original_config):
                    shutil.copy2(original_config, config_path)
                    print_master("✅ Config.json created successfully")
                else:
                    print_master("⚠️  Original config.json not found, will use fallback method")
            
            model_args.checkpoint_path = iteration_model
            model = MMEBModel.load(model_args, is_trainable=True)
            print_master(f"✅ Successfully loaded model from iteration {resume_from_iteration}")
        except Exception as e:
            print_master(f"❌ Failed to load iteration checkpoint: {e}")
            print_master("🔄 Falling back to trainer checkpoint or new model")
            if hasattr(model_args, 'checkpoint_path'):
                delattr(model_args, 'checkpoint_path')
            model = None
            use_iteration_for_weights = False
    
    if model is None and use_trainer_for_state:
        print_master(f"Loading model from trainer checkpoint: {trainer_checkpoint}")
        try:
            model_args.checkpoint_path = trainer_checkpoint
            model = MMEBModel.load(model_args, is_trainable=True)
            print_master("✅ Successfully loaded model from trainer checkpoint")
        except Exception as e:
            print_master(f"❌ Failed to load trainer checkpoint: {e}")
            print_master("🔄 Will build new model")
            if hasattr(model_args, 'checkpoint_path'):
                delattr(model_args, 'checkpoint_path')
            model = None
    
    if model is None:
        print_master("🆕 Building new model from scratch...")
        model = MMEBModel.build(model_args)
        print_master("✅ New model built successfully")
    
    # Load processor
    processor = load_processor(model_args, data_args)
    
    # # 🔥 添加这一行来优化processor
    # processor = optimize_processor_for_memory(processor, max_pixels=200704)  # 448x448

    setattr(model, 'processor', processor)

    # Load foundation model for caption generation
    # foundation_model = load_foundation_model(model_args, data_args)
    foundation_model = None  # 不预加载
    foundation_model_name = getattr(model_args, "foundation_model_name", None)  # 例如 "Qwen2-VL-7B-Instruct"

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
                    # 🔧 Use new parameter name: steps_per_iteration instead of production_max_steps
                    steps_per_iter = config.get('steps_per_iteration', config.get('production_max_steps', 1000))
                    iterative_params.update({
                        'steps_per_iteration': steps_per_iter,  # New parameter name
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
            foundation_model=None,                   # 不传实例
            foundation_model_name=foundation_model_name,  # 只传名字
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
            foundation_model_name=foundation_model_name,  # <--- 补上这一行
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

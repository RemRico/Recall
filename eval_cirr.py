#!/usr/bin/env python3
"""
CIRR Evaluation Script for Composed Image Retrieval Models
Supports flexible model loading and evaluation with the CIRREvaluator
"""

import os
import sys
import json
import torch
import logging
import re  # NEW
from pathlib import Path
from typing import Dict
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import torch.distributed as dist

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, get_backbone_name
from src.model.processor import SUPPORTED_MODELS  # NEW: for validation
from src.evaluation.cirr_evaluator import CIRREvaluator
from src.utils import print_rank, print_master



# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass



logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class CIRREvalArguments:
    """CIRR Evaluation specific arguments"""
    model_path: str = field(
        metadata={"help": "Path to the trained model checkpoint (can be checkpoint-xxx or iteration_x directory)"}
    )
    base_model_name: str = field(
        default=None,
        metadata={"help": "Base model name (e.g., Qwen/Qwen2-VL-2B-Instruct). If not provided, will try to infer from model_path"}
    )
    eval_config: str = field(
        default=None,
        metadata={"help": "Path to evaluation configuration YAML file. If not provided, will use default CIRR config"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for evaluation (default: 8 to match training evaluation). Use smaller values if OOM occurs."}
    )
    device: str = field(
        default="auto",
        metadata={"help": "Device to use: 'auto', 'cuda', 'cuda:0', etc."}
    )
    distributed: bool = field(
        default=False,
        metadata={"help": "Whether to use distributed evaluation"}
    )
    output_file: str = field(
        default=None,
        metadata={"help": "File to save evaluation results (JSON format)"}
    )
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print detailed results"}
    )
    cirr_data_dir: str = field(
        default=None,
        metadata={"help": "Override CIRR dataset directory path"}
    )
    cirr_image_dir: str = field(
        default=None,
        metadata={"help": "Override CIRR image directory path"}
    )


def setup_device(device_arg: str, distributed: bool = False) -> str:
    """Setup and return the appropriate device"""
    
    # Initialize distributed if requested
    if distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # We're in a distributed environment (torchrun)
            try:
                dist.init_process_group(backend='nccl')
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                torch.cuda.set_device(local_rank)
                device = f"cuda:{local_rank}"
                print_master(f"Distributed mode initialized: rank {dist.get_rank()}/{dist.get_world_size()}")
                print_master(f"Using device: {device}")
                return device
            except Exception as e:
                print_master(f"Failed to initialize distributed mode: {e}")
                print_master("Falling back to single GPU mode")
                distributed = False
        else:
            print_master("No distributed environment variables found, using single GPU")
            distributed = False
    
    # Single GPU/CPU setup
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print_master(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print_master("CUDA not available, using CPU")
    else:
        device = device_arg
        print_master(f"Using specified device: {device}")
    
    return device


def infer_model_name_from_path(model_path: str, quiet: bool=False) -> str:
    """Try to infer base model name from checkpoint path.
    增强：支持 qwen2_5vl / qwen2.5-vl 等多种写法，允许静默模式。
    """
    path_lower = model_path.lower()
    # 直接匹配 qwen2(.5|_5)? + 可选分隔 + vl
    if re.search(r"qwen2(\.5|_5)?[-_]?vl", path_lower):
        is_qwen25 = bool(re.search(r"qwen2(\.5|_5)", path_lower))
        size = None
        if "2b" in path_lower:
            size = "2B"
        elif "7b" in path_lower:
            size = "7B"
        elif "32b" in path_lower:
            size = "32B"
        if is_qwen25:
            base = f"Qwen2.5-VL-{size or '7B'}-Instruct"
        else:
            base = f"Qwen2-VL-{size or '7B'}-Instruct"
        model_name = f"/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/{base}" if not base.startswith("/") else base
        if not quiet:
            print_master(f"Inferred base model name from path pattern: {model_name}")
        return model_name
    # 原始逻辑回退
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            for key in ['_name_or_path', 'name_or_path', 'model_name', 'base_model_name']:
                if key in config:
                    model_name = config[key]
                    if not quiet:
                        print_master(f"Inferred base model name from config: {model_name}")
                    return model_name
        except Exception as e:
            if not quiet:
                print_master(f"Warning: Could not read config.json: {e}")
    # 最终回退
    default_name = "Qwen/Qwen2-VL-2B-Instruct"
    if not quiet:
        print_master("Warning: Could not infer base model name, using default")
    return default_name


def create_eval_config_if_needed(eval_args: CIRREvalArguments) -> str:
    """Return eval config path if provided, otherwise return None to use defaults"""
    eval_config = getattr(eval_args, "eval_config", None)
    if eval_config and os.path.exists(eval_config):
        return eval_config
    
    # Don't create temporary config file, let evaluator use default values
    print_master("No eval config provided, will use default CIRR evaluation settings")
    return None


DEFAULT_RESIZE_MAX_PIXELS = 28 * 28 * 1280


def _find_file_in_hierarchy(model_path: str, filename: str) -> Path:
    """
    Search for a file in the model directory, its parents, and common sibling folders.
    Useful for locating artifacts like preprocessor_config.json or training_args.bin.
    """
    path = Path(model_path).resolve()
    candidates = [
        path,
        path.parent if path.is_dir() else path.parent,
        path.parent / "base_model",
        path.parent.parent if path.parent != path else path.parent.parent,
    ]
    seen = set()
    for base in candidates:
        if base is None or base in seen:
            continue
        seen.add(base)
        target = base / filename
        if target.exists():
            return target
    return None


def _align_data_args_with_training(model_path: str, data_args: DataArguments, verbose=True):
    """
    Align evaluation data arguments with training metadata when possible.
    Currently focuses on resize_max_pixels so evaluation and training use consistent resolutions.
    """
    preprocessor_cfg = _find_file_in_hierarchy(model_path, "preprocessor_config.json")
    training_max_pixels = None

    if preprocessor_cfg and preprocessor_cfg.exists():
        try:
            with open(preprocessor_cfg, "r") as f:
                cfg = json.load(f)
            training_max_pixels = cfg.get("max_pixels") or cfg.get("size", {}).get("max_pixels")
        except Exception as exc:
            if verbose:
                print_master(f"Warning: Failed to read {preprocessor_cfg}: {exc}")

    user_specified = getattr(data_args, "resize_max_pixels", None)
    if user_specified is None:
        user_specified = DEFAULT_RESIZE_MAX_PIXELS

    if training_max_pixels is not None:
        if user_specified == DEFAULT_RESIZE_MAX_PIXELS:
            data_args.resize_max_pixels = training_max_pixels
            if verbose:
                print_master(f"✅ Aligning resize_max_pixels with training value: {training_max_pixels}")
        else:
            if verbose:
                print_master(f"✅ Using user-specified resize_max_pixels: {user_specified}")
    else:
        if user_specified != DEFAULT_RESIZE_MAX_PIXELS:
            if verbose:
                print_master(f"✅ Using user-specified resize_max_pixels: {user_specified}")
        else:
            if verbose:
                print_master("⚠️ Could not infer training resize_max_pixels; using library default.")


def detect_fullft_checkpoint(model_path: str) -> bool:
    """
    Detect if this is a full fine-tuning checkpoint.
    
    Heuristics:
    - Path contains "FullFT" or "fullft"
    - Has model.safetensors or pytorch_model.bin but NO adapter_config.json
    """
    path_str = str(model_path).lower()
    has_fullft_marker = 'fullft' in path_str or 'full_ft' in path_str or 'fullfinetune' in path_str
    
    # Check for model files
    has_model_file = False
    if os.path.isdir(model_path):
        for fname in ['model.safetensors', 'pytorch_model.bin', 'model.bin']:
            if os.path.exists(os.path.join(model_path, fname)):
                has_model_file = True
                break
    
    # Check for adapter config (LoRA marker)
    adapter_config = os.path.join(model_path, "adapter_config.json")
    has_adapter = os.path.exists(adapter_config)
    
    # FullFT if: path marker OR (has model file AND no adapter)
    is_fullft = has_fullft_marker or (has_model_file and not has_adapter)
    
    return is_fullft


def load_model_and_processor(eval_args: CIRREvalArguments, model_args: ModelArguments, data_args: DataArguments):
    """Load model and processor with proper error handling"""
    print_master("=" * 60)
    print_master("LOADING MODEL AND PROCESSOR")
    print_master("=" * 60)

    # 先确定潜在的 adapter / config 路径
    local_config_path = os.path.join(eval_args.model_path, "config.json")
    adapter_config_path = os.path.join(eval_args.model_path, "adapter_config.json")
    
    # Detect full fine-tuning checkpoint
    is_fullft = detect_fullft_checkpoint(eval_args.model_path)
    if is_fullft:
        print_master(f"🔍 Detected FULL FINE-TUNING checkpoint")
    else:
        print_master(f"🔍 Detected LoRA or standard checkpoint")

    # 优先：如果是 LoRA，则直接读取 adapter_config，避免先推测再覆盖的冗余日志
    base_model_name = None
    lora_mode = False
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            if 'base_model_name_or_path' in adapter_config:
                base_model_name = adapter_config['base_model_name_or_path']
                lora_mode = True
                print_master(f"Detected LoRA adapter. Base model: {base_model_name}")
        except Exception as e:
            print_master(f"Failed reading adapter_config.json (will fallback to inference): {e}")

    # 如果不是 LoRA 或者 LoRA 读取失败，再进行推测或使用用户输入
    if eval_args.base_model_name:
        base_model_name = eval_args.base_model_name
        print_master(f"Using provided base model name: {base_model_name}")
    if base_model_name is None:
        # 静默推测（减少无意义 Warning），只有最终结果打印
        inferred = infer_model_name_from_path(eval_args.model_path, quiet=True)
        base_model_name = inferred
        print_master(f"Inferred base model name: {base_model_name}")

    # 设置 model_args.model_name
    if model_args.model_name in [None, "auto-infer"]:
        model_args.model_name = base_model_name
        print_master(f"Final model_name set to: {model_args.model_name}")

    # checkpoint 路径（LoRA: 指向 adapter；全量：指向权重目录）
    model_args.checkpoint_path = eval_args.model_path
    model_args.lora = lora_mode or getattr(model_args, 'lora', False)

    # 尝试与训练时配置对齐（保留自定义的 resize_max_pixels 用于实验）
    _align_data_args_with_training(eval_args.model_path, data_args)

    # 备选分辨率（便于快速切换做对比测试）
    # data_args.resize_max_pixels = 35840   #  ~sqrt(35840)=189 (早期小分辨率/调试)
    # data_args.resize_max_pixels = 50176   # 224*224
    # data_args.resize_max_pixels = 82944   # 288*288
    # data_args.resize_max_pixels = 147456  # 384*384
    # data_args.resize_max_pixels = 262144  # 512*512 默认评测分辨率
    print_master(f"✅ Set resize_max_pixels={data_args.resize_max_pixels}")

    # Backbone 识别：优先 true config，再 fallback
    try:
        from transformers import AutoConfig
        base_cfg = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        detected_backbone = get_backbone_name(base_cfg, getattr(model_args, 'model_type', None))
        model_args.model_backbone = detected_backbone
        print_master(f"Detected backbone: {detected_backbone}")
    except Exception as e_det:
        bl = model_args.model_name.lower()
        if 'qwen2.5' in bl or 'qwen2_5' in bl:
            model_args.model_backbone = 'qwen2_5_vl'
        elif 'qwen2' in bl:
            model_args.model_backbone = 'qwen2_vl'
        elif 'llava' in bl:
            model_args.model_backbone = 'llava_next'
        else:
            model_args.model_backbone = 'qwen2_vl'
        print_master(f"Backbone detection fallback due to {e_det}: {model_args.model_backbone}")

    # 构建 / 加载模型
    model = None
    if model_args.lora:
        print_master("Loading LoRA model (base + adapter)...")
        try:
            # Keep adapters unmerged to replicate training-time evaluation exactly
            model = MMEBModel.load(model_args, is_trainable=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            print_master("✅ LoRA model loaded successfully (adapters kept unmerged)")
        except Exception as e:
            print_master(f"❌ LoRA model loading failed: {e}")
            raise
    else:
        # Choose model class based on fullft detection
        ModelClass = MMEBModelFullFT if is_fullft else MMEBModel
        model_type_str = "FULL FINE-TUNED" if is_fullft else "full"
        
        print_master(f"Loading {model_type_str} model from local checkpoint (non-LoRA)...")
        try:
            model = ModelClass.load(model_args, is_trainable=False)
            model.eval()
            print_master(f"✅ {model_type_str} model loaded successfully from local checkpoint")
        except Exception as e:
            print_master(f"{ModelClass.__name__}.load failed: {e}")
            print_master("Trying build + manual weight load fallback...")
            try:
                original_checkpoint = model_args.checkpoint_path
                model_args.checkpoint_path = None
                model = ModelClass.build(model_args)
                weight_file = None
                if os.path.isdir(eval_args.model_path):
                    for f in ["model.safetensors", "pytorch_model.bin", "model.bin"]:
                        fp = os.path.join(eval_args.model_path, f)
                        if os.path.exists(fp):
                            weight_file = fp; break
                
                if weight_file:
                    print_master(f"Loading weights from: {weight_file}")
                    if weight_file.endswith('.safetensors'):
                        from safetensors import safe_open
                        sd = {}
                        with safe_open(weight_file, framework="pt", device="cpu") as sf:
                            for k in sf.keys():
                                sd[k] = sf.get_tensor(k)
                    else:
                        sd = torch.load(weight_file, map_location='cpu')
                    
                    # For full fine-tuning, state_dict keys should have 'encoder.' prefix
                    if is_fullft and not any(k.startswith('encoder.') for k in sd.keys()):
                        print_master("[FullFT] Adding 'encoder.' prefix to state_dict keys")
                        sd = {'encoder.' + k: v for k, v in sd.items()}
                    
                    model.load_state_dict(sd, strict=False)
                    print_master("✅ Weights loaded into built model")
                else:
                    err_msg = f"未在 {eval_args.model_path} 中找到可加载的模型权重文件 (adapter_model.safetensors/adapter_model.bin/model.safetensors/pytorch_model.bin)."
                    print_master(f"❌ {err_msg}")
                    raise ValueError(err_msg)
                model_args.checkpoint_path = original_checkpoint
            except Exception as e2:
                print_master(f"❌ All loading methods failed: {e2}")
                raise

    # Processor
    print_master("Loading processor...")
    try:
        processor = load_processor(model_args, data_args)
        print_master("✅ Processor loaded successfully")
    except Exception as e:
        print_master(f"❌ Failed to load processor: {e}")
        raise

    setattr(model, 'processor', processor)
    print_master(
        f"Final evaluation args -> pooling: {getattr(model_args, 'pooling', None)}, "
        f"normalize: {getattr(model_args, 'normalize', None)}, "
        f"max_len: {getattr(data_args, 'max_len', None)}, "
        f"resize_max_pixels: {getattr(data_args, 'resize_max_pixels', None)}"
    )
    print_master("=" * 60)
    return model, processor


def run_evaluation(model, processor, eval_args: CIRREvalArguments, model_args: ModelArguments, data_args: DataArguments, device: str) -> Dict[str, float]:
    """Run CIRR evaluation and return results"""
    
    print_master("=" * 60)
    print_master("STARTING CIRR EVALUATION")
    print_master("=" * 60)
    
    # Create evaluation config
    eval_config_path = create_eval_config_if_needed(eval_args)
    
    # Use the specified batch size directly
    eval_batch_size = eval_args.batch_size
    print_master(f"Using batch_size: {eval_batch_size}")
    
    # Create CIRR evaluator
    try:
        evaluator = CIRREvaluator(
            model=model,
            processor=processor,
            data_args=data_args,
            model_args=model_args,
            device=device,
            batch_size=eval_batch_size
            # Note: Not passing eval_config_path to match training behavior
        )
        print_master("✅ CIRR evaluator created successfully")
    except Exception as e:
        print_master(f"❌ Failed to create CIRR evaluator: {e}")
        raise
    
    try:
        # Check if distributed evaluation is available (match training logic)
        import torch.distributed as dist
        use_distributed = (dist.is_initialized() and 
                         dist.get_world_size() > 1 and 
                         hasattr(evaluator, '_evaluate_distributed'))
        
        # Override with user's distributed preference if explicitly set
        if eval_args.distributed is not None:
            use_distributed = eval_args.distributed and dist.is_initialized()
        
        # Run evaluation
        if use_distributed:
            print_master(f"Using distributed evaluation across {dist.get_world_size()} GPUs")
            results = evaluator.evaluate(distributed=True)
        else:
            print_master("Using single GPU evaluation")
            results = evaluator.evaluate(distributed=False)
        
        # Add evaluation metadata
        results['evaluation_mode'] = 'distributed' if use_distributed else 'single_gpu'
        results['batch_size'] = eval_batch_size
        
        if eval_args.verbose:
            print_master("\n" + "=" * 60)
            print_master("FINAL CIRR EVALUATION RESULTS")
            print_master("=" * 60)
            
            # Format and print results
            for metric, value in results.items():
                if isinstance(value, float):
                    print_master(f"{metric}: {value:.4f}")
                else:
                    print_master(f"{metric}: {value}")
            
            print_master("=" * 60)
        
        # Clean up temporary config if created (no longer needed since we don't create temp files)
        # if eval_config_path and eval_config_path.startswith("temp_eval_config_rank"):
        #     ... cleanup code removed ...
        
        return results
        
    except Exception as e:
        print_master(f"❌ CIRR evaluation failed: {e}")
        import traceback
        print_master(f"Traceback: {traceback.format_exc()}")
        raise


def save_results(results: Dict[str, float], eval_args: CIRREvalArguments, model_args: ModelArguments):
    """Save evaluation results to file (append mode)"""
    
    if eval_args.output_file:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(eval_args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        current_result = {
            "timestamp": timestamp,
            "model_path": eval_args.model_path,
            "model_name": eval_args.base_model_name or "inferred",
            "model_backbone": getattr(model_args, 'model_backbone', 'unknown'),
            "batch_size": results.get('batch_size', eval_args.batch_size),  # Use actual batch_size from results
            "distributed": eval_args.distributed,
            "evaluation_mode": results.get('evaluation_mode', 'unknown'),
            "results": results
        }
        
        try:
            # Load existing results if file exists
            all_results = []
            if os.path.exists(eval_args.output_file):
                try:
                    with open(eval_args.output_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        # Handle both old format (single result) and new format (list of results)
                        if isinstance(existing_data, list):
                            all_results = existing_data
                        else:
                            # Convert old single result format to new list format
                            all_results = [existing_data]
                except json.JSONDecodeError:
                    print_master(f"Warning: Could not parse existing results file, starting fresh")
                    all_results = []
                except UnicodeDecodeError as e:
                    print_master(f"Warning: Could not decode existing results file ({e}), starting fresh")
                    all_results = []
                    # Try to backup the corrupted file
                    import shutil
                    backup_file = eval_args.output_file + '.backup'
                    try:
                        shutil.copy2(eval_args.output_file, backup_file)
                        print_master(f"Backed up corrupted file to: {backup_file}")
                    except Exception as backup_e:
                        print_master(f"Could not create backup: {backup_e}")
            
            # Append current result
            all_results.append(current_result)
            
            # Save updated results with proper encoding
            with open(eval_args.output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print_master(f"Results appended to: {eval_args.output_file}")
            print_master(f"Total evaluation records: {len(all_results)}")
            
        except Exception as e:
            print_master(f"Warning: Could not save results: {e}")


def main():
    """Main evaluation function"""
    
    # Parse arguments
    parser = HfArgumentParser((CIRREvalArguments, ModelArguments, DataArguments))
    
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        # Load from JSON config file
        eval_args, model_args, data_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        # Parse from command line
        eval_args, model_args, data_args = parser.parse_args_into_dataclasses()
    
    # Validate arguments
    if not eval_args.model_path:
        raise ValueError("model_path is required")
    
    if not os.path.exists(eval_args.model_path):
        raise ValueError(f"Model path does not exist: {eval_args.model_path}")
    
    # Setup device
    device = setup_device(eval_args.device, eval_args.distributed)
    
    # Load model and processor (all processes)
    model, processor = load_model_and_processor(eval_args, model_args, data_args)
    
    # Move model to device
    model = model.to(device)
    print_master(f"Model moved to device: {device}")
    
    # Run evaluation (all processes participate)
    results = run_evaluation(model, processor, eval_args, model_args, data_args, device)
    
    # Determine if this is the main process
    is_main_process = True
    if eval_args.distributed and 'RANK' in os.environ:
        rank = int(os.environ.get('RANK', 0))
        is_main_process = (rank == 0)
    
    # Save results and print completion message (only on main process)
    if is_main_process:
        save_results(results, eval_args, model_args)
        print_master("🎉 CIRR evaluation completed successfully!")
    
    # Clean up distributed (all processes)
    if eval_args.distributed and dist.is_initialized():
        dist.destroy_process_group()
    
    return results


if __name__ == "__main__":
    main()

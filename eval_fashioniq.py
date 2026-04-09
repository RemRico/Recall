#!/usr/bin/env python3
"""
FashionIQ Evaluation Script for Composed Image Retrieval Models
Supports flexible model loading and evaluation with the FashionIQEvaluator
"""

import os
import sys
import json
import torch
import logging
from typing import Tuple
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import torch.distributed as dist

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, get_backbone_name
from src.evaluation.fashioniq_evaluator import FashionIQEvaluator

logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class FashionIQEvalArguments:
    """FashionIQ Evaluation specific arguments"""
    model_path: str = field(
        metadata={"help": "Path to the trained model checkpoint (e.g., checkpoint-5000 or iteration_x directory)"}
    )
    base_model_name: str = field(
        default=None,
        metadata={"help": "Base model name (e.g., Qwen/Qwen2-VL-2B-Instruct). If not provided, will try to infer from model_path"}
    )
    eval_config: str = field(
        default="configs/fashioniq_eval_config.yaml",
        metadata={"help": "Path to evaluation configuration YAML file"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for evaluation. Use smaller values if OOM occurs."}
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
        metadata={"help": "File to save evaluation results (JSON format). If not provided, saves to model_path directory"}
    )
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print detailed results"}
    )
    fashioniq_data_dir: str = field(
        default=None,
        metadata={"help": "Override FashionIQ dataset directory path"}
    )
    fashioniq_image_dir: str = field(
        default=None,
        metadata={"help": "Override FashionIQ image directory path"}
    )


def setup_device(device_arg: str, distributed: bool = False) -> Tuple[str, bool, int, int]:
    """
    Setup and return the appropriate device
    Returns: (device, is_distributed, rank, world_size)
    """
    
    if distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            try:
                if not dist.is_initialized():
                    dist.init_process_group(backend='nccl')
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                device = f'cuda:{rank}'
                if rank == 0:
                    logger.info(f"Distributed mode: {world_size} GPUs, rank {rank}, device {device}")
                return device, True, rank, world_size
            except Exception as e:
                logger.warning(f"Failed to initialize distributed: {e}. Falling back to single GPU")
        else:
            logger.warning("Distributed mode requested but RANK/WORLD_SIZE not in env. Use: torchrun --nproc_per_node=N eval_fashioniq.py ...")
    
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = device_arg
    
    logger.info(f"Using device: {device}")
    return device, False, 0, 1


def infer_base_model_from_checkpoint(model_path: str) -> str:
    """
    Try to infer the base model name from checkpoint structure.
    Looks for adapter_config.json or config.json.
    """
    adapter_config_path = os.path.join(model_path, 'adapter_config.json')
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                base_model = adapter_config.get('base_model_name_or_path')
                if base_model:
                    logger.info(f"Inferred base model from adapter_config: {base_model}")
                    return base_model
        except Exception as e:
            logger.warning(f"Failed to read adapter_config.json: {e}")
    
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_type = config.get('model_type', '')
                if 'qwen3_vl' in model_type.lower():
                    full_path_lower = model_path.lower()
                    if 'qwen3-vl-8b' in full_path_lower or 'qwen3vl_8b' in full_path_lower:
                        return "Qwen/Qwen3-VL-8B-Instruct"
                    else:
                        logger.warning(f"Could not determine exact Qwen3 model, defaulting to Qwen3-VL-8B-Instruct")
                        return "Qwen/Qwen3-VL-8B-Instruct"
                elif 'qwen2_vl' in model_type.lower() or 'qwen2_5_vl' in model_type.lower():
                    full_path_lower = model_path.lower()
                    if 'qwen2.5-vl-7b' in full_path_lower or 'qwen2_5vl_7b' in full_path_lower:
                        return "Qwen/Qwen2.5-VL-7B-Instruct"
                    elif 'qwen2.5-vl-2b' in full_path_lower or 'qwen2_5vl_2b' in full_path_lower:
                        return "Qwen/Qwen2.5-VL-2B-Instruct"
                    elif 'qwen2-vl-7b' in full_path_lower or 'qwen2vl_7b' in full_path_lower:
                        return "Qwen/Qwen2-VL-7B-Instruct"
                    elif 'qwen2-vl-2b' in full_path_lower or 'qwen2vl_2b' in full_path_lower:
                        return "Qwen/Qwen2-VL-2B-Instruct"
                    else:
                        logger.warning(f"Could not determine exact model size from path, defaulting to Qwen2.5-VL-7B-Instruct")
                        return "Qwen/Qwen2.5-VL-7B-Instruct"
        except Exception as e:
            logger.warning(f"Failed to read config.json: {e}")
    
    full_path_lower = model_path.lower()
    if 'qwen3-vl' in full_path_lower or 'qwen3vl' in full_path_lower:
        logger.info("Inferred from path: Qwen3-VL-8B-Instruct")
        return "Qwen/Qwen3-VL-8B-Instruct"
    elif 'qwen2.5-vl-7b' in full_path_lower or 'qwen2_5vl_7b' in full_path_lower:
        logger.info("Inferred from path: Qwen2.5-VL-7B-Instruct")
        return "Qwen/Qwen2.5-VL-7B-Instruct"
    elif 'qwen2.5-vl-2b' in full_path_lower or 'qwen2_5vl_2b' in full_path_lower:
        logger.info("Inferred from path: Qwen2.5-VL-2B-Instruct")
        return "Qwen/Qwen2.5-VL-2B-Instruct"
    elif 'qwen2-vl' in full_path_lower or 'qwen2vl' in full_path_lower:
        logger.info("Inferred from path: Qwen2-VL-7B-Instruct (default)")
        return "Qwen/Qwen2-VL-7B-Instruct"
    
    raise ValueError(
        f"Could not infer base model from checkpoint at {model_path}. "
        "Please provide --base_model_name explicitly.\n"
        "Example: --base_model_name 'Qwen/Qwen2.5-VL-7B-Instruct'"
    )


def load_model_and_processor(
    model_path: str,
    base_model_name: str,
    device: str
):
    """
    Load model and processor from checkpoint (supports both LoRA and full models)
    """
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Base model: {base_model_name}")
    
    lora_mode = False
    adapter_config_path = os.path.join(model_path, 'adapter_config.json')
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            if 'base_model_name_or_path' in adapter_config:
                base_model_from_adapter = adapter_config['base_model_name_or_path']
                lora_mode = True
                logger.info(f"Detected LoRA adapter. Base model: {base_model_from_adapter}")
                if base_model_name and base_model_name != base_model_from_adapter:
                    logger.warning(f"Base model mismatch: provided={base_model_name}, adapter={base_model_from_adapter}")
                    logger.warning(f"Using adapter config: {base_model_from_adapter}")
                base_model_name = base_model_from_adapter
        except Exception as e:
            logger.warning(f"Failed reading adapter_config.json: {e}")
    
    base_model_path = base_model_name
    
    model_args = ModelArguments(
        model_name=base_model_path,
        model_type='vlm2vec',
        normalize=True,
        pooling='eos',
        lora=lora_mode,
        checkpoint_path=model_path if lora_mode else None,
    )

    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config, model_type=None)
    model_args.model_backbone = model_backbone
    logger.info(f"Model backbone: {model_backbone}")

    if hf_config.model_type in ['qwen2_5_vl', 'qwen2_vl', 'qwen3_vl']:
        model_args.model_type = hf_config.model_type
        logger.info(f"Updated model_type to: {model_args.model_type}")

    data_args = DataArguments(
        dataset_name='fashioniq',
        dataset_config=None,
        max_len=512,
    )

    logger.info("Loading processor...")
    processor = load_processor(model_args, data_args)

    if lora_mode:
        logger.info("Loading LoRA model (base + adapter)...")
        try:
            model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
            model.eval()
            logger.info("LoRA model loaded successfully")
        except Exception as e:
            logger.error(f"LoRA model loading failed: {e}")
            raise
    else:
        logger.info("Loading full model from checkpoint...")
        try:
            model_args.checkpoint_path = model_path
            model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
            model.eval()
            logger.info("Model loaded successfully from checkpoint")
        except Exception as e:
            logger.warning(f"MMEBModel.load failed: {e}")
            logger.info("Trying build + manual weight load fallback...")
            try:
                model_args.checkpoint_path = None
                model = MMEBModel.build(model_args)

                weight_file = None
                for f in ["pytorch_model.bin", "model.safetensors", "model.bin"]:
                    fp = os.path.join(model_path, f)
                    if os.path.exists(fp):
                        weight_file = fp
                        break
                
                if weight_file:
                    logger.info(f"Loading weights from: {weight_file}")
                    if weight_file.endswith('.safetensors'):
                        from safetensors import safe_open
                        sd = {}
                        with safe_open(weight_file, framework="pt", device="cpu") as sf:
                            for k in sf.keys():
                                sd[k] = sf.get_tensor(k)
                    else:
                        sd = torch.load(weight_file, map_location='cpu')

                    model.load_state_dict(sd, strict=False)
                    logger.info("Weights loaded into built model")
                else:
                    logger.warning(f"No weight file found at {model_path}, using base model weights only")
            except Exception as e2:
                logger.error(f"All loading methods failed: {e2}")
                raise

    model = model.to(device)
    model.eval()
    
    return model, processor, model_args, data_args


def main():
    parser = HfArgumentParser((FashionIQEvalArguments,))
    eval_args, = parser.parse_args_into_dataclasses()
    eval_args: FashionIQEvalArguments

    device, is_distributed, rank, world_size = setup_device(eval_args.device, eval_args.distributed)

    is_main_process = (rank == 0)

    if eval_args.base_model_name is None:
        eval_args.base_model_name = infer_base_model_from_checkpoint(eval_args.model_path)
        if is_main_process:
            logger.info(f"Inferred base model: {eval_args.base_model_name}")

    if is_main_process:
        logger.info(f"Loading model on rank {rank}...")
    
    model, processor, model_args, data_args = load_model_and_processor(
        eval_args.model_path,
        eval_args.base_model_name,
        device
    )

    eval_config_path = eval_args.eval_config
    if not os.path.exists(eval_config_path):
        if is_main_process:
            logger.warning(f"Eval config not found at {eval_config_path}, using default config")
        eval_config_path = None

    if is_main_process:
        logger.info("Creating FashionIQ evaluator...")
    
    evaluator = FashionIQEvaluator(
        model=model,
        processor=processor,
        data_args=data_args,
        model_args=model_args,
        device=device,
        batch_size=eval_args.batch_size,
        eval_config_path=eval_config_path
    )

    if eval_args.fashioniq_data_dir:
        evaluator.data_dir = eval_args.fashioniq_data_dir
        if is_main_process:
            logger.info(f"Using custom data_dir: {eval_args.fashioniq_data_dir}")
    
    if eval_args.fashioniq_image_dir:
        evaluator.image_base_dir = eval_args.fashioniq_image_dir
        if is_main_process:
            logger.info(f"Using custom image_base_dir: {eval_args.fashioniq_image_dir}")

    if is_distributed:
        dist.barrier()

    if is_main_process:
        logger.info("Starting FashionIQ evaluation...")
        total_queries = sum(len(evaluator.category_data[cat]['queries']) for cat in evaluator.categories)
        total_candidates = sum(len(evaluator.category_data[cat]['candidates']) for cat in evaluator.categories)
        logger.info(f"Evaluating {total_queries} queries against {total_candidates} candidates (split by category)")
        if is_distributed:
            logger.info(f"Using distributed evaluation with {world_size} GPUs")

    results = evaluator.evaluate(distributed=is_distributed)

    if is_distributed:
        dist.barrier()

    if is_main_process:
        if eval_args.verbose:
            logger.info("=" * 60)
            logger.info("FashionIQ Evaluation Results")
            logger.info("=" * 60)
            for metric, value in results.items():
                logger.info("%-25s: %.4f", metric, value)
            logger.info("=" * 60)

        if eval_args.output_file:
            output_file = eval_args.output_file
        else:
            output_file = os.path.join(eval_args.model_path, 'fashioniq_eval_results.json')

        results['model_path'] = eval_args.model_path
        results['base_model'] = eval_args.base_model_name
        results['batch_size'] = eval_args.batch_size
        results['distributed'] = is_distributed
        results['world_size'] = world_size
        total_queries = sum(len(evaluator.category_data[cat]['queries']) for cat in evaluator.categories)
        total_candidates = sum(len(evaluator.category_data[cat]['candidates']) for cat in evaluator.categories)
        results['num_queries'] = total_queries
        results['num_candidates'] = total_candidates
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_file}")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


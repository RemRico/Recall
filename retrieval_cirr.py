#!/usr/bin/env python3
"""
CIRR Retrieval Script

Computes embeddings for all candidates and queries, then returns top-k
retrieval results per query.  Reuses the unified encoding pipeline from
CIRREvaluator for consistency with training and evaluation.
"""

import os
import sys
import re
import json
import torch
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, get_backbone_name
from src.utils import print_rank, print_master


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class CIRRRetrievalArguments:
    """CIRR retrieval arguments"""
    model_path: str = field(
        metadata={"help": "Path to trained model checkpoint (checkpoint-xxx or iteration_x directory)"}
    )
    base_model_name: str = field(
        default=None,
        metadata={"help": "Base model name (e.g., Qwen/Qwen2-VL-2B-Instruct). Auto-inferred if not provided"}
    )
    output_file: str = field(
        default=None,
        metadata={"help": "JSON file to save retrieval results. Auto-generated if not provided"}
    )
    top_k: int = field(
        default=10,
        metadata={"help": "Number of top-k results per query (default: 10)"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for encoding (default: 8)"}
    )
    device: str = field(
        default="auto",
        metadata={"help": "Device: 'auto', 'cuda', 'cuda:0', etc."}
    )
    cirr_data_dir: str = field(
        default=None,
        metadata={"help": "CIRR data directory path (containing captions/ and image_splits/)"}
    )
    cirr_image_dir: str = field(
        default=None,
        metadata={"help": "CIRR image root directory (containing dev/test/train subdirs)"}
    )
    save_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to also save query and candidate embeddings (can produce large files)"}
    )
    distributed: bool = field(
        default=False,
        metadata={"help": "Whether to use distributed encoding via torchrun"}
    )


class CIRRRetriever:
    """
    CIRR retriever: loads data, encodes via CIRREvaluator pipeline,
    computes similarity and exports top-k results.
    """

    def __init__(
        self,
        model,
        processor,
        data_args,
        model_args,
        device='cuda',
        batch_size=8,
        cirr_data_dir=None,
        cirr_image_dir=None,
        distributed: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.data_args = data_args
        self.model_args = model_args
        self.device = device
        self.batch_size = batch_size
        self.distributed = distributed
        self.model_backbone = getattr(model_args, 'model_backbone', 'qwen2_vl')

        self._configure_data_paths(cirr_data_dir, cirr_image_dir)
        self.test_data, self.candidate_images = self._load_cirr_test_data()

        try:
            from src.evaluation.cirr_evaluator import CIRREvaluator
            self._evaluator = CIRREvaluator(
                model=self.model,
                processor=self.processor,
                data_args=self.data_args,
                model_args=self.model_args,
                device=self.device,
                batch_size=self.batch_size,
            )
        except Exception as e:
            print_master(f"Warning: CIRREvaluator creation failed, falling back to simplified encoding: {e}")
            self._evaluator = None

        print_master(f"Loaded {len(self.test_data)} queries")
        print_master(f"Loaded {len(self.candidate_images)} candidate images")

    def _configure_data_paths(self, cirr_data_dir=None, cirr_image_dir=None):
        if not cirr_data_dir or not cirr_image_dir:
            raise ValueError(
                "CIRR data paths must be provided via --cirr_data_dir and --cirr_image_dir. "
                "See README for dataset download instructions."
            )
        self.data_dir = cirr_data_dir
        self.image_base_dir = cirr_image_dir
        self.captions_file = os.path.join(self.data_dir, 'captions/cap.rc2.val.json')
        self.image_splits_file = os.path.join(self.data_dir, 'image_splits/split.rc2.val.json')

        print_master(f"CIRR data dir: {self.data_dir}")
        print_master(f"CIRR image dir: {self.image_base_dir}")

    def _load_cirr_test_data(self) -> Tuple[List[Dict], List[str]]:
        try:
            if not os.path.exists(self.captions_file):
                print_master(f"Warning: Validation query file not found: {self.captions_file}")
                return self._create_dummy_test_data()

            with open(self.captions_file, 'r') as f:
                val_queries = json.load(f)

            if os.path.exists(self.image_splits_file):
                with open(self.image_splits_file, 'r') as f:
                    val_splits = json.load(f)
                candidate_images = list(val_splits.keys())
                self.image_splits = val_splits
                print_master(f"Loaded {len(candidate_images)} candidate images from validation splits")
            else:
                print_master(f"Warning: Validation splits file not found: {self.image_splits_file}")
                candidate_images = [f"dummy_img_{i}" for i in range(100)]
                self.image_splits = {}

            print_master(f"Loaded {len(val_queries)} validation queries")
            return val_queries, candidate_images
        except Exception as e:
            print_master(f"Failed to load CIRR validation data: {e}")
            return self._create_dummy_test_data()

    def _create_dummy_test_data(self) -> Tuple[List[Dict], List[str]]:
        """Create dummy data for debugging without actual data files."""
        dummy_data = []
        for i in range(50):
            dummy_data.append({
                'pairid': i,
                'reference': f'dummy_ref_{i}',
                'target_hard': f'dummy_target_{i}',
                'caption': f'dummy modification text {i}',
                'target_soft': {},
                'img_set': {'members': [f'dummy_img_{j}' for j in range(i, i + 5)]},
            })
        candidate_images = [f"dummy_img_{i}" for i in range(100)]
        self.image_splits = {}
        return dummy_data, candidate_images

    def _encode_images(self, image_names: List[str]) -> torch.Tensor:
        if self._evaluator is not None and hasattr(self._evaluator, '_encode_images_local'):
            return self._evaluator._encode_images_local(image_names)
        print_master("Warning: Falling back to empty image embeddings, check evaluator availability")
        return torch.empty(0, 512, device=self.device)

    def _encode_composed_queries(self, queries: List[Dict]) -> torch.Tensor:
        if self._evaluator is not None and hasattr(self._evaluator, '_encode_composed_queries_local'):
            return self._evaluator._encode_composed_queries_local(queries)
        print_master("Warning: Falling back to empty query embeddings, check evaluator availability")
        return torch.empty(0, 512, device=self.device)

    def retrieve_top_k(self, top_k: int = 10, save_embeddings: bool = False) -> Dict[str, Any]:
        """Run retrieval for all queries and return top-k results."""
        print_master("Starting CIRR retrieval...")
        self.model.eval()

        use_dist = False
        if self.distributed and self._evaluator is not None:
            try:
                import torch.distributed as dist
                use_dist = dist.is_initialized() and dist.get_world_size() > 1 \
                           and hasattr(self._evaluator, '_encode_images_distributed') \
                           and hasattr(self._evaluator, '_encode_queries_distributed')
            except Exception:
                use_dist = False

        if use_dist:
            print_master("Encoding candidates (distributed)...")
            candidate_embeddings = self._evaluator._encode_images_distributed()
            print_master("Encoding queries (distributed)...")
            query_embeddings = self._evaluator._encode_queries_distributed()
        else:
            candidate_embeddings = self._encode_images(self.candidate_images)
            if candidate_embeddings.numel() == 0:
                print_master("Error: Candidate embeddings are empty, aborting retrieval")
                return {}
            query_embeddings = self._encode_composed_queries(self.test_data)
            if query_embeddings.numel() == 0:
                print_master("Error: Query embeddings are empty, aborting retrieval")
                return {}

        candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        print_master("Computing similarities...")
        similarities = torch.mm(query_embeddings, candidate_embeddings.t()).float()

        print_master("Excluding reference images...")
        image_to_idx = {img: idx for idx, img in enumerate(self.candidate_images)}
        for q_idx, q in enumerate(self.test_data):
            ref = q.get('reference')
            if ref in image_to_idx:
                similarities[q_idx, image_to_idx[ref]] = -float('inf')

        k = min(top_k, similarities.size(1))
        print_master(f"Retrieving top-{k} results per query...")
        _, topk_idx = torch.topk(similarities, k=k, dim=1, largest=True)

        results: Dict[str, Any] = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_path': getattr(self.model_args, 'checkpoint_path', 'unknown'),
                'model_backbone': getattr(self.model_args, 'model_backbone', 'unknown'),
                'total_queries': len(self.test_data),
                'total_candidates': len(self.candidate_images),
                'top_k': k,
                'batch_size': self.batch_size,
                'device': str(self.device),
                'distributed': bool(use_dist),
            },
            'queries': [],
            'candidate_images': self.candidate_images,
        }

        if save_embeddings:
            try:
                results['embeddings'] = {
                    'query_embeddings': query_embeddings.cpu().numpy().tolist(),
                    'candidate_embeddings': candidate_embeddings.cpu().numpy().tolist(),
                }
            except Exception as e:
                print_master(f"Warning: Failed to save embeddings, skipping ({e})")

        print_master("Building detailed results...")
        for q_idx, q in enumerate(self.test_data):
            indices = topk_idx[q_idx].tolist()
            scores = similarities[q_idx, indices].tolist()

            retrieval_results = []
            for rank, (ci, s) in enumerate(zip(indices, scores), start=1):
                retrieval_results.append({
                    'rank': rank,
                    'candidate_image': self.candidate_images[ci],
                    'similarity_score': float(s),
                    'candidate_index': int(ci),
                })

            qr: Dict[str, Any] = {
                'query_id': q_idx,
                'pairid': q.get('pairid', q_idx),
                'reference_image': q.get('reference'),
                'target_hard': q.get('target_hard'),
                'modification_text': q.get('caption'),
                'target_soft': q.get('target_soft', {}),
                'img_set': q.get('img_set', {}),
                'retrieval_results': retrieval_results,
            }

            tgt = q.get('target_hard')
            if tgt is not None:
                found = None
                for item in retrieval_results:
                    if item['candidate_image'] == tgt:
                        found = item['rank']
                        break
                qr['ground_truth'] = {
                    'target_hard': tgt,
                    'found_in_top_k': found is not None,
                    'rank_in_top_k': found,
                }

            results['queries'].append(qr)

        if results['queries'] and 'ground_truth' in results['queries'][0]:
            found_cnt = sum(1 for q in results['queries'] if q['ground_truth']['found_in_top_k'])
            acc_at_k = found_cnt / len(results['queries']) if results['queries'] else 0.0
            results['metadata']['accuracy_at_k'] = acc_at_k
            results['metadata']['found_in_top_k_count'] = found_cnt
            print_master(f"Accuracy@{k}: {acc_at_k:.4f} ({found_cnt}/{len(results['queries'])})")

        print_master("Retrieval complete!")
        return results


def setup_device(device_arg: str, distributed: bool = False) -> str:
    """Setup and return the appropriate device, with optional distributed init."""
    if distributed and ('RANK' in os.environ and 'WORLD_SIZE' in os.environ):
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = f"cuda:{local_rank}"
                print_master(f"Distributed initialized: rank {dist.get_rank()}/{dist.get_world_size()}")
                print_master(f"Using device: {device}")
                return device
            else:
                print_master("CUDA not available, falling back to CPU")
        except Exception as e:
            print_master(f"Distributed init failed: {e}, falling back to single GPU")

    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            try:
                print_master(f"Using CUDA device: {torch.cuda.get_device_name()}")
            except Exception:
                print_master("Using CUDA device")
        else:
            device = 'cpu'
            print_master("CUDA not available, using CPU")
    else:
        device = device_arg
        print_master(f"Using specified device: {device}")
    return device


def infer_model_name_from_path(model_path: str, quiet: bool = False) -> str:
    """Infer base model name from checkpoint path via naming conventions or config.json."""
    path_lower = model_path.lower()

    if re.search(r"qwen2(\.5|_5)?[-_]?vl", path_lower):
        is_qwen25 = bool(re.search(r"qwen2(\.5|_5)", path_lower))
        size = None
        if '2b' in path_lower:
            size = '2B'
        elif '7b' in path_lower:
            size = '7B'
        elif '32b' in path_lower:
            size = '32B'
        base = f"Qwen2.5-VL-{size or '7B'}-Instruct" if is_qwen25 else f"Qwen2-VL-{size or '7B'}-Instruct"
        model_name = f"Qwen/{base}"
        if not quiet:
            print_master(f"Inferred base model from path pattern: {model_name}")
        return model_name

    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            for key in ['_name_or_path', 'name_or_path', 'model_name', 'base_model_name']:
                if key in cfg and cfg[key]:
                    if not quiet:
                        print_master(f"Inferred base model from config: {cfg[key]}")
                    return cfg[key]
        except Exception as e:
            if not quiet:
                print_master(f"Warning: Failed to read config.json: {e}")

    default_name = 'Qwen/Qwen2-VL-2B-Instruct'
    if not quiet:
        print_master("Warning: Could not infer base model, using default")
    return default_name


def load_model_and_processor(
    retrieval_args: CIRRRetrievalArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
):
    """Load model and processor, aligned with eval_cirr.py logic."""
    print_master("=" * 60)
    print_master("LOADING MODEL AND PROCESSOR")
    print_master("=" * 60)

    base_model_name = None
    lora_mode = False

    adapter_config_path = os.path.join(retrieval_args.model_path, 'adapter_config.json')
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, 'r') as f:
                ad_cfg = json.load(f)
            if 'base_model_name_or_path' in ad_cfg:
                base_model_name = ad_cfg['base_model_name_or_path']
                lora_mode = True
                print_master(f"Detected LoRA adapter. Base model: {base_model_name}")
        except Exception as e:
            print_master(f"Failed reading adapter_config.json (will fallback to inference): {e}")

    if retrieval_args.base_model_name:
        base_model_name = retrieval_args.base_model_name
        print_master(f"Using provided base model name: {base_model_name}")

    if base_model_name is None:
        base_model_name = infer_model_name_from_path(retrieval_args.model_path, quiet=True)
        print_master(f"Inferred base model name: {base_model_name}")

    if model_args.model_name in [None, 'auto-infer']:
        model_args.model_name = base_model_name
        print_master(f"Final model_name: {model_args.model_name}")

    model_args.checkpoint_path = retrieval_args.model_path
    model_args.lora = lora_mode or getattr(model_args, 'lora', False)

    print_master("Aligning ModelArguments defaults with training config...")
    model_args.pooling = 'eos'
    model_args.normalize = True
    print_master(f"Set pooling={model_args.pooling}, normalize={model_args.normalize}")

    data_args.max_len = 512
    data_args.resize_max_pixels = 147456
    print_master(f"Set max_len={data_args.max_len}, resize_max_pixels={data_args.resize_max_pixels}")

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
        print_master(f"Backbone detection fallback ({e_det}): {model_args.model_backbone}")

    model = None
    if model_args.lora:
        print_master("Loading LoRA model (base + adapter)...")
        try:
            model = MMEBModel.load(model_args, is_trainable=False)
            model.eval()
            print_master("LoRA model loaded successfully")
        except Exception as e:
            print_master(f"LoRA model loading failed: {e}")
            raise
    else:
        print_master("Loading full model from local checkpoint (non-LoRA)...")
        try:
            model = MMEBModel.load(model_args, is_trainable=False)
            model.eval()
            print_master("Full model loaded successfully")
        except Exception as e:
            print_master(f"MMEBModel.load failed: {e}")
            print_master("Trying build + manual weight load fallback...")
            try:
                original_ckpt = model_args.checkpoint_path
                model_args.checkpoint_path = None
                model = MMEBModel.build(model_args)

                weight_file = None
                if os.path.isdir(retrieval_args.model_path):
                    for f in ["pytorch_model.bin", "model.safetensors", "model.bin"]:
                        fp = os.path.join(retrieval_args.model_path, f)
                        if os.path.exists(fp):
                            weight_file = fp
                            break
                if weight_file is None:
                    raise ValueError(f"No weight files found in {retrieval_args.model_path}")

                print_master(f"Loading weights from {weight_file}...")
                if weight_file.endswith('.safetensors'):
                    from safetensors import safe_open
                    sd = {}
                    with safe_open(weight_file, framework='pt', device='cpu') as sf:
                        for k in sf.keys():
                            sd[k] = sf.get_tensor(k)
                else:
                    sd = torch.load(weight_file, map_location='cpu')

                model.load_state_dict(sd, strict=False)
                print_master("Weights loaded into built model")
                model_args.checkpoint_path = original_ckpt
            except Exception as e2:
                print_master(f"Fallback loading failed: {e2}")
                raise

    print_master("Loading processor...")
    try:
        processor = load_processor(model_args, data_args)
        print_master("Processor loaded successfully")
    except Exception as e:
        print_master(f"Failed to load processor: {e}")
        raise

    setattr(model, 'processor', processor)
    print_master("=" * 60)
    return model, processor


def generate_output_filename(retrieval_args: CIRRRetrievalArguments) -> str:
    if retrieval_args.output_file:
        return retrieval_args.output_file

    project_root = os.path.dirname(__file__)
    base_dir = os.path.join(project_root, 'retrieval_results')

    model_path = retrieval_args.model_path.rstrip('/')
    model_name = os.path.basename(model_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{model_name}_{timestamp}")
    filename = f"cirr_retrieval_top{retrieval_args.top_k}.json"
    return os.path.join(run_dir, filename)


def save_retrieval_results(results: Dict[str, Any], output_file: str):
    try:
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print_master(f"Retrieval results saved to: {output_file}")
    except Exception as e:
        print_master(f"Failed to save retrieval results: {e}")
        print_master(f"Traceback: {traceback.format_exc()}")


def main():
    parser = HfArgumentParser((CIRRRetrievalArguments, ModelArguments, DataArguments))

    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        retrieval_args, model_args, data_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        retrieval_args, model_args, data_args = parser.parse_args_into_dataclasses()

    if not retrieval_args.model_path or not os.path.exists(retrieval_args.model_path):
        raise ValueError(f"Invalid model path: {retrieval_args.model_path}")

    device = setup_device(retrieval_args.device, retrieval_args.distributed)

    model, processor = load_model_and_processor(retrieval_args, model_args, data_args)
    model = model.to(device)
    print_master(f"Model moved to device: {device}")

    retriever = CIRRRetriever(
        model=model,
        processor=processor,
        data_args=data_args,
        model_args=model_args,
        device=device,
        batch_size=retrieval_args.batch_size,
        cirr_data_dir=retrieval_args.cirr_data_dir,
        cirr_image_dir=retrieval_args.cirr_image_dir,
        distributed=retrieval_args.distributed,
    )

    results = retriever.retrieve_top_k(top_k=retrieval_args.top_k, save_embeddings=retrieval_args.save_embeddings)

    is_main = True
    if retrieval_args.distributed:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                is_main = (dist.get_rank() == 0)
        except Exception:
            is_main = True

    if results and is_main:
        output_file = generate_output_filename(retrieval_args)
        save_retrieval_results(results, output_file)
        print_master("CIRR retrieval complete!")

    if retrieval_args.distributed:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except Exception:
            pass

    return results if results else {}


if __name__ == "__main__":
    main()

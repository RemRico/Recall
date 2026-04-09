#!/usr/bin/env python3
"""
CIRR Test Submission Script

Generates the two JSON files required by the CIRR evaluation server:
  * recall_submission_{name}.json        (global Top-50)
  * recall_subset_submission_{name}.json (group Top-3)

Reuses the unified model loading and encoding pipeline from eval_cirr.py
to ensure consistency between training, evaluation, and submission.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F
from transformers import HfArgumentParser

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.arguments import ModelArguments, DataArguments
from src.utils import print_master
from src.evaluation.cirr_evaluator import CIRREvaluator

from eval_cirr import setup_device as eval_setup_device
from eval_cirr import load_model_and_processor as eval_load_model_and_processor


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class CIRRSubmissionArguments:
    """CIRR test submission arguments"""
    model_path: str = field(metadata={"help": "Path to trained model checkpoint (checkpoint-xxx or LoRA directory)"})
    base_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Base model name (e.g., Qwen/Qwen2-VL-2B-Instruct). Auto-inferred if not provided"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Encoding batch size (default: 8)"}
    )
    device: str = field(
        default="auto",
        metadata={"help": "Device: auto/cuda/cuda:0/cpu etc."}
    )
    distributed: bool = field(
        default=False,
        metadata={"help": "Whether to use distributed encoding (requires torchrun)"}
    )
    submission_name: str = field(
        default="submission",
        metadata={"help": "Identifier in output filenames, e.g. my_model_test"}
    )
    output_dir: str = field(
        default="./submission/CIRR",
        metadata={"help": "Output directory for submission files"}
    )
    cirr_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "CIRR data root directory (containing captions/ and image_splits/)"}
    )
    cirr_image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "CIRR image root directory (containing train/ dev/ test1/ subdirs)"}
    )
    resize_max_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "Override max pixels for image resizing (default: use training config)"}
    )


def _resolve_cirr_paths(evaluator: CIRREvaluator,
                        cirr_data_dir: Optional[str],
                        cirr_image_dir: Optional[str]) -> Tuple[str, str]:
    """Resolve CIRR data/image root directories, preferring explicit arguments."""
    data_dir = cirr_data_dir if cirr_data_dir else getattr(evaluator, 'data_dir', None)
    image_dir = cirr_image_dir if cirr_image_dir else getattr(evaluator, 'image_base_dir', None)

    if not data_dir or not image_dir:
        raise ValueError(
            "CIRR data paths must be provided via --cirr_data_dir and --cirr_image_dir. "
            "See README for dataset download instructions."
        )

    print_master(f"CIRR data_dir: {data_dir}")
    print_master(f"CIRR image_dir: {image_dir}")
    return data_dir, image_dir


def _try_read_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_cirr_test1(data_dir: str) -> Tuple[List[Dict], List[str], Dict[str, str]]:
    """Load CIRR test1 queries and candidate image mapping."""
    captions_candidates = [
        os.path.join(data_dir, 'captions', 'cap.rc2.test1.json'),
        os.path.join(data_dir, 'captions', 'cap.test1.json'),
    ]
    splits_candidates = [
        os.path.join(data_dir, 'image_splits', 'split.rc2.test1.json'),
        os.path.join(data_dir, 'image_splits', 'split.test1.json'),
    ]

    captions_file = next((p for p in captions_candidates if os.path.exists(p)), None)
    splits_file = next((p for p in splits_candidates if os.path.exists(p)), None)

    if captions_file is None or splits_file is None:
        val_cap = os.path.join(data_dir, 'captions', 'cap.rc2.val.json')
        val_split = os.path.join(data_dir, 'image_splits', 'split.rc2.val.json')
        print_master("Warning: test1 files not found, falling back to validation set (for debugging only)")
        if not (os.path.exists(val_cap) and os.path.exists(val_split)):
            raise FileNotFoundError("Neither test1 nor validation files found under CIRR data dir")
        captions_file, splits_file = val_cap, val_split

    test_queries = _try_read_json(captions_file)
    image_splits = _try_read_json(splits_file)
    candidate_images = list(image_splits.keys())

    print_master(f"Loaded test queries: {len(test_queries)} from {captions_file}")
    print_master(f"Loaded candidate images: {len(candidate_images)} from {splits_file}")
    return test_queries, candidate_images, image_splits


def _encode_all_embeddings(evaluator: CIRREvaluator,
                           test_queries: List[Dict],
                           candidate_images: List[str],
                           distributed: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode candidates and queries, with optional distributed support. Returns CPU tensors."""
    evaluator.test_data = test_queries
    evaluator.candidate_images = candidate_images

    if distributed:
        import torch.distributed as dist
        if not (dist.is_initialized() and dist.get_world_size() > 1):
            print_master("Distributed not initialized, falling back to single GPU encoding")
            distributed = False

    if distributed:
        print_master("Encoding candidates (distributed)...")
        cand_emb = evaluator._encode_images_distributed()
        print_master("Encoding queries (distributed)...")
        qry_emb = evaluator._encode_queries_distributed()
    else:
        print_master("Encoding candidates (single GPU)...")
        cand_emb = evaluator._encode_images_local(candidate_images)
        print_master("Encoding queries (single GPU)...")
        qry_emb = evaluator._encode_composed_queries_local(test_queries)

    return cand_emb.cpu(), qry_emb.cpu()


def _build_submissions(test_queries: List[Dict],
                       candidate_images: List[str],
                       candidate_emb: torch.Tensor,
                       query_emb: torch.Tensor) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build submission dicts from embeddings.
    Returns (pairid_to_predictions, pairid_to_group_predictions).
    """
    candidate_emb = F.normalize(candidate_emb, p=2, dim=1).float()
    query_emb = F.normalize(query_emb, p=2, dim=1).float()

    sims = torch.mm(query_emb, candidate_emb.t())

    image_to_idx = {name: i for i, name in enumerate(candidate_images)}
    for qi, q in enumerate(test_queries):
        ref = q.get('reference')
        if ref in image_to_idx:
            sims[qi, image_to_idx[ref]] = -float('inf')

    topK = min(50, sims.size(1))
    topk_scores, topk_indices = torch.topk(sims, k=topK, dim=1, largest=True)

    pairid_to_predictions: Dict[str, List[str]] = {}
    pairid_to_group_predictions: Dict[str, List[str]] = {}

    for qi, q in enumerate(test_queries):
        pairid = str(int(q.get('pairid', qi)))
        indices = topk_indices[qi].tolist()
        names = [candidate_images[idx] for idx in indices]
        pairid_to_predictions[pairid] = names

        # Group (subset) Top-3: rank within the query's assigned group members
        img_set = q.get('img_set', {})
        members = img_set.get('members', []) if isinstance(img_set, dict) else (img_set or [])
        ref = q.get('reference')

        group_names: List[str] = []
        if members:
            group_indices: List[int] = []
            ordered_members: List[str] = []
            for member in members:
                if member == ref:
                    continue
                if member in image_to_idx:
                    group_indices.append(image_to_idx[member])
                    ordered_members.append(member)

            if group_indices:
                group_scores = sims[qi, group_indices]
                top_g = min(3, len(group_indices))
                top_indices = torch.topk(group_scores, k=top_g, dim=0, largest=True).indices.tolist()
                group_names = [ordered_members[idx] for idx in top_indices]

        pairid_to_group_predictions[pairid] = group_names

    return pairid_to_predictions, pairid_to_group_predictions


def _save_submissions(output_dir: str,
                      submission_name: str,
                      pred_global: Dict[str, List[str]],
                      pred_group: Dict[str, List[str]]):
    os.makedirs(output_dir, exist_ok=True)

    sub_global = {"version": "rc2", "metric": "recall"}
    sub_global.update(pred_global)

    sub_group = {"version": "rc2", "metric": "recall_subset"}
    sub_group.update(pred_group)

    g_path = os.path.join(output_dir, f"recall_submission_{submission_name}.json")
    s_path = os.path.join(output_dir, f"recall_subset_submission_{submission_name}.json")

    with open(g_path, 'w', encoding='utf-8') as f:
        json.dump(sub_global, f, ensure_ascii=False, indent=2)
    with open(s_path, 'w', encoding='utf-8') as f:
        json.dump(sub_group, f, ensure_ascii=False, indent=2)

    print_master(f"Saved: {g_path}")
    print_master(f"Saved: {s_path}")


def main():
    parser = HfArgumentParser(CIRRSubmissionArguments)
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        sub_args = parser.parse_json_file(json_file=sys.argv[1])[0]
    else:
        sub_args = parser.parse_args_into_dataclasses()[0]

    model_args = ModelArguments(model_name="auto-infer")
    data_args = DataArguments()
    if sub_args.resize_max_pixels is not None:
        data_args.resize_max_pixels = sub_args.resize_max_pixels

    device = eval_setup_device(sub_args.device, sub_args.distributed)

    model, processor = eval_load_model_and_processor(sub_args, model_args, data_args)
    model = model.to(device)
    setattr(model, 'processor', processor)
    print_master(f"Model moved to device: {device}")

    evaluator = CIRREvaluator(
        model=model,
        processor=processor,
        data_args=data_args,
        model_args=model_args,
        device=device,
        batch_size=sub_args.batch_size,
    )

    data_dir, image_dir = _resolve_cirr_paths(evaluator, sub_args.cirr_data_dir, sub_args.cirr_image_dir)
    test_queries, candidate_images, image_splits = _load_cirr_test1(data_dir)
    evaluator.image_base_dir = image_dir
    evaluator.image_splits = image_splits

    cand_emb, qry_emb = _encode_all_embeddings(
        evaluator,
        test_queries=test_queries,
        candidate_images=candidate_images,
        distributed=sub_args.distributed,
    )

    pred_global, pred_group = _build_submissions(
        test_queries=test_queries,
        candidate_images=candidate_images,
        candidate_emb=cand_emb,
        query_emb=qry_emb,
    )

    is_main_process = True
    if sub_args.distributed:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                is_main_process = (dist.get_rank() == 0)
        except Exception:
            is_main_process = True

    if is_main_process:
        _save_submissions(sub_args.output_dir, sub_args.submission_name, pred_global, pred_group)
        print_master("CIRR test submission files generated successfully!")

    if sub_args.distributed:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()

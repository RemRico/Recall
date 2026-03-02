#!/usr/bin/env python3
"""
CIRR 测试提交文件生成脚本（基于本项目统一编码/模型加载流程）
- 读取 CIRR test1 的查询与候选图像
- 复用本项目的处理器与编码逻辑（与训练/评估一致）
- 生成官网评测所需的两个 JSON：
  * recall_submission_{name}.json        （全集 Top-50）
  * recall_subset_submission_{name}.json （组内 Top-3）

使用方式示例请见文件末尾注释。
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

# 加入 src 路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.arguments import ModelArguments, DataArguments
from src.utils import print_master
from src.evaluation.cirr_evaluator import CIRREvaluator

# 直接复用现有评估脚本里的设备/加载逻辑
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
    """CIRR 测试提交参数"""
    model_path: str = field(metadata={"help": "已训练模型的路径（checkpoint-xxx 或 LoRA 目录）"})
    base_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "基础模型名称（如 Qwen/Qwen2-VL-2B-Instruct）。不提供则自动推断"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "编码 batch size（默认 8）"}
    )
    device: str = field(
        default="auto",
        metadata={"help": "设备：auto/cuda/cuda:0/cpu 等"}
    )
    distributed: bool = field(
        default=False,
        metadata={"help": "是否使用分布式（torch.distributed）进行加速，仅在已启动分布式环境时生效"}
    )
    submission_name: str = field(
        default="submission",
        metadata={"help": "输出文件名中的标识，例如 my_model_test"}
    )
    output_dir: str = field(
        default="./submission/CIRR",
        metadata={"help": "提交文件输出目录"}
    )
    cirr_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "CIRR 数据根目录（包含 captions/ 与 image_splits/ 的 cirr 子目录）"}
    )
    cirr_image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "CIRR 图像根目录（包含 train/ dev/ test1/ 等子目录）"}
    )
    resize_max_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "在生成提交时使用的最大像素数（覆盖训练默认，留空则沿用训练配置）"}
    )


def _resolve_cirr_paths(evaluator: CIRREvaluator,
                        cirr_data_dir: Optional[str],
                        cirr_image_dir: Optional[str]) -> Tuple[str, str]:
    """决定使用的 CIRR data/image 根目录。
    优先使用传入参数，否则使用 evaluator 配置的默认路径。
    """
    data_dir = cirr_data_dir if cirr_data_dir else getattr(evaluator, 'data_dir', None)
    image_dir = cirr_image_dir if cirr_image_dir else getattr(evaluator, 'image_base_dir', None)

    if not data_dir or not image_dir:
        # 回落到 evaluator 的默认配置（会在其 __init__ 里写入）
        data_dir = getattr(evaluator, 'data_dir', '/home/guohaiyun/yty_data/CIRR/cirr')
        image_dir = getattr(evaluator, 'image_base_dir', '/home/guohaiyun/yty_data/CIRR')

    print_master(f"CIRR data_dir: {data_dir}")
    print_master(f"CIRR image_dir: {image_dir}")
    return data_dir, image_dir


def _try_read_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_cirr_test1(data_dir: str) -> Tuple[List[Dict], List[str], Dict[str, str]]:
    """读取 CIRR test1 查询与候选图像映射。

    返回:
      - test_queries: List[Dict]
      - candidate_images: List[str]
      - image_splits: Dict[image_name -> relative_path]
    """
    # 优先使用 rc2 test1 文件名
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
        # 兜底：使用验证集，便于联调
        val_cap = os.path.join(data_dir, 'captions', 'cap.rc2.val.json')
        val_split = os.path.join(data_dir, 'image_splits', 'split.rc2.val.json')
        print_master("Warning: test1 files not found, falling back to validation set (for debugging only)")
        if not (os.path.exists(val_cap) and os.path.exists(val_split)):
            raise FileNotFoundError("Neither test1 nor validation files found under CIRR data dir")
        captions_file, splits_file = val_cap, val_split

    test_queries = _try_read_json(captions_file)
    image_splits = _try_read_json(splits_file)

    # 候选图像为 splits 的所有 key
    candidate_images = list(image_splits.keys())

    print_master(f"Loaded test queries: {len(test_queries)} from {captions_file}")
    print_master(f"Loaded candidate images: {len(candidate_images)} from {splits_file}")
    return test_queries, candidate_images, image_splits


def _encode_all_embeddings(evaluator: CIRREvaluator,
                           test_queries: List[Dict],
                           candidate_images: List[str],
                           distributed: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """编码候选与查询，支持分布式/单卡。返回 CPU 张量。"""
    # 将数据注入 evaluator，复用其本地编码/分布式编码函数
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
    """根据嵌入构建两个提交字典。
    返回 (pairid_to_predictions, pairid_to_group_predictions)
    """
    # 归一化并转 float32，避免 numpy/bfloat16 兼容问题
    candidate_emb = F.normalize(candidate_emb, p=2, dim=1).float()
    query_emb = F.normalize(query_emb, p=2, dim=1).float()

    # 相似度 [Q, C]
    sims = torch.mm(query_emb, candidate_emb.t())

    # 建立 name->idx 映射，并排除参考图像
    image_to_idx = {name: i for i, name in enumerate(candidate_images)}
    for qi, q in enumerate(test_queries):
        ref = q.get('reference')
        if ref in image_to_idx:
            sims[qi, image_to_idx[ref]] = -float('inf')

    # 计算 Top-50 全局
    topK = min(50, sims.size(1))
    topk_scores, topk_indices = torch.topk(sims, k=topK, dim=1, largest=True)

    pairid_to_predictions: Dict[str, List[str]] = {}
    pairid_to_group_predictions: Dict[str, List[str]] = {}

    for qi, q in enumerate(test_queries):
        pairid = str(int(q.get('pairid', qi)))
        # 全局 Top-50 名称
        indices = topk_indices[qi].tolist()
        names = [candidate_images[idx] for idx in indices]
        pairid_to_predictions[pairid] = names

        # 组内 Top-3：直接在所属分组内部按相似度排序，确保提交的列表始终覆盖组成员
        img_set = q.get('img_set', {})
        members = img_set.get('members', []) if isinstance(img_set, dict) else (img_set or [])
        ref = q.get('reference')

        group_names: List[str] = []
        if members:
            group_indices: List[int] = []
            ordered_members: List[str] = []
            for member in members:
                if member == ref:
                    continue  # 官方评测不应包含引用图
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
    # 解析参数：仅解析本脚本的提交参数，避免强制要求 --model_name
    parser = HfArgumentParser(CIRRSubmissionArguments)
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        sub_args = parser.parse_json_file(json_file=sys.argv[1])[0]
    else:
        sub_args = parser.parse_args_into_dataclasses()[0]

    # 构造默认的 ModelArguments 和 DataArguments，model_name 设为 auto-infer，后续由 eval_cirr 的逻辑自动推断
    model_args = ModelArguments(model_name="auto-infer")
    data_args = DataArguments()
    if sub_args.resize_max_pixels is not None:
        data_args.resize_max_pixels = sub_args.resize_max_pixels

    # 设置设备（支持分布式 torchrun 环境）
    device = eval_setup_device(sub_args.device, sub_args.distributed)

    # 加载模型与处理器（完全复用 eval_cirr 的逻辑，支持 LoRA 等）
    model, processor = eval_load_model_and_processor(sub_args, model_args, data_args)
    model = model.to(device)
    setattr(model, 'processor', processor)
    print_master(f"Model moved to device: {device}")

    # 构建 evaluator，用于复用其编码函数（指令/预处理与训练一致）
    evaluator = CIRREvaluator(
        model=model,
        processor=processor,
        data_args=data_args,
        model_args=model_args,
        device=device,
        batch_size=sub_args.batch_size,
    )

    # 使用 test1 文件，覆盖 evaluator 默认的验证集文件
    data_dir, image_dir = _resolve_cirr_paths(evaluator, sub_args.cirr_data_dir, sub_args.cirr_image_dir)
    test_queries, candidate_images, image_splits = _load_cirr_test1(data_dir)
    evaluator.image_base_dir = image_dir
    evaluator.image_splits = image_splits  # 供 _load_image 使用

    # 编码
    cand_emb, qry_emb = _encode_all_embeddings(
        evaluator,
        test_queries=test_queries,
        candidate_images=candidate_images,
        distributed=sub_args.distributed,
    )

    # 相似度与提交字典
    pred_global, pred_group = _build_submissions(
        test_queries=test_queries,
        candidate_images=candidate_images,
        candidate_emb=cand_emb,
        query_emb=qry_emb,
    )

    # 仅主进程写文件（分布式情况）
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
        print_master("✅ CIRR 测试提交文件生成完成！")

    # 结束分布式
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

"""
使用示例：
1) 单卡
python cirr_test_submission.py \
  --model_path /path/to/your/checkpoint-xxxx \
  --base_model_name "Qwen/Qwen2-VL-2B-Instruct" \
  --batch_size 8 \
  --device cuda \
  --submission_name mymodel_cirr_test \
  --output_dir ./submission/CIRR

2) 分布式（例如 8 卡）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 cirr_test_submission.py \
  --model_path experiments/IterativeCIRR_qwen2_5vl_7b_20260126_193416_r64/training_iter_0/checkpoint-3500 \
  --batch_size 8 \
  --device cuda \
  --distributed True \
  --submission_name qwen2.5vl_cirr_base_ckpt3500_r64\
  --output_dir ./submission/CIRR
"""

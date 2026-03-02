# -*- coding: utf-8 -*-
"""
FashionIQ dataset implementation atop the IterativeRetrievalDataset base class.
For now this mirrors your original minimal loader. We will expand the sample
builders in a later step to match CIRR's retrieval/augmentation pipeline.
"""

import json
import os
from typing import Dict, Any

from .base_iterative_dataset import IterativeRetrievalDataset
from src.utils import print_rank
from ...model.processor import process_input_text


class IterativeFashionIQDataset(IterativeRetrievalDataset):
    """
    Iterative FashionIQ Dataset — minimal loading (as in your original code).
    Notes:
      - FashionIQ has categories: dress, shirt, toptee
      - Captions usually live under {data_dir}/captions/cap.{category}.train.json
      - Images under {data_dir}/images
    """

    def _load_data(self) -> None:
        print_rank("Loading FashionIQ dataset...")

        data_dir = self.dataset_config.get("data_dir", "./data/FashionIQ")
        self.image_base_dir = os.path.join(data_dir, "images")

        categories = ["dress", "shirt", "toptee"]
        all_annotations = []

        for cat in categories:
            cap_file = os.path.join(data_dir, "captions", f"cap.{cat}.train.json")
            if os.path.exists(cap_file):
                with open(cap_file, "r") as f:
                    items = json.load(f)
                for it in items:
                    it["category"] = cat
                all_annotations.extend(items)

        self.annotations = all_annotations
        self.image_splits = {}  # FashionIQ typically addresses files by name directly
        print_rank(f"Loaded {len(self.annotations)} FashionIQ training samples")

        self.retrieval_candidates = []  # will be built by candidate builder later (optional)
        self.num_rows = len(self.annotations) + len(self.augmented_samples)

    def _get_original_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.annotations[idx]
        
        # FashionIQ数据格式：candidate为候选图片，target为目标图片，captions为修改文本列表
        ref_image_path = f"{sample['candidate']}.jpg"
        target_image_path = f"{sample['target']}.jpg"
        
        # FashionIQ通常有多个captions，这里取第一个
        modification_text = sample['captions'][0] if sample['captions'] else ""
        
        model_backbone = getattr(self.model_args, "model_backbone", "qwen2_vl")
        
        # Query = (reference image + modification text)
        query_text = process_input_text(
            instruction=f"Modify this image with <{modification_text}>\nRepresent the modified image in one word:",
            model_backbone=model_backbone,
            text="",
            add_image_token=True,
        )
        # Positive = target image (empty text + image token)
        pos_text = process_input_text(
            instruction="Represent the given image in one word:",
            model_backbone=model_backbone,
            text="",
            add_image_token=True,
        )
        # Negative placeholder = reference image again (same as CIRR baseline)
        neg_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text="",
            add_image_token=True,
        )
        
        ref_full_path = self._get_full_image_path(ref_image_path)
        tgt_full_path = self._get_full_image_path(target_image_path)
        original_mod_text = modification_text.strip() if isinstance(modification_text, str) else str(modification_text)

        return {
            "query_text": query_text,
            "query_image": self._load_image(ref_image_path),
            "pos_text": pos_text,
            "pos_image": self._load_image(target_image_path),
            "neg_text": neg_text,
            "neg_image": self._load_image(ref_image_path),
            "global_dataset_name": "FashionIQ",
            "reference_image": ref_full_path,  # normalized for grouping
            "reference_id": self._get_reference_id(ref_full_path),
            "is_augmented": False,
            "category": sample.get("category", ""),  # FashionIQ特有的类别信息
            "original_mod_text": original_mod_text,
        }

    def _get_augmented_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.augmented_samples[idx]
        
        model_backbone = getattr(self.model_args, "model_backbone", "qwen2_vl")
        
        query_text = process_input_text(
            instruction=f"Modify this image with <{sample['modification_text']}>\nRepresent the modified image in one word:",
            model_backbone=model_backbone,
            text="",
            add_image_token=True,
        )
        pos_text = process_input_text(
            instruction="Represent the given image in one word:",
            model_backbone=model_backbone,
            text="",
            add_image_token=True,
        )
        neg_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text="",
            add_image_token=True,
        )
        
        ref_full_path = self._get_full_image_path(sample["reference_image"])

        original_mod_text = sample.get("original_mod_text", "")
        if isinstance(original_mod_text, str):
            original_mod_text = original_mod_text.strip()

        return {
            "query_text": query_text,
            "query_image": self._load_image(sample["reference_image"]),
            "pos_text": pos_text,
            "pos_image": self._load_image(sample["target_image"]),
            "neg_text": neg_text,
            "neg_image": self._load_image(sample["reference_image"]),
            "global_dataset_name": "FashionIQ",
            "is_augmented": True,
            "original_mod_text": original_mod_text,
            "reference_image": ref_full_path,
            "reference_id": self._get_reference_id(ref_full_path),
            "category": sample.get("category", ""),  # FashionIQ特有的类别信息
        }

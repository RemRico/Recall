# -*- coding: utf-8 -*-
"""
CIRR dataset implementation atop the IterativeRetrievalDataset base class.
This file only handles CIRR-specific loading and sample materialization.
"""

import json
import os
from typing import Dict, Any

from .base_iterative_dataset import IterativeRetrievalDataset
from src.utils import print_rank
from ...model.processor import process_input_text
from ...data.dataset.base_pair_dataset import AutoPairDataset


class IterativeCIRRDataset(IterativeRetrievalDataset):
    """
    Iterative CIRR Dataset (data loading + sample building).
    Retrieval/mining/augmentation are handled by dedicated modules outside.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_original_data = True

    def set_use_original_data(self, use_original: bool):
        """Set whether to use original data or only augmented data."""
        self.use_original_data = use_original
        # Recalculate total number of rows
        self.num_rows = self.__len__()
        print_rank(f"Dataset config updated: use_original_data = {self.use_original_data}. Total samples = {self.num_rows}")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        if not self.use_original_data:
            return len(self.augmented_samples)
        return len(self.annotations) + len(self.augmented_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        if not self.use_original_data:
            # Only use augmented samples
            if idx >= len(self.augmented_samples):
                raise IndexError("Index out of range when only using augmented samples")
            return self._get_augmented_sample(idx)

        # Use both original and augmented samples
        if idx < len(self.annotations):
            return self._get_original_sample(idx)
        else:
            return self._get_augmented_sample(idx - len(self.annotations))

    # ---------- Dataset-specific loading ----------
    def _load_data(self) -> None:
        print_rank("Loading CIRR dataset...")

        data_dir = self.dataset_config.get("data_dir", "./data/CIRR")
        self.image_base_dir = self.dataset_config.get("image_base_dir", data_dir)

        captions_file = self.dataset_config.get("captions_file")
        image_splits_file = self.dataset_config.get("image_splits_file")

        if not captions_file or not image_splits_file:
            raise ValueError("CIRR requires 'captions_file' and 'image_splits_file' in dataset_config.")

        with open(captions_file, "r") as f:
            self.annotations = json.load(f)

        with open(image_splits_file, "r") as f:
            self.image_splits = json.load(f)

        print_rank(f"Loaded {len(self.annotations)} CIRR training samples")
        print_rank(f"Loaded {len(self.image_splits)} image path mappings")

        # optional: retrieval candidates will be built by retrieval.candidate_builder later
        self.retrieval_candidates = []

        self.num_rows = len(self.annotations) + len(self.augmented_samples)

    # ---------- Sample builders (same behavior as your original file) ----------
    def _get_original_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.annotations[idx]

        # Map IDs to relative paths if available
        ref_image_path = self.image_splits.get(sample["reference"], sample["reference"])
        target_image_path = self.image_splits.get(sample["target_hard"], sample["target_hard"])

        model_backbone = getattr(self.model_args, "model_backbone", "qwen2_vl")

        # Query = (reference image + modification text)
        query_text = process_input_text(
            instruction=f"Modify this image with <{sample['caption']}>\nRepresent the modified image in one word:",
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
        # Negative placeholder = reference image again (same as original baseline)
        neg_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text="",
            add_image_token=True,
        )

        ref_full_path = self._get_full_image_path(ref_image_path)
        original_mod_text = sample.get("caption", "")
        if isinstance(original_mod_text, str):
            original_mod_text = original_mod_text.strip()
        return {
            "query_text": query_text,
            "query_image": self._load_image(ref_image_path),
            "pos_text": pos_text,
            "pos_image": self._load_image(target_image_path),
            "neg_text": neg_text,
            "neg_image": self._load_image(ref_image_path),
            "global_dataset_name": "CIRR",
            # 关键：统一规范化 reference_image，确保与增强样本一致，从而能被 sampler 分到同一组
            "reference_image": ref_full_path,
            "reference_id": self._get_reference_id(ref_full_path),
            "original_mod_text": original_mod_text,
            "is_augmented": False,
        }

    def _get_augmented_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.augmented_samples[idx]

        model_backbone = getattr(self.model_args, "model_backbone", "qwen2_vl")

        # 兼容不同的增广样本字段：优先使用 caption，否则回退到 modification_text
        mod_text = sample.get("caption", sample.get("modification_text", ""))

        query_text = process_input_text(
            instruction=f"Modify this image with <{mod_text}>\nRepresent the modified image in one word:",
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

        ref_path = self._get_full_image_path(sample["reference_image"])  # 规范化

        original_mod_text = sample.get("original_mod_text", "")
        if isinstance(original_mod_text, str):
            original_mod_text = original_mod_text.strip()

        return {
            "query_text": query_text,
            "query_image": self._load_image(ref_path),
            "pos_text": pos_text,
            "pos_image": self._load_image(sample["target_image"]),
            "neg_text": neg_text,
            "neg_image": self._load_image(ref_path),
            "global_dataset_name": "CIRR",
            "is_augmented": True,
            "original_mod_text": original_mod_text,
            # 关键：与原始样本相同规则的规范化 reference_image，确保分组一致
            "reference_image": ref_path,
            "reference_id": self._get_reference_id(ref_path),
        }


AutoPairDataset.registry["IterativeCIRRDataset"] = IterativeCIRRDataset

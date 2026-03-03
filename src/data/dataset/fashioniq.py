# -*- coding: utf-8 -*-
"""
FashionIQ dataset implementation atop the IterativeRetrievalDataset base class.
For now this mirrors your original minimal loader. We will expand the sample
builders in a later step to match CIRR's retrieval/augmentation pipeline.
"""

import json
import os
from typing import Dict, Any, List
from glob import glob

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
        dataset_split = self.dataset_config.get("dataset_split", "train")  # train/val/test

        # Step 1: Load split files to get valid images for this split
        print_rank(f"Loading FashionIQ split files for split: {dataset_split}")
        valid_image_ids = self._load_split_files(data_dir, dataset_split)
        print_rank(f"Loaded {len(valid_image_ids)} valid image IDs for {dataset_split} split")

        # Step 2: Build image_splits ONLY for images in this split
        print_rank(f"Building FashionIQ image_splits (only {dataset_split} split images)...")
        self.image_splits = self._build_image_splits(valid_image_ids=valid_image_ids)
        print_rank(f"Built image_splits with {self._unique_image_count} unique images ({len(self.image_splits)} total keys for lookup)")

        # Step 3: Load annotations from caption files
        categories = ["dress", "shirt", "toptee"]
        all_annotations = []

        for cat in categories:
            cap_file = os.path.join(data_dir, "captions", f"cap.{cat}.{dataset_split}.json")
            if os.path.exists(cap_file):
                with open(cap_file, "r") as f:
                    items = json.load(f)
                for it in items:
                    it["category"] = cat
                all_annotations.extend(items)
            else:
                print_rank(f"Warning: Caption file not found: {cap_file}")

        # Step 4: Normalize annotations to CIRR-compatible format
        print_rank("Normalizing FashionIQ annotations to CIRR format...")
        self.annotations = self._normalize_annotations(all_annotations)
        print_rank(f"Loaded and normalized {len(self.annotations)} FashionIQ training samples")

        # Step 5: Build retrieval candidates from the filtered image_splits
        self.retrieval_candidates = self._build_retrieval_candidates(valid_image_ids)
        print_rank(f"Built retrieval candidates: {len(self.retrieval_candidates)} images for {dataset_split} split")
        
        self.num_rows = len(self.annotations) + len(self.augmented_samples)

    def _load_split_files(self, data_dir: str, split: str) -> set:
        """
        Load image split files to get valid image IDs for the given split.
        
        Args:
            data_dir: Base data directory
            split: Dataset split (train/val/test)
            
        Returns:
            Set of valid image IDs for this split
        """
        valid_ids = set()
        categories = ["dress", "shirt", "toptee"]
        
        for cat in categories:
            split_file = os.path.join(data_dir, "image_splits", f"split.{cat}.{split}.json")
            if os.path.exists(split_file):
                with open(split_file, "r") as f:
                    cat_ids = json.load(f)
                    valid_ids.update(cat_ids)
                    print_rank(f"  • Loaded {len(cat_ids)} image IDs from {cat} {split} split")
            else:
                print_rank(f"Warning: Split file not found: {split_file}")
        
        return valid_ids

    def _build_retrieval_candidates(self, valid_image_ids: set) -> List[str]:
        """
        Build retrieval candidates list filtered by valid image IDs from split files.
        
        Args:
            valid_image_ids: Set of valid image IDs for this split
            
        Returns:
            List of image paths that are valid for this split
        """
        candidates = []
        all_image_paths = set(self.image_splits.values())
        
        for img_path in all_image_paths:
            # Extract image ID from path (handle both flat and category structures)
            # Examples: "B007FO5H1Q.png" -> "B007FO5H1Q"
            #           "toptee/B007FO5H1Q.png" -> "B007FO5H1Q"
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            if img_name in valid_image_ids:
                candidates.append(img_path)
        
        print_rank(f"  • Filtered {len(candidates)} valid candidates from {len(all_image_paths)} total images")
        return candidates

    def _build_image_splits(self, valid_image_ids: set = None) -> Dict[str, str]:
        """
        Build image_splits mapping: image_id -> relative_path
        Supports both flat structure (images/*.png) and category structure (images/dress/*.png)
        
        Args:
            valid_image_ids: Optional set of valid image IDs to filter by. If provided,
                           only images in this set will be included in image_splits.
        """
        image_splits = {}
        unique_images = set()  # Track unique image files for accurate counting
        
        # Check if images are organized by category
        categories = ["dress", "shirt", "toptee"]
        has_category_dirs = all(
            os.path.isdir(os.path.join(self.image_base_dir, cat)) 
            for cat in categories
        )
        
        if has_category_dirs:
            print_rank("Detected category-based image organization")
            # Category-based structure: images/dress/*.png, images/shirt/*.png, etc.
            for cat in categories:
                cat_dir = os.path.join(self.image_base_dir, cat)
                if os.path.exists(cat_dir):
                    # Try both .png and .jpg extensions
                    for ext in ["*.png", "*.jpg", "*.jpeg"]:
                        pattern = os.path.join(cat_dir, ext)
                        for img_path in glob(pattern):
                            img_name = os.path.splitext(os.path.basename(img_path))[0]
                            
                            # Filter by valid_image_ids if provided
                            if valid_image_ids is not None and img_name not in valid_image_ids:
                                continue
                            
                            rel_path = os.path.join(cat, os.path.basename(img_path))
                            
                            # Track unique images
                            unique_images.add(rel_path)
                            
                            # Add multiple key formats for compatibility
                            image_splits[img_name] = rel_path  # "B007FO5H1Q" -> "toptee/B007FO5H1Q.png"
                            image_splits[f"{cat}/{img_name}"] = rel_path  # "toptee/B007FO5H1Q" -> "toptee/B007FO5H1Q.png"
                            image_splits[rel_path] = rel_path  # "toptee/B007FO5H1Q.png" -> "toptee/B007FO5H1Q.png"
        else:
            print_rank("Detected flat image organization")
            # Flat structure: images/*.png or images/*.jpg
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                pattern = os.path.join(self.image_base_dir, ext)
                for img_path in glob(pattern):
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    
                    # Filter by valid_image_ids if provided
                    if valid_image_ids is not None and img_name not in valid_image_ids:
                        continue
                    
                    rel_path = os.path.basename(img_path)
                    
                    # Track unique images
                    unique_images.add(rel_path)
                    
                    # Add key formats
                    image_splits[img_name] = rel_path  # "B007FO5H1Q" -> "B007FO5H1Q.png"
                    image_splits[rel_path] = rel_path  # "B007FO5H1Q.png" -> "B007FO5H1Q.png"
        
        # Store unique image count for accurate reporting
        self._unique_image_count = len(unique_images)
        return image_splits

    def _normalize_annotations(self, raw_annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize FashionIQ annotations to CIRR-compatible format.
        Adds 'reference', 'target_hard', and 'caption' fields for HardNegativeMiner compatibility.
        """
        normalized = []
        
        for ann in raw_annotations:
            # Create normalized annotation with CIRR-compatible fields
            normalized_ann = dict(ann)  # Copy original fields
            
            # Add CIRR-compatible fields
            normalized_ann['reference'] = ann['candidate']  # CIRR uses 'reference'
            normalized_ann['target_hard'] = ann['target']   # CIRR uses 'target_hard'
            
            # Join multiple captions with "and" for single caption field
            if 'captions' in ann and ann['captions']:
                normalized_ann['caption'] = " and ".join(ann['captions'])
            else:
                normalized_ann['caption'] = ""
            
            # Validate that required images exist in image_splits
            candidate_id = ann['candidate']
            target_id = ann['target']
            
            # Try different key formats to find the image
            candidate_found = self._find_image_path(candidate_id, ann.get('category', ''))
            target_found = self._find_image_path(target_id, ann.get('category', ''))
            
            if candidate_found and target_found:
                normalized.append(normalized_ann)
            else:
                missing = []
                if not candidate_found:
                    missing.append(f"candidate:{candidate_id}")
                if not target_found:
                    missing.append(f"target:{target_id}")
                print_rank(f"Warning: Skipping annotation due to missing images: {missing}")
        
        print_rank(f"Normalized {len(normalized)}/{len(raw_annotations)} annotations")
        return normalized

    def _find_image_path(self, image_id: str, category: str = "") -> bool:
        """
        Check if an image exists in image_splits using various key formats.
        Returns True if found, False otherwise.
        """
        # Try different key formats
        possible_keys = [
            image_id,  # "B007FO5H1Q"
            f"{image_id}.png",  # "B007FO5H1Q.png"
            f"{image_id}.jpg",  # "B007FO5H1Q.jpg"
            f"{image_id}.jpeg",  # "B007FO5H1Q.jpeg"
        ]
        
        if category:
            possible_keys.extend([
                f"{category}/{image_id}",  # "toptee/B007FO5H1Q"
                f"{category}/{image_id}.png",  # "toptee/B007FO5H1Q.png"
                f"{category}/{image_id}.jpg",  # "toptee/B007FO5H1Q.jpg"
                f"{category}/{image_id}.jpeg",  # "toptee/B007FO5H1Q.jpeg"
            ])
        
        return any(key in self.image_splits for key in possible_keys)

    def _get_image_path_from_splits(self, image_id: str, category: str = "") -> str:
        """
        Get the actual image path from image_splits using various key formats.
        Returns the relative path if found, otherwise returns a fallback path.
        """
        # Try different key formats
        possible_keys = [
            image_id,  # "B007FO5H1Q"
            f"{image_id}.png",  # "B007FO5H1Q.png"
            f"{image_id}.jpg",  # "B007FO5H1Q.jpg"
            f"{image_id}.jpeg",  # "B007FO5H1Q.jpeg"
        ]
        
        if category:
            possible_keys.extend([
                f"{category}/{image_id}",  # "toptee/B007FO5H1Q"
                f"{category}/{image_id}.png",  # "toptee/B007FO5H1Q.png"
                f"{category}/{image_id}.jpg",  # "toptee/B007FO5H1Q.jpg"
                f"{category}/{image_id}.jpeg",  # "toptee/B007FO5H1Q.jpeg"
            ])
        
        # Try to find the image in image_splits
        for key in possible_keys:
            if key in self.image_splits:
                return self.image_splits[key]
        
        # Fallback: construct path based on category and common extensions
        if category:
            # Try category-based path with different extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                fallback_path = f"{category}/{image_id}{ext}"
                if os.path.exists(self._get_full_image_path(fallback_path)):
                    return fallback_path
        
        # Final fallback: try flat structure
        for ext in ['.png', '.jpg', '.jpeg']:
            fallback_path = f"{image_id}{ext}"
            if os.path.exists(self._get_full_image_path(fallback_path)):
                return fallback_path
        
        # If nothing found, return a default path (will be handled by _load_image)
        return f"{category}/{image_id}.png" if category else f"{image_id}.png"

    def _get_original_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.annotations[idx]
        
        # Use normalized fields (added by _normalize_annotations)
        # Get image paths from image_splits using the normalized reference/target_hard fields
        ref_image_path = self._get_image_path_from_splits(sample['reference'], sample.get('category', ''))
        target_image_path = self._get_image_path_from_splits(sample['target_hard'], sample.get('category', ''))
        
        # Use the normalized caption field
        modification_text = sample.get('caption', '')
        category = sample.get('category', '')
        model_backbone = getattr(self.model_args, "model_backbone", "qwen2_vl")
        
        # Query = (reference image + modification text)
        query_text = process_input_text(
            instruction=f"Change the style of this {category} to <{modification_text}>\nRepresent this modified {category} in one word:",
            # instruction = f"Modify this image with <{modification_text}>\nRepresent the modified image in one word:",
            # instruction = f"Find an image to match the fashion image and style note.\n{modification_text}",
            model_backbone=model_backbone,
            text="",
            add_image_token=True,
        )
        # Positive = target image (empty text + image token)
        pos_text = process_input_text(
            instruction="Represent the given image in one word:",
            # instruction="Represent the given image",
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
        
        return {
            "query_text": query_text,
            "query_image": self._load_image(ref_image_path),
            "pos_text": pos_text,
            "pos_image": self._load_image(target_image_path),
            "neg_text": neg_text,
            "neg_image": self._load_image(ref_image_path),
            "global_dataset_name": "FashionIQ",
            # 🔥 关键修复：使用绝对路径，确保与增强样本的 reference_image 格式一致，
            # 这样 GroupedBatchSampler 才能将同一 reference 的原始样本和增强样本分到同一组
            "reference_image": self._get_full_image_path(ref_image_path),  # normalized for grouping
            "reference_id": self._get_reference_id(ref_image_path),
            "is_augmented": False,
            "category": sample.get("category", ""),  # FashionIQ特有的类别信息
        }

    def _get_augmented_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.augmented_samples[idx]
        
        category = sample.get('category', '')
        model_backbone = getattr(self.model_args, "model_backbone", "qwen2_vl")
        
        query_text = process_input_text(
            instruction=f"Change the style of this {category} to <{sample['modification_text']}>\nRepresent this modified {category} in one word:",
            # instruction = f"Modify this image with <{sample['modification_text']}>\nRepresent the modified image in one word:",
            # instruction = f"Find an image to match the fashion image and style note.\n{sample['modification_text']}",
            model_backbone=model_backbone,
            text="",
            add_image_token=True,
        )
        pos_text = process_input_text(
            instruction="Represent the given image in one word:",
            # instruction="Represent the given image",
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
        
        # 🔥 确保增强样本的 reference_image 也是绝对路径（可能已经是，但double-check）
        ref_img_path = sample["reference_image"]
        # 如果已经是绝对路径就保持，否则转换为绝对路径
        if not os.path.isabs(ref_img_path):
            ref_img_path = self._get_full_image_path(ref_img_path)
        
        return {
            "query_text": query_text,
            "query_image": self._load_image(sample["reference_image"]),
            "pos_text": pos_text,
            "pos_image": self._load_image(sample["target_image"]),
            "neg_text": neg_text,
            "neg_image": self._load_image(sample["reference_image"]),
            "global_dataset_name": "FashionIQ",
            "is_augmented": True,
            "original_mod_text": sample.get("original_mod_text", ""),
            "reference_image": ref_img_path,
            "reference_id": self._get_reference_id(ref_img_path),
            "category": sample.get("category", ""),  # FashionIQ特有的类别信息
        }

# 注册到数据集工厂（与 CIRR 保持一致）
from ...data.dataset.base_pair_dataset import AutoPairDataset
AutoPairDataset.registry["IterativeFashionIQDataset"] = IterativeFashionIQDataset

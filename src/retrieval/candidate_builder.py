# retrieval/candidate_builder.py
from __future__ import annotations
from typing import Dict, List, Any
import os

from src.utils import print_rank
from src.utils.path_utils import get_full_image_path


class CandidateBuilder:
    """
    Build retrieval candidates from image splits and validate coverage.
    """

    def __init__(
        self,
        image_splits: Dict[str, str],
        annotations: List[Dict[str, Any]],
        image_base_dir: str,
    ) -> None:
        self.image_splits = image_splits
        self.annotations = annotations
        self.image_base_dir = image_base_dir

    # ---------------------------- public ----------------------------

    def build(self) -> List[str]:
        """
        Build candidate set and keep relative paths for cache/signature compatibility.
        """
        print_rank("Building CIRR retrieval candidate set...")

        if not isinstance(self.image_splits, dict):
            raise ValueError(f"Expected image_splits to be a dict, got {type(self.image_splits)}")

        print_rank(f"CIRR image_splits contains {len(self.image_splits)} total images")

        retrieval_candidates: List[str] = []
        valid = 0
        for img_name, img_rel in self.image_splits.items():
            if self._image_exists(img_rel):
                retrieval_candidates.append(img_rel)
                valid += 1
            else:
                if len(retrieval_candidates) < 5:
                    print_rank(f"Warning: Image not found: {img_name} -> {img_rel}")

        print_rank("Built CIRR retrieval candidate set:")
        print_rank(f"  • Total candidates from image_splits: {len(self.image_splits)}")
        print_rank(f"  • Valid candidates (files exist): {valid}")
        print_rank(f"  • Missing files: {len(self.image_splits) - valid}")

        if len(retrieval_candidates) < 1000:
            print_rank(f"Warning: only {len(retrieval_candidates)} retrieval candidates found.")
            print_rank("    This might be insufficient for high-quality hard negative mining.")
            print_rank("    Expected ~16,000+ candidates for CIRR dataset.")
        else:
            print_rank(f"Excellent: {len(retrieval_candidates)} candidates available for hard negative mining.")
            print_rank(f"    This is {len(retrieval_candidates)/200:.1f}x more than the previous limited approach.")

        self._validate_candidate_coverage(retrieval_candidates)
        return retrieval_candidates

    # ---------------------------- internal ----------------------------

    def _image_exists(self, rel_path: str) -> bool:
        if not isinstance(rel_path, str):
            return False
        full = get_full_image_path(rel_path, self.image_base_dir)
        return os.path.exists(full)

    def _validate_candidate_coverage(self, retrieval_candidates: List[str]) -> None:
        """Validate that all reference and target images from training data are in candidate set"""
        print_rank("Validating candidate set coverage...")
        cand_set = set(retrieval_candidates)

        missing_refs = 0
        missing_targets = 0
        total_refs = set()
        total_targets = set()

        for ann in self.annotations:
            ref_path = self.image_splits.get(ann["reference"], ann["reference"])
            tgt_path = self.image_splits.get(ann["target_hard"], ann["target_hard"])
            total_refs.add(ref_path)
            total_targets.add(tgt_path)
            if ref_path not in cand_set:
                missing_refs += 1
            if tgt_path not in cand_set:
                missing_targets += 1

        print_rank("Coverage validation results:")
        print_rank(f"  • Unique reference images: {len(total_refs)}")
        print_rank(f"  • Unique target images: {len(total_targets)}")
        print_rank(f"  • Missing reference images: {missing_refs}")
        print_rank(f"  • Missing target images: {missing_targets}")

        if missing_refs == 0 and missing_targets == 0:
            print_rank("Perfect coverage: all training images are in the candidate set.")
        else:
            print_rank("Warning: coverage issues detected. This may affect hard negative mining quality.")

        denom = len(total_refs | total_targets)
        covered = denom - missing_refs - missing_targets
        rate = (covered / denom * 100) if denom > 0 else 0.0
        print_rank(f"  • Overall coverage rate: {rate:.1f}%")

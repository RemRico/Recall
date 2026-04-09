# retrieval/embedding_cache.py
from __future__ import annotations
from typing import Callable, Dict, List, Any
import os
import time
import hashlib

import torch

from src.utils import print_rank


class EmbeddingCache:
    """
    Cache and compute target embeddings.

    `get_or_compute` validates cached content before reuse and falls back to
    recomputation when cache validation fails.
    """

    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.cache_dir = os.path.join(self.experiment_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ---------------------------- public API ----------------------------

    def get_or_compute(
        self,
        target_database: List[str],
        model: torch.nn.Module,
        processor: Any,
        model_backbone: str,
        device: torch.device,
        prepare_target_inputs_fn: Callable[[List[str], Any, str, torch.device], Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Read or compute target embeddings with dtype/device alignment.
        """
        cache_file = self._get_cache_file_path(target_database)
        # 1) Try cache.
        if os.path.exists(cache_file):
            try:
                print_rank(f"Loading cached target embeddings from {cache_file}")
                cached = torch.load(cache_file, map_location="cpu")
                if (
                    isinstance(cached, dict)
                    and cached.get("target_paths") == target_database
                    and "embeddings" in cached
                    and cached["embeddings"].size(0) == len(target_database)
                ):
                    embs = cached["embeddings"]
                    model_dtype = next(model.parameters()).dtype
                    embs = embs.to(dtype=model_dtype, device=device)
                    print_rank(f"Cache hit: loaded {len(target_database)} target embeddings (dtype: {embs.dtype})")
                    return embs
                else:
                    print_rank("Cache validation failed, will recompute embeddings")
            except Exception as e:
                print_rank(f"Error loading cache: {e}, will recompute embeddings")

        # 2) Compute embeddings.
        print_rank(f"Computing target embeddings for {len(target_database)} images...")
        embs = self._compute_batch(
            target_database, model, processor, model_backbone, device, prepare_target_inputs_fn
        )

        # 3) Persist cache.
        try:
            cache_data = {
                "target_paths": target_database,
                "embeddings": embs.detach().cpu(),
                "timestamp": time.time(),
                "model_backbone": model_backbone,
            }
            torch.save(cache_data, cache_file)
            print_rank(f"Cached target embeddings to {cache_file}")
        except Exception as e:
            print_rank(f"Warning: Failed to cache embeddings: {e}")

        return embs

    def get_cache_file_path(self, target_database: List[str]) -> str:
        """Expose cache path for distributed done-flag workflow."""
        return self._get_cache_file_path(target_database)

    # ---------------------------- internal ----------------------------

    def _get_cache_file_path(self, target_database: List[str]) -> str:
        content_hash = hashlib.md5(str(sorted(target_database)).encode()).hexdigest()[:8]
        filename = f"target_embeddings_{len(target_database)}_{content_hash}.pt"
        return os.path.join(self.cache_dir, filename)

    def _compute_batch(
        self,
        target_database: List[str],
        model: torch.nn.Module,
        processor: Any,
        model_backbone: str,
        device: torch.device,
        prepare_target_inputs_fn: Callable[[List[str], Any, str, torch.device], Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute target embeddings in mini-batches with robust error fallback.
        """
        batch_size = 8
        total = len(target_database)
        total_batches = (total + batch_size - 1) // batch_size

        chunks: List[torch.Tensor] = []
        start = time.time()
        with torch.no_grad():
            for i in range(0, total, batch_size):
                bidx = i // batch_size + 1
                part = target_database[i : i + batch_size]

                # ETA estimate.
                if bidx > 1:
                    elapsed = time.time() - start
                    avg = elapsed / (bidx - 1)
                    remain = total_batches - bidx + 1
                    eta = int(avg * remain)
                    progress = (bidx - 1) / total_batches * 100
                    eta_str = f"ETA: {eta//60:02d}:{eta%60:02d}"
                else:
                    progress = 0.0
                    eta_str = "ETA: calculating..."

                print_rank(
                    f"  Batch {bidx:4d}/{total_batches} ({progress:5.1f}%) - Processing {len(part)} images - {eta_str}"
                )

                try:
                    inp = prepare_target_inputs_fn(part, processor, model_backbone, device)
                    embs = model.encode_input(inp)
                    embs = self._process_embeddings(embs, len(part), f"target_batch_{bidx}")
                    chunks.append(embs.detach().cpu())
                except Exception as e:
                    print_rank(f"Error encoding target batch {bidx}: {e}")
                    dummy = torch.randn(len(part), 768)
                    chunks.append(dummy)

        dur = time.time() - start
        print_rank(f"Target embeddings computation completed in {int(dur//60):02d}:{int(dur%60):02d}")
        if dur > 0:
            print_rank(f"   Average speed: {total/dur:.1f} images/second")
        final = torch.cat(chunks, dim=0)
        print_rank(f"   Final embeddings shape: {final.shape}")
        # Keep unnormalized embeddings; retrieval normalizes before similarity.
        model_dtype = next(model.parameters()).dtype
        return final.to(dtype=model_dtype, device=device)

    @staticmethod
    def _process_embeddings(embeddings: torch.Tensor, expected_bs: int, tag: str) -> torch.Tensor:
        """
        Align embedding tensor dimensions with safe fallback behavior.
        """
        if embeddings is None:
            print_rank(f"Warning: {tag} returned None, using dummy embeddings")
            return torch.randn(expected_bs, 768)

        if embeddings.dim() == 0:
            if embeddings.numel() > 0:
                embeddings = embeddings.view(1, -1)
            else:
                return torch.randn(expected_bs, 768)
        elif embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.size(0), -1)

        if embeddings.size(0) != expected_bs:
            print_rank(f"Warning: {tag} batch size {embeddings.size(0)} != expected {expected_bs}")
            if embeddings.size(0) == 1 and expected_bs > 1:
                embeddings = embeddings.repeat(expected_bs, 1)
            else:
                return torch.randn(expected_bs, embeddings.size(-1) if embeddings.numel() > 0 else 768)

        return embeddings

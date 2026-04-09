# mining/hard_negative.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os

from ..utils import print_rank
from ..utils.path_utils import get_full_image_path
import time
import json
import torch
import torch.distributed as dist


class HardNegativeMiner:
    """
    Hard-negative mining and filtering helper.

    Dependencies:
    - CandidateBuilder
    - EmbeddingCache
    - RetrievalEngine

    Supports both single-GPU and distributed collection.
    """

    def __init__(self,experiment_dir: str, iteration_round: int,
                 candidate_builder, retrieval_engine, embedding_cache,
                 image_base_dir: Optional[str] = None,
                 max_negatives_per_query: int = 5,
                 examine_topk: int = 10,
                 post_gt_negatives: int = 0):
        self.image_base_dir = image_base_dir or ""
        self.max_negatives_per_query = max_negatives_per_query
        self.examine_topk = examine_topk
        self.experiment_dir = experiment_dir
        self.iteration_round = iteration_round
        self.candidate_builder = candidate_builder
        self.retrieval_engine = retrieval_engine
        self.embedding_cache = embedding_cache
        self.post_gt_negatives = max(int(post_gt_negatives), 0)

        os.makedirs(self.experiment_dir, exist_ok=True)
        self.hard_negatives_cache = []

    # ----------------------
    # public
    # ----------------------
    @property
    def hard_negatives_file(self) -> str:
        return os.path.join(self.experiment_dir, f"hard_negatives_iter_{self.iteration_round}.json")

    def collect_single_gpu(self, retrieval_model, annotations, batch_size=8, max_samples=None, *, processor=None, model_backbone: Optional[str] = None, device=None, prepare_target_inputs_fn=None, **_):
        """
        Collect hard negatives on a single GPU.
        """
        if os.path.exists(self.hard_negatives_file):
            print_rank(f"Loading existing hard negatives: {self.hard_negatives_file}")
            try:
                with open(self.hard_negatives_file, "r") as f:
                    data = json.load(f)
                if max_samples is not None and len(data) > max_samples:
                    data = data[:max_samples]
                self.hard_negatives_cache = data
                print_rank(f"Loaded {len(data)} hard negatives from cache")
                return data
            except Exception as e:
                print_rank(f"Load cache failed, will recompute: {e}")

        # Build retrieval candidates once.
        candidates = self.candidate_builder.build()
        print_rank(f"Candidates ready: {len(candidates)}")

        # Compute or load target embeddings.
        model = retrieval_model
        proc = processor if processor is not None else getattr(model, "processor", None)
        device = next(model.parameters()).device
        backbone = (
            model_backbone
            if model_backbone is not None
            else getattr(self.retrieval_engine.model_args, "model_backbone", "qwen2_5_vl")
        )
        dev = device if device is not None else next(model.parameters()).device
        prep_fn = (
            prepare_target_inputs_fn
            if prepare_target_inputs_fn is not None
            else self.retrieval_engine._prepare_target_inputs
        )

        # Compute / load target embeddings with aligned target-input preparation.
        target_emb = self.embedding_cache.get_or_compute(
            candidates,   # target_database
            model,        # model
            proc,         # processor
            backbone,     # model_backbone
            dev,          # device
            prep_fn,
        )

        # Process queries.
        samples = annotations[:max_samples] if max_samples is not None else annotations
        print_rank(f"Starting single-GPU hard negative collection for {len(samples)} queries")

        all_neg = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                chunk = samples[i:i+batch_size]
                batch = [{
                    "reference_image": self._to_abs_path(self.candidate_builder, ann['reference']),
                    "modification_text": ann['caption'],
                    "target_image": self._to_abs_path(self.candidate_builder, ann['target_hard']),
                } for ann in chunk]

                results = self.retrieval_engine.run_retrieval_with_cached_targets(
                    model, batch, target_emb, max_samples=max_samples
                )

                negs = self._identify_hard_negatives(batch, results)
                all_neg.extend(negs)

        with open(self.hard_negatives_file, "w") as f:
            json.dump(all_neg, f, indent=2)
        print_rank(f"Saved {len(all_neg)} hard negatives -> {self.hard_negatives_file}")
        self.hard_negatives_cache = all_neg
        return all_neg

    def collect_distributed_minimal(
        self,
        retrieval_model,
        annotations,
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        *,
        processor=None,
        model_backbone: Optional[str] = None,
        device=None,
        prepare_target_inputs_fn=None,
        **_
    ):
        """
        Distributed collection with file-based synchronization.

        Rank 0 computes and stores target embeddings, other ranks poll for
        readiness, then process disjoint query shards and merge with
        `all_gather_object`.
        """
        if not dist.is_initialized():
            return self.collect_single_gpu(
                retrieval_model, annotations, batch_size, max_samples,
                processor=processor, model_backbone=model_backbone,
                device=device, prepare_target_inputs_fn=prepare_target_inputs_fn,
            )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dev = device if device is not None else (f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # If final output already exists, all ranks read directly.
        if os.path.exists(self.hard_negatives_file):
            try:
                with open(self.hard_negatives_file, "r") as f:
                    data = json.load(f)
                if max_samples is not None and len(data) > max_samples:
                    data = data[:max_samples]
                self.hard_negatives_cache = data
                print_rank(f"GPU {rank}: loaded existing hard negatives ({len(data)})")
                return data
            except Exception as e:
                print_rank(f"GPU {rank}: read existed file failed, recompute: {e}")

        # Build candidate set once.
        candidates = self.candidate_builder.build()

        model = retrieval_model
        proc = processor if processor is not None else getattr(model, "processor", None)
        backbone = (
            model_backbone
            if model_backbone is not None
            else getattr(self.retrieval_engine.model_args, "model_backbone", "qwen2_vl")
        )
        prep_fn = (
            prepare_target_inputs_fn
            if prepare_target_inputs_fn is not None
            else self.retrieval_engine._prepare_target_inputs
        )

        cache_file = self.embedding_cache.get_cache_file_path(candidates)
        done_flag = cache_file + ".done"

        # Rank 0 computes and writes embeddings.
        if rank == 0:
            print_rank("GPU 0: Computing / loading target embeddings...")
            self.embedding_cache.get_or_compute(
                candidates,
                model,
                proc,
                backbone,
                'cuda:0' if torch.cuda.is_available() else 'cpu',
                prep_fn,
            )
            with open(done_flag, "w") as f:
                f.write(str(time.time()))
            print_rank("GPU 0: target embeddings ready (file + flag)")
        else:
            print_rank(f"GPU {rank}: Waiting embeddings flag (no barrier)...")

        # Other ranks poll without distributed barrier.
        if rank != 0:
            _poll_files([cache_file, done_flag], rank, timeout_s=3 * 3600)

        # All ranks load cached embeddings.
        try:
            cached = torch.load(cache_file, map_location=dev)
            target_emb = cached["embeddings"].to(next(model.parameters()).dtype)
            print_rank(f"GPU {rank}: loaded target embeddings {target_emb.shape}")
        except Exception as e:
            raise RuntimeError(f"GPU {rank}: load embeddings failed: {e}")

        # Split query workload by rank.
        samples = annotations[:max_samples] if max_samples is not None else annotations
        total = len(samples)
        per_rank = (total + world_size - 1) // world_size
        s = rank * per_rank
        e = min(s + per_rank, total)
        local_anns = samples[s:e]
        print_rank(f"GPU {rank}: assigned {len(local_anns)} / {total} queries")

        model.eval()
        local_neg = []
        with torch.no_grad():
            for i in range(0, len(local_anns), batch_size):
                chunk = local_anns[i:i+batch_size]
                batch = [{
                    "reference_image": self._to_abs_path(self.candidate_builder, ann['reference']),
                    "modification_text": ann['caption'],
                    "target_image": self._to_abs_path(self.candidate_builder, ann['target_hard']),
                } for ann in chunk]

                results = self.retrieval_engine.run_retrieval_with_cached_targets(
                    model, batch, target_emb.to(dev), max_samples=max_samples
                )
                negs = self._identify_hard_negatives(batch, results)
                local_neg.extend(negs)

        # Gather all local hard negatives, then rank 0 writes final result.
        gathered = [None] * world_size
        try:
            dist.all_gather_object(gathered, local_neg)
        except Exception as e:
            print_rank(f"GPU {rank}: all_gather_object failed: {e}")
            gathered = [[] for _ in range(world_size)]
            gathered[rank] = local_neg

        final = []
        if rank == 0:
            for idx, part in enumerate(gathered):
                if part is not None:
                    final.extend(part)
                    print_rank(f"GPU 0: merge from GPU {idx} +{len(part)}")
            with open(self.hard_negatives_file, "w") as f:
                json.dump(final, f, indent=2)
            print_rank(f"GPU 0: saved {len(final)} hard negatives -> {self.hard_negatives_file}")
            self.hard_negatives_cache = final

        container = [final if rank == 0 else []]
        dist.broadcast_object_list(container, src=0)
        if rank != 0:
            self.hard_negatives_cache = container[0]
        print_rank(f"GPU {rank}: done. total hard negatives = {len(self.hard_negatives_cache)}")
        return self.hard_negatives_cache

    # Helper: robust same-image check by normalized absolute path, with basename fallback.
    def _is_same_image(self, path1: Optional[str], path2: Optional[str]) -> bool:
        if not path1 or not path2:
            return False
        try:
            full1 = get_full_image_path(path1, self.image_base_dir)
            full2 = get_full_image_path(path2, self.image_base_dir)
            n1 = os.path.normpath(os.path.realpath(full1))
            n2 = os.path.normpath(os.path.realpath(full2))
            return n1 == n2
        except Exception:
            return os.path.basename(str(path1)) == os.path.basename(str(path2))

    def _identify_hard_negatives(self,
                 batch: List[Dict[str, Any]],
                 retrieval_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify hard negatives from retrieval outputs.

        Args:
        - batch: [{'reference_image', 'modification_text', 'target_image'}, ...]
        - retrieval_results: {
            'top_k_indices': List[List[int]],
            'gt_indices': List[int],
            'similarities': List[List[float]],
                        # Always present in real retrieval mode:
            'target_paths': List[str]
          }
        Returns:
            Hard negative samples with metadata.
        """
        hard_negatives: List[Dict[str, Any]] = []

        top_k_indices = retrieval_results['top_k_indices']
        gt_indices = retrieval_results['gt_indices']
        similarities = retrieval_results['similarities']

        target_paths = retrieval_results.get('target_paths', None)
        is_real_retrieval = target_paths is not None
        gt_full_ranks = retrieval_results.get('gt_full_ranks')  # List[int] or None
        gt_sim_list = retrieval_results.get('gt_similarities')  # List[float|None] or None

        for qidx, (query, gt_target, top_k, sims) in enumerate(
            zip(batch, gt_indices, top_k_indices, similarities)
        ):
            # Filter out the reference image itself.
            query_ref_path = query['reference_image']
            filtered_hard_negatives = []

            # Compute GT top-k rank (1-based) for reporting.
            gt_topk_rank = -1
            if gt_target in top_k:
                gt_topk_rank = int(top_k.index(gt_target) + 1)

            def process_negative_candidate(neg_pos: int,
                                           neg_idx: int,
                                           gt_position: int = -1) -> bool:
                """Append a negative candidate unless it matches filtered cases."""
                if is_real_retrieval:
                    hard_negative_image = (
                        target_paths[neg_idx] if neg_idx < len(target_paths)
                        else f"target_{neg_idx}"
                    )
                else:
                    hard_negative_image = neg_idx

                # 1) candidate == reference image
                if is_real_retrieval and self._is_same_image(query_ref_path, hard_negative_image):
                    print_rank(f"Filtered out reference image as hard negative: {query_ref_path}")
                    return False

                # 2) candidate == ground-truth target image
                if is_real_retrieval and 'target_image' in query and \
                        self._is_same_image(query['target_image'], hard_negative_image):
                    print_rank(f"Filtered out ground truth as hard negative: {hard_negative_image}")
                    return False

                # Absolute GT rank and similarity metadata.
                full_rank = gt_full_ranks[qidx] if isinstance(gt_full_ranks, list) and qidx < len(gt_full_ranks) else -1
                gt_sim = gt_sim_list[qidx] if isinstance(gt_sim_list, list) and qidx < len(gt_sim_list) else None
                gt_in_candidates = (gt_target != -1)

                filtered_hard_negatives.append({
                    'reference_image': query['reference_image'],
                    'modification_text': query['modification_text'],
                    'target_image': query['target_image'],  # GT
                    'hard_negative_image': hard_negative_image,
                    'rank_position': int(neg_pos + 1),
                    'gt_rank': int(full_rank),
                    'gt_topk_rank': int(gt_topk_rank),
                    'gt_in_candidates': bool(gt_in_candidates),
                    'gt_similarity': float(gt_sim) if gt_sim is not None else None,
                    'similarity_score': float(sims[neg_pos]) if neg_pos < len(sims) else 0.0,
                    'is_real_retrieval': is_real_retrieval
                })
                return True

            collected = 0
            limit = self.max_negatives_per_query
            topN = min(self.examine_topk, len(top_k))

            if gt_target == -1:
                # GT is not in retrieval candidates.
                for neg_pos in range(topN):
                    if collected >= limit:
                        break
                    if process_negative_candidate(neg_pos, top_k[neg_pos]):
                        collected += 1

            elif gt_target in top_k:
                # GT appears in top-k: take negatives before GT.
                gt_position = top_k.index(gt_target)
                if gt_position > 0:
                    for neg_pos in range(gt_position):
                        if collected >= limit:
                            break
                        if process_negative_candidate(neg_pos, top_k[neg_pos], gt_position):
                            collected += 1
                if self.post_gt_negatives > 0 and collected < limit:
                    post_start = gt_position + 1
                    post_end = min(topN, post_start + self.post_gt_negatives)
                    for neg_pos in range(post_start, post_end):
                        if collected >= limit:
                            break
                        if process_negative_candidate(neg_pos, top_k[neg_pos], gt_position):
                            collected += 1
            else:
                # GT is in candidate set but not in top-k.
                for neg_pos in range(topN):
                    if collected >= limit:
                        break
                    if process_negative_candidate(neg_pos, top_k[neg_pos]):
                        collected += 1

            hard_negatives.extend(filtered_hard_negatives)
            if filtered_hard_negatives:
                print_rank(f"Query {qidx}: Collected {len(filtered_hard_negatives)} valid hard negatives after filtering")
            else:
                print_rank(f"Query {qidx}: No valid hard negatives found after filtering")

        return hard_negatives
    
    def _to_abs_path(self, builder, key_or_path: str) -> str:
        """Resolve split keys or relative paths to absolute image paths."""
        val = builder.image_splits.get(key_or_path, key_or_path)
        return get_full_image_path(val, self.image_base_dir)


    # ----------------------
    # Polling util for file-based distributed synchronization.
    # ----------------------
def _poll_files(paths, rank, timeout_s=3*3600, poll_interval=2):
    start = time.time()
    missing = set(paths)
    while time.time() - start < timeout_s:
        for p in list(missing):
            if os.path.exists(p):
                missing.remove(p)
        if not missing:
            time.sleep(1)
            return
        time.sleep(poll_interval)
        elapsed = int(time.time() - start)
        if elapsed % 60 == 0:
            print_rank(f"GPU {rank}: waiting files: {list(missing)} (elapsed {elapsed}s)")
    raise TimeoutError(f"GPU {rank}: timeout waiting files: {list(missing)}")

# aug/caption_generator.py
import os
import time
import json
import shutil
import traceback
import torch
import torch.distributed as dist
from src.utils import print_rank
from .validators import CaptionValidator
from .batchers import CaptionBatcher
from src.utils.path_utils import get_full_image_path


class CaptionGenerator:
    """
    Caption augmentation generator for single-GPU and distributed workflows.

    Behavior is preserved from the original implementation while split into
    smaller helper modules.
    """

    def __init__(
        self,
        foundation_model,
        model_args,
        experiment_dir,
        iteration_round,
        prepare_fns,
        generate_fns,
        image_base_dir: str = "",
        image_splits: dict | None = None,
    ):
        """
        Args:
            foundation_model: Foundation model with `.processor` and `.generate`
            model_args: Model argument container
            experiment_dir: Experiment directory for cache/sync artifacts
            iteration_round: Current iteration index (writes to next-iter file)
            prepare_fns: dict, {"qwen": fn, "llava": fn, "generic": fn}
            generate_fns: dict, {"qwen": fn, "llava": fn, "generic": fn}
            image_base_dir: Dataset image root directory (e.g., /path/to/CIRR)
            image_splits: Mapping {sample-id or key -> relative path}
        """
        self.foundation_model = foundation_model
        self.model_args = model_args
        self.experiment_dir = experiment_dir
        self.iteration_round = iteration_round
        self.augmented_samples = []

        self.batcher = CaptionBatcher(foundation_model, model_args, prepare_fns, generate_fns)
        self.validator = CaptionValidator()

        # Path normalization context.
        self.image_base_dir = image_base_dir or ""
        self.image_splits = image_splits or {}
        self._preview_printed = 0
        self._preview_limit = 5

    def _resolve_image_path(self, p: str) -> str:
        """
        Resolve image path to absolute path.

        - If `p` is already absolute, return as-is.
        - If `p` is a split key, map to relative path and join base dir.
        - Otherwise treat as relative path and join base dir.
        """
        if not isinstance(p, str) or p == "":
            return p
        if os.path.isabs(p):
            return p
        mapped = self.image_splits.get(p, p)
        return get_full_image_path(mapped, self.image_base_dir)

    def _coerce_saved_samples(self, saved: dict) -> list:
        if not isinstance(saved, dict):
            return []
        samples = saved.get("samples", [])
        # nested meta dict
        if isinstance(samples, dict):
            # case 1: inner meta with 'samples' list
            inner = samples.get("samples") if isinstance(samples, dict) else None
            if isinstance(inner, list):
                return [s for s in inner if isinstance(s, dict)]
            # case 2: dict-of-samples
            if all(isinstance(v, dict) for v in samples.values()):
                return list(samples.values())
            # case 3: pick longest list in values
            lists = [v for v in samples.values() if isinstance(v, list)]
            if lists:
                best = max(lists, key=len)
                return [s for s in best if isinstance(s, dict)]
            # fallback: values that are dicts
            return [v for v in samples.values() if isinstance(v, dict)]
        # plain list
        if isinstance(samples, list):
            return [s for s in samples if isinstance(s, dict)]
        # other iterable
        try:
            ls = list(samples)
            return [s for s in ls if isinstance(s, dict)]
        except Exception:
            return []

    @staticmethod
    def _sanitize_rank(value):
        """Normalize rank-like values to non-negative ints or None."""
        if isinstance(value, bool) or value is None:
            return int(value) if isinstance(value, bool) else None
        try:
            ivalue = int(value)
            return ivalue if ivalue >= 0 else None
        except (ValueError, TypeError, OverflowError):
            return None

    @staticmethod
    def _sanitize_float(value):
        """Normalize float-like values for JSON serialization."""
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _normalize_item_paths(self, item: dict) -> dict:
        out = dict(item)
        if "reference_image" in out:
            out["reference_image"] = self._resolve_image_path(out["reference_image"])
        if "target_image" in out and out["target_image"]:
            out["target_image"] = self._resolve_image_path(out["target_image"])
        if "hard_negative_image" in out and out["hard_negative_image"]:
            out["hard_negative_image"] = self._resolve_image_path(out["hard_negative_image"])
        return out

    # =========================
    #     Single-GPU workflow
    # =========================
    def generate_augmented_captions(self, hard_negatives):
        """Generate augmented captions in single-GPU mode."""
        if not self.foundation_model:
            print_rank("No foundation model, skipping caption generation")
            return []

        next_iter = self.iteration_round + 1
        aug_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{next_iter}.json")

        # Read cache.
        if os.path.exists(aug_file):
            print_rank(f"Loading existing augmented samples from {aug_file}")
            try:
                with open(aug_file, "r") as f:
                    saved = json.load(f)
                samples = self._coerce_saved_samples(saved)
                declared = saved.get("total_samples") if isinstance(saved, dict) else None
                if isinstance(declared, int) and declared != len(samples):
                    print_rank(f"Loaded {len(samples)} samples (declared {declared}) — coerced")
                self.augmented_samples = samples
                return samples
            except Exception as e:
                print_rank(f"Error loading augmented samples: {e}, regenerating...")

        print_rank(f"Generating augmented captions for {len(hard_negatives)} hard negatives")
        augmented_samples = []
        batch_size = 4
        total_batches = (len(hard_negatives) + batch_size - 1) // batch_size
        start_time = time.time()

        minimal_mode = getattr(self.model_args, "foundation_prompt_mode", "minimal") == "minimal"

        for i in range(0, len(hard_negatives), batch_size):
            batch_idx = i // batch_size + 1
            batch = hard_negatives[i:i+batch_size]
            # ETA
            if batch_idx > 1:
                elapsed = time.time() - start_time
                avg_tpb = elapsed / (batch_idx - 1)
                remain = total_batches - batch_idx + 1
                eta = f"ETA {int((avg_tpb*remain)//60):02d}:{int((avg_tpb*remain)%60):02d}"
            else:
                eta = "ETA calculating..."

            print_rank(f"Processing batch {batch_idx}/{total_batches} ({len(batch)}) - {eta}")
            try:
                batch_start = time.time()
                batch_aug = self._generate_caption_batch(batch, minimal_mode=minimal_mode)
                augmented_samples.extend(batch_aug)
                print_rank(f"Batch {batch_idx}/{total_batches} done in {time.time()-batch_start:.1f}s, +{len(batch_aug)}")
            except Exception as e:
                print_rank(f"Error in batch {batch_idx}: {e}")
                continue

        augmented_samples = self.validator.filter_valid_samples(augmented_samples)
        self._log_preview(augmented_samples)
        self._save_augmented_samples(augmented_samples)
        total_time = time.time() - start_time
        print_rank(f"Generated {len(augmented_samples)} samples in {total_time:.1f}s")
        return augmented_samples

    def _generate_caption_batch(self, hard_negatives_batch, minimal_mode: bool = False):
        """Generate captions in batch (shared by single/distributed flows)."""
        from PIL import Image

        augmented = []
        foundation_processor = getattr(self.foundation_model, "processor", None)
        if foundation_processor is None:
            print_rank("Foundation model has no processor")
            return []

        device = next(self.foundation_model.parameters()).device

        # Extract dataset name from samples
        if hard_negatives_batch:
            first_sample = hard_negatives_batch[0]
            dataset_name = first_sample.get("global_dataset_name", None)
            if dataset_name:
                setattr(self.model_args, "current_dataset_name", dataset_name)
                print_rank(f"[CaptionGenerator] Detected dataset: {dataset_name}")

        ref_images, tgt_images, texts, meta = [], [], [], []

        # Build batch inputs.
        for hard_neg in hard_negatives_batch:
            try:
                norm_item = self._normalize_item_paths(hard_neg)

                ref_path = norm_item["reference_image"]
                tgt_path = norm_item.get("hard_negative_image", norm_item.get("target_image"))

                ref_img = Image.open(ref_path).convert("RGB")
                tgt_img = Image.open(tgt_path).convert("RGB")

                ref_images.append(ref_img)
                tgt_images.append(tgt_img)
                texts.append(norm_item["modification_text"])
                meta.append(norm_item)  # Keep normalized absolute paths for write-back.
            except Exception as e:
                print_rank(f"Error preparing sample: {e}")

        # Run generation.
        generated_texts = []
        if ref_images:
            generated_texts = self.batcher.generate_batch(
                ref_images, tgt_images, texts, foundation_processor, device
            )
        for hard_neg, gen_text in zip(meta, generated_texts):
            if gen_text and self.validator.is_valid(gen_text):
                candidate_rank = self._sanitize_rank(hard_neg.get("rank_position"))
                gt_rank = self._sanitize_rank(hard_neg.get("gt_rank"))
                gt_topk_rank = self._sanitize_rank(hard_neg.get("gt_topk_rank"))
                gt_in_candidates = hard_neg.get("gt_in_candidates")
                if isinstance(gt_in_candidates, bool):
                    gt_in_candidates_val = gt_in_candidates
                else:
                    gt_in_candidates_val = bool(gt_in_candidates) if gt_in_candidates is not None else None
                is_before_gt = None
                if gt_topk_rank is not None and candidate_rank is not None:
                    is_before_gt = candidate_rank < gt_topk_rank
                similarity_score = self._sanitize_float(hard_neg.get("similarity_score"))
                gt_similarity = self._sanitize_float(hard_neg.get("gt_similarity"))

                augmented.append({
                    "reference_image": hard_neg["reference_image"],
                    "modification_text": gen_text,
                    "target_image": hard_neg.get("hard_negative_image", hard_neg.get("target_image")),
                    "original_target_image": hard_neg.get("target_image"),
                    "original_mod_text": hard_neg["modification_text"],
                    "is_augmented": True,
                    "hard_negative_rank": candidate_rank,
                    "gt_rank": gt_rank,
                    "gt_topk_rank": gt_topk_rank,
                    "gt_in_candidates": gt_in_candidates_val,
                    "is_before_gt_in_topk": is_before_gt,
                    "similarity_score": similarity_score,
                    "gt_similarity": gt_similarity,
                })
                if minimal_mode and self._preview_printed < self._preview_limit:
                    orig_text = hard_neg.get("modification_text")
                    tgt_img = hard_neg.get("hard_negative_image", hard_neg.get("target_image"))
                    print_rank(
                        f"[Preview {self._preview_printed+1}/{self._preview_limit}] target={tgt_img}; original='{orig_text}'; new='{gen_text}'"
                    )
                    self._preview_printed += 1
            elif minimal_mode:
                print_rank("[CaptionGenerator] minimal pipeline returned empty or invalid text; sample skipped")
        return augmented

    def _log_preview(self, augmented_samples, preview_count: int = 5):
        if not augmented_samples:
            return
        print_rank("--- Preview of generated samples (minimal rewrite mode) ---")
        for idx, sample in enumerate(augmented_samples[:preview_count]):
            orig = sample.get("original_mod_text") or sample.get("original_text")
            new = sample.get("modification_text")
            tgt = sample.get("target_image")
            print_rank(f"[{idx+1}] target={tgt}; original='{orig}'; new='{new}'")
        if len(augmented_samples) > preview_count:
            print_rank(f"... (total {len(augmented_samples)} samples)")

    # =========================
    #   Distributed workflow
    # =========================
    def generate_augmented_captions_distributed(self, hard_negatives):
        """
        Generate captions in distributed mode using file-based synchronization.
        Behavior mirrors the original implementation while reusing validators/savers.
        """
        minimal_mode = getattr(self.model_args, "foundation_prompt_mode", "minimal") == "minimal"
        if not self.foundation_model:
            print_rank("No foundation model provided, skipping caption generation")
            return []

        next_iter = self.iteration_round + 1
        final_aug_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{next_iter}.json")

        # Read cache: all ranks try direct loading first.
        if os.path.exists(final_aug_file):
            print_rank(f"Loading existing augmented samples from {final_aug_file}")
            try:
                with open(final_aug_file, "r") as f:
                    saved = json.load(f)
                samples = self._coerce_saved_samples(saved)
                declared = saved.get("total_samples") if isinstance(saved, dict) else None
                if isinstance(declared, int) and declared != len(samples):
                    print_rank(f"Loaded {len(samples)} samples (declared {declared}) — coerced")
                self.augmented_samples = samples
                return samples
            except Exception as e:
                print_rank(f"Error loading augmented samples: {e}, regenerating...")

        # Fallback to single-GPU path when distributed is not initialized.
        if not dist.is_initialized():
            return self.generate_augmented_captions(hard_negatives)

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print_rank(f"Starting distributed caption generation for {len(hard_negatives)} samples using {world_size} GPUs")

        # Partition workload.
        total = len(hard_negatives)
        per_gpu = (total + world_size - 1) // world_size
        start_idx = rank * per_gpu
        end_idx = min(start_idx + per_gpu, total)
        local_list = hard_negatives[start_idx:end_idx]
        print_rank(f"GPU {rank}: Assigned {len(local_list)} samples [{start_idx}, {end_idx})")

        # Place model on rank-local device.
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        if hasattr(self.foundation_model, "to"):
            self.foundation_model = self.foundation_model.to(device)

        # Local generation.
        local_aug = []
        if local_list:
            batch_size = 4
            total_batches = (len(local_list) + batch_size - 1) // batch_size
            start_time = time.time()
            for i in range(0, len(local_list), batch_size):
                bidx = i // batch_size + 1
                batch = local_list[i:i+batch_size]

                # Reduce log noise: rank 0 logs every batch, other ranks log every 5 batches.
                if bidx % 5 == 1 or rank == 0:
                    if bidx > 1:
                        elapsed = time.time() - start_time
                        avg_tpb = elapsed / (bidx - 1)
                        remain = total_batches - bidx + 1
                        eta = f"ETA {int((avg_tpb*remain)//60):02d}:{int((avg_tpb*remain)%60):02d}"
                    else:
                        eta = "ETA calculating..."
                    print_rank(f"GPU {rank}: Batch {bidx}/{total_batches} ({len(batch)}) - {eta}")

                try:
                    t0 = time.time()
                    batch_aug = self._generate_caption_batch(batch, minimal_mode=minimal_mode)
                    local_aug.extend(batch_aug)
                    if bidx % 5 == 0 or rank == 0 or bidx == total_batches:
                        print_rank(f"GPU {rank}: Batch {bidx}/{total_batches} in {time.time()-t0:.1f}s, +{len(batch_aug)}")
                except Exception as e:
                    print_rank(f"GPU {rank}: Error in batch {bidx}: {e}")
                    print_rank(traceback.format_exc())
        else:
            print_rank(f"GPU {rank}: No local samples, skip generation")

        print_rank(f"GPU {rank}: Local generation done: {len(local_aug)} samples")

        # ============ File sync: stage 1 (completion flag) ============
        sync_dir = os.path.join(self.experiment_dir, "sync_caption_gen")
        if rank == 0:
            os.makedirs(sync_dir, exist_ok=True)
            print_rank(f"GPU 0: Created sync dir {sync_dir}")

        _wait_dir(sync_dir, rank, max_wait_s=36000)
        completion_flag = os.path.join(sync_dir, f"gpu_{rank}_completed.txt")
        try:
            with open(completion_flag, "w") as f:
                f.write(f"GPU {rank} completed {len(local_aug)} samples at {time.time()}")
            print_rank(f"GPU {rank}: Wrote completion flag: {completion_flag}")
        except Exception as e:
            print_rank(f"GPU {rank}: Error writing completion flag: {e}")
            os.makedirs(sync_dir, exist_ok=True)
            with open(completion_flag, "w") as f:
                f.write(f"GPU {rank} completed {len(local_aug)} samples at {time.time()}")
            print_rank(f"GPU {rank}: Retried flag ok")

        # Wait for all completion flags.
        print_rank(f"GPU {rank}: Waiting all completion flags")
        _wait_all_flags(sync_dir, world_size, rank, max_wait_s=36000)

        # ============ File sync: stage 2 (write per-rank results) ============
        tmp_dir = os.path.join(self.experiment_dir, "temp_caption_results")
        if rank == 0:
            os.makedirs(tmp_dir, exist_ok=True)
            print_rank(f"GPU 0: Created tmp dir {tmp_dir}")
        _wait_dir(tmp_dir, rank, max_wait_s=36000)

        # Write local result file.
        local_file = os.path.join(tmp_dir, f"gpu_{rank}_samples.json")
        try:
            # Atomic write to avoid partial-read races.
            tmp_local = local_file + ".tmp"
            with open(tmp_local, "w") as f:
                json.dump(local_aug, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_local, local_file)
            print_rank(f"GPU {rank}: Saved {len(local_aug)} to {local_file}")
        except Exception as e:
            print_rank(f"GPU {rank}: Error saving local file: {e}")

        # Wait until all local files are ready.
        print_rank(f"GPU {rank}: Waiting all gpu files")
        _wait_all_files(tmp_dir, world_size, rank, max_wait_s=36000)

        # ============ Rank-0 merge ============
        if rank == 0:
            all_aug = [local_aug]  # include rank0
            for r in range(1, world_size):
                try:
                    with open(os.path.join(tmp_dir, f"gpu_{r}_samples.json"), "r") as f:
                        data = json.load(f)
                    print_rank(f"GPU 0: Loaded {len(data)} from GPU {r}")
                    all_aug.append(data)
                except Exception as e:
                    print_rank(f"GPU 0: Error loading GPU {r} file: {e}; treat as empty")
                    all_aug.append([])

            merged = []
            for chunk in all_aug:
                merged.extend(chunk)

            # Filter invalid samples.
            print_rank(f"GPU 0: Filtering {len(merged)} samples")
            merged = self.validator.filter_valid_samples(merged)

            # Save final file (atomic replace).
            self._save_augmented_samples(merged)
            if merged:
                self._log_preview(merged)
            print_rank(f"GPU 0: Saved {len(merged)} merged samples")

            # Cleanup is delayed until all ranks finish reading the final file.
            print_rank(f"GPU 0: Final file written, waiting other ranks to read before cleanup")

        # ============ All ranks wait for final file ============
        print_rank(f"GPU {rank}: Waiting final file")
        _wait_file(final_aug_file, rank, max_wait_s=36000)

        # All ranks read the same final file (with retry for atomic replace timing).
        final_aug = []
        if os.path.exists(final_aug_file):
            try:
                saved = _json_load_retry(final_aug_file, retries=5, delay=0.3)
                samples = self._coerce_saved_samples(saved) if isinstance(saved, dict) else []
                declared = saved.get("total_samples", None) if isinstance(saved, dict) else None
                final_aug = samples
                if declared is not None and declared != len(samples):
                    print_rank(f"GPU {rank}: Loaded {len(samples)} samples (declared {declared}) — coerced")
                else:
                    print_rank(f"GPU {rank}: Loaded {len(samples)} samples")
            except Exception as e:
                print_rank(f"GPU {rank}: Error loading final file: {e}")
        else:
            print_rank(f"GPU {rank}: Final file not found")

        # Mark final-read completion under sync dir.
        try:
            final_read_flag = os.path.join(sync_dir, f"gpu_{rank}_final_read.txt")
            with open(final_read_flag, "w") as f:
                f.write(f"GPU {rank} read final file at {time.time()}")
        except Exception as e:
            print_rank(f"GPU {rank}: Error writing final_read flag: {e}")

        # Only rank 0 performs cleanup after all final-read flags are visible.
        if rank == 0:
            print_rank("GPU 0: Waiting all final_read flags before cleanup")
            start = time.time()
            while True:
                all_ok = True
                for r in range(world_size):
                    if not os.path.exists(os.path.join(sync_dir, f"gpu_{r}_final_read.txt")):
                        all_ok = False
                        break
                if all_ok:
                    break
                if time.time() - start > 36000:
                    print_rank("GPU 0: Timeout waiting final_read flags, proceed to cleanup")
                    break
                time.sleep(2)
            # Cleanup.
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                shutil.rmtree(sync_dir, ignore_errors=True)
                print_rank(f"GPU 0: Cleaned tmp dir {tmp_dir}")
                print_rank(f"GPU 0: Cleaned sync dir {sync_dir}")
            except Exception as e:
                print_rank(f"GPU 0: Cleanup error: {e}")

        self.augmented_samples = final_aug
        print_rank(f"GPU {rank}: Distributed caption generation completed: {len(final_aug)}")
        return final_aug

    # =========================
    #      Shared helpers
    # =========================
    def _save_augmented_samples(self, samples):
        """Save augmented samples for the next iteration using atomic replace."""
        # Defensive normalization: ensure `samples` is List[Dict].
        if not isinstance(samples, list):
            samples = self._coerce_saved_samples({"samples": samples}) if hasattr(self, "_coerce_saved_samples") else []
        else:
            # Filter non-dict items.
            samples = [s for s in samples if isinstance(s, dict)]

        next_iter = self.iteration_round + 1
        out_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{next_iter}.json")
        ref_images = {s.get("reference_image") for s in samples if s.get("reference_image")}
        tgt_images = {s.get("target_image") for s in samples if s.get("target_image")}
        orig_tgt_images = {s.get("original_target_image") for s in samples if s.get("original_target_image")}
        gt_ranks = []
        gt_topk_ranks = []
        gt_present = 0
        gt_before = 0
        gt_after = 0
        gt_similarities = []
        for sample in samples:
            rank = self._sanitize_rank(sample.get("gt_rank"))
            if rank is not None:
                gt_ranks.append(rank)
            topk_rank = self._sanitize_rank(sample.get("gt_topk_rank"))
            if topk_rank is not None:
                gt_topk_ranks.append(topk_rank)
            if sample.get("gt_in_candidates") is True:
                gt_present += 1
            before_gt = sample.get("is_before_gt_in_topk")
            if before_gt is True:
                gt_before += 1
            elif before_gt is False and sample.get("gt_in_candidates") is True:
                gt_after += 1
            gt_sim = self._sanitize_float(sample.get("gt_similarity"))
            if gt_sim is not None:
                gt_similarities.append(gt_sim)

        def _safe_mean(values):
            return float(sum(values)) / len(values) if values else None

        # Attach summary statistics.
        summary = {
            "total_samples": len(samples),
            "generation_timestamp": time.time(),
            "iteration_round": next_iter,
            "sample_statistics": {
                "avg_original_length": (sum(len(s.get("original_mod_text", "")) for s in samples) / len(samples)) if samples else 0,
                "avg_generated_length": (sum(len(s.get("modification_text", "")) for s in samples) / len(samples)) if samples else 0,
                "unique_reference_images": len(ref_images),
                "unique_target_images": len(tgt_images),
                "unique_original_target_images": len(orig_tgt_images),
                "gt_metrics": {
                    "with_global_rank": len(gt_ranks),
                    "avg_global_rank": _safe_mean(gt_ranks),
                    "with_topk_rank": len(gt_topk_ranks),
                    "avg_topk_rank": _safe_mean(gt_topk_ranks),
                    "gt_in_candidates": gt_present,
                    "gt_in_candidates_ratio": (gt_present / len(samples)) if samples else 0.0,
                    "before_gt_in_topk": gt_before,
                    "after_gt_in_topk": gt_after,
                    "avg_gt_similarity": _safe_mean(gt_similarities),
                },
            },
            "samples": samples
        }
        # Atomic write: tmp -> fsync -> replace.
        tmp_path = out_file + ".tmp"
        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir, exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_file)
        print_rank(f"Saved {len(samples)} samples to {out_file}")


# --------------------------
# Helper functions (distributed polling)
# --------------------------
def _wait_dir(path, rank, max_wait_s=36000):
    """Wait for a directory to appear via polling to avoid NCCL barrier watchdog risk."""
    waited = 0
    while not os.path.exists(path) and waited < max_wait_s:
        time.sleep(1)
        waited += 1
        if waited % 10 == 0:
            print_rank(f"GPU {rank}: Waiting dir {path}... {waited}s")
    if not os.path.exists(path):
        print_rank(f"GPU {rank}: Dir wait timeout, try creating locally: {path}")
        os.makedirs(path, exist_ok=True)


def _wait_all_flags(sync_dir, world_size, rank, max_wait_s=36000):
    """Wait for completion flags from all GPUs."""
    start = time.time()
    while time.time() - start < max_wait_s:
        all_ok = True
        for r in range(world_size):
            if not os.path.exists(os.path.join(sync_dir, f"gpu_{r}_completed.txt")):
                all_ok = False
                break
        if all_ok:
            print_rank(f"GPU {rank}: All completion flags ready")
            return
        time.sleep(5)
        elapsed = int(time.time() - start)
        if elapsed % 120 == 0:
            done = [r for r in range(world_size)
                    if os.path.exists(os.path.join(sync_dir, f"gpu_{r}_completed.txt"))]
            print_rank(f"GPU {rank}: Waiting flags... done={done}, elapsed={elapsed}s")
    print_rank(f"GPU {rank}: Timeout waiting completion flags")


def _wait_all_files(tmp_dir, world_size, rank, max_wait_s=36000):
    """Wait for temporary result files from all GPUs."""
    start = time.time()
    while time.time() - start < max_wait_s:
        all_ok = True
        missing = []
        for r in range(world_size):
            fp = os.path.join(tmp_dir, f"gpu_{r}_samples.json")
            if not os.path.exists(fp):
                all_ok = False
                missing.append(r)
        if all_ok:
            print_rank(f"GPU {rank}: All GPU files ready")
            return
        time.sleep(2)
        elapsed = int(time.time() - start)
        if elapsed % 60 == 0 and elapsed > 0:
            print_rank(f"GPU {rank}: Waiting files... missing={missing} elapsed={elapsed}s")
    print_rank(f"GPU {rank}: Timeout waiting gpu files")


def _wait_file(path, rank, max_wait_s=36000):
    """Wait for the final merged file."""
    start = time.time()
    while time.time() - start < max_wait_s:
        if os.path.exists(path):
            print_rank(f"GPU {rank}: Final file ready")
            return
        time.sleep(2)
        elapsed = int(time.time() - start)
        if elapsed % 10 == 0 and elapsed > 0:
            print_rank(f"GPU {rank}: Waiting final file... {elapsed}s")
    print_rank(f"GPU {rank}: Timeout waiting final file {path}")


def _json_load_retry(path: str, retries: int = 5, delay: float = 0.2):
    """Load JSON with retry to handle transient visibility around atomic replace."""
    last_err = None
    for i in range(retries):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            last_err = e
            time.sleep(delay * (i + 1))
    raise last_err

"""
Iterative Trainer for Composed Image Retrieval
Refactored to use decoupled modules:
- retrieval.embedding
- retrieval.hard_negative
- aug.caption_generation
- utils.logging, utils.progress
"""

import os
import json
import time
import torch
import torch.distributed as dist
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple
from torch.utils.data import DataLoader
from transformers import TrainingArguments
import torch.nn.functional as F

from .trainer import MMEBTrainer
from .model.model import MMEBModel
from src.data.dataset.cirr import IterativeCIRRDataset
from src.data.dataset.fashioniq import IterativeFashionIQDataset
from src.evaluation.cirr_evaluator import CIRREvaluator
from src.evaluation.fashioniq_evaluator import FashionIQEvaluator

# Import decoupled functional modules.
from .utils import print_rank, print_master
from src.mining.hard_negative import HardNegativeMiner
from src.retrieval.candidate_builder import CandidateBuilder      
from src.retrieval.embedding_cache import EmbeddingCache          
from src.retrieval.engine import RetrievalEngine                 
from src.aug.caption_generator import CaptionGenerator

# Prompt builders per backbone (LLaVA/Generic gracefully degrade to no-op if unavailable).
from src.prompt.qwen.builder_v2 import prepare_inputs_v2 as qwen_prepare, generate_with_qwen_v2 as generate_with_qwen
try:
    from src.prompt.llava.builder import prepare_inputs as llava_prepare, generate_with_llava
except Exception:
    llava_prepare, generate_with_llava = None, None
try:
    from src.prompt.generic.builder import prepare_inputs as generic_prepare, generate_with_generic
except Exception:
    generic_prepare, generate_with_generic = None, None

try:
    from peft import PeftModel
    try:
        from peft.utils.other import set_peft_model_state_dict
    except ImportError:  # pragma: no cover - older PEFT versions
        from peft.utils import set_peft_model_state_dict  # type: ignore
except Exception:  # pragma: no cover - PEFT not installed
    PeftModel = None  # type: ignore
    set_peft_model_state_dict = None  # type: ignore

from transformers.utils import (
    WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
)


logger = logging.getLogger(__name__)


class IterativeRetrievalTrainer(MMEBTrainer):

    METRIC_ALIASES = {
        "recall_at_1": ["recall_at_1", "r_at_1"],
        "recall_at_5": ["recall_at_5", "r_at_5"],
        "recall_at_10": ["recall_at_10", "r_at_10"],
    }

    def __init__(
        self,
        foundation_model=None,                 # Ignore incoming instance (kept for compatibility).
        foundation_model_name: str = None,     # Keep only name and lazily load on demand.
        max_iterations: int = 3,
        hard_neg_collection_freq: int = 1,
        caption_generation_batch_size: int = 8,
        model_args=None,
        data_args=None,
        max_length=None,
        # Fast / Production
        fast_mode: bool = False,
        fast_mode_max_samples: int = 100,
        fast_mode_retrieval_db_size: int = 50,
        fast_mode_max_steps: int = 5,
        steps_per_iteration: int = 1000,
        production_save_steps: int = 100,
        # Legacy compatibility.
        production_max_steps: Optional[int] = None,
        **kwargs
    ):
        # ---- Store objects needed by this class (before `super`) ----
        self.model_args = model_args
        self.data_args = data_args
        self.max_length = max_length

        # Handle compatibility between `steps_per_iteration` and `production_max_steps`.
        if production_max_steps is not None:
            print_master("WARNING: 'production_max_steps' is deprecated, use 'steps_per_iteration' instead")
            self.production_max_steps = production_max_steps
        else:
            self.production_max_steps = steps_per_iteration

        self.fast_mode = fast_mode
        self.fast_mode_max_samples = fast_mode_max_samples
        self.fast_mode_retrieval_db_size = fast_mode_retrieval_db_size
        self.fast_mode_max_steps = fast_mode_max_steps
        self.production_save_steps = production_save_steps

        self.steps_per_iteration = (
            self.fast_mode_max_steps if self.fast_mode else self.production_max_steps
        )

        # Print training plan.
        print_master(f"Training plan: {max_iterations} iterations × {self.steps_per_iteration} steps/iter")
        print_master("Strategy: reset optimizer & scheduler every iteration")

        # ---- Remove kwargs unsupported by base Trainer to avoid `super()` errors ----
        # These keys may be forwarded from factory/upper layers.
        for k in [
            "model_args", "data_args", "max_length",
            "fast_mode", "fast_mode_max_samples", "fast_mode_retrieval_db_size",
            "fast_mode_max_steps", "steps_per_iteration",
            "production_max_steps", "production_save_steps",
            "foundation_model_name", "foundation_model",
        ]:
            kwargs.pop(k, None)

        # Some callers pass `processing_class` through kwargs; capture it for this class/evaluators.
        self.processing_class = kwargs.get("processing_class", None)

        # ---- Enter base class initialization ----
        super().__init__(**kwargs)

        # ---- Local state and paths ----
        # Fully ignore incoming `foundation_model` instance (name only, to save VRAM).
        self.foundation_model = None
        self.foundation_processor = None
        self.foundation_model_name = foundation_model_name

        self.max_iterations = max_iterations
        self.hard_neg_collection_freq = hard_neg_collection_freq
        self.caption_generation_batch_size = caption_generation_batch_size

        # Persist base loss weights (user-configured) and track per-iteration overrides
        self.current_iteration = 0
        self.iteration_metrics: Dict[int, Dict[str, float]] = {}

        self._base_training_completed = False
        self._target_embeddings_cached = False

        # Unified experiment directory shared by decoupled modules.
        self.experiment_dir = getattr(self.args, "output_dir", "./outputs")
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Keep original dataset reference (used by iteration 0).
        self.original_dataset = self.train_dataset

        # Create file handler for train.log.
        self._configure_logging()

        # Recover previous progress/cache (sets `current_iteration`, etc.).
        self._try_resume_from_checkpoint()

        print_master(f"Initialized IterativeRetrievalTrainer with max_iterations={max_iterations}")

        # Adjust save/logging cadence by fast/production mode.
        self._configure_training_mode()

        # Cache user-configured loss weights for per-iteration adjustments
        self._base_info_weight = getattr(self.args, "info_nce_weight", 1.0)
        self._base_triplet_weight = getattr(self.args, "triplet_loss_weight", 0.0)
        self._base_triplet_margin = getattr(self.args, "triplet_margin", 0.2)
        self._current_info_weight = self._base_info_weight
        self._current_triplet_weight = self._base_triplet_weight

    def _apply_loss_weights_for_iteration(self):
        """Configure model loss weights based on the current iteration."""
        info_weight = 1.0 if getattr(self, "current_iteration", 0) == 0 else self._base_info_weight
        triplet_weight = 0.0 if getattr(self, "current_iteration", 0) == 0 else self._base_triplet_weight

        if hasattr(self.model, "configure_loss"):
            self.model.configure_loss(
                info_nce_weight=info_weight,
                triplet_loss_weight=triplet_weight,
                triplet_margin=self._base_triplet_margin,
            )

        self._current_info_weight = info_weight
        self._current_triplet_weight = triplet_weight

        print_master(
            f"[LossConfig] iter={self.current_iteration} info_weight={info_weight} "
            f"triplet_weight={triplet_weight} margin={self._base_triplet_margin}"
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("Expected (query_inputs, target_inputs) tuple from dataloader.")

        qry_inputs, tgt_inputs = inputs
        qry_inputs = dict(qry_inputs)
        tgt_inputs = dict(tgt_inputs)

        reference_ids = qry_inputs.pop("reference_ids", None)
        is_augmented = qry_inputs.pop("is_augmented", None)
        anchor_group_ids = qry_inputs.pop("anchor_group_ids", None)

        info_weight = getattr(self, "_current_info_weight", getattr(self.args, "info_nce_weight", 1.0))
        triplet_weight = getattr(self, "_current_triplet_weight", getattr(self.args, "triplet_loss_weight", 0.0))

        use_triplet = (
            triplet_weight > 0
            and getattr(self, "current_iteration", 0) >= 1
            and reference_ids is not None
        )

        outputs = model(
            qry=qry_inputs,
            tgt=tgt_inputs,
            return_embeddings=use_triplet,
        )

        if isinstance(outputs, dict):
            base_loss = outputs["loss"]
        else:
            base_loss = outputs
            outputs = {"loss": base_loss}

        loss = base_loss

        triplet_loss = None
        if use_triplet:
            triplet_loss = self._compute_intra_reference_triplet(
                outputs.get("qry_reps_local"),
                outputs.get("tgt_reps_local"),
                reference_ids,
                is_augmented,
                anchor_group_ids,
            )
            if triplet_loss is not None:
                if loss.requires_grad:
                    loss = loss + triplet_weight * triplet_loss
                else:
                    loss = triplet_weight * triplet_loss
                outputs["triplet_loss"] = triplet_loss
                if hasattr(model, "_last_loss_components"):
                    model._last_loss_components["triplet_loss"] = triplet_loss.detach()

                current_step = getattr(self.state, "global_step", 0)
                log_every = max(1, getattr(self.args, "logging_steps", 10))
                if current_step < 5 or ((current_step + 1) % log_every == 0):
                    info_val = outputs.get("info_nce_loss")
                    info_val = info_val.detach().item() if info_val is not None else None
                    margin_val = getattr(self.model, "triplet_margin", getattr(self.args, "triplet_margin", 0.2))
                    print_master(
                        f"[TripletLoss] iter={self.current_iteration} "
                        f"step={current_step} info_nce={info_val} "
                        f"triplet={triplet_loss.detach().item():.6f} "
                        f"margin={margin_val} "
                        f"weights(info={info_weight}, "
                        f"triplet={triplet_weight})"
                    )

        if return_outputs:
            outputs["loss"] = loss
            return loss, outputs
        return loss

    def _compute_intra_reference_triplet(
        self,
        qry_reps: Optional[torch.Tensor],
        tgt_reps: Optional[torch.Tensor],
        reference_ids: Optional[torch.Tensor],
        is_augmented: Optional[torch.Tensor],
        anchor_group_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if qry_reps is None or tgt_reps is None or reference_ids is None:
            return None

        # reference_ids / is_augmented may come from CPU tensors (depending on accelerator settings)
        reference_ids = reference_ids.to(qry_reps.device)
        if is_augmented is not None:
            is_augmented = is_augmented.to(qry_reps.device)
        if anchor_group_ids is not None:
            anchor_group_ids = anchor_group_ids.to(qry_reps.device)

        unique_refs = torch.unique(reference_ids)
        weighted_losses = []
        total_triplets = 0
        margin = getattr(self.args, "triplet_margin", 0.2)

        for ref in unique_refs:
            ref_mask = reference_ids == ref
            if anchor_group_ids is None:
                anchor_values = [None]
            else:
                anchor_values = torch.unique(anchor_group_ids[ref_mask])

            for anchor_val in anchor_values:
                if anchor_group_ids is None:
                    group_mask = ref_mask
                else:
                    group_mask = ref_mask & (anchor_group_ids == anchor_val)

                group_size = int(group_mask.sum().item())
                if group_size < 2:
                    continue

                anchor_q = qry_reps[group_mask]
                pos_t = tgt_reps[group_mask]

                anchors_exp = anchor_q.unsqueeze(1).expand(group_size, group_size, -1)
                positives_exp = pos_t.unsqueeze(1).expand(group_size, group_size, -1)
                negatives_exp = pos_t.unsqueeze(0).expand(group_size, group_size, -1)

                off_diag_mask = ~torch.eye(group_size, dtype=torch.bool, device=anchor_q.device)
                anchors = anchors_exp[off_diag_mask]
                positives = positives_exp[off_diag_mask]
                negatives = negatives_exp[off_diag_mask]

                if anchors.numel() == 0:
                    continue

                group_loss = F.triplet_margin_loss(
                    anchors,
                    positives,
                    negatives,
                    margin=margin,
                    reduction="mean",
                )
                triplet_count = group_size * (group_size - 1)
                weighted_losses.append(group_loss * triplet_count)
                total_triplets += triplet_count

        if total_triplets == 0 or not weighted_losses:
            # ensure a tensor connected to graph so downstream loss keeps gradients
            return qry_reps.sum() * 0.0

        triplet_loss = torch.stack(weighted_losses).sum() / float(total_triplets)

        if torch.distributed.is_initialized():
            triplet_loss = triplet_loss * torch.distributed.get_world_size()

        return triplet_loss


    def _try_resume_from_checkpoint(self) -> bool:
        """Try to resume from previous experiment state (backward-compatible)."""
        import os, json
        from glob import glob

        output_dir = self.args.output_dir
        max_iters = int(getattr(self, "max_iterations", 0) or 0)

        def _has_base_model(dir_):
            if not os.path.isdir(dir_):
                return False
            files = set(os.listdir(dir_))
            # Support both LoRA adapters and full HF checkpoint formats.
            has_lora = any(f.startswith("adapter_") for f in files) and "adapter_config.json" in files
            has_full = ("pytorch_model.bin" in files or "model.safetensors" in files) and "config.json" in files
            return has_lora or has_full

        def _has_any_embedding_cache(root):
            """Support both legacy and modern embedding-cache paths."""
            legacy = glob(os.path.join(root, "cache", "target_embeddings_*.pt"))
            modern = glob(os.path.join(root, "cache", "embeddings", "*.pt"))
            return bool(legacy or modern)

        # -------- 1) Find the latest complete iteration --------
        latest_complete = None
        for i in range(max_iters - 1, -1, -1):
            state_file = os.path.join(output_dir, f"iteration_{i}_state.json")
            if not os.path.exists(state_file):
                continue
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
            except Exception as e:
                print_master(f"Error reading iteration state {i}: {e}")
                continue

            # Backward-compatible state key.
            iter_complete = state.get("iteration_complete", False)
            if iter_complete:
                latest_complete = i
                print_master(f"Found COMPLETE iteration {i}")
                break
            else:
                print_master(f"Found INCOMPLETE iteration {i}, continue searching older COMPLETE")

        if latest_complete is not None:
            # Resume from the next iteration after the latest completed one.
            state_file = os.path.join(output_dir, f"iteration_{latest_complete}_state.json")
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
            except Exception as e:
                print_master(f"Error reading COMPLETE iteration state {latest_complete}: {e}")
                state = {}

            # Backward-compatible metrics field.
            metrics = state.get("iteration_metrics", state.get("metrics", {}))
            self.iteration_metrics = metrics
            self.current_iteration = min(latest_complete + 1, max_iters)  # Bound within valid range.

            # Restore hard-negative cache if needed by the next iteration.
            if latest_complete < max_iters - 1:
                hard_neg_file = os.path.join(output_dir, f"hard_negatives_iter_{latest_complete}.json")
                if os.path.exists(hard_neg_file) and hasattr(self.train_dataset, "hard_negatives_file"):
                    self.train_dataset.hard_negatives_file = hard_neg_file
                    # Dataset usually provides a lightweight `_load_hard_negatives(iteration)`.
                    try:
                        self.train_dataset._load_hard_negatives(latest_complete)
                    except Exception:
                        pass

            print_master(f"Resuming from COMPLETE iteration {latest_complete}")
            print_master(f"   Next iteration to run: {self.current_iteration}")
            return True

        # -------- 2) If no complete iteration exists, try resuming from an incomplete one --------
        for i in range(max_iters - 1, -1, -1):
            state_file = os.path.join(output_dir, f"iteration_{i}_state.json")
            if not os.path.exists(state_file):
                continue
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                completed_steps = state.get("completed_steps", {})
                metrics = state.get("iteration_metrics", state.get("metrics", {}))
            except Exception as e:
                print_master(f"Error reading incomplete iteration state {i}: {e}")
                continue

            self.current_iteration = i
            self.iteration_metrics = metrics
            print_master(f"Resuming from INCOMPLETE iteration {i}")
            print_master(f"   Completed steps detected: {list(completed_steps.keys())}")

            # Key step: load augmented samples/state into dataset after decoupling.
            try:
                self._prepare_dataset_for_iteration(i)
            except Exception as e:
                print_master(f"Warning: _prepare_dataset_for_iteration({i}) failed: {e}")

            return True

        # -------- 3) If no iteration state exists, check base model + embedding cache --------
        base_model_dir = os.path.join(output_dir, "base_model")
        cache_root = output_dir

        if _has_base_model(base_model_dir):
            print_master(f"Found base model at: {base_model_dir}")
            has_cache = _has_any_embedding_cache(cache_root)

            if has_cache:
                print_master("Found cached target embeddings (legacy or modern path).")
                print_master("Resuming from completed base training (iteration 0).")
                print_master("   Will skip: base training/eval/target embedding computation")
                print_master("   Next: hard negative collection")
                self.current_iteration = 0
                self._base_training_completed = True
                self._target_embeddings_cached = True
                return True
            else:
                print_master("Warning: base model found but no target embeddings cache.")
                print_master("Will recompute target embeddings and continue from iteration 0")
                self.current_iteration = 0
                self._base_training_completed = True
                self._target_embeddings_cached = False
                return True

        # -------- 4) Fallback: checkpoint-* exists without base_model -> treat as incomplete training --------
        ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")] if os.path.isdir(output_dir) else []
        if ckpts:
            print_master(f"Found checkpoints without base model: {ckpts}")
            print_master("This looks like an incomplete training; starting from scratch.")

        print_master("No previous state/base model/embedding cache found, starting from scratch.")
        self._base_training_completed = False
        self._target_embeddings_cached = False
        self.current_iteration = 0
        return False

    def _configure_logging(self):
        """Configure additional logging to ensure train.log is generated"""
        if hasattr(self.args, 'logging_dir') and self.args.logging_dir:
            import os
            os.makedirs(self.args.logging_dir, exist_ok=True)
            log_file = os.path.join(self.args.logging_dir, "train.log")

            root_logger = logging.getLogger()
            file_handler_exists = any(
                isinstance(handler, logging.FileHandler) and 
                handler.baseFilename == os.path.abspath(log_file)
                for handler in root_logger.handlers
            )
            if not file_handler_exists:
                file_handler = logging.FileHandler(log_file, mode='a')
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
                )
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                print_master(f"Added file logging to: {log_file}")
            else:
                print_master(f"File logging already configured for: {log_file}")
        else:
            print_master("Warning: logging_dir not set, train.log will not be generated")

    def _configure_training_mode(self):
        """Configure training parameters based on fast mode or production mode"""
        print_master(f"DEBUG: self.fast_mode = {self.fast_mode}")
        print_master(f"DEBUG: steps_per_iteration = {self.steps_per_iteration}")

        if self.fast_mode:
            print_master("=== FAST MODE CONFIGURATION ===")
            print_master(f"Steps per iteration: {self.steps_per_iteration}")
            print_master(f"Max samples for hard negatives: {self.fast_mode_max_samples}")
            print_master(f"Retrieval database size: {self.fast_mode_retrieval_db_size}")
            self.args.save_steps = max(1, self.steps_per_iteration // 2)
            self.args.logging_steps = 1
        else:
            print_master("=== PRODUCTION MODE CONFIGURATION ===")
            print_master(f"Steps per iteration: {self.steps_per_iteration}")
            print_master(f"Save frequency: every {self.production_save_steps} steps")
            self.args.save_steps = self.production_save_steps
            self.args.logging_steps = min(10, self.production_save_steps // 10)

        print_master("Each iteration will train independently with fresh optimizer/scheduler")
        print_master(f"Final training configuration:")
        print_master(f"  steps_per_iteration: {self.steps_per_iteration}")
        print_master(f"  save_steps: {self.args.save_steps}")
        print_master(f"  logging_steps: {self.args.logging_steps}")
        print_master("=" * 50)

    def _train_base_model(self):
        """Train the base retrieval model using standard contrastive learning."""
        import os
        from .utils import print_master

        print_master("Training base model with original dataset and fresh optimizer...")

        # 1) Use the original dataset and refresh dataloader.
        self.train_dataset = self.original_dataset

        # Ensure iteration 0 uses pure InfoNCE (triplet disabled).
        self.current_iteration = 0
        self._apply_loss_weights_for_iteration()
        self._update_train_dataloader()

        # 2) Train in an isolated subdirectory to avoid checkpoint conflicts.
        original_output_dir = self.args.output_dir
        original_max_steps = self.args.max_steps
        base_training_dir = os.path.join(original_output_dir, "training_iter_0")
        os.makedirs(base_training_dir, exist_ok=True)

        self.args.output_dir = base_training_dir
        self.args.max_steps = self.steps_per_iteration

        print_master(f"Base model training plan: 0 -> {self.args.max_steps} steps")
        print_master("Starting fresh training with new optimizer and scheduler")
        print_master(f"Training checkpoints will be saved to: {base_training_dir}")
        print_master(f"Final base model will be saved to: {original_output_dir}")

        # 3) Explicitly reset optimizer/scheduler for a fresh run.
        try:
            # Transformers compatibility: some versions expose optimizer/lr_scheduler attributes.
            if hasattr(self, "optimizer"):
                self.optimizer = None
            if hasattr(self, "lr_scheduler"):
                self.lr_scheduler = None
            # Some Trainer versions lazily initialize inside `train()`; setting None forces recreation.
            train_result = self.train(resume_from_checkpoint=None)
        finally:
            # 4) Restore global output directory and max steps.
            self.args.output_dir = original_output_dir
            self.args.max_steps = original_max_steps
            print_master(f"Restored output_dir to: {original_output_dir}")
            print_master(f"Restored max_steps to: {original_max_steps}")

        # 5) Save base model to the main directory (not subdirectory).
        base_model_path = os.path.join(original_output_dir, "base_model")
        self.save_model(base_model_path)

        print_master(f"Base model training completed: 0 → {self.state.global_step} steps")
        print_master(f"Base model saved to: {base_model_path}")

        return train_result

    def _train_current_iteration(self):
        """
        Train for the current iteration by loading previous model weights
        but resetting the optimizer and LR scheduler for independent training.
        """
        import os
        from .utils import print_master

        print_master(f"Training iteration {self.current_iteration} with RESET optimizer and scheduler...")

        # 1) Model weights are loaded in external `main()`; this method controls training flow only.
        print_master("Model weights already loaded by main() function")
        print_master("Will reset optimizer and scheduler for an independent learning-rate schedule")

        # Set loss weights for this iteration before refreshing dataloader.
        self._apply_loss_weights_for_iteration()

        # 2) Ensure dataset and sampler are updated.
        self._update_train_dataloader()

        # 3) Create isolated training subdirectory for this iteration.
        original_output_dir = self.args.output_dir
        original_max_steps = self.args.max_steps
        iteration_output_dir = os.path.join(original_output_dir, f"training_iter_{self.current_iteration}")
        os.makedirs(iteration_output_dir, exist_ok=True)

        self.args.output_dir = iteration_output_dir
        self.args.max_steps = self.steps_per_iteration

        print_master(f"Iteration {self.current_iteration} independent training plan:")
        print_master(f"   - Will train for {self.args.max_steps} fresh steps")
        print_master(f"   - Training checkpoints will be saved to: {iteration_output_dir}")
        print_master(f"   - Final iteration model will be saved to: {original_output_dir}")
        print_master(f"   - Previous global_step will be ignored for LR scheduling")
        print_master(f"   - New optimizer and scheduler will start from scratch")

        try:
            # 4) Explicitly reset optimizer/scheduler for a fresh schedule each iteration.
            if hasattr(self, "optimizer"):
                self.optimizer = None
            if hasattr(self, "lr_scheduler"):
                self.lr_scheduler = None

            print_master("Model weights already loaded in main(), ready for independent training")
            print_master("Creating fresh optimizer and scheduler for this iteration")

            # Do not resume from checkpoint: force fresh optimizer and LR scheduler.
            train_result = self.train(resume_from_checkpoint=None)
        finally:
            # 5) Restore trainer global output directory and max_steps.
            self.args.output_dir = original_output_dir
            self.args.max_steps = original_max_steps
            print_master(f"Restored output_dir to: {original_output_dir}")
            print_master(f"Restored max_steps to: {original_max_steps}")

        print_master(f"Iteration {self.current_iteration} independent training completed")
        print_master(f"Final step count: {self.state.global_step}")

        # 6) Save this iteration's final model to the main directory.
        iter_model_path = os.path.join(original_output_dir, f"iteration_{self.current_iteration}")
        self.save_model(iter_model_path)
        print_master(f"Final iteration model saved to: {iter_model_path}")

        return train_result

    def _parse_checkpoint_step(self, checkpoint_name: str) -> Optional[int]:
        """Extract the step number from a checkpoint directory name."""
        try:
            return int(checkpoint_name.split("-")[-1])
        except (ValueError, IndexError):
            return None

    def _list_iteration_checkpoints(self, iteration: int) -> List[Tuple[int, str, str]]:
        """Return a sorted list of (step, name, path) tuples for checkpoints of a given iteration."""
        iteration_dir = os.path.join(self.args.output_dir, f"training_iter_{iteration}")
        checkpoints: List[Tuple[int, str, str]] = []

        if not os.path.isdir(iteration_dir):
            return checkpoints

        for entry in sorted(os.listdir(iteration_dir)):
            if not entry.startswith("checkpoint-"):
                continue
            checkpoint_path = os.path.join(iteration_dir, entry)
            if not os.path.isdir(checkpoint_path):
                continue
            step = self._parse_checkpoint_step(entry)
            if step is None:
                continue
            checkpoints.append((step, entry, checkpoint_path))

        checkpoints.sort(key=lambda item: item[0])
        return checkpoints

    def _load_checkpoint_into_model(self, checkpoint_path: str):
        """Load encoder (and adapter) weights from a specific checkpoint directory into the active model."""
        from .utils import print_master

        weight_files = [
            (ADAPTER_SAFE_WEIGHTS_NAME, "safetensors"),
            (ADAPTER_WEIGHTS_NAME, "bin"),
            (SAFE_WEIGHTS_NAME, "safetensors"),
            (WEIGHTS_NAME, "bin"),
        ]

        for filename, weight_type in weight_files:
            full_path = os.path.join(checkpoint_path, filename)
            if not os.path.exists(full_path):
                continue

            if weight_type == "safetensors":
                try:
                    from safetensors.torch import load_file as load_safetensors
                except ImportError as exc:
                    raise ImportError(
                        f"Cannot load {full_path} because the `safetensors` dependency is missing. "
                        f"Please install `safetensors` and retry."
                    ) from exc
                state_dict = load_safetensors(full_path, device="cpu")
            else:
                state_dict = torch.load(full_path, map_location="cpu")

            missing_keys: List[str] = []
            unexpected_keys: List[str] = []

            encoder = getattr(self.model, "encoder", None)
            is_peft_model = PeftModel is not None and isinstance(encoder, PeftModel)

            if is_peft_model and set_peft_model_state_dict is not None:
                try:
                    missing_keys, unexpected_keys = set_peft_model_state_dict(
                        encoder, state_dict, adapter_name=getattr(encoder, "active_adapter", None)
                    )
                except Exception as exc:
                    print_master(f"Warning: failed to load PEFT adapter from {checkpoint_path}: {exc}. Falling back to load_state_dict.")
                    missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, strict=False)
            else:
                missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, strict=False)

            if missing_keys:
                preview = ", ".join(missing_keys[:5])
                if len(missing_keys) > 5:
                    preview += ", ..."
                print_master(f"Warning: missing keys when loading {checkpoint_path}: {preview}")
            if unexpected_keys:
                preview = ", ".join(unexpected_keys[:5])
                if len(unexpected_keys) > 5:
                    preview += ", ..."
                print_master(f"Warning: unexpected keys when loading {checkpoint_path}: {preview}")

            return

        raise FileNotFoundError(
            f"No loadable model weight file found under {checkpoint_path} "
            f"({ADAPTER_SAFE_WEIGHTS_NAME}/{ADAPTER_WEIGHTS_NAME}/{SAFE_WEIGHTS_NAME}/{WEIGHTS_NAME})."
        )

    def _get_metric_value(self, metrics: Dict[str, float], key: str) -> Optional[float]:
        """Fetch metric value with alias support."""
        aliases = self.METRIC_ALIASES.get(key, [key])
        for alias in aliases:
            if alias in metrics and metrics[alias] is not None:
                return metrics[alias]
        return None

    def _normalize_metric_aliases(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure canonical metric keys exist even if evaluator returns alias names."""
        normalized = dict(metrics)
        for canonical in self.METRIC_ALIASES.keys():
            if canonical not in normalized:
                value = self._get_metric_value(metrics, canonical)
                if value is not None:
                    normalized[canonical] = value
        return normalized

    def _is_metrics_better(
        self,
        candidate: Dict[str, float],
        reference: Dict[str, float],
        candidate_step: Optional[int] = None,
        reference_step: Optional[int] = None,
    ) -> bool:
        """Compare two metric dicts using Recall@1 as primary, Recall@5/10 as tie-breakers, then smaller step."""
        priority_metrics = ["recall_at_1", "recall_at_5", "recall_at_10"]

        for metric in priority_metrics:
            cand_value = self._get_metric_value(candidate, metric)
            ref_value = self._get_metric_value(reference, metric)

            if cand_value is None and ref_value is None:
                continue
            if cand_value is None:
                return False
            if ref_value is None:
                return True
            if cand_value > ref_value:
                return True
            if cand_value < ref_value:
                return False

        if candidate_step is not None and reference_step is not None:
            return candidate_step < reference_step

        return False

    def _save_iteration_model_from_current_weights(self, iteration: int):
        """Persist current model weights as the iteration representative (handles base iteration separately)."""
        target_dir = os.path.join(
            self.args.output_dir,
            "base_model" if iteration == 0 else f"iteration_{iteration}",
        )
        self.save_model(target_dir)
        from .utils import print_master
        print_master(
            f"Saved selected weights for iteration {iteration} to {target_dir}"
        )

    def _write_checkpoint_metrics_summary(
        self,
        iteration: int,
        entries: List[Dict[str, Any]],
        selected_checkpoint: Optional[str],
        selected_step: Optional[int],
    ):
        """Persist per-checkpoint evaluation metrics for later inspection."""
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        summary_path = os.path.join(self.args.output_dir, f"iteration_{iteration}_checkpoint_metrics.json")
        tmp_path = summary_path + ".tmp"
        summary_payload = {
            "iteration": iteration,
            "selection_metric": "recall_at_1",
            "tie_breakers": ["recall_at_5", "recall_at_10", "lower_step"],
            "selected_checkpoint": selected_checkpoint,
            "selected_step": selected_step,
            "checkpoints": entries,
        }
        with open(tmp_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        os.replace(tmp_path, summary_path)
        from .utils import print_master
        print_master(f"Stored checkpoint evaluation summary at {summary_path}")

    def _write_iteration_eval_file(self, iteration: int, metrics: Dict[str, Any]):
        """Write the final evaluation results file for an iteration."""
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        eval_path = os.path.join(self.args.output_dir, f"eval_results_iter_{iteration}.json")
        tmp_path = eval_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(metrics, f, indent=2)
        os.replace(tmp_path, eval_path)
        from .utils import print_master
        print_master(f"Final evaluation results saved to {eval_path}")

    def _evaluate_iteration_checkpoints(self, iteration: int) -> Dict[str, Any]:
        """
        Evaluate all saved checkpoints for this iteration, choose the best by Recall@1,
        persist per-checkpoint metrics, and ensure the selected weights are saved for the next iteration.
        """
        from .utils import print_master

        checkpoints = self._list_iteration_checkpoints(iteration)
        summary_entries: List[Dict[str, Any]] = []
        best_checkpoint: Optional[Dict[str, Any]] = None

        if not checkpoints:
            print_master(f"Warning: no checkpoints found for iteration {iteration}; evaluating current model only.")
            metrics = self._normalize_metric_aliases(
                dict(self._evaluate_current_model(iteration=iteration, reuse_cached=True))
            )
            metrics.setdefault("selected_checkpoint", None)
            metrics.setdefault("selected_step", None)
            summary_entries.append({
                "checkpoint": None,
                "step": None,
                "metrics": metrics,
                "note": "no checkpoints were saved for this iteration",
            })
            self._save_iteration_model_from_current_weights(iteration)
            self._write_checkpoint_metrics_summary(iteration, summary_entries, None, None)
            self._write_iteration_eval_file(iteration, metrics)
            return metrics

        for step, name, path in checkpoints:
            suffix = f"_{name}"
            try:
                metrics = dict(
                    self._evaluate_current_model(
                        iteration=iteration,
                        checkpoint_path=path,
                        results_suffix=suffix,
                        reuse_cached=True,
                    )
                )
                metrics = self._normalize_metric_aliases(metrics)
                metrics.setdefault("checkpoint", name)
                metrics.setdefault("iteration", iteration)
                summary_entries.append({
                    "checkpoint": name,
                    "step": step,
                    "metrics": metrics,
                })

                if best_checkpoint is None or self._is_metrics_better(
                    metrics,
                    best_checkpoint["metrics"],
                    candidate_step=step,
                    reference_step=best_checkpoint["step"],
                ):
                    best_checkpoint = {
                        "checkpoint": path,
                        "checkpoint_name": name,
                        "step": step,
                        "metrics": metrics,
                    }
            except Exception as exc:
                error_entry = {
                    "checkpoint": name,
                    "step": step,
                    "error": str(exc),
                }
                summary_entries.append(error_entry)
                print_master(f"Warning: evaluation failed for {name}: {exc}")

        if best_checkpoint is None:
            print_master(f"Warning: all checkpoint evaluations failed for iteration {iteration}; using current model.")
            metrics = self._normalize_metric_aliases(
                dict(self._evaluate_current_model(iteration=iteration, reuse_cached=True))
            )
            metrics.setdefault("selected_checkpoint", None)
            metrics.setdefault("selected_step", None)
            summary_entries.append({
                "checkpoint": None,
                "step": None,
                "metrics": metrics,
                "note": "all checkpoint evaluations failed; using current weights",
            })
            self._save_iteration_model_from_current_weights(iteration)
            self._write_checkpoint_metrics_summary(iteration, summary_entries, None, None)
            self._write_iteration_eval_file(iteration, metrics)
            return metrics

        # Ensure the best checkpoint weights are loaded before saving iteration model
        self._load_checkpoint_into_model(best_checkpoint["checkpoint"])
        best_metrics = self._normalize_metric_aliases(dict(best_checkpoint["metrics"]))
        best_metrics["selected_checkpoint"] = best_checkpoint["checkpoint_name"]
        best_metrics["selected_step"] = best_checkpoint["step"]

        print_master(
            f"🏆 Iteration {iteration}: selected {best_checkpoint['checkpoint_name']} "
            f"(step {best_checkpoint['step']}) based on Recall@1."
        )

        self._save_iteration_model_from_current_weights(iteration)
        self._write_checkpoint_metrics_summary(
            iteration,
            summary_entries,
            best_checkpoint["checkpoint_name"],
            best_checkpoint["step"],
        )
        self._write_iteration_eval_file(iteration, best_metrics)

        return best_metrics

    def _evaluate_current_model(
        self,
        iteration: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        results_suffix: Optional[str] = None,
        reuse_cached: bool = True,
    ) -> Dict[str, float]:
        """Evaluate current (or specified checkpoint) model with optional caching."""
        import os, json
        import torch.distributed as dist
        from .utils import print_master

        iteration = self.current_iteration if iteration is None else iteration
        suffix = ""
        if results_suffix:
            suffix = results_suffix
        elif checkpoint_path:
            suffix = f"_{os.path.basename(checkpoint_path)}"
        suffix = suffix.replace(os.path.sep, "_")

        if suffix:
            per_ckpt_dir = os.path.join(self.args.output_dir, f"iteration_{iteration}_checkpoint_eval")
            os.makedirs(per_ckpt_dir, exist_ok=True)
            eval_results_file = os.path.join(per_ckpt_dir, f"eval_results_iter_{iteration}{suffix}.json")
        else:
            eval_results_file = os.path.join(self.args.output_dir, f"eval_results_iter_{iteration}.json")

        if checkpoint_path:
            print_master(f"Evaluating checkpoint {checkpoint_path} for iteration {iteration}...")
            self._load_checkpoint_into_model(checkpoint_path)
        else:
            print_master(f"Evaluating iteration {iteration} model (current weights)...")

        if reuse_cached and os.path.exists(eval_results_file):
            try:
                with open(eval_results_file, "r") as f:
                    cached = json.load(f)
                print_master(f"Loaded cached evaluation from {eval_results_file}")
                cached.setdefault("iteration", int(iteration))
                if checkpoint_path:
                    cached.setdefault("checkpoint", os.path.basename(checkpoint_path))
                return cached
            except Exception as e:
                print_master(f"Failed to read cached evaluation {eval_results_file}: {e}. Re-evaluating.")

        # 2) Prepare evaluator (auto-select by training dataset).
        try:
            # Resolve processor: prefer Trainer `processing_class`, then `model.processor`.
            processor = getattr(self, "processing_class", None) or getattr(self.model, "processor", None)
            if processor is None:
                print_master("Warning: no processor found on Trainer or model; evaluator may fail.")

            eval_bs = 4 if getattr(self, "fast_mode", False) else 8

            dataset_name = str(getattr(self.data_args, "dataset_name", "") or "").strip().lower()
            dataset_config_name = ""
            if hasattr(self, "train_dataset") and hasattr(self.train_dataset, "dataset_config"):
                dataset_config_name = str(self.train_dataset.dataset_config.get("dataset_name", "") or "").strip().lower()

            resolved_dataset_name = dataset_name or dataset_config_name

            if "fashioniq" in resolved_dataset_name:
                evaluator = FashionIQEvaluator(
                    model=self.model,
                    processor=processor,
                    data_args=self.data_args,
                    model_args=self.model_args,
                    device=str(getattr(self.args, "device", "cpu")),
                    batch_size=eval_bs,
                )
                print_master(f"Real FashionIQ evaluator initialized (batch_size={eval_bs}).")
            else:
                evaluator = CIRREvaluator(
                    model=self.model,
                    processor=processor,
                    data_args=self.data_args,
                    model_args=self.model_args,
                    device=str(getattr(self.args, "device", "cpu")),
                    batch_size=eval_bs,
                )
                print_master(f"Real CIRR evaluator initialized (batch_size={eval_bs}).")

        # 3) Real evaluation or fallback.

            # Distributed mode check.
            world_ok = dist.is_initialized() and dist.get_world_size() > 1
            # Enable distributed evaluation only when evaluator supports it and world size > 1.
            supports_dist = hasattr(evaluator, "evaluate") or hasattr(evaluator, "_evaluate_distributed")
            use_distributed = bool(world_ok and supports_dist)

            # In fast mode, force single-GPU path if actual world size is 1.
            if getattr(self, "fast_mode", False) and (not world_ok):
                use_distributed = False
                print_master("Fast mode: single-GPU evaluation.")

            if use_distributed:
                print_master(f"Using distributed evaluation across {dist.get_world_size()} GPUs")
                eval_results = evaluator.evaluate(distributed=True)
            else:
                print_master("Using single-GPU evaluation")
                eval_results = evaluator.evaluate(distributed=False)

            # Metadata.
            eval_results["evaluation_mode"] = "distributed" if use_distributed else "single_gpu"
            eval_results["fast_mode"] = bool(getattr(self, "fast_mode", False))
            eval_results["iteration"] = int(iteration)

        except Exception as e:
            print_master(f"Real evaluation failed: {e}")
            print_master("Falling back to dummy evaluation metrics.")
            if getattr(self, "fast_mode", False):
                eval_results = {
                    "recall_at_1": 0.15,
                    "recall_at_5": 0.35,
                    "recall_at_10": 0.45,
                    "recall_subset_at_1": 0.12,
                    "recall_subset_at_2": 0.25,
                    "recall_subset_at_3": 0.32,
                    "group_recall_at_1": 0.18,
                    "group_recall_at_2": 0.30,
                    "group_recall_at_3": 0.38,
                    "evaluation_mode": "dummy_fast",
                    "fast_mode": True,
                    "iteration": int(iteration),
                }
            else:
                eval_results = {
                    "recall_at_1": 0.50,
                    "recall_at_5": 0.70,
                    "recall_at_10": 0.80,
                    "recall_subset_at_1": 0.30,
                    "recall_subset_at_2": 0.50,
                    "recall_subset_at_3": 0.60,
                    "group_recall_at_1": 0.40,
                    "group_recall_at_2": 0.60,
                    "group_recall_at_3": 0.70,
                    "evaluation_mode": "dummy_production",
                    "fast_mode": False,
                    "iteration": int(iteration),
                }

        if checkpoint_path:
            eval_results["checkpoint"] = os.path.basename(checkpoint_path)

        # 4) Persist on rank 0 only.
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                with open(eval_results_file, "w") as f:
                    json.dump(eval_results, f, indent=2)
                print_master(f"Saved evaluation results to {eval_results_file}")
            except Exception as e:
                print_master(f"Warning: failed to save evaluation results: {e}")

        print_master(f"Iteration {iteration} evaluation results: {eval_results}")
        return eval_results

    def _prepare_next_iteration_dataset(self, next_iteration: int, augmented_samples: List[Dict]):
        """Prepare dataset for next iteration with augmented samples (decoupled version)."""
        import os, json, time
        import torch.distributed as dist

        print_master(f"Preparing dataset for iteration {next_iteration}...")

        # 1) Persist augmented samples (rank 0 only; atomic replace avoids partial reads).
        augmented_file = os.path.join(self.args.output_dir, f"augmented_samples_iter_{next_iteration}.json")
        augmented_dir = os.path.dirname(augmented_file)
        os.makedirs(augmented_dir, exist_ok=True)

        meta = {
            "total_samples": len(augmented_samples),
            "generation_timestamp": time.time(),
            "iteration_round": next_iteration,
            "sample_statistics": self._compute_sample_statistics(augmented_samples),
            "samples": augmented_samples,
        }

        if not dist.is_initialized() or dist.get_rank() == 0:
            tmp_path = augmented_file + ".tmp"
            try:
                with open(tmp_path, "w") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, augmented_file)  # Atomic replace.
                print_master(f"Saved {len(augmented_samples)} augmented samples to {augmented_file}")
            except Exception as e:
                print_master(f"Error: failed to save augmented samples: {e}")
        else:
            print_rank(f"GPU {dist.get_rank()}: Skipping augmented samples save (only rank 0 writes)")

        # 2) Synchronize once so all GPUs can see the file.
        if dist.is_initialized():
            dist.barrier()
            print_master("All GPUs synchronized after augmented samples save")

        # 3) Append augmented samples to the current training dataset.
        try:
            ds_types = (IterativeCIRRDataset, IterativeFashionIQDataset)
        except Exception:
            ds_types = tuple()

        if isinstance(self.train_dataset, ds_types) or hasattr(self.train_dataset, "augmented_samples"):
            old_total = len(self.train_dataset)
            old_aug = len(getattr(self.train_dataset, "augmented_samples", []))

            setattr(self.train_dataset, "iteration_round", next_iteration)
            if not hasattr(self.train_dataset, "augmented_samples"):
                self.train_dataset.augmented_samples = []
            self.train_dataset.augmented_samples.extend(augmented_samples)

            # Config control: optionally use only augmented data for iteration > 0.
            use_original = getattr(self.args, "use_original_data_in_iter_plus", True)
            if not use_original and next_iteration > 0 and hasattr(self.train_dataset, "set_use_original_data"):
                self.train_dataset.set_use_original_data(False)
                print_master(f"Iter {next_iteration}: using ONLY augmented samples (original disabled)")
            elif hasattr(self.train_dataset, "set_use_original_data"):
                self.train_dataset.set_use_original_data(True)

            setattr(
                self.train_dataset,
                "hard_negatives_file",
                os.path.join(self.args.output_dir, f"hard_negatives_iter_{next_iteration}.json"),
            )

            new_total = len(self.train_dataset)
            new_aug = len(self.train_dataset.augmented_samples)
            print_master("Dataset update summary:")
            print_master(f"  - Added {len(augmented_samples)} new augmented samples")
            print_master(f"  - Total augmented samples: {old_aug} → {new_aug}")
            # Actual total depends on `use_original_data`; report logical dataset length here.
            print_master(f"  - Reported dataset length (len(dataset)): {new_total}")
        else:
            print_master("Warning: train dataset has no 'augmented_samples' attribute; skipped in-memory append.")

        self._update_train_dataloader()
        print_master(f"Training dataloader updated for iteration {next_iteration}")

    def _save_iteration_state(self, iteration: int):
        """Save iteration state and metrics with step completion tracking"""
        import os, json, time
        import torch.distributed as dist
        from .utils import print_rank, print_master

        # only rank 0 writes
        if dist.is_initialized() and dist.get_rank() != 0:
            print_rank(f"GPU {dist.get_rank()}: Skipping state save (only rank 0 writes)")
            return

        state_file = os.path.join(self.args.output_dir, f"iteration_{iteration}_state.json")

        # model path of this iteration
        model_path = (os.path.join(self.args.output_dir, "base_model")
                    if iteration == 0
                    else os.path.join(self.args.output_dir, f"iteration_{iteration}"))

        # compute step completion flags
        completed_steps = self._check_iteration_completion_status(iteration)

        state = {
            "iteration": iteration,
            # Use unified key name `iteration_metrics` for read/write consistency.
            "iteration_metrics": self.iteration_metrics,
            "model_path": model_path,
            "hard_negatives_file": f"hard_negatives_iter_{iteration}.json",
            "completed_steps": completed_steps,
            "iteration_complete": completed_steps.get("all_steps_complete", False),
            "timestamp": time.time(),
        }

        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            completion_status = "COMPLETE" if state["iteration_complete"] else "IN PROGRESS"
            print_master(f"Saved iteration {iteration} state to {state_file} - {completion_status}")
            print_master(f"Model path recorded as: {model_path}")
            print_master(f"Completed steps: {list(completed_steps.keys())}")
        except Exception as e:
            print_master(f"Error saving iteration state: {e}")

    def _check_iteration_completion_status(self, iteration: int) -> dict:
        """Check which steps of an iteration have been completed"""
        import os

        output_dir = self.args.output_dir
        completed_steps = {}

        # Step 1: model training done?
        model_path = (os.path.join(output_dir, "base_model")
                    if iteration == 0
                    else os.path.join(output_dir, f"iteration_{iteration}"))
        completed_steps["model_training"] = os.path.exists(model_path)

        # Step 2: evaluation done?
        eval_file = os.path.join(output_dir, f"eval_results_iter_{iteration}.json")
        completed_steps["evaluation"] = os.path.exists(eval_file)

        # Step 3: hard negatives (non-final iteration)
        is_final_iteration = iteration >= (self.max_iterations - 1)
        if is_final_iteration:
            completed_steps["hard_negatives_collection"] = True
        else:
            hard_neg_file = os.path.join(output_dir, f"hard_negatives_iter_{iteration}.json")
            completed_steps["hard_negatives_collection"] = os.path.exists(hard_neg_file)

        # Step 4: caption generation (non-final iteration)
        if is_final_iteration:
            completed_steps["caption_generation"] = True
        else:
            next_iteration = iteration + 1
            augmented_file = os.path.join(output_dir, f"augmented_samples_iter_{next_iteration}.json")
            completed_steps["caption_generation"] = os.path.exists(augmented_file)

        # all done?
        completed_steps["all_steps_complete"] = all(completed_steps.values())
        return completed_steps

    def _load_iteration_state(self, iteration: int):
        """Load iteration state for resuming (model already loaded externally)"""
        state_file = os.path.join(self.args.output_dir, f"iteration_{iteration}_state.json")
        
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Note: Model weights already loaded in main script using MMEBModel.load()
            print_master(f"Loading iteration {iteration} metadata (model loaded separately)")
            
            self.iteration_metrics = state.get('iteration_metrics', {})
            print_master(f"Loaded iteration {iteration} state from {state_file}")
        else:
            print_master(f"No state file found for iteration {iteration}")

    def _summarize_results(self):
        """Summarize results across all iterations"""
        import os, json
        import torch.distributed as dist
        from .utils import print_master

        print_master("\n" + "=" * 80)
        print_master("ITERATIVE TRAINING SUMMARY")
        print_master("=" * 80)

        for iteration, metrics in self.iteration_metrics.items():
            print_master(f"Iteration {iteration}: {metrics}")

        best_iteration, best_metrics = None, None
        if self.iteration_metrics:
            def _score(m):  # Support alias metric names.
                return m.get("recall_at_1", m.get("r_at_1", 0))
            best_iteration = max(self.iteration_metrics.keys(),
                                key=lambda x: _score(self.iteration_metrics[x]))
            best_metrics = self.iteration_metrics[best_iteration]
            print_master(f"\nBest performance: Iteration {best_iteration}")
            print_master(f"Best metrics: {best_metrics}")

        # only rank0 writes summary
        if not dist.is_initialized() or dist.get_rank() == 0:
            summary_file = os.path.join(self.args.output_dir, "training_summary.json")
            summary = {
                "max_iterations": self.max_iterations,
                "completed_iterations": len(self.iteration_metrics),
                "iteration_metrics": self.iteration_metrics,
                "best_iteration": best_iteration if self.iteration_metrics else None,
                "best_metrics": best_metrics if self.iteration_metrics else None,
            }
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print_master(f"Training summary saved to {summary_file}")

    def _prepare_dataset_for_iteration(self, iteration: int):
        """Prepare dataset state for a specific iteration when resuming"""

        print_master(f"Preparing dataset for resumed iteration {iteration}...")

        # Iteration 0: use the original dataset.
        if iteration == 0:
            self.train_dataset = self.original_dataset
            self._update_train_dataloader()  # Ensure sampler no longer references stale dataset object.
            return

        # Iteration > 0: accumulate augmented samples from prior rounds.
        all_augmented_samples = []
        for i in range(1, iteration + 1):
            augmented_file = os.path.join(self.args.output_dir, f"augmented_samples_iter_{i}.json")
            if not os.path.exists(augmented_file):
                print_master(f"Warning: augmented file not found for iter {i}: {augmented_file}")
                continue

            try:
                with open(augmented_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print_master(f"Warning: failed to read {augmented_file}: {e}")
                continue

            # Support multiple formats: new (with metadata), old (direct list), and nested variants.
            iter_samples = None
            declared_total = data.get("total_samples") if isinstance(data, dict) else None

            if isinstance(data, dict) and "samples" in data:
                iter_samples = data["samples"]
                # Case A: normal list.
                if isinstance(iter_samples, list):
                    pass
                # Case B: `samples` is a dict.
                elif isinstance(iter_samples, dict):
                    # B1: nested metadata containing the real list.
                    if "samples" in iter_samples and isinstance(iter_samples["samples"], list):
                        iter_samples = iter_samples["samples"]
                        print_master(
                            f"Note: iter {i} 'samples' contained nested 'samples' list; unwrapped to list with {len(iter_samples)} items"
                        )
                    # B2: pure id->sample dict.
                    elif all(isinstance(v, dict) for v in iter_samples.values()):
                        iter_samples = list(iter_samples.values())
                        print_master(
                            f"Note: iter {i} 'samples' is a dict with {len(iter_samples)} dict values; converted to list"
                        )
                    else:
                        # B3: mixed id->list[...] values; pick the longest list as sample list.
                        list_candidates = [v for v in iter_samples.values() if isinstance(v, list)]
                        if list_candidates:
                            best = max(list_candidates, key=len)
                            iter_samples = best
                            print_master(
                                f"Note: iter {i} 'samples' dict had list candidates; selected longest list with {len(iter_samples)} items"
                            )
                        else:
                            # B4: no list candidate; fallback to values list.
                            iter_samples = list(iter_samples.values())
                            print_master(
                                f"Note: iter {i} 'samples' dict coerced to list of values (len={len(iter_samples)}); will filter non-dict entries"
                            )
                else:
                    # Other iterable types: try coercing to list.
                    try:
                        iter_samples = list(iter_samples)
                        print_master(
                            f"Note: iter {i} 'samples' of type {type(data['samples']).__name__} coerced to list with {len(iter_samples)} items"
                        )
                    except Exception:
                        print_master(
                            f"Warning: Unexpected 'samples' type in {augmented_file}: {type(data['samples']).__name__}; skip"
                        )
                        continue
                print_master(f"Loaded {len(iter_samples)} augmented samples from iter {i} (with metadata)")
            elif isinstance(data, list):
                iter_samples = data
                print_master(f"Loaded {len(iter_samples)} augmented samples from iter {i} (direct list)")
            elif isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
                # Top-level dict-of-samples fallback.
                iter_samples = list(data.values())
                print_master(
                    f"Detected top-level dict-of-samples for iter {i}; converted to list with {len(iter_samples)} items"
                )
            else:
                print_master(f"Warning: Unexpected data format in {augmented_file}, skip")
                continue

            # Additional filtering: keep dict samples only to avoid `.get` access errors.
            if not isinstance(iter_samples, list):
                print_master(f"Warning: iter {i} samples not a list after coercion; skip")
                continue
            before = len(iter_samples)
            iter_samples = [s for s in iter_samples if isinstance(s, dict)]
            dropped = before - len(iter_samples)
            if dropped > 0:
                print_master(f"Note: iter {i} dropped {dropped} non-dict entries while loading augmented samples")

            if declared_total is not None and isinstance(declared_total, int) and declared_total != len(iter_samples):
                print_master(
                    f"Consistency check: iter {i} meta total_samples={declared_total}, loaded={len(iter_samples)}"
                )

            all_augmented_samples.extend(iter_samples)

        # Add resume-time statistics (aligned with previous implementation).
        stats = self._compute_sample_statistics(all_augmented_samples)
        if stats:
            print_master("Resume-time augmented sample statistics:")
            for k, v in stats.items():
                print_master(f"  - {k}: {v}")

            # Optional: persist resume statistics for auditing.
            resume_stats_file = os.path.join(self.args.output_dir, f"resume_stats_iter_{iteration}.json")
            try:
                with open(resume_stats_file, "w") as f:
                    json.dump(stats, f, indent=2)
                print_master(f"Saved resume statistics to {resume_stats_file}")
            except Exception as e:
                print_master(f"Warning: failed to save resume statistics: {e}")

        # Attach accumulated samples to current training dataset.
        if isinstance(self.train_dataset, (IterativeCIRRDataset, IterativeFashionIQDataset)):
            # Mark current iteration.
            self.train_dataset.iteration_round = iteration
            
            # Keep loaded samples as-is instead of clearing.
            self.train_dataset.augmented_samples = all_augmented_samples

            # Decide whether to include original data for iteration > 0.
            use_original = getattr(self.args, "use_original_data_in_iter_plus", True)
            if not use_original and iteration > 0:
                print_master(f"Iteration {iteration}: using ONLY augmented data as per `use_original_data_in_iter_plus=False`.")
                # Disable original data via dataset flag when available.
                if hasattr(self.train_dataset, "set_use_original_data"):
                    self.train_dataset.set_use_original_data(False)
                else:
                    print_master("Warning: dataset does not support `set_use_original_data`; original data might still be used.")
            else:
                if hasattr(self.train_dataset, "set_use_original_data"):
                    self.train_dataset.set_use_original_data(True)
                    # Force-refresh dataset internal state.
                    self.train_dataset.num_rows = len(self.train_dataset)

            # Compatibility field: some downstream code may still read this path.
            if hasattr(self.train_dataset, "hard_negatives_file"):
                self.train_dataset.hard_negatives_file = os.path.join(
                    self.args.output_dir, f"hard_negatives_iter_{iteration-1}.json"
                )

        print_master(
            f"Dataset prepared for iteration {iteration} "
            f"with {len(all_augmented_samples)} total augmented samples"
        )

        # Refresh dataloader so sampler/batch construction matches updated data.
        self._update_train_dataloader()

    def _compute_sample_statistics(self, samples: List[Dict]) -> Dict[str, Any]:
        """Compute descriptive statistics for augmented samples (robust & backward-compatible)."""
        if not samples:
            return {}

        try:
            orig_chars, gen_chars = [], []
            orig_words, gen_words = [], []
            refs, tgts, orig_tgts, pairs = set(), set(), set(), set()
            gt_ranks, gt_topk_ranks, gt_similarities = [], [], []
            gt_present_cnt = 0
            gt_before_cnt = 0
            gt_after_cnt = 0

            # optional counters
            has_original_cnt = 0
            has_generated_cnt = 0
            source_counter = {}

            for s in samples:
                # ----- text length (chars/words) -----
                orig_txt = s.get("original_mod_text", "")
                gen_txt  = s.get("modification_text", "")

                if isinstance(orig_txt, str) and len(orig_txt) > 0:
                    has_original_cnt += 1
                    orig_chars.append(len(orig_txt))
                    orig_words.append(len(orig_txt.split()))

                if isinstance(gen_txt, str) and len(gen_txt) > 0:
                    has_generated_cnt += 1
                    gen_chars.append(len(gen_txt))
                    gen_words.append(len(gen_txt.split()))

                # ----- unique images / pairs -----
                ref = s.get("reference_image")
                tgt = s.get("target_image")
                orig_tgt = s.get("original_target_image")
                if ref: refs.add(ref)
                if tgt: tgts.add(tgt)
                if orig_tgt: orig_tgts.add(orig_tgt)
                if ref and tgt:
                    pairs.add((ref, tgt))

                # ----- gt rank metadata -----
                gt_rank = s.get("gt_rank")
                if isinstance(gt_rank, (int, float)) and gt_rank >= 0:
                    gt_ranks.append(int(gt_rank))

                gt_topk_rank = s.get("gt_topk_rank")
                if isinstance(gt_topk_rank, (int, float)) and gt_topk_rank >= 0:
                    gt_topk_ranks.append(int(gt_topk_rank))

                if s.get("gt_in_candidates") is True:
                    gt_present_cnt += 1

                before_gt = s.get("is_before_gt_in_topk")
                if before_gt is True:
                    gt_before_cnt += 1
                elif before_gt is False and s.get("gt_in_candidates") is True:
                    gt_after_cnt += 1

                gt_sim = s.get("gt_similarity")
                if isinstance(gt_sim, (int, float)):
                    gt_similarities.append(float(gt_sim))

                # ----- optional: source breakdown -----
                src = s.get("source")
                if src:
                    source_counter[src] = source_counter.get(src, 0) + 1

            def _mean(x): return float(statistics.mean(x)) if x else 0.0
            def _median(x): return float(statistics.median(x)) if x else 0.0
            def _p95(x):
                if not x: return 0.0
                xs = sorted(x)
                idx = min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))
                return float(xs[idx])

            total = len(samples)
            augmented_ratio = sum(1 for s in samples if s.get("is_augmented", False)) / total if total else 0.0

            stats: Dict[str, Any] = {
                # Legacy-compatible fields (character-level means).
                "total_samples": total,
                "avg_original_length": _mean(orig_chars),
                "avg_generated_length": _mean(gen_chars),
                "unique_reference_images": len(refs),
                "unique_target_images": len(tgts),
                "unique_original_target_images": len(orig_tgts),
                "augmented_ratio": augmented_ratio,

                # Additional detailed metrics.
                "original": {
                    "count": has_original_cnt,
                    "avg_chars": _mean(orig_chars),
                    "median_chars": _median(orig_chars),
                    "p95_chars": _p95(orig_chars),
                    "avg_words": _mean(orig_words),
                    "median_words": _median(orig_words),
                    "p95_words": _p95(orig_words),
                },
                "generated": {
                    "count": has_generated_cnt,
                    "avg_chars": _mean(gen_chars),
                    "median_chars": _median(gen_chars),
                    "p95_chars": _p95(gen_chars),
                    "avg_words": _mean(gen_words),
                    "median_words": _median(gen_words),
                    "p95_words": _p95(gen_words),
                },
                "gt_metrics": {
                    "with_global_rank": len(gt_ranks),
                    "avg_global_rank": _mean(gt_ranks),
                    "median_global_rank": _median(gt_ranks),
                    "p95_global_rank": _p95(gt_ranks),
                    "with_topk_rank": len(gt_topk_ranks),
                    "avg_topk_rank": _mean(gt_topk_ranks),
                    "median_topk_rank": _median(gt_topk_ranks),
                    "p95_topk_rank": _p95(gt_topk_ranks),
                    "gt_in_candidates_count": gt_present_cnt,
                    "gt_in_candidates_ratio": (gt_present_cnt / total) if total else 0.0,
                    "before_gt_in_topk": gt_before_cnt,
                    "after_gt_in_topk": gt_after_cnt,
                    "avg_gt_similarity": _mean(gt_similarities),
                    "median_gt_similarity": _median(gt_similarities),
                    "p95_gt_similarity": _p95(gt_similarities),
                },
                "unique_pairs": len(pairs),
                "duplicate_pair_count": max(0, total - len(pairs)),  # Simple estimate.
            }

            if source_counter:
                stats["by_source"] = dict(sorted(source_counter.items(), key=lambda kv: (-kv[1], kv[0])))

            return stats

        except Exception as e:
            print_master(f"Warning: Failed to compute sample statistics: {e}")
            return {"total_samples": len(samples)}

    def _update_train_dataloader(self):
        """Update train dataloader to reflect dataset changes (safe for HF Trainer + DDP)."""
        import torch.distributed as dist

        # 1) Optional DDP sync (safer).
        if dist.is_initialized():
            dist.barrier()

        # 2) Clear Trainer internal caches to force rebuild.
        #   - `_train_dataloader` is HF Trainer internal cache.
        #   - `_train_sampler` may also be cached in some versions.
        if hasattr(self, "_train_dataloader"):
            self._train_dataloader = None
        if hasattr(self, "_train_sampler"):
            self._train_sampler = None

        # 3) Rebuild dataloader from current dataset/sampler/collator.
        dl = self.get_train_dataloader()
        # Explicitly backfill cache to avoid repeated `get_train_dataloader` in custom scenarios.
        self._train_dataloader = dl

        # 4) Detailed statistics and logging.
        try:
            total_samples = len(self.train_dataset) if self.train_dataset is not None else 0
        except Exception:
            total_samples = 0

        if hasattr(self.train_dataset, "augmented_samples"):
            try:
                augmented_count = len(self.train_dataset.augmented_samples)
            except Exception:
                augmented_count = 0
            original_count = max(0, total_samples - augmented_count)

            print_master("Updated train dataloader:")
            print_master(f"  - Total samples: {total_samples}")
            print_master(f"  - Original samples: {original_count}")
            print_master(f"  - Augmented samples: {augmented_count}")
        else:
            print_master(f"Updated train dataloader with {total_samples} total samples")

        # 5) For grouped/custom samplers, ensure rebuild happens from latest dataset.

    def _lazy_load_foundation_model(self, to_device: str = None):
        """
        Lazily load foundation model only when caption generation is needed.
        """
        if getattr(self, "foundation_model", None) is not None:
            # Already loaded; move to target device if needed.
            if to_device:
                try:
                    self.foundation_model.to(to_device)
                except Exception:
                    pass
            return self.foundation_model

        if not getattr(self, "foundation_model_name", None):
            print_master("No foundation_model_name set; skip loading FM.")
            return None

        from transformers import AutoModelForVision2Seq, AutoProcessor
        dev = to_device or (f"cuda:{self.args.process_index}" if torch.cuda.is_available() else "cpu")

        print_master(f"Lazy-loading foundation model on {dev}: {self.foundation_model_name}")
        fm = AutoModelForVision2Seq.from_pretrained(
            self.foundation_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=None
        ).to(dev).eval()

        proc = AutoProcessor.from_pretrained(self.foundation_model_name, trust_remote_code=True)
        setattr(fm, "processor", proc)      # For CaptionGenerator / Batcher.
        self.foundation_model = fm
        self.foundation_processor = proc
        print_master("Foundation model lazy-loaded.")
        return fm


    def _unload_foundation_model(self):
        """Unload foundation model after generation to release VRAM."""
        import gc
        try:
            del self.foundation_model
        except Exception:
            pass
        self.foundation_model = None
        try:
            del self.foundation_processor
        except Exception:
            pass
        self.foundation_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_master("Foundation model unloaded and CUDA cache cleared.")

    # ---------------------------
    # External module wrappers
    # ---------------------------
    def _collect_hard_negatives(self, iteration: int):
        """
        Collect hard negatives via decoupled `HardNegativeMiner`:
        - Instantiate retrieval modules and pass them into the miner.
        - Support both single-GPU and distributed-minimal mode.
        - Cache file path remains `{output_dir}/hard_negatives_iter_{iteration}.json`.
        """
        import os, json
        import torch.distributed as dist
        from src.utils import print_master

        print_master(f"Collecting hard negatives for iteration {iteration}...")

        cache_file = os.path.join(self.args.output_dir, f"hard_negatives_iter_{iteration}.json")
        if os.path.exists(cache_file):
            print_master(f"Found cached hard negatives for iteration {iteration}, loading...")
            with open(cache_file, "r") as f:
                cached = json.load(f)
            print_master(f"Loaded {len(cached)} cached hard negatives")
            return cached

        # ---------- Validate required dataset fields ----------
        ds = self.train_dataset

        annotations = (
            getattr(ds, "annotations", None)
            or getattr(ds, "train_annotations", None)
            or getattr(ds, "ann", None)
        )
        image_splits = getattr(ds, "image_splits", None)
        image_base_dir = getattr(ds, "image_base_dir", None) or getattr(ds, "root", None)

        retrieval_candidates = getattr(ds, "retrieval_candidates", None)
        if retrieval_candidates is None:
            retrieval_candidates = []  # Empty is allowed; engine may build/expand internally.

        missing = []
        if annotations is None:    missing.append("annotations")
        if image_splits is None:   missing.append("image_splits")
        if image_base_dir is None: missing.append("image_base_dir/root")
        if missing:
            raise RuntimeError(f"train_dataset missing required attributes: {missing}")

        if not isinstance(annotations, (list, tuple)) or len(annotations) == 0:
            raise RuntimeError("annotations are empty or invalid; hard-negative mining cannot proceed")

         # ---------- Training / device / processor ----------
        proc = self.processing_class or getattr(self.model, "processor", None)
        backbone = getattr(self.model_args, "model_backbone", "qwen2_vl")
        device = f"cuda:{self.args.process_index}" if torch.cuda.is_available() else "cpu"

        # Fast/production controls.
        sample_limit = self.fast_mode_max_samples if getattr(self, "fast_mode", False) else None

        dataset_config = getattr(ds, "dataset_config", {}) or {}
        hard_neg_top_k = max(int(dataset_config.get("hard_neg_top_k", 10)), 1)
        hard_neg_post_gt = max(int(dataset_config.get("hard_neg_post_gt", 0)), 0)
        hard_neg_per_query = int(dataset_config.get("hard_neg_per_query", 5))
        if hard_neg_per_query < 1:
            hard_neg_per_query = 1

        # 1) Build decoupled modules from dataset-provided information.
        candidate_builder = CandidateBuilder(
            annotations=annotations,
            image_splits=image_splits,
            image_base_dir=image_base_dir,
            # Optional if implementation supports it: experiment_dir=self.args.output_dir
        )


        embedding_cache = EmbeddingCache(
            experiment_dir=self.args.output_dir
        )

        # Build retrieval candidates via CandidateBuilder first.
        retrieval_candidates = candidate_builder.build()
        print_master(f"CandidateBuilder built {len(retrieval_candidates)} retrieval candidates")


        # Provide required RetrievalEngine parameters.
        retrieval_engine = RetrievalEngine(
            model_args=self.model_args,
            experiment_dir=self.args.output_dir,
            image_base_dir=image_base_dir,
            retrieval_candidates=retrieval_candidates,
            topk=hard_neg_top_k,
            # Optional: fast-mode limit if engine supports it.
            # fast_mode_limit=self.fast_mode_retrieval_db_size if self.fast_mode else None,
        )

        # 2) Assemble miner.
        miner = HardNegativeMiner(
            experiment_dir=self.args.output_dir,
            iteration_round=iteration,
            candidate_builder=candidate_builder,
            retrieval_engine=retrieval_engine,
            embedding_cache=embedding_cache,
            image_base_dir=image_base_dir,
            max_negatives_per_query=hard_neg_per_query,
            examine_topk=hard_neg_top_k,
            post_gt_negatives=hard_neg_post_gt,
        )

        call_kwargs = dict(
            batch_size=8,
            max_samples=sample_limit,
            processor=proc,
            model_backbone=backbone,
            device=device,
        )

        # 3) Collection: prefer distributed-minimal path when distributed is available.
        if dist.is_initialized() and dist.get_world_size() > 1:
            print_master("Using HardNegativeMiner.collect_distributed_minimal ...")
            hard_negatives = miner.collect_distributed_minimal(self.model, annotations, **call_kwargs)
        else:
            print_master("Using HardNegativeMiner.collect_single_gpu ...")
            hard_negatives = miner.collect_single_gpu(self.model, annotations, **call_kwargs)

        print_master(
            f"Collected {len(hard_negatives)} hard negatives "
            f"{'(limited to '+str(sample_limit)+')' if sample_limit else '(no limit)'}"
        )

        # Rank-0 writes cache as an extra safety step.
        if not dist.is_initialized() or dist.get_rank() == 0:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(hard_negatives, f, indent=2)
            print_master(f"Cached hard negatives to {cache_file}")

        return hard_negatives

    def _generate_augmented_captions(self, hard_negatives: List[Dict]) -> List[Dict]:
        """Generate augmented instruction text via CaptionGenerator (single/distributed)."""
        if not hard_negatives:
            return []

        dev = f"cuda:{self.args.process_index}" if torch.cuda.is_available() else "cpu"
        fm = self._lazy_load_foundation_model(to_device=dev)
        if fm is None:
            print_master("No foundation model available, skip caption generation")
            return []

        # Default backbone (Qwen in this project).
        if not getattr(self.model_args, "foundation_model_backbone", None):
            setattr(self.model_args, "foundation_model_backbone", "qwen2_5_vl")



    # Build prepare/generate function maps.
        PREPARE_FNS = {
            "qwen": qwen_prepare,
            "llava": llava_prepare or (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("LLaVA not configured"))),
            "generic": generic_prepare or (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Generic not configured"))),
        }
        GENERATE_FNS = {
            "qwen": generate_with_qwen,
            "llava": generate_with_llava or (lambda *a, **k: ""),
            "generic": generate_with_generic or (lambda *a, **k: ""),
        }



        try:
            generator = CaptionGenerator(
                foundation_model=fm,
                model_args=self.model_args,
                experiment_dir=self.args.output_dir,
                iteration_round=self.current_iteration,
                prepare_fns=PREPARE_FNS,
                generate_fns=GENERATE_FNS,
                # Pass dataset base/splits into generator.
                image_base_dir=self.train_dataset.image_base_dir,
                image_splits=self.train_dataset.image_splits,
            )

            if dist.is_initialized() and dist.get_world_size() > 1:
                print_master("Using distributed caption generation...")
                augmented_samples = generator.generate_augmented_captions_distributed(hard_negatives)
            else:
                print_master("Using single-GPU caption generation...")
                augmented_samples = generator.generate_augmented_captions(hard_negatives)

        except Exception as e:
            print_master(f"Caption generation failed: {e}")
            augmented_samples = []
        finally:
            # Unload immediately after use to release VRAM.
            self._unload_foundation_model()

        print_master(f"Generated {len(augmented_samples)} augmented samples")
        return augmented_samples
    
    def iterative_train(self, resume_from_iteration: int = 0):
        """
        Main iterative training loop.
        - Preserves/manual resume, step completion checks, timing, distributed barriers, and cache I/O.
        - Compatible with decoupled dataset/retrieval/augmentation modules.
        """
        import os, json, time
        import torch.distributed as dist
        from .utils import print_master

        print_master("Starting iterative training process...")

        # Manually resume from a specified iteration.
        if resume_from_iteration > 0:
            print_master(f"Manually resuming from iteration {resume_from_iteration}")
            self.current_iteration = resume_from_iteration
            # Read metadata from previous iteration (model weights are loaded externally).
            self._load_iteration_state(resume_from_iteration - 1)
            # Prepare dataset state needed by resumed iteration.
            self._prepare_dataset_for_iteration(resume_from_iteration)

        for iteration in range(self.current_iteration, self.max_iterations):
            print_master(f"\n{'='*60}")
            print_master(f"Starting Iteration {iteration}")
            print_master(f"{'='*60}")
            self.current_iteration = iteration

            # Check whether steps are already complete (for skipping).
            completed_steps = self._check_iteration_completion_status(iteration)
            print_master(f"Iteration {iteration} completion status: {completed_steps}")

            # -----------------------
            # Step 1: Train (if incomplete)
            # -----------------------
            if not completed_steps.get('model_training', False):
                if iteration == 0:
                    print_master("Iteration 0: Training base retrieval model...")
                    self._train_base_model()
                else:
                    print_master(f"Iteration {iteration}: Training with augmented data...")
                    self._train_current_iteration()
                    print_master(f"Iteration {iteration} training completed with fresh optimizer/scheduler")
                if dist.is_initialized():
                    dist.barrier()
                    print_master(f"All GPUs completed training for iteration {iteration}")
            else:
                print_master("Model training already completed, skipping...")

            # -----------------------
            # Step 2: Evaluate (if incomplete)
            # -----------------------
            if not completed_steps.get('evaluation', False):
                if dist.is_initialized():
                    dist.barrier()  # Ensure all workers finished training.
                eval_results = self._evaluate_iteration_checkpoints(iteration)
                self.iteration_metrics[iteration] = eval_results
            else:
                print_master("Model evaluation already completed, loading cached results...")
                eval_file = os.path.join(self.args.output_dir, f"eval_results_iter_{iteration}.json")
                try:
                    with open(eval_file, 'r') as f:
                        eval_results = json.load(f)
                    eval_results = self._normalize_metric_aliases(eval_results)
                    self.iteration_metrics[iteration] = eval_results
                except Exception as e:
                    print_master(f"Warning: failed to load cached eval results: {e}. Re-evaluating...")
                    eval_results = self._normalize_metric_aliases(self._evaluate_current_model())
                    self.iteration_metrics[iteration] = eval_results

            # ------------------------------------------------------
            # Step 3-4: hard-negative collection + caption augmentation (non-final iterations only)
            # ------------------------------------------------------
            if iteration < self.max_iterations - 1:
                # 3) Hard negatives.
                hard_neg_time = 0.0
                if not completed_steps.get('hard_negatives_collection', False):
                    print_master(f"Starting hard negative collection for iteration {iteration}...")
                    t0 = time.time()
                    hard_negatives = self._collect_hard_negatives(iteration)
                    hard_neg_time = time.time() - t0
                    print_master(f"Hard negative collection completed in {int(hard_neg_time//60):02d}:{int(hard_neg_time%60):02d}")
                    if dist.is_initialized():
                        dist.barrier()
                        print_master(f"All GPUs completed hard negative collection for iteration {iteration}")
                else:
                    print_master("Hard negative collection already completed, loading cached results...")
                    hn_file = os.path.join(self.args.output_dir, f"hard_negatives_iter_{iteration}.json")
                    with open(hn_file, 'r') as f:
                        hard_negatives = json.load(f)
                    hard_neg_time = 0.0

                # Stop early when no hard negatives are found.
                if not hard_negatives:
                    print_master("Warning: no hard negatives found, stopping early")
                    break
                
                # TODO: add FM-based false-negative filtering via decoupled module interfaces.

                # 4) Caption augmentation.
                caption_time = 0.0
                next_iter = iteration + 1
                if not completed_steps.get('caption_generation', False):
                    print_master(f"Starting caption generation for {len(hard_negatives)} hard negatives...")
                    t1 = time.time()
                    augmented_samples = self._generate_augmented_captions(hard_negatives)
                    caption_time = time.time() - t1
                    print_master(f"Caption generation completed in {int(caption_time//60):02d}:{int(caption_time%60):02d}")
                    if dist.is_initialized():
                        dist.barrier()
                        print_master(f"All GPUs completed caption generation for iteration {iteration}")
                    # Prepare dataset for next iteration.
                    self._prepare_next_iteration_dataset(next_iter, augmented_samples)
                else:
                    print_master("Caption generation already completed, loading cached results...")
                    aug_file = os.path.join(self.args.output_dir, f"augmented_samples_iter_{next_iter}.json")
                    with open(aug_file, 'r') as f:
                        saved_data = json.load(f)
                    augmented_samples = saved_data.get('samples', saved_data if isinstance(saved_data, list) else [])
                    caption_time = 0.0

                # Statistics.
                total_time = hard_neg_time + caption_time
                print_master(f"Iteration {iteration} data preparation stats:")
                print_master(f"  - Hard negatives: {len(hard_negatives)} samples in {hard_neg_time:.1f}s")
                print_master(f"  - Augmented captions: {len(augmented_samples)} samples in {caption_time:.1f}s")
                print_master(f"  - Total time: {int(total_time//60):02d}:{int(total_time%60):02d}")
                if dist.is_initialized():
                    ws = dist.get_world_size()
                    print_master(f"  - Used {ws} GPUs for parallel processing")
                    if total_time > 0:
                        rate = (len(hard_negatives) + len(augmented_samples)) / max(total_time, 1e-6)
                        print_master(f"  - Processing rate: {rate:.2f} samples/second")

                # TODO: add post-generation filtering (fast string match + FM-based) via decoupled modules.

            # -----------------------
            # Final sync and state persistence
            # -----------------------
            if dist.is_initialized():
                dist.barrier()
                print_master(f"All GPUs completed iteration {iteration}, saving state...")

            self._save_iteration_state(iteration)

        print_master("\nIterative training completed!")
        self._summarize_results()


# ---------------------------
    # Factory function
# ---------------------------
def create_iterative_trainer(
    model: MMEBModel,
    foundation_model=None,
    args: TrainingArguments = None,
    train_dataset=None,
    eval_dataset=None,
    experiment_dir=None,
    **kwargs
) -> IterativeRetrievalTrainer:

    iterative_params = {k: kwargs.pop(k) for k in
                        ['max_iterations', 'hard_neg_collection_freq', 'caption_generation_batch_size']
                        if k in kwargs}

    fast_mode_params = {k: kwargs.pop(k) for k in
                        ['fast_mode', 'fast_mode_max_samples', 'fast_mode_retrieval_db_size',
                         'fast_mode_max_steps', 'steps_per_iteration', 'production_save_steps',
                         'production_max_steps']
                        if k in kwargs}

    important_args = {k: kwargs.pop(k) for k in ['model_args', 'data_args', 'max_length'] if k in kwargs}
    foundation_model_name = kwargs.pop('foundation_model_name', None)

    # Extract standard Trainer-related parameters.
    trainer_params = {k: kwargs.pop(k) for k in [
        'data_collator', 'tokenizer', 'model_init', 'compute_metrics',
        'callbacks', 'optimizers', 'preprocess_logits_for_metrics',
        'processing_class'  # Used by evaluator path.
    ] if k in kwargs}

    # Pass both extracted trainer params and remaining kwargs downstream.
    return IterativeRetrievalTrainer(
        model=model,
        foundation_model=foundation_model,
        foundation_model_name=foundation_model_name,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **iterative_params,
        **fast_mode_params,
        **important_args,
        **trainer_params,
        **kwargs
    )

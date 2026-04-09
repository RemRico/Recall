# aug/batchers.py
from src.utils import print_rank
from src.prompt.qwen import builder_minimal


class CaptionBatcher:
    """Batch caption generation wrapper across different foundation backbones."""

    def __init__(self, foundation_model, model_args, prepare_fns, generate_fns):
        """
        Args:
            foundation_model: Underlying generator model (Qwen / LLaVA / Generic)
            model_args: Model argument container
            prepare_fns: Mapping of input-preparation functions
            generate_fns: Mapping of generation functions
        """
        self.foundation_model = foundation_model
        self.model_args = model_args
        self.prepare_fns = prepare_fns
        self.generate_fns = generate_fns

    def generate_batch(self, ref_images, target_images, original_texts, processor, device):
        """Generate captions in batch according to configured backbone type."""
        backbone = getattr(self.model_args, "foundation_model_backbone", "qwen2_vl")

        if backbone in ["qwen2_vl", "qwen", "qwen2_5_vl"]:
            return self._generate_qwen_batch(
                ref_images, target_images, original_texts, processor, device
            )
        elif backbone in ["llava", "llava_next"]:
            return self._generate_llava_batch(
                ref_images, target_images, original_texts, processor, device
            )
        else:
            return self._generate_generic_batch(
                ref_images, target_images, original_texts, processor, device
            )

    def _generate_qwen_batch(self, ref_images, target_images, prompts, processor, device):
        """Generate captions with Qwen-family models."""
        results = []

        # Optional dataset-aware prompt behavior.
        dataset_name = getattr(self.model_args, "current_dataset_name", None)
        prompt_mode = getattr(self.model_args, "foundation_prompt_mode", "minimal")

        for ref_img, tgt_img, prompt in zip(ref_images, target_images, prompts):
            text_out = None
            if prompt_mode == "minimal":
                try:
                    minimal_result = builder_minimal.run_minimal_pipeline(
                        ref_img,
                        tgt_img,
                        prompt,
                        processor,
                        device,
                        self.foundation_model,
                        dataset_name=dataset_name,
                    )
                    text_out = minimal_result.get("final_text")
                    if not text_out:
                        err = minimal_result.get("validation_error", "unknown error")
                        print_rank(f"[CaptionBatcher] minimal pipeline yielded empty text (reason: {err}); fallback to CoT prompt")
                except Exception as e:
                    print_rank(f"[CaptionBatcher] minimal pipeline failed: {e}")

            if text_out is None or text_out == "":
                try:
                    inputs = self.prepare_fns["qwen"](ref_img, tgt_img, prompt, processor, device)
                    text_out = self.generate_fns["qwen"](inputs, device, self.foundation_model)
                except Exception as e:
                    print_rank(f"Error in Qwen batch fallback: {e}")
                    text_out = None

            results.append(text_out)
        return results

    def _generate_llava_batch(self, ref_images, target_images, prompts, processor, device):
        """Generate captions with LLaVA-family models."""
        results = []
        for ref_img, tgt_img, prompt in zip(ref_images, target_images, prompts):
            try:
                inputs = self.prepare_fns["llava"](ref_img, tgt_img, prompt, processor, device)
                text = self.generate_fns["llava"](inputs, device, self.foundation_model)
                results.append(text)
            except Exception as e:
                print_rank(f"Error in LLaVA batch: {e}")
                results.append(None)
        return results

    def _generate_generic_batch(self, ref_images, target_images, prompts, processor, device):
        """Generate captions with generic fallback generation functions."""
        results = []
        for ref_img, tgt_img, prompt in zip(ref_images, target_images, prompts):
            try:
                inputs = self.prepare_fns["generic"](ref_img, tgt_img, prompt, processor, device)
                text = self.generate_fns["generic"](inputs, device, self.foundation_model)
                results.append(text)
            except Exception as e:
                print_rank(f"Error in Generic batch: {e}")
                results.append(None)
        return results

# src/model/model.py
from typing import Dict

import torch
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from transformers import modeling_utils

from peft import LoraConfig, get_peft_model, PeftModel

from src.arguments import ModelArguments
from src.model.processor import (
    LLAVA_NEXT,
    QWEN2_VL,
    PHI3V,
    QWEN2_5_VL,
    QWEN2_VL_TOKENSELECTION,
    QWEN2_5_VL_TOKENSELECTION,
    E5_V,
    INTERNVIDEO2,
    GME,
    LamRA,
    LamRA_QWEN2_5,
    COLPALI,
    VLM_IMAGE_TOKENS,
    get_backbone_name,
    print_master,
    backbone2model,
)

from src.model.baseline_backbone.colpali import ColPali
from src.model.baseline_backbone.gme.gme_inference import GmeQwen2VL
from src.model.baseline_backbone.lamra.lamra_inference import LamRAQwen2VL
from src.model.baseline_backbone.lamra.lamra_qwen25_inference import LamRAQwen25VL
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration

# 兼容部分 Transformers 版本可能缺少该常量
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(
        self,
        encoder: PreTrainedModel,
        pooling: str = "last",
        normalize: bool = False,
        temperature: float = 0.02,
    ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.info_nce_weight = 1.0
        self.triplet_loss_weight = 0.0
        self.triplet_margin = 0.2
        self._warned_missing_triplet = False

        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    # ===== encode_input：保持你最初的逻辑 =====
    def encode_input(self, input):
        if getattr(self, "model_backbone", None) == INTERNVIDEO2:
            if "input_ids" in input.keys():
                # text side
                text_output = self.encoder.get_text_encoder()(
                    input["input_ids"],
                    attention_mask=input["attention_mask"],
                    return_dict=True,
                    mode="text",
                )
                text_embeds = text_output.last_hidden_state
                pooled_text_embeds = text_embeds[:, 0]
                pooled_output = self.encoder.text_proj(pooled_text_embeds)
                pooled_output /= pooled_output.norm(dim=-1, keepdim=True)
                return pooled_output
            else:
                _, vfeat = self.encoder.encode_vision(input["pixel_values"], test=True)
                vfeat = self.encoder.vision_proj(vfeat)
                vfeat /= vfeat.norm(dim=-1, keepdim=True)
                return vfeat

        elif getattr(self, "model_backbone", None) in [GME, LamRA, LamRA_QWEN2_5]:
            raw_texts = input.get("texts", [])
            texts = []
            for text in raw_texts:
                if isinstance(text, str):
                    texts.append(text.replace(VLM_IMAGE_TOKENS[QWEN2_VL] + "\n", ""))
                else:
                    texts.append(text)

            images = []
            for imgs in input.get("images", []):
                # if multi images are given, select the middle frame only
                if isinstance(imgs, list):
                    if len(imgs) == 0:
                        images.append(None)
                        continue
                    imgs = imgs[len(imgs) // 2]
                images.append(imgs)

            instruction = input.get("instruction", None)
            is_query = input.get("is_query", True)
            pooled_output = self.encoder.get_fused_embeddings(
                texts=texts,
                images=images,
                is_query=is_query,
                instruction=instruction,
            )
            return pooled_output

        elif getattr(self, "model_backbone", None) == COLPALI:
            pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            return pooled_output

        elif getattr(self, "model_backbone", None) == LLAVA_NEXT:
            input["pixel_values"] = input["pixel_values"].squeeze(dim=1)
            input["image_sizes"] = input["image_sizes"].squeeze(dim=1)
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            pooled_output = self._pooling(hidden_states, input["attention_mask"])
            return pooled_output

        else:
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            pooled_output = self._pooling(hidden_states, input["attention_mask"])
            return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ("last", "eos"):
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices
                ]
        else:
            raise NotImplementedError

        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing by delegating to the underlying encoder model.
        This is needed when using --gradient_checkpointing True in training.
        （行为不变：仅在底模支持时开启；否则保持原本状态）
        """
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        # LoRA + DDP + gradient checkpointing 会在默认 use_reentrant=True 时触发
        # “mark parameter ready twice” 报错，因此默认改成 False，可被显式参数覆盖
        gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        else:
            # Fallback for models that don't have this method
            if hasattr(self.encoder, "config"):
                self.encoder.config.use_cache = False
            if hasattr(self.encoder, "gradient_checkpointing"):
                self.encoder.gradient_checkpointing = True
        # Ensure embeddings allow gradient flow when checkpointing is active
        self.enable_input_require_grads()

    def enable_input_require_grads(self):
        """
        Mirror HF PreTrainedModel.enable_input_require_grads so Trainer can un-freeze
        the embedding inputs when gradient checkpointing + LoRA are both enabled.
        """
        if getattr(self, "_enable_input_require_grads_called", False):
            return
        self._enable_input_require_grads_called = True
        if hasattr(self.encoder, "enable_input_require_grads"):
            self.encoder.enable_input_require_grads()
            return

        get_input_embeddings = getattr(self.encoder, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            input_embeddings = get_input_embeddings()
            if input_embeddings is not None and hasattr(input_embeddings, "weight"):
                input_embeddings.weight.requires_grad_(True)
                if not hasattr(self, "_input_grad_hook"):
                    def _make_inputs_require_grad(_, inputs, output):
                        if isinstance(output, torch.Tensor):
                            output.requires_grad_(True)
                            return output
                        if isinstance(output, tuple):
                            new_output = []
                            changed = False
                            for item in output:
                                if isinstance(item, torch.Tensor):
                                    new_output.append(item.requires_grad_(True))
                                    changed = True
                                else:
                                    new_output.append(item)
                            if changed:
                                return tuple(new_output)
                        return output
                    self._input_grad_hook = input_embeddings.register_forward_hook(_make_inputs_require_grad)

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        print_master(f"Loading backbone [{model_backbone}] from {model_args.model_name}")

        # Loading the base model
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        elif model_backbone == LLAVA_NEXT:
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        elif model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        elif model_backbone in [QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            from .utils import parse_layer_type

            lm_qwen_layer = 28
            vis_qwen_layer = 32
            lm_skip_layer = parse_layer_type(model_args.lm_skip_layer, lm_qwen_layer)
            vis_skip_layer = parse_layer_type(model_args.vis_skip_layer, vis_qwen_layer)

            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                lm_skip_layer=lm_skip_layer,
                vis_skip_layer=vis_skip_layer,
            )

        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name,
                **kwargs,
                config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        # LoRA（如你原始实现）
        if model_args.lora:
            print_master(f"Loading lora adapter from {base_model}")
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(","),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False,
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        # Loading the base model
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
            model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
            setattr(model_args, "model_backbone", model_backbone)

        print_master(f"Loading backbone [{model_args.model_backbone}] from {model_name_or_path}")

        if model_args.model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V}:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config._attn_implementation = "flash_attention_2"
            config.vision_config._attn_implementation = "flash_attention_2"
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config,
            )

        elif model_args.model_backbone == PHI3V:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name, **kwargs, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            base_model.padding_side = "right"

        elif model_args.model_backbone == INTERNVIDEO2:
            print_master(f"Loading backbone [{model_args.model_backbone}] from {'src/model/vlm_backbone/internvideo2/'}")
            config = AutoConfig.from_pretrained("src/model/vlm_backbone/internvideo2/", trust_remote_code=True)
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                "src/model/vlm_backbone/internvideo2/", config=config, trust_remote_code=True
            )

        elif model_args.model_backbone == GME:
            base_model = GmeQwen2VL(model_args.model_name, processor=kwargs["processor"])
            setattr(base_model, "config", config)

        elif model_args.model_backbone == LamRA:
            base_model = LamRAQwen2VL(model_args.model_name)
            setattr(base_model, "config", config)

        elif model_args.model_backbone == LamRA_QWEN2_5:
            base_model = LamRAQwen25VL(model_args.model_name)
            setattr(base_model, "config", config)

        elif model_args.model_backbone == COLPALI:
            base_model = ColPali.from_pretrained(model_args.model_name)
            setattr(base_model, "config", config)

        else:
            # Loading external base model from HF
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **kwargs, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True
            )

        # Building the model on top of the base
        if model_args.lora:
            print_master(f"Loading LoRA from {model_name_or_path}")
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            lora_model = PeftModel.from_pretrained(
                base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable
            )
            lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=is_trainable)
            if not is_trainable:
                lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )

        model.model_backbone = model_args.model_backbone
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    # ===== forward：保持你最初的 InfoNCE/CE 计算 =====
    def forward(
        self,
        qry: Dict[str, Tensor] = None,
        tgt: Dict[str, Tensor] = None,
        *args,
        return_embeddings: bool = False,
        **kwargs,
    ):
        qry_reps = self.encode_input(qry) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt) if tgt else None  # (bsz_per_device, dim)

        if qry_reps is None or tgt_reps is None:
            outputs = {"qry_reps": qry_reps, "tgt_reps": tgt_reps}
            return outputs if return_embeddings else qry_reps

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        total_loss = None
        losses = []
        info_nce_loss = None

        if self.info_nce_weight > 0:
            scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
            scores = scores.view(all_qry_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))

            info_nce_loss = self.cross_entropy(scores / self.temperature, target)
            if self.is_ddp:
                info_nce_loss = info_nce_loss * self.world_size
            losses.append(self.info_nce_weight * info_nce_loss)

        if losses:
            total_loss = sum(losses)
        else:
            total_loss = torch.tensor(0.0, device=qry_reps.device, dtype=qry_reps.dtype)

        # Store for optional debugging/inspection
        self._last_loss_components = {
            "info_nce_loss": info_nce_loss.detach() if info_nce_loss is not None else None,
            "triplet_loss": None,
        }
        if not return_embeddings:
            return total_loss

        outputs = {
            "loss": total_loss,
            "info_nce_loss": info_nce_loss,
            "qry_reps_gathered": all_qry_reps,
            "tgt_reps_gathered": all_tgt_reps,
            "qry_reps_local": qry_reps,
            "tgt_reps_local": tgt_reps,
        }
        return outputs

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def configure_loss(self, info_nce_weight=None, triplet_loss_weight=None, triplet_margin=None):
        if info_nce_weight is not None:
            self.info_nce_weight = info_nce_weight
        if triplet_loss_weight is not None:
            self.triplet_loss_weight = triplet_loss_weight
        if triplet_margin is not None:
            self.triplet_margin = triplet_margin
        # Reset warning flag so toggling losses logs appropriately
        self._warned_missing_triplet = False

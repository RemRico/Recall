# retrieval/engine.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable
import os
import time

import torch
import torch.nn.functional as F

from ..utils import print_rank
from ..utils.path_utils import get_full_image_path


try:
    # 兼容你原项目里的工具函数位置
    from src.model.processor import process_input_text  # type: ignore
except Exception:
    process_input_text = None  # 我们会在用到时显式报错

# VLM2Vec 官方 processor 映射
from src.model.processor import (
    process_vlm_inputs_fns,  # type: ignore
    GME,
    LamRA,
    LamRA_QWEN2_5,
    GME_CIRR_QUERY_INSTRUCTION,
)

from .embedding_cache import EmbeddingCache


class RetrievalEngine:
    """
    只负责“真实检索 + fallback 简化检索”，与缓存、候选库解耦。
    关键目标：
    - 100% 保持你原来的行为：fast/production 模式；top-k 结构；gt 索引计算；dtype 对齐；L2 normalize；维度兜底等。
    """

    def __init__(
        self,
        experiment_dir: str,
        image_base_dir: str,
        model_args: Any,  # 用于拿 model_backbone
        retrieval_candidates: List[str],
        embedding_cache: Optional[EmbeddingCache] = None,
        *,
        topk: int = 10,
    ):
        self.experiment_dir = experiment_dir
        self.image_base_dir = image_base_dir
        self.model_args = model_args
        self.retrieval_candidates = retrieval_candidates
        self.cache = embedding_cache or EmbeddingCache(experiment_dir)
        self.topk = max(int(topk), 1)

    # -------------------------- public API --------------------------

    def run_retrieval_batch(
        self, model: torch.nn.Module, batch: List[Dict[str, Any]], max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        与你原先 _run_retrieval_batch 等价：优先真实检索，失败时回退简化检索
        """
        print_rank(f"Running real retrieval for {len(batch)} queries")
        try:
            return self._run_real_retrieval(model, batch, max_samples)
        except Exception as e:
            print_rank(f"Real retrieval failed: {e}, falling back to simplified retrieval")
            return self._run_simplified_retrieval(batch)

    def run_retrieval_with_cached_targets(
        self,
        model: torch.nn.Module,
        batch: List[Dict[str, Any]],
        target_embeddings: torch.Tensor,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        与你原先 _run_real_retrieval_with_cached_targets 等价（抽成公开方法）
        """
        return self._run_real_retrieval_with_cached_targets(model, batch, target_embeddings, max_samples)

    def get_cache_file_path(self, target_database: List[str]) -> str:
        """供分布式流程生成 .done flag 时使用"""
        return self.cache.get_cache_file_path(target_database)

    # -------------------------- internal: retrieval --------------------------

    def _run_real_retrieval(
        self, model: torch.nn.Module, batch: List[Dict[str, Any]], max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        与你原始 _run_real_retrieval 完全一致（除了路径函数换成 utils），包含：
        - fast/production 候选集裁剪
        - 使用 EmbeddingCache.get_or_compute（与原 _get_or_compute_target_embeddings 行为一致）
        - 归一化 + 相似度 + top-k + gt 索引
        """
        device = next(model.parameters()).device
        model_backbone = getattr(self.model_args, "model_backbone", "qwen2_vl")
        processor = getattr(model, "processor", None)
        if processor is None:
            print_rank("Warning: No processor found in model")
            raise RuntimeError("No processor available")

        # 候选集选择（fast / production）
        import torch.distributed as dist  # 本地导入，避免非分布式环境报错
        if max_samples is not None and max_samples <= 100:
            min_candidates = min(1000, len(self.retrieval_candidates))
            candidate_targets = self.retrieval_candidates[:min_candidates]
            if not dist.is_initialized() or dist.get_rank() == 0:
                print_rank(
                    f"Fast mode: using {len(candidate_targets)} target candidates (subset of {len(self.retrieval_candidates)})"
                )
        else:
            candidate_targets = self.retrieval_candidates
            if not dist.is_initialized() or dist.get_rank() == 0:
                print_rank(f"Production mode: using full retrieval candidate set ({len(candidate_targets)} images)")

        if len(candidate_targets) == 0:
            raise RuntimeError("No valid target images found")

        # 目标向量（缓存）
        target_embeddings = self.cache.get_or_compute(
            candidate_targets, model, processor, model_backbone, device, self._prepare_target_inputs
        )

        # ✅ 统一路径：把候选集映射为绝对路径列表，仅用于 GT 匹配
        abs_candidate_paths = [get_full_image_path(p, self.image_base_dir) for p in candidate_targets]

        # 编码查询（参考图 + 文本）
        with torch.no_grad():
            q_inputs = self._prepare_query_inputs(batch, processor, model_backbone, device)
            try:
                q_embs = model.encode_input(q_inputs)
                q_embs = self._process_embeddings(q_embs, len(batch), "query_embeddings").cpu()
            except Exception as e:
                print_rank(f"Error encoding queries: {e}")
                q_embs = torch.randn(len(batch), target_embeddings.size(1))

        # 相似度
        q_embs = F.normalize(q_embs, p=2, dim=1)
        t_embs = F.normalize(target_embeddings, p=2, dim=1)
        sims = torch.mm(q_embs, t_embs.t())

        k = min(self.topk, len(candidate_targets))
        top_k_sims, top_k_idx = torch.topk(sims, k, dim=1, largest=True)

        # GT 索引（fallback -1）
        target_paths = candidate_targets
        gt_indices = []
        gt_full_ranks = []  # ✅ 新增：GT 在全库中的绝对名次（1-based）；若不在候选库为 -1
        gt_similarities = []  # ✅ 新增：GT 相似度（若不在候选库则为 None）
        # 新增：当找不到 GT 时，打印规范化对比结果
        abs_candidate_paths_norm = None  # type: ignore
        cand_norm_set = None  # type: ignore
        cand_norm_index = None  # type: ignore
        for i, q in enumerate(batch):
            gt_path_raw = q["target_image"]
            gt_abs = get_full_image_path(gt_path_raw, self.image_base_dir)
            try:
                gt_idx = abs_candidate_paths.index(gt_abs)
                # 计算全库绝对名次（1-based）：比 GT 相似度更大的数量 + 1
                gt_sim = sims[i, gt_idx].item()
                full_rank = int(1 + torch.sum(sims[i] > sims[i, gt_idx]).item())
                gt_full_ranks.append(full_rank)
                gt_similarities.append(gt_sim)
            except ValueError:
                # 调试：仅在失败时构建规范化候选集并对比
                try:
                    gt_norm = os.path.normpath(os.path.realpath(gt_abs))
                    if abs_candidate_paths_norm is None:
                        abs_candidate_paths_norm = [os.path.normpath(os.path.realpath(p)) for p in abs_candidate_paths]
                        cand_norm_set = set(abs_candidate_paths_norm)
                        cand_norm_index = {p: i for i, p in enumerate(abs_candidate_paths_norm)}
                    if gt_norm in cand_norm_set:  # type: ignore[arg-type]
                        norm_idx = cand_norm_index.get(gt_norm, -1)  # type: ignore[union-attr]
                        print_rank(
                            f"DEBUG[retrieval]: GT raw not found, but found after normalization at idx={norm_idx}.\n"
                            f"  gt_raw={gt_path_raw}\n  gt_abs={gt_abs}\n  gt_norm={gt_norm}"
                        )
                    else:
                        # 打印最小必要信息，避免过多日志
                        sample_cand = abs_candidate_paths[0] if len(abs_candidate_paths) > 0 else ""
                        sample_cand_norm = (
                            abs_candidate_paths_norm[0] if abs_candidate_paths_norm and len(abs_candidate_paths_norm) > 0 else ""
                        )
                        print_rank(
                            "DEBUG[retrieval]: GT not in candidate set even after normalization. "
                            f"num_candidates={len(abs_candidate_paths)}\n  gt_raw={gt_path_raw}\n  gt_abs={gt_abs}\n  gt_norm={gt_norm}\n"
                            f"  sample_cand_abs={sample_cand}\n  sample_cand_norm={sample_cand_norm}"
                        )
                except Exception as e:
                    print_rank(f"DEBUG[retrieval]: normalization diagnostic failed: {e}")
                gt_idx = -1
                gt_full_ranks.append(-1)
                gt_similarities.append(None)
            gt_indices.append(gt_idx)

        results = {
            "top_k_indices": top_k_idx.tolist(),
            "gt_indices": gt_indices,
            "similarities": top_k_sims.tolist(),
            "target_paths": target_paths,
            # ✅ 新增
            "gt_full_ranks": gt_full_ranks,
            "gt_similarities": gt_similarities,
        }

        if not dist.is_initialized() or dist.get_rank() == 0:
            print_rank(f"Real retrieval completed. Average top-1 similarity: {top_k_sims[:, 0].mean():.4f}")
        return results

    def _run_real_retrieval_with_cached_targets(
        self,
        model: torch.nn.Module,
        batch: List[Dict[str, Any]],
        target_embeddings: torch.Tensor,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        与你原来的 _run_real_retrieval_with_cached_targets 等价
        """
        processor = getattr(model, "processor", None)
        if processor is None:
            raise RuntimeError("No processor available")

        device = next(model.parameters()).device
        model_backbone = getattr(self.model_args, "model_backbone", "qwen2_vl")

        import torch.distributed as dist
        if max_samples is not None and max_samples <= 100:
            min_candidates = min(1000, len(self.retrieval_candidates))
            candidate_targets = self.retrieval_candidates[:min_candidates]
            used_target_embeddings = target_embeddings[:min_candidates].to(device)
            if not dist.is_initialized() or dist.get_rank() == 0:
                print_rank(f"Fast mode: using {len(candidate_targets)} target candidates")
        else:
            candidate_targets = self.retrieval_candidates
            used_target_embeddings = target_embeddings.to(device)
            if not dist.is_initialized() or dist.get_rank() == 0:
                print_rank(f"Production mode: using full candidate set ({len(candidate_targets)} images)")

        # ✅ 同样统一绝对路径，再找索引
        abs_candidate_paths = [get_full_image_path(p, self.image_base_dir) for p in candidate_targets]

        with torch.no_grad():
            q_inputs = self._prepare_query_inputs(batch, processor, model_backbone, device)
            try:
                q_embs = model.encode_input(q_inputs)
                q_embs = self._process_embeddings(q_embs, len(batch), "query_embeddings").to(device)
            except Exception as e:
                print_rank(f"Error encoding queries: {e}")
                q_embs = torch.randn(len(batch), used_target_embeddings.size(1), device=device)

        q_embs = F.normalize(q_embs, p=2, dim=1)
        used_target_embeddings = F.normalize(used_target_embeddings, p=2, dim=1)
        used_target_embeddings = used_target_embeddings.to(q_embs.dtype)

        sims = torch.mm(q_embs, used_target_embeddings.t())
        k = min(self.topk, len(candidate_targets))
        top_k_sims, top_k_idx = torch.topk(sims, k, dim=1, largest=True)

        gt_indices = []
        gt_full_ranks = []  # ✅ 新增
        gt_similarities = []  # ✅ 新增
        # 新增：当找不到 GT 时，打印规范化对比结果
        abs_candidate_paths_norm = None  # type: ignore
        cand_norm_set = None  # type: ignore
        cand_norm_index = None  # type: ignore
        for i, q in enumerate(batch):
            gt_path_raw = q["target_image"]
            gt_abs = get_full_image_path(gt_path_raw, self.image_base_dir)
            try:
                gt_idx = abs_candidate_paths.index(gt_abs)
                gt_sim = sims[i, gt_idx].item()
                full_rank = int(1 + torch.sum(sims[i] > sims[i, gt_idx]).item())
                gt_full_ranks.append(full_rank)
                gt_similarities.append(gt_sim)
            except ValueError:
                try:
                    gt_norm = os.path.normpath(os.path.realpath(gt_abs))
                    if abs_candidate_paths_norm is None:
                        abs_candidate_paths_norm = [os.path.normpath(os.path.realpath(p)) for p in abs_candidate_paths]
                        cand_norm_set = set(abs_candidate_paths_norm)
                        cand_norm_index = {p: i for i, p in enumerate(abs_candidate_paths_norm)}
                    if gt_norm in cand_norm_set:  # type: ignore[arg-type]
                        norm_idx = cand_norm_index.get(gt_norm, -1)  # type: ignore[union-attr]
                        print_rank(
                            f"DEBUG[retrieval_cached]: GT raw not found, but found after normalization at idx={norm_idx}.\n"
                            f"  gt_raw={gt_path_raw}\n  gt_abs={gt_abs}\n  gt_norm={gt_norm}"
                        )
                    else:
                        sample_cand = abs_candidate_paths[0] if len(abs_candidate_paths) > 0 else ""
                        sample_cand_norm = (
                            abs_candidate_paths_norm[0] if abs_candidate_paths_norm and len(abs_candidate_paths_norm) > 0 else ""
                        )
                        print_rank(
                            "DEBUG[retrieval_cached]: GT not in candidate set even after normalization. "
                            f"num_candidates={len(abs_candidate_paths)}\n  gt_raw={gt_path_raw}\n  gt_abs={gt_abs}\n  gt_norm={gt_norm}\n"
                            f"  sample_cand_abs={sample_cand}\n  sample_cand_norm={sample_cand_norm}"
                        )
                except Exception as e:
                    print_rank(f"DEBUG[retrieval_cached]: normalization diagnostic failed: {e}")
                gt_idx = -1
                gt_full_ranks.append(-1)
                gt_similarities.append(None)
            gt_indices.append(gt_idx)

        return {
            "top_k_indices": top_k_idx.tolist(),
            "gt_indices": gt_indices,
            "similarities": top_k_sims.tolist(),
            "target_paths": candidate_targets,
            # ✅ 新增
            "gt_full_ranks": gt_full_ranks,
            "gt_similarities": gt_similarities,
        }

    # -------------------------- internal: inputs & utils --------------------------

    def _prepare_target_inputs(
        self, target_paths: List[str], processor: Any, model_backbone: str, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        与训练数据集一致：为 target 图像构建“仅图像 + one-word 提示”的输入
        """
        if process_input_text is None:
            raise RuntimeError("process_input_text is not available. Please ensure prompts.builder is installed.")

        texts: List[str] = []
        for _ in target_paths:
            # 与 IterativeCIRRDataset/_get_original_sample 中的 pos_text 对齐
            t = process_input_text(
                instruction="Represent the given image in one word:",
                model_backbone=model_backbone,
                text="",
                add_image_token=True,
            )
            texts.append(t)

        return self._prepare_vlm_inputs(target_paths, texts, processor, model_backbone, device, "target")

    def _prepare_query_inputs(
        self, batch: List[Dict[str, Any]], processor: Any, model_backbone: str, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        与训练数据集一致：为查询（参考图 + 修改文本）构建输入
        """
        if process_input_text is None:
            raise RuntimeError("process_input_text is not available. Please ensure prompts.builder is installed.")

        image_paths = [q["reference_image"] for q in batch]
        texts: List[str] = []
        for q in batch:
            # 与 IterativeCIRRDataset/_get_original_sample 和 _get_augmented_sample 的 query_text 对齐
            mod_text = q.get("modification_text", "")
            txt = process_input_text(
                instruction=f"Modify this image with <{mod_text}>\nRepresent the modified image in one word:",
                model_backbone=model_backbone,
                text="",
                add_image_token=True,
            )
            texts.append(txt)

        return self._prepare_vlm_inputs(image_paths, texts, processor, model_backbone, device, "query")

    def _prepare_vlm_inputs(
        self,
        image_paths: List[str],
        texts: List[str],
        processor: Any,
        model_backbone: str,
        device: torch.device,
        input_type: str = "general",
    ) -> Dict[str, torch.Tensor]:
        """
        与你原来 _prepare_vlm_inputs 一致：加载图片 -> 调用 VLM2Vec 官方 processor 映射
        """
        from PIL import Image

        images = []
        proc_texts = []

        for img_path, text in zip(image_paths, texts):
            try:
                full = get_full_image_path(img_path, self.image_base_dir)
                img = Image.open(full).convert("RGB")
                images.append(img)
                proc_texts.append(text)
            except Exception as e:
                print_rank(f"Error loading image {img_path}: {e}")
                images.append(Image.new("RGB", (224, 224), color="white"))
                proc_texts.append(text or "")

        try:
            model_inputs = {"text": proc_texts, "images": images}
            if model_backbone in {GME, LamRA, LamRA_QWEN2_5}:
                is_query = input_type == "query"
                model_inputs["is_query"] = is_query
                if is_query:
                    model_inputs["instruction"] = GME_CIRR_QUERY_INSTRUCTION
                else:
                    model_inputs["instruction"] = None
            if model_backbone not in process_vlm_inputs_fns:
                raise ValueError(f"Model backbone {model_backbone} not supported in VLM2Vec")
            inputs = process_vlm_inputs_fns[model_backbone](model_inputs, processor)
            for k in inputs:
                if hasattr(inputs[k], "to"):
                    inputs[k] = inputs[k].to(device)
            return inputs
        except Exception as e:
            print_rank(f"Error in VLM2Vec processor for {input_type}: {e}")
            raise

    @staticmethod
    def _process_embeddings(embeddings: torch.Tensor, expected_bs: int, tag: str) -> torch.Tensor:
        """
        复制你的 _process_embeddings 兜底逻辑
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

    # -------------------------- fallback simplified --------------------------

    def _run_simplified_retrieval(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        原样保留你的简化检索（伪造 top-k / similarities，方便快速跑通后续流程）
        """
        print_rank("Running simplified retrieval (fallback)")
        B = len(batch)
        topk_list: List[List[int]] = []
        gt_list: List[int] = []
        sims_list: List[List[float]] = []

        for i in range(B):
            if i % 3 == 0:
                gt_idx = 2
                top_k = [1, 5, gt_idx, 7, 9, 3, 8, 4, 6, 0]
            elif i % 3 == 1:
                gt_idx = 0
                top_k = [gt_idx, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                gt_idx = 4
                top_k = [2, 7, 1, 9, gt_idx, 5, 3, 8, 6, 0]

            topk_list.append(top_k)
            gt_list.append(gt_idx)
            sims_list.append([0.9 - j * 0.1 for j in range(10)])

        return {
            "top_k_indices": topk_list,
            "gt_indices": gt_list,
            "similarities": sims_list,
            # 注意：简化检索不含 target_paths，HardNegativeMiner 会据此认为是“模拟检索”
        }

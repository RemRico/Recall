import logging
import random
from collections import defaultdict
from typing import Any, List, Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


logger = logging.getLogger(__name__)


class DistributedGroupedBatchSampler(Sampler[List[int]]):
    """
        DDP-friendly grouped batch sampler.

        Samples are grouped by `reference_image` and optionally refined by
        `(reference_image, anchor_text)` when augmentation anchor text exists.
        The sampler builds global batches, slices them evenly per rank, and can
        optionally pad short batches to fixed length.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        pad_to_full: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        ref_key: str = "reference_image",
        no_ref_bucket: str = "__no_ref__",
        debug: bool = False,
        debug_max_batches: int = 5,
        debug_preview_groups: int = 3,
        debug_preview_items: int = 3,
        debug_preview_chars: int = 80,
        debug_preview_small_max_size: int = 4,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.pad_to_full = pad_to_full
        self.drop_last = drop_last
        self.seed = int(seed)
        self.ref_key = ref_key
        self.no_ref_bucket = no_ref_bucket
        self.debug = bool(debug)
        self.debug_max_batches = int(debug_max_batches)
        self.debug_preview_groups = max(0, int(debug_preview_groups))
        self.debug_preview_items = max(1, int(debug_preview_items))
        self.debug_preview_chars = max(8, int(debug_preview_chars))
        self.debug_preview_small_max_size = max(0, int(debug_preview_small_max_size))
        self.preview_cache_char_limit = max(self.debug_preview_chars, 120)
        self.preview_cache: Optional[Dict[int, Dict[str, Any]]] = {} if self.debug else None
        self.index_to_group_idx: Optional[Dict[int, int]] = None

        if world_size is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size, rank = 1, 0
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.epoch = 0

        self.all_indices: List[int] = list(range(len(self.dataset)))

        self.schema_buckets: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
        self.index_signature: Dict[int, Tuple[str, ...]] = {}
        self.index_is_augmented: Dict[int, bool] = {}

        raw_groups: Dict[object, List[int]] = defaultdict(list)
        for i in range(len(self.dataset)):
            try:
                ex = self.dataset[i]
                sig = self._signature_of_example(ex)
                if sig is not None:
                    self.index_signature[i] = sig
                    self.schema_buckets[sig].append(i)
                self.index_is_augmented[i] = bool(ex.get("is_augmented", False))

                group_key = self._build_group_key(ex)
                raw_groups[group_key].append(i)
                if self.preview_cache is not None:
                    self.preview_cache[i] = self._build_preview_payload(ex)
            except Exception as e:
                continue

        self.group_keys: List[object] = []
        self.groups: List[List[int]] = []
        self.group_base_refs: List[str] = []
        self.reference_to_group_indices: Dict[str, List[int]] = defaultdict(list)
        for key, members in raw_groups.items():
            gi = len(self.groups)
            base_ref = self._base_ref_from_group_key(key)
            self.group_keys.append(key)
            self.groups.append(members)
            self.group_base_refs.append(base_ref)
            self.reference_to_group_indices[base_ref].append(gi)
        self.num_groups = len(self.groups)
        if self.debug:
            self.index_to_group_idx = {}
            for gi, members in enumerate(self.groups):
                for sample_idx in members:
                    self.index_to_group_idx[sample_idx] = gi

        self._cached_batches = None
        self._cached_len = None

        if self.debug and (not dist.is_initialized() or self.rank == 0):
            total_aug = sum(1 for v in self.index_is_augmented.values() if v)
            total = len(self.all_indices)
            both_cnt = 0
            only_aug = 0
            only_orig = 0
            for g in self.groups:
                has_aug = any(self.index_is_augmented.get(x, False) for x in g)
                has_orig = any(not self.index_is_augmented.get(x, False) for x in g)
                if has_aug and has_orig:
                    both_cnt += 1
                elif has_aug:
                    only_aug += 1
                elif has_orig:
                    only_orig += 1
            logger.info(
                "[Sampler][rank%d] groups_built: num_groups=%d, total=%d, augmented=%d; "
                "group_types: both=%d, only_aug=%d, only_orig=%d",
                self.rank,
                self.num_groups,
                total,
                total_aug,
                both_cnt,
                only_aug,
                only_orig,
            )
            preview_indices = self._select_preview_group_indices()
            if preview_indices:
                logger.info(
                    "[Sampler][rank%d] group preview (up to %d buckets; <= %d items when possible)",
                    self.rank,
                    self.debug_preview_groups,
                    self.debug_preview_small_max_size,
                )
                for order, gi in enumerate(preview_indices):
                    key_str = self._format_group_key(self.group_keys[gi])
                    members = self.groups[gi]
                    logger.info("  group#%d (idx=%d) %s size=%d", order, gi, key_str, len(members))
                    for idx in members[: self.debug_preview_items]:
                        logger.info("    %s", self._preview_sample(idx))
                    if len(members) > self.debug_preview_items:
                        remain = len(members) - self.debug_preview_items
                        logger.info("    ... +%d more", remain)

        self._debug_pad_same_sig = 0
        self._debug_pad_global = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._cached_batches = None
        self._cached_len = None
        self._debug_pad_same_sig = 0
        self._debug_pad_global = 0

    def _signature_of_example(self, ex) -> Optional[Tuple[str, ...]]:
        try:
            if isinstance(ex, dict):
                keys = [k for k in ex.keys() if isinstance(k, str)]
                return tuple(sorted(keys))
        except Exception:
            pass
        return None

    def _pad_unique(self, base: List[int], target_size: int, rnd: random.Random) -> List[int]:
        if len(base) >= target_size:
            return base[:target_size]
        if not base:
            batch = []
            exclude = set()
            candidates_pool = self.all_indices
        else:
            batch = list(base)
            exclude = set(batch)
            sig = self.index_signature.get(batch[0])
            if sig is not None and sig in self.schema_buckets:
                candidates_pool = self.schema_buckets[sig]
            else:
                candidates_pool = self.all_indices

        need = target_size - len(batch)
        candidates = [i for i in candidates_pool if i not in exclude]
        take = min(need, len(candidates))
        if take > 0:
            batch.extend(rnd.sample(candidates, take))
            need -= take
            exclude.update(batch)
            if candidates_pool is self.all_indices:
                self._debug_pad_global += 1
            else:
                self._debug_pad_same_sig += 1

        if need > 0:
            global_cands = [i for i in self.all_indices if i not in exclude]
            take2 = min(need, len(global_cands))
            if take2 > 0:
                batch.extend(rnd.sample(global_cands, take2))
                need -= take2
                self._debug_pad_global += 1

        while len(batch) < target_size:
            pool = candidates_pool if candidates_pool else (batch if batch else self.all_indices)
            batch.append(rnd.choice(pool))
        return batch

    def _build_group_key(self, ex) -> object:
        """Build grouping key using reference path and optional anchor text."""
        ref = None
        if isinstance(ex, dict):
            ref = ex.get(self.ref_key)
        if isinstance(ref, str):
            base_key = ref
        elif ref is not None:
            base_key = str(ref)
        else:
            base_key = self.no_ref_bucket

        anchor = self._extract_anchor_text(ex)
        if anchor:
            return (base_key, anchor)
        return base_key

    def _base_ref_from_group_key(self, key: object) -> str:
        if isinstance(key, tuple) and len(key) >= 1:
            base = key[0]
        else:
            base = key
        if isinstance(base, str):
            return base
        return str(base)

    def _extract_anchor_text(self, ex) -> Optional[str]:
        if not isinstance(ex, dict):
            return None
        candidates = (
            ex.get("augmentation_group_key"),
            ex.get("original_mod_text"),
        )
        for cand in candidates:
            if isinstance(cand, str):
                stripped = cand.strip()
                if stripped:
                    return stripped
        return None

    def _select_preview_group_indices(self) -> List[int]:
        if self.num_groups == 0 or self.debug_preview_groups <= 0:
            return []
        selected: List[int] = []
        if self.debug_preview_small_max_size > 0:
            for gi, members in enumerate(self.groups):
                if len(members) <= self.debug_preview_small_max_size:
                    selected.append(gi)
                if len(selected) >= self.debug_preview_groups:
                    break
        if not selected:
            limit = min(self.debug_preview_groups, self.num_groups)
            selected = list(range(limit))
        return selected

    def _build_preview_payload(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "is_aug": bool(ex.get("is_augmented", False)),
            "reference": self._sanitize_path(ex.get(self.ref_key)),
            "anchor": self._sanitize_text(self._extract_anchor_text(ex)),
            "mod_text": self._sanitize_text(ex.get("modification_text")),
            "orig_mod_text": self._sanitize_text(ex.get("original_mod_text")),
            "caption": self._sanitize_text(ex.get("caption")),
            "target": self._sanitize_path(self._extract_target_path(ex)),
        }

    def _preview_sample(self, idx: int, max_chars: Optional[int] = None) -> str:
        info: Optional[Dict[str, Any]] = None
        if self.preview_cache is not None:
            info = self.preview_cache.get(idx)
        if info is None:
            try:
                ex = self.dataset[idx]
            except Exception as exc:
                return f"idx={idx} !{exc}"
            info = self._build_preview_payload(ex)
            if self.preview_cache is not None:
                self.preview_cache[idx] = info
        return self._format_preview(idx, info, max_chars=max_chars)

    def _format_preview(self, idx: int, info: Dict[str, Any], max_chars: Optional[int] = None) -> str:
        limit = max_chars if max_chars is not None else self.debug_preview_chars
        if limit < 4:
            limit = 4
        parts = [f"idx={idx}", "aug" if info.get("is_aug") else "orig"]
        reference = info.get("reference")
        if reference:
            parts.append(f"ref={self._short_path(reference)}")
        anchor = info.get("anchor")
        if anchor:
            parts.append(f"anchor={self._truncate(anchor, limit)}")
        mod_text = info.get("mod_text")
        caption = info.get("caption")
        orig_mod = info.get("orig_mod_text")
        if mod_text:
            parts.append(f"mod={self._truncate(mod_text, limit)}")
        elif caption:
            parts.append(f"cap={self._truncate(caption, limit)}")
        if orig_mod and (not mod_text or orig_mod != mod_text):
            parts.append(f"orig={self._truncate(orig_mod, limit)}")
        target = info.get("target")
        if target:
            parts.append(f"tgt={self._short_path(target)}")
        return " | ".join(parts)

    def _truncate(self, text: str, limit: int) -> str:
        if not text:
            return ""
        clean = str(text).replace("\n", " ").replace("\t", " ").strip()
        if len(clean) <= limit:
            return clean
        if limit <= 3:
            return clean[:limit]
        return clean[: limit - 3] + "..."

    def _short_path(self, path: str) -> str:
        if not path:
            return ""
        normalized = str(path).replace("\\", "/").strip().rstrip("/")
        parts = [p for p in normalized.split("/") if p]
        if not parts:
            return normalized
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1]

    def _sanitize_text(self, value: Optional[Any]) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value.strip()
        else:
            text = str(value).strip()
        return self._truncate(text, self.preview_cache_char_limit)

    def _sanitize_path(self, value: Optional[Any]) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            path = value.strip()
        else:
            path = str(value).strip()
        return self._truncate(path, self.preview_cache_char_limit)

    def _extract_target_path(self, ex) -> str:
        if not isinstance(ex, dict):
            return ""
        target = ex.get("target_image")
        if isinstance(target, str):
            return target
        pos_image = ex.get("pos_image")
        if pos_image:
            path = self._extract_path_from_image_dict(pos_image)
            if path:
                return path
        return ""

    def _extract_path_from_image_dict(self, image_dict) -> str:
        if not isinstance(image_dict, dict):
            return ""
        paths = image_dict.get("paths")
        if isinstance(paths, (list, tuple)) and paths:
            first = paths[0]
            if isinstance(first, str):
                return first
        return ""

    def _format_group_key(self, key: object) -> str:
        if isinstance(key, tuple) and len(key) == 2:
            base = self._short_path(key[0])
            anchor = self._truncate(str(key[1]), self.debug_preview_chars)
            return f"ref={base} | anchor={anchor}"
        if isinstance(key, str):
            return f"ref={self._short_path(key)}"
        return f"ref={str(key)}"

    def _build_all_batches(self):
        """Build global batches for the epoch and slice batches for this rank."""
        if self.shuffle:
            rnd = random.Random(self.seed + self.epoch)
            ref_order = list(self.reference_to_group_indices.keys())
            rnd.shuffle(ref_order)
        else:
            rnd = None
            ref_order = list(self.reference_to_group_indices.keys())

        group_indices: List[int] = []
        for ref in ref_order:
            per_ref_groups = list(self.reference_to_group_indices[ref])
            if self.shuffle and rnd is not None:
                rnd.shuffle(per_ref_groups)
            group_indices.extend(per_ref_groups)

        batches: List[List[int]] = []
        pending: List[int] = []
        for gi in group_indices:
            pending.extend(self.groups[gi])
            while len(pending) >= self.batch_size:
                batch = pending[: self.batch_size]
                batches.append(batch)
                pending = pending[self.batch_size :]

        if pending:
            if self.drop_last:
                pending = []
            elif self.pad_to_full:
                rnd = random.Random(self.seed + self.epoch + 2024)
                padded = self._pad_unique(list(pending), self.batch_size, rnd)
                batches.append(padded)
                pending = []
            else:
                batches.append(list(pending))
                pending = []

        my_batches = [b for i, b in enumerate(batches) if (i % self.world_size) == self.rank]

        if self.pad_to_full:
            fixed = []
            rnd = random.Random(self.seed + self.epoch + 4096 + self.rank)
            for b in my_batches:
                if 0 < len(b) < self.batch_size:
                    c = self._pad_unique(b, self.batch_size, rnd)
                    fixed.append(c)
                elif len(b) > self.batch_size:
                    fixed.append(b[: self.batch_size])
                else:
                    fixed.append(b)
            my_batches = fixed

        self._cached_batches = my_batches
        self._cached_len = len(my_batches)

        if self.debug and (not dist.is_initialized() or self.rank == 0):
            try:
                num_batches = len(my_batches)
                batches_with_aug = 0
                total_aug_items = 0
                preview_lines = []
                for bi, b in enumerate(my_batches[: self.debug_max_batches]):
                    aug_flags = [self.index_is_augmented.get(x, False) for x in b]
                    aug_count = sum(1 for f in aug_flags if f)
                    total_aug_items += aug_count
                    if aug_count > 0:
                        batches_with_aug += 1
                    sample_snippets = ", ".join(
                        self._preview_sample(idx, max_chars=self.debug_preview_chars)
                        for idx in b[: self.debug_preview_items]
                    )
                    if len(b) > self.debug_preview_items:
                        sample_snippets = f"{sample_snippets}, ..."
                    group_hint = ""
                    if self.index_to_group_idx is not None and b:
                        first_idx = b[0]
                        gi = self.index_to_group_idx.get(first_idx)
                        if gi is not None:
                            group_hint = f", group_key={self._format_group_key(self.group_keys[gi])}"
                    preview_lines.append(
                        f"  batch[{bi}] size={len(b)}, aug={aug_count}{group_hint}, samples=[{sample_snippets}]"
                    )
                if num_batches > self.debug_max_batches:
                    for b in my_batches[self.debug_max_batches:]:
                        if any(self.index_is_augmented.get(x, False) for x in b):
                            batches_with_aug += 1
                logger.info(
                    "[Sampler][rank%d][epoch%d] my_batches=%d, batches_with_aug=%d, "
                    "total_aug_items_in_preview=%d, pad_same_sig_calls=%d, pad_global_calls=%d\n%s",
                    self.rank,
                    self.epoch,
                    num_batches,
                    batches_with_aug,
                    total_aug_items,
                    self._debug_pad_same_sig,
                    self._debug_pad_global,
                    "\n".join(preview_lines),
                )
            except Exception:
                pass

    def __iter__(self) -> Iterator[List[int]]:
        if self._cached_batches is None:
            self._build_all_batches()
        for b in self._cached_batches:
            yield b

    def __len__(self) -> int:
        if self._cached_len is None:
            self._build_all_batches()
        return self._cached_len

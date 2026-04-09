"""Category-aware distributed samplers for FashionIQ training."""

import logging
import random
from collections import defaultdict
from typing import List, Dict, Iterator, Optional

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


logger = logging.getLogger(__name__)


class DistributedCategoryBatchSampler(Sampler[List[int]]):
    """
    DDP-friendly category-alternating batch sampler for FashionIQ.

    It groups samples by category, builds category-homogeneous batches,
    then distributes batches evenly across ranks.
    
    Args:
        dataset: Dataset (must contain the `category` field)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle within each category
        drop_last: Whether to drop the final incomplete batch
        seed: Random seed
        world_size: DDP world size
        rank: DDP rank
        category_key: Category field name (default: `category`)
        category_rotation: Category rotation order (default: ['dress', 'shirt', 'toptee'])
        debug: Whether to emit debug summaries
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        category_key: str = "category",
        category_rotation: Optional[List[str]] = None,
        debug: bool = False,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.category_key = category_key
        self.debug = bool(debug)

        if category_rotation is None:
            self.category_rotation = ['dress', 'shirt', 'toptee']
        else:
            self.category_rotation = list(category_rotation)

        if world_size is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size, rank = 1, 0
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.epoch = 0

        self.category_indices: Dict[str, List[int]] = defaultdict(list)

        if hasattr(self.dataset, 'annotations'):
            for i, ann in enumerate(self.dataset.annotations):
                cat = ann.get(self.category_key, "unknown")
                self.category_indices[cat].append(i)

            if hasattr(self.dataset, 'augmented_samples') and len(self.dataset.augmented_samples) > 0:
                offset = len(self.dataset.annotations)
                for i, aug_sample in enumerate(self.dataset.augmented_samples):
                    cat = aug_sample.get(self.category_key, "unknown")
                    self.category_indices[cat].append(offset + i)
        elif hasattr(self.dataset, 'get_category'):
            for i in range(len(self.dataset)):
                try:
                    cat = self.dataset.get_category(i)
                    self.category_indices[cat].append(i)
                except Exception:
                    self.category_indices["unknown"].append(i)
        else:
            logger.warning("[CategorySampler] dataset has no fast category access; building category map may be slow")
            for i in range(len(self.dataset)):
                try:
                    ex = self.dataset[i]
                    cat = ex.get(self.category_key, "unknown")
                    self.category_indices[cat].append(i)
                except Exception:
                    continue

        self.categories = sorted(self.category_indices.keys())
        self._cached_batches = None
        self._cached_len = None

        if self.debug and (not dist.is_initialized() or self.rank == 0):
            logger.info("[CategorySampler] Initialized")
            logger.info("  Categories found: %s", self.categories)
            for cat in self.categories:
                logger.info("    - %s: %d samples", cat, len(self.category_indices[cat]))
            logger.info("  Batch size: %d", self.batch_size)
            logger.info("  Category rotation: %s", self.category_rotation)
            logger.info("  DDP: world_size=%d, rank=%d", self.world_size, self.rank)

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across workers."""
        self.epoch = int(epoch)
        self._cached_batches = None
        self._cached_len = None

    def _build_all_batches(self):
        """Build all global batches and slice them for the current rank."""
        rnd = random.Random(self.seed + self.epoch)

        category_pools: Dict[str, List[int]] = {}
        for cat in self.category_rotation:
            if cat in self.category_indices:
                indices = list(self.category_indices[cat])
                if self.shuffle:
                    rnd.shuffle(indices)
                category_pools[cat] = indices
            else:
                category_pools[cat] = []

        all_batches: List[List[int]] = []
        category_idx = 0

        while True:
            if all(len(pool) == 0 for pool in category_pools.values()):
                break

            current_cat = self.category_rotation[category_idx % len(self.category_rotation)]
            category_idx += 1

            if len(category_pools[current_cat]) == 0:
                continue

            pool = category_pools[current_cat]
            if len(pool) >= self.batch_size:
                batch = pool[:self.batch_size]
                category_pools[current_cat] = pool[self.batch_size:]
            else:
                if not self.drop_last and len(pool) > 0:
                    batch = pool
                    category_pools[current_cat] = []
                else:
                    category_pools[current_cat] = []
                    continue

            all_batches.append(batch)

        if self.shuffle:
            rnd.shuffle(all_batches)

        num_batches_per_rank = len(all_batches) // self.world_size

        if self.drop_last:
            all_batches = all_batches[:num_batches_per_rank * self.world_size]
        else:
            remainder = len(all_batches) % self.world_size
            if remainder != 0:
                num_to_pad = self.world_size - remainder
                padding_batches = all_batches[:num_to_pad]
                all_batches.extend(padding_batches)
                num_batches_per_rank = len(all_batches) // self.world_size

        my_batches = [b for i, b in enumerate(all_batches) if (i % self.world_size) == self.rank]

        assert len(my_batches) == num_batches_per_rank, \
            f"Rank {self.rank}: got {len(my_batches)} batches, expected {num_batches_per_rank}"

        if not self.drop_last:
            fixed_batches = []
            for b in my_batches:
                if len(b) < self.batch_size:
                    original_len = len(b)
                    if original_len > 0:
                        while len(b) < self.batch_size:
                            b.append(b[len(b) % original_len])
                    fixed_batches.append(b)
                elif len(b) > self.batch_size:
                    fixed_batches.append(b[:self.batch_size])
                else:
                    fixed_batches.append(b)
            my_batches = fixed_batches

        self._cached_batches = my_batches
        self._cached_len = len(my_batches)

        if self.debug and (not dist.is_initialized() or self.rank == 0):
            logger.info("[CategorySampler][Epoch %d][Rank %d]", self.epoch, self.rank)
            logger.info("  Total batches (global): %d", len(all_batches))
            logger.info("  My batches: %d", len(my_batches))

            cat_counts = defaultdict(int)
            idx_to_cat = {}
            for cat, indices in self.category_indices.items():
                for idx in indices:
                    idx_to_cat[idx] = cat

            for batch in my_batches[:min(10, len(my_batches))]:
                if len(batch) > 0:
                    sample_cat = idx_to_cat.get(batch[0], "unknown")
                    cat_counts[sample_cat] += 1

            logger.info("  Category distribution (first 10 batches):")
            for cat, count in sorted(cat_counts.items()):
                logger.info("    - %s: %d batches", cat, count)

            logger.info("  Sample batches:")
            for i, batch in enumerate(my_batches[:3]):
                if len(batch) > 0:
                    sample_cat = idx_to_cat.get(batch[0], "unknown")
                    logger.info("    batch[%d]: size=%d, category=%s, ids=%s...", i, len(batch), sample_cat, batch[:3])

    def __iter__(self) -> Iterator[List[int]]:
        if self._cached_batches is None:
            self._build_all_batches()
        for b in self._cached_batches:
            yield b

    def __len__(self) -> int:
        if self._cached_len is None:
            self._build_all_batches()
        return self._cached_len


class DistributedCategoryBalancedSampler(Sampler[int]):
    """
    Category-balanced distributed sampler for non-batched DataLoader flows.

    It alternates categories to prevent domination by any single class.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 42,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        category_key: str = "category",
        debug: bool = False,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = int(seed)
        self.category_key = category_key
        self.debug = bool(debug)

        if world_size is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size, rank = 1, 0
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.epoch = 0

        self.category_indices: Dict[str, List[int]] = defaultdict(list)
        for i in range(len(self.dataset)):
            try:
                ex = self.dataset[i]
                cat = ex.get(self.category_key, "unknown")
                self.category_indices[cat].append(i)
            except Exception:
                continue

        self.categories = sorted(self.category_indices.keys())

        if self.debug and (not dist.is_initialized() or self.rank == 0):
            logger.info("[CategoryBalancedSampler] Initialized")
            for cat in self.categories:
                logger.info("  - %s: %d samples", cat, len(self.category_indices[cat]))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rnd = random.Random(self.seed + self.epoch)

        # Shuffle each category pool.
        category_pools = {}
        for cat in self.categories:
            indices = list(self.category_indices[cat])
            if self.shuffle:
                rnd.shuffle(indices)
            category_pools[cat] = indices

        # Interleave categories until all samples are consumed.
        all_indices = []
        while any(len(pool) > 0 for pool in category_pools.values()):
            for cat in self.categories:
                if len(category_pools[cat]) > 0:
                    all_indices.append(category_pools[cat].pop(0))

        # DDP: each rank takes its own shard.
        my_indices = [idx for i, idx in enumerate(all_indices) if (i % self.world_size) == self.rank]

        return iter(my_indices)

    def __len__(self) -> int:
        total = sum(len(indices) for indices in self.category_indices.values())
        return (total + self.world_size - 1) // self.world_size


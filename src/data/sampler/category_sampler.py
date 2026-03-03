"""
Category-based Batch Sampler for FashionIQ
按类别（dress/shirt/toptee）交替采样，每个batch只包含同一类别的样本
"""

import random
from collections import defaultdict
from typing import List, Dict, Iterator, Optional

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


class DistributedCategoryBatchSampler(Sampler[List[int]]):
    """
    DDP-friendly 类别交替采样器（专为FashionIQ设计）：
      - 按category（dress/shirt/toptee）分组
      - 每个batch只包含同一category的样本
      - 交替使用不同category，确保训练平衡
      - DDP友好：每个rank获得均匀的batch分配
    
    Args:
        dataset: 数据集（需要包含'category'字段）
        batch_size: 每个batch的样本数
        shuffle: 是否在每个category内打乱
        drop_last: 是否丢弃最后不足batch_size的batch
        seed: 随机种子
        world_size: DDP world size
        rank: DDP rank
        category_key: 类别字段名（默认'category'）
        category_rotation: 类别轮换顺序（默认['dress', 'shirt', 'toptee']）
        debug: 是否打印调试信息
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

        # 默认类别轮换顺序
        if category_rotation is None:
            self.category_rotation = ['dress', 'shirt', 'toptee']
        else:
            self.category_rotation = list(category_rotation)

        # DDP info
        if world_size is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size, rank = 1, 0
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.epoch = 0

        # 按类别分组索引（避免触发完整数据加载）
        self.category_indices: Dict[str, List[int]] = defaultdict(list)
        
        # 尝试快速访问category（不加载图像）
        if hasattr(self.dataset, 'annotations'):
            # FashionIQ/CIRR等有annotations属性的数据集
            # 处理原始样本（annotations）
            for i, ann in enumerate(self.dataset.annotations):
                cat = ann.get(self.category_key, "unknown")
                self.category_indices[cat].append(i)
            
            # 🆕 处理增强样本（augmented_samples）
            if hasattr(self.dataset, 'augmented_samples') and len(self.dataset.augmented_samples) > 0:
                offset = len(self.dataset.annotations)  # 增强样本的索引从原始样本数量开始
                for i, aug_sample in enumerate(self.dataset.augmented_samples):
                    cat = aug_sample.get(self.category_key, "unknown")
                    self.category_indices[cat].append(offset + i)
        elif hasattr(self.dataset, 'get_category'):
            # 如果数据集提供了快速获取category的方法
            for i in range(len(self.dataset)):
                try:
                    cat = self.dataset.get_category(i)
                    self.category_indices[cat].append(i)
                except Exception:
                    self.category_indices["unknown"].append(i)
        else:
            # 回退：完整加载（可能很慢）
            print(f"[CategorySampler] Warning: dataset has no fast category access, this may be slow...")
            for i in range(len(self.dataset)):
                try:
                    ex = self.dataset[i]
                    cat = ex.get(self.category_key, "unknown")
                    self.category_indices[cat].append(i)
                except Exception:
                    continue

        # 统计信息
        self.categories = sorted(self.category_indices.keys())
        self._cached_batches = None
        self._cached_len = None

        if self.debug and (not dist.is_initialized() or self.rank == 0):
            print(f"\n[CategorySampler] Initialized:")
            print(f"  Categories found: {self.categories}")
            for cat in self.categories:
                print(f"    - {cat}: {len(self.category_indices[cat])} samples")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Category rotation: {self.category_rotation}")
            print(f"  DDP: world_size={self.world_size}, rank={self.rank}\n")

    def set_epoch(self, epoch: int):
        """设置epoch用于随机种子"""
        self.epoch = int(epoch)
        self._cached_batches = None
        self._cached_len = None

    def _build_all_batches(self):
        """
        构建所有batches，交替使用不同category
        """
        rnd = random.Random(self.seed + self.epoch)

        # Step 1: 为每个category准备打乱后的索引
        category_pools: Dict[str, List[int]] = {}
        for cat in self.category_rotation:
            if cat in self.category_indices:
                indices = list(self.category_indices[cat])
                if self.shuffle:
                    rnd.shuffle(indices)
                category_pools[cat] = indices
            else:
                category_pools[cat] = []

        # Step 2: 交替从各category抽取batch
        all_batches: List[List[int]] = []
        category_idx = 0
        
        while True:
            # 检查是否所有category都已用尽
            if all(len(pool) == 0 for pool in category_pools.values()):
                break
            
            # 选择当前category
            current_cat = self.category_rotation[category_idx % len(self.category_rotation)]
            category_idx += 1
            
            # 跳过已用尽的category
            if len(category_pools[current_cat]) == 0:
                continue
            
            # 从当前category抽取一个batch
            pool = category_pools[current_cat]
            if len(pool) >= self.batch_size:
                batch = pool[:self.batch_size]
                category_pools[current_cat] = pool[self.batch_size:]
            else:
                # 不足一个batch
                if not self.drop_last and len(pool) > 0:
                    batch = pool
                    category_pools[current_cat] = []
                else:
                    # drop_last=True，丢弃剩余样本
                    category_pools[current_cat] = []
                    continue
            
            all_batches.append(batch)

        # Step 3: 全局打乱所有batches（保持每个batch内部是同category）
        if self.shuffle:
            rnd.shuffle(all_batches)

        # Step 4: DDP分配：确保所有rank的batch数量相同（关键：避免DDP hang）
        # 计算每个rank应该有多少batch
        num_batches_per_rank = len(all_batches) // self.world_size
        
        if self.drop_last:
            # Drop掉多余的batches，确保可以均匀分配
            all_batches = all_batches[:num_batches_per_rank * self.world_size]
        else:
            # 不drop，而是复制一些batches来padding（保证所有rank batch数相同）
            remainder = len(all_batches) % self.world_size
            if remainder != 0:
                # 需要padding
                num_to_pad = self.world_size - remainder
                # 从前面复制batches来padding
                padding_batches = all_batches[:num_to_pad]
                all_batches.extend(padding_batches)
                num_batches_per_rank = len(all_batches) // self.world_size
        
        # 使用round-robin分配
        my_batches = [b for i, b in enumerate(all_batches) if (i % self.world_size) == self.rank]
        
        # 验证：所有rank应该有相同数量的batches
        assert len(my_batches) == num_batches_per_rank, \
            f"Rank {self.rank}: got {len(my_batches)} batches, expected {num_batches_per_rank}"
        
        # 双保险：确保每个batch长度都等于batch_size（防止DDP hang）
        if not self.drop_last:
            # 如果有不足batch_size的batch，复制补齐
            fixed_batches = []
            for b in my_batches:
                if len(b) < self.batch_size:
                    # 从batch内重复样本补齐到batch_size
                    original_len = len(b)
                    if original_len > 0:
                        while len(b) < self.batch_size:
                            b.append(b[len(b) % original_len])
                    fixed_batches.append(b)
                elif len(b) > self.batch_size:
                    # 截断（理论上不应该发生）
                    fixed_batches.append(b[:self.batch_size])
                else:
                    fixed_batches.append(b)
            my_batches = fixed_batches

        self._cached_batches = my_batches
        self._cached_len = len(my_batches)

        # 调试输出
        if self.debug and (not dist.is_initialized() or self.rank == 0):
            print(f"\n[CategorySampler][Epoch {self.epoch}][Rank {self.rank}]")
            print(f"  Total batches (global): {len(all_batches)}")
            print(f"  My batches: {len(my_batches)}")
            
            # 统计每个category的batch数（基于索引反查，避免加载数据）
            cat_counts = defaultdict(int)
            idx_to_cat = {}
            for cat, indices in self.category_indices.items():
                for idx in indices:
                    idx_to_cat[idx] = cat
            
            for batch in my_batches[:min(10, len(my_batches))]:
                if len(batch) > 0:
                    sample_cat = idx_to_cat.get(batch[0], "unknown")
                    cat_counts[sample_cat] += 1
            
            print(f"  Category distribution (first 10 batches):")
            for cat, count in sorted(cat_counts.items()):
                print(f"    - {cat}: {count} batches")
            
            # 显示前几个batch的组成
            print(f"  Sample batches:")
            for i, batch in enumerate(my_batches[:3]):
                if len(batch) > 0:
                    sample_cat = idx_to_cat.get(batch[0], "unknown")
                    print(f"    batch[{i}]: size={len(batch)}, category={sample_cat}, ids={batch[:3]}...")
            print()

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
    类别平衡的分布式采样器（不成批，用于标准DataLoader）
    
    确保三个category（dress/shirt/toptee）在采样时保持平衡，
    避免某个category主导训练。
    
    使用场景：配合标准DataLoader使用，不需要batch_sampler时
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

        # DDP info
        if world_size is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size, rank = 1, 0
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.epoch = 0

        # 按类别分组
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
            print(f"\n[CategoryBalancedSampler] Initialized:")
            for cat in self.categories:
                print(f"  - {cat}: {len(self.category_indices[cat])} samples")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rnd = random.Random(self.seed + self.epoch)

        # 为每个category打乱
        category_pools = {}
        for cat in self.categories:
            indices = list(self.category_indices[cat])
            if self.shuffle:
                rnd.shuffle(indices)
            category_pools[cat] = indices

        # 交替从各category抽取，直到所有样本用尽
        all_indices = []
        while any(len(pool) > 0 for pool in category_pools.values()):
            for cat in self.categories:
                if len(category_pools[cat]) > 0:
                    all_indices.append(category_pools[cat].pop(0))

        # DDP: 每个rank取自己的份额
        my_indices = [idx for i, idx in enumerate(all_indices) if (i % self.world_size) == self.rank]

        return iter(my_indices)

    def __len__(self) -> int:
        total = sum(len(indices) for indices in self.category_indices.values())
        return (total + self.world_size - 1) // self.world_size


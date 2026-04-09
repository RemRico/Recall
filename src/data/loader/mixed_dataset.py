"""Dataset initialization utilities for iterative retrieval training."""

from typing import Any, Dict

from datasets.distributed import split_dataset_by_node

from src.data.dataset.base_pair_dataset import AutoPairDataset
from src.data.dataset.hf_datasets import interleave_datasets
from src.utils import print_master
import torch

from ..dataset.cirr import IterativeCIRRDataset
from ..dataset.fashioniq import IterativeFashionIQDataset


def init_mixed_dataset(
    dataset_config: Dict[str, Dict[str, Any]],
    model_args: Any,
    data_args: Any,
    training_args: Any,
):
    """Initialize one dataset or an interleaved dataset according to config."""
    weights = [d["weight"] for d in dataset_config.values()]
    w_sum = sum(weights)
    probs = [w / w_sum for w in weights]
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    train_datasets = []
    for data_idx, (global_dataset_name, single_dataset_cfg) in enumerate(dataset_config.items()):
        train_dataset = AutoPairDataset.instantiate(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            **single_dataset_cfg,
        )
        print_master(
            f"\t\tDataset#{data_idx} (dataset_parser={single_dataset_cfg.get('dataset_parser', 'n/a')}): "
            f"{global_dataset_name}, num_rows={train_dataset.num_rows}, prob={probs[data_idx] * 100.0}"
        )
        train_datasets.append(train_dataset)

    if training_args.interleave_batch_size and training_args.interleave_batch_size <= 1.0:
        interleave_batch_size = training_args.per_device_train_batch_size * world_size * training_args.interleave_batch_size
    else:
        interleave_batch_size = training_args.interleave_batch_size
    total_num_rows = sum([d.num_rows for d in train_datasets])
    print_master(f"\nInitializing interleave datasets:"
                 f"\n\t\tworld_size={world_size}"
                 f"\n\t\ttotal num rows={total_num_rows}"
                 f"\n\t\tglobal batch size={training_args.per_device_train_batch_size * world_size}"
                 f"\n\t\testimated num step per epoch={total_num_rows/(training_args.per_device_train_batch_size * world_size)}"
                 f"\n\t\tinterleave_batch_size={interleave_batch_size}"
                 )
    assert total_num_rows >= (training_args.per_device_train_batch_size * world_size), \
        f"total_num_rows(={total_num_rows}) must be greater than or equal to global batch size (={training_args.per_device_train_batch_size * world_size}), since the last batch will be dropped."

    if len(train_datasets) > 1:
        train_dataset = interleave_datasets(train_datasets, probabilities=probs, batch_size=interleave_batch_size,
                                            seed=training_args.seed, stopping_strategy=training_args.interleave_stopping_strategy)

        # The interleaved IterableDataset does not expose `.shard()`, so split by node here.
        if torch.distributed.is_initialized():
            print_master("Applying split_dataset_by_node to the interleaved dataset.")
            train_dataset = split_dataset_by_node(train_dataset, rank=torch.distributed.get_rank(), world_size=world_size)
    else:
        # For a single custom dataset, Trainer will call its internal `.shard()`.
        train_dataset = train_datasets[0]
        print_master("Skipping split_dataset_by_node for single custom dataset; Trainer will use .shard() method.")

    return train_dataset

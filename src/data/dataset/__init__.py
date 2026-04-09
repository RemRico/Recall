"""Dataset package exports and optional registry wiring."""

from .base_iterative_dataset import IterativeRetrievalDataset
from .cirr import IterativeCIRRDataset
from .fashioniq import IterativeFashionIQDataset

# Optional backward-compatible registration.
try:
    from .base_pair_dataset import AutoPairDataset
    AutoPairDataset.registry["IterativeCIRRDataset"] = IterativeCIRRDataset
    AutoPairDataset.registry["IterativeFashionIQDataset"] = IterativeFashionIQDataset
except Exception:
    pass

__all__ = [
    "IterativeRetrievalDataset",
    "IterativeCIRRDataset",
    "IterativeFashionIQDataset",
]

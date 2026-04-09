# utils/logging.py
from __future__ import annotations
import os
import sys
import time

def is_dist_initialized():
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False

def get_rank() -> int:
    if is_dist_initialized():
        import torch.distributed as dist
        try:
            return dist.get_rank()
        except Exception:
            return 0
    # Fallback to environment variable (some launchers inject it).
    return int(os.environ.get("RANK", "0"))

def is_main_process() -> bool:
    return get_rank() == 0

def _ts() -> str:
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return t

def print_rank(msg: str) -> None:
    """
    Unified printing for distributed/single-process modes:
    - Automatically prefixes timestamp and rank
    - Never raises, to avoid interrupting training
    """
    try:
        rank = get_rank()
        sys.stdout.write(f"[{_ts()}][rank {rank}] {msg}\n")
        sys.stdout.flush()
    except Exception:
        # Most conservative fallback.
        print(msg)

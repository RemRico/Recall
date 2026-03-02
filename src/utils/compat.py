# src/utils/compat.py
import logging as _logging
_logging.basicConfig(level=_logging.DEBUG,
                     format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = _logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hugging Face compatibility helpers
# ---------------------------------------------------------------------------
try:
    import transformers
    from packaging import version as _pkg_version
    from transformers.utils import versions as _hf_versions

    if not hasattr(_hf_versions, "_mycomp_original_require_version"):
        _hf_versions._mycomp_original_require_version = _hf_versions.require_version

        _RELAXED_REQUIREMENTS = {"transformers<4.52.0"}

        def _patched_require_version(requirement: str, hint: str | None = None):
            normalized = requirement.replace(" ", "")
            if normalized in _RELAXED_REQUIREMENTS:
                current_ver = transformers.__version__
                target_ver = normalized.split("<", 1)[-1]
                if _pkg_version.parse(current_ver) >= _pkg_version.parse(target_ver):
                    logger.warning(
                        "Bypassing requirement '%s' because transformers==%s is installed. "
                        "Please ensure downstream code is compatible.",
                        requirement,
                        current_ver,
                    )
                    return True
            return _hf_versions._mycomp_original_require_version(requirement, hint)

        _patched_require_version.__name__ = "_patched_require_version"
        _hf_versions.require_version = _patched_require_version
except Exception as _compat_err:
    logger.debug("Failed to patch transformers.require_version: %s", _compat_err)

import os
import torch

def _is_dist_initialized() -> bool:
    try:
        return torch.distributed.is_available() and torch.distributed.is_initialized()
    except Exception:
        return False

def print_rank(message: str):
    """If distributed is initialized, print the rank."""
    if _is_dist_initialized():
        logger.info(f'rank{torch.distributed.get_rank()}: ' + str(message))
    else:
        logger.info(str(message))

def print_master(message: str):
    """If distributed is initialized print only on rank 0."""
    if _is_dist_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(str(message))
    else:
        logger.info(str(message))

def find_latest_checkpoint(output_dir: str):
    """Scan the output directory and return the latest checkpoint path."""
    if not os.path.exists(output_dir):
        return None

    checkpoints = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None

    # Sort by checkpoint number and return the latest one
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return latest_checkpoint

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

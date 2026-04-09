# utils/path_utils.py
from __future__ import annotations
from typing import List, Iterable
import os
import glob

def normalize_path(path: str) -> str:
    """
    Normalize a path:
    - Expand user-home and environment variables
    - Normalize separators and relative markers such as `..`
    - Do not force absolute conversion (caller decides)
    """
    if not isinstance(path, str):
        path = str(path)
    return os.path.normpath(os.path.expandvars(os.path.expanduser(path)))

def get_full_image_path(image_path: str, base_dir: str) -> str:
    """
    Unified path join equivalent to the original behavior:
    - If input is './xxx', strip './' and join with `base_dir`
    - If input is absolute, return as-is
    - Otherwise join relative path with `base_dir`
    - Normalize once before returning
    """
    if not isinstance(image_path, str):
        image_path = str(image_path)

    base_dir = normalize_path(base_dir)
    p = image_path.strip()

    if p.startswith("./"):
        p = p[2:]
        full = os.path.join(base_dir, p)
    elif os.path.isabs(p):
        full = p
    else:
        full = os.path.join(base_dir, p)

    return normalize_path(full)

def file_exists(path: str) -> bool:
    try:
        return os.path.exists(normalize_path(path))
    except Exception:
        return False

def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def list_images_recursive(root: str, patterns: Iterable[str] = ("**/*.jpg","**/*.jpeg","**/*.png","**/*.bmp","**/*.webp","**/*.tiff")) -> List[str]:
    root = normalize_path(root)
    results: List[str] = []
    for pat in patterns:
        results.extend(glob.glob(os.path.join(root, pat), recursive=True))
    results = [normalize_path(p) for p in results]
    return sorted(results)

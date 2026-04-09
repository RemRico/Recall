# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import random
from typing import Dict

from PIL import Image as PILImage
import numpy as np

_THIS_DIR = os.path.dirname(__file__)
_TMPL = os.path.join(_THIS_DIR, "templates.txt")

def _load_templates():
    with open(_TMPL, "r", encoding="utf-8") as f:
        raw = f.read()
    hard, diversity = [], []
    bucket = None
    for line in raw.splitlines():
        t = line.strip()
        if t == "[[HARD_NEGATIVE]]":
            bucket = hard
            continue
        if t == "[[DIVERSITY]]":
            bucket = diversity
            continue
        if bucket is not None:
            if t and not t.startswith("#"):
                bucket.append(line)

    # Split by blank lines.
    def split_blocks(lines):
        blocks, cur = [], []
        for ln in lines:
            if ln.strip() == "" and cur:
                blocks.append("\n".join(cur).strip())
                cur = []
            else:
                cur.append(ln)
        if cur:
            blocks.append("\n".join(cur).strip())
        return blocks

    return split_blocks(hard), split_blocks(diversity)

_HARD, _DIV = _load_templates()

def create_llava_prompt_enhanced(original_text: str, is_hard_negative_context: bool = False) -> str:
    """
    Kept API. Choose from templates and format {text}.
    """
    pool = _HARD if is_hard_negative_context else _DIV
    tmpl = random.choice(pool) if pool else "USER: {text}\nA:"
    return tmpl.replace("{text}", original_text)

def _concat_side_by_side(ref_img, target_img):
    # `ref_img` and `target_img` are PIL images.
    ref = np.array(ref_img)
    tgt = np.array(target_img)
    h = min(ref.shape[0], tgt.shape[0])
    ref_resized = PILImage.fromarray(ref).resize((int(ref.shape[1] * h / ref.shape[0]), h))
    tgt_resized = PILImage.fromarray(tgt).resize((int(tgt.shape[1] * h / tgt.shape[0]), h))
    canvas = PILImage.new("RGB", (ref_resized.width + tgt_resized.width, h))
    canvas.paste(ref_resized, (0, 0))
    canvas.paste(tgt_resized, (ref_resized.width, 0))
    return canvas

def prepare_llava_inputs(ref_image, target_image, prompt: str, processor, device) -> Dict:
    """
    Same semantics as your _prepare_llava_inputs.
    """
    combined = _concat_side_by_side(ref_image, target_image)
    inputs = processor(
        text=prompt,
        images=combined,
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}

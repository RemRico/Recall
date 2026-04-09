# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import random
from typing import Dict

_THIS_DIR = os.path.dirname(__file__)
_TMPL = os.path.join(_THIS_DIR, "templates.txt")

def _load_lines():
    with open(_TMPL, "r", encoding="utf-8") as f:
        raw = f.read()
    hard, div = [], []
    mode = None
    for ln in raw.splitlines():
        t = ln.strip()
        if t == "[[HARD_NEGATIVE]]":
            mode = "hard"
            continue
        if t == "[[DIVERSITY]]":
            mode = "div"
            continue
        if not t:
            continue
        if mode == "hard":
            hard.append(t)
        elif mode == "div":
            div.append(t)
    return hard, div

_HARD, _DIV = _load_lines()

def create_generic_prompt_enhanced(original_text: str, is_hard_negative_context: bool = False) -> str:
    pool = _HARD if is_hard_negative_context else _DIV
    tmpl = random.choice(pool) if pool else 'Original: "{text}".'
    return tmpl.replace("{text}", original_text)

def prepare_generic_inputs(ref_image, target_image, prompt: str, processor, device) -> Dict:
    """
    Same semantics as your _prepare_generic_inputs.
    """
    enhanced_prompt = f"{prompt} (Comparing reference and target images)"
    inputs = processor(
        text=enhanced_prompt,
        images=ref_image,  # generic: single primary image; prompt mentions comparison
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}

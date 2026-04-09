# -*- coding: utf-8 -*-
"""
Prompt builder for LLaVA backbone
- Builds concatenated multi-image input with structured text.
"""

from typing import Dict
from PIL import Image


def build_llava_inputs(processor, device, ref_image_path: str, tgt_image_path: str, original_text: str) -> Dict:
    """
    Build inputs for LLaVA model given two images and original text.
    Returns a dict suitable for fm.generate(**inputs).
    """

    prompt = f"""
You are given two images: left (reference) and right (target).
The original modification text is: "{original_text}"

Your task: produce a new modification description ("text_new") that
better captures the transformation from the left to the right image.
Be concise, natural, and descriptive.
    """.strip()

    ref_img = Image.open(ref_image_path).convert("RGB")
    tgt_img = Image.open(tgt_image_path).convert("RGB")

    # Resize to the same height.
    min_h = min(ref_img.height, tgt_img.height)
    new_ref_w = int(ref_img.width * min_h / ref_img.height)
    new_tgt_w = int(tgt_img.width * min_h / tgt_img.height)
    ref_resized = ref_img.resize((new_ref_w, min_h))
    tgt_resized = tgt_img.resize((new_tgt_w, min_h))

    from PIL import Image as PILImage
    canvas = PILImage.new("RGB", (new_ref_w + new_tgt_w, min_h))
    canvas.paste(ref_resized, (0, 0))
    canvas.paste(tgt_resized, (new_ref_w, 0))

    inputs = processor(prompt, canvas, return_tensors="pt").to(device)
    return inputs

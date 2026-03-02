# -*- coding: utf-8 -*-
"""
Alternative prompt builder that focuses on direct visual evidence and stricter formatting.
This module is self-contained so it can be swapped in without touching the original builder.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Dict, List

import torch
from PIL import Image

from src.utils import print_rank

_THIS_DIR = os.path.dirname(__file__)
_SYS_TXT_V2 = os.path.join(_THIS_DIR, "system_prompt_v2.txt")
_FS_TXT_V2 = os.path.join(_THIS_DIR, "fewshot_examples_v2.txt")
_STAGES_TEMPLATE_V2_PATH = os.path.join(_THIS_DIR, "stages_template_v2.txt")

_TEXT_CACHE: Dict[str, str] = {}


def _read_text_retry(path: str, retries: int = 3, delay: float = 0.2) -> str:
    last_err = None
    for i in range(retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except (FileNotFoundError, OSError) as e:
            last_err = e
            time.sleep(delay * (i + 1))
    raise last_err  # type: ignore[misc]


def _read_text_cached(path: str) -> str:
    if path in _TEXT_CACHE and _TEXT_CACHE[path]:
        return _TEXT_CACHE[path]
    try:
        content = _read_text_retry(path)
        _TEXT_CACHE[path] = content
        return content
    except Exception as e:
        print_rank(f"[builder_v2] Warning: failed to read {path}: {e}")
        return _TEXT_CACHE.get(path, "")


def _parse_fewshot(raw: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not raw.strip():
        return items

    lines = raw.splitlines()
    mode = None
    buf_user: List[str] = []
    buf_assist: List[str] = []

    def _flush():
        if buf_user or buf_assist:
            items.append(
                {
                    "user": "\n".join(buf_user).strip(),
                    "assistant": "\n".join(buf_assist).strip(),
                }
            )
        buf_user.clear()
        buf_assist.clear()

    for ln in lines:
        token = ln.strip()
        if token == "<<<USER":
            if mode is not None:
                _flush()
            mode = "user"
            continue
        if token == ">>>ASSISTANT":
            mode = "assistant"
            continue
        if mode == "user":
            buf_user.append(ln)
        elif mode == "assistant":
            buf_assist.append(ln)
    _flush()
    return items


def _load_stage_template() -> str:
    return _read_text_cached(_STAGES_TEMPLATE_V2_PATH)


def create_prompt_instructions(original_text: str, is_hard_negative_context: bool = False) -> str:
    """
    Optional diversity in instruction; can be expanded later.
    """
    if is_hard_negative_context:
        return (
            "You are repairing a mistaken retrieval result.\n"
            f'Original edit text: "{original_text}".\n'
            "Describe the change so that the reference image would correctly match the hard negative target.\n"
            "Use vocabulary different from the original edit."
        )
    choices = [
        (
            "Rewrite the modification so it matches the TARGET image while staying close to the original intent.\n"
            f'Input edit text: "{original_text}".\n'
            "Use concise wording and only include facts visible in the TARGET image."
        ),
        (
            "Produce an alternative phrasing of the edit text that is guaranteed true for the TARGET image.\n"
            f'Current text: "{original_text}".\n'
            "Keep the meaning but adjust wording to reflect the exact visual evidence."
        ),
        (
            "Craft a corrected edit instruction based on the TARGET image.\n"
            f'Original: "{original_text}".\n'
            "Mention only details that the TARGET image confirms."
        ),
    ]
    return random.choice(choices)


def prepare_inputs_v2(
    ref_image: Image.Image,
    target_image: Image.Image,
    prompt: str,
    processor,
    device: str,
    *,
    use_fewshot: bool = True,
):
    system_prompt = _read_text_cached(_SYS_TXT_V2)
    fewshot_raw = _read_text_cached(_FS_TXT_V2) if use_fewshot else ""
    fewshots = _parse_fewshot(fewshot_raw)
    stage_template = _load_stage_template()

    conversation: List[Dict] = [{"role": "system", "content": system_prompt}]
    for ex in fewshots:
        conversation.append({"role": "user", "content": ex["user"]})
        conversation.append({"role": "assistant", "content": ex["assistant"]})

    stage_body = stage_template.format(modification_text=prompt)
    conversation.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ref_image},
                {"type": "image", "image": target_image},
                {"type": "text", "text": stage_body},
            ],
        }
    )

    chat_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        add_vision_id=True,
    )

    inputs = processor(
        text=[chat_prompt],
        images=[ref_image, target_image],
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def decode_raw_output(generated_text: str) -> str | None:
    if not generated_text:
        return None
    # Try structured JSON blocks first
    try:
        parsed = json.loads(generated_text)
        if isinstance(parsed, dict) and "text_new" in parsed:
            return parsed["text_new"]
    except Exception:
        pass
    try:
        match = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict) and "text_new" in parsed:
                return parsed["text_new"]
    except Exception:
        pass
    match = re.search(r'text_new\s*=\s*"([^"]*)"', generated_text)
    if match:
        return match.group(1)
    match = re.search(r'"text_new"\s*:\s*"([^"]*)"', generated_text)
    if match:
        return match.group(1)
    return None


@torch.no_grad()
def generate_with_qwen_v2(inputs, device: str, foundation_model):
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        output_ids = foundation_model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=foundation_model.config.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]

        processor = getattr(foundation_model, "processor", None)
        if processor is None:
            raise RuntimeError("foundation_model missing processor attribute")

        try:
            decoded = processor.decode(generated[0], skip_special_tokens=True).strip()
        except Exception:
            try:
                decoded = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            except Exception:
                tokenizer = getattr(processor, "tokenizer", None)
                decoded = tokenizer.decode(generated[0], skip_special_tokens=True).strip() if tokenizer else ""

        return decoded
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = [
    "create_prompt_instructions",
    "prepare_inputs_v2",
    "generate_with_qwen_v2",
    "decode_raw_output",
]

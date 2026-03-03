# -*- coding: utf-8 -*-
"""Minimal rewrite prompt builder for Qwen models."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Tuple

import torch
from PIL import Image

from src.utils import print_rank

_THIS_DIR = os.path.dirname(__file__)
_SYS_TXT = os.path.join(_THIS_DIR, "system_prompt_minimal.txt")
_TEMPLATE_TXT = os.path.join(_THIS_DIR, "minimal_template.txt")

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
        print_rank(f"[minimal builder] warn: failed to read {path}: {e}")
        return _TEXT_CACHE.get(path, "")


def prepare_inputs_minimal(
    ref_image: Image.Image,
    target_image: Image.Image,
    original_text: str,
    processor,
    device: str,
    dataset_name: str = None,
):
    # Select prompt files based on dataset
    if dataset_name == "FashionIQ":
        sys_txt = os.path.join(_THIS_DIR, "system_prompt_minimal_fashioniq.txt")
        template_txt = os.path.join(_THIS_DIR, "minimal_template_fashioniq.txt")
    else:
        sys_txt = _SYS_TXT
        template_txt = _TEMPLATE_TXT

    system_prompt = _read_text_cached(sys_txt)
    template = _read_text_cached(template_txt)
    payload = template.replace("{original_text}", original_text)

    conversation: List[Dict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ref_image},
                {"type": "image", "image": target_image},
                {"type": "text", "text": payload},
            ],
        },
    ]

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
    return {k: v.to(device) for k, v in inputs.items()}, system_prompt, payload, chat_prompt


@torch.no_grad()
def generate_with_qwen_minimal(inputs, device: str, foundation_model) -> str:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        output_ids = foundation_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
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


def _extract_json_block(raw_text: str) -> str | None:
    if not raw_text:
        return None
    # Try direct JSON
    raw_text = raw_text.strip()
    if raw_text.startswith("{") and raw_text.endswith("}"):
        return raw_text
    # Try code block
    match = re.search(r"```json\s*(\{.*\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback: brace extraction
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def parse_minimal_output(raw_text: str) -> Dict | None:
    json_str = _extract_json_block(raw_text)
    if not json_str:
        return None
    # Fix common formatting issues (missing commas between list items)
    json_str = re.sub(r'"\s*"-', '", "-', json_str)
    json_str = re.sub(r'"\s+"-', '", "-', json_str)
    json_str = re.sub(r'"\s*\"-', '", "-', json_str)
    json_str = re.sub(r'"\s*\n\s*"-', '",\n      "-', json_str)
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return None
        return data
    except json.JSONDecodeError:
        return None


def apply_rewrites(original_text: str, segments: List[Dict[str, str]]) -> str:
    """Apply rewrite segments to original text sequentially."""
    result = original_text
    for seg in segments:
        orig = seg.get("original_span", "")
        new = seg.get("new_span", "")
        if not orig:
            continue
        if orig not in result:
            raise ValueError(f"Original span '{orig}' not found in text")
        result = result.replace(orig, new, 1)
    return result


def validate_minimal_output(original_text: str, parsed: Dict) -> Tuple[bool, str, str]:
    if not isinstance(parsed, dict):
        return False, "Output is not a dict", ""

    required_keys = {"visual_summary", "rewrite_segments", "final_text"}
    missing = required_keys - parsed.keys()
    if missing:
        return False, f"Missing keys: {missing}", ""

    segments = parsed.get("rewrite_segments", [])
    if not isinstance(segments, list):
        return False, "rewrite_segments must be a list", ""

    for seg in segments:
        if not isinstance(seg, dict):
            return False, "segment is not a dict", ""
        if "original_span" not in seg or "new_span" not in seg:
            return False, "segment missing required fields", ""
        orig = seg["original_span"]
        if orig and orig not in original_text:
            return False, f"Span '{orig}' not in original text", ""

    try:
        reconstructed = apply_rewrites(original_text, segments)
    except ValueError as e:
        return False, str(e), ""

    final_text = parsed.get("final_text", "")
    if final_text != reconstructed:
        return False, "final_text mismatch reconstructed text", reconstructed

    # Check for contradictory contrast phrases
    if "instead of" in reconstructed.lower():
        lower = reconstructed.lower()
        parts = lower.split("instead of")
        if len(parts) >= 2:
            left_tokens = [
                t for t in re.findall(r"[a-zA-Z]+", parts[0])
                if t not in {"a", "the", "on", "in", "of", "and", "with", "is", "are", "was", "were", "to", "into", "at", "by", "for"}
            ]
            right_tokens = [
                t for t in re.findall(r"[a-zA-Z]+", parts[1])
                if t not in {"a", "the", "on", "in", "of", "and", "with", "is", "are", "was", "were", "to", "into", "at", "by", "for"}
            ]
            if left_tokens and right_tokens:
                if set(left_tokens) & set(right_tokens):
                    return False, "contradictory contrast phrase after rewrite", reconstructed

    return True, "OK", reconstructed


def clean_contrast_clause(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"\s+(instead of\b.*)$", "", text, flags=re.IGNORECASE).strip()
    if cleaned and text.endswith(".") and not cleaned.endswith("."):
        cleaned += "."
    return cleaned or text


def run_minimal_pipeline(
    ref_image: Image.Image,
    target_image: Image.Image,
    original_text: str,
    processor,
    device: str,
    foundation_model,
    dataset_name: str = None,
) -> Dict:
    inputs, system_prompt, payload, chat_prompt = prepare_inputs_minimal(
        ref_image, target_image, original_text, processor, device, dataset_name
    )
    raw_text = generate_with_qwen_minimal(inputs, device, foundation_model)

    result: Dict = {
        "raw_model_output": raw_text,
        "final_text": "",
        "parsed_output": None,
        "validation_error": None,
        "system_prompt": system_prompt,
        "template_payload": payload,
        "chat_prompt": chat_prompt,
    }

    parsed = parse_minimal_output(raw_text)
    if parsed is None:
        result["validation_error"] = "parse_error"
        return result

    result["parsed_output"] = parsed
    ok, reason, corrected = validate_minimal_output(original_text, parsed)

    if ok:
        final_text = parsed.get("final_text", "")
    else:
        result["validation_error"] = reason
        if reason == "final_text mismatch reconstructed text" and corrected:
            result["final_text_corrected"] = corrected
            final_text = corrected
        elif corrected:
            final_text = corrected
        else:
            final_text = ""

    if final_text:
        cleaned = clean_contrast_clause(final_text)
        if cleaned != final_text:
            result["final_text_contrast_cleaned"] = cleaned
            final_text = cleaned

    result["final_text"] = final_text
    return result


__all__ = [
    "prepare_inputs_minimal",
    "generate_with_qwen_minimal",
    "parse_minimal_output",
    "validate_minimal_output",
]

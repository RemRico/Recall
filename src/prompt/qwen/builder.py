# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
import random
import re
import time
from typing import Dict, List

import torch

from src.utils import print_rank

_THIS_DIR = os.path.dirname(__file__)
_SYS_TXT = os.path.join(_THIS_DIR, "system_prompt.txt")
_FS_TXT = os.path.join(_THIS_DIR, "fewshot_examples.txt")

# Cache prompt texts to reduce repeated filesystem reads under DDP.
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
    raise last_err

def _read_text_cached(path: str) -> str:
    if path in _TEXT_CACHE and _TEXT_CACHE[path]:
        return _TEXT_CACHE[path]
    try:
        content = _read_text_retry(path)
        _TEXT_CACHE[path] = content
        return content
    except Exception as e:
        print_rank(f"Warning: failed to read {path}: {e}. Using cached/empty content.")
        return _TEXT_CACHE.get(path, "")

# Warm up cache once; non-fatal if files are temporarily unavailable.
try:
    _TEXT_CACHE[_SYS_TXT] = _read_text_retry(_SYS_TXT)
    _TEXT_CACHE[_FS_TXT] = _read_text_retry(_FS_TXT)
except Exception as e:
    print_rank(f"Note: prompt warmup skipped: {e}")

# --------------------------
# Prompt creators (kept API)
# --------------------------

def create_qwen_prompt(original_text: str, is_hard_negative_context: bool = False) -> str:
    """
    Same semantics as your original _create_qwen_prompt.
    """
    if is_hard_negative_context:
        return (
            'You are analyzing a retrieval error case. Here\'s the situation:\n\n'
            f'RETRIEVAL ERROR ANALYSIS:\n'
            f'- Reference image (first image) + Original query: "{original_text}"\n'
            f'- This combination INCORRECTLY retrieved the target image (second image)\n'
            f'- The target image should NOT have been retrieved with the original query\n\n'
            'TASK: Generate a NEW modification text that would make this retrieval CORRECT.\n'
            '- Create a description that, when combined with the reference image, should CORRECTLY retrieve the target image\n'
            '- Focus on the visual differences that make the target image the RIGHT match\n'
            '- Use completely different vocabulary and approach than the original query\n\n'
            'New correct modification text:'
        )
    else:
        diversity = [
            (
                "You are an expert at describing visual changes. Looking at these two images, "
                f'I need a modification instruction.\n\nOriginal instruction: "{original_text}"\n\n'
                "Create a NEW instruction that describes how to transform the reference image to match the target image. "
                "Make it different from the original but achieve the same visual transformation. "
                "Use varied vocabulary and phrasing.\n\nNew instruction:"
            ),
            (
                "Based on comparing the reference and target images, write a fresh description of the visual change needed.\n\n"
                f'Given instruction: "{original_text}"\n\n'
                "Generate an alternative instruction that would lead to the same visual outcome but uses different words and structure. "
                "Focus on different aspects or details.\n\nAlternative instruction:"
            ),
            (
                "I have two images and need a modification description. "
                f'The original was: "{original_text}"\n\n'
                "Looking at both images, write a NEW way to describe this same transformation. "
                "Use different terminology, focus on different visual elements, or approach from a different angle.\n\n"
                "New description:"
            ),
        ]
        return random.choice(diversity)

# --------------------------
# Few-shot loader / parser
# --------------------------

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _parse_fewshot(raw: str) -> List[Dict[str, str]]:
    """
    Parse blocks:
        <<<USER
        ...text...
        >>>ASSISTANT
        ...text...
    Returns list of {"user": "...", "assistant": "..."} in order.
    """
    items, cur = [], None
    lines = raw.splitlines()
    mode = None
    buf_user, buf_assist = [], []

    def _flush():
        nonlocal buf_user, buf_assist
        if buf_user or buf_assist:
            items.append({"user": "\n".join(buf_user).strip(), "assistant": "\n".join(buf_assist).strip()})
        buf_user, buf_assist = [], []

    for ln in lines:
        if ln.strip() == "<<<USER":
            if mode is not None:
                _flush()
            mode = "user"
            continue
        if ln.strip() == ">>>ASSISTANT":
            mode = "assistant"
            continue
        if mode == "user":
            buf_user.append(ln)
        elif mode == "assistant":
            buf_assist.append(ln)
        else:
            # ignore stray lines
            pass
    _flush()
    return items

# --------------------------------
# Qwen input builder (kept API)
# --------------------------------

_STAGES_TEMPLATE = """
<Input>
    "REFERENCE": Picture 1,
    "TARGET": Picture 2,
    "Input edit text": {modification_text}

=====================
STAGES (run in order)
=====================
##Generate Captions:
- You must not refer to the input edit text in this stage.
- You must mention their unique features in their own captitons.
- Do not reuse sentences across Reference and Target. If a sentence would be identical, rewrite it to include a detail unique to that image, or write "uncertain".
- Tag mapping: "Reference" ALWAYS refers to Picture 1; "Target" ALWAYS refers to Picture 2.
- Reference: [REFERENCE] only describe the reference image(REFERENCE, Picture 1) in 3-5 literal sentences. Be thorough and accurate; list visible objects, attributes (color/material), counts, spatial relations, and readable text. Do not mention TARGET here. If unreadable, say "uncertain".
- Target: [TARGET] only describe the target image(TARGET, Picture 2) in 3-5 literal sentences. Same constraints; do not mention REFERENCE here.

##Decompose Intents(only use edit text:{modification_text}):
- One line per atomic intent, covering all content-bearing phrases in the input edit text (split by commas/"and/then"/parentheses).
- you must only decompose the input edit text, do not refer to any information from the REFERENCE or TARGET.
- Format (one per line):
    I# | span="..." | intent="..." | type=absolute|relative | objects=[... ] | note="one-line reason"
- Hints:
    - Treat parenthetical modifiers as their own intents if they encode states/gestures/poses (e.g., "(fist closed)").
    - Count words exactly ("a/one/two/…") and keep them in span.
    - Max 6 intents; if more, merge logically but keep atomicity for verifiable units.

##Generate Polar Questions:
- Create exactly one yes/no question per intent, same order and same ID.
- Format (one per line):
    I# | scope=target_only|compare_ref_target | Q="...?"
- Rules:
    - If type=absolute → scope=target_only and start with In the TARGET image, ...
    - If type=relative → scope=compare_ref_target and start with Comparing REFERENCE and TARGET, ...
    - Use only allowed auxiliaries after the prefix; no wh-words.

##Answer Questions via Image Comparison:
- Answer each question (same IDs and order) with Yes/No/Uncertain and brief evidence based on the TARGET image(Picture 2).
- Carefully consider the visibility of the object in the TARGET image(Picture 2),and only when the evidence is clear, you can answer "Yes".
- Format (one per line):
    I# | A=Yes|No|Uncertain | evidence_target="..." | evidence_ref="..."
- every question should be answered, and the answer should be based on the TARGET image only.
- Notes:
    - If scope=target_only, fill evidence_ref="".
    - If scope=compare_ref_target, you can refer to the REFERENCE image for relative information, but the answer should be based on the TARGET image only.
    - If visibility is insufficient, choose Uncertain.

##Local Edits:
- Map each intent to a local edit: Yes → keep, No → rewrite to match TARGET.
- Format (one per line):
    I# | action=keep|rewrite|generalize|delete | before="(original span or clause)" | after="(new fragment)" | reason="(short why)"

##Generate Text_new:
- Output a single fluent sentence (or two short clauses) by merging all local edits with action=keep or rewrite (skip delete; generalize if you marked generalize).
- Format (one per line):
    text_new="..."
- Rules:
    - Please keep all intents that you answered "Yes" in the local edits.
    - Use the TARGET image as the evidence source.
    - do not contain 'REFERENCE' or 'TARGET' in the text_new.
    - if rewrite, please rewrite it with target image's that feature.(e.g. "four dogs" but in target image only two dogs -> "two dogs")
    - try to keep the original format of the input edit text.(e.g. "a bird of a yellow color" -> "a bird of blue color")
""".strip()

def prepare_qwen_inputs(ref_image, target_image, prompt: str, processor, device):
    """
    Same semantics as your _prepare_qwen_inputs(ref_image, target_image, prompt, processor, device).
    - ref_image / target_image: PIL.Image.Image
    - prompt: string (usually original_text or create_qwen_prompt(...))
    """
    system_prompt = _read_text_cached(_SYS_TXT)
    fewshot_raw   = _read_text_cached(_FS_TXT)
    fewshots = _parse_fewshot(fewshot_raw)

    input_text = _STAGES_TEMPLATE.format(modification_text=prompt)
    
    conversation: List[Dict] = [{"role": "system", "content": system_prompt}]
    for ex in fewshots:
        conversation.append({"role": "user", "content": ex["user"]})
        conversation.append({"role": "assistant", "content": ex["assistant"]})

    conversation.append({
        "role": "user",
        "content": [
            {"type": "image", "image": ref_image},
            {"type": "image", "image": target_image},
            {"type": "text", "text": input_text},
        ],
    })

    text_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        add_vision_id=True
    )

    inputs = processor(
        text=[text_prompt],
        images=[ref_image, target_image],
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}


# Alias retained for `CaptionBatcher` compatibility.
prepare_inputs = prepare_qwen_inputs

def _output_translator(generated_text: str):
    """Extract text_new from model output"""
    if not generated_text:
        return None
    # 1) Raw JSON.
    try:
        parsed_json = json.loads(generated_text.strip())
        if isinstance(parsed_json, dict) and "text_new" in parsed_json:
            return parsed_json["text_new"]
    except json.JSONDecodeError:
        pass
    # 2) JSON code block.
    try:
        m = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
        if m:
            parsed_json = json.loads(m.group(1))
            if isinstance(parsed_json, dict) and "text_new" in parsed_json:
                return parsed_json["text_new"]
    except json.JSONDecodeError:
        pass
    # 3) Inline JSON object snippet.
    m = re.search(r'\{[^{}]*"text_new"\s*:\s*"([^"]*)"[^{}]*\}', generated_text, re.DOTALL)
    if m:
        return m.group(1)
    # 4) Direct `"text_new": "..."` match.
    m = re.search(r'"text_new"\s*:\s*"([^"]*)"', generated_text)
    if m:
        return m.group(1)
    # 5) `text_new="..."` match.
    m = re.search(r'text_new\s*[=:]\s*"([^"]*)"', generated_text)
    if m:
        return m.group(1)
    # 6) `Text_new` variant.
    m = re.search(r'"Text_new"\s*[:=]\s*"([^"]*)"', generated_text)
    if m:
        return m.group(1)
    return ' '

@torch.no_grad()
def generate_with_qwen(inputs, device, foundation_model):
    """Generate text with Qwen2-VL with memory-aware settings."""
    try:
        torch.cuda.empty_cache()
        output_ids = foundation_model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=foundation_model.config.eos_token_id
        )
        # Decode only newly generated tokens.
        input_len = inputs['input_ids'].shape[1]
        generated = output_ids[:, input_len:]
        proc = getattr(foundation_model, "processor", None)

        # Prefer processor.decode, with batch_decode/tokenizer fallback.
        try:
            text = proc.decode(generated[0], skip_special_tokens=True).strip()
        except Exception:
            try:
                text = proc.batch_decode(generated, skip_special_tokens=True)[0].strip()
            except Exception:
                tok = getattr(proc, "tokenizer", None)
                text = tok.decode(generated[0], skip_special_tokens=True).strip() if tok else ""

        del output_ids, generated
        torch.cuda.empty_cache()
        return _output_translator(text)

    except torch.cuda.OutOfMemoryError as e:
        print_rank(f"CUDA OOM in caption generation: {e}")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print_rank(f"Error in Qwen generation: {e}")
        torch.cuda.empty_cache()
        return None
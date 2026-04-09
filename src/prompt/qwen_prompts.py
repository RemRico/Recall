# -*- coding: utf-8 -*-
"""
Prompt builder for Qwen backbone
- Builds conversation format with rich instruction chains.
"""

from typing import Dict, List


def build_qwen_conversation(ref_image_path: str, tgt_image_path: str, original_text: str) -> List[Dict]:
    """
    Build a Qwen-style chat conversation for image modification.
    Returns a list of role/content dicts, suitable for processor.apply_chat_template.
    """

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ref_image_path},
                {"type": "image", "image": tgt_image_path},
                {
                    "type": "text",
                    "text": f"""
You are given two images (left: reference, right: target).
Original modification description: "{original_text}"

Perform the following tasks step by step:

1. **Generate Captions**: Write clear captions for both images.
2. **Compare Captions**: Highlight visual differences.
3. **Generate Polar Questions**: Ask yes/no style comparison questions.
4. **Answer Questions via Image Comparison**: Answer them based on the two images.
5. **Local Edits**: Describe fine-grained changes from reference to target.
6. **Decompose Intents**: Summarize the main modification intents.
7. **Generate Text_new**: Finally, write a single natural sentence ("text_new") that can be used as a modification query for retrieval.

Output must be valid JSON with a field "text_new".
                        """.strip(),
                },
            ],
        }
    ]

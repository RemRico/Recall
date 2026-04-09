from itertools import repeat
from typing import Optional

import logging
from dataclasses import dataclass
from transformers import ProcessorMixin
from src.arguments import DataArguments, ModelArguments, TrainingArguments
import torch
from qwen_vl_utils import smart_resize

from src.model.processor import (
    LLAVA_NEXT,
    QWEN2_VL,
    QWEN2_5_VL,
    QWEN2_VL_TOKENSELECTION,
    QWEN2_5_VL_TOKENSELECTION,
    PHI3V,
    GME,
    LamRA,
    LamRA_QWEN2_5,
    GME_CIRR_QUERY_INSTRUCTION,
    process_vlm_inputs_fns,
)
from PIL import Image
import io
import hashlib
import builtins
from src.utils import print_rank


logger = logging.getLogger(__name__)


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000


def split_and_process_vlm_inputs(model_input: dict, chunk_size: int):
    """Split aligned multimodal model inputs into chunks along batch dimension."""
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = []
    for k in keys:
        if isinstance(arg_val[k], torch.Tensor):
            chunked_tensor = arg_val[k].split(chunk_size, dim=0)
        else:
            chunked_tensor = [arg_val[k][i: i + chunk_size] for i in list(range(0, len(arg_val[k]), chunk_size))]
        chunked_tensors.append(chunked_tensor)
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]
    chunked_inputs = [{arg_key: c} for c in chunked_arg_val]

    return chunked_inputs


def split_vlm_inputs(model_input: dict, chunk_size: int):
    """Split multimodal inputs while preserving text-image alignment in each chunk."""
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]
    keys = list(arg_val.keys())

    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in ["input_ids", "attention_mask"]]

    input_ids = arg_val["input_ids"]
    positions = torch.nonzero((input_ids < 0) & (input_ids > -PHI_IMAGE_TOKEN_MAX_INPUT_ID), as_tuple=True)
    row_contain_image = torch.unique(positions[0])  # indicates which row in input_ids contain images
    num_chunks = len(chunked_tensors[0])
    chunk_image_count = []
    for chunk_idx in range(num_chunks):
        chunk_image_count.append(torch.sum(
            (row_contain_image >= chunk_idx * chunk_size) & (row_contain_image < (chunk_idx + 1) * chunk_size)).item())
    if "pixel_values" in keys:
        pixel_values = arg_val["pixel_values"]
        image_sizes = arg_val["image_sizes"]
        chunked_tensors.append(torch.split(pixel_values, chunk_image_count))
        chunked_tensors.append(torch.split(image_sizes, chunk_image_count))

    chunked_arg_val = []
    for kk, tt in zip(repeat(keys), zip(*chunked_tensors)):
        if "pixel_values" in keys and tt[2].numel() == 0:
            chunked_arg_val.append(dict(zip(kk[:2], tt[:2])))
        else:
            chunked_arg_val.append(dict(zip(kk, tt)))

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    """
    Get either qry_reps or tgt_reps.
    """
    if x["qry_reps"] is None:
        return x["tgt_reps"]
    else:
        return x["qry_reps"]


@dataclass
class TrainTextImageDataCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """Build query/positive text-image inputs for legacy train flows."""
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        return qry_inputs, pos_inputs

    def _get_batch_inputs(self, examples, text_keyname, image_keyname):
        texts, images = [], []
        for example in examples:
            if example is None or not example:
                text, image = '  ', None
            text, image = example[text_keyname], example[image_keyname]
            if type(text) == list:
                if len(text) == 0 or len(image) == 0:
                    text, image = '  ', None
                else:
                    text, image = text[0], image[0]
            texts.append(text)
            images.append(image)
        inputs = {'text': texts, 'image': images}
        return inputs


@dataclass
class MultimodalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    batch_size: Optional[int] = None

    def __post_init__(self):
        self.long_text_count = 0
        self.total_text_count = 0

    def _get_batch_inputs(self, batch, text_keyname, image_keyname, input_type: str = "query"):
        texts, visual_inputs = [], []
        
        for batch_idx, example in enumerate(batch):
            if example is None or not example:
                text, visual_input = '  ', None
            else:
                text, raw_images = example[text_keyname], example[image_keyname]

                if text and len(text) > 1000:
                    self.long_text_count += 1
                    original_len = len(text)
                    text = text[:1000]
                    if self.long_text_count <= 10:
                        print_rank(f"Long text truncated: {original_len} -> {len(text)} chars")

                self.total_text_count += 1

                if type(raw_images) == dict:
                    visual_input = []
                    assert 'resolutions' in raw_images, "we need len(raw_images['resolutions']) to determine the number of images, set it a list of None of for cases that no resizing is needed"
                    num_images = len(raw_images['resolutions'])
                    
                    for image_idx in range(num_images):
                        bytes_data = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                        path = raw_images['paths'][image_idx] if 'paths' in raw_images else None
                        image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                        
                        image = self._load_image(bytes_data, path)
                        
                        if not self.data_args.resize_use_processor and image is not None and image_resolution:
                            image = image.resize(image_resolution)
                            
                        if image is not None and (self.data_args.image_decay_factor is not None and image_resolution is None):
                            assert image_resolution is None, "image_resolution is conflicting with image_decay_factor"
                            assert self.model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION], "image_decay_factor is only supported for Qwen models"
                            max_pixels = max(self.data_args.resize_min_pixels, self.data_args.resize_max_pixels * self.data_args.image_decay_factor ** (num_images - image_idx))
                            width, height = image.size
                            resized_height, resized_width = smart_resize(
                                height, width,
                                min_pixels=self.data_args.resize_min_pixels,
                                max_pixels=max_pixels,
                            )
                            image = image.resize((resized_width, resized_height))

                        visual_input.append(image)
                else:
                            visual_input = None
            texts.append(text)
            visual_inputs.append(visual_input)
    
        inputs = {'text': texts, 'images': visual_inputs}
        if self.training_args.model_backbone in [GME, LamRA, LamRA_QWEN2_5]:
            inputs['is_query'] = (input_type == "query")
            if input_type == "query":
                inputs['instruction'] = GME_CIRR_QUERY_INSTRUCTION
            else:
                inputs['instruction'] = None
        return inputs

    def _load_image(self, bytes_data, path):
        """Load an image from bytes or local path and return RGB PIL image."""
        try:
            if bytes_data is None and path is None:
                return None
            elif bytes_data is not None:
                image = Image.open(io.BytesIO(bytes_data))
                return image.convert("RGB")
            elif path is not None:
                with Image.open(path) as img:
                    return img.convert("RGB")
        except Exception as e:
            logger.warning("Failed to load image: %s", e)
            return None

    def __call__(self, examples):
        """Collate training samples into processed query/target model inputs."""
        try:
            qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image", input_type="query")
            pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image", input_type="target")
            bs = len(qry_inputs['text'])
            assert bs > 0, 'An empty batch'
            
            if self.batch_size is not None and bs < self.batch_size:
                raise RuntimeError(f"Expect batch size {self.batch_size}, but got batch size of {bs}")
                
            process_fn = process_vlm_inputs_fns[self.training_args.model_backbone]
            processed_qry_inputs = process_fn(qry_inputs, processor=self.processor, max_length=self.data_args.max_len)
            processed_pos_inputs = process_fn(pos_inputs, processor=self.processor, max_length=self.data_args.max_len)
            
            processed_qry_inputs['text'] = [e['query_text'] for e in examples]
            processed_pos_inputs['text'] = [e['pos_text'] for e in examples]
            processed_qry_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
            processed_pos_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]

            if self.total_text_count % 1000 == 0:
                logger.info(
                    "Text stats - Total: %d, Truncated: %d (%.1f%%)",
                    self.total_text_count,
                    self.long_text_count,
                    self.long_text_count / self.total_text_count * 100,
                )

            sid_list = []
            for e in examples:
                sid = e.get("sample_ids", None)
                if isinstance(sid, torch.Tensor):
                    sid_list.append(sid.cpu())
                elif sid is None:
                    sid_list.append(torch.tensor(-1))
                else:
                    sid_list.append(torch.tensor(int(sid)))
            processed_qry_inputs["sample_ids"] = torch.stack(sid_list)
            
            ref_ids = []
            aug_flags = []
            anchor_ids = []
            for e in examples:
                ref_ids.append(int(e.get("reference_id", -1)))
                aug_flags.append(bool(e.get("is_augmented", False)))
                anchor_ids.append(self._anchor_group_id(e))
            processed_qry_inputs["reference_ids"] = torch.as_tensor(ref_ids, dtype=torch.long)
            processed_qry_inputs["is_augmented"] = torch.as_tensor(aug_flags, dtype=torch.bool)
            processed_qry_inputs["anchor_group_ids"] = torch.as_tensor(anchor_ids, dtype=torch.long)

            return processed_qry_inputs, processed_pos_inputs

        except Exception as e:
            logger.exception("Collator error: %s", e)
            logger.info("Text stats - Total: %d, Truncated: %d", self.total_text_count, self.long_text_count)
            raise

    def _anchor_group_id(self, example) -> int:
        """
        Stable hash for (reference_image, original_mod_text) grouping.
        Falls back to -1 when no anchor text is available.
        """
        if not builtins.isinstance(example, dict):
            return -1
        anchor = example.get("augmentation_group_key")
        if not anchor:
            anchor = example.get("original_mod_text")
        if anchor is None:
            return -1
        if not builtins.isinstance(anchor, str):
            anchor = str(anchor)
        anchor = anchor.strip()
        if not anchor:
            return -1
        ref = example.get("reference_image", "")
        if not builtins.isinstance(ref, str):
            ref = str(ref) if ref is not None else ""
        composite = f"{ref}||{anchor}"
        digest = hashlib.sha1(composite.encode("utf-8")).hexdigest()
        anchor_id = int(digest[-16:], 16)
        anchor_id &= (1 << 63) - 1
        return anchor_id

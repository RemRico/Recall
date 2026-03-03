from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "huggingface model name or path"})
    model_type: str = field(default=None, metadata={"help": "model type, typically includes in config file, but sometimes needs mannually add"})
    processor_name: str = field(default=None, metadata={"help": "processor_name, huggingface model name or path"})
    model_backbone: str = field(default=None, metadata={"help": "HF model type"})
    checkpoint_path: str = field(default=None, metadata={"help": "a local model path, could be a LoRA version"})
    foundation_model_name: str = field(default=None, metadata={"help": "foundation model name for iterative training caption generation"})
    foundation_prompt_mode: str = field(default="minimal", metadata={"help": "prompt mode for foundation caption generation (cot|minimal)"})
    pooling: str = field(default='last', metadata={"help": "pooling method for encoder"})
    normalize: bool = field(default=False, metadata={"help": "normalize query and passage representations"})
    temperature: float = field(default=0.02, metadata={"help": "temperature for softmax"})
    lora: bool = field(default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"})
    lora_r: int = field(default=16, metadata={"help": "lora r"})
    lora_alpha: int = field(default=64, metadata={"help": "lora alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "lora dropout"})
    lora_target_modules: str = field(default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj", metadata={"help": "lora target modules"})
    num_crops: int = field(default=16, metadata={"help": "number of crops used in image encoder"})
    uigraph_use: bool = field(default=False, metadata={"help": "Enable ui graph for token selection"})
    uigraph_diff: int = field(default=1, metadata={"help": "Pixel difference used for constructing ui graph for token selection"})
    uigraph_rand: bool = field(default=False, metadata={"help": "Enable random graph construction for token selection"})
    uimask_ratio: float = field(default=0.5, metadata={"help": "Specify the percentage of patch tokens to skip per component for token selection"})
    uimask_rand: bool = field(default=False, metadata={"help": "Enable random token selection instead of uniform selection"})
    lm_skip_layer: str = field(default='[1,28,0]', metadata={"help": "Specify the layers of the language model to skip for token selection"})
    vis_skip_layer: str = field(default='[1,32,0]', metadata={"help": "Specify the layers of the vision model to skip for token selection"})


@dataclass
class DataArguments:
    dataset_config: str = field(default=None, metadata={"help": "yaml file with dataset configuration"})
    data_basedir: str = field(default=None, metadata={"help": "Expect an absolute path to the base directory of all datasets. If set, it will be prepended to each dataset path"})
    dataset_name: str = field(default=None, metadata={"help": "huggingface dataset name"})
    subset_name: List[str] = field(default=None, metadata={"help": "Useful for datasets with subsets"})
    dataset_split: str = field(default='train', metadata={"help": "dataset split"})
    num_sample_per_subset: int = field(default=None, metadata={"help": "number of training samples per subset"})
    image_dir: str = field(default=None, metadata={"help": "Image directory path"})
    encode_output_path: str = field(default=None, metadata={"help": "encode output path"})
    max_len: int = field(default=None, metadata={"help": "The maximum total input sequence length after tokenization. Use with caution, since it may truncate text prompts due to large image lengths."},)
    embedding_type: str = field(default="", metadata={"help": "embedding type"})
    image_resolution: str = field(default=None, metadata={"help": "for models i.e. LLaVA-next and Qwen, resize images first, none means using original image resolution. This is only works when `--resize_use_processor false`."})
    resize_use_processor: bool = field(default=True, metadata={"help": "Resize visual inputs insides processor, e.g. Qwen2VLImageProcessor, instead of by our code."})
    resize_min_pixels: int = field(default=28*28*4, metadata={"help": "The min pixels of the image to resize the image. This is only works when `--resize_use_processor true`."})
    resize_max_pixels: int = field(default=28*28*1280, metadata={"help": "The max pixels of the image to resize the image. This is only works when `--resize_use_processor true`."})
    image_decay_factor: float = field(default=None, metadata={"help": "The image decay factor for resizing temporal images"})
    num_hardneg: int = field(default=0, metadata={"help": "hard negative number"})


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    group_by_reference_image: bool = field(
        default=False, metadata={"help": "Group samples by reference image into the same batch during iterative training."}
    )
    group_by_category: bool = field(
        default=False, metadata={"help": "Group FashionIQ samples by category (dress/shirt/toptee) for batch sampling."}
    )
    use_minimal_prompt: bool = field(
        default=True, metadata={"help": "Use minimal prompt mode for caption generation (automatically selects dataset-specific prompts)."}
    )
    use_original_data_in_iter_plus: bool = field(
        default=True, metadata={"help": "Whether to use original data in iterations > 0. If False, only augmented data will be used."}
    )
    sampler_debug: bool = field(
        default=True, metadata={"help": "Enable grouped sampler debug logging (summaries, previews)."}
    )
    sampler_debug_max_batches: int = field(
        default=5, metadata={"help": "Number of batches to preview in sampler debug logging."}
    )
    sampler_debug_preview_groups: int = field(
        default=3, metadata={"help": "Number of representative groups to preview when sampler debug is on."}
    )
    sampler_debug_preview_items: int = field(
        default=3, metadata={"help": "Number of samples to print per preview group or batch when debug is on."}
    )
    sampler_debug_preview_chars: int = field(
        default=80, metadata={"help": "Max characters to show per text field in sampler previews."}
    )
    sampler_debug_preview_small_max_size: int = field(
        default=4, metadata={"help": "Preferred max size of groups to display in sampler previews; set 0 to disable small-group preference."}
    )
    # evaluation_strategy: str = field(default='no', metadata={"help": "The evaluation strategy to adopt during training. Possible values are: 'no', 'steps', 'epoch'"})
    # per_device_train_batch_size: int = field(default=8, metadata={"help": "The batch size per GPU/TPU core/CPU for training."})
    image_encoder_freeze: bool = field(default=False, metadata={"help": "huggingface model name"})
    resume_from: str = field(default="none", metadata={"help": "`auto` will detect if any previous checkpoints should be resumed. or specify specific step of the checkpoint."})
    resume_from_iteration: str = field(default="none", metadata={"help": "Resume from a specific iteration model: `auto`, `iter_0`, `iter_1`, etc. This loads model weights only without training state."})
    project_name: str = field(default=None, metadata={"help": "project name"})
    logging_dir: str = field(default=None, metadata={"help": "TensorBoard log directory, also enables train.log generation"})
    logging_steps: int = field(default=1, metadata={"help": "logging steps"})
    num_train_epochs: int = field(default=1, metadata={"help": "number of training epochs"})
    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=2, metadata={"help": "query side subset size"})
    gc_p_chunk_size: int = field(default=2, metadata={"help": "target side subset size"})
    interleave_stopping_strategy: str = field(default="all_exhausted", metadata={"help": "all_exhausted or first_exhausted"})
    interleave_batch_size: float = field(default=0, metadata={"help": "Specify mini-batch size to interleave data from multi-sources, 0/None means random sampling by examples, 1 means full batch."})
    # Iterative training parameters
    max_iterations: int = field(default=3, metadata={"help": "Maximum number of iterative training rounds"})
    hard_neg_collection_freq: int = field(default=1, metadata={"help": "Frequency of hard negative collection (every N iterations)"})
    caption_generation_batch_size: int = field(default=8, metadata={"help": "Batch size for foundation model caption generation"})
    info_nce_weight: float = field(default=0.5, metadata={"help": "Weight for the InfoNCE contrastive loss"})
    triplet_loss_weight: float = field(default=0.5, metadata={"help": "Weight for the triplet margin loss"})
    triplet_margin: float = field(default=0.2, metadata={"help": "Margin used by the triplet loss"})

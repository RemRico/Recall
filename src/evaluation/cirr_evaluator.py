"""
CIRR Dataset Evaluator for Composed Image Retrieval
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any
from PIL import Image
from tqdm import tqdm

from ..model.processor import (
    GME,
    LamRA,
    LamRA_QWEN2_5,
    GME_CIRR_QUERY_INSTRUCTION,
)

try:
    from ..utils import print_rank, print_master
except ImportError:
    # Fallback for when running as script
    def print_rank(*args, **kwargs):
        print(*args, **kwargs)
    
    def print_master(*args, **kwargs):
        print(*args, **kwargs)


class CIRREvaluator:
    """
    Evaluator for CIRR dataset following the official evaluation protocol
    """
    
    def __init__(self, 
                 model,
                 processor, 
                 data_args,
                 model_args,
                 device='cuda',
                 batch_size=16,
                 eval_config_path=None):
        self.model = model
        self.processor = processor
        self.data_args = data_args
        self.model_args = model_args
        self.device = device
        self.batch_size = batch_size
        
        # Load evaluation configuration
        self.eval_config = self._load_eval_config(eval_config_path)
        
        # Override batch_size from config if provided
        if 'evaluation' in self.eval_config and 'batch_size' in self.eval_config['evaluation']:
            self.batch_size = self.eval_config['evaluation']['batch_size']
        
        # Get model backbone from model_args or config
        self.model_backbone = getattr(model_args, 'model_backbone', None)
        if self.model_backbone is None:
            self.model_backbone = self.eval_config.get('model', {}).get('default_backbone', 'qwen2_vl')
        
        # Configure CIRR data paths
        self._configure_data_paths()
        
        self.test_data, self.candidate_images = self._load_cirr_test_data()
        
        print_master(f"Loaded {len(self.test_data)} test queries")
        print_master(f"Loaded {len(self.candidate_images)} candidate images")
    
    def _load_eval_config(self, config_path=None):
        """Load evaluation configuration from YAML file"""
        import yaml
        
        if config_path is None:
            # Default config path
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(current_dir, 'configs', 'eval_config.yaml')
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print_master(f"Loaded evaluation config from: {config_path}")
                return config
            else:
                print_master(f"Config file not found: {config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            print_master(f"Error loading config file: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return default configuration when config file is not available"""
        return {
            'CIRR': {
                'data_dir': '/home/guohaiyun/yty_data/CIRR/cirr',
                'image_base_dir': '/home/guohaiyun/yty_data/CIRR',
                'validation': {
                    'queries_file': 'captions/cap.rc2.val.json',
                    'splits_file': 'image_splits/split.rc2.val.json'
                },
                'evaluation': {
                    'batch_size': 16,
                    'global_recall_k': [1, 5, 10, 50],
                    'group_recall_k': [1, 2, 3]
                }
            },
            'model': {
                'default_backbone': 'qwen2_vl',
                'max_length': 512,
                'instructions': {
                    'query_instruction': 'Represent the given image with the following modification',
                    'target_instruction': ''
                }
            }
        }
    
    def _configure_data_paths(self):
        """Configure CIRR dataset paths based on eval config and data_args"""
        cirr_config = self.eval_config.get('CIRR', {})
        
        # First try to load from data_args (for compatibility)
        if hasattr(self.data_args, 'dataset_config') and self.data_args.dataset_config:
            import yaml
            with open(self.data_args.dataset_config, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            if 'CIRR' in dataset_config:
                dataset_cirr_config = dataset_config['CIRR']
                self.data_dir = dataset_cirr_config.get('data_dir', cirr_config.get('data_dir'))
                self.image_base_dir = dataset_cirr_config.get('image_base_dir', cirr_config.get('image_base_dir'))
            else:
                # Use eval config
                self.data_dir = cirr_config.get('data_dir', '/home/guohaiyun/yty_data/CIRR/cirr')
                self.image_base_dir = cirr_config.get('image_base_dir', '/home/guohaiyun/yty_data/CIRR')
        else:
            # Use eval config
            self.data_dir = cirr_config.get('data_dir', '/home/guohaiyun/yty_data/CIRR/cirr')
            self.image_base_dir = cirr_config.get('image_base_dir', '/home/guohaiyun/yty_data/CIRR')
        
        # Set file paths using config
        validation_config = cirr_config.get('validation', {})
        queries_file = validation_config.get('queries_file', 'captions/cap.rc2.val.json')
        splits_file = validation_config.get('splits_file', 'image_splits/split.rc2.val.json')
        
        self.captions_file = os.path.join(self.data_dir, queries_file)
        self.image_splits_file = os.path.join(self.data_dir, splits_file)
        
        print_master(f"Using CIRR data directory: {self.data_dir}")
        print_master(f"Using CIRR image directory: {self.image_base_dir}")
    
    def _load_cirr_test_data(self) -> Tuple[List[Dict], List[str]]:
        """Load CIRR validation data (dev set) for testing
        
        Returns:
            Tuple[List[Dict], List[str]]: (test_queries, candidate_images)
        """
        try:
            # Use validation set as test set since test set has no GT labels
            val_queries_file = self.captions_file
            val_splits_file = self.image_splits_file
            
            if not os.path.exists(val_queries_file):
                print_master(f"Warning: CIRR validation queries file not found at {val_queries_file}")
                return self._create_dummy_test_data()
            
            # Load validation queries (complete query data including ref/target images, modification text, etc.)
            with open(val_queries_file, 'r') as f:
                val_queries = json.load(f)
            
            # Load validation image splits for candidate images
            if os.path.exists(val_splits_file):
                with open(val_splits_file, 'r') as f:
                    val_splits = json.load(f)
                # Extract candidate image names from splits
                candidate_images = list(val_splits.keys())
                self.image_splits = val_splits
                print_master(f"Loaded {len(candidate_images)} candidate images from validation splits")
            else:
                print_master(f"Warning: No validation splits file found at {val_splits_file}, using dummy candidates")
                candidate_images = [f"dummy_img_{i}" for i in range(100)]
                self.image_splits = {}
            
            print_master(f"Loaded {len(val_queries)} CIRR validation queries")
            
            return val_queries, candidate_images
            
        except Exception as e:
            print_master(f"Error loading CIRR validation data: {e}")
            return self._create_dummy_test_data()
    
    def _create_dummy_test_data(self) -> Tuple[List[Dict], List[str]]:
        """Create dummy test data when real CIRR data is not available
        
        Returns:
            Tuple[List[Dict], List[str]]: (dummy_queries, dummy_candidate_images)
        """
        dummy_data = []
        for i in range(50):  # Small dummy dataset
            dummy_data.append({
                'pairid': i,
                'reference': f'dummy_ref_{i}',
                'target_hard': f'dummy_target_{i}',
                'caption': f'Dummy modification text {i}',
                'target_soft': {},
                'img_set': {'members': [f'dummy_img_{j}' for j in range(i, i+5)]}
            })
        # Also create dummy candidate images when creating dummy test data
        candidate_images = [f"dummy_img_{i}" for i in range(100)]
        self.image_splits = {}
        return dummy_data, candidate_images
    
    def _load_image(self, image_name: str) -> Image.Image:
        """Load PIL image from name using CIRR image splits"""
        try:
            # First try to get path from image_splits mapping
            if hasattr(self, 'image_splits') and image_name in self.image_splits:
                relative_path = self.image_splits[image_name]
                # Convert relative path to absolute path
                # relative_path is like "./dev/dev-244-0-img0.png"
                if relative_path.startswith('./'):
                    relative_path = relative_path[2:]  # Remove "./"
                full_path = os.path.join(self.image_base_dir, relative_path)
                
                if os.path.exists(full_path):
                    return Image.open(full_path).convert('RGB')
            
            # Fallback: try different subdirectories
            possible_subdirs = ['dev', 'test1', 'train']
            
            for subdir in possible_subdirs:
                base_dir = os.path.join(self.image_base_dir, subdir)
                if not os.path.exists(base_dir):
                    continue
                    
                # Try different extensions
                for ext in ['.png', '.jpg', '.jpeg', '']:
                    if ext:
                        image_path = os.path.join(base_dir, f"{image_name}{ext}")
                    else:
                        image_path = os.path.join(base_dir, image_name)
                        
                    if os.path.exists(image_path):
                        return Image.open(image_path).convert('RGB')
            
            # If still not found, return dummy image
            print_rank(f"Warning: Image {image_name} not found, using dummy image")
            return Image.new('RGB', (224, 224), color=(128, 128, 128))
            
        except Exception as e:
            print_rank(f"Error loading image {image_name}: {e}")
            return Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # If not found, return a dummy image
        #print_rank(f"Warning: Image {image_name} not found, using dummy image")
        return Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    def _encode_batch(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """
        Unified encoding function using VLM2Vec's encode_input method
        This ensures consistency between training and evaluation
        
        Args:
            batch_data: Dict containing 'text' and 'images' keys
        
        Returns:
            torch.Tensor: Normalized embeddings
        """
        try:
            # Use VLM2Vec's unified processor
            from ..model.processor import process_vlm_inputs_fns
            
            # Process inputs using the same pipeline as training
            if self.model_backbone in process_vlm_inputs_fns:
                inputs = process_vlm_inputs_fns[self.model_backbone](
                    batch_data, 
                    self.processor,
                    max_length=getattr(self.data_args, 'max_len', 512)
                )
            else:
                # Fallback to standard processing
                inputs = self.processor(
                    text=batch_data['text'],
                    images=batch_data['images'],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.data_args, 'max_len', 512)
                )
            
            # Move to device
            for key in inputs:
                if hasattr(inputs[key], 'to'):
                    inputs[key] = inputs[key].to(self.device)
            
            # Use VLM2Vec's unified encode_input method (same as training)
            with torch.no_grad():
                embeddings = self.model.encode_input(inputs)
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
                
                # Handle different output formats
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)
                elif embeddings.dim() > 2:
                    embeddings = embeddings.view(embeddings.size(0), -1)
                
                return embeddings.cpu()
                
        except Exception as e:
            print_rank(f"Error in unified encoding: {e}")
            # Return dummy embeddings on error
            batch_size = len(batch_data['text'])
            return torch.randn(batch_size, 512)

    def _encode_images(self, image_names: List[str]) -> torch.Tensor:
        """Encode a batch of images using unified encoding interface (wrapper for compatibility)"""
        return self._encode_images_local(image_names)

    def _encode_composed_queries(self, queries: List[Dict]) -> torch.Tensor:
        """Encode composed queries using unified encoding interface (wrapper for compatibility)"""
        return self._encode_composed_queries_local(queries)
    
    def evaluate(self, distributed=False) -> Dict[str, float]:
        """
        Perform full evaluation on CIRR test set
        Returns recall@k metrics
        
        Args:
            distributed (bool): Whether to use distributed evaluation across multiple GPUs
        """
        print_master("Starting CIRR evaluation...")
        
        # Set model to eval mode
        self.model.eval()
        
        try:
            if distributed:
                return self._evaluate_distributed()
            else:
                return self._evaluate_single_gpu()
                
        except Exception as e:
            print_master(f"Evaluation failed: {e}")
            import traceback
            print_master(f"Traceback: {traceback.format_exc()}")
            # Return dummy metrics on failure
            return {
                'recall_at_1': 0.0,
                'recall_at_5': 0.0,
                'recall_at_10': 0.0,
                'recall_subset_at_1': 0.0,
                'recall_subset_at_2': 0.0,
                'recall_subset_at_3': 0.0,
                'group_recall_at_1': 0.0,
                'group_recall_at_2': 0.0,
                'group_recall_at_3': 0.0,
            }
    
    def _evaluate_single_gpu(self) -> Dict[str, float]:
        """Original single GPU evaluation"""
        # Encode all candidate images
        candidate_embeddings = self._encode_images(self.candidate_images)
        candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
        
        # Encode all composed queries
        query_embeddings = self._encode_composed_queries(self.test_data)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        
        # Compute similarities
        print_master("Computing similarities...")
        similarities = torch.mm(query_embeddings, candidate_embeddings.t())  # [num_queries, num_candidates]
        
        # Compute metrics using similarities (not rankings)
        metrics = self._compute_recall_metrics(similarities)
        
        print_master("Evaluation completed!")
        return metrics
    
    def _evaluate_distributed(self) -> Dict[str, float]:
        """
        Distributed evaluation across multiple GPUs
        Both candidate images and queries are processed in parallel across GPUs
        """
        try:
            import torch.distributed as dist
            
            if not dist.is_initialized():
                print_master("Warning: Distributed not initialized, falling back to single GPU")
                return self._evaluate_single_gpu()
            
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            print_master(f"Starting distributed evaluation on {world_size} GPUs")
            
            # Step 1: Distributed candidate image encoding
            print_master(f"Rank {rank}: Starting distributed candidate image encoding...")
            candidate_embeddings = self._encode_images_distributed()
            # Move to GPU for similarity computation
            candidate_embeddings = candidate_embeddings.to(self.device)
            candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
            
            # Step 2: Distributed query encoding  
            print_master(f"Rank {rank}: Starting distributed query encoding...")
            query_embeddings = self._encode_queries_distributed()
            # Move to GPU for similarity computation
            query_embeddings = query_embeddings.to(self.device)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            
            # Step 3: Compute similarities (all ranks have full embeddings now)
            # 这里所有进程都计算了一遍相似度，但是目前来看计算相似度的开销相对较小，所以暂时不优化
            print_master(f"Rank {rank}: Computing similarities...")
            print_master(f"Rank {rank}: Query embeddings shape: {query_embeddings.shape}, Candidate embeddings shape: {candidate_embeddings.shape}")
            similarities = torch.mm(query_embeddings, candidate_embeddings.t())
            
            # Step 4: Only rank 0 computes final metrics
            if rank == 0:
                print_master("Computing final metrics...")
                metrics = self._compute_recall_metrics(similarities)
                print_master(f"Distributed evaluation completed! Results: {metrics}")
                return metrics
            else:
                # Non-master ranks return empty dict (will be filled by broadcast in trainer)
                return {}
                
        except Exception as e:
            print_master(f"Distributed evaluation failed: {e}")
            import traceback
            print_master(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to single GPU evaluation on rank 0
            if not hasattr(dist, 'get_rank') or dist.get_rank() == 0:
                print_master("Falling back to single GPU evaluation")
                return self._evaluate_single_gpu()
            else:
                return {}
    
    def _encode_images_distributed(self) -> torch.Tensor:
        """
        Distributed encoding of candidate images with improved efficiency
        Each GPU processes a subset of images, then all embeddings are gathered
        """
        import torch.distributed as dist
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Ensure data is evenly divisible (pad if necessary)
        total_images = len(self.candidate_images)
        images_per_gpu = (total_images + world_size - 1) // world_size
        padded_total = images_per_gpu * world_size
        
        # Pad candidate images if necessary for even distribution
        if padded_total > total_images:
            padding_needed = padded_total - total_images
            # Repeat from beginning to pad
            padding_images = self.candidate_images[:padding_needed]
            candidate_images_padded = self.candidate_images + padding_images
            print_master(f"Padded candidate images: {total_images} -> {padded_total}")
        else:
            candidate_images_padded = self.candidate_images
        
        # Split evenly across GPUs
        start_idx = rank * images_per_gpu
        end_idx = start_idx + images_per_gpu
        
        local_images = candidate_images_padded[start_idx:end_idx]
        print_master(f"Rank {rank}: Encoding candidate images {start_idx}-{end_idx-1} ({len(local_images)} images)")
        
        # Encode local subset with mixed precision
        if len(local_images) > 0:
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                local_embeddings = self._encode_images_local(local_images)
            # Move to CPU to avoid GPU memory issues during gathering
            local_embeddings = local_embeddings.cpu()
        else:
            # Create empty tensor - we'll get actual dim from other ranks
            local_embeddings = torch.empty(0, 0)
        
        # Get embedding dimension dynamically
        local_dim = torch.tensor([local_embeddings.size(1) if local_embeddings.numel() > 0 else 0], dtype=torch.long, device=self.device)
        dim_list = [torch.zeros_like(local_dim) for _ in range(world_size)]
        dist.all_gather(dim_list, local_dim)
        
        embedding_dim = max([d.item() for d in dim_list])
        if embedding_dim == 0:
            embedding_dim = 512  # Fallback
        
        # Ensure local_embeddings has correct dimension
        if local_embeddings.numel() == 0:
            local_embeddings = torch.empty(0, embedding_dim)
        
        # Use all_gather_into_tensor for better efficiency
        local_size = images_per_gpu
        padded_embeddings = torch.zeros(local_size, embedding_dim, device=self.device)
        if local_embeddings.size(0) > 0:
            padded_embeddings[:local_embeddings.size(0)] = local_embeddings.to(self.device)
        
        # Efficient tensor gathering
        output_shape = [padded_total, embedding_dim]
        gathered_embeddings = torch.empty(output_shape, dtype=padded_embeddings.dtype, device=self.device)
        dist.all_gather_into_tensor(gathered_embeddings, padded_embeddings)
        
        # Trim back to original size and move to CPU
        final_embeddings = gathered_embeddings[:total_images].cpu()
        print_master(f"Rank {rank}: Reconstructed {final_embeddings.size(0)} candidate embeddings")
        
        return final_embeddings
    
    def _encode_queries_distributed(self) -> torch.Tensor:
        """
        Distributed encoding of queries with improved efficiency
        Each GPU processes a subset of queries, then all embeddings are gathered
        """
        import torch.distributed as dist
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Ensure data is evenly divisible (pad if necessary)
        total_queries = len(self.test_data)
        queries_per_gpu = (total_queries + world_size - 1) // world_size
        padded_total = queries_per_gpu * world_size
        
        # Pad queries if necessary for even distribution
        if padded_total > total_queries:
            padding_needed = padded_total - total_queries
            # Repeat from beginning to pad
            padding_queries = self.test_data[:padding_needed]
            test_data_padded = self.test_data + padding_queries
            print_master(f"Padded queries: {total_queries} -> {padded_total}")
        else:
            test_data_padded = self.test_data
        
        # Split evenly across GPUs
        start_idx = rank * queries_per_gpu
        end_idx = start_idx + queries_per_gpu
        
        local_queries = test_data_padded[start_idx:end_idx]
        print_master(f"Rank {rank}: Encoding queries {start_idx}-{end_idx-1} ({len(local_queries)} queries)")
        
        # Encode local subset with mixed precision
        if len(local_queries) > 0:
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                local_embeddings = self._encode_composed_queries_local(local_queries)
            # Move to CPU to avoid GPU memory issues during gathering
            local_embeddings = local_embeddings.cpu()
        else:
            # Create empty tensor - we'll get actual dim from other ranks
            local_embeddings = torch.empty(0, 0)
        
        # Get embedding dimension dynamically
        local_dim = torch.tensor([local_embeddings.size(1) if local_embeddings.numel() > 0 else 0], dtype=torch.long, device=self.device)
        dim_list = [torch.zeros_like(local_dim) for _ in range(world_size)]
        dist.all_gather(dim_list, local_dim)
        
        embedding_dim = max([d.item() for d in dim_list])
        if embedding_dim == 0:
            embedding_dim = 512  # Fallback
        
        # Ensure local_embeddings has correct dimension
        if local_embeddings.numel() == 0:
            local_embeddings = torch.empty(0, embedding_dim)
        
        # Use all_gather_into_tensor for better efficiency  
        local_size = queries_per_gpu
        padded_embeddings = torch.zeros(local_size, embedding_dim, device=self.device)
        if local_embeddings.size(0) > 0:
            padded_embeddings[:local_embeddings.size(0)] = local_embeddings.to(self.device)
        
        # Efficient tensor gathering
        output_shape = [padded_total, embedding_dim]
        gathered_embeddings = torch.empty(output_shape, dtype=padded_embeddings.dtype, device=self.device)
        dist.all_gather_into_tensor(gathered_embeddings, padded_embeddings)
        
        # Trim back to original size and move to CPU
        final_embeddings = gathered_embeddings[:total_queries].cpu()
        print_master(f"Rank {rank}: Reconstructed {final_embeddings.size(0)} query embeddings")
        
        return final_embeddings
    
    def _encode_images_local(self, image_names: List[str]) -> torch.Tensor:
        """Encode a subset of images on local GPU (used by distributed encoding)"""
        embeddings = []
        
        # Progress indication for distributed encoding
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            desc = f"Rank {rank}: Encoding images"
            # Only show progress bar on rank 0 to avoid cluttered output
            disable_tqdm = (rank != 0)
        else:
            desc = "Encoding images"
            disable_tqdm = False
        
        # Process in batches
        for i in tqdm(range(0, len(image_names), self.batch_size), desc=desc, disable=disable_tqdm):
            batch_names = image_names[i:i+self.batch_size]
            batch_images = [self._load_image(name) for name in batch_names]
            
            # 使用与 CIRR 训练集一致的目标编码指令
            from ..model.processor import process_input_text
            instruction = "Represent the given image in one word:"
            texts = [
                process_input_text(
                    instruction=instruction,
                    model_backbone=self.model_backbone,
                    text="",
                    add_image_token=True,
                )
                for _ in batch_images
            ]
            wrapped_images = [[img] for img in batch_images]
            
            batch_data = {
                'text': texts,
                'images': wrapped_images
            }
            if self.model_backbone in {GME, LamRA, LamRA_QWEN2_5}:
                batch_data['is_query'] = False
                batch_data['instruction'] = None
            
            # Use unified encoding function
            batch_embeddings = self._encode_batch(batch_data)
            embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 512, device=self.device)
    
    def _encode_composed_queries_local(self, queries: List[Dict]) -> torch.Tensor:
        """Encode a subset of composed queries on local GPU (used by distributed encoding)"""
        embeddings = []
        
        # Progress indication for distributed encoding
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            desc = f"Rank {rank}: Encoding queries"
            # Only show progress bar on rank 0 to avoid cluttered output
            disable_tqdm = (rank != 0)
        else:
            desc = "Encoding queries"
            disable_tqdm = False
        
        # Process in batches
        for i in tqdm(range(0, len(queries), self.batch_size), desc=desc, disable=disable_tqdm):
            batch_queries = queries[i:i+self.batch_size]
            
            # 与 CIRR 训练集一致：将 caption 嵌入到 instruction 中，text 传空串
            batch_images_raw = [self._load_image(q['reference']) for q in batch_queries]
            batch_images = [[img] for img in batch_images_raw]
            
            from ..model.processor import process_input_text
            batch_texts = []
            for q in batch_queries:
                cap = q.get('caption', '')
                instruction = f"Modify this image with <{cap}>\nRepresent the modified image in one word:"
                query_text = process_input_text(
                    instruction=instruction,
                    model_backbone=self.model_backbone,
                    text="",
                    add_image_token=True,
                )
                batch_texts.append(query_text)
            
            batch_data = {
                'text': batch_texts,
                'images': batch_images
            }
            if self.model_backbone in {GME, LamRA, LamRA_QWEN2_5}:
                batch_data['is_query'] = True
                batch_data['instruction'] = GME_CIRR_QUERY_INSTRUCTION
            
            # Use unified encoding function
            batch_embeddings = self._encode_batch(batch_data)
            embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 512, device=self.device)
    
    def _compute_recall_metrics(self, similarities: torch.Tensor) -> Dict[str, float]:
        """
        Compute Recall@K metrics for CIRR evaluation
        包含两种评估模式：全集Recall@K和Group Recall@K
        
        similarities: [num_queries, num_candidates] - similarity scores between queries and candidates
        """
        device = similarities.device
        num_queries, num_candidates = similarities.shape
        
        # Create mapping from image name to candidate index
        image_to_idx = {img: idx for idx, img in enumerate(self.candidate_images)}
        
        # Exclude reference images from ranking
        # 对每个query，将其reference image的分数设为负无穷，确保不会出现在top-k中
        for query_idx, query in enumerate(self.test_data):
            reference_image = query['reference']  # Use correct CIRR key name
            if reference_image in image_to_idx:
                ref_idx = image_to_idx[reference_image]
                similarities[query_idx, ref_idx] = -float('inf')
        
        # Create positive pairs matrix for global evaluation
        positive_pairs = torch.zeros((num_queries, num_candidates), dtype=torch.bool, device=device)
        valid_queries = 0
        
        for query_idx, query in enumerate(self.test_data):
            target_image = query['target_hard']  # Use correct CIRR key name
            if target_image in image_to_idx:
                target_idx = image_to_idx[target_image]
                positive_pairs[query_idx, target_idx] = True
                valid_queries += 1
        
        metrics = {}
        
        # Get recall@k values from config
        eval_config = self.eval_config.get('CIRR', {}).get('evaluation', {})
        global_recall_k = eval_config.get('global_recall_k', [1, 5, 10, 50])
        group_recall_k = eval_config.get('group_recall_k', [1, 2, 3])
        
        # 1. Compute Global Recall@K (standard evaluation)
        print_master("Computing Global Recall@K...")
        for k in global_recall_k:
            recall_scores = self._recall_at_k(similarities, positive_pairs, k)
            metrics[f'recall@{k}'] = (recall_scores > 0).float().mean().item() * 100
        
        # 2. Compute Group Recall@K (within img_set groups)
        print_master("Computing Group Recall@K...")
        group_metrics = self._compute_group_recall(similarities, image_to_idx, group_recall_k)
        metrics.update(group_metrics)
        
        # Print results
        print_master("=" * 60)
        print_master("CIRR Evaluation Results:")
        print_master("=" * 60)
        print_master(f"Valid queries: {valid_queries} / {len(self.test_data)}")
        print_master("")
        print_master("Global Recall@K:")
        for k in [1, 5, 10, 50]:
            key = f'recall@{k}'
            if key in metrics:
                print_master(f"  Recall@{k}: {metrics[key]:.2f}%")
        
        print_master("")
        print_master("Group Recall@K:")
        for k in [1, 2, 3]:
            key = f'group_recall@{k}'
            if key in metrics:
                print_master(f"  Group Recall@{k}: {metrics[key]:.2f}%")
        print_master("=" * 60)
        
        # Return metrics with original naming for compatibility
        result_metrics = {}
        for k, v in metrics.items():
            if k.startswith('recall@'):
                k_num = k.split('@')[1]
                result_metrics[f'r_at_{k_num}'] = v / 100.0  # Convert back to 0-1 range
            else:
                result_metrics[k] = v / 100.0
        
        return result_metrics
    
    def _recall_at_k(self, scores: torch.Tensor, positive_pairs: torch.Tensor, k: int) -> torch.Tensor:
        """计算Recall@K"""
        batch_size = 32
        device = scores.device
        
        # 分批处理避免内存不足
        results = []
        for start in range(0, len(scores), batch_size):
            end = start + batch_size
            batch_scores = scores[start:end]
            batch_positive = positive_pairs[start:end]
            
            # 获取Top-K索引
            topk_indices = torch.topk(batch_scores, k, dim=1)[1]
            
            # 获取每个查询的正确目标索引
            target_indices = torch.argmax(batch_positive.long(), dim=1)
            
            # 检查每个查询的正确目标是否在Top-K结果中
            correct = torch.zeros(len(target_indices), dtype=torch.float, device=device)
            for i, (target_idx, topk) in enumerate(zip(target_indices, topk_indices)):
                correct[i] = 1.0 if target_idx in topk else 0.0
            
            results.append(correct.cpu())
        
        return torch.cat(results)
    
    def _compute_group_recall(self, similarities: torch.Tensor, image_to_idx: Dict[str, int], group_recall_k: List[int]) -> Dict[str, float]:
        """计算Group Recall@K (within img_set groups)"""
        device = similarities.device
        group_scores = []
        group_positive_pairs = []
        
        # 对每个查询，创建组内评估的子集
        for query_idx, query in enumerate(self.test_data):
            target_image = query['target_hard']  # Use correct CIRR key name
            img_set = query.get('img_set', {})
            
            if target_image in image_to_idx and img_set:
                # 获取所属组的候选图像索引 - img_set is a dict with 'members' key
                group_members = img_set.get('members', []) if isinstance(img_set, dict) else img_set
                group_indices = []
                for member in group_members:
                    if member in image_to_idx:
                        group_indices.append(image_to_idx[member])
                
                if group_indices and len(group_indices) > 1:  # 至少要有2个候选才有意义
                    # 提取该查询与组内所有候选的分数
                    group_score = similarities[query_idx, group_indices]
                    
                    # 创建该组内的正样本对
                    group_positive = torch.zeros(len(group_indices), dtype=torch.bool, device=device)
                    try:
                        target_idx_in_group = group_indices.index(image_to_idx[target_image])
                        group_positive[target_idx_in_group] = True
                        
                        group_scores.append(group_score)
                        group_positive_pairs.append(group_positive)
                    except ValueError:
                        # Target not in group, skip this query
                        continue
        
        metrics = {}
        
        # 如果有有效的组评估数据
        if group_scores:
            group_scores = torch.stack(group_scores)
            group_positive_pairs = torch.stack(group_positive_pairs)
            
            # 计算Group Recall@K
            for k in group_recall_k:
                if k <= min([len(p) for p in group_positive_pairs]):
                    recall_scores = self._recall_at_k(group_scores, group_positive_pairs, k)
                    metrics[f"group_recall@{k}"] = (recall_scores > 0).float().mean().item() * 100
        else:
            print_master("Warning: No valid groups found for group recall evaluation")
            for k in group_recall_k:
                metrics[f"group_recall@{k}"] = 0.0
        
        return metrics

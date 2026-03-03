"""
FashionIQ Dataset Evaluator for Composed Image Retrieval
"""

import os
import json
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
 
try:
    from ..utils import print_rank, print_master
except ImportError:
    def print_rank(*args, **kwargs):
        print(*args, **kwargs)
    def print_master(*args, **kwargs):
        print(*args, **kwargs)


class FashionIQEvaluator:
    """
    Evaluator for FashionIQ dataset. Supports per-category evaluation and global recall@K.
    """

    def __init__(self,
                 model,
                 processor,
                 data_args,
                 model_args,
                 device: str = 'cuda',
                 batch_size: int = 16,
                 eval_config_path: str = None):
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

        # Get model backbone
        self.model_backbone = getattr(model_args, 'model_backbone', None)
        if self.model_backbone is None:
            self.model_backbone = self.eval_config.get('model', {}).get('default_backbone', 'qwen2_vl')

        # Configure data paths
        self._configure_data_paths()

        # Load queries and candidate images by category
        self.category_data = self._load_fashioniq_test_data()
        
        # Print summary
        total_queries = sum(len(self.category_data[cat]['queries']) for cat in self.categories)
        total_candidates = sum(len(self.category_data[cat]['candidates']) for cat in self.categories)
        print_master(f"Loaded {total_queries} FashionIQ validation queries across {len(self.categories)} categories")
        for cat in self.categories:
            print_master(f"  {cat}: {len(self.category_data[cat]['queries'])} queries, {len(self.category_data[cat]['candidates'])} candidates")

    # ------------------------- config & data loading -------------------------

    def _load_eval_config(self, config_path=None) -> Dict[str, Any]:
        import yaml
        if config_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(current_dir, 'configs', 'fashioniq_eval_config.yaml')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print_master(f"Loaded FashionIQ eval config: {config_path}")
                return config or self._get_default_config()
            else:
                print_master(f"Config not found: {config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            print_master(f"Error loading config {config_path}: {e}. Using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'FashionIQ': {
                'data_dir': '/home/guohaiyun/yty_data/FashionIQ',
                'image_base_dir': '/home/guohaiyun/yty_data/FashionIQ/images',
                'categories': ['dress', 'shirt', 'toptee'],
                'validation': {
                    'queries_files': {
                        'dress': 'captions/cap.dress.val.json',
                        'shirt': 'captions/cap.shirt.val.json',
                        'toptee': 'captions/cap.toptee.val.json'
                    }
                },
                'evaluation': {
                    'batch_size': 16,
                    'recall_k': [1, 5, 10]
                }
            },
            'model': {
                'default_backbone': 'qwen2_vl',
                'max_length': 512,
                # Note: Instructions are hardcoded to match training prompts exactly
            }
        }

    def _configure_data_paths(self) -> None:
        fiq_cfg = self.eval_config.get('FashionIQ', {})

        # Prefer dataset YAML path if provided
        self.data_dir = fiq_cfg.get('data_dir', '/home/guohaiyun/yty_data/FashionIQ')
        self.image_base_dir = fiq_cfg.get('image_base_dir', os.path.join(self.data_dir, 'images'))
        self.categories = fiq_cfg.get('categories', ['dress', 'shirt', 'toptee'])
        
        # Query caption files
        val_cfg = fiq_cfg.get('validation', {}).get('queries_files', {})
        if not val_cfg:
            val_cfg = {
                'dress': 'captions/cap.dress.val.json',
                'shirt': 'captions/cap.shirt.val.json',
                'toptee': 'captions/cap.toptee.val.json',
            }
        self.queries_files = {k: os.path.join(self.data_dir, v) for k, v in val_cfg.items()}
        
        # Image split files
        self.split_dir = os.path.join(self.data_dir, 'image_splits')
        self.split_files = {
            'dress': os.path.join(self.split_dir, 'split.dress.val.json'),
            'shirt': os.path.join(self.split_dir, 'split.shirt.val.json'),
            'toptee': os.path.join(self.split_dir, 'split.toptee.val.json'),
        }

        print_master(f"FashionIQ data_dir: {self.data_dir}")
        print_master(f"FashionIQ image_base_dir: {self.image_base_dir}")

    def _load_fashioniq_test_data(self) -> Dict[str, Dict[str, Any]]:
        """Load queries and candidates for each category separately from split files"""
        category_data = {}
        
        for cat in self.categories:
            # Load candidate images from split file
            split_file = self.split_files.get(cat)
            if not split_file or not os.path.exists(split_file):
                print_master(f"Warning: Missing split file for {cat}: {split_file}")
                category_data[cat] = {'queries': [], 'candidates': []}
                continue
            
            try:
                with open(split_file, 'r') as f:
                    image_ids = json.load(f)
                # Add .png extension (FashionIQ images are in PNG format)
                candidates = [f"{img_id}.png" for img_id in image_ids]
            except Exception as e:
                print_master(f"Failed to load split file {split_file}: {e}")
                candidates = []
            
            # Load queries
            qfile = self.queries_files.get(cat)
            queries = []
            if not qfile or not os.path.exists(qfile):
                print_master(f"Warning: Missing queries file for {cat}: {qfile}")
            else:
                try:
                    with open(qfile, 'r') as f:
                        items = json.load(f)
                    for it in items:
                        query = {
                            'reference': it.get('candidate', ''),
                            'target_hard': it.get('target', ''),
                            'caption': ' and '.join(it.get('captions', [])) if it.get('captions') else '',
                            'category': cat,
                        }
                        queries.append(query)
                except Exception as e:
                    print_master(f"Failed to load {qfile}: {e}")
            
            category_data[cat] = {
                'queries': queries,
                'candidates': candidates
            }
        
        return category_data

    # ------------------------------ I/O helpers ------------------------------

    def _load_image(self, rel_path: str) -> Image.Image:
        try:
            # First try the path as-is
            full = os.path.join(self.image_base_dir, rel_path)
            if os.path.exists(full):
                return Image.open(full).convert('RGB')
            
            # If path contains category prefix (e.g. "toptee/xxx.png"), try removing it for flat structure
            if os.sep in rel_path:
                # Extract just the filename
                filename = os.path.basename(rel_path)
                full_flat = os.path.join(self.image_base_dir, filename)
                if os.path.exists(full_flat):
                    return Image.open(full_flat).convert('RGB')
                # Try with alternate extensions
                root, ext = os.path.splitext(full_flat)
                for alt in ['.png', '.jpg', '.jpeg']:
                    alt_path = root + alt
                    if os.path.exists(alt_path):
                        return Image.open(alt_path).convert('RGB')
            
            # Try alternate extension swapping on original path
            root, ext = os.path.splitext(full)
            for alt in ['.png', '.jpg', '.jpeg']:
                alt_path = root + alt
                if os.path.exists(alt_path):
                    return Image.open(alt_path).convert('RGB')
            
            print_rank(f"Warning: Image not found: {rel_path}")
            return Image.new('RGB', (224, 224), color=(128, 128, 128))
        except Exception as e:
            print_rank(f"Error loading image {rel_path}: {e}")
            return Image.new('RGB', (224, 224), color=(128, 128, 128))

    # ---------------------------- encoding wrappers ----------------------------

    def _encode_batch(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        try:
            from ..model.processor import process_vlm_inputs_fns
            if self.model_backbone in process_vlm_inputs_fns:
                inputs = process_vlm_inputs_fns[self.model_backbone](
                    batch_data,
                    self.processor,
                    max_length=getattr(self.data_args, 'max_len', 512)
                )
            else:
                inputs = self.processor(
                    text=batch_data['text'],
                    images=batch_data['images'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.data_args, 'max_len', 512)
                )
            for k in inputs:
                if hasattr(inputs[k], 'to'):
                    inputs[k] = inputs[k].to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode_input(inputs)
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)
                elif embeddings.dim() > 2:
                    embeddings = embeddings = embeddings.view(embeddings.size(0), -1)
                return embeddings.cpu()
        except Exception as e:
            print_rank(f"Error in encoding: {e}")
            bs = len(batch_data['text'])
            return torch.randn(bs, 512)

    def _encode_images(self, rel_paths: List[str]) -> torch.Tensor:
        """Encode images (wrapper for compatibility, delegates to _encode_images_local)"""
        return self._encode_images_local(rel_paths)
    
    def _encode_images_local(self, rel_paths: List[str]) -> torch.Tensor:
        """Encode a subset of images on local GPU (used by both single and distributed encoding)"""
        from ..model.processor import process_input_text
        
        all_embeddings = []
        
        # Progress indication for distributed encoding
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            desc = f"Rank {rank}: Encoding images"
            # Only show progress bar on rank 0 to avoid cluttered output
            disable_tqdm = (rank != 0)
        else:
            desc = "Encoding candidate images"
            disable_tqdm = False
        
        num_batches = (len(rel_paths) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, len(rel_paths), self.batch_size), 
                      desc=desc, 
                      total=num_batches,
                      disable=disable_tqdm):
            batch_paths = rel_paths[i:i + self.batch_size]
            images: List[List[Image.Image]] = []
            texts: List[str] = []
            for p in batch_paths:
                images.append([self._load_image(p)])
                # Use the same instruction as training target images
                text = process_input_text(
                    instruction="Represent the given image in one word:",
                    # instruction="Represent the given image",
                    model_backbone=self.model_backbone,
                    text="",
                    add_image_token=True,
                )
                texts.append(text)
            batch = {'text': texts, 'images': images}
            batch_emb = self._encode_batch(batch)
            all_embeddings.append(batch_emb)
        
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, 512, device=self.device)

    def _encode_composed_queries(self, queries: List[Dict[str, Any]]) -> torch.Tensor:
        """Encode queries (wrapper for compatibility, delegates to _encode_composed_queries_local)"""
        return self._encode_composed_queries_local(queries)
    
    def _encode_composed_queries_local(self, queries: List[Dict[str, Any]]) -> torch.Tensor:
        """Encode a subset of composed queries on local GPU (used by both single and distributed encoding)"""
        from ..model.processor import process_input_text
        
        all_embeddings = []
        
        # Progress indication for distributed encoding
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            desc = f"Rank {rank}: Encoding queries"
            # Only show progress bar on rank 0 to avoid cluttered output
            disable_tqdm = (rank != 0)
        else:
            desc = "Encoding composed queries"
            disable_tqdm = False
        
        num_batches = (len(queries) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, len(queries), self.batch_size), 
                      desc=desc, 
                      total=num_batches,
                      disable=disable_tqdm):
            batch_queries = queries[i:i + self.batch_size]
            images: List[List[Image.Image]] = []
            texts: List[str] = []
            
            for q in batch_queries:
                # Build path candidates
                ref = q.get('reference', '')
                cat = q.get('category', '')
                caption = q.get('caption', '')
                rel = ref
                
                # If ref doesn't have extension, try to find it
                if not ref.lower().endswith(('.png', '.jpg', '.jpeg')):
                    matched = False
                    # Try flat structure first (just filename with extension)
                    for ext in ['.png', '.jpg', '.jpeg']:
                        candidate_flat = f"{ref}{ext}"
                        if os.path.exists(os.path.join(self.image_base_dir, candidate_flat)):
                            rel = candidate_flat
                            matched = True
                            break
                    
                    # If not found, try category-based structure
                    if not matched and cat:
                        for ext in ['.png', '.jpg', '.jpeg']:
                            candidate = os.path.join(cat, f"{ref}{ext}")
                            if os.path.exists(os.path.join(self.image_base_dir, candidate)):
                                rel = candidate
                                matched = True
                                break
                    
                    # Default fallback
                    if not matched:
                        rel = f"{ref}.png"
                images.append([self._load_image(rel)])
                # Format text with category-aware prompt and add image token
                # MUST match training prompt EXACTLY
                instruction = f"Change the style of this {cat} to <{caption}>\nRepresent the modified {cat} in one word:" if caption else ""
                # instruction = f"Modify this image with <{caption}>\nRepresent the modified image in one word:"
                # instruction = f"Find an image to match the fashion image and style note.\n{caption}"  # 🔥 Fixed: removed trailing comma
                formatted_text = process_input_text(
                    instruction=instruction,
                    model_backbone=self.model_backbone,
                    text="",
                    add_image_token=True,
                )
                texts.append(formatted_text)
            
            batch = {'text': texts, 'images': images}
            batch_emb = self._encode_batch(batch)
            all_embeddings.append(batch_emb)
        
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, 512, device=self.device)
    
    # ------------------------ distributed encoding methods ----------------------
    
    def _encode_images_distributed(self, rel_paths: List[str]) -> torch.Tensor:
        """
        Distributed encoding of candidate images with improved efficiency
        Each GPU processes a subset of images, then all embeddings are gathered
        """
        import torch.distributed as dist
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Ensure data is evenly divisible (pad if necessary)
        total_images = len(rel_paths)
        images_per_gpu = (total_images + world_size - 1) // world_size
        padded_total = images_per_gpu * world_size
        
        # Pad candidate images if necessary for even distribution
        if padded_total > total_images:
            padding_needed = padded_total - total_images
            # Repeat from beginning to pad
            padding_images = rel_paths[:padding_needed]
            rel_paths_padded = rel_paths + padding_images
            print_master(f"Padded candidate images: {total_images} -> {padded_total}")
        else:
            rel_paths_padded = rel_paths
        
        # Split evenly across GPUs
        start_idx = rank * images_per_gpu
        end_idx = start_idx + images_per_gpu
        
        local_images = rel_paths_padded[start_idx:end_idx]
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
    
    def _encode_queries_distributed(self, queries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Distributed encoding of queries with improved efficiency
        Each GPU processes a subset of queries, then all embeddings are gathered
        """
        import torch.distributed as dist
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Ensure data is evenly divisible (pad if necessary)
        total_queries = len(queries)
        queries_per_gpu = (total_queries + world_size - 1) // world_size
        padded_total = queries_per_gpu * world_size
        
        # Pad queries if necessary for even distribution
        if padded_total > total_queries:
            padding_needed = padded_total - total_queries
            # Repeat from beginning to pad
            padding_queries = queries[:padding_needed]
            queries_padded = queries + padding_queries
            print_master(f"Padded queries: {total_queries} -> {padded_total}")
        else:
            queries_padded = queries
        
        # Split evenly across GPUs
        start_idx = rank * queries_per_gpu
        end_idx = start_idx + queries_per_gpu
        
        local_queries = queries_padded[start_idx:end_idx]
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

    # -------------------------------- evaluation --------------------------------

    def evaluate(self, distributed: bool = False) -> Dict[str, float]:
        """
        Perform full evaluation on FashionIQ test set
        
        Args:
            distributed (bool): Whether to use distributed evaluation across multiple GPUs
        """
        print_master("Starting FashionIQ evaluation...")
        self.model.eval()
        
        try:
            if distributed:
                return self._evaluate_distributed()
            else:
                return self._evaluate_by_category()
        except Exception as e:
            print_master(f"FashionIQ evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'recall_at_1': 0.0,
                'recall_at_5': 0.0,
                'recall_at_10': 0.0,
            }

    def _evaluate_distributed(self) -> Dict[str, float]:
        """
        Distributed evaluation across multiple GPUs
        Both candidate images and queries are processed in parallel across GPUs
        Each category is evaluated separately, then results are averaged
        """
        try:
            import torch.distributed as dist
            
            if not dist.is_initialized():
                print_master("Warning: Distributed not initialized, falling back to single GPU")
                return self._evaluate_by_category()
            
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            print_master(f"Starting distributed FashionIQ evaluation on {world_size} GPUs")
            
            category_results = {}
            
            for cat in self.categories:
                print_master(f"\n{'='*60}")
                print_master(f"Rank {rank}: Evaluating category: {cat}")
                print_master(f"{'='*60}")
                
                queries = self.category_data[cat]['queries']
                candidates = self.category_data[cat]['candidates']
                
                if not queries or not candidates:
                    print_master(f"Rank {rank}: Skipping {cat}: no queries or candidates")
                    continue
                
                print_master(f"Rank {rank}: {cat} - Queries: {len(queries)}, Candidates: {len(candidates)}")
                
                # Step 1: Distributed candidate encoding for this category
                print_master(f"Rank {rank}: Encoding {cat} candidate images (distributed)...")
                cand_emb = self._encode_images_distributed(candidates)
                cand_emb = cand_emb.to(self.device)
                cand_emb = F.normalize(cand_emb, p=2, dim=1)
                
                # Step 2: Distributed query encoding for this category
                print_master(f"Rank {rank}: Encoding {cat} queries (distributed)...")
                qry_emb = self._encode_queries_distributed(queries)
                qry_emb = qry_emb.to(self.device)
                qry_emb = F.normalize(qry_emb, p=2, dim=1)
                
                # Step 3: Compute similarities (all ranks)
                print_master(f"Rank {rank}: Computing similarity matrix for {cat}...")
                print_master(f"Rank {rank}: Query embeddings shape: {qry_emb.shape}, Candidate embeddings shape: {cand_emb.shape}")
                sims = torch.mm(qry_emb, cand_emb.t())
                
                # Step 4: Only rank 0 computes metrics for this category
                if rank == 0:
                    print_master(f"Computing recall metrics for {cat}...")
                    metrics = self._compute_recall_metrics_single_category(sims, queries, candidates)
                    category_results[cat] = metrics
                    
                    # Print category results
                    print_master(f"\n{cat.upper()} Results:")
                    for k, v in metrics.items():
                        print_master(f"  {k}: {v:.4f}")
            
            # Only rank 0 computes final averaged results
            if rank == 0:
                print_master(f"\n{'='*60}")
                print_master("Computing average metrics across categories...")
                print_master(f"{'='*60}")
                
                avg_metrics = {}
                if category_results:
                    # Get all metric keys from first category
                    metric_keys = list(next(iter(category_results.values())).keys())
                    for key in metric_keys:
                        values = [category_results[cat][key] for cat in category_results if key in category_results[cat]]
                        avg_metrics[key] = sum(values) / len(values) if values else 0.0
                    
                    # Also add per-category metrics
                    for cat, metrics in category_results.items():
                        for key, value in metrics.items():
                            avg_metrics[f'{cat}_{key}'] = value
                
                print_master(f"Distributed FashionIQ evaluation completed! Results: {avg_metrics}")
                return avg_metrics
            else:
                # Non-master ranks return empty dict
                return {}
                
        except Exception as e:
            print_master(f"Distributed evaluation failed: {e}")
            import traceback
            print_master(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to single GPU evaluation on rank 0
            import torch.distributed as dist
            if not hasattr(dist, 'get_rank') or dist.get_rank() == 0:
                print_master("Falling back to single GPU evaluation")
                return self._evaluate_by_category()
            else:
                return {}

    def _evaluate_by_category(self) -> Dict[str, float]:
        """Evaluate each category separately and compute average metrics (single GPU)"""
        category_results = {}
        
        for cat in self.categories:
            print_master(f"\n{'='*60}")
            print_master(f"Evaluating category: {cat}")
            print_master(f"{'='*60}")
            
            queries = self.category_data[cat]['queries']
            candidates = self.category_data[cat]['candidates']
            
            if not queries or not candidates:
                print_master(f"Skipping {cat}: no queries or candidates")
                continue
            
            print_master(f"Queries: {len(queries)}, Candidates: {len(candidates)}")
            
            # Encode candidates for this category
            print_master(f"Encoding {cat} candidate images...")
            cand_emb = self._encode_images(candidates)
            cand_emb = F.normalize(cand_emb, p=2, dim=1)
            
            # Encode queries for this category
            print_master(f"Encoding {cat} composed queries...")
            qry_emb = self._encode_composed_queries(queries)
            qry_emb = F.normalize(qry_emb, p=2, dim=1)
            
            # Compute similarities
            print_master(f"Computing similarity matrix for {cat}...")
            sims = torch.mm(qry_emb, cand_emb.t())
            
            # Debug: Print first query's top-10 results and similarity scores
            if len(queries) > 0 and len(candidates) > 0:
                first_query = queries[0]
                first_sims = sims[0]
                top_k_vals, top_k_indices = torch.topk(first_sims, k=min(10, len(candidates)))
                print_master(f"\nDebug - First query:")
                print_master(f"  Reference: {first_query.get('reference', 'N/A')}")
                print_master(f"  Target: {first_query.get('target_hard', 'N/A')}")
                print_master(f"  Caption: {first_query.get('caption', 'N/A')}")
                print_master(f"  Top-10 candidates (similarity scores):")
                for i, (idx, sim) in enumerate(zip(top_k_indices[:10], top_k_vals[:10])):
                    cand_name = candidates[idx.item()]
                    is_target = '✓ TARGET' if cand_name.replace('.png', '') == first_query.get('target_hard', '') else ''
                    print_master(f"    {i+1}. {cand_name} (sim: {sim.item():.4f}) {is_target}")
            
            # Compute metrics
            print_master(f"Computing recall metrics for {cat}...")
            metrics = self._compute_recall_metrics_single_category(sims, queries, candidates)
            category_results[cat] = metrics
            
            # Print category results
            print_master(f"\n{cat.upper()} Results:")
            for k, v in metrics.items():
                print_master(f"  {k}: {v:.4f}")
        
        # Compute average across categories
        print_master(f"\n{'='*60}")
        print_master("Computing average metrics across categories...")
        print_master(f"{'='*60}")
        
        avg_metrics = {}
        if category_results:
            # Get all metric keys from first category
            metric_keys = list(next(iter(category_results.values())).keys())
            for key in metric_keys:
                values = [category_results[cat][key] for cat in category_results if key in category_results[cat]]
                avg_metrics[key] = sum(values) / len(values) if values else 0.0
            
            # Also add per-category metrics
            for cat, metrics in category_results.items():
                for key, value in metrics.items():
                    avg_metrics[f'{cat}_{key}'] = value
        
        return avg_metrics

    def _compute_recall_metrics_single_category(
        self, 
        similarities: torch.Tensor, 
        queries: List[Dict[str, Any]], 
        candidates: List[str]
    ) -> Dict[str, float]:
        """Compute recall metrics for a single category"""
        device = similarities.device
        num_queries, num_candidates = similarities.shape
        
        # Build candidate stems mapping (without extension)
        candidate_stems = {}
        for idx, cand_path in enumerate(candidates):
            # Remove extension
            stem = os.path.splitext(cand_path)[0]
            candidate_stems[stem] = idx
        
        # Exclude reference images from ranking
        for qi, q in enumerate(queries):
            ref_id = q.get('reference', '')
            if ref_id in candidate_stems:
                idx = candidate_stems[ref_id]
                similarities[qi, idx] = -float('inf')
        
        # Build positive pairs matrix
        positive = torch.zeros((num_queries, num_candidates), dtype=torch.bool, device=device)
        valid = 0
        for qi, q in enumerate(queries):
            tgt_id = q.get('target_hard', '')
            if tgt_id in candidate_stems:
                idx = candidate_stems[tgt_id]
                positive[qi, idx] = True
                valid += 1
        
        print_master(f"  Valid query-target pairs: {valid}/{num_queries}")
        
        # Get recall k list from config
        eval_cfg = self.eval_config.get('FashionIQ', {}).get('evaluation', {})
        k_list = eval_cfg.get('recall_k', [1, 5, 10, 50])
        
        # Compute recall@k
        results = {}
        for k in k_list:
            r = self._recall_at_k(similarities, positive, k)
            recall = (r > 0).float().mean().item()
            results[f'r_at_{k}'] = recall
        
        return results

    def _compute_recall_metrics(self, similarities: torch.Tensor) -> Dict[str, float]:
        device = similarities.device
        num_queries, num_candidates = similarities.shape

        # Build mapping from candidate index to category
        cand_cat = self.candidate_categories
        # Build mapping for target path resolution
        # Use set of candidate rel paths without extension to match ids
        candidate_stems = {}
        for idx, rel in enumerate(self.candidate_images):
            stem = os.path.splitext(rel)[0]  # "dress/xxxx" or "xxxx"
            candidate_stems.setdefault(stem, idx)
            # Also add just the filename stem for flat structure compatibility
            basename_stem = os.path.splitext(os.path.basename(rel))[0]  # "xxxx"
            candidate_stems.setdefault(basename_stem, idx)

        # Exclude reference image from ranking if resolvable
        for qi, q in enumerate(self.test_data):
            ref_id = q.get('reference', '')
            cat = q.get('category', '')
            matched = False
            # Try flat structure first (just image ID)
            if ref_id in candidate_stems:
                idx = candidate_stems[ref_id]
                similarities[qi, idx] = -float('inf')
                matched = True
            # Try with extensions
            if not matched:
                for ext in ['.png', '.jpg', '.jpeg', '']:
                    # Try flat: just ref_id with extension
                    key_flat = ref_id + ext if ext else ref_id
                    stem_flat = os.path.splitext(key_flat)[0]
                    if stem_flat in candidate_stems:
                        idx = candidate_stems[stem_flat]
                        similarities[qi, idx] = -float('inf')
                        matched = True
                        break
                    # Try category-based: cat/ref_id with extension
                    if cat:
                        key = os.path.join(cat, ref_id + ext) if ext else os.path.join(cat, ref_id)
                        stem = os.path.splitext(key)[0]
                        if stem in candidate_stems:
                            idx = candidate_stems[stem]
                            similarities[qi, idx] = -float('inf')
                            matched = True
                            break

        # Positive pairs matrix for global recall
        positive = torch.zeros((num_queries, num_candidates), dtype=torch.bool, device=device)
        valid = 0
        for qi, q in enumerate(self.test_data):
            tgt_id = q.get('target_hard', '')
            cat = q.get('category', '')
            matched = False
            # Try flat structure first (just image ID)
            if tgt_id in candidate_stems:
                idx = candidate_stems[tgt_id]
                positive[qi, idx] = True
                valid += 1
                matched = True
            # Try with extensions
            if not matched:
                for ext in ['.png', '.jpg', '.jpeg', '']:
                    # Try flat: just tgt_id with extension
                    key_flat = tgt_id + ext if ext else tgt_id
                    stem_flat = os.path.splitext(key_flat)[0]
                    if stem_flat in candidate_stems:
                        idx = candidate_stems[stem_flat]
                        positive[qi, idx] = True
                        valid += 1
                        matched = True
                        break
                    # Try category-based: cat/tgt_id with extension
                    if cat:
                        key = os.path.join(cat, tgt_id + ext) if ext else os.path.join(cat, tgt_id)
                        stem = os.path.splitext(key)[0]
                        if stem in candidate_stems:
                            idx = candidate_stems[stem]
                            positive[qi, idx] = True
                            valid += 1
                            matched = True
                            break

        # K list
        eval_cfg = self.eval_config.get('FashionIQ', {}).get('evaluation', {})
        k_list = eval_cfg.get('recall_k', [1, 5, 10])

        # Global recall@k
        results: Dict[str, float] = {}
        for k in k_list:
            r = self._recall_at_k(similarities, positive, k)
            results[f'recall@{k}'] = (r > 0).float().mean().item()

        # Category recall@k (restrict to same category)
        from collections import defaultdict
        cat_to_indices = defaultdict(list)
        for idx, rel in enumerate(self.candidate_images):
            c = rel.split(os.sep)[0] if os.sep in rel else ''
            cat_to_indices[c].append(idx)

        for k in k_list:
            correct = []
            for qi, q in enumerate(self.test_data):
                c = q.get('category', '')
                cand_idx_list = cat_to_indices.get(c, [])
                if not cand_idx_list:
                    correct.append(0.0)
                    continue
                scores = similarities[qi, cand_idx_list]
                pos = positive[qi, cand_idx_list]
                topk = torch.topk(scores, min(k, scores.numel()), dim=0)[1]
                try:
                    tgt_idx = torch.argmax(pos.long()).item()
                    hit = 1.0 if tgt_idx in topk.tolist() else 0.0
                except Exception:
                    hit = 0.0
                correct.append(hit)
            results[f'category_recall@{k}'] = float(sum(correct) / len(correct) if correct else 0.0)

        # Map to r_at_k for compatibility
        final = {}
        for k, v in results.items():
            if k.startswith('recall@'):
                final[f'r_at_{k.split("@")[1]}'] = v
            else:
                final[k] = v
        return final

    def _recall_at_k(self, scores: torch.Tensor, positive_pairs: torch.Tensor, k: int) -> torch.Tensor:
        batch = 32
        res = []
        device = scores.device
        for s in range(0, scores.size(0), batch):
            e = min(s + batch, scores.size(0))
            sc = scores[s:e]
            pos = positive_pairs[s:e]
            topk = torch.topk(sc, min(k, sc.size(1)), dim=1)[1]
            tgt = torch.argmax(pos.long(), dim=1)
            corr = torch.zeros(tgt.size(0), dtype=torch.float, device=device)
            for i, (t, tk) in enumerate(zip(tgt, topk)):
                corr[i] = 1.0 if t in tk else 0.0
            res.append(corr.cpu())
        return torch.cat(res) if res else torch.zeros(0)



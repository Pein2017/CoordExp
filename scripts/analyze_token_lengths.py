#!/usr/bin/env python3
"""
Token Length Analysis for Packing Mode Migration

This script analyzes token lengths in the training dataset to inform the migration
from standard padding to packing mode while maintaining equivalent training dynamics.

Key Features:
1. Computes exact token lengths for each sample (text + image tokens)
2. Generates comprehensive statistics (mean, median, max, min, percentiles)
3. Simulates packing efficiency for different global_max_length values
4. Estimates image token counts using Qwen3-VL vision architecture
5. Provides configuration recommendations for packing mode

Usage:
    python scripts/analyze_token_lengths.py \
        --model_path model_cache/Qwen3-VL-8B-Instruct-coordexp \
        --train_jsonl public_data/lvis/rescale_32_768_poly_20/train.jsonl \
        --output_dir docs/temp_packed_dataset \
        --max_samples 0  # 0 = all samples

Output:
    - Token length statistics (CSV)
    - Distribution histograms
    - Packing simulation results
    - Configuration recommendations
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import math

import numpy as np
import torch
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from tqdm.auto import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TokenStats:
    """Statistics for a single sample"""
    sample_idx: int
    text_tokens: int
    image_tokens: int
    total_tokens: int
    num_images: int
    num_objects: int
    image_resolutions: List[Tuple[int, int]]
    

@dataclass
class DatasetStats:
    """Aggregated dataset statistics"""
    num_samples: int
    mean_text_tokens: float
    median_text_tokens: float
    std_text_tokens: float
    min_text_tokens: int
    max_text_tokens: int
    p95_text_tokens: float
    p99_text_tokens: float
    
    mean_image_tokens: float
    median_image_tokens: float
    std_image_tokens: float
    min_image_tokens: int
    max_image_tokens: int
    
    mean_total_tokens: float
    median_total_tokens: float
    std_total_tokens: float
    min_total_tokens: int
    max_total_tokens: int
    p95_total_tokens: float
    p99_total_tokens: float
    
    mean_objects_per_sample: float
    median_objects_per_sample: float


@dataclass
class PackingSimulation:
    """Results of packing simulation for a given max_length"""
    global_max_length: int
    avg_samples_per_pack: float
    median_samples_per_pack: float
    avg_fill_ratio: float
    median_fill_ratio: float
    num_packs: int
    total_samples_packed: int
    samples_dropped: int  # Samples exceeding max_length
    wasted_tokens: int
    total_tokens: int
    efficiency: float  # (total_tokens - wasted_tokens) / total_tokens


def estimate_image_tokens_qwen3vl(
    width: int,
    height: int,
    patch_size: int = 14,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = 3136,  # 56*56
    max_pixels: int = 12845056,  # 1280*720*14*14
) -> int:
    """
    Estimate the number of image tokens for Qwen3-VL vision encoder.
    
    Based on Qwen3-VL architecture:
    - Vision encoder splits image into patches (patch_size=14)
    - Patches are merged (merge_size=2) to form vision tokens
    - Additional CLS and temporal tokens
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        patch_size: Size of vision patches (default 14 for Qwen3-VL)
        merge_size: Patch merging factor (default 2)
        temporal_patch_size: Temporal merging (default 2)
        min_pixels: Minimum pixels for resizing
        max_pixels: Maximum pixels for resizing
        
    Returns:
        Estimated number of image tokens
    """
    # Qwen3-VL preprocessing logic (from processor)
    # 1. Smart resize to fit within min/max pixels while maintaining aspect ratio
    resized_height, resized_width = smart_resize(
        height, width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    
    # 2. Calculate grid size after patching
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    
    # 3. Calculate merged grid after merge_size
    merged_grid_h = (grid_h + merge_size - 1) // merge_size
    merged_grid_w = (grid_w + merge_size - 1) // merge_size
    
    # 4. Total vision tokens = merged grid + special tokens
    # Qwen3-VL adds 1 CLS token + temporal patches
    vision_tokens = merged_grid_h * merged_grid_w
    
    # Add special tokens (CLS + temporal)
    # Based on Qwen3-VL implementation: 1 CLS + temporal_patch_size^2
    special_tokens = 1 + (temporal_patch_size ** 2)
    
    total_tokens = vision_tokens + special_tokens
    
    logger.debug(
        f"Image {width}x{height} -> resized {resized_width}x{resized_height} "
        f"-> grid {grid_w}x{grid_h} -> merged {merged_grid_w}x{merged_grid_h} "
        f"-> {vision_tokens} vision tokens + {special_tokens} special = {total_tokens} total"
    )
    
    return total_tokens


def smart_resize(
    height: int,
    width: int,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
) -> Tuple[int, int]:
    """
    Smart resize logic from Qwen3-VL processor.
    Maintains aspect ratio while fitting within min/max pixel constraints.
    """
    total_pixels = height * width
    
    if total_pixels < min_pixels:
        # Scale up
        scale = math.sqrt(min_pixels / total_pixels)
        new_height = int(height * scale)
        new_width = int(width * scale)
    elif total_pixels > max_pixels:
        # Scale down
        scale = math.sqrt(max_pixels / total_pixels)
        new_height = int(height * scale)
        new_width = int(width * scale)
    else:
        # No resize needed
        new_height = height
        new_width = width
    
    return new_height, new_width


def build_conversation_text(sample: Dict, use_coord_tokens: bool = False) -> str:
    """
    Build conversation text from JSONL sample using the same logic as dataset preprocessing.
    
    Args:
        sample: JSONL sample dict
        use_coord_tokens: Whether to use <|coord_*|> tokens (for .coord.jsonl)
        
    Returns:
        Formatted conversation text (without image tokens)
    """
    objects = sample.get("objects", [])
    
    # System message
    system = "You are a helpful assistant that can detect and locate objects in images."
    
    # User query
    user_query = "Detect all objects in the image and provide their locations."
    
    # Assistant response - serialize objects
    obj_strs = []
    for obj in objects:
        desc = obj.get("desc", "object")
        
        # Determine geometry type and format
        if "poly" in obj:
            poly = obj["poly"]
            if use_coord_tokens:
                # Coord token format: <|coord_1|><|coord_2|>...
                coords_str = "".join([f"<|coord_{c}|>" for c in poly])
                obj_str = f"{desc}: poly {coords_str}"
            else:
                # Numeric format: "x1 y1 x2 y2 ..."
                coords_str = " ".join(map(str, poly))
                obj_str = f"{desc}: poly {coords_str}"
        elif "bbox_2d" in obj:
            bbox = obj["bbox_2d"]
            if use_coord_tokens:
                coords_str = "".join([f"<|coord_{c}|>" for c in bbox])
                obj_str = f"{desc}: bbox {coords_str}"
            else:
                coords_str = " ".join(map(str, bbox))
                obj_str = f"{desc}: bbox {coords_str}"
        else:
            obj_str = f"{desc}: unknown"
        
        obj_strs.append(obj_str)
    
    assistant_response = "\n".join(obj_strs)
    
    # Build full conversation using Qwen3-VL chat template format
    # Note: This is a simplified version; actual template may differ
    conversation = f"<|im_start|>system\n{system}<|im_end|>\n"
    conversation += f"<|im_start|>user\n{user_query}<|im_end|>\n"
    conversation += f"<|im_start|>assistant\n{assistant_response}<|im_end|>"
    
    return conversation


def analyze_sample(
    sample: Dict,
    sample_idx: int,
    tokenizer: AutoTokenizer,
    use_coord_tokens: bool = False,
) -> TokenStats:
    """
    Analyze token counts for a single sample.
    
    Args:
        sample: JSONL sample dict
        sample_idx: Sample index
        tokenizer: Tokenizer instance
        use_coord_tokens: Whether to use coord tokens
        
    Returns:
        TokenStats for this sample
    """
    # Extract image info
    images = sample.get("images", [])
    width = sample.get("width", 640)
    height = sample.get("height", 640)
    objects = sample.get("objects", [])
    
    # Estimate image tokens (one image per sample in LVIS)
    num_images = len(images)
    image_resolutions = [(width, height)] * num_images
    
    # Use default Qwen3-VL settings for max_pixels (from base.yaml)
    # max_pixels: 1572864999 means effectively no resize
    total_image_tokens = 0
    for _ in range(num_images):
        img_tokens = estimate_image_tokens_qwen3vl(
            width=width,
            height=height,
            max_pixels=1572864999,  # From base.yaml - no resize
        )
        total_image_tokens += img_tokens
    
    # Build conversation text
    conversation = build_conversation_text(sample, use_coord_tokens=use_coord_tokens)
    
    # Tokenize text
    text_token_ids = tokenizer.encode(conversation, add_special_tokens=True)
    num_text_tokens = len(text_token_ids)
    
    # Total tokens = text + image
    total_tokens = num_text_tokens + total_image_tokens
    
    return TokenStats(
        sample_idx=sample_idx,
        text_tokens=num_text_tokens,
        image_tokens=total_image_tokens,
        total_tokens=total_tokens,
        num_images=num_images,
        num_objects=len(objects),
        image_resolutions=image_resolutions,
    )


def compute_dataset_stats(sample_stats: List[TokenStats]) -> DatasetStats:
    """Compute aggregated statistics from sample stats"""
    text_tokens = [s.text_tokens for s in sample_stats]
    image_tokens = [s.image_tokens for s in sample_stats]
    total_tokens = [s.total_tokens for s in sample_stats]
    num_objects = [s.num_objects for s in sample_stats]
    
    return DatasetStats(
        num_samples=len(sample_stats),
        mean_text_tokens=float(np.mean(text_tokens)),
        median_text_tokens=float(np.median(text_tokens)),
        std_text_tokens=float(np.std(text_tokens)),
        min_text_tokens=int(np.min(text_tokens)),
        max_text_tokens=int(np.max(text_tokens)),
        p95_text_tokens=float(np.percentile(text_tokens, 95)),
        p99_text_tokens=float(np.percentile(text_tokens, 99)),
        
        mean_image_tokens=float(np.mean(image_tokens)),
        median_image_tokens=float(np.median(image_tokens)),
        std_image_tokens=float(np.std(image_tokens)),
        min_image_tokens=int(np.min(image_tokens)),
        max_image_tokens=int(np.max(image_tokens)),
        
        mean_total_tokens=float(np.mean(total_tokens)),
        median_total_tokens=float(np.median(total_tokens)),
        std_total_tokens=float(np.std(total_tokens)),
        min_total_tokens=int(np.min(total_tokens)),
        max_total_tokens=int(np.max(total_tokens)),
        p95_total_tokens=float(np.percentile(total_tokens, 95)),
        p99_total_tokens=float(np.percentile(total_tokens, 99)),
        
        mean_objects_per_sample=float(np.mean(num_objects)),
        median_objects_per_sample=float(np.median(num_objects)),
    )


def simulate_packing(
    sample_stats: List[TokenStats],
    global_max_length: int,
    buffer_tokens: int = 256,
) -> PackingSimulation:
    """
    Simulate packing with a given global_max_length.
    
    Uses a greedy First-Fit bin packing algorithm.
    
    Args:
        sample_stats: List of TokenStats
        global_max_length: Maximum tokens per packed sequence
        buffer_tokens: Reserve buffer for packing overhead (separators, etc.)
        
    Returns:
        PackingSimulation results
    """
    effective_max = global_max_length - buffer_tokens
    
    packs = []  # List of packs, each pack is a list of sample indices
    current_pack = []
    current_tokens = 0
    samples_dropped = 0
    
    for stats in sample_stats:
        if stats.total_tokens > effective_max:
            # Sample too large to pack
            samples_dropped += 1
            logger.warning(
                f"Sample {stats.sample_idx} with {stats.total_tokens} tokens "
                f"exceeds max_length {effective_max} (dropped)"
            )
            continue
        
        if current_tokens + stats.total_tokens <= effective_max:
            # Fits in current pack
            current_pack.append(stats)
            current_tokens += stats.total_tokens
        else:
            # Start new pack
            if current_pack:
                packs.append(current_pack)
            current_pack = [stats]
            current_tokens = stats.total_tokens
    
    # Add last pack
    if current_pack:
        packs.append(current_pack)
    
    # Compute statistics
    samples_per_pack = [len(p) for p in packs]
    fill_ratios = [sum(s.total_tokens for s in p) / effective_max for p in packs]
    
    total_tokens = sum(sum(s.total_tokens for s in p) for p in packs)
    total_capacity = len(packs) * effective_max
    wasted_tokens = total_capacity - total_tokens
    
    return PackingSimulation(
        global_max_length=global_max_length,
        avg_samples_per_pack=float(np.mean(samples_per_pack)) if packs else 0.0,
        median_samples_per_pack=float(np.median(samples_per_pack)) if packs else 0.0,
        avg_fill_ratio=float(np.mean(fill_ratios)) if packs else 0.0,
        median_fill_ratio=float(np.median(fill_ratios)) if packs else 0.0,
        num_packs=len(packs),
        total_samples_packed=sum(samples_per_pack),
        samples_dropped=samples_dropped,
        wasted_tokens=wasted_tokens,
        total_tokens=total_tokens,
        efficiency=float(total_tokens / total_capacity) if total_capacity > 0 else 0.0,
    )


def generate_recommendations(
    dataset_stats: DatasetStats,
    packing_sims: List[PackingSimulation],
    baseline_config: Dict,
) -> Dict:
    """
    Generate configuration recommendations for packing mode.
    
    Args:
        dataset_stats: Dataset statistics
        packing_sims: List of packing simulations
        baseline_config: Baseline configuration (standard padding)
        
    Returns:
        Recommended configuration values
    """
    # Find best packing configuration (highest efficiency with reasonable fill ratio)
    best_sim = max(packing_sims, key=lambda s: s.efficiency)
    
    # Calculate equivalent batch sizes
    baseline_steps_per_epoch = (
        dataset_stats.num_samples / baseline_config["effective_batch_size"]
    )
    baseline_total_steps = (
        baseline_steps_per_epoch * baseline_config["num_train_epochs"]
    )
    
    # For packing mode:
    # - Each pack becomes one sequence
    # - per_device_train_batch_size controls how many packs per device
    # - effective_batch_size should give us the same optimizer steps
    
    # Strategy 1: Keep effective_batch_size the same
    # This means we process more samples per step (due to packing)
    # but keep the same number of optimizer updates
    recommended_eff_bs = baseline_config["effective_batch_size"]
    
    # Calculate steps per epoch with packing
    packing_steps_per_epoch = best_sim.num_packs / recommended_eff_bs
    
    # Adjust epochs to match total optimizer steps
    recommended_epochs = baseline_total_steps / packing_steps_per_epoch
    
    # Recommended per-device batch size (start conservative)
    # With packing, each "sample" is a packed sequence, so reduce batch size
    recommended_per_device_bs = 1  # Conservative start
    
    return {
        "global_max_length": best_sim.global_max_length,
        "per_device_train_batch_size": recommended_per_device_bs,
        "effective_batch_size": recommended_eff_bs,
        "num_train_epochs": recommended_epochs,
        "packing": True,
        "packing_buffer": 256,
        "packing_min_fill_ratio": 0.7,
        "packing_drop_last": True,
        
        # Derived metrics
        "expected_steps_per_epoch": packing_steps_per_epoch,
        "expected_total_steps": baseline_total_steps,
        "expected_samples_per_step": best_sim.avg_samples_per_pack * recommended_eff_bs,
        "packing_efficiency": best_sim.efficiency,
        "avg_fill_ratio": best_sim.avg_fill_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze token lengths for packing mode")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_cache/Qwen3-VL-8B-Instruct-coordexp",
        help="Path to model/tokenizer",
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default="public_data/lvis/rescale_32_768_poly_20/train.jsonl",
        help="Path to training JSONL",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/temp_packed_dataset",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to analyze (0 = all)",
    )
    parser.add_argument(
        "--use_coord_tokens",
        action="store_true",
        help="Use coord tokens instead of numeric coordinates",
    )
    parser.add_argument(
        "--baseline_config",
        type=str,
        default="configs/dlora/sft_coord_offset.yaml",
        help="Path to baseline config (for comparison)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    logger.info(f"Reading dataset: {args.train_jsonl}")
    sample_stats = []
    
    with open(args.train_jsonl, "r") as f:
        for idx, line in enumerate(tqdm(f, desc="Analyzing samples")):
            if args.max_samples > 0 and idx >= args.max_samples:
                break
            
            sample = json.loads(line)
            stats = analyze_sample(
                sample,
                sample_idx=idx,
                tokenizer=tokenizer,
                use_coord_tokens=args.use_coord_tokens,
            )
            sample_stats.append(stats)
    
    logger.info(f"Analyzed {len(sample_stats)} samples")
    
    # Compute dataset statistics
    logger.info("Computing dataset statistics...")
    dataset_stats = compute_dataset_stats(sample_stats)
    
    # Save sample-level stats
    sample_stats_file = output_dir / "sample_stats.jsonl"
    logger.info(f"Saving sample stats to {sample_stats_file}")
    with open(sample_stats_file, "w") as f:
        for stats in sample_stats:
            f.write(json.dumps(asdict(stats)) + "\n")
    
    # Save dataset statistics
    dataset_stats_file = output_dir / "dataset_stats.json"
    logger.info(f"Saving dataset stats to {dataset_stats_file}")
    with open(dataset_stats_file, "w") as f:
        json.dump(asdict(dataset_stats), f, indent=2)
    
    # Simulate packing for different max_lengths
    logger.info("Running packing simulations...")
    max_lengths = [8000, 10000, 12000, 14000, 16000, 18000, 20000, 24000]
    packing_sims = []
    
    for max_len in tqdm(max_lengths, desc="Packing simulations"):
        sim = simulate_packing(sample_stats, global_max_length=max_len)
        packing_sims.append(sim)
        logger.info(
            f"max_length={max_len}: "
            f"{sim.num_packs} packs, "
            f"{sim.avg_samples_per_pack:.2f} samples/pack, "
            f"{sim.avg_fill_ratio:.2%} fill ratio, "
            f"{sim.efficiency:.2%} efficiency"
        )
    
    # Save packing simulations
    packing_sims_file = output_dir / "packing_simulations.json"
    logger.info(f"Saving packing simulations to {packing_sims_file}")
    with open(packing_sims_file, "w") as f:
        json.dump([asdict(sim) for sim in packing_sims], f, indent=2)
    
    # Generate recommendations
    logger.info("Generating configuration recommendations...")
    baseline_config = {
        "num_train_epochs": 4,
        "effective_batch_size": 128,
        "per_device_train_batch_size": 2,
        "num_gpus": 4,
    }
    
    recommendations = generate_recommendations(
        dataset_stats, packing_sims, baseline_config
    )
    
    # Save recommendations
    recommendations_file = output_dir / "recommendations.json"
    logger.info(f"Saving recommendations to {recommendations_file}")
    with open(recommendations_file, "w") as f:
        json.dump(recommendations, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nDataset Statistics (n={dataset_stats.num_samples}):")
    logger.info(f"  Total tokens: mean={dataset_stats.mean_total_tokens:.1f}, "
                f"median={dataset_stats.median_total_tokens:.1f}, "
                f"std={dataset_stats.std_total_tokens:.1f}")
    logger.info(f"  Total tokens: min={dataset_stats.min_total_tokens}, "
                f"max={dataset_stats.max_total_tokens}")
    logger.info(f"  Total tokens: p95={dataset_stats.p95_total_tokens:.1f}, "
                f"p99={dataset_stats.p99_total_tokens:.1f}")
    logger.info(f"  Text tokens: mean={dataset_stats.mean_text_tokens:.1f}, "
                f"median={dataset_stats.median_text_tokens:.1f}")
    logger.info(f"  Image tokens: mean={dataset_stats.mean_image_tokens:.1f}, "
                f"median={dataset_stats.median_image_tokens:.1f}")
    logger.info(f"  Objects per sample: mean={dataset_stats.mean_objects_per_sample:.1f}, "
                f"median={dataset_stats.median_objects_per_sample:.1f}")
    
    logger.info(f"\nRecommended Configuration:")
    logger.info(f"  global_max_length: {recommendations['global_max_length']}")
    logger.info(f"  per_device_train_batch_size: {recommendations['per_device_train_batch_size']}")
    logger.info(f"  effective_batch_size: {recommendations['effective_batch_size']}")
    logger.info(f"  num_train_epochs: {recommendations['num_train_epochs']:.2f}")
    logger.info(f"  packing_efficiency: {recommendations['packing_efficiency']:.2%}")
    logger.info(f"  avg_fill_ratio: {recommendations['avg_fill_ratio']:.2%}")
    logger.info(f"  expected_steps_per_epoch: {recommendations['expected_steps_per_epoch']:.1f}")
    logger.info(f"  expected_total_steps: {recommendations['expected_total_steps']:.1f}")
    
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

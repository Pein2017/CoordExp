#!/usr/bin/env python3
"""Verify model training by comparing weights between original and merged checkpoints.

This tool verifies that coordinate tokens and other model layers were successfully
trained by comparing weights between the original checkpoint and the merged model.

Features:
  - Coordinate token verification: Compares embeddings and lm_head weights for
    coordinate tokens (0-999, typically IDs 151670-152669) to confirm training
  - Adapter checkpoint inspection: Optionally checks adapter checkpoints for
    coord_offset_adapter weights directly
  - Layer update verification: Optionally checks if other layers (vision, LLM,
    aligner) were updated during training

Usage:
  # Basic: Verify coordinate tokens were trained
  conda run -n ms python scripts/tools/verify_coord_tokens.py \\
    --original model_cache/Qwen3-VL-8B-Instruct-coordexp \\
    --merged output/debug/coord_merged

  # With adapter checkpoint verification
  conda run -n ms python scripts/tools/verify_coord_tokens.py \\
    --original model_cache/Qwen3-VL-8B-Instruct-coordexp \\
    --merged output/debug/coord_merged \\
    --adapter output/debug/coord/checkpoint-15

  # Also check other layers (vision, LLM, aligner)
  conda run -n ms python scripts/tools/verify_coord_tokens.py \\
    --original model_cache/Qwen3-VL-8B-Instruct-coordexp \\
    --merged output/debug/coord_merged \\
    --check-layers

  # Use GPU for faster loading
  conda run -n ms python scripts/tools/verify_coord_tokens.py \\
    --original model_cache/Qwen3-VL-8B-Instruct-coordexp \\
    --merged output/debug/coord_merged \\
    --device cuda:0

Exit codes:
  0: Success - coordinate tokens (and optionally other layers) were trained
  1: Failure - coordinate tokens were not trained, or other errors occurred
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def find_safetensors_files(model_path: Path):
    """Find all safetensors files in a model directory."""
    safetensors_files = sorted(model_path.glob("model*.safetensors"))
    if not safetensors_files:
        # Try pytorch_model files as fallback
        pytorch_files = sorted(model_path.glob("pytorch_model*.bin"))
        if pytorch_files:
            return pytorch_files, False
        raise ValueError(f"No model files found in {model_path}")
    return safetensors_files, True


def load_all_keys(model_path: Path):
    """Load all weight keys from model files."""
    model_files, _ = find_safetensors_files(model_path)
    all_keys = set()

    for model_file in model_files:
        with safe_open(str(model_file), framework="pt", device="cpu") as f:
            all_keys.update(f.keys())

    return sorted(all_keys)


def sample_weights(model_path: Path, key: str, device: str = "cpu"):
    """Load a specific weight tensor from model files."""
    model_files, use_safetensors = find_safetensors_files(model_path)

    for model_file in model_files:
        if use_safetensors:
            with safe_open(str(model_file), framework="pt", device=device) as f:
                if key in f.keys():
                    return f.get_tensor(key)
        else:
            state_dict = torch.load(
                str(model_file), map_location=device, weights_only=True
            )
            if key in state_dict:
                return state_dict[key]

    return None


def compare_weight(
    key: str,
    orig_path: Path,
    merged_path: Path,
    device: str = "cpu",
    threshold: float = 1e-6,
):
    """Compare a single weight between original and merged models."""
    orig_weight = sample_weights(orig_path, key, device)
    merged_weight = sample_weights(merged_path, key, device)

    if orig_weight is None or merged_weight is None:
        return None

    if orig_weight.shape != merged_weight.shape:
        return {
            "error": f"Shape mismatch: {orig_weight.shape} vs {merged_weight.shape}"
        }

    diff = (merged_weight - orig_weight).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()

    # Sample a few values to show
    num_samples = min(5, orig_weight.numel())
    flat_orig = orig_weight.flatten()[:num_samples]
    flat_merged = merged_weight.flatten()[:num_samples]
    flat_diff = diff.flatten()[:num_samples]

    return {
        "shape": orig_weight.shape,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "changed": max_diff > threshold,
        "samples": {
            "orig": [f"{v:.6f}" for v in flat_orig.tolist()],
            "merged": [f"{v:.6f}" for v in flat_merged.tolist()],
            "diff": [f"{v:.6f}" for v in flat_diff.tolist()],
        },
    }


def check_other_layers(
    orig_path: Path, merged_path: Path, device: str = "cpu", threshold: float = 1e-6
):
    """Check if other model layers (vision, LLM, aligner) were updated."""
    print("\n" + "=" * 80)
    print("CHECKING OTHER LAYERS (Vision, LLM, Aligner)")
    print("=" * 80)

    # Get all keys from merged model
    print("\n[1] Loading weight keys from merged model...")
    merged_keys = load_all_keys(merged_path)
    print(f"  Found {len(merged_keys)} weight keys")

    # Sample keys from different layers
    print("\n[2] Sampling weights from different layers...")

    # Vision layers
    vision_keys = [
        k for k in merged_keys if "visual" in k.lower() and "blocks" in k.lower()
    ][:5]
    # LLM layers
    llm_keys = [
        k
        for k in merged_keys
        if "language_model" in k.lower() and "layers" in k.lower()
    ][:5]
    # Aligner layers
    aligner_keys = [
        k for k in merged_keys if ("merger" in k.lower() or "aligner" in k.lower())
    ][:5]

    print(f"  Vision layer samples: {len(vision_keys)}")
    print(f"  LLM layer samples: {len(llm_keys)}")
    print(f"  Aligner layer samples: {len(aligner_keys)}")

    all_results = {}

    # Compare vision layers
    print("\n" + "=" * 80)
    print("VISION LAYERS")
    print("=" * 80)
    for key in vision_keys:
        result = compare_weight(key, orig_path, merged_path, device, threshold)
        all_results[key] = result
        if result is None:
            print(f"\n{key}: Not found in one or both models")
        elif "error" in result:
            print(f"\n{key}: {result['error']}")
        else:
            status = "✓ CHANGED" if result["changed"] else "✗ UNCHANGED"
            print(f"\n{key}")
            print(f"  Shape: {result['shape']}")
            print(f"  Status: {status}")
            print(f"  Max diff: {result['max_diff']:.6f}")
            print(f"  Mean diff: {result['mean_diff']:.6f}")
            print(f"  Std diff: {result['std_diff']:.6f}")
            if result["changed"]:
                print("  Sample values (orig -> merged, diff):")
                for i, (o, m, d) in enumerate(
                    zip(
                        result["samples"]["orig"],
                        result["samples"]["merged"],
                        result["samples"]["diff"],
                    )
                ):
                    print(f"    [{i}]: {o} -> {m}, diff={d}")

    # Compare LLM layers
    print("\n" + "=" * 80)
    print("LLM LAYERS")
    print("=" * 80)
    for key in llm_keys:
        result = compare_weight(key, orig_path, merged_path, device, threshold)
        all_results[key] = result
        if result is None:
            print(f"\n{key}: Not found in one or both models")
        elif "error" in result:
            print(f"\n{key}: {result['error']}")
        else:
            status = "✓ CHANGED" if result["changed"] else "✗ UNCHANGED"
            print(f"\n{key}")
            print(f"  Shape: {result['shape']}")
            print(f"  Status: {status}")
            print(f"  Max diff: {result['max_diff']:.6f}")
            print(f"  Mean diff: {result['mean_diff']:.6f}")
            print(f"  Std diff: {result['std_diff']:.6f}")
            if result["changed"]:
                print("  Sample values (orig -> merged, diff):")
                for i, (o, m, d) in enumerate(
                    zip(
                        result["samples"]["orig"],
                        result["samples"]["merged"],
                        result["samples"]["diff"],
                    )
                ):
                    print(f"    [{i}]: {o} -> {m}, diff={d}")

    # Compare aligner layers
    print("\n" + "=" * 80)
    print("ALIGNER LAYERS")
    print("=" * 80)
    for key in aligner_keys:
        result = compare_weight(key, orig_path, merged_path, device, threshold)
        all_results[key] = result
        if result is None:
            print(f"\n{key}: Not found in one or both models")
        elif "error" in result:
            print(f"\n{key}: {result['error']}")
        else:
            status = "✓ CHANGED" if result["changed"] else "✗ UNCHANGED"
            print(f"\n{key}")
            print(f"  Shape: {result['shape']}")
            print(f"  Status: {status}")
            print(f"  Max diff: {result['max_diff']:.6f}")
            print(f"  Mean diff: {result['mean_diff']:.6f}")
            print(f"  Std diff: {result['std_diff']:.6f}")
            if result["changed"]:
                print("  Sample values (orig -> merged, diff):")
                for i, (o, m, d) in enumerate(
                    zip(
                        result["samples"]["orig"],
                        result["samples"]["merged"],
                        result["samples"]["diff"],
                    )
                ):
                    print(f"    [{i}]: {o} -> {m}, diff={d}")

    # Summary
    print("\n" + "=" * 80)
    print("LAYER UPDATE SUMMARY")
    print("=" * 80)

    all_keys_to_check = vision_keys + llm_keys + aligner_keys
    changed_count = 0
    unchanged_count = 0

    for key in all_keys_to_check:
        result = all_results.get(key)
        if result and "error" not in result:
            if result["changed"]:
                changed_count += 1
            else:
                unchanged_count += 1

    print(f"\n  Changed layers: {changed_count}/{len(all_keys_to_check)}")
    print(f"  Unchanged layers: {unchanged_count}/{len(all_keys_to_check)}")

    if changed_count > 0:
        print("\n  ✓ Some layers were updated during training")
    else:
        print("\n  ✗ No layers were updated (merged model identical to original)")

    return changed_count > 0


def load_embedding_weights(model_path: Path, device: str = "cpu"):
    """Load embedding and lm_head weights directly from safetensors files."""
    print(f"\n[Loading weights from] {model_path}")

    # Load config to get model structure
    try:
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        print(
            f"[Config] vocab_size={config.vocab_size}, hidden_size={config.hidden_size}"
        )
    except Exception as e:
        print(f"[WARNING] Could not load config: {e}")
        config = None

    # Find safetensors files
    try:
        model_files, use_safetensors = find_safetensors_files(model_path)
        print(f"[Found] {len(model_files)} model file(s)")
    except Exception as e:
        print(f"[ERROR] {e}")
        return None, None

    # Try to find embed_tokens and lm_head weight keys
    # Qwen3-VL uses language_model.embed_tokens.weight and language_model.lm_head.weight
    embed_key = None
    head_key = None

    # Common key patterns
    possible_embed_keys = [
        "language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "embed_tokens.weight",
    ]
    possible_head_keys = [
        "lm_head.weight",  # Often at root level
        "language_model.lm_head.weight",
        "model.lm_head.weight",
        "model.language_model.lm_head.weight",
    ]

    # Scan all files to find the keys (lm_head might be in a different file)
    print("[Scanning] model files for weight keys...")

    if use_safetensors:
        for model_file in model_files:
            with safe_open(str(model_file), framework="pt", device=device) as f:
                keys = list(f.keys())
                for key in keys:
                    if embed_key is None and any(k in key for k in possible_embed_keys):
                        embed_key = key
                        print(f"[Found] embed_tokens key: {key} in {model_file.name}")
                    if head_key is None and any(k in key for k in possible_head_keys):
                        head_key = key
                        print(f"[Found] lm_head key: {key} in {model_file.name}")
                # Also check for "head" in general
                if head_key is None:
                    for key in keys:
                        if "head" in key.lower() and "weight" in key:
                            head_key = key
                            print(
                                f"[Found] lm_head key (by pattern): {key} in {model_file.name}"
                            )
                            break
    else:
        # PyTorch format
        for model_file in model_files:
            state_dict = torch.load(
                str(model_file), map_location=device, weights_only=True
            )
            for key in state_dict.keys():
                if embed_key is None and any(k in key for k in possible_embed_keys):
                    embed_key = key
                    print(f"[Found] embed_tokens key: {key} in {model_file.name}")
                if head_key is None and any(k in key for k in possible_head_keys):
                    head_key = key
                    print(f"[Found] lm_head key: {key} in {model_file.name}")
            # Also check for "head" in general
            if head_key is None:
                for key in state_dict.keys():
                    if "head" in key.lower() and "weight" in key:
                        head_key = key
                        print(
                            f"[Found] lm_head key (by pattern): {key} in {model_file.name}"
                        )
                        break

    if embed_key is None or head_key is None:
        print("[ERROR] Could not find embed_tokens or lm_head weights")
        if not embed_key:
            print("  Missing: embed_tokens.weight")
        if not head_key:
            print("  Missing: lm_head.weight")
        # Show available keys for debugging
        if use_safetensors and model_files:
            with safe_open(str(model_files[0]), framework="pt", device=device) as f:
                print(f"  Available keys (first 10): {list(f.keys())[:10]}")
        return None, None

    # Load weights from all files
    embed_weight = None
    head_weight = None

    if use_safetensors:
        # Load from safetensors
        for model_file in model_files:
            with safe_open(str(model_file), framework="pt", device=device) as f:
                if embed_key in f.keys() and embed_weight is None:
                    embed_weight = f.get_tensor(embed_key)
                    print(
                        f"[Loaded] {embed_key} from {model_file.name}: shape={embed_weight.shape}"
                    )
                if head_key in f.keys() and head_weight is None:
                    head_weight = f.get_tensor(head_key)
                    print(
                        f"[Loaded] {head_key} from {model_file.name}: shape={head_weight.shape}"
                    )
    else:
        # Load from PyTorch format
        for model_file in model_files:
            state_dict = torch.load(
                str(model_file), map_location=device, weights_only=True
            )
            if embed_key in state_dict and embed_weight is None:
                embed_weight = state_dict[embed_key]
                print(
                    f"[Loaded] {embed_key} from {model_file.name}: shape={embed_weight.shape}"
                )
            if head_key in state_dict and head_weight is None:
                head_weight = state_dict[head_key]
                print(
                    f"[Loaded] {head_key} from {model_file.name}: shape={head_weight.shape}"
                )

    if embed_weight is None or head_weight is None:
        print("[ERROR] Failed to load weights")
        return None, None

    print(f"[Info] Embedding shape: {embed_weight.shape}")
    print(f"[Info] LM head shape: {head_weight.shape}")

    return embed_weight, head_weight


def extract_coord_embeddings(
    embed_weight: torch.Tensor, head_weight: torch.Tensor, coord_ids: range
):
    """Extract embeddings and head weights for coordinate token IDs."""
    coord_ids_tensor = torch.tensor(list(coord_ids), dtype=torch.long)

    # Extract embeddings (vocab_size, hidden_dim)
    coord_embeds = embed_weight[coord_ids_tensor].clone()

    # Extract head weights (vocab_size, hidden_dim) - these are the output projection weights
    coord_head = head_weight[coord_ids_tensor].clone()

    return coord_embeds, coord_head, coord_ids_tensor


def compare_embeddings(
    orig_embeds,
    merged_embeds,
    orig_head,
    merged_head,
    coord_ids: range,
    threshold: float = 1e-6,
):
    """Compare embeddings and head weights between original and merged models."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Embedding comparison
    embed_diff = (merged_embeds - orig_embeds).abs()
    embed_max_diff = embed_diff.max().item()
    embed_mean_diff = embed_diff.mean().item()
    embed_std_diff = embed_diff.std().item()

    print(
        f"\n[Embeddings] Token IDs: {coord_ids.start} to {coord_ids.stop - 1} (n={len(coord_ids)})"
    )
    print(f"  Max difference: {embed_max_diff:.6f}")
    print(f"  Mean difference: {embed_mean_diff:.6f}")
    print(f"  Std difference: {embed_std_diff:.6f}")

    # Check if any tokens changed significantly
    tokens_changed = (embed_diff.max(dim=1)[0] > threshold).sum().item()
    print(f"  Tokens with changes > {threshold}: {tokens_changed}/{len(coord_ids)}")

    if embed_max_diff < threshold:
        print(
            "  ⚠️  WARNING: Embeddings appear unchanged (training may not have occurred)"
        )
    else:
        print("  ✓ Embeddings have been modified (training occurred)")

    # Head comparison
    head_diff = (merged_head - orig_head).abs()
    head_max_diff = head_diff.max().item()
    head_mean_diff = head_diff.mean().item()
    head_std_diff = head_diff.std().item()

    print(
        f"\n[LM Head] Token IDs: {coord_ids.start} to {coord_ids.stop - 1} (n={len(coord_ids)})"
    )
    print(f"  Max difference: {head_max_diff:.6f}")
    print(f"  Mean difference: {head_mean_diff:.6f}")
    print(f"  Std difference: {head_std_diff:.6f}")

    head_tokens_changed = (head_diff.max(dim=1)[0] > threshold).sum().item()
    print(
        f"  Tokens with changes > {threshold}: {head_tokens_changed}/{len(coord_ids)}"
    )

    if head_max_diff < threshold:
        print(
            "  ⚠️  WARNING: LM head weights appear unchanged (training may not have occurred)"
        )
    else:
        print("  ✓ LM head weights have been modified (training occurred)")

    # Per-token statistics
    print("\n[Per-Token Statistics]")
    print("  Embedding changes per token (max):")
    embed_per_token_max = embed_diff.max(dim=1)[0]
    print(f"    Min: {embed_per_token_max.min().item():.6f}")
    print(f"    Max: {embed_per_token_max.max().item():.6f}")
    print(f"    Mean: {embed_per_token_max.mean().item():.6f}")

    print("  Head changes per token (max):")
    head_per_token_max = head_diff.max(dim=1)[0]
    print(f"    Min: {head_per_token_max.min().item():.6f}")
    print(f"    Max: {head_per_token_max.max().item():.6f}")
    print(f"    Mean: {head_per_token_max.mean().item():.6f}")

    # Sample a few tokens to show detailed differences
    print("\n[Sample Token Details] (showing first 5 tokens)")
    for i in range(min(5, len(coord_ids))):
        token_id = coord_ids.start + i
        embed_change = embed_per_token_max[i].item()
        head_change = head_per_token_max[i].item()
        print(
            f"  Token ID {token_id}: embed_diff={embed_change:.6f}, head_diff={head_change:.6f}"
        )

    return {
        "embed_max_diff": embed_max_diff,
        "embed_mean_diff": embed_mean_diff,
        "head_max_diff": head_max_diff,
        "head_mean_diff": head_mean_diff,
        "tokens_changed_embed": tokens_changed,
        "tokens_changed_head": head_tokens_changed,
    }


def load_tokenizer_with_fallback(model_path: Path, base_model_path: Path = None):
    """Load tokenizer, trying merged model first, then base model if needed."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        print(f"  Loaded tokenizer from: {model_path}")
        return tokenizer
    except Exception as e:
        if base_model_path and base_model_path.exists():
            try:
                print(f"  Tokenizer not found in {model_path}, trying base model...")
                tokenizer = AutoTokenizer.from_pretrained(
                    str(base_model_path), trust_remote_code=True
                )
                print(f"  Loaded tokenizer from base model: {base_model_path}")
                return tokenizer
            except Exception as e2:
                print(f"  ERROR: Could not load tokenizer from base model either: {e2}")
        raise e


def find_coord_token_ids(tokenizer):
    """Find coordinate token IDs from tokenizer vocabulary."""
    coord_tokens = [f"<|coord_{i}|>" for i in range(1000)]
    found_coords = []
    coord_ids = []

    for i, token in enumerate(coord_tokens):
        if token in tokenizer.get_vocab():
            token_id = tokenizer.get_vocab()[token]
            found_coords.append((i, token_id))
            coord_ids.append(token_id)

    return found_coords, coord_ids


def verify_tokenizer_coord_tokens(tokenizer, coord_ids: range):
    """Verify that the tokenizer recognizes the coordinate tokens."""
    print("\n" + "=" * 80)
    print("TOKENIZER VERIFICATION")
    print("=" * 80)

    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Coordinate token ID range: {coord_ids.start} to {coord_ids.stop - 1}")

    if coord_ids.stop > vocab_size:
        print("⚠️  WARNING: Coordinate token IDs extend beyond vocab size!")
        return False

    # Check a few sample tokens
    print("\n[Sample Coordinate Tokens]")
    sample_ids = list(coord_ids)[:10]
    for token_id in sample_ids:
        try:
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            decoded = tokenizer.decode([token_id])
            print(f"  ID {token_id}: {token} -> '{decoded}'")
        except Exception as e:
            print(f"  ID {token_id}: ERROR - {e}")

    return True


def check_adapter_coord_offsets(
    adapter_path: Path, coord_ids: range, threshold: float = 1e-6
):
    """Check if coord_offset_adapter weights in adapter checkpoint were trained."""
    print("\n" + "=" * 80)
    print("ADAPTER CHECKPOINT VERIFICATION")
    print("=" * 80)

    adapter_file = adapter_path / "adapter_model.safetensors"
    if not adapter_file.exists():
        print(f"[ERROR] Adapter file not found: {adapter_file}")
        return False

    print(f"[Checking] {adapter_file}")

    embed_offset_key = None
    head_offset_key = None

    with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        # Find coord_offset keys
        for key in keys:
            if "coord_offset_adapter" in key:
                if "embed_offset" in key:
                    embed_offset_key = key
                elif "head_offset" in key:
                    head_offset_key = key

        if embed_offset_key is None or head_offset_key is None:
            print("[ERROR] Could not find coord_offset_adapter weights in adapter")
            coord_keys = [
                k for k in keys if "coord" in k.lower() or "offset" in k.lower()
            ]
            if coord_keys:
                print(f"  Found coord-related keys: {coord_keys}")
            return False

        print(f"[Found] {embed_offset_key}")
        print(f"[Found] {head_offset_key}")

        embed_offset = f.get_tensor(embed_offset_key)
        head_offset = f.get_tensor(head_offset_key)

        print(
            f"\n[Embed Offset] shape={embed_offset.shape}, dtype={embed_offset.dtype}"
        )
        embed_max = embed_offset.abs().max().item()
        embed_mean = embed_offset.abs().mean().item()
        embed_std = embed_offset.std().item()
        embed_nonzero = (embed_offset.abs() > threshold).sum().item()

        print(f"  Max abs value: {embed_max:.6f}")
        print(f"  Mean abs value: {embed_mean:.6f}")
        print(f"  Std: {embed_std:.6f}")
        print(
            f"  Non-zero elements (> {threshold}): {embed_nonzero}/{embed_offset.numel()}"
        )

        print(f"\n[Head Offset] shape={head_offset.shape}, dtype={head_offset.dtype}")
        head_max = head_offset.abs().max().item()
        head_mean = head_offset.abs().mean().item()
        head_std = head_offset.std().item()
        head_nonzero = (head_offset.abs() > threshold).sum().item()

        print(f"  Max abs value: {head_max:.6f}")
        print(f"  Mean abs value: {head_mean:.6f}")
        print(f"  Std: {head_std:.6f}")
        print(
            f"  Non-zero elements (> {threshold}): {head_nonzero}/{head_offset.numel()}"
        )

        # Check if offsets were trained
        embed_trained = embed_max > threshold
        head_trained = head_max > threshold

        print("\n[Training Status]")
        if embed_trained and head_trained:
            print("  ✓ SUCCESS: Coordinate offsets were trained!")
            print(
                f"    - Embed offset: {embed_nonzero}/{embed_offset.numel()} elements non-zero"
            )
            print(
                f"    - Head offset: {head_nonzero}/{head_offset.numel()} elements non-zero"
            )
            return True
        else:
            print(
                "  ✗ FAILURE: Coordinate offsets appear untrained (all zeros or near-zero)"
            )
            if not embed_trained:
                print("    - Embed offset is all zeros")
            if not head_trained:
                print("    - Head offset is all zeros")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify coordinate token training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: Verify coordinate tokens were trained
  python verify_coord_tokens.py \\
    --original model_cache/Qwen3-VL-8B-Instruct-coordexp \\
    --merged output/debug/coord_merged

  # With adapter checkpoint verification
  python verify_coord_tokens.py \\
    --original model_cache/Qwen3-VL-8B-Instruct-coordexp \\
    --merged output/debug/coord_merged \\
    --adapter output/debug/coord/checkpoint-15

  # Also check other layers (vision, LLM, aligner)
  python verify_coord_tokens.py \\
    --original model_cache/Qwen3-VL-8B-Instruct-coordexp \\
    --merged output/debug/coord_merged \\
    --check-layers

  # Use GPU for faster loading
  python verify_coord_tokens.py \\
    --original model_cache/Qwen3-VL-8B-Instruct-coordexp \\
    --merged output/debug/coord_merged \\
    --device cuda:0
        """,
    )
    parser.add_argument(
        "--original",
        type=str,
        default="model_cache/Qwen3-VL-8B-Instruct-coordexp",
        help="Path to original checkpoint",
    )
    parser.add_argument(
        "--merged",
        type=str,
        default="output/debug/coord_merged",
        help="Path to merged model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="model_cache/Qwen3-VL-8B-Instruct-coordexp",
        help="Base model path for tokenizer fallback (if merged model lacks tokenizer files)",
    )
    parser.add_argument(
        "--coord-start",
        type=int,
        default=None,
        help="Start of coordinate token ID range (auto-detect from tokenizer if not provided)",
    )
    parser.add_argument(
        "--coord-end",
        type=int,
        default=None,
        help="End of coordinate token ID range (auto-detect from tokenizer if not provided)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load models on",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Threshold for considering changes significant",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional: Path to adapter checkpoint to verify coord_offset weights directly",
    )
    parser.add_argument(
        "--check-layers",
        action="store_true",
        help="Also check if other layers (vision, LLM, aligner) were updated during training",
    )

    args = parser.parse_args()

    # Resolve paths relative to repo root
    original_path = REPO_ROOT / args.original
    merged_path = REPO_ROOT / args.merged
    base_model_path = REPO_ROOT / args.base_model if args.base_model else None

    if not original_path.exists():
        print(f"[ERROR] Original model not found: {original_path}")
        return 1

    if not merged_path.exists():
        print(f"[ERROR] Merged model not found: {merged_path}")
        return 1

    # Try to detect coordinate token IDs from tokenizer
    print("=" * 80)
    print("Loading tokenizer to detect coordinate tokens...")
    print("=" * 80)

    try:
        tokenizer = load_tokenizer_with_fallback(merged_path, base_model_path)
        found_coords, coord_ids_list = find_coord_token_ids(tokenizer)

        if coord_ids_list:
            coord_start = coord_ids_list[0]
            coord_end = coord_ids_list[-1]
            coord_ids = range(coord_start, coord_end + 1)
            print(
                f"\n[Detected] Coordinate token IDs: {coord_start} to {coord_end} (n={len(coord_ids)})"
            )
            print(
                f"  Found {len(found_coords)} coordinate tokens (0-{len(found_coords) - 1})"
            )
            if len(found_coords) > 500:
                print(
                    f"  Sample: <|coord_0|> = ID {found_coords[0][1]}, <|coord_500|> = ID {found_coords[500][1]}"
                )
        else:
            # Fall back to provided range or default
            if args.coord_start is not None and args.coord_end is not None:
                coord_start = args.coord_start
                coord_end = args.coord_end
            else:
                coord_start = 151670
                coord_end = 152669
            coord_ids = range(coord_start, coord_end + 1)
            print(
                f"\n[Using] Coordinate token IDs: {coord_start} to {coord_end} (n={len(coord_ids)})"
            )
            print(
                "  ⚠️  WARNING: Coordinate tokens not found in tokenizer, using provided/default range"
            )
            coord_ids_list = list(coord_ids)
    except Exception as e:
        print(f"[ERROR] Could not load tokenizer: {e}")
        if args.coord_start is not None and args.coord_end is not None:
            coord_start = args.coord_start
            coord_end = args.coord_end
        else:
            print("[ERROR] Cannot auto-detect coordinate tokens and no range provided")
            return 1
        coord_ids = range(coord_start, coord_end + 1)
        coord_ids_list = list(coord_ids)
        print(
            f"[Using] Coordinate token IDs: {coord_start} to {coord_end} (n={len(coord_ids)})"
        )

    # Optionally check adapter checkpoint directly
    if args.adapter:
        adapter_path = REPO_ROOT / args.adapter
        if not adapter_path.exists():
            print(f"[ERROR] Adapter checkpoint not found: {adapter_path}")
            return 1
        adapter_trained = check_adapter_coord_offsets(
            adapter_path, coord_ids, args.threshold
        )
        if not adapter_trained:
            print("\n[WARNING] Adapter checkpoint shows offsets were not trained.")
            print("  This may indicate a training configuration issue.")
            return 1
        print(
            "\n[INFO] Adapter checkpoint verification passed. Proceeding to model check..."
        )

    # Load original model weights
    orig_embed, orig_head = load_embedding_weights(original_path, args.device)
    if orig_embed is None or orig_head is None:
        return 1

    # Load merged model weights
    merged_embed, merged_head = load_embedding_weights(merged_path, args.device)
    if merged_embed is None or merged_head is None:
        return 1

    # Verify tokenizer if we have it
    if "tokenizer" in locals():
        verify_tokenizer_coord_tokens(tokenizer, coord_ids)

    # Extract coordinate token embeddings
    print("\n" + "=" * 80)
    print("EXTRACTING COORDINATE TOKEN EMBEDDINGS")
    print("=" * 80)

    orig_coord_embeds, orig_coord_head, _ = extract_coord_embeddings(
        orig_embed, orig_head, coord_ids
    )
    merged_coord_embeds, merged_coord_head, _ = extract_coord_embeddings(
        merged_embed, merged_head, coord_ids
    )

    # Compare
    stats = compare_embeddings(
        orig_coord_embeds,
        merged_coord_embeds,
        orig_coord_head,
        merged_coord_head,
        coord_ids,
        threshold=args.threshold,
    )

    # Optionally check other layers
    layers_updated = None
    if args.check_layers:
        layers_updated = check_other_layers(
            original_path, merged_path, args.device, args.threshold
        )

    # Final verdict
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    embed_trained = stats["embed_max_diff"] > args.threshold
    head_trained = stats["head_max_diff"] > args.threshold

    coord_success = embed_trained and head_trained

    if coord_success:
        print("✓ SUCCESS: Coordinate tokens (0-999) have been successfully trained!")
        print(
            f"  - Embeddings modified: {stats['tokens_changed_embed']}/{len(coord_ids)} tokens"
        )
        print(
            f"  - LM head modified: {stats['tokens_changed_head']}/{len(coord_ids)} tokens"
        )
    else:
        print("✗ FAILURE: Coordinate tokens do not appear to have been trained.")
        if not embed_trained:
            print("  - Embeddings unchanged")
        if not head_trained:
            print("  - LM head unchanged")

    if args.check_layers and layers_updated is not None:
        if layers_updated:
            print("\n✓ Other layers (vision, LLM, aligner) were updated.")
        else:
            print("\n✗ Other layers (vision, LLM, aligner) were not updated.")

    # Return success only if coordinate tokens were trained
    return 0 if coord_success else 1


if __name__ == "__main__":
    sys.exit(main())

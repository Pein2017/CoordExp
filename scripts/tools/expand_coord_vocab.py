"""One-time utility to expand Qwen3-VL vocab with coordinate tokens and save a new checkpoint.

Default source:
  <repo>/model_cache/models/Qwen/Qwen3-VL-2B-Instruct
Default output:
  <repo>/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp

Run inside the `ms` conda environment:
  python scripts/tools/expand_coord_vocab.py
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

TRANSFORMERS_RESIZE_SEED = 0
COORD_INIT_MEAN_RESIZE = "mean_resize"
COORD_INIT_NATURAL_ADJACENT = "natural_adjacent"


def build_coord_tokens(num_bins: int, include_wildcard: bool = True) -> List[str]:
    """
    Build coordinate tokens for Qwen3-VL norm1000 quantization.

    Qwen3-VL normalizes coordinates to [0, 1000] with round(). In practice we
    cap at 0..999 here per experiment needs; adjust --num-bins if you want 1000.
    """
    # Generate tokens from 0 to num_bins (inclusive)
    tokens = [f"<|coord_{i}|>" for i in range(0, num_bins + 1)]
    if include_wildcard:
        tokens = ["<|coord_*|>"] + tokens
    return tokens


def resize_token_embeddings_deterministically(
    model: Qwen3VLForConditionalGeneration,
    *,
    new_vocab_size: int,
    seed: int = TRANSFORMERS_RESIZE_SEED,
) -> None:
    """Preserve Transformers mean-resizing behavior while making it reproducible."""
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = None
    if torch.cuda.is_available():
        cuda_rng_states = torch.cuda.get_rng_state_all()

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.resize_token_embeddings(new_vocab_size, mean_resizing=True)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def iter_numeric_coord_tokens(coord_tokens: list[str]) -> list[str]:
    """Return only canonical numeric coord tokens, ordered by numeric bin."""
    numeric_tokens: list[tuple[int, str]] = []
    prefix = "<|coord_"
    suffix = "|>"
    for token in coord_tokens:
        if not token.startswith(prefix) or not token.endswith(suffix):
            continue
        raw_bin = token[len(prefix) : -len(suffix)]
        if not raw_bin.isdigit():
            continue
        numeric_tokens.append((int(raw_bin), token))
    numeric_tokens.sort(key=lambda item: item[0])
    return [token for _, token in numeric_tokens]


def build_coord_positional_features(
    num_coords: int,
    num_frequencies: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build smooth positional features for coord bins in their natural order."""
    if num_coords <= 0:
        raise ValueError("num_coords must be positive")
    if num_frequencies < 0:
        raise ValueError("num_frequencies must be non-negative")

    denominator = max(num_coords - 1, 1)
    positions = torch.arange(num_coords, device=device, dtype=dtype) / denominator
    features = [positions]
    for frequency_index in range(num_frequencies):
        frequency = 2**frequency_index
        phase = 2 * math.pi * frequency * positions
        features.append(torch.sin(phase))
        features.append(torch.cos(phase))
    return torch.stack(features, dim=1)


def _token_id_if_available(tokenizer: object, token: str, vocab_size: int) -> int | None:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    if vocab is not None and token not in vocab:
        return None
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(token_id, int):
        return None
    if token_id < 0 or token_id >= vocab_size:
        return None
    return token_id


def select_base_embedding(tokenizer: object, embedding_weight: torch.Tensor) -> torch.Tensor:
    """Use digit-token rows as the coord baseline, falling back to the full mean."""
    anchor_ids: list[int] = []
    for token in (str(i) for i in range(10)):
        token_id = _token_id_if_available(tokenizer, token, embedding_weight.shape[0])
        if token_id is not None:
            anchor_ids.append(token_id)

    if anchor_ids:
        anchor_index = torch.tensor(
            anchor_ids,
            device=embedding_weight.device,
            dtype=torch.long,
        )
        return embedding_weight.index_select(0, anchor_index).mean(dim=0)
    return embedding_weight.mean(dim=0)


def initialize_natural_adjacent_coord_rows(
    model: Qwen3VLForConditionalGeneration,
    tokenizer: object,
    coord_tokens: list[str],
    *,
    seed: int,
    scale: float,
    num_frequencies: int,
) -> None:
    """Overwrite numeric coord rows with a deterministic smooth positional prior."""
    if scale < 0:
        raise ValueError("scale must be non-negative")

    numeric_coord_tokens = iter_numeric_coord_tokens(coord_tokens)
    if not numeric_coord_tokens:
        return

    input_embeddings = model.get_input_embeddings()
    embedding_weight = input_embeddings.weight
    device = embedding_weight.device
    dtype = embedding_weight.dtype

    coord_token_ids: list[int] = []
    for token in numeric_coord_tokens:
        token_id = _token_id_if_available(tokenizer, token, embedding_weight.shape[0])
        if token_id is None:
            raise ValueError(f"numeric coord token {token!r} is missing from tokenizer")
        coord_token_ids.append(token_id)

    features = build_coord_positional_features(
        len(coord_token_ids),
        num_frequencies,
        device=device,
        dtype=dtype,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    projection = torch.randn(
        features.shape[1],
        embedding_weight.shape[1],
        device=device,
        dtype=dtype,
        generator=generator,
    ) / math.sqrt(features.shape[1])
    base_embedding = select_base_embedding(tokenizer, embedding_weight)
    initialized_rows = base_embedding.unsqueeze(0) + scale * features.matmul(projection)
    coord_index = torch.tensor(coord_token_ids, device=device, dtype=torch.long)

    with torch.no_grad():
        embedding_weight.index_copy_(0, coord_index, initialized_rows)


def _default_2b_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "model_cache" / "models" / "Qwen" / "Qwen3-VL-2B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand Qwen3-VL tokenizer with coordinate tokens."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=_default_2b_dir(),
        help="Path to the base checkpoint directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=_default_2b_dir().parent
        / (f"{_default_2b_dir().name}-coordexp"),
        help="Path to save the expanded checkpoint.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=999,
        help="Maximum bin value (generates tokens from coord_0 to coord_N, inclusive). "
        "Default 999 matches the current coord-token experiments; set 1000 if you want the upper-edge bin.",
    )
    parser.add_argument(
        "--no-wildcard",
        action="store_true",
        help="Disable adding <|coord_*|> wildcard token.",
    )
    parser.add_argument(
        "--coord-init",
        choices=(COORD_INIT_MEAN_RESIZE, COORD_INIT_NATURAL_ADJACENT),
        default=COORD_INIT_MEAN_RESIZE,
        help="Initialization strategy for numeric coord embedding rows after resize.",
    )
    parser.add_argument(
        "--coord-init-frequencies",
        type=int,
        default=8,
        help="Number of powers-of-two sinusoidal frequency bands for natural_adjacent init.",
    )
    parser.add_argument(
        "--coord-init-scale",
        type=float,
        default=0.02,
        help="Amplitude of the natural_adjacent offset added to the base embedding.",
    )
    parser.add_argument(
        "--coord-init-seed",
        type=int,
        default=TRANSFORMERS_RESIZE_SEED,
        help="Random seed for the natural_adjacent projection matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_wildcard = not args.no_wildcard

    coord_tokens = build_coord_tokens(args.num_bins, include_wildcard)
    if args.num_bins < 1000:
        print(f"[!] Warning: coord_1000 will NOT be added (num_bins={args.num_bins})")

    print(f"[+] Loading tokenizer from {args.src}")
    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    print(f"[+] Loading model from {args.src}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.src, trust_remote_code=True
    )

    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": coord_tokens},
        replace_additional_special_tokens=False,
    )
    print(f"[+] Added {added} tokens; new vocab size = {len(tokenizer)}")

    print("[+] Resizing model embeddings...")
    resize_token_embeddings_deterministically(model, new_vocab_size=len(tokenizer))
    if added > 0:
        print(
            f"[+] Resized with Transformers mean-resizing under fixed seed {TRANSFORMERS_RESIZE_SEED}."
        )
        if args.coord_init == COORD_INIT_NATURAL_ADJACENT:
            initialize_natural_adjacent_coord_rows(
                model,
                tokenizer,
                coord_tokens,
                seed=args.coord_init_seed,
                scale=args.coord_init_scale,
                num_frequencies=args.coord_init_frequencies,
            )
            print(
                "[+] Initialized numeric coord rows with natural_adjacent "
                f"prior (seed={args.coord_init_seed}, scale={args.coord_init_scale}, "
                f"frequencies={args.coord_init_frequencies})."
            )

    # Qwen3-VL default: tie-head (single shared lookup table for embedding + lm_head).
    # After resizing, force tie_word_embeddings and re-tie weights so new tokens
    # are consistent across embed_tokens and lm_head.
    if getattr(model.config, "tie_word_embeddings", None) is not True:
        print("[!] Forcing tie_word_embeddings=True (Qwen3-VL tie-head default).")
        model.config.tie_word_embeddings = True

    try:
        model.tie_weights()
    except Exception as e:
        raise RuntimeError(f"Failed to tie weights after resize: {e}") from e

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("Model has no output embeddings; cannot verify tie-head.")

    if output_embeddings.weight.data_ptr() != input_embeddings.weight.data_ptr():
        raise RuntimeError(
            "tie-head check failed: input embeddings and lm_head weights are not tied."
        )
    print("[+] Verified tie-head: embed_tokens.weight and lm_head.weight are tied.")

    args.dst.mkdir(parents=True, exist_ok=True)
    # Keep the expanded checkpoint self-contained for multimodal loaders (e.g. AutoProcessor).
    # Qwen3-VL relies on image/video preprocessor config files that are NOT written by
    # tokenizer.save_pretrained() / model.save_pretrained().
    extra_files = [
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.json",
        "README.md",
        "configuration.json",
    ]
    for name in extra_files:
        src_path = args.src / name
        dst_path = args.dst / name
        if src_path.exists() and not dst_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"[+] Copied {name} from src -> dst")

    print(f"[+] Saving tokenizer to {args.dst}")
    tokenizer.save_pretrained(args.dst)
    print(f"[+] Saving model to {args.dst}")
    model.save_pretrained(args.dst)

    tokens_path = args.dst / "coord_tokens.json"
    with tokens_path.open("w", encoding="utf-8") as f:
        json.dump(coord_tokens, f, ensure_ascii=True, indent=2)
    print(f"[+] Wrote token list to {tokens_path}")
    if args.coord_init == COORD_INIT_NATURAL_ADJACENT and added > 0:
        init_metadata_path = args.dst / "coord_init.json"
        init_metadata = {
            "schema_version": 1,
            "coord_init": COORD_INIT_NATURAL_ADJACENT,
            "num_numeric_coord_tokens": len(iter_numeric_coord_tokens(coord_tokens)),
            "num_frequencies": args.coord_init_frequencies,
            "scale": args.coord_init_scale,
            "seed": args.coord_init_seed,
        }
        with init_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(init_metadata, f, ensure_ascii=True, indent=2)
        print(f"[+] Wrote coord init metadata to {init_metadata_path}")
    print("[✓] Done. Point ms-swift configs to the new checkpoint.")


if __name__ == "__main__":
    main()

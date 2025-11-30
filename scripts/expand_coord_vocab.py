"""
One-time utility to expand Qwen3-VL vocab with coordinate tokens and save a new checkpoint.

Default source:
  /data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct
Default output:
  /data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct-coordexp

Run inside the `ms` conda environment:
  python scripts/expand_coord_vocab.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration


def build_coord_tokens(num_bins: int, include_wildcard: bool = True) -> List[str]:
    """
    Build coordinate tokens for Qwen3-VL norm1000 quantization.

    Qwen3-VL normalizes coordinates to [0, 1000] range using:
    bin = round(pixel / image_dim * 1000)

    This requires tokens for bins 0 through 1000 (inclusive), giving 1001 tokens.
    """
    # Generate tokens from 0 to num_bins (inclusive) to cover [0, 1000] range
    tokens = [f"<|coord_{i}|>" for i in range(0, num_bins + 1)]
    if include_wildcard:
        tokens = ["<|coord_*|>"] + tokens
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand Qwen3-VL tokenizer with coordinate tokens."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            "/data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct"
        ),
        help="Path to the base checkpoint directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(
            "/data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct-coordexp"
        ),
        help="Path to save the expanded checkpoint.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=1000,
        help="Maximum bin value (generates tokens from coord_0 to coord_N, inclusive). "
        "For Qwen3-VL norm1000, use 1000 to get bins [0, 1000] (1001 tokens).",
    )
    parser.add_argument(
        "--no-wildcard",
        action="store_true",
        help="Disable adding <|coord_*|> wildcard token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_wildcard = not args.no_wildcard

    coord_tokens = build_coord_tokens(args.num_bins, include_wildcard)

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
    model.resize_token_embeddings(len(tokenizer))

    args.dst.mkdir(parents=True, exist_ok=True)
    print(f"[+] Saving tokenizer to {args.dst}")
    tokenizer.save_pretrained(args.dst)
    print(f"[+] Saving model to {args.dst}")
    model.save_pretrained(args.dst)

    tokens_path = args.dst / "coord_tokens.json"
    with tokens_path.open("w", encoding="utf-8") as f:
        json.dump(coord_tokens, f, ensure_ascii=True, indent=2)
    print(f"[+] Wrote token list to {tokens_path}")
    print("[✓] Done. Point ms-swift configs to the new checkpoint.")


if __name__ == "__main__":
    main()

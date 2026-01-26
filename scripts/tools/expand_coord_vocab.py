"""
One-time utility to expand Qwen3-VL vocab with coordinate tokens and save a new checkpoint.

Default source:
  /data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct
Default output:
  /data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct-coordexp

Run inside the `ms` conda environment:
  python scripts/tools/expand_coord_vocab.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand Qwen3-VL tokenizer with coordinate tokens."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            "/data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct"
        ),
        help="Path to the base checkpoint directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(
            "/data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct-coordexp"
        ),
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
    model.resize_token_embeddings(len(tokenizer))

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
    print("[✓] Done. Point ms-swift configs to the new checkpoint.")


if __name__ == "__main__":
    main()

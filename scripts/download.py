#!/usr/bin/env python3
"""Simplified Script: Downloading Qwen3-VL Models Using ModelScope (Method 1 Only)

This script downloads the Qwen3-VL-2B model from ModelScope to a local cache
directory using the ms-swift framework.

Run with:
  python scripts/download.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    print("\n" + "=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)

    dependencies = {
        "swift": "ms-swift",
        "modelscope": "modelscope",
        "transformers": "transformers>=4.57",
        "torch": "torch",
    }

    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ All dependencies are installed!")
        return True


def download_models() -> str | None:
    """Download Qwen3-VL-2B model using Method 1: safe_snapshot_download()."""
    print("\n" + "=" * 70)
    print("METHOD 1: Using safe_snapshot_download()")
    print("=" * 70)

    try:
        from swift.llm import safe_snapshot_download

        # Set custom cache directory
        cache_dir = os.path.abspath("./model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = cache_dir

        print(f"\nCache directory set to: {cache_dir}")

        print("\n" + "-" * 70)
        print("Downloading Qwen3-VL-2B-Instruct...")
        print("-" * 70)
        model_dir_2b = safe_snapshot_download(
            "Qwen/Qwen3-VL-2B-Instruct",
            use_hf=False,  # Use ModelScope instead of HuggingFace
        )
        print("\n✓ 2B model successfully downloaded!")
        print(f"  Location: {model_dir_2b}")
        return str(model_dir_2b)

    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("  Install ms-swift: pip install ms-swift[framework]")
        return None


def expand_coord_vocab(model_dir: str, num_bins: int, include_wildcard: bool) -> str | None:
    """Expand a local checkpoint with <|coord_*> and <|coord_k|> tokens."""
    expand_script = Path(__file__).resolve().parent / "tools" / "expand_coord_vocab.py"
    expanded_dir = f"{model_dir}-coordexp"
    cmd = [
        sys.executable,
        str(expand_script),
        "--src",
        model_dir,
        "--dst",
        expanded_dir,
        "--num-bins",
        str(num_bins),
    ]
    if not include_wildcard:
        cmd.append("--no-wildcard")

    print(f"\nExpanding checkpoint with coord vocabulary:\n  {expanded_dir}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(f"\n✗ Expansion script not found: {expand_script}")
        return None
    except subprocess.CalledProcessError as exc:
        print(f"\n✗ Expansion failed with exit code {exc.returncode}")
        return None

    return expanded_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Qwen3-VL and optionally expand coord-token vocabulary."
    )
    parser.add_argument(
        "--expand-coord-vocab",
        action="store_true",
        help="Create a coord-token-expanded checkpoint (adds <|coord_*> and <|coord_k|> tokens).",
    )
    parser.add_argument(
        "--coord-num-bins",
        type=int,
        default=999,
        help="Maximum coord bin index for generated <|coord_k|> tokens (default: 999).",
    )
    parser.add_argument(
        "--no-coord-wildcard",
        action="store_true",
        help="Do not add <|coord_*|> wildcard token.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to download models."""
    args = parse_args()
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  Qwen3-VL Model Downloader (Method 1 Only)  ".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    # Check dependencies first
    if not check_dependencies():
        print("\n⚠ Some dependencies are missing. Please install them and try again.")
        return

    # Download models
    print("\n\nStarting download...\n")
    model_dir_2b = download_models()

    expanded_model_dir = None
    if model_dir_2b and args.expand_coord_vocab:
        expanded_model_dir = expand_coord_vocab(
            model_dir=model_dir_2b,
            num_bins=args.coord_num_bins,
            include_wildcard=not args.no_coord_wildcard,
        )

    final_model_dir = expanded_model_dir or model_dir_2b
    if final_model_dir:
        print("\n" + "=" * 70)
        print("DOWNLOAD COMPLETE")
        print("=" * 70)
        print("\n✓ Successfully downloaded Qwen3-VL-2B model!")
        if expanded_model_dir:
            print("✓ Also created coord-token-expanded model:")
            print(f"  2B CoordExp Model: {expanded_model_dir}")
        else:
            print(f"  2B Model: {final_model_dir}")

        print(f"\nCache Directory: {os.path.abspath('./model_cache')}")
        if expanded_model_dir:
            print("\nUse this checkpoint for stage-1 pretraining:")
            print(f"  {expanded_model_dir}")
        else:
            print("\nUse this checkpoint for inference / base model:")
            print(f"  {model_dir_2b}")

        print(
            "\nYou can now use ms-swift with this checkpoint path."
        )
    else:
        print("\n✗ Download failed!")


if __name__ == "__main__":
    main()

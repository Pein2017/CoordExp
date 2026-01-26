"""Visualization utility for CoordExp unified inference outputs.

This is a thin CLI wrapper around the unified pipeline visualizer.

Input contract: a unified artifact JSONL (default: gt_vs_pred.jsonl) where each
record contains embedded GT (`gt`) and predictions (`pred`) in pixel space.

Usage (inside repo root, ms env):
  PYTHONPATH=. conda run -n ms python vis_tools/vis_coordexp.py \
      --pred_jsonl output/infer/<run_name>/gt_vs_pred.jsonl \
      --save_dir output/infer/<run_name>/vis \
      --limit 20

Image resolution:
- If `ROOT_IMAGE_DIR` is set, images are resolved relative to it.
- Otherwise, images are resolved relative to the JSONL parent directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.infer.vis import render_vis_from_jsonl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CoordExp visualization (GT vs pred overlays)")
    ap.add_argument("--pred_jsonl", required=True, help="Path to gt_vs_pred.jsonl")
    ap.add_argument("--save_dir", required=True, help="Output directory for PNG overlays")
    ap.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max samples to render (0 = all)",
    )
    ap.add_argument(
        "--root_image_dir",
        default=None,
        help=(
            "Optional override for resolving relative image paths. "
            "Equivalent to setting ROOT_IMAGE_DIR."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.root_image_dir:
        os.environ["ROOT_IMAGE_DIR"] = str(args.root_image_dir)

    render_vis_from_jsonl(
        Path(args.pred_jsonl),
        out_dir=Path(args.save_dir),
        limit=int(args.limit),
    )


if __name__ == "__main__":
    main()

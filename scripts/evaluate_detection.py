#!/usr/bin/env python
"""
Offline detection evaluator entrypoint.

Usage (from repo root, ms env):
  python scripts/evaluate_detection.py --pred_jsonl <path> --out_dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.detection import EvalOptions, evaluate_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoordExp detection evaluator (COCO-style)."
    )
    parser.add_argument(
        "--pred_jsonl", required=True, type=Path, help="Predictions JSONL path."
    )
    parser.add_argument(
        "--out_dir",
        default=Path("eval_out"),
        type=Path,
        help="Output directory (overwrites).",
    )
    parser.add_argument(
        "--unknown-policy",
        choices=["bucket", "drop"],
        default="bucket",
        help="Bucket unknown desc into 'unknown' (default) or drop them.",
    )
    parser.add_argument(
        "--strict-parse",
        action="store_true",
        help="Abort on first parse/validation error.",
    )
    parser.add_argument(
        "--no-segm",
        action="store_false",
        dest="use_segm",
        default=True,
        help="Disable segmentation metrics/export.",
    )
    parser.add_argument(
        "--iou-thrs",
        type=float,
        nargs="+",
        default=None,
        help="IoU thresholds override (e.g., --iou-thrs 0.5 0.75). Defaults to COCO if unset.",
    )
    parser.add_argument(
        "--overlay", action="store_true", help="Render overlay samples (top FP/FN)."
    )
    parser.add_argument(
        "--overlay-k",
        type=int,
        default=12,
        help="Number of overlay samples when enabled.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="CPU workers for parsing/denorm (0=single).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = EvalOptions(
        unknown_policy=args.unknown_policy,
        strict_parse=bool(args.strict_parse),
        use_segm=bool(args.use_segm),
        iou_thrs=args.iou_thrs,
        output_dir=args.out_dir,
        overlay=bool(args.overlay),
        overlay_k=int(args.overlay_k),
        num_workers=int(args.num_workers),
    )

    summary = evaluate_and_save(args.pred_jsonl, options=options)
    print(json.dumps(summary["metrics"], indent=2))
    print("Counters:", json.dumps(summary.get("counters", {}), indent=2))


if __name__ == "__main__":
    main()

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
        description="CoordExp detection evaluator (COCO + F1-ish set matching)."
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
        "--metrics",
        choices=["coco", "f1ish", "both"],
        default="coco",
        help="Which metric suite to run: COCOeval, F1-ish (greedy set matching), or both.",
    )
    parser.add_argument(
        "--unknown-policy",
        choices=["bucket", "drop", "semantic"],
        default="semantic",
        help=(
            "How to handle predicted desc that are not an exact match to any GT desc: "
            "'bucket' -> map to 'unknown'; 'drop' -> discard; "
            "'semantic' -> map to nearest GT desc by embedding similarity."
        ),
    )
    parser.add_argument(
        "--semantic-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help=(
            "HF model id used for semantic desc matching: "
            "(1) unknown-policy semantic mapping in COCO mode, and "
            "(2) semantic-on-matched scoring in F1-ish mode."
        ),
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.6,
        help=(
            "Cosine similarity threshold for semantic matching (used by both COCO unknown-policy "
            "semantic mapping and F1-ish semantic scoring)."
        ),
    )
    parser.add_argument(
        "--semantic-fallback",
        choices=["bucket", "drop"],
        default="bucket",
        help="Fallback for semantic mapping misses (only for --unknown-policy semantic).",
    )
    parser.add_argument(
        "--semantic-device",
        default="auto",
        help="Device for semantic matcher: auto|cpu|cuda[:N].",
    )
    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=64,
        help="Batch size for semantic embedding encoding.",
    )
    parser.add_argument(
        "--f1ish-iou-thrs",
        type=float,
        nargs="+",
        default=[0.3, 0.5],
        help=(
            "IoU thresholds for F1-ish greedy matching (e.g., --f1ish-iou-thrs 0.3 0.5). "
            "When multiple are provided, 0.5 (if present) is treated as the primary threshold "
            "for `matches.jsonl` naming."
        ),
    )
    parser.add_argument(
        "--f1ish-pred-scope",
        choices=["annotated", "all"],
        default="annotated",
        help=(
            "Which predictions count for F1-ish FP: "
            "'annotated' ignores predictions whose desc is not semantically close to any GT desc "
            "in the image; 'all' counts all predictions (strict)."
        ),
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
        metrics=str(args.metrics),
        unknown_policy=args.unknown_policy,
        strict_parse=bool(args.strict_parse),
        use_segm=bool(args.use_segm),
        iou_thrs=args.iou_thrs,
        f1ish_iou_thrs=[float(x) for x in (args.f1ish_iou_thrs or [])],
        f1ish_pred_scope=str(args.f1ish_pred_scope),
        output_dir=args.out_dir,
        overlay=bool(args.overlay),
        overlay_k=int(args.overlay_k),
        num_workers=int(args.num_workers),
        semantic_model=str(args.semantic_model),
        semantic_threshold=float(args.semantic_threshold),
        semantic_fallback=str(args.semantic_fallback),
        semantic_device=str(args.semantic_device),
        semantic_batch_size=int(args.semantic_batch_size),
    )

    summary = evaluate_and_save(args.pred_jsonl, options=options)
    print(json.dumps(summary["metrics"], indent=2))
    print("Counters:", json.dumps(summary.get("counters", {}), indent=2))


if __name__ == "__main__":
    main()

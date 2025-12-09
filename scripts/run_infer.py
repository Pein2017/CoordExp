#!/usr/bin/env python
"""Unified inference CLI for CoordExp.

This script wraps ``src.infer.InferenceEngine`` and produces:
- ``pred.jsonl``: pixel-space geometries for both GT and predictions.
- ``pred.summary.json``: counters and error codes.

Usage (inside repo root, ms env):
  python scripts/run_infer.py \
      --gt_jsonl public_data/lvis/rescale_32_768_poly_20/val.coord.jsonl \
      --model_checkpoint output/ckpts/coord_lora \
      --mode coord \
      --out output/pred.jsonl \
      --temperature 0.01 --top_p 0.95 --max_new_tokens 1024 --repetition_penalty 1.05 --seed 42

Notes:
- ``--mode`` is required (no auto-detect).
- Generation config is flag-driven only; no external config files.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

from src.common.geometry import flatten_points, has_coord_tokens
from src.infer import GenerationConfig, InferenceConfig, InferenceEngine


def _detect_mode_from_gt(jsonl_path: str, *, sample_size: int = 128) -> Tuple[str, str]:
    """Heuristic to choose coord vs text mode based on GT.

    - coord if any coord tokens or values go beyond image size.
    - otherwise text.
    """
    has_tokens = False
    out_of_image = 0
    checked = 0

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if checked >= sample_size:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = eval(line) if line.startswith("{") else None  # noqa: S307
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                width = rec.get("width")
                height = rec.get("height")
                max_dim = None
                try:
                    max_dim = max(int(width), int(height))
                except Exception:
                    pass
                objs = rec.get("objects") or rec.get("gt") or []
                for obj in objs:
                    pts = flatten_points(
                        obj.get("bbox_2d")
                        or obj.get("poly")
                        or obj.get("line")
                        or obj.get("points")
                        or []
                    )
                    if pts is None or len(pts) == 0:
                        continue
                    if has_coord_tokens(pts):
                        has_tokens = True
                        break
                    if max_dim is not None and max(pts) > max_dim:
                        out_of_image += 1
                        break
                checked += 1
    except FileNotFoundError:
        return "coord", "file_not_found"

    if has_tokens:
        return "coord", "coord_tokens_found"
    if out_of_image > 0:
        return "coord", f"points_exceed_image ({out_of_image})"
    return "text", "within_image_bounds"


def parse_args() -> tuple[InferenceConfig, GenerationConfig, Optional[str]]:
    ap = argparse.ArgumentParser(description="Unified inference for CoordExp")
    ap.add_argument("--gt_jsonl", required=True, help="Path to ground-truth JSONL")
    ap.add_argument("--model_checkpoint", required=True, help="Checkpoint path")
    ap.add_argument(
        "--mode",
        required=True,
        choices=["coord", "text", "auto"],
        help="Model/GT mode (coord-token vs pixel GT), or auto-detect",
    )
    ap.add_argument("--out", default="pred.jsonl", help="Output predictions JSONL")
    ap.add_argument(
        "--summary",
        default=None,
        help="Optional summary path (defaults to pred.jsonl sibling)",
    )
    ap.add_argument("--device", default="cuda:0", help="Device for inference")
    ap.add_argument(
        "--limit", type=int, default=0, help="Max samples to process (0 = all)"
    )
    ap.add_argument(
        "--detect-samples",
        type=int,
        default=128,
        help="Samples to scan for auto mode detection (runs even if limit is small)",
    )
    ap.add_argument(
        "--force-mode",
        action="store_true",
        help="Keep user-specified mode even if detector disagrees",
    )
    ap.add_argument(
        "--pred-coord-mode",
        choices=["auto", "pixel", "norm1000"],
        default="auto",
        help="Override how prediction coords are interpreted before scaling",
    )

    # Generation flags (CLI only)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=None)

    args = ap.parse_args()

    detected_mode, reason = _detect_mode_from_gt(
        args.gt_jsonl, sample_size=max(args.detect_samples, args.limit or 0)
    )
    resolved_mode = args.mode
    note: Optional[str] = None

    if args.mode == "auto":
        resolved_mode = detected_mode
        note = f"auto-detected mode={resolved_mode} ({reason})"
    else:
        if detected_mode != args.mode and not args.force_mode:
            note = (
                f"detector suggested mode={detected_mode} ({reason}); "
                f"overriding user choice '{args.mode}'"
            )
            resolved_mode = detected_mode
        elif detected_mode != args.mode:
            note = (
                f"detector suggested mode={detected_mode} ({reason}); "
                f"kept user mode '{args.mode}' due to --force-mode"
            )

    gen_cfg = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    inf_cfg = InferenceConfig(
        gt_jsonl=args.gt_jsonl,
        model_checkpoint=args.model_checkpoint,
        mode=resolved_mode,
        pred_coord_mode=args.pred_coord_mode,
        out_path=args.out,
        summary_path=args.summary,
        device=args.device,
        limit=args.limit,
    )
    return inf_cfg, gen_cfg, note


def main() -> None:
    inf_cfg, gen_cfg, note = parse_args()
    if note:
        print(note)
    engine = InferenceEngine(inf_cfg, gen_cfg)
    out_path, summary_path = engine.infer()
    print(f"Wrote predictions to {out_path} and summary to {summary_path}")


if __name__ == "__main__":
    main()

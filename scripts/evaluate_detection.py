#!/usr/bin/env python
"""Offline detection evaluator entrypoint (YAML-first).

YAML usage:
  python scripts/evaluate_detection.py --config configs/eval/detection.yaml

Legacy usage (still supported during transition):
  python scripts/evaluate_detection.py --pred_jsonl <path> --out_dir <dir>

In the unified pipeline workflow, the evaluator consumes a single artifact
JSONL that embeds both GT and predictions per sample (e.g., gt_vs_pred.jsonl).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from src.eval.detection import EvalOptions, evaluate_and_save
from src.utils import get_logger

logger = get_logger(__name__)


def _warn_deprecated_option(key: str, source: str, value: Any) -> None:
    if value is None:
        return
    logger.warning(
        "Ignoring %s=%s from %s; description matching always uses sentence-transformers/all-MiniLM-L6-v2 and drops unmatched predictions.",
        key,
        value,
        source,
    )


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "YAML config requires PyYAML (import yaml). Install it in the ms env."
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("eval config must be a YAML mapping")
    return data


def _get(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    return cfg[key] if key in cfg else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoordExp detection evaluator (COCO + F1-ish set matching)."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config")

    # Legacy flags / overrides (CLI wins over YAML when provided)
    parser.add_argument("--pred_jsonl", type=Path, default=None, help="Artifact JSONL")
    parser.add_argument(
        "--out_dir",
        default=None,
        type=Path,
        help="Output directory (overwrites).",
    )
    parser.add_argument(
        "--metrics",
        choices=["coco", "f1ish", "both"],
        default=None,
        help="Which metric suite to run.",
    )
    parser.add_argument(
        "--unknown-policy",
        choices=["bucket", "drop", "semantic"],
        default=None,
        help="How to handle unknown desc.",
    )

    parser.add_argument(
        "--semantic-model",
        default=None,
        help="HF model id used for semantic desc matching.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold for semantic matching.",
    )
    parser.add_argument(
        "--semantic-fallback",
        choices=["bucket", "drop"],
        default=None,
        help="Fallback for semantic mapping misses (only for unknown-policy=semantic).",
    )
    parser.add_argument(
        "--semantic-device",
        default=None,
        help="Device for semantic matcher: auto|cpu|cuda[:N].",
    )
    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=None,
        help="Batch size for semantic embedding encoding.",
    )

    parser.add_argument(
        "--f1ish-iou-thrs",
        type=float,
        nargs="+",
        default=None,
        help="IoU thresholds for F1-ish greedy matching.",
    )
    parser.add_argument(
        "--f1ish-pred-scope",
        choices=["annotated", "all"],
        default=None,
        help="Which predictions count for F1-ish FP.",
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
        default=None,
        help="Disable segmentation metrics/export.",
    )
    parser.add_argument(
        "--iou-thrs",
        type=float,
        nargs="+",
        default=None,
        help="IoU thresholds override (defaults to COCO if unset).",
    )
    parser.add_argument(
        "--overlay", action="store_true", help="Render overlay samples (top FP/FN)."
    )
    parser.add_argument(
        "--overlay-k",
        type=int,
        default=None,
        help="Number of overlay samples when enabled.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="CPU workers for parsing/denorm (0=single).",
    )
    return parser.parse_args()


def _resolve_from_yaml(ycfg: Mapping[str, Any], args: argparse.Namespace) -> tuple[Path, EvalOptions]:
    pred_jsonl = args.pred_jsonl or Path(str(_get(ycfg, "pred_jsonl", "")))
    out_dir = args.out_dir or Path(str(_get(ycfg, "out_dir", "eval_out")))

    if not str(pred_jsonl):
        raise ValueError("pred_jsonl must be set (either in YAML or via --pred_jsonl)")

    metrics = str(args.metrics or _get(ycfg, "metrics", "both"))

    strict_parse = bool(args.strict_parse) if args.strict_parse else bool(_get(ycfg, "strict_parse", False))

    use_segm = (
        bool(args.use_segm)
        if args.use_segm is not None
        else bool(_get(ycfg, "use_segm", True))
    )

    _warn_deprecated_option("--unknown-policy", "CLI flag", args.unknown_policy)
    _warn_deprecated_option("--semantic-fallback", "CLI flag", args.semantic_fallback)
    yaml_unknown_policy = _get(ycfg, "unknown_policy", None)
    yaml_semantic_fallback = _get(ycfg, "semantic_fallback", None)
    config_src = f"YAML config '{args.config}'"
    _warn_deprecated_option("unknown_policy", config_src, yaml_unknown_policy)
    _warn_deprecated_option("semantic_fallback", config_src, yaml_semantic_fallback)

    options = EvalOptions(
        metrics=metrics,
        strict_parse=strict_parse,
        use_segm=use_segm,
        iou_thrs=args.iou_thrs or _get(ycfg, "iou_thrs", None),
        f1ish_iou_thrs=[
            float(x)
            for x in (
                (args.f1ish_iou_thrs if args.f1ish_iou_thrs is not None else _get(ycfg, "f1ish_iou_thrs", [0.3, 0.5]))
                or []
            )
        ],
        f1ish_pred_scope=str(args.f1ish_pred_scope or _get(ycfg, "f1ish_pred_scope", "annotated")),
        output_dir=out_dir,
        overlay=bool(args.overlay) if args.overlay else bool(_get(ycfg, "overlay", False)),
        overlay_k=int(args.overlay_k or _get(ycfg, "overlay_k", 12)),
        num_workers=int(args.num_workers or _get(ycfg, "num_workers", 0)),
        semantic_model=str(args.semantic_model or _get(ycfg, "semantic_model", "sentence-transformers/all-MiniLM-L6-v2")),
        semantic_threshold=float(args.semantic_threshold or _get(ycfg, "semantic_threshold", 0.6)),
        semantic_device=str(args.semantic_device or _get(ycfg, "semantic_device", "auto")),
        semantic_batch_size=int(args.semantic_batch_size or _get(ycfg, "semantic_batch_size", 64)),
    )

    return pred_jsonl, options


def _resolve_legacy(args: argparse.Namespace) -> tuple[Path, EvalOptions]:
    if args.pred_jsonl is None:
        raise ValueError("--pred_jsonl is required when --config is not provided")

    out_dir = args.out_dir or Path("eval_out")

    _warn_deprecated_option("--unknown-policy", "CLI flag", args.unknown_policy)
    _warn_deprecated_option("--semantic-fallback", "CLI flag", args.semantic_fallback)

    options = EvalOptions(
        metrics=str(args.metrics or "f1ish"),
        strict_parse=bool(args.strict_parse),
        use_segm=bool(args.use_segm) if args.use_segm is not None else True,
        iou_thrs=args.iou_thrs,
        f1ish_iou_thrs=[float(x) for x in (args.f1ish_iou_thrs or [0.3, 0.5])],
        f1ish_pred_scope=str(args.f1ish_pred_scope or "annotated"),
        output_dir=out_dir,
        overlay=bool(args.overlay),
        overlay_k=int(args.overlay_k or 12),
        num_workers=int(args.num_workers or 0),
        semantic_model=str(args.semantic_model or "sentence-transformers/all-MiniLM-L6-v2"),
        semantic_threshold=float(args.semantic_threshold or 0.6),
        semantic_device=str(args.semantic_device or "auto"),
        semantic_batch_size=int(args.semantic_batch_size or 64),
    )

    return args.pred_jsonl, options


def main() -> None:
    args = parse_args()

    if args.config is not None:
        ycfg = _load_yaml(args.config)
        pred_jsonl, options = _resolve_from_yaml(ycfg, args)
        print("Resolved eval config:")
        print(
            json.dumps(
                {
                    "pred_jsonl": str(pred_jsonl),
                    "out_dir": str(options.output_dir),
                    "metrics": options.metrics,
                    "use_segm": options.use_segm,
                    "overlay": options.overlay,
                    "overlay_k": options.overlay_k,
                },
                indent=2,
            )
        )
    else:
        pred_jsonl, options = _resolve_legacy(args)

    summary = evaluate_and_save(pred_jsonl, options=options)
    print(json.dumps(summary.get("metrics", {}), indent=2))
    print("Counters:", json.dumps(summary.get("counters", {}), indent=2))


if __name__ == "__main__":
    main()

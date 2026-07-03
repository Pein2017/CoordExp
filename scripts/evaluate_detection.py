#!/usr/bin/env python
"""Evaluate CoordExp-Swift scored detection artifacts with COCO bbox metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.errors import CoordExpError
from src.eval.detection_consumer import SCORED_NAME, evaluate_scored_detection_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a CoordExp-Swift inference artifact directory containing "
            "gt_vs_pred.jsonl, gt_vs_pred_scored.jsonl, and the scored provenance sidecar."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Inference artifact directory to evaluate.",
    )
    parser.add_argument(
        "--pred-jsonl",
        "--pred_jsonl",
        dest="pred_jsonl",
        type=Path,
        default=None,
        help=(
            "Compatibility alias for a path ending in gt_vs_pred_scored.jsonl; "
            "the parent directory is evaluated."
        ),
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        required=True,
        help="Output directory for metrics.json and COCO conversion artifacts.",
    )
    parser.add_argument(
        "--metrics-name",
        default="metrics.json",
        help="Metric artifact filename to write inside --out-dir.",
    )
    return parser.parse_args()


def _resolve_artifact_dir(args: argparse.Namespace) -> Path:
    if args.artifact_dir is not None and args.pred_jsonl is not None:
        raise ValueError("provide only one of --artifact-dir or --pred-jsonl")
    if args.artifact_dir is not None:
        return args.artifact_dir
    if args.pred_jsonl is not None:
        if args.pred_jsonl.name != SCORED_NAME:
            raise ValueError(
                f"--pred-jsonl must point to {SCORED_NAME}, got {args.pred_jsonl.name!r}"
            )
        return args.pred_jsonl.parent
    raise ValueError("one of --artifact-dir or --pred-jsonl is required")


def main() -> None:
    try:
        args = parse_args()
        result = evaluate_scored_detection_artifacts(
            artifact_dir=_resolve_artifact_dir(args),
            output_dir=args.out_dir,
            metrics_name=str(args.metrics_name),
        )
    except CoordExpError as exc:
        print(
            json.dumps(
                {
                    "code": exc.code,
                    "message": exc.message,
                    "context": exc.context,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "code": "eval_detection.cli_error",
                    "message": str(exc),
                    "context": {},
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from None

    print(f"metrics: {result.metrics_path}")
    print(
        json.dumps(
            result.metrics,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

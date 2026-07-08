#!/usr/bin/env python
"""Render lightweight CoordExp-Swift detection visualizations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.errors import CoordExpError
from src.vis import render_gt_vs_prediction, render_prediction_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gt_parser = subparsers.add_parser("gt-vs-pred")
    gt_parser.add_argument("--run-dir", required=True, type=Path)
    gt_parser.add_argument("--out-dir", required=True, type=Path)
    _add_common(gt_parser)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left-run-dir", required=True, type=Path)
    compare_parser.add_argument("--right-run-dir", required=True, type=Path)
    compare_parser.add_argument("--out-dir", required=True, type=Path)
    compare_parser.add_argument("--left-label", default=None)
    compare_parser.add_argument("--right-label", default=None)
    _add_common(compare_parser)
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        if args.command == "gt-vs-pred":
            result = render_gt_vs_prediction(
                args.run_dir,
                args.out_dir,
                row_ids=args.row_id,
                limit=args.limit,
                duplicate_iou_threshold=args.duplicate_iou_threshold,
            )
        elif args.command == "compare":
            result = render_prediction_comparison(
                args.left_run_dir,
                args.right_run_dir,
                args.out_dir,
                left_label=args.left_label,
                right_label=args.right_label,
                row_ids=args.row_id,
                limit=args.limit,
                duplicate_iou_threshold=args.duplicate_iou_threshold,
            )
        else:
            raise ValueError(f"unsupported visualization command: {args.command}")
    except CoordExpError as exc:
        print(
            json.dumps(
                {"code": exc.code, "message": exc.message, "context": exc.context},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except ValueError as exc:
        print(
            json.dumps(
                {"code": "vis.cli_error", "message": str(exc), "context": {}},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from None

    print(f"manifest: {result.manifest_path}")
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "image_count": len(result.image_paths),
                "images": [str(path) for path in result.image_paths],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--row-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.30)


if __name__ == "__main__":
    main()

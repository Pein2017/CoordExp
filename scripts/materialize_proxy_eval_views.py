#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.proxy_views import (
    DEFAULT_METADATA_NAMESPACE,
    materialize_proxy_eval_views,
    supported_proxy_views,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize benchmark-aligned GT views from a COCO+LVIS-proxy eval artifact."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Scored or unscored eval artifact with inline GT metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the filtered eval-view JSONLs.",
    )
    parser.add_argument(
        "--metadata-namespace",
        type=str,
        default=DEFAULT_METADATA_NAMESPACE,
        help="Proxy metadata namespace on each record.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=list(supported_proxy_views()),
        choices=list(supported_proxy_views()),
        help="Subset of eval views to materialize.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize_proxy_eval_views(
        args.input_jsonl,
        output_dir=args.output_dir,
        views=args.views,
        metadata_namespace=args.metadata_namespace,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


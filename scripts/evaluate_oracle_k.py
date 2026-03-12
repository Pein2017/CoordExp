#!/usr/bin/env python
"""Oracle-K repeated-sampling evaluator entrypoint (YAML-first)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.oracle_k import run_oracle_k_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoordExp Oracle-K repeated-sampling evaluator."
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_oracle_k_from_config(args.config)
    print("Resolved Oracle-K summary:")
    print(
        json.dumps(
            {
                "oracle_run_count": summary.get("oracle_run_count"),
                "primary_iou_thr": summary.get("primary_iou_thr"),
                "out_dir": summary.get("out_dir"),
                "primary_recovery": summary.get("primary_recovery", {}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

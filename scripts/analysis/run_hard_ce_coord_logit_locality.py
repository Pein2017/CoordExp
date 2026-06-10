#!/usr/bin/env python
"""Run hard-CE coordinate-logit and token-embedding locality stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.hard_ce_coord_logit_locality import run_study  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAML config for the hard-CE coordinate locality study.",
    )
    parser.add_argument(
        "--stages",
        required=True,
        help=(
            "Comma-separated stages: embeddings,teacher_forced,self_prefix,"
            "x1_basin_attribution,plots,report."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit override for logit extraction stages.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Optional Lane-C shard index for x1_basin_attribution.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Optional Lane-C shard count for x1_basin_attribution.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build a CPU-only Lane-C shard plan without loading the model.",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge Lane-C shard outputs without loading the model.",
    )
    return parser.parse_args()


def main() -> int:
    """Run requested study stages and print a compact JSON summary."""

    args = _parse_args()
    stages = tuple(stage.strip() for stage in str(args.stages).split(",") if stage.strip())
    result = run_study(
        config_path=args.config,
        stages=stages,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        dry_run=bool(args.dry_run),
        merge_shards=bool(args.merge_shards),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

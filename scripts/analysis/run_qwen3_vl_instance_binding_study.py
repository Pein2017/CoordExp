#!/usr/bin/env python
"""Run stages for the Qwen3-VL coord-token instance-binding mechanism study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.qwen3_vl_instance_binding import run_study_stage  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        help="Single stage or comma-separated stage list, e.g. audit,select_cases.",
    )
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summaries = []
    for raw_stage in str(args.stage).split(","):
        stage = raw_stage.strip()
        if not stage:
            continue
        summaries.append(
            run_study_stage(
                config_path=args.config,
                stage=stage,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        )
    print(json.dumps({"stages": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

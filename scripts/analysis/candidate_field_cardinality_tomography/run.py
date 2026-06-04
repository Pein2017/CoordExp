#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.candidate_field_cardinality_tomography.runner import run_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run candidate-field cardinality tomography stages.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stages", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    args = parser.parse_args()
    del args.num_shards
    stages = None if args.stages is None else tuple(item for item in args.stages.split(",") if item)
    result = run_from_config(
        args.config,
        stages=stages,
        dry_run=args.dry_run,
        allow_overwrite=args.allow_overwrite,
        shard_id=args.shard_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

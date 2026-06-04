#!/usr/bin/env python
"""CLI wrapper for the A3.2 sorted-vs-random no-newline phenotype runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.sorted_random_no_newline_phenotype.runner import run_stages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or dry-run A3.2 sorted-vs-random no-newline phenotype stages.",
    )
    parser.add_argument("--config", required=True, help="Path to the A3.2 YAML config.")
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stage list. Defaults to every A3.2 stage.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print a JSON-safe plan.")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting materialized artifacts for reruns.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard id for sharded GPU-facing stages.",
    )
    parser.add_argument(
        "--launch-context",
        action="store_true",
        help="Internal launcher guard for GPU-facing orchestration stages.",
    )
    args = parser.parse_args(argv)

    try:
        result = run_stages(
            args.config,
            stages=args.stages,
            dry_run=args.dry_run,
            allow_overwrite=args.allow_overwrite,
            shard_id=args.shard_id,
            launch_context=args.launch_context,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Run autoregressive attention evidence-routing analysis stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.autoreg_attention_evidence_routing import (  # noqa: E402
    ATTENTION_STAGES,
    build_attention_dry_run_plan,
    load_attention_config,
    materialize_attention_atlas_shard,
    materialize_attention_feasibility_shard,
    materialize_attention_select_cases_shard,
    merge_attention_shards,
    write_attention_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stages", required=True)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    return parser


def _stages(raw: str, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    stages = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not stages:
        parser.error("--stages must not be empty")
    unknown = [stage for stage in stages if stage not in ATTENTION_STAGES]
    if unknown:
        parser.error(f"unknown attention stage(s): {', '.join(unknown)}")
    return stages


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    stages = _stages(args.stages, parser)
    try:
        config = load_attention_config(args.config)
        if args.dry_run:
            payload = build_attention_dry_run_plan(
                config,
                stages=stages,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
        if args.merge_shards:
            if args.num_shards is None:
                parser.error("--merge-shards requires --num-shards")
            summary = merge_attention_shards(
                config.paths.artifact_root,
                expected_shards=args.num_shards,
            )
            if "report" in stages:
                write_attention_report(config.paths.artifact_root)
            print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
            return 0
        if "select_cases" in stages:
            if args.shard_index is None or args.num_shards is None:
                parser.error("select_cases requires --shard-index and --num-shards")
            payload = materialize_attention_select_cases_shard(
                config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
        if "feasibility" in stages:
            if args.shard_index is None or args.num_shards is None:
                parser.error("feasibility requires --shard-index and --num-shards")
            payload = materialize_attention_feasibility_shard(
                config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
        if "attention_atlas" in stages:
            if args.shard_index is None or args.num_shards is None:
                parser.error("attention_atlas requires --shard-index and --num-shards")
            payload = materialize_attention_atlas_shard(
                config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
        if stages == ("report",):
            report = write_attention_report(config.paths.artifact_root)
            print(json.dumps({"report": str(report)}, sort_keys=True, separators=(",", ":")))
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    parser.error(f"unsupported attention stage combination: {','.join(stages)}")


if __name__ == "__main__":
    raise SystemExit(main())

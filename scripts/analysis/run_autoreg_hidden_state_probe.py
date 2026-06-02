#!/usr/bin/env python
"""Run Lane-D autoregressive hidden-state probe stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.autoreg_hidden_state_probe import (  # noqa: E402
    LANE_D_STAGES,
    build_lane_d_dry_run_plan,
    lane_d_config_expected_merge_metadata,
    load_lane_d_config,
    materialize_lane_d_hidden_states_shard,
    materialize_lane_d_select_cases_shard,
    merge_lane_d_shards,
    normalize_lane_d_shard,
    write_lane_d_shards_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAML config for the Lane-D autoregressive hidden-state probe.",
    )
    parser.add_argument(
        "--stages",
        required=True,
        help=(
            "Comma-separated stages: "
            "select_cases,position_inventory,hidden_states,patching,merge,report."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Optional Lane-D shard index.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Optional Lane-D shard count.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build a CPU-only Lane-D plan without importing or loading model code.",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Plan or run Lane-D shard merge.",
    )
    return parser


def _normalize_stages(raw_stages: str, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    stages = tuple(stage.strip() for stage in str(raw_stages).split(",") if stage.strip())
    if not stages:
        parser.error("--stages must include at least one Lane-D stage")
    unknown = [stage for stage in stages if stage not in LANE_D_STAGES]
    if unknown:
        parser.error(f"unknown Lane-D stage(s): {', '.join(unknown)}")
    return stages


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    stages = _normalize_stages(args.stages, parser)
    cpu_select_case_stages = {"select_cases", "position_inventory"}

    try:
        config = load_lane_d_config(args.config)
        if args.merge_shards and not args.dry_run:
            if "patching" in stages or "report" in stages:
                parser.error("Lane-D patching/report stages are not implemented")
            if args.num_shards is None:
                parser.error("--merge-shards requires --num-shards")
            summary = merge_lane_d_shards(
                config.paths.artifact_root,
                expected_shards=args.num_shards,
                expected_metadata=lane_d_config_expected_merge_metadata(config),
            )
            print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
            return 0
        if args.dry_run:
            plan = build_lane_d_dry_run_plan(
                config,
                stages=stages,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                merge_shards=bool(args.merge_shards),
            )
            print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
            return 0
        if "patching" in stages or "report" in stages:
            parser.error("Lane-D patching/report stages are not implemented")
        if "hidden_states" in stages:
            if args.shard_index is None or args.num_shards is None:
                parser.error(
                    "non-dry-run hidden_states requires "
                    "--shard-index and --num-shards"
                )
            normalize_lane_d_shard(
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            payload: dict[str, object] = {}
            if "select_cases" in stages:
                payload["manifest"] = write_lane_d_shards_manifest(
                    config,
                    args.num_shards,
                )
                payload["select_cases_summary"] = (
                    materialize_lane_d_select_cases_shard(
                        config,
                        shard_index=args.shard_index,
                        num_shards=args.num_shards,
                    )
                )
            payload["hidden_states_summary"] = materialize_lane_d_hidden_states_shard(
                config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
        if (
            "select_cases" in stages
            and set(stages).issubset(cpu_select_case_stages)
        ):
            if args.shard_index is None or args.num_shards is None:
                parser.error(
                    "non-dry-run select_cases requires "
                    "--shard-index and --num-shards"
                )
            normalize_lane_d_shard(
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            manifest = write_lane_d_shards_manifest(config, args.num_shards)
            shard_summary = materialize_lane_d_select_cases_shard(
                config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            print(
                json.dumps(
                    {"manifest": manifest, "shard_summary": shard_summary},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    parser.error(
        "Lane-D non-dry-run execution is not implemented in Task 2; "
        "use --dry-run to inspect the CPU-only plan."
    )


if __name__ == "__main__":
    raise SystemExit(main())

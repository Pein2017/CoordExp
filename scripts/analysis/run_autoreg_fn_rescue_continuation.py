#!/usr/bin/env python
"""Run FN-rescue continuation analysis stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.autoreg_fn_rescue_continuation import (  # noqa: E402
    FN_RESCUE_STAGES,
    build_fn_rescue_dry_run_plan,
    load_fn_rescue_config,
    materialize_fn_rescue_attention_replay_shard,
    materialize_fn_rescue_decode_shard,
    materialize_fn_rescue_feasibility_shard,
    materialize_fn_rescue_select_cases_shard,
    merge_fn_rescue_shards,
    write_fn_rescue_gallery,
    write_fn_rescue_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stages", required=True)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _stages(raw: str, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    stages = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not stages:
        parser.error("--stages must not be empty")
    unknown = [stage for stage in stages if stage not in FN_RESCUE_STAGES]
    if unknown:
        parser.error(f"unknown FN-rescue stage(s): {', '.join(unknown)}")
    return stages


def _require_shard_args(stage_name: str, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.shard_index is None or args.num_shards is None:
        parser.error(f"{stage_name} requires --shard-index and --num-shards")


def _print_json(payload: object) -> None:
    print(json.dumps(payload, separators=(",", ":")))


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    stages = _stages(args.stages, parser)
    merge_only_stages = {"merge", "report", "gallery"}
    if args.merge_shards:
        invalid_merge_stages = [stage for stage in stages if stage not in merge_only_stages]
        if invalid_merge_stages:
            parser.error(
                "--merge-shards requires stages drawn from merge,report,gallery; "
                f"got {','.join(invalid_merge_stages)}"
            )
        if "merge" not in stages:
            parser.error("--merge-shards requires an explicit merge stage in --stages")
    shard_stage_map = {
        "select_cases": materialize_fn_rescue_select_cases_shard,
        "feasibility": materialize_fn_rescue_feasibility_shard,
        "rescue_decode": materialize_fn_rescue_decode_shard,
        "attention_replay": materialize_fn_rescue_attention_replay_shard,
    }
    try:
        config = load_fn_rescue_config(args.config)
        if args.dry_run:
            _print_json(
                build_fn_rescue_dry_run_plan(
                    config,
                    stages=stages,
                    shard_index=args.shard_index,
                    num_shards=args.num_shards,
                )
            )
            return 0
        if args.merge_shards:
            if args.num_shards is None:
                parser.error("--merge-shards requires --num-shards")
            payload: dict[str, object] = {}
            if "merge" in stages:
                payload["merge"] = merge_fn_rescue_shards(
                    config.paths.artifact_root,
                    expected_shards=args.num_shards,
                )
            if "report" in stages:
                payload["report"] = str(write_fn_rescue_report(config.paths.artifact_root))
            if "gallery" in stages:
                payload["gallery"] = write_fn_rescue_gallery(config.paths.artifact_root)
            _print_json(payload)
            return 0
        shard_stages = [stage for stage in stages if stage in shard_stage_map]
        if shard_stages:
            payload: dict[str, object] = {}
            for stage_name in stages:
                if stage_name not in shard_stage_map:
                    continue
                _require_shard_args(stage_name, args, parser)
                payload[stage_name] = shard_stage_map[stage_name](
                    config,
                    shard_index=args.shard_index,
                    num_shards=args.num_shards,
                )
            _print_json(payload)
            return 0
        if set(stages).issubset({"report", "gallery"}):
            payload: dict[str, object] = {}
            if "report" in stages:
                payload["report"] = str(write_fn_rescue_report(config.paths.artifact_root))
            if "gallery" in stages:
                payload["gallery"] = write_fn_rescue_gallery(config.paths.artifact_root)
            _print_json(payload)
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    parser.error(f"unsupported FN-rescue stage combination: {','.join(stages)}")


if __name__ == "__main__":
    raise SystemExit(main())

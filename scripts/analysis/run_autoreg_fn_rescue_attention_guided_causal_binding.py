#!/usr/bin/env python
"""Run Phase-3 FN-rescue attention-guided causal binding analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.autoreg_fn_rescue_attention_guided_causal_binding import (  # noqa: E402
    PHASE3_STAGES,
    build_phase3_dry_run_plan,
    load_phase3_config,
    materialize_case_linked_table,
    materialize_competitor_source,
    materialize_sink_triage,
    materialize_target_mask,
    write_phase3_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stages", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _stages(raw: str, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    stages = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not stages:
        parser.error("--stages must not be empty")
    unknown = [stage for stage in stages if stage not in PHASE3_STAGES]
    if unknown:
        parser.error(f"unknown phase3 stage(s): {', '.join(unknown)}")
    return stages


def _print_json(payload: object) -> None:
    print(json.dumps(payload, separators=(",", ":")))


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    stages = _stages(args.stages, parser)
    try:
        config = load_phase3_config(args.config)
        if args.dry_run:
            _print_json(build_phase3_dry_run_plan(config, stages=stages))
            return 0
        payload: dict[str, object] = {}
        if "target_mask" in stages:
            payload["target_mask"] = materialize_target_mask(config)
        if "competitor_source" in stages:
            payload["competitor_source"] = materialize_competitor_source(config)
        if "sink_triage" in stages:
            payload["sink_triage"] = materialize_sink_triage(config)
        if "case_linked" in stages:
            payload["case_linked"] = materialize_case_linked_table(config)
        if "report" in stages:
            payload["report"] = str(write_phase3_report(config))
        _print_json(payload)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

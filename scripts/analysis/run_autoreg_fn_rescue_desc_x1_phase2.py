#!/usr/bin/env python
"""Run FN-rescue desc->x1 phase-2 mechanism analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.autoreg_fn_rescue_desc_x1_phase2 import (  # noqa: E402
    PHASE2_STAGES,
    build_phase2_dry_run_plan,
    load_phase2_config,
    materialize_attention_mining,
    materialize_intervention_plan,
    materialize_intervention_smoke,
    write_phase2_report,
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
    unknown = [stage for stage in stages if stage not in PHASE2_STAGES]
    if unknown:
        parser.error(f"unknown phase2 stage(s): {', '.join(unknown)}")
    return stages


def _print_json(payload: object) -> None:
    print(json.dumps(payload, separators=(",", ":")))


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    stages = _stages(args.stages, parser)
    try:
        config = load_phase2_config(args.config)
        if args.dry_run:
            _print_json(build_phase2_dry_run_plan(config, stages=stages))
            return 0
        payload: dict[str, object] = {}
        if "attention_mining" in stages:
            payload["attention_mining"] = materialize_attention_mining(config)
        if "intervention_plan" in stages:
            payload["intervention_plan"] = materialize_intervention_plan(config)
        if "intervention_smoke" in stages:
            payload["intervention_smoke"] = materialize_intervention_smoke(config)
        if "report" in stages:
            payload["report"] = str(write_phase2_report(config.paths.artifact_root))
        _print_json(payload)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

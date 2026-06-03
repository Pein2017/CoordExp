#!/usr/bin/env python
"""Run Phase-5 FN-rescue desc->x1 logit binding analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.autoreg_fn_rescue_desc_x1_phase5 import (  # noqa: E402
    PHASE5_STAGES,
    build_phase5_dry_run_plan,
    load_phase5_config,
    materialize_coord_slot_logit_binding_probe,
    materialize_x1_logit_binding_probe,
    write_phase5_report,
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
    unknown = sorted(set(stages) - set(PHASE5_STAGES))
    if unknown:
        parser.error(f"unknown phase5 stage(s): {', '.join(unknown)}")
    return stages


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    stages = _stages(args.stages, parser)
    try:
        config = load_phase5_config(args.config)
        if args.dry_run:
            print(json.dumps(build_phase5_dry_run_plan(config, stages=stages), separators=(",", ":")))
            return 0
        payload: dict[str, object] = {}
        if "x1_logit_binding_probe" in stages:
            payload["x1_logit_binding_probe"] = materialize_x1_logit_binding_probe(config)
        if "coord_slot_logit_binding_probe" in stages:
            payload["coord_slot_logit_binding_probe"] = materialize_coord_slot_logit_binding_probe(config)
        if "report" in stages:
            payload["report"] = str(write_phase5_report(config))
        print(json.dumps(payload, separators=(",", ":")))
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

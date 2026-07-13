#!/usr/bin/env python3
"""Plan the Spatial Scope and History Disentanglement research execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.spatial_scope_history.calibration import PrimaryScheduleArtifact
from src.analysis.spatial_scope_history.cohort_ledger import AttemptLedger
from src.analysis.spatial_scope_history.runner import build_coordinator_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate frozen inputs and emit a dry-run whole-batch execution plan for "
            "the Spatial Scope and History Disentanglement research unit."
        )
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        required=True,
        help="Frozen PrimaryScheduleArtifact JSON artifact.",
    )
    parser.add_argument(
        "--attempt-ledger",
        type=Path,
        help="Optional append-only terminal AttemptRecord JSON Lines artifact.",
    )
    parser.add_argument(
        "--physical-gpu-token",
        action="append",
        required=True,
        help=(
            "One physical graphics-processing-unit token; repeat once per persistent "
            "worker. Each worker sees only its token and uses logical cuda:0."
        ),
    )
    parser.add_argument(
        "--output-plan",
        type=Path,
        required=True,
        help="Write-once output path for the canonical coordinator plan JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Required safety gate: validate and plan without loading a model or CUDA.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schedule_path = args.schedule.resolve(strict=True)
    schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule_artifact = PrimaryScheduleArtifact.from_artifact_dict(schedule_payload)
    schedule = schedule_artifact.schedule
    attempt_ledger = None
    if args.attempt_ledger is not None:
        ledger_path = args.attempt_ledger.resolve(strict=True)
        attempt_ledger = AttemptLedger.from_jsonl_bytes(
            ledger_path.read_bytes(),
            run_id=schedule.identity.run_id,
            schedule_sha256=schedule.fingerprint,
            execution_identity=schedule.identity.execution_identity,
        )
    plan = build_coordinator_plan(
        schedule,
        physical_gpu_tokens=tuple(args.physical_gpu_token),
        attempt_ledger=attempt_ledger,
    )
    output_path = args.output_plan.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(plan.to_artifact_dict(), handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Execute the sealed Spatial Scope and History Disentanglement schedule.

This is intentionally a thin production entrypoint.  The schedule owns request
ordering, physical batch membership, sampling seeds, and cumulative barriers;
the runner owns persistent worker lifecycle and resumability; the production
executor owns model inference and terminal artifact persistence.  This script
only binds those existing contracts to command-line paths and GPU assignments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.spatial_scope_history.calibration import PrimaryScheduleArtifact
from src.analysis.spatial_scope_history.cohort_ledger import AttemptLedger
from src.analysis.spatial_scope_history.production_executor import (
    ProductionExecutorFactory,
    PRODUCTION_EXECUTOR_FACTORY_CONFIG_SCHEMA_VERSION,
)
from src.analysis.spatial_scope_history.runner import (
    PersistentWorkerPool,
    ProcessPoolRunReceipt,
    build_coordinator_plan,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal production execution command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Execute the sealed Spatial Scope and History Disentanglement "
            "schedule with one persistent worker per physical graphics "
            "processing unit (GPU)."
        )
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        required=True,
        help="Frozen PrimaryScheduleArtifact JSON artifact.",
    )
    parser.add_argument(
        "--infer-config",
        type=Path,
        required=True,
        help="Resolved inference configuration used by each worker.",
    )
    parser.add_argument(
        "--cohort-ledger",
        type=Path,
        required=True,
        help="Accepted cohort ledger JSON Lines artifact.",
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        required=True,
        help="Exact source dataset JSON Lines artifact bound by the cohort.",
    )
    parser.add_argument(
        "--calibration-selection-receipt",
        type=Path,
        required=True,
        help="Calibration selection receipt bound by the primary schedule.",
    )
    parser.add_argument(
        "--source-runtime-identity-receipt",
        type=Path,
        required=True,
        help="Source/runtime identity receipt bound by the primary schedule.",
    )
    parser.add_argument(
        "--sampled-runtime-attestation",
        type=Path,
        required=True,
        help="Verified sampled-runtime attestation aggregate artifact.",
    )
    parser.add_argument(
        "--attempt-ledger",
        type=Path,
        required=True,
        help=(
            "Append-only terminal AttemptRecord JSON Lines artifact. Existing "
            "records are used to derive the safe resume frontier."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Root directory for write-once per-request execution artifacts.",
    )
    parser.add_argument(
        "--physical-gpu-token",
        action="append",
        required=True,
        help=(
            "One physical GPU token per persistent worker; repeat once for each "
            "GPU. A token may not contain a comma."
        ),
    )
    parser.add_argument(
        "--result-timeout-seconds",
        type=float,
        default=600.0,
        help="Maximum wait for one worker lifecycle message (default: 600).",
    )
    return parser


def execute_research_schedule(
    *,
    schedule_path: Path,
    infer_config_path: Path,
    cohort_ledger_path: Path,
    source_jsonl_path: Path,
    calibration_selection_receipt_path: Path,
    source_runtime_identity_receipt_path: Path,
    sampled_runtime_attestation_path: Path,
    attempt_ledger_path: Path,
    artifact_root: Path,
    physical_gpu_tokens: Sequence[str],
    result_timeout_seconds: float = 600.0,
    pool_factory: Callable[..., Any] = PersistentWorkerPool,
    executor_factory: Any = ProductionExecutorFactory,
) -> ProcessPoolRunReceipt | None:
    """Execute the current resume frontier through the existing worker pool.

    ``pool_factory`` and ``executor_factory`` are injectable solely for narrow
    tests; production callers use the canonical persistent pool and production
    executor factory.  An empty resume frontier is a successful no-op and does
    not start model workers.
    """

    schedule_path = schedule_path.expanduser().resolve(strict=True)
    infer_config_path = infer_config_path.expanduser().resolve(strict=True)
    cohort_ledger_path = cohort_ledger_path.expanduser().resolve(strict=True)
    source_jsonl_path = source_jsonl_path.expanduser().resolve(strict=True)
    calibration_selection_receipt_path = (
        calibration_selection_receipt_path.expanduser().resolve(strict=True)
    )
    source_runtime_identity_receipt_path = (
        source_runtime_identity_receipt_path.expanduser().resolve(strict=True)
    )
    sampled_runtime_attestation_path = (
        sampled_runtime_attestation_path.expanduser().resolve(strict=True)
    )
    attempt_ledger_path = attempt_ledger_path.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()

    if result_timeout_seconds <= 0:
        raise ValueError("result timeout must be positive")

    schedule_artifact = PrimaryScheduleArtifact.from_artifact_dict(
        _read_json_object(schedule_path)
    )
    schedule = schedule_artifact.schedule
    attempt_ledger = AttemptLedger.from_jsonl_bytes(
        attempt_ledger_path.read_bytes() if attempt_ledger_path.exists() else b"",
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
        execution_identity=schedule.identity.execution_identity,
    )
    plan = build_coordinator_plan(
        schedule,
        physical_gpu_tokens=tuple(physical_gpu_tokens),
        attempt_ledger=attempt_ledger,
    )
    if not plan.waves:
        return None

    factory_config = {
        "attempt_ledger_path": str(attempt_ledger_path),
        "calibration_selection_receipt_path": str(
            calibration_selection_receipt_path
        ),
        "cohort_ledger_path": str(cohort_ledger_path),
        "infer_config_path": str(infer_config_path),
        "primary_schedule_artifact_path": str(schedule_path),
        "sampled_runtime_attestation_path": str(sampled_runtime_attestation_path),
        "schema_version": PRODUCTION_EXECUTOR_FACTORY_CONFIG_SCHEMA_VERSION,
        "source_jsonl_path": str(source_jsonl_path),
        "source_runtime_identity_receipt_path": str(
            source_runtime_identity_receipt_path
        ),
    }
    factory = executor_factory() if isinstance(executor_factory, type) else executor_factory
    with pool_factory(
        workers=plan.workers,
        executor_factory=factory,
        factory_config=factory_config,
        result_timeout_seconds=result_timeout_seconds,
    ) as pool:
        return pool.execute_plan(plan, artifact_root=artifact_root)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, execute, and emit a compact lifecycle summary."""

    args = build_parser().parse_args(argv)
    receipt = execute_research_schedule(
        schedule_path=args.schedule,
        infer_config_path=args.infer_config,
        cohort_ledger_path=args.cohort_ledger,
        source_jsonl_path=args.source_jsonl,
        calibration_selection_receipt_path=args.calibration_selection_receipt,
        source_runtime_identity_receipt_path=args.source_runtime_identity_receipt,
        sampled_runtime_attestation_path=args.sampled_runtime_attestation,
        attempt_ledger_path=args.attempt_ledger,
        artifact_root=args.artifact_root,
        physical_gpu_tokens=args.physical_gpu_token,
        result_timeout_seconds=args.result_timeout_seconds,
    )
    if receipt is None:
        print(json.dumps({"status": "already_complete"}, sort_keys=True))
        return 0
    print(
        json.dumps(
            {
                "batch_count": len(receipt.batch_receipts),
                "status": "completed",
                "worker_count": len(receipt.worker_startups),
            },
            sort_keys=True,
        )
    )
    return 0


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())

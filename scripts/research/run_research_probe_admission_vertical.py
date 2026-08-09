#!/usr/bin/env python3
"""Run the bounded support admission vertical and the crossover CPU adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

INFRA_ROOT = Path(__file__).resolve().parents[2]
if str(INFRA_ROOT) not in sys.path:
    sys.path.insert(0, str(INFRA_ROOT))

from scripts.research import research_probe_admission_consumers as consumers  # noqa: E402
from scripts.research import resumable_natural_boundary_support_completion as support  # noqa: E402
from scripts.research import run_resumable_natural_boundary_support_shard as worker  # noqa: E402
from src.artifacts.json_values import (  # noqa: E402
    ArtifactContractError,
    canonical_json_bytes,
    json_sha256,
    publish_json_exclusive,
)
from src.artifacts.research_probe_admission import (  # noqa: E402
    AbsoluteExecutableBinding,
    DirectoryTreeBinding,
    RegularFileBinding,
    ResearchProbeAdmission,
    ReservedOutputPath,
    StrictValueBinding,
    capture_binding_manifest,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research-probe-infras/"
    "2026-08-09-production-shaped-admission-target-vertical-v1"
)
DEFAULT_CONTEXT_ID = "ctx:0a055737dcb38ca92bb8bce5"
INFER_CONFIG = consumers.ACTIVE_RESEARCH_PROBES_ROOT / (
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml"
)
ADAPTER_TENSOR = Path(
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/"
    "step-2444/adapter/adapter_model.safetensors"
)
EMBEDDING_DELTA = Path(
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/"
    "step-2444/special_token_embeddings/special_token_embeddings.safetensors"
)
BASE_MODEL = worker.EXPECTED_BASE_MODEL
SOURCE_GATE_ROOT = worker.EXPECTED_SOURCE_GATE_ROOT


class VerticalAcceptanceError(RuntimeError):
    """The bounded acceptance run cannot preserve its declared identity."""


def _file_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    encoded = resolved.read_bytes()
    return {
        "path": str(resolved),
        "raw_sha256": hashlib.sha256(encoded).hexdigest(),
        "byte_count": len(encoded),
    }


def _publish_receipt_idempotent(path: Path, value: dict[str, Any]) -> None:
    """Recover only the exact write-once receipt after uncertain publication."""

    expected = canonical_json_bytes(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise VerticalAcceptanceError(
                "existing mechanics receipt differs from exact publication bytes"
            )
        return
    try:
        publish_json_exclusive(path, value)
    except ArtifactContractError as exc:
        if (
            path.exists()
            and not path.is_symlink()
            and path.is_file()
            and path.read_bytes() == expected
        ):
            return
        raise VerticalAcceptanceError("mechanics receipt publication failed") from exc


def _expected_execution(
    *, context_id: str, physical_gpu: str
) -> tuple[dict[str, Any], support.LogicalPlan, dict[str, Any]]:
    consumer = consumers.SUPPORT_CONSUMER.resolve(strict=True)
    plan_path = consumers.SUPPORT_PLAN.resolve(strict=True)
    census = consumers.SUPPORT_CENSUS.resolve(strict=True)
    infer_config = INFER_CONFIG.resolve(strict=True)
    adapter_tensor = ADAPTER_TENSOR.resolve(strict=True)
    embedding_delta = EMBEDDING_DELTA.resolve(strict=True)
    logical_plan = support.load_logical_plan(plan_path)
    projected = support.project_logical_contexts(logical_plan, context_ids=(context_id,))
    schedule = support.plan_physical_slots(projected, slot_count=1)
    _, base_model_identity = worker._require_model_root(BASE_MODEL)
    source_gate = worker._bind_source_gate_root(SOURCE_GATE_ROOT)
    runtime_source = worker._bind_consumer_runtime_source(consumer)
    execution_identity = {
        "consumer": {
            "path": str(consumer),
            "sha256": support.file_sha256(consumer),
        },
        "consumer_runtime_source": runtime_source,
        "adapter": {
            "path": str(Path(support.__file__).resolve(strict=True)),
            "sha256": support.file_sha256(support.__file__),
        },
        "worker": {
            "path": str(Path(worker.__file__).resolve(strict=True)),
            "sha256": support.file_sha256(worker.__file__),
        },
        "logical_plan": {
            "path": str(plan_path),
            "file_sha256": logical_plan.file_sha256,
            "content_sha256": logical_plan.content_sha256,
        },
        "bounded_context_ids": [context_id],
        "schedule_sha256": schedule["content_sha256"],
        "census": {
            "path": str(census),
            "sha256": support.file_sha256(census),
        },
        "infer_config": {
            "path": str(infer_config),
            "sha256": support.file_sha256(infer_config),
        },
        "model": {
            "base_model": base_model_identity,
            "adapter_tensor": {
                "path": str(adapter_tensor),
                "sha256": support.file_sha256(adapter_tensor),
            },
            "embedding_delta": {
                "path": str(embedding_delta),
                "sha256": support.file_sha256(embedding_delta),
            },
        },
        "embedding_source_gate": source_gate,
        "runtime_policy": {
            "logical_device": "cuda:0",
            "cuda_visible_devices": physical_gpu,
            "physical_slot_index": 0,
            "physical_slot_count": 1,
        },
    }
    return execution_identity, projected, schedule


def _worker_argv(
    *,
    executable: Path,
    execution_identity: dict[str, Any],
    context_id: str,
    journal_root: Path,
    runtime_receipt: Path,
    execution_id: str,
) -> list[str]:
    return [
        str(executable),
        "-m",
        "scripts.research.run_resumable_natural_boundary_support_shard",
        "--consumer",
        execution_identity["consumer"]["path"],
        "--consumer-sha256",
        execution_identity["consumer"]["sha256"],
        "--plan",
        execution_identity["logical_plan"]["path"],
        "--plan-sha256",
        execution_identity["logical_plan"]["file_sha256"],
        "--census",
        execution_identity["census"]["path"],
        "--census-sha256",
        execution_identity["census"]["sha256"],
        "--infer-config",
        execution_identity["infer_config"]["path"],
        "--infer-config-sha256",
        execution_identity["infer_config"]["sha256"],
        "--adapter-tensor",
        execution_identity["model"]["adapter_tensor"]["path"],
        "--adapter-tensor-sha256",
        execution_identity["model"]["adapter_tensor"]["sha256"],
        "--embedding-delta",
        execution_identity["model"]["embedding_delta"]["path"],
        "--embedding-delta-sha256",
        execution_identity["model"]["embedding_delta"]["sha256"],
        "--base-model",
        execution_identity["model"]["base_model"]["path"],
        "--source-gate-root",
        execution_identity["embedding_source_gate"]["root"],
        "--journal-root",
        str(journal_root),
        "--runtime-receipt",
        str(runtime_receipt),
        "--execution-id",
        execution_id,
        "--physical-slot-index",
        "0",
        "--physical-slot-count",
        "1",
        "--context-id",
        context_id,
        "--device",
        "cuda:0",
    ]


def _support_bindings(
    *,
    execution_identity: dict[str, Any],
    executable: Path,
    worker_argv: list[str],
    expected_runtime: dict[str, str],
) -> Any:
    requests: list[Any] = list(consumers.support_binding_requests())
    requests.extend(
        [
            RegularFileBinding("support_vertical_driver", Path(__file__).resolve()),
            RegularFileBinding("support_infer_config", INFER_CONFIG),
            RegularFileBinding("support_adapter_tensor", ADAPTER_TENSOR),
            RegularFileBinding("support_embedding_delta", EMBEDDING_DELTA),
            DirectoryTreeBinding("support_base_model", BASE_MODEL),
            AbsoluteExecutableBinding("support_python", executable),
            StrictValueBinding(
                "support_consumer_runtime_source",
                execution_identity["consumer_runtime_source"],
            ),
            StrictValueBinding(
                "support_embedding_source_gate",
                execution_identity["embedding_source_gate"],
            ),
            StrictValueBinding("support_execution_identity", execution_identity),
            StrictValueBinding("support_worker_argv", worker_argv),
            StrictValueBinding("support_worker_cwd", str(INFRA_ROOT)),
            StrictValueBinding("support_expected_runtime", expected_runtime),
        ]
    )
    runtime_root = Path(execution_identity["consumer_runtime_source"]["tree"]["root"])
    for index, path in enumerate(sorted(runtime_root.rglob("*.py"))):
        requests.append(
            RegularFileBinding(f"support_runtime_source_{index:04d}", path)
        )
    source_gate_files = execution_identity["embedding_source_gate"]["files"]
    for index, reference in enumerate(
        value for _, value in sorted(source_gate_files.items())
    ):
        requests.append(
            RegularFileBinding(
                f"support_source_gate_{index:04d}", Path(reference["path"])
            )
        )
    return capture_binding_manifest(tuple(requests))


def _append_cpu_stages(
    *,
    root: Path,
    support_manifest: Any,
) -> tuple[ResearchProbeAdmission, ResearchProbeAdmission]:
    support_cpu = root / "support" / "cpu"
    support_vertical = root / "support" / "vertical"
    support_admission = ResearchProbeAdmission.create(
        root=root / "support" / "admission",
        admission_id="natural-boundary-support-bounded-v1",
        bindings=support_manifest,
        reserved_output_paths=(
            ReservedOutputPath("support_cpu", support_cpu),
            ReservedOutputPath("support_vertical", support_vertical),
        ),
        context={
            "consumer": "natural_boundary_support",
            "scope": "bounded_live_mechanics_only",
        },
    )
    support_admission.append_stage(
        stage="cpu_preflight",
        evidence=consumers.build_support_cpu_evidence(
            output_root=support_cpu,
            compatibility_receipt_path=consumers.SUPPORT_COMPATIBILITY_RECEIPT,
            plan_path=consumers.SUPPORT_PLAN,
            census_path=consumers.SUPPORT_CENSUS,
            consumer_path=consumers.SUPPORT_CONSUMER,
            consumer_test_path=consumers.SUPPORT_CONSUMER_TEST,
            merger_path=consumers.SUPPORT_MERGER,
            merger_test_path=consumers.SUPPORT_MERGER_TEST,
        ),
        attempt_id=support_admission.start_attempt(),
    )

    crossover_cpu = root / "crossover" / "cpu"
    crossover_vertical = root / "crossover" / "vertical"
    crossover_manifest = capture_binding_manifest(
        consumers.crossover_binding_requests()
    )
    crossover_admission = ResearchProbeAdmission.create(
        root=root / "crossover" / "admission",
        admission_id="k10-h20-crossover-cpu-v1",
        bindings=crossover_manifest,
        reserved_output_paths=(
            ReservedOutputPath("crossover_cpu", crossover_cpu),
            ReservedOutputPath("crossover_vertical", crossover_vertical),
        ),
        context={
            "consumer": "k10_h20_crossover",
            "scope": "cpu_compatibility_only",
        },
    )
    crossover_admission.append_stage(
        stage="cpu_preflight",
        evidence=consumers.build_crossover_cpu_evidence(
            output_root=crossover_cpu,
            compatibility_receipt_path=consumers.CROSSOVER_COMPATIBILITY_RECEIPT,
            source_preflight_path=consumers.CROSSOVER_SOURCE_PREFLIGHT,
            accepted_evidence_path=consumers.CROSSOVER_ACCEPTED_EVIDENCE,
            runner_path=consumers.CROSSOVER_RUNNER,
            runner_test_path=consumers.CROSSOVER_RUNNER_TEST,
            sealer_path=consumers.CROSSOVER_SEALER,
            finalizer_path=consumers.CROSSOVER_FINALIZER,
            finalizer_test_path=consumers.CROSSOVER_FINALIZER_TEST,
        ),
        attempt_id=crossover_admission.start_attempt(),
    )
    return support_admission, crossover_admission


def run(*, output_root: Path, physical_gpu: str, context_id: str) -> Path:
    if output_root.exists() or output_root.is_symlink():
        raise VerticalAcceptanceError("acceptance output root must be fresh")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != physical_gpu or not visible or "," in visible or visible == "-1":
        raise VerticalAcceptanceError(
            "CUDA_VISIBLE_DEVICES must equal the one explicit physical GPU"
        )
    executable = Path(sys.executable).resolve(strict=True)
    execution_identity, projected_plan, schedule = _expected_execution(
        context_id=context_id,
        physical_gpu=physical_gpu,
    )
    vertical_root = output_root / "support" / "vertical"
    journal_root = vertical_root / "journal" / "slot-000"
    runtime_receipt = vertical_root / "runtime.json"
    worker_exit_receipt = vertical_root / "worker-exit.json"
    terminal_path = vertical_root / "bounded-terminal.json"
    execution_id = "production-shaped-admission-support-slot-000-v1"
    worker_argv = _worker_argv(
        executable=executable,
        execution_identity=execution_identity,
        context_id=context_id,
        journal_root=journal_root,
        runtime_receipt=runtime_receipt,
        execution_id=execution_id,
    )
    expected_runtime = {
        "checkpoint": "S",
        "config_fingerprint": str(
            projected_plan.raw["h0_lineage"]["config_fingerprint"]
        ),
    }
    support_manifest = _support_bindings(
        execution_identity=execution_identity,
        executable=executable,
        worker_argv=worker_argv,
        expected_runtime=expected_runtime,
    )
    support_admission, crossover_admission = _append_cpu_stages(
        root=output_root,
        support_manifest=support_manifest,
    )
    try:
        support.launch_slot_worker(
            command=worker_argv,
            cwd=INFRA_ROOT,
            journal_root=journal_root,
            mechanics_receipt_path=worker_exit_receipt,
            physical_slot_index=0,
            logical_plan_file_sha256=projected_plan.file_sha256,
            schedule_sha256=schedule["content_sha256"],
            expected_context_count=1,
            timeout_seconds=7200.0,
            terminate_after_first_durable_record=False,
        )
        support.materialize_bounded_mechanics_terminal(
            plan=projected_plan,
            schedule=schedule,
            slot_roots=(journal_root,),
            execution_identity=execution_identity,
            output_path=terminal_path,
        )
        vertical_evidence = consumers.build_support_vertical_evidence(
            output_root=vertical_root / "snapshot",
            terminal_path=terminal_path,
            plan_path=consumers.SUPPORT_PLAN,
            schedule=schedule,
            slot_roots=(journal_root,),
            execution_identity=execution_identity,
            admission_bindings=support_manifest,
            worker_path=consumers.SUPPORT_WORKER,
            executable_path=executable,
            worker_argv=worker_argv,
            worker_cwd=INFRA_ROOT,
            worker_exit_receipts=(worker_exit_receipt,),
            runtime_receipts=(runtime_receipt,),
            expected_checkpoint=expected_runtime["checkpoint"],
            expected_config_fingerprint=expected_runtime["config_fingerprint"],
        )
        support_admission.append_stage(
            stage="vertical_smoke",
            evidence=vertical_evidence,
            attempt_id=support_admission.start_attempt(),
        )
        admission_receipt = support_admission.finalize()
    finally:
        support_admission.close()
        crossover_admission.close()

    support_inspection = ResearchProbeAdmission.inspect(
        output_root / "support" / "admission"
    )
    crossover_inspection = ResearchProbeAdmission.inspect(
        output_root / "crossover" / "admission"
    )
    receipt = {
        "schema_version": "research_probe_admission.acceptance_vertical.v1",
        "kind": "production_shaped_research_probe_admission_acceptance",
        "support_admission": _file_ref(admission_receipt),
        "support_completed_stages": list(support_inspection.completed_stages),
        "support_mechanically_admitted": support_inspection.mechanically_admitted,
        "crossover_completed_stages": list(crossover_inspection.completed_stages),
        "crossover_missing_stages": list(crossover_inspection.missing_stages),
        "crossover_mechanically_admitted": crossover_inspection.mechanically_admitted,
        "bounded_terminal": _file_ref(terminal_path),
        "runtime_receipt": _file_ref(runtime_receipt),
        "worker_exit_receipt": _file_ref(worker_exit_receipt),
        "execution_identity_fingerprint": json_sha256(execution_identity),
        "binding_manifest_fingerprint": support_manifest.content_fingerprint,
        "physical_gpu": physical_gpu,
        "logical_device": "cuda:0",
        "durable_context_ids": [context_id],
        "legacy_merger_closed": False,
        "scientific_interpretation": False,
        "bulk_launch_authorized": False,
        "claim_boundary": "bounded_admission_mechanics_only_no_scientific_or_launch_authority",
    }
    receipt["content_sha256"] = json_sha256(receipt)
    receipt_path = output_root / "mechanics-receipt.json"
    _publish_receipt_idempotent(receipt_path, receipt)
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--context-id", default=DEFAULT_CONTEXT_ID)
    args = parser.parse_args()
    path = run(
        output_root=args.output_root.resolve(),
        physical_gpu=args.physical_gpu,
        context_id=args.context_id,
    )
    print(json.dumps({"mechanics_receipt": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

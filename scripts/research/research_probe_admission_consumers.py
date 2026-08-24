"""Receipt-only mechanics adapters for research-probe admission consumers.

This module never imports a sibling consumer, executes a callback, loads a
model, or starts a worker.  Callers supply already-produced compatibility and
runtime receipts; this module verifies their byte bindings and returns the
shared :class:`StageEvidence` envelope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import src.artifacts.evidence_journal as evidence_journal_module
import src.artifacts.json_values as json_values_module
import src.artifacts.research_probe_admission as admission_module
from scripts.research import resumable_natural_boundary_support_completion as support
from src.artifacts.evidence_journal import ExecutionEvidenceJournal
from src.artifacts.json_values import (
    canonical_json_bytes,
    json_sha256,
    load_canonical_json,
    publish_json_exclusive,
    validate_json_value,
)
from src.artifacts.research_probe_admission import (
    AbsoluteExecutableBinding,
    BindingManifest,
    RegularFileBinding,
    StageEvidence,
    StrictValueBinding,
    TargetTreeBinding,
)
from src.common.errors import ArtifactContractError


INFRA_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_RESEARCH_PROBES_ROOT = Path("/data/CoordExp/.worktrees/research-probes")
SUPPORT_CONSUMER = (
    ACTIVE_RESEARCH_PROBES_ROOT
    / "scripts/research/run_natural_boundary_support_completion.py"
)
SUPPORT_CONSUMER_TEST = (
    ACTIVE_RESEARCH_PROBES_ROOT
    / "tests/research/test_run_natural_boundary_support_completion.py"
)
SUPPORT_MERGER = (
    ACTIVE_RESEARCH_PROBES_ROOT
    / "scripts/research/merge_natural_boundary_support_completion.py"
)
SUPPORT_MERGER_TEST = (
    ACTIVE_RESEARCH_PROBES_ROOT
    / "tests/test_merge_natural_boundary_support_completion.py"
)
SUPPORT_ADAPTER_TEST = (
    INFRA_ROOT / "tests/research/test_resumable_natural_boundary_support_completion.py"
)
SUPPORT_WORKER = (
    INFRA_ROOT / "scripts/research/run_resumable_natural_boundary_support_shard.py"
)
CONSUMER_ADMISSION_ADAPTER = Path(__file__).resolve()
ADMISSION_CORE = Path(admission_module.__file__).resolve()
EVIDENCE_JOURNAL_CORE = Path(evidence_journal_module.__file__).resolve()
JSON_VALUES_CORE = Path(json_values_module.__file__).resolve()
ADMISSION_ACCEPTANCE_DRIVER = (
    INFRA_ROOT / "scripts/research/generate_research_probe_admission_cpu_receipts.py"
)
SUPPORT_PLAN = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/support-completion-plan-v1/plan.json"
)
SUPPORT_CENSUS = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/cpu-census-v2/admission-census.json"
)
SUPPORT_COMPATIBILITY_RECEIPT = Path(
    "/data/CoordExp/outputs/research-probe-infras/"
    "2026-08-24-target-binding-cpu-compatibility-v1/"
    "support-cpu-compatibility.json"
)
CROSSOVER_RUNNER = (
    ACTIVE_RESEARCH_PROBES_ROOT / "scripts/research/run_s_k10_h20_crossover_shard.py"
)
CROSSOVER_RUNNER_TEST = (
    ACTIVE_RESEARCH_PROBES_ROOT / "tests/research/test_run_s_k10_h20_crossover_shard.py"
)
CROSSOVER_SEALER = ACTIVE_RESEARCH_PROBES_ROOT / (
    "scripts/research/seal_s_k10_h20_crossover_pre_gpu_receipt.py"
)
CROSSOVER_FINALIZER = (
    ACTIVE_RESEARCH_PROBES_ROOT / "scripts/research/finalize_s_k10_h20_crossover.py"
)
CROSSOVER_FINALIZER_TEST = (
    ACTIVE_RESEARCH_PROBES_ROOT / "tests/research/test_finalize_s_k10_h20_crossover.py"
)
CROSSOVER_SOURCE_PREFLIGHT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-07-s-k10-h20-natural-crossover/pre-gpu-evidence-v5/source-preflight-preseal.json"
)
CROSSOVER_ACCEPTED_EVIDENCE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-07-s-k10-h20-natural-crossover/evidence-v4/evidence.json"
)
CROSSOVER_COMPATIBILITY_RECEIPT = Path(
    "/data/CoordExp/outputs/research-probe-infras/"
    "2026-08-24-target-binding-cpu-compatibility-v1/"
    "crossover-cpu-compatibility.json"
)


class ConsumerAdmissionError(ValueError):
    """A receipt or already-completed bounded surface is not mechanically exact."""


def support_binding_requests(
    *,
    consumer: Path = SUPPORT_CONSUMER,
    consumer_test: Path = SUPPORT_CONSUMER_TEST,
    plan: Path = SUPPORT_PLAN,
    census: Path = SUPPORT_CENSUS,
    merger: Path = SUPPORT_MERGER,
    merger_test: Path = SUPPORT_MERGER_TEST,
    adapter_test: Path = SUPPORT_ADAPTER_TEST,
    compatibility_receipt: Path = SUPPORT_COMPATIBILITY_RECEIPT,
) -> tuple[RegularFileBinding | AbsoluteExecutableBinding | StrictValueBinding, ...]:
    """Bind support source, tests, plan/census, and a read-only verification file."""

    return (
        RegularFileBinding("support_consumer", consumer),
        RegularFileBinding("support_consumer_test", consumer_test),
        RegularFileBinding("support_plan", plan),
        RegularFileBinding("support_census", census),
        RegularFileBinding("support_merger", merger),
        RegularFileBinding("support_merger_test", merger_test),
        RegularFileBinding("support_adapter_test", adapter_test),
        RegularFileBinding("support_compatibility_receipt", compatibility_receipt),
        RegularFileBinding("support_adapter", Path(support.__file__).resolve()),
        RegularFileBinding("support_worker", SUPPORT_WORKER),
        RegularFileBinding("consumer_admission_adapter", CONSUMER_ADMISSION_ADAPTER),
        RegularFileBinding("admission_core", ADMISSION_CORE),
        RegularFileBinding("evidence_journal_core", EVIDENCE_JOURNAL_CORE),
        RegularFileBinding("json_values_core", JSON_VALUES_CORE),
        RegularFileBinding(
            "consumer_admission_acceptance", ADMISSION_ACCEPTANCE_DRIVER
        ),
        StrictValueBinding(
            "support_mechanics_boundary", {"legacy_merger_closed": False}
        ),
        *_command_binding_requests(compatibility_receipt, "support"),
    )


def crossover_binding_requests(
    *,
    runner: Path = CROSSOVER_RUNNER,
    runner_test: Path = CROSSOVER_RUNNER_TEST,
    sealer: Path = CROSSOVER_SEALER,
    finalizer: Path = CROSSOVER_FINALIZER,
    finalizer_test: Path = CROSSOVER_FINALIZER_TEST,
    source_preflight: Path = CROSSOVER_SOURCE_PREFLIGHT,
    accepted_evidence: Path = CROSSOVER_ACCEPTED_EVIDENCE,
    compatibility_receipt: Path = CROSSOVER_COMPATIBILITY_RECEIPT,
) -> tuple[RegularFileBinding | AbsoluteExecutableBinding, ...]:
    """Bind crossover production/test surfaces and opaque public receipts."""

    return (
        RegularFileBinding("crossover_runner", runner),
        RegularFileBinding("crossover_runner_test", runner_test),
        RegularFileBinding("crossover_sealer", sealer),
        RegularFileBinding("crossover_finalizer", finalizer),
        RegularFileBinding("crossover_finalizer_test", finalizer_test),
        RegularFileBinding("crossover_source_preflight", source_preflight),
        RegularFileBinding("crossover_accepted_evidence", accepted_evidence),
        RegularFileBinding("crossover_compatibility_receipt", compatibility_receipt),
        RegularFileBinding("admission_core", ADMISSION_CORE),
        RegularFileBinding("evidence_journal_core", EVIDENCE_JOURNAL_CORE),
        RegularFileBinding("json_values_core", JSON_VALUES_CORE),
        RegularFileBinding(
            "consumer_admission_acceptance", ADMISSION_ACCEPTANCE_DRIVER
        ),
        *_command_binding_requests(compatibility_receipt, "crossover"),
    )


def support_target_tree_binding(
    *, effective_binding_names: tuple[str, ...]
) -> TargetTreeBinding:
    """Declare the exact support execution target and its already-bound inputs."""

    return TargetTreeBinding(
        root=ACTIVE_RESEARCH_PROBES_ROOT,
        effective_binding_names=effective_binding_names,
    )


def crossover_target_tree_binding(
    *, effective_binding_names: tuple[str, ...]
) -> TargetTreeBinding:
    """Declare the exact crossover execution target and its already-bound inputs."""

    return TargetTreeBinding(
        root=ACTIVE_RESEARCH_PROBES_ROOT,
        effective_binding_names=effective_binding_names,
    )


def build_support_cpu_evidence(
    *,
    output_root: Path,
    compatibility_receipt_path: Path,
    plan_path: Path,
    census_path: Path,
    consumer_path: Path,
    consumer_test_path: Path,
    merger_path: Path,
    merger_test_path: Path,
    adapter_path: Path = Path(support.__file__).resolve(),
    adapter_test_path: Path = SUPPORT_ADAPTER_TEST,
) -> StageEvidence:
    """Accept one completed, canonical support CPU compatibility receipt only."""

    _require_exact_paths(
        {
            "consumer": consumer_path,
            "consumer_test": consumer_test_path,
            "merger": merger_path,
            "merger_test": merger_test_path,
            "adapter": adapter_path,
            "adapter_test": adapter_test_path,
            "plan": plan_path,
            "census": census_path,
        },
        {
            "consumer": SUPPORT_CONSUMER,
            "consumer_test": SUPPORT_CONSUMER_TEST,
            "merger": SUPPORT_MERGER,
            "merger_test": SUPPORT_MERGER_TEST,
            "adapter": Path(support.__file__).resolve(),
            "adapter_test": SUPPORT_ADAPTER_TEST,
            "plan": SUPPORT_PLAN,
            "census": SUPPORT_CENSUS,
        },
    )
    receipt = _read_hashed_receipt(
        compatibility_receipt_path, "support CPU compatibility receipt"
    )
    _require_exact_receipt(
        receipt,
        schema="research_probe_admission.support_cpu_compatibility.v1",
        kind="support_cpu_compatibility",
        files={
            "consumer": consumer_path,
            "consumer_test": consumer_test_path,
            "merger": merger_path,
            "merger_test": merger_test_path,
            "adapter": adapter_path,
            "adapter_test": adapter_test_path,
            "plan": plan_path,
            "census": census_path,
            "generator": ADMISSION_ACCEPTANCE_DRIVER,
        },
        selectors={
            "consumer_validator": "validate_execution_plan",
            "model_free_finalizer": "materialize_legacy_shard_receipts",
            "downstream_validator": "merge_support_receipts",
        },
    )
    _require_flags(
        receipt,
        {
            "consumer_validator_ran": True,
            "model_free_finalizer_ran": True,
            "downstream_validator_accepted": True,
            "model_loaded": False,
            "gpu_used": False,
            "scientific_interpretation": False,
        },
    )
    verification = _publish_fresh_receipt(
        output_root,
        "support_cpu_verification.json",
        {
            "schema_version": "research_probe_admission.consumer.support_cpu.v2",
            "kind": "verified_support_cpu_compatibility_receipt",
            "compatibility_receipt": _file_identity(compatibility_receipt_path),
            "claim_boundary": "mechanics_only_model_free_cpu_receipt_validation",
        },
    )
    return StageEvidence(
        producer_binding="support_consumer",
        validator_binding="support_merger",
        output_files=(RegularFileBinding("support_cpu_verification", verification),),
        assertions={
            "production_entrypoint_resolved": True,
            "consumer_validator_ran": True,
            "model_free_finalizer_ran": True,
            "downstream_validator_accepted": True,
            "model_loaded": False,
            "gpu_used": False,
        },
        detail={
            "compatibility_receipt": str(
                compatibility_receipt_path.resolve(strict=True)
            ),
            "mode": "receipt_adapter",
        },
    )


def validate_support_bounded_terminal(
    *,
    terminal_path: Path,
    plan_path: Path,
    schedule: Mapping[str, Any],
    slot_roots: Sequence[Path],
    execution_identity: Mapping[str, Any],
    admission_bindings: BindingManifest,
    worker_path: Path,
    executable_path: Path,
    worker_argv: Sequence[str],
    worker_cwd: Path,
    worker_exit_receipts: Sequence[Path],
    runtime_receipts: Sequence[Path],
    expected_checkpoint: str,
    expected_config_fingerprint: str,
) -> dict[str, Any]:
    """Validate complete bounded terminal/runtime evidence; never legacy closure."""

    executable = _regular_leaf(executable_path, "worker executable")
    cwd = _regular_directory(worker_cwd, "worker cwd")
    if not worker_argv or any(
        not isinstance(item, str) or not item for item in worker_argv
    ):
        raise ConsumerAdmissionError("worker argv must be a nonempty string array")
    if expected_checkpoint == "" or expected_config_fingerprint == "":
        raise ConsumerAdmissionError("checkpoint and config fingerprint are mandatory")
    _require_production_identity(
        admission_bindings,
        execution_identity,
        plan_path,
        worker_path,
        executable,
        worker_argv,
        cwd,
        expected_checkpoint,
        expected_config_fingerprint,
    )
    sealed_plan = support.load_logical_plan(plan_path)
    bounded_context_ids = execution_identity.get("bounded_context_ids")
    if not isinstance(bounded_context_ids, list):
        raise ConsumerAdmissionError(
            "vertical execution identity lacks bounded context identifiers"
        )
    plan = support.project_logical_contexts(
        sealed_plan,
        context_ids=bounded_context_ids,
    )
    checked_schedule = support.validate_schedule(schedule, plan)
    if (
        len(slot_roots) != checked_schedule["slot_count"]
        or len(worker_exit_receipts) != len(slot_roots)
        or len(runtime_receipts) != len(slot_roots)
    ):
        raise ConsumerAdmissionError(
            "bounded terminal requires one terminal, exit, and runtime receipt per scheduled slot"
        )
    terminal = _read_hashed_receipt(terminal_path, "bounded terminal")
    forbidden = {
        "legacy_merger_closed",
        "legacy_merger",
        "merger_closure",
        "eight_shard_closure",
    }
    if (
        forbidden & set(terminal)
        or "legacy" in str(terminal.get("claim_boundary", "")).lower()
    ):
        raise ConsumerAdmissionError(
            "bounded terminal must not claim legacy eight-shard merger closure"
        )
    if (
        terminal.get("kind") != "bounded_terminal_materialization"
        or terminal.get("claim_boundary")
        != "bounded_live_mechanics_only_no_scientific_receipt_or_outcome_interpretation"
    ):
        raise ConsumerAdmissionError(
            "terminal is not the bounded mechanics-only materialization"
        )
    if (
        terminal.get("logical_plan_file_sha256") != plan.file_sha256
        or terminal.get("logical_plan_content_sha256") != plan.content_sha256
        or terminal.get("schedule_sha256") != checked_schedule["content_sha256"]
    ):
        raise ConsumerAdmissionError(
            "bounded terminal plan or schedule binding drifted"
        )
    expected_ids = [context.context_id for context in plan.contexts]
    if (
        terminal.get("context_count") != len(expected_ids)
        or terminal.get("context_ids") != expected_ids
        or len(set(expected_ids)) != len(expected_ids)
    ):
        raise ConsumerAdmissionError(
            "bounded terminal context denominator is incomplete or foreign"
        )
    observed_records: dict[str, Any] = {}
    slot_terminal_digests: list[str] = []
    slot_plan_fingerprints: list[str] = []
    runtime_identities: list[dict[str, Any]] = []
    worker_exit_identities: list[dict[str, Any]] = []
    slot_journal_identities: list[dict[str, Any]] = []
    for index, (slot_root, exit_path, runtime_path) in enumerate(
        zip(slot_roots, worker_exit_receipts, runtime_receipts, strict=True)
    ):
        diagnostics = ExecutionEvidenceJournal.inspect_diagnostics(slot_root)
        if diagnostics.snapshot.terminal is None or tuple(
            diagnostics.snapshot.completed_work_item_ids
        ) != tuple(checked_schedule["slots"][index]["context_ids"]):
            raise ConsumerAdmissionError(
                "bounded terminal requires a complete terminal slot journal"
            )
        expected_identity = support._slot_identity(
            execution_identity=execution_identity,
            schedule=checked_schedule,
            slot_index=index,
        )
        if diagnostics.snapshot.execution_identity_fingerprint != json_sha256(
            expected_identity
        ):
            raise ConsumerAdmissionError("bounded terminal slot journal is foreign")
        runtime = _read_hashed_receipt(runtime_path, "runtime receipt")
        if (
            runtime.get("kind") != "live_worker_runtime_identity"
            or runtime.get("claim_boundary")
            != "live_runtime_mechanics_only_no_support_or_model_mechanism_claim"
            or runtime.get("binding") != execution_identity
            or runtime.get("runtime_identity", {}).get("normalized_device") != "cuda:0"
        ):
            raise ConsumerAdmissionError(
                "runtime receipt is foreign or does not bind logical cuda:0"
            )
        if runtime["runtime_identity"].get("checkpoint") != expected_checkpoint:
            raise ConsumerAdmissionError(
                "runtime receipt checkpoint differs from the bound expectation"
            )
        if (
            runtime["runtime_identity"].get("config_fingerprint")
            != expected_config_fingerprint
        ):
            raise ConsumerAdmissionError(
                "runtime receipt config fingerprint differs from the bound expectation"
            )
        runtime_identities.append(_file_identity(runtime_path))
        exit_receipt = _read_hashed_receipt(exit_path, "worker exit receipt")
        expected_count = checked_schedule["slots"][index]["context_count"]
        if (
            exit_receipt.get("schema_version") != support.MECHANICS_SCHEMA_VERSION
            or exit_receipt.get("kind") != "worker_exit"
            or exit_receipt.get("claim_boundary")
            != "mechanics_only_no_exit_interpretation_or_automatic_retry"
            or exit_receipt.get("physical_slot_index") != index
            or exit_receipt.get("logical_plan_file_sha256") != plan.file_sha256
            or exit_receipt.get("schedule_sha256") != checked_schedule["content_sha256"]
            or exit_receipt.get("accepted_context_count") != expected_count
            or exit_receipt.get("missing_context_count") != 0
            or exit_receipt.get("journal_terminal") is not True
            or exit_receipt.get("return_code") != 0
            or exit_receipt.get("terminating_signal") is not None
            or exit_receipt.get("external_signal_sent") is not None
            or exit_receipt.get("temporary_paths") != []
            or exit_receipt.get("journal_inspection", {}).get("status") != "validated"
            or exit_receipt.get("command") != list(worker_argv)
            or exit_receipt.get("cwd") != str(cwd)
            or exit_receipt.get("executable") != _file_identity(executable)
        ):
            raise ConsumerAdmissionError(
                "worker-exit receipt does not close the exact bounded slot"
            )
        worker_exit_identities.append(_file_identity(exit_path))
        for record in diagnostics.records:
            if record.work_item_id in observed_records:
                raise ConsumerAdmissionError(
                    "bounded terminal duplicates a durable context record"
                )
            observed_records[record.work_item_id] = record
        slot_terminal_digests.append(
            str(diagnostics.snapshot.terminal["content_sha256"])
        )
        slot_plan_fingerprints.append(diagnostics.snapshot.plan_fingerprint)
        slot_journal_identities.append(
            {
                "root": str(slot_root.resolve(strict=True)),
                "plan_fingerprint": diagnostics.snapshot.plan_fingerprint,
                "terminal_content_sha256": diagnostics.snapshot.terminal[
                    "content_sha256"
                ],
                "execution_identity_fingerprint": diagnostics.snapshot.execution_identity_fingerprint,
            }
        )
    if set(observed_records) != set(expected_ids):
        raise ConsumerAdmissionError(
            "bounded terminal journal denominator is incomplete or foreign"
        )
    ordered_records = [observed_records[context_id] for context_id in expected_ids]
    record_digests = [record.record_digest for record in ordered_records]
    payload_fingerprints = [record.payload_fingerprint for record in ordered_records]
    if (
        terminal.get("record_digests") != record_digests
        or terminal.get("payload_fingerprints") != payload_fingerprints
        or terminal.get("slot_terminal_digests") != slot_terminal_digests
        or terminal.get("slot_plan_fingerprints") != slot_plan_fingerprints
    ):
        raise ConsumerAdmissionError(
            "bounded terminal digest arrays differ from journal diagnostics"
        )
    return {
        "schema_version": "research_probe_admission.consumer.support_vertical.v2",
        "kind": "bounded_terminal_validator",
        "terminal": _file_identity(terminal_path),
        "runtime_receipts": runtime_identities,
        "worker_exit_receipts": worker_exit_identities,
        "slot_journals": slot_journal_identities,
        "logical_plan_file_sha256": plan.file_sha256,
        "schedule_sha256": checked_schedule["content_sha256"],
        "durable_work_item_count": len(expected_ids),
        "legacy_merger_closed": False,
        "model_runtime_loaded": True,
        "scientific_interpretation": False,
        "claim_boundary": "bounded_terminal_mechanics_only_never_legacy_eight_shard_merger_closure",
    }


def build_support_vertical_evidence(
    *,
    output_root: Path,
    terminal_path: Path,
    plan_path: Path,
    schedule: Mapping[str, Any],
    slot_roots: Sequence[Path],
    execution_identity: Mapping[str, Any],
    admission_bindings: BindingManifest,
    worker_path: Path,
    executable_path: Path,
    worker_argv: Sequence[str],
    worker_cwd: Path,
    worker_exit_receipts: Sequence[Path],
    runtime_receipts: Sequence[Path],
    expected_checkpoint: str,
    expected_config_fingerprint: str,
) -> StageEvidence:
    validation = validate_support_bounded_terminal(
        terminal_path=terminal_path,
        plan_path=plan_path,
        schedule=schedule,
        slot_roots=slot_roots,
        execution_identity=execution_identity,
        admission_bindings=admission_bindings,
        worker_path=worker_path,
        executable_path=executable_path,
        worker_argv=worker_argv,
        worker_cwd=worker_cwd,
        worker_exit_receipts=worker_exit_receipts,
        runtime_receipts=runtime_receipts,
        expected_checkpoint=expected_checkpoint,
        expected_config_fingerprint=expected_config_fingerprint,
    )
    outputs = _publish_vertical_snapshot(
        output_root,
        validation,
        terminal_path,
        runtime_receipts,
        worker_exit_receipts,
        slot_roots,
    )
    return StageEvidence(
        producer_binding="support_worker",
        validator_binding="consumer_admission_adapter",
        output_files=tuple(
            RegularFileBinding(f"support_vertical_{index}", path)
            for index, path in enumerate(outputs)
        ),
        assertions={
            "production_entrypoint_executed": True,
            "model_runtime_loaded": True,
            "terminal_finalizer_completed": True,
            "downstream_validator_accepted": True,
            "durable_work_item_count": validation["durable_work_item_count"],
        },
        detail={
            "consumer": "natural_boundary_support",
            "legacy_merger_closed": False,
            "mode": "receipt_adapter_snapshot",
        },
    )


def build_crossover_cpu_evidence(
    *,
    output_root: Path,
    compatibility_receipt_path: Path,
    source_preflight_path: Path,
    accepted_evidence_path: Path,
    runner_path: Path,
    runner_test_path: Path,
    sealer_path: Path,
    finalizer_path: Path,
    finalizer_test_path: Path,
) -> StageEvidence:
    """Accept a completed crossover CPU compatibility receipt, never a replay marker."""

    _require_exact_paths(
        {
            "runner": runner_path,
            "runner_test": runner_test_path,
            "sealer": sealer_path,
            "finalizer": finalizer_path,
            "finalizer_test": finalizer_test_path,
            "source_preflight": source_preflight_path,
            "accepted_evidence": accepted_evidence_path,
        },
        {
            "runner": CROSSOVER_RUNNER,
            "runner_test": CROSSOVER_RUNNER_TEST,
            "sealer": CROSSOVER_SEALER,
            "finalizer": CROSSOVER_FINALIZER,
            "finalizer_test": CROSSOVER_FINALIZER_TEST,
            "source_preflight": CROSSOVER_SOURCE_PREFLIGHT,
            "accepted_evidence": CROSSOVER_ACCEPTED_EVIDENCE,
        },
    )
    receipt = _read_hashed_receipt(
        compatibility_receipt_path, "crossover CPU compatibility receipt"
    )
    _require_exact_receipt(
        receipt,
        schema="research_probe_admission.crossover_cpu_compatibility.v1",
        kind="crossover_cpu_compatibility",
        files={
            "runner": runner_path,
            "runner_test": runner_test_path,
            "sealer": sealer_path,
            "finalizer": finalizer_path,
            "finalizer_test": finalizer_test_path,
            "source_preflight": source_preflight_path,
            "accepted_evidence": accepted_evidence_path,
            "generator": ADMISSION_ACCEPTANCE_DRIVER,
        },
        selectors={
            "source_preflight_validator": "preflight_crossover_sources",
            "model_free_finalizer": "finalize",
            "downstream_validator": "validate_evidence",
        },
    )
    _require_flags(
        receipt,
        {
            "source_preflight_validated": True,
            "model_free_finalizer_ran": True,
            "downstream_validator_accepted": True,
            "model_loaded": False,
            "gpu_used": False,
            "scientific_interpretation": False,
        },
    )
    source_preflight = _read_hashed_receipt(
        source_preflight_path,
        "crossover source preflight",
        allow_trailing_newline=True,
    )
    production = source_preflight.get("model_free_production_path")
    full_runtime_cohort = (
        production.get("full_runtime_cohort")
        if isinstance(production, Mapping)
        else None
    )
    selected_factory_contexts = (
        production.get("selected_factory_contexts")
        if isinstance(production, Mapping)
        else None
    )
    if (
        source_preflight.get("schema_version")
        != "s_k10_h20_crossover_event.v1.source_preflight.v1"
        or source_preflight.get("status") != "passed"
        or source_preflight.get("phase") != "preseal_model_free_production"
        or source_preflight.get("receipt_independent") is not True
        or source_preflight.get("gpu_used") is not False
        or source_preflight.get("model_loaded") is not False
        or source_preflight.get("backend_session_opened") is not False
        or source_preflight.get("output_root_created") is not False
        or not isinstance(production, Mapping)
        or production.get("status") != "passed"
        or production.get("gpu_used") is not False
        or production.get("model_loaded") is not False
        or production.get("model_loader_called") is not False
        or production.get("output_root_created") is not False
        or not isinstance(full_runtime_cohort, Mapping)
        or full_runtime_cohort.get("status") != "passed"
        or full_runtime_cohort.get("event_count") != 11
        or not isinstance(selected_factory_contexts, list)
        or len(selected_factory_contexts) != 3
    ):
        raise ConsumerAdmissionError(
            "crossover source preflight is not the completed model-free production path"
        )
    if receipt.get("replay_status") in {"not_replayed", "read_only_wrapper"}:
        raise ConsumerAdmissionError(
            "crossover compatibility receipt says the finalizer was not replayed"
        )
    verification = _publish_fresh_receipt(
        output_root,
        "crossover_cpu_verification.json",
        {
            "schema_version": "research_probe_admission.consumer.crossover_cpu.v2",
            "kind": "verified_crossover_cpu_compatibility_receipt",
            "compatibility_receipt": _file_identity(compatibility_receipt_path),
            "claim_boundary": "mechanics_only_model_free_cpu_receipt_validation",
        },
    )
    return StageEvidence(
        producer_binding="crossover_runner",
        validator_binding="crossover_finalizer",
        output_files=(RegularFileBinding("crossover_cpu_verification", verification),),
        assertions={
            "production_entrypoint_resolved": True,
            "consumer_validator_ran": True,
            "model_free_finalizer_ran": True,
            "downstream_validator_accepted": True,
            "model_loaded": False,
            "gpu_used": False,
        },
        detail={
            "compatibility_receipt": str(
                compatibility_receipt_path.resolve(strict=True)
            ),
            "mode": "receipt_adapter",
        },
    )


def _require_exact_receipt(
    receipt: Mapping[str, Any],
    *,
    schema: str,
    kind: str,
    files: Mapping[str, Path],
    selectors: Mapping[str, str],
) -> None:
    if (
        receipt.get("schema_version") != schema
        or receipt.get("kind") != kind
        or receipt.get("exit_code") != 0
    ):
        raise ConsumerAdmissionError(
            "compatibility receipt schema, kind, or zero exit status differs"
        )
    actual_files = receipt.get("files")
    if not isinstance(actual_files, Mapping) or set(actual_files) != set(files):
        raise ConsumerAdmissionError("compatibility receipt source file set differs")
    for name, path in files.items():
        if actual_files[name] != _file_identity(path):
            raise ConsumerAdmissionError(
                f"compatibility receipt {name} raw source binding drifted"
            )
    if receipt.get("selectors") != dict(selectors):
        raise ConsumerAdmissionError(
            "compatibility receipt command/test selectors differ"
        )
    _require_command_results(receipt, selectors)


def _require_command_results(
    receipt: Mapping[str, Any], selectors: Mapping[str, str]
) -> None:
    """Require immutable, replayable per-command receipts; never a bare boolean."""

    results = receipt.get("command_results")
    if not isinstance(results, Mapping) or set(results) != set(selectors):
        raise ConsumerAdmissionError(
            "compatibility receipt command_results set differs"
        )
    for name, selector in selectors.items():
        result = results[name]
        if not isinstance(result, Mapping) or set(result) != {
            "executable",
            "cwd",
            "argv",
            "selector",
            "return_code",
            "passed",
            "stdout",
            "stderr",
            "receipt",
            "generator",
        }:
            raise ConsumerAdmissionError("command result schema differs")
        if (
            result["selector"] != selector
            or result["return_code"] != 0
            or isinstance(result["passed"], bool)
            or not isinstance(result["passed"], int)
            or result["passed"] < 1
        ):
            raise ConsumerAdmissionError(
                "command result selector, exit, or passed count differs"
            )
        executable = _identity_to_live_file(result["executable"], "command executable")
        if not executable.is_absolute():
            raise ConsumerAdmissionError("command executable must be absolute")
        _regular_directory(result["cwd"], "command cwd")
        argv = result["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise ConsumerAdmissionError("command argv must be a nonempty string array")
        if argv[0] != str(executable):
            raise ConsumerAdmissionError(
                "command argv does not invoke the bound executable"
            )
        generator = _identity_to_live_file(result["generator"], "command generator")
        for field in ("stdout", "stderr"):
            _identity_to_live_file(result[field], f"command {field}")
        command_receipt_path = _identity_to_live_file(
            result["receipt"], "command receipt"
        )
        command_receipt = _read_hashed_receipt(command_receipt_path, "command receipt")
        expected_receipt = {
            "schema_version": "research_probe_admission.command_execution.v1",
            "kind": "command_execution",
            "generator": _file_identity(generator),
            "executable": result["executable"],
            "cwd": result["cwd"],
            "argv": argv,
            "selector": selector,
            "return_code": 0,
            "passed": result["passed"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "claim_boundary": "cpu_command_mechanics_only_no_scientific_interpretation",
        }
        expected_receipt["content_sha256"] = json_sha256(expected_receipt)
        if command_receipt != expected_receipt:
            raise ConsumerAdmissionError("durable command receipt projection differs")


def _command_binding_requests(
    receipt_path: Path, prefix: str
) -> tuple[RegularFileBinding | AbsoluteExecutableBinding, ...]:
    """Promote every command attestation leaf into the live input manifest."""

    receipt = _read_hashed_receipt(receipt_path, f"{prefix} compatibility receipt")
    results = receipt.get("command_results")
    if not isinstance(results, Mapping):
        raise ConsumerAdmissionError("compatibility receipt lacks command_results")
    bindings: list[RegularFileBinding | AbsoluteExecutableBinding] = []
    for command_name in sorted(results):
        result = results[command_name]
        if not isinstance(result, Mapping):
            raise ConsumerAdmissionError("compatibility command result is malformed")
        executable = result.get("executable")
        if not isinstance(executable, Mapping) or not isinstance(
            executable.get("path"), str
        ):
            raise ConsumerAdmissionError(
                "compatibility command executable is malformed"
            )
        binding_prefix = f"{prefix}_command_{command_name}"
        bindings.append(
            AbsoluteExecutableBinding(
                f"{binding_prefix}_executable", Path(executable["path"])
            )
        )
        for field in ("stdout", "stderr", "receipt", "generator"):
            identity = result.get(field)
            if not isinstance(identity, Mapping) or not isinstance(
                identity.get("path"), str
            ):
                raise ConsumerAdmissionError(
                    f"compatibility command {field} identity is malformed"
                )
            bindings.append(
                RegularFileBinding(f"{binding_prefix}_{field}", Path(identity["path"]))
            )
    return tuple(bindings)


def _require_exact_paths(
    supplied: Mapping[str, Path], expected: Mapping[str, Path]
) -> None:
    if set(supplied) != set(expected):
        raise ConsumerAdmissionError("consumer path role set differs")
    for name, path in supplied.items():
        if _regular_leaf(path, name) != _regular_leaf(expected[name], name):
            raise ConsumerAdmissionError(
                f"{name} is not the current production consumer path"
            )


def _identity_to_live_file(value: Any, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "raw_sha256",
        "byte_count",
    }:
        raise ConsumerAdmissionError(f"{label} identity schema differs")
    path = _regular_leaf(Path(str(value["path"])), label)
    if value != _file_identity(path):
        raise ConsumerAdmissionError(f"{label} bytes differ from receipt identity")
    return path


def _regular_directory(value: Any, label: str) -> Path:
    path = Path(str(value))
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ConsumerAdmissionError(f"{label} is missing") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or not path.is_absolute()
    ):
        raise ConsumerAdmissionError(
            f"{label} must be an absolute non-symlink directory"
        )
    return path.resolve(strict=True)


def _require_flags(receipt: Mapping[str, Any], expected: Mapping[str, bool]) -> None:
    for name, value in expected.items():
        if receipt.get(name) is not value:
            raise ConsumerAdmissionError(
                f"compatibility receipt flag {name} is not {str(value).lower()}"
            )


def _require_production_identity(
    manifest: BindingManifest,
    execution_identity: Mapping[str, Any],
    plan_path: Path,
    worker_path: Path,
    executable_path: Path,
    worker_argv: Sequence[str],
    worker_cwd: Path,
    expected_checkpoint: str,
    expected_config_fingerprint: str,
) -> None:
    """Bind the exact current worker schema to independently captured inputs."""

    if not isinstance(manifest, BindingManifest) or not isinstance(
        execution_identity, Mapping
    ):
        raise ConsumerAdmissionError(
            "vertical admission requires a typed manifest and execution identity"
        )
    entries = {str(item["name"]): item for item in manifest.to_mapping()["bindings"]}
    required = {
        "support_consumer",
        "admission_core",
        "evidence_journal_core",
        "json_values_core",
        "support_worker",
        "support_adapter",
        "support_plan",
        "support_census",
        "support_infer_config",
        "support_adapter_tensor",
        "support_embedding_delta",
        "support_base_model",
        "support_python",
        "support_consumer_runtime_source",
        "support_embedding_source_gate",
        "support_execution_identity",
        "support_worker_argv",
        "support_worker_cwd",
        "support_expected_runtime",
    }
    if not required <= set(entries):
        raise ConsumerAdmissionError(
            "vertical admission manifest lacks production source/model/runtime bindings"
        )
    _require_admission_runtime_bindings(entries)

    worker = _regular_leaf(worker_path, "worker")
    plan = _regular_leaf(plan_path, "plan")
    if worker != SUPPORT_WORKER.resolve(strict=True):
        raise ConsumerAdmissionError(
            "vertical worker is not the current production worker"
        )
    if plan != SUPPORT_PLAN.resolve(strict=True):
        raise ConsumerAdmissionError("vertical plan is not the sealed support plan")
    if _regular_leaf(SUPPORT_CONSUMER, "support consumer") != Path(
        str(entries["support_consumer"].get("path"))
    ):
        raise ConsumerAdmissionError(
            "vertical consumer is not the current production consumer"
        )

    expected_keys = {
        "consumer",
        "consumer_runtime_source",
        "adapter",
        "worker",
        "logical_plan",
        "bounded_context_ids",
        "schedule_sha256",
        "census",
        "infer_config",
        "model",
        "embedding_source_gate",
        "runtime_policy",
    }
    if set(execution_identity) != expected_keys:
        raise ConsumerAdmissionError(
            "vertical execution identity schema differs from the production worker"
        )
    if _manifest_value(entries, "support_execution_identity") != dict(
        execution_identity
    ):
        raise ConsumerAdmissionError(
            "vertical execution identity differs from admission manifest"
        )
    if _manifest_value(entries, "support_consumer_runtime_source") != dict(
        execution_identity["consumer_runtime_source"]
    ):
        raise ConsumerAdmissionError("consumer runtime identity is not manifest-bound")
    if _manifest_value(entries, "support_embedding_source_gate") != dict(
        execution_identity["embedding_source_gate"]
    ):
        raise ConsumerAdmissionError("embedding source gate is not manifest-bound")
    if _manifest_value(entries, "support_worker_argv") != list(worker_argv):
        raise ConsumerAdmissionError("worker argv is not manifest-bound")
    if _manifest_value(entries, "support_worker_cwd") != str(worker_cwd):
        raise ConsumerAdmissionError("worker cwd is not manifest-bound")
    if _manifest_value(entries, "support_expected_runtime") != {
        "checkpoint": expected_checkpoint,
        "config_fingerprint": expected_config_fingerprint,
    }:
        raise ConsumerAdmissionError("expected live runtime is not manifest-bound")

    regular_refs = {
        "support_consumer": execution_identity["consumer"],
        "support_worker": execution_identity["worker"],
        "support_adapter": execution_identity["adapter"],
        "support_census": execution_identity["census"],
        "support_infer_config": execution_identity["infer_config"],
        "support_adapter_tensor": execution_identity["model"]["adapter_tensor"],
        "support_embedding_delta": execution_identity["model"]["embedding_delta"],
    }
    for name, reference in regular_refs.items():
        if not isinstance(reference, Mapping):
            raise ConsumerAdmissionError(f"{name} execution reference is malformed")
        expected_ref = _manifest_regular_reference(entries, name)
        if dict(reference) != expected_ref:
            raise ConsumerAdmissionError(f"{name} differs from admission binding")

    sealed = support.load_logical_plan(plan)
    logical_plan = execution_identity["logical_plan"]
    if logical_plan != {
        "path": str(plan),
        "file_sha256": sealed.file_sha256,
        "content_sha256": sealed.content_sha256,
    }:
        raise ConsumerAdmissionError("logical plan identity differs from sealed input")
    plan_binding = entries["support_plan"]
    if (
        plan_binding.get("kind") != "regular_file"
        or plan_binding.get("path") != str(plan)
        or plan_binding.get("sha256") != sealed.file_sha256
    ):
        raise ConsumerAdmissionError("sealed plan is not exactly admission-bound")

    model = execution_identity["model"]
    if not isinstance(model, Mapping) or set(model) != {
        "base_model",
        "adapter_tensor",
        "embedding_delta",
    }:
        raise ConsumerAdmissionError("production model identity schema differs")
    _require_base_model_binding(entries["support_base_model"], model["base_model"])
    _require_runtime_source_bindings(
        entries, execution_identity["consumer_runtime_source"]
    )
    _require_source_gate_bindings(entries, execution_identity["embedding_source_gate"])

    executable_binding = entries["support_python"]
    if (
        executable_binding.get("kind") != "absolute_executable"
        or executable_binding.get("path") != str(executable_path)
        or executable_binding.get("sha256")
        != _file_identity(executable_path)["raw_sha256"]
    ):
        raise ConsumerAdmissionError("worker executable differs from admission binding")
    if list(worker_argv[:3]) != [
        str(executable_path),
        "-m",
        "scripts.research.run_resumable_natural_boundary_support_shard",
    ]:
        raise ConsumerAdmissionError(
            "worker argv does not invoke the bound production worker"
        )

    runtime_policy = execution_identity["runtime_policy"]
    if (
        not isinstance(runtime_policy, Mapping)
        or runtime_policy.get("logical_device") != "cuda:0"
        or runtime_policy.get("physical_slot_count") != 1
        or runtime_policy.get("physical_slot_index") != 0
        or not isinstance(runtime_policy.get("cuda_visible_devices"), str)
        or not runtime_policy["cuda_visible_devices"]
        or "," in runtime_policy["cuda_visible_devices"]
    ):
        raise ConsumerAdmissionError(
            "worker runtime policy is not single-device cuda:0"
        )


def _manifest_value(entries: Mapping[str, Mapping[str, Any]], name: str) -> Any:
    entry = entries[name]
    if entry.get("kind") != "strict_value":
        raise ConsumerAdmissionError(f"{name} must be a strict-value binding")
    return entry.get("value")


def _manifest_regular_reference(
    entries: Mapping[str, Mapping[str, Any]], name: str
) -> dict[str, Any]:
    entry = entries[name]
    if entry.get("kind") != "regular_file":
        raise ConsumerAdmissionError(f"{name} must be a regular-file binding")
    return {"path": str(entry["path"]), "sha256": str(entry["sha256"])}


def _require_admission_runtime_bindings(
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind the verifier implementation that evaluates and publishes admission."""

    expected_paths = {
        "admission_core": ADMISSION_CORE,
        "evidence_journal_core": EVIDENCE_JOURNAL_CORE,
        "json_values_core": JSON_VALUES_CORE,
    }
    for name, path in expected_paths.items():
        if name not in entries:
            raise ConsumerAdmissionError(
                f"vertical admission manifest lacks {name} verifier binding"
            )
        expected = {
            "path": str(_regular_leaf(path, name)),
            "sha256": support.file_sha256(path),
        }
        if _manifest_regular_reference(entries, name) != expected:
            raise ConsumerAdmissionError(
                f"vertical admission {name} verifier identity drifted"
            )


def _require_base_model_binding(
    binding: Mapping[str, Any], model_identity: Any
) -> None:
    if binding.get("kind") != "directory_tree" or not isinstance(
        model_identity, Mapping
    ):
        raise ConsumerAdmissionError("base model directory binding is malformed")
    files = model_identity.get("files")
    if (
        model_identity.get("path") != binding.get("path")
        or not isinstance(files, Mapping)
        or model_identity.get("file_count") != len(files)
    ):
        raise ConsumerAdmissionError(
            "base model denominator differs from admission binding"
        )
    bound_files = {
        str(item["relative_path"]): str(item["sha256"])
        for item in binding.get("files", [])
    }
    observed_files = {
        str(name): str(reference["sha256"])
        for name, reference in files.items()
        if isinstance(reference, Mapping)
    }
    if bound_files != observed_files:
        raise ConsumerAdmissionError(
            "base model file bytes differ from admission binding"
        )
    expected_aggregate = json_sha256(
        [
            {"name": name, "sha256": sha256}
            for name, sha256 in sorted(observed_files.items())
        ]
    )
    if model_identity.get("aggregate_sha256") != expected_aggregate:
        raise ConsumerAdmissionError("base model aggregate identity differs")


def _require_runtime_source_bindings(
    entries: Mapping[str, Mapping[str, Any]], runtime_identity: Any
) -> None:
    if not isinstance(runtime_identity, Mapping):
        raise ConsumerAdmissionError("consumer runtime source identity is malformed")
    tree = runtime_identity.get("tree")
    if not isinstance(tree, Mapping):
        raise ConsumerAdmissionError("consumer runtime tree identity is malformed")
    root = _regular_directory(tree.get("root"), "consumer runtime source root")
    live_paths = sorted(root.rglob("*.py"))
    if any(path.is_symlink() or not path.is_file() for path in live_paths):
        raise ConsumerAdmissionError(
            "consumer runtime tree contains a non-regular Python file"
        )
    bound = {
        str(entry["path"]): str(entry["sha256"])
        for name, entry in entries.items()
        if name.startswith("support_runtime_source_")
        and entry.get("kind") == "regular_file"
    }
    live = {
        str(path.resolve(strict=True)): support.file_sha256(path) for path in live_paths
    }
    if bound != live:
        raise ConsumerAdmissionError(
            "consumer runtime Python denominator is not admission-bound"
        )
    aggregate_rows = [
        {
            "path": path.relative_to(ACTIVE_RESEARCH_PROBES_ROOT).as_posix(),
            "sha256": live[str(path.resolve(strict=True))],
        }
        for path in live_paths
    ]
    if tree != {
        "root": str(root),
        "algorithm": "sorted_relative_path_and_sha256_json.v1",
        "python_file_count": len(aggregate_rows),
        "aggregate_sha256": json_sha256(aggregate_rows),
    }:
        raise ConsumerAdmissionError("consumer runtime tree aggregate differs")
    critical_files = runtime_identity.get("critical_files")
    if not isinstance(critical_files, Mapping):
        raise ConsumerAdmissionError(
            "consumer runtime critical-file identity is malformed"
        )
    for reference in critical_files.values():
        if not isinstance(reference, Mapping) or bound.get(
            str(reference.get("path"))
        ) != reference.get("sha256"):
            raise ConsumerAdmissionError(
                "consumer runtime critical file is not admission-bound"
            )


def _require_source_gate_bindings(
    entries: Mapping[str, Mapping[str, Any]], source_gate: Any
) -> None:
    if not isinstance(source_gate, Mapping) or not isinstance(
        source_gate.get("files"), Mapping
    ):
        raise ConsumerAdmissionError("embedding source-gate identity is malformed")
    bound = {
        str(entry["path"]): str(entry["sha256"])
        for name, entry in entries.items()
        if name.startswith("support_source_gate_")
        and entry.get("kind") == "regular_file"
    }
    expected = {
        str(reference["path"]): str(reference["sha256"])
        for reference in source_gate["files"].values()
        if isinstance(reference, Mapping)
    }
    if not expected or bound != expected:
        raise ConsumerAdmissionError(
            "embedding source-gate files are not admission-bound"
        )


def _file_identity(path: Path) -> dict[str, Any]:
    candidate = _regular_leaf(path, "bound file")
    raw = candidate.read_bytes()
    return {
        "path": str(candidate),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _read_hashed_receipt(
    path: Path, label: str, *, allow_trailing_newline: bool = False
) -> dict[str, Any]:
    candidate = _regular_leaf(path, label)
    raw = candidate.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConsumerAdmissionError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict):
        raise ConsumerAdmissionError(f"{label} is not a JSON object")
    try:
        canonical = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConsumerAdmissionError(f"{label} is not strict canonical JSON") from exc
    accepted_bytes = (
        (canonical, canonical + b"\n") if allow_trailing_newline else (canonical,)
    )
    if raw not in accepted_bytes:
        raise ConsumerAdmissionError(f"{label} bytes are not canonical JSON")
    hash_field = "content_sha256" if "content_sha256" in value else "self_sha256"
    digest = value.get(hash_field)
    body = dict(value)
    body.pop(hash_field, None)
    if not isinstance(digest, str) or digest != json_sha256(body):
        raise ConsumerAdmissionError(f"{label} content hash is invalid")
    validate_json_value(value)
    return value


def _publish_fresh_receipt(root: Path, filename: str, body: Mapping[str, Any]) -> Path:
    payload = dict(body)
    validate_json_value(payload)
    payload["content_sha256"] = json_sha256(payload)
    expected = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path = root / filename
    if root.exists() or root.is_symlink():
        try:
            metadata = root.lstat()
            children = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise ConsumerAdmissionError(
                "existing receipt root cannot be inspected"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or [item.name for item in children] != [filename]
        ):
            raise ConsumerAdmissionError(
                "existing receipt root is not the exact single-file publication shape"
            )
        existing = _regular_leaf(path, "existing verification receipt")
        if existing.read_bytes() != expected:
            raise ConsumerAdmissionError(
                "existing verification receipt differs from exact publication bytes"
            )
        return existing
    root.mkdir(parents=True, exist_ok=False)
    publish_json_exclusive(path, payload)
    return path


def _publish_vertical_snapshot(
    root: Path,
    validation: Mapping[str, Any],
    terminal: Path,
    runtimes: Sequence[Path],
    exits: Sequence[Path],
    slot_roots: Sequence[Path],
) -> tuple[Path, ...]:
    """Durably snapshot every raw JSON artifact with exact retry recovery."""

    sources: list[tuple[str, Path]] = [("terminal.json", terminal)]
    sources.extend(
        (f"runtime/{index:03d}.json", path) for index, path in enumerate(runtimes)
    )
    sources.extend(
        (f"worker-exit/{index:03d}.json", path) for index, path in enumerate(exits)
    )
    for slot_index, journal_root in enumerate(slot_roots):
        for source in sorted(
            journal_root.rglob("*.json"),
            key=lambda item: item.relative_to(journal_root).as_posix(),
        ):
            relative = source.relative_to(journal_root).as_posix()
            sources.append((f"journal/{slot_index:03d}/{relative}", source))

    expected_relative_paths = {name for name, _ in sources}
    expected_relative_paths.add("support_vertical_verification.json")
    _prepare_exact_snapshot_root(root, expected_relative_paths)

    copied: list[Path] = []
    identities: list[dict[str, Any]] = []
    for name, source in sources:
        source_file = _regular_leaf(source, "vertical evidence source")
        target = root / name
        _publish_canonical_copy(target, source_file)
        copied.append(target)
        identities.append(
            {
                "name": name,
                "source": _file_identity(source_file),
                "snapshot": _file_identity(target),
            }
        )
    snapshot = {
        "schema_version": "research_probe_admission.consumer.support_vertical_snapshot.v1",
        "kind": "flat_bound_evidence_snapshot",
        "validation": dict(validation),
        "files": identities,
        "claim_boundary": "mechanics_only_snapshot_of_terminal_runtime_exit_and_journal_evidence",
    }
    snapshot["content_sha256"] = json_sha256(snapshot)
    snapshot_path = root / "support_vertical_verification.json"
    _publish_json_idempotent(snapshot_path, snapshot)
    _prepare_exact_snapshot_root(root, expected_relative_paths, require_complete=True)
    return (snapshot_path, *copied)


def _prepare_exact_snapshot_root(
    root: Path,
    expected_relative_paths: set[str],
    *,
    require_complete: bool = False,
) -> None:
    if root.is_symlink():
        raise ConsumerAdmissionError("vertical snapshot root must not be a symlink")
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
    if not root.is_dir():
        raise ConsumerAdmissionError("vertical snapshot root must be a directory")
    observed: set[str] = set()
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if path.is_symlink():
            raise ConsumerAdmissionError("vertical snapshot contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ConsumerAdmissionError(
                "vertical snapshot contains a non-regular file"
            )
        observed.add(path.relative_to(root).as_posix())
    if not observed <= expected_relative_paths:
        raise ConsumerAdmissionError("vertical snapshot contains an unexpected output")
    if require_complete and observed != expected_relative_paths:
        raise ConsumerAdmissionError(
            "vertical snapshot output denominator is incomplete"
        )


def _publish_canonical_copy(target: Path, source: Path) -> None:
    try:
        value = load_canonical_json(source)
    except ArtifactContractError as exc:
        raise ConsumerAdmissionError(
            "vertical evidence source is not canonical JSON"
        ) from exc
    _publish_json_idempotent(target, value)


def _publish_json_idempotent(target: Path, value: Any) -> None:
    expected = canonical_json_bytes(value)
    if target.exists() or target.is_symlink():
        existing = _regular_leaf(target, "existing vertical snapshot output")
        if existing.read_bytes() != expected:
            raise ConsumerAdmissionError("existing vertical snapshot output differs")
        return
    try:
        publish_json_exclusive(target, value)
    except ArtifactContractError as exc:
        if target.exists() and not target.is_symlink():
            existing = _regular_leaf(target, "recovered vertical snapshot output")
            if existing.read_bytes() == expected:
                return
        raise ConsumerAdmissionError("vertical snapshot publication failed") from exc


def _regular_leaf(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise ConsumerAdmissionError(f"{label} path must be a Path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ConsumerAdmissionError(f"{label} is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ConsumerAdmissionError(
            f"{label} must be a declared regular non-symlink file"
        )
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ConsumerAdmissionError(f"{label} cannot be resolved") from exc


__all__ = [
    "ConsumerAdmissionError",
    "build_crossover_cpu_evidence",
    "build_support_cpu_evidence",
    "build_support_vertical_evidence",
    "crossover_binding_requests",
    "support_binding_requests",
    "validate_support_bounded_terminal",
]

"""Model-bearing training session lifetime for the V1 supervised smoke.

Design decision 8 of ``decompose-coordexp-swift-training-orchestration``: this
module owns the mutable, model-bearing lifetime that begins once the control
plane has admitted the model-free inputs -- the lifecycle counters, the
Accelerator/model/adapter/embedding/loss/optimizer/runtime assembly, cache
hydration, exact-resume admission/restoration/publication, the eval/checkpoint/
final handlers, the forward-input-provider lifetime, run finalization, and the
profile-sync policy reset.

``src/training/pipeline.py`` keeps only the public facade; it composes this
module, ``execution_plan.py``, ``control_plane.py``, and ``cache_workflow.py``.
Nothing below this owner may import it back: ``session.py`` may compose every
lower owner but must never be imported by one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any

import torch

try:
    from accelerate import Accelerator
    from accelerate.utils import broadcast_object_list
except ImportError:  # pragma: no cover - exercised only in stripped environments.
    Accelerator = None  # type: ignore[assignment]
    broadcast_object_list = None  # type: ignore[assignment]

from src.adapters import (
    build_adapter_setup_plan,
    load_default_adapter_source_gate_evidence,
    setup_dora_adapter,
)
from src.artifacts import CheckpointWriter, RunWriter
from src.artifacts.identity import base_model_weight_identity
from src.artifacts.checkpoint_payload import (
    build_inference_checkpoint_payload_identity,
)
from src.artifacts.provenance import (
    collect_execution_provenance,
    require_mapped_native_execution_attestation,
    require_pinned_runtime_baseline,
)
from src.artifacts.run_writer import admit_exact_resume_checkpoint_publication
from src.artifacts.resources import (
    collect_resource_snapshot,
    converge_rank_cpu_resources,
    merge_rank_cpu_resource_receipts,
    rank_cpu_resources_from_metric_rows,
)
from src.artifacts.training_state import (
    TRAINING_STATE_DIRECTORY,
    TRAINING_STATE_MANIFEST,
    TrainingStateExpectations,
    TrainingStatePublicationPlan,
    admit_training_state,
    build_resume_compatibility_projection,
    capture_runtime_state_expectations,
    load_training_state_manifest,
)
from src.common.errors import RuntimeContractError
from src.config.models import ForwardInputProviderMode, RunDirectory
from src.config.paths import resolve_run_directory
from src.config.resolve import resolve_qwen_runtime_controls
from src.eval import ForwardEvalRunner
from src.eval.forward import (
    EVAL_REDUCTION_REPLICATED,
)
from src.losses import LossRunner, build_token_vocabulary_groups
from src.optim import (
    build_optimizer_and_scheduler,
    build_optimizer_group_plan,
    build_scheduler_plan,
    build_trainable_surface_receipt,
)
from src.qwen import (
    build_default_special_token_selection,
    load_qwen_components,
)
from src.qwen.special_token_embeddings import (
    load_default_special_token_embedding_source_gate_evidence,
    install_special_token_embedding_deltas,
    load_special_token_embedding_deltas,
)
from src.qwen.forward import (
    set_profile_sync_timing_policy as set_qwen_profile_sync_timing_policy,
)
from src.runtime import (
    TrainRuntime,
    validate_accelerator_runtime,
)
from src.training.schedule import ResolvedStepSchedule, resolve_planned_step_schedule
from src.training.pack_cache import (
    load_rank_micro_steps_from_cache,
)
from src.training.forward_input_provider import (
    ResolvedForwardInputProviderMode,
    build_forward_input_provider,
    resolve_forward_input_provider_mode,
)
from src.training.exact_resume import (
    RankCudaDeviceBinding,
    build_exact_resume_identities,
    build_exact_resume_topology_identity,
    capture_rank_rng_snapshot,
    prepare_distributed_exact_resume_contribution,
    publish_distributed_exact_resume,
    restore_distributed_exact_resume,
)
from src.training.supervised_trainer import (
    CompletedStepObservation,
    SupervisedMicroStep,
    SupervisedTrainer,
    set_profile_sync_timing_policy as set_trainer_profile_sync_timing_policy,
)
from src.training import cache_workflow, control_plane, execution_plan, reporting


BEST_EVAL_SELECTOR_NAME = "acc_top1"

_EVAL_RESOURCE_OBSERVATION_SCOPE = (
    "process_lifetime_high_water_observed_after_evaluation"
)
_STEADY_STATE_DURATION_SCOPE = "sum_of_accepted_all_rank_max_step_durations"
_EVALUATION_DURATION_SCOPE = "sum_of_all_rank_max_evaluation_event_durations"


@dataclass(frozen=True)
class RunIdentity:
    """The rank-zero run owner admitted before any model weight is loaded.

    Design decision 8 names this record in the ``TrainingSession`` constructor.
    It carries exactly the values ``_initialize_model_free_run_owner`` already
    returned as a positional tuple; it adds no field and abstracts no storage.
    """

    run_directory: RunDirectory
    run_id: str
    run_segment_id: str
    writer: RunWriter | None
    measurement_warmup_steps: int
    provenance: Mapping[str, Any] | None
    continuation_lineage: Mapping[str, Any] | None


def _resolve_profile_sync_timing_selector() -> dict[str, bool | str]:
    """Mirror the exact training/Qwen timing gate without recording raw env."""

    return {
        "enabled": os.environ.get(cache_workflow._PROFILE_SYNC_TIMINGS_ENV) == "1",
        "source": cache_workflow._environment_selector_source(cache_workflow._PROFILE_SYNC_TIMINGS_ENV),
    }


def _begin_run_phase(
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
    phase: str,
    *,
    started_at: str | None = None,
    started_monotonic: float | None = None,
    resources: Mapping[str, Any] | None = None,
) -> None:
    if lifecycle.get("active_phase") is not None:
        raise RuntimeContractError(
            "training phase lifecycle already has an active phase",
            code="runtime.phase_already_active",
            context={
                "active_phase": lifecycle.get("active_phase"),
                "requested_phase": phase,
            },
        )
    resolved_started_monotonic = (
        time.monotonic() if started_monotonic is None else started_monotonic
    )
    if writer is not None:
        writer.begin_phase(
            phase,
            started_at=cache_workflow._utc_now() if started_at is None else started_at,
            resources=collect_resource_snapshot() if resources is None else resources,
        )
    lifecycle["active_phase"] = phase
    lifecycle["phase_started_monotonic"] = resolved_started_monotonic


def _finish_run_phase(
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
    phase: str,
    *,
    completed_monotonic: float | None = None,
    status: str = "completed",
    accepted_measured_steps: int | None = None,
    expected_measured_steps: int | None = None,
    rank_resources: Mapping[str, Any] | None = None,
    rank_details: Mapping[str, Any] | None = None,
) -> None:
    if lifecycle.get("active_phase") != phase:
        raise RuntimeContractError(
            "training phase lifecycle can finish only its active phase",
            code="runtime.phase_not_active",
            context={
                "active_phase": lifecycle.get("active_phase"),
                "requested_phase": phase,
            },
        )
    started = lifecycle.get("phase_started_monotonic")
    if not isinstance(started, (int, float)):
        raise RuntimeContractError(
            "training phase lifecycle has no monotonic start",
            code="runtime.phase_start_missing",
            context={"phase": phase},
        )
    resolved_completed_monotonic = (
        time.monotonic() if completed_monotonic is None else completed_monotonic
    )
    duration = max(0.0, float(resolved_completed_monotonic) - float(started))
    captured = lifecycle.get("phase_rank_receipts", {}).get(phase)
    if isinstance(captured, Mapping):
        if rank_resources is None and isinstance(
            captured.get("rank_resources"), Mapping
        ):
            rank_resources = captured["rank_resources"]
        if rank_details is None and isinstance(captured.get("rank_details"), Mapping):
            rank_details = captured["rank_details"]
    if writer is not None:
        writer.finish_phase(
            phase,
            status=status,
            completed_at=cache_workflow._utc_now(),
            duration_seconds=duration,
            resources=collect_resource_snapshot(),
            accepted_measured_steps=accepted_measured_steps,
            expected_measured_steps=expected_measured_steps,
            rank_resources=rank_resources,
            rank_details=rank_details,
        )
    lifecycle["active_phase"] = None
    lifecycle["phase_started_monotonic"] = None


def _phase_receipt_sink(
    lifecycle: dict[str, Any], phase: str
) -> Callable[[Mapping[str, Any]], None]:
    def capture(receipt: Mapping[str, Any]) -> None:
        receipts = lifecycle.setdefault("phase_rank_receipts", {})
        if phase in receipts:
            raise RuntimeContractError(
                "training phase rank receipt was captured more than once",
                code="runtime.phase_receipt_duplicate",
                context={"phase": phase},
            )
        receipts[phase] = dict(receipt)

    return capture


def _phase_rank_details_sink(
    lifecycle: dict[str, Any], phase: str
) -> Callable[[Mapping[str, Any]], None]:
    """Add one post-assembly rank detail receipt to an existing phase receipt."""

    def capture(receipt: Mapping[str, Any]) -> None:
        phase_receipt = lifecycle.get("phase_rank_receipts", {}).get(phase)
        rank_details = receipt.get("rank_details")
        if not isinstance(phase_receipt, dict) or not isinstance(rank_details, Mapping):
            raise RuntimeContractError(
                "post-assembly rank details require an existing phase receipt",
                code="runtime.phase_receipt_missing",
                context={"phase": phase},
            )
        if "rank_details" in phase_receipt:
            raise RuntimeContractError(
                "training phase rank details were captured more than once",
                code="runtime.phase_receipt_duplicate",
                context={"phase": phase},
            )
        phase_receipt["rank_details"] = {
            str(rank): dict(detail) if isinstance(detail, Mapping) else detail
            for rank, detail in rank_details.items()
        }

    return capture


def _resolve_shared_pinned_runtime_baseline(
    provenance: Mapping[str, Any] | None,
    *,
    attention_backend: str,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
) -> dict[str, Any]:
    """Admit every launcher's imported runtime before cache or model work."""

    local_detail: dict[str, Any] = {
        "admission_status": "running",
        "baseline": None,
    }

    def admit_local() -> dict[str, Any]:
        try:
            if provenance is None:
                raise RuntimeContractError(
                    "launcher rank has no collected execution provenance",
                    code="runtime.pinned_runtime_provenance_missing",
                    context={"rank": rank},
                )
            receipt = require_pinned_runtime_baseline(
                provenance=provenance,
                attention_backend=attention_backend,
            )
            local_detail.update(
                admission_status="completed",
                baseline=dict(receipt),
            )
            return dict(receipt)
        except Exception as exc:
            comparison = getattr(exc, "result", None)
            local_detail.update(
                admission_status="failed",
                error_type=type(exc).__name__[:128],
                error_code=str(
                    getattr(exc, "code", "runtime.pinned_runtime_baseline_rejected")
                )[:128],
                comparison=(
                    dict(comparison) if isinstance(comparison, Mapping) else None
                ),
            )
            raise RuntimeContractError(
                "pinned runtime baseline rejected the current execution stack",
                code="runtime.pinned_runtime_baseline_rejected",
                context={
                    "rank": rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1024],
                    "comparison": local_detail.get("comparison"),
                },
                cause=exc,
            ) from exc

    def validate_receipts(receipt: Mapping[str, Any]) -> None:
        rank_details = receipt.get("rank_details")
        if not isinstance(rank_details, Mapping):
            return
        baselines: list[dict[str, Any]] = []
        for detail in rank_details.values():
            baseline = detail.get("baseline") if isinstance(detail, Mapping) else None
            if not isinstance(baseline, Mapping):
                return
            baselines.append(dict(baseline))
        canonical = baselines[0] if baselines else None
        if canonical is None or any(item != canonical for item in baselines[1:]):
            raise RuntimeContractError(
                "pinned runtime baseline receipts differ across launcher ranks",
                code="runtime.pinned_runtime_baseline_rank_mismatch",
            )

    observed = control_plane._run_rank_converged_phase(
        "upstream_runtime_baseline_admission",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=admit_local,
        local_details=lambda: local_detail,
        receipt_sink=validate_receipts,
    )
    if (
        not isinstance(observed, Mapping)
        or observed.get("schema_version") != 3
        or observed.get("admitted") is not True
        or observed.get("attention_backend") != attention_backend
        or observed.get("mismatches") != []
        or not isinstance(observed.get("baseline_sha256"), str)
        or len(str(observed["baseline_sha256"])) != 64
        or not isinstance(observed.get("reference_only"), Mapping)
    ):
        raise RuntimeContractError(
            "pinned runtime baseline receipt is malformed or inconsistent",
            code="runtime.pinned_runtime_baseline_status_invalid",
        )
    return dict(observed)


_MAPPED_NATIVE_ATTESTATION_MAX_BYTES = 32 * 1024


def _bounded_mapped_native_attestation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeContractError(
            "mapped-native execution attestation is not a mapping",
            code="runtime.mapped_native_execution_status_invalid",
        )
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(
            "mapped-native execution attestation is not JSON-serializable",
            code="runtime.mapped_native_execution_status_invalid",
            cause=exc,
        ) from exc
    if len(encoded) > _MAPPED_NATIVE_ATTESTATION_MAX_BYTES:
        raise RuntimeContractError(
            "mapped-native execution attestation exceeds the bounded receipt limit",
            code="runtime.mapped_native_execution_receipt_oversized",
            context={
                "max_bytes": _MAPPED_NATIVE_ATTESTATION_MAX_BYTES,
                "observed_bytes": len(encoded),
            },
        )
    return dict(value)


def _validate_mapped_native_attestation(value: Mapping[str, Any]) -> None:
    mapped_cudnn = value.get("mapped_cudnn_components")
    if (
        value.get("schema_version") != 1
        or value.get("cuda_initialized") is not True
        or value.get("admitted") is not True
        or value.get("mismatches") != []
        or not isinstance(value.get("components"), Mapping)
        or not isinstance(mapped_cudnn, list)
        or any(not isinstance(name, str) or not name for name in mapped_cudnn)
    ):
        raise RuntimeContractError(
            "mapped-native execution attestation is malformed or rejected",
            code="runtime.mapped_native_execution_status_invalid",
        )


def _move_model_and_resolve_mapped_native_execution(
    *,
    model: Any,
    accelerator: Any,
    provenance: Mapping[str, Any] | None,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
    receipt_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Use the real CUDA model surface, then admit its mapped native DSOs."""

    local_detail: dict[str, Any] = {
        "admission_status": "running",
        "attestation": None,
    }

    def move_and_attest() -> dict[str, Any]:
        if provenance is None:
            raise RuntimeContractError(
                "mapped-native execution attestation requires execution provenance",
                code="runtime.mapped_native_execution_provenance_missing",
                context={"rank": rank},
            )
        # This is the first intentional CUDA use owned by the training pipeline.
        # The attestation must inspect mappings only after this operation.
        try:
            model.to(torch.device(accelerator.device))
        except Exception as exc:
            local_detail.update(
                admission_status="failed",
                error_type=type(exc).__name__[:128],
                error_code="runtime.mapped_native_execution_cuda_use_failed",
            )
            raise RuntimeContractError(
                "model CUDA initialization failed before mapped-native attestation",
                code="runtime.mapped_native_execution_cuda_use_failed",
                context={"rank": rank, "error_type": type(exc).__name__},
                cause=exc,
            ) from exc
        try:
            observed = require_mapped_native_execution_attestation(
                provenance=provenance
            )
            bounded = _bounded_mapped_native_attestation(observed)
            _validate_mapped_native_attestation(bounded)
        except Exception as exc:
            comparison = getattr(exc, "result", None)
            bounded_comparison: dict[str, Any] | None = None
            if isinstance(comparison, Mapping):
                try:
                    bounded_comparison = _bounded_mapped_native_attestation(comparison)
                except RuntimeContractError:
                    bounded_comparison = {
                        "schema_version": comparison.get("schema_version"),
                        "admitted": False,
                        "receipt_status": "unavailable_oversized_or_invalid",
                    }
            local_detail.update(
                admission_status="failed",
                attestation=bounded_comparison,
                error_type=type(exc).__name__[:128],
                error_code=str(
                    getattr(exc, "code", "runtime.mapped_native_execution_rejected")
                )[:128],
            )
            if isinstance(exc, RuntimeContractError):
                raise
            raise RuntimeContractError(
                "mapped-native execution attestation rejected the active CUDA stack",
                code="runtime.mapped_native_execution_rejected",
                context={
                    "rank": rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1024],
                    "attestation": bounded_comparison,
                },
                cause=exc,
            ) from exc
        local_detail.update(
            admission_status="completed",
            attestation=bounded,
        )
        return bounded

    def validate_and_capture(receipt: Mapping[str, Any]) -> None:
        if receipt_sink is not None:
            receipt_sink(receipt)
        rank_details = receipt.get("rank_details")
        if not isinstance(rank_details, Mapping):
            raise RuntimeContractError(
                "mapped-native execution rank receipts are missing",
                code="runtime.mapped_native_execution_status_invalid",
            )
        completed = [
            detail
            for detail in rank_details.values()
            if isinstance(detail, Mapping)
            and detail.get("admission_status") == "completed"
        ]
        if len(completed) != world_size:
            return
        attestations = [
            _bounded_mapped_native_attestation(detail.get("attestation"))
            for detail in completed
        ]
        canonical = attestations[0]
        if any(attestation != canonical for attestation in attestations[1:]):
            raise RuntimeContractError(
                "mapped-native execution attestations differ across launcher ranks",
                code="runtime.mapped_native_execution_rank_mismatch",
            )

    observed = control_plane._run_rank_converged_phase(
        "mapped_native_execution_attestation",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=move_and_attest,
        local_details=lambda: local_detail,
        receipt_sink=validate_and_capture,
    )
    if not isinstance(observed, Mapping):
        raise RuntimeContractError(
            "mapped-native execution admission returned an invalid receipt",
            code="runtime.mapped_native_execution_status_invalid",
        )
    result = _bounded_mapped_native_attestation(observed)
    _validate_mapped_native_attestation(result)
    return result


def _resolve_converged_profile_sync_timing_selector(
    *,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
) -> dict[str, bool | str]:
    """Resolve and freeze one synchronization-timing selector on every rank."""

    local_detail: dict[str, Any] = {"resolution": None}

    def resolve_local() -> dict[str, bool | str]:
        resolved = _resolve_profile_sync_timing_selector()
        local_detail["resolution"] = dict(resolved)
        return resolved

    def validate_receipts(receipt: Mapping[str, Any]) -> None:
        rank_details = receipt.get("rank_details")
        if not isinstance(rank_details, Mapping):
            return
        resolutions = [
            dict(detail["resolution"])
            for detail in rank_details.values()
            if isinstance(detail, Mapping)
            and isinstance(detail.get("resolution"), Mapping)
        ]
        if len(resolutions) != world_size:
            return
        canonical = resolutions[0]
        if any(item != canonical for item in resolutions[1:]):
            raise RuntimeContractError(
                "profile synchronization timing selector differs across ranks",
                code="runtime.profile_sync_timing_resolution_mismatch",
                context={"rank_resolutions": resolutions},
            )

    observed = control_plane._run_rank_converged_phase(
        "profile_sync_timing_resolution",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=resolve_local,
        local_details=lambda: local_detail,
        receipt_sink=validate_receipts,
    )
    if (
        not isinstance(observed, Mapping)
        or not isinstance(observed.get("enabled"), bool)
        or observed.get("source")
        not in {"default", "COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS"}
    ):
        raise RuntimeContractError(
            "profile synchronization timing selector receipt is invalid",
            code="runtime.profile_sync_timing_resolution_invalid",
        )
    return dict(observed)


def _resolve_converged_forward_input_provider_mode(
    configured_mode: ForwardInputProviderMode,
    *,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
    receipt_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> ResolvedForwardInputProviderMode:
    """Resolve one identical provider policy on every launcher rank.

    This boundary deliberately runs on the temporary CPU control plane.  The
    deprecated environment selector is rank-local, so accepting only the local
    result would allow different trainer/provider behavior across ranks.
    """

    local_detail: dict[str, Any] = {
        "resolution_status": "running",
        "resolution": None,
    }

    def resolve_local() -> ResolvedForwardInputProviderMode:
        try:
            resolved = resolve_forward_input_provider_mode(configured_mode)
        except Exception as exc:
            code = getattr(exc, "code", "python_exception")
            local_detail.update(
                resolution_status="failed",
                error_type=type(exc).__name__[:128],
                error_code=str(code)[:128],
            )
            raise
        local_detail.update(
            resolution_status="completed",
            resolution=resolved.to_receipt_dict(),
        )
        return resolved

    def validate_and_capture(receipt: Mapping[str, Any]) -> None:
        if receipt_sink is not None:
            receipt_sink(receipt)
        rank_details = receipt.get("rank_details")
        if not isinstance(rank_details, Mapping):
            return
        resolutions: dict[str, dict[str, Any]] = {}
        for rank_key, detail in rank_details.items():
            resolution = (
                detail.get("resolution") if isinstance(detail, Mapping) else None
            )
            if not isinstance(resolution, Mapping):
                # A failed resolver is handled by the common phase failure
                # path after this sink has preserved its bounded diagnostics.
                return
            resolutions[str(rank_key)] = dict(resolution)
        canonical = resolutions.get("0")
        if canonical is None or any(
            resolution != canonical for resolution in resolutions.values()
        ):
            raise RuntimeContractError(
                "forward input provider resolution differs across launcher ranks",
                code="training.forward_input_provider_resolution_mismatch",
                context={"rank_resolutions": resolutions},
            )

    return control_plane._run_rank_converged_phase(
        "config_provenance_resolution",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=resolve_local,
        local_details=lambda: local_detail,
        receipt_sink=validate_and_capture,
    )


_MODEL_FREE_PREFLIGHT_PHASES = (
    "cache_identity_resolution",
    "cache_publication_admission",
    "train_rank_hydration",
)


def _preflight_receipt_sink(
    lifecycle: dict[str, Any],
) -> Callable[[Mapping[str, Any]], None]:
    def capture(receipt: Mapping[str, Any]) -> None:
        rank_details = receipt.get("rank_details")
        rank_resources = receipt.get("rank_resources")
        if not isinstance(rank_details, Mapping) or not isinstance(
            rank_resources, Mapping
        ):
            raise RuntimeContractError(
                "cache preflight convergence returned no bounded phase trace",
                code="runtime.phase_status_invalid",
                context={"phase": "cache_preflight"},
            )
        world_size = int(rank_resources["world_size"])
        pending: list[dict[str, Any]] = []
        for phase in _MODEL_FREE_PREFLIGHT_PHASES:
            phase_rows: list[tuple[int, Mapping[str, Any]]] = []
            for rank in range(world_size):
                detail = rank_details.get(str(rank))
                trace = (
                    detail.get("phase_trace") if isinstance(detail, Mapping) else None
                )
                row = trace.get(phase) if isinstance(trace, Mapping) else None
                if isinstance(row, Mapping):
                    phase_rows.append((rank, row))
            if not phase_rows:
                continue
            if len(phase_rows) != world_size:
                raise RuntimeContractError(
                    "cache preflight phase trace has partial rank coverage",
                    code="runtime.phase_status_invalid",
                    context={"phase": phase},
                )
            phase_resources = converge_rank_cpu_resources(
                [(rank, row["resource_snapshot"]) for rank, row in phase_rows],
                world_size=world_size,
            )
            statuses = [str(row.get("status")) for _, row in phase_rows]
            if any(status not in {"completed", "failed"} for status in statuses):
                raise RuntimeContractError(
                    "cache preflight phase trace has an invalid status",
                    code="runtime.phase_status_invalid",
                    context={"phase": phase},
                )
            status = "failed" if "failed" in statuses else "completed"
            duration_seconds = max(
                float(row.get("duration_seconds", 0.0)) for _, row in phase_rows
            )
            per_rank_details = {
                str(rank): {
                    "status": str(row["status"]),
                    "duration_seconds": float(row["duration_seconds"]),
                    **(
                        {"failure": dict(row["failure"])}
                        if isinstance(row.get("failure"), Mapping)
                        else {}
                    ),
                    **(
                        {"details": dict(row["details"])}
                        if isinstance(row.get("details"), Mapping)
                        else {}
                    ),
                }
                for rank, row in phase_rows
            }
            pending.append(
                {
                    "phase": phase,
                    "status": status,
                    "duration_seconds": duration_seconds,
                    "rank_resources": phase_resources,
                    "rank_details": per_rank_details,
                }
            )
            if status == "failed":
                break
        lifecycle["pending_preflight_phases"] = pending
        lifecycle["cache_admission_rank_resources"] = rank_resources

    return capture


def _persist_preflight_phase_receipts(
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
) -> None:
    if writer is None or lifecycle.get("preflight_phases_persisted") is True:
        return
    for row in lifecycle.get("pending_preflight_phases", ()):
        if not isinstance(row, Mapping):
            continue
        phase = str(row["phase"])
        common = {
            "completed_at": cache_workflow._utc_now(),
            "duration_seconds": float(row["duration_seconds"]),
            "rank_resources": row["rank_resources"],
            "rank_details": row["rank_details"],
        }
        if row["status"] == "failed":
            writer.record_failed_phase(
                phase,
                reason="one_or_more_ranks_failed_at_phase_boundary",
                **common,
            )
        else:
            writer.record_completed_phase(phase, **common)
    lifecycle["preflight_phases_persisted"] = True


def _fail_active_run_phase(writer: RunWriter | None, lifecycle: dict[str, Any]) -> None:
    phase = lifecycle.get("active_phase")
    if isinstance(phase, str):
        captured = lifecycle.get("phase_rank_receipts", {}).get(phase)
        rank_resources = (
            captured.get("rank_resources")
            if isinstance(captured, Mapping)
            and isinstance(captured.get("rank_resources"), Mapping)
            else rank_cpu_resources_from_metric_rows(
                None,
                world_size=int(lifecycle.get("world_size", 1)),
            )
        )
        _finish_run_phase(
            writer,
            lifecycle,
            phase,
            status="failed",
            rank_resources=rank_resources,
        )


def _record_terminal_measurement_summaries(
    writer: RunWriter | None,
    lifecycle: Mapping[str, Any],
) -> None:
    if writer is None:
        return
    expected = int(lifecycle.get("expected_measured_steps", 0))
    accepted = int(lifecycle.get("accepted_measured_steps", 0))
    steady_duration = float(lifecycle.get("steady_state_duration_seconds", 0.0))
    world_size = int(lifecycle.get("world_size", 1))
    steady_rank_resources = (
        lifecycle["steady_state_rank_resources"]
        if isinstance(lifecycle.get("steady_state_rank_resources"), Mapping)
        else rank_cpu_resources_from_metric_rows(None, world_size=world_size)
    )
    if expected == 0:
        writer.record_phase_summary(
            "steady_state",
            status="not_run",
            completed_at=None,
            duration_seconds=0.0,
            duration_scope=_STEADY_STATE_DURATION_SCOPE,
            accepted_measured_steps=accepted,
            expected_measured_steps=expected,
            reason="no_post_warmup_steps",
            rank_resources=steady_rank_resources,
        )
    else:
        writer.record_phase_summary(
            "steady_state",
            status="completed",
            completed_at=cache_workflow._utc_now(),
            duration_seconds=steady_duration,
            duration_scope=_STEADY_STATE_DURATION_SCOPE,
            accepted_measured_steps=accepted,
            expected_measured_steps=expected,
            rank_resources=steady_rank_resources,
        )

    eval_events = int(lifecycle.get("evaluation_event_count", 0))
    evaluation_rank_resources = (
        lifecycle["evaluation_rank_resources"]
        if isinstance(lifecycle.get("evaluation_rank_resources"), Mapping)
        else rank_cpu_resources_from_metric_rows(None, world_size=world_size)
    )
    if eval_events == 0:
        writer.record_phase_summary(
            "evaluation_execution",
            status="not_run",
            completed_at=None,
            duration_seconds=0.0,
            duration_scope=_EVALUATION_DURATION_SCOPE,
            event_count=0,
            reason="no_scheduled_evaluation_executed",
            resource_observation_scope=_EVAL_RESOURCE_OBSERVATION_SCOPE,
            rank_resources=evaluation_rank_resources,
        )
    else:
        writer.record_phase_summary(
            "evaluation_execution",
            status="completed",
            completed_at=cache_workflow._utc_now(),
            duration_seconds=float(lifecycle.get("evaluation_duration_seconds", 0.0)),
            duration_scope=_EVALUATION_DURATION_SCOPE,
            event_count=eval_events,
            resource_observation_scope=_EVAL_RESOURCE_OBSERVATION_SCOPE,
            resource_high_water_observed_after_events=(
                lifecycle.get("evaluation_resource_high_water")
                if isinstance(lifecycle.get("evaluation_resource_high_water"), Mapping)
                else None
            ),
            rank_resources=evaluation_rank_resources,
        )


def _record_evaluation_failure(
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
    *,
    planned_step_id: int,
    duration_seconds: float,
    error: Exception,
) -> None:
    lifecycle["evaluation_failed"] = True
    if writer is None or lifecycle.get("evaluation_summary_recorded") is True:
        return
    error_code = (
        error.code if isinstance(error, RuntimeContractError) else "python_exception"
    )
    rank_resources = rank_cpu_resources_from_metric_rows(
        None,
        world_size=int(lifecycle.get("world_size", 1)),
    )
    writer.record_phase_summary(
        "evaluation_execution",
        status="failed",
        completed_at=cache_workflow._utc_now(),
        duration_seconds=max(0.0, float(duration_seconds)),
        duration_scope="failed_event_local_rank_elapsed_before_exception",
        event_count=int(lifecycle.get("evaluation_event_count", 0)),
        reason=(
            f"evaluation_exception_at_step_{int(planned_step_id)}:"
            f"{type(error).__name__}:{error_code}"
        ),
        resource_observation_scope=_EVAL_RESOURCE_OBSERVATION_SCOPE,
        rank_resources=rank_resources,
    )
    lifecycle["evaluation_summary_recorded"] = True


def _record_evaluation_summary_after_failure(
    writer: RunWriter | None,
    lifecycle: Mapping[str, Any],
) -> None:
    if writer is None or lifecycle.get("evaluation_summary_recorded") is True:
        return
    phases = writer.read_run()["measurement"]["phases"]
    if "evaluation_execution" in phases:
        return
    eval_events = int(lifecycle.get("evaluation_event_count", 0))
    rank_resources = (
        lifecycle["evaluation_rank_resources"]
        if isinstance(lifecycle.get("evaluation_rank_resources"), Mapping)
        else rank_cpu_resources_from_metric_rows(
            None,
            world_size=int(lifecycle.get("world_size", 1)),
        )
    )
    if eval_events == 0:
        writer.record_phase_summary(
            "evaluation_execution",
            status="not_run",
            completed_at=None,
            duration_seconds=0.0,
            duration_scope=_EVALUATION_DURATION_SCOPE,
            event_count=0,
            reason="run_failed_before_any_evaluation_executed",
            resource_observation_scope=_EVAL_RESOURCE_OBSERVATION_SCOPE,
            rank_resources=rank_resources,
        )
        return
    writer.record_phase_summary(
        "evaluation_execution",
        status="completed",
        completed_at=cache_workflow._utc_now(),
        duration_seconds=float(lifecycle.get("evaluation_duration_seconds", 0.0)),
        duration_scope=_EVALUATION_DURATION_SCOPE,
        event_count=eval_events,
        reason="completed_events_aggregated_before_later_run_failure",
        resource_observation_scope=_EVAL_RESOURCE_OBSERVATION_SCOPE,
        resource_high_water_observed_after_events=(
            lifecycle.get("evaluation_resource_high_water")
            if isinstance(lifecycle.get("evaluation_resource_high_water"), Mapping)
            else None
        ),
        rank_resources=rank_resources,
    )


def _merge_scalar_high_water(
    current: Mapping[str, float] | None,
    observed: Mapping[str, float],
) -> dict[str, float]:
    merged = (
        {}
        if current is None
        else {str(key): float(value) for key, value in current.items()}
    )
    for key, value in observed.items():
        merged[str(key)] = max(float(value), merged.get(str(key), float("-inf")))
    return {key: merged[key] for key in sorted(merged)}


def _initialize_artifact_owner(
    *,
    accelerator: Any,
    run_directory: RunDirectory,
    run_id: str,
    created_at: str,
    resolved_config: Any,
    provenance: Mapping[str, Any] | None = None,
    measurement_context: Mapping[str, Any] | None = None,
    entry_started_at: str | None = None,
    segment_id: str | None = None,
    continuation_lineage: Mapping[str, Any] | None = None,
) -> RunWriter | None:
    writer: RunWriter | None = None
    status: dict[str, Any] | None = None
    if bool(accelerator.is_main_process):
        try:
            writer = RunWriter.initialize(
                run_dir=run_directory.run_dir,
                run_id=run_id,
                run_name=run_directory.run_name,
                artifact_root=run_directory.artifact_root,
                collision_outcome=run_directory.collision_policy,
                created_at=created_at,
                config_fingerprint=resolved_config.fingerprint,
                resolved_config=resolved_config.to_artifact_dict(),
                world_size=int(accelerator.num_processes),
                provenance=provenance,
                measurement_context=measurement_context,
                entry_started_at=entry_started_at,
                segment_id=segment_id,
                continuation_lineage=continuation_lineage,
            )
            status = {"ok": True}
        except Exception as exc:
            status = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    try:
        objects: list[Any] = [status]
        if int(accelerator.num_processes) > 1:
            broadcast = getattr(accelerator, "broadcast_object_list", None)
            if callable(broadcast):
                result = broadcast(objects, from_process=0)
                if result is not None:
                    objects = result
            elif broadcast_object_list is None:
                raise RuntimeContractError(
                    "artifact initialization handshake requires accelerate",
                    code="runtime.artifact_init_broadcast_unavailable",
                )
            else:
                broadcast_object_list(objects, from_process=0)
        shared = objects[0]
        if not isinstance(shared, Mapping) or not bool(shared.get("ok")):
            raise RuntimeContractError(
                "rank zero failed to initialize shared run artifacts",
                code="runtime.artifact_initialization_failed",
                context={
                    "error_type": str(
                        shared.get("error_type", "unknown")
                        if isinstance(shared, Mapping)
                        else "invalid_status"
                    ),
                    "error": str(
                        shared.get("error", "unknown error")
                        if isinstance(shared, Mapping)
                        else shared
                    ),
                },
            )
    except BaseException as exc:
        if writer is not None:
            try:
                writer.finalize(
                    status="failed",
                    updated_at=cache_workflow._utc_now(),
                    completed_steps=0,
                    consumed_packs=0,
                    checkpoint_event_count=0,
                    optimizer_update_status=None,
                    finite_status=None,
                    terminal_error=(
                        "artifact initialization handshake failed: "
                        f"{type(exc).__name__[:control_plane._CACHE_PREFLIGHT_IDENTITY_LIMIT]}"
                    ),
                )
            except BaseException:
                pass
        raise
    if bool(accelerator.is_main_process):
        return writer
    return None


def _resolve_resume_continuation_lineage(
    config: Any,
    *,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
) -> dict[str, Any] | None:
    resume = getattr(config, "resume", None)
    if resume is None:
        return None
    if resume.mode == "disabled" or resume.checkpoint_dir is None:
        return None
    checkpoint_dir = Path(resume.checkpoint_dir).expanduser().resolve()
    local_detail: dict[str, Any] = {"lineage": None}

    def inspect_local() -> dict[str, Any]:
        manifest = load_training_state_manifest(checkpoint_dir)
        if manifest.world_size != world_size:
            raise RuntimeContractError(
                "exact resume requires the checkpoint world size",
                code="training.resume_world_size_mismatch",
                context={
                    "checkpoint_world_size": manifest.world_size,
                    "current_world_size": world_size,
                },
            )
        manifest_path = (
            checkpoint_dir / TRAINING_STATE_DIRECTORY / TRAINING_STATE_MANIFEST
        )
        lineage = {
            "parent_run_id": manifest.parent_run_id,
            "parent_segment_id": manifest.parent_segment_id,
            "parent_checkpoint_identity": {
                "resolved_path": str(checkpoint_dir),
                "checkpoint_step": manifest.checkpoint_step,
                "training_state_manifest_file_sha256": cache_workflow._file_sha256(manifest_path),
                "training_state_aggregate_digest": manifest.aggregate_digest,
            },
            "parent_continuation_index": manifest.continuation_index,
            "continuation_index": manifest.continuation_index + 1,
        }
        local_detail["lineage"] = lineage
        return lineage

    def validate_receipts(receipt: Mapping[str, Any]) -> None:
        rank_details = receipt.get("rank_details")
        if not isinstance(rank_details, Mapping):
            return
        lineages = [
            dict(detail["lineage"])
            for detail in rank_details.values()
            if isinstance(detail, Mapping)
            and isinstance(detail.get("lineage"), Mapping)
        ]
        if len(lineages) != world_size:
            return
        if any(lineage != lineages[0] for lineage in lineages[1:]):
            raise RuntimeContractError(
                "exact resume lineage differs across launcher ranks",
                code="training.resume_lineage_rank_mismatch",
            )

    observed = control_plane._run_rank_converged_phase(
        "resume_lineage_admission",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=inspect_local,
        local_details=lambda: local_detail,
        receipt_sink=validate_receipts,
    )
    if not isinstance(observed, Mapping):
        raise RuntimeContractError(
            "exact resume lineage admission returned an invalid receipt",
            code="training.resume_lineage_invalid",
        )
    return dict(observed)


def _initialize_model_free_run_owner(
    *,
    config: Any,
    resolved_config: Any,
    repo_root: Path,
    launch_rank: int,
    launch_world_size: int,
    preflight_gatherer: Callable[[Any], Sequence[Any]] | None,
    measurement_context: Mapping[str, Any] | None,
    entry_started_at: str,
) -> RunIdentity:
    model_free_control_plane = control_plane._build_model_free_control_plane(
        rank=launch_rank,
        world_size=launch_world_size,
        rank_report_gatherer=preflight_gatherer,
    )
    run_directory = _resolve_shared_run_directory(
        config,
        cwd=repo_root,
        accelerator=model_free_control_plane,
    )
    provenance = collect_execution_provenance(repository_root=repo_root)
    continuation_lineage = _resolve_resume_continuation_lineage(
        config,
        rank=launch_rank,
        world_size=launch_world_size,
        rank_report_gatherer=preflight_gatherer,
    )
    created_at = cache_workflow._utc_now()
    run_id = _run_id(config.run.name, resolved_config.fingerprint)
    if continuation_lineage is not None:
        run_id = f"{run_id}-continuation-{continuation_lineage['continuation_index']}"
    segment_id = (
        "segment-"
        + hashlib.sha256(
            f"{run_id}:{run_directory.run_dir}".encode("utf-8")
        ).hexdigest()[:32]
    )
    resolved_measurement_context = {
        "comparison_arm": "unclassified",
        "wall_clock_scope": "training_entry_to_terminal_artifact",
        "warmup_exclusion_steps": 1,
        "workload_identity": resolved_config.fingerprint,
        "world_size": launch_world_size,
        **({} if measurement_context is None else dict(measurement_context)),
    }
    resolved_measurement_context["workload_identity"] = resolved_config.fingerprint
    resolved_measurement_context["world_size"] = launch_world_size
    resolved_measurement_context["profile_sync_timings"] = (
        _resolve_profile_sync_timing_selector()
    )
    measurement_warmup_steps = resolved_measurement_context.get(
        "warmup_exclusion_steps"
    )
    if (
        isinstance(measurement_warmup_steps, bool)
        or not isinstance(measurement_warmup_steps, int)
        or measurement_warmup_steps < 1
    ):
        raise RuntimeContractError(
            "measurement warm-up exclusion must be a positive integer",
            code="runtime.measurement_warmup_invalid",
            context={"warmup_exclusion_steps": measurement_warmup_steps},
        )
    writer = _initialize_artifact_owner(
        accelerator=model_free_control_plane,
        run_directory=run_directory,
        run_id=run_id,
        created_at=created_at,
        resolved_config=resolved_config,
        provenance=provenance if launch_rank == 0 else None,
        measurement_context=resolved_measurement_context,
        entry_started_at=entry_started_at,
        segment_id=segment_id,
        # Bind the parent only after all checkpoint components and current
        # runtime identities are admitted.  A failed attempt must not claim a
        # corrupt or incompatible parent merely because its manifest parsed.
        continuation_lineage=None,
    )
    return RunIdentity(
        run_directory=run_directory,
        run_id=run_id,
        run_segment_id=segment_id,
        writer=writer,
        measurement_warmup_steps=measurement_warmup_steps,
        provenance=provenance,
        continuation_lineage=continuation_lineage,
    )


def new_training_lifecycle(
    *,
    plan: execution_plan.TrainingExecutionPlan,
    run_identity: RunIdentity,
) -> dict[str, Any]:
    """Return the session's mutable lifecycle counters at their entry values.

    The counters exist before the session because the facade's pre-model
    ``config_provenance_resolution`` and ``cache_admission`` phases already
    record into them; ``TrainingSession`` owns them from construction onward.
    """

    return {
        "completed_steps": 0,
        "consumed_packs": 0,
        "checkpoint_event_count": 0,
        "optimizer_update_status": None,
        "finite_status": None,
        "active_phase": None,
        "phase_started_monotonic": None,
        "measurement_warmup_steps": run_identity.measurement_warmup_steps,
        "accepted_measured_steps": 0,
        "expected_measured_steps": 0,
        "steady_state_duration_seconds": 0.0,
        "evaluation_event_count": 0,
        "evaluation_duration_seconds": 0.0,
        "evaluation_resource_high_water": None,
        "evaluation_rank_resources": None,
        "evaluation_summary_recorded": False,
        "steady_state_rank_resources": None,
        "phase_rank_receipts": {},
        "pending_preflight_phases": [],
        "preflight_phases_persisted": False,
        "resolved_max_steps": None,
        "world_size": plan.launch_world_size,
        "entry_started_at": plan.entry_started_at,
        "entry_started_monotonic": plan.entry_started_monotonic,
    }


def admit_runtime_baseline_policies(
    *,
    plan: execution_plan.TrainingExecutionPlan,
    rank_control_plane: control_plane.RankControlPlane,
    run_identity: RunIdentity,
    lifecycle: dict[str, Any],
    runtime_determinism: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Open ``config_provenance_resolution`` and admit the runtime baseline."""

    writer = run_identity.writer
    config = plan.resolved_config.config
    _begin_run_phase(
        writer,
        lifecycle,
        "config_provenance_resolution",
        started_at=plan.entry_started_at,
        started_monotonic=plan.entry_started_monotonic,
        resources=dict(plan.entry_resources),
    )
    pinned_runtime_baseline = _resolve_shared_pinned_runtime_baseline(
        run_identity.provenance,
        attention_backend=str(config.model.attn_implementation),
        rank=plan.launch_rank,
        world_size=plan.launch_world_size,
        rank_report_gatherer=rank_control_plane.gatherer,
    )
    if writer is not None:
        writer.bind_policy_identity(
            "upstream_runtime_baseline",
            pinned_runtime_baseline,
        )
        if runtime_determinism is None:
            raise RuntimeContractError(
                "runtime determinism policy is unavailable after convergence",
                code="runtime.determinism_rank_mismatch",
            )
        writer.bind_policy_identity(
            "runtime_determinism",
            cache_workflow._runtime_determinism_run_policy(
                runtime_determinism,
                pinned_runtime_baseline=pinned_runtime_baseline,
            ),
        )
    return pinned_runtime_baseline


def apply_converged_profile_sync_policy(
    *,
    plan: execution_plan.TrainingExecutionPlan,
    rank_control_plane: control_plane.RankControlPlane,
) -> Mapping[str, Any]:
    """Converge the profile-sync selector and install the process-wide policy.

    The caller must treat the policy as configured from the moment this call
    returns; ``reset_profile_sync_timing_policies`` is its only release.
    """

    profile_sync_timings = _resolve_converged_profile_sync_timing_selector(
        rank=plan.launch_rank,
        world_size=plan.launch_world_size,
        rank_report_gatherer=rank_control_plane.gatherer,
    )
    set_qwen_profile_sync_timing_policy(bool(profile_sync_timings["enabled"]))
    set_trainer_profile_sync_timing_policy(bool(profile_sync_timings["enabled"]))
    return profile_sync_timings


def reset_profile_sync_timing_policies() -> None:
    """Release the process-wide Qwen/trainer profile-sync timing policy."""

    set_qwen_profile_sync_timing_policy(None)
    set_trainer_profile_sync_timing_policy(None)


def admit_forward_input_provider_policy(
    *,
    plan: execution_plan.TrainingExecutionPlan,
    rank_control_plane: control_plane.RankControlPlane,
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
    profile_sync_timings: Mapping[str, Any],
) -> ResolvedForwardInputProviderMode:
    """Bind the applied profile-sync identity, admit the provider, close phase.

    The profile-sync policy identity is bound here rather than in
    ``apply_converged_profile_sync_policy`` so that the process-wide policy is
    already installed - and therefore already releasable by the caller - before
    any writer I/O can fail.  This is the literal Wave-0 statement order.
    """

    config = plan.resolved_config.config
    if writer is not None:
        writer.bind_policy_identity(
            "profile_sync_timings",
            {"schema_version": 1, **profile_sync_timings},
        )
    resolved_forward_input_provider = _resolve_converged_forward_input_provider_mode(
        getattr(
            config.training,
            "forward_input_provider_mode",
            "synchronous",
        ),
        rank=plan.launch_rank,
        world_size=plan.launch_world_size,
        rank_report_gatherer=rank_control_plane.gatherer,
        receipt_sink=_phase_receipt_sink(lifecycle, "config_provenance_resolution"),
    )
    _finish_run_phase(writer, lifecycle, "config_provenance_resolution")
    return resolved_forward_input_provider


def admit_training_cache_workflow(
    *,
    plan: execution_plan.TrainingExecutionPlan,
    rank_control_plane: control_plane.RankControlPlane,
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
) -> Mapping[str, Any]:
    """Converge the model-free cache preflight and persist its receipts."""

    preflight_phase_trace: dict[str, Any] = {}
    preflight = rank_control_plane.converge(
        "cache_preflight",
        lambda: cache_workflow._resolve_model_free_training_preflight(
            config=plan.resolved_config.config,
            config_path=plan.resolved_config.entry_config_path,
            repo_root=plan.repo_root,
            rank=plan.launch_rank,
            world_size=plan.launch_world_size,
            rank_report_gatherer=rank_control_plane.gatherer,
            phase_trace=preflight_phase_trace,
        ),
        local_details=lambda: {"phase_trace": preflight_phase_trace},
        receipt_sink=_preflight_receipt_sink(lifecycle),
    )
    _persist_preflight_phase_receipts(writer, lifecycle)
    return preflight


def open_admitted_accelerator(
    *,
    plan: execution_plan.TrainingExecutionPlan,
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
) -> Any:
    """Open ``cache_admission`` and construct the run's Accelerator.

    Accelerator construction stays outside the convergence claim: a rank that
    fails before a post-Accelerator process group exists cannot safely publish
    status to peers.  The caller binds the result into the control plane, whose
    boundary then covers every live rank.
    """

    _begin_run_phase(writer, lifecycle, "cache_admission")
    return _build_accelerator(plan.resolved_config.config.training.precision)


def admit_accelerator_runtime(
    *,
    plan: execution_plan.TrainingExecutionPlan,
    rank_control_plane: control_plane.RankControlPlane,
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
    accelerator: Any,
) -> None:
    """Converge the post-Accelerator runtime check and close ``cache_admission``."""

    config = plan.resolved_config.config

    def validate_post_accelerator_runtime() -> None:
        validate_accelerator_runtime(
            accelerator,
            expected_mixed_precision=config.training.precision,
        )
        accelerate_rank = int(accelerator.process_index)
        accelerate_world_size = int(accelerator.num_processes)
        if (
            accelerate_rank != plan.launch_rank
            or accelerate_world_size != plan.launch_world_size
        ):
            raise RuntimeContractError(
                "Accelerate identity disagrees with the admitted model-free launch identity",
                code="runtime.preflight_accelerate_identity_mismatch",
                context={
                    "preflight_rank": plan.launch_rank,
                    "preflight_world_size": plan.launch_world_size,
                    "accelerate_rank": accelerate_rank,
                    "accelerate_world_size": accelerate_world_size,
                },
            )

    rank_control_plane.converge(
        "accelerator_runtime_preflight",
        validate_post_accelerator_runtime,
        receipt_sink=_phase_receipt_sink(lifecycle, "cache_admission"),
    )
    _finish_run_phase(writer, lifecycle, "cache_admission")
    if writer is not None:
        writer.record_phase_not_run(
            "cache_preparation",
            reason="single_process_preparation_precedes_training_launch",
        )
        writer.record_phase_not_run(
            "cache_publication",
            reason="immutable_cache_hit",
        )


def publish_training_entry_failure(
    *,
    plan: execution_plan.TrainingExecutionPlan,
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
    error: BaseException,
) -> None:
    """Best-effort terminal publication for a failed training entry.

    Every step is individually guarded: a publication failure must never
    replace the primary exception the caller is already raising.
    """

    if writer is None:
        return
    try:
        _persist_preflight_phase_receipts(writer, lifecycle)
    except BaseException:
        pass
    try:
        _fail_active_run_phase(writer, lifecycle)
    except BaseException:
        pass
    try:
        measurement = writer.read_run()["measurement"]
        if measurement.get("terminal_phase") is None:
            writer.record_failed_phase(
                "cache_admission",
                completed_at=cache_workflow._utc_now(),
                duration_seconds=max(
                    0.0, time.monotonic() - plan.entry_started_monotonic
                ),
                reason="preflight_control_plane_failure",
                rank_resources=rank_cpu_resources_from_metric_rows(
                    None,
                    world_size=plan.launch_world_size,
                ),
            )
    except BaseException:
        pass
    try:
        _record_evaluation_summary_after_failure(writer, lifecycle)
    except BaseException:
        pass
    try:
        writer.finalize(
            status="failed",
            updated_at=cache_workflow._utc_now(),
            completed_steps=int(lifecycle["completed_steps"]),
            consumed_packs=int(lifecycle["consumed_packs"]),
            checkpoint_event_count=int(lifecycle["checkpoint_event_count"]),
            optimizer_update_status=lifecycle["optimizer_update_status"],
            finite_status=lifecycle["finite_status"],
            terminal_error=f"{type(error).__name__}: {error}",
            entry_started_monotonic=plan.entry_started_monotonic,
        )
    except BaseException:
        pass


class TrainingSession:
    """The one model-bearing lifetime of a training entry (design decision 8).

    The facade admits the model-free inputs, binds the Accelerator through the
    control plane, and then constructs exactly one session.  ``run`` executes
    the fixed initialized-training choreography in a literal ordered body -
    there is no phase subclass, registry, callback container, or alternate
    backend.  ``fail`` publishes the terminal failure best-effort without
    swallowing the primary exception, and ``close`` is idempotent and never
    performs a success finalization.
    """

    def __init__(
        self,
        *,
        plan: execution_plan.TrainingExecutionPlan,
        control_plane: control_plane.RankControlPlane,
        writer: RunWriter | None,
        run_identity: RunIdentity,
        cache_preflight: Mapping[str, Any] | None,
        admitted_policies: Mapping[str, Any],
        lifecycle: dict[str, Any],
    ) -> None:
        self.plan = plan
        self.control_plane = control_plane
        self.writer = writer
        self.run_identity = run_identity
        self.cache_preflight = cache_preflight
        self.admitted_policies = MappingProxyType(dict(admitted_policies))
        self.lifecycle = lifecycle
        self._closed = False

    def run(self) -> dict[str, Any]:
        return _run_initialized_training(
            repo_root=self.plan.repo_root,
            resolved_config=self.plan.resolved_config,
            config=self.plan.resolved_config.config,
            accelerator=self.control_plane.accelerator,
            run_directory=self.run_identity.run_directory,
            run_id=self.run_identity.run_id,
            run_segment_id=self.run_identity.run_segment_id,
            writer=self.writer,
            lifecycle=self.lifecycle,
            rank_report_gatherer=self.control_plane.gatherer,
            preflight=self.cache_preflight,
            resolved_forward_input_provider=self.admitted_policies[
                "forward_input_provider"
            ],
            provenance=self.run_identity.provenance,
            continuation_lineage=self.run_identity.continuation_lineage,
            pinned_runtime_baseline=self.admitted_policies[
                "pinned_runtime_baseline"
            ],
            profile_sync_timings=self.admitted_policies["profile_sync_timings"],
        )

    def fail(self, error: BaseException) -> None:
        publish_training_entry_failure(
            plan=self.plan,
            writer=self.writer,
            lifecycle=self.lifecycle,
            error=error,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        reset_profile_sync_timing_policies()


def _run_initialized_training(
    *,
    repo_root: Path,
    resolved_config: Any,
    config: Any,
    accelerator: Any,
    run_directory: RunDirectory,
    run_id: str,
    run_segment_id: str,
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
    resolved_forward_input_provider: ResolvedForwardInputProviderMode,
    provenance: Mapping[str, Any] | None,
    preflight: Mapping[str, Any] | None = None,
    continuation_lineage: Mapping[str, Any] | None = None,
    pinned_runtime_baseline: Mapping[str, Any] | None = None,
    profile_sync_timings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if preflight is None:
        pack_cache_root, pack_cache_root_receipt = cache_workflow._resolve_pack_cache_root(repo_root)
    else:
        if int(preflight["rank"]) != int(accelerator.process_index) or int(
            preflight["world_size"]
        ) != int(accelerator.num_processes):
            raise RuntimeContractError(
                "model-free cache preflight identity changed before training assembly",
                code="runtime.preflight_identity_mismatch",
                context={
                    "preflight_rank": int(preflight["rank"]),
                    "preflight_world_size": int(preflight["world_size"]),
                    "accelerate_rank": int(accelerator.process_index),
                    "accelerate_world_size": int(accelerator.num_processes),
                },
            )
        pack_cache_root = Path(preflight["cache_root"])
        pack_cache_root_receipt = dict(preflight["cache_root_receipt"])

    def assemble_model_surface() -> tuple[Any, Any, Any]:
        _begin_run_phase(writer, lifecycle, "model_loading")
        components = load_qwen_components(config, load_model=True)
        if components.model is None:
            raise RuntimeContractError(
                "training pipeline requires load_qwen_components(..., load_model=True)",
                code="training.model_not_loaded",
                context={"base_model": config.model.base_model},
            )
        if preflight is None:
            runtime_controls = resolve_qwen_runtime_controls(
                config,
                tokenizer_vocab_size=components.token_identity.tokenizer_vocab_size,
                model_logits_dtype=config.training.precision,
            )
            del runtime_controls
        else:
            expected_components = preflight["components"]
            expected_identity = {
                "base_model_path": str(expected_components.base_model_path),
                "base_config_sha256": str(expected_components.base_config_sha256),
                "tokenizer_sha256": str(expected_components.tokenizer_sha256),
                "processor": expected_components.processor_identity.to_artifact_dict(),
                "tokens": expected_components.token_identity.to_artifact_dict(),
            }
            observed_identity = {
                "base_model_path": str(components.base_model_path),
                "base_config_sha256": str(components.base_config_sha256),
                "tokenizer_sha256": str(components.tokenizer_sha256),
                "processor": components.processor_identity.to_artifact_dict(),
                "tokens": components.token_identity.to_artifact_dict(),
            }
            if observed_identity != expected_identity:
                raise RuntimeContractError(
                    "model-load identities changed after cache preflight admission",
                    code="training.cache_preflight_identity_drift",
                    context={
                        "expected": expected_identity,
                        "observed": observed_identity,
                    },
                )
        adapter_evidence = load_default_adapter_source_gate_evidence(repo_root)
        adapter_plan = build_adapter_setup_plan(
            config.adapter,
            adapter_evidence,
            base_model_path=components.base_model_path,
        )
        adapter_result = setup_dora_adapter(components.model, adapter_plan)
        model = adapter_result.model

        special_token_selection = build_default_special_token_selection(
            config.model.special_token_embeddings,
            components.token_identity,
        )
        special_token_evidence = (
            load_default_special_token_embedding_source_gate_evidence(repo_root)
        )
        special_token_result = install_special_token_embedding_deltas(
            model,
            special_token_selection,
            source_gate=special_token_evidence,
        )
        model = special_token_result.model
        if getattr(adapter_plan, "mode", None) == "warm_start_expand_dora":
            if adapter_plan.repaired_embedding_payload_path is None:
                raise RuntimeContractError(
                    "warm_start_expand_dora requires repaired embedding payload path",
                    code="adapter.warm_start_embedding_payload_required",
                )
            load_special_token_embedding_deltas(
                special_token_result,
                adapter_plan.repaired_embedding_payload_path,
                expected_base_model_path=components.base_model_path,
                expected_base_config_sha256=components.base_config_sha256,
                expected_tokenizer_sha256=components.tokenizer_sha256,
            )
        enable_training_memory_savers(model)
        if writer is not None:
            writer.bind_policy_identity(
                "packing",
                cache_workflow._packing_policy_receipt(config),
            )
            writer.bind_policy_identity(
                "attention_proof",
                {
                    "schema_version": 1,
                    "attention_implementation": str(config.model.attn_implementation),
                    "proof_policy": str(config.model.fa2_branch_proof),
                },
            )
            writer.bind_policy_identity(
                "resume",
                {
                    "schema_version": 1,
                    "mode": str(
                        getattr(getattr(config, "resume", None), "mode", "disabled")
                    ),
                    "supported_boundary": "optimizer_step_accumulation_zero",
                    "world_size_policy": "exact_same_world_size",
                },
            )
        return components, adapter_result, special_token_result

    assembled = control_plane._run_rank_converged_phase(
        "model_loading",
        rank=int(accelerator.process_index),
        world_size=int(accelerator.num_processes),
        rank_report_gatherer=rank_report_gatherer,
        body=assemble_model_surface,
        receipt_sink=_phase_receipt_sink(lifecycle, "model_loading"),
    )
    components, adapter_result, special_token_result = assembled
    model = special_token_result.model
    _move_model_and_resolve_mapped_native_execution(
        model=model,
        accelerator=accelerator,
        provenance=provenance,
        rank=int(accelerator.process_index),
        world_size=int(accelerator.num_processes),
        rank_report_gatherer=rank_report_gatherer,
        receipt_sink=_phase_rank_details_sink(lifecycle, "model_loading"),
    )
    _finish_run_phase(writer, lifecycle, "model_loading")

    if preflight is None:
        vocab_groups = build_token_vocabulary_groups(
            components.token_identity,
            tokenizer=components.tokenizer,
        )
        _begin_run_phase(writer, lifecycle, "cache_admission")
        train_cache = cache_workflow._resolve_or_build_train_pack_cache(
            config,
            components,
            vocab_groups,
            repo_root=repo_root,
            accelerator=accelerator,
            rank=int(accelerator.process_index),
            verification_level="manifest",
            cache_root=pack_cache_root,
        )
        schedule = resolve_planned_step_schedule(
            config,
            packs_per_epoch=train_cache["micro_step_count"],
            world_size=int(accelerator.num_processes),
            source_config_path=str(resolved_config.entry_config_path),
        )
        train_micro_steps = load_rank_micro_steps_from_cache(
            train_cache["cache_dir"],
            cache_root=pack_cache_root,
            expected_fingerprint=str(train_cache["fingerprint"]),
            schedule=schedule,
            rank=int(accelerator.process_index),
            world_size=int(accelerator.num_processes),
        )
    else:
        vocab_groups = preflight["vocab_groups"]
        train_cache = dict(preflight["train_cache"])
        schedule = preflight["schedule"]
        train_micro_steps = tuple(preflight["train_micro_steps"])
    lifecycle["resolved_max_steps"] = schedule.resolved_max_steps
    lifecycle["expected_measured_steps"] = max(
        0,
        int(schedule.resolved_max_steps)
        - int(lifecycle.get("measurement_warmup_steps", 0)),
    )
    if writer is not None:
        writer.bind_schedule(resolved_max_steps=schedule.resolved_max_steps)
        cache_workflow._bind_cache_materialization(writer, "train", train_cache)
    train_micro_steps = cache_workflow._attach_image_processors_to_micro_steps(
        train_micro_steps,
        image_processor=cache_workflow._qwen_image_processor(components),
    )
    train_micro_steps = cache_workflow._apply_fa2_branch_proof_policy(train_micro_steps, config)
    if preflight is None:
        _finish_run_phase(writer, lifecycle, "cache_admission")
    if writer is not None and preflight is None:
        for phase in ("cache_preparation", "cache_publication"):
            phase_receipt = train_cache["phase_receipt"][phase]
            if phase_receipt["status"] == "completed":
                writer.record_completed_phase(
                    phase,
                    completed_at=cache_workflow._utc_now(),
                    duration_seconds=float(phase_receipt["duration_seconds"]),
                    resources=collect_resource_snapshot(),
                )
            else:
                writer.record_phase_not_run(
                    phase,
                    reason=str(phase_receipt["status"]),
                )
    _begin_run_phase(writer, lifecycle, "optimizer_runtime_assembly")

    def assemble_optimizer_runtime() -> tuple[LossRunner, TrainRuntime, Any]:
        loss_runner = LossRunner.from_config(config.losses)
        optimizer_group_plan = build_optimizer_group_plan(
            model,
            config.optimizer,
            adapter_receipt=adapter_result.receipt,
            special_token_receipt=special_token_result.receipt,
        )
        scheduler_plan = build_scheduler_plan(
            config.optimizer,
            total_training_steps=schedule.resolved_max_steps,
        )
        del scheduler_plan
        optimizer, scheduler = build_optimizer_and_scheduler(
            config.optimizer,
            optimizer_group_plan,
            total_training_steps=schedule.resolved_max_steps,
        )
        trainable_surface_receipt = build_trainable_surface_receipt(
            model,
            adapter_receipt=adapter_result.receipt,
            special_token_receipt=special_token_result.receipt,
            optimizer_group_plan=optimizer_group_plan,
        )
        runtime = TrainRuntime(
            runtime_config=config.runtime,
            runtime_batch=schedule.runtime_batch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_mixed_precision=config.training.precision,
            max_grad_norm=config.training.max_grad_norm,
            accelerator=accelerator,
            rank_report_gatherer=rank_report_gatherer,
        )
        return loss_runner, runtime, trainable_surface_receipt

    loss_runner, runtime, trainable_surface_receipt = control_plane._run_rank_converged_phase(
        "optimizer_runtime_assembly",
        rank=int(accelerator.process_index),
        world_size=int(accelerator.num_processes),
        rank_report_gatherer=rank_report_gatherer,
        body=assemble_optimizer_runtime,
        receipt_sink=_phase_receipt_sink(lifecycle, "optimizer_runtime_assembly"),
    )
    _finish_run_phase(writer, lifecycle, "optimizer_runtime_assembly")

    _begin_run_phase(writer, lifecycle, "evaluation_hydration")
    eval_cache = (
        cache_workflow._resolve_eval_pack_cache(
            config,
            components,
            vocab_groups,
            repo_root=repo_root,
            accelerator=accelerator,
            rank=int(accelerator.process_index),
            cache_root=pack_cache_root,
        )
        if preflight is None
        else preflight["eval_cache"]
    )
    eval_reduction_receipt = (
        cache_workflow._resolve_converged_eval_reduction_receipt(
            pack_count=(
                None if eval_cache is None else int(eval_cache["micro_step_count"])
            ),
            rank=int(accelerator.process_index),
            world_size=int(accelerator.num_processes),
            rank_report_gatherer=rank_report_gatherer,
        )
        if preflight is None
        else dict(preflight["eval_reduction"])
    )
    eval_pack_count_for_consensus: int | None
    if eval_cache is None:
        # Train-reuse-as-eval fallback: `eval_micro_steps` is already this
        # rank's own train shard, not a canonical replicated eval set, so
        # disjoint sharding (which requires a shared canonical pack order
        # every rank can partition identically) does not apply here. Any
        # scheduled eval in this configuration is already rejected before
        # training starts (config/schedule validation requires an explicit
        # eval source), so this branch never actually drives a live eval
        # invocation; it is left byte-identical to pre-Wave-4 behavior.
        # `pack_count` has no canonical cross-rank meaning here (each rank's
        # train shard size legitimately differs), so the consensus check
        # below only validates `reduction_mode` identity for this branch.
        eval_micro_steps = train_micro_steps
        eval_reduction_mode = str(eval_reduction_receipt["effective_mode"])
        eval_pack_count_for_consensus = eval_reduction_receipt["pack_count"]
        control_plane._run_rank_converged_phase(
            "evaluation_hydration",
            rank=int(accelerator.process_index),
            world_size=int(accelerator.num_processes),
            rank_report_gatherer=rank_report_gatherer,
            body=lambda: None,
            local_details=lambda: {
                "assignment_mode": EVAL_REDUCTION_REPLICATED,
                "canonical_ordinal_total_count": {
                    "status": "unavailable",
                    "reason": "no_explicit_eval_publication",
                },
                "canonical_ordinal_assigned_count": {
                    "status": "unavailable",
                    "reason": "no_explicit_eval_publication",
                },
                "retained_micro_step_count": len(eval_micro_steps),
                "selective_decode": {
                    "decoded_micro_step_count": {
                        "status": "unavailable",
                        "reason": "no_selective_eval_hydration",
                    },
                    "decoded_chunk_count": {
                        "status": "unavailable",
                        "reason": "no_selective_eval_hydration",
                    },
                    "payload_bytes_read": {
                        "status": "unavailable",
                        "reason": "no_selective_eval_hydration",
                    },
                },
            },
            receipt_sink=_phase_receipt_sink(lifecycle, "evaluation_hydration"),
        )
    else:
        (
            eval_micro_steps,
            eval_reduction_mode,
            eval_pack_count_for_consensus,
        ) = cache_workflow._hydrate_eval_micro_steps_from_cache(
            eval_cache,
            cache_root=pack_cache_root,
            rank=int(accelerator.process_index),
            world_size=int(accelerator.num_processes),
            rank_report_gatherer=rank_report_gatherer,
            reduction_receipt=eval_reduction_receipt,
            receipt_sink=_phase_receipt_sink(lifecycle, "evaluation_hydration"),
        )
        eval_micro_steps = cache_workflow._attach_image_processors_to_micro_steps(
            eval_micro_steps,
            image_processor=cache_workflow._qwen_image_processor(components),
        )
        if writer is not None:
            cache_workflow._bind_cache_materialization(writer, "eval", eval_cache)
    # Opus HOLD P2-B: every world rank calls this SAME consensus collective
    # unconditionally, after rank-selective hydration but before
    # eval.forward's mode-dependent denominator-gather collective. If ranks
    # somehow disagree on reduction_mode (config drift, a resolution bug),
    # some ranks would otherwise call the
    # disjoint-shard denominator gather while others never call it at all --
    # a silent collective deadlock discovered only much later, if ever. This
    # check fails closed immediately instead, using the exact same bounded
    # rank-report gatherer as every other collective (no new framework).
    runtime.validate_eval_reduction_consensus(
        reduction_mode=eval_reduction_mode,
        pack_count=eval_pack_count_for_consensus,
    )
    eval_micro_steps = cache_workflow._apply_fa2_branch_proof_policy(eval_micro_steps, config)
    _finish_run_phase(writer, lifecycle, "evaluation_hydration")
    if writer is not None:
        writer.bind_policy_identity(
            "eval_reduction",
            {
                "schema_version": 1,
                **eval_reduction_receipt,
            },
        )
        writer.bind_policy_identity(
            "cache",
            {
                "schema_version": 1,
                "root": pack_cache_root_receipt,
                "train": {
                    "format_version": str(train_cache["format_version"]),
                    "fingerprint": str(train_cache["fingerprint"]),
                },
                "eval": (
                    None
                    if eval_cache is None
                    else {
                        "format_version": str(eval_cache["format_version"]),
                        "fingerprint": str(eval_cache["fingerprint"]),
                    }
                ),
            },
        )
    forward_input_provider_mode = resolved_forward_input_provider.resolved_mode
    full_train_micro_steps = tuple(train_micro_steps)
    trainer_micro_steps = full_train_micro_steps
    start_planned_step_id = 1
    exact_training_state_callback_factory: Callable[[int, Path], None] | None = None
    resume_config = getattr(config, "resume", None)
    if str(getattr(resume_config, "mode", "disabled")) == "exact_same_world_size":
        rank = int(accelerator.process_index)
        world_size = int(accelerator.num_processes)
        if pinned_runtime_baseline is None or profile_sync_timings is None:
            raise RuntimeContractError(
                "exact resume requires admitted dependency and profile-sync identities",
                code="training.resume_identity_incomplete",
            )
        unwrap_model = getattr(accelerator, "unwrap_model", None)
        if not callable(unwrap_model):
            raise RuntimeContractError(
                "exact resume requires Accelerator.unwrap_model",
                code="training.resume_unwrap_model_unavailable",
            )
        exact_model = unwrap_model(runtime.model)
        scaler = getattr(accelerator, "scaler", None)
        topology_bindings = _gather_exact_resume_topology_bindings(
            rank=rank,
            world_size=world_size,
            rank_report_gatherer=rank_report_gatherer,
        )
        resolved_config_artifact = resolved_config.to_artifact_dict()
        exact_policy = _exact_resume_policy_payload(
            packing=cache_workflow._packing_policy_receipt(config),
            input_provider={
                "mode": forward_input_provider_mode,
                **resolved_forward_input_provider.to_receipt_dict(),
            },
            attention={
                "attention_implementation": str(config.model.attn_implementation),
                "proof_policy": str(config.model.fa2_branch_proof),
            },
            profile_sync=dict(profile_sync_timings),
            eval_reduction={
                **eval_reduction_receipt,
            },
            resume=resume_config.model_dump(mode="json"),
        )
        exact_identities = control_plane._run_rank_converged_phase(
            "exact_resume_identity_resolution",
            rank=rank,
            world_size=world_size,
            rank_report_gatherer=rank_report_gatherer,
            body=lambda: _build_pipeline_exact_resume_identities(
                base_model_path=components.base_model_path,
                train_cache=train_cache,
                eval_cache=eval_cache,
                pinned_runtime_baseline=pinned_runtime_baseline,
                policy=exact_policy,
                resolved_config=resolved_config_artifact,
                topology_bindings=topology_bindings,
                trainable_surface=trainable_surface_receipt.to_artifact_dict(),
            ),
        )

        def exact_status_gather(local_status: Any) -> Sequence[Any]:
            if world_size == 1:
                return (local_status,)
            if rank_report_gatherer is None:
                raise RuntimeContractError(
                    "exact resume requires an all-rank status gatherer",
                    code="training.resume_status_gather_unavailable",
                )
            return tuple(rank_report_gatherer(local_status))

        resume_checkpoint_dir = resume_config.checkpoint_dir
        if resume_checkpoint_dir is not None:
            checkpoint_dir = Path(resume_checkpoint_dir).expanduser().resolve()
            checkpoint_manifest = control_plane._run_rank_converged_phase(
                "exact_resume_manifest_selection",
                rank=rank,
                world_size=world_size,
                rank_report_gatherer=rank_report_gatherer,
                body=lambda: load_training_state_manifest(checkpoint_dir),
            )
            checkpoint_step = int(checkpoint_manifest.checkpoint_step)
            if checkpoint_step >= int(schedule.resolved_max_steps):
                raise RuntimeContractError(
                    "an exact-resume checkpoint has no remaining planned step",
                    code="training.resume_terminal_checkpoint",
                    context={
                        "checkpoint_step": checkpoint_step,
                        "resolved_max_steps": int(schedule.resolved_max_steps),
                    },
                )
            read_only_admission = control_plane._run_rank_converged_phase(
                "exact_resume_read_only_admission",
                rank=rank,
                world_size=world_size,
                rank_report_gatherer=rank_report_gatherer,
                body=lambda: _read_only_admit_pipeline_exact_resume(
                    checkpoint_dir,
                    checkpoint_step=checkpoint_step,
                    rank=rank,
                    world_size=world_size,
                    identities=exact_identities,
                    resolved_config=resolved_config_artifact,
                    model=exact_model,
                    optimizer=runtime.optimizer,
                    scheduler=runtime.scheduler,
                    scaler=scaler,
                    train_micro_steps=full_train_micro_steps,
                    train_cache=train_cache,
                    schedule=schedule,
                ),
            )
            continuation_lineage = _reconcile_admitted_resume_continuation_lineage(
                continuation_lineage,
                read_only_admission.continuation_lineage,
            )
            _assert_training_state_manifest_snapshot(
                checkpoint_dir,
                read_only_admission.manifest_snapshot,
            )
            if writer is not None:
                writer.bind_continuation_lineage(continuation_lineage)
            _assert_training_state_manifest_snapshot(
                checkpoint_dir,
                read_only_admission.manifest_snapshot,
            )
            restored = restore_distributed_exact_resume(
                checkpoint_dir,
                checkpoint_step=checkpoint_step,
                expected_manifest_digest=read_only_admission.manifest_digest,
                rank=rank,
                world_size=world_size,
                identities=exact_identities,
                resolved_config=resolved_config_artifact,
                model=exact_model,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                scaler=scaler,
                cuda_device_topology=("cuda:0",),
                gather_status=exact_status_gather,
            )
            if restored.manifest_digest != read_only_admission.manifest_digest:
                raise RuntimeContractError(
                    "exact-resume manifest changed after read-only admission",
                    code="training.resume_manifest_drift",
                )
            applied_cursor = _apply_exact_resume_cursor_state(
                read_only_admission.cursor,
                next_rank_local_micro_step=(
                    read_only_admission.next_rank_local_micro_step
                ),
                checkpoint_step=checkpoint_step,
                train_micro_steps=full_train_micro_steps,
                train_cache=train_cache,
                schedule=schedule,
                runtime=runtime,
                lifecycle=lifecycle,
                rank=rank,
                world_size=world_size,
            )
            trainer_micro_steps = applied_cursor.trainer_micro_steps
            start_planned_step_id = applied_cursor.start_planned_step_id

        continuation_index = (
            0
            if continuation_lineage is None
            else int(continuation_lineage["continuation_index"])
        )

        def publish_exact_training_state(
            checkpoint_step: int, checkpoint_dir: Path
        ) -> None:
            def prepare_publication() -> tuple[
                TrainingStatePublicationPlan, dict[str, Any], int
            ]:
                cursor, next_micro_step = _build_exact_resume_cursor(
                    checkpoint_step=checkpoint_step,
                    consumed_micro_steps=int(lifecycle["consumed_packs"]),
                    train_micro_steps=full_train_micro_steps,
                    train_cache=train_cache,
                    schedule=schedule,
                    runtime=runtime,
                    rank=rank,
                    world_size=world_size,
                )
                plan = _build_exact_resume_publication_plan(
                    checkpoint_step=checkpoint_step,
                    run_id=run_id,
                    run_segment_id=run_segment_id,
                    continuation_index=continuation_index,
                    world_size=world_size,
                    identities=exact_identities,
                    resolved_config=resolved_config_artifact,
                    scheduler_applicable=runtime.scheduler is not None,
                    scaler_applicable=scaler is not None,
                )
                return plan, cursor, next_micro_step

            plan, cursor, next_micro_step = control_plane._run_rank_converged_phase(
                "exact_resume_prepublication",
                rank=rank,
                world_size=world_size,
                rank_report_gatherer=rank_report_gatherer,
                body=prepare_publication,
            )
            payload = prepare_distributed_exact_resume_contribution(
                plan=plan,
                rank=rank,
                model=exact_model,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                scaler=scaler,
                cursor=cursor,
                next_rank_local_micro_step=next_micro_step,
                rng_snapshot=None,
                gather_status=exact_status_gather,
            )
            publish_distributed_exact_resume(
                checkpoint_dir,
                plan=plan,
                rank=rank,
                local_payload=payload,
                barrier=accelerator.wait_for_everyone,
                gather_status=exact_status_gather,
            )

        exact_training_state_callback_factory = publish_exact_training_state
    forward_input_provider = build_forward_input_provider(forward_input_provider_mode)
    if (forward_input_provider is None) != (
        forward_input_provider_mode == "legacy_fused"
    ):
        raise RuntimeContractError(
            "forward input provider disposition disagrees with the resolved mode",
            code="training.forward_input_provider_disposition_invalid",
            context={"resolved_mode": forward_input_provider_mode},
        )
    checkpoint_writer = CheckpointWriter(run_directory.run_dir)
    eval_by_step: dict[int, dict[str, Any]] = {}
    committed_checkpoint_steps: set[int] = set()
    checkpoint_handler = _checkpoint_handler(
        checkpoint_writer,
        model=runtime.model,
        runtime=runtime,
        adapter_name=str(getattr(adapter_result.receipt, "adapter_name", "default")),
        special_token_result=special_token_result,
        schedule=schedule,
        base_model_path=components.base_model_path,
        base_config_sha256=components.base_config_sha256,
        tokenizer_sha256=components.tokenizer_sha256,
        writer=writer,
        eval_by_step=eval_by_step,
        committed_steps=committed_checkpoint_steps,
        lifecycle=lifecycle,
        save_final=config.checkpoint.save_final,
        exact_training_state_callback_factory=exact_training_state_callback_factory,
    )
    trainer = SupervisedTrainer(
        model=model,
        schedule=schedule,
        pack_stream=iter(trainer_micro_steps),
        start_planned_step_id=start_planned_step_id,
        loss_runner=loss_runner,
        runtime=runtime,
        on_completed_step=reporting.CompletedStepReporter(
            writer=writer,
            lifecycle=lifecycle,
            runtime=runtime,
            resource_collector=collect_resource_snapshot,
        ),
        on_checkpoint=checkpoint_handler,
        on_eval=_eval_forward_handler(
            model=runtime.model,
            runtime=runtime,
            eval_micro_steps=eval_micro_steps,
            loss_runner=loss_runner,
            writer=writer,
            eval_source=config.data.eval.model_dump(mode="json")
            if config.data.eval is not None
            else None,
            eval_by_step=eval_by_step,
            reduction_mode=eval_reduction_mode,
            lifecycle=lifecycle,
        ),
        on_final=_final_handler(
            checkpoint_handler=checkpoint_handler,
            committed_steps=committed_checkpoint_steps,
            save_final=config.checkpoint.save_final,
        ),
        forward_input_provider=forward_input_provider,
    )
    # Bind only after the trainer accepted the provider: `SupervisedTrainer`
    # fails closed (P2-G) if the provider is paired with a non-streaming
    # loss runner (production-dead, Wave-5-deletion-bound), so the run
    # record never claims a provider mode that turned out to be unused.
    if writer is not None:
        writer.bind_forward_input_provider_mode(
            forward_input_provider_mode,
            resolution=resolved_forward_input_provider.to_receipt_dict(),
        )
        writer.bind_policy_identity(
            "input_provider",
            {
                "schema_version": 1,
                "mode": forward_input_provider_mode,
                **resolved_forward_input_provider.to_receipt_dict(),
            },
        )
    try:
        _begin_run_phase(writer, lifecycle, "first_optimizer_step")
        result = trainer.run()
        if lifecycle.get("active_phase") is not None:
            _fail_active_run_phase(writer, lifecycle)
        latest = result.latest_observation
        if writer is not None:
            _record_terminal_measurement_summaries(writer, lifecycle)
            entry_started_monotonic = lifecycle.get("entry_started_monotonic")
            writer.finalize(
                status="completed",
                updated_at=cache_workflow._utc_now(),
                completed_steps=result.completed_steps,
                consumed_packs=(
                    result.consumed_micro_steps
                    if start_planned_step_id == 1
                    else int(lifecycle["consumed_packs"])
                ),
                checkpoint_event_count=result.scheduled_event_counts.get(
                    "checkpoint", 0
                ),
                optimizer_update_status=None
                if latest is None
                else latest.optimizer_update_status,
                finite_status=None if latest is None else latest.finite_status,
                entry_started_monotonic=(
                    float(entry_started_monotonic)
                    if isinstance(entry_started_monotonic, (int, float))
                    and not isinstance(entry_started_monotonic, bool)
                    else None
                ),
            )
        return {
            "run_dir": str(run_directory.run_dir),
            "run_id": run_id,
            "resolved_config_fingerprint": resolved_config.fingerprint,
            "completed_steps": result.completed_steps,
            "consumed_micro_steps": (
                result.consumed_micro_steps
                if start_planned_step_id == 1
                else int(lifecycle["consumed_packs"])
            ),
            "scheduled_event_counts": dict(result.scheduled_event_counts),
        }
    finally:
        if forward_input_provider is not None:
            forward_input_provider.close()


def enable_training_memory_savers(model: Any) -> dict[str, Any]:
    train_mode_enabled = _enable_train_mode(model)
    use_cache_disabled = _disable_use_cache(model)
    gradient_checkpointing_kwargs = {"use_reentrant": False}
    gradient_checkpointing_enabled, applied_gradient_checkpointing_kwargs = (
        _enable_gradient_checkpointing(
            model,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )
    )
    input_require_grads_enabled = _call_first_available(
        model,
        "enable_input_require_grads",
    )
    return {
        "train_mode_enabled": train_mode_enabled,
        "model_training": _model_training_state(model),
        "gradient_checkpointing_enabled": gradient_checkpointing_enabled,
        "gradient_checkpointing_kwargs": applied_gradient_checkpointing_kwargs,
        "input_require_grads_enabled": input_require_grads_enabled,
        "use_cache_disabled": use_cache_disabled,
    }


def _checkpoint_handler(
    checkpoint_writer: CheckpointWriter,
    *,
    model: Any,
    runtime: Any | None = None,
    adapter_name: str,
    special_token_result: Any,
    schedule: ResolvedStepSchedule,
    base_model_path: Path,
    base_config_sha256: str,
    tokenizer_sha256: str,
    writer: RunWriter | None,
    eval_by_step: Mapping[int, Mapping[str, Any]],
    committed_steps: set[int],
    lifecycle: dict[str, Any],
    save_final: bool = True,
    exact_training_state_callback_factory: Callable[[int, Path], None] | None = None,
) -> Any:
    def handle(scheduled_event: Any, observation: CompletedStepObservation) -> None:
        accelerator = getattr(runtime, "accelerator", runtime)
        step = int(scheduled_event.planned_step_id)
        eval_observation = eval_by_step.get(step)
        is_final = save_final and step == schedule.resolved_max_steps
        publication_started_at = cache_workflow._utc_now()
        publication_started_monotonic = time.monotonic()
        completed_event_recorded = False
        if is_final:
            _begin_run_phase(
                writer,
                lifecycle,
                "checkpoint_publication",
                started_monotonic=publication_started_monotonic,
            )
        exact_training_state_callback = (
            None
            if exact_training_state_callback_factory is None
            else lambda checkpoint_dir: exact_training_state_callback_factory(
                step, checkpoint_dir
            )
        )
        try:
            checkpoint_writer.write_checkpoint(
                step=step,
                accelerator=accelerator,
                model=model,
                adapter_name=adapter_name,
                special_token_result=special_token_result,
                base_model_path=base_model_path,
                base_config_sha256=base_config_sha256,
                tokenizer_sha256=tokenizer_sha256,
                run_writer=writer,
                is_final=is_final,
                best_candidate=None
                if eval_observation is None
                else {
                    "completed": True,
                    "selector": BEST_EVAL_SELECTOR_NAME,
                    "value": eval_observation.get(BEST_EVAL_SELECTOR_NAME),
                    "optimizer_update_status": observation.optimizer_update_status,
                    "finite_status": observation.finite_status,
                },
                exact_training_state_callback=exact_training_state_callback,
            )
            if writer is not None:
                checkpoint_dir = writer.checkpoints_dir / f"step-{step}"
                inference_payload_identity = (
                    build_inference_checkpoint_payload_identity(checkpoint_dir)
                )
                checkpoint_identity = None
                if exact_training_state_callback is not None:
                    manifest = load_training_state_manifest(checkpoint_dir)
                    manifest_path = (
                        checkpoint_dir
                        / TRAINING_STATE_DIRECTORY
                        / TRAINING_STATE_MANIFEST
                    )
                    checkpoint_identity = {
                        "checkpoint_step": step,
                        "resolved_path": str(checkpoint_dir.resolve()),
                        "training_state_aggregate_digest": manifest.aggregate_digest,
                        "training_state_manifest_file_sha256": cache_workflow._file_sha256(
                            manifest_path
                        ),
                    }
                publication_completed_monotonic = time.monotonic()
                writer.record_checkpoint_publication_event(
                    step=step,
                    status="completed",
                    started_at=publication_started_at,
                    completed_at=cache_workflow._utc_now(),
                    duration_seconds=max(
                        0.0,
                        publication_completed_monotonic - publication_started_monotonic,
                    ),
                    is_final=is_final,
                    exact_training_state_enabled=(
                        exact_training_state_callback is not None
                    ),
                    checkpoint_identity=checkpoint_identity,
                    inference_payload_identity=inference_payload_identity,
                    committed_progress={
                        "schema": "coordexp-swift-checkpoint-committed-progress",
                        "schema_version": 1,
                        "completed_steps": step,
                        "consumed_packs": int(lifecycle.get("consumed_packs", 0)),
                        "optimizer_update_status": observation.optimizer_update_status,
                        "finite_status": observation.finite_status,
                    },
                    failure_code=None,
                )
                completed_event_recorded = True
            else:
                publication_completed_monotonic = time.monotonic()
            committed_steps.add(step)
            lifecycle["checkpoint_event_count"] = (
                int(lifecycle.get("checkpoint_event_count", 0)) + 1
            )
            if is_final:
                _finish_run_phase(
                    writer,
                    lifecycle,
                    "checkpoint_publication",
                    completed_monotonic=publication_completed_monotonic,
                )
        except BaseException as exc:
            if writer is not None and not completed_event_recorded:
                failure_code = getattr(exc, "code", None)
                if (
                    not isinstance(failure_code, str)
                    or not failure_code
                    or len(failure_code) > 128
                    or not failure_code.isascii()
                ):
                    failure_code = type(exc).__name__[:128]
                writer.record_checkpoint_publication_event(
                    step=step,
                    status="failed",
                    started_at=publication_started_at,
                    completed_at=cache_workflow._utc_now(),
                    duration_seconds=max(
                        0.0, time.monotonic() - publication_started_monotonic
                    ),
                    is_final=is_final,
                    exact_training_state_enabled=(
                        exact_training_state_callback is not None
                    ),
                    checkpoint_identity=None,
                    inference_payload_identity=None,
                    committed_progress=None,
                    failure_code=failure_code,
                )
            raise

    return handle


def _build_exact_resume_cursor(
    *,
    checkpoint_step: int,
    consumed_micro_steps: int,
    train_micro_steps: Sequence[Any],
    train_cache: Mapping[str, Any],
    schedule: ResolvedStepSchedule,
    runtime: Any,
    rank: int,
    world_size: int,
) -> tuple[dict[str, Any], int]:
    """Build the exact next rank-local stream position at one save boundary."""

    counters = {
        name: int(getattr(runtime, name))
        for name in (
            "optimizer_step_count",
            "scheduler_step_count",
            "zero_grad_count",
        )
    }
    cursor = _exact_resume_cursor_from_counters(
        checkpoint_step=checkpoint_step,
        consumed_micro_steps=consumed_micro_steps,
        train_micro_steps=train_micro_steps,
        train_cache=train_cache,
        schedule=schedule,
        counters=counters,
        rank=rank,
        world_size=world_size,
    )
    return cursor, consumed_micro_steps


def _build_pipeline_exact_resume_identities(
    *,
    base_model_path: Path,
    train_cache: Mapping[str, Any],
    eval_cache: Mapping[str, Any] | None,
    pinned_runtime_baseline: Mapping[str, Any],
    policy: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    topology_bindings: Sequence[RankCudaDeviceBinding],
    trainable_surface: Mapping[str, Any],
) -> Mapping[str, str]:
    base_weights = base_model_weight_identity(base_model_path)
    cache_identity = {
        "schema": "coordexp-swift-exact-resume-cache-identity-v1",
        "train": _exact_resume_cache_binding(train_cache),
        "eval": (
            None if eval_cache is None else _exact_resume_cache_binding(eval_cache)
        ),
    }
    return build_exact_resume_identities(
        base_model=str(base_weights["aggregate_sha256"]),
        cache=cache_workflow._sha256_json(cache_identity),
        dependencies=_exact_resume_dependency_identity(pinned_runtime_baseline),
        policy=cache_workflow._sha256_json(dict(policy)),
        resolved_config=resolved_config,
        topology=build_exact_resume_topology_identity(topology_bindings),
        trainable_surface=cache_workflow._sha256_json(dict(trainable_surface)),
    )


def _exact_resume_dependency_identity(
    pinned_runtime_baseline: Mapping[str, Any],
) -> str:
    if pinned_runtime_baseline.get("admitted") is not True:
        raise RuntimeContractError(
            "exact resume requires the admitted pinned runtime baseline",
            code="training.resume_dependency_not_admitted",
            context={"mismatches": list(pinned_runtime_baseline.get("mismatches", []))},
        )
    projection = {
        "schema": "coordexp-swift-exact-resume-dependencies-v1",
        "attention_backend": str(pinned_runtime_baseline["attention_backend"]),
        "baseline_sha256": str(pinned_runtime_baseline["baseline_sha256"]),
        "baseline_schema_version": int(pinned_runtime_baseline["schema_version"]),
    }
    return cache_workflow._sha256_json(projection)


def _exact_resume_cache_binding(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "determinants_sha256": str(cache["determinants_sha256"]),
        "fingerprint": str(cache["fingerprint"]),
        "format_version": str(cache["format_version"]),
        "manifest_sha256": str(cache["manifest_sha256"]),
    }


def _gather_exact_resume_topology_bindings(
    *,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
) -> tuple[RankCudaDeviceBinding, ...]:
    local_snapshot = control_plane._run_rank_converged_phase(
        "exact_resume_topology_capture",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=lambda: capture_rank_rng_snapshot(rank=rank, world_size=world_size),
    )
    local = local_snapshot.cuda_device_binding
    observed: Sequence[Any]
    if world_size == 1:
        observed = (local,)
    else:
        if rank_report_gatherer is None:
            raise RuntimeContractError(
                "exact-resume topology identity requires an all-rank gatherer",
                code="training.resume_topology_gather_unavailable",
            )
        observed = tuple(rank_report_gatherer(local))
    if len(observed) != world_size or any(
        not isinstance(item, RankCudaDeviceBinding) for item in observed
    ):
        raise RuntimeContractError(
            "exact-resume topology gather returned an invalid rank set",
            code="training.resume_topology_invalid",
        )
    return tuple(sorted(observed, key=lambda item: item.rank))


@dataclass(frozen=True)
class _AppliedExactResumeCursor:
    start_planned_step_id: int
    trainer_micro_steps: tuple[Any, ...]


@dataclass(frozen=True)
class _ReadOnlyExactResumeAdmission:
    manifest_digest: str
    continuation_lineage: Mapping[str, Any]
    manifest_snapshot: "_TrainingStateManifestSnapshot"
    cursor: Mapping[str, Any]
    next_rank_local_micro_step: int


@dataclass(frozen=True)
class _TrainingStateManifestSnapshot:
    file_sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


def _snapshot_training_state_manifest(
    checkpoint_dir: Path | str,
) -> _TrainingStateManifestSnapshot:
    manifest_path = (
        Path(checkpoint_dir).expanduser().resolve()
        / TRAINING_STATE_DIRECTORY
        / TRAINING_STATE_MANIFEST
    )
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeContractError(
            "exact-resume manifest path must be one regular file",
            code="training.resume_manifest_path_invalid",
            context={"manifest_path": str(manifest_path)},
        )
    before = manifest_path.stat(follow_symlinks=False)
    file_sha256 = cache_workflow._file_sha256(manifest_path)
    after = manifest_path.stat(follow_symlinks=False)
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if before_identity != after_identity:
        raise RuntimeContractError(
            "exact-resume manifest path changed while it was fingerprinted",
            code="training.resume_manifest_path_replaced",
            context={"manifest_path": str(manifest_path)},
        )
    return _TrainingStateManifestSnapshot(
        file_sha256=file_sha256,
        device=after_identity[0],
        inode=after_identity[1],
        size=after_identity[2],
        mtime_ns=after_identity[3],
    )


def _assert_training_state_manifest_snapshot(
    checkpoint_dir: Path | str,
    expected: _TrainingStateManifestSnapshot,
) -> None:
    observed = _snapshot_training_state_manifest(checkpoint_dir)
    if observed != expected:
        raise RuntimeContractError(
            "exact-resume manifest path changed after read-only admission",
            code="training.resume_manifest_path_replaced",
            context={
                "checkpoint_dir": str(Path(checkpoint_dir).expanduser().resolve()),
                "expected_manifest_file_sha256": expected.file_sha256,
                "observed_manifest_file_sha256": observed.file_sha256,
            },
        )


def _reconcile_admitted_resume_continuation_lineage(
    preliminary: Mapping[str, Any] | None,
    admitted: Mapping[str, Any],
) -> dict[str, Any]:
    authoritative = dict(admitted)
    if preliminary is None or dict(preliminary) != authoritative:
        raise RuntimeContractError(
            "preliminary continuation lineage differs from the admitted checkpoint",
            code="training.resume_lineage_admission_mismatch",
            context={
                "preliminary_present": preliminary is not None,
                "preliminary_manifest_file_sha256": (
                    None
                    if preliminary is None
                    else preliminary.get("parent_checkpoint_identity", {}).get(
                        "training_state_manifest_file_sha256"
                    )
                ),
                "admitted_manifest_file_sha256": authoritative[
                    "parent_checkpoint_identity"
                ]["training_state_manifest_file_sha256"],
            },
        )
    return authoritative


def _exact_resume_policy_payload(
    *,
    packing: Mapping[str, Any],
    input_provider: Mapping[str, Any],
    attention: Mapping[str, Any],
    profile_sync: Mapping[str, Any],
    eval_reduction: Mapping[str, Any],
    resume: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return semantic policy identity shared by parent and continuation runs."""

    del resume
    return {
        "schema": "coordexp-swift-exact-resume-policy-v1",
        "attention": dict(attention),
        "eval_reduction": dict(eval_reduction),
        "input_provider": dict(input_provider),
        "packing": dict(packing),
        "profile_sync": dict(profile_sync),
    }


def _build_exact_resume_publication_plan(
    *,
    checkpoint_step: int,
    run_id: str,
    run_segment_id: str,
    continuation_index: int,
    world_size: int,
    identities: Mapping[str, str],
    resolved_config: Mapping[str, Any],
    scheduler_applicable: bool,
    scaler_applicable: bool,
) -> TrainingStatePublicationPlan:
    return TrainingStatePublicationPlan(
        parent_run_id=run_id,
        parent_segment_id=run_segment_id,
        checkpoint_step=checkpoint_step,
        continuation_index=continuation_index,
        world_size=world_size,
        identities=identities,
        scheduler_applicable=scheduler_applicable,
        scaler_applicable=scaler_applicable,
        resolved_config=resolved_config,
        resume_compatibility=build_resume_compatibility_projection(resolved_config),
        accumulation_microstep=0,
    )


def _apply_exact_resume_cursor_state(
    cursor: Mapping[str, Any],
    *,
    next_rank_local_micro_step: int,
    checkpoint_step: int,
    train_micro_steps: Sequence[Any],
    train_cache: Mapping[str, Any],
    schedule: ResolvedStepSchedule,
    runtime: Any,
    lifecycle: dict[str, Any],
    rank: int,
    world_size: int,
) -> _AppliedExactResumeCursor:
    admitted = _validate_exact_resume_cursor(
        cursor,
        next_rank_local_micro_step=next_rank_local_micro_step,
        checkpoint_step=checkpoint_step,
        train_micro_steps=train_micro_steps,
        train_cache=train_cache,
        schedule=schedule,
        rank=rank,
        world_size=world_size,
    )
    counters = admitted["runtime_counters"]
    for name, value in counters.items():
        setattr(runtime, name, int(value))
    lifecycle["completed_steps"] = checkpoint_step
    lifecycle["consumed_packs"] = int(next_rank_local_micro_step)
    return _AppliedExactResumeCursor(
        start_planned_step_id=int(admitted["start_planned_step_id"]),
        trainer_micro_steps=tuple(train_micro_steps[next_rank_local_micro_step:]),
    )


def _restored_exact_resume_cursor_state(cursor: Mapping[str, Any]) -> dict[str, Any]:
    restored: dict[str, Any] = {}
    for owner in ("data", "pack"):
        envelope = cursor.get(owner)
        if not isinstance(envelope, Mapping) or not isinstance(
            envelope.get("state"), Mapping
        ):
            raise RuntimeContractError(
                "restored exact-resume cursor has an invalid owner envelope",
                code="training.resume_cursor_invalid",
                context={"owner": owner},
            )
        restored[owner] = dict(envelope["state"])
    return restored


def _read_only_admit_pipeline_exact_resume(
    checkpoint_dir: Path | str,
    *,
    checkpoint_step: int,
    rank: int,
    world_size: int,
    identities: Mapping[str, str],
    resolved_config: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    train_micro_steps: Sequence[Any],
    train_cache: Mapping[str, Any],
    schedule: ResolvedStepSchedule,
) -> _ReadOnlyExactResumeAdmission:
    resolved_checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    manifest_snapshot_before = _snapshot_training_state_manifest(
        resolved_checkpoint_dir
    )
    expectations = TrainingStateExpectations(
        checkpoint_step=checkpoint_step,
        world_size=world_size,
        identities=identities,
        scheduler_applicable=scheduler is not None,
        scaler_applicable=scaler is not None,
        resolved_config=resolved_config,
        runtime_state=capture_runtime_state_expectations(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        ),
        resume_compatibility=build_resume_compatibility_projection(resolved_config),
    )
    admitted = admit_training_state(
        resolved_checkpoint_dir,
        expectations,
        current_rank=rank,
    )
    manifest_snapshot_after = _snapshot_training_state_manifest(resolved_checkpoint_dir)
    if manifest_snapshot_after != manifest_snapshot_before:
        raise RuntimeContractError(
            "exact-resume manifest path changed during read-only admission",
            code="training.resume_manifest_path_replaced",
            context={
                "checkpoint_dir": str(resolved_checkpoint_dir),
                "before_manifest_file_sha256": (manifest_snapshot_before.file_sha256),
                "after_manifest_file_sha256": manifest_snapshot_after.file_sha256,
            },
        )
    decoded = admitted.decoded_rank
    if (
        tuple(decoded.cuda_device_topology) != ("cuda:0",)
        or int(decoded.cuda_device_count) != 1
    ):
        raise RuntimeContractError(
            "exact-resume rank state has an incompatible CUDA topology",
            code="training.resume_topology_invalid",
            context={
                "cuda_device_count": int(decoded.cuda_device_count),
                "cuda_device_topology": list(decoded.cuda_device_topology),
            },
        )
    decoded_cursor = decoded.cursor
    next_rank_local_micro_step = (
        decoded_cursor.get("next_rank_local_micro_step")
        if isinstance(decoded_cursor, Mapping)
        else None
    )
    if (
        isinstance(next_rank_local_micro_step, bool)
        or not isinstance(next_rank_local_micro_step, int)
        or next_rank_local_micro_step < 0
    ):
        raise RuntimeContractError(
            "restored exact-resume cursor position is invalid",
            code="training.resume_cursor_invalid",
            context={
                "field": "decoded.cursor.next_rank_local_micro_step",
                "value": next_rank_local_micro_step,
            },
        )
    cursor = _restored_exact_resume_cursor_state(decoded_cursor)
    _validate_exact_resume_cursor(
        cursor,
        next_rank_local_micro_step=next_rank_local_micro_step,
        checkpoint_step=checkpoint_step,
        train_micro_steps=train_micro_steps,
        train_cache=train_cache,
        schedule=schedule,
        rank=rank,
        world_size=world_size,
    )
    manifest = admitted.manifest
    admit_exact_resume_checkpoint_publication(
        resolved_checkpoint_dir,
        checkpoint_step=checkpoint_step,
        training_state_manifest_file_sha256=manifest_snapshot_after.file_sha256,
        training_state_aggregate_digest=str(manifest.aggregate_digest),
        parent_run_id=str(manifest.parent_run_id),
        parent_segment_id=str(manifest.parent_segment_id),
    )
    continuation_lineage = {
        "parent_run_id": manifest.parent_run_id,
        "parent_segment_id": manifest.parent_segment_id,
        "parent_checkpoint_identity": {
            "resolved_path": str(resolved_checkpoint_dir),
            "checkpoint_step": int(manifest.checkpoint_step),
            "training_state_manifest_file_sha256": (
                manifest_snapshot_after.file_sha256
            ),
            "training_state_aggregate_digest": str(manifest.aggregate_digest),
        },
        "parent_continuation_index": int(manifest.continuation_index),
        "continuation_index": int(manifest.continuation_index) + 1,
    }
    return _ReadOnlyExactResumeAdmission(
        manifest_digest=str(admitted.manifest.aggregate_digest),
        continuation_lineage=continuation_lineage,
        manifest_snapshot=manifest_snapshot_after,
        cursor=cursor,
        next_rank_local_micro_step=next_rank_local_micro_step,
    )


def _validate_exact_resume_cursor(
    cursor: Mapping[str, Any],
    *,
    next_rank_local_micro_step: int,
    checkpoint_step: int,
    train_micro_steps: Sequence[Any],
    train_cache: Mapping[str, Any],
    schedule: ResolvedStepSchedule,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Recompute and exact-compare a restored cursor before stream slicing."""

    if checkpoint_step >= int(schedule.resolved_max_steps):
        raise RuntimeContractError(
            "an exact-resume checkpoint has no remaining planned step",
            code="training.resume_terminal_checkpoint",
            context={
                "checkpoint_step": checkpoint_step,
                "resolved_max_steps": int(schedule.resolved_max_steps),
            },
        )
    if not isinstance(cursor, Mapping) or set(cursor) != {"data", "pack"}:
        raise RuntimeContractError(
            "exact-resume cursor must contain data and pack owner states",
            code="training.resume_cursor_invalid",
        )
    data = cursor.get("data")
    if not isinstance(data, Mapping):
        raise RuntimeContractError(
            "exact-resume data cursor is invalid",
            code="training.resume_cursor_invalid",
        )
    counters = data.get("runtime_counters")
    expected_counter_names = {
        "optimizer_step_count",
        "scheduler_step_count",
        "zero_grad_count",
    }
    if not isinstance(counters, Mapping) or set(counters) != expected_counter_names:
        raise RuntimeContractError(
            "exact-resume runtime counters are incomplete",
            code="training.resume_cursor_invalid",
        )
    normalized_counters: dict[str, int] = {}
    for name in sorted(expected_counter_names):
        value = counters[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeContractError(
                "exact-resume runtime counter is invalid",
                code="training.resume_cursor_invalid",
                context={"counter": name, "value": value},
            )
        normalized_counters[name] = value
    expected = _exact_resume_cursor_from_counters(
        checkpoint_step=checkpoint_step,
        consumed_micro_steps=next_rank_local_micro_step,
        train_micro_steps=train_micro_steps,
        train_cache=train_cache,
        schedule=schedule,
        counters=normalized_counters,
        rank=rank,
        world_size=world_size,
    )
    if dict(cursor) != expected:
        raise RuntimeContractError(
            "exact-resume cursor differs from the admitted cache and schedule",
            code="training.resume_cursor_mismatch",
        )
    return {
        "next_rank_local_micro_step": next_rank_local_micro_step,
        "start_planned_step_id": checkpoint_step + 1,
        "runtime_counters": normalized_counters,
    }


def _exact_resume_cursor_from_counters(
    *,
    checkpoint_step: int,
    consumed_micro_steps: int,
    train_micro_steps: Sequence[Any],
    train_cache: Mapping[str, Any],
    schedule: ResolvedStepSchedule,
    counters: Mapping[str, int],
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    grad_accum_steps = int(schedule.runtime_batch.resolved_grad_accum_steps)
    expected_next = checkpoint_step * grad_accum_steps
    if consumed_micro_steps != expected_next:
        raise RuntimeContractError(
            "consumed rank-local micro-steps disagree with the optimizer boundary",
            code="training.resume_cursor_position_mismatch",
            context={
                "checkpoint_step": checkpoint_step,
                "consumed_micro_steps": consumed_micro_steps,
                "expected_micro_steps": expected_next,
            },
        )
    if consumed_micro_steps < 0 or consumed_micro_steps > len(train_micro_steps):
        raise RuntimeContractError(
            "exact-resume cursor is outside the admitted rank-local stream",
            code="training.resume_cursor_position_mismatch",
            context={
                "consumed_micro_steps": consumed_micro_steps,
                "stream_length": len(train_micro_steps),
            },
        )
    if rank < 0 or rank >= world_size or world_size <= 0:
        raise RuntimeContractError(
            "exact-resume cursor has an invalid distributed identity",
            code="training.resume_cursor_invalid",
        )
    cache_identity = {
        "fingerprint": str(train_cache["fingerprint"]),
        "format_version": str(train_cache["format_version"]),
    }
    next_pack = (
        None
        if consumed_micro_steps == len(train_micro_steps)
        else _exact_resume_pack_identity(train_micro_steps[consumed_micro_steps])
    )
    common = {
        "checkpoint_step": checkpoint_step,
        "next_rank_local_micro_step": consumed_micro_steps,
        "rank": rank,
        "resolved_grad_accum_steps": grad_accum_steps,
        "resolved_max_steps": int(schedule.resolved_max_steps),
        "total_rank_local_micro_steps": len(train_micro_steps),
        "world_size": world_size,
    }
    return {
        "data": {
            "schema": "coordexp-swift-rank-data-cursor-v1",
            **common,
            "runtime_counters": dict(counters),
        },
        "pack": {
            "schema": "coordexp-swift-rank-pack-cursor-v1",
            **common,
            "cache": cache_identity,
            "next_pack": next_pack,
        },
    }


def _exact_resume_pack_identity(micro_step: Any) -> dict[str, Any]:
    pack = getattr(micro_step, "pack", None)
    segments = getattr(pack, "segments", None)
    input_ids = getattr(pack, "input_ids", None)
    pack_index = getattr(pack, "pack_index", None)
    if (
        isinstance(pack_index, bool)
        or not isinstance(pack_index, int)
        or pack_index < 0
        or not isinstance(segments, Sequence)
        or not isinstance(input_ids, Sequence)
    ):
        raise RuntimeContractError(
            "rank-local micro-step has no canonical pack identity",
            code="training.resume_cursor_invalid",
        )
    example_ids = [str(getattr(segment, "example_id", "")) for segment in segments]
    if not example_ids or any(not value for value in example_ids):
        raise RuntimeContractError(
            "rank-local pack has no canonical example identities",
            code="training.resume_cursor_invalid",
        )
    return {
        "example_ids": example_ids,
        "pack_index": pack_index,
        "sequence_length": len(input_ids),
    }


def _eval_forward_handler(
    *,
    model: Any,
    runtime: Any,
    eval_micro_steps: tuple[SupervisedMicroStep, ...],
    loss_runner: LossRunner,
    writer: RunWriter | None,
    eval_source: dict[str, Any] | None,
    eval_by_step: dict[int, dict[str, Any]],
    reduction_mode: str = EVAL_REDUCTION_REPLICATED,
    lifecycle: dict[str, Any] | None = None,
    resource_collector: Callable[[], Mapping[str, Any]] = collect_resource_snapshot,
) -> Any:
    lifecycle_state = {} if lifecycle is None else lifecycle

    def handle(scheduled_event: Any, observation: CompletedStepObservation) -> None:
        event_started_monotonic = time.monotonic()
        try:
            result = ForwardEvalRunner(
                model=model,
                micro_step_stream=iter(eval_micro_steps),
                loss_runner=loss_runner,
                eval_source=eval_source,
                runtime=runtime,
                reduction_mode=reduction_mode,
                world_size=int(getattr(runtime, "world_size", 1)),
                rank=int(getattr(runtime, "rank", 0)),
            ).run(
                planned_step_id=scheduled_event.planned_step_id,
                trigger_reasons=scheduled_event.trigger_reasons,
            )
            local_duration_seconds = max(
                0.0, time.monotonic() - event_started_monotonic
            )
            post_eval_resources = dict(resource_collector())
            local_measurement = {
                "eval_duration_seconds": local_duration_seconds,
                **reporting._resource_scalar_metrics(post_eval_resources),
            }
            gathered_measurement = runtime.gather_metrics(
                local_measurement,
                planned_step_id=int(scheduled_event.planned_step_id),
                split="eval.measurement",
            )
            reduced_measurement = (
                gathered_measurement.get("metrics")
                if isinstance(gathered_measurement, Mapping)
                else None
            )
            if not isinstance(reduced_measurement, Mapping):
                raise RuntimeContractError(
                    "evaluation measurement reduction returned no scalar mapping",
                    code="runtime.eval_measurement_reduction_failed",
                )
            event_rank_resources = rank_cpu_resources_from_metric_rows(
                gathered_measurement.get("per_rank_metrics")
                if isinstance(gathered_measurement, Mapping)
                else None,
                world_size=int(getattr(runtime, "world_size", 1)),
            )
            row = result.to_logging_row()
            row["optimizer_update_status"] = observation.optimizer_update_status
            row["finite_status"] = observation.finite_status
            row.update(dict(reduced_measurement))
            row["resource_observation_scope"] = _EVAL_RESOURCE_OBSERVATION_SCOPE
            per_rank_measurement = reporting._per_rank_measurement(gathered_measurement)
            if per_rank_measurement is not None:
                row["per_rank_measurement"] = per_rank_measurement
            eval_by_step[int(scheduled_event.planned_step_id)] = row
            reporting._append_logging_row_shared(writer=writer, row=row, runtime=runtime)
            lifecycle_state["evaluation_event_count"] = (
                int(lifecycle_state.get("evaluation_event_count", 0)) + 1
            )
            lifecycle_state["evaluation_duration_seconds"] = float(
                lifecycle_state.get("evaluation_duration_seconds", 0.0)
            ) + float(reduced_measurement["eval_duration_seconds"])
            reduced_resources = {
                str(key): float(value)
                for key, value in reduced_measurement.items()
                if str(key).startswith("resource/")
            }
            lifecycle_state["evaluation_resource_high_water"] = (
                _merge_scalar_high_water(
                    lifecycle_state.get("evaluation_resource_high_water")
                    if isinstance(
                        lifecycle_state.get("evaluation_resource_high_water"),
                        Mapping,
                    )
                    else None,
                    reduced_resources,
                )
            )
            lifecycle_state["evaluation_rank_resources"] = (
                merge_rank_cpu_resource_receipts(
                    lifecycle_state.get("evaluation_rank_resources")
                    if isinstance(
                        lifecycle_state.get("evaluation_rank_resources"), Mapping
                    )
                    else None,
                    event_rank_resources,
                )
            )
        except Exception as exc:
            _record_evaluation_failure(
                writer,
                lifecycle_state,
                planned_step_id=int(scheduled_event.planned_step_id),
                duration_seconds=max(0.0, time.monotonic() - event_started_monotonic),
                error=exc,
            )
            raise

    return handle


def _final_handler(
    *, checkpoint_handler: Any, committed_steps: set[int], save_final: bool = True
) -> Any:
    def handle(scheduled_event: Any, observation: CompletedStepObservation) -> None:
        if save_final and int(scheduled_event.planned_step_id) not in committed_steps:
            checkpoint_handler(scheduled_event, observation)

    return handle


def _resolve_shared_run_directory(
    config: Any,
    *,
    cwd: Path,
    accelerator: Any,
) -> RunDirectory:
    """Resolve one rank-zero run path and share its compact descriptor."""

    descriptor: dict[str, Any] | None = None
    if bool(accelerator.is_main_process):
        try:
            selected = resolve_run_directory(config, cwd=cwd)
            descriptor = {
                "ok": True,
                "run_name": selected.run_name,
                "artifact_root": str(selected.artifact_root),
                "run_dir": str(selected.run_dir),
                "collision_policy": selected.collision_policy,
            }
        except Exception as exc:
            descriptor = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    objects: list[Any] = [descriptor]
    if int(accelerator.num_processes) > 1:
        broadcast = getattr(accelerator, "broadcast_object_list", None)
        if callable(broadcast):
            result = broadcast(objects, from_process=0)
            if result is not None:
                objects = result
        elif broadcast_object_list is None:
            raise RuntimeContractError(
                "distributed run descriptor broadcast requires accelerate",
                code="runtime.run_descriptor_broadcast_unavailable",
            )
        else:
            broadcast_object_list(objects, from_process=0)
    shared = objects[0]
    if not isinstance(shared, Mapping):
        raise RuntimeContractError(
            "run descriptor broadcast returned an invalid payload",
            code="runtime.run_descriptor_invalid",
        )
    if not bool(shared.get("ok")):
        raise RuntimeContractError(
            "rank zero failed to resolve the shared run directory",
            code="runtime.run_directory_resolution_failed",
            context={
                "error_type": str(shared.get("error_type", "unknown")),
                "error": str(shared.get("error", "unknown error")),
            },
        )
    return RunDirectory(
        run_name=str(shared["run_name"]),
        artifact_root=Path(str(shared["artifact_root"])),
        run_dir=Path(str(shared["run_dir"])),
        collision_policy=str(shared["collision_policy"]),
    )


def _disable_use_cache(model: Any) -> list[str]:
    disabled: list[str] = []
    seen_config_ids: set[int] = set()
    for owner_name, owner in _model_and_base_model_owners(model):
        for config_path, config in _config_owners(owner):
            config_id = id(config)
            if config_id in seen_config_ids:
                continue
            seen_config_ids.add(config_id)
            if not hasattr(config, "use_cache"):
                continue
            if getattr(config, "use_cache") is not False:
                setattr(config, "use_cache", False)
            disabled.append(f"{owner_name}.{config_path}")
    return disabled


def _build_accelerator(training_precision: str) -> Any:
    if Accelerator is None:
        raise RuntimeContractError(
            "training requires the accelerate package",
            code="runtime.accelerate_unavailable",
        )
    kwargs: dict[str, Any] = {
        # CoordExp-Swift owns planned-step loss normalization and optimizer
        # cadence. Accelerate's accumulation counter would additionally divide
        # loss inside accelerator.backward(), so keep it neutral and use
        # TrainRuntime.no_sync for intermediate microsteps.
        "gradient_accumulation_steps": 1,
        "mixed_precision": str(training_precision),
    }
    return Accelerator(**kwargs)


def _config_owners(owner: Any) -> tuple[tuple[str, Any], ...]:
    config = getattr(owner, "config", None)
    if config is None:
        return ()
    owners: list[tuple[str, Any]] = [("config", config)]
    for nested_name in ("text_config", "language_config"):
        nested = getattr(config, nested_name, None)
        if nested is not None:
            owners.append((f"config.{nested_name}", nested))
    return tuple(owners)


def _enable_train_mode(model: Any) -> bool:
    train = getattr(model, "train", None)
    if not callable(train):
        return False
    train()
    return True


def _model_training_state(model: Any) -> bool | None:
    training = getattr(model, "training", None)
    if training is None:
        return None
    return bool(training)


def _enable_gradient_checkpointing(
    model: Any,
    *,
    gradient_checkpointing_kwargs: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    for _owner_name, owner in _model_and_base_model_owners(model):
        method = getattr(owner, "gradient_checkpointing_enable", None)
        if not callable(method):
            continue
        try:
            method(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            return True, dict(gradient_checkpointing_kwargs)
        except TypeError:
            method()
            return True, None
    return False, None


def _call_first_available(model: Any, method_name: str) -> bool:
    for _owner_name, owner in _model_and_base_model_owners(model):
        method = getattr(owner, method_name, None)
        if callable(method):
            method()
            return True
    return False


def _model_and_base_model_owners(model: Any) -> tuple[tuple[str, Any], ...]:
    owners: list[tuple[str, Any]] = [("model", model)]
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base_model = get_base_model()
        except Exception:
            base_model = None
        if base_model is not None and base_model is not model:
            owners.append(("base_model", base_model))
    return tuple(owners)


def _run_id(run_name: str, fingerprint: str) -> str:
    return f"{run_name}-{fingerprint[:12]}"

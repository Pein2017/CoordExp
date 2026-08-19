"""Training pipeline assembly for the V1 supervised smoke."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shlex
import time
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
from src.augmentation.factory import build_augmentation_processor
from src.augmentation.processor import AugmentationMaterializationResult
from src.artifacts import CheckpointWriter, RunWriter
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
    merge_resource_high_water,
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
from src.config.loader import load_train_config
from src.config.models import ForwardInputProviderMode, RunDirectory
from src.config.paths import resolve_run_directory
from src.config.resolve import resolve_qwen_runtime_controls
from src.data import load_raw_examples
from src.eval import ForwardEvalRunner
from src.eval.forward import (
    EVAL_REDUCTION_DISJOINT_SHARD,
    EVAL_REDUCTION_REPLICATED,
    resolve_active_eval_reduction_mode,
    resolve_eval_reduction_control,
)
from src.losses import LossRunner, build_token_vocabulary_groups
from src.optim import (
    build_optimizer_and_scheduler,
    build_optimizer_group_plan,
    build_scheduler_plan,
    build_trainable_surface_receipt,
)
from src.packing import (
    ONLINE_WINDOW_BINPACK,
    PackPlan,
    PackedSequence,
    build_pack_plan_policy_identity,
    build_packed_supervision,
    create_pack_plan,
    replay_pack_plan,
    stream_online_pack_plan_fragments,
    verify_pack_plan_stream_fragments,
)
from src.qwen import (
    QwenImageEncoding,
    attach_qwen_image_processor,
    build_default_special_token_selection,
    build_qwen_position_inputs,
    encode_rendered_example,
    load_qwen_components,
)
from src.qwen.parity import base_model_weight_identity
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
    TrainingSeedReceipt,
    seed_training_runtime,
    validate_accelerator_runtime,
)
from src.supervision import (
    build_token_sequence_from_packed_supervision,
    index_token_atoms_by_pack,
)
from src.templates import render_example
from src.training.schedule import ResolvedStepSchedule, resolve_planned_step_schedule
from src.training.pack_cache import (
    DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS,
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    PACKING_CACHE_VERSION,
    PackingCacheInvalidError,
    build_packing_cache_materialization,
    build_packing_cache_determinants,
    build_packing_cache_fingerprint,
    cache_dir_for_fingerprint,
    load_cache_manifest,
    load_rank_eval_micro_steps_from_cache,
    load_rank_micro_steps_from_cache,
    manifest_path,
    packing_cache_fingerprint_from_determinants,
    write_micro_step_cache,
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
from src.training import control_plane, execution_plan


TRAIN_SPLIT = "train"
_PACK_CACHE_WORKER_CONTEXT: dict[str, Any] | None = None
BEST_EVAL_SELECTOR_NAME = "acc_top1"
_PACK_CACHE_ROOT_ENV = "COORDEXP_SWIFT_PACK_CACHE_ROOT"
_FORWARD_INPUT_PROVIDER_MODE_ENV = "COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE"
_EVAL_REDUCTION_MODE_ENV = "COORDEXP_SWIFT_EVAL_REDUCTION_MODE"
_PROFILE_SYNC_TIMINGS_ENV = "COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS"
_RECEIPT_ENVIRONMENT_SELECTORS = frozenset(
    {
        _PACK_CACHE_ROOT_ENV,
        _FORWARD_INPUT_PROVIDER_MODE_ENV,
        _EVAL_REDUCTION_MODE_ENV,
        _PROFILE_SYNC_TIMINGS_ENV,
    }
)

_EVAL_RESOURCE_OBSERVATION_SCOPE = (
    "process_lifetime_high_water_observed_after_evaluation"
)
_STEADY_STATE_DURATION_SCOPE = "sum_of_accepted_all_rank_max_step_durations"
_EVALUATION_DURATION_SCOPE = "sum_of_all_rank_max_evaluation_event_durations"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _environment_selector_source(name: str) -> str:
    """Return a bounded selector source without persisting its raw value."""

    if name not in _RECEIPT_ENVIRONMENT_SELECTORS:
        raise RuntimeContractError(
            "environment selector is not approved for the run receipt",
            code="runtime.environment_selector_unsupported",
            context={"name": name},
        )
    return name if name in os.environ else "default"


def _resolve_pack_cache_root(repo_root: Path) -> tuple[Path, dict[str, str]]:
    """Resolve the production cache root and its compact allowlisted source."""

    raw = os.environ.get(_PACK_CACHE_ROOT_ENV)
    root = (
        repo_root / ".cache" / "coordexp_swift" / "packing"
        if raw is None
        else Path(raw)
    )
    return root, {
        "resolved_root": str(root.resolve()),
        "source": _environment_selector_source(_PACK_CACHE_ROOT_ENV),
    }


def _resolve_profile_sync_timing_selector() -> dict[str, bool | str]:
    """Mirror the exact training/Qwen timing gate without recording raw env."""

    return {
        "enabled": os.environ.get(_PROFILE_SYNC_TIMINGS_ENV) == "1",
        "source": _environment_selector_source(_PROFILE_SYNC_TIMINGS_ENV),
    }


def _launcher_device_mapping(*, rank: int, world_size: int) -> dict[str, Any]:
    raw_local_rank = os.environ.get("LOCAL_RANK")
    if raw_local_rank is None:
        if world_size != 1:
            raise RuntimeContractError(
                "distributed strict runtime identity requires LOCAL_RANK",
                code="runtime.determinism_launcher_mapping_invalid",
                context={"rank": rank, "world_size": world_size},
            )
        local_rank = 0
    elif re.fullmatch(r"0|[1-9][0-9]*", raw_local_rank) is None:
        raise RuntimeContractError(
            "LOCAL_RANK must be a strict nonnegative decimal integer",
            code="runtime.determinism_launcher_mapping_invalid",
            context={"rank": rank, "world_size": world_size},
        )
    else:
        local_rank = int(raw_local_rank)
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_devices: list[str] | None = None
    if raw_visible is not None:
        entries = raw_visible.split(",")
        if (
            not entries
            or len(entries) > 64
            or any(
                not entry or len(entry) > 128 or not entry.isascii()
                for entry in entries
            )
        ):
            raise RuntimeContractError(
                "CUDA_VISIBLE_DEVICES violates the bounded launcher mapping contract",
                code="runtime.determinism_launcher_mapping_invalid",
                context={"rank": rank, "world_size": world_size},
            )
        if local_rank >= len(entries):
            raise RuntimeContractError(
                "LOCAL_RANK is outside CUDA_VISIBLE_DEVICES",
                code="runtime.determinism_launcher_mapping_invalid",
                context={
                    "local_rank": local_rank,
                    "visible_device_count": len(entries),
                },
            )
        visible_devices = entries
    return {
        "cuda_visible_devices": visible_devices,
        "local_rank": local_rank,
        "logical_cuda_device": local_rank,
        "rank": rank,
        "world_size": world_size,
    }


def _establish_converged_runtime_determinism(
    runtime_config: Any,
    *,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
    phase: str,
) -> dict[str, Any]:
    """Establish one pre-CUDA policy and attest it on the CPU control plane."""

    local_detail: dict[str, Any] = {
        "runtime_determinism": None,
        "application": None,
        "launcher": None,
    }
    local_receipt: TrainingSeedReceipt | None = None
    converged_details: list[dict[str, Any]] = []

    def establish_local() -> TrainingSeedReceipt:
        nonlocal local_receipt
        launcher = _launcher_device_mapping(rank=rank, world_size=world_size)
        receipt = seed_training_runtime(
            int(runtime_config.seed),
            determinism_mode=str(runtime_config.determinism.mode),
            phase=phase,
        )
        local_receipt = receipt
        local_detail.update(
            runtime_determinism=receipt.to_policy_identity_dict(),
            application=receipt.to_artifact_dict(),
            launcher=launcher,
        )
        return receipt

    def validate_receipts(receipt: Mapping[str, Any]) -> None:
        rank_details = receipt.get("rank_details")
        if not isinstance(rank_details, Mapping):
            return
        details = [
            dict(rank_details[str(index)])
            for index in range(world_size)
            if str(index) in rank_details
        ]
        if len(details) != world_size:
            raise RuntimeContractError(
                "runtime determinism receipt lacks the complete launcher rank set",
                code="runtime.determinism_rank_mismatch",
            )
        policies = [detail.get("runtime_determinism") for detail in details]
        canonical = policies[0]
        if not isinstance(canonical, Mapping) or any(
            policy != canonical for policy in policies[1:]
        ):
            raise RuntimeContractError(
                "runtime determinism policy differs across launcher ranks",
                code="runtime.determinism_rank_mismatch",
            )
        launchers = [detail.get("launcher") for detail in details]
        for expected_rank, item in enumerate(launchers):
            if not isinstance(item, Mapping):
                raise RuntimeContractError(
                    "runtime determinism launcher mapping is incomplete",
                    code="runtime.determinism_rank_mismatch",
                )
            expected_fields = {
                "cuda_visible_devices",
                "local_rank",
                "logical_cuda_device",
                "rank",
                "world_size",
            }
            local_rank = item.get("local_rank")
            visible_devices = item.get("cuda_visible_devices")
            if (
                set(item) != expected_fields
                or item.get("rank") != expected_rank
                or item.get("world_size") != world_size
                or isinstance(local_rank, bool)
                or not isinstance(local_rank, int)
                or local_rank < 0
                or item.get("logical_cuda_device") != local_rank
                or (
                    visible_devices is not None
                    and (
                        not isinstance(visible_devices, list)
                        or local_rank >= len(visible_devices)
                    )
                )
            ):
                raise RuntimeContractError(
                    "runtime determinism launcher mapping differs from the rank set",
                    code="runtime.determinism_rank_mismatch",
                )
        applications = [detail.get("application") for detail in details]
        if any(not isinstance(item, Mapping) for item in applications):
            raise RuntimeContractError(
                "runtime determinism application receipt is incomplete",
                code="runtime.determinism_rank_mismatch",
            )
        if canonical.get("mode") == "strict_cuda_replay_v1" and any(
            item.get("cuda_initialized") is not False
            for item in applications
            if isinstance(item, Mapping)
        ):
            raise RuntimeContractError(
                "strict runtime determinism was not established before CUDA",
                code="runtime.determinism_rank_mismatch",
            )
        converged_details[:] = details

    control_plane._run_rank_converged_phase(
        "config_provenance_resolution",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=establish_local,
        local_details=lambda: local_detail,
        receipt_sink=validate_receipts,
    )
    if local_receipt is None or len(converged_details) != world_size:
        raise RuntimeContractError(
            "runtime determinism convergence returned no admitted receipt",
            code="runtime.determinism_rank_mismatch",
        )
    return {
        **local_receipt.to_policy_identity_dict(),
        "application_receipt": local_receipt.to_artifact_dict(),
        "launcher_attestations": [
            dict(detail["launcher"]) for detail in converged_details
        ],
        "pre_apply_cuda_initialized": bool(local_receipt.cuda_initialized),
        "pre_apply_cuda_initialized_by_rank": {
            str(index): bool(detail["application"]["cuda_initialized"])
            for index, detail in enumerate(converged_details)
        },
    }


def _runtime_determinism_run_policy(
    converged: Mapping[str, Any],
    *,
    pinned_runtime_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_sha256 = pinned_runtime_baseline.get("baseline_sha256")
    if (
        not isinstance(baseline_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", baseline_sha256) is None
    ):
        raise RuntimeContractError(
            "runtime determinism policy requires the admitted baseline digest",
            code="runtime.determinism_baseline_invalid",
        )
    return {
        **dict(converged),
        "pinned_runtime_baseline_sha256": baseline_sha256,
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
            started_at=_utc_now() if started_at is None else started_at,
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
            completed_at=_utc_now(),
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


def _resolve_eval_reduction_receipt(
    *, pack_count: int | None, world_size: int
) -> dict[str, Any]:
    """Resolve the complete rank-local eval selector without model state."""

    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise RuntimeContractError(
            "eval reduction world size must be a positive integer",
            code="training.eval_reduction_resolution_invalid",
            context={"world_size": world_size},
        )
    if pack_count is not None and (
        isinstance(pack_count, bool)
        or not isinstance(pack_count, int)
        or pack_count <= 0
    ):
        raise RuntimeContractError(
            "eval reduction pack count must be positive when an eval cache exists",
            code="training.eval_reduction_resolution_invalid",
            context={"pack_count": pack_count},
        )
    control = resolve_eval_reduction_control()
    effective_mode = (
        EVAL_REDUCTION_REPLICATED
        if pack_count is None
        else resolve_active_eval_reduction_mode(
            pack_count=pack_count,
            world_size=world_size,
        )
    )
    return {
        "control": control,
        "effective_mode": effective_mode,
        "source": _environment_selector_source(_EVAL_REDUCTION_MODE_ENV),
        "pack_count": pack_count,
        "world_size": world_size,
    }


def _resolve_converged_eval_reduction_receipt(
    *,
    pack_count: int | None,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
) -> dict[str, Any]:
    """Freeze one exact eval selector receipt on the model-free control plane."""

    local_detail: dict[str, Any] = {"resolution": None}

    def resolve_local() -> dict[str, Any]:
        resolved = _resolve_eval_reduction_receipt(
            pack_count=pack_count,
            world_size=world_size,
        )
        local_detail["resolution"] = dict(resolved)
        return resolved

    def validate_receipts(receipt: Mapping[str, Any]) -> None:
        rank_details = receipt.get("rank_details")
        if not isinstance(rank_details, Mapping):
            return
        resolutions = {
            str(rank_key): dict(detail["resolution"])
            for rank_key, detail in rank_details.items()
            if isinstance(detail, Mapping)
            and isinstance(detail.get("resolution"), Mapping)
        }
        if len(resolutions) != world_size:
            return
        canonical = resolutions.get("0")
        if canonical is None or any(
            resolution != canonical for resolution in resolutions.values()
        ):
            raise RuntimeContractError(
                "eval reduction resolution differs across launcher ranks",
                code="training.eval_reduction_resolution_mismatch",
                context={"rank_resolutions": resolutions},
            )

    observed = control_plane._run_rank_converged_phase(
        "eval_reduction_resolution",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=resolve_local,
        local_details=lambda: local_detail,
        receipt_sink=validate_receipts,
    )
    expected_fields = {
        "control",
        "effective_mode",
        "source",
        "pack_count",
        "world_size",
    }
    if not isinstance(observed, Mapping) or set(observed) != expected_fields:
        raise RuntimeContractError(
            "eval reduction resolution returned an invalid receipt",
            code="training.eval_reduction_resolution_invalid",
        )
    return dict(observed)


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
            "completed_at": _utc_now(),
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
            completed_at=_utc_now(),
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
            completed_at=_utc_now(),
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
        completed_at=_utc_now(),
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
        completed_at=_utc_now(),
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


def _scheduler_lr_metrics(scheduler_artifact: Any) -> dict[str, float]:
    if not isinstance(scheduler_artifact, Mapping):
        return {}
    learning_rates = scheduler_artifact.get("learning_rates")
    if not isinstance(learning_rates, Sequence):
        return {}
    metrics: dict[str, float] = {}
    for item in learning_rates:
        if not isinstance(item, Mapping):
            continue
        group_index = item.get("group_index")
        lr = item.get("lr")
        if group_index is None or lr is None:
            continue
        metrics[f"lr/group_{int(group_index)}"] = float(lr)
    return metrics


def _resource_scalar_metrics(snapshot: Mapping[str, Any]) -> dict[str, float]:
    cpu = snapshot.get("cpu")
    gpu = snapshot.get("gpu")
    metrics: dict[str, float] = {}
    if isinstance(cpu, Mapping):
        for field in ("max_rss_bytes", "io_read_bytes", "io_write_bytes"):
            value = cpu.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                metrics[f"resource/cpu_{field}"] = float(value)
    if isinstance(gpu, Mapping) and gpu.get("initialized") is True:
        for field in (
            "max_memory_allocated_bytes",
            "max_memory_reserved_bytes",
        ):
            value = gpu.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                metrics[f"resource/gpu_{field}"] = float(value)
    return metrics


def _per_rank_measurement(
    gathered: Mapping[str, Any],
) -> dict[str, dict[str, float]] | None:
    per_rank = gathered.get("per_rank_metrics")
    if not isinstance(per_rank, Mapping):
        return None
    prefixes = (
        "eval_duration_seconds",
        "input_build_seconds",
        "input_wait_seconds",
        "resource/",
        "step_duration_seconds",
    )
    receipt: dict[str, dict[str, float]] = {}
    for rank, metrics in per_rank.items():
        if not isinstance(metrics, Mapping):
            continue
        selected = {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and any(
                str(key) == prefix or str(key).startswith(prefix) for prefix in prefixes
            )
        }
        if selected:
            receipt[str(rank)] = {key: selected[key] for key in sorted(selected)}
    return receipt or None


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
                    updated_at=_utc_now(),
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


def _train_logging_handler(
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
    runtime: Any,
    *,
    resource_collector: Callable[[], Mapping[str, Any]] | None = None,
) -> Any:
    def handle(observation: CompletedStepObservation) -> None:
        lifecycle.update(
            completed_steps=observation.planned_step_id,
            consumed_packs=int(lifecycle.get("consumed_packs", 0))
            + observation.micro_step_count,
            optimizer_update_status=observation.optimizer_update_status,
            finite_status=observation.finite_status,
        )
        loss_bundle = dict(observation.loss_bundle_artifact)
        metrics = loss_bundle.get("metrics", {})
        accuracy_stats = loss_bundle.get("accuracy_stats")
        scalar_metrics = _scheduler_lr_metrics(observation.scheduler_artifact)
        if isinstance(metrics, Mapping):
            scalar_metrics.update({str(name): value for name, value in metrics.items()})
        timing_fields = {
            "step_duration_seconds": observation.step_duration_seconds,
            "input_build_seconds": observation.input_build_seconds,
            "input_wait_seconds": observation.input_wait_seconds,
        }
        scalar_metrics.update(
            {
                name: float(value)
                for name, value in timing_fields.items()
                if value is not None
            }
        )
        resource_snapshot = (
            None if resource_collector is None else dict(resource_collector())
        )
        if resource_snapshot is not None:
            scalar_metrics.update(_resource_scalar_metrics(resource_snapshot))
        gathered = runtime.gather_metrics(
            scalar_metrics,
            planned_step_id=observation.planned_step_id,
            split=TRAIN_SPLIT,
            accuracy_stats=accuracy_stats
            if isinstance(accuracy_stats, Mapping)
            else None,
        )
        reduced = gathered.get("metrics") if isinstance(gathered, Mapping) else None
        if not isinstance(reduced, Mapping):
            raise RuntimeContractError(
                "train metric reduction returned no scalar mapping",
                code="runtime.train_metric_reduction_failed",
            )
        reduced_accuracy_stats = (
            gathered.get("accuracy_stats") if isinstance(gathered, Mapping) else None
        )
        if any(key in reduced for key in ("acc_top1", "acc_top5")) and not isinstance(
            reduced_accuracy_stats, Mapping
        ):
            raise RuntimeContractError(
                "train metric reduction returned no global integer accuracy_stats",
                code="runtime.train_accuracy_stats_reduction_failed",
            )
        world_size = int(
            getattr(
                runtime,
                "world_size",
                getattr(getattr(runtime, "accelerator", runtime), "num_processes", 1),
            )
        )
        rank_resources = rank_cpu_resources_from_metric_rows(
            gathered.get("per_rank_metrics") if isinstance(gathered, Mapping) else None,
            world_size=world_size,
        )
        row: dict[str, Any] = {
            "step": observation.planned_step_id,
            "split": TRAIN_SPLIT,
            "micro_step_count": observation.micro_step_count,
            "optimizer_update_status": observation.optimizer_update_status,
            "finite_status": observation.finite_status,
            **dict(reduced),
        }
        if isinstance(reduced_accuracy_stats, Mapping):
            row["accuracy_stats"] = dict(reduced_accuracy_stats)
        per_rank_measurement = _per_rank_measurement(gathered)
        if per_rank_measurement is not None:
            row["per_rank_measurement"] = per_rank_measurement
        _append_logging_row_shared(writer=writer, row=row, runtime=runtime)
        warmup_steps = lifecycle.get("measurement_warmup_steps")
        reduced_step_duration = reduced.get("step_duration_seconds")
        if (
            isinstance(warmup_steps, int)
            and observation.planned_step_id > warmup_steps
            and observation.optimizer_update_status == "applied"
            and observation.finite_status == "finite"
            and isinstance(reduced_step_duration, (int, float))
            and not isinstance(reduced_step_duration, bool)
            and math.isfinite(float(reduced_step_duration))
            and float(reduced_step_duration) >= 0.0
        ):
            lifecycle["accepted_measured_steps"] = (
                int(lifecycle.get("accepted_measured_steps", 0)) + 1
            )
            lifecycle["steady_state_duration_seconds"] = float(
                lifecycle.get("steady_state_duration_seconds", 0.0)
            ) + float(reduced_step_duration)
            lifecycle["steady_state_rank_resources"] = merge_rank_cpu_resource_receipts(
                lifecycle.get("steady_state_rank_resources")
                if isinstance(lifecycle.get("steady_state_rank_resources"), Mapping)
                else None,
                rank_resources,
            )
        active_phase = lifecycle.get("active_phase")
        if (
            active_phase == "first_optimizer_step"
            and observation.optimizer_update_status == "applied"
        ):
            _finish_run_phase(
                writer,
                lifecycle,
                "first_optimizer_step",
                rank_resources=rank_resources,
            )

        resolved_max_steps = lifecycle.get("resolved_max_steps")
        is_final_step = (
            isinstance(resolved_max_steps, int)
            and observation.planned_step_id == resolved_max_steps
        )
        if is_final_step and lifecycle.get("active_phase") == "first_optimizer_step":
            _finish_run_phase(
                writer,
                lifecycle,
                "first_optimizer_step",
                status="failed",
                rank_resources=rank_resources,
            )

    return handle


def _append_logging_row_shared(
    *, writer: RunWriter | None, row: Mapping[str, Any], runtime: Any
) -> None:
    """Append on rank zero and make its bounded outcome common to every rank."""
    accelerator = getattr(runtime, "accelerator", runtime)
    is_main = bool(
        getattr(
            runtime, "is_main_process", getattr(accelerator, "is_main_process", True)
        )
    )
    status: dict[str, Any] = {"ok": True}
    if is_main:
        try:
            if writer is None:
                raise RuntimeError("rank zero has no run writer")
            writer.append_logging_row(row)
        except BaseException as exc:
            status = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:1024],
            }
    values: list[Any] = [status]
    if (
        int(getattr(runtime, "world_size", getattr(accelerator, "num_processes", 1)))
        > 1
    ):
        broadcast = getattr(accelerator, "broadcast_object_list", None)
        if callable(broadcast):
            result = broadcast(values, from_process=0)
            if result is not None:
                values = result
        elif broadcast_object_list is not None:
            broadcast_object_list(values, from_process=0)
        else:
            raise RuntimeContractError(
                "logging outcome broadcast requires accelerate",
                code="runtime.logging_broadcast_unavailable",
            )
    shared = values[0]
    if not isinstance(shared, Mapping) or not bool(shared.get("ok")):
        error = (
            shared.get("error", "invalid status")
            if isinstance(shared, Mapping)
            else shared
        )
        raise RuntimeContractError(
            f"rank zero logging append failed: {error}",
            code="runtime.logging_append_failed",
        )


def _hydrate_eval_micro_steps_from_cache(
    eval_cache: Mapping[str, Any],
    *,
    cache_root: Path,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
    reduction_receipt: Mapping[str, Any] | None = None,
    receipt_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[tuple[SupervisedMicroStep, ...], str, int]:
    """Hydrate one exact eval ordinal assignment without a full-load fallback."""

    hydration_details: dict[str, Any] = {}

    def hydrate() -> tuple[tuple[SupervisedMicroStep, ...], str, int]:
        raw_pack_count = eval_cache.get("micro_step_count")
        if (
            not isinstance(raw_pack_count, int)
            or isinstance(raw_pack_count, bool)
            or raw_pack_count <= 0
        ):
            raise RuntimeContractError(
                "evaluation cache must declare a positive micro-step count",
                code="training.eval_cache_count_invalid",
                context={"micro_step_count": raw_pack_count},
            )
        pack_count = int(raw_pack_count)
        resolved_reduction = (
            _resolve_eval_reduction_receipt(
                pack_count=pack_count,
                world_size=world_size,
            )
            if reduction_receipt is None
            else dict(reduction_receipt)
        )
        expected_fields = {
            "control",
            "effective_mode",
            "source",
            "pack_count",
            "world_size",
        }
        if (
            set(resolved_reduction) != expected_fields
            or resolved_reduction.get("pack_count") != pack_count
            or resolved_reduction.get("world_size") != world_size
            or resolved_reduction.get("effective_mode")
            not in {EVAL_REDUCTION_REPLICATED, EVAL_REDUCTION_DISJOINT_SHARD}
        ):
            raise RuntimeContractError(
                "eval hydration disagrees with the model-free reduction receipt",
                code="training.eval_reduction_receipt_mismatch",
                context={
                    "admitted_pack_count": resolved_reduction.get("pack_count"),
                    "hydration_pack_count": pack_count,
                    "admitted_world_size": resolved_reduction.get("world_size"),
                    "hydration_world_size": world_size,
                },
            )
        reduction_mode = str(resolved_reduction["effective_mode"])

        if reduction_mode == EVAL_REDUCTION_DISJOINT_SHARD:
            # The selective loader already applies the canonical modulo
            # assignment. Production must not partition this shard again.
            loader_rank = rank
            loader_world_size = world_size
        elif world_size > 1 and pack_count < world_size:
            # Structural replicated fallback: empty disjoint ranks cannot
            # participate in the accepted evaluation contract. Hydrate the
            # exact full ordinal sequence through the selective API itself.
            loader_rank = 0
            loader_world_size = 1
        else:
            # Single-rank execution and the explicit internal replicated
            # control both request the one exact all-ordinal assignment. This
            # is not a recovery path for a failed selective load.
            loader_rank = 0
            loader_world_size = 1

        shard = load_rank_eval_micro_steps_from_cache(
            eval_cache["cache_dir"],
            cache_root=cache_root,
            expected_fingerprint=str(eval_cache["fingerprint"]),
            rank=loader_rank,
            world_size=loader_world_size,
        )
        total_ordinal_count = int(shard.total_ordinal_count)
        if total_ordinal_count != pack_count:
            raise RuntimeContractError(
                "selective eval hydration disagrees with the admitted pack count",
                code="training.eval_hydration_total_count_mismatch",
                context={
                    "admitted_pack_count": pack_count,
                    "loaded_total_ordinal_count": total_ordinal_count,
                },
            )
        expected_ordinals = tuple(
            range(loader_rank, total_ordinal_count, loader_world_size)
        )
        observed_ordinals = tuple(shard.canonical_ordinals)
        if observed_ordinals != expected_ordinals:
            raise RuntimeContractError(
                "selective eval hydration returned a noncanonical ordinal assignment",
                code="training.eval_hydration_ordinal_mismatch",
                context={
                    "loader_rank": loader_rank,
                    "loader_world_size": loader_world_size,
                    "expected_count": len(expected_ordinals),
                    "observed_count": len(observed_ordinals),
                },
            )
        micro_steps = tuple(shard.micro_steps)
        if len(micro_steps) != len(expected_ordinals):
            raise RuntimeContractError(
                "selective eval hydration payload count disagrees with its ordinals",
                code="training.eval_hydration_payload_count_mismatch",
                context={
                    "ordinal_count": len(expected_ordinals),
                    "payload_count": len(micro_steps),
                },
            )
        hydration_details.update(
            {
                "assignment_mode": reduction_mode,
                "loader_rank": loader_rank,
                "loader_world_size": loader_world_size,
                "canonical_ordinal_total_count": total_ordinal_count,
                "canonical_ordinal_assigned_count": len(expected_ordinals),
                "retained_micro_step_count": len(micro_steps),
                "selective_decode": {
                    "decoded_micro_step_count": len(micro_steps),
                    "decoded_chunk_count": {
                        "status": "unavailable",
                        "reason": "selective_loader_counter_not_exposed",
                    },
                    "payload_bytes_read": {
                        "status": "unavailable",
                        "reason": "selective_loader_byte_counter_not_exposed",
                    },
                },
            }
        )
        return micro_steps, reduction_mode, total_ordinal_count

    return control_plane._run_rank_converged_phase(
        "evaluation_hydration",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=hydrate,
        local_details=lambda: hydration_details,
        receipt_sink=receipt_sink,
    )


def _cache_preparation_argv(config_path: str | Path) -> tuple[str, ...]:
    return (
        "python",
        "-m",
        "src.prepare_train_cache",
        "--config",
        str(Path(config_path).expanduser().resolve()),
    )


def _cache_preparation_environment() -> dict[str, str]:
    return dict(control_plane._STRICT_CACHE_PREPARATION_ENVIRONMENT)


def _occupied_cache_validation_category(
    cache_dir: Path,
    *,
    expected_fingerprint: str,
    error: BaseException,
) -> str:
    manifest_file = manifest_path(cache_dir)
    if not manifest_file.is_file():
        return "publication_manifest_missing"
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "publication_manifest_malformed"
    if not isinstance(payload, Mapping):
        return "publication_manifest_malformed"
    if payload.get("version") != PACKING_CACHE_VERSION:
        return "retired_or_unknown_version"
    if payload.get("status") != "complete":
        return "publication_incomplete"
    if payload.get("fingerprint") != expected_fingerprint:
        return "semantic_fingerprint_mismatch"
    error_text = str(error).lower()
    if "sha256" in error_text or "checksum" in error_text or "digest" in error_text:
        return "required_payload_digest_mismatch"
    if "pickle" in error_text or "payload" in error_text or "micro-step" in error_text:
        return "required_payload_invalid"
    if "chunk" in error_text:
        return "chunk_plan_or_payload_invalid"
    return "current_publication_invalid"


def _cache_preflight_error(
    *,
    cache_dir: Path,
    cache_root: Path,
    config_path: str | Path,
    fingerprint: str,
    split: str,
    error: BaseException,
) -> RuntimeContractError:
    common_context: dict[str, Any] = {
        "split": split,
        "cache_root": str(cache_root),
        "expected_cache_target": str(cache_dir),
        "cache_version": PACKING_CACHE_VERSION,
        "fingerprint": fingerprint,
    }
    if not cache_dir.exists():
        preparation_argv = _cache_preparation_argv(config_path)
        preparation_env = _cache_preparation_environment()
        preparation_command = shlex.join(
            [
                *(f"{name}={value}" for name, value in preparation_env.items()),
                *preparation_argv,
            ]
        )
        return RuntimeContractError(
            "required packing cache is not prepared; "
            f"expected cache target: {cache_dir}; prepare with: {preparation_command}",
            code="training.pack_cache_not_prepared",
            context={
                **common_context,
                "validation_category": "expected_target_missing",
                "automatic_recovery": "single_process_preparation_required",
                "preparation_argv": list(preparation_argv),
                "preparation_env": preparation_env,
                "preparation_command": preparation_command,
            },
            cause=error,
        )
    return RuntimeContractError(
        "required v3 packing cache target is occupied by an invalid immutable "
        f"publication: {cache_dir}",
        code="training.pack_cache_immutable_collision",
        context={
            **common_context,
            "validation_category": _occupied_cache_validation_category(
                cache_dir,
                expected_fingerprint=fingerprint,
                error=error,
            ),
            "automatic_recovery": "unavailable",
        },
        cause=error,
    )


def _admit_model_free_pack_cache(
    config: Any,
    components: Any,
    *,
    vocab_groups: Any,
    dataset: Any,
    split: str,
    cache_root: Path,
    config_path: str | Path,
    verification_level: str,
    resolved_fingerprint: str | None = None,
) -> dict[str, Any]:
    fingerprint = (
        build_packing_cache_fingerprint(
            config,
            components,
            dataset=dataset,
            split=split,
            vocab_groups=vocab_groups,
        )
        if resolved_fingerprint is None
        else resolved_fingerprint
    )
    cache_dir = cache_dir_for_fingerprint(cache_root, fingerprint)
    admission_started = time.monotonic()
    try:
        manifest = load_cache_manifest(
            cache_dir,
            cache_root=cache_root,
            expected_fingerprint=fingerprint,
            level=verification_level,
        )
    except PackingCacheInvalidError as exc:
        raise _cache_preflight_error(
            cache_dir=cache_dir,
            cache_root=cache_root,
            config_path=config_path,
            fingerprint=fingerprint,
            split=split,
            error=exc,
        ) from exc
    admission_seconds = time.monotonic() - admission_started
    cache_manifest_path = manifest_path(cache_dir)
    return {
        "cache_dir": cache_dir,
        "format_version": str(manifest["version"]),
        "fingerprint": fingerprint,
        "micro_step_count": int(manifest["micro_step_count"]),
        "chunk_count": len(manifest["chunks"]),
        "chunk_size": int(manifest["chunk_size"]),
        "status": manifest["status"],
        "build_status": "hit",
        "manifest_path": cache_manifest_path,
        "manifest_sha256": _file_sha256(cache_manifest_path),
        "determinants_sha256": _sha256_json(manifest["determinants"]),
        "chunk_sha256s": [str(chunk["sha256"]) for chunk in manifest["chunks"]],
        "materialization": manifest.get("materialization"),
        "augmentation": manifest.get("augmentation"),
        "phase_receipt": {
            "cache_preparation": {
                "status": "not_run_cache_hit",
                "duration_seconds": 0.0,
            },
            "cache_publication": {
                "status": "not_run_cache_hit",
                "duration_seconds": 0.0,
            },
            "cache_admission": {
                "status": "completed",
                "duration_seconds": admission_seconds,
                "verification_level": verification_level,
            },
        },
    }


def _resolve_model_free_training_preflight(
    *,
    config: Any,
    config_path: str | Path,
    repo_root: Path,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None = None,
    phase_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve and validate all required cache state before expensive setup."""

    started = time.monotonic()
    trace = {} if phase_trace is None else phase_trace

    def measured(phase: str, body: Callable[[], Any]) -> Any:
        phase_started = time.monotonic()
        status = "completed"
        failure: dict[str, str] | None = None
        try:
            return body()
        except Exception as exc:
            status = "failed"
            code = getattr(exc, "code", "python_exception")
            failure = {
                "error_type": type(exc).__name__[:128],
                "error_code": str(code)[:128],
            }
            raise
        finally:
            receipt: dict[str, Any] = {
                "status": status,
                "duration_seconds": max(0.0, time.monotonic() - phase_started),
                "resource_snapshot": collect_resource_snapshot(),
            }
            if failure is not None:
                receipt["failure"] = failure
            trace[phase] = receipt

    def resolve_identities() -> dict[str, Any]:
        components = load_qwen_components(config, load_model=False)
        vocab_groups = build_token_vocabulary_groups(
            components.token_identity,
            tokenizer=components.tokenizer,
        )
        resolve_qwen_runtime_controls(
            config,
            tokenizer_vocab_size=components.token_identity.tokenizer_vocab_size,
            model_logits_dtype=config.training.precision,
        )
        cache_root, cache_root_receipt = _resolve_pack_cache_root(repo_root)
        return {
            "components": components,
            "vocab_groups": vocab_groups,
            "cache_root": cache_root,
            "cache_root_receipt": cache_root_receipt,
            "train_fingerprint": build_packing_cache_fingerprint(
                config,
                components,
                dataset=config.data.train,
                split=TRAIN_SPLIT,
                vocab_groups=vocab_groups,
            ),
            "eval_fingerprint": (
                None
                if config.data.eval is None
                else build_packing_cache_fingerprint(
                    config,
                    components,
                    dataset=config.data.eval,
                    split="eval.forward",
                    vocab_groups=vocab_groups,
                )
            ),
        }

    identities = measured("cache_identity_resolution", resolve_identities)
    trace["cache_identity_resolution"]["details"] = {
        "train_identity_resolved": True,
        "eval_identity_resolved": identities["eval_fingerprint"] is not None,
    }
    components = identities["components"]
    vocab_groups = identities["vocab_groups"]
    cache_root = identities["cache_root"]
    cache_root_receipt = identities["cache_root_receipt"]

    def admit_publications() -> tuple[dict[str, Any], Any, dict[str, Any] | None]:
        train_cache = _admit_model_free_pack_cache(
            config,
            components,
            vocab_groups=vocab_groups,
            dataset=config.data.train,
            split=TRAIN_SPLIT,
            cache_root=cache_root,
            config_path=config_path,
            verification_level="manifest",
            resolved_fingerprint=str(identities["train_fingerprint"]),
        )
        schedule = resolve_planned_step_schedule(
            config,
            packs_per_epoch=int(train_cache["micro_step_count"]),
            world_size=world_size,
            source_config_path=str(config_path),
        )
        eval_cache = None
        if config.data.eval is not None:
            eval_cache = _admit_model_free_pack_cache(
                config,
                components,
                vocab_groups=vocab_groups,
                dataset=config.data.eval,
                split="eval.forward",
                cache_root=cache_root,
                config_path=config_path,
                verification_level="payloads",
                resolved_fingerprint=str(identities["eval_fingerprint"]),
            )
        return train_cache, schedule, eval_cache

    train_cache, schedule, eval_cache = measured(
        "cache_publication_admission", admit_publications
    )
    trace["cache_publication_admission"]["details"] = {
        "train_verification_level": "manifest",
        "eval_verification_level": (
            "not_applicable" if eval_cache is None else "payloads"
        ),
        "train_manifest_micro_step_count": int(train_cache["micro_step_count"]),
        "eval_manifest_micro_step_count": (
            0 if eval_cache is None else int(eval_cache["micro_step_count"])
        ),
        "immutable_publication_admitted": True,
    }

    eval_reduction = _resolve_converged_eval_reduction_receipt(
        pack_count=(
            None if eval_cache is None else int(eval_cache["micro_step_count"])
        ),
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
    )
    trace["cache_publication_admission"]["details"]["eval_reduction"] = dict(
        eval_reduction
    )

    def hydrate_train_rank() -> tuple[SupervisedMicroStep, ...]:
        try:
            return load_rank_micro_steps_from_cache(
                train_cache["cache_dir"],
                cache_root=cache_root,
                expected_fingerprint=str(train_cache["fingerprint"]),
                schedule=schedule,
                rank=rank,
                world_size=world_size,
            )
        except PackingCacheInvalidError as exc:
            raise _cache_preflight_error(
                cache_dir=Path(train_cache["cache_dir"]),
                cache_root=cache_root,
                config_path=config_path,
                fingerprint=str(train_cache["fingerprint"]),
                split=TRAIN_SPLIT,
                error=exc,
            ) from exc

    train_micro_steps = measured("train_rank_hydration", hydrate_train_rank)
    trace["train_rank_hydration"]["details"] = {
        "retained_micro_step_count": len(train_micro_steps),
        "rank": rank,
        "world_size": world_size,
    }
    return {
        "components": components,
        "vocab_groups": vocab_groups,
        "cache_root": cache_root,
        "cache_root_receipt": cache_root_receipt,
        "train_cache": train_cache,
        "train_micro_steps": train_micro_steps,
        "eval_cache": eval_cache,
        "eval_reduction": eval_reduction,
        "schedule": schedule,
        "rank": rank,
        "world_size": world_size,
        "duration_seconds": time.monotonic() - started,
        "phase_trace": trace,
    }


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
                "training_state_manifest_file_sha256": _file_sha256(manifest_path),
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
) -> tuple[
    RunDirectory,
    str,
    str,
    RunWriter | None,
    int,
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
]:
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
    created_at = _utc_now()
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
    return (
        run_directory,
        run_id,
        segment_id,
        writer,
        measurement_warmup_steps,
        provenance,
        continuation_lineage,
    )


def run_training_pipeline(
    config_path: str | Path,
    *,
    measurement_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = execution_plan.build_training_execution_plan(
        config_path,
        measurement_context=measurement_context,
    )
    entry_started_at = plan.entry_started_at
    entry_started_monotonic = plan.entry_started_monotonic
    entry_resource_snapshot = dict(plan.entry_resources)
    repo_root = plan.repo_root
    resolved_config = plan.resolved_config
    config = resolved_config.config
    launch_rank = plan.launch_rank
    launch_world_size = plan.launch_world_size
    rank_control_plane: control_plane.RankControlPlane | None = None
    runtime_determinism: Mapping[str, Any] | None = None
    try:
        rank_control_plane = control_plane.RankControlPlane.open(
            rank=launch_rank,
            world_size=launch_world_size,
        )
        runtime_determinism = _establish_converged_runtime_determinism(
            config.runtime,
            rank=launch_rank,
            world_size=launch_world_size,
            rank_report_gatherer=rank_control_plane.gatherer,
            phase="pipeline_entry",
        )
        (
            run_directory,
            run_id,
            run_segment_id,
            writer,
            measurement_warmup_steps,
            provenance,
            continuation_lineage,
        ) = _initialize_model_free_run_owner(
            config=config,
            resolved_config=resolved_config,
            repo_root=repo_root,
            launch_rank=launch_rank,
            launch_world_size=launch_world_size,
            preflight_gatherer=rank_control_plane.gatherer,
            measurement_context=plan.measurement_context,
            entry_started_at=entry_started_at,
        )
    except BaseException:
        if rank_control_plane is not None:
            rank_control_plane.close()
        raise
    lifecycle: dict[str, Any] = {
        "completed_steps": 0,
        "consumed_packs": 0,
        "checkpoint_event_count": 0,
        "optimizer_update_status": None,
        "finite_status": None,
        "active_phase": None,
        "phase_started_monotonic": None,
        "measurement_warmup_steps": measurement_warmup_steps,
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
        "world_size": launch_world_size,
        "entry_started_at": entry_started_at,
        "entry_started_monotonic": entry_started_monotonic,
    }
    preflight: Mapping[str, Any] | None = None
    resolved_forward_input_provider: ResolvedForwardInputProviderMode | None = None
    profile_sync_policy_configured = False
    accelerator: Any | None = None
    preflight_phase_trace: dict[str, Any] = {}
    try:
        _begin_run_phase(
            writer,
            lifecycle,
            "config_provenance_resolution",
            started_at=entry_started_at,
            started_monotonic=entry_started_monotonic,
            resources=entry_resource_snapshot,
        )
        pinned_runtime_baseline = _resolve_shared_pinned_runtime_baseline(
            provenance,
            attention_backend=str(config.model.attn_implementation),
            rank=launch_rank,
            world_size=launch_world_size,
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
                _runtime_determinism_run_policy(
                    runtime_determinism,
                    pinned_runtime_baseline=pinned_runtime_baseline,
                ),
            )
        profile_sync_timings = _resolve_converged_profile_sync_timing_selector(
            rank=launch_rank,
            world_size=launch_world_size,
            rank_report_gatherer=rank_control_plane.gatherer,
        )
        set_qwen_profile_sync_timing_policy(bool(profile_sync_timings["enabled"]))
        set_trainer_profile_sync_timing_policy(bool(profile_sync_timings["enabled"]))
        profile_sync_policy_configured = True
        if writer is not None:
            writer.bind_policy_identity(
                "profile_sync_timings",
                {"schema_version": 1, **profile_sync_timings},
            )
        resolved_forward_input_provider = (
            _resolve_converged_forward_input_provider_mode(
                getattr(
                    config.training,
                    "forward_input_provider_mode",
                    "synchronous",
                ),
                rank=launch_rank,
                world_size=launch_world_size,
                rank_report_gatherer=rank_control_plane.gatherer,
                receipt_sink=_phase_receipt_sink(
                    lifecycle, "config_provenance_resolution"
                ),
            )
        )
        _finish_run_phase(writer, lifecycle, "config_provenance_resolution")
        preflight = rank_control_plane.converge(
            "cache_preflight",
            lambda: _resolve_model_free_training_preflight(
                config=config,
                config_path=resolved_config.entry_config_path,
                repo_root=repo_root,
                rank=launch_rank,
                world_size=launch_world_size,
                rank_report_gatherer=rank_control_plane.gatherer,
                phase_trace=preflight_phase_trace,
            ),
            local_details=lambda: {"phase_trace": preflight_phase_trace},
            receipt_sink=_preflight_receipt_sink(lifecycle),
        )
        _persist_preflight_phase_receipts(writer, lifecycle)
        rank_control_plane.close()
        _begin_run_phase(writer, lifecycle, "cache_admission")
        # Accelerator construction stays outside the convergence claim: a rank
        # that fails before a post-Accelerator process group exists cannot safely
        # publish status to peers. The boundary below covers every live rank only
        # after construction has established that common distributed surface.
        accelerator = _build_accelerator(config.training.precision)
        rank_control_plane.bind_accelerator(accelerator)

        def validate_post_accelerator_runtime() -> None:
            validate_accelerator_runtime(
                accelerator,
                expected_mixed_precision=config.training.precision,
            )
            accelerate_rank = int(accelerator.process_index)
            accelerate_world_size = int(accelerator.num_processes)
            if (
                accelerate_rank != launch_rank
                or accelerate_world_size != launch_world_size
            ):
                raise RuntimeContractError(
                    "Accelerate identity disagrees with the admitted model-free launch identity",
                    code="runtime.preflight_accelerate_identity_mismatch",
                    context={
                        "preflight_rank": launch_rank,
                        "preflight_world_size": launch_world_size,
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
        return _run_initialized_training(
            repo_root=repo_root,
            resolved_config=resolved_config,
            config=config,
            accelerator=accelerator,
            run_directory=run_directory,
            run_id=run_id,
            run_segment_id=run_segment_id,
            writer=writer,
            lifecycle=lifecycle,
            rank_report_gatherer=rank_control_plane.gatherer,
            preflight=preflight,
            resolved_forward_input_provider=resolved_forward_input_provider,
            provenance=provenance,
            continuation_lineage=continuation_lineage,
            pinned_runtime_baseline=pinned_runtime_baseline,
            profile_sync_timings=profile_sync_timings,
        )
    except BaseException as exc:
        if writer is not None:
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
                        completed_at=_utc_now(),
                        duration_seconds=max(
                            0.0, time.monotonic() - entry_started_monotonic
                        ),
                        reason="preflight_control_plane_failure",
                        rank_resources=rank_cpu_resources_from_metric_rows(
                            None,
                            world_size=launch_world_size,
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
                    updated_at=_utc_now(),
                    completed_steps=int(lifecycle["completed_steps"]),
                    consumed_packs=int(lifecycle["consumed_packs"]),
                    checkpoint_event_count=int(lifecycle["checkpoint_event_count"]),
                    optimizer_update_status=lifecycle["optimizer_update_status"],
                    finite_status=lifecycle["finite_status"],
                    terminal_error=f"{type(exc).__name__}: {exc}",
                    entry_started_monotonic=entry_started_monotonic,
                )
            except BaseException:
                pass
        raise
    finally:
        if profile_sync_policy_configured:
            set_qwen_profile_sync_timing_policy(None)
            set_trainer_profile_sync_timing_policy(None)
        rank_control_plane.close()


def _packing_policy_receipt(config: Any) -> dict[str, Any]:
    packing = config.packing
    return {
        "schema_version": 2,
        "global_max_length": int(packing.global_max_length),
        "planner": build_pack_plan_policy_identity(
            policy=str(packing.policy),
            window_size=packing.window_size,
            lookahead=packing.lookahead,
            seed=int(packing.seed),
            worker_count=int(packing.worker_count),
            cursor_byte_budget=int(packing.cursor_byte_budget),
            fragment_item_budget=int(packing.fragment_item_budget),
            fragment_byte_budget=int(packing.fragment_byte_budget),
        ),
        "fragment_pack_budget": packing.max_packs_per_fragment,
        "train_order": str(config.data.train_order),
        "intra_image_object_order": str(config.template.object_ordering),
    }


def prepare_training_pack_caches(config_path: str | Path) -> dict[str, Any]:
    """Materialize all packing caches before distributed model startup."""

    entry_started_at = _utc_now()
    entry_started_monotonic = time.monotonic()
    initial_resources = collect_resource_snapshot()
    repo_root = Path.cwd().resolve()
    resolved_config = load_train_config(config_path)
    config = resolved_config.config
    runtime_determinism = _establish_converged_runtime_determinism(
        config.runtime,
        rank=0,
        world_size=1,
        rank_report_gatherer=None,
        phase="pack_cache_preparation",
    )
    provenance = collect_execution_provenance(repository_root=repo_root)
    pinned_runtime_baseline = require_pinned_runtime_baseline(
        provenance=provenance,
        attention_backend=str(config.model.attn_implementation),
    )
    components = load_qwen_components(config, load_model=False)
    vocab_groups = build_token_vocabulary_groups(
        components.token_identity,
        tokenizer=components.tokenizer,
    )
    config_phase_seconds = time.monotonic() - entry_started_monotonic
    pack_cache_root, pack_cache_root_receipt = _resolve_pack_cache_root(repo_root)
    train_cache = _resolve_or_build_train_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        accelerator=None,
        rank=0,
        verification_level="payloads",
        cache_root=pack_cache_root,
    )
    eval_cache = _resolve_eval_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        accelerator=None,
        rank=0,
        cache_root=pack_cache_root,
    )
    final_resources = collect_resource_snapshot()
    resource_high_water = merge_resource_high_water(initial_resources, final_resources)
    split_caches = [train_cache, *([] if eval_cache is None else [eval_cache])]
    aggregate_phases = {
        phase: _aggregate_cache_phase(split_caches, phase)
        for phase in ("cache_preparation", "cache_publication", "cache_admission")
    }
    return {
        "entry_config_path": str(resolved_config.entry_config_path),
        "resolved_config_fingerprint": resolved_config.fingerprint,
        "model_loaded": False,
        "provenance": provenance,
        "policy_identities": {
            "upstream_runtime_baseline": pinned_runtime_baseline,
            "runtime_determinism": _runtime_determinism_run_policy(
                runtime_determinism,
                pinned_runtime_baseline=pinned_runtime_baseline,
            ),
            "packing": _packing_policy_receipt(config),
            "cache": {
                "schema_version": 1,
                "root": pack_cache_root_receipt,
                "train_fingerprint": str(train_cache["fingerprint"]),
                "eval_fingerprint": (
                    None if eval_cache is None else str(eval_cache["fingerprint"])
                ),
            },
        },
        "measurement": {
            "schema_version": 1,
            "context": {
                "comparison_arm": "compatibility_reference",
                "wall_clock_scope": "prepare_training_pack_caches_entry_to_return",
                "workload_identity": resolved_config.fingerprint,
                "world_size": 1,
            },
            "started_at": entry_started_at,
            "completed_at": _utc_now(),
            "duration_seconds": time.monotonic() - entry_started_monotonic,
            "phases": {
                "config_provenance_resolution": {
                    "status": "completed",
                    "duration_seconds": config_phase_seconds,
                },
                **aggregate_phases,
            },
            "resource_high_water": resource_high_water,
        },
        "train": _pack_cache_preparation_receipt(train_cache),
        "eval": (
            None if eval_cache is None else _pack_cache_preparation_receipt(eval_cache)
        ),
    }


def _aggregate_cache_phase(
    split_caches: Sequence[Mapping[str, Any]], phase: str
) -> dict[str, Any]:
    receipts = [cache["phase_receipt"][phase] for cache in split_caches]
    statuses = {str(receipt["status"]) for receipt in receipts}
    duration_seconds = sum(float(receipt["duration_seconds"]) for receipt in receipts)
    if statuses == {"not_run_cache_hit"}:
        return {
            "status": "not_run",
            "reason": "all_cache_hits",
            "duration_seconds": duration_seconds,
        }
    result: dict[str, Any] = {
        "status": "completed",
        "duration_seconds": duration_seconds,
    }
    if "not_run_cache_hit" in statuses:
        result["reason"] = "mixed_cache_hits_and_builds"
    return result


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
        pack_cache_root, pack_cache_root_receipt = _resolve_pack_cache_root(repo_root)
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
                _packing_policy_receipt(config),
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
        train_cache = _resolve_or_build_train_pack_cache(
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
        _bind_cache_materialization(writer, "train", train_cache)
    train_micro_steps = _attach_image_processors_to_micro_steps(
        train_micro_steps,
        image_processor=_qwen_image_processor(components),
    )
    train_micro_steps = _apply_fa2_branch_proof_policy(train_micro_steps, config)
    if preflight is None:
        _finish_run_phase(writer, lifecycle, "cache_admission")
    if writer is not None and preflight is None:
        for phase in ("cache_preparation", "cache_publication"):
            phase_receipt = train_cache["phase_receipt"][phase]
            if phase_receipt["status"] == "completed":
                writer.record_completed_phase(
                    phase,
                    completed_at=_utc_now(),
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
        _resolve_eval_pack_cache(
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
        _resolve_converged_eval_reduction_receipt(
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
        ) = _hydrate_eval_micro_steps_from_cache(
            eval_cache,
            cache_root=pack_cache_root,
            rank=int(accelerator.process_index),
            world_size=int(accelerator.num_processes),
            rank_report_gatherer=rank_report_gatherer,
            reduction_receipt=eval_reduction_receipt,
            receipt_sink=_phase_receipt_sink(lifecycle, "evaluation_hydration"),
        )
        eval_micro_steps = _attach_image_processors_to_micro_steps(
            eval_micro_steps,
            image_processor=_qwen_image_processor(components),
        )
        if writer is not None:
            _bind_cache_materialization(writer, "eval", eval_cache)
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
    eval_micro_steps = _apply_fa2_branch_proof_policy(eval_micro_steps, config)
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
            packing=_packing_policy_receipt(config),
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
        on_completed_step=_train_logging_handler(
            writer,
            lifecycle,
            runtime,
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
                updated_at=_utc_now(),
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


def build_base_micro_steps(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    materialization_workers: int | None = None,
) -> tuple[SupervisedMicroStep, ...]:
    return _build_micro_steps_for_dataset(
        config,
        components,
        vocab_groups,
        dataset=config.data.train,
        split=TRAIN_SPLIT,
        materialization_workers=materialization_workers,
    )


def _resolve_or_build_train_pack_cache(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    repo_root: Path,
    accelerator: Any,
    rank: int = 0,
    verification_level: str,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    return _resolve_or_build_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        dataset=config.data.train,
        split=TRAIN_SPLIT,
        accelerator=accelerator,
        rank=rank,
        verification_level=verification_level,
        cache_root=cache_root,
        build_micro_steps=lambda workers: build_base_micro_steps(
            config,
            components,
            vocab_groups,
            materialization_workers=workers,
        ),
    )


def _resolve_eval_pack_cache(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    repo_root: Path,
    accelerator: Any,
    rank: int = 0,
    cache_root: Path | None = None,
) -> dict[str, Any] | None:
    if config.data.eval is None:
        return None
    return _resolve_or_build_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        dataset=config.data.eval,
        split="eval.forward",
        accelerator=accelerator,
        rank=rank,
        verification_level="payloads",
        cache_root=cache_root,
        build_micro_steps=lambda workers: _build_micro_steps_for_dataset(
            config,
            components,
            vocab_groups,
            dataset=config.data.eval,
            split="eval.forward",
            materialization_workers=workers,
        ),
    )


def _resolve_or_build_pack_cache(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    repo_root: Path,
    dataset: Any,
    split: str,
    accelerator: Any,
    rank: int = 0,
    build_micro_steps: Callable[[int], Sequence[SupervisedMicroStep]],
    materialization_workers: int | None = None,
    verification_level: str,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    if cache_root is None:
        cache_root, _ = _resolve_pack_cache_root(repo_root)
    world_size = 1 if accelerator is None else int(accelerator.num_processes)
    determinants = build_packing_cache_determinants(
        config,
        components,
        dataset=dataset,
        split=split,
        vocab_groups=vocab_groups,
    )
    fingerprint = packing_cache_fingerprint_from_determinants(determinants)

    def revalidate_determinants() -> Mapping[str, Any]:
        return build_packing_cache_determinants(
            config,
            components,
            dataset=dataset,
            split=split,
            vocab_groups=vocab_groups,
        )

    resolved_materialization_workers = _resolve_pack_cache_materialization_workers(
        materialization_workers
    )
    materialization = build_packing_cache_materialization(
        workers=resolved_materialization_workers,
        strategy=PACKING_CACHE_MATERIALIZATION_STRATEGY,
    )
    cache_dir = cache_dir_for_fingerprint(cache_root, fingerprint)
    cache_complete_before = False
    preparation_seconds = 0.0
    publication_seconds = 0.0
    admission_seconds = 0.0
    try:
        admission_started = time.monotonic()
        try:
            manifest = load_cache_manifest(
                cache_dir,
                cache_root=cache_root,
                expected_fingerprint=fingerprint,
                level=verification_level,
            )
            admission_seconds = time.monotonic() - admission_started
            cache_complete_before = True
        except PackingCacheInvalidError as exc:
            admission_seconds = time.monotonic() - admission_started
            if cache_dir.exists():
                raise _cache_preflight_error(
                    cache_dir=cache_dir,
                    cache_root=cache_root,
                    config_path=repo_root / "<cache-preparation>",
                    fingerprint=fingerprint,
                    split=split,
                    error=exc,
                ) from exc
            if world_size > 1:
                raise RuntimeContractError(
                    "distributed training requires prepared packing caches; run "
                    "`python -m src.prepare_train_cache --config <path>` before "
                    "`accelerate launch`",
                    code="training.pack_cache_not_prepared",
                    context={
                        "cache_root": str(cache_root),
                        "expected_cache_target": str(cache_dir),
                        "cache_version": PACKING_CACHE_VERSION,
                        "fingerprint": fingerprint,
                        "world_size": world_size,
                    },
                )
            preparation_started = time.monotonic()
            micro_steps = tuple(build_micro_steps(resolved_materialization_workers))
            preparation_seconds = time.monotonic() - preparation_started
            publication_started = time.monotonic()
            manifest = write_micro_step_cache(
                cache_dir,
                micro_steps,
                cache_root=cache_root,
                fingerprint=fingerprint,
                determinants=determinants,
                materialization=materialization,
                determinant_revalidator=revalidate_determinants,
                augmentation=_augmentation_receipt_from_micro_steps(micro_steps),
            )
            publication_seconds = time.monotonic() - publication_started
            admission_started = time.monotonic()
            manifest = load_cache_manifest(
                cache_dir,
                cache_root=cache_root,
                expected_fingerprint=fingerprint,
                level=verification_level,
            )
            admission_seconds = time.monotonic() - admission_started
    except RuntimeContractError:
        raise
    except PackingCacheInvalidError as exc:
        if cache_dir.exists():
            raise RuntimeContractError(
                "required v3 packing cache target is occupied by an invalid "
                f"immutable publication: {cache_dir}",
                code="training.pack_cache_immutable_collision",
                context={
                    "split": split,
                    "cache_root": str(cache_root),
                    "expected_cache_target": str(cache_dir),
                    "cache_version": PACKING_CACHE_VERSION,
                    "fingerprint": fingerprint,
                    "validation_category": _occupied_cache_validation_category(
                        cache_dir,
                        expected_fingerprint=fingerprint,
                        error=exc,
                    ),
                    "automatic_recovery": "unavailable",
                },
            ) from exc
        raise RuntimeContractError(
            "packing cache publication failed before the expected target became visible",
            code="training.pack_cache_resolution_failed",
            context={
                "split": split,
                "cache_root": str(cache_root),
                "expected_cache_target": str(cache_dir),
                "cache_version": PACKING_CACHE_VERSION,
                "fingerprint": fingerprint,
                "error_type": type(exc).__name__,
            },
        ) from exc
    except BaseException as exc:
        raise RuntimeContractError(
            "failed to resolve packing cache",
            code="training.pack_cache_resolution_failed",
            context={
                "rank": rank,
                "cache_root": str(cache_root),
                "expected_cache_target": str(cache_dir),
                "cache_version": PACKING_CACHE_VERSION,
                "fingerprint": fingerprint,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1024],
            },
        ) from exc
    cache_manifest_path = manifest_path(cache_dir)
    return {
        "cache_dir": cache_dir,
        "format_version": str(manifest["version"]),
        "fingerprint": fingerprint,
        "micro_step_count": int(manifest["micro_step_count"]),
        "chunk_count": len(manifest["chunks"]),
        "chunk_size": int(manifest["chunk_size"]),
        "status": manifest["status"],
        "build_status": (
            "waited" if rank != 0 else ("hit" if cache_complete_before else "built")
        ),
        "manifest_path": cache_manifest_path,
        "manifest_sha256": _file_sha256(cache_manifest_path),
        "determinants_sha256": _sha256_json(manifest["determinants"]),
        "chunk_sha256s": [str(chunk["sha256"]) for chunk in manifest["chunks"]],
        "materialization": manifest.get("materialization"),
        "augmentation": manifest.get("augmentation"),
        "phase_receipt": {
            "cache_preparation": {
                "status": "completed"
                if not cache_complete_before
                else "not_run_cache_hit",
                "duration_seconds": preparation_seconds,
            },
            "cache_publication": {
                "status": "completed"
                if not cache_complete_before
                else "not_run_cache_hit",
                "duration_seconds": publication_seconds,
            },
            "cache_admission": {
                "status": "completed",
                "duration_seconds": admission_seconds,
                "verification_level": verification_level,
            },
        },
    }


def _pack_cache_preparation_receipt(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(cache["status"]),
        "build_status": str(cache["build_status"]),
        "cache_dir": str(cache["cache_dir"]),
        "format_version": str(cache["format_version"]),
        "fingerprint": str(cache["fingerprint"]),
        "manifest_path": str(cache["manifest_path"]),
        "manifest_sha256": str(cache["manifest_sha256"]),
        "micro_step_count": int(cache["micro_step_count"]),
        "phase_receipt": dict(cache["phase_receipt"]),
    }


def _bind_cache_materialization(
    writer: RunWriter, split: str, cache: Mapping[str, Any]
) -> None:
    writer.bind_materialization(
        split,
        cache_format_version=str(cache["format_version"]),
        semantic_fingerprint=str(cache["fingerprint"]),
        determinant_digest=str(cache["determinants_sha256"]),
    )


def _attach_image_processors_to_micro_steps(
    micro_steps: Sequence[SupervisedMicroStep],
    *,
    image_processor: Any,
) -> tuple[SupervisedMicroStep, ...]:
    if image_processor is None:
        raise RuntimeContractError(
            "cached Qwen image encodings require a runtime image_processor",
            code="training.qwen_image_processor_missing",
            context={},
        )
    return tuple(
        replace(
            micro_step,
            encoded_examples=tuple(
                _attach_image_processor_to_encoded_example(
                    encoded_example,
                    image_processor=image_processor,
                )
                for encoded_example in micro_step.encoded_examples
            ),
        )
        for micro_step in micro_steps
    )


def _attach_image_processor_to_encoded_example(
    encoded_example: Any,
    *,
    image_processor: Any,
) -> Any:
    image_encoding = getattr(encoded_example, "image_encoding", None)
    if not isinstance(image_encoding, QwenImageEncoding):
        return encoded_example
    return replace(
        encoded_example,
        image_encoding=attach_qwen_image_processor(image_encoding, image_processor),
    )


def _apply_fa2_branch_proof_policy(
    micro_steps: Sequence[SupervisedMicroStep],
    config: Any,
) -> tuple[SupervisedMicroStep, ...]:
    policy = config.model.fa2_branch_proof
    configured: list[SupervisedMicroStep] = []
    for local_index, micro_step in enumerate(micro_steps):
        capture = policy == "every_forward" or (
            policy == "first_micro_step" and local_index == 0
        )
        configured.append(
            replace(
                micro_step,
                fa2_branch_evidence=None,
                capture_fa2_branch=capture,
                require_fa2_branch_proof=capture,
                fa2_branch_proof_policy=policy,
            )
        )
    return tuple(configured)


def _qwen_image_processor(components: Any) -> Any:
    processor = getattr(components, "processor", None)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeContractError(
            "Qwen components must expose processor.image_processor for lazy image packing",
            code="training.qwen_image_processor_missing",
            context={"processor_type": type(processor).__name__},
        )
    return image_processor


def _build_encoded_examples_for_dataset(
    config: Any,
    components: Any,
    raw_examples: Sequence[Any],
    *,
    materialization_workers: int | None = None,
) -> tuple[Any, ...]:
    workers = _resolve_pack_cache_materialization_workers(materialization_workers)
    if workers == 1:
        return tuple(
            _render_and_encode_example(
                raw_example,
                config=config,
                components=components,
            )
            for raw_example in raw_examples
        )
    return _encode_examples_with_fork_process_pool(
        config,
        components,
        raw_examples,
        workers=workers,
    )


def _encode_examples_with_fork_process_pool(
    config: Any,
    components: Any,
    raw_examples: Sequence[Any],
    *,
    workers: int,
) -> tuple[Any, ...]:
    mp_context = _fork_multiprocessing_context(workers)
    global _PACK_CACHE_WORKER_CONTEXT
    _PACK_CACHE_WORKER_CONTEXT = {
        "config": config,
        "components": components,
        "raw_examples": tuple(raw_examples),
    }
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context,
        ) as executor:
            futures = [
                executor.submit(_encode_example_worker, index)
                for index in range(len(raw_examples))
            ]
            indexed_results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
    finally:
        _PACK_CACHE_WORKER_CONTEXT = None
    return _restore_encoded_example_order(
        indexed_results,
        expected_count=len(raw_examples),
    )


def _encode_example_worker(index: int) -> tuple[int, Any]:
    context = _PACK_CACHE_WORKER_CONTEXT
    if context is None:
        raise RuntimeContractError(
            "packing cache worker context was not initialized",
            code="training.pack_cache_worker_context_missing",
            context={"index": index},
        )
    raw_examples = context["raw_examples"]
    raw_example = raw_examples[index]
    encoded = _render_and_encode_example(
        raw_example,
        config=context["config"],
        components=context["components"],
    )
    return index, encoded


def _render_and_encode_example(
    raw_example: Any,
    *,
    config: Any,
    components: Any,
) -> Any:
    rendered = render_example(
        raw_example,
        config.template,
        object_order_seed=_object_order_seed(config, raw_example.example_id),
    )
    return encode_rendered_example(
        raw_example,
        rendered,
        components=components,
        processor_config=config.model.processor,
        global_max_length=config.packing.global_max_length,
        materialize_image_pixels=False,
    )


def _restore_encoded_example_order(
    indexed_results: Sequence[tuple[int, Any]],
    *,
    expected_count: int,
) -> tuple[Any, ...]:
    ordered: list[Any | None] = [None for _ in range(expected_count)]
    seen: set[int] = set()
    for index, encoded_example in indexed_results:
        if index < 0 or index >= expected_count:
            raise RuntimeContractError(
                "packing cache worker returned an out-of-range example index",
                code="training.pack_cache_worker_index",
                context={"index": index, "expected_count": expected_count},
            )
        if index in seen:
            raise RuntimeContractError(
                "packing cache worker returned a duplicate example index",
                code="training.pack_cache_worker_index_duplicate",
                context={"index": index},
            )
        seen.add(index)
        ordered[index] = encoded_example
    if len(seen) != expected_count:
        missing = sorted(set(range(expected_count)) - seen)
        raise RuntimeContractError(
            "packing cache workers did not return every encoded example",
            code="training.pack_cache_worker_index_missing",
            context={"missing_indices": missing[:16], "missing_count": len(missing)},
        )
    return tuple(encoded_example for encoded_example in ordered)


def _resolve_pack_cache_materialization_workers(workers: int | None) -> int:
    resolved_workers = (
        DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS if workers is None else workers
    )
    if isinstance(resolved_workers, bool) or not isinstance(resolved_workers, int):
        raise RuntimeContractError(
            "packing cache materialization workers must be an integer",
            code="training.pack_cache_workers_invalid",
            context={"workers": resolved_workers},
        )
    if resolved_workers <= 0:
        raise RuntimeContractError(
            "packing cache materialization workers must be positive",
            code="training.pack_cache_workers_invalid",
            context={"workers": resolved_workers},
        )
    return resolved_workers


def _fork_multiprocessing_context(workers: int) -> Any:
    available_start_methods = tuple(multiprocessing.get_all_start_methods())
    if "fork" not in available_start_methods:
        raise RuntimeContractError(
            "parallel packing-cache materialization requires multiprocessing fork",
            code="training.pack_cache_workers_unavailable",
            context={
                "workers": workers,
                "available_start_methods": available_start_methods,
            },
        )
    try:
        return multiprocessing.get_context("fork")
    except ValueError as exc:
        raise RuntimeContractError(
            "parallel packing-cache materialization could not acquire fork context",
            code="training.pack_cache_workers_unavailable",
            context={
                "workers": workers,
                "available_start_methods": available_start_methods,
            },
            cause=exc,
        ) from exc


def _build_micro_steps_for_dataset(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    dataset: Any,
    split: str,
    materialization_workers: int | None = None,
) -> tuple[SupervisedMicroStep, ...]:
    if dataset is None:
        raise RuntimeContractError(
            "micro-step construction requires an explicit dataset split",
            code="training.dataset_split_missing",
            context={"split": split},
        )
    augmentation_result = _materialize_raw_examples_for_dataset(
        config,
        dataset,
        split=split,
    )
    raw_examples = augmentation_result.examples
    encoded_examples = _build_encoded_examples_for_dataset(
        config,
        components,
        raw_examples,
        materialization_workers=materialization_workers,
    )
    packs, pack_plan_receipt, fragment_by_pack = _materialize_pack_plan(
        config,
        encoded_examples,
    )
    supervision = build_packed_supervision(packs, encoded_examples)
    token_atoms_by_pack = index_token_atoms_by_pack(supervision)
    micro_steps: list[SupervisedMicroStep] = []
    for pack in packs:
        pack_examples = _encoded_examples_for_pack(pack, encoded_examples)
        position_inputs = build_qwen_position_inputs(
            pack,
            pack_examples,
            image_token_id=_image_token_id(components),
        )
        token_sequence = build_token_sequence_from_packed_supervision(
            pack,
            token_atoms_by_pack.get(pack.pack_index, ()),
        )
        micro_steps.append(
            SupervisedMicroStep(
                pack=pack,
                encoded_examples=pack_examples,
                position_inputs=position_inputs,
                token_sequence=token_sequence,
                vocab_groups=vocab_groups,
                metadata={
                    "split": split,
                    "pack_id": pack.pack_index,
                    "example_ids": [segment.example_id for segment in pack.segments],
                    "augmentation_receipt": augmentation_result.receipt,
                    "pack_plan": {
                        **pack_plan_receipt,
                        "fragment_sha256": fragment_by_pack[pack.pack_index],
                    },
                },
                expected_vocab_size=components.token_identity.tokenizer_vocab_size,
                fa2_model_dtype=config.training.precision,
                capture_fa2_branch=config.model.fa2_branch_proof == "every_forward",
                require_fa2_branch_proof=config.model.fa2_branch_proof
                == "every_forward",
            )
        )
    if not micro_steps:
        raise RuntimeContractError(
            "micro-step construction produced no packs",
            code="training.empty_micro_step_plan",
            context={"split": split},
        )
    return tuple(micro_steps)


def _materialize_pack_plan(
    config: Any,
    encoded_examples: Sequence[Any],
) -> tuple[tuple[PackedSequence, ...], dict[str, Any], dict[int, str]]:
    """Plan and authenticate exact memberships before supervision is built."""

    packing = config.packing
    policy_identity = build_pack_plan_policy_identity(
        policy=str(packing.policy),
        window_size=packing.window_size,
        lookahead=packing.lookahead,
        seed=int(packing.seed),
        worker_count=int(packing.worker_count),
        cursor_byte_budget=int(packing.cursor_byte_budget),
        fragment_item_budget=int(packing.fragment_item_budget),
        fragment_byte_budget=int(packing.fragment_byte_budget),
    )
    common_kwargs = {
        "global_max_length": int(packing.global_max_length),
        "seed": int(packing.seed),
        "worker_count": int(packing.worker_count),
        "cursor_byte_budget": int(packing.cursor_byte_budget),
        "fragment_item_budget": int(packing.fragment_item_budget),
        "fragment_byte_budget": int(packing.fragment_byte_budget),
    }
    if packing.policy != ONLINE_WINDOW_BINPACK:
        plan = create_pack_plan(
            encoded_examples,
            policy=str(packing.policy),
            window_size=packing.window_size,
            lookahead=packing.lookahead,
            **common_kwargs,
        )
        _reject_pack_plan_omissions((plan,))
        packs = replay_pack_plan(plan, encoded_examples)
        receipt = {
            "schema_version": 1,
            "mode": "complete_plan",
            "policy_identity": policy_identity,
            "plan_sha256": plan.canonical_sha256,
            "fragment_chain_sha256": None,
            "fragment_count": 1,
            "source_input_count": len(encoded_examples),
            "emitted_pack_count": len(packs),
        }
        return (
            packs,
            receipt,
            {pack.pack_index: plan.canonical_sha256 for pack in packs},
        )

    fragments: list[PackPlan] = []
    stream_receipt = stream_online_pack_plan_fragments(
        lambda: iter(encoded_examples),
        fragment_sink=fragments.append,
        lookahead=int(packing.lookahead),
        max_packs_per_fragment=int(packing.max_packs_per_fragment),
        **common_kwargs,
    )
    verify_pack_plan_stream_fragments(stream_receipt, fragments)
    _reject_pack_plan_omissions(fragments)
    materialized_fragments = tuple(
        (fragment, replay_pack_plan(fragment, encoded_examples))
        for fragment in fragments
    )
    packs = tuple(
        pack for _, fragment_packs in materialized_fragments for pack in fragment_packs
    )
    fragment_by_pack = {
        pack.pack_index: fragment.canonical_sha256
        for fragment, fragment_packs in materialized_fragments
        for pack in fragment_packs
    }
    receipt = {
        "schema_version": 1,
        "mode": "bounded_online_fragments",
        "policy_identity": policy_identity,
        "plan_sha256": None,
        "fragment_chain_sha256": stream_receipt.fragment_chain_sha256,
        "fragment_count": stream_receipt.fragment_count,
        "source_input_count": stream_receipt.source_input_count,
        "emitted_pack_count": stream_receipt.emitted_pack_count,
        "max_fragment_items_observed": stream_receipt.max_fragment_items_observed,
        "max_fragment_bytes_observed": stream_receipt.max_fragment_bytes_observed,
        "max_pending_items_observed": stream_receipt.max_pending_items_observed,
        "max_cursor_bytes_observed": stream_receipt.max_cursor_bytes_observed,
    }
    return packs, receipt, fragment_by_pack


def _reject_pack_plan_omissions(plans: Sequence[PackPlan]) -> None:
    rejected = [item for plan in plans for item in plan.rejected_examples]
    if not rejected:
        return
    raise RuntimeContractError(
        "pack planning rejected one or more encoded examples",
        code="training.pack_plan_rejected_examples",
        context={
            "rejected_count": len(rejected),
            "example_ids": [item.example_id for item in rejected[:16]],
        },
    )


def _materialize_raw_examples_for_dataset(
    config: Any,
    dataset: Any,
    *,
    split: str,
) -> AugmentationMaterializationResult:
    raw_examples = load_raw_examples(dataset)
    processor = build_augmentation_processor(config, split=split)
    return processor.materialize(
        raw_examples,
        split=split,
        object_ordering=config.template.object_ordering,
    )


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
        publication_started_at = _utc_now()
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
                        "training_state_manifest_file_sha256": _file_sha256(
                            manifest_path
                        ),
                    }
                publication_completed_monotonic = time.monotonic()
                writer.record_checkpoint_publication_event(
                    step=step,
                    status="completed",
                    started_at=publication_started_at,
                    completed_at=_utc_now(),
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
                    completed_at=_utc_now(),
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
        cache=_sha256_json(cache_identity),
        dependencies=_exact_resume_dependency_identity(pinned_runtime_baseline),
        policy=_sha256_json(dict(policy)),
        resolved_config=resolved_config,
        topology=build_exact_resume_topology_identity(topology_bindings),
        trainable_surface=_sha256_json(dict(trainable_surface)),
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
    return _sha256_json(projection)


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
    file_sha256 = _file_sha256(manifest_path)
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
                **_resource_scalar_metrics(post_eval_resources),
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
            per_rank_measurement = _per_rank_measurement(gathered_measurement)
            if per_rank_measurement is not None:
                row["per_rank_measurement"] = per_rank_measurement
            eval_by_step[int(scheduled_event.planned_step_id)] = row
            _append_logging_row_shared(writer=writer, row=row, runtime=runtime)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _augmentation_receipt_from_micro_steps(
    micro_steps: Sequence[SupervisedMicroStep],
) -> dict[str, Any] | None:
    for micro_step in micro_steps:
        metadata = getattr(micro_step, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        receipt = metadata.get("augmentation_receipt")
        if isinstance(receipt, Mapping):
            return dict(receipt)
    return None


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


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


def _encoded_examples_for_pack(
    pack: PackedSequence,
    encoded_examples: Sequence[Any],
) -> tuple[Any, ...]:
    examples_by_id = {
        str(getattr(example, "example_id")): example for example in encoded_examples
    }
    return tuple(examples_by_id[segment.example_id] for segment in pack.segments)


def _object_order_seed(config: Any, example_id: str) -> int | None:
    del example_id
    if config.template.object_ordering == "random":
        return int(config.runtime.seed)
    return None


def _image_token_id(components: Any) -> int | None:
    tokenizer = getattr(components, "tokenizer", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    token_id = convert("<|image_pad|>")
    return None if token_id is None else int(token_id)


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

#!/usr/bin/env python3
"""Two-phase real-Qwen packed-versus-separate Wave 2 parity probe.

``prepare`` is CPU/model-free and freezes every semantic determinant before
results exist. ``run`` revalidates that exact plan, then performs the authorized
single-GPU BF16 comparison.  Neither phase reads or writes the packing cache.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field, replace
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
from types import MethodType
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapters import (  # noqa: E402
    build_adapter_setup_plan,
    load_default_adapter_source_gate_evidence,
    setup_dora_adapter,
)
from src.augmentation.factory import build_augmentation_processor  # noqa: E402
from src.artifacts.provenance import collect_dependency_provenance  # noqa: E402
from src.config import load_train_config  # noqa: E402
from src.data import load_raw_examples  # noqa: E402
from src.losses import LossContext, LossRunner, build_token_vocabulary_groups  # noqa: E402
from src.packing import build_packed_supervision, plan_packed_sequences  # noqa: E402
from src.qwen import (  # noqa: E402
    QwenImageEncoding,
    attach_qwen_image_processor,
    build_default_special_token_selection,
    build_qwen_forward_inputs,
    build_qwen_position_inputs,
    encode_rendered_example,
    load_qwen_components,
    run_qwen_forward,
)
from src.qwen.parity import (  # noqa: E402
    BF16_ATOL,
    BF16_RTOL,
    MAX_GRADIENT_PARAMETER_SAMPLES,
    PARITY_ATTEMPT_MARKER_SCHEMA,
    PARITY_FAILURE_EVIDENCE_SCHEMA,
    PARITY_PLAN_SCHEMA,
    PARITY_RECEIPT_SCHEMA,
    ParityContractError,
    assert_absent_artifact_target,
    assert_dependency_provenance_equal,
    assert_model_weight_identity_equal,
    assert_plan_revalidated,
    attest_qwen_component_identity,
    attest_v3_runtime_config,
    base_model_weight_identity,
    bounded_failure,
    compare_gradient_inventories,
    compare_packed_gradient_repeat,
    compare_keyed_logits,
    compare_semantic_atom_inventories,
    compare_shared_denominators,
    compare_tensors,
    config_compatibility_projection,
    concrete_trainable_inventory,
    finalize_attempt_marker,
    finalize_plan,
    frozen_parent_v2_identity,
    frozen_trainable_inventory_declaration,
    frozen_tolerances,
    load_strict_json,
    merged_boundary_forward_inputs,
    negative_discriminator,
    repo_identity,
    restore_rng,
    selected_logits_by_semantic_key,
    semantic_atom_inventory,
    semantic_atom_key_inventory_sha256,
    sha256_file,
    sha256_json,
    snapshot_rng,
    snapshot_trainable_gradients,
    source_owner_identity,
    tensor_sha256,
    trainable_value_identity,
    validate_attempt_marker,
    validate_concrete_trainable_inventory,
    validate_parent_v2_plan_file,
    validate_parity_plan,
    validate_parity_receipt,
    validate_v3_config_identity,
    validate_denominator_comparison_artifact,
    validate_dependency_provenance,
    write_strict_json_atomic,
)
from src.qwen.special_token_embeddings import (  # noqa: E402
    install_special_token_embedding_deltas,
    load_default_special_token_embedding_source_gate_evidence,
    load_special_token_embedding_deltas,
)
from src.supervision import (  # noqa: E402
    build_token_sequence_from_packed_supervision,
)
from src.templates import render_example  # noqa: E402
from src.training.pipeline import (  # noqa: E402
    _build_accelerator,
    enable_training_memory_savers,
)
from src.runtime import validate_accelerator_runtime  # noqa: E402


SOURCE_OWNERS = (
    "src/qwen/parity.py",
    "scripts/probes/coordexp_swift/wave2_packed_parity.py",
    "src/config/loader.py",
    "src/artifacts/provenance.py",
    "src/artifacts/resources.py",
    "src/data/jsonl.py",
    "src/templates/renderer.py",
    "src/qwen/encoding.py",
    "src/qwen/images.py",
    "src/packing/planner.py",
    "src/packing/supervision.py",
    "src/supervision/tokens.py",
    "src/qwen/positions.py",
    "src/qwen/fa2.py",
    "src/qwen/forward.py",
    "src/losses/runner.py",
    "src/adapters/source_gates.py",
    "src/adapters/dora.py",
    "src/qwen/special_token_embeddings.py",
    "src/training/pipeline.py",
    "src/runtime/__init__.py",
    "src/runtime/train_runtime.py",
)
HOST_MEMORY_CEILING_BYTES = 64 * 1024**3
DEVICE_MEMORY_CEILING_BYTES = 76 * 1024**3
GPU_IDLE_MEMORY_LIMIT_BYTES = 1024**3
GPU_IDLE_UTILIZATION_LIMIT_PERCENT = 5
DEVICE_SAMPLER_INTERVAL_SECONDS = 0.25
DEVICE_SAMPLER_MAX_SAMPLES = 8_192
FLASH_ATTENTION_DETERMINISTIC_VALUE = "1"


@dataclass(frozen=True)
class ParityMaterials:
    resolved: Any
    components: Any
    raw_examples: tuple[Any, ...]
    encoded_examples: tuple[Any, ...]
    packed: Any
    packed_positions: Any
    packed_tokens: Any
    references: tuple[tuple[Any, Any, Any, tuple[Any, ...]], ...]
    vocab_groups: Any
    augmentation_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutedArm:
    name: str
    total_loss: torch.Tensor
    loss_artifact: dict[str, Any]
    keyed_logits: dict[Any, torch.Tensor]
    gradients: tuple[Any, ...]
    forward_receipts: tuple[dict[str, Any], ...]
    forward_logits_dtypes: tuple[str, ...]
    autocast_observations: tuple[dict[str, Any], ...]
    forward_elapsed_ns: int
    backward_call_count: int
    backward_events: tuple[dict[str, Any], ...]
    gradient_clear_count: int


@dataclass
class _FailureEvidenceAccumulator:
    """Bounded, tensor-free evidence retained across a terminal probe failure."""

    requested_device: str
    stage_reached: str = "initialized"
    completed_phases: list[str] = dataclass_field(default_factory=list)
    fields: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)

    def reach(self, stage: str) -> None:
        self.stage_reached = stage

    def complete_phase(self, phase: str) -> None:
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)

    def record(self, name: str, value: Mapping[str, Any]) -> None:
        self.fields[name] = dict(value)

    def evidence(self) -> dict[str, Any]:
        ordered_fields = (
            "source_identity",
            "model_identity_attestation",
            "execution",
            "attempt_marker",
            "trainable_inventory",
            "arms",
            "proof",
            "comparisons",
            "negative_discriminator",
            "timings",
            "gpu_memory",
            "measurement",
        )
        return {
            "schema": PARITY_FAILURE_EVIDENCE_SCHEMA,
            "stage_reached": self.stage_reached,
            "completed_phases": list(self.completed_phases),
            "completed_fields": [
                name for name in ordered_fields if bool(self.fields.get(name))
            ],
        }


class _BoundedDeviceSampler:
    """Bounded whole-device nvidia-smi sampler for shared-use HWM evidence."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._sample_count = 0
        self._hwm_memory_used_bytes = 0
        self._hwm_utilization_percent = 0
        self._first_monotonic_ns: int | None = None
        self._last_monotonic_ns: int | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise ParityContractError(
                "GPU sampler cannot be started twice",
                code="qwen.parity.gpu_sampler_state",
                context={},
            )
        self._record_one()
        self._thread = threading.Thread(
            target=self._run,
            name="wave2-bounded-nvidia-smi-sampler",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(DEVICE_SAMPLER_INTERVAL_SECONDS):
            try:
                self._record_one()
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                self._stop.set()
                return

    def _record_one(self) -> None:
        with self._lock:
            next_index = self._sample_count
            if next_index >= DEVICE_SAMPLER_MAX_SAMPLES:
                raise ParityContractError(
                    "GPU sampler exceeded its frozen sample bound",
                    code="qwen.parity.gpu_sampler_bound",
                    context={"maximum_samples": DEVICE_SAMPLER_MAX_SAMPLES},
                )
        sample = _nvidia_smi_sample(self._device, sample_index=next_index)
        with self._lock:
            if self._sample_count != next_index:
                raise ParityContractError(
                    "GPU sampler index changed concurrently",
                    code="qwen.parity.gpu_sampler_state",
                    context={},
                )
            monotonic_ns = int(sample["monotonic_ns"])
            self._sample_count += 1
            self._hwm_memory_used_bytes = max(
                self._hwm_memory_used_bytes, int(sample["memory_used_bytes"])
            )
            self._hwm_utilization_percent = max(
                self._hwm_utilization_percent, int(sample["utilization_percent"])
            )
            if self._first_monotonic_ns is None:
                self._first_monotonic_ns = monotonic_ns
            self._last_monotonic_ns = monotonic_ns

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise ParityContractError(
                "bounded GPU sampler failed during probe execution",
                code="qwen.parity.gpu_sampler",
                context={"error": type(error).__name__},
                cause=error,
            ) from error

    def stop(self, *, raise_error: bool = True) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=15.0)
            if thread.is_alive():
                raise ParityContractError(
                    "bounded GPU sampler did not stop",
                    code="qwen.parity.gpu_sampler_stop",
                    context={},
                )
        try:
            self._record_one()
        except BaseException as exc:
            with self._lock:
                if self._error is None:
                    self._error = exc
        if raise_error:
            self.raise_if_failed()
        return self.artifact(status="completed" if self._error is None else "failed")

    def artifact(self, *, status: str = "running") -> dict[str, Any]:
        with self._lock:
            return {
                "status": status,
                "interval_seconds": DEVICE_SAMPLER_INTERVAL_SECONDS,
                "sample_count": int(self._sample_count),
                "maximum_samples": DEVICE_SAMPLER_MAX_SAMPLES,
                "hwm_memory_used_bytes": int(self._hwm_memory_used_bytes),
                "hwm_utilization_percent": int(self._hwm_utilization_percent),
                "first_monotonic_ns": self._first_monotonic_ns,
                "last_monotonic_ns": self._last_monotonic_ns,
            }


def build_plan(
    *,
    config_path: str | Path,
    parent_v2_plan_path: str | Path,
    source_indices: tuple[int, int] = (0, 1),
) -> tuple[dict[str, Any], ParityMaterials]:
    _require_flash_attention_deterministic()
    materials = _materialize(config_path=config_path, source_indices=source_indices)
    resolved = materials.resolved
    config = resolved.config
    samples = []
    for source_index, raw, encoded in zip(
        source_indices,
        materials.raw_examples,
        materials.encoded_examples,
        strict=True,
    ):
        rendered = render_example(
            raw,
            config.template,
            object_order_seed=(
                int(config.runtime.seed)
                if config.template.object_ordering == "random"
                else None
            ),
        )
        samples.append(
            {
                "source_index": source_index,
                "example_id": raw.example_id,
                "source": raw.source.to_artifact_dict(),
                "raw_example_sha256": sha256_json(raw.to_artifact_dict()),
                "image_path": str(raw.image.path),
                "image_content_sha256": sha256_file(raw.image.path),
                "rendered_sha256": sha256_json(rendered.to_artifact_dict()),
                "encoded_sha256": sha256_json(encoded.to_artifact_dict()),
                "input_ids_sha256": _int_sequence_sha256(encoded.input_ids),
                "supervision_sha256": sha256_json(
                    [span.to_artifact_dict() for span in encoded.supervised_token_spans]
                ),
                "object_row_identity": [
                    {
                        "object_id": obj.object_id,
                        "description": obj.description,
                        "bbox": list(obj.bbox),
                    }
                    for obj in raw.objects
                ],
            }
        )

    packed_boundaries = tuple(materials.packed_positions.segment_boundaries)
    merged_boundaries = (0, materials.packed.length)
    references = []
    for pack, positions, tokens, _examples in materials.references:
        references.append(
            {
                "example_id": pack.segments[0].example_id,
                "segment_boundaries": list(positions.segment_boundaries),
                "input_ids_sha256": _int_sequence_sha256(pack.input_ids),
                "position_ids_sha256": tensor_sha256(positions.position_ids),
                "supervision_sha256": sha256_json(tokens.to_artifact_dict()),
            }
        )
    loss_normalization_preflight, *_ = _build_loss_normalization_preflight(materials)

    parent_v2 = _load_and_attest_parent_v2_plan(parent_v2_plan_path)
    compatibility_projection = config_compatibility_projection(resolved.config_dict)
    runtime_config_attestation = {
        "schema": "coordexp-swift-wave2-runtime-config-attestation-v1",
        "status": "passed",
        "config_fingerprint": resolved.fingerprint,
        "field_path": "training.forward_input_provider_mode",
        "required_value": "synchronous",
        "resolved_value": str(config.training.forward_input_provider_mode),
        "compatibility_projection_sha256": sha256_json(compatibility_projection),
    }
    plan_body = {
        "schema": PARITY_PLAN_SCHEMA,
        "status": "prepared",
        "config_identity": {
            "entry_path": str(resolved.entry_config_path),
            "fingerprint": resolved.fingerprint,
            "schema_version": resolved.schema_version,
            "loader_version": resolved.loader_version,
            "resolved_config_sha256": sha256_json(resolved.config_dict),
            "sources": [source.to_artifact_dict() for source in resolved.sources],
            "compatibility_projection": compatibility_projection,
            "runtime_config_attestation": runtime_config_attestation,
        },
        "repo_identity": repo_identity(REPO_ROOT),
        "dependency_identity": _dependency_identity(),
        "model_identity": materials.components.to_artifact_dict(),
        "model_weight_identity": base_model_weight_identity(
            materials.components.base_model_path
        ),
        "parent_v2": frozen_parent_v2_identity(),
        "selection": {
            "split": "train",
            "source_indices": list(source_indices),
            "example_ids": [raw.example_id for raw in materials.raw_examples],
            "selection_policy": "explicit_source_indices_no_switching",
        },
        "samples": samples,
        "arms": {
            "packed_primary": {
                "example_ids": [
                    segment.example_id for segment in materials.packed.segments
                ],
                "segment_boundaries": list(packed_boundaries),
                "input_ids_sha256": _int_sequence_sha256(materials.packed.input_ids),
                "position_ids_sha256": tensor_sha256(
                    materials.packed_positions.position_ids
                ),
                "supervision_sha256": sha256_json(
                    materials.packed_tokens.to_artifact_dict()
                ),
                "image_content_sha256": [
                    item["image_content_sha256"] for item in samples
                ],
                "loss_wiring": "one_microstep_one_shared_denominator",
                "proof_scope": "packed_forward_only_before_backward",
            },
            "packed_repeat": {
                "example_ids": [
                    segment.example_id for segment in materials.packed.segments
                ],
                "segment_boundaries": list(packed_boundaries),
                "input_ids_sha256": _int_sequence_sha256(materials.packed.input_ids),
                "position_ids_sha256": tensor_sha256(
                    materials.packed_positions.position_ids
                ),
                "supervision_sha256": sha256_json(
                    materials.packed_tokens.to_artifact_dict()
                ),
                "image_content_sha256": [
                    item["image_content_sha256"] for item in samples
                ],
                "loss_wiring": "one_microstep_one_immediate_backward",
                "proof_scope": "disabled_identical_repeat",
            },
            "separate_reference": {
                "microstep_count": 2,
                "examples": references,
                "loss_wiring": "two_microsteps_two_immediate_backwards_one_initial_clear",
            },
            "packed_merged_boundary_negative": {
                "example_ids": [
                    segment.example_id for segment in materials.packed.segments
                ],
                "segment_boundaries": list(merged_boundaries),
                "input_ids_sha256": _int_sequence_sha256(materials.packed.input_ids),
                "position_ids_sha256": tensor_sha256(
                    materials.packed_positions.position_ids
                ),
                "supervision_sha256": sha256_json(
                    materials.packed_tokens.to_artifact_dict()
                ),
                "image_content_sha256": [
                    item["image_content_sha256"] for item in samples
                ],
                "changed_fields": ["fa2_varlen_boundaries"],
                "proof_scope": "disabled_test_only_corruption",
            },
        },
        "semantic_atom_inventory": list(
            semantic_atom_inventory(materials.packed_tokens)
        ),
        "trainable_mechanism": {
            "adapter": config.adapter.model_dump(mode="json"),
            "special_token_embeddings": config.model.special_token_embeddings.model_dump(
                mode="json"
            ),
            "losses": config.losses.model_dump(mode="json"),
            "model_precision": config.training.precision,
            "attention_implementation": config.model.attn_implementation,
            "gradient_checkpointing": {
                "enabled": True,
                "use_reentrant": False,
                "use_cache": False,
                "train_mode": True,
            },
        },
        "trainable_inventory_declaration": frozen_trainable_inventory_declaration(),
        "determinism": {
            "seed": int(config.runtime.seed),
            "torch_manual_seed": int(config.runtime.seed),
            "cuda_manual_seed_all": int(config.runtime.seed),
            "snapshot_restore_cpu_rng_between_arms": True,
            "snapshot_restore_all_cuda_rng_between_arms": True,
            "same_model_state": True,
            "same_train_mode": True,
            "optimizer": None,
            "sample_switching": False,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "flash_attention_deterministic": os.environ.get(
                "FLASH_ATTENTION_DETERMINISTIC"
            ),
            "augmentation_receipt": dict(materials.augmentation_receipt),
        },
        "tolerances": frozen_tolerances(),
        "loss_normalization_preflight": loss_normalization_preflight,
        "source_owners": source_owner_identity(REPO_ROOT, SOURCE_OWNERS),
    }
    plan = finalize_plan(plan_body)
    _assert_v3_matches_parent_v2(plan, parent_v2)
    return plan, materials


def prepare_command(args: argparse.Namespace) -> int:
    _require_flash_attention_deterministic()
    plan_path = assert_absent_artifact_target(args.plan)
    plan, _ = build_plan(
        config_path=args.config,
        parent_v2_plan_path=args.parent_v2_plan,
        source_indices=(args.first_index, args.second_index),
    )
    write_strict_json_atomic(plan_path, plan)
    print(
        json.dumps(
            {
                "status": "prepared",
                "plan": str(plan_path),
                "plan_sha256": plan["plan_sha256"],
                "example_ids": plan["selection"]["example_ids"],
            },
            sort_keys=True,
        )
    )
    return 0


def _load_and_attest_parent_v2_plan(path: str | Path) -> dict[str, Any]:
    return validate_parent_v2_plan_file(path)


def _assert_v3_matches_parent_v2(
    v3_plan: Mapping[str, Any], parent_v2: Mapping[str, Any]
) -> None:
    """Reject any v3 workload substitution relative to the immutable parent."""

    validate_v3_config_identity(
        v3_plan.get("config_identity", {}),
        parent_v2_config_identity=parent_v2.get("config_identity", {}),
    )
    exact_pairs = (
        ("model_identity", "model_identity"),
        ("model_weight_identity", "model_weight_identity"),
        ("selection", "selection"),
        ("samples", "samples"),
        ("semantic_atom_inventory", "semantic_atom_inventory"),
        ("trainable_mechanism", "trainable_mechanism"),
        ("loss_normalization_preflight", "loss_normalization_preflight"),
    )
    mismatches = [
        current_key
        for current_key, parent_key in exact_pairs
        if v3_plan.get(current_key) != parent_v2.get(parent_key)
    ]
    v3_arms = v3_plan.get("arms", {})
    v2_arms = parent_v2.get("arms", {})
    if not isinstance(v3_arms, Mapping) or not isinstance(v2_arms, Mapping):
        mismatches.append("arms")
    else:
        packed_parent = v2_arms.get("packed_clean")
        if v3_arms.get("packed_primary") != packed_parent:
            mismatches.append("arms.packed_primary")
        repeat = v3_arms.get("packed_repeat")
        if not isinstance(repeat, Mapping) or not isinstance(packed_parent, Mapping):
            mismatches.append("arms.packed_repeat")
        else:
            repeat_semantics = {
                key: value
                for key, value in repeat.items()
                if key not in {"loss_wiring", "proof_scope"}
            }
            parent_semantics = {
                key: value
                for key, value in packed_parent.items()
                if key not in {"loss_wiring", "proof_scope"}
            }
            if repeat_semantics != parent_semantics:
                mismatches.append("arms.packed_repeat")
        separate = v3_arms.get("separate_reference")
        parent_separate = v2_arms.get("separate_reference")
        if not isinstance(separate, Mapping) or not isinstance(
            parent_separate, Mapping
        ):
            mismatches.append("arms.separate_reference")
        else:
            separate_semantics = {
                key: value for key, value in separate.items() if key != "loss_wiring"
            }
            parent_semantics = {
                key: value
                for key, value in parent_separate.items()
                if key != "loss_wiring"
            }
            if separate_semantics != parent_semantics:
                mismatches.append("arms.separate_reference")
        if v3_arms.get("packed_merged_boundary_negative") != v2_arms.get(
            "packed_merged_boundary_negative"
        ):
            mismatches.append("arms.packed_merged_boundary_negative")
    if mismatches:
        raise ParityContractError(
            "v3 plan differs from the frozen parent v2 workload",
            code="qwen.parity.parent_v2_workload_drift",
            context={"fields": sorted(set(mismatches))},
        )


def _run_command_identity(args: argparse.Namespace) -> dict[str, Any]:
    argv = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "run",
        "--plan",
        str(Path(args.plan).expanduser().resolve()),
        "--parent-v2-plan",
        str(Path(args.parent_v2_plan).expanduser().resolve()),
        "--receipt",
        str(Path(args.receipt).expanduser().resolve()),
        "--attempt-marker",
        str(Path(args.attempt_marker).expanduser().resolve()),
        "--device",
        str(args.device),
    ]
    return {
        "schema": "coordexp-swift-wave2-v3-command-identity-v1",
        "argv": argv,
        "argv_sha256": sha256_json(argv),
        "script_sha256": sha256_file(__file__),
    }


def _publish_attempt_start_marker(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
    receipt_target: str | Path,
    command_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    concrete_inventory: Mapping[str, Any],
    failure_evidence: _FailureEvidenceAccumulator | None = None,
    inject_before: Any | None = None,
    inject_after: Any | None = None,
) -> dict[str, Any]:
    """Atomically consume the one v3 attempt immediately before GPU setup."""

    target = assert_absent_artifact_target(path)
    if inject_before is not None:
        inject_before()
    marker = finalize_attempt_marker(
        {
            "schema": PARITY_ATTEMPT_MARKER_SCHEMA,
            "status": "attempt_started",
            "plan_sha256": plan["plan_sha256"],
            "receipt_target": str(Path(receipt_target).expanduser().resolve()),
            "command_identity": dict(command_identity),
            "source_identity": dict(source_identity),
            "concrete_trainable_inventory": dict(concrete_inventory),
        }
    )
    validate_attempt_marker(
        marker,
        expected_plan=plan,
        expected_receipt_target=receipt_target,
    )
    reference = {
        "path": str(target),
        "schema": marker["schema"],
        "status": marker["status"],
        "marker_sha256": marker["marker_sha256"],
        "expected_receipt_target": marker["receipt_target"],
    }

    def record_linked_attempt() -> None:
        if failure_evidence is not None:
            failure_evidence.record("source_identity", dict(source_identity))
            failure_evidence.record(
                "trainable_inventory",
                {"concrete": dict(concrete_inventory)},
            )
            failure_evidence.record("attempt_marker", reference)
            failure_evidence.reach("attempt_started")

    write_strict_json_atomic(target, marker, on_linked=record_linked_attempt)
    persisted = validate_attempt_marker(
        load_strict_json(target),
        expected_plan=plan,
        expected_receipt_target=receipt_target,
    )
    persisted_reference = {
        "path": str(target),
        "schema": persisted["schema"],
        "status": persisted["status"],
        "marker_sha256": persisted["marker_sha256"],
        "expected_receipt_target": persisted["receipt_target"],
    }
    if persisted != marker or persisted_reference != reference:
        raise ParityContractError(
            "persisted attempt marker differs from the locally validated publication",
            code="qwen.parity.marker_persistence_mismatch",
            context={},
        )
    if inject_after is not None:
        inject_after()
    return reference


def run_command(args: argparse.Namespace) -> int:
    receipt_path = Path(args.receipt).expanduser()
    plan: dict[str, Any] | None = None
    evidence = _FailureEvidenceAccumulator(requested_device=str(args.device))
    evidence.record("execution", {"requested_device": str(args.device)})
    try:
        receipt_path = assert_absent_artifact_target(receipt_path)
        attempt_marker_path = assert_absent_artifact_target(args.attempt_marker)
        evidence.reach("receipt_target_preflight")
        _require_flash_attention_deterministic()
        plan = validate_parity_plan(load_strict_json(args.plan))
        evidence.reach("plan_loaded")
        evidence.record("source_identity", _planned_source_identity(plan))
        selection = plan["selection"]
        current, materials = build_plan(
            config_path=plan["config_identity"]["entry_path"],
            parent_v2_plan_path=args.parent_v2_plan,
            source_indices=tuple(selection["source_indices"]),
        )
        assert_plan_revalidated(plan, current)
        evidence.reach("plan_revalidated")
        receipt = _execute_real_probe(
            plan,
            materials,
            device_text=args.device,
            receipt_path=receipt_path,
            attempt_marker_path=attempt_marker_path,
            command_identity=_run_command_identity(args),
            failure_evidence=evidence,
        )
        validate_parity_receipt(
            receipt,
            expected_plan=plan,
            expected_receipt_target=receipt_path,
        )
        receipt_linked = False

        def record_linked_receipt() -> None:
            nonlocal receipt_linked
            receipt_linked = True

        try:
            write_strict_json_atomic(
                receipt_path,
                receipt,
                on_linked=record_linked_receipt,
            )
        except BaseException as publication_exc:
            if receipt_linked:
                try:
                    persisted = load_strict_json(receipt_path)
                    validate_parity_receipt(
                        persisted,
                        expected_plan=plan,
                        expected_receipt_target=receipt_path,
                    )
                    if persisted != receipt:
                        raise ParityContractError(
                            "post-link passed receipt differs from the intended receipt",
                            code="qwen.parity.receipt_recovery_mismatch",
                            context={},
                        )
                except BaseException:
                    pass
                else:
                    print(
                        json.dumps(
                            {
                                "terminal_status": "passed",
                                "receipt": str(receipt_path),
                                "receipt_persisted": True,
                                "post_link_recovery": True,
                                "publication_warning": bounded_failure(publication_exc),
                            },
                            sort_keys=True,
                        )
                    )
                    return 0
            raise
        print(
            json.dumps(
                {"terminal_status": "passed", "receipt": str(receipt_path)},
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        failure_receipt = _failure_receipt(
            plan,
            device_text=args.device,
            exc=exc,
            evidence=evidence,
        )
        failure_receipt_linked = False

        def record_linked_failure_receipt() -> None:
            nonlocal failure_receipt_linked
            failure_receipt_linked = True

        try:
            validate_parity_receipt(failure_receipt, expected_plan=plan)
            write_strict_json_atomic(
                receipt_path,
                failure_receipt,
                on_linked=record_linked_failure_receipt,
            )
        except BaseException as publication_exc:
            if failure_receipt_linked:
                try:
                    persisted = load_strict_json(receipt_path)
                    validate_parity_receipt(persisted, expected_plan=plan)
                    if persisted != failure_receipt:
                        raise ParityContractError(
                            "post-link failure receipt differs from the intended receipt",
                            code="qwen.parity.failure_receipt_recovery_mismatch",
                            context={},
                        )
                except BaseException:
                    pass
                else:
                    print(
                        json.dumps(
                            {
                                "terminal_status": failure_receipt["terminal_status"],
                                "receipt": str(receipt_path),
                                "receipt_persisted": True,
                                "failure": bounded_failure(exc),
                                "post_link_recovery": True,
                                "publication_warning": bounded_failure(publication_exc),
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                    )
                    return 1
            publication = _publish_publication_failure_sidecar(
                receipt_path,
                primary_failure=exc,
                publication_failure=publication_exc,
            )
            print(
                json.dumps(
                    {
                        "terminal_status": "failed",
                        "receipt": str(receipt_path),
                        "receipt_persisted": False,
                        "failure": bounded_failure(exc),
                        "publication_failure": publication,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 1
        print(
            json.dumps(
                {
                    "terminal_status": failure_receipt["terminal_status"],
                    "receipt": str(receipt_path),
                    "receipt_persisted": True,
                    "failure": bounded_failure(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


def _planned_source_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_identity": plan.get("config_identity"),
        "repo_identity": plan.get("repo_identity"),
        "dependency_identity": plan.get("dependency_identity"),
        "model_identity": plan.get("model_identity"),
        "model_weight_identity": plan.get("model_weight_identity"),
        "source_owners": plan.get("source_owners"),
    }


def _require_flash_attention_deterministic() -> str:
    observed = os.environ.get("FLASH_ATTENTION_DETERMINISTIC")
    if observed != FLASH_ATTENTION_DETERMINISTIC_VALUE:
        raise ParityContractError(
            "Wave 2 v3 requires FLASH_ATTENTION_DETERMINISTIC=1",
            code="qwen.parity.flash_attention_deterministic",
            context={"observed": observed},
        )
    return observed


def _publication_failure_sidecar_path(receipt_path: str | Path) -> Path:
    requested = Path(receipt_path).expanduser()
    return requested.with_name(f"{requested.name}.publication-failure.json")


def _publish_publication_failure_sidecar(
    receipt_path: str | Path,
    *,
    primary_failure: BaseException,
    publication_failure: BaseException,
) -> dict[str, Any]:
    sidecar_path = _publication_failure_sidecar_path(receipt_path)
    payload = {
        "schema": "coordexp-swift-wave2-receipt-publication-failure-v1",
        "terminal_status": "publication_failed",
        "receipt": str(Path(receipt_path).expanduser().resolve(strict=False)),
        "receipt_persisted": False,
        "primary_failure": bounded_failure(primary_failure),
        "publication_failure": bounded_failure(publication_failure),
    }
    sidecar_linked = False

    def record_linked_sidecar() -> None:
        nonlocal sidecar_linked
        sidecar_linked = True

    try:
        sidecar_target = assert_absent_artifact_target(sidecar_path)
        write_strict_json_atomic(
            sidecar_target,
            payload,
            on_linked=record_linked_sidecar,
        )
    except BaseException as sidecar_exc:
        if sidecar_linked:
            try:
                persisted = load_strict_json(sidecar_path)
                if persisted != payload:
                    raise ParityContractError(
                        "post-link publication sidecar differs from the intended sidecar",
                        code="qwen.parity.sidecar_recovery_mismatch",
                        context={},
                    )
            except BaseException:
                pass
            else:
                return {
                    **payload,
                    "sidecar": str(sidecar_path.resolve(strict=False)),
                    "sidecar_persisted": True,
                    "sidecar_post_link_recovery": True,
                    "sidecar_publication_warning": bounded_failure(sidecar_exc),
                }
        return {
            **payload,
            "sidecar": str(sidecar_path.resolve(strict=False)),
            "sidecar_persisted": False,
            "sidecar_failure": bounded_failure(sidecar_exc),
        }
    return {
        **payload,
        "sidecar": str(sidecar_target),
        "sidecar_persisted": True,
    }


def _materialize(
    *,
    config_path: str | Path,
    source_indices: tuple[int, int],
) -> ParityMaterials:
    if (
        len(source_indices) != 2
        or len(set(source_indices)) != 2
        or any(index < 0 for index in source_indices)
    ):
        raise ParityContractError(
            "parity preparation requires two distinct non-negative source indices",
            code="qwen.parity.selection_pair",
            context={"source_indices": list(source_indices)},
        )
    resolved = load_train_config(config_path)
    config = resolved.config
    components = load_qwen_components(config, load_model=False)
    requested_max = max(source_indices)
    source_examples = load_raw_examples(
        config.data.train, sample_limit=requested_max + 1
    )
    if len(source_examples) <= requested_max:
        raise ParityContractError(
            "source index exceeds the configured training split",
            code="qwen.parity.selection_bounds",
            context={
                "source_indices": list(source_indices),
                "loaded_count": len(source_examples),
            },
        )
    selected_source = tuple(source_examples[index] for index in source_indices)
    augmentation = build_augmentation_processor(config, split="train").materialize(
        selected_source,
        split="train",
        object_ordering=config.template.object_ordering,
    )
    raw_examples = tuple(augmentation.examples)
    encoded_examples = tuple(
        encode_rendered_example(
            raw,
            render_example(
                raw,
                config.template,
                object_order_seed=(
                    int(config.runtime.seed)
                    if config.template.object_ordering == "random"
                    else None
                ),
            ),
            components=components,
            processor_config=config.model.processor,
            global_max_length=config.packing.global_max_length,
            materialize_image_pixels=False,
        )
        for raw in raw_examples
    )
    packed_candidates = plan_packed_sequences(
        encoded_examples,
        global_max_length=config.packing.global_max_length,
    )
    if len(packed_candidates) != 1 or len(packed_candidates[0].segments) != 2:
        raise ParityContractError(
            "selected source pair does not form one two-segment production pack",
            code="qwen.parity.selection_not_one_pack",
            context={
                "source_indices": list(source_indices),
                "pack_count": len(packed_candidates),
                "segment_counts": [len(pack.segments) for pack in packed_candidates],
            },
        )
    packed = packed_candidates[0]
    image_token_id = _image_token_id(components)
    packed_positions = build_qwen_position_inputs(
        packed,
        encoded_examples,
        image_token_id=image_token_id,
    )
    packed_supervision = build_packed_supervision((packed,), encoded_examples)
    packed_tokens = build_token_sequence_from_packed_supervision(
        packed, packed_supervision
    )
    references = []
    for encoded in encoded_examples:
        (pack,) = plan_packed_sequences(
            (encoded,),
            global_max_length=config.packing.global_max_length,
        )
        positions = build_qwen_position_inputs(
            pack,
            (encoded,),
            image_token_id=image_token_id,
        )
        supervision = build_packed_supervision((pack,), (encoded,))
        tokens = build_token_sequence_from_packed_supervision(pack, supervision)
        references.append((pack, positions, tokens, (encoded,)))
    vocab_groups = build_token_vocabulary_groups(
        components.token_identity,
        tokenizer=components.tokenizer,
    )
    atom_comparison = compare_semantic_atom_inventories(
        packed_tokens,
        tuple(reference[2] for reference in references),
    )
    if not atom_comparison["passed"]:
        raise ParityContractError(
            "packed and separate semantic atom inventories differ during preparation",
            code="qwen.parity.atom_alignment",
            context=atom_comparison,
        )
    return ParityMaterials(
        resolved=resolved,
        components=components,
        raw_examples=raw_examples,
        encoded_examples=encoded_examples,
        packed=packed,
        packed_positions=packed_positions,
        packed_tokens=packed_tokens,
        references=tuple(references),
        vocab_groups=vocab_groups,
        augmentation_receipt=augmentation.receipt,
    )


def _record_comparisons_and_enforce_clean_gate(
    failure_evidence: _FailureEvidenceAccumulator,
    *,
    comparisons: Mapping[str, Any],
    negative_discriminator_artifact: Mapping[str, Any],
) -> None:
    """Persist completed comparison evidence before enforcing terminal gates."""

    failure_evidence.reach("comparisons")
    failure_evidence.record("comparisons", comparisons)
    failure_evidence.record("negative_discriminator", negative_discriminator_artifact)
    attestation = negative_discriminator_artifact.get("attestation")
    if not isinstance(attestation, Mapping):
        raise ParityContractError(
            "negative attestation is absent",
            code="qwen.parity.negative_attestation",
            context={},
        )
    if (
        attestation.get("boundary_mismatch_detected") is not True
        or attestation.get("proof_disabled") is not True
    ):
        raise ParityContractError(
            "negative attestation did not expose the merged boundary",
            code="qwen.parity.negative_attestation",
            context=dict(attestation),
        )
    if negative_discriminator_artifact.get("detected") is not True:
        raise ParityContractError(
            "merged-boundary negative control was not detected",
            code="qwen.parity.negative_undetected",
            context={},
        )
    repeat_gate = comparisons["packed_repeat_measurability"]
    if repeat_gate.get("passed") is not True:
        raise ParityContractError(
            "same-packed repeat is not measurable within the frozen gate",
            code="qwen.parity.packed_repeat_unmeasurable",
            context={
                "status": repeat_gate.get("status"),
                "coverage_passed": repeat_gate.get("coverage_passed"),
                "max_abs_diff": repeat_gate.get("max_abs_diff"),
                "max_abs_threshold": repeat_gate.get("max_abs_threshold"),
            },
        )
    clean_status = {
        "atoms": bool(comparisons["semantic_atoms"]["passed"]),
        "denominators": bool(comparisons["denominators"]["passed"]),
    }
    for arm_name in ("packed_primary_vs_separate", "packed_repeat_vs_separate"):
        arm = comparisons[arm_name]
        clean_status[f"{arm_name}.logits"] = bool(arm["supervised_logits"]["passed"])
        clean_status[f"{arm_name}.loss"] = bool(arm["loss"]["passed"])
        clean_status[f"{arm_name}.loss_terms"] = bool(
            arm["cross_arm_bf16_loss_term_scalars"]["passed"]
        )
        clean_status[f"{arm_name}.gradients"] = bool(arm["gradients"]["passed"])
    if not all(clean_status.values()):
        raise ParityContractError(
            "clean packed-versus-separate parity failed",
            code="qwen.parity.clean_failed",
            context=clean_status,
        )


def _merge_completed_failure_execution(
    retained_execution: Mapping[str, Any],
    completed_execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain completed preflight evidence in a post-comparison failure shape."""

    retained_keys = {
        "requested_device",
        "device",
        "gpu_idle_preflight",
        "accelerator",
        "model_dtype",
        "train_mode",
        "use_cache",
        "optimizer",
        "memory_savers",
        "adapter",
        "special_token_embeddings",
        "trainable_value_identity_before",
        "prepared_model_attestation",
    }
    completed_keys = {
        "device",
        "model_dtype",
        "train_mode",
        "use_cache",
        "optimizer",
        "memory_savers",
        "adapter",
        "special_token_embeddings",
        "trainable_value_identity_before",
        "trainable_value_identity_after",
        "accelerator",
    }
    if (
        set(retained_execution) != retained_keys
        or set(completed_execution) != completed_keys
    ):
        raise ParityContractError(
            "completed execution evidence cannot be merged from an unexpected shape",
            code="qwen.parity.failure_execution_merge_shape",
            context={
                "retained_fields": sorted(str(key) for key in retained_execution),
                "completed_fields": sorted(str(key) for key in completed_execution),
            },
        )
    requested_device = retained_execution["requested_device"]
    device = retained_execution["device"]
    idle = retained_execution["gpu_idle_preflight"]
    retained_accelerator = retained_execution["accelerator"]
    completed_accelerator = completed_execution["accelerator"]
    if (
        not isinstance(idle, Mapping)
        or not isinstance(retained_accelerator, Mapping)
        or not isinstance(completed_accelerator, Mapping)
    ):
        raise ParityContractError(
            "completed execution evidence contains a non-mapping subtree",
            code="qwen.parity.failure_execution_merge_shape",
            context={},
        )
    common_fields = retained_keys & completed_keys - {"accelerator"}
    if (
        requested_device != device
        or completed_execution["device"] != device
        or idle.get("requested_device") != device
        or any(
            retained_execution[field] != completed_execution[field]
            for field in common_fields
        )
        or any(
            completed_accelerator.get(field) != value
            for field, value in retained_accelerator.items()
        )
        or completed_accelerator.get("prepared_model_attestation")
        != retained_execution["prepared_model_attestation"]
    ):
        raise ParityContractError(
            "completed execution evidence contradicts retained preflight or setup evidence",
            code="qwen.parity.failure_execution_merge_binding",
            context={},
        )
    return {
        "requested_device": requested_device,
        "gpu_idle_preflight": dict(idle),
        **dict(completed_execution),
    }


def _assert_gradient_receipt_capacity(
    trainable_value_inventory: Sequence[Mapping[str, Any]],
) -> None:
    if len(trainable_value_inventory) > MAX_GRADIENT_PARAMETER_SAMPLES:
        raise ParityContractError(
            "trainable inventory exceeds the full gradient-receipt row capacity",
            code="qwen.parity.gradient_artifact_bound",
            context={
                "trainable_parameter_count": len(trainable_value_inventory),
                "maximum": MAX_GRADIENT_PARAMETER_SAMPLES,
            },
        )


def _assert_inventory_matches_setup_receipts(
    concrete_inventory: Mapping[str, Any],
    *,
    adapter_receipt: Any,
    special_token_receipt: Any,
) -> None:
    """Bind the live inventory to the two owners that created trainables."""

    rows = concrete_inventory["parameters"]
    adapter_names = set(str(name) for name in adapter_receipt.trainable_names)
    delta_names = set(str(name) for name in special_token_receipt.delta_parameter_names)
    observed_adapter_names = {
        str(row["name"]) for row in rows if row["group"] != "special_token_delta"
    }
    observed_delta_names = {
        str(row["name"]) for row in rows if row["group"] == "special_token_delta"
    }
    if (
        observed_adapter_names != adapter_names
        or observed_delta_names != delta_names
        or len(delta_names) != 1
    ):
        raise ParityContractError(
            "concrete trainable inventory differs from adapter or delta setup receipts",
            code="qwen.parity.trainable_setup_receipt_binding",
            context={
                "adapter_expected_count": len(adapter_names),
                "adapter_observed_count": len(observed_adapter_names),
                "delta_expected_count": len(delta_names),
                "delta_observed_count": len(observed_delta_names),
            },
        )


def _execute_real_probe(
    plan: Mapping[str, Any],
    materials: ParityMaterials,
    *,
    device_text: str,
    receipt_path: Path,
    attempt_marker_path: Path,
    command_identity: Mapping[str, Any],
    failure_evidence: _FailureEvidenceAccumulator,
) -> dict[str, Any]:
    device, gpu_idle_preflight = _preflight_cuda(device_text)
    failure_evidence.reach("cuda_preflight")
    failure_evidence.record(
        "execution",
        {
            "requested_device": device_text,
            "device": str(device),
            "gpu_idle_preflight": dict(gpu_idle_preflight),
        },
    )
    device_sampler = _BoundedDeviceSampler(device)
    device_sampler.start()
    try:
        receipt = _execute_real_probe_with_sampler(
            plan,
            materials,
            device=device,
            gpu_idle_preflight=gpu_idle_preflight,
            device_sampler=device_sampler,
            receipt_path=receipt_path,
            attempt_marker_path=attempt_marker_path,
            command_identity=command_identity,
            failure_evidence=failure_evidence,
        )
    except BaseException:
        sampler_artifact = device_sampler.stop(raise_error=False)
        _record_failure_resources(
            failure_evidence,
            device=device,
            device_sampler=sampler_artifact,
        )
        raise
    sampler_artifact = device_sampler.stop()
    try:
        gpu_resources = receipt["measurement"]["resources"]["gpu"]
        gpu_resources["device_sampler"] = sampler_artifact
        gpu_resources["device_used_hwm_bytes"] = int(
            sampler_artifact["hwm_memory_used_bytes"]
        )
        ceiling_comparison = {
            "host_rss_below": receipt["measurement"]["resources"]["host"][
                "rss_hwm_bytes"
            ]
            < HOST_MEMORY_CEILING_BYTES,
            "torch_reserved_below": gpu_resources["torch_peak_reserved_bytes"]
            < DEVICE_MEMORY_CEILING_BYTES,
            "device_sampler_below": gpu_resources["device_used_hwm_bytes"]
            < DEVICE_MEMORY_CEILING_BYTES,
        }
        receipt["measurement"]["resources"]["ceilings"]["comparison"] = (
            ceiling_comparison
        )
        if not all(ceiling_comparison.values()):
            raise ParityContractError(
                "Wave 2 resource ceiling failed after bounded sampler completion",
                code="qwen.parity.device_memory_ceiling",
                context={
                    "comparison": ceiling_comparison,
                    "device_sampler_hwm_bytes": gpu_resources["device_used_hwm_bytes"],
                },
            )
    except BaseException:
        _record_failure_resources(
            failure_evidence,
            device=device,
            device_sampler=sampler_artifact,
        )
        raise
    _record_failure_resources(
        failure_evidence,
        device=device,
        device_sampler=sampler_artifact,
    )
    return receipt


def _record_failure_resources(
    failure_evidence: _FailureEvidenceAccumulator,
    *,
    device: torch.device,
    device_sampler: Mapping[str, Any],
) -> None:
    try:
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    except BaseException:
        return
    failure_evidence.record(
        "gpu_memory",
        {
            "scope": "whole_probe_until_failure",
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
    )
    measurement = dict(failure_evidence.fields.get("measurement", {}))
    if not measurement:
        measurement = {
            "schema": "coordexp-swift-wave2-failure-measurement-v1",
            "completed_phases": [],
            "phase_boundary_samples": [],
        }
    measurement["failure_resource_summary"] = {
        "device": str(device),
        "torch_peak_allocated_bytes": peak_allocated,
        "torch_peak_reserved_bytes": peak_reserved,
        "device_sampler": dict(device_sampler),
    }
    failure_evidence.record("measurement", measurement)


def _build_loss_normalization_preflight(
    materials: ParityMaterials,
) -> tuple[dict[str, Any], Any, Any, Any, tuple[Any, ...], tuple[Any, ...]]:
    packed_token_sequences = (materials.packed_tokens,)
    separate_token_sequences = tuple(reference[2] for reference in materials.references)
    if len(packed_token_sequences) != 1 or len(separate_token_sequences) != 2:
        raise ParityContractError(
            "Wave 2 normalization preflight requires literal one-packed/two-separate arms",
            code="qwen.parity.denominator_arm_shape",
            context={
                "packed_context_count": len(packed_token_sequences),
                "separate_context_count": len(separate_token_sequences),
            },
        )
    loss_runner = LossRunner.from_config(materials.resolved.config.losses)
    packed_loss_plan = loss_runner.prepare_planned_step(packed_token_sequences)
    separate_loss_plan = loss_runner.prepare_planned_step(separate_token_sequences)
    comparison = compare_shared_denominators(
        packed_loss_plan,
        separate_loss_plan,
        packed_context_count=1,
        separate_context_count=2,
    )
    if not comparison["passed"]:
        raise ParityContractError(
            "packed and separate arms differ in loss-normalization semantics or context accounting",
            code="qwen.parity.denominator_mismatch",
            context=comparison,
        )
    validate_denominator_comparison_artifact(comparison)
    return (
        comparison,
        loss_runner,
        packed_loss_plan,
        separate_loss_plan,
        packed_token_sequences,
        separate_token_sequences,
    )


def _attest_fresh_runtime_config_immediately_before_marker(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-read the exact config identity at the final pre-marker boundary."""

    identity = plan["config_identity"]
    resolved = load_train_config(identity["entry_path"])
    observed_identity = {
        "entry_path": str(resolved.entry_config_path),
        "fingerprint": resolved.fingerprint,
        "schema_version": resolved.schema_version,
        "loader_version": resolved.loader_version,
        "resolved_config_sha256": sha256_json(resolved.config_dict),
        "sources": [source.to_artifact_dict() for source in resolved.sources],
    }
    expected_identity = {
        key: identity[key]
        for key in (
            "entry_path",
            "fingerprint",
            "schema_version",
            "loader_version",
            "resolved_config_sha256",
            "sources",
        )
    }
    if observed_identity != expected_identity:
        raise ParityContractError(
            "live config identity drifted during CPU model setup",
            code="qwen.parity.runtime_config_identity_drift",
            context={
                "expected_sha256": sha256_json(expected_identity),
                "observed_sha256": sha256_json(observed_identity),
            },
        )
    attestation = attest_v3_runtime_config(identity, resolved.config_dict)
    if attestation != identity["runtime_config_attestation"]:
        raise ParityContractError(
            "live runtime config attestation drifted during CPU model setup",
            code="qwen.parity.runtime_config_attestation_drift",
            context={},
        )
    return attestation


def _execute_real_probe_with_sampler(
    plan: Mapping[str, Any],
    materials: ParityMaterials,
    *,
    device: torch.device,
    gpu_idle_preflight: dict[str, Any],
    device_sampler: _BoundedDeviceSampler,
    receipt_path: Path,
    attempt_marker_path: Path,
    command_identity: Mapping[str, Any],
    failure_evidence: _FailureEvidenceAccumulator,
) -> dict[str, Any]:
    deterministic_value = _require_flash_attention_deterministic()
    planned_determinism = plan.get("determinism")
    if (
        not isinstance(planned_determinism, Mapping)
        or planned_determinism.get("flash_attention_deterministic")
        != deterministic_value
    ):
        raise ParityContractError(
            "runtime FlashAttention determinism differs from the authenticated plan",
            code="qwen.parity.flash_attention_deterministic_drift",
            context={
                "planned": (
                    None
                    if not isinstance(planned_determinism, Mapping)
                    else planned_determinism.get("flash_attention_deterministic")
                ),
                "observed": deterministic_value,
            },
        )
    phases: list[dict[str, Any]] = []
    resource_samples: list[dict[str, Any]] = []
    runtime_config_attestation = attest_v3_runtime_config(
        plan["config_identity"],
        materials.resolved.config_dict,
    )
    if (
        runtime_config_attestation
        != plan["config_identity"]["runtime_config_attestation"]
    ):
        raise ParityContractError(
            "live runtime config attestation differs from the authenticated plan",
            code="qwen.parity.runtime_config_attestation_drift",
            context={},
        )
    config = materials.resolved.config
    if config.training.precision != "bf16":
        raise ParityContractError(
            "Wave 2 real parity probe requires production BF16",
            code="qwen.parity.precision",
            context={"precision": config.training.precision},
        )
    (
        denominator_comparison,
        loss_runner,
        packed_loss_plan,
        separate_loss_plan,
        packed_token_sequences,
        separate_token_sequences,
    ) = _build_loss_normalization_preflight(materials)
    failure_evidence.reach("normalization_preflight")
    if denominator_comparison != plan["loss_normalization_preflight"]:
        raise ParityContractError(
            "runtime loss-normalization preflight differs from the authenticated plan",
            code="qwen.parity.denominator_preflight_drift",
            context={
                "planned_sha256": sha256_json(plan["loss_normalization_preflight"]),
                "runtime_sha256": sha256_json(denominator_comparison),
            },
        )
    torch.manual_seed(int(config.runtime.seed))
    model_setup_start = time.perf_counter_ns()
    components = load_qwen_components(config, load_model=True)
    if components.model is None:
        raise ParityContractError(
            "real parity probe did not load a model",
            code="qwen.parity.model_missing",
            context={},
        )
    model_identity_attestation = attest_qwen_component_identity(
        plan["model_identity"],
        components.to_artifact_dict(),
    )
    post_load_weight_identity = base_model_weight_identity(components.base_model_path)
    assert_model_weight_identity_equal(
        plan["model_weight_identity"],
        post_load_weight_identity,
    )
    post_load_dependency_identity = _dependency_identity()
    assert_dependency_provenance_equal(
        plan["dependency_identity"],
        post_load_dependency_identity,
    )
    adapter_evidence = load_default_adapter_source_gate_evidence(REPO_ROOT)
    adapter_plan = build_adapter_setup_plan(
        config.adapter,
        adapter_evidence,
        base_model_path=components.base_model_path,
    )
    adapter_result = setup_dora_adapter(components.model, adapter_plan)
    selection = build_default_special_token_selection(
        config.model.special_token_embeddings,
        components.token_identity,
    )
    special_evidence = load_default_special_token_embedding_source_gate_evidence(
        REPO_ROOT
    )
    special_result = install_special_token_embedding_deltas(
        adapter_result.model,
        selection,
        source_gate=special_evidence,
    )
    if adapter_plan.mode == "warm_start_expand_dora":
        if adapter_plan.repaired_embedding_payload_path is None:
            raise ParityContractError(
                "warm-start parity run requires repaired embedding payload",
                code="qwen.parity.embedding_payload",
                context={},
            )
        load_special_token_embedding_deltas(
            special_result,
            adapter_plan.repaired_embedding_payload_path,
            expected_base_model_path=components.base_model_path,
            expected_base_config_sha256=components.base_config_sha256,
            expected_tokenizer_sha256=components.tokenizer_sha256,
        )
    model = special_result.model
    memory_savers = enable_training_memory_savers(model)
    if not memory_savers["gradient_checkpointing_enabled"]:
        raise ParityContractError(
            "real parity probe requires production gradient checkpointing",
            code="qwen.parity.gradient_checkpointing",
            context=memory_savers,
        )
    concrete_inventory = concrete_trainable_inventory(model)
    validate_concrete_trainable_inventory(
        concrete_inventory,
        declaration=plan["trainable_inventory_declaration"],
    )
    _assert_inventory_matches_setup_receipts(
        concrete_inventory,
        adapter_receipt=adapter_result.receipt,
        special_token_receipt=special_result.receipt,
    )
    _assert_gradient_receipt_capacity(concrete_inventory["parameters"])
    loaded_source_identity = {
        **_planned_source_identity(plan),
        "dependency_identity": post_load_dependency_identity,
        "model_identity": components.to_artifact_dict(),
        "model_weight_identity": post_load_weight_identity,
    }
    _attest_fresh_runtime_config_immediately_before_marker(plan)
    _append_final_gpu_idle_check(device, gpu_idle_preflight=gpu_idle_preflight)
    device_sampler.raise_if_failed()
    attempt_marker = _publish_attempt_start_marker(
        attempt_marker_path,
        plan=plan,
        receipt_target=receipt_path,
        command_identity=command_identity,
        source_identity=loaded_source_identity,
        concrete_inventory=concrete_inventory,
        failure_evidence=failure_evidence,
    )
    torch.cuda.manual_seed_all(int(config.runtime.seed))
    accelerator = _build_exact_one_rank_accelerator(
        device=device,
        training_precision=config.training.precision,
    )
    accelerator_identity = _validate_exact_accelerator_runtime(
        accelerator,
        expected_device=device,
        expected_mixed_precision=config.training.precision,
    )
    failure_evidence.reach("accelerator_ready")
    failure_evidence.record(
        "execution",
        {
            **failure_evidence.fields["execution"],
            "accelerator": dict(accelerator_identity),
        },
    )
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    prepared = accelerator.prepare(model)
    if isinstance(prepared, (tuple, list)):
        if len(prepared) != 1:
            raise ParityContractError(
                "Accelerator returned an invalid prepared-model arity",
                code="qwen.parity.accelerator_prepare",
                context={"arity": len(prepared)},
            )
        model = prepared[0]
    else:
        model = prepared
    model.train()
    _assert_model_runtime(model)
    prepared_model_attestation = _assert_prepared_accelerator_model(
        accelerator,
        model=model,
        expected_device=device,
    )
    value_identity_before = trainable_value_identity(model)
    _assert_gradient_receipt_capacity(value_identity_before)
    failure_evidence.record("model_identity_attestation", model_identity_attestation)
    failure_evidence.record(
        "execution",
        {
            **failure_evidence.fields["execution"],
            "model_dtype": "torch.bfloat16",
            "train_mode": bool(model.training),
            "use_cache": False,
            "optimizer": None,
            "memory_savers": memory_savers,
            "adapter": adapter_result.receipt.to_artifact_dict(),
            "special_token_embeddings": special_result.receipt.to_artifact_dict(),
            "trainable_value_identity_before": list(value_identity_before),
            "prepared_model_attestation": prepared_model_attestation,
        },
    )
    _finish_phase(phases, "model_setup", model_setup_start)
    failure_evidence.reach("model_setup")
    failure_evidence.complete_phase("model_setup")
    _record_failure_progress(failure_evidence, phases, resource_samples)
    _enforce_resource_ceilings(
        device,
        phase="model_setup",
        samples=resource_samples,
        device_sampler=device_sampler,
    )

    input_start = time.perf_counter_ns()
    packed_inputs = _forward_inputs(
        materials.packed,
        materials.encoded_examples,
        materials.packed_positions,
        materials.packed_tokens,
        device=device,
        proof_policy="first_micro_step",
        image_processor=components.processor.image_processor,
    )
    reference_inputs = tuple(
        _forward_inputs(
            pack,
            examples,
            positions,
            tokens,
            device=device,
            proof_policy="disabled",
            image_processor=components.processor.image_processor,
        )
        for pack, positions, tokens, examples in materials.references
    )
    corrupted_inputs = merged_boundary_forward_inputs(packed_inputs)
    rng = snapshot_rng()
    _finish_phase(phases, "input_construction", input_start)
    failure_evidence.reach("input_construction")
    failure_evidence.complete_phase("input_construction")
    _record_failure_progress(failure_evidence, phases, resource_samples)
    _enforce_resource_ceilings(
        device,
        phase="input_construction",
        samples=resource_samples,
        device_sampler=device_sampler,
    )

    restore_rng(rng)
    _zero_grad(model)
    warmup_start = time.perf_counter_ns()
    with _capture_inner_cuda_autocast(model) as warmup_autocast_rows:
        warmup = run_qwen_forward(
            model,
            packed_inputs,
            expected_vocab_size=materials.components.token_identity.tokenizer_vocab_size,
            capture_fa2_branch=False,
            require_fa2_branch_proof=False,
        )
    warmup_autocast = _validate_inner_autocast_observations(
        warmup_autocast_rows,
        path="unmeasured_no_proof_warmup",
    )
    warmup_logits_dtype = _assert_forward_output_dtype(
        warmup,
        path="unmeasured_no_proof_warmup",
    )
    _sync(device)
    del warmup
    _zero_grad(model)
    _finish_phase(phases, "warmup_forward", warmup_start)
    failure_evidence.reach("warmup_forward")
    failure_evidence.complete_phase("warmup_forward")
    _record_failure_progress(failure_evidence, phases, resource_samples)
    _enforce_resource_ceilings(
        device,
        phase="warmup_forward",
        samples=resource_samples,
        device_sampler=device_sampler,
    )

    restore_rng(rng)
    _zero_grad(model)
    proof_off_phase_start = time.perf_counter_ns()
    proof_off_start = _timed_start(device)
    with _capture_inner_cuda_autocast(model) as proof_off_autocast_rows:
        proof_off = run_qwen_forward(
            model,
            packed_inputs,
            expected_vocab_size=materials.components.token_identity.tokenizer_vocab_size,
            capture_fa2_branch=False,
            require_fa2_branch_proof=False,
        )
    proof_off_autocast = _validate_inner_autocast_observations(
        proof_off_autocast_rows,
        path="timed_proof_off",
    )
    proof_off_logits_dtype = _assert_forward_output_dtype(
        proof_off,
        path="timed_proof_off",
    )
    proof_off_ns = _timed_end(device, proof_off_start)
    del proof_off
    _zero_grad(model)
    _finish_phase(phases, "proof_off_forward", proof_off_phase_start)
    failure_evidence.reach("proof_off_forward")
    failure_evidence.complete_phase("proof_off_forward")
    _record_failure_progress(failure_evidence, phases, resource_samples)
    _enforce_resource_ceilings(
        device,
        phase="proof_off_forward",
        samples=resource_samples,
        device_sampler=device_sampler,
    )

    restore_rng(rng)
    clean_start = time.perf_counter_ns()
    primary = _execute_arm(
        name="packed_primary",
        model=model,
        forward_inputs=(packed_inputs,),
        token_sequences=packed_token_sequences,
        loss_runner=loss_runner,
        loss_plan=packed_loss_plan,
        vocab_groups=materials.vocab_groups,
        expected_vocab_size=materials.components.token_identity.tokenizer_vocab_size,
        capture_proof=True,
        accelerator=accelerator,
        expected_inventory=concrete_inventory,
    )
    _finish_phase(phases, "packed_primary", clean_start)
    failure_evidence.reach("packed_primary")
    failure_evidence.complete_phase("packed_primary")
    failure_evidence.record("arms", {"packed_primary": _bounded_arm_artifact(primary)})
    clean_proof = primary.forward_receipts[0]["fa2_varlen"]["proof"]
    if not isinstance(clean_proof, Mapping):
        raise ParityContractError(
            "packed primary arm did not return a proof mapping",
            code="qwen.parity.proof_missing",
            context={"value_type": type(clean_proof).__name__},
        )
    failure_evidence.record("proof", dict(clean_proof))
    failure_evidence.record(
        "timings",
        {
            "clock": "time.perf_counter_ns_with_cuda_synchronize",
            "scope": "forward_only_same_packed_inputs_train_mode_no_backward",
            "proof_on_forward_ns": primary.forward_elapsed_ns,
            "proof_off_forward_ns": proof_off_ns,
            "proof_overhead_ns": primary.forward_elapsed_ns - proof_off_ns,
        },
    )
    _record_failure_progress(failure_evidence, phases, resource_samples)
    _enforce_resource_ceilings(
        device,
        phase="packed_primary",
        samples=resource_samples,
        device_sampler=device_sampler,
    )
    restore_rng(rng)
    repeat_start = time.perf_counter_ns()
    repeat = _execute_arm(
        name="packed_repeat",
        model=model,
        forward_inputs=(packed_inputs,),
        token_sequences=packed_token_sequences,
        loss_runner=loss_runner,
        loss_plan=packed_loss_plan,
        vocab_groups=materials.vocab_groups,
        expected_vocab_size=materials.components.token_identity.tokenizer_vocab_size,
        capture_proof=False,
        accelerator=accelerator,
        expected_inventory=concrete_inventory,
    )
    _finish_phase(phases, "packed_repeat", repeat_start)
    failure_evidence.reach("packed_repeat")
    failure_evidence.complete_phase("packed_repeat")
    failure_evidence.record(
        "arms",
        {
            **failure_evidence.fields["arms"],
            "packed_repeat": _bounded_arm_artifact(repeat),
        },
    )
    _record_failure_progress(failure_evidence, phases, resource_samples)
    _enforce_resource_ceilings(
        device,
        phase="packed_repeat",
        samples=resource_samples,
        device_sampler=device_sampler,
    )
    restore_rng(rng)
    reference_start = time.perf_counter_ns()
    reference = _execute_arm(
        name="separate_reference",
        model=model,
        forward_inputs=reference_inputs,
        token_sequences=separate_token_sequences,
        loss_runner=loss_runner,
        loss_plan=separate_loss_plan,
        vocab_groups=materials.vocab_groups,
        expected_vocab_size=materials.components.token_identity.tokenizer_vocab_size,
        capture_proof=False,
        accelerator=accelerator,
        expected_inventory=concrete_inventory,
    )
    _finish_phase(phases, "separate_reference", reference_start)
    failure_evidence.reach("separate_reference")
    failure_evidence.complete_phase("separate_reference")
    failure_evidence.record(
        "arms",
        {
            **failure_evidence.fields["arms"],
            "separate_reference": _bounded_arm_artifact(reference),
        },
    )
    _record_failure_progress(failure_evidence, phases, resource_samples)
    _enforce_resource_ceilings(
        device,
        phase="separate_reference",
        samples=resource_samples,
        device_sampler=device_sampler,
    )
    restore_rng(rng)
    negative_start = time.perf_counter_ns()
    negative = _execute_arm(
        name="packed_merged_boundary_negative",
        model=model,
        forward_inputs=(corrupted_inputs,),
        token_sequences=packed_token_sequences,
        loss_runner=loss_runner,
        loss_plan=packed_loss_plan,
        vocab_groups=materials.vocab_groups,
        expected_vocab_size=materials.components.token_identity.tokenizer_vocab_size,
        capture_proof=False,
        accelerator=accelerator,
        expected_inventory=concrete_inventory,
        perform_backward=False,
    )
    _finish_phase(phases, "negative_control", negative_start)
    failure_evidence.reach("negative_control")
    failure_evidence.complete_phase("negative_control")
    failure_evidence.record(
        "arms",
        {
            **failure_evidence.fields["arms"],
            "packed_merged_boundary_negative": _bounded_arm_artifact(negative),
        },
    )
    _record_failure_progress(failure_evidence, phases, resource_samples)
    _enforce_resource_ceilings(
        device,
        phase="negative_control",
        samples=resource_samples,
        device_sampler=device_sampler,
    )

    comparison_start = time.perf_counter_ns()
    atom_comparison = compare_semantic_atom_inventories(
        materials.packed_tokens,
        separate_token_sequences,
    )
    primary_logit_comparison = compare_keyed_logits(
        primary.keyed_logits, reference.keyed_logits
    )
    repeat_logit_comparison = compare_keyed_logits(
        repeat.keyed_logits, reference.keyed_logits
    )
    primary_loss_comparison = compare_tensors(
        primary.total_loss,
        reference.total_loss,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    ).to_artifact_dict()
    primary_loss_comparison["passed"] = primary_loss_comparison.pop("allclose")
    repeat_loss_comparison = compare_tensors(
        repeat.total_loss,
        reference.total_loss,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    ).to_artifact_dict()
    repeat_loss_comparison["passed"] = repeat_loss_comparison.pop("allclose")
    primary_loss_term_scalars = _compare_loss_terms(
        primary.loss_artifact, reference.loss_artifact
    )
    repeat_loss_term_scalars = _compare_loss_terms(
        repeat.loss_artifact, reference.loss_artifact
    )
    primary_gradient_comparison = compare_gradient_inventories(
        primary.gradients, reference.gradients
    )
    repeat_gradient_comparison = compare_gradient_inventories(
        repeat.gradients, reference.gradients
    )
    packed_repeat_comparison = compare_packed_gradient_repeat(
        primary.gradients, repeat.gradients
    )

    negative_logits = compare_keyed_logits(primary.keyed_logits, negative.keyed_logits)
    negative_loss_raw = compare_tensors(
        primary.total_loss,
        negative.total_loss,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    discriminator = negative_discriminator(
        clean_boundaries=packed_inputs.fa2_varlen_plan.segment_boundaries,
        negative_boundaries=corrupted_inputs.fa2_varlen_plan.segment_boundaries,
        logits_allclose=bool(negative_logits["passed"]),
        loss_allclose=negative_loss_raw.allclose,
        gradients_allclose=True,
    )
    negative_varlen = negative.forward_receipts[0]["fa2_varlen"]
    negative_proof = negative_varlen["proof"]
    observed_negative_boundaries = negative_varlen.get("segment_boundaries")
    negative_attestation = {
        "status": "rejected_against_frozen_clean_boundary",
        "expected_clean_boundaries": list(
            packed_inputs.fa2_varlen_plan.segment_boundaries
        ),
        "observed_negative_boundaries": observed_negative_boundaries,
        "boundary_mismatch_detected": observed_negative_boundaries
        != list(packed_inputs.fa2_varlen_plan.segment_boundaries),
        "proof_disabled": negative_proof is None,
        "executed_negative_varlen_receipt": negative_varlen,
    }
    value_identity_after = trainable_value_identity(model)
    if value_identity_before != value_identity_after:
        raise ParityContractError(
            "trainable parameter values changed without an optimizer",
            code="qwen.parity.trainable_value_drift",
            context={},
        )
    proof = primary.forward_receipts[0]["fa2_varlen"]["proof"]
    if not isinstance(proof, Mapping) or proof.get("status") != "pass":
        raise ParityContractError(
            "packed clean arm did not produce all-layer proof",
            code="qwen.parity.proof_missing",
            context={},
        )
    execution_artifact = {
        "device": str(device),
        "model_dtype": "torch.bfloat16",
        "train_mode": bool(model.training),
        "use_cache": False,
        "optimizer": None,
        "memory_savers": memory_savers,
        "adapter": adapter_result.receipt.to_artifact_dict(),
        "special_token_embeddings": special_result.receipt.to_artifact_dict(),
        "trainable_value_identity_before": list(value_identity_before),
        "trainable_value_identity_after": list(value_identity_after),
        "accelerator": {
            **accelerator_identity,
            "prepared_model_attestation": prepared_model_attestation,
            "prepared": True,
            "prepare_route": "accelerator.prepare",
            "backward_route": "accelerator.backward",
            "output_conversion": "convert_outputs_to_fp32",
            "observed_forward_logits_dtypes": {
                "unmeasured_no_proof_warmup": [warmup_logits_dtype],
                "timed_proof_off": [proof_off_logits_dtype],
                "packed_primary": list(primary.forward_logits_dtypes),
                "packed_repeat": list(repeat.forward_logits_dtypes),
                "separate_reference": list(reference.forward_logits_dtypes),
                "packed_merged_boundary_negative": list(negative.forward_logits_dtypes),
            },
            "observed_inner_autocast": {
                "unmeasured_no_proof_warmup": [warmup_autocast],
                "timed_proof_off": [proof_off_autocast],
                "packed_primary": list(primary.autocast_observations),
                "packed_repeat": list(repeat.autocast_observations),
                "separate_reference": list(reference.autocast_observations),
                "packed_merged_boundary_negative": list(negative.autocast_observations),
            },
        },
    }
    comparison_artifact = {
        "semantic_atoms": atom_comparison,
        "denominators": denominator_comparison,
        "packed_primary_vs_separate": {
            "supervised_logits": primary_logit_comparison,
            "loss": primary_loss_comparison,
            "cross_arm_bf16_loss_term_scalars": primary_loss_term_scalars,
            "gradients": primary_gradient_comparison,
        },
        "packed_repeat_vs_separate": {
            "supervised_logits": repeat_logit_comparison,
            "loss": repeat_loss_comparison,
            "cross_arm_bf16_loss_term_scalars": repeat_loss_term_scalars,
            "gradients": repeat_gradient_comparison,
        },
        "packed_repeat_measurability": packed_repeat_comparison,
    }
    failure_evidence.record(
        "execution",
        _merge_completed_failure_execution(
            failure_evidence.fields.get("execution", {}),
            execution_artifact,
        ),
    )
    negative_artifact = {
        **discriminator,
        "attestation": negative_attestation,
        "supervised_logits": negative_logits,
        "total_loss": negative_loss_raw.to_artifact_dict(),
        "gradients": {"status": "not_executed", "acceptance_metric": False},
    }
    _record_comparisons_and_enforce_clean_gate(
        failure_evidence,
        comparisons=comparison_artifact,
        negative_discriminator_artifact=negative_artifact,
    )
    clean_gate = True
    _record_failure_progress(failure_evidence, phases, resource_samples)
    _finish_phase(phases, "comparison", comparison_start)
    failure_evidence.complete_phase("comparison")
    failure_evidence.reach("finalization")
    _record_failure_progress(failure_evidence, phases, resource_samples)
    _enforce_resource_ceilings(
        device,
        phase="comparison",
        samples=resource_samples,
        device_sampler=device_sampler,
    )
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    receipt = {
        "schema": PARITY_RECEIPT_SCHEMA,
        "terminal_status": "passed",
        "plan_sha256": plan["plan_sha256"],
        "source_identity": {
            "config_identity": plan["config_identity"],
            "repo_identity": plan["repo_identity"],
            "dependency_identity": plan["dependency_identity"],
            "model_identity": components.to_artifact_dict(),
            "model_weight_identity": post_load_weight_identity,
            "source_owners": plan["source_owners"],
        },
        "parent_v2": plan["parent_v2"],
        "attempt_marker": attempt_marker,
        "trainable_inventory": {
            "concrete": concrete_inventory,
        },
        "model_identity_attestation": model_identity_attestation,
        "execution": execution_artifact,
        "arms": {
            "packed_primary": _arm_artifact(primary),
            "packed_repeat": _arm_artifact(repeat),
            "separate_reference": _arm_artifact(reference),
            "packed_merged_boundary_negative": _arm_artifact(negative),
        },
        "proof": dict(proof),
        "comparisons": {
            **comparison_artifact,
        },
        "negative_discriminator": negative_artifact,
        "timings": {
            "clock": "time.perf_counter_ns_with_cuda_synchronize",
            "scope": "forward_only_same_packed_inputs_train_mode_no_backward",
            "proof_on_forward_ns": primary.forward_elapsed_ns,
            "proof_off_forward_ns": proof_off_ns,
            "proof_overhead_ns": primary.forward_elapsed_ns - proof_off_ns,
            "ordering": [
                "unmeasured_no_proof_warmup",
                "timed_proof_off",
                "timed_proof_on_packed_primary",
            ],
            "timed_sample_count_per_mode": 1,
            "warmup_sample_count": 1,
            "steady_step_inclusion": False,
        },
        "gpu_memory": {
            "scope": "whole_probe_including_model_transfer_and_all_forward_backward_arms",
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "measurement": _measurement_artifact(
            plan=plan,
            materials=materials,
            device=device,
            gpu_idle_preflight=gpu_idle_preflight,
            phases=phases,
            resource_samples=resource_samples,
            peak_allocated=peak_allocated,
            peak_reserved=peak_reserved,
            accelerator_identity=accelerator_identity,
            device_sampler=device_sampler.artifact(),
            clean_gate=clean_gate,
            discriminator=discriminator,
            proof=proof,
        ),
        "failure": None,
    }
    return receipt


def _execute_arm(
    *,
    name: str,
    model: Any,
    forward_inputs: Sequence[Any],
    token_sequences: Sequence[Any],
    loss_runner: LossRunner,
    loss_plan: Any,
    vocab_groups: Any,
    expected_vocab_size: int,
    capture_proof: bool,
    accelerator: Any,
    expected_inventory: Mapping[str, Any],
    perform_backward: bool = True,
) -> ExecutedArm:
    if len(forward_inputs) != len(token_sequences):
        raise ParityContractError(
            "parity arm forward/token sequence counts differ",
            code="qwen.parity.arm_shape",
            context={"name": name},
        )
    _zero_grad(model)
    total_loss_fp32: torch.Tensor | None = None
    keyed_logits: dict[Any, torch.Tensor] = {}
    artifacts = []
    receipts = []
    forward_logits_dtypes = []
    autocast_observations = []
    backward_events: list[dict[str, Any]] = []
    elapsed_ns = 0
    for index, (inputs, tokens) in enumerate(
        zip(forward_inputs, token_sequences, strict=True)
    ):
        sync_gradients = index == len(forward_inputs) - 1
        with _production_accumulation_context(
            accelerator,
            model=model,
            sync_gradients=sync_gradients,
            enabled=perform_backward,
        ) as context_kind:
            start = _timed_start(inputs.input_ids.device)
            with _capture_inner_cuda_autocast(model) as autocast_rows:
                forward = run_qwen_forward(
                    model,
                    inputs,
                    expected_vocab_size=expected_vocab_size,
                    capture_fa2_branch=capture_proof and index == 0,
                    require_fa2_branch_proof=capture_proof and index == 0,
                )
            autocast_observations.append(
                _validate_inner_autocast_observations(
                    autocast_rows,
                    path=f"{name}[{index}]",
                )
            )
            forward_logits_dtypes.append(
                _assert_forward_output_dtype(
                    forward,
                    path=f"{name}[{index}]",
                )
            )
            elapsed_ns += _timed_end(inputs.input_ids.device, start)
            context = LossContext(
                logits=forward.logits,
                token_sequence=tokens,
                vocab_groups=vocab_groups,
                logits_position_ids=forward.logits_position_ids,
            )
            selected = selected_logits_by_semantic_key(context)
            overlap = set(keyed_logits) & set(selected)
            if overlap:
                raise ParityContractError(
                    "parity arm produced duplicate semantic logits",
                    code="qwen.parity.duplicate_atom_key",
                    context={"name": name, "duplicate_count": len(overlap)},
                )
            keyed_logits.update(selected)
            bundle = loss_runner.compute_micro_step(
                context,
                loss_plan,
                local_micro_step_index=index,
            )
            detached_loss = bundle.total_loss.detach().float().cpu()
            total_loss_fp32 = (
                detached_loss
                if total_loss_fp32 is None
                else total_loss_fp32 + detached_loss
            )
            artifacts.append(bundle.to_artifact_dict())
            receipts.append(forward.receipt.to_artifact_dict())
            if perform_backward:
                accelerator.backward(bundle.total_loss)
        if perform_backward:
            backward_events.append(
                {
                    "microstep_index": index,
                    "forward_ordinal": index + 1,
                    "loss_ordinal": index + 1,
                    "backward_ordinal": len(backward_events) + 1,
                    "immediate_after_loss": True,
                    "sync_gradients": sync_gradients,
                    "accumulation_context": context_kind,
                }
            )
    if total_loss_fp32 is None:
        raise ParityContractError(
            "parity arm contains no microsteps",
            code="qwen.parity.arm_empty",
            context={"name": name},
        )
    gradients = (
        snapshot_trainable_gradients(
            model,
            expected_inventory=expected_inventory,
            gradient_provenance_dtype="torch.bfloat16",
        )
        if perform_backward
        else ()
    )
    loss_artifact = loss_runner.finalize_planned_step(artifacts, loss_plan)
    return ExecutedArm(
        name=name,
        total_loss=total_loss_fp32,
        loss_artifact=loss_artifact,
        keyed_logits=keyed_logits,
        gradients=gradients,
        forward_receipts=tuple(receipts),
        forward_logits_dtypes=tuple(forward_logits_dtypes),
        autocast_observations=tuple(autocast_observations),
        forward_elapsed_ns=elapsed_ns,
        backward_call_count=len(backward_events),
        backward_events=tuple(backward_events),
        gradient_clear_count=1,
    )


@contextmanager
def _production_accumulation_context(
    accelerator: Any,
    *,
    model: Any,
    sync_gradients: bool,
    enabled: bool = True,
):
    if not enabled:
        yield "no_backward"
        return
    if sync_gradients:
        yield "sync_gradients"
        return
    no_sync = getattr(accelerator, "no_sync", None)
    if not callable(no_sync):
        raise ParityContractError(
            "streaming reference requires Accelerator.no_sync for an intermediate microstep",
            code="qwen.parity.streaming_no_sync",
            context={},
        )
    with no_sync(model):
        yield "accelerator.no_sync"


def _assert_forward_output_dtype(forward: Any, *, path: str) -> str:
    logits = getattr(forward, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise ParityContractError(
            "Accelerator-prepared forward did not return tensor logits",
            code="qwen.parity.accelerator_output",
            context={"path": path, "value_type": type(logits).__name__},
        )
    observed = str(logits.dtype)
    if observed != "torch.float32":
        raise ParityContractError(
            "Accelerator BF16 forward must convert outputs to FP32",
            code="qwen.parity.accelerator_output_dtype",
            context={"path": path, "observed": observed},
        )
    return observed


def _forward_inputs(
    pack: Any,
    examples: Sequence[Any],
    positions: Any,
    tokens: Any,
    *,
    device: torch.device,
    proof_policy: str,
    image_processor: Any,
) -> Any:
    attached_examples = _attach_runtime_image_processor(
        examples,
        image_processor=image_processor,
    )
    logits_positions = tuple(
        sorted({int(atom.causal_logits_position) for atom in tokens.atoms})
    )
    return build_qwen_forward_inputs(
        pack,
        attached_examples,
        positions,
        logits_to_keep_positions=logits_positions,
        device=device,
        fa2_branch_proof_policy=proof_policy,
    )


def _attach_runtime_image_processor(
    examples: Sequence[Any],
    *,
    image_processor: Any,
) -> tuple[Any, ...]:
    """Reattach the loaded processor without changing frozen example semantics."""

    if image_processor is None:
        raise ParityContractError(
            "real parity inputs require runtime processor.image_processor",
            code="qwen.parity.image_processor_missing",
            context={},
        )
    attached_examples = []
    for encoded in examples:
        image_encoding = getattr(encoded, "image_encoding", None)
        if not isinstance(image_encoding, QwenImageEncoding):
            raise ParityContractError(
                "real parity inputs require QwenImageEncoding plans",
                code="qwen.parity.image_encoding",
                context={"encoded_type": type(encoded).__name__},
            )
        artifact_fn = getattr(encoded, "to_artifact_dict", None)
        if not callable(artifact_fn):
            raise ParityContractError(
                "real parity encoded example lacks semantic artifact identity",
                code="qwen.parity.encoded_artifact",
                context={"encoded_type": type(encoded).__name__},
            )
        frozen_before = artifact_fn()
        attached = replace(
            encoded,
            image_encoding=attach_qwen_image_processor(
                image_encoding,
                image_processor,
            ),
        )
        if attached.to_artifact_dict() != frozen_before:
            raise ParityContractError(
                "runtime image-processor attachment changed frozen example semantics",
                code="qwen.parity.image_attachment_drift",
                context={"example_id": getattr(encoded, "example_id", None)},
            )
        attached_examples.append(attached)
    return tuple(attached_examples)


def _compare_loss_terms(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    left_terms = {str(item["name"]): item for item in left["terms"]}
    right_terms = {str(item["name"]): item for item in right["terms"]}
    missing = sorted(set(left_terms) - set(right_terms))
    extra = sorted(set(right_terms) - set(left_terms))
    rows: list[dict[str, Any]] = []
    passed = not missing and not extra
    fields = [
        "raw_loss",
        "weighted_loss",
        "segment_mean_numerator",
        "token_weighted_diagnostic",
    ]
    for name in sorted(set(left_terms) & set(right_terms)):
        field_rows: list[dict[str, Any]] = []
        for field in fields:
            left_value = torch.tensor(
                float(left_terms[name][field]), dtype=torch.float32
            )
            right_value = torch.tensor(
                float(right_terms[name][field]), dtype=torch.float32
            )
            comparison = compare_tensors(
                left_value,
                right_value,
                rtol=BF16_RTOL,
                atol=BF16_ATOL,
            )
            passed = passed and comparison.allclose
            field_rows.append(
                {
                    "field": field,
                    "packed_value": float(left_value.item()),
                    "separate_shared_value": float(right_value.item()),
                    "raw_delta": float((left_value - right_value).item()),
                    **comparison.to_artifact_dict(),
                }
            )
        rows.append({"term_name": name, "fields": field_rows})
    return {
        "passed": passed,
        "source_forward_dtype": "torch.bfloat16",
        "comparison_dtype": "torch.float32",
        "rtol": BF16_RTOL,
        "atol": BF16_ATOL,
        "fields": fields,
        "missing": missing,
        "extra": extra,
        "terms": rows,
    }


def _arm_artifact(arm: ExecutedArm) -> dict[str, Any]:
    forward_only = arm.backward_call_count == 0
    return {
        "name": arm.name,
        "microstep_count": len(arm.forward_receipts),
        "total_loss_fp32": float(arm.total_loss.item()),
        "loss_artifact": arm.loss_artifact,
        "semantic_logit_rows": len(arm.keyed_logits),
        "semantic_key_inventory_sha256": semantic_atom_key_inventory_sha256(
            tuple(sorted(arm.keyed_logits))
        ),
        "gradient_inventory": [record.to_inventory_dict() for record in arm.gradients],
        "forward_receipts": list(arm.forward_receipts),
        "forward_logits_dtypes": list(arm.forward_logits_dtypes),
        "inner_autocast": list(arm.autocast_observations),
        "backward_cadence": {
            "microstep_count": len(arm.forward_receipts),
            "gradient_clear_count": arm.gradient_clear_count,
            "harness_backward_call_count": arm.backward_call_count,
            "harness_policy": (
                "forward_only_negative_control"
                if forward_only
                else "one_immediate_accelerator_backward_per_microstep"
            ),
            "events": list(arm.backward_events),
            "production_backward_call_count": arm.backward_call_count,
            "production_policy": (
                "not_applicable_forward_only_negative"
                if forward_only
                else "one_runtime_backward_per_microstep_with_accumulation_context"
            ),
            "cadence_matches_production": (
                forward_only
                or (
                    arm.backward_call_count == len(arm.forward_receipts)
                    and all(
                        event["immediate_after_loss"] is True
                        for event in arm.backward_events
                    )
                )
            ),
        },
    }


def _bounded_arm_artifact(arm: ExecutedArm) -> dict[str, Any]:
    artifact = _arm_artifact(arm)
    inventory = artifact["gradient_inventory"]
    artifact["gradient_inventory"] = inventory[:64]
    artifact["gradient_inventory_count"] = len(inventory)
    artifact["gradient_inventory_omitted"] = max(0, len(inventory) - 64)
    return artifact


def _record_failure_progress(
    evidence: _FailureEvidenceAccumulator,
    phases: Sequence[Mapping[str, Any]],
    resource_samples: Sequence[Mapping[str, Any]],
) -> None:
    evidence.record(
        "measurement",
        {
            "schema": "coordexp-swift-wave2-failure-measurement-v1",
            "completed_phases": [dict(phase) for phase in phases],
            "phase_boundary_samples": [dict(sample) for sample in resource_samples],
        },
    )


def _failure_receipt(
    plan: Mapping[str, Any] | None,
    *,
    device_text: str,
    exc: BaseException,
    evidence: _FailureEvidenceAccumulator | None = None,
) -> dict[str, Any]:
    plan_sha = "0" * 64 if plan is None else str(plan.get("plan_sha256", "0" * 64))
    source_identity = {} if plan is None else _planned_source_identity(plan)
    evidence_fields = {} if evidence is None else evidence.fields
    marker_reference = evidence_fields.get("attempt_marker", {})
    failure = bounded_failure(exc)
    if evidence is not None:
        failure["evidence"] = evidence.evidence()
    terminal_status = (
        "unmeasurable"
        if failure.get("code") == "qwen.parity.packed_repeat_unmeasurable"
        else "failed"
    )
    return {
        "schema": PARITY_RECEIPT_SCHEMA,
        "terminal_status": terminal_status,
        "plan_sha256": plan_sha,
        "parent_v2": (
            frozen_parent_v2_identity()
            if plan is None
            else plan.get("parent_v2", frozen_parent_v2_identity())
        ),
        "attempt_marker": marker_reference,
        "trainable_inventory": (
            {"concrete": evidence_fields["trainable_inventory"]["concrete"]}
            if marker_reference and "trainable_inventory" in evidence_fields
            else {}
        ),
        "source_identity": evidence_fields.get("source_identity", source_identity),
        "model_identity_attestation": evidence_fields.get(
            "model_identity_attestation", {}
        ),
        "execution": evidence_fields.get(
            "execution", {"requested_device": device_text}
        ),
        "arms": evidence_fields.get("arms", {}),
        "proof": evidence_fields.get("proof", {}),
        "comparisons": evidence_fields.get("comparisons", {}),
        "negative_discriminator": evidence_fields.get("negative_discriminator", {}),
        "timings": evidence_fields.get("timings", {}),
        "gpu_memory": evidence_fields.get("gpu_memory", {}),
        "measurement": evidence_fields.get("measurement", {}),
        "failure": failure,
    }


def _dependency_identity() -> dict[str, Any]:
    collected = collect_dependency_provenance()
    accelerate_origin = collected.get("accelerate", {}).get("imported_origin", {})
    if (
        not isinstance(accelerate_origin, Mapping)
        or accelerate_origin.get("status") != "available"
        or not isinstance(accelerate_origin.get("value"), str)
    ):
        raise ParityContractError(
            "Accelerate imported source root is unavailable",
            code="qwen.parity.accelerate_source_unavailable",
            context={},
        )
    package_root = Path(accelerate_origin["value"]).expanduser().resolve().parent
    runtime_paths = (
        "accelerator.py",
        "state.py",
        "utils/modeling.py",
        "utils/operations.py",
    )
    runtime_sources = []
    for relative in runtime_paths:
        source = (package_root / relative).resolve()
        try:
            source.relative_to(package_root)
        except ValueError as exc:
            raise ParityContractError(
                "Accelerate runtime source escaped its package root",
                code="qwen.parity.accelerate_source_path",
                context={"relative_path": relative},
                cause=exc,
            ) from exc
        runtime_sources.append(
            {
                "relative_path": relative,
                "resolved_path": str(source),
                "sha256": sha256_file(source),
            }
        )
    return validate_dependency_provenance(
        {
            "schema": "coordexp-swift-wave2-dependency-identity-v1",
            "collector": "src.artifacts.provenance.collect_dependency_provenance",
            "collected": collected,
            "accelerate_runtime_sources": runtime_sources,
        }
    )


def _image_token_id(components: Any) -> int | None:
    convert = getattr(components.tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    value = convert("<|image_pad|>")
    return None if value is None else int(value)


def _int_sequence_sha256(values: Sequence[int]) -> str:
    return sha256_json([int(value) for value in values])


def _preflight_cuda(device_text: str) -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise ParityContractError(
            "real parity run requires CUDA",
            code="qwen.parity.cuda_unavailable",
            context={},
        )
    device = torch.device(device_text)
    if device.type != "cuda" or device.index is None:
        raise ParityContractError(
            "real parity run requires one explicit CUDA device index",
            code="qwen.parity.cuda_device",
            context={"device": device_text},
        )
    if device.index < 0 or device.index >= torch.cuda.device_count():
        raise ParityContractError(
            "requested CUDA device index is unavailable",
            code="qwen.parity.cuda_device",
            context={"device": device_text, "device_count": torch.cuda.device_count()},
        )
    samples = _collect_idle_samples(
        device,
        check_name="initial_before_cpu_model_work",
    )
    first = samples[0]
    return device, {
        "status": "passed",
        "requested_device": str(device),
        "physical_index": first["physical_index"],
        "uuid": first["uuid"],
        "memory_limit_bytes": GPU_IDLE_MEMORY_LIMIT_BYTES,
        "utilization_limit_percent": GPU_IDLE_UTILIZATION_LIMIT_PERCENT,
        "sampler": {
            "command": "nvidia-smi query-gpu index,uuid,memory.used,utilization.gpu",
            "sample_count": 3,
            "interval_seconds": 0.2,
            "cuda_visible_device_mapping": _cuda_device_selector(device),
        },
        "checks": [
            {
                "name": "initial_before_cpu_model_work",
                "samples": samples,
            }
        ],
    }


def _append_final_gpu_idle_check(
    device: torch.device,
    *,
    gpu_idle_preflight: dict[str, Any],
) -> None:
    checks = gpu_idle_preflight.get("checks")
    if not isinstance(checks, list) or len(checks) != 1:
        raise ParityContractError(
            "GPU idle preflight cannot append the final check",
            code="qwen.parity.gpu_preflight_state",
            context={
                "check_count": None if not isinstance(checks, list) else len(checks)
            },
        )
    checks.append(
        {
            "name": "final_before_gpu_work",
            "samples": _collect_idle_samples(
                device,
                check_name="final_before_gpu_work",
            ),
        }
    )


def _collect_idle_samples(
    device: torch.device,
    *,
    check_name: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sample_index in range(3):
        samples.append(_nvidia_smi_sample(device, sample_index=sample_index))
        if sample_index < 2:
            time.sleep(0.2)
    failed = [
        sample
        for sample in samples
        if sample["memory_used_bytes"] >= GPU_IDLE_MEMORY_LIMIT_BYTES
        or sample["utilization_percent"] >= GPU_IDLE_UTILIZATION_LIMIT_PERCENT
    ]
    if failed:
        raise ParityContractError(
            "selected GPU is not idle enough for the Wave 2 probe",
            code="qwen.parity.gpu_not_idle",
            context={
                "check_name": check_name,
                "device": str(device),
                "memory_limit_bytes": GPU_IDLE_MEMORY_LIMIT_BYTES,
                "utilization_limit_percent": GPU_IDLE_UTILIZATION_LIMIT_PERCENT,
                "failed_samples": failed,
            },
        )
    return samples


def _nvidia_smi_sample(
    device: torch.device,
    *,
    sample_index: int,
) -> dict[str, Any]:
    selector = _cuda_device_selector(device)
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                selector,
            ),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ParityContractError(
            "nvidia-smi sampling failed",
            code="qwen.parity.gpu_sampler",
            context={"device": str(device), "error": type(exc).__name__},
            cause=exc,
        ) from exc
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ParityContractError(
            "nvidia-smi sampler did not resolve one physical GPU",
            code="qwen.parity.gpu_sampler",
            context={"device": str(device), "row_count": len(rows)},
        )
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise ParityContractError(
            "nvidia-smi sampler returned an unexpected row",
            code="qwen.parity.gpu_sampler",
            context={"device": str(device), "field_count": len(fields)},
        )
    try:
        physical_index = int(fields[0])
        memory_used_bytes = int(fields[2]) * 1024**2
        utilization_percent = int(fields[3])
    except ValueError as exc:
        raise ParityContractError(
            "nvidia-smi sampler returned non-integer resource fields",
            code="qwen.parity.gpu_sampler",
            context={"device": str(device)},
            cause=exc,
        ) from exc
    return {
        "sample_index": int(sample_index),
        "monotonic_ns": time.perf_counter_ns(),
        "physical_index": physical_index,
        "uuid": fields[1],
        "memory_used_bytes": memory_used_bytes,
        "utilization_percent": utilization_percent,
    }


def _cuda_device_selector(device: torch.device) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return str(device.index)
    entries = [entry.strip() for entry in visible.split(",")]
    if (
        device.index is None
        or device.index >= len(entries)
        or not entries[device.index]
    ):
        raise ParityContractError(
            "CUDA_VISIBLE_DEVICES does not map the requested logical device",
            code="qwen.parity.cuda_visible_mapping",
            context={"device": str(device), "visible_entry_count": len(entries)},
        )
    return entries[device.index]


def _build_exact_one_rank_accelerator(
    *,
    device: torch.device,
    training_precision: str,
) -> Any:
    """Construct the production Accelerator seam in a fresh explicit-device process."""

    try:
        from accelerate.state import AcceleratorState, PartialState
    except ImportError as exc:
        raise ParityContractError(
            "Accelerate state APIs are unavailable",
            code="qwen.parity.accelerator_state",
            context={},
            cause=exc,
        ) from exc
    shared_state_present = bool(getattr(PartialState, "_shared_state", {})) or bool(
        getattr(AcceleratorState, "_shared_state", {})
    )
    distributed_initialized = bool(
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    launcher_keys = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
    )
    launcher_contamination = sorted(key for key in launcher_keys if key in os.environ)
    if shared_state_present or distributed_initialized or launcher_contamination:
        raise ParityContractError(
            "Wave 2 Accelerator must be constructed in a fresh one-rank process",
            code="qwen.parity.accelerator_not_fresh",
            context={
                "accelerate_shared_state_present": shared_state_present,
                "torch_distributed_initialized": distributed_initialized,
                "launcher_environment_keys": launcher_contamination,
            },
        )
    required_device = str(device)
    configured_device = os.environ.get("ACCELERATE_TORCH_DEVICE")
    if configured_device is not None and configured_device != required_device:
        raise ParityContractError(
            "ACCELERATE_TORCH_DEVICE conflicts with the requested probe device",
            code="qwen.parity.accelerator_device_override",
            context={"expected": required_device, "observed": configured_device},
        )
    os.environ["ACCELERATE_TORCH_DEVICE"] = required_device
    torch.cuda.set_device(device)
    return _build_accelerator(training_precision)


def _validate_exact_accelerator_runtime(
    accelerator: Any,
    *,
    expected_device: torch.device,
    expected_mixed_precision: str,
) -> dict[str, Any]:
    validate_accelerator_runtime(
        accelerator,
        expected_mixed_precision=expected_mixed_precision,
    )
    distributed_type = getattr(accelerator, "distributed_type", None)
    distributed_name = getattr(distributed_type, "name", str(distributed_type))
    observed_device = torch.device(accelerator.device)
    local_rank = int(getattr(accelerator, "local_process_index", -1))
    scaler = getattr(accelerator, "scaler", None)
    identity = {
        "distributed_type": distributed_name,
        "rank": int(accelerator.process_index),
        "local_rank": local_rank,
        "world_size": int(accelerator.num_processes),
        "device": str(observed_device),
        "cuda_current_device": int(torch.cuda.current_device()),
        "mixed_precision": str(accelerator.mixed_precision),
        "native_amp": bool(getattr(accelerator, "native_amp", False)),
        "gradient_accumulation_steps": int(
            getattr(accelerator, "gradient_accumulation_steps", -1)
        ),
        "scaler": None if scaler is None else type(scaler).__name__,
        "accelerate_torch_device": os.environ.get("ACCELERATE_TORCH_DEVICE"),
    }
    expected_index = expected_device.index
    if (
        distributed_name != "NO"
        or identity["rank"] != 0
        or local_rank != 0
        or identity["world_size"] != 1
        or observed_device != expected_device
        or identity["cuda_current_device"] != expected_index
        or identity["mixed_precision"] != "bf16"
        or identity["native_amp"] is not True
        or identity["gradient_accumulation_steps"] != 1
        or scaler is not None
        or identity["accelerate_torch_device"] != str(expected_device)
    ):
        raise ParityContractError(
            "Wave 2 Accelerator runtime differs from the frozen one-rank BF16 seam",
            code="qwen.parity.accelerator_identity",
            context=identity,
        )
    return identity


def _assert_prepared_accelerator_model(
    accelerator: Any,
    *,
    model: Any,
    expected_device: torch.device,
) -> dict[str, Any]:
    from accelerate.utils.operations import ConvertOutputsToFp32

    registered_models = tuple(getattr(accelerator, "_models", ()))
    if not any(candidate is model for candidate in registered_models):
        raise ParityContractError(
            "Accelerator did not register the exact prepared model",
            code="qwen.parity.accelerator_prepare",
            context={"registered_model_count": len(registered_models)},
        )
    wrong_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.device != expected_device
    ]
    wrong_buffers = [
        name
        for name, buffer in model.named_buffers()
        if buffer.device != expected_device
    ]
    if wrong_parameters or wrong_buffers:
        raise ParityContractError(
            "Accelerator-prepared model contains tensors on another device",
            code="qwen.parity.accelerator_model_device",
            context={
                "parameter_count": len(wrong_parameters),
                "buffer_count": len(wrong_buffers),
                "parameter_examples": wrong_parameters[:8],
                "buffer_examples": wrong_buffers[:8],
            },
        )
    if not callable(getattr(model, "_original_forward", None)):
        raise ParityContractError(
            "Accelerator-prepared model lacks the original forward binding",
            code="qwen.parity.accelerator_forward_wrapper",
            context={},
        )
    prepared_forward = getattr(model, "forward", None)
    if isinstance(prepared_forward, MethodType):
        wrapper_owner = prepared_forward.__func__
        binding_branch = "bound_method"
    else:
        wrapper_owner = prepared_forward
        binding_branch = "direct_callable"
    output_wrapper = getattr(wrapper_owner, "__wrapped__", None)
    wrapped_forward = getattr(output_wrapper, "model_forward", None)
    original_forward = getattr(model, "_original_forward", None)
    original_identity = (
        original_forward.__func__
        if isinstance(original_forward, MethodType)
        else original_forward
    )
    try:
        unwrapped_identity = inspect.unwrap(wrapped_forward)
    except (ValueError, TypeError) as exc:
        raise ParityContractError(
            "Accelerator forward wrapper chain cannot be unwrapped",
            code="qwen.parity.accelerator_forward_wrapper",
            context={"binding_branch": binding_branch},
            cause=exc,
        ) from exc
    if (
        type(output_wrapper) is not ConvertOutputsToFp32
        or getattr(wrapper_owner, "__wrapped__", None) is not output_wrapper
        or getattr(output_wrapper, "__wrapped__", None) is not wrapped_forward
        or not callable(wrapped_forward)
        or unwrapped_identity is not original_identity
    ):
        raise ParityContractError(
            "Accelerator-prepared model lacks ConvertOutputsToFp32",
            code="qwen.parity.accelerator_forward_wrapper",
            context={
                "binding_branch": binding_branch,
                "prepared_forward_type": type(prepared_forward).__name__,
                "wrapper_owner_type": type(wrapper_owner).__name__,
                "output_wrapper_type": type(output_wrapper).__name__,
                "wrapped_forward_callable": callable(wrapped_forward),
                "unwrapped_matches_original": unwrapped_identity is original_identity,
            },
        )
    return {
        "binding_branch": binding_branch,
        "prepared_forward_type": type(prepared_forward).__name__,
        "wrapper_owner_type": type(wrapper_owner).__name__,
        "output_wrapper_type": type(output_wrapper).__name__,
        "wrapper_identity_chain_verified": True,
        "unwrapped_original_identity_verified": True,
    }


@contextmanager
def _capture_inner_cuda_autocast(model: Any):
    attention_modules = [
        module
        for module in model.modules()
        if type(module).__name__ == "Qwen3VLTextAttention"
    ]
    if not attention_modules:
        raise ParityContractError(
            "prepared model exposes no Qwen3VL text-attention module",
            code="qwen.parity.autocast_probe_topology",
            context={},
        )
    attention_modules.sort(key=lambda module: int(getattr(module, "layer_idx", -1)))
    target = attention_modules[0]
    rows: list[dict[str, Any]] = []

    def record_autocast(module: Any, _args: Any) -> None:
        rows.append(
            {
                "module_type": type(module).__name__,
                "layer_idx": int(getattr(module, "layer_idx", -1)),
                "cuda_autocast_enabled": bool(torch.is_autocast_enabled("cuda")),
                "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
            }
        )

    handle = target.register_forward_pre_hook(record_autocast)
    try:
        yield rows
    finally:
        handle.remove()


def _validate_inner_autocast_observations(
    rows: Sequence[Mapping[str, Any]],
    *,
    path: str,
) -> dict[str, Any]:
    if len(rows) != 1:
        raise ParityContractError(
            "inner attention autocast probe did not observe exactly one forward",
            code="qwen.parity.autocast_probe_count",
            context={"path": path, "count": len(rows)},
        )
    row = dict(rows[0])
    if (
        row.get("module_type") != "Qwen3VLTextAttention"
        or row.get("layer_idx") != 0
        or row.get("cuda_autocast_enabled") is not True
        or row.get("cuda_autocast_dtype") != "torch.bfloat16"
    ):
        raise ParityContractError(
            "inner Qwen attention did not execute under CUDA BF16 autocast",
            code="qwen.parity.autocast_probe",
            context={"path": path, "observation": row},
        )
    return row


def _assert_model_runtime(model: Any) -> None:
    if not bool(model.training):
        raise ParityContractError(
            "parity model must remain in train mode",
            code="qwen.parity.train_mode",
            context={},
        )
    configurations = [getattr(model, "config", None)]
    base = getattr(model, "base_model", None)
    configurations.append(getattr(base, "config", None))
    stale = [
        config
        for config in configurations
        if config is not None and getattr(config, "use_cache", False)
    ]
    if stale:
        raise ParityContractError(
            "parity model must have use_cache disabled",
            code="qwen.parity.use_cache",
            context={"stale_config_count": len(stale)},
        )


def _finish_phase(
    phases: list[dict[str, Any]],
    name: str,
    start_ns: int,
) -> None:
    end_ns = time.perf_counter_ns()
    phases.append(
        {
            "name": name,
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
            "duration_ns": max(1, int(end_ns - start_ns)),
            "status": "completed",
        }
    )


def _host_resource_snapshot() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB. CoordExp production nodes are Linux.
    rss_hwm_bytes = int(usage.ru_maxrss) * 1024
    io_values = {"read_bytes": 0, "write_bytes": 0}
    try:
        for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition(":")
            if separator and name in io_values:
                io_values[name] = int(value.strip())
    except (OSError, UnicodeError, ValueError) as exc:
        raise ParityContractError(
            "host I/O counters are unavailable",
            code="qwen.parity.host_resource_sampler",
            context={"error": type(exc).__name__},
            cause=exc,
        ) from exc
    return {
        "rss_hwm_bytes": rss_hwm_bytes,
        "io_read_bytes": io_values["read_bytes"],
        "io_write_bytes": io_values["write_bytes"],
    }


def _enforce_resource_ceilings(
    device: torch.device,
    *,
    phase: str,
    samples: list[dict[str, Any]],
    device_sampler: _BoundedDeviceSampler,
) -> None:
    _sync(device)
    device_sampler.raise_if_failed()
    host = _host_resource_snapshot()
    gpu = _nvidia_smi_sample(device, sample_index=len(samples))
    torch_cuda = {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    sampler_snapshot = device_sampler.artifact()
    sample = {
        "phase": phase,
        "monotonic_ns": time.perf_counter_ns(),
        "host": host,
        "gpu": gpu,
        "torch_cuda": torch_cuda,
    }
    samples.append(sample)
    if host["rss_hwm_bytes"] >= HOST_MEMORY_CEILING_BYTES:
        raise ParityContractError(
            "Wave 2 host-memory ceiling exceeded",
            code="qwen.parity.host_memory_ceiling",
            context={
                "phase": phase,
                "observed_bytes": host["rss_hwm_bytes"],
                "ceiling_bytes": HOST_MEMORY_CEILING_BYTES,
            },
        )
    if (
        gpu["memory_used_bytes"] >= DEVICE_MEMORY_CEILING_BYTES
        or torch_cuda["max_reserved_bytes"] >= DEVICE_MEMORY_CEILING_BYTES
        or sampler_snapshot["hwm_memory_used_bytes"] >= DEVICE_MEMORY_CEILING_BYTES
    ):
        raise ParityContractError(
            "Wave 2 device-memory ceiling exceeded",
            code="qwen.parity.device_memory_ceiling",
            context={
                "phase": phase,
                "observed_device_bytes": gpu["memory_used_bytes"],
                "observed_process_peak_reserved_bytes": torch_cuda[
                    "max_reserved_bytes"
                ],
                "observed_device_sampler_hwm_bytes": sampler_snapshot[
                    "hwm_memory_used_bytes"
                ],
                "ceiling_bytes": DEVICE_MEMORY_CEILING_BYTES,
            },
        )


def _measurement_artifact(
    *,
    plan: Mapping[str, Any],
    materials: ParityMaterials,
    device: torch.device,
    gpu_idle_preflight: Mapping[str, Any],
    phases: Sequence[Mapping[str, Any]],
    resource_samples: Sequence[Mapping[str, Any]],
    peak_allocated: int,
    peak_reserved: int,
    accelerator_identity: Mapping[str, Any],
    device_sampler: Mapping[str, Any],
    clean_gate: bool,
    discriminator: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    if not resource_samples:
        raise ParityContractError(
            "Wave 2 measurement requires resource samples",
            code="qwen.parity.measurement_resources",
            context={},
        )
    host_samples = [sample["host"] for sample in resource_samples]
    pack_length = int(materials.packed.length)
    max_length = int(materials.packed.global_max_length)
    not_applicable_reasons = {
        "cache": "probe constructs two examples directly and never reads or writes packing cache",
        "eval": "forward-backward parity probe has no evaluation phase",
        "checkpoint": "probe performs no optimizer step or checkpoint publication",
        "warmup": "training scheduler warmup is not applicable; attention warmup is recorded as a phase",
        "per_step": "probe is not a steady-state optimizer-step timing run",
    }
    return {
        "schema": "coordexp-swift-wave2-measurement-v1",
        "runtime_launch_identity": {
            "rank": int(accelerator_identity["rank"]),
            "local_rank": int(accelerator_identity["local_rank"]),
            "world_size": int(accelerator_identity["world_size"]),
            "process_count": 1,
            "mode": "single_process_explicit_device",
            "device": str(accelerator_identity["device"]),
            "mixed_precision": str(accelerator_identity["mixed_precision"]),
        },
        "gpu_idle_preflight": dict(gpu_idle_preflight),
        "policies": {
            "seed": int(materials.resolved.config.runtime.seed),
            "config_fingerprint": str(plan["config_identity"]["fingerprint"]),
            "packing_policy": "source_order_next_fit_two_selected_examples",
            "provider_policy": "synchronous_cpu_build_then_explicit_device_transfer",
            "attention_backend": str(
                materials.resolved.config.model.attn_implementation
            ),
            "attention_proof_policy": "bounded_first_packed_forward",
        },
        "phases": [dict(phase) for phase in phases],
        "resources": {
            "host": {
                "rss_hwm_bytes": max(
                    int(item["rss_hwm_bytes"]) for item in host_samples
                ),
                "io_read_bytes_hwm": max(
                    int(item["io_read_bytes"]) for item in host_samples
                ),
                "io_write_bytes_hwm": max(
                    int(item["io_write_bytes"]) for item in host_samples
                ),
                "source": "getrusage_ru_maxrss_and_proc_self_io",
            },
            "gpu": {
                "device": str(device),
                "torch_peak_allocated_bytes": int(peak_allocated),
                "torch_peak_reserved_bytes": int(peak_reserved),
                "device_used_hwm_bytes": int(device_sampler["hwm_memory_used_bytes"]),
                "source": "torch_cuda_process_peak_and_bounded_nvidia_smi_sampler",
                "device_sampler": dict(device_sampler),
            },
            "ceilings": {
                "status": "passed",
                "host_bytes": HOST_MEMORY_CEILING_BYTES,
                "device_bytes": DEVICE_MEMORY_CEILING_BYTES,
                "comparison": {
                    "host_rss_below": max(
                        int(item["rss_hwm_bytes"]) for item in host_samples
                    )
                    < HOST_MEMORY_CEILING_BYTES,
                    "torch_reserved_below": int(peak_reserved)
                    < DEVICE_MEMORY_CEILING_BYTES,
                    "device_sampler_below": int(device_sampler["hwm_memory_used_bytes"])
                    < DEVICE_MEMORY_CEILING_BYTES,
                },
            },
            "phase_boundary_samples": [dict(sample) for sample in resource_samples],
        },
        "pack_utilization": {
            "pack_length": pack_length,
            "global_max_length": max_length,
            "unused_tokens": max_length - pack_length,
            "utilization_ratio": pack_length / max_length,
            "segment_count": len(materials.packed.segments),
        },
        "semantic_result": {
            "clean_parity_passed": bool(clean_gate),
            "negative_forward_signal_detected": bool(discriminator["detected"]),
            "all_layer_proof_passed": proof.get("status") == "pass",
        },
        "eligibility": {
            "terminal_eligible": True,
            "steady_state_timing_eligible": False,
            "steady_state_reason": "single decision-grade proof probe excludes warmup and proof timings from steady state",
        },
        "not_applicable": {
            name: {"status": "not_applicable", "reason": reason}
            for name, reason in not_applicable_reasons.items()
        },
    }


def _zero_grad(model: Any) -> None:
    model.zero_grad(set_to_none=True)


def _timed_start(device: torch.device | str) -> int:
    _sync(device)
    return time.perf_counter_ns()


def _timed_end(device: torch.device | str, start_ns: int) -> int:
    _sync(device)
    return max(1, time.perf_counter_ns() - start_ns)


def _sync(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or execute the immutable Wave 2 packed Qwen parity probe."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="CPU/model-free immutable plan preparation"
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--plan", required=True)
    prepare.add_argument("--parent-v2-plan", required=True)
    prepare.add_argument("--first-index", type=int, default=0)
    prepare.add_argument("--second-index", type=int, default=1)
    prepare.set_defaults(handler=prepare_command)
    run = subparsers.add_parser(
        "run", help="Revalidate plan and execute one real BF16 GPU probe"
    )
    run.add_argument("--plan", required=True)
    run.add_argument("--parent-v2-plan", required=True)
    run.add_argument("--receipt", required=True)
    run.add_argument("--attempt-marker", required=True)
    run.add_argument(
        "--device", required=True, help="Explicit device, for example cuda:0"
    )
    run.set_defaults(handler=run_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

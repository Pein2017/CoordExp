"""Decision-grade packed-versus-separate Qwen parity helpers.

This module owns the immutable plan and numerical comparison contracts used by
the Wave 2 GPU probe.  It deliberately has no import-time CUDA or model-loading
side effects so plan preparation and all unit tests remain CPU-only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
import subprocess
from typing import Any

import torch

from src.common.errors import RuntimeContractError


PARITY_PLAN_SCHEMA = "coordexp-swift-wave2-packed-parity-plan-v3"
PARITY_RECEIPT_SCHEMA = "coordexp-swift-wave2-packed-parity-receipt-v3"
PARITY_ATTEMPT_MARKER_SCHEMA = (
    "coordexp-swift-wave2-packed-parity-attempt-start-marker-v1"
)
LEGACY_PARITY_PLAN_SCHEMA_V2 = "coordexp-swift-wave2-packed-parity-plan-v2"
LEGACY_PARITY_RECEIPT_SCHEMA_V2 = "coordexp-swift-wave2-packed-parity-receipt-v2"
PARITY_FAILURE_EVIDENCE_SCHEMA = (
    "coordexp-swift-wave2-packed-parity-failure-evidence-v2"
)
LEGACY_PARITY_FAILURE_EVIDENCE_SCHEMA_V1 = (
    "coordexp-swift-wave2-packed-parity-failure-evidence-v1"
)
PLAN_STATUS = "prepared"
TERMINAL_STATUSES = frozenset({"passed", "failed", "unmeasurable"})
BF16_RTOL = 5.0e-3
BF16_ATOL = 5.0e-3
FP32_RTOL = 1.0e-4
FP32_ATOL = 1.0e-5
LOSS_RTOL = 1.0e-5
LOSS_ATOL = 1.0e-6
PACKED_REPEAT_MAX_ABS = 2.5e-3
EXPECTED_TRAINABLE_PARAMETER_COUNT = 589
EXPECTED_TRAINABLE_STRUCTURE = {
    "lora_A": 196,
    "lora_B": 196,
    "dora_magnitude": 196,
    "special_token_delta": 1,
}
TRAINABLE_PARAMETER_SUFFIXES = {
    "lora_A": ".lora_A.default.weight",
    "lora_B": ".lora_B.default.weight",
    "dora_magnitude": ".lora_magnitude_vector.default.weight",
    "special_token_delta": ".shared_embed_delta",
}
BF16_COMPUTE_PROVENANCE_DTYPE = "torch.bfloat16"
FP32_COMPARISON_DTYPE = "torch.float32"
FROZEN_PARENT_V2_PLAN_SHA256 = (
    "e5c2b1eb0c7ff99b7a66de8dc172f5af0331959d9b5f12a07e61096dd79b4197"
)
FROZEN_PARENT_V2_PLAN_FILE_SHA256 = (
    "aadcfe046938315d90df05633b1da00d5a86e368eee3304b9d822183c1cdb681"
)
FROZEN_PARENT_V2_CONFIG_FINGERPRINT = (
    "de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d"
)
HISTORICAL_V3_RUNTIME_CONFIG_FINGERPRINT = (
    "0f8fda29362a46e67cecccdd5fee7d7539fafc7d91b52f49b4d4b036556224a1"
)
FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT = (
    "8f2076bfe2f7f7fcac0c9ea2c210f91595f6bdebf08d03dbb54ef080f529286c"
)
_IMMUTABLE_WAVE2_V3_PLAN_SHA256 = (
    "03834e309357dace9d7b51a6c51186b14b09bea8953b3c3d12e82609cff1a49d"
)
_IMMUTABLE_WAVE2_V3_PLAN_PAYLOAD_SHA256 = (
    "6f27e22b3e8a2c8aff626e9062afc0d664cee04e77341d0d4fdf7723c08a3483"
)
_IMMUTABLE_WAVE2_V3_FAILURE_RECEIPT_PAYLOAD_SHA256 = (
    "8bdea86554c64b656b6a60325c2b5d69a12d9e58b2a7cbfdac01e0a9f9aabe13"
)
LEGACY_CONFIG_COMPATIBILITY_PROJECTION_SCHEMA_V1 = (
    "coordexp-swift-wave2-config-compatibility-projection-v1"
)
CONFIG_COMPATIBILITY_PROJECTION_SCHEMA = (
    "coordexp-swift-wave2-config-compatibility-projection-v2"
)
RUNTIME_CONFIG_ATTESTATION_SCHEMA = "coordexp-swift-wave2-runtime-config-attestation-v1"
_COMPATIBILITY_DEFAULT_PATH_VALUES = (
    ("training.forward_input_provider_mode", "synchronous"),
    ("packing.policy", "source_order_next_fit"),
    ("packing.window_size", None),
    ("packing.lookahead", None),
    ("packing.seed", 0),
    ("packing.worker_count", 1),
    ("packing.fragment_item_budget", 1024),
    ("packing.fragment_byte_budget", 4_194_304),
    ("packing.cursor_byte_budget", 65_536),
    ("packing.max_packs_per_fragment", None),
    ("resume", {"checkpoint_dir": None, "mode": "disabled"}),
)
_WAVE2_COMPATIBILITY_DEFAULT_PATH_VALUES = (
    *_COMPATIBILITY_DEFAULT_PATH_VALUES,
    ("runtime.determinism", {"mode": "legacy"}),
)
_COMPATIBILITY_DEFAULT_PATH = _COMPATIBILITY_DEFAULT_PATH_VALUES[0][0]
_COMPATIBILITY_DEFAULT_VALUE = _COMPATIBILITY_DEFAULT_PATH_VALUES[0][1]
FROZEN_PARENT_V2_MODEL_WEIGHT_SHA256 = (
    "e128f5f42f1a042702efc1eed5a787da36a4d586a17b538671260996056284aa"
)
FROZEN_PARENT_V2_EXAMPLE_IDS = (
    "coco2017_train_000000000009::aug:hflip",
    "coco2017_train_000000000025::aug:hvflip",
)
MODEL_WEIGHT_IDENTITY_SCHEMA = "coordexp-swift-base-model-weights-v1"
MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA = (
    "coordexp-swift-base-model-weight-hash-execution-policy-v1"
)
DEPENDENCY_IDENTITY_SCHEMA = "coordexp-swift-wave2-dependency-identity-v1"
QWEN_COMPONENT_IDENTITY_ATTESTATION_SCHEMA = (
    "coordexp-swift-wave2-qwen-component-identity-attestation-v1"
)
QWEN_PATCH_EMBED_LINEARIZATION_NAME = "qwen3_vl_patch_embed_linearization"
QWEN_COMPONENT_STABLE_FIELDS = (
    "base_model_path",
    "base_config_sha256",
    "tokenizer_sha256",
    "attn_implementation",
    "processor",
    "model",
    "tokens",
    "package_versions",
)
DENOMINATOR_SEMANTIC_FIELDS = (
    "term_name",
    "denominator_scope",
    "eligible_segment_count",
    "selected_atom_count",
    "skipped_segment_count",
)
_DENOMINATOR_ARTIFACT_FIELDS = frozenset(
    (*DENOMINATOR_SEMANTIC_FIELDS, "context_count")
)
_PLAN_COUNT_FIELDS = frozenset(
    {
        "count/supervised_atoms",
        "count/eligible_segments",
        "count/skipped_segments",
        "count/packs",
        "count/examples",
    }
)
_QWEN_COMPONENT_IDENTITY_FIELDS = frozenset(
    (*QWEN_COMPONENT_STABLE_FIELDS, "load_model", "runtime_patches")
)
_QWEN_RUNTIME_PATCH_RECEIPT_FIELDS = frozenset(
    {
        "name",
        "policy",
        "applied",
        "reason",
        "owner_path",
        "original_class",
        "owner_class",
        "patched_class",
        "projection_class",
        "original_forward_sha256",
        "replacement_forward_sha256",
        "in_channels",
        "temporal_patch_size",
        "patch_size",
        "embed_dim",
        "weight_shape",
        "bias",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "equivalence_probe",
    }
)
MAX_WEIGHT_INDEX_BYTES = 16 * 1024 * 1024
MAX_WEIGHT_DECLARATIONS = 100_000
MAX_WEIGHT_SHARDS = 256
MAX_WEIGHT_SHARD_BYTES = 16 * 1024 * 1024 * 1024
MAX_WEIGHT_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_GRADIENT_DIAGNOSTIC_NAMES = 4_096
MAX_GRADIENT_PARAMETER_SAMPLES = 4_096
MAX_FAILURE_EVIDENCE_BYTES = 8 * 1024 * 1024
_FAILURE_EVIDENCE_FIELDS = (
    "source_identity",
    "model_identity_attestation",
    "execution",
    "arms",
    "proof",
    "comparisons",
    "negative_discriminator",
    "timings",
    "gpu_memory",
    "measurement",
)
_FAILURE_STAGES = (
    "initialized",
    "receipt_target_preflight",
    "plan_loaded",
    "plan_revalidated",
    "cuda_preflight",
    "normalization_preflight",
    "accelerator_ready",
    "model_setup",
    "input_construction",
    "warmup_forward",
    "proof_off_forward",
    "packed_clean",
    "separate_reference",
    "negative_control",
    "comparisons",
    "finalization",
)
_FAILURE_PHASES = (
    "model_setup",
    "input_construction",
    "warmup_forward",
    "proof_off_forward",
    "packed_clean",
    "separate_reference",
    "negative_control",
    "comparison",
)
_FAILURE_STAGE_PHASE_COUNT = {
    "initialized": 0,
    "receipt_target_preflight": 0,
    "plan_loaded": 0,
    "plan_revalidated": 0,
    "cuda_preflight": 0,
    "normalization_preflight": 0,
    "accelerator_ready": 0,
    "model_setup": 1,
    "input_construction": 2,
    "warmup_forward": 3,
    "proof_off_forward": 4,
    "packed_clean": 5,
    "separate_reference": 6,
    "negative_control": 7,
    "comparisons": 7,
    "finalization": 8,
}
_FAILURE_STAGE_BASE_FIELDS = {
    "initialized": ("execution",),
    "receipt_target_preflight": ("execution",),
    "plan_loaded": ("source_identity", "execution"),
    "plan_revalidated": ("source_identity", "execution"),
    "cuda_preflight": ("source_identity", "execution"),
    "normalization_preflight": ("source_identity", "execution"),
    "accelerator_ready": ("source_identity", "execution"),
    "model_setup": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "measurement",
    ),
    "input_construction": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "measurement",
    ),
    "warmup_forward": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "measurement",
    ),
    "proof_off_forward": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "measurement",
    ),
    "packed_clean": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "arms",
        "proof",
        "timings",
        "measurement",
    ),
    "separate_reference": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "arms",
        "proof",
        "timings",
        "measurement",
    ),
    "negative_control": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "arms",
        "proof",
        "timings",
        "measurement",
    ),
    "comparisons": tuple(
        field for field in _FAILURE_EVIDENCE_FIELDS if field != "gpu_memory"
    ),
    "finalization": tuple(
        field for field in _FAILURE_EVIDENCE_FIELDS if field != "gpu_memory"
    ),
}

_V3_FAILURE_FIELD_ORDER = (
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
_V3_FAILURE_STAGES = (
    "initialized",
    "receipt_target_preflight",
    "plan_loaded",
    "plan_revalidated",
    "cuda_preflight",
    "normalization_preflight",
    "attempt_started",
    "accelerator_ready",
    "model_setup",
    "input_construction",
    "warmup_forward",
    "proof_off_forward",
    "packed_primary",
    "packed_repeat",
    "separate_reference",
    "negative_control",
    "comparisons",
    "finalization",
)
_V3_FAILURE_PHASES = (
    "model_setup",
    "input_construction",
    "warmup_forward",
    "proof_off_forward",
    "packed_primary",
    "packed_repeat",
    "separate_reference",
    "negative_control",
    "comparison",
)
_V3_FAILURE_STAGE_PHASE_COUNT = {
    "initialized": 0,
    "receipt_target_preflight": 0,
    "plan_loaded": 0,
    "plan_revalidated": 0,
    "cuda_preflight": 0,
    "normalization_preflight": 0,
    "attempt_started": 0,
    "accelerator_ready": 0,
    "model_setup": 1,
    "input_construction": 2,
    "warmup_forward": 3,
    "proof_off_forward": 4,
    "packed_primary": 5,
    "packed_repeat": 6,
    "separate_reference": 7,
    "negative_control": 8,
    "comparisons": 8,
    "finalization": 9,
}
_V3_FAILURE_STAGE_REQUIRED_FIELDS = {
    "initialized": ("execution",),
    "receipt_target_preflight": ("execution",),
    "plan_loaded": ("source_identity", "execution"),
    "plan_revalidated": ("source_identity", "execution"),
    "cuda_preflight": ("source_identity", "execution"),
    "normalization_preflight": ("source_identity", "execution"),
    "attempt_started": (
        "source_identity",
        "execution",
        "attempt_marker",
        "trainable_inventory",
    ),
    "accelerator_ready": (
        "source_identity",
        "execution",
        "attempt_marker",
        "trainable_inventory",
    ),
    "model_setup": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "attempt_marker",
        "trainable_inventory",
        "measurement",
    ),
    "input_construction": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "attempt_marker",
        "trainable_inventory",
        "measurement",
    ),
    "warmup_forward": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "attempt_marker",
        "trainable_inventory",
        "measurement",
    ),
    "proof_off_forward": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "attempt_marker",
        "trainable_inventory",
        "measurement",
    ),
    "packed_primary": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "attempt_marker",
        "trainable_inventory",
        "arms",
        "proof",
        "timings",
        "measurement",
    ),
    "packed_repeat": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "attempt_marker",
        "trainable_inventory",
        "arms",
        "proof",
        "timings",
        "measurement",
    ),
    "separate_reference": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "attempt_marker",
        "trainable_inventory",
        "arms",
        "proof",
        "timings",
        "measurement",
    ),
    "negative_control": (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "attempt_marker",
        "trainable_inventory",
        "arms",
        "proof",
        "timings",
        "measurement",
    ),
    "comparisons": tuple(
        field for field in _V3_FAILURE_FIELD_ORDER if field != "gpu_memory"
    ),
    "finalization": tuple(
        field for field in _V3_FAILURE_FIELD_ORDER if field != "gpu_memory"
    ),
}
_V3_FAILURE_STAGE_ARM_NAMES = {
    "packed_primary": ("packed_primary",),
    "packed_repeat": ("packed_primary", "packed_repeat"),
    "separate_reference": (
        "packed_primary",
        "packed_repeat",
        "separate_reference",
    ),
    "negative_control": (
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "packed_merged_boundary_negative",
    ),
    "comparisons": (
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "packed_merged_boundary_negative",
    ),
    "finalization": (
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "packed_merged_boundary_negative",
    ),
}


class ParityContractError(RuntimeContractError):
    """Fail-closed packed-parity contract violation."""


@dataclass(frozen=True, order=True)
class SemanticAtomKey:
    """Pack-position-independent identity for one supervised causal target."""

    example_id: str
    logical_target_position: int
    logical_target_end: int
    token_id: int
    token_type: str
    object_id: str | None
    field: str | None
    source: str | None

    @classmethod
    def from_atom(cls, atom: Any) -> "SemanticAtomKey":
        logical_end = getattr(atom, "logical_target_end", None)
        if logical_end is None:
            logical_end = int(getattr(atom, "logical_target_position")) + 1
        return cls(
            example_id=_non_empty_text(getattr(atom, "example_id", None), "example_id"),
            logical_target_position=_non_negative_int(
                getattr(atom, "logical_target_position", None),
                "logical_target_position",
            ),
            logical_target_end=_positive_int(logical_end, "logical_target_end"),
            token_id=_non_negative_int(getattr(atom, "token_id", None), "token_id"),
            token_type=_non_empty_text(getattr(atom, "token_type", None), "token_type"),
            object_id=_optional_text(getattr(atom, "object_id", None), "object_id"),
            field=_optional_text(getattr(atom, "field", None), "field"),
            source=_optional_text(getattr(atom, "source", None), "source"),
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "logical_target_position": self.logical_target_position,
            "logical_target_end": self.logical_target_end,
            "token_id": self.token_id,
            "token_type": self.token_type,
            "object_id": self.object_id,
            "field": self.field,
            "source": self.source,
        }


@dataclass(frozen=True)
class TensorComparison:
    allclose: bool
    rtol: float
    atol: float
    max_abs_diff: float
    max_rel_diff: float
    compared_value_count: int

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "allclose": self.allclose,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "compared_value_count": self.compared_value_count,
        }


@dataclass(frozen=True)
class GradientRecord:
    name: str
    shape: tuple[int, ...]
    parameter_dtype: str
    grad: torch.Tensor | None
    gradient_dtype: str | None = None
    gradient_provenance_dtype: str | None = BF16_COMPUTE_PROVENANCE_DTYPE

    @property
    def resolved_gradient_dtype(self) -> str | None:
        if self.grad is None:
            return None
        return self.gradient_dtype or str(self.grad.dtype)

    def to_inventory_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "parameter_dtype": self.parameter_dtype,
            "parameter_storage_dtype": self.parameter_dtype,
            "gradient_dtype": self.resolved_gradient_dtype,
            "gradient_provenance_dtype": self.gradient_provenance_dtype,
            "grad_status": "none" if self.grad is None else "finite",
            "gradient_sha256": None if self.grad is None else tensor_sha256(self.grad),
        }


@dataclass(frozen=True)
class RngSnapshot:
    cpu: torch.Tensor
    cuda: tuple[torch.Tensor, ...]


def canonical_json_bytes(value: Any) -> bytes:
    """Encode strict deterministic JSON and reject non-finite values."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ParityContractError(
            "parity artifact is not strict JSON",
            code="qwen.parity.strict_json",
            context={"value_type": type(value).__name__, "error": str(exc)},
            cause=exc,
        ) from exc
    return text.encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ParityContractError(
            "parity identity file is unreadable",
            code="qwen.parity.identity_file",
            context={"path": str(source), "error": type(exc).__name__},
            cause=exc,
        ) from exc
    return digest.hexdigest()


def write_strict_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    on_linked: Callable[[], None] | None = None,
) -> Path:
    """Publish one strict JSON artifact without exposing partial bytes."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ParityContractError(
                "parity artifact target already exists",
                code="qwen.parity.artifact_collision",
                context={"path": str(target)},
                cause=exc,
            ) from exc
        if on_linked is not None:
            on_linked()
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError as exc:
            raise ParityContractError(
                "parity artifact directory cannot be opened for durability sync",
                code="qwen.parity.artifact_directory_sync",
                context={
                    "path": str(target.parent),
                    "error_type": type(exc).__name__,
                },
                cause=exc,
            ) from exc
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def assert_absent_artifact_target(path: str | Path) -> Path:
    """Fail before expensive work unless an artifact can be published safely."""

    requested = Path(path).expanduser()
    target = requested.resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise ParityContractError(
            "parity artifact target already exists",
            code="qwen.parity.artifact_collision",
            context={"path": str(target)},
        )
    parent = target.parent
    if not parent.is_dir():
        raise ParityContractError(
            "parity artifact parent must already exist",
            code="qwen.parity.artifact_parent",
            context={"path": str(parent)},
        )
    current = parent
    while current != current.parent:
        if current.is_symlink():
            raise ParityContractError(
                "parity artifact path must not traverse symlinks",
                code="qwen.parity.artifact_symlink",
                context={"path": str(current)},
            )
        current = current.parent
    return target


def load_strict_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ParityContractError(
            "parity artifact is not readable strict JSON",
            code="qwen.parity.json_read",
            context={"path": str(source), "error": type(exc).__name__},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        raise ParityContractError(
            "parity artifact root must be a JSON object",
            code="qwen.parity.json_shape",
            context={"path": str(source), "value_type": type(payload).__name__},
        )
    canonical_json_bytes(payload)
    return payload


def frozen_parent_v2_identity() -> dict[str, Any]:
    """Return the only historical workload that a Wave 2 v3 plan may inherit."""

    return {
        "schema": "coordexp-swift-wave2-parent-v2-identity-v1",
        "plan_schema": LEGACY_PARITY_PLAN_SCHEMA_V2,
        "receipt_schema": LEGACY_PARITY_RECEIPT_SCHEMA_V2,
        "plan_sha256": FROZEN_PARENT_V2_PLAN_SHA256,
        "plan_file_sha256": FROZEN_PARENT_V2_PLAN_FILE_SHA256,
        "config_fingerprint": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
        "model_weight_aggregate_sha256": FROZEN_PARENT_V2_MODEL_WEIGHT_SHA256,
        "seed": 17,
        "source_indices": [0, 1],
        "example_ids": list(FROZEN_PARENT_V2_EXAMPLE_IDS),
    }


def validate_parent_v2_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Fail unless a v3 plan binds the exact authenticated v2 workload."""

    expected = frozen_parent_v2_identity()
    _expect_exact_keys(identity, set(expected), owner="parent_v2")
    if dict(identity) != expected:
        drifted = sorted(key for key in expected if identity.get(key) != expected[key])
        raise ParityContractError(
            "Wave 2 v3 parent identity differs from the frozen v2 workload",
            code="qwen.parity.parent_v2_identity",
            context={"drifted_fields": drifted},
        )
    return expected


def validate_parent_v2_plan_file(path: str | Path) -> dict[str, Any]:
    """Authenticate the exact immutable v2 plan selected by the v3 workload."""

    source = Path(path).expanduser().resolve()
    if sha256_file(source) != FROZEN_PARENT_V2_PLAN_FILE_SHA256:
        raise ParityContractError(
            "parent v2 plan file bytes differ from the frozen artifact",
            code="qwen.parity.parent_v2_plan_file",
            context={"path": str(source)},
        )
    plan = load_strict_json(source)
    validated = _validate_parity_plan_v2(
        plan,
        allow_authenticated_frozen_parent_without_cuda_runtime=True,
    )
    if (
        validated["plan_sha256"] != FROZEN_PARENT_V2_PLAN_SHA256
        or validated["config_identity"].get("fingerprint")
        != FROZEN_PARENT_V2_CONFIG_FINGERPRINT
        or validated["model_weight_identity"].get("aggregate_sha256")
        != FROZEN_PARENT_V2_MODEL_WEIGHT_SHA256
        or validated["selection"].get("source_indices") != [0, 1]
        or validated["selection"].get("example_ids")
        != list(FROZEN_PARENT_V2_EXAMPLE_IDS)
        or validated["determinism"].get("seed") != 17
    ):
        raise ParityContractError(
            "parent v2 plan contents differ from the frozen workload",
            code="qwen.parity.parent_v2_plan_identity",
            context={"path": str(source)},
        )
    return validated


def config_compatibility_projection(
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only the enumerated later compatibility defaults from live config."""

    current = json.loads(canonical_json_bytes(dict(resolved_config)))
    current_sha256 = sha256_json(current)
    if current_sha256 != FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT:
        raise ParityContractError(
            "Wave 2 v3 resolved config differs from the frozen runtime identity",
            code="qwen.parity.runtime_config_drift",
            context={
                "expected": FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT,
                "observed": current_sha256,
            },
        )
    _remove_exact_config_path_values(
        current,
        _wave2_config_compatibility_removed_path_values(),
    )
    projected_sha256 = sha256_json(current)
    if projected_sha256 != FROZEN_PARENT_V2_CONFIG_FINGERPRINT:
        raise ParityContractError(
            "Wave 2 v3 config projection differs from the immutable parent",
            code="qwen.parity.config_projection_drift",
            context={
                "expected": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
                "observed": projected_sha256,
            },
        )
    projection = {
        "schema": CONFIG_COMPATIBILITY_PROJECTION_SCHEMA,
        "current_config_sha256": current_sha256,
        "removed_path_values": _wave2_config_compatibility_removed_path_values(),
        "projected_config_sha256": projected_sha256,
        "parent_config_fingerprint": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
        "policy": "remove_exact_enumerated_later_strict_defaults",
    }
    return validate_config_compatibility_projection(
        projection,
        resolved_config=resolved_config,
    )


def validate_config_compatibility_projection(
    projection: Mapping[str, Any],
    *,
    resolved_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate current v2 projection or immutable historical v1 evidence."""

    schema = projection.get("schema")
    if schema == CONFIG_COMPATIBILITY_PROJECTION_SCHEMA:
        expected = _current_config_compatibility_projection()
    elif schema == LEGACY_CONFIG_COMPATIBILITY_PROJECTION_SCHEMA_V1:
        expected = _historical_config_compatibility_projection_v1()
    else:
        expected = _current_config_compatibility_projection()
    _expect_exact_keys(
        projection,
        set(expected),
        owner="config_compatibility_projection",
    )
    if dict(projection) != expected:
        drifted = sorted(
            key for key in expected if projection.get(key) != expected[key]
        )
        raise ParityContractError(
            "Wave 2 config compatibility projection is not an exact supported projection",
            code="qwen.parity.config_projection_attestation",
            context={"drifted_fields": drifted},
        )
    if resolved_config is not None:
        current = json.loads(canonical_json_bytes(dict(resolved_config)))
        current_sha256 = sha256_json(current)
        if current_sha256 != expected["current_config_sha256"]:
            raise ParityContractError(
                "live resolved config does not satisfy the frozen compatibility projection",
                code="qwen.parity.runtime_config_drift",
                context={
                    "expected_sha256": expected["current_config_sha256"],
                    "observed_sha256": current_sha256,
                },
            )
        _remove_exact_config_path_values(current, expected["removed_path_values"])
        if sha256_json(current) != expected["projected_config_sha256"]:
            raise ParityContractError(
                "live resolved config has drift beyond the enumerated defaults",
                code="qwen.parity.config_projection_drift",
                context={},
            )
    canonical_json_bytes(projection)
    return dict(projection)


def current_config_compatibility_removed_path_values() -> list[dict[str, Any]]:
    """Return a fresh canonical copy of the exact current compatibility removals."""

    return json.loads(
        canonical_json_bytes(
            [
                {"path": path, "value": value}
                for path, value in _COMPATIBILITY_DEFAULT_PATH_VALUES
            ]
        )
    )


def _wave2_config_compatibility_removed_path_values() -> list[dict[str, Any]]:
    """Return Wave 2's exact current-to-parent compatibility removals."""

    return json.loads(
        canonical_json_bytes(
            [
                {"path": path, "value": value}
                for path, value in _WAVE2_COMPATIBILITY_DEFAULT_PATH_VALUES
            ]
        )
    )


def _current_config_compatibility_projection() -> dict[str, Any]:
    return {
        "schema": CONFIG_COMPATIBILITY_PROJECTION_SCHEMA,
        "current_config_sha256": FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT,
        "removed_path_values": _wave2_config_compatibility_removed_path_values(),
        "projected_config_sha256": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
        "parent_config_fingerprint": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
        "policy": "remove_exact_enumerated_later_strict_defaults",
    }


def _historical_config_compatibility_projection_v1() -> dict[str, Any]:
    return {
        "schema": LEGACY_CONFIG_COMPATIBILITY_PROJECTION_SCHEMA_V1,
        "current_config_sha256": HISTORICAL_V3_RUNTIME_CONFIG_FINGERPRINT,
        "removed_path_values": [
            {
                "path": _COMPATIBILITY_DEFAULT_PATH,
                "value": _COMPATIBILITY_DEFAULT_VALUE,
            }
        ],
        "projected_config_sha256": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
        "parent_config_fingerprint": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
        "policy": "remove_exactly_one_later_strict_default",
    }


def _remove_exact_config_path_values(
    config: dict[str, Any],
    path_values: Sequence[Mapping[str, Any]],
) -> None:
    for row in path_values:
        path = row.get("path")
        if not isinstance(path, str) or not path:
            raise ParityContractError(
                "config projection path is invalid",
                code="qwen.parity.config_projection_path",
                context={"path": path},
            )
        parts = path.split(".")
        owner: Any = config
        for part in parts[:-1]:
            if not isinstance(owner, dict) or part not in owner:
                raise ParityContractError(
                    "config projection path is missing",
                    code="qwen.parity.config_projection_path",
                    context={"path": path},
                )
            owner = owner[part]
        leaf = parts[-1]
        if not isinstance(owner, dict) or leaf not in owner:
            raise ParityContractError(
                "config projection path is missing",
                code="qwen.parity.config_projection_path",
                context={"path": path},
            )
        expected_value = row.get("value")
        observed_value = owner[leaf]
        if observed_value != expected_value:
            raise ParityContractError(
                "config projection value is not the exact compatibility default",
                code="qwen.parity.config_projection_value",
                context={
                    "path": path,
                    "expected": expected_value,
                    "observed": observed_value,
                },
            )
        del owner[leaf]


def validate_v3_config_identity(
    identity: Mapping[str, Any],
    *,
    resolved_config: Mapping[str, Any] | None = None,
    parent_v2_config_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate exact current identity or immutable historical v1 identity."""

    _expect_exact_keys(
        identity,
        {
            "entry_path",
            "fingerprint",
            "schema_version",
            "loader_version",
            "resolved_config_sha256",
            "sources",
            "compatibility_projection",
            "runtime_config_attestation",
        },
        owner="config_identity",
    )
    projection = validate_config_compatibility_projection(
        _mapping(
            identity["compatibility_projection"],
            "config_identity.compatibility_projection",
        ),
        resolved_config=resolved_config,
    )
    expected_config_fingerprint = projection["current_config_sha256"]
    if (
        identity["fingerprint"] != expected_config_fingerprint
        or identity["resolved_config_sha256"] != expected_config_fingerprint
    ):
        raise ParityContractError(
            "Wave 2 v3 plan does not retain the exact current runtime config identity",
            code="qwen.parity.runtime_config_identity",
            context={
                "fingerprint": identity["fingerprint"],
                "resolved_config_sha256": identity["resolved_config_sha256"],
            },
        )
    expected_attestation = {
        "schema": RUNTIME_CONFIG_ATTESTATION_SCHEMA,
        "status": "passed",
        "config_fingerprint": expected_config_fingerprint,
        "field_path": _COMPATIBILITY_DEFAULT_PATH,
        "required_value": _COMPATIBILITY_DEFAULT_VALUE,
        "resolved_value": _COMPATIBILITY_DEFAULT_VALUE,
        "compatibility_projection_sha256": sha256_json(projection),
    }
    if identity["runtime_config_attestation"] != expected_attestation:
        raise ParityContractError(
            "Wave 2 v3 runtime config attestation is invalid",
            code="qwen.parity.runtime_config_attestation",
            context={},
        )
    if parent_v2_config_identity is not None:
        parent = dict(parent_v2_config_identity)
        stable_fields = (
            "entry_path",
            "schema_version",
            "loader_version",
            "sources",
        )
        drifted = [
            field for field in stable_fields if identity[field] != parent.get(field)
        ]
        if (
            parent.get("fingerprint") != FROZEN_PARENT_V2_CONFIG_FINGERPRINT
            or parent.get("resolved_config_sha256")
            != FROZEN_PARENT_V2_CONFIG_FINGERPRINT
            or drifted
        ):
            raise ParityContractError(
                "Wave 2 v3 config identity differs beyond the allowed default projection",
                code="qwen.parity.config_projection_identity_drift",
                context={"drifted_fields": drifted},
            )
    canonical_json_bytes(identity)
    return dict(identity)


def attest_v3_runtime_config(
    config_identity: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reassert the live synchronous mode against the recorded v3 identity."""

    identity = validate_v3_config_identity(
        config_identity,
        resolved_config=resolved_config,
    )
    return dict(
        _mapping(
            identity["runtime_config_attestation"],
            "config_identity.runtime_config_attestation",
        )
    )


def frozen_trainable_inventory_declaration() -> dict[str, Any]:
    """Return the model-free structural declaration for the frozen v3 surface."""

    return {
        "schema": "coordexp-swift-wave2-trainable-inventory-declaration-v1",
        "total_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
        "group_counts": dict(EXPECTED_TRAINABLE_STRUCTURE),
        "strict_name_suffixes": dict(TRAINABLE_PARAMETER_SUFFIXES),
        "expected_gradient_presence": "required_finite_every_clean_arm",
        "aggregate_nonzero_gradient_required": True,
        "compute_provenance_dtype": BF16_COMPUTE_PROVENANCE_DTYPE,
        "comparison_dtype": FP32_COMPARISON_DTYPE,
    }


def validate_trainable_inventory_declaration(
    declaration: Mapping[str, Any],
) -> dict[str, Any]:
    expected = frozen_trainable_inventory_declaration()
    _expect_exact_keys(declaration, set(expected), owner="trainable_declaration")
    if dict(declaration) != expected:
        drifted = sorted(
            key for key in expected if declaration.get(key) != expected[key]
        )
        raise ParityContractError(
            "trainable inventory declaration differs from the frozen v3 surface",
            code="qwen.parity.trainable_declaration",
            context={"drifted_fields": drifted},
        )
    return expected


def _trainable_group(name: str) -> str:
    matches = [
        group
        for group, suffix in TRAINABLE_PARAMETER_SUFFIXES.items()
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ParityContractError(
            "trainable parameter does not match exactly one frozen suffix rule",
            code="qwen.parity.trainable_name",
            context={"name": name, "matched_groups": matches},
        )
    return matches[0]


def concrete_trainable_inventory(
    model: Any,
    *,
    expected_gradient_dtype: str = FP32_COMPARISON_DTYPE,
    compute_provenance_dtype: str = BF16_COMPUTE_PROVENANCE_DTYPE,
) -> dict[str, Any]:
    """Bind exact post-setup trainable names/shapes/dtypes without reading grads."""

    if expected_gradient_dtype != FP32_COMPARISON_DTYPE:
        raise ParityContractError(
            "v3 expected gradient dtype must remain FP32",
            code="qwen.parity.trainable_gradient_dtype",
            context={"expected_gradient_dtype": expected_gradient_dtype},
        )
    if compute_provenance_dtype != BF16_COMPUTE_PROVENANCE_DTYPE:
        raise ParityContractError(
            "v3 trainable compute provenance must remain BF16",
            code="qwen.parity.trainable_compute_provenance",
            context={"compute_provenance_dtype": compute_provenance_dtype},
        )
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        text_name = _non_empty_text(str(name), "trainable.name")
        rows.append(
            {
                "name": text_name,
                "group": _trainable_group(text_name),
                "shape": [int(item) for item in parameter.shape],
                "parameter_storage_dtype": str(parameter.dtype),
                "expected_gradient_dtype": expected_gradient_dtype,
                "compute_provenance_dtype": compute_provenance_dtype,
            }
        )
    rows.sort(key=lambda row: str(row["name"]))
    body = {
        "schema": "coordexp-swift-wave2-concrete-trainable-inventory-v1",
        "declaration_sha256": sha256_json(frozen_trainable_inventory_declaration()),
        "total_count": len(rows),
        "group_counts": {
            group: sum(row["group"] == group for row in rows)
            for group in EXPECTED_TRAINABLE_STRUCTURE
        },
        "parameters": rows,
    }
    inventory = {**body, "inventory_sha256": sha256_json(body)}
    return validate_concrete_trainable_inventory(inventory)


def validate_concrete_trainable_inventory(
    inventory: Mapping[str, Any],
    *,
    declaration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an exact post-setup inventory before marker/GPU publication."""

    expected_declaration = validate_trainable_inventory_declaration(
        declaration or frozen_trainable_inventory_declaration()
    )
    _expect_exact_keys(
        inventory,
        {
            "schema",
            "declaration_sha256",
            "total_count",
            "group_counts",
            "parameters",
            "inventory_sha256",
        },
        owner="concrete_trainable_inventory",
    )
    if inventory["schema"] != "coordexp-swift-wave2-concrete-trainable-inventory-v1":
        raise ParityContractError(
            "concrete trainable inventory schema is unsupported",
            code="qwen.parity.trainable_inventory_schema",
            context={"schema": inventory["schema"]},
        )
    if inventory["declaration_sha256"] != sha256_json(expected_declaration):
        raise ParityContractError(
            "concrete inventory is not bound to the frozen declaration",
            code="qwen.parity.trainable_inventory_declaration",
            context={},
        )
    rows = _list_of_mappings(inventory["parameters"], "trainable.parameters")
    names: list[str] = []
    observed_counts = {group: 0 for group in EXPECTED_TRAINABLE_STRUCTURE}
    for row in rows:
        _expect_exact_keys(
            row,
            {
                "name",
                "group",
                "shape",
                "parameter_storage_dtype",
                "expected_gradient_dtype",
                "compute_provenance_dtype",
            },
            owner="trainable.parameter",
        )
        name = _non_empty_text(row["name"], "trainable.parameter.name")
        group = _non_empty_text(row["group"], "trainable.parameter.group")
        if group != _trainable_group(name):
            raise ParityContractError(
                "trainable parameter group contradicts its strict suffix",
                code="qwen.parity.trainable_group",
                context={"name": name, "group": group},
            )
        shape = _int_list(row["shape"], "trainable.parameter.shape")
        if not shape or any(item <= 0 for item in shape):
            raise ParityContractError(
                "trainable parameter shape must be non-empty and positive",
                code="qwen.parity.trainable_shape",
                context={"name": name, "shape": shape},
            )
        _non_empty_text(
            row["parameter_storage_dtype"], "trainable.parameter.storage_dtype"
        )
        if (
            row["expected_gradient_dtype"] != FP32_COMPARISON_DTYPE
            or row["compute_provenance_dtype"] != BF16_COMPUTE_PROVENANCE_DTYPE
        ):
            raise ParityContractError(
                "trainable parameter gradient dtype/provenance contract drifted",
                code="qwen.parity.trainable_dtype_contract",
                context={"name": name},
            )
        names.append(name)
        observed_counts[group] += 1
    if names != sorted(set(names)):
        raise ParityContractError(
            "concrete trainable names must be unique and sorted",
            code="qwen.parity.trainable_name_inventory",
            context={},
        )
    if (
        inventory["total_count"] != len(rows)
        or len(rows) != EXPECTED_TRAINABLE_PARAMETER_COUNT
        or dict(_mapping(inventory["group_counts"], "trainable.group_counts"))
        != observed_counts
        or observed_counts != EXPECTED_TRAINABLE_STRUCTURE
    ):
        raise ParityContractError(
            "concrete trainable inventory does not match the frozen 589 surface",
            code="qwen.parity.trainable_inventory_count",
            context={"total_count": len(rows), "group_counts": observed_counts},
        )
    fingerprint = _sha256_text(
        inventory["inventory_sha256"], "trainable.inventory_sha256"
    )
    body = dict(inventory)
    del body["inventory_sha256"]
    if sha256_json(body) != fingerprint:
        raise ParityContractError(
            "concrete trainable inventory fingerprint is invalid",
            code="qwen.parity.trainable_inventory_fingerprint",
            context={},
        )
    canonical_json_bytes(inventory)
    return dict(inventory)


def finalize_attempt_marker(payload: Mapping[str, Any]) -> dict[str, Any]:
    marker = dict(payload)
    if "marker_sha256" in marker:
        raise ParityContractError(
            "unfinalized attempt marker must not contain marker_sha256",
            code="qwen.parity.marker_prefingerprinted",
            context={},
        )
    marker["marker_sha256"] = sha256_json(marker)
    return validate_attempt_marker(marker)


def validate_attempt_marker(
    marker: Mapping[str, Any],
    *,
    expected_plan: Mapping[str, Any] | None = None,
    expected_receipt_target: str | Path | None = None,
) -> dict[str, Any]:
    _expect_exact_keys(
        marker,
        {
            "schema",
            "status",
            "plan_sha256",
            "receipt_target",
            "command_identity",
            "source_identity",
            "concrete_trainable_inventory",
            "marker_sha256",
        },
        owner="attempt_marker",
    )
    if (
        marker["schema"] != PARITY_ATTEMPT_MARKER_SCHEMA
        or marker["status"] != "attempt_started"
    ):
        raise ParityContractError(
            "attempt marker schema or status is unsupported",
            code="qwen.parity.marker_header",
            context={"schema": marker["schema"], "status": marker["status"]},
        )
    plan_sha256 = _sha256_text(marker["plan_sha256"], "marker.plan_sha256")
    receipt_target = _non_empty_text(marker["receipt_target"], "marker.receipt_target")
    if not Path(receipt_target).is_absolute():
        raise ParityContractError(
            "attempt marker receipt target must be absolute",
            code="qwen.parity.marker_receipt_target",
            context={"receipt_target": receipt_target},
        )
    command_identity = _mapping(marker["command_identity"], "marker.command_identity")
    source_identity = _mapping(marker["source_identity"], "marker.source_identity")
    if not command_identity or not source_identity:
        raise ParityContractError(
            "attempt marker requires command and source identities",
            code="qwen.parity.marker_identity",
            context={},
        )
    canonical_json_bytes(command_identity)
    canonical_json_bytes(source_identity)
    validate_concrete_trainable_inventory(
        _mapping(
            marker["concrete_trainable_inventory"],
            "marker.concrete_trainable_inventory",
        )
    )
    if expected_plan is not None:
        plan = validate_parity_plan(expected_plan)
        if plan_sha256 != plan["plan_sha256"]:
            raise ParityContractError(
                "attempt marker is not bound to the authenticated v3 plan",
                code="qwen.parity.marker_plan_binding",
                context={},
            )
    if expected_receipt_target is not None and receipt_target != str(
        Path(expected_receipt_target).expanduser().resolve(strict=False)
    ):
        raise ParityContractError(
            "attempt marker receipt target differs from the audited target",
            code="qwen.parity.marker_receipt_binding",
            context={},
        )
    fingerprint = _sha256_text(marker["marker_sha256"], "marker.marker_sha256")
    body = dict(marker)
    del body["marker_sha256"]
    if sha256_json(body) != fingerprint:
        raise ParityContractError(
            "attempt marker fingerprint does not authenticate its body",
            code="qwen.parity.marker_fingerprint",
            context={},
        )
    canonical_json_bytes(marker)
    return dict(marker)


def finalize_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a self-authenticating fingerprint to a complete plan body."""

    plan = dict(payload)
    if "plan_sha256" in plan:
        raise ParityContractError(
            "unfinalized plan must not contain plan_sha256",
            code="qwen.parity.plan_prefingerprinted",
            context={},
        )
    plan["plan_sha256"] = sha256_json(plan)
    validate_parity_plan(plan)
    return plan


def attest_qwen_component_identity(
    expected_plan_identity: Mapping[str, Any],
    loaded_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Attest one exact model-free-to-loaded Qwen component transition."""

    expected = _validate_qwen_component_identity(
        expected_plan_identity,
        owner="expected_plan_model_identity",
        expected_load_model=False,
    )
    loaded = _validate_qwen_component_identity(
        loaded_identity,
        owner="loaded_model_identity",
        expected_load_model=True,
    )
    expected_projection = {
        field: expected[field] for field in QWEN_COMPONENT_STABLE_FIELDS
    }
    loaded_projection = {field: loaded[field] for field in QWEN_COMPONENT_STABLE_FIELDS}
    drifted_fields = [
        field
        for field in QWEN_COMPONENT_STABLE_FIELDS
        if expected_projection[field] != loaded_projection[field]
    ]
    if drifted_fields:
        raise ParityContractError(
            "loaded Qwen component identity differs from the prepared plan",
            code="qwen.parity.model_identity_drift",
            context={"drifted_fields": drifted_fields},
        )

    expected_patch = _only_patch_receipt(
        expected["runtime_patches"], owner="expected_plan_model_identity"
    )
    loaded_patch = _only_patch_receipt(
        loaded["runtime_patches"], owner="loaded_model_identity"
    )
    policy = _validate_model_free_patch_receipt(expected_patch)
    _validate_loaded_patch_receipt(loaded_patch, policy=policy)
    transition = {
        "name": QWEN_PATCH_EMBED_LINEARIZATION_NAME,
        "policy": policy,
        "expected_reason": "model_not_loaded",
        "loaded_reason": loaded_patch["reason"],
        "loaded_applied": loaded_patch["applied"],
        "expected_receipt_sha256": sha256_json(expected_patch),
        "loaded_receipt_sha256": sha256_json(loaded_patch),
    }
    return {
        "schema": QWEN_COMPONENT_IDENTITY_ATTESTATION_SCHEMA,
        "status": "pass",
        "expected_plan_model_identity": expected,
        "expected_plan_model_identity_sha256": sha256_json(expected),
        "loaded_model_identity_sha256": sha256_json(loaded),
        "stable_projection_fields": list(QWEN_COMPONENT_STABLE_FIELDS),
        "stable_projection_sha256": sha256_json(expected_projection),
        "runtime_patch_transition": transition,
    }


def validate_qwen_component_identity_attestation(
    attestation: Mapping[str, Any],
    *,
    loaded_model_identity: Mapping[str, Any],
    expected_plan_model_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute a receipt attestation against its loaded source identity."""

    _expect_exact_keys(
        attestation,
        {
            "schema",
            "status",
            "expected_plan_model_identity",
            "expected_plan_model_identity_sha256",
            "loaded_model_identity_sha256",
            "stable_projection_fields",
            "stable_projection_sha256",
            "runtime_patch_transition",
        },
        owner="model_identity_attestation",
    )
    embedded_expected = _mapping(
        attestation["expected_plan_model_identity"],
        "model_identity_attestation.expected_plan_model_identity",
    )
    if expected_plan_model_identity is not None and dict(embedded_expected) != dict(
        expected_plan_model_identity
    ):
        raise ParityContractError(
            "receipt model identity attestation differs from the authenticated plan",
            code="qwen.parity.model_identity_plan_mismatch",
            context={},
        )
    recomputed = attest_qwen_component_identity(
        embedded_expected,
        loaded_model_identity,
    )
    if dict(attestation) != recomputed:
        raise ParityContractError(
            "receipt model identity attestation is inconsistent",
            code="qwen.parity.model_identity_attestation",
            context={},
        )
    return recomputed


def validate_cross_arm_bf16_loss_term_scalars(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete v2 cross-arm BF16-derived scalar comparison."""

    _expect_exact_keys(
        artifact,
        {
            "passed",
            "source_forward_dtype",
            "comparison_dtype",
            "rtol",
            "atol",
            "fields",
            "missing",
            "extra",
            "terms",
        },
        owner="cross_arm_bf16_loss_term_scalars",
    )
    fields = [
        "raw_loss",
        "weighted_loss",
        "segment_mean_numerator",
        "token_weighted_diagnostic",
    ]
    if (
        artifact["source_forward_dtype"] != "torch.bfloat16"
        or artifact["comparison_dtype"] != "torch.float32"
        or artifact["rtol"] != BF16_RTOL
        or artifact["atol"] != BF16_ATOL
        or artifact["fields"] != fields
    ):
        raise ParityContractError(
            "cross-arm loss-term scalar identity or tolerance band is invalid",
            code="qwen.parity.loss_term_scalar_header",
            context={
                "source_forward_dtype": artifact["source_forward_dtype"],
                "comparison_dtype": artifact["comparison_dtype"],
                "rtol": artifact["rtol"],
                "atol": artifact["atol"],
                "fields": artifact["fields"],
            },
        )
    missing = _text_list_allow_empty(artifact["missing"], "loss_term_scalars.missing")
    extra = _text_list_allow_empty(artifact["extra"], "loss_term_scalars.extra")
    if len(missing) != len(set(missing)) or len(extra) != len(set(extra)):
        raise ParityContractError(
            "cross-arm loss-term inventory deltas contain duplicates",
            code="qwen.parity.loss_term_scalar_inventory",
            context={"missing": missing, "extra": extra},
        )
    terms = _list_of_mappings(artifact["terms"], "loss_term_scalars.terms")
    if not terms:
        raise ParityContractError(
            "cross-arm loss-term scalar comparison requires non-empty terms",
            code="qwen.parity.loss_term_scalar_inventory",
            context={},
        )
    term_names: list[str] = []
    nested_allclose = True
    for term in terms:
        _expect_exact_keys(
            term, {"term_name", "fields"}, owner="loss_term_scalars.term"
        )
        term_name = _non_empty_text(term["term_name"], "loss_term_scalars.term_name")
        term_names.append(term_name)
        field_rows = _list_of_mappings(
            term["fields"], f"loss_term_scalars.{term_name}.fields"
        )
        observed_fields = [row.get("field") for row in field_rows]
        if observed_fields != fields:
            raise ParityContractError(
                "cross-arm loss-term scalar field inventory is invalid",
                code="qwen.parity.loss_term_scalar_fields",
                context={"term_name": term_name, "fields": observed_fields},
            )
        for row in field_rows:
            field = str(row["field"])
            _expect_exact_keys(
                row,
                {
                    "field",
                    "packed_value",
                    "separate_shared_value",
                    "raw_delta",
                    "allclose",
                    "rtol",
                    "atol",
                    "max_abs_diff",
                    "max_rel_diff",
                    "compared_value_count",
                },
                owner=f"loss_term_scalars.{term_name}.{field}",
            )
            packed_value = _finite_number(
                row["packed_value"], f"loss_term_scalars.{term_name}.{field}.packed"
            )
            separate_value = _finite_number(
                row["separate_shared_value"],
                f"loss_term_scalars.{term_name}.{field}.separate",
            )
            raw_delta = _finite_number(
                row["raw_delta"], f"loss_term_scalars.{term_name}.{field}.delta"
            )
            expected = compare_tensors(
                torch.tensor(packed_value, dtype=torch.float32),
                torch.tensor(separate_value, dtype=torch.float32),
                rtol=BF16_RTOL,
                atol=BF16_ATOL,
            ).to_artifact_dict()
            expected_row = {
                "field": field,
                "packed_value": float(
                    torch.tensor(packed_value, dtype=torch.float32).item()
                ),
                "separate_shared_value": float(
                    torch.tensor(separate_value, dtype=torch.float32).item()
                ),
                "raw_delta": float(
                    (
                        torch.tensor(packed_value, dtype=torch.float32)
                        - torch.tensor(separate_value, dtype=torch.float32)
                    ).item()
                ),
                **expected,
            }
            if dict(row) != expected_row or raw_delta != expected_row["raw_delta"]:
                raise ParityContractError(
                    "cross-arm loss-term scalar comparison row is inconsistent",
                    code="qwen.parity.loss_term_scalar_row",
                    context={"term_name": term_name, "field": field},
                )
            nested_allclose = nested_allclose and bool(row["allclose"])
    if len(term_names) != len(set(term_names)):
        raise ParityContractError(
            "cross-arm loss-term scalar comparison contains duplicate terms",
            code="qwen.parity.loss_term_scalar_inventory",
            context={"term_names": term_names},
        )
    expected_passed = not missing and not extra and nested_allclose
    if artifact["passed"] is not expected_passed:
        raise ParityContractError(
            "cross-arm loss-term scalar top-level status is inconsistent",
            code="qwen.parity.loss_term_scalar_status",
            context={"expected": expected_passed, "observed": artifact["passed"]},
        )
    canonical_json_bytes(artifact)
    return dict(artifact)


def validate_denominator_comparison_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete literal one-packed/two-separate denominator proof."""

    _expect_exact_keys(
        artifact,
        {
            "passed",
            "semantic_fields",
            "expected_context_counts",
            "packed",
            "separate_shared",
            "plan_normalization",
            "failures",
        },
        owner="denominator_comparison",
    )
    if artifact["semantic_fields"] != list(DENOMINATOR_SEMANTIC_FIELDS):
        raise ParityContractError(
            "denominator semantic-field inventory is invalid",
            code="qwen.parity.denominator_artifact",
            context={"fields": artifact["semantic_fields"]},
        )
    expected_counts = _mapping(
        artifact["expected_context_counts"], "denominator.expected_context_counts"
    )
    _expect_exact_keys(
        expected_counts,
        {"packed", "separate_shared"},
        owner="denominator.expected_context_counts",
    )
    if dict(expected_counts) != {"packed": 1, "separate_shared": 2}:
        raise ParityContractError(
            "denominator comparison arm shape differs from literal Wave 2 scope",
            code="qwen.parity.denominator_arm_shape",
            context=dict(expected_counts),
        )
    failures = artifact["failures"]
    if not isinstance(failures, list) or failures:
        raise ParityContractError(
            "passed denominator comparison must have an empty failure inventory",
            code="qwen.parity.denominator_artifact",
            context={"failures": failures},
        )
    if artifact["passed"] is not True:
        raise ParityContractError(
            "denominator comparison status is not passed",
            code="qwen.parity.denominator_artifact",
            context={"passed": artifact["passed"]},
        )
    packed = _validate_denominator_artifact_mapping(
        artifact["packed"], owner="denominator.packed", expected_context_count=1
    )
    separate = _validate_denominator_artifact_mapping(
        artifact["separate_shared"],
        owner="denominator.separate_shared",
        expected_context_count=2,
    )
    if set(packed) != set(separate):
        raise ParityContractError(
            "denominator term inventories differ",
            code="qwen.parity.denominator_artifact",
            context={"packed": sorted(packed), "separate_shared": sorted(separate)},
        )
    for term_name in sorted(packed):
        for field in DENOMINATOR_SEMANTIC_FIELDS:
            if packed[term_name][field] != separate[term_name][field]:
                raise ParityContractError(
                    "denominator semantic fields differ across arms",
                    code="qwen.parity.denominator_artifact",
                    context={"term_name": term_name, "field": field},
                )
    normalization = _mapping(
        artifact["plan_normalization"], "denominator.plan_normalization"
    )
    _expect_exact_keys(
        normalization,
        {"packed", "separate_shared"},
        owner="denominator.plan_normalization",
    )
    packed_normalization = _validate_plan_normalization_mapping(
        normalization["packed"],
        owner="denominator.plan_normalization.packed",
        denominators=packed,
        expected_context_count=1,
    )
    separate_normalization = _validate_plan_normalization_mapping(
        normalization["separate_shared"],
        owner="denominator.plan_normalization.separate_shared",
        denominators=separate,
        expected_context_count=2,
    )
    for field in (
        "denominator_scope",
        "world_size",
        "rank",
        "backend_gradient_scale",
        "token_type_gate_groups",
    ):
        if packed_normalization[field] != separate_normalization[field]:
            raise ParityContractError(
                "denominator plan normalization differs across arms",
                code="qwen.parity.denominator_artifact",
                context={"field": field},
            )
    for field in sorted(_PLAN_COUNT_FIELDS - {"count/packs"}):
        if (
            packed_normalization["counts"][field]
            != separate_normalization["counts"][field]
        ):
            raise ParityContractError(
                "denominator plan semantic counts differ across arms",
                code="qwen.parity.denominator_artifact",
                context={"field": field},
            )
    canonical_json_bytes(artifact)
    return dict(artifact)


def _validate_parity_plan_v2(
    plan: Mapping[str, Any],
    *,
    allow_authenticated_frozen_parent_without_cuda_runtime: bool = False,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "status",
        "config_identity",
        "repo_identity",
        "dependency_identity",
        "model_identity",
        "model_weight_identity",
        "selection",
        "samples",
        "arms",
        "semantic_atom_inventory",
        "trainable_mechanism",
        "determinism",
        "tolerances",
        "loss_normalization_preflight",
        "source_owners",
        "plan_sha256",
    }
    _expect_exact_keys(plan, expected_keys, owner="plan")
    if plan["schema"] != LEGACY_PARITY_PLAN_SCHEMA_V2 or plan["status"] != PLAN_STATUS:
        raise ParityContractError(
            "parity plan schema or status is unsupported",
            code="qwen.parity.plan_header",
            context={"schema": plan["schema"], "status": plan["status"]},
        )
    validate_dependency_provenance(
        _mapping(plan["dependency_identity"], "dependency_identity"),
        allow_historical_without_cuda_runtime=(
            allow_authenticated_frozen_parent_without_cuda_runtime
        ),
    )
    planned_model_identity = _validate_qwen_component_identity(
        _mapping(plan["model_identity"], "model_identity"),
        owner="model_identity",
        expected_load_model=False,
    )
    _validate_model_free_patch_receipt(
        _only_patch_receipt(
            planned_model_identity["runtime_patches"],
            owner="model_identity",
        )
    )
    validate_model_weight_identity(
        _mapping(plan["model_weight_identity"], "model_weight_identity")
    )
    source_owners = _list_of_mappings(plan["source_owners"], "source_owners")
    source_paths: list[str] = []
    for row in source_owners:
        _expect_exact_keys(row, {"path", "sha256"}, owner="source_owner")
        source_paths.append(_non_empty_text(row["path"], "source_owner.path"))
        _sha256_text(row["sha256"], "source_owner.sha256")
    if len(source_paths) != len(set(source_paths)):
        raise ParityContractError(
            "parity plan source-owner inventory contains duplicates",
            code="qwen.parity.source_owner_duplicate",
            context={"paths": source_paths},
        )
    required_source_owners = {
        "src/qwen/parity.py",
        "scripts/probes/coordexp_swift/wave2_packed_parity.py",
        "src/artifacts/provenance.py",
        "src/artifacts/resources.py",
    }
    missing_source_owners = sorted(required_source_owners - set(source_paths))
    if missing_source_owners:
        raise ParityContractError(
            "parity plan omits a required executed source owner",
            code="qwen.parity.source_owner_missing",
            context={"missing": missing_source_owners},
        )
    selection = _mapping(plan["selection"], "selection")
    _expect_exact_keys(
        selection,
        {"split", "source_indices", "example_ids", "selection_policy"},
        owner="selection",
    )
    indices = _int_list(selection["source_indices"], "selection.source_indices")
    example_ids = _text_list(selection["example_ids"], "selection.example_ids")
    if len(indices) != 2 or len(set(indices)) != 2 or len(example_ids) != 2:
        raise ParityContractError(
            "parity plan requires exactly two distinct source examples",
            code="qwen.parity.selection_pair",
            context={"source_indices": indices, "example_ids": example_ids},
        )
    samples = _list_of_mappings(plan["samples"], "samples")
    if len(samples) != 2:
        raise ParityContractError(
            "parity plan must freeze exactly two samples",
            code="qwen.parity.sample_count",
            context={"count": len(samples)},
        )
    inventory = _list_of_mappings(
        plan["semantic_atom_inventory"], "semantic_atom_inventory"
    )
    if not inventory:
        raise ParityContractError(
            "parity plan requires supervised semantic atoms",
            code="qwen.parity.empty_atoms",
            context={},
        )
    keys = tuple(_key_from_mapping(item) for item in inventory)
    if len(set(keys)) != len(keys):
        raise ParityContractError(
            "parity plan semantic atom keys must be unique",
            code="qwen.parity.duplicate_atom_key",
            context={"atom_count": len(keys)},
        )
    arms = _mapping(plan["arms"], "arms")
    _expect_exact_keys(
        arms,
        {"packed_clean", "separate_reference", "packed_merged_boundary_negative"},
        owner="arms",
    )
    clean = _mapping(arms["packed_clean"], "arms.packed_clean")
    negative = _mapping(
        arms["packed_merged_boundary_negative"],
        "arms.packed_merged_boundary_negative",
    )
    clean_boundaries = _int_list(clean.get("segment_boundaries"), "clean boundaries")
    negative_boundaries = _int_list(
        negative.get("segment_boundaries"), "negative boundaries"
    )
    if len(clean_boundaries) != 3 or len(negative_boundaries) != 2:
        raise ParityContractError(
            "clean and merged-boundary arm shapes are invalid",
            code="qwen.parity.boundary_shape",
            context={
                "clean_boundaries": clean_boundaries,
                "negative_boundaries": negative_boundaries,
            },
        )
    if clean_boundaries == negative_boundaries:
        raise ParityContractError(
            "negative-control boundary must differ from clean boundary",
            code="qwen.parity.negative_boundary_equal",
            context={"segment_boundaries": clean_boundaries},
        )
    tolerances = _mapping(plan["tolerances"], "tolerances")
    expected_tolerances = legacy_v2_tolerances()
    if tolerances != expected_tolerances:
        raise ParityContractError(
            "parity tolerances differ from the frozen Wave 2 contract",
            code="qwen.parity.tolerance_drift",
            context={"expected": expected_tolerances, "observed": tolerances},
        )
    validate_denominator_comparison_artifact(
        _mapping(
            plan["loss_normalization_preflight"],
            "loss_normalization_preflight",
        )
    )
    fingerprint = _sha256_text(plan["plan_sha256"], "plan_sha256")
    body = dict(plan)
    del body["plan_sha256"]
    observed = sha256_json(body)
    if observed != fingerprint:
        raise ParityContractError(
            "parity plan fingerprint does not authenticate the plan body",
            code="qwen.parity.plan_fingerprint",
            context={"expected": fingerprint, "observed": observed},
        )
    canonical_json_bytes(plan)
    return dict(plan)


def validate_parity_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the current v3 plan; historical v2 plans use a separate reader."""

    expected_keys = {
        "schema",
        "status",
        "config_identity",
        "repo_identity",
        "dependency_identity",
        "model_identity",
        "model_weight_identity",
        "parent_v2",
        "selection",
        "samples",
        "arms",
        "semantic_atom_inventory",
        "trainable_mechanism",
        "trainable_inventory_declaration",
        "determinism",
        "tolerances",
        "loss_normalization_preflight",
        "source_owners",
        "plan_sha256",
    }
    _expect_exact_keys(plan, expected_keys, owner="plan")
    if plan["schema"] != PARITY_PLAN_SCHEMA or plan["status"] != PLAN_STATUS:
        raise ParityContractError(
            "parity plan schema or status is unsupported",
            code="qwen.parity.plan_header",
            context={"schema": plan["schema"], "status": plan["status"]},
        )
    parent = validate_parent_v2_identity(_mapping(plan["parent_v2"], "parent_v2"))
    immutable_historical_plan = (
        plan.get("plan_sha256") == _IMMUTABLE_WAVE2_V3_PLAN_SHA256
        and sha256_json(plan) == _IMMUTABLE_WAVE2_V3_PLAN_PAYLOAD_SHA256
    )
    validate_dependency_provenance(
        _mapping(plan["dependency_identity"], "dependency_identity"),
        allow_historical_without_cuda_runtime=immutable_historical_plan,
    )
    planned_model_identity = _validate_qwen_component_identity(
        _mapping(plan["model_identity"], "model_identity"),
        owner="model_identity",
        expected_load_model=False,
    )
    _validate_model_free_patch_receipt(
        _only_patch_receipt(
            planned_model_identity["runtime_patches"], owner="model_identity"
        )
    )
    weight_identity = validate_model_weight_identity(
        _mapping(plan["model_weight_identity"], "model_weight_identity")
    )
    validate_v3_config_identity(_mapping(plan["config_identity"], "config_identity"))
    if weight_identity["aggregate_sha256"] != parent["model_weight_aggregate_sha256"]:
        raise ParityContractError(
            "v3 base-model identity differs from its frozen parent",
            code="qwen.parity.parent_v2_workload",
            context={},
        )
    source_owners = _list_of_mappings(plan["source_owners"], "source_owners")
    source_paths: list[str] = []
    for row in source_owners:
        _expect_exact_keys(row, {"path", "sha256"}, owner="source_owner")
        source_paths.append(_non_empty_text(row["path"], "source_owner.path"))
        _sha256_text(row["sha256"], "source_owner.sha256")
    required_source_owners = {
        "src/qwen/parity.py",
        "scripts/probes/coordexp_swift/wave2_packed_parity.py",
        "src/artifacts/provenance.py",
        "src/artifacts/resources.py",
    }
    if len(source_paths) != len(set(source_paths)) or not required_source_owners <= set(
        source_paths
    ):
        raise ParityContractError(
            "v3 plan source-owner inventory is incomplete or duplicate",
            code="qwen.parity.source_owner_inventory",
            context={
                "missing": sorted(required_source_owners - set(source_paths)),
            },
        )
    selection = _mapping(plan["selection"], "selection")
    _expect_exact_keys(
        selection,
        {"split", "source_indices", "example_ids", "selection_policy"},
        owner="selection",
    )
    if (
        selection["split"] != "train"
        or selection["source_indices"] != parent["source_indices"]
        or selection["example_ids"] != parent["example_ids"]
        or selection["selection_policy"] != "explicit_source_indices_no_switching"
    ):
        raise ParityContractError(
            "v3 selection differs from its frozen parent",
            code="qwen.parity.parent_v2_selection",
            context=dict(selection),
        )
    samples = _list_of_mappings(plan["samples"], "samples")
    if [sample.get("example_id") for sample in samples] != parent["example_ids"]:
        raise ParityContractError(
            "v3 sample identities differ from its frozen parent",
            code="qwen.parity.parent_v2_samples",
            context={},
        )
    atoms = _list_of_mappings(
        plan["semantic_atom_inventory"], "semantic_atom_inventory"
    )
    atom_keys = tuple(_key_from_mapping(row) for row in atoms)
    if (
        not atom_keys
        or len(atom_keys) != len(set(atom_keys))
        or atom_keys != tuple(sorted(atom_keys))
    ):
        raise ParityContractError(
            "v3 semantic atom inventory is empty, duplicate, or unordered",
            code="qwen.parity.semantic_atom_inventory",
            context={"count": len(atom_keys)},
        )
    validate_trainable_inventory_declaration(
        _mapping(
            plan["trainable_inventory_declaration"],
            "trainable_inventory_declaration",
        )
    )
    arms = _mapping(plan["arms"], "arms")
    _expect_exact_keys(
        arms,
        {
            "packed_primary",
            "packed_repeat",
            "separate_reference",
            "packed_merged_boundary_negative",
        },
        owner="arms",
    )
    primary = _mapping(arms["packed_primary"], "arms.packed_primary")
    repeat = _mapping(arms["packed_repeat"], "arms.packed_repeat")
    negative = _mapping(
        arms["packed_merged_boundary_negative"],
        "arms.packed_merged_boundary_negative",
    )
    identity_fields = {
        "example_ids",
        "segment_boundaries",
        "input_ids_sha256",
        "position_ids_sha256",
        "supervision_sha256",
        "image_content_sha256",
    }
    if any(primary.get(field) != repeat.get(field) for field in identity_fields):
        raise ParityContractError(
            "packed repeat is not input-identical to packed primary",
            code="qwen.parity.packed_repeat_identity",
            context={},
        )
    primary_boundaries = _int_list(
        primary.get("segment_boundaries"), "arms.packed_primary.segment_boundaries"
    )
    negative_boundaries = _int_list(
        negative.get("segment_boundaries"),
        "arms.packed_merged_boundary_negative.segment_boundaries",
    )
    if primary_boundaries != [0, 1436, 2822] or negative_boundaries != [0, 2822]:
        raise ParityContractError(
            "v3 clean or negative boundary differs from the frozen parent",
            code="qwen.parity.parent_v2_boundaries",
            context={
                "primary": primary_boundaries,
                "negative": negative_boundaries,
            },
        )
    separate = _mapping(arms["separate_reference"], "arms.separate_reference")
    if (
        separate.get("microstep_count") != 2
        or separate.get("loss_wiring")
        != "two_microsteps_two_immediate_backwards_one_initial_clear"
    ):
        raise ParityContractError(
            "separate reference does not declare production streaming backward",
            code="qwen.parity.separate_cadence",
            context={},
        )
    determinism = _mapping(plan["determinism"], "determinism")
    for field in (
        "seed",
        "torch_manual_seed",
        "cuda_manual_seed_all",
    ):
        if determinism.get(field) != parent["seed"]:
            raise ParityContractError(
                "v3 deterministic seed differs from the frozen parent",
                code="qwen.parity.parent_v2_seed",
                context={"field": field, "value": determinism.get(field)},
            )
    if determinism.get("flash_attention_deterministic") != "1":
        raise ParityContractError(
            "v3 probe plan requires deterministic FlashAttention",
            code="qwen.parity.flash_attention_deterministic",
            context={"observed": determinism.get("flash_attention_deterministic")},
        )
    if _mapping(plan["tolerances"], "tolerances") != frozen_tolerances():
        raise ParityContractError(
            "parity tolerances differ from the frozen Wave 2 v3 contract",
            code="qwen.parity.tolerance_drift",
            context={},
        )
    validate_denominator_comparison_artifact(
        _mapping(plan["loss_normalization_preflight"], "loss_normalization_preflight")
    )
    fingerprint = _sha256_text(plan["plan_sha256"], "plan_sha256")
    body = dict(plan)
    del body["plan_sha256"]
    if sha256_json(body) != fingerprint:
        raise ParityContractError(
            "parity plan fingerprint does not authenticate the plan body",
            code="qwen.parity.plan_fingerprint",
            context={},
        )
    canonical_json_bytes(plan)
    return dict(plan)


def _validate_parity_receipt_v2(
    receipt: Mapping[str, Any],
    *,
    expected_plan_model_identity: Mapping[str, Any] | None = None,
    expected_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _expect_exact_keys(
        receipt,
        {
            "schema",
            "terminal_status",
            "plan_sha256",
            "source_identity",
            "model_identity_attestation",
            "execution",
            "arms",
            "proof",
            "comparisons",
            "negative_discriminator",
            "timings",
            "gpu_memory",
            "measurement",
            "failure",
        },
        owner="receipt",
    )
    if receipt["schema"] != LEGACY_PARITY_RECEIPT_SCHEMA_V2:
        raise ParityContractError(
            "parity receipt schema is unsupported",
            code="qwen.parity.receipt_schema",
            context={"schema": receipt["schema"]},
        )
    status = receipt["terminal_status"]
    if status not in {"passed", "failed"}:
        raise ParityContractError(
            "parity receipt must have a terminal status",
            code="qwen.parity.receipt_status",
            context={"terminal_status": status},
        )
    _sha256_text(receipt["plan_sha256"], "plan_sha256")
    if status == "passed":
        if expected_plan is None:
            raise ParityContractError(
                "passed receipt validation requires its authenticated plan",
                code="qwen.parity.receipt_plan_required",
                context={},
            )
        authenticated_plan = _validate_parity_plan_v2(expected_plan)
        if receipt["plan_sha256"] != authenticated_plan["plan_sha256"]:
            raise ParityContractError(
                "passed receipt is not bound to the authenticated plan",
                code="qwen.parity.receipt_plan_binding",
                context={},
            )
        plan_model_identity = _mapping(
            authenticated_plan["model_identity"], "expected_plan.model_identity"
        )
        if expected_plan_model_identity is not None and dict(
            expected_plan_model_identity
        ) != dict(plan_model_identity):
            raise ParityContractError(
                "separately supplied plan model identity contradicts the plan",
                code="qwen.parity.receipt_plan_binding",
                context={},
            )
        if receipt["failure"] is not None:
            raise ParityContractError(
                "passed receipt cannot contain a failure",
                code="qwen.parity.receipt_failure",
                context={},
            )
        source_identity = _mapping(receipt["source_identity"], "source_identity")
        _expect_exact_keys(
            source_identity,
            {
                "config_identity",
                "repo_identity",
                "dependency_identity",
                "model_identity",
                "model_weight_identity",
                "source_owners",
            },
            owner="source_identity",
        )
        validate_dependency_provenance(
            _mapping(source_identity["dependency_identity"], "dependency_identity")
        )
        validate_model_weight_identity(
            _mapping(source_identity["model_weight_identity"], "model_weight_identity")
        )
        expected_source_projection = {
            "config_identity": authenticated_plan["config_identity"],
            "repo_identity": authenticated_plan["repo_identity"],
            "dependency_identity": authenticated_plan["dependency_identity"],
            "model_weight_identity": authenticated_plan["model_weight_identity"],
            "source_owners": authenticated_plan["source_owners"],
        }
        if {
            key: source_identity[key] for key in expected_source_projection
        } != expected_source_projection:
            raise ParityContractError(
                "passed receipt source identity differs from its authenticated plan",
                code="qwen.parity.receipt_source_binding",
                context={},
            )
        validate_qwen_component_identity_attestation(
            _mapping(
                receipt["model_identity_attestation"],
                "model_identity_attestation",
            ),
            loaded_model_identity=_mapping(
                source_identity["model_identity"], "source_identity.model_identity"
            ),
            expected_plan_model_identity=plan_model_identity,
        )
        _validate_success_execution(_mapping(receipt["execution"], "execution"))
        discriminator = _mapping(
            receipt["negative_discriminator"], "negative_discriminator"
        )
        if discriminator.get("detected") is not True:
            raise ParityContractError(
                "passed receipt requires a detected negative control",
                code="qwen.parity.receipt_negative",
                context={},
            )
        forward_detected_by = discriminator.get("forward_detected_by")
        if (
            not isinstance(forward_detected_by, list)
            or not forward_detected_by
            or not set(forward_detected_by) <= {"supervised_logits", "total_loss"}
            or discriminator.get("gradient_only_is_insufficient") is not True
        ):
            raise ParityContractError(
                "passed receipt requires a forward numerical negative signal",
                code="qwen.parity.receipt_negative_forward",
                context={"forward_detected_by": forward_detected_by},
            )
        comparisons = _mapping(receipt["comparisons"], "comparisons")
        _expect_exact_keys(
            comparisons,
            {
                "semantic_atoms",
                "denominators",
                "supervised_logits",
                "loss",
                "cross_arm_bf16_loss_term_scalars",
                "gradients",
            },
            owner="comparisons",
        )
        validate_denominator_comparison_artifact(
            _mapping(comparisons["denominators"], "comparisons.denominators")
        )
        validate_cross_arm_bf16_loss_term_scalars(
            _mapping(
                comparisons["cross_arm_bf16_loss_term_scalars"],
                "comparisons.cross_arm_bf16_loss_term_scalars",
            )
        )
        for name in (
            "semantic_atoms",
            "denominators",
            "supervised_logits",
            "loss",
            "cross_arm_bf16_loss_term_scalars",
            "gradients",
        ):
            result = _mapping(comparisons.get(name), f"comparisons.{name}")
            if result.get("passed") is not True:
                raise ParityContractError(
                    "passed receipt contains a failed clean comparison",
                    code="qwen.parity.receipt_comparison",
                    context={"comparison": name},
                )
        proof = _mapping(receipt["proof"], "proof")
        if proof.get("status") != "pass":
            raise ParityContractError(
                "passed receipt requires all-layer proof",
                code="qwen.parity.receipt_proof",
                context={"proof_status": proof.get("status")},
            )
        validate_measurement_contract(_mapping(receipt["measurement"], "measurement"))
        _validate_success_timings(_mapping(receipt["timings"], "timings"))
        gpu_memory = _mapping(receipt["gpu_memory"], "gpu_memory")
        _expect_exact_keys(
            gpu_memory,
            {"scope", "peak_allocated_bytes", "peak_reserved_bytes"},
            owner="gpu_memory",
        )
        measurement_gpu = _mapping(
            _mapping(receipt["measurement"], "measurement")["resources"]["gpu"],
            "measurement.resources.gpu",
        )
        if (
            gpu_memory["scope"]
            != "whole_probe_including_model_transfer_and_all_forward_backward_arms"
            or int(gpu_memory["peak_allocated_bytes"])
            != int(measurement_gpu["torch_peak_allocated_bytes"])
            or int(gpu_memory["peak_reserved_bytes"])
            != int(measurement_gpu["torch_peak_reserved_bytes"])
        ):
            raise ParityContractError(
                "Wave 2 GPU memory summary is inconsistent",
                code="qwen.parity.receipt_gpu_memory",
                context=dict(gpu_memory),
            )
    else:
        failure = _mapping(receipt["failure"], "failure")
        _validate_failed_receipt(
            receipt,
            failure=failure,
            expected_plan=expected_plan,
        )
    canonical_json_bytes(receipt)
    return dict(receipt)


def validate_historical_parity_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read an immutable failed v2 receipt without making it a v3 result."""

    if (
        receipt.get("schema") != LEGACY_PARITY_RECEIPT_SCHEMA_V2
        or receipt.get("terminal_status") != "failed"
    ):
        raise ParityContractError(
            "historical reader accepts only immutable failed v2 receipts",
            code="qwen.parity.historical_receipt_scope",
            context={
                "schema": receipt.get("schema"),
                "terminal_status": receipt.get("terminal_status"),
            },
        )
    return _validate_parity_receipt_v2(receipt, expected_plan=expected_plan)


def _validate_v3_marker_reference(
    reference: Mapping[str, Any],
    *,
    expected_plan: Mapping[str, Any],
    expected_receipt_target: str | Path | None = None,
) -> dict[str, Any]:
    _expect_exact_keys(
        reference,
        {
            "path",
            "schema",
            "status",
            "marker_sha256",
            "expected_receipt_target",
        },
        owner="receipt.attempt_marker",
    )
    path = Path(_non_empty_text(reference["path"], "attempt_marker.path"))
    if not path.is_absolute():
        raise ParityContractError(
            "receipt attempt-marker path must be absolute",
            code="qwen.parity.receipt_marker_path",
            context={},
        )
    if (
        reference["schema"] != PARITY_ATTEMPT_MARKER_SCHEMA
        or reference["status"] != "attempt_started"
    ):
        raise ParityContractError(
            "receipt attempt-marker reference header is invalid",
            code="qwen.parity.receipt_marker_header",
            context={},
        )
    marker_sha256 = _sha256_text(
        reference["marker_sha256"], "attempt_marker.marker_sha256"
    )
    referenced_receipt_target = _non_empty_text(
        reference["expected_receipt_target"],
        "attempt_marker.expected_receipt_target",
    )
    if not Path(referenced_receipt_target).is_absolute():
        raise ParityContractError(
            "receipt attempt-marker target reference must be absolute",
            code="qwen.parity.receipt_marker_target",
            context={},
        )
    if expected_receipt_target is not None and referenced_receipt_target != str(
        Path(expected_receipt_target).expanduser().resolve(strict=False)
    ):
        raise ParityContractError(
            "receipt attempt-marker target differs from the audited receipt path",
            code="qwen.parity.receipt_marker_target",
            context={},
        )
    marker = validate_attempt_marker(
        load_strict_json(path),
        expected_plan=expected_plan,
        expected_receipt_target=referenced_receipt_target,
    )
    if marker["marker_sha256"] != marker_sha256:
        raise ParityContractError(
            "receipt attempt-marker reference differs from the immutable marker",
            code="qwen.parity.receipt_marker_binding",
            context={},
        )
    return dict(reference)


def _validate_v3_loss_artifact(
    artifact: Mapping[str, Any],
    *,
    owner: str,
    expected_term_names: Sequence[str] | None = None,
    expected_term_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    total_loss = _finite_number(artifact.get("total_loss"), f"{owner}.total_loss")
    terms = _list_of_mappings(artifact.get("terms"), f"{owner}.terms")
    if not terms:
        raise ParityContractError(
            "v3 finalized loss artifact has no terms",
            code="qwen.parity.arm_loss_terms",
            context={"owner": owner},
        )
    names: list[str] = []
    weighted_sum = 0.0
    for term in terms:
        name = _non_empty_text(term.get("name"), f"{owner}.term.name")
        for field in (
            "raw_loss",
            "weighted_loss",
            "segment_mean_numerator",
            "token_weighted_diagnostic",
        ):
            _finite_number(term.get(field), f"{owner}.{name}.{field}")
        weight = _finite_number(term.get("weight"), f"{owner}.{name}.weight")
        if weight < 0.0 or not math.isclose(
            float(term["weighted_loss"]),
            float(term["raw_loss"]) * weight,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise ParityContractError(
                "v3 finalized loss term weighted value is inconsistent",
                code="qwen.parity.arm_loss_weight",
                context={"owner": owner, "term": name},
            )
        weighted_sum += float(term["weighted_loss"])
        names.append(name)
    if names != list(dict.fromkeys(names)):
        raise ParityContractError(
            "v3 finalized loss artifact contains duplicate terms",
            code="qwen.parity.arm_loss_terms",
            context={"owner": owner},
        )
    if expected_term_names is not None and set(names) != set(expected_term_names):
        raise ParityContractError(
            "v3 finalized loss artifact term inventory differs from its plan",
            code="qwen.parity.arm_loss_terms",
            context={
                "owner": owner,
                "expected": sorted(expected_term_names),
                "observed": sorted(names),
            },
        )
    if expected_term_weights is not None:
        observed_weights = {str(term["name"]): float(term["weight"]) for term in terms}
        expected_weights = {
            str(name): float(weight) for name, weight in expected_term_weights.items()
        }
        if observed_weights != expected_weights:
            raise ParityContractError(
                "v3 finalized loss term weights differ from the authenticated plan",
                code="qwen.parity.arm_loss_weight_binding",
                context={"owner": owner},
            )
    if not math.isclose(
        total_loss,
        weighted_sum,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    ):
        raise ParityContractError(
            "v3 finalized weighted terms do not sum to total loss",
            code="qwen.parity.arm_loss_terms",
            context={"owner": owner},
        )
    return dict(artifact)


def _planned_loss_term_names(plan: Mapping[str, Any]) -> tuple[str, ...]:
    preflight = _mapping(
        plan["loss_normalization_preflight"], "plan.loss_normalization_preflight"
    )
    packed = _mapping(preflight["packed"], "plan.loss_normalization_preflight.packed")
    names = tuple(str(name) for name in packed)
    if not names or len(names) != len(set(names)):
        raise ParityContractError(
            "planned loss term inventory is empty or duplicate",
            code="qwen.parity.arm_loss_terms",
            context={},
        )
    return names


def _planned_loss_term_weights(
    plan: Mapping[str, Any],
) -> dict[str, float] | None:
    """Return only weights exposed authoritatively by the authenticated plan."""

    mechanism = _mapping(plan["trainable_mechanism"], "plan.trainable_mechanism")
    losses = mechanism.get("losses")
    if losses is None:
        return None
    losses_mapping = _mapping(losses, "plan.trainable_mechanism.losses")
    protected = losses_mapping.get("protected")
    if protected is None:
        return None
    protected_mapping = _mapping(protected, "plan.trainable_mechanism.losses.protected")
    planned_names = _planned_loss_term_names(plan)
    weights: dict[str, float] = {}
    for name in planned_names:
        term = protected_mapping.get(name)
        if term is None:
            return None
        term_mapping = _mapping(
            term, f"plan.trainable_mechanism.losses.protected.{name}"
        )
        if "weight" not in term_mapping:
            return None
        weight = _finite_number(
            term_mapping["weight"],
            f"plan.trainable_mechanism.losses.protected.{name}.weight",
        )
        if weight < 0.0:
            raise ParityContractError(
                "authenticated plan exposes a negative loss weight",
                code="qwen.parity.arm_loss_weight_binding",
                context={"term": name},
            )
        weights[name] = weight
    return weights


def _loss_term_comparison_from_serialized_arms(
    left_arm: Mapping[str, Any],
    right_arm: Mapping[str, Any],
) -> dict[str, Any]:
    left_artifact = _validate_v3_loss_artifact(
        _mapping(left_arm.get("loss_artifact"), "left_arm.loss_artifact"),
        owner="left_arm.loss_artifact",
    )
    right_artifact = _validate_v3_loss_artifact(
        _mapping(right_arm.get("loss_artifact"), "right_arm.loss_artifact"),
        owner="right_arm.loss_artifact",
    )
    left_terms = {
        str(term["name"]): term
        for term in _list_of_mappings(left_artifact["terms"], "left_arm.terms")
    }
    right_terms = {
        str(term["name"]): term
        for term in _list_of_mappings(right_artifact["terms"], "right_arm.terms")
    }
    missing = sorted(set(left_terms) - set(right_terms))
    extra = sorted(set(right_terms) - set(left_terms))
    fields = [
        "raw_loss",
        "weighted_loss",
        "segment_mean_numerator",
        "token_weighted_diagnostic",
    ]
    rows: list[dict[str, Any]] = []
    passed = not missing and not extra
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


def _validate_v3_arm(
    arm: Mapping[str, Any],
    *,
    expected_name: str,
    expected_microsteps: int,
    expected_inventory: Mapping[str, Any],
    expected_loss_terms: Sequence[str],
    expected_loss_weights: Mapping[str, float] | None,
    expected_semantic_atom_count: int,
    expected_semantic_atom_keys: Sequence[SemanticAtomKey],
    expected_boundaries: Sequence[Sequence[int]],
    expected_proof: Mapping[str, Any] | None = None,
    forward_only: bool = False,
    bounded_failure: bool = False,
) -> None:
    expected_keys = {
        "name",
        "microstep_count",
        "total_loss_fp32",
        "loss_artifact",
        "semantic_logit_rows",
        "semantic_key_inventory_sha256",
        "gradient_inventory",
        "forward_receipts",
        "forward_logits_dtypes",
        "inner_autocast",
        "backward_cadence",
    }
    if bounded_failure:
        expected_keys.update(
            {
                "gradient_inventory_count",
                "gradient_inventory_omitted",
            }
        )
    _expect_exact_keys(
        arm,
        expected_keys,
        owner=f"arms.{expected_name}",
    )
    if arm["name"] != expected_name or arm["microstep_count"] != expected_microsteps:
        raise ParityContractError(
            "v3 arm name or microstep count is invalid",
            code="qwen.parity.arm_identity",
            context={"expected_name": expected_name},
        )
    total_loss = _finite_number(
        arm["total_loss_fp32"], f"arms.{expected_name}.total_loss"
    )
    loss_artifact = _validate_v3_loss_artifact(
        _mapping(arm["loss_artifact"], f"arms.{expected_name}.loss_artifact"),
        owner=f"arms.{expected_name}.loss_artifact",
        expected_term_names=expected_loss_terms,
        expected_term_weights=expected_loss_weights,
    )
    artifact_total_loss = _finite_number(
        loss_artifact.get("total_loss"),
        f"arms.{expected_name}.loss_artifact.total_loss",
    )
    if not math.isclose(
        total_loss,
        artifact_total_loss,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    ):
        raise ParityContractError(
            "v3 arm total loss contradicts its finalized loss artifact",
            code="qwen.parity.arm_loss_binding",
            context={"name": expected_name},
        )
    semantic_logit_rows = _positive_int(
        arm["semantic_logit_rows"], f"arms.{expected_name}.logit_rows"
    )
    if semantic_logit_rows != expected_semantic_atom_count:
        raise ParityContractError(
            "v3 arm semantic-logit coverage differs from its authenticated plan",
            code="qwen.parity.arm_semantic_logit_coverage",
            context={
                "name": expected_name,
                "expected": expected_semantic_atom_count,
                "observed": semantic_logit_rows,
            },
        )
    observed_semantic_digest = _sha256_text(
        arm["semantic_key_inventory_sha256"],
        f"arms.{expected_name}.semantic_key_inventory_sha256",
    )
    expected_semantic_digest = semantic_atom_key_inventory_sha256(
        tuple(expected_semantic_atom_keys)
    )
    if observed_semantic_digest != expected_semantic_digest:
        raise ParityContractError(
            "v3 arm semantic-logit keys differ from its authenticated plan",
            code="qwen.parity.arm_semantic_logit_key_binding",
            context={"name": expected_name},
        )
    forward_receipts = _list_of_mappings(
        arm["forward_receipts"], f"arms.{expected_name}.forward_receipts"
    )
    forward_logits_dtypes = _text_list_allow_empty(
        arm["forward_logits_dtypes"],
        f"arms.{expected_name}.forward_logits_dtypes",
    )
    autocast_rows = _list_of_mappings(
        arm["inner_autocast"], f"arms.{expected_name}.inner_autocast"
    )
    normalized_boundaries = [
        _int_list(value, f"arms.{expected_name}.expected_boundaries")
        for value in expected_boundaries
    ]
    if (
        len(normalized_boundaries) != expected_microsteps
        or len(forward_receipts) != expected_microsteps
        or forward_logits_dtypes != [FP32_COMPARISON_DTYPE] * expected_microsteps
        or len(autocast_rows) != expected_microsteps
    ):
        raise ParityContractError(
            "v3 arm forward evidence inventory is incomplete",
            code="qwen.parity.arm_forward_evidence",
            context={"name": expected_name},
        )
    for index, (forward_receipt, autocast_row, boundaries) in enumerate(
        zip(
            forward_receipts,
            autocast_rows,
            normalized_boundaries,
            strict=True,
        )
    ):
        fa2 = _mapping(
            forward_receipt.get("fa2_varlen"),
            f"arms.{expected_name}.forward_receipts[{index}].fa2_varlen",
        )
        if (
            fa2.get("segment_boundaries") != boundaries
            or "proof" not in fa2
            or fa2["proof"]
            != (expected_proof if expected_name == "packed_primary" else None)
        ):
            raise ParityContractError(
                "v3 arm forward receipt is not bound to its planned FA2 route",
                code="qwen.parity.arm_forward_evidence",
                context={"name": expected_name, "microstep_index": index},
            )
        _expect_exact_keys(
            autocast_row,
            {
                "module_type",
                "layer_idx",
                "cuda_autocast_enabled",
                "cuda_autocast_dtype",
            },
            owner=f"arms.{expected_name}.inner_autocast[{index}]",
        )
        if (
            autocast_row["module_type"] != "Qwen3VLTextAttention"
            or autocast_row["layer_idx"] != 0
            or autocast_row["cuda_autocast_enabled"] is not True
            or autocast_row["cuda_autocast_dtype"] != "torch.bfloat16"
        ):
            raise ParityContractError(
                "v3 arm did not execute its inner attention under BF16 autocast",
                code="qwen.parity.arm_forward_evidence",
                context={"name": expected_name, "microstep_index": index},
            )
    gradients = _list_of_mappings(
        arm["gradient_inventory"], f"arms.{expected_name}.gradient_inventory"
    )
    concrete_rows = _list_of_mappings(
        expected_inventory["parameters"], "concrete_trainable_inventory.parameters"
    )
    published_gradient_count = len(gradients)
    if bounded_failure:
        total_gradient_count = _non_negative_int(
            arm["gradient_inventory_count"],
            f"arms.{expected_name}.gradient_inventory_count",
        )
        omitted_gradient_count = _non_negative_int(
            arm["gradient_inventory_omitted"],
            f"arms.{expected_name}.gradient_inventory_omitted",
        )
        if (
            published_gradient_count > 64
            or total_gradient_count != published_gradient_count + omitted_gradient_count
        ):
            raise ParityContractError(
                "bounded failure arm gradient accounting is invalid",
                code="qwen.parity.arm_gradient_count",
                context={"name": expected_name},
            )
    else:
        total_gradient_count = published_gradient_count
    if forward_only:
        if gradients or total_gradient_count != 0:
            raise ParityContractError(
                "forward-only negative arm must not publish gradients",
                code="qwen.parity.negative_gradient_scope",
                context={},
            )
    else:
        if total_gradient_count != EXPECTED_TRAINABLE_PARAMETER_COUNT:
            raise ParityContractError(
                "clean arm does not preserve all 589 gradient rows",
                code="qwen.parity.arm_gradient_count",
                context={"name": expected_name, "count": total_gradient_count},
            )
        expected_by_name = {str(row["name"]): row for row in concrete_rows}
        observed_names: list[str] = []
        for row in gradients:
            _expect_exact_keys(
                row,
                {
                    "name",
                    "shape",
                    "parameter_dtype",
                    "parameter_storage_dtype",
                    "gradient_dtype",
                    "gradient_provenance_dtype",
                    "grad_status",
                    "gradient_sha256",
                },
                owner=f"arms.{expected_name}.gradient",
            )
            name = _non_empty_text(row["name"], "arm.gradient.name")
            expected = expected_by_name.get(name)
            if expected is None:
                raise ParityContractError(
                    "clean arm contains an unexpected trainable gradient",
                    code="qwen.parity.arm_gradient_identity",
                    context={"name": name},
                )
            if (
                row["shape"] != expected["shape"]
                or row["parameter_dtype"] != expected["parameter_storage_dtype"]
                or row["parameter_storage_dtype"] != expected["parameter_storage_dtype"]
                or row["gradient_dtype"] != expected["expected_gradient_dtype"]
                or row["gradient_provenance_dtype"]
                != expected["compute_provenance_dtype"]
                or row["grad_status"] != "finite"
            ):
                raise ParityContractError(
                    "clean arm gradient metadata differs from concrete inventory",
                    code="qwen.parity.arm_gradient_identity",
                    context={"name": name},
                )
            _sha256_text(row["gradient_sha256"], "arm.gradient.sha256")
            observed_names.append(name)
        expected_names = sorted(expected_by_name)
        if bounded_failure:
            expected_names = expected_names[:published_gradient_count]
        if observed_names != expected_names:
            raise ParityContractError(
                "clean arm gradient names differ from concrete inventory",
                code="qwen.parity.arm_gradient_identity",
                context={},
            )
    cadence = _mapping(arm["backward_cadence"], f"arms.{expected_name}.cadence")
    _expect_exact_keys(
        cadence,
        {
            "microstep_count",
            "gradient_clear_count",
            "harness_backward_call_count",
            "harness_policy",
            "events",
            "production_backward_call_count",
            "production_policy",
            "cadence_matches_production",
        },
        owner=f"arms.{expected_name}.cadence",
    )
    expected_backward_count = 0 if forward_only else expected_microsteps
    events = _list_of_mappings(cadence["events"], f"arms.{expected_name}.events")
    if (
        cadence["microstep_count"] != expected_microsteps
        or cadence["gradient_clear_count"] != 1
        or cadence["harness_backward_call_count"] != expected_backward_count
        or cadence["production_backward_call_count"] != expected_backward_count
        or len(events) != expected_backward_count
        or cadence["cadence_matches_production"] is not True
        or cadence["harness_policy"]
        != (
            "forward_only_negative_control"
            if forward_only
            else "one_immediate_accelerator_backward_per_microstep"
        )
        or cadence["production_policy"]
        != (
            "not_applicable_forward_only_negative"
            if forward_only
            else "one_runtime_backward_per_microstep_with_accumulation_context"
        )
    ):
        raise ParityContractError(
            "v3 arm backward cadence is invalid",
            code="qwen.parity.arm_cadence",
            context={"name": expected_name},
        )
    for index, event in enumerate(events):
        _expect_exact_keys(
            event,
            {
                "microstep_index",
                "forward_ordinal",
                "loss_ordinal",
                "backward_ordinal",
                "immediate_after_loss",
                "sync_gradients",
                "accumulation_context",
            },
            owner=f"arms.{expected_name}.events[{index}]",
        )
        expected_sync = index == expected_microsteps - 1
        expected_context = "sync_gradients" if expected_sync else "accelerator.no_sync"
        if (
            event.get("microstep_index") != index
            or event.get("forward_ordinal") != index + 1
            or event.get("loss_ordinal") != index + 1
            or event.get("backward_ordinal") != index + 1
            or event.get("immediate_after_loss") is not True
            or event.get("sync_gradients") is not expected_sync
            or event.get("accumulation_context") != expected_context
        ):
            raise ParityContractError(
                "v3 arm backward event order is invalid",
                code="qwen.parity.arm_cadence_event",
                context={"name": expected_name, "index": index},
            )


def _validate_v3_negative_discriminator(
    value: Mapping[str, Any],
    *,
    expected_semantic_atom_count: int,
    expected_semantic_atom_keys: Sequence[SemanticAtomKey],
    require_detected: bool = True,
) -> None:
    _expect_exact_keys(
        value,
        {
            "detected",
            "boundary_changed",
            "clean_boundaries",
            "negative_boundaries",
            "forward_detected_by",
            "diagnostic_detected_by",
            "gradient_only_is_insufficient",
            "attestation",
            "supervised_logits",
            "total_loss",
            "gradients",
        },
        owner="negative_discriminator_v3",
    )
    clean = _int_list(value["clean_boundaries"], "negative.clean_boundaries")
    negative = _int_list(value["negative_boundaries"], "negative.negative_boundaries")
    logits = _mapping(value["supervised_logits"], "negative.supervised_logits")
    total_loss = _mapping(value["total_loss"], "negative.total_loss")
    _validate_keyed_logits_comparison_artifact(
        logits,
        expected_row_count=expected_semantic_atom_count,
        expected_keys=expected_semantic_atom_keys,
    )
    _validate_tensor_comparison_artifact(total_loss, owner="negative.total_loss")
    expected_forward = [
        name
        for name, allclose in (
            ("supervised_logits", bool(logits["passed"])),
            ("total_loss", bool(total_loss["allclose"])),
        )
        if not allclose
    ]
    expected_detected = bool(expected_forward)
    gradients = _mapping(value["gradients"], "negative.gradients")
    _expect_exact_keys(
        gradients, {"status", "acceptance_metric"}, owner="negative.gradients"
    )
    attestation = _mapping(value["attestation"], "negative.attestation")
    _expect_exact_keys(
        attestation,
        {
            "status",
            "expected_clean_boundaries",
            "observed_negative_boundaries",
            "boundary_mismatch_detected",
            "proof_disabled",
            "executed_negative_varlen_receipt",
        },
        owner="negative.attestation",
    )
    executed = _mapping(
        attestation["executed_negative_varlen_receipt"],
        "negative.executed_negative_varlen_receipt",
    )
    if (
        clean != [0, 1436, 2822]
        or negative != [0, 2822]
        or value["boundary_changed"] is not True
        or value["forward_detected_by"] != expected_forward
        or value["detected"] is not expected_detected
        or (require_detected and not expected_detected)
        or (not require_detected and expected_detected)
        or value["gradient_only_is_insufficient"] is not True
        or value["diagnostic_detected_by"] != []
        or gradients != {"status": "not_executed", "acceptance_metric": False}
        or attestation["status"] != "rejected_against_frozen_clean_boundary"
        or attestation["expected_clean_boundaries"] != clean
        or attestation["observed_negative_boundaries"] != negative
        or attestation["boundary_mismatch_detected"] is not True
        or attestation["proof_disabled"] is not True
        or executed.get("segment_boundaries") != negative
        or executed.get("proof") is not None
    ):
        raise ParityContractError(
            "v3 boundary-only negative evidence is inconsistent",
            code="qwen.parity.negative_attestation",
            context={},
        )


def validate_parity_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_plan_model_identity: Mapping[str, Any] | None = None,
    expected_plan: Mapping[str, Any] | None = None,
    expected_receipt_target: str | Path | None = None,
) -> dict[str, Any]:
    """Validate only the decision-bearing Wave 2 v3 receipt schema."""

    _expect_exact_keys(
        receipt,
        {
            "schema",
            "terminal_status",
            "plan_sha256",
            "parent_v2",
            "attempt_marker",
            "trainable_inventory",
            "source_identity",
            "model_identity_attestation",
            "execution",
            "arms",
            "proof",
            "comparisons",
            "negative_discriminator",
            "timings",
            "gpu_memory",
            "measurement",
            "failure",
        },
        owner="receipt",
    )
    if receipt["schema"] != PARITY_RECEIPT_SCHEMA:
        raise ParityContractError(
            "parity receipt schema is unsupported",
            code="qwen.parity.receipt_schema",
            context={"schema": receipt["schema"]},
        )
    status = receipt["terminal_status"]
    if status not in TERMINAL_STATUSES:
        raise ParityContractError(
            "parity receipt must have a v3 terminal status",
            code="qwen.parity.receipt_status",
            context={"terminal_status": status},
        )
    validate_parent_v2_identity(_mapping(receipt["parent_v2"], "receipt.parent_v2"))
    plan_sha256 = _sha256_text(receipt["plan_sha256"], "receipt.plan_sha256")
    plan: dict[str, Any] | None = None
    if expected_plan is not None:
        plan = validate_parity_plan(expected_plan)
        if plan_sha256 != plan["plan_sha256"]:
            raise ParityContractError(
                "v3 receipt is not bound to its authenticated plan",
                code="qwen.parity.receipt_plan_binding",
                context={},
            )
        if receipt["parent_v2"] != plan["parent_v2"]:
            raise ParityContractError(
                "v3 receipt parent identity differs from its plan",
                code="qwen.parity.receipt_parent_binding",
                context={},
            )
    elif status != "failed" or plan_sha256 != "0" * 64:
        raise ParityContractError(
            "post-plan v3 receipt validation requires its authenticated plan",
            code="qwen.parity.receipt_plan_required",
            context={},
        )
    marker_reference = _mapping(receipt["attempt_marker"], "receipt.attempt_marker")
    inventory_root = _mapping(
        receipt["trainable_inventory"], "receipt.trainable_inventory"
    )
    arms = _mapping(receipt["arms"], "receipt.arms")
    comparisons = _mapping(receipt["comparisons"], "receipt.comparisons")
    failure = receipt["failure"]
    if status == "passed":
        if plan is None or failure is not None:
            raise ParityContractError(
                "passed v3 receipt requires its plan and no failure",
                code="qwen.parity.receipt_failure",
                context={},
            )
        _validate_v3_marker_reference(
            marker_reference,
            expected_plan=plan,
            expected_receipt_target=expected_receipt_target,
        )
        source_identity = _mapping(receipt["source_identity"], "source_identity")
        _expect_exact_keys(
            source_identity,
            {
                "config_identity",
                "repo_identity",
                "dependency_identity",
                "model_identity",
                "model_weight_identity",
                "source_owners",
            },
            owner="source_identity",
        )
        expected_source_projection = {
            "config_identity": plan["config_identity"],
            "repo_identity": plan["repo_identity"],
            "dependency_identity": plan["dependency_identity"],
            "model_weight_identity": plan["model_weight_identity"],
            "source_owners": plan["source_owners"],
        }
        if {
            key: source_identity[key] for key in expected_source_projection
        } != expected_source_projection:
            raise ParityContractError(
                "passed receipt source identity differs from its authenticated plan",
                code="qwen.parity.receipt_source_binding",
                context={},
            )
        plan_model_identity = _mapping(
            plan["model_identity"], "expected_plan.model_identity"
        )
        if expected_plan_model_identity is not None and dict(
            expected_plan_model_identity
        ) != dict(plan_model_identity):
            raise ParityContractError(
                "separately supplied model identity contradicts the plan",
                code="qwen.parity.receipt_plan_binding",
                context={},
            )
        validate_qwen_component_identity_attestation(
            _mapping(
                receipt["model_identity_attestation"],
                "model_identity_attestation",
            ),
            loaded_model_identity=_mapping(
                source_identity["model_identity"], "source_identity.model_identity"
            ),
            expected_plan_model_identity=plan_model_identity,
        )
        _expect_exact_keys(inventory_root, {"concrete"}, owner="trainable_inventory")
        concrete = validate_concrete_trainable_inventory(
            _mapping(inventory_root["concrete"], "trainable_inventory.concrete"),
            declaration=_mapping(
                plan["trainable_inventory_declaration"],
                "plan.trainable_inventory_declaration",
            ),
        )
        expected_semantic_atom_keys = tuple(
            _key_from_mapping(row)
            for row in _list_of_mappings(
                plan["semantic_atom_inventory"], "plan.semantic_atom_inventory"
            )
        )
        expected_semantic_atom_count = len(expected_semantic_atom_keys)
        expected_loss_terms = _planned_loss_term_names(plan)
        expected_loss_weights = _planned_loss_term_weights(plan)
        marker_payload = validate_attempt_marker(
            load_strict_json(marker_reference["path"]), expected_plan=plan
        )
        if (
            source_identity != marker_payload["source_identity"]
            or concrete != marker_payload["concrete_trainable_inventory"]
        ):
            raise ParityContractError(
                "passed receipt differs from its immutable attempt marker",
                code="qwen.parity.receipt_marker_binding",
                context={},
            )
        _expect_exact_keys(
            arms,
            {
                "packed_primary",
                "packed_repeat",
                "separate_reference",
                "packed_merged_boundary_negative",
            },
            owner="receipt.arms",
        )
        _validate_v3_arm(
            _mapping(arms["packed_primary"], "arms.packed_primary"),
            expected_name="packed_primary",
            expected_microsteps=1,
            expected_inventory=concrete,
            expected_loss_terms=expected_loss_terms,
            expected_loss_weights=expected_loss_weights,
            expected_semantic_atom_count=expected_semantic_atom_count,
            expected_semantic_atom_keys=expected_semantic_atom_keys,
            expected_boundaries=([0, 1436, 2822],),
            expected_proof=_mapping(receipt["proof"], "receipt.proof"),
        )
        _validate_v3_arm(
            _mapping(arms["packed_repeat"], "arms.packed_repeat"),
            expected_name="packed_repeat",
            expected_microsteps=1,
            expected_inventory=concrete,
            expected_loss_terms=expected_loss_terms,
            expected_loss_weights=expected_loss_weights,
            expected_semantic_atom_count=expected_semantic_atom_count,
            expected_semantic_atom_keys=expected_semantic_atom_keys,
            expected_boundaries=([0, 1436, 2822],),
        )
        _validate_v3_arm(
            _mapping(arms["separate_reference"], "arms.separate_reference"),
            expected_name="separate_reference",
            expected_microsteps=2,
            expected_inventory=concrete,
            expected_loss_terms=expected_loss_terms,
            expected_loss_weights=expected_loss_weights,
            expected_semantic_atom_count=expected_semantic_atom_count,
            expected_semantic_atom_keys=expected_semantic_atom_keys,
            expected_boundaries=([0, 1436], [0, 1386]),
        )
        _validate_v3_arm(
            _mapping(
                arms["packed_merged_boundary_negative"],
                "arms.packed_merged_boundary_negative",
            ),
            expected_name="packed_merged_boundary_negative",
            expected_microsteps=1,
            expected_inventory=concrete,
            expected_loss_terms=expected_loss_terms,
            expected_loss_weights=expected_loss_weights,
            expected_semantic_atom_count=expected_semantic_atom_count,
            expected_semantic_atom_keys=expected_semantic_atom_keys,
            expected_boundaries=([0, 2822],),
            forward_only=True,
        )
        _expect_exact_keys(
            comparisons,
            {
                "semantic_atoms",
                "denominators",
                "packed_primary_vs_separate",
                "packed_repeat_vs_separate",
                "packed_repeat_measurability",
            },
            owner="receipt.comparisons",
        )
        for comparison_name in (
            "packed_primary_vs_separate",
            "packed_repeat_vs_separate",
        ):
            comparison = _mapping(
                comparisons[comparison_name], f"comparisons.{comparison_name}"
            )
            _expect_exact_keys(
                comparison,
                {
                    "supervised_logits",
                    "loss",
                    "cross_arm_bf16_loss_term_scalars",
                    "gradients",
                },
                owner=f"comparisons.{comparison_name}",
            )
            logits = _mapping(
                comparison["supervised_logits"],
                f"{comparison_name}.supervised_logits",
            )
            loss = _mapping(comparison["loss"], f"{comparison_name}.loss")
            _validate_keyed_logits_comparison_artifact(
                logits,
                expected_row_count=expected_semantic_atom_count,
                expected_keys=expected_semantic_atom_keys,
            )
            _validate_total_loss_comparison_artifact(loss)
            left_arm_name = (
                "packed_primary"
                if comparison_name == "packed_primary_vs_separate"
                else "packed_repeat"
            )
            if dict(loss) != _total_loss_comparison_from_serialized_arms(
                _mapping(arms[left_arm_name], f"arms.{left_arm_name}"),
                _mapping(arms["separate_reference"], "arms.separate_reference"),
            ):
                raise ParityContractError(
                    "published loss comparison is not derived from its arm totals",
                    code="qwen.parity.receipt_loss_binding",
                    context={"comparison": comparison_name},
                )
            if logits["passed"] is not True or loss["passed"] is not True:
                raise ParityContractError(
                    "passed receipt contains a failed forward comparison",
                    code="qwen.parity.receipt_comparison",
                    context={"comparison": comparison_name},
                )
            loss_terms = _mapping(
                comparison["cross_arm_bf16_loss_term_scalars"],
                f"{comparison_name}.loss_terms",
            )
            validated_loss_terms = validate_cross_arm_bf16_loss_term_scalars(loss_terms)
            if validated_loss_terms != _loss_term_comparison_from_serialized_arms(
                _mapping(arms[left_arm_name], f"arms.{left_arm_name}"),
                _mapping(arms["separate_reference"], "arms.separate_reference"),
            ):
                raise ParityContractError(
                    "published loss-term comparison is not derived from its arms",
                    code="qwen.parity.receipt_loss_term_binding",
                    context={"comparison": comparison_name},
                )
            if validated_loss_terms["passed"] is not True:
                raise ParityContractError(
                    "passed receipt contains failed per-term scalars",
                    code="qwen.parity.receipt_comparison",
                    context={"comparison": comparison_name},
                )
            if (
                validate_gradient_comparison_artifact(
                    _mapping(comparison["gradients"], f"{comparison_name}.gradients"),
                    expected_inventory=concrete,
                )["passed"]
                is not True
            ):
                raise ParityContractError(
                    "passed receipt contains failed gradients",
                    code="qwen.parity.receipt_comparison",
                    context={"comparison": comparison_name},
                )
        validate_denominator_comparison_artifact(
            _mapping(comparisons["denominators"], "comparisons.denominators")
        )
        _validate_semantic_atom_comparison_artifact(
            _mapping(comparisons["semantic_atoms"], "comparisons.semantic_atoms"),
            expected_count=expected_semantic_atom_count,
            expected_keys=expected_semantic_atom_keys,
        )
        repeat = validate_packed_gradient_repeat_artifact(
            _mapping(
                comparisons["packed_repeat_measurability"],
                "comparisons.packed_repeat_measurability",
            ),
            expected_inventory=concrete,
        )
        if repeat["passed"] is not True:
            raise ParityContractError(
                "passed receipt contains an unmeasurable packed repeat",
                code="qwen.parity.receipt_repeat",
                context={},
            )
        proof = _mapping(receipt["proof"], "receipt.proof")
        _validate_fa2_proof_artifact(proof)
        planned_primary_boundaries = _int_list(
            _mapping(plan["arms"], "plan.arms")["packed_primary"]["segment_boundaries"],
            "plan.arms.packed_primary.segment_boundaries",
        )
        if proof["segment_boundaries"] != planned_primary_boundaries:
            raise ParityContractError(
                "all-layer proof boundaries differ from the authenticated plan",
                code="qwen.parity.receipt_proof",
                context={},
            )
        _validate_v3_negative_discriminator(
            _mapping(receipt["negative_discriminator"], "negative_discriminator"),
            expected_semantic_atom_count=expected_semantic_atom_count,
            expected_semantic_atom_keys=expected_semantic_atom_keys,
        )
        execution = _mapping(receipt["execution"], "execution")
        _validate_success_execution(execution, v3=True)
        _validate_v3_execution_arm_evidence(execution, arms=arms)
        _validate_success_timings(_mapping(receipt["timings"], "timings"), v3=True)
        measurement = validate_measurement_contract(
            _mapping(receipt["measurement"], "measurement"), v3=True
        )
        gpu_memory = _mapping(receipt["gpu_memory"], "gpu_memory")
        _expect_exact_keys(
            gpu_memory,
            {"scope", "peak_allocated_bytes", "peak_reserved_bytes"},
            owner="gpu_memory",
        )
        measurement_gpu = _mapping(
            _mapping(measurement["resources"], "measurement.resources")["gpu"],
            "measurement.resources.gpu",
        )
        if (
            gpu_memory["scope"]
            != "whole_probe_including_model_transfer_and_all_forward_backward_arms"
            or int(gpu_memory["peak_allocated_bytes"])
            != int(measurement_gpu["torch_peak_allocated_bytes"])
            or int(gpu_memory["peak_reserved_bytes"])
            != int(measurement_gpu["torch_peak_reserved_bytes"])
        ):
            raise ParityContractError(
                "Wave 2 v3 GPU memory summary is inconsistent",
                code="qwen.parity.receipt_gpu_memory",
                context=dict(gpu_memory),
            )
    else:
        failure_mapping = _mapping(failure, "receipt.failure")
        _non_empty_text(failure_mapping.get("type"), "failure.type")
        _non_empty_text(failure_mapping.get("message"), "failure.message")
        evidence = _mapping(failure_mapping.get("evidence"), "failure.evidence")
        _expect_exact_keys(
            evidence,
            {"schema", "stage_reached", "completed_phases", "completed_fields"},
            owner="failure.evidence",
        )
        stage = _non_empty_text(evidence["stage_reached"], "failure.stage_reached")
        if (
            evidence["schema"] != PARITY_FAILURE_EVIDENCE_SCHEMA
            or stage not in _V3_FAILURE_STAGES
        ):
            raise ParityContractError(
                "v3 failure phase-state evidence is invalid",
                code="qwen.parity.failure_evidence",
                context={"stage": stage},
            )
        completed_fields = _text_list_allow_empty(
            evidence["completed_fields"], "completed_fields"
        )
        observed_fields = [
            field
            for field in _V3_FAILURE_FIELD_ORDER
            if bool(
                receipt[field]
                if field not in {"attempt_marker", "trainable_inventory"}
                else (marker_reference if field == "attempt_marker" else inventory_root)
            )
        ]
        if completed_fields != observed_fields:
            raise ParityContractError(
                "v3 failure completed-field inventory contradicts retained evidence",
                code="qwen.parity.failure_completed_fields",
                context={
                    "expected": observed_fields,
                    "observed": completed_fields,
                },
            )
        completed_phases = _text_list_allow_empty(
            evidence["completed_phases"], "completed_phases"
        )
        expected_phases = list(
            _V3_FAILURE_PHASES[: _V3_FAILURE_STAGE_PHASE_COUNT[stage]]
        )
        if completed_phases != expected_phases:
            raise ParityContractError(
                "v3 failure completed phases do not match the reached stage",
                code="qwen.parity.failure_phase_prefix",
                context={
                    "stage": stage,
                    "expected": expected_phases,
                    "observed": completed_phases,
                },
            )
        required_fields = set(_V3_FAILURE_STAGE_REQUIRED_FIELDS[stage])
        allowed_fields = set(required_fields)
        if _V3_FAILURE_STAGES.index(stage) >= _V3_FAILURE_STAGES.index(
            "cuda_preflight"
        ):
            allowed_fields.update({"gpu_memory", "measurement"})
        observed_field_set = set(observed_fields)
        if (
            not required_fields <= observed_field_set
            or not observed_field_set <= allowed_fields
            or (
                "gpu_memory" in observed_field_set
                and "measurement" not in observed_field_set
            )
        ):
            raise ParityContractError(
                "v3 failure completed fields do not match the reached stage",
                code="qwen.parity.failure_completed_fields",
                context={
                    "stage": stage,
                    "required": sorted(required_fields),
                    "allowed": sorted(allowed_fields),
                    "observed": observed_fields,
                },
            )
        source_identity = _mapping(receipt["source_identity"], "source_identity")
        if source_identity:
            _expect_exact_keys(
                source_identity,
                {
                    "config_identity",
                    "repo_identity",
                    "dependency_identity",
                    "model_identity",
                    "model_weight_identity",
                    "source_owners",
                },
                owner="source_identity",
            )
        if plan is not None and source_identity:
            plan_owned = {
                "config_identity": plan["config_identity"],
                "repo_identity": plan["repo_identity"],
                "dependency_identity": plan["dependency_identity"],
                "model_weight_identity": plan["model_weight_identity"],
                "source_owners": plan["source_owners"],
            }
            if any(
                source_identity.get(key) != value for key, value in plan_owned.items()
            ):
                raise ParityContractError(
                    "failed receipt source identity differs from its authenticated plan",
                    code="qwen.parity.receipt_source_binding",
                    context={},
                )
        stage_index = _V3_FAILURE_STAGES.index(stage)
        marker_stage_index = _V3_FAILURE_STAGES.index("attempt_started")
        model_stage_index = _V3_FAILURE_STAGES.index("model_setup")
        concrete: dict[str, Any] | None = None
        if stage_index >= marker_stage_index:
            if plan is None:
                raise ParityContractError(
                    "post-marker v3 failure requires its authenticated plan",
                    code="qwen.parity.failure_plan_required",
                    context={},
                )
            _validate_v3_marker_reference(
                marker_reference,
                expected_plan=plan,
                expected_receipt_target=expected_receipt_target,
            )
            _expect_exact_keys(
                inventory_root, {"concrete"}, owner="trainable_inventory"
            )
            concrete = validate_concrete_trainable_inventory(
                _mapping(inventory_root["concrete"], "trainable_inventory.concrete")
            )
            marker_payload = validate_attempt_marker(
                load_strict_json(marker_reference["path"]), expected_plan=plan
            )
            if (
                source_identity != marker_payload["source_identity"]
                or concrete != marker_payload["concrete_trainable_inventory"]
            ):
                raise ParityContractError(
                    "post-marker failure evidence differs from the immutable marker",
                    code="qwen.parity.failure_marker_binding",
                    context={},
                )
            if not {"attempt_marker", "trainable_inventory"} <= set(completed_fields):
                raise ParityContractError(
                    "post-marker failure omits marker or concrete inventory evidence",
                    code="qwen.parity.failure_marker_evidence",
                    context={},
                )
        elif marker_reference or inventory_root:
            raise ParityContractError(
                "pre-marker failure cannot claim marker or concrete inventory",
                code="qwen.parity.failure_marker_stage",
                context={},
            )
        if plan is not None and source_identity and stage_index < marker_stage_index:
            if source_identity["model_identity"] != plan["model_identity"]:
                raise ParityContractError(
                    "pre-marker failure model identity differs from its plan",
                    code="qwen.parity.receipt_source_binding",
                    context={},
                )
        if stage_index >= model_stage_index:
            if plan is None:
                raise ParityContractError(
                    "post-model failure requires its authenticated plan",
                    code="qwen.parity.failure_plan_required",
                    context={},
                )
            validate_qwen_component_identity_attestation(
                _mapping(
                    receipt["model_identity_attestation"],
                    "model_identity_attestation",
                ),
                loaded_model_identity=_mapping(
                    source_identity["model_identity"],
                    "source_identity.model_identity",
                ),
                expected_plan_model_identity=_mapping(
                    plan["model_identity"], "expected_plan.model_identity"
                ),
            )
        _validate_rich_failure_execution(
            receipt,
            stage=stage,
            v3=True,
            expected_plan=plan,
        )
        expected_arm_names = _V3_FAILURE_STAGE_ARM_NAMES.get(stage, ())
        if set(arms) != set(expected_arm_names):
            raise ParityContractError(
                "v3 failure arm inventory does not match the reached stage",
                code="qwen.parity.failure_arms",
                context={
                    "stage": stage,
                    "expected": list(expected_arm_names),
                    "observed": sorted(arms),
                },
            )
        proof = _mapping(receipt["proof"], "receipt.proof")
        if bool(proof) != bool(expected_arm_names):
            raise ParityContractError(
                "v3 failure proof presence does not match the reached stage",
                code="qwen.parity.failure_arms",
                context={"stage": stage},
            )
        if expected_arm_names:
            if concrete is None or proof.get("status") != "pass":
                raise ParityContractError(
                    "v3 failure arm evidence lacks concrete inventory or proof",
                    code="qwen.parity.failure_arms",
                    context={"stage": stage},
                )
            _validate_fa2_proof_artifact(proof)
            if (
                plan is None
                or proof["segment_boundaries"]
                != _mapping(plan["arms"], "plan.arms")["packed_primary"][
                    "segment_boundaries"
                ]
            ):
                raise ParityContractError(
                    "v3 failure proof boundaries differ from the authenticated plan",
                    code="qwen.parity.failure_proof",
                    context={},
                )
            arm_contract = {
                "packed_primary": (1, False),
                "packed_repeat": (1, False),
                "separate_reference": (2, False),
                "packed_merged_boundary_negative": (1, True),
            }
            expected_semantic_atom_keys = tuple(
                _key_from_mapping(row)
                for row in _list_of_mappings(
                    plan["semantic_atom_inventory"], "plan.semantic_atom_inventory"
                )
            )
            expected_semantic_atom_count = len(expected_semantic_atom_keys)
            expected_loss_terms = _planned_loss_term_names(plan)
            expected_loss_weights = _planned_loss_term_weights(plan)
            for arm_name in expected_arm_names:
                microsteps, forward_only = arm_contract[arm_name]
                _validate_v3_arm(
                    _mapping(arms[arm_name], f"arms.{arm_name}"),
                    expected_name=arm_name,
                    expected_microsteps=microsteps,
                    expected_inventory=concrete,
                    expected_loss_terms=expected_loss_terms,
                    expected_loss_weights=expected_loss_weights,
                    expected_semantic_atom_count=expected_semantic_atom_count,
                    expected_semantic_atom_keys=expected_semantic_atom_keys,
                    expected_boundaries={
                        "packed_primary": ([0, 1436, 2822],),
                        "packed_repeat": ([0, 1436, 2822],),
                        "separate_reference": ([0, 1436], [0, 1386]),
                        "packed_merged_boundary_negative": ([0, 2822],),
                    }[arm_name],
                    expected_proof=proof if arm_name == "packed_primary" else None,
                    forward_only=forward_only,
                    bounded_failure=True,
                )
        if stage_index >= _V3_FAILURE_STAGES.index("comparisons"):
            _expect_exact_keys(
                comparisons,
                {
                    "semantic_atoms",
                    "denominators",
                    "packed_primary_vs_separate",
                    "packed_repeat_vs_separate",
                    "packed_repeat_measurability",
                },
                owner="receipt.comparisons",
            )
            for comparison_name in (
                "packed_primary_vs_separate",
                "packed_repeat_vs_separate",
            ):
                comparison = _mapping(
                    comparisons[comparison_name], f"comparisons.{comparison_name}"
                )
                _expect_exact_keys(
                    comparison,
                    {
                        "supervised_logits",
                        "loss",
                        "cross_arm_bf16_loss_term_scalars",
                        "gradients",
                    },
                    owner=f"comparisons.{comparison_name}",
                )
                _validate_keyed_logits_comparison_artifact(
                    _mapping(
                        comparison["supervised_logits"],
                        f"{comparison_name}.supervised_logits",
                    ),
                    expected_row_count=expected_semantic_atom_count,
                    expected_keys=expected_semantic_atom_keys,
                )
                loss = _mapping(comparison["loss"], f"{comparison_name}.loss")
                _validate_total_loss_comparison_artifact(loss)
                left_arm_name = (
                    "packed_primary"
                    if comparison_name == "packed_primary_vs_separate"
                    else "packed_repeat"
                )
                if dict(loss) != _total_loss_comparison_from_serialized_arms(
                    _mapping(arms[left_arm_name], f"arms.{left_arm_name}"),
                    _mapping(arms["separate_reference"], "arms.separate_reference"),
                ):
                    raise ParityContractError(
                        "failure loss comparison is not derived from its arm totals",
                        code="qwen.parity.failure_loss_binding",
                        context={"comparison": comparison_name},
                    )
                validate_gradient_comparison_artifact(
                    _mapping(comparison["gradients"], f"{comparison_name}.gradients"),
                    expected_inventory=concrete,
                )
                loss_terms = _mapping(
                    comparison["cross_arm_bf16_loss_term_scalars"],
                    f"{comparison_name}.loss_terms",
                )
                if validate_cross_arm_bf16_loss_term_scalars(
                    loss_terms
                ) != _loss_term_comparison_from_serialized_arms(
                    _mapping(arms[left_arm_name], f"arms.{left_arm_name}"),
                    _mapping(arms["separate_reference"], "arms.separate_reference"),
                ):
                    raise ParityContractError(
                        "failure loss-term comparison is not derived from its arms",
                        code="qwen.parity.failure_loss_term_binding",
                        context={"comparison": comparison_name},
                    )
            validate_denominator_comparison_artifact(
                _mapping(comparisons["denominators"], "comparisons.denominators")
            )
            _validate_semantic_atom_comparison_artifact(
                _mapping(comparisons["semantic_atoms"], "comparisons.semantic_atoms"),
                expected_count=expected_semantic_atom_count,
                expected_keys=expected_semantic_atom_keys,
            )
            validate_packed_gradient_repeat_artifact(
                _mapping(
                    comparisons["packed_repeat_measurability"],
                    "comparisons.packed_repeat_measurability",
                ),
                expected_inventory=concrete,
            )
            _validate_v3_negative_discriminator(
                _mapping(receipt["negative_discriminator"], "negative_discriminator"),
                expected_semantic_atom_count=expected_semantic_atom_count,
                expected_semantic_atom_keys=expected_semantic_atom_keys,
                require_detected=failure_mapping.get("code")
                != "qwen.parity.negative_undetected",
            )
        _validate_rich_failure_timing_and_resources(
            receipt,
            stage=stage,
            completed_phases=completed_phases,
            v3=True,
        )
        if status == "unmeasurable":
            repeat = validate_packed_gradient_repeat_artifact(
                _mapping(
                    comparisons.get("packed_repeat_measurability"),
                    "comparisons.packed_repeat_measurability",
                ),
                expected_inventory=concrete,
            )
            if repeat["status"] != "unmeasurable":
                raise ParityContractError(
                    "unmeasurable receipt lacks an unmeasurable repeat gate",
                    code="qwen.parity.receipt_unmeasurable",
                    context={},
                )
    canonical_json_bytes(receipt)
    return dict(receipt)


def _validate_failed_receipt(
    receipt: Mapping[str, Any],
    *,
    failure: Mapping[str, Any],
    expected_plan: Mapping[str, Any] | None,
) -> None:
    """Validate archived minimal v2 failures or strict rich v1 evidence."""

    _non_empty_text(failure.get("type"), "failure.type")
    message = _non_empty_text(failure.get("message"), "failure.message")
    if len(message) > 1000:
        raise ParityContractError(
            "failure message exceeds its bounded receipt allowance",
            code="qwen.parity.failure_message_bound",
            context={"length": len(message), "maximum": 1000},
        )
    if "code" in failure:
        code = _non_empty_text(failure["code"], "failure.code")
        if len(code) > 160:
            raise ParityContractError(
                "failure code exceeds its bounded receipt allowance",
                code="qwen.parity.failure_code_bound",
                context={"length": len(code), "maximum": 160},
            )
    evidence = failure.get("evidence")
    if evidence is None:
        _validate_archived_minimal_failure(receipt, failure=failure)
        return
    _validate_rich_failure_evidence(
        receipt,
        failure=failure,
        evidence=_mapping(evidence, "failure.evidence"),
        expected_plan=expected_plan,
    )


def _validate_archived_minimal_failure(
    receipt: Mapping[str, Any], *, failure: Mapping[str, Any]
) -> None:
    """Keep the actual pre-remediation v2 failure shape readable."""

    allowed_failure_keys = {"type", "message", "code"}
    if not set(failure) <= allowed_failure_keys:
        raise ParityContractError(
            "archived minimal failure contains unsupported fields",
            code="qwen.parity.failure_minimal_fields",
            context={"fields": sorted(str(key) for key in failure)},
        )
    for field in (
        "model_identity_attestation",
        "arms",
        "proof",
        "comparisons",
        "negative_discriminator",
        "timings",
        "gpu_memory",
        "measurement",
    ):
        if _mapping(receipt[field], field):
            raise ParityContractError(
                "archived minimal failure cannot claim completed evidence",
                code="qwen.parity.failure_minimal_completed",
                context={"field": field},
            )
    execution = _mapping(receipt["execution"], "execution")
    if set(execution) != {"requested_device"}:
        raise ParityContractError(
            "archived minimal failure execution shape is invalid",
            code="qwen.parity.failure_minimal_execution",
            context={"fields": sorted(str(key) for key in execution)},
        )
    _non_empty_text(execution["requested_device"], "execution.requested_device")
    source_identity = _mapping(receipt["source_identity"], "source_identity")
    if source_identity:
        _expect_exact_keys(
            source_identity,
            {
                "config_identity",
                "repo_identity",
                "dependency_identity",
                "model_identity",
                "model_weight_identity",
                "source_owners",
            },
            owner="source_identity",
        )


def _validate_rich_failure_evidence(
    receipt: Mapping[str, Any],
    *,
    failure: Mapping[str, Any],
    evidence: Mapping[str, Any],
    expected_plan: Mapping[str, Any] | None,
) -> None:
    if set(failure) != (
        {"type", "message", "evidence"} | ({"code"} if "code" in failure else set())
    ):
        raise ParityContractError(
            "rich failure envelope fields are invalid",
            code="qwen.parity.failure_rich_fields",
            context={"fields": sorted(str(key) for key in failure)},
        )
    _expect_exact_keys(
        evidence,
        {"schema", "stage_reached", "completed_phases", "completed_fields"},
        owner="failure.evidence",
    )
    if evidence["schema"] != LEGACY_PARITY_FAILURE_EVIDENCE_SCHEMA_V1:
        raise ParityContractError(
            "rich failure evidence schema is unsupported",
            code="qwen.parity.failure_evidence_schema",
            context={"schema": evidence["schema"]},
        )
    stage = _non_empty_text(evidence["stage_reached"], "failure.evidence.stage_reached")
    if stage not in _FAILURE_STAGES:
        raise ParityContractError(
            "rich failure stage is unsupported",
            code="qwen.parity.failure_stage",
            context={"stage_reached": stage},
        )
    authenticated_plan: Mapping[str, Any] | None = None
    if stage not in {"initialized", "receipt_target_preflight"}:
        if expected_plan is None:
            raise ParityContractError(
                "post-plan rich failure validation requires its authenticated plan",
                code="qwen.parity.failure_plan_required",
                context={"stage_reached": stage},
            )
        authenticated_plan = _validate_parity_plan_v2(expected_plan)
        if receipt["plan_sha256"] != authenticated_plan["plan_sha256"]:
            raise ParityContractError(
                "rich failure receipt is not bound to the authenticated plan",
                code="qwen.parity.failure_plan_binding",
                context={"stage_reached": stage},
            )
    expected_phases = list(_FAILURE_PHASES[: _FAILURE_STAGE_PHASE_COUNT[stage]])
    completed_phases = evidence["completed_phases"]
    if completed_phases != expected_phases:
        raise ParityContractError(
            "rich failure completed phases do not match the reached stage",
            code="qwen.parity.failure_phases",
            context={
                "stage_reached": stage,
                "expected": expected_phases,
                "observed": completed_phases,
            },
        )
    completed_fields = evidence["completed_fields"]
    observed_completed_fields = [
        field
        for field in _FAILURE_EVIDENCE_FIELDS
        if bool(_mapping(receipt[field], field))
    ]
    base_fields = list(_FAILURE_STAGE_BASE_FIELDS[stage])
    allowed_field_sets = [base_fields]
    if stage in {"cuda_preflight", "normalization_preflight", "accelerator_ready"}:
        allowed_field_sets.append(
            [
                field
                for field in _FAILURE_EVIDENCE_FIELDS
                if field in {*base_fields, "gpu_memory", "measurement"}
            ]
        )
    elif _FAILURE_STAGE_PHASE_COUNT[stage] > 0:
        allowed_field_sets.append(
            [
                field
                for field in _FAILURE_EVIDENCE_FIELDS
                if field in {*base_fields, "gpu_memory"}
            ]
        )
    if (
        completed_fields != observed_completed_fields
        or completed_fields not in allowed_field_sets
    ):
        raise ParityContractError(
            "rich failure completed fields do not match its stage contract",
            code="qwen.parity.failure_completed_fields",
            context={
                "declared": completed_fields,
                "observed": observed_completed_fields,
                "allowed": allowed_field_sets,
            },
        )
    _validate_rich_failure_source_identity(
        receipt,
        stage=stage,
        expected_plan=authenticated_plan,
    )
    _validate_rich_failure_execution(receipt, stage=stage)
    _validate_rich_failure_arms_and_proof(receipt, stage=stage)
    _validate_rich_failure_timing_and_resources(
        receipt,
        stage=stage,
        completed_phases=completed_phases,
    )
    comparisons = _mapping(receipt["comparisons"], "comparisons")
    if comparisons:
        _expect_exact_keys(
            comparisons,
            {
                "semantic_atoms",
                "denominators",
                "supervised_logits",
                "loss",
                "cross_arm_bf16_loss_term_scalars",
                "gradients",
            },
            owner="comparisons",
        )
        _validate_semantic_atom_comparison_artifact(
            _mapping(comparisons["semantic_atoms"], "comparisons.semantic_atoms")
        )
        _validate_keyed_logits_comparison_artifact(
            _mapping(
                comparisons["supervised_logits"],
                "comparisons.supervised_logits",
            )
        )
        _validate_total_loss_comparison_artifact(
            _mapping(comparisons["loss"], "comparisons.loss")
        )
        validate_denominator_comparison_artifact(
            _mapping(comparisons["denominators"], "comparisons.denominators")
        )
        validate_cross_arm_bf16_loss_term_scalars(
            _mapping(
                comparisons["cross_arm_bf16_loss_term_scalars"],
                "comparisons.cross_arm_bf16_loss_term_scalars",
            )
        )
        _validate_gradient_comparison_artifact(
            _mapping(comparisons["gradients"], "comparisons.gradients")
        )
        _validate_negative_discriminator_artifact(
            _mapping(receipt["negative_discriminator"], "negative_discriminator"),
            arms=_mapping(receipt["arms"], "arms"),
        )
        if failure.get("code") == "qwen.parity.clean_failed" and all(
            bool(_mapping(comparisons[name], name).get("passed"))
            for name in comparisons
        ):
            raise ParityContractError(
                "clean-failed receipt contains no failed clean comparison",
                code="qwen.parity.failure_comparison",
                context={},
            )
    evidence_payload = {
        field: receipt[field]
        for field in _FAILURE_EVIDENCE_FIELDS
        if bool(_mapping(receipt[field], field))
    }
    evidence_bytes = len(canonical_json_bytes(evidence_payload))
    if evidence_bytes > MAX_FAILURE_EVIDENCE_BYTES:
        raise ParityContractError(
            "rich failure evidence exceeds its serialized byte bound",
            code="qwen.parity.failure_evidence_bound",
            context={
                "observed_bytes": evidence_bytes,
                "maximum": MAX_FAILURE_EVIDENCE_BYTES,
            },
        )


def _validate_rich_failure_source_identity(
    receipt: Mapping[str, Any],
    *,
    stage: str,
    expected_plan: Mapping[str, Any] | None,
) -> None:
    source_identity = _mapping(receipt["source_identity"], "source_identity")
    attestation = _mapping(
        receipt["model_identity_attestation"], "model_identity_attestation"
    )
    if not source_identity:
        if stage not in {"initialized", "receipt_target_preflight"} or attestation:
            raise ParityContractError(
                "rich failure source identity is absent after plan loading",
                code="qwen.parity.failure_source_identity",
                context={"stage_reached": stage},
            )
        return
    _expect_exact_keys(
        source_identity,
        {
            "config_identity",
            "repo_identity",
            "dependency_identity",
            "model_identity",
            "model_weight_identity",
            "source_owners",
        },
        owner="source_identity",
    )
    if expected_plan is None:
        raise ParityContractError(
            "rich failure source identity lacks an authenticated plan binding",
            code="qwen.parity.failure_plan_required",
            context={"stage_reached": stage},
        )
    _mapping(source_identity["config_identity"], "source_identity.config_identity")
    _mapping(source_identity["repo_identity"], "source_identity.repo_identity")
    validate_dependency_provenance(
        _mapping(
            source_identity["dependency_identity"],
            "source_identity.dependency_identity",
        )
    )
    validate_model_weight_identity(
        _mapping(
            source_identity["model_weight_identity"],
            "source_identity.model_weight_identity",
        )
    )
    loaded = bool(attestation)
    model_identity = _validate_qwen_component_identity(
        _mapping(source_identity["model_identity"], "source_identity.model_identity"),
        owner="source_identity.model_identity",
        expected_load_model=loaded,
    )
    source_owners = _list_of_mappings(
        source_identity["source_owners"], "source_identity.source_owners"
    )
    owner_paths: list[str] = []
    for row in source_owners:
        _expect_exact_keys(row, {"path", "sha256"}, owner="source_owner")
        owner_paths.append(_non_empty_text(row["path"], "source_owner.path"))
        _sha256_text(row["sha256"], "source_owner.sha256")
    if len(owner_paths) != len(set(owner_paths)):
        raise ParityContractError(
            "rich failure source owner inventory contains duplicates",
            code="qwen.parity.failure_source_identity",
            context={},
        )
    expected_source_projection = {
        "config_identity": expected_plan["config_identity"],
        "repo_identity": expected_plan["repo_identity"],
        "dependency_identity": expected_plan["dependency_identity"],
        "model_weight_identity": expected_plan["model_weight_identity"],
        "source_owners": expected_plan["source_owners"],
    }
    observed_source_projection = {
        key: source_identity[key] for key in expected_source_projection
    }
    if observed_source_projection != expected_source_projection:
        raise ParityContractError(
            "rich failure source identity differs from its authenticated plan",
            code="qwen.parity.failure_source_binding",
            context={},
        )
    if loaded:
        validate_qwen_component_identity_attestation(
            attestation,
            loaded_model_identity=model_identity,
            expected_plan_model_identity=_mapping(
                expected_plan["model_identity"], "expected_plan.model_identity"
            ),
        )
    elif model_identity != expected_plan["model_identity"]:
        raise ParityContractError(
            "pre-load model identity differs from its authenticated plan",
            code="qwen.parity.failure_source_binding",
            context={},
        )


def _is_authenticated_immutable_wave2_v3_failure_receipt(
    receipt: Mapping[str, Any], *, expected_plan: Mapping[str, Any] | None
) -> bool:
    if expected_plan is None:
        return False
    failure = _mapping(receipt["failure"], "failure")
    evidence = _mapping(failure["evidence"], "failure.evidence")
    source_identity = _mapping(receipt["source_identity"], "source_identity")
    config_identity = _mapping(
        source_identity["config_identity"], "source_identity.config_identity"
    )
    return (
        expected_plan.get("schema") == PARITY_PLAN_SCHEMA
        and expected_plan.get("plan_sha256") == _IMMUTABLE_WAVE2_V3_PLAN_SHA256
        and sha256_json(expected_plan) == _IMMUTABLE_WAVE2_V3_PLAN_PAYLOAD_SHA256
        and receipt.get("schema") == PARITY_RECEIPT_SCHEMA
        and receipt.get("terminal_status") == "failed"
        and receipt.get("plan_sha256") == _IMMUTABLE_WAVE2_V3_PLAN_SHA256
        and failure.get("code") == "qwen.parity.clean_failed"
        and evidence.get("schema") == PARITY_FAILURE_EVIDENCE_SCHEMA
        and evidence.get("stage_reached") == "comparisons"
        and config_identity.get("fingerprint")
        == HISTORICAL_V3_RUNTIME_CONFIG_FINGERPRINT
        and sha256_json(receipt) == _IMMUTABLE_WAVE2_V3_FAILURE_RECEIPT_PAYLOAD_SHA256
    )


def _validate_rich_failure_execution(
    receipt: Mapping[str, Any],
    *,
    stage: str,
    v3: bool = False,
    expected_plan: Mapping[str, Any] | None = None,
) -> None:
    execution = _mapping(receipt["execution"], "execution")
    if stage in {"comparisons", "finalization"}:
        if not v3:
            _validate_success_execution(execution)
            return
        success_keys = {
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
        if set(execution) == success_keys and stage == "comparisons":
            if _is_authenticated_immutable_wave2_v3_failure_receipt(
                receipt, expected_plan=expected_plan
            ):
                _validate_success_execution(execution, v3=True)
                return
        _expect_exact_keys(
            execution,
            {*success_keys, "requested_device", "gpu_idle_preflight"},
            owner="execution",
        )
        completed_execution = {key: execution[key] for key in success_keys}
        _validate_success_execution(completed_execution, v3=True)
        requested_device = _non_empty_text(
            execution["requested_device"], "execution.requested_device"
        )
        if requested_device != completed_execution["device"]:
            raise ParityContractError(
                "completed failure execution changed its requested CUDA device",
                code="qwen.parity.failure_execution",
                context={
                    "requested": requested_device,
                    "device": completed_execution["device"],
                },
            )
        _validate_completed_gpu_idle_preflight(
            _mapping(execution["gpu_idle_preflight"], "execution.gpu_idle_preflight"),
            expected_device=requested_device,
        )
        return
    if stage in {
        "initialized",
        "receipt_target_preflight",
        "plan_loaded",
        "plan_revalidated",
    }:
        _expect_exact_keys(execution, {"requested_device"}, owner="execution")
        _non_empty_text(execution["requested_device"], "execution.requested_device")
        return
    base_keys = {"requested_device", "device", "gpu_idle_preflight"}
    pre_model_stages = {"cuda_preflight", "normalization_preflight"}
    if v3:
        pre_model_stages.add("attempt_started")
    if stage in pre_model_stages:
        _expect_exact_keys(execution, base_keys, owner="execution")
    elif stage == "accelerator_ready":
        _expect_exact_keys(execution, {*base_keys, "accelerator"}, owner="execution")
    else:
        _expect_exact_keys(
            execution,
            {
                *base_keys,
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
            },
            owner="execution",
        )
        if (
            execution["model_dtype"] != "torch.bfloat16"
            or execution["train_mode"] is not True
            or execution["use_cache"] is not False
            or execution["optimizer"] is not None
        ):
            raise ParityContractError(
                "rich failure model execution identity is invalid",
                code="qwen.parity.failure_execution",
                context={"stage_reached": stage},
            )
        for field in (
            "memory_savers",
            "adapter",
            "special_token_embeddings",
            "prepared_model_attestation",
        ):
            _mapping(execution[field], f"execution.{field}")
        if not isinstance(execution["trainable_value_identity_before"], list):
            raise ParityContractError(
                "rich failure trainable identity is invalid",
                code="qwen.parity.failure_execution",
                context={},
            )
    requested = _non_empty_text(
        execution["requested_device"], "execution.requested_device"
    )
    device = _non_empty_text(execution["device"], "execution.device")
    if requested != device or not device.startswith("cuda:"):
        raise ParityContractError(
            "rich failure requested and resolved devices differ",
            code="qwen.parity.failure_execution",
            context={"requested": requested, "device": device},
        )
    idle = _mapping(execution["gpu_idle_preflight"], "execution.gpu_idle_preflight")
    if idle.get("status") != "passed" or idle.get("requested_device") != device:
        raise ParityContractError(
            "rich failure GPU preflight identity is invalid",
            code="qwen.parity.failure_execution",
            context={},
        )
    if "accelerator" in execution:
        accelerator = _mapping(execution["accelerator"], "execution.accelerator")
        _expect_exact_keys(
            accelerator,
            {
                "distributed_type",
                "rank",
                "local_rank",
                "world_size",
                "device",
                "cuda_current_device",
                "mixed_precision",
                "native_amp",
                "gradient_accumulation_steps",
                "scaler",
                "accelerate_torch_device",
            },
            owner="execution.accelerator",
        )
        try:
            expected_cuda_index = int(device.rsplit(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ParityContractError(
                "rich failure CUDA device identity is invalid",
                code="qwen.parity.failure_execution",
                context={"device": device},
                cause=exc,
            ) from exc
        required = {
            "distributed_type": "NO",
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": device,
            "mixed_precision": "bf16",
            "native_amp": True,
            "gradient_accumulation_steps": 1,
            "scaler": None,
            "accelerate_torch_device": device,
            "cuda_current_device": expected_cuda_index,
        }
        if any(accelerator.get(key) != value for key, value in required.items()):
            raise ParityContractError(
                "rich failure Accelerator identity is invalid",
                code="qwen.parity.failure_execution",
                context={},
            )


def _validate_rich_failure_arms_and_proof(
    receipt: Mapping[str, Any], *, stage: str
) -> None:
    arms = _mapping(receipt["arms"], "arms")
    proof = _mapping(receipt["proof"], "proof")
    expected_names: list[str] = []
    if _FAILURE_STAGE_PHASE_COUNT[stage] >= 5:
        expected_names.append("packed_clean")
    if _FAILURE_STAGE_PHASE_COUNT[stage] >= 6:
        expected_names.append("separate_reference")
    if _FAILURE_STAGE_PHASE_COUNT[stage] >= 7:
        expected_names.append("packed_merged_boundary_negative")
    proof_presence_is_valid = bool(proof) == bool(expected_names)
    if set(arms) != set(expected_names) or not proof_presence_is_valid:
        raise ParityContractError(
            "rich failure arm/proof inventory does not match its stage",
            code="qwen.parity.failure_arms",
            context={"expected": expected_names, "observed": list(arms)},
        )
    if not expected_names:
        return
    _validate_fa2_proof_artifact(proof)
    expected_microsteps = {
        "packed_clean": 1,
        "separate_reference": 2,
        "packed_merged_boundary_negative": 1,
    }
    for name in expected_names:
        _validate_failure_arm_artifact(
            _mapping(arms[name], f"arms.{name}"),
            expected_name=name,
            expected_microsteps=expected_microsteps[name],
            root_proof=proof,
        )


def _validate_failure_arm_artifact(
    arm: Mapping[str, Any],
    *,
    expected_name: str,
    expected_microsteps: int,
    root_proof: Mapping[str, Any],
) -> None:
    _expect_exact_keys(
        arm,
        {
            "name",
            "microstep_count",
            "total_loss_fp32",
            "loss_artifact",
            "semantic_logit_rows",
            "gradient_inventory",
            "forward_receipts",
            "forward_logits_dtypes",
            "inner_autocast",
            "backward_cadence",
            "gradient_inventory_count",
            "gradient_inventory_omitted",
        },
        owner=f"arms.{expected_name}",
    )
    if (
        arm["name"] != expected_name
        or arm["microstep_count"] != expected_microsteps
        or not math.isfinite(float(arm["total_loss_fp32"]))
    ):
        raise ParityContractError(
            "rich failure arm identity is invalid",
            code="qwen.parity.failure_arms",
            context={"arm": expected_name},
        )
    _mapping(arm["loss_artifact"], f"arms.{expected_name}.loss_artifact")
    _non_negative_int(
        arm["semantic_logit_rows"], f"arms.{expected_name}.semantic_logit_rows"
    )
    inventory = _list_of_mappings(
        arm["gradient_inventory"], f"arms.{expected_name}.gradient_inventory"
    )
    inventory_count = _positive_int(
        arm["gradient_inventory_count"],
        f"arms.{expected_name}.gradient_inventory_count",
    )
    omitted = _non_negative_int(
        arm["gradient_inventory_omitted"],
        f"arms.{expected_name}.gradient_inventory_omitted",
    )
    if inventory_count != len(inventory) + omitted or len(inventory) > 64:
        raise ParityContractError(
            "rich failure arm gradient inventory accounting is invalid",
            code="qwen.parity.failure_arms",
            context={"arm": expected_name},
        )
    inventory_names: list[str] = []
    for row in inventory:
        _expect_exact_keys(
            row,
            {
                "name",
                "shape",
                "parameter_dtype",
                "parameter_storage_dtype",
                "gradient_dtype",
                "gradient_provenance_dtype",
                "grad_status",
                "gradient_sha256",
            },
            owner=f"arms.{expected_name}.gradient",
        )
        inventory_names.append(_non_empty_text(row["name"], "gradient.name"))
        if row["parameter_storage_dtype"] != row["parameter_dtype"]:
            raise ParityContractError(
                "rich failure arm parameter dtype is inconsistent",
                code="qwen.parity.failure_arms",
                context={"arm": expected_name},
            )
        if row["grad_status"] == "finite":
            _sha256_text(row["gradient_sha256"], "gradient.gradient_sha256")
        elif row["grad_status"] != "none" or row["gradient_sha256"] is not None:
            raise ParityContractError(
                "rich failure arm gradient status is invalid",
                code="qwen.parity.failure_arms",
                context={"arm": expected_name},
            )
    if len(inventory_names) != len(set(inventory_names)):
        raise ParityContractError(
            "rich failure arm gradient names contain duplicates",
            code="qwen.parity.failure_arms",
            context={"arm": expected_name},
        )
    forward_receipts = _list_of_mappings(
        arm["forward_receipts"], f"arms.{expected_name}.forward_receipts"
    )
    dtypes = arm["forward_logits_dtypes"]
    autocast = arm["inner_autocast"]
    if (
        len(forward_receipts) != expected_microsteps
        or dtypes != ["torch.float32"] * expected_microsteps
        or not isinstance(autocast, list)
        or len(autocast) != expected_microsteps
    ):
        raise ParityContractError(
            "rich failure arm forward inventory is invalid",
            code="qwen.parity.failure_arms",
            context={"arm": expected_name},
        )
    for index, forward in enumerate(forward_receipts):
        fa2 = _mapping(forward.get("fa2_varlen"), "forward.fa2_varlen")
        observed_proof = fa2.get("proof")
        expected_proof = (
            root_proof if expected_name == "packed_clean" and index == 0 else None
        )
        if observed_proof != expected_proof:
            raise ParityContractError(
                "rich failure arm proof binding is invalid",
                code="qwen.parity.failure_arms",
                context={"arm": expected_name, "index": index},
            )
    cadence = _mapping(
        arm["backward_cadence"], f"arms.{expected_name}.backward_cadence"
    )
    _expect_exact_keys(
        cadence,
        {
            "microstep_count",
            "harness_backward_call_count",
            "harness_policy",
            "production_backward_call_count",
            "production_policy",
            "cadence_matches_production",
        },
        owner="backward_cadence",
    )
    if (
        cadence["microstep_count"] != expected_microsteps
        or cadence["harness_backward_call_count"] != 1
        or cadence["production_backward_call_count"] != expected_microsteps
        or cadence["cadence_matches_production"] is not (expected_microsteps == 1)
    ):
        raise ParityContractError(
            "rich failure backward cadence is inconsistent",
            code="qwen.parity.failure_arms",
            context={"arm": expected_name},
        )


def _validate_fa2_proof_artifact(proof: Mapping[str, Any]) -> None:
    _expect_exact_keys(
        proof,
        {
            "status",
            "observed_branch",
            "segment_boundaries",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
            "resolved_attention_implementation",
            "model_dtype",
            "branch_evidence_from_explicit_varlen_kwargs",
            "flash_fn_called",
            "flash_varlen_fn_called",
            "pad_fn_called",
            "unpad_fn_called",
            "observed_call",
            "topology",
            "expected_text_layer_count",
            "observed_text_layer_count",
            "text_layer_events",
            "vision_attention_events",
            "unrelated_attention_events",
        },
        owner="proof",
    )
    boundaries = _int_list(proof["segment_boundaries"], "proof.segment_boundaries")
    expected_max_length = (
        max(right - left for left, right in zip(boundaries, boundaries[1:]))
        if len(boundaries) >= 2
        else 0
    )
    text_events = proof["text_layer_events"]
    topology = _mapping(proof["topology"], "proof.topology")
    _expect_exact_keys(
        topology,
        {
            "topology_id",
            "text_model_name",
            "text_model_class",
            "configured_text_layer_count",
            "text_layers",
        },
        owner="proof.topology",
    )
    topology_id = _non_empty_text(topology["topology_id"], "proof.topology_id")
    _non_empty_text(topology["text_model_name"], "proof.text_model_name")
    _non_empty_text(topology["text_model_class"], "proof.text_model_class")
    configured_layer_count = _positive_int(
        topology["configured_text_layer_count"],
        "proof.configured_text_layer_count",
    )
    text_layers = topology["text_layers"]
    if (
        proof["status"] != "pass"
        or proof["observed_branch"] != "padding_free_varlen"
        or proof["resolved_attention_implementation"] != "flash_attention_2"
        or proof["model_dtype"] != "torch.bfloat16"
        or proof["branch_evidence_from_explicit_varlen_kwargs"] is not True
        or proof["flash_fn_called"] is not False
        or proof["flash_varlen_fn_called"] is not True
        or proof["pad_fn_called"] is not False
        or proof["unpad_fn_called"] is not False
        or len(boundaries) < 2
        or boundaries[0] != 0
        or any(right <= left for left, right in zip(boundaries, boundaries[1:]))
        or proof["max_length_q"] != expected_max_length
        or proof["max_length_k"] != expected_max_length
        or proof["cu_seq_lens_q"] != boundaries
        or proof["cu_seq_lens_k"] != boundaries
        or not isinstance(text_layers, list)
        or not isinstance(text_events, list)
        or configured_layer_count != len(text_layers)
        or proof["expected_text_layer_count"] != len(text_layers)
        or proof["observed_text_layer_count"] != len(text_events)
        or len(text_events) != len(text_layers)
        or not text_events
        or len(text_events) > 256
    ):
        raise ParityContractError(
            "rich failure all-layer proof is inconsistent",
            code="qwen.parity.failure_proof",
            context={},
        )
    expected_layers: list[dict[str, Any]] = []
    for expected_index, layer_value in enumerate(text_layers):
        layer = _mapping(layer_value, "proof.topology.text_layer")
        _expect_exact_keys(
            layer,
            {
                "topology_id",
                "identity",
                "module_name",
                "module_class",
                "layer_idx",
                "configured_backend",
            },
            owner="proof.topology.text_layer",
        )
        module_name = _non_empty_text(
            layer["module_name"], "proof.topology.text_layer.module_name"
        )
        layer_idx = _non_negative_int(
            layer["layer_idx"], "proof.topology.text_layer.layer_idx"
        )
        if (
            layer_idx != expected_index
            or layer["topology_id"] != topology_id
            or layer["identity"] != f"{module_name}#{layer_idx}"
            or layer["configured_backend"] != "flash_attention_2"
        ):
            raise ParityContractError(
                "rich failure expected text-layer topology is invalid",
                code="qwen.parity.failure_proof",
                context={"layer_idx": layer_idx},
            )
        _non_empty_text(layer["module_class"], "proof.topology.text_layer.module_class")
        expected_layers.append(dict(layer))
    if [layer["layer_idx"] for layer in expected_layers] != list(
        range(configured_layer_count)
    ):
        raise ParityContractError(
            "rich failure expected text-layer indices are not contiguous",
            code="qwen.parity.failure_proof",
            context={},
        )
    observed_identities: list[str] = []
    for expected, event in zip(expected_layers, text_events, strict=True):
        row = _mapping(event, "proof.text_layer_event")
        if (
            row.get("kind") != "text"
            or row.get("topology_id") != expected["topology_id"]
            or row.get("identity") != expected["identity"]
            or row.get("module_name") != expected["module_name"]
            or row.get("module_class") != expected["module_class"]
            or row.get("layer_idx") != expected["layer_idx"]
            or row.get("configured_backend") != expected["configured_backend"]
            or row.get("registry_key") != "flash_attention_2"
            or row.get("registry_had_local_override") is not False
            or row.get("completion_status") != "completed"
            or row.get("attention_mask") is not None
            or row.get("cu_seq_lens_q") != boundaries
            or row.get("cu_seq_lens_k") != boundaries
            or row.get("max_length_q") != proof["max_length_q"]
            or row.get("max_length_k") != proof["max_length_k"]
            or row.get("flash_fn_call_count") != 0
            or row.get("flash_varlen_fn_call_count") != 1
            or row.get("pad_fn_call_count") != 0
            or row.get("unpad_fn_call_count") != 0
            or not isinstance(row.get("varlen_calls"), list)
            or len(row["varlen_calls"]) != 1
        ):
            raise ParityContractError(
                "rich failure text-layer proof event is invalid",
                code="qwen.parity.failure_proof",
                context={},
            )
        observed_identities.append(str(row["identity"]))
    if observed_identities != [str(layer["identity"]) for layer in expected_layers]:
        raise ParityContractError(
            "rich failure text-layer proof identities are invalid",
            code="qwen.parity.failure_proof",
            context={},
        )
    if proof["observed_call"] != text_events[0]["varlen_calls"][0]:
        raise ParityContractError(
            "rich failure observed FA2 call differs from its first text-layer event",
            code="qwen.parity.failure_proof",
            context={},
        )
    if proof["unrelated_attention_events"] != []:
        raise ParityContractError(
            "rich failure proof contains unrelated attention events",
            code="qwen.parity.failure_proof",
            context={},
        )


def _validate_rich_failure_timing_and_resources(
    receipt: Mapping[str, Any],
    *,
    stage: str,
    completed_phases: Sequence[str],
    v3: bool = False,
) -> None:
    timings = _mapping(receipt["timings"], "timings")
    phase_count = (
        _V3_FAILURE_STAGE_PHASE_COUNT[stage]
        if v3
        else _FAILURE_STAGE_PHASE_COUNT[stage]
    )
    if phase_count < 5:
        if timings:
            raise ParityContractError(
                "rich failure timing exists before packed-clean completion",
                code="qwen.parity.failure_timings",
                context={},
            )
    else:
        _expect_exact_keys(
            timings,
            {
                "clock",
                "scope",
                "proof_on_forward_ns",
                "proof_off_forward_ns",
                "proof_overhead_ns",
            },
            owner="timings",
        )
        proof_on = _positive_int(timings["proof_on_forward_ns"], "proof_on")
        proof_off = _positive_int(timings["proof_off_forward_ns"], "proof_off")
        if (
            timings["clock"] != "time.perf_counter_ns_with_cuda_synchronize"
            or timings["scope"]
            != "forward_only_same_packed_inputs_train_mode_no_backward"
            or timings["proof_overhead_ns"] != proof_on - proof_off
        ):
            raise ParityContractError(
                "rich failure proof timing is inconsistent",
                code="qwen.parity.failure_timings",
                context={},
            )
    gpu_memory = _mapping(receipt["gpu_memory"], "gpu_memory")
    measurement = _mapping(receipt["measurement"], "measurement")
    if gpu_memory:
        _expect_exact_keys(
            gpu_memory,
            {"scope", "peak_allocated_bytes", "peak_reserved_bytes"},
            owner="gpu_memory",
        )
        if gpu_memory["scope"] != "whole_probe_until_failure":
            raise ParityContractError(
                "rich failure GPU memory scope is invalid",
                code="qwen.parity.failure_resources",
                context={},
            )
        _non_negative_int(gpu_memory["peak_allocated_bytes"], "peak_allocated")
        _non_negative_int(gpu_memory["peak_reserved_bytes"], "peak_reserved")
    if measurement:
        allowed_keys = {
            "schema",
            "completed_phases",
            "phase_boundary_samples",
            "failure_resource_summary",
        }
        if set(measurement) not in (
            {"schema", "completed_phases", "phase_boundary_samples"},
            allowed_keys,
        ):
            raise ParityContractError(
                "rich failure measurement fields are invalid",
                code="qwen.parity.failure_resources",
                context={"fields": sorted(str(key) for key in measurement)},
            )
        if (
            measurement["schema"] != "coordexp-swift-wave2-failure-measurement-v1"
            or [row.get("name") for row in measurement["completed_phases"]]
            != list(completed_phases)
            or not isinstance(measurement["phase_boundary_samples"], list)
        ):
            raise ParityContractError(
                "rich failure measurement is inconsistent with stage evidence",
                code="qwen.parity.failure_resources",
                context={},
            )
        summary = measurement.get("failure_resource_summary")
        if bool(summary) is not bool(gpu_memory):
            raise ParityContractError(
                "rich failure GPU summary and measurement disagree",
                code="qwen.parity.failure_resources",
                context={},
            )
        if summary:
            summary = _mapping(summary, "failure_resource_summary")
            if (
                summary.get("torch_peak_allocated_bytes")
                != gpu_memory["peak_allocated_bytes"]
                or summary.get("torch_peak_reserved_bytes")
                != gpu_memory["peak_reserved_bytes"]
            ):
                raise ParityContractError(
                    "rich failure GPU resource summaries differ",
                    code="qwen.parity.failure_resources",
                    context={},
                )


def _validate_success_execution(
    execution: Mapping[str, Any], *, v3: bool = False
) -> None:
    _expect_exact_keys(
        execution,
        {
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
        },
        owner="execution",
    )
    if (
        not str(execution["device"]).startswith("cuda:")
        or execution["model_dtype"] != "torch.bfloat16"
        or execution["train_mode"] is not True
        or execution["use_cache"] is not False
        or execution["optimizer"] is not None
        or execution["trainable_value_identity_before"]
        != execution["trainable_value_identity_after"]
    ):
        raise ParityContractError(
            "Wave 2 execution identity is invalid",
            code="qwen.parity.receipt_execution",
            context={
                "device": execution["device"],
                "model_dtype": execution["model_dtype"],
                "train_mode": execution["train_mode"],
                "use_cache": execution["use_cache"],
                "optimizer": execution["optimizer"],
            },
        )
    accelerator = _mapping(execution["accelerator"], "execution.accelerator")
    _expect_exact_keys(
        accelerator,
        {
            "distributed_type",
            "rank",
            "local_rank",
            "world_size",
            "device",
            "cuda_current_device",
            "mixed_precision",
            "native_amp",
            "gradient_accumulation_steps",
            "scaler",
            "accelerate_torch_device",
            "prepared_model_attestation",
            "prepared",
            "prepare_route",
            "backward_route",
            "output_conversion",
            "observed_forward_logits_dtypes",
            "observed_inner_autocast",
        },
        owner="execution.accelerator",
    )
    prepared_model_attestation = _mapping(
        accelerator.get("prepared_model_attestation"),
        "execution.accelerator.prepared_model_attestation",
    )
    _expect_exact_keys(
        prepared_model_attestation,
        {
            "binding_branch",
            "prepared_forward_type",
            "wrapper_owner_type",
            "output_wrapper_type",
            "wrapper_identity_chain_verified",
            "unwrapped_original_identity_verified",
        },
        owner="execution.accelerator.prepared_model_attestation",
    )
    binding_branch = prepared_model_attestation.get("binding_branch")
    prepared_forward_type = prepared_model_attestation.get("prepared_forward_type")
    if (
        binding_branch not in {"bound_method", "direct_callable"}
        or not isinstance(prepared_forward_type, str)
        or not prepared_forward_type
        or not isinstance(prepared_model_attestation.get("wrapper_owner_type"), str)
        or not prepared_model_attestation["wrapper_owner_type"]
        or prepared_model_attestation.get("output_wrapper_type")
        != "ConvertOutputsToFp32"
        or prepared_model_attestation.get("wrapper_identity_chain_verified") is not True
        or prepared_model_attestation.get("unwrapped_original_identity_verified")
        is not True
        or (binding_branch == "bound_method" and prepared_forward_type != "method")
        or (binding_branch == "direct_callable" and prepared_forward_type == "method")
    ):
        raise ParityContractError(
            "Wave 2 prepared-model wrapper attestation is invalid",
            code="qwen.parity.receipt_accelerator_wrapper",
            context=dict(prepared_model_attestation),
        )
    try:
        expected_cuda_index = int(str(execution["device"]).rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ParityContractError(
            "Wave 2 execution CUDA device identity is invalid",
            code="qwen.parity.receipt_accelerator",
            context={"device": execution["device"]},
            cause=exc,
        ) from exc
    required_exact = {
        "distributed_type": "NO",
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "mixed_precision": "bf16",
        "native_amp": True,
        "gradient_accumulation_steps": 1,
        "scaler": None,
        "prepared": True,
        "prepare_route": "accelerator.prepare",
        "backward_route": "accelerator.backward",
        "output_conversion": "convert_outputs_to_fp32",
        "cuda_current_device": expected_cuda_index,
    }
    mismatches = {
        key: {"expected": expected, "observed": accelerator.get(key)}
        for key, expected in required_exact.items()
        if accelerator.get(key) != expected
    }
    if (
        mismatches
        or accelerator.get("device") != execution["device"]
        or accelerator.get("accelerate_torch_device") != execution["device"]
    ):
        raise ParityContractError(
            "Wave 2 Accelerator receipt is invalid",
            code="qwen.parity.receipt_accelerator",
            context={"mismatches": mismatches},
        )
    expected_forward_paths = (
        {
            "unmeasured_no_proof_warmup",
            "timed_proof_off",
            "packed_primary",
            "packed_repeat",
            "separate_reference",
            "packed_merged_boundary_negative",
        }
        if v3
        else {
            "unmeasured_no_proof_warmup",
            "timed_proof_off",
            "packed_clean",
            "separate_reference",
            "packed_merged_boundary_negative",
        }
    )
    logits_dtypes = _mapping(
        accelerator.get("observed_forward_logits_dtypes"),
        "observed_forward_logits_dtypes",
    )
    autocast = _mapping(
        accelerator.get("observed_inner_autocast"), "observed_inner_autocast"
    )
    if (
        set(logits_dtypes) != expected_forward_paths
        or set(autocast) != expected_forward_paths
    ):
        raise ParityContractError(
            "Wave 2 Accelerator observation inventory is incomplete",
            code="qwen.parity.receipt_accelerator_observations",
            context={},
        )
    for path in sorted(expected_forward_paths):
        dtype_rows = logits_dtypes[path]
        autocast_rows = autocast[path]
        if (
            not isinstance(dtype_rows, list)
            or not dtype_rows
            or any(item != "torch.float32" for item in dtype_rows)
            or not isinstance(autocast_rows, list)
            or len(autocast_rows) != len(dtype_rows)
        ):
            raise ParityContractError(
                "Wave 2 forward output/autocast observations are invalid",
                code="qwen.parity.receipt_accelerator_observations",
                context={"path": path},
            )
        for row in autocast_rows:
            observation = _mapping(row, f"autocast.{path}")
            _expect_exact_keys(
                observation,
                {
                    "module_type",
                    "layer_idx",
                    "cuda_autocast_enabled",
                    "cuda_autocast_dtype",
                },
                owner=f"autocast.{path}",
            )
            if (
                observation.get("module_type") != "Qwen3VLTextAttention"
                or observation.get("layer_idx") != 0
                or observation.get("cuda_autocast_enabled") is not True
                or observation.get("cuda_autocast_dtype") != "torch.bfloat16"
            ):
                raise ParityContractError(
                    "Wave 2 inner autocast observation is invalid",
                    code="qwen.parity.receipt_accelerator_observations",
                    context={"path": path, "observation": dict(observation)},
                )


def _validate_v3_execution_arm_evidence(
    execution: Mapping[str, Any], *, arms: Mapping[str, Any]
) -> None:
    accelerator = _mapping(execution["accelerator"], "execution.accelerator")
    logits = _mapping(
        accelerator["observed_forward_logits_dtypes"],
        "execution.accelerator.observed_forward_logits_dtypes",
    )
    autocast = _mapping(
        accelerator["observed_inner_autocast"],
        "execution.accelerator.observed_inner_autocast",
    )
    for arm_name in (
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "packed_merged_boundary_negative",
    ):
        arm = _mapping(arms[arm_name], f"arms.{arm_name}")
        if (
            logits.get(arm_name) != arm["forward_logits_dtypes"]
            or autocast.get(arm_name) != arm["inner_autocast"]
        ):
            raise ParityContractError(
                "arm forward evidence differs from Accelerator observations",
                code="qwen.parity.receipt_arm_execution_binding",
                context={"arm": arm_name},
            )


def _validate_success_timings(timings: Mapping[str, Any], *, v3: bool = False) -> None:
    _expect_exact_keys(
        timings,
        {
            "clock",
            "scope",
            "proof_on_forward_ns",
            "proof_off_forward_ns",
            "proof_overhead_ns",
            "ordering",
            "timed_sample_count_per_mode",
            "warmup_sample_count",
            "steady_step_inclusion",
        },
        owner="timings",
    )
    proof_on = _positive_int(timings["proof_on_forward_ns"], "proof_on_forward_ns")
    proof_off = _positive_int(timings["proof_off_forward_ns"], "proof_off_forward_ns")
    overhead = timings["proof_overhead_ns"]
    if isinstance(overhead, bool) or not isinstance(overhead, int):
        raise ParityContractError(
            "Wave 2 proof overhead must be an integer duration delta",
            code="qwen.parity.receipt_timing",
            context={"proof_overhead_ns": overhead},
        )
    if (
        timings["clock"] != "time.perf_counter_ns_with_cuda_synchronize"
        or timings["scope"] != "forward_only_same_packed_inputs_train_mode_no_backward"
        or overhead != proof_on - proof_off
        or timings["ordering"]
        != [
            "unmeasured_no_proof_warmup",
            "timed_proof_off",
            ("timed_proof_on_packed_primary" if v3 else "timed_proof_on_packed_clean"),
        ]
        or timings["timed_sample_count_per_mode"] != 1
        or timings["warmup_sample_count"] != 1
        or timings["steady_step_inclusion"] is not False
    ):
        raise ParityContractError(
            "Wave 2 proof timing contract is invalid",
            code="qwen.parity.receipt_timing",
            context=dict(timings),
        )


def _validate_completed_gpu_idle_preflight(
    idle: Mapping[str, Any], *, expected_device: str
) -> None:
    """Validate the bounded two-check preflight retained after GPU execution."""

    _expect_exact_keys(
        idle,
        {
            "status",
            "requested_device",
            "physical_index",
            "uuid",
            "memory_limit_bytes",
            "utilization_limit_percent",
            "sampler",
            "checks",
        },
        owner="gpu_idle_preflight",
    )
    checks = _list_of_mappings(idle["checks"], "gpu_idle_preflight.checks")
    expected_check_names = {"initial_before_cpu_model_work", "final_before_gpu_work"}
    observed_check_names = {check.get("name") for check in checks}
    if (
        idle["status"] != "passed"
        or idle["requested_device"] != expected_device
        or len(checks) != len(expected_check_names)
        or observed_check_names != expected_check_names
    ):
        raise ParityContractError(
            "GPU idle preflight is incomplete",
            code="qwen.parity.measurement_gpu_idle",
            context={
                "status": idle["status"],
                "requested_device": idle["requested_device"],
                "expected_device": expected_device,
                "expected_checks": sorted(expected_check_names),
                "observed_checks": sorted(str(item) for item in observed_check_names),
            },
        )
    memory_limit = _positive_int(idle["memory_limit_bytes"], "memory_limit_bytes")
    utilization_limit = _positive_int(
        idle["utilization_limit_percent"], "utilization_limit_percent"
    )
    idle_sampler = _mapping(idle["sampler"], "gpu_idle_preflight.sampler")
    _expect_exact_keys(
        idle_sampler,
        {
            "command",
            "sample_count",
            "interval_seconds",
            "cuda_visible_device_mapping",
        },
        owner="gpu_idle_preflight.sampler",
    )
    if (
        idle_sampler["sample_count"] != 3
        or idle_sampler["interval_seconds"] != 0.2
        or not isinstance(idle_sampler["command"], str)
        or not isinstance(idle_sampler["cuda_visible_device_mapping"], str)
    ):
        raise ParityContractError(
            "GPU idle preflight sampler contract is invalid",
            code="qwen.parity.measurement_gpu_idle",
            context=dict(idle_sampler),
        )
    for check in checks:
        _expect_exact_keys(check, {"name", "samples"}, owner="gpu_idle_check")
        samples = _list_of_mappings(
            check["samples"], f"gpu_idle_preflight.{check['name']}.samples"
        )
        if len(samples) != 3:
            raise ParityContractError(
                "GPU idle preflight check requires exactly three samples",
                code="qwen.parity.measurement_gpu_idle",
                context={"name": check["name"], "sample_count": len(samples)},
            )
        for sample in samples:
            _validate_device_sample(sample, owner="gpu_idle_sample")
            if (
                sample["physical_index"] != idle["physical_index"]
                or sample["uuid"] != idle["uuid"]
            ):
                raise ParityContractError(
                    "GPU idle preflight samples changed physical device",
                    code="qwen.parity.measurement_gpu_idle",
                    context=dict(sample),
                )
            if (
                _non_negative_int(sample.get("memory_used_bytes"), "memory_used_bytes")
                >= memory_limit
                or _non_negative_int(
                    sample.get("utilization_percent"), "utilization_percent"
                )
                >= utilization_limit
            ):
                raise ParityContractError(
                    "GPU idle preflight sample exceeds the frozen limit",
                    code="qwen.parity.measurement_gpu_idle",
                    context=dict(sample),
                )


def validate_measurement_contract(
    measurement: Mapping[str, Any], *, v3: bool = False
) -> dict[str, Any]:
    _expect_exact_keys(
        measurement,
        {
            "schema",
            "runtime_launch_identity",
            "gpu_idle_preflight",
            "policies",
            "phases",
            "resources",
            "pack_utilization",
            "semantic_result",
            "eligibility",
            "not_applicable",
        },
        owner="measurement",
    )
    if measurement["schema"] != "coordexp-swift-wave2-measurement-v1":
        raise ParityContractError(
            "Wave 2 measurement schema is unsupported",
            code="qwen.parity.measurement_schema",
            context={"schema": measurement["schema"]},
        )
    launch = _mapping(measurement["runtime_launch_identity"], "runtime_launch_identity")
    _expect_exact_keys(
        launch,
        {
            "rank",
            "local_rank",
            "world_size",
            "process_count",
            "mode",
            "device",
            "mixed_precision",
        },
        owner="runtime_launch_identity",
    )
    if (
        launch.get("rank") != 0
        or launch.get("local_rank") != 0
        or launch.get("world_size") != 1
        or launch.get("process_count") != 1
        or launch.get("mode") != "single_process_explicit_device"
        or launch.get("mixed_precision") != "bf16"
        or not str(launch.get("device", "")).startswith("cuda:")
    ):
        raise ParityContractError(
            "Wave 2 measurement requires exact one-rank launch identity",
            code="qwen.parity.measurement_launch",
            context=dict(launch),
        )
    idle = _mapping(measurement["gpu_idle_preflight"], "gpu_idle_preflight")
    _validate_completed_gpu_idle_preflight(
        idle,
        expected_device=str(measurement["runtime_launch_identity"]["device"]),
    )
    policies = _mapping(measurement["policies"], "policies")
    _expect_exact_keys(
        policies,
        {
            "seed",
            "config_fingerprint",
            "packing_policy",
            "provider_policy",
            "attention_backend",
            "attention_proof_policy",
        },
        owner="policies",
    )
    _non_negative_int(policies["seed"], "policies.seed")
    _sha256_text(policies["config_fingerprint"], "config_fingerprint")
    _non_empty_text(policies["packing_policy"], "packing_policy")
    _non_empty_text(policies["provider_policy"], "provider_policy")
    if policies["attention_backend"] != "flash_attention_2":
        raise ParityContractError(
            "Wave 2 measurement requires the frozen FA2 backend",
            code="qwen.parity.measurement_backend",
            context={"attention_backend": policies["attention_backend"]},
        )
    if policies["attention_proof_policy"] != "bounded_first_packed_forward":
        raise ParityContractError(
            "Wave 2 measurement proof policy is invalid",
            code="qwen.parity.measurement_proof_policy",
            context={"attention_proof_policy": policies["attention_proof_policy"]},
        )
    phases = _list_of_mappings(measurement["phases"], "phases")
    expected_phase_order = [
        "model_setup",
        "input_construction",
        "warmup_forward",
        "proof_off_forward",
        *(["packed_primary", "packed_repeat"] if v3 else ["packed_clean"]),
        "separate_reference",
        "negative_control",
        "comparison",
    ]
    expected_phases = set(expected_phase_order)
    observed_phase_order = [str(phase.get("name")) for phase in phases]
    if observed_phase_order != expected_phase_order:
        raise ParityContractError(
            "Wave 2 measurement phase inventory is incomplete",
            code="qwen.parity.measurement_phases",
            context={
                "expected": sorted(expected_phases),
                "observed": observed_phase_order,
            },
        )
    previous_end = 0
    for phase in phases:
        _expect_exact_keys(
            phase,
            {"name", "start_ns", "end_ns", "duration_ns", "status"},
            owner="phase",
        )
        start = _non_negative_int(phase["start_ns"], "phase.start_ns")
        end = _positive_int(phase["end_ns"], "phase.end_ns")
        duration = _positive_int(phase["duration_ns"], "phase.duration_ns")
        if (
            phase["status"] != "completed"
            or end - start != duration
            or start < previous_end
        ):
            raise ParityContractError(
                "Wave 2 measurement phase is invalid",
                code="qwen.parity.measurement_phase",
                context=dict(phase),
            )
        previous_end = end
    resources = _mapping(measurement["resources"], "resources")
    _expect_exact_keys(
        resources,
        {"host", "gpu", "ceilings", "phase_boundary_samples"},
        owner="resources",
    )
    host = _mapping(resources["host"], "resources.host")
    gpu = _mapping(resources["gpu"], "resources.gpu")
    ceilings = _mapping(resources["ceilings"], "resources.ceilings")
    _expect_exact_keys(
        host,
        {"rss_hwm_bytes", "io_read_bytes_hwm", "io_write_bytes_hwm", "source"},
        owner="resources.host",
    )
    _expect_exact_keys(
        gpu,
        {
            "device",
            "torch_peak_allocated_bytes",
            "torch_peak_reserved_bytes",
            "device_used_hwm_bytes",
            "source",
            "device_sampler",
        },
        owner="resources.gpu",
    )
    _expect_exact_keys(
        ceilings,
        {"status", "host_bytes", "device_bytes", "comparison"},
        owner="resources.ceilings",
    )
    for field in ("rss_hwm_bytes", "io_read_bytes_hwm", "io_write_bytes_hwm"):
        _non_negative_int(host.get(field), f"resources.host.{field}")
    for field in (
        "torch_peak_allocated_bytes",
        "torch_peak_reserved_bytes",
        "device_used_hwm_bytes",
    ):
        _non_negative_int(gpu.get(field), f"resources.gpu.{field}")
    host_ceiling = _positive_int(ceilings.get("host_bytes"), "ceilings.host_bytes")
    device_ceiling = _positive_int(
        ceilings.get("device_bytes"), "ceilings.device_bytes"
    )
    ceiling_comparison = _mapping(ceilings.get("comparison"), "ceilings.comparison")
    _expect_exact_keys(
        ceiling_comparison,
        {"host_rss_below", "torch_reserved_below", "device_sampler_below"},
        owner="ceilings.comparison",
    )
    observed_below = {
        "host_rss_below": int(host["rss_hwm_bytes"]) < host_ceiling,
        "torch_reserved_below": int(gpu["torch_peak_reserved_bytes"]) < device_ceiling,
        "device_sampler_below": int(gpu["device_used_hwm_bytes"]) < device_ceiling,
    }
    if (
        ceilings.get("status") != "passed"
        or dict(ceiling_comparison) != observed_below
        or not all(observed_below.values())
    ):
        raise ParityContractError(
            "Wave 2 resource ceiling status is not passed",
            code="qwen.parity.measurement_ceiling",
            context={"ceilings": dict(ceilings), "observed": observed_below},
        )
    sampler = _mapping(gpu["device_sampler"], "resources.gpu.device_sampler")
    _expect_exact_keys(
        sampler,
        {
            "status",
            "interval_seconds",
            "sample_count",
            "maximum_samples",
            "hwm_memory_used_bytes",
            "hwm_utilization_percent",
            "first_monotonic_ns",
            "last_monotonic_ns",
        },
        owner="resources.gpu.device_sampler",
    )
    if sampler["status"] != "completed":
        raise ParityContractError(
            "Wave 2 bounded GPU sampler did not complete",
            code="qwen.parity.measurement_sampler",
            context=dict(sampler),
        )
    sample_count = _positive_int(sampler["sample_count"], "sampler.sample_count")
    maximum_samples = _positive_int(
        sampler["maximum_samples"], "sampler.maximum_samples"
    )
    if sample_count > maximum_samples:
        raise ParityContractError(
            "Wave 2 GPU sampler exceeded its sample bound",
            code="qwen.parity.measurement_sampler",
            context={"sample_count": sample_count, "maximum_samples": maximum_samples},
        )
    interval = sampler["interval_seconds"]
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or interval <= 0
    ):
        raise ParityContractError(
            "Wave 2 GPU sampler interval is invalid",
            code="qwen.parity.measurement_sampler",
            context={"interval_seconds": interval},
        )
    _non_negative_int(sampler["hwm_memory_used_bytes"], "sampler.hwm_memory_used_bytes")
    utilization_hwm = _non_negative_int(
        sampler["hwm_utilization_percent"], "sampler.hwm_utilization_percent"
    )
    first_sample_ns = _positive_int(
        sampler["first_monotonic_ns"], "sampler.first_monotonic_ns"
    )
    last_sample_ns = _positive_int(
        sampler["last_monotonic_ns"], "sampler.last_monotonic_ns"
    )
    if utilization_hwm > 100 or last_sample_ns < first_sample_ns:
        raise ParityContractError(
            "Wave 2 GPU sampler HWM or timestamps are invalid",
            code="qwen.parity.measurement_sampler",
            context=dict(sampler),
        )
    if int(sampler["hwm_memory_used_bytes"]) != int(gpu["device_used_hwm_bytes"]):
        raise ParityContractError(
            "Wave 2 GPU sampler HWM disagrees with the resource summary",
            code="qwen.parity.measurement_sampler",
            context={},
        )
    phase_samples = _list_of_mappings(
        resources["phase_boundary_samples"], "resources.phase_boundary_samples"
    )
    if [sample.get("phase") for sample in phase_samples] != expected_phase_order:
        raise ParityContractError(
            "Wave 2 phase-boundary resource inventory is incomplete",
            code="qwen.parity.measurement_resources",
            context={
                "expected": sorted(expected_phases),
                "observed": sorted(
                    str(sample.get("phase")) for sample in phase_samples
                ),
            },
        )
    for sample in phase_samples:
        _validate_phase_resource_sample(sample)
    host_summary = {
        "rss_hwm_bytes": max(
            int(sample["host"]["rss_hwm_bytes"]) for sample in phase_samples
        ),
        "io_read_bytes_hwm": max(
            int(sample["host"]["io_read_bytes"]) for sample in phase_samples
        ),
        "io_write_bytes_hwm": max(
            int(sample["host"]["io_write_bytes"]) for sample in phase_samples
        ),
    }
    if any(int(host[field]) != value for field, value in host_summary.items()):
        raise ParityContractError(
            "Wave 2 host resource summary disagrees with phase boundaries",
            code="qwen.parity.measurement_resources",
            context={"expected": host_summary, "observed": dict(host)},
        )
    phase_peak_allocated = max(
        int(sample["torch_cuda"]["max_allocated_bytes"]) for sample in phase_samples
    )
    phase_peak_reserved = max(
        int(sample["torch_cuda"]["max_reserved_bytes"]) for sample in phase_samples
    )
    if phase_peak_allocated != int(
        gpu["torch_peak_allocated_bytes"]
    ) or phase_peak_reserved != int(gpu["torch_peak_reserved_bytes"]):
        raise ParityContractError(
            "Wave 2 CUDA peak summary disagrees with phase-boundary peaks",
            code="qwen.parity.measurement_peak_mismatch",
            context={
                "phase_peak_allocated": phase_peak_allocated,
                "summary_peak_allocated": gpu["torch_peak_allocated_bytes"],
                "phase_peak_reserved": phase_peak_reserved,
                "summary_peak_reserved": gpu["torch_peak_reserved_bytes"],
            },
        )
    utilization = _mapping(measurement["pack_utilization"], "pack_utilization")
    _expect_exact_keys(
        utilization,
        {
            "pack_length",
            "global_max_length",
            "unused_tokens",
            "utilization_ratio",
            "segment_count",
        },
        owner="pack_utilization",
    )
    pack_length = _positive_int(utilization["pack_length"], "pack_length")
    global_max_length = _positive_int(
        utilization["global_max_length"], "global_max_length"
    )
    unused_tokens = _non_negative_int(utilization["unused_tokens"], "unused_tokens")
    _positive_int(utilization["segment_count"], "segment_count")
    ratio = utilization.get("utilization_ratio")
    if (
        not isinstance(ratio, (int, float))
        or isinstance(ratio, bool)
        or not 0 < ratio <= 1
        or pack_length > global_max_length
        or unused_tokens != global_max_length - pack_length
        or abs(float(ratio) - pack_length / global_max_length) > 1.0e-12
    ):
        raise ParityContractError(
            "pack utilization ratio is invalid",
            code="qwen.parity.measurement_utilization",
            context={"utilization_ratio": ratio},
        )
    semantic = _mapping(measurement["semantic_result"], "semantic_result")
    required_semantic = {
        "clean_parity_passed",
        "negative_forward_signal_detected",
        "all_layer_proof_passed",
    }
    if set(semantic) != required_semantic or not all(
        semantic[field] is True for field in required_semantic
    ):
        raise ParityContractError(
            "Wave 2 semantic result is not fully passed",
            code="qwen.parity.measurement_semantic",
            context=dict(semantic),
        )
    eligibility = _mapping(measurement["eligibility"], "eligibility")
    _expect_exact_keys(
        eligibility,
        {
            "terminal_eligible",
            "steady_state_timing_eligible",
            "steady_state_reason",
        },
        owner="eligibility",
    )
    if (
        eligibility.get("terminal_eligible") is not True
        or eligibility.get("steady_state_timing_eligible") is not False
        or not isinstance(eligibility.get("steady_state_reason"), str)
    ):
        raise ParityContractError(
            "Wave 2 eligibility fields are invalid",
            code="qwen.parity.measurement_eligibility",
            context=dict(eligibility),
        )
    not_applicable = _mapping(measurement["not_applicable"], "not_applicable")
    expected_na = {"cache", "eval", "checkpoint", "warmup", "per_step"}
    if set(not_applicable) != expected_na:
        raise ParityContractError(
            "Wave 2 typed not-applicable inventory is incomplete",
            code="qwen.parity.measurement_na",
            context={
                "expected": sorted(expected_na),
                "observed": sorted(not_applicable),
            },
        )
    for name, record in not_applicable.items():
        item = _mapping(record, f"not_applicable.{name}")
        _expect_exact_keys(
            item,
            {"status", "reason"},
            owner=f"not_applicable.{name}",
        )
        if (
            item.get("status") != "not_applicable"
            or not isinstance(item.get("reason"), str)
            or not item["reason"]
        ):
            raise ParityContractError(
                "Wave 2 not-applicable record is invalid",
                code="qwen.parity.measurement_na",
                context={"name": name, "record": dict(item)},
            )
    canonical_json_bytes(measurement)
    return dict(measurement)


def legacy_v2_tolerances() -> dict[str, Any]:
    return {
        "bf16_logits_and_total_loss": {"rtol": BF16_RTOL, "atol": BF16_ATOL},
        "cross_arm_bf16_loss_term_scalars": {
            "rtol": BF16_RTOL,
            "atol": BF16_ATOL,
        },
        "same_forward_fp32_protected_loss_scalars": {
            "rtol": LOSS_RTOL,
            "atol": LOSS_ATOL,
        },
        "gradients_by_parameter_dtype": {
            "torch.bfloat16": {"rtol": BF16_RTOL, "atol": BF16_ATOL},
            "torch.float32": {"rtol": FP32_RTOL, "atol": FP32_ATOL},
        },
        "negative_control": "outside_at_least_one_frozen_supervised_logit_or_total_loss_tolerance",
        "widening_after_results": False,
    }


def frozen_tolerances() -> dict[str, Any]:
    """Return the deliberately small Wave 2 v3 numerical contract."""

    return {
        "bf16_derived_fp32_comparison": {
            "source_compute_dtype": BF16_COMPUTE_PROVENANCE_DTYPE,
            "comparison_dtype": FP32_COMPARISON_DTYPE,
            "rtol": BF16_RTOL,
            "atol": BF16_ATOL,
        },
        "same_packed_repeat": {
            "comparison_dtype": FP32_COMPARISON_DTYPE,
            "max_abs": PACKED_REPEAT_MAX_ABS,
        },
        "negative_control": (
            "outside_at_least_one_frozen_supervised_logit_or_total_loss_tolerance"
        ),
        "storage_dtype_selects_tolerance": False,
        "adaptive_tolerance": False,
        "widening_after_results": False,
    }


def base_model_weight_identity(
    model_root: str | Path,
    *,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Content-bind the bounded base-model safetensors payload."""

    identity, _ = base_model_weight_identity_with_execution_policy(
        model_root,
        max_workers=max_workers,
    )
    return identity


def base_model_weight_identity_with_execution_policy(
    model_root: str | Path,
    *,
    max_workers: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Content-bind base weights and return the schema-neutral hash policy."""

    _validate_requested_weight_hash_workers(max_workers)

    root = Path(model_root).expanduser().resolve()
    if not root.is_dir():
        raise ParityContractError(
            "base-model weight root is not a directory",
            code="qwen.parity.weight_root",
            context={"path": str(root)},
        )
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index_identity, index_bytes = _stable_file_identity(
            index_path,
            root=root,
            max_bytes=MAX_WEIGHT_INDEX_BYTES,
            return_bytes=True,
        )
        try:
            index_payload = json.loads(
                index_bytes.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ParityContractError(
                "base-model weight index is not strict UTF-8 JSON",
                code="qwen.parity.weight_index_json",
                context={"path": str(index_path)},
                cause=exc,
            ) from exc
        if not isinstance(index_payload, dict):
            raise ParityContractError(
                "base-model weight index root must be a mapping",
                code="qwen.parity.weight_index_shape",
                context={"path": str(index_path)},
            )
        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ParityContractError(
                "base-model weight index must declare a non-empty weight_map",
                code="qwen.parity.weight_index_shape",
                context={"path": str(index_path)},
            )
        if len(weight_map) > MAX_WEIGHT_DECLARATIONS:
            raise ParityContractError(
                "base-model weight index exceeds the declaration bound",
                code="qwen.parity.weight_declaration_bound",
                context={
                    "count": len(weight_map),
                    "maximum": MAX_WEIGHT_DECLARATIONS,
                },
            )
        shard_names: set[str] = set()
        for tensor_name, shard_name in weight_map.items():
            _non_empty_text(tensor_name, "weight_map tensor name")
            shard_names.add(_safe_relative_weight_path(shard_name))
        if len(shard_names) > MAX_WEIGHT_SHARDS:
            raise ParityContractError(
                "base-model weight index exceeds the shard bound",
                code="qwen.parity.weight_shard_bound",
                context={"count": len(shard_names), "maximum": MAX_WEIGHT_SHARDS},
            )
        shard_paths = [root / shard_name for shard_name in sorted(shard_names)]
        _preflight_weight_file_set(shard_paths)
        resolved_workers = _resolve_weight_hash_workers(
            max_workers,
            payload_file_count=len(shard_paths),
        )
        shards = _hash_weight_shards(
            shard_paths,
            root=root,
            resolved_workers=resolved_workers,
        )
        mode = "indexed_safetensors"
        index_artifact: dict[str, Any] | None = index_identity
        declaration_count = len(weight_map)
    else:
        standalone = root / "model.safetensors"
        discovered: list[Path] = []
        try:
            for candidate in root.iterdir():
                if candidate.name.endswith(".safetensors"):
                    discovered.append(candidate)
                    if len(discovered) > MAX_WEIGHT_SHARDS:
                        raise ParityContractError(
                            "unindexed base-model weight discovery exceeds its bound",
                            code="qwen.parity.weight_shard_bound",
                            context={
                                "count": len(discovered),
                                "maximum": MAX_WEIGHT_SHARDS,
                            },
                        )
        except OSError as exc:
            raise ParityContractError(
                "base-model weight root cannot be inventoried",
                code="qwen.parity.weight_root",
                context={"path": str(root), "error": type(exc).__name__},
                cause=exc,
            ) from exc
        discovered.sort()
        if not standalone.is_file() or discovered != [standalone]:
            raise ParityContractError(
                "base-model weights require an index or one standalone model.safetensors",
                code="qwen.parity.weight_layout",
                context={
                    "root": str(root),
                    "discovered_count": len(discovered),
                    "discovered": [item.name for item in discovered[:16]],
                },
            )
        _preflight_weight_file_set([standalone])
        resolved_workers = _resolve_weight_hash_workers(
            max_workers,
            payload_file_count=1,
        )
        shards = _hash_weight_shards(
            [standalone],
            root=root,
            resolved_workers=resolved_workers,
        )
        mode = "standalone_safetensors"
        index_artifact = None
        declaration_count = None
    total_bytes = sum(int(item["size_bytes"]) for item in shards)
    if total_bytes > MAX_WEIGHT_TOTAL_BYTES:
        raise ParityContractError(
            "base-model weight payload exceeds the total byte bound",
            code="qwen.parity.weight_total_bound",
            context={"total_bytes": total_bytes, "maximum": MAX_WEIGHT_TOTAL_BYTES},
        )
    determinants = {
        "schema": MODEL_WEIGHT_IDENTITY_SCHEMA,
        "root": str(root),
        "mode": mode,
        "index": index_artifact,
        "declaration_count": declaration_count,
        "shards": shards,
        "shard_count": len(shards),
        "total_bytes": total_bytes,
        "bounds": {
            "max_index_bytes": MAX_WEIGHT_INDEX_BYTES,
            "max_declarations": MAX_WEIGHT_DECLARATIONS,
            "max_shards": MAX_WEIGHT_SHARDS,
            "max_shard_bytes": MAX_WEIGHT_SHARD_BYTES,
            "max_total_bytes": MAX_WEIGHT_TOTAL_BYTES,
        },
    }
    identity = {**determinants, "aggregate_sha256": sha256_json(determinants)}
    execution_policy = {
        "schema": MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA,
        "strategy": "thread_pool_file_sha256",
        "resolved_workers": resolved_workers,
        "payload_file_count": len(shards),
    }
    return validate_model_weight_identity(identity), execution_policy


def validate_model_weight_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one bounded indexed or standalone safetensors identity."""

    receipt = dict(identity)
    _expect_exact_keys(
        receipt,
        {
            "schema",
            "root",
            "mode",
            "index",
            "declaration_count",
            "shards",
            "shard_count",
            "total_bytes",
            "bounds",
            "aggregate_sha256",
        },
        owner="model_weight_identity",
    )
    if receipt["schema"] != MODEL_WEIGHT_IDENTITY_SCHEMA:
        raise ParityContractError(
            "base-model weight identity schema is unsupported",
            code="qwen.parity.weight_schema",
            context={"schema": receipt["schema"]},
        )
    _non_empty_text(receipt["root"], "model_weight_identity.root")
    mode = receipt["mode"]
    if mode not in {"indexed_safetensors", "standalone_safetensors"}:
        raise ParityContractError(
            "base-model weight identity mode is unsupported",
            code="qwen.parity.weight_layout",
            context={"mode": mode},
        )
    expected_bounds = {
        "max_index_bytes": MAX_WEIGHT_INDEX_BYTES,
        "max_declarations": MAX_WEIGHT_DECLARATIONS,
        "max_shards": MAX_WEIGHT_SHARDS,
        "max_shard_bytes": MAX_WEIGHT_SHARD_BYTES,
        "max_total_bytes": MAX_WEIGHT_TOTAL_BYTES,
    }
    bounds = _mapping(receipt["bounds"], "model_weight_identity.bounds")
    if dict(bounds) != expected_bounds:
        raise ParityContractError(
            "base-model weight identity bounds drifted",
            code="qwen.parity.weight_bounds",
            context={"expected": expected_bounds, "observed": dict(bounds)},
        )
    shards = _list_of_mappings(receipt["shards"], "model_weight_identity.shards")
    shard_count = _positive_int(receipt["shard_count"], "shard_count")
    if shard_count != len(shards) or shard_count > MAX_WEIGHT_SHARDS:
        raise ParityContractError(
            "base-model weight shard count is invalid",
            code="qwen.parity.weight_shard_bound",
            context={"declared": shard_count, "observed": len(shards)},
        )
    shard_paths: list[str] = []
    observed_total = 0
    for shard in shards:
        _expect_exact_keys(
            shard, {"path", "size_bytes", "sha256"}, owner="weight_shard"
        )
        shard_path = _safe_relative_weight_path(shard["path"])
        shard_paths.append(shard_path)
        size_bytes = _positive_int(shard["size_bytes"], "weight_shard.size_bytes")
        if size_bytes > MAX_WEIGHT_SHARD_BYTES:
            raise ParityContractError(
                "base-model weight shard exceeds the byte bound",
                code="qwen.parity.weight_file_bound",
                context={"path": shard_path, "size_bytes": size_bytes},
            )
        observed_total += size_bytes
        _sha256_text(shard["sha256"], "weight_shard.sha256")
    if shard_paths != sorted(set(shard_paths)):
        raise ParityContractError(
            "base-model weight shard paths must be unique and sorted",
            code="qwen.parity.weight_shard_order",
            context={"paths": shard_paths},
        )
    total_bytes = _positive_int(receipt["total_bytes"], "total_bytes")
    if total_bytes != observed_total or total_bytes > MAX_WEIGHT_TOTAL_BYTES:
        raise ParityContractError(
            "base-model weight total bytes are inconsistent",
            code="qwen.parity.weight_total_bound",
            context={"declared": total_bytes, "observed": observed_total},
        )
    if mode == "indexed_safetensors":
        index = _mapping(receipt["index"], "model_weight_identity.index")
        _expect_exact_keys(
            index, {"path", "size_bytes", "sha256"}, owner="weight_index"
        )
        if index["path"] != "model.safetensors.index.json":
            raise ParityContractError(
                "base-model weight index path is invalid",
                code="qwen.parity.weight_index_path",
                context={"path": index["path"]},
            )
        index_size = _positive_int(index["size_bytes"], "weight_index.size_bytes")
        if index_size > MAX_WEIGHT_INDEX_BYTES:
            raise ParityContractError(
                "base-model weight index exceeds its byte bound",
                code="qwen.parity.weight_file_bound",
                context={"size_bytes": index_size},
            )
        _sha256_text(index["sha256"], "weight_index.sha256")
        declarations = _positive_int(receipt["declaration_count"], "declaration_count")
        if declarations > MAX_WEIGHT_DECLARATIONS:
            raise ParityContractError(
                "base-model weight declaration count exceeds its bound",
                code="qwen.parity.weight_declaration_bound",
                context={"count": declarations},
            )
    elif (
        receipt["index"] is not None
        or receipt["declaration_count"] is not None
        or shard_paths != ["model.safetensors"]
    ):
        raise ParityContractError(
            "standalone safetensors identity has indexed-layout residue",
            code="qwen.parity.weight_layout",
            context={"shard_paths": shard_paths},
        )
    fingerprint = _sha256_text(
        receipt["aggregate_sha256"], "model_weight_identity.aggregate_sha256"
    )
    body = dict(receipt)
    del body["aggregate_sha256"]
    observed_fingerprint = sha256_json(body)
    if fingerprint != observed_fingerprint:
        raise ParityContractError(
            "base-model weight identity fingerprint is invalid",
            code="qwen.parity.weight_fingerprint",
            context={"expected": fingerprint, "observed": observed_fingerprint},
        )
    canonical_json_bytes(receipt)
    return receipt


def validate_dependency_provenance(
    provenance: Mapping[str, Any],
    *,
    allow_historical_without_cuda_runtime: bool = False,
) -> dict[str, Any]:
    """Require the authoritative imported FA2 binary identity."""

    receipt = dict(provenance)
    _expect_exact_keys(
        receipt,
        {"schema", "collector", "collected", "accelerate_runtime_sources"},
        owner="dependency_identity",
    )
    if (
        receipt["schema"] != DEPENDENCY_IDENTITY_SCHEMA
        or receipt["collector"]
        != "src.artifacts.provenance.collect_dependency_provenance"
    ):
        raise ParityContractError(
            "Wave 2 dependency identity header is unsupported",
            code="qwen.parity.dependency_schema",
            context={
                "schema": receipt["schema"],
                "collector": receipt["collector"],
            },
        )
    collected = _mapping(receipt["collected"], "dependency_identity.collected")
    historical_components = {
        "ms-swift",
        "transformers",
        "flash-attn",
        "flash_attn_2_cuda",
        "torch",
        "accelerate",
        "peft",
        "tokenizers",
    }
    expected_components = {*historical_components, "cuda-runtime"}
    observed_components = set(collected)
    historical_inventory = (
        allow_historical_without_cuda_runtime
        and observed_components == historical_components
    )
    if observed_components != expected_components and not historical_inventory:
        raise ParityContractError(
            "dependency provenance component inventory is incomplete",
            code="qwen.parity.dependency_inventory",
            context={
                "missing": sorted(expected_components - observed_components),
                "unknown": sorted(observed_components - expected_components),
            },
        )
    if not historical_inventory:
        cuda_runtime = _mapping(collected.get("cuda-runtime"), "cuda-runtime")
        _expect_exact_keys(
            cuda_runtime,
            {
                "distribution",
                "import_name",
                "distribution_version",
                "distribution_record",
                "role",
                "origin_resolution",
                "loaded_soname",
                "distribution_relative_path",
                "distribution_origin",
                "imported_origin",
                "loaded_origin_matches_distribution",
                "origin_kind",
                "sha256",
                "size_bytes",
                "elf_build_id",
                "source_repository",
                "source_identities",
                "native_identities",
            },
            owner="cuda-runtime",
        )
        fixed_cuda_runtime_fields = {
            "distribution": "nvidia-cuda-runtime-cu12",
            "import_name": None,
            "role": "runtime_dependency",
            "origin_resolution": "loaded_shared_object",
            "loaded_soname": "libcudart.so.12",
            "distribution_relative_path": ("nvidia/cuda_runtime/lib/libcudart.so.12"),
            "loaded_origin_matches_distribution": True,
            "origin_kind": "binary",
            "source_identities": {},
            "native_identities": {},
        }
        drifted_cuda_runtime_fields = sorted(
            key
            for key, expected in fixed_cuda_runtime_fields.items()
            if cuda_runtime.get(key) != expected
        )
        if drifted_cuda_runtime_fields:
            raise ParityContractError(
                "CUDA runtime dependency provenance is unsupported",
                code="qwen.parity.cuda_runtime_provenance",
                context={"drifted_fields": drifted_cuda_runtime_fields},
            )
        distribution_version = _mapping(
            cuda_runtime["distribution_version"],
            "cuda-runtime.distribution_version",
        )
        _expect_exact_keys(
            distribution_version,
            {"status", "value"},
            owner="cuda-runtime.distribution_version",
        )
        if distribution_version["status"] != "available":
            raise ParityContractError(
                "CUDA runtime distribution version is unavailable",
                code="qwen.parity.cuda_runtime_provenance",
                context={"status": distribution_version["status"]},
            )
        _non_empty_text(
            distribution_version["value"],
            "cuda-runtime.distribution_version.value",
        )
        distribution_record = _mapping(
            cuda_runtime["distribution_record"],
            "cuda-runtime.distribution_record",
        )
        _expect_exact_keys(
            distribution_record,
            {"status", "value"},
            owner="cuda-runtime.distribution_record",
        )
        if distribution_record["status"] != "available":
            raise ParityContractError(
                "CUDA runtime distribution record is unavailable",
                code="qwen.parity.cuda_runtime_provenance",
                context={"status": distribution_record["status"]},
            )
        distribution_record_value = _mapping(
            distribution_record["value"],
            "cuda-runtime.distribution_record.value",
        )
        _expect_exact_keys(
            distribution_record_value,
            {"sha256", "size_bytes"},
            owner="cuda-runtime.distribution_record.value",
        )
        _sha256_text(
            distribution_record_value["sha256"],
            "cuda-runtime.distribution_record.value.sha256",
        )
        _positive_int(
            distribution_record_value["size_bytes"],
            "cuda-runtime.distribution_record.value.size_bytes",
        )
        for field in ("distribution_origin", "imported_origin"):
            identity = _mapping(cuda_runtime[field], f"cuda-runtime.{field}")
            _expect_exact_keys(
                identity,
                {"status", "value"},
                owner=f"cuda-runtime.{field}",
            )
            if identity["status"] != "available":
                raise ParityContractError(
                    "CUDA runtime binary origin is unavailable",
                    code="qwen.parity.cuda_runtime_provenance",
                    context={"field": field, "status": identity["status"]},
                )
            _non_empty_text(identity["value"], f"cuda-runtime.{field}.value")
        for field, validator in (
            ("sha256", _sha256_text),
            ("size_bytes", _positive_int),
            ("elf_build_id", _non_empty_text),
        ):
            identity = _mapping(cuda_runtime[field], f"cuda-runtime.{field}")
            _expect_exact_keys(
                identity,
                {"status", "value"},
                owner=f"cuda-runtime.{field}",
            )
            if identity["status"] != "available":
                raise ParityContractError(
                    "CUDA runtime binary identity is unavailable",
                    code="qwen.parity.cuda_runtime_provenance",
                    context={"field": field, "status": identity["status"]},
                )
            validator(identity["value"], f"cuda-runtime.{field}.value")
        source_repository = _mapping(
            cuda_runtime["source_repository"], "cuda-runtime.source_repository"
        )
        if source_repository != {
            "status": "unavailable",
            "reason": "not_source_origin",
        }:
            raise ParityContractError(
                "CUDA runtime source provenance is unsupported",
                code="qwen.parity.cuda_runtime_provenance",
                context={},
            )
    binary = _mapping(collected.get("flash_attn_2_cuda"), "flash_attn_2_cuda")
    imported_origin = _mapping(
        binary.get("imported_origin"), "flash_attn_2_cuda.imported_origin"
    )
    digest = _mapping(binary.get("sha256"), "flash_attn_2_cuda.sha256")
    if (
        binary.get("role") != "runtime_dependency"
        or binary.get("origin_kind") != "binary"
        or imported_origin.get("status") != "available"
        or digest.get("status") != "available"
    ):
        raise ParityContractError(
            "flash_attn_2_cuda binary provenance is unavailable",
            code="qwen.parity.flash_binary_unavailable",
            context={
                "role": binary.get("role"),
                "origin_kind": binary.get("origin_kind"),
                "origin_status": imported_origin.get("status"),
                "sha256_status": digest.get("status"),
            },
        )
    _non_empty_text(imported_origin.get("value"), "flash_attn_2_cuda origin")
    _sha256_text(digest.get("value"), "flash_attn_2_cuda sha256")
    runtime_sources = _list_of_mappings(
        receipt["accelerate_runtime_sources"], "accelerate_runtime_sources"
    )
    expected_runtime_paths = [
        "accelerator.py",
        "state.py",
        "utils/modeling.py",
        "utils/operations.py",
    ]
    observed_runtime_paths: list[str] = []
    for row in runtime_sources:
        _expect_exact_keys(
            row,
            {"relative_path", "resolved_path", "sha256"},
            owner="accelerate_runtime_source",
        )
        relative = _non_empty_text(
            row["relative_path"], "accelerate_runtime_source.relative_path"
        )
        observed_runtime_paths.append(relative)
        _non_empty_text(row["resolved_path"], "accelerate_runtime_source.resolved_path")
        _sha256_text(row["sha256"], "accelerate_runtime_source.sha256")
    if observed_runtime_paths != expected_runtime_paths:
        raise ParityContractError(
            "Accelerate runtime source identity is incomplete or unordered",
            code="qwen.parity.accelerate_source_inventory",
            context={
                "expected": expected_runtime_paths,
                "observed": observed_runtime_paths,
            },
        )
    canonical_json_bytes(receipt)
    return receipt


def assert_dependency_provenance_equal(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    expected_receipt = validate_dependency_provenance(expected)
    observed_receipt = validate_dependency_provenance(observed)
    if expected_receipt != observed_receipt:
        raise ParityContractError(
            "runtime dependency provenance changed after plan preparation",
            code="qwen.parity.dependency_drift",
            context={
                "expected_sha256": sha256_json(expected_receipt),
                "observed_sha256": sha256_json(observed_receipt),
            },
        )


def assert_model_weight_identity_equal(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    expected_receipt = validate_model_weight_identity(expected)
    observed_receipt = validate_model_weight_identity(observed)
    if expected_receipt != observed_receipt:
        raise ParityContractError(
            "base-model weight identity changed after plan preparation",
            code="qwen.parity.weight_identity_drift",
            context={
                "expected_aggregate_sha256": expected_receipt.get("aggregate_sha256"),
                "observed_aggregate_sha256": observed_receipt.get("aggregate_sha256"),
            },
        )


def semantic_atom_inventory(
    token_sequences: Any | Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    sequences = _as_sequence_tuple(token_sequences)
    rows: list[dict[str, Any]] = []
    keys: set[SemanticAtomKey] = set()
    for sequence in sequences:
        atoms = getattr(sequence, "atoms", None)
        if atoms is None:
            raise ParityContractError(
                "semantic atom inventory requires TokenSequence-like inputs",
                code="qwen.parity.atom_sequence",
                context={"value_type": type(sequence).__name__},
            )
        for atom in atoms:
            key = SemanticAtomKey.from_atom(atom)
            if key in keys:
                raise ParityContractError(
                    "semantic atom inventory contains a duplicate key",
                    code="qwen.parity.duplicate_atom_key",
                    context={"key": key.to_artifact_dict()},
                )
            keys.add(key)
            rows.append(key.to_artifact_dict())
    return tuple(sorted(rows, key=lambda item: _key_from_mapping(item)))


def semantic_atom_key_inventory_sha256(
    keys: Sequence[SemanticAtomKey | Mapping[str, Any]],
) -> str:
    """Hash one ordered semantic-key inventory using its canonical artifact form."""

    rows: list[dict[str, Any]] = []
    for value in keys:
        key = value if isinstance(value, SemanticAtomKey) else _key_from_mapping(value)
        rows.append(key.to_artifact_dict())
    return sha256_json(rows)


def compare_semantic_atom_inventories(
    left: Any | Sequence[Any],
    right: Any | Sequence[Any],
) -> dict[str, Any]:
    left_rows = semantic_atom_inventory(left)
    right_rows = semantic_atom_inventory(right)
    left_keys = {_key_from_mapping(item) for item in left_rows}
    right_keys = {_key_from_mapping(item) for item in right_rows}
    missing = sorted(left_keys - right_keys)
    extra = sorted(right_keys - left_keys)
    return {
        "passed": not missing and not extra,
        "left_count": len(left_keys),
        "right_count": len(right_keys),
        "left_key_inventory_sha256": semantic_atom_key_inventory_sha256(left_rows),
        "right_key_inventory_sha256": semantic_atom_key_inventory_sha256(right_rows),
        "missing": [key.to_artifact_dict() for key in missing[:16]],
        "extra": [key.to_artifact_dict() for key in extra[:16]],
    }


def selected_logits_by_semantic_key(
    context: Any,
) -> dict[SemanticAtomKey, torch.Tensor]:
    selector = getattr(context, "select_logits_fp32", None)
    if not callable(selector):
        raise ParityContractError(
            "selected-logit extraction requires LossContext",
            code="qwen.parity.loss_context",
            context={"value_type": type(context).__name__},
        )
    logits, _targets, atoms = selector(token_types=None)
    if logits.ndim != 2 or int(logits.shape[0]) != len(atoms):
        raise ParityContractError(
            "selected logits do not align with semantic atoms",
            code="qwen.parity.logit_alignment",
            context={"logits_shape": list(logits.shape), "atom_count": len(atoms)},
        )
    result: dict[SemanticAtomKey, torch.Tensor] = {}
    for row_index, atom in enumerate(atoms):
        key = SemanticAtomKey.from_atom(atom)
        if key in result:
            raise ParityContractError(
                "selected logits contain a duplicate semantic key",
                code="qwen.parity.duplicate_atom_key",
                context={"key": key.to_artifact_dict()},
            )
        result[key] = logits[row_index].detach().float().cpu()
    return result


def compare_keyed_logits(
    left: Mapping[SemanticAtomKey, torch.Tensor],
    right: Mapping[SemanticAtomKey, torch.Tensor],
    *,
    rtol: float = BF16_RTOL,
    atol: float = BF16_ATOL,
) -> dict[str, Any]:
    left_keys = set(left)
    right_keys = set(right)
    missing = sorted(left_keys - right_keys)
    extra = sorted(right_keys - left_keys)
    if missing or extra:
        return {
            "passed": False,
            "key_alignment": False,
            "left_count": len(left_keys),
            "right_count": len(right_keys),
            "semantic_key_inventory_sha256": None,
            "missing": [item.to_artifact_dict() for item in missing[:16]],
            "extra": [item.to_artifact_dict() for item in extra[:16]],
            "rtol": rtol,
            "atol": atol,
            "rows": [],
            "max_abs_diff": None,
            "max_rel_diff": None,
        }
    rows: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    allclose = True
    for key in sorted(left_keys):
        comparison = compare_tensors(left[key], right[key], rtol=rtol, atol=atol)
        max_abs = max(max_abs, comparison.max_abs_diff)
        max_rel = max(max_rel, comparison.max_rel_diff)
        allclose = allclose and comparison.allclose
        target_id = key.token_id
        left_row = left[key]
        right_row = right[key]
        rows.append(
            {
                "key": key.to_artifact_dict(),
                "left_row_sha256": tensor_sha256(left_row),
                "right_row_sha256": tensor_sha256(right_row),
                "left_target_logit_fp32": float(left_row[target_id].item()),
                "right_target_logit_fp32": float(right_row[target_id].item()),
                **comparison.to_artifact_dict(),
            }
        )
    return {
        "passed": allclose,
        "key_alignment": True,
        "left_count": len(left_keys),
        "right_count": len(right_keys),
        "semantic_key_inventory_sha256": semantic_atom_key_inventory_sha256(
            tuple(sorted(left_keys))
        ),
        "missing": [],
        "extra": [],
        "rtol": rtol,
        "atol": atol,
        "row_count": len(rows),
        "rows": rows,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
    }


def compare_tensors(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> TensorComparison:
    if tuple(left.shape) != tuple(right.shape):
        raise ParityContractError(
            "tensor comparison shapes differ",
            code="qwen.parity.tensor_shape",
            context={"left": list(left.shape), "right": list(right.shape)},
        )
    left_fp32 = left.detach().float().cpu()
    right_fp32 = right.detach().float().cpu()
    if not bool(torch.isfinite(left_fp32).all()) or not bool(
        torch.isfinite(right_fp32).all()
    ):
        raise ParityContractError(
            "tensor comparison requires finite inputs",
            code="qwen.parity.tensor_nonfinite",
            context={},
        )
    diff = (left_fp32 - right_fp32).abs()
    denominator = torch.maximum(right_fp32.abs(), torch.full_like(right_fp32, atol))
    relative = diff / denominator
    return TensorComparison(
        allclose=bool(torch.allclose(left_fp32, right_fp32, rtol=rtol, atol=atol)),
        rtol=float(rtol),
        atol=float(atol),
        max_abs_diff=0.0 if diff.numel() == 0 else float(diff.max().item()),
        max_rel_diff=0.0 if relative.numel() == 0 else float(relative.max().item()),
        compared_value_count=int(diff.numel()),
    )


def snapshot_trainable_gradients(
    model: Any,
    *,
    gradient_provenance_dtype: str | None = None,
    expected_inventory: Mapping[str, Any] | None = None,
) -> tuple[GradientRecord, ...]:
    records: list[GradientRecord] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        gradient_dtype = None if grad is None else str(grad.dtype)
        if grad is not None:
            grad = grad.detach().float().cpu().clone()
            if not bool(torch.isfinite(grad).all()):
                raise ParityContractError(
                    "trainable gradient is non-finite",
                    code="qwen.parity.gradient_nonfinite",
                    context={"name": name},
                )
        records.append(
            GradientRecord(
                name=str(name),
                shape=tuple(int(item) for item in parameter.shape),
                parameter_dtype=str(parameter.dtype),
                grad=grad,
                gradient_dtype=gradient_dtype,
                gradient_provenance_dtype=gradient_provenance_dtype,
            )
        )
    if not records:
        raise ParityContractError(
            "parity run found no trainable parameters",
            code="qwen.parity.trainable_empty",
            context={},
        )
    if expected_inventory is not None:
        inventory = validate_concrete_trainable_inventory(expected_inventory)
        expected_rows = {
            str(row["name"]): row
            for row in _list_of_mappings(
                inventory["parameters"], "expected_inventory.parameters"
            )
        }
        observed = {record.name: record for record in records}
        missing = sorted(set(expected_rows) - set(observed))
        extra = sorted(set(observed) - set(expected_rows))
        mismatches: list[dict[str, Any]] = []
        aggregate_nonzero = False
        for name in sorted(set(expected_rows) & set(observed)):
            expected = expected_rows[name]
            record = observed[name]
            fields = []
            if list(record.shape) != expected["shape"]:
                fields.append("shape")
            if record.parameter_dtype != expected["parameter_storage_dtype"]:
                fields.append("parameter_storage_dtype")
            if record.resolved_gradient_dtype != expected["expected_gradient_dtype"]:
                fields.append("gradient_dtype")
            if record.gradient_provenance_dtype != expected["compute_provenance_dtype"]:
                fields.append("compute_provenance_dtype")
            if record.grad is None:
                fields.append("gradient_presence")
            else:
                aggregate_nonzero = aggregate_nonzero or bool(
                    torch.count_nonzero(record.grad).item()
                )
            if fields:
                mismatches.append({"name": name, "fields": fields})
        if missing or extra or mismatches or not aggregate_nonzero:
            raise ParityContractError(
                "arm gradient inventory does not match the exact v3 trainable surface",
                code="qwen.parity.gradient_coverage",
                context={
                    "missing": missing,
                    "extra": extra,
                    "mismatches": mismatches,
                    "aggregate_nonzero": aggregate_nonzero,
                },
            )
    return tuple(sorted(records, key=lambda record: record.name))


def _compare_gradient_inventories_v2(
    left: Sequence[GradientRecord],
    right: Sequence[GradientRecord],
) -> dict[str, Any]:
    left_by_name = _gradient_records_by_name(left, "left")
    right_by_name = _gradient_records_by_name(right, "right")
    missing_all = sorted(set(left_by_name) - set(right_by_name))
    extra_all = sorted(set(right_by_name) - set(left_by_name))
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    reason_counts = {
        "missing_name": len(missing_all),
        "extra_name": len(extra_all),
        "shape_mismatch": 0,
        "parameter_storage_dtype_mismatch": 0,
        "gradient_dtype_mismatch": 0,
        "gradient_provenance_dtype_mismatch": 0,
        "none_status_mismatch": 0,
        "both_none": 0,
        "both_zero": 0,
        "one_or_both_nonzero": 0,
        "tolerance_failure": 0,
        "no_nonzero_gradient_signal": 0,
    }
    status_counts = {
        "matched_parameter_count": 0,
        "compared_tensor_count": 0,
        "compared_value_count": 0,
        "both_none_count": 0,
        "none_status_mismatch_count": 0,
        "both_zero_count": 0,
        "one_or_both_nonzero_count": 0,
    }
    numerical_summary = {
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "max_left_l2_norm": 0.0,
        "max_right_l2_norm": 0.0,
        "max_abs_l2_norm_diff": 0.0,
        "max_rel_l2_norm_diff": 0.0,
    }
    per_parameter_dtype: dict[str, dict[str, Any]] = {}
    per_gradient_dtype: dict[str, int] = {}
    per_gradient_provenance_dtype: dict[str, int] = {}
    nonzero_gradient_signal = False
    for name in sorted(set(left_by_name) & set(right_by_name)):
        status_counts["matched_parameter_count"] += 1
        left_record = left_by_name[name]
        right_record = right_by_name[name]
        if left_record.shape != right_record.shape:
            reason_counts["shape_mismatch"] += 1
            failures.append(
                {
                    "name": name,
                    "reason": "shape",
                    "left": list(left_record.shape),
                    "right": list(right_record.shape),
                }
            )
            continue
        if left_record.parameter_dtype != right_record.parameter_dtype:
            reason_counts["parameter_storage_dtype_mismatch"] += 1
            failures.append(
                {
                    "name": name,
                    "reason": "dtype",
                    "left": left_record.parameter_dtype,
                    "right": right_record.parameter_dtype,
                }
            )
            continue
        left_gradient_dtype = left_record.resolved_gradient_dtype
        right_gradient_dtype = right_record.resolved_gradient_dtype
        if (left_record.grad is None) != (right_record.grad is None):
            reason_counts["none_status_mismatch"] += 1
            status_counts["none_status_mismatch_count"] += 1
            failures.append(
                {
                    "name": name,
                    "reason": "none_status",
                    "left_none": left_record.grad is None,
                    "right_none": right_record.grad is None,
                }
            )
            continue
        if left_gradient_dtype != right_gradient_dtype:
            reason_counts["gradient_dtype_mismatch"] += 1
            failures.append(
                {
                    "name": name,
                    "reason": "gradient_dtype",
                    "left": left_gradient_dtype,
                    "right": right_gradient_dtype,
                }
            )
            continue
        if (
            left_record.gradient_provenance_dtype
            != right_record.gradient_provenance_dtype
        ):
            reason_counts["gradient_provenance_dtype_mismatch"] += 1
            failures.append(
                {
                    "name": name,
                    "reason": "gradient_provenance_dtype",
                    "left": left_record.gradient_provenance_dtype,
                    "right": right_record.gradient_provenance_dtype,
                }
            )
            continue
        tolerance = gradient_tolerance(left_record.parameter_dtype)
        dtype_summary = per_parameter_dtype.setdefault(
            left_record.parameter_dtype,
            {
                "parameter_count": 0,
                "both_none_count": 0,
                "both_zero_count": 0,
                "nonzero_count": 0,
                "tolerance_failure_count": 0,
                "compared_value_count": 0,
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
                "max_abs_l2_norm_diff": 0.0,
                "max_rel_l2_norm_diff": 0.0,
                "rtol": tolerance["rtol"],
                "atol": tolerance["atol"],
            },
        )
        dtype_summary["parameter_count"] += 1
        gradient_dtype_key = left_gradient_dtype or "none"
        per_gradient_dtype[gradient_dtype_key] = (
            per_gradient_dtype.get(gradient_dtype_key, 0) + 1
        )
        provenance_key = left_record.gradient_provenance_dtype or "unspecified"
        per_gradient_provenance_dtype[provenance_key] = (
            per_gradient_provenance_dtype.get(provenance_key, 0) + 1
        )
        if left_record.grad is None:
            reason_counts["both_none"] += 1
            status_counts["both_none_count"] += 1
            dtype_summary["both_none_count"] += 1
            rows.append(
                {
                    "name": name,
                    "shape": list(left_record.shape),
                    "parameter_dtype": left_record.parameter_dtype,
                    "parameter_storage_dtype": left_record.parameter_dtype,
                    "gradient_dtype": None,
                    "gradient_provenance_dtype": (
                        left_record.gradient_provenance_dtype
                    ),
                    "grad_status": "none",
                    "parity_aligned": True,
                    "coverage_present": False,
                    "allclose": True,
                    "tolerance_selected_by": "parameter_storage_dtype",
                    **tolerance,
                }
            )
            continue
        left_nonzero = bool(torch.count_nonzero(left_record.grad).item())
        right_nonzero = bool(torch.count_nonzero(right_record.grad).item())
        any_nonzero = left_nonzero or right_nonzero
        nonzero_gradient_signal = nonzero_gradient_signal or any_nonzero
        if any_nonzero:
            reason_counts["one_or_both_nonzero"] += 1
            status_counts["one_or_both_nonzero_count"] += 1
            dtype_summary["nonzero_count"] += 1
        else:
            reason_counts["both_zero"] += 1
            status_counts["both_zero_count"] += 1
            dtype_summary["both_zero_count"] += 1
        comparison = compare_tensors(
            left_record.grad,
            right_record.grad,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
        )
        left_norm = float(torch.linalg.vector_norm(left_record.grad.float()).item())
        right_norm = float(torch.linalg.vector_norm(right_record.grad.float()).item())
        norm_abs_diff = abs(left_norm - right_norm)
        norm_rel_diff = norm_abs_diff / max(abs(right_norm), tolerance["atol"])
        status_counts["compared_tensor_count"] += 1
        status_counts["compared_value_count"] += comparison.compared_value_count
        dtype_summary["compared_value_count"] += comparison.compared_value_count
        for summary in (numerical_summary, dtype_summary):
            summary["max_abs_diff"] = max(
                float(summary["max_abs_diff"]), comparison.max_abs_diff
            )
            summary["max_rel_diff"] = max(
                float(summary["max_rel_diff"]), comparison.max_rel_diff
            )
            summary["max_abs_l2_norm_diff"] = max(
                float(summary["max_abs_l2_norm_diff"]), norm_abs_diff
            )
            summary["max_rel_l2_norm_diff"] = max(
                float(summary["max_rel_l2_norm_diff"]), norm_rel_diff
            )
        numerical_summary["max_left_l2_norm"] = max(
            numerical_summary["max_left_l2_norm"], left_norm
        )
        numerical_summary["max_right_l2_norm"] = max(
            numerical_summary["max_right_l2_norm"], right_norm
        )
        row = {
            "name": name,
            "shape": list(left_record.shape),
            "parameter_dtype": left_record.parameter_dtype,
            "parameter_storage_dtype": left_record.parameter_dtype,
            "gradient_dtype": left_gradient_dtype,
            "gradient_provenance_dtype": left_record.gradient_provenance_dtype,
            "grad_status": "finite",
            "left_zero": not left_nonzero,
            "right_zero": not right_nonzero,
            "parity_aligned": comparison.allclose,
            "coverage_present": True,
            "tolerance_selected_by": "parameter_storage_dtype",
            "left_l2_norm": left_norm,
            "right_l2_norm": right_norm,
            "abs_l2_norm_diff": norm_abs_diff,
            "rel_l2_norm_diff": norm_rel_diff,
            **comparison.to_artifact_dict(),
        }
        rows.append(row)
        if not comparison.allclose:
            reason_counts["tolerance_failure"] += 1
            dtype_summary["tolerance_failure_count"] += 1
            failures.append(
                {"name": name, "reason": "tolerance", **comparison.to_artifact_dict()}
            )
    if not nonzero_gradient_signal:
        reason_counts["no_nonzero_gradient_signal"] = 1
        failures.append(
            {
                "name": "<aggregate>",
                "reason": "no_nonzero_gradient_signal",
            }
        )
    parity_failure_reasons = {
        "shape",
        "dtype",
        "gradient_dtype",
        "gradient_provenance_dtype",
        "none_status",
        "tolerance",
    }
    parity_aligned = (
        not missing_all
        and not extra_all
        and not any(item["reason"] in parity_failure_reasons for item in failures)
    )
    trainable_coverage_passed = (
        not missing_all
        and not extra_all
        and reason_counts["both_none"] == 0
        and reason_counts["none_status_mismatch"] == 0
        and nonzero_gradient_signal
    )
    if (
        len(missing_all) > MAX_GRADIENT_DIAGNOSTIC_NAMES
        or len(extra_all) > MAX_GRADIENT_DIAGNOSTIC_NAMES
        or len(failures) > MAX_GRADIENT_DIAGNOSTIC_NAMES
        or len(rows) > MAX_GRADIENT_PARAMETER_SAMPLES
    ):
        raise ParityContractError(
            "gradient diagnostic inventory exceeds its bounded receipt capacity",
            code="qwen.parity.gradient_artifact_bound",
            context={
                "missing": len(missing_all),
                "extra": len(extra_all),
                "failures": len(failures),
                "parameters": len(rows),
            },
        )
    return {
        "passed": parity_aligned and trainable_coverage_passed,
        "parity_aligned": parity_aligned,
        "trainable_coverage_passed": trainable_coverage_passed,
        "coverage_policy": {
            "every_matched_trainable_requires_gradient": True,
            "aggregate_nonzero_gradient_required": True,
            "both_none_is_parity_aligned": True,
            "both_none_satisfies_coverage": False,
        },
        "left_count": len(left_by_name),
        "right_count": len(right_by_name),
        "matched_count": status_counts["matched_parameter_count"],
        "missing_name_count": len(missing_all),
        "extra_name_count": len(extra_all),
        "missing_names": missing_all,
        "extra_names": extra_all,
        "reason_counts": reason_counts,
        "failures": failures,
        "parameter_sample_count": len(rows),
        "parameter_samples_omitted": 0,
        "parameters": rows,
        "status_counts": status_counts,
        "per_parameter_storage_dtype": per_parameter_dtype,
        "per_gradient_dtype": per_gradient_dtype,
        "per_gradient_provenance_dtype": per_gradient_provenance_dtype,
        "numerical_summary": numerical_summary,
        "tolerance_policy": {
            "selected_by": "parameter_storage_dtype",
            "frozen_bands": frozen_tolerances()["gradients_by_parameter_dtype"],
            "gradient_provenance_is_diagnostic_only": True,
        },
        "nonzero_gradient_signal": nonzero_gradient_signal,
    }


def compare_gradient_inventories(
    left: Sequence[GradientRecord],
    right: Sequence[GradientRecord],
) -> dict[str, Any]:
    """Compare BF16-compute-derived gradients after detached FP32 upcast."""

    left_by_name = _gradient_records_by_name(left, "left")
    right_by_name = _gradient_records_by_name(right, "right")
    missing = sorted(set(left_by_name) - set(right_by_name))
    extra = sorted(set(right_by_name) - set(left_by_name))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    nonzero_gradient_signal = False
    compared_value_count = 0
    max_abs_diff = 0.0
    for name in sorted(set(left_by_name) & set(right_by_name)):
        left_record = left_by_name[name]
        right_record = right_by_name[name]
        if left_record.shape != right_record.shape:
            failures.append({"name": name, "reason": "shape_mismatch"})
            continue
        if left_record.parameter_dtype != right_record.parameter_dtype:
            failures.append(
                {"name": name, "reason": "parameter_storage_dtype_mismatch"}
            )
            continue
        if (
            left_record.gradient_provenance_dtype != BF16_COMPUTE_PROVENANCE_DTYPE
            or right_record.gradient_provenance_dtype != BF16_COMPUTE_PROVENANCE_DTYPE
        ):
            failures.append({"name": name, "reason": "compute_provenance_mismatch"})
            continue
        if left_record.grad is None or right_record.grad is None:
            failures.append({"name": name, "reason": "none_status"})
            continue
        if left_record.resolved_gradient_dtype != right_record.resolved_gradient_dtype:
            failures.append({"name": name, "reason": "gradient_dtype_mismatch"})
            continue
        left_fp32 = left_record.grad.detach().float().cpu()
        right_fp32 = right_record.grad.detach().float().cpu()
        if not bool(torch.isfinite(left_fp32).all()) or not bool(
            torch.isfinite(right_fp32).all()
        ):
            failures.append({"name": name, "reason": "nonfinite_gradient"})
            continue
        comparison = compare_tensors(
            left_fp32,
            right_fp32,
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )
        left_nonzero = bool(torch.count_nonzero(left_fp32).item())
        right_nonzero = bool(torch.count_nonzero(right_fp32).item())
        nonzero_gradient_signal = (
            nonzero_gradient_signal or left_nonzero or right_nonzero
        )
        compared_value_count += comparison.compared_value_count
        max_abs_diff = max(max_abs_diff, comparison.max_abs_diff)
        row = {
            "name": name,
            "shape": list(left_record.shape),
            "parameter_storage_dtype": left_record.parameter_dtype,
            "gradient_dtype": left_record.resolved_gradient_dtype,
            "compute_provenance_dtype": BF16_COMPUTE_PROVENANCE_DTYPE,
            "comparison_dtype": FP32_COMPARISON_DTYPE,
            **comparison.to_artifact_dict(),
        }
        rows.append(row)
        if not comparison.allclose:
            failures.append({"name": name, "reason": "tolerance_failure"})
    coverage_passed = (
        not missing
        and not extra
        and len(rows) == len(left_by_name) == len(right_by_name)
        and not any(
            row["reason"]
            in {
                "shape_mismatch",
                "parameter_storage_dtype_mismatch",
                "gradient_dtype_mismatch",
                "compute_provenance_mismatch",
                "none_status",
                "nonfinite_gradient",
            }
            for row in failures
        )
        and nonzero_gradient_signal
    )
    if not nonzero_gradient_signal:
        failures.append({"name": "<aggregate>", "reason": "no_nonzero_gradient_signal"})
    parity_passed = not any(row["reason"] == "tolerance_failure" for row in failures)
    artifact = {
        "passed": coverage_passed and parity_passed,
        "coverage_passed": coverage_passed,
        "parity_passed": parity_passed,
        "source_compute_dtype": BF16_COMPUTE_PROVENANCE_DTYPE,
        "comparison_dtype": FP32_COMPARISON_DTYPE,
        "rtol": BF16_RTOL,
        "atol": BF16_ATOL,
        "left_count": len(left_by_name),
        "right_count": len(right_by_name),
        "matched_count": len(set(left_by_name) & set(right_by_name)),
        "missing_names": missing,
        "extra_names": extra,
        "nonzero_gradient_signal": nonzero_gradient_signal,
        "compared_value_count": compared_value_count,
        "max_abs_diff": max_abs_diff,
        "parameters": rows,
        "failures": failures,
    }
    if (
        max(len(missing), len(extra), len(failures), len(rows))
        > MAX_GRADIENT_PARAMETER_SAMPLES
    ):
        raise ParityContractError(
            "gradient diagnostic inventory exceeds its bounded receipt capacity",
            code="qwen.parity.gradient_artifact_bound",
            context={},
        )
    canonical_json_bytes(artifact)
    return artifact


def compare_packed_gradient_repeat(
    primary: Sequence[GradientRecord],
    repeat: Sequence[GradientRecord],
) -> dict[str, Any]:
    """Apply the fixed same-packed measurability gate in detached FP32."""

    try:
        primary_by_name = _gradient_records_by_name(primary, "packed_primary")
        repeat_by_name = _gradient_records_by_name(repeat, "packed_repeat")
    except ParityContractError as exc:
        return {
            "status": "unmeasurable",
            "passed": False,
            "coverage_passed": False,
            "comparison_dtype": FP32_COMPARISON_DTYPE,
            "max_abs_threshold": PACKED_REPEAT_MAX_ABS,
            "max_abs_diff": None,
            "compared_parameter_count": 0,
            "compared_value_count": 0,
            "failures": [bounded_failure(exc)],
        }
    missing = sorted(set(primary_by_name) - set(repeat_by_name))
    extra = sorted(set(repeat_by_name) - set(primary_by_name))
    failures: list[dict[str, Any]] = []
    if missing:
        failures.append({"reason": "missing_names", "names": missing})
    if extra:
        failures.append({"reason": "extra_names", "names": extra})
    max_abs = 0.0
    compared_parameters = 0
    compared_values = 0
    primary_has_nonzero = False
    repeat_has_nonzero = False
    for name in sorted(set(primary_by_name) & set(repeat_by_name)):
        left = primary_by_name[name]
        right = repeat_by_name[name]
        structural = {
            "shape": (left.shape, right.shape),
            "parameter_storage_dtype": (
                left.parameter_dtype,
                right.parameter_dtype,
            ),
            "gradient_dtype": (
                left.resolved_gradient_dtype,
                right.resolved_gradient_dtype,
            ),
            "compute_provenance_dtype": (
                left.gradient_provenance_dtype,
                right.gradient_provenance_dtype,
            ),
        }
        mismatch = [field for field, pair in structural.items() if pair[0] != pair[1]]
        if mismatch:
            failures.append(
                {"reason": "structural_mismatch", "name": name, "fields": mismatch}
            )
            continue
        if left.grad is None or right.grad is None:
            failures.append({"reason": "missing_gradient", "name": name})
            continue
        left_fp32 = left.grad.detach().float().cpu()
        right_fp32 = right.grad.detach().float().cpu()
        if not bool(torch.isfinite(left_fp32).all()) or not bool(
            torch.isfinite(right_fp32).all()
        ):
            failures.append({"reason": "nonfinite_gradient", "name": name})
            continue
        difference = (left_fp32 - right_fp32).abs()
        primary_has_nonzero = primary_has_nonzero or bool(
            torch.count_nonzero(left_fp32).item()
        )
        repeat_has_nonzero = repeat_has_nonzero or bool(
            torch.count_nonzero(right_fp32).item()
        )
        max_abs = max(
            max_abs,
            0.0 if difference.numel() == 0 else float(difference.max().item()),
        )
        compared_parameters += 1
        compared_values += int(difference.numel())
    if not primary_has_nonzero or not repeat_has_nonzero:
        failures.append(
            {
                "reason": "no_nonzero_gradient_signal",
                "packed_primary_nonzero": primary_has_nonzero,
                "packed_repeat_nonzero": repeat_has_nonzero,
            }
        )
    coverage_passed = (
        not failures
        and compared_parameters == len(primary_by_name) == len(repeat_by_name)
        and compared_parameters == EXPECTED_TRAINABLE_PARAMETER_COUNT
    )
    if not coverage_passed:
        status = "unmeasurable"
        passed = False
    elif max_abs > PACKED_REPEAT_MAX_ABS:
        status = "unmeasurable"
        passed = False
        failures.append(
            {
                "reason": "max_abs_exceeded",
                "observed": max_abs,
                "maximum": PACKED_REPEAT_MAX_ABS,
            }
        )
    else:
        status = "measurable"
        passed = True
    artifact = {
        "status": status,
        "passed": passed,
        "coverage_passed": coverage_passed,
        "comparison_dtype": FP32_COMPARISON_DTYPE,
        "max_abs_threshold": PACKED_REPEAT_MAX_ABS,
        "max_abs_diff": max_abs if coverage_passed else None,
        "compared_parameter_count": compared_parameters,
        "compared_value_count": compared_values,
        "failures": failures,
    }
    canonical_json_bytes(artifact)
    return artifact


def validate_gradient_comparison_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the compact v3 FP32-computed BF16-derived gradient gate."""

    _expect_exact_keys(
        artifact,
        {
            "passed",
            "coverage_passed",
            "parity_passed",
            "source_compute_dtype",
            "comparison_dtype",
            "rtol",
            "atol",
            "left_count",
            "right_count",
            "matched_count",
            "missing_names",
            "extra_names",
            "nonzero_gradient_signal",
            "compared_value_count",
            "max_abs_diff",
            "parameters",
            "failures",
        },
        owner="gradient_comparison_v3",
    )
    if (
        artifact["source_compute_dtype"] != BF16_COMPUTE_PROVENANCE_DTYPE
        or artifact["comparison_dtype"] != FP32_COMPARISON_DTYPE
        or float(artifact["rtol"]) != BF16_RTOL
        or float(artifact["atol"]) != BF16_ATOL
    ):
        raise ParityContractError(
            "v3 gradient comparison dtype or tolerance contract drifted",
            code="qwen.parity.gradient_v3_header",
            context={},
        )
    for field in (
        "passed",
        "coverage_passed",
        "parity_passed",
        "nonzero_gradient_signal",
    ):
        if not isinstance(artifact[field], bool):
            raise ParityContractError(
                "v3 gradient comparison status is not boolean",
                code="qwen.parity.gradient_v3_status",
                context={"field": field},
            )
    left_count = _non_negative_int(artifact["left_count"], "gradient.left_count")
    right_count = _non_negative_int(artifact["right_count"], "gradient.right_count")
    matched_count = _non_negative_int(
        artifact["matched_count"], "gradient.matched_count"
    )
    missing = _text_list_allow_empty(artifact["missing_names"], "gradient.missing")
    extra = _text_list_allow_empty(artifact["extra_names"], "gradient.extra")
    rows = _list_of_mappings(artifact["parameters"], "gradient.parameters")
    failures = _list_of_mappings(artifact["failures"], "gradient.failures")
    if (
        max(len(missing), len(extra), len(rows), len(failures))
        > MAX_GRADIENT_PARAMETER_SAMPLES
    ):
        raise ParityContractError(
            "v3 gradient artifact exceeds its bounded capacity",
            code="qwen.parity.gradient_artifact_bound",
            context={},
        )
    names: list[str] = []
    compared_values = 0
    max_abs = 0.0
    row_allclose = True
    for row in rows:
        _expect_exact_keys(
            row,
            {
                "name",
                "shape",
                "parameter_storage_dtype",
                "gradient_dtype",
                "compute_provenance_dtype",
                "comparison_dtype",
                "allclose",
                "rtol",
                "atol",
                "max_abs_diff",
                "max_rel_diff",
                "compared_value_count",
            },
            owner="gradient.parameter_v3",
        )
        names.append(_non_empty_text(row["name"], "gradient.parameter.name"))
        if (
            row["compute_provenance_dtype"] != BF16_COMPUTE_PROVENANCE_DTYPE
            or row["comparison_dtype"] != FP32_COMPARISON_DTYPE
            or float(row["rtol"]) != BF16_RTOL
            or float(row["atol"]) != BF16_ATOL
        ):
            raise ParityContractError(
                "v3 gradient parameter row uses a forbidden dtype/tolerance policy",
                code="qwen.parity.gradient_v3_row",
                context={"name": row["name"]},
            )
        _positive_int_list(row["shape"], "gradient.parameter.shape")
        _non_empty_text(
            row["parameter_storage_dtype"], "gradient.parameter.storage_dtype"
        )
        _non_empty_text(row["gradient_dtype"], "gradient.parameter.gradient_dtype")
        compared_values += _non_negative_int(
            row["compared_value_count"], "gradient.parameter.compared_value_count"
        )
        max_abs = max(
            max_abs,
            _finite_number(row["max_abs_diff"], "gradient.parameter.max_abs_diff"),
        )
        row_allclose = row_allclose and row["allclose"] is True
    if names != sorted(set(names)):
        raise ParityContractError(
            "v3 gradient parameter rows must be unique and sorted",
            code="qwen.parity.gradient_v3_inventory",
            context={},
        )
    if expected_inventory is not None:
        concrete = validate_concrete_trainable_inventory(expected_inventory)
        expected_rows = _list_of_mappings(
            concrete["parameters"], "expected_inventory.parameters"
        )
        expected_by_name = {
            _non_empty_text(row["name"], "expected_inventory.parameter.name"): row
            for row in expected_rows
        }
        expected_names = sorted(expected_by_name)
        if (
            names != expected_names
            or left_count != EXPECTED_TRAINABLE_PARAMETER_COUNT
            or right_count != EXPECTED_TRAINABLE_PARAMETER_COUNT
            or matched_count != EXPECTED_TRAINABLE_PARAMETER_COUNT
            or len(rows) != EXPECTED_TRAINABLE_PARAMETER_COUNT
        ):
            raise ParityContractError(
                "v3 gradient comparison is not the exact concrete 589 inventory",
                code="qwen.parity.gradient_v3_inventory_binding",
                context={
                    "expected_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
                    "observed_row_count": len(rows),
                    "left_count": left_count,
                    "right_count": right_count,
                    "matched_count": matched_count,
                },
            )
        for row in rows:
            expected = expected_by_name[str(row["name"])]
            if (
                row["shape"] != expected["shape"]
                or row["parameter_storage_dtype"] != expected["parameter_storage_dtype"]
                or row["gradient_dtype"] != expected["expected_gradient_dtype"]
                or row["compute_provenance_dtype"]
                != expected["compute_provenance_dtype"]
            ):
                raise ParityContractError(
                    "v3 gradient comparison row differs from the concrete inventory",
                    code="qwen.parity.gradient_v3_inventory_binding",
                    context={"name": row["name"]},
                )
    structural_failure = any(
        failure.get("reason") != "tolerance_failure" for failure in failures
    )
    tolerance_failure = any(
        failure.get("reason") == "tolerance_failure" for failure in failures
    )
    expected_coverage = (
        not missing
        and not extra
        and not structural_failure
        and artifact["nonzero_gradient_signal"] is True
        and len(rows) == left_count == right_count == matched_count
    )
    expected_parity = row_allclose and not tolerance_failure
    if (
        artifact["coverage_passed"] is not expected_coverage
        or artifact["parity_passed"] is not expected_parity
        or artifact["passed"] is not (expected_coverage and expected_parity)
        or artifact["compared_value_count"] != compared_values
        or float(artifact["max_abs_diff"]) != max_abs
    ):
        raise ParityContractError(
            "v3 gradient comparison aggregate contradicts its rows",
            code="qwen.parity.gradient_v3_aggregate",
            context={},
        )
    canonical_json_bytes(artifact)
    return dict(artifact)


def validate_packed_gradient_repeat_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _expect_exact_keys(
        artifact,
        {
            "status",
            "passed",
            "coverage_passed",
            "comparison_dtype",
            "max_abs_threshold",
            "max_abs_diff",
            "compared_parameter_count",
            "compared_value_count",
            "failures",
        },
        owner="packed_repeat",
    )
    failures = _list_of_mappings(artifact["failures"], "packed_repeat.failures")
    compared_parameter_count = _non_negative_int(
        artifact["compared_parameter_count"], "packed_repeat.parameter_count"
    )
    compared_value_count = _non_negative_int(
        artifact["compared_value_count"], "packed_repeat.value_count"
    )
    expected_value_count: int | None = None
    if expected_inventory is not None:
        concrete = validate_concrete_trainable_inventory(expected_inventory)
        expected_value_count = sum(
            math.prod(
                _positive_int_list(
                    row["shape"], "packed_repeat.expected_inventory.shape"
                )
            )
            for row in _list_of_mappings(
                concrete["parameters"], "packed_repeat.expected_inventory.parameters"
            )
        )
    if (
        artifact["comparison_dtype"] != FP32_COMPARISON_DTYPE
        or float(artifact["max_abs_threshold"]) != PACKED_REPEAT_MAX_ABS
        or artifact["status"] not in {"measurable", "unmeasurable"}
    ):
        raise ParityContractError(
            "same-packed repeat header drifted",
            code="qwen.parity.repeat_header",
            context={},
        )
    if artifact["status"] == "measurable":
        if (
            artifact["passed"] is not True
            or artifact["coverage_passed"] is not True
            or failures
            or compared_parameter_count != EXPECTED_TRAINABLE_PARAMETER_COUNT
            or compared_value_count <= 0
            or (
                expected_value_count is not None
                and compared_value_count != expected_value_count
            )
            or artifact["max_abs_diff"] is None
            or float(artifact["max_abs_diff"]) > PACKED_REPEAT_MAX_ABS
        ):
            raise ParityContractError(
                "measurable same-packed repeat contradicts its evidence",
                code="qwen.parity.repeat_aggregate",
                context={},
            )
    elif artifact["passed"] is not False or not failures:
        raise ParityContractError(
            "unmeasurable same-packed repeat requires bounded failure evidence",
            code="qwen.parity.repeat_aggregate",
            context={},
        )
    canonical_json_bytes(artifact)
    return dict(artifact)


def _validate_tensor_comparison_artifact(
    value: Mapping[str, Any],
    *,
    owner: str,
    status_field: str = "allclose",
) -> None:
    _expect_exact_keys(
        value,
        {
            status_field,
            "rtol",
            "atol",
            "max_abs_diff",
            "max_rel_diff",
            "compared_value_count",
        },
        owner=owner,
    )
    if not isinstance(value[status_field], bool):
        raise ParityContractError(
            "tensor comparison status must be boolean",
            code="qwen.parity.failure_comparison_artifact",
            context={"owner": owner},
        )
    for field in ("rtol", "atol", "max_abs_diff", "max_rel_diff"):
        observed = value[field]
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ParityContractError(
                "tensor comparison numerical summary is invalid",
                code="qwen.parity.failure_comparison_artifact",
                context={"owner": owner, "field": field},
            )
        if not math.isfinite(float(observed)) or float(observed) < 0.0:
            raise ParityContractError(
                "tensor comparison numerical summary is invalid",
                code="qwen.parity.failure_comparison_artifact",
                context={"owner": owner, "field": field},
            )
    _non_negative_int(value["compared_value_count"], f"{owner}.compared_value_count")


def _validate_semantic_atom_comparison_artifact(
    value: Mapping[str, Any],
    *,
    expected_count: int | None = None,
    expected_keys: Sequence[SemanticAtomKey] | None = None,
) -> None:
    _expect_exact_keys(
        value,
        {
            "passed",
            "left_count",
            "right_count",
            "left_key_inventory_sha256",
            "right_key_inventory_sha256",
            "missing",
            "extra",
        },
        owner="semantic_atom_comparison",
    )
    if not isinstance(value["passed"], bool):
        raise ParityContractError(
            "semantic-atom comparison status is invalid",
            code="qwen.parity.failure_comparison_artifact",
            context={},
        )
    left_count = _non_negative_int(value["left_count"], "semantic_atoms.left_count")
    right_count = _non_negative_int(value["right_count"], "semantic_atoms.right_count")
    missing = _list_of_mappings(value["missing"], "semantic_atoms.missing")
    extra = _list_of_mappings(value["extra"], "semantic_atoms.extra")
    if len(missing) > 16 or len(extra) > 16:
        raise ParityContractError(
            "semantic-atom diagnostics exceed their bound",
            code="qwen.parity.failure_comparison_artifact",
            context={},
        )
    missing_keys = [_key_from_mapping(row) for row in missing]
    extra_keys = [_key_from_mapping(row) for row in extra]
    left_digest = _sha256_text(
        value["left_key_inventory_sha256"],
        "semantic_atoms.left_key_inventory_sha256",
    )
    right_digest = _sha256_text(
        value["right_key_inventory_sha256"],
        "semantic_atoms.right_key_inventory_sha256",
    )
    expected_digest = (
        semantic_atom_key_inventory_sha256(tuple(expected_keys))
        if expected_keys is not None
        else None
    )
    if (
        missing_keys != sorted(set(missing_keys))
        or extra_keys != sorted(set(extra_keys))
        or bool(value["passed"]) != (not missing and not extra)
        or (value["passed"] and left_count != right_count)
        or (not value["passed"] and not (missing or extra))
        or (
            expected_count is not None
            and (left_count != expected_count or right_count != expected_count)
        )
        or (
            expected_digest is not None
            and (left_digest != expected_digest or right_digest != expected_digest)
        )
    ):
        raise ParityContractError(
            "semantic-atom comparison is internally inconsistent",
            code="qwen.parity.failure_comparison_artifact",
            context={},
        )


def _validate_keyed_logits_comparison_artifact(
    value: Mapping[str, Any],
    *,
    expected_row_count: int | None = None,
    expected_keys: Sequence[SemanticAtomKey] | None = None,
) -> None:
    common = {
        "passed",
        "key_alignment",
        "left_count",
        "right_count",
        "semantic_key_inventory_sha256",
        "missing",
        "extra",
        "rtol",
        "atol",
        "rows",
        "max_abs_diff",
        "max_rel_diff",
    }
    aligned = value.get("key_alignment") is True
    _expect_exact_keys(
        value,
        common | ({"row_count"} if aligned else set()),
        owner="keyed_logits_comparison",
    )
    if not isinstance(value["passed"], bool) or not isinstance(
        value["key_alignment"], bool
    ):
        raise ParityContractError(
            "keyed-logit comparison status is invalid",
            code="qwen.parity.failure_comparison_artifact",
            context={},
        )
    if float(value["rtol"]) != BF16_RTOL or float(value["atol"]) != BF16_ATOL:
        raise ParityContractError(
            "keyed-logit comparison tolerance drifted",
            code="qwen.parity.failure_comparison_artifact",
            context={},
        )
    missing = _list_of_mappings(value["missing"], "keyed_logits.missing")
    extra = _list_of_mappings(value["extra"], "keyed_logits.extra")
    rows = _list_of_mappings(value["rows"], "keyed_logits.rows")
    left_count = _non_negative_int(value["left_count"], "keyed_logits.left_count")
    right_count = _non_negative_int(value["right_count"], "keyed_logits.right_count")
    if expected_row_count is not None and (
        left_count != expected_row_count or right_count != expected_row_count
    ):
        raise ParityContractError(
            "keyed-logit coverage differs from the authenticated semantic inventory",
            code="qwen.parity.semantic_logit_coverage",
            context={
                "expected": expected_row_count,
                "left_count": left_count,
                "right_count": right_count,
            },
        )
    if (
        len(missing) > 16
        or len(extra) > 16
        or len(rows) > MAX_GRADIENT_PARAMETER_SAMPLES
    ):
        raise ParityContractError(
            "keyed-logit diagnostics exceed their bound",
            code="qwen.parity.failure_comparison_artifact",
            context={},
        )
    for row in (*missing, *extra):
        _key_from_mapping(row)
    if not aligned:
        if (
            value["passed"]
            or rows
            or not (missing or extra)
            or left_count - len(missing) != right_count - len(extra)
            or value["semantic_key_inventory_sha256"] is not None
        ):
            raise ParityContractError(
                "unaligned keyed-logit comparison is inconsistent",
                code="qwen.parity.failure_comparison_artifact",
                context={},
            )
        if value["max_abs_diff"] is not None or value["max_rel_diff"] is not None:
            raise ParityContractError(
                "unaligned keyed-logit comparison contains numerical claims",
                code="qwen.parity.failure_comparison_artifact",
                context={},
            )
        return
    if (
        missing
        or extra
        or value["row_count"] != len(rows)
        or left_count != len(rows)
        or right_count != len(rows)
    ):
        raise ParityContractError(
            "aligned keyed-logit inventory is inconsistent",
            code="qwen.parity.failure_comparison_artifact",
            context={},
        )
    observed_keys: list[SemanticAtomKey] = []
    max_abs = 0.0
    max_rel = 0.0
    allclose = True
    row_keys = {
        "key",
        "left_row_sha256",
        "right_row_sha256",
        "left_target_logit_fp32",
        "right_target_logit_fp32",
        "allclose",
        "rtol",
        "atol",
        "max_abs_diff",
        "max_rel_diff",
        "compared_value_count",
    }
    for index, row in enumerate(rows):
        _expect_exact_keys(row, row_keys, owner="keyed_logits.row")
        observed_keys.append(
            _key_from_mapping(_mapping(row["key"], "keyed_logits.key"))
        )
        _sha256_text(row["left_row_sha256"], "keyed_logits.left_row_sha256")
        _sha256_text(row["right_row_sha256"], "keyed_logits.right_row_sha256")
        for field in ("left_target_logit_fp32", "right_target_logit_fp32"):
            if not math.isfinite(float(row[field])):
                raise ParityContractError(
                    "keyed-logit target scalar is invalid",
                    code="qwen.parity.failure_comparison_artifact",
                    context={"row": index, "field": field},
                )
        comparison = {
            key: row[key]
            for key in row_keys
            if key
            not in {
                "key",
                "left_row_sha256",
                "right_row_sha256",
                "left_target_logit_fp32",
                "right_target_logit_fp32",
            }
        }
        _validate_tensor_comparison_artifact(comparison, owner="keyed_logits.row")
        if float(row["rtol"]) != BF16_RTOL or float(row["atol"]) != BF16_ATOL:
            raise ParityContractError(
                "keyed-logit row tolerance drifted",
                code="qwen.parity.failure_comparison_artifact",
                context={"row": index},
            )
        allclose = allclose and bool(row["allclose"])
        max_abs = max(max_abs, float(row["max_abs_diff"]))
        max_rel = max(max_rel, float(row["max_rel_diff"]))
    if observed_keys != sorted(set(observed_keys)):
        raise ParityContractError(
            "keyed-logit row keys are duplicate or unordered",
            code="qwen.parity.failure_comparison_artifact",
            context={},
        )
    observed_digest = semantic_atom_key_inventory_sha256(tuple(observed_keys))
    published_digest = _sha256_text(
        value["semantic_key_inventory_sha256"],
        "keyed_logits.semantic_key_inventory_sha256",
    )
    expected_key_tuple = tuple(expected_keys) if expected_keys is not None else None
    if published_digest != observed_digest or (
        expected_key_tuple is not None and observed_keys != list(expected_key_tuple)
    ):
        raise ParityContractError(
            "keyed-logit semantic keys differ from the authenticated plan inventory",
            code="qwen.parity.semantic_logit_key_binding",
            context={},
        )
    if (
        value["passed"] is not allclose
        or float(value["max_abs_diff"]) != max_abs
        or float(value["max_rel_diff"]) != max_rel
    ):
        raise ParityContractError(
            "keyed-logit aggregate is inconsistent with its rows",
            code="qwen.parity.failure_comparison_artifact",
            context={},
        )


def _validate_total_loss_comparison_artifact(value: Mapping[str, Any]) -> None:
    _validate_tensor_comparison_artifact(
        value, owner="total_loss", status_field="passed"
    )
    if (
        float(value["rtol"]) != BF16_RTOL
        or float(value["atol"]) != BF16_ATOL
        or value["compared_value_count"] != 1
    ):
        raise ParityContractError(
            "total-loss comparison contract drifted",
            code="qwen.parity.failure_comparison_artifact",
            context={},
        )


def _total_loss_comparison_from_serialized_arms(
    left_arm: Mapping[str, Any],
    right_arm: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = compare_tensors(
        torch.tensor(
            _finite_number(left_arm.get("total_loss_fp32"), "left_arm.total_loss"),
            dtype=torch.float32,
        ),
        torch.tensor(
            _finite_number(right_arm.get("total_loss_fp32"), "right_arm.total_loss"),
            dtype=torch.float32,
        ),
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    ).to_artifact_dict()
    comparison["passed"] = comparison.pop("allclose")
    return comparison


def _validate_negative_discriminator_artifact(
    value: Mapping[str, Any], *, arms: Mapping[str, Any]
) -> None:
    _expect_exact_keys(
        value,
        {
            "detected",
            "boundary_changed",
            "clean_boundaries",
            "negative_boundaries",
            "forward_detected_by",
            "diagnostic_detected_by",
            "gradient_only_is_insufficient",
            "attestation",
            "supervised_logits",
            "total_loss",
            "gradients",
        },
        owner="negative_discriminator",
    )
    clean = _int_list(value["clean_boundaries"], "negative.clean_boundaries")
    negative = _int_list(value["negative_boundaries"], "negative.negative_boundaries")
    if (
        len(clean) < 2
        or len(negative) < 2
        or clean[0] != 0
        or negative[0] != 0
        or any(right <= left for left, right in zip(clean, clean[1:]))
        or any(right <= left for left, right in zip(negative, negative[1:]))
    ):
        raise ParityContractError(
            "negative discriminator boundaries are invalid",
            code="qwen.parity.failure_negative_artifact",
            context={},
        )
    logits = _mapping(value["supervised_logits"], "negative.supervised_logits")
    total_loss = _mapping(value["total_loss"], "negative.total_loss")
    gradients = _mapping(value["gradients"], "negative.gradients")
    _validate_keyed_logits_comparison_artifact(logits)
    _validate_tensor_comparison_artifact(total_loss, owner="negative.total_loss")
    _validate_gradient_comparison_artifact(gradients)
    expected_forward = [
        name
        for name, aligned in (
            ("supervised_logits", bool(logits["passed"])),
            ("total_loss", bool(total_loss["allclose"])),
        )
        if not aligned
    ]
    expected_diagnostic = [] if gradients["passed"] else ["trainable_gradients"]
    boundary_changed = clean != negative
    if (
        value["boundary_changed"] is not boundary_changed
        or value["forward_detected_by"] != expected_forward
        or value["diagnostic_detected_by"] != expected_diagnostic
        or value["detected"] is not (boundary_changed and bool(expected_forward))
        or value["gradient_only_is_insufficient"] is not True
    ):
        raise ParityContractError(
            "negative discriminator aggregate is inconsistent",
            code="qwen.parity.failure_negative_artifact",
            context={},
        )
    attestation = _mapping(value["attestation"], "negative.attestation")
    _expect_exact_keys(
        attestation,
        {
            "status",
            "expected_clean_boundaries",
            "observed_negative_boundaries",
            "boundary_mismatch_detected",
            "proof_disabled",
            "executed_negative_varlen_receipt",
        },
        owner="negative.attestation",
    )
    negative_arm = _mapping(arms.get("packed_merged_boundary_negative"), "negative.arm")
    forwards = _list_of_mappings(negative_arm["forward_receipts"], "negative.forwards")
    executed = _mapping(
        attestation["executed_negative_varlen_receipt"], "negative.executed_varlen"
    )
    arm_executed = _mapping(forwards[0].get("fa2_varlen"), "negative.arm_varlen")
    if (
        attestation["status"] != "rejected_against_frozen_clean_boundary"
        or attestation["expected_clean_boundaries"] != clean
        or attestation["observed_negative_boundaries"] != negative
        or attestation["boundary_mismatch_detected"] is not boundary_changed
        or attestation["proof_disabled"] is not True
        or executed != arm_executed
        or executed.get("segment_boundaries") != negative
        or executed.get("proof") is not None
    ):
        raise ParityContractError(
            "negative discriminator attestation is inconsistent",
            code="qwen.parity.failure_negative_artifact",
            context={},
        )


def _validate_gradient_comparison_artifact(value: Mapping[str, Any]) -> None:
    required = {
        "passed",
        "parity_aligned",
        "trainable_coverage_passed",
        "coverage_policy",
        "left_count",
        "right_count",
        "matched_count",
        "missing_name_count",
        "extra_name_count",
        "missing_names",
        "extra_names",
        "reason_counts",
        "failures",
        "parameter_sample_count",
        "parameter_samples_omitted",
        "parameters",
        "status_counts",
        "per_parameter_storage_dtype",
        "per_gradient_dtype",
        "per_gradient_provenance_dtype",
        "numerical_summary",
        "tolerance_policy",
        "nonzero_gradient_signal",
    }
    _expect_exact_keys(value, required, owner="gradient_comparison")
    for field in (
        "passed",
        "parity_aligned",
        "trainable_coverage_passed",
        "nonzero_gradient_signal",
    ):
        if not isinstance(value[field], bool):
            raise ParityContractError(
                "gradient comparison status must be boolean",
                code="qwen.parity.gradient_artifact",
                context={"field": field},
            )
    for field in (
        "left_count",
        "right_count",
        "matched_count",
        "missing_name_count",
        "extra_name_count",
        "parameter_sample_count",
        "parameter_samples_omitted",
    ):
        _non_negative_int(value[field], f"gradient_comparison.{field}")
    missing = value["missing_names"]
    extra = value["extra_names"]
    failures = value["failures"]
    parameters = value["parameters"]
    if not all(
        isinstance(item, list) for item in (missing, extra, failures, parameters)
    ) or any(
        len(items) > maximum
        for items, maximum in (
            (missing, MAX_GRADIENT_DIAGNOSTIC_NAMES),
            (extra, MAX_GRADIENT_DIAGNOSTIC_NAMES),
            (failures, MAX_GRADIENT_DIAGNOSTIC_NAMES),
            (parameters, MAX_GRADIENT_PARAMETER_SAMPLES),
        )
    ):
        raise ParityContractError(
            "gradient comparison bounded diagnostics are invalid",
            code="qwen.parity.gradient_artifact",
            context={},
        )
    if (
        any(not isinstance(name, str) or not name for name in (*missing, *extra))
        or missing != sorted(set(missing))
        or extra != sorted(set(extra))
        or value["missing_name_count"] != len(missing)
        or value["extra_name_count"] != len(extra)
        or value["matched_count"] != value["left_count"] - len(missing)
        or value["matched_count"] != value["right_count"] - len(extra)
        or value["parameter_sample_count"] != len(parameters)
        or value["parameter_samples_omitted"] != 0
    ):
        raise ParityContractError(
            "gradient comparison inventory accounting is inconsistent",
            code="qwen.parity.gradient_artifact",
            context={},
        )
    coverage_policy = _mapping(value["coverage_policy"], "coverage_policy")
    if coverage_policy != {
        "every_matched_trainable_requires_gradient": True,
        "aggregate_nonzero_gradient_required": True,
        "both_none_is_parity_aligned": True,
        "both_none_satisfies_coverage": False,
    }:
        raise ParityContractError(
            "gradient coverage policy drifted",
            code="qwen.parity.gradient_artifact",
            context={},
        )
    tolerance_policy = _mapping(value["tolerance_policy"], "tolerance_policy")
    if tolerance_policy != {
        "selected_by": "parameter_storage_dtype",
        "frozen_bands": frozen_tolerances()["gradients_by_parameter_dtype"],
        "gradient_provenance_is_diagnostic_only": True,
    }:
        raise ParityContractError(
            "gradient comparison tolerance policy drifted",
            code="qwen.parity.gradient_artifact",
            context={},
        )

    reason_keys = (
        "missing_name",
        "extra_name",
        "shape_mismatch",
        "parameter_storage_dtype_mismatch",
        "gradient_dtype_mismatch",
        "gradient_provenance_dtype_mismatch",
        "none_status_mismatch",
        "both_none",
        "both_zero",
        "one_or_both_nonzero",
        "tolerance_failure",
        "no_nonzero_gradient_signal",
    )
    expected_reason_counts = {key: 0 for key in reason_keys}
    expected_reason_counts["missing_name"] = len(missing)
    expected_reason_counts["extra_name"] = len(extra)
    expected_status = {
        "matched_parameter_count": value["matched_count"],
        "compared_tensor_count": 0,
        "compared_value_count": 0,
        "both_none_count": 0,
        "none_status_mismatch_count": 0,
        "both_zero_count": 0,
        "one_or_both_nonzero_count": 0,
    }
    expected_numerical = {
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "max_left_l2_norm": 0.0,
        "max_right_l2_norm": 0.0,
        "max_abs_l2_norm_diff": 0.0,
        "max_rel_l2_norm_diff": 0.0,
    }
    expected_per_parameter_dtype: dict[str, dict[str, Any]] = {}
    expected_per_gradient_dtype: dict[str, int] = {}
    expected_per_provenance_dtype: dict[str, int] = {}
    parameter_names: list[str] = []
    tolerance_failure_names: set[str] = set()
    finite_comparisons_by_name: dict[str, dict[str, Any]] = {}
    finite_nonzero = False
    for raw_row in parameters:
        row = _mapping(raw_row, "gradient.parameters.row")
        name = _non_empty_text(row.get("name"), "gradient.parameters.name")
        parameter_names.append(name)
        grad_status = row.get("grad_status")
        common_keys = {
            "name",
            "shape",
            "parameter_dtype",
            "parameter_storage_dtype",
            "gradient_dtype",
            "gradient_provenance_dtype",
            "grad_status",
            "parity_aligned",
            "coverage_present",
            "allclose",
            "tolerance_selected_by",
            "rtol",
            "atol",
        }
        finite_keys = {
            "left_zero",
            "right_zero",
            "left_l2_norm",
            "right_l2_norm",
            "abs_l2_norm_diff",
            "rel_l2_norm_diff",
            "max_abs_diff",
            "max_rel_diff",
            "compared_value_count",
        }
        _expect_exact_keys(
            row,
            common_keys | (finite_keys if grad_status == "finite" else set()),
            owner="gradient.parameters.row",
        )
        shape = row["shape"]
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in shape
        ):
            raise ParityContractError(
                "gradient parameter shape is invalid",
                code="qwen.parity.gradient_artifact",
                context={"name": name},
            )
        parameter_dtype = _non_empty_text(row["parameter_dtype"], "parameter_dtype")
        if row["parameter_storage_dtype"] != parameter_dtype:
            raise ParityContractError(
                "gradient parameter storage dtype is inconsistent",
                code="qwen.parity.gradient_artifact",
                context={"name": name},
            )
        tolerance = gradient_tolerance(parameter_dtype)
        if (
            row["tolerance_selected_by"] != "parameter_storage_dtype"
            or float(row["rtol"]) != tolerance["rtol"]
            or float(row["atol"]) != tolerance["atol"]
        ):
            raise ParityContractError(
                "gradient row tolerance policy drifted",
                code="qwen.parity.gradient_artifact",
                context={"name": name},
            )
        dtype_summary = expected_per_parameter_dtype.setdefault(
            parameter_dtype,
            {
                "parameter_count": 0,
                "both_none_count": 0,
                "both_zero_count": 0,
                "nonzero_count": 0,
                "tolerance_failure_count": 0,
                "compared_value_count": 0,
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
                "max_abs_l2_norm_diff": 0.0,
                "max_rel_l2_norm_diff": 0.0,
                "rtol": tolerance["rtol"],
                "atol": tolerance["atol"],
            },
        )
        dtype_summary["parameter_count"] += 1
        gradient_dtype = row["gradient_dtype"] or "none"
        provenance_dtype = row["gradient_provenance_dtype"] or "unspecified"
        expected_per_gradient_dtype[gradient_dtype] = (
            expected_per_gradient_dtype.get(gradient_dtype, 0) + 1
        )
        expected_per_provenance_dtype[provenance_dtype] = (
            expected_per_provenance_dtype.get(provenance_dtype, 0) + 1
        )
        if grad_status == "none":
            if (
                row["gradient_dtype"] is not None
                or row["parity_aligned"] is not True
                or row["coverage_present"] is not False
                or row["allclose"] is not True
            ):
                raise ParityContractError(
                    "both-None gradient row is inconsistent",
                    code="qwen.parity.gradient_artifact",
                    context={"name": name},
                )
            expected_reason_counts["both_none"] += 1
            expected_status["both_none_count"] += 1
            dtype_summary["both_none_count"] += 1
            continue
        if grad_status != "finite" or not isinstance(row["gradient_dtype"], str):
            raise ParityContractError(
                "gradient row status is invalid",
                code="qwen.parity.gradient_artifact",
                context={"name": name},
            )
        comparison = {
            "allclose": row["allclose"],
            "rtol": row["rtol"],
            "atol": row["atol"],
            "max_abs_diff": row["max_abs_diff"],
            "max_rel_diff": row["max_rel_diff"],
            "compared_value_count": row["compared_value_count"],
        }
        _validate_tensor_comparison_artifact(
            comparison, owner="gradient.parameters.row"
        )
        finite_comparisons_by_name[name] = comparison
        if (
            row["parity_aligned"] is not row["allclose"]
            or row["coverage_present"] is not True
        ):
            raise ParityContractError(
                "finite gradient row status is inconsistent",
                code="qwen.parity.gradient_artifact",
                context={"name": name},
            )
        left_norm = float(row["left_l2_norm"])
        right_norm = float(row["right_l2_norm"])
        norm_abs = abs(left_norm - right_norm)
        norm_rel = norm_abs / max(abs(right_norm), tolerance["atol"])
        if (
            any(
                not math.isfinite(float(row[field])) or float(row[field]) < 0.0
                for field in (
                    "left_l2_norm",
                    "right_l2_norm",
                    "abs_l2_norm_diff",
                    "rel_l2_norm_diff",
                )
            )
            or not math.isclose(
                float(row["abs_l2_norm_diff"]), norm_abs, rel_tol=1e-12, abs_tol=1e-12
            )
            or not math.isclose(
                float(row["rel_l2_norm_diff"]), norm_rel, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ParityContractError(
                "gradient norm summary is inconsistent",
                code="qwen.parity.gradient_artifact",
                context={"name": name},
            )
        any_nonzero = not (row["left_zero"] and row["right_zero"])
        if not isinstance(row["left_zero"], bool) or not isinstance(
            row["right_zero"], bool
        ):
            raise ParityContractError(
                "gradient zero status is invalid",
                code="qwen.parity.gradient_artifact",
                context={"name": name},
            )
        finite_nonzero = finite_nonzero or any_nonzero
        reason = "one_or_both_nonzero" if any_nonzero else "both_zero"
        expected_reason_counts[reason] += 1
        expected_status[f"{reason}_count"] += 1
        dtype_summary["nonzero_count" if any_nonzero else "both_zero_count"] += 1
        expected_status["compared_tensor_count"] += 1
        expected_status["compared_value_count"] += row["compared_value_count"]
        dtype_summary["compared_value_count"] += row["compared_value_count"]
        for summary in (expected_numerical, dtype_summary):
            summary["max_abs_diff"] = max(
                float(summary["max_abs_diff"]), float(row["max_abs_diff"])
            )
            summary["max_rel_diff"] = max(
                float(summary["max_rel_diff"]), float(row["max_rel_diff"])
            )
            summary["max_abs_l2_norm_diff"] = max(
                float(summary["max_abs_l2_norm_diff"]), norm_abs
            )
            summary["max_rel_l2_norm_diff"] = max(
                float(summary["max_rel_l2_norm_diff"]), norm_rel
            )
        expected_numerical["max_left_l2_norm"] = max(
            expected_numerical["max_left_l2_norm"], left_norm
        )
        expected_numerical["max_right_l2_norm"] = max(
            expected_numerical["max_right_l2_norm"], right_norm
        )
        if not row["allclose"]:
            tolerance_failure_names.add(name)
            expected_reason_counts["tolerance_failure"] += 1
            dtype_summary["tolerance_failure_count"] += 1
    if parameter_names != sorted(set(parameter_names)):
        raise ParityContractError(
            "gradient parameter rows are duplicate or unordered",
            code="qwen.parity.gradient_artifact",
            context={},
        )

    failure_names: set[str] = set()
    structural_count = 0
    reason_map = {
        "shape": "shape_mismatch",
        "dtype": "parameter_storage_dtype_mismatch",
        "gradient_dtype": "gradient_dtype_mismatch",
        "gradient_provenance_dtype": "gradient_provenance_dtype_mismatch",
        "none_status": "none_status_mismatch",
    }
    observed_tolerance_names: set[str] = set()
    aggregate_no_signal = False
    for raw_failure in failures:
        failure = _mapping(raw_failure, "gradient.failures.row")
        name = _non_empty_text(failure.get("name"), "gradient.failure.name")
        reason = failure.get("reason")
        if reason == "no_nonzero_gradient_signal":
            _expect_exact_keys(failure, {"name", "reason"}, owner="gradient.failure")
            if name != "<aggregate>" or aggregate_no_signal:
                raise ParityContractError(
                    "gradient aggregate failure is invalid",
                    code="qwen.parity.gradient_artifact",
                    context={},
                )
            aggregate_no_signal = True
            continue
        if name in failure_names:
            raise ParityContractError(
                "gradient failure names are duplicate",
                code="qwen.parity.gradient_artifact",
                context={"name": name},
            )
        failure_names.add(name)
        if reason == "tolerance":
            _validate_tensor_comparison_artifact(
                {
                    key: failure[key]
                    for key in (
                        "allclose",
                        "rtol",
                        "atol",
                        "max_abs_diff",
                        "max_rel_diff",
                        "compared_value_count",
                    )
                },
                owner="gradient.failure.tolerance",
            )
            _expect_exact_keys(
                failure,
                {
                    "name",
                    "reason",
                    "allclose",
                    "rtol",
                    "atol",
                    "max_abs_diff",
                    "max_rel_diff",
                    "compared_value_count",
                },
                owner="gradient.failure.tolerance",
            )
            if failure["allclose"] is not False:
                raise ParityContractError(
                    "gradient tolerance failure claims allclose",
                    code="qwen.parity.gradient_artifact",
                    context={"name": name},
                )
            if {
                key: failure[key]
                for key in (
                    "allclose",
                    "rtol",
                    "atol",
                    "max_abs_diff",
                    "max_rel_diff",
                    "compared_value_count",
                )
            } != finite_comparisons_by_name.get(name):
                raise ParityContractError(
                    "gradient tolerance failure differs from its parameter row",
                    code="qwen.parity.gradient_artifact",
                    context={"name": name},
                )
            observed_tolerance_names.add(name)
            continue
        expected_reason = reason_map.get(reason)
        if expected_reason is None:
            raise ParityContractError(
                "gradient failure reason is unsupported",
                code="qwen.parity.gradient_artifact",
                context={"reason": reason},
            )
        structural_count += 1
        expected_reason_counts[expected_reason] += 1
        if reason in {"shape", "dtype", "gradient_dtype", "gradient_provenance_dtype"}:
            _expect_exact_keys(
                failure, {"name", "reason", "left", "right"}, owner="gradient.failure"
            )
        else:
            _expect_exact_keys(
                failure,
                {"name", "reason", "left_none", "right_none"},
                owner="gradient.failure",
            )
            if (
                not isinstance(failure["left_none"], bool)
                or not isinstance(failure["right_none"], bool)
                or failure["left_none"] is failure["right_none"]
            ):
                raise ParityContractError(
                    "gradient None-status failure is inconsistent",
                    code="qwen.parity.gradient_artifact",
                    context={"name": name},
                )
            expected_status["none_status_mismatch_count"] += 1
    if observed_tolerance_names != tolerance_failure_names:
        raise ParityContractError(
            "gradient tolerance failures disagree with parameter rows",
            code="qwen.parity.gradient_artifact",
            context={},
        )
    expected_no_signal = not finite_nonzero
    expected_reason_counts["no_nonzero_gradient_signal"] = int(expected_no_signal)
    if aggregate_no_signal is not expected_no_signal:
        raise ParityContractError(
            "gradient aggregate signal failure is inconsistent",
            code="qwen.parity.gradient_artifact",
            context={},
        )
    if len(parameters) + structural_count != value["matched_count"]:
        raise ParityContractError(
            "gradient matched inventory is incomplete",
            code="qwen.parity.gradient_artifact",
            context={},
        )
    observed_reason_counts = _mapping(value["reason_counts"], "reason_counts")
    observed_status = _mapping(value["status_counts"], "status_counts")
    if (
        dict(observed_reason_counts) != expected_reason_counts
        or dict(observed_status) != expected_status
    ):
        raise ParityContractError(
            "gradient reason or status counts are inconsistent",
            code="qwen.parity.gradient_artifact",
            context={},
        )
    if (
        dict(
            _mapping(
                value["per_parameter_storage_dtype"], "per_parameter_storage_dtype"
            )
        )
        != expected_per_parameter_dtype
        or dict(_mapping(value["per_gradient_dtype"], "per_gradient_dtype"))
        != expected_per_gradient_dtype
        or dict(
            _mapping(
                value["per_gradient_provenance_dtype"], "per_gradient_provenance_dtype"
            )
        )
        != expected_per_provenance_dtype
        or dict(_mapping(value["numerical_summary"], "numerical_summary"))
        != expected_numerical
    ):
        raise ParityContractError(
            "gradient numerical or dtype summaries are inconsistent",
            code="qwen.parity.gradient_artifact",
            context={},
        )
    expected_parity = (
        not missing
        and not extra
        and structural_count == 0
        and not tolerance_failure_names
    )
    expected_coverage = (
        not missing
        and not extra
        and expected_reason_counts["both_none"] == 0
        and expected_reason_counts["none_status_mismatch"] == 0
        and finite_nonzero
    )
    if (
        value["nonzero_gradient_signal"] is not finite_nonzero
        or value["parity_aligned"] is not expected_parity
        or value["trainable_coverage_passed"] is not expected_coverage
        or value["passed"] is not (expected_parity and expected_coverage)
    ):
        raise ParityContractError(
            "gradient aggregate status contradicts bounded evidence",
            code="qwen.parity.gradient_artifact",
            context={},
        )


def gradient_tolerance(parameter_dtype: str) -> dict[str, float]:
    if parameter_dtype in {"torch.bfloat16", "bfloat16", "bf16"}:
        return {"rtol": BF16_RTOL, "atol": BF16_ATOL}
    if parameter_dtype in {"torch.float32", "float32", "fp32"}:
        return {"rtol": FP32_RTOL, "atol": FP32_ATOL}
    raise ParityContractError(
        "trainable parameter dtype has no frozen gradient tolerance",
        code="qwen.parity.gradient_dtype",
        context={"parameter_dtype": parameter_dtype},
    )


def trainable_value_identity(model: Any) -> tuple[dict[str, Any], ...]:
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            rows.append(
                {
                    "name": str(name),
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "value_sha256": tensor_sha256(parameter),
                }
            )
    if not rows:
        raise ParityContractError(
            "trainable value identity found no trainable parameters",
            code="qwen.parity.trainable_empty",
            context={},
        )
    return tuple(rows)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    header = canonical_json_bytes(
        {"shape": list(value.shape), "dtype": str(value.dtype)}
    )
    byte_view = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + byte_view).hexdigest()


def snapshot_rng() -> RngSnapshot:
    cuda_states: tuple[torch.Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return RngSnapshot(cpu=torch.get_rng_state().clone(), cuda=cuda_states)


def restore_rng(snapshot: RngSnapshot) -> None:
    torch.set_rng_state(snapshot.cpu)
    if snapshot.cuda:
        if not torch.cuda.is_available():
            raise ParityContractError(
                "CUDA RNG snapshot cannot be restored without CUDA",
                code="qwen.parity.rng_cuda_unavailable",
                context={},
            )
        observed_count = torch.cuda.device_count()
        if observed_count != len(snapshot.cuda):
            raise ParityContractError(
                "CUDA device count changed across parity arms",
                code="qwen.parity.rng_device_count",
                context={"expected": len(snapshot.cuda), "observed": observed_count},
            )
        torch.cuda.set_rng_state_all(list(snapshot.cuda))


def merged_boundary_forward_inputs(clean_inputs: Any) -> Any:
    """Test-only negative control changing only explicit FA varlen boundaries."""

    from src.qwen.fa2 import Fa2VarlenPlan

    plan = getattr(clean_inputs, "fa2_varlen_plan", None)
    if not isinstance(plan, Fa2VarlenPlan) or plan.segment_count != 2:
        raise ParityContractError(
            "merged-boundary negative requires one clean two-segment FA2 plan",
            code="qwen.parity.negative_plan",
            context={"plan_type": type(plan).__name__},
        )
    total = int(plan.segment_boundaries[-1])
    device = plan.cu_seq_lens_q.device
    merged_cu = torch.tensor([0, total], dtype=torch.int32, device=device)
    merged = Fa2VarlenPlan(
        segment_boundaries=(0, total),
        segment_lengths=(total,),
        cu_seq_lens_q=merged_cu,
        cu_seq_lens_k=merged_cu.clone(),
        max_length_q=total,
        max_length_k=total,
        attention_mask=None,
        branch_evidence_required=False,
    )
    receipt = getattr(clean_inputs, "receipt", None)
    if receipt is None:
        raise ParityContractError(
            "merged-boundary negative requires Qwen forward receipt",
            code="qwen.parity.negative_receipt",
            context={},
        )
    corrupted = replace(
        clean_inputs,
        fa2_varlen_plan=merged,
        receipt=replace(
            receipt, fa2_varlen_plan=merged, fa2_branch_proof_policy="disabled"
        ),
    )
    assert_boundary_only_corruption(clean_inputs, corrupted)
    return corrupted


def assert_boundary_only_corruption(clean: Any, corrupted: Any) -> None:
    clean_plan = clean.fa2_varlen_plan
    corrupted_plan = corrupted.fa2_varlen_plan
    if clean_plan.segment_boundaries == corrupted_plan.segment_boundaries:
        raise ParityContractError(
            "negative-control boundary did not change",
            code="qwen.parity.negative_boundary_equal",
            context={},
        )
    tensor_fields = ("input_ids", "position_ids", "pixel_values", "image_grid_thw")
    changed = [
        name
        for name in tensor_fields
        if not torch.equal(getattr(clean, name), getattr(corrupted, name))
    ]
    scalar_fields = ("pack_index", "logits_to_keep", "logits_position_ids")
    changed.extend(
        name
        for name in scalar_fields
        if not _parity_value_equal(getattr(clean, name), getattr(corrupted, name))
    )
    if changed:
        raise ParityContractError(
            "negative control changed non-boundary model inputs",
            code="qwen.parity.negative_scope",
            context={"changed_fields": changed},
        )


def negative_discriminator(
    *,
    clean_boundaries: Sequence[int],
    negative_boundaries: Sequence[int],
    logits_allclose: bool,
    loss_allclose: bool,
    gradients_allclose: bool,
) -> dict[str, Any]:
    boundary_changed = tuple(clean_boundaries) != tuple(negative_boundaries)
    forward_detected_by = [
        name
        for name, allclose in (
            ("supervised_logits", logits_allclose),
            ("total_loss", loss_allclose),
        )
        if not allclose
    ]
    diagnostic_detected_by = [] if gradients_allclose else ["trainable_gradients"]
    return {
        "detected": boundary_changed and bool(forward_detected_by),
        "boundary_changed": boundary_changed,
        "clean_boundaries": list(clean_boundaries),
        "negative_boundaries": list(negative_boundaries),
        "forward_detected_by": forward_detected_by,
        "diagnostic_detected_by": diagnostic_detected_by,
        "gradient_only_is_insufficient": True,
    }


def compare_shared_denominators(
    packed_plan: Any,
    separate_shared_plan: Any,
    *,
    packed_context_count: int,
    separate_context_count: int,
) -> dict[str, Any]:
    """Compare loss-normalization semantics while accounting for arm shape."""

    expected_context_counts = {
        "packed": _positive_int(packed_context_count, "packed_context_count"),
        "separate_shared": _positive_int(
            separate_context_count, "separate_context_count"
        ),
    }
    packed = _denominator_artifact(packed_plan, owner="packed")
    separate_shared = _denominator_artifact(
        separate_shared_plan, owner="separate_shared"
    )
    plan_normalization = {
        "packed": _plan_normalization_artifact(
            packed_plan,
            denominators=packed,
            owner="packed",
        ),
        "separate_shared": _plan_normalization_artifact(
            separate_shared_plan,
            denominators=separate_shared,
            owner="separate_shared",
        ),
    }
    failures: list[dict[str, Any]] = []
    packed_terms = set(packed)
    separate_terms = set(separate_shared)
    if packed_terms != separate_terms:
        failures.append(
            {
                "reason": "term_inventory_mismatch",
                "field": "term_name",
                "missing_in_separate_shared": sorted(packed_terms - separate_terms),
                "extra_in_separate_shared": sorted(separate_terms - packed_terms),
            }
        )
    for term_name in sorted(packed_terms & separate_terms):
        packed_term = packed[term_name]
        separate_term = separate_shared[term_name]
        for field in DENOMINATOR_SEMANTIC_FIELDS:
            if packed_term[field] != separate_term[field]:
                failures.append(
                    {
                        "reason": "semantic_field_mismatch",
                        "term_name": term_name,
                        "field": field,
                        "packed": packed_term[field],
                        "separate_shared": separate_term[field],
                    }
                )
        for arm, artifact, expected_count in (
            ("packed", packed_term, expected_context_counts["packed"]),
            (
                "separate_shared",
                separate_term,
                expected_context_counts["separate_shared"],
            ),
        ):
            if artifact["context_count"] != expected_count:
                failures.append(
                    {
                        "reason": "context_count_mismatch",
                        "term_name": term_name,
                        "arm": arm,
                        "expected": expected_count,
                        "observed": artifact["context_count"],
                    }
                )
    packed_normalization = plan_normalization["packed"]
    separate_normalization = plan_normalization["separate_shared"]
    for arm, normalization in plan_normalization.items():
        observed_context_count = normalization["counts"]["count/packs"]
        expected_context_count = expected_context_counts[arm]
        if observed_context_count != expected_context_count:
            failures.append(
                {
                    "reason": "plan_context_count_mismatch",
                    "arm": arm,
                    "expected": expected_context_count,
                    "observed": observed_context_count,
                }
            )
    for field in (
        "denominator_scope",
        "world_size",
        "rank",
        "backend_gradient_scale",
        "token_type_gate_groups",
    ):
        if packed_normalization[field] != separate_normalization[field]:
            failures.append(
                {
                    "reason": "plan_normalization_field_mismatch",
                    "field": field,
                    "packed": packed_normalization[field],
                    "separate_shared": separate_normalization[field],
                }
            )
    for field in sorted(_PLAN_COUNT_FIELDS - {"count/packs"}):
        packed_value = packed_normalization["counts"][field]
        separate_value = separate_normalization["counts"][field]
        if packed_value != separate_value:
            failures.append(
                {
                    "reason": "plan_count_semantic_mismatch",
                    "field": field,
                    "packed": packed_value,
                    "separate_shared": separate_value,
                }
            )
    return {
        "passed": not failures,
        "semantic_fields": list(DENOMINATOR_SEMANTIC_FIELDS),
        "expected_context_counts": expected_context_counts,
        "packed": packed,
        "separate_shared": separate_shared,
        "plan_normalization": plan_normalization,
        "failures": failures,
    }


def repo_identity(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    head = _git(root, "rev-parse", "HEAD").strip()
    # Probe artifacts are intentionally untracked and may be written between
    # prepare and run.  Bind tracked repository state here; every harness and
    # production owner file (including untracked new owners) is content-bound
    # separately through ``source_owner_identity``.
    status = _git(root, "status", "--short", "--untracked-files=no")
    diff = _git(root, "diff", "--binary", "HEAD", "--", ".")
    return {
        "root": str(root),
        "head": _git_object_id(head, "git HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def source_owner_identity(
    repo_root: str | Path, paths: Sequence[str]
) -> list[dict[str, str]]:
    root = Path(repo_root).expanduser().resolve()
    rows = []
    for relative in paths:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ParityContractError(
                "source owner escaped repository root",
                code="qwen.parity.source_owner_path",
                context={"path": str(path)},
                cause=exc,
            ) from exc
        rows.append({"path": relative, "sha256": sha256_file(path)})
    return rows


def assert_plan_revalidated(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    validate_parity_plan(expected)
    validate_parity_plan(observed)
    if dict(expected) != dict(observed):
        raise ParityContractError(
            "parity plan determinants changed before execution",
            code="qwen.parity.plan_drift",
            context={
                "expected_plan_sha256": expected["plan_sha256"],
                "observed_plan_sha256": observed["plan_sha256"],
            },
        )


def bounded_failure(exc: BaseException) -> dict[str, str]:
    message = " ".join(str(exc).split())[:1000] or type(exc).__name__
    code = getattr(exc, "code", None)
    result = {"type": type(exc).__name__, "message": message}
    if isinstance(code, str) and code:
        result["code"] = code[:160]
    return result


def _denominator_artifact(plan: Any, *, owner: str) -> dict[str, Any]:
    denominators = getattr(plan, "denominators", None)
    if not isinstance(denominators, Mapping) or not denominators:
        raise ParityContractError(
            "loss plan does not expose a non-empty denominator inventory",
            code="qwen.parity.denominator_shape",
            context={"owner": owner, "plan_type": type(plan).__name__},
        )
    result: dict[str, dict[str, Any]] = {}
    for raw_name, denominator in sorted(denominators.items()):
        name = _non_empty_text(raw_name, f"{owner}.denominator_name")
        artifact_fn = getattr(denominator, "to_artifact_dict", None)
        if not callable(artifact_fn):
            raise ParityContractError(
                "loss denominator does not expose an artifact serializer",
                code="qwen.parity.denominator_shape",
                context={"owner": owner, "term_name": name},
            )
        artifact = _mapping(artifact_fn(), f"{owner}.denominators.{name}")
        observed_fields = set(artifact)
        if observed_fields != set(_DENOMINATOR_ARTIFACT_FIELDS):
            raise ParityContractError(
                "loss denominator artifact fields differ from production schema",
                code="qwen.parity.denominator_schema",
                context={
                    "owner": owner,
                    "term_name": name,
                    "missing": sorted(
                        set(_DENOMINATOR_ARTIFACT_FIELDS) - observed_fields
                    ),
                    "unknown": sorted(
                        observed_fields - set(_DENOMINATOR_ARTIFACT_FIELDS)
                    ),
                },
            )
        term_name = _non_empty_text(artifact["term_name"], f"{owner}.{name}.term_name")
        denominator_scope = _non_empty_text(
            artifact["denominator_scope"], f"{owner}.{name}.denominator_scope"
        )
        if term_name != name:
            raise ParityContractError(
                "loss denominator artifact term name differs from its inventory key",
                code="qwen.parity.denominator_schema",
                context={
                    "owner": owner,
                    "inventory_key": name,
                    "term_name": term_name,
                },
            )
        result[name] = {
            "term_name": term_name,
            "denominator_scope": denominator_scope,
            "eligible_segment_count": _positive_int(
                artifact["eligible_segment_count"],
                f"{owner}.{name}.eligible_segment_count",
            ),
            "selected_atom_count": _positive_int(
                artifact["selected_atom_count"],
                f"{owner}.{name}.selected_atom_count",
            ),
            "skipped_segment_count": _non_negative_int(
                artifact["skipped_segment_count"],
                f"{owner}.{name}.skipped_segment_count",
            ),
            "context_count": _positive_int(
                artifact["context_count"], f"{owner}.{name}.context_count"
            ),
        }
    return result


def _plan_normalization_artifact(
    plan: Any,
    *,
    denominators: Mapping[str, Mapping[str, Any]],
    owner: str,
) -> dict[str, Any]:
    denominator_scope = _non_empty_text(
        getattr(plan, "denominator_scope", None), f"{owner}.denominator_scope"
    )
    observed_scopes = sorted(
        {str(artifact["denominator_scope"]) for artifact in denominators.values()}
    )
    if observed_scopes != [denominator_scope]:
        raise ParityContractError(
            "loss plan denominator scope differs from its term artifacts",
            code="qwen.parity.denominator_plan_schema",
            context={
                "owner": owner,
                "plan_scope": denominator_scope,
                "term_scopes": observed_scopes,
            },
        )
    world_size = _positive_int(getattr(plan, "world_size", None), f"{owner}.world_size")
    rank = _non_negative_int(getattr(plan, "rank", None), f"{owner}.rank")
    backend_gradient_scale = getattr(plan, "backend_gradient_scale", None)
    if (
        world_size != 1
        or rank != 0
        or isinstance(backend_gradient_scale, bool)
        or not isinstance(backend_gradient_scale, (int, float))
        or not math.isfinite(float(backend_gradient_scale))
        or float(backend_gradient_scale) != 1.0
    ):
        raise ParityContractError(
            "Wave 2 denominator comparison requires exact one-rank normalization",
            code="qwen.parity.denominator_plan_schema",
            context={
                "owner": owner,
                "world_size": world_size,
                "rank": rank,
                "backend_gradient_scale": backend_gradient_scale,
            },
        )
    groups = getattr(plan, "token_type_gate_groups", None)
    if not isinstance(groups, tuple) or any(
        not isinstance(group, str) or not group for group in groups
    ):
        raise ParityContractError(
            "loss plan token-type groups differ from production schema",
            code="qwen.parity.denominator_plan_schema",
            context={"owner": owner, "value_type": type(groups).__name__},
        )
    counts = _mapping(getattr(plan, "counts", None), f"{owner}.counts")
    _expect_exact_keys(counts, set(_PLAN_COUNT_FIELDS), owner=f"{owner}.counts")
    checked_counts = {
        field: _non_negative_int(counts[field], f"{owner}.counts.{field}")
        for field in sorted(_PLAN_COUNT_FIELDS)
    }
    base_ce = denominators.get("base_ce")
    if base_ce is None:
        raise ParityContractError(
            "loss denominator inventory omits the base_ce normalization owner",
            code="qwen.parity.denominator_plan_schema",
            context={"owner": owner, "terms": sorted(denominators)},
        )
    expected_base_counts = {
        "count/supervised_atoms": base_ce["selected_atom_count"],
        "count/eligible_segments": base_ce["eligible_segment_count"],
        "count/skipped_segments": base_ce["skipped_segment_count"],
    }
    inconsistent_counts = {
        field: {"expected": expected, "observed": checked_counts[field]}
        for field, expected in expected_base_counts.items()
        if checked_counts[field] != expected
    }
    if inconsistent_counts or checked_counts["count/examples"] <= 0:
        raise ParityContractError(
            "loss plan counts disagree with denominator semantics",
            code="qwen.parity.denominator_plan_schema",
            context={
                "owner": owner,
                "inconsistent": inconsistent_counts,
                "example_count": checked_counts["count/examples"],
            },
        )
    return {
        "denominator_scope": denominator_scope,
        "world_size": world_size,
        "rank": rank,
        "backend_gradient_scale": float(backend_gradient_scale),
        "token_type_gate_groups": list(groups),
        "counts": checked_counts,
    }


def _validate_denominator_artifact_mapping(
    value: Any,
    *,
    owner: str,
    expected_context_count: int,
) -> dict[str, dict[str, Any]]:
    denominators = _mapping(value, owner)
    if not denominators:
        raise ParityContractError(
            "denominator artifact inventory must be non-empty",
            code="qwen.parity.denominator_artifact",
            context={"owner": owner},
        )
    result: dict[str, dict[str, Any]] = {}
    for raw_name, raw_artifact in denominators.items():
        name = _non_empty_text(raw_name, f"{owner}.term_name")
        artifact = _mapping(raw_artifact, f"{owner}.{name}")
        _expect_exact_keys(
            artifact,
            set(_DENOMINATOR_ARTIFACT_FIELDS),
            owner=f"{owner}.{name}",
        )
        term_name = _non_empty_text(artifact["term_name"], f"{owner}.{name}.term_name")
        if term_name != name:
            raise ParityContractError(
                "denominator artifact term name differs from its inventory key",
                code="qwen.parity.denominator_artifact",
                context={"owner": owner, "key": name, "term_name": term_name},
            )
        row = {
            "term_name": term_name,
            "denominator_scope": _non_empty_text(
                artifact["denominator_scope"], f"{owner}.{name}.denominator_scope"
            ),
            "eligible_segment_count": _positive_int(
                artifact["eligible_segment_count"],
                f"{owner}.{name}.eligible_segment_count",
            ),
            "selected_atom_count": _positive_int(
                artifact["selected_atom_count"],
                f"{owner}.{name}.selected_atom_count",
            ),
            "skipped_segment_count": _non_negative_int(
                artifact["skipped_segment_count"],
                f"{owner}.{name}.skipped_segment_count",
            ),
            "context_count": _positive_int(
                artifact["context_count"], f"{owner}.{name}.context_count"
            ),
        }
        if row["context_count"] != expected_context_count:
            raise ParityContractError(
                "denominator artifact context count differs from literal arm shape",
                code="qwen.parity.denominator_arm_shape",
                context={
                    "owner": owner,
                    "term_name": name,
                    "expected": expected_context_count,
                    "observed": row["context_count"],
                },
            )
        result[name] = row
    return result


def _validate_plan_normalization_mapping(
    value: Any,
    *,
    owner: str,
    denominators: Mapping[str, Mapping[str, Any]],
    expected_context_count: int,
) -> dict[str, Any]:
    normalization = _mapping(value, owner)
    _expect_exact_keys(
        normalization,
        {
            "denominator_scope",
            "world_size",
            "rank",
            "backend_gradient_scale",
            "token_type_gate_groups",
            "counts",
        },
        owner=owner,
    )
    denominator_scope = _non_empty_text(
        normalization["denominator_scope"], f"{owner}.denominator_scope"
    )
    if any(
        denominator["denominator_scope"] != denominator_scope
        for denominator in denominators.values()
    ):
        raise ParityContractError(
            "plan normalization scope differs from denominator artifacts",
            code="qwen.parity.denominator_artifact",
            context={"owner": owner},
        )
    world_size = _positive_int(normalization["world_size"], f"{owner}.world_size")
    rank = _non_negative_int(normalization["rank"], f"{owner}.rank")
    scale = _finite_number(
        normalization["backend_gradient_scale"], f"{owner}.backend_gradient_scale"
    )
    if world_size != 1 or rank != 0 or scale != 1.0:
        raise ParityContractError(
            "plan normalization differs from exact one-rank Wave 2 scope",
            code="qwen.parity.denominator_artifact",
            context={
                "owner": owner,
                "world_size": world_size,
                "rank": rank,
                "scale": scale,
            },
        )
    groups = _text_list_allow_empty(
        normalization["token_type_gate_groups"], f"{owner}.token_type_gate_groups"
    )
    if not groups or len(groups) != len(set(groups)):
        raise ParityContractError(
            "plan normalization token-type groups are empty or duplicated",
            code="qwen.parity.denominator_artifact",
            context={"owner": owner, "groups": groups},
        )
    counts = _mapping(normalization["counts"], f"{owner}.counts")
    _expect_exact_keys(counts, set(_PLAN_COUNT_FIELDS), owner=f"{owner}.counts")
    checked_counts = {
        field: _non_negative_int(counts[field], f"{owner}.counts.{field}")
        for field in sorted(_PLAN_COUNT_FIELDS)
    }
    if checked_counts["count/packs"] != expected_context_count:
        raise ParityContractError(
            "plan normalization pack count differs from literal arm shape",
            code="qwen.parity.denominator_arm_shape",
            context={
                "owner": owner,
                "expected": expected_context_count,
                "observed": checked_counts["count/packs"],
            },
        )
    base_ce = denominators.get("base_ce")
    if base_ce is None:
        raise ParityContractError(
            "plan normalization omits base_ce denominator",
            code="qwen.parity.denominator_artifact",
            context={"owner": owner},
        )
    expected_counts = {
        "count/supervised_atoms": base_ce["selected_atom_count"],
        "count/eligible_segments": base_ce["eligible_segment_count"],
        "count/skipped_segments": base_ce["skipped_segment_count"],
    }
    if any(
        checked_counts[field] != expected for field, expected in expected_counts.items()
    ):
        raise ParityContractError(
            "plan normalization counts disagree with base_ce denominator",
            code="qwen.parity.denominator_artifact",
            context={"owner": owner},
        )
    if checked_counts["count/examples"] <= 0:
        raise ParityContractError(
            "plan normalization example count must be positive",
            code="qwen.parity.denominator_artifact",
            context={"owner": owner},
        )
    return {
        "denominator_scope": denominator_scope,
        "world_size": world_size,
        "rank": rank,
        "backend_gradient_scale": scale,
        "token_type_gate_groups": groups,
        "counts": checked_counts,
    }


def _safe_relative_weight_path(value: Any) -> str:
    text = _non_empty_text(value, "weight shard path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
        or not text.endswith(".safetensors")
    ):
        raise ParityContractError(
            "weight shard declaration is not a safe relative safetensors path",
            code="qwen.parity.weight_shard_path",
            context={"path": text[:240]},
        )
    return path.as_posix()


def _preflight_weight_file_set(paths: Sequence[Path]) -> None:
    total_bytes = 0
    for path in paths:
        try:
            file_stat = path.lstat()
        except OSError as exc:
            raise ParityContractError(
                "declared weight identity file is unavailable",
                code="qwen.parity.weight_file_unavailable",
                context={"path": str(path), "error": type(exc).__name__},
                cause=exc,
            ) from exc
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise ParityContractError(
                "declared weight identity file must be a regular non-symlink",
                code="qwen.parity.weight_file_type",
                context={"path": str(path)},
            )
        if file_stat.st_size > MAX_WEIGHT_SHARD_BYTES:
            raise ParityContractError(
                "declared weight identity file exceeds its byte bound",
                code="qwen.parity.weight_file_bound",
                context={
                    "path": str(path),
                    "size_bytes": file_stat.st_size,
                    "maximum": MAX_WEIGHT_SHARD_BYTES,
                },
            )
        total_bytes += int(file_stat.st_size)
        if total_bytes > MAX_WEIGHT_TOTAL_BYTES:
            raise ParityContractError(
                "base-model weight payload exceeds the total byte bound",
                code="qwen.parity.weight_total_bound",
                context={
                    "total_bytes": total_bytes,
                    "maximum": MAX_WEIGHT_TOTAL_BYTES,
                },
            )


def _validate_requested_weight_hash_workers(max_workers: int | None) -> None:
    if max_workers is None:
        return
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise ParityContractError(
            "base-model weight hash workers must be an integer",
            code="qwen.parity.weight_hash_workers",
            context={"value_type": type(max_workers).__name__},
        )
    if max_workers <= 0:
        raise ParityContractError(
            "base-model weight hash workers must be positive",
            code="qwen.parity.weight_hash_workers",
            context={"max_workers": max_workers},
        )


def _resolve_weight_hash_workers(
    max_workers: int | None,
    *,
    payload_file_count: int,
) -> int:
    _validate_requested_weight_hash_workers(max_workers)
    if payload_file_count <= 0:
        raise ParityContractError(
            "base-model weight hash payload inventory is empty",
            code="qwen.parity.weight_layout",
            context={"payload_file_count": payload_file_count},
        )
    if max_workers is not None:
        return min(max_workers, payload_file_count, 4)
    available_cpus = os.cpu_count() or 1
    if available_cpus <= 0:
        available_cpus = 1
    return min(payload_file_count, 4, available_cpus)


def _weight_hash_executor_error(
    exc: Exception,
    *,
    stage: str,
    path: Path | None = None,
) -> ParityContractError:
    context: dict[str, Any] = {
        "stage": stage,
        "error": type(exc).__name__,
    }
    if path is not None:
        context["path"] = path.name
    return ParityContractError(
        "base-model weight hash executor failed",
        code="qwen.parity.weight_hash_executor",
        context=context,
        cause=exc,
    )


def _hash_weight_shards(
    shard_paths: Sequence[Path],
    *,
    root: Path,
    resolved_workers: int,
) -> list[dict[str, Any]]:
    try:
        executor = ThreadPoolExecutor(max_workers=resolved_workers)
    except Exception as exc:
        raise _weight_hash_executor_error(exc, stage="create") from exc

    futures: list[Future[tuple[dict[str, Any], bytes]]] = []
    primary_error: BaseException | None = None
    try:
        submit_error: Exception | None = None
        for shard_path in shard_paths:
            try:
                futures.append(
                    executor.submit(
                        _stable_file_identity,
                        shard_path,
                        root=root,
                        max_bytes=MAX_WEIGHT_SHARD_BYTES,
                    )
                )
            except Exception as exc:
                submit_error = exc
                break

        results: list[dict[str, Any] | None] = [None] * len(futures)
        contract_errors: list[tuple[int, ParityContractError]] = []
        operational_errors: list[tuple[int, Exception]] = []
        failure_seen = submit_error is not None
        for index, future in enumerate(futures):
            if failure_seen:
                future.cancel()
            try:
                results[index] = future.result()[0]
            except ParityContractError as exc:
                contract_errors.append((index, exc))
                failure_seen = True
                for pending in futures[index + 1 :]:
                    pending.cancel()
            except Exception as exc:
                operational_errors.append((index, exc))
                failure_seen = True
                for pending in futures[index + 1 :]:
                    pending.cancel()

        if contract_errors:
            raise min(contract_errors, key=lambda item: item[0])[1]
        if submit_error is not None:
            raise _weight_hash_executor_error(submit_error, stage="submit")
        if operational_errors:
            failed_index, operational_error = min(
                operational_errors,
                key=lambda item: item[0],
            )
            raise _weight_hash_executor_error(
                operational_error,
                stage="result",
                path=shard_paths[failed_index],
            )
        if len(results) != len(shard_paths) or any(item is None for item in results):
            raise ParityContractError(
                "base-model weight hash executor returned an incomplete inventory",
                code="qwen.parity.weight_hash_executor",
                context={
                    "stage": "result",
                    "expected": len(shard_paths),
                    "observed": sum(item is not None for item in results),
                },
            )
        return [item for item in results if item is not None]
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except Exception as exc:
            if primary_error is None:
                raise _weight_hash_executor_error(exc, stage="shutdown") from exc


def _stable_file_identity(
    path: Path,
    *,
    root: Path,
    max_bytes: int,
    return_bytes: bool = False,
) -> tuple[dict[str, Any], bytes]:
    candidate = path
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ParityContractError(
            "weight identity path escaped the model root",
            code="qwen.parity.weight_shard_path",
            context={"path": str(candidate), "root": str(root)},
            cause=exc,
        ) from exc
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        try:
            if cursor.is_symlink():
                raise ParityContractError(
                    "weight identity path contains a symlink",
                    code="qwen.parity.weight_file_type",
                    context={"path": relative, "symlink_component": str(cursor)},
                )
        except OSError as exc:
            raise ParityContractError(
                "weight identity path component is unavailable",
                code="qwen.parity.weight_file_unavailable",
                context={"path": relative, "error": type(exc).__name__},
                cause=exc,
            ) from exc
    try:
        path_stat = candidate.lstat()
    except OSError as exc:
        raise ParityContractError(
            "declared weight identity file is unavailable",
            code="qwen.parity.weight_file_unavailable",
            context={"path": relative, "error": type(exc).__name__},
            cause=exc,
        ) from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise ParityContractError(
            "declared weight identity file must be a regular non-symlink",
            code="qwen.parity.weight_file_type",
            context={"path": relative},
        )
    if path_stat.st_size > max_bytes:
        raise ParityContractError(
            "declared weight identity file exceeds its byte bound",
            code="qwen.parity.weight_file_bound",
            context={
                "path": relative,
                "size_bytes": path_stat.st_size,
                "maximum": max_bytes,
            },
        )
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    observed_bytes = 0
    try:
        with candidate.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                observed_bytes += len(chunk)
                if return_bytes:
                    chunks.append(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ParityContractError(
            "declared weight identity file could not be read",
            code="qwen.parity.weight_file_unavailable",
            context={"path": relative, "error": type(exc).__name__},
            cause=exc,
        ) from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ParityContractError(
            "declared weight identity file changed while hashing",
            code="qwen.parity.weight_file_drift",
            context={"path": relative},
        )
    if observed_bytes != before.st_size:
        raise ParityContractError(
            "declared weight identity file produced a short snapshot",
            code="qwen.parity.weight_file_drift",
            context={
                "path": relative,
                "expected": before.st_size,
                "observed": observed_bytes,
            },
        )
    payload = b"".join(chunks) if return_bytes else b""
    return (
        {
            "path": relative,
            "size_bytes": int(before.st_size),
            "sha256": digest.hexdigest(),
        },
        payload,
    )


def _gradient_records_by_name(
    records: Sequence[GradientRecord], owner: str
) -> dict[str, GradientRecord]:
    result: dict[str, GradientRecord] = {}
    for record in records:
        if record.name in result:
            raise ParityContractError(
                "gradient inventory contains a duplicate parameter name",
                code="qwen.parity.gradient_duplicate",
                context={"owner": owner, "name": record.name},
            )
        if record.grad is not None and not bool(torch.isfinite(record.grad).all()):
            raise ParityContractError(
                "gradient inventory contains non-finite values",
                code="qwen.parity.gradient_nonfinite",
                context={"owner": owner, "name": record.name},
            )
        result[record.name] = record
    return result


def _as_sequence_tuple(value: Any | Sequence[Any]) -> tuple[Any, ...]:
    if hasattr(value, "atoms"):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    raise ParityContractError(
        "semantic inventory input must be a TokenSequence or sequence",
        code="qwen.parity.atom_sequence",
        context={"value_type": type(value).__name__},
    )


def _parity_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    return bool(left == right)


def _validate_device_sample(sample: Mapping[str, Any], *, owner: str) -> None:
    _expect_exact_keys(
        sample,
        {
            "sample_index",
            "monotonic_ns",
            "physical_index",
            "uuid",
            "memory_used_bytes",
            "utilization_percent",
        },
        owner=owner,
    )
    _non_negative_int(sample["sample_index"], f"{owner}.sample_index")
    _positive_int(sample["monotonic_ns"], f"{owner}.monotonic_ns")
    _non_negative_int(sample["physical_index"], f"{owner}.physical_index")
    _non_empty_text(sample["uuid"], f"{owner}.uuid")
    _non_negative_int(sample["memory_used_bytes"], f"{owner}.memory_used_bytes")
    utilization = _non_negative_int(
        sample["utilization_percent"], f"{owner}.utilization_percent"
    )
    if utilization > 100:
        raise ParityContractError(
            "GPU utilization sample exceeds 100 percent",
            code="qwen.parity.measurement_sampler",
            context={"owner": owner, "utilization_percent": utilization},
        )


def _validate_phase_resource_sample(sample: Mapping[str, Any]) -> None:
    _expect_exact_keys(
        sample,
        {"phase", "monotonic_ns", "host", "gpu", "torch_cuda"},
        owner="phase_boundary_sample",
    )
    _non_empty_text(sample["phase"], "phase_boundary_sample.phase")
    _positive_int(sample["monotonic_ns"], "phase_boundary_sample.monotonic_ns")
    host = _mapping(sample["host"], "phase_boundary_sample.host")
    _expect_exact_keys(
        host,
        {"rss_hwm_bytes", "io_read_bytes", "io_write_bytes"},
        owner="phase_boundary_sample.host",
    )
    for field in ("rss_hwm_bytes", "io_read_bytes", "io_write_bytes"):
        _non_negative_int(host[field], f"phase_boundary_sample.host.{field}")
    _validate_device_sample(
        _mapping(sample["gpu"], "phase_boundary_sample.gpu"),
        owner="phase_boundary_sample.gpu",
    )
    torch_cuda = _mapping(sample["torch_cuda"], "phase_boundary_sample.torch_cuda")
    _expect_exact_keys(
        torch_cuda,
        {"max_allocated_bytes", "max_reserved_bytes"},
        owner="phase_boundary_sample.torch_cuda",
    )
    _non_negative_int(
        torch_cuda["max_allocated_bytes"],
        "phase_boundary_sample.torch_cuda.max_allocated_bytes",
    )
    _non_negative_int(
        torch_cuda["max_reserved_bytes"],
        "phase_boundary_sample.torch_cuda.max_reserved_bytes",
    )


def _key_from_mapping(value: Mapping[str, Any]) -> SemanticAtomKey:
    _expect_exact_keys(
        value,
        {
            "example_id",
            "logical_target_position",
            "logical_target_end",
            "token_id",
            "token_type",
            "object_id",
            "field",
            "source",
        },
        owner="semantic_atom_key",
    )
    return SemanticAtomKey(
        example_id=_non_empty_text(value["example_id"], "example_id"),
        logical_target_position=_non_negative_int(
            value["logical_target_position"], "logical_target_position"
        ),
        logical_target_end=_positive_int(
            value["logical_target_end"], "logical_target_end"
        ),
        token_id=_non_negative_int(value["token_id"], "token_id"),
        token_type=_non_empty_text(value["token_type"], "token_type"),
        object_id=_optional_text(value["object_id"], "object_id"),
        field=_optional_text(value["field"], "field"),
        source=_optional_text(value["source"], "source"),
    )


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ParityContractError(
            "repository identity command failed",
            code="qwen.parity.git_identity",
            context={"args": list(args), "error": type(exc).__name__},
            cause=exc,
        ) from exc
    return result.stdout


def _validate_qwen_component_identity(
    value: Mapping[str, Any],
    *,
    owner: str,
    expected_load_model: bool,
) -> dict[str, Any]:
    identity = _mapping(value, owner)
    _expect_exact_keys(identity, set(_QWEN_COMPONENT_IDENTITY_FIELDS), owner=owner)
    _non_empty_text(identity["base_model_path"], f"{owner}.base_model_path")
    _sha256_text(identity["base_config_sha256"], f"{owner}.base_config_sha256")
    _sha256_text(identity["tokenizer_sha256"], f"{owner}.tokenizer_sha256")
    _non_empty_text(identity["attn_implementation"], f"{owner}.attn_implementation")
    for field in ("processor", "model", "tokens", "package_versions"):
        _mapping(identity[field], f"{owner}.{field}")
    if identity["load_model"] is not expected_load_model:
        raise ParityContractError(
            "Qwen component load state differs from the required transition",
            code="qwen.parity.model_identity_load_state",
            context={
                "owner": owner,
                "expected": expected_load_model,
                "observed": identity["load_model"],
            },
        )
    _mapping(identity["runtime_patches"], f"{owner}.runtime_patches")
    canonical_json_bytes(identity)
    return dict(identity)


def _only_patch_receipt(value: Any, *, owner: str) -> Mapping[str, Any]:
    patches = _mapping(value, f"{owner}.runtime_patches")
    if set(patches) != {QWEN_PATCH_EMBED_LINEARIZATION_NAME}:
        raise ParityContractError(
            "Qwen component identity requires exactly one runtime patch receipt",
            code="qwen.parity.model_identity_patch_inventory",
            context={"owner": owner, "patches": sorted(str(key) for key in patches)},
        )
    receipt = _mapping(
        patches[QWEN_PATCH_EMBED_LINEARIZATION_NAME],
        f"{owner}.runtime_patches.{QWEN_PATCH_EMBED_LINEARIZATION_NAME}",
    )
    _expect_exact_keys(
        receipt,
        set(_QWEN_RUNTIME_PATCH_RECEIPT_FIELDS),
        owner=f"{owner}.runtime_patch_receipt",
    )
    if receipt["name"] != QWEN_PATCH_EMBED_LINEARIZATION_NAME:
        raise ParityContractError(
            "Qwen runtime patch receipt name is inconsistent",
            code="qwen.parity.model_identity_patch_name",
            context={"owner": owner, "name": receipt["name"]},
        )
    return receipt


def _validate_model_free_patch_receipt(receipt: Mapping[str, Any]) -> str:
    policy = receipt["policy"]
    if policy not in {"enabled", "disabled"}:
        raise ParityContractError(
            "planned Qwen runtime patch policy is unsupported",
            code="qwen.parity.model_identity_patch_policy",
            context={"policy": policy},
        )
    required = {
        "name": QWEN_PATCH_EMBED_LINEARIZATION_NAME,
        "policy": policy,
        "applied": False,
        "reason": "model_not_loaded",
    }
    mismatches = {
        key: {"expected": expected, "observed": receipt.get(key)}
        for key, expected in required.items()
        if receipt.get(key) != expected
    }
    optional = set(_QWEN_RUNTIME_PATCH_RECEIPT_FIELDS) - set(required)
    non_null_optional = sorted(key for key in optional if receipt.get(key) is not None)
    if mismatches or non_null_optional:
        raise ParityContractError(
            "planned Qwen runtime patch receipt is not model-free",
            code="qwen.parity.model_identity_patch_expected",
            context={
                "mismatches": mismatches,
                "non_null_optional": non_null_optional,
            },
        )
    return str(policy)


def _validate_loaded_patch_receipt(receipt: Mapping[str, Any], *, policy: str) -> None:
    if receipt.get("policy") != policy:
        raise ParityContractError(
            "loaded Qwen runtime patch policy differs from the prepared plan",
            code="qwen.parity.model_identity_patch_policy",
            context={"expected": policy, "observed": receipt.get("policy")},
        )
    if policy == "disabled":
        required = {
            "name": QWEN_PATCH_EMBED_LINEARIZATION_NAME,
            "policy": "disabled",
            "applied": False,
            "reason": "policy_disabled",
            "owner_path": "model.visual.patch_embed",
        }
        mismatches = {
            key: {"expected": expected, "observed": receipt.get(key)}
            for key, expected in required.items()
            if receipt.get(key) != expected
        }
        optional = set(_QWEN_RUNTIME_PATCH_RECEIPT_FIELDS) - set(required)
        non_null_optional = sorted(
            key for key in optional if receipt.get(key) is not None
        )
        if mismatches or non_null_optional:
            raise ParityContractError(
                "disabled Qwen runtime patch receipt is inconsistent",
                code="qwen.parity.model_identity_patch_disabled",
                context={
                    "mismatches": mismatches,
                    "non_null_optional": non_null_optional,
                },
            )
        return

    required = {
        "name": QWEN_PATCH_EMBED_LINEARIZATION_NAME,
        "policy": "enabled",
        "applied": True,
        "reason": "conv3d_kernel_stride_equivalent_linear_projection",
        "owner_path": "model.visual.patch_embed",
        "original_class": "Qwen3VLVisionPatchEmbed",
        "owner_class": "Qwen3VLVisionPatchEmbed",
        "patched_class": "LinearizedQwen3VLPatchEmbed",
        "projection_class": "Conv3d",
    }
    mismatches = {
        key: {"expected": expected, "observed": receipt.get(key)}
        for key, expected in required.items()
        if receipt.get(key) != expected
    }
    if mismatches:
        raise ParityContractError(
            "enabled Qwen runtime patch receipt owner or reason is inconsistent",
            code="qwen.parity.model_identity_patch_enabled",
            context={"mismatches": mismatches},
        )
    _sha256_text(
        receipt["original_forward_sha256"],
        "runtime_patch.original_forward_sha256",
    )
    _sha256_text(
        receipt["replacement_forward_sha256"],
        "runtime_patch.replacement_forward_sha256",
    )
    in_channels = _positive_int(receipt["in_channels"], "runtime_patch.in_channels")
    temporal = _positive_int(
        receipt["temporal_patch_size"], "runtime_patch.temporal_patch_size"
    )
    patch_size = _positive_int(receipt["patch_size"], "runtime_patch.patch_size")
    embed_dim = _positive_int(receipt["embed_dim"], "runtime_patch.embed_dim")
    weight_shape = _positive_int_list(
        receipt["weight_shape"], "runtime_patch.weight_shape"
    )
    kernel_size = _positive_int_list(
        receipt["kernel_size"], "runtime_patch.kernel_size"
    )
    stride = _positive_int_list(receipt["stride"], "runtime_patch.stride")
    padding = _int_list(receipt["padding"], "runtime_patch.padding")
    dilation = _positive_int_list(receipt["dilation"], "runtime_patch.dilation")
    if not isinstance(receipt["bias"], bool):
        raise ParityContractError(
            "enabled Qwen runtime patch bias identity must be boolean",
            code="qwen.parity.model_identity_patch_shape",
            context={"bias": receipt["bias"]},
        )
    expected_kernel = [temporal, patch_size, patch_size]
    expected_weight = [embed_dim, in_channels, *expected_kernel]
    if (
        weight_shape != expected_weight
        or kernel_size != expected_kernel
        or stride != expected_kernel
        or padding != [0, 0, 0]
        or dilation != [1, 1, 1]
        or receipt["groups"] != 1
    ):
        raise ParityContractError(
            "enabled Qwen runtime patch Conv3d shape invariants are invalid",
            code="qwen.parity.model_identity_patch_shape",
            context={
                "weight_shape": weight_shape,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": receipt["groups"],
            },
        )
    probe = _mapping(receipt["equivalence_probe"], "runtime_patch.equivalence_probe")
    _expect_exact_keys(
        probe,
        {
            "scope",
            "sample_count",
            "input_features",
            "max_abs_diff",
            "grad_max_abs_diff",
        },
        owner="runtime_patch.equivalence_probe",
    )
    expected_input_features = in_channels * temporal * patch_size * patch_size
    if (
        probe["scope"] != "cpu_float32_forward_and_grad_small_canary"
        or probe["sample_count"] != 2
        or probe["input_features"] != expected_input_features
    ):
        raise ParityContractError(
            "enabled Qwen runtime patch equivalence probe identity is invalid",
            code="qwen.parity.model_identity_patch_probe",
            context=dict(probe),
        )
    for field in ("max_abs_diff", "grad_max_abs_diff"):
        value = probe[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value) > 1.0e-4
        ):
            raise ParityContractError(
                "enabled Qwen runtime patch equivalence probe exceeds tolerance",
                code="qwen.parity.model_identity_patch_probe",
                context={"field": field, "value": value, "maximum": 1.0e-4},
            )


def _positive_int_list(value: Any, owner: str) -> list[int]:
    integers = _int_list(value, owner)
    if any(item <= 0 for item in integers):
        raise ParityContractError(
            "parity field must be a positive integer list",
            code="qwen.parity.integer",
            context={"owner": owner, "value": integers},
        )
    return integers


def _expect_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, owner: str
) -> None:
    if not isinstance(value, Mapping):
        raise ParityContractError(
            "parity object must be a mapping",
            code="qwen.parity.mapping",
            context={"owner": owner, "value_type": type(value).__name__},
        )
    observed = set(value)
    if observed != expected:
        raise ParityContractError(
            "parity object fields differ from strict schema",
            code="qwen.parity.fields",
            context={
                "owner": owner,
                "missing": sorted(expected - observed),
                "unknown": sorted(observed - expected),
            },
        )


def _mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParityContractError(
            "parity field must be a mapping",
            code="qwen.parity.mapping",
            context={"owner": owner, "value_type": type(value).__name__},
        )
    return value


def _list_of_mappings(value: Any, owner: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ParityContractError(
            "parity field must be a list of mappings",
            code="qwen.parity.mapping_list",
            context={"owner": owner, "value_type": type(value).__name__},
        )
    return value


def _int_list(value: Any, owner: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ParityContractError(
            "parity field must be an integer list",
            code="qwen.parity.int_list",
            context={"owner": owner},
        )
    return value


def _text_list(value: Any, owner: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ParityContractError(
            "parity field must be a non-empty string list",
            code="qwen.parity.text_list",
            context={"owner": owner},
        )
    return value


def _text_list_allow_empty(value: Any, owner: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ParityContractError(
            "parity field must be a string list",
            code="qwen.parity.text_list",
            context={"owner": owner},
        )
    return value


def _non_empty_text(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParityContractError(
            "parity field must be a non-empty string",
            code="qwen.parity.text",
            context={"owner": owner, "value_type": type(value).__name__},
        )
    return value


def _sha256_text(value: Any, owner: str) -> str:
    text = _non_empty_text(value, owner)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ParityContractError(
            "parity identity must be a lowercase SHA-256 digest",
            code="qwen.parity.sha256",
            context={"owner": owner},
        )
    return text


def _git_object_id(value: Any, owner: str) -> str:
    text = _non_empty_text(value, owner)
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise ParityContractError(
            "repository HEAD must be a lowercase Git object id",
            code="qwen.parity.git_object_id",
            context={"owner": owner, "length": len(text)},
        )
    return text


def _optional_text(value: Any, owner: str) -> str | None:
    if value is None:
        return None
    return _non_empty_text(value, owner)


def _non_negative_int(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ParityContractError(
            "parity field must be a non-negative integer",
            code="qwen.parity.integer",
            context={"owner": owner, "value": value},
        )
    return value


def _positive_int(value: Any, owner: str) -> int:
    integer = _non_negative_int(value, owner)
    if integer <= 0:
        raise ParityContractError(
            "parity field must be a positive integer",
            code="qwen.parity.integer",
            context={"owner": owner, "value": value},
        )
    return integer


def _finite_number(value: Any, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ParityContractError(
            "parity field must be a finite number",
            code="qwen.parity.number",
            context={"owner": owner, "value": value},
        )
    return float(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


__all__ = [
    "BF16_ATOL",
    "BF16_RTOL",
    "CONFIG_COMPATIBILITY_PROJECTION_SCHEMA",
    "DENOMINATOR_SEMANTIC_FIELDS",
    "FP32_ATOL",
    "FP32_RTOL",
    "FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT",
    "GradientRecord",
    "LOSS_ATOL",
    "LOSS_RTOL",
    "PARITY_PLAN_SCHEMA",
    "PARITY_RECEIPT_SCHEMA",
    "QWEN_COMPONENT_IDENTITY_ATTESTATION_SCHEMA",
    "QWEN_COMPONENT_STABLE_FIELDS",
    "ParityContractError",
    "RngSnapshot",
    "SemanticAtomKey",
    "assert_boundary_only_corruption",
    "assert_dependency_provenance_equal",
    "assert_model_weight_identity_equal",
    "assert_plan_revalidated",
    "attest_qwen_component_identity",
    "attest_v3_runtime_config",
    "bounded_failure",
    "base_model_weight_identity",
    "canonical_json_bytes",
    "compare_gradient_inventories",
    "compare_keyed_logits",
    "compare_semantic_atom_inventories",
    "compare_shared_denominators",
    "compare_tensors",
    "config_compatibility_projection",
    "current_config_compatibility_removed_path_values",
    "finalize_plan",
    "frozen_tolerances",
    "gradient_tolerance",
    "load_strict_json",
    "merged_boundary_forward_inputs",
    "negative_discriminator",
    "repo_identity",
    "restore_rng",
    "selected_logits_by_semantic_key",
    "semantic_atom_inventory",
    "sha256_file",
    "sha256_json",
    "snapshot_rng",
    "snapshot_trainable_gradients",
    "source_owner_identity",
    "tensor_sha256",
    "trainable_value_identity",
    "validate_parity_plan",
    "validate_parity_receipt",
    "validate_v3_config_identity",
    "validate_cross_arm_bf16_loss_term_scalars",
    "validate_config_compatibility_projection",
    "validate_denominator_comparison_artifact",
    "validate_qwen_component_identity_attestation",
    "validate_dependency_provenance",
    "validate_measurement_contract",
    "validate_model_weight_identity",
    "write_strict_json_atomic",
]

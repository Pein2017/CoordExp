#!/usr/bin/env python3
"""One-shot Wave 3 zero-weight protected-loss GPU comparison.

``prepare`` is model-free and reads one authenticated micro-step from the
immutable private Wave 0 cache.  ``run`` is a bounded parent controller.  Its
single child revalidates the plan, installs model/adapter/embedding-delta state
on CPU, derives the exact classified 589-row inventory, and only then publishes
the one-shot marker immediately before GPU setup and the oracle/A/B/B/A arms.
No phase writes a cache and the internal worker refuses direct invocation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from typing import Any
import weakref

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapters import (  # noqa: E402
    build_adapter_setup_plan,
    load_default_adapter_source_gate_evidence,
    setup_dora_adapter,
)
from src.artifacts.provenance import collect_execution_provenance  # noqa: E402
from src.artifacts.resources import collect_resource_snapshot  # noqa: E402
from src.config import load_train_config  # noqa: E402
from src.losses import LossContext, LossRunner  # noqa: E402
from src.qwen import (  # noqa: E402
    QwenImageEncoding,
    attach_qwen_image_processor,
    build_default_special_token_selection,
    build_qwen_forward_inputs,
    load_qwen_components,
    run_qwen_forward,
)
from src.qwen.parity import (  # noqa: E402
    BF16_ATOL,
    BF16_RTOL,
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    EXPECTED_TRAINABLE_STRUCTURE,
    LOSS_ATOL,
    LOSS_RTOL,
    ParityContractError,
    assert_model_weight_identity_equal,
    base_model_weight_identity,
    concrete_trainable_inventory,
    current_config_compatibility_removed_path_values,
    load_strict_json,
    restore_rng,
    sha256_file,
    sha256_json,
    snapshot_rng,
    tensor_sha256,
    validate_concrete_trainable_inventory,
    validate_model_weight_identity,
    write_strict_json_atomic,
)
from src.qwen.special_token_embeddings import (  # noqa: E402
    install_special_token_embedding_deltas,
    load_default_special_token_embedding_source_gate_evidence,
    load_special_token_embedding_deltas,
)
from src.training import pack_cache as pack_cache_module  # noqa: E402
from src.training.pipeline import enable_training_memory_savers  # noqa: E402


PLAN_SCHEMA = "coordexp-swift-wave3-zero-weight-plan-v4"
MARKER_SCHEMA = "coordexp-swift-wave3-zero-weight-attempt-start-marker-v4"
RECEIPT_SCHEMA = "coordexp-swift-wave3-zero-weight-receipt-v4"
PUBLICATION_FAILURE_SCHEMA = "coordexp-swift-wave3-zero-weight-publication-failure-v1"
EXPECTED_BASE_MODEL_WEIGHT_AGGREGATE_SHA256 = (
    "e128f5f42f1a042702efc1eed5a787da36a4d586a17b538671260996056284aa"
)
HISTORICAL_R2_ROOT = REPO_ROOT / (
    "outputs/probes/coordexp_swift/wave3_zero_weight/2026-08-10-r2"
)
HISTORICAL_R2_PLAN_FILE_SHA256 = (
    "fcf83a271ff60ffa13c74d5cd6ab8b1591c4458f862b8c717cb1b55198833a42"
)
HISTORICAL_R2_MARKER_FILE_SHA256 = (
    "ee5f83166398392b3fdefb24e96fc2641fa9a3b711ed9780e3543e9b8185347d"
)
HISTORICAL_R2_RECEIPT_FILE_SHA256 = (
    "3aac3c2b17d00e7e88c25397fb5cdbf26d6c15b1c2749991706ea3df0f982fdd"
)
HISTORICAL_R2_PLAN_SHA256 = (
    "02453b5d1085bc783cfb3ce1e104fb6f5ffc5d27ac4ecb3e379df670b81692f3"
)
HISTORICAL_R2_MARKER_SHA256 = (
    "10478f5586b34ac928705817bb70ee66059d131ebfb72f36a3d122069a54d4e7"
)
HISTORICAL_R2_RECEIPT_SHA256 = (
    "6b96e2fa23b7c2cf12a4d75aaa961734ac6d7bfead242a943c2aff769a668117"
)
HISTORICAL_R2_PLAN_SCHEMA = "coordexp-swift-wave3-zero-weight-plan-v2"
HISTORICAL_R2_MARKER_SCHEMA = "coordexp-swift-wave3-zero-weight-attempt-start-marker-v2"
HISTORICAL_R2_RECEIPT_SCHEMA = "coordexp-swift-wave3-zero-weight-receipt-v2"
HISTORICAL_R3_V3_ROOT = REPO_ROOT / (
    "outputs/probes/coordexp_swift/wave3_zero_weight/2026-08-10-r3-v3"
)
HISTORICAL_R3_V3_PLAN_FILE_SHA256 = (
    "f4936e01729722ce297f9d9a09bd791967fcb741ada827d32b1321deaa86288e"
)
HISTORICAL_R3_V3_MARKER_FILE_SHA256 = (
    "27b2b957d6f96b9a454d3394773fbc65672b673824e14b45c0f5373847c9b297"
)
HISTORICAL_R3_V3_RECEIPT_FILE_SHA256 = (
    "6283baa2b72c0e730bb671179980423564188b9d910778cf29ba66042298958c"
)
HISTORICAL_R3_V3_PLAN_SHA256 = (
    "0d2f9c575968ab1f79c530ac6a501190b183fc6e29fd835cbeb5ce555f98a90a"
)
HISTORICAL_R3_V3_MARKER_SHA256 = (
    "3df128f6ea3b4c12cf822dbcb5258f5a0da0f74d58a7a6bd46a92e09fbf400d7"
)
HISTORICAL_R3_V3_RECEIPT_SHA256 = (
    "b0be5a695abfb4c657d60afba82988f8381159664859e62ea19b55ace0dd148e"
)
HISTORICAL_R3_V3_PLAN_SCHEMA = "coordexp-swift-wave3-zero-weight-plan-v3"
HISTORICAL_R3_V3_MARKER_SCHEMA = (
    "coordexp-swift-wave3-zero-weight-attempt-start-marker-v3"
)
HISTORICAL_R3_V3_RECEIPT_SCHEMA = "coordexp-swift-wave3-zero-weight-receipt-v3"
TERMINAL_STATUSES = frozenset({"passed", "failed"})
FROZEN_CONFIG_PATH = REPO_ROOT / (
    "configs/coordexp_swift/smoke/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_"
    "llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
)
FROZEN_CONFIG_FINGERPRINT = (
    "8f2076bfe2f7f7fcac0c9ea2c210f91595f6bdebf08d03dbb54ef080f529286c"
)
LEGACY_FROZEN_CONFIG_FINGERPRINT = (
    "eaaacb94ca1aa3dc3a2080711da081dac99cd282696ac673bc6f701b828a8bd4"
)
CONFIG_COMPATIBILITY_PROJECTION_SCHEMA = (
    "coordexp-swift-wave3-config-compatibility-projection-v2"
)
RUNTIME_CONFIG_ATTESTATION_SCHEMA = "coordexp-swift-wave3-runtime-config-attestation-v1"
FORWARD_INPUT_PROVIDER_PATH = "training.forward_input_provider_mode"
FORWARD_INPUT_PROVIDER_VALUE = "synchronous"
W0_CACHE_DIR = Path(
    "/tmp/coordexp-wave0-baseline-FIU33RQX/cache/"
    "c6be15b8d7840524829c70e1fa5729accf600ae366d97fccf32778eca84c7de1"
)
W0_CACHE_FINGERPRINT = (
    "c6be15b8d7840524829c70e1fa5729accf600ae366d97fccf32778eca84c7de1"
)
W0_MANIFEST_SHA256 = "636c701efd1cf94af0f88bdb1d86fcef0529d47609c2b48fbf433a7172da7a4e"
W0_CACHE_VERSION = "coordexp-swift-pack-cache-v2"
W0_FIRST_MEASURED_PACK_ORDINAL = 16
W0_FIRST_MEASURED_SELECTION = {
    "planned_step_id": 3,
    "rank": 0,
    "world_size": 8,
    "local_micro_step_index": 0,
    "global_pack_presentation_index": 48,
    "cache_pack_ordinal": W0_FIRST_MEASURED_PACK_ORDINAL,
}
ARM_ORDER = (
    "zero_reference",
    "zero_optimized",
    "nonzero_optimized",
    "nonzero_reference",
)
ORACLE_ARM_NAME = "same_logits_oracle"
COMPARISON_ORDER = (
    "raw_diagnostic_same_logits",
    "zero_reference_base_only_total",
    "zero_optimized_base_only_total",
    "zero_total",
    "zero_gradients",
    "nonzero_total",
    "nonzero_gradients",
    "nonzero_control_graph",
)
PASSED_PHASE_ORDER = (
    "plan_revalidated",
    "shared_gpu_preflight",
    "model_setup_cpu",
    "attempt_started",
    "model_setup_gpu",
    "input_construction",
    "same_logits_oracle",
    *(f"arm:{name}" for name in ARM_ORDER),
    "comparisons",
)
EXPECTED_TRAINABLE_COUNT = EXPECTED_TRAINABLE_PARAMETER_COUNT
SHARED_GPU_BASELINE_SCHEMA = "coordexp-swift-wave3-shared-gpu-baseline-v1"
SHARED_GPU_OBSERVATION_SCHEMA = "coordexp-swift-wave3-shared-gpu-observation-v1"
SHARED_GPU_SUBSET_SWEEP_SCHEMA = "coordexp-swift-wave3-shared-gpu-subset-sweep-v2"
PROCESS_CLEANUP_SCHEMA = "coordexp-swift-wave3-process-cleanup-v1"
SHARED_GPU_TOTAL_MEMORY_MIB = 81920
SHARED_GPU_MAX_PREEXISTING_MEMORY_MIB = 49152
SHARED_GPU_REQUIRED_HEADROOM_MIB = 32768
SHARED_GPU_SAMPLE_COUNT = 2
SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS = 2.0
# Compatibility aliases are intentionally non-authoritative.  Active v4 plans
# and receipts use the shared-GPU names and validators below.
GPU_IDLE_MEMORY_LIMIT_MIB = SHARED_GPU_MAX_PREEXISTING_MEMORY_MIB
GPU_IDLE_UTILIZATION_LIMIT_PERCENT = 100
HOST_RSS_CEILING_BYTES = 64 * 1024**3
DEVICE_MEMORY_CEILING_BYTES = 76 * 1024**3
ARM_WALL_CEILING_SECONDS = 30 * 60
RUN_WALL_CEILING_SECONDS = 3 * 60 * 60
CONTROLLER_POLL_SECONDS = 0.25
CONTROLLER_TERM_GRACE_SECONDS = 5.0
PROCESS_CLEANUP_TIMEOUT_SECONDS = 10.0
PROCESS_CLEANUP_POLL_SECONDS = 0.05
MAX_TRACKED_PROCESSES = 4096
MAX_JSON_BYTES = 16 * 1024 * 1024
EVENT_PREFIX = "WAVE3_EVENT "
CONTROLLER_PID_ENV = "COORDEXP_WAVE3_CONTROLLER_PID"
CONTROLLER_TOKEN_ENV = "COORDEXP_WAVE3_CONTROLLER_TOKEN"
CONTROLLER_BASELINE_ENV = "COORDEXP_WAVE3_SHARED_GPU_BASELINE"
SOURCE_OWNERS = (
    "scripts/probes/coordexp_swift/wave3_zero_weight_gpu.py",
    "src/losses/runner.py",
    "src/losses/token_type_gate.py",
    "src/losses/normalizers.py",
    "src/qwen/forward.py",
    "src/training/pack_cache.py",
)


def shared_gpu_claim_boundary() -> dict[str, Any]:
    """Return the fixed evidence boundary for a shared-load Wave 3 run."""

    return {
        "admissible": ["correctness", "plumbing", "numerical_equivalence"],
        "nonpromotional": [
            "timing",
            "resource_usage",
            "efficiency",
            "zero_weight_performance",
        ],
        "promotion_allowed": False,
        "reason": "shared_preexisting_gpu_load",
    }


class Wave3ProbeError(RuntimeError):
    """A fail-closed Wave 3 probe contract violation."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class Wave3ArtifactPublicationError(Wave3ProbeError):
    """A typed publication failure with exact ownership/reload evidence."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        linked_by_this_call: bool,
        reloaded_exact: bool,
    ) -> None:
        super().__init__(message, code=code)
        self.linked_by_this_call = linked_by_this_call
        self.reloaded_exact = reloaded_exact


@dataclass
class ArmExecution:
    artifact: dict[str, Any]
    total_loss: torch.Tensor
    gradients: dict[str, torch.Tensor]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _finalize(payload: Mapping[str, Any], *, hash_field: str) -> dict[str, Any]:
    result = dict(payload)
    if hash_field in result:
        raise Wave3ProbeError("hash field must be finalized once", code="wave3.hash")
    result[hash_field] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _validate_finalized(
    payload: Mapping[str, Any], *, schema: str, hash_field: str
) -> dict[str, Any]:
    result = dict(payload)
    if result.get("schema") != schema:
        raise Wave3ProbeError("artifact schema is incompatible", code="wave3.schema")
    digest = result.pop(hash_field, None)
    if not _is_sha256(digest):
        raise Wave3ProbeError("artifact hash is missing", code="wave3.hash")
    expected = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    if digest != expected:
        raise Wave3ProbeError("artifact hash does not match", code="wave3.hash")
    result[hash_field] = digest
    return result


def _validate_historical_r2_failed_bytes(
    *, plan_bytes: bytes, marker_bytes: bytes, receipt_bytes: bytes
) -> dict[str, Any]:
    rows = (
        ("plan", plan_bytes, HISTORICAL_R2_PLAN_FILE_SHA256),
        ("marker", marker_bytes, HISTORICAL_R2_MARKER_FILE_SHA256),
        ("receipt", receipt_bytes, HISTORICAL_R2_RECEIPT_FILE_SHA256),
    )
    if any(
        not isinstance(data, bytes)
        or not data
        or len(data) > MAX_JSON_BYTES
        or hashlib.sha256(data).hexdigest() != expected
        for _owner, data, expected in rows
    ):
        raise Wave3ProbeError(
            "historical r2 raw artifact identity drifted",
            code="wave3.historical_r2",
        )
    try:
        plan = json.loads(plan_bytes.decode("utf-8"))
        marker = json.loads(marker_bytes.decode("utf-8"))
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise Wave3ProbeError(
            "historical r2 artifacts are not strict JSON",
            code="wave3.historical_r2",
        ) from exc
    expected_targets = {
        "plan": str(HISTORICAL_R2_ROOT / "plan.json"),
        "attempt_marker": str(HISTORICAL_R2_ROOT / "attempt-marker.json"),
        "receipt": str(HISTORICAL_R2_ROOT / "terminal-receipt.json"),
        "publication_failure": str(
            HISTORICAL_R2_ROOT / "receipt-publication-failure.json"
        ),
    }
    if (
        not isinstance(plan, dict)
        or not isinstance(marker, dict)
        or not isinstance(receipt, dict)
        or plan.get("schema") != HISTORICAL_R2_PLAN_SCHEMA
        or marker.get("schema") != HISTORICAL_R2_MARKER_SCHEMA
        or receipt.get("schema") != HISTORICAL_R2_RECEIPT_SCHEMA
        or plan.get("status") != "prepared"
        or marker.get("status") != "attempt_started"
        or receipt.get("status") != "failed"
        or plan.get("plan_sha256") != HISTORICAL_R2_PLAN_SHA256
        or marker.get("plan_sha256") != HISTORICAL_R2_PLAN_SHA256
        or marker.get("marker_sha256") != HISTORICAL_R2_MARKER_SHA256
        or receipt.get("plan_sha256") != HISTORICAL_R2_PLAN_SHA256
        or receipt.get("receipt_sha256") != HISTORICAL_R2_RECEIPT_SHA256
        or plan.get("artifact_targets") != expected_targets
        or marker.get("receipt_target") != expected_targets["receipt"]
        or marker.get("publication_failure_target")
        != expected_targets["publication_failure"]
        or receipt.get("attempt_marker")
        != {
            "status": "published",
            "path": expected_targets["attempt_marker"],
            "marker_sha256": HISTORICAL_R2_MARKER_SHA256,
        }
        or not isinstance(receipt.get("terminal_reason"), dict)
        or receipt["terminal_reason"].get("code") != "wave3.host_watchdog"
        or receipt.get("source_identity") != plan.get("source_identity")
        or marker.get("source_identity") != plan.get("source_identity")
    ):
        raise Wave3ProbeError(
            "historical r2 failed chain is not the frozen exact chain",
            code="wave3.historical_r2",
        )
    return {
        "status": "historical_non_executable",
        "root": str(HISTORICAL_R2_ROOT),
        "plan_file_sha256": HISTORICAL_R2_PLAN_FILE_SHA256,
        "plan_sha256": HISTORICAL_R2_PLAN_SHA256,
        "marker_file_sha256": HISTORICAL_R2_MARKER_FILE_SHA256,
        "marker_sha256": HISTORICAL_R2_MARKER_SHA256,
        "receipt_file_sha256": HISTORICAL_R2_RECEIPT_FILE_SHA256,
        "receipt_sha256": HISTORICAL_R2_RECEIPT_SHA256,
        "terminal_reason_code": "wave3.host_watchdog",
    }


def load_historical_r2_failed_evidence(
    root: str | Path = HISTORICAL_R2_ROOT,
) -> dict[str, Any]:
    requested = Path(root).expanduser().resolve()
    if requested != HISTORICAL_R2_ROOT.resolve():
        raise Wave3ProbeError(
            "historical r2 reader only accepts the frozen root",
            code="wave3.historical_r2",
        )
    paths = tuple(
        requested / name
        for name in ("plan.json", "attempt-marker.json", "terminal-receipt.json")
    )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise Wave3ProbeError(
            "historical r2 artifact path is unavailable",
            code="wave3.historical_r2",
        )
    try:
        plan_bytes, marker_bytes, receipt_bytes = (path.read_bytes() for path in paths)
    except OSError as exc:
        raise Wave3ProbeError(
            "historical r2 artifacts cannot be read",
            code="wave3.historical_r2",
        ) from exc
    return _validate_historical_r2_failed_bytes(
        plan_bytes=plan_bytes,
        marker_bytes=marker_bytes,
        receipt_bytes=receipt_bytes,
    )


def load_historical_r3_v3_failed_evidence(
    root: str | Path = HISTORICAL_R3_V3_ROOT,
) -> dict[str, Any]:
    """Authenticate the consumed v3 chain without treating it as executable."""

    requested = Path(root).expanduser().resolve()
    if requested != HISTORICAL_R3_V3_ROOT.resolve():
        raise Wave3ProbeError(
            "historical r3-v3 reader only accepts the frozen root",
            code="wave3.historical_r3_v3",
        )
    paths = tuple(
        requested / name
        for name in ("plan.json", "attempt-marker.json", "terminal-receipt.json")
    )
    expected_file_hashes = (
        HISTORICAL_R3_V3_PLAN_FILE_SHA256,
        HISTORICAL_R3_V3_MARKER_FILE_SHA256,
        HISTORICAL_R3_V3_RECEIPT_FILE_SHA256,
    )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise Wave3ProbeError(
            "historical r3-v3 artifact path is unavailable",
            code="wave3.historical_r3_v3",
        )
    try:
        raw = tuple(path.read_bytes() for path in paths)
    except OSError as exc:
        raise Wave3ProbeError(
            "historical r3-v3 artifacts cannot be read",
            code="wave3.historical_r3_v3",
        ) from exc
    if any(
        not data
        or len(data) > MAX_JSON_BYTES
        or hashlib.sha256(data).hexdigest() != expected
        for data, expected in zip(raw, expected_file_hashes, strict=True)
    ):
        raise Wave3ProbeError(
            "historical r3-v3 raw artifact identity drifted",
            code="wave3.historical_r3_v3",
        )
    try:
        plan, marker, receipt = (json.loads(data.decode("utf-8")) for data in raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise Wave3ProbeError(
            "historical r3-v3 artifacts are not strict JSON",
            code="wave3.historical_r3_v3",
        ) from exc
    targets = {
        "plan": str(requested / "plan.json"),
        "attempt_marker": str(requested / "attempt-marker.json"),
        "receipt": str(requested / "terminal-receipt.json"),
        "publication_failure": str(requested / "receipt-publication-failure.json"),
    }
    if (
        not isinstance(plan, dict)
        or not isinstance(marker, dict)
        or not isinstance(receipt, dict)
        or plan.get("schema") != HISTORICAL_R3_V3_PLAN_SCHEMA
        or marker.get("schema") != HISTORICAL_R3_V3_MARKER_SCHEMA
        or receipt.get("schema") != HISTORICAL_R3_V3_RECEIPT_SCHEMA
        or plan.get("status") != "prepared"
        or marker.get("status") != "attempt_started"
        or receipt.get("status") != "failed"
        or plan.get("plan_sha256") != HISTORICAL_R3_V3_PLAN_SHA256
        or marker.get("plan_sha256") != HISTORICAL_R3_V3_PLAN_SHA256
        or marker.get("marker_sha256") != HISTORICAL_R3_V3_MARKER_SHA256
        or receipt.get("plan_sha256") != HISTORICAL_R3_V3_PLAN_SHA256
        or receipt.get("receipt_sha256") != HISTORICAL_R3_V3_RECEIPT_SHA256
        or plan.get("artifact_targets") != targets
        or marker.get("receipt_target") != targets["receipt"]
        or marker.get("publication_failure_target") != targets["publication_failure"]
        or receipt.get("attempt_marker")
        != {
            "status": "published",
            "path": targets["attempt_marker"],
            "marker_sha256": HISTORICAL_R3_V3_MARKER_SHA256,
        }
        or not isinstance(receipt.get("terminal_reason"), dict)
        or receipt["terminal_reason"].get("code") != "wave3.accelerator"
        or marker.get("source_identity") != plan.get("source_identity")
        or receipt.get("source_identity") != plan.get("source_identity")
    ):
        raise Wave3ProbeError(
            "historical r3-v3 failed chain is not the frozen exact chain",
            code="wave3.historical_r3_v3",
        )
    return {
        "status": "historical_non_executable",
        "root": str(HISTORICAL_R3_V3_ROOT),
        "plan_file_sha256": HISTORICAL_R3_V3_PLAN_FILE_SHA256,
        "plan_sha256": HISTORICAL_R3_V3_PLAN_SHA256,
        "marker_file_sha256": HISTORICAL_R3_V3_MARKER_FILE_SHA256,
        "marker_sha256": HISTORICAL_R3_V3_MARKER_SHA256,
        "receipt_file_sha256": HISTORICAL_R3_V3_RECEIPT_FILE_SHA256,
        "receipt_sha256": HISTORICAL_R3_V3_RECEIPT_SHA256,
        "terminal_reason_code": "wave3.accelerator",
    }


def frozen_config_compatibility_projection() -> dict[str, Any]:
    """Return the sole accepted current-to-legacy config projection."""

    return {
        "schema": CONFIG_COMPATIBILITY_PROJECTION_SCHEMA,
        "current_config_sha256": FROZEN_CONFIG_FINGERPRINT,
        "removed_path_values": current_config_compatibility_removed_path_values(),
        "projected_config_sha256": LEGACY_FROZEN_CONFIG_FINGERPRINT,
        "legacy_config_fingerprint": LEGACY_FROZEN_CONFIG_FINGERPRINT,
        "policy": "remove_exact_enumerated_later_strict_defaults",
    }


def validate_config_compatibility_projection(
    value: Mapping[str, Any],
    *,
    resolved_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate exactly the enumerated later compatibility defaults."""

    expected = frozen_config_compatibility_projection()
    projection = _mapping(value, "config.compatibility_projection")
    _require_exact_fields(
        projection,
        set(expected),
        owner="config.compatibility_projection",
        code="wave3.config_projection",
    )
    if dict(projection) != expected:
        raise Wave3ProbeError(
            "config compatibility projection is not the frozen exact projection",
            code="wave3.config_projection",
        )
    if resolved_config is not None:
        current = json.loads(canonical_json_bytes(dict(resolved_config)))
        if sha256_json(current) != FROZEN_CONFIG_FINGERPRINT:
            raise Wave3ProbeError(
                "current resolved config identity drifted",
                code="wave3.config_identity",
            )
        _remove_exact_config_path_values(
            current,
            expected["removed_path_values"],
        )
        if sha256_json(current) != LEGACY_FROZEN_CONFIG_FINGERPRINT:
            raise Wave3ProbeError(
                "resolved config differs beyond the enumerated defaults",
                code="wave3.config_projection",
            )
    return dict(projection)


def build_config_compatibility_projection(
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and revalidate the exact enumerated compatibility projection."""

    projection = frozen_config_compatibility_projection()
    return validate_config_compatibility_projection(
        projection,
        resolved_config=resolved_config,
    )


def _remove_exact_config_path_values(
    config: dict[str, Any],
    path_values: Sequence[Mapping[str, Any]],
) -> None:
    for row in path_values:
        path = row.get("path")
        if not isinstance(path, str) or not path:
            raise Wave3ProbeError(
                "config projection path is invalid",
                code="wave3.config_projection",
            )
        parts = path.split(".")
        owner: Any = config
        for part in parts[:-1]:
            if not isinstance(owner, dict) or part not in owner:
                raise Wave3ProbeError(
                    "config projection path is missing",
                    code="wave3.config_projection",
                )
            owner = owner[part]
        leaf = parts[-1]
        if not isinstance(owner, dict) or leaf not in owner:
            raise Wave3ProbeError(
                "config projection path is missing",
                code="wave3.config_projection",
            )
        if owner[leaf] != row.get("value"):
            if path == FORWARD_INPUT_PROVIDER_PATH:
                raise Wave3ProbeError(
                    "runtime provider mode is not synchronous",
                    code="wave3.config_provider",
                )
            raise Wave3ProbeError(
                "config projection value is not the exact compatibility default",
                code="wave3.config_projection",
            )
        del owner[leaf]


def frozen_runtime_config_attestation(
    projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validated_projection = validate_config_compatibility_projection(
        frozen_config_compatibility_projection() if projection is None else projection
    )
    return {
        "schema": RUNTIME_CONFIG_ATTESTATION_SCHEMA,
        "status": "passed",
        "config_fingerprint": FROZEN_CONFIG_FINGERPRINT,
        "resolved_config_sha256": FROZEN_CONFIG_FINGERPRINT,
        "field_path": FORWARD_INPUT_PROVIDER_PATH,
        "required_value": FORWARD_INPUT_PROVIDER_VALUE,
        "resolved_value": FORWARD_INPUT_PROVIDER_VALUE,
        "compatibility_projection_sha256": sha256_json(validated_projection),
    }


def attest_runtime_config(
    config_identity: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reassert the full current identity and provider immediately pre-marker."""

    config = _mapping(config_identity, "config")
    projection = validate_config_compatibility_projection(
        _mapping(
            config.get("compatibility_projection"), "config.compatibility_projection"
        ),
        resolved_config=resolved_config,
    )
    expected = frozen_runtime_config_attestation(projection)
    if (
        config.get("fingerprint") != FROZEN_CONFIG_FINGERPRINT
        or config.get("resolved_sha256") != FROZEN_CONFIG_FINGERPRINT
        or config.get("forward_input_provider_mode") != FORWARD_INPUT_PROVIDER_VALUE
        or config.get("runtime_config_attestation") != expected
    ):
        raise Wave3ProbeError(
            "recorded runtime config identity drifted",
            code="wave3.config_identity",
        )
    return expected


def _attest_fresh_runtime_config_immediately_before_marker(
    plan: Mapping[str, Any],
    *,
    initial_resolved_config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Reload and bind the full live config after CPU setup, before GPU work."""

    config_identity = _mapping(plan.get("config"), "config")
    fresh = load_train_config(config_identity["entry_path"])
    fresh_config = dict(fresh.config_dict)
    if (
        fresh.fingerprint != FROZEN_CONFIG_FINGERPRINT
        or sha256_json(fresh_config) != FROZEN_CONFIG_FINGERPRINT
        or fresh_config != dict(initial_resolved_config)
        or fresh.config.training.forward_input_provider_mode
        != FORWARD_INPUT_PROVIDER_VALUE
        or fresh.config.training.precision != "bf16"
    ):
        raise Wave3ProbeError(
            "fresh pre-marker runtime config identity drifted",
            code="wave3.config_identity",
        )
    attestation = attest_runtime_config(config_identity, fresh_config)
    return fresh, attestation


def validate_runtime_config_attestation(value: Any) -> dict[str, Any]:
    attestation = _mapping(value, "runtime_config_attestation")
    expected = frozen_runtime_config_attestation()
    _require_exact_fields(
        attestation,
        set(expected),
        owner="runtime_config_attestation",
        code="wave3.config_attestation",
    )
    if dict(attestation) != expected:
        raise Wave3ProbeError(
            "runtime config attestation drifted",
            code="wave3.config_attestation",
        )
    return dict(attestation)


def _validate_model_weight_identity(value: Any) -> dict[str, Any]:
    identity = _mapping(value, "model_weight_identity")
    try:
        return validate_model_weight_identity(identity)
    except Exception as exc:
        raise Wave3ProbeError(
            "base-model weight identity is invalid",
            code="wave3.model_weight_identity",
        ) from exc


def _assert_current_model_weight_identity(
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        validated = validate_model_weight_identity(expected)
        observed = base_model_weight_identity(validated["root"])
        assert_model_weight_identity_equal(validated, observed)
    except Exception as exc:
        raise Wave3ProbeError(
            "base-model weight identity changed after plan preparation",
            code="wave3.model_weight_identity",
        ) from exc
    return observed


def validate_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    plan = _validate_finalized(payload, schema=PLAN_SCHEMA, hash_field="plan_sha256")
    _require_exact_fields(
        plan,
        {
            "schema",
            "status",
            "model_weight_identity",
            "config",
            "cache",
            "workload",
            "loss_contract",
            "execution_contract",
            "artifact_targets",
            "source_identity",
            "provenance",
            "plan_sha256",
        },
        owner="plan",
        code="wave3.plan_fields",
    )
    if plan["status"] != "prepared":
        raise Wave3ProbeError("plan status is invalid", code="wave3.plan_fields")
    model_weight_identity = _validate_model_weight_identity(
        plan["model_weight_identity"]
    )
    if (
        model_weight_identity["aggregate_sha256"]
        != EXPECTED_BASE_MODEL_WEIGHT_AGGREGATE_SHA256
    ):
        raise Wave3ProbeError(
            "base-model weight identity is not the frozen payload",
            code="wave3.model_weight_identity",
        )
    config = _mapping(plan["config"], "config")
    _require_exact_fields(
        config,
        {
            "entry_path",
            "fingerprint",
            "resolved_sha256",
            "forward_input_provider_mode",
            "compatibility_projection",
            "runtime_config_attestation",
            "model_identity",
        },
        owner="config",
        code="wave3.config_identity",
    )
    projection = validate_config_compatibility_projection(
        _mapping(
            config.get("compatibility_projection"),
            "config.compatibility_projection",
        )
    )
    attestation = validate_runtime_config_attestation(
        config.get("runtime_config_attestation")
    )
    if (
        Path(str(config["entry_path"])).resolve() != FROZEN_CONFIG_PATH.resolve()
        or config["fingerprint"] != FROZEN_CONFIG_FINGERPRINT
        or config["resolved_sha256"] != FROZEN_CONFIG_FINGERPRINT
        or config["forward_input_provider_mode"] != FORWARD_INPUT_PROVIDER_VALUE
        or attestation != frozen_runtime_config_attestation(projection)
        or not isinstance(config["model_identity"], Mapping)
        or not config["model_identity"]
        or config["model_identity"].get("load_model") is not False
        or not _is_sha256(config["model_identity"].get("base_config_sha256"))
        or not _is_sha256(config["model_identity"].get("tokenizer_sha256"))
        or model_weight_identity["root"]
        != str(Path(str(config["model_identity"].get("base_model_path"))).resolve())
    ):
        raise Wave3ProbeError(
            "frozen config identity drifted", code="wave3.config_identity"
        )
    cache = _mapping(plan["cache"], "cache")
    _require_exact_fields(
        cache,
        {
            "path",
            "access",
            "version",
            "fingerprint",
            "manifest_sha256",
            "chunk_sha256",
            "shared_cache_writes",
        },
        owner="cache",
        code="wave3.cache_identity",
    )
    if (
        Path(str(cache["path"])).resolve() != W0_CACHE_DIR.resolve()
        or cache["access"] != "read_only_private_immutable"
        or cache["version"] != W0_CACHE_VERSION
        or cache["fingerprint"] != W0_CACHE_FINGERPRINT
        or cache["manifest_sha256"] != W0_MANIFEST_SHA256
        or not _is_sha256(cache["chunk_sha256"])
        or cache["shared_cache_writes"] is not False
    ):
        raise Wave3ProbeError(
            "frozen cache identity drifted", code="wave3.cache_identity"
        )
    workload = _mapping(plan["workload"], "workload")
    _require_exact_fields(
        workload,
        {
            "selection",
            "pack_index",
            "example_ids",
            "segment_bounds",
            "segment_boundaries",
            "pack_length",
            "input_sha256",
            "supervision_sha256",
        },
        owner="workload",
        code="wave3.workload_hash",
    )
    if workload.get("selection") != W0_FIRST_MEASURED_SELECTION:
        raise Wave3ProbeError("Wave 0 selection drifted", code="wave3.selection")
    if not _is_sha256(workload.get("input_sha256")) or not _is_sha256(
        workload.get("supervision_sha256")
    ):
        raise Wave3ProbeError("workload hashes are invalid", code="wave3.workload_hash")
    execution = _mapping(plan["execution_contract"], "execution_contract")
    _require_exact_fields(
        execution,
        {
            "arm_order",
            "oracle_arm_precedes_measured_arms",
            "model_forward_count_per_arm",
            "backward_count_per_measured_arm",
            "state_restore_between_arms",
            "wall_clock",
            "arm_wall_ceiling_seconds",
            "run_wall_ceiling_seconds",
            "host_rss_ceiling_bytes",
            "device_memory_ceiling_bytes",
            "controller_poll_seconds",
            "process_cleanup_timeout_seconds",
            "max_tracked_processes",
            "terminal_order",
            "shared_gpu_total_memory_mib",
            "shared_gpu_max_preexisting_memory_mib",
            "shared_gpu_required_headroom_mib",
            "shared_gpu_preflight_sample_count",
            "shared_gpu_minimum_sample_interval_seconds",
            "shared_gpu_utilization_disposition",
            "shared_gpu_baseline_process_policy",
            "baseline_process_action",
            "claim_boundary",
            "requested_device",
            "retry",
            "sample_switch",
            "tolerance_change",
            "prepare_argv",
            "prepare_argv_sha256",
            "run_argv",
            "run_argv_sha256",
            "controller_argv",
            "controller_argv_sha256",
        },
        owner="execution_contract",
        code="wave3.execution_contract",
    )
    if tuple(execution.get("arm_order", ())) != ARM_ORDER:
        raise Wave3ProbeError("arm order drifted", code="wave3.arm_order")
    if (
        execution.get("model_forward_count_per_arm") != 1
        or execution.get("backward_count_per_measured_arm") != 1
        or execution.get("oracle_arm_precedes_measured_arms") is not True
        or execution.get("state_restore_between_arms")
        != "trainable_values_named_buffers_train_mode_and_rng_exact"
        or execution.get("wall_clock")
        != "external_monotonic_watchdog_and_cuda_synchronized_perf_counter"
        or execution.get("arm_wall_ceiling_seconds") != ARM_WALL_CEILING_SECONDS
        or execution.get("run_wall_ceiling_seconds") != RUN_WALL_CEILING_SECONDS
        or execution.get("host_rss_ceiling_bytes") != HOST_RSS_CEILING_BYTES
        or execution.get("device_memory_ceiling_bytes") != DEVICE_MEMORY_CEILING_BYTES
        or execution.get("controller_poll_seconds") != CONTROLLER_POLL_SECONDS
        or execution.get("process_cleanup_timeout_seconds")
        != PROCESS_CLEANUP_TIMEOUT_SECONDS
        or execution.get("max_tracked_processes") != MAX_TRACKED_PROCESSES
        or execution.get("terminal_order")
        != "process_tree_cleanup_then_two_sample_gpu_subset_sweep"
        or execution.get("shared_gpu_total_memory_mib") != SHARED_GPU_TOTAL_MEMORY_MIB
        or execution.get("shared_gpu_max_preexisting_memory_mib")
        != SHARED_GPU_MAX_PREEXISTING_MEMORY_MIB
        or execution.get("shared_gpu_required_headroom_mib")
        != SHARED_GPU_REQUIRED_HEADROOM_MIB
        or execution.get("shared_gpu_preflight_sample_count") != SHARED_GPU_SAMPLE_COUNT
        or execution.get("shared_gpu_minimum_sample_interval_seconds")
        != SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS
        or execution.get("shared_gpu_utilization_disposition")
        != "observational_only_not_admission_or_promotion"
        or execution.get("shared_gpu_baseline_process_policy")
        != "stable_exact_gpu_uuid_driver_pid_then_post_probe_subset"
        or execution.get("baseline_process_action")
        != "observe_only_never_signal_or_reset"
        or execution.get("claim_boundary") != shared_gpu_claim_boundary()
        or execution.get("retry") is not False
        or execution.get("sample_switch") is not False
        or execution.get("tolerance_change") is not False
    ):
        raise Wave3ProbeError(
            "one-forward invariant drifted", code="wave3.forward_count"
        )
    for owner in ("prepare", "run", "controller"):
        argv = execution.get(f"{owner}_argv")
        if not isinstance(argv, list) or not all(
            isinstance(item, str) and item for item in argv
        ):
            raise Wave3ProbeError(
                f"{owner} argv is invalid", code=f"wave3.{owner}_argv"
            )
        if execution.get(f"{owner}_argv_sha256") != sha256_json(argv):
            raise Wave3ProbeError(
                f"{owner} argv hash drifted", code=f"wave3.{owner}_argv"
            )
    loss = _mapping(plan["loss_contract"], "loss_contract")
    _require_exact_fields(
        loss,
        {
            "zero_weight",
            "nonzero_control_weight",
            "raw_diagnostic_tolerance",
            "bf16_derived_tolerance",
            "raw_oracle",
            "zero_total_reference",
            "optimized_zero_total",
            "expected_trainable_gradient_count",
        },
        owner="loss_contract",
        code="wave3.loss_contract",
    )
    if (
        loss.get("zero_weight") != 0.0
        or not isinstance(loss.get("nonzero_control_weight"), float)
        or loss.get("nonzero_control_weight", 0.0) <= 0.0
        or loss.get("expected_trainable_gradient_count") != EXPECTED_TRAINABLE_COUNT
    ):
        raise Wave3ProbeError("loss contract drifted", code="wave3.loss_contract")
    if loss.get("raw_diagnostic_tolerance") != {"rtol": LOSS_RTOL, "atol": LOSS_ATOL}:
        raise Wave3ProbeError(
            "raw diagnostic tolerance drifted", code="wave3.tolerance"
        )
    if loss.get("bf16_derived_tolerance") != {"rtol": BF16_RTOL, "atol": BF16_ATOL}:
        raise Wave3ProbeError("BF16 tolerance drifted", code="wave3.tolerance")
    targets = _mapping(plan["artifact_targets"], "artifact_targets")
    _require_exact_fields(
        targets,
        {"plan", "attempt_marker", "receipt", "publication_failure"},
        owner="artifact_targets",
        code="wave3.artifact_targets",
    )
    if len(set(str(value) for value in targets.values())) != 4:
        raise Wave3ProbeError(
            "artifact targets must be distinct", code="wave3.artifact_targets"
        )
    for value in targets.values():
        if not Path(str(value)).is_absolute():
            raise Wave3ProbeError(
                "artifact targets must be absolute", code="wave3.artifact_targets"
            )
    expected_prepare_argv = _prepare_argv(
        config_path=config["entry_path"],
        cache_dir=cache["path"],
        plan_path=targets["plan"],
        receipt_path=targets["receipt"],
        attempt_marker_path=targets["attempt_marker"],
        publication_failure_path=targets["publication_failure"],
        device=execution["requested_device"],
    )
    expected_run_argv = _artifact_argv(
        command="run",
        plan_path=targets["plan"],
        receipt_path=targets["receipt"],
        attempt_marker_path=targets["attempt_marker"],
        publication_failure_path=targets["publication_failure"],
        device=execution["requested_device"],
    )
    expected_controller_argv = _artifact_argv(
        command="controller",
        plan_path=targets["plan"],
        receipt_path=targets["receipt"],
        attempt_marker_path=targets["attempt_marker"],
        publication_failure_path=targets["publication_failure"],
        device=execution["requested_device"],
    )
    if execution["prepare_argv"] != expected_prepare_argv:
        raise Wave3ProbeError(
            "prepare argv is not target-bound", code="wave3.prepare_argv"
        )
    if execution["run_argv"] != expected_run_argv:
        raise Wave3ProbeError("run argv is not target-bound", code="wave3.run_argv")
    if execution["controller_argv"] != expected_controller_argv:
        raise Wave3ProbeError(
            "controller argv is not target-bound", code="wave3.controller_argv"
        )
    source = _mapping(plan["source_identity"], "source_identity")
    if set(source) != set(SOURCE_OWNERS) or not all(
        _is_sha256(value) for value in source.values()
    ):
        raise Wave3ProbeError(
            "source identity is incomplete", code="wave3.source_identity"
        )
    provenance = _mapping(plan["provenance"], "provenance")
    if set(provenance) != {"schema_version", "repository", "dependencies", "runtime"}:
        raise Wave3ProbeError(
            "provenance identity is incomplete", code="wave3.provenance"
        )
    return plan


def validate_marker(
    payload: Mapping[str, Any], *, expected_plan: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    marker = _validate_finalized(
        payload, schema=MARKER_SCHEMA, hash_field="marker_sha256"
    )
    _require_exact_fields(
        marker,
        {
            "schema",
            "status",
            "plan_sha256",
            "model_weight_identity",
            "receipt_target",
            "publication_failure_target",
            "requested_device",
            "command_identity",
            "publication_owner_token_sha256",
            "source_identity",
            "runtime_config_attestation",
            "runtime_source_identities",
            "cpu_installation",
            "provenance_sha256",
            "shared_gpu_preflight",
            "pre_marker_gpu_observation",
            "claim_boundary",
            "concrete_trainable_inventory",
            "initial_buffer_inventory",
            "published_monotonic_ns",
            "marker_sha256",
        },
        owner="attempt_marker",
        code="wave3.marker",
    )
    if marker.get("status") != "attempt_started":
        raise Wave3ProbeError("attempt marker status is invalid", code="wave3.marker")
    runtime_config_attestation = validate_runtime_config_attestation(
        marker.get("runtime_config_attestation")
    )
    model_weight_identity = _validate_model_weight_identity(
        marker.get("model_weight_identity")
    )
    if not _is_sha256(marker.get("plan_sha256")):
        raise Wave3ProbeError("attempt marker plan is invalid", code="wave3.marker")
    for field in ("receipt_target", "publication_failure_target"):
        if (
            not isinstance(marker.get(field), str)
            or not Path(marker[field]).is_absolute()
        ):
            raise Wave3ProbeError(
                "attempt marker target is invalid", code="wave3.marker"
            )
    if marker.get("requested_device") != _parse_cuda_device_text(
        str(marker.get("requested_device"))
    ):
        raise Wave3ProbeError("attempt marker device is invalid", code="wave3.marker")
    command = _mapping(marker.get("command_identity"), "command_identity")
    if not _is_sha256(marker.get("publication_owner_token_sha256")):
        raise Wave3ProbeError(
            "attempt marker publication owner is invalid", code="wave3.marker"
        )
    _require_exact_fields(
        command,
        {
            "run_argv",
            "run_argv_sha256",
            "controller_argv",
            "controller_argv_sha256",
            "worker_argv",
            "worker_argv_sha256",
        },
        owner="command_identity",
        code="wave3.marker",
    )
    for owner in ("run", "controller", "worker"):
        argv = command[f"{owner}_argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) and item for item in argv)
            or command[f"{owner}_argv_sha256"] != sha256_json(argv)
        ):
            raise Wave3ProbeError(
                "marker command identity drifted", code="wave3.marker"
            )
    _validate_source_identity(marker.get("source_identity"))
    _validate_runtime_source_identities(marker.get("runtime_source_identities"))
    _validate_cpu_installation(marker.get("cpu_installation"))
    _validate_concrete_inventory(marker.get("concrete_trainable_inventory"))
    _validate_buffer_inventory(marker.get("initial_buffer_inventory"))
    shared_gpu_preflight = _validate_shared_gpu_baseline(
        marker.get("shared_gpu_preflight")
    )
    _validate_shared_gpu_observation(
        marker.get("pre_marker_gpu_observation"),
        baseline=shared_gpu_preflight,
        expected_stage="pre_marker",
    )
    if marker.get("claim_boundary") != shared_gpu_claim_boundary():
        raise Wave3ProbeError(
            "attempt marker claim boundary drifted", code="wave3.marker"
        )
    if not _is_sha256(marker.get("provenance_sha256")) or not _is_nonnegative_int(
        marker.get("published_monotonic_ns")
    ):
        raise Wave3ProbeError(
            "attempt marker identity is incomplete", code="wave3.marker"
        )
    if expected_plan is not None:
        plan = validate_plan(expected_plan)
        execution = _mapping(plan["execution_contract"], "execution_contract")
        targets = _mapping(plan["artifact_targets"], "artifact_targets")
        expected_worker_argv = _artifact_argv(
            command="_worker",
            plan_path=targets["plan"],
            receipt_path=targets["receipt"],
            attempt_marker_path=targets["attempt_marker"],
            publication_failure_path=targets["publication_failure"],
            device=execution["requested_device"],
        )
        prepared_model_identity = _mapping(
            plan["config"]["model_identity"], "config.model_identity"
        )
        loaded_model_identity = _mapping(
            marker["runtime_source_identities"]["model"]["artifact"],
            "runtime.model_identity",
        )
        stable_model_fields = (
            "base_model_path",
            "base_config_sha256",
            "tokenizer_sha256",
            "attn_implementation",
            "processor",
            "model",
            "tokens",
            "package_versions",
        )
        if (
            marker["plan_sha256"] != plan["plan_sha256"]
            or model_weight_identity != plan["model_weight_identity"]
            or marker["receipt_target"] != targets["receipt"]
            or marker["publication_failure_target"] != targets["publication_failure"]
            or marker["requested_device"] != execution["requested_device"]
            or marker["source_identity"] != plan["source_identity"]
            or marker["claim_boundary"] != execution["claim_boundary"]
            or runtime_config_attestation
            != plan["config"]["runtime_config_attestation"]
            or command["run_argv"] != execution["run_argv"]
            or command["controller_argv"] != execution["controller_argv"]
            or command["worker_argv"] != expected_worker_argv
            or any(
                prepared_model_identity.get(field) != loaded_model_identity.get(field)
                for field in stable_model_fields
            )
        ):
            raise Wave3ProbeError("attempt marker plan drifted", code="wave3.marker")
    return marker


def validate_receipt(
    payload: Mapping[str, Any], *, expected_plan: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    receipt = _validate_finalized(
        payload, schema=RECEIPT_SCHEMA, hash_field="receipt_sha256"
    )
    _require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "plan_sha256",
            "model_weight_identity",
            "plan_binding",
            "source_identity",
            "terminal_reason",
            "attempt_marker",
            "completed_phases",
            "completed_arm_order",
            "counts",
            "oracle",
            "arms",
            "comparisons",
            "efficiency",
            "runtime",
            "receipt_sha256",
        },
        owner="receipt",
        code="wave3.receipt",
    )
    if receipt.get("status") not in TERMINAL_STATUSES:
        raise Wave3ProbeError("receipt status is not terminal", code="wave3.receipt")
    receipt_weight_identity = receipt.get("model_weight_identity")
    _validate_source_identity(
        receipt.get("source_identity"), allow_empty=expected_plan is None
    )
    plan_binding = _mapping(receipt.get("plan_binding"), "plan_binding")
    _require_exact_fields(
        plan_binding,
        {
            "plan_sha256",
            "source_identity_sha256",
            "prepare_argv_sha256",
            "run_argv_sha256",
            "controller_argv_sha256",
            "model_weight_identity_sha256",
        },
        owner="plan_binding",
        code="wave3.receipt",
    )
    if expected_plan is not None:
        plan = validate_plan(expected_plan)
        execution = _mapping(plan["execution_contract"], "execution_contract")
        expected_binding = {
            "plan_sha256": plan["plan_sha256"],
            "source_identity_sha256": sha256_json(plan["source_identity"]),
            "prepare_argv_sha256": execution["prepare_argv_sha256"],
            "run_argv_sha256": execution["run_argv_sha256"],
            "controller_argv_sha256": execution["controller_argv_sha256"],
            "model_weight_identity_sha256": plan["model_weight_identity"][
                "aggregate_sha256"
            ],
        }
        validated_receipt_weight_identity = _validate_model_weight_identity(
            receipt_weight_identity
        )
        if (
            receipt.get("plan_sha256") != plan["plan_sha256"]
            or validated_receipt_weight_identity != plan["model_weight_identity"]
            or receipt["source_identity"] != plan["source_identity"]
            or dict(plan_binding) != expected_binding
        ):
            raise Wave3ProbeError("receipt plan drifted", code="wave3.receipt")
    elif (
        receipt.get("plan_sha256") is not None
        or receipt_weight_identity is not None
        or any(value is not None for value in plan_binding.values())
    ):
        raise Wave3ProbeError("unbound receipt claims a plan", code="wave3.receipt")
    marker_ref = _validate_marker_reference(
        receipt.get("attempt_marker"), expected_plan
    )
    phases = receipt.get("completed_phases")
    if not isinstance(phases, list) or not all(
        isinstance(item, str) for item in phases
    ):
        raise Wave3ProbeError("receipt phases are invalid", code="wave3.receipt")
    if phases != list(PASSED_PHASE_ORDER[: len(phases)]):
        raise Wave3ProbeError(
            "receipt phases are not an exact prefix", code="wave3.receipt"
        )
    completed_arms = receipt.get("completed_arm_order")
    if not isinstance(completed_arms, list) or completed_arms != list(
        ARM_ORDER[: len(completed_arms)]
    ):
        raise Wave3ProbeError("receipt arm order is invalid", code="wave3.receipt")
    counts = _mapping(receipt.get("counts", {}), "counts")
    _require_exact_fields(
        counts,
        {"model_forwards", "backwards"},
        owner="counts",
        code="wave3.receipt",
    )
    oracle = _validate_oracle_receipt(receipt.get("oracle"))
    arms = _mapping(receipt.get("arms"), "arms")
    _require_exact_fields(arms, set(ARM_ORDER), owner="arms", code="wave3.receipt")
    validated_arms = {
        name: _validate_arm_envelope(name, arms[name]) for name in ARM_ORDER
    }
    completed_from_artifacts = [
        name for name in ARM_ORDER if validated_arms[name]["status"] == "completed"
    ]
    if completed_from_artifacts != completed_arms:
        raise Wave3ProbeError(
            "receipt arm evidence contradicts order", code="wave3.receipt"
        )
    expected_forwards = (1 if oracle["status"] == "completed" else 0) + len(
        completed_from_artifacts
    )
    if counts != {
        "model_forwards": expected_forwards,
        "backwards": len(completed_from_artifacts),
    }:
        raise Wave3ProbeError(
            "receipt counts are not recomputable", code="wave3.receipt"
        )
    runtime = _validate_runtime_receipt(
        receipt.get("runtime"), allow_limit_exceeded=receipt["status"] == "failed"
    )
    comparisons = _validate_comparison_receipts(
        receipt.get("comparisons"), oracle=oracle, arms=validated_arms
    )
    efficiency = _validate_efficiency_receipt(
        receipt.get("efficiency"), arms=validated_arms
    )
    if (marker_ref["status"] == "published") is not ("attempt_started" in phases):
        raise Wave3ProbeError(
            "receipt marker phase relation drifted", code="wave3.receipt"
        )
    if (oracle["status"] == "completed") is not ("same_logits_oracle" in phases):
        raise Wave3ProbeError(
            "receipt oracle phase relation drifted", code="wave3.receipt"
        )
    phase_arms = [name for name in ARM_ORDER if f"arm:{name}" in phases]
    if phase_arms != completed_from_artifacts:
        raise Wave3ProbeError(
            "receipt arm phase relation drifted", code="wave3.receipt"
        )
    if (runtime["status"] in {"cpu_ready", "gpu_ready"}) is not (
        "model_setup_cpu" in phases
    ) or (runtime["status"] == "gpu_ready") is not ("model_setup_gpu" in phases):
        raise Wave3ProbeError(
            "receipt runtime phase relation drifted", code="wave3.receipt"
        )
    if "comparisons" in phases:
        if (
            any(value is None for value in comparisons.values())
            or efficiency["status"] != "completed"
        ):
            raise Wave3ProbeError(
                "receipt comparison phase is incomplete", code="wave3.receipt"
            )
    elif efficiency["status"] == "completed":
        raise Wave3ProbeError(
            "receipt efficiency precedes comparison phase", code="wave3.receipt"
        )
    if marker_ref["status"] == "published":
        if expected_plan is None:
            raise Wave3ProbeError(
                "marker-bound receipt requires a plan", code="wave3.receipt"
            )
        marker_path = Path(marker_ref["path"])
        if not marker_path.is_file():
            raise Wave3ProbeError(
                "published marker is unavailable", code="wave3.receipt"
            )
        marker = validate_marker(_load_json(marker_path), expected_plan=expected_plan)
        if (
            marker["marker_sha256"] != marker_ref["marker_sha256"]
            or runtime["runtime_config_attestation"]
            != marker["runtime_config_attestation"]
            or runtime["model_weight_identity"] != marker["model_weight_identity"]
            or runtime["runtime_source_identities"]
            != marker["runtime_source_identities"]
            or runtime["concrete_trainable_inventory"]
            != marker["concrete_trainable_inventory"]
            or runtime["initial_buffer_inventory"] != marker["initial_buffer_inventory"]
            or runtime["shared_gpu_contract"]["baseline"]
            != marker["shared_gpu_preflight"]
            or runtime["shared_gpu_contract"]["pre_marker_observation"]
            != marker["pre_marker_gpu_observation"]
        ):
            raise Wave3ProbeError(
                "receipt marker binding drifted", code="wave3.receipt"
            )
    if receipt["status"] == "passed":
        if phases != list(PASSED_PHASE_ORDER):
            raise Wave3ProbeError(
                "passed receipt phases are incomplete", code="wave3.receipt"
            )
        if (
            marker_ref["status"] != "published"
            or runtime["status"] != "gpu_ready"
            or runtime["shared_gpu_contract"]["status"] != "post_probe_verified"
            or runtime["shared_gpu_contract"]["post_probe_preterminal_observation"][
                "status"
            ]
            != "passed"
            or runtime["process_cleanup"]["status"] != "passed"
        ):
            raise Wave3ProbeError(
                "passed receipt lacks runtime marker binding", code="wave3.receipt"
            )
        if (
            oracle["status"] != "completed"
            or any(value["status"] != "completed" for value in validated_arms.values())
            or any(
                value is None or not value["passed"] for value in comparisons.values()
            )
            or efficiency["status"] != "completed"
        ):
            raise Wave3ProbeError(
                "passed receipt evidence is incomplete", code="wave3.receipt"
            )
        initial_buffer_sha = runtime["initial_buffer_inventory"]["inventory_sha256"]
        transitions = [
            oracle["buffers"],
            *(validated_arms[name]["artifact"]["buffers"] for name in ARM_ORDER),
        ]
        if any(
            transition[side]["inventory_sha256"] != initial_buffer_sha
            for transition in transitions
            for side in ("before", "restored")
        ):
            raise Wave3ProbeError(
                "arm buffers are not runtime-inventory bound", code="wave3.receipt"
            )
        expected_gradient_rows = [
            {
                "name": row["name"],
                "shape": row["shape"],
                "storage_dtype": row["parameter_storage_dtype"],
            }
            for row in runtime["concrete_trainable_inventory"]["parameters"]
        ]
        for name in ARM_ORDER:
            observed_gradient_rows = [
                {
                    "name": row["name"],
                    "shape": row["shape"],
                    "storage_dtype": row["storage_dtype"],
                }
                for row in validated_arms[name]["artifact"]["gradients"]["rows"]
            ]
            if observed_gradient_rows != expected_gradient_rows:
                raise Wave3ProbeError(
                    "arm gradients are not inventory-bound", code="wave3.receipt"
                )
        if receipt.get("terminal_reason") is not None:
            raise Wave3ProbeError(
                "passed receipt has a terminal error", code="wave3.receipt"
            )
    else:
        reason = _mapping(receipt.get("terminal_reason"), "terminal_reason")
        _require_exact_fields(
            reason,
            {"code", "type", "message"},
            owner="terminal_reason",
            code="wave3.receipt",
        )
        if not all(
            isinstance(reason[field], str) and reason[field] for field in reason
        ):
            raise Wave3ProbeError(
                "failed receipt reason is empty", code="wave3.receipt"
            )
    return receipt


def validate_publication_failure(
    payload: Mapping[str, Any], *, expected_plan: Mapping[str, Any]
) -> dict[str, Any]:
    sidecar = _validate_finalized(
        payload,
        schema=PUBLICATION_FAILURE_SCHEMA,
        hash_field="publication_failure_sha256",
    )
    _require_exact_fields(
        sidecar,
        {
            "schema",
            "status",
            "plan_sha256",
            "receipt_target",
            "publication_failure_target",
            "attempt_marker",
            "source_identity",
            "intended_receipt_sha256",
            "terminal_reason",
            "publication_failure_sha256",
        },
        owner="publication_failure",
        code="wave3.publication_failure",
    )
    plan = validate_plan(expected_plan)
    targets = _mapping(plan["artifact_targets"], "artifact_targets")
    _validate_marker_reference(sidecar["attempt_marker"], plan)
    _validate_source_identity(sidecar["source_identity"])
    reason = _mapping(sidecar["terminal_reason"], "terminal_reason")
    _require_exact_fields(
        reason,
        {"code", "type", "message"},
        owner="terminal_reason",
        code="wave3.publication_failure",
    )
    if (
        sidecar["status"] != "receipt_publication_failed"
        or sidecar["plan_sha256"] != plan["plan_sha256"]
        or sidecar["receipt_target"] != targets["receipt"]
        or sidecar["publication_failure_target"] != targets["publication_failure"]
        or sidecar["source_identity"] != plan["source_identity"]
        or not _is_sha256(sidecar["intended_receipt_sha256"])
        or not all(isinstance(reason[field], str) and reason[field] for field in reason)
    ):
        raise Wave3ProbeError(
            "publication failure sidecar drifted", code="wave3.publication_failure"
        )
    return sidecar


def _validate_source_identity(
    value: Any, *, allow_empty: bool = False
) -> dict[str, str]:
    source = _mapping(value, "source_identity")
    if allow_empty and not source:
        return {}
    if set(source) != set(SOURCE_OWNERS) or not all(
        _is_sha256(item) for item in source.values()
    ):
        raise Wave3ProbeError(
            "source identity is incomplete", code="wave3.source_identity"
        )
    return dict(source)


def _validate_runtime_source_identities(value: Any) -> dict[str, Any]:
    identities = _mapping(value, "runtime_source_identities")
    _require_exact_fields(
        identities,
        {"model", "adapter", "embedding_delta"},
        owner="runtime_source_identities",
        code="wave3.runtime_source_identity",
    )
    result: dict[str, Any] = {}
    for owner in ("model", "adapter", "embedding_delta"):
        row = _mapping(identities[owner], owner)
        _require_exact_fields(
            row,
            {"artifact", "sha256"},
            owner=f"runtime_source_identities.{owner}",
            code="wave3.runtime_source_identity",
        )
        artifact = _mapping(row["artifact"], f"{owner}.artifact")
        if not artifact or not _is_sha256(row["sha256"]):
            raise Wave3ProbeError(
                "authoritative runtime source identity is empty",
                code="wave3.runtime_source_identity",
            )
        try:
            digest = sha256_json(artifact)
        except (TypeError, ValueError) as exc:
            raise Wave3ProbeError(
                "runtime source identity is not canonical JSON",
                code="wave3.runtime_source_identity",
            ) from exc
        if row["sha256"] != digest:
            raise Wave3ProbeError(
                "runtime source identity digest drifted",
                code="wave3.runtime_source_identity",
            )
        if owner == "model":
            if (
                not {
                    "base_model_path",
                    "base_config_sha256",
                    "tokenizer_sha256",
                    "load_model",
                    "model",
                    "tokens",
                    "package_versions",
                }.issubset(artifact)
                or not Path(str(artifact["base_model_path"])).is_absolute()
                or not _is_sha256(artifact["base_config_sha256"])
                or not _is_sha256(artifact["tokenizer_sha256"])
                or artifact["load_model"] is not True
                or not isinstance(artifact["model"], Mapping)
                or not artifact["model"]
                or not isinstance(artifact["tokens"], Mapping)
                or not artifact["tokens"]
                or not isinstance(artifact["package_versions"], Mapping)
                or not artifact["package_versions"]
            ):
                raise Wave3ProbeError(
                    "model source identity is not authoritative",
                    code="wave3.runtime_source_identity",
                )
        elif owner == "adapter":
            if set(artifact) != {"plan", "setup_receipt"}:
                raise Wave3ProbeError(
                    "adapter source identity fields drifted",
                    code="wave3.runtime_source_identity",
                )
            adapter_plan = _mapping(artifact["plan"], "adapter.plan")
            setup = _mapping(artifact["setup_receipt"], "adapter.setup_receipt")
            if (
                adapter_plan.get("adapter_type") != "dora"
                or not isinstance(adapter_plan.get("source_gate"), Mapping)
                or not adapter_plan.get("source_gate")
                or setup.get("adapter_type") != "dora"
                or setup.get("mode") != adapter_plan.get("mode")
                or not isinstance(setup.get("trainable_names"), list)
                or not setup.get("trainable_names")
            ):
                raise Wave3ProbeError(
                    "adapter source identity is not authoritative",
                    code="wave3.runtime_source_identity",
                )
        else:
            if set(artifact) != {"install_receipt", "load_receipt"}:
                raise Wave3ProbeError(
                    "embedding-delta source identity fields drifted",
                    code="wave3.runtime_source_identity",
                )
            install = _mapping(
                artifact["install_receipt"], "embedding_delta.install_receipt"
            )
            if (
                install.get("semantics") != "additive_delta"
                or install.get("tensor_key") != "shared_embed_delta"
                or not isinstance(install.get("delta_parameter_names"), list)
                or not install.get("delta_parameter_names")
                or (
                    artifact["load_receipt"] is not None
                    and not isinstance(artifact["load_receipt"], Mapping)
                )
            ):
                raise Wave3ProbeError(
                    "embedding-delta source identity is not authoritative",
                    code="wave3.runtime_source_identity",
                )
        result[owner] = dict(row)
    return result


def _validate_cpu_installation(value: Any) -> dict[str, Any]:
    installation = _mapping(value, "cpu_installation")
    _require_exact_fields(
        installation,
        {"model", "adapter", "embedding_delta"},
        owner="cpu_installation",
        code="wave3.cpu_installation",
    )
    for owner in ("model", "adapter", "embedding_delta"):
        row = _mapping(installation[owner], f"cpu_installation.{owner}")
        _require_exact_fields(
            row,
            {"parameter_count", "buffer_count", "all_cpu"},
            owner=f"cpu_installation.{owner}",
            code="wave3.cpu_installation",
        )
        if (
            not _is_nonnegative_int(row["parameter_count"])
            or not _is_nonnegative_int(row["buffer_count"])
            or row["all_cpu"] is not True
        ):
            raise Wave3ProbeError(
                "CPU installation evidence drifted", code="wave3.cpu_installation"
            )
    return dict(installation)


def _validate_concrete_inventory(value: Any) -> dict[str, Any]:
    inventory = _mapping(value, "concrete_trainable_inventory")
    try:
        validated = validate_concrete_trainable_inventory(inventory)
    except Exception as exc:
        raise Wave3ProbeError(
            "concrete trainable inventory is invalid", code="wave3.inventory"
        ) from exc
    if (
        validated["total_count"] != EXPECTED_TRAINABLE_COUNT
        or validated["group_counts"] != EXPECTED_TRAINABLE_STRUCTURE
    ):
        raise Wave3ProbeError(
            "concrete trainable inventory is not exact", code="wave3.inventory"
        )
    return validated


def _validate_compute_process_rows(value: Any, *, owner: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise Wave3ProbeError(
            "GPU compute process inventory is malformed",
            code="wave3.gpu_baseline",
        )
    rows: list[dict[str, Any]] = []
    for row_value in value:
        row = _mapping(row_value, owner)
        _require_exact_fields(
            row,
            {"gpu_uuid", "driver_pid"},
            owner=owner,
            code="wave3.gpu_baseline",
        )
        if (
            not isinstance(row["gpu_uuid"], str)
            or not row["gpu_uuid"]
            or not _is_nonnegative_int(row["driver_pid"])
            or row["driver_pid"] == 0
        ):
            raise Wave3ProbeError(
                "GPU compute process inventory is malformed",
                code="wave3.gpu_baseline",
            )
        rows.append(dict(row))
    ordered = sorted(rows, key=lambda row: (row["gpu_uuid"], row["driver_pid"]))
    if rows != ordered or len(
        {(row["gpu_uuid"], row["driver_pid"]) for row in rows}
    ) != len(rows):
        raise Wave3ProbeError(
            "GPU compute process inventory is not canonical",
            code="wave3.gpu_baseline",
        )
    return rows


def _validate_shared_gpu_baseline(value: Any) -> dict[str, Any]:
    receipt = _mapping(value, "shared_gpu_preflight")
    _require_exact_fields(
        receipt,
        {
            "schema",
            "mode",
            "requested_device",
            "cuda_visible_devices",
            "selector",
            "gpu_identity",
            "limits",
            "utilization_disposition",
            "baseline_compute_processes",
            "samples",
        },
        owner="shared_gpu_preflight",
        code="wave3.gpu_preflight",
    )
    if (
        receipt["schema"] != SHARED_GPU_BASELINE_SCHEMA
        or receipt["mode"] != "shared_preexisting_compute"
        or receipt["requested_device"]
        != _parse_cuda_device_text(str(receipt["requested_device"]))
        or receipt["utilization_disposition"]
        != "observational_only_not_admission_or_promotion"
        or not isinstance(receipt["selector"], str)
        or not receipt["selector"]
        or (
            receipt["cuda_visible_devices"] is not None
            and not isinstance(receipt["cuda_visible_devices"], str)
        )
    ):
        raise Wave3ProbeError(
            "shared GPU preflight identity drifted", code="wave3.gpu_preflight"
        )
    identity = _mapping(receipt["gpu_identity"], "shared_gpu_preflight.gpu_identity")
    _require_exact_fields(
        identity,
        {"physical_index", "uuid", "total_memory_mib"},
        owner="shared_gpu_preflight.gpu_identity",
        code="wave3.gpu_preflight",
    )
    if (
        not _is_nonnegative_int(identity["physical_index"])
        or not isinstance(identity["uuid"], str)
        or not identity["uuid"]
        or identity["total_memory_mib"] != SHARED_GPU_TOTAL_MEMORY_MIB
    ):
        raise Wave3ProbeError(
            "shared GPU physical identity drifted", code="wave3.gpu_preflight"
        )
    limits = _mapping(receipt["limits"], "shared_gpu_preflight.limits")
    expected_limits = {
        "sample_count": SHARED_GPU_SAMPLE_COUNT,
        "minimum_sample_interval_seconds": SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS,
        "max_preexisting_memory_mib": SHARED_GPU_MAX_PREEXISTING_MEMORY_MIB,
        "required_headroom_mib": SHARED_GPU_REQUIRED_HEADROOM_MIB,
    }
    if dict(limits) != expected_limits:
        raise Wave3ProbeError(
            "shared GPU preflight limits drifted", code="wave3.gpu_preflight"
        )
    baseline_processes = _validate_compute_process_rows(
        receipt["baseline_compute_processes"],
        owner="shared_gpu_preflight.baseline_compute_process",
    )
    samples = receipt["samples"]
    if not isinstance(samples, list) or len(samples) != SHARED_GPU_SAMPLE_COUNT:
        raise Wave3ProbeError(
            "GPU preflight receipt drifted", code="wave3.gpu_preflight"
        )
    prior_monotonic_ns: int | None = None
    for index, sample_value in enumerate(samples):
        sample = _mapping(sample_value, "shared_gpu_sample")
        _require_exact_fields(
            sample,
            {
                "sample_index",
                "monotonic_ns",
                "memory_used_mib",
                "headroom_mib",
                "utilization_percent",
                "compute_processes",
            },
            owner="shared_gpu_sample",
            code="wave3.gpu_preflight",
        )
        memory_used_mib = sample["memory_used_mib"]
        monotonic_ns = sample["monotonic_ns"]
        if (
            sample["sample_index"] != index
            or not _is_nonnegative_int(monotonic_ns)
            or not _is_nonnegative_int(memory_used_mib)
            or memory_used_mib > SHARED_GPU_MAX_PREEXISTING_MEMORY_MIB
            or sample["headroom_mib"] != SHARED_GPU_TOTAL_MEMORY_MIB - memory_used_mib
            or sample["headroom_mib"] < SHARED_GPU_REQUIRED_HEADROOM_MIB
            or not _is_nonnegative_int(sample["utilization_percent"])
            or sample["utilization_percent"] > 100
            or _validate_compute_process_rows(
                sample["compute_processes"], owner="shared_gpu_sample.compute_process"
            )
            != baseline_processes
        ):
            raise Wave3ProbeError(
                "GPU preflight sample drifted", code="wave3.gpu_preflight"
            )
        if prior_monotonic_ns is not None and monotonic_ns - prior_monotonic_ns < int(
            SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS * 1_000_000_000
        ):
            raise Wave3ProbeError(
                "shared GPU samples are too close", code="wave3.gpu_preflight"
            )
        prior_monotonic_ns = monotonic_ns
    return dict(receipt)


def _validate_shared_gpu_observation(
    value: Any,
    *,
    baseline: Mapping[str, Any],
    expected_stage: str,
    allow_new: bool = False,
) -> dict[str, Any]:
    validated_baseline = _validate_shared_gpu_baseline(baseline)
    observation = _mapping(value, "shared_gpu_observation")
    _require_exact_fields(
        observation,
        {
            "schema",
            "stage",
            "monotonic_ns",
            "gpu_identity",
            "memory_used_mib",
            "headroom_mib",
            "utilization_percent",
            "compute_processes",
            "missing_baseline_processes",
            "new_processes",
        },
        owner="shared_gpu_observation",
        code="wave3.gpu_postcheck",
    )
    current = _validate_compute_process_rows(
        observation["compute_processes"], owner="shared_gpu_observation.compute_process"
    )
    missing = _validate_compute_process_rows(
        observation["missing_baseline_processes"],
        owner="shared_gpu_observation.missing_process",
    )
    new = _validate_compute_process_rows(
        observation["new_processes"], owner="shared_gpu_observation.new_process"
    )
    baseline_rows = validated_baseline["baseline_compute_processes"]
    baseline_keys = {(row["gpu_uuid"], row["driver_pid"]) for row in baseline_rows}
    current_keys = {(row["gpu_uuid"], row["driver_pid"]) for row in current}
    expected_missing = [
        row
        for row in baseline_rows
        if (row["gpu_uuid"], row["driver_pid"]) not in current_keys
    ]
    expected_new = [
        row
        for row in current
        if (row["gpu_uuid"], row["driver_pid"]) not in baseline_keys
    ]
    identity = validated_baseline["gpu_identity"]
    memory_used_mib = observation["memory_used_mib"]
    if (
        observation["schema"] != SHARED_GPU_OBSERVATION_SCHEMA
        or observation["stage"] != expected_stage
        or not _is_nonnegative_int(observation["monotonic_ns"])
        or observation["gpu_identity"] != identity
        or not _is_nonnegative_int(memory_used_mib)
        or memory_used_mib > identity["total_memory_mib"]
        or observation["headroom_mib"] != identity["total_memory_mib"] - memory_used_mib
        or not _is_nonnegative_int(observation["utilization_percent"])
        or observation["utilization_percent"] > 100
        or missing != expected_missing
        or new != expected_new
        or (new and not allow_new)
    ):
        raise Wave3ProbeError(
            "shared GPU postcheck is not a baseline subset",
            code="wave3.gpu_new_process" if new else "wave3.gpu_postcheck",
        )
    return dict(observation)


def _validate_gpu_process_subset_sweep(
    value: Any, *, baseline: Mapping[str, Any], expected_stage: str
) -> dict[str, Any]:
    validated_baseline = _validate_shared_gpu_baseline(baseline)
    sweep = _mapping(value, "shared_gpu_subset_sweep")
    _require_exact_fields(
        sweep,
        {
            "schema",
            "stage",
            "sample_count",
            "minimum_sample_interval_seconds",
            "status",
            "samples",
            "terminal_reason",
        },
        owner="shared_gpu_subset_sweep",
        code="wave3.gpu_postcheck",
    )
    samples = sweep["samples"]
    if (
        sweep["schema"] != SHARED_GPU_SUBSET_SWEEP_SCHEMA
        or sweep["stage"] != expected_stage
        or sweep["sample_count"] != SHARED_GPU_SAMPLE_COUNT
        or sweep["minimum_sample_interval_seconds"]
        != SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS
        or not isinstance(samples, list)
        or len(samples) != SHARED_GPU_SAMPLE_COUNT
    ):
        raise Wave3ProbeError(
            "shared GPU subset sweep shape drifted", code="wave3.gpu_postcheck"
        )
    previous_ns: int | None = None
    failures: list[dict[str, str]] = []
    for index, sample_value in enumerate(samples):
        sample = _mapping(sample_value, "shared_gpu_subset_sweep.sample")
        _require_exact_fields(
            sample,
            {"sample_index", "monotonic_ns", "status", "observation", "error"},
            owner="shared_gpu_subset_sweep.sample",
            code="wave3.gpu_postcheck",
        )
        if (
            sample["sample_index"] != index
            or not _is_nonnegative_int(sample["monotonic_ns"])
            or sample["status"] not in {"passed", "failed"}
        ):
            raise Wave3ProbeError(
                "shared GPU subset sample drifted", code="wave3.gpu_postcheck"
            )
        observed_ns = sample["monotonic_ns"]
        if previous_ns is not None and observed_ns - previous_ns < int(
            SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS * 1_000_000_000
        ):
            raise Wave3ProbeError(
                "shared GPU subset sweeps are too close",
                code="wave3.gpu_postcheck",
            )
        previous_ns = observed_ns
        error = sample["error"]
        observation = sample["observation"]
        if sample["status"] == "passed":
            if error is not None or observation is None:
                raise Wave3ProbeError(
                    "passed GPU subset sample lacks evidence",
                    code="wave3.gpu_postcheck",
                )
            _validate_shared_gpu_observation(
                observation,
                baseline=validated_baseline,
                expected_stage=expected_stage,
            )
        else:
            error_row = _mapping(error, "shared_gpu_subset_sweep.sample.error")
            _require_exact_fields(
                error_row,
                {"code", "type", "message"},
                owner="shared_gpu_subset_sweep.sample.error",
                code="wave3.gpu_postcheck",
            )
            if not all(
                isinstance(error_row[field], str) and error_row[field]
                for field in ("code", "type", "message")
            ):
                raise Wave3ProbeError(
                    "failed GPU subset sample has empty error",
                    code="wave3.gpu_postcheck",
                )
            if observation is not None:
                _validate_shared_gpu_observation(
                    observation,
                    baseline=validated_baseline,
                    expected_stage=expected_stage,
                    allow_new=True,
                )
            failures.append(dict(error_row))
    expected_status = "failed" if failures else "passed"
    if sweep["status"] != expected_status:
        raise Wave3ProbeError(
            "shared GPU subset sweep status contradicts samples",
            code="wave3.gpu_postcheck",
        )
    terminal_reason = sweep["terminal_reason"]
    if failures:
        if terminal_reason != failures[0]:
            raise Wave3ProbeError(
                "shared GPU subset sweep terminal reason drifted",
                code="wave3.gpu_postcheck",
            )
    elif terminal_reason is not None:
        raise Wave3ProbeError(
            "passed GPU subset sweep has an error", code="wave3.gpu_postcheck"
        )
    return dict(sweep)


def _validate_marker_reference(
    value: Any, expected_plan: Mapping[str, Any] | None
) -> dict[str, Any]:
    reference = _mapping(value, "attempt_marker")
    _require_exact_fields(
        reference,
        {"status", "path", "marker_sha256"},
        owner="attempt_marker",
        code="wave3.receipt",
    )
    status = reference["status"]
    if status not in {"not_published", "published"}:
        raise Wave3ProbeError(
            "marker reference status is invalid", code="wave3.receipt"
        )
    path = reference["path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise Wave3ProbeError("marker reference path is invalid", code="wave3.receipt")
    if expected_plan is not None:
        plan = validate_plan(expected_plan)
        if path != plan["artifact_targets"]["attempt_marker"]:
            raise Wave3ProbeError(
                "marker reference target drifted", code="wave3.receipt"
            )
    if (status == "published") != _is_sha256(reference["marker_sha256"]):
        raise Wave3ProbeError(
            "marker reference digest is invalid", code="wave3.receipt"
        )
    if status == "not_published" and reference["marker_sha256"] is not None:
        raise Wave3ProbeError("unpublished marker has a digest", code="wave3.receipt")
    return dict(reference)


def _validate_buffer_inventory(value: Any) -> dict[str, Any]:
    inventory = _mapping(value, "buffer_inventory")
    _require_exact_fields(
        inventory,
        {"count", "rows", "inventory_sha256"},
        owner="buffer_inventory",
        code="wave3.buffers",
    )
    rows = inventory["rows"]
    if not isinstance(rows, list):
        raise Wave3ProbeError("buffer rows are invalid", code="wave3.buffers")
    names: list[str] = []
    for row_value in rows:
        row = _mapping(row_value, "buffer_row")
        _require_exact_fields(
            row,
            {"name", "shape", "dtype", "numel", "finite", "sha256"},
            owner="buffer_row",
            code="wave3.buffers",
        )
        shape = row["shape"]
        if (
            not isinstance(row["name"], str)
            or not row["name"]
            or not isinstance(shape, list)
            or not all(_is_nonnegative_int(item) for item in shape)
            or row["numel"] != math.prod(shape)
            or not isinstance(row["dtype"], str)
            or not row["dtype"]
            or row["finite"] is not True
            or not _is_sha256(row["sha256"])
        ):
            raise Wave3ProbeError("buffer row is malformed", code="wave3.buffers")
        names.append(row["name"])
    if names != sorted(set(names)) or inventory["count"] != len(rows):
        raise Wave3ProbeError(
            "buffer inventory names/count drifted", code="wave3.buffers"
        )
    body = {"count": inventory["count"], "rows": rows}
    if inventory["inventory_sha256"] != sha256_json(body):
        raise Wave3ProbeError("buffer inventory digest drifted", code="wave3.buffers")
    return dict(inventory)


def _validate_buffer_transition(value: Any) -> dict[str, Any]:
    transition = _mapping(value, "buffer_transition")
    _require_exact_fields(
        transition,
        {
            "before",
            "after",
            "restored",
            "after_matches_before",
            "restoration_matches_initial",
        },
        owner="buffer_transition",
        code="wave3.buffers",
    )
    before = _validate_buffer_inventory(transition["before"])
    after = _validate_buffer_inventory(transition["after"])
    _validate_buffer_inventory(transition["restored"])
    if (
        transition["after_matches_before"]
        is not (after["inventory_sha256"] == before["inventory_sha256"])
        or transition["restoration_matches_initial"] is not True
    ):
        raise Wave3ProbeError(
            "buffer restoration relation drifted", code="wave3.buffers"
        )
    return dict(transition)


def _validate_scalar_comparison(
    value: Any, *, expected_rtol: float, expected_atol: float
) -> dict[str, Any]:
    row = _mapping(value, "scalar_comparison")
    _require_exact_fields(
        row,
        {
            "left",
            "right",
            "rtol",
            "atol",
            "max_abs_diff",
            "threshold",
            "finite",
            "passed",
        },
        owner="scalar_comparison",
        code="wave3.comparison",
    )
    if row["rtol"] != expected_rtol or row["atol"] != expected_atol:
        raise Wave3ProbeError("comparison tolerance drifted", code="wave3.comparison")
    if row["finite"] is False:
        if (
            row["passed"] is not False
            or row["max_abs_diff"] is not None
            or row["threshold"] is not None
            or not all(
                item is None or _is_finite_number(item)
                for item in (row["left"], row["right"])
            )
        ):
            raise Wave3ProbeError(
                "non-finite comparison relation drifted", code="wave3.comparison"
            )
        return dict(row)
    values = (row["left"], row["right"], row["max_abs_diff"], row["threshold"])
    if not all(_is_finite_number(item) for item in values) or row["finite"] is not True:
        raise Wave3ProbeError(
            "comparison values are non-finite", code="wave3.comparison"
        )
    difference = abs(float(row["left"]) - float(row["right"]))
    threshold = expected_atol + expected_rtol * abs(float(row["right"]))
    passed = difference <= threshold
    if (
        not _float_exact(row["max_abs_diff"], difference)
        or not _float_exact(row["threshold"], threshold)
        or row["passed"] is not passed
    ):
        raise Wave3ProbeError("comparison relation drifted", code="wave3.comparison")
    return dict(row)


def _validate_gradient_comparison(value: Any) -> dict[str, Any]:
    result = _mapping(value, "gradient_comparison")
    _require_exact_fields(
        result,
        {
            "rtol",
            "atol",
            "left_count",
            "right_count",
            "left_names_sha256",
            "right_names_sha256",
            "missing",
            "extra",
            "mismatched",
            "rows",
            "global_max_abs_diff",
            "passed",
        },
        owner="gradient_comparison",
        code="wave3.comparison",
    )
    if result["rtol"] != BF16_RTOL or result["atol"] != BF16_ATOL:
        raise Wave3ProbeError("gradient tolerance drifted", code="wave3.comparison")
    rows = result["rows"]
    if not isinstance(rows, list) or not rows:
        raise Wave3ProbeError(
            "gradient comparison rows are empty", code="wave3.comparison"
        )
    left_names: list[str] = []
    right_names: list[str] = []
    mismatched: list[str] = []
    maxima: list[float] = []
    for row_value in rows:
        row = _mapping(row_value, "gradient_comparison_row")
        _require_exact_fields(
            row,
            {
                "name",
                "left_present",
                "right_present",
                "left_shape",
                "right_shape",
                "left_dtype",
                "right_dtype",
                "left_finite",
                "right_finite",
                "element_count",
                "within_count",
                "max_abs_diff",
                "max_threshold_excess",
                "passed",
            },
            owner="gradient_comparison_row",
            code="wave3.comparison",
        )
        name = row["name"]
        if not isinstance(name, str) or not name:
            raise Wave3ProbeError(
                "gradient comparison name is empty", code="wave3.comparison"
            )
        if row["left_present"]:
            left_names.append(name)
        if row["right_present"]:
            right_names.append(name)
        common = row["left_present"] is True and row["right_present"] is True
        row_passed = bool(
            common
            and row["left_shape"] == row["right_shape"]
            and row["left_dtype"] == "torch.float32"
            and row["right_dtype"] == "torch.float32"
            and row["left_finite"] is True
            and row["right_finite"] is True
            and _is_nonnegative_int(row["element_count"])
            and row["within_count"] == row["element_count"]
            and _is_finite_number(row["max_abs_diff"])
            and _is_finite_number(row["max_threshold_excess"])
            and float(row["max_threshold_excess"]) <= 0.0
        )
        if row["passed"] is not row_passed:
            raise Wave3ProbeError(
                "gradient row relation drifted", code="wave3.comparison"
            )
        if not row_passed:
            mismatched.append(name)
        if common and _is_finite_number(row["max_abs_diff"]):
            maxima.append(float(row["max_abs_diff"]))
    left_names = sorted(left_names)
    right_names = sorted(right_names)
    missing = sorted(set(left_names) - set(right_names))
    extra = sorted(set(right_names) - set(left_names))
    if (
        result["left_count"] != len(left_names)
        or result["right_count"] != len(right_names)
        or result["left_names_sha256"] != sha256_json(left_names)
        or result["right_names_sha256"] != sha256_json(right_names)
        or result["missing"] != missing
        or result["extra"] != extra
        or result["mismatched"] != sorted(mismatched)
        or not _float_exact(result["global_max_abs_diff"], max(maxima, default=0.0))
        or result["passed"] is not (not missing and not extra and not mismatched)
    ):
        raise Wave3ProbeError(
            "gradient comparison summary drifted", code="wave3.comparison"
        )
    return dict(result)


def _validate_oracle_receipt(value: Any) -> dict[str, Any]:
    oracle = _mapping(value, "oracle")
    _require_exact_fields(
        oracle,
        {
            "status",
            "model_forward_count",
            "backward_count",
            "same_logits_object",
            "logits_dtype",
            "logits_sha256",
            "reference_requires_grad",
            "optimized_requires_grad",
            "optimized_grad_fn",
            "raw_diagnostic_comparison",
            "buffers",
        },
        owner="oracle",
        code="wave3.receipt",
    )
    if oracle["status"] == "not_run":
        expected = _empty_oracle()
        if dict(oracle) != expected:
            raise Wave3ProbeError("not-run oracle is not exact", code="wave3.receipt")
        return dict(oracle)
    if oracle["status"] != "completed":
        raise Wave3ProbeError("oracle status is invalid", code="wave3.receipt")
    if (
        oracle["model_forward_count"] != 1
        or oracle["backward_count"] != 0
        or oracle["same_logits_object"] is not True
        or oracle["logits_dtype"] != "torch.float32"
        or not _is_sha256(oracle["logits_sha256"])
        or oracle["reference_requires_grad"] is not True
        or oracle["optimized_requires_grad"] is not False
        or oracle["optimized_grad_fn"] is not None
    ):
        raise Wave3ProbeError("oracle evidence drifted", code="wave3.receipt")
    _validate_scalar_comparison(
        oracle["raw_diagnostic_comparison"],
        expected_rtol=LOSS_RTOL,
        expected_atol=LOSS_ATOL,
    )
    _validate_buffer_transition(oracle["buffers"])
    return dict(oracle)


def _validate_arm_envelope(name: str, value: Any) -> dict[str, Any]:
    envelope = _mapping(value, f"arm.{name}")
    _require_exact_fields(
        envelope,
        {"status", "artifact"},
        owner=f"arm.{name}",
        code="wave3.receipt",
    )
    if envelope["status"] == "not_run":
        if envelope["artifact"] is not None:
            raise Wave3ProbeError("not-run arm has evidence", code="wave3.receipt")
        return dict(envelope)
    if envelope["status"] != "completed":
        raise Wave3ProbeError("arm status is invalid", code="wave3.receipt")
    artifact = _mapping(envelope["artifact"], f"arm.{name}.artifact")
    _require_exact_fields(
        artifact,
        {
            "name",
            "mode",
            "implementation",
            "model_forward_count",
            "backward_count",
            "wall_seconds",
            "total_loss",
            "base_only_total_loss",
            "diagnostic_requires_grad",
            "diagnostic_grad_fn",
            "nonzero_control_differentiable",
            "graph_residue",
            "resources_before",
            "resources_after",
            "gradients",
            "buffers",
        },
        owner=f"arm.{name}.artifact",
        code="wave3.receipt",
    )
    expected_mode = "zero" if name.startswith("zero_") else "nonzero"
    expected_implementation = "reference" if name.endswith("reference") else "optimized"
    if (
        artifact["name"] != name
        or artifact["mode"] != expected_mode
        or artifact["implementation"] != expected_implementation
        or artifact["model_forward_count"] != 1
        or artifact["backward_count"] != 1
        or not _is_finite_number(artifact["wall_seconds"])
        or not 0.0 <= float(artifact["wall_seconds"]) <= ARM_WALL_CEILING_SECONDS
        or not _is_finite_number(artifact["total_loss"])
        or not _is_finite_number(artifact["base_only_total_loss"])
    ):
        raise Wave3ProbeError("arm scalar evidence drifted", code="wave3.receipt")
    expected_grad = not (
        expected_mode == "zero" and expected_implementation == "optimized"
    )
    if artifact["diagnostic_requires_grad"] is not expected_grad:
        raise Wave3ProbeError("arm graph evidence drifted", code="wave3.receipt")
    if expected_grad != (
        isinstance(artifact["diagnostic_grad_fn"], str)
        and bool(artifact["diagnostic_grad_fn"])
    ):
        raise Wave3ProbeError("arm grad_fn evidence drifted", code="wave3.receipt")
    if artifact["nonzero_control_differentiable"] is not (expected_mode == "nonzero"):
        raise Wave3ProbeError("arm nonzero control drifted", code="wave3.receipt")
    _validate_graph_residue(artifact["graph_residue"], optimized_zero=not expected_grad)
    _validate_resource_snapshot(artifact["resources_before"])
    _validate_resource_snapshot(artifact["resources_after"])
    _validate_gradient_snapshot(artifact["gradients"])
    _validate_buffer_transition(artifact["buffers"])
    return dict(envelope)


def _validate_graph_residue(value: Any, *, optimized_zero: bool) -> None:
    graph = _mapping(value, "graph_residue")
    _require_exact_fields(
        graph,
        {
            "saved_graph_tensor_count",
            "live_saved_graph_tensors_after_backward_and_release",
            "diagnostic_tensor_live_after_release",
            "optimized_zero_diagnostic_released",
        },
        owner="graph_residue",
        code="wave3.receipt",
    )
    if (
        not _is_nonnegative_int(graph["saved_graph_tensor_count"])
        or not _is_nonnegative_int(
            graph["live_saved_graph_tensors_after_backward_and_release"]
        )
        or graph["live_saved_graph_tensors_after_backward_and_release"]
        > graph["saved_graph_tensor_count"]
        or not isinstance(graph["diagnostic_tensor_live_after_release"], bool)
        or graph["optimized_zero_diagnostic_released"]
        is not (optimized_zero and not graph["diagnostic_tensor_live_after_release"])
    ):
        raise Wave3ProbeError("graph residue relation drifted", code="wave3.receipt")


def _validate_resource_snapshot(value: Any) -> dict[str, Any]:
    snapshot = _mapping(value, "resource_snapshot")
    _require_exact_fields(
        snapshot,
        {"schema_version", "cpu", "gpu"},
        owner="resource_snapshot",
        code="wave3.resources",
    )
    cpu = _mapping(snapshot["cpu"], "resource.cpu")
    gpu = _mapping(snapshot["gpu"], "resource.gpu")
    _require_exact_fields(
        cpu,
        {"scope", "max_rss_bytes", "io_read_bytes", "io_write_bytes"},
        owner="resource.cpu",
        code="wave3.resources",
    )
    _require_exact_fields(
        gpu,
        {
            "scope",
            "initialized",
            "device_index",
            "max_memory_allocated_bytes",
            "max_memory_reserved_bytes",
        },
        owner="resource.gpu",
        code="wave3.resources",
    )
    if (
        snapshot["schema_version"] != 1
        or cpu["scope"] != "current_process"
        or not _is_nonnegative_int(cpu["max_rss_bytes"])
        or cpu["max_rss_bytes"] > HOST_RSS_CEILING_BYTES
        or gpu["scope"] != "current_process_current_device"
        or gpu["initialized"] is not True
        or not _is_nonnegative_int(gpu["device_index"])
        or not _is_nonnegative_int(gpu["max_memory_allocated_bytes"])
        or not _is_nonnegative_int(gpu["max_memory_reserved_bytes"])
        or gpu["max_memory_allocated_bytes"] > DEVICE_MEMORY_CEILING_BYTES
        or gpu["max_memory_reserved_bytes"] > DEVICE_MEMORY_CEILING_BYTES
    ):
        raise Wave3ProbeError(
            "resource snapshot exceeds contract", code="wave3.resources"
        )
    return dict(snapshot)


def _validate_gradient_snapshot(value: Any) -> dict[str, Any]:
    snapshot = _mapping(value, "gradient_snapshot")
    _require_exact_fields(
        snapshot,
        {"count", "rows", "rows_sha256", "aggregate_nonzero"},
        owner="gradient_snapshot",
        code="wave3.gradient",
    )
    rows = snapshot["rows"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_TRAINABLE_COUNT:
        raise Wave3ProbeError("gradient rows are incomplete", code="wave3.gradient")
    names: list[str] = []
    for row_value in rows:
        row = _mapping(row_value, "gradient_row")
        _require_exact_fields(
            row,
            {
                "name",
                "shape",
                "storage_dtype",
                "gradient_dtype",
                "comparison_dtype",
                "numel",
                "finite",
                "sha256",
                "l2_norm",
            },
            owner="gradient_row",
            code="wave3.gradient",
        )
        shape = row["shape"]
        if (
            not isinstance(row["name"], str)
            or not row["name"]
            or not isinstance(shape, list)
            or not shape
            or not all(_is_nonnegative_int(item) for item in shape)
            or row["numel"] != math.prod(shape)
            or not isinstance(row["storage_dtype"], str)
            or row["gradient_dtype"] != row["storage_dtype"]
            or row["comparison_dtype"] != "torch.float32"
            or row["finite"] is not True
            or not _is_sha256(row["sha256"])
            or not _is_finite_number(row["l2_norm"])
            or float(row["l2_norm"]) < 0.0
        ):
            raise Wave3ProbeError("gradient row is invalid", code="wave3.gradient")
        names.append(row["name"])
    if (
        names != sorted(set(names))
        or snapshot["count"] != len(rows)
        or snapshot["rows_sha256"] != sha256_json(rows)
        or snapshot["aggregate_nonzero"] is not True
    ):
        raise Wave3ProbeError(
            "gradient snapshot digest/count drifted", code="wave3.gradient"
        )
    return dict(snapshot)


def _validate_process_identity_rows(value: Any, *, owner: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise Wave3ProbeError(
            "process cleanup rows are malformed", code="wave3.process_cleanup"
        )
    rows: list[dict[str, Any]] = []
    for row_value in value:
        row = _mapping(row_value, owner)
        _require_exact_fields(
            row,
            {
                "pid",
                "state",
                "parent_pid",
                "process_group_id",
                "session_id",
                "start_time_ticks",
            },
            owner=owner,
            code="wave3.process_cleanup",
        )
        if (
            not _is_nonnegative_int(row["pid"])
            or row["pid"] == 0
            or not isinstance(row["state"], str)
            or len(row["state"]) != 1
            or not all(
                _is_nonnegative_int(row[field])
                for field in (
                    "parent_pid",
                    "process_group_id",
                    "session_id",
                    "start_time_ticks",
                )
            )
        ):
            raise Wave3ProbeError(
                "process cleanup row is invalid", code="wave3.process_cleanup"
            )
        rows.append(dict(row))
    if len({_process_identity_key(row) for row in rows}) != len(rows):
        raise Wave3ProbeError(
            "process cleanup rows contain duplicates", code="wave3.process_cleanup"
        )
    return rows


def _empty_process_cleanup() -> dict[str, Any]:
    return {
        "schema": PROCESS_CLEANUP_SCHEMA,
        "status": "not_applicable",
        "reason": None,
        "leader_pid": None,
        "cleanup_timeout_seconds": PROCESS_CLEANUP_TIMEOUT_SECONDS,
        "tracked_processes": [],
        "term_signals": [],
        "kill_signals": [],
        "leader_absent": None,
        "descendants_absent": None,
        "sessions_absent": None,
        "surviving_processes": [],
        "completed_monotonic_ns": None,
        "terminal_reason": None,
    }


def _validate_process_cleanup_receipt(value: Any) -> dict[str, Any]:
    cleanup = _mapping(value, "runtime.process_cleanup")
    _require_exact_fields(
        cleanup,
        set(_empty_process_cleanup()),
        owner="runtime.process_cleanup",
        code="wave3.process_cleanup",
    )
    if (
        cleanup["schema"] != PROCESS_CLEANUP_SCHEMA
        or cleanup["cleanup_timeout_seconds"] != PROCESS_CLEANUP_TIMEOUT_SECONDS
    ):
        raise Wave3ProbeError(
            "process cleanup contract drifted", code="wave3.process_cleanup"
        )
    tracked = _validate_process_identity_rows(
        cleanup["tracked_processes"], owner="runtime.process_cleanup.tracked"
    )
    term = _validate_process_identity_rows(
        cleanup["term_signals"], owner="runtime.process_cleanup.term"
    )
    killed = _validate_process_identity_rows(
        cleanup["kill_signals"], owner="runtime.process_cleanup.kill"
    )
    survivors = _validate_process_identity_rows(
        cleanup["surviving_processes"], owner="runtime.process_cleanup.survivor"
    )
    tracked_keys = {_process_identity_key(row) for row in tracked}
    if any(
        _process_identity_key(row) not in tracked_keys
        for row in (*term, *killed, *survivors)
    ):
        raise Wave3ProbeError(
            "process cleanup evidence is not tracked-tree bound",
            code="wave3.process_cleanup",
        )
    status = cleanup["status"]
    if status == "not_applicable":
        if dict(cleanup) != _empty_process_cleanup():
            raise Wave3ProbeError(
                "not-applicable cleanup has evidence", code="wave3.process_cleanup"
            )
    elif status == "passed":
        if (
            not _is_nonnegative_int(cleanup["leader_pid"])
            or not isinstance(cleanup["reason"], str)
            or not cleanup["reason"]
            or cleanup["leader_absent"] is not True
            or cleanup["descendants_absent"] is not True
            or cleanup["sessions_absent"] is not True
            or survivors
            or not _is_nonnegative_int(cleanup["completed_monotonic_ns"])
            or cleanup["terminal_reason"] is not None
        ):
            raise Wave3ProbeError(
                "passed process cleanup lacks absence proof",
                code="wave3.process_cleanup",
            )
    elif status == "failed":
        reason = _mapping(cleanup["terminal_reason"], "process_cleanup.error")
        _require_exact_fields(
            reason,
            {"code", "type", "message"},
            owner="process_cleanup.error",
            code="wave3.process_cleanup",
        )
        if not all(
            isinstance(reason[field], str) and reason[field]
            for field in ("code", "type", "message")
        ):
            raise Wave3ProbeError(
                "failed process cleanup has empty error",
                code="wave3.process_cleanup",
            )
    else:
        raise Wave3ProbeError(
            "process cleanup status is invalid", code="wave3.process_cleanup"
        )
    return dict(cleanup)


def _validate_shared_gpu_runtime_contract(value: Any) -> dict[str, Any]:
    contract = _mapping(value, "runtime.shared_gpu_contract")
    _require_exact_fields(
        contract,
        {
            "status",
            "claim_boundary",
            "baseline_process_action",
            "baseline",
            "pre_marker_observation",
            "post_probe_preterminal_observation",
        },
        owner="runtime.shared_gpu_contract",
        code="wave3.gpu_contract",
    )
    if (
        contract["claim_boundary"] != shared_gpu_claim_boundary()
        or contract["baseline_process_action"] != "observe_only_never_signal_or_reset"
        or contract["status"]
        not in {
            "not_started",
            "baseline_bound",
            "pre_marker_bound",
            "post_probe_verified",
            "post_probe_failed",
        }
    ):
        raise Wave3ProbeError(
            "shared GPU runtime contract drifted", code="wave3.gpu_contract"
        )
    baseline = contract["baseline"]
    pre_marker = contract["pre_marker_observation"]
    post_probe = contract["post_probe_preterminal_observation"]
    expected_status = "not_started"
    if baseline is not None:
        validated_baseline = _validate_shared_gpu_baseline(baseline)
        expected_status = "baseline_bound"
        if pre_marker is not None:
            _validate_shared_gpu_observation(
                pre_marker,
                baseline=validated_baseline,
                expected_stage="pre_marker",
            )
            expected_status = "pre_marker_bound"
        if post_probe is not None:
            if pre_marker is None:
                raise Wave3ProbeError(
                    "post-probe GPU evidence precedes pre-marker evidence",
                    code="wave3.gpu_contract",
                )
            _validate_gpu_process_subset_sweep(
                post_probe,
                baseline=validated_baseline,
                expected_stage="post_probe_preterminal",
            )
            expected_status = (
                "post_probe_verified"
                if post_probe.get("status") == "passed"
                else "post_probe_failed"
            )
    elif pre_marker is not None or post_probe is not None:
        raise Wave3ProbeError(
            "shared GPU observations lack a baseline", code="wave3.gpu_contract"
        )
    if contract["status"] != expected_status:
        raise Wave3ProbeError(
            "shared GPU runtime status contradicts evidence",
            code="wave3.gpu_contract",
        )
    return dict(contract)


def _validate_runtime_receipt(
    value: Any, *, allow_limit_exceeded: bool = False
) -> dict[str, Any]:
    runtime = _mapping(value, "runtime")
    _require_exact_fields(
        runtime,
        {
            "status",
            "requested_device",
            "precision",
            "memory_savers",
            "cache_access",
            "model_weight_identity",
            "runtime_config_attestation",
            "runtime_source_identities",
            "concrete_trainable_inventory",
            "initial_buffer_inventory",
            "resource_limits",
            "controller",
            "shared_gpu_contract",
            "process_cleanup",
        },
        owner="runtime",
        code="wave3.runtime",
    )
    limits = _mapping(runtime["resource_limits"], "resource_limits")
    expected_limits = _resource_limits()
    if dict(limits) != expected_limits:
        raise Wave3ProbeError("runtime limits drifted", code="wave3.runtime")
    _validate_shared_gpu_runtime_contract(runtime["shared_gpu_contract"])
    process_cleanup = _validate_process_cleanup_receipt(runtime["process_cleanup"])
    controller = _mapping(runtime["controller"], "controller")
    _require_exact_fields(
        controller,
        {
            "status",
            "worker_pid",
            "termination",
            "max_host_rss_bytes",
            "max_gpu_memory_bytes",
        },
        owner="runtime.controller",
        code="wave3.runtime",
    )
    if (
        controller["status"] not in {"not_started", "running", "exited", "terminated"}
        or (
            controller["worker_pid"] is not None
            and not _is_nonnegative_int(controller["worker_pid"])
        )
        or not _is_nonnegative_int(controller["max_host_rss_bytes"])
        or not _is_nonnegative_int(controller["max_gpu_memory_bytes"])
        or (
            not allow_limit_exceeded
            and controller["max_host_rss_bytes"] > HOST_RSS_CEILING_BYTES
        )
        or (
            not allow_limit_exceeded
            and controller["max_gpu_memory_bytes"] > DEVICE_MEMORY_CEILING_BYTES
        )
    ):
        raise Wave3ProbeError(
            "controller runtime evidence drifted", code="wave3.runtime"
        )
    has_termination = isinstance(controller["termination"], str) and bool(
        controller["termination"]
    )
    if (
        controller["status"] == "running"
        or (controller["status"] == "terminated") != has_termination
        or (controller["status"] == "terminated" and controller["worker_pid"] is None)
        or (
            controller["status"] == "not_started"
            and (
                controller["worker_pid"] is not None
                or controller["termination"] is not None
            )
        )
    ):
        raise Wave3ProbeError(
            "controller terminal relation drifted", code="wave3.runtime"
        )
    if runtime["status"] == "not_started":
        if any(
            runtime[field] is not None
            for field in (
                "memory_savers",
                "model_weight_identity",
                "runtime_config_attestation",
                "runtime_source_identities",
                "concrete_trainable_inventory",
                "initial_buffer_inventory",
            )
        ):
            raise Wave3ProbeError(
                "not-started runtime has setup evidence", code="wave3.runtime"
            )
    elif runtime["status"] in {"cpu_ready", "gpu_ready"}:
        if not isinstance(runtime["memory_savers"], Mapping):
            raise Wave3ProbeError(
                "runtime memory saver evidence is missing", code="wave3.runtime"
            )
        validate_runtime_config_attestation(runtime["runtime_config_attestation"])
        _validate_model_weight_identity(runtime["model_weight_identity"])
        _validate_runtime_source_identities(runtime["runtime_source_identities"])
        _validate_concrete_inventory(runtime["concrete_trainable_inventory"])
        _validate_buffer_inventory(runtime["initial_buffer_inventory"])
    else:
        raise Wave3ProbeError("runtime status is invalid", code="wave3.runtime")
    if (
        runtime["precision"] != "bf16"
        or runtime["cache_access"] != "read_only_no_writes"
        or runtime["requested_device"]
        != _parse_cuda_device_text(runtime["requested_device"])
    ):
        raise Wave3ProbeError("runtime identity drifted", code="wave3.runtime")
    if runtime["controller"]["worker_pid"] is None:
        if process_cleanup["status"] != "not_applicable":
            raise Wave3ProbeError(
                "cleanup evidence exists without a worker", code="wave3.process_cleanup"
            )
    elif process_cleanup["status"] == "not_applicable":
        raise Wave3ProbeError(
            "spawned worker lacks cleanup evidence", code="wave3.process_cleanup"
        )
    return dict(runtime)


def _validate_comparison_receipts(
    value: Any, *, oracle: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    comparisons = _mapping(value, "comparisons")
    _require_exact_fields(
        comparisons,
        set(COMPARISON_ORDER),
        owner="comparisons",
        code="wave3.comparison",
    )
    result: dict[str, Any] = {}
    scalar_names = {
        "raw_diagnostic_same_logits": (LOSS_RTOL, LOSS_ATOL),
        "zero_reference_base_only_total": (LOSS_RTOL, LOSS_ATOL),
        "zero_optimized_base_only_total": (LOSS_RTOL, LOSS_ATOL),
        "zero_total": (BF16_RTOL, BF16_ATOL),
        "nonzero_total": (BF16_RTOL, BF16_ATOL),
    }
    for name in COMPARISON_ORDER:
        row = comparisons[name]
        if row is None:
            result[name] = None
        elif name in scalar_names:
            result[name] = _validate_scalar_comparison(
                row,
                expected_rtol=scalar_names[name][0],
                expected_atol=scalar_names[name][1],
            )
        elif name.endswith("_gradients"):
            result[name] = _validate_gradient_comparison(row)
        else:
            control = _mapping(row, name)
            _require_exact_fields(
                control,
                {"reference_differentiable", "optimized_differentiable", "passed"},
                owner=name,
                code="wave3.comparison",
            )
            if control["passed"] is not (
                control["reference_differentiable"] is True
                and control["optimized_differentiable"] is True
            ):
                raise Wave3ProbeError(
                    "nonzero control relation drifted", code="wave3.comparison"
                )
            result[name] = dict(control)
    if (
        all(arms[name]["status"] == "completed" for name in ARM_ORDER)
        and oracle["status"] == "completed"
    ):
        artifacts = {name: arms[name]["artifact"] for name in ARM_ORDER}
        scalar_bindings = {
            "raw_diagnostic_same_logits": oracle["raw_diagnostic_comparison"],
            "zero_reference_base_only_total": compare_scalar(
                artifacts["zero_reference"]["total_loss"],
                artifacts["zero_reference"]["base_only_total_loss"],
                rtol=LOSS_RTOL,
                atol=LOSS_ATOL,
            ),
            "zero_optimized_base_only_total": compare_scalar(
                artifacts["zero_optimized"]["total_loss"],
                artifacts["zero_optimized"]["base_only_total_loss"],
                rtol=LOSS_RTOL,
                atol=LOSS_ATOL,
            ),
            "zero_total": compare_scalar(
                artifacts["zero_reference"]["total_loss"],
                artifacts["zero_optimized"]["total_loss"],
                rtol=BF16_RTOL,
                atol=BF16_ATOL,
            ),
            "nonzero_total": compare_scalar(
                artifacts["nonzero_reference"]["total_loss"],
                artifacts["nonzero_optimized"]["total_loss"],
                rtol=BF16_RTOL,
                atol=BF16_ATOL,
            ),
        }
        if any(result[name] != expected for name, expected in scalar_bindings.items()):
            raise Wave3ProbeError(
                "comparison is not arm-bound", code="wave3.comparison"
            )
        control_expected = {
            "reference_differentiable": artifacts["nonzero_reference"][
                "nonzero_control_differentiable"
            ],
            "optimized_differentiable": artifacts["nonzero_optimized"][
                "nonzero_control_differentiable"
            ],
            "passed": True,
        }
        if result["nonzero_control_graph"] != control_expected:
            raise Wave3ProbeError(
                "control comparison is not arm-bound", code="wave3.comparison"
            )
        for comparison_name, left_name, right_name in (
            ("zero_gradients", "zero_reference", "zero_optimized"),
            ("nonzero_gradients", "nonzero_reference", "nonzero_optimized"),
        ):
            gradient = result[comparison_name]
            if gradient is None:
                raise Wave3ProbeError(
                    "gradient comparison is absent", code="wave3.comparison"
                )
            left_rows = artifacts[left_name]["gradients"]["rows"]
            right_rows = artifacts[right_name]["gradients"]["rows"]
            if (
                gradient["left_count"] != len(left_rows)
                or gradient["right_count"] != len(right_rows)
                or gradient["left_names_sha256"]
                != sha256_json([row["name"] for row in left_rows])
                or gradient["right_names_sha256"]
                != sha256_json([row["name"] for row in right_rows])
            ):
                raise Wave3ProbeError(
                    "gradient comparison is not arm-bound", code="wave3.comparison"
                )
    return result


def _validate_efficiency_receipt(
    value: Any, *, arms: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    efficiency = _mapping(value, "efficiency")
    _require_exact_fields(
        efficiency,
        {"status", "result"},
        owner="efficiency",
        code="wave3.efficiency",
    )
    if efficiency["status"] == "not_run":
        if efficiency["result"] is not None:
            raise Wave3ProbeError(
                "not-run efficiency has evidence", code="wave3.efficiency"
            )
        return dict(efficiency)
    if efficiency["status"] != "completed":
        raise Wave3ProbeError("efficiency status is invalid", code="wave3.efficiency")
    result = _mapping(efficiency["result"], "efficiency.result")
    _require_exact_fields(
        result,
        {
            "scope",
            "reference",
            "optimized",
            "optimized_minus_reference",
            "promotion_claim",
        },
        owner="efficiency.result",
        code="wave3.efficiency",
    )
    if any(
        arms[name]["status"] != "completed"
        for name in ("zero_reference", "zero_optimized")
    ):
        raise Wave3ProbeError(
            "efficiency lacks completed arms", code="wave3.efficiency"
        )
    expected = _efficiency_from_artifacts(
        arms["zero_reference"]["artifact"], arms["zero_optimized"]["artifact"]
    )
    if dict(result) != expected:
        raise Wave3ProbeError("efficiency is not recomputable", code="wave3.efficiency")
    return dict(efficiency)


def _validate_published_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = payload.get("schema")
    schema_hash_fields = {
        PLAN_SCHEMA: "plan_sha256",
        MARKER_SCHEMA: "marker_sha256",
        RECEIPT_SCHEMA: "receipt_sha256",
        PUBLICATION_FAILURE_SCHEMA: "publication_failure_sha256",
    }
    hash_field = schema_hash_fields.get(schema)
    if hash_field is None:
        canonical_json_bytes(payload)
        return dict(payload)
    return _validate_finalized(payload, schema=str(schema), hash_field=hash_field)


def _reload_exact_published_payload(target: Path, intended: Mapping[str, Any]) -> bool:
    try:
        if target.stat().st_size > MAX_JSON_BYTES:
            return False
        persisted = _validate_published_payload(load_strict_json(target))
        validated_intended = _validate_published_payload(intended)
    except BaseException:
        return False
    return persisted == validated_intended


def publish_json_absent(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    on_linked: Any | None = None,
) -> Path:
    """No-replace publish, directory-fsync, then strict reload and hash-check."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise Wave3ProbeError(
            f"artifact target is a symlink: {requested}",
            code="wave3.immutable_collision",
        )
    target = requested.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    if len(data) > MAX_JSON_BYTES:
        raise Wave3ProbeError(
            "artifact exceeds JSON byte bound", code="wave3.artifact_size"
        )
    _validate_published_payload(payload)
    linked_by_this_call = False

    def record_linked() -> None:
        nonlocal linked_by_this_call
        linked_by_this_call = True
        if on_linked is not None:
            on_linked()

    try:
        write_strict_json_atomic(target, payload, on_linked=record_linked)
        if not _reload_exact_published_payload(target, payload):
            raise Wave3ProbeError(
                "published artifact failed strict reload/hash validation",
                code="wave3.artifact_persistence",
            )
    except BaseException as exc:
        if not linked_by_this_call and isinstance(exc, ParityContractError):
            if exc.code == "qwen.parity.artifact_collision":
                raise Wave3ProbeError(
                    f"artifact target already exists: {target}",
                    code="wave3.immutable_collision",
                ) from exc
        reloaded_exact = bool(
            linked_by_this_call and _reload_exact_published_payload(target, payload)
        )
        raise Wave3ArtifactPublicationError(
            "artifact publication failed after own link"
            if linked_by_this_call
            else "artifact publication failed before link",
            code=(
                "wave3.artifact_post_link"
                if linked_by_this_call
                else "wave3.artifact_pre_link"
            ),
            linked_by_this_call=linked_by_this_call,
            reloaded_exact=reloaded_exact,
        ) from exc
    return target


def build_plan(
    *,
    config_path: str | Path,
    cache_dir: str | Path,
    plan_path: str | Path,
    receipt_path: str | Path,
    attempt_marker_path: str | Path,
    publication_failure_path: str | Path,
    device: str,
) -> dict[str, Any]:
    config_target = Path(config_path).expanduser().resolve(strict=True)
    if config_target != FROZEN_CONFIG_PATH.resolve(strict=True):
        raise Wave3ProbeError(
            "frozen config path drifted", code="wave3.config_identity"
        )
    cache_path = Path(cache_dir).expanduser().resolve(strict=True)
    if cache_path != W0_CACHE_DIR.resolve(strict=True):
        raise Wave3ProbeError(
            "private Wave 0 cache path drifted", code="wave3.cache_path"
        )
    manifest, micro_step = _load_frozen_micro_step(cache_path)
    targets = {
        "plan": str(Path(plan_path).expanduser().resolve()),
        "attempt_marker": str(Path(attempt_marker_path).expanduser().resolve()),
        "receipt": str(Path(receipt_path).expanduser().resolve()),
        "publication_failure": str(
            Path(publication_failure_path).expanduser().resolve()
        ),
    }
    if len(set(targets.values())) != 4:
        raise Wave3ProbeError(
            "artifact targets must be distinct", code="wave3.artifact_targets"
        )
    requested_device = _parse_cuda_device_text(device)
    resolved = load_train_config(config_target)
    if resolved.fingerprint != FROZEN_CONFIG_FINGERPRINT:
        raise Wave3ProbeError(
            "frozen config fingerprint drifted", code="wave3.config_identity"
        )
    compatibility_projection = build_config_compatibility_projection(
        resolved.config_dict
    )
    runtime_config_attestation = frozen_runtime_config_attestation(
        compatibility_projection
    )
    components = load_qwen_components(resolved.config, load_model=False)
    model_weight_identity = base_model_weight_identity(components.base_model_path)
    if (
        model_weight_identity["aggregate_sha256"]
        != EXPECTED_BASE_MODEL_WEIGHT_AGGREGATE_SHA256
    ):
        raise Wave3ProbeError(
            "frozen base-model weight identity drifted",
            code="wave3.model_weight_identity",
        )
    selection = _workload_identity(micro_step)
    prepare_argv = _prepare_argv(
        config_path=config_target,
        cache_dir=cache_path,
        plan_path=targets["plan"],
        receipt_path=targets["receipt"],
        attempt_marker_path=targets["attempt_marker"],
        publication_failure_path=targets["publication_failure"],
        device=requested_device,
    )
    run_argv = _artifact_argv(
        command="run",
        plan_path=targets["plan"],
        receipt_path=targets["receipt"],
        attempt_marker_path=targets["attempt_marker"],
        publication_failure_path=targets["publication_failure"],
        device=requested_device,
    )
    controller_argv = _artifact_argv(
        command="controller",
        plan_path=targets["plan"],
        receipt_path=targets["receipt"],
        attempt_marker_path=targets["attempt_marker"],
        publication_failure_path=targets["publication_failure"],
        device=requested_device,
    )
    plan = _finalize(
        {
            "schema": PLAN_SCHEMA,
            "status": "prepared",
            "model_weight_identity": model_weight_identity,
            "config": {
                "entry_path": str(resolved.entry_config_path),
                "fingerprint": resolved.fingerprint,
                "resolved_sha256": sha256_json(resolved.config_dict),
                "forward_input_provider_mode": str(
                    resolved.config.training.forward_input_provider_mode
                ),
                "compatibility_projection": compatibility_projection,
                "runtime_config_attestation": runtime_config_attestation,
                "model_identity": components.to_artifact_dict(),
            },
            "cache": {
                "path": str(cache_path),
                "access": "read_only_private_immutable",
                "version": manifest["version"],
                "fingerprint": manifest["fingerprint"],
                "manifest_sha256": W0_MANIFEST_SHA256,
                "chunk_sha256": manifest["chunks"][0]["sha256"],
                "shared_cache_writes": False,
            },
            "workload": selection,
            "loss_contract": {
                "zero_weight": 0.0,
                "nonzero_control_weight": float(
                    resolved.config.losses.protected.token_type_gate.weight
                ),
                "raw_diagnostic_tolerance": {"rtol": LOSS_RTOL, "atol": LOSS_ATOL},
                "bf16_derived_tolerance": {"rtol": BF16_RTOL, "atol": BF16_ATOL},
                "raw_oracle": "one_frozen_fp32_logits_tensor_two_diagnostic_paths",
                "zero_total_reference": "base_and_nonzero_terms_plus_graph_connected_raw_times_zero",
                "optimized_zero_total": "base_and_nonzero_terms_only",
                "expected_trainable_gradient_count": EXPECTED_TRAINABLE_COUNT,
            },
            "execution_contract": {
                "arm_order": list(ARM_ORDER),
                "oracle_arm_precedes_measured_arms": True,
                "model_forward_count_per_arm": 1,
                "backward_count_per_measured_arm": 1,
                "state_restore_between_arms": (
                    "trainable_values_named_buffers_train_mode_and_rng_exact"
                ),
                "wall_clock": (
                    "external_monotonic_watchdog_and_cuda_synchronized_perf_counter"
                ),
                "arm_wall_ceiling_seconds": ARM_WALL_CEILING_SECONDS,
                "host_rss_ceiling_bytes": HOST_RSS_CEILING_BYTES,
                "device_memory_ceiling_bytes": DEVICE_MEMORY_CEILING_BYTES,
                "controller_poll_seconds": CONTROLLER_POLL_SECONDS,
                "process_cleanup_timeout_seconds": PROCESS_CLEANUP_TIMEOUT_SECONDS,
                "max_tracked_processes": MAX_TRACKED_PROCESSES,
                "terminal_order": (
                    "process_tree_cleanup_then_two_sample_gpu_subset_sweep"
                ),
                "shared_gpu_total_memory_mib": SHARED_GPU_TOTAL_MEMORY_MIB,
                "shared_gpu_max_preexisting_memory_mib": (
                    SHARED_GPU_MAX_PREEXISTING_MEMORY_MIB
                ),
                "shared_gpu_required_headroom_mib": SHARED_GPU_REQUIRED_HEADROOM_MIB,
                "shared_gpu_preflight_sample_count": SHARED_GPU_SAMPLE_COUNT,
                "shared_gpu_minimum_sample_interval_seconds": (
                    SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS
                ),
                "shared_gpu_utilization_disposition": (
                    "observational_only_not_admission_or_promotion"
                ),
                "shared_gpu_baseline_process_policy": (
                    "stable_exact_gpu_uuid_driver_pid_then_post_probe_subset"
                ),
                "baseline_process_action": "observe_only_never_signal_or_reset",
                "claim_boundary": shared_gpu_claim_boundary(),
                "run_wall_ceiling_seconds": RUN_WALL_CEILING_SECONDS,
                "requested_device": requested_device,
                "retry": False,
                "sample_switch": False,
                "tolerance_change": False,
                "prepare_argv": prepare_argv,
                "prepare_argv_sha256": sha256_json(prepare_argv),
                "run_argv": run_argv,
                "run_argv_sha256": sha256_json(run_argv),
                "controller_argv": controller_argv,
                "controller_argv_sha256": sha256_json(controller_argv),
            },
            "artifact_targets": targets,
            "source_identity": {
                path: sha256_file(REPO_ROOT / path) for path in SOURCE_OWNERS
            },
            "provenance": collect_execution_provenance(repository_root=REPO_ROOT),
        },
        hash_field="plan_sha256",
    )
    return validate_plan(plan)


def compare_scalar(
    left: float, right: float, *, rtol: float, atol: float
) -> dict[str, Any]:
    values_finite = math.isfinite(left) and math.isfinite(right)
    difference = abs(left - right) if values_finite else None
    threshold = atol + rtol * abs(right) if values_finite else None
    return {
        "left": left if math.isfinite(left) else None,
        "right": right if math.isfinite(right) else None,
        "rtol": rtol,
        "atol": atol,
        "max_abs_diff": difference,
        "threshold": threshold,
        "finite": bool(values_finite),
        "passed": bool(
            values_finite
            and difference is not None
            and threshold is not None
            and difference <= threshold
        ),
    }


def compare_gradient_maps(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    missing = sorted(set(left) - set(right))
    extra = sorted(set(right) - set(left))
    rows: list[dict[str, Any]] = []
    mismatched: list[str] = []
    maximum = 0.0
    for name in sorted(set(left) | set(right)):
        left_present = name in left
        right_present = name in right
        lhs = left[name].detach().float().cpu() if left_present else None
        rhs = right[name].detach().float().cpu() if right_present else None
        shape_match = bool(
            lhs is not None and rhs is not None and lhs.shape == rhs.shape
        )
        left_finite = bool(lhs is not None and torch.isfinite(lhs).all())
        right_finite = bool(rhs is not None and torch.isfinite(rhs).all())
        if shape_match and left_finite and right_finite:
            assert lhs is not None and rhs is not None
            differences = torch.abs(lhs - rhs)
            thresholds = BF16_ATOL + BF16_RTOL * torch.abs(rhs)
            within = differences <= thresholds
            element_count = int(differences.numel())
            within_count = int(torch.count_nonzero(within).item())
            max_abs_diff = float(differences.max().item()) if element_count else 0.0
            max_threshold_excess = (
                float((differences - thresholds).max().item())
                if element_count
                else -BF16_ATOL
            )
            maximum = max(maximum, max_abs_diff)
        else:
            element_count = 0
            within_count = 0
            max_abs_diff = 0.0
            max_threshold_excess = 0.0
        passed = bool(
            left_present
            and right_present
            and shape_match
            and left_finite
            and right_finite
            and within_count == element_count
            and max_threshold_excess <= 0.0
        )
        if not passed:
            mismatched.append(name)
        rows.append(
            {
                "name": name,
                "left_present": left_present,
                "right_present": right_present,
                "left_shape": None if lhs is None else list(lhs.shape),
                "right_shape": None if rhs is None else list(rhs.shape),
                "left_dtype": None if lhs is None else str(lhs.dtype),
                "right_dtype": None if rhs is None else str(rhs.dtype),
                "left_finite": left_finite,
                "right_finite": right_finite,
                "element_count": element_count,
                "within_count": within_count,
                "max_abs_diff": max_abs_diff,
                "max_threshold_excess": max_threshold_excess,
                "passed": passed,
            }
        )
    left_names = sorted(left)
    right_names = sorted(right)
    return {
        "rtol": BF16_RTOL,
        "atol": BF16_ATOL,
        "left_count": len(left),
        "right_count": len(right),
        "left_names_sha256": sha256_json(left_names),
        "right_names_sha256": sha256_json(right_names),
        "missing": missing,
        "extra": extra,
        "mismatched": sorted(mismatched),
        "rows": rows,
        "global_max_abs_diff": maximum,
        "passed": not missing and not extra and not mismatched,
    }


def _efficiency_comparison(arms: Mapping[str, ArmExecution]) -> dict[str, Any]:
    return _efficiency_from_artifacts(
        arms["zero_reference"].artifact, arms["zero_optimized"].artifact
    )


def _efficiency_from_artifacts(
    reference_artifact: Mapping[str, Any], optimized_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    def metrics(artifact: Mapping[str, Any]) -> dict[str, float | int]:
        gpu = _mapping(artifact["resources_after"]["gpu"], "arm.gpu")
        cpu = _mapping(artifact["resources_after"]["cpu"], "arm.cpu")
        return {
            "wall_seconds": float(artifact["wall_seconds"]),
            "peak_allocated_bytes": int(gpu["max_memory_allocated_bytes"]),
            "peak_reserved_bytes": int(gpu["max_memory_reserved_bytes"]),
            "max_host_rss_bytes": int(cpu["max_rss_bytes"]),
        }

    reference = metrics(reference_artifact)
    optimized = metrics(optimized_artifact)
    return {
        "scope": "single_forward_loss_backward_per_arm",
        "reference": reference,
        "optimized": optimized,
        "optimized_minus_reference": {
            key: optimized[key] - reference[key] for key in reference
        },
        "promotion_claim": "nonpromotional_shared_load_observation",
    }


def assert_single_forward_same_logits(
    *,
    forward_count: int,
    backward_count: int,
    logits: torch.Tensor,
    diagnostic_logits: Sequence[torch.Tensor],
) -> None:
    if forward_count != 1:
        raise Wave3ProbeError(
            "arm must execute exactly one model forward", code="wave3.forward_count"
        )
    if backward_count not in {0, 1}:
        raise Wave3ProbeError(
            "arm backward count is invalid", code="wave3.backward_count"
        )
    if not diagnostic_logits or any(value is not logits for value in diagnostic_logits):
        raise Wave3ProbeError(
            "diagnostic paths must consume the exact same logits tensor",
            code="wave3.same_logits",
        )


def prepare_command(args: argparse.Namespace) -> int:
    target = _assert_absent_target(args.plan)
    _assert_absent_target(args.receipt)
    _assert_absent_target(args.attempt_marker)
    _assert_absent_target(args.publication_failure)
    plan = build_plan(
        config_path=args.config,
        cache_dir=args.cache_dir,
        plan_path=args.plan,
        receipt_path=args.receipt,
        attempt_marker_path=args.attempt_marker,
        publication_failure_path=args.publication_failure,
        device=args.device,
    )
    post_link_recovery = False
    try:
        publish_json_absent(target, plan)
    except Wave3ArtifactPublicationError as exc:
        if not (exc.linked_by_this_call and exc.reloaded_exact):
            raise
        post_link_recovery = True
    print(
        json.dumps(
            {
                "status": "prepared",
                "plan": str(target),
                "plan_sha256": plan["plan_sha256"],
                "post_link_recovery": post_link_recovery,
            },
            sort_keys=True,
        )
    )
    return 0


def run_command(args: argparse.Namespace) -> int:
    return _controller_command(args)


def controller_command(args: argparse.Namespace) -> int:
    return _controller_command(args)


def _load_and_bind_active_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Admit only one exact active v4 plan before any execution lifecycle."""

    plan = validate_plan(_load_json(args.plan))
    targets = _mapping(plan["artifact_targets"], "artifact_targets")
    if (
        Path(args.plan).expanduser().resolve() != Path(str(targets["plan"]))
        or Path(args.receipt).expanduser().resolve() != Path(str(targets["receipt"]))
        or Path(args.attempt_marker).expanduser().resolve()
        != Path(str(targets["attempt_marker"]))
        or Path(args.publication_failure).expanduser().resolve()
        != Path(str(targets["publication_failure"]))
        or str(args.device) != plan["execution_contract"]["requested_device"]
    ):
        raise Wave3ProbeError(
            "entry arguments differ from the active plan",
            code="wave3.run_binding",
        )
    return plan


def _controller_command(args: argparse.Namespace) -> int:
    plan = _load_and_bind_active_plan(args)
    run_started = time.monotonic()
    targets = _mapping(plan["artifact_targets"], "artifact_targets")
    receipt_target = Path(str(targets["receipt"]))
    marker_target = Path(str(targets["attempt_marker"]))
    publication_failure_target = Path(str(targets["publication_failure"]))
    evidence = _empty_evidence(
        requested_device=plan["execution_contract"]["requested_device"],
        marker_path=marker_target,
        plan=plan,
    )
    process: subprocess.Popen[bytes] | None = None
    shared_gpu_baseline: dict[str, Any] | None = None
    controller_token_sha256: str | None = None
    controller_state: dict[str, Any] = {
        "status": "not_started",
        "worker_pid": None,
        "termination": None,
        "max_host_rss_bytes": 0,
        "max_gpu_memory_bytes": 0,
        "current_arm": None,
        "current_arm_started": None,
        "terminal_receipt": None,
        "post_probe_preterminal_observation": None,
        "process_cleanup": _empty_process_cleanup(),
        "owned_process_identities": {},
        "owned_session_ids": [],
    }
    terminal: dict[str, Any] | None = None
    primary_exc: BaseException | None = None
    device: torch.device | None = None
    try:
        device = _parse_cuda_device(args.device)
        evidence = _empty_evidence(
            requested_device=str(device), marker_path=marker_target, plan=plan
        )
        _assert_absent_target(receipt_target)
        _assert_absent_target(marker_target)
        _assert_absent_target(publication_failure_target)
        evidence["completed_phases"].append("plan_revalidated")
        _assert_deadline(run_started)
        shared_gpu_baseline = _capture_shared_gpu_baseline(device)
        evidence["runtime"]["shared_gpu_contract"].update(
            {
                "status": "baseline_bound",
                "baseline": shared_gpu_baseline,
            }
        )
        evidence["completed_phases"].append("shared_gpu_preflight")
        worker_env = os.environ.copy()
        worker_env[CONTROLLER_PID_ENV] = str(os.getpid())
        controller_token = os.urandom(32).hex()
        controller_token_sha256 = hashlib.sha256(
            controller_token.encode("ascii")
        ).hexdigest()
        worker_env[CONTROLLER_TOKEN_ENV] = controller_token
        worker_env[CONTROLLER_BASELINE_ENV] = canonical_json_bytes(
            shared_gpu_baseline
        ).decode("ascii")
        process = subprocess.Popen(
            _worker_argv(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=worker_env,
        )
        controller_state.update(
            {
                "status": "running",
                "worker_pid": process.pid,
            }
        )
        _record_owned_process_tree(process.pid, controller_state)
        _watch_worker(
            process,
            device=device,
            run_started=run_started,
            evidence=evidence,
            controller_state=controller_state,
        )
        received = controller_state.get("terminal_receipt")
        if not isinstance(received, Mapping):
            raise Wave3ProbeError(
                "worker exited without a terminal receipt", code="wave3.controller"
            )
        terminal = dict(received)
    except BaseException as exc:
        primary_exc = exc
    if process is not None:
        cleanup_reason = (
            str(getattr(primary_exc, "code", type(primary_exc).__name__))
            if primary_exc is not None
            else "worker_terminal_cleanup"
        )
        if device is None or shared_gpu_baseline is None:
            raise Wave3ProbeError(
                "spawned worker lacks its admitted GPU baseline",
                code="wave3.gpu_baseline",
            )
        _finalize_spawned_worker(
            process,
            device=device,
            shared_gpu_baseline=shared_gpu_baseline,
            controller_state=controller_state,
            reason=cleanup_reason,
        )
    cleanup = controller_state["process_cleanup"]
    sweep = controller_state["post_probe_preterminal_observation"]
    terminal_gate_exc: BaseException | None = primary_exc
    if terminal_gate_exc is None and cleanup["status"] == "failed":
        terminal_gate_exc = Wave3ProbeError(
            "worker process cleanup failed",
            code="wave3.process_cleanup_incomplete",
        )
    if (
        terminal_gate_exc is None
        and isinstance(sweep, Mapping)
        and sweep.get("status") == "failed"
    ):
        terminal_gate_exc = Wave3ProbeError(
            "post-probe GPU subset sweep failed",
            code="wave3.gpu_postcheck",
        )
    if terminal_gate_exc is not None or terminal is None:
        _consume_marker_if_present(
            marker_target,
            plan=plan,
            evidence=evidence,
            publication_owner_token_sha256=controller_token_sha256,
        )
        terminal = _failure_receipt(
            plan,
            evidence=evidence,
            exc=(
                terminal_gate_exc
                if terminal_gate_exc is not None
                else Wave3ProbeError(
                    "worker produced no terminal receipt", code="wave3.controller"
                )
            ),
        )
    controller_state["status"] = (
        "terminated" if controller_state.get("termination") else "exited"
    )
    terminal = _bind_controller_receipt(terminal, controller_state)
    expected_plan = plan
    try:
        validate_receipt(terminal, expected_plan=expected_plan)
        publication = _publish_terminal_receipt(
            receipt_target=receipt_target,
            publication_failure_target=publication_failure_target,
            receipt=terminal,
            plan=plan,
        )
    except BaseException as publication_exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "code": str(
                        getattr(
                            publication_exc,
                            "code",
                            type(publication_exc).__name__,
                        )
                    ),
                    "receipt": str(receipt_target),
                    "publication_failure": str(publication_failure_target),
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "receipt": str(receipt_target),
                **publication,
            },
            sort_keys=True,
        )
    )
    return 0 if terminal["status"] == "passed" else 1


def _finalize_spawned_worker(
    process: subprocess.Popen[Any],
    *,
    device: torch.device,
    shared_gpu_baseline: Mapping[str, Any],
    controller_state: dict[str, Any],
    reason: str,
) -> None:
    try:
        cleanup = _cleanup_worker_process_tree(
            process,
            controller_state=controller_state,
            reason=reason,
        )
    except BaseException as exc:
        tracked = [
            dict(row)
            for row in controller_state.get("owned_process_identities", {}).values()
            if isinstance(row, Mapping)
        ]
        cleanup = {
            "schema": PROCESS_CLEANUP_SCHEMA,
            "status": "failed",
            "reason": reason,
            "leader_pid": process.pid,
            "cleanup_timeout_seconds": PROCESS_CLEANUP_TIMEOUT_SECONDS,
            "tracked_processes": tracked,
            "term_signals": [],
            "kill_signals": [],
            "leader_absent": False,
            "descendants_absent": False,
            "sessions_absent": False,
            "surviving_processes": tracked,
            "completed_monotonic_ns": time.monotonic_ns(),
            "terminal_reason": {
                "code": str(getattr(exc, "code", type(exc).__name__)),
                "type": type(exc).__name__,
                "message": str(exc)[:2048] or "process cleanup failed",
            },
        }
    controller_state["process_cleanup"] = cleanup
    controller_state["post_probe_preterminal_observation"] = (
        _capture_gpu_process_subset_sweep(
            device,
            shared_gpu_baseline,
            stage="post_probe_preterminal",
        )
    )


def worker_command(args: argparse.Namespace) -> int:
    plan = _load_and_bind_active_plan(args)
    controller_token = _require_controller_parent()
    targets = _mapping(plan["artifact_targets"], "artifact_targets")
    marker_target = Path(str(targets["attempt_marker"]))
    evidence = _empty_evidence(
        requested_device=plan["execution_contract"]["requested_device"],
        marker_path=marker_target,
        plan=plan,
    )
    started = time.monotonic()
    try:
        device = _parse_cuda_device(args.device)
        shared_gpu_baseline = _load_controller_shared_gpu_baseline(device)
        evidence = _empty_evidence(
            requested_device=str(device), marker_path=marker_target, plan=plan
        )
        evidence["runtime"]["shared_gpu_contract"].update(
            {
                "status": "baseline_bound",
                "baseline": shared_gpu_baseline,
            }
        )
        current = build_plan(
            config_path=plan["config"]["entry_path"],
            cache_dir=plan["cache"]["path"],
            plan_path=targets["plan"],
            receipt_path=targets["receipt"],
            attempt_marker_path=targets["attempt_marker"],
            publication_failure_path=targets["publication_failure"],
            device=str(device),
        )
        if current != plan:
            raise Wave3ProbeError(
                "prepared plan no longer revalidates", code="wave3.plan_drift"
            )
        evidence["completed_phases"].extend(
            ["plan_revalidated", "shared_gpu_preflight"]
        )
        terminal = _execute_gpu_probe(
            plan,
            device=device,
            evidence=evidence,
            run_started=started,
            marker_target=marker_target,
            publication_owner_token_sha256=hashlib.sha256(
                controller_token.encode("ascii")
            ).hexdigest(),
            worker_argv=_worker_argv(args),
            shared_gpu_baseline=shared_gpu_baseline,
            emit=lambda event: _emit_worker_event(event),
        )
    except BaseException as exc:
        terminal = _failure_receipt(plan, evidence=evidence, exc=exc)
    _emit_worker_event({"event": "terminal", "receipt": terminal})
    return 0 if terminal["status"] == "passed" else 1


def _require_controller_parent() -> str:
    parent_text = os.environ.get(CONTROLLER_PID_ENV)
    token = os.environ.get(CONTROLLER_TOKEN_ENV)
    if (
        parent_text != str(os.getppid())
        or not isinstance(token, str)
        or len(token) != 64
        or any(char not in "0123456789abcdef" for char in token)
    ):
        raise Wave3ProbeError(
            "internal worker requires its live one-shot controller parent",
            code="wave3.controller_parent",
        )
    return token


def _load_controller_shared_gpu_baseline(device: torch.device) -> dict[str, Any]:
    text = os.environ.get(CONTROLLER_BASELINE_ENV)
    if (
        not isinstance(text, str)
        or not text
        or len(text.encode("utf-8")) > MAX_JSON_BYTES
    ):
        raise Wave3ProbeError(
            "internal worker lacks the controller GPU baseline",
            code="wave3.gpu_baseline",
        )
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise Wave3ProbeError(
            "controller GPU baseline is malformed", code="wave3.gpu_baseline"
        ) from exc
    baseline = _validate_shared_gpu_baseline(value)
    if baseline["requested_device"] != str(device):
        raise Wave3ProbeError(
            "controller GPU baseline selects another device",
            code="wave3.gpu_baseline",
        )
    return baseline


def _emit_worker_event(event: Mapping[str, Any]) -> None:
    data = canonical_json_bytes(event)
    if len(data) > MAX_JSON_BYTES:
        raise Wave3ProbeError("controller event is too large", code="wave3.controller")
    sys.stdout.buffer.write(EVENT_PREFIX.encode("ascii") + data + b"\n")
    sys.stdout.buffer.flush()


def _watch_worker(
    process: subprocess.Popen[bytes],
    *,
    device: torch.device,
    run_started: float,
    evidence: dict[str, Any],
    controller_state: dict[str, Any],
) -> None:
    if process.stdout is None or process.stderr is None:
        raise Wave3ProbeError("worker pipes are unavailable", code="wave3.controller")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    stdout_buffer = bytearray()
    stderr_bytes = bytearray()
    last_gpu_sample = 0.0
    while True:
        now = time.monotonic()
        _record_owned_process_tree(process.pid, controller_state)
        if now - run_started > RUN_WALL_CEILING_SECONDS:
            raise Wave3ProbeError(
                "external controller exceeded 3-hour run deadline",
                code="wave3.run_wall",
            )
        arm_started = controller_state.get("current_arm_started")
        if (
            arm_started is not None
            and now - float(arm_started) > ARM_WALL_CEILING_SECONDS
        ):
            raise Wave3ProbeError(
                "external controller exceeded 30-minute arm deadline",
                code="wave3.arm_wall",
            )
        rss = _process_tree_rss_bytes(process.pid)
        controller_state["max_host_rss_bytes"] = max(
            int(controller_state["max_host_rss_bytes"]), rss
        )
        if rss > HOST_RSS_CEILING_BYTES:
            raise Wave3ProbeError(
                "external controller exceeded 64 GiB host RSS",
                code="wave3.host_rss",
            )
        if now - last_gpu_sample >= 1.0:
            gpu_bytes = _gpu_memory_used_bytes(device)
            controller_state["max_gpu_memory_bytes"] = max(
                int(controller_state["max_gpu_memory_bytes"]), gpu_bytes
            )
            if gpu_bytes > DEVICE_MEMORY_CEILING_BYTES:
                raise Wave3ProbeError(
                    "external controller exceeded 76 GiB GPU memory",
                    code="wave3.device_memory",
                )
            last_gpu_sample = now
        for key, _mask in selector.select(timeout=CONTROLLER_POLL_SECONDS):
            chunk = os.read(key.fileobj.fileno(), 65536)
            if not chunk:
                try:
                    selector.unregister(key.fileobj)
                except KeyError:
                    pass
                continue
            if key.data == "stderr":
                if len(stderr_bytes) < 65536:
                    stderr_bytes.extend(chunk[: 65536 - len(stderr_bytes)])
                continue
            stdout_buffer.extend(chunk)
            if len(stdout_buffer) > MAX_JSON_BYTES + len(EVENT_PREFIX) + 1:
                raise Wave3ProbeError(
                    "worker event stream exceeds bound", code="wave3.controller"
                )
            while b"\n" in stdout_buffer:
                line, _, remainder = stdout_buffer.partition(b"\n")
                stdout_buffer = bytearray(remainder)
                _consume_worker_event_line(
                    line, evidence=evidence, controller_state=controller_state
                )
        returncode = process.poll()
        if returncode is None:
            continue
        for stream, owner in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            remainder = stream.read() or b""
            if owner == "stdout":
                stdout_buffer.extend(remainder)
            elif len(stderr_bytes) < 65536:
                stderr_bytes.extend(remainder[: 65536 - len(stderr_bytes)])
        for line in bytes(stdout_buffer).splitlines():
            _consume_worker_event_line(
                line, evidence=evidence, controller_state=controller_state
            )
        terminal = controller_state.get("terminal_receipt")
        if not isinstance(terminal, Mapping):
            detail = stderr_bytes.decode("utf-8", errors="replace")[-2048:]
            raise Wave3ProbeError(
                f"worker exited without terminal receipt: {detail}",
                code="wave3.controller",
            )
        status = terminal.get("status")
        if (status == "passed" and returncode != 0) or (
            status == "failed" and returncode == 0
        ):
            raise Wave3ProbeError(
                "worker exit code contradicts receipt", code="wave3.controller"
            )
        return


def _consume_worker_event_line(
    line: bytes,
    *,
    evidence: dict[str, Any],
    controller_state: dict[str, Any],
) -> None:
    prefix = EVENT_PREFIX.encode("ascii")
    if not line.startswith(prefix):
        return
    try:
        event = json.loads(line[len(prefix) :])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Wave3ProbeError(
            "worker event is malformed", code="wave3.controller"
        ) from exc
    if not isinstance(event, Mapping) or not isinstance(event.get("event"), str):
        raise Wave3ProbeError("worker event is malformed", code="wave3.controller")
    kind = event["event"]
    if kind == "arm_started":
        arm = event.get("arm")
        if (
            arm not in (ORACLE_ARM_NAME, *ARM_ORDER)
            or controller_state.get("current_arm") is not None
        ):
            raise Wave3ProbeError(
                "worker arm event order is invalid", code="wave3.controller"
            )
        controller_state["current_arm"] = arm
        controller_state["current_arm_started"] = time.monotonic()
    elif kind == "arm_completed":
        if event.get("arm") != controller_state.get("current_arm"):
            raise Wave3ProbeError(
                "worker arm completion is invalid", code="wave3.controller"
            )
        controller_state["current_arm"] = None
        controller_state["current_arm_started"] = None
    elif kind == "progress":
        incoming = event.get("evidence")
        if not isinstance(incoming, Mapping):
            raise Wave3ProbeError(
                "worker progress is malformed", code="wave3.controller"
            )
        for field in (
            "attempt_marker",
            "completed_phases",
            "completed_arm_order",
            "counts",
            "oracle",
            "arms",
            "comparisons",
            "efficiency",
            "runtime",
        ):
            if field not in incoming:
                raise Wave3ProbeError(
                    "worker progress is incomplete", code="wave3.controller"
                )
            evidence[field] = incoming[field]
    elif kind == "terminal":
        if controller_state.get("terminal_receipt") is not None:
            raise Wave3ProbeError(
                "worker emitted two terminal receipts", code="wave3.controller"
            )
        receipt = event.get("receipt")
        if not isinstance(receipt, Mapping):
            raise Wave3ProbeError(
                "worker terminal receipt is malformed", code="wave3.controller"
            )
        controller_state["terminal_receipt"] = dict(receipt)
    else:
        raise Wave3ProbeError("worker event kind is unknown", code="wave3.controller")


def _parse_process_stat(text: str) -> dict[str, Any]:
    close = text.rfind(")")
    if close <= 0:
        raise ValueError("missing process comm terminator")
    pid_text = text[: text.find(" ")]
    fields = text[close + 2 :].split()
    if len(fields) < 20:
        raise ValueError("short process stat")
    return {
        "pid": int(pid_text),
        "state": fields[0],
        "parent_pid": int(fields[1]),
        "process_group_id": int(fields[2]),
        "session_id": int(fields[3]),
        "start_time_ticks": int(fields[19]),
    }


def _read_process_identity(pid: int) -> dict[str, Any] | None:
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        row = _parse_process_stat(text)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (PermissionError, OSError, UnicodeError, ValueError) as exc:
        raise Wave3ProbeError(
            "process cleanup inventory is unavailable",
            code="wave3.process_cleanup_inventory",
        ) from exc
    if row["pid"] != int(pid):
        raise Wave3ProbeError(
            "process cleanup inventory PID drifted",
            code="wave3.process_cleanup_inventory",
        )
    return row


def _snapshot_process_table() -> dict[int, dict[str, Any]]:
    table: dict[int, dict[str, Any]] = {}
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError as exc:
        raise Wave3ProbeError(
            "process cleanup cannot enumerate /proc",
            code="wave3.process_cleanup_inventory",
        ) from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        row = _read_process_identity(int(entry.name))
        if row is not None:
            table[row["pid"]] = row
    return table


def _process_identity_key(row: Mapping[str, Any]) -> str:
    return f"{int(row['pid'])}:{int(row['start_time_ticks'])}"


def _same_process_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return int(left["pid"]) == int(right["pid"]) and int(
        left["start_time_ticks"]
    ) == int(right["start_time_ticks"])


def _record_owned_process_tree(
    worker_pid: int, controller_state: dict[str, Any]
) -> list[dict[str, Any]]:
    table = _snapshot_process_table()
    tracked_value = controller_state.setdefault("owned_process_identities", {})
    if not isinstance(tracked_value, dict):
        raise Wave3ProbeError(
            "controller owned-process inventory is malformed",
            code="wave3.process_cleanup_inventory",
        )
    sessions_value = controller_state.setdefault("owned_session_ids", [])
    if not isinstance(sessions_value, list) or not all(
        _is_nonnegative_int(value) for value in sessions_value
    ):
        raise Wave3ProbeError(
            "controller owned-session inventory is malformed",
            code="wave3.process_cleanup_inventory",
        )
    leader = table.get(int(worker_pid))
    if not tracked_value and leader is not None:
        tracked_value[_process_identity_key(leader)] = leader
    live_tracked: dict[int, dict[str, Any]] = {}
    for row_value in tracked_value.values():
        row = _mapping(row_value, "owned_process_identity")
        current = table.get(int(row["pid"]))
        if current is not None and _same_process_identity(row, current):
            live_tracked[current["pid"]] = current
    owned_pids = set(live_tracked)
    owned_sessions = set(int(value) for value in sessions_value)
    owned_sessions.update(row["session_id"] for row in live_tracked.values())
    if leader is not None and (
        not tracked_value
        or any(_same_process_identity(leader, row) for row in tracked_value.values())
    ):
        owned_pids.add(leader["pid"])
        owned_sessions.add(leader["session_id"])
    changed = True
    while changed:
        changed = False
        for row in table.values():
            if row["pid"] in owned_pids:
                continue
            if row["parent_pid"] in owned_pids or row["session_id"] in owned_sessions:
                owned_pids.add(row["pid"])
                owned_sessions.add(row["session_id"])
                changed = True
    for pid in sorted(owned_pids):
        row = table[pid]
        tracked_value[_process_identity_key(row)] = row
    if len(tracked_value) > MAX_TRACKED_PROCESSES:
        raise Wave3ProbeError(
            "owned process tree exceeds the cleanup bound",
            code="wave3.process_cleanup_bound",
        )
    controller_state["owned_session_ids"] = sorted(owned_sessions)
    return [
        dict(row)
        for _key, row in sorted(
            tracked_value.items(), key=lambda item: (item[1]["pid"], item[0])
        )
    ]


def _live_tracked_processes(
    worker_pid: int, controller_state: dict[str, Any]
) -> list[dict[str, Any]]:
    _record_owned_process_tree(worker_pid, controller_state)
    table = _snapshot_process_table()
    result: list[dict[str, Any]] = []
    for row_value in controller_state["owned_process_identities"].values():
        row = _mapping(row_value, "owned_process_identity")
        current = table.get(int(row["pid"]))
        if current is not None and _same_process_identity(row, current):
            result.append(dict(current))
    return sorted(result, key=lambda row: row["pid"])


def _signal_exact_process(row: Mapping[str, Any], sig: signal.Signals) -> bool:
    current = _read_process_identity(int(row["pid"]))
    if current is None or not _same_process_identity(row, current):
        return False
    try:
        os.kill(int(row["pid"]), sig)
    except ProcessLookupError:
        return False
    return True


def _cleanup_worker_process_tree(
    process: subprocess.Popen[Any],
    *,
    controller_state: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + PROCESS_CLEANUP_TIMEOUT_SECONDS
    terminal_reason: dict[str, str] | None = None
    term_signals: list[dict[str, Any]] = []
    kill_signals: list[dict[str, Any]] = []
    try:
        tracked = _record_owned_process_tree(process.pid, controller_state)
        live = _live_tracked_processes(process.pid, controller_state)
        for row in sorted(live, key=lambda value: value["pid"] == process.pid):
            if _signal_exact_process(row, signal.SIGTERM):
                term_signals.append(dict(row))
        if term_signals and controller_state.get("termination") is None:
            controller_state["termination"] = reason
        while time.monotonic() < deadline:
            process.poll()
            live = _live_tracked_processes(process.pid, controller_state)
            if not live:
                break
            time.sleep(PROCESS_CLEANUP_POLL_SECONDS)
        if live:
            for row in sorted(live, key=lambda value: value["pid"] == process.pid):
                if _signal_exact_process(row, signal.SIGKILL):
                    kill_signals.append(dict(row))
            while time.monotonic() < deadline:
                process.poll()
                live = _live_tracked_processes(process.pid, controller_state)
                if not live:
                    break
                time.sleep(PROCESS_CLEANUP_POLL_SECONDS)
        remaining = max(0.0, deadline - time.monotonic())
        if process.poll() is None:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
        survivors = _live_tracked_processes(process.pid, controller_state)
        tracked = _record_owned_process_tree(process.pid, controller_state)
    except BaseException as exc:
        tracked = [
            dict(row)
            for row in controller_state.get("owned_process_identities", {}).values()
            if isinstance(row, Mapping)
        ]
        survivors = tracked
        terminal_reason = {
            "code": str(getattr(exc, "code", type(exc).__name__)),
            "type": type(exc).__name__,
            "message": str(exc)[:2048] or "process cleanup failed",
        }
    leader_identity = next(
        (row for row in tracked if row.get("pid") == process.pid), None
    )
    current_leader = _read_process_identity(process.pid)
    leader_absent = (
        leader_identity is None
        or current_leader is None
        or not _same_process_identity(leader_identity, current_leader)
    )
    descendant_survivors = [row for row in survivors if row.get("pid") != process.pid]
    session_ids = set(
        int(value) for value in controller_state.get("owned_session_ids", [])
    )
    try:
        session_survivors = [
            row
            for row in _snapshot_process_table().values()
            if row["session_id"] in session_ids
            and any(_same_process_identity(row, tracked_row) for tracked_row in tracked)
        ]
    except BaseException as exc:
        session_survivors = list(survivors)
        if terminal_reason is None:
            terminal_reason = {
                "code": str(getattr(exc, "code", type(exc).__name__)),
                "type": type(exc).__name__,
                "message": str(exc)[:2048] or "session verification failed",
            }
    descendants_absent = not descendant_survivors
    sessions_absent = not session_survivors
    passed = (
        terminal_reason is None
        and leader_absent
        and descendants_absent
        and sessions_absent
        and not survivors
    )
    if not passed and terminal_reason is None:
        terminal_reason = {
            "code": "wave3.process_cleanup_incomplete",
            "type": "Wave3ProbeError",
            "message": "worker leader, descendant, or session remains after cleanup",
        }
    return {
        "schema": PROCESS_CLEANUP_SCHEMA,
        "status": "passed" if passed else "failed",
        "reason": str(reason),
        "leader_pid": process.pid,
        "cleanup_timeout_seconds": PROCESS_CLEANUP_TIMEOUT_SECONDS,
        "tracked_processes": sorted(tracked, key=lambda row: row["pid"]),
        "term_signals": term_signals,
        "kill_signals": kill_signals,
        "leader_absent": leader_absent,
        "descendants_absent": descendants_absent,
        "sessions_absent": sessions_absent,
        "surviving_processes": sorted(survivors, key=lambda row: row["pid"]),
        "completed_monotonic_ns": time.monotonic_ns(),
        "terminal_reason": terminal_reason,
    }


def _terminate_worker(
    process: subprocess.Popen[Any],
    *,
    controller_state: dict[str, Any],
    reason: str,
) -> None:
    cleanup = _cleanup_worker_process_tree(
        process, controller_state=controller_state, reason=reason
    )
    controller_state["process_cleanup"] = cleanup


_TERMINAL_PROC_STATES = frozenset({"Z", "X", "x"})


def _read_watchdog_proc_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (PermissionError, OSError) as exc:
        raise Wave3ProbeError(
            "host RSS watchdog is unavailable", code="wave3.host_watchdog"
        ) from exc


def _proc_status_rss_bytes(status: str) -> int | None:
    for line in status.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) < 2 or not fields[1].isdigit():
            raise Wave3ProbeError(
                "host RSS watchdog is malformed",
                code="wave3.host_watchdog",
            )
        return int(fields[1]) * 1024
    return None


def _proc_status_state(status: str) -> str | None:
    for line in status.splitlines():
        if line.startswith("State:"):
            fields = line.split()
            return fields[1] if len(fields) >= 2 else None
    return None


def _rechecked_process_rss_bytes(pid: int) -> int | None:
    for attempt in range(3):
        status = _read_watchdog_proc_text(Path(f"/proc/{pid}/status"))
        if status is None:
            return None
        rss = _proc_status_rss_bytes(status)
        if rss is not None:
            return rss
        if _proc_status_state(status) in _TERMINAL_PROC_STATES:
            return None
        if attempt < 2:
            time.sleep(0.001)
    raise Wave3ProbeError("host RSS watchdog lacks VmRSS", code="wave3.host_watchdog")


def _process_tree_rss_bytes(root_pid: int) -> int:
    pending = [int(root_pid)]
    seen: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        status = _read_watchdog_proc_text(Path(f"/proc/{pid}/status"))
        if status is None:
            status = _read_watchdog_proc_text(Path(f"/proc/{pid}/status"))
            if status is None:
                continue
        children = _read_watchdog_proc_text(Path(f"/proc/{pid}/task/{pid}/children"))
        if children is None:
            rechecked_status = _read_watchdog_proc_text(Path(f"/proc/{pid}/status"))
            if rechecked_status is None or (
                _proc_status_rss_bytes(rechecked_status) is None
                and _proc_status_state(rechecked_status) in _TERMINAL_PROC_STATES
            ):
                continue
            raise Wave3ProbeError(
                "host RSS watchdog is unavailable", code="wave3.host_watchdog"
            )
        pending.extend(int(item) for item in children.split() if item.isdigit())
        rss = _proc_status_rss_bytes(status)
        if rss is None:
            rss = _rechecked_process_rss_bytes(pid)
            if rss is None:
                continue
        total += rss
    return total


def _gpu_memory_used_bytes(device: torch.device) -> int:
    selector = _cuda_device_selector(device)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                selector,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or len(rows) != 1:
            raise ValueError("malformed nvidia-smi output")
        memory_mib = int(rows[0])
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise Wave3ProbeError(
            "GPU watchdog is unavailable", code="wave3.gpu_watchdog"
        ) from exc
    if memory_mib < 0:
        raise Wave3ProbeError("GPU watchdog is invalid", code="wave3.gpu_watchdog")
    return memory_mib * 1024**2


def _consume_marker_if_present(
    marker_target: Path,
    *,
    plan: Mapping[str, Any] | None,
    evidence: dict[str, Any],
    publication_owner_token_sha256: str | None,
) -> None:
    if (
        plan is None
        or not _is_sha256(publication_owner_token_sha256)
        or not marker_target.is_file()
    ):
        return
    marker = validate_marker(_load_json(marker_target), expected_plan=plan)
    if marker["publication_owner_token_sha256"] != publication_owner_token_sha256:
        return
    evidence["attempt_marker"] = {
        "status": "published",
        "path": str(marker_target),
        "marker_sha256": marker["marker_sha256"],
    }
    runtime = evidence["runtime"]
    runtime["shared_gpu_contract"].update(
        {
            "status": "pre_marker_bound",
            "baseline": marker["shared_gpu_preflight"],
            "pre_marker_observation": marker["pre_marker_gpu_observation"],
        }
    )
    if runtime["status"] == "not_started":
        runtime.update(
            {
                "status": "cpu_ready",
                "memory_savers": {},
                "model_weight_identity": marker["model_weight_identity"],
                "runtime_config_attestation": marker["runtime_config_attestation"],
                "runtime_source_identities": marker["runtime_source_identities"],
                "concrete_trainable_inventory": marker["concrete_trainable_inventory"],
                "initial_buffer_inventory": marker["initial_buffer_inventory"],
            }
        )
    for phase in ("model_setup_cpu", "attempt_started"):
        if phase not in evidence["completed_phases"]:
            expected_index = PASSED_PHASE_ORDER.index(phase)
            evidence["completed_phases"] = list(
                PASSED_PHASE_ORDER[: expected_index + 1]
            )


def _bind_controller_receipt(
    receipt: Mapping[str, Any], controller_state: Mapping[str, Any]
) -> dict[str, Any]:
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    runtime = dict(_mapping(body["runtime"], "runtime"))
    shared_gpu_contract = dict(
        _mapping(runtime["shared_gpu_contract"], "runtime.shared_gpu_contract")
    )
    post_probe_observation = controller_state.get("post_probe_preterminal_observation")
    if post_probe_observation is not None:
        shared_gpu_contract["post_probe_preterminal_observation"] = (
            post_probe_observation
        )
        shared_gpu_contract["status"] = (
            "post_probe_verified"
            if post_probe_observation.get("status") == "passed"
            else "post_probe_failed"
        )
    runtime["shared_gpu_contract"] = shared_gpu_contract
    runtime["process_cleanup"] = controller_state.get(
        "process_cleanup", _empty_process_cleanup()
    )
    runtime["controller"] = {
        "status": controller_state["status"],
        "worker_pid": controller_state["worker_pid"],
        "termination": controller_state["termination"],
        "max_host_rss_bytes": int(controller_state["max_host_rss_bytes"]),
        "max_gpu_memory_bytes": int(controller_state["max_gpu_memory_bytes"]),
    }
    body["runtime"] = runtime
    return _finalize(body, hash_field="receipt_sha256")


def _publish_terminal_receipt(
    *,
    receipt_target: Path,
    publication_failure_target: Path,
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    try:
        publish_json_absent(receipt_target, receipt)
        return {"receipt_linked_and_reloaded": True, "post_link_recovery": False}
    except BaseException as exc:
        if (
            isinstance(exc, Wave3ArtifactPublicationError)
            and exc.linked_by_this_call
            and exc.reloaded_exact
        ):
            return {
                "receipt_linked_and_reloaded": True,
                "post_link_recovery": True,
                "publication_warning": {
                    "code": exc.code,
                    "type": type(exc.__cause__).__name__,
                    "message": str(exc.__cause__)[:2048]
                    if exc.__cause__ is not None
                    else str(exc)[:2048],
                },
            }
        if plan is None:
            raise Wave3ProbeError(
                "receipt publication failed before plan binding",
                code="wave3.receipt_publication",
            ) from exc
        sidecar = _finalize(
            {
                "schema": PUBLICATION_FAILURE_SCHEMA,
                "status": "receipt_publication_failed",
                "plan_sha256": plan["plan_sha256"],
                "receipt_target": str(receipt_target),
                "publication_failure_target": str(publication_failure_target),
                "attempt_marker": receipt["attempt_marker"],
                "source_identity": plan["source_identity"],
                "intended_receipt_sha256": receipt["receipt_sha256"],
                "terminal_reason": {
                    "code": str(getattr(exc, "code", type(exc).__name__)),
                    "type": type(exc).__name__,
                    "message": str(exc)[:2048] or "receipt publication failed",
                },
            },
            hash_field="publication_failure_sha256",
        )
        validate_publication_failure(sidecar, expected_plan=plan)
        try:
            publish_json_absent(publication_failure_target, sidecar)
        except Wave3ArtifactPublicationError as sidecar_exc:
            if not (sidecar_exc.linked_by_this_call and sidecar_exc.reloaded_exact):
                raise
        raise Wave3ProbeError(
            "receipt publication failed; immutable failure sidecar published",
            code="wave3.receipt_publication",
        ) from exc


def _execute_gpu_probe(
    plan: Mapping[str, Any],
    *,
    device: torch.device,
    evidence: dict[str, Any],
    run_started: float,
    marker_target: Path,
    publication_owner_token_sha256: str,
    worker_argv: list[str],
    shared_gpu_baseline: Mapping[str, Any],
    emit: Any,
) -> dict[str, Any]:
    _assert_deadline(run_started)
    if not _is_sha256(publication_owner_token_sha256):
        raise Wave3ProbeError(
            "publication owner token identity is invalid",
            code="wave3.controller_parent",
        )
    resolved = load_train_config(plan["config"]["entry_path"])
    if resolved.fingerprint != FROZEN_CONFIG_FINGERPRINT:
        raise Wave3ProbeError(
            "runtime config identity drifted", code="wave3.config_identity"
        )
    initial_runtime_config_attestation = attest_runtime_config(
        _mapping(plan["config"], "config"), resolved.config_dict
    )
    if resolved.config.training.precision != "bf16":
        raise Wave3ProbeError("probe requires production BF16", code="wave3.precision")
    _manifest, micro_step = _load_frozen_micro_step(Path(plan["cache"]["path"]))
    if _workload_identity(micro_step) != plan["workload"]:
        raise Wave3ProbeError(
            "runtime workload identity drifted", code="wave3.workload_drift"
        )
    current_model_weight_identity = _assert_current_model_weight_identity(
        plan["model_weight_identity"]
    )
    components = load_qwen_components(resolved.config, load_model=True)
    if components.model is None:
        raise Wave3ProbeError(
            "model loader returned no model", code="wave3.model_missing"
        )
    cpu_installation = {
        "model": _assert_model_cpu_resident(components.model),
    }
    adapter_plan = build_adapter_setup_plan(
        resolved.config.adapter,
        load_default_adapter_source_gate_evidence(REPO_ROOT),
        base_model_path=components.base_model_path,
    )
    adapter_result = setup_dora_adapter(components.model, adapter_plan)
    cpu_installation["adapter"] = _assert_model_cpu_resident(adapter_result.model)
    selection = build_default_special_token_selection(
        resolved.config.model.special_token_embeddings, components.token_identity
    )
    special_result = install_special_token_embedding_deltas(
        adapter_result.model,
        selection,
        source_gate=load_default_special_token_embedding_source_gate_evidence(
            REPO_ROOT
        ),
    )
    embedding_load_receipt: Mapping[str, Any] | None = None
    if adapter_plan.mode == "warm_start_expand_dora":
        payload = adapter_plan.repaired_embedding_payload_path
        if payload is None:
            raise Wave3ProbeError(
                "warm-start embedding payload is missing",
                code="wave3.embedding_payload",
            )
        embedding_load_receipt = load_special_token_embedding_deltas(
            special_result,
            payload,
            expected_base_model_path=components.base_model_path,
            expected_base_config_sha256=components.base_config_sha256,
            expected_tokenizer_sha256=components.tokenizer_sha256,
        ).to_artifact_dict()
    cpu_installation["embedding_delta"] = _assert_model_cpu_resident(
        special_result.model
    )
    model = special_result.model
    memory_savers = enable_training_memory_savers(model)
    inventory = _concrete_trainable_inventory(model)
    runtime_source_identities = _runtime_source_identities(
        components=components,
        adapter_plan=adapter_plan,
        adapter_result=adapter_result,
        special_result=special_result,
        embedding_load_receipt=embedding_load_receipt,
    )
    evidence["runtime"].update(
        {
            "status": "cpu_ready",
            "memory_savers": memory_savers,
            "model_weight_identity": current_model_weight_identity,
            "runtime_config_attestation": initial_runtime_config_attestation,
            "runtime_source_identities": runtime_source_identities,
            "concrete_trainable_inventory": inventory,
            "initial_buffer_inventory": _snapshot_named_buffers(model),
        }
    )
    evidence["completed_phases"].append("model_setup_cpu")
    _assert_deadline(run_started)
    resolved, runtime_config_attestation = (
        _attest_fresh_runtime_config_immediately_before_marker(
            plan,
            initial_resolved_config=resolved.config_dict,
        )
    )
    if runtime_config_attestation != initial_runtime_config_attestation:
        raise Wave3ProbeError(
            "runtime config attestation changed before marker",
            code="wave3.config_attestation",
        )
    _admit_exact_one_rank_accelerator_environment(device=device)
    validated_shared_gpu_baseline = _validate_shared_gpu_baseline(shared_gpu_baseline)
    pre_marker_observation = _assert_gpu_process_subset(
        device,
        validated_shared_gpu_baseline,
        stage="pre_marker",
    )
    evidence["runtime"]["shared_gpu_contract"].update(
        {
            "status": "pre_marker_bound",
            "baseline": validated_shared_gpu_baseline,
            "pre_marker_observation": pre_marker_observation,
        }
    )
    marker = _finalize(
        {
            "schema": MARKER_SCHEMA,
            "status": "attempt_started",
            "plan_sha256": plan["plan_sha256"],
            "model_weight_identity": current_model_weight_identity,
            "receipt_target": plan["artifact_targets"]["receipt"],
            "publication_failure_target": plan["artifact_targets"][
                "publication_failure"
            ],
            "requested_device": str(device),
            "command_identity": {
                "run_argv": plan["execution_contract"]["run_argv"],
                "run_argv_sha256": plan["execution_contract"]["run_argv_sha256"],
                "controller_argv": plan["execution_contract"]["controller_argv"],
                "controller_argv_sha256": plan["execution_contract"][
                    "controller_argv_sha256"
                ],
                "worker_argv": worker_argv,
                "worker_argv_sha256": sha256_json(worker_argv),
            },
            "publication_owner_token_sha256": publication_owner_token_sha256,
            "source_identity": plan["source_identity"],
            "runtime_config_attestation": runtime_config_attestation,
            "runtime_source_identities": runtime_source_identities,
            "cpu_installation": cpu_installation,
            "provenance_sha256": sha256_json(plan["provenance"]),
            "shared_gpu_preflight": validated_shared_gpu_baseline,
            "pre_marker_gpu_observation": pre_marker_observation,
            "claim_boundary": shared_gpu_claim_boundary(),
            "concrete_trainable_inventory": inventory,
            "initial_buffer_inventory": evidence["runtime"]["initial_buffer_inventory"],
            "published_monotonic_ns": time.monotonic_ns(),
        },
        hash_field="marker_sha256",
    )
    validate_marker(marker, expected_plan=plan)
    marker_reference = {
        "status": "published",
        "path": str(marker_target),
        "marker_sha256": marker["marker_sha256"],
    }

    def record_linked_marker() -> None:
        evidence["attempt_marker"] = marker_reference
        evidence["completed_phases"].append("attempt_started")

    publish_json_absent(marker_target, marker, on_linked=record_linked_marker)
    emit({"event": "progress", "evidence": evidence})
    accelerator = _build_exact_one_rank_accelerator(device=device)
    model.to(device)
    model = accelerator.prepare(model)
    model.train()
    if _concrete_trainable_inventory(model) != inventory:
        raise Wave3ProbeError(
            "trainable inventory changed after prepare", code="wave3.inventory"
        )
    initial_buffer_inventory = _snapshot_named_buffers(model)
    if initial_buffer_inventory != marker["initial_buffer_inventory"]:
        raise Wave3ProbeError(
            "named buffer inventory changed during GPU setup", code="wave3.buffers"
        )
    evidence["runtime"].update(
        {
            "status": "gpu_ready",
            "initial_buffer_inventory": initial_buffer_inventory,
        }
    )
    evidence["completed_phases"].append("model_setup_gpu")
    forward_inputs = _build_forward_inputs(micro_step, components, device=device)
    evidence["completed_phases"].append("input_construction")
    initial_values = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    initial_buffers = {
        name: buffer.detach().cpu().clone() for name, buffer in model.named_buffers()
    }
    rng = snapshot_rng()
    configured = LossRunner.from_config(resolved.config.losses)
    optimized = replace(configured, token_type_gate_weight=0.0)
    differentiable = configured
    plan_loss = optimized.prepare_planned_step((micro_step,))

    def restore_state() -> None:
        restore_rng(rng)
        model.train()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(
                        initial_values[name].to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    )
                    parameter.grad = None
            observed_buffer_names = [name for name, _buffer in model.named_buffers()]
            if observed_buffer_names != list(initial_buffers):
                raise Wave3ProbeError(
                    "named buffer inventory changed", code="wave3.buffers"
                )
            for name, buffer in model.named_buffers():
                buffer.copy_(
                    initial_buffers[name].to(device=buffer.device, dtype=buffer.dtype)
                )

    restore_state()
    oracle_before = _snapshot_named_buffers(model)
    emit({"event": "arm_started", "arm": ORACLE_ARM_NAME})
    oracle = _execute_oracle_arm(
        model=model,
        forward_inputs=forward_inputs,
        micro_step=micro_step,
        differentiable=differentiable,
        optimized=optimized,
        loss_plan=plan_loss,
    )
    oracle_after = _snapshot_named_buffers(model)
    restore_state()
    oracle_restored = _snapshot_named_buffers(model)
    oracle["status"] = "completed"
    oracle["buffers"] = _buffer_transition(
        before=oracle_before,
        after=oracle_after,
        restored=oracle_restored,
        initial=initial_buffer_inventory,
    )
    emit({"event": "arm_completed", "arm": ORACLE_ARM_NAME})
    evidence["oracle"] = oracle
    evidence["counts"]["model_forwards"] += 1
    evidence["completed_phases"].append("same_logits_oracle")
    emit({"event": "progress", "evidence": evidence})
    if not oracle["raw_diagnostic_comparison"]["passed"]:
        raise Wave3ProbeError(
            "same-logits raw diagnostic comparison failed", code="wave3.raw_diagnostic"
        )

    arms: dict[str, ArmExecution] = {}
    for name in ARM_ORDER:
        _assert_deadline(run_started)
        restore_state()
        buffers_before = _snapshot_named_buffers(model)
        mode = "zero" if name.startswith("zero_") else "nonzero"
        implementation = "reference" if name.endswith("reference") else "optimized"
        emit({"event": "arm_started", "arm": name})
        arms[name] = _execute_measured_arm(
            name=name,
            mode=mode,
            implementation=implementation,
            model=model,
            accelerator=accelerator,
            forward_inputs=forward_inputs,
            micro_step=micro_step,
            configured=configured,
            differentiable=differentiable,
            optimized=optimized,
            loss_plan=plan_loss,
            device=device,
            expected_inventory=inventory,
        )
        buffers_after = _snapshot_named_buffers(model)
        restore_state()
        buffers_restored = _snapshot_named_buffers(model)
        arms[name].artifact["buffers"] = _buffer_transition(
            before=buffers_before,
            after=buffers_after,
            restored=buffers_restored,
            initial=initial_buffer_inventory,
        )
        evidence["arms"][name] = {
            "status": "completed",
            "artifact": arms[name].artifact,
        }
        evidence["completed_arm_order"].append(name)
        evidence["counts"]["model_forwards"] += 1
        evidence["counts"]["backwards"] += 1
        evidence["completed_phases"].append(f"arm:{name}")
        emit({"event": "arm_completed", "arm": name})
        emit({"event": "progress", "evidence": evidence})

    comparisons = {
        "raw_diagnostic_same_logits": oracle["raw_diagnostic_comparison"],
        "zero_reference_base_only_total": compare_scalar(
            arms["zero_reference"].artifact["total_loss"],
            arms["zero_reference"].artifact["base_only_total_loss"],
            rtol=LOSS_RTOL,
            atol=LOSS_ATOL,
        ),
        "zero_optimized_base_only_total": compare_scalar(
            arms["zero_optimized"].artifact["total_loss"],
            arms["zero_optimized"].artifact["base_only_total_loss"],
            rtol=LOSS_RTOL,
            atol=LOSS_ATOL,
        ),
        "zero_total": compare_scalar(
            float(arms["zero_reference"].total_loss.item()),
            float(arms["zero_optimized"].total_loss.item()),
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        ),
        "zero_gradients": compare_gradient_maps(
            arms["zero_reference"].gradients, arms["zero_optimized"].gradients
        ),
        "nonzero_total": compare_scalar(
            float(arms["nonzero_reference"].total_loss.item()),
            float(arms["nonzero_optimized"].total_loss.item()),
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        ),
        "nonzero_gradients": compare_gradient_maps(
            arms["nonzero_reference"].gradients,
            arms["nonzero_optimized"].gradients,
        ),
        "nonzero_control_graph": {
            "reference_differentiable": arms["nonzero_reference"].artifact[
                "nonzero_control_differentiable"
            ],
            "optimized_differentiable": arms["nonzero_optimized"].artifact[
                "nonzero_control_differentiable"
            ],
            "passed": bool(
                arms["nonzero_reference"].artifact["nonzero_control_differentiable"]
                and arms["nonzero_optimized"].artifact["nonzero_control_differentiable"]
            ),
        },
    }
    evidence["comparisons"] = comparisons
    if not all(bool(row["passed"]) for row in comparisons.values()):
        raise Wave3ProbeError(
            "one or more comparison bands failed", code="wave3.comparison"
        )
    efficiency = _efficiency_comparison(arms)
    evidence["efficiency"] = {"status": "completed", "result": efficiency}
    evidence["completed_phases"].append("comparisons")
    emit({"event": "progress", "evidence": evidence})
    _assert_deadline(run_started)
    receipt = _finalize(
        {
            "schema": RECEIPT_SCHEMA,
            "status": "passed",
            "plan_sha256": plan["plan_sha256"],
            "model_weight_identity": plan["model_weight_identity"],
            "plan_binding": _plan_binding(plan),
            "source_identity": plan["source_identity"],
            "terminal_reason": None,
            "attempt_marker": evidence["attempt_marker"],
            "completed_phases": list(evidence["completed_phases"]),
            "completed_arm_order": list(evidence["completed_arm_order"]),
            "counts": dict(evidence["counts"]),
            "oracle": oracle,
            "arms": dict(evidence["arms"]),
            "comparisons": comparisons,
            "efficiency": evidence["efficiency"],
            "runtime": evidence["runtime"],
        },
        hash_field="receipt_sha256",
    )
    return receipt


def _execute_oracle_arm(
    *,
    model: Any,
    forward_inputs: Any,
    micro_step: Any,
    differentiable: LossRunner,
    optimized: LossRunner,
    loss_plan: Any,
) -> dict[str, Any]:
    torch.cuda.synchronize(forward_inputs.input_ids.device)
    forward = run_qwen_forward(
        model,
        forward_inputs,
        expected_vocab_size=micro_step.expected_vocab_size,
        capture_fa2_branch=False,
        require_fa2_branch_proof=False,
    )
    context = LossContext(
        logits=forward.logits,
        token_sequence=micro_step.token_sequence,
        vocab_groups=micro_step.vocab_groups,
        logits_position_ids=forward.logits_position_ids,
    )
    reference = differentiable.compute_micro_step(
        context, loss_plan, local_micro_step_index=0
    )
    candidate = optimized.compute_micro_step(
        context, loss_plan, local_micro_step_index=0
    )
    if forward.logits.dtype != torch.float32:
        raise Wave3ProbeError(
            "strict oracle logits must be FP32", code="wave3.oracle_dtype"
        )
    assert_single_forward_same_logits(
        forward_count=1,
        backward_count=0,
        logits=forward.logits,
        diagnostic_logits=(context.logits, context.logits),
    )
    left = float(
        reference.term_by_name("token_type_gate").raw_loss.detach().float().item()
    )
    right = float(
        candidate.term_by_name("token_type_gate").raw_loss.detach().float().item()
    )
    reference_gate = reference.term_by_name("token_type_gate").raw_loss
    candidate_gate = candidate.term_by_name("token_type_gate").raw_loss
    if (
        not reference_gate.requires_grad
        or candidate_gate.requires_grad
        or candidate_gate.grad_fn is not None
    ):
        raise Wave3ProbeError(
            "same-logits graph attachment contract failed", code="wave3.oracle_graph"
        )
    return {
        "model_forward_count": 1,
        "backward_count": 0,
        "same_logits_object": True,
        "logits_dtype": str(forward.logits.dtype),
        "logits_sha256": tensor_sha256(forward.logits.detach().float().cpu()),
        "reference_requires_grad": bool(reference_gate.requires_grad),
        "optimized_requires_grad": bool(candidate_gate.requires_grad),
        "optimized_grad_fn": None,
        "raw_diagnostic_comparison": compare_scalar(
            left, right, rtol=LOSS_RTOL, atol=LOSS_ATOL
        ),
    }


def _execute_measured_arm(
    *,
    name: str,
    mode: str,
    implementation: str,
    model: Any,
    accelerator: Any,
    forward_inputs: Any,
    micro_step: Any,
    configured: LossRunner,
    differentiable: LossRunner,
    optimized: LossRunner,
    loss_plan: Any,
    device: torch.device,
    expected_inventory: Mapping[str, Any],
) -> ArmExecution:
    torch.cuda.reset_peak_memory_stats(device)
    resources_before = collect_resource_snapshot()
    saved_graph_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def save_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.grad_fn is not None:
            saved_graph_refs.append(weakref.ref(tensor))
        return tensor

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(save_tensor, lambda tensor: tensor):
        forward = run_qwen_forward(
            model,
            forward_inputs,
            expected_vocab_size=micro_step.expected_vocab_size,
            capture_fa2_branch=False,
            require_fa2_branch_proof=False,
        )
        context = LossContext(
            logits=forward.logits,
            token_sequence=micro_step.token_sequence,
            vocab_groups=micro_step.vocab_groups,
            logits_position_ids=forward.logits_position_ids,
        )
        if forward.logits.dtype != torch.float32:
            raise Wave3ProbeError(
                "measured logits must be graph-connected FP32",
                code="wave3.logits_dtype",
            )
        base_only_total: torch.Tensor
        if mode == "zero" and implementation == "reference":
            bundle = differentiable.compute_micro_step(
                context, loss_plan, local_micro_step_index=0
            )
            gate = bundle.term_by_name("token_type_gate")
            base_only_total = sum(
                (
                    term.weighted_loss
                    for term in bundle.terms
                    if term.name != "token_type_gate"
                ),
                gate.raw_loss.new_zeros(()),
            )
            objective = base_only_total + gate.raw_loss * 0.0
        elif mode == "zero":
            bundle = optimized.compute_micro_step(
                context, loss_plan, local_micro_step_index=0
            )
            gate = bundle.term_by_name("token_type_gate")
            base_only_total = sum(
                (
                    term.weighted_loss
                    for term in bundle.terms
                    if term.name != "token_type_gate"
                ),
                gate.raw_loss.new_zeros(()),
            )
            objective = bundle.total_loss
            if not torch.equal(objective.detach(), base_only_total.detach()):
                raise Wave3ProbeError(
                    "optimized zero objective is not base-only", code="wave3.base_only"
                )
        else:
            bundle = configured.compute_micro_step(
                context, loss_plan, local_micro_step_index=0
            )
            gate = bundle.term_by_name("token_type_gate")
            base_only_total = sum(
                (
                    term.weighted_loss
                    for term in bundle.terms
                    if term.name != "token_type_gate"
                ),
                gate.raw_loss.new_zeros(()),
            )
            objective = bundle.total_loss
        detached_total = objective.detach().float().cpu().clone()
        detached_base_only_total = base_only_total.detach().float().cpu().clone()
        if not torch.isfinite(detached_total) or not torch.isfinite(
            detached_base_only_total
        ):
            raise Wave3ProbeError(
                "arm loss is non-finite", code="wave3.non_finite_loss"
            )
        accelerator.backward(objective)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    assert_single_forward_same_logits(
        forward_count=1,
        backward_count=1,
        logits=forward.logits,
        diagnostic_logits=(context.logits,),
    )
    if elapsed > ARM_WALL_CEILING_SECONDS:
        raise Wave3ProbeError("arm exceeded 30-minute ceiling", code="wave3.arm_wall")
    gradients, gradient_rows = _snapshot_gradients(
        model, expected_inventory=expected_inventory
    )
    resources_after = collect_resource_snapshot()
    gpu = _mapping(resources_after.get("gpu", {}), "gpu")
    cpu = _mapping(resources_after.get("cpu", {}), "cpu")
    allocated = gpu.get("max_memory_allocated_bytes")
    reserved = gpu.get("max_memory_reserved_bytes")
    if not isinstance(allocated, int) or not isinstance(reserved, int):
        raise Wave3ProbeError(
            "CUDA resource counters are unavailable", code="wave3.device_memory"
        )
    if (
        allocated > DEVICE_MEMORY_CEILING_BYTES
        or reserved > DEVICE_MEMORY_CEILING_BYTES
    ):
        raise Wave3ProbeError(
            "arm exceeded 76 GiB CUDA ceiling", code="wave3.device_memory"
        )
    max_rss = cpu.get("max_rss_bytes")
    if not isinstance(max_rss, int) or max_rss > HOST_RSS_CEILING_BYTES:
        raise Wave3ProbeError(
            "arm exceeded 64 GiB host RSS ceiling", code="wave3.host_rss"
        )
    gate_requires_grad = bool(gate.raw_loss.requires_grad)
    gate_grad_fn = (
        None if gate.raw_loss.grad_fn is None else type(gate.raw_loss.grad_fn).__name__
    )
    diagnostic_ref = weakref.ref(gate.raw_loss)
    if mode == "zero" and implementation == "optimized" and gate_requires_grad:
        raise Wave3ProbeError(
            "optimized zero-weight diagnostic retained autograd",
            code="wave3.optimized_graph_residue",
        )
    if (mode == "nonzero" or implementation == "reference") and not gate_requires_grad:
        raise Wave3ProbeError(
            "differentiable diagnostic control lost autograd",
            code="wave3.nonzero_control",
        )
    del bundle, objective, gate, context, forward
    gc.collect()
    diagnostic_live = diagnostic_ref() is not None
    if mode == "zero" and implementation == "optimized" and diagnostic_live:
        raise Wave3ProbeError(
            "optimized zero-weight diagnostic survived graph release",
            code="wave3.optimized_graph_residue",
        )
    graph_residue = {
        "saved_graph_tensor_count": len(saved_graph_refs),
        "live_saved_graph_tensors_after_backward_and_release": sum(
            ref() is not None for ref in saved_graph_refs
        ),
        "diagnostic_tensor_live_after_release": diagnostic_live,
        "optimized_zero_diagnostic_released": bool(
            mode == "zero" and implementation == "optimized" and not diagnostic_live
        ),
    }
    artifact = {
        "name": name,
        "mode": mode,
        "implementation": implementation,
        "model_forward_count": 1,
        "backward_count": 1,
        "wall_seconds": elapsed,
        "total_loss": float(detached_total.item()),
        "base_only_total_loss": float(detached_base_only_total.item()),
        "diagnostic_requires_grad": gate_requires_grad,
        "diagnostic_grad_fn": gate_grad_fn,
        "nonzero_control_differentiable": bool(
            mode == "nonzero" and gate_requires_grad
        ),
        "graph_residue": graph_residue,
        "resources_before": resources_before,
        "resources_after": resources_after,
        "gradients": gradient_rows,
    }
    return ArmExecution(
        artifact=artifact, total_loss=detached_total, gradients=gradients
    )


def _snapshot_gradients(
    model: Any, *, expected_inventory: Mapping[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    gradients: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    aggregate_nonzero = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise Wave3ProbeError(
                "trainable gradient is missing or non-finite", code="wave3.gradient"
            )
        detached = gradient.detach().float().cpu().clone()
        aggregate_nonzero = aggregate_nonzero or bool(
            torch.count_nonzero(detached).item()
        )
        gradients[name] = detached
        rows.append(
            {
                "name": name,
                "shape": list(detached.shape),
                "storage_dtype": str(parameter.dtype),
                "gradient_dtype": str(gradient.dtype),
                "comparison_dtype": str(detached.dtype),
                "numel": int(detached.numel()),
                "finite": True,
                "sha256": tensor_sha256(detached),
                "l2_norm": float(torch.linalg.vector_norm(detached.double()).item()),
            }
        )
    if len(rows) != EXPECTED_TRAINABLE_COUNT or not aggregate_nonzero:
        raise Wave3ProbeError(
            "gradient inventory is incomplete or all zero", code="wave3.gradient"
        )
    rows.sort(key=lambda row: str(row["name"]))
    expected = [
        {
            "name": row.get("name"),
            "shape": row.get("shape"),
            "dtype": row.get("parameter_storage_dtype"),
        }
        for row in expected_inventory["parameters"]
    ]
    observed = [
        {
            "name": row["name"],
            "shape": row["shape"],
            "dtype": row["storage_dtype"],
        }
        for row in rows
    ]
    if observed != expected:
        raise Wave3ProbeError(
            "gradient inventory differs from the model", code="wave3.gradient"
        )
    snapshot = {
        "count": len(rows),
        "rows": rows,
        "rows_sha256": sha256_json(rows),
        "aggregate_nonzero": aggregate_nonzero,
    }
    _validate_gradient_snapshot(snapshot)
    return gradients, snapshot


def _concrete_trainable_inventory(model: Any) -> dict[str, Any]:
    try:
        return concrete_trainable_inventory(model)
    except Exception as exc:
        raise Wave3ProbeError(
            "trainable inventory is not the exact classified 589 surface",
            code="wave3.inventory",
        ) from exc


def _snapshot_named_buffers(model: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, buffer in model.named_buffers():
        detached = buffer.detach().cpu().contiguous()
        finite = bool(torch.isfinite(detached).all())
        if not finite:
            raise Wave3ProbeError(
                f"named buffer is non-finite: {name}", code="wave3.buffers"
            )
        rows.append(
            {
                "name": str(name),
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "numel": int(detached.numel()),
                "finite": finite,
                "sha256": tensor_sha256(detached),
            }
        )
    rows.sort(key=lambda row: str(row["name"]))
    body = {"count": len(rows), "rows": rows}
    inventory = {**body, "inventory_sha256": sha256_json(body)}
    return _validate_buffer_inventory(inventory)


def _assert_model_cpu_resident(model: Any) -> dict[str, Any]:
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    if any(tensor.device.type != "cpu" for tensor in (*parameters, *buffers)):
        raise Wave3ProbeError(
            "model/adapter/delta installation left CPU before marker publication",
            code="wave3.cpu_installation",
        )
    receipt = {
        "parameter_count": len(parameters),
        "buffer_count": len(buffers),
        "all_cpu": True,
    }
    _validate_cpu_installation(
        {"model": receipt, "adapter": receipt, "embedding_delta": receipt}
    )
    return receipt


def _buffer_transition(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    restored: Mapping[str, Any],
    initial: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "before": dict(before),
        "after": dict(after),
        "restored": dict(restored),
        "after_matches_before": (
            after["inventory_sha256"] == before["inventory_sha256"]
        ),
        "restoration_matches_initial": (
            restored["inventory_sha256"] == initial["inventory_sha256"]
            and before["inventory_sha256"] == initial["inventory_sha256"]
        ),
    }
    if result["restoration_matches_initial"] is not True:
        raise Wave3ProbeError(
            "named buffers were not restored exactly", code="wave3.buffers"
        )
    return _validate_buffer_transition(result)


def _runtime_source_identities(
    *,
    components: Any,
    adapter_plan: Any,
    adapter_result: Any,
    special_result: Any,
    embedding_load_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    model_artifact = components.to_artifact_dict()
    adapter_artifact = {
        "plan": adapter_plan.to_artifact_dict(),
        "setup_receipt": adapter_result.receipt.to_artifact_dict(),
    }
    embedding_artifact = {
        "install_receipt": special_result.receipt.to_artifact_dict(),
        "load_receipt": None
        if embedding_load_receipt is None
        else dict(embedding_load_receipt),
    }
    identities = {
        owner: {"artifact": artifact, "sha256": sha256_json(artifact)}
        for owner, artifact in (
            ("model", model_artifact),
            ("adapter", adapter_artifact),
            ("embedding_delta", embedding_artifact),
        )
    }
    return _validate_runtime_source_identities(identities)


def _build_forward_inputs(
    micro_step: Any, components: Any, *, device: torch.device
) -> Any:
    processor = getattr(components.processor, "image_processor", None)
    if processor is None:
        raise Wave3ProbeError(
            "runtime image processor is missing", code="wave3.image_processor"
        )
    attached = []
    for encoded in micro_step.encoded_examples:
        image_encoding = getattr(encoded, "image_encoding", None)
        if not isinstance(image_encoding, QwenImageEncoding):
            raise Wave3ProbeError(
                "cached image encoding is invalid", code="wave3.image_encoding"
            )
        before = encoded.to_artifact_dict()
        updated = replace(
            encoded,
            image_encoding=attach_qwen_image_processor(image_encoding, processor),
        )
        if updated.to_artifact_dict() != before:
            raise Wave3ProbeError(
                "processor attachment changed semantics", code="wave3.image_attachment"
            )
        attached.append(updated)
    logits_positions = tuple(
        sorted(
            {
                int(atom.causal_logits_position)
                for atom in micro_step.token_sequence.atoms
            }
        )
    )
    return build_qwen_forward_inputs(
        micro_step.pack,
        tuple(attached),
        micro_step.position_inputs,
        logits_to_keep_positions=logits_positions,
        device=device,
        fa2_branch_proof_policy="disabled",
    )


def _load_frozen_micro_step(cache_dir: Path) -> tuple[dict[str, Any], Any]:
    manifest_path = cache_dir / "manifest.json"
    if sha256_file(manifest_path) != W0_MANIFEST_SHA256:
        raise Wave3ProbeError(
            "Wave 0 manifest identity drifted", code="wave3.cache_identity"
        )
    manifest = _load_json(manifest_path)
    if (
        manifest.get("version") != W0_CACHE_VERSION
        or manifest.get("fingerprint") != W0_CACHE_FINGERPRINT
        or manifest.get("status") != "complete"
        or manifest.get("micro_step_count") != 32
    ):
        raise Wave3ProbeError(
            "Wave 0 manifest contract drifted", code="wave3.cache_identity"
        )
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 1:
        raise Wave3ProbeError("Wave 0 chunk plan drifted", code="wave3.cache_identity")
    start, steps = pack_cache_module._load_validated_chunk(cache_dir, chunks[0])
    ordinal = W0_FIRST_MEASURED_PACK_ORDINAL
    if not (start <= ordinal < start + len(steps)):
        raise Wave3ProbeError(
            "measured pack is outside authenticated chunk", code="wave3.selection"
        )
    micro_step = steps[ordinal - start]
    if int(micro_step.pack.pack_index) != ordinal:
        raise Wave3ProbeError("measured pack ordinal drifted", code="wave3.selection")
    return manifest, micro_step


def _workload_identity(micro_step: Any) -> dict[str, Any]:
    pack = micro_step.pack
    input_payload = {
        "pack": pack.to_artifact_dict(),
        "input_ids_sha256": _int_sequence_sha256(pack.input_ids),
        "encoded_examples": [
            item.to_artifact_dict() for item in micro_step.encoded_examples
        ],
        "position_inputs": micro_step.position_inputs.to_artifact_dict(),
        "position_ids_sha256": tensor_sha256(micro_step.position_inputs.position_ids),
        "metadata": dict(micro_step.metadata or {}),
        "expected_vocab_size": int(micro_step.expected_vocab_size),
    }
    supervision_payload = {
        "token_sequence": micro_step.token_sequence.to_artifact_dict(),
        "input_ids_sha256": _int_sequence_sha256(micro_step.token_sequence.input_ids),
        "vocab_groups": micro_step.vocab_groups.to_artifact_dict(),
        "vocab_membership_sha256": sha256_json(
            {
                name: list(micro_step.vocab_groups.allowed_ids(name))
                for name in ("desc_text", "schema", "coordinate", "eos")
            }
        ),
    }
    return {
        "selection": dict(W0_FIRST_MEASURED_SELECTION),
        "pack_index": int(pack.pack_index),
        "example_ids": [segment.example_id for segment in pack.segments],
        "segment_bounds": [
            [int(segment.start), int(segment.end)] for segment in pack.segments
        ],
        "segment_boundaries": list(micro_step.position_inputs.segment_boundaries),
        "pack_length": int(pack.length),
        "input_sha256": sha256_json(input_payload),
        "supervision_sha256": sha256_json(supervision_payload),
    }


def _failure_receipt(
    plan: Mapping[str, Any] | None, *, evidence: Mapping[str, Any], exc: BaseException
) -> dict[str, Any]:
    return _finalize(
        {
            "schema": RECEIPT_SCHEMA,
            "status": "failed",
            "plan_sha256": None if plan is None else plan.get("plan_sha256"),
            "model_weight_identity": (
                None if plan is None else dict(plan["model_weight_identity"])
            ),
            "plan_binding": _plan_binding(plan),
            "source_identity": {} if plan is None else dict(plan["source_identity"]),
            "terminal_reason": {
                "code": str(getattr(exc, "code", type(exc).__name__)),
                "type": type(exc).__name__,
                "message": str(exc)[:2048],
            },
            "attempt_marker": dict(evidence["attempt_marker"]),
            "completed_phases": list(evidence.get("completed_phases", [])),
            "completed_arm_order": list(evidence.get("completed_arm_order", [])),
            "counts": dict(
                evidence.get("counts", {"model_forwards": 0, "backwards": 0})
            ),
            "oracle": dict(evidence["oracle"]),
            "arms": dict(evidence["arms"]),
            "comparisons": dict(evidence["comparisons"]),
            "efficiency": dict(evidence["efficiency"]),
            "runtime": dict(evidence["runtime"]),
        },
        hash_field="receipt_sha256",
    )


def _empty_oracle() -> dict[str, Any]:
    return {
        "status": "not_run",
        "model_forward_count": 0,
        "backward_count": 0,
        "same_logits_object": None,
        "logits_dtype": None,
        "logits_sha256": None,
        "reference_requires_grad": None,
        "optimized_requires_grad": None,
        "optimized_grad_fn": None,
        "raw_diagnostic_comparison": None,
        "buffers": None,
    }


def _resource_limits() -> dict[str, Any]:
    return {
        "host_rss_ceiling_bytes": HOST_RSS_CEILING_BYTES,
        "device_memory_ceiling_bytes": DEVICE_MEMORY_CEILING_BYTES,
        "arm_wall_ceiling_seconds": ARM_WALL_CEILING_SECONDS,
        "run_wall_ceiling_seconds": RUN_WALL_CEILING_SECONDS,
        "controller_poll_seconds": CONTROLLER_POLL_SECONDS,
        "retry": False,
    }


def _empty_runtime(requested_device: str) -> dict[str, Any]:
    return {
        "status": "not_started",
        "requested_device": _parse_cuda_device_text(requested_device),
        "precision": "bf16",
        "memory_savers": None,
        "cache_access": "read_only_no_writes",
        "model_weight_identity": None,
        "runtime_config_attestation": None,
        "runtime_source_identities": None,
        "concrete_trainable_inventory": None,
        "initial_buffer_inventory": None,
        "resource_limits": _resource_limits(),
        "shared_gpu_contract": {
            "status": "not_started",
            "claim_boundary": shared_gpu_claim_boundary(),
            "baseline_process_action": "observe_only_never_signal_or_reset",
            "baseline": None,
            "pre_marker_observation": None,
            "post_probe_preterminal_observation": None,
        },
        "process_cleanup": _empty_process_cleanup(),
        "controller": {
            "status": "not_started",
            "worker_pid": None,
            "termination": None,
            "max_host_rss_bytes": 0,
            "max_gpu_memory_bytes": 0,
        },
    }


def _empty_evidence(
    *, requested_device: str, marker_path: Path, plan: Mapping[str, Any] | None
) -> dict[str, Any]:
    return {
        "requested_device": _parse_cuda_device_text(requested_device),
        "attempt_marker": {
            "status": "not_published",
            "path": str(marker_path.resolve()),
            "marker_sha256": None,
        },
        "completed_phases": [],
        "completed_arm_order": [],
        "counts": {"model_forwards": 0, "backwards": 0},
        "oracle": _empty_oracle(),
        "arms": {name: {"status": "not_run", "artifact": None} for name in ARM_ORDER},
        "comparisons": {name: None for name in COMPARISON_ORDER},
        "efficiency": {"status": "not_run", "result": None},
        "runtime": _empty_runtime(requested_device),
        "plan_sha256": None if plan is None else plan.get("plan_sha256"),
    }


def _plan_binding(plan: Mapping[str, Any] | None) -> dict[str, Any]:
    if plan is None:
        return {
            "plan_sha256": None,
            "source_identity_sha256": None,
            "prepare_argv_sha256": None,
            "run_argv_sha256": None,
            "controller_argv_sha256": None,
            "model_weight_identity_sha256": None,
        }
    execution = _mapping(plan["execution_contract"], "execution_contract")
    return {
        "plan_sha256": plan["plan_sha256"],
        "source_identity_sha256": sha256_json(plan["source_identity"]),
        "prepare_argv_sha256": execution["prepare_argv_sha256"],
        "run_argv_sha256": execution["run_argv_sha256"],
        "controller_argv_sha256": execution["controller_argv_sha256"],
        "model_weight_identity_sha256": plan["model_weight_identity"][
            "aggregate_sha256"
        ],
    }


def _parse_cuda_device_text(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("cuda:"):
        raise Wave3ProbeError(
            "Wave 3 run requires explicit cuda:<index>", code="wave3.device_preflight"
        )
    suffix = value.removeprefix("cuda:")
    if not suffix.isdigit():
        raise Wave3ProbeError(
            "Wave 3 run requires explicit cuda:<index>", code="wave3.device_preflight"
        )
    return f"cuda:{int(suffix)}"


def _parse_cuda_device(value: str) -> torch.device:
    normalized = _parse_cuda_device_text(value)
    if not torch.cuda.is_available():
        raise Wave3ProbeError("CUDA is unavailable", code="wave3.device_preflight")
    device = torch.device(normalized)
    if device.index is None or device.index >= torch.cuda.device_count():
        raise Wave3ProbeError(
            "requested CUDA device is unavailable", code="wave3.device_preflight"
        )
    return device


def _query_shared_gpu_state(device: torch.device) -> dict[str, Any]:
    selector = _cuda_device_selector(device)
    try:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                selector,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        process_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
                "-i",
                selector,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Wave3ProbeError(
            "shared GPU preflight is unavailable", code="wave3.gpu_preflight"
        ) from exc
    gpu_rows = [line.strip() for line in gpu_result.stdout.splitlines() if line.strip()]
    if gpu_result.returncode != 0 or len(gpu_rows) != 1:
        raise Wave3ProbeError(
            "shared GPU preflight is malformed", code="wave3.gpu_preflight"
        )
    fields = [item.strip() for item in gpu_rows[0].split(",")]
    if len(fields) != 5:
        raise Wave3ProbeError(
            "shared GPU preflight is malformed", code="wave3.gpu_preflight"
        )
    try:
        physical_index = int(fields[0])
        total_memory_mib = int(fields[2])
        memory_used_mib = int(fields[3])
        utilization_percent = int(fields[4])
    except ValueError as exc:
        raise Wave3ProbeError(
            "shared GPU preflight is malformed", code="wave3.gpu_preflight"
        ) from exc
    gpu_uuid = fields[1]
    if (
        physical_index < 0
        or not gpu_uuid
        or total_memory_mib <= 0
        or memory_used_mib < 0
        or memory_used_mib > total_memory_mib
        or utilization_percent < 0
        or utilization_percent > 100
        or process_result.returncode != 0
    ):
        raise Wave3ProbeError(
            "shared GPU preflight is malformed", code="wave3.gpu_preflight"
        )
    processes: list[dict[str, Any]] = []
    for text in process_result.stdout.splitlines():
        stripped = text.strip()
        if not stripped:
            continue
        process_fields = [item.strip() for item in stripped.split(",")]
        if len(process_fields) not in {2, 3}:
            raise Wave3ProbeError(
                "shared GPU process inventory is malformed",
                code="wave3.gpu_preflight",
            )
        try:
            driver_pid = int(process_fields[1])
        except ValueError as exc:
            raise Wave3ProbeError(
                "shared GPU process inventory is malformed",
                code="wave3.gpu_preflight",
            ) from exc
        if process_fields[0] != gpu_uuid or driver_pid <= 0:
            raise Wave3ProbeError(
                "shared GPU process inventory does not match selected GPU",
                code="wave3.gpu_preflight",
            )
        processes.append({"gpu_uuid": gpu_uuid, "driver_pid": driver_pid})
    processes.sort(key=lambda row: (row["gpu_uuid"], row["driver_pid"]))
    if len({(row["gpu_uuid"], row["driver_pid"]) for row in processes}) != len(
        processes
    ):
        raise Wave3ProbeError(
            "shared GPU process inventory contains duplicates",
            code="wave3.gpu_preflight",
        )
    return {
        "gpu_identity": {
            "physical_index": physical_index,
            "uuid": gpu_uuid,
            "total_memory_mib": total_memory_mib,
        },
        "memory_used_mib": memory_used_mib,
        "headroom_mib": total_memory_mib - memory_used_mib,
        "utilization_percent": utilization_percent,
        "compute_processes": processes,
    }


def _capture_shared_gpu_baseline(device: torch.device) -> dict[str, Any]:
    selector = _cuda_device_selector(device)
    samples: list[dict[str, Any]] = []
    identity: dict[str, Any] | None = None
    baseline_processes: list[dict[str, Any]] | None = None
    for sample_index in range(SHARED_GPU_SAMPLE_COUNT):
        state = _query_shared_gpu_state(device)
        if state["gpu_identity"]["total_memory_mib"] != SHARED_GPU_TOTAL_MEMORY_MIB:
            raise Wave3ProbeError(
                "selected GPU total memory is not the frozen 81920 MiB shape",
                code="wave3.gpu_baseline",
            )
        if (
            state["memory_used_mib"] > SHARED_GPU_MAX_PREEXISTING_MEMORY_MIB
            or state["headroom_mib"] < SHARED_GPU_REQUIRED_HEADROOM_MIB
        ):
            raise Wave3ProbeError(
                "selected GPU lacks the required shared-run headroom",
                code="wave3.gpu_busy",
            )
        if identity is None:
            identity = dict(state["gpu_identity"])
            baseline_processes = list(state["compute_processes"])
        elif (
            state["gpu_identity"] != identity
            or state["compute_processes"] != baseline_processes
        ):
            raise Wave3ProbeError(
                "selected GPU baseline identity changed during admission",
                code="wave3.gpu_baseline",
            )
        samples.append(
            {
                "sample_index": sample_index,
                "monotonic_ns": time.monotonic_ns(),
                "memory_used_mib": state["memory_used_mib"],
                "headroom_mib": state["headroom_mib"],
                "utilization_percent": state["utilization_percent"],
                "compute_processes": list(state["compute_processes"]),
            }
        )
        if sample_index + 1 < SHARED_GPU_SAMPLE_COUNT:
            time.sleep(SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS)
    if identity is None or baseline_processes is None:
        raise Wave3ProbeError(
            "shared GPU baseline has no samples", code="wave3.gpu_baseline"
        )
    baseline = {
        "schema": SHARED_GPU_BASELINE_SCHEMA,
        "mode": "shared_preexisting_compute",
        "requested_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selector": selector,
        "gpu_identity": identity,
        "limits": {
            "sample_count": SHARED_GPU_SAMPLE_COUNT,
            "minimum_sample_interval_seconds": SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS,
            "max_preexisting_memory_mib": SHARED_GPU_MAX_PREEXISTING_MEMORY_MIB,
            "required_headroom_mib": SHARED_GPU_REQUIRED_HEADROOM_MIB,
        },
        "utilization_disposition": ("observational_only_not_admission_or_promotion"),
        "baseline_compute_processes": baseline_processes,
        "samples": samples,
    }
    return _validate_shared_gpu_baseline(baseline)


def _assert_gpu_process_subset(
    device: torch.device, baseline: Mapping[str, Any], *, stage: str
) -> dict[str, Any]:
    validated = _validate_shared_gpu_baseline(baseline)
    state = _query_shared_gpu_state(device)
    if state["gpu_identity"] != validated["gpu_identity"]:
        raise Wave3ProbeError(
            "selected GPU identity changed after baseline",
            code="wave3.gpu_postcheck",
        )
    baseline_rows = validated["baseline_compute_processes"]
    current_rows = state["compute_processes"]
    baseline_keys = {(row["gpu_uuid"], row["driver_pid"]) for row in baseline_rows}
    current_keys = {(row["gpu_uuid"], row["driver_pid"]) for row in current_rows}
    missing = [
        row
        for row in baseline_rows
        if (row["gpu_uuid"], row["driver_pid"]) not in current_keys
    ]
    new = [
        row
        for row in current_rows
        if (row["gpu_uuid"], row["driver_pid"]) not in baseline_keys
    ]
    observation = {
        "schema": SHARED_GPU_OBSERVATION_SCHEMA,
        "stage": stage,
        "monotonic_ns": time.monotonic_ns(),
        "gpu_identity": state["gpu_identity"],
        "memory_used_mib": state["memory_used_mib"],
        "headroom_mib": state["headroom_mib"],
        "utilization_percent": state["utilization_percent"],
        "compute_processes": current_rows,
        "missing_baseline_processes": missing,
        "new_processes": new,
    }
    try:
        return _validate_shared_gpu_observation(
            observation, baseline=validated, expected_stage=stage
        )
    except Wave3ProbeError as exc:
        exc.evidence = observation
        raise


def _capture_gpu_process_subset_sweep(
    device: torch.device, baseline: Mapping[str, Any], *, stage: str
) -> dict[str, Any]:
    validated = _validate_shared_gpu_baseline(baseline)
    samples: list[dict[str, Any]] = []
    first_error: dict[str, str] | None = None
    for sample_index in range(SHARED_GPU_SAMPLE_COUNT):
        try:
            observation = _assert_gpu_process_subset(device, validated, stage=stage)
            sample = {
                "sample_index": sample_index,
                "monotonic_ns": observation["monotonic_ns"],
                "status": "passed",
                "observation": observation,
                "error": None,
            }
        except BaseException as exc:
            error = {
                "code": str(getattr(exc, "code", type(exc).__name__)),
                "type": type(exc).__name__,
                "message": str(exc)[:2048] or "GPU subset sweep failed",
            }
            if first_error is None:
                first_error = error
            observed = getattr(exc, "evidence", None)
            sample = {
                "sample_index": sample_index,
                "monotonic_ns": (
                    observed["monotonic_ns"]
                    if isinstance(observed, Mapping)
                    and _is_nonnegative_int(observed.get("monotonic_ns"))
                    else time.monotonic_ns()
                ),
                "status": "failed",
                "observation": dict(observed)
                if isinstance(observed, Mapping)
                else None,
                "error": error,
            }
        samples.append(sample)
        if sample_index + 1 < SHARED_GPU_SAMPLE_COUNT:
            time.sleep(SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS)
    sweep = {
        "schema": SHARED_GPU_SUBSET_SWEEP_SCHEMA,
        "stage": stage,
        "sample_count": SHARED_GPU_SAMPLE_COUNT,
        "minimum_sample_interval_seconds": (SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS),
        "status": "failed" if first_error is not None else "passed",
        "samples": samples,
        "terminal_reason": first_error,
    }
    return _validate_gpu_process_subset_sweep(
        sweep, baseline=validated, expected_stage=stage
    )


def _assert_gpu_idle(device: torch.device) -> dict[str, Any]:
    """Retired private-idle helper retained only as an internal compatibility alias."""

    return _capture_shared_gpu_baseline(device)


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
        raise Wave3ProbeError(
            "CUDA_VISIBLE_DEVICES does not map the requested device",
            code="wave3.cuda_visible_mapping",
        )
    return entries[device.index]


def _admit_exact_one_rank_accelerator_environment(*, device: torch.device) -> None:
    """Pin a fresh direct one-rank Accelerate device without touching CUDA."""

    try:
        from accelerate.state import AcceleratorState, PartialState
    except ImportError as exc:
        raise Wave3ProbeError(
            "Accelerate state APIs are unavailable",
            code="wave3.accelerator_state",
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
        raise Wave3ProbeError(
            "probe requires a fresh direct one-rank process",
            code="wave3.accelerator_not_fresh",
        )
    required_device = str(device)
    configured_device = os.environ.get("ACCELERATE_TORCH_DEVICE")
    if configured_device is not None and configured_device != required_device:
        raise Wave3ProbeError(
            "ACCELERATE_TORCH_DEVICE conflicts with the requested probe device",
            code="wave3.accelerator_device_override",
        )
    os.environ["ACCELERATE_TORCH_DEVICE"] = required_device


def _build_exact_one_rank_accelerator(*, device: torch.device) -> Any:
    torch.cuda.set_device(device)
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="bf16")
    _validate_exact_accelerator_runtime(
        accelerator,
        expected_device=device,
        expected_mixed_precision="bf16",
    )
    return accelerator


def _validate_exact_accelerator_runtime(
    accelerator: Any,
    *,
    expected_device: torch.device,
    expected_mixed_precision: str,
) -> dict[str, Any]:
    distributed_type = getattr(accelerator, "distributed_type", None)
    distributed_name = getattr(distributed_type, "name", str(distributed_type))
    try:
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
    except (AttributeError, TypeError, ValueError) as exc:
        raise Wave3ProbeError(
            "Accelerator runtime identity is incomplete",
            code="wave3.accelerator_identity",
        ) from exc
    expected_index = expected_device.index
    if (
        distributed_name != "NO"
        or identity["rank"] != 0
        or local_rank != 0
        or identity["world_size"] != 1
        or observed_device != expected_device
        or identity["cuda_current_device"] != expected_index
        or expected_mixed_precision != "bf16"
        or identity["mixed_precision"] != expected_mixed_precision
        or identity["native_amp"] is not True
        or identity["gradient_accumulation_steps"] != 1
        or scaler is not None
        or identity["accelerate_torch_device"] != str(expected_device)
    ):
        raise Wave3ProbeError(
            "Accelerator runtime differs from the frozen one-rank BF16 seam",
            code="wave3.accelerator_identity",
        )
    return identity


def _assert_deadline(started: float) -> None:
    elapsed = time.monotonic() - started
    if elapsed > RUN_WALL_CEILING_SECONDS:
        raise Wave3ProbeError(
            "probe exceeded total wall ceiling", code="wave3.run_wall"
        )


def _prepare_argv(
    *,
    config_path: str | Path,
    cache_dir: str | Path,
    plan_path: str | Path,
    receipt_path: str | Path,
    attempt_marker_path: str | Path,
    publication_failure_path: str | Path,
    device: str,
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "prepare",
        "--config",
        str(Path(config_path).expanduser().resolve()),
        "--cache-dir",
        str(Path(cache_dir).expanduser().resolve()),
        "--plan",
        str(Path(plan_path).expanduser().resolve()),
        "--receipt",
        str(Path(receipt_path).expanduser().resolve()),
        "--attempt-marker",
        str(Path(attempt_marker_path).expanduser().resolve()),
        "--publication-failure",
        str(Path(publication_failure_path).expanduser().resolve()),
        "--device",
        _parse_cuda_device_text(device),
    ]


def _artifact_argv(
    *,
    command: str,
    plan_path: str | Path,
    receipt_path: str | Path,
    attempt_marker_path: str | Path,
    publication_failure_path: str | Path,
    device: str,
) -> list[str]:
    if command not in {"run", "controller", "_worker"}:
        raise Wave3ProbeError("artifact command is invalid", code="wave3.argv")
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        command,
        "--plan",
        str(Path(plan_path).expanduser().resolve()),
        "--receipt",
        str(Path(receipt_path).expanduser().resolve()),
        "--attempt-marker",
        str(Path(attempt_marker_path).expanduser().resolve()),
        "--publication-failure",
        str(Path(publication_failure_path).expanduser().resolve()),
        "--device",
        _parse_cuda_device_text(device),
    ]


def _run_argv(args: argparse.Namespace) -> list[str]:
    return _artifact_argv(
        command="run",
        plan_path=args.plan,
        receipt_path=args.receipt,
        attempt_marker_path=args.attempt_marker,
        publication_failure_path=args.publication_failure,
        device=args.device,
    )


def _controller_argv(args: argparse.Namespace) -> list[str]:
    return _artifact_argv(
        command="controller",
        plan_path=args.plan,
        receipt_path=args.receipt,
        attempt_marker_path=args.attempt_marker,
        publication_failure_path=args.publication_failure,
        device=args.device,
    )


def _worker_argv(args: argparse.Namespace) -> list[str]:
    return _artifact_argv(
        command="_worker",
        plan_path=args.plan,
        receipt_path=args.receipt,
        attempt_marker_path=args.attempt_marker,
        publication_failure_path=args.publication_failure,
        device=args.device,
    )


def _assert_absent_target(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise Wave3ProbeError(
            f"artifact target already exists: {requested}",
            code="wave3.immutable_collision",
        )
    target = requested.resolve()
    if target.exists():
        raise Wave3ProbeError(
            f"artifact target already exists: {target}",
            code="wave3.immutable_collision",
        )
    return target


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve(strict=True)
    if target.stat().st_size > MAX_JSON_BYTES:
        raise Wave3ProbeError(
            "JSON input exceeds byte bound", code="wave3.artifact_size"
        )
    try:
        payload = load_strict_json(target)
    except ParityContractError as exc:
        raise Wave3ProbeError(
            "JSON artifact is not strict JSON", code="wave3.json"
        ) from exc
    if not isinstance(payload, dict):
        raise Wave3ProbeError("JSON artifact must be an object", code="wave3.json")
    return payload


def _mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Wave3ProbeError(f"{owner} must be a mapping", code="wave3.mapping")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], *, owner: str, code: str
) -> None:
    if set(value) != expected:
        raise Wave3ProbeError(
            f"{owner} fields are not exact",
            code=code,
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _float_exact(value: Any, expected: float) -> bool:
    return _is_finite_number(value) and float(value) == float(expected)


def _int_sequence_sha256(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, default=W0_CACHE_DIR)
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--receipt", type=Path, required=True)
    prepare.add_argument("--attempt-marker", type=Path, required=True)
    prepare.add_argument("--publication-failure", type=Path, required=True)
    prepare.add_argument("--device", required=True)
    prepare.set_defaults(func=prepare_command)
    for name, function in (
        ("run", run_command),
        ("controller", controller_command),
        ("_worker", worker_command),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
        command.add_argument("--attempt-marker", type=Path, required=True)
        command.add_argument("--publication-failure", type=Path, required=True)
        command.add_argument("--device", required=True)
        command.set_defaults(func=function)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Wave3ProbeError as exc:
        if args.command not in {"run", "controller", "_worker"}:
            raise
        print(
            json.dumps(
                {
                    "status": "failed",
                    "command": args.command,
                    "code": exc.code,
                    "type": type(exc).__name__,
                    "message": str(exc)[:2048],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare and execute the one-shot Wave 7 r7 exact-resume successor."""

from __future__ import annotations

import argparse
import ctypes
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qwen.parity import (  # noqa: E402
    assert_absent_artifact_target,
    base_model_weight_identity_with_execution_policy,
    canonical_json_bytes,
    write_strict_json_atomic,
)
from src.artifacts import provenance as execution_provenance  # noqa: E402
from src.config.loader import load_train_config  # noqa: E402
from src.config.paths import resolve_run_directory  # noqa: E402
from src.training.input_attestation import (  # noqa: E402
    CACHE_ATTESTATION_SCHEMA,
    MODEL_ATTESTATION_SCHEMA,
    validate_training_input_attestations,
)
from scripts.probes.coordexp_swift import (  # noqa: E402
    wave7_determinism_preflight as determinism_preflight,
)


REQUEST_SCHEMA = "coordexp-swift-wave7-exact-resume-sequence-request-v6"
PLAN_SCHEMA = "coordexp-swift-wave7-exact-resume-sequence-plan-v6"
MARKER_SCHEMA = "coordexp-swift-wave7-exact-resume-sequence-marker-v6"
RECEIPT_SCHEMA = "coordexp-swift-wave7-exact-resume-sequence-receipt-v6"
AMENDMENT_SCHEMA = "coordexp-swift-wave7-r7-amendment-v4"
AMENDMENT_EFFECTIVE_DATE = "2026-08-12"
AMENDMENT_AUTHORITY_PATH = (
    REPO_ROOT
    / "openspec/changes/harden-optimize-coordexp-swift-training-infrastructure/"
    "measurement-plan.md"
).resolve()
FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256 = (
    "6b088bc3ba7befcca26963b08ef493ce67a14dbfa190158b4ec270a410332730"
)
AMENDMENT_SECTION_ANCHOR = "### 2026-08-12: Wave 7 r7 one-shot successor authorization"
AMENDMENT_SCOPE = {
    "sequence": "one_fresh_r7_one_shot_successor",
    "predecessor": "immutable_r4_r5_and_r6_failures_historical_non_executable",
    "gpu_admission": "shared_preexisting_subset_v1",
    "claim_scope": "correctness_plumbing_numerical_artifact_interruption_exact_resume_only",
    "performance_promotion": False,
    "automatic_r8": False,
}
AMENDMENT_AUTHORITY_PHRASES = (
    "The user explicitly authorized exactly one fresh Wave 7 `r7` one-shot successor",
    "This is not an r6 retry",
    "no retry, root switch, or automatic Wave 7 `r8` successor",
)
PUBLICATION_FAILURE_SCHEMA = (
    "coordexp-swift-wave7-exact-resume-sequence-publication-failure-v1"
)
PRE_CHILD_RECEIPT_SCHEMA = "coordexp-swift-wave7-pre-child-gate-v1"
FINAL_RECEIPT_SCHEMA = "coordexp-swift-wave7-exact-resume-comparison-v2"
LEGACY_R4_RECEIPT_SCHEMA = "coordexp-swift-wave7-exact-resume-comparison-v1"
PREDECESSOR_R5_RECEIPT_SCHEMA = "coordexp-swift-wave7-exact-resume-sequence-receipt-v4"
PREDECESSOR_R5_MARKER_SCHEMA = "coordexp-swift-wave7-exact-resume-sequence-marker-v4"
DETERMINISM_PREFLIGHT_RECEIPT_SCHEMA = (
    "coordexp-swift-wave7-r5-determinism-preflight-v4"
)
DETERMINISM_PREFLIGHT_PLAN_SCHEMA = "coordexp-swift-wave7-determinism-preflight-plan-v4"
DETERMINISM_PREFLIGHT_MARKER_SCHEMA = (
    "coordexp-swift-wave7-determinism-preflight-attempt-start-marker-v4"
)
RUNTIME_ADMISSION_SCHEMA = "coordexp-swift-wave7-r5-runtime-admission-v1"
NATIVE_REFERENCE_SCHEMA = "coordexp-swift-wave8-native-runtime-attestation-v1"
FROZEN_R4_FAILURE_FILE_SHA256 = (
    "1d56e8139500ac09e62f57b2d2ce074401bc7596a64bee3b3915ff5036575e5b"
)
FROZEN_R4_FAILURE_PAYLOAD_SHA256 = (
    "13a39f43f1e504f48a2e70a8037d5ea7ccfd919003ffa1f89417e1b592973f4a"
)
FROZEN_R5_PLAN_FILE_SHA256 = (
    "ea3cef1d96412cd3f425e9c7f0c0cd562ceafa2fe59c39c225755898dbb66a1f"
)
FROZEN_R5_PLAN_PAYLOAD_SHA256 = (
    "73c502d69ec8b999a9e451620702b54eb1d573bf775e18d32edde08d216f53c4"
)
FROZEN_R5_MARKER_FILE_SHA256 = (
    "58aa7425dc6c019a23e790ba072495f0fb787b805eabaf922f079c35eddb19b5"
)
FROZEN_R5_MARKER_PAYLOAD_SHA256 = (
    "a620caa0e64a12be980120e310a914571c78ee0889d6da0760aae208f3c82224"
)
FROZEN_R5_FAILURE_FILE_SHA256 = (
    "d5244fb6d96b24b1cfac4438628ae141496f7357c286dff5454c19c9f39731b1"
)
FROZEN_R5_FAILURE_PAYLOAD_SHA256 = (
    "f560cbd7bfeac8e25f031021d9d0053c3df0d69961d52f1b854754ce4f4e7656"
)
FROZEN_R5_FAILURE_PATH = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r5/"
    "sequence-receipt.json"
)
FROZEN_R6_PREFLIGHT_ROOT = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r6/"
    "determinism-preflight"
).resolve()
FROZEN_R6_PREFLIGHT_PLAN_PATH = FROZEN_R6_PREFLIGHT_ROOT / "plan.json"
FROZEN_R6_PREFLIGHT_MARKER_PATH = FROZEN_R6_PREFLIGHT_ROOT / "attempt-start-marker.json"
FROZEN_R6_PREFLIGHT_FAILURE_PATH = FROZEN_R6_PREFLIGHT_ROOT / "terminal-receipt.json"
FROZEN_R6_PREFLIGHT_PLAN_FILE_SHA256 = (
    "2d5d5108c07e02ff964f5b3e6e4e42ce84e58bd0c17a8310dd7e2f3e6147e6ec"
)
FROZEN_R6_PREFLIGHT_PLAN_PAYLOAD_SHA256 = (
    "9b5c820c5f2c366c994a15732b99498b4072d1d552d9a7048c6afb65e0532a8f"
)
FROZEN_R6_PREFLIGHT_MARKER_FILE_SHA256 = (
    "f19a894ac3113aecc3a126dab7e3327389518c9e12e34d7806ad401adf239ce9"
)
FROZEN_R6_PREFLIGHT_MARKER_PAYLOAD_SHA256 = (
    "967fc87daabc19dd82d9d5a0f2567a2700da71bf9cb3773f904da63488585d7e"
)
FROZEN_R6_PREFLIGHT_FAILURE_FILE_SHA256 = (
    "eaf20c86a3fa95d7fa6533faaf7fb0e9b8fe938aa6f62d2ba725c3c604a64784"
)
FROZEN_R6_PREFLIGHT_FAILURE_PAYLOAD_SHA256 = (
    "4b400e094c494099b611e85d78b2156ce53b0fe372e12de6ff780c493b5770ab"
)
R7_SEQUENCE_ROOT = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-core-4"
).resolve()
R7_PRIVATE_CACHE_ROOT = (
    REPO_ROOT / "outputs/probes/coordexp_swift/private_v3_cache/2026-08-12-wave7-core-4"
).resolve()
R7_REQUEST_PATH = R7_SEQUENCE_ROOT / "request-v6.json"
R7_PLAN_PATH = R7_SEQUENCE_ROOT / "sequence-plan-v6.json"
MAX_AUTHORIZED_PHASE_SECONDS = 600.0
MAX_AUTHORIZED_WALL_SECONDS = 2400.0
MAX_AUTHORIZED_GPU_DEVICE_SECONDS = 14400.0
PRE_CHILD_DIGEST_PLACEHOLDER = "{PRE_CHILD_RECEIPT_PAYLOAD_SHA256}"
PHASE_ORDER = (
    "uninterrupted",
    "interrupted_parent",
    "pre_child",
    "verify_pre_child",
    "resume_child",
    "final_compare",
)
GPU_PHASES = ("uninterrupted", "interrupted_parent", "resume_child")
RUN_ROLES = ("uninterrupted", "interrupted_parent", "resume_child")
TARGET_NAMES = (
    "sequence_marker",
    "sequence_receipt",
    "publication_failure_sidecar",
    "interruption_marker",
    "interruption_receipt",
    "pre_child_receipt",
    "final_receipt",
)
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "FLASH_ATTENTION_DETERMINISTIC": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ARGV_ITEMS = 512
MAX_ARG_CHARS = 16_384
MAX_IDENTITY_FILES = 512
MAX_PROCESS_GRAPH = 4096
MAX_PHASE_OUTPUT_BYTES = 64 * 1024
GPU_DEVICE_COUNT = 8
COST_SECONDS_DECIMAL_PLACES = 9
SHARED_GPU_ADMISSION_MODE = "shared_preexisting_subset_v1"
SHARED_GPU_MEMORY_TOTAL_MIB = 81_920
SHARED_GPU_MEMORY_USED_CEILING_MIB = 49_152
SHARED_GPU_MEMORY_HEADROOM_FLOOR_MIB = 32_768
SHARED_GPU_BASELINE_STABILITY_SECONDS = 2.0
SHARED_GPU_POST_CLEANUP_STABILITY_SECONDS = 2.0
MAX_GPU_CSV_BYTES = 1024 * 1024
SEQUENCE_CLAIM_SCOPE = {
    "gpu_occupancy": "shared_preexisting_compute_processes_allowed",
    "establishes": [
        "correctness_evidence",
        "artifact_evidence",
        "exact_resume_correctness_evidence",
    ],
    "does_not_establish": [
        "throughput",
        "performance",
        "resource_capacity_or_promotion",
    ],
}
PYTHON_EXECUTABLE = Path(sys.executable).resolve()
_ACCELERATE_EXECUTABLE_TEXT = shutil.which("accelerate")
ACCELERATE_EXECUTABLE = (
    None
    if _ACCELERATE_EXECUTABLE_TEXT is None
    else Path(_ACCELERATE_EXECUTABLE_TEXT).resolve()
)
TRAIN_ENTRYPOINT = REPO_ROOT / "src/train.py"
REQUEST_PRODUCER_SOURCE = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_request.py"
)
INPUT_ATTESTATION_SOURCE = REPO_ROOT / "src/training/input_attestation.py"
MODEL_WEIGHT_IDENTITY_SOURCE = Path(
    base_model_weight_identity_with_execution_policy.__code__.co_filename
).resolve()
INTERRUPT_CONTROLLER = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_interrupt.py"
)
COMPARATOR_V2 = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_compare_v2.py"
)
TRAIN_PORTS = {
    "uninterrupted": 29681,
    "interrupted_parent": 29682,
    "resume_child": 29683,
}
FROZEN_V1_COMPARATOR_SHA256 = (
    "dfbb4d63c22d5c0db78c296514af1fe8fe24c28299175c90547b1a2917064034"
)
FROZEN_V2_COMPARATOR_SHA256 = (
    "2e3b6ea6d3cb7bb4bb43fc0d077b83e26b70d141a5401e3c1df6a7ee6f7b17bf"
)
FROZEN_DETERMINISM_PREFLIGHT_SOURCE_SHA256 = (
    "52d6e2a3b709ede73f5794eefdbfd6c57997714af29f5329e936b8953492346c"
)
FROZEN_PARITY_SOURCE_SHA256 = (
    "61d8460bb731d3243315b368a313e7f84d2008e415fdf3db17ed3834dad54deb"
)
FROZEN_INPUT_ATTESTATION_SOURCE_SHA256 = (
    "40202d0248a8929727204b38351ae742238eee6f5beb0ad5c2ff306425296508"
)
FROZEN_REQUEST_PRODUCER_SOURCE_SHA256 = (
    "540101ead19eeb82a7a2821954371a01e57e936f1484d8a0cf5130b330af71ac"
)
REQUIRED_SOURCE_PATHS = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_compare.py",
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_compare_v2.py",
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_interrupt.py",
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_determinism_preflight.py",
    REQUEST_PRODUCER_SOURCE,
    INPUT_ATTESTATION_SOURCE,
    MODEL_WEIGHT_IDENTITY_SOURCE,
    TRAIN_ENTRYPOINT,
)
DETERMINISM_CLAIM_SCOPE = {
    "classification": "plumbing_only",
    "establishes": [
        "two_fresh_eight_rank_launches_execute_the_fixed_synthetic_workload",
        "fixed_rank_digests_are_exactly_equal_across_launches",
        "active_cuda_native_mappings_match_the_pinned_admission",
    ],
    "does_not_establish": [
        "model_loading",
        "training_quality",
        "throughput",
        "performance_under_shared_gpu_occupancy",
        "resource_comparisons_under_shared_gpu_occupancy",
        "exact_resume_correctness",
    ],
}
_FORBIDDEN_SHELLS = frozenset(
    {"sh", "bash", "dash", "zsh", "fish", "csh", "tcsh", "pwsh", "powershell"}
)
_FORBIDDEN_CONTROL_TOKENS = frozenset(
    {"retry", "while", "until", "for", "&&", "||", ";"}
)
_TERMINAL_PROCESS_STATES = frozenset({"X", "x"})
_PR_SET_CHILD_SUBREAPER = 36
_PHASE_PROCESS_FIELDS = frozenset(
    {
        "pid",
        "pgid",
        "term_sent",
        "kill_sent",
        "reaped",
        "captured_process_graph",
        "remaining_pids",
        "remaining_pgids",
        "signal_events",
        "graph_overflow",
        "observed_process_count",
        "stdout_tail",
        "stdout_truncated",
        "stderr_tail",
        "stderr_truncated",
    }
)
_PROCESS_GRAPH_FIELDS = frozenset(
    {
        "pid",
        "parent_pid",
        "process_group_id",
        "session_id",
        "state",
        "start_time_ticks",
        "depth",
    }
)
_SIGNAL_EVENT_KINDS = frozenset(
    {
        "signal_group_sigterm",
        "signal_group_sigkill",
        "signal_pid_sigterm",
        "signal_pid_sigkill",
    }
)
MAX_SIGNAL_EVENTS = MAX_PROCESS_GRAPH * 4


class Wave7SequenceError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.context = dict(context or {})
        super().__init__(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise Wave7SequenceError(
            "identity file is unreadable",
            code="wave7_sequence.identity_drift",
            context={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    return digest.hexdigest()


def _strict_json_loads(encoded: bytes, *, owner: str) -> Any:
    if len(encoded) > MAX_JSON_BYTES:
        raise Wave7SequenceError(
            f"{owner} exceeds the JSON size bound",
            code="wave7_sequence.json_oversize",
            context={"owner": owner, "size": len(encoded)},
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant: {value}")

    try:
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise Wave7SequenceError(
            f"{owner} is not strict JSON",
            code="wave7_sequence.json_malformed",
            context={"owner": owner, "error_type": type(exc).__name__},
        ) from exc


def _strict_json_file(path: Path, *, owner: str) -> dict[str, Any]:
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise Wave7SequenceError(
            f"{owner} is unreadable",
            code="wave7_sequence.identity_drift",
            context={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    value = _strict_json_loads(encoded, owner=owner)
    if not isinstance(value, dict):
        raise Wave7SequenceError(
            f"{owner} must be one JSON object",
            code="wave7_sequence.json_schema",
        )
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, owner: str) -> None:
    observed = set(value)
    if observed != expected:
        raise Wave7SequenceError(
            f"{owner} fields are not exact",
            code="wave7_sequence.json_schema",
            context={
                "owner": owner,
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            },
        )


def _digest(value: Any, *, owner: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Wave7SequenceError(
            f"{owner} must be one lowercase sha256 digest",
            code="wave7_sequence.json_schema",
        )
    return value


def _positive_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Wave7SequenceError(
            f"{owner} must be a positive finite number",
            code="wave7_sequence.json_schema",
        )
    result = float(value)
    if not (result > 0.0 and result < 7 * 24 * 60 * 60):
        raise Wave7SequenceError(
            f"{owner} is outside the allowed bound",
            code="wave7_sequence.json_schema",
        )
    return result


def _freeze_cost_seconds(value: int | float) -> float:
    return round(float(value), COST_SECONDS_DECIMAL_PLACES)


def _validated_cost_seconds(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise Wave7SequenceError(
            f"{owner} must be one finite nonnegative number",
            code="wave7_sequence.terminal_schema",
        )
    result = _freeze_cost_seconds(value)
    if float(value) != result:
        raise Wave7SequenceError(
            f"{owner} exceeds the frozen receipt precision",
            code="wave7_sequence.terminal_schema",
        )
    return result


def _absolute_path(value: Any, *, owner: str, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise Wave7SequenceError(
            f"{owner} must be one absolute path",
            code="wave7_sequence.path",
        )
    requested = Path(value)
    if not requested.is_absolute():
        raise Wave7SequenceError(
            f"{owner} must be absolute",
            code="wave7_sequence.path",
            context={"path": value},
        )
    resolved = requested.resolve(strict=must_exist)
    if resolved != requested:
        raise Wave7SequenceError(
            f"{owner} must already be canonical",
            code="wave7_sequence.path",
            context={"path": value, "resolved": str(resolved)},
        )
    current = resolved if resolved.exists() else resolved.parent
    while current != current.parent:
        if current.is_symlink():
            raise Wave7SequenceError(
                f"{owner} must not traverse symlinks",
                code="wave7_sequence.path",
                context={"path": str(current)},
            )
        current = current.parent
    return resolved


def _file_identity(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
    except OSError as exc:
        raise Wave7SequenceError(
            "identity file cannot be stated",
            code="wave7_sequence.identity_drift",
            context={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise Wave7SequenceError(
            "identity path is not a regular file",
            code="wave7_sequence.identity_drift",
            context={"path": str(path)},
        )
    return {
        "path": str(path),
        "size": info.st_size,
        "sha256": _sha256_file(path),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _directory_identity(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
    except OSError as exc:
        raise Wave7SequenceError(
            "identity directory cannot be stated",
            code="wave7_sequence.identity_drift",
            context={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    if not stat.S_ISDIR(info.st_mode):
        raise Wave7SequenceError(
            "identity path is not a directory",
            code="wave7_sequence.identity_drift",
            context={"path": str(path)},
        )
    return {
        "path": str(path),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _verify_file_identity(identity: Mapping[str, Any]) -> None:
    _exact_fields(
        identity,
        {"path", "size", "sha256", "uid", "gid", "mode", "device", "inode"},
        owner="file identity",
    )
    path = _absolute_path(identity["path"], owner="file identity path", must_exist=True)
    observed = _file_identity(path)
    if observed != dict(identity):
        raise Wave7SequenceError(
            "file identity drifted",
            code="wave7_sequence.identity_drift",
            context={
                "path": str(path),
                "expected": dict(identity),
                "observed": observed,
            },
        )


def _verify_directory_identity(identity: Mapping[str, Any]) -> None:
    _exact_fields(
        identity,
        {"path", "uid", "gid", "mode", "device", "inode"},
        owner="directory identity",
    )
    path = _absolute_path(
        identity["path"], owner="directory identity path", must_exist=True
    )
    observed = _directory_identity(path)
    if observed != dict(identity):
        raise Wave7SequenceError(
            "directory ownership identity drifted",
            code="wave7_sequence.identity_drift",
            context={
                "path": str(path),
                "expected": dict(identity),
                "observed": observed,
            },
        )


def _verify_signed_payload(
    payload: Mapping[str, Any], *, expected_payload_sha256: str, owner: str
) -> None:
    observed = payload.get("receipt_payload_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_payload_sha256", None)
    recomputed = _sha256_bytes(canonical_json_bytes(unsigned))
    if observed != expected_payload_sha256 or recomputed != expected_payload_sha256:
        raise Wave7SequenceError(
            f"{owner} payload digest is invalid",
            code="wave7_sequence.identity_drift",
            context={
                "owner": owner,
                "expected": expected_payload_sha256,
                "observed": observed,
                "recomputed": recomputed,
            },
        )


def _verify_receipt_binding(
    value: Mapping[str, Any],
    *,
    owner: str,
    extra_fields: set[str] | None = None,
) -> dict[str, Any]:
    fields = {"path", "file_sha256", "payload_sha256", "schema", "status"}
    fields.update(extra_fields or set())
    _exact_fields(value, fields, owner=owner)
    path = _absolute_path(value["path"], owner=f"{owner} path", must_exist=True)
    file_sha256 = _digest(value["file_sha256"], owner=f"{owner} file sha256")
    payload_sha256 = _digest(value["payload_sha256"], owner=f"{owner} payload sha256")
    if _sha256_file(path) != file_sha256:
        raise Wave7SequenceError(
            f"{owner} file digest drifted",
            code="wave7_sequence.identity_drift",
            context={"path": str(path)},
        )
    payload = _strict_json_file(path, owner=owner)
    _verify_signed_payload(payload, expected_payload_sha256=payload_sha256, owner=owner)
    if (
        payload.get("schema") != value["schema"]
        or payload.get("status") != value["status"]
    ):
        raise Wave7SequenceError(
            f"{owner} schema or status drifted",
            code="wave7_sequence.identity_drift",
        )
    return payload


def _validate_predecessor_sequence_failure(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the immutable r5 receipt-v4 only as predecessor evidence."""

    try:
        _exact_fields(
            value,
            {"path", "file_sha256", "payload_sha256", "schema", "status"},
            owner="predecessor r5 failure binding",
        )
        if (
            value.get("path") != str(FROZEN_R5_FAILURE_PATH)
            or value.get("schema") != PREDECESSOR_R5_RECEIPT_SCHEMA
            or value.get("status") != "failed"
            or value.get("file_sha256") != FROZEN_R5_FAILURE_FILE_SHA256
            or value.get("payload_sha256") != FROZEN_R5_FAILURE_PAYLOAD_SHA256
        ):
            raise Wave7SequenceError(
                "predecessor binding is not the immutable r5 failure",
                code="wave7_sequence.predecessor_r5",
            )
        payload = _verify_receipt_binding(
            value, owner="predecessor r5 sequence failure"
        )
        _exact_fields(
            payload,
            {
                "schema",
                "status",
                "failure",
                "plan",
                "marker",
                "input_attestations",
                "phase_records",
                "pre_child_receipt",
                "final_receipt",
                "bounded_cleanup",
                "cost",
                "claim_scope",
                "final_gpu_recovery",
                "completed_at",
                "receipt_payload_sha256",
            },
            owner="predecessor r5 sequence failure",
        )
        expected_failure = {
            "phase": "uninterrupted",
            "code": "wave7_sequence.phase_failed",
            "message": "phase returned nonzero",
            "error_type": "Wave7SequenceError",
        }
        plan = payload.get("plan")
        marker = payload.get("marker")
        phases = payload.get("phase_records")
        if (
            payload.get("schema") != PREDECESSOR_R5_RECEIPT_SCHEMA
            or payload.get("status") != "failed"
            or payload.get("failure") != expected_failure
            or not isinstance(plan, dict)
            or set(plan) != {"path", "file_sha256", "payload_sha256"}
            or plan.get("file_sha256") != FROZEN_R5_PLAN_FILE_SHA256
            or plan.get("payload_sha256") != FROZEN_R5_PLAN_PAYLOAD_SHA256
            or not isinstance(marker, dict)
            or set(marker) != {"path", "file_sha256", "payload_sha256", "schema"}
            or marker.get("file_sha256") != FROZEN_R5_MARKER_FILE_SHA256
            or marker.get("payload_sha256") != FROZEN_R5_MARKER_PAYLOAD_SHA256
            or marker.get("schema") != PREDECESSOR_R5_MARKER_SCHEMA
            or not isinstance(phases, list)
            or len(phases) != len(PHASE_ORDER)
            or payload.get("pre_child_receipt") is not None
            or payload.get("final_receipt") is not None
            or payload.get("claim_scope") != SEQUENCE_CLAIM_SCOPE
            or not isinstance(payload.get("input_attestations"), dict)
            or not isinstance(payload.get("completed_at"), str)
            or not payload["completed_at"]
        ):
            raise Wave7SequenceError(
                "predecessor r5 failure projection is not exact",
                code="wave7_sequence.predecessor_r5",
            )

        first = phases[0]
        process = first.get("process") if isinstance(first, dict) else None
        if (
            not isinstance(first, dict)
            or first.get("phase") != "uninterrupted"
            or first.get("status") != "failed"
            or first.get("attempt_count") != 1
            or isinstance(first.get("attempt_count"), bool)
            or first.get("return_code") != 1
            or isinstance(first.get("return_code"), bool)
            or first.get("launched_command_sha256") != first.get("command_sha256")
            or not isinstance(process, dict)
            or process.get("reaped") is not True
            or process.get("remaining_pids") != []
            or process.get("remaining_pgids") != []
            or not isinstance(first.get("gpu_before_launch"), dict)
            or not isinstance(first.get("gpu_after_cleanup"), dict)
        ):
            raise Wave7SequenceError(
                "predecessor r5 first phase is not the exact failed attempt",
                code="wave7_sequence.predecessor_r5",
            )
        unattempted_fields = {
            "launched_command_sha256",
            "started_at",
            "completed_at",
            "return_code",
            "duration_seconds",
            "process",
            "gpu_before_launch",
            "gpu_after_cleanup",
        }
        for expected_phase, record in zip(PHASE_ORDER[1:], phases[1:], strict=True):
            if (
                not isinstance(record, dict)
                or record.get("phase") != expected_phase
                or record.get("status") != "not_started"
                or record.get("attempt_count") != 0
                or isinstance(record.get("attempt_count"), bool)
                or any(record.get(name) is not None for name in unattempted_fields)
            ):
                raise Wave7SequenceError(
                    "predecessor r5 later phase state is not unattempted",
                    code="wave7_sequence.predecessor_r5",
                )

        bounded_cleanup = payload.get("bounded_cleanup")
        if not isinstance(bounded_cleanup, dict) or bounded_cleanup != {
            "policy": "TERM_KILL_reap_started_process_group_only",
            "records": [{"phase": "uninterrupted", **process}],
            "artifacts_deleted": False,
        }:
            raise Wave7SequenceError(
                "predecessor r5 bounded cleanup projection is not exact",
                code="wave7_sequence.predecessor_r5",
            )
        if payload.get("cost") != {
            "wall_seconds": 51.002604775,
            "wall_seconds_cap": 2400,
            "gpu_device_count": 8,
            "gpu_device_seconds": 160.753348712,
            "gpu_device_seconds_cap": 14400,
        }:
            raise Wave7SequenceError(
                "predecessor r5 cost projection is not exact",
                code="wave7_sequence.predecessor_r5",
            )
        _validate_shared_gpu_recovery(
            payload.get("final_gpu_recovery"),
            expected_stability_seconds=SHARED_GPU_POST_CLEANUP_STABILITY_SECONDS,
            expected_phase="preterminal",
        )
    except Wave7SequenceError as exc:
        if exc.code == "wave7_sequence.predecessor_r5":
            raise
        raise Wave7SequenceError(
            "predecessor r5 evidence failed dedicated validation",
            code="wave7_sequence.predecessor_r5",
            context={"nested_code": exc.code},
        ) from exc
    return payload


def _validate_predecessor_preflight_failure(
    value: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Authenticate the consumed r6 preflight chain as historical evidence only."""

    try:
        _exact_fields(
            value,
            {
                "historical_non_executable",
                "plan",
                "attempt_marker",
                "terminal_receipt",
            },
            owner="predecessor r6 preflight failure",
        )
        if value.get("historical_non_executable") is not True:
            raise Wave7SequenceError(
                "r6 predecessor evidence must remain historical and non-executable",
                code="wave7_sequence.predecessor_r6_preflight",
            )
        plan_binding = value.get("plan")
        marker_binding = value.get("attempt_marker")
        terminal_binding = value.get("terminal_receipt")
        if not all(
            isinstance(binding, dict)
            for binding in (plan_binding, marker_binding, terminal_binding)
        ):
            raise Wave7SequenceError(
                "r6 predecessor chain bindings must be objects",
                code="wave7_sequence.predecessor_r6_preflight",
            )
        expected_bindings = (
            (
                plan_binding,
                FROZEN_R6_PREFLIGHT_PLAN_PATH,
                FROZEN_R6_PREFLIGHT_PLAN_FILE_SHA256,
                FROZEN_R6_PREFLIGHT_PLAN_PAYLOAD_SHA256,
                DETERMINISM_PREFLIGHT_PLAN_SCHEMA,
                "prepared",
            ),
            (
                marker_binding,
                FROZEN_R6_PREFLIGHT_MARKER_PATH,
                FROZEN_R6_PREFLIGHT_MARKER_FILE_SHA256,
                FROZEN_R6_PREFLIGHT_MARKER_PAYLOAD_SHA256,
                DETERMINISM_PREFLIGHT_MARKER_SCHEMA,
                "started",
            ),
            (
                terminal_binding,
                FROZEN_R6_PREFLIGHT_FAILURE_PATH,
                FROZEN_R6_PREFLIGHT_FAILURE_FILE_SHA256,
                FROZEN_R6_PREFLIGHT_FAILURE_PAYLOAD_SHA256,
                DETERMINISM_PREFLIGHT_RECEIPT_SCHEMA,
                "failed",
            ),
        )
        for (
            binding,
            path,
            file_sha256,
            payload_sha256,
            schema,
            status,
        ) in expected_bindings:
            if binding != {
                "path": str(path),
                "file_sha256": file_sha256,
                "payload_sha256": payload_sha256,
                "schema": schema,
                "status": status,
            }:
                raise Wave7SequenceError(
                    "r6 predecessor binding is not the immutable failed chain",
                    code="wave7_sequence.predecessor_r6_preflight",
                )

        plan = _verify_payload_binding(
            plan_binding,
            owner="predecessor r6 preflight plan",
            digest_field="plan_payload_sha256",
        )
        marker = _verify_receipt_binding(
            marker_binding, owner="predecessor r6 preflight marker"
        )
        terminal = _verify_receipt_binding(
            terminal_binding, owner="predecessor r6 preflight terminal"
        )
        if (
            plan.get("schema") != DETERMINISM_PREFLIGHT_PLAN_SCHEMA
            or plan.get("status") != "prepared"
            or marker.get("schema") != DETERMINISM_PREFLIGHT_MARKER_SCHEMA
            or marker.get("status") != "started"
            or marker.get("plan_sha256") != FROZEN_R6_PREFLIGHT_PLAN_PAYLOAD_SHA256
            or terminal.get("schema") != DETERMINISM_PREFLIGHT_RECEIPT_SCHEMA
            or terminal.get("status") != "failed"
            or terminal.get("world_size") != GPU_DEVICE_COUNT
            or terminal.get("launch_count") != 0
            or terminal.get("mismatches") != ["KeyboardInterrupt"]
            or terminal.get("plan_sha256") != FROZEN_R6_PREFLIGHT_PLAN_PAYLOAD_SHA256
            or terminal.get("attempt_marker") != dict(marker_binding)
            or terminal.get("processes") != []
            or terminal.get("rank_receipts") != []
            or terminal.get("comparisons") is not None
            or terminal.get("gpu_shared_occupancy_sweeps") != []
            or terminal.get("cleanup")
            != {"all_process_groups_exited": None, "bounded": True}
            or terminal.get("claim_scope") != DETERMINISM_CLAIM_SCOPE
        ):
            raise Wave7SequenceError(
                "r6 predecessor terminal is not the exact consumed preflight failure",
                code="wave7_sequence.predecessor_r6_preflight",
            )
    except Wave7SequenceError as exc:
        if exc.code == "wave7_sequence.predecessor_r6_preflight":
            raise
        raise Wave7SequenceError(
            "r6 predecessor evidence failed dedicated validation",
            code="wave7_sequence.predecessor_r6_preflight",
            context={"nested_code": exc.code},
        ) from exc
    return {
        "plan": plan,
        "attempt_marker": marker,
        "terminal_receipt": terminal,
    }


def _verify_payload_binding(
    value: Mapping[str, Any], *, owner: str, digest_field: str
) -> dict[str, Any]:
    _exact_fields(
        value,
        {"path", "file_sha256", "payload_sha256", "schema", "status"},
        owner=owner,
    )
    path = _absolute_path(value["path"], owner=f"{owner} path", must_exist=True)
    file_sha256 = _digest(value["file_sha256"], owner=f"{owner} file sha256")
    payload_sha256 = _digest(value["payload_sha256"], owner=f"{owner} payload sha256")
    if _sha256_file(path) != file_sha256:
        raise Wave7SequenceError(
            f"{owner} file digest drifted",
            code="wave7_sequence.identity_drift",
            context={"path": str(path)},
        )
    payload = _strict_json_file(path, owner=owner)
    observed = payload.get(digest_field)
    unsigned = dict(payload)
    unsigned.pop(digest_field, None)
    if (
        observed != payload_sha256
        or _sha256_bytes(canonical_json_bytes(unsigned)) != payload_sha256
        or payload.get("schema") != value["schema"]
        or payload.get("status") != value["status"]
    ):
        raise Wave7SequenceError(
            f"{owner} payload binding drifted",
            code="wave7_sequence.identity_drift",
        )
    return payload


def _validate_r7_amendment_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _verify_payload_binding(
        value, owner="r7 amendment", digest_field="amendment_sha256"
    )
    _exact_fields(
        payload,
        {
            "schema",
            "status",
            "effective_date",
            "section_anchor",
            "scope",
            "authority",
            "amendment_sha256",
        },
        owner="r7 amendment payload",
    )
    authority = payload.get("authority")
    if not isinstance(authority, dict):
        raise Wave7SequenceError(
            "r7 amendment authority binding is malformed",
            code="wave7_sequence.amendment",
        )
    _exact_fields(
        authority,
        {"path", "file_sha256", "section_sha256"},
        owner="r7 amendment authority",
    )
    authority_path = _absolute_path(
        authority["path"], owner="r7 amendment authority path", must_exist=True
    )
    try:
        authority_text = authority_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise Wave7SequenceError(
            "r7 amendment authority is unreadable",
            code="wave7_sequence.amendment",
        ) from exc
    start = authority_text.find(AMENDMENT_SECTION_ANCHOR)
    next_heading = (
        -1
        if start < 0
        else authority_text.find("\n### ", start + len(AMENDMENT_SECTION_ANCHOR))
    )
    section = (
        ""
        if start < 0
        else (
            authority_text[start:]
            if next_heading < 0
            else authority_text[start:next_heading]
        )
    )
    normalized_section = " ".join(section.split())
    if (
        payload.get("schema") != AMENDMENT_SCHEMA
        or payload.get("status") != "approved"
        or payload.get("effective_date") != AMENDMENT_EFFECTIVE_DATE
        or payload.get("section_anchor") != AMENDMENT_SECTION_ANCHOR
        or payload.get("scope") != AMENDMENT_SCOPE
        or authority_path != AMENDMENT_AUTHORITY_PATH
        or authority.get("file_sha256") != FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256
        or _sha256_file(authority_path) != FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256
        or authority.get("section_sha256") != _sha256_bytes(section.encode("utf-8"))
        or any(
            phrase not in normalized_section for phrase in AMENDMENT_AUTHORITY_PHRASES
        )
    ):
        raise Wave7SequenceError(
            "r7 amendment is not bound to the exact live authority section",
            code="wave7_sequence.amendment",
        )
    return payload


def _verify_source_inventory(value: Any) -> tuple[list[Path], list[dict[str, Any]]]:
    if not isinstance(value, list) or not value or len(value) > MAX_IDENTITY_FILES:
        raise Wave7SequenceError(
            "source inventory is invalid", code="wave7_sequence.source_inventory"
        )
    paths: list[Path] = []
    for row in value:
        if not isinstance(row, dict):
            raise Wave7SequenceError(
                "source inventory row is malformed",
                code="wave7_sequence.source_inventory",
            )
        _exact_fields(row, {"path", "size_bytes", "sha256"}, owner="source row")
        path = _absolute_path(row["path"], owner="source path", must_exist=True)
        expected_size = row["size_bytes"]
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or path.stat().st_size != expected_size
            or _sha256_file(path) != _digest(row["sha256"], owner="source sha256")
        ):
            raise Wave7SequenceError(
                "source inventory identity drifted",
                code="wave7_sequence.source_inventory",
                context={"path": str(path)},
            )
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise Wave7SequenceError(
            "source inventory contains duplicates",
            code="wave7_sequence.source_inventory",
        )
    return paths, [dict(row) for row in value]


def _validate_input_attestations(
    *,
    cache_payload: Mapping[str, Any],
    model_payload: Mapping[str, Any],
    config_paths: Mapping[str, Path],
) -> dict[str, Any]:
    maximum = cache_payload.get("max_cache_payload_bytes")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise Wave7SequenceError(
            "cache input attestation payload bound is invalid",
            code="wave7_sequence.input_attestation",
        )
    try:
        result = validate_training_input_attestations(
            cache_attestation=cache_payload,
            model_attestation=model_payload,
            config_paths={role: config_paths[role] for role in RUN_ROLES},
            validate_cache_payloads=True,
            rehash_model_weights=True,
            max_cache_payload_bytes=maximum,
        )
    except BaseException as exc:
        raise Wave7SequenceError(
            "native cache/model input attestation validation failed",
            code="wave7_sequence.input_attestation",
            context={"error_type": type(exc).__name__},
        ) from exc
    expected = {
        "status": "passed",
        "cache_attestation_sha256": cache_payload.get("attestation_sha256"),
        "model_attestation_sha256": model_payload.get("attestation_sha256"),
        "cache_payloads_validated": True,
        "model_weights_rehashed": True,
        "measured_cache_payload_bytes": cache_payload.get("measured_payload_bytes"),
        "materialization_policy": {
            split: cache_payload.get("splits", {}).get(split, {}).get("materialization")
            for split in ("train", "eval.forward")
        },
    }
    if "weight_hash_execution_policy" in model_payload:
        expected["weight_hash_execution_policy"] = model_payload[
            "weight_hash_execution_policy"
        ]
    if result != expected:
        raise Wave7SequenceError(
            "native cache/model input attestation projection is not exact",
            code="wave7_sequence.input_attestation",
        )
    return expected


def _validate_determinism_preflight_evidence(
    *, plan_binding: Mapping[str, Any], terminal_binding: Mapping[str, Any]
) -> dict[str, Any]:
    _exact_fields(
        plan_binding,
        {"path", "file_sha256", "payload_sha256", "schema", "status"},
        owner="determinism preflight plan binding",
    )
    _exact_fields(
        terminal_binding,
        {"path", "file_sha256", "payload_sha256", "schema", "status"},
        owner="determinism preflight terminal binding",
    )
    plan_path = _absolute_path(
        plan_binding["path"],
        owner="determinism preflight plan path",
        must_exist=True,
    )
    if (
        plan_binding["schema"] != DETERMINISM_PREFLIGHT_PLAN_SCHEMA
        or plan_binding["status"] != "prepared"
        or _sha256_file(plan_path)
        != _digest(
            plan_binding["file_sha256"],
            owner="determinism preflight plan file sha256",
        )
    ):
        raise Wave7SequenceError(
            "determinism preflight plan binding is not the frozen v4 plan",
            code="wave7_sequence.determinism_preflight",
        )
    try:
        preflight_plan = determinism_preflight._validate_plan(
            determinism_preflight._load_json(plan_path), plan_path=plan_path
        )
    except Wave7SequenceError:
        raise
    except BaseException as exc:
        raise Wave7SequenceError(
            "determinism preflight plan failed production validation",
            code="wave7_sequence.determinism_preflight",
            context={"error_type": type(exc).__name__},
        ) from exc
    if preflight_plan["plan_payload_sha256"] != _digest(
        plan_binding["payload_sha256"],
        owner="determinism preflight plan payload sha256",
    ):
        raise Wave7SequenceError(
            "determinism preflight plan payload binding drifted",
            code="wave7_sequence.determinism_preflight",
        )
    terminal_path = _absolute_path(
        terminal_binding["path"],
        owner="determinism preflight terminal path",
        must_exist=True,
    )
    terminal_file_sha256 = _digest(
        terminal_binding["file_sha256"],
        owner="determinism preflight terminal file sha256",
    )
    terminal_payload_sha256 = _digest(
        terminal_binding["payload_sha256"],
        owner="determinism preflight terminal payload sha256",
    )
    try:
        terminal = determinism_preflight.verify_receipt(
            terminal_path,
            expected_file_sha256=terminal_file_sha256,
            plan_path=plan_path,
            expected_plan_file_sha256=plan_binding["file_sha256"],
        )
    except BaseException as exc:
        raise Wave7SequenceError(
            "determinism preflight terminal failed production evidence verification",
            code="wave7_sequence.determinism_preflight",
            context={"error_type": type(exc).__name__},
        ) from exc
    if (
        terminal_binding["schema"] != DETERMINISM_PREFLIGHT_RECEIPT_SCHEMA
        or terminal_binding["status"] != "passed"
        or terminal["receipt_payload_sha256"] != terminal_payload_sha256
        or terminal["plan_sha256"] != preflight_plan["plan_payload_sha256"]
    ):
        raise Wave7SequenceError(
            "determinism preflight terminal is not bound to its frozen plan",
            code="wave7_sequence.determinism_preflight",
        )
    return terminal


def _validate_request_digest(request: Mapping[str, Any]) -> str:
    observed = _digest(
        request.get("request_payload_sha256"), owner="request payload sha256"
    )
    unsigned = dict(request)
    unsigned.pop("request_payload_sha256", None)
    recomputed = _sha256_bytes(canonical_json_bytes(unsigned))
    if observed != recomputed:
        raise Wave7SequenceError(
            "request payload digest is invalid",
            code="wave7_sequence.request_digest",
            context={"expected": recomputed, "observed": observed},
        )
    return observed


def _validate_determinism_process(value: Any, *, launch_id: str) -> None:
    if not isinstance(value, dict):
        raise Wave7SequenceError(
            "determinism process record is malformed",
            code="wave7_sequence.determinism_preflight",
        )
    _exact_fields(
        value,
        {
            "launch_id",
            "pid",
            "returncode",
            "termination",
            "process_scope",
            "cleanup_verified",
            "stdout_tail",
            "stderr_tail",
        },
        owner="determinism process record",
    )
    pid = value["pid"]
    scope = value["process_scope"]
    if (
        value["launch_id"] != launch_id
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or value["returncode"] != 0
        or value["termination"] != "exited"
        or value["cleanup_verified"] is not True
        or not isinstance(value["stdout_tail"], str)
        or not isinstance(value["stderr_tail"], str)
        or not isinstance(scope, dict)
    ):
        raise Wave7SequenceError(
            "determinism process result or cleanup status is not passed",
            code="wave7_sequence.determinism_preflight",
        )
    _exact_fields(
        scope,
        {
            "session_id",
            "leader",
            "observed_members",
            "term_sent",
            "kill_sent",
            "remaining_members",
            "leader_reaped",
            "cleanup_failure",
            "cleanup_verified",
        },
        owner="determinism process cleanup scope",
    )
    leader = scope["leader"]
    observed_members = scope["observed_members"]
    if (
        scope["session_id"] != pid
        or not isinstance(leader, dict)
        or set(leader) != {"pid", "start_time_ticks"}
        or leader["pid"] != pid
        or isinstance(leader["start_time_ticks"], bool)
        or not isinstance(leader["start_time_ticks"], int)
        or not isinstance(observed_members, list)
        or not observed_members
        or scope["remaining_members"] != []
        or not isinstance(scope["term_sent"], bool)
        or not isinstance(scope["kill_sent"], bool)
        or scope["leader_reaped"] is not True
        or scope["cleanup_failure"] is not None
        or scope["cleanup_verified"] is not True
    ):
        raise Wave7SequenceError(
            "determinism cleanup scope is not fail-closed and complete",
            code="wave7_sequence.determinism_preflight",
        )
    member_keys: list[tuple[int, int]] = []
    for member in observed_members:
        if not isinstance(member, dict):
            raise Wave7SequenceError(
                "determinism cleanup member is malformed",
                code="wave7_sequence.determinism_preflight",
            )
        _exact_fields(
            member,
            {
                "pid",
                "state",
                "parent_pid",
                "process_group_id",
                "session_id",
                "start_time_ticks",
            },
            owner="determinism cleanup member",
        )
        if (
            member["session_id"] != pid
            or not isinstance(member["pid"], int)
            or isinstance(member["pid"], bool)
            or not isinstance(member["start_time_ticks"], int)
            or isinstance(member["start_time_ticks"], bool)
            or not isinstance(member["state"], str)
            or len(member["state"]) != 1
        ):
            raise Wave7SequenceError(
                "determinism cleanup member identity is invalid",
                code="wave7_sequence.determinism_preflight",
            )
        member_keys.append((member["pid"], member["start_time_ticks"]))
    if (
        len(member_keys) != len(set(member_keys))
        or (pid, leader["start_time_ticks"]) not in member_keys
    ):
        raise Wave7SequenceError(
            "determinism cleanup identity inventory is incomplete",
            code="wave7_sequence.determinism_preflight",
        )


def _validate_determinism_preflight_payload(payload: Mapping[str, Any]) -> None:
    _exact_fields(
        payload,
        {
            "schema",
            "status",
            "world_size",
            "launch_count",
            "mismatches",
            "plan_sha256",
            "attempt_marker",
            "processes",
            "rank_receipts",
            "comparisons",
            "gpu_shared_occupancy_sweeps",
            "cleanup",
            "claim_scope",
            "receipt_payload_sha256",
        },
        owner="8-rank determinism preflight terminal",
    )
    comparisons = payload["comparisons"]
    cleanup = payload["cleanup"]
    processes = payload["processes"]
    rank_receipts = payload["rank_receipts"]
    sweeps = payload["gpu_shared_occupancy_sweeps"]
    if (
        payload["schema"] != DETERMINISM_PREFLIGHT_RECEIPT_SCHEMA
        or payload["status"] != "passed"
        or payload["world_size"] != 8
        or payload["launch_count"] != 2
        or payload["mismatches"] != []
        or not isinstance(payload["attempt_marker"], dict)
        or not isinstance(processes, list)
        or len(processes) != 2
        or not isinstance(rank_receipts, list)
        or len(rank_receipts) != 16
        or not isinstance(sweeps, list)
        or len(sweeps) != 3
        or payload["claim_scope"] != DETERMINISM_CLAIM_SCOPE
    ):
        raise Wave7SequenceError(
            "determinism preflight terminal is not an exact passed two-launch receipt",
            code="wave7_sequence.determinism_preflight",
        )
    _digest(payload["plan_sha256"], owner="determinism preflight plan sha256")
    if not isinstance(comparisons, dict):
        raise Wave7SequenceError(
            "determinism comparisons are malformed",
            code="wave7_sequence.determinism_preflight",
        )
    _exact_fields(
        comparisons,
        {
            "launch_digest_equal",
            "device_mapping_equal",
            "native_mapping_equal",
            "rank_count_per_launch",
            "mismatch_ranks",
            "launch_a_aggregate_sha256",
            "launch_b_aggregate_sha256",
        },
        owner="determinism comparisons",
    )
    marker_payload = _verify_receipt_binding(
        payload["attempt_marker"], owner="determinism attempt marker"
    )
    if (
        payload["attempt_marker"]["schema"] != DETERMINISM_PREFLIGHT_MARKER_SCHEMA
        or payload["attempt_marker"]["status"] != "started"
        or marker_payload.get("schema") != DETERMINISM_PREFLIGHT_MARKER_SCHEMA
    ):
        raise Wave7SequenceError(
            "determinism attempt marker binding is not the frozen v4 marker",
            code="wave7_sequence.determinism_preflight",
        )
    for process, launch_id in zip(processes, ("launch-a", "launch-b"), strict=True):
        _validate_determinism_process(process, launch_id=launch_id)
    launch_a = _digest(
        comparisons["launch_a_aggregate_sha256"],
        owner="launch-a aggregate sha256",
    )
    launch_b = _digest(
        comparisons["launch_b_aggregate_sha256"],
        owner="launch-b aggregate sha256",
    )
    if (
        comparisons["launch_digest_equal"] is not True
        or comparisons["device_mapping_equal"] is not True
        or comparisons["native_mapping_equal"] is not True
        or comparisons["rank_count_per_launch"] != 8
        or comparisons["mismatch_ranks"] != []
        or launch_a != launch_b
    ):
        raise Wave7SequenceError(
            "determinism comparison gate is not exact",
            code="wave7_sequence.determinism_preflight",
        )
    for sweep, phase in zip(
        sweeps, ("post-launch-a", "post-launch-b", "preterminal"), strict=True
    ):
        if not isinstance(sweep, dict):
            raise Wave7SequenceError(
                "determinism shared-GPU sweep is malformed",
                code="wave7_sequence.determinism_preflight",
            )
        _exact_fields(
            sweep,
            {
                "phase",
                "samples",
                "device_inventory_matches_baseline",
                "observed_compute_processes",
                "new_compute_processes",
                "admitted",
            },
            owner="determinism shared-GPU sweep",
        )
        samples = sweep["samples"]
        if (
            sweep["phase"] != phase
            or not isinstance(samples, list)
            or len(samples) != 2
            or not isinstance(sweep["observed_compute_processes"], list)
            or sweep["new_compute_processes"] != []
            or sweep["device_inventory_matches_baseline"] is not True
            or sweep["admitted"] is not True
        ):
            raise Wave7SequenceError(
                "determinism shared-GPU recovery sweep is not admitted",
                code="wave7_sequence.determinism_preflight",
            )
        monotonic_values: list[int] = []
        for sample_index, sample in enumerate(samples):
            if (
                not isinstance(sample, dict)
                or sample.get("sample_index") != sample_index
                or isinstance(sample.get("sample_monotonic_ns"), bool)
                or not isinstance(sample.get("sample_monotonic_ns"), int)
                or sample["sample_monotonic_ns"] <= 0
            ):
                raise Wave7SequenceError(
                    "determinism shared-GPU recovery sample is malformed",
                    code="wave7_sequence.determinism_preflight",
                )
            monotonic_values.append(sample["sample_monotonic_ns"])
        if monotonic_values[1] - monotonic_values[0] < 2_000_000_000:
            raise Wave7SequenceError(
                "determinism shared-GPU recovery samples are less than two seconds apart",
                code="wave7_sequence.determinism_preflight",
            )
    if not isinstance(cleanup, dict):
        raise Wave7SequenceError(
            "determinism cleanup is malformed",
            code="wave7_sequence.determinism_preflight",
        )
    _exact_fields(
        cleanup, {"all_process_groups_exited", "bounded"}, owner="determinism cleanup"
    )
    if cleanup != {"all_process_groups_exited": True, "bounded": True}:
        raise Wave7SequenceError(
            "determinism cleanup gate is not exact",
            code="wave7_sequence.determinism_preflight",
        )


def _validate_native_reference_binding(value: Mapping[str, Any]) -> None:
    _exact_fields(
        value,
        {"path", "file_sha256", "payload_sha256", "schema", "status"},
        owner="runtime native reference binding",
    )
    path = _absolute_path(
        value["path"], owner="runtime native reference path", must_exist=True
    )
    expected_file_sha256 = _digest(
        value["file_sha256"], owner="runtime native reference file sha256"
    )
    expected_payload_sha256 = _digest(
        value["payload_sha256"], owner="runtime native reference payload sha256"
    )
    if _sha256_file(path) != expected_file_sha256:
        raise Wave7SequenceError(
            "runtime native reference file digest drifted",
            code="wave7_sequence.runtime_admission",
        )
    receipt = _strict_json_file(path, owner="runtime native reference")
    _exact_fields(
        receipt,
        {
            "schema",
            "created_at",
            "repository_root",
            "source_identity",
            "attention_backend",
            "pinned_runtime_baseline",
            "cuda",
            "mapped_native_execution",
            "terminal_status",
            "receipt_sha256",
        },
        owner="runtime native reference",
    )
    unsigned = dict(receipt)
    observed_payload_sha256 = unsigned.pop("receipt_sha256", None)
    recomputed_payload_sha256 = _sha256_bytes(canonical_json_bytes(unsigned))
    source_path = Path(execution_provenance.__file__).resolve()
    source_identity = receipt["source_identity"]
    baseline = receipt["pinned_runtime_baseline"]
    mapped = receipt["mapped_native_execution"]
    if (
        observed_payload_sha256 != expected_payload_sha256
        or recomputed_payload_sha256 != expected_payload_sha256
        or receipt["schema"] != NATIVE_REFERENCE_SCHEMA
        or receipt["terminal_status"] != "passed"
        or value["schema"] != NATIVE_REFERENCE_SCHEMA
        or value["status"] != "passed"
        or receipt["repository_root"] != str(REPO_ROOT)
        or receipt["attention_backend"] != "flash_attention_2"
        or source_identity
        != {"path": str(source_path), "sha256": _sha256_file(source_path)}
        or baseline
        != {
            "schema_version": 3,
            "baseline_sha256": execution_provenance.PINNED_RUNTIME_BASELINE_SHA256,
            "admitted": True,
        }
        or not isinstance(mapped, dict)
        or mapped.get("schema_version") != 1
        or mapped.get("cuda_initialized") is not True
        or mapped.get("admitted") is not True
        or mapped.get("mismatches") != []
    ):
        raise Wave7SequenceError(
            "runtime native reference is not the exact passed reference",
            code="wave7_sequence.runtime_admission",
        )


def _validate_late_native_expectations(value: Any) -> None:
    expected_distributions = {
        "flash_attention_2": "flash-attn",
        "libcublas": "nvidia-cublas-cu12",
        "libnccl": "nvidia-nccl-cu12",
    }
    if not isinstance(value, dict):
        raise Wave7SequenceError(
            "runtime late-native expectations are malformed",
            code="wave7_sequence.runtime_admission",
        )
    _exact_fields(
        value, set(expected_distributions), owner="runtime late-native expectations"
    )
    for component, expected_distribution in expected_distributions.items():
        identity = value[component]
        if not isinstance(identity, dict):
            raise Wave7SequenceError(
                "runtime late-native identity is malformed",
                code="wave7_sequence.runtime_admission",
                context={"component": component},
            )
        _exact_fields(
            identity,
            {"distribution", "soname", "path", "size_bytes", "file_sha256"},
            owner=f"runtime late-native identity {component}",
        )
        path = _absolute_path(
            identity["path"],
            owner=f"runtime late-native path {component}",
            must_exist=True,
        )
        try:
            info = path.stat()
        except OSError as exc:
            raise Wave7SequenceError(
                "runtime late-native identity is unreadable",
                code="wave7_sequence.runtime_admission",
                context={"component": component},
            ) from exc
        if (
            identity["distribution"] != expected_distribution
            or identity["soname"] != path.name
            or isinstance(identity["size_bytes"], bool)
            or identity["size_bytes"] != info.st_size
            or not stat.S_ISREG(info.st_mode)
            or _sha256_file(path)
            != _digest(
                identity["file_sha256"],
                owner=f"runtime late-native sha256 {component}",
            )
        ):
            raise Wave7SequenceError(
                "runtime late-native identity drifted",
                code="wave7_sequence.runtime_admission",
                context={"component": component, "path": str(path)},
            )


def _collect_live_execution_provenance() -> dict[str, Any]:
    value = execution_provenance.collect_execution_provenance(repository_root=REPO_ROOT)
    if not isinstance(value, dict):
        raise Wave7SequenceError(
            "live execution provenance is malformed",
            code="wave7_sequence.runtime_provenance",
        )
    return value


def _require_pinned_runtime_baseline(provenance: Mapping[str, Any]) -> dict[str, Any]:
    value = execution_provenance.require_pinned_runtime_baseline(
        provenance=provenance,
        attention_backend="flash_attention_2",
    )
    if not isinstance(value, dict):
        raise Wave7SequenceError(
            "pinned runtime baseline result is malformed",
            code="wave7_sequence.runtime_admission",
        )
    return value


def _validate_runtime_admission_payload(
    payload: Mapping[str, Any],
    *,
    expected_provenance_sha256: str,
    require_live_provenance: bool,
) -> None:
    _exact_fields(
        payload,
        {
            "schema",
            "status",
            "model_loaded",
            "cuda_initialized",
            "repository_root",
            "source_identity",
            "native_reference",
            "attention_backend",
            "provenance",
            "pinned_runtime_baseline",
            "late_native_expectations",
            "receipt_payload_sha256",
        },
        owner="runtime admission receipt",
    )
    provenance = payload["provenance"]
    source_path = Path(execution_provenance.__file__).resolve()
    if (
        payload["schema"] != RUNTIME_ADMISSION_SCHEMA
        or payload["status"] != "passed"
        or payload["model_loaded"] is not False
        or payload["cuda_initialized"] is not False
        or payload["repository_root"] != str(REPO_ROOT)
        or payload["attention_backend"] != "flash_attention_2"
        or payload["source_identity"]
        != {"path": str(source_path), "file_sha256": _sha256_file(source_path)}
        or not isinstance(provenance, dict)
        or set(provenance)
        != {"schema_version", "repository", "dependencies", "runtime"}
        or provenance.get("schema_version") != 1
    ):
        raise Wave7SequenceError(
            "runtime admission identity is not exact",
            code="wave7_sequence.runtime_admission",
        )
    observed_provenance_sha256 = _sha256_bytes(canonical_json_bytes(provenance))
    if observed_provenance_sha256 != expected_provenance_sha256:
        raise Wave7SequenceError(
            "request provenance digest is not bound to runtime admission",
            code="wave7_sequence.runtime_provenance",
            context={
                "expected": expected_provenance_sha256,
                "observed": observed_provenance_sha256,
            },
        )
    native_reference = payload["native_reference"]
    if not isinstance(native_reference, dict):
        raise Wave7SequenceError(
            "runtime native reference binding is malformed",
            code="wave7_sequence.runtime_admission",
        )
    _validate_native_reference_binding(native_reference)
    try:
        pinned = _require_pinned_runtime_baseline(provenance)
    except BaseException as exc:
        if isinstance(exc, Wave7SequenceError):
            raise
        raise Wave7SequenceError(
            "runtime provenance is not admitted by the pinned baseline",
            code="wave7_sequence.runtime_admission",
            context={"error_type": type(exc).__name__},
        ) from exc
    if payload["pinned_runtime_baseline"] != pinned:
        raise Wave7SequenceError(
            "runtime admission baseline comparison drifted",
            code="wave7_sequence.runtime_admission",
        )
    _validate_late_native_expectations(payload["late_native_expectations"])
    if require_live_provenance:
        try:
            live_provenance = _collect_live_execution_provenance()
        except BaseException as exc:
            if isinstance(exc, Wave7SequenceError):
                raise
            raise Wave7SequenceError(
                "live execution provenance collection failed",
                code="wave7_sequence.runtime_provenance",
                context={"error_type": type(exc).__name__},
            ) from exc
        live_sha256 = _sha256_bytes(canonical_json_bytes(live_provenance))
        if live_sha256 != expected_provenance_sha256 or live_provenance != provenance:
            raise Wave7SequenceError(
                "live execution provenance drifted from runtime admission",
                code="wave7_sequence.runtime_provenance",
                context={
                    "expected": expected_provenance_sha256,
                    "observed": live_sha256,
                },
            )


def _validate_argv(value: Any, *, phase: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > MAX_ARGV_ITEMS:
        raise Wave7SequenceError(
            "phase argv count is invalid",
            code="wave7_sequence.argv",
            context={"phase": phase},
        )
    result: list[str] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, str)
            or not item
            or len(item) > MAX_ARG_CHARS
            or "\x00" in item
            or "\n" in item
        ):
            raise Wave7SequenceError(
                "phase argv item is invalid",
                code="wave7_sequence.argv",
                context={"phase": phase, "index": index},
            )
        result.append(item)
    executable = Path(result[0]).name.lower()
    if executable in _FORBIDDEN_SHELLS or any(
        item.strip().lower() in _FORBIDDEN_CONTROL_TOKENS for item in result
    ):
        raise Wave7SequenceError(
            "shell control or retry commands are forbidden",
            code="wave7_sequence.no_retry",
            context={"phase": phase},
        )
    placeholder_count = sum(item.count(PRE_CHILD_DIGEST_PLACEHOLDER) for item in result)
    expected_count = 1 if phase in {"verify_pre_child", "resume_child"} else 0
    if placeholder_count != expected_count:
        raise Wave7SequenceError(
            "pre-child digest placeholder placement is invalid",
            code="wave7_sequence.argv",
            context={"phase": phase, "count": placeholder_count},
        )
    return result


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _expected_r7_run_roots() -> dict[str, Path]:
    return {role: R7_SEQUENCE_ROOT / "runs" / role for role in RUN_ROLES}


def _expected_r7_targets() -> dict[str, Path]:
    return {
        "sequence_marker": R7_SEQUENCE_ROOT / "sequence-marker.json",
        "sequence_receipt": R7_SEQUENCE_ROOT / "sequence-receipt.json",
        "publication_failure_sidecar": (
            R7_SEQUENCE_ROOT / "sequence-receipt.publication-failure.json"
        ),
        "interruption_marker": (
            R7_SEQUENCE_ROOT / "interrupted-parent-attempt-marker.json"
        ),
        "interruption_receipt": (
            R7_SEQUENCE_ROOT / "interrupted-parent-termination-receipt.json"
        ),
        "pre_child_receipt": R7_SEQUENCE_ROOT / "pre-child-receipt.json",
        "final_receipt": (R7_SEQUENCE_ROOT / "exact-resume-comparison-receipt-v2.json"),
    }


def _require_r7_external_path(value: Mapping[str, Any], *, owner: str) -> None:
    path = _absolute_path(value.get("path"), owner=f"{owner} path", must_exist=True)
    if not path.is_relative_to(R7_SEQUENCE_ROOT):
        raise Wave7SequenceError(
            f"{owner} is outside the canonical r7 namespace",
            code="wave7_sequence.r7_namespace",
            context={"path": str(path), "root": str(R7_SEQUENCE_ROOT)},
        )


def _validate_topology(
    run_roots: Mapping[str, str], targets: Mapping[str, str], *, require_absent: bool
) -> tuple[dict[str, Path], dict[str, Path]]:
    _exact_fields(run_roots, set(RUN_ROLES), owner="run_roots")
    _exact_fields(targets, set(TARGET_NAMES), owner="targets")
    roots = {
        name: _absolute_path(value, owner=f"run root {name}", must_exist=False)
        for name, value in run_roots.items()
    }
    target_paths = {
        name: _absolute_path(value, owner=f"target {name}", must_exist=False)
        for name, value in targets.items()
    }
    if roots != _expected_r7_run_roots() or target_paths != _expected_r7_targets():
        raise Wave7SequenceError(
            "run roots or targets are outside the exact r7 namespace",
            code="wave7_sequence.r7_namespace",
        )
    combined = tuple(
        [(f"run_roots.{key}", value) for key, value in roots.items()]
        + [(f"targets.{key}", value) for key, value in target_paths.items()]
    )
    for index, (left_name, left) in enumerate(combined):
        for right_name, right in combined[index + 1 :]:
            if _paths_overlap(left, right):
                raise Wave7SequenceError(
                    "sequence paths overlap",
                    code="wave7_sequence.path_topology",
                    context={
                        "left": left_name,
                        "left_path": str(left),
                        "right": right_name,
                        "right_path": str(right),
                    },
                )
    if require_absent:
        for name, path in combined:
            if path.exists() or path.is_symlink():
                raise Wave7SequenceError(
                    "sequence target already exists",
                    code="wave7_sequence.target_exists",
                    context={"owner": name, "path": str(path)},
                )
            try:
                assert_absent_artifact_target(path)
            except BaseException as exc:
                raise Wave7SequenceError(
                    "sequence target is not safely publishable",
                    code="wave7_sequence.target_exists",
                    context={"owner": name, "path": str(path)},
                ) from exc
    return roots, target_paths


def _train_command(config_path: Path, *, port: int) -> list[str]:
    if ACCELERATE_EXECUTABLE is None:
        raise Wave7SequenceError(
            "the production accelerate executable is unavailable",
            code="wave7_sequence.command_owner",
        )
    return [
        str(ACCELERATE_EXECUTABLE),
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(GPU_DEVICE_COUNT),
        "--main_process_port",
        str(port),
        "--module",
        "src.train",
        "--config",
        str(config_path),
    ]


def _expected_commands(
    *,
    config_paths: Mapping[str, Path],
    roots: Mapping[str, Path],
    targets: Mapping[str, Path],
    policy: Mapping[str, Any],
    provenance_sha256: str,
) -> dict[str, list[str]]:
    interrupt_sha256 = _sha256_file(INTERRUPT_CONTROLLER)
    uninterrupted = _train_command(
        config_paths["uninterrupted"], port=TRAIN_PORTS["uninterrupted"]
    )
    parent = _train_command(
        config_paths["interrupted_parent"], port=TRAIN_PORTS["interrupted_parent"]
    )
    child = _train_command(
        config_paths["resume_child"], port=TRAIN_PORTS["resume_child"]
    )
    interrupted_config_args = [
        item
        for source in load_train_config(config_paths["interrupted_parent"]).sources
        for item in ("--config", str(source.path))
    ]
    common_comparator = [
        "--uninterrupted-run-dir",
        str(roots["uninterrupted"]),
        "--interrupted-parent-run-dir",
        str(roots["interrupted_parent"]),
    ]
    comparison_tail = [
        "--interruption-marker",
        str(targets["interruption_marker"]),
        "--termination-receipt",
        str(targets["interruption_receipt"]),
    ]
    identity_tail = [
        "--expected-interrupt-source-sha256",
        interrupt_sha256,
        "--expected-source-sha256",
        FROZEN_V2_COMPARATOR_SHA256,
        "--expected-provenance-sha256",
        provenance_sha256,
    ]
    return {
        "uninterrupted": uninterrupted,
        "interrupted_parent": [
            str(PYTHON_EXECUTABLE),
            str(INTERRUPT_CONTROLLER),
            "--parent-run-dir",
            str(roots["interrupted_parent"]),
            "--expected-checkpoint-step",
            "3",
            "--marker",
            str(targets["interruption_marker"]),
            "--receipt",
            str(targets["interruption_receipt"]),
            "--timeout-seconds",
            str(policy["phase_timeout_seconds"]["interrupted_parent"]),
            "--term-grace-seconds",
            str(policy["term_grace_seconds"]),
            "--kill-grace-seconds",
            str(policy["kill_grace_seconds"]),
            "--stability-seconds",
            str(policy["gpu_baseline_stability_seconds"]),
            "--poll-seconds",
            "0.01",
            "--source",
            str(INTERRUPT_CONTROLLER),
            "--source",
            str(TRAIN_ENTRYPOINT),
            "--source",
            str(REQUIRED_SOURCE_PATHS[0]),
            "--source",
            str(COMPARATOR_V2),
            *interrupted_config_args,
            "--",
            *parent,
        ],
        "pre_child": [
            str(PYTHON_EXECUTABLE),
            str(COMPARATOR_V2),
            "pre-child",
            *common_comparator,
            *comparison_tail,
            "--output",
            str(targets["pre_child_receipt"]),
            *identity_tail,
        ],
        "verify_pre_child": [
            str(PYTHON_EXECUTABLE),
            str(COMPARATOR_V2),
            "verify-pre-child",
            "--receipt",
            str(targets["pre_child_receipt"]),
            "--expected-payload-sha256",
            PRE_CHILD_DIGEST_PLACEHOLDER,
        ],
        "resume_child": [
            str(PYTHON_EXECUTABLE),
            str(Path(__file__).resolve()),
            "launch-child",
            "--receipt",
            str(targets["pre_child_receipt"]),
            "--expected-payload-sha256",
            PRE_CHILD_DIGEST_PLACEHOLDER,
            "--config",
            str(config_paths["resume_child"]),
            "--run-root",
            str(roots["resume_child"]),
            "--parent-run-root",
            str(roots["interrupted_parent"]),
            "--",
            *child,
        ],
        "final_compare": [
            str(PYTHON_EXECUTABLE),
            str(COMPARATOR_V2),
            "compare",
            *common_comparator,
            "--resume-child-run-dir",
            str(roots["resume_child"]),
            *comparison_tail,
            "--output",
            str(targets["final_receipt"]),
            *identity_tail,
        ],
    }


def _validate_exact_commands(
    commands: Mapping[str, list[str]],
    *,
    config_paths: Mapping[str, Path],
    roots: Mapping[str, Path],
    targets: Mapping[str, Path],
    policy: Mapping[str, Any],
    provenance_sha256: str,
) -> None:
    expected = _expected_commands(
        config_paths=config_paths,
        roots=roots,
        targets=targets,
        policy=policy,
        provenance_sha256=provenance_sha256,
    )
    for phase in PHASE_ORDER:
        if commands[phase] != expected[phase]:
            raise Wave7SequenceError(
                "phase command does not match its frozen production grammar",
                code="wave7_sequence.command_contract",
                context={"phase": phase},
            )


def _validate_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "phase_order",
        "gpu_phases",
        "gpu_admission_mode",
        "gpu_memory_total_mib",
        "gpu_memory_used_ceiling_mib",
        "gpu_memory_headroom_floor_mib",
        "gpu_post_cleanup_stability_seconds",
        "no_retry",
        "max_attempts_per_phase",
        "preserve_failed_artifacts",
        "phase_timeout_seconds",
        "term_grace_seconds",
        "kill_grace_seconds",
        "gpu_baseline_stability_seconds",
        "max_total_wall_seconds",
        "max_total_gpu_device_seconds",
    }
    _exact_fields(value, expected, owner="policy")
    if value["phase_order"] != list(PHASE_ORDER) or value["gpu_phases"] != list(
        GPU_PHASES
    ):
        raise Wave7SequenceError(
            "phase order or GPU phase set is not frozen",
            code="wave7_sequence.policy",
        )
    if (
        value["gpu_admission_mode"] != SHARED_GPU_ADMISSION_MODE
        or isinstance(value["gpu_memory_total_mib"], bool)
        or not isinstance(value["gpu_memory_total_mib"], int)
        or value["gpu_memory_total_mib"] != SHARED_GPU_MEMORY_TOTAL_MIB
        or isinstance(value["gpu_memory_used_ceiling_mib"], bool)
        or not isinstance(value["gpu_memory_used_ceiling_mib"], int)
        or value["gpu_memory_used_ceiling_mib"] != SHARED_GPU_MEMORY_USED_CEILING_MIB
        or isinstance(value["gpu_memory_headroom_floor_mib"], bool)
        or not isinstance(value["gpu_memory_headroom_floor_mib"], int)
        or value["gpu_memory_headroom_floor_mib"]
        != SHARED_GPU_MEMORY_HEADROOM_FLOOR_MIB
        or isinstance(value["gpu_baseline_stability_seconds"], bool)
        or not isinstance(value["gpu_baseline_stability_seconds"], (int, float))
        or float(value["gpu_baseline_stability_seconds"])
        != SHARED_GPU_BASELINE_STABILITY_SECONDS
        or isinstance(value["gpu_post_cleanup_stability_seconds"], bool)
        or not isinstance(value["gpu_post_cleanup_stability_seconds"], (int, float))
        or float(value["gpu_post_cleanup_stability_seconds"])
        != SHARED_GPU_POST_CLEANUP_STABILITY_SECONDS
    ):
        raise Wave7SequenceError(
            "shared GPU admission mode and memory ceiling are frozen",
            code="wave7_sequence.policy",
        )
    if (
        value["no_retry"] is not True
        or value["max_attempts_per_phase"] != 1
        or isinstance(value["max_attempts_per_phase"], bool)
        or value["preserve_failed_artifacts"] is not True
    ):
        raise Wave7SequenceError(
            "at-most-once forensic preservation policy is required",
            code="wave7_sequence.policy",
        )
    timeouts = value["phase_timeout_seconds"]
    if not isinstance(timeouts, dict):
        raise Wave7SequenceError(
            "phase timeouts must be one object", code="wave7_sequence.policy"
        )
    _exact_fields(timeouts, set(PHASE_ORDER), owner="phase timeouts")
    normalized = dict(value)
    normalized["phase_timeout_seconds"] = {
        phase: _positive_number(timeouts[phase], owner=f"timeout {phase}")
        for phase in PHASE_ORDER
    }
    for field in (
        "term_grace_seconds",
        "kill_grace_seconds",
        "gpu_baseline_stability_seconds",
        "gpu_post_cleanup_stability_seconds",
        "max_total_wall_seconds",
        "max_total_gpu_device_seconds",
    ):
        normalized[field] = _positive_number(value[field], owner=field)
    if (
        any(
            timeout > MAX_AUTHORIZED_PHASE_SECONDS
            for timeout in normalized["phase_timeout_seconds"].values()
        )
        or normalized["max_total_wall_seconds"] > MAX_AUTHORIZED_WALL_SECONDS
        or normalized["max_total_gpu_device_seconds"]
        > MAX_AUTHORIZED_GPU_DEVICE_SECONDS
        or normalized["max_total_gpu_device_seconds"]
        > normalized["max_total_wall_seconds"] * GPU_DEVICE_COUNT
    ):
        raise Wave7SequenceError(
            "sequence cost policy exceeds the r7 authorization",
            code="wave7_sequence.policy",
        )
    return normalized


def _validate_oracles(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "pre_child": {
            "schema": PRE_CHILD_RECEIPT_SCHEMA,
            "status": "passed",
            "child_launch_authorized": True,
            "mismatches": [],
        },
        "final": {
            "schema": FINAL_RECEIPT_SCHEMA,
            "status": "passed",
            "mismatches": [],
        },
    }
    if value != expected:
        raise Wave7SequenceError(
            "comparison oracles are not the frozen Wave 7 oracles",
            code="wave7_sequence.oracle",
        )
    return expected


def _configured_run_dir(config: Any) -> Path:
    root = Path(config.run.artifact_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()
    run_name = config.run.output_dir or config.run.name
    relative = Path(run_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise Wave7SequenceError(
            "config run output escapes its artifact root",
            code="wave7_sequence.config_run_root",
        )
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise Wave7SequenceError(
            "config run output escapes its artifact root",
            code="wave7_sequence.config_run_root",
        )
    return result


def _resolved_config_projection(
    path: Path,
    *,
    role: str,
    roots: Mapping[str, Path],
    require_absent_run: bool,
) -> dict[str, Any]:
    try:
        resolved = load_train_config(path)
    except BaseException as exc:
        raise Wave7SequenceError(
            "training config failed production model-free validation",
            code="wave7_sequence.config_invalid",
            context={"role": role, "path": str(path), "error_type": type(exc).__name__},
        ) from exc
    config = resolved.config
    if config.runtime.determinism.mode != "strict_cuda_replay_v1":
        raise Wave7SequenceError(
            "training config does not require strict CUDA replay",
            code="wave7_sequence.config_determinism",
            context={"role": role},
        )
    configured_run_dir = _configured_run_dir(config)
    if require_absent_run:
        try:
            production_run_dir = resolve_run_directory(config, cwd=REPO_ROOT).run_dir
        except BaseException as exc:
            raise Wave7SequenceError(
                "training config run directory failed production validation",
                code="wave7_sequence.config_run_root",
                context={"role": role, "error_type": type(exc).__name__},
            ) from exc
        if production_run_dir != configured_run_dir:
            raise Wave7SequenceError(
                "training config run directory resolvers disagree",
                code="wave7_sequence.config_run_root",
                context={"role": role},
            )
    if (
        config.run.name != role
        or config.run.collision_policy != "fail"
        or configured_run_dir != roots[role]
    ):
        raise Wave7SequenceError(
            "training config role or run root is not exact",
            code="wave7_sequence.config_run_root",
            context={
                "role": role,
                "run_name": config.run.name,
                "configured_run_dir": str(configured_run_dir),
                "expected_run_dir": str(roots[role]),
            },
        )
    expected_checkpoint_dir = (
        roots["interrupted_parent"] / "checkpoints" / "step-3"
        if role == "resume_child"
        else None
    )
    observed_checkpoint_dir = (
        None
        if config.resume.checkpoint_dir is None
        else Path(config.resume.checkpoint_dir).resolve()
    )
    if (
        config.resume.mode != "exact_same_world_size"
        or observed_checkpoint_dir != expected_checkpoint_dir
    ):
        raise Wave7SequenceError(
            "training config resume mode or checkpoint directory is not exact",
            code="wave7_sequence.config_resume",
            context={
                "role": role,
                "mode": config.resume.mode,
                "checkpoint_dir": (
                    None
                    if observed_checkpoint_dir is None
                    else str(observed_checkpoint_dir)
                ),
                "expected_checkpoint_dir": (
                    None
                    if expected_checkpoint_dir is None
                    else str(expected_checkpoint_dir)
                ),
            },
        )
    if (
        config.training.max_steps != 5
        or config.training.forward_input_provider_mode != "synchronous"
        or config.eval.forward.every_fraction is not None
        or config.eval.forward.steps != (3,)
        or config.checkpoint.every_fraction is not None
        or config.checkpoint.steps != (3, 5)
        or config.checkpoint.save_final is not True
    ):
        raise Wave7SequenceError(
            "training config step and cadence contract is not exact",
            code="wave7_sequence.config_cadence",
            context={"role": role},
        )
    full_projection = resolved.config_dict
    semantic_projection = {
        key: value
        for key, value in full_projection.items()
        if key not in {"run", "resume"}
    }
    return {
        "role": role,
        "entry_path": str(resolved.entry_config_path),
        "schema_version": resolved.schema_version,
        "loader_version": resolved.loader_version,
        "fingerprint": resolved.fingerprint,
        "sources": [source.to_artifact_dict() for source in resolved.sources],
        "resolved_run_dir": str(configured_run_dir),
        "config_projection": full_projection,
        "semantic_projection": semantic_projection,
        "semantic_projection_sha256": _sha256_bytes(
            canonical_json_bytes(semantic_projection)
        ),
    }


def _resolved_config_inventory(
    paths: Mapping[str, Path],
    *,
    roots: Mapping[str, Path],
    require_absent_runs: bool,
) -> dict[str, dict[str, Any]]:
    inventory = {
        role: _resolved_config_projection(
            paths[role],
            role=role,
            roots=roots,
            require_absent_run=require_absent_runs,
        )
        for role in RUN_ROLES
    }
    reference = inventory["uninterrupted"]["semantic_projection"]
    if any(
        inventory[role]["semantic_projection"] != reference for role in RUN_ROLES[1:]
    ):
        raise Wave7SequenceError(
            "non-run/non-resume config semantics differ across roles",
            code="wave7_sequence.config_semantic_mismatch",
        )
    return inventory


def _build_plan_payload(request_path: Path) -> dict[str, Any]:
    request_path = _absolute_path(
        str(request_path.expanduser().resolve()), owner="request", must_exist=True
    )
    if request_path != R7_REQUEST_PATH:
        raise Wave7SequenceError(
            "request path is outside the exact r7 namespace",
            code="wave7_sequence.r7_namespace",
        )
    request = _strict_json_file(request_path, owner="sequence request")
    _exact_fields(
        request,
        {
            "schema",
            "status",
            "request_payload_sha256",
            "amendment",
            "legacy_r4_failure",
            "predecessor_sequence_failure",
            "predecessor_preflight_failure",
            "determinism_preflight_plan",
            "determinism_preflight",
            "runtime_receipt",
            "cache_input_attestation",
            "model_input_attestation",
            "source_inventory",
            "config_paths",
            "provenance_sha256",
            "environment",
            "run_roots",
            "targets",
            "commands",
            "oracles",
            "policy",
        },
        owner="sequence request",
    )
    if request["schema"] != REQUEST_SCHEMA or request["status"] != "prepared":
        raise Wave7SequenceError(
            "sequence request schema is unsupported",
            code="wave7_sequence.json_schema",
        )
    request_payload_sha256 = _validate_request_digest(request)
    provenance_sha256 = _digest(request["provenance_sha256"], owner="provenance sha256")

    amendment = request["amendment"]
    if not isinstance(amendment, dict):
        raise Wave7SequenceError(
            "amendment must be an object", code="wave7_sequence.json_schema"
        )
    _validate_r7_amendment_binding(amendment)

    legacy = request["legacy_r4_failure"]
    predecessor = request["predecessor_sequence_failure"]
    predecessor_preflight = request["predecessor_preflight_failure"]
    determinism_plan = request["determinism_preflight_plan"]
    determinism = request["determinism_preflight"]
    runtime = request["runtime_receipt"]
    if not all(
        isinstance(value, dict)
        for value in (
            legacy,
            predecessor,
            predecessor_preflight,
            determinism_plan,
            determinism,
            runtime,
        )
    ):
        raise Wave7SequenceError(
            "receipt bindings must be objects", code="wave7_sequence.json_schema"
        )
    for binding, owner in (
        (determinism_plan, "determinism preflight plan"),
        (determinism, "determinism preflight terminal"),
        (runtime, "runtime receipt"),
    ):
        _require_r7_external_path(binding, owner=owner)
    legacy_payload = _verify_receipt_binding(legacy, owner="legacy r4 failure")
    if (
        legacy["schema"] != LEGACY_R4_RECEIPT_SCHEMA
        or legacy["status"] != "failed"
        or legacy["file_sha256"] != FROZEN_R4_FAILURE_FILE_SHA256
        or legacy["payload_sha256"] != FROZEN_R4_FAILURE_PAYLOAD_SHA256
        or legacy_payload.get("mismatches") in (None, [])
    ):
        raise Wave7SequenceError(
            "legacy r4 evidence must remain an immutable failure",
            code="wave7_sequence.legacy_r4",
        )
    _validate_predecessor_sequence_failure(predecessor)
    _validate_predecessor_preflight_failure(predecessor_preflight)
    _validate_determinism_preflight_evidence(
        plan_binding=determinism_plan, terminal_binding=determinism
    )
    runtime_payload = _verify_receipt_binding(runtime, owner="runtime receipt")
    if runtime["schema"] != RUNTIME_ADMISSION_SCHEMA or runtime["status"] != "passed":
        raise Wave7SequenceError(
            "runtime receipt binding is not the strict r5 admission",
            code="wave7_sequence.runtime_admission",
        )
    _validate_runtime_admission_payload(
        runtime_payload,
        expected_provenance_sha256=provenance_sha256,
        require_live_provenance=True,
    )
    runtime_provenance = runtime_payload.get("provenance")
    repository = (
        runtime_provenance.get("repository")
        if isinstance(runtime_provenance, dict)
        else None
    )
    execution_digest = (
        repository.get("execution_relevant_digest")
        if isinstance(repository, dict)
        else None
    )
    compare_provenance_sha256 = (
        execution_digest.get("value")
        if isinstance(execution_digest, dict)
        and execution_digest.get("status") == "available"
        else None
    )
    if (
        not isinstance(compare_provenance_sha256, str)
        or len(compare_provenance_sha256) != 64
        or any(character not in "0123456789abcdef" for character in compare_provenance_sha256)
    ):
        raise Wave7SequenceError(
            "runtime receipt omits the available execution-relevant provenance digest",
            code="wave7_sequence.runtime_admission",
        )

    source_paths, source_inventory = _verify_source_inventory(
        request["source_inventory"]
    )
    controller_path = Path(__file__).resolve()
    if controller_path not in source_paths:
        raise Wave7SequenceError(
            "source inventory omits the sequence controller",
            code="wave7_sequence.source_inventory",
        )
    required_sources = {
        *REQUIRED_SOURCE_PATHS[:4],
        REQUEST_PRODUCER_SOURCE,
        INPUT_ATTESTATION_SOURCE,
        MODEL_WEIGHT_IDENTITY_SOURCE,
    }
    missing_required_sources = required_sources - set(source_paths)
    if missing_required_sources:
        raise Wave7SequenceError(
            "source path inventory omits a required Wave 7 owner",
            code="wave7_sequence.source_inventory",
            context={"missing": sorted(str(path) for path in missing_required_sources)},
        )
    command_owners = {
        PYTHON_EXECUTABLE,
        TRAIN_ENTRYPOINT,
        INTERRUPT_CONTROLLER,
        COMPARATOR_V2,
        REQUEST_PRODUCER_SOURCE,
        INPUT_ATTESTATION_SOURCE,
    }
    if ACCELERATE_EXECUTABLE is not None:
        command_owners.add(ACCELERATE_EXECUTABLE)
    missing_command_owners = command_owners - set(source_paths)
    if missing_command_owners:
        raise Wave7SequenceError(
            "source inventory omits a frozen command owner",
            code="wave7_sequence.command_owner",
            context={"missing": sorted(str(path) for path in missing_command_owners)},
        )
    v1_comparator = REQUIRED_SOURCE_PATHS[0]
    if _sha256_file(v1_comparator) != FROZEN_V1_COMPARATOR_SHA256:
        raise Wave7SequenceError(
            "the frozen v1 comparator identity drifted",
            code="wave7_sequence.source_inventory",
            context={"path": str(v1_comparator)},
        )
    v2_comparator = REQUIRED_SOURCE_PATHS[1]
    if _sha256_file(v2_comparator) != FROZEN_V2_COMPARATOR_SHA256:
        raise Wave7SequenceError(
            "the frozen v2 comparator identity drifted",
            code="wave7_sequence.source_inventory",
            context={"path": str(v2_comparator)},
        )
    determinism_source = REQUIRED_SOURCE_PATHS[3]
    if _sha256_file(determinism_source) != FROZEN_DETERMINISM_PREFLIGHT_SOURCE_SHA256:
        raise Wave7SequenceError(
            "the frozen determinism preflight source identity drifted",
            code="wave7_sequence.source_inventory",
            context={"path": str(determinism_source)},
        )
    if _sha256_file(MODEL_WEIGHT_IDENTITY_SOURCE) != FROZEN_PARITY_SOURCE_SHA256:
        raise Wave7SequenceError(
            "the frozen parallel model-weight identity owner drifted",
            code="wave7_sequence.source_inventory",
            context={"path": str(MODEL_WEIGHT_IDENTITY_SOURCE)},
        )
    for source_path, expected_sha256, owner in (
        (
            INPUT_ATTESTATION_SOURCE,
            FROZEN_INPUT_ATTESTATION_SOURCE_SHA256,
            "input attestation",
        ),
        (
            REQUEST_PRODUCER_SOURCE,
            FROZEN_REQUEST_PRODUCER_SOURCE_SHA256,
            "request producer",
        ),
    ):
        if _sha256_file(source_path) != expected_sha256:
            raise Wave7SequenceError(
                f"the frozen {owner} owner drifted",
                code="wave7_sequence.source_inventory",
                context={"path": str(source_path)},
            )
    sources = [_file_identity(path) for path in sorted(source_paths)]

    config_values = request["config_paths"]
    if not isinstance(config_values, dict):
        raise Wave7SequenceError(
            "config_paths must be an object", code="wave7_sequence.json_schema"
        )
    _exact_fields(config_values, set(RUN_ROLES), owner="config_paths")
    config_paths = {
        role: _absolute_path(
            config_values[role], owner=f"config {role}", must_exist=True
        )
        for role in RUN_ROLES
    }
    configs = {role: _file_identity(config_paths[role]) for role in RUN_ROLES}

    cache_binding = request["cache_input_attestation"]
    model_binding = request["model_input_attestation"]
    if not isinstance(cache_binding, dict) or not isinstance(model_binding, dict):
        raise Wave7SequenceError(
            "cache and model input attestation bindings must be objects",
            code="wave7_sequence.json_schema",
        )
    cache_payload = _verify_payload_binding(
        cache_binding,
        owner="cache input attestation",
        digest_field="attestation_sha256",
    )
    model_payload = _verify_payload_binding(
        model_binding,
        owner="model input attestation",
        digest_field="attestation_sha256",
    )
    if (
        cache_payload.get("schema") != CACHE_ATTESTATION_SCHEMA
        or cache_payload.get("status") != "passed"
        or model_payload.get("schema") != MODEL_ATTESTATION_SCHEMA
        or model_payload.get("status") != "passed"
    ):
        raise Wave7SequenceError(
            "cache/model input attestation schema or status is unsupported",
            code="wave7_sequence.input_attestation",
        )
    input_validation = _validate_input_attestations(
        cache_payload=cache_payload,
        model_payload=model_payload,
        config_paths=config_paths,
    )
    cache_root = _absolute_path(
        cache_payload.get("cache_root"), owner="cache root", must_exist=True
    )
    model_root = _absolute_path(
        model_payload.get("model_root"), owner="model root", must_exist=True
    )
    if (
        cache_root != R7_PRIVATE_CACHE_ROOT
        or not cache_root.is_dir()
        or not model_root.is_dir()
        or _paths_overlap(cache_root, model_root)
    ):
        raise Wave7SequenceError(
            "cache and model roots must be distinct directories",
            code="wave7_sequence.path_topology",
        )
    input_attestations = {
        "cache": {
            "binding": dict(cache_binding),
            "payload": cache_payload,
            "root": _directory_identity(cache_root),
        },
        "model": {
            "binding": dict(model_binding),
            "payload": model_payload,
            "root": _directory_identity(model_root),
        },
        "validation": input_validation,
    }
    environment = request["environment"]
    if not isinstance(environment, dict):
        raise Wave7SequenceError(
            "environment must be one object", code="wave7_sequence.environment"
        )
    expected_environment = {
        **ENVIRONMENT,
        "COORDEXP_SWIFT_PACK_CACHE_ROOT": str(cache_root),
    }
    if environment != expected_environment:
        raise Wave7SequenceError(
            "r7 environment is not exact",
            code="wave7_sequence.environment",
            context={"expected": expected_environment, "observed": environment},
        )

    run_values = request["run_roots"]
    target_values = request["targets"]
    if not isinstance(run_values, dict) or not isinstance(target_values, dict):
        raise Wave7SequenceError(
            "run roots and targets must be objects", code="wave7_sequence.json_schema"
        )
    roots, targets = _validate_topology(run_values, target_values, require_absent=True)
    resolved_configs = _resolved_config_inventory(
        config_paths, roots=roots, require_absent_runs=True
    )

    command_values = request["commands"]
    if not isinstance(command_values, dict):
        raise Wave7SequenceError(
            "commands must be an object", code="wave7_sequence.argv"
        )
    _exact_fields(command_values, set(PHASE_ORDER), owner="commands")
    commands = {
        phase: _validate_argv(command_values[phase], phase=phase)
        for phase in PHASE_ORDER
    }

    oracle_values = request["oracles"]
    policy_values = request["policy"]
    if not isinstance(oracle_values, dict) or not isinstance(policy_values, dict):
        raise Wave7SequenceError(
            "oracles and policy must be objects", code="wave7_sequence.json_schema"
        )
    oracles = _validate_oracles(oracle_values)
    policy = _validate_policy(policy_values)
    _validate_exact_commands(
        commands,
        config_paths=config_paths,
        roots=roots,
        targets=targets,
        policy=policy,
        provenance_sha256=compare_provenance_sha256,
    )

    owner_paths = {
        request_path,
        controller_path,
        *source_paths,
        *(Path(identity["path"]) for identity in configs.values()),
        *(
            Path(source["path"])
            for projection in resolved_configs.values()
            for source in projection["sources"]
        ),
        *(
            Path(binding["path"])
            for binding in (
                amendment,
                legacy,
                predecessor,
                determinism_plan,
                determinism,
                runtime,
                cache_binding,
                model_binding,
            )
        ),
        *(
            Path(predecessor_preflight[name]["path"])
            for name in ("plan", "attempt_marker", "terminal_receipt")
        ),
    }
    owner_directories = {
        cache_root,
        model_root,
        *(path.parent for path in roots.values()),
        *(path.parent for path in targets.values()),
    }
    ownership = {
        "files": [_file_identity(path) for path in sorted(owner_paths)],
        "directories": [
            _directory_identity(path) for path in sorted(owner_directories)
        ],
    }
    request_identity = {
        "path": str(request_path),
        "file_sha256": _sha256_file(request_path),
        "payload_sha256": request_payload_sha256,
    }
    payload = {
        "schema": PLAN_SCHEMA,
        "request": request_identity,
        "controller_source": _file_identity(controller_path),
        "amendment": dict(amendment),
        "legacy_r4_failure": dict(legacy),
        "predecessor_sequence_failure": dict(predecessor),
        "predecessor_preflight_failure": dict(predecessor_preflight),
        "determinism_preflight_plan": dict(determinism_plan),
        "determinism_preflight": dict(determinism),
        "runtime_receipt": dict(runtime),
        "sources": sources,
        "request_source_inventory": source_inventory,
        "configs": configs,
        "resolved_configs": resolved_configs,
        "input_attestations": input_attestations,
        "provenance_sha256": provenance_sha256,
        "environment": expected_environment,
        "run_roots": {role: str(roots[role]) for role in RUN_ROLES},
        "targets": {name: str(targets[name]) for name in TARGET_NAMES},
        "commands": commands,
        "command_sha256": {
            phase: _sha256_bytes(canonical_json_bytes(commands[phase]))
            for phase in PHASE_ORDER
        },
        "oracles": oracles,
        "policy": policy,
        "ownership": ownership,
    }
    payload["plan_payload_sha256"] = _sha256_bytes(canonical_json_bytes(payload))
    return payload


def build_plan(request_path: Path, plan_path: Path) -> dict[str, Any]:
    if plan_path.expanduser().resolve() != R7_PLAN_PATH:
        raise Wave7SequenceError(
            "plan path is outside the exact r7 namespace",
            code="wave7_sequence.r7_namespace",
        )
    target = assert_absent_artifact_target(plan_path)
    payload = _build_plan_payload(request_path)
    _validate_r7_amendment_binding(payload["amendment"])
    _validate_predecessor_sequence_failure(payload["predecessor_sequence_failure"])
    _validate_predecessor_preflight_failure(payload["predecessor_preflight_failure"])
    write_strict_json_atomic(target, payload)
    return payload


def prepare(request_path: Path, plan_path: Path) -> int:
    try:
        build_plan(request_path, plan_path)
    except BaseException as exc:
        print(
            f"{getattr(exc, 'code', 'wave7_sequence.prepare')}: {exc}",
            file=sys.stderr,
        )
        return 2
    return 0


def _load_plan(plan_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = _absolute_path(
        str(plan_path.expanduser().resolve()), owner="plan", must_exist=True
    )
    if plan_path != R7_PLAN_PATH:
        raise Wave7SequenceError(
            "plan path is outside the exact r7 namespace",
            code="wave7_sequence.r7_namespace",
        )
    plan = _strict_json_file(plan_path, owner="sequence plan")
    expected_fields = {
        "schema",
        "request",
        "controller_source",
        "amendment",
        "legacy_r4_failure",
        "predecessor_sequence_failure",
        "predecessor_preflight_failure",
        "determinism_preflight_plan",
        "determinism_preflight",
        "runtime_receipt",
        "sources",
        "request_source_inventory",
        "configs",
        "resolved_configs",
        "input_attestations",
        "provenance_sha256",
        "environment",
        "run_roots",
        "targets",
        "commands",
        "command_sha256",
        "oracles",
        "policy",
        "ownership",
        "plan_payload_sha256",
    }
    _exact_fields(plan, expected_fields, owner="sequence plan")
    if plan["schema"] != PLAN_SCHEMA:
        raise Wave7SequenceError(
            "sequence plan schema is unsupported", code="wave7_sequence.json_schema"
        )
    expected_payload_sha256 = _digest(
        plan["plan_payload_sha256"], owner="plan payload sha256"
    )
    unsigned = dict(plan)
    unsigned.pop("plan_payload_sha256")
    if _sha256_bytes(canonical_json_bytes(unsigned)) != expected_payload_sha256:
        raise Wave7SequenceError(
            "sequence plan payload digest is invalid",
            code="wave7_sequence.plan_digest",
        )
    plan_identity = {
        "path": str(plan_path),
        "file_sha256": _sha256_file(plan_path),
        "payload_sha256": expected_payload_sha256,
    }
    return plan, plan_identity


def _revalidate_plan_inputs(
    plan: Mapping[str, Any], *, plan_identity: Mapping[str, Any] | None = None
) -> None:
    if plan_identity is not None:
        plan_path = _absolute_path(
            plan_identity["path"], owner="plan identity path", must_exist=True
        )
        if _sha256_file(plan_path) != plan_identity["file_sha256"]:
            raise Wave7SequenceError(
                "sequence plan file drifted after admission",
                code="wave7_sequence.identity_drift",
                context={"path": str(plan_path)},
            )
    controller = plan["controller_source"]
    if not isinstance(controller, dict):
        raise Wave7SequenceError(
            "controller identity is malformed", code="wave7_sequence.json_schema"
        )
    _verify_file_identity(controller)
    sources = plan["sources"]
    configs = plan["configs"]
    resolved_configs = plan["resolved_configs"]
    ownership = plan["ownership"]
    if (
        not isinstance(sources, list)
        or not isinstance(configs, dict)
        or not isinstance(resolved_configs, dict)
        or not isinstance(ownership, dict)
    ):
        raise Wave7SequenceError(
            "plan identity inventories are malformed",
            code="wave7_sequence.json_schema",
        )
    for identity in sources:
        if not isinstance(identity, dict):
            raise Wave7SequenceError(
                "source identity is malformed", code="wave7_sequence.json_schema"
            )
        _verify_file_identity(identity)
    request_source_inventory = plan["request_source_inventory"]
    source_paths, observed_request_sources = _verify_source_inventory(
        request_source_inventory
    )
    if observed_request_sources != request_source_inventory or set(source_paths) != {
        Path(identity["path"]) for identity in sources
    }:
        raise Wave7SequenceError(
            "request and plan source inventories disagree",
            code="wave7_sequence.source_inventory",
        )
    _exact_fields(configs, set(RUN_ROLES), owner="plan configs")
    for identity in configs.values():
        if not isinstance(identity, dict):
            raise Wave7SequenceError(
                "config identity is malformed", code="wave7_sequence.json_schema"
            )
        _verify_file_identity(identity)
    run_roots = plan["run_roots"]
    if not isinstance(run_roots, dict):
        raise Wave7SequenceError(
            "plan run roots are malformed", code="wave7_sequence.json_schema"
        )
    _exact_fields(run_roots, set(RUN_ROLES), owner="plan run roots")
    observed_resolved_configs = _resolved_config_inventory(
        {role: Path(configs[role]["path"]) for role in RUN_ROLES},
        roots={role: Path(run_roots[role]) for role in RUN_ROLES},
        require_absent_runs=False,
    )
    if observed_resolved_configs != resolved_configs:
        raise Wave7SequenceError(
            "resolved training config projection drifted after admission",
            code="wave7_sequence.identity_drift",
        )
    receipt_payloads: dict[str, dict[str, Any]] = {}
    for name in ("legacy_r4_failure", "runtime_receipt"):
        binding = plan[name]
        if not isinstance(binding, dict):
            raise Wave7SequenceError(
                "receipt binding is malformed", code="wave7_sequence.json_schema"
            )
        receipt_payloads[name] = _verify_receipt_binding(binding, owner=name)
    predecessor = plan["predecessor_sequence_failure"]
    if not isinstance(predecessor, dict):
        raise Wave7SequenceError(
            "predecessor r5 binding is malformed",
            code="wave7_sequence.json_schema",
        )
    _validate_predecessor_sequence_failure(predecessor)
    predecessor_preflight = plan["predecessor_preflight_failure"]
    if not isinstance(predecessor_preflight, dict):
        raise Wave7SequenceError(
            "predecessor r6 preflight binding is malformed",
            code="wave7_sequence.json_schema",
        )
    _validate_predecessor_preflight_failure(predecessor_preflight)
    amendment = plan["amendment"]
    if not isinstance(amendment, dict):
        raise Wave7SequenceError(
            "amendment binding is malformed", code="wave7_sequence.json_schema"
        )
    _validate_r7_amendment_binding(amendment)
    runtime_payload = receipt_payloads["runtime_receipt"]
    for name, owner in (
        ("determinism_preflight_plan", "determinism preflight plan"),
        ("determinism_preflight", "determinism preflight terminal"),
        ("runtime_receipt", "runtime receipt"),
    ):
        _require_r7_external_path(plan[name], owner=owner)
    if plan["runtime_receipt"].get("schema") != RUNTIME_ADMISSION_SCHEMA:
        raise Wave7SequenceError(
            "runtime receipt binding is not the strict r5 admission",
            code="wave7_sequence.runtime_admission",
        )
    _validate_runtime_admission_payload(
        runtime_payload,
        expected_provenance_sha256=_digest(
            plan["provenance_sha256"], owner="plan provenance sha256"
        ),
        require_live_provenance=True,
    )
    determinism_plan = plan["determinism_preflight_plan"]
    determinism_terminal = plan["determinism_preflight"]
    if not isinstance(determinism_plan, dict) or not isinstance(
        determinism_terminal, dict
    ):
        raise Wave7SequenceError(
            "determinism preflight bindings are malformed",
            code="wave7_sequence.json_schema",
        )
    _validate_determinism_preflight_evidence(
        plan_binding=determinism_plan,
        terminal_binding=determinism_terminal,
    )
    attestations = plan["input_attestations"]
    if not isinstance(attestations, dict):
        raise Wave7SequenceError(
            "input attestations are malformed", code="wave7_sequence.json_schema"
        )
    _exact_fields(
        attestations, {"cache", "model", "validation"}, owner="input attestations"
    )
    live_payloads: dict[str, dict[str, Any]] = {}
    for name, digest_field in (
        ("cache", "attestation_sha256"),
        ("model", "attestation_sha256"),
    ):
        row = attestations[name]
        if (
            not isinstance(row, dict)
            or set(row) != {"binding", "payload", "root"}
            or not isinstance(row["binding"], dict)
            or not isinstance(row["payload"], dict)
            or not isinstance(row["root"], dict)
        ):
            raise Wave7SequenceError(
                f"{name} input attestation is malformed",
                code="wave7_sequence.json_schema",
            )
        _verify_directory_identity(row["root"])
        live_payloads[name] = _verify_payload_binding(
            row["binding"],
            owner=f"{name} input attestation",
            digest_field=digest_field,
        )
        if live_payloads[name] != row["payload"]:
            raise Wave7SequenceError(
                f"{name} input attestation payload drifted",
                code="wave7_sequence.input_attestation",
            )
    if Path(attestations["cache"]["root"]["path"]) != R7_PRIVATE_CACHE_ROOT:
        raise Wave7SequenceError(
            "live cache identity is outside the exact r7 namespace",
            code="wave7_sequence.r7_namespace",
        )
    observed_validation = _validate_input_attestations(
        cache_payload=live_payloads["cache"],
        model_payload=live_payloads["model"],
        config_paths={role: Path(configs[role]["path"]) for role in RUN_ROLES},
    )
    if observed_validation != attestations["validation"]:
        raise Wave7SequenceError(
            "input attestation validation projection drifted",
            code="wave7_sequence.input_attestation",
        )
    _exact_fields(ownership, {"files", "directories"}, owner="ownership")
    for identity in ownership["files"]:
        _verify_file_identity(identity)
    for identity in ownership["directories"]:
        _verify_directory_identity(identity)


def _validate_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    expected = plan["environment"]
    if not isinstance(expected, dict):
        raise Wave7SequenceError(
            "plan environment is malformed", code="wave7_sequence.environment"
        )
    cache_root = plan["input_attestations"]["cache"]["root"]["path"]
    frozen = {**ENVIRONMENT, "COORDEXP_SWIFT_PACK_CACHE_ROOT": cache_root}
    if expected != frozen:
        raise Wave7SequenceError(
            "plan environment is not the frozen r7 environment",
            code="wave7_sequence.environment",
        )
    observed = {name: os.environ.get(name) for name in expected}
    if observed != expected:
        raise Wave7SequenceError(
            "process environment does not match the r5 plan",
            code="wave7_sequence.environment",
            context={"expected": expected, "observed": observed},
        )
    return dict(expected)


def _bounded_nvidia_csv(command: list[str], *, owner: str) -> list[str]:
    with (
        tempfile.TemporaryFile() as stdout_sink,
        tempfile.TemporaryFile() as stderr_sink,
    ):
        try:
            subprocess.run(
                command,
                check=True,
                stdout=stdout_sink,
                stderr=stderr_sink,
                timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise Wave7SequenceError(
                f"{owner} cannot be sampled",
                code="wave7_sequence.gpu_inventory",
                context={"error_type": type(exc).__name__, "command": command},
            ) from exc
        for stream, sink in (("stdout", stdout_sink), ("stderr", stderr_sink)):
            observed_bytes = sink.seek(0, os.SEEK_END)
            if observed_bytes > MAX_GPU_CSV_BYTES:
                raise Wave7SequenceError(
                    f"{owner} {stream} exceeds the CSV evidence bound",
                    code="wave7_sequence.gpu_inventory",
                    context={
                        "stream": stream,
                        "maximum_bytes": MAX_GPU_CSV_BYTES,
                        "observed_bytes": observed_bytes,
                    },
                )
        stdout_sink.seek(0)
        try:
            stdout = stdout_sink.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Wave7SequenceError(
                f"{owner} stdout is not UTF-8",
                code="wave7_sequence.gpu_inventory",
                context={"stream": "stdout", "error_type": type(exc).__name__},
            ) from exc
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _gpu_compute_sample(
    *,
    memory_total_mib: int,
    memory_used_ceiling_mib: int,
    memory_headroom_floor_mib: int,
) -> dict[str, Any]:
    gpu_lines = _bounded_nvidia_csv(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        owner="GPU device inventory",
    )
    compute_lines = _bounded_nvidia_csv(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        owner="GPU compute inventory",
    )
    gpu_inventory: list[dict[str, Any]] = []
    for line in gpu_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5 or any(not field for field in fields):
            raise Wave7SequenceError(
                "GPU device inventory row is malformed",
                code="wave7_sequence.gpu_inventory",
                context={"row": line},
            )
        (
            index_text,
            gpu_uuid,
            memory_total_text,
            memory_used_text,
            utilization_text,
        ) = fields
        if (
            not index_text.isdigit()
            or not memory_total_text.isdigit()
            or not memory_used_text.isdigit()
            or not utilization_text.isdigit()
        ):
            raise Wave7SequenceError(
                "GPU device inventory values are malformed",
                code="wave7_sequence.gpu_inventory",
                context={"row": line},
            )
        index = int(index_text)
        observed_memory_total_mib = int(memory_total_text)
        memory_used_mib = int(memory_used_text)
        memory_headroom_mib = observed_memory_total_mib - memory_used_mib
        utilization_gpu_percent = int(utilization_text)
        if not 0 <= utilization_gpu_percent <= 100:
            raise Wave7SequenceError(
                "GPU utilization observation is outside its numeric range",
                code="wave7_sequence.gpu_inventory",
                context={"row": line},
            )
        if (
            observed_memory_total_mib != memory_total_mib
            or memory_used_mib > memory_used_ceiling_mib
            or memory_headroom_mib < memory_headroom_floor_mib
        ):
            raise Wave7SequenceError(
                "GPU memory capacity, use, or headroom violates shared admission",
                code="wave7_sequence.gpu_memory_ceiling",
                context={
                    "gpu_index": index,
                    "gpu_uuid": gpu_uuid,
                    "memory_total_mib": observed_memory_total_mib,
                    "memory_used_mib": memory_used_mib,
                    "memory_used_ceiling_mib": memory_used_ceiling_mib,
                    "memory_headroom_mib": memory_headroom_mib,
                    "memory_headroom_floor_mib": memory_headroom_floor_mib,
                },
            )
        gpu_inventory.append(
            {
                "index": index,
                "gpu_uuid": gpu_uuid,
                "memory_total_mib": observed_memory_total_mib,
                "memory_used_mib": memory_used_mib,
                "memory_headroom_mib": memory_headroom_mib,
                "utilization_gpu_percent": utilization_gpu_percent,
            }
        )
    expected_indices = list(range(GPU_DEVICE_COUNT))
    observed_indices = [row["index"] for row in gpu_inventory]
    observed_uuids = [row["gpu_uuid"] for row in gpu_inventory]
    if (
        observed_indices != expected_indices
        or len(set(observed_uuids)) != GPU_DEVICE_COUNT
    ):
        raise Wave7SequenceError(
            "GPU device inventory must be the exact ordered 0-7 inventory",
            code="wave7_sequence.gpu_inventory",
            context={"indices": observed_indices, "gpu_uuids": observed_uuids},
        )

    compute_inventory: list[dict[str, Any]] = []
    for line in compute_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise Wave7SequenceError(
                "GPU compute inventory row is malformed",
                code="wave7_sequence.gpu_inventory",
                context={"row": line},
            )
        gpu_uuid, driver_pid_text = fields
        if gpu_uuid not in observed_uuids or int(driver_pid_text) <= 0:
            raise Wave7SequenceError(
                "GPU compute inventory row is not bound to the device inventory",
                code="wave7_sequence.gpu_inventory",
                context={"row": line},
            )
        compute_inventory.append(
            {"gpu_uuid": gpu_uuid, "driver_pid": int(driver_pid_text)}
        )
    compute_inventory.sort(key=lambda row: (row["gpu_uuid"], row["driver_pid"]))
    compute_keys = [(row["gpu_uuid"], row["driver_pid"]) for row in compute_inventory]
    if len(compute_keys) != len(set(compute_keys)):
        raise Wave7SequenceError(
            "GPU compute inventory contains duplicate rows",
            code="wave7_sequence.gpu_inventory",
        )
    return {
        "checked_at": _utc_now(),
        "gpu_inventory": gpu_inventory,
        "compute_inventory": compute_inventory,
    }


def _device_identity(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"index": row["index"], "gpu_uuid": row["gpu_uuid"]}
        for row in sample["gpu_inventory"]
    ]


def _validate_gpu_sample(
    value: Any,
    *,
    memory_total_mib: int,
    memory_used_ceiling_mib: int,
    memory_headroom_floor_mib: int,
    owner: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Wave7SequenceError(
            f"{owner} must be one object", code="wave7_sequence.json_schema"
        )
    _exact_fields(
        value,
        {"checked_at", "gpu_inventory", "compute_inventory"},
        owner=owner,
    )
    if not isinstance(value["checked_at"], str):
        raise Wave7SequenceError(
            f"{owner} timestamp is malformed", code="wave7_sequence.json_schema"
        )
    gpu_inventory = value["gpu_inventory"]
    compute_inventory = value["compute_inventory"]
    if not isinstance(gpu_inventory, list) or not isinstance(compute_inventory, list):
        raise Wave7SequenceError(
            f"{owner} inventories must be lists", code="wave7_sequence.json_schema"
        )
    if len(gpu_inventory) != GPU_DEVICE_COUNT:
        raise Wave7SequenceError(
            f"{owner} must bind exactly eight GPU rows",
            code="wave7_sequence.gpu_inventory",
        )
    gpu_uuids: list[str] = []
    for expected_index, row in enumerate(gpu_inventory):
        if not isinstance(row, dict):
            raise Wave7SequenceError(
                f"{owner} GPU row must be an object",
                code="wave7_sequence.json_schema",
            )
        _exact_fields(
            row,
            {
                "index",
                "gpu_uuid",
                "memory_total_mib",
                "memory_used_mib",
                "memory_headroom_mib",
                "utilization_gpu_percent",
            },
            owner=f"{owner} GPU row",
        )
        observed_memory_total_mib = row["memory_total_mib"]
        memory_used_mib = row["memory_used_mib"]
        memory_headroom_mib = row["memory_headroom_mib"]
        utilization = row["utilization_gpu_percent"]
        if (
            row["index"] != expected_index
            or isinstance(row["index"], bool)
            or not isinstance(row["index"], int)
            or not isinstance(row["gpu_uuid"], str)
            or not row["gpu_uuid"]
            or isinstance(observed_memory_total_mib, bool)
            or not isinstance(observed_memory_total_mib, int)
            or observed_memory_total_mib != memory_total_mib
            or isinstance(memory_used_mib, bool)
            or not isinstance(memory_used_mib, int)
            or memory_used_mib < 0
            or memory_used_mib > memory_used_ceiling_mib
            or isinstance(memory_headroom_mib, bool)
            or not isinstance(memory_headroom_mib, int)
            or memory_headroom_mib != observed_memory_total_mib - memory_used_mib
            or memory_headroom_mib < memory_headroom_floor_mib
            or isinstance(utilization, bool)
            or not isinstance(utilization, int)
            or not 0 <= utilization <= 100
        ):
            raise Wave7SequenceError(
                f"{owner} GPU row is invalid",
                code="wave7_sequence.gpu_inventory",
                context={"row": row},
            )
        gpu_uuids.append(row["gpu_uuid"])
    if len(set(gpu_uuids)) != GPU_DEVICE_COUNT:
        raise Wave7SequenceError(
            f"{owner} GPU UUID inventory is not unique",
            code="wave7_sequence.gpu_inventory",
        )
    compute_keys: list[tuple[str, int]] = []
    for row in compute_inventory:
        if not isinstance(row, dict):
            raise Wave7SequenceError(
                f"{owner} compute row must be an object",
                code="wave7_sequence.json_schema",
            )
        _exact_fields(row, {"gpu_uuid", "driver_pid"}, owner=f"{owner} compute row")
        if (
            row["gpu_uuid"] not in gpu_uuids
            or isinstance(row["driver_pid"], bool)
            or not isinstance(row["driver_pid"], int)
            or row["driver_pid"] <= 0
        ):
            raise Wave7SequenceError(
                f"{owner} compute row is invalid",
                code="wave7_sequence.gpu_inventory",
                context={"row": row},
            )
        compute_keys.append((row["gpu_uuid"], row["driver_pid"]))
    if compute_keys != sorted(set(compute_keys)):
        raise Wave7SequenceError(
            f"{owner} compute inventory is not unique and sorted",
            code="wave7_sequence.gpu_inventory",
        )
    return value


def _validate_shared_gpu_baseline(
    value: Any, *, expected_stability_seconds: float | None = None
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Wave7SequenceError(
            "shared GPU baseline must be one object",
            code="wave7_sequence.json_schema",
        )
    _exact_fields(
        value,
        {
            "mode",
            "stability_seconds",
            "observed_stability_seconds",
            "sample_monotonic_ns",
            "memory_total_mib",
            "memory_used_ceiling_mib",
            "memory_headroom_floor_mib",
            "device_inventory",
            "compute_inventory",
            "samples",
        },
        owner="shared GPU baseline",
    )
    stability_seconds = _positive_number(
        value["stability_seconds"], owner="GPU baseline stability seconds"
    )
    observed_stability_seconds = _positive_number(
        value["observed_stability_seconds"],
        owner="observed GPU baseline stability seconds",
    )
    sample_monotonic_ns = value["sample_monotonic_ns"]
    required_interval_ns = int(SHARED_GPU_BASELINE_STABILITY_SECONDS * 1_000_000_000)
    if (
        value["mode"] != SHARED_GPU_ADMISSION_MODE
        or stability_seconds != SHARED_GPU_BASELINE_STABILITY_SECONDS
        or not isinstance(sample_monotonic_ns, list)
        or len(sample_monotonic_ns) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in sample_monotonic_ns
        )
        or sample_monotonic_ns[1] - sample_monotonic_ns[0] < required_interval_ns
        or observed_stability_seconds
        != (sample_monotonic_ns[1] - sample_monotonic_ns[0]) / 1_000_000_000
        or isinstance(value["memory_total_mib"], bool)
        or not isinstance(value["memory_total_mib"], int)
        or value["memory_total_mib"] != SHARED_GPU_MEMORY_TOTAL_MIB
        or isinstance(value["memory_used_ceiling_mib"], bool)
        or not isinstance(value["memory_used_ceiling_mib"], int)
        or value["memory_used_ceiling_mib"] != SHARED_GPU_MEMORY_USED_CEILING_MIB
        or isinstance(value["memory_headroom_floor_mib"], bool)
        or not isinstance(value["memory_headroom_floor_mib"], int)
        or value["memory_headroom_floor_mib"] != SHARED_GPU_MEMORY_HEADROOM_FLOOR_MIB
        or (
            expected_stability_seconds is not None
            and stability_seconds != expected_stability_seconds
        )
    ):
        raise Wave7SequenceError(
            "shared GPU baseline policy is not the frozen plan policy",
            code="wave7_sequence.gpu_baseline",
        )
    samples = value["samples"]
    if not isinstance(samples, list) or len(samples) != 2:
        raise Wave7SequenceError(
            "shared GPU baseline requires exactly two samples",
            code="wave7_sequence.gpu_baseline",
        )
    first = _validate_gpu_sample(
        samples[0],
        memory_total_mib=SHARED_GPU_MEMORY_TOTAL_MIB,
        memory_used_ceiling_mib=SHARED_GPU_MEMORY_USED_CEILING_MIB,
        memory_headroom_floor_mib=SHARED_GPU_MEMORY_HEADROOM_FLOOR_MIB,
        owner="shared GPU baseline sample 1",
    )
    second = _validate_gpu_sample(
        samples[1],
        memory_total_mib=SHARED_GPU_MEMORY_TOTAL_MIB,
        memory_used_ceiling_mib=SHARED_GPU_MEMORY_USED_CEILING_MIB,
        memory_headroom_floor_mib=SHARED_GPU_MEMORY_HEADROOM_FLOOR_MIB,
        owner="shared GPU baseline sample 2",
    )
    if (
        value["device_inventory"] != _device_identity(first)
        or value["device_inventory"] != _device_identity(second)
        or value["compute_inventory"] != first["compute_inventory"]
        or value["compute_inventory"] != second["compute_inventory"]
    ):
        raise Wave7SequenceError(
            "shared GPU baseline samples and frozen inventories disagree",
            code="wave7_sequence.gpu_baseline",
        )
    return value


def _attest_shared_gpu_baseline(
    *,
    stability_seconds: float,
    memory_total_mib: int,
    memory_used_ceiling_mib: int,
    memory_headroom_floor_mib: int,
) -> dict[str, Any]:
    first = _gpu_compute_sample(
        memory_total_mib=memory_total_mib,
        memory_used_ceiling_mib=memory_used_ceiling_mib,
        memory_headroom_floor_mib=memory_headroom_floor_mib,
    )
    first_completed_ns = time.monotonic_ns()
    time.sleep(stability_seconds)
    second = _gpu_compute_sample(
        memory_total_mib=memory_total_mib,
        memory_used_ceiling_mib=memory_used_ceiling_mib,
        memory_headroom_floor_mib=memory_headroom_floor_mib,
    )
    second_completed_ns = time.monotonic_ns()
    observed_stability_seconds = (second_completed_ns - first_completed_ns) / 1e9
    first_devices = _device_identity(first)
    second_devices = _device_identity(second)
    if (
        first_devices != second_devices
        or first["compute_inventory"] != second["compute_inventory"]
    ):
        raise Wave7SequenceError(
            "shared GPU baseline inventory changed during the stability window",
            code="wave7_sequence.gpu_baseline",
            context={"first": first, "second": second},
        )
    baseline = {
        "mode": SHARED_GPU_ADMISSION_MODE,
        "stability_seconds": stability_seconds,
        "observed_stability_seconds": observed_stability_seconds,
        "sample_monotonic_ns": [first_completed_ns, second_completed_ns],
        "memory_total_mib": memory_total_mib,
        "memory_used_ceiling_mib": memory_used_ceiling_mib,
        "memory_headroom_floor_mib": memory_headroom_floor_mib,
        "device_inventory": first_devices,
        "compute_inventory": list(first["compute_inventory"]),
        "samples": [first, second],
    }
    return _validate_shared_gpu_baseline(
        baseline, expected_stability_seconds=stability_seconds
    )


def _assert_shared_gpu_subset(*, baseline: Mapping[str, Any]) -> dict[str, Any]:
    baseline = _validate_shared_gpu_baseline(baseline)
    sample = _gpu_compute_sample(
        memory_total_mib=int(baseline["memory_total_mib"]),
        memory_used_ceiling_mib=int(baseline["memory_used_ceiling_mib"]),
        memory_headroom_floor_mib=int(baseline["memory_headroom_floor_mib"]),
    )
    observed_devices = _device_identity(sample)
    if observed_devices != baseline["device_inventory"]:
        raise Wave7SequenceError(
            "GPU UUID/index inventory drifted from the sequence marker baseline",
            code="wave7_sequence.gpu_inventory",
            context={
                "baseline": baseline["device_inventory"],
                "observed": observed_devices,
            },
        )
    baseline_keys = {
        (row["gpu_uuid"], row["driver_pid"]) for row in baseline["compute_inventory"]
    }
    observed_keys = {
        (row["gpu_uuid"], row["driver_pid"]) for row in sample["compute_inventory"]
    }
    added = sorted(observed_keys - baseline_keys)
    if added:
        raise Wave7SequenceError(
            "new GPU compute processes appeared after the sequence marker",
            code="wave7_sequence.gpu_added_process",
            context={
                "added": [
                    {"gpu_uuid": gpu_uuid, "driver_pid": driver_pid}
                    for gpu_uuid, driver_pid in added
                ]
            },
        )
    return sample


def _validate_shared_gpu_recovery(
    value: Any,
    *,
    expected_stability_seconds: float,
    expected_phase: str,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Wave7SequenceError(
            "post-cleanup shared GPU recovery must be one object",
            code="wave7_sequence.gpu_recovery",
        )
    _exact_fields(
        value,
        {"phase", "stability_seconds", "sample_monotonic_ns", "samples"},
        owner="post-cleanup shared GPU recovery",
    )
    stability_seconds = _positive_number(
        value["stability_seconds"], owner="GPU recovery stability seconds"
    )
    sample_monotonic_ns = value["sample_monotonic_ns"]
    samples = value["samples"]
    if (
        value["phase"] != expected_phase
        or stability_seconds != expected_stability_seconds
        or stability_seconds != SHARED_GPU_POST_CLEANUP_STABILITY_SECONDS
        or not isinstance(sample_monotonic_ns, list)
        or len(sample_monotonic_ns) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in sample_monotonic_ns
        )
        or sample_monotonic_ns[1] - sample_monotonic_ns[0]
        < int(stability_seconds * 1_000_000_000)
        or not isinstance(samples, list)
        or len(samples) != 2
    ):
        raise Wave7SequenceError(
            "post-cleanup shared GPU recovery is not an exact two-sample sweep",
            code="wave7_sequence.gpu_recovery",
        )
    validated_samples = []
    for index, sample in enumerate(samples, start=1):
        validated_samples.append(
            _validate_gpu_sample(
                sample,
                memory_total_mib=SHARED_GPU_MEMORY_TOTAL_MIB,
                memory_used_ceiling_mib=SHARED_GPU_MEMORY_USED_CEILING_MIB,
                memory_headroom_floor_mib=SHARED_GPU_MEMORY_HEADROOM_FLOOR_MIB,
                owner=f"post-cleanup shared GPU recovery sample {index}",
            )
        )
    if baseline is not None:
        admitted_baseline = _validate_shared_gpu_baseline(baseline)
        baseline_keys = {
            (row["gpu_uuid"], row["driver_pid"])
            for row in admitted_baseline["compute_inventory"]
        }
        for sample in validated_samples:
            observed_keys = {
                (row["gpu_uuid"], row["driver_pid"])
                for row in sample["compute_inventory"]
            }
            if _device_identity(sample) != admitted_baseline[
                "device_inventory"
            ] or not observed_keys.issubset(baseline_keys):
                raise Wave7SequenceError(
                    "shared GPU recovery is not a subset of the marker baseline",
                    code="wave7_sequence.gpu_recovery",
                )
    return value


def _attest_shared_gpu_recovery(
    *, baseline: Mapping[str, Any], stability_seconds: float, phase: str
) -> dict[str, Any]:
    first = _assert_shared_gpu_subset(baseline=baseline)
    first_completed_ns = time.monotonic_ns()
    time.sleep(stability_seconds)
    second = _assert_shared_gpu_subset(baseline=baseline)
    recovery = {
        "phase": phase,
        "stability_seconds": stability_seconds,
        "sample_monotonic_ns": [first_completed_ns, time.monotonic_ns()],
        "samples": [first, second],
    }
    return _validate_shared_gpu_recovery(
        recovery,
        expected_stability_seconds=stability_seconds,
        expected_phase=phase,
        baseline=baseline,
    )


def _enable_child_subreaper() -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except (AttributeError, OSError) as exc:
        raise Wave7SequenceError(
            "child-subreaper support is unavailable",
            code="wave7_sequence.process_graph",
        ) from exc
    if result != 0:
        raise Wave7SequenceError(
            "child-subreaper activation failed",
            code="wave7_sequence.process_graph",
            context={"errno": ctypes.get_errno()},
        )


def _read_process_identity(pid: int) -> dict[str, Any] | None:
    try:
        encoded = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, UnicodeError) as exc:
        raise Wave7SequenceError(
            "process identity is unreadable",
            code="wave7_sequence.process_graph",
            context={"pid": pid},
        ) from exc
    close = encoded.rfind(")")
    fields = encoded[close + 2 :].split() if close >= 0 else []
    if len(fields) <= 19:
        raise Wave7SequenceError(
            "process identity is malformed",
            code="wave7_sequence.process_graph",
            context={"pid": pid},
        )
    try:
        return {
            "pid": pid,
            "parent_pid": int(fields[1]),
            "process_group_id": int(fields[2]),
            "session_id": int(fields[3]),
            "state": fields[0],
            "start_time_ticks": int(fields[19]),
            "depth": 0,
        }
    except (IndexError, ValueError) as exc:
        raise Wave7SequenceError(
            "process identity is malformed",
            code="wave7_sequence.process_graph",
            context={"pid": pid},
        ) from exc


def _process_snapshot() -> dict[int, dict[str, Any]]:
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as exc:
        raise Wave7SequenceError(
            "process inventory is unavailable",
            code="wave7_sequence.process_graph",
        ) from exc
    snapshot: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        identity = _read_process_identity(int(entry.name))
        if identity is not None:
            snapshot[identity["pid"]] = identity
    return snapshot


def _identity_matches(current: Mapping[str, Any], captured: Mapping[str, Any]) -> bool:
    return current.get("pid") == captured.get("pid") and current.get(
        "start_time_ticks"
    ) == captured.get("start_time_ticks")


def _controller_children() -> dict[int, int]:
    controller_pid = os.getpid()
    return {
        int(identity["pid"]): int(identity["start_time_ticks"])
        for identity in _process_snapshot().values()
        if identity["parent_pid"] == controller_pid
    }


def _capture_descendant_graph(root_pid: int) -> tuple[dict[str, Any], ...]:
    snapshot = _process_snapshot()
    root = snapshot.get(root_pid)
    if root is None:
        return ()
    children: dict[int, list[int]] = {}
    for identity in snapshot.values():
        children.setdefault(int(identity["parent_pid"]), []).append(
            int(identity["pid"])
        )
    pending = [(root_pid, 0)]
    seen: set[int] = set()
    result: list[dict[str, Any]] = []
    while pending:
        pid, depth = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        identity = snapshot.get(pid)
        if identity is None:
            continue
        result.append({**identity, "depth": depth})
        if len(result) > MAX_PROCESS_GRAPH:
            raise Wave7SequenceError(
                "phase process graph exceeds the cleanup bound",
                code="wave7_sequence.process_graph",
                context={"maximum": MAX_PROCESS_GRAPH},
            )
        pending.extend((child, depth + 1) for child in children.get(pid, []))
    return tuple(sorted(result, key=lambda item: (item["depth"], item["pid"])))


def _expand_cleanup_graph(
    graph: Sequence[Mapping[str, Any]],
    *,
    baseline_controller_children: Mapping[int, int],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    snapshot = _process_snapshot()
    known = {int(item["pid"]): dict(item) for item in graph}
    discovered: dict[int, dict[str, Any]] = {}
    changed = True
    while changed:
        changed = False
        for current in sorted(snapshot.values(), key=lambda item: item["pid"]):
            pid = int(current["pid"])
            if pid in known or pid in discovered:
                continue
            parent = discovered.get(int(current["parent_pid"]))
            if parent is None:
                captured_parent = known.get(int(current["parent_pid"]))
                live_parent = snapshot.get(int(current["parent_pid"]))
                if (
                    captured_parent is not None
                    and live_parent is not None
                    and _identity_matches(live_parent, captured_parent)
                ):
                    parent = captured_parent
            adopted = (
                current["parent_pid"] == os.getpid()
                and baseline_controller_children.get(pid) != current["start_time_ticks"]
            )
            if parent is None and not adopted:
                continue
            depth = (
                int(parent["depth"]) + 1
                if parent is not None
                else max(
                    (
                        int(item["depth"])
                        for item in (*known.values(), *discovered.values())
                    ),
                    default=0,
                )
                + 1
            )
            discovered[pid] = {**current, "depth": depth}
            changed = True
    combined = tuple(
        sorted(
            (*known.values(), *discovered.values()),
            key=lambda item: (item["depth"], item["pid"]),
        )
    )
    return combined, tuple(
        sorted(discovered.values(), key=lambda item: (item["depth"], item["pid"]))
    )


def _signal_cleanup_graph(
    graph: Sequence[Mapping[str, Any]],
    signum: signal.Signals,
    *,
    events: list[dict[str, Any]],
) -> None:
    snapshot = _process_snapshot()
    known = {int(item["pid"]): item for item in graph}
    controller_pgid = os.getpgrp()
    proven_groups: dict[int, int] = {}
    for pgid in {int(item["process_group_id"]) for item in graph}:
        if pgid == controller_pgid:
            continue
        leader = known.get(pgid)
        members = [
            item for item in snapshot.values() if int(item["process_group_id"]) == pgid
        ]
        if (
            leader is not None
            and snapshot.get(pgid) is not None
            and _identity_matches(snapshot[pgid], leader)
            and all(
                int(member["pid"]) in known
                and _identity_matches(member, known[int(member["pid"])])
                for member in members
            )
        ):
            proven_groups[pgid] = max(
                int(item["depth"])
                for item in graph
                if int(item["process_group_id"]) == pgid
            )
    for depth in sorted({int(item["depth"]) for item in graph}, reverse=True):
        for pgid in sorted(
            (
                pgid
                for pgid, group_depth in proven_groups.items()
                if group_depth == depth
            ),
            reverse=True,
        ):
            try:
                os.killpg(pgid, signum)
                outcome = "sent"
            except ProcessLookupError:
                outcome = "absent"
            events.append(
                {
                    "kind": f"signal_group_{signum.name.lower()}",
                    "pgid": pgid,
                    "depth": depth,
                    "outcome": outcome,
                }
            )
        for captured in sorted(
            (item for item in graph if int(item["depth"]) == depth),
            key=lambda item: int(item["pid"]),
            reverse=True,
        ):
            current = _read_process_identity(int(captured["pid"]))
            if current is None or not _identity_matches(current, captured):
                outcome = "absent"
            elif current["state"] in _TERMINAL_PROCESS_STATES:
                outcome = "terminal"
            else:
                try:
                    os.kill(int(captured["pid"]), signum)
                    outcome = "sent"
                except ProcessLookupError:
                    outcome = "absent"
            events.append(
                {
                    "kind": f"signal_pid_{signum.name.lower()}",
                    "pid": int(captured["pid"]),
                    "depth": depth,
                    "outcome": outcome,
                }
            )


def _reap_captured_children(
    process: subprocess.Popen[bytes], graph: Sequence[Mapping[str, Any]]
) -> list[int]:
    process.poll()
    reaped: list[int] = []
    for captured in graph:
        pid = int(captured["pid"])
        if pid == process.pid:
            continue
        try:
            observed_pid, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            continue
        if observed_pid:
            reaped.append(observed_pid)
    return reaped


def _remaining_cleanup_graph(
    graph: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    snapshot = _process_snapshot()
    remaining = [
        dict(captured)
        for captured in graph
        if snapshot.get(int(captured["pid"])) is not None
        and _identity_matches(snapshot[int(captured["pid"])], captured)
    ]
    remaining_pgids = sorted(
        {
            int(captured["process_group_id"])
            for captured in remaining
            if int(captured["process_group_id"]) != os.getpgrp()
        }
    )
    return remaining, remaining_pgids


def _wait_cleanup_graph(
    process: subprocess.Popen[bytes],
    graph: Sequence[Mapping[str, Any]],
    *,
    seconds: float,
    signum: signal.Signals,
    baseline_controller_children: Mapping[int, int],
    events: list[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], list[dict[str, Any]], list[int], list[int]]:
    deadline = time.monotonic() + seconds
    current_graph = tuple(dict(item) for item in graph)
    reaped: list[int] = []
    empty_samples = 0
    while True:
        current_graph, discovered = _expand_cleanup_graph(
            current_graph,
            baseline_controller_children=baseline_controller_children,
        )
        if discovered:
            _signal_cleanup_graph(discovered, signum, events=events)
        reaped.extend(_reap_captured_children(process, current_graph))
        remaining, remaining_pgids = _remaining_cleanup_graph(current_graph)
        if not remaining:
            empty_samples += 1
            if empty_samples >= 2:
                return current_graph, remaining, remaining_pgids, reaped
        else:
            empty_samples = 0
        if remaining and time.monotonic() >= deadline:
            return current_graph, remaining, remaining_pgids, reaped
        time.sleep(
            0.01 if not remaining else min(0.01, max(0.0, deadline - time.monotonic()))
        )


class _BoundedOutputTail:
    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self.total_bytes = 0
        self.tail = bytearray()

    def append(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if len(chunk) >= self.maximum_bytes:
            self.tail[:] = chunk[-self.maximum_bytes :]
            return
        overflow = len(self.tail) + len(chunk) - self.maximum_bytes
        if overflow > 0:
            del self.tail[:overflow]
        self.tail.extend(chunk)

    def diagnostic(self) -> tuple[str, bool]:
        text = bytes(self.tail).decode("utf-8", errors="replace")
        encoded_size = len(text.encode("utf-8"))
        trimmed = False
        if encoded_size > self.maximum_bytes:
            suffix_size = 0
            start = len(text)
            for character in reversed(text):
                width = len(character.encode("utf-8"))
                if suffix_size + width > self.maximum_bytes:
                    break
                suffix_size += width
                start -= 1
            text = text[start:]
            trimmed = start > 0
        return text, self.total_bytes > self.maximum_bytes or trimmed


def _drain_phase_output(
    pipe: Any, collector: _BoundedOutputTail, errors: list[BaseException]
) -> None:
    try:
        while chunk := pipe.read(64 * 1024):
            collector.append(chunk)
    except BaseException as exc:
        errors.append(exc)


def _start_phase_output_capture(
    process: subprocess.Popen[bytes],
) -> tuple[
    _BoundedOutputTail,
    _BoundedOutputTail,
    list[BaseException],
    tuple[threading.Thread, threading.Thread],
]:
    if process.stdout is None or process.stderr is None:
        raise Wave7SequenceError(
            "phase output pipes were not created",
            code="wave7_sequence.output_capture",
        )
    stdout = _BoundedOutputTail(MAX_PHASE_OUTPUT_BYTES)
    stderr = _BoundedOutputTail(MAX_PHASE_OUTPUT_BYTES)
    errors: list[BaseException] = []
    threads = (
        threading.Thread(
            target=_drain_phase_output,
            args=(process.stdout, stdout, errors),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_phase_output,
            args=(process.stderr, stderr, errors),
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()
    return stdout, stderr, errors, threads


def _attach_phase_diagnostics(
    record: dict[str, Any],
    capture: tuple[
        _BoundedOutputTail,
        _BoundedOutputTail,
        list[BaseException],
        tuple[threading.Thread, threading.Thread],
    ],
    *,
    join_timeout: float,
) -> None:
    stdout, stderr, errors, threads = capture
    for thread in threads:
        thread.join(timeout=max(0.1, join_timeout))
    if any(thread.is_alive() for thread in threads) or errors:
        raise Wave7SequenceError(
            "phase output capture did not terminate cleanly",
            code="wave7_sequence.output_capture",
            context={
                "reader_alive": [thread.is_alive() for thread in threads],
                "error_types": [type(exc).__name__ for exc in errors],
            },
        )
    stdout_tail, stdout_truncated = stdout.diagnostic()
    stderr_tail, stderr_truncated = stderr.diagnostic()
    record.update(
        {
            "stdout_tail": stdout_tail,
            "stdout_truncated": stdout_truncated,
            "stderr_tail": stderr_tail,
            "stderr_truncated": stderr_truncated,
        }
    )


def _attach_fallback_phase_diagnostics(record: dict[str, Any]) -> None:
    record.update(
        {
            "stdout_tail": "",
            "stdout_truncated": True,
            "stderr_tail": "",
            "stderr_truncated": True,
        }
    )


def _terminate_started_process(
    process: subprocess.Popen[bytes],
    *,
    initial_graph: Sequence[Mapping[str, Any]],
    baseline_controller_children: Mapping[int, int],
    term_grace: float,
    kill_grace: float,
) -> dict[str, Any]:
    graph, _ = _expand_cleanup_graph(
        initial_graph,
        baseline_controller_children=baseline_controller_children,
    )
    if not graph:
        root = _read_process_identity(process.pid)
        if root is not None:
            graph = ({**root, "depth": 0},)
    record = {
        "pid": process.pid,
        "pgid": process.pid,
        "term_sent": False,
        "kill_sent": False,
        "reaped": False,
        "captured_process_graph": [],
        "remaining_pids": [],
        "remaining_pgids": [],
        "signal_events": [],
        "graph_overflow": False,
        "observed_process_count": 0,
    }
    events: list[dict[str, Any]] = record["signal_events"]
    if graph:
        record["term_sent"] = True
        _signal_cleanup_graph(graph, signal.SIGTERM, events=events)
        graph, remaining, remaining_pgids, _ = _wait_cleanup_graph(
            process,
            graph,
            seconds=term_grace,
            signum=signal.SIGTERM,
            baseline_controller_children=baseline_controller_children,
            events=events,
        )
        if remaining:
            record["kill_sent"] = True
            _signal_cleanup_graph(graph, signal.SIGKILL, events=events)
            graph, remaining, remaining_pgids, _ = _wait_cleanup_graph(
                process,
                graph,
                seconds=kill_grace,
                signum=signal.SIGKILL,
                baseline_controller_children=baseline_controller_children,
                events=events,
            )
    else:
        remaining, remaining_pgids = [], []
    try:
        process.wait(timeout=kill_grace)
    except subprocess.TimeoutExpired as exc:
        raise Wave7SequenceError(
            "phase leader could not be reaped within the cleanup bound",
            code="wave7_sequence.cleanup",
            context={"pid": process.pid},
        ) from exc
    graph, _ = _expand_cleanup_graph(
        graph,
        baseline_controller_children=baseline_controller_children,
    )
    _reap_captured_children(process, graph)
    remaining, remaining_pgids = _remaining_cleanup_graph(graph)
    record["graph_overflow"] = len(graph) > MAX_PROCESS_GRAPH
    record["observed_process_count"] = len(graph)
    record["captured_process_graph"] = [
        dict(item) for item in graph[:MAX_PROCESS_GRAPH]
    ]
    record["remaining_pids"] = [int(item["pid"]) for item in remaining]
    record["remaining_pgids"] = remaining_pgids
    record["reaped"] = (
        process.poll() is not None and not remaining and not remaining_pgids
    )
    if record["reaped"] is not True:
        raise Wave7SequenceError(
            "captured phase process graph survived bounded cleanup",
            code="wave7_sequence.cleanup",
            context={"cleanup": record},
        )
    if record["graph_overflow"] is True:
        raise Wave7SequenceError(
            "phase process graph exceeded the evidence bound",
            code="wave7_sequence.process_graph",
            context={"cleanup": record, "maximum": MAX_PROCESS_GRAPH},
        )
    return record


def _run_phase(
    phase: str,
    argv: list[str],
    *,
    environment: Mapping[str, str],
    timeout: float,
    term_grace: float,
    kill_grace: float,
) -> tuple[int, float, dict[str, Any]]:
    started = time.monotonic()
    child_environment = os.environ.copy()
    child_environment.update(environment)
    _enable_child_subreaper()
    baseline_controller_children = _controller_children()
    try:
        process = subprocess.Popen(
            argv,
            cwd=REPO_ROOT,
            env=child_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise Wave7SequenceError(
            "phase process could not be launched",
            code="wave7_sequence.launch",
            context={
                "phase": phase,
                "error_type": type(exc).__name__,
                "duration_seconds": time.monotonic() - started,
            },
        ) from exc
    output_capture: (
        tuple[
            _BoundedOutputTail,
            _BoundedOutputTail,
            list[BaseException],
            tuple[threading.Thread, threading.Thread],
        ]
        | None
    ) = None
    owned_graph: tuple[dict[str, Any], ...] = ()
    stage = "output_capture_setup"
    try:
        output_capture = _start_phase_output_capture(process)
        stage = "process_graph_capture"
        owned_graph = _capture_descendant_graph(process.pid)
        stage = "process_wait"
        return_code = process.wait(timeout=timeout)
        deadline = time.monotonic() + kill_grace
        empty_samples = 0
        while True:
            owned_graph, _ = _expand_cleanup_graph(
                owned_graph,
                baseline_controller_children=baseline_controller_children,
            )
            _reap_captured_children(process, owned_graph)
            remaining, remaining_pgids = _remaining_cleanup_graph(owned_graph)
            if not remaining and not remaining_pgids:
                empty_samples += 1
                if empty_samples >= 2:
                    break
            else:
                empty_samples = 0
            if time.monotonic() >= deadline:
                raise Wave7SequenceError(
                    "phase leader exited while its descendant/session graph survived",
                    code="wave7_sequence.cleanup",
                )
            time.sleep(0.01)
        if len(owned_graph) > MAX_PROCESS_GRAPH:
            raise Wave7SequenceError(
                "phase process graph exceeded the evidence bound",
                code="wave7_sequence.process_graph",
            )
        cleanup = {
            "pid": process.pid,
            "pgid": process.pid,
            "term_sent": False,
            "kill_sent": False,
            "reaped": True,
            "captured_process_graph": [dict(item) for item in owned_graph],
            "remaining_pids": [],
            "remaining_pgids": [],
            "signal_events": [],
            "graph_overflow": False,
            "observed_process_count": len(owned_graph),
        }
        stage = "output_capture_finalize"
        _attach_phase_diagnostics(
            cleanup, output_capture, join_timeout=term_grace + kill_grace
        )
        return return_code, time.monotonic() - started, cleanup
    except BaseException as exc:
        cleanup = _terminate_started_process(
            process,
            initial_graph=owned_graph,
            baseline_controller_children=baseline_controller_children,
            term_grace=term_grace,
            kill_grace=kill_grace,
        )
        if output_capture is None:
            _attach_fallback_phase_diagnostics(cleanup)
        else:
            try:
                _attach_phase_diagnostics(
                    cleanup,
                    output_capture,
                    join_timeout=term_grace + kill_grace,
                )
            except BaseException:
                _attach_fallback_phase_diagnostics(cleanup)
        if isinstance(exc, subprocess.TimeoutExpired):
            message = "phase exceeded its fixed timeout"
            code = "wave7_sequence.timeout"
        elif isinstance(exc, Wave7SequenceError):
            message = str(exc)
            code = exc.code
        elif stage in {"output_capture_setup", "output_capture_finalize"}:
            message = "phase output capture failed after process start"
            code = "wave7_sequence.output_capture"
        elif stage == "process_graph_capture":
            message = "phase process graph could not be captured"
            code = "wave7_sequence.process_graph"
        else:
            message = "phase controller was interrupted after process start"
            code = "wave7_sequence.interrupted"
        context: dict[str, Any] = {
            "phase": phase,
            "error_type": type(exc).__name__,
            "duration_seconds": time.monotonic() - started,
            "cleanup": cleanup,
        }
        if code == "wave7_sequence.timeout":
            context["timeout_seconds"] = timeout
        raise Wave7SequenceError(
            message,
            code=code,
            context=context,
        ) from exc


def _phase_timeout_budget(
    *,
    phase: str,
    policy: Mapping[str, Any],
    wall_elapsed: float,
    gpu_device_seconds: float,
) -> float:
    candidates = [
        float(policy["phase_timeout_seconds"][phase]),
        float(policy["max_total_wall_seconds"]) - wall_elapsed,
        (float(policy["max_total_gpu_device_seconds"]) - gpu_device_seconds)
        / GPU_DEVICE_COUNT,
    ]
    timeout = min(candidates)
    if timeout <= 0.0:
        raise Wave7SequenceError(
            "phase has no remaining launch budget",
            code="wave7_sequence.cost_cap",
            context={"phase": phase, "candidate_seconds": candidates},
        )
    return timeout


def _validate_comparison_receipt(
    path: Path, *, oracle: Mapping[str, Any], owner: str
) -> dict[str, Any]:
    try:
        path = _absolute_path(str(path), owner=f"{owner} path", must_exist=True)
    except Wave7SequenceError:
        raise
    except OSError as exc:
        raise Wave7SequenceError(
            f"{owner} is unavailable",
            code="wave7_sequence.comparison_gate",
            context={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    try:
        info = path.lstat()
    except OSError as exc:
        raise Wave7SequenceError(
            f"{owner} cannot be stated",
            code="wave7_sequence.comparison_gate",
            context={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise Wave7SequenceError(
            f"{owner} must be one regular non-symlink file",
            code="wave7_sequence.comparison_gate",
            context={"path": str(path)},
        )
    payload = _strict_json_file(path, owner=owner)
    digest = _digest(payload.get("receipt_payload_sha256"), owner=f"{owner} digest")
    _verify_signed_payload(payload, expected_payload_sha256=digest, owner=owner)
    for key, expected in oracle.items():
        if payload.get(key) != expected:
            raise Wave7SequenceError(
                f"{owner} did not satisfy its frozen oracle",
                code="wave7_sequence.comparison_gate",
                context={
                    "field": key,
                    "expected": expected,
                    "observed": payload.get(key),
                },
            )
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "receipt_payload_sha256": digest,
        "schema": payload["schema"],
        "status": payload["status"],
        "mismatches": payload["mismatches"],
        "validated": True,
    }


def _verify_child_gate(receipt_path: Path, expected_payload_sha256: str) -> None:
    expected = _digest(
        expected_payload_sha256, owner="child gate expected payload sha256"
    )
    payload = _strict_json_file(receipt_path, owner="child gate receipt")
    _verify_signed_payload(
        payload, expected_payload_sha256=expected, owner="child gate receipt"
    )
    if (
        payload.get("schema") != PRE_CHILD_RECEIPT_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("child_launch_authorized") is not True
        or payload.get("mismatches") != []
    ):
        raise Wave7SequenceError(
            "child gate is not an exact launch authorization",
            code="wave7_sequence.child_gate",
        )


def launch_child(
    *,
    receipt_path: Path,
    expected_payload_sha256: str,
    config_path: Path,
    run_root: Path,
    parent_run_root: Path,
    launcher: Sequence[str],
) -> int:
    """Verify the immutable gate before transferring control to training."""

    try:
        receipt_path = _absolute_path(
            str(receipt_path), owner="child gate receipt", must_exist=True
        )
        config_path = _absolute_path(
            str(config_path), owner="child config", must_exist=True
        )
        run_root = _absolute_path(
            str(run_root), owner="child run root", must_exist=False
        )
        parent_run_root = _absolute_path(
            str(parent_run_root), owner="parent run root", must_exist=True
        )
        _verify_child_gate(receipt_path, expected_payload_sha256)
        if run_root.exists() or run_root.is_symlink():
            raise Wave7SequenceError(
                "child run root appeared before gate transfer",
                code="wave7_sequence.child_gate",
            )
        resolved = load_train_config(config_path)
        config = resolved.config
        expected_checkpoint = parent_run_root / "checkpoints" / "step-3"
        if (
            config.run.name != "resume_child"
            or config.run.collision_policy != "fail"
            or _configured_run_dir(config) != run_root
            or config.resume.mode != "exact_same_world_size"
            or config.resume.checkpoint_dir is None
            or Path(config.resume.checkpoint_dir).resolve() != expected_checkpoint
            or config.runtime.determinism.mode != "strict_cuda_replay_v1"
        ):
            raise Wave7SequenceError(
                "child config no longer matches the frozen resume role",
                code="wave7_sequence.child_gate",
            )
        expected_launcher = _train_command(
            config_path, port=TRAIN_PORTS["resume_child"]
        )
        if list(launcher) != expected_launcher:
            raise Wave7SequenceError(
                "child launcher does not match the frozen production grammar",
                code="wave7_sequence.command_contract",
            )
        os.execve(expected_launcher[0], expected_launcher, os.environ.copy())
    except BaseException as exc:
        print(
            f"{getattr(exc, 'code', 'wave7_sequence.child_gate')}: {exc}",
            file=sys.stderr,
        )
        return 1
    raise AssertionError("os.execve returned unexpectedly")


def _phase_record(
    phase: str, command_sha256: str, environment_sha256: str
) -> dict[str, Any]:
    return {
        "phase": phase,
        "status": "not_started",
        "attempt_count": 0,
        "command_sha256": command_sha256,
        "launched_command_sha256": None,
        "environment_sha256": environment_sha256,
        "started_at": None,
        "completed_at": None,
        "return_code": None,
        "duration_seconds": None,
        "process": None,
        "gpu_before_launch": None,
        "gpu_after_cleanup": None,
    }


def _failure_record(phase: str, exc: BaseException) -> dict[str, str]:
    code = getattr(exc, "code", "wave7_sequence.unexpected")
    if not isinstance(code, str) or not code:
        code = "wave7_sequence.unexpected"
    message = str(exc)
    return {
        "phase": phase,
        "code": code,
        "message": message if message else type(exc).__name__,
        "error_type": type(exc).__name__,
    }


def _failed_gpu_recovery(phase: str, exc: BaseException) -> dict[str, Any]:
    return {
        "phase": phase,
        "status": "failed",
        "failure": _failure_record(phase, exc),
    }


def _attempt_final_gpu_recovery(
    *, baseline: Mapping[str, Any], stability_seconds: float
) -> tuple[dict[str, Any], dict[str, str] | None]:
    try:
        recovery = _attest_shared_gpu_recovery(
            baseline=baseline,
            stability_seconds=stability_seconds,
            phase="preterminal",
        )
    except BaseException as exc:
        recovery = _failed_gpu_recovery("preterminal", exc)
        return recovery, recovery["failure"]
    return recovery, None


def _signed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_payload_sha256"] = _sha256_bytes(canonical_json_bytes(result))
    return result


def _validate_marker_payload(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_fields(
        payload,
        {
            "schema",
            "plan",
            "controller_source",
            "request",
            "immutable_receipts",
            "identities",
            "environment",
            "absence_attestation",
            "command_sha256",
            "phase_order",
            "gpu_admission",
            "claim_scope",
            "published_at",
            "receipt_payload_sha256",
        },
        owner="sequence marker",
    )
    digest = _digest(
        payload["receipt_payload_sha256"], owner="sequence marker payload sha256"
    )
    _verify_signed_payload(
        payload,
        expected_payload_sha256=digest,
        owner="sequence marker",
    )
    expected_receipts = {
        name: plan[name]
        for name in (
            "amendment",
            "legacy_r4_failure",
            "predecessor_sequence_failure",
            "predecessor_preflight_failure",
            "determinism_preflight_plan",
            "determinism_preflight",
            "runtime_receipt",
        )
    }
    expected_identities = {
        "sources": plan["sources"],
        "request_source_inventory": plan["request_source_inventory"],
        "configs": plan["configs"],
        "resolved_configs": plan["resolved_configs"],
        "input_attestations": plan["input_attestations"],
        "provenance_sha256": plan["provenance_sha256"],
        "ownership": plan["ownership"],
    }
    expected_absence = {
        "run_roots": plan["run_roots"],
        "targets": plan["targets"],
        "status": "all_absent",
    }
    if (
        payload["schema"] != MARKER_SCHEMA
        or payload["plan"] != plan_identity
        or payload["controller_source"] != plan["controller_source"]
        or payload["request"] != plan["request"]
        or payload["immutable_receipts"] != expected_receipts
        or payload["identities"] != expected_identities
        or payload["environment"] != plan["environment"]
        or payload["absence_attestation"] != expected_absence
        or payload["command_sha256"] != plan["command_sha256"]
        or payload["phase_order"] != list(PHASE_ORDER)
        or payload["claim_scope"] != SEQUENCE_CLAIM_SCOPE
    ):
        raise Wave7SequenceError(
            "sequence marker does not match the frozen v6 plan",
            code="wave7_sequence.marker_schema",
        )
    baseline = _validate_shared_gpu_baseline(
        payload["gpu_admission"],
        expected_stability_seconds=plan["policy"]["gpu_baseline_stability_seconds"],
    )
    return baseline


def _verify_marker_identity(
    marker_identity: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_fields(
        marker_identity,
        {"path", "file_sha256", "payload_sha256", "schema"},
        owner="sequence marker identity",
    )
    path = _absolute_path(
        marker_identity["path"], owner="sequence marker path", must_exist=True
    )
    if (
        marker_identity["schema"] != MARKER_SCHEMA
        or _sha256_file(path) != marker_identity["file_sha256"]
    ):
        raise Wave7SequenceError(
            "sequence marker identity drifted or changed schema",
            code="wave7_sequence.marker_schema",
        )
    payload = _strict_json_file(path, owner="sequence marker")
    if payload.get("receipt_payload_sha256") != marker_identity["payload_sha256"]:
        raise Wave7SequenceError(
            "sequence marker payload binding drifted",
            code="wave7_sequence.marker_schema",
        )
    return _validate_marker_payload(payload, plan=plan, plan_identity=plan_identity)


def _validate_phase_process(value: Any, *, owner: str) -> None:
    if not isinstance(value, dict):
        raise Wave7SequenceError(
            f"{owner} must be one object",
            code="wave7_sequence.terminal_schema",
        )
    _exact_fields(value, set(_PHASE_PROCESS_FIELDS), owner=owner)
    pid = value["pid"]
    captured_graph = value["captured_process_graph"]
    observed_process_count = value["observed_process_count"]
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or value["pgid"] != pid
        or any(
            not isinstance(value[name], bool)
            for name in (
                "term_sent",
                "kill_sent",
                "reaped",
                "graph_overflow",
                "stdout_truncated",
                "stderr_truncated",
            )
        )
        or value["reaped"] is not True
        or value["graph_overflow"] is not False
        or not isinstance(captured_graph, list)
        or len(captured_graph) > MAX_PROCESS_GRAPH
        or isinstance(observed_process_count, bool)
        or not isinstance(observed_process_count, int)
        or observed_process_count != len(captured_graph)
        or value["remaining_pids"] != []
        or value["remaining_pgids"] != []
        or not isinstance(value["signal_events"], list)
        or len(value["signal_events"]) > MAX_SIGNAL_EVENTS
    ):
        raise Wave7SequenceError(
            f"{owner} cleanup shape is invalid",
            code="wave7_sequence.terminal_schema",
        )
    graph_order: list[tuple[int, int]] = []
    graph_pids: set[int] = set()
    for row in captured_graph:
        if not isinstance(row, dict) or set(row) != set(_PROCESS_GRAPH_FIELDS):
            raise Wave7SequenceError(
                f"{owner} process graph row shape is invalid",
                code="wave7_sequence.terminal_schema",
            )
        integer_bounds = {
            "pid": 1,
            "parent_pid": 0,
            "process_group_id": 1,
            "session_id": 1,
            "start_time_ticks": 0,
            "depth": 0,
        }
        if any(
            isinstance(row[name], bool)
            or not isinstance(row[name], int)
            or row[name] < minimum
            for name, minimum in integer_bounds.items()
        ) or not (isinstance(row["state"], str) and len(row["state"]) == 1):
            raise Wave7SequenceError(
                f"{owner} process graph row types are invalid",
                code="wave7_sequence.terminal_schema",
            )
        row_pid = row["pid"]
        graph_order.append((row["depth"], row_pid))
        if row_pid in graph_pids:
            raise Wave7SequenceError(
                f"{owner} process graph repeats a pid",
                code="wave7_sequence.terminal_schema",
            )
        graph_pids.add(row_pid)
    if graph_order != sorted(graph_order):
        raise Wave7SequenceError(
            f"{owner} process graph order is invalid",
            code="wave7_sequence.terminal_schema",
        )
    for event in value["signal_events"]:
        kind = event.get("kind") if isinstance(event, dict) else None
        if (
            not isinstance(event, dict)
            or not isinstance(kind, str)
            or kind not in _SIGNAL_EVENT_KINDS
        ):
            raise Wave7SequenceError(
                f"{owner} signal event kind is invalid",
                code="wave7_sequence.terminal_schema",
            )
        is_group = kind.startswith("signal_group_")
        target = "pgid" if is_group else "pid"
        expected_fields = {"kind", target, "depth", "outcome"}
        allowed_outcomes = (
            {"sent", "absent"}
            if is_group
            else {
                "sent",
                "absent",
                "terminal",
            }
        )
        if (
            set(event) != expected_fields
            or isinstance(event[target], bool)
            or not isinstance(event[target], int)
            or event[target] <= 0
            or isinstance(event["depth"], bool)
            or not isinstance(event["depth"], int)
            or event["depth"] < 0
            or not isinstance(event["outcome"], str)
            or event["outcome"] not in allowed_outcomes
        ):
            raise Wave7SequenceError(
                f"{owner} signal event row is invalid",
                code="wave7_sequence.terminal_schema",
            )
    for stream in ("stdout", "stderr"):
        tail = value[f"{stream}_tail"]
        if (
            not isinstance(tail, str)
            or len(tail.encode("utf-8")) > MAX_PHASE_OUTPUT_BYTES
        ):
            raise Wave7SequenceError(
                f"{owner} {stream} diagnostic exceeds its byte cap",
                code="wave7_sequence.terminal_schema",
            )


def _validate_failed_gpu_recovery(
    value: Any, *, expected_phase: str, owner: str
) -> None:
    if not isinstance(value, dict):
        raise Wave7SequenceError(
            f"{owner} must be one object",
            code="wave7_sequence.terminal_schema",
        )
    _exact_fields(value, {"phase", "status", "failure"}, owner=owner)
    failure = value["failure"]
    if (
        value["phase"] != expected_phase
        or value["status"] != "failed"
        or not isinstance(failure, dict)
        or set(failure) != {"phase", "code", "message", "error_type"}
        or failure.get("phase") != expected_phase
        or any(
            not isinstance(failure.get(name), str) or not failure[name]
            for name in ("code", "message", "error_type")
        )
    ):
        raise Wave7SequenceError(
            f"{owner} is malformed",
            code="wave7_sequence.terminal_schema",
        )


def _validate_terminal_payload(payload: Mapping[str, Any]) -> None:
    _exact_fields(
        payload,
        {
            "schema",
            "status",
            "failure",
            "plan",
            "marker",
            "input_attestations",
            "phase_records",
            "pre_child_receipt",
            "final_receipt",
            "bounded_cleanup",
            "cost",
            "claim_scope",
            "final_gpu_recovery",
            "completed_at",
        },
        owner="sequence terminal receipt",
    )
    if (
        payload["schema"] != RECEIPT_SCHEMA
        or payload["status"] not in {"passed", "failed"}
        or payload["claim_scope"] != SEQUENCE_CLAIM_SCOPE
        or not isinstance(payload["phase_records"], list)
        or len(payload["phase_records"]) != len(PHASE_ORDER)
    ):
        raise Wave7SequenceError(
            "sequence terminal receipt schema or claim scope is invalid",
            code="wave7_sequence.terminal_schema",
        )
    for owner, binding, fields in (
        (
            "terminal plan binding",
            payload["plan"],
            {"path", "file_sha256", "payload_sha256"},
        ),
        (
            "terminal marker binding",
            payload["marker"],
            {"path", "file_sha256", "payload_sha256", "schema"},
        ),
    ):
        if not isinstance(binding, dict):
            raise Wave7SequenceError(
                f"{owner} is malformed", code="wave7_sequence.terminal_schema"
            )
        _exact_fields(binding, fields, owner=owner)
        _digest(binding["file_sha256"], owner=f"{owner} file sha256")
        _digest(binding["payload_sha256"], owner=f"{owner} payload sha256")
    plan_binding = payload["plan"]
    plan_path = _absolute_path(
        plan_binding["path"], owner="terminal plan path", must_exist=True
    )
    live_plan, observed_plan_binding = _load_plan(plan_path)
    if observed_plan_binding != plan_binding:
        raise Wave7SequenceError(
            "terminal plan binding is not live",
            code="wave7_sequence.terminal_schema",
        )
    predecessor = live_plan["predecessor_sequence_failure"]
    if not isinstance(predecessor, dict):
        raise Wave7SequenceError(
            "terminal live plan predecessor binding is malformed",
            code="wave7_sequence.terminal_schema",
        )
    _validate_predecessor_sequence_failure(predecessor)
    predecessor_preflight = live_plan["predecessor_preflight_failure"]
    if not isinstance(predecessor_preflight, dict):
        raise Wave7SequenceError(
            "terminal live plan r6 predecessor binding is malformed",
            code="wave7_sequence.terminal_schema",
        )
    _validate_predecessor_preflight_failure(predecessor_preflight)
    amendment = live_plan["amendment"]
    if not isinstance(amendment, dict):
        raise Wave7SequenceError(
            "terminal live plan amendment binding is malformed",
            code="wave7_sequence.terminal_schema",
        )
    _validate_r7_amendment_binding(amendment)
    if payload["input_attestations"] != live_plan["input_attestations"]:
        raise Wave7SequenceError(
            "terminal input attestation projection differs from the live plan",
            code="wave7_sequence.terminal_schema",
        )
    policy = _validate_policy(live_plan["policy"])
    failure = payload["failure"]
    if payload["status"] == "passed":
        if failure is not None:
            raise Wave7SequenceError(
                "passed terminal receipt reports a failure",
                code="wave7_sequence.terminal_schema",
            )
    elif (
        not isinstance(failure, dict)
        or set(failure) != {"phase", "code", "message", "error_type"}
        or any(
            not isinstance(failure.get(name), str) or not failure[name]
            for name in ("phase", "code", "message", "error_type")
        )
    ):
        raise Wave7SequenceError(
            "failed terminal receipt lacks one exact failure",
            code="wave7_sequence.terminal_schema",
        )
    attempted_wall_seconds = 0.0
    attempted_gpu_duration_seconds = 0.0
    records_by_phase: dict[str, dict[str, Any]] = {}
    for expected_phase, record in zip(
        PHASE_ORDER, payload["phase_records"], strict=True
    ):
        if not isinstance(record, dict):
            raise Wave7SequenceError(
                "sequence phase record must be one object",
                code="wave7_sequence.terminal_schema",
            )
        _exact_fields(
            record,
            {
                "phase",
                "status",
                "attempt_count",
                "command_sha256",
                "launched_command_sha256",
                "environment_sha256",
                "started_at",
                "completed_at",
                "return_code",
                "duration_seconds",
                "process",
                "gpu_before_launch",
                "gpu_after_cleanup",
            },
            owner="sequence phase record",
        )
        if record["phase"] != expected_phase:
            raise Wave7SequenceError(
                "sequence phase record order drifted",
                code="wave7_sequence.terminal_schema",
            )
        records_by_phase[expected_phase] = record
        attempt_count = record["attempt_count"]
        if (
            isinstance(attempt_count, bool)
            or not isinstance(attempt_count, int)
            or attempt_count not in {0, 1}
        ):
            raise Wave7SequenceError(
                "sequence phase attempt count is invalid",
                code="wave7_sequence.terminal_schema",
            )
        if attempt_count == 0:
            if record["duration_seconds"] is not None:
                raise Wave7SequenceError(
                    "unattempted phase reports a duration",
                    code="wave7_sequence.terminal_schema",
                )
        else:
            duration = _validated_cost_seconds(
                record["duration_seconds"],
                owner=f"{expected_phase} duration seconds",
            )
            attempted_wall_seconds = _freeze_cost_seconds(
                attempted_wall_seconds + duration
            )
            if expected_phase in GPU_PHASES:
                attempted_gpu_duration_seconds = _freeze_cost_seconds(
                    attempted_gpu_duration_seconds + duration
                )
        status = record["status"]
        process = record["process"]
        clean_process = False
        if isinstance(process, dict):
            try:
                _validate_phase_process(
                    process, owner=f"{expected_phase} phase process"
                )
            except Wave7SequenceError as exc:
                raise Wave7SequenceError(
                    "sequence phase process evidence is invalid",
                    code="wave7_sequence.terminal_schema",
                    context={"phase": expected_phase},
                ) from exc
            clean_process = True
        launched_digest = record["launched_command_sha256"]
        expected_command_digest = live_plan["command_sha256"][expected_phase]
        expected_environment_digest = _sha256_bytes(
            canonical_json_bytes(live_plan["environment"])
        )
        if (
            status not in {"not_started", "passed", "failed"}
            or record["command_sha256"] != expected_command_digest
            or record["environment_sha256"] != expected_environment_digest
            or (
                launched_digest is not None
                and (
                    not isinstance(launched_digest, str)
                    or len(launched_digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in launched_digest
                    )
                )
            )
        ):
            raise Wave7SequenceError(
                "sequence phase state or identity is invalid",
                code="wave7_sequence.terminal_schema",
            )
        if status == "not_started":
            if attempt_count != 0 or any(
                record[name] is not None
                for name in (
                    "launched_command_sha256",
                    "started_at",
                    "completed_at",
                    "return_code",
                    "duration_seconds",
                    "process",
                    "gpu_before_launch",
                    "gpu_after_cleanup",
                )
            ):
                raise Wave7SequenceError(
                    "unattempted phase contains attempted-phase evidence",
                    code="wave7_sequence.terminal_schema",
                )
        elif status == "passed":
            if (
                attempt_count != 1
                or record["return_code"] != 0
                or isinstance(record["return_code"], bool)
                or not isinstance(record["started_at"], str)
                or not record["started_at"]
                or not isinstance(record["completed_at"], str)
                or not record["completed_at"]
                or launched_digest is None
                or not clean_process
                or record["gpu_after_cleanup"] is None
                or (record["gpu_before_launch"] is None)
                == (expected_phase in GPU_PHASES)
            ):
                raise Wave7SequenceError(
                    "passed phase lacks exact execution and cleanup evidence",
                    code="wave7_sequence.terminal_schema",
                )
        elif attempt_count == 0:
            if (
                record["launched_command_sha256"] is not None
                or record["started_at"] is not None
                or record["return_code"] is not None
                or record["duration_seconds"] is not None
                or record["process"] is not None
                or record["gpu_after_cleanup"] is not None
                or not isinstance(record["completed_at"], str)
                or not record["completed_at"]
                or (
                    record["gpu_before_launch"] is not None
                    and expected_phase not in GPU_PHASES
                )
            ):
                raise Wave7SequenceError(
                    "failed pre-launch phase contains attempted-phase evidence",
                    code="wave7_sequence.terminal_schema",
                )
        elif (
            attempt_count != 1
            or launched_digest is None
            or not isinstance(record["started_at"], str)
            or not record["started_at"]
            or not isinstance(record["completed_at"], str)
            or not record["completed_at"]
            or (process is not None and not clean_process)
            or (
                process is None
                and failure is not None
                and failure.get("phase") == expected_phase
                and failure.get("code") != "wave7_sequence.launch"
            )
            or (
                record["return_code"] is not None
                and (
                    isinstance(record["return_code"], bool)
                    or not isinstance(record["return_code"], int)
                )
            )
            or (record["gpu_before_launch"] is None) == (expected_phase in GPU_PHASES)
        ):
            raise Wave7SequenceError(
                "failed attempted phase lacks exact execution or cleanup evidence",
                code="wave7_sequence.terminal_schema",
            )
        try:
            if record["gpu_before_launch"] is not None:
                _validate_gpu_sample(
                    record["gpu_before_launch"],
                    memory_total_mib=SHARED_GPU_MEMORY_TOTAL_MIB,
                    memory_used_ceiling_mib=SHARED_GPU_MEMORY_USED_CEILING_MIB,
                    memory_headroom_floor_mib=SHARED_GPU_MEMORY_HEADROOM_FLOOR_MIB,
                    owner=f"{expected_phase} gpu_before_launch",
                )
            if record["gpu_after_cleanup"] is not None:
                phase_recovery = record["gpu_after_cleanup"]
                if isinstance(phase_recovery, dict) and set(phase_recovery) == {
                    "phase",
                    "status",
                    "failure",
                }:
                    if record["status"] != "failed":
                        raise Wave7SequenceError(
                            "passed phase reports failed GPU recovery",
                            code="wave7_sequence.terminal_schema",
                        )
                    _validate_failed_gpu_recovery(
                        phase_recovery,
                        expected_phase=f"post-{expected_phase}",
                        owner=f"{expected_phase} failed gpu_after_cleanup",
                    )
                else:
                    _validate_shared_gpu_recovery(
                        phase_recovery,
                        expected_stability_seconds=(
                            SHARED_GPU_POST_CLEANUP_STABILITY_SECONDS
                        ),
                        expected_phase=f"post-{expected_phase}",
                    )
        except Wave7SequenceError as exc:
            raise Wave7SequenceError(
                "sequence phase nested evidence is invalid",
                code="wave7_sequence.terminal_schema",
                context={"phase": expected_phase, "nested_code": exc.code},
            ) from exc

    failed_phases = [
        phase for phase in PHASE_ORDER if records_by_phase[phase]["status"] == "failed"
    ]
    phase_statuses = [records_by_phase[phase]["status"] for phase in PHASE_ORDER]
    if payload["status"] == "passed":
        state_machine_valid = phase_statuses == ["passed"] * len(PHASE_ORDER)
    elif len(failed_phases) == 1:
        failed_index = PHASE_ORDER.index(failed_phases[0])
        state_machine_valid = (
            failure["phase"] == failed_phases[0]
            and phase_statuses[:failed_index] == ["passed"] * failed_index
            and phase_statuses[failed_index + 1 :]
            == ["not_started"] * (len(PHASE_ORDER) - failed_index - 1)
        )
    elif not failed_phases and failure["phase"] == "preterminal":
        state_machine_valid = phase_statuses == ["passed"] * len(PHASE_ORDER)
    elif not failed_phases and failure["phase"] == "sequence_controller":
        state_machine_valid = phase_statuses == ["not_started"] * len(PHASE_ORDER)
    else:
        state_machine_valid = False
    if not state_machine_valid:
        raise Wave7SequenceError(
            "terminal failure is not bound to the first failed phase",
            code="wave7_sequence.terminal_schema",
        )
    if failed_phases:
        failed_record = records_by_phase[failed_phases[0]]
        returned_nonzero = (
            isinstance(failed_record["return_code"], int)
            and not isinstance(failed_record["return_code"], bool)
            and failed_record["return_code"] != 0
        )
        if returned_nonzero != (failure["code"] == "wave7_sequence.phase_failed"):
            raise Wave7SequenceError(
                "terminal failure code disagrees with the failed phase outcome",
                code="wave7_sequence.terminal_schema",
            )

    cleanup = payload["bounded_cleanup"]
    expected_cleanup_records = [
        {"phase": phase, **records_by_phase[phase]["process"]}
        for phase in PHASE_ORDER
        if isinstance(records_by_phase[phase]["process"], dict)
    ]
    if (
        not isinstance(cleanup, dict)
        or set(cleanup) != {"policy", "records", "artifacts_deleted"}
        or cleanup["policy"] != "TERM_KILL_reap_started_process_group_only"
        or cleanup["artifacts_deleted"] is not False
        or cleanup["records"] != expected_cleanup_records
    ):
        raise Wave7SequenceError(
            "terminal cleanup ledger is inconsistent with phase cleanup",
            code="wave7_sequence.terminal_schema",
        )

    def validate_comparison_binding(
        value: Any,
        *,
        schema: str,
        oracle: Mapping[str, Any],
        owner: str,
    ) -> None:
        expected_fields = {
            "path",
            "file_sha256",
            "receipt_payload_sha256",
            "schema",
            "status",
            "mismatches",
            "validated",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected_fields
            or value["schema"] != schema
            or value["status"] != "passed"
            or value["mismatches"] != []
            or value["validated"] is not True
        ):
            raise Wave7SequenceError(
                f"{owner} is inconsistent with completed phases",
                code="wave7_sequence.terminal_schema",
            )
        try:
            live_binding = _validate_comparison_receipt(
                Path(value["path"]), oracle=oracle, owner=owner
            )
            _digest(value["file_sha256"], owner=f"{owner} file sha256")
            _digest(
                value["receipt_payload_sha256"],
                owner=f"{owner} payload sha256",
            )
            if live_binding != value:
                raise Wave7SequenceError(
                    f"{owner} differs from its frozen binding",
                    code="wave7_sequence.terminal_schema",
                )
        except Wave7SequenceError as exc:
            raise Wave7SequenceError(
                f"{owner} live evidence is invalid",
                code="wave7_sequence.terminal_schema",
            ) from exc

    pre_child_completed = records_by_phase["pre_child"]["status"] == "passed"
    final_completed = records_by_phase["final_compare"]["status"] == "passed"
    if pre_child_completed:
        validate_comparison_binding(
            payload["pre_child_receipt"],
            schema=PRE_CHILD_RECEIPT_SCHEMA,
            oracle=live_plan["oracles"]["pre_child"],
            owner="terminal pre-child receipt",
        )
    elif payload["pre_child_receipt"] is not None:
        raise Wave7SequenceError(
            "uncompleted pre-child phase reports validated receipt evidence",
            code="wave7_sequence.terminal_schema",
        )
    if final_completed:
        validate_comparison_binding(
            payload["final_receipt"],
            schema=FINAL_RECEIPT_SCHEMA,
            oracle=live_plan["oracles"]["final"],
            owner="terminal final comparison receipt",
        )
    elif payload["final_receipt"] is not None:
        raise Wave7SequenceError(
            "uncompleted final phase reports validated receipt evidence",
            code="wave7_sequence.terminal_schema",
        )

    pre_child_payload_sha256 = (
        None
        if payload["pre_child_receipt"] is None
        else payload["pre_child_receipt"]["receipt_payload_sha256"]
    )
    for phase in PHASE_ORDER:
        record = records_by_phase[phase]
        if record["attempt_count"] == 0:
            continue
        argv = list(live_plan["commands"][phase])
        if phase in {"verify_pre_child", "resume_child"}:
            if pre_child_payload_sha256 is None:
                raise Wave7SequenceError(
                    "attempted child-gated phase lacks live pre-child evidence",
                    code="wave7_sequence.terminal_schema",
                )
            argv = [
                item.replace(
                    PRE_CHILD_DIGEST_PLACEHOLDER,
                    pre_child_payload_sha256,
                )
                for item in argv
            ]
        expected_launched_sha256 = _sha256_bytes(canonical_json_bytes(argv))
        if record["launched_command_sha256"] != expected_launched_sha256:
            raise Wave7SequenceError(
                "launched phase command differs from the live frozen argv",
                code="wave7_sequence.terminal_schema",
                context={"phase": phase},
            )

    try:
        marker_path = _absolute_path(
            payload["marker"]["path"], owner="terminal marker path", must_exist=True
        )
        marker_payload = _strict_json_file(marker_path, owner="terminal marker")
        baseline = _validate_shared_gpu_baseline(marker_payload.get("gpu_admission"))
        if failure is None or failure["phase"] != "sequence_controller":
            baseline = _verify_marker_identity(
                payload["marker"], plan=live_plan, plan_identity=plan_binding
            )
        recovery = payload["final_gpu_recovery"]
        if isinstance(recovery, dict) and set(recovery) == {
            "phase",
            "status",
            "failure",
        }:
            recovery_failure = recovery["failure"]
            if (
                payload["status"] != "failed"
                or recovery["phase"] != "preterminal"
                or recovery["status"] != "failed"
                or not isinstance(recovery_failure, dict)
                or set(recovery_failure) != {"phase", "code", "message", "error_type"}
                or recovery_failure.get("phase") != "preterminal"
                or any(
                    not isinstance(recovery_failure.get(name), str)
                    or not recovery_failure[name]
                    for name in ("code", "message", "error_type")
                )
                or (
                    not failed_phases
                    and failure["phase"] == "preterminal"
                    and recovery_failure != failure
                )
            ):
                raise Wave7SequenceError(
                    "failed final GPU recovery evidence is incoherent",
                    code="wave7_sequence.terminal_schema",
                )
        else:
            _validate_shared_gpu_recovery(
                recovery,
                expected_stability_seconds=SHARED_GPU_POST_CLEANUP_STABILITY_SECONDS,
                expected_phase="preterminal",
                baseline=baseline,
            )
        for phase in PHASE_ORDER:
            phase_recovery = records_by_phase[phase]["gpu_after_cleanup"]
            if phase_recovery is not None:
                if isinstance(phase_recovery, dict) and set(phase_recovery) == {
                    "phase",
                    "status",
                    "failure",
                }:
                    if records_by_phase[phase]["status"] != "failed":
                        raise Wave7SequenceError(
                            "passed phase reports failed GPU recovery",
                            code="wave7_sequence.terminal_schema",
                        )
                    _validate_failed_gpu_recovery(
                        phase_recovery,
                        expected_phase=f"post-{phase}",
                        owner=f"{phase} failed gpu_after_cleanup",
                    )
                else:
                    _validate_shared_gpu_recovery(
                        phase_recovery,
                        expected_stability_seconds=(
                            SHARED_GPU_POST_CLEANUP_STABILITY_SECONDS
                        ),
                        expected_phase=f"post-{phase}",
                        baseline=baseline,
                    )
    except Wave7SequenceError as exc:
        if exc.code == "wave7_sequence.terminal_schema":
            raise
        raise Wave7SequenceError(
            "terminal final GPU recovery evidence is invalid",
            code="wave7_sequence.terminal_schema",
        ) from exc
    cost = payload["cost"]
    if not isinstance(cost, dict):
        raise Wave7SequenceError(
            "sequence terminal cost is malformed",
            code="wave7_sequence.terminal_schema",
        )
    _exact_fields(
        cost,
        {
            "wall_seconds",
            "wall_seconds_cap",
            "gpu_device_count",
            "gpu_device_seconds",
            "gpu_device_seconds_cap",
        },
        owner="sequence terminal cost",
    )
    wall_seconds = _validated_cost_seconds(
        cost["wall_seconds"], owner="terminal wall seconds"
    )
    wall_seconds_cap = _validated_cost_seconds(
        cost["wall_seconds_cap"], owner="terminal wall seconds cap"
    )
    gpu_device_seconds = _validated_cost_seconds(
        cost["gpu_device_seconds"], owner="terminal GPU device seconds"
    )
    gpu_device_seconds_cap = _validated_cost_seconds(
        cost["gpu_device_seconds_cap"], owner="terminal GPU device seconds cap"
    )
    derived_gpu_device_seconds = _freeze_cost_seconds(
        attempted_gpu_duration_seconds * GPU_DEVICE_COUNT
    )
    if (
        cost["gpu_device_count"] != GPU_DEVICE_COUNT
        or isinstance(cost["gpu_device_count"], bool)
        or wall_seconds_cap != float(policy["max_total_wall_seconds"])
        or gpu_device_seconds_cap != float(policy["max_total_gpu_device_seconds"])
        or wall_seconds < attempted_wall_seconds
        or wall_seconds > wall_seconds_cap
        or gpu_device_seconds != derived_gpu_device_seconds
        or gpu_device_seconds > gpu_device_seconds_cap
    ):
        raise Wave7SequenceError(
            "sequence terminal cost does not derive from its live plan and attempts",
            code="wave7_sequence.terminal_schema",
        )
    if payload["status"] == "passed":
        by_phase = {record["phase"]: record for record in payload["phase_records"]}
        cleanup = payload["bounded_cleanup"]
        marker_binding = payload["marker"]
        marker_path = _absolute_path(
            marker_binding["path"], owner="terminal marker path", must_exist=True
        )
        if (
            marker_binding["schema"] != MARKER_SCHEMA
            or _sha256_file(marker_path) != marker_binding["file_sha256"]
        ):
            raise Wave7SequenceError(
                "passed terminal marker binding is not live",
                code="wave7_sequence.terminal_schema",
            )
        marker_payload = _strict_json_file(marker_path, owner="terminal marker")
        _verify_signed_payload(
            marker_payload,
            expected_payload_sha256=marker_binding["payload_sha256"],
            owner="terminal marker",
        )
        if marker_payload.get("schema") != MARKER_SCHEMA:
            raise Wave7SequenceError(
                "passed terminal marker schema is not frozen",
                code="wave7_sequence.terminal_schema",
            )
        baseline = _validate_shared_gpu_baseline(marker_payload.get("gpu_admission"))
        if (
            payload["failure"] is not None
            or payload["final_gpu_recovery"] is None
            or any(
                by_phase[phase]["status"] != "passed"
                or by_phase[phase]["attempt_count"] != 1
                or isinstance(by_phase[phase]["attempt_count"], bool)
                or by_phase[phase]["return_code"] != 0
                or isinstance(by_phase[phase]["return_code"], bool)
                or not isinstance(by_phase[phase]["duration_seconds"], (int, float))
                or isinstance(by_phase[phase]["duration_seconds"], bool)
                or by_phase[phase]["duration_seconds"] < 0
                or not isinstance(by_phase[phase]["started_at"], str)
                or not isinstance(by_phase[phase]["completed_at"], str)
                or not isinstance(by_phase[phase]["launched_command_sha256"], str)
                or len(by_phase[phase]["launched_command_sha256"]) != 64
                or not isinstance(by_phase[phase]["process"], dict)
                or by_phase[phase]["process"].get("reaped") is not True
                or by_phase[phase]["process"].get("remaining_pids") != []
                or by_phase[phase]["process"].get("remaining_pgids") != []
                or by_phase[phase]["gpu_after_cleanup"] is None
                for phase in PHASE_ORDER
            )
            or any(by_phase[phase]["gpu_before_launch"] is None for phase in GPU_PHASES)
            or not isinstance(payload["pre_child_receipt"], dict)
            or payload["pre_child_receipt"].get("validated") is not True
            or payload["pre_child_receipt"].get("schema") != PRE_CHILD_RECEIPT_SCHEMA
            or payload["pre_child_receipt"].get("status") != "passed"
            or payload["pre_child_receipt"].get("mismatches") != []
            or not isinstance(payload["final_receipt"], dict)
            or payload["final_receipt"].get("validated") is not True
            or payload["final_receipt"].get("schema") != FINAL_RECEIPT_SCHEMA
            or payload["final_receipt"].get("status") != "passed"
            or payload["final_receipt"].get("mismatches") != []
            or not isinstance(cleanup, dict)
            or cleanup.get("policy") != "TERM_KILL_reap_started_process_group_only"
            or cleanup.get("artifacts_deleted") is not False
            or not isinstance(cleanup.get("records"), list)
            or len(cleanup["records"]) != len(PHASE_ORDER)
            or [item.get("phase") for item in cleanup["records"]] != list(PHASE_ORDER)
            or any(
                not isinstance(item, dict)
                or item.get("reaped") is not True
                or item.get("remaining_pids") != []
                or item.get("remaining_pgids") != []
                for item in cleanup["records"]
            )
            or not isinstance(cost, dict)
            or set(cost)
            != {
                "wall_seconds",
                "wall_seconds_cap",
                "gpu_device_count",
                "gpu_device_seconds",
                "gpu_device_seconds_cap",
            }
            or cost.get("gpu_device_count") != GPU_DEVICE_COUNT
            or any(
                isinstance(cost.get(name), bool)
                or not isinstance(cost.get(name), (int, float))
                or cost[name] < 0
                for name in (
                    "wall_seconds",
                    "wall_seconds_cap",
                    "gpu_device_seconds",
                    "gpu_device_seconds_cap",
                )
            )
            or cost.get("wall_seconds", float("inf")) > cost.get("wall_seconds_cap", -1)
            or cost.get("gpu_device_seconds", float("inf"))
            > cost.get("gpu_device_seconds_cap", -1)
        ):
            raise Wave7SequenceError(
                "successful sequence receipt lacks final shared-GPU recovery evidence",
                code="wave7_sequence.terminal_schema",
            )
        _validate_shared_gpu_recovery(
            payload["final_gpu_recovery"],
            expected_stability_seconds=SHARED_GPU_POST_CLEANUP_STABILITY_SECONDS,
            expected_phase="preterminal",
            baseline=baseline,
        )
        for phase in PHASE_ORDER:
            _validate_shared_gpu_recovery(
                by_phase[phase]["gpu_after_cleanup"],
                expected_stability_seconds=SHARED_GPU_POST_CLEANUP_STABILITY_SECONDS,
                expected_phase=f"post-{phase}",
                baseline=baseline,
            )


def _publish_terminal_or_sidecar(
    terminal_path: Path,
    sidecar_path: Path,
    payload: Mapping[str, Any],
    *,
    marker_identity: Mapping[str, Any],
) -> bool:
    try:
        _validate_terminal_payload(payload)
        write_strict_json_atomic(terminal_path, _signed_payload(payload))
    except BaseException as exc:
        if terminal_path.exists() or terminal_path.is_symlink():
            print(
                "wave7_sequence.publication_ambiguous: terminal path appeared during publication",
                file=sys.stderr,
            )
            return False
        sidecar_payload = _signed_payload(
            {
                "schema": PUBLICATION_FAILURE_SCHEMA,
                "terminal_receipt_path": str(terminal_path),
                "marker": dict(marker_identity),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "published_at": _utc_now(),
            }
        )
        try:
            write_strict_json_atomic(sidecar_path, sidecar_payload)
        except BaseException as sidecar_exc:
            print(
                f"wave7_sequence.publication_failure: terminal and sidecar failed: {type(sidecar_exc).__name__}",
                file=sys.stderr,
            )
            return False
        print(
            f"wave7_sequence.publication_failure: terminal publication failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return False
    if sidecar_path.exists() or sidecar_path.is_symlink():
        print(
            "wave7_sequence.publication_mutual_exclusion: sidecar appeared after terminal publication",
            file=sys.stderr,
        )
        return False
    return True


def _emergency_terminal(
    plan: Mapping[str, Any],
    plan_identity: Mapping[str, Any],
    marker_identity: Mapping[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    started = time.monotonic()
    environment_sha256 = _sha256_bytes(canonical_json_bytes(plan["environment"]))
    failure = _failure_record("sequence_controller", exc)
    try:
        marker_path = _absolute_path(
            marker_identity["path"], owner="emergency marker path", must_exist=True
        )
        marker_payload = _strict_json_file(marker_path, owner="emergency marker")
        baseline = _validate_shared_gpu_baseline(marker_payload.get("gpu_admission"))
        final_gpu_recovery, _ = _attempt_final_gpu_recovery(
            baseline=baseline,
            stability_seconds=plan["policy"]["gpu_post_cleanup_stability_seconds"],
        )
    except BaseException as recovery_exc:
        final_gpu_recovery = {
            "phase": "preterminal",
            "status": "failed",
            "failure": _failure_record("preterminal", recovery_exc),
        }
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "failed",
        "failure": failure,
        "plan": dict(plan_identity),
        "marker": dict(marker_identity),
        "input_attestations": plan["input_attestations"],
        "phase_records": [
            _phase_record(phase, plan["command_sha256"][phase], environment_sha256)
            for phase in PHASE_ORDER
        ],
        "pre_child_receipt": None,
        "final_receipt": None,
        "claim_scope": dict(SEQUENCE_CLAIM_SCOPE),
        "final_gpu_recovery": final_gpu_recovery,
        "bounded_cleanup": {
            "policy": "TERM_KILL_reap_started_process_group_only",
            "records": [],
            "artifacts_deleted": False,
        },
        "cost": {
            "wall_seconds": _freeze_cost_seconds(time.monotonic() - started),
            "wall_seconds_cap": plan["policy"]["max_total_wall_seconds"],
            "gpu_device_count": GPU_DEVICE_COUNT,
            "gpu_device_seconds": 0.0,
            "gpu_device_seconds_cap": plan["policy"]["max_total_gpu_device_seconds"],
        },
        "completed_at": _utc_now(),
    }


def _execute_after_marker(
    plan: dict[str, Any],
    plan_identity: dict[str, Any],
    marker_identity: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    environment = _validate_environment(plan)
    environment_sha256 = _sha256_bytes(canonical_json_bytes(environment))
    records = [
        _phase_record(phase, plan["command_sha256"][phase], environment_sha256)
        for phase in PHASE_ORDER
    ]
    record_by_phase = {record["phase"]: record for record in records}
    started = time.monotonic()
    gpu_device_seconds = 0.0
    pre_child_receipt: dict[str, Any] | None = None
    final_receipt: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    cleanup_records: list[dict[str, Any]] = []
    final_gpu_recovery: dict[str, Any] | None = None
    policy = plan["policy"]
    roots = {name: Path(value) for name, value in plan["run_roots"].items()}
    targets = {name: Path(value) for name, value in plan["targets"].items()}
    baseline = _verify_marker_identity(
        marker_identity, plan=plan, plan_identity=plan_identity
    )

    for phase in PHASE_ORDER:
        record = record_by_phase[phase]
        post_cleanup_attempted = False
        gpu_duration_counted = False
        try:
            wall_elapsed = time.monotonic() - started
            argv = list(plan["commands"][phase])
            if phase in {"verify_pre_child", "resume_child"}:
                if pre_child_receipt is None:
                    raise Wave7SequenceError(
                        "pre-child receipt was not validated",
                        code="wave7_sequence.comparison_gate",
                    )
                argv = [
                    item.replace(
                        PRE_CHILD_DIGEST_PLACEHOLDER,
                        pre_child_receipt["receipt_payload_sha256"],
                    )
                    for item in argv
                ]
            _revalidate_plan_inputs(plan, plan_identity=plan_identity)
            environment = _validate_environment(plan)
            baseline = _verify_marker_identity(
                marker_identity, plan=plan, plan_identity=plan_identity
            )
            if phase in GPU_PHASES:
                record["gpu_before_launch"] = _assert_shared_gpu_subset(
                    baseline=baseline
                )
            wall_elapsed = time.monotonic() - started
            phase_timeout = _phase_timeout_budget(
                phase=phase,
                policy=policy,
                wall_elapsed=wall_elapsed,
                gpu_device_seconds=gpu_device_seconds,
            )
            record["status"] = "started"
            record["attempt_count"] = 1
            record["launched_command_sha256"] = _sha256_bytes(
                canonical_json_bytes(argv)
            )
            record["started_at"] = _utc_now()
            return_code, duration, cleanup = _run_phase(
                phase,
                argv,
                environment=environment,
                timeout=phase_timeout,
                term_grace=policy["term_grace_seconds"],
                kill_grace=policy["kill_grace_seconds"],
            )
            cleanup_records.append({"phase": phase, **cleanup})
            record["completed_at"] = _utc_now()
            record["return_code"] = return_code
            record["duration_seconds"] = _freeze_cost_seconds(duration)
            record["process"] = cleanup
            primary_exc: BaseException | None = None
            if return_code != 0:
                primary_exc = Wave7SequenceError(
                    "phase returned nonzero",
                    code="wave7_sequence.phase_failed",
                    context={"return_code": return_code},
                )
            if phase in GPU_PHASES:
                gpu_device_seconds += duration * GPU_DEVICE_COUNT
                gpu_duration_counted = True
                if (
                    primary_exc is None
                    and gpu_device_seconds > policy["max_total_gpu_device_seconds"]
                ):
                    raise Wave7SequenceError(
                        "total GPU device-seconds cap was exceeded",
                        code="wave7_sequence.cost_cap",
                        context={"gpu_device_seconds": gpu_device_seconds},
                    )
            post_cleanup_attempted = True
            try:
                record["gpu_after_cleanup"] = _attest_shared_gpu_recovery(
                    baseline=baseline,
                    stability_seconds=policy["gpu_post_cleanup_stability_seconds"],
                    phase=f"post-{phase}",
                )
            except BaseException as recovery_exc:
                record["gpu_after_cleanup"] = _failed_gpu_recovery(
                    f"post-{phase}", recovery_exc
                )
                if primary_exc is None:
                    primary_exc = recovery_exc
            if primary_exc is not None:
                raise primary_exc
            if phase in GPU_PHASES and not roots[phase].is_dir():
                raise Wave7SequenceError(
                    "GPU phase did not publish its planned run root",
                    code="wave7_sequence.phase_output",
                    context={"path": str(roots[phase])},
                )
            if phase == "interrupted_parent":
                for name in ("interruption_marker", "interruption_receipt"):
                    if not targets[name].is_file():
                        raise Wave7SequenceError(
                            "interrupt controller did not publish required evidence",
                            code="wave7_sequence.phase_output",
                            context={"target": name},
                        )
            if phase == "pre_child":
                pre_child_receipt = _validate_comparison_receipt(
                    targets["pre_child_receipt"],
                    oracle=plan["oracles"]["pre_child"],
                    owner="pre-child receipt",
                )
            if phase == "final_compare":
                final_receipt = _validate_comparison_receipt(
                    targets["final_receipt"],
                    oracle=plan["oracles"]["final"],
                    owner="final comparison receipt",
                )
            record["status"] = "passed"
        except BaseException as exc:
            record["status"] = "failed"
            if record["completed_at"] is None:
                record["completed_at"] = _utc_now()
            if record["process"] is None and isinstance(
                getattr(exc, "context", None), dict
            ):
                duration = exc.context.get("duration_seconds")
                if (
                    not isinstance(duration, bool)
                    and isinstance(duration, (int, float))
                    and duration >= 0
                ):
                    record["duration_seconds"] = _freeze_cost_seconds(duration)
                cleanup = exc.context.get("cleanup")
                if isinstance(cleanup, dict):
                    record["process"] = cleanup
                    cleanup_records.append({"phase": phase, **cleanup})
            if (
                phase in GPU_PHASES
                and not gpu_duration_counted
                and isinstance(record["duration_seconds"], (int, float))
                and not isinstance(record["duration_seconds"], bool)
            ):
                gpu_device_seconds += (
                    float(record["duration_seconds"]) * GPU_DEVICE_COUNT
                )
                gpu_duration_counted = True
            if (
                record["status"] == "failed"
                and record["attempt_count"] == 1
                and not post_cleanup_attempted
                and isinstance(record["process"], dict)
                and record["process"].get("reaped") is True
            ):
                try:
                    post_cleanup_attempted = True
                    record["gpu_after_cleanup"] = _attest_shared_gpu_recovery(
                        baseline=baseline,
                        stability_seconds=policy["gpu_post_cleanup_stability_seconds"],
                        phase=f"post-{phase}",
                    )
                except BaseException as recovery_exc:
                    record["gpu_after_cleanup"] = _failed_gpu_recovery(
                        f"post-{phase}", recovery_exc
                    )
            failure = _failure_record(phase, exc)
            break

    if failure is None:
        try:
            _revalidate_plan_inputs(plan, plan_identity=plan_identity)
            _validate_environment(plan)
            baseline = _verify_marker_identity(
                marker_identity, plan=plan, plan_identity=plan_identity
            )
        except BaseException as exc:
            failure = _failure_record("preterminal", exc)

    final_gpu_recovery, recovery_failure = _attempt_final_gpu_recovery(
        baseline=baseline,
        stability_seconds=policy["gpu_post_cleanup_stability_seconds"],
    )
    if failure is None and recovery_failure is not None:
        failure = recovery_failure

    try:
        _revalidate_plan_inputs(plan, plan_identity=plan_identity)
        _validate_environment(plan)
        baseline = _verify_marker_identity(
            marker_identity, plan=plan, plan_identity=plan_identity
        )
        for binding, oracle, owner in (
            (
                pre_child_receipt,
                plan["oracles"]["pre_child"],
                "preterminal pre-child receipt",
            ),
            (
                final_receipt,
                plan["oracles"]["final"],
                "preterminal final comparison receipt",
            ),
        ):
            if binding is None:
                continue
            live_binding = _validate_comparison_receipt(
                Path(binding["path"]), oracle=oracle, owner=owner
            )
            if live_binding != binding:
                raise Wave7SequenceError(
                    f"{owner} differs from its frozen binding",
                    code="wave7_sequence.comparison_gate",
                )
    except BaseException as exc:
        if failure is None:
            failure = _failure_record("preterminal", exc)

    wall_seconds = _freeze_cost_seconds(time.monotonic() - started)
    derived_gpu_device_seconds = _freeze_cost_seconds(
        sum(
            float(record["duration_seconds"]) * GPU_DEVICE_COUNT
            for record in records
            if record["phase"] in GPU_PHASES
            and record["attempt_count"] == 1
            and isinstance(record["duration_seconds"], (int, float))
            and not isinstance(record["duration_seconds"], bool)
        )
    )
    status = "passed" if failure is None else "failed"
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "failure": failure,
        "plan": plan_identity,
        "marker": marker_identity,
        "input_attestations": plan["input_attestations"],
        "phase_records": records,
        "pre_child_receipt": pre_child_receipt,
        "final_receipt": final_receipt,
        "claim_scope": dict(SEQUENCE_CLAIM_SCOPE),
        "final_gpu_recovery": final_gpu_recovery,
        "bounded_cleanup": {
            "policy": "TERM_KILL_reap_started_process_group_only",
            "records": cleanup_records,
            "artifacts_deleted": False,
        },
        "cost": {
            "wall_seconds": wall_seconds,
            "wall_seconds_cap": policy["max_total_wall_seconds"],
            "gpu_device_count": GPU_DEVICE_COUNT,
            "gpu_device_seconds": derived_gpu_device_seconds,
            "gpu_device_seconds_cap": policy["max_total_gpu_device_seconds"],
        },
        "completed_at": _utc_now(),
    }
    return (0 if status == "passed" else 1), receipt


def execute(plan_path: Path) -> int:
    try:
        plan, plan_identity = _load_plan(plan_path)
        if not isinstance(plan["run_roots"], dict) or not isinstance(
            plan["targets"], dict
        ):
            raise Wave7SequenceError(
                "plan paths are malformed", code="wave7_sequence.json_schema"
            )
        _, targets = _validate_topology(
            plan["run_roots"], plan["targets"], require_absent=True
        )
        _revalidate_plan_inputs(plan, plan_identity=plan_identity)
        environment = _validate_environment(plan)
        gpu_baseline = _attest_shared_gpu_baseline(
            stability_seconds=plan["policy"]["gpu_baseline_stability_seconds"],
            memory_total_mib=plan["policy"]["gpu_memory_total_mib"],
            memory_used_ceiling_mib=plan["policy"]["gpu_memory_used_ceiling_mib"],
            memory_headroom_floor_mib=plan["policy"]["gpu_memory_headroom_floor_mib"],
        )
        marker_payload = _signed_payload(
            {
                "schema": MARKER_SCHEMA,
                "plan": plan_identity,
                "controller_source": plan["controller_source"],
                "request": plan["request"],
                "immutable_receipts": {
                    name: plan[name]
                    for name in (
                        "amendment",
                        "legacy_r4_failure",
                        "predecessor_sequence_failure",
                        "predecessor_preflight_failure",
                        "determinism_preflight_plan",
                        "determinism_preflight",
                        "runtime_receipt",
                    )
                },
                "identities": {
                    "sources": plan["sources"],
                    "request_source_inventory": plan["request_source_inventory"],
                    "configs": plan["configs"],
                    "resolved_configs": plan["resolved_configs"],
                    "input_attestations": plan["input_attestations"],
                    "provenance_sha256": plan["provenance_sha256"],
                    "ownership": plan["ownership"],
                },
                "environment": environment,
                "absence_attestation": {
                    "run_roots": plan["run_roots"],
                    "targets": plan["targets"],
                    "status": "all_absent",
                },
                "command_sha256": plan["command_sha256"],
                "phase_order": list(PHASE_ORDER),
                "gpu_admission": gpu_baseline,
                "claim_scope": dict(SEQUENCE_CLAIM_SCOPE),
                "published_at": _utc_now(),
            }
        )
        _validate_marker_payload(marker_payload, plan=plan, plan_identity=plan_identity)
        _validate_r7_amendment_binding(plan["amendment"])
        _validate_predecessor_sequence_failure(plan["predecessor_sequence_failure"])
        _validate_predecessor_preflight_failure(plan["predecessor_preflight_failure"])
        marker_path = targets["sequence_marker"]
        linked = [False]
        try:
            write_strict_json_atomic(
                marker_path,
                marker_payload,
                on_linked=lambda: linked.__setitem__(0, True),
            )
        except BaseException as exc:
            if not linked[0]:
                raise
            marker_identity = {
                "path": str(marker_path),
                "file_sha256": _sha256_file(marker_path),
                "payload_sha256": marker_payload["receipt_payload_sha256"],
                "schema": MARKER_SCHEMA,
            }
            terminal = _emergency_terminal(plan, plan_identity, marker_identity, exc)
            _publish_terminal_or_sidecar(
                targets["sequence_receipt"],
                targets["publication_failure_sidecar"],
                terminal,
                marker_identity=marker_identity,
            )
            print(
                f"{getattr(exc, 'code', 'wave7_sequence.marker_publication')}: marker durability publication failed after link",
                file=sys.stderr,
            )
            return 2
        marker_identity = {
            "path": str(marker_path),
            "file_sha256": _sha256_file(marker_path),
            "payload_sha256": marker_payload["receipt_payload_sha256"],
            "schema": MARKER_SCHEMA,
        }
        try:
            return_code, terminal = _execute_after_marker(
                plan, plan_identity, marker_identity
            )
        except BaseException as exc:
            return_code = 1
            terminal = _emergency_terminal(plan, plan_identity, marker_identity, exc)
        published = _publish_terminal_or_sidecar(
            targets["sequence_receipt"],
            targets["publication_failure_sidecar"],
            terminal,
            marker_identity=marker_identity,
        )
        return return_code if published else 2
    except BaseException as exc:
        print(
            f"{getattr(exc, 'code', 'wave7_sequence.preflight')}: {exc}",
            file=sys.stderr,
        )
        return 2


def verify_plan(plan_path: Path, *, expected_file_sha256: str) -> int:
    try:
        expected = _digest(expected_file_sha256, owner="expected plan file sha256")
        plan, plan_identity = _load_plan(plan_path)
        if plan_identity["file_sha256"] != expected:
            raise Wave7SequenceError(
                "sequence plan file digest does not match the independent expectation",
                code="wave7_sequence.plan_verify",
            )
        if not isinstance(plan["run_roots"], dict) or not isinstance(
            plan["targets"], dict
        ):
            raise Wave7SequenceError(
                "plan paths are malformed", code="wave7_sequence.json_schema"
            )
        _validate_topology(plan["run_roots"], plan["targets"], require_absent=True)
        _revalidate_plan_inputs(plan, plan_identity=plan_identity)
        _validate_environment(plan)
        request = plan["request"]
        if not isinstance(request, dict):
            raise Wave7SequenceError(
                "plan request identity is malformed",
                code="wave7_sequence.plan_verify",
            )
        _exact_fields(
            request,
            {"path", "file_sha256", "payload_sha256"},
            owner="plan request identity",
        )
        request_path = _absolute_path(
            request["path"], owner="plan request path", must_exist=True
        )
        if (
            request_path != R7_REQUEST_PATH
            or _sha256_file(request_path) != request["file_sha256"]
        ):
            raise Wave7SequenceError(
                "live request file identity drifted",
                code="wave7_sequence.plan_verify",
            )
        rebuilt = _build_plan_payload(request_path)
        if rebuilt != plan:
            raise Wave7SequenceError(
                "live request no longer reconstructs the exact plan",
                code="wave7_sequence.plan_verify",
            )
    except BaseException as exc:
        print(
            f"{getattr(exc, 'code', 'wave7_sequence.plan_verify')}: {exc}",
            file=sys.stderr,
        )
        return 2
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--request", type=Path, required=True)
    prepare_parser.add_argument("--plan", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify-plan")
    verify_parser.add_argument("--plan", type=Path, required=True)
    verify_parser.add_argument("--expected-file-sha256", required=True)
    child_parser = subparsers.add_parser("launch-child")
    child_parser.add_argument("--receipt", type=Path, required=True)
    child_parser.add_argument("--expected-payload-sha256", required=True)
    child_parser.add_argument("--config", type=Path, required=True)
    child_parser.add_argument("--run-root", type=Path, required=True)
    child_parser.add_argument("--parent-run-root", type=Path, required=True)
    child_parser.add_argument("launcher", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        return prepare(args.request, args.plan)
    if args.command == "run":
        return execute(args.plan)
    if args.command == "verify-plan":
        return verify_plan(args.plan, expected_file_sha256=args.expected_file_sha256)
    launcher = list(args.launcher)
    if launcher[:1] == ["--"]:
        launcher = launcher[1:]
    return launch_child(
        receipt_path=args.receipt,
        expected_payload_sha256=args.expected_payload_sha256,
        config_path=args.config,
        run_root=args.run_root,
        parent_run_root=args.parent_run_root,
        launcher=launcher,
    )


if __name__ == "__main__":
    raise SystemExit(main())

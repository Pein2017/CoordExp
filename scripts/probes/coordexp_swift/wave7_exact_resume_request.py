#!/usr/bin/env python3
"""Author the immutable Wave 7 r7 input leaves and request-v6 commit point."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.loader import load_train_config  # noqa: E402
from src.qwen.parity import (  # noqa: E402
    assert_absent_artifact_target,
    base_model_weight_identity_with_execution_policy,
    canonical_json_bytes,
    write_strict_json_atomic,
)
from src.training.input_attestation import (  # noqa: E402
    CACHE_ATTESTATION_SCHEMA as CACHE_ATTESTATION_SCHEMA,
    MODEL_ATTESTATION_SCHEMA as MODEL_ATTESTATION_SCHEMA,
    build_training_input_attestations,
)


AMENDMENT_SCHEMA = "coordexp-swift-wave7-r7-amendment-v4"
REQUEST_SCHEMA = "coordexp-swift-wave7-exact-resume-sequence-request-v6"
FROZEN_PARITY_SOURCE_SHA256 = (
    "61d8460bb731d3243315b368a313e7f84d2008e415fdf3db17ed3834dad54deb"
)
AMENDMENT_EFFECTIVE_DATE = "2026-08-12"
AMENDMENT_AUTHORITY_PATH = (
    REPO_ROOT
    / "openspec/changes/harden-optimize-coordexp-swift-training-infrastructure/"
    "measurement-plan.md"
).resolve()
FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256 = (
    "6b088bc3ba7befcca26963b08ef493ce67a14dbfa190158b4ec270a410332730"
)
FROZEN_AMENDMENT_AUTHORITY_SECTION_SHA256 = (
    "6b94dafab01c0845e32a807a1d2b8dc5d31e677d5ae81e448a3110a7dd05a491"
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
_AUTHORITY_PHRASES = (
    "The user explicitly authorized exactly one fresh Wave 7 `r7` one-shot successor.",
    "This is not an r6 retry",
    "no retry, root switch, or automatic Wave 7 `r8` successor",
)
RUN_ROLES = ("uninterrupted", "interrupted_parent", "resume_child")
PHASE_ORDER = (
    "uninterrupted",
    "interrupted_parent",
    "pre_child",
    "verify_pre_child",
    "resume_child",
    "final_compare",
)
GPU_PHASES = ("uninterrupted", "interrupted_parent", "resume_child")
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "FLASH_ATTENTION_DETERMINISTIC": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
TRAIN_PORTS = {
    "uninterrupted": 29681,
    "interrupted_parent": 29682,
    "resume_child": 29683,
}
PRE_CHILD_DIGEST_PLACEHOLDER = "{PRE_CHILD_RECEIPT_PAYLOAD_SHA256}"
FROZEN_R4_FAILURE_FILE_SHA256 = (
    "1d56e8139500ac09e62f57b2d2ce074401bc7596a64bee3b3915ff5036575e5b"
)
FROZEN_R5_FAILURE_FILE_SHA256 = (
    "d5244fb6d96b24b1cfac4438628ae141496f7357c286dff5454c19c9f39731b1"
)
FROZEN_R5_FAILURE_PATH = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r5/"
    "sequence-receipt.json"
).resolve()
FROZEN_R5_FAILURE_PAYLOAD_SHA256 = (
    "f560cbd7bfeac8e25f031021d9d0053c3df0d69961d52f1b854754ce4f4e7656"
)
FROZEN_R5_PLAN = {
    "path": (
        "/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/"
        "wave7_exact_resume/2026-08-11-r5/sequence-plan-v4.json"
    ),
    "file_sha256": ("ea3cef1d96412cd3f425e9c7f0c0cd562ceafa2fe59c39c225755898dbb66a1f"),
    "payload_sha256": (
        "73c502d69ec8b999a9e451620702b54eb1d573bf775e18d32edde08d216f53c4"
    ),
}
FROZEN_R5_MARKER = {
    "path": (
        "/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/"
        "wave7_exact_resume/2026-08-11-r5/sequence-marker.json"
    ),
    "file_sha256": ("58aa7425dc6c019a23e790ba072495f0fb787b805eabaf922f079c35eddb19b5"),
    "payload_sha256": (
        "a620caa0e64a12be980120e310a914571c78ee0889d6da0760aae208f3c82224"
    ),
    "schema": "coordexp-swift-wave7-exact-resume-sequence-marker-v4",
}
FROZEN_R5_PHASE_STATE = (
    (
        "uninterrupted",
        "failed",
        1,
        1,
        "e213a547b4da9566908a1798fadbae844902388d95cc4f45c5c3ee0f070c2c2c",
        "e213a547b4da9566908a1798fadbae844902388d95cc4f45c5c3ee0f070c2c2c",
    ),
    (
        "interrupted_parent",
        "not_started",
        0,
        None,
        "a7280faa0cd98d58d730cdd5c33745cd118f82f4e26b680a6c0b64c8febf2cd6",
        None,
    ),
    (
        "pre_child",
        "not_started",
        0,
        None,
        "fdb4ace0bbf7135c6c550554c12d25a2eead16ab119162c3b3552d645bf38381",
        None,
    ),
    (
        "verify_pre_child",
        "not_started",
        0,
        None,
        "14046ea79a2850fdac7f57e7ace24d057a2c44d91599162022e10c8631e2e240",
        None,
    ),
    (
        "resume_child",
        "not_started",
        0,
        None,
        "1fe2d366c1228370ecdd38bb7d8033c3271a1034f4bc7efd717967bb2aa5e7fd",
        None,
    ),
    (
        "final_compare",
        "not_started",
        0,
        None,
        "8f49c9b274656fbe86d8f467f6c3c41cbfc8a6776b908a676fd1b31dd2cd7753",
        None,
    ),
)
FROZEN_R5_BOUNDED_CLEANUP_SHA256 = (
    "2d902a3e8ada43a139e06f66e41651115680a18ac07a3614a97d709257db4e27"
)
FROZEN_R5_FINAL_GPU_RECOVERY_SHA256 = (
    "c65f6f94837b05e3d3977920a4f771cff206050efa5e7fc8ec7f0ff2a31f1f8c"
)
FROZEN_R6_PREFLIGHT_PLAN_PATH = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r6/"
    "determinism-preflight/plan.json"
).resolve()
FROZEN_R6_PREFLIGHT_PLAN_FILE_SHA256 = (
    "2d5d5108c07e02ff964f5b3e6e4e42ce84e58bd0c17a8310dd7e2f3e6147e6ec"
)
FROZEN_R6_PREFLIGHT_PLAN_PAYLOAD_SHA256 = (
    "9b5c820c5f2c366c994a15732b99498b4072d1d552d9a7048c6afb65e0532a8f"
)
FROZEN_R6_PREFLIGHT_MARKER_PATH = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r6/"
    "determinism-preflight/attempt-start-marker.json"
).resolve()
FROZEN_R6_PREFLIGHT_MARKER_FILE_SHA256 = (
    "f19a894ac3113aecc3a126dab7e3327389518c9e12e34d7806ad401adf239ce9"
)
FROZEN_R6_PREFLIGHT_MARKER_PAYLOAD_SHA256 = (
    "967fc87daabc19dd82d9d5a0f2567a2700da71bf9cb3773f904da63488585d7e"
)
FROZEN_R6_PREFLIGHT_TERMINAL_PATH = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r6/"
    "determinism-preflight/terminal-receipt.json"
).resolve()
FROZEN_R6_PREFLIGHT_TERMINAL_FILE_SHA256 = (
    "eaf20c86a3fa95d7fa6533faaf7fb0e9b8fe938aa6f62d2ba725c3c604a64784"
)
FROZEN_R6_PREFLIGHT_TERMINAL_PAYLOAD_SHA256 = (
    "4b400e094c494099b611e85d78b2156ce53b0fe372e12de6ff780c493b5770ab"
)
LEAF_NAMES = {
    "amendment": "amendment-v4.json",
    "cache": "cache-input-attestation.json",
    "model": "model-input-attestation.json",
    "request": "request-v6.json",
}
R7_SEQUENCE_ROOT = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-core-4"
).resolve()
R7_CACHE_ROOT = (
    REPO_ROOT / "outputs/probes/coordexp_swift/private_v3_cache/2026-08-12-wave7-core-4"
).resolve()
R7_CONFIG_PATHS = {
    "uninterrupted": R7_SEQUENCE_ROOT / "configs/uninterrupted.yaml",
    "interrupted_parent": R7_SEQUENCE_ROOT / "configs/interrupted-parent.yaml",
    "resume_child": R7_SEQUENCE_ROOT / "configs/resume-child.yaml",
}
R7_RUN_ROOTS = {role: R7_SEQUENCE_ROOT / "runs" / role for role in RUN_ROLES}
R7_TARGETS = {
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
    "final_receipt": R7_SEQUENCE_ROOT / "exact-resume-comparison-receipt-v2.json",
}
R7_CACHE_PREPARATION_RECEIPT = R7_CACHE_ROOT / "preparation-receipt.json"
R7_RUNTIME_RECEIPT = R7_SEQUENCE_ROOT / "runtime/runtime-admission.json"
R7_DETERMINISM_PREFLIGHT_PLAN = R7_SEQUENCE_ROOT / "determinism-preflight/plan.json"
R7_DETERMINISM_PREFLIGHT_RECEIPT = (
    R7_SEQUENCE_ROOT / "determinism-preflight/terminal-receipt.json"
)
PYTHON_EXECUTABLE = Path(sys.executable).resolve()
TRAIN_ENTRYPOINT = REPO_ROOT / "src/train.py"
SEQUENCE_CONTROLLER = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_sequence.py"
)
INTERRUPT_CONTROLLER = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_interrupt.py"
)
COMPARATOR_V1 = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_compare.py"
)
COMPARATOR_V2 = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_compare_v2.py"
)
DETERMINISM_PREFLIGHT = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_determinism_preflight.py"
)
PRE_CHILD_SCHEMA = "coordexp-swift-wave7-pre-child-gate-v1"
FINAL_SCHEMA = "coordexp-swift-wave7-exact-resume-comparison-v2"


class RequestAuthoringError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(payload: Mapping[str, Any], *, digest_field: str) -> dict[str, Any]:
    body = dict(payload)
    return {**body, digest_field: _sha256_bytes(canonical_json_bytes(body))}


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _strict_json(path: Path, *, owner: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        info = resolved.lstat()
    except OSError as exc:
        raise RequestAuthoringError(f"{owner} is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or resolved.is_symlink()
        or info.st_size > 16 * 1024 * 1024
    ):
        raise RequestAuthoringError(f"{owner} is not a bounded regular file")
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RequestAuthoringError(f"{owner} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RequestAuthoringError(f"{owner} root is not an object")
    canonical_json_bytes(value)
    return value


def _build_amendment_payload(authority_path: str | Path) -> dict[str, Any]:
    requested = Path(authority_path).expanduser()
    if requested.is_symlink():
        raise RequestAuthoringError("amendment authority must not be a symlink")
    path = requested.resolve()
    if path != AMENDMENT_AUTHORITY_PATH:
        raise RequestAuthoringError("amendment authority is not the canonical file")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RequestAuthoringError("amendment authority is unreadable") from exc
    file_sha256 = _sha256_file(path)
    if file_sha256 != FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256:
        raise RequestAuthoringError("amendment authority file identity drifted")
    start = text.find(AMENDMENT_SECTION_ANCHOR)
    if start < 0:
        raise RequestAuthoringError("amendment authority section anchor drifted")
    next_heading = text.find("\n### ", start + len(AMENDMENT_SECTION_ANCHOR))
    section = text[start:] if next_heading < 0 else text[start:next_heading]
    section_sha256 = _sha256_bytes(section.encode("utf-8"))
    if section_sha256 != FROZEN_AMENDMENT_AUTHORITY_SECTION_SHA256:
        raise RequestAuthoringError("amendment authority section identity drifted")
    normalized_section = " ".join(section.split())
    if any(phrase not in normalized_section for phrase in _AUTHORITY_PHRASES):
        raise RequestAuthoringError(
            "amendment authority scope or approval text drifted"
        )
    body = {
        "schema": AMENDMENT_SCHEMA,
        "status": "approved",
        "effective_date": AMENDMENT_EFFECTIVE_DATE,
        "section_anchor": AMENDMENT_SECTION_ANCHOR,
        "scope": dict(AMENDMENT_SCOPE),
        "authority": {
            "path": str(path),
            "file_sha256": file_sha256,
            "section_sha256": section_sha256,
        },
    }
    return _signed(body, digest_field="amendment_sha256")


def _binding_for_payload(
    path: Path,
    payload: Mapping[str, Any],
    *,
    digest_field: str,
    status_field: str = "status",
) -> dict[str, Any]:
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256_bytes(encoded),
        "payload_sha256": payload[digest_field],
        "schema": payload["schema"],
        "status": payload[status_field],
    }


def _bind_external_receipt(
    path_value: str | Path,
    *,
    owner: str,
    expected_schema: str | None = None,
    expected_status: str | None = None,
    fixed_file_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    payload = _strict_json(path, owner=owner)
    file_sha256 = _sha256_file(path)
    if fixed_file_sha256 is not None and file_sha256 != fixed_file_sha256:
        raise RequestAuthoringError(f"{owner} immutable file identity drifted")
    schema = payload.get("schema")
    status = payload.get("status", payload.get("terminal_status"))
    if expected_schema is not None and schema != expected_schema:
        raise RequestAuthoringError(f"{owner} schema drifted")
    if expected_status is not None and status != expected_status:
        raise RequestAuthoringError(f"{owner} status is not {expected_status}")
    digest_field = next(
        (
            field
            for field in (
                "receipt_payload_sha256",
                "plan_payload_sha256",
                "receipt_sha256",
            )
            if field in payload
        ),
        None,
    )
    if digest_field is None:
        raise RequestAuthoringError(f"{owner} has no authenticated payload digest")
    body = dict(payload)
    observed = body.pop(digest_field)
    if observed != _sha256_bytes(canonical_json_bytes(body)):
        raise RequestAuthoringError(f"{owner} payload digest is invalid")
    return {
        "path": str(path),
        "file_sha256": file_sha256,
        "payload_sha256": observed,
        "schema": schema,
        "status": status,
    }


def _validate_predecessor_r5_projection(payload: Mapping[str, Any]) -> None:
    try:
        phase_state = tuple(
            (
                row["phase"],
                row["status"],
                row["attempt_count"],
                row["return_code"],
                row["command_sha256"],
                row["launched_command_sha256"],
            )
            for row in payload["phase_records"]
        )
        cleanup = payload["bounded_cleanup"]
        cleanup_records = cleanup["records"]
        recovery = payload["final_gpu_recovery"]
        cost = payload["cost"]
        exact = (
            payload["schema"] == "coordexp-swift-wave7-exact-resume-sequence-receipt-v4"
            and payload["status"] == "failed"
            and payload["failure"]["phase"] == "uninterrupted"
            and payload["failure"]["code"] == "wave7_sequence.phase_failed"
            and payload["plan"] == FROZEN_R5_PLAN
            and payload["marker"] == FROZEN_R5_MARKER
            and phase_state == FROZEN_R5_PHASE_STATE
            and payload["pre_child_receipt"] is None
            and payload["final_receipt"] is None
            and cleanup["policy"] == "TERM_KILL_reap_started_process_group_only"
            and cleanup["artifacts_deleted"] is False
            and len(cleanup_records) == 1
            and cleanup_records[0]["phase"] == "uninterrupted"
            and cleanup_records[0]["reaped"] is True
            and cleanup_records[0]["remaining_pids"] == []
            and cleanup_records[0]["remaining_pgids"] == []
            and _sha256_bytes(canonical_json_bytes(cleanup))
            == FROZEN_R5_BOUNDED_CLEANUP_SHA256
            and recovery["phase"] == "preterminal"
            and recovery["stability_seconds"] == 2.0
            and recovery["sample_monotonic_ns"]
            == [32590162785217312, 32590165072888582]
            and len(recovery["samples"]) == 2
            and all(
                len(sample["compute_inventory"]) == 8 for sample in recovery["samples"]
            )
            and _sha256_bytes(canonical_json_bytes(recovery))
            == FROZEN_R5_FINAL_GPU_RECOVERY_SHA256
            and cost
            == {
                "gpu_device_count": 8,
                "gpu_device_seconds": 160.753348712,
                "gpu_device_seconds_cap": 14400.0,
                "wall_seconds": 51.002604775,
                "wall_seconds_cap": 2400.0,
            }
        )
    except (KeyError, TypeError):
        exact = False
    if not exact:
        raise RequestAuthoringError(
            "predecessor r5 failure decision-bearing projection drifted"
        )


def _bind_predecessor_sequence_failure(path_value: str | Path) -> dict[str, Any]:
    requested = Path(path_value).expanduser()
    if requested.is_symlink():
        raise RequestAuthoringError(
            "predecessor r5 failure is not the canonical regular file"
        )
    path = requested.resolve()
    if path != FROZEN_R5_FAILURE_PATH:
        raise RequestAuthoringError("predecessor r5 failure is not the canonical file")
    payload = _strict_json(path, owner="predecessor r5 failure")
    file_sha256 = _sha256_file(path)
    if file_sha256 != FROZEN_R5_FAILURE_FILE_SHA256:
        raise RequestAuthoringError(
            "predecessor r5 failure immutable file identity drifted"
        )
    body = dict(payload)
    observed = body.pop("receipt_payload_sha256", None)
    if (
        observed is None
        or observed != _sha256_bytes(canonical_json_bytes(body))
        or observed != FROZEN_R5_FAILURE_PAYLOAD_SHA256
    ):
        raise RequestAuthoringError(
            "predecessor r5 failure authenticated payload identity drifted"
        )
    _validate_predecessor_r5_projection(payload)
    return {
        "path": str(path),
        "file_sha256": file_sha256,
        "payload_sha256": observed,
        "schema": payload["schema"],
        "status": payload["status"],
    }


def _bind_immutable_r6_preflight_leaf(
    path_value: str | Path,
    *,
    owner: str,
    canonical_path: Path,
    frozen_file_sha256: str,
    frozen_payload_sha256: str,
    digest_field: str,
    expected_schema: str,
    expected_status: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    requested = Path(path_value).expanduser()
    if requested.is_symlink() or requested.resolve() != canonical_path:
        raise RequestAuthoringError(f"{owner} is not the canonical file")
    payload = _strict_json(canonical_path, owner=owner)
    file_sha256 = _sha256_file(canonical_path)
    body = dict(payload)
    observed = body.pop(digest_field, None)
    if (
        file_sha256 != frozen_file_sha256
        or observed != frozen_payload_sha256
        or observed != _sha256_bytes(canonical_json_bytes(body))
        or payload.get("schema") != expected_schema
        or payload.get("status") != expected_status
    ):
        raise RequestAuthoringError(f"{owner} immutable identity drifted")
    return payload, {
        "path": str(canonical_path),
        "file_sha256": file_sha256,
        "payload_sha256": observed,
        "schema": payload["schema"],
        "status": payload["status"],
    }


def _bind_predecessor_preflight_failure(
    plan_path: str | Path,
    marker_path: str | Path,
    terminal_path: str | Path,
) -> dict[str, Any]:
    plan_payload, plan = _bind_immutable_r6_preflight_leaf(
        plan_path,
        owner="predecessor r6 preflight plan",
        canonical_path=FROZEN_R6_PREFLIGHT_PLAN_PATH,
        frozen_file_sha256=FROZEN_R6_PREFLIGHT_PLAN_FILE_SHA256,
        frozen_payload_sha256=FROZEN_R6_PREFLIGHT_PLAN_PAYLOAD_SHA256,
        digest_field="plan_payload_sha256",
        expected_schema="coordexp-swift-wave7-determinism-preflight-plan-v4",
        expected_status="prepared",
    )
    marker_payload, marker = _bind_immutable_r6_preflight_leaf(
        marker_path,
        owner="predecessor r6 preflight attempt marker",
        canonical_path=FROZEN_R6_PREFLIGHT_MARKER_PATH,
        frozen_file_sha256=FROZEN_R6_PREFLIGHT_MARKER_FILE_SHA256,
        frozen_payload_sha256=FROZEN_R6_PREFLIGHT_MARKER_PAYLOAD_SHA256,
        digest_field="receipt_payload_sha256",
        expected_schema=(
            "coordexp-swift-wave7-determinism-preflight-attempt-start-marker-v4"
        ),
        expected_status="started",
    )
    terminal_payload, terminal = _bind_immutable_r6_preflight_leaf(
        terminal_path,
        owner="predecessor r6 preflight terminal",
        canonical_path=FROZEN_R6_PREFLIGHT_TERMINAL_PATH,
        frozen_file_sha256=FROZEN_R6_PREFLIGHT_TERMINAL_FILE_SHA256,
        frozen_payload_sha256=FROZEN_R6_PREFLIGHT_TERMINAL_PAYLOAD_SHA256,
        digest_field="receipt_payload_sha256",
        expected_schema="coordexp-swift-wave7-r5-determinism-preflight-v4",
        expected_status="failed",
    )
    exact_chain = (
        marker_payload.get("plan_sha256") == FROZEN_R6_PREFLIGHT_PLAN_PAYLOAD_SHA256
        and terminal_payload.get("plan_sha256")
        == FROZEN_R6_PREFLIGHT_PLAN_PAYLOAD_SHA256
        and terminal_payload.get("attempt_marker") == marker
        and terminal_payload.get("launch_count") == 0
        and terminal_payload.get("mismatches") == ["KeyboardInterrupt"]
        and terminal_payload.get("comparisons") is None
        and terminal_payload.get("rank_receipts") == []
        and plan_payload.get("artifact_targets", {}).get("attempt_marker")
        == str(FROZEN_R6_PREFLIGHT_MARKER_PATH)
        and plan_payload.get("artifact_targets", {}).get("terminal_receipt")
        == str(FROZEN_R6_PREFLIGHT_TERMINAL_PATH)
    )
    if not exact_chain:
        raise RequestAuthoringError("predecessor r6 preflight failure chain drifted")
    return {
        "historical_non_executable": True,
        "plan": plan,
        "attempt_marker": marker,
        "terminal_receipt": terminal,
    }


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RequestAuthoringError(f"source inventory path is not regular: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(info.st_size),
        "sha256": _sha256_file(resolved),
    }


def _source_inventory() -> list[dict[str, Any]]:
    accelerate_text = shutil.which("accelerate")
    if accelerate_text is None:
        raise RequestAuthoringError("production accelerate executable is unavailable")
    paths = {
        PYTHON_EXECUTABLE,
        Path(accelerate_text).resolve(),
        TRAIN_ENTRYPOINT,
        SEQUENCE_CONTROLLER,
        INTERRUPT_CONTROLLER,
        COMPARATOR_V1,
        COMPARATOR_V2,
        DETERMINISM_PREFLIGHT,
        Path(__file__).resolve(),
        Path(build_training_input_attestations.__code__.co_filename).resolve(),
        Path(
            base_model_weight_identity_with_execution_policy.__code__.co_filename
        ).resolve(),
    }
    inventory = [_file_identity(path) for path in sorted(paths)]
    parity_path = Path(
        base_model_weight_identity_with_execution_policy.__code__.co_filename
    ).resolve()
    parity_identity = next(row for row in inventory if row["path"] == str(parity_path))
    if parity_identity["sha256"] != FROZEN_PARITY_SOURCE_SHA256:
        raise RequestAuthoringError("frozen parity helper owner identity drifted")
    return inventory


def _train_command(config_path: Path, *, port: int) -> list[str]:
    accelerate = shutil.which("accelerate")
    if accelerate is None:
        raise RequestAuthoringError("production accelerate executable is unavailable")
    return [
        str(Path(accelerate).resolve()),
        "launch",
        "--multi_gpu",
        "--num_processes",
        "8",
        "--main_process_port",
        str(port),
        "--module",
        "src.train",
        "--config",
        str(config_path.resolve()),
    ]


def _expected_commands(
    *,
    configs: Mapping[str, Path],
    roots: Mapping[str, Path],
    targets: Mapping[str, Path],
    phase_timeout_seconds: float,
    term_grace_seconds: float,
    kill_grace_seconds: float,
    baseline_stability_seconds: float,
    provenance_sha256: str,
) -> dict[str, list[str]]:
    trains = {
        role: _train_command(configs[role], port=TRAIN_PORTS[role])
        for role in RUN_ROLES
    }
    identity_tail = [
        "--expected-interrupt-source-sha256",
        _sha256_file(INTERRUPT_CONTROLLER),
        "--expected-source-sha256",
        _sha256_file(COMPARATOR_V2),
        "--expected-provenance-sha256",
        provenance_sha256,
    ]
    interrupted_config_args = [
        item
        for source in load_train_config(configs["interrupted_parent"]).sources
        for item in ("--config", str(source.path))
    ]
    common = [
        "--uninterrupted-run-dir",
        str(roots["uninterrupted"]),
        "--interrupted-parent-run-dir",
        str(roots["interrupted_parent"]),
    ]
    return {
        "uninterrupted": trains["uninterrupted"],
        "interrupted_parent": [
            str(PYTHON_EXECUTABLE),
            str(INTERRUPT_CONTROLLER.resolve()),
            "--parent-run-dir",
            str(roots["interrupted_parent"]),
            "--expected-checkpoint-step",
            "3",
            "--marker",
            str(targets["interruption_marker"]),
            "--receipt",
            str(targets["interruption_receipt"]),
            "--timeout-seconds",
            str(phase_timeout_seconds),
            "--term-grace-seconds",
            str(term_grace_seconds),
            "--kill-grace-seconds",
            str(kill_grace_seconds),
            "--stability-seconds",
            str(baseline_stability_seconds),
            "--poll-seconds",
            "0.01",
            "--source",
            str(INTERRUPT_CONTROLLER.resolve()),
            "--source",
            str(TRAIN_ENTRYPOINT.resolve()),
            "--source",
            str(COMPARATOR_V1.resolve()),
            "--source",
            str(COMPARATOR_V2.resolve()),
            *interrupted_config_args,
            "--",
            *trains["interrupted_parent"],
        ],
        "pre_child": [
            str(PYTHON_EXECUTABLE),
            str(COMPARATOR_V2.resolve()),
            "pre-child",
            *common,
            "--interruption-marker",
            str(targets["interruption_marker"]),
            "--termination-receipt",
            str(targets["interruption_receipt"]),
            "--output",
            str(targets["pre_child_receipt"]),
            *identity_tail,
        ],
        "verify_pre_child": [
            str(PYTHON_EXECUTABLE),
            str(COMPARATOR_V2.resolve()),
            "verify-pre-child",
            "--receipt",
            str(targets["pre_child_receipt"]),
            "--expected-payload-sha256",
            PRE_CHILD_DIGEST_PLACEHOLDER,
        ],
        "resume_child": [
            str(PYTHON_EXECUTABLE),
            str(SEQUENCE_CONTROLLER.resolve()),
            "launch-child",
            "--receipt",
            str(targets["pre_child_receipt"]),
            "--expected-payload-sha256",
            PRE_CHILD_DIGEST_PLACEHOLDER,
            "--config",
            str(configs["resume_child"]),
            "--run-root",
            str(roots["resume_child"]),
            "--parent-run-root",
            str(roots["interrupted_parent"]),
            "--",
            *trains["resume_child"],
        ],
        "final_compare": [
            str(PYTHON_EXECUTABLE),
            str(COMPARATOR_V2.resolve()),
            "compare",
            *common,
            "--resume-child-run-dir",
            str(roots["resume_child"]),
            "--interruption-marker",
            str(targets["interruption_marker"]),
            "--termination-receipt",
            str(targets["interruption_receipt"]),
            "--output",
            str(targets["final_receipt"]),
            *identity_tail,
        ],
    }


def _validate_configs_and_roots(
    configs: Mapping[str, Path], roots: Mapping[str, Path]
) -> None:
    for role in RUN_ROLES:
        config = load_train_config(configs[role]).config
        run_root = (
            Path(config.run.artifact_root) / (config.run.output_dir or config.run.name)
        ).resolve()
        expected_checkpoint = (
            roots["interrupted_parent"] / "checkpoints" / "step-3"
            if role == "resume_child"
            else None
        )
        observed_checkpoint = (
            None
            if config.resume.checkpoint_dir is None
            else Path(config.resume.checkpoint_dir).resolve()
        )
        if (
            config.run.name != role
            or config.run.collision_policy != "fail"
            or run_root != roots[role]
            or config.resume.mode != "exact_same_world_size"
            or observed_checkpoint != expected_checkpoint
            or config.runtime.determinism.mode != "strict_cuda_replay_v1"
            or config.training.max_steps != 5
            or config.training.forward_input_provider_mode != "synchronous"
            or config.eval.forward.every_fraction is not None
            or config.eval.forward.steps != (3,)
            or config.checkpoint.every_fraction is not None
            or config.checkpoint.steps != (3, 5)
            or config.checkpoint.save_final is not True
        ):
            raise RequestAuthoringError(
                f"{role} config does not match the frozen r7 grammar"
            )


def _validate_path_topology(
    roots: Mapping[str, Path], targets: Mapping[str, Path]
) -> None:
    rows = [(f"run_roots.{name}", path) for name, path in roots.items()]
    rows.extend((f"targets.{name}", path) for name, path in targets.items())
    for index, (left_name, left) in enumerate(rows):
        for right_name, right in rows[index + 1 :]:
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                raise RequestAuthoringError(
                    f"request roots/targets overlap: {left_name} and {right_name}"
                )
    for name, path in rows:
        try:
            assert_absent_artifact_target(path)
        except Exception as exc:
            raise RequestAuthoringError(
                f"request root/target is not absent and publishable: {name}"
            ) from exc


def _validate_r7_request_namespace(
    args: argparse.Namespace,
    configs: Mapping[str, Path],
    roots: Mapping[str, Path],
    targets: Mapping[str, Path],
) -> None:
    raw_paths = [
        *(Path(getattr(args, f"{role}_config")).expanduser() for role in RUN_ROLES),
        *(Path(getattr(args, f"{role}_run_root")).expanduser() for role in RUN_ROLES),
        *(Path(getattr(args, name)).expanduser() for name in R7_TARGETS),
        Path(args.cache_root).expanduser(),
        Path(args.cache_preparation_receipt).expanduser(),
        Path(args.runtime_receipt).expanduser(),
        Path(args.determinism_preflight_plan).expanduser(),
        Path(args.determinism_preflight_receipt).expanduser(),
    ]
    if any(path.is_symlink() for path in raw_paths):
        raise RequestAuthoringError(
            "request path is not in the exact fresh r7 namespace"
        )
    exact = (
        dict(configs) == R7_CONFIG_PATHS
        and dict(roots) == R7_RUN_ROOTS
        and dict(targets) == R7_TARGETS
        and Path(args.cache_root).expanduser().resolve() == R7_CACHE_ROOT
        and Path(args.cache_preparation_receipt).expanduser().resolve()
        == R7_CACHE_PREPARATION_RECEIPT
        and Path(args.runtime_receipt).expanduser().resolve() == R7_RUNTIME_RECEIPT
        and Path(args.determinism_preflight_plan).expanduser().resolve()
        == R7_DETERMINISM_PREFLIGHT_PLAN
        and Path(args.determinism_preflight_receipt).expanduser().resolve()
        == R7_DETERMINISM_PREFLIGHT_RECEIPT
    )
    if not exact:
        raise RequestAuthoringError(
            "request paths do not match the exact fresh r7 namespace"
        )


def _positive_cost(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise RequestAuthoringError(f"{field} must be a positive finite number")
    result = float(value)
    if result == float("inf") or result != result:
        raise RequestAuthoringError(f"{field} must be a positive finite number")
    return result


def _validate_authorized_costs(args: argparse.Namespace) -> None:
    for field in (
        "phase_timeout_seconds",
        "term_grace_seconds",
        "kill_grace_seconds",
        "gpu_baseline_stability_seconds",
        "gpu_post_cleanup_stability_seconds",
        "max_total_wall_seconds",
        "max_total_gpu_device_seconds",
    ):
        setattr(args, field, _positive_cost(getattr(args, field), field=field))
    if args.phase_timeout_seconds > 600:
        raise RequestAuthoringError("phase_timeout_seconds exceeds r7 authority")
    if args.max_total_wall_seconds > 2400:
        raise RequestAuthoringError("max_total_wall_seconds exceeds r7 authority")
    if args.max_total_gpu_device_seconds > 14400:
        raise RequestAuthoringError("max_total_gpu_device_seconds exceeds r7 authority")
    if args.max_total_gpu_device_seconds > args.max_total_wall_seconds * 8:
        raise RequestAuthoringError(
            "max_total_gpu_device_seconds exceeds the eight-device wall-time envelope"
        )


def _request_payload(
    args: argparse.Namespace,
    *,
    amendment: dict[str, Any],
    cache: dict[str, Any],
    model: dict[str, Any],
    leaf_paths: Mapping[str, Path],
) -> dict[str, Any]:
    configs = {
        role: Path(getattr(args, f"{role}_config")).expanduser().resolve()
        for role in RUN_ROLES
    }
    roots = {
        role: Path(getattr(args, f"{role}_run_root")).expanduser().resolve()
        for role in RUN_ROLES
    }
    targets = {
        name: Path(getattr(args, name)).expanduser().resolve()
        for name in (
            "sequence_marker",
            "sequence_receipt",
            "publication_failure_sidecar",
            "interruption_marker",
            "interruption_receipt",
            "pre_child_receipt",
            "final_receipt",
        )
    }
    _validate_r7_request_namespace(args, configs, roots, targets)
    _validate_configs_and_roots(configs, roots)
    _validate_path_topology(roots, targets)
    _validate_authorized_costs(args)
    external = {
        "legacy_r4_failure": _bind_external_receipt(
            args.legacy_r4_failure,
            owner="legacy r4 failure",
            fixed_file_sha256=FROZEN_R4_FAILURE_FILE_SHA256,
        ),
        "predecessor_sequence_failure": _bind_predecessor_sequence_failure(
            args.predecessor_r5_failure
        ),
        "predecessor_preflight_failure": _bind_predecessor_preflight_failure(
            args.predecessor_r6_preflight_plan,
            args.predecessor_r6_preflight_marker,
            args.predecessor_r6_preflight_terminal,
        ),
        "determinism_preflight_plan": _bind_external_receipt(
            args.determinism_preflight_plan,
            owner="determinism preflight plan",
            expected_schema="coordexp-swift-wave7-determinism-preflight-plan-v4",
            expected_status="prepared",
        ),
        "determinism_preflight": _bind_external_receipt(
            args.determinism_preflight_receipt,
            owner="determinism preflight receipt",
            expected_schema="coordexp-swift-wave7-r5-determinism-preflight-v4",
            expected_status="passed",
        ),
        "runtime_receipt": _bind_external_receipt(
            args.runtime_receipt,
            owner="runtime receipt",
            expected_schema="coordexp-swift-wave7-r5-runtime-admission-v1",
            expected_status="passed",
        ),
    }
    runtime_payload = _strict_json(Path(args.runtime_receipt), owner="runtime receipt")
    if (
        runtime_payload.get("model_loaded") is not False
        or runtime_payload.get("cuda_initialized") is not False
    ):
        raise RequestAuthoringError(
            "runtime receipt is not model-free pre-CUDA evidence"
        )
    provenance = runtime_payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RequestAuthoringError("runtime receipt omits executed provenance")
    provenance_sha256 = _sha256_bytes(canonical_json_bytes(dict(provenance)))
    repository = provenance.get("repository")
    execution_digest = (
        repository.get("execution_relevant_digest")
        if isinstance(repository, Mapping)
        else None
    )
    compare_provenance_sha256 = (
        execution_digest.get("value")
        if isinstance(execution_digest, Mapping)
        and execution_digest.get("status") == "available"
        else None
    )
    if (
        not isinstance(compare_provenance_sha256, str)
        or len(compare_provenance_sha256) != 64
        or any(character not in "0123456789abcdef" for character in compare_provenance_sha256)
    ):
        raise RequestAuthoringError(
            "runtime receipt omits the available execution-relevant provenance digest"
        )
    commands = _expected_commands(
        configs=configs,
        roots=roots,
        targets=targets,
        phase_timeout_seconds=args.phase_timeout_seconds,
        term_grace_seconds=args.term_grace_seconds,
        kill_grace_seconds=args.kill_grace_seconds,
        baseline_stability_seconds=args.gpu_baseline_stability_seconds,
        provenance_sha256=compare_provenance_sha256,
    )
    body = {
        "schema": REQUEST_SCHEMA,
        "status": "prepared",
        "amendment": _binding_for_payload(
            leaf_paths["amendment"], amendment, digest_field="amendment_sha256"
        ),
        **external,
        "cache_input_attestation": _binding_for_payload(
            leaf_paths["cache"], cache, digest_field="attestation_sha256"
        ),
        "model_input_attestation": _binding_for_payload(
            leaf_paths["model"], model, digest_field="attestation_sha256"
        ),
        "source_inventory": _source_inventory(),
        "config_paths": {role: str(configs[role]) for role in RUN_ROLES},
        "provenance_sha256": provenance_sha256,
        "environment": {
            **ENVIRONMENT,
            "COORDEXP_SWIFT_PACK_CACHE_ROOT": str(
                Path(args.cache_root).expanduser().resolve()
            ),
        },
        "run_roots": {role: str(roots[role]) for role in RUN_ROLES},
        "targets": {name: str(path) for name, path in targets.items()},
        "commands": commands,
        "oracles": {
            "pre_child": {
                "schema": PRE_CHILD_SCHEMA,
                "status": "passed",
                "child_launch_authorized": True,
                "mismatches": [],
            },
            "final": {"schema": FINAL_SCHEMA, "status": "passed", "mismatches": []},
        },
        "policy": {
            "phase_order": list(PHASE_ORDER),
            "gpu_phases": list(GPU_PHASES),
            "gpu_admission_mode": "shared_preexisting_subset_v1",
            "gpu_memory_total_mib": 81_920,
            "gpu_memory_used_ceiling_mib": 49_152,
            "gpu_memory_headroom_floor_mib": 32_768,
            "no_retry": True,
            "max_attempts_per_phase": 1,
            "preserve_failed_artifacts": True,
            "phase_timeout_seconds": {
                phase: args.phase_timeout_seconds for phase in PHASE_ORDER
            },
            "term_grace_seconds": args.term_grace_seconds,
            "kill_grace_seconds": args.kill_grace_seconds,
            "gpu_baseline_stability_seconds": args.gpu_baseline_stability_seconds,
            "gpu_post_cleanup_stability_seconds": args.gpu_post_cleanup_stability_seconds,
            "max_total_wall_seconds": args.max_total_wall_seconds,
            "max_total_gpu_device_seconds": args.max_total_gpu_device_seconds,
        },
    }
    return _signed(body, digest_field="request_payload_sha256")


def _publish_bundle(
    *, output_root: str | Path, payloads: Mapping[str, Mapping[str, Any]]
) -> dict[str, Path]:
    if set(payloads) != set(LEAF_NAMES):
        raise RequestAuthoringError("publication bundle leaf inventory is not exact")
    requested_root = Path(output_root).expanduser()
    if requested_root.is_symlink():
        raise RequestAuthoringError(
            "output root must be an existing non-symlink directory"
        )
    root = requested_root.resolve()
    if root != R7_SEQUENCE_ROOT:
        raise RequestAuthoringError(
            "output root does not match the exact fresh r7 namespace"
        )
    if not root.is_dir():
        raise RequestAuthoringError(
            "output root must be an existing non-symlink directory"
        )
    paths = {name: root / filename for name, filename in LEAF_NAMES.items()}
    for path in paths.values():
        assert_absent_artifact_target(path)
    for name in ("amendment", "cache", "model"):
        write_strict_json_atomic(paths[name], payloads[name])
        observed = _strict_json(paths[name], owner=f"published {name} leaf")
        if observed != dict(payloads[name]):
            raise RequestAuthoringError(f"published {name} leaf failed strict readback")
    request_linked = [False]
    try:
        write_strict_json_atomic(
            paths["request"],
            payloads["request"],
            on_linked=lambda: request_linked.__setitem__(0, True),
        )
    except BaseException as exc:
        if not request_linked[0]:
            raise
        # The absent-only request link is the final bundle commit point. A later
        # durability sync failure must not invite a retry of an owned request.
        observed = _strict_json(paths["request"], owner="committed request leaf")
        if observed != dict(payloads["request"]):
            raise RequestAuthoringError(
                "committed request leaf failed strict readback"
            ) from exc
    else:
        observed = _strict_json(paths["request"], owner="published request leaf")
        if observed != dict(payloads["request"]):
            raise RequestAuthoringError("published request leaf failed strict readback")
    return paths


def _author(args: argparse.Namespace) -> dict[str, Path]:
    requested_output_root = Path(args.output_root).expanduser()
    if requested_output_root.is_symlink():
        raise RequestAuthoringError(
            "output root does not match the exact fresh r7 namespace"
        )
    output_root = requested_output_root.resolve()
    leaf_paths = {name: output_root / filename for name, filename in LEAF_NAMES.items()}
    for path in leaf_paths.values():
        assert_absent_artifact_target(path)
    amendment = _build_amendment_payload(args.amendment_authority)
    configs = {role: getattr(args, f"{role}_config") for role in RUN_ROLES}
    cache, model = build_training_input_attestations(
        config_paths=configs,
        cache_root=args.cache_root,
        cache_preparation_receipt_path=args.cache_preparation_receipt,
        expected_model_root=args.model_root,
        max_cache_payload_bytes=args.max_cache_payload_bytes,
    )
    request = _request_payload(
        args,
        amendment=amendment,
        cache=cache,
        model=model,
        leaf_paths=leaf_paths,
    )
    return _publish_bundle(
        output_root=output_root,
        payloads={
            "amendment": amendment,
            "cache": cache,
            "model": model,
            "request": request,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    author = subparsers.add_parser(
        "author", help="Publish one absent-only request-v6 bundle."
    )
    author.add_argument("--amendment-authority", required=True)
    for role in RUN_ROLES:
        author.add_argument(
            f"--{role.replace('_', '-')}-config", dest=f"{role}_config", required=True
        )
    author.add_argument("--cache-root", required=True)
    author.add_argument("--cache-preparation-receipt", required=True)
    author.add_argument("--model-root", required=True)
    author.add_argument("--legacy-r4-failure", required=True)
    author.add_argument("--predecessor-r5-failure", required=True)
    author.add_argument("--predecessor-r6-preflight-plan", required=True)
    author.add_argument("--predecessor-r6-preflight-marker", required=True)
    author.add_argument("--predecessor-r6-preflight-terminal", required=True)
    author.add_argument("--determinism-preflight-plan", required=True)
    author.add_argument("--determinism-preflight-receipt", required=True)
    author.add_argument("--runtime-receipt", required=True)
    author.add_argument("--output-root", required=True)
    for role in RUN_ROLES:
        author.add_argument(
            f"--{role.replace('_', '-')}-run-root",
            dest=f"{role}_run_root",
            required=True,
        )
    for name in (
        "sequence_marker",
        "sequence_receipt",
        "publication_failure_sidecar",
        "interruption_marker",
        "interruption_receipt",
        "pre_child_receipt",
        "final_receipt",
    ):
        author.add_argument(f"--{name.replace('_', '-')}", dest=name, required=True)
    author.add_argument("--max-cache-payload-bytes", type=int, required=True)
    author.add_argument("--phase-timeout-seconds", type=float, required=True)
    author.add_argument("--term-grace-seconds", type=float, required=True)
    author.add_argument("--kill-grace-seconds", type=float, required=True)
    author.add_argument("--gpu-baseline-stability-seconds", type=float, required=True)
    author.add_argument(
        "--gpu-post-cleanup-stability-seconds", type=float, required=True
    )
    author.add_argument("--max-total-wall-seconds", type=float, required=True)
    author.add_argument("--max-total-gpu-device-seconds", type=float, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "author":
        raise RequestAuthoringError("unsupported request producer command")
    paths = _author(args)
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

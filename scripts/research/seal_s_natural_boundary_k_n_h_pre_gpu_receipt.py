#!/usr/bin/env python3
"""Seal the S natural-boundary K/N/H cohort before any model load.

This is a CPU-only launch boundary.  The sealer consumes already materialized
JSON artifacts and exact source-file identities; it never imports torch,
creates execution roots, or loads a checkpoint.  It deterministically
inventories the selected base-model payload and ``src/**/*.py`` runtime tree so
that unlisted loader code or model files cannot drift.  A receipt is canonical
JSON with a self hash and is write-once.  ``validate_receipt`` is intended for
the cohort executor and shard runner and re-hashes every bound input before
CUDA/model construction.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
LEGACY_CONTEXT_UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
SCHEMA_VERSION = "s_natural_boundary_k_n_h_pre_gpu_receipt.v1"
FOCUSED_TEST_EVIDENCE_SCHEMA_VERSION = "s_natural_boundary_k_n_h_focused_test_receipt.v1"
RUNTIME_EVIDENCE_SCHEMA_VERSION = "s_natural_boundary_k_n_h_runtime_evidence.v1"
EVIDENCE_MATERIALIZER_ROLE = "evidence_materializer"
STATUS = "sealed_pre_gpu"
CHECKPOINT = "S"
STEP = 2444
SUBSTRATE = "four-coordinate geo_sorted_xy"
SHARD_COUNT = 8
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ARM_ORDER = (
    "K00", "K01", "K10", "K11", "K12", "K13", "K14T", "K14B",
    "N00", "N01", "N10", "N20", "H00", "H10", "H20",
)
CODE_ROLES = (
    "frozen_gate",
    "probe",
    "attention_actuator",
    "residual_actuator",
    "cohort_validator",
    "live_executor",
    "legacy_owner_interface_runner",
    "shard_planner",
    "shard_runner",
    "merger",
    "analyzer_finalizer",
    "sealer",
    EVIDENCE_MATERIALIZER_ROLE,
)
CODE_ROLE_PATHS = {
    "frozen_gate": REPO_ROOT / "scripts/research/run_s_primary_natural_boundary_gate.py",
    "probe": REPO_ROOT / "scripts/research/run_natural_boundary_routing_history_probe.py",
    "attention_actuator": REPO_ROOT / "scripts/research/natural_boundary_attention_actuators.py",
    "residual_actuator": REPO_ROOT / "scripts/research/natural_boundary_residual_actuators.py",
    "cohort_validator": REPO_ROOT / "scripts/research/run_s_natural_boundary_k_n_h_cohort.py",
    "live_executor": REPO_ROOT / "scripts/research/s_natural_boundary_k_n_h_live_executor.py",
    "legacy_owner_interface_runner": REPO_ROOT / "scripts/research/run_static_dynamic_owner_interface_experiment.py",
    "shard_planner": REPO_ROOT / "scripts/research/plan_s_natural_boundary_k_n_h_execution.py",
    "shard_runner": REPO_ROOT / "scripts/research/run_s_natural_boundary_k_n_h_shard.py",
    "merger": REPO_ROOT / "scripts/research/merge_s_natural_boundary_k_n_h_shards.py",
    "analyzer_finalizer": REPO_ROOT / "scripts/research/analyze_s_natural_boundary_k_n_h_evidence.py",
    "sealer": REPO_ROOT / "scripts/research/seal_s_natural_boundary_k_n_h_pre_gpu_receipt.py",
    EVIDENCE_MATERIALIZER_ROLE: REPO_ROOT / "scripts/research/materialize_s_natural_boundary_k_n_h_pre_gpu_evidence.py",
}
# The executor identity schema predates the CPU evidence materializer.  Keep
# its exact runtime contract stable; the materializer remains bound in the
# parent pre-GPU receipt and both evidence artifacts.
RUNTIME_IDENTITY_CODE_ROLES = tuple(
    role for role in CODE_ROLES if role != EVIDENCE_MATERIALIZER_ROLE
)
H0_IDENTITY_FILE_ROLES = (
    "resolved_config",
    "run_manifest",
    "summary",
    "pred_token_trace",
    "image_plan",
    "adapter_config",
    "adapter_tensor",
    "embedding_metadata",
    "embedding_tensor",
)
H0_LOADER_RELATIVE_PATHS = {
    "resolved_config": Path("configs/resolved.json"),
    "run_manifest": Path("run_manifest.json"),
    "summary": Path("summary.json"),
    "pred_token_trace": Path("pred_token_trace.jsonl"),
    "image_plan": Path("image_plan.jsonl"),
}
BASE_MODEL_FILE_ROLES = (
    "config",
    "model_index",
    "weight_shard_1",
    "weight_shard_2",
    "tokenizer",
    "tokenizer_config",
    "added_tokens",
    "special_tokens_map",
    "chat_template_jinja",
    "chat_template_json",
    "preprocessor_config",
)
BASE_MODEL_ROLE_BASENAMES = {
    "config": "config.json",
    "model_index": "model.safetensors.index.json",
    "weight_shard_1": "model-00001-of-00002.safetensors",
    "weight_shard_2": "model-00002-of-00002.safetensors",
    "tokenizer": "tokenizer.json",
    "tokenizer_config": "tokenizer_config.json",
    "added_tokens": "added_tokens.json",
    "special_tokens_map": "special_tokens_map.json",
    "chat_template_jinja": "chat_template.jinja",
    "chat_template_json": "chat_template.json",
    "preprocessor_config": "preprocessor_config.json",
}
SHARD_PHYSICAL_DEVICES = {
    f"shard-{index:03d}": str(index) for index in range(SHARD_COUNT)
}
FROZEN_ELIGIBILITY_PREDICATES = frozenset(
    {
        "checkpoint",
        "native_fn",
        "strict_complete_row",
        "natural_boundary_valid",
        "eligible_except_support",
        "verified_support",
        "covered_owner_ids_nonempty",
        "geometry_launch_eligible",
    }
)
DEVICE_POLICY = {
    "device_count": 1,
    "logical_device": "cuda:0",
    "shard_physical_devices": SHARD_PHYSICAL_DEVICES,
}
REQUIRED_FOCUSED_TEST_PATHS = (
    REPO_ROOT / "tests/research/test_run_s_primary_natural_boundary_gate.py",
    REPO_ROOT / "tests/research/test_run_natural_boundary_routing_history_probe.py",
    REPO_ROOT / "tests/research/test_natural_boundary_attention_actuators.py",
    REPO_ROOT / "tests/research/test_natural_boundary_residual_actuators.py",
    REPO_ROOT / "tests/research/test_run_s_natural_boundary_k_n_h_cohort.py",
    REPO_ROOT / "tests/research/test_run_static_dynamic_owner_interface_experiment.py",
    REPO_ROOT / "tests/research/test_s_natural_boundary_k_n_h_live_executor.py",
    REPO_ROOT / "tests/research/test_s_natural_boundary_k_n_h_sharded_execution.py",
    REPO_ROOT / "tests/research/test_analyze_s_natural_boundary_k_n_h_evidence.py",
    REPO_ROOT / "tests/research/test_seal_s_natural_boundary_k_n_h_pre_gpu_receipt.py",
)
FOCUSED_TEST_COMMAND = "python -m pytest -q " + " ".join(
    str(path) for path in REQUIRED_FOCUSED_TEST_PATHS
)
GATE_ARTIFACTS = ("result", "runtime_identity", "terminal_summary", "launch_log")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WILDCARD_CHARS = frozenset("*?[]")


class PreGpuReceiptError(ValueError):
    """Raised when a pre-GPU cohort receipt is incomplete or drifted."""


ReceiptError = PreGpuReceiptError


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize only finite, deterministic JSON values."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreGpuReceiptError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def document_self_sha256(document: Mapping[str, Any]) -> str:
    body = dict(document)
    body.pop("self_sha256", None)
    return sha256_json(body)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PreGpuReceiptError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreGpuReceiptError(f"{label} must be a non-empty string")
    return value.strip()


def _reject_wildcards(value: Any, label: str) -> None:
    if isinstance(value, str) and any(char in value for char in WILDCARD_CHARS):
        raise PreGpuReceiptError(f"{label} uses wildcard/discovery syntax")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_wildcards(nested, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_wildcards(nested, f"{label}[{index}]")


def _absolute(value: str | Path, label: str) -> Path:
    _reject_wildcards(str(value), label)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise PreGpuReceiptError(f"{label} must be an absolute path")
    # ``resolve`` is safe only after checking every lexical component.  A
    # symlink anywhere in the chain would make the identity mutable.
    cursor = candidate
    while True:
        if cursor.is_symlink():
            raise PreGpuReceiptError(f"{label} must not traverse a symlink: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise PreGpuReceiptError(f"cannot resolve {label}: {candidate}") from exc
    if resolved != candidate:
        raise PreGpuReceiptError(f"{label} resolves through a symlink: {candidate}")
    return resolved


def _regular_file(value: str | Path, label: str) -> Path:
    path = _absolute(value, label)
    if path.is_symlink() or not path.is_file():
        raise PreGpuReceiptError(f"{label} must be an existing regular non-symlink file: {path}")
    return path


def _directory(value: str | Path, label: str, *, exists: bool = True) -> Path:
    path = _absolute(value, label)
    if path.is_symlink() or (exists and not path.is_dir()) or (not exists and path.exists()):
        state = "existing regular non-symlink directory" if exists else "absent non-symlink path"
        raise PreGpuReceiptError(f"{label} must be {state}: {path}")
    return path


def sha256_file(value: str | Path) -> str:
    path = _regular_file(value, "hash source")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PreGpuReceiptError(f"cannot read {path}") from exc
    return digest.hexdigest()


def _regular_file_inventory(
    root_value: str | Path,
    label: str,
    *,
    suffix: str | None = None,
    known_file_refs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Hash one complete, deterministic regular-file inventory.

    ``suffix`` limits the owned surface (the runtime tree is exactly
    ``src/**/*.py``); the base-model inventory deliberately has no filter.
    Symlink directories and in-scope symlink files are rejected rather than
    silently omitted.
    """

    root = _directory(root_value, label)
    known = dict(known_file_refs or {})
    entries: list[dict[str, Any]] = []
    try:
        for current_raw, directory_names, file_names in os.walk(root, followlinks=False):
            current = Path(current_raw)
            directory_names.sort()
            file_names.sort()
            for name in directory_names:
                candidate = current / name
                if candidate.is_symlink():
                    raise PreGpuReceiptError(
                        f"{label} must not contain a symlink directory: {candidate}"
                    )
            for name in file_names:
                candidate = current / name
                if suffix is not None and candidate.suffix != suffix:
                    continue
                if candidate.is_symlink() or not candidate.is_file():
                    raise PreGpuReceiptError(
                        f"{label} contains a non-regular or symlink file: {candidate}"
                    )
                relative = candidate.relative_to(root).as_posix()
                prior = known.get(str(candidate))
                if prior is not None:
                    digest = _sha(prior.get("sha256"), f"{label} known file SHA-256")
                    size_bytes = prior.get("size_bytes")
                    if not isinstance(size_bytes, int) or size_bytes < 0:
                        raise PreGpuReceiptError(f"{label} known file size is invalid")
                else:
                    digest = sha256_file(candidate)
                    size_bytes = candidate.stat().st_size
                entries.append(
                    {
                        "relative_path": relative,
                        "sha256": digest,
                        "size_bytes": size_bytes,
                    }
                )
    except OSError as exc:
        raise PreGpuReceiptError(f"cannot inventory {label}: {root}") from exc
    entries.sort(key=lambda item: item["relative_path"])
    if not entries:
        raise PreGpuReceiptError(f"{label} inventory is empty")
    return {
        "root": str(root),
        "file_count": len(entries),
        "files": entries,
        "inventory_sha256": sha256_json(entries),
    }


def _validate_inventory(
    value: Any,
    label: str,
    *,
    expected_root: str | Path,
    suffix: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreGpuReceiptError(f"{label} inventory is missing")
    observed = _regular_file_inventory(expected_root, label, suffix=suffix)
    if dict(value) != observed:
        raise PreGpuReceiptError(f"{label} inventory added, missing, or drifted")
    return observed


def _path_value(value: Any, label: str) -> tuple[Path, str | None]:
    if isinstance(value, (str, Path)):
        return _regular_file(value, label), None
    if not isinstance(value, Mapping):
        raise PreGpuReceiptError(f"{label} must be an exact path/ref object")
    path = value.get("path") or value.get("file")
    if path is None:
        raise PreGpuReceiptError(f"{label}.path is missing")
    return _regular_file(path, label), value.get("sha256")


def _file_ref(value: Any, label: str, *, expected_sha256: str | None = None) -> dict[str, Any]:
    path, supplied = _path_value(value, label)
    expected = expected_sha256 if expected_sha256 is not None else supplied
    if expected is not None:
        _sha(expected, f"{label}.sha256")
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise PreGpuReceiptError(f"{label} raw SHA-256 drifted")
    result = {"path": str(path), "sha256": observed, "size_bytes": path.stat().st_size}
    return result


def _json_file_ref(
    value: Any,
    label: str,
    *,
    canonical: bool = True,
    require_semantic_hash: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ref = _file_ref(value, label)
    path = Path(ref["path"])
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreGpuReceiptError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(parsed, Mapping):
        raise PreGpuReceiptError(f"{label} must contain a JSON object")
    document = dict(parsed)
    canonical_json_bytes(document)
    if canonical and raw != canonical_json_bytes(document) + b"\n":
        raise PreGpuReceiptError(f"{label} must be canonical JSON with one trailing newline")
    semantic_keys = (
        "self_sha256",
        "content_sha256",
        "plan_content_sha256",
        "plan_sha256",
        "result_sha256",
        "receipt_sha256",
        "identity_sha256",
        "census_self_sha256",
    )
    declared: dict[str, str] = {}
    schema = str(document.get("schema_version", ""))
    for key in semantic_keys:
        if key in document:
            declared[key] = _sha(document[key], f"{label}.{key}")
            owns_hash = (
                key in {"self_sha256", "content_sha256"}
                or (key == "plan_content_sha256" and schema == "natural_boundary_owner_support_completion_plan.v1")
                or (key == "plan_sha256" and schema == "s_natural_boundary_k_n_h_execution_plan.v1")
            )
            if owns_hash:
                body = dict(document)
                body.pop(key, None)
                if declared[key] != sha256_json(body):
                    raise PreGpuReceiptError(f"{label}.{key} semantic hash mismatch")
    if require_semantic_hash and not declared:
        raise PreGpuReceiptError(f"{label} lacks a semantic self/content hash")
    ref["semantic_sha256"] = (
        declared.get("self_sha256")
        or declared.get("content_sha256")
        or declared.get("plan_content_sha256")
        or declared.get("plan_sha256")
        or declared.get("identity_sha256")
        or declared.get("result_sha256")
        or declared.get("census_self_sha256")
    )
    return ref, document


def _h0_image_plan_identity(ref: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the exact H0 image-plan rows used by geometry preflight."""

    path = Path(ref["path"])
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PreGpuReceiptError(f"cannot read S H0 image plan: {path}") from exc
    if not raw.endswith(b"\n"):
        raise PreGpuReceiptError("S H0 image plan must end with one JSONL newline")
    raw_lines = raw.splitlines()
    if not raw_lines or any(not line for line in raw_lines):
        raise PreGpuReceiptError("S H0 image plan must contain non-empty JSONL rows")
    normalized_fields = (
        "row_id",
        "row_index",
        "status",
        "error",
        "declared_width",
        "declared_height",
        "decoded_width",
        "decoded_height",
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "expected_image_grid_thw",
        "observed_image_grid_thw",
        "raw_patch_rows",
        "merged_visual_tokens",
    )
    normalized_rows: list[dict[str, Any]] = []
    row_ids: set[str] = set()
    for index, raw_line in enumerate(raw_lines):
        try:
            value = json.loads(raw_line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PreGpuReceiptError(f"S H0 image plan row {index} is invalid JSON") from exc
        if not isinstance(value, Mapping) or raw_line != canonical_json_bytes(value):
            raise PreGpuReceiptError(
                f"S H0 image plan row {index} is not canonical JSON"
            )
        row = dict(value)
        if any(field not in row for field in normalized_fields):
            raise PreGpuReceiptError(
                f"S H0 image plan row {index} lacks normalized geometry fields"
            )
        row_id = row["row_id"]
        if not isinstance(row_id, str) or not row_id or row_id in row_ids:
            raise PreGpuReceiptError("S H0 image plan row_id identity is missing/duplicate")
        row_ids.add(row_id)
        if row["row_index"] != index:
            raise PreGpuReceiptError("S H0 image plan row_index order drifted")
        scalar_fields = (
            "declared_width",
            "declared_height",
            "decoded_width",
            "decoded_height",
            "patch_size",
            "temporal_patch_size",
            "merge_size",
            "raw_patch_rows",
            "merged_visual_tokens",
        )
        if any(
            isinstance(row[field], bool)
            or not isinstance(row[field], int)
            or row[field] <= 0
            for field in scalar_fields
        ):
            raise PreGpuReceiptError(
                f"S H0 image plan row {index} has invalid positive integer geometry"
            )
        grids = (
            row["expected_image_grid_thw"],
            row["observed_image_grid_thw"],
        )
        if any(
            not isinstance(grid, list)
            or len(grid) != 3
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in grid
            )
            for grid in grids
        ):
            raise PreGpuReceiptError(f"S H0 image plan row {index} grid is malformed")
        observed_grid = grids[1]
        merge_size = row["merge_size"]
        grid_product = observed_grid[0] * observed_grid[1] * observed_grid[2]
        if (
            row["status"] != "ok"
            or row["error"] is not None
            or grids[0] != observed_grid
            or grid_product % (merge_size * merge_size) != 0
            or grid_product // (merge_size * merge_size)
            != row["merged_visual_tokens"]
        ):
            raise PreGpuReceiptError(
                f"S H0 image plan row {index} is not a successful exact grid plan"
            )
        normalized = {field: row[field] for field in normalized_fields}
        normalized_rows.append(normalized)
    row_digests = [
        {
            "row_index": row["row_index"],
            "row_id": row["row_id"],
            "sha256": sha256_json(row),
        }
        for row in normalized_rows
    ]
    return {
        "path": str(path),
        "raw_sha256": ref["sha256"],
        "row_count": len(normalized_rows),
        "normalized_rows_sha256": sha256_json(normalized_rows),
        "row_digests": row_digests,
        "row_digests_sha256": sha256_json(row_digests),
    }


def _write_once(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    target = _absolute(path, "receipt output")
    payload = canonical_json_bytes(document) + b"\n"
    parent = target.parent
    if parent.exists() and parent.is_symlink():
        raise PreGpuReceiptError(f"receipt output parent is a symlink: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(f"immutable receipt collision: {target}")
        return {"status": "sealed", "path": str(target), "sha256": sha256_bytes(payload), "byte_identical": True}
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(f"immutable receipt collision: {target}")
        return {"status": "sealed", "path": str(target), "sha256": sha256_bytes(payload), "byte_identical": True}
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {"status": "sealed", "path": str(target), "sha256": sha256_bytes(payload), "byte_identical": False}


def _self_checked_json(value: Any, label: str, *, canonical: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _json_file_ref(value, label, canonical=canonical, require_semantic_hash=True)
    if "self_sha256" in document and document["self_sha256"] != document_self_sha256(document):
        raise PreGpuReceiptError(f"{label}.self_sha256 semantic hash mismatch")
    if "self_sha256" not in document:
        # A few frozen producer schemas use content/plan/result hashes instead
        # of self_sha256.  Only accept those named hashes when they were
        # independently checked above; never synthesize a fallback.
        if not any(key in document for key in ("content_sha256", "plan_content_sha256", "plan_sha256", "result_sha256", "identity_sha256", "census_self_sha256")):
            raise PreGpuReceiptError(f"{label} lacks a valid self hash")
    return ref, document


def _artifact_ref(value: Any, label: str, *, canonical: bool = True) -> dict[str, Any]:
    ref, document = _self_checked_json(value, label, canonical=canonical)
    ref["schema_version"] = document.get("schema_version")
    ref["status"] = document.get("status")
    ref["unit_id"] = document.get("unit_id")
    ref["checkpoint"] = document.get("checkpoint")
    ref["step"] = document.get("step")
    if "self_sha256" in document:
        ref["self_sha256"] = document["self_sha256"]
    return ref


def _validate_primary(document: Mapping[str, Any], label: str) -> None:
    primary = document.get("primary")
    if not isinstance(primary, Mapping) or dict(primary) != {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE}:
        raise PreGpuReceiptError(f"{label} is not S step-2444 {SUBSTRATE}")


def _validate_authority(value: Any) -> dict[str, Any]:
    ref = _file_ref(value, "unit authority")
    if isinstance(value, Mapping):
        for field in ("unit_id", "status", "scope"):
            _text(value.get(field), f"unit authority.{field}")
        if value.get("unit_id") != UNIT_ID:
            raise PreGpuReceiptError("unit authority is not this unit")
        if value.get("status") not in {"active", "active_user_authorized_exception", "sealed"}:
            raise PreGpuReceiptError("unit authority status is not active/sealed")
        ref.update({"unit_id": value["unit_id"], "status": value["status"], "scope": value["scope"]})
    else:
        raise PreGpuReceiptError("unit authority requires explicit unit_id/status/scope metadata")
    return ref


def _validate_gate_artifacts(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(GATE_ARTIFACTS):
        raise PreGpuReceiptError(f"gate_v3_artifacts must contain exactly {GATE_ARTIFACTS}")
    refs: dict[str, Any] = {}
    docs: dict[str, Any] = {}
    for role in GATE_ARTIFACTS:
        if role == "launch_log":
            refs[role] = _file_ref(value[role], "gate-v3 launch log")
            continue
        ref, document = _json_file_ref(value[role], f"gate-v3 {role}", canonical=False, require_semantic_hash=True)
        if document.get("unit_id") not in {None, UNIT_ID}:
            raise PreGpuReceiptError(f"gate-v3 {role} belongs to another unit")
        if document.get("checkpoint") not in {None, CHECKPOINT}:
            raise PreGpuReceiptError(f"gate-v3 {role} is not checkpoint S")
        if role == "result":
            if (
                document.get("schema_version") != "s_primary_natural_boundary_gate.v1"
                or document.get("gpu_launch_authorized") is not False
                or document.get("no_training") is not True
                or document.get("arm_order") != list(ARM_ORDER)
            ):
                raise PreGpuReceiptError("gate-v3 result is not the frozen no-training S gate evidence")
            body = dict(document)
            declared_result = body.pop("result_sha256", None)
            if declared_result != sha256_json(body):
                raise PreGpuReceiptError("gate-v3 result_sha256 mismatch")
        elif role == "runtime_identity":
            if document.get("gpu_launch_authorized") is not False or document.get("no_training") is not True:
                raise PreGpuReceiptError("gate-v3 runtime identity crosses the frozen no-training boundary")
            body = dict(document)
            declared_identity = body.pop("identity_sha256", None)
            if declared_identity != sha256_json(body):
                raise PreGpuReceiptError("gate-v3 runtime identity_sha256 mismatch")
        elif role == "terminal_summary" and document.get("status") != "completed":
            raise PreGpuReceiptError("gate-v3 terminal summary is not completed")
        refs[role] = ref
        docs[role] = document
    result = docs.get("result", {})
    runtime = docs.get("runtime_identity", {})
    terminal = docs.get("terminal_summary", {})
    if result.get("event_id") and terminal.get("event_id") and result.get("event_id") != terminal.get("event_id"):
        raise PreGpuReceiptError("gate-v3 result/terminal event identity differs")
    if result.get("result_sha256") and terminal.get("result_sha256") and result.get("result_sha256") != terminal.get("result_sha256"):
        raise PreGpuReceiptError("gate-v3 terminal result hash differs")
    if runtime.get("checkpoint") not in {None, CHECKPOINT}:
        raise PreGpuReceiptError("gate-v3 runtime identity is not checkpoint S")
    return refs, docs


def _validate_census_v3(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _self_checked_json(value, "census-v3", canonical=True)
    if document.get("schema_version") != "natural_boundary_owner_admission_census.v3" or document.get("census_revision") != "census-v3" or document.get("status") != "sealed" or document.get("unit_id") != UNIT_ID:
        raise PreGpuReceiptError("census-v3 schema/status/unit identity drifted")
    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != 784:
        raise PreGpuReceiptError("census-v3 must contain exactly 784 rows")
    return ref, document


def _validate_manifest(value: Any, census_ref: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _self_checked_json(value, "admitted cohort manifest", canonical=True)
    if document.get("schema_version") != "s_natural_boundary_admitted_event_manifest.v3" or document.get("status") != "sealed" or document.get("unit_id") != UNIT_ID:
        raise PreGpuReceiptError("manifest schema/status/unit identity drifted")
    _validate_primary(document, "manifest")
    if document.get("arm_order") != list(ARM_ORDER):
        raise PreGpuReceiptError("manifest arm order differs from frozen S K/N/H order")
    source_census = document.get("source_census")
    if not isinstance(source_census, Mapping) or source_census.get("revision") != "census-v3" or source_census.get("sha256") != census_ref["sha256"]:
        raise PreGpuReceiptError("manifest source_census is not the exact census-v3 identity")
    events = document.get("events")
    if not isinstance(events, list) or not events:
        raise PreGpuReceiptError("manifest has no admitted events")
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or event.get("event_index") != index or event.get("checkpoint") != CHECKPOINT or event.get("step") != STEP or event.get("event_sha256") != sha256_json({k: v for k, v in event.items() if k != "event_sha256"}):
            raise PreGpuReceiptError(f"manifest event {index} is not an exact S event identity")
        owner_refs = event.get("owner_refs")
        eligibility = event.get("eligibility")
        predicates = (
            eligibility.get("predicates")
            if isinstance(eligibility, Mapping)
            else None
        )
        if (
            event.get("admission") != "admitted"
            or not isinstance(owner_refs, Mapping)
            or any(
                isinstance(owner_refs.get(key), bool)
                or not isinstance(owner_refs.get(key), int)
                or owner_refs[key] < 0
                for key in (
                    "source_panel_object_index",
                    "derived_panel_object_index",
                )
            )
            or not isinstance(owner_refs.get("covered_owner_ids"), list)
            or not owner_refs["covered_owner_ids"]
            or any(
                not isinstance(owner_id, str) or not owner_id
                for owner_id in owner_refs["covered_owner_ids"]
            )
            or owner_refs.get("covered_A_owner_id")
            != owner_refs["covered_owner_ids"][-1]
            or not isinstance(predicates, Mapping)
            or set(predicates) != FROZEN_ELIGIBILITY_PREDICATES
            or any(value is not True for value in predicates.values())
        ):
            raise PreGpuReceiptError(
                f"manifest event {index} admission identity/predicates drifted"
            )
    image_count = len({event["image_id"] for event in events})
    if document.get("event_count") != len(events) or document.get("image_count") != image_count:
        raise PreGpuReceiptError(
            "manifest event_count/image_count must equal exact admitted event identities"
        )
    admission_gate = document.get("admission_gate")
    if (
        not isinstance(admission_gate, Mapping)
        or admission_gate.get("minimum_event_count") != 3
        or admission_gate.get("minimum_image_count") != 2
    ):
        raise PreGpuReceiptError("manifest checkpoint replication floor identity drifted")
    try:
        from scripts.research.run_s_natural_boundary_k_n_h_cohort import validate_manifest as validate_cohort_manifest

        producer_info = validate_cohort_manifest(ref["path"])
    except Exception as exc:
        raise PreGpuReceiptError(f"manifest fails frozen cohort validator: {exc}") from exc
    if producer_info.get("manifest_sha256") != ref["sha256"] or producer_info.get("manifest_self_sha256") != document.get("self_sha256"):
        raise PreGpuReceiptError("frozen cohort validator returned a different manifest identity")
    return ref, document


def _validate_legacy_context_binding(
    manifest: Mapping[str, Any],
    cohort_identity: Mapping[str, Any],
    cohort_manifest_value: Any,
) -> dict[str, Any]:
    """Bind the exact materializer-selected legacy context pair."""

    declared = manifest.get("legacy_context_cohort")
    required = {
        "status",
        "path",
        "sha256",
        "manifest_path",
        "manifest_sha256",
        "event_count",
    }
    if not isinstance(declared, Mapping) or set(declared) != required:
        raise PreGpuReceiptError(
            "manifest legacy_context_cohort identity block is missing or incomplete"
        )
    if declared.get("status") != "bound":
        raise PreGpuReceiptError("manifest legacy_context_cohort is not bound")
    cohort_ref, cohort_document = _json_file_ref(
        cohort_identity,
        "legacy context cohort",
        canonical=True,
        require_semantic_hash=False,
    )
    companion_ref, companion_document = _json_file_ref(
        cohort_manifest_value,
        "legacy context cohort companion manifest",
        canonical=True,
        require_semantic_hash=False,
    )
    cohort_path = Path(cohort_ref["path"])
    if cohort_path.suffix != ".json":
        raise PreGpuReceiptError("legacy context cohort path must end in .json")
    expected_companion = cohort_path.with_name(
        cohort_path.name.replace(".json", ".manifest.json")
    )
    event_count = declared.get("event_count")
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 1:
        raise PreGpuReceiptError("legacy context cohort event_count must be positive")
    if (
        declared.get("path") != cohort_ref["path"]
        or declared.get("sha256") != cohort_ref["sha256"]
        or declared.get("manifest_path") != companion_ref["path"]
        or declared.get("manifest_sha256") != companion_ref["sha256"]
        or companion_ref["path"] != str(expected_companion)
    ):
        raise PreGpuReceiptError(
            "legacy context cohort path/content identity differs from materializer manifest"
        )
    context_events = cohort_document.get("events")
    if not isinstance(context_events, list) or len(context_events) != event_count:
        raise PreGpuReceiptError("legacy context cohort event_count differs from its content")
    context_owner_ids = [
        event.get("gt_owner_id") if isinstance(event, Mapping) else None
        for event in context_events
    ]
    if any(not isinstance(owner_id, str) or not owner_id for owner_id in context_owner_ids):
        raise PreGpuReceiptError("legacy context cohort contains a malformed owner identity")
    if len(set(context_owner_ids)) != len(context_owner_ids):
        raise PreGpuReceiptError("legacy context cohort contains duplicate owner identities")
    context_plan_fields = {
        "cell_count",
        "grid_cols",
        "grid_rows",
        "grid_thw",
        "image_height",
        "image_width",
        "merge_size",
        "merged_visual_tokens",
        "observed_image_grid_thw",
        "premerge_grid_cols",
        "premerge_grid_rows",
    }
    context_plan_bindings: list[dict[str, Any]] = []
    for index, event in enumerate(context_events):
        assert isinstance(event, Mapping)
        panel_identity = event.get("panel_identity")
        coco_ann_id = (
            panel_identity.get("coco_ann_id")
            if isinstance(panel_identity, Mapping)
            else None
        )
        # Human-refined panel additions use stable negative annotation IDs.
        # Their sign distinguishes them from source COCO annotations; identity
        # requires an exact integer, not a non-negative source-only ID.
        if isinstance(coco_ann_id, bool) or not isinstance(coco_ann_id, int):
            raise PreGpuReceiptError(
                "legacy context cohort event "
                f"{index} lacks an integer panel_identity.coco_ann_id"
            )
        geometry_by_checkpoint = event.get("geometry_by_checkpoint")
        geometry = (
            geometry_by_checkpoint.get(CHECKPOINT)
            if isinstance(geometry_by_checkpoint, Mapping)
            else None
        )
        plan_identity = (
            geometry.get("image_plan_identity")
            if isinstance(geometry, Mapping)
            else None
        )
        if not isinstance(plan_identity, Mapping) or set(plan_identity) != context_plan_fields:
            raise PreGpuReceiptError(
                f"legacy context cohort event {index} image_plan_identity is incomplete"
            )
        canonical_json_bytes(plan_identity)
        cell_count = plan_identity.get("cell_count")
        merged_visual_tokens = plan_identity.get("merged_visual_tokens")
        if (
            isinstance(cell_count, bool)
            or not isinstance(cell_count, int)
            or cell_count <= 0
            or merged_visual_tokens != cell_count
        ):
            raise PreGpuReceiptError(
                f"legacy context cohort event {index} image-plan cell count drifted"
            )
        image_id = event.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id < 0:
            raise PreGpuReceiptError(
                f"legacy context cohort event {index} image identity is malformed"
            )
        context_plan_bindings.append(
            {
                "gt_owner_id": event["gt_owner_id"],
                "image_id": image_id,
                "image_plan_identity": dict(plan_identity),
            }
        )
    admitted_events = manifest.get("events")
    dynamic_only = manifest.get("dynamic_only")
    if not isinstance(admitted_events, list) or not isinstance(dynamic_only, Mapping):
        raise PreGpuReceiptError(
            "manifest lacks admitted/dynamic-only context ownership needed for legacy binding"
        )
    admitted_owner_ids = [
        event.get("owner_refs", {}).get("gt_owner_id")
        if isinstance(event, Mapping) and isinstance(event.get("owner_refs"), Mapping)
        else None
        for event in admitted_events
    ]
    dynamic_owner_ids = dynamic_only.get("owner_ids")
    if not isinstance(dynamic_owner_ids, list) or any(
        not isinstance(owner_id, str) or not owner_id for owner_id in dynamic_owner_ids
    ):
        raise PreGpuReceiptError("manifest dynamic_only owner identity list is malformed")
    intended_owner_ids = admitted_owner_ids + list(dynamic_owner_ids)
    if (
        any(not isinstance(owner_id, str) or not owner_id for owner_id in admitted_owner_ids)
        or len(set(intended_owner_ids)) != len(intended_owner_ids)
        or set(context_owner_ids) != set(intended_owner_ids)
    ):
        raise PreGpuReceiptError(
            "legacy context cohort does not equal admitted plus dynamic-only owner set"
        )
    if (
        companion_document.get("schema_version")
        != "static_dynamic_owner_interface_cohort.v1.manifest"
        or companion_document.get("unit_id") != LEGACY_CONTEXT_UNIT_ID
        or companion_document.get("status") != "sealed"
        or companion_document.get("cohort_path") != cohort_ref["path"]
        or companion_document.get("cohort_sha256") != declared.get("sha256")
        or companion_document.get("event_count") != event_count
        or companion_document.get("owner_ids") != context_owner_ids
        or companion_document.get("owner_ids_sha256") != sha256_json(context_owner_ids)
    ):
        raise PreGpuReceiptError("legacy context companion manifest linkage drifted")
    cohort_ref["semantic_sha256"] = sha256_json(cohort_document)
    companion_ref["semantic_sha256"] = sha256_json(companion_document)
    return {
        "status": "bound",
        "event_count": event_count,
        "owner_ids_sha256": sha256_json(context_owner_ids),
        "image_plan_bindings_sha256": sha256_json(context_plan_bindings),
        "cohort": cohort_ref,
        "companion_manifest": companion_ref,
    }


def _validate_execution_plan(value: Any, manifest_ref: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _self_checked_json(value, "execution plan", canonical=True)
    if document.get("schema_version") != "s_natural_boundary_k_n_h_execution_plan.v1" or document.get("status") != "planned" or document.get("unit_id") != UNIT_ID:
        raise PreGpuReceiptError("execution plan schema/status/unit identity drifted")
    _validate_primary(document, "execution plan")
    declared = document.get("manifest_sha256")
    if declared != manifest_ref["sha256"] or document.get("manifest_self_sha256") != manifest.get("self_sha256"):
        raise PreGpuReceiptError("execution plan is bound to a different manifest")
    if document.get("shard_count") != SHARD_COUNT or not isinstance(document.get("shards"), list) or len(document["shards"]) != SHARD_COUNT:
        raise PreGpuReceiptError("execution plan must contain exactly eight shards")
    if document.get("arm_order") != list(ARM_ORDER):
        raise PreGpuReceiptError("execution plan arm order differs from frozen cohort")
    if document.get("claim_scope") != _claim_scope(manifest):
        raise PreGpuReceiptError("execution plan claim scope differs from exact manifest counts")
    for flag in ("no_outcome_adaptive_selection", "no_event_reorder", "no_sweep", "no_a3", "no_2x2", "no_p4"):
        if document.get(flag) is not True:
            raise PreGpuReceiptError(f"execution plan flag {flag} is not sealed")
    manifest_events = manifest.get("events")
    assert isinstance(manifest_events, list)
    expected_refs = [
        {
            "event_index": event["event_index"],
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "event_sha256": event["event_sha256"],
        }
        for event in manifest_events
    ]
    if document.get("events") != expected_refs:
        raise PreGpuReceiptError("execution plan top-level event order differs from manifest")
    assigned: list[dict[str, Any]] = []
    for index, shard in enumerate(document["shards"]):
        if not isinstance(shard, Mapping) or shard.get("shard_index") != index or shard.get("shard_id") != f"shard-{index:03d}":
            raise PreGpuReceiptError("execution plan shard identity/order drifted")
        refs = shard.get("events")
        if not isinstance(refs, list) or shard.get("event_indices") != [ref.get("event_index") for ref in refs if isinstance(ref, Mapping)]:
            raise PreGpuReceiptError(f"execution plan shard {index} event assignment is malformed")
        if any(not isinstance(ref, Mapping) for ref in refs):
            raise PreGpuReceiptError(f"execution plan shard {index} contains a malformed event ref")
        assigned.extend(dict(ref) for ref in refs)
    if sorted(assigned, key=lambda ref: ref["event_index"]) != expected_refs or len({ref["event_index"] for ref in assigned}) != len(assigned):
        raise PreGpuReceiptError("execution plan shards do not cover the exact manifest once")
    try:
        from scripts.research.plan_s_natural_boundary_k_n_h_execution import validate_plan as validate_frozen_plan
        from scripts.research.run_s_natural_boundary_k_n_h_cohort import validate_manifest as validate_cohort_manifest

        producer_manifest = validate_cohort_manifest(manifest_ref["path"])
        producer_plan = validate_frozen_plan(ref["path"], manifest_info=producer_manifest)
    except Exception as exc:
        raise PreGpuReceiptError(f"execution plan fails frozen planner validator: {exc}") from exc
    if producer_plan.get("plan_sha256") != document.get("plan_sha256"):
        raise PreGpuReceiptError("frozen planner validator returned a different plan identity")
    return ref, document


def _validate_identity(value: Any, label: str, *, require_directory: bool = False, require_file: bool = True) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreGpuReceiptError(f"{label} identity must be an object")
    result = dict(value)
    _text(result.get("id"), f"{label}.id")
    if result.get("checkpoint") not in {None, CHECKPOINT} or result.get("step") not in {None, STEP}:
        raise PreGpuReceiptError(f"{label} is not S step-2444")
    path = result.get("path") or result.get("root")
    if path is None:
        raise PreGpuReceiptError(f"{label}.path/root is missing")
    if require_directory:
        root = _directory(path, label)
        result["path"] = str(root)
    elif require_file:
        result.update(_file_ref(result, label))
    if result.get("sha256") is not None:
        _sha(result["sha256"], f"{label}.sha256")
    return result


def _validate_h0_identity(
    value: Any,
    *,
    require_bound_inventory: bool = False,
) -> dict[str, Any]:
    """Bind the exact loader-selected H0 directory and model payload files."""

    result = _validate_identity(value, "S H0", require_directory=True, require_file=False)
    if result.get("checkpoint") != CHECKPOINT or result.get("step") != STEP:
        raise PreGpuReceiptError("H0 identity is not S step-2444")
    if not isinstance(value, Mapping):
        raise PreGpuReceiptError("S H0 identity must be an object")
    h0_dir_value = value.get("h0_dir")
    if h0_dir_value is None:
        raise PreGpuReceiptError("S H0 identity requires the exact loader-selected h0_dir")
    h0_dir = _directory(h0_dir_value, "S H0 checkpoint directory")
    root = Path(result["path"])
    if h0_dir.parent != root:
        raise PreGpuReceiptError("S H0 checkpoint directory must be one exact child of h0_root")
    raw_files = value.get("identity_files")
    if not isinstance(raw_files, Mapping) or set(raw_files) != set(H0_IDENTITY_FILE_ROLES):
        raise PreGpuReceiptError(
            f"S H0 identity_files must contain exactly {H0_IDENTITY_FILE_ROLES}"
        )
    files = {
        role: _file_ref(raw_files[role], f"S H0 identity file {role}")
        for role in H0_IDENTITY_FILE_ROLES
    }
    for role, relative in H0_LOADER_RELATIVE_PATHS.items():
        if Path(files[role]["path"]) != h0_dir / relative:
            raise PreGpuReceiptError(
                f"S H0 {role} must equal exact loader path {h0_dir / relative}"
            )
    result["h0_dir"] = str(h0_dir)
    result["identity_files"] = files
    result["identity_files_sha256"] = sha256_json(
        {role: files[role]["sha256"] for role in H0_IDENTITY_FILE_ROLES}
    )
    image_plan_identity = _h0_image_plan_identity(files["image_plan"])
    declared_image_plan_identity = value.get("image_plan_identity")
    if (
        require_bound_inventory
        and not isinstance(declared_image_plan_identity, Mapping)
    ):
        raise PreGpuReceiptError("S H0 identity lacks normalized image-plan identity")
    if (
        declared_image_plan_identity is not None
        and dict(declared_image_plan_identity) != image_plan_identity
    ):
        raise PreGpuReceiptError("S H0 normalized image-plan identity drifted")
    result["image_plan_identity"] = image_plan_identity
    base_model_value = value.get("base_model_dir")
    if base_model_value is None:
        raise PreGpuReceiptError("S H0 identity requires the exact loader-selected base_model_dir")
    base_model_dir = _directory(base_model_value, "S base model directory")
    raw_base_files = value.get("base_model_files")
    if not isinstance(raw_base_files, Mapping) or set(raw_base_files) != set(BASE_MODEL_FILE_ROLES):
        raise PreGpuReceiptError(
            f"S H0 base_model_files must contain exactly {BASE_MODEL_FILE_ROLES}"
        )
    base_files = {
        role: _file_ref(raw_base_files[role], f"S base model file {role}")
        for role in BASE_MODEL_FILE_ROLES
    }
    for role, ref in base_files.items():
        expected_path = base_model_dir / BASE_MODEL_ROLE_BASENAMES[role]
        if Path(ref["path"]) != expected_path:
            raise PreGpuReceiptError(
                f"S base model role {role} must equal exact loader path {expected_path}"
            )
    base_inventory = _regular_file_inventory(
        base_model_dir,
        "S base model directory",
        known_file_refs={ref["path"]: ref for ref in base_files.values()},
    )
    declared_inventory = value.get("base_model_inventory")
    if require_bound_inventory and not isinstance(declared_inventory, Mapping):
        raise PreGpuReceiptError("S H0 identity lacks the complete base-model inventory")
    if declared_inventory is not None and dict(declared_inventory) != base_inventory:
        raise PreGpuReceiptError("S base model inventory added, missing, or drifted")
    inventory_by_path = {
        str(base_model_dir / item["relative_path"]): item
        for item in base_inventory["files"]
    }
    for role, ref in base_files.items():
        inventory_ref = inventory_by_path.get(ref["path"])
        if inventory_ref is None or inventory_ref["sha256"] != ref["sha256"]:
            raise PreGpuReceiptError(
                f"S base model role {role} is absent from the complete payload inventory"
            )

    def read_json_ref(ref: Mapping[str, Any], label: str) -> dict[str, Any]:
        try:
            parsed = json.loads(Path(ref["path"]).read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PreGpuReceiptError(f"cannot read {label} JSON") from exc
        if not isinstance(parsed, Mapping):
            raise PreGpuReceiptError(f"{label} must be a JSON object")
        canonical_json_bytes(parsed)
        return dict(parsed)

    resolved = read_json_ref(files["resolved_config"], "S H0 resolved config")
    run_manifest = read_json_ref(files["run_manifest"], "S H0 run manifest")
    adapter_config = read_json_ref(files["adapter_config"], "S adapter config")
    embedding_metadata = read_json_ref(files["embedding_metadata"], "S embedding metadata")
    resolved_config = resolved.get("config")
    resolved_adapter = (
        resolved_config.get("adapter")
        if isinstance(resolved_config, Mapping)
        else None
    )
    resolved_embedding = (
        resolved_config.get("embedding_delta")
        if isinstance(resolved_config, Mapping)
        else None
    )
    adapter_root_value = (
        resolved_adapter.get("path")
        if isinstance(resolved_adapter, Mapping)
        else None
    )
    embedding_root_value = (
        resolved_embedding.get("path")
        if isinstance(resolved_embedding, Mapping)
        else None
    )
    if adapter_root_value is None or embedding_root_value is None:
        raise PreGpuReceiptError(
            "S H0 resolved config lacks exact adapter/embedding-delta loader paths"
        )
    adapter_root = _directory(adapter_root_value, "S adapter directory")
    embedding_root = _directory(
        embedding_root_value,
        "S embedding-delta directory",
    )
    expected_payload_paths = {
        "adapter_config": adapter_root / "adapter_config.json",
        "adapter_tensor": adapter_root / "adapter_model.safetensors",
        "embedding_metadata": embedding_root / "special_token_embeddings.json",
        "embedding_tensor": embedding_root / "special_token_embeddings.safetensors",
    }
    for role, expected_path in expected_payload_paths.items():
        if Path(files[role]["path"]) != expected_path:
            raise PreGpuReceiptError(
                f"S H0 {role} must equal exact selected loader path {expected_path}"
            )
    resolved_base = (
        resolved.get("config", {}).get("model", {}).get("base_model")
        if isinstance(resolved.get("config"), Mapping)
        and isinstance(resolved["config"].get("model"), Mapping)
        else None
    )
    manifest_base = (
        run_manifest.get("model_identity", {}).get("base", {}).get("path")
        if isinstance(run_manifest.get("model_identity"), Mapping)
        and isinstance(run_manifest["model_identity"].get("base"), Mapping)
        else None
    )
    manifest_adapter = run_manifest.get("adapter_identity")
    adapter_payload = (
        manifest_adapter.get("adapter_payload_evidence")
        if isinstance(manifest_adapter, Mapping)
        else None
    )
    backend_session = run_manifest.get("backend_session")
    session_model = (
        backend_session.get("model_identity")
        if isinstance(backend_session, Mapping)
        else None
    )
    manifest_embedding = (
        session_model.get("embedding_delta", {}).get("identity")
        if isinstance(session_model, Mapping)
        and isinstance(session_model.get("embedding_delta"), Mapping)
        else None
    )
    expected_base = str(base_model_dir)
    if (
        resolved_base != expected_base
        or manifest_base != expected_base
        or adapter_config.get("base_model_name_or_path") != expected_base
        or embedding_metadata.get("base_model_path") != expected_base
        or embedding_metadata.get("base_config_sha256") != base_files["config"]["sha256"]
        or not isinstance(manifest_adapter, Mapping)
        or manifest_adapter.get("adapter_path") != str(adapter_root)
        or not isinstance(adapter_payload, Mapping)
        or adapter_payload.get("config_path") != files["adapter_config"]["path"]
        or adapter_payload.get("tensor_path") != files["adapter_tensor"]["path"]
        or not isinstance(manifest_embedding, Mapping)
        or manifest_embedding.get("delta_path") != str(embedding_root)
        or manifest_embedding.get("metadata_path") != files["embedding_metadata"]["path"]
    ):
        raise PreGpuReceiptError("S H0/base-model lineage differs across resolved loader identities")
    result["base_model_dir"] = expected_base
    result["base_model_files"] = base_files
    result["base_model_files_sha256"] = sha256_json(
        {role: base_files[role]["sha256"] for role in BASE_MODEL_FILE_ROLES}
    )
    result["base_model_inventory"] = base_inventory
    result["base_model_inventory_sha256"] = base_inventory["inventory_sha256"]
    return result


def _validate_support_inputs(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreGpuReceiptError("support_inputs must be an object")
    required = ("plan", "census_v2", "shard_receipts", "merge_ledger", "merge_receipt")
    if set(value) != set(required):
        raise PreGpuReceiptError(f"support_inputs must contain exactly {required}")
    refs: dict[str, Any] = {}
    for key in ("plan", "census_v2", "merge_ledger", "merge_receipt"):
        refs[key] = _artifact_ref(value[key], f"support {key}", canonical=True)
    raw_shards = value["shard_receipts"]
    if not isinstance(raw_shards, Sequence) or isinstance(raw_shards, (str, bytes, bytearray)) or len(raw_shards) != SHARD_COUNT:
        raise PreGpuReceiptError("support shard receipts must contain exactly eight files")
    shard_refs: list[dict[str, Any]] = []
    for index, item in enumerate(raw_shards):
        ref = _artifact_ref(item, f"support shard receipt {index}", canonical=True)
        shard_refs.append(ref)
    refs["shard_receipts"] = shard_refs
    return refs


def _validate_runtime(runtime: Any, forced_math: Any, device_policy: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(runtime, Mapping) or not isinstance(forced_math, Mapping) or not isinstance(device_policy, Mapping):
        raise PreGpuReceiptError("runtime, forced_math, and device_policy are required objects")
    runtime_doc = dict(runtime)
    for key in ("python_version", "torch_version", "transformers_version", "backend", "dtype", "attn_implementation"):
        _text(runtime_doc.get(key), f"runtime.{key}")
    if (
        runtime_doc["backend"].lower() not in {"hf", "huggingface"}
        or runtime_doc["dtype"].lower() not in {"fp32", "float32"}
        or runtime_doc["attn_implementation"].lower() != "sdpa"
        or runtime_doc.get("no_training") is not True
    ):
        raise PreGpuReceiptError("runtime must be no-training HF fp32/SDPA")
    try:
        installed = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch_version": metadata.version("torch"),
            "transformers_version": metadata.version("transformers"),
        }
    except metadata.PackageNotFoundError as exc:
        raise PreGpuReceiptError(
            f"installed runtime package is unavailable: {exc.name}"
        ) from exc
    for key, observed in installed.items():
        if runtime_doc[key] != observed:
            raise PreGpuReceiptError(
                f"runtime.{key} differs from installed CPU-observed version {observed}"
            )
    math_doc = dict(forced_math)
    if math_doc.get("enabled") is not True or str(math_doc.get("backend", "")).upper() != "MATH":
        raise PreGpuReceiptError("forced-MATH policy is not enabled")
    policy = dict(device_policy)
    if policy != DEVICE_POLICY:
        raise PreGpuReceiptError(
            "device policy must bind shard-000..007 exactly to physical GPUs 0..7 "
            "with one logical cuda:0 per shard"
        )
    return runtime_doc, math_doc, policy


def _validate_runtime_evidence(
    value: Any,
    *,
    runtime: Any,
    forced_math: Any,
    device_policy: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _json_file_ref(
        value,
        "installed runtime evidence",
        canonical=True,
        require_semantic_hash=False,
    )
    if set(document) != {
        "schema_version", "status", "producer", "cpu_only", "gpu_used",
        "model_loaded", "runtime", "forced_math", "device_policy",
    }:
        raise PreGpuReceiptError("installed runtime evidence has an unexpected schema")
    if document.get("schema_version") != RUNTIME_EVIDENCE_SCHEMA_VERSION:
        raise PreGpuReceiptError("installed runtime evidence schema drifted")
    producer = _file_ref(document.get("producer"), "installed runtime evidence producer")
    materializer_path = CODE_ROLE_PATHS[EVIDENCE_MATERIALIZER_ROLE].resolve(strict=True)
    if producer["path"] != str(materializer_path):
        raise PreGpuReceiptError("installed runtime evidence producer is not the exact materializer")
    if producer["sha256"] != sha256_file(materializer_path):
        raise PreGpuReceiptError("installed runtime evidence producer hash drifted")
    if (
        document.get("status") != "passed"
        or document.get("cpu_only") is not True
        or document.get("gpu_used") is not False
        or document.get("model_loaded") is not False
    ):
        raise PreGpuReceiptError("installed runtime evidence is not a passing CPU-only receipt")
    if (
        document.get("runtime") != runtime
        or document.get("forced_math") != forced_math
        or document.get("device_policy") != device_policy
    ):
        raise PreGpuReceiptError("installed runtime evidence differs from runtime/MATH/device policy")
    _validate_runtime(runtime, forced_math, device_policy)
    return ref, document


def _validate_code(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(CODE_ROLES):
        raise PreGpuReceiptError(f"source_files must contain exactly frozen code roles {CODE_ROLES}")
    refs: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for role in CODE_ROLES:
        ref = _file_ref(value[role], f"source file {role}")
        if ref["path"] != str(CODE_ROLE_PATHS[role].resolve(strict=True)):
            raise PreGpuReceiptError(
                f"source role {role} must bind exact repository owner {CODE_ROLE_PATHS[role]}"
            )
        refs[role] = {"path": ref["path"], "sha256": ref["sha256"], "size_bytes": ref["size_bytes"], "role": role}
        hashes[role] = ref["sha256"]
    return refs, hashes


def _validate_tests(focused_tests: Any, test_receipt: Any) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(test_receipt, (str, Path, Mapping)):
        raise PreGpuReceiptError("focused test result must be an exact JSON file ref")
    if isinstance(test_receipt, Mapping) and "path" not in test_receipt and "file" not in test_receipt:
        raise PreGpuReceiptError("focused test result requires an exact file path and raw hash")
    receipt_ref, receipt = _json_file_ref(
        test_receipt,
        "focused test result",
        canonical=True,
        require_semantic_hash=False,
    )
    if set(receipt) != {
        "schema_version", "status", "producer", "command", "execution_argv", "cwd",
        "exit_code", "passed_test_count", "focused_tests", "stdout", "stderr", "output",
        "output_sha256",
    }:
        raise PreGpuReceiptError("focused test result has an unexpected schema")
    if receipt.get("schema_version") != FOCUSED_TEST_EVIDENCE_SCHEMA_VERSION:
        raise PreGpuReceiptError("focused test result schema drifted")
    producer = _file_ref(receipt.get("producer"), "focused test result producer")
    materializer_path = CODE_ROLE_PATHS[EVIDENCE_MATERIALIZER_ROLE].resolve(strict=True)
    if producer["path"] != str(materializer_path):
        raise PreGpuReceiptError("focused test result producer is not the exact materializer")
    if producer["sha256"] != sha256_file(materializer_path):
        raise PreGpuReceiptError("focused test result producer hash drifted")
    if receipt.get("status") != "passed" or receipt.get("exit_code") != 0:
        raise PreGpuReceiptError("focused test result is not a passing zero-exit receipt")
    if receipt.get("command") != FOCUSED_TEST_COMMAND:
        raise PreGpuReceiptError("focused test result command differs from exact suite command")
    observed_argv = receipt.get("execution_argv")
    expected_argv_tail = ["-m", "pytest", "-q", *(str(path) for path in REQUIRED_FOCUSED_TEST_PATHS)]
    if (
        not isinstance(observed_argv, list)
        or len(observed_argv) != len(expected_argv_tail) + 1
        or not isinstance(observed_argv[0], str)
        or observed_argv[1:] != expected_argv_tail
        or receipt.get("cwd") != str(REPO_ROOT)
    ):
        raise PreGpuReceiptError("focused test result execution environment differs from the exact suite")
    try:
        observed_executable = Path(observed_argv[0]).resolve(strict=True)
        expected_executable = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise PreGpuReceiptError("focused test result executable is not resolvable") from exc
    if not observed_executable.is_file() or observed_executable != expected_executable:
        raise PreGpuReceiptError("focused test result executable differs from the current interpreter")
    if not isinstance(receipt.get("stdout"), str) or not isinstance(receipt.get("stderr"), str):
        raise PreGpuReceiptError("focused test result must preserve stdout and stderr")
    if receipt.get("output") != receipt["stdout"] + receipt["stderr"]:
        raise PreGpuReceiptError("focused test result output does not bind stdout and stderr")
    if receipt.get("output_sha256") != sha256_bytes(receipt["output"].encode("utf-8")):
        raise PreGpuReceiptError("focused test result output raw hash drifted")
    entries = receipt.get("focused_tests")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)) or not entries:
        raise PreGpuReceiptError("focused test result lacks exact test identities")
    refs: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        refs.append(_file_ref(item, f"focused test {index}"))
    required_paths = {str(path.resolve(strict=True)) for path in REQUIRED_FOCUSED_TEST_PATHS}
    observed_paths = {ref["path"] for ref in refs}
    if observed_paths != required_paths:
        missing = sorted(required_paths - observed_paths)
        foreign = sorted(observed_paths - required_paths)
        raise PreGpuReceiptError(
            f"focused test result differs from exact conclusion-critical suite; missing={missing}, foreign={foreign}"
        )
    if len(refs) != len(REQUIRED_FOCUSED_TEST_PATHS):
        raise PreGpuReceiptError("focused test result contains duplicate test identities")
    passed_match = re.fullmatch(r"\d+", str(receipt.get("passed_test_count", "")))
    if passed_match is None or int(receipt["passed_test_count"]) < len(REQUIRED_FOCUSED_TEST_PATHS):
        raise PreGpuReceiptError("focused test result pass count is missing or implausibly small")
    if re.search(rf"(?m)(?:^|\s){receipt['passed_test_count']} passed(?:,|\s|$)", receipt["output"]) is None:
        raise PreGpuReceiptError("focused test result output does not attest the exact pass count")
    if focused_tests is not None:
        if not isinstance(focused_tests, Sequence) or isinstance(focused_tests, (str, bytes, bytearray)) or len(focused_tests) != len(refs):
            raise PreGpuReceiptError("focused_tests disagree with passing test result")
        supplied = [_file_ref(item, f"focused test supplied {index}") for index, item in enumerate(focused_tests)]
        if [(item["path"], item["sha256"]) for item in supplied] != [(item["path"], item["sha256"]) for item in refs]:
            raise PreGpuReceiptError("focused test identities disagree with passing test result")
    execution = {
        "status": "passed",
        "command": FOCUSED_TEST_COMMAND,
        "exit_code": 0,
        "test_count": len(refs),
        "tests": refs,
    }
    return execution, receipt, refs, receipt_ref


def _validate_output_binding(shard_roots: Any, final_merge_root: Any, *, phase: str) -> dict[str, Any]:
    if phase not in {"prelaunch", "runtime", "postrun"}:
        raise PreGpuReceiptError("output validation phase must be prelaunch, runtime, or postrun")
    if not isinstance(shard_roots, Sequence) or isinstance(shard_roots, (str, bytes, bytearray)) or len(shard_roots) != SHARD_COUNT:
        raise PreGpuReceiptError("output binding requires exactly eight shard roots")
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(shard_roots):
        if isinstance(item, Mapping):
            if item.get("shard_index") != index:
                raise PreGpuReceiptError("shard root indices must be ordered 0..7")
            raw = item.get("path")
        else:
            raw = item
        if raw is None:
            raise PreGpuReceiptError(f"shard root {index} path is missing")
        if phase == "prelaunch":
            path = _directory(raw, f"shard root {index}", exists=False)
            status = "reserved_absent_pre_gpu"
        elif phase == "postrun":
            path = _directory(raw, f"shard root {index}", exists=True)
            status = "regular_postrun_root"
        else:
            path = _absolute(raw, f"shard root {index}")
            if path.exists() and (path.is_symlink() or not path.is_dir()):
                raise PreGpuReceiptError(f"runtime shard root {index} is not a regular directory")
            status = "authorized_runtime_root"
        if str(path) in seen:
            raise PreGpuReceiptError("shard roots must be distinct")
        seen.add(str(path))
        refs.append({"shard_index": index, "path": str(path), "status": status})
    if phase == "prelaunch":
        final = _directory(final_merge_root, "final merge root", exists=False)
        status = "reserved_absent_pre_gpu"
    elif phase == "postrun":
        final = _directory(final_merge_root, "final merge root", exists=True)
        status = "regular_postrun_root"
    else:
        final = _absolute(final_merge_root, "final merge root")
        if final.exists() and (final.is_symlink() or not final.is_dir()):
            raise PreGpuReceiptError("runtime final merge root is not a regular directory")
        status = "authorized_runtime_root"
    if str(final) in seen:
        raise PreGpuReceiptError("final merge root must differ from shard roots")
    return {"shard_roots": refs, "final_merge_root": {"path": str(final), "status": status}, "phase": phase}


def _event_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    events = manifest.get("events")
    assert isinstance(events, list)
    hashes = [event["event_sha256"] for event in events]
    images = sorted({int(event["image_id"]) for event in events})
    bindings = [
        {
            "event_index": event["event_index"],
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "event_sha256": event["event_sha256"],
        }
        for event in events
    ]
    return {
        "checkpoint": CHECKPOINT,
        "step": STEP,
        "event_count": len(events),
        "image_count": len(images),
        # Kept for the first-event gate compatibility surface.  Cohort
        # authorization is owned by the ordered plural fields below.
        "event_sha256": hashes[0],
        "event_sha256s": hashes,
        "authorized_event_sha256s": hashes,
        "event_ids": [event["event_id"] for event in events],
        "bindings": bindings,
        "event_bindings_sha256": sha256_json(bindings),
        "authorized_event_order_sha256": sha256_json(hashes),
    }


def _claim_scope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    events = manifest.get("events")
    assert isinstance(events, list)
    event_count = len(events)
    image_count = len({int(event["image_id"]) for event in events})
    checkpoint_qualified = event_count >= 3 and image_count >= 2
    return {
        "execution_scope": (
            "checkpoint_replication_candidate"
            if checkpoint_qualified
            else "case_study"
        ),
        "event_count": event_count,
        "image_count": image_count,
        "minimum_checkpoint_event_count": 3,
        "minimum_checkpoint_image_count": 2,
        "checkpoint_claim_qualified": checkpoint_qualified,
        "static_direction_claim_qualified": checkpoint_qualified,
        "training_claim_qualified": False,
        "subfloor_execution_authorized": not checkpoint_qualified,
    }


def _shard_authorizations(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    events = manifest.get("events")
    shards = plan.get("shards")
    roots = output_binding.get("shard_roots")
    assert isinstance(events, list) and isinstance(shards, list) and isinstance(roots, list)
    by_index = {event["event_index"]: event for event in events}
    result: list[dict[str, Any]] = []
    for index, (shard, root) in enumerate(zip(shards, roots, strict=True)):
        refs = shard["events"]
        bindings = [
            {
                "event_index": ref["event_index"],
                "event_id": ref["event_id"],
                "event_sha256": by_index[ref["event_index"]]["event_sha256"],
            }
            for ref in refs
        ]
        authorization = {
            "shard_id": f"shard-{index:03d}",
            "shard_index": index,
            "output_root": root["path"],
            "physical_device": SHARD_PHYSICAL_DEVICES[f"shard-{index:03d}"],
            "logical_device": "cuda:0",
            "device_count": 1,
            "event_indices": [binding["event_index"] for binding in bindings],
            "event_sha256s": [binding["event_sha256"] for binding in bindings],
            "event_bindings": bindings,
        }
        authorization["authorization_sha256"] = sha256_json(authorization)
        result.append(authorization)
    return result


def build_receipt(
    *,
    output: str | Path,
    unit_authority: Any,
    gate_v3_artifacts: Any,
    support_inputs: Any,
    census_v3: Any,
    manifest: Any,
    execution_plan: Any,
    config: Any,
    checkpoint: Any,
    h0: Any,
    panel: Any,
    cohort: Any,
    cohort_manifest: Any,
    runtime: Any,
    runtime_evidence: Any,
    forced_math: Any,
    device_policy: Any,
    shard_roots: Any,
    final_merge_root: Any,
    source_files: Any,
    focused_tests: Any,
    test_receipt: Any,
) -> dict[str, Any]:
    """Build (but do not write) a strict pre-GPU receipt."""

    if checkpoint != {"checkpoint": CHECKPOINT, "step": STEP} and checkpoint != {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE}:
        raise PreGpuReceiptError("checkpoint identity must be exactly S step-2444")
    authority = _validate_authority(unit_authority)
    gate_refs, gate_docs = _validate_gate_artifacts(gate_v3_artifacts)
    support_refs = _validate_support_inputs(support_inputs)
    census_ref, census_doc = _validate_census_v3(census_v3)
    manifest_ref, manifest_doc = _validate_manifest(manifest, census_ref)
    plan_ref, plan_doc = _validate_execution_plan(execution_plan, manifest_ref, manifest_doc)
    config_id = _validate_identity(config, "S config")
    if config_id.get("checkpoint") != CHECKPOINT or config_id.get("step") != STEP:
        raise PreGpuReceiptError("config identity is not S step-2444")
    h0_id = _validate_h0_identity(h0)
    h0_dir_ref = {
        "path": h0_id["h0_dir"],
        "identity_files_sha256": h0_id["identity_files_sha256"],
        "image_plan_identity": h0_id["image_plan_identity"],
    }
    panel_id = _validate_identity(panel, "S panel")
    cohort_id = _validate_identity(cohort, "S cohort")
    if cohort_id.get("checkpoint") not in {None, CHECKPOINT} or cohort_id.get("step") not in {None, STEP}:
        raise PreGpuReceiptError("cohort identity is not S")
    if isinstance(cohort, Mapping) and cohort.get("frozen_arms") != list(ARM_ORDER):
        raise PreGpuReceiptError("cohort frozen arm order differs from S K/N/H")
    legacy_context = _validate_legacy_context_binding(
        manifest_doc,
        cohort_id,
        cohort_manifest,
    )
    manifest_panel = manifest_doc.get("panel")
    if not isinstance(manifest_panel, Mapping) or (
        manifest_panel.get("id") != panel_id.get("id")
        or manifest_panel.get("path") != panel_id.get("path")
        or manifest_panel.get("sha256") != panel_id.get("sha256")
    ):
        raise PreGpuReceiptError("manifest panel identity differs from exact live panel")
    manifest_cohort = manifest_doc.get("cohort")
    if not isinstance(manifest_cohort, Mapping) or (
        manifest_cohort.get("id") != cohort_id.get("id")
        or manifest_cohort.get("frozen_arms") != list(ARM_ORDER)
        or manifest_cohort.get("sha256") != cohort_id.get("manifest_identity_sha256")
    ):
        raise PreGpuReceiptError("manifest cohort identity differs from exact live cohort")
    runtime_doc, math_doc, policy_doc = _validate_runtime(runtime, forced_math, device_policy)
    runtime_evidence_ref, runtime_evidence_doc = _validate_runtime_evidence(
        runtime_evidence,
        runtime=runtime_doc,
        forced_math=math_doc,
        device_policy=policy_doc,
    )
    output_binding = _validate_output_binding(shard_roots, final_merge_root, phase="prelaunch")
    code_refs, code_hashes = _validate_code(source_files)
    src_runtime_tree = _regular_file_inventory(
        SRC_ROOT,
        "src runtime tree",
        suffix=".py",
    )
    tests_doc, test_result, test_refs, test_result_ref = _validate_tests(focused_tests, test_receipt)
    event_binding = _event_binding(manifest_doc)
    claim_scope = _claim_scope(manifest_doc)
    shard_authorizations = _shard_authorizations(plan_doc, manifest_doc, output_binding)
    immutable_inputs = {
        "manifest": manifest_ref,
        "census": census_ref,
        "config": {"path": config_id["path"], "sha256": config_id["sha256"]},
        "panel": {"path": panel_id["path"], "sha256": panel_id["sha256"]},
        "cohort": {"path": cohort_id["path"], "sha256": cohort_id["sha256"]},
        "cohort_manifest": legacy_context["companion_manifest"],
        "h0_root": {"path": h0_id["path"], "sha256": h0_id.get("sha256")},
        "base_model_dir": {
            "path": h0_id["base_model_dir"],
            "base_model_files_sha256": h0_id["base_model_files_sha256"],
            "base_model_inventory_sha256": h0_id["base_model_inventory_sha256"],
        },
        "execution_plan": plan_ref,
    }
    immutable_inputs["h0_dir"] = h0_dir_ref
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "unit_id": UNIT_ID,
        "checkpoint": CHECKPOINT,
        "step": STEP,
        "primary": {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE},
        "event_binding": event_binding,
        "claim_scope": claim_scope,
        "authorized_event_sha256s": event_binding["authorized_event_sha256s"],
        "event_bindings_sha256": event_binding["event_bindings_sha256"],
        "authorized_event_order_sha256": event_binding["authorized_event_order_sha256"],
        "authorized_shards": shard_authorizations,
        "unit_authority": authority,
        "gate_v3_artifacts": gate_refs,
        "support_inputs": support_refs,
        "immutable_inputs": immutable_inputs,
        "manifest_self_sha256": manifest_doc["self_sha256"],
        "source_census_sha256": census_ref["sha256"],
        "primary_identity": {
            "checkpoint": CHECKPOINT,
            "step": STEP,
            "substrate": SUBSTRATE,
            "config": config_id,
            "h0": h0_id,
            "panel": panel_id,
            "cohort": cohort_id,
            "legacy_context": legacy_context,
        },
        "runtime": runtime_doc,
        "runtime_evidence_ref": runtime_evidence_ref,
        "runtime_evidence": runtime_evidence_doc,
        "forced_math": math_doc,
        "device_policy": policy_doc,
        "output_binding": output_binding,
        "output_roots": output_binding,
        "source_files": code_refs,
        "code_identity": {
            "required_roles": list(CODE_ROLES),
            "sha256": code_hashes,
            "src_runtime_tree_sha256": src_runtime_tree["inventory_sha256"],
        },
        "src_runtime_tree": src_runtime_tree,
        "focused_test_execution": tests_doc,
        "focused_test_result": test_result,
        "focused_test_result_ref": test_result_ref,
        "focused_test_file_hashes": {str(index): ref["sha256"] for index, ref in enumerate(test_refs)},
        "gate_identity": {
            "result_sha256": gate_docs.get("result", {}).get("result_sha256"),
            "runtime_identity_sha256": gate_docs.get("runtime_identity", {}).get("identity_sha256"),
            "event_id": gate_docs.get("result", {}).get("event_id"),
        },
        "runtime_identity_contract": {
            "required": True,
            "scope": "exact_shard_and_every_arm_event",
            "receipt_path_field": "pre_gpu_receipt_path",
            "receipt_sha256_field": "pre_gpu_receipt_sha256",
            "receipt_self_sha256_field": "pre_gpu_receipt_self_sha256",
            "code_hashes_field": "code_hashes",
            "src_runtime_tree_sha256_field": "src_runtime_tree.inventory_sha256",
            "base_model_inventory_sha256_field": "input_hashes.base_model_inventory_sha256",
            "shard_device_assignment_field": "device_assignment",
            "observed_cuda_visible_devices_required": True,
            "repeat_exact_code_hashes": True,
        },
    }
    document["self_sha256"] = document_self_sha256(document)
    # The output itself is part of the authority; keep it absolute and do not
    # create/reserve the path until seal_receipt is called.
    document["output_policy"] = {"receipt_path": str(_absolute(output, "receipt output")), "overwrite": False, "collision": "reject_nonidentical"}
    document["self_sha256"] = document_self_sha256(document)
    return document


def _validate_ref(ref: Any, label: str, *, directory: bool = False) -> dict[str, Any]:
    if not isinstance(ref, Mapping):
        raise PreGpuReceiptError(f"{label} identity is missing")
    path = ref.get("path")
    if path is None:
        raise PreGpuReceiptError(f"{label}.path is missing")
    if directory:
        observed_path = _directory(path, label)
        if str(observed_path) != str(Path(path).expanduser()):
            raise PreGpuReceiptError(f"{label} path is not exact")
        if ref.get("sha256") is not None:
            _sha(ref["sha256"], f"{label}.sha256")
        return dict(ref)
    observed = _file_ref(path, label)
    if ref.get("sha256") != observed["sha256"]:
        raise PreGpuReceiptError(f"{label} hash drifted")
    if ref.get("size_bytes") is not None and ref.get("size_bytes") != observed["size_bytes"]:
        raise PreGpuReceiptError(f"{label} size drifted")
    return observed


def runtime_identity_binding(
    document: Mapping[str, Any],
    *,
    receipt_path: str | Path,
    shard_id: str | int,
    observed_cuda_visible_devices: str,
) -> dict[str, Any]:
    """Return the exact shard identity every arm/event must repeat."""

    validate_receipt(document, receipt_path=receipt_path, phase="runtime")
    path = str(_regular_file(receipt_path, "pre-GPU receipt"))
    authorization = _get_shard_authorization(document, shard_id=shard_id)
    if (
        not isinstance(observed_cuda_visible_devices, str)
        or observed_cuda_visible_devices != authorization["physical_device"]
    ):
        raise PreGpuReceiptError(
            "observed CUDA_VISIBLE_DEVICES differs from the authorized shard assignment"
        )
    immutable = document["immutable_inputs"]
    primary = document["primary_identity"]
    h0 = primary["h0"]
    input_hashes = {
        "manifest_raw_sha256": immutable["manifest"]["sha256"],
        "manifest_self_sha256": document["manifest_self_sha256"],
        "census_v3_raw_sha256": immutable["census"]["sha256"],
        "census_v3_self_sha256": immutable["census"].get("semantic_sha256"),
        "execution_plan_raw_sha256": immutable["execution_plan"]["sha256"],
        "execution_plan_sha256": immutable["execution_plan"].get("semantic_sha256"),
        "config_sha256": immutable["config"]["sha256"],
        "panel_sha256": immutable["panel"]["sha256"],
        "cohort_sha256": immutable["cohort"]["sha256"],
        "legacy_context_cohort_semantic_sha256": primary["legacy_context"]["cohort"]["semantic_sha256"],
        "legacy_context_manifest_raw_sha256": immutable["cohort_manifest"]["sha256"],
        "legacy_context_manifest_semantic_sha256": immutable["cohort_manifest"]["semantic_sha256"],
        "h0_identity_files_sha256": h0["identity_files_sha256"],
        "h0_image_plan_raw_sha256": h0["image_plan_identity"]["raw_sha256"],
        "h0_image_plan_normalized_rows_sha256": h0["image_plan_identity"]["normalized_rows_sha256"],
        "legacy_context_image_plan_bindings_sha256": primary["legacy_context"]["image_plan_bindings_sha256"],
        "base_model_files_sha256": h0["base_model_files_sha256"],
        "base_model_inventory_sha256": h0["base_model_inventory_sha256"],
        "src_runtime_tree_sha256": document["src_runtime_tree"]["inventory_sha256"],
        "event_bindings_sha256": document["event_bindings_sha256"],
        "authorized_event_order_sha256": document["authorized_event_order_sha256"],
        "authorized_shards_sha256": sha256_json(
            [item["authorization_sha256"] for item in document["authorized_shards"]]
        ),
    }
    body: dict[str, Any] = {
        "schema_version": "s_natural_boundary_k_n_h_runtime_identity_binding.v1",
        "unit_id": UNIT_ID,
        "checkpoint": CHECKPOINT,
        "step": STEP,
        "substrate": SUBSTRATE,
        "pre_gpu_receipt_path": path,
        "pre_gpu_receipt_sha256": sha256_file(path),
        "pre_gpu_receipt_self_sha256": document["self_sha256"],
        "input_hashes": input_hashes,
        "input_paths": {
            key: immutable[key]["path"]
            for key in (
                "manifest", "census", "execution_plan", "config", "panel",
                "cohort", "cohort_manifest", "h0_root", "h0_dir", "base_model_dir",
            )
        },
        "code_hashes": {
            role: document["code_identity"]["sha256"][role]
            for role in RUNTIME_IDENTITY_CODE_ROLES
        },
        "src_runtime_tree": {
            "root": document["src_runtime_tree"]["root"],
            "file_count": document["src_runtime_tree"]["file_count"],
            "inventory_sha256": document["src_runtime_tree"]["inventory_sha256"],
        },
        "h0_image_plan": dict(h0["image_plan_identity"]),
        "runtime": dict(document["runtime"]),
        "forced_math": dict(document["forced_math"]),
        "device_policy": dict(document["device_policy"]),
        "device_assignment": {
            "shard_id": authorization["shard_id"],
            "shard_index": authorization["shard_index"],
            "physical_device": authorization["physical_device"],
            "observed_cuda_visible_devices": observed_cuda_visible_devices,
            "logical_device": authorization["logical_device"],
            "device_count": authorization["device_count"],
            "authorization_sha256": authorization["authorization_sha256"],
        },
        "claim_scope": dict(document["claim_scope"]),
        "no_training": True,
    }
    body["binding_sha256"] = sha256_json(body)
    return body


def validate_runtime_identity(
    identity: Mapping[str, Any],
    *,
    receipt_path: str | Path,
    receipt: Mapping[str, Any],
    shard_id: str | int,
    observed_cuda_visible_devices: str,
    expected_code_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise PreGpuReceiptError("runtime identity must be an object")
    expected = runtime_identity_binding(
        receipt,
        receipt_path=receipt_path,
        shard_id=shard_id,
        observed_cuda_visible_devices=observed_cuda_visible_devices,
    )
    if expected_code_hashes is not None and dict(expected_code_hashes) != expected["code_hashes"]:
        raise PreGpuReceiptError("caller expected code hashes differ from receipt")
    if dict(identity) != expected:
        raise PreGpuReceiptError("runtime identity differs from exact pre-GPU binding")
    return expected


def _normalize_shard_id(shard_id: str | int) -> tuple[str, int]:
    if isinstance(shard_id, bool):
        raise PreGpuReceiptError("shard_id must be an index or shard-NNN string")
    if isinstance(shard_id, int):
        index = shard_id
        normalized = f"shard-{index:03d}"
    elif isinstance(shard_id, str) and re.fullmatch(r"shard-[0-9]{3}", shard_id):
        normalized = shard_id
        index = int(shard_id[6:])
    else:
        raise PreGpuReceiptError("shard_id must be an index or shard-NNN string")
    if not 0 <= index < SHARD_COUNT or normalized != f"shard-{index:03d}":
        raise PreGpuReceiptError("shard_id is outside the frozen 0..7 range")
    return normalized, index


def _get_shard_authorization(
    document: Mapping[str, Any],
    *,
    shard_id: str | int,
) -> dict[str, Any]:
    normalized, index = _normalize_shard_id(shard_id)
    shards = document.get("authorized_shards")
    if not isinstance(shards, list) or len(shards) != SHARD_COUNT:
        raise PreGpuReceiptError("receipt authorized_shards is incomplete")
    authorization = shards[index]
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("shard_id") != normalized
        or authorization.get("shard_index") != index
    ):
        raise PreGpuReceiptError("receipt shard authorization identity drifted")
    body = dict(authorization)
    supplied = body.pop("authorization_sha256", None)
    if supplied != sha256_json(body):
        raise PreGpuReceiptError("receipt shard authorization hash mismatch")
    expected_device = SHARD_PHYSICAL_DEVICES[normalized]
    if (
        authorization.get("physical_device") != expected_device
        or authorization.get("logical_device") != "cuda:0"
        or authorization.get("device_count") != 1
    ):
        raise PreGpuReceiptError("receipt shard device assignment drifted")
    return dict(authorization)


def validate_shard_authorization(
    document: Mapping[str, Any],
    *,
    shard_id: str | int,
    output_root: str | Path,
    event_sha256s: Sequence[str],
    observed_cuda_visible_devices: str,
) -> dict[str, Any]:
    """Validate one runner-owned shard assignment after receipt validation."""

    authorization = _get_shard_authorization(document, shard_id=shard_id)
    expected_root = str(_absolute(output_root, "authorized shard output root"))
    if authorization.get("output_root") != expected_root:
        raise PreGpuReceiptError("runner output root differs from receipt authorization")
    observed = list(event_sha256s)
    for position, digest in enumerate(observed):
        _sha(digest, f"shard event_sha256s[{position}]")
    if observed != authorization.get("event_sha256s"):
        raise PreGpuReceiptError("runner event assignment differs from receipt authorization")
    if (
        not isinstance(observed_cuda_visible_devices, str)
        or observed_cuda_visible_devices != authorization["physical_device"]
    ):
        raise PreGpuReceiptError(
            "observed CUDA_VISIBLE_DEVICES differs from the authorized shard assignment"
        )
    return dict(authorization)


def validate_receipt(
    document: Mapping[str, Any],
    *,
    receipt_path: str | Path | None = None,
    expected_paths: Mapping[str, str | Path] | None = None,
    phase: str = "prelaunch",
) -> None:
    """Re-hash a sealed receipt; this function is safe before model load."""

    if not isinstance(document, Mapping):
        raise PreGpuReceiptError("pre-GPU receipt must be a JSON object")
    canonical_json_bytes(document)
    required = {
        "schema_version", "status", "unit_id", "checkpoint", "step", "primary",
        "event_binding", "claim_scope", "authorized_event_sha256s", "event_bindings_sha256",
        "authorized_event_order_sha256", "authorized_shards", "unit_authority",
        "gate_v3_artifacts", "support_inputs", "immutable_inputs",
        "manifest_self_sha256", "source_census_sha256", "primary_identity",
        "code_identity", "source_files", "src_runtime_tree", "focused_test_execution",
        "focused_test_result", "focused_test_result_ref", "output_binding", "output_roots", "runtime",
        "runtime_evidence_ref", "runtime_evidence", "forced_math", "device_policy", "output_policy", "self_sha256",
    }
    missing = sorted(required - set(document))
    if missing:
        raise PreGpuReceiptError(f"pre-GPU receipt is missing fields: {missing}")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("status") != STATUS or document.get("unit_id") != UNIT_ID or document.get("checkpoint") != CHECKPOINT or document.get("step") != STEP:
        raise PreGpuReceiptError("pre-GPU receipt schema/status/unit identity drifted")
    if document.get("self_sha256") != document_self_sha256(document):
        raise PreGpuReceiptError("pre-GPU receipt self hash mismatch")
    primary = document.get("primary")
    if not isinstance(primary, Mapping) or dict(primary) != {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE}:
        raise PreGpuReceiptError("pre-GPU receipt primary identity drifted")
    immutable = document.get("immutable_inputs")
    if not isinstance(immutable, Mapping):
        raise PreGpuReceiptError("immutable_inputs is missing")
    for key in ("config", "panel", "cohort", "cohort_manifest"):
        _validate_ref(immutable.get(key), f"immutable input {key}")
    _validate_ref(immutable.get("h0_root"), "immutable input h0_root", directory=True)
    _validate_ref(immutable.get("base_model_dir"), "immutable input base_model_dir", directory=True)
    if "h0_dir" in immutable:
        _validate_ref(immutable.get("h0_dir"), "immutable input h0_dir", directory=True)
    census_ref, _census_doc = _validate_census_v3(immutable.get("census"))
    manifest_ref, manifest_doc = _validate_manifest(immutable.get("manifest"), census_ref)
    plan_ref, plan_doc = _validate_execution_plan(immutable.get("execution_plan"), manifest_ref, manifest_doc)
    if document.get("manifest_self_sha256") != manifest_doc.get("self_sha256") or document.get("source_census_sha256") != census_ref["sha256"]:
        raise PreGpuReceiptError("receipt manifest/census authority hashes drifted")
    if expected_paths:
        for key, expected in expected_paths.items():
            if key not in immutable:
                raise PreGpuReceiptError(f"expected path binding {key} is absent")
            bound = immutable[key].get("path") if isinstance(immutable[key], Mapping) else None
            expected_path = _absolute(expected, f"expected path {key}")
            if bound != str(expected_path):
                raise PreGpuReceiptError(f"immutable input {key} path differs from live input")
            if key in {"h0_root", "h0_dir"}:
                _directory(expected_path, f"expected path {key}")
            else:
                observed = sha256_file(expected_path)
                if immutable[key].get("sha256") != observed:
                    raise PreGpuReceiptError(f"immutable input {key} hash differs from live input")
    expected_events = _event_binding(manifest_doc)
    if document.get("event_binding") != expected_events:
        raise PreGpuReceiptError("receipt full-manifest event authorization drifted")
    if (
        document.get("authorized_event_sha256s") != expected_events["authorized_event_sha256s"]
        or document.get("event_bindings_sha256") != expected_events["event_bindings_sha256"]
        or document.get("authorized_event_order_sha256") != expected_events["authorized_event_order_sha256"]
    ):
        raise PreGpuReceiptError("receipt ordered event authorization aliases drifted")
    if document.get("claim_scope") != _claim_scope(manifest_doc):
        raise PreGpuReceiptError("receipt case-study/checkpoint claim scope drifted")
    _validate_authority(document.get("unit_authority"))
    _validate_gate_artifacts(document.get("gate_v3_artifacts"))
    _validate_support_inputs(document.get("support_inputs"))
    primary_identity = document.get("primary_identity")
    if not isinstance(primary_identity, Mapping) or primary_identity.get("checkpoint") != CHECKPOINT or primary_identity.get("step") != STEP or primary_identity.get("substrate") != SUBSTRATE:
        raise PreGpuReceiptError("receipt primary_identity is not S step-2444")
    for name, label in (
        ("config", "S config"),
        ("panel", "S panel"),
        ("cohort", "S cohort"),
    ):
        value = primary_identity.get(name)
        observed = _validate_identity(value, label)
        if observed.get("path") != immutable[name].get("path"):
            raise PreGpuReceiptError(f"primary_identity {name} differs from immutable input")
    observed_h0 = _validate_h0_identity(
        primary_identity.get("h0"),
        require_bound_inventory=True,
    )
    if (
        observed_h0.get("path") != immutable["h0_root"].get("path")
        or observed_h0.get("h0_dir") != immutable["h0_dir"].get("path")
        or observed_h0.get("identity_files_sha256") != immutable["h0_dir"].get("identity_files_sha256")
        or observed_h0.get("image_plan_identity") != immutable["h0_dir"].get("image_plan_identity")
        or observed_h0.get("base_model_dir") != immutable["base_model_dir"].get("path")
        or observed_h0.get("base_model_files_sha256") != immutable["base_model_dir"].get("base_model_files_sha256")
        or observed_h0.get("base_model_inventory_sha256") != immutable["base_model_dir"].get("base_model_inventory_sha256")
    ):
        raise PreGpuReceiptError("primary_identity H0 differs from exact immutable H0 binding")
    observed_panel = primary_identity["panel"]
    observed_cohort = primary_identity["cohort"]
    observed_legacy_context = _validate_legacy_context_binding(
        manifest_doc,
        observed_cohort,
        immutable["cohort_manifest"],
    )
    if primary_identity.get("legacy_context") != observed_legacy_context:
        raise PreGpuReceiptError("primary_identity legacy context pair drifted")
    if (
        observed_legacy_context["cohort"]["path"] != immutable["cohort"]["path"]
        or observed_legacy_context["cohort"]["sha256"] != immutable["cohort"]["sha256"]
        or observed_legacy_context["companion_manifest"] != immutable["cohort_manifest"]
    ):
        raise PreGpuReceiptError("immutable legacy context pair differs from manifest authority")
    if manifest_doc.get("panel") != {
        "id": observed_panel.get("id"),
        "path": observed_panel.get("path"),
        "sha256": observed_panel.get("sha256"),
    }:
        raise PreGpuReceiptError("manifest panel differs from primary_identity panel")
    manifest_cohort = manifest_doc.get("cohort")
    if not isinstance(manifest_cohort, Mapping) or (
        manifest_cohort.get("id") != observed_cohort.get("id")
        or manifest_cohort.get("frozen_arms") != list(ARM_ORDER)
        or manifest_cohort.get("sha256") != observed_cohort.get("manifest_identity_sha256")
    ):
        raise PreGpuReceiptError("manifest cohort differs from primary_identity cohort")
    _validate_code(document.get("source_files"))
    observed_src_runtime_tree = _validate_inventory(
        document.get("src_runtime_tree"),
        "src runtime tree",
        expected_root=SRC_ROOT,
        suffix=".py",
    )
    code = document.get("code_identity")
    if (
        not isinstance(code, Mapping)
        or code.get("required_roles") != list(CODE_ROLES)
        or not isinstance(code.get("sha256"), Mapping)
        or dict(code["sha256"])
        != {
            role: _validate_ref(
                document["source_files"].get(role),
                f"source file {role}",
            )["sha256"]
            for role in CODE_ROLES
        }
        or code.get("src_runtime_tree_sha256")
        != observed_src_runtime_tree["inventory_sha256"]
    ):
        raise PreGpuReceiptError("code identity hashes drifted")
    _validate_runtime(document.get("runtime"), document.get("forced_math"), document.get("device_policy"))
    observed_runtime_ref, observed_runtime_evidence = _validate_runtime_evidence(
        document.get("runtime_evidence_ref"),
        runtime=document.get("runtime"),
        forced_math=document.get("forced_math"),
        device_policy=document.get("device_policy"),
    )
    if observed_runtime_evidence != document.get("runtime_evidence") or observed_runtime_ref.get("sha256") != document["runtime_evidence_ref"].get("sha256"):
        raise PreGpuReceiptError("installed runtime evidence raw/content binding drifted")
    _tests, observed_test_result, _test_refs, observed_test_ref = _validate_tests(
        None, document.get("focused_test_result_ref")
    )
    if observed_test_result != document.get("focused_test_result"):
        raise PreGpuReceiptError("focused test result file differs from sealed result")
    if observed_test_ref.get("sha256") != document["focused_test_result_ref"].get("sha256"):
        raise PreGpuReceiptError("focused test result raw hash drifted")
    outputs = document.get("output_binding")
    if not isinstance(outputs, Mapping):
        raise PreGpuReceiptError("output_binding is missing")
    if document.get("output_roots") != outputs or outputs.get("phase") != "prelaunch":
        raise PreGpuReceiptError("output root aliases/status drifted")
    observed_outputs = _validate_output_binding(
        [item for item in outputs.get("shard_roots", [])],
        outputs.get("final_merge_root", {}).get("path") if isinstance(outputs.get("final_merge_root"), Mapping) else None,
        phase=phase,
    )
    expected_shards = _shard_authorizations(plan_doc, manifest_doc, outputs)
    if document.get("authorized_shards") != expected_shards:
        raise PreGpuReceiptError("authorized shard membership/root bindings drifted")
    if [item["path"] for item in observed_outputs["shard_roots"]] != [item["path"] for item in outputs["shard_roots"]] or observed_outputs["final_merge_root"]["path"] != outputs["final_merge_root"]["path"]:
        raise PreGpuReceiptError("validated output roots differ from receipt authority")
    if receipt_path is not None:
        path = _regular_file(receipt_path, "pre-GPU receipt")
        raw = path.read_bytes()
        if raw != canonical_json_bytes(document) + b"\n":
            raise PreGpuReceiptError("receipt file is not canonical JSON with one trailing newline")
        policy = document.get("output_policy")
        if not isinstance(policy, Mapping) or policy.get("receipt_path") != str(path) or policy.get("overwrite") is not False:
            raise PreGpuReceiptError("receipt file path differs from sealed output policy")


def seal_receipt(document: Mapping[str, Any], *, output: str | Path | None = None) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise PreGpuReceiptError("receipt must be an object")
    validate_receipt(document, phase="prelaunch")
    target = output or (document.get("output_policy", {}).get("receipt_path") if isinstance(document.get("output_policy"), Mapping) else None)
    if target is None:
        raise PreGpuReceiptError("receipt output path is missing")
    if isinstance(document.get("output_policy"), Mapping) and document["output_policy"].get("receipt_path") != str(_absolute(target, "receipt output")):
        raise PreGpuReceiptError("receipt output differs from sealed output policy")
    return _write_once(target, document)


def _role_paths(values: Sequence[str], roles: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise PreGpuReceiptError(f"{label} must use ROLE=/absolute/path syntax")
        role, raw_path = item.split("=", 1)
        if role not in roles or role in result:
            raise PreGpuReceiptError(f"{label} has unknown/duplicate role {role!r}")
        result[role] = _regular_file(raw_path, f"{label} {role}")
    if set(result) != set(roles):
        raise PreGpuReceiptError(f"{label} roles differ; expected {tuple(roles)}")
    return result


def _read_canonical_object(path: Path, label: str) -> dict[str, Any]:
    _ref, document = _json_file_ref(path, label, canonical=True)
    return document


def _build_from_cli(args: argparse.Namespace) -> dict[str, Any]:
    support_shards = list(args.support_shard_receipt)
    if len(support_shards) != SHARD_COUNT:
        raise PreGpuReceiptError("builder requires exactly eight --support-shard-receipt values")
    shard_roots = list(args.shard_root)
    if len(shard_roots) != SHARD_COUNT:
        raise PreGpuReceiptError("builder requires exactly eight --shard-root values")
    h0_files = _role_paths(args.h0_file, H0_IDENTITY_FILE_ROLES, "--h0-file")
    base_files = _role_paths(args.base_model_file, BASE_MODEL_FILE_ROLES, "--base-model-file")
    runtime_evidence = _read_canonical_object(args.runtime_evidence, "installed runtime evidence")
    manifest_document = _read_canonical_object(args.manifest, "admitted cohort manifest")
    manifest_cohort = manifest_document.get("cohort")
    if not isinstance(manifest_cohort, Mapping):
        raise PreGpuReceiptError("manifest cohort identity is missing")
    config_ref = _file_ref(args.config, "S config")
    panel_ref = _file_ref(args.panel, "S panel")
    cohort_ref = _file_ref(args.cohort, "S cohort")
    document = build_receipt(
        output=args.output,
        unit_authority={
            "path": str(args.unit_authority),
            "sha256": sha256_file(args.unit_authority),
            "unit_id": UNIT_ID,
            "status": args.authority_status,
            "scope": args.authority_scope,
        },
        gate_v3_artifacts={
            "result": args.gate_result,
            "runtime_identity": args.gate_runtime_identity,
            "terminal_summary": args.gate_terminal_summary,
            "launch_log": args.gate_launch_log,
        },
        support_inputs={
            "plan": args.support_plan,
            "census_v2": args.census_v2,
            "shard_receipts": support_shards,
            "merge_ledger": args.support_merge_ledger,
            "merge_receipt": args.support_merge_receipt,
        },
        census_v3=args.census_v3,
        manifest=args.manifest,
        execution_plan=args.execution_plan,
        config={
            "id": args.config_id,
            **config_ref,
            "checkpoint": CHECKPOINT,
            "step": STEP,
        },
        checkpoint={"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE},
        h0={
            "id": args.h0_id,
            "path": str(args.h0_root),
            "h0_dir": str(args.h0_dir),
            "checkpoint": CHECKPOINT,
            "step": STEP,
            "identity_files": h0_files,
            "base_model_dir": str(args.base_model_dir),
            "base_model_files": base_files,
        },
        panel={"id": args.panel_id, **panel_ref},
        cohort={
            "id": args.cohort_id,
            **cohort_ref,
            "manifest_identity_sha256": manifest_cohort.get("sha256"),
            "checkpoint": CHECKPOINT,
            "step": STEP,
            "frozen_arms": list(ARM_ORDER),
        },
        cohort_manifest=args.cohort_manifest,
        runtime=runtime_evidence.get("runtime"),
        runtime_evidence=args.runtime_evidence,
        forced_math=runtime_evidence.get("forced_math"),
        device_policy=runtime_evidence.get("device_policy"),
        shard_roots=shard_roots,
        final_merge_root=args.final_merge_root,
        source_files=CODE_ROLE_PATHS,
        focused_tests=REQUIRED_FOCUSED_TEST_PATHS,
        test_receipt=args.focused_test_receipt,
    )
    seal_result = seal_receipt(document)
    return {
        **seal_result,
        "self_sha256": document["self_sha256"],
        "event_count": document["event_binding"]["event_count"],
        "shard_count": len(document["authorized_shards"]),
    }


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit-authority", type=Path, required=True)
    parser.add_argument("--authority-status", required=True)
    parser.add_argument("--authority-scope", required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--gate-runtime-identity", type=Path, required=True)
    parser.add_argument("--gate-terminal-summary", type=Path, required=True)
    parser.add_argument("--gate-launch-log", type=Path, required=True)
    parser.add_argument("--support-plan", type=Path, required=True)
    parser.add_argument("--census-v2", type=Path, required=True)
    parser.add_argument("--support-shard-receipt", type=Path, action="append", required=True)
    parser.add_argument("--support-merge-ledger", type=Path, required=True)
    parser.add_argument("--support-merge-receipt", type=Path, required=True)
    parser.add_argument("--census-v3", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--h0-root", type=Path, required=True)
    parser.add_argument("--h0-dir", type=Path, required=True)
    parser.add_argument("--h0-id", required=True)
    parser.add_argument("--h0-file", action="append", required=True, metavar="ROLE=/ABS/PATH")
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--base-model-file", action="append", required=True, metavar="ROLE=/ABS/PATH")
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    parser.add_argument("--focused-test-receipt", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, action="append", required=True)
    parser.add_argument("--final-merge-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build and write one exact pre-GPU receipt")
    _add_build_arguments(build)
    validate = subparsers.add_parser("validate", help="validate an existing receipt")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--phase", choices=("prelaunch", "runtime", "postrun"), required=True)
    existing = subparsers.add_parser("seal-existing", help="write-once copy of a prepared canonical receipt")
    existing.add_argument("--input", type=Path, required=True)
    existing.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            result = _build_from_cli(args)
        elif args.command == "validate":
            value = _read_canonical_object(args.receipt, "pre-GPU receipt")
            validate_receipt(value, receipt_path=args.receipt, phase=args.phase)
            result = {"status": "valid", "path": str(args.receipt.resolve()), "phase": args.phase}
        else:
            raw = args.input.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                raise PreGpuReceiptError("input receipt must be an object")
            if raw != canonical_json_bytes(value) + b"\n":
                raise PreGpuReceiptError("input receipt is not canonical JSON with one trailing newline")
            validate_receipt(value, phase="prelaunch")
            result = seal_receipt(value, output=args.output)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (PreGpuReceiptError, FileExistsError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Strict read-only Wave 7 uninterrupted-versus-resumed artifact comparator."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.coordexp_swift import (  # noqa: E402
    wave7_exact_resume_interrupt as interrupt,
)
from src.artifacts.checkpoints import _validate_adapter_payload  # noqa: E402
from src.artifacts.training_state import (  # noqa: E402
    REQUIRED_RNG_KINDS,
    AdmittedTrainingState,
    DecodedRankTrainingState,
    TrainingStateExpectations,
    TrainingStateManifest,
    admit_training_state,
    build_resume_compatibility_projection,
    load_training_state_manifest,
)
from src.qwen.parity import (  # noqa: E402
    ParityContractError,
    assert_absent_artifact_target,
    canonical_json_bytes,
    write_strict_json_atomic,
)
from src.qwen.special_token_embeddings import (  # noqa: E402
    inspect_special_token_embedding_delta_payload,
)


RECEIPT_SCHEMA = "coordexp-swift-wave7-exact-resume-comparison-v1"
AUDITED_INTERRUPT_MARKER_SCHEMA = "coordexp-swift-wave7-interrupt-marker-v1"
AUDITED_INTERRUPT_RECEIPT_SCHEMA = "coordexp-swift-wave7-interrupt-receipt-v1"
AUDITED_INTERRUPT_SOURCE_SHA256 = (
    "7018e7e177d74940e9397caa6152f5335d55e68bb1e98e55705725a749ccdbdd"
)
AUDITED_INTERRUPT_PATH = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_interrupt.py"
)
EXPECTED_WORLD_SIZE = 8
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_LOG_ROWS = 10_000
MAX_MISMATCH_ROWS = 50_000
HEX = frozenset("0123456789abcdef")
STRICT_IDENTITY_KEYS = (
    "base_model",
    "cache",
    "dependencies",
    "policy",
    "resume_compatibility",
    "topology",
    "trainable_surface",
)
RUN_FIELDS = frozenset(
    {
        "artifact_root",
        "checkpoint_event_count",
        "collision_outcome",
        "completed_at",
        "completed_steps",
        "config_fingerprint",
        "consumed_packs",
        "continuation",
        "created_at",
        "final_finite_status",
        "final_optimizer_update_status",
        "forward_input_provider_mode",
        "forward_input_provider_resolution",
        "materializations",
        "measurement",
        "policy_identities",
        "provenance",
        "resolved_config_path",
        "resolved_max_steps",
        "run_dir",
        "run_id",
        "run_name",
        "runtime",
        "status",
        "terminal_error",
        "updated_at",
        "warning_counts",
    }
)
RESOLUTION_FIELDS = frozenset(
    {
        "entry_config_path",
        "fingerprint",
        "loader_version",
        "path_origins",
        "schema_version",
        "sources",
    }
)
MARKER_FIELDS = frozenset(
    {
        "captured_pgid_set",
        "captured_pid_set",
        "captured_process_graph",
        "checkpoint",
        "checkpoint_publication_event",
        "checkpoint_publication_run",
        "config_hashes",
        "gpu_compute_apps_baseline",
        "launch",
        "launcher",
        "marker_target",
        "receipt_target",
        "schema",
        "script",
        "source_hashes",
        "target_binding",
        "timestamp",
    }
)
TERMINATION_RECEIPT_FIELDS = frozenset(
    {
        "checkpoint",
        "checkpoint_publication",
        "config_hashes",
        "duration_seconds",
        "errors",
        "events",
        "finished_at",
        "launch",
        "marker",
        "postconditions",
        "request",
        "schema",
        "script",
        "source_hashes",
        "started_at",
        "status",
        "termination",
        "target_binding",
    }
)
POSTCONDITION_FIELDS = frozenset(
    {
        "checkpoint_publication_event_unchanged",
        "checkpoint_publication_run_final_file_sha256",
        "final_json_absent",
        "late_write_detected",
        "manifest_final_aggregate_digest",
        "manifest_final_file_sha256",
        "manifest_unchanged",
        "max_logged_train_step",
        "nvidia_compute_apps",
        "nvidia_compute_apps_added",
        "nvidia_compute_apps_baseline",
        "run_not_completed",
        "run_status",
        "stability_after",
        "stability_before",
        "step_5_absent",
    }
)
TERMINATION_FIELDS = frozenset(
    {
        "capture_completed_monotonic",
        "captured_process_graph",
        "cleanup_errors",
        "captured_pgids",
        "captured_pids",
        "duration_seconds",
        "launcher_exited",
        "launcher_returncode",
        "post_marker_discovered_pids",
        "reaped",
        "remaining_pgids",
        "remaining_pids",
    }
)
MARKER_OBSERVATION_FIELDS = frozenset(
    {
        "expected_file_sha256",
        "file_sha256",
        "final_file_sha256",
        "path",
        "published",
        "strict_payload_equal",
        "unchanged",
    }
)
TIMING_FIELDS = frozenset(
    {"step_duration_seconds", "input_build_seconds", "input_wait_seconds"}
)
CHECKPOINT_PUBLICATION_EVENT_FIELDS = frozenset(
    {
        "checkpoint_identity",
        "checkpoint_path",
        "completed_at",
        "duration_clock",
        "duration_seconds",
        "exact_training_state_enabled",
        "failure_code",
        "is_final",
        "started_at",
        "status",
        "step",
    }
)
CHECKPOINT_PUBLICATION_RUN_FIELDS = frozenset({"file_sha256", "path", "size"})
CHECKPOINT_PUBLICATION_RECEIPT_FIELDS = frozenset(
    {"event", "run_file_sha256", "run_path", "run_size"}
)


class Wave7CompareError(RuntimeError):
    """Typed fail-closed comparator contract error."""

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


@dataclass(frozen=True)
class CompareRequest:
    uninterrupted_run_dir: Path
    interrupted_parent_run_dir: Path
    resume_child_run_dir: Path
    interruption_marker: Path
    termination_receipt: Path
    output: Path
    expected_interrupt_source_sha256: str
    expected_source_sha256: str
    expected_provenance_sha256: str


@dataclass(frozen=True)
class RunArtifacts:
    role: str
    root: Path
    run: Mapping[str, Any]
    config: Mapping[str, Any]
    logs: tuple[Mapping[str, Any], ...]
    hashes: Mapping[str, Any]


def _digest_arg(value: str) -> str:
    if len(value) != 64 or any(character not in HEX for character in value):
        raise argparse.ArgumentTypeError("must be one lowercase SHA-256 digest")
    return value


def parse_request(argv: Sequence[str] | None = None) -> CompareRequest:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted-run-dir", type=Path, required=True)
    parser.add_argument("--interrupted-parent-run-dir", type=Path, required=True)
    parser.add_argument("--resume-child-run-dir", type=Path, required=True)
    parser.add_argument("--interruption-marker", type=Path, required=True)
    parser.add_argument("--termination-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-interrupt-source-sha256", type=_digest_arg, required=True
    )
    parser.add_argument("--expected-source-sha256", type=_digest_arg, required=True)
    parser.add_argument("--expected-provenance-sha256", type=_digest_arg, required=True)
    args = parser.parse_args(argv)
    roots = tuple(
        path.expanduser().resolve()
        for path in (
            args.uninterrupted_run_dir,
            args.interrupted_parent_run_dir,
            args.resume_child_run_dir,
        )
    )
    if len(set(roots)) != 3:
        parser.error("the three run directories must be distinct")
    return CompareRequest(
        uninterrupted_run_dir=roots[0],
        interrupted_parent_run_dir=roots[1],
        resume_child_run_dir=roots[2],
        interruption_marker=args.interruption_marker.expanduser().resolve(),
        termination_receipt=args.termination_receipt.expanduser().resolve(),
        output=args.output.expanduser().resolve(strict=False),
        expected_interrupt_source_sha256=args.expected_interrupt_source_sha256,
        expected_source_sha256=args.expected_source_sha256,
        expected_provenance_sha256=args.expected_provenance_sha256,
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _validate_input_topology(request: CompareRequest) -> None:
    paths = {
        "uninterrupted_run_dir": request.uninterrupted_run_dir,
        "interrupted_parent_run_dir": request.interrupted_parent_run_dir,
        "resume_child_run_dir": request.resume_child_run_dir,
        "interruption_marker": request.interruption_marker,
        "termination_receipt": request.termination_receipt,
        "output": request.output,
    }
    items = tuple(paths.items())
    overlaps = [
        {
            "left": left_name,
            "left_path": str(left_path),
            "right": right_name,
            "right_path": str(right_path),
        }
        for index, (left_name, left_path) in enumerate(items)
        for right_name, right_path in items[index + 1 :]
        if _paths_overlap(left_path, right_path)
    ]
    if overlaps:
        raise Wave7CompareError(
            "run trees and comparator artifacts must be pairwise disjoint and non-ancestor",
            code="wave7_compare.path_topology",
            context={"overlaps": overlaps},
        )


def _validate_interrupt_source_pin(request: CompareRequest) -> dict[str, Any]:
    canonical_path = AUDITED_INTERRUPT_PATH.resolve()
    observed_module_path = Path(interrupt.__file__).resolve()
    observed_sha256 = _sha256_file(
        _regular_file(canonical_path, owner="audited interrupt source")
    )
    expected_schemas = {
        "marker": AUDITED_INTERRUPT_MARKER_SCHEMA,
        "receipt": AUDITED_INTERRUPT_RECEIPT_SCHEMA,
    }
    observed_schemas = {
        "marker": interrupt.MARKER_SCHEMA,
        "receipt": interrupt.RECEIPT_SCHEMA,
    }
    if (
        request.expected_interrupt_source_sha256 != AUDITED_INTERRUPT_SOURCE_SHA256
        or observed_sha256 != AUDITED_INTERRUPT_SOURCE_SHA256
        or observed_module_path != canonical_path
        or observed_schemas != expected_schemas
    ):
        raise Wave7CompareError(
            "interrupt controller source or schema differs from the audited pin",
            code="wave7_compare.interrupt_source_mismatch",
            context={
                "audited_path": str(canonical_path),
                "expected_sha256": AUDITED_INTERRUPT_SOURCE_SHA256,
                "expected_schemas": expected_schemas,
                "observed_module_path": str(observed_module_path),
                "observed_sha256": observed_sha256,
                "observed_schemas": observed_schemas,
                "requested_sha256": request.expected_interrupt_source_sha256,
            },
        )
    return {
        "marker_schema": AUDITED_INTERRUPT_MARKER_SCHEMA,
        "path": str(canonical_path),
        "receipt_schema": AUDITED_INTERRUPT_RECEIPT_SCHEMA,
        "sha256": AUDITED_INTERRUPT_SOURCE_SHA256,
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_loads(encoded: bytes, *, owner: str) -> Any:
    if len(encoded) > MAX_JSON_BYTES:
        raise Wave7CompareError(
            f"{owner} exceeds the JSON size bound",
            code="wave7_compare.json_oversize",
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
        raise ValueError(f"non-finite value: {value}")

    try:
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise Wave7CompareError(
            f"{owner} is not strict JSON",
            code="wave7_compare.json_malformed",
            context={"owner": owner, "error_type": type(exc).__name__},
        ) from exc


def _reject_symlink_components(path: Path, *, owner: str) -> None:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise Wave7CompareError(
                f"{owner} contains a symlink",
                code="wave7_compare.unsafe_path",
                context={"path": str(current)},
            )


def _regular_file(path: Path, *, owner: str) -> Path:
    requested = path.expanduser().absolute()
    _reject_symlink_components(requested, owner=owner)
    try:
        mode = requested.lstat().st_mode
    except OSError as exc:
        raise Wave7CompareError(
            f"{owner} is unavailable",
            code="wave7_compare.input_missing",
            context={"path": str(requested)},
        ) from exc
    if not stat.S_ISREG(mode):
        raise Wave7CompareError(
            f"{owner} is not a regular file",
            code="wave7_compare.unsafe_path",
            context={"path": str(requested)},
        )
    return requested


def _strict_json_file(path: Path, *, owner: str) -> dict[str, Any]:
    regular = _regular_file(path, owner=owner)
    first = regular.read_bytes()
    value = _strict_json_loads(first, owner=owner)
    if first != regular.read_bytes():
        raise Wave7CompareError(
            f"{owner} changed while loading",
            code="wave7_compare.input_drift",
            context={"path": str(regular)},
        )
    if not isinstance(value, Mapping):
        raise Wave7CompareError(
            f"{owner} must be an object", code="wave7_compare.schema"
        )
    return dict(value)


def _require_fields(
    value: Mapping[str, Any], expected: frozenset[str], owner: str
) -> None:
    if set(value) != expected:
        raise Wave7CompareError(
            f"{owner} has an invalid field set",
            code="wave7_compare.schema",
            context={
                "owner": owner,
                "missing": sorted(expected - set(value)),
                "unexpected": sorted(set(value) - expected),
            },
        )


def _strict_logging_file(path: Path, *, owner: str) -> tuple[dict[str, Any], ...]:
    regular = _regular_file(path, owner=owner)
    first = regular.read_bytes()
    if len(first) > MAX_JSON_BYTES:
        raise Wave7CompareError(
            f"{owner} exceeds the size bound", code="wave7_compare.json_oversize"
        )
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(first.splitlines(), start=1):
        if not line.strip():
            raise Wave7CompareError(
                f"{owner} contains a blank row", code="wave7_compare.logging_schema"
            )
        raw = _strict_json_loads(line, owner=f"{owner} row {index}")
        if not isinstance(raw, Mapping):
            raise Wave7CompareError(
                f"{owner} row must be an object",
                code="wave7_compare.logging_schema",
            )
        row = dict(raw)
        step = row.get("step")
        split = row.get("split")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or split not in {"train", "eval"}
            or not isinstance(row.get("non_finite_fields"), list)
        ):
            raise Wave7CompareError(
                f"{owner} row has invalid base fields",
                code="wave7_compare.logging_schema",
                context={"row": index},
            )
        rows.append(row)
        if len(rows) > MAX_LOG_ROWS:
            raise Wave7CompareError(
                f"{owner} exceeds the row bound", code="wave7_compare.logging_schema"
            )
    if first != regular.read_bytes():
        raise Wave7CompareError(
            f"{owner} changed while loading", code="wave7_compare.input_drift"
        )
    return tuple(rows)


def _load_run(root: Path, *, role: str) -> RunArtifacts:
    if not root.is_dir() or root.is_symlink():
        raise Wave7CompareError(
            f"{role} run directory is unavailable",
            code="wave7_compare.run_missing",
            context={"path": str(root)},
        )
    run_path = root / "run.json"
    config_path = root / "resolved_config.json"
    logging_path = root / "logging.jsonl"
    run = _strict_json_file(run_path, owner=f"{role} run.json")
    _require_fields(run, RUN_FIELDS, f"{role} run.json")
    config = _strict_json_file(config_path, owner=f"{role} resolved_config.json")
    _require_fields(config, frozenset({"config", "resolution"}), f"{role} config")
    if not isinstance(config["config"], Mapping) or not isinstance(
        config["resolution"], Mapping
    ):
        raise Wave7CompareError(
            f"{role} resolved config is malformed", code="wave7_compare.config_schema"
        )
    _require_fields(
        config["resolution"], RESOLUTION_FIELDS, f"{role} config resolution"
    )
    if run["resolved_config_path"] != "resolved_config.json":
        raise Wave7CompareError(
            f"{role} run selects a noncanonical resolved config",
            code="wave7_compare.config_schema",
        )
    if run["run_dir"] != str(root) or run["runtime"] != {
        "world_size": EXPECTED_WORLD_SIZE
    }:
        raise Wave7CompareError(
            f"{role} run identity is inconsistent",
            code="wave7_compare.run_identity",
        )
    if run["config_fingerprint"] != config["resolution"]["fingerprint"]:
        raise Wave7CompareError(
            f"{role} run/config fingerprint differs",
            code="wave7_compare.config_identity",
        )
    logs = _strict_logging_file(logging_path, owner=f"{role} logging.jsonl")
    return RunArtifacts(
        role=role,
        root=root,
        run=run,
        config=config,
        logs=logs,
        hashes={
            "run_json_sha256": _sha256_file(run_path),
            "resolved_config_sha256": _sha256_file(config_path),
            "logging_jsonl_sha256": _sha256_file(logging_path),
        },
    )


def _safe_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _sha256_bytes(
                value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ),
        }
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, str):
        return value[:1024]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return {"type": type(value).__name__}


def _add_mismatch(
    mismatches: list[dict[str, Any]],
    *,
    code: str,
    scope: str,
    path: str,
    expected: Any,
    observed: Any,
) -> None:
    if len(mismatches) >= MAX_MISMATCH_ROWS:
        raise Wave7CompareError(
            "comparison mismatch count exceeds the complete receipt bound",
            code="wave7_compare.mismatch_bound",
            context={"maximum": MAX_MISMATCH_ROWS},
        )
    mismatches.append(
        {
            "code": code,
            "expected": _safe_json(expected),
            "observed": _safe_json(observed),
            "path": path,
            "scope": scope,
        }
    )


def _error_mismatch(exc: BaseException, *, scope: str) -> dict[str, Any]:
    return {
        "code": str(getattr(exc, "code", "wave7_compare.unexpected"))[:128],
        "expected": None,
        "observed": {
            "context": _safe_json(getattr(exc, "context", {})),
            "message": str(exc)[:2048],
            "type": type(exc).__name__,
        },
        "path": scope,
        "scope": scope,
    }


def _available_digest(value: Any, *, owner: str, length: int = 64) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"status", "value"}
        or value.get("status") != "available"
        or not isinstance(value.get("value"), str)
        or len(value["value"]) != length
        or any(character not in HEX for character in value["value"])
    ):
        raise Wave7CompareError(
            f"{owner} is not available", code="wave7_compare.provenance"
        )
    return str(value["value"])


def _compare_run_contracts(
    runs: Mapping[str, RunArtifacts],
    request: CompareRequest,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = runs["uninterrupted"]
    parent = runs["interrupted_parent"]
    child = runs["resume_child"]
    expected_status = {
        "uninterrupted": ("completed", 5),
        "interrupted_parent": ("not_completed", 3),
        "resume_child": ("completed", 5),
    }
    commits: dict[str, str] = {}
    provenance_digests: dict[str, str] = {}
    for role, artifacts in runs.items():
        status, completed = expected_status[role]
        observed_status = artifacts.run["status"]
        status_ok = (
            observed_status == "completed"
            if status == "completed"
            else observed_status != "completed"
        )
        if not status_ok or artifacts.run["completed_steps"] != completed:
            _add_mismatch(
                mismatches,
                code="wave7_compare.run_status",
                scope=role,
                path="run.status_and_completed_steps",
                expected={"status": status, "completed_steps": completed},
                observed={
                    "status": observed_status,
                    "completed_steps": artifacts.run["completed_steps"],
                },
            )
        provenance = artifacts.run["provenance"]
        if not isinstance(provenance, Mapping) or not isinstance(
            provenance.get("repository"), Mapping
        ):
            raise Wave7CompareError(
                f"{role} provenance is malformed", code="wave7_compare.provenance"
            )
        repository = provenance["repository"]
        commits[role] = _available_digest(
            repository.get("commit"), owner=f"{role} commit", length=40
        )
        provenance_digests[role] = _available_digest(
            repository.get("execution_relevant_digest"),
            owner=f"{role} execution digest",
        )
    if len(set(commits.values())) != 1:
        _add_mismatch(
            mismatches,
            code="wave7_compare.commit_mismatch",
            scope="runs",
            path="provenance.repository.commit",
            expected=commits["uninterrupted"],
            observed=commits,
        )
    if set(provenance_digests.values()) != {request.expected_provenance_sha256}:
        _add_mismatch(
            mismatches,
            code="wave7_compare.provenance_mismatch",
            scope="runs",
            path="provenance.repository.execution_relevant_digest",
            expected=request.expected_provenance_sha256,
            observed=provenance_digests,
        )
    run_ids = {role: artifacts.run["run_id"] for role, artifacts in runs.items()}
    segment_ids = {
        role: artifacts.run["continuation"]["segment_id"]
        for role, artifacts in runs.items()
    }
    if len(set(run_ids.values())) != 3 or len(set(segment_ids.values())) != 3:
        _add_mismatch(
            mismatches,
            code="wave7_compare.run_identity_mismatch",
            scope="runs",
            path="run_id_and_segment_id",
            expected="three distinct run IDs and segment IDs",
            observed={"run_ids": run_ids, "segment_ids": segment_ids},
        )
    count_contract = {
        "uninterrupted": {
            "completed_steps": reference.run["completed_steps"],
            "consumed_packs": reference.run["consumed_packs"],
            "checkpoint_event_count": reference.run["checkpoint_event_count"],
        },
        "interrupted_parent": {
            "completed_steps": parent.run["completed_steps"],
            "consumed_packs": parent.run["consumed_packs"],
            "checkpoint_event_count": parent.run["checkpoint_event_count"],
        },
        "resume_child": {
            "completed_steps": child.run["completed_steps"],
            "consumed_packs": child.run["consumed_packs"],
            "checkpoint_event_count": child.run["checkpoint_event_count"],
        },
    }
    counts_match = (
        count_contract["uninterrupted"]["completed_steps"]
        == count_contract["resume_child"]["completed_steps"]
        == 5
        and count_contract["interrupted_parent"]["completed_steps"] == 3
        and count_contract["uninterrupted"]["consumed_packs"]
        == count_contract["resume_child"]["consumed_packs"]
        and count_contract["interrupted_parent"]["consumed_packs"]
        < count_contract["resume_child"]["consumed_packs"]
        and count_contract["uninterrupted"]["checkpoint_event_count"]
        == count_contract["interrupted_parent"]["checkpoint_event_count"]
        + count_contract["resume_child"]["checkpoint_event_count"]
    )
    if not counts_match:
        _add_mismatch(
            mismatches,
            code="wave7_compare.run_count_mismatch",
            scope="runs",
            path="completed_steps_consumed_packs_checkpoint_events",
            expected=(
                "reference and child totals equal; parent stops at step 3; "
                "parent plus child checkpoint events equal reference"
            ),
            observed=count_contract,
        )
    policy_reference = reference.run["policy_identities"]
    dependency_reference = reference.run["provenance"]["dependencies"]
    for role, artifacts in runs.items():
        if artifacts.run["policy_identities"] != policy_reference:
            _add_mismatch(
                mismatches,
                code="wave7_compare.policy_mismatch",
                scope=role,
                path="run.policy_identities",
                expected=policy_reference,
                observed=artifacts.run["policy_identities"],
            )
        if artifacts.run["provenance"]["dependencies"] != dependency_reference:
            _add_mismatch(
                mismatches,
                code="wave7_compare.dependency_mismatch",
                scope=role,
                path="run.provenance.dependencies",
                expected=dependency_reference,
                observed=artifacts.run["provenance"]["dependencies"],
            )
    projections = {
        role: dict(build_resume_compatibility_projection(artifacts.config))
        for role, artifacts in runs.items()
    }
    if (
        len(
            {
                _sha256_bytes(canonical_json_bytes(value))
                for value in projections.values()
            }
        )
        != 1
    ):
        _add_mismatch(
            mismatches,
            code="wave7_compare.resume_compatibility_mismatch",
            scope="configs",
            path="resume_compatibility",
            expected=projections["uninterrupted"],
            observed=projections,
        )
    full_hashes = {
        role: _sha256_bytes(canonical_json_bytes(artifacts.config))
        for role, artifacts in runs.items()
    }
    if len(set(full_hashes.values())) != 3:
        _add_mismatch(
            mismatches,
            code="wave7_compare.full_config_identity",
            scope="configs",
            path="resolved_config",
            expected="three distinct full run/config identities",
            observed=full_hashes,
        )
    reference_resume = reference.config["config"].get("resume")
    parent_resume = parent.config["config"].get("resume")
    child_resume = child.config["config"].get("resume")
    expected_child_checkpoint = str(parent.root / "checkpoints/step-3")
    if (
        reference_resume != {"checkpoint_dir": None, "mode": "exact_same_world_size"}
        or parent_resume != reference_resume
        or child_resume
        != {
            "checkpoint_dir": expected_child_checkpoint,
            "mode": "exact_same_world_size",
        }
    ):
        _add_mismatch(
            mismatches,
            code="wave7_compare.resume_selection",
            scope="configs",
            path="config.resume",
            expected={
                "reference_and_parent": reference_resume,
                "child_checkpoint_dir": expected_child_checkpoint,
            },
            observed={
                "reference": reference_resume,
                "parent": parent_resume,
                "child": child_resume,
            },
        )
    return {
        "commit": commits,
        "execution_relevant_digest": provenance_digests,
        "full_resolved_config_sha256": full_hashes,
        "run_counts": count_contract,
        "run_ids": run_ids,
        "segment_ids": segment_ids,
        "resume_compatibility_sha256": _sha256_bytes(
            canonical_json_bytes(projections["uninterrupted"])
        ),
    }


def _validate_identity_inventory(value: Any, *, owner: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > 4_096:
        raise Wave7CompareError(
            f"{owner} must be one nonempty bounded identity inventory",
            code="wave7_compare.interrupt_inventory",
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "size"}:
            raise Wave7CompareError(
                f"{owner} row has an invalid schema",
                code="wave7_compare.interrupt_inventory",
                context={"index": index},
            )
        path_value = raw["path"]
        digest = raw["sha256"]
        size = raw["size"]
        if (
            not isinstance(path_value, str)
            or not Path(path_value).is_absolute()
            or path_value in seen
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in HEX for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise Wave7CompareError(
                f"{owner} row is invalid",
                code="wave7_compare.interrupt_inventory",
                context={"index": index},
            )
        path = _regular_file(Path(path_value), owner=f"{owner} row {index}")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise Wave7CompareError(
                f"{owner} no longer matches its pinned file",
                code="wave7_compare.interrupt_inventory",
                context={"index": index, "path": path_value},
            )
        seen.add(path_value)
        rows.append(dict(raw))
    return rows


def _validate_interrupt_bindings(
    marker: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    request: CompareRequest,
    parent: RunArtifacts,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    source_hashes = _validate_identity_inventory(
        marker.get("source_hashes"), owner="interrupt source_hashes"
    )
    config_hashes = _validate_identity_inventory(
        marker.get("config_hashes"), owner="interrupt config_hashes"
    )
    if receipt.get("source_hashes") != source_hashes:
        _add_mismatch(
            mismatches,
            code="wave7_compare.interrupt_inventory",
            scope="interruption",
            path="source_hashes",
            expected=source_hashes,
            observed=receipt.get("source_hashes"),
        )
    if receipt.get("config_hashes") != config_hashes:
        _add_mismatch(
            mismatches,
            code="wave7_compare.interrupt_inventory",
            scope="interruption",
            path="config_hashes",
            expected=config_hashes,
            observed=receipt.get("config_hashes"),
        )
    script_path = AUDITED_INTERRUPT_PATH.resolve()
    expected_script = {
        "path": str(script_path),
        "sha256": request.expected_interrupt_source_sha256,
    }
    if (
        marker.get("script") != expected_script
        or receipt.get("script") != expected_script
    ):
        _add_mismatch(
            mismatches,
            code="wave7_compare.interrupt_inventory",
            scope="interruption",
            path="interrupt_script",
            expected=expected_script,
            observed={"marker": marker.get("script"), "receipt": receipt.get("script")},
        )
    required_source_rows = {
        str(Path(__file__).resolve()): request.expected_source_sha256,
        str(script_path): expected_script["sha256"],
    }
    observed_sources = {row["path"]: row["sha256"] for row in source_hashes}
    if any(
        observed_sources.get(path) != digest
        for path, digest in required_source_rows.items()
    ):
        _add_mismatch(
            mismatches,
            code="wave7_compare.interrupt_inventory",
            scope="interruption",
            path="required_broad_source_hashes",
            expected=required_source_rows,
            observed=observed_sources,
        )
    expected_target = {
        "artifact_root": parent.run["artifact_root"],
        "launcher_config": marker.get("target_binding", {}).get("launcher_config")
        if isinstance(marker.get("target_binding"), Mapping)
        else None,
        "resolved_config_fingerprint": parent.run["config_fingerprint"],
        "resolved_config_sources": parent.config["resolution"]["sources"],
        "run_dir": str(parent.root),
        "run_name": parent.run["run_name"],
    }
    marker_target = marker.get("target_binding")
    receipt_target = receipt.get("target_binding")
    if marker_target != expected_target or receipt_target != expected_target:
        _add_mismatch(
            mismatches,
            code="wave7_compare.interruption_binding",
            scope="interruption",
            path="target_binding",
            expected=expected_target,
            observed={"marker": marker_target, "receipt": receipt_target},
        )
    config_by_path = {row["path"]: row["sha256"] for row in config_hashes}
    resolved_sources = parent.config["resolution"]["sources"]
    if not isinstance(resolved_sources, list) or any(
        not isinstance(row, Mapping)
        or set(row) != {"path", "sha256"}
        or config_by_path.get(row["path"]) != row["sha256"]
        for row in resolved_sources
    ):
        _add_mismatch(
            mismatches,
            code="wave7_compare.interruption_binding",
            scope="interruption",
            path="resolved_config_sources",
            expected=resolved_sources,
            observed=config_by_path,
        )
    if expected_target["launcher_config"] not in config_by_path:
        _add_mismatch(
            mismatches,
            code="wave7_compare.interruption_binding",
            scope="interruption",
            path="launcher_config",
            expected="launcher config present in pinned config inventory",
            observed=expected_target["launcher_config"],
        )
    resolved_source_paths = {
        row["path"] for row in resolved_sources if isinstance(row, Mapping)
    }
    if expected_target["launcher_config"] not in resolved_source_paths:
        _add_mismatch(
            mismatches,
            code="wave7_compare.interruption_binding",
            scope="interruption",
            path="launcher_config_resolved_source",
            expected=sorted(resolved_source_paths),
            observed=expected_target["launcher_config"],
        )
    if marker.get("marker_target") != str(request.interruption_marker) or marker.get(
        "receipt_target"
    ) != str(request.termination_receipt):
        _add_mismatch(
            mismatches,
            code="wave7_compare.interruption_binding",
            scope="interruption",
            path="artifact_targets",
            expected={
                "marker": str(request.interruption_marker),
                "receipt": str(request.termination_receipt),
            },
            observed={
                "marker": marker.get("marker_target"),
                "receipt": marker.get("receipt_target"),
            },
        )
    marker_launcher = marker.get("launcher")
    receipt_launch = receipt.get("launch")
    launcher_argv = (
        marker_launcher.get("argv") if isinstance(marker_launcher, Mapping) else None
    )
    expected_launcher_sha256 = (
        _sha256_bytes(canonical_json_bytes(launcher_argv))
        if isinstance(launcher_argv, list)
        else None
    )
    launcher_bound = (
        isinstance(marker_launcher, Mapping)
        and set(marker_launcher) == {"argv", "argv_sha256"}
        and isinstance(receipt_launch, Mapping)
        and marker_launcher.get("argv_sha256") == expected_launcher_sha256
        and receipt_launch.get("argv") == launcher_argv
        and receipt_launch.get("argv_sha256") == expected_launcher_sha256
        and marker.get("launch")
        == {"pgid": receipt_launch.get("pgid"), "pid": receipt_launch.get("pid")}
    )
    if not launcher_bound:
        _add_mismatch(
            mismatches,
            code="wave7_compare.interruption_binding",
            scope="interruption",
            path="launcher_argv_and_process_identity",
            expected={
                "argv_sha256": expected_launcher_sha256,
                "marker_launch": marker.get("launch"),
            },
            observed={"launcher": marker_launcher, "receipt_launch": receipt_launch},
        )
    return {
        "config_hashes": config_hashes,
        "interrupt_script": expected_script,
        "source_hashes": source_hashes,
        "target_binding": expected_target,
    }


def _validate_interrupt_publication_binding(
    marker: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    parent: RunArtifacts,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    marker_event = marker.get("checkpoint_publication_event")
    marker_run = marker.get("checkpoint_publication_run")
    receipt_publication = receipt.get("checkpoint_publication")
    if not isinstance(marker_event, Mapping):
        raise Wave7CompareError(
            "interruption marker publication event is missing",
            code="wave7_compare.interruption_schema",
        )
    if not isinstance(marker_run, Mapping):
        raise Wave7CompareError(
            "interruption marker publication run identity is missing",
            code="wave7_compare.interruption_schema",
        )
    if not isinstance(receipt_publication, Mapping):
        raise Wave7CompareError(
            "termination receipt publication binding is missing",
            code="wave7_compare.interruption_schema",
        )
    _require_fields(
        marker_event,
        CHECKPOINT_PUBLICATION_EVENT_FIELDS,
        "interruption checkpoint publication event",
    )
    _require_fields(
        marker_run,
        CHECKPOINT_PUBLICATION_RUN_FIELDS,
        "interruption checkpoint publication run",
    )
    _require_fields(
        receipt_publication,
        CHECKPOINT_PUBLICATION_RECEIPT_FIELDS,
        "termination checkpoint publication binding",
    )
    receipt_event = receipt_publication.get("event")
    if not isinstance(receipt_event, Mapping):
        raise Wave7CompareError(
            "termination receipt publication event is missing",
            code="wave7_compare.interruption_schema",
        )
    _require_fields(
        receipt_event,
        CHECKPOINT_PUBLICATION_EVENT_FIELDS,
        "termination checkpoint publication event",
    )
    measurement = parent.run.get("measurement")
    parent_events = (
        measurement.get("checkpoint_publication_events")
        if isinstance(measurement, Mapping)
        else None
    )
    matching_events = (
        [
            dict(event)
            for event in parent_events
            if isinstance(event, Mapping)
            and event.get("step") == 3
            and not isinstance(event.get("step"), bool)
        ]
        if isinstance(parent_events, list)
        else []
    )
    run_path = parent.root / "run.json"
    run_size = run_path.stat().st_size
    run_sha256 = _sha256_file(run_path)
    expected_event = matching_events[0] if len(matching_events) == 1 else None
    expected_marker_run = {
        "file_sha256": run_sha256,
        "path": str(run_path),
        "size": run_size,
    }
    expected_receipt_publication = {
        "event": expected_event,
        "run_file_sha256": run_sha256,
        "run_path": str(run_path),
        "run_size": run_size,
    }
    if (
        dict(marker_event) != expected_event
        or dict(marker_run) != expected_marker_run
        or dict(receipt_publication) != expected_receipt_publication
    ):
        _add_mismatch(
            mismatches,
            code="wave7_compare.interruption_binding",
            scope="interruption",
            path="checkpoint_publication",
            expected={
                "event": expected_event,
                "marker_run": expected_marker_run,
                "receipt": expected_receipt_publication,
            },
            observed={
                "event": marker_event,
                "marker_run": marker_run,
                "receipt": receipt_publication,
            },
        )
    return expected_receipt_publication


def _load_interrupt_evidence(
    request: CompareRequest,
    parent: RunArtifacts,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    marker = _strict_json_file(request.interruption_marker, owner="interruption marker")
    receipt = _strict_json_file(
        request.termination_receipt, owner="termination receipt"
    )
    _require_fields(marker, MARKER_FIELDS, "interruption marker")
    _require_fields(receipt, TERMINATION_RECEIPT_FIELDS, "termination receipt")
    bindings = _validate_interrupt_bindings(
        marker,
        receipt,
        request=request,
        parent=parent,
        mismatches=mismatches,
    )
    if marker.get("schema") != AUDITED_INTERRUPT_MARKER_SCHEMA:
        raise Wave7CompareError(
            "interruption marker schema is unsupported",
            code="wave7_compare.interruption_schema",
        )
    if receipt.get("schema") != AUDITED_INTERRUPT_RECEIPT_SCHEMA:
        raise Wave7CompareError(
            "termination receipt schema is unsupported",
            code="wave7_compare.interruption_schema",
        )
    publication_binding = _validate_interrupt_publication_binding(
        marker,
        receipt,
        parent=parent,
        mismatches=mismatches,
    )
    postconditions = receipt.get("postconditions")
    termination = receipt.get("termination")
    marker_observation = receipt.get("marker")
    if (
        not isinstance(postconditions, Mapping)
        or not isinstance(termination, Mapping)
        or not isinstance(marker_observation, Mapping)
    ):
        raise Wave7CompareError(
            "termination receipt is incomplete",
            code="wave7_compare.interruption_schema",
        )
    _require_fields(postconditions, POSTCONDITION_FIELDS, "termination postconditions")
    _require_fields(termination, TERMINATION_FIELDS, "termination details")
    _require_fields(
        marker_observation, MARKER_OBSERVATION_FIELDS, "termination marker observation"
    )
    marker_sha256 = _sha256_file(request.interruption_marker)
    expected_marker_observation = {
        "expected_file_sha256": marker_sha256,
        "file_sha256": marker_sha256,
        "final_file_sha256": marker_sha256,
        "path": str(request.interruption_marker),
        "published": True,
        "strict_payload_equal": True,
        "unchanged": True,
    }
    expected_gate = (
        receipt.get("status") == "passed"
        and receipt.get("errors") == []
        and marker_observation == expected_marker_observation
        and postconditions.get("late_write_detected") is False
        and postconditions.get("max_logged_train_step") == 3
        and postconditions.get("step_5_absent") is True
        and postconditions.get("final_json_absent") is True
        and postconditions.get("manifest_unchanged") is True
        and postconditions.get("checkpoint_publication_event_unchanged") is True
        and postconditions.get("checkpoint_publication_run_final_file_sha256")
        == publication_binding["run_file_sha256"]
        and postconditions.get("run_not_completed") is True
        and postconditions.get("nvidia_compute_apps") == []
        and postconditions.get("nvidia_compute_apps_added") == []
        and termination.get("cleanup_errors") == []
        and termination.get("launcher_exited") is True
        and termination.get("remaining_pids") == []
        and termination.get("remaining_pgids") == []
    )
    if not expected_gate:
        _add_mismatch(
            mismatches,
            code="wave7_compare.interruption_gate",
            scope="interruption",
            path="termination_receipt",
            expected="passed bounded termination with immutable step-3",
            observed={
                "receipt": receipt.get("status"),
                "postconditions": postconditions,
            },
        )
    if marker.get("checkpoint") != receipt.get("checkpoint"):
        _add_mismatch(
            mismatches,
            code="wave7_compare.interruption_binding",
            scope="interruption",
            path="checkpoint",
            expected=marker.get("checkpoint"),
            observed=receipt.get("checkpoint"),
        )
    if receipt.get("request") != {
        "expected_checkpoint_step": 3,
        "parent_run_dir": str(parent.root),
        "timeout_seconds": receipt.get("request", {}).get("timeout_seconds")
        if isinstance(receipt.get("request"), Mapping)
        else None,
    }:
        _add_mismatch(
            mismatches,
            code="wave7_compare.interruption_binding",
            scope="interruption",
            path="request.parent_run_dir",
            expected=str(parent.root),
            observed=receipt.get("request"),
        )
    current_tree = interrupt._snapshot_run_tree(parent.root)
    expected_tree = postconditions["stability_after"]
    if not isinstance(expected_tree, Mapping) or (
        current_tree["entry_count"] != expected_tree.get("entry_count")
        or current_tree["fingerprint"] != expected_tree.get("fingerprint")
    ):
        _add_mismatch(
            mismatches,
            code="wave7_compare.parent_tree_changed",
            scope="interrupted_parent",
            path="run_tree",
            expected=expected_tree,
            observed={
                "entry_count": current_tree["entry_count"],
                "fingerprint": current_tree["fingerprint"],
            },
        )
    return {
        "marker_file_sha256": marker_sha256,
        "termination_receipt_file_sha256": _sha256_file(request.termination_receipt),
        "checkpoint": receipt.get("checkpoint"),
        "checkpoint_publication": publication_binding,
        "identity_bindings": bindings,
        "parent_tree": {
            "entry_count": current_tree["entry_count"],
            "fingerprint": current_tree["fingerprint"],
        },
    }


def _manifest_expectations(
    manifest: TrainingStateManifest,
) -> TrainingStateExpectations:
    return TrainingStateExpectations(
        checkpoint_step=manifest.checkpoint_step,
        world_size=manifest.world_size,
        identities=manifest.identities,
        scheduler_applicable=manifest.scheduler_applicable,
        scaler_applicable=manifest.scaler_applicable,
        rng_kinds=REQUIRED_RNG_KINDS,
    )


def _checkpoint_inventory(checkpoint_dir: Path) -> dict[str, Any]:
    _reject_symlink_components(checkpoint_dir, owner="checkpoint")
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise Wave7CompareError(
            "checkpoint directory is missing",
            code="wave7_compare.pruning",
            context={"path": str(checkpoint_dir)},
        )
    total = 0
    training_state = 0
    count = 0
    for path in sorted(checkpoint_dir.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise Wave7CompareError(
                "checkpoint contains a non-regular entry",
                code="wave7_compare.unsafe_path",
                context={"path": str(path)},
            )
        size = path.stat().st_size
        total += size
        count += 1
        if path.is_relative_to(checkpoint_dir / "training_state"):
            training_state += size
    return {
        "checkpoint_total_bytes": total,
        "file_count": count,
        "inference_payload_bytes": total - training_state,
        "training_state_bytes": training_state,
    }


def _validate_inference_payload(checkpoint_dir: Path) -> dict[str, Any]:
    adapter = checkpoint_dir / "adapter"
    _validate_adapter_payload(adapter)
    adapter_config = _regular_file(
        adapter / "adapter_config.json", owner="adapter config"
    )
    adapter_weights = _regular_file(
        adapter / "adapter_model.safetensors", owner="adapter weights"
    )
    special = inspect_special_token_embedding_delta_payload(
        checkpoint_dir / "special_token_embeddings"
    )
    return {
        "adapter": {
            "config_sha256": _sha256_file(adapter_config),
            "weights_sha256": _sha256_file(adapter_weights),
            "status": "bounded_structural_validated_and_read",
        },
        "special_token_embeddings": {
            "fingerprint": special["fingerprint"],
            "file_count": special["file_count"],
            "status": "bounded_structural_validated_and_read",
        },
    }


def _load_manifest_checked(checkpoint_dir: Path, *, step: int) -> TrainingStateManifest:
    manifest = load_training_state_manifest(checkpoint_dir)
    if (
        manifest.checkpoint_step != step
        or manifest.world_size != EXPECTED_WORLD_SIZE
        or tuple(rank.rank for rank in manifest.ranks)
        != tuple(range(EXPECTED_WORLD_SIZE))
    ):
        raise Wave7CompareError(
            "training-state manifest identifies the wrong boundary or rank set",
            code="wave7_compare.manifest_identity",
            context={
                "path": str(checkpoint_dir),
                "step": manifest.checkpoint_step,
                "world_size": manifest.world_size,
                "ranks": [rank.rank for rank in manifest.ranks],
            },
        )
    return manifest


def _admit_rank(
    checkpoint_dir: Path,
    manifest: TrainingStateManifest,
    rank: int,
) -> AdmittedTrainingState:
    admitted = admit_training_state(
        checkpoint_dir,
        _manifest_expectations(manifest),
        current_rank=rank,
    )
    if not isinstance(admitted, AdmittedTrainingState):
        raise Wave7CompareError(
            "training-state admission returned an unexpected result",
            code="wave7_compare.training_state",
        )
    return admitted


def _exact_state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_exact_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)):
        return (
            isinstance(right, Sequence)
            and not isinstance(right, (str, bytes, bytearray))
            and len(left) == len(right)
            and all(_exact_state_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    if hasattr(result, "all"):
        return bool(result.all())
    return False


def _numeric_classification(owner: str) -> tuple[str, float, float]:
    if owner in {"scheduler", "scaler"}:
        return "fp32_same_forward_or_control_state", 1e-4, 1e-5
    return "bf16_derived_or_model_update", 5e-3, 5e-3


def _compare_runtime_state(
    left: Any,
    right: Any,
    *,
    owner: str,
    path: str,
    scope: str,
    mismatches: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            _add_mismatch(
                mismatches,
                code="wave7_compare.tensor_type",
                scope=scope,
                path=path,
                expected=left,
                observed=right,
            )
            return
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            _add_mismatch(
                mismatches,
                code="wave7_compare.tensor_structure",
                scope=scope,
                path=path,
                expected={"dtype": str(left.dtype), "shape": list(left.shape)},
                observed={"dtype": str(right.dtype), "shape": list(right.shape)},
            )
            return
        if not left.dtype.is_floating_point:
            if not torch.equal(left, right):
                _add_mismatch(
                    mismatches,
                    code="wave7_compare.nonfloat_state_drift",
                    scope=scope,
                    path=path,
                    expected=left,
                    observed=right,
                )
            return
        classification, rtol, atol = _numeric_classification(owner)
        left_fp32 = left.detach().cpu().to(torch.float32)
        right_fp32 = right.detach().cpu().to(torch.float32)
        finite = bool(
            torch.isfinite(left_fp32).all() and torch.isfinite(right_fp32).all()
        )
        if finite and left_fp32.numel():
            delta = (left_fp32 - right_fp32).abs()
            max_abs = float(delta.max().item())
            denominator = left_fp32.abs().clamp_min(1e-30)
            max_rel = float((delta / denominator).max().item())
            passed = bool(torch.allclose(left_fp32, right_fp32, rtol=rtol, atol=atol))
        else:
            max_abs = math.inf
            max_rel = math.inf
            passed = False
        bucket = summary.setdefault(
            classification,
            {
                "atol": atol,
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
                "rtol": rtol,
                "tensor_count": 0,
                "worst_abs_path": None,
                "worst_rel_path": None,
            },
        )
        bucket["tensor_count"] += 1
        if max_abs >= bucket["max_abs_diff"]:
            bucket["max_abs_diff"] = max_abs
            bucket["worst_abs_path"] = path
        if max_rel >= bucket["max_rel_diff"]:
            bucket["max_rel_diff"] = max_rel
            bucket["worst_rel_path"] = path
        if not passed:
            _add_mismatch(
                mismatches,
                code="wave7_compare.tensor_drift",
                scope=scope,
                path=path,
                expected={"classification": classification, "rtol": rtol, "atol": atol},
                observed={"max_abs_diff": max_abs, "max_rel_diff": max_rel},
            )
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            _add_mismatch(
                mismatches,
                code="wave7_compare.state_type",
                scope=scope,
                path=path,
                expected=type(left).__name__,
                observed=type(right).__name__,
            )
            return
        if set(left) != set(right):
            _add_mismatch(
                mismatches,
                code="wave7_compare.state_fields",
                scope=scope,
                path=path,
                expected=sorted(str(key) for key in left),
                observed=sorted(str(key) for key in right),
            )
            return
        for key in sorted(left, key=str):
            _compare_runtime_state(
                left[key],
                right[key],
                owner=owner,
                path=f"{path}.{key}",
                scope=scope,
                mismatches=mismatches,
                summary=summary,
            )
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)):
        if not isinstance(right, Sequence) or isinstance(
            right, (str, bytes, bytearray)
        ):
            _add_mismatch(
                mismatches,
                code="wave7_compare.state_type",
                scope=scope,
                path=path,
                expected=type(left).__name__,
                observed=type(right).__name__,
            )
            return
        if len(left) != len(right):
            _add_mismatch(
                mismatches,
                code="wave7_compare.state_length",
                scope=scope,
                path=path,
                expected=len(left),
                observed=len(right),
            )
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare_runtime_state(
                left_item,
                right_item,
                owner=owner,
                path=f"{path}[{index}]",
                scope=scope,
                mismatches=mismatches,
                summary=summary,
            )
        return
    if isinstance(left, float) or isinstance(right, float):
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            passed = False
            max_abs = math.inf
        else:
            classification, rtol, atol = _numeric_classification(owner)
            max_abs = abs(float(left) - float(right))
            passed = math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)
            bucket = summary.setdefault(
                classification,
                {
                    "atol": atol,
                    "max_abs_diff": 0.0,
                    "max_rel_diff": 0.0,
                    "rtol": rtol,
                    "tensor_count": 0,
                    "worst_abs_path": None,
                    "worst_rel_path": None,
                },
            )
            if max_abs >= bucket["max_abs_diff"]:
                bucket["max_abs_diff"] = max_abs
                bucket["worst_abs_path"] = path
        if not passed:
            _add_mismatch(
                mismatches,
                code="wave7_compare.float_state_drift",
                scope=scope,
                path=path,
                expected=left,
                observed=right,
            )
        return
    if left != right:
        _add_mismatch(
            mismatches,
            code="wave7_compare.nonfloat_state_drift",
            scope=scope,
            path=path,
            expected=left,
            observed=right,
        )


def _cursor_projection(decoded: DecodedRankTrainingState) -> dict[str, Any]:
    cursor = decoded.cursor
    data = cursor["data"]["state"]
    pack = cursor["pack"]["state"]
    next_pack = pack.get("next_pack")
    return {
        "next_rank_local_micro_step": cursor["next_rank_local_micro_step"],
        "next_pack": next_pack,
        "next_example_ids": (
            None if next_pack is None else list(next_pack.get("example_ids", []))
        ),
        "runtime_counters": dict(data.get("runtime_counters", {})),
    }


def _compare_decoded_rank(
    left: DecodedRankTrainingState,
    right: DecodedRankTrainingState,
    *,
    scope: str,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    if left.signature != right.signature or dict(left.structure) != dict(
        right.structure
    ):
        _add_mismatch(
            mismatches,
            code="wave7_compare.structure_signature",
            scope=scope,
            path="structure_signature",
            expected={"signature": left.signature, "structure": left.structure},
            observed={"signature": right.signature, "structure": right.structure},
        )
    if dict(left.cursor) != dict(right.cursor):
        _add_mismatch(
            mismatches,
            code="wave7_compare.cursor_drift",
            scope=scope,
            path="cursor",
            expected=left.cursor,
            observed=right.cursor,
        )
    rng_pairs = {
        "python": (left.python_rng_state, right.python_rng_state),
        "numpy": (left.numpy_rng_state, right.numpy_rng_state),
        "torch_cpu": (left.torch_cpu_rng_state, right.torch_cpu_rng_state),
        "torch_cuda": (left.torch_cuda_rng_states, right.torch_cuda_rng_states),
    }
    for name, (expected, observed) in rng_pairs.items():
        if not _exact_state_equal(expected, observed):
            _add_mismatch(
                mismatches,
                code="wave7_compare.rng_drift",
                scope=scope,
                path=f"rng.{name}",
                expected=expected,
                observed=observed,
            )
    numeric_summary: dict[str, Any] = {}
    for owner in ("trainable_model", "optimizer", "scheduler", "scaler"):
        _compare_runtime_state(
            getattr(left, owner),
            getattr(right, owner),
            owner=owner,
            path=owner,
            scope=scope,
            mismatches=mismatches,
            summary=numeric_summary,
        )
    return {
        "cursor": _cursor_projection(left),
        "numeric_summary": numeric_summary,
        "rng_exact": all(
            _exact_state_equal(expected, observed)
            for expected, observed in rng_pairs.values()
        ),
        "signature": left.signature,
    }


def _compare_checkpoint_pair(
    left_dir: Path,
    right_dir: Path,
    *,
    step: int,
    scope: str,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    before = len(mismatches)
    left_manifest = _load_manifest_checked(left_dir, step=step)
    right_manifest = _load_manifest_checked(right_dir, step=step)
    for key in STRICT_IDENTITY_KEYS:
        if left_manifest.identities[key] != right_manifest.identities[key]:
            _add_mismatch(
                mismatches,
                code="wave7_compare.manifest_identity_mismatch",
                scope=scope,
                path=f"identities.{key}",
                expected=left_manifest.identities[key],
                observed=right_manifest.identities[key],
            )
    rank_receipts: list[dict[str, Any]] = []
    for rank in range(EXPECTED_WORLD_SIZE):
        left = _admit_rank(left_dir, left_manifest, rank).decoded_rank
        right = _admit_rank(right_dir, right_manifest, rank).decoded_rank
        rank_receipts.append(
            {
                "rank": rank,
                **_compare_decoded_rank(
                    left,
                    right,
                    scope=f"{scope}.rank-{rank:05d}",
                    mismatches=mismatches,
                ),
            }
        )
    return {
        "left_manifest_aggregate_digest": left_manifest.aggregate_digest,
        "rank_comparisons": rank_receipts,
        "right_manifest_aggregate_digest": right_manifest.aggregate_digest,
        "status": "passed" if len(mismatches) == before else "failed",
    }


def _load_checkpoint_surfaces(
    runs: Mapping[str, RunArtifacts],
    mismatches: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, TrainingStateManifest]]:
    selections = {
        "reference_step_3": (runs["uninterrupted"].root / "checkpoints/step-3", 3),
        "reference_step_5": (runs["uninterrupted"].root / "checkpoints/step-5", 5),
        "parent_step_3": (runs["interrupted_parent"].root / "checkpoints/step-3", 3),
        "child_step_5": (runs["resume_child"].root / "checkpoints/step-5", 5),
    }
    surfaces: dict[str, Any] = {}
    manifests: dict[str, TrainingStateManifest] = {}
    for name, (checkpoint_dir, step) in selections.items():
        manifest = _load_manifest_checked(checkpoint_dir, step=step)
        manifests[name] = manifest
        surfaces[name] = {
            "inference": _validate_inference_payload(checkpoint_dir),
            "manifest_aggregate_digest": manifest.aggregate_digest,
            "manifest_file_sha256": _sha256_file(
                checkpoint_dir / "training_state/manifest.json"
            ),
            "path": str(checkpoint_dir),
            "sizes": _checkpoint_inventory(checkpoint_dir),
        }
    reference_ids = manifests["reference_step_3"].identities
    for name, manifest in manifests.items():
        for key in STRICT_IDENTITY_KEYS:
            if manifest.identities[key] != reference_ids[key]:
                _add_mismatch(
                    mismatches,
                    code="wave7_compare.manifest_identity_mismatch",
                    scope=name,
                    path=f"identities.{key}",
                    expected=reference_ids[key],
                    observed=manifest.identities[key],
                )
    return surfaces, manifests


def _validate_lineage(
    runs: Mapping[str, RunArtifacts],
    manifests: Mapping[str, TrainingStateManifest],
    checkpoint_surfaces: Mapping[str, Any],
    interruption: Mapping[str, Any],
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = runs["uninterrupted"]
    parent = runs["interrupted_parent"]
    child = runs["resume_child"]
    parent_surface = checkpoint_surfaces["parent_step_3"]
    parent_identity = {
        "checkpoint_step": 3,
        "resolved_path": str(parent.root / "checkpoints/step-3"),
        "training_state_aggregate_digest": parent_surface["manifest_aggregate_digest"],
        "training_state_manifest_file_sha256": parent_surface["manifest_file_sha256"],
    }
    expected_child = {
        "continuation_index": 1,
        "parent": {
            "checkpoint_identity": parent_identity,
            "continuation_index": 0,
            "run_id": parent.run["run_id"],
            "segment_id": parent.run["continuation"]["segment_id"],
        },
        "schema_version": 1,
        "segment_id": child.run["continuation"]["segment_id"],
    }
    if (
        reference.run["continuation"]["continuation_index"] != 0
        or parent.run["continuation"]["continuation_index"] != 0
    ):
        _add_mismatch(
            mismatches,
            code="wave7_compare.lineage_mismatch",
            scope="lineage",
            path="reference_parent_continuation_index",
            expected=0,
            observed={
                "reference": reference.run["continuation"],
                "parent": parent.run["continuation"],
            },
        )
    if child.run["continuation"] != expected_child:
        _add_mismatch(
            mismatches,
            code="wave7_compare.lineage_mismatch",
            scope="lineage",
            path="resume_child.continuation",
            expected=expected_child,
            observed=child.run["continuation"],
        )
    interruption_checkpoint = interruption.get("checkpoint")
    expected_interrupt = {
        "aggregate_digest": parent_identity["training_state_aggregate_digest"],
        "checkpoint_step": 3,
        "file_sha256": parent_identity["training_state_manifest_file_sha256"],
        "path": str(parent.root / "checkpoints/step-3/training_state/manifest.json"),
        "rank_count": EXPECTED_WORLD_SIZE,
        "size": (parent.root / "checkpoints/step-3/training_state/manifest.json")
        .stat()
        .st_size,
        "world_size": EXPECTED_WORLD_SIZE,
    }
    if interruption_checkpoint != expected_interrupt:
        _add_mismatch(
            mismatches,
            code="wave7_compare.lineage_mismatch",
            scope="lineage",
            path="interruption.checkpoint",
            expected=expected_interrupt,
            observed=interruption_checkpoint,
        )
    manifest_owner_expectations = {
        "reference_step_3": (
            reference.run["run_id"],
            reference.run["continuation"]["segment_id"],
            0,
        ),
        "reference_step_5": (
            reference.run["run_id"],
            reference.run["continuation"]["segment_id"],
            0,
        ),
        "parent_step_3": (
            parent.run["run_id"],
            parent.run["continuation"]["segment_id"],
            0,
        ),
        "child_step_5": (
            child.run["run_id"],
            child.run["continuation"]["segment_id"],
            1,
        ),
    }
    for name, (run_id, segment_id, index) in manifest_owner_expectations.items():
        manifest = manifests[name]
        observed = (
            manifest.parent_run_id,
            manifest.parent_segment_id,
            manifest.continuation_index,
        )
        expected = (run_id, segment_id, index)
        if observed != expected:
            _add_mismatch(
                mismatches,
                code="wave7_compare.lineage_mismatch",
                scope="lineage",
                path=f"{name}.manifest_owner",
                expected=expected,
                observed=observed,
            )
    return {
        "continuation_index": child.run["continuation"].get("continuation_index"),
        "parent_checkpoint_identity": parent_identity,
        "parent_run_id": parent.run["run_id"],
        "parent_segment_id": parent.run["continuation"]["segment_id"],
    }


def _projection_for_log(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in row.items()
        if key not in TIMING_FIELDS
        and key != "per_rank_measurement"
        and not key.startswith("resource/")
    }
    non_finite = result.get("non_finite_fields")
    if isinstance(non_finite, list):
        result["non_finite_fields"] = sorted(
            field
            for field in non_finite
            if field not in TIMING_FIELDS and not field.startswith("resource/")
        )
    return result


def _excluded_log_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        key: value
        for key, value in row.items()
        if key in TIMING_FIELDS
        or key == "per_rank_measurement"
        or key.startswith("resource/")
    }
    return {
        "split": row["split"],
        "step": row["step"],
        "values": values,
    }


def _compare_log_value(
    left: Any,
    right: Any,
    *,
    path: str,
    scope: str,
    mismatches: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if (
            not isinstance(left, Mapping)
            or not isinstance(right, Mapping)
            or set(left) != set(right)
        ):
            _add_mismatch(
                mismatches,
                code="wave7_compare.logging_fields",
                scope=scope,
                path=path,
                expected=left,
                observed=right,
            )
            return
        for key in sorted(left):
            _compare_log_value(
                left[key],
                right[key],
                path=f"{path}.{key}",
                scope=scope,
                mismatches=mismatches,
                summary=summary,
            )
        return
    if isinstance(left, list) or isinstance(right, list):
        if (
            not isinstance(left, list)
            or not isinstance(right, list)
            or len(left) != len(right)
        ):
            _add_mismatch(
                mismatches,
                code="wave7_compare.logging_value",
                scope=scope,
                path=path,
                expected=left,
                observed=right,
            )
            return
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _compare_log_value(
                a,
                b,
                path=f"{path}[{index}]",
                scope=scope,
                mismatches=mismatches,
                summary=summary,
            )
        return
    numeric = (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    )
    if numeric:
        leaf = path.rsplit(".", maxsplit=1)[-1]
        exact = leaf == "step" or leaf.endswith("_count") or leaf.startswith("lr/")
        exact = exact or leaf.startswith("count/")
        if exact:
            classification = "exact_count_id_lr"
            rtol = 0.0
            atol = 0.0
            passed = type(left) is type(right) and left == right
        elif leaf.startswith("loss/") or "diagnostic" in leaf.lower():
            classification = "bf16_loss_or_diagnostic"
            rtol = 5e-3
            atol = 5e-3
            passed = math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)
        else:
            classification = "aggregate_derived_metric"
            rtol = 1e-6
            atol = 1e-8
            passed = math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)
        difference = abs(float(left) - float(right))
        relative = difference / max(abs(float(left)), 1e-30)
        bucket = summary.setdefault(
            classification,
            {
                "atol": atol,
                "comparison_count": 0,
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
                "rtol": rtol,
                "worst_abs_path": None,
                "worst_rel_path": None,
            },
        )
        bucket["comparison_count"] += 1
        if difference >= bucket["max_abs_diff"]:
            bucket["max_abs_diff"] = difference
            bucket["worst_abs_path"] = path
        if relative >= bucket["max_rel_diff"]:
            bucket["max_rel_diff"] = relative
            bucket["worst_rel_path"] = path
    else:
        passed = left == right
    if not passed:
        _add_mismatch(
            mismatches,
            code="wave7_compare.logging_value",
            scope=scope,
            path=path,
            expected=left,
            observed=right,
        )


def _index_logs(
    rows: Sequence[Mapping[str, Any]], *, role: str
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["split"]), int(row["step"]))
        if key in result:
            raise Wave7CompareError(
                f"{role} repeats one split/step logging row",
                code="wave7_compare.logging_schema",
                context={"split": key[0], "step": key[1]},
            )
        result[key] = row
    return result


def _compare_logs(
    runs: Mapping[str, RunArtifacts], mismatches: list[dict[str, Any]]
) -> dict[str, Any]:
    indexed = {
        role: _index_logs(artifacts.logs, role=role) for role, artifacts in runs.items()
    }
    numeric_summary: dict[str, Any] = {}
    expected_keys = {
        "uninterrupted": {*(("train", step) for step in range(1, 6)), ("eval", 3)},
        "interrupted_parent": {*(("train", step) for step in range(1, 4)), ("eval", 3)},
        "resume_child": {("train", 4), ("train", 5)},
    }
    for role, expected in expected_keys.items():
        if set(indexed[role]) != expected:
            _add_mismatch(
                mismatches,
                code="wave7_compare.logging_inventory",
                scope=role,
                path="logging.split_steps",
                expected=sorted(expected),
                observed=sorted(indexed[role]),
            )
    for step in range(1, 6):
        candidate_role = "interrupted_parent" if step <= 3 else "resume_child"
        left = indexed["uninterrupted"].get(("train", step))
        right = indexed[candidate_role].get(("train", step))
        if left is not None and right is not None:
            _compare_log_value(
                _projection_for_log(left),
                _projection_for_log(right),
                path=f"train.step-{step}",
                scope="combined_training_logs",
                mismatches=mismatches,
                summary=numeric_summary,
            )
    reference_eval = indexed["uninterrupted"].get(("eval", 3))
    parent_eval = indexed["interrupted_parent"].get(("eval", 3))
    if reference_eval is not None and parent_eval is not None:
        _compare_log_value(
            _projection_for_log(reference_eval),
            _projection_for_log(parent_eval),
            path="eval.step-3",
            scope="evaluation_logs",
            mismatches=mismatches,
            summary=numeric_summary,
        )
    return {
        "child_replayed_step_3_eval": ("eval", 3) in indexed["resume_child"],
        "excluded_observations": {
            role: [_excluded_log_observation(row) for row in artifacts.logs]
            for role, artifacts in runs.items()
        },
        "excluded_from_numerical_equality": [
            "input_build_seconds",
            "input_wait_seconds",
            "per_rank_measurement",
            "resource/*",
            "step_duration_seconds",
        ],
        "numeric_summary": numeric_summary,
        "train_step_ids": {
            role: sorted(step for split, step in rows if split == "train")
            for role, rows in indexed.items()
        },
    }


def _validate_aliases(
    runs: Mapping[str, RunArtifacts], mismatches: list[dict[str, Any]]
) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for role in ("uninterrupted", "resume_child"):
        alias = _strict_json_file(
            runs[role].root / "checkpoints/final.json", owner=f"{role} final alias"
        )
        expected = {"checkpoint_path": "checkpoints/step-5", "step": 5}
        if alias != expected:
            _add_mismatch(
                mismatches,
                code="wave7_compare.final_alias",
                scope=role,
                path="checkpoints/final.json",
                expected=expected,
                observed=alias,
            )
        aliases[role] = alias
    parent_final = runs["interrupted_parent"].root / "checkpoints/final.json"
    if parent_final.exists() or parent_final.is_symlink():
        _add_mismatch(
            mismatches,
            code="wave7_compare.parent_final_alias",
            scope="interrupted_parent",
            path="checkpoints/final.json",
            expected="absent",
            observed=str(parent_final),
        )
    aliases["interrupted_parent"] = None
    return aliases


def _load_best_alias(run: RunArtifacts) -> dict[str, Any] | None:
    path = run.root / "checkpoints/best.json"
    if not (path.exists() or path.is_symlink()):
        return None
    value = _strict_json_file(path, owner=f"{run.role} best alias")
    if set(value) != {"checkpoint_path", "selector", "step", "value"}:
        raise Wave7CompareError(
            f"{run.role} best alias has an invalid field set",
            code="wave7_compare.checkpoint_selection",
        )
    step = value["step"]
    selector = value["selector"]
    metric = value["value"]
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
        or not isinstance(selector, str)
        or not selector
        or isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(float(metric))
        or value["checkpoint_path"] != f"checkpoints/step-{step}"
    ):
        raise Wave7CompareError(
            f"{run.role} best alias is malformed",
            code="wave7_compare.checkpoint_selection",
        )
    return value


def _selection_eligibility(
    run: RunArtifacts, alias: Mapping[str, Any] | None
) -> dict[str, Any]:
    if alias is None:
        return {"eligible": False, "reason": "best_alias_absent"}
    step = int(alias["step"])
    indexed = _index_logs(run.logs, role=run.role)
    train = indexed.get(("train", step))
    evaluation = indexed.get(("eval", step))
    selector = str(alias["selector"])
    metric = None if evaluation is None else evaluation.get(selector)
    metric_matches = (
        isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and math.isfinite(float(metric))
        and math.isclose(
            float(metric), float(alias["value"]), rel_tol=1e-6, abs_tol=1e-8
        )
    )
    checkpoint_present = (run.root / str(alias["checkpoint_path"])).is_dir()
    eligible = (
        train is not None
        and train.get("optimizer_update_status") == "applied"
        and train.get("finite_status") == "finite"
        and evaluation is not None
        and metric_matches
        and checkpoint_present
    )
    return {
        "checkpoint_present": checkpoint_present,
        "eligible": eligible,
        "eval_present": evaluation is not None,
        "finite_status": None if train is None else train.get("finite_status"),
        "metric_matches_alias": metric_matches,
        "optimizer_update_status": (
            None if train is None else train.get("optimizer_update_status")
        ),
    }


def _compare_checkpoint_selection(
    runs: Mapping[str, RunArtifacts], mismatches: list[dict[str, Any]]
) -> dict[str, Any]:
    aliases = {role: _load_best_alias(run) for role, run in runs.items()}
    eligibility = {
        role: _selection_eligibility(runs[role], alias)
        for role, alias in aliases.items()
    }
    before = len(mismatches)
    for role in ("uninterrupted", "interrupted_parent"):
        if aliases[role] is None or eligibility[role].get("eligible") is not True:
            _add_mismatch(
                mismatches,
                code="wave7_compare.checkpoint_selection",
                scope=role,
                path="checkpoints/best.json.eligibility",
                expected="present and eligible",
                observed={"alias": aliases[role], "eligibility": eligibility[role]},
            )
    if (
        aliases["resume_child"] is not None
        and eligibility["resume_child"].get("eligible") is not True
    ):
        _add_mismatch(
            mismatches,
            code="wave7_compare.checkpoint_selection",
            scope="resume_child",
            path="checkpoints/best.json.eligibility",
            expected="absent or eligible",
            observed={
                "alias": aliases["resume_child"],
                "eligibility": eligibility["resume_child"],
            },
        )
    candidates = [
        (role, aliases[role])
        for role in ("interrupted_parent", "resume_child")
        if aliases[role] is not None and eligibility[role].get("eligible") is True
    ]
    combined_role: str | None = None
    combined: Mapping[str, Any] | None = None
    if candidates:
        combined_role, combined = max(
            candidates, key=lambda item: float(item[1]["value"])
        )
    reference = aliases["uninterrupted"]
    exact_fields_match = (
        reference is not None
        and combined is not None
        and all(
            reference[key] == combined[key]
            for key in ("checkpoint_path", "selector", "step")
        )
    )
    value_matches = (
        reference is not None
        and combined is not None
        and math.isclose(
            float(reference["value"]),
            float(combined["value"]),
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
    )
    if not exact_fields_match or not value_matches:
        _add_mismatch(
            mismatches,
            code="wave7_compare.checkpoint_selection",
            scope="checkpoint_selection",
            path="reference_vs_combined_parent_child",
            expected=reference,
            observed={"alias": combined, "owner": combined_role},
        )
    return {
        "combined_owner": combined_role,
        "combined_parent_child": combined,
        "eligibility": eligibility,
        "reference": reference,
        "status": "passed" if len(mismatches) == before else "failed",
        "value_tolerance": {"atol": 1e-8, "rtol": 1e-6},
    }


def _publication_measurements(
    runs: Mapping[str, RunArtifacts],
    checkpoint_surfaces: Mapping[str, Any],
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_steps = {
        "uninterrupted": (3, 5),
        "interrupted_parent": (3,),
        "resume_child": (5,),
    }
    surface_names = {
        ("uninterrupted", 3): "reference_step_3",
        ("uninterrupted", 5): "reference_step_5",
        ("interrupted_parent", 3): "parent_step_3",
        ("resume_child", 5): "child_step_5",
    }
    before = len(mismatches)
    by_run: dict[str, Any] = {}
    total_duration = 0.0
    total_checkpoint_bytes = 0
    total_training_state_bytes = 0
    total_inference_bytes = 0
    total_count = 0
    for role, artifacts in runs.items():
        measurement = artifacts.run.get("measurement")
        raw_events = (
            measurement.get("checkpoint_publication_events")
            if isinstance(measurement, Mapping)
            else None
        )
        valid_events: list[dict[str, Any]] = []
        if isinstance(raw_events, list):
            for raw in raw_events:
                if (
                    not isinstance(raw, Mapping)
                    or set(raw) != CHECKPOINT_PUBLICATION_EVENT_FIELDS
                ):
                    continue
                duration = raw["duration_seconds"]
                step = raw["step"]
                surface_name = (
                    surface_names.get((role, step))
                    if isinstance(step, int) and not isinstance(step, bool)
                    else None
                )
                surface = (
                    checkpoint_surfaces.get(surface_name)
                    if surface_name is not None
                    else None
                )
                expected_identity = (
                    {
                        "checkpoint_step": step,
                        "resolved_path": surface["path"],
                        "training_state_aggregate_digest": surface[
                            "manifest_aggregate_digest"
                        ],
                        "training_state_manifest_file_sha256": surface[
                            "manifest_file_sha256"
                        ],
                    }
                    if surface is not None
                    else None
                )
                identity = raw["checkpoint_identity"]
                if (
                    raw["status"] != "completed"
                    or not isinstance(raw["started_at"], str)
                    or not raw["started_at"]
                    or not isinstance(raw["completed_at"], str)
                    or not raw["completed_at"]
                    or isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(float(duration))
                    or float(duration) < 0
                    or isinstance(step, bool)
                    or not isinstance(step, int)
                    or step <= 0
                    or raw["duration_clock"] != "monotonic"
                    or raw["checkpoint_path"] != f"checkpoints/step-{step}"
                    or not isinstance(raw["is_final"], bool)
                    or raw["is_final"] is not (step == 5)
                    or raw["exact_training_state_enabled"] is not True
                    or raw["failure_code"] is not None
                    or expected_identity is None
                    or not isinstance(identity, Mapping)
                    or set(identity)
                    != {
                        "checkpoint_step",
                        "resolved_path",
                        "training_state_aggregate_digest",
                        "training_state_manifest_file_sha256",
                    }
                    or isinstance(identity["checkpoint_step"], bool)
                    or not isinstance(identity["checkpoint_step"], int)
                    or dict(identity) != expected_identity
                ):
                    continue
                valid_event = dict(raw)
                valid_event["duration_seconds"] = float(duration)
                valid_events.append(valid_event)
        observed_steps = tuple(event["step"] for event in valid_events)
        contract_ok = (
            observed_steps == expected_steps[role]
            and isinstance(raw_events, list)
            and len(valid_events) == len(raw_events)
            and artifacts.run["checkpoint_event_count"] == len(valid_events)
        )
        if not contract_ok:
            _add_mismatch(
                mismatches,
                code="wave7_compare.checkpoint_publication",
                scope=role,
                path="run.measurement.checkpoint_publication_events",
                expected={
                    "checkpoint_event_count": len(expected_steps[role]),
                    "completed_steps": list(expected_steps[role]),
                    "event_fields": sorted(CHECKPOINT_PUBLICATION_EVENT_FIELDS),
                    "exact_training_state_enabled": True,
                    "identity_bound_to_strict_manifest": True,
                },
                observed={
                    "checkpoint_event_count": artifacts.run["checkpoint_event_count"],
                    "events": raw_events,
                },
            )
        event_receipts: list[dict[str, Any]] = []
        for event in valid_events:
            surface_name = surface_names.get((role, int(event["step"])))
            if surface_name is None:
                continue
            sizes = checkpoint_surfaces[surface_name]["sizes"]
            event_receipts.append({**event, "sizes": sizes})
            total_duration += float(event["duration_seconds"])
            total_checkpoint_bytes += int(sizes["checkpoint_total_bytes"])
            total_training_state_bytes += int(sizes["training_state_bytes"])
            total_inference_bytes += int(sizes["inference_payload_bytes"])
            total_count += 1
        by_run[role] = {
            "checkpoint_count": len(event_receipts),
            "events": event_receipts,
            "total_duration_seconds": sum(
                float(event["duration_seconds"]) for event in event_receipts
            ),
        }
    return {
        "by_run": by_run,
        "cadence_cost": {
            "five_step_smoke_only": True,
            "mean_publication_seconds_per_checkpoint": (
                None if total_count == 0 else total_duration / total_count
            ),
            "total_checkpoint_bytes": total_checkpoint_bytes,
            "total_checkpoint_count": total_count,
            "total_inference_payload_bytes": total_inference_bytes,
            "total_publication_seconds": total_duration,
            "total_training_state_bytes": total_training_state_bytes,
        },
        "status": "passed" if len(mismatches) == before else "failed",
    }


def _publish(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        write_strict_json_atomic(path, payload)
    except ParityContractError as exc:
        raise Wave7CompareError(
            "comparison receipt publication failed",
            code="wave7_compare.publication",
            context={"publisher_code": exc.code, "path": str(path)},
        ) from exc
    persisted = path.read_bytes()
    if _strict_json_loads(persisted, owner="published comparison receipt") != dict(
        payload
    ):
        raise Wave7CompareError(
            "comparison receipt did not reload exactly",
            code="wave7_compare.publication_reload",
        )


def _discard_staged_receipt(stage: Path) -> None:
    try:
        stage.unlink(missing_ok=True)
    finally:
        try:
            stage.parent.rmdir()
        except FileNotFoundError:
            pass


def _stage_receipt_payload(output: Path, payload: Mapping[str, Any]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent)
    )
    stage = stage_dir / "receipt.json"
    try:
        _publish(stage, payload)
    except BaseException:
        _discard_staged_receipt(stage)
        raise
    return stage


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_exact_owned_provisional(
    stage: Path, output: Path, payload: Mapping[str, Any]
) -> bool:
    try:
        stage_stat = stage.stat(follow_symlinks=False)
        output_stat = output.stat(follow_symlinks=False)
        if (
            stage_stat.st_dev != output_stat.st_dev
            or stage_stat.st_ino != output_stat.st_ino
            or _strict_json_loads(
                output.read_bytes(), owner="provisional comparison receipt"
            )
            != dict(payload)
        ):
            return False
        output.unlink()
        _fsync_directory(output.parent)
        return True
    except (OSError, Wave7CompareError):
        return False


def _commit_staged_receipt(
    stage: Path,
    output: Path,
    payload: Mapping[str, Any],
    *,
    interrupted_parent_run_dir: Path,
    expected_parent_identity: Mapping[str, Any],
) -> dict[str, Any]:
    linked = False
    try:
        assert_absent_artifact_target(output)
        os.link(stage, output, follow_symlinks=False)
        linked = True
        _fsync_directory(output.parent)
        if _strict_json_loads(
            output.read_bytes(), owner="committed comparison receipt"
        ) != dict(payload):
            raise Wave7CompareError(
                "committed comparison receipt differs from its staged payload",
                code="wave7_compare.publication_reload",
            )
        observed_parent_identity = _tree_identity(
            interrupt._snapshot_run_tree(interrupted_parent_run_dir)
        )
        if observed_parent_identity != dict(expected_parent_identity):
            if not _remove_exact_owned_provisional(stage, output, payload):
                raise Wave7CompareError(
                    "refusing to remove a provisional target that is not the exact staged receipt",
                    code="wave7_compare.provisional_ownership",
                    context={"path": str(output)},
                )
            linked = False
        return observed_parent_identity
    except BaseException:
        if linked:
            _remove_exact_owned_provisional(stage, output, payload)
        raise
    finally:
        _discard_staged_receipt(stage)


def _sign_payload(payload: dict[str, Any], mismatches: Sequence[Any]) -> None:
    payload["status"] = "passed" if not mismatches else "failed"
    unsigned = dict(payload)
    unsigned.pop("receipt_payload_sha256")
    payload["receipt_payload_sha256"] = _sha256_bytes(canonical_json_bytes(unsigned))


def _tree_identity(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_count": snapshot["entry_count"],
        "fingerprint": snapshot["fingerprint"],
    }


def execute(request: CompareRequest) -> int:
    try:
        _validate_input_topology(request)
        interrupt_source_pin = _validate_interrupt_source_pin(request)
        assert_absent_artifact_target(request.output)
        initial_parent_tree = interrupt._snapshot_run_tree(
            request.interrupted_parent_run_dir
        )
    except Exception as exc:
        code = getattr(exc, "code", "wave7_compare.output_collision")
        print(f"{code}: {exc}", file=sys.stderr)
        return 2
    source_sha256 = _sha256_file(Path(__file__).resolve())
    mismatches: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "claim_boundaries": {
            "bitwise_cross_launch_reproducibility": "not_claimed",
            "cadence_benefit": "not_claimed_beyond_five_step_smoke",
            "inference_validation": (
                "bounded_structural_production_validators_not_runtime_model_load"
            ),
            "performance_scope": (
                "checkpoint_bytes_and_declared_publication_events_only"
            ),
            "task_8_2_completion": (
                "not_claimed_without_independent_runtime_inference_load"
            ),
            "workload_scope": "one_uninterrupted_and_one_step3_interrupted_resume_to_step5",
        },
        "checkpoint_selection": {},
        "comparison_contract": {
            "exact": [
                "commit",
                "execution_relevant_digest",
                "cache_policy_dependency_topology_identities",
                "resume_compatibility",
                "rank_cursor_next_pack_example_ids_counters_rng",
                "runtime_structure_signature",
                "nonfloating_state",
                "step_ids_counts_lr_status",
                "checkpoint_inventory_lineage_aliases",
            ],
            "tolerances": {
                "bf16_derived_or_model_update": {"atol": 5e-3, "rtol": 5e-3},
                "fp32_same_forward_or_control_state": {"atol": 1e-5, "rtol": 1e-4},
                "bf16_loss_or_diagnostic": {"atol": 5e-3, "rtol": 5e-3},
                "aggregate_derived_metrics": {"atol": 1e-8, "rtol": 1e-6},
            },
        },
        "input_hashes": {},
        "interruption": {},
        "interrupt_source_pin": interrupt_source_pin,
        "lineage": {},
        "log_comparison": {},
        "mismatches": mismatches,
        "publication_measurements": {},
        "parent_tree_toctou": {
            "final_precommit": None,
            "initial": _tree_identity(initial_parent_tree),
            "prepublication": None,
        },
        "receipt_payload_sha256": None,
        "runs": {},
        "schema": RECEIPT_SCHEMA,
        "source": {
            "expected_sha256": request.expected_source_sha256,
            "observed_sha256": source_sha256,
            "path": str(Path(__file__).resolve()),
        },
        "state_comparisons": {},
        "status": "failed",
        "storage": {},
    }
    if source_sha256 != request.expected_source_sha256:
        _add_mismatch(
            mismatches,
            code="wave7_compare.source_mismatch",
            scope="source",
            path="comparator.sha256",
            expected=request.expected_source_sha256,
            observed=source_sha256,
        )
    else:
        try:
            runs = {
                "uninterrupted": _load_run(
                    request.uninterrupted_run_dir, role="uninterrupted"
                ),
                "interrupted_parent": _load_run(
                    request.interrupted_parent_run_dir, role="interrupted_parent"
                ),
                "resume_child": _load_run(
                    request.resume_child_run_dir, role="resume_child"
                ),
            }
            payload["runs"] = {
                role: {
                    "hashes": artifacts.hashes,
                    "path": str(artifacts.root),
                    "role": role,
                    "run_id": artifacts.run["run_id"],
                    "segment_id": artifacts.run["continuation"]["segment_id"],
                }
                for role, artifacts in runs.items()
            }
            payload["input_hashes"] = {
                role: dict(artifacts.hashes) for role, artifacts in runs.items()
            }
            payload["input_hashes"].update(
                interruption_marker_sha256=_sha256_file(request.interruption_marker),
                termination_receipt_sha256=_sha256_file(request.termination_receipt),
            )
            payload["run_contract"] = _compare_run_contracts(runs, request, mismatches)
            payload["interruption"] = _load_interrupt_evidence(
                request, runs["interrupted_parent"], mismatches
            )
            checkpoint_surfaces, manifests = _load_checkpoint_surfaces(runs, mismatches)
            payload["storage"] = checkpoint_surfaces
            payload["lineage"] = _validate_lineage(
                runs,
                manifests,
                checkpoint_surfaces,
                payload["interruption"],
                mismatches,
            )
            payload["state_comparisons"] = {
                "step_3": _compare_checkpoint_pair(
                    runs["uninterrupted"].root / "checkpoints/step-3",
                    runs["interrupted_parent"].root / "checkpoints/step-3",
                    step=3,
                    scope="reference_vs_parent_step_3",
                    mismatches=mismatches,
                ),
                "step_5": _compare_checkpoint_pair(
                    runs["uninterrupted"].root / "checkpoints/step-5",
                    runs["resume_child"].root / "checkpoints/step-5",
                    step=5,
                    scope="reference_vs_child_step_5",
                    mismatches=mismatches,
                ),
            }
            payload["log_comparison"] = _compare_logs(runs, mismatches)
            if payload["log_comparison"]["child_replayed_step_3_eval"]:
                _add_mismatch(
                    mismatches,
                    code="wave7_compare.eval_replay",
                    scope="resume_child",
                    path="logging.eval.step-3",
                    expected="absent",
                    observed="present",
                )
            payload["aliases"] = _validate_aliases(runs, mismatches)
            payload["checkpoint_selection"] = _compare_checkpoint_selection(
                runs, mismatches
            )
            payload["publication_measurements"] = _publication_measurements(
                runs, checkpoint_surfaces, mismatches
            )
        except BaseException as exc:
            if len(mismatches) >= MAX_MISMATCH_ROWS:
                mismatches.clear()
            mismatches.append(_error_mismatch(exc, scope="comparison"))
    try:
        prepublication_parent_tree = interrupt._snapshot_run_tree(
            request.interrupted_parent_run_dir
        )
        prepublication_identity = _tree_identity(prepublication_parent_tree)
        payload["parent_tree_toctou"]["prepublication"] = prepublication_identity
        if prepublication_identity != payload["parent_tree_toctou"]["initial"]:
            _add_mismatch(
                mismatches,
                code="wave7_compare.parent_tree_changed",
                scope="interrupted_parent",
                path="run_tree_during_comparison",
                expected=payload["parent_tree_toctou"]["initial"],
                observed=prepublication_identity,
            )
    except BaseException as exc:
        mismatches.append(_error_mismatch(exc, scope="parent_tree_prepublication"))
    _sign_payload(payload, mismatches)
    staged: Path | None = None
    try:
        staged = _stage_receipt_payload(request.output, payload)
        final_precommit_identity = _tree_identity(
            interrupt._snapshot_run_tree(request.interrupted_parent_run_dir)
        )
        payload["parent_tree_toctou"]["final_precommit"] = final_precommit_identity
        if final_precommit_identity != payload["parent_tree_toctou"]["prepublication"]:
            _add_mismatch(
                mismatches,
                code="wave7_compare.parent_tree_changed",
                scope="interrupted_parent",
                path="run_tree_at_publication_boundary",
                expected=payload["parent_tree_toctou"]["prepublication"],
                observed=final_precommit_identity,
            )
        _discard_staged_receipt(staged)
        staged = None
        _sign_payload(payload, mismatches)
        if mismatches:
            _publish(request.output, payload)
        else:
            commit_boundary_identity = _tree_identity(
                interrupt._snapshot_run_tree(request.interrupted_parent_run_dir)
            )
            if commit_boundary_identity != final_precommit_identity:
                _add_mismatch(
                    mismatches,
                    code="wave7_compare.parent_tree_changed",
                    scope="interrupted_parent",
                    path="run_tree_at_final_absent_commit",
                    expected=final_precommit_identity,
                    observed=commit_boundary_identity,
                )
                payload["parent_tree_toctou"]["final_precommit"] = (
                    commit_boundary_identity
                )
                _sign_payload(payload, mismatches)
                _publish(request.output, payload)
            else:
                staged = _stage_receipt_payload(request.output, payload)
                postlink_identity = _commit_staged_receipt(
                    staged,
                    request.output,
                    payload,
                    interrupted_parent_run_dir=request.interrupted_parent_run_dir,
                    expected_parent_identity=commit_boundary_identity,
                )
                staged = None
                if postlink_identity != commit_boundary_identity:
                    _add_mismatch(
                        mismatches,
                        code="wave7_compare.parent_tree_changed",
                        scope="interrupted_parent",
                        path="run_tree_after_final_link",
                        expected=commit_boundary_identity,
                        observed=postlink_identity,
                    )
                    payload["parent_tree_toctou"]["final_precommit"] = postlink_identity
                    _sign_payload(payload, mismatches)
                    _publish(request.output, payload)
    except BaseException as exc:
        if staged is not None:
            _discard_staged_receipt(staged)
        print(
            f"{getattr(exc, 'code', 'wave7_compare.unexpected')}: {exc}",
            file=sys.stderr,
        )
        return 2
    if mismatches:
        for row in mismatches[:16]:
            print(f"{row['code']}: {row['path']}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        request = parse_request(argv)
    except Wave7CompareError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    return execute(request)


if __name__ == "__main__":
    raise SystemExit(main())

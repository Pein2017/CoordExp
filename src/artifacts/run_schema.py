"""Pure strict-JSON schema owner for ``RunWriter`` artifacts.

Per design decision 10, this module owns strict JSON normalization/
serialization, bounded detail validation, timestamp/number/lineage
validation, measurement payload construction, and logging-row normalization
moved verbatim out of ``src/artifacts/run_writer.py``. Nothing here performs
filesystem I/O or reads the wall clock; ``RunWriter`` remains the only owner
that reads/writes run files and sequences transitions, importing these pure
functions rather than reimplementing them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.artifacts.resources import validate_rank_cpu_resource_receipt
from src.common.errors import ArtifactContractError

_CONTINUATION_LINEAGE_FIELDS = frozenset(
    {
        "continuation_index",
        "parent_checkpoint_identity",
        "parent_continuation_index",
        "parent_run_id",
        "parent_segment_id",
    }
)
_PARENT_CHECKPOINT_IDENTITY_FIELDS = frozenset(
    {
        "checkpoint_step",
        "resolved_path",
        "training_state_aggregate_digest",
        "training_state_manifest_file_sha256",
    }
)
_COMMITTED_PROGRESS_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "completed_steps",
        "consumed_packs",
        "optimizer_update_status",
        "finite_status",
    }
)
_UNPHASED_FAILURE_PHASE = "unphased_failure"
_PHASE_NAMES = frozenset(
    {
        "cache_admission",
        "cache_identity_resolution",
        "cache_preparation",
        "cache_publication",
        "cache_publication_admission",
        "checkpoint_publication",
        "config_provenance_resolution",
        "evaluation_hydration",
        "evaluation_execution",
        "first_optimizer_step",
        "model_loading",
        "optimizer_runtime_assembly",
        "steady_state",
        "train_rank_hydration",
        _UNPHASED_FAILURE_PHASE,
    }
)


def _normalize_logging_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("split") not in {"train", "eval"}:
        raise ArtifactContractError(
            "logging split must be train or eval",
            code="run_writer.invalid_logging_split",
        )
    step = row.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ArtifactContractError(
            "logging step must be a positive integer",
            code="run_writer.invalid_logging_step",
        )
    normalized = deepcopy(dict(row))
    fields: list[str] = []
    normalized = _replace_non_finite(normalized, path="", fields=fields)
    declared = normalized.get("non_finite_fields", [])
    if not isinstance(declared, list) or not all(
        isinstance(item, str) for item in declared
    ):
        raise ArtifactContractError(
            "non_finite_fields must be a list of field names",
            code="run_writer.invalid_non_finite_fields",
        )
    normalized["non_finite_fields"] = sorted(set(declared).union(fields))
    _reject_non_finite(normalized)
    return normalized


def _provenance_payload(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if provenance is not None:
        return _strict_mapping(provenance, field="provenance")
    unavailable = {"status": "unavailable", "reason": "not_collected_by_caller"}
    return {
        "schema_version": 1,
        "repository": {
            "commit": dict(unavailable),
            "state": "unavailable",
            "tracked_changes_present": None,
            "untracked_changes_present": None,
            "execution_relevant_changes": {
                "count": 0,
                "path_classes": {},
                "truncated": False,
            },
            "execution_relevant_digest": dict(unavailable),
        },
        "dependencies": {},
        "runtime": {},
    }


def _continuation_payload(
    *,
    run_id: str,
    segment_id: str,
    lineage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if lineage is None:
        return {
            "schema_version": 1,
            "segment_id": segment_id,
            "continuation_index": 0,
            "parent": None,
        }
    if not isinstance(lineage, Mapping) or set(lineage) != _CONTINUATION_LINEAGE_FIELDS:
        _invalid_continuation_lineage(
            "continuation lineage must contain the complete exact field set"
        )
    parent_run_id = _validate_lineage_identity(
        lineage["parent_run_id"], field="parent_run_id"
    )
    parent_segment_id = _validate_lineage_identity(
        lineage["parent_segment_id"], field="parent_segment_id"
    )
    if parent_run_id == run_id or parent_segment_id == segment_id:
        _invalid_continuation_lineage(
            "a continuation cannot identify itself as its parent"
        )
    parent_index = _validate_lineage_index(
        lineage["parent_continuation_index"],
        field="parent_continuation_index",
    )
    continuation_index = _validate_lineage_index(
        lineage["continuation_index"], field="continuation_index"
    )
    if continuation_index <= 0 or continuation_index != parent_index + 1:
        _invalid_continuation_lineage(
            "continuation index must advance its parent index exactly once"
        )
    checkpoint_identity = _validate_parent_checkpoint_identity(
        lineage["parent_checkpoint_identity"]
    )
    return {
        "schema_version": 1,
        "segment_id": segment_id,
        "continuation_index": continuation_index,
        "parent": {
            "run_id": parent_run_id,
            "segment_id": parent_segment_id,
            "checkpoint_identity": checkpoint_identity,
            "continuation_index": parent_index,
        },
    }


def _validate_segment_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or not value.isascii()
    ):
        raise ArtifactContractError(
            "segment identity must be a nonempty bounded ASCII string",
            code="run_writer.invalid_segment_id",
        )
    return value


def _validate_lineage_identity(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or not value.isascii()
    ):
        _invalid_continuation_lineage(
            f"{field} must be a nonempty bounded ASCII string"
        )
    return value


def _validate_lineage_index(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _invalid_continuation_lineage(f"{field} must be a nonnegative integer")
    return value


def _validate_parent_checkpoint_identity(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _PARENT_CHECKPOINT_IDENTITY_FIELDS
    ):
        _invalid_continuation_lineage(
            "parent checkpoint identity must contain the exact resolved path and digest fields"
        )
    resolved_path = value["resolved_path"]
    if (
        not isinstance(resolved_path, str)
        or not resolved_path
        or len(resolved_path) > 4096
        or not Path(resolved_path).is_absolute()
        or str(Path(resolved_path).resolve()) != resolved_path
    ):
        _invalid_continuation_lineage(
            "parent checkpoint identity path must be absolute and resolved"
        )
    checkpoint_step = value["checkpoint_step"]
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step <= 0
    ):
        _invalid_continuation_lineage(
            "parent checkpoint step must be a positive integer"
        )
    manifest_digest = value["training_state_manifest_file_sha256"]
    aggregate_digest = value["training_state_aggregate_digest"]
    if not _is_lowercase_sha256(manifest_digest) or not _is_lowercase_sha256(
        aggregate_digest
    ):
        _invalid_continuation_lineage(
            "parent training-state identities must be lowercase SHA-256 digests"
        )
    return {
        "resolved_path": resolved_path,
        "checkpoint_step": checkpoint_step,
        "training_state_manifest_file_sha256": manifest_digest,
        "training_state_aggregate_digest": aggregate_digest,
    }


def _is_lowercase_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _invalid_continuation_lineage(message: str) -> None:
    raise ArtifactContractError(
        message,
        code="run_writer.invalid_continuation_lineage",
    )


def _checkpoint_publication_identity(
    value: Mapping[str, Any],
    *,
    step: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    fields = _PARENT_CHECKPOINT_IDENTITY_FIELDS
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ArtifactContractError(
            "checkpoint publication identity must contain the complete exact field set",
            code="run_writer.invalid_checkpoint_publication_event",
        )
    resolved_path = value["resolved_path"]
    if (
        not isinstance(resolved_path, str)
        or resolved_path != str(checkpoint_dir.resolve())
        or value["checkpoint_step"] != step
        or not _is_lowercase_sha256(value["training_state_manifest_file_sha256"])
        or not _is_lowercase_sha256(value["training_state_aggregate_digest"])
    ):
        raise ArtifactContractError(
            "checkpoint publication identity disagrees with its committed checkpoint",
            code="run_writer.invalid_checkpoint_publication_event",
        )
    return {
        "checkpoint_step": step,
        "resolved_path": resolved_path,
        "training_state_aggregate_digest": value["training_state_aggregate_digest"],
        "training_state_manifest_file_sha256": value[
            "training_state_manifest_file_sha256"
        ],
    }


def _checkpoint_committed_progress(
    value: Mapping[str, Any],
    *,
    step: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _COMMITTED_PROGRESS_FIELDS:
        raise ArtifactContractError(
            "checkpoint committed progress must contain the complete exact field set",
            code="run_writer.invalid_checkpoint_publication_event",
        )
    completed_steps = value["completed_steps"]
    consumed_packs = value["consumed_packs"]
    if (
        value["schema"] != "coordexp-swift-checkpoint-committed-progress"
        or value["schema_version"] != 1
        or isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps != step
        or isinstance(consumed_packs, bool)
        or not isinstance(consumed_packs, int)
        or consumed_packs < 0
    ):
        raise ArtifactContractError(
            "checkpoint committed progress disagrees with its publication step",
            code="run_writer.invalid_checkpoint_publication_event",
        )
    for field in ("optimizer_update_status", "finite_status"):
        status = value[field]
        if status is not None and (
            not isinstance(status, str)
            or not status
            or len(status) > 128
            or not status.isascii()
        ):
            raise ArtifactContractError(
                "checkpoint committed progress status is invalid",
                code="run_writer.invalid_checkpoint_publication_event",
                context={"field": field},
            )
    return {
        "schema": "coordexp-swift-checkpoint-committed-progress",
        "schema_version": 1,
        "completed_steps": completed_steps,
        "consumed_packs": consumed_packs,
        "optimizer_update_status": value["optimizer_update_status"],
        "finite_status": value["finite_status"],
    }


def _measurement_payload(
    context: Mapping[str, Any] | None,
    *,
    entry_started_at: str,
) -> dict[str, Any]:
    _validate_timestamp(entry_started_at, field="entry_started_at")
    payload = (
        {
            "comparison_arm": "unclassified",
            "wall_clock_scope": "training_entry_to_terminal_artifact",
            "warmup_exclusion_steps": None,
        }
        if context is None
        else _strict_mapping(context, field="measurement context")
    )
    return {
        "schema_version": 1,
        "context": payload,
        "active_phase": None,
        "terminal_phase": None,
        "terminal_phase_status": None,
        "last_completed_phase": None,
        "failure_phase": None,
        "phase_order": [],
        "phases": {},
        "checkpoint_publication_events": [],
        "resource_high_water": None,
        "accepted_measured_steps": 0,
        "expected_measured_steps": None,
        "steady_state_eligible": False,
        "entry_to_terminal": {
            "status": "running",
            "started_at": entry_started_at,
            "completed_at": None,
            "duration_seconds": None,
            "clock": "monotonic",
            "boundary": (
                "training_entry_to_terminal_state_durable_before_measurement_annotation"
            ),
        },
    }


def _attach_rank_phase_receipt(
    receipt: dict[str, Any],
    *,
    rank_resources: Mapping[str, Any] | None,
    rank_details: Mapping[str, Any] | None,
) -> None:
    if rank_resources is None:
        if rank_details is not None:
            raise ArtifactContractError(
                "rank details require a rank resource receipt",
                code="run_writer.invalid_additive_receipt",
                context={"field": "rank_details"},
            )
        return
    resources = validate_rank_cpu_resource_receipt(rank_resources)
    receipt["rank_resources"] = resources
    if rank_details is None:
        return
    details = _strict_mapping(rank_details, field="rank details")
    expected_ranks = {str(rank) for rank in range(int(resources["world_size"]))}
    if set(details) != expected_ranks:
        raise ArtifactContractError(
            "rank details must cover every resource rank exactly once",
            code="run_writer.invalid_additive_receipt",
            context={"field": "rank_details"},
        )
    _validate_bounded_rank_detail_value(details, depth=0)
    encoded = _strict_json_dumps(details, compact=True).encode("utf-8")
    if len(encoded) > 32 * 1024:
        raise ArtifactContractError(
            "rank details exceed the bounded artifact size",
            code="run_writer.invalid_additive_receipt",
            context={"field": "rank_details"},
        )
    receipt["rank_details"] = details


def _validate_bounded_rank_detail_value(value: Any, *, depth: int) -> None:
    if depth > 6:
        raise ArtifactContractError(
            "rank details exceed the bounded nesting depth",
            code="run_writer.invalid_additive_receipt",
            context={"field": "rank_details"},
        )
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
    elif isinstance(value, str):
        if len(value) <= 256 and value.isascii():
            return
    elif isinstance(value, Mapping):
        if len(value) <= 128 and all(
            isinstance(key, str) and 0 < len(key) <= 64 and key.isascii()
            for key in value
        ):
            for item in value.values():
                _validate_bounded_rank_detail_value(item, depth=depth + 1)
            return
    elif isinstance(value, (list, tuple)) and len(value) <= 128:
        for item in value:
            _validate_bounded_rank_detail_value(item, depth=depth + 1)
        return
    raise ArtifactContractError(
        "rank details contain an unbounded or unsupported value",
        code="run_writer.invalid_additive_receipt",
        context={"field": "rank_details"},
    )


def _strict_mapping(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactContractError(
            f"{field} must be a mapping",
            code="run_writer.invalid_additive_receipt",
            context={"field": field},
        )
    payload = deepcopy(dict(value))
    _strict_json_dumps(payload)
    return payload


def _validate_phase_name(phase: str) -> None:
    if phase not in _PHASE_NAMES:
        raise ArtifactContractError(
            "run phase name is not supported",
            code="run_writer.invalid_phase_name",
            context={"phase": phase},
        )


def _validate_timestamp(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ArtifactContractError(
            "phase timestamps must be nonempty strings",
            code="run_writer.invalid_phase_timestamp",
            context={"field": field},
        )


def _validate_nonnegative_finite(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ArtifactContractError(
            f"{field} must be finite and nonnegative",
            code="run_writer.invalid_phase_summary",
            context={"field": field},
        )
    return float(value)


def _validate_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactContractError(
            f"{field} must be a nonnegative integer",
            code="run_writer.invalid_measured_step_count",
            context={"field": field},
        )
    return value


def _replace_non_finite(value: Any, *, path: str, fields: list[str]) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        fields.append(path)
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _replace_non_finite(
                item, path=f"{path}.{key}" if path else str(key), fields=fields
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _replace_non_finite(item, path=f"{path}[{index}]", fields=fields)
            for index, item in enumerate(value)
        ]
    return value


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactContractError(
            "non-finite value remains after logging normalization",
            code="run_writer.non_finite_remains",
        )
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


def _strict_json_dumps(payload: Any, *, compact: bool = False) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":") if compact else None,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(
            "artifact payload is not strict-JSON serializable",
            code="run_writer.not_json_serializable",
            cause=exc,
        ) from exc

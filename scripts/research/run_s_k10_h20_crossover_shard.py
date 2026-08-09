#!/usr/bin/env python3
"""Run one immutable S K10/H20 natural-boundary crossover shard.

This runner is intentionally a small composition lane beside the frozen
15-arm S executor.  It owns no new model implementation and never mutates the
old cohort's ``ARM_ORDER``.  CPU preflight validates every source binding,
plan/event identity, and the pre-GPU receipt before the live executor or gate
is imported.  A production event then uses one natural-boundary gate/context
and records the technical K00 control plus C00/K10/H20/C11 cells in a fresh
write-once root.

The Python API has an injected event-runner seam for contract tests.  The CLI
crosses the existing guarded live executor only after preflight succeeds.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from scripts.research import natural_boundary_attention_actuators as attention  # noqa: E402


UNIT_ID = "2026-08-07-s-k10-h20-natural-crossover"
SCHEMA_VERSION = "s_k10_h20_crossover_event.v1"
PLAN_SCHEMA_VERSION = "s_k10_h20_crossover_plan.v1"
RUNTIME_IDENTITY_SCHEMA_VERSION = "s_k10_h20_crossover_runtime_identity.v1"
TERMINAL_SCHEMA_VERSION = "s_k10_h20_crossover_terminal_summary.v1"
AGGREGATE_RECEIPT_SCHEMA_VERSION = "s_k10_h20_crossover_shard_receipt.v1"
FAILURE_SCHEMA_VERSION = f"{SCHEMA_VERSION}.failure.v1"
PRE_GPU_PROBE_SCHEMA_VERSION = f"{SCHEMA_VERSION}.pre_gpu_probe.v1"
SOURCE_PREFLIGHT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.source_preflight.v1"

CHECKPOINT = "S"
STEP = 2444
MAX_ROWS = 3
MAX_ROW_TOKENS = 256
SHARD_COUNT = 3
CELLS = ("C00", "C10", "C01", "C11")
TECHNICAL_ARM = "K00"
TRANSPORT_ARMS = {"C00": "K01", "C10": "K10", "C01": "H20", "C11": "K10"}
NOOP_TOLERANCE = 1e-4
MANIFEST_IDENTITY_FIELDS = (
    "event_index",
    "event_id",
    "image_id",
    "event_sha256",
    "prefix_sha256",
    "geometry_sha256",
    "target_owner_id",
    "covered_owner_ids",
)
PLAN_EVENT_FIELDS = (*MANIFEST_IDENTITY_FIELDS, "source_qualification")
SOURCE_QUALIFICATION = {
    "k10_target_strict_release": True,
    "h20_factor_qualified": True,
    "h20_nondegenerate": True,
}


class CrossoverRunnerError(ValueError):
    """Raised for a preflight, composition, runtime, or receipt contract error."""


class CrossoverTechnicalInvalid(CrossoverRunnerError):
    """Raised when an event cannot be interpreted as a scientific outcome."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CrossoverRunnerError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    """Hash a directory's relative regular-file inventory without symlink traversal."""

    entries: list[dict[str, Any]] = []
    for child in sorted(path.rglob("*")):
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            raise CrossoverRunnerError(f"source directory contains a symlink: {child}")
        if child.is_file():
            # Match the pre-GPU sealer's canonical directory inventory so the
            # runner can recompute and compare one exact raw binding.
            entries.append(
                {
                    "relative_path": relative,
                    "sha256": _sha256_file(child),
                    "size_bytes": child.stat().st_size,
                }
            )
        elif not child.is_dir():
            raise CrossoverRunnerError(f"source directory contains a non-regular entry: {child}")
    entries.sort(key=lambda item: item["relative_path"])
    return sha256_json(entries)


def _regular_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise CrossoverRunnerError(f"{label} is not a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _regular_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise CrossoverRunnerError(f"{label} is not a regular non-symlink directory: {path}")
    return path.resolve(strict=True)


def _read_json(source: str | Path | Mapping[str, Any], label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(source, Mapping):
        document = dict(source)
        return document, {"inline": True, "sha256": sha256_json(document), "raw": None, "path": None}
    path = _regular_file(source, label)
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CrossoverRunnerError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise CrossoverRunnerError(f"{label} must be a JSON object")
    return dict(document), {"path": str(path), "sha256": sha256_bytes(raw), "raw": raw, "inline": False}


_NONPERSISTENT_CALLBACK_KEYS = frozenset(
    {"attention_mask", "score_bias", "score_bias_callback"}
)


def _stripped_tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().cpu().contiguous()
    return {
        "tensor_stripped": True,
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "sha256": attention.sha256_tensor(detached),
    }


def _detached_json(value: Any, *, label: str, key: str | None = None) -> Any:
    """Detach only named callback payloads; reject all other executable state."""

    if key in _NONPERSISTENT_CALLBACK_KEYS:
        if isinstance(value, torch.Tensor):
            return _stripped_tensor_metadata(value)
        if value is None:
            return None
        # The callback itself is deliberately omitted.  Its JSON-safe actuator
        # receipt/hash is retained in the sibling ``receipt`` field.
        if callable(value) or not isinstance(value, (str, int, float, bool, Mapping, list, tuple)):
            return {"nonpersistent_key": key, "stripped": True}
        return _detached_json(value, label=label, key=None)
    if isinstance(value, torch.Tensor):
        raise CrossoverRunnerError(f"{label} contains an unexpected tensor outside a callback payload")
    if callable(value):
        raise CrossoverRunnerError(f"{label} contains an unexpected callable outside a callback payload")
    if isinstance(value, Mapping):
        # These fields are executable/non-persistent callback payloads.  Drop
        # them at the mapping boundary instead of leaving a placeholder that
        # could be mistaken for a replayable mask or callback.  The immutable
        # actuator receipt/hash is retained in its explicit sibling field.
        return {
            str(name): _detached_json(item, label=f"{label}.{name}", key=str(name))
            for name, item in value.items()
            if str(name) not in _NONPERSISTENT_CALLBACK_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_detached_json(item, label=f"{label}[]") for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise CrossoverRunnerError(f"{label} contains an unexpected object of type {type(value).__name__}")


def _canonical_detached(value: Any, *, label: str) -> dict[str, Any]:
    detached = _detached_json(value, label=label)
    try:
        payload = json.loads(_canonical(detached).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CrossoverRunnerError(f"{label} is not JSON-safe after detachment: {exc}") from exc
    if not isinstance(payload, dict):
        raise CrossoverRunnerError(f"{label} must serialize as a JSON object")
    return payload


def _write_once_json(path: Path, document: Mapping[str, Any]) -> tuple[bool, str]:
    if path.is_symlink():
        raise CrossoverRunnerError(f"refusing to write through symlink: {path}")
    payload = _canonical(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"immutable receipt collision: {path}")
        return True, sha256_bytes(payload)
    flags = 0
    import os

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return False, sha256_bytes(payload)


def _write_once_bytes(path: Path, payload: bytes) -> tuple[bool, str]:
    if path.is_symlink():
        raise CrossoverRunnerError(f"refusing to write through symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"immutable receipt collision: {path}")
        return True, sha256_bytes(payload)
    import os

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return False, sha256_bytes(payload)


def _self_hash(document: Mapping[str, Any], key: str = "self_sha256") -> str:
    body = dict(document)
    declared = body.pop(key, None)
    if declared is None:
        return sha256_json(body)
    if not isinstance(declared, str) or len(declared) != 64:
        raise CrossoverRunnerError(f"{key} must be a lowercase SHA-256")
    observed = sha256_json(body)
    if declared != observed:
        raise CrossoverRunnerError(f"{key} does not match canonical document body")
    return declared


def _plan_self_hash(document: Mapping[str, Any]) -> tuple[str, str | None]:
    body = dict(document)
    declared_plan = body.pop("plan_sha256", None)
    declared_self = body.pop("plan_self_sha256", body.pop("self_sha256", None))
    observed = sha256_json(body)
    if declared_plan is not None and declared_plan != observed:
        raise CrossoverRunnerError("plan_sha256 does not match canonical plan body")
    if declared_self is not None and declared_self != observed:
        raise CrossoverRunnerError("plan_self_sha256 does not match canonical plan body")
    return str(declared_plan or observed), str(declared_self or observed)


def _normalize_shard_id(value: str | int, *, shard_count: int = SHARD_COUNT) -> tuple[str, int]:
    if isinstance(value, bool):
        raise CrossoverRunnerError("shard id must be an integer or shard-NNN string")
    if isinstance(value, int):
        index = value
        shard = f"shard-{index:03d}" if 0 <= index < shard_count else ""
    elif isinstance(value, str):
        if not value.startswith("shard-"):
            raise CrossoverRunnerError("shard id must use shard-NNN syntax")
        try:
            index = int(value[6:])
        except ValueError as exc:
            raise CrossoverRunnerError("shard id must use shard-NNN syntax") from exc
        shard = value
    else:
        raise CrossoverRunnerError("shard id must be an integer or shard-NNN string")
    if not 0 <= index < shard_count or shard != f"shard-{index:03d}":
        raise CrossoverRunnerError(f"shard id is outside the frozen 0..{shard_count - 1} range")
    return shard, index


def _identity_mismatch(label: str, field: str, expected: Any, observed: Any) -> None:
    raise CrossoverRunnerError(
        f"{label}.{field} differs: expected={expected!r}, observed={observed!r}"
    )


def _require_event_ref(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if set(value) != set(PLAN_EVENT_FIELDS):
        raise CrossoverRunnerError(
            f"{label} fields differ from exact plan projection: "
            f"expected={list(PLAN_EVENT_FIELDS)!r}, observed={list(value)!r}"
        )
    index = value["event_index"]
    image = value["image_id"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise CrossoverRunnerError(f"{label}.event_index is invalid")
    if isinstance(image, bool) or not isinstance(image, int) or image < 0:
        raise CrossoverRunnerError(f"{label}.image_id is invalid")
    if not isinstance(value["event_id"], str) or not value["event_id"]:
        raise CrossoverRunnerError(f"{label}.event_id is invalid")
    digest = _require_sha256(value["event_sha256"], f"{label}.event_sha256")
    normalized = {
        "event_index": int(index),
        "event_id": str(value["event_id"]),
        "image_id": int(image),
        "event_sha256": digest,
    }
    for key in ("prefix_sha256", "geometry_sha256"):
        normalized[key] = _require_sha256(value[key], f"{label}.{key}")
    target_owner_id = value["target_owner_id"]
    if not isinstance(target_owner_id, str) or not target_owner_id:
        raise CrossoverRunnerError(f"{label}.target_owner_id is invalid")
    covered_owner_ids = value["covered_owner_ids"]
    if (
        not isinstance(covered_owner_ids, list)
        or any(not isinstance(owner_id, str) or not owner_id for owner_id in covered_owner_ids)
        or len(set(covered_owner_ids)) != len(covered_owner_ids)
    ):
        raise CrossoverRunnerError(f"{label}.covered_owner_ids is invalid")
    qualification = value["source_qualification"]
    if not isinstance(qualification, Mapping) or dict(qualification) != SOURCE_QUALIFICATION:
        _identity_mismatch(
            label,
            "source_qualification",
            SOURCE_QUALIFICATION,
            dict(qualification) if isinstance(qualification, Mapping) else qualification,
        )
    normalized["target_owner_id"] = target_owner_id
    normalized["covered_owner_ids"] = list(covered_owner_ids)
    normalized["source_qualification"] = dict(SOURCE_QUALIFICATION)
    return normalized


def _manifest_event_projection(
    event: Mapping[str, Any], *, label: str = "manifest event"
) -> dict[str, Any]:
    declared_event_sha256 = _require_sha256(
        event.get("event_sha256"), f"{label}.event_sha256"
    )
    event_body = dict(event)
    event_body.pop("event_sha256", None)
    observed_event_sha256 = sha256_json(event_body)
    if declared_event_sha256 != observed_event_sha256:
        _identity_mismatch(
            label,
            "event_sha256",
            declared_event_sha256,
            observed_event_sha256,
        )

    natural_boundary = event.get("natural_boundary")
    if not isinstance(natural_boundary, Mapping):
        raise CrossoverRunnerError(f"{label}.natural_boundary is missing or malformed")
    prefix_sha256 = _require_sha256(
        natural_boundary.get("prefix_sha256"),
        f"{label}.natural_boundary.prefix_sha256",
    )
    history_sha256 = _require_sha256(
        natural_boundary.get("history_sha256"),
        f"{label}.natural_boundary.history_sha256",
    )
    if prefix_sha256 != history_sha256:
        _identity_mismatch(
            label,
            "natural_boundary.history_sha256",
            prefix_sha256,
            history_sha256,
        )

    geometry = event.get("geometry")
    if not isinstance(geometry, Mapping):
        raise CrossoverRunnerError(f"{label}.geometry is missing or malformed")
    nested_geometry_sha256 = _require_sha256(
        geometry.get("geometry_sha256"), f"{label}.geometry.geometry_sha256"
    )
    geometry_body = dict(geometry)
    geometry_body.pop("geometry_sha256", None)
    observed_geometry_sha256 = sha256_json(geometry_body)
    if nested_geometry_sha256 != observed_geometry_sha256:
        _identity_mismatch(
            label,
            "geometry.geometry_sha256",
            nested_geometry_sha256,
            observed_geometry_sha256,
        )
    top_level_geometry_sha256 = _require_sha256(
        event.get("geometry_sha256"), f"{label}.geometry_sha256"
    )
    if nested_geometry_sha256 != top_level_geometry_sha256:
        _identity_mismatch(
            label,
            "geometry_sha256",
            nested_geometry_sha256,
            top_level_geometry_sha256,
        )

    owner_refs = event.get("owner_refs")
    if not isinstance(owner_refs, Mapping):
        raise CrossoverRunnerError(f"{label}.owner_refs is missing or malformed")
    target_owner_id = geometry.get("target_owner_id")
    owner_ref_target = owner_refs.get("gt_owner_id")
    if not isinstance(target_owner_id, str) or not target_owner_id:
        raise CrossoverRunnerError(f"{label}.geometry.target_owner_id is invalid")
    if target_owner_id != owner_ref_target:
        _identity_mismatch(
            label,
            "owner_refs.gt_owner_id",
            target_owner_id,
            owner_ref_target,
        )
    covered_owner_ids = owner_refs.get("covered_owner_ids")
    if (
        not isinstance(covered_owner_ids, list)
        or any(not isinstance(owner_id, str) or not owner_id for owner_id in covered_owner_ids)
        or len(set(covered_owner_ids)) != len(covered_owner_ids)
    ):
        raise CrossoverRunnerError(f"{label}.owner_refs.covered_owner_ids is invalid")

    projected = {
        "event_index": event.get("event_index"),
        "event_id": event.get("event_id"),
        "image_id": event.get("image_id"),
        "event_sha256": declared_event_sha256,
        "prefix_sha256": prefix_sha256,
        "geometry_sha256": nested_geometry_sha256,
        "target_owner_id": target_owner_id,
        "covered_owner_ids": list(covered_owner_ids),
        "source_qualification": dict(SOURCE_QUALIFICATION),
    }
    normalized = _require_event_ref(projected, label=f"{label} projection")
    normalized.pop("source_qualification")
    return normalized


def _event_from_manifest(manifest: Mapping[str, Any], ref: Mapping[str, Any]) -> dict[str, Any]:
    normalized_ref = _require_event_ref(ref, label="selected plan event")
    events = manifest.get("events")
    if not isinstance(events, list):
        raise CrossoverRunnerError("manifest events are missing or malformed")
    matches = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event_index") == normalized_ref["event_index"]
        and event.get("event_id") == normalized_ref["event_id"]
    ]
    if len(matches) != 1:
        index_matches = [
            event
            for event in events
            if isinstance(event, Mapping)
            and event.get("event_index") == normalized_ref["event_index"]
        ]
        if len(index_matches) == 1:
            _identity_mismatch(
                "manifest event lookup",
                "event_id",
                normalized_ref["event_id"],
                index_matches[0].get("event_id"),
            )
        raise CrossoverRunnerError(
            "manifest lacks exactly one event matching "
            f"event_index={normalized_ref['event_index']!r}, "
            f"event_id={normalized_ref['event_id']!r}; observed_count={len(matches)}"
        )
    event = dict(matches[0])
    observed_projection = _manifest_event_projection(
        event, label=f"manifest event {normalized_ref['event_id']}"
    )
    for key in MANIFEST_IDENTITY_FIELDS:
        if observed_projection[key] != normalized_ref[key]:
            _identity_mismatch(
                f"manifest event {normalized_ref['event_id']}",
                key,
                normalized_ref[key],
                observed_projection[key],
            )
    return event


def _extract_plan_events(plan: Mapping[str, Any], *, shard_id: str, shard_index: int) -> list[dict[str, Any]]:
    global_events_raw = plan.get("events")
    if not isinstance(global_events_raw, list):
        one = plan.get("event")
        global_events_raw = [one] if isinstance(one, Mapping) else []
    global_events = [_require_event_ref(event, label=f"plan.events[{index}]") for index, event in enumerate(global_events_raw)]
    if [event["event_index"] for event in global_events] != sorted(event["event_index"] for event in global_events):
        raise CrossoverRunnerError("plan events are not in exact event-index order")
    shards = plan.get("shards")
    selected_raw: Any = None
    if isinstance(shards, list):
        selected = [item for item in shards if isinstance(item, Mapping) and item.get("shard_id") == shard_id]
        if len(selected) != 1 and 0 <= shard_index < len(shards):
            candidate = shards[shard_index]
            selected = [candidate] if isinstance(candidate, Mapping) else []
        if len(selected) == 1:
            selected_raw = selected[0]
    elif isinstance(shards, Mapping):
        selected_raw = shards.get(shard_id, shards.get(str(shard_index)))
    if selected_raw is None:
        selected_raw = {"events": global_events}
    if not isinstance(selected_raw, Mapping):
        raise CrossoverRunnerError("plan shard entry is malformed")
    refs_raw = selected_raw.get("events")
    if refs_raw is None:
        indices = selected_raw.get("event_indices")
        if isinstance(indices, list):
            by_index = {event["event_index"]: event for event in global_events}
            refs_raw = [by_index.get(index) for index in indices]
    if not isinstance(refs_raw, list):
        raise CrossoverRunnerError("plan shard lacks an events list")
    refs: list[dict[str, Any]] = []
    for index, ref in enumerate(refs_raw):
        if not isinstance(ref, Mapping):
            raise CrossoverRunnerError(f"plan shard event {index} is malformed")
        normalized = _require_event_ref(ref, label=f"plan shard events[{index}]")
        if normalized not in global_events:
            raise CrossoverRunnerError("plan shard event is not in global plan order")
        refs.append(normalized)
    if refs != sorted(refs, key=lambda item: item["event_index"]):
        raise CrossoverRunnerError("plan shard events are not in exact order")
    return refs


@dataclass(frozen=True)
class CrossoverPreflight:
    plan: dict[str, Any]
    plan_sha256: str
    plan_self_sha256: str
    plan_source_sha256: str
    manifest: dict[str, Any]
    manifest_sha256: str
    manifest_path: str
    event: dict[str, Any]
    shard_id: str
    shard_index: int
    output_root: str
    pre_gpu_receipt: dict[str, Any]
    pre_gpu_receipt_path: str
    pre_gpu_receipt_raw_sha256: str
    pre_gpu_receipt_self_sha256: str
    source_paths: dict[str, str]
    source_hashes: dict[str, str]


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CrossoverRunnerError(f"{label} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CrossoverRunnerError(f"{label} must be a lowercase SHA-256") from exc
    if value != value.lower():
        raise CrossoverRunnerError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute_absent(path_value: Any, label: str) -> Path:
    if not isinstance(path_value, (str, Path)) or not str(path_value):
        raise CrossoverRunnerError(f"{label} path is missing")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise CrossoverRunnerError(f"{label} path must be absolute")
    cursor = path
    while True:
        if cursor.is_symlink():
            raise CrossoverRunnerError(f"{label} path traverses a symlink: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    resolved = path.resolve(strict=False)
    if resolved.exists() or resolved.is_symlink():
        raise CrossoverRunnerError(f"{label} must be absent and non-symlink: {resolved}")
    return resolved


def _path_ref_value(value: Any, label: str, *, directory: bool | None = None) -> tuple[str, str]:
    """Resolve one sealer path reference and recompute its raw identity."""

    if not isinstance(value, Mapping):
        raise CrossoverRunnerError(f"{label} must be a path reference")
    raw_path = value.get("path") or value.get("file") or value.get("root")
    if not isinstance(raw_path, str) or not raw_path:
        raise CrossoverRunnerError(f"{label}.path is missing")
    path = Path(raw_path).expanduser()
    if path.is_symlink():
        raise CrossoverRunnerError(f"{label} must not be a symlink: {path}")
    if directory is True:
        path = _regular_directory(path, label)
        observed = _sha256_directory(path)
    elif directory is False:
        path = _regular_file(path, label)
        observed = _sha256_file(path)
    else:
        if path.is_dir():
            path = _regular_directory(path, label)
            observed = _sha256_directory(path)
        else:
            path = _regular_file(path, label)
            observed = _sha256_file(path)
    declared = value.get("sha256")
    if declared is not None and _require_sha256(declared, f"{label}.sha256") != observed:
        raise CrossoverRunnerError(f"{label} raw SHA-256 differs from the sealed binding")
    return str(path), observed


def _require_exact_operator_contract(plan: Mapping[str, Any]) -> None:
    try:
        from scripts.research import materialize_s_k10_h20_crossover_plan as planner
    except Exception as exc:  # pragma: no cover - import is deterministic in repo
        raise CrossoverRunnerError(f"cannot import crossover plan materializer: {exc}") from exc
    if plan.get("schema_version") != planner.SCHEMA_VERSION or plan.get("status") != "planned" or plan.get("unit_id") != UNIT_ID:
        raise CrossoverRunnerError("plan schema/status/unit identity drifted")
    if plan.get("primary") != dict(planner.PRIMARY):
        raise CrossoverRunnerError("plan primary checkpoint/step/substrate differs from S step 2444")
    if plan.get("cell_order") != list(CELLS) or plan.get("technical_control") != TECHNICAL_CELL:
        raise CrossoverRunnerError("plan cell order/technical control is not C00,C10,C01,C11/C00")
    if plan.get("event_count") != 3 or plan.get("image_count") != 3:
        raise CrossoverRunnerError("plan must bind exactly three events and three images")
    if plan.get("events") is None or not isinstance(plan["events"], list) or len(plan["events"]) != 3:
        raise CrossoverRunnerError("plan must bind exactly three selected events")
    expected_ids = tuple(planner.EVENT_IDS)
    actual_ids = tuple(event.get("event_id") for event in plan["events"] if isinstance(event, Mapping))
    if actual_ids != expected_ids:
        raise CrossoverRunnerError("plan selected event order differs from frozen materializer")
    if len({event.get("image_id") for event in plan["events"] if isinstance(event, Mapping)}) != 3:
        raise CrossoverRunnerError("plan selected events must have three distinct images")
    device_plan = plan.get("device_plan")
    if (
        not isinstance(device_plan, Mapping)
        or set(device_plan) != {f"shard-{index:03d}" for index in range(SHARD_COUNT)}
        or any(
            not isinstance(device_plan[shard_id], str) or not device_plan[shard_id]
            for shard_id in device_plan
        )
    ):
        raise CrossoverRunnerError("plan device assignment must bind exactly three physical devices")
    shards = plan.get("shards")
    if not isinstance(shards, list) or len(shards) != SHARD_COUNT:
        raise CrossoverRunnerError("plan must contain exactly three shards")
    for index, shard_doc in enumerate(shards):
        expected_id = f"shard-{index:03d}"
        if not isinstance(shard_doc, Mapping):
            raise CrossoverRunnerError(f"plan {expected_id} entry is malformed")
        if (
            shard_doc.get("shard_index") != index
            or shard_doc.get("shard_id") != expected_id
            or shard_doc.get("physical_device") != device_plan[expected_id]
            or shard_doc.get("logical_device") != "cuda:0"
        ):
            raise CrossoverRunnerError(f"plan {expected_id} device identity drifted")
        events = shard_doc.get("events")
        if not isinstance(events, list) or len(events) != 1 or events[0] != shard_doc.get("event") or events[0] != plan["events"][index]:
            raise CrossoverRunnerError(f"plan {expected_id} must bind its exact one event")
    operator = plan.get("operator_contract")
    expected_operator = {
        "admission_mode": planner.OPENER_MODE,
        "opener_injected": False,
        "opener_generated_by_model": True,
        "use_cache": False,
        "max_rows": planner.MAX_ROWS,
        "max_row_tokens": planner.MAX_ROW_TOKENS,
        "full_endpoint_vectors_required": True,
        "no_training": True,
    }
    if not isinstance(operator, Mapping) or dict(operator) != expected_operator:
        raise CrossoverRunnerError("plan operator contract differs from natural no-cache max_rows=3/max_row_tokens=256")
    for key, expected in planner.FROZEN_FLAGS.items():
        if plan.get(key) is not expected:
            raise CrossoverRunnerError(f"plan frozen flag {key} differs")
    if plan.get("no_legacy_no_2x2_reuse") is not True or plan.get("no_legacy_plan_reselection") is not True:
        raise CrossoverRunnerError("plan legacy reuse/reselection guard is missing")
    source_bindings = plan.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != {"evidence", "receipt", "manifest", "census", "original_plan", "gate_result"}:
        raise CrossoverRunnerError("plan source bindings are incomplete")


TECHNICAL_CELL = "C00"


def validate_preflight(
    plan: str | Path | Mapping[str, Any],
    pre_gpu_receipt: str | Path | Mapping[str, Any],
    output_root: str | Path,
    *,
    shard_id: str | int,
    manifest: str | Path | Mapping[str, Any] | None = None,
    census: str | Path | None = None,
    config: str | Path | None = None,
    panel: str | Path | None = None,
    cohort: str | Path | None = None,
    h0_root: str | Path | None = None,
    h0_dir: str | Path | None = None,
    cohort_manifest: str | Path | None = None,
    base_model_dir: str | Path | None = None,
) -> CrossoverPreflight:
    """Validate all source and identity bindings without importing a model."""

    output_source = Path(output_root).expanduser()
    if output_source.is_symlink() or output_source.exists():
        raise CrossoverRunnerError(f"fresh output root must be absent and non-symlink: {output_source}")
    output_requested = output_source.resolve(strict=False)
    if not output_requested.is_absolute():  # pragma: no cover - resolve of Path is absolute
        raise CrossoverRunnerError("output root must be absolute")

    # Both the planner and the pre-GPU sealer are authoritative.  Their
    # validators re-read the exact bound source paths and reject byte drift;
    # this runner only consumes their returned documents after those checks.
    if isinstance(plan, Mapping):
        raise CrossoverRunnerError("crossover plan must be supplied as its canonical materialized path")
    if isinstance(pre_gpu_receipt, Mapping):
        raise CrossoverRunnerError("pre-GPU receipt must be supplied as its canonical sealed path")
    plan_path = _regular_file(plan, "crossover plan")
    receipt_path = _regular_file(pre_gpu_receipt, "pre-GPU receipt")
    try:
        from scripts.research import materialize_s_k10_h20_crossover_plan as planner
        from scripts.research import seal_s_k10_h20_crossover_pre_gpu_receipt as sealer
    except Exception as exc:  # pragma: no cover - repository imports are deterministic
        raise CrossoverRunnerError(f"cannot import crossover CPU validators: {exc}") from exc
    try:
        planner_info = planner.validate_plan(plan_path)
    except Exception as exc:
        raise CrossoverRunnerError(f"crossover plan deterministic validation failed: {exc}") from exc
    plan_doc, plan_info = _read_json(plan_path, "crossover plan")
    if planner_info.get("plan") != plan_doc:
        raise CrossoverRunnerError("materializer returned a plan different from the canonical plan path")
    _require_exact_operator_contract(plan_doc)
    plan_sha = str(planner_info.get("plan_sha256") or plan_doc.get("self_sha256"))
    plan_self = str(planner_info.get("self_sha256") or plan_doc.get("self_sha256"))
    _require_sha256(plan_sha, "plan_sha256")
    _require_sha256(plan_self, "plan_self_sha256")

    shard, shard_index = _normalize_shard_id(shard_id)

    try:
        validate_receipt = sealer.validate_pre_gpu_receipt
        validation_kwargs = {
            "phase": "runtime",
            "shard_id": shard,
            "output_root": str(output_requested),
        }
        try:
            signature = inspect.signature(validate_receipt)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if accepts_kwargs:
                sealer_info = validate_receipt(receipt_path, **validation_kwargs)
            else:
                accepted = {
                    name: value
                    for name, value in validation_kwargs.items()
                    if name in signature.parameters
                }
                sealer_info = validate_receipt(receipt_path, **accepted)
        except (TypeError, ValueError):
            sealer_info = validate_receipt(receipt_path)
    except Exception as exc:
        raise CrossoverRunnerError(f"pre-GPU receipt validation failed: {exc}") from exc
    if not isinstance(sealer_info, Mapping) or not isinstance(sealer_info.get("receipt"), Mapping):
        raise CrossoverRunnerError("pre-GPU validator did not return receipt document")
    receipt_doc = dict(sealer_info["receipt"])
    if receipt_doc.get("schema_version") != getattr(sealer, "SCHEMA_VERSION", None) or receipt_doc.get("status") != getattr(sealer, "STATUS", None) or receipt_doc.get("unit_id") != UNIT_ID:
        raise CrossoverRunnerError("pre-GPU receipt schema/status/unit identity drifted")
    receipt_self = _require_sha256(sealer_info.get("self_sha256"), "pre_gpu_receipt_self_sha256")
    if receipt_doc.get("self_sha256") != receipt_self:
        raise CrossoverRunnerError("pre-GPU validator self hash differs from receipt")
    receipt_raw_sha = _sha256_file(receipt_path)
    if sealer_info.get("path") not in {None, str(receipt_path)}:
        raise CrossoverRunnerError("pre-GPU validator path differs from supplied canonical receipt path")
    plan_ref = receipt_doc.get("plan")
    if not isinstance(plan_ref, Mapping) or plan_ref.get("path") != str(plan_path):
        raise CrossoverRunnerError("pre-GPU receipt is not bound to the supplied canonical plan path")
    if receipt_doc.get("plan_self_sha256") != plan_self:
        raise CrossoverRunnerError("pre-GPU receipt plan self hash differs from validated plan")

    # The sealer reserves one parent execution root.  A shard may only write
    # its own exact child; arbitrary fresh roots would sever receipt lineage.
    roots = receipt_doc.get("roots")
    if not isinstance(roots, Mapping) or not isinstance(roots.get("execution_root"), Mapping):
        raise CrossoverRunnerError("pre-GPU receipt execution_root binding is missing")
    execution_root = roots["execution_root"].get("path")
    if roots["execution_root"].get("status") != "reserved_absent_pre_gpu":
        raise CrossoverRunnerError("pre-GPU execution root is not reserved absent before GPU")
    execution_parent = Path(str(execution_root)).expanduser().resolve(strict=False)

    device_plan = plan_doc.get("device_plan")
    receipt_device_plan = receipt_doc.get("device_plan")
    if not isinstance(device_plan, Mapping) or not isinstance(receipt_device_plan, Mapping) or dict(receipt_device_plan) != dict(device_plan):
        raise CrossoverRunnerError("pre-GPU receipt device plan differs from validated plan")
    expected_physical = device_plan.get(shard)
    shard_doc = plan_doc["shards"][shard_index]
    if shard_doc.get("physical_device") != expected_physical:
        raise CrossoverRunnerError("plan shard physical device differs from device_plan")
    expected_output = (execution_parent / shard).resolve(strict=False)
    if output_requested != expected_output:
        raise CrossoverRunnerError("output root must be exactly execution_root/shard_id from pre-GPU receipt")

    # All ten loader inputs are explicit sealer bindings.  CLI values are
    # optional only where the receipt can supply the exact path/hash itself;
    # an absent binding is an error, never an inferred sibling or glob.
    bindings = receipt_doc.get("input_bindings")
    if not isinstance(bindings, Mapping):
        raise CrossoverRunnerError("pre-GPU receipt input_bindings are missing")
    required_inputs = (
        "manifest",
        "census",
        "execution_plan",
        "config",
        "panel",
        "cohort",
        "cohort_manifest",
        "h0_root",
        "h0_dir",
        "base_model_dir",
    )
    missing_bindings = [key for key in required_inputs if key not in bindings]
    if missing_bindings:
        raise CrossoverRunnerError("pre-GPU bridge missing bound input fields: " + ", ".join(missing_bindings))
    if isinstance(manifest, Mapping):
        raise CrossoverRunnerError("source manifest must be supplied as its canonical path")
    explicit: dict[str, str | Path | None] = {
        "manifest": manifest,
        "census": census,
        "config": config,
        "panel": panel,
        "cohort": cohort,
        "cohort_manifest": cohort_manifest,
        "h0_root": h0_root,
        "h0_dir": h0_dir,
        "base_model_dir": base_model_dir,
    }
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    directory_inputs = {"h0_root", "h0_dir", "base_model_dir"}
    for key in required_inputs:
        sealed_path, sealed_hash = _path_ref_value(bindings[key], f"pre-GPU input {key}", directory=key in directory_inputs)
        candidate = explicit.get(key)
        if candidate is not None:
            candidate_path = (_regular_directory(candidate, key) if key in directory_inputs else _regular_file(candidate, key))
            if str(candidate_path) != sealed_path:
                raise CrossoverRunnerError(f"explicit {key} path differs from pre-GPU binding")
        source_paths[key] = sealed_path
        source_hashes[key] = sealed_hash
    plan_manifest_ref = plan_doc.get("source_bindings", {}).get("manifest")
    plan_census_ref = plan_doc.get("source_bindings", {}).get("census")
    if not isinstance(plan_manifest_ref, Mapping) or source_paths["manifest"] != plan_manifest_ref.get("path"):
        raise CrossoverRunnerError("manifest path differs from deterministic plan source binding")
    if not isinstance(plan_census_ref, Mapping) or source_paths["census"] != plan_census_ref.get("path"):
        raise CrossoverRunnerError("census path differs from deterministic plan source binding")
    if source_hashes["manifest"] != plan_manifest_ref.get("raw_sha256"):
        raise CrossoverRunnerError("manifest bytes differ from deterministic plan source binding")
    if source_hashes["census"] != plan_census_ref.get("raw_sha256"):
        raise CrossoverRunnerError("census bytes differ from deterministic plan source binding")
    source_paths["pre_gpu_receipt"] = str(receipt_path)
    source_hashes["pre_gpu_receipt"] = receipt_raw_sha
    source_hashes["pre_gpu_receipt_self_sha256"] = receipt_self
    source_hashes["plan_file_sha256"] = plan_info["sha256"]
    source_hashes["plan_sha256"] = plan_sha

    manifest_path = source_paths["manifest"]
    manifest_doc, manifest_info = _read_json(manifest_path, "source manifest")
    if manifest_info["sha256"] != source_hashes["manifest"]:
        raise CrossoverRunnerError("source manifest bytes changed during preflight")
    # Every selected event carries the plan's prefix and geometry digests.
    refs = _extract_plan_events(plan_doc, shard_id=shard, shard_index=shard_index)
    if len(refs) != 1:
        raise CrossoverRunnerError("one crossover event root is required per shard")
    planned_event = _require_event_ref(
        plan_doc["events"][shard_index], label=f"plan.events[{shard_index}]"
    )
    if planned_event != refs[0]:
        raise CrossoverRunnerError(
            f"plan {shard} event projection differs from global plan event"
        )
    event = _event_from_manifest(manifest_doc, refs[0])
    for key, expected in (
        ("checkpoint", CHECKPOINT),
        ("step", STEP),
        ("substrate", "four-coordinate geo_sorted_xy"),
    ):
        if event.get(key) != expected:
            _identity_mismatch(
                f"manifest event {event['event_id']}", key, expected, event.get(key)
            )

    return CrossoverPreflight(
        plan=dict(plan_doc),
        plan_sha256=plan_sha,
        plan_self_sha256=plan_self,
        plan_source_sha256=plan_info["sha256"],
        manifest=dict(manifest_doc),
        manifest_sha256=manifest_info["sha256"],
        manifest_path=manifest_path,
        event=event,
        shard_id=shard,
        shard_index=shard_index,
        output_root=str(output_requested),
        pre_gpu_receipt=receipt_doc,
        pre_gpu_receipt_path=str(receipt_path),
        pre_gpu_receipt_raw_sha256=receipt_raw_sha,
        pre_gpu_receipt_self_sha256=receipt_self,
        source_paths=source_paths,
        source_hashes=source_hashes,
    )


def _recursive_mappings(value: Any) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        found.append(value)
        for item in value.values():
            found.extend(_recursive_mappings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_recursive_mappings(item))
    return found


def _first_mapping(value: Any, keys: Sequence[str]) -> Mapping[str, Any] | None:
    for mapping in _recursive_mappings(value):
        if all(key in mapping for key in keys):
            return mapping
    return None


ATTESTATION_KEYS = ("layer_consumption_attestation", "all_layer_consumption_attestation")
# This unit builds K01/K10/H20 (``base``) and the composed C11 receipt
# (``composed``).  The additive-dose ``declared_layer_count`` placeholder
# belongs to K14 in the other unit and must stay unrecognized here so it cannot
# enter a crossover receipt unnoticed.
ACCEPTED_CONSTRUCTION_PLACEHOLDER_KINDS = frozenset({"base", "composed"})


def _construction_placeholder_kind(value: Any, *, layer_count: int) -> str | None:
    kind = attention.construction_consumption_placeholder_kind(
        value, expected_layer_count=layer_count
    )
    return kind if kind in ACCEPTED_CONSTRUCTION_PLACEHOLDER_KINDS else None


def _runtime_attestation_sites(receipt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the attestation sites a live forward overwrites with evidence.

    The gate binds its observed all-layer receipt to the scalar call receipt and
    to the actuator receipt it wrapped.  A composed C11 mask keeps its K10/H20
    ``children`` receipts verbatim, so those nested placeholders are never
    promoted here and can never stand in for runtime proof.
    """

    containers: list[Any] = [receipt, receipt.get("attention_actuation_receipt")]
    attention_mask = receipt.get("attention_mask")
    if isinstance(attention_mask, Mapping):
        containers.append(attention_mask.get("receipt"))
    sites: list[Mapping[str, Any]] = []
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in ATTESTATION_KEYS:
            value = container.get(key)
            if isinstance(value, Mapping):
                sites.append(value)
    return sites


def require_unattested_consumption(
    receipt: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require both pre-forward keys to be an exact construction placeholder.

    A looser check here would seal a preflight receipt whose consumption
    metadata the live runner and the finalizer would later reject.
    """

    validated: list[dict[str, Any]] = []
    for key in ATTESTATION_KEYS:
        value = receipt.get(key)
        if _construction_placeholder_kind(value, layer_count=28) is None:
            raise CrossoverTechnicalInvalid(
                f"{label}.{key} does not retain the pre-forward "
                "required/unattested consumption contract"
            )
        validated.append(dict(value))
    return validated[0], validated[1]


def _validate_natural_result(
    result: Mapping[str, Any],
    *,
    label: str,
    expected_mask_sha256: str | None = None,
    require_mask: bool = False,
    require_layer_count: int = 28,
) -> dict[str, Any]:
    raw = _canonical_detached(result, label=label)
    if raw.get("admission_mode") not in {None, "pre_opener_natural"}:
        raise CrossoverTechnicalInvalid(f"{label} did not use pre_opener_natural admission")
    if raw.get("opener_injected") is True or raw.get("synthetic_opener_injections", 0) != 0:
        raise CrossoverTechnicalInvalid(f"{label} injected an opener")
    if raw.get("use_cache") is True:
        raise CrossoverTechnicalInvalid(f"{label} enabled cache")
    scalar = raw.get("runtime_scalar_receipts", raw.get("scalar_receipts"))
    if scalar is None:
        scalar = []
    if not isinstance(scalar, list):
        raise CrossoverTechnicalInvalid(f"{label} scalar receipts are malformed")
    if "runtime_scalar_forward_count" in raw and raw["runtime_scalar_forward_count"] != len(scalar):
        raise CrossoverTechnicalInvalid(f"{label} scalar forward count differs from receipts")
    if "scalar_forward_count" in raw and not isinstance(raw["scalar_forward_count"], int):
        raise CrossoverTechnicalInvalid(f"{label} scalar_forward_count is malformed")
    if require_mask and not scalar:
        raise CrossoverTechnicalInvalid(f"{label} lacks scalar mask-consumption receipts")
    mask_hashes: set[str] = set()
    attestations: list[Mapping[str, Any]] = []
    runtime_sites: list[Mapping[str, Any]] = []
    token_hash_count = 0
    mrope_hash_count = 0
    finite_logit_count = 0
    for index, receipt in enumerate(scalar):
        if not isinstance(receipt, Mapping):
            raise CrossoverTechnicalInvalid(f"{label} scalar receipt {index} is malformed")
        if receipt.get("use_cache") is not False:
            raise CrossoverTechnicalInvalid(f"{label} scalar receipt {index} did not set use_cache=false")
        if receipt.get("input_ids_sha256") or receipt.get("prefix_token_ids_sha256") or receipt.get("input_token_hash"):
            token_hash_count += 1
        if receipt.get("mrope_hash") or receipt.get("mrope_sha256") or receipt.get("position_ids_sha256"):
            mrope_hash_count += 1
        for mapping in _recursive_mappings(receipt):
            for key, value in mapping.items():
                if "mask_sha256" in str(key) or str(key) in {"mask_hash", "attention_mask_hash"}:
                    if isinstance(value, str):
                        mask_hashes.add(value)
                if str(key) in {"logits_finite", "finite_logits", "finite"} and value is False:
                    raise CrossoverTechnicalInvalid(f"{label} scalar receipt {index} contains non-finite logits")
                if str(key) in {"logits_finite", "finite_logits", "finite"} and value is True:
                    finite_logit_count += 1
                if str(key) in ATTESTATION_KEYS and isinstance(value, Mapping):
                    attestations.append(value)
        runtime_sites.extend(_runtime_attestation_sites(receipt))
    if require_mask:
        if expected_mask_sha256 is not None and expected_mask_sha256 not in mask_hashes:
            raise CrossoverTechnicalInvalid(f"{label} scalar receipts do not bind the exact mask hash")
        for item in attestations:
            # A real C11 compose receipt carries its K10/H20 children's
            # construction placeholders, which the gate never overwrites.  They
            # are accepted only in their exact pre-forward shape and never
            # count as consumption evidence; every other attestation must be a
            # passed all-N-layer runtime receipt.
            if _construction_placeholder_kind(item, layer_count=require_layer_count) is not None:
                continue
            if item.get("passed") is not True:
                raise CrossoverTechnicalInvalid(f"{label} lacks passed all-layer consumption attestation")
            layer_count = item.get("layer_count")
            layer_indices = item.get("layer_indices")
            if layer_count is not None and layer_count != require_layer_count:
                raise CrossoverTechnicalInvalid(f"{label} consumption attestation is not all-{require_layer_count}-layer")
            if layer_count is None and (not isinstance(layer_indices, list) or len(layer_indices) != require_layer_count):
                raise CrossoverTechnicalInvalid(f"{label} consumption attestation lacks all-{require_layer_count}-layer identity")
        if not any(
            _construction_placeholder_kind(item, layer_count=require_layer_count) is None
            for item in runtime_sites
        ):
            raise CrossoverTechnicalInvalid(
                f"{label} lacks a runtime all-{require_layer_count}-layer consumption attestation"
            )
        if token_hash_count != len(scalar) or mrope_hash_count != len(scalar):
            raise CrossoverTechnicalInvalid(f"{label} scalar receipts lack token/prefix/MRoPE hashes")
    elif scalar and (token_hash_count != len(scalar) or mrope_hash_count != len(scalar)):
        raise CrossoverTechnicalInvalid(f"{label} scalar receipts lack token/prefix/MRoPE hashes")
    if scalar and finite_logit_count and finite_logit_count < len(scalar):
        raise CrossoverTechnicalInvalid(f"{label} scalar receipts lack finite-logit attestations")
    return raw


def _validate_c00_parity(result: Mapping[str, Any]) -> None:
    candidates = [
        result.get("full_logit_parity"),
        result.get("c00_parity"),
        result.get("parity"),
    ]
    parity = next((item for item in candidates if isinstance(item, Mapping)), None)
    if parity is None or parity.get("passed") is not True:
        raise CrossoverTechnicalInvalid("C00 explicit K01 control lacks passed full-vocabulary parity")
    delta = parity.get("per_forward_max_abs_delta", parity.get("max_abs_logit_delta", parity.get("max_abs_delta", 0.0)))
    if not isinstance(delta, (int, float)) or float(delta) > NOOP_TOLERANCE:
        raise CrossoverTechnicalInvalid(f"C00 full-vocabulary parity exceeded {NOOP_TOLERANCE}")
    if parity.get("candidate_step_count") is not None and parity.get("reference_step_count") != parity.get("candidate_step_count"):
        raise CrossoverTechnicalInvalid("C00 parity trajectory lengths differ")


def _factory_kwargs(factory: Any) -> dict[str, Any]:
    names = (
        "image_key_positions",
        "b_exclusive_positions",
        "a_exclusive_positions",
        "background_positions",
        "same_class_competitor_positions",
        "latest_terminal_key_positions",
        "latest_row_key_positions",
        "row_major_order",
        "layer_count",
        "head_count",
        "device",
        "dtype",
        "max_sequence_length",
    )
    result = {name: getattr(factory, name) for name in names if hasattr(factory, name)}
    return result


class _C11CompositionCallback:
    """Fresh per-step K10/H20 factory construction for the C11 transport arm."""

    protocol = "s_k10_h20_crossover_c11_callback.v1"

    def __init__(self, k10_factory: Any, h20_factory: Any) -> None:
        self.k10_kwargs = _factory_kwargs(k10_factory)
        self.h20_kwargs = _factory_kwargs(h20_factory)
        self.receipts: list[dict[str, Any]] = []
        self.last_receipt: dict[str, Any] | None = None

    def __call__(
        self,
        *,
        sequence_length: int | None = None,
        query_position: int | None = None,
        device: torch.device | str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Build one fresh C11 mask from the gate's scalar-step protocol.

        ``_arm_attention_callback`` dispatches callbacks with keyword-only
        ``sequence_length``/``query_position``/``device`` when their
        signature has no positional parameters.  Keep this seam operator
        neutral: contextual gate metadata is accepted and ignored, while all
        three scalar fields are validated before rebuilding the K10/H20
        factories for this one step.
        """

        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
            raise CrossoverTechnicalInvalid(
                "C11 callback requires integer sequence_length"
            )
        if sequence_length <= 0:
            raise CrossoverTechnicalInvalid(
                "C11 callback requires positive sequence_length"
            )
        if isinstance(query_position, bool) or not isinstance(query_position, int):
            raise CrossoverTechnicalInvalid(
                "C11 callback requires integer query_position"
            )
        if not 0 <= query_position < sequence_length:
            raise CrossoverTechnicalInvalid(
                "C11 callback query_position must be within sequence_length"
            )
        if device is None:
            raise CrossoverTechnicalInvalid("C11 callback requires device")
        if not isinstance(device, (str, torch.device)):
            raise CrossoverTechnicalInvalid(
                "C11 callback device must be a torch.device or string"
            )
        try:
            target_device = torch.device(device)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise CrossoverTechnicalInvalid(
                f"C11 callback device is invalid: {device!r}"
            ) from exc

        k_factory = attention.build_scalar_step_factory("K10", **self.k10_kwargs)
        h_factory = attention.build_scalar_step_factory("H20", **self.h20_kwargs)
        k10 = k_factory.build(
            sequence_length,
            query_position=query_position,
            device=target_device,
        )
        h20 = h_factory.build(
            sequence_length,
            query_position=query_position,
            device=target_device,
        )
        composed = attention.compose_k10_h20(k10, h20, cell_id="C11")
        receipt = composed.receipt()
        receipt["transport_arm_id"] = "K10"
        receipt["inner_actuator_id"] = "C11"
        receipt["composition_arm_id"] = "C11"
        self.last_receipt = receipt
        self.receipts.append(dict(receipt))
        return {
            "attention_mask": composed.attention_mask,
            "actuator_id": "C11",
            "protocol": self.protocol,
            "receipt": receipt,
            "score_bias": None,
            "score_bias_callback": None,
        }


def _call_event_runner(runner: Callable[..., Any], event: Mapping[str, Any], preflight: CrossoverPreflight) -> Any:
    try:
        signature = inspect.signature(runner)
        kwargs = {
            "event": event,
            "preflight": preflight,
            "shard_id": preflight.shard_id,
        }
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return runner(**kwargs)
        accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
        if accepted:
            return runner(**accepted)
        return runner(event)
    except (TypeError, ValueError):
        return runner(event)


def _normalize_injected_outputs(value: Any) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise CrossoverTechnicalInvalid("injected crossover event runner must return an object")
    technical = value.get("technical_control")
    cells_value = value.get("cells")
    runtime_identity = value.get("runtime_identity", {})
    if isinstance(technical, Mapping) and isinstance(cells_value, Mapping):
        technical_raw = technical.get("K00", {}).get("result") if isinstance(technical.get("K00"), Mapping) else None
        cells = {
            cell: cells_value[cell].get("result") if isinstance(cells_value[cell], Mapping) else cells_value[cell]
            for cell in CELLS
            if cell in cells_value
        }
        if not isinstance(technical_raw, Mapping) or set(cells) != set(CELLS):
            raise CrossoverTechnicalInvalid("injected outputs have incomplete technical/cell set")
        return technical_raw, cells, dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {}
    arms = value.get("arms")
    if isinstance(arms, Mapping):
        key_map = {"K00": "K00", "K01": "C00", "K10": "C10", "H20": "C01"}
        if not all(key in arms and isinstance(arms[key], Mapping) for key in key_map):
            raise CrossoverTechnicalInvalid("injected arms lack K00/K01/K10/H20 outputs")
        cells = {cell: arms[source] for source, cell in key_map.items() if cell != "K00"}
        # An injected C11 can be supplied explicitly for CPU composition tests.
        c11 = value.get("C11") or value.get("c11")
        if isinstance(c11, Mapping):
            cells["C11"] = c11
        if "C11" not in cells:
            raise CrossoverTechnicalInvalid("injected outputs lack C11")
        return arms["K00"], cells, dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {}
    raise CrossoverTechnicalInvalid("injected event output must provide technical_control/cells or arms")


def _derive_pre_gpu_runtime_identity(preflight: CrossoverPreflight) -> dict[str, Any]:
    """Consume the sealer's exact per-shard bridge; never rebuild one locally."""

    try:
        from scripts.research import seal_s_k10_h20_crossover_pre_gpu_receipt as sealer
    except Exception as exc:  # pragma: no cover - repository import is deterministic
        raise CrossoverTechnicalInvalid(f"cannot import pre-GPU runtime identity sealer: {exc}") from exc
    binder = getattr(sealer, "runtime_identity_binding", None)
    validator = getattr(sealer, "validate_runtime_identity", None)
    if not callable(binder) or not callable(validator):
        raise CrossoverTechnicalInvalid(
            "pre-GPU bridge missing authoritative runtime_identity_binding/validate_runtime_identity"
        )
    expected_physical = str(preflight.plan["device_plan"][preflight.shard_id])
    observed_physical = os.environ.get("CUDA_VISIBLE_DEVICES")
    if observed_physical != expected_physical:
        raise CrossoverTechnicalInvalid(
            f"runtime CUDA_VISIBLE_DEVICES={observed_physical!r} differs from sealed device {expected_physical!r}"
        )
    try:
        identity = binder(
            preflight.pre_gpu_receipt_path,
            receipt_path=preflight.pre_gpu_receipt_path,
            shard_id=preflight.shard_id,
            observed_cuda_visible_devices=observed_physical,
        )
    except Exception as exc:
        raise CrossoverTechnicalInvalid(f"pre-GPU runtime identity binding failed: {exc}") from exc
    if not isinstance(identity, Mapping):
        raise CrossoverTechnicalInvalid("pre-GPU runtime identity binding is not an object")
    identity_document = dict(identity)
    try:
        validator(
            identity_document,
            preflight.pre_gpu_receipt_path,
            receipt_path=preflight.pre_gpu_receipt_path,
            shard_id=preflight.shard_id,
            observed_cuda_visible_devices=observed_physical,
        )
    except Exception as exc:
        raise CrossoverTechnicalInvalid(f"pre-GPU runtime identity revalidation failed: {exc}") from exc
    required = {
        "input_paths",
        "input_hashes",
        "code_hashes",
        "runtime",
        "device_assignment",
        "pre_gpu_receipt_path",
        "pre_gpu_receipt_sha256",
        "pre_gpu_receipt_self_sha256",
    }
    missing = sorted(key for key in required if key not in identity_document)
    if missing:
        raise CrossoverTechnicalInvalid("pre-GPU bridge missing bound fields: " + ", ".join(missing))
    assignment = identity_document["device_assignment"]
    if not isinstance(assignment, Mapping) or assignment.get("shard_id") != preflight.shard_id or str(assignment.get("physical_device")) != expected_physical or assignment.get("observed_cuda_visible_devices") != expected_physical:
        raise CrossoverTechnicalInvalid("pre-GPU bridge device assignment differs from sealed physical device")
    if identity_document["pre_gpu_receipt_path"] != preflight.pre_gpu_receipt_path or identity_document["pre_gpu_receipt_self_sha256"] != preflight.pre_gpu_receipt_self_sha256:
        raise CrossoverTechnicalInvalid("pre-GPU bridge receipt path/self binding differs")
    return identity_document


def _execution_environment(preflight: CrossoverPreflight) -> dict[str, str | None]:
    expected_physical = str(preflight.plan["device_plan"][preflight.shard_id])
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing not in {None, expected_physical}:
        raise CrossoverTechnicalInvalid(
            f"CUDA_VISIBLE_DEVICES={existing!r} differs from sealed physical device {expected_physical!r}"
        )
    values = {
        "CUDA_VISIBLE_DEVICES": expected_physical,
        "S_NATURAL_BOUNDARY_MANIFEST": preflight.source_paths["manifest"],
        "S_NATURAL_BOUNDARY_CENSUS": preflight.source_paths["census"],
        "S_NATURAL_BOUNDARY_CONFIG": preflight.source_paths["config"],
        "S_NATURAL_BOUNDARY_PANEL": preflight.source_paths["panel"],
        "S_NATURAL_BOUNDARY_COHORT": preflight.source_paths["cohort"],
        "S_NATURAL_BOUNDARY_COHORT_MANIFEST": preflight.source_paths["cohort_manifest"],
        "S_NATURAL_BOUNDARY_H0_ROOT": preflight.source_paths["h0_root"],
        "S_NATURAL_BOUNDARY_H0_DIR": preflight.source_paths["h0_dir"],
        "S_NATURAL_BOUNDARY_PRE_GPU_RECEIPT": preflight.pre_gpu_receipt_path,
        "S_NATURAL_BOUNDARY_SHARD_ID": preflight.shard_id,
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: value for key, value in values.items() if value is not None})
    return previous


def _restore_execution_environment(previous: Mapping[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _require_admitted_history(
    exact_history_token_ids: Sequence[int], event: Mapping[str, Any]
) -> None:
    natural_boundary = event.get("natural_boundary")
    history_token_ids = (
        natural_boundary.get("history_token_ids")
        if isinstance(natural_boundary, Mapping)
        else None
    )
    if not isinstance(history_token_ids, list):
        raise CrossoverTechnicalInvalid(
            "admitted event lacks required natural_boundary.history_token_ids"
        )
    if tuple(exact_history_token_ids) != tuple(history_token_ids):
        raise CrossoverTechnicalInvalid(
            "natural context history differs from admitted event"
        )


def _prepare_live_event(
    executor: Any,
    preflight: CrossoverPreflight,
    runtime_identity: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    """Use existing executor protected validation/load as transport only."""

    if callable(getattr(executor, "prepare_crossover_event", None)):
        prepare = executor.prepare_crossover_event
        kwargs = {"preflight": preflight, "runtime_identity": runtime_identity}
        try:
            signature = inspect.signature(prepare)
            if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
                prepared = prepare(preflight.event, **kwargs)
            else:
                accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
                prepared = prepare(preflight.event, **accepted)
        except (TypeError, ValueError):
            prepared = prepare(preflight.event, preflight=preflight)
        if not isinstance(prepared, tuple) or len(prepared) != 6:
            raise CrossoverTechnicalInvalid("prepare_crossover_event must return executor,binding,gate,context,boundary,identity")
        return prepared  # type: ignore[return-value]
    live = importlib.import_module("scripts.research.s_natural_boundary_k_n_h_live_executor")
    gate = importlib.import_module("scripts.research.run_s_primary_natural_boundary_gate")
    cohort = importlib.import_module("scripts.research.run_s_natural_boundary_k_n_h_cohort")
    if not callable(getattr(executor, "_validate_inputs", None)) or not callable(getattr(executor, "_load_once", None)):
        raise CrossoverTechnicalInvalid("live executor lacks protected validation/load transport")
    target, row, paths = executor._validate_inputs(preflight.event, cohort.ARM_ORDER)
    executor._load_once(paths, runtime_identity)
    runtime_event = live._runtime_event(target, row)
    orchestrator = getattr(executor, "_orchestrator", None)
    adapter = getattr(executor, "_adapter", None)
    if orchestrator is None or adapter is None:
        raise CrossoverTechnicalInvalid("live executor did not expose loaded orchestrator/adapter")
    matches = [
        candidate
        for candidate in getattr(orchestrator, "events", ())
        if isinstance(candidate, Mapping) and live._is_exact_legacy_s_event(candidate, runtime_event)
    ]
    if len(matches) != 1:
        raise CrossoverTechnicalInvalid("event does not resolve to exactly one legacy S event")
    legacy = importlib.import_module("scripts.research.run_static_dynamic_owner_interface_experiment")
    seeded = legacy._make_event_context(adapter, matches[0], history_resolver=live._ledger_exact_history_resolver)
    runtime = getattr(seeded, "runtime", None)
    if runtime is None:
        raise CrossoverTechnicalInvalid("loaded event context lacks runtime")
    binding = gate.LiveRuntimeBinding(adapter=adapter, runtime=runtime, event=runtime_event, seeded_context=seeded, identity=getattr(executor, "_identity", {}))
    context, boundary, identity = gate.build_natural_event_context(
        binding,
        event_id=runtime_event.get("event_id"),
        max_rows=MAX_ROWS,
        max_row_tokens=MAX_ROW_TOKENS,
    )
    _require_admitted_history(context.exact_history_token_ids, preflight.event)
    gate_runner = gate.SPrimaryNaturalBoundaryGate(binding, context, boundary, identity, max_rows=MAX_ROWS, max_row_tokens=MAX_ROW_TOKENS)
    return executor, binding, gate_runner, context, boundary, dict(identity)


def _run_live_event(preflight: CrossoverPreflight, *, executor: Any, runtime_identity: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    _executor, binding, gate_runner, context, _boundary, identity = _prepare_live_event(executor, preflight, runtime_identity)
    gate = importlib.import_module("scripts.research.run_s_primary_natural_boundary_gate")
    factories = gate.build_live_attention_mask_actuators(binding, context)
    technical = gate_runner.run_arm(TECHNICAL_ARM)
    c00 = gate_runner.run_arm("K01", attention_mask_actuator=factories["K01"])
    c10 = gate_runner.run_arm("K10", attention_mask_actuator=factories["K10"])
    c01 = gate_runner.run_arm("H20", attention_mask_actuator=factories["H20"])
    c11_callback = _C11CompositionCallback(factories["K10"], factories["H20"])
    c11 = gate_runner.run_arm("K10", attention_mask_actuator=c11_callback)
    cells = {"C00": c00, "C10": c10, "C01": c01, "C11": c11}
    return technical, cells, identity


def _runtime_identity(preflight: CrossoverPreflight, identity: Mapping[str, Any] | None) -> dict[str, Any]:
    source_hashes = dict(preflight.source_hashes)
    source_hashes.update(
        {
            "plan_file_sha256": preflight.plan_source_sha256,
            "plan_sha256": preflight.plan_sha256,
            "plan_self_sha256": preflight.plan_self_sha256,
            "pre_gpu_receipt_sha256": preflight.pre_gpu_receipt_raw_sha256,
            "pre_gpu_receipt_self_sha256": preflight.pre_gpu_receipt_self_sha256,
        }
    )
    code_paths = {
        "runner": Path(__file__).resolve(),
        "attention_actuators": Path(attention.__file__).resolve(),
    }
    code_hashes = {name: _sha256_file(path) for name, path in code_paths.items() if path.is_file()}
    body = {
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "status": "completed",
        "unit_id": UNIT_ID,
        "shard_id": preflight.shard_id,
        "event_id": preflight.event["event_id"],
        "event_index": preflight.event["event_index"],
        "image_id": preflight.event["image_id"],
        "checkpoint": CHECKPOINT,
        "step": STEP,
        "device": {
            "logical_device": "cuda:0",
            "physical_device": str(preflight.plan["device_plan"][preflight.shard_id]),
        },
        "code_hashes": code_hashes,
        "input_hashes": source_hashes,
        "plan_sha256": preflight.plan_sha256,
        "plan_self_sha256": preflight.plan_self_sha256,
        "pre_gpu_receipt_self_sha256": preflight.pre_gpu_receipt_self_sha256,
    }
    if isinstance(identity, Mapping):
        supplied_code = identity.get("code_hashes")
        if isinstance(supplied_code, Mapping):
            code_hashes.update(
                {
                    str(name): _require_sha256(value, f"runtime_identity.code_hashes.{name}")
                    for name, value in supplied_code.items()
                }
            )
            body["code_hashes"] = code_hashes
        supplied_inputs = identity.get("input_hashes")
        if isinstance(supplied_inputs, Mapping):
            for name, value in supplied_inputs.items():
                if isinstance(value, str) and len(value) == 64:
                    source_hashes[str(name)] = value
            body["input_hashes"] = source_hashes
        for key in ("runtime", "model", "backend", "cuda", "model_identity", "device_assignment"):
            if key in identity:
                body[key] = _detached_json(identity[key], label=f"runtime_identity.{key}")
        assignment = identity.get("device_assignment")
        if isinstance(assignment, Mapping):
            body["device"] = {
                "logical_device": assignment.get("logical_device", "cuda:0"),
                "physical_device": str(assignment.get("physical_device")),
            }
    result = dict(body)
    result["self_sha256"] = sha256_json(body)
    return result


def _validate_and_wrap_outputs(
    preflight: CrossoverPreflight,
    technical: Mapping[str, Any],
    cells: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    technical_raw = _validate_natural_result(technical, label="technical K00", require_mask=False)
    if set(cells) != set(CELLS):
        raise CrossoverTechnicalInvalid("crossover cells are incomplete")
    expected_hashes: dict[str, str | None] = {}
    normalized: dict[str, Mapping[str, Any]] = {}
    for cell in CELLS:
        result = cells[cell]
        if not isinstance(result, Mapping):
            raise CrossoverTechnicalInvalid(f"{cell} result is malformed")
        # The operator receipt is required even when its natural scientific
        # endpoint is unmatched, invalid, or STOP.
        actuator_receipt = _first_mapping(result, ("mask_sha256", "mask_shape"))
        expected_hashes[cell] = actuator_receipt.get("mask_sha256") if actuator_receipt else None
        normalized[cell] = _validate_natural_result(
            result,
            label=cell,
            expected_mask_sha256=expected_hashes[cell],
            require_mask=True,
        )
    _validate_c00_parity(normalized["C00"])
    c11_receipt = _first_mapping(normalized["C11"], ("cell_id", "mask_sha256"))
    if c11_receipt is not None:
        if c11_receipt.get("cell_id") != "C11":
            raise CrossoverTechnicalInvalid("C11 result hides a non-C11 actuator identity")
        if c11_receipt.get("transport_arm_id") not in {None, "K10"}:
            raise CrossoverTechnicalInvalid("C11 transport arm identity differs from K10")
    return technical_raw, normalized


def _cell_document(cell: str, result: Mapping[str, Any]) -> dict[str, Any]:
    document = {
        "cell_id": cell,
        "transport_arm_id": TRANSPORT_ARMS[cell],
        "result": dict(result),
    }
    if cell == "C11":
        document["composition_arm_id"] = "C11"
    return document


def _terminal_document(
    preflight: CrossoverPreflight,
    result_document: Mapping[str, Any],
) -> dict[str, Any]:
    cells = result_document.get("cells", {})
    cell_bindings: dict[str, Any] = {}
    if isinstance(cells, Mapping):
        for cell in CELLS:
            value = cells.get(cell)
            raw = value.get("result") if isinstance(value, Mapping) else None
            if not isinstance(raw, Mapping):
                continue
            cell_bindings[cell] = {
                "cell_id": cell,
                "transport_arm_id": TRANSPORT_ARMS[cell],
                "composition_arm_id": "C11" if cell == "C11" else None,
                "admission_mode": raw.get("admission_mode"),
                "opener_injected": raw.get("opener_injected"),
                "terminal_reason": raw.get("terminal_reason", raw.get("stop_reason")),
                "stop_reason": raw.get("stop_reason"),
                "raw_result_sha256": sha256_json(_detached_json(raw, label=f"terminal.{cell}")),
            }
    body = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": "completed",
        "unit_id": UNIT_ID,
        "shard_id": preflight.shard_id,
        "event_id": preflight.event["event_id"],
        "event_index": preflight.event["event_index"],
        "image_id": preflight.event["image_id"],
        "result_sha256": result_document.get("result_sha256"),
        "cells": cell_bindings,
    }
    terminal = dict(body)
    terminal["self_sha256"] = sha256_json(body)
    return terminal


def _aggregate_receipt(
    preflight: CrossoverPreflight,
    *,
    result_raw_sha256: str,
    runtime_raw_sha256: str,
    terminal_raw_sha256: str,
    result_sha256: str,
    runtime_identity_sha256: str,
    terminal_self_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema_version": AGGREGATE_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "unit_id": UNIT_ID,
        "shard_id": preflight.shard_id,
        "event_id": preflight.event["event_id"],
        "event_index": preflight.event["event_index"],
        "image_id": preflight.event["image_id"],
        "plan_sha256": preflight.plan_sha256,
        "plan_self_sha256": preflight.plan_self_sha256,
        "pre_gpu_receipt_self_sha256": preflight.pre_gpu_receipt_self_sha256,
        "result_sha256": result_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "terminal_summary_self_sha256": terminal_self_sha256,
        "result_raw_sha256": result_raw_sha256,
        "runtime_identity_raw_sha256": runtime_raw_sha256,
        "terminal_summary_raw_sha256": terminal_raw_sha256,
    }
    receipt = dict(body)
    receipt["self_sha256"] = sha256_json(body)
    return receipt


def _persist_failure(output_root: Path, exc: BaseException) -> None:
    try:
        if output_root.is_symlink() or (output_root.exists() and not output_root.is_dir()):
            return
        output_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "status": "failed",
            "unit_id": UNIT_ID,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        failure["self_sha256"] = sha256_json({key: value for key, value in failure.items() if key != "self_sha256"})
        _write_once_json(output_root / "failure.json", failure)
        _write_once_bytes(output_root / "failure.stderr", (f"{type(exc).__name__}: {exc}\n").encode("utf-8", errors="backslashreplace"))
    except Exception:
        # The original exception remains the authoritative failure.  Never
        # overwrite or mask an existing immutable failure receipt.
        return


def _run_shard_core(
    preflight: CrossoverPreflight,
    *,
    event_runner: Callable[..., Any] | None = None,
    executor: Any | None = None,
) -> dict[str, Any]:
    root = Path(preflight.output_root)
    if root.exists() or root.is_symlink():
        raise CrossoverRunnerError("output root appeared after preflight")
    try:
        previous_environment = _execution_environment(preflight)
        try:
            runtime_binding = _derive_pre_gpu_runtime_identity(preflight)
            if event_runner is not None:
                outputs = _call_event_runner(event_runner, preflight.event, preflight)
                technical, cells, live_identity = _normalize_injected_outputs(outputs)
            else:
                if executor is None:
                    module = importlib.import_module("scripts.research.s_natural_boundary_k_n_h_live_executor")
                    executor = module.SNaturalBoundaryKNHLiveExecutor()
                configure = getattr(executor, "configure_pre_gpu_identity", None)
                if not callable(configure):
                    raise CrossoverTechnicalInvalid("live executor lacks configure_pre_gpu_identity transport seam")
                runtime = runtime_binding.get("runtime", {})
                configure(runtime_binding, runtime_versions=runtime if isinstance(runtime, Mapping) else {})
                technical, cells, live_identity = _run_live_event(
                    preflight,
                    executor=executor,
                    runtime_identity=runtime_binding,
                )
        finally:
            _restore_execution_environment(previous_environment)
        technical_raw, normalized = _validate_and_wrap_outputs(preflight, technical, cells)
        result_body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "unit_id": UNIT_ID,
            "shard_id": preflight.shard_id,
            "event_id": preflight.event["event_id"],
            "image_id": preflight.event["image_id"],
            "event_index": preflight.event["event_index"],
            "plan_sha256": preflight.plan_sha256,
            "plan_self_sha256": preflight.plan_self_sha256,
            "pre_gpu_receipt_self_sha256": preflight.pre_gpu_receipt_self_sha256,
            "source_event_sha256": preflight.event["event_sha256"],
            "runtime_identity_sha256": None,
            "technical_control": {"K00": {"transport_arm_id": "K00", "result": technical_raw}},
            "cells": {cell: _cell_document(cell, normalized[cell]) for cell in CELLS},
        }
        combined_identity = dict(runtime_binding)
        if isinstance(live_identity, Mapping):
            combined_identity.update(live_identity)
        runtime = _runtime_identity(preflight, combined_identity)
        runtime_sha = runtime["self_sha256"]
        result_body["runtime_identity_sha256"] = runtime_sha
        result_document = dict(result_body)
        result_document["result_sha256"] = sha256_json(result_body)
        terminal = _terminal_document(preflight, result_document)
        root.mkdir(parents=True, exist_ok=False)
        _write_once_json(root / "result.json", result_document)
        _, runtime_raw_sha = _write_once_json(root / "runtime_identity.json", runtime)
        _, terminal_raw_sha = _write_once_json(root / "terminal_summary.json", terminal)
        result_raw_sha = sha256_bytes(_canonical(result_document) + b"\n")
        aggregate = _aggregate_receipt(
            preflight,
            result_raw_sha256=result_raw_sha,
            runtime_raw_sha256=runtime_raw_sha,
            terminal_raw_sha256=terminal_raw_sha,
            result_sha256=result_document["result_sha256"],
            runtime_identity_sha256=runtime_sha,
            terminal_self_sha256=terminal["self_sha256"],
        )
        _write_once_json(root / "aggregate.receipt.json", aggregate)
        return {
            "result": result_document,
            "runtime_identity": runtime,
            "terminal_summary": terminal,
            "aggregate_receipt": aggregate,
            "output_root": str(root),
        }
    except Exception as exc:
        _persist_failure(root, exc)
        raise


def run_crossover_shard(
    plan: str | Path | Mapping[str, Any],
    pre_gpu_receipt: str | Path | Mapping[str, Any],
    output_root: str | Path,
    *,
    shard_id: str | int,
    manifest: str | Path | Mapping[str, Any] | None = None,
    census: str | Path | None = None,
    config: str | Path | None = None,
    panel: str | Path | None = None,
    cohort: str | Path | None = None,
    h0_root: str | Path | None = None,
    h0_dir: str | Path | None = None,
    cohort_manifest: str | Path | None = None,
    base_model_dir: str | Path | None = None,
    event_runner: Callable[..., Any] | None = None,
    executor: Any | None = None,
) -> dict[str, Any]:
    preflight = validate_preflight(
        plan,
        pre_gpu_receipt,
        output_root,
        shard_id=shard_id,
        manifest=manifest,
        census=census,
        config=config,
        panel=panel,
        cohort=cohort,
        h0_root=h0_root,
        h0_dir=h0_dir,
        cohort_manifest=cohort_manifest,
        base_model_dir=base_model_dir,
    )
    return _run_shard_core(preflight, event_runner=event_runner, executor=executor)


def _run_shard_test_only(
    plan: Mapping[str, Any],
    pre_gpu_receipt: Mapping[str, Any],
    output_root: str | Path,
    *,
    shard_id: str | int,
    manifest: Mapping[str, Any],
    event_runner: Callable[..., Any],
) -> dict[str, Any]:
    """CPU seam used by focused runner tests; no loader/model import."""

    return run_crossover_shard(
        plan,
        pre_gpu_receipt,
        output_root,
        shard_id=shard_id,
        manifest=manifest,
        event_runner=event_runner,
    )


class _ModelFreeFactoryAdapter:
    """Expose sealed architecture metadata to factory code without a model."""

    def __init__(self, source: Any, *, layer_count: int, head_count: int) -> None:
        self._source = source
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(
                    num_hidden_layers=layer_count,
                    num_attention_heads=head_count,
                )
            )
        )
        self.model_device = torch.device("cpu")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


def _model_free_factory_adapter(
    preflight: CrossoverPreflight, processor_adapter: Any
) -> tuple[_ModelFreeFactoryAdapter, dict[str, Any]]:
    base_model_dir = _regular_directory(
        preflight.source_paths["base_model_dir"], "sealed base model directory"
    )
    config_path = _regular_file(
        base_model_dir / "config.json", "sealed base model config"
    )
    config, config_info = _read_json(config_path, "sealed base model config")
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        raise CrossoverTechnicalInvalid(
            "sealed base model config lacks text_config architecture metadata"
        )
    layer_count = text_config.get("num_hidden_layers")
    head_count = text_config.get("num_attention_heads")
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
        or isinstance(head_count, bool)
        or not isinstance(head_count, int)
        or head_count <= 0
    ):
        raise CrossoverTechnicalInvalid(
            "sealed base model config has invalid layer/head counts"
        )
    configured_base = getattr(
        getattr(getattr(processor_adapter, "config", None), "model", None),
        "base_model",
        None,
    )
    if not isinstance(configured_base, str):
        raise CrossoverTechnicalInvalid(
            "processor-only adapter lacks a resolved base-model path"
        )
    observed_base = Path(configured_base).expanduser().resolve(strict=True)
    if observed_base != base_model_dir:
        _identity_mismatch(
            "model-free factory metadata",
            "base_model_dir",
            str(base_model_dir),
            str(observed_base),
        )
    adapter = _ModelFreeFactoryAdapter(
        processor_adapter, layer_count=layer_count, head_count=head_count
    )
    receipt = {
        "status": "passed",
        "source": "sealed_base_model_config_metadata_only",
        "base_model_dir": str(base_model_dir),
        "base_model_inventory_sha256": preflight.source_hashes["base_model_dir"],
        "config_path": str(config_path),
        "config_raw_sha256": config_info["sha256"],
        "layer_count": layer_count,
        "head_count": head_count,
        "device": "cpu",
        "executable_model_present": False,
        "model_loader_called": False,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return adapter, receipt


def _preflight_model_free_production_path(
    preflight: CrossoverPreflight,
    *,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Run every model-free production seam that precedes the first forward."""

    live = importlib.import_module(
        "scripts.research.s_natural_boundary_k_n_h_live_executor"
    )
    gate = importlib.import_module(
        "scripts.research.run_s_primary_natural_boundary_gate"
    )
    legacy = importlib.import_module(
        "scripts.research.run_static_dynamic_owner_interface_experiment"
    )
    output_root = Path(preflight.output_root)
    if output_root.exists() or output_root.is_symlink():
        raise CrossoverTechnicalInvalid(
            "model-free production preflight requires an absent output root"
        )
    orchestrator = legacy.OwnerInterfaceOrchestrator(
        checkpoint=CHECKPOINT,
        stage="all",
        output_dir=output_root,
        fail_collision=False,
        config_path=paths["config"],
        panel_path=paths["panel"],
        cohort_path=paths["cohort"],
        h0_root=paths["h0_root"],
        h0_dir=paths["h0_dir"],
    )
    cpu_identity = orchestrator._load_cpu_contract()
    if output_root.exists() or output_root.is_symlink():
        raise CrossoverTechnicalInvalid(
            "_load_cpu_contract created the reserved output root"
        )
    processor_adapter = orchestrator._load_ineligible_materialization_adapter(
        cpu_identity
    )
    processor_identity = getattr(
        processor_adapter, "processor_only_identity", None
    )
    if (
        not isinstance(processor_identity, Mapping)
        or processor_identity.get("load_model") is not False
        or processor_identity.get("model_present") is not False
        or processor_identity.get("backend_session_opened") is not False
        or getattr(processor_adapter, "model", None) is not None
        or getattr(processor_adapter, "session", None) is not None
    ):
        raise CrossoverTechnicalInvalid(
            "processor-only production preflight loaded a model or backend session"
        )
    full_runtime_cohort = live._preflight_full_runtime_cohort(
        orchestrator, paths, processor_adapter=processor_adapter
    )
    processor_contexts = full_runtime_cohort.get("processor_context_bindings")
    if (
        full_runtime_cohort.get("status") != "passed"
        or full_runtime_cohort.get("event_count") != 11
        or not isinstance(processor_contexts, list)
        or len(processor_contexts) != 11
    ):
        raise CrossoverTechnicalInvalid(
            "model-free full-runtime cohort preflight did not prove all 11 events"
        )

    factory_adapter, factory_metadata = _model_free_factory_adapter(
        preflight, processor_adapter
    )
    census, _census_info = _read_json(paths["census"], "source census")
    rows = census.get("rows")
    if not isinstance(rows, list):
        raise CrossoverTechnicalInvalid("source census lacks rows")
    selected_refs = [
        _require_event_ref(event, label=f"plan.events[{index}]")
        for index, event in enumerate(preflight.plan.get("events", ()))
        if isinstance(event, Mapping)
    ]
    if len(selected_refs) != SHARD_COUNT:
        raise CrossoverTechnicalInvalid(
            "model-free production preflight requires all three selected events"
        )
    selected_contexts: list[dict[str, Any]] = []
    attention_arms = tuple(sorted(gate.ATTENTION_ARMS))

    for ref in selected_refs:
        target = _event_from_manifest(preflight.manifest, ref)
        owner_refs = target["owner_refs"]
        row_matches = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("checkpoint") == CHECKPOINT
            and row.get("gt_owner_id") == owner_refs["gt_owner_id"]
            and row.get("image_id") == target["image_id"]
            and row.get("source_panel_object_index")
            == owner_refs["source_panel_object_index"]
            and row.get("derived_panel_object_index")
            == owner_refs["derived_panel_object_index"]
        ]
        if len(row_matches) != 1:
            raise CrossoverTechnicalInvalid(
                f"selected event {target['event_id']} did not resolve to one census row"
            )
        row = row_matches[0]
        runtime_event = live._runtime_event(target, row)
        legacy_matches = [
            event
            for event in orchestrator.events
            if isinstance(event, Mapping)
            and live._is_exact_legacy_s_event(event, runtime_event)
        ]
        if len(legacy_matches) != 1:
            raise CrossoverTechnicalInvalid(
                f"selected event {target['event_id']} did not resolve to one legacy context"
            )
        seeded = legacy._make_event_context(
            processor_adapter,
            legacy_matches[0],
            history_resolver=live._ledger_exact_history_resolver,
        )
        runtime = getattr(seeded, "runtime", None)
        if runtime is None:
            raise CrossoverTechnicalInvalid(
                f"selected event {target['event_id']} lacks processor runtime"
            )
        context_binding = gate.LiveRuntimeBinding(
            adapter=processor_adapter,
            runtime=runtime,
            event=runtime_event,
            seeded_context=seeded,
        )
        context, boundary, natural_identity = gate.build_natural_event_context(
            context_binding, event_id=str(target["event_id"])
        )
        _require_admitted_history(context.exact_history_token_ids, target)
        admitted_history = tuple(target["natural_boundary"]["history_token_ids"])
        seeded_ids = tuple(
            int(value) for value in seeded.prefix_ids.detach().cpu().reshape(-1).tolist()
        )
        expected_seeded = (
            tuple(context.prompt_token_ids)
            + admitted_history
            + (int(processor_adapter.wrapper_contract.object_ref_start_token_id),)
        )
        if seeded_ids != expected_seeded:
            raise CrossoverTechnicalInvalid(
                f"selected event {target['event_id']} seeded context differs from ledger history"
            )
        if tuple(boundary.natural_prefix_token_ids) != tuple(
            context.prefix_token_ids
        ):
            raise CrossoverTechnicalInvalid(
                f"selected event {target['event_id']} natural boundary differs from context"
            )

        factory_binding = gate.LiveRuntimeBinding(
            adapter=factory_adapter,
            runtime=runtime,
            event=runtime_event,
            seeded_context=seeded,
        )
        factories = gate.build_live_attention_mask_actuators(
            factory_binding, context
        )
        if set(factories) != set(attention_arms):
            raise CrossoverTechnicalInvalid(
                f"selected event {target['event_id']} attention factory set drifted"
            )
        sequence_length = len(context.prefix_token_ids)
        factory_contracts: dict[str, dict[str, Any]] = {}
        for arm in attention_arms:
            factory = factories[arm]
            factory_contracts[arm] = {
                "protocol": getattr(factory, "protocol", None),
                "arm_id": getattr(factory, "arm_id", None),
                "image_key_positions_sha256": sha256_json(
                    list(getattr(factory, "image_key_positions", ()))
                ),
                "b_exclusive_positions_sha256": sha256_json(
                    list(getattr(factory, "b_exclusive_positions", ()))
                ),
                "latest_row_key_positions_sha256": sha256_json(
                    list(getattr(factory, "latest_row_key_positions", ()))
                ),
                "layer_count": getattr(factory, "layer_count", None),
                "head_count": getattr(factory, "head_count", None),
                "device": str(getattr(factory, "device", "")),
            }
        built_factory_receipts: dict[str, dict[str, Any]] = {}
        for arm in ("K01", "K10", "H20"):
            actuator = factories[arm].build(
                sequence_length,
                query_position=sequence_length - 1,
                device="cpu",
            )
            receipt = _detached_json(
                actuator.receipt(),
                label=f"selected event {target['event_id']} factory {arm}",
            )
            for tensor in (actuator.attention_mask, actuator.score_bias):
                if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                    raise CrossoverTechnicalInvalid(
                        f"selected event {target['event_id']} factory {arm} used a GPU"
                    )
            consumption, all_layer_consumption = require_unattested_consumption(
                receipt,
                label=f"selected event {target['event_id']} factory {arm}",
            )
            built_factory_receipts[arm] = {
                "status": receipt.get("status"),
                "mask_sha256": receipt.get("mask_sha256"),
                "selected_positions": list(receipt.get("selected_positions", ())),
                "layer_consumption_attestation": consumption,
                "all_layer_consumption_attestation": all_layer_consumption,
                "no_op_parity": dict(receipt.get("no_op_parity", {})),
                "receipt_sha256": sha256_json(receipt),
            }
        k14_positions = gate._resolve_k14_reference_positions(factory_binding)
        k14_factory_positions = {
            "K14T": list(factories["K14T"].b_exclusive_positions),
            "K14B": list(factories["K14B"]._background_selection or ()),
        }
        for arm, observed_positions in k14_factory_positions.items():
            if observed_positions != list(k14_positions[arm]):
                _identity_mismatch(
                    f"selected event {target['event_id']} K14 geometry",
                    arm,
                    list(k14_positions[arm]),
                    observed_positions,
                )
        c11 = _C11CompositionCallback(factories["K10"], factories["H20"])
        c11_callback = gate._arm_attention_callback(c11, "K10")
        c11_output = c11_callback(
            None,
            input_ids=torch.tensor(
                [list(context.prefix_token_ids)], dtype=torch.long, device="cpu"
            ),
            step=0,
            row_index=0,
        )
        c11_receipt = _detached_json(
            c11_output["receipt"],
            label=f"selected event {target['event_id']} C11 factory",
        )
        require_unattested_consumption(
            c11_receipt,
            label=f"selected event {target['event_id']} C11 factory",
        )
        selected_contexts.append(
            {
                "event_id": target["event_id"],
                "event_index": target["event_index"],
                "image_id": target["image_id"],
                "exact_history_sha256": sha256_json(list(admitted_history)),
                "seeded_prefix_sha256": sha256_json(list(seeded_ids)),
                "natural_context_sha256": sha256_json(context.receipt()),
                "natural_identity_sha256": sha256_json(natural_identity),
                "attention_factory_arms": list(attention_arms),
                "attention_factory_contracts": factory_contracts,
                "built_factory_receipts": built_factory_receipts,
                "k14_reference_positions": {
                    arm: list(values) for arm, values in k14_positions.items()
                },
                "c11_receipt": c11_receipt,
            }
        )

    if (
        getattr(processor_adapter, "model", None) is not None
        or getattr(processor_adapter, "session", None) is not None
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise CrossoverTechnicalInvalid(
            "model-free production preflight crossed the model/output boundary"
        )
    body = {
        "status": "passed",
        "cpu_contract": {
            "status": "passed",
            "event_count": cpu_identity.get("event_count"),
            "identity_sha256": sha256_json(cpu_identity),
        },
        "full_runtime_cohort": dict(full_runtime_cohort),
        "processor_only": {
            "status": "passed",
            "load_model": False,
            "model_present": False,
            "backend_session_opened": False,
            "identity_sha256": sha256_json(processor_identity),
        },
        "factory_model_metadata": factory_metadata,
        "selected_factory_contexts": selected_contexts,
        "selected_event_count": len(selected_contexts),
        "gpu_used": False,
        "model_loaded": False,
        "model_loader_called": False,
        "output_root_created": False,
    }
    result = dict(body)
    result["receipt_sha256"] = sha256_json(body)
    return result


def preflight_crossover_sources(
    plan: str | Path,
    *,
    manifest: str | Path,
    census: str | Path,
    config: str | Path,
    panel: str | Path,
    cohort: str | Path,
    cohort_manifest: str | Path,
    h0_root: str | Path,
    h0_dir: str | Path,
    base_model_dir: str | Path,
    execution_root: str | Path,
    final_root: str | Path,
) -> dict[str, Any]:
    """Produce receipt-independent model-free evidence for the v4 sealer."""

    from scripts.research import materialize_s_k10_h20_crossover_plan as planner
    from scripts.research import seal_s_k10_h20_crossover_pre_gpu_receipt as sealer

    plan_path = _regular_file(plan, "crossover plan")
    planner_info = planner.validate_plan(plan_path)
    plan_doc, plan_info = _read_json(plan_path, "crossover plan")
    if planner_info.get("plan") != plan_doc:
        raise CrossoverRunnerError(
            "materializer returned a plan different from the canonical plan path"
        )
    _require_exact_operator_contract(plan_doc)
    execution_path = _absolute_absent(execution_root, "reserved execution root")
    final_path = _absolute_absent(final_root, "reserved final root")
    if execution_path == final_path:
        raise CrossoverRunnerError("reserved execution/final roots must be distinct")

    raw_sources: dict[str, str | Path] = {
        "manifest": manifest,
        "census": census,
        "execution_plan": plan_path,
        "config": config,
        "panel": panel,
        "cohort": cohort,
        "cohort_manifest": cohort_manifest,
        "h0_root": h0_root,
        "h0_dir": h0_dir,
        "base_model_dir": base_model_dir,
    }
    directory_labels = {"h0_root", "h0_dir", "base_model_dir"}
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    source_bindings: dict[str, dict[str, Any]] = {}
    for label, raw_source in raw_sources.items():
        path = (
            _regular_directory(raw_source, label)
            if label in directory_labels
            else _regular_file(raw_source, label)
        )
        digest = (
            _sha256_directory(path)
            if label in directory_labels
            else _sha256_file(path)
        )
        source_paths[label] = str(path)
        source_hashes[label] = digest
        source_bindings[label] = {
            "path": str(path),
            "sha256": digest,
            "kind": "directory" if label in directory_labels else "file",
        }
    manifest_source = plan_doc.get("source_bindings", {}).get("manifest")
    census_source = plan_doc.get("source_bindings", {}).get("census")
    if (
        not isinstance(manifest_source, Mapping)
        or manifest_source.get("path") != source_paths["manifest"]
        or manifest_source.get("raw_sha256") != source_hashes["manifest"]
    ):
        raise CrossoverRunnerError(
            "pre-seal manifest differs from deterministic plan binding"
        )
    if (
        not isinstance(census_source, Mapping)
        or census_source.get("path") != source_paths["census"]
        or census_source.get("raw_sha256") != source_hashes["census"]
    ):
        raise CrossoverRunnerError(
            "pre-seal census differs from deterministic plan binding"
        )
    manifest_doc, manifest_info = _read_json(
        source_paths["manifest"], "source manifest"
    )
    first_event = _event_from_manifest(manifest_doc, plan_doc["events"][0])
    preflight = CrossoverPreflight(
        plan=dict(plan_doc),
        plan_sha256=str(planner_info.get("plan_sha256") or plan_doc["self_sha256"]),
        plan_self_sha256=str(
            planner_info.get("self_sha256") or plan_doc["self_sha256"]
        ),
        plan_source_sha256=plan_info["sha256"],
        manifest=dict(manifest_doc),
        manifest_sha256=manifest_info["sha256"],
        manifest_path=source_paths["manifest"],
        event=first_event,
        shard_id="preseal-all-shards",
        shard_index=-1,
        output_root=str(execution_path),
        pre_gpu_receipt={},
        pre_gpu_receipt_path="",
        pre_gpu_receipt_raw_sha256="0" * 64,
        pre_gpu_receipt_self_sha256="0" * 64,
        source_paths=source_paths,
        source_hashes=source_hashes,
    )
    live_paths = {
        label: Path(source_paths[label])
        for label in (
            "manifest",
            "census",
            "config",
            "panel",
            "cohort",
            "cohort_manifest",
            "h0_root",
            "h0_dir",
        )
    }
    production_path = _preflight_model_free_production_path(
        preflight, paths=live_paths
    )
    code_bindings: dict[str, dict[str, str]] = {}
    for role, raw_path in sealer.CODE_ROLE_PATHS.items():
        path = _regular_file(raw_path, f"source file {role}")
        code_bindings[role] = {"path": str(path), "sha256": _sha256_file(path)}
    test_bindings: dict[str, dict[str, str]] = {}
    for role, raw_path in sealer.TEST_PATHS.items():
        path = _regular_file(raw_path, f"test file {role}")
        test_bindings[role] = {"path": str(path), "sha256": _sha256_file(path)}
    body = {
        "schema_version": SOURCE_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "phase": "preseal_model_free_production",
        "unit_id": UNIT_ID,
        "plan_binding": {
            "path": str(plan_path),
            "raw_sha256": plan_info["sha256"],
            "self_sha256": plan_doc["self_sha256"],
        },
        "source_bindings": source_bindings,
        "code_bindings": code_bindings,
        "test_bindings": test_bindings,
        "reserved_roots": {
            "execution_root": {
                "path": str(execution_path),
                "status": "reserved_absent_pre_gpu",
            },
            "final_root": {
                "path": str(final_path),
                "status": "reserved_absent_pre_gpu",
            },
        },
        "model_free_production_path": production_path,
        "gpu_used": False,
        "model_loaded": False,
        "backend_session_opened": False,
        "output_root_created": False,
        "receipt_independent": True,
    }
    if execution_path.exists() or final_path.exists():
        raise CrossoverTechnicalInvalid(
            "pre-seal source preflight created a reserved root"
        )
    result = dict(body)
    result["self_sha256"] = sha256_json(body)
    return result


def preflight_crossover(
    plan: str | Path | Mapping[str, Any],
    pre_gpu_receipt: str | Path | Mapping[str, Any],
    output_root: str | Path,
    *,
    shard_id: str | int,
    manifest: str | Path | Mapping[str, Any] | None = None,
    census: str | Path | None = None,
    config: str | Path | None = None,
    panel: str | Path | None = None,
    cohort: str | Path | None = None,
    h0_root: str | Path | None = None,
    h0_dir: str | Path | None = None,
    cohort_manifest: str | Path | None = None,
    base_model_dir: str | Path | None = None,
    executor: Any | None = None,
) -> dict[str, Any]:
    """Exercise the exact launch source seam without loading a model or writing output."""

    preflight = validate_preflight(
        plan,
        pre_gpu_receipt,
        output_root,
        shard_id=shard_id,
        manifest=manifest,
        census=census,
        config=config,
        panel=panel,
        cohort=cohort,
        h0_root=h0_root,
        h0_dir=h0_dir,
        cohort_manifest=cohort_manifest,
        base_model_dir=base_model_dir,
    )
    live = importlib.import_module(
        "scripts.research.s_natural_boundary_k_n_h_live_executor"
    )
    cohort_module = importlib.import_module(
        "scripts.research.run_s_natural_boundary_k_n_h_cohort"
    )
    if executor is None:
        executor = live.SNaturalBoundaryKNHLiveExecutor()
    if any(
        getattr(executor, attribute, None) is not None
        for attribute in ("_adapter", "_orchestrator")
    ):
        raise CrossoverTechnicalInvalid(
            "source preflight requires a fresh executor with no loaded model"
        )
    configure = getattr(executor, "configure_pre_gpu_identity", None)
    validate_inputs = getattr(executor, "_validate_inputs", None)
    if not callable(configure) or not callable(validate_inputs):
        raise CrossoverTechnicalInvalid(
            "live executor lacks CPU source-preflight configuration/validation seams"
        )

    previous_environment = _execution_environment(preflight)
    try:
        runtime_identity = _derive_pre_gpu_runtime_identity(preflight)
        runtime_versions = runtime_identity.get("runtime")
        if not isinstance(runtime_versions, Mapping):
            raise CrossoverTechnicalInvalid(
                "pre-GPU runtime identity lacks runtime versions"
            )
        configure(runtime_identity, runtime_versions=runtime_versions)
        target, row, paths = validate_inputs(preflight.event, cohort_module.ARM_ORDER)
        production_path = _preflight_model_free_production_path(
            preflight, paths=paths
        )
    finally:
        _restore_execution_environment(previous_environment)

    if not isinstance(target, Mapping) or dict(target) != preflight.event:
        raise CrossoverTechnicalInvalid(
            "source preflight did not retain the exact admitted full manifest event"
        )
    if not isinstance(row, Mapping):
        raise CrossoverTechnicalInvalid(
            "source preflight did not resolve an exact census row"
        )
    if not isinstance(paths, Mapping):
        raise CrossoverTechnicalInvalid(
            "source preflight did not resolve exact input paths"
        )
    if any(
        getattr(executor, attribute, None) is not None
        for attribute in ("_adapter", "_orchestrator")
    ):
        raise CrossoverTechnicalInvalid("source preflight loaded model state")

    live_path_labels = {
        "manifest",
        "census",
        "config",
        "panel",
        "cohort",
        "cohort_manifest",
        "h0_root",
        "h0_dir",
        "pre_gpu_receipt",
    }
    if set(paths) != live_path_labels:
        raise CrossoverTechnicalInvalid(
            "source preflight live path labels differ: "
            f"expected={sorted(live_path_labels)!r}, observed={sorted(paths)!r}"
        )
    for label, raw_path in paths.items():
        path = Path(raw_path).expanduser().resolve(strict=True)
        sealed_path = preflight.source_paths[label]
        if sealed_path != str(path):
            _identity_mismatch(
                "source preflight paths", str(label), sealed_path, str(path)
            )

    resolved_paths: dict[str, dict[str, str]] = {}
    for label, sealed_path in sorted(preflight.source_paths.items()):
        path = Path(sealed_path).expanduser().resolve(strict=True)
        path_sha256 = preflight.source_hashes.get(label)
        if path_sha256 is None:
            raise CrossoverTechnicalInvalid(
                f"source preflight path {label!r} lacks a sealed hash"
            )
        resolved_paths[str(label)] = {
            "path": str(path),
            "sha256": _require_sha256(
                path_sha256, f"source preflight paths.{label}.sha256"
            ),
        }

    owner_refs = target.get("owner_refs")
    if not isinstance(owner_refs, Mapping):  # pragma: no cover - projection rejects first
        raise CrossoverTechnicalInvalid("source preflight target owner refs are missing")
    resolved_identity = _manifest_event_projection(
        target, label=f"source preflight event {target.get('event_id')}"
    )
    resolved_identity["source_qualification"] = dict(SOURCE_QUALIFICATION)
    resolved_identity.update(
        {
            "checkpoint": target.get("checkpoint"),
            "step": target.get("step"),
            "substrate": target.get("substrate"),
        }
    )
    resolved_census_row = {
        "checkpoint": row.get("checkpoint"),
        "gt_owner_id": row.get("gt_owner_id"),
        "image_id": row.get("image_id"),
        "source_panel_object_index": row.get("source_panel_object_index"),
        "derived_panel_object_index": row.get("derived_panel_object_index"),
        "row_sha256": sha256_json(row),
    }
    body = {
        "schema_version": SOURCE_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "unit_id": UNIT_ID,
        "shard_id": preflight.shard_id,
        "plan_sha256": preflight.plan_sha256,
        "plan_self_sha256": preflight.plan_self_sha256,
        "pre_gpu_receipt_self_sha256": preflight.pre_gpu_receipt_self_sha256,
        "runtime_identity_sha256": sha256_json(runtime_identity),
        "resolved_identity": resolved_identity,
        "resolved_census_row": resolved_census_row,
        "resolved_paths": resolved_paths,
        "model_free_production_path": production_path,
        "gpu_used": False,
        "model_loaded": False,
        "output_root_created": False,
    }
    if Path(preflight.output_root).exists() or Path(preflight.output_root).is_symlink():
        raise CrossoverTechnicalInvalid("source preflight created the reserved output root")
    result = dict(body)
    result["self_sha256"] = sha256_json(body)
    return result


def run_cpu_preflight_probe(
    *, source_preflight: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Emit canonical composition and installed-Qwen transport evidence."""

    try:
        k10 = attention.build_k10(
            sequence_length=8,
            image_key_positions=(1, 2, 3, 4),
            b_exclusive_positions=(3, 4),
            query_position=7,
        )
        h20 = attention.build_h20(
            sequence_length=8,
            query_positions=(7,),
            latest_row_key_positions=(5, 6),
        )
        composed = attention.compose_k10_h20(k10, h20)
        composition = {
            "status": "passed",
            "arm_id": composed.arm_id,
            "cell_id": composed.receipt().get("cell_id"),
            "mask_sha256": composed.receipt().get("mask_sha256"),
            "logical_and_passed": composed.receipt().get("logical_and_passed"),
            "component_changed_cell_union": composed.receipt().get("component_changed_cell_union"),
        }
    except Exception as exc:
        composition = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
    try:
        qwen = attention.run_installed_qwen_cpu_probe()
    except Exception as exc:  # pragma: no cover - environment-specific
        qwen = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
    if source_preflight is None:
        source_document: dict[str, Any] = {"status": "not_exercised"}
    elif source_preflight.get("status") == "passed":
        source_document = _detached_json(
            source_preflight, label="source_preflight"
        )
    else:
        raise CrossoverRunnerError("source preflight evidence is not passing")
    body = {
        "schema_version": PRE_GPU_PROBE_SCHEMA_VERSION,
        "status": "passed"
        if composition.get("status") == "passed"
        and qwen.get("status") == "passed"
        and source_document.get("status") in {"passed", "not_exercised"}
        else "blocked",
        "unit_id": UNIT_ID,
        "composition": composition,
        "installed_qwen_consumption": _detached_json(qwen, label="installed_qwen_consumption"),
        "source_preflight": source_document,
        "gpu_used": False,
        "model_loaded": False,
        "no_gpu_launch": True,
    }
    result = dict(body)
    result["self_sha256"] = sha256_json(body)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--pre-gpu-receipt", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--census", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--cohort-manifest", type=Path)
    parser.add_argument("--h0-root", type=Path)
    parser.add_argument("--h0-dir", type=Path)
    parser.add_argument("--base-model-dir", type=Path)
    parser.add_argument("--shard-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execution-root", type=Path)
    parser.add_argument("--final-root", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--cpu-preflight", action="store_true")
    modes.add_argument("--preseal-source-preflight", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preseal_source_preflight:
        source_arguments = {
            "plan": args.plan,
            "manifest": args.manifest,
            "census": args.census,
            "config": args.config,
            "panel": args.panel,
            "cohort": args.cohort,
            "cohort_manifest": args.cohort_manifest,
            "h0_root": args.h0_root,
            "h0_dir": args.h0_dir,
            "base_model_dir": args.base_model_dir,
            "execution_root": args.execution_root,
            "final_root": args.final_root,
        }
        missing = [
            f"--{name.replace('_', '-')}"
            for name, value in source_arguments.items()
            if value is None
        ]
        if missing:
            blocked = {
                "schema_version": SOURCE_PREFLIGHT_SCHEMA_VERSION,
                "status": "blocked",
                "phase": "preseal_model_free_production",
                "unit_id": UNIT_ID,
                "error": f"missing required arguments: {missing}",
                "receipt_independent": True,
                "gpu_used": False,
                "model_loaded": False,
                "backend_session_opened": False,
                "output_root_created": False,
            }
            blocked["self_sha256"] = sha256_json(blocked)
            print(_canonical(blocked).decode("utf-8"), file=sys.stderr)
            return 2
        try:
            plan_path = source_arguments.pop("plan")
            document = preflight_crossover_sources(
                plan_path,  # type: ignore[arg-type]
                **source_arguments,  # type: ignore[arg-type]
            )
            print(_canonical(document).decode("utf-8"))
            return 0
        except Exception as exc:
            blocked = {
                "schema_version": SOURCE_PREFLIGHT_SCHEMA_VERSION,
                "status": "blocked",
                "phase": "preseal_model_free_production",
                "unit_id": UNIT_ID,
                "error": f"{type(exc).__name__}: {exc}",
                "receipt_independent": True,
                "gpu_used": False,
                "model_loaded": False,
                "backend_session_opened": False,
                "output_root_created": False,
            }
            blocked["self_sha256"] = sha256_json(blocked)
            print(_canonical(blocked).decode("utf-8"), file=sys.stderr)
            return 2
    if args.cpu_preflight:
        source_arguments = {
            "--plan": args.plan,
            "--pre-gpu-receipt": args.pre_gpu_receipt,
            "--manifest": args.manifest,
            "--census": args.census,
            "--config": args.config,
            "--panel": args.panel,
            "--cohort": args.cohort,
            "--cohort-manifest": args.cohort_manifest,
            "--h0-root": args.h0_root,
            "--h0-dir": args.h0_dir,
            "--base-model-dir": args.base_model_dir,
            "--shard-id": args.shard_id,
            "--output-root": args.output_root,
        }
        required_source_arguments = (
            "--plan",
            "--pre-gpu-receipt",
            "--manifest",
            "--shard-id",
            "--output-root",
        )
        provided = [name for name, value in source_arguments.items() if value is not None]
        if not provided:
            print(_canonical(run_cpu_preflight_probe()).decode("utf-8"))
            return 0
        missing = [
            name
            for name in required_source_arguments
            if source_arguments[name] is None
        ]
        if missing:
            blocked = {
                "schema_version": PRE_GPU_PROBE_SCHEMA_VERSION,
                "status": "blocked",
                "unit_id": UNIT_ID,
                "source_preflight": {
                    "status": "blocked",
                    "error": f"partial source preflight arguments; missing required arguments: {missing}",
                },
                "gpu_used": False,
                "model_loaded": False,
                "no_gpu_launch": True,
            }
            blocked["self_sha256"] = sha256_json(blocked)
            print(_canonical(blocked).decode("utf-8"), file=sys.stderr)
            return 2
        try:
            shard_value: str | int
            try:
                shard_value = int(args.shard_id)
            except (TypeError, ValueError):
                shard_value = str(args.shard_id)
            source_document = preflight_crossover(
                args.plan,
                args.pre_gpu_receipt,
                args.output_root,
                shard_id=shard_value,
                manifest=args.manifest,
                census=args.census,
                config=args.config,
                panel=args.panel,
                cohort=args.cohort,
                cohort_manifest=args.cohort_manifest,
                h0_root=args.h0_root,
                h0_dir=args.h0_dir,
                base_model_dir=args.base_model_dir,
            )
            print(
                _canonical(
                    run_cpu_preflight_probe(source_preflight=source_document)
                ).decode("utf-8")
            )
            return 0
        except Exception as exc:
            blocked = {
                "schema_version": PRE_GPU_PROBE_SCHEMA_VERSION,
                "status": "blocked",
                "unit_id": UNIT_ID,
                "source_preflight": {
                    "status": "blocked",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "gpu_used": False,
                "model_loaded": False,
                "no_gpu_launch": True,
            }
            blocked["self_sha256"] = sha256_json(blocked)
            print(_canonical(blocked).decode("utf-8"), file=sys.stderr)
            return 2
    required = {
        "--plan": args.plan,
        "--pre-gpu-receipt": args.pre_gpu_receipt,
        "--manifest": args.manifest,
        "--shard-id": args.shard_id,
        "--output-root": args.output_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        print(_canonical({"status": "blocked", "error": f"missing required arguments: {missing}"}).decode("utf-8"), file=sys.stderr)
        return 2
    try:
        shard_value: str | int
        try:
            shard_value = int(args.shard_id)
        except (TypeError, ValueError):
            shard_value = str(args.shard_id)
        result = run_crossover_shard(
            args.plan,
            args.pre_gpu_receipt,
            args.output_root,
            shard_id=shard_value,
            manifest=args.manifest,
            census=args.census,
            config=args.config,
            panel=args.panel,
            cohort=args.cohort,
            cohort_manifest=args.cohort_manifest,
            h0_root=args.h0_root,
            h0_dir=args.h0_dir,
            base_model_dir=args.base_model_dir,
        )
        print(_canonical({"status": "completed", "result_sha256": result["result"]["result_sha256"]}).decode("utf-8"))
        return 0
    except Exception as exc:
        print(_canonical({"status": "blocked", "error": str(exc)}).decode("utf-8"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

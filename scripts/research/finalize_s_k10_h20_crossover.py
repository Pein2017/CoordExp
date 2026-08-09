#!/usr/bin/env python3
"""Finalize the fixed S/K10 x H20 crossover on CPU.

The runner writes one event per shard.  This module is deliberately the only
consumer that turns those event roots into endpoint evidence.  It validates
the immutable plan, pre-GPU receipt, runtime identities, and every raw natural
trajectory before deriving any endpoint.  Runner summaries are treated as
receipts only; endpoint vectors are reconstructed from rows and their
owner-bookkeeping records.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


UNIT_ID = "2026-08-07-s-k10-h20-natural-crossover"
PLAN_SCHEMA_VERSION = "s_k10_h20_crossover_plan.v1"
PRE_GPU_SCHEMA_VERSION = "s_k10_h20_crossover_pre_gpu_receipt.v1"
EVENT_SCHEMA_VERSION = "s_k10_h20_crossover_event.v1"
RUNTIME_SCHEMA_VERSION = "s_k10_h20_crossover_runtime_identity.v1"
TERMINAL_SCHEMA_VERSION = "s_k10_h20_crossover_terminal_summary.v1"
SHARD_RECEIPT_SCHEMA_VERSION = "s_k10_h20_crossover_shard_receipt.v1"
EVIDENCE_SCHEMA_VERSION = "s_k10_h20_crossover_evidence.v1"
RECEIPT_SCHEMA_VERSION = "s_k10_h20_crossover_evidence.receipt.v1"

PRIMARY = {"checkpoint": "S", "step": 2444, "substrate": "four-coordinate geo_sorted_xy"}
EVENT_IDS = ("gt:2299:29", "gt:13348:14", "gt:16228:15")
# ``shard_index`` is the local 0/1/2 execution ordinal.  These are the
# source-manifest event ordinals and must never be replaced by the local
# ordinal when binding a result back to its plan event.
EVENT_INDICES = (2, 5, 8)
CELL_ORDER = ("C00", "C10", "C01", "C11")
TECHNICAL_CELL = "K00"
ALL_CELLS = (*CELL_ORDER, TECHNICAL_CELL)
TRANSPORT_ARMS = {"C00": "K01", "C10": "K10", "C01": "H20", "C11": "K10"}
COMPONENT_ARMS = {"C11": "C11"}
METRICS = (
    "complete_rows",
    "strict_count",
    "target_release",
    "unmatched",
    "duplicates",
    "ambiguous",
    "malformed",
    "invalid",
    "STOP",
    "row_admission",
)
SCIENTIFIC_TERMINALS = frozenset(
    {"closure", "native_stop", "invalid", "over_continuation", "max_budget"}
)
REF_KINDS = frozenset({"file", "directory"})
CONSUMPTION_SCHEMA_VERSION = "natural_boundary_attention_actuators.v1.layer_consumption.v1"
PLACEHOLDER_BASE_KEYS = frozenset(
    {"schema_version", "required", "status", "exact_same_tensor_all_layers_required"}
)
CONTRASTS = {
    "C10-C00_static": ("C10", "C00", "static"),
    "C01-C00_history": ("C01", "C00", "history"),
    "C11-C01_static_given_H": ("C11", "C01", "static_given_H"),
    "C11-C10_history_given_K": ("C11", "C10", "history_given_K"),
}
SHA256_RE = set("0123456789abcdef")
DEVICE_PLAN = {"shard-000": "0", "shard-001": "1", "shard-002": "7"}

_PER_SHARD_RUNTIME_FIELDS = frozenset(
    {
        "shard_id",
        "event_id",
        "event_index",
        "image_id",
        "device",
        "device_assignment",
        "event_binding",
        "self_sha256",
    }
)
_RUNTIME_CODE_ALIASES = {
    "runner": "crossover_runner",
    "attention_actuators": "attention_actuator",
    "finalizer": "crossover_finalizer",
    "materializer": "plan_materializer",
    "sealer": "pre_gpu_sealer",
    "live_executor": "base_live_executor",
}
_RUNTIME_INPUT_EXTRAS = frozenset(
    {
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
        "pre_gpu_receipt",
        "pre_gpu_receipt_self_sha256",
        "plan_file_sha256",
        "plan_sha256",
        "plan_self_sha256",
    }
)
# Only a validated finalization successor may admit this runner-local key, and
# only together with the exact parent pins below.  It is deliberately not part
# of the generic allow-list above.
_SUCCESSOR_RUNTIME_INPUT_EXTRAS = frozenset({"pre_gpu_receipt_sha256"})
FINALIZATION_RECEIPT_SCHEMA_VERSION = "s_k10_h20_crossover_finalization_receipt.v1"
FINALIZATION_RECEIPT_STATUS = "sealed_post_execution"
AUTHORIZED_FINALIZATION_SLOTS = ("source_files.crossover_finalizer", "test_files.crossover_finalizer_test")
FINALIZATION_TOOL_ROLES = ("finalizer", "finalizer_test", "successor_sealer", "successor_sealer_test")
FINALIZATION_SHARD_ARTIFACTS = ("result.json", "runtime_identity.json", "terminal_summary.json", "aggregate.receipt.json")


class EvidenceContractError(ValueError):
    """Raised when a source cannot support a mechanically valid endpoint."""


FinalizerError = EvidenceContractError


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceContractError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def document_self_sha256(document: Mapping[str, Any], field: str = "self_sha256") -> str:
    body = dict(document)
    body.pop(field, None)
    return sha256_json(body)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in SHA256_RE for ch in value):
        raise EvidenceContractError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceContractError(f"{label} must be a non-empty string")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise EvidenceContractError(f"{label} must be a JSON boolean")
    return value


def _int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceContractError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise EvidenceContractError(f"{label} must be finite numeric")
    return float(value)


def _absolute(value: str | Path, label: str) -> Path:
    target = Path(value).expanduser()
    if not target.is_absolute():
        raise EvidenceContractError(f"{label} must be an absolute path")
    cursor = target
    while True:
        if cursor.is_symlink():
            raise EvidenceContractError(f"{label} must not traverse a symlink: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    try:
        resolved = target.resolve(strict=False)
    except OSError as exc:
        raise EvidenceContractError(f"cannot resolve {label}: {target}") from exc
    if resolved != target:
        raise EvidenceContractError(f"{label} resolves through a symlink: {target}")
    return resolved


def _regular_file(value: str | Path, label: str) -> Path:
    target = _absolute(value, label)
    if target.is_symlink() or not target.is_file():
        raise EvidenceContractError(f"{label} must be an existing regular non-symlink file: {target}")
    return target


def _regular_dir(value: str | Path, label: str) -> Path:
    target = _absolute(value, label)
    if target.is_symlink() or not target.is_dir():
        raise EvidenceContractError(f"{label} must be an existing regular non-symlink directory: {target}")
    return target


def _stream_sha256(target: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceContractError(f"cannot read {label}: {target}") from exc
    return digest.hexdigest()


def sha256_file(value: str | Path) -> str:
    target = _regular_file(value, "hash source")
    return _stream_sha256(target, "hash source")


def _directory_inventory(root: Path, label: str) -> tuple[str, int]:
    """Return the sealer's deterministic directory identity and total size.

    This must stay byte-identical to ``_directory_inventory`` in the pre-GPU
    sealer and the runner: walk without following symlinks, reject symlink
    directories and non-regular files, record ``relative_path``/``sha256``/
    ``size_bytes`` sorted by ``relative_path``, reject an empty inventory, and
    hash the entry list as canonical JSON.
    """

    entries: list[dict[str, Any]] = []
    try:
        for current_raw, dirs, files in os.walk(root, followlinks=False):
            current = Path(current_raw)
            dirs.sort()
            files.sort()
            for name in dirs:
                if (current / name).is_symlink():
                    raise EvidenceContractError(f"{label} contains a symlink directory: {current / name}")
            for name in files:
                path = current / name
                if path.is_symlink() or not path.is_file():
                    raise EvidenceContractError(f"{label} contains a non-regular file: {path}")
                entries.append(
                    {
                        "relative_path": path.relative_to(root).as_posix(),
                        "sha256": _stream_sha256(path, label),
                        "size_bytes": path.stat().st_size,
                    }
                )
    except OSError as exc:
        raise EvidenceContractError(f"cannot inventory {label}: {root}") from exc
    entries.sort(key=lambda item: item["relative_path"])
    if not entries:
        raise EvidenceContractError(f"{label} inventory is empty")
    return sha256_json(entries), sum(item["size_bytes"] for item in entries)


def _read_json(value: str | Path | Mapping[str, Any], label: str, *, canonical: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(value, Mapping):
        document = dict(value)
        canonical_json_bytes(document)
        return document, {"path": None, "raw_sha256": sha256_json(document), "size_bytes": len(canonical_json_bytes(document))}
    target = _regular_file(value, label)
    try:
        raw = target.read_bytes()
        parsed = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceContractError(f"{label} is not readable JSON: {target}") from exc
    if not isinstance(parsed, Mapping):
        raise EvidenceContractError(f"{label} must contain a JSON object")
    document = dict(parsed)
    canonical_json_bytes(document)
    if canonical and raw != canonical_json_bytes(document) + b"\n":
        raise EvidenceContractError(f"{label} must be canonical JSON with one trailing newline")
    return document, {"path": str(target), "raw_sha256": sha256_bytes(raw), "size_bytes": len(raw)}


def _ref(value: Any, label: str, *, require_path: bool = True, expected_sha256: str | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceContractError(f"{label} must be a path reference")
    raw_path = value.get("path") or value.get("file") or value.get("root")
    if raw_path is None:
        if require_path:
            raise EvidenceContractError(f"{label}.path is missing")
        return dict(value)
    # The sealer declares ``kind`` on every binding it writes.  A binding that
    # predates that field is a file binding, and a declared kind is never
    # inferred away: a file kind pointing at a directory (or the reverse) is a
    # drifted binding, not a shape to guess at.
    kind = value.get("kind", "file")
    if kind not in REF_KINDS:
        raise EvidenceContractError(f"{label}.kind must be one of {sorted(REF_KINDS)}")
    if kind == "directory":
        target = _regular_dir(raw_path, label)
        digest, size = _directory_inventory(target, label)
    else:
        target = _regular_file(raw_path, label)
        digest = sha256_file(target)
        size = target.stat().st_size
    if expected_sha256 is not None:
        # An authorized finalization successor replaces exactly this binding's
        # parent-sealed hash; the parent document itself is never rewritten.
        if digest != _sha(expected_sha256, f"{label}.successor_sha256"):
            raise EvidenceContractError(f"{label} live file differs from the successor-authorized hash")
    else:
        supplied = value.get("sha256", value.get("raw_sha256"))
        if supplied is not None and _sha(supplied, f"{label}.sha256") != digest:
            raise EvidenceContractError(f"{label} raw SHA-256 drifted")
        supplied_size = value.get("size_bytes")
        if supplied_size is not None and _int(supplied_size, f"{label}.size_bytes") != size:
            raise EvidenceContractError(f"{label} size_bytes drifted")
    result = dict(value)
    result["path"] = str(target)
    if "sha256" in result:
        result["sha256"] = digest
    if "raw_sha256" in result:
        result["raw_sha256"] = digest
    result.setdefault("sha256", digest)
    result.setdefault("size_bytes", size)
    return result


def _self_check(document: Mapping[str, Any], label: str, *, field: str = "self_sha256") -> None:
    if field not in document:
        raise EvidenceContractError(f"{label}.{field} is missing")
    declared = _sha(document[field], f"{label}.{field}")
    if declared != document_self_sha256(document, field):
        raise EvidenceContractError(f"{label}.{field} mismatch")


def _hash_ref(value: Any, label: str, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Validate a nested path/hash source binding without trusting its labels."""

    if not isinstance(value, Mapping):
        raise EvidenceContractError(f"{label} must be an object")
    result = dict(value)
    if "path" in result:
        checked = _ref(result, label, expected_sha256=expected_sha256)
        result.update(checked)
    for key in ("self_sha256", "semantic_self_sha256", "raw_sha256", "sha256"):
        if key in result:
            _sha(result[key], f"{label}.{key}")
    return result


def _validate_plan(document: Mapping[str, Any], info: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != PLAN_SCHEMA_VERSION or document.get("unit_id") != UNIT_ID or document.get("status") != "planned":
        raise EvidenceContractError("plan schema/status/unit identity drifted")
    if document.get("primary") != PRIMARY:
        raise EvidenceContractError("plan primary S step-2444 identity drifted")
    _self_check(document, "plan")
    if document.get("event_count") != 3 or document.get("image_count") != 3:
        raise EvidenceContractError("plan must bind three events and three images")
    events = document.get("events")
    if not isinstance(events, list) or len(events) != 3:
        raise EvidenceContractError("plan events must contain exactly three entries")
    if tuple(item.get("event_id") for item in events if isinstance(item, Mapping)) != EVENT_IDS:
        raise EvidenceContractError("plan event order differs from frozen selected order")
    images: set[int] = set()
    event_refs: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise EvidenceContractError(f"plan event {index} is malformed")
        if event.get("event_index") != EVENT_INDICES[index] or event.get("event_id") != EVENT_IDS[index]:
            raise EvidenceContractError(f"plan event {index} order/identity drifted")
        image_id = _int(event.get("image_id"), f"plan event {index}.image_id")
        images.add(image_id)
        # The selected plan carries the source manifest's event hash.  It is
        # intentionally not a hash of this smaller plan event projection.
        _sha(event.get("event_sha256"), f"plan event {index}.event_sha256")
        event_refs.append(dict(event))
    if len(images) != 3:
        raise EvidenceContractError("plan events must have three distinct image IDs")
    device_plan = document.get("device_plan")
    if device_plan != DEVICE_PLAN:
        raise EvidenceContractError("plan device_plan must bind shard-000/001/002 to physical 0/1/7")
    if document.get("cell_order") != list(CELL_ORDER) or document.get("technical_control") != "C00":
        raise EvidenceContractError("plan cell order/technical control drifted")
    cells = document.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != set(CELL_ORDER):
        raise EvidenceContractError("plan cell contracts are incomplete")
    expected_plan_cells = {
        "C00": {"source_arm": "K01"},
        "C10": {"source_arm": "K10"},
        "C01": {"source_arm": "H20"},
        "C11": {"source_arms": ["K10", "H20"]},
    }
    for cell, expected in expected_plan_cells.items():
        value = cells[cell]
        if not isinstance(value, Mapping):
            raise EvidenceContractError(f"plan cell {cell} contract is malformed")
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise EvidenceContractError(f"plan cell {cell}.{key} drifted")
    shards = document.get("shards")
    if not isinstance(shards, list) or len(shards) != 3:
        raise EvidenceContractError("plan must contain exactly three shards")
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping) or shard.get("shard_index") != index or shard.get("shard_id") != f"shard-{index:03d}":
            raise EvidenceContractError(f"plan shard {index} identity drifted")
        if shard.get("physical_device") != DEVICE_PLAN[f"shard-{index:03d}"] or shard.get("logical_device") != "cuda:0":
            raise EvidenceContractError(f"plan shard {index} device assignment drifted")
        shard_events = shard.get("events")
        if not isinstance(shard_events, list) or len(shard_events) != 1 or shard_events[0] != event_refs[index]:
            raise EvidenceContractError(f"plan shard {index} event binding drifted")
    for key in ("no_event_reorder", "no_reselection", "no_sweep", "no_a3", "no_p4", "no_training"):
        if key in document and document[key] is not True:
            raise EvidenceContractError(f"plan safety flag {key} is false")
    if "use_cache" in document and document["use_cache"] is not False:
        raise EvidenceContractError("plan use_cache boundary drifted")
    operator = document.get("operator_contract")
    if operator is not None:
        if not isinstance(operator, Mapping) or operator.get("admission_mode") != "pre_opener_natural" or operator.get("opener_injected") is not False or operator.get("use_cache") is not False:
            raise EvidenceContractError("plan operator admission/cache contract drifted")
    sources = document.get("source_bindings")
    if not isinstance(sources, Mapping) or not sources:
        raise EvidenceContractError("plan source bindings are missing")
    for key, value in sources.items():
        _hash_ref(value, f"plan.source_bindings.{key}")
    return {
        "document": dict(document),
        "path": info.get("path"),
        "raw_sha256": info["raw_sha256"],
        "size_bytes": info.get("size_bytes"),
        "self_sha256": document["self_sha256"],
        "events": event_refs,
    }


def validate_plan(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    document, info = _read_json(value, "crossover plan")
    return _validate_plan(document, info)


def _self_hash_field(document: Mapping[str, Any], label: str) -> str:
    """Return a document's own semantic self hash under its declared field."""

    field = "self_sha256" if "self_sha256" in document else "result_sha256"
    if field not in document:
        raise EvidenceContractError(f"{label} carries no semantic self hash")
    _self_check(document, label, field=field)
    return str(document[field])


def _successor_file_ref(value: Any, label: str, *, expected_path: Path | None = None) -> Path:
    """Require a successor tool/authority ref to match the live file exactly."""

    if not isinstance(value, Mapping):
        raise EvidenceContractError(f"{label} must be an object")
    target = _regular_file(value.get("path"), label)
    if expected_path is not None and target != expected_path:
        raise EvidenceContractError(f"{label} path differs from the running finalization tool")
    if _sha(value.get("sha256"), f"{label}.sha256") != sha256_file(target):
        raise EvidenceContractError(f"{label} differs from the live file")
    return target


def _validate_finalization_receipt(
    value: str | Path | Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    pre_gpu_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the one authorized post-execution successor receipt.

    The successor exists because the parent receipt binds this finalizer's own
    hash while the completed shards bind the parent's semantic self hash.  It
    may advance exactly two named slots; everything else it declares must still
    equal the live artifact, and it may never bind an evidence hash.
    """

    document, info = _read_json(value, "finalization receipt")
    if (
        document.get("schema_version") != FINALIZATION_RECEIPT_SCHEMA_VERSION
        or document.get("status") != FINALIZATION_RECEIPT_STATUS
        or document.get("unit_id") != UNIT_ID
    ):
        raise EvidenceContractError("finalization receipt schema/status/unit identity drifted")
    _self_check(document, "finalization receipt")

    authorization = document.get("authorization")
    if not isinstance(authorization, Mapping):
        raise EvidenceContractError("finalization receipt authorization is missing")
    for key, expected in (
        ("cpu_only", True),
        ("gpu_used", False),
        ("model_loaded", False),
        ("no_training", True),
        ("endpoint_semantics_unchanged", True),
    ):
        if _bool(authorization.get(key), f"finalization receipt authorization.{key}") is not expected:
            raise EvidenceContractError(f"finalization receipt authorization.{key} must be {str(expected).lower()}")
    _successor_file_ref(authorization.get("authority"), "finalization authority document")

    parent = document.get("parent")
    if not isinstance(parent, Mapping) or not isinstance(parent.get("document"), Mapping):
        raise EvidenceContractError("finalization receipt parent binding is incomplete")
    parent_ref = parent.get("receipt")
    if not isinstance(parent_ref, Mapping):
        raise EvidenceContractError("finalization receipt parent receipt binding is missing")
    parent_path = _regular_file(parent_ref.get("path"), "finalization parent receipt")
    if pre_gpu_path is not None and parent_path != _absolute(pre_gpu_path, "supplied pre-GPU receipt"):
        raise EvidenceContractError("finalization receipt binds a different parent receipt path")
    parent_document, parent_info = _read_json(parent_path, "finalization parent receipt")
    parent_raw = parent_info["raw_sha256"]
    parent_self = _self_hash_field(parent_document, "finalization parent receipt")
    if _sha(parent_ref.get("raw_sha256"), "finalization parent receipt.raw_sha256") != parent_raw:
        raise EvidenceContractError("finalization receipt parent raw SHA-256 differs from the live parent")
    if _sha(parent_ref.get("self_sha256"), "finalization parent receipt.self_sha256") != parent_self:
        raise EvidenceContractError("finalization receipt parent self SHA-256 differs from the live parent")
    if dict(parent["document"]) != parent_document:
        raise EvidenceContractError("finalization receipt parent document differs from the live parent")

    plan_ref = parent.get("plan")
    if not isinstance(plan_ref, Mapping):
        raise EvidenceContractError("finalization receipt plan binding is missing")
    plan_path = _regular_file(plan_ref.get("path"), "finalization plan")
    if plan.get("path") and plan_path != _absolute(plan["path"], "supplied plan"):
        raise EvidenceContractError("finalization receipt binds a different plan path")
    if _sha(plan_ref.get("raw_sha256"), "finalization plan.raw_sha256") != plan["raw_sha256"]:
        raise EvidenceContractError("finalization receipt plan raw SHA-256 differs from the supplied plan")
    if _sha(plan_ref.get("self_sha256"), "finalization plan.self_sha256") != plan["self_sha256"]:
        raise EvidenceContractError("finalization receipt plan self SHA-256 differs from the supplied plan")

    execution = document.get("execution")
    if not isinstance(execution, Mapping) or not isinstance(execution.get("root"), Mapping):
        raise EvidenceContractError("finalization receipt execution binding is incomplete")
    if execution["root"].get("status") != "complete_immutable":
        raise EvidenceContractError("finalization receipt execution root is not declared complete_immutable")
    execution_root = _regular_dir(execution["root"].get("path"), "finalization execution root")
    shards = execution.get("shards")
    if not isinstance(shards, list) or len(shards) != len(EVENT_IDS):
        raise EvidenceContractError("finalization receipt must bind exactly three shards")
    for index, shard in enumerate(shards):
        shard_id = f"shard-{index:03d}"
        label = f"finalization {shard_id}"
        if not isinstance(shard, Mapping) or shard.get("shard_id") != shard_id:
            raise EvidenceContractError(f"{label} identity/order drifted")
        event = plan["events"][index]
        if (
            shard.get("event_id") != EVENT_IDS[index]
            or shard.get("event_id") != event["event_id"]
            or shard.get("event_index") != event["event_index"]
            or shard.get("image_id") != event["image_id"]
            or shard.get("physical_device") != DEVICE_PLAN[shard_id]
        ):
            raise EvidenceContractError(f"{label} event/device binding differs from the plan")
        root = _regular_dir(shard.get("root"), f"{label} root")
        if root != execution_root / shard_id:
            raise EvidenceContractError(f"{label} root is not execution_root/{shard_id}")
        artifacts = shard.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(FINALIZATION_SHARD_ARTIFACTS):
            raise EvidenceContractError(f"{label} artifact set drifted")
        for name in FINALIZATION_SHARD_ARTIFACTS:
            ref = artifacts[name]
            if not isinstance(ref, Mapping):
                raise EvidenceContractError(f"{label}/{name} binding is malformed")
            artifact_path = _regular_file(ref.get("path"), f"{label}/{name}")
            if artifact_path != root / name:
                raise EvidenceContractError(f"{label}/{name} path is not under its shard root")
            artifact_document, artifact_info = _read_json(artifact_path, f"{label}/{name}")
            if _sha(ref.get("raw_sha256"), f"{label}/{name}.raw_sha256") != artifact_info["raw_sha256"]:
                raise EvidenceContractError(f"{label}/{name} raw SHA-256 differs from the immutable artifact")
            if _sha(ref.get("self_sha256"), f"{label}/{name}.self_sha256") != _self_hash_field(
                artifact_document, f"{label}/{name}"
            ):
                raise EvidenceContractError(f"{label}/{name} self SHA-256 differs from the immutable artifact")

    evidence_binding = document.get("evidence_root")
    if not isinstance(evidence_binding, Mapping) or evidence_binding.get("state") != "absent_at_seal":
        raise EvidenceContractError("finalization receipt evidence root binding is incomplete")
    if any(key in evidence_binding for key in ("sha256", "raw_sha256", "self_sha256")):
        raise EvidenceContractError("finalization receipt must not bind an evidence hash")
    evidence_root = _absolute(evidence_binding.get("path"), "finalization evidence root")
    if evidence_root.is_symlink() or evidence_root.exists():
        raise EvidenceContractError("finalization evidence root must still be absent and non-symlink")

    drift = document.get("authorized_drift")
    if not isinstance(drift, Mapping) or set(drift) != set(AUTHORIZED_FINALIZATION_SLOTS):
        raise EvidenceContractError(
            f"finalization receipt must authorize exactly {list(AUTHORIZED_FINALIZATION_SLOTS)}"
        )
    tools = document.get("finalization_tools")
    if not isinstance(tools, Mapping) or set(tools) != set(FINALIZATION_TOOL_ROLES):
        raise EvidenceContractError(f"finalization receipt tools must be exactly {list(FINALIZATION_TOOL_ROLES)}")
    finalizer_path = _successor_file_ref(
        tools["finalizer"], "finalization tool finalizer", expected_path=_regular_file(Path(__file__).resolve(), "running finalizer")
    )
    finalizer_test_path = _successor_file_ref(tools["finalizer_test"], "finalization tool finalizer_test")
    _successor_file_ref(tools["successor_sealer"], "finalization tool successor_sealer")
    _successor_file_ref(tools["successor_sealer_test"], "finalization tool successor_sealer_test")

    expected_slot_paths = {
        "source_files.crossover_finalizer": finalizer_path,
        "test_files.crossover_finalizer_test": finalizer_test_path,
    }
    allowance: dict[str, str] = {}
    for slot in AUTHORIZED_FINALIZATION_SLOTS:
        entry = drift[slot]
        group_name, _, role = slot.partition(".")
        if not isinstance(entry, Mapping) or entry.get("group") != group_name or entry.get("role") != role:
            raise EvidenceContractError(f"finalization receipt {slot} identity drifted")
        parent_group = parent_document.get(group_name)
        if not isinstance(parent_group, Mapping) or not isinstance(parent_group.get(role), Mapping):
            raise EvidenceContractError(f"finalization receipt {slot} is not a parent binding")
        parent_binding = parent_group[role]
        slot_path = _regular_file(entry.get("path"), f"finalization {slot}")
        if slot_path != _absolute(parent_binding.get("path"), f"parent {slot}"):
            raise EvidenceContractError(f"finalization receipt {slot} path differs from the parent binding")
        if slot_path != expected_slot_paths[slot]:
            raise EvidenceContractError(f"finalization receipt {slot} path is not the declared finalization tool")
        if _sha(entry.get("old_sha256"), f"finalization {slot}.old_sha256") != parent_binding.get("sha256"):
            raise EvidenceContractError(f"finalization receipt {slot} old hash differs from the parent binding")
        if _sha(entry.get("new_sha256"), f"finalization {slot}.new_sha256") != sha256_file(slot_path):
            raise EvidenceContractError(f"finalization receipt {slot} new hash differs from the live file")
        if not _text(entry.get("cause"), f"finalization {slot}.cause"):
            raise EvidenceContractError(f"finalization receipt {slot} states no cause")
        allowance[slot] = entry["new_sha256"]

    unchanged = document.get("unchanged_parent_bindings")
    if not isinstance(unchanged, Mapping):
        raise EvidenceContractError("finalization receipt unchanged parent bindings are missing")
    for group_name in ("input_bindings", "source_files", "test_files"):
        parent_group = parent_document.get(group_name)
        declared = unchanged.get(group_name)
        if not isinstance(parent_group, Mapping) or not isinstance(declared, Mapping):
            raise EvidenceContractError(f"finalization receipt unchanged {group_name} are missing")
        expected_roles = {role for role in parent_group if f"{group_name}.{role}" not in AUTHORIZED_FINALIZATION_SLOTS}
        if set(declared) != expected_roles:
            raise EvidenceContractError(f"finalization receipt unchanged {group_name} set differs from the parent")
        for role, entry in declared.items():
            if not isinstance(entry, Mapping) or entry.get("sha256") != parent_group[role].get("sha256"):
                raise EvidenceContractError(f"finalization receipt unchanged {group_name}.{role} differs from the parent")

    pins = document.get("runtime_input_pins")
    expected_pins = {
        "pre_gpu_receipt": parent_raw,
        "pre_gpu_receipt_sha256": parent_raw,
        "pre_gpu_receipt_self_sha256": parent_self,
    }
    if not isinstance(pins, Mapping) or dict(pins) != expected_pins:
        raise EvidenceContractError("finalization receipt runtime input pins differ from the recomputed parent")
    _hash_ref(document.get("producer"), "finalization receipt producer")

    return {
        "document": dict(document),
        "path": info.get("path"),
        "raw_sha256": info["raw_sha256"],
        "size_bytes": info.get("size_bytes"),
        "self_sha256": document["self_sha256"],
        "allowance": allowance,
        "parent_raw_sha256": parent_raw,
        "parent_self_sha256": parent_self,
        "evidence_root": str(evidence_root),
    }


def validate_finalization_receipt(
    value: str | Path | Mapping[str, Any],
    plan: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    plan_info = plan if isinstance(plan, Mapping) and "self_sha256" in plan else validate_plan(plan)
    return _validate_finalization_receipt(value, plan_info)


def _validate_pre_gpu(
    document: Mapping[str, Any],
    info: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    successor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if document.get("schema_version") != PRE_GPU_SCHEMA_VERSION or document.get("status") != "sealed_pre_gpu" or document.get("unit_id") != UNIT_ID:
        raise EvidenceContractError("pre-GPU receipt schema/status/unit identity drifted")
    _self_check(document, "pre-GPU receipt")
    if document.get("no_training") is not True or document.get("use_cache") is True:
        raise EvidenceContractError("pre-GPU receipt crosses no-training/cache boundary")
    if document.get("plan_self_sha256") != plan["self_sha256"]:
        raise EvidenceContractError("pre-GPU receipt binds a different plan")
    plan_ref = document.get("plan")
    if isinstance(plan_ref, Mapping):
        _hash_ref(plan_ref, "pre-GPU plan")
        if plan_ref.get("path") and plan["path"] and str(Path(plan_ref["path"]).resolve()) != str(Path(plan["path"]).resolve()):
            raise EvidenceContractError("pre-GPU plan path differs from supplied plan")
    bindings = document.get("event_bindings")
    expected_bindings = [
        {
            "event_index": event["event_index"],
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "event_sha256": event["event_sha256"],
            **({"prefix_sha256": event["prefix_sha256"]} if "prefix_sha256" in event else {}),
            **({"geometry_sha256": event["geometry_sha256"]} if "geometry_sha256" in event else {}),
        }
        for event in plan["events"]
    ]
    if bindings is None or bindings != expected_bindings:
        raise EvidenceContractError("pre-GPU event/prefix/geometry bindings drifted")
    selection = document.get("source_selection")
    if selection is not None:
        if not isinstance(selection, Mapping) or selection.get("event_ids") != list(EVENT_IDS) or selection.get("event_count") != 3 or selection.get("image_count") != 3 or selection.get("cell_order") != list(CELL_ORDER) or selection.get("opener_injected") is not False or selection.get("use_cache") is not False:
            raise EvidenceContractError("pre-GPU source selection drifted")
    device_plan = document.get("device_plan")
    if isinstance(device_plan, Mapping) and "device_plan" in device_plan:
        device_plan = device_plan["device_plan"]
    if device_plan != DEVICE_PLAN:
        raise EvidenceContractError("pre-GPU device plan must bind shard-000/001/002 to physical 0/1/7")
    policy = document.get("execution_policy")
    if policy is not None:
        if not isinstance(policy, Mapping) or any(policy.get(key) is not True for key in ("no_event_reorder", "no_reselection", "no_sweep", "no_a3", "no_p4", "no_training", "at_most_once")):
            raise EvidenceContractError("pre-GPU execution policy is unsafe")
    runtime = document.get("runtime")
    if runtime is not None:
        if not isinstance(runtime, Mapping) or runtime.get("no_training") is not True or runtime.get("gpu_used") is True or runtime.get("model_loaded") is True:
            raise EvidenceContractError("pre-GPU runtime is not a CPU-only no-training identity")
    for group_name in ("input_bindings", "source_files", "test_files"):
        group = document.get(group_name)
        if group is None and group_name == "test_files":
            continue
        if not isinstance(group, Mapping) or not group:
            raise EvidenceContractError(f"pre-GPU {group_name} are incomplete")
        allowance = dict(successor["allowance"]) if successor is not None else {}
        for key, value in group.items():
            _hash_ref(
                value,
                f"pre-GPU {group_name}.{key}",
                expected_sha256=allowance.get(f"{group_name}.{key}"),
            )
    input_hashes = document.get("input_hashes")
    if not isinstance(input_hashes, Mapping) or not input_hashes:
        raise EvidenceContractError("pre-GPU input_hashes are missing")
    for key, value in input_hashes.items():
        _sha(value, f"pre-GPU input_hashes.{key}")
    roots = document.get("roots")
    if not isinstance(roots, Mapping):
        raise EvidenceContractError("pre-GPU roots are missing")
    for root_name in ("execution_root", "final_root"):
        binding = roots.get(root_name)
        if not isinstance(binding, Mapping):
            raise EvidenceContractError(f"pre-GPU roots.{root_name} is missing")
        root_path = binding.get("path")
        if not isinstance(root_path, str) or not root_path:
            raise EvidenceContractError(f"pre-GPU roots.{root_name}.path is missing")
        _absolute(root_path, f"pre-GPU roots.{root_name}")
    return {
        "document": dict(document),
        "path": info.get("path"),
        "raw_sha256": info["raw_sha256"],
        "size_bytes": info.get("size_bytes"),
        "self_sha256": document["self_sha256"],
    }


def validate_pre_gpu_receipt(
    value: str | Path | Mapping[str, Any],
    plan: Mapping[str, Any] | str | Path,
    *,
    successor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(plan, Mapping) and isinstance(plan.get("document"), Mapping) and "self_sha256" in plan:
        plan_info = dict(plan)
    elif isinstance(plan, Mapping) and "self_sha256" in plan:
        plan_info = _validate_plan(dict(plan), {"path": None, "raw_sha256": sha256_json(plan), "size_bytes": len(canonical_json_bytes(plan))})
    else:
        plan_info = validate_plan(plan)
    document, info = _read_json(value, "pre-GPU receipt")
    return _validate_pre_gpu(document, info, plan_info, successor=successor)


def _recursive_false_flags(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered == "use_cache" and nested is not False:
                raise EvidenceContractError(f"{label}.{key} must be false")
            if lowered in {"opener_injected", "synthetic_opener_injections"} and nested not in (False, 0):
                raise EvidenceContractError(f"{label}.{key} injects an opener")
            _recursive_false_flags(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _recursive_false_flags(nested, f"{label}[{index}]")


def _find_layer_attestation(receipt: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Find the real per-forward attestation without accepting a callback stub."""

    direct = receipt.get("layer_consumption_attestation") or receipt.get("all_layer_consumption_attestation")
    if isinstance(direct, Mapping):
        return direct
    attention_receipt = receipt.get("attention_actuation_receipt")
    if isinstance(attention_receipt, Mapping):
        nested = attention_receipt.get("layer_consumption_attestation") or attention_receipt.get("all_layer_consumption_attestation")
        if isinstance(nested, Mapping):
            return nested
    attention_mask = receipt.get("attention_mask")
    if isinstance(attention_mask, Mapping):
        nested = attention_mask.get("receipt")
        if isinstance(nested, Mapping):
            nested = nested.get("layer_consumption_attestation") or nested.get("all_layer_consumption_attestation")
            if isinstance(nested, Mapping):
                return nested
    return None


def _construction_placeholder_kind(attestation: Mapping[str, Any]) -> str | None:
    """Name the exact pre-forward placeholder ``attestation`` is, if any.

    These are the two shapes ``natural_boundary_attention_actuators`` puts on a
    receipt this unit consumes: the ``base`` one every built K01/K10/H20
    actuator carries, and the ``composed`` one ``compose_k10_h20`` writes on a
    C11 receipt, which also pins the composed geometry and an explicit
    ``passed: false``.  This module stays standard-library only -- it is the
    CPU evidence consumer and must not import the torch-backed producer -- so
    the shapes are restated here and pinned against the real producer in
    ``tests/research/test_finalize_s_k10_h20_crossover.py``.
    """

    keys = set(attestation)
    if (
        attestation.get("schema_version") != CONSUMPTION_SCHEMA_VERSION
        or attestation.get("required") is not True
        or attestation.get("status") != "unattested"
        or attestation.get("exact_same_tensor_all_layers_required") is not True
    ):
        return None
    if keys == PLACEHOLDER_BASE_KEYS:
        return "base"
    if keys != PLACEHOLDER_BASE_KEYS | {"declared_layer_count", "declared_sequence_length", "passed"}:
        return None
    declared_length = attestation.get("declared_sequence_length")
    if (
        attestation.get("declared_layer_count") != 28
        or isinstance(declared_length, bool)
        or not isinstance(declared_length, int)
        or declared_length <= 0
        or attestation.get("passed") is not False
    ):
        return None
    return "composed"


def _validate_layer_consumption(
    receipt: Mapping[str, Any],
    label: str,
    *,
    required: bool = True,
) -> None:
    """Validate no-cache identity and, when required, an exact all-28 proof."""

    if receipt.get("use_cache") is not False:
        raise EvidenceContractError(f"{label} is not a no-cache scalar receipt")
    attestation = _find_layer_attestation(receipt)
    if attestation is None:
        if required:
            raise EvidenceContractError(f"{label} lacks explicit all-28-layer consumption")
        return
    if not required:
        # Lightweight scalar receipts are recorded before a model forward and
        # therefore carry only an exact construction placeholder.  It is not
        # evidence of consumption and must never satisfy runtime proof.
        if _construction_placeholder_kind(attestation) is not None:
            return
    missing = attestation.get("missing_layers", [])
    repeated = attestation.get("repeated_layers", [])
    errors = attestation.get("errors", [])
    if attestation.get("passed") is not True or missing != [] or repeated != [] or errors != []:
        raise EvidenceContractError(f"{label} all-layer consumption did not pass")
    layer_count = attestation.get("layer_count")
    layer_indices = attestation.get("layer_indices")
    if layer_count != 28 or layer_indices != list(range(28)):
        raise EvidenceContractError(f"{label} does not attest exactly layers 0..27")


def _token_id_list(value: Any, label: str, *, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise EvidenceContractError(f"{label} must be a token-ID list")
    result: list[int] = []
    for index, token in enumerate(value):
        result.append(_int(token, f"{label}[{index}]"))
    return result


def _receipt_hash(receipt: Mapping[str, Any], label: str, names: Sequence[str]) -> str:
    for name in names:
        value = receipt.get(name)
        if value is not None:
            return _sha(value, f"{label}.{name}")
    raise EvidenceContractError(f"{label} is missing {'/'.join(names)}")


def _validate_receipt_sequence(
    receipts: Sequence[Any],
    *,
    label: str,
    prefix_tokens: Sequence[int],
    generated_tokens: Sequence[int],
    require_layer_attestation: bool,
    runtime_receipt: bool,
) -> tuple[list[str], list[str]]:
    if not receipts:
        raise EvidenceContractError(f"{label} is empty")
    input_hashes: list[str] = []
    mrope_hashes: list[str] = []
    base = list(prefix_tokens)
    for index, value in enumerate(receipts):
        if not isinstance(value, Mapping):
            raise EvidenceContractError(f"{label}[{index}] is malformed")
        step = _int(value.get("step"), f"{label}[{index}].step")
        if step != index:
            raise EvidenceContractError(f"{label}[{index}] step/order is not contiguous")
        sequence_length_value = value.get("sequence_length")
        if runtime_receipt:
            sequence_length = _int(sequence_length_value, f"{label}[{index}].sequence_length")
            if sequence_length != len(base) + step:
                raise EvidenceContractError(f"{label}[{index}] sequence_length differs from growing prefix")
        elif sequence_length_value is not None and _int(sequence_length_value, f"{label}[{index}].sequence_length") != len(base) + step:
            raise EvidenceContractError(f"{label}[{index}] sequence_length differs from growing prefix")
        expected_ids = base + list(generated_tokens[:step])
        expected_hash = sha256_json(expected_ids)
        observed_hash = _receipt_hash(value, f"{label}[{index}]", ("input_ids_sha256", "prefix_token_ids_sha256", "input_token_hash"))
        if observed_hash != expected_hash:
            raise EvidenceContractError(f"{label}[{index}] input hash differs from raw growing prefix")
        supplied_ids = value.get("input_ids")
        if supplied_ids is not None and _token_id_list(supplied_ids, f"{label}[{index}].input_ids") != expected_ids:
            raise EvidenceContractError(f"{label}[{index}] input_ids differ from raw growing prefix")
        mrope_value = value.get("mrope_hash", value.get("mrope_sha256"))
        mrope = _sha(mrope_value, f"{label}[{index}].mrope_hash") if mrope_value is not None else ""
        input_hashes.append(observed_hash)
        mrope_hashes.append(mrope)
        _validate_layer_consumption(value, f"{label}[{index}]", required=require_layer_attestation)
    if runtime_receipt and len(mrope_hashes) > 1 and any(left == right for left, right in zip(mrope_hashes, mrope_hashes[1:])):
        raise EvidenceContractError(f"{label} M-RoPE hashes do not change at every growing step")
    return input_hashes, mrope_hashes


def _validate_scalars(
    raw: Mapping[str, Any],
    label: str,
    *,
    prefix_tokens: Sequence[int],
    generated_tokens: Sequence[int],
    require_layer_attestation: bool,
) -> dict[str, Any]:
    scalar = raw.get("scalar_receipts")
    runtime = raw.get("runtime_scalar_receipts")
    if not isinstance(scalar, list) or not scalar:
        raise EvidenceContractError(f"{label} scalar receipt list is incomplete")
    if not isinstance(runtime, list) or not runtime:
        raise EvidenceContractError(f"{label} runtime scalar receipt list is incomplete")
    if len(scalar) != len(runtime):
        raise EvidenceContractError(f"{label} scalar/runtime receipt lengths differ")
    if raw.get("scalar_forward_count") != len(scalar) or raw.get("runtime_scalar_forward_count") != len(runtime):
        raise EvidenceContractError(f"{label} scalar count summaries disagree with receipts")
    scalar_hashes, scalar_mrope = _validate_receipt_sequence(
        scalar,
        label=f"{label}.scalar_receipts",
        prefix_tokens=prefix_tokens,
        generated_tokens=generated_tokens,
        require_layer_attestation=False,
        runtime_receipt=False,
    )
    runtime_hashes, runtime_mrope = _validate_receipt_sequence(
        runtime,
        label=f"{label}.runtime_scalar_receipts",
        prefix_tokens=prefix_tokens,
        generated_tokens=generated_tokens,
        require_layer_attestation=require_layer_attestation,
        runtime_receipt=True,
    )
    if scalar_hashes != runtime_hashes:
        raise EvidenceContractError(f"{label} scalar/runtime input identities differ")
    return {
        "scalar_forward_count": len(scalar),
        "runtime_scalar_forward_count": len(runtime),
        "input_ids_hashes": runtime_hashes,
        "mrope_hashes": runtime_mrope,
        "initial_mrope_hash": runtime_mrope[0],
    }


def _owner_id_list(value: Any, label: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise EvidenceContractError(f"{label} must be an owner-ID list")
    result: list[str] = []
    for index, owner in enumerate(value):
        owner_text = _text(owner, f"{label}[{index}]")
        if owner_text in result:
            raise EvidenceContractError(f"{label} contains duplicate owner IDs")
        result.append(owner_text)
    return result


def _strict_owner(row: Mapping[str, Any], label: str) -> tuple[str | None, bool, str | None]:
    match = row.get("owner_match")
    status = None
    owner = None
    if isinstance(match, Mapping):
        status = match.get("status")
        owner_value = match.get("owner_id", match.get("matched_owner_id"))
        if owner_value is not None:
            owner = _text(owner_value, f"{label}.owner_match.owner_id")
        if status not in {"unique", "matched", "unmatched", "ambiguous"}:
            raise EvidenceContractError(f"{label}.owner_match.status is unknown")
        if status in {"unmatched", "ambiguous"} and owner is not None:
            raise EvidenceContractError(f"{label} neutral owner match claims an owner")
        if status in {"unique", "matched"} and owner is None:
            raise EvidenceContractError(f"{label} matched owner identity is missing")
        if not isinstance(match.get("source_specific"), bool) or not isinstance(match.get("physical_match"), bool):
            raise EvidenceContractError(f"{label} owner source/physical flags are missing")
    else:
        if row.get("status") in {"closure", "accepted", "complete"}:
            raise EvidenceContractError(f"{label} complete row lacks owner_match")
    complete = row.get("status") in {"closure", "accepted", "complete"} or row.get("complete") is True
    strict = bool(
        complete
        and status in {"unique", "matched"}
        and isinstance(match, Mapping)
        and match.get("source_specific") is True
        and match.get("physical_match") is True
        and owner is not None
    )
    if not complete and isinstance(match, Mapping):
        raise EvidenceContractError(f"{label} non-complete row carries stale owner identity")
    return owner, strict, status


def _parity(raw: Mapping[str, Any], label: str) -> dict[str, Any]:
    value = raw.get("parity_control") or raw.get("c00_parity") or raw.get("full_logit_parity")
    if not isinstance(value, Mapping):
        raise EvidenceContractError(f"{label} parity receipt is missing")
    if value.get("passed") is not True:
        raise EvidenceContractError(f"{label} parity receipt did not pass")
    tolerance = _finite(value.get("tolerance"), f"{label}.parity.tolerance")
    delta = value.get("per_forward_max_abs_delta", value.get("max_abs_delta"))
    if delta is not None and _finite(delta, f"{label}.parity.delta") > min(tolerance, 1e-4):
        raise EvidenceContractError(f"{label} parity delta exceeds tolerance")
    return dict(value)


def _validate_natural_raw(raw: Mapping[str, Any], expected_arm: str, label: str, *, event: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("arm_id") != expected_arm:
        raise EvidenceContractError(f"{label}.arm_id differs from transport arm {expected_arm}")
    try:
        canonical_json_bytes(raw)
    except EvidenceContractError:
        raise
    except Exception as exc:
        raise EvidenceContractError(f"{label} production natural parser validation failed: {exc}") from exc
    if raw.get("admission_mode") != "pre_opener_natural" or raw.get("opener_injected") is not False or raw.get("synthetic_opener_injections") != 0:
        raise EvidenceContractError(f"{label} is not exact natural pre-opener admission")
    if raw.get("opener_seeded") is True or raw.get("seed_provenance") not in (None, False, "") or raw.get("opener_seed_provenance") not in (None, False, ""):
        raise EvidenceContractError(f"{label} carries seeded opener provenance")
    if raw.get("event_id") is not None and raw.get("event_id") != event.get("event_id"):
        raise EvidenceContractError(f"{label}.event_id differs from plan event")
    if expected_arm == TECHNICAL_CELL:
        def has_attention_callback(value: Any) -> bool:
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    if str(key) in {"attention_actuation_receipt", "attention_mask_actuator", "score_bias_callback"} and isinstance(nested, Mapping):
                        return True
                    if has_attention_callback(nested):
                        return True
            elif isinstance(value, list):
                return any(has_attention_callback(item) for item in value)
            return False
        if has_attention_callback(raw):
            raise EvidenceContractError(f"{label} technical K00 control carries an attention callback receipt")
    _recursive_false_flags(raw, label)
    prefix = raw.get("prefix")
    prefix_identity = raw.get("prefix_identity")
    if not isinstance(prefix, Mapping):
        raise EvidenceContractError(f"{label}.prefix is missing")
    exact_history = _token_id_list(prefix.get("exact_history_token_ids"), f"{label}.prefix.exact_history_token_ids", allow_empty=False)
    prefix_tokens = _token_id_list(prefix.get("prefix_token_ids"), f"{label}.prefix.prefix_token_ids", allow_empty=False)
    history_sha = sha256_json(exact_history)
    full_prefix_sha = sha256_json(prefix_tokens)
    if prefix.get("prefix_sha256", prefix.get("history_sha256")) not in (None, history_sha):
        raise EvidenceContractError(f"{label} history-row prefix hash differs from exact history tokens")
    if prefix.get("prefix_token_ids_sha256") not in (None, full_prefix_sha):
        raise EvidenceContractError(f"{label} full prefix token hash differs from raw prefix tokens")
    if isinstance(prefix_identity, Mapping):
        natural_tokens = prefix_identity.get("natural_prefix_token_ids")
        if natural_tokens is not None and _token_id_list(natural_tokens, f"{label}.prefix_identity.natural_prefix_token_ids", allow_empty=False) != prefix_tokens:
            raise EvidenceContractError(f"{label} prefix identity tokens differ from raw full prefix")
        natural_sha = prefix_identity.get("natural_prefix_sha256", prefix_identity.get("prefix_sha256"))
        if natural_sha is not None and natural_sha != full_prefix_sha:
            raise EvidenceContractError(f"{label} prefix identity hash differs from raw full prefix")
    if prefix.get("event_id") is not None and prefix.get("event_id") != event.get("event_id"):
        raise EvidenceContractError(f"{label}.prefix.event_id differs from plan event")
    generated_tokens = _token_id_list(raw.get("generated_token_ids"), f"{label}.generated_token_ids")
    if raw.get("generated_token_ids_sha256") not in (None, sha256_json(generated_tokens)):
        raise EvidenceContractError(f"{label}.generated_token_ids hash differs from tokens")
    opener = _int(raw.get("opener_token_id"), f"{label}.opener_token_id")
    initial_last = _int(raw.get("initial_prefix_last_token_id"), f"{label}.initial_prefix_last_token_id")
    if initial_last != prefix_tokens[-1] or initial_last == opener:
        raise EvidenceContractError(f"{label} natural prefix already ends with opener")
    first = _int(raw.get("first_generated_token_id"), f"{label}.first_generated_token_id")
    opener_generated = _bool(raw.get("opener_generated_by_model"), f"{label}.opener_generated_by_model")
    if opener_generated != (first == opener):
        raise EvidenceContractError(f"{label} opener_generated_by_model disagrees with first token")
    native_stop_ids = raw.get("native_stop_token_ids")
    if not isinstance(native_stop_ids, list):
        raise EvidenceContractError(f"{label}.native_stop_token_ids is malformed")
    native_stop_ids = [
        _int(token, f"{label}.native_stop_token_ids[{index}]")
        for index, token in enumerate(native_stop_ids)
    ]
    if len(set(native_stop_ids)) != len(native_stop_ids):
        raise EvidenceContractError(f"{label}.native_stop_token_ids contains duplicates")
    terminal = raw.get("terminal_reason")
    if terminal not in SCIENTIFIC_TERMINALS:
        raise EvidenceContractError(f"{label}.terminal_reason is unknown")
    if first in native_stop_ids and terminal != "native_stop":
        raise EvidenceContractError(f"{label} native terminal token disagrees with terminal_reason")
    scalar = _validate_scalars(
        raw,
        label,
        prefix_tokens=prefix_tokens,
        generated_tokens=generated_tokens,
        require_layer_attestation=expected_arm != TECHNICAL_CELL,
    )
    prefix_sha = history_sha
    mrope = scalar["initial_mrope_hash"]
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        raise EvidenceContractError(f"{label}.rows is missing")
    strict_sequence: list[str] = []
    book_value = raw.get("owner_bookkeeping")
    covered_value = raw.get("covered_owner_ids")
    if covered_value is None and isinstance(book_value, Mapping):
        covered_value = book_value.get("covered_owner_ids")
    if covered_value is None and rows and isinstance(rows[0], Mapping) and isinstance(rows[0].get("owner_bookkeeping"), Mapping):
        covered_value = rows[0]["owner_bookkeeping"].get("covered_owner_ids_before")
    covered = _owner_id_list(covered_value if covered_value is not None else [], f"{label}.covered_owner_ids")
    covered_set = set(covered)
    seen: set[str] = set()
    parsed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("row_index") != index:
            raise EvidenceContractError(f"{label}.rows[{index}] identity/order is malformed")
        row_synthetic_injections = row.get("synthetic_opener_injections", 0)
        if (
            row.get("admission_mode") != "pre_opener_natural"
            or row.get("opener_injected") is not False
            or isinstance(row_synthetic_injections, bool)
            or not isinstance(row_synthetic_injections, int)
            or row_synthetic_injections != 0
        ):
            raise EvidenceContractError(f"{label}.rows[{index}] is not natural pre-opener admission")
        if row.get("opener_seeded") is True or row.get("seed_provenance") not in (None, False, "") or row.get("opener_seed_provenance") not in (None, False, ""):
            raise EvidenceContractError(f"{label}.rows[{index}] carries seeded opener provenance")
        if row.get("opener_token_id") != opener:
            raise EvidenceContractError(f"{label}.rows[{index}] opener identity differs")
        row_tokens = _token_id_list(row.get("token_ids"), f"{label}.rows[{index}].token_ids", allow_empty=True)
        if row.get("token_ids_sha256") != sha256_json(row_tokens):
            raise EvidenceContractError(f"{label}.rows[{index}] token hash differs from tokens")
        row_first_value = row.get("first_generated_token_id")
        if row_tokens:
            row_first = _int(row_first_value, f"{label}.rows[{index}].first_generated_token_id")
            if row_first != row_tokens[0]:
                raise EvidenceContractError(f"{label}.rows[{index}] first token differs from trajectory")
        else:
            row_first = None
            if row_first_value is not None:
                raise EvidenceContractError(f"{label}.rows[{index}] first token differs from empty trajectory")
        row_last = _int(row.get("initial_prefix_last_token_id"), f"{label}.rows[{index}].initial_prefix_last_token_id")
        if row_last == opener:
            raise EvidenceContractError(f"{label}.rows[{index}] prefix already ends with opener")
        if index == 0 and row_last != initial_last:
            raise EvidenceContractError(f"{label} arm/first-row natural-boundary receipts disagree")
        row_generated = _bool(row.get("opener_generated_by_model"), f"{label}.rows[{index}].opener_generated_by_model")
        if row_generated != bool(row_tokens and row_first == opener):
            raise EvidenceContractError(f"{label}.rows[{index}] opener provenance disagrees with first token")
        row_started_value = row.get("row_started")
        if row_started_value is not None:
            row_started_value = _bool(row_started_value, f"{label}.rows[{index}].row_started")
            if row_started_value != row_generated:
                raise EvidenceContractError(f"{label}.rows[{index}] row_started disagrees with opener provenance")
        row_started = row_generated
        status = row.get("status")
        if status not in SCIENTIFIC_TERMINALS:
            raise EvidenceContractError(f"{label}.rows[{index}] has unknown scientific status")
        if status == "closure" and not row_generated:
            raise EvidenceContractError(f"{label}.rows[{index}] closure lacks model-generated opener")
        stop_positions = [position for position, token in enumerate(row_tokens) if token in native_stop_ids]
        if stop_positions and (status != "native_stop" or stop_positions != [len(row_tokens) - 1]):
            raise EvidenceContractError(f"{label}.rows[{index}] native STOP token is misplaced")
        if status == "native_stop" and (not row_tokens or row_tokens[-1] not in native_stop_ids):
            raise EvidenceContractError(f"{label}.rows[{index}] native STOP token is not terminal")
        if row_tokens and row_first in native_stop_ids and status != "native_stop":
            raise EvidenceContractError(f"{label}.rows[{index}] native terminal token disagrees with status")
        if index == 0 and (row_first != first or row_generated != opener_generated):
            raise EvidenceContractError(f"{label} arm/first-row natural-boundary receipts disagree")
        owner, strict, match_status = _strict_owner(row, f"{label}.rows[{index}]")
        covered_repeat = bool(strict and owner in covered_set)
        duplicate = bool(strict and owner in seen)
        seen_before = sorted(covered_set | seen)
        row_book = row.get("owner_bookkeeping")
        if not isinstance(row_book, Mapping):
            raise EvidenceContractError(f"{label}.rows[{index}] owner bookkeeping is missing")
        expected_row_book = {
            "covered_owner_ids_before": covered,
            "seen_owner_ids_before": seen_before,
            "matched_owner_id": owner,
            "strict_physical_owner_match": strict,
            "covered_repeat": covered_repeat,
            "duplicate": duplicate,
            "new_target_owner": bool(strict and owner not in covered_set),
        }
        for key, expected in expected_row_book.items():
            if key not in row_book or row_book[key] != expected:
                raise EvidenceContractError(f"{label}.rows[{index}] bookkeeping differs for {key}")
        complete = status in {"closure", "accepted", "complete"} or row.get("complete") is True
        stop = bool(row.get("stop") is True or status == "native_stop" or row.get("stop_reason") == "native_stop")
        parsed_rows.append({"owner": owner, "strict": strict, "status": status, "match_status": match_status, "covered_repeat": covered_repeat, "duplicate": duplicate, "complete": complete, "stop": stop, "row_started": row_started, "token_ids": row_tokens})
        if strict and owner is not None:
            strict_sequence.append(owner)
            seen.add(owner)
    trajectory_tokens = [token for row in parsed_rows for token in row["token_ids"]]
    lookahead_token: int | None = None
    if generated_tokens != trajectory_tokens:
        suffix = generated_tokens[len(trajectory_tokens) :]
        if generated_tokens[: len(trajectory_tokens)] != trajectory_tokens or len(suffix) != 1:
            raise EvidenceContractError(f"{label}.generated_token_ids differs from row token trajectory")
        last_row = rows[-1]
        last_status = last_row.get("status")
        if last_status == "closure" and terminal == "native_stop" and suffix[0] in native_stop_ids:
            lookahead_token = suffix[0]
        elif last_status == "over_continuation" and terminal == "over_continuation":
            over_continuation = last_row.get("over_continuation")
            if not isinstance(over_continuation, Mapping) or over_continuation.get("selected_token_id") != suffix[0]:
                raise EvidenceContractError(f"{label} over-continuation token disagrees with trajectory")
            # The producer routes a between-row STOP and a natural opener before
            # it can call the lookahead an over-continuation, so neither token
            # can ever be the surplus one here.
            if suffix[0] == opener or suffix[0] in native_stop_ids:
                raise EvidenceContractError(f"{label} over-continuation lookahead token is foreign to the natural boundary")
            lookahead_token = suffix[0]
        else:
            raise EvidenceContractError(f"{label}.generated_token_ids differs from row token trajectory")
    row_native_stop = any(row["status"] == "native_stop" for row in parsed_rows)
    if row_native_stop and terminal != "native_stop":
        raise EvidenceContractError(f"{label} native STOP row disagrees with terminal_reason")
    if terminal == "native_stop" and not row_native_stop and lookahead_token not in native_stop_ids:
        raise EvidenceContractError(f"{label} native STOP terminal lacks an exact terminal token")
    if terminal == "over_continuation" and (lookahead_token is None or rows[-1].get("status") != "over_continuation"):
        raise EvidenceContractError(f"{label} over-continuation terminal lacks its exact lookahead token")
    nonclosure = [index for index, row in enumerate(parsed_rows) if row["status"] != "closure"]
    if nonclosure and nonclosure != [len(parsed_rows) - 1]:
        raise EvidenceContractError(f"{label}.rows continue after a terminal scientific outcome")
    parse = {
        "valid_rows": sum(item["complete"] for item in parsed_rows),
        "duplicate_rows": sum(item["duplicate"] for item in parsed_rows),
        "unmatched_rows": sum(item["complete"] and not item["strict"] for item in parsed_rows),
        "ambiguous_rows": sum(item["complete"] and item["match_status"] == "ambiguous" for item in parsed_rows),
        "malformed_rows": sum(item["status"] in {"over_continuation", "malformed"} for item in parsed_rows),
        "invalid_rows": sum(item["status"] in {"invalid", "over_continuation", "max_budget"} for item in parsed_rows),
    }
    book = raw.get("owner_bookkeeping")
    if not isinstance(book, Mapping) or not isinstance(book.get("parse"), Mapping) or not isinstance(book.get("stop"), Mapping):
        raise EvidenceContractError(f"{label}.owner_bookkeeping aggregate is incomplete")
    declared_parse = dict(book["parse"])
    if set(declared_parse) != set(parse) or declared_parse != parse:
        raise EvidenceContractError(f"{label}.owner_bookkeeping.parse differs from rows")
    row_stop_reason = rows[-1].get("stop_reason", rows[-1].get("status"))
    lookahead_terminal = rows[-1].get("status") == "closure" and terminal in {
        "native_stop",
        "over_continuation",
        "max_budget",
    }
    if (
        (row_stop_reason != terminal and not lookahead_terminal)
        or book.get("horizon_status") != terminal
        or book["stop"].get("stop_reason") != terminal
        or book["stop"].get("stopped") != (terminal != "closure")
    ):
        raise EvidenceContractError(f"{label} terminal/STOP receipt differs from rows")
    row_entry = book.get("row_entry")
    if not isinstance(row_entry, Mapping):
        raise EvidenceContractError(f"{label}.owner_bookkeeping.row_entry is missing")
    expected_row_entry = {
        "admission_mode": "pre_opener_natural",
        "first_generated_token_id": first,
        "opener_generated_by_model": opener_generated,
        "opener_injected": False,
        "row_started": opener_generated,
    }
    for key, expected in expected_row_entry.items():
        if row_entry.get(key) != expected:
            raise EvidenceContractError(f"{label}.owner_bookkeeping.row_entry differs for {key}")
    for index, row in enumerate(rows):
        row_book = row.get("owner_bookkeeping")
        if row_book.get("parse") != parse or row_book.get("stop") != dict(book["stop"]) or row_book.get("horizon_status") != terminal:
            raise EvidenceContractError(f"{label}.rows[{index}] aggregate bookkeeping differs")
        if row_book.get("row_entry") != row_entry:
            raise EvidenceContractError(f"{label}.rows[{index}] row_entry bookkeeping differs")
    strict_set = list(dict.fromkeys(strict_sequence))
    expected_lists = {
        "raw_endpoint_owner_ids": sorted(strict_set),
        "covered_repeat_owner_ids": sorted(set(strict_set) & covered_set),
        "new_target_owner_ids": sorted(set(strict_set) - covered_set),
    }
    for key, expected in expected_lists.items():
        if key in book and _owner_id_list(book[key], f"{label}.owner_bookkeeping.{key}") != expected:
            raise EvidenceContractError(f"{label}.{key} differs from rows")
    if "strict_physical_owner_match_count" in book and book["strict_physical_owner_match_count"] != len(strict_sequence):
        raise EvidenceContractError(f"{label} strict owner count differs from rows")
    target = raw.get("target_owner_id", event.get("target_owner_id", event.get("event_id")))
    target = _text(target, f"{label}.target_owner_id")
    owner_utility = _extract_utilities(raw, covered_set, parsed_rows, label)
    return {
        "raw": dict(raw),
        "prefix_sha256": prefix_sha,
        "full_prefix_sha256": full_prefix_sha,
        "mrope_hash": mrope,
        "mrope_hashes": list(scalar["mrope_hashes"]),
        "opener_token_id": raw["opener_token_id"],
        "initial_prefix_last_token_id": initial_last,
        "first_generated_token_id": first,
        "opener_generated_by_model": opener_generated,
        "terminal_reason": terminal,
        "rows": parsed_rows,
        "strict_owner_sequence": strict_sequence,
        "strict_owner_ids": sorted(set(strict_sequence)),
        "covered_owner_ids": covered,
        "target_owner_id": target,
        "target_release": target in set(strict_sequence),
        "covered_repeat_owner_ids": sorted(set(strict_sequence) & covered_set),
        "uncovered_owner_gain_ids": sorted(set(strict_sequence) - covered_set),
        "parse": parse,
        "owner_utility": owner_utility,
        "scalar": scalar,
        "native_stop": terminal == "native_stop" or any(item["stop"] for item in parsed_rows),
        "row_admission": sum(int(row["row_started"]) for row in parsed_rows),
        "mechanically_valid": True,
    }


def _extract_utilities(raw: Mapping[str, Any], covered: set[str], rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any] | None]:
    book = raw.get("owner_bookkeeping")
    if not isinstance(book, Mapping):
        return {"horizon_1": None, "horizon_3": None}
    horizons = book.get("horizons") if isinstance(book.get("horizons"), Mapping) else {}
    result: dict[str, dict[str, Any] | None] = {}
    for horizon, count in (("horizon_1", 1), ("horizon_3", 3)):
        value = horizons.get(horizon, horizons.get(horizon.removeprefix("horizon_")))
        if value is None and horizon in book and isinstance(book[horizon], Mapping):
            value = book[horizon]
        if value is None and horizon == "horizon_3" and all(key in book for key in ("G", "K", "L")):
            value = {key: book[key] for key in ("G", "K", "L")}
        if value is None:
            result[horizon] = None
            continue
        if not isinstance(value, Mapping):
            raise EvidenceContractError(f"{label}.{horizon} utility is malformed")
        try:
            gained = _owner_id_list(value.get("G"), f"{label}.{horizon}.G")
            kept = _owner_id_list(value.get("K"), f"{label}.{horizon}.K")
            lost = _owner_id_list(value.get("L"), f"{label}.{horizon}.L")
        except EvidenceContractError:
            raise
        if set(gained) & (set(kept) | set(lost)) or set(kept) & set(lost):
            raise EvidenceContractError(f"{label}.{horizon} G/K/L sets overlap")
        net = value.get("net", len(gained) - len(lost))
        if _finite(net, f"{label}.{horizon}.net") != len(gained) - len(lost):
            raise EvidenceContractError(f"{label}.{horizon}.net differs from G/L")
        result[horizon] = {"G": gained, "K": kept, "L": lost, "net": len(gained) - len(lost), "net_charged": len(gained) - len(lost) - int(raw.get("owner_bookkeeping", {}).get("parse", {}).get("unmatched_rows", 0))}
    return result


def _derive_owner_utilities(parsed_cells: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Derive descriptive owner-set transitions against C00 from raw rows.

    The live gate intentionally emits no horizon G/K/L maps.  This derivation
    uses the same first N row positions for every cell, retains unmatched rows
    as a charge, and is descriptive even when source-specific tau is later
    unqualified.
    """

    base = parsed_cells["C00"]
    result: dict[str, dict[str, Any]] = {}
    for cell, parsed in parsed_cells.items():
        horizons: dict[str, Any] = {}
        for horizon, limit in (("horizon_1", 1), ("horizon_3", 3)):
            base_rows = list(base.get("rows", []))[:limit]
            treatment_rows = list(parsed.get("rows", []))[:limit]
            base_ids = sorted({row["owner"] for row in base_rows if row.get("strict") and row.get("owner") is not None})
            treatment_ids = sorted({row["owner"] for row in treatment_rows if row.get("strict") and row.get("owner") is not None})
            base_set = set(base_ids)
            treatment_set = set(treatment_ids)
            gained = sorted(treatment_set - base_set)
            kept = sorted(treatment_set & base_set)
            lost = sorted(base_set - treatment_set)
            unmatched = sum(bool(row.get("complete") and not row.get("strict")) for row in treatment_rows)
            net = len(gained) - len(lost)
            horizons[horizon] = {
                "row_limit": limit,
                "reference_cell": "C00",
                "estimand": "treatment-minus-C00 owner-set delta over first N rows",
                "base_owner_ids": base_ids,
                "treatment_owner_ids": treatment_ids,
                "G": gained,
                "K": kept,
                "L": lost,
                "net": net,
                "treatment_unmatched_rows": unmatched,
                "net_charged": net - unmatched,
            }
        result[cell] = horizons
    return result


def _validate_c11_receipts(cell: Mapping[str, Any], parsed: Mapping[str, Any], label: str) -> None:
    if cell.get("composition_arm_id") != "C11":
        raise EvidenceContractError("C11 composition_arm_id is missing or foreign")
    raw = parsed["raw"]
    mappings: list[Mapping[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            mappings.append(value)
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(raw)
    composition = raw.get("composition_receipt") or raw.get("composition")
    if not isinstance(composition, Mapping):
        composition = next((item for item in mappings if item.get("arm_id") == "C11" and (item.get("cell_id") == "C11" or item.get("composition_arm_id") == "C11")), None)
    transport = raw.get("transport_receipt") or raw.get("transport")
    if not isinstance(transport, Mapping):
        transport = next((item for item in mappings if item.get("transport_arm_id") == "K10" or item.get("inner_actuator_id") == "C11"), None)
    if not isinstance(composition, Mapping) or not isinstance(transport, Mapping):
        raise EvidenceContractError("C11 component/transport receipts are missing")
    component_ids = composition.get("component_arm_ids", composition.get("component_order"))
    if component_ids not in (["K10", "H20"], {"K10", "H20"}):
        children = composition.get("children")
        component_ids = [item.get("arm_id") for item in children] if isinstance(children, list) and all(isinstance(item, Mapping) for item in children) else None
    if component_ids != ["K10", "H20"] and component_ids != {"K10", "H20"}:
        raise EvidenceContractError("C11 component receipt does not bind K10 and H20")
    if composition.get("status") not in {"ready", "passed", "validated", "complete"}:
        raise EvidenceContractError("C11 composition receipt is invalid")
    if composition.get("arm_id") not in {None, "C11"} or composition.get("cell_id") not in {None, "C11"}:
        raise EvidenceContractError("C11 composition receipt identity is invalid")
    if composition.get("component_order") is not None and composition.get("component_order") != ["K10", "H20"]:
        raise EvidenceContractError("C11 component order drifted")
    if composition.get("component_changed_cell_union") is not None and composition.get("component_changed_cell_union") is not True:
        raise EvidenceContractError("C11 component union proof failed")
    if composition.get("exact_scope") is not None and composition.get("exact_scope") is not True:
        raise EvidenceContractError("C11 component scope proof failed")
    children = composition.get("children")
    if children is not None:
        if not isinstance(children, list) or len(children) != 2 or [item.get("arm_id") for item in children if isinstance(item, Mapping)] != ["K10", "H20"]:
            raise EvidenceContractError("C11 child component receipts are incomplete")
        for child in children:
            if not isinstance(child, Mapping):
                raise EvidenceContractError("C11 child component receipt is malformed")
            if child.get("receipt_sha256") is not None and isinstance(child.get("receipt"), Mapping) and child["receipt_sha256"] != sha256_json(child["receipt"]):
                raise EvidenceContractError("C11 child component receipt hash differs")
    if transport.get("arm_id", transport.get("transport_arm_id")) not in {"K10", "C11"} or transport.get("status") not in {"ready", "passed", "validated", "complete"}:
        raise EvidenceContractError("C11 transport receipt is invalid")
    if transport.get("use_cache") is True or transport.get("opener_injected") is True:
        raise EvidenceContractError("C11 transport receipt crosses natural/cache boundary")
    for key in ("prefix_sha256", "mrope_hash"):
        if key in transport and transport[key] != parsed[key]:
            raise EvidenceContractError(f"C11 transport {key} differs from raw endpoint")


def _sealed_code_hashes(pre_gpu: Mapping[str, Any]) -> dict[str, str]:
    source_files = pre_gpu.get("document", {}).get("source_files") if isinstance(pre_gpu.get("document"), Mapping) else None
    if not isinstance(source_files, Mapping) or not source_files:
        raise EvidenceContractError("pre-GPU source_files are missing for runtime code identity")
    result: dict[str, str] = {}
    for role, value in source_files.items():
        if not isinstance(value, Mapping):
            raise EvidenceContractError(f"pre-GPU source_files.{role} is malformed")
        digest = value.get("sha256", value.get("raw_sha256"))
        result[str(role)] = _sha(digest, f"pre-GPU source_files.{role}.sha256")
    return result


def _sealed_input_hashes(pre_gpu: Mapping[str, Any]) -> dict[str, str]:
    document = pre_gpu.get("document")
    values = document.get("input_hashes") if isinstance(document, Mapping) else None
    if not isinstance(values, Mapping) or not values:
        raise EvidenceContractError("pre-GPU input_hashes are missing for runtime input identity")
    return {str(key): _sha(value, f"pre-GPU input_hashes.{key}") for key, value in values.items()}


def _validate_runtime_hash_maps(
    document: Mapping[str, Any],
    pre_gpu: Mapping[str, Any],
    label: str,
    *,
    successor: Mapping[str, Any] | None = None,
) -> None:
    sealed_code = _sealed_code_hashes(pre_gpu)
    code = document.get("code_hashes")
    if not isinstance(code, Mapping) or not code:
        raise EvidenceContractError(f"{label}.code_hashes are missing")
    for role, digest in sealed_code.items():
        if code.get(role) != digest:
            raise EvidenceContractError(f"{label}.code_hashes.{role} differs from sealed source")
    for role, digest in code.items():
        _sha(digest, f"{label}.code_hashes.{role}")
        if role not in sealed_code and role not in _RUNTIME_CODE_ALIASES:
            raise EvidenceContractError(f"{label}.code_hashes contains an unexpected runner-local role {role}")
        if role in _RUNTIME_CODE_ALIASES and role not in sealed_code:
            source_role = _RUNTIME_CODE_ALIASES[role]
            if source_role not in sealed_code:
                raise EvidenceContractError(f"{label}.code_hashes.{role} is not explicitly sealed as a runner-local alias")
            if digest != sealed_code[source_role]:
                raise EvidenceContractError(f"{label}.code_hashes.{role} differs from sealed {source_role}")

    sealed_inputs = _sealed_input_hashes(pre_gpu)
    inputs = document.get("input_hashes")
    if not isinstance(inputs, Mapping) or not inputs:
        raise EvidenceContractError(f"{label}.input_hashes are missing")
    for key, digest in sealed_inputs.items():
        if inputs.get(key) != digest:
            raise EvidenceContractError(f"{label}.input_hashes.{key} differs from sealed input")
    for key, digest in inputs.items():
        _sha(digest, f"{label}.input_hashes.{key}")
        if key in sealed_inputs or key in _RUNTIME_INPUT_EXTRAS:
            continue
        if successor is not None and key in _SUCCESSOR_RUNTIME_INPUT_EXTRAS:
            continue
        raise EvidenceContractError(f"{label}.input_hashes contains an unexpected runner-local key {key}")
    if successor is not None:
        # The successor admits the runner-recorded parent hashes only when all
        # three pin the recomputed parent receipt exactly.
        expected_pins = {
            "pre_gpu_receipt": successor["parent_raw_sha256"],
            "pre_gpu_receipt_sha256": successor["parent_raw_sha256"],
            "pre_gpu_receipt_self_sha256": successor["parent_self_sha256"],
        }
        for key, expected in expected_pins.items():
            if inputs.get(key) != expected:
                raise EvidenceContractError(
                    f"{label}.input_hashes.{key} does not pin the recomputed parent receipt"
                )


def _expected_device_assignment(expected_shard: str, expected_event: Mapping[str, Any]) -> dict[str, Any]:
    try:
        shard_index = int(expected_shard.removeprefix("shard-"))
    except ValueError as exc:  # pragma: no cover - caller controls shard syntax
        raise EvidenceContractError(f"invalid shard id {expected_shard}") from exc
    physical = DEVICE_PLAN.get(expected_shard)
    if physical is None:
        raise EvidenceContractError(f"unknown shard device {expected_shard}")
    body = {
        "shard_id": expected_shard,
        "shard_index": shard_index,
        "physical_device": physical,
        "observed_cuda_visible_devices": physical,
        "logical_device": "cuda:0",
        "device_count": 1,
        "event_index": expected_event["event_index"],
        "event_id": expected_event["event_id"],
        "event_sha256": expected_event["event_sha256"],
    }
    body["authorization_sha256"] = sha256_json({key: value for key, value in body.items() if key != "authorization_sha256"})
    return body


def _validate_runtime(
    document: Mapping[str, Any],
    info: Mapping[str, Any],
    expected_shard: str,
    expected_event: Mapping[str, Any],
    pre_gpu: Mapping[str, Any],
    *,
    successor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if document.get("schema_version") != RUNTIME_SCHEMA_VERSION or document.get("status") not in {"completed", "attested"} or document.get("unit_id") != UNIT_ID:
        raise EvidenceContractError("runtime identity schema/status/unit drifted")
    _self_check(document, "runtime identity")
    allowed_runtime = {
        "schema_version", "status", "unit_id", "shard_id", "event_id", "event_index", "image_id", "checkpoint", "step",
        "device", "code_hashes", "input_hashes", "plan_sha256", "plan_self_sha256", "pre_gpu_receipt_self_sha256", "self_sha256",
        "runtime", "model", "backend", "cuda", "model_identity", "device_assignment", "device_policy", "event_binding", "input_paths", "forced_math", "checkpoint", "step", "substrate", "claim_scope", "no_training",
    }
    if not set(document).issubset(allowed_runtime):
        raise EvidenceContractError("runtime identity contains an unknown field")
    if document.get("shard_id") != expected_shard or document.get("event_id") != expected_event["event_id"] or document.get("event_index") != expected_event["event_index"] or document.get("image_id") != expected_event["image_id"]:
        raise EvidenceContractError("runtime identity shard/event differs")
    _validate_runtime_hash_maps(document, pre_gpu, "runtime identity", successor=successor)
    device = document.get("device")
    if not isinstance(device, Mapping):
        raise EvidenceContractError("runtime identity device is missing")
    if device.get("logical_device") != "cuda:0" or str(device.get("physical_device")) != DEVICE_PLAN[expected_shard]:
        raise EvidenceContractError("runtime identity device assignment differs")
    assignment = document.get("device_assignment")
    expected_assignment = _expected_device_assignment(expected_shard, expected_event)
    if not isinstance(assignment, Mapping) or dict(assignment) != expected_assignment:
        raise EvidenceContractError("runtime identity nested device_assignment differs from sealed shard/event authorization")
    if document.get("event_binding") is not None:
        binding = document["event_binding"]
        if not isinstance(binding, Mapping) or any(binding.get(key) != expected_event.get(key) for key in ("event_index", "event_id", "image_id", "event_sha256")):
            raise EvidenceContractError("runtime identity event_binding differs from plan event")
    if document.get("device_policy") is not None:
        policy = document["device_policy"]
        if not isinstance(policy, Mapping) or policy.get("device_plan") != DEVICE_PLAN or policy.get("logical_device") not in {None, "cuda:0"}:
            raise EvidenceContractError("runtime identity device_policy differs from sealed plan")
    if document.get("input_paths") is not None:
        sealed_bindings = pre_gpu.get("document", {}).get("input_bindings") if isinstance(pre_gpu.get("document"), Mapping) else None
        if not isinstance(document["input_paths"], Mapping) or not isinstance(sealed_bindings, Mapping) or any(document["input_paths"].get(key) != value.get("path") for key, value in sealed_bindings.items() if isinstance(value, Mapping)):
            raise EvidenceContractError("runtime identity input_paths differ from sealed inputs")
    if document.get("pre_gpu_receipt_self_sha256") is not None and document["pre_gpu_receipt_self_sha256"] != pre_gpu["self_sha256"]:
        raise EvidenceContractError("runtime identity pre-GPU binding differs")
    return {"document": dict(document), "path": info.get("path"), "raw_sha256": info["raw_sha256"], "size_bytes": info.get("size_bytes"), "self_sha256": document["self_sha256"]}


def _validate_terminal(document: Mapping[str, Any], info: Mapping[str, Any], result: Mapping[str, Any], expected_shard: str, event: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != TERMINAL_SCHEMA_VERSION or document.get("status") != "completed" or document.get("unit_id") != UNIT_ID:
        raise EvidenceContractError("terminal summary schema/status/unit drifted")
    _self_check(document, "terminal summary")
    if set(document) != {"schema_version", "status", "unit_id", "shard_id", "event_id", "event_index", "image_id", "result_sha256", "cells", "self_sha256"}:
        raise EvidenceContractError("terminal summary fields are incomplete or unknown")
    if document.get("shard_id") != expected_shard or document.get("event_id") != event["event_id"] or document.get("event_index") != event["event_index"] or document.get("image_id") != event["image_id"] or document.get("result_sha256") != result["result_sha256"]:
        raise EvidenceContractError("terminal summary event/result binding differs")
    if document.get("admission_mode") not in {None, "pre_opener_natural"} or document.get("opener_injected") not in {None, False}:
        raise EvidenceContractError("terminal summary admission is not natural")
    terminal_cells = document.get("cells", document.get("cell_terminal", document.get("terminals")))
    if not isinstance(terminal_cells, Mapping) or set(terminal_cells) != set(CELL_ORDER):
        raise EvidenceContractError("terminal summary cell terminal bindings are incomplete")
    result_cells = result.get("cells")
    for cell in CELL_ORDER:
        binding = terminal_cells[cell]
        source = result_cells.get(cell) if isinstance(result_cells, Mapping) else None
        raw = source.get("result") if isinstance(source, Mapping) else None
        if not isinstance(binding, Mapping) or binding.get("cell_id") != cell or not isinstance(raw, Mapping):
            raise EvidenceContractError(f"terminal summary {cell} binding is malformed")
        if binding.get("transport_arm_id") != source.get("transport_arm_id"):
            raise EvidenceContractError(f"terminal summary {cell} transport identity differs")
        if binding.get("composition_arm_id") != ("C11" if cell == "C11" else None):
            raise EvidenceContractError(f"terminal summary {cell} composition identity differs")
        if binding.get("admission_mode") != raw.get("admission_mode") or binding.get("opener_injected") != raw.get("opener_injected"):
            raise EvidenceContractError(f"terminal summary {cell} admission differs")
        terminal_reason = raw.get("terminal_reason", raw.get("stop_reason"))
        raw_stop_reason = raw.get("stop_reason")
        if binding.get("terminal_reason") != terminal_reason or binding.get("stop_reason") != raw_stop_reason:
            raise EvidenceContractError(f"terminal summary {cell} terminal differs")
        if binding.get("raw_result_sha256") != sha256_json(raw):
            raise EvidenceContractError(f"terminal summary {cell} raw result hash differs")
    return {"document": dict(document), "path": info.get("path"), "raw_sha256": info["raw_sha256"], "size_bytes": info.get("size_bytes"), "self_sha256": document["self_sha256"]}


def _validate_aggregate_receipt(document: Mapping[str, Any], info: Mapping[str, Any], *, result: Mapping[str, Any], result_info: Mapping[str, Any], runtime: Mapping[str, Any], runtime_info: Mapping[str, Any], terminal: Mapping[str, Any], terminal_info: Mapping[str, Any], plan: Mapping[str, Any], pre_gpu: Mapping[str, Any], shard: str, event: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != SHARD_RECEIPT_SCHEMA_VERSION or document.get("status") != "completed" or document.get("unit_id") != UNIT_ID:
        raise EvidenceContractError("aggregate receipt schema/status/unit drifted")
    _self_check(document, "aggregate receipt")
    if set(document) != {"schema_version", "status", "unit_id", "shard_id", "event_id", "event_index", "image_id", "plan_sha256", "plan_self_sha256", "pre_gpu_receipt_self_sha256", "result_sha256", "runtime_identity_sha256", "terminal_summary_self_sha256", "result_raw_sha256", "runtime_identity_raw_sha256", "terminal_summary_raw_sha256", "self_sha256"}:
        raise EvidenceContractError("aggregate receipt fields are incomplete or unknown")
    if document.get("shard_id") != shard or document.get("event_id") != event["event_id"] or document.get("event_index") != event["event_index"] or document.get("image_id") != event["image_id"]:
        raise EvidenceContractError("aggregate receipt shard/event binding differs")
    expected = {
        "result_raw_sha256": result_info["raw_sha256"],
        "result_sha256": result["result_sha256"],
        "runtime_identity_raw_sha256": runtime_info["raw_sha256"],
        "runtime_identity_sha256": runtime["self_sha256"],
        "terminal_summary_raw_sha256": terminal_info["raw_sha256"],
        "terminal_summary_self_sha256": terminal["self_sha256"],
        "plan_sha256": plan["self_sha256"],
        "plan_self_sha256": plan["self_sha256"],
        "pre_gpu_receipt_self_sha256": pre_gpu["self_sha256"],
    }
    for key, value in expected.items():
        if key in document and document[key] != value:
            raise EvidenceContractError(f"aggregate receipt {key} differs")
        if key not in document:
            raise EvidenceContractError(f"aggregate receipt {key} is missing")
    return {"document": dict(document), "path": info.get("path"), "raw_sha256": info["raw_sha256"], "size_bytes": info.get("size_bytes"), "self_sha256": document["self_sha256"]}


def _load_shard(
    root_value: str | Path,
    index: int,
    plan: Mapping[str, Any],
    pre_gpu: Mapping[str, Any],
    *,
    successor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _regular_dir(root_value, f"shard[{index}]")
    expected_names = {"result.json", "runtime_identity.json", "terminal_summary.json", "aggregate.receipt.json"}
    names = {child.name for child in root.iterdir()}
    if names != expected_names:
        raise EvidenceContractError(f"shard[{index}] is partial, duplicated, or contains a foreign file")
    paths = {name: _regular_file(root / name, f"shard[{index}]/{name}") for name in expected_names}
    result, result_info = _read_json(paths["result.json"], f"shard[{index}].result")
    runtime, runtime_info = _read_json(paths["runtime_identity.json"], f"shard[{index}].runtime_identity")
    terminal, terminal_info = _read_json(paths["terminal_summary.json"], f"shard[{index}].terminal_summary")
    aggregate, aggregate_info = _read_json(paths["aggregate.receipt.json"], f"shard[{index}].aggregate.receipt")
    event = plan["events"][index]
    shard_id = f"shard-{index:03d}"
    required_result = {"schema_version", "status", "unit_id", "shard_id", "event_id", "image_id", "event_index", "plan_sha256", "plan_self_sha256", "pre_gpu_receipt_self_sha256", "source_event_sha256", "runtime_identity_sha256", "technical_control", "cells", "result_sha256"}
    if set(result) != required_result or result.get("schema_version") != EVENT_SCHEMA_VERSION or result.get("status") != "completed" or result.get("unit_id") != UNIT_ID:
        raise EvidenceContractError(f"shard[{index}] result schema/keys drifted")
    if result.get("shard_id") != shard_id or result.get("event_id") != event["event_id"] or result.get("image_id") != event["image_id"] or result.get("event_index") != event["event_index"] or result.get("plan_sha256") != plan["self_sha256"] or result.get("plan_self_sha256") != plan["self_sha256"] or result.get("pre_gpu_receipt_self_sha256") != pre_gpu["self_sha256"] or result.get("source_event_sha256") != event["event_sha256"]:
        raise EvidenceContractError(f"shard[{index}] result source/event binding differs")
    _self_check(result, f"shard[{index}].result", field="result_sha256")
    runtime_info_checked = _validate_runtime(runtime, runtime_info, shard_id, event, pre_gpu, successor=successor)
    if result.get("runtime_identity_sha256") != runtime_info_checked["self_sha256"]:
        raise EvidenceContractError(f"shard[{index}] result/runtime identity hash differs")
    terminal_checked = _validate_terminal(terminal, terminal_info, result, shard_id, event)
    aggregate_checked = _validate_aggregate_receipt(aggregate, aggregate_info, result=result, result_info=result_info, runtime=runtime, runtime_info=runtime_info, terminal=terminal, terminal_info=terminal_info, plan=plan, pre_gpu=pre_gpu, shard=shard_id, event=event)
    technical = result.get("technical_control")
    if not isinstance(technical, Mapping) or set(technical) != {TECHNICAL_CELL}:
        raise EvidenceContractError(f"shard[{index}] technical K00 control is missing")
    control = technical[TECHNICAL_CELL]
    if not isinstance(control, Mapping) or control.get("transport_arm_id") != TECHNICAL_CELL or not isinstance(control.get("result"), Mapping):
        raise EvidenceContractError(f"shard[{index}] K00 technical control is malformed")
    control_raw = dict(control["result"])
    _validate_natural_raw(control_raw, TECHNICAL_CELL, f"shard[{index}].K00", event=event)
    cells = result.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != set(CELL_ORDER):
        raise EvidenceContractError(f"shard[{index}] cell set/order differs")
    parsed_cells: dict[str, dict[str, Any]] = {}
    for cell in CELL_ORDER:
        value = cells[cell]
        required = {"cell_id", "transport_arm_id", "result"} | ({"composition_arm_id"} if cell == "C11" else set())
        if not isinstance(value, Mapping) or set(value) != required or value.get("cell_id") != cell or value.get("transport_arm_id") != TRANSPORT_ARMS[cell] or not isinstance(value.get("result"), Mapping):
            raise EvidenceContractError(f"shard[{index}] {cell} cell contract drifted")
        parsed = _validate_natural_raw(dict(value["result"]), TRANSPORT_ARMS[cell], f"shard[{index}].{cell}", event=event)
        if cell == "C11":
            _validate_c11_receipts(value, parsed, f"shard[{index}].C11")
        parsed_cells[cell] = parsed
    parity = _parity(parsed_cells["C00"]["raw"], f"shard[{index}].C00")
    runtime_identity_signature = {
        key: value
        for key, value in runtime.items()
        if key not in _PER_SHARD_RUNTIME_FIELDS
    }
    return {
        "root": str(root),
        "root_sha256": sha256_json(sorted((name, info["raw_sha256"], info["size_bytes"]) for name, info in (("result", result_info), ("runtime_identity", runtime_info), ("terminal_summary", terminal_info), ("aggregate_receipt", aggregate_info)))),
        "artifacts": {
            "result": {"path": str(paths["result.json"]), "sha256": result_info["raw_sha256"], "size": result_info["size_bytes"], "self_sha256": result["result_sha256"]},
            "runtime_identity": {"path": str(paths["runtime_identity.json"]), "sha256": runtime_info["raw_sha256"], "size": runtime_info["size_bytes"], "self_sha256": runtime["self_sha256"]},
            "terminal_summary": {"path": str(paths["terminal_summary.json"]), "sha256": terminal_info["raw_sha256"], "size": terminal_info["size_bytes"], "self_sha256": terminal["self_sha256"]},
            "aggregate_receipt": {"path": str(paths["aggregate.receipt.json"]), "sha256": aggregate_info["raw_sha256"], "size": aggregate_info["size_bytes"], "self_sha256": aggregate["self_sha256"]},
        },
        "event": dict(event),
        "result": result,
        "result_info": result_info,
        "runtime": runtime,
        "runtime_info": runtime_info,
        "runtime_checked": runtime_info_checked,
        "terminal": terminal,
        "terminal_checked": terminal_checked,
        "aggregate": aggregate,
        "aggregate_checked": aggregate_checked,
        "cells": parsed_cells,
        "control": control_raw,
        "parity": parity,
        "runtime_identity_signature": runtime_identity_signature,
    }


def _event_metric(endpoint: Mapping[str, Any]) -> dict[str, int]:
    parse = endpoint["parse"]
    return {
        "complete_rows": int(parse["valid_rows"]),
        "strict_count": len(endpoint["strict_owner_sequence"]),
        "target_release": int(endpoint["target_release"]),
        "unmatched": int(parse["unmatched_rows"]),
        "duplicates": int(parse["duplicate_rows"]),
        "ambiguous": int(parse["ambiguous_rows"]),
        "malformed": int(parse["malformed_rows"]),
        "invalid": int(parse["invalid_rows"]),
        "STOP": int(endpoint["native_stop"]),
        "row_admission": int(endpoint["row_admission"]),
    }


def _endpoint_document(cell: str, event: Mapping[str, Any], parsed: Mapping[str, Any]) -> dict[str, Any]:
    raw = parsed["raw"]
    metrics = _event_metric(parsed)
    outcome = "matched" if parsed["mechanically_valid"] and metrics["strict_count"] and not any(metrics[key] for key in ("unmatched", "duplicates", "ambiguous", "malformed", "invalid", "STOP")) else "scientific_outcome"
    if metrics["unmatched"]:
        outcome = "unmatched"
    elif metrics["duplicates"]:
        outcome = "duplicate"
    elif metrics["ambiguous"]:
        outcome = "ambiguous"
    elif metrics["malformed"]:
        outcome = "malformed"
    elif metrics["invalid"]:
        outcome = "invalid_token_grammar"
    elif metrics["STOP"]:
        outcome = "native_STOP"
    return {
        "cell_id": cell,
        "event_id": event["event_id"],
        "image_id": event["image_id"],
        "event_index": event["event_index"],
        "mechanically_valid": True,
        "scientific_status": outcome,
        "first_token": {
            "token_id": parsed["first_generated_token_id"],
            "opener_token_id": parsed["opener_token_id"],
            "admission_mode": "pre_opener_natural",
            "opener_generated_by_model": parsed["opener_generated_by_model"],
        },
        "terminal_reason": parsed["terminal_reason"],
        "metrics": metrics,
        "complete_rows": metrics["complete_rows"],
        "strict_owner_sequence": list(parsed["strict_owner_sequence"]),
        "strict_owner_ids": list(parsed["strict_owner_ids"]),
        "strict_count": metrics["strict_count"],
        "target_owner_id": parsed["target_owner_id"],
        "target_release": parsed["target_release"],
        "covered_owner_ids": list(parsed["covered_owner_ids"]),
        "covered_repeat_owner_ids": list(parsed["covered_repeat_owner_ids"]),
        "uncovered_owner_gain_ids": list(parsed["uncovered_owner_gain_ids"]),
        "parse": dict(parsed["parse"]),
        "owner_utility": parsed["owner_utility"],
        "prefix_sha256": parsed["prefix_sha256"],
        "mrope_hash": parsed["mrope_hash"],
        "raw_result_sha256": sha256_json(raw),
    }


def _numeric_contrasts(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    contrasts: dict[str, Any] = {}
    taus: dict[str, Any] = {}
    for name, (left, right, semantic) in CONTRASTS.items():
        per_event: list[dict[str, Any]] = []
        for event in events:
            cells = event["cells"]
            lv = cells[left]["metrics"]
            rv = cells[right]["metrics"]
            # Contrast names are deliberately ``left-minus-right``.  Keep
            # their arithmetic aligned with the published key rather than
            # silently reversing the scientific sign.
            deltas = {metric: lv[metric] - rv[metric] for metric in METRICS}
            per_event.append({"event_id": event["event_id"], "image_id": event["image_id"], "delta": deltas})
        contrasts[name] = {"semantic": semantic, "left": left, "right": right, "per_event": per_event, "metrics": {metric: [item["delta"][metric] for item in per_event] for metric in METRICS}}
    for metric in METRICS:
        per_event_tau: list[dict[str, Any]] = []
        for event in events:
            if event["source_specific_event_status"] == "qualified":
                values = {cell: event["cells"][cell]["metrics"][metric] for cell in CELL_ORDER}
                value = values["C11"] - values["C10"] - values["C01"] + values["C00"]
                per_event_tau.append({
                    "event_id": event["event_id"],
                    "image_id": event["image_id"],
                    "status": "qualified",
                    "unqualified_reason": None,
                    "value": value,
                })
            else:
                per_event_tau.append({
                    "event_id": event["event_id"],
                    "image_id": event["image_id"],
                    "status": "unqualified",
                    "unqualified_reason": event["source_specific_unqualified_reason"],
                    "value": None,
                })
        qualified_count = sum(item["status"] == "qualified" for item in per_event_tau)
        status = "qualified" if qualified_count == len(per_event_tau) else "unqualified"
        taus[metric] = {
            "status": status,
            "unqualified_reason": (
                None
                if status == "qualified"
                else "aggregate source-specific tau requires qualified endpoints for all three frozen events"
            ),
            "qualified_event_count": qualified_count,
            "event_count": len(per_event_tau),
            "per_event": per_event_tau,
            "values": [item["value"] for item in per_event_tau],
            "mean": (
                sum(item["value"] for item in per_event_tau) / len(per_event_tau)
                if status == "qualified"
                else None
            ),
        }
    return contrasts, taus


def _utilities(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in ("horizon_1", "horizon_3"):
        horizon_data: dict[str, Any] = {
            "matched_neutral": {cell: [] for cell in CELL_ORDER},
            "net_charged": {cell: [] for cell in CELL_ORDER},
            "per_event": [],
            "available": True,
        }
        neutral_taus: list[float | None] = []
        charged_taus: list[float | None] = []
        qualified_count = 0
        unqualified_event_ids: list[str] = []
        for event in events:
            utilities = {cell: event["cells"][cell]["owner_utility"].get(horizon) for cell in CELL_ORDER}
            missing = [cell for cell, utility in utilities.items() if not isinstance(utility, Mapping)]
            source_qualified = event["source_specific_event_status"] == "qualified"
            if not missing:
                # Keep descriptive per-cell values available on unqualified
                # events; only the aggregate source-specific tau is withheld.
                neutral = {cell: float(utilities[cell]["net"]) for cell in CELL_ORDER}
                charged = {cell: float(utilities[cell]["net_charged"]) for cell in CELL_ORDER}
            if source_qualified and not missing:
                neutral_tau = neutral["C11"] - neutral["C10"] - neutral["C01"] + neutral["C00"]
                charged_tau = charged["C11"] - charged["C10"] - charged["C01"] + charged["C00"]
                qualified_count += 1
                status = "qualified"
                reason = None
            else:
                if missing:
                    neutral = {cell: None for cell in CELL_ORDER}
                    charged = {cell: None for cell in CELL_ORDER}
                neutral_tau = None
                charged_tau = None
                status = "unqualified"
                reason = event["source_specific_unqualified_reason"] if not source_qualified else f"{horizon} owner utility is missing for cells: {','.join(missing)}"
                unqualified_event_ids.append(event["event_id"])
                horizon_data["available"] = False
            for cell in CELL_ORDER:
                horizon_data["matched_neutral"][cell].append(neutral[cell])
                horizon_data["net_charged"][cell].append(charged[cell])
            neutral_taus.append(neutral_tau)
            charged_taus.append(charged_tau)
            horizon_data["per_event"].append({
                "event_id": event["event_id"],
                "image_id": event["image_id"],
                "status": status,
                "unqualified_reason": reason,
                "matched_neutral": neutral if status == "qualified" else None,
                "net_charged": charged if status == "qualified" else None,
                "matched_neutral_tau": neutral_tau,
                "net_charged_tau": charged_tau,
            })
        status = "qualified" if qualified_count == len(events) else "unqualified"
        horizon_data["status"] = status
        horizon_data["unqualified_reason"] = (
            None
            if status == "qualified"
            else f"source-specific {horizon} utility requires qualified utility endpoints for all three frozen events; unqualified events: {','.join(unqualified_event_ids)}"
        )
        horizon_data["qualified_event_count"] = qualified_count
        horizon_data["event_count"] = len(events)
        horizon_data["matched_neutral"]["tau_per_event"] = neutral_taus
        horizon_data["net_charged"]["tau_per_event"] = charged_taus
        horizon_data["matched_neutral"]["tau"] = sum(neutral_taus) / len(events) if status == "qualified" else None
        horizon_data["net_charged"]["tau"] = sum(charged_taus) / len(events) if status == "qualified" else None
        result[horizon] = horizon_data
    return result


def _producer() -> dict[str, Any]:
    path = _regular_file(Path(__file__).resolve(), "finalizer producer")
    return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}


def _write_once(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    target = _absolute(path, "evidence output")
    payload = canonical_json_bytes(document) + b"\n"
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(f"immutable evidence collision: {target}")
        return {"path": str(target), "raw_sha256": sha256_bytes(payload), "byte_identical": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return {"path": str(target), "raw_sha256": sha256_bytes(payload), "byte_identical": False}


def finalize(
    plan: str | Path | Mapping[str, Any],
    pre_gpu_receipt: str | Path | Mapping[str, Any],
    shard_roots: Sequence[str | Path],
    *,
    output: str | Path | None = None,
    receipt_output: str | Path | None = None,
    finalization_receipt: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the three immutable shard roots and write evidence/receipt.

    Without ``finalization_receipt`` the parent pre-GPU receipt is the only
    authority, so a post-execution consumer repair is rejected by the parent's
    own binding of this file.  With it, exactly the two authorized slots are
    consumed from the successor and every other binding still live-validates.
    """

    if isinstance(shard_roots, (str, bytes, bytearray)) or not isinstance(shard_roots, Sequence) or len(shard_roots) != 3:
        raise EvidenceContractError("shard_roots must contain exactly three paths in plan order")
    plan_info = validate_plan(plan)
    successor_info: dict[str, Any] | None = None
    if finalization_receipt is not None:
        successor_info = _validate_finalization_receipt(
            finalization_receipt,
            plan_info,
            pre_gpu_path=pre_gpu_receipt if isinstance(pre_gpu_receipt, (str, Path)) else None,
        )
    pre_gpu_info = validate_pre_gpu_receipt(pre_gpu_receipt, plan_info, successor=successor_info)
    pre_gpu_document = pre_gpu_info["document"]
    roots = pre_gpu_document.get("roots")
    if not isinstance(roots, Mapping):
        raise EvidenceContractError("pre-GPU roots are missing")
    execution_binding = roots.get("execution_root")
    final_binding = roots.get("final_root")
    if not isinstance(execution_binding, Mapping) or not isinstance(final_binding, Mapping):
        raise EvidenceContractError("pre-GPU execution/final root bindings are incomplete")
    execution_path = execution_binding.get("path")
    final_path = final_binding.get("path")
    if not isinstance(execution_path, str) or not isinstance(final_path, str):
        raise EvidenceContractError("pre-GPU execution/final root paths are missing")
    execution_root = _absolute(execution_path, "pre-GPU execution root")
    final_root = _absolute(final_path, "pre-GPU final root")
    if execution_root == final_root:
        raise EvidenceContractError("pre-GPU execution and final roots must be distinct")
    if successor_info is not None and successor_info["evidence_root"] != str(final_root):
        raise EvidenceContractError("finalization receipt evidence root differs from the sealed final root")
    for index, root_value in enumerate(shard_roots):
        expected_root = execution_root / f"shard-{index:03d}"
        supplied_root = _absolute(root_value, f"shard[{index}]")
        if supplied_root != expected_root:
            raise EvidenceContractError(f"shard[{index}] must be exactly execution_root/shard-{index:03d}")
    for output_name, output_value in (("evidence output", output), ("receipt output", receipt_output)):
        if output_value is not None and _absolute(output_value, output_name).parent != final_root:
            raise EvidenceContractError(f"{output_name} must be directly under the sealed final root")
    shards = [_load_shard(root, index, plan_info, pre_gpu_info, successor=successor_info) for index, root in enumerate(shard_roots)]
    signatures = [shard["runtime_identity_signature"] for shard in shards]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise EvidenceContractError("runtime/code/input identities differ across shards")
    events: list[dict[str, Any]] = []
    prefix_by_event: dict[str, str] = {}
    full_prefix_by_event: dict[str, str] = {}
    mrope_by_event: dict[str, str] = {}
    for shard in shards:
        event = shard["event"]
        cells: dict[str, Any] = {}
        parsed_cells = {cell: shard["cells"][cell] for cell in CELL_ORDER}
        utilities = _derive_owner_utilities(parsed_cells)
        for cell in CELL_ORDER:
            parsed = shard["cells"][cell]
            if event["event_id"] in prefix_by_event and prefix_by_event[event["event_id"]] != parsed["prefix_sha256"]:
                raise EvidenceContractError(f"{event['event_id']} prefix identity drifted across cells")
            if event["event_id"] in full_prefix_by_event and full_prefix_by_event[event["event_id"]] != parsed["full_prefix_sha256"]:
                raise EvidenceContractError(f"{event['event_id']} full natural prefix drifted across cells")
            if event["event_id"] in mrope_by_event and mrope_by_event[event["event_id"]] != parsed["mrope_hash"]:
                raise EvidenceContractError(f"{event['event_id']} M-RoPE identity drifted across cells")
            if event.get("prefix_sha256") is not None and event["prefix_sha256"] != parsed["prefix_sha256"]:
                raise EvidenceContractError(f"{event['event_id']} history-row prefix identity differs from plan")
            prefix_by_event[event["event_id"]] = parsed["prefix_sha256"]
            full_prefix_by_event[event["event_id"]] = parsed["full_prefix_sha256"]
            mrope_by_event[event["event_id"]] = parsed["mrope_hash"]
            endpoint = _endpoint_document(cell, event, parsed)
            endpoint["owner_utility"] = utilities[cell]
            cells[cell] = endpoint
        unqualified_cells = [
            cell
            for cell in CELL_ORDER
            if not (
                cells[cell]["mechanically_valid"]
                and cells[cell]["strict_count"] > 0
                and cells[cell]["strict_count"] == cells[cell]["complete_rows"]
                and not any(cells[cell]["metrics"][metric] for metric in ("unmatched", "duplicates", "ambiguous", "malformed", "invalid", "STOP"))
            )
        ]
        source_qualified = not unqualified_cells
        source_reason = (
            None
            if source_qualified
            else f"source-specific tau and utility require qualified endpoints in C00,C10,C01,C11; unqualified cells: {','.join(unqualified_cells)}"
        )
        for cell in CELL_ORDER:
            utility = cells[cell]["owner_utility"]
            cells[cell]["owner_utility"] = {
                "source_specific_status": "qualified" if source_qualified else "unqualified",
                "unqualified_reason": source_reason,
                "horizon_1": utility.get("horizon_1"),
                "horizon_3": utility.get("horizon_3"),
            }
        events.append({
            "event_index": event["event_index"],
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "cells": cells,
            "source_specific_event_status": "qualified" if source_qualified else "unqualified",
            "source_specific_unqualified_reason": source_reason,
            "source_specific_unqualified_cells": unqualified_cells,
            "technical_control": {"cell_id": TECHNICAL_CELL, "parity": shard["parity"], "result_sha256": sha256_json(shard["control"])},
        })
    if tuple(event["event_id"] for event in events) != EVENT_IDS:
        raise EvidenceContractError("event order drifted")
    contrasts, taus = _numeric_contrasts(events)
    source_status = "qualified" if all(event["source_specific_event_status"] == "qualified" for event in events) else "unqualified"
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "complete",
        "unit_id": UNIT_ID,
        "primary": dict(PRIMARY),
        "claim_scope": "case-level selected-three-event evidence only",
        "narrative": "case-level selected-three-event evidence only; this is not checkpoint, training, controller, or production evidence",
        "tau_scope": "source-qualified endpoint arithmetic only; no hidden-state interaction claim",
        "producer": _producer(),
        "source_bindings": {
            "plan": {"path": plan_info["path"], "sha256": plan_info["raw_sha256"], "size": plan_info["size_bytes"], "self_sha256": plan_info["self_sha256"]},
            "pre_gpu_receipt": {"path": pre_gpu_info["path"], "sha256": pre_gpu_info["raw_sha256"], "size": pre_gpu_info["size_bytes"], "self_sha256": pre_gpu_info["self_sha256"]},
            "shards": [
                {"shard_id": shard["result"]["shard_id"], "event_id": shard["event"]["event_id"], "path": shard["root"], "sha256": shard["root_sha256"], "artifacts": shard["artifacts"]}
                for shard in shards
            ],
            **(
                {
                    "finalization_successor": {
                        "path": successor_info["path"],
                        "sha256": successor_info["raw_sha256"],
                        "self_sha256": successor_info["self_sha256"],
                    }
                }
                if successor_info is not None
                else {}
            ),
        },
        "denominators": {"event_count": 3, "image_count": 3, "event_ids": list(EVENT_IDS), "image_ids": [event["image_id"] for event in events]},
        "cell_order": list(CELL_ORDER),
        "events": events,
        "endpoint_vectors": {event["event_id"]: {cell: event["cells"][cell]["metrics"] for cell in CELL_ORDER} for event in events},
        "component_contrasts": contrasts,
        "component_tau": taus,
        "tau": taus,
        "utilities": _utilities(events),
        "source_specific_crossover_status": source_status,
        "source_specific_crossover_qualified": source_status == "qualified",
        "next": {
            "crossover_completed": "observation",
            "authorize_training": False,
            "authorize_A3": False,
            "authorize_P4": False,
            "authorize_production": False,
            "strongest_alternative": "hard oracle bottleneck/grounding collapse versus separable history routing",
            "outcome_to_next_discriminator": {
                "unmatched": "repeat source-specific owner matching and inspect grounding before history interpretation",
                "duplicates": "compare covered-repeat bookkeeping against owner-set transition",
                "invalid_token_grammar": "validate native grammar/token budget before endpoint comparison",
                "native_STOP": "separate admission/STOP from routing and inspect first-token release",
                "matched": "compare component contrasts and owner-set transitions across horizons",
            },
        },
        "safety": {"crossover_completed": "observation", "authorize_training": False, "authorize_A3": False, "authorize_P4": False, "authorize_production": False},
    }
    evidence["self_sha256"] = document_self_sha256(evidence)
    result: dict[str, Any] = {"evidence": evidence}
    if output is not None:
        result["write"] = _write_once(output, evidence)
    if receipt_output is not None:
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "unit_id": UNIT_ID,
            "evidence_sha256": evidence["self_sha256"],
            "event_count": 3,
            "image_count": 3,
            "event_ids": list(EVENT_IDS),
            "cell_order": list(CELL_ORDER),
            "source_bindings": evidence["source_bindings"],
            "denominators": evidence["denominators"],
            "source_specific_crossover_status": source_status,
            "component_tau": taus,
            "component_contrasts_sha256": sha256_json(contrasts),
            "utilities": evidence["utilities"],
            "utilities_sha256": sha256_json(evidence["utilities"]),
            "endpoint_summary": evidence["endpoint_vectors"],
            "endpoint_summary_sha256": sha256_json(evidence["endpoint_vectors"]),
        }
        receipt["self_sha256"] = document_self_sha256(receipt)
        result["receipt"] = receipt
        result["receipt_write"] = _write_once(receipt_output, receipt)
    return result


def finalize_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return finalize(*args, **kwargs)


def validate_evidence(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != EVIDENCE_SCHEMA_VERSION or document.get("unit_id") != UNIT_ID:
        raise EvidenceContractError("evidence schema/unit identity mismatch")
    _self_check(document, "evidence")
    if document.get("source_specific_crossover_status") not in {"qualified", "unqualified"}:
        raise EvidenceContractError("source_specific_crossover_status must be qualified or unqualified")
    if document.get("denominators", {}).get("event_count") != 3 or document.get("denominators", {}).get("image_count") != 3:
        raise EvidenceContractError("evidence denominators drifted")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pre-gpu-receipt", type=Path, required=True)
    parser.add_argument("--shard", dest="shards", action="append", type=Path, required=True, help="one shard root; repeat exactly three times")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", dest="receipt_output", type=Path, required=True)
    parser.add_argument(
        "--finalization-receipt",
        dest="finalization_receipt",
        type=Path,
        default=None,
        help="the one sealed post-execution successor receipt, when authorized",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = finalize(
            args.plan,
            args.pre_gpu_receipt,
            args.shards,
            output=args.output,
            receipt_output=args.receipt_output,
            finalization_receipt=args.finalization_receipt,
        )
    except (EvidenceContractError, FileExistsError, OSError, ValueError) as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(value.get("receipt", value["evidence"])).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

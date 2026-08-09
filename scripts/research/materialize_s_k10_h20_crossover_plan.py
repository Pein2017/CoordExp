#!/usr/bin/env python3
"""Materialize the fixed three-event S K10×H20 crossover plan.

The planner is deliberately CPU-only.  It reads the sealed v3 evidence and
its source lineage, recomputes the intersection used for selection, and emits
one immutable plan.  It never imports torch, loads a model, or chooses events
from a result produced by this successor unit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


UNIT_ID = "2026-08-07-s-k10-h20-natural-crossover"
SOURCE_UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
SCHEMA_VERSION = "s_k10_h20_crossover_plan.v1"
SOURCE_EVIDENCE_SCHEMA_VERSION = "s_natural_boundary_k_n_h_evidence.v3"
SOURCE_RECEIPT_SCHEMA_VERSION = "s_natural_boundary_k_n_h_evidence.v3.receipt"
MANIFEST_SCHEMA_VERSION = "s_natural_boundary_admitted_event_manifest.v3"
ORIGINAL_PLAN_SCHEMA_VERSION = "s_natural_boundary_k_n_h_execution_plan.v1"
GATE_SCHEMA_VERSION = "s_primary_natural_boundary_gate.v1"
PRIMARY = {
    "checkpoint": "S",
    "step": 2444,
    "substrate": "four-coordinate geo_sorted_xy",
}
EXPECTED_SOURCE_EVIDENCE_RAW_SHA256 = "77f90bc48f8126ee1b071767308db603e1b64c0d19a4ca3e5ad8906cfc837598"
EXPECTED_SOURCE_EVIDENCE_SELF_SHA256 = "e04cb63540f9e3a1b27b938ada14f694f6b7208564adcaf2c82656915c62bb95"
EXPECTED_SOURCE_RECEIPT_RAW_SHA256 = "fa11800ec7ad832646162f77c9e5a6df71d2e9ca951144e40b2fd060df46bf8e"
EXPECTED_SOURCE_RECEIPT_SELF_SHA256 = "80aa70f46899474ee1f8646c02d9c061babfac9bd58534f4cfa1d75aa9df12fd"

CELL_ORDER = ("C00", "C10", "C01", "C11")
EVENT_IDS = ("gt:2299:29", "gt:13348:14", "gt:16228:15")
# Physical GPU 2 is occupied by an unrelated live process at launch planning
# time; reserve the current free device 7 instead of evicting or co-locating.
DEVICE_PLAN = {"shard-000": "0", "shard-001": "1", "shard-002": "7"}
MAX_ROWS = 3
MAX_ROW_TOKENS = 256
OPENER_MODE = "pre_opener_natural"

FROZEN_FLAGS = {
    "no_event_reorder": True,
    "no_reselection": True,
    "no_sweep": True,
    "no_a3": True,
    "no_p4": True,
    "no_training": True,
    "use_cache": False,
    "opener_injected": False,
}


class PlanError(ValueError):
    """Raised for a source, plan, or identity contract violation."""


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
        raise PlanError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    target = _regular_file(path, "hash source")
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PlanError(f"cannot read {target}: {exc}") from exc
    return digest.hexdigest()


def document_self_sha256(document: Mapping[str, Any], field: str = "self_sha256") -> str:
    body = dict(document)
    body.pop(field, None)
    return sha256_json(body)


def _absolute(value: str | Path, label: str) -> Path:
    target = Path(value).expanduser()
    if not target.is_absolute():
        raise PlanError(f"{label} must be an absolute path")
    cursor = target
    while True:
        if cursor.is_symlink():
            raise PlanError(f"{label} must not traverse a symlink: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    try:
        resolved = target.resolve(strict=False)
    except OSError as exc:
        raise PlanError(f"cannot resolve {label}: {target}") from exc
    if resolved != target:
        raise PlanError(f"{label} resolves through a symlink: {target}")
    return resolved


def _regular_file(value: str | Path, label: str) -> Path:
    target = _absolute(value, label)
    if target.is_symlink() or not target.is_file():
        raise PlanError(f"{label} must be an existing regular non-symlink file: {target}")
    return target


def _read_json(
    value: str | Path | Mapping[str, Any],
    label: str,
    *,
    canonical: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(value, Mapping):
        document = dict(value)
        canonical_json_bytes(document)
        return document, {"inline": True, "raw_sha256": sha256_json(document), "path": None}
    path = _regular_file(value, label)
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"{label} is not readable JSON: {path}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise PlanError(f"{label} must contain a JSON object")
    document = dict(parsed)
    canonical_json_bytes(document)
    if canonical and raw != canonical_json_bytes(document) + b"\n":
        raise PlanError(f"{label} must be canonical JSON with one trailing newline")
    return document, {"path": str(path), "raw_sha256": sha256_bytes(raw), "size_bytes": len(raw)}


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise PlanError(f"{label} must be a lowercase SHA-256")
    return value


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise PlanError(f"{label} must be boolean")
    return value


def _int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise PlanError(f"{label} must be >= {minimum}")
    return int(value)


def _write_once(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    target = _absolute(path, "plan output")
    payload = canonical_json_bytes(document) + b"\n"
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(f"immutable plan collision: {target}")
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


def _source_ref(
    value: str | Path | Mapping[str, Any],
    label: str,
    *,
    expected_raw: str | None = None,
    canonical: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, info = _read_json(value, label, canonical=canonical)
    if info.get("path") is None:
        raw_sha = info["raw_sha256"]
        path = None
    else:
        raw_sha = str(info["raw_sha256"])
        path = info["path"]
    if expected_raw is not None and raw_sha != expected_raw:
        raise PlanError(f"{label} raw SHA-256 differs from frozen source")
    semantic = document.get("self_sha256")
    if semantic is not None:
        _sha(semantic, f"{label}.self_sha256")
        if semantic != document_self_sha256(document):
            raise PlanError(f"{label}.self_sha256 mismatch")
    return (
        {
            "path": path,
            "raw_sha256": raw_sha,
            "size_bytes": info.get("size_bytes"),
            "semantic_self_sha256": semantic or sha256_json(document),
        },
        document,
    )


def _required_source_identity(document: Mapping[str, Any], label: str) -> None:
    if document.get("unit_id") != SOURCE_UNIT_ID:
        raise PlanError(f"{label} belongs to another unit")
    if document.get("primary") != PRIMARY:
        raise PlanError(f"{label} primary S substrate identity drifted")


def _validate_source_evidence(value: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _source_ref(value, "source v3 evidence", expected_raw=EXPECTED_SOURCE_EVIDENCE_RAW_SHA256)
    if document.get("schema_version") != SOURCE_EVIDENCE_SCHEMA_VERSION or document.get("status") != "complete":
        raise PlanError("source v3 evidence schema/status drifted")
    _required_source_identity(document, "source v3 evidence")
    if document.get("decision_owner") != "S" or document.get("contrast_semantics") != "intervention_minus_own_baseline_only":
        raise PlanError("source v3 evidence decision/contrast ownership drifted")
    if document.get("self_sha256") != EXPECTED_SOURCE_EVIDENCE_SELF_SHA256:
        raise PlanError("source v3 evidence semantic self hash differs from frozen v3")
    events = document.get("events")
    if not isinstance(events, list):
        raise PlanError("source v3 evidence events are missing")
    by_id = {event.get("event_id"): event for event in events if isinstance(event, Mapping)}
    if len(by_id) != len(events):
        raise PlanError("source v3 evidence event identities are not unique")
    selected: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        arms = event.get("arms")
        disposition = event.get("dynamic_factor_disposition")
        k10 = arms.get("K10") if isinstance(arms, Mapping) else None
        h20 = disposition.get("H20") if isinstance(disposition, Mapping) else None
        if (
            isinstance(k10, Mapping)
            and k10.get("target_strict_release") is True
            and isinstance(h20, Mapping)
            and h20.get("factor_qualified") is True
            and h20.get("degenerate_grammar_disruption") is False
        ):
            selected.append(event)
    actual_ids = tuple(event.get("event_id") for event in selected)
    if actual_ids != EVENT_IDS:
        raise PlanError(f"frozen K10∩H20 selection changed: {actual_ids!r}")
    return ref, document


def _validate_source_receipt(value: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _source_ref(value, "source v3 receipt", expected_raw=EXPECTED_SOURCE_RECEIPT_RAW_SHA256)
    if document.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION or document.get("status") != "complete":
        raise PlanError("source v3 receipt schema/status drifted")
    if document.get("unit_id") != SOURCE_UNIT_ID:
        raise PlanError("source v3 receipt belongs to another unit")
    if document.get("evidence_raw_file_sha256") != EXPECTED_SOURCE_EVIDENCE_RAW_SHA256:
        raise PlanError("source v3 receipt does not bind exact evidence raw hash")
    if document.get("evidence_semantic_self_sha256") != EXPECTED_SOURCE_EVIDENCE_SELF_SHA256:
        raise PlanError("source v3 receipt does not bind exact evidence semantic hash")
    if document.get("self_sha256") != EXPECTED_SOURCE_RECEIPT_SELF_SHA256:
        raise PlanError("source v3 receipt semantic self hash differs from frozen receipt")
    return ref, document


def _validate_manifest(value: str | Path | Mapping[str, Any], census_ref: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _source_ref(value, "source admitted manifest")
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION or document.get("status") != "sealed":
        raise PlanError("source admitted manifest schema/status drifted")
    if document.get("unit_id") != SOURCE_UNIT_ID or document.get("primary") != PRIMARY:
        raise PlanError("source admitted manifest identity drifted")
    source_census = document.get("source_census")
    if not isinstance(source_census, Mapping) or source_census.get("sha256") != census_ref["raw_sha256"]:
        raise PlanError("source manifest is not bound to supplied census")
    events = document.get("events")
    if not isinstance(events, list):
        raise PlanError("source admitted manifest events are missing")
    for event in events:
        if not isinstance(event, Mapping):
            raise PlanError("source admitted manifest event is malformed")
        declared = event.get("event_sha256")
        if not isinstance(declared, str) or declared != sha256_json({k: v for k, v in event.items() if k != "event_sha256"}):
            raise PlanError(f"source manifest event {event.get('event_id')} hash mismatch")
    return ref, document


def _validate_census(value: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _source_ref(value, "source census")
    if document.get("schema_version") != "natural_boundary_owner_admission_census.v3" or document.get("status") != "sealed":
        raise PlanError("source census schema/status drifted")
    if document.get("unit_id") != SOURCE_UNIT_ID or document.get("census_revision") != "census-v3":
        raise PlanError("source census identity drifted")
    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != 784:
        raise PlanError("source census must contain exactly 784 rows")
    return ref, document


def _validate_original_plan(value: str | Path | Mapping[str, Any], manifest_ref: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _source_ref(value, "source original execution plan")
    if document.get("schema_version") != ORIGINAL_PLAN_SCHEMA_VERSION or document.get("status") != "planned":
        raise PlanError("source original execution plan schema/status drifted")
    if document.get("unit_id") != SOURCE_UNIT_ID or document.get("primary") != PRIMARY:
        raise PlanError("source original execution plan identity drifted")
    if document.get("manifest_sha256") != manifest_ref["raw_sha256"] or document.get("manifest_self_sha256") != manifest.get("self_sha256"):
        raise PlanError("source original execution plan is not bound to supplied manifest")
    if document.get("no_2x2") is not True or document.get("no_a3") is not True or document.get("no_p4") is not True or document.get("no_sweep") is not True:
        raise PlanError("source original execution plan safety flags drifted")
    return ref, document


def _validate_gate(value: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # The frozen gate writer retained a human-readable raw result.  Bind its
    # exact bytes while validating the finite semantic body independently.
    ref, document = _source_ref(value, "source gate result", canonical=False)
    if document.get("schema_version") != GATE_SCHEMA_VERSION or document.get("unit_id") != SOURCE_UNIT_ID:
        raise PlanError("source gate result schema/unit drifted")
    if document.get("checkpoint") != "S" or document.get("event_id") != "gt:5001:15":
        raise PlanError("source gate result event identity drifted")
    if document.get("gpu_launch_authorized") is not False or document.get("no_training") is not True:
        raise PlanError("source gate result crosses the no-GPU/no-training boundary")
    declared = document.get("result_sha256")
    if declared != document_self_sha256(document, "result_sha256"):
        raise PlanError("source gate result self hash mismatch")
    return ref, document


def _source_event_binding(event: Mapping[str, Any], manifest_event: Mapping[str, Any]) -> dict[str, Any]:
    event_id = event.get("event_id")
    if event_id != manifest_event.get("event_id") or event.get("image_id") != manifest_event.get("image_id"):
        raise PlanError(f"source evidence/manifest event identity differs for {event_id}")
    if not isinstance(event_id, str) or event_id not in EVENT_IDS:
        raise PlanError(f"unexpected selected event identity: {event_id}")
    natural = manifest_event.get("natural_boundary")
    geometry = manifest_event.get("geometry")
    if not isinstance(natural, Mapping) or not isinstance(geometry, Mapping):
        raise PlanError(f"source event {event_id} lacks natural prefix or geometry")
    prefix_hash = natural.get("prefix_sha256")
    if prefix_hash != natural.get("history_sha256"):
        raise PlanError(f"source event {event_id} prefix/history hashes differ")
    _sha(prefix_hash, f"source event {event_id}.prefix_sha256")
    geometry_hash = geometry.get("geometry_sha256")
    if geometry_hash != sha256_json({k: v for k, v in geometry.items() if k != "geometry_sha256"}):
        raise PlanError(f"source event {event_id} geometry hash mismatch")
    manifest_hash = manifest_event.get("event_sha256")
    if manifest_hash != sha256_json({k: v for k, v in manifest_event.items() if k != "event_sha256"}):
        raise PlanError(f"source event {event_id} event hash mismatch")
    if natural.get("pre_opener_natural") is not True or natural.get("opener_injected") is not False or natural.get("opener_seeded") is not False:
        raise PlanError(f"source event {event_id} is not the exact natural prefix")
    return {
        "event_index": _int(manifest_event.get("event_index"), f"source event {event_id}.event_index", minimum=0),
        "event_id": event_id,
        "image_id": _int(manifest_event.get("image_id"), f"source event {event_id}.image_id", minimum=0),
        "event_sha256": manifest_hash,
        "prefix_sha256": prefix_hash,
        "geometry_sha256": geometry_hash,
        "target_owner_id": event.get("target_owner_id"),
        "covered_owner_ids": list(manifest_event.get("owner_refs", {}).get("covered_owner_ids", [])),
        "source_qualification": {
            "k10_target_strict_release": True,
            "h20_factor_qualified": True,
            "h20_nondegenerate": True,
        },
    }


def _endpoint_summary(event: Mapping[str, Any], arm: str) -> dict[str, Any]:
    arms = event.get("arms")
    value = arms.get(arm) if isinstance(arms, Mapping) else None
    if not isinstance(value, Mapping):
        raise PlanError(f"source event {event.get('event_id')} lacks endpoint arm {arm}")
    fields = (
        "target_strict_release",
        "strict_physical_owner_ids",
        "strict_physical_owner_match_count",
        "uncovered_owner_gain_ids",
        "covered_repeat_owner_ids",
        "complete_natural_rows",
        "duplicate_rows",
        "invalid_rows",
        "malformed_rows",
        "unmatched_rows",
        "native_stop",
        "terminal_reason",
        "mechanically_valid",
        "scientific_unmatched_is_valid",
        "admission_mode",
        "opener_generated_by_model",
    )
    result = {field: value.get(field) for field in fields}
    result["endpoint_sha256"] = sha256_json(result)
    return result


def _cell_contract(cell_id: str) -> dict[str, Any]:
    if cell_id == "C00":
        return {
            "cell_id": "C00",
            "label": "explicit all-allowed causal no-op",
            "baseline_arm": "K01",
            "source_arm": "K01",
            "technical_control": True,
            "hidden_parity_arm": "K00",
            "semantics": "all allowed keys, explicit K01-style no-op; hidden K00 full-vocabulary parity",
        }
    if cell_id == "C10":
        return {
            "cell_id": "C10",
            "label": "K10 static routing",
            "baseline_arm": "C00",
            "source_arm": "K10",
            "technical_control": False,
            "semantics": "fresh crossover runner K10 operator under the exact natural prefix",
        }
    if cell_id == "C01":
        return {
            "cell_id": "C01",
            "label": "H20 history factor",
            "baseline_arm": "C00",
            "source_arm": "H20",
            "technical_control": False,
            "semantics": "fresh crossover runner H20 operator under the exact natural prefix",
        }
    return {
        "cell_id": "C11",
        "label": "K10 and H20 crossover",
        "baseline_arm": "C00",
        "source_arms": ["K10", "H20"],
        "technical_control": False,
        "semantics": "fresh simultaneous K10∧H20 operator; no source endpoint is presumed",
    }


def _build_document(
    *,
    source_evidence: Any,
    source_receipt: Any,
    manifest: Any,
    census: Any,
    original_plan: Any,
    gate_result: Any,
) -> dict[str, Any]:
    evidence_ref, evidence_doc = _validate_source_evidence(source_evidence)
    receipt_ref, receipt_doc = _validate_source_receipt(source_receipt)
    census_ref, census_doc = _validate_census(census)
    manifest_ref, manifest_doc = _validate_manifest(manifest, census_ref)
    original_plan_ref, original_plan_doc = _validate_original_plan(original_plan, manifest_ref, manifest_doc)
    gate_ref, gate_doc = _validate_gate(gate_result)
    if receipt_doc.get("evidence_raw_file_sha256") != evidence_ref["raw_sha256"]:
        raise PlanError("source receipt/evidence raw identity differs")

    evidence_events = {event["event_id"]: event for event in evidence_doc["events"] if isinstance(event, Mapping)}
    manifest_events = {event["event_id"]: event for event in manifest_doc["events"] if isinstance(event, Mapping)}
    selected = []
    for event_id in EVENT_IDS:
        if event_id not in evidence_events or event_id not in manifest_events:
            raise PlanError(f"selected source event is absent: {event_id}")
        selected.append(_source_event_binding(evidence_events[event_id], manifest_events[event_id]))
    if tuple(ref["event_id"] for ref in selected) != EVENT_IDS:
        raise PlanError("selected event order drifted")

    source_endpoint_vectors = {
        ref["event_id"]: {
            arm: _endpoint_summary(evidence_events[ref["event_id"]], arm)
            for arm in ("K01", "K10", "H00", "H20")
        }
        for ref in selected
    }
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "unit_id": UNIT_ID,
        "primary": dict(PRIMARY),
        "source_bindings": {
            "evidence": evidence_ref,
            "receipt": receipt_ref,
            "manifest": manifest_ref,
            "census": census_ref,
            "original_plan": original_plan_ref,
            "gate_result": gate_ref,
        },
        "source_identities": {
            "source_unit_id": SOURCE_UNIT_ID,
            "evidence_schema": SOURCE_EVIDENCE_SCHEMA_VERSION,
            "evidence_raw_sha256": evidence_ref["raw_sha256"],
            "evidence_self_sha256": evidence_doc["self_sha256"],
            "receipt_raw_sha256": receipt_ref["raw_sha256"],
            "receipt_self_sha256": receipt_doc["self_sha256"],
            "manifest_raw_sha256": manifest_ref["raw_sha256"],
            "manifest_self_sha256": manifest_doc["self_sha256"],
            "census_raw_sha256": census_ref["raw_sha256"],
            "census_self_sha256": census_doc["self_sha256"],
            "original_plan_raw_sha256": original_plan_ref["raw_sha256"],
            "original_plan_self_sha256": original_plan_doc["plan_sha256"],
            "gate_result_raw_sha256": gate_ref["raw_sha256"],
            "gate_result_self_sha256": gate_doc["result_sha256"],
        },
        "selection_rule": {
            "rule_id": "intersection_k10_target_strict_release_h20_factor_qualified_nondegenerate_v3",
            "source_evidence_schema": SOURCE_EVIDENCE_SCHEMA_VERSION,
            "k10": {"arm": "K10", "field": "target_strict_release", "equals": True},
            "h20": {
                "factor": "H20",
                "field": "factor_qualified",
                "equals": True,
                "degenerate_grammar_disruption": False,
            },
            "expected_event_ids": list(EVENT_IDS),
            "expected_event_count": 3,
            "expected_distinct_image_count": 3,
        },
        "events": selected,
        "event_count": 3,
        "image_count": 3,
        "cell_order": list(CELL_ORDER),
        "technical_control": "C00",
        "cells": {cell_id: _cell_contract(cell_id) for cell_id in CELL_ORDER},
        "source_endpoint_vectors": source_endpoint_vectors,
        "estimands": {
            "endpoint_vector": [
                "target_strict_release",
                "strict_physical_owner_ids",
                "strict_physical_owner_match_count",
                "uncovered_owner_gain_ids",
                "covered_repeat_owner_ids",
                "complete_natural_rows",
                "duplicate_rows",
                "invalid_rows",
                "malformed_rows",
                "unmatched_rows",
                "native_stop",
                "terminal_reason",
                "mechanically_valid",
                "scientific_unmatched_is_valid",
            ],
            "contrast": "intervention_minus_own_baseline_only",
            "tau": "componentwise descriptive endpoint deltas only",
            "charged_utility": "componentwise descriptive gained-minus-lost utility only",
            "source_specific_crossover": "unqualified unless C00,C10,C01,C11 each have qualified source-specific endpoints",
            "unmatched": "mechanically valid scientific outcome, not technical invalidity",
            "training_claim": False,
        },
        "operator_contract": {
            "admission_mode": OPENER_MODE,
            "opener_injected": False,
            "opener_generated_by_model": True,
            "use_cache": False,
            "max_rows": MAX_ROWS,
            "max_row_tokens": MAX_ROW_TOKENS,
            "full_endpoint_vectors_required": True,
            "no_training": True,
        },
        "device_plan": dict(DEVICE_PLAN),
        "shards": [
            {
                "shard_index": index,
                "shard_id": f"shard-{index:03d}",
                "physical_device": DEVICE_PLAN[f"shard-{index:03d}"],
                "logical_device": "cuda:0",
                "event": selected[index],
                "events": [selected[index]],
            }
            for index in range(3)
        ],
        **FROZEN_FLAGS,
        "no_legacy_no_2x2_reuse": True,
        "no_legacy_plan_reselection": True,
        "claim_boundary": {
            "execution_scope": "one_fixed_three_shard_no_training_launch",
            "source_specific_crossover_qualified": False,
            "training_claim_qualified": False,
        },
    }
    plan["self_sha256"] = document_self_sha256(plan)
    return plan


def build_plan(
    source_evidence: str | Path | Mapping[str, Any],
    source_receipt: str | Path | Mapping[str, Any],
    manifest: str | Path | Mapping[str, Any],
    census: str | Path | Mapping[str, Any],
    original_plan: str | Path | Mapping[str, Any],
    gate_result: str | Path | Mapping[str, Any],
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Build and optionally write the immutable crossover plan."""

    document = _build_document(
        source_evidence=source_evidence,
        source_receipt=source_receipt,
        manifest=manifest,
        census=census,
        original_plan=original_plan,
        gate_result=gate_result,
    )
    if output is not None:
        _write_once(output, document)
    return document


def _validate_plan_document(document: Mapping[str, Any], plan_path: Path | None = None) -> dict[str, Any]:
    if document.get("schema_version") != SCHEMA_VERSION or document.get("status") != "planned" or document.get("unit_id") != UNIT_ID:
        raise PlanError("plan schema/status/unit identity drifted")
    if document.get("primary") != PRIMARY:
        raise PlanError("plan primary identity drifted")
    if document.get("self_sha256") != document_self_sha256(document):
        raise PlanError("plan self_sha256 mismatch")
    if document.get("cell_order") != list(CELL_ORDER) or document.get("technical_control") != "C00":
        raise PlanError("plan cell order/technical control drifted")
    if document.get("event_count") != 3 or document.get("image_count") != 3:
        raise PlanError("plan event/image count drifted")
    if document.get("events") is None or not isinstance(document["events"], list) or len(document["events"]) != 3:
        raise PlanError("plan must contain exactly three events")
    if tuple(event.get("event_id") for event in document["events"] if isinstance(event, Mapping)) != EVENT_IDS:
        raise PlanError("plan selected event order drifted")
    if len({event.get("image_id") for event in document["events"] if isinstance(event, Mapping)}) != 3:
        raise PlanError("plan selected events must use three distinct images")
    if document.get("device_plan") != DEVICE_PLAN:
        raise PlanError("plan device assignment drifted")
    if document.get("shards") is None or not isinstance(document["shards"], list) or len(document["shards"]) != 3:
        raise PlanError("plan must contain exactly three shards")
    for index, shard in enumerate(document["shards"]):
        shard_id = f"shard-{index:03d}"
        if not isinstance(shard, Mapping) or shard.get("shard_index") != index or shard.get("shard_id") != shard_id or shard.get("physical_device") != DEVICE_PLAN[shard_id]:
            raise PlanError(f"plan shard {index} identity/device drifted")
        events = shard.get("events")
        if not isinstance(events, list) or len(events) != 1 or events[0] != shard.get("event") or events[0] != document["events"][index]:
            raise PlanError(f"plan shard {index} must contain exactly its one event")
    for key, expected in FROZEN_FLAGS.items():
        if document.get(key) is not expected:
            raise PlanError(f"plan flag {key} drifted")
    if document.get("no_legacy_no_2x2_reuse") is not True or document.get("no_legacy_plan_reselection") is not True:
        raise PlanError("plan legacy no-2x2/reselection guard drifted")
    op = document.get("operator_contract")
    if not isinstance(op, Mapping) or op.get("admission_mode") != OPENER_MODE or op.get("opener_injected") is not False or op.get("use_cache") is not False or op.get("max_rows") != MAX_ROWS or op.get("max_row_tokens") != MAX_ROW_TOKENS:
        raise PlanError("plan operator contract drifted")
    selection = document.get("selection_rule")
    if not isinstance(selection, Mapping) or selection.get("expected_event_ids") != list(EVENT_IDS) or selection.get("expected_event_count") != 3 or selection.get("expected_distinct_image_count") != 3:
        raise PlanError("plan selection rule drifted")
    refs = document.get("source_bindings")
    if not isinstance(refs, Mapping) or set(refs) != {"evidence", "receipt", "manifest", "census", "original_plan", "gate_result"}:
        raise PlanError("plan source bindings are incomplete")
    return {"plan": dict(document), "plan_sha256": document["self_sha256"], "self_sha256": document["self_sha256"], "path": str(plan_path) if plan_path else None}


def validate_plan(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Validate a materialized plan and all bound source identities."""

    if isinstance(value, Mapping):
        document = dict(value)
        path = None
    else:
        path = _regular_file(value, "plan")
        raw = path.read_bytes()
        try:
            parsed = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PlanError(f"plan is not readable JSON: {path}: {exc}") from exc
        if not isinstance(parsed, Mapping) or raw != canonical_json_bytes(parsed) + b"\n":
            raise PlanError("plan must be canonical JSON with one trailing newline")
        document = dict(parsed)
    structural = _validate_plan_document(document, path)
    sources = document["source_bindings"]
    # Re-read and validate source documents by the paths sealed in the plan.
    evidence = sources["evidence"].get("path")
    receipt = sources["receipt"].get("path")
    manifest = sources["manifest"].get("path")
    census = sources["census"].get("path")
    original = sources["original_plan"].get("path")
    gate = sources["gate_result"].get("path")
    if not all(isinstance(item, str) and item for item in (evidence, receipt, manifest, census, original, gate)):
        raise PlanError("plan source bindings must retain exact source paths")
    expected = build_plan(evidence, receipt, manifest, census, original, gate)
    if expected != document:
        raise PlanError("plan does not equal deterministic materialization from bound sources")
    return structural


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, help_text in (
        ("source-evidence", "sealed v3 evidence.json"),
        ("source-receipt", "sealed v3 evidence.receipt.json"),
        ("manifest", "source admitted event manifest"),
        ("census", "source census-v3 document"),
        ("original-plan", "source execution plan"),
        ("gate-result", "source gate result.json"),
    ):
        parser.add_argument(f"--{name}", type=Path, required=True, help=help_text)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = build_plan(
            args.source_evidence,
            args.source_receipt,
            args.manifest,
            args.census,
            args.original_plan,
            args.gate_result,
            args.output,
        )
    except (PlanError, FileExistsError, OSError, ValueError) as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(document).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

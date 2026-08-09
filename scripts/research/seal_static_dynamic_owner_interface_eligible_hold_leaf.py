#!/usr/bin/env python3
"""Seal the exhausted S/ordinal-11 eligible technical-HOLD evidence leaf.

This is an administrative evidence sealer.  It never imports a model runner,
loads a checkpoint, calls an actuator, or creates synthetic scientific output.
The accepted input is deliberately singular: the frozen S ordinal-11 cohort
event and its two already-existing, pre-actuator failed attempt directories.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
SCHEMA_VERSION = "static_dynamic_owner_interface_eligible_hold_leaf.v1"
RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.receipt.v1"
RUNTIME_SCHEMA_VERSION = "static_dynamic_owner_interface_experiment.v1"
RUNTIME_ATTESTATION_SCHEMA_VERSION = f"{RUNTIME_SCHEMA_VERSION}.runtime_attestation.v1"

WORKTREE_ROOT = Path("/data/CoordExp/.worktrees/research-probes")
EXPERIMENT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover"
)
UNIT_PATH = WORKTREE_ROOT / (
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-08-05-static-dynamic-owner-interface-crossover/unit.md"
)
TASKS_PATH = UNIT_PATH.with_name("tasks.md")
COHORT_PATH = EXPERIMENT_ROOT / "cohort/s-step2444-final-support.json"
COHORT_MANIFEST_PATH = EXPERIMENT_ROOT / "cohort/s-step2444-final-support.manifest.json"
INITIAL_ATTEMPT_DIR = EXPERIMENT_ROOT / "p1-p4/scored-smoke/s-step2444-ordinal11-all-live"
REPAIR_ATTEMPT_DIR = EXPERIMENT_ROOT / "p1-p4/scored-smoke/s-step2444-ordinal11-all-live-repair1"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "p1-p4/eligible-hold/s-step2444-ordinal11"

LEAF_FILENAME = "eligible_hold_leaf.json"
RECEIPT_FILENAME = "eligible_hold_leaf.receipt.json"

EXPECTED_UNIT_SHA256 = "da7cd13cfe83990647a68fca45f7f00ecf65fbd87e61856dcf53e1b7fe701513"
EXPECTED_TASKS_SHA256 = "ecb7eb1a8d20d6dbfd09f42638b2bbcabdd5ab262707c52a2ba8ae0f07d4128c"
EXPECTED_COHORT_SHA256 = "4bab115670d2896b98ac866dd2d4a650986c1ed35288783f2acfbe4e9340dcdb"
EXPECTED_COHORT_MANIFEST_SHA256 = "18437e0fde19400b5097bd6ce8066baa4c9908353bf2bbd7d1a512fc6b933196"
EXPECTED_CONFIG_SHA256 = "d2217208bc3e419bc9d8c621b4842da262d216780eb1dfc88eade0edf1358f6b"
EXPECTED_CONFIG_FINGERPRINT = "a5ba21ab3e495fa22770ae5d156372d7894fec61f4ac905553db52540b8ead12"
EXPECTED_DERIVED_PANEL_SHA256 = "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23"
EXPECTED_DERIVED_RECEIPT_SHA256 = "cd1273f627f7bdfcb16e9ca6e4a50e9d51f0163081ea7458d6d3deabe9bb82c0"
EXPECTED_SOURCE_PANEL_SHA256 = "01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8"
EXPECTED_H0_LEDGER_SHA256 = "5a73390ce66a0c318964d684a0c75c8a3756d8cde4708a9015eb2628aafca3d9"
EXPECTED_SUPPORT_LEDGER_SHA256 = "dc2df6f4370526c51d212d49acd5b809fa33e71f3fd21482d2cd98a2cc94afb5"
EXPECTED_HELPER_SCHEMA_SHA256 = "a4a259ce9939c4b6692351e57f8e30ad66e18dc4e85d686ee5cd5e117d5cf621"
EXPECTED_PREFIX_SHA256 = "872748d64025444e40d097f2f8605b7dd2d89b4b13b1666b820b0a1f73cfe238"
EXPECTED_PHYSICAL_DEVICE_UUID = "GPU-8d43cb78-19ca-2f59-3179-7ea166cb1a4e"

P1_CELLS = (
    "K00",
    "K01",
    "K10",
    "K11",
    "K12",
    "K13",
    *(f"{arm}_block{block}" for block in (13, 23) for arm in ("R00", "R10", "R11", "R12")),
    "R00_block27",
    "R10_block27",
)
P2_CELLS = tuple(
    f"{arm}.horizon_{horizon}"
    for arm in ("D00", "D01", "D10", "D11", "D12", "D20", "D21")
    for horizon in (1, 3)
)
P3_CELLS = tuple(
    f"{cell}.horizon_{horizon}"
    for cell in ("Y00", "Y10", "Y01", "Y11")
    for horizon in (1, 3)
)
P4_CELLS = (
    "target_b_complete_row_nll",
    "uncovered_b_vs_covered_a_margin_loss",
    "fixed_sum_coupled",
)

ABSENT_ATTEMPT_ARTIFACTS = (
    "exact_prefix_manifest.json",
    "intervention_manifest.json",
    "per_event_results.jsonl",
    "gradient_receipt.json",
    "terminal_summary.json",
)


class HoldSealError(ValueError):
    """Raised when the singular eligible-HOLD evidence contract is not exact."""


@dataclass(frozen=True)
class AttemptSpec:
    role: str
    runtime_identity_sha256: str
    runtime_identity_size: int
    file_census_sha256: str
    pid: int
    timestamp_utc: str
    cause_code: str
    cause_summary: str


INITIAL_ATTEMPT = AttemptSpec(
    role="initial",
    runtime_identity_sha256="87dcd16ca3c4fc40997fac9bc70b67af163cc9eb02fd3c38026d685fe401213a",
    runtime_identity_size=159170,
    file_census_sha256="1463f696a5cd7f22688b3a2716817de331861f86b196bedae093b02689f1e337",
    pid=3311728,
    timestamp_utc="2026-08-05T23:33:23.409360Z",
    cause_code="boolean_receipt_false_positive_on_identity_field",
    cause_summary=(
        "Lead observed the production identity field same_class_competitor_owner_id "
        "being rejected as a boolean receipt."
    ),
)
REPAIR_ATTEMPT = AttemptSpec(
    role="repair1",
    runtime_identity_sha256="95d227c1c0f81ca7339c3f7c854199871d1c1118ca8203eedfec4d1cf09fcca5",
    runtime_identity_size=159170,
    file_census_sha256="0aaf15e57b64bcc22a0ced6749debdcea79147a51b902c4ef6046e02536e9208",
    pid=3322959,
    timestamp_utc="2026-08-05T23:36:57.470177Z",
    cause_code="stored_single_image_grid_shape_rejected_pre_forward",
    cause_summary=(
        "Lead observed the stored rank-1 image_grid_thw reaching a helper that "
        "required a native batched grid before the first static-arm forward."
    ),
)


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
        raise HoldSealError(f"value is not canonical finite JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HoldSealError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def document_self_sha256(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("self_sha256", None)
    return sha256_json(payload)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HoldSealError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HoldSealError(f"{label} must be a JSON object")
    canonical_json_bytes(payload)
    return payload


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise HoldSealError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise HoldSealError(f"{label} mismatch: expected {expected!r}, observed {value!r}")


def validate_frozen_contract_file(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise HoldSealError(f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {observed}")
    return {
        "path": str(resolved),
        "sha256": observed,
        "size_bytes": resolved.stat().st_size,
    }


def _artifact_ref(role: str, path: Path, expected_sha256: str) -> dict[str, Any]:
    ref = validate_frozen_contract_file(path, expected_sha256=expected_sha256, label=role)
    return {"role": role, **ref}


def _source_entry(
    sources: Mapping[str, Any],
    key: str,
    *,
    list_entry: bool,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    raw = sources.get(key)
    if list_entry:
        if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], Mapping):
            raise HoldSealError(f"cohort sources.{key} must contain exactly one source")
        entry = dict(raw[0])
    else:
        if not isinstance(raw, Mapping):
            raise HoldSealError(f"cohort sources.{key} must be an object")
        entry = dict(raw)
    _require_exact(entry.get("sha256"), expected_sha256, f"cohort sources.{key}.sha256")
    path = Path(str(entry.get("path"))).expanduser().resolve(strict=True)
    if sha256_file(path) != expected_sha256:
        raise HoldSealError(f"cohort sources.{key} bytes differ from the frozen SHA-256")
    return path, entry


def _one_ledger_record(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise HoldSealError(f"{label}.records must be a JSON array")
    hits = [
        item
        for item in records
        if isinstance(item, Mapping)
        and item.get("gt_owner_id") == "gt:5001:15"
        and str(item.get("image_id")) == "5001"
    ]
    if len(hits) != 1:
        raise HoldSealError(f"{label} must contain exactly one S ordinal-11 owner record")
    return dict(hits[0])


def validate_production_binding() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    refs = [
        _artifact_ref("cohort", COHORT_PATH, EXPECTED_COHORT_SHA256),
        _artifact_ref("cohort_manifest", COHORT_MANIFEST_PATH, EXPECTED_COHORT_MANIFEST_SHA256),
    ]
    cohort = _read_json(COHORT_PATH, "cohort")
    _require_exact(cohort.get("unit_id"), UNIT_ID, "cohort unit_id")
    _require_exact(cohort.get("schema_version"), "static_dynamic_owner_interface_cohort.v1", "cohort schema")
    events = cohort.get("events")
    if not isinstance(events, list):
        raise HoldSealError("cohort events must be a JSON array")
    matches = [item for item in events if isinstance(item, Mapping) and item.get("ordinal") == 11]
    if len(matches) != 1:
        raise HoldSealError("cohort must contain exactly one ordinal-11 event")
    event = dict(matches[0])
    _require_exact(event.get("gt_owner_id"), "gt:5001:15", "cohort event owner")
    _require_exact(event.get("image_id"), 5001, "cohort event image")
    _require_exact(event.get("source_panel_object_index"), 15, "cohort event source index")
    _require_exact(event.get("disposition"), "established", "cohort event disposition")
    _require_exact(event.get("geometry_status"), "available", "cohort event geometry status")
    _require_exact(
        event.get("geometry_mechanical_disposition"),
        "eligible_verified_pair_regions",
        "cohort event geometry disposition",
    )
    _require_exact(event.get("geometry_launch_eligible"), True, "cohort event geometry eligibility")
    regions = event.get("image_cell_regions")
    if not isinstance(regions, Mapping):
        raise HoldSealError("cohort event image_cell_regions must be an object")
    expected_region_counts = {
        "a_exclusive": 30,
        "b_exclusive": 140,
        "background": 140,
        "same_class_competitor": 20,
        "shared_core": 14,
    }
    for name, count in expected_region_counts.items():
        values = regions.get(name)
        if not isinstance(values, list) or len(values) != count:
            raise HoldSealError(f"cohort event region {name} differs from the frozen geometry")
    pairs = event.get("A_B")
    if not isinstance(pairs, Mapping) or not isinstance(pairs.get("S"), Mapping):
        raise HoldSealError("cohort event lacks the S A/B pair")
    pair = dict(pairs["S"])
    _require_exact(pair.get("pair_status"), "verified_pair", "cohort S pair status")
    a = pair.get("A_latest_covered")
    b = pair.get("B_verified_uncovered")
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        raise HoldSealError("cohort S verified pair lacks A or B")
    expected_a = {
        "gt_owner_id": "gt:5001:2",
        "natural_boundary": 0,
        "source_panel_object_index": 2,
        "strict_complete_row": True,
    }
    expected_b = {
        "exact_prefix_sha256": EXPECTED_PREFIX_SHA256,
        "gt_owner_id": "gt:5001:15",
        "natural_boundary": 1,
        "strict_complete_row": False,
        "verified_support": True,
    }
    _require_exact(dict(a), expected_a, "cohort S covered-A binding")
    _require_exact(dict(b), expected_b, "cohort S uncovered-B binding")

    manifest = _read_json(COHORT_MANIFEST_PATH, "cohort manifest")
    _require_exact(manifest.get("unit_id"), UNIT_ID, "cohort manifest unit_id")
    _require_exact(manifest.get("cohort_sha256"), EXPECTED_COHORT_SHA256, "cohort manifest binding")
    _require_exact(manifest.get("event_count"), 32, "cohort manifest event count")
    sources = cohort.get("sources")
    if not isinstance(sources, Mapping):
        raise HoldSealError("cohort sources must be an object")
    source_specs = (
        ("derived_panel", False, EXPECTED_DERIVED_PANEL_SHA256),
        ("derived_receipt", False, EXPECTED_DERIVED_RECEIPT_SHA256),
        ("source_panel", False, EXPECTED_SOURCE_PANEL_SHA256),
        ("h0_ledgers", True, EXPECTED_H0_LEDGER_SHA256),
        ("support_ledgers", True, EXPECTED_SUPPORT_LEDGER_SHA256),
    )
    resolved_sources: dict[str, Path] = {}
    for key, list_entry, digest in source_specs:
        source_path, _entry = _source_entry(sources, key, list_entry=list_entry, expected_sha256=digest)
        resolved_sources[key] = source_path
        refs.append(_artifact_ref(f"cohort_source:{key}", source_path, digest))

    h0_payload = _read_json(resolved_sources["h0_ledgers"], "native H0 ledger")
    support_payload = _read_json(resolved_sources["support_ledgers"], "support ledger")
    h0_record = _one_ledger_record(h0_payload, label="native H0 ledger")
    support_record = _one_ledger_record(support_payload, label="support ledger")
    for label, record in (("native H0", h0_record), ("support", support_record)):
        _require_exact(record.get("checkpoint"), "S", f"{label} record checkpoint")
        _require_exact(record.get("natural_boundary"), 1, f"{label} record boundary")
        _require_exact(record.get("exact_prefix_sha256"), EXPECTED_PREFIX_SHA256, f"{label} record prefix")
        _require_exact(record.get("covered_owner_ids"), ["gt:5001:2"], f"{label} record covered owners")
        _require_exact(record.get("strict_complete_row"), False, f"{label} record strict row")
    _require_exact(h0_record.get("latest_covered_owner_id"), "gt:5001:2", "native H0 latest owner")
    _require_exact(h0_record.get("history_complete"), True, "native H0 history completeness")
    _require_exact(support_record.get("verified_support"), True, "support record verification")
    _require_exact(support_record.get("intervention"), "none", "support record intervention")

    event_binding = {
        "checkpoint": "S",
        "cohort_eligibility": "eligible_verified_pair",
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "event_id": "gt:5001:15",
        "expected_h0_exact_prefix_sha256": EXPECTED_PREFIX_SHA256,
        "geometry_disposition": "eligible_verified_pair_regions",
        "geometry_status": "available",
        "image_id": 5001,
        "natural_boundary": 1,
        "ordinal": 11,
        "source_panel_object_index": 15,
    }
    h0_binding = {
        "covered_a_owner_id": "gt:5001:2",
        "native_h0_ledger_path": str(resolved_sources["h0_ledgers"]),
        "native_h0_ledger_sha256": EXPECTED_H0_LEDGER_SHA256,
        "native_h0_record_sha256": sha256_json(h0_record),
        "support_ledger_path": str(resolved_sources["support_ledgers"]),
        "support_ledger_sha256": EXPECTED_SUPPORT_LEDGER_SHA256,
        "support_record_sha256": sha256_json(support_record),
        "target_b_owner_id": "gt:5001:15",
        "verified_support": True,
    }
    return event_binding, h0_binding, refs


def _validate_runtime_attestation(identity: Mapping[str, Any], spec: AttemptSpec) -> dict[str, Any]:
    attestation = identity.get("runtime_attestation")
    if not isinstance(attestation, Mapping):
        raise HoldSealError(f"attempt {spec.role} runtime attestation is missing")
    if attestation.get("status") != "validated" or attestation.get("passed") is not True:
        raise HoldSealError(f"attempt {spec.role} runtime attestation is not validated")
    _require_exact(attestation.get("schema_version"), RUNTIME_ATTESTATION_SCHEMA_VERSION, f"attempt {spec.role} attestation schema")
    _require_exact(attestation.get("checkpoint"), "S", f"attempt {spec.role} attestation checkpoint")
    _require_exact(attestation.get("config_fingerprint"), EXPECTED_CONFIG_FINGERPRINT, f"attempt {spec.role} attestation config")
    _require_exact(attestation.get("pid"), spec.pid, f"attempt {spec.role} pid")
    _require_exact(attestation.get("timestamp_utc"), spec.timestamp_utc, f"attempt {spec.role} timestamp")
    _require_exact(attestation.get("physical_device_uuid"), EXPECTED_PHYSICAL_DEVICE_UUID, f"attempt {spec.role} GPU UUID")
    support_runtime = attestation.get("support_runtime_identity")
    if not isinstance(support_runtime, Mapping) or support_runtime.get("status") != "validated" or support_runtime.get("passed") is not True:
        raise HoldSealError(f"attempt {spec.role} support runtime identity is not validated")
    for key in (
        "device",
        "effective_device",
        "normalized_device",
        "torch_current_device",
        "cuda_visible_devices",
        "physical_device_id",
    ):
        _require_exact(support_runtime.get(key), attestation.get(key), f"attempt {spec.role} support runtime {key}")
    return {
        "checkpoint": "S",
        "passed": True,
        "physical_device_uuid": EXPECTED_PHYSICAL_DEVICE_UUID,
        "pid": spec.pid,
        "status": "validated",
        "timestamp_utc": spec.timestamp_utc,
    }


def validate_attempt_directory(
    path: Path,
    *,
    expected_directory: Path,
    spec: AttemptSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved != Path(expected_directory).expanduser().resolve(strict=True):
        raise HoldSealError(f"attempt {spec.role} directory differs from the exact sealed path")
    entries = sorted(resolved.iterdir(), key=lambda item: item.name)
    if len(entries) != 1 or entries[0].name != "runtime_identity.json" or entries[0].is_symlink() or not entries[0].is_file():
        raise HoldSealError(f"attempt {spec.role} must contain exactly runtime_identity.json and no other entry")
    identity_path = entries[0]
    observed_size = identity_path.stat().st_size
    observed_hash = sha256_file(identity_path)
    if observed_hash != spec.runtime_identity_sha256:
        raise HoldSealError(
            f"attempt {spec.role} runtime identity SHA-256 mismatch: "
            f"expected {spec.runtime_identity_sha256}, observed {observed_hash}"
        )
    if observed_size != spec.runtime_identity_size:
        raise HoldSealError(f"attempt {spec.role} runtime identity size mismatch")
    census = [
        {
            "relative_path": "runtime_identity.json",
            "sha256": observed_hash,
            "size_bytes": observed_size,
        }
    ]
    if sha256_json(census) != spec.file_census_sha256:
        raise HoldSealError(f"attempt {spec.role} file census SHA-256 mismatch")

    identity = _read_json(identity_path, f"attempt {spec.role} runtime identity")
    expected_scalars = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": "S",
        "stage": "all",
        "event_count": 1,
        "dry_run": False,
        "cohort_path": str(COHORT_PATH),
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "panel_sha256": EXPECTED_DERIVED_PANEL_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "resolved_config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
    }
    for key, expected in expected_scalars.items():
        _require_exact(identity.get(key), expected, f"attempt {spec.role} runtime {key}")
    helper_schema = identity.get("helper_schema")
    if not isinstance(helper_schema, Mapping):
        raise HoldSealError(f"attempt {spec.role} helper schema is missing")
    _require_exact(helper_schema.get("sha256"), EXPECTED_HELPER_SCHEMA_SHA256, f"attempt {spec.role} helper schema")
    panel_path = Path(str(identity.get("panel_path"))).expanduser().resolve(strict=True)
    if sha256_file(panel_path) != EXPECTED_DERIVED_PANEL_SHA256:
        raise HoldSealError(f"attempt {spec.role} panel bytes differ from frozen identity")
    config_path = Path(str(identity.get("config_path"))).expanduser().resolve(strict=True)
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise HoldSealError(f"attempt {spec.role} config bytes differ from frozen identity")
    h0 = identity.get("h0")
    if not isinstance(h0, Mapping):
        raise HoldSealError(f"attempt {spec.role} H0 identity is missing")
    h0_root = Path(str(h0.get("root"))).expanduser().resolve(strict=True)
    h0_refs: list[dict[str, Any]] = []
    for relative, key in (
        ("summary.json", "summary_sha256"),
        ("run_manifest.json", "run_manifest_sha256"),
        ("configs/resolved.json", "resolved_config_sha256"),
    ):
        artifact = h0_root / relative
        digest = _require_sha256(h0.get(key), f"attempt {spec.role} H0 {key}")
        h0_refs.append(_artifact_ref(f"attempt:{spec.role}:h0:{relative}", artifact, digest))
    attestation_summary = _validate_runtime_attestation(identity, spec)
    attempt_document = {
        "cause": {
            "code": spec.cause_code,
            "evidence_level": "lead_observed_unattested",
            "summary": spec.cause_summary,
            "verbatim_stderr": None,
        },
        "directory": str(resolved),
        "file_census": census,
        "file_census_sha256": spec.file_census_sha256,
        "persisted_failure_log": {"path": None, "sha256": None, "status": "unavailable"},
        "role": spec.role,
        "runtime_attestation": attestation_summary,
        "runtime_identity_path": str(identity_path),
        "runtime_identity_sha256": observed_hash,
        "runtime_identity_size_bytes": observed_size,
    }
    refs = [
        {
            "role": f"attempt:{spec.role}:runtime_identity",
            "path": str(identity_path),
            "sha256": observed_hash,
            "size_bytes": observed_size,
        },
        _artifact_ref(f"attempt:{spec.role}:config", config_path, EXPECTED_CONFIG_SHA256),
        _artifact_ref(f"attempt:{spec.role}:panel", panel_path, EXPECTED_DERIVED_PANEL_SHA256),
        *h0_refs,
    ]
    return attempt_document, refs


def invalid_cell_disposition() -> dict[str, Any]:
    return {
        "execution_status": "not_sealed",
        "metrics": None,
        "model_output": None,
        "reason_code": "pre_actuator_technical_failure_repair_exhausted",
        "scientific_observation": None,
        "status": "invalid/uninterpretable",
    }


def _matrix_dispositions() -> dict[str, Any]:
    stage_cells = {"p1": P1_CELLS, "p2": P2_CELLS, "p3": P3_CELLS, "p4": P4_CELLS}
    return {
        stage: {
            "cells": {cell: invalid_cell_disposition() for cell in cells},
            "origin": "administrative_disposition_not_model_output",
            "status": "invalid/uninterpretable",
        }
        for stage, cells in stage_cells.items()
    }


def _expected_classification() -> dict[str, Any]:
    return {
        "actuators_called": None,
        "complete_non_scored": False,
        "matrix_status": "eligible_pre_actuator_hold",
        "scored": False,
    }


def _expected_execution_evidence() -> dict[str, Any]:
    return {
        "actual_actuator_invocation_count": None,
        "actual_model_forward_count": None,
        "lead_observation_evidence_level": "unattested",
        "lead_observed_pre_actuator": True,
        "persisted_artifacts_absent": list(ABSENT_ATTEMPT_ARTIFACTS),
        "persisted_event_result_count": 0,
        "persisted_failure_log": {"path": None, "sha256": None, "status": "unavailable"},
        "persisted_gradient_receipt_count": 0,
        "persisted_intervention_receipt_count": 0,
        "persisted_terminal_count": 0,
        "receipt_bearing_actuator_cell_count": 0,
        "receipt_bearing_scientific_cell_count": 0,
    }


def validate_leaf_document(leaf: Mapping[str, Any]) -> None:
    required_keys = {
        "schema_version",
        "unit_id",
        "status",
        "kind",
        "receipt_path",
        "frozen_contract",
        "event_binding",
        "h0_binding",
        "classification",
        "repair_policy",
        "execution_evidence",
        "attempt_lineage",
        "matrix_dispositions",
        "self_sha256",
    }
    if set(leaf) != required_keys:
        raise HoldSealError("leaf top-level schema is not exact")
    _require_exact(leaf.get("schema_version"), SCHEMA_VERSION, "leaf schema")
    _require_exact(leaf.get("unit_id"), UNIT_ID, "leaf unit_id")
    _require_exact(leaf.get("status"), "sealed", "leaf status")
    _require_exact(leaf.get("kind"), "eligible_pre_actuator_technical_hold", "leaf kind")
    if leaf.get("classification") != _expected_classification():
        raise HoldSealError("leaf classification confuses eligible HOLD with scored or complete_non_scored evidence")
    event_binding = leaf.get("event_binding")
    if not isinstance(event_binding, Mapping) or event_binding.get("cohort_eligibility") != "eligible_verified_pair":
        raise HoldSealError("leaf event binding is not the eligible verified S pair")
    expected_event_fields = {
        "checkpoint": "S",
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "event_id": "gt:5001:15",
        "expected_h0_exact_prefix_sha256": EXPECTED_PREFIX_SHA256,
        "geometry_disposition": "eligible_verified_pair_regions",
        "geometry_status": "available",
        "image_id": 5001,
        "natural_boundary": 1,
        "ordinal": 11,
        "source_panel_object_index": 15,
    }
    for key, value in expected_event_fields.items():
        _require_exact(event_binding.get(key), value, f"leaf event binding {key}")
    if leaf.get("execution_evidence") != _expected_execution_evidence():
        raise HoldSealError("leaf actual execution counts or persisted-evidence boundary are not exact")
    attempts = leaf.get("attempt_lineage")
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise HoldSealError("leaf attempt lineage must contain exactly initial and repair1")
    for attempt, spec in zip(attempts, (INITIAL_ATTEMPT, REPAIR_ATTEMPT), strict=True):
        if not isinstance(attempt, Mapping) or attempt.get("role") != spec.role:
            raise HoldSealError("leaf attempt lineage order is not initial then repair1")
        _require_exact(attempt.get("runtime_identity_sha256"), spec.runtime_identity_sha256, f"leaf attempt {spec.role} hash")
        _require_exact(attempt.get("file_census_sha256"), spec.file_census_sha256, f"leaf attempt {spec.role} census")
        cause = attempt.get("cause")
        if not isinstance(cause, Mapping) or cause.get("evidence_level") != "lead_observed_unattested" or cause.get("verbatim_stderr") is not None:
            raise HoldSealError(f"leaf attempt {spec.role} cause is not explicitly unattested")
        if attempt.get("persisted_failure_log") != {"path": None, "sha256": None, "status": "unavailable"}:
            raise HoldSealError(f"leaf attempt {spec.role} invents a persisted failure log")
    repair_policy = leaf.get("repair_policy")
    expected_repair_policy = {
        "attempt_count": 2,
        "attempt_roles": ["initial", "repair1"],
        "exhausted": True,
        "repair_count": 1,
        "rule": "one_exact_repair_then_invalid_uninterpretable",
        "unit_sha256": EXPECTED_UNIT_SHA256,
    }
    if repair_policy != expected_repair_policy:
        raise HoldSealError("leaf repair exhaustion contract is not exact")
    matrix = leaf.get("matrix_dispositions")
    expected_cells = {"p1": P1_CELLS, "p2": P2_CELLS, "p3": P3_CELLS, "p4": P4_CELLS}
    if not isinstance(matrix, Mapping) or set(matrix) != set(expected_cells):
        raise HoldSealError("leaf administrative matrix stages are not exact")
    expected_cell = invalid_cell_disposition()
    for stage, cell_ids in expected_cells.items():
        stage_record = matrix.get(stage)
        if not isinstance(stage_record, Mapping):
            raise HoldSealError(f"leaf administrative matrix stage {stage} is missing")
        if stage_record.get("status") != "invalid/uninterpretable" or stage_record.get("origin") != "administrative_disposition_not_model_output":
            raise HoldSealError(f"leaf administrative matrix stage {stage} is not a HOLD disposition")
        cells = stage_record.get("cells")
        if not isinstance(cells, Mapping) or set(cells) != set(cell_ids):
            raise HoldSealError(f"leaf administrative matrix stage {stage} cell IDs are not exact")
        if any(cell != expected_cell for cell in cells.values()):
            raise HoldSealError(f"leaf administrative cell {stage} carries a scientific observation or non-HOLD field")
    observed_self = _require_sha256(leaf.get("self_sha256"), "leaf self_sha256")
    if observed_self != document_self_sha256(leaf):
        raise HoldSealError("leaf self_sha256 mismatch")


def build_leaf_document(*, receipt_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unit_ref = _artifact_ref("frozen_unit", UNIT_PATH, EXPECTED_UNIT_SHA256)
    tasks_ref = _artifact_ref("frozen_tasks", TASKS_PATH, EXPECTED_TASKS_SHA256)
    event_binding, h0_binding, source_refs = validate_production_binding()
    initial, initial_refs = validate_attempt_directory(
        INITIAL_ATTEMPT_DIR,
        expected_directory=INITIAL_ATTEMPT_DIR,
        spec=INITIAL_ATTEMPT,
    )
    repair, repair_refs = validate_attempt_directory(
        REPAIR_ATTEMPT_DIR,
        expected_directory=REPAIR_ATTEMPT_DIR,
        spec=REPAIR_ATTEMPT,
    )
    if initial["runtime_attestation"]["timestamp_utc"] >= repair["runtime_attestation"]["timestamp_utc"]:
        raise HoldSealError("repair1 timestamp must be strictly later than the initial attempt")
    leaf: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "sealed",
        "kind": "eligible_pre_actuator_technical_hold",
        "receipt_path": str(Path(receipt_path).expanduser().resolve()),
        "frozen_contract": {
            "tasks": {key: value for key, value in tasks_ref.items() if key != "role"},
            "unit": {key: value for key, value in unit_ref.items() if key != "role"},
        },
        "event_binding": event_binding,
        "h0_binding": h0_binding,
        "classification": _expected_classification(),
        "repair_policy": {
            "attempt_count": 2,
            "attempt_roles": ["initial", "repair1"],
            "exhausted": True,
            "repair_count": 1,
            "rule": "one_exact_repair_then_invalid_uninterpretable",
            "unit_sha256": EXPECTED_UNIT_SHA256,
        },
        "execution_evidence": _expected_execution_evidence(),
        "attempt_lineage": [initial, repair],
        "matrix_dispositions": _matrix_dispositions(),
    }
    leaf["self_sha256"] = document_self_sha256(leaf)
    validate_leaf_document(leaf)
    refs = [unit_ref, tasks_ref, *source_refs, *initial_refs, *repair_refs]
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for ref in refs:
        unique[(str(ref["role"]), str(ref["path"]))] = ref
    return leaf, [unique[key] for key in sorted(unique)]


def _leaf_bytes(leaf: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(leaf) + b"\n"


def validate_receipt_document(receipt: Mapping[str, Any], *, leaf_path: Path, leaf: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "unit_id",
        "kind",
        "leaf_path",
        "leaf_sha256",
        "leaf_self_sha256",
        "input_set_sha256",
        "inputs",
        "self_sha256",
    }
    if set(receipt) != required:
        raise HoldSealError("receipt top-level schema is not exact")
    _require_exact(receipt.get("schema_version"), RECEIPT_SCHEMA_VERSION, "receipt schema")
    _require_exact(receipt.get("unit_id"), UNIT_ID, "receipt unit_id")
    _require_exact(receipt.get("kind"), "eligible_pre_actuator_technical_hold", "receipt kind")
    resolved_leaf = Path(leaf_path).expanduser().resolve(strict=True)
    _require_exact(receipt.get("leaf_path"), str(resolved_leaf), "receipt leaf path")
    _require_exact(receipt.get("leaf_sha256"), sha256_file(resolved_leaf), "receipt leaf SHA-256")
    _require_exact(receipt.get("leaf_self_sha256"), leaf.get("self_sha256"), "receipt leaf self hash")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise HoldSealError("receipt inputs must be a non-empty array")
    if receipt.get("input_set_sha256") != sha256_json(inputs):
        raise HoldSealError("receipt input_set_sha256 mismatch")
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping):
            raise HoldSealError(f"receipt input {index} is not an object")
        path = Path(str(item.get("path"))).expanduser().resolve(strict=True)
        _require_exact(item.get("sha256"), sha256_file(path), f"receipt input {index} SHA-256")
        _require_exact(item.get("size_bytes"), path.stat().st_size, f"receipt input {index} size")
    observed_self = _require_sha256(receipt.get("self_sha256"), "receipt self_sha256")
    if observed_self != document_self_sha256(receipt):
        raise HoldSealError("receipt self_sha256 mismatch")


def _write_exclusive(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"output collision: refusing to overwrite {path}") from exc


def seal(output_dir: Path) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output collision: refusing to use non-empty directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    leaf_path = output / LEAF_FILENAME
    receipt_path = output / RECEIPT_FILENAME
    leaf, inputs = build_leaf_document(receipt_path=receipt_path)
    leaf_content = _leaf_bytes(leaf)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "kind": "eligible_pre_actuator_technical_hold",
        "leaf_path": str(leaf_path),
        "leaf_sha256": sha256_bytes(leaf_content),
        "leaf_self_sha256": leaf["self_sha256"],
        "input_set_sha256": sha256_json(inputs),
        "inputs": inputs,
    }
    receipt["self_sha256"] = document_self_sha256(receipt)
    receipt_content = canonical_json_bytes(receipt) + b"\n"
    _write_exclusive(leaf_path, leaf_content)
    _write_exclusive(receipt_path, receipt_content)
    persisted_leaf = _read_json(leaf_path, "persisted eligible HOLD leaf")
    persisted_receipt = _read_json(receipt_path, "persisted eligible HOLD receipt")
    validate_leaf_document(persisted_leaf)
    validate_receipt_document(persisted_receipt, leaf_path=leaf_path, leaf=persisted_leaf)
    return {
        "status": "sealed",
        "kind": "eligible_pre_actuator_technical_hold",
        "leaf_path": str(leaf_path),
        "leaf_sha256": sha256_file(leaf_path),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = seal(args.output_dir)
    except (HoldSealError, FileExistsError, OSError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

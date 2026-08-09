#!/usr/bin/env python3
"""Finalize the sealed S step-2444 K/N/H cohort without model work.

This CPU-only reducer accepts only the canonical manifest/census/plan/gate and
the completed eight-shard merge.  It reuses the production validators before
interpreting any row, keeps scientific unmatched/duplicate/malformed/STOP
outcomes separate from technical invalidity, and emits only conditional
follow-up flags.  It never authorizes crossover, A3, P4, or training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.merge_s_natural_boundary_k_n_h_shards import (  # noqa: E402
    MERGED_AGGREGATE_SCHEMA_VERSION,
    MERGED_RECEIPT_SCHEMA_VERSION,
    ShardMergeError,
    _canonical_file,
    _validate_hash_document,
    _validate_shards_root,
    validate_shard,
)
from scripts.research.plan_s_natural_boundary_k_n_h_execution import (  # noqa: E402
    SHARD_COUNT,
    ExecutionPlanError,
    _shard_id,
    _validate_gate_receipt,
    validate_plan,
)
from scripts.research.run_s_natural_boundary_k_n_h_cohort import (  # noqa: E402
    ARM_ORDER,
    CHECKPOINT,
    EVENT_SCHEMA_VERSION,
    STEP,
    SUBSTRATE,
    UNIT_ID,
    CohortContractError,
    _load_completed_event,
    _validate_execution_qualification,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    validate_arm_result,
    validate_manifest as validate_cohort_manifest,
)
from scripts.research.run_s_natural_boundary_k_n_h_shard import (  # noqa: E402
    _event_root_name,
    _validate_event_plan_binding,
)


PRIMARY = {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE}
SCHEMA_VERSION = "s_natural_boundary_k_n_h_evidence.v3"
RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.receipt"
CENSUS_SCHEMA_VERSION = "natural_boundary_owner_admission_census.v3"
K13_MASK_RECEIPT_SCHEMA_VERSION = "natural_boundary_attention_actuators.v1.mask_receipt.v1"
K13_LAYER_RECEIPT_SCHEMA_VERSION = "natural_boundary_attention_actuators.v1.layer_consumption.v1"
K13_NOT_APPLICABLE_REASON = "same-class competitor keys are unavailable in the current scalar prefix"
OWN_BASELINE = {
    "K10": "K01",
    "K11": "K01",
    "K12": "K01",
    "K13": "K01",
    "K14T": "K01",
    "K14B": "K01",
    "N10": "N01",
    "N20": "N01",
    "H10": "H00",
    "H20": "H00",
}
DYNAMIC_INTERVENTIONS = ("N10", "N20", "H10", "H20")
PARSE_KEYS = (
    "valid_rows",
    "duplicate_rows",
    "unmatched_rows",
    "ambiguous_rows",
    "malformed_rows",
    "invalid_rows",
)
SCIENTIFIC_TERMINALS = {"closure", "native_stop", "invalid", "over_continuation", "max_budget"}

# ``_delta`` is a small, sealed contrast contract.  Keep every emitted key in
# exactly one partition so a future field cannot silently become a dynamic
# endpoint discriminator (or silently be ignored by the disposition).
_DELTA_ENDPOINT_BOOLEAN_KEYS = (
    "first_token_class_changed",
    "terminal_reason_changed",
)
_DELTA_ENDPOINT_NUMERIC_KEYS = (
    "strict_physical_owner_match_count_delta",
    "unmatched_rows_delta",
    "ambiguous_rows_delta",
    "duplicate_rows_delta",
    "malformed_rows_delta",
    "invalid_rows_delta",
    "target_strict_release_delta",
)
_DELTA_ENDPOINT_IDENTITY_KEYS = (
    "strict_owner_ids_added",
    "strict_owner_ids_removed",
)
_DELTA_DERIVED_KEYS = (
    "complete_natural_rows_delta",
    "covered_repeat_owner_ids_added",
    "covered_repeat_owner_ids_removed",
    "uncovered_owner_gain_ids_added",
    "uncovered_owner_gain_ids_removed",
)
_DELTA_METADATA_KEYS = (
    "intervention_arm",
    "baseline_arm",
    "status",
    "unmatched_is_scientific_not_technical",
)
_DELTA_PARTITIONS = {
    "endpoint_boolean": _DELTA_ENDPOINT_BOOLEAN_KEYS,
    "endpoint_numeric": _DELTA_ENDPOINT_NUMERIC_KEYS,
    "endpoint_identity": _DELTA_ENDPOINT_IDENTITY_KEYS,
    "derived": _DELTA_DERIVED_KEYS,
    "metadata": _DELTA_METADATA_KEYS,
}
_DELTA_ALL_KEYS = frozenset(key for keys in _DELTA_PARTITIONS.values() for key in keys)


class EvidenceAnalysisError(ValueError):
    """Raised before evidence emission when a formal input is uninterpretable."""


def document_self_sha256(document: Mapping[str, Any], field: str = "self_sha256") -> str:
    body = dict(document)
    body.pop(field, None)
    return sha256_json(body)


def sha256_file(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise EvidenceAnalysisError(f"hash source is not a regular non-symlink file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_path(source: str | Path, label: str) -> Path:
    raw = Path(source).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise EvidenceAnalysisError(f"{label} is not a regular non-symlink file: {raw}")
    return raw.resolve(strict=True)


def _strict_json_document(
    source: str | Path,
    label: str,
    *,
    require_canonical: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one regular JSON object and validate its semantic JSON domain.

    Most sealed inputs are immutable canonical documents.  Gate-v3 result and
    runtime files are the exception: their raw bytes are bound by the plan,
    while the planner authenticates their canonical semantic bodies.
    """

    path = _strict_path(source, label)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceAnalysisError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise EvidenceAnalysisError(f"{label} must be a JSON object")
    document = dict(value)
    try:
        canonical = canonical_json_bytes(document)
    except (CohortContractError, TypeError, ValueError) as exc:
        raise EvidenceAnalysisError(f"{label} is not finite canonical JSON") from exc
    if require_canonical and raw != canonical + b"\n":
        raise EvidenceAnalysisError(f"{label} is not canonical JSON with one trailing newline")
    return document, {"path": str(path), "raw_file_sha256": sha256_bytes(raw), "size_bytes": len(raw)}


def _strict_canonical_document(source: str | Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return _strict_json_document(source, label, require_canonical=True)


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise EvidenceAnalysisError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceAnalysisError(f"{label} must be a non-negative integer")
    return int(value)


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise EvidenceAnalysisError(f"{label} must be a JSON boolean")
    return value


def _require_id_list(value: Any, label: str, *, sorted_unique: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise EvidenceAnalysisError(f"{label} must be an owner-ID array")
    result = list(value)
    if len(set(result)) != len(result):
        raise EvidenceAnalysisError(f"{label} contains duplicate owner IDs")
    if sorted_unique and result != sorted(result):
        raise EvidenceAnalysisError(f"{label} must be sorted")
    return result


def validate_census(
    source: str | Path,
    *,
    manifest_info: Mapping[str, Any],
) -> dict[str, Any]:
    census, source_info = _strict_canonical_document(source, "census-v3")
    if (
        census.get("schema_version") != CENSUS_SCHEMA_VERSION
        or census.get("status") != "sealed"
        or census.get("unit_id") != UNIT_ID
        or census.get("census_revision") not in {None, "census-v3"}
    ):
        raise EvidenceAnalysisError("census is not the sealed census-v3 for this unit")
    self_sha = _require_sha(census.get("self_sha256"), "census.self_sha256")
    if self_sha != document_self_sha256(census):
        raise EvidenceAnalysisError("census self_sha256 mismatch")
    rows = census.get("rows")
    if not isinstance(rows, list) or len(rows) != 784 or any(not isinstance(row, Mapping) for row in rows):
        raise EvidenceAnalysisError("census-v3 must contain exactly 784 owner rows")
    normalized = [dict(row) for row in rows]
    checkpoint_counts = Counter(row.get("checkpoint") for row in normalized)
    if checkpoint_counts != Counter({"S": 392, "A": 392}):
        raise EvidenceAnalysisError(f"census-v3 checkpoint denominator must be S=392/A=392, observed {dict(checkpoint_counts)}")
    keys = [(row.get("checkpoint"), row.get("gt_owner_id")) for row in normalized]
    if len(keys) != len(set(keys)) or any(not isinstance(owner_id, str) or not owner_id for _, owner_id in keys):
        raise EvidenceAnalysisError("census-v3 checkpoint/owner keys are malformed or duplicated")

    manifest_census = manifest_info["document"].get("source_census")
    if not isinstance(manifest_census, Mapping):
        raise EvidenceAnalysisError("manifest source_census binding is missing")
    if (
        manifest_census.get("revision") != "census-v3"
        or manifest_census.get("hash_semantics") != "canonical_json_document_with_trailing_newline"
        or manifest_census.get("sha256") != source_info["raw_file_sha256"]
    ):
        raise EvidenceAnalysisError("manifest source_census raw-byte binding differs from supplied census-v3")
    declared_path = manifest_census.get("path")
    if not isinstance(declared_path, str) or _strict_path(declared_path, "manifest source_census.path") != Path(source_info["path"]):
        raise EvidenceAnalysisError("manifest source_census path differs from supplied census-v3")

    by_key = {(str(row["checkpoint"]), str(row["gt_owner_id"])): row for row in normalized}
    for event in manifest_info["events"]:
        owner_id = event["owner_refs"]["gt_owner_id"]
        row = by_key.get(("S", owner_id))
        if row is None:
            raise EvidenceAnalysisError(f"manifest event owner is absent from S census: {owner_id}")
        if row.get("image_id") != event.get("image_id") or event.get("checkpoint") != "S":
            raise EvidenceAnalysisError(f"manifest event owner/image/checkpoint differs from census: {owner_id}")
        for field in ("source_panel_object_index", "derived_panel_object_index"):
            event_value = event["owner_refs"].get(field)
            row_value = row.get(field)
            if isinstance(event_value, bool) or not isinstance(event_value, int) or event_value < 0 or event_value != row_value:
                raise EvidenceAnalysisError(f"manifest event {owner_id} {field} membership differs from census")
        covered = event["owner_refs"].get("covered_owner_ids")
        if not isinstance(covered, list) or not covered:
            raise EvidenceAnalysisError(f"manifest event {owner_id} lacks covered-owner membership")
        for covered_owner in covered:
            covered_row = by_key.get(("S", covered_owner))
            if covered_row is None or covered_row.get("image_id") != event.get("image_id"):
                raise EvidenceAnalysisError(f"event {owner_id} covered owner is foreign to the S image: {covered_owner}")
    return {
        "document": census,
        "source": source_info,
        "raw_file_sha256": source_info["raw_file_sha256"],
        "semantic_self_sha256": self_sha,
        "rows": normalized,
        "by_key": by_key,
        "checkpoint_counts": dict(checkpoint_counts),
    }


def _validate_gate_binding(gate_receipt: str | Path, plan_info: Mapping[str, Any]) -> dict[str, Any]:
    gate_document, gate_source = _strict_json_document(gate_receipt, "gate result", require_canonical=False)
    runtime_path = Path(gate_source["path"]).with_name("runtime_identity.json")
    runtime_document, runtime_source = _strict_json_document(
        runtime_path,
        "gate runtime identity",
        require_canonical=False,
    )
    try:
        gate_info = _validate_gate_receipt(gate_source["path"])
    except (ExecutionPlanError, OSError, ValueError) as exc:
        raise EvidenceAnalysisError(f"gate validation failed: {exc}") from exc
    plan = plan_info["document"]
    expected = {
        "gate_sha256": gate_info["gate_sha256"],
        "gate_result_sha256": gate_info["gate_result_sha256"],
        "gate_runtime_identity_sha256": gate_info["gate_runtime_identity_sha256"],
        "gate_runtime_identity_raw_sha256": gate_info["gate_runtime_identity_raw_sha256"],
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise EvidenceAnalysisError("execution plan gate raw/self binding differs from supplied gate")
    if gate_info["gate_sha256"] != gate_source["raw_file_sha256"]:
        raise EvidenceAnalysisError("gate raw hash validation is internally inconsistent")
    if runtime_document.get("identity_sha256") != gate_info["gate_runtime_identity_sha256"]:
        raise EvidenceAnalysisError("gate runtime semantic self hash differs from validated gate binding")
    if runtime_source["raw_file_sha256"] != gate_info["gate_runtime_identity_raw_sha256"]:
        raise EvidenceAnalysisError("gate runtime raw hash differs from validated gate binding")
    return {
        "document": gate_document,
        "raw_file_sha256": gate_source["raw_file_sha256"],
        "semantic_self_sha256": gate_info["gate_result_sha256"],
        "runtime_identity": {
            "raw_file_sha256": runtime_source["raw_file_sha256"],
            "semantic_self_sha256": runtime_document["identity_sha256"],
        },
    }


def _validate_merged_set(
    aggregate_path: str | Path,
    *,
    shards_root: str | Path,
    manifest_info: Mapping[str, Any],
    plan_info: Mapping[str, Any],
) -> dict[str, Any]:
    claim_scope = _validate_execution_qualification(manifest_info)
    if plan_info["document"].get("claim_scope") != claim_scope:
        raise EvidenceAnalysisError("execution plan claim_scope differs from manifest-derived scope")
    aggregate_source = _strict_path(aggregate_path, "merged aggregate")
    receipt_source = aggregate_source.with_name("aggregate.receipt.json")
    try:
        aggregate, aggregate_raw_sha = _canonical_file(aggregate_source, label="merged.aggregate")
        receipt, receipt_raw_sha = _canonical_file(receipt_source, label="merged.receipt")
        aggregate_semantic_sha = _validate_hash_document(aggregate, "aggregate_sha256", "merged.aggregate")
        receipt_semantic_sha = _validate_hash_document(receipt, "receipt_sha256", "merged.receipt")
    except (ShardMergeError, OSError, ValueError) as exc:
        raise EvidenceAnalysisError(f"merged aggregate validation failed: {exc}") from exc
    root_source = Path(shards_root).expanduser()
    if root_source.is_symlink():
        raise EvidenceAnalysisError(f"shards root is a symlink: {root_source}")
    root = root_source.resolve(strict=True)
    try:
        _validate_shards_root(root)
        shard_infos = [
            validate_shard(
                root / _shard_id(index),
                shard_id=index,
                manifest_info=manifest_info,
                plan_info=plan_info,
            )
            for index in range(SHARD_COUNT)
        ]
    except (ShardMergeError, CohortContractError, OSError, ValueError) as exc:
        raise EvidenceAnalysisError(f"eight-shard validation failed: {exc}") from exc
    all_events = sorted(
        [event for shard_info in shard_infos for event in shard_info["events"]],
        key=lambda event: event["event_index"],
    )
    scalar_total = sum(info["scalar_forward_count"] for info in shard_infos)
    runtime_total = sum(info["runtime_scalar_forward_count"] for info in shard_infos)
    expected_qualification = claim_scope
    expected_shards = [
        {
            "shard_id": info["shard_id"],
            "shard_index": info["shard_index"],
            "aggregate_sha256": info["aggregate_sha256"],
            "receipt_sha256": info["receipt_sha256"],
            "event_count": len(info["events"]),
            "scalar_forward_count": info["scalar_forward_count"],
            "runtime_scalar_forward_count": info["runtime_scalar_forward_count"],
        }
        for info in shard_infos
    ]
    identity_checks = {
        "schema_version": MERGED_AGGREGATE_SCHEMA_VERSION,
        "status": "completed",
        "unit_id": UNIT_ID,
        "primary": PRIMARY,
        "claim_scope": claim_scope,
        "plan_sha256": plan_info["plan_sha256"],
        "manifest_self_sha256": manifest_info["manifest_self_sha256"],
        "manifest_sha256": manifest_info["manifest_sha256"],
        "gate_sha256": plan_info["document"]["gate_sha256"],
        "gate_result_sha256": plan_info["document"]["gate_result_sha256"],
        "gate_runtime_identity_sha256": plan_info["document"].get("gate_runtime_identity_sha256"),
        "gate_runtime_identity_raw_sha256": plan_info["document"].get("gate_runtime_identity_raw_sha256"),
        "arm_order": list(ARM_ORDER),
        "shard_count": SHARD_COUNT,
        "shards": expected_shards,
        "event_count": len(all_events),
        "distinct_image_count": len({event["image_id"] for event in all_events}),
        "scalar_forward_count": scalar_total,
        "runtime_scalar_forward_count": runtime_total,
        "execution_qualification": expected_qualification,
        "events": all_events,
    }
    for key, expected in identity_checks.items():
        if aggregate.get(key) != expected:
            raise EvidenceAnalysisError(f"merged aggregate field differs from validated shards: {key}")
    for flag in ("no_event_reorder", "no_outcome_adaptive_selection", "no_sweep", "no_a3", "no_2x2", "no_p4"):
        if aggregate.get(flag) is not True:
            raise EvidenceAnalysisError(f"merged aggregate flag is not sealed: {flag}")
    if receipt.get("schema_version") != MERGED_RECEIPT_SCHEMA_VERSION or receipt.get("status") != "completed":
        raise EvidenceAnalysisError("merged receipt schema/status differs from completed merge")
    receipt_checks = {
        "unit_id": UNIT_ID,
        "plan_sha256": plan_info["plan_sha256"],
        "manifest_sha256": manifest_info["manifest_sha256"],
        "gate_sha256": plan_info["document"]["gate_sha256"],
        "gate_result_sha256": plan_info["document"]["gate_result_sha256"],
        "gate_runtime_identity_sha256": plan_info["document"].get("gate_runtime_identity_sha256"),
        "gate_runtime_identity_raw_sha256": plan_info["document"].get("gate_runtime_identity_raw_sha256"),
        "aggregate_sha256": aggregate_semantic_sha,
        "shard_count": SHARD_COUNT,
        "event_count": len(all_events),
        "distinct_image_count": len({event["image_id"] for event in all_events}),
        "scalar_forward_count": scalar_total,
        "runtime_scalar_forward_count": runtime_total,
        "execution_qualification": expected_qualification,
        "claim_scope": claim_scope,
        "shards": expected_shards,
        "event_roots": all_events,
    }
    for key, expected in receipt_checks.items():
        if receipt.get(key) != expected:
            raise EvidenceAnalysisError(f"merged receipt field differs from validated shards: {key}")
    for info in shard_infos:
        if (
            info["aggregate"].get("claim_scope") != claim_scope
            or info["receipt"].get("claim_scope") != claim_scope
        ):
            raise EvidenceAnalysisError(
                f"{info['shard_id']} claim_scope differs from manifest-derived scope"
            )
    shard_bindings = []
    for info in shard_infos:
        shard_root = root / info["shard_id"]
        shard_bindings.append({
            "shard_id": info["shard_id"],
            "aggregate_raw_file_sha256": sha256_file(shard_root / "aggregate.json"),
            "aggregate_semantic_self_sha256": info["aggregate_sha256"],
            "receipt_raw_file_sha256": sha256_file(shard_root / "aggregate.receipt.json"),
            "receipt_semantic_self_sha256": info["receipt_sha256"],
        })
    return {
        "aggregate": aggregate,
        "receipt": receipt,
        "events": all_events,
        "shard_infos": shard_infos,
        "claim_scope": claim_scope,
        "shards_root": root,
        "bindings": {
            "aggregate": {
                "raw_file_sha256": aggregate_raw_sha,
                "semantic_self_sha256": aggregate_semantic_sha,
            },
            "receipt": {
                "raw_file_sha256": receipt_raw_sha,
                "semantic_self_sha256": receipt_semantic_sha,
            },
            "shards": shard_bindings,
        },
    }


def _strict_owner_match(row: Mapping[str, Any]) -> tuple[str | None, bool, str | None]:
    match = row.get("owner_match")
    if not isinstance(match, Mapping):
        return None, False, None
    status = match.get("status") if isinstance(match.get("status"), str) else None
    owner_id = match.get("owner_id") if isinstance(match.get("owner_id"), str) and match.get("owner_id") else None
    strict = bool(
        row.get("status") == "closure"
        and status in {"unique", "matched"}
        and match.get("physical_match") is True
        and match.get("source_specific") is True
        and owner_id is not None
    )
    return owner_id, strict, status


def _validate_natural_receipts(arm: Mapping[str, Any], arm_id: str) -> dict[str, Any]:
    if arm.get("arm_id") != arm_id:
        raise EvidenceAnalysisError(f"arm_id mismatch: expected {arm_id}, observed {arm.get('arm_id')}")
    if arm.get("admission_mode") != "pre_opener_natural":
        raise EvidenceAnalysisError(f"{arm_id} is not a pre-opener natural arm")
    if arm.get("opener_injected") is not False or arm.get("synthetic_opener_injections") != 0:
        raise EvidenceAnalysisError(f"{arm_id} injected an opener")
    if (
        arm.get("opener_seeded") is True
        or arm.get("seed_provenance") not in (None, False, "")
        or arm.get("opener_seed_provenance") not in (None, False, "")
    ):
        raise EvidenceAnalysisError(f"{arm_id} carries seeded-opener provenance")
    opener = _require_nonnegative_int(arm.get("opener_token_id"), f"{arm_id}.opener_token_id")
    initial_last = _require_nonnegative_int(arm.get("initial_prefix_last_token_id"), f"{arm_id}.initial_prefix_last_token_id")
    if initial_last == opener:
        raise EvidenceAnalysisError(f"{arm_id} natural prefix already ends with object_ref_start")
    first = _require_nonnegative_int(arm.get("first_generated_token_id"), f"{arm_id}.first_generated_token_id")
    opener_generated = _require_bool(arm.get("opener_generated_by_model"), f"{arm_id}.opener_generated_by_model")
    if opener_generated != (first == opener):
        raise EvidenceAnalysisError(f"{arm_id} opener_generated_by_model disagrees with first token")
    native_stop_ids = arm.get("native_stop_token_ids")
    if not isinstance(native_stop_ids, list) or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in native_stop_ids):
        raise EvidenceAnalysisError(f"{arm_id}.native_stop_token_ids is malformed")
    if len(set(native_stop_ids)) != len(native_stop_ids):
        raise EvidenceAnalysisError(f"{arm_id}.native_stop_token_ids contains duplicates")
    terminal = arm.get("terminal_reason")
    if terminal not in SCIENTIFIC_TERMINALS:
        raise EvidenceAnalysisError(f"{arm_id}.terminal_reason is unknown")
    if terminal == "native_stop" and first not in native_stop_ids:
        raise EvidenceAnalysisError(f"{arm_id} native STOP first token is not a native terminal")
    if first in native_stop_ids and terminal != "native_stop":
        raise EvidenceAnalysisError(f"{arm_id} native terminal token disagrees with terminal_reason")
    first_class = "object_ref_start" if first == opener else ("native_stop" if first in native_stop_ids else "other_invalid")
    rows = arm.get("rows")
    if not isinstance(rows, list) or not rows:
        raise EvidenceAnalysisError(f"{arm_id}.rows is missing")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("row_index") != index:
            raise EvidenceAnalysisError(f"{arm_id}.rows[{index}] identity is malformed")
        row_synthetic_injections = row.get("synthetic_opener_injections", 0)
        if (
            row.get("admission_mode") != "pre_opener_natural"
            or row.get("opener_injected") is not False
            or isinstance(row_synthetic_injections, bool)
            or not isinstance(row_synthetic_injections, int)
            or row_synthetic_injections != 0
        ):
            raise EvidenceAnalysisError(f"{arm_id}.rows[{index}] is not exact natural pre-opener release")
        if (
            row.get("opener_seeded") is True
            or row.get("seed_provenance") not in (None, False, "")
            or row.get("opener_seed_provenance") not in (None, False, "")
        ):
            raise EvidenceAnalysisError(f"{arm_id}.rows[{index}] carries seeded-opener provenance")
        if row.get("opener_token_id") != opener:
            raise EvidenceAnalysisError(f"{arm_id}.rows[{index}] opener identity differs from arm")
        row_last = _require_nonnegative_int(row.get("initial_prefix_last_token_id"), f"{arm_id}.rows[{index}].initial_prefix_last_token_id")
        if row_last == opener:
            raise EvidenceAnalysisError(f"{arm_id}.rows[{index}] prefix already ends with opener")
        tokens = row.get("token_ids")
        if not isinstance(tokens, list) or not tokens:
            raise EvidenceAnalysisError(f"{arm_id}.rows[{index}].token_ids is empty or malformed")
        row_first = _require_nonnegative_int(row.get("first_generated_token_id"), f"{arm_id}.rows[{index}].first_generated_token_id")
        if tokens[0] != row_first:
            raise EvidenceAnalysisError(f"{arm_id}.rows[{index}] first token disagrees with row tokens")
        row_generated = _require_bool(row.get("opener_generated_by_model"), f"{arm_id}.rows[{index}].opener_generated_by_model")
        if row_generated != (row_first == opener):
            raise EvidenceAnalysisError(f"{arm_id}.rows[{index}] opener provenance disagrees with first token")
        if row.get("row_started") is not None and row.get("row_started") is not row_generated:
            raise EvidenceAnalysisError(f"{arm_id}.rows[{index}] row_started disagrees with opener provenance")
        if row.get("status") == "closure" and not row_generated:
            raise EvidenceAnalysisError(f"{arm_id}.rows[{index}] closure lacks model-generated opener")
        if row.get("status") == "native_stop" and row_first not in native_stop_ids:
            raise EvidenceAnalysisError(f"{arm_id}.rows[{index}] native STOP token is not terminal")
        if index == 0 and (row_last != initial_last or row_first != first or row_generated != opener_generated):
            raise EvidenceAnalysisError(f"{arm_id} arm/first-row natural-boundary receipts disagree")
    return {
        "opener_token_id": opener,
        "initial_prefix_last_token_id": initial_last,
        "first_generated_token_id": first,
        "opener_generated_by_model": opener_generated,
        "first_token_class": first_class,
        "terminal_reason": terminal,
    }


def _k13_region_geometry(event: Mapping[str, Any], event_id: str) -> tuple[str | None, list[Any], Mapping[str, Any]]:
    regions = event.get("image_cell_regions")
    receipts = event.get("image_cell_region_receipts")
    if not isinstance(regions, Mapping) or not isinstance(receipts, Mapping):
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 sealed geometry lacks image regions or receipts"
        )
    competitor_cells = regions.get("same_class_competitor")
    competitor_receipt = receipts.get("same_class_competitor")
    if not isinstance(competitor_cells, list) or not isinstance(competitor_receipt, Mapping):
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 sealed competitor geometry is malformed"
        )
    owner = event.get("same_class_competitor_owner_id")
    if owner is not None and (not isinstance(owner, str) or not owner):
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 sealed competitor owner identity is malformed"
        )
    return owner, competitor_cells, competitor_receipt


def _validate_region_receipt(
    receipt: Mapping[str, Any],
    *,
    cells: list[Any],
    event_id: str,
    applicable: bool,
) -> None:
    status = "available" if applicable else "not_applicable"
    if receipt.get("status") != status or receipt.get("available") is not applicable:
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 sealed competitor region disposition disagrees with event geometry"
        )
    if receipt.get("cell_indices") != cells or receipt.get("visual_indices") != cells:
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 sealed competitor region cells disagree with receipt"
        )
    weights = receipt.get("fractional_weights")
    if not isinstance(weights, list):
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 sealed competitor region weights are malformed"
        )
    weight_indices = [item.get("cell_index") for item in weights if isinstance(item, Mapping)]
    if (
        len(weight_indices) != len(weights)
        or weight_indices != cells
        or any(
            isinstance(item.get("overlap_fraction"), bool)
            or not isinstance(item.get("overlap_fraction"), (int, float))
            or not math.isfinite(float(item["overlap_fraction"]))
            or not 0.0 <= float(item["overlap_fraction"]) <= 1.0
            for item in weights
            if isinstance(item, Mapping)
        )
    ):
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 sealed competitor region weight cells disagree"
        )
    if receipt.get("cell_count") != len(cells):
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 sealed competitor region cell count disagrees"
        )
    if receipt.get("cell_indices_sha256") != sha256_json(cells):
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 sealed competitor region cell hash disagrees"
        )
    if receipt.get("weights_sha256") != sha256_json(weights):
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 sealed competitor region weight hash disagrees"
        )
    if applicable:
        if not cells or receipt.get("not_measured_reason") is not None:
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 applicable competitor region is empty or not measured"
            )
    elif (
        cells
        or receipt.get("not_measured_reason") != "no_verified_same_class_competitor"
        or receipt.get("weight_sum") != 0.0
    ):
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 not_applicable competitor region is not canonical"
        )


def _k13_attestation(
    value: Any,
    *,
    event_id: str,
    label: str,
    not_applicable: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != K13_LAYER_RECEIPT_SCHEMA_VERSION:
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 {label} layer attestation schema is malformed"
        )
    if not_applicable:
        # The producer's construction-time stub intentionally carries no
        # unit_id.  It is a four-key declaration that the all-layer check was
        # required but could not be attested because no competitor existed.
        # Keep this raw mapping unchanged: the nested and all-layer aliases
        # must remain exactly equal to one another, including their key set.
        expected = {
            "schema_version": K13_LAYER_RECEIPT_SCHEMA_VERSION,
            "required": True,
            "status": "unattested",
            "exact_same_tensor_all_layers_required": True,
        }
        if dict(value) != expected:
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 {label} not_applicable layer attestation is malformed"
            )
    else:
        if value.get("unit_id") != UNIT_ID:
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 {label} layer attestation unit is foreign"
            )
        if (
            value.get("passed") is not True
            or value.get("all_layers_identical") is not True
            or value.get("missing_layers") != []
            or value.get("repeated_layers") != []
            or value.get("errors") != []
            or "status" in value
        ):
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 {label} applicable layer attestation is malformed"
            )
    return dict(value)


def _k13_applicability(arm: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    """Classify K13 from the executed attention and layer receipts.

    K13 can be mechanically inapplicable when the frozen event has no
    same-class competitor.  That disposition is carried by the live gate's
    executed per-forward receipts; trajectory shape or endpoint outcomes are
    intentionally not used as a proxy.  The two receipt paths must agree for
    every forward, and every forward in an event must carry the same
    disposition.
    """

    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise EvidenceAnalysisError("K13 event identity is malformed")
    owner, competitor_cells, competitor_region_receipt = _k13_region_geometry(event, event_id)
    event_applicable = owner is not None
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in competitor_cells)
        or competitor_cells != sorted(competitor_cells)
        or len(set(competitor_cells)) != len(competitor_cells)
    ):
        raise EvidenceAnalysisError(f"event {event_id}/K13 sealed competitor cells are not sorted")
    _validate_region_receipt(
        competitor_region_receipt,
        cells=competitor_cells,
        event_id=event_id,
        applicable=event_applicable,
    )
    if event_applicable != bool(competitor_cells):
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 sealed competitor owner and cells disagree"
        )

    runtime_receipts = arm.get("runtime_scalar_receipts")
    if not isinstance(runtime_receipts, list) or not runtime_receipts:
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 lacks executed runtime applicability receipts"
        )

    observed: list[tuple[str, str | None, str | None, str | None, bool | None]] = []
    for index, runtime_receipt in enumerate(runtime_receipts):
        if not isinstance(runtime_receipt, Mapping):
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} is malformed"
            )
        attention = runtime_receipt.get("attention_actuation_receipt")
        layer = runtime_receipt.get("layer_consumption_attestation")
        if not isinstance(attention, Mapping) or not isinstance(layer, Mapping):
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} lacks canonical applicability receipts"
            )

        if attention.get("schema_version") != K13_MASK_RECEIPT_SCHEMA_VERSION:
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} mask receipt schema is foreign"
            )
        if attention.get("unit_id") != UNIT_ID or attention.get("arm_id") != "K13":
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} mask receipt identity is foreign"
            )
        if attention.get("factory_arm_id") != "K13":
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} factory identity is foreign"
            )
        attention_status = attention.get("status")
        expected_status = "ready" if event_applicable else "not_applicable"
        if attention_status != expected_status:
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} status disagrees with sealed geometry"
            )
        image_positions = attention.get("image_key_positions")
        fixed_image_positions = attention.get("fixed_absolute_image_key_positions")
        active_image_positions = attention.get("active_image_key_positions")
        if (
            not isinstance(image_positions, list)
            or not image_positions
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in image_positions)
            or image_positions != sorted(image_positions)
            or len(set(image_positions)) != len(image_positions)
            or fixed_image_positions != image_positions
            or active_image_positions != image_positions
        ):
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} image key positions are malformed"
            )
        fixed_competitor = attention.get("fixed_absolute_same_class_competitor_positions")
        active_competitor = attention.get("active_same_class_competitor_positions")
        selected_competitor = attention.get("selected_positions")
        fixed_hash = attention.get("fixed_competitor_positions_sha256")
        selected_hash = attention.get("selected_positions_sha256")
        if (
            not isinstance(fixed_competitor, list)
            or not isinstance(active_competitor, list)
            or not isinstance(selected_competitor, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in fixed_competitor)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in active_competitor)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in selected_competitor)
            or fixed_competitor != sorted(fixed_competitor)
            or active_competitor != sorted(active_competitor)
            or selected_competitor != sorted(selected_competitor)
            or len(set(fixed_competitor)) != len(fixed_competitor)
            or len(set(active_competitor)) != len(active_competitor)
            or len(set(selected_competitor)) != len(selected_competitor)
            or attention.get("selected_key_count") != len(selected_competitor)
            or fixed_hash != sha256_json(fixed_competitor)
            or selected_hash != sha256_json(selected_competitor)
        ):
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} competitor positions or hashes are malformed"
            )
        if event_applicable:
            if (
                not fixed_competitor
                or not active_competitor
                or not selected_competitor
                or not set(active_competitor).issubset(fixed_competitor)
                or selected_competitor != active_competitor
            ):
                raise EvidenceAnalysisError(
                    f"event {event_id}/K13 runtime receipt {index} applicable competitor positions are malformed"
                )
        elif fixed_competitor or active_competitor or selected_competitor:
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} not_applicable competitor positions are nonempty"
            )
        query_positions = attention.get("query_positions")
        if (
            not isinstance(query_positions, list)
            or not query_positions
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in query_positions)
            or query_positions != sorted(query_positions)
            or len(set(query_positions)) != len(query_positions)
            or attention.get("query_positions_sha256") != sha256_json(query_positions)
        ):
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} query position hash is malformed"
            )
        layer_status = layer.get("status")
        if layer.get("passed") is not True:
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} layer attestation did not pass"
            )

        nested_layer = attention.get("layer_consumption_attestation")
        nested_layer = _k13_attestation(
            nested_layer,
            event_id=event_id,
            label=f"runtime receipt {index} nested",
            not_applicable=not event_applicable,
        )
        all_layer = attention.get("all_layer_consumption_attestation")
        if not isinstance(all_layer, Mapping) or dict(all_layer) != nested_layer:
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} nested layer attestation aliases disagree"
            )
        direct_layer = _k13_attestation(
            layer,
            event_id=event_id,
            label=f"runtime receipt {index} direct",
            not_applicable=False,
        ) if event_applicable else None

        if not event_applicable:
            if layer_status != "not_applicable":
                raise EvidenceAnalysisError(
                    f"event {event_id}/K13 runtime receipt {index} has conflicting not_applicable attestation"
                )
            # The callback's construction receipt is ``unattested`` when no
            # operator exists; the executed direct attestation above is the
            # canonical not-applicable proof.
            if attention.get("reason") != K13_NOT_APPLICABLE_REASON:
                raise EvidenceAnalysisError(
                    f"event {event_id}/K13 runtime receipt {index} not_applicable reason is non-canonical"
                )
            if dict(layer) != {"passed": True, "status": "not_applicable"}:
                raise EvidenceAnalysisError(
                    f"event {event_id}/K13 runtime receipt {index} direct not_applicable attestation is malformed"
                )
            if direct_layer is not None or nested_layer.get("status") != "unattested":
                raise EvidenceAnalysisError(
                    f"event {event_id}/K13 runtime receipt {index} nested attestation conflicts with not_applicable"
                )
            observed.append((attention_status, attention["reason"], layer_status, nested_layer.get("status"), None))
            continue

        # The live all-layer attestor reports ``passed=True`` without a
        # status key for an applicable forward.  A status or failed nested
        # attestation here is evidence of a malformed or conflicting receipt.
        reason = attention.get("reason")
        if reason is not None:
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} applicable receipt carries a reason"
            )
        if direct_layer != nested_layer:
            raise EvidenceAnalysisError(
                f"event {event_id}/K13 runtime receipt {index} applicable attestations disagree"
            )
        observed.append((attention_status, None, layer_status, None, True))

    first_status, first_reason, first_layer_status, first_nested_status, first_nested_passed = observed[0]
    if any(
        status != first_status
        or reason != first_reason
        or layer_status != first_layer_status
        or nested_status != first_nested_status
        or nested_passed != first_nested_passed
        for status, reason, layer_status, nested_status, nested_passed in observed[1:]
    ):
        raise EvidenceAnalysisError(
            f"event {event_id}/K13 applicability receipts conflict across forwards"
        )
    if first_status == "not_applicable":
        return {
            "status": "not_applicable",
            "reason": first_reason,
            "measured": False,
            "excluded_from_effect_metrics": True,
            "executed_forward_count": len(observed),
        }
    return {
        "status": "applicable",
        "reason": None,
        "measured": True,
        "excluded_from_effect_metrics": False,
        "executed_forward_count": len(observed),
    }


def _summarize_arm(
    arm: Mapping[str, Any],
    *,
    arm_id: str,
    event: Mapping[str, Any],
    census_info: Mapping[str, Any],
    claim_scope: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        validated_arm = validate_arm_result(arm, arm_id)
    except (CohortContractError, ValueError) as exc:
        raise EvidenceAnalysisError(f"formal arm validation failed for {event['event_id']}/{arm_id}: {exc}") from exc
    if validated_arm["executor_identity"]["pre_gpu"].get("claim_scope") != dict(claim_scope):
        raise EvidenceAnalysisError(
            f"{event['event_id']}/{arm_id} pre-GPU claim_scope differs from manifest-derived scope"
        )
    natural = _validate_natural_receipts(arm, arm_id)
    prefix = event["natural_boundary"]["prefix_token_ids"]
    if not prefix or natural["initial_prefix_last_token_id"] != prefix[-1]:
        raise EvidenceAnalysisError(
            f"{event['event_id']}/{arm_id} initial prefix receipt differs from the sealed event"
        )
    covered_ids = sorted(event["owner_refs"]["covered_owner_ids"])
    covered = set(covered_ids)
    rows = arm["rows"]
    seen_endpoint: set[str] = set()
    seen_before: set[str] = set(covered)
    strict_row_owner_ids: list[str] = []
    parsed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        owner_id, strict, match_status = _strict_owner_match(row)
        if row.get("status") == "closure" and not isinstance(row.get("owner_match"), Mapping):
            raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id}/row-{index} closure lacks owner_match")
        if row.get("status") == "closure" and row.get("owner_match_status") != match_status:
            raise EvidenceAnalysisError(
                f"{event['event_id']}/{arm_id}/row-{index} owner_match_status differs from owner_match"
            )
        if row.get("status") != "closure" and (row.get("owner_match") is not None or row.get("owner_match_status") is not None):
            raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id}/row-{index} non-closure carries stale owner identity")
        if strict:
            census_row = census_info["by_key"].get(("S", owner_id))
            if census_row is None or census_row.get("image_id") != event["image_id"]:
                raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id} strict owner is foreign to S image: {owner_id}")
        covered_repeat = bool(strict and owner_id in covered)
        duplicate = bool(strict and owner_id in seen_endpoint)
        if strict and owner_id is not None:
            strict_row_owner_ids.append(owner_id)
            seen_endpoint.add(owner_id)
        parsed_rows.append({
            "owner_id": owner_id,
            "strict": strict,
            "match_status": match_status,
            "covered_repeat": covered_repeat,
            "duplicate": duplicate,
            "status": row.get("status"),
        })
        row_book = row.get("owner_bookkeeping")
        if not isinstance(row_book, Mapping):
            raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id}/row-{index} lacks owner bookkeeping")
        expected_row = {
            "covered_owner_ids_before": covered_ids,
            "seen_owner_ids_before": sorted(seen_before),
            "matched_owner_id": owner_id,
            "strict_physical_owner_match": strict,
            "covered_repeat": covered_repeat,
            "duplicate": duplicate,
            "new_target_owner": bool(strict and owner_id not in covered),
        }
        for key, expected in expected_row.items():
            if row_book.get(key) != expected:
                raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id}/row-{index} bookkeeping differs: {key}")
        if strict and owner_id is not None:
            seen_before.add(owner_id)

    derived_parse = {
        "valid_rows": sum(item["status"] == "closure" for item in parsed_rows),
        "duplicate_rows": sum(item["duplicate"] for item in parsed_rows),
        "unmatched_rows": sum(item["status"] == "closure" and not item["strict"] for item in parsed_rows),
        "ambiguous_rows": sum(item["status"] == "closure" and item["match_status"] == "ambiguous" for item in parsed_rows),
        "malformed_rows": sum(item["status"] == "over_continuation" for item in parsed_rows),
        "invalid_rows": sum(item["status"] in {"invalid", "over_continuation", "max_budget"} for item in parsed_rows),
    }
    book = arm.get("owner_bookkeeping")
    if not isinstance(book, Mapping) or not isinstance(book.get("parse"), Mapping) or not isinstance(book.get("stop"), Mapping):
        raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id} owner bookkeeping is incomplete")
    declared_parse = dict(book["parse"])
    if set(declared_parse) != set(PARSE_KEYS):
        raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id} parse keys differ from frozen contract")
    for key in PARSE_KEYS:
        _require_nonnegative_int(declared_parse[key], f"{event['event_id']}/{arm_id}.parse.{key}")
    if declared_parse != derived_parse:
        raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id} parse counts differ from exact row classification")
    endpoint_ids = sorted(set(strict_row_owner_ids))
    covered_repeat_ids = sorted(set(endpoint_ids) & covered)
    new_target_ids = sorted(set(endpoint_ids) - covered)
    declared_lists = {
        "raw_endpoint_owner_ids": endpoint_ids,
        "covered_repeat_owner_ids": covered_repeat_ids,
        "new_target_owner_ids": new_target_ids,
    }
    for key, expected in declared_lists.items():
        if _require_id_list(book.get(key), f"{event['event_id']}/{arm_id}.{key}") != expected:
            raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id} {key} differs from exact closure rows")
    exact_counts = {
        "row_count": len(rows),
        "strict_physical_owner_match_count": len(strict_row_owner_ids),
        "duplicate_count": derived_parse["duplicate_rows"],
        "unmatched_count": derived_parse["unmatched_rows"],
    }
    for key, expected in exact_counts.items():
        if _require_nonnegative_int(book.get(key), f"{event['event_id']}/{arm_id}.{key}") != expected:
            raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id} {key} differs from exact rows")
    expected_stop = {"stopped": natural["terminal_reason"] != "closure", "stop_reason": natural["terminal_reason"]}
    if dict(book["stop"]) != expected_stop or book.get("horizon_status") != natural["terminal_reason"]:
        raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id} stop/horizon bookkeeping differs from trajectory")
    row_entry = book.get("row_entry")
    if not isinstance(row_entry, Mapping):
        raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id} row_entry bookkeeping is missing")
    entry_expected = {
        "admission_mode": "pre_opener_natural",
        "first_generated_token_id": natural["first_generated_token_id"],
        "opener_generated_by_model": natural["opener_generated_by_model"],
        "opener_injected": False,
        "row_started": natural["opener_generated_by_model"],
    }
    for key, expected in entry_expected.items():
        if row_entry.get(key) != expected:
            raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id} row_entry differs: {key}")
    for index, row in enumerate(rows):
        row_book = row["owner_bookkeeping"]
        if row_book.get("parse") != derived_parse or row_book.get("stop") != expected_stop or row_book.get("horizon_status") != natural["terminal_reason"]:
            raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id}/row-{index} aggregate bookkeeping differs")
        if row_book.get("row_entry") != row_entry:
            raise EvidenceAnalysisError(f"{event['event_id']}/{arm_id}/row-{index} row_entry bookkeeping differs")

    target_owner = event["event_id"]
    fully_measured_other = bool(
        derived_parse["valid_rows"] > 0
        and derived_parse["valid_rows"] == len(rows)
        and derived_parse["unmatched_rows"] == 0
        and derived_parse["ambiguous_rows"] == 0
        and derived_parse["duplicate_rows"] == 0
        and derived_parse["malformed_rows"] == 0
        and derived_parse["invalid_rows"] == 0
        and target_owner not in endpoint_ids
        and bool(endpoint_ids)
    )
    return {
        "arm_id": arm_id,
        "mechanically_valid": True,
        "admission_mode": "pre_opener_natural",
        "first_token_class": natural["first_token_class"],
        "first_generated_token_id": natural["first_generated_token_id"],
        "opener_generated_by_model": natural["opener_generated_by_model"],
        "terminal_reason": natural["terminal_reason"],
        "complete_natural_rows": derived_parse["valid_rows"],
        "strict_physical_owner_match_count": len(strict_row_owner_ids),
        "strict_physical_owner_ids": endpoint_ids,
        "target_owner_id": target_owner,
        "target_strict_release": target_owner in endpoint_ids,
        "unmatched_rows": derived_parse["unmatched_rows"],
        "duplicate_rows": derived_parse["duplicate_rows"],
        "ambiguous_rows": derived_parse["ambiguous_rows"],
        "malformed_rows": derived_parse["malformed_rows"],
        "invalid_rows": derived_parse["invalid_rows"],
        "native_stop": natural["terminal_reason"] == "native_stop",
        "covered_repeat_owner_ids": covered_repeat_ids,
        "uncovered_owner_gain_ids": new_target_ids,
        "fully_measured_strict_other_owner": fully_measured_other,
        "scientific_unmatched_is_valid": derived_parse["unmatched_rows"] > 0,
        "parse": derived_parse,
    }


def _delta_count(value: Any, label: str) -> int:
    """Normalize one non-negative arm count or boolean release to an int."""

    if isinstance(value, bool):
        return int(value)
    return _require_nonnegative_int(value, label)


def _validate_delta_partition(delta: Mapping[str, Any]) -> None:
    """Validate the closed ``_delta`` key partition before disposition."""

    observed = set(delta)
    if observed != _DELTA_ALL_KEYS:
        missing = sorted(_DELTA_ALL_KEYS - observed)
        extra = sorted(observed - _DELTA_ALL_KEYS)
        raise EvidenceAnalysisError(
            f"delta fields are not exactly partitioned (missing={missing}, extra={extra})"
        )
    if delta["status"] != "measured":
        raise EvidenceAnalysisError("delta status is not measured")
    if delta["unmatched_is_scientific_not_technical"] is not True:
        raise EvidenceAnalysisError("delta unmatched semantics are not scientific")
    for key in _DELTA_ENDPOINT_BOOLEAN_KEYS:
        if not isinstance(delta[key], bool):
            raise EvidenceAnalysisError(f"delta {key} must be boolean")
    for key in _DELTA_ENDPOINT_NUMERIC_KEYS + ("complete_natural_rows_delta",):
        if isinstance(delta[key], bool) or not isinstance(delta[key], int):
            raise EvidenceAnalysisError(f"delta {key} must be an integer")
    for key in _DELTA_ENDPOINT_IDENTITY_KEYS + (
        "covered_repeat_owner_ids_added",
        "covered_repeat_owner_ids_removed",
        "uncovered_owner_gain_ids_added",
        "uncovered_owner_gain_ids_removed",
    ):
        _require_id_list(delta[key], f"delta.{key}")

    strict_added = set(delta["strict_owner_ids_added"])
    strict_removed = set(delta["strict_owner_ids_removed"])
    for key in ("covered_repeat_owner_ids_added", "uncovered_owner_gain_ids_added"):
        if not set(delta[key]).issubset(strict_added):
            raise EvidenceAnalysisError(f"delta {key} is outside strict_owner_ids_added")
    for key in ("covered_repeat_owner_ids_removed", "uncovered_owner_gain_ids_removed"):
        if not set(delta[key]).issubset(strict_removed):
            raise EvidenceAnalysisError(f"delta {key} is outside strict_owner_ids_removed")

    expected_complete = (
        delta["strict_physical_owner_match_count_delta"]
        + delta["unmatched_rows_delta"]
    )
    if delta["complete_natural_rows_delta"] != expected_complete:
        raise EvidenceAnalysisError(
            "complete_natural_rows_delta must equal strict_physical_owner_match_count_delta "
            "+ unmatched_rows_delta"
        )


def _delta(intervention: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    numeric = (
        "complete_natural_rows",
        "strict_physical_owner_match_count",
        "unmatched_rows",
        "duplicate_rows",
        "ambiguous_rows",
        "malformed_rows",
        "invalid_rows",
        "target_strict_release",
    )
    result: dict[str, Any] = {
        "intervention_arm": intervention["arm_id"],
        "baseline_arm": baseline["arm_id"],
        "status": "measured",
    }
    for key in numeric:
        left = _delta_count(intervention[key], f"intervention.{key}")
        right = _delta_count(baseline[key], f"baseline.{key}")
        result[f"{key}_delta"] = left - right
    result["first_token_class_changed"] = intervention["first_token_class"] != baseline["first_token_class"]
    result["terminal_reason_changed"] = intervention["terminal_reason"] != baseline["terminal_reason"]

    list_fields = (
        "strict_physical_owner_ids",
        "covered_repeat_owner_ids",
        "uncovered_owner_gain_ids",
    )
    normalized: dict[str, tuple[list[str], list[str]]] = {}
    for key in list_fields:
        normalized[key] = (
            _require_id_list(intervention[key], f"intervention.{key}"),
            _require_id_list(baseline[key], f"baseline.{key}"),
        )
    strict_intervention, strict_baseline = normalized["strict_physical_owner_ids"]
    strict_intervention_set = set(strict_intervention)
    strict_baseline_set = set(strict_baseline)
    for source_key in ("covered_repeat_owner_ids", "uncovered_owner_gain_ids"):
        intervention_ids, baseline_ids = normalized[source_key]
        if not set(intervention_ids).issubset(strict_intervention_set):
            raise EvidenceAnalysisError(
                f"intervention.{source_key} is outside strict_physical_owner_ids"
            )
        if not set(baseline_ids).issubset(strict_baseline_set):
            raise EvidenceAnalysisError(
                f"baseline.{source_key} is outside strict_physical_owner_ids"
            )
    result["strict_owner_ids_added"] = sorted(strict_intervention_set - strict_baseline_set)
    result["strict_owner_ids_removed"] = sorted(strict_baseline_set - strict_intervention_set)
    for source_key, added_key, removed_key in (
        ("covered_repeat_owner_ids", "covered_repeat_owner_ids_added", "covered_repeat_owner_ids_removed"),
        ("uncovered_owner_gain_ids", "uncovered_owner_gain_ids_added", "uncovered_owner_gain_ids_removed"),
    ):
        intervention_ids, baseline_ids = normalized[source_key]
        result[added_key] = sorted(set(intervention_ids) - set(baseline_ids))
        result[removed_key] = sorted(set(baseline_ids) - set(intervention_ids))
    result["unmatched_is_scientific_not_technical"] = True
    _validate_delta_partition(result)
    return result


def _dynamic_disposition(
    intervention: Mapping[str, Any],
    baseline: Mapping[str, Any],
    delta: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_delta_partition(delta)
    endpoint_change_reasons = sorted(
        [
            key
            for key in _DELTA_ENDPOINT_BOOLEAN_KEYS
            if delta[key]
        ]
        + [
            key
            for key in _DELTA_ENDPOINT_NUMERIC_KEYS
            if delta[key] != 0
        ]
        + [
            key
            for key in _DELTA_ENDPOINT_IDENTITY_KEYS
            if delta[key]
        ]
    )
    changed = bool(endpoint_change_reasons)
    degenerate = bool(
        baseline["first_token_class"] == "object_ref_start"
        and intervention["first_token_class"] == "other_invalid"
        and intervention["terminal_reason"] == "invalid"
        and intervention["complete_natural_rows"] == 0
    )
    return {
        "endpoint_changed": changed,
        "endpoint_change_reasons": endpoint_change_reasons,
        "degenerate_grammar_disruption": degenerate,
        "factor_qualified": changed and not degenerate,
        "crossover_candidate": changed and not degenerate,
        "status": "degenerate_grammar_disruption" if degenerate else ("qualified_endpoint_change" if changed else "no_endpoint_change"),
    }


def _validate_result_bindings(
    document: Mapping[str, Any],
    *,
    ref: Mapping[str, Any] | None,
    event: Mapping[str, Any],
    manifest_info: Mapping[str, Any],
    census_info: Mapping[str, Any],
) -> str:
    if (
        document.get("schema_version") != EVENT_SCHEMA_VERSION
        or document.get("manifest_sha256") != manifest_info["manifest_sha256"]
        or document.get("manifest_self_sha256") != manifest_info["manifest_self_sha256"]
        or document.get("source_census_sha256") != census_info["raw_file_sha256"]
    ):
        raise EvidenceAnalysisError(f"event {event['event_id']} manifest/census binding is missing or foreign")
    if not isinstance(ref, Mapping) or ref.get("result_sha256") is None:
        raise EvidenceAnalysisError(f"event {event['event_id']} merged result reference hash is missing")
    result_self = _require_sha(document.get("result_sha256"), f"event {event['event_id']}.result_sha256")
    if result_self != ref.get("result_sha256") or result_self != document_self_sha256(document, "result_sha256"):
        raise EvidenceAnalysisError(f"event {event['event_id']} result reference/self hash mismatch")
    return result_self


def _load_validated_event_documents(
    *,
    merged_info: Mapping[str, Any],
    manifest_info: Mapping[str, Any],
    census_info: Mapping[str, Any],
    plan_info: Mapping[str, Any],
    claim_scope: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ref_by_index = {ref["event_index"]: ref for ref in merged_info["events"]}
    rows: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for event in manifest_info["events"]:
        shard_index = event["event_index"] % SHARD_COUNT
        shard = _shard_id(shard_index)
        event_root = merged_info["shards_root"] / shard / _event_root_name(event)
        result_path = event_root / "result.json"
        terminal_path = event_root / "terminal_summary.json"
        result_raw, result_source = _strict_canonical_document(result_path, f"event {event['event_id']} result")
        terminal_raw, terminal_source = _strict_canonical_document(terminal_path, f"event {event['event_id']} terminal")
        try:
            document = _load_completed_event(event_root, event=event, manifest_info=manifest_info)
        except (CohortContractError, OSError, ValueError) as exc:
            raise EvidenceAnalysisError(f"event {event['event_id']} production validation failed: {exc}") from exc
        if document is None:
            raise EvidenceAnalysisError(f"event {event['event_id']} completed document is missing")
        _validate_event_plan_binding(document, plan_sha256=plan_info["plan_sha256"], shard=shard, shard_index=shard_index)
        ref = ref_by_index.get(event["event_index"])
        result_self = _validate_result_bindings(
            document,
            ref=ref,
            event=event,
            manifest_info=manifest_info,
            census_info=census_info,
        )
        if terminal_raw.get("result_sha256") != result_self:
            raise EvidenceAnalysisError(f"event {event['event_id']} terminal/result semantic hash mismatch")
        terminal_self = _require_sha(terminal_raw.get("self_sha256"), f"event {event['event_id']}.terminal.self_sha256")
        if terminal_self != document_self_sha256(terminal_raw):
            raise EvidenceAnalysisError(f"event {event['event_id']} terminal self hash mismatch")
        arms = document.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(ARM_ORDER):
            raise EvidenceAnalysisError(f"event {event['event_id']} arm set is incomplete")
        summaries = {
            arm: _summarize_arm(
                arms[arm],
                arm_id=arm,
                event=event,
                census_info=census_info,
                claim_scope=claim_scope,
            )
            for arm in ARM_ORDER
        }
        k13_applicability = _k13_applicability(arms["K13"], event)
        deltas = {
            arm: _delta(summaries[arm], summaries[baseline])
            for arm, baseline in OWN_BASELINE.items()
            if not (
                arm == "K13"
                and k13_applicability["status"] == "not_applicable"
            )
        }
        dynamic = {
            arm: _dynamic_disposition(summaries[arm], summaries[OWN_BASELINE[arm]], deltas[arm])
            for arm in DYNAMIC_INTERVENTIONS
        }
        rows.append({
            "event_index": event["event_index"],
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "target_owner_id": event["event_id"],
            "arms": summaries,
            "k13_applicability": k13_applicability,
            "intervention_minus_own_baseline": deltas,
            "dynamic_factor_disposition": dynamic,
            "post_opener_seeded_diagnostic": {
                "status": "not_present_in_frozen_cohort",
                "pooled_with_primary": False,
            },
        })
        bindings.append({
            "event_index": event["event_index"],
            "event_id": event["event_id"],
            "shard_id": shard,
            "result": {
                "raw_file_sha256": result_source["raw_file_sha256"],
                "semantic_self_sha256": result_self,
            },
            "terminal": {
                "raw_file_sha256": terminal_source["raw_file_sha256"],
                "semantic_self_sha256": terminal_self,
            },
        })
    return rows, bindings


def _qualification(
    events: Sequence[Mapping[str, Any]],
    *,
    claim_scope: Mapping[str, Any],
) -> dict[str, Any]:
    k10_cases = [
        event for event in events
        if event["arms"]["K01"]["mechanically_valid"]
        and event["arms"]["K10"]["target_strict_release"]
    ]
    k10_images = sorted({event["image_id"] for event in k10_cases})
    k14_cases = []
    k14_rows = []
    for event in events:
        target = event["target_owner_id"]
        k01 = event["arms"]["K01"]
        k14t = event["arms"]["K14T"]
        k14b = event["arms"]["K14B"]
        control_status = (
            "native_stop_or_grammar"
            if k14b["native_stop"]
            else "fully_measured_strict_other_owner"
            if k14b["fully_measured_strict_other_owner"]
            else "target_B"
            if target in k14b["strict_physical_owner_ids"]
            else "unmatched"
            if k14b["unmatched_rows"] > 0
            else "invalid_or_unmeasured"
        )
        qualified = bool(
            k01["mechanically_valid"]
            and k14t["target_strict_release"]
            and control_status in {"native_stop_or_grammar", "fully_measured_strict_other_owner"}
        )
        if qualified:
            k14_cases.append(event)
        k14_rows.append({
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "qualified": qualified,
            "K01_valid": k01["mechanically_valid"],
            "K14T_strict_target": k14t["target_strict_release"],
            "K14B_control_status": control_status,
        })
    k14_images = sorted({event["image_id"] for event in k14_cases})
    k10_checkpoint_observed = len(k10_cases) >= 3 and len(k10_images) >= 2
    k14_checkpoint_observed = len(k14_cases) >= 3 and len(k14_images) >= 2
    checkpoint_scope_open = claim_scope["checkpoint_claim_qualified"] is True
    static_scope_open = claim_scope["static_direction_claim_qualified"] is True
    k10_checkpoint = checkpoint_scope_open and k10_checkpoint_observed
    k14_checkpoint = checkpoint_scope_open and k14_checkpoint_observed
    checkpoint_level_qualified = k10_checkpoint or k14_checkpoint
    static_direction_qualified = static_scope_open and checkpoint_level_qualified
    return {
        "K10_hard_oracle": {
            "case_level_event_count": len(k10_cases),
            "case_level_event_ids": [event["event_id"] for event in k10_cases],
            "case_level_image_count": len(k10_images),
            "case_level_image_ids": k10_images,
            "case_level_floor": {"minimum_events": 1},
            "case_level_status": "qualified" if k10_cases else "hold",
            "hard_replication_floor": {"minimum_events": 2, "minimum_images": 2},
            "hard_replication_status": "qualified" if len(k10_cases) >= 2 and len(k10_images) >= 2 else "hold",
            "checkpoint_floor": {"minimum_events": 3, "minimum_images": 2},
            "checkpoint_observed_before_scope_gate": k10_checkpoint_observed,
            "checkpoint_status": "qualified" if k10_checkpoint else "hold",
        },
        "K14_finite_salience": {
            "events": k14_rows,
            "qualifying_event_count": len(k14_cases),
            "qualifying_image_count": len(k14_images),
            "checkpoint_floor": {"minimum_events": 3, "minimum_images": 2},
            "checkpoint_observed_before_scope_gate": k14_checkpoint_observed,
            "checkpoint_status": "qualified" if k14_checkpoint else "hold",
        },
        "scope_gate": {
            "execution_scope": claim_scope["execution_scope"],
            "checkpoint_scope_open": checkpoint_scope_open,
            "static_direction_scope_open": static_scope_open,
            "training_scope_open": False,
        },
        "checkpoint_level_status": "qualified" if checkpoint_level_qualified else "hold",
        "static_direction_status": "qualified" if static_direction_qualified else "hold",
        "training_status": "hold",
    }


def _absolute_without_resolve(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _reject_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise EvidenceAnalysisError(f"{label} resolves through a symlink: {current}")


def _write_once(path: str | Path, document: Mapping[str, Any]) -> str:
    destination = _absolute_without_resolve(path)
    _reject_symlink_components(destination.parent, "output parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, "output parent")
    payload = canonical_json_bytes(document) + b"\n"
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
            raise EvidenceAnalysisError(f"immutable output collision or symlink: {destination}")
        return sha256_bytes(payload)
    parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(
            destination.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=parent_fd,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.unlink(destination.name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return sha256_bytes(payload)


def _producer_ref() -> dict[str, Any]:
    """Return the immutable identity of this CPU analyzer source."""

    source = Path(__file__).absolute()
    # Check the import entry point before resolving it.  Resolving first would
    # hide a symlinked producer and make the provenance claim depend on the
    # target rather than the source path actually executed.
    if source.is_symlink() or not source.is_file():
        raise EvidenceAnalysisError(
            f"analyzer producer is not a regular non-symlink file: {source}"
        )
    path = source.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise EvidenceAnalysisError(
            f"analyzer producer is not a regular non-symlink file: {path}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def analyze_cohort(
    aggregate: str | Path,
    manifest: str | Path,
    census: str | Path,
    *,
    plan: str | Path,
    gate_receipt: str | Path,
    shards_root: str | Path,
    output: str | Path | None = None,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    manifest_document, manifest_source = _strict_canonical_document(manifest, "event manifest")
    try:
        manifest_info = validate_cohort_manifest(manifest_source["path"])
    except (CohortContractError, OSError, ValueError) as exc:
        raise EvidenceAnalysisError(f"canonical cohort manifest validation failed: {exc}") from exc
    if manifest_document.get("self_sha256") != manifest_info["manifest_self_sha256"]:
        raise EvidenceAnalysisError("manifest raw document/self binding differs from canonical validator")
    census_info = validate_census(census, manifest_info=manifest_info)
    plan_document, plan_source = _strict_canonical_document(plan, "execution plan")
    try:
        plan_info = validate_plan(plan_source["path"], manifest_info=manifest_info)
    except (ExecutionPlanError, CohortContractError, OSError, ValueError) as exc:
        raise EvidenceAnalysisError(f"execution plan validation failed: {exc}") from exc
    if plan_document.get("plan_sha256") != plan_info["plan_sha256"]:
        raise EvidenceAnalysisError("plan raw document/self binding differs from canonical validator")
    gate_info = _validate_gate_binding(gate_receipt, plan_info)
    merged_info = _validate_merged_set(
        aggregate,
        shards_root=shards_root,
        manifest_info=manifest_info,
        plan_info=plan_info,
    )
    claim_scope = merged_info["claim_scope"]
    events, event_bindings = _load_validated_event_documents(
        merged_info=merged_info,
        manifest_info=manifest_info,
        census_info=census_info,
        plan_info=plan_info,
        claim_scope=claim_scope,
    )
    qualification = _qualification(events, claim_scope=claim_scope)
    producer = _producer_ref()
    claim_decision = {
        "execution_scope": claim_scope["execution_scope"],
        "analyzed_event_count": len(events),
        "all_case_events_analyzed": len(events) == claim_scope["event_count"],
        "checkpoint_claim_qualified": bool(
            claim_scope["checkpoint_claim_qualified"]
            and qualification["checkpoint_level_status"] == "qualified"
        ),
        "static_direction_claim_qualified": bool(
            claim_scope["static_direction_claim_qualified"]
            and qualification["static_direction_status"] == "qualified"
        ),
        "training_claim_qualified": False,
        "checkpoint_claim_status": qualification["checkpoint_level_status"],
        "static_direction_claim_status": qualification["static_direction_status"],
        "training_claim_status": "hold",
    }
    dynamic_summary = {
        arm: {
            "event_count": len(events),
            "endpoint_change_event_count": sum(event["dynamic_factor_disposition"][arm]["endpoint_changed"] for event in events),
            "degenerate_grammar_disruption_event_count": sum(event["dynamic_factor_disposition"][arm]["degenerate_grammar_disruption"] for event in events),
            "factor_qualified_event_count": sum(event["dynamic_factor_disposition"][arm]["factor_qualified"] for event in events),
        }
        for arm in DYNAMIC_INTERVENTIONS
    }
    k13_measured_events = [
        event for event in events if event["k13_applicability"]["status"] == "applicable"
    ]
    k13_not_applicable_events = [
        event for event in events if event["k13_applicability"]["status"] == "not_applicable"
    ]
    k13_not_applicable_reasons = dict(
        sorted(
            Counter(
                event["k13_applicability"]["reason"]
                for event in k13_not_applicable_events
            ).items()
        )
    )
    arm_denominators = {
        arm: {
            "event_count": len(k13_measured_events) if arm == "K13" else len(events),
            "complete_natural_rows": sum(
                event["arms"][arm]["complete_natural_rows"]
                for event in (k13_measured_events if arm == "K13" else events)
            ),
            "strict_physical_owner_matches": sum(
                event["arms"][arm]["strict_physical_owner_match_count"]
                for event in (k13_measured_events if arm == "K13" else events)
            ),
            "target_strict_release_events": sum(
                event["arms"][arm]["target_strict_release"]
                for event in (k13_measured_events if arm == "K13" else events)
            ),
            "unmatched_rows": sum(
                event["arms"][arm]["unmatched_rows"]
                for event in (k13_measured_events if arm == "K13" else events)
            ),
            "duplicate_rows": sum(
                event["arms"][arm]["duplicate_rows"]
                for event in (k13_measured_events if arm == "K13" else events)
            ),
            "malformed_rows": sum(
                event["arms"][arm]["malformed_rows"]
                for event in (k13_measured_events if arm == "K13" else events)
            ),
            "invalid_rows": sum(
                event["arms"][arm]["invalid_rows"]
                for event in (k13_measured_events if arm == "K13" else events)
            ),
            "native_stop_events": sum(
                event["arms"][arm]["native_stop"]
                for event in (k13_measured_events if arm == "K13" else events)
            ),
            "first_token_classes": dict(
                sorted(
                    Counter(
                        event["arms"][arm]["first_token_class"]
                        for event in (k13_measured_events if arm == "K13" else events)
                    ).items()
                )
            ),
            **(
                {
                    "observed_event_count": len(events),
                    "measured_event_count": len(k13_measured_events),
                    "not_applicable_event_count": len(k13_not_applicable_events),
                    "not_applicable_reasons": k13_not_applicable_reasons,
                }
                if arm == "K13"
                else {}
            ),
        }
        for arm in ARM_ORDER
    }
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "unit_id": UNIT_ID,
        "decision_owner": "S",
        "producer": producer,
        "primary": PRIMARY,
        "claim_scope": claim_scope,
        "claim_decision": claim_decision,
        "input_bindings": {
            "manifest": {
                "raw_file_sha256": manifest_source["raw_file_sha256"],
                "semantic_self_sha256": manifest_info["manifest_self_sha256"],
            },
            "census": {
                "raw_file_sha256": census_info["raw_file_sha256"],
                "semantic_self_sha256": census_info["semantic_self_sha256"],
            },
            "plan": {
                "raw_file_sha256": plan_source["raw_file_sha256"],
                "semantic_self_sha256": plan_info["plan_sha256"],
            },
            "gate": gate_info,
            "merged": merged_info["bindings"],
            "events": event_bindings,
        },
        "denominators": {
            "census_owner_rows": 784,
            "census_checkpoint_owner_rows": census_info["checkpoint_counts"],
            "manifest_events": len(events),
            "distinct_images": len({event["image_id"] for event in events}),
            "shards": SHARD_COUNT,
            "arm_event_denominators": arm_denominators,
            "k13_applicability": {
                "observed_event_count": len(events),
                "measured_event_count": len(k13_measured_events),
                "not_applicable_event_count": len(k13_not_applicable_events),
                "not_applicable_reasons": k13_not_applicable_reasons,
            },
        },
        "events": events,
        "qualification": qualification,
        "dynamic_history_effects": dynamic_summary,
        "post_opener_seeded_diagnostics": {
            "status": "not_present_in_frozen_cohort",
            "pooled_with_pre_opener_primary": False,
            "interpretation": "conditional owner realization only if separately measured",
        },
        "next_step_flags": {
            "static_checkpoint_floor_observed": claim_decision["static_direction_claim_qualified"],
            "dynamic_factor_observed": any(value["factor_qualified_event_count"] > 0 for value in dynamic_summary.values()),
            "authorize_crossover": False,
            "authorize_a3": False,
            "authorize_p4": False,
            "authorize_training": False,
            "note": "Outcome flags are conditional observations only and confer no execution authority.",
        },
        "contrast_semantics": "intervention_minus_own_baseline_only",
        "unmatched_semantics": "mechanically_valid_scientific_outcome_not_technical_invalid",
    }
    evidence["self_sha256"] = document_self_sha256(evidence)
    evidence_payload_sha = sha256_bytes(canonical_json_bytes(evidence) + b"\n")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "unit_id": UNIT_ID,
        "decision_owner": "S",
        "producer": producer,
        "claim_scope": claim_scope,
        "claim_decision": claim_decision,
        "evidence_raw_file_sha256": evidence_payload_sha,
        "evidence_semantic_self_sha256": evidence["self_sha256"],
        "event_count": len(events),
        "distinct_image_count": len({event["image_id"] for event in events}),
        "shard_count": SHARD_COUNT,
        "dynamic_history_effects": evidence["dynamic_history_effects"],
        "k13_applicability": evidence["denominators"]["k13_applicability"],
        "checkpoint_level_status": qualification["checkpoint_level_status"],
        "next_step_flags": evidence["next_step_flags"],
    }
    receipt["self_sha256"] = document_self_sha256(receipt)
    result: dict[str, Any] = {"evidence": evidence, "receipt": receipt}
    if output is not None:
        observed = _write_once(output, evidence)
        if observed != evidence_payload_sha:
            raise EvidenceAnalysisError("written evidence raw hash differs from receipt")
        result["path"] = str(_absolute_without_resolve(output))
    if receipt_output is not None:
        result["receipt_raw_file_sha256"] = _write_once(receipt_output, receipt)
        result["receipt_path"] = str(_absolute_without_resolve(receipt_output))
    return result


finalize_evidence = analyze_cohort


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--gate-receipt", required=True, type=Path)
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", dest="receipt_output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze_cohort(
            args.aggregate,
            args.manifest,
            args.census,
            plan=args.plan,
            gate_receipt=args.gate_receipt,
            shards_root=args.shards_root,
            output=args.output,
            receipt_output=args.receipt_output,
        )
        print(json.dumps({
            "status": result["evidence"]["status"],
            "evidence_self_sha256": result["evidence"]["self_sha256"],
            "receipt_self_sha256": result["receipt"]["self_sha256"],
        }, sort_keys=True))
        return 0
    except (EvidenceAnalysisError, CohortContractError, ExecutionPlanError, ShardMergeError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

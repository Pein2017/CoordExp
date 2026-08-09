#!/usr/bin/env python3
"""Run the sealed S natural K/N/H cohort from an admitted event manifest.

The manifest is the only cohort authority.  This runner deliberately accepts
an injected event executor: production code can bind the existing exact
runtime, while contract tests can exercise the complete immutable/resume
surface without loading a checkpoint.  The runner never sorts events, adds an
arm, infers an eligibility threshold, or turns parser/STOP outcomes into
technical failures.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Protocol


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
SCHEMA_VERSION = "s_natural_boundary_admitted_event_manifest.v3"
RUNNER_SCHEMA_VERSION = "s_natural_boundary_k_n_h_cohort_runner.v3"
EVENT_SCHEMA_VERSION = f"{RUNNER_SCHEMA_VERSION}.event.v1"
AGGREGATE_SCHEMA_VERSION = f"{RUNNER_SCHEMA_VERSION}.aggregate.v1"
RECEIPT_SCHEMA_VERSION = f"{RUNNER_SCHEMA_VERSION}.receipt.v1"
CHECKPOINT = "S"
STEP = 2444
SUBSTRATE = "four-coordinate geo_sorted_xy"
SOURCE_CENSUS_REVISION = "census-v3"
ARM_ORDER = (
    "K00",
    "K01",
    "K10",
    "K11",
    "K12",
    "K13",
    "K14T",
    "K14B",
    "N00",
    "N01",
    "N10",
    "N20",
    "H00",
    "H10",
    "H20",
)
ARM_SET = frozenset(ARM_ORDER)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EVENT_ID_RE = re.compile(r"^gt:[0-9]+:[0-9]+$")
ROW_STATUSES = frozenset({"closure", "native_stop", "invalid", "over_continuation", "max_budget"})
STOP_REASONS = frozenset({"closure", "native_stop", "invalid", "over_continuation", "max_budget", None})
EXECUTOR_IDENTITY_SCHEMA_VERSION = "s_natural_boundary_k_n_h_executor_identity.v1"
PRE_GPU_IDENTITY_SCHEMA_VERSION = "s_natural_boundary_k_n_h_runtime_identity_binding.v1"
FULL_RUNTIME_COHORT_PREFLIGHT_EVENT_COUNT = 11
EXECUTOR_CODE_ROLES = (
    "frozen_gate", "probe", "attention_actuator", "residual_actuator",
    "cohort_validator", "live_executor", "legacy_owner_interface_runner",
    "shard_planner", "shard_runner", "merger", "analyzer_finalizer", "sealer",
)
ELIGIBILITY_PREDICATE_KEYS = frozenset({
    "checkpoint", "native_fn", "strict_complete_row", "natural_boundary_valid",
    "verified_support", "eligible_except_support", "geometry_launch_eligible",
    "covered_owner_ids_nonempty",
})
PRE_GPU_INPUT_HASH_KEYS = frozenset({
    "manifest_raw_sha256", "manifest_self_sha256", "census_v3_raw_sha256",
    "census_v3_self_sha256", "execution_plan_raw_sha256", "execution_plan_sha256",
    "config_sha256", "panel_sha256", "cohort_sha256", "h0_identity_files_sha256",
    "base_model_files_sha256", "base_model_inventory_sha256",
    "src_runtime_tree_sha256", "event_bindings_sha256",
    "authorized_event_order_sha256", "authorized_shards_sha256",
    "legacy_context_cohort_semantic_sha256",
    "legacy_context_manifest_raw_sha256",
    "legacy_context_manifest_semantic_sha256",
    "h0_image_plan_raw_sha256", "h0_image_plan_normalized_rows_sha256",
    "legacy_context_image_plan_bindings_sha256",
})
PRE_GPU_INPUT_PATH_KEYS = frozenset({
    "manifest", "census", "execution_plan", "config", "panel", "cohort", "cohort_manifest",
    "h0_root", "h0_dir", "base_model_dir",
})


class CohortContractError(ValueError):
    """Raised when a manifest, result, or immutable output is incompatible."""


class EventExecutor(Protocol):
    def __call__(self, event: Mapping[str, Any], *, arm_order: Sequence[str]) -> Mapping[str, Any]: ...


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
        raise CohortContractError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        candidate = source.resolve()
        raise CohortContractError(f"hash source is not a regular file: {candidate}")
    candidate = source.resolve(strict=True)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CohortContractError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CohortContractError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CohortContractError(f"{label} must be a non-negative integer")
    return int(value)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CohortContractError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise CohortContractError(f"{label} must be a finite number")
    return number


def _document_self_sha256(document: Mapping[str, Any]) -> str:
    body = dict(document)
    body.pop("self_sha256", None)
    return sha256_json(body)


def _validate_census_identity(value: Any) -> str:
    if not isinstance(value, Mapping) or value.get("revision") != SOURCE_CENSUS_REVISION:
        raise CohortContractError("manifest source_census is not census-v3")
    census_sha = _sha(value.get("sha256"), "source_census.sha256")
    census_path = value.get("path")
    if census_path is None:
        return census_sha
    candidate = Path(str(census_path)).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise CohortContractError("source_census.path is not a regular file")
    semantics = value.get("hash_semantics", "file_sha256")
    if semantics == "file_sha256":
        observed = sha256_file(candidate)
    elif semantics == "document_self_sha256":
        try:
            parsed = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CohortContractError(f"cannot read source_census document: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise CohortContractError("source_census document must be an object")
        observed = _document_self_sha256(parsed)
    elif semantics == "canonical_json_document_with_trailing_newline":
        try:
            raw = candidate.read_bytes()
            parsed = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CohortContractError(f"cannot read source_census document: {exc}") from exc
        if not isinstance(parsed, Mapping) or raw != canonical_json_bytes(parsed) + b"\n":
            raise CohortContractError("source_census is not canonical JSON with one trailing newline")
        observed = sha256_bytes(raw)
    else:
        raise CohortContractError(f"unsupported source_census.hash_semantics: {semantics!r}")
    if observed != census_sha:
        raise CohortContractError("source_census.path hash differs from declared sha256")
    return census_sha


def _read_json(source: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(source, (str, Path)):
        source_path = Path(source).expanduser()
        if source_path.is_symlink() or not source_path.is_file():
            path = source_path.resolve()
            raise CohortContractError(f"manifest is not a regular file: {path}")
        path = source_path.resolve(strict=True)
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CohortContractError(f"cannot read JSON manifest {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise CohortContractError("manifest must be a JSON object")
        return dict(value), {"path": str(path), "sha256": sha256_bytes(raw), "size_bytes": len(raw)}
    if not isinstance(source, Mapping):
        raise CohortContractError("manifest must be a JSON path or mapping")
    value = dict(source)
    return value, {"inline": True, "sha256": sha256_json(value), "size_bytes": None}


def _validate_source_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CohortContractError(f"{label} identity must be an object")
    identity = dict(value)
    _text(identity.get("id"), f"{label}.id")
    digest = _sha(identity.get("sha256"), f"{label}.sha256")
    if label == "cohort":
        frozen_arms = identity.get("frozen_arms")
        if not isinstance(frozen_arms, list) or tuple(frozen_arms) != ARM_ORDER:
            raise CohortContractError("cohort.frozen_arms differs from the frozen S K/N/H tuple")
    path = identity.get("path")
    if path is not None:
        candidate = Path(str(path)).expanduser()
        if candidate.is_symlink():
            raise CohortContractError(f"{label}.path must not be a symlink")
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise CohortContractError(f"{label}.path is not a regular file")
        observed = sha256_file(resolved)
        if observed != digest:
            raise CohortContractError(f"{label}.path hash differs from declared sha256")
        identity["path"] = str(resolved)
    elif label in {"cohort", "operator", "backend"}:
        identity_body = {key: item for key, item in identity.items() if key != "sha256"}
        if digest != sha256_json(identity_body):
            raise CohortContractError(f"{label}.sha256 does not bind its identity body")
    return identity


def _validate_thresholds(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise CohortContractError(f"{label} must be a non-empty threshold object")
    thresholds = dict(value)
    for key, threshold in thresholds.items():
        _text(key, f"{label} key")
        if isinstance(threshold, Mapping):
            _validate_thresholds(threshold, f"{label}.{key}")
        elif isinstance(threshold, (list, tuple)):
            for index, item in enumerate(threshold):
                _finite_number(item, f"{label}.{key}[{index}]")
        else:
            _finite_number(threshold, f"{label}.{key}")
    canonical_json_bytes(thresholds)
    return thresholds


def _validate_token_list(value: Any, label: str, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise CohortContractError(f"{label} must be a non-empty integer list")
    result: list[int] = []
    for index, token in enumerate(value):
        result.append(_nonnegative_int(token, f"{label}[{index}]"))
    return result


def _validate_opener_contract(value: Any, event_label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CohortContractError(f"{event_label}.natural_boundary.opener_token_contract is missing")
    contract = dict(value)
    expected = {
        "status": "runner_resolved",
        "resolver": "serialization_successor_runner",
        "token_name": "<|object_ref_start|>",
        "resolution_semantics": "pre_opener_natural_prefix_ends_before_object_ref_start",
    }
    if any(contract.get(key) != expected[key] for key in expected):
        raise CohortContractError(f"{event_label} opener token contract identity drifted")
    declared = _sha(contract.get("contract_sha256"), f"{event_label}.opener_token_contract.contract_sha256")
    if declared != sha256_json({"token_name": expected["token_name"], "resolver": expected["resolver"]}):
        raise CohortContractError(f"{event_label} opener token contract hash mismatch")
    return contract


def _validate_natural_boundary(value: Any, event_label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CohortContractError(f"{event_label}.natural_boundary must be an object")
    boundary = dict(value)
    if boundary.get("pre_opener_natural") is not True:
        raise CohortContractError(f"{event_label} is not a pre-opener natural boundary")
    if boundary.get("opener_seeded") is not False:
        raise CohortContractError(f"{event_label} has a seeded opener")
    if boundary.get("opener_injected") is not False:
        raise CohortContractError(f"{event_label} allows opener injection")
    if boundary.get("synthetic_opener_injections") != 0:
        raise CohortContractError(f"{event_label} permits synthetic opener injections")
    if boundary.get("opener_token_id") is not None:
        raise CohortContractError(f"{event_label}.opener_token_id must remain null until runner resolution")
    boundary["opener_token_contract"] = _validate_opener_contract(
        boundary.get("opener_token_contract"), event_label
    )
    prefix = _validate_token_list(boundary.get("prefix_token_ids"), f"{event_label}.prefix_token_ids")
    history = _validate_token_list(boundary.get("history_token_ids"), f"{event_label}.history_token_ids", allow_empty=True)
    if boundary.get("prefix_sha256") != sha256_json(prefix):
        raise CohortContractError(f"{event_label} prefix hash mismatch")
    if boundary.get("history_sha256") != sha256_json(history):
        raise CohortContractError(f"{event_label} history hash mismatch")
    boundary["prefix_token_ids"] = prefix
    boundary["history_token_ids"] = history
    return boundary


def _validate_event(
    event: Any,
    index: int,
    rule_id: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise CohortContractError(f"events[{index}] must be an object")
    item = dict(event)
    label = f"events[{index}]"
    if item.get("event_index") != index:
        raise CohortContractError(f"{label}.event_index breaks manifest order")
    event_id = _text(item.get("event_id"), f"{label}.event_id")
    if EVENT_ID_RE.fullmatch(event_id) is None:
        raise CohortContractError(f"{label}.event_id must be a gt:image:index identity")
    _nonnegative_int(item.get("image_id"), f"{label}.image_id")
    if item.get("checkpoint") != CHECKPOINT or item.get("step") != STEP or item.get("substrate") != SUBSTRATE:
        raise CohortContractError(f"{label} is not on the frozen S step-2444 substrate")
    if item.get("admission") != "admitted":
        raise CohortContractError(f"{label} is not admitted")
    owner_refs = item.get("owner_refs")
    if not isinstance(owner_refs, Mapping) or not owner_refs:
        raise CohortContractError(f"{label}.owner_refs must be non-empty")
    owner_id = _text(owner_refs.get("gt_owner_id"), f"{label}.owner_refs.gt_owner_id")
    if owner_id != event_id:
        raise CohortContractError(f"{label}.event_id differs from owner_refs.gt_owner_id")
    event_image_id = int(event_id.split(":")[1])
    if item["image_id"] != event_image_id:
        raise CohortContractError(f"{label}.image_id differs from event_id")
    for key in ("source_panel_object_index", "derived_panel_object_index"):
        _nonnegative_int(owner_refs.get(key), f"{label}.owner_refs.{key}")
    covered_owner_ids = owner_refs.get("covered_owner_ids")
    if (
        not isinstance(covered_owner_ids, list)
        or not covered_owner_ids
        or any(not isinstance(value, str) or not value for value in covered_owner_ids)
        or len(set(covered_owner_ids)) != len(covered_owner_ids)
        or owner_refs.get("covered_A_owner_id") != covered_owner_ids[-1]
    ):
        raise CohortContractError(f"{label}.owner_refs covered-owner order is invalid")
    eligibility = item.get("eligibility")
    if not isinstance(eligibility, Mapping):
        raise CohortContractError(f"{label}.eligibility must be explicit")
    if eligibility.get("admitted") is not True or eligibility.get("rule_id") != rule_id:
        raise CohortContractError(f"{label}.eligibility admission predicate is not true")
    event_thresholds = _validate_thresholds(eligibility.get("thresholds"), f"{label}.eligibility.thresholds")
    if event_thresholds != dict(thresholds):
        raise CohortContractError(f"{label}.eligibility.thresholds differ from manifest thresholds")
    predicates = eligibility.get("predicates")
    if (
        not isinstance(predicates, Mapping)
        or set(predicates) != ELIGIBILITY_PREDICATE_KEYS
        or any(value is not True for value in predicates.values())
    ):
        raise CohortContractError(f"{label}.eligibility.predicates must be the exact all-true frozen set")
    canonical_json_bytes(predicates)
    item["natural_boundary"] = _validate_natural_boundary(item.get("natural_boundary"), label)
    supplied_event_hash = _sha(item.get("event_sha256"), f"{label}.event_sha256")
    body = dict(item)
    body.pop("event_sha256", None)
    if supplied_event_hash != sha256_json(body):
        raise CohortContractError(f"{label}.event_sha256 mismatch")
    return item


def validate_manifest(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the immutable manifest identity plus ordered events."""

    document, source_info = _read_json(source)
    canonical_json_bytes(document)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise CohortContractError("manifest schema_version is not census-v3 admitted-event v3")
    if document.get("status") != "sealed" or document.get("unit_id") != UNIT_ID:
        raise CohortContractError("manifest status or unit identity drifted")
    primary = document.get("primary")
    if not isinstance(primary, Mapping) or dict(primary) != {
        "checkpoint": CHECKPOINT,
        "step": STEP,
        "substrate": SUBSTRATE,
    }:
        raise CohortContractError("manifest primary identity is not frozen S step-2444")
    arm_order = tuple(document.get("arm_order", ()))
    if arm_order != ARM_ORDER:
        raise CohortContractError("manifest arm_order differs from the frozen S K/N/H tuple")
    census_sha = _validate_census_identity(document.get("source_census"))
    source_identities = {
        key: _validate_source_identity(document.get(key), key)
        for key in ("panel", "cohort", "operator", "backend")
    }
    eligibility = document.get("eligibility")
    if not isinstance(eligibility, Mapping):
        raise CohortContractError("manifest eligibility must be explicit")
    top_level_rule_id = _text(eligibility.get("rule_id"), "eligibility.rule_id")
    thresholds = _validate_thresholds(eligibility.get("thresholds"), "eligibility.thresholds")
    admission_gate = document.get("admission_gate")
    if not isinstance(admission_gate, Mapping):
        raise CohortContractError("manifest admission_gate must be explicit")
    _text(admission_gate.get("status"), "admission_gate.status")
    _nonnegative_int(admission_gate.get("minimum_event_count"), "admission_gate.minimum_event_count")
    _nonnegative_int(admission_gate.get("minimum_image_count"), "admission_gate.minimum_image_count")
    events_raw = document.get("events")
    if not isinstance(events_raw, list) or not events_raw:
        raise CohortContractError("manifest events must be a non-empty ordered list")
    events = [
        _validate_event(
            item,
            index,
            top_level_rule_id,
            thresholds,
        )
        for index, item in enumerate(events_raw)
    ]
    ids = [item["event_id"] for item in events]
    if len(ids) != len(set(ids)):
        raise CohortContractError("manifest contains duplicate event_id")
    owner_ids = [item["owner_refs"]["gt_owner_id"] for item in events]
    if len(owner_ids) != len(set(owner_ids)):
        raise CohortContractError("manifest contains duplicate owner reference")
    supplied_self_hash = _sha(document.get("self_sha256"), "manifest.self_sha256")
    if supplied_self_hash != _document_self_sha256(document):
        raise CohortContractError("manifest self_sha256 mismatch")
    return {
        "document": document,
        "source": source_info,
        "manifest_sha256": source_info["sha256"],
        "manifest_self_sha256": supplied_self_hash,
        "source_census_sha256": census_sha,
        "source_identities": source_identities,
        "events": events,
        "arm_order": ARM_ORDER,
        "thresholds": thresholds,
        "admission_gate": dict(admission_gate),
    }


def _validate_arm_order(arms: Sequence[str]) -> tuple[str, ...]:
    observed = tuple(str(item) for item in arms)
    if observed != ARM_ORDER:
        raise CohortContractError("requested arm set/order is not the frozen S K/N/H tuple")
    return observed


def _validate_execution_qualification(manifest_info: Mapping[str, Any]) -> dict[str, Any]:
    gate = manifest_info.get("admission_gate")
    if not isinstance(gate, Mapping) or gate.get("status") != "deferred_to_successor_runner":
        raise CohortContractError("manifest admission_gate is not delegated to this successor runner")
    minimum_events = _nonnegative_int(gate.get("minimum_event_count"), "admission_gate.minimum_event_count")
    minimum_images = _nonnegative_int(gate.get("minimum_image_count"), "admission_gate.minimum_image_count")
    if minimum_events < 3 or minimum_images < 2:
        raise CohortContractError("admission_gate is below the execution qualification floor (3 events/2 images)")
    events = manifest_info["events"]
    image_count = len({event["image_id"] for event in events})
    qualified = len(events) >= minimum_events and image_count >= minimum_images
    return {
        "execution_scope": "checkpoint_replication_candidate" if qualified else "case_study",
        "checkpoint_claim_qualified": qualified,
        "static_direction_claim_qualified": qualified,
        "training_claim_qualified": False,
        "subfloor_execution_authorized": not qualified,
        "event_count": len(events),
        "image_count": image_count,
        "minimum_checkpoint_event_count": minimum_events,
        "minimum_checkpoint_image_count": minimum_images,
    }


def _finite_result(result: Mapping[str, Any], label: str) -> None:
    try:
        canonical_json_bytes(result)
    except CohortContractError as exc:
        raise CohortContractError(f"{label} is not canonical finite JSON") from exc


def _validate_full_runtime_cohort_preflight(value: Any, label: str) -> dict[str, Any]:
    """Validate the exact processor-only eleven-event preflight receipt."""

    if not isinstance(value, Mapping):
        raise CohortContractError(f"{label} must be an object")
    receipt = dict(value)
    required = {
        "status",
        "event_count",
        "event_identities_sha256",
        "authoritative_bindings_sha256",
        "processor_context_bindings",
        "processor_context_bindings_sha256",
        "cohort_path",
        "cohort_sha256",
        "cohort_manifest_path",
        "cohort_manifest_sha256",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise CohortContractError(f"{label} fields are incomplete")
    if receipt.get("status") != "passed":
        raise CohortContractError(f"{label}.status is not passed")
    if _nonnegative_int(receipt.get("event_count"), f"{label}.event_count") != FULL_RUNTIME_COHORT_PREFLIGHT_EVENT_COUNT:
        raise CohortContractError(
            f"{label}.event_count is not the exact {FULL_RUNTIME_COHORT_PREFLIGHT_EVENT_COUNT}-event preflight"
        )
    for key in (
        "event_identities_sha256",
        "authoritative_bindings_sha256",
        "processor_context_bindings_sha256",
        "cohort_sha256",
        "cohort_manifest_sha256",
    ):
        _sha(receipt.get(key), f"{label}.{key}")
    for key in ("cohort_path", "cohort_manifest_path"):
        if not Path(_text(receipt.get(key), f"{label}.{key}")).is_absolute():
            raise CohortContractError(f"{label}.{key} is not absolute")
    bindings = receipt.get("processor_context_bindings")
    if (
        not isinstance(bindings, list)
        or len(bindings) != FULL_RUNTIME_COHORT_PREFLIGHT_EVENT_COUNT
        or any(
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("event_id"), str)
            or not binding.get("event_id")
            for binding in bindings
        )
    ):
        raise CohortContractError(f"{label}.processor_context_bindings are not the exact eleven-event receipt")
    if receipt["processor_context_bindings_sha256"] != sha256_json(bindings):
        raise CohortContractError(f"{label}.processor_context_bindings_sha256 mismatch")
    _finite_result(receipt, label)
    receipt_hash = receipt["receipt_sha256"]
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    if receipt_hash != sha256_json(body):
        raise CohortContractError(f"{label}.receipt_sha256 mismatch")
    return receipt


def _validate_parse_outcomes(
    result: Mapping[str, Any], arm_label: str, opener_token_id: int
) -> dict[str, Any]:
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CohortContractError(f"{arm_label}.rows must be a full natural trajectory")
    statuses: Counter[str] = Counter()
    trajectory_tokens: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CohortContractError(f"{arm_label}.rows[{index}] must be an object")
        if row.get("row_index") != index:
            raise CohortContractError(f"{arm_label}.rows[{index}] row_index is not contiguous")
        if row.get("admission_mode") != "pre_opener_natural":
            raise CohortContractError(f"{arm_label}.rows[{index}] is not a natural admission")
        if row.get("opener_injected") is not False or row.get("synthetic_opener_injections", 0) != 0:
            raise CohortContractError(f"{arm_label}.rows[{index}] injected an opener")
        opener_generated = row.get("opener_generated_by_model")
        if not isinstance(opener_generated, bool):
            raise CohortContractError(f"{arm_label}.rows[{index}] lacks opener provenance")
        token_ids = _validate_token_list(
            row.get("token_ids"), f"{arm_label}.rows[{index}].token_ids", allow_empty=True
        )
        if opener_generated != bool(token_ids and token_ids[0] == opener_token_id):
            raise CohortContractError(f"{arm_label}.rows[{index}] opener provenance disagrees with first token")
        row_started = row.get("row_started")
        if row_started is not None and (
            not isinstance(row_started, bool) or row_started != opener_generated
        ):
            raise CohortContractError(f"{arm_label}.rows[{index}] row_started disagrees with opener provenance")
        first_generated = row.get("first_generated_token_id")
        if first_generated is not None and (
            _nonnegative_int(first_generated, f"{arm_label}.rows[{index}].first_generated_token_id")
            != (token_ids[0] if token_ids else None)
        ):
            raise CohortContractError(f"{arm_label}.rows[{index}] first-token receipt disagrees with trajectory")
        trajectory_tokens.extend(token_ids)
        if row.get("token_ids_sha256") != sha256_json(token_ids):
            raise CohortContractError(f"{arm_label}.rows[{index}] token hash mismatch")
        status = row.get("status")
        if status not in ROW_STATUSES:
            raise CohortContractError(f"{arm_label}.rows[{index}] has unknown scientific status")
        statuses[str(status)] += 1
    generated = _validate_token_list(
        result.get("generated_token_ids"), f"{arm_label}.generated_token_ids", allow_empty=True
    )
    if generated != trajectory_tokens:
        suffix = generated[len(trajectory_tokens) :]
        terminal = result.get("terminal_reason")
        if not (
            len(generated) == len(trajectory_tokens) + 1
            and len(suffix) == 1
            and rows[-1].get("status") in {"closure", "over_continuation"}
            and terminal in {"native_stop", "over_continuation"}
        ):
            raise CohortContractError(f"{arm_label}.generated_token_ids disagrees with row trajectory")
        over_continuation = rows[-1].get("over_continuation")
        if terminal == "over_continuation" and isinstance(over_continuation, Mapping):
            if over_continuation.get("selected_token_id") != suffix[0]:
                raise CohortContractError(f"{arm_label} over-continuation token disagrees with trajectory")
    nonclosure = [index for index, row in enumerate(rows) if row.get("status") != "closure"]
    if nonclosure and nonclosure != [len(rows) - 1]:
        raise CohortContractError(f"{arm_label}.rows continue after a terminal scientific outcome")
    bookkeeping = result.get("owner_bookkeeping")
    if not isinstance(bookkeeping, Mapping):
        raise CohortContractError(f"{arm_label}.owner_bookkeeping is missing")
    parse = bookkeeping.get("parse")
    stop = bookkeeping.get("stop")
    if not isinstance(parse, Mapping) or not isinstance(stop, Mapping):
        raise CohortContractError(f"{arm_label} parser/STOP outcome receipts are missing")
    parse_counts: dict[str, int] = {}
    for key in ("valid_rows", "duplicate_rows", "unmatched_rows", "ambiguous_rows", "malformed_rows", "invalid_rows"):
        parse_counts[key] = _nonnegative_int(parse.get(key), f"{arm_label}.owner_bookkeeping.parse.{key}")
    expected_valid = statuses["closure"]
    expected_malformed = statuses["over_continuation"]
    expected_invalid = statuses["invalid"] + statuses["over_continuation"] + statuses["max_budget"]
    if parse_counts["valid_rows"] != expected_valid:
        raise CohortContractError(f"{arm_label} valid-row count disagrees with trajectory")
    if parse_counts["malformed_rows"] != expected_malformed or parse_counts["invalid_rows"] != expected_invalid:
        raise CohortContractError(f"{arm_label} malformed/invalid counts disagree with trajectory")
    for key in ("duplicate_rows", "unmatched_rows", "ambiguous_rows"):
        if parse_counts[key] > expected_valid:
            raise CohortContractError(f"{arm_label} {key} count exceeds closure rows")
    stop_reason = stop.get("stop_reason")
    if stop_reason not in STOP_REASONS or not isinstance(stop.get("stopped"), bool):
        raise CohortContractError(f"{arm_label}.owner_bookkeeping.stop is malformed")
    terminal = result.get("terminal_reason")
    if terminal not in {value for value in STOP_REASONS if value is not None}:
        raise CohortContractError(f"{arm_label}.terminal_reason is malformed")
    last = rows[-1]
    row_stop_reason = last.get("stop_reason", last.get("status"))
    lookahead_terminal = rows[-1].get("status") == "closure" and terminal in {
        "native_stop",
        "over_continuation",
        "max_budget",
    }
    if (
        (row_stop_reason != terminal and not lookahead_terminal)
        or stop_reason != terminal
        or stop["stopped"] != (terminal != "closure")
    ):
        raise CohortContractError(f"{arm_label} STOP/terminal receipts disagree with trajectory")
    if isinstance(bookkeeping.get("row_count"), int) and bookkeeping.get("row_count") != len(rows):
        raise CohortContractError(f"{arm_label}.owner_bookkeeping.row_count disagrees with trajectory")
    return {
        "row_status_counts": dict(sorted(statuses.items())),
        "parse": parse_counts,
        "stop": {"stopped": bool(stop["stopped"]), "stop_reason": stop_reason},
        "terminal_reason": terminal,
    }


def _validate_parity(value: Any, arm_label: str) -> dict[str, Any]:
    """Require the complete parity receipt emitted by the scalar runtime."""

    if not isinstance(value, Mapping):
        raise CohortContractError(f"{arm_label}.full_logit_parity is missing")
    parity = dict(value)
    status = parity.get("status")
    if status not in {"reference_captured", "measured", "not_measured"}:
        raise CohortContractError(f"{arm_label}.full_logit_parity.status is unknown")
    reference_arm = parity.get("reference_arm")
    if reference_arm is not None:
        _text(reference_arm, f"{arm_label}.full_logit_parity.reference_arm")
    for key in ("reference_step_count", "candidate_step_count"):
        _nonnegative_int(parity.get(key), f"{arm_label}.full_logit_parity.{key}")
    delta = parity.get("per_forward_max_abs_delta")
    if delta is not None and _finite_number(delta, f"{arm_label}.full_logit_parity.per_forward_max_abs_delta") < 0:
        raise CohortContractError(f"{arm_label}.full_logit_parity.per_forward_max_abs_delta is negative")
    tolerance = _finite_number(parity.get("tolerance"), f"{arm_label}.full_logit_parity.tolerance")
    if tolerance < 0:
        raise CohortContractError(f"{arm_label}.full_logit_parity.tolerance is negative")
    if not isinstance(parity.get("passed"), bool):
        raise CohortContractError(f"{arm_label}.full_logit_parity.passed must be boolean")
    return parity


def _validate_executor_identity(value: Any, arm_label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CohortContractError(f"executor result {arm_label}.executor_identity is missing")
    identity = dict(value)
    if set(identity) != {
        "schema_version",
        "pre_gpu",
        "observed",
        "full_runtime_cohort_preflight",
        "identity_sha256",
    }:
        raise CohortContractError(f"executor result {arm_label}.executor_identity fields are incomplete")
    if identity.get("schema_version") != EXECUTOR_IDENTITY_SCHEMA_VERSION:
        raise CohortContractError(f"executor result {arm_label}.executor_identity schema drifted")
    pre_gpu = identity.get("pre_gpu")
    if not isinstance(pre_gpu, Mapping):
        raise CohortContractError(f"executor result {arm_label}.executor_identity.pre_gpu is missing")
    pre_gpu_doc = dict(pre_gpu)
    required_pre_gpu = {
        "schema_version", "unit_id", "checkpoint", "step", "substrate",
        "pre_gpu_receipt_path", "pre_gpu_receipt_sha256",
        "pre_gpu_receipt_self_sha256", "input_hashes", "input_paths",
        "code_hashes", "src_runtime_tree", "h0_image_plan", "runtime", "forced_math", "device_policy",
        "device_assignment",
        "claim_scope",
        "no_training", "binding_sha256",
    }
    if set(pre_gpu_doc) != required_pre_gpu:
        raise CohortContractError(f"executor result {arm_label}.executor_identity.pre_gpu fields are incomplete")
    if (
        pre_gpu_doc.get("schema_version") != PRE_GPU_IDENTITY_SCHEMA_VERSION
        or pre_gpu_doc.get("unit_id") != UNIT_ID
        or pre_gpu_doc.get("checkpoint") != CHECKPOINT
        or pre_gpu_doc.get("step") != STEP
        or pre_gpu_doc.get("substrate") != SUBSTRATE
        or pre_gpu_doc.get("no_training") is not True
    ):
        raise CohortContractError(f"executor result {arm_label}.executor_identity.pre_gpu identity drifted")
    receipt_path = _text(pre_gpu_doc.get("pre_gpu_receipt_path"), f"{arm_label}.executor_identity.pre_gpu.pre_gpu_receipt_path")
    if not Path(receipt_path).is_absolute():
        raise CohortContractError(f"executor result {arm_label}.executor_identity receipt path is not absolute")
    _sha(pre_gpu_doc.get("pre_gpu_receipt_sha256"), f"{arm_label}.executor_identity.pre_gpu.pre_gpu_receipt_sha256")
    _sha(pre_gpu_doc.get("pre_gpu_receipt_self_sha256"), f"{arm_label}.executor_identity.pre_gpu.pre_gpu_receipt_self_sha256")
    input_hashes = pre_gpu_doc.get("input_hashes")
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != PRE_GPU_INPUT_HASH_KEYS:
        raise CohortContractError(f"executor result {arm_label}.executor_identity input hashes are incomplete")
    for key, digest in input_hashes.items():
        _sha(digest, f"{arm_label}.executor_identity.pre_gpu.input_hashes.{key}")
    input_paths = pre_gpu_doc.get("input_paths")
    if not isinstance(input_paths, Mapping) or set(input_paths) != PRE_GPU_INPUT_PATH_KEYS:
        raise CohortContractError(f"executor result {arm_label}.executor_identity input paths are incomplete")
    for key, path in input_paths.items():
        if not Path(_text(path, f"{arm_label}.executor_identity.pre_gpu.input_paths.{key}")).is_absolute():
            raise CohortContractError(f"executor result {arm_label}.executor_identity input path {key} is not absolute")
    code_hashes = pre_gpu_doc.get("code_hashes")
    if not isinstance(code_hashes, Mapping) or set(code_hashes) != set(EXECUTOR_CODE_ROLES):
        raise CohortContractError(f"executor result {arm_label}.executor_identity code hashes are incomplete")
    for role, digest in code_hashes.items():
        _sha(digest, f"{arm_label}.executor_identity.pre_gpu.code_hashes.{role}")
    src_runtime_tree = pre_gpu_doc.get("src_runtime_tree")
    if not isinstance(src_runtime_tree, Mapping) or set(src_runtime_tree) != {
        "root", "file_count", "inventory_sha256"
    }:
        raise CohortContractError(
            f"executor result {arm_label}.executor_identity src runtime tree is incomplete"
        )
    if not Path(_text(src_runtime_tree.get("root"), f"{arm_label}.executor_identity.pre_gpu.src_runtime_tree.root")).is_absolute():
        raise CohortContractError(f"executor result {arm_label}.executor_identity src root is not absolute")
    _nonnegative_int(src_runtime_tree.get("file_count"), f"{arm_label}.executor_identity.pre_gpu.src_runtime_tree.file_count")
    if _sha(src_runtime_tree.get("inventory_sha256"), f"{arm_label}.executor_identity.pre_gpu.src_runtime_tree.inventory_sha256") != input_hashes["src_runtime_tree_sha256"]:
        raise CohortContractError(f"executor result {arm_label}.executor_identity src hashes differ")
    h0_image_plan = pre_gpu_doc.get("h0_image_plan")
    if not isinstance(h0_image_plan, Mapping) or set(h0_image_plan) != {
        "path", "raw_sha256", "row_count", "normalized_rows_sha256",
        "row_digests", "row_digests_sha256",
    }:
        raise CohortContractError(f"executor result {arm_label}.executor_identity H0 image plan is incomplete")
    if not Path(_text(h0_image_plan.get("path"), f"{arm_label}.executor_identity.pre_gpu.h0_image_plan.path")).is_absolute():
        raise CohortContractError(f"executor result {arm_label}.executor_identity H0 image plan path is not absolute")
    if (
        _sha(h0_image_plan.get("raw_sha256"), f"{arm_label}.executor_identity.pre_gpu.h0_image_plan.raw_sha256") != input_hashes["h0_image_plan_raw_sha256"]
        or _sha(h0_image_plan.get("normalized_rows_sha256"), f"{arm_label}.executor_identity.pre_gpu.h0_image_plan.normalized_rows_sha256") != input_hashes["h0_image_plan_normalized_rows_sha256"]
    ):
        raise CohortContractError(f"executor result {arm_label}.executor_identity H0 image plan hashes differ")
    row_count = _nonnegative_int(h0_image_plan.get("row_count"), f"{arm_label}.executor_identity.pre_gpu.h0_image_plan.row_count")
    row_digests = h0_image_plan.get("row_digests")
    if not isinstance(row_digests, list) or len(row_digests) != row_count or any(
        not isinstance(row, Mapping)
        or set(row) != {"row_index", "row_id", "sha256"}
        or row.get("row_index") != index
        or not isinstance(row.get("row_id"), str)
        or not SHA256_RE.fullmatch(str(row.get("sha256", "")))
        for index, row in enumerate(row_digests)
    ) or _sha(h0_image_plan.get("row_digests_sha256"), f"{arm_label}.executor_identity.pre_gpu.h0_image_plan.row_digests_sha256") != sha256_json(row_digests):
        raise CohortContractError(f"executor result {arm_label}.executor_identity H0 image plan rows drifted")
    if not isinstance(pre_gpu_doc.get("runtime"), Mapping) or not isinstance(pre_gpu_doc.get("forced_math"), Mapping) or not isinstance(pre_gpu_doc.get("device_policy"), Mapping):
        raise CohortContractError(f"executor result {arm_label}.executor_identity runtime policy is incomplete")
    device_policy = pre_gpu_doc["device_policy"]
    expected_physical_devices = {f"shard-{index:03d}": str(index) for index in range(8)}
    if device_policy != {
        "device_count": 1,
        "logical_device": "cuda:0",
        "shard_physical_devices": expected_physical_devices,
    }:
        raise CohortContractError(f"executor result {arm_label}.executor_identity device policy drifted")
    assignment = pre_gpu_doc.get("device_assignment")
    if not isinstance(assignment, Mapping) or set(assignment) != {
        "shard_id", "shard_index", "physical_device", "observed_cuda_visible_devices",
        "logical_device", "device_count", "authorization_sha256",
    }:
        raise CohortContractError(f"executor result {arm_label}.executor_identity device assignment is incomplete")
    shard_index = _nonnegative_int(assignment.get("shard_index"), f"{arm_label}.executor_identity.pre_gpu.device_assignment.shard_index")
    shard_id = _text(assignment.get("shard_id"), f"{arm_label}.executor_identity.pre_gpu.device_assignment.shard_id")
    physical_device = _text(assignment.get("physical_device"), f"{arm_label}.executor_identity.pre_gpu.device_assignment.physical_device")
    if (
        not 0 <= shard_index < 8
        or shard_id != f"shard-{shard_index:03d}"
        or physical_device != str(shard_index)
        or assignment.get("observed_cuda_visible_devices") != physical_device
        or assignment.get("logical_device") != "cuda:0"
        or assignment.get("device_count") != 1
        or expected_physical_devices.get(shard_id) != physical_device
    ):
        raise CohortContractError(f"executor result {arm_label}.executor_identity device assignment drifted")
    _sha(assignment.get("authorization_sha256"), f"{arm_label}.executor_identity.pre_gpu.device_assignment.authorization_sha256")
    claim_scope = pre_gpu_doc.get("claim_scope")
    if not isinstance(claim_scope, Mapping) or set(claim_scope) != {
        "execution_scope", "event_count", "image_count",
        "minimum_checkpoint_event_count", "minimum_checkpoint_image_count",
        "checkpoint_claim_qualified", "static_direction_claim_qualified",
        "training_claim_qualified", "subfloor_execution_authorized",
    }:
        raise CohortContractError(f"executor result {arm_label}.executor_identity claim scope is incomplete")
    checkpoint_qualified = (
        claim_scope.get("event_count", 0) >= 3 and claim_scope.get("image_count", 0) >= 2
        if isinstance(claim_scope.get("event_count"), int) and isinstance(claim_scope.get("image_count"), int)
        else None
    )
    if (
        checkpoint_qualified is None
        or claim_scope.get("minimum_checkpoint_event_count") != 3
        or claim_scope.get("minimum_checkpoint_image_count") != 2
        or claim_scope.get("checkpoint_claim_qualified") is not checkpoint_qualified
        or claim_scope.get("static_direction_claim_qualified") is not checkpoint_qualified
        or claim_scope.get("training_claim_qualified") is not False
        or claim_scope.get("subfloor_execution_authorized") is not (not checkpoint_qualified)
        or claim_scope.get("execution_scope") != (
            "checkpoint_replication_candidate" if checkpoint_qualified else "case_study"
        )
    ):
        raise CohortContractError(f"executor result {arm_label}.executor_identity claim scope drifted")
    binding_sha = _sha(pre_gpu_doc.get("binding_sha256"), f"{arm_label}.executor_identity.pre_gpu.binding_sha256")
    pre_gpu_body = dict(pre_gpu_doc)
    pre_gpu_body.pop("binding_sha256", None)
    if binding_sha != sha256_json(pre_gpu_body):
        raise CohortContractError(f"executor result {arm_label}.executor_identity pre-GPU hash mismatch")
    _validate_full_runtime_cohort_preflight(
        identity.get("full_runtime_cohort_preflight"),
        f"{arm_label}.executor_identity.full_runtime_cohort_preflight",
    )
    observed = identity.get("observed")
    if not isinstance(observed, Mapping) or set(observed) != {"model", "backend", "device", "cuda", "config_sha256", "runtime_versions"}:
        raise CohortContractError(f"executor result {arm_label}.executor_identity observed evidence is incomplete")
    model = observed.get("model")
    required_model = {
        "checkpoint", "step", "substrate", "h0_dir", "base_model_path",
        "adapter_path", "embedding_delta_path", "resolved_config_sha256",
        "h0_identity_files_sha256", "base_model_files_sha256",
        "base_model_inventory_sha256",
    }
    if not isinstance(model, Mapping) or set(model) != required_model:
        raise CohortContractError(f"executor result {arm_label}.executor_identity model evidence is incomplete")
    if model.get("checkpoint") != CHECKPOINT or model.get("step") != STEP or model.get("substrate") != SUBSTRATE:
        raise CohortContractError(f"executor result {arm_label}.executor_identity model identity drifted")
    for key in ("h0_dir", "base_model_path", "adapter_path", "embedding_delta_path"):
        _text(model.get(key), f"{arm_label}.executor_identity.observed.model.{key}")
    if model.get("h0_dir") != input_paths["h0_dir"] or model.get("base_model_path") != input_paths["base_model_dir"]:
        raise CohortContractError(f"executor result {arm_label}.executor_identity model paths differ from pre-GPU binding")
    for key in (
        "resolved_config_sha256", "h0_identity_files_sha256",
        "base_model_files_sha256", "base_model_inventory_sha256",
    ):
        _sha(model.get(key), f"{arm_label}.executor_identity.observed.model.{key}")
    if (
        model.get("h0_identity_files_sha256") != input_hashes["h0_identity_files_sha256"]
        or model.get("base_model_files_sha256") != input_hashes["base_model_files_sha256"]
        or model.get("base_model_inventory_sha256") != input_hashes["base_model_inventory_sha256"]
    ):
        raise CohortContractError(f"executor result {arm_label}.executor_identity model inventories differ")
    backend = observed.get("backend")
    if not isinstance(backend, Mapping) or set(backend) != {
        "type", "session_class", "model_dtype", "attention_implementation",
        "generation", "model_training",
    }:
        raise CohortContractError(f"executor result {arm_label}.executor_identity backend evidence is incomplete")
    generation = backend.get("generation")
    if (
        backend.get("type") != "hf"
        or backend.get("session_class") != "src.inference.hf_backend.HFBackendSession"
        or backend.get("model_dtype") != "fp32"
        or backend.get("attention_implementation") != "sdpa"
        or backend.get("model_training") is not False
        or not isinstance(generation, Mapping)
        or generation != {"mode": "greedy", "temperature": 0.0, "top_p": 1.0}
    ):
        raise CohortContractError(f"executor result {arm_label}.executor_identity backend contract drifted")
    if _text(observed.get("device"), f"{arm_label}.executor_identity.observed.device") != "cuda:0":
        raise CohortContractError(f"executor result {arm_label}.executor_identity logical device drifted")
    cuda = observed.get("cuda")
    if not isinstance(cuda, Mapping) or cuda.get("passed") is not True:
        raise CohortContractError(f"executor result {arm_label}.executor_identity CUDA attestation did not pass")
    visible = cuda.get("cuda_visible_devices")
    if (
        cuda.get("logical_model_device") != "cuda:0"
        or not isinstance(visible, Mapping)
        or visible.get("raw") != physical_device
        or visible.get("tokens") != [physical_device]
        or visible.get("selected_physical_device") != physical_device
    ):
        raise CohortContractError(f"executor result {arm_label}.executor_identity CUDA assignment drifted")
    config_sha = _sha(observed.get("config_sha256"), f"{arm_label}.executor_identity.observed.config_sha256")
    if config_sha != input_hashes["config_sha256"]:
        raise CohortContractError(f"executor result {arm_label}.executor_identity config hashes differ")
    runtime_versions = observed.get("runtime_versions")
    if not isinstance(runtime_versions, Mapping) or set(runtime_versions) != {
        "python_version", "torch_version", "transformers_version"
    } or any(pre_gpu_doc["runtime"].get(key) != value for key, value in runtime_versions.items()):
        raise CohortContractError(f"executor result {arm_label}.executor_identity runtime versions drifted")
    identity_sha = _sha(identity.get("identity_sha256"), f"{arm_label}.executor_identity.identity_sha256")
    identity_body = dict(identity)
    identity_body.pop("identity_sha256", None)
    if identity_sha != sha256_json(identity_body):
        raise CohortContractError(f"executor result {arm_label}.executor_identity hash mismatch")
    return identity


def validate_arm_result(result: Any, arm: str) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise CohortContractError(f"executor result for {arm} must be an object")
    value = dict(result)
    _finite_result(value, f"executor result {arm}")
    if value.get("arm_id") != arm:
        raise CohortContractError(f"executor result arm_id mismatch for {arm}")
    if value.get("admission_mode") != "pre_opener_natural":
        raise CohortContractError(f"executor result {arm} did not use natural pre-opener admission")
    if value.get("opener_injected") is not False or value.get("synthetic_opener_injections") != 0:
        raise CohortContractError(f"executor result {arm} injected an opener")
    resolved_opener_token_id = _nonnegative_int(
        value.get("opener_token_id"), f"executor result {arm}.opener_token_id"
    )
    outcomes = _validate_parse_outcomes(value, arm, resolved_opener_token_id)
    for index, row in enumerate(value["rows"]):
        if row.get("opener_token_id") != resolved_opener_token_id:
            raise CohortContractError(f"executor result {arm}.rows[{index}] opener identity mismatch")
    if value.get("no_cache_scalar_recompute") is not True:
        raise CohortContractError(f"executor result {arm} is not a full no-cache natural trajectory")
    generated = _validate_token_list(value.get("generated_token_ids"), f"{arm}.generated_token_ids", allow_empty=True)
    opener_generated = value.get("opener_generated_by_model")
    if not isinstance(opener_generated, bool):
        raise CohortContractError(f"executor result {arm} lacks opener provenance")
    first_row_tokens = value["rows"][0]["token_ids"]
    if opener_generated != bool(first_row_tokens and first_row_tokens[0] == resolved_opener_token_id):
        raise CohortContractError(f"executor result {arm} opener provenance disagrees with first row")
    first_generated = value.get("first_generated_token_id")
    if first_generated is not None and (
        _nonnegative_int(first_generated, f"executor result {arm}.first_generated_token_id")
        != (generated[0] if generated else None)
    ):
        raise CohortContractError(f"executor result {arm} first-token receipt disagrees with trajectory")
    if value.get("generated_token_ids_sha256") != sha256_json(generated):
        raise CohortContractError(f"executor result {arm} generated-token hash mismatch")
    _nonnegative_int(value.get("scalar_forward_count"), f"{arm}.scalar_forward_count")
    receipts = value.get("scalar_receipts")
    runtime_receipts = value.get("runtime_scalar_receipts")
    runtime_count = _nonnegative_int(
        value.get("runtime_scalar_forward_count"), f"{arm}.runtime_scalar_forward_count"
    )
    if (
        not isinstance(receipts, list)
        or not isinstance(runtime_receipts, list)
        or len(receipts) != len(runtime_receipts)
        or value["scalar_forward_count"] != len(receipts)
        or runtime_count != len(runtime_receipts)
        or not receipts
    ):
        raise CohortContractError(f"executor result {arm} lacks complete scalar trajectory receipts")
    for receipt_list, label in ((receipts, "scalar_receipts"), (runtime_receipts, "runtime_scalar_receipts")):
        for index, receipt in enumerate(receipt_list):
            if not isinstance(receipt, Mapping) or receipt.get("use_cache") is not False:
                raise CohortContractError(f"executor result {arm}.{label}[{index}] is not no-cache")
    parity = _validate_parity(value.get("full_logit_parity"), arm)
    if parity["status"] == "reference_captured" and parity["reference_step_count"] != value["scalar_forward_count"]:
        raise CohortContractError(f"executor result {arm} parity/reference count disagrees with scalar trajectory")
    if parity["status"] == "measured":
        if parity["candidate_step_count"] != value["scalar_forward_count"] or parity["passed"] is not True:
            raise CohortContractError(f"executor result {arm} measured parity is incomplete or failed")
    executor_identity = _validate_executor_identity(value.get("executor_identity"), arm)
    return {
        "result": value,
        "outcomes": outcomes,
        "generated_token_count": len(generated),
        "resolved_opener_token_id": resolved_opener_token_id,
        "parity": parity,
        "executor_identity": executor_identity,
    }


def _normalize_executor_output(raw: Mapping[str, Any], event_id: str) -> Mapping[str, Any]:
    if "result" in raw and isinstance(raw.get("result"), Mapping):
        raw = raw["result"]
    if "arms" in raw:
        arms = raw.get("arms")
    else:
        arms = raw
    if not isinstance(arms, Mapping):
        raise CohortContractError(f"executor output for {event_id} lacks arms mapping")
    return arms


def _event_result_document(
    manifest_info: Mapping[str, Any],
    event: Mapping[str, Any],
    arms: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    resolved_opener_token_ids: Mapping[str, int],
) -> dict[str, Any]:
    executor_identity = arms[ARM_ORDER[0]].get("executor_identity")
    body = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "primary": {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE},
        "manifest_self_sha256": manifest_info["manifest_self_sha256"],
        "manifest_sha256": manifest_info["manifest_sha256"],
        "source_census_sha256": manifest_info["source_census_sha256"],
        "source_identities": manifest_info["source_identities"],
        "arm_order": list(ARM_ORDER),
        "event": dict(event),
        "event_sha256": event["event_sha256"],
        "opener_token_contract": event["natural_boundary"]["opener_token_contract"],
        "resolved_opener_token_ids": dict(resolved_opener_token_ids),
        "arms": {arm: arms[arm] for arm in ARM_ORDER},
        "scientific_outcomes": {arm: outcomes[arm] for arm in ARM_ORDER},
        "executor_identity": executor_identity,
        "natural_boundary_no_seed": True,
        "full_natural_trajectory": True,
        "no_event_reorder": True,
        "no_sweep": True,
        "no_a3": True,
        "no_2x2": True,
        "no_p4": True,
    }
    document = dict(body)
    document["result_sha256"] = sha256_json(body)
    _finite_result(document, f"event {event['event_id']} result")
    return document


def _write_once(path: Path, document: Mapping[str, Any]) -> bool:
    payload = canonical_json_bytes(document) + b"\n"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"immutable output collision: {path}")
        return True
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
    return False


def _load_completed_event(root: Path, *, event: Mapping[str, Any], manifest_info: Mapping[str, Any]) -> dict[str, Any] | None:
    result_path = root / "result.json"
    terminal_path = root / "terminal_summary.json"
    if not root.exists():
        return None
    if (
        root.is_symlink()
        or not root.is_dir()
        or result_path.is_symlink()
        or terminal_path.is_symlink()
        or not result_path.is_file()
        or not terminal_path.is_file()
    ):
        raise CohortContractError(f"event root is partial or non-regular: {root}")
    if {child.name for child in root.iterdir()} != {"result.json", "terminal_summary.json"}:
        raise CohortContractError(f"event root {root} contains an unexpected file")
    try:
        result_raw = result_path.read_bytes()
        terminal_raw = terminal_path.read_bytes()
        result = json.loads(result_raw)
        terminal = json.loads(terminal_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CohortContractError(f"cannot resume event root {root}: {exc}") from exc
    if not isinstance(result, Mapping) or not isinstance(terminal, Mapping):
        raise CohortContractError(f"event root {root} receipts must be objects")
    if terminal_raw != canonical_json_bytes(terminal) + b"\n":
        raise CohortContractError(f"event root {root} terminal receipt is not canonical JSON")
    terminal_body = dict(terminal)
    terminal_self_sha256 = terminal_body.pop("self_sha256", None)
    if (
        _sha(terminal_self_sha256, f"event root {root} terminal_summary.self_sha256")
        != sha256_json(terminal_body)
        or set(terminal) != {
            "schema_version",
            "status",
            "unit_id",
            "event_index",
            "event_id",
            "event_sha256",
            "manifest_self_sha256",
            "manifest_sha256",
            "arm_order",
            "result_sha256",
            "self_sha256",
        }
    ):
        raise CohortContractError(f"event root {root} terminal receipt self hash mismatch")
    if (
        result.get("schema_version") != EVENT_SCHEMA_VERSION
        or result.get("unit_id") != UNIT_ID
        or result.get("primary")
        != {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE}
        or result.get("arm_order") != list(ARM_ORDER)
        or result.get("event") != dict(event)
        or result.get("event_sha256") != event["event_sha256"]
        or result.get("manifest_self_sha256") != manifest_info["manifest_self_sha256"]
        or result.get("manifest_sha256") != manifest_info["manifest_sha256"]
        or result.get("source_census_sha256") != manifest_info["source_census_sha256"]
        or result.get("source_identities") != manifest_info["source_identities"]
    ):
        raise CohortContractError(f"event root {root} binds a different manifest/event")
    if any(result.get(flag) is not True for flag in (
        "natural_boundary_no_seed",
        "full_natural_trajectory",
        "no_event_reorder",
        "no_sweep",
        "no_a3",
        "no_2x2",
        "no_p4",
    )):
        raise CohortContractError(f"event root {root} has an incompatible execution flag")
    arm_values = result.get("arms")
    if not isinstance(arm_values, Mapping) or set(arm_values) != ARM_SET:
        raise CohortContractError(f"event root {root} has an incompatible arm set/order")
    validated = {arm: validate_arm_result(arm_values[arm], arm) for arm in ARM_ORDER}
    executor_identities = [validated[arm]["executor_identity"] for arm in ARM_ORDER]
    if any(identity != executor_identities[0] for identity in executor_identities[1:]):
        raise CohortContractError(f"event root {root} executor identities differ across arms")
    if result.get("executor_identity") != executor_identities[0]:
        raise CohortContractError(f"event root {root} executor identity receipt mismatch")
    resolved_opener_token_ids = {
        arm: validated[arm]["resolved_opener_token_id"] for arm in ARM_ORDER
    }
    if len(set(resolved_opener_token_ids.values())) != 1 or result.get("resolved_opener_token_ids") != resolved_opener_token_ids:
        raise CohortContractError(f"event root {root} has incompatible opener resolution")
    expected_outcomes = {arm: validated[arm]["outcomes"] for arm in ARM_ORDER}
    if result.get("scientific_outcomes") != expected_outcomes:
        raise CohortContractError(f"event root {root} scientific outcome receipt mismatch")
    body = dict(result)
    result_hash = body.pop("result_sha256", None)
    if result_hash != sha256_json(body):
        raise CohortContractError(f"event root {root} result hash mismatch")
    if (
        terminal.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or terminal.get("status") != "completed"
        or terminal.get("unit_id") != UNIT_ID
        or terminal.get("event_index") != event["event_index"]
        or terminal.get("event_id") != event["event_id"]
        or terminal.get("event_sha256") != event["event_sha256"]
        or terminal.get("manifest_self_sha256") != manifest_info["manifest_self_sha256"]
        or terminal.get("manifest_sha256") != manifest_info["manifest_sha256"]
        or terminal.get("arm_order") != list(ARM_ORDER)
        or terminal.get("result_sha256") != result_hash
    ):
        raise CohortContractError(f"event root {root} terminal receipt mismatch")
    return dict(result)


def _validate_output_contents(root: Path, *, manifest_info: Mapping[str, Any], mode: str) -> None:
    """Reject foreign roots/files before resuming or admitting new execution."""

    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise CohortContractError(f"aggregate output root is not a regular directory: {root}")
    expected_events = {
        _event_root_name(event["event_index"], event["event_id"])
        for event in manifest_info["events"]
    } if mode == "execute" else set()
    allowed = {"aggregate.json", "aggregate.receipt.json"} | expected_events
    for child in root.iterdir():
        if child.name not in allowed:
            raise CohortContractError(f"cohort output contains an unexpected file or foreign event root: {child}")
        if child.name in expected_events and (child.is_symlink() or not child.is_dir()):
            raise CohortContractError(f"cohort event root is not a regular directory: {child}")
        if child.name in {"aggregate.json", "aggregate.receipt.json"} and child.is_symlink():
            raise CohortContractError(f"cohort aggregate receipt is a symlink: {child}")


def _event_root_name(index: int, event_id: str) -> str:
    return f"event-{index:06d}-{event_id.replace(':', '-') }"


def run_cohort(
    manifest: str | Path | Mapping[str, Any],
    output_root: str | Path,
    *,
    executor: EventExecutor | None,
    arms: Sequence[str] = ARM_ORDER,
    mode: str = "execute",
) -> dict[str, Any]:
    """Validate and execute the ordered S cohort, or write a contract receipt."""

    manifest_info = validate_manifest(manifest)
    _validate_arm_order(arms)
    if mode not in {"execute", "contract"}:
        raise CohortContractError("mode must be execute or contract")
    if mode == "execute" and executor is None:
        raise CohortContractError("execute mode requires an injected event executor")
    execution_qualification = _validate_execution_qualification(manifest_info)
    root_source = Path(output_root).expanduser()
    if root_source.is_symlink():
        raise CohortContractError(f"aggregate output root is a symlink: {root_source}")
    root = root_source.resolve()
    if root.exists() and not root.is_dir():
        raise CohortContractError(f"aggregate output root is not a regular directory: {root}")
    _validate_output_contents(root, manifest_info=manifest_info, mode=mode)
    event_refs: list[dict[str, Any]] = []
    if mode == "execute":
        assert executor is not None
        for event in manifest_info["events"]:
            event_root = root / _event_root_name(event["event_index"], event["event_id"])
            existing = _load_completed_event(event_root, event=event, manifest_info=manifest_info)
            if existing is not None:
                event_document = existing
            else:
                raw = executor(event, arm_order=ARM_ORDER)
                if not isinstance(raw, Mapping):
                    raise CohortContractError(f"executor output for {event['event_id']} must be an object")
                arm_values = _normalize_executor_output(raw, event["event_id"])
                if tuple(arm_values) != ARM_ORDER:
                    raise CohortContractError(f"executor arm set/order differs for {event['event_id']}")
                validated = {arm: validate_arm_result(arm_values[arm], arm) for arm in ARM_ORDER}
                executor_identities = [validated[arm]["executor_identity"] for arm in ARM_ORDER]
                if any(identity != executor_identities[0] for identity in executor_identities[1:]):
                    raise CohortContractError(
                        f"executor identities differ across arms for {event['event_id']}"
                    )
                resolved_opener_token_ids = {
                    arm: validated[arm]["resolved_opener_token_id"] for arm in ARM_ORDER
                }
                if len(set(resolved_opener_token_ids.values())) != 1:
                    raise CohortContractError(
                        f"executor opener token resolution differs across arms for {event['event_id']}"
                    )
                event_document = _event_result_document(
                    manifest_info,
                    event,
                    {arm: validated[arm]["result"] for arm in ARM_ORDER},
                    {arm: validated[arm]["outcomes"] for arm in ARM_ORDER},
                    resolved_opener_token_ids,
                )
                _write_once(event_root / "result.json", event_document)
                terminal = {
                    "schema_version": RECEIPT_SCHEMA_VERSION,
                    "status": "completed",
                    "unit_id": UNIT_ID,
                    "event_index": event["event_index"],
                    "event_id": event["event_id"],
                    "event_sha256": event["event_sha256"],
                    "manifest_self_sha256": manifest_info["manifest_self_sha256"],
                    "manifest_sha256": manifest_info["manifest_sha256"],
                    "arm_order": list(ARM_ORDER),
                    "result_sha256": event_document["result_sha256"],
                }
                terminal["self_sha256"] = sha256_json(terminal)
                _write_once(event_root / "terminal_summary.json", terminal)
            event_refs.append(
                {
                    "event_index": event["event_index"],
                    "event_id": event["event_id"],
                    "image_id": event["image_id"],
                    "event_sha256": event["event_sha256"],
                    "result_sha256": event_document["result_sha256"],
                    "root": event_root.relative_to(root).as_posix(),
                }
            )
    if mode == "contract":
        execution_qualification = dict(execution_qualification)
    body = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "completed" if mode == "execute" else "contract_validated",
        "mode": mode,
        "primary": {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE},
        "manifest_self_sha256": manifest_info["manifest_self_sha256"],
        "manifest_sha256": manifest_info["manifest_sha256"],
        "source_census_sha256": manifest_info["source_census_sha256"],
        "source_identities": manifest_info["source_identities"],
        "arm_order": list(ARM_ORDER),
        "event_count": len(manifest_info["events"]),
        "execution_qualification": execution_qualification,
        "events": event_refs,
        "no_event_reorder": True,
        "no_sweep": True,
        "no_a3": True,
        "no_2x2": True,
        "no_p4": True,
    }
    aggregate = dict(body)
    aggregate["aggregate_sha256"] = sha256_json(body)
    _write_once(root / "aggregate.json", aggregate)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": body["status"],
        "unit_id": UNIT_ID,
        "mode": mode,
        "manifest_self_sha256": manifest_info["manifest_self_sha256"],
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "event_count": len(event_refs) if mode == "execute" else len(manifest_info["events"]),
        "distinct_image_count": execution_qualification["image_count"],
        "execution_qualification": execution_qualification,
        "arm_order": list(ARM_ORDER),
        "event_roots": event_refs,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    _write_once(root / "aggregate.receipt.json", receipt)
    return {"aggregate": aggregate, "receipt": receipt, "manifest": manifest_info}


def load_executor(spec: str) -> EventExecutor:
    if ":" not in spec:
        raise CohortContractError("executor must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise CohortContractError(f"executor is not callable: {spec}")
    return function


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("contract", "execute"), default="contract")
    parser.add_argument("--executor", default=None, help="module:function event executor for execute mode")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        executor = load_executor(args.executor) if args.mode == "execute" and args.executor else None
        result = run_cohort(args.manifest, args.output_root, executor=executor, mode=args.mode)
        print(json.dumps({"status": result["aggregate"]["status"], "aggregate_sha256": result["aggregate"]["aggregate_sha256"]}, sort_keys=True))
        return 0
    except (CohortContractError, FileExistsError, OSError, ImportError, AttributeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

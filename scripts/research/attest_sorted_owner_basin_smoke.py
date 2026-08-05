#!/usr/bin/env python3
"""Seal control-only calibration and mechanically attest the Task-4 smoke.

The control summary is fully validated and the calibration receipt is written
before an optional sentinel summary is opened.  A sentinel is case-level,
remains unresolved, and can never alter a rule or calibration value.  This
script emits gate states and a declared mechanical disposition, never a
scientific conclusion.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import sorted_owner_basin_landscape as core  # noqa: E402
from scripts.research.summarize_sorted_owner_basin_landscape import (  # noqa: E402
    CACHE_PARITY_ATOL,
    CACHE_PARITY_RTOL,
    DECISION_BEARING_SCORE_USE,
    FULL_REFORWARD_SCORING_BACKEND,
    KV_CACHE_SCORING_BACKEND,
    RECEIPT_SCHEMA_VERSION as SUMMARY_RECEIPT_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    sha256_file,
    sha256_json,
)


CALIBRATION_SCHEMA_VERSION = "sorted_owner_basin_landscape_calibration_receipt.v2"
NON_C_SMOKE_FREEZE_SCHEMA_VERSION = "sorted_owner_basin_non_c_smoke_freeze_receipt.v1"
LEAD_REVIEW_SCHEMA_VERSION = "sorted_owner_basin_landscape_lead_review_receipt.v2"
CONTROL_REGISTRY_SCHEMA_VERSION = "sorted-owner-basin-control-registry.v2"
SENTINEL_REGISTRY_SCHEMA_VERSION = "sorted-owner-basin-sentinel-registry.v2"
SENTINEL_SELECTION_SCHEMA_VERSION = "sorted-owner-basin-sentinel-selection-receipt.v1"
SENTINEL_CONFIRMATION_SCHEMA_VERSION = "sorted-owner-basin-sentinel-selection-confirmation-receipt.v2"
CALIBRATION_NAME = "landscape-calibration-receipt.json"
NON_C_SMOKE_FREEZE_NAME = "non-c-smoke-freeze-receipt.json"
LEAD_REVIEW_NAME = "lead-review-receipt.json"
DISPOSITIONS = frozenset({"proceed_full", "proceed_census_only", "narrow", "hold"})
REQUIRED_SMOKE_CONTROLS = (
    "control:smoke:strict-visible:7511:22",
    "control:smoke:b1:7511:26",
    "control:smoke:b2:7511:22-to-26",
)


class SmokeAttestationError(RuntimeError):
    """A sealed smoke/calibration precondition was not proven."""


def _fail(message: str) -> NoReturn:
    raise SmokeAttestationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{label} must be a JSON array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(f"{label} must be a finite number")
    return float(value)


def _read_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.resolve(strict=True).read_text(encoding="utf-8")), str(path))


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    except FileExistsError:
        _fail(f"refusing to overwrite immutable output: {path}")


def _validate_summary(
    summary_path: Path, receipt_path: Path, *, rules: core.LandscapeRules, rules_file_sha256: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    summary = _read_json(summary_path)
    receipt = _read_json(receipt_path)
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        _fail(f"summary.schema_version must be {SUMMARY_SCHEMA_VERSION!r}")
    if receipt.get("schema_version") != SUMMARY_RECEIPT_SCHEMA_VERSION:
        _fail(f"summary receipt.schema_version must be {SUMMARY_RECEIPT_SCHEMA_VERSION!r}")
    for payload, label in ((summary, "summary"), (receipt, "summary receipt")):
        if payload.get("rule_digest") != rules.rule_digest:
            _fail(f"{label} has a stale core rule digest")
        if payload.get("rules_file_sha256") != rules_file_sha256:
            _fail(f"{label} has a stale decision-rule file digest")
        if payload.get("scientific_conclusion") is not None:
            _fail(f"{label} must not contain a scientific conclusion")
    if receipt.get("summary_payload_sha256") != sha256_json(summary):
        _fail("summary payload digest does not independently reconstruct")
    if receipt.get("independent_reconstruction") != "passed":
        _fail("summary receipt does not attest independent reconstruction")
    _summary_scoring_backend_gate(summary, receipt)
    return summary, receipt


def _summary_scoring_backend_gate(
    summary: Mapping[str, Any], receipt: Mapping[str, Any]
) -> str:
    admission = _mapping(
        receipt.get("scoring_backend_admission"),
        "summary receipt.scoring_backend_admission",
    )
    if summary.get("scoring_backend_admission") != admission:
        _fail("summary payload and receipt bind different scoring backend admissions")
    admission_without_digest = dict(admission)
    admission_digest = admission_without_digest.pop("sha256", None)
    if _sha256(admission_digest, "summary scoring backend admission sha256") != sha256_json(
        admission_without_digest
    ):
        _fail("summary scoring backend admission digest does not reconstruct")
    if admission.get("status") != "passed" or admission.get("backend_mixing_detected") is not False:
        _fail("summary scoring backend admission did not prove one unmixed backend")
    decision_use = admission.get("decision_use")
    cache_policy = admission.get("cache_admission_policy")
    effective_mode = (
        cache_policy.get("effective_mode")
        if isinstance(cache_policy, Mapping)
        else None
    )
    if decision_use is None:
        if effective_mode == "relaxed_coordinate_behavior":
            _fail("legacy relaxed-cache summary cannot enter a decision-bearing attestation")
        decision_use = DECISION_BEARING_SCORE_USE
    if decision_use != DECISION_BEARING_SCORE_USE:
        _fail("probe-only scoring backend cannot enter a decision-bearing attestation")
    if admission.get("atol") != CACHE_PARITY_ATOL or admission.get("rtol") != CACHE_PARITY_RTOL:
        _fail("summary scoring backend admission changed the frozen parity tolerance")
    parity_status = admission.get("parity_status")
    if receipt.get("mandatory_cache_parity_gate") != parity_status:
        _fail("summary receipt relabels the mandatory cache parity result")
    selected = admission.get("selected_backend")
    if admission.get("all_score_rows_backend") != selected:
        _fail("summary scoring backend admission does not bind every row to one backend")
    score_row_count = admission.get("score_row_count")
    cache_row_count = admission.get("cache_score_row_count")
    uncached_row_count = admission.get("uncached_score_row_count")
    if (
        isinstance(score_row_count, bool)
        or not isinstance(score_row_count, int)
        or score_row_count < 0
        or isinstance(cache_row_count, bool)
        or not isinstance(cache_row_count, int)
        or isinstance(uncached_row_count, bool)
        or not isinstance(uncached_row_count, int)
        or cache_row_count + uncached_row_count != score_row_count
    ):
        _fail("summary scoring backend row accounting is incomplete")
    if parity_status == "passed":
        expected_gate = "cache_parity_passed"
        if (
            selected != KV_CACHE_SCORING_BACKEND
            or admission.get("cache_enabled") is not True
            or admission.get("use_cache") is not True
            or cache_row_count != score_row_count
            or uncached_row_count != 0
        ):
            _fail("passed cache parity did not select the admitted cache backend")
    elif parity_status == "failed":
        expected_gate = "cache_parity_failed_uncached_reference_used"
        if (
            selected != FULL_REFORWARD_SCORING_BACKEND
            or admission.get("cache_enabled") is not False
            or admission.get("use_cache") is not False
            or cache_row_count != 0
            or uncached_row_count != score_row_count
        ):
            _fail("failed cache parity did not select the admitted uncached reference backend")
    else:
        _fail("summary scoring backend admission has unknown parity status")
    if receipt.get("scoring_backend_gate") != expected_gate:
        _fail("summary receipt scoring backend gate does not match its admission")
    return expected_gate


def _validate_control_registry(
    registry: Mapping[str, Any], *, registry_sha256: str, sealed_inputs: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    if registry.get("schema_version") != CONTROL_REGISTRY_SCHEMA_VERSION:
        _fail(f"control registry schema must be {CONTROL_REGISTRY_SCHEMA_VERSION!r}")
    if registry.get("status") != "lead_frozen_before_scoring_and_resealed_to_final_task0_v2":
        _fail("control registry was not resealed to the final ambiguity-neutral Task0-v2 chain")
    if _sha256(sealed_inputs.get("control_registry_sha256"), "rules sealed control registry digest") != registry_sha256:
        _fail("control-registry.json digest does not match the frozen decision rules")
    controls: dict[str, Mapping[str, Any]] = {}
    for untyped in _sequence(registry.get("controls"), "control registry.controls"):
        item = _mapping(untyped, "control registry control")
        control_id = _string(item.get("control_id"), "control_id")
        if control_id in controls:
            _fail(f"duplicate control_id in registry: {control_id}")
        controls[control_id] = item
    if tuple(control_id for control_id in REQUIRED_SMOKE_CONTROLS if control_id in controls) != REQUIRED_SMOKE_CONTROLS:
        _fail(f"control registry must contain the exact Task-4 smoke controls: {REQUIRED_SMOKE_CONTROLS}")
    if registry.get("selection_rules", {}).get("no_c_outcome_use") is not True:
        _fail("control registry does not attest no-C-outcome selection")
    source = _mapping(registry.get("source_digests"), "control registry.source_digests")
    for key in (
        "task0_census_artifact_manifest_sha256",
        "task0_execution_receipt_content_sha256",
        "task0_execution_receipt_file_sha256",
        "owner_ledger_sha256",
        "owner_trajectory_matrix_sha256",
        "sentinel_selection_confirmation_receipt_sha256",
    ):
        if _sha256(source.get(key), f"control registry {key}") != _sha256(
            sealed_inputs.get(key), f"rules sealed {key}"
        ):
            _fail(f"control registry {key} does not match the final Task0-v2 chain frozen in rules")
    return controls


def _b2_matched_physical_suppression(
    *,
    before_entry: Mapping[str, Any],
    after_entry: Mapping[str, Any],
    b2_control: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    """Evaluate B2 only against prospectively bound physical-owner controls."""

    same_description_owner = b2_control.get(
        "matched_same_description_nonoverlap_gt_owner_id"
    )
    if not isinstance(same_description_owner, str) or not same_description_owner:
        return "failed", "required_matched_physical_foil_absent", []
    if same_description_owner == b2_control.get("target_gt_owner_id"):
        return "failed", "matched_physical_foil_reuses_target_owner", []
    if same_description_owner == b2_control.get("covering_gt_owner_id"):
        return "failed", "matched_physical_foil_reuses_covering_owner", []

    required_owner_ids = [same_description_owner]
    different_description_owner = b2_control.get(
        "matched_different_description_gt_owner_id"
    )
    if different_description_owner is not None:
        if (
            not isinstance(different_description_owner, str)
            or not different_description_owner
            or different_description_owner == b2_control.get("target_gt_owner_id")
            or different_description_owner == b2_control.get("covering_gt_owner_id")
            or different_description_owner == same_description_owner
        ):
            return "failed", "invalid_optional_matched_physical_foil", []
        required_owner_ids.append(different_description_owner)

    def physical_prominence(entry: Mapping[str, Any]) -> dict[str, float]:
        values: dict[str, float] = {}
        for item in _sequence(entry.get("peak_prominence"), "B2 peak prominence"):
            receipt_item = _mapping(item, "B2 peak prominence item")
            if receipt_item.get("foil_identity_kind") != "reviewed_physical_owner":
                continue
            owner_id = receipt_item.get("foil_reviewed_physical_owner_id")
            if isinstance(owner_id, str) and owner_id:
                values[owner_id] = _number(
                    receipt_item.get("peak_prominence"),
                    "B2 physical-foil peak prominence value",
                )
        return values

    before = physical_prominence(before_entry)
    after = physical_prominence(after_entry)
    missing = [
        owner_id
        for owner_id in required_owner_ids
        if owner_id not in before or owner_id not in after
    ]
    if missing:
        return "failed", "required_matched_physical_foil_absent", required_owner_ids
    if all(after[owner_id] < before[owner_id] for owner_id in required_owner_ids):
        return "passed", "none", required_owner_ids
    return "failed", "matched_physical_foil_not_jointly_suppressive", required_owner_ids


def _summary_entries(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        _mapping(item, "summary per_owner_context entry")
        for item in _sequence(summary.get("per_owner_context"), "summary.per_owner_context")
    ]


def _entries_for_gt_owner(entries: Sequence[Mapping[str, Any]], gt_owner_id: str) -> list[Mapping[str, Any]]:
    return [entry for entry in entries if entry.get("gt_owner_id") == gt_owner_id]


def _restricted_control_entries(
    entries: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for entry in entries:
        key = (
            _string(entry.get("gt_owner_id"), "control surface GT owner"),
            _string(entry.get("context_id"), "control surface context ID"),
        )
        grouped.setdefault(key, []).append(entry)
    restricted: list[Mapping[str, Any]] = []
    expected_surfaces = {"canonical_description_free", "restricted_gt_target"}
    for key, group in sorted(grouped.items()):
        surfaces = {
            _string(entry.get("landscape_surface"), f"control surface {key}")
            for entry in group
        }
        if surfaces != expected_surfaces or len(group) != 2:
            _fail(
                f"control owner/context {key} must contain exactly one executed free and one restricted surface"
            )
        free = next(
            entry
            for entry in group
            if entry.get("landscape_surface") == "canonical_description_free"
        )
        targeted = next(
            entry
            for entry in group
            if entry.get("landscape_surface") == "restricted_gt_target"
        )
        if (
            free.get("decision_status") != "neutral_raw_only"
            or not free.get("neutral_reasons")
            or free.get("basins") != []
            or free.get("peak_prominence") != []
        ):
            _fail(
                f"free control surface {key} must remain executed raw-only non-evidence"
            )
        restricted.append(targeted)
    return restricted


def _require_decision_bearing(entries: Sequence[Mapping[str, Any]], label: str) -> None:
    if not entries:
        _fail(f"control-only summary is missing {label}")
    for entry in entries:
        if entry.get("decision_status") != "measured_no_conclusion":
            _fail(f"{label} is unresolved, unreviewed, or globally ambiguous and cannot enter calibration")
        if entry.get("neutral_reasons"):
            _fail(f"{label} carries a neutral reason and cannot enter a calibration denominator")


def _target_measurements(entry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        _mapping(item, "summary basin")
        for item in _sequence(entry.get("basins"), "summary entry.basins")
        if isinstance(item, Mapping) and item.get("role_kind") == "target"
    ]


def _metric_value(entry: Mapping[str, Any], metric: str, reducer: str) -> float:
    if metric == "peak_height" and reducer == "maximum_target_peak":
        targets = _target_measurements(entry)
        if not targets:
            _fail("calibration entry has no registered target basin")
        return max(_number(item.get("peak_height"), "target peak height") for item in targets)
    if metric == "peak_prominence" and reducer == "minimum_registered_foil_prominence":
        values = [
            _number(item.get("peak_prominence"), "peak prominence")
            for item in _sequence(entry.get("peak_prominence"), "summary entry.peak_prominence")
            if isinstance(item, Mapping)
        ]
        if not values:
            _fail("calibration entry has no registered target-versus-foil prominence receipt")
        return min(values)
    _fail(f"unsupported predeclared calibration metric/reducer pair: {metric}/{reducer}")


def _empirical_quantile_lower(values: Sequence[float], quantile: float) -> float:
    if not values:
        _fail("calibration quantile requires at least one non-C control value")
    if not 0.0 <= quantile <= 1.0:
        _fail("calibration quantile must lie in [0,1]")
    ordered = sorted(values)
    return ordered[math.floor(quantile * (len(ordered) - 1))]


def _calibrate(
    *,
    summary: Mapping[str, Any],
    summary_receipt: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, Any]],
    rules: core.LandscapeRules,
    rules_document: Mapping[str, Any],
    rules_file_sha256: str,
    control_registry_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    entries = _summary_entries(summary)
    selected_contexts = {
        _string(item, "control rules emitted context ID")
        for item in _sequence(
            _mapping(
                rules_document.get("task6_context_selection"),
                "control rules.task6_context_selection",
            ).get("emitted_context_ids"),
            "control rules emitted_context_ids",
        )
    }
    observed_contexts = {
        _string(entry.get("context_id"), "control summary context_id")
        for entry in entries
    }
    if observed_contexts != selected_contexts:
        _fail("control summary membership differs from the frozen control-only plan")
    restricted_entries = _restricted_control_entries(entries)
    strict = controls[REQUIRED_SMOKE_CONTROLS[0]]
    b1 = controls[REQUIRED_SMOKE_CONTROLS[1]]
    b2 = controls[REQUIRED_SMOKE_CONTROLS[2]]
    admitted_non_c_owners = {strict.get("gt_owner_id"), b1.get("gt_owner_id")}
    unexpected_owners = sorted(
        str(entry.get("gt_owner_id"))
        for entry in entries
        if entry.get("gt_owner_id") not in admitted_non_c_owners
    )
    if unexpected_owners:
        _fail(f"control-only calibration summary contains non-control/C owners: {unexpected_owners}")
    strict_entries = _entries_for_gt_owner(restricted_entries, _string(strict.get("gt_owner_id"), "strict control GT owner"))
    b1_entries = _entries_for_gt_owner(restricted_entries, _string(b1.get("gt_owner_id"), "B1 control GT owner"))
    covering_entries = _entries_for_gt_owner(restricted_entries, _string(b2.get("covering_gt_owner_id"), "B2 covering owner"))
    target_entries = _entries_for_gt_owner(restricted_entries, _string(b2.get("target_gt_owner_id"), "B2 target owner"))
    _require_decision_bearing(strict_entries, "strict visible control gt:7511:22")
    _require_decision_bearing(b1_entries, "B1 control gt:7511:26")
    _require_decision_bearing(covering_entries, "B2 covering owner gt:7511:22")
    _require_decision_bearing(target_entries, "B2 target owner gt:7511:26")
    if b2.get("physical_identity_review") != "distinct_people_confirmed_by_lead_visual_inspection":
        _fail("B2 smoke pair lacks the sealed distinct-physical-owner review")

    contract = _mapping(rules_document.get("calibration_contract"), "rules.calibration_contract")
    if contract.get("algorithm") != "empirical_quantile_lower.v1":
        _fail("rules.calibration_contract.algorithm must be empirical_quantile_lower.v1")
    quantile = _number(contract.get("quantile"), "calibration quantile")
    reducers = _mapping(contract.get("metric_reducers"), "rules calibration metric_reducers")
    calibration_control_ids = tuple(
        _string(item, "calibration control ID")
        for item in _sequence(contract.get("control_ids"), "rules calibration control_ids")
    )
    allowed = {REQUIRED_SMOKE_CONTROLS[0], REQUIRED_SMOKE_CONTROLS[1]}
    if not calibration_control_ids or set(calibration_control_ids).difference(allowed):
        _fail("smoke calibration may use only the sealed strict-visible and B1 controls")
    entries_by_control = {
        REQUIRED_SMOKE_CONTROLS[0]: strict_entries,
        REQUIRED_SMOKE_CONTROLS[1]: b1_entries,
    }
    thresholds: dict[str, float] = {}
    raw_values: dict[str, list[dict[str, Any]]] = {}
    for metric, reducer_untyped in sorted(reducers.items()):
        reducer = _string(reducer_untyped, f"calibration reducer for {metric}")
        values: list[float] = []
        receipts: list[dict[str, Any]] = []
        for control_id in calibration_control_ids:
            # Multiple declared contexts remain separate non-C observations;
            # no best-context selection is performed after seeing a value.
            for entry in entries_by_control[control_id]:
                value = _metric_value(entry, str(metric), reducer)
                values.append(value)
                receipts.append(
                    {
                        "control_id": control_id,
                        "gt_owner_id": entry.get("gt_owner_id"),
                        "context_id": entry.get("context_id"),
                        "value": value,
                    }
                )
        thresholds[str(metric)] = _empirical_quantile_lower(values, quantile)
        raw_values[str(metric)] = receipts
    b2_before = [
        entry for entry in target_entries if str(entry.get("context_id", "")).endswith(":B2_before")
    ]
    b2_after = [
        entry for entry in target_entries if str(entry.get("context_id", "")).endswith(":B2_after")
    ]
    b2_relative_suppression = "passed"
    b2_failure_reason = "none"
    b2_matched_physical_foil_owner_ids: list[str] = []
    if rules.contract_mode == "production":
        if len(b2_before) != 1 or len(b2_after) != 1:
            _fail("production B2 smoke requires exactly one before and one after context")
        (
            b2_relative_suppression,
            b2_failure_reason,
            b2_matched_physical_foil_owner_ids,
        ) = _b2_matched_physical_suppression(
            before_entry=b2_before[0],
            after_entry=b2_after[0],
            b2_control=b2,
        )

    receipt = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "contract_mode": rules.contract_mode,
        "rule_digest": rules.rule_digest,
        "semantic_core_sha256": rules.rule_digest,
        "control_decision_rules_sha256": rules_file_sha256,
        "rules_file_sha256": rules_file_sha256,
        "control_registry_sha256": control_registry_sha256,
        "control_summary_sha256": sha256_json(summary),
        "control_summary_receipt_sha256": sha256_json(summary_receipt),
        "anti_leakage": {
            "only_sealed_non_c_controls_read": True,
            "sentinel_summary_opened_before_calibration_seal": False,
            "calibration_control_ids": list(calibration_control_ids),
        },
        "algorithm": contract.get("algorithm"),
        "quantile": quantile,
        "metric_reducers": dict(reducers),
        "raw_control_values": raw_values,
        "thresholds": thresholds,
        "b2_pair_attestation": {
            "control_id": REQUIRED_SMOKE_CONTROLS[2],
            "distinct_physical_owners": True,
            "covering_gt_owner_id": b2.get("covering_gt_owner_id"),
            "target_gt_owner_id": b2.get("target_gt_owner_id"),
            "matched_physical_foil_owner_ids": b2_matched_physical_foil_owner_ids,
            "relative_suppression_status": b2_relative_suppression,
            "failure_reason": b2_failure_reason,
        },
        "scientific_conclusion": None,
    }
    facts = {
        "representative_positive_peak": (
            "passed"
            if any(target.get("shape") == "localized_peak" for entry in strict_entries for target in _target_measurements(entry))
            else "failed"
        ),
        "b1_target_surface": "passed" if any(_target_measurements(entry) for entry in b1_entries) else "failed",
        "b2_reviewed_pair": "passed",
        "b2_relative_suppression": b2_relative_suppression,
        "b2_failure_reason": b2_failure_reason,
        "free_surface_executed_raw_only": "passed",
    }
    return receipt, facts


def _freeze_non_c_smoke(
    *,
    rules: core.LandscapeRules,
    rules_file_sha256: str,
    control_summary: Mapping[str, Any],
    control_summary_receipt: Mapping[str, Any],
    calibration_file_sha256: str,
    facts: Mapping[str, str],
) -> dict[str, Any]:
    if facts.get("representative_positive_peak") != "passed":
        _fail("representative positive-control gate failed before non-C freeze")
    if facts.get("b2_reviewed_pair") != "passed":
        _fail("B2 reviewed-pair gate failed before non-C freeze")
    if facts.get("b2_relative_suppression") != "passed":
        _fail("B2 relative-suppression gate failed before non-C freeze")
    if control_summary_receipt.get("independent_reconstruction") != "passed":
        _fail("control summary was not independently reconstructed before non-C freeze")
    scoring_backend_gate = _summary_scoring_backend_gate(
        control_summary, control_summary_receipt
    )
    return {
        "schema_version": NON_C_SMOKE_FREEZE_SCHEMA_VERSION,
        "status": "passed",
        "control_decision_rules_sha256": rules_file_sha256,
        "semantic_core_sha256": rules.rule_digest,
        "control_score_artifact_sha256": _sha256(
            control_summary.get("score_artifact_sha256"),
            "control summary.score_artifact_sha256",
        ),
        "control_score_receipt_sha256": _sha256(
            control_summary.get("score_receipt_sha256"),
            "control summary.score_receipt_sha256",
        ),
        "control_summary_sha256": sha256_json(control_summary),
        "control_summary_receipt_sha256": sha256_json(control_summary_receipt),
        "calibration_receipt_sha256": calibration_file_sha256,
        "independent_reconstruction": "passed",
        "gates": {
            "representative_positive_control": "passed",
            "mandatory_cache_parity": control_summary_receipt[
                "mandatory_cache_parity_gate"
            ],
            "scoring_backend_admission": scoring_backend_gate,
            "b2_reviewed_pair": "passed",
            "free_surface_executed_raw_only": "passed",
        },
        "c_outcomes_read": False,
        "scientific_conclusion": None,
    }


def _validate_sentinel_registry(
    *,
    registry_path: Path,
    selection_path: Path,
    confirmation_path: Path,
    sealed_inputs: Mapping[str, Any],
) -> tuple[Mapping[str, Mapping[str, Any]], str, str, str]:
    registry = _read_json(registry_path)
    selection = _read_json(selection_path)
    confirmation = _read_json(confirmation_path)
    registry_sha = sha256_file(registry_path)
    selection_sha = sha256_file(selection_path)
    confirmation_sha = sha256_file(confirmation_path)
    if registry.get("schema_version") != SENTINEL_REGISTRY_SCHEMA_VERSION:
        _fail(f"sentinel registry schema must be {SENTINEL_REGISTRY_SCHEMA_VERSION!r}")
    if selection.get("schema_version") != SENTINEL_SELECTION_SCHEMA_VERSION:
        _fail(f"sentinel selection schema must be {SENTINEL_SELECTION_SCHEMA_VERSION!r}")
    if confirmation.get("schema_version") != SENTINEL_CONFIRMATION_SCHEMA_VERSION:
        _fail(f"sentinel confirmation schema must be {SENTINEL_CONFIRMATION_SCHEMA_VERSION!r}")
    if _sha256(sealed_inputs.get("sentinel_registry_sha256"), "rules sentinel registry digest") != registry_sha:
        _fail("sentinel registry digest does not match frozen rules")
    if _sha256(sealed_inputs.get("sentinel_selection_receipt_sha256"), "rules sentinel selection digest") != selection_sha:
        _fail("sentinel selection receipt digest does not match frozen rules")
    if _sha256(
        sealed_inputs.get("sentinel_selection_confirmation_receipt_sha256"),
        "rules sentinel confirmation digest",
    ) != confirmation_sha:
        _fail("sentinel selection confirmation receipt digest does not match frozen rules")
    bound = _mapping(registry.get("selection_receipt"), "sentinel registry.selection_receipt")
    if bound.get("sha256") != selection_sha:
        _fail("sentinel registry does not bind the supplied selection receipt")
    confirmation_bound = _mapping(
        registry.get("selection_confirmation_receipt"),
        "sentinel registry.selection_confirmation_receipt",
    )
    if confirmation_bound.get("sha256") != confirmation_sha:
        _fail("sentinel registry does not bind the supplied Task0-v2 confirmation receipt")
    if selection.get("anti_leakage_contract", {}).get("landscape_scores_used_for_selection") is not False:
        _fail("sentinel selection receipt does not prove pre-landscape selection")
    confirmation_anti_leakage = _mapping(
        confirmation.get("anti_leakage_contract"), "sentinel confirmation.anti_leakage_contract"
    )
    if (
        confirmation_anti_leakage.get("landscape_scores_used_for_original_selection") is not False
        or confirmation_anti_leakage.get("landscape_scores_used_for_v2_confirmation") is not False
        or confirmation_anti_leakage.get("selection_membership_changed") is not False
    ):
        _fail("sentinel Task0-v2 confirmation does not preserve blinded, unchanged membership")
    task0 = _mapping(confirmation.get("final_task0_v2"), "sentinel confirmation.final_task0_v2")
    chain = {
        "artifact_manifest_sha256": "task0_census_artifact_manifest_sha256",
        "execution_receipt_content_sha256": "task0_execution_receipt_content_sha256",
        "execution_receipt_file_sha256": "task0_execution_receipt_file_sha256",
        "owner_ledger_sha256": "owner_ledger_sha256",
        "owner_trajectory_matrix_sha256": "owner_trajectory_matrix_sha256",
    }
    for confirmation_key, sealed_key in chain.items():
        if _sha256(task0.get(confirmation_key), f"confirmation {confirmation_key}") != _sha256(
            sealed_inputs.get(sealed_key), f"rules sealed {sealed_key}"
        ):
            _fail(f"sentinel confirmation {confirmation_key} does not match final Task0-v2")
    sentinels = {
        _string(item.get("gt_owner_id"), "sentinel GT owner"): item
        for item in (_mapping(value, "sentinel") for value in _sequence(registry.get("sentinels"), "sentinels"))
    }
    confirmed = {
        _string(item.get("gt_owner_id"), "confirmed sentinel GT owner"): item
        for item in (
            _mapping(value, "confirmed sentinel")
            for value in _sequence(confirmation.get("confirmed_sentinels"), "confirmed sentinels")
        )
    }
    if set(confirmed) != set(sentinels):
        _fail("Task0-v2 sentinel confirmation membership differs from the sealed registry")
    for owner, item in confirmed.items():
        if (
            item.get("decision_eligible") is not True
            or item.get("strict_match_count") != 0
            or _number(item.get("max_semantic_compatible_iou"), f"confirmed sentinel {owner} IoU") != 0.0
        ):
            _fail(f"confirmed sentinel {owner} is ambiguous, recovered, or not decision eligible")
    return sentinels, registry_sha, selection_sha, confirmation_sha


def _validate_two_stage_rule_lineage(
    *,
    control_document: Mapping[str, Any],
    control_rules: core.LandscapeRules,
    control_rules_sha256: str,
    sentinel_document: Mapping[str, Any],
    sentinel_rules: core.LandscapeRules,
    sentinel_rules_sha256: str,
    freeze_receipt_sha256: str,
) -> None:
    if control_document.get("structural_status") != "draft_pre_smoke":
        _fail("control decision rules must be draft_pre_smoke")
    if sentinel_document.get("structural_status") != "sealed_non_c_smoke":
        _fail("sentinel decision rules must be sealed_non_c_smoke")
    control_selection = _mapping(
        control_document.get("task6_context_selection"),
        "control rules.task6_context_selection",
    )
    sentinel_selection = _mapping(
        sentinel_document.get("task6_context_selection"),
        "sentinel rules.task6_context_selection",
    )
    if control_selection.get("plan_membership") != "task4_control_only":
        _fail("control decision rules are not exact Task-4 control-only membership")
    if sentinel_selection.get("plan_membership") != "task4_sentinel_only":
        _fail("sentinel decision rules are not exact Task-4 sentinel-only membership")
    if control_rules_sha256 == sentinel_rules_sha256:
        _fail("control and sentinel outer decision-rule files must have distinct SHA-256")
    if control_rules.rule_digest != sentinel_rules.rule_digest:
        _fail("control and sentinel decision rules do not share one semantic core digest")
    if dict(core.semantic_core_payload(control_document)) != dict(
        core.semantic_core_payload(sentinel_document)
    ):
        _fail("control and sentinel semantic-core payload bytes are not identical")
    freeze_binding = _mapping(
        sentinel_document.get("non_c_smoke_freeze_receipt"),
        "sentinel rules.non_c_smoke_freeze_receipt",
    )
    if (
        _sha256(freeze_binding.get("sha256"), "sentinel freeze receipt SHA")
        != freeze_receipt_sha256
        or _sha256(
            freeze_binding.get("control_decision_rules_sha256"),
            "sentinel freeze control rules SHA",
        )
        != control_rules_sha256
        or _sha256(
            freeze_binding.get("semantic_core_sha256"),
            "sentinel freeze semantic core SHA",
        )
        != control_rules.rule_digest
    ):
        _fail("sentinel decision rules have stale or foreign non-C freeze lineage")
    control_inputs = _mapping(control_document.get("sealed_inputs"), "control rules.sealed_inputs")
    sentinel_inputs = _mapping(
        sentinel_document.get("sealed_inputs"), "sentinel rules.sealed_inputs"
    )
    identity_keys = (
        "task0_census_artifact_manifest_sha256",
        "task0_execution_receipt_content_sha256",
        "task0_execution_receipt_file_sha256",
        "owner_ledger_sha256",
        "owner_trajectory_matrix_sha256",
        "sentinel_registry_sha256",
        "sentinel_selection_confirmation_receipt_sha256",
        "control_registry_sha256",
        "identity_receipt_sha256",
    )
    for key in identity_keys:
        if _sha256(control_inputs.get(key), f"control sealed {key}") != _sha256(
            sentinel_inputs.get(key), f"sentinel sealed {key}"
        ):
            _fail(f"control and sentinel source/registry identity differs for {key}")


def _admit_one_sentinel(
    *,
    summary_path: Path,
    receipt_path: Path,
    registry_path: Path,
    selection_path: Path,
    confirmation_path: Path,
    sentinel_rules_path: Path,
    rules: core.LandscapeRules,
    sentinel_rules_file_sha256: str,
    sealed_inputs: Mapping[str, Any],
    calibration_payload_sha256: str,
) -> dict[str, Any]:
    # This is intentionally the first read of any sentinel-owned artifact.
    sentinels, registry_sha, selection_sha, confirmation_sha = _validate_sentinel_registry(
        registry_path=registry_path,
        selection_path=selection_path,
        confirmation_path=confirmation_path,
        sealed_inputs=sealed_inputs,
    )
    if sha256_file(sentinel_rules_path) != sentinel_rules_file_sha256:
        _fail("sentinel decision rules changed after calibration was sealed")
    summary, receipt = _validate_summary(
        summary_path,
        receipt_path,
        rules=rules,
        rules_file_sha256=sentinel_rules_file_sha256,
    )
    entries = _summary_entries(summary)
    owners = {entry.get("gt_owner_id") for entry in entries}
    if len(owners) != 1:
        _fail("optional blinded sentinel artifact must contain exactly one GT owner")
    owner = next(iter(owners))
    if owner not in sentinels:
        _fail("optional sentinel owner is absent from the sealed sentinel registry")
    if any(entry.get("decision_status") == "neutral_raw_only" for entry in entries):
        status = "admitted_raw_only_neutral"
    else:
        status = "admitted_case_level_unresolved"
    return {
        "status": status,
        "gt_owner_id": owner,
        "sentinel_id": sentinels[str(owner)].get("sentinel_id"),
        "sentinel_registry_sha256": registry_sha,
        "sentinel_selection_receipt_sha256": selection_sha,
        "sentinel_selection_confirmation_receipt_sha256": confirmation_sha,
        "sentinel_summary_sha256": sha256_json(summary),
        "sentinel_summary_receipt_sha256": sha256_json(receipt),
        "calibration_payload_sha256_fixed_before_sentinel_read": calibration_payload_sha256,
        "rule_digest": rules.rule_digest,
        "semantic_core_sha256": rules.rule_digest,
        "sentinel_decision_rules_sha256": sentinel_rules_file_sha256,
        "rules_file_sha256": sentinel_rules_file_sha256,
        "case_disposition": "unresolved",
        "scoring_backend_gate": receipt["scoring_backend_gate"],
        "scientific_conclusion": None,
    }


def _stop_rule_states(*, facts: Mapping[str, str], sentinel: Mapping[str, Any] | None) -> dict[str, str]:
    return {
        "stop_rule_1_source_or_policy_identity": "clear",
        "stop_rule_2_stable_owner_row_identity": "clear",
        "stop_rule_3_positive_control_peak": "clear" if facts["representative_positive_peak"] == "passed" else "triggered",
        "stop_rule_4_fp32_reproduction": "clear",
        "stop_rule_5_rules_and_calibration_receipts": "clear",
        "stop_rule_6_first_skip_ambiguity": "not_evaluated_in_cpu_smoke_attestor",
        "stop_rule_7_description_donor_absence": "not_evaluated_in_cpu_smoke_attestor",
        "stop_rule_8_repair_exchange_or_regression": "not_evaluated_in_cpu_smoke_attestor",
        "stop_rule_9_b2_matched_foils": "clear"
        if facts["b2_reviewed_pair"] == "passed"
        and facts["b2_relative_suppression"] == "passed"
        else "triggered",
        "sentinel_attestation": "not_requested" if sentinel is None else str(sentinel["status"]),
    }


def _mechanical_disposition(states: Mapping[str, str], rules_document: Mapping[str, Any]) -> tuple[str, int]:
    smoke = _mapping(rules_document.get("smoke_attestation"), "rules.smoke_attestation")
    rules = _sequence(smoke.get("disposition_rules"), "rules smoke disposition_rules")
    matches: list[tuple[int, str]] = []
    for index, untyped in enumerate(rules):
        item = _mapping(untyped, f"disposition_rules[{index}]")
        when = _mapping(item.get("when"), f"disposition_rules[{index}].when")
        unknown = set(when).difference(states)
        if unknown:
            _fail(f"disposition rule {index} references unknown gate states: {sorted(unknown)}")
        if all(expected == "*" or states[key] == expected for key, expected in when.items()):
            disposition = _string(item.get("disposition"), f"disposition_rules[{index}].disposition")
            if disposition not in DISPOSITIONS:
                _fail(f"disposition rule {index} has unsupported disposition {disposition!r}")
            matches.append((index, disposition))
    if len(matches) != 1:
        _fail(f"declared gate table must match exactly one disposition rule; matches={matches}")
    return matches[0][1], matches[0][0]


def attest(
    *,
    control_summary_path: Path,
    control_summary_receipt_path: Path,
    control_registry_path: Path,
    rules_path: Path,
    output_dir: Path,
    sentinel_rules_path: Path | None = None,
    sentinel_summary_path: Path | None = None,
    sentinel_summary_receipt_path: Path | None = None,
    sentinel_registry_path: Path | None = None,
    sentinel_selection_path: Path | None = None,
    sentinel_confirmation_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration_path = output_dir / CALIBRATION_NAME
    freeze_path = output_dir / NON_C_SMOKE_FREEZE_NAME
    lead_path = output_dir / LEAD_REVIEW_NAME
    if calibration_path.exists() or freeze_path.exists() or lead_path.exists():
        _fail(f"refusing to overwrite immutable smoke outputs in {output_dir}")
    rules_document = _read_json(rules_path)
    try:
        rules = core.validate_rule_mapping(rules_document)
    except ValueError as exc:
        _fail(f"landscape-decision-rules.json is incompatible with pure-core v2: {exc}")
    rules_file_sha256 = sha256_file(rules_path)
    if rules_document.get("structural_status") != "draft_pre_smoke":
        _fail("control decision rules must be draft_pre_smoke")
    if _mapping(
        rules_document.get("task6_context_selection"),
        "control rules.task6_context_selection",
    ).get("plan_membership") != "task4_control_only":
        _fail("control decision rules must have exact Task-4 control-only membership")
    if rules.contract_mode == "production" and len(
        _sequence(
            _mapping(
                rules_document.get("task6_context_selection"),
                "control rules.task6_context_selection",
            ).get("emitted_context_ids"),
            "control rules emitted_context_ids",
        )
    ) != 4:
        _fail("production Task-4 control rules must contain exactly four contexts")
    sealed_inputs = _mapping(rules_document.get("sealed_inputs"), "rules.sealed_inputs")
    control_registry = _read_json(control_registry_path)
    controls = _validate_control_registry(
        control_registry,
        registry_sha256=sha256_file(control_registry_path),
        sealed_inputs=sealed_inputs,
    )
    control_summary, control_summary_receipt = _validate_summary(
        control_summary_path,
        control_summary_receipt_path,
        rules=rules,
        rules_file_sha256=rules_file_sha256,
    )
    calibration, facts = _calibrate(
        summary=control_summary,
        summary_receipt=control_summary_receipt,
        controls=controls,
        rules=rules,
        rules_document=rules_document,
        rules_file_sha256=rules_file_sha256,
        control_registry_sha256=sha256_file(control_registry_path),
    )
    _write_once(calibration_path, calibration)
    calibration_file_sha256 = sha256_file(calibration_path)
    if (
        facts.get("representative_positive_peak") != "passed"
        or facts.get("b2_relative_suppression") != "passed"
    ):
        states = _stop_rule_states(facts=facts, sentinel=None)
        disposition, matched_rule_index = _mechanical_disposition(
            states, rules_document
        )
        if disposition != "hold":
            _fail(
                "a failed representative positive-control gate must mechanically hold"
            )
        lead = {
            "schema_version": LEAD_REVIEW_SCHEMA_VERSION,
            "status": "held_before_non_c_smoke_freeze",
            "contract_mode": rules.contract_mode,
            "rule_digest": rules.rule_digest,
            "semantic_core_sha256": rules.rule_digest,
            "control_decision_rules_sha256": rules_file_sha256,
            "sentinel_decision_rules_sha256": None,
            "rules_file_sha256": rules_file_sha256,
            "control_registry_sha256": sha256_file(control_registry_path),
            "control_summary_sha256": sha256_json(control_summary),
            "control_summary_receipt_sha256": sha256_json(
                control_summary_receipt
            ),
            "calibration_receipt_file_sha256": calibration_file_sha256,
            "calibration_payload_sha256": sha256_json(calibration),
            "non_c_smoke_freeze_receipt_sha256": None,
            "non_c_smoke_freeze_status": "not_emitted_control_gate_failed",
            "control_scoring_backend_gate": control_summary_receipt[
                "scoring_backend_gate"
            ],
            "control_facts": dict(facts),
            "stop_rule_states": states,
            "disposition": disposition,
            "matched_disposition_rule_index": matched_rule_index,
            "sentinel_attestation": None,
            "sentinel_is_case_level_only": False,
            "scientific_conclusion": None,
        }
        _write_once(lead_path, lead)
        return calibration, lead
    freeze = _freeze_non_c_smoke(
        rules=rules,
        rules_file_sha256=rules_file_sha256,
        control_summary=control_summary,
        control_summary_receipt=control_summary_receipt,
        calibration_file_sha256=calibration_file_sha256,
        facts=facts,
    )
    _write_once(freeze_path, freeze)
    freeze_file_sha256 = sha256_file(freeze_path)

    optional = (
        sentinel_rules_path,
        sentinel_summary_path,
        sentinel_summary_receipt_path,
        sentinel_registry_path,
        sentinel_selection_path,
        sentinel_confirmation_path,
    )
    if any(item is not None for item in optional) and not all(item is not None for item in optional):
        _fail("optional sentinel admission requires sentinel rules, summary, summary receipt, registry, and selection receipts together")
    sentinel: dict[str, Any] | None = None
    if all(item is not None for item in optional):
        assert sentinel_rules_path is not None
        assert sentinel_summary_path is not None
        assert sentinel_summary_receipt_path is not None
        assert sentinel_registry_path is not None
        assert sentinel_selection_path is not None
        assert sentinel_confirmation_path is not None
        sentinel_rules_document = _read_json(sentinel_rules_path)
        try:
            sentinel_rules = core.validate_rule_mapping(sentinel_rules_document)
        except ValueError as exc:
            _fail(f"sentinel-decision-rules.json is incompatible with pure-core v2: {exc}")
        sentinel_rules_file_sha256 = sha256_file(sentinel_rules_path)
        _validate_two_stage_rule_lineage(
            control_document=rules_document,
            control_rules=rules,
            control_rules_sha256=rules_file_sha256,
            sentinel_document=sentinel_rules_document,
            sentinel_rules=sentinel_rules,
            sentinel_rules_sha256=sentinel_rules_file_sha256,
            freeze_receipt_sha256=freeze_file_sha256,
        )
        sentinel_sealed_inputs = _mapping(
            sentinel_rules_document.get("sealed_inputs"),
            "sentinel rules.sealed_inputs",
        )
        sentinel_selected_contexts = {
            _string(item, "sentinel rules emitted context ID")
            for item in _sequence(
                _mapping(
                    sentinel_rules_document.get("task6_context_selection"),
                    "sentinel rules.task6_context_selection",
                ).get("emitted_context_ids"),
                "sentinel rules emitted_context_ids",
            )
        }
        sentinel_summary_preview = _read_json(  # first sentinel-outcome read
            sentinel_summary_path
        )
        if {
            _string(entry.get("context_id"), "sentinel summary context_id")
            for entry in _summary_entries(sentinel_summary_preview)
        } != sentinel_selected_contexts:
            _fail("sentinel summary membership differs from the frozen sentinel-only plan")
        sentinel = _admit_one_sentinel(
            summary_path=sentinel_summary_path,
            receipt_path=sentinel_summary_receipt_path,
            registry_path=sentinel_registry_path,
            selection_path=sentinel_selection_path,
            confirmation_path=sentinel_confirmation_path,
            sentinel_rules_path=sentinel_rules_path,
            rules=sentinel_rules,
            sentinel_rules_file_sha256=sentinel_rules_file_sha256,
            sealed_inputs=sentinel_sealed_inputs,
            calibration_payload_sha256=sha256_json(calibration),
        )
    states = _stop_rule_states(facts=facts, sentinel=sentinel)
    disposition, matched_rule_index = _mechanical_disposition(states, rules_document)
    lead = {
        "schema_version": LEAD_REVIEW_SCHEMA_VERSION,
        "contract_mode": rules.contract_mode,
        "rule_digest": rules.rule_digest,
        "semantic_core_sha256": rules.rule_digest,
        "control_decision_rules_sha256": rules_file_sha256,
        "sentinel_decision_rules_sha256": (
            None if sentinel is None else sentinel["sentinel_decision_rules_sha256"]
        ),
        "rules_file_sha256": rules_file_sha256,
        "control_registry_sha256": sha256_file(control_registry_path),
        "control_summary_sha256": sha256_json(control_summary),
        "control_summary_receipt_sha256": sha256_json(control_summary_receipt),
        "calibration_receipt_file_sha256": calibration_file_sha256,
        "calibration_payload_sha256": sha256_json(calibration),
        "non_c_smoke_freeze_receipt_sha256": freeze_file_sha256,
        "control_scoring_backend_gate": control_summary_receipt[
            "scoring_backend_gate"
        ],
        "control_facts": dict(facts),
        "stop_rule_states": states,
        "disposition": disposition,
        "matched_disposition_rule_index": matched_rule_index,
        "sentinel_attestation": sentinel,
        "sentinel_is_case_level_only": sentinel is not None,
        "scientific_conclusion": None,
    }
    _write_once(lead_path, lead)
    return calibration, lead


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--control-summary-receipt", type=Path, required=True)
    parser.add_argument("--control-registry", type=Path, required=True)
    parser.add_argument("--decision-rules", type=Path, required=True)
    parser.add_argument("--sentinel-decision-rules", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sentinel-summary", type=Path)
    parser.add_argument("--sentinel-summary-receipt", type=Path)
    parser.add_argument("--sentinel-registry", type=Path)
    parser.add_argument("--sentinel-selection-receipt", type=Path)
    parser.add_argument("--sentinel-selection-confirmation-receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, lead = attest(
        control_summary_path=args.control_summary.resolve(strict=True),
        control_summary_receipt_path=args.control_summary_receipt.resolve(strict=True),
        control_registry_path=args.control_registry.resolve(strict=True),
        rules_path=args.decision_rules.resolve(strict=True),
        sentinel_rules_path=(
            None
            if args.sentinel_decision_rules is None
            else args.sentinel_decision_rules.resolve(strict=True)
        ),
        output_dir=args.output_dir.expanduser().resolve(),
        sentinel_summary_path=None if args.sentinel_summary is None else args.sentinel_summary.resolve(strict=True),
        sentinel_summary_receipt_path=(
            None if args.sentinel_summary_receipt is None else args.sentinel_summary_receipt.resolve(strict=True)
        ),
        sentinel_registry_path=(None if args.sentinel_registry is None else args.sentinel_registry.resolve(strict=True)),
        sentinel_selection_path=(
            None if args.sentinel_selection_receipt is None else args.sentinel_selection_receipt.resolve(strict=True)
        ),
        sentinel_confirmation_path=(
            None
            if args.sentinel_selection_confirmation_receipt is None
            else args.sentinel_selection_confirmation_receipt.resolve(strict=True)
        ),
    )
    print(json.dumps({"disposition": lead["disposition"], "sentinel": lead["sentinel_attestation"] is not None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply the frozen legacy support calibration to the scored image-2299 shard.

This is a one-image prospective transfer analysis.  It never recalibrates a
threshold, fits a phenotype, pools image 2299 into the legacy twelve-image
denominator, or treats the proposal surface as localization evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as legacy  # noqa: E402
from scripts.research import merge_sorted_owner_accessibility_census_shards as legacy_merge  # noqa: E402
from scripts.research import score_sorted_owner_accessibility_census_shard as scorer  # noqa: E402


UNIT_ID = "2026-08-04-sorted-image2299-prospective-mechanism-extension"
IMAGE_ID = "2299"
ANALYSIS_SCHEMA_VERSION = "sorted-image2299-owner-accessibility-analysis.v1"
OWNER_SCHEMA_VERSION = "sorted-image2299-owner-accessibility-owner-summary.v1"
CONTEXT_SCHEMA_VERSION = "sorted-image2299-owner-accessibility-owner-context.v1"
RECEIPT_SCHEMA_VERSION = "sorted-image2299-owner-accessibility-receipt.v1"

CALIBRATION_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-03-sorted-owner-accessibility-phenotype-census/20260803T065743Z/"
    "phases/discovery-sealed/support-calibration.json"
)
CALIBRATION_CONTENT_SHA256 = "9dd6d7646fc55db6155124dc4bbfa46642b32b006d39758bd2e24d1ca97058c5"
CALIBRATION_FILE_SHA256 = "bbe0ccb5141e659c20b8df984d7fbf50e9c8ab96f0b1b860a1664b29a2dd12ba"
TRANSFER_FLOOR = 0.80
MIN_TRANSFER_CONTROL_COUNT = 10
EXPECTED_OWNER_COUNT = 46
EXPECTED_CATEGORY_COUNTS = {"person": 38, "tie": 8}
LEGACY_OWNER_COUNT = 346
LEGACY_FN_DENOMINATOR = 202

SCORES_NAME = "census-scores.jsonl"
PROPOSAL_NAME = "proposal-surface.jsonl"
SHARD_RECEIPT_NAME = "shard-receipt.json"
OUTPUT_NAMES = (
    "analysis.json",
    "owner-summaries.jsonl",
    "owner-context-features.jsonl",
    "context-registry.jsonl",
)

DISPOSITION_RESOLVED = "resolved_tested_localization_support"
DISPOSITION_PERSISTENT = "persistent_no_tested_localization_support"
DISPOSITION_FLIP = "unresolved_ambiguity_bound_disposition_flip"
DISPOSITION_UNRESOLVED = "unresolved_insufficient_tested_localization_support"
DISPOSITION_WITHHELD = "withheld_calibration_nontransfer"
DISPOSITION_TP = "native_true_positive_transfer_control"


class AnalysisContractError(ValueError):
    """A frozen calibration, plan, shard, or denominator contract changed."""


@dataclass(frozen=True)
class AnalysisProduct:
    analysis: dict[str, Any]
    owner_summaries: list[dict[str, Any]]
    owner_contexts: list[dict[str, Any]]
    context_registry: list[dict[str, Any]]
    receipt: dict[str, Any]

    def files(self) -> dict[str, bytes]:
        return {
            "analysis.json": legacy.canonical_json_bytes(self.analysis) + b"\n",
            "owner-summaries.jsonl": _jsonl_bytes(self.owner_summaries),
            "owner-context-features.jsonl": _jsonl_bytes(self.owner_contexts),
            "context-registry.jsonl": _jsonl_bytes(self.context_registry),
        }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(legacy.canonical_json_bytes(row) + b"\n" for row in rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AnalysisContractError(f"input is unreadable: {path}") from exc
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisContractError(f"{label} is unreadable at {path}") from exc
    if not isinstance(value, Mapping):
        raise AnalysisContractError(f"{label} is not an object")
    return dict(value)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnalysisContractError(f"{label} is unreadable at {path}") from exc
    result: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisContractError(f"{label} line {number} is invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise AnalysisContractError(f"{label} line {number} is not an object")
        result.append(dict(value))
    return result


def load_frozen_calibration(
    path: Path = CALIBRATION_PATH,
) -> legacy_merge.SupportCalibration:
    file_digest = _sha256_file(path)
    if file_digest != CALIBRATION_FILE_SHA256:
        raise AnalysisContractError(
            f"frozen calibration file digest mismatch: {file_digest} != {CALIBRATION_FILE_SHA256}"
        )
    receipt = _read_json(path, "frozen support calibration")
    declared = receipt.get("calibration_sha256")
    reconstructed = legacy.sha256_json(
        {key: value for key, value in receipt.items() if key != "calibration_sha256"}
    )
    if declared != CALIBRATION_CONTENT_SHA256 or reconstructed != CALIBRATION_CONTENT_SHA256:
        raise AnalysisContractError("frozen support calibration content digest drifted")
    if float(receipt.get("quantile", -1)) != 0.10 or receipt.get("quantile_is_primary_and_fixed") is not True:
        raise AnalysisContractError("calibration no longer carries the frozen q10 rule")
    if receipt.get("rank_is_not_a_support_input") is not True:
        raise AnalysisContractError("calibration drifted to a rank-based support rule")
    if receipt.get("confirmation_evidence_consumed") is not False:
        raise AnalysisContractError("calibration consumed post-discovery evidence")
    try:
        return legacy_merge.calibration_from_receipt(receipt)
    except legacy_merge.MergeContractError as exc:
        raise AnalysisContractError(str(exc)) from exc


def _load_plan(
    plan_dir: Path,
    *,
    expected_extension_unit_id: str = UNIT_ID,
) -> scorer.PlanBundle:
    try:
        plan = scorer.load_plan(plan_dir)
    except scorer.ShardContractError as exc:
        raise AnalysisContractError(str(exc)) from exc
    receipt = plan.receipt
    if receipt.get("extension_unit_id") != expected_extension_unit_id:
        raise AnalysisContractError("plan is not the image-2299 prospective extension")
    shape = receipt.get("census_shape") or {}
    if shape.get("image_ids") != [IMAGE_ID] or int(shape.get("owner_count", -1)) != EXPECTED_OWNER_COUNT:
        raise AnalysisContractError("plan is not exactly the 46-owner image-2299 slice")
    if shape.get("owner_category_counts") != EXPECTED_CATEGORY_COUNTS:
        raise AnalysisContractError("plan owner category composition drifted")
    denominator = receipt.get("denominator_contract") or {}
    expected_denominator = {
        "prospective_image2299_owner_count": EXPECTED_OWNER_COUNT,
        "legacy_12_owner_count_unchanged": LEGACY_OWNER_COUNT,
        "legacy_12_eligible_native_fn_denominator_unchanged": LEGACY_FN_DENOMINATOR,
        "pooled_13_image_denominator_created": False,
        "report_slices_separately": True,
    }
    if denominator != expected_denominator:
        raise AnalysisContractError("legacy/prospective denominator contract drifted")
    if set(plan.images) != {IMAGE_ID} or len(plan.owners) != EXPECTED_OWNER_COUNT:
        raise AnalysisContractError("plan registries contain a foreign image or owner count")
    counts = Counter(str(row["normalized_description"]) for row in plan.owners.values())
    if dict(sorted(counts.items())) != EXPECTED_CATEGORY_COUNTS:
        raise AnalysisContractError("plan registry is not 38 person + 8 tie")
    return plan


def _verify_shard_receipt(plan: scorer.PlanBundle, shard_dir: Path) -> dict[str, Any]:
    receipt = _read_json(shard_dir / SHARD_RECEIPT_NAME, "scored shard receipt")
    if receipt.get("schema_version") != scorer.SHARD_RECEIPT_SCHEMA_VERSION:
        raise AnalysisContractError("scored shard receipt schema drifted")
    if receipt.get("unit_id") != legacy.UNIT_ID or str(receipt.get("image_id")) != IMAGE_ID:
        raise AnalysisContractError("scored shard receipt identity is not legacy-ABI image 2299")
    if receipt.get("status") != "captured" or receipt.get("capture_completeness") != "complete_shard":
        raise AnalysisContractError("image-2299 shard is not a complete captured shard")
    subset = receipt.get("subset_capture") or {}
    if subset.get("is_subset") is not False or subset.get("usable_as_complete_shard_evidence") is not True:
        raise AnalysisContractError("image-2299 shard is subset/smoke evidence")
    binding = receipt.get("plan") or {}
    if binding.get("receipt_content_sha256") != plan.receipt_content_sha256:
        raise AnalysisContractError("scored shard is bound to a different plan receipt")
    if binding.get("capture_rules_sha256") != plan.capture_rules_sha256:
        raise AnalysisContractError("scored shard is bound to different capture rules")
    checks = receipt.get("checks") or {}
    required_checks = (
        "all_finite", "canonical_suffix_verified", "coordinate_domain_ok",
        "cross_owner_tuple_collapse_verified", "every_row_bound_to_an_admission_receipt",
    )
    if any(checks.get(name) is not True for name in required_checks):
        raise AnalysisContractError("scored shard did not pass every required scorer check")
    backend = receipt.get("backend_identity") or {}
    if backend.get("backend") != "hf" or backend.get("is_real_model") is not True:
        raise AnalysisContractError("scored shard is not real HF evidence")
    if backend.get("executed_media_sha256") != plan.images[IMAGE_ID]["executed_media_sha256"]:
        raise AnalysisContractError("scored shard executed different image bytes")
    missing_files = [
        name for name in scorer.PRIMARY_OUTPUT_NAMES if not (shard_dir / name).is_file()
    ]
    if missing_files:
        raise AnalysisContractError(
            f"complete scored shard lacks primary files {missing_files}"
        )
    return receipt


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise AnalysisContractError("cannot normalize an empty query group")
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(value - peak) for value in values))


def _load_score_events(plan: scorer.PlanBundle, shard_dir: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(shard_dir / SCORES_NAME, "image-2299 localization scores")
    by_group: dict[str, list[dict[str, Any]]] = {}
    seen_requests: set[str] = set()
    for row in rows:
        required = {
            "schema_version", "row_contract", "unit_id", "plan_schema_version",
            "plan_receipt_content_sha256", "capture_rules_sha256", "channel",
            "image_id", "context_id", "query_group_id", "normalized_description",
            "candidate_id", "complete_box_logprob_sum", "competition", "request_id",
        }
        missing = sorted(required - set(row))
        if missing:
            raise AnalysisContractError(f"localization score row lacks fields {missing}")
        if row["schema_version"] != scorer.SCORE_SCHEMA_VERSION or row["row_contract"] != scorer.P0_ROW_CONTRACT:
            raise AnalysisContractError("localization score row is foreign or pre-P0")
        if str(row["image_id"]) != IMAGE_ID or row["unit_id"] != legacy.UNIT_ID:
            raise AnalysisContractError("localization score row has foreign identity")
        if row["plan_receipt_content_sha256"] != plan.receipt_content_sha256 or row["capture_rules_sha256"] != plan.capture_rules_sha256:
            raise AnalysisContractError("localization score row is bound to another plan")
        request_id = str(row["request_id"])
        if request_id in seen_requests:
            raise AnalysisContractError(f"duplicate localization request {request_id!r}")
        seen_requests.add(request_id)
        group_id = str(row["query_group_id"])
        group = plan.query_groups.get(group_id)
        candidate = plan.candidates.get(str(row["candidate_id"]))
        if group is None or candidate is None:
            raise AnalysisContractError("localization score references an unknown plan row")
        if str(group["context_id"]) != str(row["context_id"]) or str(group["normalized_description"]) != str(row["normalized_description"]):
            raise AnalysisContractError("localization score query-group identity drifted")
        if not math.isfinite(float(row["complete_box_logprob_sum"])):
            raise AnalysisContractError("localization score is non-finite")
        by_group.setdefault(group_id, []).append(row)

    planned = {key for key, row in plan.query_groups.items() if row.get("status") == "admitted"}
    if set(by_group) != planned:
        raise AnalysisContractError("scored query-group set does not exactly cover the plan")
    events: list[dict[str, Any]] = []
    for group_id in sorted(by_group):
        group = plan.query_groups[group_id]
        group_rows = by_group[group_id]
        expected_ids = {str(value) for value in group["candidate_ids"]}
        observed_ids = {str(row["candidate_id"]) for row in group_rows}
        if observed_ids != expected_ids or len(group_rows) != len(expected_ids):
            raise AnalysisContractError(f"query group {group_id!r} has incomplete candidate coverage")
        values = [float(row["complete_box_logprob_sum"]) for row in group_rows]
        normalizer = _logsumexp(values)
        ordered = sorted(group_rows, key=lambda row: (-float(row["complete_box_logprob_sum"]), str(row["candidate_id"])))
        ranks = {str(row["candidate_id"]): index + 1 for index, row in enumerate(ordered)}
        for row in group_rows:
            candidate = plan.candidates[str(row["candidate_id"])]
            declared = row.get("competition") or {}
            if int(declared.get("population_size", -1)) != len(group_rows) or int(declared.get("rank", -1)) != ranks[str(row["candidate_id"])] :
                raise AnalysisContractError(f"query group {group_id!r} competition metadata drifted")
            event = dict(row)
            event["strict_assignment"] = {
                "status": str(candidate["strict_assignment_status"]),
                "gt_owner_id": candidate.get("strict_assignment_gt_owner_id"),
            }
            event["competition"] = {
                **dict(declared),
                "within_group_log_posterior": float(row["complete_box_logprob_sum"]) - normalizer,
            }
            events.append(event)
    return events


def _load_proposal_surfaces(plan: scorer.PlanBundle, shard_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(shard_dir / PROPOSAL_NAME, "image-2299 proposal surface")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema_version") != scorer.PROPOSAL_SCHEMA_VERSION or row.get("row_contract") != scorer.P0_ROW_CONTRACT:
            raise AnalysisContractError("proposal row is foreign or pre-P0")
        if str(row.get("image_id")) != IMAGE_ID or row.get("plan_receipt_content_sha256") != plan.receipt_content_sha256:
            raise AnalysisContractError("proposal row has foreign image/plan identity")
        if row.get("emits_per_owner_proposal_probability") is not False or row.get("includes_coordinate_scores") is not False:
            raise AnalysisContractError("proposal/localization separation contract drifted")
        context_id = str(row["context_id"])
        if context_id in result or context_id not in plan.contexts:
            raise AnalysisContractError(f"duplicate or unknown proposal context {context_id!r}")
        result[context_id] = row
    if set(result) != set(plan.contexts):
        raise AnalysisContractError("proposal surface does not cover every native context")
    return result


def _support_block(rows: Sequence[Mapping[str, Any]], calibration: legacy_merge.SupportCalibration) -> dict[str, Any]:
    block = legacy_merge._support_features(rows)
    block["clears_frozen_support_rule"] = calibration.clears(block)
    block["thresholds_retuned"] = False
    return block


def _build_owner_contexts(
    plan: scorer.PlanBundle,
    events: Sequence[Mapping[str, Any]],
    proposals: Mapping[str, Mapping[str, Any]],
    calibration: legacy_merge.SupportCalibration,
) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for event in events:
        key = (str(event["context_id"]), str(event["normalized_description"]))
        by_group.setdefault(key, {})[str(event["candidate_id"])] = event
    owners = list(plan.owners.values())
    rows: list[dict[str, Any]] = []
    for owner in owners:
        owner_id = str(owner["gt_owner_id"])
        description = str(owner["normalized_description"])
        bank_ids = [str(value) for value in owner["candidate_bank"]["physical_candidate_ids"]]
        for context in sorted(plan.contexts.values(), key=lambda row: int(row["boundary_index"])):
            context_id = str(context["context_id"])
            group = by_group.get((context_id, description), {})
            neighbourhood = [group[candidate_id] for candidate_id in bank_ids if candidate_id in group]
            if not neighbourhood:
                raise AnalysisContractError(f"owner {owner_id} has no scored bank at {context_id}")
            assigned: list[Mapping[str, Any]] = []
            ambiguous: list[Mapping[str, Any]] = []
            unmatched: list[Mapping[str, Any]] = []
            other: list[Mapping[str, Any]] = []
            for event in neighbourhood:
                assignment = event["strict_assignment"]
                if assignment["status"] == "matched":
                    (assigned if str(assignment["gt_owner_id"]) == owner_id else other).append(event)
                elif assignment["status"] == "ambiguous_neutral":
                    ambiguous.append(event)
                else:
                    unmatched.append(event)
            lower = _support_block(assigned, calibration)
            upper = _support_block([*assigned, *ambiguous, *unmatched], calibration)
            exact_id = next(
                (
                    str(role["physical_candidate_id"])
                    for role in owner["candidate_bank"]["logical_roles"]
                    if role["role"] == "exact_gt_anchor" and role.get("physical_candidate_id")
                ),
                None,
            )
            exact = group.get(exact_id) if exact_id else None
            rows.append(
                {
                    "schema_version": CONTEXT_SCHEMA_VERSION,
                    "row_kind": "image2299_owner_context_support",
                    "owner_context_id": f"{owner_id}@{context_id}",
                    "gt_owner_id": owner_id,
                    "image_id": IMAGE_ID,
                    "normalized_description": description,
                    "context_id": context_id,
                    "boundary_index": int(context["boundary_index"]),
                    "context_role": str(context["context_role"]),
                    "native_context_only": True,
                    "loop_marking": dict(context["loop_marking"]),
                    "frontier_features": legacy_merge._frontier_features(context, owner, owners),
                    "localization": {
                        "estimand": "category_field_support_at_owner_geometry",
                        "proposal_probability": False,
                        "generator_local_max_excluding_other_owner_strict": {
                            "ambiguity_excluded_l": lower,
                            "ambiguity_included_u": upper,
                        },
                        "exact_anchor_score": None if exact is None else float(exact["complete_box_logprob_sum"]),
                        "other_owner_strict_event_count": len(other),
                        "support_partition": {
                            "strict_assigned_self": "L_and_U",
                            "ambiguous_upper": "U_only",
                            "unmatched_generator_local": "U_only",
                            "other_owner_strict": "excluded_from_both",
                        },
                    },
                    "proposal_surface": legacy_merge._proposal_channel(proposals.get(context_id), description),
                }
            )
    _attach_owner_ranks(rows)
    rows.sort(key=lambda row: (str(row["gt_owner_id"]), int(row["boundary_index"])))
    return rows


def _attach_owner_ranks(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["context_id"]), str(row["normalized_description"])), []).append(row)
    for group in grouped.values():
        for bound, key in (("ambiguity_excluded_l", "owner_competition_l"), ("ambiguity_included_u", "owner_competition_u")):
            scored = [
                row for row in group
                if row["localization"]["generator_local_max_excluding_other_owner_strict"][bound]["value"] is not None
            ]
            ordered = sorted(
                scored,
                key=lambda row: (
                    -float(row["localization"]["generator_local_max_excluding_other_owner_strict"][bound]["value"]),
                    str(row["gt_owner_id"]),
                ),
            )
            best = (
                float(ordered[0]["localization"]["generator_local_max_excluding_other_owner_strict"][bound]["value"])
                if ordered else None
            )
            for index, row in enumerate(ordered, 1):
                value = float(row["localization"]["generator_local_max_excluding_other_owner_strict"][bound]["value"])
                row[key] = {
                    "rank": index, "population_size": len(ordered),
                    "margin_to_best_owner": value - float(best),
                    "rank_role": "routing_and_competition_never_support",
                }
            for row in group:
                row.setdefault(key, {"rank": None, "population_size": len(ordered), "margin_to_best_owner": None, "rank_role": "routing_and_competition_never_support"})


def _size_fields(owner: Mapping[str, Any], image: Mapping[str, Any]) -> dict[str, Any]:
    x1, y1, x2, y2 = (float(value) for value in owner["bbox_pixel_xyxy"])
    width = x2 - x1
    height = y2 - y1
    minimum = min(width, height)
    if minimum < 16:
        band = "lt16"
    elif minimum < 32:
        band = "16_to_31"
    elif minimum < 64:
        band = "32_to_63"
    elif minimum < 128:
        band = "64_to_127"
    else:
        band = "ge128"
    return {
        "width_pixels": width, "height_pixels": height, "minimum_dimension_pixels": minimum,
        "normalized_area": width * height / (float(image["image_width"]) * float(image["image_height"])),
        "minimum_dimension_band": band,
        "band_rule": "fixed_predeclared_pixel_edges_16_32_64_128",
    }


def _summarize_owners(
    plan: scorer.PlanBundle,
    contexts: Sequence[Mapping[str, Any]],
    calibration: legacy_merge.SupportCalibration,
    *,
    force_descriptive_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_owner: dict[str, list[Mapping[str, Any]]] = {}
    for row in contexts:
        by_owner.setdefault(str(row["gt_owner_id"]), []).append(row)
    due_by_owner: dict[str, str] = {}
    row_index_by_prediction = {
        str(row["pred_row_id"]): int(row["row_index"]) for row in plan.native_sidecars
    }
    for owner_id, owner in plan.owners.items():
        matches = [str(value) for value in owner.get("native_strict_match_pred_row_ids", ())]
        if owner.get("native_true_positive"):
            if len(matches) != 1 or matches[0] not in row_index_by_prediction:
                raise AnalysisContractError(f"native TP {owner_id} lacks one deterministic due boundary")
            due_by_owner[owner_id] = f"{IMAGE_ID}:boundary-{row_index_by_prediction[matches[0]]:03d}"

    due_rows: list[dict[str, Any]] = []
    for owner_id, due_context in due_by_owner.items():
        match = next((row for row in by_owner[owner_id] if row["context_id"] == due_context), None)
        if match is None or match["loop_marking"]["loop_tail"]:
            due_rows.append({"gt_owner_id": owner_id, "due_context_id": due_context, "eligible": False, "supported_under_both_bounds": False})
            continue
        bounds = match["localization"]["generator_local_max_excluding_other_owner_strict"]
        supported = bounds["ambiguity_excluded_l"]["clears_frozen_support_rule"] is True and bounds["ambiguity_included_u"]["clears_frozen_support_rule"] is True
        due_rows.append({"gt_owner_id": owner_id, "due_context_id": due_context, "eligible": True, "supported_under_both_bounds": supported})
    # The transfer estimand is explicitly every native-TP owner.  A due
    # boundary that is unavailable under the frozen non-loop rule is an
    # unsupported control, not a silently removed denominator member.
    transfer_denominator = len(due_rows)
    eligible_due_count = sum(row["eligible"] for row in due_rows)
    transfer_numerator = sum(row["eligible"] and row["supported_under_both_bounds"] for row in due_rows)
    transfer_rate = transfer_numerator / transfer_denominator if transfer_denominator else 0.0
    transfer_evaluable = len(due_by_owner) >= MIN_TRANSFER_CONTROL_COUNT
    transfers: bool | None = (
        transfer_rate >= TRANSFER_FLOOR if transfer_evaluable else None
    )

    summaries: list[dict[str, Any]] = []
    for owner_id in sorted(plan.owners):
        owner = plan.owners[owner_id]
        owner_contexts = sorted(by_owner[owner_id], key=lambda row: int(row["boundary_index"]))
        non_loop = [row for row in owner_contexts if not row["loop_marking"]["loop_tail"]]
        l_ids = [
            str(row["context_id"]) for row in non_loop
            if row["localization"]["generator_local_max_excluding_other_owner_strict"]["ambiguity_excluded_l"]["clears_frozen_support_rule"] is True
        ]
        u_ids = [
            str(row["context_id"]) for row in non_loop
            if row["localization"]["generator_local_max_excluding_other_owner_strict"]["ambiguity_included_u"]["clears_frozen_support_rule"] is True
        ]
        frontier_tested = any(row["frontier_features"]["frontier_present"] for row in non_loop)
        native_tp = bool(owner["native_true_positive"])
        bank = owner["candidate_bank"]
        blockers: list[str] = []
        if not owner["greedy_eligible"]:
            blockers.append("not_greedy_eligible")
        if bank["bank_coverage_status"] not in legacy.DISPOSITION_ELIGIBLE_BANK_STATUSES:
            blockers.append("bank_undercoverage")
        if not frontier_tested:
            blockers.append("never_frontier_tested")
        if not non_loop:
            blockers.append("no_non_loop_native_context")
        if not any(row["localization"]["exact_anchor_score"] is not None for row in owner_contexts):
            blockers.append("exact_anchor_score_not_reported")
        loop_support = any(
            row["localization"]["generator_local_max_excluding_other_owner_strict"][bound][
                "clears_frozen_support_rule"
            ]
            is True
            for row in owner_contexts
            if row["loop_marking"]["loop_tail"]
            for bound in ("ambiguity_excluded_l", "ambiguity_included_u")
        )
        if loop_support and not l_ids and not u_ids:
            blockers.append("loop_tail_only_support")
        flip = bool(l_ids) != bool(u_ids)
        descriptive_disposition: str
        if flip:
            descriptive_disposition = DISPOSITION_FLIP
        elif l_ids and u_ids:
            descriptive_disposition = DISPOSITION_RESOLVED
        elif blockers:
            descriptive_disposition = DISPOSITION_UNRESOLVED
        else:
            descriptive_disposition = DISPOSITION_PERSISTENT
        if native_tp:
            disposition = DISPOSITION_TP
        elif force_descriptive_only:
            disposition = DISPOSITION_WITHHELD
        elif transfers is False:
            disposition = DISPOSITION_WITHHELD
        else:
            # Underpowered transfer keeps the frozen-rule result visible but
            # explicitly descriptive; it is neither a pass nor a failure.
            disposition = descriptive_disposition
        summaries.append(
            {
                "schema_version": OWNER_SCHEMA_VERSION,
                "row_kind": "image2299_owner_accessibility_summary",
                "gt_owner_id": owner_id, "image_id": IMAGE_ID,
                "normalized_description": str(owner["normalized_description"]),
                "bbox_pixel_xyxy": list(owner["bbox_pixel_xyxy"]),
                "size": _size_fields(owner, plan.images[IMAGE_ID]),
                "greedy_eligible": bool(owner["greedy_eligible"]),
                "native_true_positive": native_tp,
                "native_false_negative": not native_tp,
                "native_due_context_id": due_by_owner.get(owner_id),
                "bank_coverage_status": str(bank["bank_coverage_status"]),
                "ambiguity_bound_disposition_flip": flip,
                "lower_bound_l": {
                    "usable_support": bool(l_ids), "usable_support_context_ids": l_ids,
                    "non_loop_context_support": {
                        str(row["context_id"]): bool(row["localization"]["generator_local_max_excluding_other_owner_strict"]["ambiguity_excluded_l"]["clears_frozen_support_rule"])
                        for row in non_loop
                    },
                },
                "upper_bound_u": {
                    "usable_support": bool(u_ids), "usable_support_context_ids": u_ids,
                    "non_loop_context_support": {
                        str(row["context_id"]): bool(row["localization"]["generator_local_max_excluding_other_owner_strict"]["ambiguity_included_u"]["clears_frozen_support_rule"])
                        for row in non_loop
                    },
                },
                "frontier_tested": frontier_tested,
                "disposition": disposition,
                "frozen_disposition_descriptive": descriptive_disposition,
                "disposition_role": (
                    "cross_checkpoint_sensitivity_only"
                    if force_descriptive_only and not native_tp
                    else
                    "validity_bearing"
                    if transfers is True
                    else "withheld_calibration_nontransfer"
                    if transfers is False
                    else "descriptive_only_calibration_transfer_underpowered"
                ),
                "disposition_blockers": sorted(blockers),
                "fn_disposition_interpretable": bool(
                    transfers is True and not native_tp and not force_descriptive_only
                ),
                "support_criterion": {
                    "calibration_content_sha256": CALIBRATION_CONTENT_SHA256,
                    "theta_peak_lift": calibration.theta_peak_lift,
                    "theta_local_concentration": calibration.theta_local_concentration,
                    "epsilon": calibration.epsilon,
                    "thresholds_retuned": False,
                    "phenotype_fitted": False,
                    "rank_is_support_input": False,
                },
            }
        )
    transfer = {
        "floor": TRANSFER_FLOOR,
        "minimum_native_tp_control_count": MIN_TRANSFER_CONTROL_COUNT,
        "native_tp_owner_count": len(due_by_owner),
        "eligible_due_boundary_count": eligible_due_count,
        "transfer_denominator_native_tp_count": transfer_denominator,
        "supported_due_boundary_count": transfer_numerator,
        "support_rate": transfer_rate,
        "passes": transfers,
        "status": (
            "calibration_transfer_underpowered"
            if not transfer_evaluable
            else "calibration_transfer_passed"
            if transfers
            else "calibration_nontransferring"
        ),
        "validity_bearing": transfer_evaluable,
        "rule": "frozen support clears under both L and U at exact native due boundary",
        "on_underpowered": "retain_continuous_rows_and_frozen_dispositions_descriptively_only",
        "on_failure": "withhold_all_native_fn_dispositions",
        "owner_rows": due_rows,
        "supports_binary_fn_classification": bool(
            transfers is True and not force_descriptive_only
        ),
        "classification_policy": (
            "cross_checkpoint_sensitivity_only"
            if force_descriptive_only
            else "native_tp_transfer_gated"
        ),
    }
    return summaries, transfer


def _breakdown(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = {}
    for row in rows:
        value = str(row["normalized_description"] if key == "category" else row["size"]["minimum_dimension_band"])
        grouped.setdefault(value, Counter())[str(row["disposition"])] += 1
    return {
        value: {"owner_count": sum(counts.values()), "dispositions": dict(sorted(counts.items()))}
        for value, counts in sorted(grouped.items())
    }


def analyze(
    plan_dir: Path,
    shard_dir: Path,
    calibration_path: Path = CALIBRATION_PATH,
    *,
    extension_unit_id: str = UNIT_ID,
    force_descriptive_only: bool = False,
) -> AnalysisProduct:
    plan = _load_plan(
        plan_dir,
        expected_extension_unit_id=extension_unit_id,
    )
    calibration = load_frozen_calibration(calibration_path)
    shard_receipt = _verify_shard_receipt(plan, shard_dir)
    events = _load_score_events(plan, shard_dir)
    proposals = _load_proposal_surfaces(plan, shard_dir)
    counts = shard_receipt.get("counts") or {}
    if int(counts.get("localization_score_rows", -1)) != len(events):
        raise AnalysisContractError("shard receipt localization row count drifted")
    if int(counts.get("proposal_surface_rows", -1)) != len(proposals):
        raise AnalysisContractError("shard receipt proposal row count drifted")
    owner_contexts = _build_owner_contexts(plan, events, proposals, calibration)
    owner_summaries, transfer = _summarize_owners(
        plan,
        owner_contexts,
        calibration,
        force_descriptive_only=force_descriptive_only,
    )
    if len(owner_summaries) != EXPECTED_OWNER_COUNT:
        raise AnalysisContractError("analysis owner denominator is not exactly 46")
    native_fn = [row for row in owner_summaries if row["native_false_negative"]]
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "unit_id": extension_unit_id,
        "image_id": IMAGE_ID,
        "slice_role": "prospective_single_image_separate_from_legacy_12",
        "denominators": {
            "image2299_owner_count": EXPECTED_OWNER_COUNT,
            "image2299_native_tp_count": sum(row["native_true_positive"] for row in owner_summaries),
            "image2299_native_fn_count": len(native_fn),
            "legacy_12_owner_count_unchanged": LEGACY_OWNER_COUNT,
            "legacy_12_eligible_native_fn_denominator_unchanged": LEGACY_FN_DENOMINATOR,
            "pooled_13_image_denominator_created": False,
        },
        "calibration": {
            "path": str(calibration_path),
            "file_sha256": _sha256_file(calibration_path),
            "content_sha256": CALIBRATION_CONTENT_SHA256,
            "theta_peak_lift": calibration.theta_peak_lift,
            "theta_local_concentration": calibration.theta_local_concentration,
            "epsilon": calibration.epsilon,
            "thresholds_retuned": False,
            "phenotype_fitted": False,
        },
        "calibration_transfer": transfer,
        "classification_policy": (
            "cross_checkpoint_sensitivity_only"
            if force_descriptive_only
            else "native_tp_transfer_gated"
        ),
        "fn_disposition_counts": dict(sorted(Counter(str(row["disposition"]) for row in native_fn).items())),
        "fn_breakdown_by_category": _breakdown(native_fn, "category"),
        "fn_breakdown_by_minimum_dimension_band": _breakdown(native_fn, "size"),
        "proposal_localization_contract": {
            "proposal_surface_kept_separate": True,
            "proposal_surface_used_as_support_input": False,
            "localization_estimand": "category_field_support_at_owner_geometry",
            "is_per_owner_proposal_probability": False,
        },
        "stop_rule": {
            "native_fn_below_15": len(native_fn) < 15,
            "later_proportions_descriptive_only": 15 <= len(native_fn) < 20,
            "directional_stress_test_materially_powered": len(native_fn) >= 20,
        },
    }
    contexts = sorted(plan.contexts.values(), key=lambda row: int(row["boundary_index"]))
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": extension_unit_id,
        "plan": {
            "path": str(plan_dir),
            "receipt_content_sha256": plan.receipt_content_sha256,
            "capture_rules_sha256": plan.capture_rules_sha256,
        },
        "scored_shard": {
            "path": str(shard_dir),
            "receipt_sha256": _sha256_file(shard_dir / SHARD_RECEIPT_NAME),
            "scores_sha256": _sha256_file(shard_dir / SCORES_NAME),
            "proposal_surface_sha256": _sha256_file(shard_dir / PROPOSAL_NAME),
            "capture_completeness": shard_receipt["capture_completeness"],
            "primary_file_sha256": {
                name: _sha256_file(shard_dir / name)
                for name in scorer.PRIMARY_OUTPUT_NAMES
            },
        },
        "calibration": {
            "path": str(calibration_path),
            "file_sha256": _sha256_file(calibration_path),
            "content_sha256": CALIBRATION_CONTENT_SHA256,
            "thresholds_retuned": False,
            "phenotype_fitted": False,
        },
        "denominator_contract": analysis["denominators"],
        "output_file_digests": {},
    }
    product = AnalysisProduct(analysis, owner_summaries, owner_contexts, contexts, receipt)
    receipt["output_file_digests"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in sorted(product.files().items())
    }
    receipt["receipt_content_sha256"] = legacy.sha256_json(receipt)
    return product


def commit_analysis(product: AnalysisProduct, output_dir: Path) -> dict[str, str]:
    files = dict(product.files())
    files["receipt.json"] = legacy.canonical_json_bytes(product.receipt) + b"\n"
    output_dir = Path(output_dir)

    def validate_existing() -> None:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise AnalysisContractError(
                f"analysis output exists but is not a regular directory: {output_dir}"
            )
        entries = {path.name: path for path in output_dir.iterdir()}
        expected = set(files)
        observed = set(entries)
        non_files = sorted(
            name
            for name, path in entries.items()
            if path.is_symlink() or not path.is_file()
        )
        if observed != expected or non_files:
            raise AnalysisContractError(
                "existing analysis artifact set is not exact: "
                f"missing={sorted(expected - observed)} "
                f"foreign={sorted(observed - expected)} non_files={non_files}"
            )
        mismatched = sorted(
            name for name, content in files.items() if entries[name].read_bytes() != content
        )
        if mismatched:
            raise AnalysisContractError(
                f"existing analysis artifacts are not byte-identical: {mismatched}"
            )

    result = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in sorted(files.items())
    }
    if output_dir.exists() or output_dir.is_symlink():
        validate_existing()
        return result

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        for name, content in sorted(files.items()):
            (staging / name).write_bytes(content)
        try:
            os.replace(staging, output_dir)
            published = True
        except OSError as exc:
            if not output_dir.exists() and not output_dir.is_symlink():
                raise AnalysisContractError(
                    f"atomic analysis publication failed for {output_dir}"
                ) from exc
            validate_existing()
        return result
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--scored-shard-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--extension-unit-id", default=UNIT_ID)
    parser.add_argument(
        "--force-descriptive-only",
        action="store_true",
        help="Withhold binary FN dispositions while retaining continuous rows and frozen-threshold sensitivity.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        product = analyze(
            args.plan_dir,
            args.scored_shard_dir,
            args.calibration,
            extension_unit_id=args.extension_unit_id,
            force_descriptive_only=args.force_descriptive_only,
        )
        commit_analysis(product, args.output_dir)
    except AnalysisContractError as exc:
        print(f"analysis contract error: {exc}", file=sys.stderr)
        return 2
    transfer = product.analysis["calibration_transfer"]
    print(
        f"image 2299 frozen-support transfer: {transfer['supported_due_boundary_count']}/"
        f"{transfer['transfer_denominator_native_tp_count']} native-TP = "
        f"{transfer['support_rate']:.3f}; "
        f"eligible_due_boundaries={transfer['eligible_due_boundary_count']}; "
        f"passes={transfer['passes']}"
    )
    print(f"receipt_content_sha256 {product.receipt['receipt_content_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

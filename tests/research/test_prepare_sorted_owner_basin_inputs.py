"""Focused contracts for the CPU-only sorted owner-basin input author."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.research.build_sorted_owner_basin_candidates import (
    build_sorted_owner_basin_candidates,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from scripts.research.prepare_sorted_owner_basin_inputs import (
    COHORT_ASSIGNMENT_SCHEMA_VERSION,
    CONTEXT_SCHEMA_VERSION,
    CONTROL_REGISTRY_SCHEMA_VERSION,
    IDENTITY_RECEIPT_SCHEMA_VERSION,
    InputPlanError,
    NON_C_SMOKE_FREEZE_SCHEMA_VERSION,
    SAMPLING_SUPPORT_SCHEMA_VERSION,
    SENTINEL_CONFIRMATION_SCHEMA_VERSION,
    SENTINEL_REGISTRY_SCHEMA_VERSION,
    _task4_calibration_control_ids,
    _validate_task4_production_control_contexts,
    _validate_task4_production_sentinel_contexts,
    prepare_sorted_owner_basin_inputs,
)
from scripts.research.sorted_owner_basin_landscape import (
    CoordinateBox,
    build_semantic_core_payload,
    enumerate_target_x1_anchors,
    enumerate_target_y1_anchors,
    validate_rule_mapping,
    semantic_core_payload,
)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [101, *[200 + ord(character) for character in text]]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _digest(label: str) -> str:
    return sha256_json({"fixture": label})


def _frozen_document(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {**value, field: sha256_json(value)}


def _fixture(
    root: Path,
    *,
    include_scan: bool = True,
    include_background: bool = True,
    decision_eligible: bool = True,
    include_real_task6_mix: bool = False,
    repair_decision_eligible: bool = True,
    repair_cohort: str = "no_free_spatial_support",
) -> dict[str, Any]:
    panel = root / "panel.jsonl"
    panel_objects = [
        {
            "bbox_2d": [
                "<|coord_100|>",
                "<|coord_200|>",
                "<|coord_110|>",
                "<|coord_220|>",
            ],
            "desc": "person",
            "category_id": 1,
        }
    ]
    if include_real_task6_mix:
        panel_objects.append(
            {
                "bbox_2d": [
                    "<|coord_300|>",
                    "<|coord_300|>",
                    "<|coord_310|>",
                    "<|coord_320|>",
                ],
                "desc": "person",
                "category_id": 1,
            }
        )
        panel_objects.append(
            {
                "bbox_2d": [
                    "<|coord_350|>",
                    "<|coord_350|>",
                    "<|coord_360|>",
                    "<|coord_370|>",
                ],
                "desc": "person",
                "category_id": 1,
            }
        )
    _write_jsonl(
        panel,
        [
            {
                "image_id": "17",
                "width": 1000,
                "height": 1000,
                "objects": panel_objects,
            }
        ],
    )
    owner_ledger = root / "owner-ledger.jsonl"
    owner_rows = [
        {
            "schema_version": "sorted-owner-basin-owner-ledger.v2",
            "gt_owner_id": "gt:17:0",
            "diagnostic_owner_id": "diagnostic:gt:17:0",
            "mapped_gt_owner_id": "gt:17:0",
            "image_id": "17",
            "original_annotation_index": 0,
            "description": "person",
            "normalized_description": "person",
            "official_coco_category_id": 1,
            "source_digests": {"panel": sha256_file(panel)},
            "ambiguity_receipt_ids": [],
            "decision_eligibility": {
                "greedy_natural": {"eligible": True, "status": "eligible"},
                "k16_any_hit": {"eligible": True, "status": "eligible"},
                "greedy_k16_paired": {
                    "eligible": decision_eligible,
                    "status": (
                        "eligible"
                        if decision_eligible
                        else "excluded_by_either_policy_global_ambiguity"
                    ),
                },
            },
        }
    ]
    if include_real_task6_mix:
        owner_rows.append(
            {
                **owner_rows[0],
                "gt_owner_id": "gt:17:1",
                "diagnostic_owner_id": "diagnostic:gt:17:1",
                "mapped_gt_owner_id": "gt:17:1",
                "original_annotation_index": 1,
                "decision_eligibility": {
                    "greedy_natural": {"eligible": True, "status": "eligible"},
                    "k16_any_hit": {"eligible": True, "status": "eligible"},
                    "greedy_k16_paired": {
                        "eligible": repair_decision_eligible,
                        "status": (
                            "eligible"
                            if repair_decision_eligible
                            else "excluded_by_either_policy_global_ambiguity"
                        ),
                    },
                },
            }
        )
        owner_rows.append(
            {
                **owner_rows[1],
                "gt_owner_id": "gt:17:2",
                "diagnostic_owner_id": "diagnostic:gt:17:2",
                "mapped_gt_owner_id": "gt:17:2",
                "original_annotation_index": 2,
                "decision_eligibility": {
                    "greedy_natural": {"eligible": True, "status": "eligible"},
                    "k16_any_hit": {"eligible": True, "status": "eligible"},
                    "greedy_k16_paired": {"eligible": True, "status": "eligible"},
                },
            }
        )
    _write_jsonl(owner_ledger, owner_rows)
    execution_receipt = root / "execution-receipt.json"
    execution_receipt_document = _frozen_document(
        {"schema_version": "sorted-owner-basin-task0-execution-receipt.v2"},
        "execution_receipt_content_sha256",
    )
    execution_receipt_id = execution_receipt_document[
        "execution_receipt_content_sha256"
    ]
    _write_json(execution_receipt, execution_receipt_document)
    ambiguity_receipts = root / "ambiguity-receipts.jsonl"
    ambiguity_receipts.write_bytes(b"")
    owner_trajectory_matrix = root / "owner-trajectory-matrix.jsonl"
    _write_jsonl(
        owner_trajectory_matrix,
        [
            {
                "schema_version": "sorted-owner-basin-owner-trajectory-matrix.v2",
                "gt_owner_id": "gt:17:0",
                "policy_stratum": "primary_rp_1.00",
                "decode_mode": "greedy",
                "seed": 0,
                "strict_match_presence": False,
                "max_semantic_compatible_iou": 0.0,
                "ambiguity_receipt_ids": [],
                "global_ambiguity_presence": False,
                "execution_receipt_content_sha256": execution_receipt_id,
            }
        ],
    )
    manifest = root / "artifact-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "sorted-owner-basin-census-artifact-manifest.v2",
            "artifacts": {
                "owner_ledger": {
                    "path": owner_ledger.name,
                    "sha256": sha256_file(owner_ledger),
                },
                "execution_receipt": {
                    "path": execution_receipt.name,
                    "sha256": sha256_file(execution_receipt),
                },
                "ambiguity_receipts": {
                    "path": ambiguity_receipts.name,
                    "sha256": sha256_file(ambiguity_receipts),
                },
                "owner_trajectory_matrix": {
                    "path": owner_trajectory_matrix.name,
                    "sha256": sha256_file(owner_trajectory_matrix),
                },
            },
            "execution_receipt_content_sha256": execution_receipt_id,
            "sources": {
                "panel": {"path": str(panel.resolve()), "sha256": sha256_file(panel)}
            },
            "primary_metrics": {
                "matched_rp_1_0_k16_any_hit": {
                    "ambiguity_neutral_paired_set": {
                        "owner_denominator": (1 if decision_eligible else 0)
                        + int(include_real_task6_mix)
                        * (1 + int(repair_decision_eligible)),
                        "excluded_neutral_gt_owner_ids": sorted(
                            ([] if decision_eligible else ["gt:17:0"])
                            + (
                                []
                                if not include_real_task6_mix
                                or repair_decision_eligible
                                else ["gt:17:1"]
                            )
                        ),
                    }
                }
            },
        },
    )
    task0_digest = sha256_file(manifest)
    source_digests = {"task0_census_artifact_manifest_sha256": task0_digest}

    cohort = root / "cohort-assignments.jsonl"
    cohort_rows: list[dict[str, Any]] = [
        {
            "schema_version": COHORT_ASSIGNMENT_SCHEMA_VERSION,
            "gt_owner_id": "gt:17:0",
            "diagnostic_owner_id": "diagnostic:gt:17:0",
            "image_id": "17",
            "cohort": "no_free_spatial_support",
            "natural_spatial_support": {
                "no_free_spatial_support": True,
                "meaningful_loose_iou_diagnostic": {
                    "metric": "intersection_over_union",
                    "lower_threshold": 0.05,
                    "predeclared_threshold": 0.10,
                    "upper_threshold": 0.15,
                    "cohort_rule_independent_of_diagnostic_thresholds": True,
                },
            },
            "source_digests": source_digests,
        }
    ]
    if include_real_task6_mix:
        cohort_rows.append(
            {
                **cohort_rows[0],
                "gt_owner_id": "gt:17:1",
                "diagnostic_owner_id": "diagnostic:gt:17:1",
                "cohort": repair_cohort,
                **(
                    {
                        "primary_eligibility": {
                            "included_in_b1_or_no_free": False,
                            "included_in_repair_or_calibration": False,
                            "status": (
                                "excluded_unreviewed_or_subthreshold_positive_overlap"
                                if repair_cohort == "positive_overlap_neutral"
                                else "excluded_global_strict_ambiguity"
                            ),
                        },
                        "primary_strict_support": {
                            "global_ambiguity_present": (
                                repair_cohort == "strict_ambiguity_neutral"
                            )
                        },
                        "natural_spatial_support": {
                            **cohort_rows[0]["natural_spatial_support"],
                            "positive_overlap_neutral": (
                                repair_cohort == "positive_overlap_neutral"
                            ),
                        },
                    }
                    if repair_cohort
                    in {
                        "positive_overlap_neutral",
                        "strict_ambiguity_neutral",
                    }
                    else {}
                ),
            }
        )
        cohort_rows.append(
            {
                **cohort_rows[0],
                "gt_owner_id": "gt:17:2",
                "diagnostic_owner_id": "diagnostic:gt:17:2",
            }
        )
    _write_jsonl(cohort, cohort_rows)
    sampling = root / "sampling-support.jsonl"
    sampling_rows = [
        {
            "schema_version": SAMPLING_SUPPORT_SCHEMA_VERSION,
            "support_panel": "matched_primary_k16",
            "registration_id": "fixed-seeds-21001-through-21016",
            "gt_owner_id": "gt:17:0",
            "image_id": "17",
            "policy_stratum": "primary_rp_1.00",
            "source_digests": source_digests,
        }
    ]
    if include_real_task6_mix:
        sampling_rows.append({**sampling_rows[0], "gt_owner_id": "gt:17:1"})
        sampling_rows.append({**sampling_rows[0], "gt_owner_id": "gt:17:2"})
    _write_jsonl(sampling, sampling_rows)
    prompt = [1, 2, 3]
    self_prefix = [4, 5]
    exact = prompt + self_prefix
    contexts = root / "first-skip-contexts.jsonl"
    context_source_digests = {
        **source_digests,
        "task0_census_artifact_manifest": task0_digest,
        "task0_execution_receipt": sha256_file(execution_receipt),
        "task0_ambiguity_receipts": sha256_file(ambiguity_receipts),
    }
    context_rows: list[dict[str, Any]] = [
        {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "record_type": "exact_context",
            "context_id": "context:17:root",
            "context_kind": "root",
            "gt_owner_id": "gt:17:0",
            "image_id": "17",
            "policy_stratum": "primary_rp_1.00",
            "decode_mode": "greedy",
            "seed": 0,
            "status": "admitted",
            "eligibility": "control_diagnostic_only",
            "source_pred_row_id": None,
            "registry_id": "control:17:0",
            "source_binding": {
                "upstream_artifact_digests": context_source_digests,
                "trajectory_id": "trajectory:17:greedy:0",
            },
            "exact_prefix": {
                "prompt_token_ids": prompt,
                "prompt_token_ids_sha256": sha256_json(prompt),
                "self_prefix_generated_token_ids": self_prefix,
                "self_prefix_generated_token_ids_sha256": sha256_json(self_prefix),
                "model_input_token_ids": exact,
                "model_input_token_ids_sha256": sha256_json(exact),
            },
        },
        {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "record_type": "first_skip_candidate_receipt",
            "context_id": "receipt:first-skip:17:0",
            "context_kind": "first_skip_candidate_receipt",
            "gt_owner_id": "gt:17:0",
            "status": "admitted_candidate",
            "source_binding": {
                "upstream_artifact_digests": context_source_digests,
                "trajectory_id": "trajectory:17:greedy:0",
            },
            "exact_prefix": None,
        },
    ]
    if include_real_task6_mix:
        for context_kind in ("P_pre", "P_post"):
            context_rows.append(
                {
                    **context_rows[0],
                    "context_id": f"ctx:first-skip:17:repair:{context_kind}",
                    "context_kind": context_kind,
                    "gt_owner_id": "gt:17:1",
                    "eligibility": (
                        "clean_first_skip_repair_candidate_pending_native_replay"
                    ),
                    "source_pred_row_id": "pred:17:successor",
                    "registry_id": None,
                }
            )
        for role_name, registry_id in (
            ("b1", "control:17:0:b1"),
            ("rescued", "control:17:0:rescued"),
        ):
            context_rows.append(
                {
                    **context_rows[0],
                    "context_id": f"ctx:control:17:{role_name}:root",
                    "eligibility": "control_diagnostic_only",
                    "registry_id": registry_id,
                }
            )
        context_rows.append(
            {
                **context_rows[0],
                "context_id": "ctx:sentinel:17:natural_stop",
                "context_kind": "natural_stop",
                "eligibility": "sentinel_diagnostic_only",
                "registry_id": "sentinel:17:0",
            }
        )
        for context_kind in ("B2_before", "B2_after"):
            context_rows.append(
                {
                    **context_rows[0],
                    "context_id": f"ctx:control:17:b2:{context_kind}",
                    "context_kind": context_kind,
                    "status": "admitted_reviewed_candidate_only",
                    "eligibility": "reviewed_candidate_only",
                    "source_pred_row_id": "pred:17:b2-covering",
                    "registry_id": "control:17:b2",
                }
            )
        context_rows.append(
            {
                **context_rows[0],
                "context_id": "ctx:control:17:b2:reference_scan_position",
                "context_kind": "reference_scan_position",
                "gt_owner_id": "gt:17:2",
                "status": "grounding_or_order_conditioned_accessibility_unresolved",
                "eligibility": "not_control_diagnostic_eligible",
                "source_pred_row_id": None,
                "registry_id": "control:17:b2",
                "exact_prefix": None,
                "details": {
                    "control_ids": ["control:17:b2"],
                    "control_owner_id": "gt:17:2",
                    "control_roles": ["b2_distinct_same_description_pair"],
                    "strata": ["fixture"],
                    "control_original_annotation_index": 2,
                    "unresolved_reason": "no_native_row_follows_control_reference_position",
                    "repair_eligible": False,
                },
            }
        )
    _write_jsonl(contexts, context_rows)
    original_selection = root / "sentinel-selection-receipt.json"
    _write_json(original_selection, {"fixture": "original pre-score selection"})
    task0_chain = {
        "root": str(root.resolve()),
        "artifact_manifest_sha256": task0_digest,
        "execution_receipt_content_sha256": execution_receipt_id,
        "execution_receipt_file_sha256": sha256_file(execution_receipt),
        "owner_ledger_sha256": sha256_file(owner_ledger),
        "owner_trajectory_matrix_sha256": sha256_file(owner_trajectory_matrix),
    }
    confirmation = root / "sentinel-selection-confirmation-receipt.json"
    _write_json(
        confirmation,
        {
            "schema_version": SENTINEL_CONFIRMATION_SCHEMA_VERSION,
            "confirmation_status": (
                "lead_reconfirmed_against_final_task0_v2_before_landscape_scoring"
            ),
            "claim_scope": "outcome_selected_case_studies_only_no_prevalence",
            "anti_leakage_contract": {
                "landscape_scores_used_for_original_selection": False,
                "landscape_scores_used_for_v2_confirmation": False,
                "selection_membership_changed": False,
            },
            "original_selection_receipt": {
                "path": original_selection.name,
                "sha256": sha256_file(original_selection),
            },
            "final_task0_v2": task0_chain,
            "confirmed_sentinels": [
                {
                    "gt_owner_id": "gt:17:0",
                    "trajectory_count": 1,
                    "max_semantic_compatible_iou": 0.0,
                    "strict_match_count": 0,
                    "decision_eligible": True,
                }
            ],
        },
    )
    sentinel = root / "sentinel-registry.json"
    sentinel_source_digests = {
        **source_digests,
        "task0_v2_root": str(root.resolve()),
        "task0_execution_receipt_content_sha256": execution_receipt_id,
        "task0_execution_receipt_file_sha256": sha256_file(execution_receipt),
        "task0_owner_ledger_sha256": sha256_file(owner_ledger),
        "task0_owner_trajectory_matrix_sha256": sha256_file(owner_trajectory_matrix),
    }
    _write_json(
        sentinel,
        {
            "schema_version": SENTINEL_REGISTRY_SCHEMA_VERSION,
            "selection_status": (
                "lead_frozen_before_scoring_and_reconfirmed_against_final_task0_v2"
            ),
            "claim_scope": "outcome_selected_case_studies_only_no_prevalence",
            "source_digests": sentinel_source_digests,
            "selection_receipt": {
                "path": original_selection.name,
                "sha256": sha256_file(original_selection),
            },
            "selection_confirmation_receipt": {
                "path": confirmation.name,
                "sha256": sha256_file(confirmation),
            },
            "sentinels": [
                {
                    "sentinel_id": "sentinel:17:0",
                    "gt_owner_id": "gt:17:0",
                    "prior_non_recovery_status": (
                        "verified_primary_natural_zero_spatial_support"
                    ),
                }
            ],
        },
    )
    provenance = {
        "artifact_path": "/immutable/review.json",
        "artifact_sha256": _digest("review"),
        "source_row_id": "review-row-1",
    }
    members = []
    if include_background:
        members.append(
            {
                "owner_id": "gt:17:0",
                "context_id": "context:17:root",
                "bank_name": "equal_size_background",
                "box": [400, 400, 410, 420],
                "identity_kind": "owner_neutral_foil_geometry",
                "identity_id": "foil-geometry:background-1",
                "extent_submode": "equal_size_background",
                "review_status": "reviewed_explicit_geometry",
                "provenance": provenance,
            }
        )
    if include_scan:
        members.append(
            {
                "owner_id": "gt:17:0",
                "context_id": "context:17:root",
                "bank_name": "sorted_scan",
                "box": [700, 600, 710, 620],
                "identity_kind": "owner_neutral_foil_geometry",
                "identity_id": "foil-geometry:scan-1",
                "extent_submode": "sorted_scan",
                "review_status": "reviewed_explicit_geometry",
                "provenance": provenance,
            }
        )
    if include_real_task6_mix:
        for context_id in (
            "ctx:control:17:b1:root",
            "ctx:control:17:rescued:root",
            "ctx:sentinel:17:natural_stop",
        ):
            if include_background:
                members.append(
                    {
                        "owner_id": "gt:17:0",
                        "context_id": context_id,
                        "bank_name": "equal_size_background",
                        "box": [450, 450, 460, 470],
                        "identity_kind": "owner_neutral_foil_geometry",
                        "identity_id": f"foil-geometry:{context_id}:background",
                        "extent_submode": "equal_size_background",
                        "review_status": "reviewed_explicit_geometry",
                        "provenance": provenance,
                    }
                )
            if include_scan:
                members.append(
                    {
                        "owner_id": "gt:17:0",
                        "context_id": context_id,
                        "bank_name": "sorted_scan",
                        "box": [650, 650, 660, 670],
                        "identity_kind": "owner_neutral_foil_geometry",
                        "identity_id": f"foil-geometry:{context_id}:scan",
                        "extent_submode": "sorted_scan",
                        "review_status": "reviewed_explicit_geometry",
                        "provenance": provenance,
                    }
                )
        for context_kind in ("B2_before", "B2_after"):
            context_id = f"ctx:control:17:b2:{context_kind}"
            if include_background:
                members.append(
                    {
                        "owner_id": "gt:17:0",
                        "context_id": context_id,
                        "bank_name": "equal_size_background",
                        "box": [500, 500, 510, 520],
                        "identity_kind": "owner_neutral_foil_geometry",
                        "identity_id": f"foil-geometry:{context_kind}:background",
                        "extent_submode": "equal_size_background",
                        "review_status": "reviewed_explicit_geometry",
                        "provenance": provenance,
                    }
                )
            if include_scan:
                members.append(
                    {
                        "owner_id": "gt:17:0",
                        "context_id": context_id,
                        "bank_name": "sorted_scan",
                        "box": [600, 600, 610, 620],
                        "identity_kind": "owner_neutral_foil_geometry",
                        "identity_id": f"foil-geometry:{context_kind}:scan",
                        "extent_submode": "sorted_scan",
                        "review_status": "reviewed_explicit_geometry",
                        "provenance": provenance,
                    }
                )
            members.append(
                {
                    "owner_id": "gt:17:0",
                    "context_id": context_id,
                    "bank_name": "covered_same_description",
                    "box": [300, 300, 310, 320],
                    "identity_kind": "gt_owner",
                    "identity_id": "gt:17:2",
                    "extent_submode": "same_description_whole",
                    "review_status": "reviewed_explicit_geometry",
                    "provenance": provenance,
                }
            )
    review_rows: dict[str, dict[str, Any]] = {}
    member_review_ids: list[str] = []
    bank_names = {
        "equal_size_background": "background",
        "sorted_scan": "scan",
        "covered_same_description": "covered",
    }
    for member in members:
        bank_name = bank_names[str(member["bank_name"])]
        review_identity = (
            member["identity_id"] if bank_name == "covered" else "owner_neutral"
        )
        review_key = {
            "gt_owner_id": member["owner_id"],
            "bank_name": bank_name,
            "box": member["box"],
            "extent_submode": member["extent_submode"],
            "identity": review_identity,
        }
        source_row_id = f"fixture-foil-review:{sha256_json(review_key)}"
        member_review_ids.append(source_row_id)
        review_rows.setdefault(
            source_row_id,
            {
                "source_row_id": source_row_id,
                "gt_owner_id": member["owner_id"],
                "image_id": "17",
                "bank_name": bank_name,
                "box": member["box"],
                "extent_submode": member["extent_submode"],
                **(
                    {"identity_gt_owner_id": member["identity_id"]}
                    if bank_name == "covered"
                    else {}
                ),
                "review_statement": "fixture pre-score geometry review",
            },
        )
    foil_review = root / "candidate-bank-foil-review.json"
    _write_json(
        foil_review,
        {
            "schema_version": "sorted-owner-basin-foil-geometry-review.v1",
            "status": "lead_reviewed_and_frozen_before_landscape_scoring",
            "claim_scope": "candidate_bank_geometry_only_no_owner_outcome_or_threshold_selection",
            "anti_leakage_contract": {
                "landscape_scores_used_for_selection": False,
                "forced_continuation_scores_used_for_selection": False,
            },
            "reviewed_geometries": [review_rows[key] for key in sorted(review_rows)],
        },
    )
    foil_review_digest = sha256_file(foil_review)
    for member, source_row_id in zip(members, member_review_ids, strict=True):
        member["provenance"] = {
            "artifact_path": foil_review.name,
            "artifact_sha256": foil_review_digest,
            "source_row_id": source_row_id,
        }

    control = root / "control-registry.json"
    controls = [
        {
            "control_id": "control:17:0",
            "role": "strict_visible_true_positive",
            "gt_owner_id": "gt:17:0",
            "strata": ["fixture"],
        },
        {
            "control_id": "control:17:0:b1",
            "role": "b1_loose_only",
            "gt_owner_id": "gt:17:0",
            "strata": ["fixture"],
        },
        {
            "control_id": "control:17:0:rescued",
            "role": "strict_rescued",
            "gt_owner_id": "gt:17:0",
            "strata": ["fixture"],
        },
    ]
    if include_real_task6_mix:
        controls.append(
            {
                "control_id": "control:17:b2",
                "role": "b2_distinct_same_description_pair",
                "covering_gt_owner_id": "gt:17:2",
                "target_gt_owner_id": "gt:17:0",
                "strata": ["fixture"],
            }
        )
    _write_json(
        control,
        {
            "schema_version": CONTROL_REGISTRY_SCHEMA_VERSION,
            "status": "lead_frozen_before_scoring_and_resealed_to_final_task0_v2",
            "source_digests": {
                "task0_v2_root": str(root.resolve()),
                "task0_census_artifact_manifest_sha256": task0_digest,
                "task0_execution_receipt_content_sha256": execution_receipt_id,
                "task0_execution_receipt_file_sha256": sha256_file(execution_receipt),
                "owner_ledger_sha256": sha256_file(owner_ledger),
                "owner_trajectory_matrix_sha256": sha256_file(owner_trajectory_matrix),
                "sentinel_selection_confirmation_receipt_sha256": sha256_file(
                    confirmation
                ),
                "candidate_bank_foil_review_sha256": foil_review_digest,
            },
            "candidate_bank_review": {
                "path": foil_review.name,
                "sha256": foil_review_digest,
                "status": "lead_reviewed_and_frozen_before_landscape_scoring",
            },
            "controls": controls,
            "candidate_bank_members": members,
        },
    )
    identity = root / "identity-receipt.json"
    identity_content = {
        "schema_version": IDENTITY_RECEIPT_SCHEMA_VERSION,
        "status": "frozen",
        "tokenizer": {"path": str(root), "identity_sha256": _digest("tokenizer")},
        "model": {"identity_sha256": _digest("model")},
        "runtime": {"identity_sha256": _digest("runtime")},
        "coordinate_vocabulary": {
            "coordinate_min": 0,
            "coordinate_max": 999,
            "token_id_start": 1000,
            "token_id_end_exclusive": 2000,
        },
        "model_vocab_size": 100000,
        "schema_tokens": {
            "object_ref_start_token_id": 10,
            "object_ref_end_token_id": 11,
            "box_start_token_id": 12,
            "box_end_token_id": 13,
        },
    }
    _write_json(identity, _frozen_document(identity_content, "receipt_digest"))
    paths = {
        "census_manifest": manifest,
        "owner_ledger": owner_ledger,
        "cohort_assignments": cohort,
        "sampling_support": sampling,
        "exact_contexts": contexts,
        "sentinel_registry": sentinel,
        "sentinel_confirmation_receipt": confirmation,
        "identity_receipt": identity,
        "control_registry": control,
    }
    return {
        "paths": paths,
        "expected": {name: sha256_file(path) for name, path in paths.items()},
    }


def _prepare(
    root: Path,
    output: Path,
    *,
    include_context_ids: tuple[str, ...] = (),
    **fixture_kwargs: Any,
) -> dict[str, Any]:
    fixture = _fixture(root, **fixture_kwargs)
    prepare_sorted_owner_basin_inputs(
        **fixture["paths"],
        output_dir=output,
        expected_sha256=fixture["expected"],
        tokenizer=_Tokenizer(),
        contract_mode="test_fixture",
        include_context_ids=include_context_ids,
    )
    return fixture


def _reseal_fixture_review(fixture: dict[str, Any], review: dict[str, Any]) -> None:
    control_path = fixture["paths"]["control_registry"]
    review_path = control_path.parent / "candidate-bank-foil-review.json"
    _write_json(review_path, review)
    review_digest = sha256_file(review_path)
    control = json.loads(control_path.read_text())
    control["candidate_bank_review"]["sha256"] = review_digest
    control["source_digests"]["candidate_bank_foil_review_sha256"] = review_digest
    for member in control["candidate_bank_members"]:
        member["provenance"]["artifact_sha256"] = review_digest
    _write_json(control_path, control)
    fixture["expected"]["control_registry"] = sha256_file(control_path)


def _freeze_receipt_for_control_rules(
    path: Path, control_rules_path: Path, *, c_outcomes_read: bool = False
) -> dict[str, Any]:
    rules = json.loads(control_rules_path.read_text(encoding="utf-8"))
    receipt = {
        "schema_version": NON_C_SMOKE_FREEZE_SCHEMA_VERSION,
        "status": "passed",
        "control_decision_rules_sha256": sha256_file(control_rules_path),
        "semantic_core_sha256": rules["semantic_core"]["sha256"],
        "control_score_artifact_sha256": _digest("control scores"),
        "control_score_receipt_sha256": _digest("control score receipt"),
        "control_summary_sha256": _digest("control summary"),
        "control_summary_receipt_sha256": _digest("control summary receipt"),
        "calibration_receipt_sha256": _digest("calibration receipt"),
        "independent_reconstruction": "passed",
        "gates": {
            "representative_positive_control": "passed",
            "mandatory_cache_parity": "passed",
            "b2_reviewed_pair": "passed",
            "free_surface_executed_raw_only": "passed",
        },
        "c_outcomes_read": c_outcomes_read,
        "scientific_conclusion": None,
    }
    _write_json(path, receipt)
    return receipt


def test_production_task4_control_contexts_require_exact_7511_membership() -> None:
    strict_id = "control:smoke:strict-visible:7511:22"
    b1_id = "control:smoke:b1:7511:26"
    b2_id = "control:smoke:b2:7511:22-to-26"
    controls = [
        {"control_id": strict_id, "role": "strict_visible_true_positive"},
        {"control_id": b1_id, "role": "b1_loose_only"},
        {"control_id": b2_id, "role": "b2_distinct_same_description_pair"},
    ]
    contexts = [
        {
            "context_id": "ctx:control-owner:gt:7511:22:root",
            "registry_id": b2_id,
            "details": {
                "control_ids": [b2_id, strict_id],
                "control_roles": [
                    "b2_distinct_same_description_pair",
                    "strict_visible_true_positive",
                ],
            },
        },
        {
            "context_id": "ctx:control-owner:gt:7511:26:root",
            "registry_id": b1_id,
            "details": {
                "control_ids": [b1_id, b2_id],
                "control_roles": [
                    "b1_loose_only",
                    "b2_distinct_same_description_pair",
                ],
            },
        },
        {
            "context_id": "ctx:B2:control:smoke:b2:7511:22-to-26:B2_before",
            "registry_id": b2_id,
            "details": {"control_ids": [b2_id]},
        },
        {
            "context_id": "ctx:B2:control:smoke:b2:7511:22-to-26:B2_after",
            "registry_id": b2_id,
            "details": {"control_ids": [b2_id]},
        },
    ]

    _validate_task4_production_control_contexts(contexts, controls)

    cross_image_strict_id = "control:wall-bowl:strict-visible:13923:5"
    cross_image_contexts = [
        {
            **contexts[0],
            "context_id": "ctx:control-owner:gt:13923:5:root",
            "registry_id": cross_image_strict_id,
            "details": {
                "control_ids": [cross_image_strict_id],
                "control_roles": ["strict_visible_true_positive"],
            },
        },
        *contexts[1:],
    ]
    with pytest.raises(InputPlanError, match="exact authorized four image-7511"):
        _validate_task4_production_control_contexts(
            cross_image_contexts,
            [
                *controls,
                {
                    "control_id": cross_image_strict_id,
                    "role": "strict_visible_true_positive",
                },
            ],
        )


def test_production_task4_calibration_uses_only_representative_smoke_controls() -> None:
    controls = [
        {
            "control_id": "control:smoke:strict-visible:7511:22",
            "role": "strict_visible_true_positive",
            "strata": ["representative_smoke", "far_person_calibration"],
        },
        {
            "control_id": "control:smoke:b1:7511:26",
            "role": "b1_loose_only",
            "strata": ["representative_smoke", "far_person_calibration"],
        },
        {
            "control_id": "control:far-person:b1:7511:15",
            "role": "b1_loose_only",
            "strata": ["far_person_calibration"],
        },
        {
            "control_id": "control:wall-bowl:strict-visible:13923:5",
            "role": "strict_visible_true_positive",
            "strata": ["wall_bowl_fallback_calibration"],
        },
        {
            "control_id": "control:wall-bowl:b1:16228:47",
            "role": "b1_loose_only",
            "strata": ["wall_bowl_fallback_calibration"],
        },
    ]
    assert _task4_calibration_control_ids(
        controls, contract_mode="production"
    ) == [
        "control:smoke:b1:7511:26",
        "control:smoke:strict-visible:7511:22",
    ]

    stale = [dict(control) for control in controls]
    stale[0] = {**stale[0], "strata": ["far_person_calibration"]}
    with pytest.raises(InputPlanError, match="stale role or stratum"):
        _task4_calibration_control_ids(stale, contract_mode="production")


def test_production_task4_sentinel_is_exact_far_person_root() -> None:
    sentinel = {
        "context_id": "ctx:sentinel:sentinel:far-person:7511:10:root",
        "gt_owner_id": "gt:7511:10",
        "registry_id": "sentinel:far-person:7511:10",
        "context_kind": "root",
        "eligibility": "sentinel_diagnostic_only",
    }
    _validate_task4_production_sentinel_contexts([sentinel])

    with pytest.raises(InputPlanError, match="exact far-person"):
        _validate_task4_production_sentinel_contexts(
            [{**sentinel, "context_id": "ctx:sentinel:alternate:root"}]
        )
    with pytest.raises(InputPlanError, match="exactly one sentinel context"):
        _validate_task4_production_sentinel_contexts(
            [sentinel, {**sentinel, "context_id": "ctx:sentinel:alternate:root"}]
        )


def test_full_interior_plus_size_aware_margin_and_valid_extent_grid(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    _prepare(tmp_path / "inputs", output)
    rules_document = json.loads((output / "landscape-decision-rules.json").read_text())
    receipt = json.loads((output / "input-plan-receipt.json").read_text())
    rules = validate_rule_mapping(rules_document)
    box = CoordinateBox.from_values(100, 200, 110, 220)

    assert [item.value for item in enumerate_target_x1_anchors(box, rules)] == list(
        range(98, 112)
    )
    assert [item.value for item in enumerate_target_y1_anchors(box, rules)] == list(
        range(198, 222)
    )
    seeds = json.loads((output / "candidate-bank-seeds.json").read_text())["seeds"]
    target = [seed for seed in seeds if seed["bank_name"] == "target"]
    extent_ids = {seed["extent_grid_member"]["extent_id"] for seed in target}
    assert {
        "gt_whole",
        "scale_0p80",
        "scale_1p20",
        "aspect_0p80",
        "aspect_1p20",
    } <= extent_ids
    assert all(
        seed["box"][2] > seed["box"][0] and seed["box"][3] > seed["box"][1]
        for seed in seeds
    )
    assert rules_document["structural_status"] == "draft_pre_smoke"
    assert rules_document["calibration"]["numeric_decision_threshold"] is None
    assert rules_document["free_coordinate_tree"]["execution_status"].startswith(
        "declared_not_executed"
    )
    assert receipt["free_coordinate_tree"]["executed"] is False
    assert (
        rules_document["free_coordinate_tree"]["selector"]
        == receipt["free_coordinate_tree"]["selector"]
    )
    selector = rules_document["free_coordinate_tree"]["selector"]
    assert selector["x1_selection"] == "top_64_by_raw_x1_log_probability"
    assert (
        selector["anchor_pool_selection"]["tie_break"]
        == "higher_joint_raw_then_ascending_x1_then_y1"
    )
    assert (
        selector["anchor_pool_selection"]["stop_when_max_min_distance_below_bins"] == 24
    )
    assert selector["null_semantics"] == "bounded_search_null_is_non_evidence"
    assert receipt["c_label_emitted"] is False


def test_equal_measure_equal_size_foils_and_exact_token_binding(tmp_path: Path) -> None:
    output = tmp_path / "out"
    _prepare(tmp_path / "inputs", output)
    rules = json.loads((output / "landscape-decision-rules.json").read_text())
    ledger = json.loads((output / "owner-context-ledger.jsonl").read_text().strip())
    seeds = json.loads((output / "candidate-bank-seeds.json").read_text())["seeds"]
    background = next(seed for seed in seeds if seed["bank_name"] == "background")

    assert (
        rules["bank_proposal_measure"]["target"]
        == rules["bank_proposal_measure"]["background"]
    )
    assert background["box"][2] - background["box"][0] == 10
    assert background["box"][3] - background["box"][1] == 20
    assert ledger["canonical_description"]["token_ids"] == _Tokenizer().encode(
        "person", add_special_tokens=False
    )
    assert ledger["canonical_description"]["token_ids_sha256"] == sha256_json(
        ledger["canonical_description"]["token_ids"]
    )
    assert ledger["context_tokens"]["token_ids"] == [1, 2, 3, 4, 5]
    assert ledger["prompt_prefix_token_count"] == 3
    assert ledger["context_tokens"]["split"] == {
        "prompt": [0, 3],
        "self_prefix": [3, 5],
    }
    assert ledger["vocabulary_attestation"]["tokenizer_identity_sha256"] == _digest(
        "tokenizer"
    )
    registry = rules["token_registry"]
    assert (
        len(registry["coordinate_bin_to_token_id"]["coordinate_bin_token_ids"]) == 1000
    )
    assert registry["coordinate_bin_to_token_id"]["coordinate_bin_token_ids"][:2] == [
        1000,
        1001,
    ]
    assert ledger["token_registry"] == registry


def test_score_fields_are_forbidden_before_membership_authoring(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs")
    control_path = fixture["paths"]["control_registry"]
    control = json.loads(control_path.read_text())
    control["candidate_bank_members"][0]["preview_score"] = -17.0
    _write_json(control_path, control)
    fixture["expected"]["control_registry"] = sha256_file(control_path)

    with pytest.raises(InputPlanError, match="forbidden score-dependent field"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )
    assert not (tmp_path / "out").exists()


def test_candidate_review_file_hash_is_strictly_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs")
    review_path = (
        fixture["paths"]["control_registry"].parent / "candidate-bank-foil-review.json"
    )
    review = json.loads(review_path.read_text())
    review["review_date_utc"] = "2099-01-01"
    _write_json(review_path, review)

    with pytest.raises(InputPlanError, match="foil review digest is stale"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing review row"),
        ("duplicate", "duplicate candidate-bank review row"),
        ("geometry_drift", "drifts from review row"),
        ("score_field", "forbidden score-dependent field"),
    ],
)
def test_candidate_members_must_match_unique_score_blind_review_rows(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    review_path = (
        fixture["paths"]["control_registry"].parent / "candidate-bank-foil-review.json"
    )
    review = json.loads(review_path.read_text())
    if mutation == "missing":
        review["reviewed_geometries"] = []
    elif mutation == "duplicate":
        review["reviewed_geometries"].append(dict(review["reviewed_geometries"][0]))
    elif mutation == "geometry_drift":
        review["reviewed_geometries"][0]["box"] = [401, 400, 411, 420]
    elif mutation == "score_field":
        review["reviewed_geometries"][0]["preview_score"] = -1.0
    else:
        raise AssertionError(mutation)
    _reseal_fixture_review(fixture, review)

    with pytest.raises(InputPlanError, match=message):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )


def test_missing_required_control_role_keeps_calibration_unresolved(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    control_path = fixture["paths"]["control_registry"]
    control = json.loads(control_path.read_text())
    control["controls"] = [
        item for item in control["controls"] if item["role"] != "strict_rescued"
    ]
    _write_json(control_path, control)
    fixture["expected"]["control_registry"] = sha256_file(control_path)

    with pytest.raises(InputPlanError, match="calibration unresolved: missing roles"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )


def test_task0_paired_neutral_owner_cannot_enter_decision_inputs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs", decision_eligible=False)
    with pytest.raises(InputPlanError, match="decision-neutral ambiguity owner"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )


@pytest.mark.parametrize(
    "neutral_cohort", ["positive_overlap_neutral", "strict_ambiguity_neutral"]
)
def test_neutral_cohorts_parse_but_cannot_select_a_decision_context(
    tmp_path: Path, neutral_cohort: str
) -> None:
    fixture = _fixture(
        tmp_path / "inputs",
        include_real_task6_mix=True,
        repair_decision_eligible=False,
        repair_cohort=neutral_cohort,
    )
    output = tmp_path / "full"
    prepare_sorted_owner_basin_inputs(
        **fixture["paths"],
        output_dir=output,
        expected_sha256=fixture["expected"],
        tokenizer=_Tokenizer(),
        contract_mode="test_fixture",
    )
    ledger = [
        json.loads(line)
        for line in (output / "owner-context-ledger.jsonl").read_text().splitlines()
    ]
    assert all(row["gt_owner_id"] != "gt:17:1" for row in ledger)

    with pytest.raises(InputPlanError, match="registry-ineligible contexts"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "selected",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
            include_context_ids=("ctx:first-skip:17:repair:P_pre",),
        )


def test_confirmation_receipt_hash_is_bound_through_both_registries(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    confirmation_path = fixture["paths"]["sentinel_confirmation_receipt"]
    confirmation = json.loads(confirmation_path.read_text())
    confirmation["confirmation_date_utc"] = "2099-01-01"
    _write_json(confirmation_path, confirmation)
    fixture["expected"]["sentinel_confirmation_receipt"] = sha256_file(
        confirmation_path
    )

    with pytest.raises(InputPlanError, match="confirmation receipt digest is stale"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )
    assert not (tmp_path / "out").exists()


def test_confirmation_non_recovery_assertion_is_revalidated(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs")
    confirmation_path = fixture["paths"]["sentinel_confirmation_receipt"]
    confirmation = json.loads(confirmation_path.read_text())
    confirmation["confirmed_sentinels"][0]["strict_match_count"] = 1
    _write_json(confirmation_path, confirmation)
    confirmation_digest = sha256_file(confirmation_path)
    fixture["expected"]["sentinel_confirmation_receipt"] = confirmation_digest

    sentinel_path = fixture["paths"]["sentinel_registry"]
    sentinel = json.loads(sentinel_path.read_text())
    sentinel["selection_confirmation_receipt"]["sha256"] = confirmation_digest
    _write_json(sentinel_path, sentinel)
    fixture["expected"]["sentinel_registry"] = sha256_file(sentinel_path)

    control_path = fixture["paths"]["control_registry"]
    control = json.loads(control_path.read_text())
    control["source_digests"]["sentinel_selection_confirmation_receipt_sha256"] = (
        confirmation_digest
    )
    _write_json(control_path, control)
    fixture["expected"]["control_registry"] = sha256_file(control_path)

    with pytest.raises(InputPlanError, match="violates eligibility/non-recovery"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )


def test_production_rejects_nonfinal_original_selection_receipt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    with pytest.raises(InputPlanError, match="original sentinel selection receipt"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
        )


@pytest.mark.parametrize("missing", ["scan", "background"])
def test_missing_required_foil_keeps_context_unresolved_and_writes_nothing(
    tmp_path: Path, missing: str
) -> None:
    fixture = _fixture(
        tmp_path / "inputs",
        include_scan=missing != "scan",
        include_background=missing != "background",
    )
    with pytest.raises(InputPlanError, match="unresolved: missing reviewed foil banks"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )
    assert not (tmp_path / "out").exists()


def test_stale_exact_context_digest_fails_before_writing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs")
    context_path = fixture["paths"]["exact_contexts"]
    rows = [json.loads(line) for line in context_path.read_text().splitlines()]
    exact_row = next(row for row in rows if row["record_type"] == "exact_context")
    exact_row["exact_prefix"]["model_input_token_ids_sha256"] = _digest("stale")
    _write_jsonl(context_path, rows)
    fixture["expected"]["exact_contexts"] = sha256_file(context_path)

    with pytest.raises(InputPlanError, match="stale literal token ids"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )


def test_task6_audit_receipt_must_bind_final_task0_chain(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs")
    context_path = fixture["paths"]["exact_contexts"]
    rows = [json.loads(line) for line in context_path.read_text().splitlines()]
    audit_row = next(
        row for row in rows if row["record_type"] == "first_skip_candidate_receipt"
    )
    audit_row["source_binding"]["upstream_artifact_digests"][
        "task0_execution_receipt"
    ] = _digest("stale execution receipt")
    _write_jsonl(context_path, rows)
    fixture["expected"]["exact_contexts"] = sha256_file(context_path)

    with pytest.raises(InputPlanError, match="disagrees with final Task-0 v2"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )
    assert not (tmp_path / "out").exists()


def test_task6_repair_contexts_are_audited_but_not_selected(tmp_path: Path) -> None:
    output = tmp_path / "out"
    _prepare(tmp_path / "inputs", output, include_real_task6_mix=True)
    rules = json.loads((output / "landscape-decision-rules.json").read_text())
    receipt = json.loads((output / "input-plan-receipt.json").read_text())
    ledger_rows = [
        json.loads(line)
        for line in (output / "owner-context-ledger.jsonl").read_text().splitlines()
    ]
    seeds = json.loads((output / "candidate-bank-seeds.json").read_text())["seeds"]

    skipped = [
        "ctx:first-skip:17:repair:P_post",
        "ctx:first-skip:17:repair:P_pre",
    ]
    selection = rules["task6_context_selection"]
    assert selection["mode"] == "all_resolved_sealed"
    assert selection["requested_context_ids"] == []
    assert selection["skipped_resolved_context_ids"] == []
    assert selection["skipped_repair_context_ids"] == skipped
    assert selection["skipped_repair_context_count"] == 2
    assert selection["score_repair_contexts"] is False
    assert receipt["task6_context_selection"] == selection
    assert receipt["counts"]["skipped_task7_task8_repair_contexts"] == 2
    assert selection["skipped_unresolved_control_context_ids"] == [
        "ctx:control:17:b2:reference_scan_position"
    ]
    assert selection["skipped_unresolved_control_context_count"] == 1
    assert selection["unresolved_sentinel_contexts_skipped"] is False
    assert receipt["counts"]["skipped_unresolved_control_contexts"] == 1
    assert {row["context_provenance"]["context_kind"] for row in ledger_rows} == {
        "root",
        "B2_before",
        "B2_after",
    }
    assert all(row["gt_owner_id"] == "gt:17:0" for row in ledger_rows)
    assert all(seed["diagnostic_owner_id"] != "diagnostic:gt:17:1" for seed in seeds)


def test_explicit_four_context_selection_is_canonical_and_deterministic(
    tmp_path: Path,
) -> None:
    selected = (
        "context:17:root",
        "ctx:control:17:b1:root",
        "ctx:control:17:b2:B2_before",
        "ctx:control:17:b2:B2_after",
    )
    fixture = _fixture(tmp_path / "inputs", include_real_task6_mix=True)
    for output, requested in (
        (tmp_path / "forward", selected),
        (tmp_path / "reverse", tuple(reversed(selected))),
    ):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=output,
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
            include_context_ids=requested,
        )
    for name in (
        "landscape-decision-rules.json",
        "owner-context-ledger.jsonl",
        "candidate-bank-seeds.json",
        "input-plan-receipt.json",
    ):
        assert (tmp_path / "forward" / name).read_bytes() == (
            tmp_path / "reverse" / name
        ).read_bytes()

    rules = json.loads(
        (tmp_path / "forward" / "landscape-decision-rules.json").read_text()
    )
    receipt = json.loads((tmp_path / "forward" / "input-plan-receipt.json").read_text())
    ledger = [
        json.loads(line)
        for line in (tmp_path / "forward" / "owner-context-ledger.jsonl")
        .read_text()
        .splitlines()
    ]
    seeds = json.loads((tmp_path / "forward" / "candidate-bank-seeds.json").read_text())
    selection = rules["task6_context_selection"]
    expected_payload = {
        "mode": "explicit_context_ids",
        "requested_context_ids": sorted(selected),
        "emitted_context_ids": sorted(selected),
        "skipped_resolved_context_ids": [],
        "plan_membership": "task4_control_only",
    }
    assert all(selection[key] == value for key, value in expected_payload.items())
    assert selection["context_selection_sha256"] == sha256_json(expected_payload)
    assert receipt["task6_context_selection"] == selection
    assert {row["context_id"] for row in ledger} == set(selected)
    assert {seed["context_id"] for seed in seeds["seeds"]} == set(selected)
    assert seeds["context_selection_sha256"] == selection["context_selection_sha256"]


def test_control_and_sentinel_plans_have_distinct_outer_sha_and_identical_core(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs", include_real_task6_mix=True)
    control_contexts = (
        "context:17:root",
        "ctx:control:17:b1:root",
        "ctx:control:17:b2:B2_before",
        "ctx:control:17:b2:B2_after",
    )
    control_output = tmp_path / "control"
    prepare_sorted_owner_basin_inputs(
        **fixture["paths"],
        output_dir=control_output,
        expected_sha256=fixture["expected"],
        tokenizer=_Tokenizer(),
        contract_mode="test_fixture",
        include_context_ids=control_contexts,
    )
    freeze_path = tmp_path / "non-c-smoke-freeze-receipt.json"
    _freeze_receipt_for_control_rules(
        freeze_path, control_output / "landscape-decision-rules.json"
    )
    sentinel_expected = {
        **fixture["expected"],
        "non_c_smoke_freeze_receipt": sha256_file(freeze_path),
    }
    sentinel_output = tmp_path / "sentinel"
    prepare_sorted_owner_basin_inputs(
        **fixture["paths"],
        output_dir=sentinel_output,
        expected_sha256=sentinel_expected,
        tokenizer=_Tokenizer(),
        contract_mode="test_fixture",
        structural_status="sealed_non_c_smoke",
        include_context_ids=("ctx:sentinel:17:natural_stop",),
        non_c_smoke_freeze_receipt=freeze_path,
    )
    control_rules_path = control_output / "landscape-decision-rules.json"
    sentinel_rules_path = sentinel_output / "landscape-decision-rules.json"
    control_rules = json.loads(control_rules_path.read_text(encoding="utf-8"))
    sentinel_rules = json.loads(sentinel_rules_path.read_text(encoding="utf-8"))

    assert sha256_file(control_rules_path) != sha256_file(sentinel_rules_path)
    assert control_rules["semantic_core"] == sentinel_rules["semantic_core"]
    assert semantic_core_payload(control_rules) == semantic_core_payload(sentinel_rules)
    changed_top_level_fields = {
        key
        for key in set(control_rules) | set(sentinel_rules)
        if control_rules.get(key) != sentinel_rules.get(key)
    }
    assert changed_top_level_fields == {
        "structural_status",
        "task6_context_selection",
        "candidate_materializer",
        "sealed_inputs",
        "non_c_smoke_freeze_receipt",
    }
    assert control_rules["task6_context_selection"]["plan_membership"] == "task4_control_only"
    assert sentinel_rules["task6_context_selection"]["plan_membership"] == "task4_sentinel_only"
    assert control_rules["task6_context_selection"]["emitted_context_ids"] == sorted(
        control_contexts
    )
    assert sentinel_rules["task6_context_selection"]["emitted_context_ids"] == [
        "ctx:sentinel:17:natural_stop"
    ]
    assert "non_c_smoke_freeze_receipt" not in control_rules
    assert sentinel_rules["non_c_smoke_freeze_receipt"]["sha256"] == sha256_file(
        freeze_path
    )
    sentinel_seeds = json.loads(
        (sentinel_output / "candidate-bank-seeds.json").read_text(encoding="utf-8")
    )
    sentinel_receipt = json.loads(
        (sentinel_output / "input-plan-receipt.json").read_text(encoding="utf-8")
    )
    assert sentinel_seeds["non_c_smoke_freeze_receipt_sha256"] == sha256_file(
        freeze_path
    )
    assert sentinel_receipt["non_c_smoke_freeze_receipt_sha256"] == sha256_file(
        freeze_path
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("stale_expected", "digest is stale"),
        ("wrong_core", "different semantic core"),
        ("c_read", "not C-blind"),
    ],
)
def test_sealed_sentinel_rejects_stale_wrong_or_c_read_freeze(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixture = _fixture(tmp_path / "inputs", include_real_task6_mix=True)
    control_output = tmp_path / "control"
    prepare_sorted_owner_basin_inputs(
        **fixture["paths"],
        output_dir=control_output,
        expected_sha256=fixture["expected"],
        tokenizer=_Tokenizer(),
        contract_mode="test_fixture",
        include_context_ids=(
            "context:17:root",
            "ctx:control:17:b1:root",
            "ctx:control:17:b2:B2_before",
            "ctx:control:17:b2:B2_after",
        ),
    )
    freeze_path = tmp_path / "freeze.json"
    receipt = _freeze_receipt_for_control_rules(
        freeze_path,
        control_output / "landscape-decision-rules.json",
        c_outcomes_read=mutation == "c_read",
    )
    if mutation == "wrong_core":
        receipt["semantic_core_sha256"] = "f" * 64
        _write_json(freeze_path, receipt)
    expected_freeze = (
        "0" * 64 if mutation == "stale_expected" else sha256_file(freeze_path)
    )
    with pytest.raises(InputPlanError, match=message):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "sentinel",
            expected_sha256={
                **fixture["expected"],
                "non_c_smoke_freeze_receipt": expected_freeze,
            },
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
            structural_status="sealed_non_c_smoke",
            include_context_ids=("ctx:sentinel:17:natural_stop",),
            non_c_smoke_freeze_receipt=freeze_path,
        )


def test_sealed_sentinel_requires_separate_freeze_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs", include_real_task6_mix=True)
    with pytest.raises(InputPlanError, match="requires --non-c-smoke-freeze-receipt"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
            structural_status="sealed_non_c_smoke",
            include_context_ids=("ctx:sentinel:17:natural_stop",),
        )


def test_semantic_drift_changes_core_digest_and_stale_binding_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    _prepare(tmp_path / "inputs", output)
    rules = json.loads((output / "landscape-decision-rules.json").read_text())
    original_digest = rules["semantic_core"]["sha256"]
    rules["free_coordinate_tree"]["selector"]["x1_selection"] = (
        "top_63_by_raw_x1_log_probability"
    )
    with pytest.raises(ValueError, match="not the canonical projection"):
        semantic_core_payload(rules)
    changed_payload = build_semantic_core_payload(rules)
    assert sha256_json(changed_payload) != original_digest


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        ("ctx:does-not-exist", "unknown context ids"),
        (
            "ctx:control:17:b2:reference_scan_position",
            "unresolved control contexts",
        ),
        ("ctx:first-skip:17:repair:P_pre", "repair-only contexts"),
    ],
)
def test_explicit_context_selection_rejects_non_emittable_ids(
    tmp_path: Path, requested: str, message: str
) -> None:
    fixture = _fixture(tmp_path / "inputs", include_real_task6_mix=True)
    with pytest.raises(InputPlanError, match=message):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
            include_context_ids=(requested,),
        )


def test_explicit_context_selection_rejects_duplicates(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs", include_real_task6_mix=True)
    with pytest.raises(InputPlanError, match="duplicate context id"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
            include_context_ids=("context:17:root", "context:17:root"),
        )


def test_unresolved_sentinel_reference_context_remains_blocking(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    context_path = fixture["paths"]["exact_contexts"]
    rows = [json.loads(line) for line in context_path.read_text().splitlines()]
    sentinel = next(row for row in rows if row["record_type"] == "exact_context")
    sentinel.update(
        {
            "context_id": "ctx:sentinel:17:reference_scan_position",
            "context_kind": "reference_scan_position",
            "status": "grounding_or_order_conditioned_accessibility_unresolved",
            "eligibility": "not_c_eligible_not_repair_eligible",
            "source_pred_row_id": None,
            "exact_prefix": None,
        }
    )
    _write_jsonl(context_path, rows)
    fixture["expected"]["exact_contexts"] = sha256_file(context_path)

    with pytest.raises(InputPlanError, match="explicit control-only exception"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=tmp_path / "out",
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )


def test_same_inputs_produce_byte_identical_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs")
    for output in (tmp_path / "one", tmp_path / "two"):
        prepare_sorted_owner_basin_inputs(
            **fixture["paths"],
            output_dir=output,
            expected_sha256=fixture["expected"],
            tokenizer=_Tokenizer(),
            contract_mode="test_fixture",
        )
    for name in (
        "landscape-decision-rules.json",
        "owner-context-ledger.jsonl",
        "candidate-bank-seeds.json",
        "input-plan-receipt.json",
    ):
        assert (tmp_path / "one" / name).read_bytes() == (
            tmp_path / "two" / name
        ).read_bytes()


def test_authored_files_are_accepted_by_candidate_builder_v2(tmp_path: Path) -> None:
    authored = tmp_path / "authored"
    _prepare(tmp_path / "inputs", authored)
    rules = authored / "landscape-decision-rules.json"
    ledger = authored / "owner-context-ledger.jsonl"
    seeds = authored / "candidate-bank-seeds.json"
    receipt = build_sorted_owner_basin_candidates(
        owner_context_ledger=ledger,
        landscape_decision_rules=rules,
        bank_seeds=seeds,
        output_jsonl=tmp_path / "landscape-candidates.jsonl",
        receipt=tmp_path / "landscape-candidates-receipt.json",
        expected_owner_context_ledger_sha256=sha256_file(ledger),
        expected_landscape_decision_rules_sha256=sha256_file(rules),
        expected_bank_seeds_sha256=sha256_file(seeds),
    )
    assert (
        receipt["candidate_membership"]["score_dependent_selection_executed"] is False
    )
    assert receipt["counts"]["conditional_y1_score_plans"] == 14


def test_direct_cli_help(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/prepare_sorted_owner_basin_inputs.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--cohort-assignments" in result.stdout
    assert "--exact-contexts" in result.stdout

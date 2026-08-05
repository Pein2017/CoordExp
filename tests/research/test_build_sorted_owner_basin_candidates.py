"""CPU contract tests for pre-score sorted-owner candidate materialization."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.research.build_sorted_owner_basin_candidates import (
    BANK_SEEDS_SCHEMA_VERSION,
    COORDINATE_SPACE_NAME,
    MATERIALIZER_RULES_SCHEMA_VERSION,
    OFFICIAL_COCO_NAMESPACE,
    OWNER_CONTEXT_LEDGER_SCHEMA_VERSION,
    build_sorted_owner_basin_candidates,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from scripts.research.sorted_owner_basin_landscape import (
    BasinRegistration,
    RULES_SCHEMA_VERSION,
)


def _digest(label: str) -> str:
    return sha256_json({"fixture": label})


def _space() -> dict:
    return {
        "name": COORDINATE_SPACE_NAME,
        "min": 0,
        "max": 9,
        "contract_kind": "test_fixture",
        "bin_to_extent_conversion": "floor",
        "non_production_reason": "small complete-vocabulary CPU fixture",
    }


def _vocabulary_attestation() -> dict:
    return {
        "tokenizer_identity_sha256": _digest("tokenizer"),
        "model_identity_sha256": _digest("model"),
        "runtime_identity_sha256": _digest("runtime"),
    }


def _token_registry() -> dict:
    coordinate_token_ids = list(range(1000, 2000))
    payload = {
        "schema_tokens": {
            "object_ref_start_token_id": 900,
            "object_ref_end_token_id": 901,
            "box_start_token_id": 902,
            "box_end_token_id": 903,
        },
        "coordinate_bin_to_token_id": {
            "bin_min": 0,
            "bin_max": 999,
            "token_id_start": 1000,
            "token_id_end_exclusive": 2000,
            "mapping": "fixture_explicit_registry",
            "coordinate_bin_token_ids": coordinate_token_ids,
            "coordinate_bin_token_ids_sha256": sha256_json(coordinate_token_ids),
        },
        "model_vocab_size": 4096,
        "vocabulary_attestation": _vocabulary_attestation(),
        "identity_receipt_digest": _digest("identity-receipt"),
    }
    return {**payload, "registry_sha256": sha256_json(payload)}


def _structural_row_wrapper(registry: dict) -> dict:
    return {
        "composition": [
            "object_ref_start_token_id",
            "canonical_description_token_ids",
            "object_ref_end_token_id",
            "box_start_token_id",
            "x1_coordinate_token_id",
            "y1_coordinate_token_id",
            "x2_coordinate_token_id",
            "y2_coordinate_token_id",
            "box_end_token_id",
        ],
        "schema_tokens": registry["schema_tokens"],
        "coordinate_bin_token_ids_sha256": registry["coordinate_bin_to_token_id"][
            "coordinate_bin_token_ids_sha256"
        ],
    }


def _score_channels() -> dict:
    return {
        "raw_fp32": {
            "role": "primary_model_likelihood",
            "repetition_penalty": None,
        },
        "rp_1_00": {"role": "auxiliary_policy_score", "repetition_penalty": 1.0},
        "rp_1_10": {"role": "auxiliary_policy_score", "repetition_penalty": 1.10},
        "shared_forward_rule": "derive both policy views from one byte-identical raw forward",
    }


def _provenance(source_row_id: str) -> dict:
    return {
        "artifact_path": "/immutable/reviewed-foils.jsonl",
        "artifact_sha256": _digest("reviewed-foils"),
        "source_row_id": source_row_id,
    }


def _rules() -> dict:
    upstream = {
        "owner_ledger": _digest("owner-ledger"),
        "context_ledger": _digest("context-ledger"),
    }
    members = [
        {
            "foil_member_id": "background-1",
            "diagnostic_owner_id": "diag:17:gt-0",
            "context_id": "ctx:17:pre-row-3",
            "bank_name": "background",
            "source_id": "background-1",
            "identity_kind": "registered_geometry",
            "identity_id": "foil-geometry:background:17:1",
            "provenance": _provenance("row-21"),
        },
        {
            "foil_member_id": "covered-1",
            "diagnostic_owner_id": "diag:17:gt-0",
            "context_id": "ctx:17:pre-row-3",
            "bank_name": "covered",
            "source_id": "covered-1",
            "identity_kind": "reviewed_physical_owner",
            "identity_id": "gt:17:1",
            "provenance": _provenance("row-22"),
        },
        {
            "foil_member_id": "scan-1",
            "diagnostic_owner_id": "diag:17:gt-0",
            "context_id": "ctx:17:pre-row-3",
            "bank_name": "scan",
            "source_id": "scan-1",
            "identity_kind": "registered_geometry",
            "identity_id": "foil-geometry:scan:17:1",
            "provenance": _provenance("row-23"),
        },
    ]
    token_registry = _token_registry()
    return {
        "schema_version": RULES_SCHEMA_VERSION,
        "contract_mode": "test_fixture",
        "geometry_identity": {
            "schema": "canonical_round_bin_times_extent_over_1000.v1",
            "coordinate_denominator": 1000,
        },
        "coordinate_bins": {"min": 0, "max": 9},
        "target_anchor": {
            "margin_fraction": 0.0,
            "min_margin_bins": 0,
            "max_margin_bins": 0,
        },
        "bank_order": ["target", "background", "covered", "scan", "part"],
        "proposal_measures": {
            "target_measure": {
                "comparability_group": "matched",
                "normalization": "full_domain_normalized_weighted_sum",
                "bank_weights": {"target": 1.0, "part": 1.0},
            },
            "foil_measure": {
                "comparability_group": "matched",
                "normalization": "full_domain_normalized_weighted_sum",
                "bank_weights": {"background": 1.0, "covered": 1.0, "scan": 1.0},
            },
        },
        "bank_proposal_measure": {
            "target": "target_measure",
            "background": "foil_measure",
            "covered": "foil_measure",
            "scan": "foil_measure",
            "part": "target_measure",
        },
        "spatial_clustering": {
            "owner_link_iou_min": 0.1,
            "owner_link_center_distance_max": 4.0,
            "extent_submode_iou_min": 0.75,
        },
        "shape": {
            "near_peak_logprob_delta": 0.3,
            "wide_ridge_min_candidates": 2,
            "multi_submode_min": 2,
            "merged_extent_submodes": [],
            "scan_bank_names": ["scan"],
        },
        "declared_extent_submodes": ["whole", "background", "scan"],
        "registered_basin_roles": {
            "fixture-covered-role": {
                "kind": "foil",
                "foil_set_id": "fixture-foils-v1",
                "identity_kind": "reviewed_physical_owner",
                "allowed_bank_names": ["covered"],
            },
            "fixture-neutral-foil-role": {
                "kind": "foil",
                "foil_set_id": "fixture-foils-v1",
                "identity_kind": "registered_geometry",
                "allowed_bank_names": ["background", "scan"],
            },
            "fixture-target-role": {
                "kind": "target",
                "foil_set_id": "fixture-foils-v1",
                "identity_kind": "reviewed_physical_owner",
                "allowed_bank_names": ["target", "part"],
            },
        },
        "prominence": {"functional": "peak_height_difference"},
        "model_vocab_size": token_registry["model_vocab_size"],
        "schema_tokens": {
            **token_registry["schema_tokens"],
            "coordinate_token_id_start": token_registry["coordinate_bin_to_token_id"][
                "token_id_start"
            ],
            "coordinate_token_id_end_exclusive": token_registry[
                "coordinate_bin_to_token_id"
            ]["token_id_end_exclusive"],
        },
        "owner_canonical_descriptions": {"gt:17:0": _description("person", [100, 101])},
        "token_registry": token_registry,
        "structural_row_wrapper": _structural_row_wrapper(token_registry),
        "score_channels": _score_channels(),
        "candidate_materializer": {
            "schema_version": MATERIALIZER_RULES_SCHEMA_VERSION,
            "status": "sealed",
            "coordinate_space": _space(),
            "coco_namespace": {
                "name": OFFICIAL_COCO_NAMESPACE,
                "id_space": "official_gapped",
            },
            "target_bank_name": "target",
            "bank_roles": {
                "target": "fixture-target-role",
                "background": "fixture-neutral-foil-role",
                "covered": "fixture-covered-role",
                "scan": "fixture-neutral-foil-role",
                "part": "fixture-target-role",
            },
            "foil_set": {
                "foil_set_id": "fixture-foils-v1",
                "members": members,
                "members_sha256": sha256_json(
                    sorted(members, key=lambda item: item["foil_member_id"])
                ),
            },
            "p_x1_y1_pruning": {
                "mode": "upper_bound_pruning",
                "declaration": "later scorer prunes only after complete-y1 scoring",
                "threshold": -17.0,
            },
            "free_search_budget": {
                "x1_branch_budget": 64,
                "y1_branch_budget_per_x1": 32,
                "extent_branch_budget_per_anchor": 16,
                "spatial_diversification": {
                    "algorithm": "deterministic_farthest_point_xy_anchor_selection",
                    "tie_break": "ascending_x1_then_y1",
                    "minimum_center_distance_bins": 24,
                },
            },
            "vocabulary_attestation": _vocabulary_attestation(),
            "upstream_digests": upstream,
        },
    }


def _description(text: str, token_ids: list[int]) -> dict:
    forced_prefix = [900, *token_ids, 901, 902]
    return {
        "text": text,
        "text_sha256": sha256_json(text),
        "token_ids": token_ids,
        "token_ids_sha256": sha256_json(token_ids),
        "forced_row_prefix_through_box_start_token_ids": forced_prefix,
        "forced_row_prefix_through_box_start_sha256": sha256_json(forced_prefix),
    }


def _ledger(rules: dict) -> dict:
    token_registry = rules["token_registry"]
    context_provenance = {
        "context_kind": "P_pre",
        "context_status": "admitted_reviewed_candidate_only",
        "eligibility": "decision_bearing",
        "source_pred_row_id": "pred-row-3",
        "registry_id": "registry:ctx-17-pre",
        "foreign_keys": {"owner_ledger": "gt:17:0", "prediction_row": "pred-row-3"},
        "source_binding": {
            "artifact_sha256": _digest("context-ledger"),
            "source_row_id": "context-row-3",
        },
        "review_status": "reviewed_candidate",
    }
    source_lineage = {
        key: context_provenance[key]
        for key in (
            "source_pred_row_id",
            "review_status",
            "registry_id",
            "foreign_keys",
            "source_binding",
        )
    }
    return {
        "schema_version": OWNER_CONTEXT_LEDGER_SCHEMA_VERSION,
        "owner_status": "resolved",
        "diagnostic_owner_id": "diag:17:gt-0",
        "gt_owner_id": "gt:17:0",
        "context_id": "ctx:17:pre-row-3",
        "prompt_prefix_token_count": 2,
        "native_repetition_penalty_stratum": 1.0,
        "source_pred_row_id": "pred-row-3",
        "image_id": "17",
        "image_identity": _digest("image-17"),
        "image_size": {"width": 640, "height": 480},
        "ground_truth": {
            "box": [2, 3, 5, 6],
            "category": {"namespace": OFFICIAL_COCO_NAMESPACE, "category_id": 1},
        },
        "canonical_description": _description("person", [100, 101]),
        "context_tokens": {
            "token_ids": [1, 2, 3],
            "token_ids_sha256": sha256_json([1, 2, 3]),
            "prompt_prefix_token_count": 2,
            "prompt_token_ids_sha256": sha256_json([1, 2]),
            "self_prefix_generated_token_ids_sha256": sha256_json([3]),
            "split": {"prompt": [0, 2], "self_prefix": [2, 3]},
            "copy_semantics": "literal_task6_model_input_token_ids_no_decode_or_retokenize",
        },
        "context_provenance": context_provenance,
        "source_review_foreign_key_lineage": source_lineage,
        "coordinate_space": dict(rules["candidate_materializer"]["coordinate_space"]),
        "vocabulary_attestation": _vocabulary_attestation(),
        "token_registry": token_registry,
        "runtime_vocabulary_receipt": {
            "model_vocab_size": token_registry["model_vocab_size"],
            "token_registry_sha256": token_registry["registry_sha256"],
            "identity_receipt_digest": token_registry["identity_receipt_digest"],
            **_vocabulary_attestation(),
        },
        "upstream_digests": rules["candidate_materializer"]["upstream_digests"],
    }


def _seeds(
    *, rules_digest: str, ledger_digest: str, rules: dict, score_value: float
) -> dict:
    foil_set_digest = rules["candidate_materializer"]["foil_set"]["members_sha256"]
    return {
        "schema_version": BANK_SEEDS_SCHEMA_VERSION,
        "landscape_decision_rules_sha256": rules_digest,
        "owner_context_ledger_sha256": ledger_digest,
        "foil_set_sha256": foil_set_digest,
        "seeds": [
            {
                "diagnostic_owner_id": "diag:17:gt-0",
                "context_id": "ctx:17:pre-row-3",
                "bank_name": "target",
                "role_id": "fixture-target-role",
                "source_id": "gt-template",
                "box": [0, 0, 3, 3],
                "extent_submode": "whole",
                "expansion": "anchor_translate",
                "identity_kind": "reviewed_physical_owner",
                "identity_id": "gt:17:0",
                "score_fields_from_untrusted_preview": {"complete_box": score_value},
            },
            {
                "diagnostic_owner_id": "diag:17:gt-0",
                "context_id": "ctx:17:pre-row-3",
                "bank_name": "background",
                "role_id": "fixture-neutral-foil-role",
                "source_id": "background-1",
                "box": [6, 6, 9, 9],
                "extent_submode": "background",
                "expansion": "exact",
                "identity_kind": "registered_geometry",
                "identity_id": "foil-geometry:background:17:1",
                "foil_member_id": "background-1",
                "foil_provenance": _provenance("row-21"),
                "score_fields_from_untrusted_preview": {"complete_box": score_value},
            },
            {
                "diagnostic_owner_id": "diag:17:gt-0",
                "context_id": "ctx:17:pre-row-3",
                "bank_name": "covered",
                "role_id": "fixture-covered-role",
                "source_id": "covered-1",
                "box": [5, 1, 8, 4],
                "extent_submode": "whole",
                "expansion": "exact",
                "identity_kind": "reviewed_physical_owner",
                "identity_id": "gt:17:1",
                "foil_member_id": "covered-1",
                "foil_provenance": _provenance("row-22"),
                "score_fields_from_untrusted_preview": {"complete_box": score_value},
            },
            {
                "diagnostic_owner_id": "diag:17:gt-0",
                "context_id": "ctx:17:pre-row-3",
                "bank_name": "scan",
                "role_id": "fixture-neutral-foil-role",
                "source_id": "scan-1",
                "box": [0, 6, 3, 9],
                "extent_submode": "scan",
                "expansion": "exact",
                "identity_kind": "registered_geometry",
                "identity_id": "foil-geometry:scan:17:1",
                "foil_member_id": "scan-1",
                "foil_provenance": _provenance("row-23"),
                "score_fields_from_untrusted_preview": {"complete_box": score_value},
            },
        ],
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _materialize(
    tmp_path: Path,
    *,
    score_value: float = 0.0,
    rules_document: dict | None = None,
    ledger_mutator: Callable[[dict], None] | None = None,
    seeds_mutator: Callable[[dict], None] | None = None,
) -> tuple[Path, Path, dict]:
    rules = _rules() if rules_document is None else rules_document
    rules_path = tmp_path / "landscape-decision-rules.json"
    _write_json(rules_path, rules)
    ledger_path = tmp_path / "owner-context-ledger.jsonl"
    ledger = _ledger(rules)
    if ledger_mutator is not None:
        ledger_mutator(ledger)
    _write_json(ledger_path, ledger)
    seeds_path = tmp_path / "bank-seeds.json"
    seeds = _seeds(
        rules_digest=sha256_file(rules_path),
        ledger_digest=sha256_file(ledger_path),
        rules=rules,
        score_value=score_value,
    )
    if seeds_mutator is not None:
        seeds_mutator(seeds)
    _write_json(seeds_path, seeds)
    output_path = tmp_path / "landscape-candidates.jsonl"
    receipt_path = tmp_path / "landscape-candidates-receipt.json"
    receipt = build_sorted_owner_basin_candidates(
        owner_context_ledger=ledger_path,
        landscape_decision_rules=rules_path,
        bank_seeds=seeds_path,
        output_jsonl=output_path,
        receipt=receipt_path,
        expected_owner_context_ledger_sha256=sha256_file(ledger_path),
        expected_landscape_decision_rules_sha256=sha256_file(rules_path),
        expected_bank_seeds_sha256=sha256_file(seeds_path),
    )
    return output_path, receipt_path, receipt


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rules_with_unselected_context_foil() -> dict:
    rules = _rules()
    members = rules["candidate_materializer"]["foil_set"]["members"]
    members.append(
        {
            "foil_member_id": "background-unselected",
            "diagnostic_owner_id": "diag:99:gt-0",
            "context_id": "ctx:99:pre-row-1",
            "bank_name": "background",
            "source_id": "background-unselected",
            "identity_kind": "registered_geometry",
            "identity_id": "foil-geometry:background:99:1",
            "provenance": _provenance("row-99"),
        }
    )
    rules["candidate_materializer"]["foil_set"]["members_sha256"] = sha256_json(
        sorted(members, key=lambda item: item["foil_member_id"])
    )
    return rules


def test_global_foil_registry_does_not_require_seed_for_unselected_context(
    tmp_path: Path,
) -> None:
    rules = _rules_with_unselected_context_foil()

    _, _, receipt = _materialize(tmp_path, rules_document=rules)

    assert receipt["inputs"]["foil_set_sha256"] == rules["candidate_materializer"][
        "foil_set"
    ]["members_sha256"]


def test_selected_context_still_requires_every_frozen_foil_seed(
    tmp_path: Path,
) -> None:
    def omit_selected_foil(seeds: dict) -> None:
        seeds["seeds"] = [
            seed
            for seed in seeds["seeds"]
            if seed.get("foil_member_id") != "scan-1"
        ]

    with pytest.raises(
        ValueError, match="bank seeds omit frozen foil provenance: scan-1"
    ):
        _materialize(tmp_path, seeds_mutator=omit_selected_foil)


def test_seed_for_unselected_context_is_rejected_as_unknown_identity(
    tmp_path: Path,
) -> None:
    rules = _rules_with_unselected_context_foil()

    def add_unselected_context_seed(seeds: dict) -> None:
        seeds["seeds"].append(
            {
                "diagnostic_owner_id": "diag:99:gt-0",
                "context_id": "ctx:99:pre-row-1",
                "bank_name": "background",
                "role_id": "fixture-neutral-foil-role",
                "source_id": "background-unselected",
                "box": [6, 6, 9, 9],
                "extent_submode": "background",
                "expansion": "exact",
                "identity_kind": "registered_geometry",
                "identity_id": "foil-geometry:background:99:1",
                "foil_member_id": "background-unselected",
                "foil_provenance": _provenance("row-99"),
            }
        )

    with pytest.raises(ValueError, match="unknown owner/context identity"):
        _materialize(
            tmp_path,
            rules_document=rules,
            seeds_mutator=add_unselected_context_seed,
        )


def test_content_addressed_candidate_and_receipt_digests_are_deterministic(
    tmp_path: Path,
) -> None:
    output_one, receipt_one, result_one = _materialize(tmp_path / "one")
    output_two, receipt_two, result_two = _materialize(tmp_path / "two")

    assert output_one.read_bytes() == output_two.read_bytes()
    assert receipt_one.read_bytes() == receipt_two.read_bytes()
    assert result_one == result_two
    candidates = [
        row
        for row in _records(output_one)
        if row["record_type"] == "complete_box_candidate"
    ]
    assert candidates
    assert all(
        row["candidate_id"].startswith("landscape-candidate:sha256:")
        for row in candidates
    )
    assert all(
        row["unique_box_id"].startswith("landscape-unique-box:sha256:")
        for row in candidates
    )
    assert all(
        row["geometry_identity"]["schema"]
        == "canonical_round_bin_times_extent_over_1000.v1"
        for row in candidates
    )
    assert all(
        row["core_rule_digest"] == result_one["inputs"]["core_rule_digest"]
        for row in candidates
    )
    assert result_one["output_jsonl"]["sha256"] == sha256_file(output_one)


def test_v2_rows_freeze_literal_executable_scorer_requests(tmp_path: Path) -> None:
    output_path, _, receipt = _materialize(tmp_path)
    records = _records(output_path)
    free_roots = [
        row
        for row in records
        if row["record_type"] == "free_coordinate_tree_root_request"
    ]
    plans = [
        row for row in records if row["record_type"] == "conditional_y1_score_plan"
    ]
    candidates = [
        row for row in records if row["record_type"] == "complete_box_candidate"
    ]
    root_prefix = [1, 2, 3, 900, 100, 101, 901, 902]

    assert len(free_roots) == 1
    assert free_roots[0]["prefix_token_ids"] == root_prefix
    assert free_roots[0]["prompt_prefix_token_count"] == 2
    assert free_roots[0]["scan_slot"] == "x1"
    assert (
        free_roots[0]["execution_status"]
        == "declared_not_executed_by_candidate_materializer"
    )
    assert free_roots[0]["dynamic_traversal_seam"]["budget"] == {
        "x1_branch_budget": 64,
        "y1_branch_budget_per_x1": 32,
        "extent_branch_budget_per_anchor": 16,
        "spatial_diversification": {
            "algorithm": "deterministic_farthest_point_xy_anchor_selection",
            "tie_break": "ascending_x1_then_y1",
            "minimum_center_distance_bins": 24,
        },
    }

    assert plans
    for plan in plans:
        x1 = plan["x1"]
        assert plan["request_kind"] == "dense_scan"
        assert plan["prefix_token_ids"] == root_prefix
        assert plan["root_prefix_token_ids"] == root_prefix
        assert plan["fixed_coord_bin_values"] == [x1]
        assert plan["fixed_coord_token_ids"] == [1000 + x1]
        assert plan["scan_slot"] == "y1"
        assert plan["fixed_coord_tokens_materialized_in_prefix"] is False
        assert plan["expected_prefix_after_fixed_token_ids"] == [
            *root_prefix,
            1000 + x1,
        ]
        assert plan["expected_prefix_after_fixed_is_attestation_only"] is True
        assert plan["core_request_attestation"]["attestation_sha256"]

    assert candidates
    for candidate in candidates:
        bins = [candidate["box"][slot] for slot in ("x1", "y1", "x2", "y2")]
        tokens = [1000 + coordinate_bin for coordinate_bin in bins]
        assert candidate["request_kind"] == "complete_box"
        assert candidate["prefix_token_ids"] == root_prefix
        assert candidate["coordinate_bin_values"] == bins
        assert candidate["coordinate_token_ids"] == tokens
        assert candidate["coord_token_ids"] == tokens
        assert candidate["prompt_prefix_token_count"] == 2
        assert candidate["source_pred_row_id"] == "pred-row-3"
        assert candidate["source_review_foreign_key_lineage"]["foreign_keys"]
        assert candidate["proposal_digest"]

    channels = {sha256_json(row["analysis_channels"]) for row in plans + candidates}
    assert len(channels) == 1
    analysis_channels = plans[0]["analysis_channels"]
    assert analysis_channels["shared_raw_forward"] is True
    assert analysis_channels["duplicate_raw_forward_per_policy_view_allowed"] is False
    assert [
        view["repetition_penalty"] for view in analysis_channels["policy_views"]
    ] == [
        1.0,
        1.10,
    ]
    assert receipt["counts"]["free_coordinate_tree_root_requests"] == 1
    assert receipt["free_search"]["executed"] is False
    assert (
        receipt["executable_request_contract"][
            "duplicate_raw_forward_per_policy_view_allowed"
        ]
        is False
    )


def test_v2_fails_if_exact_prompt_split_is_missing(tmp_path: Path) -> None:
    def remove_prompt_count(ledger: dict) -> None:
        del ledger["prompt_prefix_token_count"]

    with pytest.raises(ValueError, match="prompt_prefix_token_count"):
        _materialize(tmp_path, ledger_mutator=remove_prompt_count)
    assert not (tmp_path / "landscape-candidates.jsonl").exists()


def test_v2_fails_if_explicit_coordinate_registry_digest_is_stale(
    tmp_path: Path,
) -> None:
    rules = _rules()
    token_ids = rules["token_registry"]["coordinate_bin_to_token_id"][
        "coordinate_bin_token_ids"
    ]
    token_ids[7], token_ids[8] = token_ids[8], token_ids[7]

    with pytest.raises(
        ValueError, match="coordinate-bin token registry digest is stale"
    ):
        _materialize(tmp_path, rules_document=rules)
    assert not (tmp_path / "landscape-candidates.jsonl").exists()


def test_rule_mutation_invalidates_the_sealed_digest_before_writing(
    tmp_path: Path,
) -> None:
    rules = _rules()
    rules_path = tmp_path / "rules.json"
    _write_json(rules_path, rules)
    ledger_path = tmp_path / "ledger.jsonl"
    _write_json(ledger_path, _ledger(rules))
    original_rules_digest = sha256_file(rules_path)
    seeds_path = tmp_path / "seeds.json"
    _write_json(
        seeds_path,
        _seeds(
            rules_digest=original_rules_digest,
            ledger_digest=sha256_file(ledger_path),
            rules=rules,
            score_value=0.0,
        ),
    )
    rules["target_anchor"]["max_margin_bins"] = 1
    _write_json(rules_path, rules)

    with pytest.raises(ValueError, match="rules digest"):
        build_sorted_owner_basin_candidates(
            owner_context_ledger=ledger_path,
            landscape_decision_rules=rules_path,
            bank_seeds=seeds_path,
            output_jsonl=tmp_path / "landscape-candidates.jsonl",
            receipt=tmp_path / "receipt.json",
            expected_owner_context_ledger_sha256=sha256_file(ledger_path),
            expected_landscape_decision_rules_sha256=original_rules_digest,
            expected_bank_seeds_sha256=sha256_file(seeds_path),
        )
    assert not (tmp_path / "landscape-candidates.jsonl").exists()


def test_complete_y1_plan_retains_terminal_invalid_box_audit_and_completeness_digest(
    tmp_path: Path,
) -> None:
    output_path, _, _ = _materialize(tmp_path)
    records = _records(output_path)
    plans = [
        row for row in records if row["record_type"] == "conditional_y1_score_plan"
    ]
    candidates = [
        row for row in records if row["record_type"] == "complete_box_candidate"
    ]

    assert len(plans) == 3  # Full x1 GT interior: 2, 3, 4.
    assert all(len(row["complete_conditional_y1"]) == 10 for row in plans)
    assert all(
        row["complete_conditional_y1"][-1]
        == {
            "y1": 9,
            "is_target_anchor": False,
            "can_form_valid_box": False,
            "invalid_box_reason": "no_later_representable_y2",
        }
        for row in plans
    )
    completeness_digests = {
        row["completeness_plan_digest"] for row in plans + candidates
    }
    assert len(completeness_digests) == 1
    assert next(iter(completeness_digests)).startswith(
        "restricted-completeness-plan:sha256:"
    )
    assert all(
        row["scored_completeness_receipt_sha256"] is None for row in plans + candidates
    )


def test_neutral_foils_and_distinct_covered_owner_keep_registered_prominence_identity(
    tmp_path: Path,
) -> None:
    output_path, _, receipt = _materialize(tmp_path)
    candidates = {
        row["bank_name"]: row
        for row in _records(output_path)
        if row["record_type"] == "complete_box_candidate"
        and row["bank_name"] != "target"
    }

    for bank_name in ("background", "scan"):
        candidate = candidates[bank_name]
        assert candidate["identity_kind"] == "registered_geometry"
        assert candidate["identity_id"].startswith(f"foil-geometry:{bank_name}:")
        assert candidate["physical_owner_hint"] is None
        assert candidate["physical_owner_registration_allowed"] is False
        assert candidate["prominence_eligibility"] == "registered_owner_neutral_foil"
        assert candidate["role_kind"] == "foil"
        assert candidate["foil_set_id"] == "fixture-foils-v1"
        registration = BasinRegistration(
            basin_id=f"fixture-{bank_name}-basin",
            role_id=candidate["role_id"],
            identity_kind=candidate["identity_kind"],
            reviewed_physical_owner_id=candidate["reviewed_physical_owner_id"],
            registered_geometry_id=candidate["registered_geometry_id"],
            context_id=candidate["context_id"],
            foil_set_id=candidate["foil_set_id"],
            rule_digest=candidate["core_rule_digest"],
            conditional_y1_completeness_digest=candidate["completeness_plan_digest"],
        )
        assert registration.registered_geometry_id == candidate["identity_id"]
        with pytest.raises(ValueError, match="cannot carry physical-owner ID"):
            BasinRegistration(
                basin_id=f"invalid-{bank_name}-basin",
                role_id=candidate["role_id"],
                identity_kind="registered_geometry",
                reviewed_physical_owner_id="gt:17:0",
                registered_geometry_id=candidate["identity_id"],
                context_id=candidate["context_id"],
                foil_set_id=candidate["foil_set_id"],
                rule_digest=candidate["core_rule_digest"],
                conditional_y1_completeness_digest=candidate[
                    "completeness_plan_digest"
                ],
            )

    covered = candidates["covered"]
    assert covered["identity_kind"] == "reviewed_physical_owner"
    assert covered["identity_id"] == "gt:17:1"
    assert covered["identity_id"] != covered["gt_owner_id"]
    assert covered["physical_owner_hint"] == "gt:17:1"
    assert covered["physical_owner_registration_allowed"] is True
    assert covered["prominence_eligibility"] == "registered_physical_owner_foil"
    covered_registration = BasinRegistration(
        basin_id="fixture-covered-basin",
        role_id=covered["role_id"],
        identity_kind=covered["identity_kind"],
        reviewed_physical_owner_id=covered["reviewed_physical_owner_id"],
        registered_geometry_id=covered["registered_geometry_id"],
        context_id=covered["context_id"],
        foil_set_id=covered["foil_set_id"],
        rule_digest=covered["core_rule_digest"],
        conditional_y1_completeness_digest=covered["completeness_plan_digest"],
    )
    assert covered_registration.reviewed_physical_owner_id == "gt:17:1"
    assert receipt["production_admission"]["eligible"] is False
    assert receipt["production_admission"]["status"] == "non_production_test_fixture"


def test_untrusted_score_fields_cannot_change_candidate_membership(
    tmp_path: Path,
) -> None:
    low_output, _, _ = _materialize(tmp_path / "low", score_value=-1000000.0)
    high_output, _, _ = _materialize(tmp_path / "high", score_value=1000000.0)

    assert low_output.read_bytes() == high_output.read_bytes()
    result = json.loads(
        (tmp_path / "low" / "landscape-candidates-receipt.json").read_text()
    )
    assert result["candidate_membership"] == {
        "score_dependent_selection_executed": False,
        "selection": "frozen_pre_score_only",
        "unique_box_mass_semantics": "one_score_per_owner_context_description_box",
    }
    assert b"score_fields_from_untrusted_preview" not in low_output.read_bytes()


def test_write_once_output_refuses_overwrite(tmp_path: Path) -> None:
    output_path, receipt_path, _ = _materialize(tmp_path)
    rules_path = tmp_path / "landscape-decision-rules.json"
    ledger_path = tmp_path / "owner-context-ledger.jsonl"
    seeds_path = tmp_path / "bank-seeds.json"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_sorted_owner_basin_candidates(
            owner_context_ledger=ledger_path,
            landscape_decision_rules=rules_path,
            bank_seeds=seeds_path,
            output_jsonl=output_path,
            receipt=receipt_path,
            expected_owner_context_ledger_sha256=sha256_file(ledger_path),
            expected_landscape_decision_rules_sha256=sha256_file(rules_path),
            expected_bank_seeds_sha256=sha256_file(seeds_path),
        )


def test_production_contract_rejects_fixture_coordinate_bins_and_conversion(
    tmp_path: Path,
) -> None:
    rules = _rules()
    rules["contract_mode"] = "production"
    rules["coordinate_bins"]["max"] = 999
    rules["candidate_materializer"]["coordinate_space"] = {
        "name": COORDINATE_SPACE_NAME,
        "min": 0,
        "max": 999,
        "contract_kind": "production",
        "bin_to_extent_conversion": "floor",
    }
    rules_path = tmp_path / "rules.json"
    _write_json(rules_path, rules)
    ledger = _ledger(rules)
    ledger_path = tmp_path / "ledger.jsonl"
    _write_json(ledger_path, ledger)
    seeds_path = tmp_path / "seeds.json"
    _write_json(
        seeds_path,
        _seeds(
            rules_digest=sha256_file(rules_path),
            ledger_digest=sha256_file(ledger_path),
            rules=rules,
            score_value=0.0,
        ),
    )

    with pytest.raises(ValueError, match="production coordinate contract"):
        build_sorted_owner_basin_candidates(
            owner_context_ledger=ledger_path,
            landscape_decision_rules=rules_path,
            bank_seeds=seeds_path,
            output_jsonl=tmp_path / "candidates.jsonl",
            receipt=tmp_path / "receipt.json",
            expected_owner_context_ledger_sha256=sha256_file(ledger_path),
            expected_landscape_decision_rules_sha256=sha256_file(rules_path),
            expected_bank_seeds_sha256=sha256_file(seeds_path),
        )


def test_production_contract_emits_admissible_canonical_geometry_receipts(
    tmp_path: Path,
) -> None:
    rules = _rules()
    rules["contract_mode"] = "production"
    rules["coordinate_bins"] = {"min": 0, "max": 999}
    rules["candidate_materializer"]["coordinate_space"] = {
        "name": COORDINATE_SPACE_NAME,
        "min": 0,
        "max": 999,
        "contract_kind": "production",
        "bin_to_extent_conversion": "round(value*extent/1000)",
    }

    output_path, _, receipt = _materialize(tmp_path, rules_document=rules)
    candidates = [
        row
        for row in _records(output_path)
        if row["record_type"] == "complete_box_candidate"
    ]
    assert receipt["production_admission"]["eligible"] is True
    assert receipt["production_admission"]["status"] == "production"
    assert all(row["contract_mode"] == "production" for row in candidates)
    assert all(row["coordinate_space"]["min"] == 0 for row in candidates)
    assert all(row["coordinate_space"]["max"] == 999 for row in candidates)
    assert all(row["geometry_identity"]["identity_digest"] for row in candidates)


def test_direct_cli_help_bootstraps_the_repository_import_path(tmp_path: Path) -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/build_sorted_owner_basin_candidates.py"
    )
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--owner-context-ledger" in result.stdout

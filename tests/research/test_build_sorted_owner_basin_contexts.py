from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.research.build_sorted_owner_basin_contexts import (
    AMBIGUITY_SCHEMA_VERSION,
    CONTEXT_SCHEMA_VERSION,
    CONTROL_REGISTRY_SCHEMA_VERSION,
    ContextContractError,
    MATRIX_SCHEMA_VERSION,
    NATIVE_REPLAY_SCHEMA_VERSION,
    OWNER_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    SENTINEL_CONFIRMATION_SCHEMA_VERSION,
    SENTINEL_REGISTRY_SCHEMA_VERSION,
    SENTINEL_SELECTION_SCHEMA_VERSION,
    STRUCTURAL_REGISTRY_SCHEMA_VERSION,
    TASK0_EXECUTION_SCHEMA_VERSION,
    TASK0_MANIFEST_SCHEMA_VERSION,
    build_sorted_owner_basin_contexts,
    sha256_file,
    sha256_json,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _content_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _row_tokens(description_token: int, coordinate_offset: int) -> list[int]:
    return [
        10,
        description_token,
        11,
        12,
        100 + coordinate_offset,
        101 + coordinate_offset,
        102 + coordinate_offset,
        103 + coordinate_offset,
        13,
    ]


def _fixture(tmp_path: Path, *, ambiguous_row: bool = False) -> dict[str, Any]:
    root = tmp_path / "task0-v2"
    root.mkdir()
    prompt_ids = [1, 2, 3]
    generated_ids = [
        *_row_tokens(50, 0),
        *_row_tokens(52, 4),
        *_row_tokens(53, 8),
        *_row_tokens(53, 8),
    ]
    greedy = tmp_path / "greedy.json"
    _write_json(
        greedy,
        {
            "schema_version": "current_seeded_sampled_rollouts.v1",
            "experiment_mode": "natural",
            "rollout_count": 1,
            "config": {
                "decode_mode": "greedy",
                "seeds": [0],
                "repetition_penalty": 1.0,
            },
            "rollouts": [
                {
                    "image_id": "scene",
                    "decode_mode": "greedy",
                    "seed": 0,
                    "stop_reason": "im_end",
                    "prompt_token_ids": prompt_ids,
                    "prompt_token_ids_sha256": sha256_json(prompt_ids),
                    "generated_token_ids": generated_ids,
                    "generated_token_ids_sha256": sha256_json(generated_ids),
                    "predictions": {
                        "predictions": [
                            {"generated_order": index} for index in range(4)
                        ],
                        "dropped_predictions": [],
                    },
                }
            ],
        },
    )
    greedy_digest = sha256_file(greedy)
    trajectory_id = "trajectory:sorted:rp1.00:greedy:0:scene"
    execution_receipt: dict[str, Any] = {
        "schema_version": TASK0_EXECUTION_SCHEMA_VERSION,
        "execution_status": "completed",
        "execution_surface": "deterministic_cpu_census_no_model_inference",
        "inputs": {"greedy_artifact": {"path": str(greedy), "sha256": greedy_digest}},
        "prompt_media_identities": [
            {
                "trajectory_id": trajectory_id,
                "prompt_token_ids_sha256": sha256_json(prompt_ids),
                "generated_token_ids_sha256": sha256_json(generated_ids),
            }
        ],
    }
    execution_receipt["execution_receipt_content_sha256"] = _content_digest(
        execution_receipt
    )
    content_digest = execution_receipt["execution_receipt_content_sha256"]
    execution_path = root / "execution-receipt.json"
    _write_json(execution_path, execution_receipt)

    ambiguity_id = "ambiguity:fixture:row-1"
    ambiguity_rows = (
        [
            {
                "schema_version": AMBIGUITY_SCHEMA_VERSION,
                "ambiguity_receipt_id": ambiguity_id,
                "trajectory_id": trajectory_id,
                "image_id": "scene",
                "decode_mode": "greedy",
                "seed": 0,
                "pred_row_ids": ["pred:sorted:greedy:0:scene:1"],
                "gt_owner_ids": ["gt:scene:2"],
                "execution_receipt_content_sha256": content_digest,
                "source_digests": {
                    "rollout_artifact": greedy_digest,
                    "execution_receipt_content": content_digest,
                },
            }
        ]
        if ambiguous_row
        else []
    )
    ambiguity_path = root / "ambiguity-receipts.jsonl"
    _write_jsonl(ambiguity_path, ambiguity_rows)

    owners = []
    # Same y1 but different x1 makes the reference-position regression observable.
    annotation_indices = (40, 30, 20, 10)
    for index, name in enumerate(("a", "b", "c", "d")):
        owner_id = f"gt:scene:{index}"
        owner_is_ambiguous = ambiguous_row and index == 2
        owners.append(
            {
                "schema_version": OWNER_SCHEMA_VERSION,
                "gt_owner_id": owner_id,
                "diagnostic_owner_id": f"diagnostic:{owner_id}",
                "image_id": "scene",
                "original_annotation_index": annotation_indices[index],
                "description": name,
                "normalized_description": name,
                "bbox_xyxy": [index * 10, 0, index * 10 + 5, 5],
                "ambiguity_receipt_ids": [ambiguity_id] if owner_is_ambiguous else [],
                "decision_eligibility": {
                    "greedy_natural": {
                        "eligible": not owner_is_ambiguous,
                        "status": "eligible"
                        if not owner_is_ambiguous
                        else "globally_ambiguous_neutral",
                    },
                    "k16_any_hit": {"eligible": True, "status": "eligible"},
                    "greedy_k16_paired": {
                        "eligible": not owner_is_ambiguous,
                        "status": "eligible"
                        if not owner_is_ambiguous
                        else "excluded_by_either_policy_global_ambiguity",
                    },
                },
                "execution_receipt_content_sha256": content_digest,
                "source_digests": {
                    "panel": "1" * 64,
                    "execution_receipt_content": content_digest,
                },
            }
        )
    owner_path = root / "owner-ledger.jsonl"
    _write_jsonl(owner_path, owners)

    assigned_owner_ids = ["gt:scene:0", "gt:scene:2", "gt:scene:3", None]
    prediction_rows = []
    for index, owner_id in enumerate(assigned_owner_ids):
        ambiguous = ambiguous_row and index == 1
        reviewed_duplicate = index == 3
        prediction_rows.append(
            {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "pred_row_id": f"pred:sorted:greedy:0:scene:{index}",
                "trajectory_id": trajectory_id,
                "image_id": "scene",
                "decode_mode": "greedy",
                "seed": 0,
                "original_row_index": index,
                "row_kind": "complete_prediction",
                "description": ("a", "c", "d", "d")[index],
                "normalized_description": ("a", "c", "d", "d")[index],
                "bbox_xyxy": (
                    [0, 0, 5, 5],
                    [20, 0, 25, 5],
                    [30, 0, 35, 5],
                    [30, 0, 35, 5],
                )[index],
                "strict_match_status": "ambiguous_neutral"
                if ambiguous
                else "unmatched"
                if reviewed_duplicate
                else "matched",
                "strict_match_gt_owner_id": owner_id,
                "physical_relation": "covered_owner_duplicate"
                if reviewed_duplicate
                else None,
                "diagnostic_owner_id": "diagnostic:gt:scene:3"
                if reviewed_duplicate
                else None,
                "assignment_status": "reviewed_resolved"
                if reviewed_duplicate
                else None,
                "ambiguity_receipt_ids": [ambiguity_id] if ambiguous else [],
                "execution_receipt_content_sha256": content_digest,
                "source_digests": {
                    "rollout_artifact": greedy_digest,
                    "execution_receipt_content": content_digest,
                },
            }
        )
    prediction_path = root / "prediction-row-ledger.jsonl"
    _write_jsonl(prediction_path, prediction_rows)

    matrix_rows = []
    for index in range(4):
        pred_ids = (
            ["pred:sorted:greedy:0:scene:0"]
            if index == 0
            else ["pred:sorted:greedy:0:scene:1"]
            if index == 2
            else ["pred:sorted:greedy:0:scene:2"]
            if index == 3
            else []
        )
        owner_is_ambiguous = ambiguous_row and index == 2
        matrix_rows.append(
            {
                "schema_version": MATRIX_SCHEMA_VERSION,
                "gt_owner_id": f"gt:scene:{index}",
                "trajectory_id": trajectory_id,
                "image_id": "scene",
                "decode_mode": "greedy",
                "seed": 0,
                "matched_pred_row_ids": pred_ids,
                "global_ambiguity_presence": owner_is_ambiguous,
                "ambiguity_receipt_ids": [ambiguity_id] if owner_is_ambiguous else [],
                "execution_receipt_content_sha256": content_digest,
                "source_digests": {
                    "panel": "1" * 64,
                    "rollout_artifact": greedy_digest,
                    "execution_receipt_content": content_digest,
                },
            }
        )
    matrix_path = root / "owner-trajectory-matrix.jsonl"
    _write_jsonl(matrix_path, matrix_rows)

    native_path = root / "native-replay.jsonl"
    _write_jsonl(
        native_path,
        [
            {
                "schema_version": NATIVE_REPLAY_SCHEMA_VERSION,
                "trajectory_id": trajectory_id,
                "execution_receipt_content_sha256": content_digest,
                "source_digests": {
                    "rollout_artifact": greedy_digest,
                    "execution_receipt_content": content_digest,
                },
            }
        ],
    )
    matcher_path = root / "matcher-contract.json"
    _write_json(
        matcher_path,
        {
            "schema_version": "sorted-owner-basin-matcher.v2",
            "execution_receipt_content_sha256": content_digest,
        },
    )
    artifact_paths = {
        "execution_receipt": execution_path,
        "matcher_contract": matcher_path,
        "owner_ledger": owner_path,
        "prediction_ledger": prediction_path,
        "owner_trajectory_matrix": matrix_path,
        "native_replay": native_path,
        "ambiguity_receipts": ambiguity_path,
    }
    manifest_path = root / "artifact-manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": TASK0_MANIFEST_SCHEMA_VERSION,
            "census_schema_version": "sorted-owner-basin-census.v2",
            "execution_receipt_content_sha256": content_digest,
            "artifacts": {
                name: {"path": path.name, "sha256": sha256_file(path)}
                for name, path in artifact_paths.items()
            },
            "sources": {
                "panel": {"sha256": "1" * 64},
                "matched_rp_1_0_greedy": {"sha256": greedy_digest},
                "matched_rp_1_0_sampled_shards": [
                    {"sha256": "2" * 64},
                    {"sha256": "3" * 64},
                ],
            },
            "forced_continuation_absence": {
                "status": "proved_by_structural_source_scan",
                "forbidden_marker_paths": [],
            },
        },
    )
    manifest_digest = sha256_file(manifest_path)

    experiment = tmp_path / "experiment"
    experiment.mkdir()
    selection_path = experiment / "sentinel-selection-receipt.json"
    _write_json(
        selection_path,
        {
            "schema_version": SENTINEL_SELECTION_SCHEMA_VERSION,
            "selection_status": "lead_reviewed_and_frozen_before_landscape_scoring",
            "source_artifacts": {
                "human_refined_panel_sha256": "1" * 64,
                "matched_rp_1_0_greedy_sha256": greedy_digest,
                "matched_rp_1_0_sampled_shard_0_sha256": "2" * 64,
                "matched_rp_1_0_sampled_shard_1_sha256": "3" * 64,
            },
        },
    )
    selection_digest = sha256_file(selection_path)
    confirmation_path = experiment / "sentinel-selection-confirmation-receipt.json"
    _write_json(
        confirmation_path,
        {
            "schema_version": SENTINEL_CONFIRMATION_SCHEMA_VERSION,
            "confirmation_status": "lead_reconfirmed_against_final_task0_v2_before_landscape_scoring",
            "original_selection_receipt": {
                "path": selection_path.name,
                "sha256": selection_digest,
            },
            "final_task0_v2": {
                "root": str(root.resolve()),
                "artifact_manifest_sha256": manifest_digest,
                "execution_receipt_content_sha256": content_digest,
                "execution_receipt_file_sha256": sha256_file(execution_path),
                "owner_ledger_sha256": sha256_file(owner_path),
                "owner_trajectory_matrix_sha256": sha256_file(matrix_path),
            },
            "confirmed_sentinels": [
                {
                    "gt_owner_id": "gt:scene:1",
                    "decision_eligible": True,
                    "trajectory_count": 17,
                    "strict_match_count": 0,
                    "max_semantic_compatible_iou": 0.0,
                }
            ],
        },
    )
    confirmation_digest = sha256_file(confirmation_path)
    sentinel_path = experiment / "sentinel-registry.json"
    _write_json(
        sentinel_path,
        {
            "schema_version": SENTINEL_REGISTRY_SCHEMA_VERSION,
            "selection_status": "lead_frozen_before_scoring_and_reconfirmed_against_final_task0_v2",
            "selection_receipt": {
                "path": selection_path.name,
                "sha256": selection_digest,
            },
            "selection_confirmation_receipt": {
                "path": confirmation_path.name,
                "sha256": confirmation_digest,
            },
            "source_digests": {
                "human_refined_panel_sha256": "1" * 64,
                "matched_rp_1_0_greedy_sha256": greedy_digest,
                "matched_rp_1_0_sampled_shard_0_sha256": "2" * 64,
                "matched_rp_1_0_sampled_shard_1_sha256": "3" * 64,
                "task0_v2_root": str(root.resolve()),
                "task0_census_artifact_manifest_sha256": manifest_digest,
                "task0_execution_receipt_content_sha256": content_digest,
                "task0_execution_receipt_file_sha256": sha256_file(execution_path),
                "task0_owner_ledger_sha256": sha256_file(owner_path),
                "task0_owner_trajectory_matrix_sha256": sha256_file(matrix_path),
            },
            "sentinels": [
                {
                    "sentinel_id": "sentinel-b",
                    "gt_owner_id": "gt:scene:1",
                    "prior_non_recovery_status": "verified_primary_natural_zero_spatial_support",
                }
            ],
        },
    )
    control_path = experiment / "control-registry.json"
    _write_json(
        control_path,
        {
            "schema_version": CONTROL_REGISTRY_SCHEMA_VERSION,
            "status": "lead_frozen_before_scoring_and_resealed_to_final_task0_v2",
            "source_digests": {
                "task0_v2_root": str(root.resolve()),
                "task0_census_artifact_manifest_sha256": manifest_digest,
                "task0_execution_receipt_content_sha256": content_digest,
                "task0_execution_receipt_file_sha256": sha256_file(execution_path),
                "owner_ledger_sha256": sha256_file(owner_path),
                "owner_trajectory_matrix_sha256": sha256_file(matrix_path),
                "sentinel_selection_confirmation_receipt_sha256": confirmation_digest,
            },
            "controls": [
                {
                    "control_id": "control:strict-visible:a",
                    "role": "strict_visible_true_positive",
                    "gt_owner_id": "gt:scene:0",
                    "image_id": "scene",
                    "strata": ["fixture_calibration"],
                },
                {
                    "control_id": "control:b1:b",
                    "role": "b1_loose_only",
                    "gt_owner_id": "gt:scene:1",
                    "image_id": "scene",
                    "strata": ["fixture_calibration"],
                },
                {
                    "control_id": "control:b2:c-to-b",
                    "role": "b2_distinct_same_description_pair",
                    "covering_gt_owner_id": "gt:scene:2",
                    "target_gt_owner_id": "gt:scene:1",
                    "image_id": "scene",
                    "covering_pred_row_id": "pred:sorted:greedy:0:scene:1",
                    "physical_identity_review": "distinct_people_confirmed_by_lead_visual_inspection",
                },
            ],
        },
    )
    structural_path = tmp_path / "structural-token-registry.json"
    _write_json(
        structural_path,
        {
            "schema_version": STRUCTURAL_REGISTRY_SCHEMA_VERSION,
            "status": "sealed",
            "source_digests": {"greedy": greedy_digest},
            "tokens": {
                "object_ref_start": 10,
                "object_ref_end": 11,
                "box_start": 12,
                "box_end": 13,
                "coordinate_token_id_min": 100,
                "coordinate_token_id_max": 199,
            },
            "terminal": {
                "storage": "excluded",
                "token_ids": [],
                "stop_reason": "im_end",
            },
            "chronology": {
                "short_gap_max_intervening_rows": 2,
                "cyclic_min_occurrences": 3,
            },
        },
    )
    paths = {
        "greedy": greedy,
        "owner_ledger": owner_path,
        "prediction_row_ledger": prediction_path,
        "owner_trajectory_matrix": matrix_path,
        "structural_token_registry": structural_path,
        "sentinel_registry": sentinel_path,
        "reviewed_b2_registry": control_path,
    }
    return {
        "paths": paths,
        "digests": {name: sha256_file(path) for name, path in paths.items()},
    }


def _build(fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    paths = fixture["paths"]
    return dict(
        build_sorted_owner_basin_contexts(
            greedy_artifact=paths["greedy"],
            owner_ledger=paths["owner_ledger"],
            prediction_row_ledger=paths["prediction_row_ledger"],
            owner_trajectory_matrix=paths["owner_trajectory_matrix"],
            structural_token_registry=paths["structural_token_registry"],
            sentinel_registry=paths["sentinel_registry"],
            reviewed_b2_registry=paths["reviewed_b2_registry"],
            output_dir=output,
            expected_digests=fixture["digests"],
            execution_argv=[sys.executable, "fixture-context-builder"],
        )
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_true_first_skip_freezes_only_earliest_transition(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "output"

    receipt = _build(fixture, output)
    rows = _rows(output / "first-skip-contexts.jsonl")
    exact = [row for row in rows if row["record_type"] == "exact_context"]
    p_rows = [row for row in exact if row["context_kind"] in {"P_pre", "P_post"}]
    candidates = [
        row for row in rows if row["record_type"] == "first_skip_candidate_receipt"
    ]

    assert [(row["context_kind"], row["source_pred_row_id"]) for row in p_rows] == [
        ("P_post", "pred:sorted:greedy:0:scene:1"),
        ("P_pre", "pred:sorted:greedy:0:scene:1"),
    ]
    assert {
        len(row["exact_prefix"]["self_prefix_generated_token_ids"]) for row in p_rows
    } == {9, 18}
    assert sum(row["status"] == "admitted_candidate" for row in candidates) == 1
    repeated = next(
        row
        for row in candidates
        if row["details"]["candidate_reason"] == "first_skipped_owner_already_frozen"
    )
    assert repeated["source_pred_row_id"] == "pred:sorted:greedy:0:scene:2"
    assert repeated["exact_prefix"] is None
    assert receipt["counts"]["first_skip_candidate_receipt_rows"] == 3
    assert receipt["counts"]["first_skip_candidate_denominator"] == 2
    assert receipt["counts"]["admitted_first_skip_candidates"] == 1
    assert receipt["counts"]["rejected_neutral_first_skip_candidates"] == 1
    assert receipt["counts"]["non_candidate_rejected_history_receipts"] == 1


def test_sealed_control_owners_receive_deduplicated_exact_position_contexts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "output"

    receipt = _build(fixture, output)
    rows = _rows(output / "first-skip-contexts.jsonl")
    control_rows = [
        row
        for row in rows
        if row["record_type"] == "exact_context"
        and row["eligibility"] == "control_diagnostic_only"
    ]

    assert receipt["counts"]["control_diagnostic_context_rows"] == 9
    assert receipt["counts"]["resolved_control_diagnostic_context_rows"] == 9
    assert receipt["counts"]["unresolved_control_reference_contexts"] == 0
    assert {
        owner_id: sorted(
            row["context_kind"]
            for row in control_rows
            if row["gt_owner_id"] == owner_id
        )
        for owner_id in {str(row["gt_owner_id"]) for row in control_rows}
    } == {
        owner_id: ["natural_stop", "reference_scan_position", "root"]
        for owner_id in ("gt:scene:0", "gt:scene:1", "gt:scene:2")
    }
    b_control_rows = [row for row in control_rows if row["gt_owner_id"] == "gt:scene:1"]
    assert len(b_control_rows) == 3
    assert all(
        row["details"]["control_ids"] == ["control:b1:b", "control:b2:c-to-b"]
        for row in b_control_rows
    )
    assert all(row["exact_prefix"] is not None for row in control_rows)
    assert (
        len([row for row in rows if row["context_kind"] in {"B2_before", "B2_after"}])
        == 2
    )


def test_ambiguous_history_is_neutral_and_reconstructibly_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, ambiguous_row=True)
    output = tmp_path / "output"

    receipt = _build(fixture, output)
    rows = _rows(output / "first-skip-contexts.jsonl")
    candidates = [
        row for row in rows if row["record_type"] == "first_skip_candidate_receipt"
    ]
    first = next(row for row in candidates if row["source_pred_row_id"].endswith(":1"))

    assert first["status"] == "rejected_neutral_history"
    assert first["details"]["first_contamination"] == {
        "reason": "globally_ambiguous_owner_or_row",
        "pred_row_id": "pred:sorted:greedy:0:scene:1",
        "original_row_index": 1,
        "row_kind": "complete_prediction",
        "strict_match_status": "ambiguous_neutral",
        "deterministic_gt_owner_id": "gt:scene:2",
        "ambiguity_receipt_ids": ["ambiguity:fixture:row-1"],
    }
    assert first["exact_prefix"] is None
    assert not [row for row in rows if row["context_kind"] in {"P_pre", "P_post"}]
    assert receipt["first_skip_candidate_denominator"]["stop_rule_6"] == {
        "rule": "stop if globally ambiguous first-skip candidates exceed half of the reconstructible first-skip candidate denominator",
        "globally_ambiguous_candidate_count": 2,
        "candidate_denominator": 2,
        "globally_ambiguous_fraction": 1.0,
        "triggered": True,
    }
    b2 = next(row for row in rows if row["record_type"] == "b2_candidate_receipt")
    assert b2["status"] == "rejected_neutral_global_ambiguity"
    chronology = _rows(output / "natural-duplication-chronology.jsonl")
    assert not [row for row in chronology if row.get("gt_owner_id") == "gt:scene:2"]


def test_reviewed_duplicate_history_has_explicit_rejection_and_neutral_geometry(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "output"

    _build(fixture, output)
    contexts = _rows(output / "first-skip-contexts.jsonl")
    duplicate = next(
        row
        for row in contexts
        if row.get("source_pred_row_id") == "pred:sorted:greedy:0:scene:3"
    )
    assert duplicate["details"]["candidate_reason"] == "duplicate_or_repeat_owner"
    assert duplicate["details"]["first_contamination"]["original_row_index"] == 3
    assert duplicate["exact_prefix"] is None
    chronology = _rows(output / "natural-duplication-chronology.jsonl")
    diagnostic = next(
        row
        for row in chronology
        if row["record_type"] == "neutral_semantic_geometry_recurrence"
    )
    assert diagnostic["from_original_row_index"] == 2
    assert diagnostic["to_original_row_index"] == 3
    assert diagnostic["geometry_relation"] == "byte_identical_bbox"
    assert diagnostic["physical_owner_assignment"] == "neutral_not_assigned"
    assert diagnostic["causal_duplication_inferred"] is False


def test_reference_scan_uses_y1_then_x1_then_original_index(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "output"

    _build(fixture, output)
    rows = _rows(output / "first-skip-contexts.jsonl")
    reference = next(
        row
        for row in rows
        if row["context_kind"] == "reference_scan_position"
        and row["registry_id"] == "sentinel-b"
    )

    assert reference["source_pred_row_id"] == "pred:sorted:greedy:0:scene:1"
    assert reference["details"]["sentinel_original_annotation_index"] == 30
    assert len(reference["exact_prefix"]["self_prefix_generated_token_ids"]) == 9


def test_accepts_only_the_optional_candidate_bank_review_source_digest(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    control_path = fixture["paths"]["reviewed_b2_registry"]
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control["source_digests"]["candidate_bank_foil_review_sha256"] = "a" * 64
    control["candidate_bank_review"] = {
        "path": "candidate-bank-foil-review.json",
        "sha256": "a" * 64,
        "status": "lead_reviewed_and_frozen_before_landscape_scoring",
    }
    control["candidate_bank_members"] = [{"ignored_by_task6": True}]
    _write_json(control_path, control)
    fixture["digests"]["reviewed_b2_registry"] = sha256_file(control_path)

    receipt = _build(fixture, tmp_path / "output")

    assert (
        receipt["input_digests"]["reviewed_b2_registry"]
        == fixture["digests"]["reviewed_b2_registry"]
    )


def test_rejects_unknown_control_source_digest_even_with_optional_review_digest(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    control_path = fixture["paths"]["reviewed_b2_registry"]
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control["source_digests"]["candidate_bank_foil_review_sha256"] = "a" * 64
    control["source_digests"]["unrecognized_future_sha256"] = "b" * 64
    _write_json(control_path, control)
    fixture["digests"]["reviewed_b2_registry"] = sha256_file(control_path)

    with pytest.raises(ContextContractError, match="unexpected source-digest contract"):
        _build(fixture, tmp_path / "output")


def test_rejects_v1_manifest_without_fallback(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = fixture["paths"]["owner_ledger"].parent / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "sorted-owner-basin-census-artifact-manifest.v1"
    _write_json(manifest_path, manifest)

    with pytest.raises(ContextContractError, match="v1 fallback is forbidden"):
        _build(fixture, tmp_path / "output")


def test_receipt_binds_command_repository_runtime_and_atomic_exclusive_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "output"

    receipt = _build(fixture, output)

    assert receipt["execution_status"] == "completed"
    assert receipt["command"]["argv"][-1] == "fixture-context-builder"
    assert receipt["repository"]["head"]
    assert receipt["runtime"]["execution_device"] == "cpu"
    assert receipt["runtime"]["gpu_model_or_inference_used"] is False
    assert receipt["task0_v2_binding"]["v1_fallback_allowed"] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite or reuse"):
        _build(fixture, output)


def test_direct_cli_help() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = repository_root / "scripts/research/build_sorted_owner_basin_contexts.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--expected-owner-ledger-sha256" in completed.stdout
    assert CONTEXT_SCHEMA_VERSION == "sorted-owner-basin-first-skip-context.v2"

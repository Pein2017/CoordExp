from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.research import build_sorted_owner_basin_census as census
from scripts.research import build_sorted_owner_basin_cohorts as sut


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _identity(root: Path) -> dict[str, object]:
    base = root / "base"
    adapter = root / "adapter"
    delta = root / "delta"
    for directory in (base, adapter, delta):
        directory.mkdir(parents=True)
        (directory / "fixture.bin").write_bytes(b"fixture")
    return {
        "base": {"path": str(base)},
        "adapter": {"adapter_path": str(adapter)},
        "embedding_delta": {"identity": {"delta_path": str(delta)}},
    }


def _rollout(
    image_id: str,
    seed: int,
    mode: str,
    predictions: list[tuple[str, list[float]]],
) -> dict[str, object]:
    prompt_ids = [11, 12, 13]
    generated_ids = [100 + seed, 101 + seed]
    return {
        "image_id": image_id,
        "example_id": image_id,
        "seed": seed,
        "decode_mode": mode,
        "stop_reason": "im_end",
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": _json_hash(prompt_ids),
        "generated_token_ids": generated_ids,
        "generated_token_ids_sha256": _json_hash(generated_ids),
        "executed_media_sha256": "media-sha",
        "observed_image_grid_thw": [1, 1, 1],
        "predictions": {
            "parse_status": "accepted",
            "metric_bearing": True,
            "dropped_prediction_count": 0,
            "dropped_predictions": [],
            "valid_prediction_count": len(predictions),
            "predictions": [
                {
                    "description": description,
                    "bbox": bbox,
                    "generated_order": index,
                    "object_span_id": f"{image_id}:{seed}:{index}",
                }
                for index, (description, bbox) in enumerate(predictions)
            ],
        },
    }


def _artifact(
    path: Path,
    *,
    identity: dict[str, object],
    mode: str,
    rollouts: list[dict[str, object]],
) -> Path:
    image_ids = sorted({str(item["image_id"]) for item in rollouts})
    prompt_ids = [11, 12, 13]
    payload = {
        "schema_version": census.ROLLOUT_SCHEMA_VERSION,
        "rollout_count": len(rollouts),
        "config": {
            "decode_mode": mode,
            "max_new_tokens": sut.EXPECTED_HORIZON,
            "model_dtype": "fp32",
            "repetition_penalty": 1.0,
            "temperature": 0.0 if mode == "greedy" else 0.4,
            "top_p": 1.0 if mode == "greedy" else 0.95,
            "seeds": [0] if mode == "greedy" else list(sut.EXPECTED_SAMPLED_SEEDS),
            "image_ids": image_ids,
            "resolved_fingerprint": "fixture-fingerprint",
        },
        "model_identity": identity,
        "prompt_metadata": {
            image_id: {
                "prompt_token_ids": prompt_ids,
                "chat_text_sha256": "chat-sha",
                "image_sha256": "source-image-sha",
            }
            for image_id in image_ids
        },
        "rollouts": rollouts,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _production_run(root: Path, identity: dict[str, object]) -> Path:
    root.mkdir()
    scored = root / "gt_vs_pred_scored.jsonl"
    _write_jsonl(
        scored,
        [
            {
                "example_id": "scene",
                "row_id": "scene",
                "row_index": 0,
                "gt": [],
                "pred": [
                    {
                        "description": "horse",
                        "bbox": [60, 0, 70, 10],
                        "generated_order": 0,
                    }
                ],
            }
        ],
    )
    manifest = root / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "terminal_status": "completed",
                "generation_policy": {
                    "do_sample": False,
                    "max_new_tokens": sut.EXPECTED_HORIZON,
                    "repetition_penalty": 1.1,
                    "temperature": 0,
                    "top_p": 1,
                },
                "model_identity": identity,
                "artifacts": {"gt_vs_pred_scored": scored.name},
            }
        ),
        encoding="utf-8",
    )
    return root


def _fixture(tmp_path: Path, *, ambiguous: bool = False, reviewed_bird: bool = False) -> dict[str, Path]:
    identity = _identity(tmp_path / "components")
    panel = tmp_path / "panel.jsonl"
    panel.write_text(
        json.dumps(
            {
                "image_id": "scene",
                "width": 100,
                "height": 100,
                "objects": [
                    {"bbox": [0, 0, 10, 10], "desc": "cat", "category_id": 1},
                    (
                        {"bbox": [0, 0, 10, 10], "desc": "cat", "category_id": 1}
                        if ambiguous
                        else {"bbox": [20, 0, 30, 10], "desc": "dog", "category_id": 2}
                    ),
                    {"bbox": [40, 0, 50, 10], "desc": "bird", "category_id": 3},
                    {"bbox": [60, 0, 70, 10], "desc": "horse", "category_id": 4},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    greedy = _artifact(
        tmp_path / "greedy.json",
        identity=identity,
        mode="greedy",
        rollouts=[
            _rollout(
                "scene",
                0,
                "greedy",
                [
                    ("cat", [0, 0, 10, 10]),
                    ("bird", [49.5, 0, 59.5, 10]),
                    ("tree", [80, 0, 90, 10]),
                ],
            )
        ],
    )
    sampled_rollouts = [
        _rollout(
            "scene",
            seed,
            "sampled",
            ([("dog", [20, 0, 30, 10])] if seed == 21001 and not ambiguous else []),
        )
        for seed in sut.EXPECTED_SAMPLED_SEEDS
    ]
    shard0 = _artifact(
        tmp_path / "shard0.json",
        identity=identity,
        mode="sampled",
        rollouts=sampled_rollouts[:8],
    )
    shard1 = _artifact(
        tmp_path / "shard1.json",
        identity=identity,
        mode="sampled",
        rollouts=sampled_rollouts[8:],
    )
    rp110 = _production_run(tmp_path / "rp110", identity)
    census_dir = tmp_path / "census"
    for attempt in range(5):
        built = census.build_census(
            panel_path=panel,
            greedy_path=greedy,
            sampled_paths=(shard0, shard1),
            production_manifest_path=rp110 / "run_manifest.json",
            require_full_panel=False,
            hash_model_components=False,
        )
        try:
            census.write_census(census_dir, built)
            break
        except census.CensusContractError as error:
            if "repository state changed" not in str(error) or attempt == 4:
                raise
    census_manifest = json.loads((census_dir / "artifact-manifest.json").read_text())

    sources = census_manifest["sources"]
    original_source_digests = {
        "task0_census_artifact_manifest_sha256": _file_hash(
            census_dir / "artifact-manifest.json"
        ),
        "human_refined_panel_sha256": sources["panel"]["sha256"],
        "matched_rp_1_0_greedy_sha256": sources["matched_rp_1_0_greedy"]["sha256"],
        "matched_rp_1_0_sampled_shard_0_sha256": sources[
            "matched_rp_1_0_sampled_shards"
        ][0]["sha256"],
        "matched_rp_1_0_sampled_shard_1_sha256": sources[
            "matched_rp_1_0_sampled_shards"
        ][1]["sha256"],
    }
    visual_image = tmp_path / "visual.jpg"
    far_image = tmp_path / "far.jpg"
    visual_image.write_bytes(b"visual")
    far_image.write_bytes(b"far")
    selection_receipt = tmp_path / "sentinel-selection-receipt.json"
    selection_receipt.write_text(
        json.dumps(
            {
                "schema_version": sut.EXPECTED_SENTINEL_SELECTION_RECEIPT_SCHEMA,
                "selection_status": "lead_reviewed_and_frozen_before_landscape_scoring",
                "claim_scope": "outcome_selected_case_studies_only_no_prevalence",
                "anti_leakage_contract": {"landscape_scores_used_for_selection": False},
                "source_artifacts": original_source_digests,
                "visual_review": {
                    "image_id": "scene",
                    "image_path": str(visual_image),
                    "image_sha256": _file_hash(visual_image),
                    "owners": [],
                },
                "deterministic_far_person_selection": {
                    "image_id": "scene",
                    "image_path": str(far_image),
                    "image_sha256": _file_hash(far_image),
                    "owners": [{"gt_owner_id": "gt:scene:3"}],
                },
            }
        ),
        encoding="utf-8",
    )
    task0_receipt = json.loads((census_dir / "execution-receipt.json").read_text())
    sentinel_source_digests = {
        **original_source_digests,
        "task0_v2_root": str(census_dir.resolve()),
        "task0_execution_receipt_content_sha256": task0_receipt[
            "execution_receipt_content_sha256"
        ],
        "task0_execution_receipt_file_sha256": _file_hash(
            census_dir / "execution-receipt.json"
        ),
        "task0_owner_ledger_sha256": _file_hash(census_dir / "owner-ledger.jsonl"),
        "task0_owner_trajectory_matrix_sha256": _file_hash(
            census_dir / "owner-trajectory-matrix.jsonl"
        ),
    }
    confirmation_receipt = tmp_path / "sentinel-selection-confirmation-receipt.json"
    confirmation_receipt.write_text(
        json.dumps(
            {
                "schema_version": sut.EXPECTED_SENTINEL_CONFIRMATION_RECEIPT_SCHEMA,
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
                    "path": selection_receipt.name,
                    "sha256": _file_hash(selection_receipt),
                },
                "final_task0_v2": {
                    "root": str(census_dir.resolve()),
                    "artifact_manifest_sha256": sentinel_source_digests[
                        "task0_census_artifact_manifest_sha256"
                    ],
                    "execution_receipt_content_sha256": sentinel_source_digests[
                        "task0_execution_receipt_content_sha256"
                    ],
                    "execution_receipt_file_sha256": sentinel_source_digests[
                        "task0_execution_receipt_file_sha256"
                    ],
                    "owner_ledger_sha256": sentinel_source_digests[
                        "task0_owner_ledger_sha256"
                    ],
                    "owner_trajectory_matrix_sha256": sentinel_source_digests[
                        "task0_owner_trajectory_matrix_sha256"
                    ],
                },
                "confirmed_sentinels": [
                    {
                        "gt_owner_id": "gt:scene:3",
                        "trajectory_count": 17,
                        "max_semantic_compatible_iou": 0,
                        "strict_match_count": 0,
                        "decision_eligible": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sentinel_registry = tmp_path / "sentinel-registry.json"
    sentinel_registry.write_text(
        json.dumps(
            {
                "schema_version": sut.EXPECTED_SENTINEL_SCHEMA,
                "selection_status": (
                    "lead_frozen_before_scoring_and_reconfirmed_against_final_task0_v2"
                ),
                "claim_scope": "outcome_selected_case_studies_only_no_prevalence",
                "selection_receipt": {
                    "path": selection_receipt.name,
                    "sha256": _file_hash(selection_receipt),
                },
                "selection_confirmation_receipt": {
                    "path": confirmation_receipt.name,
                    "sha256": _file_hash(confirmation_receipt),
                },
                "source_digests": sentinel_source_digests,
                "sentinels": [
                    {
                        "sentinel_id": "sentinel:fixture:horse",
                        "sentinel_kind": "fixture",
                        "gt_owner_id": "gt:scene:3",
                        "image_id": "scene",
                        "original_annotation_index": 3,
                        "description": "horse",
                        "bbox_xyxy": [60.0, 0.0, 70.0, 10.0],
                        "prior_non_recovery_status": (
                            "verified_primary_natural_zero_spatial_support"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    control_registry = tmp_path / "control-registry.json"
    control_registry.write_text(
        json.dumps(
            {
                "schema_version": sut.EXPECTED_CONTROL_SCHEMA,
                "status": "lead_frozen_before_scoring_and_resealed_to_final_task0_v2",
                "source_digests": {
                    "task0_v2_root": str(census_dir.resolve()),
                    "task0_census_artifact_manifest_sha256": _file_hash(
                        census_dir / "artifact-manifest.json"
                    ),
                    "task0_execution_receipt_content_sha256": task0_receipt[
                        "execution_receipt_content_sha256"
                    ],
                    "task0_execution_receipt_file_sha256": _file_hash(
                        census_dir / "execution-receipt.json"
                    ),
                    "owner_ledger_sha256": _file_hash(census_dir / "owner-ledger.jsonl"),
                    "owner_trajectory_matrix_sha256": _file_hash(
                        census_dir / "owner-trajectory-matrix.jsonl"
                    ),
                    "sentinel_selection_confirmation_receipt_sha256": _file_hash(
                        confirmation_receipt
                    ),
                },
                "controls": (
                    []
                    if ambiguous
                    else [
                        {
                            "control_id": "control:fixture:cat",
                            "role": "strict_visible_true_positive",
                            "gt_owner_id": "gt:scene:0",
                            "matched_pred_row_id": "pred:sorted:greedy:0:scene:0",
                        },
                        *(
                            [
                                {
                                    "control_id": "control:fixture:bird-b1",
                                    "role": "b1_loose_only",
                                    "gt_owner_id": "gt:scene:2",
                                }
                            ]
                            if reviewed_bird
                            else []
                        ),
                    ]
                ),
            }
        ),
        encoding="utf-8",
    )
    return {
        "census": census_dir,
        "sentinels": sentinel_registry,
        "controls": control_registry,
        "rp110": rp110,
    }


def _build(paths: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    return sut.build_cohorts(
        census_dir=paths["census"],
        sentinel_registry_path=paths["sentinels"],
        control_registry_path=paths["controls"],
        require_full_panel=False,
        **kwargs,
    )


def _by_owner(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["gt_owner_id"]: row for row in payload["cohort_assignments"]}


def test_any_positive_overlap_blocks_no_free_and_thresholds_are_diagnostic(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    payload = _build(paths, rp110_run_dir=None)
    owners = _by_owner(payload)

    assert owners["gt:scene:0"]["cohort"] == "greedy_strict_present"
    assert owners["gt:scene:1"]["cohort"] == "strict_rescued"
    bird = owners["gt:scene:2"]
    assert bird["cohort"] == "positive_overlap_neutral"
    assert bird["natural_spatial_support"]["any_positive_overlap"] is True
    assert bird["natural_spatial_support"]["no_free_spatial_support"] is False
    assert bird["natural_spatial_support"]["max_semantic_compatible_gt_coverage"] > 0
    diagnostic = bird["natural_spatial_support"]["meaningful_loose_iou_diagnostic"]
    assert diagnostic["positive_at_threshold"] == {
        "lower": False,
        "predeclared": False,
        "upper": False,
    }
    assert diagnostic["stable_across_threshold_band"] is True
    assert diagnostic["cohort_rule_independent_of_diagnostic_thresholds"] is True
    assert bird["natural_spatial_support"]["blocks_no_free"] is True
    assert bird["natural_spatial_support"]["admits_loose_only_b1"] is False
    assert bird["primary_eligibility"]["status"] == (
        "excluded_unreviewed_or_subthreshold_positive_overlap"
    )
    assert bird["primary_eligibility"]["included_in_b1_or_no_free"] is False
    assert bird["primary_eligibility"]["included_in_repair_or_calibration"] is False
    horse = owners["gt:scene:3"]
    assert horse["cohort"] == "no_free_spatial_support"
    assert horse["sentinel"]["outcome_selected_no_prevalence"] is True

    queued = {row["pred_row_id"]: row for row in payload["manual_review_queue"]}
    assert "pred:sorted:greedy:0:scene:1" in queued
    assert queued["pred:sorted:greedy:0:scene:1"]["physical_axis_status"] == "unresolved"
    assert queued["pred:sorted:greedy:0:scene:1"]["semantic_axis_status"] == "unresolved"
    assert queued["pred:sorted:greedy:0:scene:1"]["physical_relation"] is None
    assert queued["pred:sorted:greedy:0:scene:1"]["original_row_index"] == 1

    output = tmp_path / "cohorts"
    written = sut.write_cohorts(output, payload, command_line=["fixture"])
    assert set(path.name for path in written.values()) == {
        "cohort-assignments.jsonl",
        "sampling-support.jsonl",
        "manual-review-queue.jsonl",
        "cohort-receipt.json",
        "artifact-manifest.json",
    }
    receipt = json.loads((output / "cohort-receipt.json").read_text())
    assert receipt["status_enums"]["cohort"] == list(sut.COHORT_STATUSES)
    assert receipt["execution_status"] == "completed_cpu_only"
    assert receipt["runtime"]["execution_device"] == "cpu"
    assert receipt["repository"]["tracked_dirty_diff_sha256"]
    assert receipt["receipt_sha256"] == sut._receipt_digest(receipt)
    assert receipt["input_digests"]["sentinel_selection_confirmation_receipt"] == (
        _file_hash(paths["sentinels"].parent / "sentinel-selection-confirmation-receipt.json")
    )


def test_reviewed_b1_control_can_admit_subthreshold_positive_overlap(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, reviewed_bird=True)
    bird = _by_owner(_build(paths, rp110_run_dir=None))["gt:scene:2"]

    assert bird["natural_spatial_support"]["meaningful_loose_iou_diagnostic"][
        "stable_meaningful_positive_across_threshold_band"
    ] is False
    assert bird["natural_spatial_support"]["reviewed_b1_control"] is True
    assert bird["cohort"] == "loose_only_b1"


def test_ambiguous_optimal_owners_and_rows_are_neutral_excluded(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, ambiguous=True)
    payload = _build(paths, rp110_run_dir=None)
    owners = _by_owner(payload)

    for owner_id in ("gt:scene:0", "gt:scene:1"):
        assert owners[owner_id]["cohort"] == "strict_ambiguity_neutral"
        eligibility = owners[owner_id]["primary_eligibility"]
        assert eligibility["status"] == "excluded_global_strict_ambiguity"
        assert eligibility["included_in_greedy_tp_fn"] is False
        assert eligibility["included_in_strict_rescued"] is False
        assert eligibility["included_in_b1_or_no_free"] is False
        assert eligibility["included_in_repair_or_calibration"] is False
    ambiguous_row = next(
        row
        for row in payload["manual_review_queue"]
        if row["pred_row_id"] == "pred:sorted:greedy:0:scene:0"
    )
    assert ambiguous_row["strict_match_status"] == "ambiguous_neutral"
    assert ambiguous_row["ambiguity_receipt_ids"]
    assert ambiguous_row["decision_eligibility"] == {
        "status": "excluded_global_strict_ambiguity",
        "included_in_b2_repair_or_calibration": False,
        "raw_diagnostics_preserved": True,
    }


def test_output_directory_creation_is_exclusive_even_when_existing_directory_is_empty(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    payload = _build(paths, rp110_run_dir=None)
    output = tmp_path / "already-exists"
    output.mkdir()

    with pytest.raises(sut.CohortContractError, match="existing output directory"):
        sut.write_cohorts(output, payload, command_line=["fixture"])


def test_rp110_positive_is_separate_and_never_changes_primary_cohort(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    without_rp110 = _by_owner(_build(paths, rp110_run_dir=None))["gt:scene:3"]
    with_payload = _build(paths, rp110_run_dir=paths["rp110"])
    with_rp110 = _by_owner(with_payload)["gt:scene:3"]

    assert without_rp110["cohort"] == with_rp110["cohort"] == "no_free_spatial_support"
    assert with_rp110["auxiliary_positive_evidence"]["rp1_10_greedy_status"] == (
        "strict_positive"
    )
    assert with_rp110["auxiliary_positive_evidence"]["blocks_later_absence_claim"] is True
    support = next(
        row
        for row in with_payload["sampling_support"]
        if row["gt_owner_id"] == "gt:scene:3"
        and row["support_panel"] == "auxiliary_rp1_10_greedy"
    )
    assert support["excluded_from_primary_k16_denominator"] is True
    assert support["evidence_semantics"] == "positive_blocks_later_absence_claim"


def test_registered_sampling_null_is_non_evidence(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    artifact = tmp_path / "registered-predictions.jsonl"
    _write_jsonl(artifact, [])
    policy_receipt = tmp_path / "registered-policy.json"
    policy = {
        "schema_version": sut.REGISTERED_POLICY_RECEIPT_SCHEMA_VERSION,
        "status": "frozen_before_generation_and_scoring",
        "registration_id": "fixture-null-registration",
        "policy": {
            "decode_mode": "sampled",
            "repetition_penalty": 1.0,
            "temperature": 0.4,
            "top_p": 0.95,
            "seeds": [31001],
            "max_new_tokens": sut.EXPECTED_HORIZON,
            "donor_selection_rule": "first strict row in ascending seed order",
        },
        "source_artifact": {"path": str(artifact), "sha256": _file_hash(artifact)},
    }
    policy["receipt_sha256"] = sut._receipt_digest(policy)
    policy_receipt.write_text(json.dumps(policy), encoding="utf-8")
    support_path = tmp_path / "registered-support.jsonl"
    rows = [
        {
            "schema_version": sut.REGISTERED_SUPPORT_SCHEMA_VERSION,
            "registration_id": "fixture-null-registration",
            "gt_owner_id": f"gt:scene:{index}",
            "policy_stratum": "registered_rp_1.00",
            "decode_mode": "sampled",
            "support_status": "null_no_positive_evidence",
            "supporting_pred_row_ids": [],
            "max_semantic_compatible_iou": 0.0,
            "source_artifact": {"path": str(artifact), "sha256": _file_hash(artifact)},
            "policy_receipt": {
                "path": str(policy_receipt),
                "sha256": _file_hash(policy_receipt),
            },
        }
        for index in range(4)
    ]
    _write_jsonl(support_path, rows)

    payload = _build(
        paths,
        rp110_run_dir=None,
        registered_support_paths=[support_path],
    )
    horse = _by_owner(payload)["gt:scene:3"]
    assert horse["cohort"] == "no_free_spatial_support"
    assert horse["auxiliary_positive_evidence"]["registered_sampling_any_status"] == (
        "null_no_positive_evidence"
    )
    assert horse["auxiliary_positive_evidence"]["blocks_later_absence_claim"] is False
    row = next(
        item
        for item in payload["sampling_support"]
        if item["gt_owner_id"] == "gt:scene:3"
        and item["support_panel"] == "registered_sampling"
    )
    assert row["evidence_semantics"] == "null_is_no_positive_evidence_not_absence_evidence"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("empty_positive", "requires nonempty evidence"),
        ("bogus_id", "unknown row"),
        ("wrong_owner", "wrong owner/semantic relation"),
        ("unfrozen", "policy is not frozen"),
    ],
)
def test_registered_sampling_rejects_invalid_evidence(
    tmp_path: Path, case: str, message: str
) -> None:
    paths = _fixture(tmp_path)
    artifact = tmp_path / "registered-predictions.jsonl"
    prediction_id = "pred:sorted:registered:31001:scene:0"
    predictions = [
        {
            "schema_version": sut.REGISTERED_PREDICTION_SCHEMA_VERSION,
            "registration_id": "fixture-registration",
            "pred_row_id": prediction_id,
            "image_id": "scene",
            "policy_stratum": "registered_rp_1.00",
            "decode_mode": "sampled",
            "seed": 31001,
            "original_row_index": 0,
            "normalized_description": "cat",
            "bbox_xyxy": [0, 0, 10, 10],
        }
    ]
    _write_jsonl(artifact, predictions)
    policy_receipt = tmp_path / "registered-policy.json"
    policy = {
        "schema_version": sut.REGISTERED_POLICY_RECEIPT_SCHEMA_VERSION,
        "status": (
            "draft_not_frozen" if case == "unfrozen" else "frozen_before_generation_and_scoring"
        ),
        "registration_id": "fixture-registration",
        "policy": {
            "decode_mode": "sampled",
            "repetition_penalty": 1.0,
            "temperature": 0.4,
            "top_p": 0.95,
            "seeds": [31001],
            "max_new_tokens": sut.EXPECTED_HORIZON,
            "donor_selection_rule": "first strict row in ascending seed order",
        },
        "source_artifact": {"path": str(artifact), "sha256": _file_hash(artifact)},
    }
    policy["receipt_sha256"] = sut._receipt_digest(policy)
    policy_receipt.write_text(json.dumps(policy), encoding="utf-8")
    claimed_owner = "gt:scene:1" if case == "wrong_owner" else "gt:scene:0"
    supporting_ids = (
        []
        if case == "empty_positive"
        else (["pred:bogus"] if case == "bogus_id" else [prediction_id])
    )
    rows = []
    for index in range(4):
        owner_id = f"gt:scene:{index}"
        is_claim = owner_id == claimed_owner
        is_correct_companion = case == "wrong_owner" and owner_id == "gt:scene:0"
        rows.append(
            {
                "schema_version": sut.REGISTERED_SUPPORT_SCHEMA_VERSION,
                "registration_id": "fixture-registration",
                "gt_owner_id": owner_id,
                "policy_stratum": "registered_rp_1.00",
                "decode_mode": "sampled",
                "support_status": (
                    "strict_positive"
                    if is_claim or is_correct_companion
                    else "null_no_positive_evidence"
                ),
                "supporting_pred_row_ids": (
                    [prediction_id]
                    if is_correct_companion
                    else (supporting_ids if is_claim else [])
                ),
                "max_semantic_compatible_iou": 1.0 if owner_id == "gt:scene:0" else 0.0,
                "source_artifact": {"path": str(artifact), "sha256": _file_hash(artifact)},
                "policy_receipt": {
                    "path": str(policy_receipt),
                    "sha256": _file_hash(policy_receipt),
                },
            }
        )
    support_path = tmp_path / "registered-support.jsonl"
    _write_jsonl(support_path, rows)

    with pytest.raises(sut.CohortContractError, match=message):
        _build(
            paths,
            rp110_run_dir=None,
            registered_support_paths=[support_path],
        )


def _refresh_manifest_artifact(census_dir: Path, key: str, filename: str) -> None:
    path = census_dir / "artifact-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"][key]["sha256"] = _file_hash(census_dir / filename)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_rejects_stale_task0_digest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    owner_path = paths["census"] / "owner-ledger.jsonl"
    owner_path.write_text(owner_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(sut.CohortContractError, match="stale digest"):
        _build(paths, rp110_run_dir=None)


def test_rejects_task0_v1_matcher_even_with_refreshed_artifact_digest(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    matcher_path = paths["census"] / "matcher-contract.json"
    matcher = json.loads(matcher_path.read_text())
    matcher["schema_version"] = "sorted-owner-basin-matcher.v1"
    matcher["matcher_id"] = "sorted-owner-basin-matcher.v1"
    matcher_path.write_text(json.dumps(matcher), encoding="utf-8")
    _refresh_manifest_artifact(
        paths["census"], "matcher_contract", "matcher-contract.json"
    )

    with pytest.raises(sut.CohortContractError, match="unsupported schema"):
        _build(paths, rp110_run_dir=None)


def test_rejects_task0_receipt_without_final_cpu_runtime_binding(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt_path = paths["census"] / "execution-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["runtime"]["execution_device"] = "cuda"
    receipt["execution_receipt_content_sha256"] = sut._task0_receipt_digest(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest = json.loads((paths["census"] / "artifact-manifest.json").read_text())
    manifest["execution_receipt_content_sha256"] = receipt[
        "execution_receipt_content_sha256"
    ]
    matcher_path = paths["census"] / "matcher-contract.json"
    matcher = json.loads(matcher_path.read_text())
    matcher["execution_receipt_content_sha256"] = receipt[
        "execution_receipt_content_sha256"
    ]
    matcher_path.write_text(json.dumps(matcher), encoding="utf-8")

    with pytest.raises(sut.CohortContractError, match="CPU-only execution"):
        sut._validate_task0_execution_receipt(
            receipt_path,
            manifest,
            {"matcher": matcher_path},
        )


def test_rejects_stale_sentinel_confirmation_source_binding(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    confirmation_path = paths["sentinels"].parent / (
        "sentinel-selection-confirmation-receipt.json"
    )
    confirmation = json.loads(confirmation_path.read_text())
    confirmation["final_task0_v2"]["artifact_manifest_sha256"] = "0" * 64
    confirmation_path.write_text(json.dumps(confirmation), encoding="utf-8")
    confirmation_digest = _file_hash(confirmation_path)

    sentinel_registry = json.loads(paths["sentinels"].read_text())
    sentinel_registry["selection_confirmation_receipt"]["sha256"] = confirmation_digest
    paths["sentinels"].write_text(json.dumps(sentinel_registry), encoding="utf-8")
    control_path = paths["controls"]
    control_registry = json.loads(control_path.read_text())
    control_registry["source_digests"][
        "sentinel_selection_confirmation_receipt_sha256"
    ] = confirmation_digest
    control_path.write_text(json.dumps(control_registry), encoding="utf-8")

    with pytest.raises(sut.CohortContractError, match="stale final Task0"):
        _build(paths, rp110_run_dir=None)


def test_rejects_missing_k16_seed_even_with_refreshed_file_digest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    matrix_path = paths["census"] / "owner-trajectory-matrix.jsonl"
    rows = [
        json.loads(line)
        for line in matrix_path.read_text().splitlines()
        if line.strip()
    ]
    rows = [
        row
        for row in rows
        if not (
            row["gt_owner_id"] == "gt:scene:3"
            and row["decode_mode"] == "sampled"
            and row["seed"] == 21016
        )
    ]
    _write_jsonl(matrix_path, rows)
    _refresh_manifest_artifact(
        paths["census"], "owner_trajectory_matrix", "owner-trajectory-matrix.jsonl"
    )

    with pytest.raises(sut.CohortContractError, match="complete owner-by-K16"):
        _build(paths, rp110_run_dir=None)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("forced_continuation", True, "forced-continuation"),
        ("policy_stratum", "production_rp_1.10_greedy", "mixes a non-primary policy"),
        ("pred_row_id", "pred:sorted:greedy:0:scene:99", "renumbered"),
    ],
)
def test_rejects_forced_policy_mixed_or_renumbered_rows(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    paths = _fixture(tmp_path)
    prediction_path = paths["census"] / "prediction-row-ledger.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
    rows[0][field] = value
    _write_jsonl(prediction_path, rows)
    _refresh_manifest_artifact(
        paths["census"], "prediction_ledger", "prediction-row-ledger.jsonl"
    )

    with pytest.raises(sut.CohortContractError, match=message):
        _build(paths, rp110_run_dir=None)


def test_direct_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/research/build_sorted_owner_basin_cohorts.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--census-dir" in result.stdout
    assert "--control-registry" in result.stdout
    assert "--rp110-run-dir" in result.stdout
    assert "--output-dir" in result.stdout

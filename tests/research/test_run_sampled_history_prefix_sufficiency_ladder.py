from __future__ import annotations

import hashlib
import json

import pytest

from scripts.research.validate_sampled_history_prefix_sufficiency_ladder_union import validate_union
from scripts.research.run_sampled_history_prefix_sufficiency_ladder import (
    StageTwoValidationError,
    reconstruct_sampled_prefix_ladder,
    summarize_continuation,
)


def _row(index: int, token_ids: list[int], owner: str | None = None, *, stop: str = "complete_row") -> dict[str, object]:
    return {
        "row_index": index,
        "status": "success" if stop == "complete_row" else "success",
        "accepted_complete_row": stop == "complete_row",
        "raw_generated_token_ids": token_ids,
        "row_stop": {"stop_reason": stop},
        "strict_matched_owner_ids": [] if owner is None else [owner],
        "entity_matches": [] if owner is None else [{"status": "matched", "matched_entity_id": owner}],
    }


def test_reconstructs_exact_prefixes_without_retokenizing() -> None:
    rows = [_row(0, [10, 11], "a"), _row(1, [12, 13], "b"), _row(2, [14, 15], "target")]
    image = {"trajectories": [{"mode": "sample", "seed": 13, "horizon_rows_generated": 3, "rows": rows}]}
    target = {
        "image_id": "1",
        "target_owner_id": "target",
        "sampled_seed": 13,
        "first_sampled_row_index_zero_based": 2,
        "target_parent_prefix_sha256": "b3b0b6b4c1a9f4c3d2d2f4a1f9c65e22d454c5b846c8b7f3dcb4c1e51b6d0bf2",
    }
    # Use the helper's hash instead of a hand-maintained digest.
    from scripts.research.run_local_branch_causal_value import hash_prefix_token_ids

    target["target_parent_prefix_sha256"] = hash_prefix_token_ids([10, 11, 12, 13])
    result = reconstruct_sampled_prefix_ladder(image, target)
    assert [item["k"] for item in result["prefixes"]] == [0, 1, 2]
    assert result["prefixes"][0]["prefix_token_ids"] == []
    assert result["prefixes"][1]["prefix_token_ids"] == [10, 11]
    assert result["prefixes"][2]["prefix_token_ids"] == [10, 11, 12, 13]
    assert result["prefixes"][2]["prefix_owner_ids"] == ["a", "b"]


def test_reconstruction_refuses_parent_hash_drift() -> None:
    rows = [_row(0, [10], "a"), _row(1, [11], "target")]
    image = {"trajectories": [{"mode": "sample", "seed": 13, "rows": rows}]}
    target = {
        "image_id": "1",
        "target_owner_id": "target",
        "sampled_seed": 13,
        "first_sampled_row_index_zero_based": 1,
        "target_parent_prefix_sha256": "wrong",
    }
    with pytest.raises(StageTwoValidationError, match="parent hash"):
        reconstruct_sampled_prefix_ladder(image, target)


def test_reconstruction_refuses_non_clean_target_row() -> None:
    rows = [_row(0, [10], "a"), _row(1, [11], "target", stop="terminal")]
    image = {"trajectories": [{"mode": "sample", "seed": 13, "rows": rows}]}
    target = {
        "image_id": "1",
        "target_owner_id": "target",
        "sampled_seed": 13,
        "first_sampled_row_index_zero_based": 1,
        "target_parent_prefix_sha256": "unused",
    }
    with pytest.raises(StageTwoValidationError, match="clean complete"):
        reconstruct_sampled_prefix_ladder(image, target)


def test_summary_records_target_hit_duplicates_and_terminal() -> None:
    rows = [
        _row(2, [20], "old"),
        _row(3, [21], "target"),
        _row(3, [21], "target"),
        _row(4, [22], stop="terminal"),
    ]
    result = summarize_continuation(
        rows,
        prefix_owner_ids=["old"],
        target_owner_id="target",
        target_start_row_index=2,
        generated_token_count=3,
        total_token_budget=512,
        horizon_rows_complete=False,
    )
    assert result["reachability_state"] == "hit"
    assert result["target_first_hit_row_index"] == 3
    assert result["duplicate_owner_ids"] == ["old", "target"]
    assert result["natural_terminal_row_index"] == 4
    assert result["malformed_row_indices"] == []


def test_summary_separates_malformed_from_terminal() -> None:
    result = summarize_continuation(
        [_row(2, [20], stop="malformed_limit")],
        prefix_owner_ids=[],
        target_owner_id="target",
        target_start_row_index=2,
        generated_token_count=1,
        total_token_budget=512,
        horizon_rows_complete=False,
    )
    assert result["reachability_state"] == "unresolved"
    assert result["malformed_row_indices"] == [2]
    assert result["natural_terminal_row_index"] is None


def test_budget_censor_preserves_malformed_evidence() -> None:
    result = summarize_continuation(
        [_row(2, [20], stop="malformed_limit")],
        prefix_owner_ids=[],
        target_owner_id="target",
        target_start_row_index=2,
        generated_token_count=512,
        total_token_budget=512,
        horizon_rows_complete=False,
    )
    assert result["reachability_state"] == "right_censored"
    assert result["malformed_row_indices"] == [2]
    assert result["unresolved_row_indices"] == []


def test_summary_distinguishes_clean_miss_and_right_censor() -> None:
    clean_miss = summarize_continuation(
        [_row(2, [20], "other"), _row(3, [21], stop="terminal")],
        prefix_owner_ids=[],
        target_owner_id="target",
        target_start_row_index=2,
        generated_token_count=2,
        total_token_budget=512,
        horizon_rows_complete=False,
    )
    censored = summarize_continuation(
        [_row(2, [20], "other")],
        prefix_owner_ids=[],
        target_owner_id="target",
        target_start_row_index=2,
        generated_token_count=512,
        total_token_budget=512,
        horizon_rows_complete=False,
    )
    assert clean_miss["reachability_state"] == "clean_miss"
    assert censored["reachability_state"] == "right_censored"


def test_horizon_complete_without_terminal_is_clean_miss() -> None:
    result = summarize_continuation(
        [_row(2, [20], "other")],
        prefix_owner_ids=[],
        target_owner_id="target",
        target_start_row_index=2,
        generated_token_count=1,
        total_token_budget=512,
        horizon_rows_complete=True,
    )
    assert result["reachability_state"] == "clean_miss"


def test_complete_syntax_with_unresolved_owner_is_unresolved() -> None:
    row = _row(2, [20], "other")
    row["entity_matches"] = [
        {"status": "matched", "matched_entity_id": "other"},
        {"status": "ambiguous", "candidates": [{"entity_id": "target", "iou": 0.2}]},
    ]
    result = summarize_continuation(
        [row],
        prefix_owner_ids=[],
        target_owner_id="target",
        target_start_row_index=2,
        generated_token_count=1,
        total_token_budget=512,
        horizon_rows_complete=True,
    )
    assert result["reachability_state"] == "unresolved"
    assert result["target_relevant_ambiguity"][0]["candidate_iou"] == 0.2


def test_unrelated_ambiguity_is_review_warning_not_global_unresolved() -> None:
    row = _row(2, [20], "other")
    row["entity_matches"] = [
        {
            "status": "ambiguous",
            "prediction_index": 0,
            "candidates": [{"entity_id": "unrelated", "iou": 0.2}],
        }
    ]
    result = summarize_continuation(
        [row],
        prefix_owner_ids=[],
        target_owner_id="target",
        target_start_row_index=2,
        generated_token_count=1,
        total_token_budget=512,
        horizon_rows_complete=True,
    )
    assert result["reachability_state"] == "clean_miss"
    assert result["unresolved_row_indices"] == []
    assert result["crop_review_warnings"]


def _write_stage_two_admission(tmp_path, image_ids=("1", "2")):
    stage_one_records = []
    for image_id in image_ids:
        stage_path = tmp_path / f"stage-one-{image_id}.json"
        stage_path.write_text(
            json.dumps({
                "schema_version": "sampled_history_target_reachability.v1",
                "phase": "stage_one_extended_root_greedy_screen",
                "images": [{
                    "image_id": image_id,
                    "target_classification": {"label": "delayed"},
                }],
            }),
            encoding="utf-8",
        )
        stage_one_records.append({
            "image_id": image_id,
            "path": str(stage_path.resolve()),
            "sha256": hashlib.sha256(stage_path.read_bytes()).hexdigest(),
        })
    union_path = tmp_path / "stage-one-union.json"
    union_path.write_text(
        json.dumps({
            "schema_version": "sampled_history_target_reachability.union.v1",
            "passed": True,
            "images": stage_one_records,
        }),
        encoding="utf-8",
    )
    union_sha = hashlib.sha256(union_path.read_bytes()).hexdigest()
    entries = {
        image_id: {
            "image_id": image_id,
            "target_owner_id": f"owner-{image_id}",
            "final_recall_stratum": "delayed",
            "run_prefix_ladder": True,
            "primary_causal_admission": True,
            "allowed_claim": "target_reachability_only",
        }
        for image_id in image_ids
    }
    admission_path = tmp_path / "stage-two-admission.json"
    admission_path.write_text(
        json.dumps({
            "schema_version": 1,
            "unit_id": "2026-07-19-sampled-history-target-reachability-and-complete-row-value",
            "source_stage_one_union": str(union_path.resolve()),
            "source_stage_one_union_sha256": union_sha,
            "targets": list(entries.values()),
        }),
        encoding="utf-8",
    )
    return admission_path, entries


def test_stage_two_union_requires_shared_identity(tmp_path) -> None:
    admission_path, entries = _write_stage_two_admission(tmp_path)

    def write(path, image_id, *, dtype="fp32"):
        path.write_text(
            json.dumps({
                "schema_version": "sampled_history_target_reachability.stage_two.v1",
                "phase": "stage_two_prefix_sufficiency_ladder",
                "source_identity": {"stage_two_runner_sha256": "runner", "reused_local_branch_helper_sha256": "helper"},
                "frozen_inputs": {
                    "manifest_sha256": "manifest",
                    "stage_two_admission": str(admission_path.resolve()),
                    "stage_two_admission_sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
                    "stage_one_union": str((tmp_path / "stage-one-union.json").resolve()),
                    "stage_one_union_sha256": hashlib.sha256((tmp_path / "stage-one-union.json").read_bytes()).hexdigest(),
                    "stage2_admission_entries": entries,
                },
                "model_identity": {"model": "same"},
                "config": {"model_dtype": dtype, "device": "cuda:0"},
                "images": [{
                    "image_id": image_id,
                    "stage2_admission_entry": entries[image_id],
                    "prefix_evaluations": [{
                        "reachability_state": "clean_miss",
                        "prefix": {"prefix_token_ids_sha256": "prefix"},
                        "continuation": {"rows": []},
                    }],
                }],
            }),
            encoding="utf-8",
        )

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write(first, "1")
    write(second, "2")
    assert validate_union(shard_paths=[first, second], admission_path=admission_path)["passed"] is True
    write(second, "2", dtype="fp16")
    with pytest.raises(ValueError, match="identity conflict"):
        validate_union(shard_paths=[first, second], admission_path=admission_path)

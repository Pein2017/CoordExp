from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.run_sampled_history_target_reachability import (
    StageOneValidationError,
    classify_target_from_extended_rows,
    compare_execution_identity,
    first_eight_parity,
    plausible_target_ambiguity,
    _find_sample_target,
    sampled_owner_union,
    strict_owner_union,
    validate_frozen_inputs,
    sha256_file,
)


def _row(index: int, token_ids: list[int], *, owner: str | None = None, stop: str = "complete_row") -> dict[str, object]:
    return {
        "row_index": index,
        "status": "success" if stop != "failed" else "failed",
        "raw_generated_token_ids": token_ids,
        "row_stop": {"stop_reason": stop},
        "strict_matched_owner_ids": [] if owner is None else [owner],
        "entity_matches": [] if owner is None else [{"status": "matched", "matched_entity_id": owner}],
    }


def test_first_eight_parity_requires_exact_raw_rows() -> None:
    rows = [_row(i, [i, 9]) for i in range(8)]
    source = {"rows": [_row(i, [i, 9]) for i in range(8)]}
    assert first_eight_parity(rows, source)["passed"] is True
    changed = list(rows)
    changed[3] = _row(3, [3, 8])
    result = first_eight_parity(changed, source)
    assert result["passed"] is False
    assert result["checks"][3]["raw_token_ids_equal"] is False


def test_execution_identity_requires_all_available_prompt_and_media_fields() -> None:
    prompt = {"image_sha256": "img", "chat_text_sha256": "chat", "prompt_token_ids_sha256": "prompt", "width": 1216, "height": 736}
    runtime = {"executed_media_sha256": "media", "executed_prompt_token_ids_sha256": "exec", "observed_image_grid_thw": [1, 46, 76]}
    assert compare_execution_identity(observed_prompt=prompt, observed_runtime=runtime, discovery_prompt=prompt, discovery_runtime=runtime)["passed"] is True
    changed = dict(prompt, chat_text_sha256="different")
    result = compare_execution_identity(observed_prompt=changed, observed_runtime=runtime, discovery_prompt=prompt, discovery_runtime=runtime)
    assert result["passed"] is False
    assert result["checks"]["chat_text_sha256"] is False


def test_target_is_delayed_only_after_row_eight() -> None:
    rows = [_row(i, [i, 9], owner=f"owner-{i}") for i in range(8)]
    rows.append(_row(8, [8, 9], owner="target"))
    result = classify_target_from_extended_rows(rows, target_owner_id="target")
    assert result["label"] == "delayed"
    assert result["target_first_row_index"] == 8


def test_terminal_before_target_is_terminally_omitted() -> None:
    rows = [_row(0, [1, 2], owner="other"), _row(1, [3], stop="terminal")]
    result = classify_target_from_extended_rows(rows, target_owner_id="target")
    assert result["label"] == "terminally_omitted"
    assert result["terminal_row_index"] == 1


def test_malformed_is_not_silently_called_terminal() -> None:
    rows = [_row(0, [1, 2], owner="other"), _row(1, [3], stop="malformed_limit")]
    result = classify_target_from_extended_rows(rows, target_owner_id="target")
    assert result["label"] == "unresolved"
    assert result["refusal_reason"] == "malformed_or_failed_before_target"


def test_ambiguous_target_then_terminal_is_unresolved() -> None:
    ambiguous = _row(0, [1, 2])
    ambiguous["entity_matches"] = [{
        "status": "ambiguous",
        "prediction_index": 0,
        "candidates": [{"entity_id": "target", "iou": 0.12, "center_distance_norm": 0.02}],
    }]
    assert plausible_target_ambiguity(ambiguous, target_owner_id="target")["candidate_iou"] == 0.12
    result = classify_target_from_extended_rows([ambiguous, _row(1, [3], stop="terminal")], target_owner_id="target")
    assert result["label"] == "unresolved"
    assert result["refusal_reason"] == "plausible_target_ambiguity_before_terminal"


def test_ambiguous_target_at_budget_cap_is_unresolved() -> None:
    ambiguous = _row(0, [1] * 512)
    ambiguous["entity_matches"] = [{
        "status": "unmatched",
        "prediction_index": 0,
        "candidates": [{"entity_id": "target", "iou": 0.03}],
    }]
    result = classify_target_from_extended_rows([ambiguous], target_owner_id="target")
    assert result["label"] == "unresolved"
    assert result["refusal_reason"] == "plausible_target_ambiguity_at_budget_cap"


def test_budget_cap_is_right_censored() -> None:
    rows = [_row(0, [1] * 512)]
    result = classify_target_from_extended_rows(rows, target_owner_id="target")
    assert result["label"] == "right_censored"
    assert result["generated_token_count"] == 512


def test_target_before_row_eight_is_not_a_delayed_result() -> None:
    rows = [_row(0, [1, 2], owner="target")]
    result = classify_target_from_extended_rows(rows, target_owner_id="target")
    assert result["label"] == "unresolved"


def test_mixed_owner_target_row_is_unresolved() -> None:
    row = _row(8, [8, 9], owner="target")
    row["strict_matched_owner_ids"] = ["target", "neighbor"]
    row["entity_matches"] = [
        {"status": "matched", "matched_entity_id": "target"},
        {"status": "matched", "matched_entity_id": "neighbor"},
    ]
    result = classify_target_from_extended_rows([row], target_owner_id="target")
    assert result["label"] == "unresolved"
    assert result["refusal_reason"] == "target_owner_mixed_or_ambiguous_row"


def test_strict_owner_union_recomputes_from_match_evidence() -> None:
    trajectory = {
        "rows": [
            {"entity_matches": [{"status": "matched", "matched_entity_id": "a"}]},
            {"entity_matches": [{"status": "unmatched", "matched_entity_id": None}], "strict_matched_owner_ids": ["should-not-count"]},
            {"strict_matched_owner_ids": ["b"]},
        ]
    }
    assert strict_owner_union(trajectory) == {"a", "b"}


def test_sampled_owner_union_visits_each_sampled_trajectory_once() -> None:
    image = {
        "trajectories": [
            {"mode": "greedy", "rows": [{"strict_matched_owner_ids": ["greedy"]}]},
            {"mode": "sample", "seed": 11, "rows": [{"strict_matched_owner_ids": ["a", "b"]}]},
            {"mode": "sample", "seed": 12, "rows": [{"strict_matched_owner_ids": ["b", "c"]}]},
        ]
    }
    assert sampled_owner_union(image) == {"a", "b", "c"}


def test_frozen_sample_target_requires_clean_single_owner_row() -> None:
    image = {
        "prefix_evaluations": [{
            "prefix": {
                "prefix_token_ids_sha256": "prefix",
                "natural_actions": [{
                    "mode": "sample",
                    "seed": 11,
                    "row_index": 1,
                    "status": "success",
                    "row_stop": {"stop_reason": "complete_row"},
                    "strict_matched_owner_ids": ["target", "neighbor"],
                    "raw_generated_token_ids": [1, 2],
                }],
            },
        }]
    }
    with pytest.raises(StageOneValidationError, match="non-clean sampled row"):
        _find_sample_target(
            image,
            {
                "image_id": "1",
                "target_owner_id": "target",
                "sampled_seed": 11,
                "first_sampled_row_index_zero_based": 1,
                "target_parent_prefix_sha256": "prefix",
            },
        )


def test_frozen_inputs_reject_missing_discovery_shards(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 2,
        "images": [{"image_id": 1}],
    }
    targets = {
        "schema_version": 1,
        "unit_id": "2026-07-19-sampled-history-target-reachability-and-complete-row-value",
        "source_artifacts": {},
        "provisional_targets": [],
        "negative_discovery_controls": [],
        "root_decision_diagnostic": {},
    }
    manifest_path = tmp_path / "manifest.json"
    targets_path = tmp_path / "targets.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    targets_path.write_text(json.dumps(targets), encoding="utf-8")
    with pytest.raises(StageOneValidationError, match="frozen discovery manifest"):
        validate_frozen_inputs(manifest_path=manifest_path, provisional_targets_path=targets_path, shard_paths=[])


def _write_union_fixture(tmp_path: Path, *, device_per_image: bool = False) -> tuple[Path, list[Path]]:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"images": [{"image_id": i} for i in range(8)]}), encoding="utf-8")
    manifest_sha256 = sha256_file(manifest_path)
    shards: list[Path] = []
    for image_id in range(8):
        path = tmp_path / f"shard-{image_id}.json"
        path.write_text(json.dumps({
            "schema_version": "sampled_history_target_reachability.v1",
            "phase": "stage_one_extended_root_greedy_screen",
            "source_identity": {
                "stage_one_runner_sha256": "runner-sha",
                "reused_local_branch_helper_sha256": "helper-sha",
            },
            "frozen_inputs": {"manifest_sha256": manifest_sha256},
            "model_identity": {"model": "same"},
            "config": {"dtype": "fp32", "device": f"cuda:{image_id}" if device_per_image else "cuda:0"},
            "images": [{"image_id": image_id}],
        }), encoding="utf-8")
        shards.append(path)
    return manifest_path, shards

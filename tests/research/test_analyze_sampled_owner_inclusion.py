from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.analyze_sampled_owner_inclusion import (
    SCHEMA_VERSION,
    analyze_artifact_paths,
    parse_args,
    write_analysis,
)


def _prediction(
    trajectory_id: str, index: int, category: str, box: list[float]
) -> dict[str, object]:
    return {
        "description": category,
        "generated_order": index,
        "bbox": box,
        "object_span_id": f"{trajectory_id}:span-{index}",
    }


def _row(
    sample_index: int,
    predictions: list[tuple[str, list[float]]],
    *,
    stop_reason: str = "im_end",
    parse_status: str = "accepted",
    dropped_prediction_count: int = 0,
) -> dict[str, object]:
    trajectory_id = f"sample-{sample_index:02d}"
    parser: dict[str, object] = {
        "parse_status": parse_status,
        "predictions": [
            _prediction(trajectory_id, index, category, box)
            for index, (category, box) in enumerate(predictions)
        ],
    }
    if dropped_prediction_count:
        parser.update(
            {
                "dropped_prediction_count": dropped_prediction_count,
                "dropped_predictions": [
                    {"generated_order": 99, "reason": "geometry_invalid"}
                ],
            }
        )
    return {
        "image_id": "scene",
        "example_id": "example-scene",
        "trajectory_id": trajectory_id,
        "sample_index": sample_index,
        "decode_mode": "sampled",
        "stop_reason": stop_reason,
        "prompt_token_ids_sha256": "prompt-hash",
        "source_image_file_sha256": "source-image-hash",
        "executed_rgb_sha256": "executed-image-hash",
        "generated_token_ids_sha256": f"generated-{sample_index}",
        "predictions": parser,
    }


def _artifact(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "coordexp_vllm_trajectory_panel.v2",
        "experiment_mode": "experiment_local_vllm_trajectory_panel",
        "config": {
            "decode_mode": "sampled",
            "panel_mode": "sampled_only",
            "temperature": 0.4,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "max_new_tokens": 1024,
            "sample_count": 16,
            "sample_index_range": [0, 15],
            "resolved_fingerprint": "synthetic-config",
        },
        "model_identity": {"execution_model_identity": {"checkpoint": "synthetic"}},
        "prompt_metadata": {
            "example-scene": {
                "prompt_token_ids_sha256": "prompt-hash",
                "source_image_file_sha256": "source-image-hash",
                "width": 100,
                "height": 100,
            }
        },
        "rollout_count": len(rows),
        "rollouts": rows,
    }


def _write_fixture(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    owners: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pool = tmp_path / "candidate-pool.jsonl"
    pool.write_text(
        json.dumps(
            {
                "image_id": "scene",
                "width": 100,
                "height": 100,
                "objects": owners
                or [
                    {"owner_id": "stable", "category": "cat", "bbox": [0, 0, 10, 10]},
                    {"owner_id": "occasional", "category": "dog", "bbox": [20, 0, 30, 10]},
                    {"owner_id": "unseen", "category": "bird", "bbox": [30, 0, 40, 10]},
                    {"owner_id": "person-a", "category": "person", "bbox": [40, 0, 50, 10]},
                    {"owner_id": "person-b", "category": "person", "bbox": [40, 0, 50, 10]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "sampled-batch-00000.json"
    artifact.write_text(json.dumps(_artifact(rows)), encoding="utf-8")
    return artifact, pool


def _panel_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_index in range(16):
        predictions: list[tuple[str, list[float]]] = [("cat", [0, 0, 10, 10])]
        if sample_index == 0:
            predictions.append(("dog", [20, 0, 30, 10]))
        if sample_index == 1:
            predictions.append(("person", [40, 0, 50, 10]))
        if sample_index == 2:
            predictions.append(("boat", [60, 0, 70, 10]))
        if sample_index == 3:
            predictions.append(("cat", [0, 0, 10, 10]))
        rows.append(_row(sample_index, predictions))
    return rows


def _analyze(
    tmp_path: Path, rows: list[dict[str, object]], *, row_budget: int | None = None
) -> dict[str, object]:
    artifact, pool = _write_fixture(tmp_path, rows)
    return analyze_artifact_paths(
        [artifact],
        pool,
        require_full_panel=False,
        panel_root=tmp_path,
        row_budget=row_budget,
    )


def test_strata_keep_entity_support_separate_from_geometry_and_unresolved_rows(
    tmp_path: Path,
) -> None:
    result = _analyze(tmp_path, _panel_rows())
    owners = {record["owner_id"]: record for record in result["owner_records"]}

    stable = owners["scene:stable"]
    occasional = owners["scene:occasional"]
    unseen = owners["scene:unseen"]
    person_a = owners["scene:person-a"]
    assert stable["entity_support"] == {
        "sample_count": 16,
        "unambiguous_matched_trajectory_count": 16,
        "ambiguous_candidate_trajectory_count": 0,
        "q_lower": 1.0,
        "q_upper": 1.0,
        "stratum": "stable",
    }
    assert occasional["entity_support"]["stratum"] == "occasional"
    assert occasional["entity_support"]["q_lower"] == pytest.approx(1 / 16)
    assert occasional["entity_support"]["q_upper"] == pytest.approx(1 / 16)
    assert unseen["entity_support"]["stratum"] == "unseen"
    assert person_a["entity_support"] == {
        "sample_count": 16,
        "unambiguous_matched_trajectory_count": 0,
        "ambiguous_candidate_trajectory_count": 1,
        "q_lower": 0.0,
        "q_upper": 1 / 16,
        "stratum": "uncertain",
    }
    assert stable["geometry_evidence"]["unambiguous_iou_min"] == 1.0
    assert person_a["geometry_evidence"]["ambiguous_candidate_iou_max"] == 1.0
    assert result["summary"]["unresolved_prediction_count"] == 1
    assert result["summary"]["matching_policy"]["unmatched_predictions_are_unresolved_not_hallucinations"]


def test_ambiguity_only_changes_upper_bound_and_duplicate_stays_non_supporting(
    tmp_path: Path,
) -> None:
    result = _analyze(tmp_path, _panel_rows())
    occurrences = result["candidate_occurrences"]
    ambiguity = [
        item
        for item in occurrences
        if item["entity_evidence_kind"] == "ambiguous_candidate_owner"
    ]
    assert {item["owner_id"] for item in ambiguity} == {
        "scene:person-a",
        "scene:person-b",
    }
    assert all(item["sample_index"] == 1 for item in ambiguity)
    assert result["summary"]["duplicate_prediction_count"] == 1
    sample_two = next(
        item for item in result["trajectory_records"] if item["sample_index"] == 2
    )
    assert sample_two["unresolved_prediction_receipts"][0]["entity_status"] == "unresolved_pending_crop_review"
    assert "hallucination" not in sample_two["unresolved_prediction_receipts"][0]["entity_status"]


def test_malformed_drops_keep_valid_rows_and_are_preserved_as_parser_evidence(
    tmp_path: Path,
) -> None:
    rows = _panel_rows()
    rows[4] = _row(
        4,
        [("cat", [0, 0, 10, 10])],
        parse_status="accepted_with_drops",
        dropped_prediction_count=1,
    )
    result = _analyze(tmp_path, rows)
    record = next(item for item in result["trajectory_records"] if item["sample_index"] == 4)
    assert record["unambiguous_owner_ids"] == ["scene:stable"]
    assert record["parser_evidence"]["parse_status"] == "accepted_with_drops"
    assert record["parser_evidence"]["dropped_prediction_count"] == 1
    assert result["panel"]["parser_status_counts"]["accepted_with_drops"] == 1


def test_duplicate_image_sample_pair_and_non_natural_stop_fail_validation(tmp_path: Path) -> None:
    duplicate_rows = _panel_rows()
    duplicate_rows[-1] = _row(14, [("cat", [0, 0, 10, 10])])
    with pytest.raises(ValueError, match="duplicate image/sample_index pair"):
        _analyze(tmp_path / "duplicate", duplicate_rows)

    bad_stop_rows = _panel_rows()
    bad_stop_rows[0] = _row(0, [("cat", [0, 0, 10, 10])], stop_reason="length")
    with pytest.raises(ValueError, match="does not end with im_end"):
        _analyze(tmp_path / "length", bad_stop_rows)


def test_each_image_requires_all_sixteen_indices_and_full_mode_exact_counts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not exactly 0..15"):
        _analyze(tmp_path / "missing-index", _panel_rows()[:-1])

    artifact, pool = _write_fixture(tmp_path / "full-mode", _panel_rows())
    with pytest.raises(ValueError, match="expected 2432"):
        analyze_artifact_paths([artifact], pool, require_full_panel=True)


def test_row_budget_excludes_later_owner_but_full_mode_keeps_it(tmp_path: Path) -> None:
    rows = _panel_rows()
    rows[0] = _row(
        0,
        [("cat", [0, 0, 10, 10])] * 16 + [("bird", [30, 0, 40, 10])],
    )
    full = _analyze(tmp_path / "full", rows)
    bounded = _analyze(tmp_path / "bounded", rows, row_budget=16)
    full_bird = next(item for item in full["owner_records"] if item["owner_id"] == "scene:unseen")
    bounded_bird = next(
        item for item in bounded["owner_records"] if item["owner_id"] == "scene:unseen"
    )
    assert full_bird["entity_support"]["q_lower"] == pytest.approx(1 / 16)
    assert bounded_bird["entity_support"]["q_lower"] == 0.0
    assert bounded_bird["entity_support"]["stratum"] == "unseen"
    full_trajectory = next(item for item in full["trajectory_records"] if item["sample_index"] == 0)
    bounded_trajectory = next(
        item for item in bounded["trajectory_records"] if item["sample_index"] == 0
    )
    assert full_trajectory["complete_parsed_row_count"] == 17
    assert full_trajectory["evaluated_row_count"] == 17
    assert bounded_trajectory["complete_parsed_row_count"] == 17
    assert bounded_trajectory["evaluated_row_count"] == 16
    assert full["panel"]["row_budget"] is None
    assert bounded["panel"]["row_budget"] == bounded["summary"]["row_budget"] == 16


def test_invalid_row_budget_fails_api_and_cli() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        analyze_artifact_paths([], Path("."), row_budget=0)
    assert parse_args(["--output-dir", "output", "--row-budget", "16"]).row_budget == 16
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "output", "--row-budget", "0"])


def test_immutable_output_binds_compact_occurrences_without_token_arrays(tmp_path: Path) -> None:
    result = _analyze(tmp_path / "analysis", _panel_rows())
    output = tmp_path / "published"
    assert write_analysis(result, output) == output
    assert {path.name for path in output.iterdir()} == {
        "summary.json",
        "owner-support.jsonl",
        "trajectory-owners.jsonl",
        "candidate-occurrences.jsonl",
        "receipt.json",
    }
    occurrence = json.loads((output / "candidate-occurrences.jsonl").read_text().splitlines()[0])
    assert occurrence["schema_version"] == SCHEMA_VERSION
    assert {
        "image_id",
        "sample_index",
        "trajectory_id",
        "generated_row_index",
        "prediction_id",
        "owner_id",
        "intersection_over_union",
        "artifact_path",
        "artifact_sha256",
    } <= set(occurrence)
    assert "generated_token_ids" not in occurrence["rollout_identity"]
    with pytest.raises(FileExistsError, match="immutable output"):
        write_analysis(result, output)

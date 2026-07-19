"""Focused tests for matched native trajectory inventory."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research.inventory_matched_objective_native_coordinate_trajectories import (
    build_inventory,
)


POLICY = {
    "temperature": 0.4,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
}


def _pure_rollout(seed: int, *, include_dropped: bool = False) -> dict:
    predictions = [
        {
            "description": "person",
            "coord_bins": [100 + seed, 200, 300, 400],
            "bbox": [10 + seed, 20, 30 + seed, 40],
            "generated_order": 0,
            "object_span_id": f"pure:{seed}:accepted",
        }
    ]
    dropped = []
    if include_dropped:
        dropped.append(
            {
                "description": "person",
                "coord_bins": [1, 2, 3, 4],
                "bbox": [1, 2, 3, 4],
                "generated_order": 1,
                "object_span_id": f"pure:{seed}:dropped",
            }
        )
    return {
        "seed": seed,
        "stop_reason": "eos",
        "predictions": {
            "valid_prediction_count": len(predictions),
            "predictions": predictions,
            "dropped_prediction_count": len(dropped),
            "dropped_predictions": dropped,
        },
    }


def _request(seed: int) -> dict:
    return {
        "request_id": f"request-{seed}",
        "image_id": 1,
        "cell_index": seed,
        "sampling_seed": seed,
        "schedule_index": seed,
        "arm": {
            "arm_code": "FULL_BAG_K",
            "calls_per_image": 16,
            "history_policy": "fresh_base_prompt_per_call",
        },
    }


def _gaussian_bundle(request: dict, *, include_dropped: bool = False) -> dict:
    seed = int(request["sampling_seed"])
    receipts = [
        {
            "generated_row_index": 0,
            "normalized_category_name": "person",
            "coordinate_bins": [100 + seed, 200, 300, 400],
            "parsed_bbox_xyxy": [10 + seed, 20, 30 + seed, 40],
            "parse_status": "accepted",
            "prediction_validity": "accepted",
        }
    ]
    if include_dropped:
        receipts.append(
            {
                "generated_row_index": 1,
                "normalized_category_name": "person",
                "coordinate_bins": [1, 2, 3, 4],
                "parsed_bbox_xyxy": [1, 2, 3, 4],
                "parse_status": "dropped",
                "prediction_validity": "invalid",
                "drop_reason": "malformed",
            }
        )
    return {
        "schema_version": "terminal-output-bundle.v1",
        "request_id": request["request_id"],
        "scheduled_request": copy.deepcopy(request),
        "execution_evidence": {
            "image_id": 1,
            "sampling_seed": seed,
            "arm": {"arm_code": "FULL_BAG_K"},
        },
        "decode_result": {
            "execution_contract_anchor": {
                "decode_generation_policy": copy.deepcopy(POLICY),
            }
        },
        "call_diagnostics": {
            "valid_prediction_count": 1,
            "invalid_row_count": int(include_dropped),
            "malformed_row_count": int(include_dropped),
        },
        "parse_score_receipts": receipts,
        "stop_reason": "eos",
    }


def _write_fixture(root: Path) -> dict[str, Path]:
    cases_path = root / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "one-person",
                        "image_id": 1,
                        "category": "person",
                        "reference_object_identifier": "person-1",
                        "reference_box_xyxy": [10, 20, 30, 40],
                        "image_dimensions": {"width": 100, "height": 100},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    pure_root = root / "pure"
    pure_root.mkdir()
    pure_document = {
        "schema_version": "current_seeded_sampled_rollouts.v1",
        "config": {**POLICY, "seeds": list(range(16))},
        "rollout_count": 16,
        "rollouts": [_pure_rollout(seed, include_dropped=seed == 0) for seed in range(16)],
        "prompt_metadata": {
            "coco2017_val_000000000001": {"width": 100, "height": 100}
        },
    }
    (pure_root / "image-1.json").write_text(json.dumps(pure_document), encoding="utf-8")

    schedule_path = root / "schedule.json"
    schedule_path.write_text(
        json.dumps({"schedule": {"requests": [_request(seed) for seed in range(16)]}}),
        encoding="utf-8",
    )

    calls_root = root / "calls"
    for seed in range(16):
        bundle_path = calls_root / f"bundle-{seed}" / "terminal-output-bundle.json"
        bundle_path.parent.mkdir(parents=True)
        bundle_path.write_text(
            json.dumps(_gaussian_bundle(_request(seed), include_dropped=seed == 0)),
            encoding="utf-8",
        )
    return {
        "cases": cases_path,
        "pure": pure_root,
        "schedule": schedule_path,
        "calls": calls_root,
    }


def test_build_inventory_preserves_ordered_trajectories_and_dropped_candidates(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    result = build_inventory(
        cases_path=paths["cases"],
        pure_artifact_root=paths["pure"],
        gaussian_schedule_path=paths["schedule"],
        gaussian_calls_root=paths["calls"],
    )

    case = result["cases"][0]
    assert result["schema_version"].endswith(".v1")
    assert case["denominators"]["trajectory_count_per_source"] == 16
    assert [item["seed"] for item in case["trajectories"]["pure_cross_entropy"]] == list(range(16))
    assert [item["cell_index"] for item in case["trajectories"]["gaussian_ranked_probability_score"]] == list(range(16))
    assert case["source_totals"]["pure_cross_entropy"]["relevant_category_valid"] == 16
    assert case["source_totals"]["pure_cross_entropy"]["relevant_category_dropped"] == 1
    assert case["source_totals"]["gaussian_ranked_probability_score"]["relevant_category_dropped"] == 1
    candidate = case["trajectories"]["pure_cross_entropy"][0]["predictions"][0]
    assert candidate["reference_iou"] == pytest.approx(1.0)
    assert candidate["per_boundary_bin_error"] == [0, 0, 0, 0]
    assert candidate["per_boundary_bin_absolute_error"] == [0, 0, 0, 0]
    assert "raw_candidate" not in candidate
    assert case["candidate_pair_review_queue_total"] == 256
    assert len(case["candidate_pair_review_queue"]) == 256
    assert all(item["admission"] == "non_admitting_review_queue_only" for item in case["candidate_pair_review_queue"])


def test_inventory_rejects_mismatched_pure_sampling_policy(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    pure_path = paths["pure"] / "image-1.json"
    document = json.loads(pure_path.read_text(encoding="utf-8"))
    document["config"]["max_new_tokens"] = 256
    pure_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="max_new_tokens"):
        build_inventory(
            cases_path=paths["cases"],
            pure_artifact_root=paths["pure"],
            gaussian_schedule_path=paths["schedule"],
            gaussian_calls_root=paths["calls"],
        )

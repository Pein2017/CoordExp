from __future__ import annotations

import copy

import pytest

from probes.training_set_completion.paired_evaluation import (
    _ledger_image,
    _matchable_rows_with_geometry_debt,
    _outside_coco80,
    _physical_statuses,
    compare,
    score_readback,
)


def _target(owner: str, box: list[int], description: str = "person") -> dict:
    return {
        "image_id": 1,
        "owner_id": owner,
        "reference_coord_bins_1000": box,
        "description": description,
        "class_status": "verified_coco80",
    }


def _prediction(name: str, box: list[int], description: str, span: str) -> dict:
    return {
        "prediction_id": name,
        "generated_order": int(name[1:]),
        "coord_bins_1000": box,
        "description": description,
        "raw_span_sha256": span,
    }


def test_annotation_unmatched_stays_separate_from_physical_false_positive() -> None:
    targets = [_target("owner-a", [0, 0, 10, 10]), _target("owner-b", [20, 20, 30, 30])]
    predictions = [_prediction("p0", [0, 0, 10, 10], "figurine", "known-false"), _prediction("p1", [40, 40, 50, 50], "person", "new")]
    ledger = _ledger_image(targets, predictions, threshold=0.5)
    physical = _physical_statuses(
        image_id=1,
        predictions=predictions,
        current_matches=[{"prediction_id": "p0", "reference_owner_id": "owner-a", "iou": 1.0}],
        current_targets=targets,
        exact_reviews={(1, "known-false"): {"physical_status": "false"}},
    )

    assert ledger["matched_count"] == 1
    assert ledger["missing_owner_ids"] == ["owner-b"]
    assert ledger["annotation_unmatched_prediction_ids"] == ["p1"]
    assert physical["matched_current_known_owner"] == 1
    assert physical["confirmed_fp"] == 0
    assert physical["physical_unknown"] == 1
    assert _outside_coco80(predictions) == [
        {"prediction_id": "p0", "generated_order": 0, "description": "figurine"}
    ]


def test_prior_repeat_is_recomputed_from_current_generated_order() -> None:
    prior_repeat = {
        (1, "repeat-span"): {
            "physical_status": "repeat",
            "owner_id": "one-owner",
            "class": "correct",
            "extent": "same",
        }
    }
    alone = _physical_statuses(
        image_id=1,
        predictions=[_prediction("p0", [0, 0, 10, 10], "person", "repeat-span")],
        current_matches=[],
        current_targets=[],
        exact_reviews=prior_repeat,
    )
    assert alone["reviewed_physical_repeat"] == 0
    assert alone["rows"][0]["physical_status"] == "exact_prior_reviewed_owner"

    paired = _physical_statuses(
        image_id=1,
        predictions=[
            _prediction("p0", [0, 0, 10, 10], "person", "repeat-span"),
            _prediction("p1", [0, 0, 10, 10], "person", "repeat-span"),
        ],
        current_matches=[],
        current_targets=[],
        exact_reviews=prior_repeat,
    )
    assert paired["reviewed_physical_repeat"] == 1
    assert paired["rows"][0]["physical_status"] == "exact_prior_reviewed_owner"
    assert paired["rows"][1]["physical_status"] == "recomputed_repeat_of_accepted_owner"


def test_compare_reports_retention_without_composite() -> None:
    def result(covered: list[str]) -> dict:
        return {
            "schema": "training_set_completion.dual_start_paired_evaluation.v1",
            "sources": {
                "preparation": {"path": "preparation.json", "sha256": "p", "size_bytes": 1},
                "readback_admission": {"readback": {"path": "readback.json", "sha256": "a", "size_bytes": 1}},
            },
            "matching": {"primary": "IoU >= 0.5"},
            "metric_contract": {"primary": "scoped218 FN count"},
            "per_image": [
                {
                    "image_id": 1,
                    "ledgers_iou_0_5": {
                        name: {"covered_owner_ids": covered}
                        for name in ("scoped218", "historical232", "current-known248")
                    },
                }
            ],
            "aggregate": {
                "ledgers_iou_0_5": {
                    name: {"covered_owner_ids": covered}
                    for name in ("scoped218", "historical232", "current-known248")
                }
            },
        }

    compared = compare(baseline=result(["a", "b"]), endpoint=result(["b", "c"]), label="A-endpoint")
    assert compared["per_ledger"]["scoped218"] == {
        "retained": [{"image_id": 1, "owner_id": "b"}],
        "gained": [{"image_id": 1, "owner_id": "c"}],
        "lost": [{"image_id": 1, "owner_id": "a"}],
        "retained_count": 1,
        "gained_count": 1,
        "lost_count": 1,
    }
    assert "No scalar composite" in compared["disposition"]


def test_compare_rejects_mismatched_metric_contract() -> None:
    base = {
        "schema": "training_set_completion.dual_start_paired_evaluation.v1",
        "sources": {
            "preparation": {"path": "preparation.json", "sha256": "p", "size_bytes": 1},
            "readback_admission": {},
        },
        "matching": {"primary": "IoU >= 0.5"},
        "metric_contract": {"primary": "scoped218 FN count"},
        "per_image": [
            {
                "image_id": 1,
                "ledgers_iou_0_5": {
                    name: {"covered_owner_ids": []}
                    for name in ("scoped218", "historical232", "current-known248")
                },
            }
        ],
    }
    endpoint = copy.deepcopy(base)
    endpoint["metric_contract"] = {"primary": "different"}
    with pytest.raises(ValueError, match="metric contract"):
        compare(baseline=base, endpoint=endpoint, label="bad")


def test_direct_readback_cannot_bypass_source_admission(tmp_path) -> None:
    preparation = tmp_path / "preparation.json"
    preparation.write_text("{}")
    with pytest.raises(ValueError, match="explicit legacy admission"):
        score_readback(
            preparation_path=preparation,
            readback_path=tmp_path / "unadmitted-readback.json",
            label="unadmitted",
        )


def test_nominal_parser_row_with_bad_geometry_remains_raw_debt() -> None:
    matchable, dropped = _matchable_rows_with_geometry_debt(
        {
            "pred": [
                {
                    "generated_order": 0,
                    "description": "person",
                    "coord_bins": [10, 20, 10, 30],
                    "raw_span_sha256": "bad-geometry",
                }
            ],
            "dropped_predictions": [],
        }
    )
    assert matchable == []
    assert dropped[0]["drop_reason"] == "consumer_geometry_invalid_after_native_parse"

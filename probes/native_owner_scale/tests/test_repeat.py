from __future__ import annotations

import json
from pathlib import Path

import pytest

from probes.native_owner_scale.repeat import (
    CAP,
    DUPLICATE_IOU_THRESHOLD,
    DEFAULT_SOURCE,
    _assert_boundary_invariant,
    _record_census,
    _span_lineage,
    is_strict_repeat,
    project_raw128,
    run_census,
)


def _parsed(pred, *, gt=None):
    return {
        "parse_status": "accepted",
        "image_width": 100,
        "image_height": 100,
        "pred": pred,
        "gt": gt or [],
        "dropped_predictions": [],
    }


def _row(example_id, parsed, *, stop="im_end"):
    ids = [1, 151645] if stop == "im_end" else list(range(CAP))
    return {
        "example_id": example_id,
        "image_id": int(example_id),
        "split": "test",
        "image_path": "/tmp/not-used.jpg",
        "image_content_sha256": None,
        "image_width": 100,
        "image_height": 100,
        "parsed": parsed,
        "token_ids": ids,
        "token_ids_sha256": "x",
        "stop_reason": stop,
        "source_record_digest": "record",
        "declared_score": {},
    }


def _obj(category, box):
    return {"description": category, "bbox": list(box)}


def test_threshold_is_exclusive_and_boundary_receipt_has_teeth():
    assert not is_strict_repeat(DUPLICATE_IOU_THRESHOLD)
    assert _assert_boundary_invariant()["above_is_repeat"]
    assert not _assert_boundary_invariant()["equal_is_repeat"]


def test_class_blind_later_row_counted_once_even_with_many_prior_matches():
    parsed = _parsed([
        _obj("person", (0, 0, 10, 10)),
        _obj("chair", (0, 0, 10, 10)),
        _obj("chair", (0, 0, 10, 10)),
    ])
    result = _record_census(_row("1", parsed))
    rows = result["strict_repeat_rows"]
    assert len(rows) == 2
    assert [r["later_pred_index"] for r in rows] == [1, 2]
    assert rows[0]["seed_pred_index"] == 0
    assert rows[1]["seed_pred_index"] == 0


def test_subthreshold_same_category_drift_is_held_not_negative():
    parsed = _parsed([
        _obj("person", (0, 0, 20, 20)),
        _obj("person", (0.7, 0, 20.7, 20)),
    ])
    result = _record_census(_row("2", parsed))
    assert result["strict_repeat_rows"] == []
    assert len(result["subthreshold_candidates"]) == 1
    candidate = result["subthreshold_candidates"][0]
    assert candidate["best_iou"] < DUPLICATE_IOU_THRESHOLD
    assert candidate["negative_authorized"] is False
    assert candidate["review_status"] == "unknown_pending_visual_review"


def test_supported_seed_is_annotation_relative_and_unmatched_is_unknown():
    parsed = _parsed(
        [_obj("person", (0, 0, 20, 20)), _obj("person", (0.7, 0, 20.7, 20))],
        gt=[_obj("person", (0, 0, 20, 20))],
    )
    result = _record_census(_row("3", parsed))
    candidate = result["subthreshold_candidates"][0]
    assert candidate["seed_support"] == "supported"
    assert candidate["later_support"] == "supported"
    # An unmatched seed is never silently relabeled as hallucination/negative.
    parsed_unknown = _parsed([_obj("person", (50, 50, 60, 60)), _obj("person", (50.7, 50, 60.7, 60))])
    unknown = _record_census(_row("4", parsed_unknown))["subthreshold_candidates"][0]
    assert unknown["seed_support"] == "unknown_or_unmatched"
    assert unknown["negative_authorized"] is False


def test_literal_span_lineage_reconstructs_generated_prefix_boundary():
    record = {
        "parsed": {
            "raw_decode_text": "abcDEF",
            "pred": [{
                "description": "person",
                "char_start": 3,
                "char_end": 6,
                "raw_span_text": "DEF",
                "raw_span_sha256": None,
                "generated_order": 1,
            }],
        },
        "token_ids": [10, 11, 12],
    }
    result = _span_lineage(
        record,
        0,
        tokenizer=object(),
        tokenizer_ids=[10, 11, 12],
        token_offsets=[(0, 1), (1, 3), (3, 6)],
        prompt_token_ids=[90, 91],
    )
    assert result["token_span_status"] == "exact"
    assert result["token_start"] == 2
    assert result["token_end_exclusive"] == 3
    assert result["deployment_prefix"]["generated_tokens_before_row"] == 2
    assert result["deployment_prefix"]["combined_prefix_token_count"] == 4


def test_final_selection_projection_reuses_frozen_cards_without_new_units(tmp_path):
    source = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-09-12-native-owner-scale-and-state/scale/preparation/"
        "selection-v2-remainder.json"
    )
    census = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-09-12-native-owner-scale-and-state/repeat/census.json"
    )
    review = census.parent / "review-manifest.json"
    if not (source.is_file() and census.is_file() and review.is_file()):
        pytest.skip("final selection projection artifacts are not mounted")
    result = project_raw128(
        source,
        census,
        review_manifest_path=review,
        output_path=tmp_path / "projection.json",
    )
    assert result["projection_kind"] == "existing_source384_census_filtered_by_A_selection_v2"
    assert result["counts"]["source_records"] == 128
    assert result["counts"]["strict_repeat_rows"] == 582
    assert result["frozen_review_selection"]["reused_without_change"] is True
    assert result["discarded_internal_review_selection"]["published"] is False


@pytest.mark.parametrize("expected", [582])
def test_frozen_exposed384_source_consumer(expected):
    if not DEFAULT_SOURCE.is_file():
        pytest.skip("frozen exposed384 source is not mounted")
    census = run_census(DEFAULT_SOURCE, universe_path=DEFAULT_SOURCE)
    assert census["original_source_population"] == 384
    assert census["counts"]["source_records"] == 384
    assert census["counts"]["strict_repeat_rows"] == expected
    assert census["counts"]["cap"] == 4
    assert census["counts"]["parser_drop_rows"] == 792
    assert census["counts"]["invalid_geometry_rows"] == 788
    assert census["counts"]["malformed_rows"] == 4

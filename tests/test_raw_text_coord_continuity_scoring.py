from __future__ import annotations

import torch

from src.analysis.unmatched_proposal_verifier import PreparedExample
from src.analysis.raw_text_coord_continuity_scoring import (
    lexical_features_for_candidate,
    replace_bbox_slot_value,
    score_span_logprobs,
)


def test_score_span_logprobs_supports_multi_token_chunk() -> None:
    logits = torch.full((1, 5, 16), -20.0)
    input_ids = torch.tensor([[1, 3, 4, 5, 2]])
    logits[0, 0, 3] = 5.0
    logits[0, 1, 4] = 4.0
    logits[0, 2, 5] = 3.0

    result = score_span_logprobs(
        logits=logits,
        input_ids=input_ids,
        batch_idx=0,
        positions=[1, 2, 3],
    )

    assert result["count"] == 3
    assert result["mean_logprob"] > -0.1
    assert result["sum_logprob"] > -0.3


def test_score_span_logprobs_rejects_nonpositive_positions() -> None:
    logits = torch.zeros((1, 2, 4))
    input_ids = torch.tensor([[0, 1]])

    try:
        score_span_logprobs(
            logits=logits,
            input_ids=input_ids,
            batch_idx=0,
            positions=[0],
        )
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("expected ValueError for position 0")


def test_prepared_example_accepts_legacy_four_field_shape() -> None:
    prepared = PreparedExample(
        full_text="demo",
        assistant_text="demo",
        desc_positions=[1],
        full_input_ids=[1, 2, 3],
    )

    assert prepared.assistant_start == 0
    assert prepared.assistant_input_ids == []


def test_replace_bbox_slot_value_preserves_json_boundaries() -> None:
    assistant_text = '[{"desc":"book","bbox_2d":[199,200,210,250]}]'
    replaced = replace_bbox_slot_value(
        assistant_text=assistant_text,
        slot="x1",
        original_bbox=(199, 200, 210, 250),
        candidate_value=231,
    )

    assert replaced == '[{"desc":"book","bbox_2d":[231,200,210,250]}]'


def test_replace_bbox_slot_value_can_target_later_duplicate_bbox() -> None:
    assistant_text = '[{"desc":"a","bbox_2d":[1,2,3,4]},{"desc":"b","bbox_2d":[1,2,3,4]}]'
    replaced = replace_bbox_slot_value(
        assistant_text=assistant_text,
        slot="x1",
        original_bbox=(1, 2, 3, 4),
        candidate_value=9,
        object_index=1,
    )

    assert replaced == '[{"desc":"a","bbox_2d":[1,2,3,4]},{"desc":"b","bbox_2d":[9,2,3,4]}]'


def test_lexical_features_capture_numeric_and_token_shape() -> None:
    features = lexical_features_for_candidate(
        candidate_value=210,
        center_value=199,
        gt_value=200,
        tokenizer_tokens=["2", "1", "0"],
        center_tokens=["1", "9", "9"],
    )

    assert features["numeric_distance_to_center"] == 11
    assert features["numeric_distance_to_gt"] == 10
    assert features["digit_length_match"] == 1
    assert features["token_count"] == 3


def test_lexical_features_use_true_character_edit_distance() -> None:
    features = lexical_features_for_candidate(
        candidate_value=121,
        center_value=212,
        gt_value=121,
        tokenizer_tokens=["1", "2", "1"],
        center_tokens=["2", "1", "2"],
    )

    assert features["char_edit_distance"] == 2

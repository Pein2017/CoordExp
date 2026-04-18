from __future__ import annotations

import torch

from src.analysis.unmatched_proposal_verifier import PreparedExample
from src.analysis.raw_text_coord_continuity_scoring import score_span_logprobs


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

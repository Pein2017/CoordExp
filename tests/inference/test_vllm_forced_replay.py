from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams

from src.inference.vllm_forced_replay import (
    EXPECTED_TOKEN_IDS_KEY,
    CoordExpForcedSequenceLogitsProcessor,
)


def _processor() -> CoordExpForcedSequenceLogitsProcessor:
    return CoordExpForcedSequenceLogitsProcessor(
        SimpleNamespace(),
        torch.device("cpu"),
        False,
    )


def test_forced_sequence_selects_the_expected_token_for_each_decode_step() -> None:
    params = SamplingParams(
        extra_args={EXPECTED_TOKEN_IDS_KEY: [2, 4]},
    )
    per_request = _processor().new_req_logits_processor(params)
    assert per_request is not None

    first = per_request([], torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0]))
    second = per_request([2], torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0]))

    assert torch.isneginf(first[[0, 1, 3, 4]]).all()
    assert first[2].item() == 0.0
    assert torch.isneginf(second[[0, 1, 2, 3]]).all()
    assert second[4].item() == 0.0


@pytest.mark.parametrize(
    "value",
    [[], [True], [-1], [1.0], "1"],
)
def test_forced_sequence_rejects_invalid_expected_tokens(value: object) -> None:
    params = SamplingParams(extra_args={EXPECTED_TOKEN_IDS_KEY: value})

    with pytest.raises(ValueError, match=EXPECTED_TOKEN_IDS_KEY):
        CoordExpForcedSequenceLogitsProcessor.validate_params(params)


def test_forced_sequence_rejects_steps_past_the_expected_continuation() -> None:
    params = SamplingParams(extra_args={EXPECTED_TOKEN_IDS_KEY: [2]})
    per_request = _processor().new_req_logits_processor(params)
    assert per_request is not None

    with pytest.raises(RuntimeError, match="advanced beyond"):
        per_request([2], torch.zeros(5))

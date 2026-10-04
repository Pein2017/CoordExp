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


def test_current_sampler_preserves_raw_logprobs_before_real_forcing() -> None:
    from src.inference.vllm_forced_replay import verify_raw_logprob_semantics

    evidence = verify_raw_logprob_semantics()
    assert evidence["status"] == "verified_cpu_sampler_ordering"
    assert evidence["forced_token_id"] == 2
    assert evidence["raw_chosen_logprob"] < -1.0
    assert evidence["processed_chosen_logprob"] == 0.0
    assert evidence["execution_qualification"] == "not_established"


def test_forced_sequence_rejects_token_outside_logits_vocabulary() -> None:
    params = SamplingParams(extra_args={EXPECTED_TOKEN_IDS_KEY: [5]})
    per_request = _processor().new_req_logits_processor(params)
    assert per_request is not None
    with pytest.raises(RuntimeError, match="outside the logits vocabulary"):
        per_request([], torch.zeros(5))


@pytest.mark.parametrize("counterexample", ["reversed_order", "alias_raw_logits"])
def test_raw_sampler_witness_rejects_wrong_order_or_raw_alias(
    monkeypatch: pytest.MonkeyPatch, counterexample: str,
) -> None:
    from vllm.v1.sample.sampler import Sampler
    from src.common.errors import RuntimeContractError
    from src.inference.vllm_forced_replay import verify_raw_logprob_semantics

    if counterexample == "reversed_order":
        original = Sampler.forward
        def reversed_order(self, logits, sampling_metadata, *args, **kwargs):
            for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
                logits = processor.apply(logits)
            return original(self, logits, sampling_metadata, *args, **kwargs)
        monkeypatch.setattr(Sampler, "forward", reversed_order)
    else:
        monkeypatch.setattr(Sampler, "compute_logprobs", staticmethod(lambda logits: logits))
    with pytest.raises(RuntimeContractError) as exc_info:
        verify_raw_logprob_semantics()
    assert exc_info.value.code == "vllm_backend.raw_semantics_unverified"


def test_installed_sampler_rejects_out_of_vocabulary_forced_token() -> None:
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.logits_processor.interface import BatchUpdate
    from vllm.v1.sample.sampler import Sampler

    processor = _processor()
    params = SamplingParams(extra_args={EXPECTED_TOKEN_IDS_KEY: [5]})
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=[(0, params, [], [])], moved=[]))
    metadata = SimpleNamespace(
        bad_words_token_ids={}, no_penalties=True, output_token_ids=[[]],
        allowed_token_ids_mask=None, logitsprocs=LogitsProcessors([processor]),
        thinking_budget_state_holder=None,
    )
    with pytest.raises(RuntimeError, match="outside the logits vocabulary"):
        Sampler().apply_logits_processors(torch.zeros(1, 5), metadata, False)

"""vLLM 0.14.1 logits processor for exact decode-time likelihood replay."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from vllm import SamplingParams
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


EXPECTED_TOKEN_IDS_KEY = "coordexp_expected_token_ids"


class CoordExpForcedSequenceLogitsProcessor(AdapterLogitsProcessor):
    """Force one pre-recorded token per decode step for raw-logprob replay.

    vLLM computes ``raw_logprobs`` before applying non-argmax-invariant logits
    processors. This processor therefore controls the sampled continuation
    without changing the raw chosen-token likelihood returned by the engine.
    """

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        value = params.extra_args and params.extra_args.get(EXPECTED_TOKEN_IDS_KEY)
        if value is None:
            return
        _validate_expected_token_ids(value)

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        value = params.extra_args and params.extra_args.get(EXPECTED_TOKEN_IDS_KEY)
        if value is None:
            return None
        expected = _validate_expected_token_ids(value)
        return _ForcedSequence(expected)


class _ForcedSequence:
    def __init__(self, expected_token_ids: Sequence[int]) -> None:
        self._expected_token_ids = tuple(expected_token_ids)

    def __call__(
        self,
        output_token_ids: Sequence[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        step = len(output_token_ids)
        if step >= len(self._expected_token_ids):
            raise RuntimeError(
                "vLLM forced replay advanced beyond the expected continuation"
            )
        token_id = self._expected_token_ids[step]
        if token_id >= logits.shape[-1]:
            raise RuntimeError(
                "vLLM forced replay token id is outside the logits vocabulary"
            )
        logits.fill_(-torch.inf)
        logits[token_id] = 0.0
        return logits


def _validate_expected_token_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            f"{EXPECTED_TOKEN_IDS_KEY} must be a nonempty integer sequence"
        )
    token_ids = tuple(value)
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError(
            f"{EXPECTED_TOKEN_IDS_KEY} must contain nonnegative integers"
        )
    return token_ids

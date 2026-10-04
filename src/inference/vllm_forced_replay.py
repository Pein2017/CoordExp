"""vLLM 0.29.0 logits processor for exact decode-time likelihood replay."""

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


def verify_raw_logprob_semantics() -> dict[str, Any]:
    """Check installed CPU sampler ordering with this real forcing processor.

    This witnesses the raw/processed sampling seam, not native model execution.
    """
    from importlib import metadata
    from types import SimpleNamespace
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.logits_processor.interface import BatchUpdate
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.sampler import Sampler
    from src.common.errors import RuntimeContractError
    from src.config.inference import inspect_vllm_runtime_version

    version = inspect_vllm_runtime_version(observed_version=metadata.version("vllm"))
    if version["status"] != "supported":
        raise RuntimeContractError("raw sampler API version is unsupported", code="vllm_backend.raw_semantics_unverified", context=version)
    logits = torch.tensor([[3.0, 1.0, -2.0, 0.5]], dtype=torch.float32)
    forced_token_id = 2
    expected = logits.log_softmax(dim=-1)
    observed = {}
    for mode in ("raw_logprobs", "processed_logprobs"):
        processor = CoordExpForcedSequenceLogitsProcessor(SimpleNamespace(), torch.device("cpu"), False)
        params = SamplingParams(extra_args={EXPECTED_TOKEN_IDS_KEY: [forced_token_id]})
        processor.validate_params(params)
        processor.update_state(BatchUpdate(batch_size=1, removed=[], added=[(0, params, [], [])], moved=[]))
        empty = torch.empty(0)
        sampling = SamplingMetadata(
            temperature=None, all_greedy=True, all_random=False, top_p=None, top_k=None,
            generators={}, max_num_logprobs=-1, no_penalties=True, prompt_token_ids=None,
            frequency_penalties=empty, presence_penalties=empty, repetition_penalties=empty,
            output_token_ids=[[]], allowed_token_ids_mask=None, bad_words_token_ids={},
            logitsprocs=LogitsProcessors([processor]),
        )
        output = Sampler(logprobs_mode=mode)(logits.clone(), sampling)
        if output.sampled_token_ids.item() != forced_token_id or output.logprobs_tensors is None:
            raise RuntimeContractError("installed sampler did not retain forced token", code="vllm_backend.raw_semantics_unverified")
        observed[mode] = output.logprobs_tensors.logprobs
    if not torch.allclose(observed["raw_logprobs"], expected, atol=1e-6, rtol=0) or observed["processed_logprobs"][0, forced_token_id].item() != 0.0:
        raise RuntimeContractError("installed sampler raw/processed ordering differs", code="vllm_backend.raw_semantics_unverified")
    return {
        "status": "verified_cpu_sampler_ordering",
        "version": version,
        "sampler": "vllm.v1.sample.sampler.Sampler.forward",
        "forced_token_id": forced_token_id,
        "raw_chosen_logprob": observed["raw_logprobs"][0, forced_token_id].item(),
        "processed_chosen_logprob": observed["processed_logprobs"][0, forced_token_id].item(),
        "execution_qualification": "not_established",
    }

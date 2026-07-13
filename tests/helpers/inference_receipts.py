"""Canonical decode-result fixtures backed by production receipt builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.inference.backend import (
    DecodeGenerationPolicy,
    DecodeRequest,
    DecodeResult,
    TokenTrace,
    build_decode_execution_receipt,
    effective_generation_arguments,
)


def build_greedy_decode_result(
    *,
    request_id: str,
    token_trace: Sequence[TokenTrace],
    raw_generated_text: str,
    parser_text: str | None = None,
    prompt_token_ids: Sequence[int] = (11, 12),
    model_identity: Mapping[str, object] | None = None,
    tokenizer_identity: Mapping[str, object] | None = None,
    generation_config_fingerprint: str = "gen-fp",
    max_new_tokens: int = 64,
    repetition_penalty: float = 1.0,
    stop_reason: str = "length",
    strip_policy: str = "none",
    request_execution_index: int = 0,
    batch_request_order_fingerprint: str | None = None,
    decode_request: DecodeRequest | None = None,
) -> DecodeResult:
    """Build a greedy result whose receipt is bound by the production contract."""

    prompt_ids = [int(token_id) for token_id in prompt_token_ids]
    traces = list(token_trace)
    generated_ids = [int(trace.token_id) for trace in traces]
    model_id = dict(model_identity or {"family": "unit"})
    tokenizer_id = dict(tokenizer_identity or {"sha256": "tok"})
    request = decode_request or DecodeRequest(
        request_id=request_id,
        prompt_token_ids=prompt_ids,
        model_inputs={},
        generation_policy=DecodeGenerationPolicy.greedy(
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        ),
        sampling_seed=None,
    )
    receipt = build_decode_execution_receipt(
        request=request,
        generated_token_ids=generated_ids,
        token_trace=traces,
        stop_reason=stop_reason,
        model_identity=model_id,
        tokenizer_identity=tokenizer_id,
        generation_config_fingerprint=generation_config_fingerprint,
        executed_generation_arguments=effective_generation_arguments(
            request.generation_policy,
            eos_token_id=151645,
            pad_token_id=0,
        ),
        request_execution_index=request_execution_index,
        batch_request_order_fingerprint=batch_request_order_fingerprint,
        runtime_identity={"runtime": "inference-fixture"},
    )
    return DecodeResult(
        request_id=request_id,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        prompt_token_ids=prompt_ids,
        generated_token_ids=generated_ids,
        raw_generated_text=raw_generated_text,
        parser_text=raw_generated_text if parser_text is None else parser_text,
        strip_policy=strip_policy,
        stop_reason=stop_reason,
        model_identity=model_id,
        tokenizer_identity=tokenizer_id,
        generation_config_fingerprint=generation_config_fingerprint,
        token_trace=traces,
        execution_receipt=receipt,
    )

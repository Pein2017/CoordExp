from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..rollout_matching.parsing import decode_pieces as _decode_pieces


def rollout_many_vllm_colocate(
    *,
    owner: Any,
    samples: Sequence[Mapping[str, Any]],
    logger: Any,
    with_logprobs: bool = False,
    request_index_offset: int = 0,
    decode_override: Optional[Mapping[str, Any]] = None,
) -> List[Any]:
    decode_request = owner._resolve_rollout_decode_request(
        decode_override=decode_override
    )
    decode_mode = str(decode_request.decode_mode)
    if decode_mode == "beam":
        raise ValueError(
            "vLLM rollout backend does not support decode_mode=beam; "
            "use greedy or sampling overrides instead"
        )

    max_new_tokens = int(decode_request.max_new_tokens)
    temperature = float(decode_request.temperature)
    top_p = float(decode_request.top_p)
    top_k = int(decode_request.top_k)
    repetition_penalty = float(decode_request.repetition_penalty)

    if with_logprobs and float(temperature) > 0.0:
        raise ValueError(
            "eval-step confidence scoring requires decoding.temperature=0.0 "
            f"(greedy), got {float(temperature)}"
        )

    try:
        from swift.llm import InferRequest, RequestConfig
    except (ImportError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "swift.llm.RequestConfig and InferRequest are required for vLLM rollouts"
        ) from exc

    request_kwargs = owner._rollout_vllm_request_config_kwargs(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    gs = int(getattr(getattr(owner, "state", None), "global_step", 0) or 0)
    seed_base = int(owner._derive_rollout_seed_base(global_step=gs))
    request_index_offset_i = max(0, int(request_index_offset))
    request_kwargs["seed"] = int(
        owner._normalize_rollout_seed_int32(int(seed_base + request_index_offset_i))
    )
    if with_logprobs:
        request_kwargs["logprobs"] = True
    request_config = RequestConfig(**request_kwargs)

    infer_requests: List[Any] = []
    for s in samples:
        msgs = s.get("messages")
        if not isinstance(msgs, list):
            raise ValueError("rollout-matching samples must contain messages (list)")
        infer_requests.append(InferRequest(messages=msgs))

    offload_cm = (
        nullcontext()
        if bool(getattr(owner, "_eval_vllm_window_active", False))
        else owner._maybe_rollout_offload_context(rollout_backend="vllm")
    )
    with offload_cm:
        owner._sync_vllm_rollout_model_if_needed()
        outs: List[Any] = owner._vllm_infer_tp_group(infer_requests, request_config)

    if len(outs) != len(infer_requests):
        raise RuntimeError("vLLM returned unexpected number of outputs")

    results: List[Any] = []
    for out_idx, out in enumerate(outs):
        if isinstance(out, Exception):
            raise RuntimeError(
                "vLLM decode failed for a rollout sample "
                f"(sample_idx={int(out_idx)}): {out!r}"
            ) from out

        text = ""
        token_ids: List[int] = []
        prompt_ids: List[int] = []
        choice_logprobs = None
        try:
            text = str(out.choices[0].message.content or "")
            token_ids = [int(t) for t in (out.choices[0].token_ids or [])]
            prompt_ids = [int(t) for t in (getattr(out, "prompt_token_ids", None) or [])]
            prompt_ids = owner.__class__._strip_left_padding_token_ids(
                prompt_ids,
                pad_token_id=getattr(owner.tokenizer, "pad_token_id", None),
            )
            choice_logprobs = getattr(out.choices[0], "logprobs", None)
        except (TypeError, ValueError):
            text = ""
            token_ids = []
            prompt_ids = []
            choice_logprobs = None

        if with_logprobs:
            token_logprobs, generated_token_text = (
                owner.__class__._extract_swift_choice_logprobs(choice_logprobs)
            )
            if len(token_logprobs) != len(generated_token_text):
                trace_pair_len = min(len(token_logprobs), len(generated_token_text))
                logger.warning(
                    "vLLM rollout trace payload length mismatch; clamping to common length. "
                    "sample_idx=%s token_ids=%s logprobs=%s text=%s",
                    int(out_idx),
                    int(len(token_ids)),
                    int(len(token_logprobs)),
                    int(len(generated_token_text)),
                )
                token_logprobs = token_logprobs[:trace_pair_len]
                generated_token_text = generated_token_text[:trace_pair_len]
            if len(token_logprobs) > len(token_ids):
                excess_trace = int(len(token_logprobs) - len(token_ids))
                logger.debug(
                    "vLLM rollout trace longer than token_ids; dropping trailing trace entries. "
                    "sample_idx=%s token_ids=%s logprobs=%s text=%s excess=%s",
                    int(out_idx),
                    int(len(token_ids)),
                    int(len(token_logprobs)),
                    int(len(generated_token_text)),
                    int(excess_trace),
                )
                token_logprobs = token_logprobs[: len(token_ids)]
                generated_token_text = generated_token_text[: len(token_ids)]
            elif len(token_logprobs) < len(token_ids):
                logger.warning(
                    "vLLM rollout trace shorter than token_ids; keeping shorter trace for fallback scoring. "
                    "sample_idx=%s token_ids=%s logprobs=%s text=%s",
                    int(out_idx),
                    int(len(token_ids)),
                    int(len(token_logprobs)),
                    int(len(generated_token_text)),
                )

            trace_token_ids = token_ids[: len(token_logprobs)]
            generated_token_text = [
                str(t)
                for t in _decode_pieces(
                    tokenizer=owner.tokenizer,
                    token_ids=trace_token_ids,
                )
            ]
            results.append(
                (
                    token_ids,
                    text,
                    decode_mode,
                    prompt_ids,
                    token_logprobs,
                    generated_token_text,
                )
            )
        else:
            results.append((token_ids, text, decode_mode, prompt_ids))

    return results

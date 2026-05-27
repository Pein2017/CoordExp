from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.infer.backend import (
    build_swift_request_config_from_decode_request,
    decode_token_pieces_with_tokenizer,
    import_swift_infer_request_and_config,
    normalize_vllm_trace_response,
    strip_left_padding_token_ids,
)
from src.infer.runtime import (
    build_decode_request_from_rollout_facts,
    resolve_rollout_decode_facts_from_owner,
)


@dataclass(frozen=True)
class VLLMColocateRolloutHandles:
    """Lifecycle handles needed by the colocated vLLM rollout core."""

    decode_facts: Any
    global_step: int
    seed_base: int
    normalize_seed_fn: Any
    eval_window_active: bool
    offload_context_fn: Any
    sync_model_fn: Any
    infer_tp_group_fn: Any
    tokenizer: Any


def resolve_vllm_colocate_rollout_handles_from_owner(
    *,
    owner: Any,
) -> VLLMColocateRolloutHandles:
    """Translate a Stage-2 owner into colocated vLLM rollout handles."""

    gs = int(getattr(getattr(owner, "state", None), "global_step", 0) or 0)
    return VLLMColocateRolloutHandles(
        decode_facts=resolve_rollout_decode_facts_from_owner(owner),
        global_step=int(gs),
        seed_base=int(owner._derive_rollout_seed_base(global_step=gs)),
        normalize_seed_fn=owner._normalize_rollout_seed_int32,
        eval_window_active=bool(getattr(owner, "_eval_vllm_window_active", False)),
        offload_context_fn=owner._maybe_rollout_offload_context,
        sync_model_fn=owner._sync_vllm_rollout_model_if_needed,
        infer_tp_group_fn=lambda *, infer_requests, request_config: vllm_infer_tp_group(
            owner=owner,
            infer_requests=infer_requests,
            request_config=request_config,
        ),
        tokenizer=owner.tokenizer,
    )


def vllm_infer_tp_group(
    *,
    owner: Any,
    infer_requests: List[Dict[str, Any]],
    request_config: Any,
) -> List[Any]:
    engine = owner._ensure_vllm_engine()
    tp = int(owner._vllm_tp_size)

    vcfg = owner._cfg("vllm", None)
    infer_batch_size: Optional[int] = None
    if isinstance(vcfg, Mapping):
        raw = vcfg.get("infer_batch_size", None)
        if raw is not None:
            try:
                infer_batch_size = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rollout_matching.vllm.infer_batch_size must be an int"
                ) from exc
            if infer_batch_size <= 0:
                infer_batch_size = None

    def _infer_batched(reqs: List[Dict[str, Any]]) -> List[Any]:
        if not reqs:
            return []
        if infer_batch_size is None or infer_batch_size >= len(reqs):
            return engine.infer(reqs, request_config=request_config, use_tqdm=False)
        outs: List[Any] = []
        for i in range(0, len(reqs), infer_batch_size):
            outs.extend(
                engine.infer(
                    reqs[i : i + infer_batch_size],
                    request_config=request_config,
                    use_tqdm=False,
                )
            )
        return outs

    if tp <= 1:
        return _infer_batched(infer_requests)

    import torch.distributed as dist

    group = owner._vllm_tp_group
    local_rank = int(dist.get_rank(group=group))
    local_len = int(len(infer_requests))
    all_lens: List[int] = [0 for _ in range(tp)]
    dist.all_gather_object(all_lens, local_len, group=group)
    start_idx = sum(int(x) for x in all_lens[:local_rank])
    end_idx = start_idx + local_len

    gathered: List[List[Dict[str, Any]]] = [[] for _ in range(tp)]
    dist.all_gather_object(gathered, infer_requests, group=group)
    flat: List[Dict[str, Any]] = [x for sub in gathered for x in sub]

    outs = _infer_batched(flat)
    return outs[start_idx:end_idx]


def rollout_many_vllm_colocate(
    *,
    owner: Any,
    samples: Sequence[Mapping[str, Any]],
    logger: Any,
    with_logprobs: bool = False,
    request_index_offset: int = 0,
    decode_override: Optional[Mapping[str, Any]] = None,
) -> List[Any]:
    return rollout_many_vllm_colocate_with_handles(
        handles=resolve_vllm_colocate_rollout_handles_from_owner(owner=owner),
        samples=samples,
        logger=logger,
        with_logprobs=with_logprobs,
        request_index_offset=request_index_offset,
        decode_override=decode_override,
    )


def rollout_many_vllm_colocate_with_handles(
    *,
    handles: VLLMColocateRolloutHandles,
    samples: Sequence[Mapping[str, Any]],
    logger: Any,
    with_logprobs: bool = False,
    request_index_offset: int = 0,
    decode_override: Optional[Mapping[str, Any]] = None,
) -> List[Any]:
    del logger
    rollout_facts = handles.decode_facts
    decode_request = build_decode_request_from_rollout_facts(
        rollout_facts,
        decode_override=decode_override
    )
    decode_mode = str(decode_request.decode_mode)
    if decode_mode == "beam":
        raise ValueError(
            "vLLM rollout backend does not support decode_mode=beam; "
            "use greedy or sampling overrides instead"
        )

    temperature = float(decode_request.temperature)

    if with_logprobs and float(temperature) > 0.0:
        raise ValueError(
            "eval-step confidence scoring requires decoding.temperature=0.0 "
            f"(greedy), got {float(temperature)}"
        )

    InferRequest, _RequestConfig = import_swift_infer_request_and_config()
    request_index_offset_i = max(0, int(request_index_offset))
    request_config = build_swift_request_config_from_decode_request(
        decode_request,
        seed=handles.normalize_seed_fn(
            int(handles.seed_base + request_index_offset_i)
        ),
        trace_logprobs=bool(with_logprobs),
    )

    infer_requests: List[Any] = []
    for s in samples:
        msgs = s.get("messages")
        if not isinstance(msgs, list):
            raise ValueError("rollout-matching samples must contain messages (list)")
        infer_requests.append(InferRequest(messages=msgs))

    offload_cm = nullcontext() if handles.eval_window_active else handles.offload_context_fn(rollout_backend="vllm")
    with offload_cm:
        if not handles.eval_window_active:
            handles.sync_model_fn()
        outs: List[Any] = handles.infer_tp_group_fn(
            infer_requests=infer_requests,
            request_config=request_config,
        )

    if len(outs) != len(infer_requests):
        raise RuntimeError("vLLM returned unexpected number of outputs")

    results: List[Any] = []
    for out_idx, out in enumerate(outs):
        if isinstance(out, Exception):
            raise RuntimeError(
                "vLLM decode failed for a rollout sample "
                f"(sample_idx={int(out_idx)}): {out!r}"
            ) from out

        try:
            choices = getattr(out, "choices")
            if not choices:
                raise RuntimeError("missing choices")
            choice0 = choices[0]
            text = str(getattr(getattr(choice0, "message", None), "content", "") or "")
            token_ids_raw = getattr(choice0, "token_ids", None)
            if not isinstance(token_ids_raw, list):
                raise RuntimeError("missing token_ids")
            token_ids = [int(t) for t in token_ids_raw]
            prompt_ids_raw = getattr(out, "prompt_token_ids", None)
            if not isinstance(prompt_ids_raw, list):
                raise RuntimeError("missing prompt_token_ids")
            prompt_ids = [int(t) for t in prompt_ids_raw]
            prompt_ids = strip_left_padding_token_ids(
                prompt_ids,
                pad_token_id=getattr(handles.tokenizer, "pad_token_id", None),
            ) or prompt_ids
            choice_logprobs = getattr(choice0, "logprobs", None)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "malformed vLLM decode output "
                f"(sample_idx={int(out_idx)}): {exc}"
            ) from exc

        if with_logprobs:
            result = normalize_vllm_trace_response(
                {
                    "prompt_token_ids": prompt_ids,
                    "choices": [
                        {
                            "message": {"content": text},
                            "token_ids": token_ids,
                            "logprobs": choice_logprobs,
                        }
                    ],
                },
                trace_logprobs=True,
                backend_mode="ms-swift",
                pad_token_id=getattr(handles.tokenizer, "pad_token_id", None),
            )
            token_ids = [int(t) for t in (result.generated_token_ids or [])]
            prompt_ids = [int(t) for t in (result.prompt_token_ids or [])]
            token_logprobs = [float(t) for t in (result.generated_logprobs or [])]
            generated_token_text = list(
                result.generated_tokens
                or decode_token_pieces_with_tokenizer(
                    tokenizer=handles.tokenizer,
                    token_ids=token_ids,
                )
            )
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


__all__ = ["vllm_infer_tp_group", "rollout_many_vllm_colocate"]

from __future__ import annotations

from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple


def rollout_many_vllm(
    *,
    owner: Any,
    samples: Sequence[Mapping[str, Any]],
    debug_samples: Optional[Sequence[Mapping[str, Any]]] = None,
    request_index_offset: int = 0,
    decode_override: Optional[Mapping[str, Any]] = None,
) -> List[Tuple[List[int], str, str, List[int]]]:
    """Shared vLLM dispatch across colocate and server modes."""

    mode = owner._vllm_mode()
    if mode == "server":
        if decode_override is None:
            return owner._rollout_many_vllm_server(
                samples,
                debug_samples=debug_samples,
                request_index_offset=int(request_index_offset),
            )
        return owner._rollout_many_vllm_server(
            samples,
            debug_samples=debug_samples,
            request_index_offset=int(request_index_offset),
            decode_override=decode_override,
        )
    if decode_override is None:
        return owner._rollout_many_vllm_colocate(
            samples,
            request_index_offset=int(request_index_offset),
        )
    return owner._rollout_many_vllm_colocate(
        samples,
        request_index_offset=int(request_index_offset),
        decode_override=decode_override,
    )


def rollout_many_vllm_traced(
    *,
    owner: Any,
    samples: Sequence[Mapping[str, Any]],
    debug_samples: Optional[Sequence[Mapping[str, Any]]] = None,
    request_index_offset: int = 0,
    decode_override: Optional[Mapping[str, Any]] = None,
) -> List[Tuple[List[int], str, str, List[int], List[float], List[str]]]:
    """Shared vLLM dispatch that also captures per-token logprobs."""

    mode = owner._vllm_mode()
    if mode == "server":
        out = owner._rollout_many_vllm_server(
            samples,
            debug_samples=debug_samples,
            request_index_offset=int(request_index_offset),
            with_logprobs=True,
            decode_override=decode_override,
        )
    else:
        if decode_override is None:
            out = owner._rollout_many_vllm_colocate(
                samples,
                with_logprobs=True,
                request_index_offset=int(request_index_offset),
            )
        else:
            out = owner._rollout_many_vllm_colocate(
                samples,
                with_logprobs=True,
                request_index_offset=int(request_index_offset),
                decode_override=decode_override,
            )
    return [
        (
            list(token_ids),
            str(text),
            str(decode_mode),
            list(prompt_ids),
            [float(lp) for lp in token_logprobs],
            [str(t) for t in generated_token_text],
        )
        for (
            token_ids,
            text,
            decode_mode,
            prompt_ids,
            token_logprobs,
            generated_token_text,
        ) in out
    ]


def rollout_many(
    *,
    owner: Any,
    samples: Sequence[Mapping[str, Any]],
    prompt_variant_override: Optional[str] = None,
    rollout_backend: Optional[Literal["hf", "vllm"]] = None,
    decode_override: Optional[Mapping[str, Any]] = None,
    request_index_offset: int = 0,
) -> List[Tuple[List[int], str, str, List[int]]]:
    """Shared backend dispatch entrypoint for rollout-aligned training/eval."""

    rollout_context = owner._current_rollout_context()
    backend = (
        rollout_backend
        if rollout_backend is not None
        else owner._effective_rollout_backend(context=rollout_context)
    )
    samples_for_rollout = owner._prepare_samples_for_rollout(
        samples,
        prompt_variant_override=prompt_variant_override,
        rollout_backend=backend,
    )

    if backend == "hf":
        if decode_override is None:
            return owner._rollout_many_hf(samples_for_rollout)
        return owner._rollout_many_hf(
            samples_for_rollout,
            decode_override=decode_override,
        )

    if backend == "vllm":
        mode = owner._vllm_mode()
        request_index_offset_base = max(0, int(request_index_offset))
        if mode == "server":
            chunk_size = max(
                1,
                int(
                    owner._rollout_decode_batch_size_per_rank(
                        rollout_backend=backend,
                        rollout_context=rollout_context,
                    )
                ),
            )
            if int(len(samples_for_rollout)) > 0:
                chunk_size = min(chunk_size, int(len(samples_for_rollout)))

            out: List[Tuple[List[int], str, str, List[int]]] = []

            for off in range(0, int(len(samples_for_rollout)), int(chunk_size)):
                chunk_samples = samples_for_rollout[int(off) : int(off + chunk_size)]
                chunk_debug_samples = samples[int(off) : int(off + chunk_size)]
                if decode_override is None:
                    chunk_out = owner._rollout_many_vllm(
                        chunk_samples,
                        debug_samples=chunk_debug_samples,
                        request_index_offset=int(request_index_offset_base + off),
                    )
                else:
                    chunk_out = owner._rollout_many_vllm(
                        chunk_samples,
                        debug_samples=chunk_debug_samples,
                        request_index_offset=int(request_index_offset_base + off),
                        decode_override=decode_override,
                    )
                out.extend(chunk_out)
        else:
            if decode_override is None:
                out = owner._rollout_many_vllm(
                    samples_for_rollout,
                    debug_samples=samples,
                    request_index_offset=int(request_index_offset_base),
                )
            else:
                out = owner._rollout_many_vllm(
                    samples_for_rollout,
                    debug_samples=samples,
                    request_index_offset=int(request_index_offset_base),
                    decode_override=decode_override,
                )
        return out

    raise AssertionError("unreachable")

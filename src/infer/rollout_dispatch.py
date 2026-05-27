from __future__ import annotations

"""Rollout backend dispatch helpers owned by the shared inference runtime."""

from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple

from swift.utils import unwrap_model_for_generation

from src.infer.backend import rollout_many_hf, rollout_many_hf_traced
from src.infer.backend_vllm_infer import rollout_many_vllm_colocate
from src.infer.backend_vllm_server import rollout_many_vllm_server
from src.infer.prompt import prepare_rollout_prompt_samples_from_owner
from src.infer.runtime import (
    build_decode_request_from_rollout_owner,
    current_rollout_context_from_owner,
    effective_rollout_backend_from_owner,
    vllm_mode_from_rollout_owner,
)


def _owner_logger(owner: Any) -> Any:
    logger = getattr(owner, "_rollout_logger", None)
    if logger is not None:
        return logger
    import logging

    return logging.getLogger(__name__)


def rollout_many_vllm(
    *,
    owner: Any,
    samples: Sequence[Mapping[str, Any]],
    debug_samples: Optional[Sequence[Mapping[str, Any]]] = None,
    request_index_offset: int = 0,
    decode_override: Optional[Mapping[str, Any]] = None,
) -> List[Tuple[List[int], str, str, List[int]]]:
    """Shared vLLM dispatch across colocate and server modes."""

    mode = vllm_mode_from_rollout_owner(owner)
    if mode == "server":
        return rollout_many_vllm_server(
            owner=owner,
            logger=_owner_logger(owner),
            samples=samples,
            debug_samples=debug_samples,
            request_index_offset=int(request_index_offset),
            decode_override=decode_override,
        )
    return rollout_many_vllm_colocate(
        owner=owner,
        logger=_owner_logger(owner),
        samples=samples,
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

    mode = vllm_mode_from_rollout_owner(owner)
    if mode == "server":
        out = rollout_many_vllm_server(
            owner=owner,
            logger=_owner_logger(owner),
            samples=samples,
            debug_samples=debug_samples,
            request_index_offset=int(request_index_offset),
            with_logprobs=True,
            decode_override=decode_override,
        )
    else:
        out = rollout_many_vllm_colocate(
            owner=owner,
            logger=_owner_logger(owner),
            samples=samples,
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

    rollout_context = current_rollout_context_from_owner(owner)
    backend = (
        rollout_backend
        if rollout_backend is not None
        else effective_rollout_backend_from_owner(owner, context=rollout_context)
    )
    samples_for_rollout = prepare_rollout_prompt_samples_from_owner(
        owner,
        samples,
        prompt_variant_override=prompt_variant_override,
        rollout_backend=backend,
    )

    if backend == "hf":
        return rollout_many_hf(
            owner=owner,
            samples=samples_for_rollout,
            decode_request=build_decode_request_from_rollout_owner(
                owner,
                decode_override=decode_override,
            ),
            unwrap_model_for_generation_fn=unwrap_model_for_generation,
        )

    if backend == "vllm":
        mode = vllm_mode_from_rollout_owner(owner)
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
                chunk_out = rollout_many_vllm(
                    owner=owner,
                    samples=chunk_samples,
                    debug_samples=chunk_debug_samples,
                    request_index_offset=int(request_index_offset_base + off),
                    decode_override=decode_override,
                )
                out.extend(chunk_out)
        else:
            out = rollout_many_vllm(
                owner=owner,
                samples=samples_for_rollout,
                debug_samples=samples,
                request_index_offset=int(request_index_offset_base),
                decode_override=decode_override,
            )
        return out

    raise AssertionError("unreachable")


def rollout_many_traced(
    *,
    owner: Any,
    samples: Sequence[Mapping[str, Any]],
    prompt_variant_override: Optional[str] = None,
    rollout_backend: Optional[Literal["hf", "vllm"]] = None,
    decode_override: Optional[Mapping[str, Any]] = None,
    request_index_offset: int = 0,
) -> List[Tuple[List[int], str, str, List[int], List[float], List[str]]]:
    """Shared rollout dispatch entrypoint that requires generated-token traces."""

    rollout_context = current_rollout_context_from_owner(owner)
    backend = (
        rollout_backend
        if rollout_backend is not None
        else effective_rollout_backend_from_owner(owner, context=rollout_context)
    )
    samples_for_rollout = prepare_rollout_prompt_samples_from_owner(
        owner,
        samples,
        prompt_variant_override=prompt_variant_override,
        rollout_backend=backend,
    )

    if backend == "hf":
        return rollout_many_hf_traced(
            owner=owner,
            samples=samples_for_rollout,
            decode_request=build_decode_request_from_rollout_owner(
                owner,
                decode_override=decode_override,
            ),
            unwrap_model_for_generation_fn=unwrap_model_for_generation,
        )

    if backend == "vllm":
        return rollout_many_vllm_traced(
            owner=owner,
            samples=samples_for_rollout,
            debug_samples=samples,
            request_index_offset=max(0, int(request_index_offset)),
            decode_override=decode_override,
        )

    raise AssertionError("unreachable")

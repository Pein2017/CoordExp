from __future__ import annotations

"""Rollout backend dispatch helpers owned by the shared inference runtime."""

from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple

from swift.utils import unwrap_model_for_generation

from src.infer.backend import (
    resolve_hf_rollout_backend_handles_from_owner,
    rollout_many_hf,
    rollout_many_hf_traced,
    rollout_many_hf_traced_with_handles,
    rollout_many_hf_with_handles,
)
from src.infer.backend_vllm_infer import (
    resolve_vllm_colocate_rollout_handles_from_owner,
    rollout_many_vllm_colocate,
    rollout_many_vllm_colocate_with_handles,
)
from src.infer.backend_vllm_server import (
    resolve_vllm_server_rollout_handles_from_owner,
    rollout_many_vllm_server,
    rollout_many_vllm_server_with_handles,
    validate_vllm_server_decode_request,
)
from src.infer.prompt import prepare_rollout_prompt_samples_from_owner
from src.infer.runtime import (
    build_decode_request_from_rollout_facts,
    current_rollout_context_from_owner,
    effective_rollout_backend_from_owner,
    resolve_rollout_decode_facts_from_owner,
    vllm_mode_from_rollout_owner,
)


@dataclass(frozen=True)
class RolloutDispatchHandles:
    """Resolved backend handles consumed by owner-free rollout dispatch cores."""

    backend: Literal["hf", "vllm"]
    vllm_mode: Literal["colocate", "server"]
    decode_request: Any
    logger: Any
    hf_handles_fn: Any = None
    vllm_colocate_handles_fn: Any = None
    vllm_server_handles_fn: Any = None
    vllm_server_chunk_size_fn: Any = None


def _owner_logger(owner: Any) -> Any:
    logger = getattr(owner, "_rollout_logger", None)
    if logger is not None:
        return logger
    import logging

    return logging.getLogger(__name__)


def resolve_rollout_dispatch_handles_from_owner(
    *,
    owner: Any,
    backend: Literal["hf", "vllm"],
    decode_request: Any,
) -> RolloutDispatchHandles:
    """Translate the Stage-2 owner into handles for shared rollout dispatch."""

    logger = _owner_logger(owner)
    if backend == "hf":
        return RolloutDispatchHandles(
            backend="hf",
            vllm_mode="colocate",
            decode_request=decode_request,
            logger=logger,
            hf_handles_fn=lambda: resolve_hf_rollout_backend_handles_from_owner(
                owner=owner,
                unwrap_model_for_generation_fn=unwrap_model_for_generation,
            ),
        )

    mode = vllm_mode_from_rollout_owner(owner)
    if mode == "server":
        legacy_chunk_fn = getattr(owner, "_rollout_decode_batch_size_per_rank", None)

        def _server_handles() -> Any:
            return resolve_vllm_server_rollout_handles_from_owner(
                owner=owner,
                logger=logger,
            )

        def _server_chunk_size() -> int:
            if callable(legacy_chunk_fn):
                return int(
                    legacy_chunk_fn(
                        rollout_backend="vllm",
                        rollout_context="train",
                    )
                )
            return int(_server_handles().per_rank_chunk_fn())

        return RolloutDispatchHandles(
            backend="vllm",
            vllm_mode="server",
            decode_request=decode_request,
            logger=logger,
            vllm_server_handles_fn=_server_handles,
            vllm_server_chunk_size_fn=_server_chunk_size,
        )

    return RolloutDispatchHandles(
        backend="vllm",
        vllm_mode="colocate",
        decode_request=decode_request,
        logger=logger,
        vllm_colocate_handles_fn=lambda: resolve_vllm_colocate_rollout_handles_from_owner(
            owner=owner,
        ),
    )


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


def rollout_many_with_handles(
    *,
    handles: RolloutDispatchHandles,
    samples_for_rollout: Sequence[Mapping[str, Any]],
    debug_samples: Sequence[Mapping[str, Any]],
    request_index_offset: int = 0,
    decode_override: Optional[Mapping[str, Any]] = None,
) -> List[Tuple[List[int], str, str, List[int]]]:
    """Owner-free backend dispatch core for rollout-aligned training/eval."""

    if handles.backend == "hf":
        return rollout_many_hf_with_handles(
            handles=handles.hf_handles_fn(),
            samples=samples_for_rollout,
            decode_request=handles.decode_request,
        )

    request_index_offset_base = max(0, int(request_index_offset))
    if handles.vllm_mode == "server":
        validate_vllm_server_decode_request(
            decode_request=handles.decode_request,
            with_logprobs=False,
        )
        chunk_size_fn = handles.vllm_server_chunk_size_fn
        chunk_size = max(1, int(chunk_size_fn()))
        if int(len(samples_for_rollout)) > 0:
            chunk_size = min(chunk_size, int(len(samples_for_rollout)))

        out: List[Tuple[List[int], str, str, List[int]]] = []
        for off in range(0, int(len(samples_for_rollout)), int(chunk_size)):
            chunk_samples = samples_for_rollout[int(off) : int(off + chunk_size)]
            chunk_debug_samples = debug_samples[int(off) : int(off + chunk_size)]
            chunk_out = rollout_many_vllm_server_with_handles(
                handles=handles.vllm_server_handles_fn(),
                logger=handles.logger,
                samples=chunk_samples,
                debug_samples=chunk_debug_samples,
                request_index_offset=int(request_index_offset_base + off),
                decode_override=decode_override,
            )
            out.extend(chunk_out)
        return out

    return rollout_many_vllm_colocate_with_handles(
        handles=handles.vllm_colocate_handles_fn(),
        logger=handles.logger,
        samples=samples_for_rollout,
        request_index_offset=int(request_index_offset_base),
        decode_override=decode_override,
    )


def rollout_many_traced_with_handles(
    *,
    handles: RolloutDispatchHandles,
    samples_for_rollout: Sequence[Mapping[str, Any]],
    debug_samples: Sequence[Mapping[str, Any]],
    request_index_offset: int = 0,
    decode_override: Optional[Mapping[str, Any]] = None,
) -> List[Tuple[List[int], str, str, List[int], List[float], List[str]]]:
    """Owner-free traced rollout dispatch core."""

    if handles.backend == "hf":
        return rollout_many_hf_traced_with_handles(
            handles=handles.hf_handles_fn(),
            samples=samples_for_rollout,
            decode_request=handles.decode_request,
        )

    if handles.vllm_mode == "server":
        validate_vllm_server_decode_request(
            decode_request=handles.decode_request,
            with_logprobs=True,
        )
        out = rollout_many_vllm_server_with_handles(
            handles=handles.vllm_server_handles_fn(),
            logger=handles.logger,
            samples=samples_for_rollout,
            debug_samples=debug_samples,
            request_index_offset=max(0, int(request_index_offset)),
            with_logprobs=True,
            decode_override=decode_override,
        )
    else:
        out = rollout_many_vllm_colocate_with_handles(
            handles=handles.vllm_colocate_handles_fn(),
            logger=handles.logger,
            samples=samples_for_rollout,
            with_logprobs=True,
            request_index_offset=max(0, int(request_index_offset)),
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
    decode_facts = resolve_rollout_decode_facts_from_owner(owner)
    decode_request = build_decode_request_from_rollout_facts(
        decode_facts,
        decode_override=decode_override,
    )
    samples_for_rollout = prepare_rollout_prompt_samples_from_owner(
        owner,
        samples,
        prompt_variant_override=prompt_variant_override,
        rollout_backend=backend,
    )
    return rollout_many_with_handles(
        handles=resolve_rollout_dispatch_handles_from_owner(
            owner=owner,
            backend=backend,
            decode_request=decode_request,
        ),
        samples_for_rollout=samples_for_rollout,
        debug_samples=samples,
        request_index_offset=request_index_offset,
        decode_override=decode_override,
    )


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
    decode_facts = resolve_rollout_decode_facts_from_owner(owner)
    decode_request = build_decode_request_from_rollout_facts(
        decode_facts,
        decode_override=decode_override,
    )
    samples_for_rollout = prepare_rollout_prompt_samples_from_owner(
        owner,
        samples,
        prompt_variant_override=prompt_variant_override,
        rollout_backend=backend,
    )
    return rollout_many_traced_with_handles(
        handles=resolve_rollout_dispatch_handles_from_owner(
            owner=owner,
            backend=backend,
            decode_request=decode_request,
        ),
        samples_for_rollout=samples_for_rollout,
        debug_samples=samples,
        request_index_offset=request_index_offset,
        decode_override=decode_override,
    )

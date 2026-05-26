from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Mapping


@lru_cache(maxsize=2)
def vllm_engine_args_fields(*, async_engine: bool = False) -> frozenset[str]:
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for rollout_matching.vllm compatibility checks. "
            "Use rollout_backend=hf to bypass vLLM."
        ) from exc

    cls = AsyncEngineArgs if bool(async_engine) else EngineArgs
    fields = set(getattr(cls, "__dataclass_fields__", {}) or {})
    fields.update(inspect.signature(cls).parameters)
    return frozenset(str(name) for name in fields)


def validate_vllm_engine_kwargs(
    kwargs: Mapping[str, Any],
    *,
    context: str,
    async_engine: bool = False,
    required: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    supported = vllm_engine_args_fields(async_engine=async_engine)
    out = dict(kwargs)

    unsupported = sorted(str(key) for key in out if str(key) not in supported)
    if unsupported:
        raise RuntimeError(
            f"{context} contains vLLM EngineArgs keys unsupported by the installed "
            f"vLLM package: {unsupported}. Remove the config key or upgrade to a "
            "compatible vLLM build before launching server-mode rollouts."
        )

    missing_required = sorted(str(key) for key in required if str(key) not in supported)
    if missing_required:
        raise RuntimeError(
            f"{context} requires vLLM EngineArgs keys unsupported by the installed "
            f"vLLM package: {missing_required}. Upgrade vLLM or change rollout backend."
        )

    return out

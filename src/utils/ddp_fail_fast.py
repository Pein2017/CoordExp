from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import torch

T = TypeVar("T")


class DDPFailFastError(RuntimeError):
    """Raised when a DDP-coordinated fail-fast abort triggers."""


@dataclass(frozen=True)
class DDPContext:
    dist: Any
    rank: int
    world_size: int
    backend: str | None
    coordination_device: torch.device


def _import_dist() -> Any | None:
    try:
        import torch.distributed as dist
    except Exception:
        return None
    return dist


def _coordination_device(dist: Any, *, model: Any | None) -> torch.device:
    backend = None
    try:
        backend = str(dist.get_backend())
    except Exception:
        backend = None

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise DDPFailFastError(
                "torch.distributed backend is NCCL but CUDA is not available"
            )
        if model is not None:
            try:
                device = getattr(model, "device", None)
                if isinstance(device, torch.device) and device.type == "cuda":
                    return device
            except Exception:
                pass
            try:
                device = next(model.parameters()).device
                if isinstance(device, torch.device) and device.type == "cuda":
                    return device
            except Exception:
                pass
        return torch.device("cuda", int(torch.cuda.current_device()))

    return torch.device("cpu")


def maybe_ddp_context(*, model: Any | None = None) -> DDPContext | None:
    """Return a minimal DDP context when torch.distributed is initialized (world_size>1)."""

    dist = _import_dist()
    if dist is None or (not dist.is_available()) or (not dist.is_initialized()):
        return None

    world_size = int(dist.get_world_size())
    if world_size <= 1:
        return None

    rank = int(dist.get_rank())
    backend = None
    try:
        backend = str(dist.get_backend())
    except Exception:
        backend = None

    return DDPContext(
        dist=dist,
        rank=int(rank),
        world_size=int(world_size),
        backend=backend,
        coordination_device=_coordination_device(dist, model=model),
    )


def _all_reduce_int(dist: Any, value: int, *, op: Any, device: torch.device) -> int:
    t = torch.tensor([int(value)], dtype=torch.int32, device=device)
    dist.all_reduce(t, op=op)
    return int(t.item())


def ddp_any_rank_fail_fast(
    *,
    where: str,
    fn: Callable[[], T],
    model: Any | None = None,
) -> T:
    """Execute `fn` and coordinate an abort if any rank throws.

    Notes:
      - This coordination works only when every rank reaches the coordination step.
      - This helper is intended for DDP-critical regions where all ranks are expected
        to execute the same control flow.
      - Non-failing ranks raise a minimal, deterministic message; the failing rank
        chains the original exception as the cause.
    """

    ctx = maybe_ddp_context(model=model)
    if ctx is None:
        return fn()

    local_failed = False
    local_exc: Exception | None = None
    try:
        result = fn()
    except Exception as exc:
        local_failed = True
        local_exc = exc
        result = None  # type: ignore[assignment]

    any_failed = _all_reduce_int(
        ctx.dist,
        1 if local_failed else 0,
        op=ctx.dist.ReduceOp.MAX,
        device=ctx.coordination_device,
    )

    local_rank_for_min = int(ctx.rank) if local_failed else int(ctx.world_size + 10_000)
    failing_rank = _all_reduce_int(
        ctx.dist,
        local_rank_for_min,
        op=ctx.dist.ReduceOp.MIN,
        device=ctx.coordination_device,
    )

    if int(any_failed) != 0:
        msg = (
            "DDP fail-fast abort: "
            f"where={str(where)} failing_rank={int(failing_rank)} "
            f"rank={int(ctx.rank)}/{int(ctx.world_size)}"
        )
        if local_failed and local_exc is not None:
            summary = f"{local_exc.__class__.__name__}: {local_exc}"
            raise DDPFailFastError(f"{msg} error={summary}") from local_exc
        raise DDPFailFastError(msg)

    return result


def ddp_rank0_coordinated_fail_fast(
    *,
    where: str,
    fn_rank0_only: Callable[[], T],
    model: Any | None = None,
    barrier: Callable[[], None] | None = None,
) -> T | None:
    """Run a rank0-only side effect, broadcasting failure to all ranks.

    The caller may pass a bounded `barrier` callable to align entry/exit.

    Returns:
      - On rank0 (DDP or non-DDP): returns the result of `fn_rank0_only`.
      - On non-rank0 under DDP: returns None.
    """

    ctx = maybe_ddp_context(model=model)
    if ctx is None:
        return fn_rank0_only()

    if barrier is not None:
        barrier()

    local_failed = False
    local_exc: Exception | None = None
    local_msg = ""
    result: T | None = None

    if int(ctx.rank) == 0:
        try:
            result = fn_rank0_only()
        except Exception as exc:
            local_failed = True
            local_exc = exc
            local_msg = f"{exc.__class__.__name__}: {exc}"

    failed_flag = torch.tensor(
        [1 if local_failed else 0], dtype=torch.int32, device=ctx.coordination_device
    )
    ctx.dist.broadcast(failed_flag, src=0)

    msg_list: list[Any] = [local_msg]
    ctx.dist.broadcast_object_list(msg_list, src=0)
    msg = str(msg_list[0])

    if barrier is not None:
        barrier()

    if int(failed_flag.item()) != 0:
        base = (
            "DDP fail-fast abort (rank0 side effect): "
            f"where={str(where)} rank={int(ctx.rank)}/{int(ctx.world_size)}"
        )
        if int(ctx.rank) == 0 and local_exc is not None:
            raise DDPFailFastError(f"{base} error={msg}") from local_exc
        raise DDPFailFastError(f"{base} error={msg}")

    return result

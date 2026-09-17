"""Completion training's SUM and state-consensus protocol.

Callers supply already globally normalized gradients. No world-size division,
cohort, partition, optimizer recipe, launcher or training loop is selected here.
Fingerprint encoding is unchanged from the retained dual-start producer.
"""

from __future__ import annotations
import hashlib
import resource
from typing import Any, Callable, Mapping, Sequence, TypeVar
import torch
import torch.distributed as dist
from src.runtime.distributed import gather_objects as _gather_all
from probes.training_set_completion import artifacts

COLLECTIVE_TIMEOUT_SECONDS = 600
T = TypeVar("T")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class DistributedTrainingError(RuntimeError):
    """A failure that every live rank has observed and accepted."""


def _coordination_device() -> torch.device:
    return (
        torch.device("cuda", torch.cuda.current_device())
        if dist.get_backend() == "nccl"
        else torch.device("cpu")
    )


def raise_if_rank_failed(error: BaseException | None, *, phase: str) -> None:
    """Propagate an ordinary local failure before starting the next collective phase."""
    local_failed = torch.tensor(
        [int(error is not None)], dtype=torch.int32, device=_coordination_device()
    )
    dist.all_reduce(local_failed, op=dist.ReduceOp.MAX)
    if not int(local_failed.item()):
        return
    local_message = None if error is None else f"{type(error).__name__}: {error}"
    messages: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(messages, local_message)
    failures = "; ".join(
        f"rank {rank}: {message}"
        for rank, message in enumerate(messages)
        if message is not None
    )
    raise DistributedTrainingError(f"{phase} failed collectively: {failures}")


def coordinated_call(function: Callable[[], T], *, phase: str) -> T:
    value: T | None = None
    error: BaseException | None = None
    try:
        value = function()
    except Exception as exc:
        error = exc
    raise_if_rank_failed(error, phase=phase)
    return value  # type: ignore[return-value]


def sum_gradients_(named: Sequence[tuple[str, torch.nn.Parameter]]) -> None:
    """SUM already-global-normalized gradients in a fixed parameter order."""
    preflight_error: BaseException | None = None
    try:
        for name, parameter in named:
            if parameter.grad is None:
                raise ValueError(f"missing local gradient before SUM: {name}")
            if not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError(f"nonfinite local gradient before SUM: {name}")
    except Exception as exc:
        preflight_error = exc
    raise_if_rank_failed(preflight_error, phase="gradient_sum_preflight")
    for _, parameter in named:
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    postflight_error: BaseException | None = None
    try:
        for name, parameter in named:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError(f"nonfinite global gradient after SUM: {name}")
    except Exception as exc:
        postflight_error = exc
    raise_if_rank_failed(postflight_error, phase="gradient_sum_postflight")


def _hash_tensor(hasher: Any, *, name: str, tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous()
    hasher.update(
        artifacts.canonical(
            {"name": name, "shape": list(cpu.shape), "dtype": str(cpu.dtype)}
        )
    )
    hasher.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())


def state_fingerprint(
    named: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Hash trainable parameters and Adam state without persisting per-rank copies."""
    parameters = hashlib.sha256()
    optimizer_state = hashlib.sha256()
    steps: set[int] = set()
    for name, parameter in named:
        _hash_tensor(parameters, name=name, tensor=parameter)
        state = optimizer.state.get(parameter, {})
        optimizer_state.update(
            artifacts.canonical({"parameter": name, "keys": sorted(state)})
        )
        for key in sorted(state):
            value = state[key]
            if isinstance(value, torch.Tensor):
                _hash_tensor(optimizer_state, name=f"{name}:{key}", tensor=value)
                if key == "step" and value.numel() == 1:
                    steps.add(int(value.item()))
            else:
                optimizer_state.update(
                    artifacts.canonical({"name": f"{name}:{key}", "value": value})
                )
                if key == "step":
                    steps.add(int(value))
    return {
        "parameter_sha256": parameters.hexdigest(),
        "optimizer_sha256": optimizer_state.hexdigest(),
        "optimizer_steps": sorted(steps),
        "parameter_count": len(named),
        "scalar_count": sum(parameter.numel() for _, parameter in named),
    }


def gather_objects(value: T) -> list[T]:
    values = _gather_all(value)
    return [item for item in values if item is not None]


def require_consensus(value: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    values = gather_objects(dict(value))
    require(len(values) == dist.get_world_size(), f"{label} rank receipt missing")
    require(
        len({artifacts.digest(item) for item in values}) == 1,
        f"{label} differs across ranks",
    )
    return values


def resource_receipt(device: torch.device) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    if device.type == "cuda":
        receipt.update(
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
    return receipt

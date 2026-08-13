"""Exact in-memory transactions for bounded Human-13 optimizer proposals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any
import uuid

import torch


@dataclass
class UpdateCounter:
    """Mutable applied-update counter included in each transaction."""

    value: int = 0


@dataclass(frozen=True)
class TrainingStateSnapshot:
    transaction_id: str
    parameter_values: tuple[tuple[str, torch.Tensor], ...]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    update_count: int
    cpu_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...] | None
    state_digest: str


@dataclass(frozen=True)
class TransactionReceipt:
    decision: str
    transaction_id: str
    before_state_digest: str
    after_state_digest: str


def _hash_value(hasher: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"tensor\0")
        hasher.update(str(tensor.dtype).encode())
        hasher.update(b"\0")
        hasher.update(repr(tuple(tensor.shape)).encode())
        hasher.update(b"\0")
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        hasher.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_value(hasher, key)
            _hash_value(hasher, value[key])
        return
    if isinstance(value, (tuple, list)):
        hasher.update(b"tuple\0" if isinstance(value, tuple) else b"list\0")
        for item in value:
            _hash_value(hasher, item)
        return
    if isinstance(value, bool):
        hasher.update(b"bool\1" if value else b"bool\0")
        return
    if isinstance(value, int):
        hasher.update(b"int\0" + str(value).encode() + b"\0")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("transaction state contains a nonfinite float")
        hasher.update(b"float\0" + struct.pack("!d", value))
        return
    if isinstance(value, str):
        hasher.update(b"str\0" + value.encode() + b"\0")
        return
    if value is None:
        hasher.update(b"none\0")
        return
    raise TypeError(f"unsupported transaction state type: {type(value).__name__}")


def _state_digest(value: Any) -> str:
    hasher = hashlib.sha256()
    _hash_value(hasher, value)
    return hasher.hexdigest()


class TrainingStateTransaction:
    """Snapshot and atomically accept or reject one optimizer mutation.

    Only caller-bound trainable tensors are copied. The complete optimizer and
    scheduler state are included, so AdamW moments and its step counter cannot
    leak across a rejected proposal.
    """

    def __init__(
        self,
        named_trainable_parameters: Sequence[tuple[str, torch.nn.Parameter]],
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        update_counter: UpdateCounter,
        capture_cuda: bool = True,
    ) -> None:
        bound = tuple(named_trainable_parameters)
        names = tuple(name for name, _ in bound)
        parameters = tuple(parameter for _, parameter in bound)
        if not bound or len(set(names)) != len(names):
            raise ValueError(
                "bound trainable parameter names must be nonempty and unique"
            )
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("bound trainable parameters must be unique")
        if any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("every bound parameter must require gradients")
        optimizer_parameters = tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        if {id(parameter) for parameter in optimizer_parameters} != {
            id(parameter) for parameter in parameters
        }:
            raise ValueError(
                "optimizer contains a parameter outside the bound trainable surface"
            )
        if len(optimizer_parameters) != len(parameters):
            raise ValueError("optimizer repeats a bound trainable parameter")
        self._named_parameters = bound
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._update_counter = update_counter
        self._capture_cuda = bool(capture_cuda)
        self._active_transaction_id: str | None = None

    def _cuda_rng_states(self) -> tuple[torch.Tensor, ...] | None:
        if not self._capture_cuda or not torch.cuda.is_available():
            return None
        return tuple(state.clone() for state in torch.cuda.get_rng_state_all())

    def _current_payload(self) -> dict[str, Any]:
        return {
            "parameters": tuple(
                (name, parameter.detach().clone())
                for name, parameter in self._named_parameters
            ),
            "optimizer": copy.deepcopy(self._optimizer.state_dict()),
            "scheduler": (
                copy.deepcopy(self._scheduler.state_dict())
                if self._scheduler is not None
                else None
            ),
            "update_count": int(self._update_counter.value),
            "cpu_rng": torch.get_rng_state().clone(),
            "cuda_rng": self._cuda_rng_states(),
        }

    def state_digest(self) -> str:
        return _state_digest(self._current_payload())

    def begin(self) -> TrainingStateSnapshot:
        if self._active_transaction_id is not None:
            raise RuntimeError("a training transaction is already active")
        transaction_id = uuid.uuid4().hex
        payload = self._current_payload()
        digest = _state_digest(payload)
        self._active_transaction_id = transaction_id
        return TrainingStateSnapshot(
            transaction_id=transaction_id,
            parameter_values=payload["parameters"],
            optimizer_state=payload["optimizer"],
            scheduler_state=payload["scheduler"],
            update_count=payload["update_count"],
            cpu_rng_state=payload["cpu_rng"],
            cuda_rng_states=payload["cuda_rng"],
            state_digest=digest,
        )

    def _require_active(self, snapshot: TrainingStateSnapshot) -> None:
        if snapshot.transaction_id != self._active_transaction_id:
            raise RuntimeError("snapshot does not belong to the active transaction")

    @torch.no_grad()
    def restore(self, snapshot: TrainingStateSnapshot) -> None:
        self._require_active(snapshot)
        live_by_name = dict(self._named_parameters)
        if tuple(live_by_name) != tuple(name for name, _ in snapshot.parameter_values):
            raise RuntimeError(
                "snapshot trainable surface no longer matches live parameters"
            )
        for name, saved in snapshot.parameter_values:
            live = live_by_name[name]
            live.copy_(saved.to(device=live.device, dtype=live.dtype))
        self._optimizer.load_state_dict(copy.deepcopy(snapshot.optimizer_state))
        if self._scheduler is not None:
            if snapshot.scheduler_state is None:
                raise RuntimeError("snapshot is missing bound scheduler state")
            self._scheduler.load_state_dict(copy.deepcopy(snapshot.scheduler_state))
        elif snapshot.scheduler_state is not None:
            raise RuntimeError("snapshot unexpectedly contains scheduler state")
        self._update_counter.value = int(snapshot.update_count)
        torch.set_rng_state(snapshot.cpu_rng_state.clone())
        if snapshot.cuda_rng_states is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("cannot restore captured CUDA RNG without CUDA")
            if len(snapshot.cuda_rng_states) != torch.cuda.device_count():
                raise RuntimeError("CUDA device count changed during transaction")
            torch.cuda.set_rng_state_all(
                [state.clone() for state in snapshot.cuda_rng_states]
            )
        restored_digest = self.state_digest()
        if restored_digest != snapshot.state_digest:
            raise RuntimeError(
                "restored training state digest does not match its snapshot: "
                f"{restored_digest} != {snapshot.state_digest}"
            )

    def accept(self, snapshot: TrainingStateSnapshot) -> TransactionReceipt:
        self._require_active(snapshot)
        after = self.state_digest()
        self._active_transaction_id = None
        return TransactionReceipt(
            decision="accepted_committed",
            transaction_id=snapshot.transaction_id,
            before_state_digest=snapshot.state_digest,
            after_state_digest=after,
        )

    def reject(self, snapshot: TrainingStateSnapshot) -> TransactionReceipt:
        self._require_active(snapshot)
        self.restore(snapshot)
        after = self.state_digest()
        self._active_transaction_id = None
        return TransactionReceipt(
            decision="rejected_restored",
            transaction_id=snapshot.transaction_id,
            before_state_digest=snapshot.state_digest,
            after_state_digest=after,
        )


__all__ = [
    "TrainingStateSnapshot",
    "TrainingStateTransaction",
    "TransactionReceipt",
    "UpdateCounter",
]

"""Bounded depth-one CPU forward-input lookahead for supervised training.

Two implementations of the same planned-step-scoped lifecycle protocol:
`SynchronousForwardInputProvider` (reference semantic path, no thread) and
`OverlappedForwardInputProvider` (one producer thread, `queue.Queue(maxsize=1)`
plus a `threading.Semaphore(1)` build slot, CPU-only producer work; device
transfer stays on the consuming step). Both providers build on CPU
(`device=None`) then move to the target device separately in `take()`, so
`total_build_inputs_ns` (feeding `input_build_seconds`) means the same thing
in both modes: CPU construction only. The H2D move itself is real time that
shows up inside the caller's `step_duration_seconds`, never inside
`input_build_seconds` or `input_wait_seconds` (queue-wait only). The
provider owns preparation; the trainer owns step boundaries by calling
`begin_planned_step` / `take` / `end_planned_step` / `close` explicitly.

The build slot enforces the true depth-one bound: the producer must acquire
it before starting a build and it is released by the consumer immediately
after dequeuing (before the H2D move), so the producer can never have a
second fully-built item in existence while the first sits unconsumed — the
`queue.Queue(maxsize=1)` alone does not prevent that, since a fast producer
could otherwise finish building item k+1 while item k is still queued.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import os
import queue
import threading
import time
from typing import Any, Protocol

import torch

from src.common.errors import RuntimeContractError
from src.qwen.forward import QwenForwardInputs, build_qwen_forward_inputs
from src.training.supervised_trainer import _logits_positions_to_keep as _logits_to_keep_positions


_QUEUE_POLL_SECONDS = 0.05
_JOIN_TIMEOUT_SECONDS = 30.0

_FORWARD_INPUT_PROVIDER_MODE_ENV = "COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE"
OVERLAPPED_MODE = "overlapped"
SYNCHRONOUS_MODE = "synchronous"
_FORWARD_INPUT_PROVIDER_MODES = frozenset({OVERLAPPED_MODE, SYNCHRONOUS_MODE})


class ForwardInputProvider(Protocol):
    """Planned-step-scoped lifecycle owned by the trainer."""

    last_take_wait_seconds: float

    def begin_planned_step(
        self, planned_step_id: int, moved_micro_steps: Sequence[Any]
    ) -> None: ...

    def take(self, ordinal: int, micro_step: Any) -> QwenForwardInputs: ...

    def end_planned_step(self) -> None: ...

    def close(self) -> None: ...


def resolve_forward_input_provider_mode() -> str:
    """Debug-only mode switch (env var, not YAML); default is synchronous.

    Overlap is implemented and semantically proven equivalent (see
    `tasks.md` 3.1-3.4), but is not the shipped default: the world_size=1
    M3 evidence collected against the depth-one-corrected code was
    inconclusive (2/5 repetitions favored overlap; see
    `implementation-notes.md` "M3"), so the stop rule keeps the
    synchronous reference path as the default until the exact 8-rank
    harness confirms a real win. `overlapped` remains selectable through
    this switch for measurement.
    """

    raw = os.environ.get(_FORWARD_INPUT_PROVIDER_MODE_ENV)
    if raw is None:
        return SYNCHRONOUS_MODE
    if raw not in _FORWARD_INPUT_PROVIDER_MODES:
        raise RuntimeContractError(
            "forward input provider mode override must be 'overlapped' or 'synchronous'",
            code="training.forward_input_provider_mode_invalid",
            context={"value": raw, "env_var": _FORWARD_INPUT_PROVIDER_MODE_ENV},
        )
    return raw


def build_forward_input_provider(mode: str) -> ForwardInputProvider:
    if mode == SYNCHRONOUS_MODE:
        return SynchronousForwardInputProvider()
    if mode == OVERLAPPED_MODE:
        return OverlappedForwardInputProvider()
    raise RuntimeContractError(
        "forward input provider mode must be 'overlapped' or 'synchronous'",
        code="training.forward_input_provider_mode_invalid",
        context={"value": mode},
    )


@dataclass(frozen=True)
class _PreparedItem:
    ordinal: int
    forward_inputs: QwenForwardInputs | None
    error: BaseException | None


class _StepBoundState:
    """Shared idle/active bookkeeping and ordinal/pack skew validation."""

    def __init__(self) -> None:
        self.active = False
        self.moved_micro_steps: tuple[Any, ...] = ()
        self.next_ordinal = 0

    def begin(self, moved_micro_steps: Sequence[Any]) -> None:
        if self.active:
            raise RuntimeContractError(
                "forward input provider begin_planned_step called while a step is already active",
                code="training.forward_input_provider_already_active",
            )
        self.moved_micro_steps = tuple(moved_micro_steps)
        self.next_ordinal = 0
        self.active = True

    def validate_take(self, ordinal: int, micro_step: Any) -> Any:
        if not self.active:
            raise RuntimeContractError(
                "forward input provider take called while no step is active",
                code="training.forward_input_provider_not_active",
                context={"ordinal": ordinal},
            )
        if ordinal != self.next_ordinal or not (0 <= ordinal < len(self.moved_micro_steps)):
            raise RuntimeContractError(
                "forward input provider take called with an unexpected ordinal",
                code="training.forward_input_provider_ordinal_skew",
                context={
                    "ordinal": ordinal,
                    "expected_ordinal": self.next_ordinal,
                    "moved_micro_step_count": len(self.moved_micro_steps),
                },
            )
        expected_micro_step = self.moved_micro_steps[ordinal]
        expected_pack_index = _pack_index(expected_micro_step)
        observed_pack_index = _pack_index(micro_step)
        if expected_pack_index != observed_pack_index:
            raise RuntimeContractError(
                "forward input provider take called with a mismatched pack identity",
                code="training.forward_input_provider_pack_skew",
                context={
                    "ordinal": ordinal,
                    "expected_pack_index": expected_pack_index,
                    "observed_pack_index": observed_pack_index,
                },
            )
        return expected_micro_step

    def advance(self, ordinal: int) -> None:
        self.next_ordinal = ordinal + 1

    def end(self) -> None:
        if not self.active:
            raise RuntimeContractError(
                "forward input provider end_planned_step called while no step is active",
                code="training.forward_input_provider_not_active",
            )
        self.active = False
        self.moved_micro_steps = ()
        self.next_ordinal = 0


def _pack_index(micro_step: Any) -> int:
    pack = getattr(micro_step, "pack", None)
    pack_index = getattr(pack, "pack_index", None)
    if pack_index is None:
        raise RuntimeContractError(
            "forward input provider requires a micro-step with pack.pack_index",
            code="training.forward_input_provider_missing_pack_index",
            context={"micro_step_type": type(micro_step).__name__},
        )
    return int(pack_index)


def _assert_not_closed(closed: bool) -> None:
    if closed:
        raise RuntimeContractError(
            "forward input provider used after close",
            code="training.forward_input_provider_closed",
        )


def _build_forward_inputs(
    micro_step: Any, *, device: torch.device | str | None
) -> QwenForwardInputs:
    return build_qwen_forward_inputs(
        micro_step.pack,
        micro_step.encoded_examples,
        micro_step.position_inputs,
        logits_to_keep_positions=_logits_to_keep_positions(micro_step),
        device=device,
        fa2_branch_proof_policy=micro_step.fa2_branch_proof_policy,
    )


def _move_forward_inputs_to_device(
    forward_inputs: QwenForwardInputs, device: torch.device | str | None
) -> QwenForwardInputs:
    torch_device = torch.device("cpu") if device is None else torch.device(device)
    fa2_plan = forward_inputs.fa2_varlen_plan
    moved_fa2_plan = replace(
        fa2_plan,
        cu_seq_lens_q=fa2_plan.cu_seq_lens_q.to(device=torch_device),
        cu_seq_lens_k=fa2_plan.cu_seq_lens_k.to(device=torch_device),
    )
    logits_to_keep = forward_inputs.logits_to_keep
    moved_logits_to_keep = (
        logits_to_keep.to(device=torch_device)
        if isinstance(logits_to_keep, torch.Tensor)
        else logits_to_keep
    )
    return replace(
        forward_inputs,
        input_ids=forward_inputs.input_ids.to(device=torch_device),
        position_ids=forward_inputs.position_ids.to(device=torch_device),
        pixel_values=forward_inputs.pixel_values.to(device=torch_device),
        image_grid_thw=forward_inputs.image_grid_thw.to(device=torch_device),
        fa2_varlen_plan=moved_fa2_plan,
        logits_to_keep=moved_logits_to_keep,
    )


class SynchronousForwardInputProvider:
    """Reference semantic path: builds forward inputs on demand, no thread."""

    def __init__(self) -> None:
        self._state = _StepBoundState()
        self._closed = False
        self.last_take_wait_seconds = 0.0

    def begin_planned_step(
        self, planned_step_id: int, moved_micro_steps: Sequence[Any]
    ) -> None:
        del planned_step_id
        _assert_not_closed(self._closed)
        self._state.begin(moved_micro_steps)

    def take(self, ordinal: int, micro_step: Any) -> QwenForwardInputs:
        expected_micro_step = self._state.validate_take(ordinal, micro_step)
        self._state.advance(ordinal)
        self.last_take_wait_seconds = 0.0
        # Build on CPU then move separately (matching the overlapped
        # provider's two-phase construction) so this reference path's
        # `total_build_inputs_ns` receipt means CPU-only build time too —
        # comparable to the overlapped provider's, not H2D-inclusive.
        forward_inputs = _build_forward_inputs(expected_micro_step, device=None)
        return _move_forward_inputs_to_device(
            forward_inputs, expected_micro_step.forward_device
        )

    def end_planned_step(self) -> None:
        self._state.end()

    def close(self) -> None:
        self._closed = True


def _cancellation_aware_acquire(
    slot: threading.Semaphore, cancel_event: threading.Event
) -> bool:
    while not cancel_event.is_set():
        if slot.acquire(timeout=_QUEUE_POLL_SECONDS):
            return True
    return False


def _cancellation_aware_put(
    item_queue: "queue.Queue[_PreparedItem]",
    item: _PreparedItem,
    cancel_event: threading.Event,
) -> bool:
    while not cancel_event.is_set():
        try:
            item_queue.put(item, timeout=_QUEUE_POLL_SECONDS)
            return True
        except queue.Full:
            continue
    return False


def _cancellation_aware_get(
    item_queue: "queue.Queue[_PreparedItem]",
    cancel_event: threading.Event,
) -> _PreparedItem | None:
    while True:
        try:
            return item_queue.get(timeout=_QUEUE_POLL_SECONDS)
        except queue.Empty:
            if cancel_event.is_set():
                return None
            continue


def _drain_queue(item_queue: "queue.Queue[_PreparedItem]") -> None:
    while True:
        try:
            item_queue.get_nowait()
        except queue.Empty:
            return


def _produce_forward_inputs(
    moved_micro_steps: tuple[Any, ...],
    item_queue: "queue.Queue[_PreparedItem]",
    cancel_event: threading.Event,
    build_slot: threading.Semaphore,
) -> None:
    for ordinal, micro_step in enumerate(moved_micro_steps):
        if cancel_event.is_set():
            return
        # Acquire the depth-one build slot BEFORE starting to build. The
        # consumer releases it immediately after dequeuing the previous
        # item, so this blocks until at most one built-but-unconsumed item
        # exists — the queue's maxsize=1 alone does not guarantee that,
        # since a fast producer could otherwise finish building ordinal
        # k+1 while ordinal k still sits in the queue.
        if not _cancellation_aware_acquire(build_slot, cancel_event):
            return
        if cancel_event.is_set():
            return
        try:
            forward_inputs = _build_forward_inputs(micro_step, device=None)
        except BaseException as exc:  # noqa: BLE001 - poison item preserves original type
            _cancellation_aware_put(
                item_queue,
                _PreparedItem(ordinal=ordinal, forward_inputs=None, error=exc),
                cancel_event,
            )
            return
        if not _cancellation_aware_put(
            item_queue,
            _PreparedItem(ordinal=ordinal, forward_inputs=forward_inputs, error=None),
            cancel_event,
        ):
            return


class OverlappedForwardInputProvider:
    """Depth-one CPU lookahead: one producer thread per planned step."""

    def __init__(self) -> None:
        self._state = _StepBoundState()
        self._closed = False
        self._queue: "queue.Queue[_PreparedItem] | None" = None
        self._cancel_event: threading.Event | None = None
        self._build_slot: threading.Semaphore | None = None
        self._thread: threading.Thread | None = None
        self.last_take_wait_seconds = 0.0

    def begin_planned_step(
        self, planned_step_id: int, moved_micro_steps: Sequence[Any]
    ) -> None:
        del planned_step_id
        _assert_not_closed(self._closed)
        self._state.begin(moved_micro_steps)
        item_queue: "queue.Queue[_PreparedItem]" = queue.Queue(maxsize=1)
        cancel_event = threading.Event()
        build_slot = threading.Semaphore(1)
        thread = threading.Thread(
            target=_produce_forward_inputs,
            args=(self._state.moved_micro_steps, item_queue, cancel_event, build_slot),
            name="coordexp-forward-input-provider",
            daemon=True,
        )
        self._queue = item_queue
        self._cancel_event = cancel_event
        self._build_slot = build_slot
        self._thread = thread
        thread.start()

    def take(self, ordinal: int, micro_step: Any) -> QwenForwardInputs:
        expected_micro_step = self._state.validate_take(ordinal, micro_step)
        item_queue = self._queue
        cancel_event = self._cancel_event
        build_slot = self._build_slot
        if item_queue is None or cancel_event is None or build_slot is None:
            raise RuntimeContractError(
                "forward input provider is missing its active queue",
                code="training.forward_input_provider_missing_queue",
                context={"ordinal": ordinal},
            )
        wait_start = time.monotonic()
        item = _cancellation_aware_get(item_queue, cancel_event)
        self.last_take_wait_seconds = max(0.0, time.monotonic() - wait_start)
        if item is None:
            raise RuntimeContractError(
                "forward input provider producer was cancelled before preparing the requested item",
                code="training.forward_input_provider_cancelled_before_take",
                context={"ordinal": ordinal},
            )
        # Release the build slot immediately after dequeuing — before the
        # H2D move below — so the producer may start building the next
        # ordinal while this item's device transfer/forward runs. This is
        # the depth-one bound: releasing any earlier would let the
        # producer build ahead of an unconsumed item.
        build_slot.release()
        if item.ordinal != ordinal:
            raise RuntimeContractError(
                "forward input provider produced an item for an unexpected ordinal",
                code="training.forward_input_provider_producer_ordinal_skew",
                context={"ordinal": ordinal, "produced_ordinal": item.ordinal},
            )
        self._state.advance(ordinal)
        if item.error is not None:
            raise item.error
        if item.forward_inputs is None:
            raise RuntimeContractError(
                "forward input provider produced neither prepared inputs nor an error",
                code="training.forward_input_provider_empty_item",
                context={"ordinal": ordinal},
            )
        return _move_forward_inputs_to_device(
            item.forward_inputs, expected_micro_step.forward_device
        )

    def end_planned_step(self) -> None:
        self._state.end()
        cancel_event = self._cancel_event
        item_queue = self._queue
        thread = self._thread
        if cancel_event is not None:
            cancel_event.set()
        if item_queue is not None:
            _drain_queue(item_queue)
        if thread is not None:
            thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                raise RuntimeContractError(
                    "forward input provider producer thread did not exit within the bounded join timeout",
                    code="training.forward_input_provider_join_timeout",
                    context={"timeout_seconds": _JOIN_TIMEOUT_SECONDS},
                )
        self._thread = None
        self._queue = None
        self._cancel_event = None
        self._build_slot = None

    def close(self) -> None:
        if self._state.active:
            self.end_planned_step()
        self._closed = True


__all__ = [
    "ForwardInputProvider",
    "OVERLAPPED_MODE",
    "OverlappedForwardInputProvider",
    "SYNCHRONOUS_MODE",
    "SynchronousForwardInputProvider",
    "build_forward_input_provider",
    "resolve_forward_input_provider_mode",
]

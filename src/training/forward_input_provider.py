"""Explicit forward-input preparation modes for supervised training.

Two providers implement the same planned-step-scoped lifecycle protocol:
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
The third strict mode, `legacy_fused`, deliberately builds no provider: a
`None` disposition leaves the existing trainer-owned fused device-direct
construction path active.

The build slot enforces the true depth-one bound: the producer must acquire
it before starting a build and it is released by the consumer immediately
after dequeuing (before the H2D move), so the producer can never have a
second fully-built item in existence while the first sits unconsumed — the
`queue.Queue(maxsize=1)` alone does not prevent that, since a fast producer
could otherwise finish building item k+1 while item k is still queued.
Before queue publication, the producer enforces both the exact logical bytes
of its CPU Qwen tensor payload and the current process lifetime max RSS. The
logical payload observable stays available for precise ownership accounting;
the process max-RSS check owns the separate 64-GiB Wave 5 stop contract,
including allocator and other process-resident memory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import os
import queue
import threading
import time
from typing import Any, Literal, Protocol

import torch

from src.artifacts.resources import collect_resource_snapshot
from src.common.errors import RuntimeContractError
from src.config.models import ForwardInputProviderMode
from src.qwen.forward import QwenForwardInputs, build_qwen_forward_inputs


_QUEUE_POLL_SECONDS = 0.05
_JOIN_TIMEOUT_SECONDS = 30.0
DEFAULT_RESIDENT_CPU_TENSOR_PAYLOAD_CEILING_BYTES = 64 * 1024**3
DEFAULT_PROCESS_MAX_RSS_CEILING_BYTES = 64 * 1024**3

_FORWARD_INPUT_PROVIDER_MODE_ENV = "COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE"
LEGACY_FUSED_MODE = "legacy_fused"
OVERLAPPED_MODE = "overlapped"
SYNCHRONOUS_MODE = "synchronous"
_FORWARD_INPUT_PROVIDER_MODES = frozenset(
    {LEGACY_FUSED_MODE, OVERLAPPED_MODE, SYNCHRONOUS_MODE}
)

ForwardInputProviderModeSource = Literal[
    "strict_config", "deprecated_environment_override"
]


@dataclass(frozen=True)
class ResolvedForwardInputProviderMode:
    """Strict mode plus the complete source needed by the run receipt."""

    configured_mode: ForwardInputProviderMode
    resolved_mode: ForwardInputProviderMode
    source: ForwardInputProviderModeSource
    environment_variable: str | None = None

    @property
    def is_semantic_override(self) -> bool:
        return self.resolved_mode != self.configured_mode

    @property
    def provider_disposition(self) -> Literal["none", "synchronous", "overlapped"]:
        if self.resolved_mode == LEGACY_FUSED_MODE:
            return "none"
        return self.resolved_mode

    @property
    def input_build_owner(
        self,
    ) -> Literal[
        "trainer_fused_device_direct",
        "provider_consumer_cpu",
        "provider_producer_cpu",
    ]:
        if self.resolved_mode == LEGACY_FUSED_MODE:
            return "trainer_fused_device_direct"
        if self.resolved_mode == SYNCHRONOUS_MODE:
            return "provider_consumer_cpu"
        return "provider_producer_cpu"

    @property
    def device_transfer_owner(
        self,
    ) -> Literal["trainer_fused_build", "provider_consumer"]:
        if self.resolved_mode == LEGACY_FUSED_MODE:
            return "trainer_fused_build"
        return "provider_consumer"

    @property
    def lookahead_depth(self) -> Literal[0, 1]:
        return 1 if self.resolved_mode == OVERLAPPED_MODE else 0

    def to_receipt_dict(self) -> dict[str, object]:
        return {
            "configured_mode": self.configured_mode,
            "resolved_mode": self.resolved_mode,
            "source": self.source,
            "environment_variable": self.environment_variable,
            "is_semantic_override": self.is_semantic_override,
            "provider_disposition": self.provider_disposition,
            "input_build_owner": self.input_build_owner,
            "device_transfer_owner": self.device_transfer_owner,
            "lookahead_depth": self.lookahead_depth,
        }


class ForwardInputProvider(Protocol):
    """Planned-step-scoped lifecycle owned by the trainer."""

    last_take_wait_seconds: float

    def begin_planned_step(
        self, planned_step_id: int, moved_micro_steps: Sequence[Any]
    ) -> None: ...

    def take(self, ordinal: int, micro_step: Any) -> QwenForwardInputs: ...

    def end_planned_step(self) -> None: ...

    def close(self) -> None: ...


def resolve_forward_input_provider_mode(
    configured_mode: ForwardInputProviderMode,
) -> ResolvedForwardInputProviderMode:
    """Resolve strict config plus the one bounded deprecated environment source.

    Callers must persist ``to_receipt_dict()`` rather than only the resolved
    string.  This makes a diagnostic environment override visible and prevents
    production assembly from silently treating it as authored strict config.
    """

    configured_mode = _validate_mode(configured_mode, source="strict_config")
    raw = os.environ.get(_FORWARD_INPUT_PROVIDER_MODE_ENV)
    if raw is None:
        return ResolvedForwardInputProviderMode(
            configured_mode=configured_mode,
            resolved_mode=configured_mode,
            source="strict_config",
        )
    resolved_mode = _validate_mode(
        raw,
        source="deprecated_environment_override",
        environment_variable=_FORWARD_INPUT_PROVIDER_MODE_ENV,
    )
    return ResolvedForwardInputProviderMode(
        configured_mode=configured_mode,
        resolved_mode=resolved_mode,
        source="deprecated_environment_override",
        environment_variable=_FORWARD_INPUT_PROVIDER_MODE_ENV,
    )


def _validate_mode(
    mode: str,
    *,
    source: ForwardInputProviderModeSource,
    environment_variable: str | None = None,
) -> ForwardInputProviderMode:
    if mode not in _FORWARD_INPUT_PROVIDER_MODES:
        raise RuntimeContractError(
            "forward input provider mode must be 'legacy_fused', 'overlapped', or "
            "'synchronous'",
            code="training.forward_input_provider_mode_invalid",
            context={
                "value": mode,
                "source": source,
                "environment_variable": environment_variable,
            },
        )
    return mode  # type: ignore[return-value]


def build_forward_input_provider(
    mode: ForwardInputProviderMode,
) -> ForwardInputProvider | None:
    mode = _validate_mode(mode, source="strict_config")
    if mode == LEGACY_FUSED_MODE:
        return None
    if mode == SYNCHRONOUS_MODE:
        return SynchronousForwardInputProvider()
    if mode == OVERLAPPED_MODE:
        return OverlappedForwardInputProvider()
    raise AssertionError(f"unhandled validated forward input provider mode: {mode}")


@dataclass(frozen=True)
class _PreparedItem:
    ordinal: int
    forward_inputs: QwenForwardInputs | None
    error: BaseException | None
    resident_cpu_tensor_payload_bytes: int
    process_max_rss_bytes: int | None


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
        if ordinal != self.next_ordinal or not (
            0 <= ordinal < len(self.moved_micro_steps)
        ):
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
        logits_to_keep_positions=micro_step.token_sequence.causal_logits_positions(),
        device=device,
        fa2_branch_proof_policy=micro_step.fa2_branch_proof_policy,
    )


def _resident_cpu_tensor_payload_bytes(
    forward_inputs: QwenForwardInputs,
) -> int:
    """Return exact logical bytes for provider-owned Qwen tensor payloads.

    This is deliberately narrower than process RSS: it accounts only the
    tensors owned by ``QwenForwardInputs`` that the provider may retain while
    preparing lookahead. Receipt metadata, Python-object overhead, allocator
    overhead, and consumer-owned tensors are outside this observable.
    """

    tensors = (
        ("input_ids", forward_inputs.input_ids),
        ("position_ids", forward_inputs.position_ids),
        ("pixel_values", forward_inputs.pixel_values),
        ("image_grid_thw", forward_inputs.image_grid_thw),
        (
            "fa2_varlen_plan.cu_seq_lens_q",
            forward_inputs.fa2_varlen_plan.cu_seq_lens_q,
        ),
        (
            "fa2_varlen_plan.cu_seq_lens_k",
            forward_inputs.fa2_varlen_plan.cu_seq_lens_k,
        ),
    )
    total_bytes = 0
    for field_path, tensor in tensors:
        total_bytes += _cpu_tensor_payload_bytes(tensor, field_path=field_path)
    if isinstance(forward_inputs.logits_to_keep, torch.Tensor):
        total_bytes += _cpu_tensor_payload_bytes(
            forward_inputs.logits_to_keep,
            field_path="logits_to_keep",
        )
    return total_bytes


def _cpu_tensor_payload_bytes(tensor: torch.Tensor, *, field_path: str) -> int:
    if tensor.device.type != "cpu":
        raise RuntimeContractError(
            "forward input provider producer may retain only CPU tensor payloads",
            code="training.forward_input_provider_producer_tensor_device_invalid",
            context={
                "field": field_path,
                "device_type": tensor.device.type,
            },
        )
    return int(tensor.numel()) * int(tensor.element_size())


def _validate_resident_cpu_tensor_payload_ceiling(
    forward_inputs: QwenForwardInputs,
    *,
    ordinal: int,
    ceiling_bytes: int,
) -> int:
    payload_bytes = _resident_cpu_tensor_payload_bytes(forward_inputs)
    if payload_bytes > ceiling_bytes:
        raise RuntimeContractError(
            "forward input provider prepared CPU tensor payload exceeds its bounded ceiling",
            code="training.forward_input_provider_host_tensor_payload_ceiling_exceeded",
            context={
                "ordinal": ordinal,
                "pack_index": int(forward_inputs.pack_index),
                "payload_bytes": payload_bytes,
                "ceiling_bytes": ceiling_bytes,
            },
        )
    return payload_bytes


def _read_current_process_max_rss_bytes() -> object:
    """Read the canonical current-process lifetime max RSS measurement."""

    snapshot = collect_resource_snapshot(cuda_api=None)
    cpu = snapshot.get("cpu")
    if not isinstance(cpu, Mapping):
        return {"status": "unavailable", "reason": "resource_cpu_snapshot_invalid"}
    return cpu.get(
        "max_rss_bytes",
        {"status": "unavailable", "reason": "resource_max_rss_missing"},
    )


def _validate_process_max_rss_ceiling(
    *,
    ordinal: int,
    pack_index: int,
    ceiling_bytes: int,
    reader: Callable[[], object],
) -> int:
    try:
        observed = reader()
    except Exception as exc:
        raise RuntimeContractError(
            "forward input provider cannot enforce its current-process max-RSS ceiling",
            code="training.forward_input_provider_process_max_rss_unavailable",
            context={
                "ordinal": ordinal,
                "pack_index": pack_index,
                "ceiling_bytes": ceiling_bytes,
                "reason": "reader_error",
                "error_type": type(exc).__name__,
            },
        ) from exc

    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
        reason = "invalid_measurement"
        if isinstance(observed, Mapping):
            status = observed.get("status")
            observed_reason = observed.get("reason")
            if status == "unavailable" and isinstance(observed_reason, str):
                reason = observed_reason
        raise RuntimeContractError(
            "forward input provider cannot enforce its current-process max-RSS ceiling",
            code="training.forward_input_provider_process_max_rss_unavailable",
            context={
                "ordinal": ordinal,
                "pack_index": pack_index,
                "ceiling_bytes": ceiling_bytes,
                "reason": reason,
            },
        )
    if observed > ceiling_bytes:
        raise RuntimeContractError(
            "forward input provider current-process max RSS exceeds its bounded ceiling",
            code="training.forward_input_provider_process_max_rss_ceiling_exceeded",
            context={
                "ordinal": ordinal,
                "pack_index": pack_index,
                "process_max_rss_bytes": observed,
                "ceiling_bytes": ceiling_bytes,
            },
        )
    return observed


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

    lookahead_depth = 0
    max_prepared_items = 0

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
    resident_cpu_tensor_payload_ceiling_bytes: int,
    process_max_rss_ceiling_bytes: int,
    process_max_rss_reader: Callable[[], object],
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
            resident_cpu_tensor_payload_bytes = (
                _validate_resident_cpu_tensor_payload_ceiling(
                    forward_inputs,
                    ordinal=ordinal,
                    ceiling_bytes=resident_cpu_tensor_payload_ceiling_bytes,
                )
            )
            process_max_rss_bytes = _validate_process_max_rss_ceiling(
                ordinal=ordinal,
                pack_index=int(forward_inputs.pack_index),
                ceiling_bytes=process_max_rss_ceiling_bytes,
                reader=process_max_rss_reader,
            )
        except BaseException as exc:  # noqa: BLE001 - poison item preserves original type
            _cancellation_aware_put(
                item_queue,
                _PreparedItem(
                    ordinal=ordinal,
                    forward_inputs=None,
                    error=exc,
                    resident_cpu_tensor_payload_bytes=0,
                    process_max_rss_bytes=None,
                ),
                cancel_event,
            )
            return
        if not _cancellation_aware_put(
            item_queue,
            _PreparedItem(
                ordinal=ordinal,
                forward_inputs=forward_inputs,
                error=None,
                resident_cpu_tensor_payload_bytes=resident_cpu_tensor_payload_bytes,
                process_max_rss_bytes=process_max_rss_bytes,
            ),
            cancel_event,
        ):
            return


class OverlappedForwardInputProvider:
    """Depth-one, payload- and process-RSS-bounded CPU lookahead."""

    lookahead_depth = 1
    max_prepared_items = 1

    def __init__(
        self,
        *,
        _resident_cpu_tensor_payload_ceiling_bytes: int | None = None,
        _process_max_rss_ceiling_bytes: int | None = None,
        _process_max_rss_reader: Callable[[], object] | None = None,
    ) -> None:
        payload_ceiling_bytes = (
            DEFAULT_RESIDENT_CPU_TENSOR_PAYLOAD_CEILING_BYTES
            if _resident_cpu_tensor_payload_ceiling_bytes is None
            else _resident_cpu_tensor_payload_ceiling_bytes
        )
        if isinstance(payload_ceiling_bytes, bool) or not isinstance(
            payload_ceiling_bytes, int
        ):
            raise TypeError("resident CPU tensor payload ceiling must be an integer")
        if payload_ceiling_bytes <= 0:
            raise ValueError("resident CPU tensor payload ceiling must be positive")
        process_rss_ceiling_bytes = (
            DEFAULT_PROCESS_MAX_RSS_CEILING_BYTES
            if _process_max_rss_ceiling_bytes is None
            else _process_max_rss_ceiling_bytes
        )
        if isinstance(process_rss_ceiling_bytes, bool) or not isinstance(
            process_rss_ceiling_bytes, int
        ):
            raise TypeError("process max-RSS ceiling must be an integer")
        if process_rss_ceiling_bytes <= 0:
            raise ValueError("process max-RSS ceiling must be positive")
        process_max_rss_reader = (
            _read_current_process_max_rss_bytes
            if _process_max_rss_reader is None
            else _process_max_rss_reader
        )
        if not callable(process_max_rss_reader):
            raise TypeError("process max-RSS reader must be callable")
        self._state = _StepBoundState()
        self._closed = False
        self.resident_cpu_tensor_payload_ceiling_bytes = payload_ceiling_bytes
        self.process_max_rss_ceiling_bytes = process_rss_ceiling_bytes
        self._process_max_rss_reader = process_max_rss_reader
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
            args=(
                self._state.moved_micro_steps,
                item_queue,
                cancel_event,
                build_slot,
                self.resident_cpu_tensor_payload_ceiling_bytes,
                self.process_max_rss_ceiling_bytes,
                self._process_max_rss_reader,
            ),
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
    "DEFAULT_PROCESS_MAX_RSS_CEILING_BYTES",
    "DEFAULT_RESIDENT_CPU_TENSOR_PAYLOAD_CEILING_BYTES",
    "ForwardInputProvider",
    "LEGACY_FUSED_MODE",
    "OVERLAPPED_MODE",
    "OverlappedForwardInputProvider",
    "ResolvedForwardInputProviderMode",
    "SYNCHRONOUS_MODE",
    "SynchronousForwardInputProvider",
    "build_forward_input_provider",
    "resolve_forward_input_provider_mode",
]

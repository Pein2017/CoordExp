from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import src.training.forward_input_provider as forward_input_provider_module
from src.common.errors import QwenForwardContractError, RuntimeContractError
from src.config.models import RuntimeBatchResolution
from src.packing.planner import plan_packed_sequences
from src.qwen.forward import QwenForwardInputs, build_qwen_forward_inputs
from src.qwen.positions import build_qwen_position_inputs
from src.runtime import GateDecision
from src.training.forward_input_provider import (
    OverlappedForwardInputProvider,
    SynchronousForwardInputProvider,
    build_forward_input_provider,
)
from src.training.schedule import ResolvedStepSchedule
from src.training.supervised_trainer import SupervisedMicroStep, SupervisedTrainer


IMAGE_TOKEN_ID = 151655


def _make_micro_step(index: int, *, forward_device: Any = None) -> SupervisedMicroStep:
    examples = (
        FakeEncodedExample(
            example_id=f"ex-{index}",
            input_ids=(10, 11, 12, *([IMAGE_TOKEN_ID] * 6), 13, 14),
            image_pad_physical_start=3,
            image_pad_physical_end=9,
            image_encoding=_fake_image_encoding(fill=float(index) + 1.0),
        ),
    )
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    return SupervisedMicroStep(
        pack=pack,
        encoded_examples=examples,
        position_inputs=positions,
        token_sequence=SimpleNamespace(atoms=None),
        vocab_groups=None,
        forward_device=forward_device,
    )


def _fake_image_encoding(*, fill: float) -> "FakeImageEncoding":
    grid = (1, 4, 6)
    return FakeImageEncoding(
        image_grid_thw=grid,
        pixel_values=torch.full((grid[0] * grid[1] * grid[2], 8), fill),
        plan=FakeImagePlan(merge_size=2),
    )


@dataclass(frozen=True)
class FakeEncodedExample:
    example_id: str
    input_ids: tuple[int, ...]
    image_pad_physical_start: int
    image_pad_physical_end: int
    image_encoding: "FakeImageEncoding"


@dataclass(frozen=True)
class FakeImageEncoding:
    image_grid_thw: tuple[int, int, int]
    pixel_values: torch.Tensor
    plan: "FakeImagePlan"


@dataclass(frozen=True)
class FakeImagePlan:
    merge_size: int


def _wait_until(predicate: Any, *, timeout: float = 5.0, interval: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met within bounded timeout")


# (a) prepared inputs byte-identical to synchronous construction.
def test_synchronous_and_overlapped_providers_produce_byte_identical_forward_inputs() -> None:
    micro_step = _make_micro_step(0)
    reference = build_qwen_forward_inputs(
        micro_step.pack, micro_step.encoded_examples, micro_step.position_inputs
    )

    sync_provider = SynchronousForwardInputProvider()
    overlapped_provider = OverlappedForwardInputProvider()
    try:
        sync_provider.begin_planned_step(1, (micro_step,))
        sync_inputs = sync_provider.take(0, micro_step)
        sync_provider.end_planned_step()

        overlapped_provider.begin_planned_step(1, (micro_step,))
        overlapped_inputs = overlapped_provider.take(0, micro_step)
        overlapped_provider.end_planned_step()
    finally:
        sync_provider.close()
        overlapped_provider.close()

    for candidate in (sync_inputs, overlapped_inputs):
        assert torch.equal(candidate.input_ids, reference.input_ids)
        assert torch.equal(candidate.position_ids, reference.position_ids)
        assert torch.equal(candidate.pixel_values, reference.pixel_values)
        assert torch.equal(candidate.image_grid_thw, reference.image_grid_thw)
        assert torch.equal(
            candidate.fa2_varlen_plan.cu_seq_lens_q, reference.fa2_varlen_plan.cu_seq_lens_q
        )
        assert torch.equal(
            candidate.fa2_varlen_plan.cu_seq_lens_k, reference.fa2_varlen_plan.cu_seq_lens_k
        )
    assert sync_inputs.input_ids.device.type == "cpu"
    assert overlapped_inputs.input_ids.device.type == "cpu"


# (b) producer materialization failure surfaces at the exact affected ordinal
# with the original error type; earlier ordinals complete normally.
def test_overlapped_provider_surfaces_producer_exception_at_the_affected_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    micro_step_0 = _make_micro_step(0)
    micro_step_1 = _make_micro_step(1)
    real_build = forward_input_provider_module.build_qwen_forward_inputs

    class SyntheticProducerFailure(QwenForwardContractError):
        pass

    def flaky_build(pack: Any, encoded_examples: Any, position_inputs: Any, **kwargs: Any) -> Any:
        if encoded_examples[0].example_id == micro_step_1.encoded_examples[0].example_id:
            raise SyntheticProducerFailure(
                "synthetic producer failure", code="test.synthetic_producer_failure"
            )
        return real_build(pack, encoded_examples, position_inputs, **kwargs)

    monkeypatch.setattr(forward_input_provider_module, "build_qwen_forward_inputs", flaky_build)

    provider = OverlappedForwardInputProvider()
    try:
        provider.begin_planned_step(1, (micro_step_0, micro_step_1))
        first = provider.take(0, micro_step_0)
        assert isinstance(first, QwenForwardInputs)
        with pytest.raises(SyntheticProducerFailure):
            provider.take(1, micro_step_1)
    finally:
        provider.close()


# (c) consumer error mid-step: end_planned_step after only partial
# consumption must cancel and join the producer thread cleanly.
def test_overlapped_provider_end_planned_step_after_partial_consumption_joins_cleanly() -> None:
    steps = tuple(_make_micro_step(index) for index in range(3))
    provider = OverlappedForwardInputProvider()
    provider.begin_planned_step(1, steps)
    first = provider.take(0, steps[0])
    assert isinstance(first, QwenForwardInputs)

    start = time.monotonic()
    provider.end_planned_step()
    elapsed = time.monotonic() - start

    assert elapsed < 5.0
    assert provider._thread is None
    provider.close()


# (d) finite-gate early break discards prepared-but-unconsumed items; the
# next planned step starts clean with no stale or skewed item.
def test_overlapped_provider_discards_unconsumed_items_and_starts_next_step_clean() -> None:
    provider = OverlappedForwardInputProvider()
    try:
        first_steps = tuple(_make_micro_step(index) for index in range(3))
        provider.begin_planned_step(1, first_steps)
        provider.take(0, first_steps[0])
        provider.take(1, first_steps[1])
        # Finite-gate early break: ordinal 2 is never consumed.
        provider.end_planned_step()

        # Scheduled eval/checkpoint handlers would run here with the
        # provider idle; there is nothing to call on the provider itself.

        second_steps = tuple(_make_micro_step(index) for index in range(2))
        provider.begin_planned_step(2, second_steps)
        second_first = provider.take(0, second_steps[0])
        second_second = provider.take(1, second_steps[1])
        assert isinstance(second_first, QwenForwardInputs)
        assert isinstance(second_second, QwenForwardInputs)
        provider.end_planned_step()
    finally:
        provider.close()


# (e) a full depth-one queue must not deadlock shutdown.
def test_overlapped_provider_end_planned_step_does_not_deadlock_on_a_full_queue() -> None:
    steps = tuple(_make_micro_step(index) for index in range(3))
    provider = OverlappedForwardInputProvider()
    provider.begin_planned_step(1, steps)
    # Deliberately never call take(): the producer fills the depth-one
    # queue with ordinal 0, then blocks trying to acquire the build slot
    # for ordinal 1 (never released, since no consumer ever dequeues
    # ordinal 0) — not blocked on put(), which succeeds immediately once
    # the slot is held (see the P1-1 depth-one build-slot fix).
    _wait_until(lambda: provider._queue is not None and provider._queue.qsize() >= 1)

    start = time.monotonic()
    provider.end_planned_step()
    elapsed = time.monotonic() - start

    assert elapsed < 5.0
    provider.close()


# (f) ordinal and pack-identity skew both fail closed.
def test_provider_take_fails_closed_on_ordinal_skew() -> None:
    steps = tuple(_make_micro_step(index) for index in range(2))
    for provider in (SynchronousForwardInputProvider(), OverlappedForwardInputProvider()):
        provider.begin_planned_step(1, steps)
        with pytest.raises(RuntimeContractError) as exc_info:
            provider.take(1, steps[1])  # ordinal 0 is expected first
        assert exc_info.value.code == "training.forward_input_provider_ordinal_skew"
        provider.close()


def test_provider_take_fails_closed_on_pack_identity_skew() -> None:
    steps = tuple(_make_micro_step(index) for index in range(2))
    wrong_micro_step = SimpleNamespace(pack=SimpleNamespace(pack_index=999))
    for provider in (SynchronousForwardInputProvider(), OverlappedForwardInputProvider()):
        provider.begin_planned_step(1, steps)
        with pytest.raises(RuntimeContractError) as exc_info:
            provider.take(0, wrong_micro_step)
        assert exc_info.value.code == "training.forward_input_provider_pack_skew"
        provider.close()


# (g) at most one prepared item beyond the executing step is ever resident.
def test_overlapped_provider_gates_next_build_until_prior_item_is_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decisive depth-one proof: the build slot (not just queue maxsize=1)
    prevents the producer from starting ordinal k+1's build while ordinal
    k's built item is still queued/unconsumed. A queue.Queue(maxsize=1)
    alone does NOT guarantee this — a fast producer can finish building
    k+1 and block only on put() while k sits in the queue, momentarily
    holding two built items. This test fails against that older shape: it
    asserts the SECOND build has not even started after a bounded wait,
    before any `take()` call is made.
    """
    steps = tuple(_make_micro_step(index) for index in range(3))
    build_started: list[int] = []
    real_build = forward_input_provider_module._build_forward_inputs

    def recording_build(micro_step: Any, *, device: Any) -> Any:
        example_id = micro_step.encoded_examples[0].example_id
        build_started.append(int(example_id.split("-")[-1]))
        return real_build(micro_step, device=device)

    monkeypatch.setattr(forward_input_provider_module, "_build_forward_inputs", recording_build)

    provider = OverlappedForwardInputProvider()
    try:
        provider.begin_planned_step(1, steps)
        _wait_until(lambda: len(build_started) >= 1)
        # Give the producer ample opportunity to race ahead if the
        # depth-one bound were not enforced by the build slot.
        time.sleep(0.3)
        assert build_started == [0], (
            "ordinal 1's build must not start while ordinal 0's built item "
            "is still queued/unconsumed"
        )

        first = provider.take(0, steps[0])
        assert isinstance(first, QwenForwardInputs)
        _wait_until(lambda: len(build_started) >= 2)
        time.sleep(0.3)
        assert build_started == [0, 1], (
            "ordinal 2's build must not start while ordinal 1's built item "
            "is still queued/unconsumed"
        )

        second = provider.take(1, steps[1])
        assert isinstance(second, QwenForwardInputs)
        _wait_until(lambda: len(build_started) >= 3)
        assert build_started == [0, 1, 2]

        third = provider.take(2, steps[2])
        assert isinstance(third, QwenForwardInputs)
        provider.end_planned_step()
    finally:
        provider.close()


def test_overlapped_provider_built_minus_taken_never_exceeds_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same depth-one bound, expressed as the invariant
    `builds_completed - items_dequeued <= 1`, sampled continuously from a
    background watcher thread across a full 5-micro-step planned step with
    no artificial delay — proving the bound holds under ordinary timing,
    not just at the one instant the other test inspects. `items_dequeued`
    is measured at the exact internal dequeue point (`_cancellation_aware_get`
    returning a real item), the same event that triggers the build-slot
    release — not after the caller's `take()` (which also performs the
    H2D move) returns, which would race against the producer and produce
    false positives unrelated to the actual depth-one bound."""

    steps = tuple(_make_micro_step(index) for index in range(5))
    builds_completed = 0
    items_dequeued = 0
    lock = threading.Lock()
    real_build = forward_input_provider_module._build_forward_inputs
    real_get = forward_input_provider_module._cancellation_aware_get
    violations: list[tuple[int, int]] = []
    watch_stop = threading.Event()

    def recording_build(micro_step: Any, *, device: Any) -> Any:
        nonlocal builds_completed
        result = real_build(micro_step, device=device)
        with lock:
            builds_completed += 1
        return result

    def recording_get(item_queue: Any, cancel_event: Any) -> Any:
        nonlocal items_dequeued
        item = real_get(item_queue, cancel_event)
        if item is not None:
            with lock:
                items_dequeued += 1
        return item

    monkeypatch.setattr(forward_input_provider_module, "_build_forward_inputs", recording_build)
    monkeypatch.setattr(forward_input_provider_module, "_cancellation_aware_get", recording_get)

    def watch() -> None:
        while not watch_stop.is_set():
            with lock:
                outstanding = builds_completed - items_dequeued
                if outstanding > 1:
                    violations.append((builds_completed, items_dequeued))
            time.sleep(0.002)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    provider = OverlappedForwardInputProvider()
    try:
        provider.begin_planned_step(1, steps)
        for ordinal, step in enumerate(steps):
            provider.take(ordinal, step)
        provider.end_planned_step()
    finally:
        watch_stop.set()
        watcher.join(timeout=5.0)
        provider.close()

    assert violations == []
    assert builds_completed == 5
    assert items_dequeued == 5


# Lifecycle state-machine contract errors: idle -> active(step) -> idle,
# entered only by begin/end; a second begin without end, take outside
# active, and use after close all fail closed.
def test_provider_lifecycle_state_machine_fails_closed_on_misuse() -> None:
    for provider in (SynchronousForwardInputProvider(), OverlappedForwardInputProvider()):
        micro_step = _make_micro_step(0)
        with pytest.raises(RuntimeContractError) as exc_info:
            provider.take(0, micro_step)
        assert exc_info.value.code == "training.forward_input_provider_not_active"

        provider.begin_planned_step(1, (micro_step,))
        with pytest.raises(RuntimeContractError) as exc_info:
            provider.begin_planned_step(2, (micro_step,))
        assert exc_info.value.code == "training.forward_input_provider_already_active"
        provider.take(0, micro_step)
        provider.end_planned_step()

        with pytest.raises(RuntimeContractError) as exc_info:
            provider.end_planned_step()
        assert exc_info.value.code == "training.forward_input_provider_not_active"

        provider.close()
        with pytest.raises(RuntimeContractError) as exc_info:
            provider.begin_planned_step(3, (micro_step,))
        assert exc_info.value.code == "training.forward_input_provider_closed"


def test_provider_close_is_idempotent_for_both_modes() -> None:
    for provider in (SynchronousForwardInputProvider(), OverlappedForwardInputProvider()):
        provider.close()
        provider.close()  # must not raise


def test_overlapped_provider_leaves_no_leaked_thread_after_close() -> None:
    steps = tuple(_make_micro_step(index) for index in range(2))
    baseline = threading.active_count()
    provider = OverlappedForwardInputProvider()
    provider.begin_planned_step(1, steps)
    provider.take(0, steps[0])
    provider.close()
    assert threading.active_count() == baseline


# Producer MUST never perform CUDA/H2D work: it always builds on CPU
# (device=None) regardless of the micro-step's eventual target device.
def test_overlapped_provider_producer_always_builds_on_cpu_regardless_of_target_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    micro_step = _make_micro_step(0, forward_device="cuda:7")
    observed_devices: list[Any] = []
    real_build = forward_input_provider_module.build_qwen_forward_inputs

    def recording_build(
        pack: Any, encoded_examples: Any, position_inputs: Any, *, device: Any = None, **kwargs: Any
    ) -> Any:
        observed_devices.append(device)
        return real_build(pack, encoded_examples, position_inputs, device=device, **kwargs)

    monkeypatch.setattr(forward_input_provider_module, "build_qwen_forward_inputs", recording_build)

    provider = OverlappedForwardInputProvider()
    provider.begin_planned_step(1, (micro_step,))
    _wait_until(lambda: len(observed_devices) >= 1)
    provider.end_planned_step()
    provider.close()

    assert observed_devices == [None]


# P2-A: both provider modes must build on CPU and move separately, so
# `total_build_inputs_ns` (feeding `input_build_seconds`) means the same
# thing — CPU-only construction — in both modes and is comparable.
def test_both_provider_modes_build_on_cpu_and_move_separately_for_comparable_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    micro_step = _make_micro_step(0)  # forward_device=None (cpu)
    observed_build_devices: list[Any] = []
    observed_move_devices: list[Any] = []
    real_build = forward_input_provider_module._build_forward_inputs
    real_move = forward_input_provider_module._move_forward_inputs_to_device

    def recording_build(step: Any, *, device: Any) -> Any:
        observed_build_devices.append(device)
        return real_build(step, device=device)

    def recording_move(forward_inputs: Any, device: Any) -> Any:
        observed_move_devices.append(device)
        return real_move(forward_inputs, device)

    monkeypatch.setattr(forward_input_provider_module, "_build_forward_inputs", recording_build)
    monkeypatch.setattr(
        forward_input_provider_module, "_move_forward_inputs_to_device", recording_move
    )

    for provider in (SynchronousForwardInputProvider(), OverlappedForwardInputProvider()):
        try:
            provider.begin_planned_step(1, (micro_step,))
            forward_inputs = provider.take(0, micro_step)
            assert isinstance(forward_inputs, QwenForwardInputs)
            provider.end_planned_step()
        finally:
            provider.close()

    # Every build call was CPU-only (device=None); every provider mode also
    # performed a separate move call — the H2D/device-move step is real
    # work that both modes share identically, never folded into build time
    # for one mode but not the other.
    assert observed_build_devices == [None, None]
    assert observed_move_devices == [None, None]


# P2-C: prove the device move actually reaches every tensor field that
# `to_model_kwargs()` sends to the model — using torch's "meta" device
# (no real CUDA required) so this fails if any field is left on CPU.
def test_move_to_device_covers_every_tensor_field_used_by_to_model_kwargs() -> None:
    micro_step = _make_micro_step(0, forward_device="meta")
    for provider in (SynchronousForwardInputProvider(), OverlappedForwardInputProvider()):
        try:
            provider.begin_planned_step(1, (micro_step,))
            forward_inputs = provider.take(0, micro_step)
        finally:
            provider.end_planned_step()
            provider.close()

        model_kwargs = forward_inputs.to_model_kwargs()
        tensor_fields = {
            key: value for key, value in model_kwargs.items() if isinstance(value, torch.Tensor)
        }
        assert tensor_fields, "sanity: to_model_kwargs must expose at least one tensor field"
        for key, tensor in tensor_fields.items():
            assert tensor.device.type == "meta", (
                f"{type(provider).__name__}: to_model_kwargs()[{key!r}] was not moved to "
                f"the target device (still {tensor.device.type})"
            )


def test_resolve_forward_input_provider_mode_defaults_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE", raising=False)
    assert forward_input_provider_module.resolve_forward_input_provider_mode() == "synchronous"

    monkeypatch.setenv("COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE", "overlapped")
    assert forward_input_provider_module.resolve_forward_input_provider_mode() == "overlapped"

    monkeypatch.setenv("COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE", "bogus")
    with pytest.raises(RuntimeContractError) as exc_info:
        forward_input_provider_module.resolve_forward_input_provider_mode()
    assert exc_info.value.code == "training.forward_input_provider_mode_invalid"


# --- M3 loss-stream equivalence: a fixed multi-step run through the real
# providers and real build_qwen_forward_inputs/run_qwen_forward, with a
# deterministic echo model whose logits are a function of the actual
# constructed tensors. Any divergence in provider-built inputs between
# modes would change the loss stream; this proves there is none.


class _EchoModel:
    def __init__(self, *, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=vocab_size))

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        input_ids = kwargs["input_ids"]
        pixel_values = kwargs["pixel_values"]
        logits_to_keep = kwargs["logits_to_keep"]
        seq_length = (
            int(logits_to_keep.numel())
            if isinstance(logits_to_keep, torch.Tensor)
            else int(input_ids.shape[1])
        )
        value = float(input_ids.sum()) + float(pixel_values.sum())
        logits = torch.full((1, seq_length, self.vocab_size), value, dtype=torch.float32)
        return SimpleNamespace(logits=logits, loss=None, past_key_values=None, rope_deltas=None)


class _EchoLossBundle:
    def __init__(self, total_loss: torch.Tensor) -> None:
        self.total_loss = total_loss

    def to_artifact_dict(self) -> dict[str, float]:
        return {"total_loss": float(self.total_loss.detach().cpu())}


class _EchoStreamingLossRunner:
    def prepare_planned_step(self, moved_micro_steps: tuple[Any, ...]) -> dict[str, int]:
        return {"micro_step_count": len(moved_micro_steps)}

    def compute_micro_step(
        self, context: dict[str, torch.Tensor], plan: dict[str, int], *, local_micro_step_index: int
    ) -> _EchoLossBundle:
        del plan, local_micro_step_index
        return _EchoLossBundle(total_loss=context["logits"].sum())

    def finalize_planned_step(
        self, micro_loss_artifacts: tuple[dict[str, float], ...], plan: dict[str, int]
    ) -> dict[str, Any]:
        del plan
        total = sum(item["total_loss"] for item in micro_loss_artifacts)
        return {"total_loss": total, "metrics": {"loss/total": total}, "diagnostics": {}}


class _EchoRuntime:
    """Deterministic runtime double: flags specific consumption-order call
    indices as an unsafe pre-backward gate to force a synthetic finite-gate
    early break, independent of provider mode."""

    def __init__(self, *, unsafe_call_indices: set[int] | None = None) -> None:
        self.unsafe_call_indices = unsafe_call_indices or set()
        self._call_index = 0

    def move_micro_step(
        self, micro_step: Any, *, planned_step_id: int, local_micro_step_index: int
    ) -> Any:
        del planned_step_id, local_micro_step_index
        return micro_step

    def accumulation_context(self, *, sync_gradients: bool) -> Any:
        del sync_gradients
        return nullcontext()

    def pre_backward(self, bundle: Any, *, planned_step_id: int) -> GateDecision:
        del bundle
        call_index = self._call_index
        self._call_index += 1
        unsafe = call_index in self.unsafe_call_indices
        return GateDecision(
            stage="pre_backward_scalar",
            planned_step_id=planned_step_id,
            world_size=1,
            ranks=(0,),
            all_ranks_safe=not unsafe,
            should_call_backward=not unsafe,
            should_call_optimizer_step=False,
            should_clear_gradients=unsafe,
            optimizer_update_status="skipped_non_finite_scalar" if unsafe else "pending_backward",
            finite_status="non_finite" if unsafe else "finite",
            reason_codes=(),
            rank_diagnostics=(),
            diagnostics={},
        )

    def backward(
        self, loss: torch.Tensor, *, planned_step_id: int, sync_gradients: bool = True
    ) -> None:
        del loss, planned_step_id, sync_gradients

    def post_backward(self, *, planned_step_id: int) -> GateDecision:
        return GateDecision(
            stage="post_backward",
            planned_step_id=planned_step_id,
            world_size=1,
            ranks=(0,),
            all_ranks_safe=True,
            should_call_backward=False,
            should_call_optimizer_step=True,
            should_clear_gradients=False,
            optimizer_update_status="applied",
            finite_status="finite",
            reason_codes=(),
            rank_diagnostics=(),
            diagnostics={},
        )

    def clip_gradients(self, *, planned_step_id: int) -> None:
        del planned_step_id

    def optimizer_step(self, *, planned_step_id: int) -> None:
        del planned_step_id

    def scheduler_step(self, *, planned_step_id: int) -> None:
        del planned_step_id
        return None

    def zero_gradients(self, *, planned_step_id: int) -> None:
        del planned_step_id


def _echo_schedule(*, resolved_max_steps: int, grad_accum_steps: int) -> ResolvedStepSchedule:
    return ResolvedStepSchedule(
        resolved_max_steps=resolved_max_steps,
        packs_per_epoch=100,
        requested_pack_presentations=resolved_max_steps * grad_accum_steps,
        actual_pack_presentations=resolved_max_steps * grad_accum_steps,
        tail_fill_pack_count=0,
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=grad_accum_steps,
            resolved_grad_accum_steps=grad_accum_steps,
        ),
        events={"checkpoint": (), "eval.forward": (), "final": ()},
    )


def _run_echo_stream(
    provider_mode: str,
    micro_steps: tuple[SupervisedMicroStep, ...],
    *,
    grad_accum_steps: int,
    unsafe_call_indices: set[int] | None = None,
) -> list[Any]:
    provider = build_forward_input_provider(provider_mode)
    observations: list[Any] = []
    trainer = SupervisedTrainer(
        model=_EchoModel(vocab_size=5),
        schedule=_echo_schedule(
            resolved_max_steps=len(micro_steps) // grad_accum_steps,
            grad_accum_steps=grad_accum_steps,
        ),
        pack_stream=iter(micro_steps),
        loss_context_factory=lambda micro_step, forward_result: {"logits": forward_result.logits},
        loss_runner=_EchoStreamingLossRunner(),
        runtime=_EchoRuntime(unsafe_call_indices=unsafe_call_indices),
        on_completed_step=observations.append,
        forward_input_provider=provider,
    )
    try:
        trainer.run()
    finally:
        provider.close()
    return observations


def _echo_micro_steps(count: int) -> tuple[SupervisedMicroStep, ...]:
    return tuple(
        SupervisedMicroStep(
            pack=step.pack,
            encoded_examples=step.encoded_examples,
            position_inputs=step.position_inputs,
            token_sequence=step.token_sequence,
            vocab_groups=step.vocab_groups,
            expected_vocab_size=5,
        )
        for step in (_make_micro_step(index) for index in range(count))
    )


def test_loss_stream_and_gate_decisions_are_identical_between_provider_modes_on_a_fixed_run() -> None:
    micro_steps = _echo_micro_steps(4)  # a fixed 2-step run, grad_accum=2

    sync_observations = _run_echo_stream("synchronous", micro_steps, grad_accum_steps=2)
    overlapped_observations = _run_echo_stream("overlapped", micro_steps, grad_accum_steps=2)

    assert len(sync_observations) == len(overlapped_observations) == 2
    for sync_obs, overlapped_obs in zip(sync_observations, overlapped_observations, strict=True):
        assert sync_obs.loss_bundle_artifact == overlapped_obs.loss_bundle_artifact
        assert sync_obs.optimizer_update_status == overlapped_obs.optimizer_update_status
        assert sync_obs.finite_status == overlapped_obs.finite_status
        assert sync_obs.micro_step_count == overlapped_obs.micro_step_count


def test_loss_stream_and_gate_decisions_are_identical_between_provider_modes_on_a_synthetic_early_break() -> None:
    micro_steps = _echo_micro_steps(4)  # a fixed 2-step run, grad_accum=2
    # Call index 0 is the first planned step's first (and only consumed)
    # micro-step: force its pre-backward gate unsafe to trigger a
    # finite-gate early break identically in both provider modes.
    unsafe_call_indices = {0}

    sync_observations = _run_echo_stream(
        "synchronous", micro_steps, grad_accum_steps=2, unsafe_call_indices=unsafe_call_indices
    )
    overlapped_observations = _run_echo_stream(
        "overlapped", micro_steps, grad_accum_steps=2, unsafe_call_indices=unsafe_call_indices
    )

    assert len(sync_observations) == len(overlapped_observations) == 2
    first_sync, first_overlapped = sync_observations[0], overlapped_observations[0]
    assert first_sync.optimizer_update_status == "skipped_non_finite_scalar"
    assert first_overlapped.optimizer_update_status == "skipped_non_finite_scalar"
    assert first_sync.micro_step_count == first_overlapped.micro_step_count == 1
    for sync_obs, overlapped_obs in zip(sync_observations, overlapped_observations, strict=True):
        assert sync_obs.loss_bundle_artifact == overlapped_obs.loss_bundle_artifact
        assert sync_obs.optimizer_update_status == overlapped_obs.optimizer_update_status
        assert sync_obs.finite_status == overlapped_obs.finite_status


def test_build_forward_input_provider_selects_implementation_by_mode() -> None:
    assert isinstance(
        forward_input_provider_module.build_forward_input_provider("synchronous"),
        SynchronousForwardInputProvider,
    )
    assert isinstance(
        forward_input_provider_module.build_forward_input_provider("overlapped"),
        OverlappedForwardInputProvider,
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        forward_input_provider_module.build_forward_input_provider("bogus")
    assert exc_info.value.code == "training.forward_input_provider_mode_invalid"

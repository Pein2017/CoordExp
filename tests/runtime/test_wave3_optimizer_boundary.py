"""Wave-3 tasks 3.1/3.2: the runtime-owned optimizer boundary.

Every test here exercises the ONE runtime owner of the post-backward optimizer
boundary: the closed `optimizer_boundary_action` decision that converges on
every rank BEFORE any wrapper call, the fp16 call order (exactly-once unscale,
finite/norm inspection, action-specific clipping, one wrapper call), the
post-wrapper `all_skipped | none_skipped | mixed` consensus, and the single
runtime-owned `AppliedUpdateReceipt`.

fp16 is simulated on CPU through duck-typed accelerator/scaler fakes: the
semantics under test are the ORDER and the CONSENSUS, not CUDA arithmetic.
Genuine CUDA fp16 arms are task 3.8's GPU probes.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.common.errors import RuntimeContractError
from src.config.models import RuntimeBatchResolution, RuntimeConfig
from src.runtime.finite_gates import GateDecision
from src.runtime.optimizer_boundary import (
    AppliedUpdateReceipt,
    OptimizerBoundaryTerminal,
)
from src.runtime.train_runtime import TrainRuntime


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeScaler:
    """The bounded slice of `torch.cuda.amp.GradScaler` the runtime reads."""

    def __init__(self, *, found_inf: bool = False) -> None:
        self.found_inf = found_inf
        self.found_inf_reads: list[Any] = []

    def _found_inf_per_device(self, optimizer: Any = None) -> dict[str, torch.Tensor]:
        self.found_inf_reads.append(optimizer)
        return {"cpu": torch.tensor(1.0 if self.found_inf else 0.0)}


class FakeAccelerator:
    def __init__(
        self,
        *,
        process_index: int = 0,
        num_processes: int = 1,
        mixed_precision: str | None = "bf16",
        scaler: FakeScaler | None = None,
        events: list[str] | None = None,
        step_was_skipped: bool = False,
        unscale_error: Exception | None = None,
        expose_skip_flag: bool = True,
    ) -> None:
        self.process_index = process_index
        self.num_processes = num_processes
        self.device = torch.device("cpu")
        self.is_main_process = process_index == 0
        self.distributed_type = SimpleNamespace(
            name="NO" if num_processes == 1 else "MULTI_GPU"
        )
        self.gradient_accumulation_steps = 1
        self.mixed_precision = mixed_precision
        self.scaler = scaler
        self.events = events if events is not None else []
        self.unscale_calls: list[Any] = []
        self.unscale_error = unscale_error
        self._step_was_skipped = bool(step_was_skipped)
        self._expose_skip_flag = bool(expose_skip_flag)

    # -- prepare / accumulation -------------------------------------------
    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        return tuple(objects)

    @contextmanager
    def no_sync(self, model: Any) -> Any:
        yield

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    # -- fp16 surface ------------------------------------------------------
    def unscale_gradients(self, optimizer: Any = None) -> None:
        self.events.append("unscale")
        self.unscale_calls.append(optimizer)
        if self.unscale_error is not None:
            raise self.unscale_error

    @property
    def optimizer_step_was_skipped(self) -> bool:
        if not self._expose_skip_flag:
            raise AttributeError("optimizer_step_was_skipped")
        return self._step_was_skipped

    def set_step_was_skipped(self, value: bool) -> None:
        self._step_was_skipped = bool(value)

    # -- PROHIBITED --------------------------------------------------------
    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> None:
        self.events.append("accelerator_clip")
        raise AssertionError(
            "accelerator.clip_grad_norm_ is prohibited: it may unscale a second time"
        )


class RecordingOptimizer(torch.optim.SGD):
    """Wrapper stand-in: records the call and can suppress its own update."""

    def __init__(self, parameters: Any, *, events: list[str], lr: float = 0.1) -> None:
        super().__init__(parameters, lr=lr)
        self.events = events
        self.step_calls = 0
        self.suppress_update = False
        self.accelerator: FakeAccelerator | None = None
        self.skip_flag_after_step: bool | None = None

    def step(self, closure: Any | None = None) -> Any:
        self.events.append("wrapper_step")
        self.step_calls += 1
        if self.accelerator is not None and self.skip_flag_after_step is not None:
            self.accelerator.set_step_was_skipped(self.skip_flag_after_step)
        if self.suppress_update:
            return None
        return super().step(closure)


class RecordingScheduler:
    def __init__(self, *, events: list[str]) -> None:
        self.events = events
        self.last_epoch = 0

    def step(self) -> None:
        self.events.append("scheduler_step")
        self.last_epoch += 1

    def get_last_lr(self) -> list[float]:
        return [0.1]


class PeerGatherer:
    """Two-rank gatherer double with an explicitly authored peer report."""

    def __init__(
        self,
        *,
        peer_gradient: dict[str, Any] | None = None,
        peer_step_was_skipped: bool | None = None,
    ) -> None:
        self.peer_gradient = peer_gradient or {}
        self.peer_step_was_skipped = peer_step_was_skipped
        self.calls: list[str] = []

    def __call__(self, local_report: Any) -> tuple[Any, Any]:
        from dataclasses import replace as _replace

        from src.runtime.finite_gates import (
            RankGradientFiniteReport,
            RankScalarFiniteReport,
        )
        from src.runtime.optimizer_boundary import RankPostWrapperReport

        if isinstance(local_report, RankGradientFiniteReport):
            self.calls.append("gradient")
            peer = _replace(local_report, rank=1, **self.peer_gradient)
            return local_report, peer
        if isinstance(local_report, RankPostWrapperReport):
            self.calls.append("post_wrapper")
            skipped = (
                local_report.step_was_skipped
                if self.peer_step_was_skipped is None
                else bool(self.peer_step_was_skipped)
            )
            peer = _replace(local_report, rank=1, step_was_skipped=skipped)
            return local_report, peer
        if isinstance(local_report, RankScalarFiniteReport):
            self.calls.append("scalar")
            return local_report, _replace(local_report, rank=1)
        raise AssertionError(type(local_report).__name__)


def _runtime(
    *,
    accelerator: FakeAccelerator,
    events: list[str],
    world_size: int = 1,
    max_grad_norm: float | None = None,
    gatherer: Any | None = None,
    with_scheduler: bool = True,
    grad_value: float | None = 1.0,
) -> TrainRuntime:
    accelerator.events = events
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = RecordingOptimizer(model.parameters(), events=events)
    optimizer.accelerator = accelerator
    scheduler = RecordingScheduler(events=events) if with_scheduler else None
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig.model_validate(
            {"seed": 17, "determinism": {"mode": "legacy"}}
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=world_size,
            effective_batch_size=world_size,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_mixed_precision=(accelerator.mixed_precision or "no"),
        max_grad_norm=max_grad_norm,
        accelerator=accelerator,
        rank_report_gatherer=gatherer,
    )
    if grad_value is not None:
        for parameter in runtime.model.parameters():
            parameter.grad = torch.full_like(parameter, float(grad_value))
    return runtime


def _fp16(**kwargs: Any) -> FakeAccelerator:
    kwargs.setdefault("scaler", FakeScaler())
    kwargs.setdefault("mixed_precision", "fp16")
    return FakeAccelerator(**kwargs)


# ==========================================================================
# Family A - closed boundary action converged before any wrapper call
# ==========================================================================


def test_pre_backward_rejection_selects_supported_not_attempted() -> None:
    accelerator = FakeAccelerator()
    events: list[str] = []
    runtime = _runtime(accelerator=accelerator, events=events)
    decision = GateDecision(
        stage="pre_backward_scalar",
        planned_step_id=4,
        world_size=1,
        ranks=(0,),
        all_ranks_safe=False,
        should_call_backward=False,
        should_call_optimizer_step=False,
        should_clear_gradients=True,
        optimizer_update_status="skipped_non_finite_scalar",
        finite_status="non_finite",
        reason_codes=("rank0:non_finite_scalar",),
        rank_diagnostics=(),
        diagnostics={},
        optimizer_boundary_action="not_attempted",
    )
    receipt = runtime.execute_optimizer_boundary(decision, planned_step_id=4)
    assert receipt.action == "not_attempted"
    assert receipt.attempted is False
    assert receipt.applied is False
    assert receipt.step_was_skipped is False
    assert receipt.optimizer_update_status == "skipped_non_finite_scalar"
    assert receipt.mutation_state == "unchanged"
    assert events == []


def test_non_scaler_gradient_rejection_selects_supported_not_attempted() -> None:
    accelerator = FakeAccelerator()
    events: list[str] = []
    runtime = _runtime(accelerator=accelerator, events=events, grad_value=float("nan"))
    decision = runtime.post_backward(planned_step_id=5)
    assert decision.optimizer_boundary_action == "not_attempted"
    assert decision.terminal_reason is None
    assert decision.should_call_optimizer_step is False
    receipt = runtime.execute_optimizer_boundary(decision, planned_step_id=5)
    assert receipt.optimizer_update_status == "skipped_gradient_or_overflow"
    assert receipt.attempted is False
    assert runtime.optimizer_step_count == 0
    assert events == []


def test_non_scaler_finite_gradients_select_apply() -> None:
    accelerator = FakeAccelerator()
    events: list[str] = []
    runtime = _runtime(accelerator=accelerator, events=events)
    decision = runtime.post_backward(planned_step_id=6)
    assert decision.optimizer_boundary_action == "apply"
    assert decision.scaler_active is False
    assert accelerator.unscale_calls == []


def test_uniform_fp16_overflow_selects_scaler_skip_on_every_rank() -> None:
    accelerator = _fp16(num_processes=2, scaler=FakeScaler(found_inf=True))
    events: list[str] = []
    gatherer = PeerGatherer(
        peer_gradient={"gradients_finite": False, "scaler_found_inf": True}
    )
    runtime = _runtime(
        accelerator=accelerator,
        events=events,
        world_size=2,
        gatherer=gatherer,
        grad_value=float("inf"),
    )
    decision = runtime.post_backward(planned_step_id=7)
    assert decision.optimizer_boundary_action == "scaler_skip"
    assert decision.terminal_reason is None
    assert decision.should_call_optimizer_step is True
    assert decision.scaler_active is True


def test_mixed_fp16_overflow_candidacy_is_terminal_before_the_wrapper() -> None:
    accelerator = _fp16(num_processes=2, scaler=FakeScaler(found_inf=True))
    events: list[str] = []
    gatherer = PeerGatherer(
        peer_gradient={
            "gradients_finite": True,
            "scaler_found_inf": False,
            "backend_overflow": False,
            "grad_norm": 1.0,
        }
    )
    runtime = _runtime(
        accelerator=accelerator,
        events=events,
        world_size=2,
        gatherer=gatherer,
        grad_value=float("inf"),
    )
    decision = runtime.post_backward(planned_step_id=8)
    assert decision.optimizer_boundary_action is None
    assert decision.terminal_reason == "pre_wrapper_mixed_scaler_overflow"
    with pytest.raises(OptimizerBoundaryTerminal) as excinfo:
        runtime.execute_optimizer_boundary(decision, planned_step_id=8)
    assert "wrapper_step" not in events
    receipt = excinfo.value.receipt
    assert receipt.terminal is True
    assert receipt.terminal_reason == "pre_wrapper_mixed_scaler_overflow"


def test_divergent_active_scaler_candidacy_is_terminal_before_the_wrapper() -> None:
    accelerator = _fp16(num_processes=2)
    events: list[str] = []
    gatherer = PeerGatherer(
        peer_gradient={"scaler_active": False, "unscale_completed": False}
    )
    runtime = _runtime(
        accelerator=accelerator, events=events, world_size=2, gatherer=gatherer
    )
    decision = runtime.post_backward(planned_step_id=9)
    assert decision.terminal_reason == "pre_wrapper_scaler_candidacy_divergent"
    assert decision.optimizer_boundary_action is None


def test_unrelated_unsafe_fp16_state_is_terminal_and_never_raises_rank_locally() -> None:
    accelerator = _fp16(
        num_processes=2, unscale_error=RuntimeError("inf check tensor is on cpu")
    )
    events: list[str] = []
    gatherer = PeerGatherer(peer_gradient={})
    runtime = _runtime(
        accelerator=accelerator, events=events, world_size=2, gatherer=gatherer
    )
    decision = runtime.post_backward(planned_step_id=10)
    # The local unscale failure MUST travel through the same all-rank
    # collective instead of raising before consensus.
    assert gatherer.calls == ["gradient"]
    assert decision.terminal_reason == "pre_wrapper_unrelated_unsafe"


def test_pre_call_logic_ignores_a_stale_wrapper_skip_flag() -> None:
    accelerator = _fp16(step_was_skipped=True)
    events: list[str] = []
    runtime = _runtime(accelerator=accelerator, events=events)
    decision = runtime.post_backward(planned_step_id=11)
    assert decision.optimizer_boundary_action == "apply"


# ==========================================================================
# Family B - fp16 ownership and call order
# ==========================================================================


def test_fp16_unscales_exactly_once_before_finite_inspection_and_norm() -> None:
    scaler = FakeScaler()
    accelerator = _fp16(scaler=scaler)
    events: list[str] = []
    runtime = _runtime(accelerator=accelerator, events=events, max_grad_norm=0.5)
    decision = runtime.post_backward(planned_step_id=12)
    assert accelerator.unscale_calls == [runtime.optimizer]
    assert events == ["unscale"]
    # `found_inf` is populated BY unscale, so it must be read after it, and
    # for THIS optimizer.
    assert scaler.found_inf_reads and scaler.found_inf_reads[-1] is runtime.optimizer
    assert decision.pre_clip_grad_norm_rank_max == pytest.approx(1.0)
    runtime.execute_optimizer_boundary(decision, planned_step_id=12)
    assert accelerator.unscale_calls == [runtime.optimizer]
    assert events == ["unscale", "clip", "wrapper_step"]


def test_apply_clips_with_a_non_unscaling_primitive_before_the_wrapper() -> None:
    accelerator = _fp16()
    events: list[str] = []
    runtime = _runtime(accelerator=accelerator, events=events, max_grad_norm=0.25)
    decision = runtime.post_backward(planned_step_id=13)
    runtime.execute_optimizer_boundary(decision, planned_step_id=13)
    assert events == ["unscale", "clip", "wrapper_step"]
    assert "accelerator_clip" not in events
    norms = [
        float(torch.linalg.vector_norm(parameter.grad.float()))
        for parameter in runtime.model.parameters()
    ]
    assert max(norms) <= 0.25 + 1e-6


def test_scaler_skip_performs_no_clip_and_calls_the_wrapper_exactly_once() -> None:
    accelerator = _fp16(scaler=FakeScaler(found_inf=True))
    events: list[str] = []
    runtime = _runtime(
        accelerator=accelerator,
        events=events,
        max_grad_norm=0.25,
        grad_value=float("inf"),
    )
    optimizer = runtime.optimizer
    optimizer.suppress_update = True
    optimizer.skip_flag_after_step = True
    decision = runtime.post_backward(planned_step_id=14)
    assert decision.optimizer_boundary_action == "scaler_skip"
    receipt = runtime.execute_optimizer_boundary(decision, planned_step_id=14)
    assert events == ["unscale", "wrapper_step"]
    assert optimizer.step_calls == 1
    assert receipt.action == "scaler_skip"
    assert receipt.attempted is True
    assert receipt.applied is False
    assert receipt.step_was_skipped is True


def test_accelerator_clip_grad_norm_is_never_called_on_any_branch() -> None:
    for grad_value, found_inf in ((1.0, False), (float("inf"), True)):
        accelerator = _fp16(scaler=FakeScaler(found_inf=found_inf))
        events: list[str] = []
        runtime = _runtime(
            accelerator=accelerator,
            events=events,
            max_grad_norm=1.0,
            grad_value=grad_value,
        )
        runtime.optimizer.suppress_update = found_inf
        runtime.optimizer.skip_flag_after_step = found_inf
        decision = runtime.post_backward(planned_step_id=15)
        runtime.execute_optimizer_boundary(decision, planned_step_id=15)
        assert "accelerator_clip" not in events


def test_production_source_contains_no_accelerator_clip_grad_norm_call() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src"
    hits = [
        f"{path.relative_to(root.parent)}:{index}"
        for path in root.rglob("*.py")
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "clip_grad_norm_(" in line and "torch.nn.utils.clip_grad_norm_(" not in line
    ]
    assert hits == []


# ==========================================================================
# Family C - post-wrapper consensus
# ==========================================================================


def test_apply_accepts_only_none_skipped() -> None:
    accelerator = _fp16(num_processes=2)
    events: list[str] = []
    gatherer = PeerGatherer(peer_step_was_skipped=False)
    runtime = _runtime(
        accelerator=accelerator, events=events, world_size=2, gatherer=gatherer
    )
    runtime.optimizer.skip_flag_after_step = False
    decision = runtime.post_backward(planned_step_id=16)
    receipt = runtime.execute_optimizer_boundary(decision, planned_step_id=16)
    assert gatherer.calls == ["gradient", "post_wrapper"]
    assert receipt.post_wrapper_outcome == "none_skipped"
    assert receipt.optimizer_update_status == "applied"
    assert receipt.terminal is False


def test_scaler_skip_accepts_only_all_skipped() -> None:
    accelerator = _fp16(num_processes=2, scaler=FakeScaler(found_inf=True))
    events: list[str] = []
    gatherer = PeerGatherer(
        peer_gradient={"gradients_finite": False, "scaler_found_inf": True},
        peer_step_was_skipped=True,
    )
    runtime = _runtime(
        accelerator=accelerator,
        events=events,
        world_size=2,
        gatherer=gatherer,
        grad_value=float("inf"),
    )
    runtime.optimizer.suppress_update = True
    runtime.optimizer.skip_flag_after_step = True
    decision = runtime.post_backward(planned_step_id=17)
    receipt = runtime.execute_optimizer_boundary(decision, planned_step_id=17)
    assert receipt.post_wrapper_outcome == "all_skipped"
    assert receipt.optimizer_update_status == "skipped_scaler_overflow"
    assert receipt.terminal is False
    assert receipt.group_learning_rates == (None,)


def test_post_wrapper_mixed_is_terminal_with_nullable_application_state() -> None:
    accelerator = _fp16(num_processes=2)
    events: list[str] = []
    gatherer = PeerGatherer(peer_step_was_skipped=True)
    runtime = _runtime(
        accelerator=accelerator, events=events, world_size=2, gatherer=gatherer
    )
    runtime.optimizer.skip_flag_after_step = False
    decision = runtime.post_backward(planned_step_id=18)
    with pytest.raises(OptimizerBoundaryTerminal) as excinfo:
        runtime.execute_optimizer_boundary(decision, planned_step_id=18)
    receipt = excinfo.value.receipt
    assert gatherer.calls == ["gradient", "post_wrapper"]
    assert receipt.attempted is True
    assert receipt.applied is None
    assert receipt.step_was_skipped is None
    assert receipt.group_learning_rates == (None,)
    assert receipt.mutation_state == "divergent_or_unknown"
    assert receipt.terminal_reason == "post_wrapper_mixed"


def test_apply_with_all_skipped_is_terminal_but_keeps_known_truth() -> None:
    accelerator = _fp16(num_processes=2)
    events: list[str] = []
    gatherer = PeerGatherer(peer_step_was_skipped=True)
    runtime = _runtime(
        accelerator=accelerator, events=events, world_size=2, gatherer=gatherer
    )
    runtime.optimizer.skip_flag_after_step = True
    decision = runtime.post_backward(planned_step_id=19)
    with pytest.raises(OptimizerBoundaryTerminal) as excinfo:
        runtime.execute_optimizer_boundary(decision, planned_step_id=19)
    receipt = excinfo.value.receipt
    assert receipt.applied is False
    assert receipt.step_was_skipped is True
    assert receipt.group_learning_rates == (None,)
    assert receipt.terminal_reason == "post_wrapper_apply_all_skipped"


def test_scaler_skip_with_none_skipped_is_terminal_and_retains_applied_lrs() -> None:
    accelerator = _fp16(num_processes=2, scaler=FakeScaler(found_inf=True))
    events: list[str] = []
    gatherer = PeerGatherer(
        peer_gradient={"gradients_finite": False, "scaler_found_inf": True},
        peer_step_was_skipped=False,
    )
    runtime = _runtime(
        accelerator=accelerator,
        events=events,
        world_size=2,
        gatherer=gatherer,
        grad_value=float("inf"),
    )
    runtime.optimizer.skip_flag_after_step = False
    decision = runtime.post_backward(planned_step_id=20)
    with pytest.raises(OptimizerBoundaryTerminal) as excinfo:
        runtime.execute_optimizer_boundary(decision, planned_step_id=20)
    receipt = excinfo.value.receipt
    assert receipt.applied is True
    assert receipt.step_was_skipped is False
    assert receipt.group_learning_rates == (pytest.approx(0.1),)
    assert receipt.mutation_state == "applied_unsafe"
    assert receipt.terminal_reason == "post_wrapper_scaler_skip_none_skipped"


def test_no_rank_raises_from_its_local_skip_flag_before_the_consensus() -> None:
    accelerator = _fp16(num_processes=2)
    events: list[str] = []
    gatherer = PeerGatherer(peer_step_was_skipped=False)
    runtime = _runtime(
        accelerator=accelerator, events=events, world_size=2, gatherer=gatherer
    )
    # This rank's own wrapper reports a skip that contradicts `apply`; the
    # peer's does not. The local rank must still JOIN the consensus.
    runtime.optimizer.skip_flag_after_step = True
    decision = runtime.post_backward(planned_step_id=21)
    with pytest.raises(OptimizerBoundaryTerminal):
        runtime.execute_optimizer_boundary(decision, planned_step_id=21)
    assert gatherer.calls == ["gradient", "post_wrapper"]


def test_non_fp16_apply_runs_no_post_wrapper_collective() -> None:
    accelerator = FakeAccelerator(num_processes=2)
    events: list[str] = []
    gatherer = PeerGatherer()
    runtime = _runtime(
        accelerator=accelerator, events=events, world_size=2, gatherer=gatherer
    )
    decision = runtime.post_backward(planned_step_id=22)
    receipt = runtime.execute_optimizer_boundary(decision, planned_step_id=22)
    assert gatherer.calls == ["gradient"]
    assert receipt.optimizer_update_status == "applied"
    assert receipt.post_wrapper_outcome is None


# ==========================================================================
# Family D - runtime-owned receipt constructors
# ==========================================================================


def test_not_attempted_constructor_reports_truthful_booleans_and_null_lrs() -> None:
    receipt = AppliedUpdateReceipt.not_attempted(31, 3, "skipped_non_finite_scalar")
    assert receipt.planned_step_id == 31
    assert receipt.attempted is False
    assert receipt.applied is False
    assert receipt.step_was_skipped is False
    assert receipt.group_learning_rates == (None, None, None)
    assert receipt.mutation_state == "unchanged"
    assert receipt.terminal is False


def test_terminal_not_attempted_after_unscale_is_divergent_not_unchanged() -> None:
    unchanged = AppliedUpdateReceipt.terminal_not_attempted(
        32, 1, "pre_wrapper_unrelated_unsafe", unscale_completed=False
    )
    divergent = AppliedUpdateReceipt.terminal_not_attempted(
        32, 1, "pre_wrapper_unrelated_unsafe", unscale_completed=True
    )
    assert unchanged.mutation_state == "unchanged"
    assert divergent.mutation_state == "divergent_or_unknown"
    for receipt in (unchanged, divergent):
        assert receipt.terminal is True
        assert receipt.attempted is False
        assert receipt.applied is False
        assert receipt.step_was_skipped is False
        assert receipt.group_learning_rates == (None,)


def test_terminal_post_wrapper_constructor_matrix() -> None:
    mixed = AppliedUpdateReceipt.terminal_post_wrapper(
        33, 2, action="apply", outcome="mixed", pre_call_learning_rates=(0.1, 0.2)
    )
    assert (mixed.applied, mixed.step_was_skipped) == (None, None)
    assert mixed.group_learning_rates == (None, None)
    assert mixed.mutation_state == "divergent_or_unknown"

    contradicted_apply = AppliedUpdateReceipt.terminal_post_wrapper(
        33, 2, action="apply", outcome="all_skipped", pre_call_learning_rates=(0.1, 0.2)
    )
    assert contradicted_apply.applied is False
    assert contradicted_apply.step_was_skipped is True
    assert contradicted_apply.group_learning_rates == (None, None)

    contradicted_skip = AppliedUpdateReceipt.terminal_post_wrapper(
        33,
        2,
        action="scaler_skip",
        outcome="none_skipped",
        pre_call_learning_rates=(0.1, 0.2),
    )
    assert contradicted_skip.applied is True
    assert contradicted_skip.step_was_skipped is False
    assert contradicted_skip.group_learning_rates == (0.1, 0.2)
    for receipt in (mixed, contradicted_apply, contradicted_skip):
        assert receipt.terminal is True
        assert receipt.attempted is True


def test_receipt_cannot_be_constructed_outside_its_runtime_constructors() -> None:
    with pytest.raises(RuntimeContractError) as excinfo:
        AppliedUpdateReceipt(
            planned_step_id=1,
            action="apply",
            attempted=True,
            applied=True,
            step_was_skipped=False,
            mutation_state="applied",
            optimizer_update_status="applied",
            terminal=False,
            terminal_reason=None,
            reason=None,
            post_wrapper_outcome=None,
            group_learning_rates=(0.1,),
        )
    assert excinfo.value.code == "runtime.update_receipt_direct_construction"


def test_no_consumer_synthesizes_receipt_booleans_from_an_exception() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src"
    producers = sorted(
        str(path.relative_to(root.parent))
        for path in root.rglob("*.py")
        if "AppliedUpdateReceipt." in path.read_text(encoding="utf-8")
    )
    # Only the runtime owner may PRODUCE a receipt. Reporting, session, and
    # artifact code may hold one, but may never build boundary booleans (least
    # of all from an exception handler).
    assert producers == ["src/runtime/train_runtime.py"]
    consumers = sorted(
        str(path.relative_to(root.parent))
        for path in root.rglob("*.py")
        if "AppliedUpdateReceipt" in path.read_text(encoding="utf-8")
    )
    assert consumers == [
        "src/runtime/optimizer_boundary.py",
        "src/runtime/train_runtime.py",
        "src/training/supervised_trainer.py",
    ]


# ==========================================================================
# Family E - counters
# ==========================================================================


def test_optimizer_step_count_counts_every_completed_wrapper_invocation() -> None:
    accelerator = _fp16(scaler=FakeScaler(found_inf=True))
    events: list[str] = []
    runtime = _runtime(accelerator=accelerator, events=events, grad_value=float("inf"))
    runtime.optimizer.suppress_update = True
    runtime.optimizer.skip_flag_after_step = True
    decision = runtime.post_backward(planned_step_id=41)
    runtime.execute_optimizer_boundary(decision, planned_step_id=41)
    assert runtime.optimizer_step_count == 1
    assert runtime.scheduler_step_count == 0


def test_optimizer_step_count_does_not_advance_for_not_attempted() -> None:
    accelerator = FakeAccelerator()
    events: list[str] = []
    runtime = _runtime(accelerator=accelerator, events=events, grad_value=float("nan"))
    decision = runtime.post_backward(planned_step_id=42)
    runtime.execute_optimizer_boundary(decision, planned_step_id=42)
    assert runtime.optimizer_step_count == 0
    runtime.scheduler_step(planned_step_id=42)
    assert runtime.scheduler_step_count == 1


def test_pre_wrapper_terminal_advances_no_counter_but_post_wrapper_does() -> None:
    accelerator = _fp16(num_processes=2, scaler=FakeScaler(found_inf=True))
    events: list[str] = []
    gatherer = PeerGatherer(
        peer_gradient={
            "gradients_finite": True,
            "scaler_found_inf": False,
            "backend_overflow": False,
            "grad_norm": 1.0,
        }
    )
    runtime = _runtime(
        accelerator=accelerator,
        events=events,
        world_size=2,
        gatherer=gatherer,
        grad_value=float("inf"),
    )
    decision = runtime.post_backward(planned_step_id=43)
    with pytest.raises(OptimizerBoundaryTerminal):
        runtime.execute_optimizer_boundary(decision, planned_step_id=43)
    assert runtime.optimizer_step_count == 0
    assert runtime.scheduler_step_count == 0
    assert "scheduler_step" not in events

    post_accelerator = _fp16(num_processes=2)
    post_events: list[str] = []
    post_runtime = _runtime(
        accelerator=post_accelerator,
        events=post_events,
        world_size=2,
        gatherer=PeerGatherer(peer_step_was_skipped=True),
    )
    post_runtime.optimizer.skip_flag_after_step = False
    post_decision = post_runtime.post_backward(planned_step_id=44)
    with pytest.raises(OptimizerBoundaryTerminal):
        post_runtime.execute_optimizer_boundary(post_decision, planned_step_id=44)
    assert post_runtime.optimizer_step_count == 1
    assert post_runtime.scheduler_step_count == 0
    assert "scheduler_step" not in post_events


def test_terminal_boundary_clears_gradients_as_cleanup() -> None:
    accelerator = _fp16(num_processes=2, scaler=FakeScaler(found_inf=True))
    gatherer = PeerGatherer(
        peer_gradient={
            "gradients_finite": True,
            "scaler_found_inf": False,
            "backend_overflow": False,
            "grad_norm": 1.0,
        }
    )
    runtime = _runtime(
        accelerator=accelerator,
        events=[],
        world_size=2,
        gatherer=gatherer,
        grad_value=float("inf"),
    )
    decision = runtime.post_backward(planned_step_id=45)
    with pytest.raises(OptimizerBoundaryTerminal):
        runtime.execute_optimizer_boundary(decision, planned_step_id=45)
    assert all(parameter.grad is None for parameter in runtime.model.parameters())
    assert runtime.zero_grad_count == 1


# ==========================================================================
# Family F - applied learning rates
# ==========================================================================


def test_applied_lr_is_the_pre_call_group_value_not_the_scheduled_one() -> None:
    accelerator = FakeAccelerator()
    events: list[str] = []
    runtime = _runtime(accelerator=accelerator, events=events)
    for group in runtime.optimizer.param_groups:
        group["lr"] = 0.07
    decision = runtime.post_backward(planned_step_id=51)
    receipt = runtime.execute_optimizer_boundary(decision, planned_step_id=51)
    # A scheduler advance AFTER the update must not retro-label the row.
    for group in runtime.optimizer.param_groups:
        group["lr"] = 0.99
    runtime.scheduler_step(planned_step_id=51)
    assert receipt.group_learning_rates == (pytest.approx(0.07),)


def test_skipped_and_not_attempted_outcomes_report_null_lrs() -> None:
    accelerator = FakeAccelerator()
    runtime = _runtime(accelerator=accelerator, events=[], grad_value=float("nan"))
    decision = runtime.post_backward(planned_step_id=52)
    receipt = runtime.execute_optimizer_boundary(decision, planned_step_id=52)
    assert receipt.group_learning_rates == (None,)


# ==========================================================================
# Family G - plumbing through the completed-step observation
# ==========================================================================


def test_completed_step_observation_carries_the_runtime_receipt() -> None:
    from src.training.supervised_trainer import CompletedStepObservation

    receipt = AppliedUpdateReceipt.not_attempted(61, 1, "skipped_non_finite_scalar")
    observation = CompletedStepObservation(
        planned_step_id=61,
        micro_step_count=1,
        loss_bundle_artifact={},
        optimizer_update_status=receipt.optimizer_update_status,
        finite_status="non_finite",
        update_receipt=receipt,
    )
    assert observation.update_receipt is receipt


def test_unreadable_post_wrapper_flag_converges_as_unknown_not_a_local_raise() -> None:
    accelerator = _fp16(num_processes=2, expose_skip_flag=False)
    events: list[str] = []
    gatherer = PeerGatherer(peer_step_was_skipped=False)
    runtime = _runtime(
        accelerator=accelerator, events=events, world_size=2, gatherer=gatherer
    )
    decision = runtime.post_backward(planned_step_id=71)
    with pytest.raises(OptimizerBoundaryTerminal) as excinfo:
        runtime.execute_optimizer_boundary(decision, planned_step_id=71)
    # The rank that cannot observe its own outcome still JOINS the consensus.
    assert gatherer.calls == ["gradient", "post_wrapper"]
    receipt = excinfo.value.receipt
    assert receipt.terminal_reason == "post_wrapper_mixed"
    assert receipt.applied is None
    assert receipt.mutation_state == "divergent_or_unknown"

"""P1-2: declared fp16 requires an active, enabled GradScaler.

Two layers are covered here, matching the delta scenario "Declared fp16
without an active GradScaler":

1. A uniform launch-time refusal where the resolved precision declaration and
   the accelerator are both visible, before any training collective and before
   any training-side mutation (``accelerator.prepare``).
2. An all-rank consensus backstop: when every gathered report declares fp16 and
   no rank is a scaler candidate, the boundary is TERMINAL rather than the
   retained bf16/non-scaler ``apply``. The rank-local report never raises on
   its own, so this backstop cannot introduce a pre-collective desync.

bf16/fp32 runs (``declared_fp16`` false, no scaler) keep the retained branch
verbatim, and the existing fp16 paths are unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.common.errors import RuntimeContractError
from src.config.models import RuntimeBatchResolution, RuntimeConfig
from src.runtime.finite_gates import (
    RankGradientFiniteReport,
    build_gradient_finite_report,
    reduce_gradient_overflow_reports,
)
from src.runtime.optimizer_boundary import (
    TERMINAL_PRE_WRAPPER_FP16_SCALER_MISSING,
    TERMINAL_PRE_WRAPPER_MIXED_SCALER_OVERFLOW,
    TERMINAL_PRE_WRAPPER_SCALER_CANDIDACY_DIVERGENT,
)
from src.runtime.train_runtime import TrainRuntime, validate_accelerator_runtime


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _Scaler:
    """The bounded GradScaler slice the runtime reads."""

    def __init__(self, *, enabled: bool = True, found_inf: bool = False) -> None:
        self._enabled = bool(enabled)
        self.found_inf = bool(found_inf)

    def is_enabled(self) -> bool:
        return self._enabled

    def _found_inf_per_device(self, optimizer: Any = None) -> dict[str, torch.Tensor]:
        return {"cpu": torch.tensor(1.0 if self.found_inf else 0.0)}


class _Accelerator:
    """The bounded accelerator surface `TrainRuntime` reads at construction."""

    def __init__(
        self,
        *,
        mixed_precision: str | None = "fp16",
        scaler: Any | None = None,
        num_processes: int = 1,
        process_index: int = 0,
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
        self.events: list[str] = []
        self.prepare_calls = 0
        self.unscale_calls: list[Any] = []

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        self.prepare_calls += 1
        return tuple(objects)

    @contextmanager
    def no_sync(self, model: Any) -> Any:
        yield

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def unscale_gradients(self, optimizer: Any = None) -> None:
        self.events.append("unscale")
        self.unscale_calls.append(optimizer)


def _runtime(
    accelerator: _Accelerator,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    model: torch.nn.Module | None = None,
) -> TrainRuntime:
    model = model if model is not None else torch.nn.Linear(1, 1, bias=False)
    return TrainRuntime(
        runtime_config=RuntimeConfig.model_validate(
            {"seed": 17, "determinism": {"mode": "legacy"}}
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=accelerator.num_processes,
            effective_batch_size=accelerator.num_processes,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        expected_mixed_precision=str(accelerator.mixed_precision),
        max_grad_norm=None,
        accelerator=accelerator,
        rank_report_gatherer=None,
    )


def _grad_report(
    rank: int,
    *,
    world_size: int = 2,
    declared_fp16: bool = True,
    scaler_active: bool = False,
    gradients_finite: bool = True,
    grad_norm: float | None = 1.0,
    unscale_completed: bool = False,
    scaler_found_inf: bool = False,
    report_error_code: str | None = None,
) -> RankGradientFiniteReport:
    return RankGradientFiniteReport(
        planned_step_id=7,
        rank=rank,
        world_size=world_size,
        gradients_finite=gradients_finite,
        backend_overflow=False,
        grad_norm=grad_norm,
        scaler_active=scaler_active,
        unscale_completed=unscale_completed,
        scaler_found_inf=scaler_found_inf,
        report_error_code=report_error_code,
        declared_fp16=declared_fp16,
    )


# ==========================================================================
# Layer 1 - uniform launch-time refusal
# ==========================================================================


def test_declared_fp16_without_scaler_is_refused_at_preflight() -> None:
    accelerator = _Accelerator(mixed_precision="fp16", scaler=None)
    with pytest.raises(RuntimeContractError) as exc_info:
        validate_accelerator_runtime(accelerator, expected_mixed_precision="fp16")
    assert exc_info.value.code == "runtime.fp16_scaler_missing"
    assert exc_info.value.context["scaler_present"] is False
    assert exc_info.value.context["mixed_precision"] == "fp16"


def test_declared_fp16_with_disabled_scaler_is_refused_at_preflight() -> None:
    accelerator = _Accelerator(
        mixed_precision="fp16", scaler=_Scaler(enabled=False)
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        validate_accelerator_runtime(accelerator, expected_mixed_precision="fp16")
    assert exc_info.value.code == "runtime.fp16_scaler_missing"
    assert exc_info.value.context["scaler_present"] is True
    assert exc_info.value.context["scaler_enabled"] is False


@pytest.mark.parametrize("scaler", [None, _Scaler(enabled=False)])
def test_declared_fp16_without_active_scaler_is_refused_before_prepare(
    scaler: Any,
) -> None:
    accelerator = _Accelerator(mixed_precision="fp16", scaler=scaler)
    with pytest.raises(RuntimeContractError) as exc_info:
        _runtime(accelerator)
    assert exc_info.value.code == "runtime.fp16_scaler_missing"
    # Refusal precedes every training-side mutation and every collective.
    assert accelerator.prepare_calls == 0


def test_declared_fp16_with_an_active_scaler_constructs() -> None:
    accelerator = _Accelerator(mixed_precision="fp16", scaler=_Scaler())
    validate_accelerator_runtime(accelerator, expected_mixed_precision="fp16")
    runtime = _runtime(accelerator)
    assert runtime._active_fp16_scaler() is accelerator.scaler
    assert runtime.declared_fp16 is True


@pytest.mark.parametrize("precision", ["bf16", "no"])
def test_non_fp16_runs_without_a_scaler_are_untouched(precision: str) -> None:
    accelerator = _Accelerator(mixed_precision=precision, scaler=None)
    validate_accelerator_runtime(accelerator, expected_mixed_precision=precision)
    runtime = _runtime(accelerator)
    assert runtime._active_fp16_scaler() is None
    assert runtime.declared_fp16 is False
    assert accelerator.prepare_calls == 1


def test_mixed_precision_mismatch_still_fires_before_the_scaler_refusal() -> None:
    accelerator = _Accelerator(mixed_precision="fp16", scaler=None)
    with pytest.raises(RuntimeContractError) as exc_info:
        validate_accelerator_runtime(accelerator, expected_mixed_precision="bf16")
    assert exc_info.value.code == "runtime.mixed_precision_mismatch"


# ==========================================================================
# Layer 2 - all-rank consensus backstop
# ==========================================================================


def test_local_report_never_raises_on_declared_fp16_without_a_scaler() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.ones(2)
    report = build_gradient_finite_report(
        [parameter],
        planned_step_id=7,
        rank=0,
        world_size=2,
        backend_overflow=False,
        scaler_active=False,
        declared_fp16=True,
    )
    assert report.declared_fp16 is True
    assert report.scaler_active is False


def test_all_ranks_declared_fp16_without_a_scaler_is_terminal_when_safe() -> None:
    decision = reduce_gradient_overflow_reports(
        (_grad_report(0), _grad_report(1)),
    )
    assert decision.optimizer_boundary_action is None
    assert decision.terminal_reason == TERMINAL_PRE_WRAPPER_FP16_SCALER_MISSING
    assert decision.should_call_optimizer_step is False
    assert decision.optimizer_update_status == (
        f"terminal_{TERMINAL_PRE_WRAPPER_FP16_SCALER_MISSING}"
    )
    assert f"terminal:{TERMINAL_PRE_WRAPPER_FP16_SCALER_MISSING}" in (
        decision.reason_codes
    )


def test_all_ranks_declared_fp16_without_a_scaler_is_terminal_when_unsafe() -> None:
    decision = reduce_gradient_overflow_reports(
        (
            _grad_report(0),
            _grad_report(1, gradients_finite=False),
        ),
    )
    # NOT reclassified as the retained bf16 `not_attempted` boundary.
    assert decision.optimizer_boundary_action is None
    assert decision.terminal_reason == TERMINAL_PRE_WRAPPER_FP16_SCALER_MISSING
    assert decision.should_call_optimizer_step is False


def test_declared_fp16_carries_into_the_rank_diagnostics() -> None:
    decision = reduce_gradient_overflow_reports(
        (_grad_report(0), _grad_report(1)),
    )
    assert [row["declared_fp16"] for row in decision.rank_diagnostics] == [True, True]


# -- retained bf16/fp32 branch, verbatim -----------------------------------


def test_non_fp16_no_scaler_safe_still_applies() -> None:
    decision = reduce_gradient_overflow_reports(
        (
            _grad_report(0, declared_fp16=False),
            _grad_report(1, declared_fp16=False),
        ),
    )
    assert decision.optimizer_boundary_action == "apply"
    assert decision.terminal_reason is None
    assert decision.optimizer_update_status == "ready_to_step"
    assert decision.should_call_optimizer_step is True


def test_non_fp16_no_scaler_unsafe_is_supported_not_attempted() -> None:
    decision = reduce_gradient_overflow_reports(
        (
            _grad_report(0, declared_fp16=False),
            _grad_report(1, declared_fp16=False, gradients_finite=False),
        ),
    )
    assert decision.optimizer_boundary_action == "not_attempted"
    assert decision.terminal_reason is None
    assert decision.should_call_optimizer_step is False


# -- existing fp16 paths, unchanged ----------------------------------------


def test_uniform_scaler_overflow_still_selects_scaler_skip() -> None:
    decision = reduce_gradient_overflow_reports(
        (
            _grad_report(
                0,
                scaler_active=True,
                unscale_completed=True,
                scaler_found_inf=True,
            ),
            _grad_report(
                1,
                scaler_active=True,
                unscale_completed=True,
                scaler_found_inf=True,
            ),
        ),
    )
    assert decision.optimizer_boundary_action == "scaler_skip"
    assert decision.terminal_reason is None


def test_scaler_candidacy_divergence_still_governs_partial_scaler_ranks() -> None:
    decision = reduce_gradient_overflow_reports(
        (
            _grad_report(0, scaler_active=True, unscale_completed=True),
            _grad_report(1, scaler_active=False),
        ),
    )
    assert decision.optimizer_boundary_action is None
    assert decision.terminal_reason == (
        TERMINAL_PRE_WRAPPER_SCALER_CANDIDACY_DIVERGENT
    )


def test_mixed_scaler_overflow_is_still_terminal() -> None:
    decision = reduce_gradient_overflow_reports(
        (
            _grad_report(
                0,
                scaler_active=True,
                unscale_completed=True,
                scaler_found_inf=True,
            ),
            _grad_report(1, scaler_active=True, unscale_completed=True),
        ),
    )
    assert decision.optimizer_boundary_action is None
    assert decision.terminal_reason == TERMINAL_PRE_WRAPPER_MIXED_SCALER_OVERFLOW


def test_uniform_fp16_scaler_ranks_still_apply_when_safe() -> None:
    decision = reduce_gradient_overflow_reports(
        (
            _grad_report(0, scaler_active=True, unscale_completed=True),
            _grad_report(1, scaler_active=True, unscale_completed=True),
        ),
    )
    assert decision.optimizer_boundary_action == "apply"
    assert decision.terminal_reason is None


# -- runtime wiring --------------------------------------------------------


def test_post_backward_reports_the_runtime_precision_declaration() -> None:
    accelerator = _Accelerator(mixed_precision="fp16", scaler=_Scaler())
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runtime = _runtime(accelerator, optimizer=optimizer, model=model)
    for parameter in runtime.model.parameters():
        parameter.grad = torch.full_like(parameter, 1.0)
    decision = runtime.post_backward(planned_step_id=7)
    assert decision.rank_diagnostics[0]["declared_fp16"] is True
    assert decision.rank_diagnostics[0]["scaler_active"] is True
    assert decision.optimizer_boundary_action == "apply"

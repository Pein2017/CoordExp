from __future__ import annotations

import json
import pytest
import torch

from src.common.errors import RuntimeContractError
from src.losses import LossBundle, LossTermResult, SegmentBalancedDenominator
from src.runtime.finite_gates import (
    RankGradientFiniteReport,
    RankScalarFiniteReport,
    build_gradient_finite_report,
    reduce_gradient_overflow_reports,
    reduce_scalar_finite_reports,
)


def test_scalar_gate_allows_backward_only_after_complete_safe_rank_consensus() -> None:
    reports = (
        RankScalarFiniteReport.from_loss_bundle(
            _bundle(total=3.0),
            planned_step_id=7,
            rank=0,
            world_size=2,
        ),
        RankScalarFiniteReport.from_loss_bundle(
            _bundle(total=4.0),
            planned_step_id=7,
            rank=1,
            world_size=2,
        ),
    )

    decision = reduce_scalar_finite_reports(reports)

    assert decision.stage == "pre_backward_scalar"
    assert decision.all_ranks_safe
    assert decision.should_call_backward
    assert not decision.should_call_optimizer_step
    assert not decision.should_clear_gradients
    assert decision.optimizer_update_status == "pending_backward"
    assert decision.reason_codes == ()


def test_scalar_gate_skips_backward_and_update_for_any_rank_non_finite_loss() -> None:
    reports = (
        RankScalarFiniteReport.from_loss_bundle(
            _bundle(total=float("nan"), base=float("nan")),
            planned_step_id=8,
            rank=0,
            world_size=2,
        ),
        RankScalarFiniteReport.from_loss_bundle(
            _bundle(total=1.0),
            planned_step_id=8,
            rank=1,
            world_size=2,
        ),
    )

    decision = reduce_scalar_finite_reports(reports)

    assert not decision.all_ranks_safe
    assert not decision.should_call_backward
    assert not decision.should_call_optimizer_step
    assert decision.should_clear_gradients
    assert decision.optimizer_update_status == "skipped_non_finite_scalar"
    assert decision.finite_status == "non_finite"
    assert decision.reason_codes == ("rank0:non_finite_scalar",)
    assert decision.rank_diagnostics[0]["terms"]["base_ce"]["finite"] is False


def test_scalar_gate_includes_rank_local_zero_eligible_error_without_deadlock() -> None:
    reports = (
        RankScalarFiniteReport.from_error(
            planned_step_id=9,
            rank=0,
            world_size=2,
            error_code="loss.segment_balanced_zero_eligible",
            error_message="rank has no eligible coordinate segments",
            term_eligible_segment_counts={"token_type_gate": 0},
        ),
        RankScalarFiniteReport.from_loss_bundle(
            _bundle(total=1.0),
            planned_step_id=9,
            rank=1,
            world_size=2,
        ),
    )

    decision = reduce_scalar_finite_reports(reports)

    assert not decision.should_call_backward
    assert decision.should_clear_gradients
    assert decision.reason_codes == ("rank0:loss.segment_balanced_zero_eligible",)
    assert decision.rank_diagnostics[0]["terms"]["token_type_gate"]["eligible_segments"] == 0


def test_gate_rejects_incomplete_or_inconsistent_rank_sets() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        reduce_scalar_finite_reports(
            (
                RankScalarFiniteReport.from_loss_bundle(
                    _bundle(total=1.0),
                    planned_step_id=1,
                    rank=0,
                    world_size=2,
                ),
            )
        )
    assert exc_info.value.code == "runtime.gate_rank_coverage"

    with pytest.raises(RuntimeContractError) as exc_info:
        reduce_gradient_overflow_reports(
            (
                RankGradientFiniteReport(
                    planned_step_id=1,
                    rank=0,
                    world_size=2,
                    gradients_finite=True,
                    backend_overflow=False,
                    grad_norm=1.0,
                ),
                RankGradientFiniteReport(
                    planned_step_id=2,
                    rank=1,
                    world_size=2,
                    gradients_finite=True,
                    backend_overflow=False,
                    grad_norm=1.0,
                ),
            )
        )
    assert exc_info.value.code == "runtime.gate_planned_step"


def test_gradient_gate_allows_optimizer_step_only_when_all_ranks_safe() -> None:
    decision = reduce_gradient_overflow_reports(
        (
            RankGradientFiniteReport(
                planned_step_id=10,
                rank=0,
                world_size=2,
                gradients_finite=True,
                backend_overflow=False,
                grad_norm=3.0,
            ),
            RankGradientFiniteReport(
                planned_step_id=10,
                rank=1,
                world_size=2,
                gradients_finite=True,
                backend_overflow=False,
                grad_norm=4.0,
            ),
        )
    )

    assert decision.stage == "post_backward_gradient"
    assert decision.all_ranks_safe
    assert not decision.should_call_backward
    assert decision.should_call_optimizer_step
    assert not decision.should_clear_gradients
    assert decision.optimizer_update_status == "ready_to_step"
    assert decision.diagnostics["max_grad_norm"] == 4.0


def test_gradient_gate_skips_optimizer_step_for_non_finite_grad_or_overflow() -> None:
    decision = reduce_gradient_overflow_reports(
        (
            RankGradientFiniteReport(
                planned_step_id=11,
                rank=0,
                world_size=2,
                gradients_finite=False,
                backend_overflow=False,
                grad_norm=float("nan"),
            ),
            RankGradientFiniteReport(
                planned_step_id=11,
                rank=1,
                world_size=2,
                gradients_finite=True,
                backend_overflow=True,
                grad_norm=1.0,
            ),
        )
    )

    assert not decision.all_ranks_safe
    assert not decision.should_call_optimizer_step
    assert decision.should_clear_gradients
    assert decision.optimizer_update_status == "skipped_gradient_or_overflow"
    assert decision.reason_codes == (
        "rank0:non_finite_gradient",
        "rank1:backend_overflow",
    )


def test_gradient_gate_skips_optimizer_step_for_non_finite_grad_norm() -> None:
    report = RankGradientFiniteReport(
        planned_step_id=12,
        rank=0,
        world_size=1,
        gradients_finite=True,
        backend_overflow=False,
        grad_norm=float("inf"),
    )
    decision = reduce_gradient_overflow_reports(
        (report,)
    )

    assert not report.is_safe()
    assert not decision.should_call_optimizer_step
    assert decision.should_clear_gradients
    assert decision.optimizer_update_status == "skipped_gradient_or_overflow"
    assert decision.reason_codes == ("rank0:non_finite_grad_norm",)
    assert decision.rank_diagnostics[0]["grad_norm"] is None
    assert decision.rank_diagnostics[0]["grad_norm_finite"] is False
    json.dumps(decision.to_artifact_dict(), allow_nan=False)


def test_gradient_gate_treats_missing_grad_norm_as_unsafe() -> None:
    report = RankGradientFiniteReport(
        planned_step_id=12,
        rank=0,
        world_size=1,
        gradients_finite=True,
        backend_overflow=False,
        grad_norm=None,
    )

    decision = reduce_gradient_overflow_reports((report,))

    assert not report.is_safe()
    assert not decision.should_call_optimizer_step
    assert decision.reason_codes == ("rank0:missing_grad_norm",)
    assert decision.rank_diagnostics[0]["grad_norm"] is None
    assert decision.rank_diagnostics[0]["grad_norm_finite"] is False


def test_gradient_gate_uses_non_finite_gradient_as_clearer_reason() -> None:
    decision = reduce_gradient_overflow_reports(
        (
            RankGradientFiniteReport(
                planned_step_id=12,
                rank=0,
                world_size=1,
                gradients_finite=False,
                backend_overflow=False,
                grad_norm=None,
            ),
        )
    )

    assert not decision.should_call_optimizer_step
    assert decision.reason_codes == ("rank0:non_finite_gradient",)
    assert decision.diagnostics["unsafe_rank_count"] == 1


def test_build_gradient_finite_report_scans_parameter_grads() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([float("inf")])

    report = build_gradient_finite_report(
        (first, second),
        planned_step_id=12,
        rank=0,
        world_size=1,
        backend_overflow=False,
    )

    assert not report.gradients_finite
    assert report.grad_norm == pytest.approx(float("inf"))


def test_build_gradient_finite_report_marks_absent_gradients_as_missing_norm() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    report = build_gradient_finite_report(
        (parameter,),
        planned_step_id=13,
        rank=0,
        world_size=1,
        backend_overflow=False,
    )
    decision = reduce_gradient_overflow_reports((report,))

    assert report.gradients_finite
    assert report.grad_norm is None
    assert not decision.should_call_optimizer_step
    assert decision.reason_codes == ("rank0:missing_grad_norm",)


def test_build_gradient_report_non_finite_norm_triggers_global_skip() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([1e38, 1e38])
    report = build_gradient_finite_report(
        (parameter,),
        planned_step_id=13,
        rank=0,
        world_size=1,
        backend_overflow=False,
    )

    decision = reduce_gradient_overflow_reports((report,))

    assert report.gradients_finite
    assert report.grad_norm == pytest.approx(float("inf"))
    assert not decision.should_call_optimizer_step
    assert decision.reason_codes == ("rank0:non_finite_grad_norm",)


def test_gradient_gate_counts_unsafe_ranks_separately_from_reasons() -> None:
    decision = reduce_gradient_overflow_reports(
        (
            RankGradientFiniteReport(
                planned_step_id=14,
                rank=0,
                world_size=1,
                gradients_finite=False,
                backend_overflow=True,
                grad_norm=float("nan"),
            ),
        )
    )

    assert decision.diagnostics["unsafe_rank_count"] == 1
    assert decision.diagnostics["unsafe_reason_count"] == 2
    assert len(decision.reason_codes) == 2
    json.dumps(decision.to_artifact_dict(), allow_nan=False)


def test_build_gradient_report_rejects_sparse_gradients_with_runtime_error() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.sparse_coo_tensor(
        indices=torch.tensor([[0]]),
        values=torch.tensor([1.0]),
        size=(2,),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_gradient_finite_report(
            (parameter,),
            planned_step_id=15,
            rank=0,
            world_size=1,
            backend_overflow=False,
        )

    assert exc_info.value.code == "runtime.sparse_gradient_unsupported"


def test_build_gradient_report_rejects_compressed_sparse_gradients() -> None:
    parameter = _FakeParameter(
        grad=torch.sparse_csr_tensor(
            crow_indices=torch.tensor([0, 1, 1]),
            col_indices=torch.tensor([0]),
            values=torch.tensor([1.0]),
            size=(2, 2),
        )
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_gradient_finite_report(
            (parameter,),
            planned_step_id=16,
            rank=0,
            world_size=1,
            backend_overflow=False,
        )

    assert exc_info.value.code == "runtime.sparse_gradient_unsupported"


def _bundle(
    *,
    total: float,
    base: float | None = None,
    gate: float | None = None,
) -> LossBundle:
    base_value = total if base is None else base
    gate_value = 0.5 if gate is None else gate
    terms = (
        _term("base_ce", base_value, selected_count=4, eligible_segments=2),
        _term("token_type_gate", gate_value, selected_count=4, eligible_segments=2),
    )
    finite_status = {
        "total_loss": "finite" if torch.isfinite(torch.tensor(total)).item() else "non_finite",
        "terms": {
            term.name: "finite"
            if torch.isfinite(term.weighted_loss.detach()).item()
            else "non_finite"
            for term in terms
        },
    }
    return LossBundle(
        total_loss=torch.tensor(total, dtype=torch.float32),
        terms=terms,
        metrics={},
        counts={"count/supervised_atoms": 4},
        diagnostics={},
        finite_status=finite_status,
    )


class _FakeParameter:
    def __init__(self, *, grad: torch.Tensor) -> None:
        self.grad = grad


def _term(
    name: str,
    value: float,
    *,
    selected_count: int,
    eligible_segments: int,
) -> LossTermResult:
    tensor = torch.tensor(value, dtype=torch.float32)
    return LossTermResult(
        name=name,
        raw_loss=tensor,
        weighted_loss=tensor,
        weight=1.0,
        segment_mean_numerator=tensor.detach() * eligible_segments,
        denominator=SegmentBalancedDenominator(
            term_name=name,
            denominator_scope="planned_step",
            eligible_segment_count=eligible_segments,
            selected_atom_count=selected_count,
            skipped_segment_count=0,
            context_count=1,
        ),
        reducer_name="segment_balanced",
        selected_count=selected_count,
        skipped_count=0,
        math_dtype="float32",
        token_weighted_diagnostic=tensor.detach(),
        diagnostics={},
    )

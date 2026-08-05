from __future__ import annotations

from typing import Any

import pytest
import torch

from src.common.errors import LossContractError
from src.config.models import (
    CoordGaussianRPSLossConfig,
    LossesConfig,
    ProtectedLossesConfig,
    TokenTypeGateLossConfig,
    WeightedLossConfig,
)
from src.coordinate_targets import CoordinateLossTarget
from src.losses import (
    BaseTokenCE,
    CoordGaussianRPSLoss,
    LossContext,
    LossRunner,
    PlannedStepLossSlice,
    TokenTypeGateLoss,
    TokenVocabularyGroups,
    reduce_segment_balanced_planned_step,
)
from src.packing.planner import PackedSegment
from src.supervision import TokenAtom, TokenSequence


def test_loss_runner_streaming_planned_step_reproduces_weighted_metrics_and_top_level_accuracy() -> None:
    # Streaming equivalent of the deleted batch `LossRunner.compute`: proves
    # the three-call streaming protocol (prepare/compute_micro_step/finalize)
    # reproduces the exact planned-step numeric oracle computed independently
    # from the low-level segment-balanced reducer, over unequal per-context
    # atom counts and micro-step ordering (2 vs 1 atoms across two packs).
    contexts = (
        _context(
            _logits(
                (
                    (0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                    (0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0),
                    (0.0, 0.0, 0.0, 1.0, 5.0, 7.0, 0.0, 2.0),
                    (0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 1.0),
                )
            ),
            (
                _segment(0, 0, 2),
                _segment(1, 2, 4),
            ),
            (
                _atom(segment_index=0, target_position=1, token_id=7),
                _atom(segment_index=1, target_position=3, token_id=5, token_type="eos"),
            ),
        ),
        _context(
            _logits(
                (
                    (0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 1.0),
                    (0.0, 0.0, 0.0, 1.0, 6.0, 0.0, 0.0, 2.0),
                )
            ),
            (_segment(0, 0, 2),),
            (_atom(segment_index=0, target_position=1, token_id=3, token_type="coordinate"),),
            pack_index=1,
        ),
    )
    runner = _runner(base_ce_weight=2.0, token_type_gate_weight=0.5)

    plan = runner.prepare_planned_step(
        tuple(context.token_sequence for context in contexts)
    )
    micro_bundles = tuple(
        runner.compute_micro_step(context, plan, local_micro_step_index=local_index)
        for local_index, context in enumerate(contexts)
    )
    bundle = runner.finalize_planned_step(
        tuple(micro_bundle.to_artifact_dict() for micro_bundle in micro_bundles),
        plan,
    )

    expected_base = reduce_segment_balanced_planned_step(
        tuple(
            PlannedStepLossSlice(
                term_name="base_ce",
                context=context,
                per_atom_losses=BaseTokenCE().per_atom_loss(context),
            )
            for context in contexts
        )
    )
    expected_gate = reduce_segment_balanced_planned_step(
        tuple(
            PlannedStepLossSlice(
                term_name="token_type_gate",
                context=context,
                per_atom_losses=TokenTypeGateLoss().per_atom_loss(context),
            )
            for context in contexts
        )
    )
    expected_total = expected_base.loss * 2.0 + expected_gate.loss * 0.5

    base_term = next(term for term in bundle["terms"] if term["name"] == "base_ce")
    gate_term = next(term for term in bundle["terms"] if term["name"] == "token_type_gate")

    assert bundle["total_loss"] == pytest.approx(float(expected_total.detach()))
    assert base_term["weight"] == 2.0
    assert base_term["raw_loss"] == pytest.approx(float(expected_base.loss.detach()))
    assert base_term["weighted_loss"] == pytest.approx(
        float((expected_base.loss * 2.0).detach())
    )
    assert bundle["metrics"]["loss/total"] == pytest.approx(float(expected_total.detach()))
    assert bundle["metrics"]["loss/base_ce"] == pytest.approx(
        float((expected_base.loss * 2.0).detach())
    )
    assert bundle["metrics"]["loss/token_type_gate"] == pytest.approx(
        float((expected_gate.loss * 0.5).detach())
    )
    assert gate_term["diagnostics"]["selected_count_by_token_type"] == {
        "desc_text": 1,
        "schema": 0,
        "coordinate": 1,
        "eos": 1,
    }
    assert "acc_top1/base_ce" not in bundle["metrics"]
    assert "acc_top5/base_ce" not in bundle["metrics"]
    assert bundle["metrics"]["acc_top1"] == pytest.approx(2 / 3)
    assert bundle["metrics"]["acc_top5"] == pytest.approx(1.0)
    assert bundle["counts"]["count/supervised_atoms"] == 3
    assert bundle["counts"]["count/eligible_segments"] == 3
    assert bundle["counts"]["count/skipped_segments"] == 0
    assert bundle["counts"]["count/packs"] == 2
    assert bundle["metrics"]["count/supervised_atoms"] == 3.0
    assert bundle["finite_status"]["total_loss"] == "finite"
    assert bundle["finite_status"]["terms"]["base_ce"] == "finite"
    # Rank-local sufficient statistics (exact integers): summing the
    # per-micro-step stats over unequal atom counts (2 vs 1) reproduces the
    # same integers as computing accuracy over the concatenated contexts.
    assert bundle["accuracy_stats"]["accuracy_atom_count"] == 3
    assert isinstance(bundle["accuracy_stats"]["top1_correct"], int)
    assert isinstance(bundle["accuracy_stats"]["top5_correct"], int)


def _single_micro_step_plan_and_artifact() -> tuple[LossRunner, Any, dict]:
    context = _context(
        _logits(
            (
                (0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0),
            )
        ),
        (_segment(0, 0, 2),),
        (_atom(segment_index=0, target_position=1, token_id=7),),
    )
    runner = _runner(base_ce_weight=1.0, token_type_gate_weight=0.1)
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    return runner, plan, bundle.to_artifact_dict()


def test_loss_runner_finalize_rejects_micro_artifact_missing_accuracy_stats() -> None:
    runner, plan, artifact = _single_micro_step_plan_and_artifact()
    del artifact["accuracy_stats"]

    with pytest.raises(LossContractError) as exc_info:
        runner.finalize_planned_step((artifact,), plan)
    assert exc_info.value.code == "loss.accuracy_stats_missing"


def test_loss_runner_finalize_rejects_micro_artifact_malformed_accuracy_field() -> None:
    runner, plan, artifact = _single_micro_step_plan_and_artifact()
    artifact["accuracy_stats"] = {
        **artifact["accuracy_stats"],
        "top1_correct": -1,
    }

    with pytest.raises(LossContractError) as exc_info:
        runner.finalize_planned_step((artifact,), plan)
    assert exc_info.value.code == "loss.accuracy_stats_field_type"


def test_loss_runner_finalize_rejects_micro_artifact_correct_exceeding_atoms() -> None:
    runner, plan, artifact = _single_micro_step_plan_and_artifact()
    artifact["accuracy_stats"] = {
        "top1_correct": 5,
        "top5_correct": 0,
        "accuracy_atom_count": 1,
    }

    with pytest.raises(LossContractError) as exc_info:
        runner.finalize_planned_step((artifact,), plan)
    assert exc_info.value.code == "loss.accuracy_stats_correct_exceeds_atoms"


def test_loss_runner_global_streaming_denominator_scales_for_ddp_mean() -> None:
    context = _context(
        _logits(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0),
                (0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0),
            ),
            requires_grad=True,
        ),
        (_segment(0, 0, 2),),
        (_atom(segment_index=0, target_position=1, token_id=7),),
    )
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.0,
        token_type_gate_groups=("desc_text",),
    )
    local_base_loss = BaseTokenCE().per_atom_loss(context).sum()

    def gatherer(local_payload):
        peer = {
            name: {
                **dict(payload),
                "eligible_segment_count": 3,
                "selected_atom_count": 3,
                "skipped_segment_count": 0,
                "context_count": 3,
            }
            for name, payload in local_payload.items()
        }
        return (local_payload, peer)

    plan = runner.prepare_planned_step(
        (context.token_sequence,),
        denominator_gatherer=gatherer,
        world_size=2,
        rank=0,
    )
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    term = bundle.term_by_name("base_ce")
    assert plan.denominator_scope == "planned_step_global"
    assert plan.backend_gradient_scale == 2.0
    assert term.denominator.denominator_scope == "planned_step_global"
    assert term.denominator.eligible_segment_count == 4
    assert torch.allclose(term.raw_loss, local_base_loss / 4.0 * 2.0)
    assert term.diagnostics["backend_gradient_scale"] == 2.0


def test_loss_runner_streaming_finalization_preserves_coord_gaussian_rps_diagnostics() -> None:
    context = _context(
        _logits(
            (
                (0.0, 0.0, 8.0, 2.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 7.0, 1.0, 0.0, 0.0, 1.0),
            ),
            requires_grad=True,
        ),
        (_segment(0, 0, 2),),
        (
            _atom(
                segment_index=0,
                target_position=1,
                token_id=3,
                token_type="coordinate",
                coordinate_target=CoordinateLossTarget(
                    bbox=(2, 3, 8, 13),
                    slot_index=0,
                ),
            ),
        ),
    )
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.25,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
        coord_gaussian_rps_weight=1.0,
        coord_gaussian_rps=CoordGaussianRPSLoss(
            gaussian_weight=0.5,
            rps_weight=0.2,
            temperature=1.0,
            gaussian_r95_axis_fraction=0.5,
            gaussian_r95_cap_bins=4,
            gaussian_r95_min_bins=1,
            gaussian_r95_fallback_bins=4,
        ),
    )
    plan = runner.prepare_planned_step((context.token_sequence,))
    micro_bundle = runner.compute_micro_step(
        context,
        plan,
        local_micro_step_index=0,
    )

    artifact = runner.finalize_planned_step(
        (micro_bundle.to_artifact_dict(),),
        plan,
    )

    coord_term = next(
        term for term in artifact["terms"] if term["name"] == "coord_gaussian_rps"
    )
    diagnostics = coord_term["diagnostics"]
    assert diagnostics["target_r95_radius_mean"] == pytest.approx(3.0)
    assert diagnostics["target_r95_radius_max"] == pytest.approx(3.0)
    assert diagnostics["gaussian_ce_mean"] > 0.0
    assert diagnostics["rps_mean"] >= 0.0
    assert diagnostics["target_entropy_mean"] > 0.0
    assert diagnostics["target_peak_prob_mean"] > 0.0


def test_loss_runner_requires_explicit_configured_weights() -> None:
    config = LossesConfig(
        normalizer="segment_balanced",
        protected=ProtectedLossesConfig(
            base_ce=WeightedLossConfig(weight=1.7),
            token_type_gate=TokenTypeGateLossConfig(
                weight=0.25,
                groups=("coordinate", "eos"),
            ),
            coord_gaussian_rps=CoordGaussianRPSLossConfig(
                weight=0.5,
                gaussian_weight=0.5,
                rps_weight=0.2,
                temperature=1.0,
                gaussian_r95_axis_fraction=0.04,
                gaussian_r95_cap_bins=8,
                gaussian_r95_min_bins=1,
                gaussian_r95_fallback_bins=8,
            ),
        ),
    )

    runner = LossRunner.from_config(config)

    assert runner.base_ce_weight == 1.7
    assert runner.token_type_gate_weight == 0.25
    assert runner.token_type_gate_groups == ("coordinate", "eos")
    assert runner.coord_gaussian_rps_weight == 0.5
    assert runner.coord_gaussian_rps is not None
    with pytest.raises(TypeError):
        LossRunner()  # type: ignore[call-arg]


def test_loss_runner_keeps_objective_differentiable_but_metrics_detached() -> None:
    logits = _logits(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0),
            (0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0),
        ),
        requires_grad=True,
    )
    context = _context(
        logits,
        (_segment(0, 0, 2),),
        (_atom(segment_index=0, target_position=1, token_id=7),),
    )

    runner = _runner(base_ce_weight=1.0, token_type_gate_weight=0.1)
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    assert bundle.total_loss.requires_grad
    assert bundle.term_by_name("base_ce").weighted_loss.requires_grad
    assert all(isinstance(value, float) for value in bundle.metrics.values())
    bundle.total_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_loss_runner_records_non_finite_status_without_raising() -> None:
    logits = _logits(
        (
            (0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0),
        )
    )
    context = _context(
        logits,
        (_segment(0, 0, 2),),
        (_atom(segment_index=0, target_position=1, token_id=3, token_type="coordinate"),),
    )

    runner = _runner(base_ce_weight=1.0, token_type_gate_weight=0.1)
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    assert bundle.finite_status["total_loss"] == "non_finite"
    assert bundle.finite_status["terms"]["base_ce"] == "non_finite"
    assert bundle.metrics["finite/total_loss"] == 0.0


def test_loss_runner_filters_token_type_gate_groups() -> None:
    context = _context(
        _logits(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0),
                (0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 2.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 1.0),
            )
        ),
        (
            _segment(0, 0, 2),
            _segment(1, 2, 4),
        ),
        (
            _atom(segment_index=0, target_position=1, token_id=7),
            _atom(segment_index=1, target_position=3, token_id=5, token_type="eos"),
        ),
    )
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.1,
        token_type_gate_groups=("eos",),
    )

    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    gate = bundle.term_by_name("token_type_gate")
    assert gate.selected_count == 1
    assert gate.denominator.eligible_segment_count == 1
    assert gate.denominator.skipped_segment_count == 1
    assert bundle.metrics["loss/token_type_gate/segment_count"] == 1.0

    no_coordinate = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.1,
        token_type_gate_groups=("coordinate",),
    )
    with pytest.raises(LossContractError) as exc_info:
        no_coordinate.prepare_planned_step((context.token_sequence,))
    assert exc_info.value.code == "loss.segment_balanced_zero_eligible"


def test_loss_runner_preserves_compact_logits_positions_when_filtering_gate_groups() -> None:
    context = _context(
        _logits(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0),
                (0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 1.0),
            )
        ),
        (_segment(0, 0, 4),),
        (
            _atom(segment_index=0, target_position=2, token_id=7),
            _atom(segment_index=0, target_position=3, token_id=3, token_type="coordinate"),
        ),
        pack_length=4,
        logits_position_ids=(1, 2),
    )

    runner = _runner(base_ce_weight=1.0, token_type_gate_weight=0.1)
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    assert bundle.counts["count/supervised_atoms"] == 2
    assert bundle.term_by_name("base_ce").selected_count == 2
    assert bundle.term_by_name("token_type_gate").selected_count == 2


def _context(
    logits: torch.Tensor,
    segments: tuple[PackedSegment, ...],
    atoms: tuple[TokenAtom, ...],
    *,
    pack_index: int = 0,
    pack_length: int | None = None,
    logits_position_ids: tuple[int, ...] | None = None,
) -> LossContext:
    resolved_pack_length = int(logits.shape[1]) if pack_length is None else pack_length
    return LossContext(
        logits=logits,
        token_sequence=TokenSequence(
            pack_index=pack_index,
            input_ids=tuple(0 for _ in range(resolved_pack_length)),
            segments=tuple(
                PackedSegment(
                    pack_index=pack_index,
                    segment_index=segment.segment_index,
                    example_index=segment.example_index,
                    example_id=segment.example_id,
                    start=segment.start,
                    end=segment.end,
                )
                for segment in segments
            ),
            atoms=tuple(
                TokenAtom(
                    pack_index=pack_index,
                    segment_index=atom.segment_index,
                    example_index=atom.example_index,
                    example_id=atom.example_id,
                    target_position=atom.target_position,
                    token_id=atom.token_id,
                    token_type=atom.token_type,
                    text=atom.text,
                    logical_target_position=atom.logical_target_position,
                    object_id=atom.object_id,
                    field=atom.field,
                    source=atom.source,
                    coordinate_target=atom.coordinate_target,
                )
                for atom in atoms
            ),
            spans=(),
        ),
        vocab_groups=_groups(),
        logits_position_ids=logits_position_ids,
    )


def _runner(
    *,
    base_ce_weight: float = 1.0,
    token_type_gate_weight: float = 0.1,
) -> LossRunner:
    return LossRunner(
        base_ce_weight=base_ce_weight,
        token_type_gate_weight=token_type_gate_weight,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
    )


def _logits(
    rows: tuple[tuple[float, ...], ...],
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    logits = torch.tensor((rows,), dtype=torch.float32)
    if requires_grad:
        logits.requires_grad_()
    return logits


def _segment(segment_index: int, start: int, end: int) -> PackedSegment:
    return PackedSegment(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        start=start,
        end=end,
    )


def _atom(
    *,
    segment_index: int,
    target_position: int,
    token_id: int,
    token_type: str = "desc_text",
    coordinate_target: CoordinateLossTarget | None = None,
) -> TokenAtom:
    return TokenAtom(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        target_position=target_position,
        token_id=token_id,
        token_type=token_type,
        text="x",
        logical_target_position=target_position,
        object_id="obj-1" if coordinate_target is not None else None,
        field="bbox[0]" if coordinate_target is not None else None,
        source="unit",
        coordinate_target=coordinate_target,
    )


def _groups() -> TokenVocabularyGroups:
    return TokenVocabularyGroups(
        vocab_size=8,
        desc_text=(7,),
        schema=(1, 2),
        coordinate=(3, 4),
        eos=(5,),
        blocked=(0, 6),
    )

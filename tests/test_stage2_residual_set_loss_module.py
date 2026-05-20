from __future__ import annotations

import pytest
import torch

from src.trainers.teacher_forcing.contracts import (
    PipelineModuleSpec,
    TeacherForcingContext,
)
from src.trainers.teacher_forcing.objective_atoms import project_stage2_objective_atoms
from src.trainers.teacher_forcing.objective_pipeline import run_teacher_forcing_pipeline
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.vocab import RoleVocab


def make_role_vocab(*, text_ids: set[int] | None = None) -> RoleVocab:
    return RoleVocab(
        schema_token_ids=frozenset({20, 21}),
        text_token_ids=frozenset(text_ids or {10, 11}),
        coord_token_ids=frozenset({30, 31}),
        stop_token_id=40,
    )


def make_spec(
    *,
    coverage_strength: float = 0.0,
    weight: float = 1.0,
) -> PipelineModuleSpec:
    return PipelineModuleSpec(
        name="residual_set_correction",
        enabled=True,
        weight=weight,
        channels=("A", "B"),
        config={"coverage_strength": coverage_strength},
    )


def make_ir(
    *,
    batch_index: int = 0,
    logit_position: int = 0,
    target_position: int = 1,
    selected_token_id: int = 10,
    valid_token_ids: frozenset[int] = frozenset({10, 11}),
    coverage_target_weights: dict[int, float] | None = None,
    loss_weight: float = 1.0,
    action_weights: tuple[float, ...] | None = None,
    support_provenance: tuple[str, ...] = ("labeled",),
    position_space: str | None = "segment_local",
) -> TeacherForcingTargetIR:
    if action_weights is not None:
        loss_weight = max(float(w) for w in action_weights)
    atom = SupervisionAtom(
        batch_index=batch_index,
        logit_position=logit_position,
        target_position=target_position,
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=valid_token_ids,
        selected_token_id=selected_token_id,
        latent_valid_token_ids=valid_token_ids,
        coverage_target_weights=coverage_target_weights,
        loss_tags=frozenset({"residual_set"}),
        loss_weight=loss_weight,
        coord_role=None,
        provenance={
            "support_provenance": support_provenance,
            "action_weights": action_weights,
            "correction_kind": "selected_path_singleton",
            "source_position_kind": "unit_test",
        },
    )
    metadata: dict[str, object] = {"objective": "residual_set_correction"}
    if position_space is not None:
        metadata["position_space"] = str(position_space)
    return TeacherForcingTargetIR(
        schema_version=TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
        atoms=(atom,),
        metadata=metadata,
    )


def make_empty_ir() -> TeacherForcingTargetIR:
    return TeacherForcingTargetIR(
        schema_version=TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
        atoms=(),
        metadata={
            "objective": "residual_set_correction",
            "position_space": "segment_local",
        },
    )


def make_context(
    *,
    channel: str = "B",
    logits: torch.Tensor | None = None,
    logits_ce: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    target_ir: TeacherForcingTargetIR | None = None,
    role_vocab: RoleVocab | None = None,
    include_role_vocab: bool = True,
    include_residual_sidecar: bool = True,
    segment_start: int = 0,
    segment_len: int | None = None,
) -> TeacherForcingContext:
    if input_ids is None:
        input_ids = torch.tensor([[99, 10]], dtype=torch.long)
    if logits is None:
        logits = torch.full((*input_ids.shape, 50), -20.0, dtype=torch.float32)
        logits[0, 0, 10] = 20.0
        logits[0, 0, 11] = 20.0
    if logits_ce is None:
        logits_ce = logits.clone()
    if segment_len is None:
        segment_len = int(input_ids.shape[1]) - int(segment_start)
    if target_ir is None:
        target_ir = make_ir()
    segment_meta = {"encoded_len": int(segment_len)}
    if include_residual_sidecar:
        segment_meta["residual_set_target_ir"] = target_ir
    meta = [segment_meta]
    extra = {}
    if include_role_vocab:
        extra["role_vocab"] = role_vocab or make_role_vocab()
    return TeacherForcingContext(
        channel=channel,
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits_ce,
        meta=meta,
        coord_token_ids=(30, 31),
        extra=extra,
    )


def test_valid_set_marginal_is_not_selected_token_ce() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    context = make_context()
    result = run_residual_set_correction_module(context=context, spec=make_spec())

    assert result.loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert result.metrics["stage2_ab/channel_b/residual_set/atom_count"] == 1.0
    assert result.metrics["stage2_ab/channel_b/residual_set/loss"] == pytest.approx(
        result.loss.item()
    )
    selected_only_ce = -torch.log_softmax(context.logits[0, 0], dim=-1)[10]
    assert selected_only_ce.item() > 0.5


def test_coverage_strength_zero_disables_coverage_and_one_penalizes_imbalance() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    logits = torch.full((1, 2, 50), -20.0, dtype=torch.float32)
    logits[0, 0, 10] = 20.0
    logits[0, 0, 11] = 0.0
    target_ir = make_ir(coverage_target_weights={10: 0.5, 11: 0.5})
    context = make_context(logits=logits, target_ir=target_ir)

    no_coverage = run_residual_set_correction_module(
        context=context,
        spec=make_spec(coverage_strength=0.0),
    )
    with_coverage = run_residual_set_correction_module(
        context=context,
        spec=make_spec(coverage_strength=1.0),
    )

    assert no_coverage.metrics["stage2_ab/channel_b/residual_set/component/coverage"] == 0.0
    assert with_coverage.loss.item() > no_coverage.loss.item() + 5.0
    assert (
        with_coverage.metrics["stage2_ab/channel_b/residual_set/component/coverage"]
        > 5.0
    )


@pytest.mark.parametrize("coverage_strength", [True, float("nan"), float("inf"), -0.1])
def test_coverage_config_rejects_bool_nonfinite_and_negative_values(
    coverage_strength: object,
) -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        build_residual_set_correction_config,
    )

    with pytest.raises((TypeError, ValueError), match="coverage_strength"):
        build_residual_set_correction_config({"coverage_strength": coverage_strength})


def test_no_role_vocab_fails_closed_with_clear_error() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    with pytest.raises(ValueError, match="role_vocab"):
        run_residual_set_correction_module(
            context=make_context(include_role_vocab=False),
            spec=make_spec(),
        )


def test_b_channel_requires_residual_set_target_ir_sidecar() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    with pytest.raises(ValueError, match="residual_set_target_ir"):
        run_residual_set_correction_module(
            context=make_context(include_residual_sidecar=False),
            spec=make_spec(),
        )


def test_present_empty_residual_set_target_ir_returns_zero() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    result = run_residual_set_correction_module(
        context=make_context(target_ir=make_empty_ir()),
        spec=make_spec(),
    )

    assert result.loss.item() == 0.0
    assert result.metrics["stage2_ab/channel_b/residual_set/atom_count"] == 0.0


def test_packed_segments_rebase_segment_local_residual_ir_positions() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    input_ids = torch.tensor([[99, 10, 88, 11]], dtype=torch.long)
    logits = torch.full((1, 4, 50), -20.0, dtype=torch.float32)
    logits[0, 0, 10] = 20.0
    logits[0, 2, 11] = 20.0
    context = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits.clone(),
        meta=[
            {
                "encoded_len": 2,
                # Residual-set sidecars are stored segment-local before module rebasing.
                "residual_set_target_ir": make_ir(
                    selected_token_id=10,
                    valid_token_ids=frozenset({10}),
                ),
            },
            {
                "encoded_len": 2,
                "residual_set_target_ir": make_ir(
                    selected_token_id=11,
                    valid_token_ids=frozenset({11}),
                ),
            },
        ],
        coord_token_ids=(30, 31),
        extra={"role_vocab": make_role_vocab()},
    )

    result = run_residual_set_correction_module(context=context, spec=make_spec())

    for segment_meta in context.meta:
        target_ir = segment_meta["residual_set_target_ir"]
        assert target_ir.metadata["position_space"] == "segment_local"
        assert (
            target_ir.atoms[0].provenance["correction_kind"]
            == "selected_path_singleton"
        )
        assert target_ir.atoms[0].provenance["source_position_kind"] == "unit_test"
    assert result.metrics["stage2_ab/channel_b/residual_set/atom_count"] == 2.0
    assert result.loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_batch_tensor_residual_ir_uses_declared_positions_without_offset() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    input_ids = torch.tensor([[99, 10, 88, 11]], dtype=torch.long)
    logits = torch.full((1, 4, 50), -20.0, dtype=torch.float32)
    logits[0, 2, 11] = 20.0
    target_ir = make_ir(
        batch_index=0,
        logit_position=2,
        target_position=3,
        selected_token_id=11,
        valid_token_ids=frozenset({11}),
        position_space="batch_tensor",
    )
    context = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits.clone(),
        meta=[
            {"encoded_len": 2, "residual_set_target_ir": make_empty_ir()},
            {"encoded_len": 2, "residual_set_target_ir": target_ir},
        ],
        coord_token_ids=(30, 31),
        extra={"role_vocab": make_role_vocab()},
    )

    result = run_residual_set_correction_module(context=context, spec=make_spec())

    assert result.metrics["stage2_ab/channel_b/residual_set/atom_count"] == 1.0
    assert result.loss.item() == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.parametrize(
    "position_space,error_match",
    [
        (None, "position_space is required"),
        ("mystery_space", "position_space must be"),
    ],
)
def test_non_empty_residual_ir_requires_known_position_space(
    position_space: str | None,
    error_match: str,
) -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    with pytest.raises(ValueError, match=error_match):
        run_residual_set_correction_module(
            context=make_context(target_ir=make_ir(position_space=position_space)),
            spec=make_spec(),
        )


@pytest.mark.parametrize(
    "target_ir,error_match",
    [
        (
            make_ir(logit_position=0, target_position=2),
            r"target_position = logit_position \+ 1",
        ),
        (make_ir(selected_token_id=11), "selected_token_id must match input_ids"),
    ],
)
def test_shift_or_selected_token_mismatch_is_rejected_by_validation(
    target_ir: TeacherForcingTargetIR,
    error_match: str,
) -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    with pytest.raises(ValueError, match=error_match):
        run_residual_set_correction_module(
            context=make_context(target_ir=target_ir),
            spec=make_spec(),
        )


def test_module_reads_full_logits_row_not_logits_ce_shifted_view() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    logits = torch.full((1, 2, 50), -20.0, dtype=torch.float32)
    logits[0, 0, 10] = 20.0
    logits[0, 0, 11] = 20.0
    logits_ce = torch.full((1, 2, 50), -20.0, dtype=torch.float32)
    logits_ce[0, 0, 10] = -20.0
    logits_ce[0, 0, 11] = -20.0
    logits_ce[0, 0, 12] = 20.0

    result = run_residual_set_correction_module(
        context=make_context(logits=logits, logits_ce=logits_ce),
        spec=make_spec(),
    )

    assert result.loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_non_b_channel_is_zero_noop_even_without_role_vocab() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    context = make_context(channel="A", include_role_vocab=False)
    result = run_residual_set_correction_module(
        context=context,
        spec=PipelineModuleSpec(
            name="residual_set_correction",
            config={"coverage_strength": True},
        ),
    )

    assert result.loss.item() == 0.0
    assert result.metrics["stage2_ab/channel_b/residual_set/atom_count"] == 0.0
    assert result.state["residual_set_correction_contrib"].item() == 0.0


def test_mixed_support_helper_uses_max_action_weight_and_preserves_provenance() -> None:
    ir = make_ir(
        action_weights=(0.25, 0.75),
        support_provenance=("labeled", "ul"),
    )

    atom = ir.atoms[0]
    assert atom.loss_weight == 0.75
    assert atom.provenance["support_provenance"] == ("labeled", "ul")


def test_state_contribution_and_pipeline_projection_route_are_visible() -> None:
    logits = torch.zeros((1, 2, 50), dtype=torch.float32)
    context = make_context(
        logits=logits,
        role_vocab=make_role_vocab(text_ids={10, 11, 12}),
    )
    pipeline_out = run_teacher_forcing_pipeline(
        context=context,
        objective_specs=[
            {
                "name": "residual_set_correction",
                "weight": 2.0,
                "channels": ["B"],
                "config": {"coverage_strength": 0.0},
            }
        ],
        diagnostics_specs=[],
    )

    assert "residual_set_correction_contrib" in pipeline_out.state
    assert pipeline_out.state["residual_set_correction_contrib"].shape == torch.Size([])

    atoms = project_stage2_objective_atoms(
        pipeline_result=pipeline_out,
        objective_specs=[
            {
                "name": "residual_set_correction",
                "weight": 2.0,
                "channels": ["B"],
                "config": {"coverage_strength": 0.0},
            }
        ],
        text_provenance="B_rollout_text",
        coord_provenance=None,
    )

    assert atoms["loss/B_rollout_text/residual_set"] == pytest.approx(
        float(pipeline_out.total_loss.detach().cpu().item())
    )

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
    config: dict[str, object] | None = None,
    weight: float = 1.0,
) -> PipelineModuleSpec:
    return PipelineModuleSpec(
        name="residual_set_correction",
        enabled=True,
        weight=weight,
        surfaces=("rollout_correction",),
        config=dict(config or {}),
    )


def make_ir(
    *,
    batch_index: int = 0,
    logit_position: int = 0,
    target_position: int = 1,
    selected_token_id: int = 10,
    selected_token_role: TokenRole = TokenRole.TEXT,
    allowed_token_roles: frozenset[TokenRole] | None = None,
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
        allowed_token_roles=allowed_token_roles or frozenset({selected_token_role}),
        selected_token_role=selected_token_role,
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


def make_multi_atom_ir(
    atoms: tuple[SupervisionAtom, ...],
    *,
    position_space: str = "batch_tensor",
) -> TeacherForcingTargetIR:
    return TeacherForcingTargetIR(
        schema_version=TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
        atoms=atoms,
        metadata={
            "objective": "residual_set_correction",
            "position_space": position_space,
        },
    )


def make_context(
    *,
    channel: str = "rollout_correction",
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
    assert result.metrics["stage2_rollout_correction/residual_set/atom_count"] == 1.0
    assert result.metrics["stage2_rollout_correction/residual_set/sequence_loss"] == pytest.approx(
        result.loss.item()
    )
    selected_only_ce = -torch.log_softmax(context.logits[0, 0], dim=-1)[10]
    assert selected_only_ce.item() > 0.5


def test_residual_set_module_counts_ambiguous_strict_and_mismatch_atoms() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    ambiguous = SupervisionAtom(
        batch_index=0,
        logit_position=0,
        target_position=1,
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({10, 11}),
        selected_token_id=10,
        latent_valid_token_ids=frozenset({10, 11}),
        coverage_target_weights=None,
        loss_tags=frozenset({"residual_set"}),
        loss_weight=1.0,
        coord_role=None,
        provenance={"source_position_kind": "unit_test"},
    )
    corrected = SupervisionAtom(
        batch_index=0,
        logit_position=1,
        target_position=2,
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({11}),
        selected_token_id=11,
        latent_valid_token_ids=frozenset({11}),
        coverage_target_weights=None,
        loss_tags=frozenset({"residual_set"}),
        loss_weight=1.0,
        coord_role=None,
        provenance={
            "source_position_kind": "unit_test",
            "allow_target_token_mismatch": True,
            "target_token_mismatch": True,
            "live_token_id": 10,
        },
    )
    target_ir = make_multi_atom_ir((ambiguous, corrected), position_space="batch_tensor")
    input_ids = torch.tensor([[99, 10, 10]], dtype=torch.long)
    logits = torch.full((1, 3, 50), -20.0, dtype=torch.float32)
    logits[0, 0, 10] = 20.0
    logits[0, 0, 11] = 20.0
    logits[0, 1, 10] = 20.0
    context = make_context(
        input_ids=input_ids,
        logits=logits,
        target_ir=target_ir,
        role_vocab=make_role_vocab(text_ids={10, 11}),
        segment_len=3,
    )

    result = run_residual_set_correction_module(context=context, spec=make_spec())

    assert result.metrics["stage2_rollout_correction/residual_set/ambiguous_token_targets"] == 1.0
    assert result.metrics["stage2_rollout_correction/residual_set/strict_token_targets"] == 1.0
    assert result.metrics["stage2_rollout_correction/residual_set/target_token_mismatch"] == 1.0
    assert result.loss.item() > 10.0


def test_target_token_mismatch_requires_explicit_first_error_provenance() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    atom = SupervisionAtom(
        batch_index=0,
        logit_position=0,
        target_position=1,
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({11}),
        selected_token_id=11,
        latent_valid_token_ids=frozenset({11}),
        coverage_target_weights=None,
        loss_tags=frozenset({"residual_set"}),
        loss_weight=1.0,
        coord_role=None,
        provenance={"source_position_kind": "unit_test"},
    )
    context = make_context(
        input_ids=torch.tensor([[99, 10]], dtype=torch.long),
        target_ir=make_multi_atom_ir((atom,), position_space="batch_tensor"),
        role_vocab=make_role_vocab(text_ids={10, 11}),
        segment_len=2,
    )

    with pytest.raises(ValueError, match="selected_token_id must match"):
        run_residual_set_correction_module(context=context, spec=make_spec())


def test_residual_set_module_rejects_stale_coverage_strength_config() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    with pytest.raises(ValueError, match="coverage_strength"):
        run_residual_set_correction_module(
            context=make_context(),
            spec=make_spec(config={"coverage_strength": 1.0}),
        )


@pytest.mark.parametrize(
    "stale_key",
    [
        "coord_span_policy",
        "ul_geometry",
        "artifact_policy",
        "loss_duplicate_burst_unlikelihood",
        "bbox_geo",
        "bbox_size_aux",
        "coord_reg",
        "coord_gate",
        "text_gate",
    ],
)
def test_residual_set_module_rejects_removed_or_stale_config_keys(
    stale_key: str,
) -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    with pytest.raises(ValueError, match=stale_key):
        run_residual_set_correction_module(
            context=make_context(),
            spec=make_spec(config={stale_key: 1.0}),
        )


def test_residual_set_module_applies_lambda_inner_to_valid_set_term() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    context = make_context(
        target_ir=make_ir(valid_token_ids=frozenset({10})),
        role_vocab=make_role_vocab(text_ids={10, 11}),
    )

    default_weight = run_residual_set_correction_module(
        context=context,
        spec=make_spec(config={"lambda_type": 1.0, "lambda_inner": 1.0}),
    )
    disabled_inner = run_residual_set_correction_module(
        context=context,
        spec=make_spec(config={"lambda_type": 1.0, "lambda_inner": 0.0}),
    )

    assert default_weight.loss.item() > disabled_inner.loss.item() + 0.5
    assert disabled_inner.metrics["stage2_rollout_correction/residual_set/inner_loss"] == 0.0


def test_residual_set_type_loss_is_default_on() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    logits = torch.full((1, 2, 50), -20.0, dtype=torch.float32)
    logits[0, 0, 30] = 20.0
    context = make_context(
        logits=logits,
        target_ir=make_ir(valid_token_ids=frozenset({10})),
        role_vocab=make_role_vocab(text_ids={10}),
    )

    default_weight = run_residual_set_correction_module(
        context=context,
        spec=make_spec(config={"lambda_inner": 0.0}),
    )
    disabled_type = run_residual_set_correction_module(
        context=context,
        spec=make_spec(config={"lambda_type": 0.0, "lambda_inner": 0.0}),
    )

    assert default_weight.loss.item() > 1.0
    assert default_weight.metrics["stage2_rollout_correction/residual_set/type_loss"] == pytest.approx(
        default_weight.loss.item()
    )
    assert disabled_type.loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_singleton_valid_set_equals_hard_ce() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    logits = torch.zeros((1, 2, 50), dtype=torch.float32)
    logits[0, 0, 10] = 0.25
    logits[0, 0, 11] = 1.25
    context = make_context(
        logits=logits,
        target_ir=make_ir(valid_token_ids=frozenset({10})),
        role_vocab=make_role_vocab(text_ids={10, 11}),
    )

    result = run_residual_set_correction_module(context=context, spec=make_spec())
    expected = -torch.log_softmax(logits[0, 0], dim=-1)[10]

    assert result.loss.item() == pytest.approx(expected.item())


def test_stop_singleton_uses_stop_role_only() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    logits = torch.full((1, 2, 50), -20.0, dtype=torch.float32)
    logits[0, 0, 40] = 20.0
    context = make_context(
        input_ids=torch.tensor([[99, 40]], dtype=torch.long),
        logits=logits,
        target_ir=make_ir(
            selected_token_id=40,
            selected_token_role=TokenRole.STOP,
            valid_token_ids=frozenset({40}),
        ),
        role_vocab=make_role_vocab(),
    )

    result = run_residual_set_correction_module(context=context, spec=make_spec())

    assert result.loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert result.metrics["stage2_rollout_correction/residual_set/eos_targets"] == 1.0
    assert result.metrics["stage2_rollout_correction/residual_set/continue_targets"] == 0.0


def test_no_role_vocab_fails_closed_with_clear_error() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    with pytest.raises(ValueError, match="role_vocab"):
        run_residual_set_correction_module(
            context=make_context(include_role_vocab=False),
            spec=make_spec(),
        )


def test_rollout_correction_requires_residual_set_target_ir_sidecar() -> None:
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
    expected_zero_keys = {
        "sequence_count",
        "atom_count",
        "atom_weight_sum",
        "raw_atom_loss_sum",
        "sequence_loss",
        "type_loss",
        "inner_loss",
        "wrong_type_mass",
        "valid_set_mass",
        "dirty_prefix_sequence_count",
        "committed_gt_rows",
        "committed_ul_rows",
        "pending_ul_candidates",
        "promoted_ul_clusters",
        "uncommitted_invalid_geometry",
        "uncommitted_malformed",
        "uncommitted_duplicate",
        "uncommitted_fp_or_unpromoted",
        "spatial_wrong_desc_conflict",
        "label_conflict_atoms",
        "label_conflict_no_atom",
        "eos_targets",
        "continue_targets",
        "dirty_prefix_reencoded",
        "clean_success_skipped",
    }
    for key in expected_zero_keys:
        assert result.metrics[f"stage2_rollout_correction/residual_set/{key}"] == 0.0


def test_packed_segments_rebase_segment_local_residual_ir_positions() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    input_ids = torch.tensor([[99, 10, 88, 11]], dtype=torch.long)
    logits = torch.full((1, 4, 50), -20.0, dtype=torch.float32)
    logits[0, 0, 10] = 20.0
    logits[0, 2, 11] = 20.0
    context = TeacherForcingContext(
        channel="rollout_correction",
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
    assert result.metrics["stage2_rollout_correction/residual_set/atom_count"] == 2.0
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
        channel="rollout_correction",
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

    assert result.metrics["stage2_rollout_correction/residual_set/atom_count"] == 1.0
    assert result.loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_compact_metrics_include_builder_and_decode_slices() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    context = make_context()
    context.meta[0]["decode_mode"] = "greedy"
    context.meta[0]["residual_set_metrics"] = {
        "scanner_dirty_context_span_count": 1.0,
        "scanner_row_decision/committed": 2.0,
        "committed_ul_rows": 1.0,
        "pending_ul_candidates": 3.0,
        "promoted_ul_clusters": 4.0,
        "scanner_row_decision/invalid_geometry": 5.0,
        "scanner_row_decision/malformed_span": 6.0,
        "scanner_row_decision/trailing_incomplete": 0.5,
        "scanner_row_decision/duplicate_burst": 7.0,
        "scanner_row_decision/unmatched_dirty_context": 8.0,
        "scanner_row_decision/spatial_wrong_desc_conflict": 9.0,
        "label_conflict_no_atom": 10.0,
        "dirty_prefix_reencoded": 11.0,
        "clean_success_skipped": 12.0,
    }
    target_ir = context.meta[0]["residual_set_target_ir"]
    atom = target_ir.atoms[0]
    context.meta[0]["residual_set_target_ir"] = make_multi_atom_ir(
        (
            SupervisionAtom(
                batch_index=atom.batch_index,
                logit_position=atom.logit_position,
                target_position=atom.target_position,
                allowed_token_roles=atom.allowed_token_roles,
                selected_token_role=atom.selected_token_role,
                valid_token_ids=atom.valid_token_ids,
                selected_token_id=atom.selected_token_id,
                latent_valid_token_ids=atom.latent_valid_token_ids,
                coverage_target_weights=atom.coverage_target_weights,
                loss_tags=atom.loss_tags,
                loss_weight=atom.loss_weight,
                coord_role=atom.coord_role,
                provenance={
                    **dict(atom.provenance),
                    "correction_kind": "spatial_wrong_desc_conflict",
                },
            ),
        ),
        position_space="segment_local",
    )

    result = run_residual_set_correction_module(context=context, spec=make_spec())

    metrics = result.metrics
    prefix = "stage2_rollout_correction/residual_set"
    assert metrics[f"{prefix}/dirty_prefix_sequence_count"] == 1.0
    assert metrics[f"{prefix}/committed_gt_rows"] == 2.0
    assert metrics[f"{prefix}/committed_ul_rows"] == 1.0
    assert metrics[f"{prefix}/pending_ul_candidates"] == 3.0
    assert metrics[f"{prefix}/promoted_ul_clusters"] == 4.0
    assert metrics[f"{prefix}/uncommitted_invalid_geometry"] == 5.0
    assert metrics[f"{prefix}/uncommitted_malformed"] == 6.5
    assert metrics[f"{prefix}/uncommitted_duplicate"] == 7.0
    assert metrics[f"{prefix}/uncommitted_fp_or_unpromoted"] == 8.0
    assert metrics[f"{prefix}/spatial_wrong_desc_conflict"] == 9.0
    assert metrics[f"{prefix}/label_conflict_atoms"] == 1.0
    assert metrics[f"{prefix}/label_conflict_no_atom"] == 10.0
    assert metrics[f"{prefix}/dirty_prefix_reencoded"] == 11.0
    assert metrics[f"{prefix}/clean_success_skipped"] == 12.0
    assert metrics[f"{prefix}/decode_mode/greedy/sequence_count"] == 1.0
    assert metrics[f"{prefix}/decode_mode/greedy/atom_count"] == 1.0


def test_no_atom_residual_set_metrics_survive_zero_loss_result() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    context = make_context(target_ir=make_empty_ir())
    context.meta[0]["residual_set_metrics"] = {
        "clean_success_skipped": 1.0,
        "label_conflict_no_atom": 2.0,
        "scanner_row_decision/malformed_span": 3.0,
        "scanner_row_decision/trailing_incomplete": 4.0,
    }

    result = run_residual_set_correction_module(context=context, spec=make_spec())

    prefix = "stage2_rollout_correction/residual_set"
    assert result.loss.item() == 0.0
    assert result.metrics[f"{prefix}/sequence_count"] == 0.0
    assert result.metrics[f"{prefix}/atom_count"] == 0.0
    assert result.metrics[f"{prefix}/clean_success_skipped"] == 1.0
    assert result.metrics[f"{prefix}/label_conflict_no_atom"] == 2.0
    assert result.metrics[f"{prefix}/uncommitted_malformed"] == 7.0


def test_sequence_losses_are_weight_normalized_then_batch_meaned() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    input_ids = torch.tensor([[99, 10, 11], [99, 10, 0]], dtype=torch.long)
    logits = torch.zeros((2, 3, 50), dtype=torch.float32)
    logits[0, 0, 10] = 0.5
    logits[0, 0, 11] = 1.5
    logits[0, 1, 10] = 1.0
    logits[0, 1, 11] = 0.25
    logits[1, 0, 10] = -0.25
    logits[1, 0, 11] = 1.75

    seq0_atom0 = make_ir(
        batch_index=0,
        logit_position=0,
        target_position=1,
        selected_token_id=10,
        valid_token_ids=frozenset({10}),
        loss_weight=1.0,
        position_space="batch_tensor",
    ).atoms[0]
    seq0_atom1 = make_ir(
        batch_index=0,
        logit_position=1,
        target_position=2,
        selected_token_id=11,
        valid_token_ids=frozenset({11}),
        loss_weight=3.0,
        position_space="batch_tensor",
    ).atoms[0]
    seq1_atom0 = make_ir(
        batch_index=1,
        logit_position=0,
        target_position=1,
        selected_token_id=10,
        valid_token_ids=frozenset({10}),
        loss_weight=10.0,
        position_space="batch_tensor",
    ).atoms[0]
    context = TeacherForcingContext(
        channel="rollout_correction",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits.clone(),
        meta=[
            {"residual_set_target_ir": make_multi_atom_ir((seq0_atom0, seq0_atom1))},
            {"residual_set_target_ir": make_multi_atom_ir((seq1_atom0,))},
        ],
        coord_token_ids=(30, 31),
        extra={"role_vocab": make_role_vocab(text_ids={10, 11})},
    )

    result = run_residual_set_correction_module(context=context, spec=make_spec())
    ce00 = -torch.log_softmax(logits[0, 0], dim=-1)[10]
    ce01 = -torch.log_softmax(logits[0, 1], dim=-1)[11]
    ce10 = -torch.log_softmax(logits[1, 0], dim=-1)[10]
    expected_seq0 = (ce00 + 3.0 * ce01) / 4.0
    expected_seq1 = ce10
    expected = (expected_seq0 + expected_seq1) / 2.0

    assert result.loss.item() == pytest.approx(expected.item())
    assert result.metrics["stage2_rollout_correction/residual_set/sequence_count"] == 2.0
    assert result.metrics["stage2_rollout_correction/residual_set/atom_count"] == 3.0
    assert result.metrics["stage2_rollout_correction/residual_set/atom_weight_sum"] == 14.0


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


def test_non_rollout_correction_surface_is_zero_noop_even_without_role_vocab() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    context = make_context(channel="offline_sft", include_role_vocab=False)
    result = run_residual_set_correction_module(
        context=context,
        spec=PipelineModuleSpec(
            name="residual_set_correction",
            config={},
        ),
    )

    assert result.loss.item() == 0.0
    assert result.metrics["stage2_rollout_correction/residual_set/atom_count"] == 0.0
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
                "config": {},
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
                "config": {},
            }
        ],
        text_provenance="rollout_correction_text",
        coord_provenance=None,
    )

    assert atoms["loss/rollout_correction_text/residual_set"] == pytest.approx(
        float(pipeline_out.total_loss.detach().cpu().item())
    )


def test_stage2_trie_ce_legacy_alias_routes_to_residual_state_valid_set() -> None:
    logits = torch.zeros((1, 2, 50), dtype=torch.float32)
    context = make_context(
        logits=logits,
        role_vocab=make_role_vocab(text_ids={10, 11, 12}),
    )
    objective_specs = [
        {
            "name": "stage2_trie_ce",
            "weight": 2.0,
            "application": {"preset": "rollout_trie_hard_ce"},
            "config": {},
        }
    ]

    pipeline_out = run_teacher_forcing_pipeline(
        context=context,
        objective_specs=objective_specs,
        diagnostics_specs=[],
    )

    assert pipeline_out.metrics["stage2_trie/residual_state_alias"] == 1.0
    assert "stage2_trie_ce_contrib" in pipeline_out.state
    assert "residual_set_correction_contrib" in pipeline_out.state

    atoms = project_stage2_objective_atoms(
        pipeline_result=pipeline_out,
        objective_specs=objective_specs,
        text_provenance="rollout_correction_text",
        coord_provenance=None,
    )

    assert atoms["loss/rollout_correction_text/residual_state_trie_ce"] == pytest.approx(
        float(pipeline_out.total_loss.detach().cpu().item())
    )
    assert "loss/rollout_correction_text/trie_ce" not in atoms


def test_pipeline_rejects_removed_residual_set_live_config_key() -> None:
    context = make_context()

    with pytest.raises(ValueError, match="bbox_geo"):
        run_teacher_forcing_pipeline(
            context=context,
            objective_specs=[
                {
                    "name": "residual_set_correction",
                    "weight": 1.0,
                    "config": {"bbox_geo": 1.0},
                }
            ],
            diagnostics_specs=[],
        )


def test_pipeline_accepts_residual_set_runtime_config_keys() -> None:
    context = make_context()

    result = run_teacher_forcing_pipeline(
        context=context,
        objective_specs=[
            {
                "name": "residual_set_correction",
                "weight": 1.0,
                "config": {
                    "expected_num_rollouts": 4,
                    "base_seed": 17,
                    "lambda_type": 1.0,
                    "lambda_inner": 1.0,
                    "lambda_ul_promoted": 0.5,
                    "ul_consensus_ratio": 1.0,
                    "min_ul_valid_rollouts": 2,
                    "strict_builder_invariants": True,
                },
            }
        ],
        diagnostics_specs=[],
    )

    assert result.metrics["stage2_rollout_correction/residual_set/atom_count"] == 1.0

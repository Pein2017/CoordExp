from __future__ import annotations

import importlib.util

import pytest
import torch

from src.training.objectives.runner import ObjectiveRunner
from src.training.objectives.teacher_forcing import teacher_forcing_atom_loss
from src.training.objectives.types import LabelLogitRowMap, ObjectiveSpec
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.distributions import TeacherForcingTargetDistribution
from src.training.supervision.spans import SupervisionSpan
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.vocab import RoleVocab


def _role_vocab(
    *,
    text_ids: set[int] | None = None,
    schema_ids: set[int] | None = None,
    coord_ids: set[int] | None = None,
    stop_id: int = 9,
) -> RoleVocab:
    return RoleVocab(
        text_token_ids=frozenset(text_ids or {1, 2, 3}),
        schema_token_ids=frozenset(schema_ids or {4, 5}),
        coord_token_ids=frozenset(coord_ids or {6, 7}),
        stop_token_id=stop_id,
    )


def _atom(
    *,
    batch_index: int = 0,
    logit_position: int = 0,
    target_position: int = 1,
    allowed_token_roles: frozenset[TokenRole] = frozenset({TokenRole.TEXT}),
    selected_token_role: TokenRole = TokenRole.TEXT,
    valid_token_ids: frozenset[int] = frozenset({1}),
    selected_token_id: int = 1,
    coverage_target_weights: dict[int, float] | None = None,
    loss_weight: float = 1.0,
    coord_role: str | None = None,
    provenance: dict[str, object] | None = None,
) -> SupervisionAtom:
    return SupervisionAtom(
        batch_index=batch_index,
        logit_position=logit_position,
        target_position=target_position,
        allowed_token_roles=allowed_token_roles,
        selected_token_role=selected_token_role,
        valid_token_ids=valid_token_ids,
        selected_token_id=selected_token_id,
        latent_valid_token_ids=valid_token_ids,
        coverage_target_weights=coverage_target_weights,
        loss_tags=frozenset({"test"}),
        loss_weight=loss_weight,
        coord_role=coord_role,
        provenance=provenance or {},
    )


def _ir(*atoms: SupervisionAtom) -> TeacherForcingTargetIR:
    return TeacherForcingTargetIR(schema_version=1, atoms=atoms, metadata={})


def _span(ir: TeacherForcingTargetIR, *, sample_id: str = "sample-1") -> SupervisionSpan:
    return SupervisionSpan(
        sample_id=sample_id,
        role="schema",
        label_positions=tuple(atom.target_position for atom in ir.atoms),
        distribution=TeacherForcingTargetDistribution(target_ir=ir),
    )


def _run(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    ir: TeacherForcingTargetIR,
    role_vocab: RoleVocab | None = None,
    coverage_strength: float = 0.0,
    token_type_mass_weight: float = 0.0,
    continuation_margin_weight: float = 0.0,
    bbox_positive_area_weight: float = 0.0,
    label_rows: LabelLogitRowMap | None = None,
) -> torch.Tensor:
    result = ObjectiveRunner().run(
        logits=logits,
        supervision=SupervisionBatch(spans=(_span(ir),), batch_id="batch-1"),
        objectives=(
            ObjectiveSpec(
                "teacher_forcing",
                config={
                    "input_ids": input_ids,
                    "role_vocab": role_vocab or _role_vocab(),
                    "coverage_strength": coverage_strength,
                    "token_type_mass_weight": token_type_mass_weight,
                    "continuation_margin_weight": continuation_margin_weight,
                    "bbox_positive_area_weight": bbox_positive_area_weight,
                },
            ),
        ),
        label_rows=label_rows,
        sample_id_to_batch_index={"sample-1": 0},
    )
    assert "teacher_forcing" in result.objectives
    assert result.state["label_row_map"] is not None
    return result.objectives["teacher_forcing"].loss


def test_rank_2_logits_rejected_even_when_batch_is_one() -> None:
    ir = _ir(_atom())

    with pytest.raises(ValueError, match=r"rank-3 logits \[batch, seq, vocab\]"):
        _run(
            logits=torch.zeros((2, 10), dtype=torch.float32),
            input_ids=torch.tensor([[0, 1]], dtype=torch.long),
            ir=ir,
        )


def test_sliced_logits_rejected_when_prefix_shape_does_not_match_input_ids() -> None:
    ir = _ir(_atom())

    with pytest.raises(ValueError, match=r"logits.shape\[:2\] == input_ids.shape\[:2\]"):
        _run(
            logits=torch.zeros((1, 1, 10), dtype=torch.float32),
            input_ids=torch.tensor([[0, 1]], dtype=torch.long),
            ir=ir,
        )


def test_selected_token_mismatch_is_rejected() -> None:
    ir = _ir(_atom(selected_token_id=1))

    with pytest.raises(ValueError, match="selected_token_id must match input_ids"):
        _run(
            logits=torch.zeros((1, 2, 10), dtype=torch.float32),
            input_ids=torch.tensor([[0, 2]], dtype=torch.long),
            ir=ir,
        )


def test_selected_token_mismatch_can_be_first_error_opd_correction() -> None:
    ir = _ir(
        _atom(
            selected_token_id=1,
            valid_token_ids=frozenset({1}),
            provenance={
                "allow_target_token_mismatch": True,
                "target_token_mismatch": True,
                "live_token_id": 2,
            },
        )
    )
    logits = torch.full((1, 2, 10), -5.0, dtype=torch.float32)
    logits[0, 0, 1] = 5.0

    loss = _run(
        logits=logits,
        input_ids=torch.tensor([[0, 2]], dtype=torch.long),
        ir=ir,
    )

    assert torch.isfinite(loss)
    assert float(loss) < 0.01


def test_valid_token_role_vocab_mismatch_rejected_before_probability_math() -> None:
    logits = torch.zeros((1, 2, 10), dtype=torch.float32)
    logits[0, 0, 6] = float("nan")
    ir = _ir(
        _atom(
            allowed_token_roles=frozenset({TokenRole.TEXT}),
            valid_token_ids=frozenset({1, 6}),
            selected_token_id=1,
        )
    )

    with pytest.raises(ValueError, match="valid_token_ids must be inside allowed role vocab"):
        _run(logits=logits, input_ids=torch.tensor([[0, 1]]), ir=ir)


def test_float_input_ids_with_fractional_target_rejected_before_probability_math() -> None:
    logits = torch.zeros((1, 2, 10), dtype=torch.float32)
    logits[0, 0, 1] = float("nan")
    ir = _ir(_atom(selected_token_id=1))

    with pytest.raises(TypeError, match="input_ids.*torch.long"):
        _run(logits=logits, input_ids=torch.tensor([[0.0, 1.5]]), ir=ir)


@pytest.mark.parametrize(
    "input_ids",
    (
        torch.tensor([[False, True]], dtype=torch.bool),
        torch.tensor([[0.0 + 0.0j, 1.0 + 0.0j]], dtype=torch.complex64),
    ),
)
def test_non_integer_input_ids_rejected(input_ids: torch.Tensor) -> None:
    ir = _ir(_atom(selected_token_id=1))

    with pytest.raises(TypeError, match="input_ids.*torch.long"):
        _run(
            logits=torch.zeros((1, 2, 10), dtype=torch.float32),
            input_ids=input_ids,
            ir=ir,
        )


def test_input_ids_with_grad_history_rejected_before_probability_math() -> None:
    logits = torch.zeros((1, 2, 10), dtype=torch.float32)
    logits[0, 0, 1] = float("nan")
    ir = _ir(_atom(selected_token_id=1))
    input_ids = torch.tensor([[0.0, 1.0]], dtype=torch.float32, requires_grad=True)

    with pytest.raises(ValueError, match="input_ids.*requires_grad"):
        _run(logits=logits, input_ids=input_ids, ir=ir)


@pytest.mark.parametrize("loss_weight", (float("nan"), -1.0))
def test_invalid_atom_loss_weight_rejected_before_probability_math(
    loss_weight: float,
) -> None:
    logits = torch.zeros((1, 2, 10), dtype=torch.float32)
    logits[0, 0, 1] = float("nan")
    ir = _ir(_atom(selected_token_id=1, loss_weight=loss_weight))

    with pytest.raises(ValueError, match="loss_weight must be finite and nonnegative"):
        _run(logits=logits, input_ids=torch.tensor([[0, 1]], dtype=torch.long), ir=ir)


@pytest.mark.parametrize(
    "input_ids,error_match",
    (
        (torch.tensor([[-100, 1]], dtype=torch.long), "masked logit position"),
        (torch.tensor([[0, -100]], dtype=torch.long), "masked target position"),
    ),
)
def test_masked_logit_or_target_position_is_rejected(
    input_ids: torch.Tensor,
    error_match: str,
) -> None:
    ir = _ir(_atom(selected_token_id=1))

    with pytest.raises(ValueError, match=error_match):
        _run(logits=torch.zeros((1, 2, 10), dtype=torch.float32), input_ids=input_ids, ir=ir)


def test_mismatched_redundant_logit_position_rejected_against_label_row_map() -> None:
    ir = _ir(_atom(logit_position=1, target_position=1))
    row_map = LabelLogitRowMap(
        time_steps=2,
        vocab_size=10,
        batch_size=1,
        sample_id_to_batch_index={"sample-1": 0},
    )

    with pytest.raises(ValueError, match="logit_position.*LabelLogitRowMap"):
        _run(
            logits=torch.zeros((1, 2, 10), dtype=torch.float32),
            input_ids=torch.tensor([[0, 1]], dtype=torch.long),
            ir=ir,
            label_rows=row_map,
        )


def test_ambiguous_atom_with_coverage_disabled_uses_valid_set_marginal_only() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, -1.0, 3.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        valid_token_ids=frozenset({1, 2}),
        selected_token_id=1,
        coverage_target_weights={1: 0.9, 2: 0.1},
    )
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)

    actual = _run(logits=logits, input_ids=input_ids, ir=_ir(atom), coverage_strength=0.0)

    probs = torch.softmax(logits[0, 0], dim=-1)
    expected = -torch.log(probs[1] + probs[2])
    assert actual.item() == pytest.approx(expected.item())


def test_singleton_atom_equals_hard_ce_inside_allowed_role_union() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 1.25, -0.5, 2.0, -1.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    atom = _atom(valid_token_ids=frozenset({3}), selected_token_id=3)
    input_ids = torch.tensor([[0, 3]], dtype=torch.long)

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        role_vocab=_role_vocab(text_ids={1, 2, 3}),
    )

    expected = -torch.log_softmax(logits[0, 0], dim=-1)[3]
    assert actual.item() == pytest.approx(expected.item())


def test_coverage_strength_zero_disables_coverage() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, -1.0, 3.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    atom = _atom(
        valid_token_ids=frozenset({1, 2}),
        selected_token_id=1,
        coverage_target_weights={1: 1.0, 2: 0.0},
    )
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)

    without_weights = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(_atom(valid_token_ids=frozenset({1, 2}), selected_token_id=1)),
        coverage_strength=0.0,
    )
    with_zero_coverage = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        coverage_strength=0.0,
    )

    assert with_zero_coverage.item() == pytest.approx(without_weights.item())


def test_coverage_strength_one_adds_within_valid_ce() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, -1.0, 3.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        valid_token_ids=frozenset({1, 2}),
        selected_token_id=1,
        coverage_target_weights={1: 0.25, 2: 0.75},
    )
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)

    actual = _run(logits=logits, input_ids=input_ids, ir=_ir(atom), coverage_strength=1.0)

    manual = teacher_forcing_atom_loss(
        logits[0, 0],
        atom=atom,
        role_vocab=_role_vocab(),
        coverage_strength=1.0,
    )
    assert actual.item() == pytest.approx(manual.total.item())
    assert manual.coverage.item() > 0.0


def test_token_type_mass_adds_extra_family_pressure_beyond_valid_nll() -> None:
    role_vocab = _role_vocab(text_ids={1}, schema_ids={4}, coord_ids={6}, stop_id=9)
    logits = torch.full((1, 2, 12), -4.0, dtype=torch.float32)
    logits[0, 0, 1] = 2.0
    logits[0, 0, 4] = 3.0
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    atom = _atom(valid_token_ids=frozenset({1}), selected_token_id=1)

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        role_vocab=role_vocab,
        token_type_mass_weight=0.5,
    )

    probs = torch.softmax(logits[0, 0], dim=-1)
    family_denominator = probs[1] + probs[4] + probs[6] + probs[9]
    expected_valid = -torch.log_softmax(logits[0, 0], dim=-1)[1]
    expected_type = -torch.log(probs[1] / family_denominator)
    assert actual.item() == pytest.approx((expected_valid + 0.5 * expected_type).item())


def test_token_type_mass_excludes_control_tokens_from_family_denominator() -> None:
    role_vocab = _role_vocab(text_ids={1}, schema_ids={4}, coord_ids={6}, stop_id=9)
    logits = torch.full((1, 2, 12), -6.0, dtype=torch.float32)
    logits[0, 0, 1] = 2.0
    logits[0, 0, 4] = 1.0
    logits[0, 0, 11] = 8.0
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    atom = _atom(valid_token_ids=frozenset({1}), selected_token_id=1)

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        role_vocab=role_vocab,
        token_type_mass_weight=1.0,
    )

    probs = torch.softmax(logits[0, 0], dim=-1)
    family_denominator = probs[1] + probs[4] + probs[6] + probs[9]
    expected = -torch.log_softmax(logits[0, 0], dim=-1)[1] - torch.log(
        probs[1] / family_denominator
    )
    assert actual.item() == pytest.approx(expected.item())


def test_continuation_margin_trains_continue_vs_stop_boundary() -> None:
    role_vocab = _role_vocab(text_ids={1, 2}, schema_ids={4}, coord_ids={6}, stop_id=9)
    logits = torch.full((1, 2, 12), -5.0, dtype=torch.float32)
    logits[0, 0, 1] = 2.0
    logits[0, 0, 2] = 0.25
    logits[0, 0, 9] = 1.5
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    atom = _atom(
        valid_token_ids=frozenset({1}),
        selected_token_id=1,
        provenance={
            "continuation_boundary": True,
            "continuation_target": "continue",
            "continuation_token_ids": frozenset({1, 2}),
            "stop_token_id": 9,
        },
    )

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        role_vocab=role_vocab,
        continuation_margin_weight=0.75,
    )

    probs = torch.softmax(logits[0, 0], dim=-1)
    opener_mass = probs[1] + probs[2]
    expected_valid = -torch.log_softmax(logits[0, 0], dim=-1)[1]
    expected_margin = -torch.log(opener_mass / (opener_mass + probs[9]))
    assert actual.item() == pytest.approx(
        (expected_valid + 0.75 * expected_margin).item()
    )


def test_continuation_margin_trains_terminal_stop_against_opener() -> None:
    role_vocab = _role_vocab(text_ids={1, 2}, schema_ids={4}, coord_ids={6}, stop_id=9)
    logits = torch.full((1, 2, 12), -5.0, dtype=torch.float32)
    logits[0, 0, 1] = 2.0
    logits[0, 0, 2] = 1.0
    logits[0, 0, 9] = 0.25
    input_ids = torch.tensor([[0, 9]], dtype=torch.long)
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.STOP}),
        selected_token_role=TokenRole.STOP,
        valid_token_ids=frozenset({9}),
        selected_token_id=9,
        provenance={
            "continuation_boundary": True,
            "continuation_target": "stop",
            "continuation_token_ids": frozenset({1, 2}),
            "stop_token_id": 9,
        },
    )

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        role_vocab=role_vocab,
        continuation_margin_weight=0.5,
    )

    probs = torch.softmax(logits[0, 0], dim=-1)
    opener_mass = probs[1] + probs[2]
    expected_valid = -torch.log_softmax(logits[0, 0], dim=-1)[9]
    expected_margin = -torch.log(probs[9] / (opener_mass + probs[9]))
    assert actual.item() == pytest.approx(
        (expected_valid + 0.5 * expected_margin).item()
    )


def test_continuation_margin_requires_nonempty_boundary_opener_ids() -> None:
    role_vocab = _role_vocab(text_ids={1, 2}, schema_ids={4}, coord_ids={6}, stop_id=9)
    logits = torch.zeros((1, 2, 12), dtype=torch.float32)
    atom = _atom(
        valid_token_ids=frozenset({1}),
        selected_token_id=1,
        provenance={
            "continuation_boundary": True,
            "continuation_target": "continue",
            "continuation_token_ids": frozenset(),
            "stop_token_id": 9,
        },
    )

    with pytest.raises(ValueError, match="continuation_token_ids must be non-empty"):
        _run(
            logits=logits,
            input_ids=torch.tensor([[0, 1]], dtype=torch.long),
            ir=_ir(atom),
            role_vocab=role_vocab,
            continuation_margin_weight=1.0,
        )


def test_bbox_positive_area_penalizes_invalid_x2_mass() -> None:
    role_vocab = _role_vocab(text_ids={1}, schema_ids={4}, coord_ids={6, 7, 8}, stop_id=9)
    logits = torch.full((1, 2, 12), -5.0, dtype=torch.float32)
    logits[0, 0, 6] = 1.0
    logits[0, 0, 7] = 2.0
    logits[0, 0, 8] = 0.0
    input_ids = torch.tensor([[0, 6]], dtype=torch.long)
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.COORD}),
        selected_token_role=TokenRole.COORD,
        valid_token_ids=frozenset({6}),
        selected_token_id=6,
        coord_role="x2",
        provenance={
            "bbox_positive_area": True,
            "bbox_positive_area_valid_token_ids": frozenset({6, 8}),
            "bbox_positive_area_invalid_token_ids": frozenset({7}),
        },
    )

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        role_vocab=role_vocab,
        bbox_positive_area_weight=0.25,
    )

    probs = torch.softmax(logits[0, 0], dim=-1)
    expected_valid = -torch.log_softmax(logits[0, 0], dim=-1)[6]
    expected_area = -torch.log((probs[6] + probs[8]) / (probs[6] + probs[7] + probs[8]))
    assert actual.item() == pytest.approx((expected_valid + 0.25 * expected_area).item())


def test_bbox_positive_area_uses_explicit_coord_bins_not_sorted_token_ids() -> None:
    role_vocab = _role_vocab(
        text_ids={1},
        schema_ids={4},
        coord_ids={6, 7, 8, 10},
        stop_id=9,
    )
    logits = torch.full((1, 2, 12), -6.0, dtype=torch.float32)
    logits[0, 0, 6] = 1.0
    logits[0, 0, 7] = 4.0
    logits[0, 0, 8] = 3.5
    logits[0, 0, 10] = 0.5
    input_ids = torch.tensor([[0, 10]], dtype=torch.long)
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.COORD}),
        selected_token_role=TokenRole.COORD,
        valid_token_ids=frozenset({10}),
        selected_token_id=10,
        coord_role="y2",
        provenance={
            "bbox_positive_area": True,
            "bbox_positive_area_valid_token_ids": frozenset({6, 10}),
            "bbox_positive_area_invalid_token_ids": frozenset({7, 8}),
        },
    )

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        role_vocab=role_vocab,
        bbox_positive_area_weight=1.0,
    )

    probs = torch.softmax(logits[0, 0], dim=-1)
    explicit_area = -torch.log(
        (probs[6] + probs[10]) / (probs[6] + probs[7] + probs[8] + probs[10])
    )
    sorted_cut_area = -torch.log(
        (probs[8] + probs[10]) / (probs[6] + probs[7] + probs[8] + probs[10])
    )
    expected = -torch.log_softmax(logits[0, 0], dim=-1)[10] + explicit_area
    assert actual.item() == pytest.approx(expected.item())
    assert actual.item() != pytest.approx(
        (-torch.log_softmax(logits[0, 0], dim=-1)[10] + sorted_cut_area).item()
    )


def test_continuation_margin_contribution_uses_boundary_denominator_not_all_atoms() -> None:
    role_vocab = _role_vocab(text_ids={1, 2, 3}, schema_ids={4}, coord_ids={6}, stop_id=9)
    logits = torch.full((1, 3, 12), -5.0, dtype=torch.float32)
    logits[0, 0, 1] = 1.0
    logits[0, 0, 2] = 0.5
    logits[0, 0, 9] = 2.0
    logits[0, 1, 3] = 2.5
    input_ids = torch.tensor([[0, 1, 3]], dtype=torch.long)
    boundary_atom = _atom(
        logit_position=0,
        target_position=1,
        valid_token_ids=frozenset({1}),
        selected_token_id=1,
        provenance={
            "continuation_boundary": True,
            "continuation_target": "continue",
            "continuation_token_ids": frozenset({1, 2}),
            "stop_token_id": 9,
        },
    )
    ordinary_atom = _atom(
        logit_position=1,
        target_position=2,
        valid_token_ids=frozenset({3}),
        selected_token_id=3,
    )

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(boundary_atom, ordinary_atom),
        role_vocab=role_vocab,
        continuation_margin_weight=1.0,
    )

    valid_mean = (
        -torch.log_softmax(logits[0, 0], dim=-1)[1]
        - torch.log_softmax(logits[0, 1], dim=-1)[3]
    ) / 2.0
    probs = torch.softmax(logits[0, 0], dim=-1)
    boundary_loss = -torch.log((probs[1] + probs[2]) / (probs[1] + probs[2] + probs[9]))
    assert actual.item() == pytest.approx((valid_mean + boundary_loss).item())


def test_bbox_positive_area_contribution_uses_tail_denominator_not_all_atoms() -> None:
    role_vocab = _role_vocab(text_ids={1}, schema_ids={4}, coord_ids={6, 7, 8}, stop_id=9)
    logits = torch.full((1, 3, 12), -5.0, dtype=torch.float32)
    logits[0, 0, 6] = 1.0
    logits[0, 0, 7] = 2.0
    logits[0, 1, 1] = 2.5
    input_ids = torch.tensor([[0, 6, 1]], dtype=torch.long)
    tail_atom = _atom(
        logit_position=0,
        target_position=1,
        allowed_token_roles=frozenset({TokenRole.COORD}),
        selected_token_role=TokenRole.COORD,
        valid_token_ids=frozenset({6}),
        selected_token_id=6,
        coord_role="x2",
        provenance={
            "bbox_positive_area": True,
            "bbox_positive_area_valid_token_ids": frozenset({6, 8}),
            "bbox_positive_area_invalid_token_ids": frozenset({7}),
        },
    )
    ordinary_atom = _atom(
        logit_position=1,
        target_position=2,
        valid_token_ids=frozenset({1}),
        selected_token_id=1,
    )

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(tail_atom, ordinary_atom),
        role_vocab=role_vocab,
        bbox_positive_area_weight=1.0,
    )

    valid_mean = (
        -torch.log_softmax(logits[0, 0], dim=-1)[6]
        - torch.log_softmax(logits[0, 1], dim=-1)[1]
    ) / 2.0
    probs = torch.softmax(logits[0, 0], dim=-1)
    area_loss = -torch.log((probs[6] + probs[8]) / (probs[6] + probs[7] + probs[8]))
    assert actual.item() == pytest.approx((valid_mean + area_loss).item())


def test_bbox_positive_area_overlapping_valid_invalid_sets_reject() -> None:
    role_vocab = _role_vocab(text_ids={1}, schema_ids={4}, coord_ids={6, 7, 8}, stop_id=9)
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.COORD}),
        selected_token_role=TokenRole.COORD,
        valid_token_ids=frozenset({6}),
        selected_token_id=6,
        coord_role="x2",
        provenance={
            "bbox_positive_area": True,
            "bbox_positive_area_valid_token_ids": frozenset({6, 7}),
            "bbox_positive_area_invalid_token_ids": frozenset({7, 8}),
        },
    )

    with pytest.raises(ValueError, match="disjoint"):
        _run(
            logits=torch.zeros((1, 2, 12), dtype=torch.float32),
            input_ids=torch.tensor([[0, 6]], dtype=torch.long),
            ir=_ir(atom),
            role_vocab=role_vocab,
            bbox_positive_area_weight=1.0,
        )


def test_bbox_positive_area_partial_coord_partition_rejects() -> None:
    role_vocab = _role_vocab(text_ids={1}, schema_ids={4}, coord_ids={6, 7, 8}, stop_id=9)
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.COORD}),
        selected_token_role=TokenRole.COORD,
        valid_token_ids=frozenset({6}),
        selected_token_id=6,
        coord_role="y2",
        provenance={
            "bbox_positive_area": True,
            "bbox_positive_area_valid_token_ids": frozenset({6}),
            "bbox_positive_area_invalid_token_ids": frozenset({7}),
        },
    )

    with pytest.raises(ValueError, match="partition all coord token ids"):
        _run(
            logits=torch.zeros((1, 2, 12), dtype=torch.float32),
            input_ids=torch.tensor([[0, 6]], dtype=torch.long),
            ir=_ir(atom),
            role_vocab=role_vocab,
            bbox_positive_area_weight=1.0,
        )


def test_bbox_positive_area_selected_tail_coord_in_invalid_set_rejects() -> None:
    role_vocab = _role_vocab(text_ids={1}, schema_ids={4}, coord_ids={6, 7, 8}, stop_id=9)
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.COORD}),
        selected_token_role=TokenRole.COORD,
        valid_token_ids=frozenset({6}),
        selected_token_id=6,
        coord_role="x2",
        provenance={
            "bbox_positive_area": True,
            "bbox_positive_area_valid_token_ids": frozenset({7, 8}),
            "bbox_positive_area_invalid_token_ids": frozenset({6}),
        },
    )

    with pytest.raises(ValueError, match="selected_token_id must be in bbox_positive_area_valid_token_ids"):
        _run(
            logits=torch.zeros((1, 2, 12), dtype=torch.float32),
            input_ids=torch.tensor([[0, 6]], dtype=torch.long),
            ir=_ir(atom),
            role_vocab=role_vocab,
            bbox_positive_area_weight=1.0,
        )


def test_src_training_objectives_runner_objective_runner_owns_execution() -> None:
    assert ObjectiveRunner.__module__ == "src.training.objectives.runner"
    result = ObjectiveRunner().run(
        logits=torch.zeros((1, 2, 10), dtype=torch.float32),
        supervision=SupervisionBatch(spans=(_span(_ir(_atom())),)),
        objectives=(
            ObjectiveSpec(
                "teacher_forcing",
                config={
                    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
                    "role_vocab": _role_vocab(),
                },
            ),
        ),
        sample_id_to_batch_index={"sample-1": 0},
    )
    assert result.objectives["teacher_forcing"].span_count == 1


def test_no_new_parallel_runner_package_is_created_under_src_objectives() -> None:
    assert importlib.util.find_spec("src.objectives") is None

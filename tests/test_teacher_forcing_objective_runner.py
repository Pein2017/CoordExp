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
        coord_role=None,
        provenance={},
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

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from src.metrics.events import flatten_metric_events
from src.trainers.metrics import teacher_forcing as trainer_teacher_forcing_metrics
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
        coord_role=None,
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
    label_rows: LabelLogitRowMap | None = None,
) -> torch.Tensor:
    result = _run_result(
        logits=logits,
        input_ids=input_ids,
        ir=ir,
        role_vocab=role_vocab,
        coverage_strength=coverage_strength,
        label_rows=label_rows,
    )
    assert "teacher_forcing" in result.objectives
    assert result.state["label_row_map"] is not None
    return result.objectives["teacher_forcing"].loss


def _run_result(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    ir: TeacherForcingTargetIR,
    role_vocab: RoleVocab | None = None,
    coverage_strength: float = 0.0,
    label_rows: LabelLogitRowMap | None = None,
    extra_config: dict[str, object] | None = None,
):
    config: dict[str, object] = {
        "input_ids": input_ids,
        "role_vocab": role_vocab or _role_vocab(),
        "coverage_strength": coverage_strength,
    }
    if extra_config:
        config.update(extra_config)
    return ObjectiveRunner().run(
        logits=logits,
        supervision=SupervisionBatch(spans=(_span(ir),), batch_id="batch-1"),
        objectives=(ObjectiveSpec("teacher_forcing", config=config),),
        label_rows=label_rows,
        sample_id_to_batch_index={"sample-1": 0},
    )


def _manual_family_masses(
    logits: torch.Tensor,
    role_vocab: RoleVocab,
) -> dict[str, torch.Tensor]:
    family_logits = torch.stack(
        (
            torch.logsumexp(
                logits[list(sorted(role_vocab.schema_token_ids))], dim=-1
            ),
            torch.logsumexp(logits[list(sorted(role_vocab.text_token_ids))], dim=-1),
            torch.logsumexp(logits[list(sorted(role_vocab.coord_token_ids))], dim=-1),
            torch.logsumexp(logits[list(sorted(role_vocab.stop_token_ids))], dim=-1),
        )
    )
    family_probs = torch.softmax(family_logits, dim=-1)
    return {
        "schema": family_probs[0],
        "desc": family_probs[1],
        "coord": family_probs[2],
        "stop": family_probs[3],
    }


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


def test_token_type_mass_rewards_selected_family_over_competing_families() -> None:
    role_vocab = _role_vocab(
        text_ids={1, 2},
        schema_ids={3},
        coord_ids={4},
        stop_id=5,
    )
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({1}),
        selected_token_id=1,
    )
    good_logits = torch.full((8,), -4.0, dtype=torch.float32)
    good_logits[1] = 5.0
    good_logits[2] = 4.0
    good_logits[3] = 0.0
    good_logits[4] = -1.0
    good_logits[5] = -2.0
    bad_logits = good_logits.clone()
    bad_logits[1] = 0.0
    bad_logits[2] = -1.0
    bad_logits[3] = 5.0
    good_expected = _manual_family_masses(good_logits, role_vocab)
    bad_expected = _manual_family_masses(bad_logits, role_vocab)

    good = teacher_forcing_atom_loss(
        good_logits,
        atom=atom,
        role_vocab=role_vocab,
        coverage_strength=0.0,
        token_type_mass_enabled=True,
    )
    bad = teacher_forcing_atom_loss(
        bad_logits,
        atom=atom,
        role_vocab=role_vocab,
        coverage_strength=0.0,
        token_type_mass_enabled=True,
    )

    assert good.target_family == "desc"
    assert good.token_type_mass.item() < bad.token_type_mass.item()
    assert good.token_type_mass.item() == pytest.approx(
        -torch.log(good_expected["desc"]).item()
    )
    assert bad.token_type_mass.item() == pytest.approx(
        -torch.log(bad_expected["desc"]).item()
    )
    assert good.target_family_mass.item() == pytest.approx(
        good_expected["desc"].item()
    )
    assert bad.target_family_mass.item() == pytest.approx(bad_expected["desc"].item())
    for family in ("schema", "desc", "coord", "stop"):
        assert good.family_masses[family].item() == pytest.approx(
            good_expected[family].item()
        )
        assert bad.family_masses[family].item() == pytest.approx(
            bad_expected[family].item()
        )


@pytest.mark.parametrize(
    ("token_type_mass_weight", "error_type"),
    (
        (True, TypeError),
        (False, TypeError),
        ("1.0", TypeError),
        (None, TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (-1.0, ValueError),
    ),
)
def test_token_type_mass_weight_rejects_invalid_direct_values(
    token_type_mass_weight: object,
    error_type: type[Exception],
) -> None:
    logits = torch.zeros((10,), dtype=torch.float32)
    atom = _atom()

    with pytest.raises(error_type, match="token_type_mass_weight"):
        teacher_forcing_atom_loss(
            logits,
            atom=atom,
            role_vocab=_role_vocab(),
            coverage_strength=0.0,
            token_type_mass_enabled=True,
            token_type_mass_weight=token_type_mass_weight,  # type: ignore[arg-type]
        )


def test_token_type_mass_excludes_out_of_family_logits_from_denominator() -> None:
    role_vocab = _role_vocab(
        text_ids={2},
        schema_ids={1},
        coord_ids={3},
        stop_id=4,
    )
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({2}),
        selected_token_id=2,
    )
    base_logits = torch.full((10,), -3.0, dtype=torch.float32)
    base_logits[1] = 0.4
    base_logits[2] = 1.2
    base_logits[3] = -0.3
    base_logits[4] = -0.5
    spiked_logits = base_logits.clone()
    spiked_logits[8] = 100.0

    expected_masses = _manual_family_masses(base_logits, role_vocab)
    expected = -torch.log(expected_masses["desc"])

    base = teacher_forcing_atom_loss(
        base_logits,
        atom=atom,
        role_vocab=role_vocab,
        coverage_strength=0.0,
        token_type_mass_enabled=True,
    )
    spiked = teacher_forcing_atom_loss(
        spiked_logits,
        atom=atom,
        role_vocab=role_vocab,
        coverage_strength=0.0,
        token_type_mass_enabled=True,
    )

    assert base.token_type_mass.item() == pytest.approx(expected.item(), abs=1e-5)
    assert spiked.token_type_mass.item() == pytest.approx(expected.item(), abs=1e-5)


def test_token_type_mass_stop_token_uses_stop_family_not_schema_family() -> None:
    role_vocab = _role_vocab(
        text_ids={2},
        schema_ids={1},
        coord_ids={3},
        stop_id=4,
    )
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.STOP}),
        selected_token_role=TokenRole.STOP,
        valid_token_ids=frozenset({4}),
        selected_token_id=4,
    )
    logits = torch.full((8,), -4.0, dtype=torch.float32)
    logits[1] = 10.0
    logits[2] = 0.0
    logits[3] = 0.0
    logits[4] = -2.0

    loss = teacher_forcing_atom_loss(
        logits,
        atom=atom,
        role_vocab=role_vocab,
        coverage_strength=0.0,
        token_type_mass_enabled=True,
    )

    assert loss.target_family == "stop"
    assert loss.family_masses["schema"].item() > 0.99
    assert loss.token_type_mass.item() > 10.0


def test_objective_emits_token_type_mass_events_with_active_atom_denominators() -> None:
    role_vocab = _role_vocab(
        text_ids={1, 2},
        schema_ids={4},
        coord_ids={6},
        stop_id=9,
    )
    desc_atom = _atom(
        logit_position=0,
        target_position=1,
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({1}),
        selected_token_id=1,
    )
    coord_atom = _atom(
        logit_position=2,
        target_position=3,
        allowed_token_roles=frozenset({TokenRole.COORD}),
        selected_token_role=TokenRole.COORD,
        valid_token_ids=frozenset({6}),
        selected_token_id=6,
    )
    logits = torch.full((1, 4, 12), -5.0, dtype=torch.float32)
    logits[0, 0, 1] = 3.0
    logits[0, 0, 2] = 2.0
    logits[0, 0, 4] = 0.5
    logits[0, 0, 6] = -1.0
    logits[0, 0, 9] = -2.0
    logits[0, 2, 1] = -0.5
    logits[0, 2, 2] = -0.25
    logits[0, 2, 4] = -1.0
    logits[0, 2, 6] = 2.5
    logits[0, 2, 9] = -3.0

    result = _run_result(
        logits=logits,
        input_ids=torch.tensor([[0, 1, 0, 6]], dtype=torch.long),
        ir=_ir(desc_atom, coord_atom),
        role_vocab=role_vocab,
        extra_config={
            "token_type_mass_enabled": True,
            "token_type_mass_weight": 0.2,
        },
    )

    objective = result.objectives["teacher_forcing"]
    events_by_key = {event.key: event for event in objective.metric_events}
    flat = flatten_metric_events(objective.metric_events)
    raw = events_by_key["teacher_forcing/loss/token_type_mass"]
    contribution = events_by_key[
        "teacher_forcing/loss/token_type_mass/contribution"
    ]

    assert raw.reducer == "weighted_mean"
    assert contribution.reducer == "weighted_mean"
    assert raw.denominator == pytest.approx(2.0)
    assert contribution.denominator == pytest.approx(2.0)
    assert contribution.numerator == pytest.approx(raw.numerator * 0.2)
    assert raw.metric_surface == "objective_loss"
    assert contribution.metric_surface == "objective_loss"
    assert raw.diagnostic_only is False
    assert contribution.diagnostic_only is False
    assert "teacher_forcing/type/desc_mass_at_desc" in events_by_key
    assert "teacher_forcing/type/coord_mass_at_coord" in events_by_key
    assert "teacher_forcing/type/schema_mass_at_schema" not in events_by_key
    assert "teacher_forcing/type/stop_mass_at_stop" not in events_by_key
    assert events_by_key["teacher_forcing/type/desc_mass_at_desc"].denominator == 1.0
    assert events_by_key["teacher_forcing/type/coord_mass_at_coord"].denominator == 1.0
    assert all(
        events_by_key[key].metric_surface == "type_family_mass"
        for key in (
            "teacher_forcing/type/desc_mass_at_desc",
            "teacher_forcing/type/coord_mass_at_coord",
        )
    )
    assert all(
        events_by_key[key].diagnostic_only is False
        for key in (
            "teacher_forcing/type/desc_mass_at_desc",
            "teacher_forcing/type/coord_mass_at_coord",
        )
    )
    assert flat["teacher_forcing/loss/token_type_mass/contribution"] == pytest.approx(
        flat["teacher_forcing/loss/token_type_mass"] * 0.2
    )
    assert not any(key.endswith("_weighted") for key in flat)


def test_trainer_token_type_mass_helpers_read_terms_config() -> None:
    object_cfg = SimpleNamespace(
        terms=SimpleNamespace(
            token_type_mass=SimpleNamespace(enabled=True, weight=0.2),
        ),
    )
    mapping_cfg = {
        "terms": {
            "token_type_mass": {
                "enabled": True,
                "weight": 0.2,
            },
        },
    }

    assert trainer_teacher_forcing_metrics._token_type_mass_enabled(object_cfg) is True
    assert trainer_teacher_forcing_metrics._token_type_mass_weight(
        object_cfg
    ) == pytest.approx(0.2)
    assert trainer_teacher_forcing_metrics._token_type_mass_enabled(mapping_cfg) is True
    assert trainer_teacher_forcing_metrics._token_type_mass_weight(
        mapping_cfg
    ) == pytest.approx(0.2)
    assert trainer_teacher_forcing_metrics._token_type_mass_enabled(None) is False
    assert trainer_teacher_forcing_metrics._token_type_mass_weight(None) == pytest.approx(
        1.0
    )


@pytest.mark.parametrize("config_style", ("object", "mapping"))
@pytest.mark.parametrize("enabled", ("false", 1, None))
def test_trainer_token_type_mass_enabled_rejects_non_bool_values(
    config_style: str,
    enabled: object,
) -> None:
    cfg = _token_type_mass_cfg(
        config_style,
        enabled=enabled,
        weight=1.0,
    )

    with pytest.raises(TypeError, match=r"token_type_mass\.enabled"):
        trainer_teacher_forcing_metrics._token_type_mass_enabled(cfg)


@pytest.mark.parametrize("config_style", ("object", "mapping"))
@pytest.mark.parametrize(
    ("weight", "error_type"),
    (
        ("1.0", TypeError),
        (True, TypeError),
        (None, TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (-1.0, ValueError),
    ),
)
def test_trainer_token_type_mass_weight_rejects_invalid_values(
    config_style: str,
    weight: object,
    error_type: type[Exception],
) -> None:
    cfg = _token_type_mass_cfg(
        config_style,
        enabled=True,
        weight=weight,
    )

    with pytest.raises(error_type, match=r"token_type_mass\.weight"):
        trainer_teacher_forcing_metrics._token_type_mass_weight(cfg)


def _token_type_mass_cfg(
    config_style: str,
    *,
    enabled: object,
    weight: object,
) -> object:
    if config_style == "object":
        return SimpleNamespace(
            terms=SimpleNamespace(
                token_type_mass=SimpleNamespace(enabled=enabled, weight=weight),
            ),
        )
    if config_style == "mapping":
        return {
            "terms": {
                "token_type_mass": {
                    "enabled": enabled,
                    "weight": weight,
                },
            },
        }
    raise AssertionError(f"unknown config style: {config_style}")


def test_trainer_supervision_rejects_duplicate_sample_ids_before_mapping() -> None:
    target_irs = (_ir(_atom()), _ir(_atom()))

    with pytest.raises(ValueError, match="duplicate sample ID"):
        trainer_teacher_forcing_metrics._build_teacher_forcing_supervision(
            target_irs=target_irs,
            sample_ids=("sample-1", "sample-1"),
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

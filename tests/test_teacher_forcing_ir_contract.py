from __future__ import annotations

import pytest
import torch

from src.training.teacher_forcing.constants import (
    MARGINAL_SCOPE_SAMPLED_PATH_NEXT_TOKEN,
    TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
    TEACHER_FORCING_TARGET_IR_KEY,
)
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.validation import validate_target_ir
from src.training.teacher_forcing.vocab import RoleVocab


def make_test_role_vocab(
    *,
    text_ids: set[int] | None = None,
    schema_ids: set[int] | None = None,
    coord_ids: set[int] | None = None,
    stop_id: int = 401,
) -> RoleVocab:
    return RoleVocab(
        schema_token_ids=frozenset(schema_ids or {201}),
        text_token_ids=frozenset(text_ids or {101}),
        coord_token_ids=frozenset(coord_ids or {301}),
        stop_token_id=stop_id,
    )


def make_atom(
    *,
    batch_index: int = 0,
    logit_position: int = 0,
    target_position: int = 1,
    allowed_token_roles: frozenset[TokenRole] = frozenset({TokenRole.TEXT}),
    selected_token_role: TokenRole = TokenRole.TEXT,
    valid_token_ids: frozenset[int] = frozenset({101}),
    selected_token_id: int = 101,
    latent_valid_token_ids: frozenset[int] | None = None,
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
        latent_valid_token_ids=valid_token_ids if latent_valid_token_ids is None else latent_valid_token_ids,
        coverage_target_weights=None,
        loss_tags=frozenset({"singleton"}),
        loss_weight=loss_weight,
        coord_role=None,
        provenance={},
    )


def make_ir(atom: SupervisionAtom) -> TeacherForcingTargetIR:
    return TeacherForcingTargetIR(schema_version=1, atoms=(atom,), metadata={})


def test_target_ir_key_and_marginal_scope_are_canonical() -> None:
    assert TEACHER_FORCING_TARGET_IR_KEY == "teacher_forcing_target_ir"
    assert MARGINAL_SCOPE_SAMPLED_PATH_NEXT_TOKEN == "sampled_path_next_token"


def test_token_role_contract_is_exact() -> None:
    assert [role.value for role in TokenRole] == ["SCHEMA", "TEXT", "COORD", "STOP"]


def test_atom_requires_causal_next_token_positions() -> None:
    atom = make_atom(logit_position=3, target_position=5)

    with pytest.raises(ValueError, match=r"target_position = logit_position \+ 1"):
        validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 9, 9, 9, 9, 101]]))


def test_selected_token_must_match_input_ids_at_canonical_target_position() -> None:
    atom = make_atom(target_position=2, logit_position=1, selected_token_id=101)

    with pytest.raises(ValueError, match="selected_token_id must match input_ids"):
        validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 9, 102]]))


@pytest.mark.parametrize("field_name", ["batch_index", "logit_position", "target_position"])
def test_negative_atom_positions_are_rejected_before_tensor_indexing(field_name: str) -> None:
    kwargs = {
        "batch_index": 0,
        "logit_position": 0,
        "target_position": 1,
    }
    kwargs[field_name] = -1
    atom = make_atom(**kwargs)

    with pytest.raises(ValueError, match=rf"atoms\[0\].*{field_name}"):
        validate_target_ir(
            make_ir(atom),
            input_ids=torch.tensor([[9, 101], [9, 101]]),
            role_vocab=make_test_role_vocab(),
        )


def test_selected_token_must_be_inside_valid_token_ids() -> None:
    atom = make_atom(valid_token_ids=frozenset({102}), selected_token_id=101)

    with pytest.raises(ValueError, match="selected_token_id must be in valid_token_ids"):
        validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 101]]))


def test_allowed_token_roles_must_be_nonempty() -> None:
    atom = make_atom(
        allowed_token_roles=frozenset(),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({101}),
        selected_token_id=101,
    )

    with pytest.raises(ValueError, match="allowed_token_roles must be nonempty"):
        validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 101]]))


def test_selected_token_role_must_be_allowed() -> None:
    atom = make_atom(
        allowed_token_roles=frozenset({TokenRole.SCHEMA}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({201}),
        selected_token_id=201,
    )

    with pytest.raises(ValueError, match="selected_token_role must be in allowed_token_roles"):
        validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 201]]))


def test_trainable_atoms_require_nonempty_valid_token_ids() -> None:
    atom = make_atom(valid_token_ids=frozenset(), selected_token_id=101)

    with pytest.raises(ValueError, match="valid_token_ids must be nonempty"):
        validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 101]]))


def test_trainable_atoms_require_role_vocab_for_fail_closed_validation() -> None:
    atom = make_atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({101}),
        selected_token_id=101,
    )

    with pytest.raises(ValueError, match="role_vocab is required"):
        validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 101]]))


def test_role_vocab_rejects_schema_ownership_of_im_end_stop_id() -> None:
    with pytest.raises(ValueError, match="stop_token_id must not appear in schema_token_ids"):
        RoleVocab(
            schema_token_ids=frozenset({201, 401}),
            text_token_ids=frozenset({101}),
            coord_token_ids=frozenset({301}),
            stop_token_id=401,
        )


def test_valid_tokens_must_belong_to_allowed_role_vocab() -> None:
    vocab = make_test_role_vocab(
        text_ids={101},
        schema_ids={201},
        coord_ids={301},
        stop_id=401,
    )
    atom = make_atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({101, 301}),
        selected_token_id=101,
    )

    with pytest.raises(ValueError, match="valid_token_ids must be inside allowed role vocab"):
        validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 101]]), role_vocab=vocab)


def test_text_schema_is_the_only_supported_mixed_role_set() -> None:
    vocab = make_test_role_vocab(text_ids={101}, schema_ids={201})
    atom = make_atom(
        allowed_token_roles=frozenset({TokenRole.TEXT, TokenRole.SCHEMA}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({101, 201}),
        selected_token_id=101,
    )

    validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 101]]), role_vocab=vocab)


@pytest.mark.parametrize(
    "roles",
    [
        frozenset({TokenRole.TEXT, TokenRole.COORD}),
        frozenset({TokenRole.SCHEMA, TokenRole.COORD}),
        frozenset({TokenRole.COORD, TokenRole.STOP}),
        frozenset({TokenRole.TEXT, TokenRole.STOP}),
    ],
)
def test_unsupported_mixed_role_sets_are_rejected(roles: frozenset[TokenRole]) -> None:
    atom = make_atom(
        allowed_token_roles=roles,
        selected_token_role=next(iter(roles)),
        valid_token_ids=frozenset({101}),
        selected_token_id=101,
    )

    with pytest.raises(ValueError, match="unsupported mixed allowed_token_roles"):
        validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 101]]))


def test_stop_vocab_ownership_is_separate_from_schema() -> None:
    vocab = make_test_role_vocab(schema_ids={201}, stop_id=401)
    atom = make_atom(
        allowed_token_roles=frozenset({TokenRole.SCHEMA}),
        selected_token_role=TokenRole.SCHEMA,
        valid_token_ids=frozenset({401}),
        selected_token_id=401,
    )

    with pytest.raises(ValueError, match="valid_token_ids must be inside allowed role vocab"):
        validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 401]]), role_vocab=vocab)


def test_stop_atoms_can_only_target_configured_im_end_token() -> None:
    vocab = make_test_role_vocab(stop_id=401)
    atom = make_atom(
        allowed_token_roles=frozenset({TokenRole.STOP}),
        selected_token_role=TokenRole.STOP,
        valid_token_ids=frozenset({999}),
        selected_token_id=999,
    )

    with pytest.raises(ValueError, match=r"STOP atoms can target only configured <\|im_end\|> token id"):
        validate_target_ir(make_ir(atom), input_ids=torch.tensor([[9, 999]]), role_vocab=vocab)


def test_canonical_target_position_keeps_redundant_logit_position_validation() -> None:
    atom = make_atom(logit_position=1, target_position=2, selected_token_id=101)

    validate_target_ir(
        make_ir(atom),
        input_ids=torch.tensor([[9, 9, 101]]),
        role_vocab=make_test_role_vocab(),
    )


def test_ir_canonicalizes_mutable_constructor_inputs() -> None:
    coverage_target_weights = {101: 1.0}
    provenance = {"source": "builder"}
    metadata = {"run": "smoke"}
    atom = SupervisionAtom(
        batch_index=0,
        logit_position=0,
        target_position=1,
        allowed_token_roles={TokenRole.TEXT},
        selected_token_role=TokenRole.TEXT,
        valid_token_ids={101},
        selected_token_id=101,
        latent_valid_token_ids={101, 102},
        coverage_target_weights=coverage_target_weights,
        loss_tags={"singleton"},
        loss_weight=1.0,
        coord_role=None,
        provenance=provenance,
    )
    atoms = [atom]
    ir = TeacherForcingTargetIR(
        schema_version=TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
        atoms=atoms,
        metadata=metadata,
    )

    coverage_target_weights[101] = 2.0
    provenance["source"] = "mutated"
    metadata["run"] = "mutated"
    atoms.clear()

    assert atom.allowed_token_roles == frozenset({TokenRole.TEXT})
    assert atom.valid_token_ids == frozenset({101})
    assert atom.latent_valid_token_ids == frozenset({101, 102})
    assert atom.loss_tags == frozenset({"singleton"})
    assert atom.coverage_target_weights[101] == 1.0
    assert atom.provenance["source"] == "builder"
    assert ir.atoms == (atom,)
    assert ir.metadata["run"] == "smoke"


def test_ir_mapping_fields_are_shallowly_immutable() -> None:
    atom = SupervisionAtom(
        batch_index=0,
        logit_position=0,
        target_position=1,
        allowed_token_roles={TokenRole.TEXT},
        selected_token_role=TokenRole.TEXT,
        valid_token_ids={101},
        selected_token_id=101,
        latent_valid_token_ids={101},
        coverage_target_weights={101: 1.0},
        loss_tags={"singleton"},
        loss_weight=1.0,
        coord_role=None,
        provenance={"source": "builder"},
    )
    ir = TeacherForcingTargetIR(
        schema_version=TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
        atoms=[atom],
        metadata={"run": "smoke"},
    )

    with pytest.raises(TypeError):
        atom.coverage_target_weights[101] = 2.0
    with pytest.raises(TypeError):
        atom.provenance["source"] = "mutated"
    with pytest.raises(TypeError):
        ir.metadata["run"] = "mutated"


def test_validate_target_ir_rejects_unsupported_schema_version() -> None:
    atom = make_atom()
    ir = TeacherForcingTargetIR(schema_version=999, atoms=(atom,), metadata={})

    with pytest.raises(ValueError, match="schema_version"):
        validate_target_ir(ir, input_ids=torch.tensor([[9, 101]]), role_vocab=make_test_role_vocab())

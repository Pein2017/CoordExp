from __future__ import annotations

import math
from numbers import Real
from typing import Any

from .constants import TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION
from .ir import SupervisionAtom, TeacherForcingTargetIR
from .roles import TokenRole
from .vocab import RoleVocab

_SUPPORTED_MIXED_ROLE_SET = frozenset({TokenRole.TEXT, TokenRole.SCHEMA})


def validate_target_ir(
    target_ir: TeacherForcingTargetIR,
    *,
    input_ids: Any,
    role_vocab: RoleVocab | None = None,
) -> None:
    if target_ir.schema_version != TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION:
        raise ValueError(
            "teacher_forcing_target_ir.schema_version: "
            f"unsupported schema_version {target_ir.schema_version}; "
            f"expected {TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION}"
        )
    for atom_index, atom in enumerate(target_ir.atoms):
        _validate_atom(atom_index, atom, input_ids=input_ids, role_vocab=role_vocab)


def _validate_atom(
    atom_index: int,
    atom: SupervisionAtom,
    *,
    input_ids: Any,
    role_vocab: RoleVocab | None,
) -> None:
    prefix = f"teacher_forcing_target_ir.atoms[{atom_index}]"
    _validate_loss_weight(atom.loss_weight, prefix=prefix)
    _validate_nonnegative_index(atom.batch_index, field_name="batch_index", prefix=prefix)
    _validate_nonnegative_index(atom.logit_position, field_name="logit_position", prefix=prefix)
    _validate_nonnegative_index(atom.target_position, field_name="target_position", prefix=prefix)
    if atom.target_position != atom.logit_position + 1:
        raise ValueError(f"{prefix}: target_position = logit_position + 1 is required")
    if not atom.allowed_token_roles:
        raise ValueError(f"{prefix}: allowed_token_roles must be nonempty")
    if atom.selected_token_role not in atom.allowed_token_roles:
        raise ValueError(f"{prefix}: selected_token_role must be in allowed_token_roles")
    if len(atom.allowed_token_roles) > 1 and atom.allowed_token_roles != _SUPPORTED_MIXED_ROLE_SET:
        raise ValueError(f"{prefix}: unsupported mixed allowed_token_roles")
    if not atom.valid_token_ids:
        raise ValueError(f"{prefix}: valid_token_ids must be nonempty for target-like atoms")

    live_token_id = _input_token_id(input_ids, atom.batch_index, atom.target_position, prefix=prefix)
    if atom.selected_token_id != live_token_id:
        raise ValueError(f"{prefix}: selected_token_id must match input_ids at target_position")
    if atom.selected_token_id not in atom.valid_token_ids:
        raise ValueError(f"{prefix}: selected_token_id must be in valid_token_ids")

    if role_vocab is None:
        raise ValueError(f"{prefix}: role_vocab is required for target-like atoms")
    if TokenRole.STOP in atom.allowed_token_roles:
        _validate_stop_atom(atom, role_vocab=role_vocab, prefix=prefix)
    _validate_valid_ids_inside_role_vocab(atom, role_vocab=role_vocab, prefix=prefix)


def _validate_stop_atom(atom: SupervisionAtom, *, role_vocab: RoleVocab, prefix: str) -> None:
    configured_stop_ids = role_vocab.stop_token_ids
    if atom.valid_token_ids != configured_stop_ids or atom.selected_token_id != role_vocab.stop_token_id:
        raise ValueError(f"{prefix}: STOP atoms can target only configured <|im_end|> token id")


def _validate_valid_ids_inside_role_vocab(
    atom: SupervisionAtom,
    *,
    role_vocab: RoleVocab,
    prefix: str,
) -> None:
    allowed_role_vocab = role_vocab.token_ids_for_roles(atom.allowed_token_roles)
    if not atom.valid_token_ids.issubset(allowed_role_vocab):
        raise ValueError(f"{prefix}: valid_token_ids must be inside allowed role vocab")


def _validate_nonnegative_index(value: int, *, field_name: str, prefix: str) -> None:
    if value < 0:
        raise ValueError(f"{prefix}: {field_name} must be nonnegative")


def _validate_loss_weight(value: float, *, prefix: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{prefix}: loss_weight must be a finite nonnegative real number")
    loss_weight = float(value)
    if not math.isfinite(loss_weight) or loss_weight < 0.0:
        raise ValueError(f"{prefix}: loss_weight must be a finite nonnegative real number")


def _input_token_id(input_ids: Any, batch_index: int, target_position: int, *, prefix: str) -> int:
    try:
        value = input_ids[batch_index, target_position]
    except Exception as exc:
        raise ValueError(f"{prefix}: target_position is out of bounds for input_ids") from exc
    if hasattr(value, "item"):
        value = value.item()
    return int(value)

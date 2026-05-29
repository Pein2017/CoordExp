"""Teacher-forcing fixed eval/probe encoded-cache contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from src.training.teacher_forcing.constants import (
    TEACHER_FORCING_TARGET_IR_KEY,
    TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
)
from src.training.teacher_forcing.ir import TeacherForcingTargetIR
from src.training.teacher_forcing.validation import validate_target_ir
from src.training.teacher_forcing.vocab import RoleVocab


def build_fixed_eval_probe_cache_key(
    *,
    tokenizer_fingerprint: str,
    chat_template_fingerprint: str,
    serialization_policy: str,
    description_normalization_policy: str,
    rollin_policy: str,
    rollin_policy_version: int,
    rollin_epoch: int,
    rollin_base_seed: int | None = None,
    target_ir_schema_version: int = TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
    max_length: int,
    rollin_seed: int | None = None,
) -> dict[str, Any]:
    """Build a complete fixed eval/probe cache key for teacher-forcing payloads."""

    base_seed = rollin_base_seed if rollin_base_seed is not None else rollin_seed
    key: dict[str, Any] = {
        "key_version": 1,
        "cache_scope": "teacher_forcing_fixed_eval_probe",
        "tokenizer_fingerprint": _require_nonempty_str(
            tokenizer_fingerprint,
            field_name="tokenizer_fingerprint",
        ),
        "chat_template_fingerprint": _require_nonempty_str(
            chat_template_fingerprint,
            field_name="chat_template_fingerprint",
        ),
        "serialization_policy": _require_nonempty_str(
            serialization_policy,
            field_name="serialization_policy",
        ),
        "description_normalization_policy": _require_nonempty_str(
            description_normalization_policy,
            field_name="description_normalization_policy",
        ),
        "rollin_policy": {
            "name": _require_nonempty_str(
                rollin_policy,
                field_name="rollin_policy.name",
            ),
            "version": _require_int(
                rollin_policy_version,
                field_name="rollin_policy.version",
                minimum=1,
            ),
            "base_seed": _require_int(
                base_seed,
                field_name="rollin_policy.base_seed",
                minimum=0,
            ),
            "epoch": _require_int(
                rollin_epoch,
                field_name="rollin_policy.epoch",
                minimum=0,
            ),
        },
        "target_ir_schema_version": _require_int(
            target_ir_schema_version,
            field_name="target_ir_schema_version",
            minimum=1,
        ),
        "max_length": _require_int(max_length, field_name="max_length", minimum=1),
    }
    key["fingerprint_sha256"] = _stable_sha256(key)
    return key


def build_fixed_eval_probe_payload(
    *,
    input_ids: Any,
    teacher_forcing_target_ir: TeacherForcingTargetIR,
) -> dict[str, Any]:
    """Build the minimal cache payload required for fixed eval/probe reuse."""

    if type(teacher_forcing_target_ir) is not TeacherForcingTargetIR:
        raise TypeError("teacher_forcing_target_ir must be a TeacherForcingTargetIR")
    return {
        "input_ids": _copy_input_ids(input_ids),
        TEACHER_FORCING_TARGET_IR_KEY: teacher_forcing_target_ir,
    }


def load_fixed_eval_probe_payload(
    payload: Mapping[str, Any],
    *,
    role_vocab: RoleVocab,
) -> dict[str, Any]:
    """Validate and return a fixed eval/probe teacher-forcing cache payload."""

    if not isinstance(payload, Mapping):
        raise TypeError(
            "teacher_forcing fixed eval/probe cache payload must be a mapping"
        )
    if "input_ids" not in payload:
        raise ValueError(
            "teacher_forcing fixed eval/probe cache payload missing input_ids"
        )
    if TEACHER_FORCING_TARGET_IR_KEY not in payload:
        raise ValueError(
            "teacher_forcing fixed eval/probe cache payload missing "
            f"{TEACHER_FORCING_TARGET_IR_KEY}"
        )

    input_ids = _copy_input_ids(payload["input_ids"])
    target_ir = payload[TEACHER_FORCING_TARGET_IR_KEY]
    if type(target_ir) is not TeacherForcingTargetIR:
        raise TypeError("teacher_forcing_target_ir must be a TeacherForcingTargetIR")

    validate_target_ir(
        target_ir,
        input_ids=_validation_input_ids(input_ids),
        role_vocab=role_vocab,
    )
    return {
        "input_ids": input_ids,
        TEACHER_FORCING_TARGET_IR_KEY: target_ir,
    }


class _SequenceInputIds:
    def __init__(self, input_ids: tuple[Any, ...]) -> None:
        self._input_ids = input_ids

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2:
            batch_index, target_position = key
            return self._input_ids[batch_index][target_position]
        return self._input_ids[key]


def _require_nonempty_str(value: Any, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a non-empty string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be a non-empty string")
    return stripped


def _require_int(value: Any, *, field_name: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        comparator = ">=" if minimum == 0 else ">"
        threshold = minimum if minimum == 0 else minimum - 1
        raise ValueError(f"{field_name} must be {comparator} {threshold}")
    return value


def _copy_input_ids(input_ids: Any) -> Any:
    if hasattr(input_ids, "detach") and hasattr(input_ids, "clone"):
        return input_ids.detach().clone()
    if isinstance(input_ids, list | tuple):
        return _freeze_sequence(input_ids)
    return copy.deepcopy(input_ids)


def _freeze_sequence(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_freeze_sequence(item) for item in value)
    return value


def _validation_input_ids(input_ids: Any) -> Any:
    if isinstance(input_ids, tuple):
        return _SequenceInputIds(input_ids)
    return input_ids


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "build_fixed_eval_probe_cache_key",
    "build_fixed_eval_probe_payload",
    "load_fixed_eval_probe_payload",
]

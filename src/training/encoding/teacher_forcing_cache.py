"""Teacher-forcing fixed eval/probe encoded-cache contracts."""

from __future__ import annotations

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
    rollin_seed: int,
    rollin_epoch: int,
    target_ir_schema_version: int = TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
    max_length: int,
) -> dict[str, Any]:
    """Build a complete fixed eval/probe cache key for teacher-forcing payloads."""

    key: dict[str, Any] = {
        "key_version": 1,
        "cache_scope": "teacher_forcing_fixed_eval_probe",
        "tokenizer_fingerprint": str(tokenizer_fingerprint),
        "chat_template_fingerprint": str(chat_template_fingerprint),
        "serialization_policy": str(serialization_policy),
        "description_normalization_policy": str(description_normalization_policy),
        "rollin_policy": {
            "name": str(rollin_policy),
            "version": int(rollin_policy_version),
            "seed": int(rollin_seed),
            "epoch": int(rollin_epoch),
        },
        "target_ir_schema_version": int(target_ir_schema_version),
        "max_length": int(max_length),
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
        "input_ids": input_ids,
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

    input_ids = payload["input_ids"]
    target_ir = payload[TEACHER_FORCING_TARGET_IR_KEY]
    if type(target_ir) is not TeacherForcingTargetIR:
        raise TypeError("teacher_forcing_target_ir must be a TeacherForcingTargetIR")

    validate_target_ir(target_ir, input_ids=input_ids, role_vocab=role_vocab)
    return {
        "input_ids": input_ids,
        TEACHER_FORCING_TARGET_IR_KEY: target_ir,
    }


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "build_fixed_eval_probe_cache_key",
    "build_fixed_eval_probe_payload",
    "load_fixed_eval_probe_payload",
]

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.training.encoding.teacher_forcing_cache import (
    build_fixed_eval_probe_cache_key,
    build_fixed_eval_probe_payload,
    load_fixed_eval_probe_payload,
)
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_KEY
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.vocab import RoleVocab
from src.training_runtime import (
    collect_training_runtime_preflight,
    resolve_training_runtime_plan,
    validate_training_runtime_preflight,
)


def make_training_config(
    *,
    objective_id: str = "teacher_forcing",
    rollin_policy: str = "random_permutation",
) -> SimpleNamespace:
    return SimpleNamespace(
        custom=SimpleNamespace(trainer_variant="stage2_two_channel"),
        objective=SimpleNamespace(
            id=objective_id,
            target_ir=SimpleNamespace(
                rollin_policy=SimpleNamespace(
                    name=rollin_policy,
                    base_seed=17,
                )
            ),
        ),
        training=SimpleNamespace(
            encoded_cache=SimpleNamespace(enabled=False),
        ),
    )


def make_role_vocab() -> RoleVocab:
    return RoleVocab(
        schema_token_ids=frozenset({201}),
        text_token_ids=frozenset({101}),
        coord_token_ids=frozenset({301}),
        stop_token_id=401,
    )


def make_atom(
    *,
    selected_token_id: int = 101,
    target_position: int = 1,
    logit_position: int = 0,
) -> SupervisionAtom:
    return SupervisionAtom(
        batch_index=0,
        logit_position=logit_position,
        target_position=target_position,
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({101}),
        selected_token_id=selected_token_id,
        latent_valid_token_ids=frozenset({101}),
        coverage_target_weights=None,
        loss_tags=frozenset({"singleton"}),
        loss_weight=1.0,
        coord_role=None,
        provenance={},
    )


def make_target_ir(
    *,
    schema_version: int = 1,
    selected_token_id: int = 101,
) -> TeacherForcingTargetIR:
    return TeacherForcingTargetIR(
        schema_version=schema_version,
        atoms=(make_atom(selected_token_id=selected_token_id),),
        metadata={},
    )


def test_epoch_varying_training_rollin_rejects_encoded_cache_enabled() -> None:
    config = make_training_config(
        objective_id="teacher_forcing",
        rollin_policy="random_permutation",
    )
    config.training.encoded_cache.enabled = True

    with pytest.raises(ValueError, match="teacher_forcing encoded training cache"):
        validate_training_runtime_preflight(
            config,
            runtime_plan=resolve_training_runtime_plan(
                config.custom.trainer_variant,
            ),
        )


def test_collect_preflight_records_teacher_forcing_cache_bypass_reason() -> None:
    config = make_training_config(
        objective_id="teacher_forcing",
        rollin_policy="random_permutation",
    )
    config.training.encoded_cache.enabled = True

    result = collect_training_runtime_preflight(
        config,
        runtime_plan=resolve_training_runtime_plan(config.custom.trainer_variant),
    )

    assert result.encoded_cache.enabled is True
    assert result.encoded_cache.allowed is False
    assert (
        result.encoded_cache.bypass_reason
        == "teacher_forcing_epoch_varying_rollin"
    )


def test_fixed_eval_probe_cache_key_tracks_teacher_forcing_contract_fields() -> None:
    key = build_fixed_eval_probe_cache_key(
        tokenizer_fingerprint="tok-a",
        chat_template_fingerprint="chat-b",
        serialization_policy="marker_delimited",
        description_normalization_policy="strip_collapse_space",
        rollin_policy="random_permutation",
        rollin_policy_version=1,
        rollin_seed=123,
        rollin_epoch=4,
        target_ir_schema_version=1,
        max_length=1024,
    )

    assert key["tokenizer_fingerprint"] == "tok-a"
    assert key["chat_template_fingerprint"] == "chat-b"
    assert key["serialization_policy"] == "marker_delimited"
    assert key["description_normalization_policy"] == "strip_collapse_space"
    assert key["rollin_policy"] == {
        "name": "random_permutation",
        "version": 1,
        "seed": 123,
        "epoch": 4,
    }
    assert key["target_ir_schema_version"] == 1
    assert key["max_length"] == 1024
    assert key["fingerprint_sha256"]


def test_fixed_eval_probe_payload_contains_input_ids_and_target_ir() -> None:
    target_ir = make_target_ir()
    payload = build_fixed_eval_probe_payload(
        input_ids=torch.tensor([[9, 101]]),
        teacher_forcing_target_ir=target_ir,
    )

    assert set(payload) == {"input_ids", TEACHER_FORCING_TARGET_IR_KEY}
    assert payload[TEACHER_FORCING_TARGET_IR_KEY] is target_ir


def test_fixed_eval_probe_cache_load_validates_payload_alignment() -> None:
    valid_payload = build_fixed_eval_probe_payload(
        input_ids=torch.tensor([[9, 101]]),
        teacher_forcing_target_ir=make_target_ir(),
    )

    loaded = load_fixed_eval_probe_payload(
        valid_payload,
        role_vocab=make_role_vocab(),
    )
    assert (
        loaded[TEACHER_FORCING_TARGET_IR_KEY]
        is valid_payload[TEACHER_FORCING_TARGET_IR_KEY]
    )

    mismatched_payload = build_fixed_eval_probe_payload(
        input_ids=torch.tensor([[9, 102]]),
        teacher_forcing_target_ir=make_target_ir(),
    )
    with pytest.raises(ValueError, match="selected_token_id must match input_ids"):
        load_fixed_eval_probe_payload(
            mismatched_payload,
            role_vocab=make_role_vocab(),
        )

    unsupported_schema_payload = build_fixed_eval_probe_payload(
        input_ids=torch.tensor([[9, 101]]),
        teacher_forcing_target_ir=make_target_ir(schema_version=999),
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_fixed_eval_probe_payload(
            unsupported_schema_payload,
            role_vocab=make_role_vocab(),
        )

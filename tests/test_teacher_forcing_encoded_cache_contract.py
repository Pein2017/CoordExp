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
    cache_namespace: str = "encoded_sample_cache",
    ineligible_policy: str = "error",
) -> SimpleNamespace:
    cache_cfg = SimpleNamespace(
        enabled=False,
        ineligible_policy=ineligible_policy,
    )
    return SimpleNamespace(
        custom=SimpleNamespace(trainer_variant="stage2_rollout_correction"),
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
            **{cache_namespace: cache_cfg},
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
    config.training.encoded_sample_cache.enabled = True

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
        ineligible_policy="bypass",
    )
    config.training.encoded_sample_cache.enabled = True

    result = collect_training_runtime_preflight(
        config,
        runtime_plan=resolve_training_runtime_plan(config.custom.trainer_variant),
    )

    assert result.encoded_cache.enabled is True
    assert result.encoded_cache.allowed is False
    assert result.encoded_cache.ineligible_policy == "bypass"
    assert (
        result.encoded_cache.bypass_reason
        == "teacher_forcing_epoch_varying_rollin"
    )
    assert result.encoded_cache.namespace == "encoded_sample_cache"


def test_validate_preflight_allows_teacher_forcing_cache_bypass_policy() -> None:
    config = make_training_config(ineligible_policy="bypass")
    config.training.encoded_sample_cache.enabled = True

    result = validate_training_runtime_preflight(
        config,
        runtime_plan=resolve_training_runtime_plan(config.custom.trainer_variant),
    )

    assert result.encoded_cache.allowed is False
    assert result.encoded_cache.ineligible_policy == "bypass"
    assert (
        result.encoded_cache.bypass_reason
        == "teacher_forcing_epoch_varying_rollin"
    )


def test_encoded_cache_alias_remains_compatibility_only() -> None:
    config = make_training_config(cache_namespace="encoded_cache")
    config.training.encoded_cache.enabled = True

    with pytest.raises(ValueError, match="teacher_forcing encoded training cache"):
        validate_training_runtime_preflight(
            config,
            runtime_plan=resolve_training_runtime_plan(
                config.custom.trainer_variant,
            ),
        )


def _fixed_eval_probe_key(**updates: object) -> dict[str, object]:
    payload = {
        "tokenizer_fingerprint": "tok-a",
        "chat_template_fingerprint": "chat-b",
        "serialization_policy": "marker_delimited",
        "description_normalization_policy": "strip_collapse_space",
        "rollin_policy": "random_permutation",
        "rollin_policy_version": 1,
        "rollin_base_seed": 123,
        "rollin_epoch": 4,
        "target_ir_schema_version": 1,
        "max_length": 1024,
    }
    payload.update(updates)
    return build_fixed_eval_probe_cache_key(**payload)


def test_fixed_eval_probe_cache_key_tracks_teacher_forcing_contract_fields() -> None:
    key = _fixed_eval_probe_key()

    assert key["tokenizer_fingerprint"] == "tok-a"
    assert key["chat_template_fingerprint"] == "chat-b"
    assert key["serialization_policy"] == "marker_delimited"
    assert key["description_normalization_policy"] == "strip_collapse_space"
    assert key["rollin_policy"] == {
        "name": "random_permutation",
        "version": 1,
        "base_seed": 123,
        "epoch": 4,
    }
    assert key["target_ir_schema_version"] == 1
    assert key["max_length"] == 1024
    assert key["fingerprint_sha256"]


def test_fixed_eval_probe_cache_key_digest_is_deterministic() -> None:
    left = _fixed_eval_probe_key()
    right = _fixed_eval_probe_key()

    assert left["fingerprint_sha256"] == right["fingerprint_sha256"]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tokenizer_fingerprint", "tok-b"),
        ("chat_template_fingerprint", "chat-c"),
        ("serialization_policy", "json_v2"),
        ("description_normalization_policy", "lowercase"),
        ("rollin_policy", "sorted"),
        ("rollin_policy_version", 2),
        ("rollin_base_seed", 124),
        ("rollin_epoch", 5),
        ("target_ir_schema_version", 2),
        ("max_length", 2048),
    ],
)
def test_fixed_eval_probe_cache_key_digest_tracks_each_surface(
    field_name: str,
    value: object,
) -> None:
    baseline = _fixed_eval_probe_key()
    changed = _fixed_eval_probe_key(**{field_name: value})

    assert changed["fingerprint_sha256"] != baseline["fingerprint_sha256"]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("tokenizer_fingerprint", "", ValueError),
        ("chat_template_fingerprint", None, TypeError),
        ("serialization_policy", "  ", ValueError),
        ("description_normalization_policy", None, TypeError),
        ("rollin_policy", "", ValueError),
        ("rollin_policy_version", True, TypeError),
        ("rollin_base_seed", False, TypeError),
        ("rollin_base_seed", None, TypeError),
        ("rollin_epoch", False, TypeError),
        ("rollin_epoch", None, TypeError),
        ("target_ir_schema_version", True, TypeError),
        ("target_ir_schema_version", 0, ValueError),
        ("max_length", True, TypeError),
        ("max_length", 0, ValueError),
    ],
)
def test_fixed_eval_probe_cache_key_rejects_malformed_fields(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        _fixed_eval_probe_key(**{field_name: value})


def test_fixed_eval_probe_payload_contains_input_ids_and_target_ir() -> None:
    target_ir = make_target_ir()
    input_ids = torch.tensor([[9, 101]])
    payload = build_fixed_eval_probe_payload(
        input_ids=input_ids,
        teacher_forcing_target_ir=target_ir,
    )
    input_ids[0, 1] = 999

    assert set(payload) == {"input_ids", TEACHER_FORCING_TARGET_IR_KEY}
    assert payload[TEACHER_FORCING_TARGET_IR_KEY] is target_ir
    assert payload["input_ids"][0, 1].item() == 101


def test_fixed_eval_probe_payload_freezes_simple_list_input_ids() -> None:
    input_ids = [[9, 101]]

    payload = build_fixed_eval_probe_payload(
        input_ids=input_ids,
        teacher_forcing_target_ir=make_target_ir(),
    )
    input_ids[0][1] = 999

    assert payload["input_ids"] == ((9, 101),)


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
    valid_payload["input_ids"][0, 1] = 999
    assert loaded["input_ids"][0, 1].item() == 101

    list_payload = build_fixed_eval_probe_payload(
        input_ids=[[9, 101]],
        teacher_forcing_target_ir=make_target_ir(),
    )
    loaded_list = load_fixed_eval_probe_payload(
        list_payload,
        role_vocab=make_role_vocab(),
    )
    assert loaded_list["input_ids"] == ((9, 101),)

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

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.sft as sft_module
from src.sft import (
    PackingRuntimeConfig,
    _build_pipeline_manifest,
    _apply_rollout_decode_batch_size_override,
    _is_rollout_matching_variant,
    _is_stage1_set_continuation_variant,
    _validate_static_packing_accumulation_windows,
    _validate_stage1_static_packing_policy,
    resolve_trainer_cls,
)
from src.training_runtime import (
    resolve_training_runtime_plan,
    resolve_training_runtime_profile,
)


@pytest.mark.parametrize(
    ("variant", "replacement"),
    [
        ("stage2_ab_training", "stage2_two_channel"),
        ("rollout_matching_sft", "stage2_rollout_aligned"),
    ],
)
def test_resolve_trainer_cls_removed_variants_fail_through_runtime_plan(
    variant: str,
    replacement: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_trainer_cls(SimpleNamespace(trainer_variant=variant))

    message = str(exc_info.value)
    assert f"custom.trainer_variant={variant} has been removed" in message
    assert f"use {replacement}" in message


@pytest.mark.parametrize(
    "variant",
    [
        None,
        "",
        "stage1_set_continuation",
        "stage2_two_channel",
        "stage2_rollout_aligned",
    ],
)
def test_sft_variant_helpers_agree_with_runtime_plan(variant: str | None) -> None:
    plan = resolve_training_runtime_plan(variant)

    assert _is_rollout_matching_variant(variant) is (
        plan.post_rollout_packing_owner is not None
    )
    assert _is_stage1_set_continuation_variant(variant) is (
        plan.collator_family == "stage1_set_continuation"
    )


def test_validate_stage1_static_packing_policy_rejects_stage1_dynamic_mode() -> None:
    with pytest.raises(
        ValueError,
        match="deprecated and unsupported for Stage-1",
    ):
        _validate_stage1_static_packing_policy(
            packing_cfg=PackingRuntimeConfig(enabled=True, mode="dynamic"),
            trainer_variant=None,
        )


@pytest.mark.parametrize(
    ("eval_packing", "message"),
    [
        (False, "training\\.packing=false"),
        (True, "training\\.eval_packing=false"),
    ],
)
def test_validate_stage1_static_packing_policy_preserves_set_continuation_errors(
    eval_packing: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_stage1_static_packing_policy(
            packing_cfg=PackingRuntimeConfig(
                enabled=True,
                mode="static",
                eval_packing=eval_packing,
            ),
            trainer_variant="stage1_set_continuation",
        )


@pytest.mark.parametrize("variant", ["stage2_two_channel", "stage2_rollout_aligned"])
def test_validate_stage1_static_packing_policy_allows_stage2_trainer_owned_packing(
    variant: str,
) -> None:
    plan = resolve_training_runtime_plan(variant)
    assert plan.post_rollout_packing_owner == "trainer"

    _validate_stage1_static_packing_policy(
        packing_cfg=PackingRuntimeConfig(enabled=True, mode="dynamic"),
        trainer_variant=variant,
    )


def test_static_packing_accumulation_warning_is_skipped_for_trainer_owned_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        sft_module.logger,
        "warning",
        lambda message, *args: warnings.append(str(message)),
    )

    _validate_static_packing_accumulation_windows(
        packing_cfg=PackingRuntimeConfig(enabled=True, mode="static"),
        trainer_variant="stage2_two_channel",
        per_rank_batches_est=1,
        gradient_accumulation_steps=2,
        world_size=1,
        dataloader_drop_last=False,
    )

    assert warnings == []

    _validate_static_packing_accumulation_windows(
        packing_cfg=PackingRuntimeConfig(enabled=True, mode="static"),
        trainer_variant=None,
        per_rank_batches_est=1,
        gradient_accumulation_steps=2,
        world_size=1,
        dataloader_drop_last=False,
    )

    assert warnings


@pytest.mark.parametrize(
    ("trainer_variant", "required_namespace"),
    [
        ("stage2_two_channel", "stage2_ab.pipeline"),
        ("stage2_rollout_aligned", "rollout_matching.pipeline"),
    ],
)
def test_pipeline_manifest_missing_pipeline_error_uses_runtime_namespace(
    trainer_variant: str,
    required_namespace: str,
) -> None:
    with pytest.raises(ValueError, match=required_namespace):
        _build_pipeline_manifest(
            {},
            default_objective=["token_ce"],
            default_diagnostics=["coord_diag"],
            trainer_variant=trainer_variant,
            config_path="configs/example.yaml",
            run_name="runtime-profile-manifest-test",
            seed=17,
        )


@pytest.mark.parametrize("variant", [None, "", "stage1_set_continuation"])
def test_rollout_decode_batch_size_override_skips_non_rollout_profiles(
    variant: str | None,
) -> None:
    profile = resolve_training_runtime_profile(variant)
    assert profile.rollout_runtime_owned is False

    train_args = SimpleNamespace(
        trainer_variant=variant,
        training_args=SimpleNamespace(per_device_eval_batch_size=3),
    )

    assert (
        _apply_rollout_decode_batch_size_override(
            train_args=train_args,
            training_config=SimpleNamespace(),
        )
        == 1
    )
    assert train_args.training_args.per_device_eval_batch_size == 3


@pytest.mark.parametrize("variant", ["stage2_two_channel", "stage2_rollout_aligned"])
def test_rollout_decode_batch_size_override_uses_rollout_runtime_profile(
    variant: str,
) -> None:
    profile = resolve_training_runtime_profile(variant)
    assert profile.rollout_runtime_owned is True

    train_args = SimpleNamespace(
        trainer_variant=variant,
        training_args=SimpleNamespace(per_device_eval_batch_size=2),
    )

    resolved = _apply_rollout_decode_batch_size_override(
        train_args=train_args,
        training_config=SimpleNamespace(
            rollout_matching={"eval_decode_batch_size": "5"},
        ),
    )

    assert resolved == 5
    assert train_args.per_device_eval_batch_size == 5
    assert train_args.training_args.per_device_eval_batch_size == 5

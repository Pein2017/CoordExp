from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import src.sft as sft_module
from src.bootstrap.trainer_setup import compose_trainer_class
from src.sft import (
    EncodedSampleCacheRuntimeConfig,
    PackingRuntimeConfig,
    _apply_sft_encoded_sample_cache_preflight,
    _build_pipeline_manifest,
    _build_encoded_sample_cache_request,
    _apply_rollout_decode_batch_size_override,
    _is_rollout_matching_variant,
    _validate_sft_runtime_preflight,
    _validate_static_packing_accumulation_windows,
    _validate_stage1_static_packing_policy,
    resolve_trainer_cls,
)
from src.training_runtime import (
    resolve_training_runtime_plan,
    resolve_training_runtime_profile,
)
from src.trainers.metrics.mixins import TeacherForcingObjectiveMixin
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.vocab import RoleVocab


@pytest.mark.parametrize(
    ("variant", "replacement"),
    [
        ("stage2_ab_training", "stage2_two_channel"),
        ("rollout_matching_sft", "stage2_two_channel"),
        ("stage2_rollout_aligned", "stage2_two_channel"),
        ("stage2_rollout_runtime", "stage2_two_channel"),
        ("stage1_set_continuation", "prefix_rollin_et_rmp_ce"),
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
        "stage2_two_channel",
    ],
)
def test_sft_variant_helpers_agree_with_runtime_plan(variant: str | None) -> None:
    plan = resolve_training_runtime_plan(variant)

    assert _is_rollout_matching_variant(variant) is (
        plan.post_rollout_packing_owner is not None
    )


def test_sft_rejects_removed_stage1_set_continuation_variant() -> None:
    with pytest.raises(ValueError, match=r"stage1_set_continuation.*removed"):
        resolve_training_runtime_plan("stage1_set_continuation")


def test_validate_stage1_static_packing_policy_rejects_stage1_dynamic_mode() -> None:
    with pytest.raises(
        ValueError,
        match="deprecated and unsupported for Stage-1",
    ):
        _validate_stage1_static_packing_policy(
            packing_cfg=PackingRuntimeConfig(enabled=True, mode="dynamic"),
            trainer_variant=None,
        )


def test_validate_stage1_static_packing_policy_allows_stage2_trainer_owned_packing() -> None:
    variant = "stage2_two_channel"
    plan = resolve_training_runtime_plan(variant)
    assert plan.post_rollout_packing_owner == "trainer"

    _validate_stage1_static_packing_policy(
        packing_cfg=PackingRuntimeConfig(enabled=True, mode="dynamic"),
        trainer_variant=variant,
    )


def test_sft_runtime_preflight_rejects_teacher_forcing_encoded_sample_cache() -> None:
    config = SimpleNamespace(
        objective=SimpleNamespace(
            id="teacher_forcing",
            target_ir=SimpleNamespace(
                rollin_policy=SimpleNamespace(
                    name="random_permutation",
                    base_seed=17,
                )
            ),
        ),
        training={
            "encoded_sample_cache": {
                "enabled": True,
                "root_dir": "/tmp/coordexp-cache",
            }
        },
    )

    with pytest.raises(ValueError, match="teacher_forcing encoded training cache"):
        _validate_sft_runtime_preflight(
            training_config=config,
            runtime_plan=resolve_training_runtime_plan("stage2_two_channel"),
        )


def test_sft_runtime_preflight_bypasses_teacher_forcing_encoded_sample_cache() -> None:
    config = SimpleNamespace(
        objective=SimpleNamespace(
            id="teacher_forcing",
            target_ir=SimpleNamespace(
                rollin_policy=SimpleNamespace(
                    name="random_permutation",
                    base_seed=17,
                )
            ),
        ),
        training={
            "encoded_sample_cache": {
                "enabled": True,
                "root_dir": "/tmp/coordexp-cache",
                "ineligible_policy": "bypass",
            }
        },
    )

    preflight = _validate_sft_runtime_preflight(
        training_config=config,
        runtime_plan=resolve_training_runtime_plan("stage2_two_channel"),
    )
    decision = _apply_sft_encoded_sample_cache_preflight(
        encoded_sample_cache_cfg=EncodedSampleCacheRuntimeConfig(
            enabled=True,
            root_dir="/tmp/coordexp-cache",
            ineligible_policy="bypass",
        ),
        preflight_result=preflight,
    )

    assert decision.encoded_sample_cache_cfg.enabled is False
    assert (
        decision.bypass_reason
        == "teacher_forcing_epoch_varying_rollin"
    )
    assert decision.bypass_info_for_split(
        dataset_split="train",
        dataset_jsonl="/tmp/train.jsonl",
    ) == {
        "enabled": True,
        "status": "bypassed",
        "reason": "teacher_forcing_epoch_varying_rollin",
        "policy": "bypass",
        "dataset_split": "train",
        "dataset_jsonl": "/tmp/train.jsonl",
    }
    assert (
        _build_encoded_sample_cache_request(
            runtime_cfg=decision.encoded_sample_cache_cfg,
            training_config=SimpleNamespace(global_max_length=1024, template={}),
            custom_config=SimpleNamespace(
                user_prompt="prompt",
                emit_norm="none",
                json_format="standard",
                bbox_format="xyxy",
                detection_sequence_format="coordjson",
                object_ordering="random_permutation",
                object_field_order="desc_first",
                use_summary=False,
                offline_max_pixels=None,
                coord_tokens=None,
            ),
            template=SimpleNamespace(max_length=1024),
            train_args=SimpleNamespace(max_model_len=1024),
            dataset_seed=17,
            dataset_jsonl="/tmp/train.jsonl",
            dataset_split="train",
            dataset_mode="dense",
        )
        is None
    )


def test_compose_trainer_class_adds_teacher_forcing_objective_mixin() -> None:
    trainer_cls = compose_trainer_class(
        trainer_cls=object,
        trainer_variant="",
        instability_monitor_cfg=None,
        token_type_cfg=None,
        bbox_geo_cfg=None,
        bbox_size_aux_cfg=None,
        coord_soft_ce_w1_cfg=None,
        sft_structural_close_cfg=None,
        recursive_detection_ce_cfg=None,
        teacher_forcing_objective_cfg=SimpleNamespace(enabled=True),
    )

    assert issubclass(trainer_cls, TeacherForcingObjectiveMixin)


def test_teacher_forcing_objective_mixin_computes_loss_through_runner() -> None:
    class _Model:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            return SimpleNamespace(logits=self.logits)

    class _BaseTrainer:
        def compute_loss(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("base trainer CE path should not own teacher_forcing")

    class _Trainer(TeacherForcingObjectiveMixin, _BaseTrainer):
        pass

    atom = SupervisionAtom(
        batch_index=0,
        logit_position=0,
        target_position=1,
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({1, 2}),
        selected_token_id=1,
        latent_valid_token_ids=frozenset({1, 2}),
        coverage_target_weights=None,
        loss_tags=frozenset({"pure_valid_set_marginal"}),
        loss_weight=1.0,
        coord_role=None,
        provenance={},
    )
    target_ir = TeacherForcingTargetIR(
        schema_version=1,
        atoms=(atom,),
        metadata={"serialization_policy": "marker_delimited"},
    )
    logits = torch.tensor(
        [[[0.0, 2.0, 1.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    inputs = {
        "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, 1]], dtype=torch.long),
        "teacher_forcing_target_ir": (target_ir,),
    }
    trainer = _Trainer()
    trainer.teacher_forcing_objective_cfg = SimpleNamespace(
        profile="pure_valid_set_marginal",
        modules=SimpleNamespace(
            within_valid_coverage=SimpleNamespace(coverage_strength=0.0),
        ),
    )
    trainer.teacher_forcing_role_vocab = RoleVocab(
        text_token_ids=frozenset({1, 2}),
        schema_token_ids=frozenset(),
        coord_token_ids=frozenset(),
        stop_token_id=9,
    )

    loss, outputs = trainer.compute_loss(
        _Model(logits),
        inputs,
        return_outputs=True,
    )

    expected = -torch.log(torch.softmax(logits[0, 0], dim=-1)[[1, 2]].sum())
    assert loss.item() == pytest.approx(expected.item())
    assert outputs.logits is logits


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


@pytest.mark.parametrize("variant", [None, ""])
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


def test_rollout_decode_batch_size_override_uses_rollout_runtime_profile() -> None:
    variant = "stage2_two_channel"
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

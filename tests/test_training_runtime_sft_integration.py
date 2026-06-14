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
    validate_training_runtime_preflight,
    resolve_training_runtime_plan,
    resolve_training_runtime_profile,
)
from src.training_runtime.stage2_projection import (
    apply_stage2_runtime_projection,
    resolve_stage2_runtime_projection,
)
from src.trainers.batch_extras import BatchExtras, stash_batch_extras
from src.trainers.metrics.mixins import (
    PrefixDenoisingObjectiveMixin,
    TeacherForcingObjectiveMixin,
)
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.vocab import RoleVocab


@pytest.mark.parametrize(
    ("variant", "replacement"),
    [
        ("stage2_ab_training", "stage2_rollout_correction"),
        ("stage2_two_channel", "stage2_rollout_correction"),
        ("rollout_matching_sft", "stage2_rollout_correction"),
        ("stage2_rollout_aligned", "stage2_rollout_correction"),
        ("stage2_rollout_runtime", "stage2_rollout_correction"),
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
        "stage2_rollout_correction",
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


def test_sft_rejects_unknown_non_empty_trainer_variant() -> None:
    with pytest.raises(
        ValueError,
        match=r"custom\.trainer_variant=experimental_unknown.*not supported",
    ):
        resolve_training_runtime_plan("experimental_unknown")


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
    variant = "stage2_rollout_correction"
    plan = resolve_training_runtime_plan(variant)
    assert plan.post_rollout_packing_owner == "trainer"

    _validate_stage1_static_packing_policy(
        packing_cfg=PackingRuntimeConfig(enabled=True, mode="dynamic"),
        trainer_variant=variant,
    )


def test_teacher_forcing_stage2_packing_fails_runtime_preflight() -> None:
    config = SimpleNamespace(
        objective=SimpleNamespace(id="teacher_forcing"),
        custom=SimpleNamespace(trainer_variant="stage2_rollout_correction"),
        training={"packing": True},
    )

    with pytest.raises(
        ValueError,
        match=r"teacher_forcing.*stage2_rollout_correction.*packing",
    ):
        validate_training_runtime_preflight(
            config,
            runtime_plan=resolve_training_runtime_plan("stage2_rollout_correction"),
        )


def test_non_teacher_forcing_stage2_packing_stays_trainer_owned() -> None:
    config = SimpleNamespace(
        objective=SimpleNamespace(id="stage2_rollout_correction"),
        custom=SimpleNamespace(trainer_variant="stage2_rollout_correction"),
        training={"packing": True},
    )

    preflight = validate_training_runtime_preflight(
        config,
        runtime_plan=resolve_training_runtime_plan("stage2_rollout_correction"),
    )

    assert preflight.runtime_plan.post_rollout_packing_owner == "trainer"


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
            runtime_plan=resolve_training_runtime_plan("stage2_rollout_correction"),
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
        runtime_plan=resolve_training_runtime_plan("stage2_rollout_correction"),
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


def test_compose_trainer_class_uses_prefix_denoising_objective_when_enabled() -> None:
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
        prefix_denoising_cfg=SimpleNamespace(
            enabled=True,
            current_object_kl=SimpleNamespace(weight=0.0),
        ),
        prefix_denoising_runtime={"packing_enabled": False, "kl_weight": 0.0},
    )

    assert issubclass(trainer_cls, PrefixDenoisingObjectiveMixin)
    assert not issubclass(trainer_cls, TeacherForcingObjectiveMixin)
    assert trainer_cls.prefix_denoising_packing_enabled is False


def test_compose_trainer_class_rejects_prefix_denoising_packed_until_task7() -> None:
    with pytest.raises(ValueError, match="Task 7 boundary rewriting"):
        compose_trainer_class(
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
            prefix_denoising_cfg=SimpleNamespace(
                enabled=True,
                current_object_kl=SimpleNamespace(weight=0.0),
            ),
            prefix_denoising_runtime={"packing_enabled": True, "kl_weight": 0.0},
        )


def test_compose_trainer_class_rejects_positive_prefix_kl_until_task6() -> None:
    with pytest.raises(ValueError, match="Task 6 KL support"):
        compose_trainer_class(
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
            prefix_denoising_cfg=SimpleNamespace(
                enabled=True,
                current_object_kl=SimpleNamespace(weight=0.05),
            ),
            prefix_denoising_runtime={"packing_enabled": False, "kl_weight": 0.05},
        )


def test_compose_trainer_class_requires_prefix_runtime_when_enabled() -> None:
    with pytest.raises(ValueError, match="prefix_denoising_runtime"):
        compose_trainer_class(
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
            prefix_denoising_cfg=SimpleNamespace(enabled=True),
        )


def test_prefix_denoising_objective_reads_stashed_extras_and_strips_sidecars() -> None:
    class _Model:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits
            self.calls: list[dict[str, object]] = []
            self.training = True
            self.config = SimpleNamespace(model_type="unit")

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            return SimpleNamespace(logits=self.logits)

    class _BaseTrainer:
        custom_metrics = None
        model = None

        def compute_loss(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("base trainer CE path should not own prefix denoising")

    class _Trainer(PrefixDenoisingObjectiveMixin, _BaseTrainer):
        prefix_denoising_packing_enabled = False

    logits = torch.full((1, 6, 4), -5.0, dtype=torch.float32)
    labels = torch.full((1, 6), -100, dtype=torch.long)
    labels[0, 2] = 1
    labels[0, 5] = 3
    logits[0, 1, 1] = 8.0
    logits[0, 4, 3] = 8.0
    inputs = {
        "input_ids": torch.tensor([[0, 9, 1, 2, 8, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "labels": labels,
        "prefix_denoising_hybrid": "must-not-be-read-from-inputs",
        "sample_id": "unit-0",
    }
    extras = BatchExtras(
        prefix_denoising_hybrid=(object(),),
        prefix_denoising_segment_meta=(
            (
                {
                    "batch_index": 0,
                    "token_start": 0,
                    "token_end": 3,
                    "branch_id": "clean_full",
                    "segment_id": "unit:clean",
                },
                {
                    "batch_index": 0,
                    "token_start": 3,
                    "token_end": 6,
                    "branch_id": "noisy_full",
                    "segment_id": "unit:noisy",
                },
            ),
        ),
    )
    trainer = _Trainer()
    stash_batch_extras(trainer, extras)

    model = _Model(logits)
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

    expected = 0.5 * torch.nn.functional.cross_entropy(
        logits[0, 1].unsqueeze(0),
        torch.tensor([1]),
    ) + 0.5 * torch.nn.functional.cross_entropy(
        logits[0, 4].unsqueeze(0),
        torch.tensor([3]),
    )
    assert loss.item() == pytest.approx(expected.item())
    assert outputs.logits is logits
    assert len(model.calls) == 1
    assert "labels" not in model.calls[0]
    assert "prefix_denoising_hybrid" not in model.calls[0]
    assert "prefix_denoising_segment_meta" not in model.calls[0]
    assert "sample_id" not in model.calls[0]


def test_prefix_denoising_real_mro_refreshes_batch_extras_between_loss_calls() -> None:
    class _Model:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.training = True
            self.config = SimpleNamespace(model_type="unit")

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            input_ids = kwargs["input_ids"]
            vocab = 5
            logits = torch.full(
                (int(input_ids.shape[0]), int(input_ids.shape[1]), vocab),
                -6.0,
                dtype=torch.float32,
            )
            if int(input_ids[0, 1]) == 11:
                logits[0, 1, 1] = 9.0
                logits[0, 4, 3] = 9.0
            else:
                logits[0, 2, 2] = 9.0
                logits[0, 4, 4] = 9.0
            return SimpleNamespace(logits=logits)

    class _BaseTrainer:
        custom_metrics = None
        model = None
        args = SimpleNamespace(gradient_accumulation_steps=1)

    trainer_cls = compose_trainer_class(
        trainer_cls=_BaseTrainer,
        trainer_variant="",
        instability_monitor_cfg=None,
        token_type_cfg=None,
        bbox_geo_cfg=None,
        bbox_size_aux_cfg=None,
        coord_soft_ce_w1_cfg=None,
        sft_structural_close_cfg=None,
        recursive_detection_ce_cfg=None,
        teacher_forcing_objective_cfg=SimpleNamespace(enabled=True),
        prefix_denoising_cfg=SimpleNamespace(
            enabled=True,
            current_object_kl=SimpleNamespace(weight=0.0),
        ),
        prefix_denoising_runtime={"packing_enabled": False, "kl_weight": 0.0},
    )
    trainer = trainer_cls()
    model = _Model()

    first_loss = trainer.compute_loss(
        model,
        {
            "input_ids": torch.tensor([[0, 11, 1, 0, 13, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 1, -100, -100, 3]], dtype=torch.long),
            "prefix_denoising_hybrid": (object(),),
            "prefix_denoising_segment_meta": (
                (
                    {
                        "batch_index": 0,
                        "token_start": 0,
                        "token_end": 3,
                        "branch_id": "clean_full",
                        "segment_id": "first:clean",
                    },
                    {
                        "batch_index": 0,
                        "token_start": 3,
                        "token_end": 6,
                        "branch_id": "noisy_full",
                        "segment_id": "first:noisy",
                    },
                ),
            ),
        },
    )
    second_loss = trainer.compute_loss(
        model,
        {
            "input_ids": torch.tensor([[0, 0, 12, 0, 0, 14]], dtype=torch.long),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "labels": torch.tensor([[-100, -100, -100, 2, -100, 4]], dtype=torch.long),
            "prefix_denoising_hybrid": (object(),),
            "prefix_denoising_segment_meta": (
                (
                    {
                        "batch_index": 0,
                        "token_start": 0,
                        "token_end": 4,
                        "branch_id": "clean_full",
                        "segment_id": "second:clean",
                    },
                    {
                        "batch_index": 0,
                        "token_start": 4,
                        "token_end": 6,
                        "branch_id": "noisy_full",
                        "segment_id": "second:noisy",
                    },
                ),
            ),
        },
    )

    assert len(model.calls) == 2
    assert first_loss.item() < 0.001
    assert second_loss.item() < 0.001


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
        "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "output_router_logits": True,
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

    model = _Model(logits)
    loss, outputs = trainer.compute_loss(
        model,
        inputs,
        return_outputs=True,
    )

    expected = -torch.log(torch.softmax(logits[0, 0], dim=-1)[[1, 2]].sum())
    assert loss.item() == pytest.approx(expected.item())
    assert outputs.logits is logits
    assert len(model.calls) == 1
    assert "teacher_forcing_target_ir" not in model.calls[0]
    assert "labels" not in model.calls[0]
    assert model.calls[0]["position_ids"] is inputs["position_ids"]
    assert model.calls[0]["output_router_logits"] is True


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
        trainer_variant="stage2_rollout_correction",
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
        ("stage2_rollout_correction", "stage2_rollout_correction.pipeline"),
    ],
)
def test_pipeline_manifest_missing_pipeline_error_uses_runtime_namespace(
    trainer_variant: str,
    required_namespace: str,
) -> None:
    with pytest.raises(ValueError, match=required_namespace):
        _build_pipeline_manifest(
            {},
            default_objective=["residual_set_correction"],
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
    variant = "stage2_rollout_correction"
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


def _runtime_projection_custom_config(
    *,
    extra_prompt_variant: str | None = "default",
) -> SimpleNamespace:
    extra = {}
    if extra_prompt_variant is not None:
        extra["prompt_variant"] = extra_prompt_variant
    return SimpleNamespace(
        extra=extra,
        object_ordering="sorted",
        object_field_order="desc_first",
        bbox_format="xyxy",
        detection_sequence_format="coordjson",
    )


def _runtime_projection_packing_config() -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        packing_length=1024,
        buffer_size=8,
        min_fill_ratio=0.75,
        drop_last=False,
    )


def test_stage2_runtime_projection_records_authored_policy_sources() -> None:
    projection = resolve_stage2_runtime_projection(
        training_config=SimpleNamespace(
            rollout_matching={
                "rollout_backend": "hf",
                "eval_rollout_backend": "hf",
                "prompt_variant": "coco_80",
                "eval_prompt_variant": "default",
                "decoding": {"temperature": 0.2},
            },
            stage2_rollout_correction={
                "pipeline": {
                    "objective": [
                        {
                            "name": "residual_set_correction",
                            "enabled": True,
                        }
                    ],
                    "diagnostics": [],
                },
                "correction": {"insertion_order": "sorted"},
            },
        ),
        custom_config=_runtime_projection_custom_config(),
        packing_cfg=_runtime_projection_packing_config(),
        trainer_variant="stage2_rollout_correction",
        config_path="configs/stage2.yaml",
        run_name="projection-test",
        seed=17,
    )

    assert projection.rollout_matching_cfg["prompt_variant"] == "coco_80"
    assert projection.rollout_matching_cfg["eval_prompt_variant"] == "default"
    assert projection.policy_sources["rollout_matching.prompt_variant"] == (
        "rollout_matching.prompt_variant"
    )
    assert projection.policy_sources["rollout_matching.eval_prompt_variant"] == (
        "rollout_matching.eval_prompt_variant"
    )
    assert projection.policy_sources["packing.enabled"] == "training.packing"
    assert projection.policy_sources["object_ordering"] == "custom.object_ordering"
    assert projection.stage2_policy_provenance["runtime_policy_sources"] == (
        projection.policy_sources
    )
    assert projection.stage2_policy_provenance["runtime_compatibility_fallbacks"] == []


def test_stage2_runtime_projection_marks_custom_extra_prompt_compat_fallback() -> None:
    projection = resolve_stage2_runtime_projection(
        training_config=SimpleNamespace(
            rollout_matching={
                "rollout_backend": "hf",
                "eval_rollout_backend": "hf",
                "decoding": {},
            },
            stage2_rollout_correction={
                "pipeline": {
                    "objective": [
                        {
                            "name": "residual_set_correction",
                            "enabled": True,
                        }
                    ],
                    "diagnostics": [],
                },
                "correction": {},
            },
        ),
        custom_config=_runtime_projection_custom_config(
            extra_prompt_variant="coco_80"
        ),
        packing_cfg=_runtime_projection_packing_config(),
        trainer_variant="stage2_rollout_correction",
        config_path="configs/stage2.yaml",
        run_name="projection-test",
        seed=17,
    )

    assert projection.rollout_matching_cfg["prompt_variant"] == "coco_80"
    assert projection.rollout_matching_cfg["eval_prompt_variant"] == "coco_80"
    assert projection.policy_sources["rollout_matching.prompt_variant"] == (
        "custom.extra.prompt_variant_compat_fallback"
    )
    assert projection.policy_sources["rollout_matching.eval_prompt_variant"] == (
        "custom.extra.prompt_variant_compat_fallback"
    )
    assert projection.stage2_policy_provenance["runtime_compatibility_fallbacks"] == [
        "custom.extra.prompt_variant"
    ]


def test_apply_stage2_runtime_projection_sets_trainer_boundary_attrs() -> None:
    calls: list[str] = []

    class _Trainer:
        def _validate_rollout_matching_cfg(self) -> None:
            calls.append("validated")

    trainer = _Trainer()
    projection = resolve_stage2_runtime_projection(
        training_config=SimpleNamespace(
            rollout_matching={
                "rollout_backend": "hf",
                "eval_rollout_backend": "hf",
                "decoding": {},
            },
            stage2_rollout_correction={
                "pipeline": {
                    "objective": [
                        {
                            "name": "residual_set_correction",
                            "enabled": True,
                        }
                    ],
                    "diagnostics": [],
                },
                "correction": {},
            },
        ),
        custom_config=_runtime_projection_custom_config(
            extra_prompt_variant=None
        ),
        packing_cfg=_runtime_projection_packing_config(),
        trainer_variant="stage2_rollout_correction",
        config_path="configs/stage2.yaml",
        run_name="projection-test",
        seed=17,
    )

    apply_stage2_runtime_projection(trainer, projection)

    assert calls == ["validated"]
    assert trainer.rollout_matching_cfg is projection.rollout_matching_cfg
    assert trainer.stage2_rollout_correction_cfg is (
        projection.stage2_rollout_correction_cfg
    )
    assert trainer.stage2_pipeline_manifest is projection.stage2_pipeline_manifest
    assert trainer.stage2_policy_provenance is projection.stage2_policy_provenance

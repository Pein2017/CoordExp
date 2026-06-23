from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import src.trainers.metrics.teacher_forcing as teacher_forcing_metrics
import src.sft as sft_module
from src.bootstrap.trainer_setup import compose_trainer_class
from src.sft import (
    EncodedSampleCacheRuntimeConfig,
    PackingRuntimeConfig,
    _apply_sft_encoded_sample_cache_preflight,
    _build_pipeline_manifest,
    _build_cfg_only_summary,
    _build_effective_runtime_payload,
    _build_encoded_sample_cache_request,
    _build_normalized_training_hierarchy_identity,
    _apply_rollout_decode_batch_size_override,
    _is_rollout_matching_variant,
    _uses_stage1_detection_dataset_builder,
    _validate_sft_runtime_preflight,
    _validate_static_packing_accumulation_windows,
    _validate_stage1_static_packing_policy,
    resolve_trainer_cls,
)
from src.config.schema import DetectionTrainingConfig
from src.config.loader import ConfigLoader
from src.detection.runtime import (
    build_detection_dataset,
    build_detection_runtime_custom_shim,
    detection_mode,
)
from test_detection_training_config_contract import (
    _detection_payload,
    _stage2_rollout_correction_payload,
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
from src.trainers.metrics.mixins import TeacherForcingObjectiveMixin
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.vocab import RoleVocab
from src.metrics.events import weighted_mean_event
from src.training.sidecars import SupervisionSidecars, TrainingSidecars

PUBLIC_RESEARCH_TF_EPOCH_VARYING_ROLLIN_BYPASS_REASON = (
    "research_teacher_forcing_epoch_varying_rollin"
)


@dataclass(init=False)
class _FakeTrainArguments:
    tuner_type: str | None = None
    training_args: SimpleNamespace = field(default_factory=SimpleNamespace)

    def __init__(self, **kwargs):
        self.training_args = SimpleNamespace()
        for key, value in kwargs.items():
            setattr(self, key, value)


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
        objective=SimpleNamespace(id="research_teacher_forcing"),
        custom=SimpleNamespace(trainer_variant="stage2_rollout_correction"),
        training={"packing": True},
    )

    with pytest.raises(
        ValueError,
        match=r"research_teacher_forcing.*stage2_rollout_correction.*packing",
    ):
        validate_training_runtime_preflight(
            config,
            runtime_plan=resolve_training_runtime_plan("stage2_rollout_correction"),
        )


def test_teacher_forcing_stage2_pipeline_packing_fails_default_runtime_preflight() -> None:
    config = SimpleNamespace(
        pipeline=SimpleNamespace(id="stage2_rollout_correction"),
        objective=SimpleNamespace(id="research_teacher_forcing"),
        training={"packing": True},
    )

    with pytest.raises(
        ValueError,
        match=r"research_teacher_forcing.*pipeline\.id=stage2_rollout_correction.*packing",
    ):
        validate_training_runtime_preflight(config)


@pytest.mark.parametrize(
    ("pipeline_id", "objective_id"),
    [
        ("stage1_standard_sft", "standard_ce"),
        ("stage1_research_teacher_forcing", "research_teacher_forcing"),
    ],
)
def test_stage1_pipeline_ids_use_default_runtime_plan_without_explicit_runtime_plan(
    pipeline_id: str,
    objective_id: str,
) -> None:
    result = validate_training_runtime_preflight(
        SimpleNamespace(
            pipeline=SimpleNamespace(id=pipeline_id),
            objective=SimpleNamespace(id=objective_id),
            training={"packing": True},
        )
    )

    assert result.runtime_plan.variant == ""
    assert result.runtime_plan.post_rollout_packing_owner is None


def test_detection_stage2_pipeline_id_builds_stage2_runtime_train_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.config.loader.TrainArguments", _FakeTrainArguments)
    cfg = DetectionTrainingConfig.from_mapping(_stage2_rollout_correction_payload())

    train_args = ConfigLoader.build_train_arguments(cfg)

    assert getattr(train_args, "trainer_variant", None) == "stage2_rollout_correction"
    assert resolve_training_runtime_plan(train_args.trainer_variant).variant == (
        "stage2_rollout_correction"
    )
    trainer_cls = resolve_trainer_cls(train_args)
    assert trainer_cls.__name__ == "Stage2RolloutCorrectionTrainerWithFinalCheckpoint"
    assert any(cls.__name__ == "Stage2RolloutCorrectionTrainer" for cls in trainer_cls.__mro__)


def test_loader_materializes_stage2_pipeline_id_as_detection_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.config.loader.TrainArguments", _FakeTrainArguments)
    payload = _stage2_rollout_correction_payload()
    assert "objective" not in payload

    prompts = ConfigLoader.resolve_prompts(payload)
    cfg = ConfigLoader._materialize_training_config(payload, prompts)

    assert isinstance(cfg, DetectionTrainingConfig)
    assert cfg.pipeline.id == "stage2_rollout_correction"
    assert cfg.objective is None

    train_args = ConfigLoader.build_train_arguments(cfg)
    assert getattr(train_args, "trainer_variant", None) == "stage2_rollout_correction"


def test_cfg_only_summary_uses_pipeline_identity_not_trainer_variant() -> None:
    cfg = DetectionTrainingConfig.from_mapping(_stage2_rollout_correction_payload())

    summary = _build_cfg_only_summary(
        config_path="configs/stage2/rollout_correction/smoke/unit.yaml",
        training_config=cfg,
        train_args=SimpleNamespace(
            run_name="stage2-unit",
            output_dir="output/stage2-unit",
            max_steps=1,
            eval_strategy="no",
            eval_steps=0,
            save_strategy="no",
            save_steps=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
        ),
        rank_context={
            "world_size": 1,
            "local_world_size": 1,
            "local_rank": 0,
        },
    )

    assert "trainer_variant" not in summary
    assert summary["pipeline"] == {"id": "stage2_rollout_correction"}


def test_loader_routes_malformed_target_hierarchy_to_detection_schema() -> None:
    payload = _stage2_rollout_correction_payload()
    payload.pop("prompt")

    prompts = ConfigLoader.resolve_prompts(payload)

    with pytest.raises(ValueError, match=r"Missing detection config sections: .*prompt"):
        ConfigLoader._materialize_training_config(payload, prompts)


def test_stage2_target_hierarchy_rejects_stage1_detection_dataset_path() -> None:
    cfg = DetectionTrainingConfig.from_mapping(_stage2_rollout_correction_payload())
    custom_config = build_detection_runtime_custom_shim(cfg)

    with pytest.raises(
        ValueError,
        match=r"stage2_rollout_correction.*Stage-1 detection dataset runtime",
    ):
        detection_mode(cfg)

    with pytest.raises(
        ValueError,
        match=r"stage2_rollout_correction.*build_detection_dataset",
    ):
        build_detection_dataset(
            "unused.jsonl",
            swift_template=SimpleNamespace(),
            training_config=cfg,
            custom_config=custom_config,
            system_prompt=None,
            seed=17,
            sample_limit=None,
            dataset_name="stage2_guard",
        )


def test_sft_entrypoint_routes_stage2_target_hierarchy_away_from_stage1_builder() -> None:
    stage1_cfg = DetectionTrainingConfig.from_mapping(_detection_payload())
    stage2_cfg = DetectionTrainingConfig.from_mapping(_stage2_rollout_correction_payload())

    assert _uses_stage1_detection_dataset_builder(
        detection_config=stage1_cfg,
        runtime_plan=resolve_training_runtime_plan(None),
    )
    assert not _uses_stage1_detection_dataset_builder(
        detection_config=stage2_cfg,
        runtime_plan=resolve_training_runtime_plan("stage2_rollout_correction"),
    )
    assert not _uses_stage1_detection_dataset_builder(
        detection_config=None,
        runtime_plan=resolve_training_runtime_plan(None),
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
            id="research_teacher_forcing",
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

    with pytest.raises(ValueError, match="research_teacher_forcing encoded training cache"):
        _validate_sft_runtime_preflight(
            training_config=config,
            runtime_plan=resolve_training_runtime_plan("stage2_rollout_correction"),
        )


def test_sft_runtime_preflight_bypasses_teacher_forcing_encoded_sample_cache() -> None:
    config = SimpleNamespace(
        objective=SimpleNamespace(
            id="research_teacher_forcing",
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
        == PUBLIC_RESEARCH_TF_EPOCH_VARYING_ROLLIN_BYPASS_REASON
    )
    assert decision.bypass_reason != "teacher_forcing_epoch_varying_rollin"
    assert decision.bypass_info_for_split(
        dataset_split="train",
        dataset_jsonl="/tmp/train.jsonl",
    ) == {
        "enabled": True,
        "status": "bypassed",
        "reason": PUBLIC_RESEARCH_TF_EPOCH_VARYING_ROLLIN_BYPASS_REASON,
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


def test_effective_runtime_uses_token_embeddings_adapter_artifact_key() -> None:
    training_config = DetectionTrainingConfig.from_mapping(_detection_payload())

    payload = _build_effective_runtime_payload(
        training_config=training_config,
        train_args=SimpleNamespace(
            run_name="test-run",
            output_dir="/tmp/out",
            logging_dir="/tmp/logs",
            save_strategy="no",
            save_last_epoch=False,
            save_only_model=False,
            seed=17,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            eval_strategy="no",
            eval_steps=0,
            max_steps=-1,
            num_train_epochs=1,
            dataloader_drop_last=False,
            max_model_len=1024,
            group_by_length=False,
        ),
        trainer_variant=None,
        dataset_seed=17,
        checkpoint_mode="artifact_only",
        packing_cfg=PackingRuntimeConfig(enabled=False),
        encoded_sample_cache_cfg=EncodedSampleCacheRuntimeConfig(enabled=False),
        train_jsonl=training_config.data.train_jsonl,
        val_jsonl=training_config.data.val_jsonl,
        pipeline_manifest={"checksum": "abc"},
    )

    assert "token_rows" not in payload
    adapter = payload["token_embeddings_adapter"]
    assert adapter["enabled"] is True
    assert adapter["tie_head"] is True
    assert adapter["expected_trainable_row_count"] == 1002
    assert adapter["groups"]["coord_geometry"]["expected_row_count"] == 1000
    assert adapter["groups"]["compact_structure"]["expected_row_count"] == 2


def test_effective_runtime_omits_private_trainer_variant_selector() -> None:
    training_config = DetectionTrainingConfig.from_mapping(
        _stage2_rollout_correction_payload()
    )

    payload = _build_effective_runtime_payload(
        training_config=training_config,
        train_args=SimpleNamespace(
            run_name="stage2-test-run",
            output_dir="/tmp/out",
            logging_dir="/tmp/logs",
            save_strategy="no",
            save_last_epoch=False,
            save_only_model=False,
            seed=17,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            eval_strategy="no",
            eval_steps=0,
            max_steps=-1,
            num_train_epochs=1,
            dataloader_drop_last=False,
            max_model_len=1024,
            group_by_length=False,
        ),
        trainer_variant="stage2_rollout_correction",
        dataset_seed=17,
        checkpoint_mode="artifact_only",
        packing_cfg=PackingRuntimeConfig(enabled=True, mode="dynamic"),
        encoded_sample_cache_cfg=EncodedSampleCacheRuntimeConfig(enabled=False),
        train_jsonl=training_config.data.train_jsonl,
        val_jsonl=training_config.data.val_jsonl,
        pipeline_manifest={"checksum": "abc"},
    )

    assert "trainer_variant" not in payload
    assert payload["training_hierarchy"]["pipeline"]["id"] == (
        "stage2_rollout_correction"
    )


def test_stage2_normalized_hierarchy_records_internal_objective_identity() -> None:
    training_config = DetectionTrainingConfig.from_mapping(
        _stage2_rollout_correction_payload()
    )

    hierarchy = _build_normalized_training_hierarchy_identity(
        training_config=training_config,
        template=SimpleNamespace(max_length=1024),
        train_args=SimpleNamespace(max_model_len=1024),
        packing_length=1024,
    )

    assert hierarchy["pipeline"]["id"] == "stage2_rollout_correction"
    assert hierarchy["objective"]["id"] == "residual_set_correction"
    assert hierarchy["objective"]["stage2_objectives"] == [
        "residual_set_correction"
    ]


def test_effective_runtime_records_normalized_training_hierarchy_identity() -> None:
    training_config = DetectionTrainingConfig.from_mapping(_detection_payload())

    payload = _build_effective_runtime_payload(
        training_config=training_config,
        train_args=SimpleNamespace(
            run_name="test-run",
            output_dir="/tmp/out",
            logging_dir="/tmp/logs",
            save_strategy="no",
            save_last_epoch=False,
            save_only_model=False,
            seed=17,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            eval_strategy="no",
            eval_steps=0,
            max_steps=-1,
            num_train_epochs=1,
            dataloader_drop_last=False,
            max_model_len=1024,
            group_by_length=False,
            model="model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp",
        ),
        trainer_variant=None,
        dataset_seed=17,
        checkpoint_mode="artifact_only",
        packing_cfg=PackingRuntimeConfig(
            enabled=True,
            mode="static",
            packing_length=1024,
        ),
        encoded_sample_cache_cfg=EncodedSampleCacheRuntimeConfig(enabled=False),
        train_jsonl=training_config.data.train_jsonl,
        val_jsonl=training_config.data.val_jsonl,
        pipeline_manifest={"checksum": "abc"},
    )

    hierarchy = payload["training_hierarchy"]
    assert hierarchy["pipeline"]["id"] == "stage1_standard_sft"
    assert hierarchy["objective"]["id"] == "standard_ce"
    assert hierarchy["detection_template"]["id"] == "compact"
    assert hierarchy["sample_factory"]["id"] == "detection_sequence"
    assert hierarchy["sample_factory"]["target_sequence"] == {
        "object_ordering": "random_permutation",
        "object_field_order": "desc_first",
        "bbox_format": "xyxy",
        "coordinate_surface": "coord_token",
        "strict_parse": True,
    }
    assert hierarchy["prompt"]["variant"] == "coco_80"
    assert isinstance(hierarchy["prompt"]["template_hash"], str)
    assert hierarchy["tokenizer"]["id"].endswith("Qwen3-VL-2B-Instruct-coordexp")
    assert hierarchy["chat_template"]["identity"] == "unknown_chat_template"
    assert hierarchy["packing"]["length"] == 1024


def test_effective_runtime_uses_public_encoded_sample_cache_bypass_reason() -> None:
    config = SimpleNamespace(
        objective=SimpleNamespace(
            id="research_teacher_forcing",
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

    payload = _build_effective_runtime_payload(
        training_config=config,
        train_args=SimpleNamespace(
            run_name="test-run",
            output_dir="/tmp/out",
            logging_dir="/tmp/logs",
            save_strategy="no",
            save_last_epoch=False,
            save_only_model=False,
            seed=17,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            eval_strategy="no",
            eval_steps=0,
            max_steps=-1,
            num_train_epochs=1,
            dataloader_drop_last=False,
            max_model_len=1024,
            group_by_length=False,
        ),
        trainer_variant="stage2_rollout_correction",
        dataset_seed=17,
        checkpoint_mode="artifact_only",
        packing_cfg=PackingRuntimeConfig(enabled=False),
        encoded_sample_cache_cfg=decision.encoded_sample_cache_cfg,
        train_jsonl="/tmp/train.jsonl",
        val_jsonl=None,
        pipeline_manifest={"checksum": "abc"},
        train_encoded_sample_cache_info=decision.bypass_info_for_split(
            dataset_split="train",
            dataset_jsonl="/tmp/train.jsonl",
        ),
    )

    encoded_cache = payload["encoded_sample_cache"]
    assert (
        encoded_cache["reason"]
        == PUBLIC_RESEARCH_TF_EPOCH_VARYING_ROLLIN_BYPASS_REASON
    )
    assert encoded_cache["reason"] != "teacher_forcing_epoch_varying_rollin"
    assert encoded_cache["train"]["reason"] == encoded_cache["reason"]


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


def test_teacher_forcing_objective_mixin_passes_coverage_ledger_to_bridge_and_logs_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Metric:
        def __init__(self) -> None:
            self.values: list[float] = []

        def update(self, value: float) -> None:
            self.values.append(float(value))

    class _BaseTrainer:
        def compute_loss(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("base trainer CE path should not own teacher_forcing")

    class _Trainer(TeacherForcingObjectiveMixin, _BaseTrainer):
        pass

    class _FakeBridge:
        settings_seen: Any = None
        training_sidecars_seen: Any = None
        raw_batch_seen: Any = None

        def __init__(self, *, settings=None, **_kwargs: Any) -> None:
            self.__class__.settings_seen = settings

        def compute_loss(self, **kwargs: Any):
            self.__class__.training_sidecars_seen = kwargs.get("training_sidecars")
            self.__class__.raw_batch_seen = dict(kwargs["raw_batch"])
            return SimpleNamespace(
                loss=torch.tensor(1.25, dtype=torch.float32),
                outputs=SimpleNamespace(marker="bridge-outputs"),
                metric_events=(
                    weighted_mean_event(
                        "teacher_forcing/loss/coverage_ledger_auxiliary_weighted",
                        0.5,
                        2.0,
                        unit="object",
                        objective_id="coverage_ledger",
                    ),
                ),
            )

    monkeypatch.setattr(teacher_forcing_metrics, "TrainerLossBridge", _FakeBridge)
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
    sidecars = TrainingSidecars(
        supervision=SupervisionSidecars(payloads=("ledger-sidecar-placeholder",))
    )
    inputs = {
        "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, 1]], dtype=torch.long),
        "teacher_forcing_target_ir": (target_ir,),
        "training_sidecars": sidecars,
    }
    trainer = _Trainer()
    trainer.teacher_forcing_objective_cfg = SimpleNamespace(
        profile="pure_valid_set_marginal",
        modules=SimpleNamespace(
            within_valid_coverage=SimpleNamespace(coverage_strength=0.0),
            coverage_ledger=SimpleNamespace(
                enabled=True,
                coverage_weight=0.5,
                region_anchor_weight=0.25,
                temperature=1.0,
                pos_weight=1.0,
            ),
        ),
    )
    trainer.teacher_forcing_role_vocab = RoleVocab(
        text_token_ids=frozenset({1, 2}),
        schema_token_ids=frozenset(),
        coord_token_ids=frozenset(),
        stop_token_id=9,
    )
    metrics = defaultdict(_Metric)
    trainer.custom_metrics = {"train": metrics}

    loss, outputs = trainer.compute_loss(
        SimpleNamespace(),
        inputs,
        return_outputs=True,
    )

    assert loss.item() == pytest.approx(1.25)
    assert outputs.marker == "bridge-outputs"
    assert _FakeBridge.settings_seen.coverage_ledger is (
        trainer.teacher_forcing_objective_cfg.modules.coverage_ledger
    )
    assert _FakeBridge.training_sidecars_seen is sidecars
    assert "training_sidecars" not in _FakeBridge.raw_batch_seen
    assert metrics[
        "teacher_forcing/loss/coverage_ledger_auxiliary_weighted"
    ].values == [pytest.approx(0.5)]


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
            pipeline={"id": "stage2_rollout_correction"},
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


def test_stage2_runtime_projection_rejects_removed_rollout_pipeline_with_target_hierarchy_guidance() -> None:
    with pytest.raises(
        ValueError,
        match=r"rollout_matching\.pipeline.*stage2_rollout_correction\.pipeline.*pipeline\.id=stage2_rollout_correction",
    ):
        resolve_stage2_runtime_projection(
            training_config=SimpleNamespace(
                pipeline={"id": "stage2_rollout_correction"},
                rollout_matching={"pipeline": {"objective": []}},
                stage2_rollout_correction={"pipeline": {"objective": []}},
            ),
            custom_config=_runtime_projection_custom_config(),
            packing_cfg=_runtime_projection_packing_config(),
            trainer_variant="stage2_rollout_correction",
            config_path="configs/stage2.yaml",
            run_name="projection-test",
            seed=17,
        )


def test_stage2_runtime_projection_marks_custom_extra_prompt_compat_fallback() -> None:
    projection = resolve_stage2_runtime_projection(
        training_config=SimpleNamespace(
            pipeline={"id": "stage2_rollout_correction"},
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


def test_stage2_runtime_projection_labels_target_hierarchy_sources() -> None:
    cfg = DetectionTrainingConfig.from_mapping(_stage2_rollout_correction_payload())
    projection = resolve_stage2_runtime_projection(
        training_config=cfg,
        custom_config=build_detection_runtime_custom_shim(cfg),
        packing_cfg=_runtime_projection_packing_config(),
        trainer_variant="stage2_rollout_correction",
        config_path="configs/stage2.yaml",
        run_name="projection-test",
        seed=17,
    )

    assert projection is not None
    assert projection.rollout_matching_cfg["prompt_variant"] == "coco_80"
    assert projection.rollout_matching_cfg["eval_prompt_variant"] == "coco_80"
    assert projection.rollout_matching_cfg["object_ordering"] == "random_permutation"
    assert projection.rollout_matching_cfg["object_field_order"] == "desc_first"
    assert projection.rollout_matching_cfg["bbox_format"] == "xyxy"
    assert projection.rollout_matching_cfg["detection_sequence_format"] == "compact_full"
    assert projection.policy_sources["rollout_matching.prompt_variant"] == (
        "prompt.variant"
    )
    assert projection.policy_sources["rollout_matching.eval_prompt_variant"] == (
        "prompt.variant"
    )
    assert projection.policy_sources["object_ordering"] == (
        "sample_factory.target_sequence.object_ordering"
    )
    assert projection.policy_sources["object_field_order"] == (
        "sample_factory.target_sequence.object_field_order"
    )
    assert projection.policy_sources["bbox_format"] == (
        "sample_factory.target_sequence.bbox_format"
    )
    assert projection.policy_sources["detection_sequence_format"] == (
        "detection_template.id+sample_factory.id"
    )
    assert projection.stage2_policy_provenance[
        "runtime_compatibility_fallbacks"
    ] == []
    assert projection.stage2_policy_provenance["pipeline"]["id"] == (
        "stage2_rollout_correction"
    )
    assert "trainer_variant" not in projection.stage2_policy_provenance
    assert all(
        not source.startswith("custom.")
        for key, source in projection.policy_sources.items()
        if not key.startswith("packing.")
    )


def test_apply_stage2_runtime_projection_sets_trainer_boundary_attrs() -> None:
    calls: list[str] = []

    class _Trainer:
        def _validate_rollout_matching_cfg(self) -> None:
            calls.append("validated")

    trainer = _Trainer()
    projection = resolve_stage2_runtime_projection(
        training_config=SimpleNamespace(
            pipeline={"id": "stage2_rollout_correction"},
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

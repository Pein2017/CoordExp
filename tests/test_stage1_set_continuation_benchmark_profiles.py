from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.bootstrap.experiment_manifest import write_experiment_manifest_file
from src.config.loader import ConfigLoader
from src.sft import (
    EncodedSampleCacheRuntimeConfig,
    PackingRuntimeConfig,
    _build_effective_runtime_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "stage1" / "set_continuation"
PRODUCTION_CONFIG = CONFIG_ROOT / "production.yaml"
SOTA_STAGE1_CHECKPOINT = (
    "output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/"
    "epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full"
)


def _load_production_config():
    return ConfigLoader.load_materialized_training_config(str(PRODUCTION_CONFIG))


def test_only_one_set_continuation_entry_config_is_checked_in() -> None:
    yaml_files = sorted(path.name for path in CONFIG_ROOT.glob("*.yaml"))

    assert yaml_files == ["production.yaml"]


def test_production_profile_resolves_and_pins_common_contract() -> None:
    cfg = _load_production_config()

    assert cfg.benchmark.group_id == "stage1_set_continuation_full_features"
    assert cfg.benchmark.control_group_id == "stage1_sft_sota1332"
    assert cfg.benchmark.intended_variable
    assert cfg.benchmark.comparability_label == "accuracy-comparable"
    assert cfg.model["model"] == SOTA_STAGE1_CHECKPOINT
    assert cfg.training["output_root"] == (
        "/data/CoordExp/output_remote/stage1_2b/set_continuation"
    )
    assert cfg.training["logging_root"] == (
        "/data/CoordExp/output_remote/stage1_2b/set_continuation/tb"
    )
    assert (
        cfg.training["artifact_subdir"]
        == "coco1024_sota1332_setcont_pem_close_suppress"
    )
    assert (
        cfg.training["run_name"]
        == "setcont-coco1024-sota1332-pem-close-suppress"
    )
    assert cfg.training["packing"] is False
    assert cfg.training["eval_packing"] is False
    assert cfg.training["encoded_sample_cache"]["enabled"] is False
    assert cfg.training["seed"] == 17
    assert cfg.training["num_train_epochs"] == 4
    assert cfg.training["effective_batch_size"] == 128
    assert cfg.training["learning_rate"] == pytest.approx(5.0e-5)
    assert cfg.training["vit_lr"] == pytest.approx(1.0e-5)
    assert cfg.training["aligner_lr"] == pytest.approx(5.0e-5)
    assert cfg.training["metric_for_best_model"] == "eval_det_bbox_AP"
    assert cfg.training["greater_is_better"] is True
    assert cfg.training.get("max_steps") is None
    assert cfg.custom.train_sample_limit is None
    assert str(cfg.custom.train_jsonl).endswith(
        "public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl"
    )
    assert str(cfg.custom.val_jsonl).endswith(
        "public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl"
    )
    assert cfg.template["max_pixels"] == 1048576
    assert cfg.custom.extra["prompt_variant"] == "coco_80"
    assert cfg.custom.object_field_order == "desc_first"
    assert cfg.custom.object_ordering == "sorted"
    assert cfg.custom.coord_tokens.enabled is True
    assert cfg.custom.coord_tokens.skip_bbox_norm is True
    assert cfg.custom.coord_offset.enabled is True
    assert cfg.custom.coord_offset.tie_head is True
    assert cfg.custom.coord_offset.embed_lr == pytest.approx(5.0e-5)
    assert cfg.custom.coord_offset.head_lr == pytest.approx(5.0e-5)
    assert cfg.custom.coord_soft_ce_w1.enabled is False
    assert cfg.custom.bbox_geo.enabled is False
    assert cfg.custom.bbox_size_aux.enabled is False
    assert cfg.custom.eval_detection.enabled is True
    assert cfg.custom.eval_detection.metrics == "both"
    assert cfg.custom.eval_detection.score_mode == "confidence_postop"
    assert cfg.custom.eval_detection.limit == 200
    assert cfg.custom.eval_detection.max_new_tokens == 3084
    assert cfg.custom.eval_detection.distributed is True
    assert cfg.custom.eval_detection.temperature == pytest.approx(0.0)
    assert cfg.custom.eval_detection.top_p == pytest.approx(1.0)
    assert cfg.custom.eval_detection.repetition_penalty == pytest.approx(1.05)
    assert cfg.custom.eval_detection.f1ish_pred_scope == "annotated"

    report = cfg.custom.extra["benchmark_report"]
    assert report["eval_scope"] == "val200"
    assert report["eval_view"] == "coco_map_with_logprob_confidence_plus_f1ish_annotated"
    assert report["dataset_choice"] == "original_coco_coord_token_train_surface"
    assert report["dataset_ablation_note"] == "original_coco_first_lvis_proxy_followup"
    assert report["prediction_surface"] == "coord_token_xyxy"
    assert report["score_mode"] == "confidence_postop_bbox_logprob_confidence_exp"
    assert report["decoding_controls"] == "greedy_temp0_top_p1_rep1p05"
    assert report["same_budget_label"] == "repeated_forward_correctness_v1"
    assert (
        report["sparse_label_caveat"]
        == "annotated_view_may_count_unlabeled_true_positives_as_fp"
    )


def test_production_profile_enables_all_set_continuation_features() -> None:
    cfg = _load_production_config()
    sc = cfg.custom.stage1_set_continuation
    pem = sc.positive_evidence_margin

    assert cfg.custom.trainer_variant == "stage1_set_continuation"
    assert sc is not None
    assert sc.candidates.mode == "exact"
    assert sc.candidates.max_candidates is None
    assert sc.subset_sampling.empty_prefix_ratio == pytest.approx(0.30)
    assert sc.subset_sampling.random_subset_ratio == pytest.approx(0.50)
    assert sc.subset_sampling.leave_one_out_ratio == pytest.approx(0.20)
    assert sc.subset_sampling.full_prefix_ratio == pytest.approx(0.0)
    assert sc.subset_sampling.prefix_order == "random"
    assert sc.structural_close.close_start_suppression_weight == pytest.approx(0.05)
    assert sc.structural_close.final_schema_close_weight == pytest.approx(0.0)
    assert pem.objective == "threshold_loss"
    assert pem.threshold_space == "full_entry_logZ"
    assert pem.rho == pytest.approx(0.90)
    assert pem.log_rho is None
    assert pem.threshold_calibration == "fixed_rho_0.90_no_external_evaluator_v1"
    assert sc.metric_schema_version == "stage1_set_continuation_metrics_v1"
    assert cfg.deepspeed is not None
    assert cfg.deepspeed.enabled is False


def test_effective_runtime_records_production_set_continuation_provenance() -> None:
    cfg = _load_production_config()

    runtime = _build_effective_runtime_payload(
        training_config=cfg,
        train_args=SimpleNamespace(
            output_dir="out",
            logging_dir="logs",
            run_name="unit",
            seed=17,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=128,
            max_steps=-1,
            num_train_epochs=4,
        ),
        trainer_variant=cfg.custom.trainer_variant,
        dataset_seed=17,
        checkpoint_mode="artifact_only",
        packing_cfg=PackingRuntimeConfig(enabled=False),
        encoded_sample_cache_cfg=EncodedSampleCacheRuntimeConfig(enabled=False),
        train_jsonl=cfg.custom.train_jsonl,
        val_jsonl=cfg.custom.val_jsonl,
        pipeline_manifest={"checksum": "abc123"},
    )

    assert runtime["benchmark_group_id"] == "stage1_set_continuation_full_features"
    assert runtime["control_group_id"] == "stage1_sft_sota1332"
    assert runtime["intended_variable"] == cfg.benchmark.intended_variable
    assert runtime["comparability_label"] == "accuracy-comparable"
    assert runtime["deepspeed"] == {
        "enabled": False,
        "config": "zero2",
        "train_args_deepspeed": None,
    }
    sc_runtime = runtime["stage1_set_continuation"]
    assert sc_runtime["candidate_scoring_mode"] == "exact"
    assert sc_runtime["logZ_estimator"] == "exact"
    assert sc_runtime["collator_path"].endswith(
        "build_stage1_set_continuation_collator"
    )
    assert sc_runtime["remove_unused_columns"] is False
    assert sc_runtime["packing_policy"] == {
        "training.packing": "rejected",
        "training.eval_packing": "rejected",
        "static_pack_plan": "not_built",
        "reason": "runtime_subset_candidate_branch_sampling_requires_unpacked_rows",
    }
    assert sc_runtime["prefix_attach_mode"] == "repeated_forward"
    assert sc_runtime["branch_isolation"] == "independent_forward"
    assert sc_runtime["branch_attention_mask"] == {
        "enabled": False,
        "reason": "naive_repeated_independent_forwards_do_not_share_candidate_sequence",
    }
    assert sc_runtime["prefix_gradient"] == "non_detached_recomputed_per_branch"
    assert sc_runtime["metric_schema_version"] == "stage1_set_continuation_metrics_v1"
    assert sc_runtime["positive_evidence_margin"] == {
        "objective": "threshold_loss",
        "threshold_space": "full_entry_logZ",
        "rho": 0.90,
        "threshold_calibration": "fixed_rho_0.90_no_external_evaluator_v1",
    }
    assert sc_runtime["effective_coord_slot_scoring"] == "coord_token_vocab_full_entry"
    assert sc_runtime["raw_text_integer_coordinates"] == "unsupported"
    assert (
        sc_runtime["realized_branch_token_budget"]["v1_execution"]
        == "naive_repeated_forward_no_prefix_cache"
    )
    assert sc_runtime["realized_prefix_mode_coverage"]["source"] == "trainer_metrics"
    assert sc_runtime["realized_aux_settings"]["coord_soft_ce_w1"]["enabled"] is False


def test_experiment_manifest_mirrors_production_runtime_summary(tmp_path: Path) -> None:
    effective_runtime = {
        "trainer_variant": "stage1_set_continuation",
        "deepspeed": {
            "enabled": False,
            "config": "zero2",
            "train_args_deepspeed": None,
        },
        "benchmark": {"group_id": "stage1_set_continuation_full_features"},
        "benchmark_group_id": "stage1_set_continuation_full_features",
        "control_group_id": "stage1_sft_sota1332",
        "intended_variable": "full-entry MP plus close-start suppression and PEM",
        "comparability_label": "accuracy-comparable",
        "stage1_set_continuation": {
            "candidate_scoring_mode": "exact",
            "logZ_estimator": "exact",
            "prefix_attach_mode": "repeated_forward",
            "branch_isolation": "independent_forward",
            "prefix_gradient": "non_detached_recomputed_per_branch",
            "collator_path": "src.data_collators.stage1_set_continuation_collator.build_stage1_set_continuation_collator",
            "packing_policy": {"training.packing": "rejected"},
            "metric_schema_version": "stage1_set_continuation_metrics_v1",
        },
    }

    out_path = write_experiment_manifest_file(
        output_dir=tmp_path,
        config_path="configs/stage1/set_continuation/production.yaml",
        base_config_path=None,
        run_name="unit",
        dataset_seed=17,
        experiment=None,
        effective_runtime=effective_runtime,
        pipeline_manifest=None,
        run_metadata=None,
        manifest_files={"effective_runtime": "effective_runtime.json"},
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    summary = payload["runtime_summary"]
    assert summary["deepspeed"] == {
        "enabled": False,
        "config": "zero2",
        "train_args_deepspeed": None,
    }
    assert summary["benchmark_group_id"] == "stage1_set_continuation_full_features"
    assert summary["control_group_id"] == "stage1_sft_sota1332"
    assert summary["stage1_set_continuation"]["candidate_scoring_mode"] == "exact"
    assert summary["stage1_set_continuation"]["packing_policy"][
        "training.packing"
    ] == "rejected"
    assert summary["benchmark"]["group_id"] == "stage1_set_continuation_full_features"

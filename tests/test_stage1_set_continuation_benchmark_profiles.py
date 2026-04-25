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

GROUP_CONFIGS = {
    "group_a_sft": CONFIG_ROOT / "group_a_sft.yaml",
    "group_b_sft_weak_schema_close": CONFIG_ROOT
    / "group_b_sft_weak_schema_close.yaml",
    "group_c_exact_mp": CONFIG_ROOT / "group_c_exact_mp.yaml",
    "group_d_mp_anti_close": CONFIG_ROOT / "group_d_mp_anti_close.yaml",
    "group_e_pem_replace": CONFIG_ROOT / "group_e_pem_replace.yaml",
    "group_f_leave_one_out": CONFIG_ROOT / "group_f_leave_one_out.yaml",
}


def _load(group_id: str):
    return ConfigLoader.load_materialized_training_config(str(GROUP_CONFIGS[group_id]))


@pytest.mark.parametrize("group_id", sorted(GROUP_CONFIGS))
def test_benchmark_profiles_resolve_and_pin_common_contract(group_id: str) -> None:
    cfg = _load(group_id)

    assert cfg.benchmark.group_id == group_id
    assert cfg.benchmark.intended_variable
    assert cfg.benchmark.comparability_label == "accuracy-comparable"
    assert cfg.training["packing"] is False
    assert cfg.training["eval_packing"] is False
    assert cfg.training["encoded_sample_cache"]["enabled"] is False
    assert cfg.training["seed"] == 17
    assert cfg.training["num_train_epochs"] == 4
    assert cfg.training["effective_batch_size"] == 128
    assert str(cfg.model["model"]).endswith("Qwen3-VL-2B-Instruct-coordexp")
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
    assert cfg.custom.coord_soft_ce_w1.enabled is False
    assert cfg.custom.bbox_geo.enabled is False
    assert cfg.custom.bbox_size_aux.enabled is False
    assert cfg.custom.eval_detection.enabled is True
    assert cfg.custom.eval_detection.metrics == "f1ish"
    assert cfg.custom.eval_detection.limit == 200
    assert cfg.custom.eval_detection.max_new_tokens == 1024
    assert cfg.custom.eval_detection.temperature == pytest.approx(0.0)
    assert cfg.custom.eval_detection.top_p == pytest.approx(1.0)
    assert cfg.custom.eval_detection.repetition_penalty == pytest.approx(1.0)
    report = cfg.custom.extra["benchmark_report"]
    assert report["eval_scope"] == "val200"
    assert report["eval_view"] == "f1ish_annotated"
    assert report["prediction_surface"] == "coord_token_xyxy"
    assert (
        report["sparse_label_caveat"]
        == "annotated_view_may_count_unlabeled_true_positives_as_fp"
    )


def test_group_a_is_ordinary_sft_baseline() -> None:
    cfg = _load("group_a_sft")

    assert cfg.custom.trainer_variant is None
    assert cfg.custom.stage1_set_continuation is None
    assert cfg.custom.sft_structural_close.enabled is False
    assert cfg.benchmark.control_group_id is None


def test_group_b_is_ordinary_sft_with_weak_global_schema_close() -> None:
    cfg = _load("group_b_sft_weak_schema_close")

    assert cfg.custom.trainer_variant is None
    assert cfg.custom.stage1_set_continuation is None
    assert cfg.custom.sft_structural_close.enabled is True
    assert cfg.custom.sft_structural_close.final_close_weight == pytest.approx(0.0)
    assert cfg.benchmark.control_group_id == "group_a_sft"


@pytest.mark.parametrize(
    ("group_id", "control_group_id"),
    [
        ("group_c_exact_mp", "group_a_sft"),
        ("group_d_mp_anti_close", "group_c_exact_mp"),
        ("group_e_pem_replace", "group_c_exact_mp"),
        ("group_f_leave_one_out", "group_c_exact_mp"),
    ],
)
def test_set_continuation_profiles_pin_variant_and_branch_semantics(
    group_id: str, control_group_id: str
) -> None:
    cfg = _load(group_id)
    sc = cfg.custom.stage1_set_continuation

    assert cfg.custom.trainer_variant == "stage1_set_continuation"
    assert sc is not None
    assert cfg.benchmark.control_group_id == control_group_id
    assert sc.candidates.mode == "exact"
    assert sc.candidates.max_candidates is None
    assert sc.subset_sampling.prefix_order == "random"
    assert (
        sc.metric_schema_version == "stage1_set_continuation_metrics_v1"
    )


def test_group_c_and_d_use_initial_subset_mixture_and_d_adds_anti_close() -> None:
    group_c = _load("group_c_exact_mp").custom.stage1_set_continuation
    group_d = _load("group_d_mp_anti_close").custom.stage1_set_continuation

    assert group_c.subset_sampling.empty_prefix_ratio == pytest.approx(0.30)
    assert group_c.subset_sampling.random_subset_ratio == pytest.approx(0.50)
    assert group_c.subset_sampling.leave_one_out_ratio == pytest.approx(0.20)
    assert group_c.subset_sampling.full_prefix_ratio == pytest.approx(0.0)
    assert group_c.structural_close.anti_close_weight == pytest.approx(0.0)
    assert group_d.structural_close.anti_close_weight == pytest.approx(0.05)
    assert group_d.structural_close.final_close_weight == pytest.approx(0.0)


def test_group_e_uses_replacement_pem_with_fixed_rho_provenance() -> None:
    cfg = _load("group_e_pem_replace").custom.stage1_set_continuation
    pem = cfg.positive_evidence_margin

    assert pem.mode == "replace_mp"
    assert pem.threshold_space == "full_entry_logZ"
    assert pem.rho == pytest.approx(0.90)
    assert pem.log_rho is None
    assert pem.threshold_calibration == "fixed_rho_0.90_no_external_evaluator_v1"


def test_group_f_emphasizes_leave_one_out_prefixes() -> None:
    cfg = _load("group_f_leave_one_out").custom.stage1_set_continuation

    assert cfg.subset_sampling.empty_prefix_ratio == pytest.approx(0.10)
    assert cfg.subset_sampling.random_subset_ratio == pytest.approx(0.15)
    assert cfg.subset_sampling.leave_one_out_ratio == pytest.approx(0.75)
    assert cfg.subset_sampling.full_prefix_ratio == pytest.approx(0.0)


def test_effective_runtime_records_benchmark_and_set_continuation_provenance() -> None:
    cfg = _load("group_e_pem_replace")

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

    assert runtime["benchmark_group_id"] == "group_e_pem_replace"
    assert runtime["control_group_id"] == "group_c_exact_mp"
    assert runtime["intended_variable"] == cfg.benchmark.intended_variable
    assert runtime["comparability_label"] == "accuracy-comparable"
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
    assert sc_runtime["prefix_gradient"] == "non_detached_recomputed_per_branch"
    assert sc_runtime["metric_schema_version"] == "stage1_set_continuation_metrics_v1"
    assert sc_runtime["positive_evidence_margin"]["mode"] == "replace_mp"
    assert (
        sc_runtime["positive_evidence_margin"]["threshold_calibration"]
        == "fixed_rho_0.90_no_external_evaluator_v1"
    )
    assert sc_runtime["effective_coord_slot_scoring"] == "coord_token_vocab_full_entry"
    assert sc_runtime["realized_branch_token_budget"]["source"] == "trainer_metrics"
    assert sc_runtime["realized_prefix_mode_coverage"]["source"] == "trainer_metrics"
    assert sc_runtime["realized_aux_settings"]["coord_soft_ce_w1"]["enabled"] is False


def test_experiment_manifest_mirrors_benchmark_runtime_summary(tmp_path: Path) -> None:
    effective_runtime = {
        "trainer_variant": "stage1_set_continuation",
        "benchmark": {"group_id": "group_c_exact_mp"},
        "benchmark_group_id": "group_c_exact_mp",
        "control_group_id": "group_a_sft",
        "intended_variable": "full-entry multi-positive objective",
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
        config_path="configs/stage1/set_continuation/group_c_exact_mp.yaml",
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
    assert summary["benchmark_group_id"] == "group_c_exact_mp"
    assert summary["control_group_id"] == "group_a_sft"
    assert summary["stage1_set_continuation"]["candidate_scoring_mode"] == "exact"
    assert summary["stage1_set_continuation"]["packing_policy"][
        "training.packing"
    ] == "rejected"
    assert summary["benchmark"]["group_id"] == "group_c_exact_mp"

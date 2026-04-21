from pathlib import Path

from src.analysis.raw_text_coordinate_mechanism_study import (
    load_study_config,
    plan_stage_cells,
    run_study,
)


def test_load_study_config_parses_two_fixed_raw_text_models(tmp_path: Path) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        """
run:
  name: raw-text-coordinate-mechanism
  output_dir: output/analysis
  stages: [contract, case_bank, confirmatory, shortlist, exploratory, representation, review, report]

models:
  base_only:
    alias: base_only
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: null
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]
  base_plus_adapter:
    alias: base_plus_adapter
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: output/stage1_2b/demo/checkpoint-552
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]

dataset:
  train_jsonl: public_data/coco/rescale_32_1024_bbox_max60/train.norm.jsonl
  val_jsonl: public_data/coco/rescale_32_1024_bbox_max60/val.norm.jsonl

execution:
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  reuse_existing: true

review:
  fp_budget: 15
  fn_budget: 5
        """.strip(),
        encoding="utf-8",
    )

    cfg = load_study_config(config_path)

    assert cfg.models.base_only.adapter_path is None
    assert cfg.models.base_plus_adapter.adapter_path.endswith("checkpoint-552")
    assert cfg.review.fp_budget == 15
    assert tuple(cfg.run.stages) == (
        "contract",
        "case_bank",
        "confirmatory",
        "shortlist",
        "exploratory",
        "representation",
        "review",
        "report",
    )


def test_plan_stage_cells_splits_models_across_available_gpus() -> None:
    stage_cells = plan_stage_cells(
        stage="exploratory",
        gpu_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        model_aliases=("base_only", "base_plus_adapter"),
        branch_names=("duplicate_fp", "fn", "heatmap", "perturb"),
    )

    assert len(stage_cells) == 8
    assert {cell["gpu_id"] for cell in stage_cells} == set(range(8))
    assert stage_cells[0]["model_alias"] == "base_only"
    assert stage_cells[-1]["branch_name"] == "perturb"


def test_run_study_materializes_run_dir_and_summary(tmp_path: Path) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        f"""
run:
  name: smoke-study
  output_dir: {tmp_path.as_posix()}
  stages: [contract, case_bank]

models:
  base_only:
    alias: base_only
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: null
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]
  base_plus_adapter:
    alias: base_plus_adapter
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: output/stage1_2b/demo/checkpoint-552
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]

dataset:
  train_jsonl: public_data/coco/rescale_32_1024_bbox_max60/train.norm.jsonl
  val_jsonl: public_data/coco/rescale_32_1024_bbox_max60/val.norm.jsonl

execution:
  gpu_ids: [0]
  reuse_existing: true

review:
  fp_budget: 2
  fn_budget: 1
        """.strip(),
        encoding="utf-8",
    )

    result = run_study(config_path)
    run_dir = tmp_path / "smoke-study"

    assert result["run_dir"] == str(run_dir)
    assert (run_dir / "stage_manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "case_bank.jsonl").exists()


def test_run_study_respects_stage_and_branch_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        f"""
run:
  name: override-study
  output_dir: {tmp_path.as_posix()}
  stages: [contract, case_bank, confirmatory]

models:
  base_only:
    alias: base_only
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: null
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]
  base_plus_adapter:
    alias: base_plus_adapter
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: output/stage1_2b/demo/checkpoint-552
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]

dataset:
  train_jsonl: public_data/coco/rescale_32_1024_bbox_max60/train.norm.jsonl
  val_jsonl: public_data/coco/rescale_32_1024_bbox_max60/val.norm.jsonl

execution:
  gpu_ids: [0]
  reuse_existing: true

review:
  fp_budget: 2
  fn_budget: 1
        """.strip(),
        encoding="utf-8",
    )

    result = run_study(
        config_path,
        stage_override="contract",
        model_alias="base_only",
        branch_name="duplicate_fp",
    )
    manifest = (tmp_path / "override-study" / "stage_manifest.json").read_text(
        encoding="utf-8"
    )

    assert result["requested_stage"] == "contract"
    assert '"stages": [\n    "contract"\n  ]' in manifest
    assert '"requested_model_alias": "base_only"' in manifest
    assert '"requested_branch_name": "duplicate_fp"' in manifest

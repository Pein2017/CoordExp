from pathlib import Path

from src.analysis.raw_text_coordinate_mechanism_study import (
    load_study_config,
    plan_stage_cells,
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

from __future__ import annotations

from pathlib import Path

import json

import pytest

import src.analysis.raw_text_coord_continuity_probe as continuity_probe_module
from src.analysis.duplication_collapse_analysis import mine_duplicate_like_rows
from src.analysis.duplication_followup import build_prefix_perturbation_variants
from src.analysis.raw_text_coord_continuity_report import (
    build_xy_heatmap_grid,
    compute_basin_metrics,
    compute_vision_lift_rows,
    summarize_wrong_anchor_advantage,
    write_report_bundle,
)
from src.analysis.raw_text_coord_continuity_probe import (
    _resolve_audit_model_info,
    build_random_cohort,
    build_study_hard_cases,
    load_study_config,
    render_pretty_inline_assistant_text,
    run_phase0_audit,
    run_study,
)
from src.infer.checkpoints import (
    AdapterCheckpointInfo,
    CoordOffsetAdapterSpec,
    ResolvedInferenceCheckpoint,
)


def test_load_study_config_parses_lanes_and_cohorts(tmp_path: Path) -> None:
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        """
run:
  name: raw-text-continuity
  output_dir: output/analysis
  stages: [audit, pilot, canonical, bad_basin, dense_scene, report]

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    prompt_surface: upper_bound
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/demo-checkpoint
    prompt_surface: canonical

cohorts:
  val_headline:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
    sample_count: 500
    seed: 17
  train_supplemental:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
    sample_count: 1500
    seed: 29
        """.strip(),
        encoding="utf-8",
    )

    cfg = load_study_config(config_path)

    assert cfg.run.stages == (
        "audit",
        "pilot",
        "canonical",
        "bad_basin",
        "dense_scene",
        "report",
    )
    assert cfg.models.base.alias == "base"
    assert cfg.models.pure_ce.prompt_surface == "canonical"
    assert cfg.cohorts.val_headline.sample_count == 500
    assert cfg.cohorts.train_supplemental.seed == 29


def test_load_study_config_rejects_unknown_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        """
run:
  name: raw-text-continuity
  output_dir: output/analysis
  stages: [audit, typo_stage]

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    prompt_surface: upper_bound
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/demo-checkpoint
    prompt_surface: canonical

cohorts:
  val_headline:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
    sample_count: 500
    seed: 17
  train_supplemental:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
    sample_count: 1500
    seed: 29
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported stage"):
        load_study_config(config_path)


def test_run_study_materializes_run_dir_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Tokenizer:
        def tokenize(self, text: str) -> list[str]:
            return list(text)

    monkeypatch.setattr(
        continuity_probe_module,
        "_load_tokenizer_for_audit",
        lambda model_path: _Tokenizer(),
    )
    out_dir = tmp_path / "analysis"
    val_jsonl = tmp_path / "val.jsonl"
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl.write_text(
        "\n".join(
            json.dumps({"image_id": idx, "image": f"val_{idx}.jpg"})
            for idx in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    train_jsonl.write_text(
        "\n".join(
            json.dumps({"image_id": idx, "image": f"train_{idx}.jpg"})
            for idx in range(12)
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        f"""
run:
  name: raw-text-continuity-smoke
  output_dir: {out_dir.as_posix()}
  stages: [audit, pilot]

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    prompt_surface: upper_bound
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/demo-checkpoint
    prompt_surface: canonical

cohorts:
  val_headline:
    jsonl_path: {val_jsonl.as_posix()}
    sample_count: 8
    seed: 17
  train_supplemental:
    jsonl_path: {train_jsonl.as_posix()}
    sample_count: 16
    seed: 29
        """.strip(),
        encoding="utf-8",
    )

    result = run_study(config_path)

    run_dir = out_dir / "raw-text-continuity-smoke"
    summary_path = run_dir / "summary.json"
    assert result["run_dir"] == str(run_dir)
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_name"] == "raw-text-continuity-smoke"
    assert summary["stages"] == ["audit", "pilot"]
    assert summary["models"]["base"]["alias"] == "base"
    val_manifest = run_dir / "cohorts" / "val_headline.jsonl"
    train_manifest = run_dir / "cohorts" / "train_supplemental.jsonl"
    assert val_manifest.exists()
    assert train_manifest.exists()
    val_rows = [json.loads(line) for line in val_manifest.read_text(encoding="utf-8").splitlines()]
    train_rows = [json.loads(line) for line in train_manifest.read_text(encoding="utf-8").splitlines()]
    assert [row["image_id"] for row in val_rows] == [7, 3, 0, 5, 1, 9, 2, 4]
    assert len(train_rows) == 12
    assert summary["cohorts"]["val_headline"]["num_rows"] == 8
    assert summary["cohorts"]["train_supplemental"]["num_rows"] == 12
    assert not (run_dir / "report.md").exists()
    assert not (run_dir / "per_coord_scores.jsonl").exists()
    assert not (run_dir / "hard_cases.jsonl").exists()


def test_run_study_resolves_config_relative_cohort_paths_when_cwd_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Tokenizer:
        def tokenize(self, text: str) -> list[str]:
            return list(text)

    monkeypatch.setattr(
        continuity_probe_module,
        "_load_tokenizer_for_audit",
        lambda model_path: _Tokenizer(),
    )
    config_dir = tmp_path / "configs"
    data_dir = config_dir / "data"
    outside_dir = tmp_path / "outside"
    out_dir = tmp_path / "analysis"
    data_dir.mkdir(parents=True)
    outside_dir.mkdir()
    for name, count in (("val", 4), ("train", 6)):
        (data_dir / f"{name}.jsonl").write_text(
            "\n".join(
                json.dumps({"image_id": idx, "image": f"{name}_{idx}.jpg"})
                for idx in range(count)
            )
            + "\n",
            encoding="utf-8",
        )
    config_path = config_dir / "probe.yaml"
    config_path.write_text(
        f"""
run:
  name: cwd-independent
  output_dir: {out_dir.as_posix()}
  stages: [audit]

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    prompt_surface: upper_bound
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/demo-checkpoint
    prompt_surface: canonical

cohorts:
  val_headline:
    jsonl_path: data/val.jsonl
    sample_count: 3
    seed: 17
  train_supplemental:
    jsonl_path: data/train.jsonl
    sample_count: 5
    seed: 29
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(outside_dir)

    result = run_study(config_path)

    run_dir = out_dir / "cwd-independent"
    assert result["run_dir"] == str(run_dir)
    assert (run_dir / "cohorts" / "val_headline.jsonl").exists()
    assert (run_dir / "cohorts" / "train_supplemental.jsonl").exists()


def test_run_phase0_audit_records_requested_numbers() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return [ord(ch) for ch in text]

        def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
            return [chr(tok_id) for tok_id in token_ids]

        def tokenize(self, text: str) -> list[str]:
            return list(text)

    class _Scorer:
        tokenizer = _Tokenizer()

    audit = run_phase0_audit(_Scorer())

    values = [row["value"] for row in audit["numbers"]]
    assert values == [0, 1, 9, 10, 99, 100, 199, 200, 210, 999]
    assert audit["numbers"][3]["tokens"] == ["1", "0"]
    assert audit["surface_forms"]["canonical_label"] == "pretty_inline"
    variants = {
        row["label"]: row for row in audit["surface_forms"]["variants"]
    }
    assert variants["pretty_inline"]["token_count"] > variants["compact"]["token_count"]
    assert variants["compact"]["first_diff_vs_pretty_inline"] == 11


def test_resolve_audit_model_info_uses_base_processor_for_adapter_shorthand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = tmp_path / "base-model"
    adapter_dir = tmp_path / "adapter-checkpoint"
    base_dir.mkdir()
    adapter_dir.mkdir()
    monkeypatch.setattr(
        continuity_probe_module,
        "resolve_inference_checkpoint",
        lambda model_checkpoint: ResolvedInferenceCheckpoint(
            checkpoint_mode="adapter_shorthand",
            requested_model_checkpoint=str(adapter_dir),
            requested_adapter_checkpoint=None,
            resolved_base_model_checkpoint=str(base_dir),
            resolved_adapter_checkpoint=str(adapter_dir),
            adapter_info=AdapterCheckpointInfo(
                path=str(adapter_dir),
                base_model_name_or_path=str(base_dir),
                modules_to_save=("coord_offset_adapter",),
                coord_offset_spec=CoordOffsetAdapterSpec(
                    coord_ids=(1, 2, 3),
                    tie_head=True,
                ),
            ),
        ),
    )

    info = _resolve_audit_model_info(adapter_dir)

    assert info["checkpoint_mode"] == "adapter_shorthand"
    assert info["resolved_base_model_checkpoint"] == str(base_dir)
    assert info["resolved_adapter_checkpoint"] == str(adapter_dir)
    assert info["processor_source"] == str(base_dir)
    assert info["processor_source_is_local"] is True
    assert info["has_coord_offset_adapter"] is True


def test_build_random_cohort_is_deterministic() -> None:
    rows = [{"image_id": idx, "image": f"img_{idx}.jpg"} for idx in range(10)]

    left = build_random_cohort(rows, sample_count=4, seed=17)
    right = build_random_cohort(rows, sample_count=4, seed=17)

    assert [row["image_id"] for row in left] == [row["image_id"] for row in right]


def test_render_pretty_inline_assistant_text_reorders_bbox_first_rows() -> None:
    row = {
        "objects": [
            {
                "bbox_2d": [699, 284, 722, 336],
                "desc": "clock",
                "category_id": 85,
            }
        ]
    }

    assistant_text = render_pretty_inline_assistant_text(
        row,
        object_field_order="desc_first",
    )

    assert assistant_text == (
        '{"objects": [{"desc": "clock", "bbox_2d": [699, 284, 722, 336]}]}'
    )


def test_build_study_hard_cases_prefers_duplicate_prone_rows() -> None:
    rows = [
        {"image_id": 1, "pred_count": 8, "max_desc_count": 3, "same_desc_duplicate_pair_count": 0},
        {"image_id": 2, "pred_count": 17, "max_desc_count": 17, "same_desc_duplicate_pair_count": 9},
    ]

    hard = build_study_hard_cases(rows, max_cases=1)

    assert hard[0]["image_id"] == 2


def test_mine_duplicate_like_rows_prefers_same_desc_local_duplicates(
    tmp_path: Path,
) -> None:
    gt_vs_pred_path = tmp_path / "gt_vs_pred.jsonl"
    gt_vs_pred_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image_id": 1,
                        "image": "duplicate.jpg",
                        "pred": [
                            {"desc": "person", "bbox_2d": [10, 10, 110, 110]},
                            {"desc": "person", "bbox_2d": [12, 12, 112, 112]},
                            {"desc": "dog", "bbox_2d": [300, 300, 360, 380]},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "image_id": 2,
                        "image": "separated.jpg",
                        "pred": [
                            {"desc": "person", "bbox_2d": [10, 10, 110, 110]},
                            {"desc": "person", "bbox_2d": [250, 250, 350, 350]},
                            {"desc": "dog", "bbox_2d": [400, 400, 450, 470]},
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = mine_duplicate_like_rows(
        gt_vs_pred_path=gt_vs_pred_path,
        max_cases=2,
        min_pred_objects=3,
        min_duplicate_pairs=1,
        duplicate_iou_threshold=0.7,
    )

    assert [row["image_id"] for row in rows] == [1]
    assert rows[0]["same_desc_duplicate_pair_count"] == 1
    assert rows[0]["top_desc"] == "person"


def test_mine_duplicate_like_rows_exposes_local_duplicate_knobs(
    tmp_path: Path,
) -> None:
    gt_vs_pred_path = tmp_path / "gt_vs_pred.jsonl"
    gt_vs_pred_path.write_text(
        json.dumps(
            {
                "image_id": 1,
                "image": "local-but-low-iou.jpg",
                "pred": [
                    {"desc": "person", "bbox_2d": [10, 10, 110, 110]},
                    {"desc": "person", "bbox_2d": [80, 80, 180, 180]},
                    {"desc": "dog", "bbox_2d": [300, 300, 360, 380]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    no_fallback_rows = mine_duplicate_like_rows(
        gt_vs_pred_path=gt_vs_pred_path,
        max_cases=1,
        min_pred_objects=3,
        min_duplicate_pairs=1,
        duplicate_iou_threshold=0.7,
        size_ratio_min=1.01,
        local_center_radius_scale=0.0,
    )
    fallback_rows = mine_duplicate_like_rows(
        gt_vs_pred_path=gt_vs_pred_path,
        max_cases=1,
        min_pred_objects=3,
        min_duplicate_pairs=1,
        duplicate_iou_threshold=0.7,
        size_ratio_min=0.75,
        local_center_radius_scale=1.0,
    )

    assert no_fallback_rows == []
    assert [row["image_id"] for row in fallback_rows] == [1]


def test_mine_duplicate_like_rows_preserves_zero_line_idx_in_selection_reason(
    tmp_path: Path,
) -> None:
    gt_vs_pred_path = tmp_path / "gt_vs_pred.jsonl"
    gt_vs_pred_path.write_text(
        json.dumps(
            {
                "image_id": 1,
                "image": "duplicate.jpg",
                "pred": [
                    {"desc": "person", "bbox_2d": [10, 10, 110, 110]},
                    {"desc": "person", "bbox_2d": [12, 12, 112, 112]},
                    {"desc": "dog", "bbox_2d": [300, 300, 360, 380]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = mine_duplicate_like_rows(
        gt_vs_pred_path=gt_vs_pred_path,
        max_cases=1,
        min_pred_objects=3,
        min_duplicate_pairs=1,
        duplicate_iou_threshold=0.7,
    )

    assert rows[0]["line_idx"] == 0
    assert "line_idx=0" in str(rows[0]["selection_reason"])


def test_compute_basin_metrics_uses_gt_center() -> None:
    rows = [
        {"candidate_value": 209, "score": 0.0, "gt_value": 210},
        {"candidate_value": 210, "score": 0.0, "gt_value": 210},
        {"candidate_value": 212, "score": 0.0, "gt_value": 210},
        {"candidate_value": 214, "score": 0.0, "gt_value": 210},
        {"candidate_value": 218, "score": 0.0, "gt_value": 210},
        {"candidate_value": 226, "score": 0.0, "gt_value": 210},
    ]

    metrics = compute_basin_metrics(rows, center_key="gt_value")

    assert metrics["mass_at_1"] == pytest.approx(2.0 / 6.0)
    assert metrics["mass_at_2"] == pytest.approx(3.0 / 6.0)
    assert metrics["mass_at_4"] == pytest.approx(4.0 / 6.0)
    assert metrics["mass_at_8"] == pytest.approx(5.0 / 6.0)
    assert metrics["mass_at_16"] == pytest.approx(1.0)
    assert metrics["mass_at_1"] <= metrics["mass_at_2"] <= metrics["mass_at_4"] <= metrics["mass_at_8"] <= metrics["mass_at_16"]
    assert metrics["local_expected_abs_error"] == pytest.approx(31.0 / 6.0)
    assert metrics["half_height_width"] == pytest.approx(16.0)


def test_compute_vision_lift_rows_pairs_correct_and_swapped() -> None:
    rows = [
        {"case_id": "a", "slot": "x1", "image_condition": "correct", "gt_score": -0.1},
        {"case_id": "a", "slot": "x1", "image_condition": "swapped", "gt_score": -0.9},
        {"case_id": "b", "slot": "y2", "image_condition": "correct", "gt_score": -0.4},
        {"case_id": "b", "slot": "y2", "image_condition": "swapped", "gt_score": -0.6},
        {"case_id": "c", "slot": "x2", "image_condition": "correct", "gt_score": -0.2},
    ]

    lifted = compute_vision_lift_rows(rows)

    assert [row["case_id"] for row in lifted] == ["a", "b"]
    assert [row["slot"] for row in lifted] == ["x1", "y2"]
    assert lifted[0]["vision_lift"] == pytest.approx(0.8)
    assert lifted[1]["vision_lift"] == pytest.approx(0.2)


def test_compute_basin_metrics_rejects_non_finite_scores() -> None:
    rows = [
        {"candidate_value": 199, "score": float("-inf"), "gt_value": 200},
        {"candidate_value": 200, "score": float("-inf"), "gt_value": 200},
    ]

    with pytest.raises(ValueError, match="finite"):
        compute_basin_metrics(rows, center_key="gt_value")


def test_compute_vision_lift_rows_rejects_duplicate_conditions() -> None:
    rows = [
        {"case_id": "a", "slot": "x1", "image_condition": "correct", "gt_score": -0.1},
        {"case_id": "a", "slot": "x1", "image_condition": "correct", "gt_score": -0.4},
        {"case_id": "a", "slot": "x1", "image_condition": "swapped", "gt_score": -0.9},
    ]

    with pytest.raises(ValueError, match="duplicate image_condition"):
        compute_vision_lift_rows(rows)


def test_summarize_wrong_anchor_advantage_prefers_pred_center_when_requested() -> None:
    rows = [
        {"candidate_value": 101, "score": -0.2, "gt_value": 140, "pred_value": 100},
        {"candidate_value": 104, "score": -0.3, "gt_value": 140, "pred_value": 100},
        {"candidate_value": 140, "score": -0.9, "gt_value": 140, "pred_value": 100},
    ]

    summary = summarize_wrong_anchor_advantage(rows)

    assert summary["pred_center_mass_at_4"] > summary["gt_center_mass_at_4"]
    assert summary["wrong_anchor_advantage_at_4"] > 0.0


def test_build_xy_heatmap_grid_preserves_cartesian_order() -> None:
    rows = [
        {"candidate_x1": 10, "candidate_y1": 20, "score": -0.1},
        {"candidate_x1": 10, "candidate_y1": 21, "score": -0.2},
        {"candidate_x1": 11, "candidate_y1": 20, "score": -0.3},
        {"candidate_x1": 11, "candidate_y1": 21, "score": -0.4},
    ]

    heatmap = build_xy_heatmap_grid(rows)

    assert heatmap["x_values"] == [10, 11]
    assert heatmap["y_values"] == [20, 21]
    assert heatmap["z_matrix"] == [[-0.1, -0.3], [-0.2, -0.4]]


def test_build_xy_heatmap_grid_rejects_duplicate_cells() -> None:
    rows = [
        {"candidate_x1": 10, "candidate_y1": 20, "score": -0.1},
        {"candidate_x1": 10, "candidate_y1": 20, "score": -0.9},
    ]

    with pytest.raises(ValueError, match="duplicate heatmap cell"):
        build_xy_heatmap_grid(rows)


def test_build_xy_heatmap_grid_rejects_sparse_grids() -> None:
    rows = [
        {"candidate_x1": 10, "candidate_y1": 20, "score": -0.1},
        {"candidate_x1": 10, "candidate_y1": 21, "score": -0.2},
        {"candidate_x1": 11, "candidate_y1": 20, "score": -0.3},
    ]

    with pytest.raises(ValueError, match="missing heatmap cell"):
        build_xy_heatmap_grid(rows)


def test_build_prefix_perturbation_variants_exposes_expected_labels() -> None:
    variants = build_prefix_perturbation_variants(
        prefix_objects=[{"desc": "person", "bbox_2d": [10, 10, 20, 20]}],
        source_index_in_prefix=0,
        gt_next={"desc": "person", "bbox_2d": [12, 14, 24, 28]},
    )

    assert [label for label, _ in variants] == [
        "drop_source",
        "replace_source_with_gt_next",
        "interp_source_to_gt_next_0p5",
        "source_x1y1_from_gt_next",
    ]


def test_write_report_bundle_materializes_required_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "probe"

    write_report_bundle(
        out_dir=out_dir,
        summary={"questions": {"q1": "inconclusive"}},
        report_md="# Demo\n",
        per_coord_rows=[{"case_id": "a", "slot": "x1", "candidate_value": 100, "score": -0.1}],
        hard_cases=[{"case_id": "hard-1"}],
    )

    assert (out_dir / "report.md").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "per_coord_scores.jsonl").exists()
    assert (out_dir / "hard_cases.jsonl").exists()


def test_run_study_materializes_report_bundle_when_report_stage_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Tokenizer:
        def tokenize(self, text: str) -> list[str]:
            return list(text)

    monkeypatch.setattr(
        continuity_probe_module,
        "_load_tokenizer_for_audit",
        lambda model_path: _Tokenizer(),
    )
    out_dir = tmp_path / "analysis"
    val_jsonl = tmp_path / "val.jsonl"
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl.write_text(json.dumps({"image_id": 1, "image": "val_1.jpg"}) + "\n", encoding="utf-8")
    train_jsonl.write_text(
        json.dumps({"image_id": 2, "image": "train_2.jpg"}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        f"""
run:
  name: report-stage
  output_dir: {out_dir.as_posix()}
  stages: [audit, report]

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    prompt_surface: upper_bound
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/demo-checkpoint
    prompt_surface: canonical

cohorts:
  val_headline:
    jsonl_path: {val_jsonl.as_posix()}
    sample_count: 1
    seed: 17
  train_supplemental:
    jsonl_path: {train_jsonl.as_posix()}
    sample_count: 1
    seed: 29
        """.strip(),
        encoding="utf-8",
    )

    run_study(config_path)

    run_dir = out_dir / "report-stage"
    assert (run_dir / "report.md").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "per_coord_scores.jsonl").exists()
    assert (run_dir / "hard_cases.jsonl").exists()


def test_run_study_materializes_audit_artifacts_for_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Tokenizer:
        def tokenize(self, text: str) -> list[str]:
            return list(text)

    def _fake_load_tokenizer_for_audit(model_path: Path) -> _Tokenizer:
        assert model_path.exists()
        return _Tokenizer()

    monkeypatch.setattr(
        continuity_probe_module,
        "_load_tokenizer_for_audit",
        _fake_load_tokenizer_for_audit,
    )
    out_dir = tmp_path / "analysis"
    base_model = tmp_path / "base-model"
    pure_model = tmp_path / "pure-model"
    base_model.mkdir()
    pure_model.mkdir()
    val_jsonl = tmp_path / "val.jsonl"
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl.write_text(json.dumps({"image_id": 1, "image": "val_1.jpg"}) + "\n", encoding="utf-8")
    train_jsonl.write_text(
        json.dumps({"image_id": 2, "image": "train_2.jpg"}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        f"""
run:
  name: audit-stage
  output_dir: {out_dir.as_posix()}
  stages: [audit]

models:
  base:
    alias: base
    path: {base_model.as_posix()}
    prompt_surface: upper_bound
  pure_ce:
    alias: pure_ce
    path: {pure_model.as_posix()}
    prompt_surface: canonical

cohorts:
  val_headline:
    jsonl_path: {val_jsonl.as_posix()}
    sample_count: 1
    seed: 17
  train_supplemental:
    jsonl_path: {train_jsonl.as_posix()}
    sample_count: 1
    seed: 29
        """.strip(),
        encoding="utf-8",
    )

    run_study(config_path)

    run_dir = out_dir / "audit-stage"
    assert (run_dir / "audit" / "base_tokenization.json").exists()
    assert (run_dir / "audit" / "pure_ce_tokenization.json").exists()
    summary = json.loads((run_dir / "audit" / "summary.json").read_text(encoding="utf-8"))
    assert summary["serialization_surface"] == "pretty_inline"

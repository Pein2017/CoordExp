from __future__ import annotations

from pathlib import Path

import json

import pytest

from src.analysis.duplication_collapse_analysis import mine_duplicate_like_rows
from src.analysis.raw_text_coord_continuity_probe import (
    build_random_cohort,
    build_study_hard_cases,
    load_study_config,
    run_phase0_audit,
    run_study,
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


def test_run_study_materializes_run_dir_and_summary(tmp_path: Path) -> None:
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


def test_run_study_resolves_config_relative_cohort_paths_when_cwd_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        def tokenize(self, text: str) -> list[str]:
            return list(text)

    class _Scorer:
        tokenizer = _Tokenizer()

    audit = run_phase0_audit(_Scorer())

    values = [row["value"] for row in audit["numbers"]]
    assert values == [0, 1, 9, 10, 99, 100, 199, 200, 210, 999]
    assert audit["numbers"][3]["tokens"] == ["1", "0"]


def test_build_random_cohort_is_deterministic() -> None:
    rows = [{"image_id": idx, "image": f"img_{idx}.jpg"} for idx in range(10)]

    left = build_random_cohort(rows, sample_count=4, seed=17)
    right = build_random_cohort(rows, sample_count=4, seed=17)

    assert [row["image_id"] for row in left] == [row["image_id"] for row in right]


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

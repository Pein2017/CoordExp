from __future__ import annotations

from pathlib import Path

import json

import pytest

from src.analysis.raw_text_coord_continuity_probe import (
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
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
    sample_count: 8
    seed: 17
  train_supplemental:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
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

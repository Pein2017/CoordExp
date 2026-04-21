from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.analysis.raw_text_coord_heatmap_probe as heatmap_module
from src.analysis.raw_text_coord_heatmap_probe import (
    load_heatmap_config,
    run_heatmap_probe,
    select_heatmap_cases,
    summarize_heatmap_rows,
)


def test_load_heatmap_config_parses_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "heatmap.yaml"
    config_path.write_text(
        """
run:
  name: heatmap-demo
  output_dir: output/analysis

selection:
  selected_cases_jsonl: output/analysis/demo/selected_cases.jsonl
  max_cases: 3

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/demo-checkpoint

scoring:
  device: cuda:6

heatmap:
  grid_radius: 16
  batch_size: 32
  centers: [pred, gt]
  variants: [baseline, replace_source_with_gt_next]
        """.strip(),
        encoding="utf-8",
    )

    cfg = load_heatmap_config(config_path)

    assert cfg.run.name == "heatmap-demo"
    assert cfg.selection.max_cases == 3
    assert cfg.scoring.device == "cuda:6"
    assert cfg.heatmap.grid_radius == 16
    assert cfg.heatmap.batch_size == 32
    assert cfg.heatmap.centers == ("pred", "gt")


def test_select_heatmap_cases_uses_selection_rank_then_advantage() -> None:
    rows = [
        {"case_id": "b", "selection_rank": 2, "selection_wrong_anchor_advantage_at_4": 0.3},
        {"case_id": "a", "selection_rank": 1, "selection_wrong_anchor_advantage_at_4": 0.1},
        {"case_id": "c", "selection_rank": 3, "selection_wrong_anchor_advantage_at_4": 0.8},
    ]

    selected = select_heatmap_cases(rows, max_cases=2)

    assert [row["case_id"] for row in selected] == ["a", "b"]


def test_summarize_heatmap_rows_counts_grids_and_case_ids() -> None:
    rows = [
        {
            "model_alias": "pure_ce",
            "case_id": "case-a",
            "variant": "baseline",
            "center_kind": "pred",
            "candidate_x1": 100,
            "candidate_y1": 200,
            "score": -1.0,
        },
        {
            "model_alias": "pure_ce",
            "case_id": "case-a",
            "variant": "baseline",
            "center_kind": "pred",
            "candidate_x1": 101,
            "candidate_y1": 200,
            "score": -0.5,
        },
        {
            "model_alias": "base",
            "case_id": "case-b",
            "variant": "replace_source_with_gt_next",
            "center_kind": "gt",
            "candidate_x1": 300,
            "candidate_y1": 400,
            "score": -2.0,
        },
    ]

    summary = summarize_heatmap_rows(rows)

    assert summary["num_heatmap_rows"] == 3
    assert summary["num_grids"] == 2
    assert summary["case_ids"] == ["case-a", "case-b"]


def test_run_heatmap_probe_materializes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "analysis"
    selected_cases_jsonl = tmp_path / "selected_cases.jsonl"
    selected_cases_jsonl.write_text(
        json.dumps({"case_id": "case-a", "selection_rank": 1}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "heatmap.yaml"
    config_path.write_text(
        f"""
run:
  name: heatmap-run
  output_dir: {out_dir.as_posix()}

selection:
  selected_cases_jsonl: {selected_cases_jsonl.as_posix()}
  max_cases: 1

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/demo-checkpoint

scoring:
  device: cuda:6

heatmap:
  grid_radius: 8
  batch_size: 8
  centers: [pred]
  variants: [baseline]
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        heatmap_module,
        "_run_selected_heatmap_cases",
        lambda **kwargs: {
            "per_cell_rows": [
                {
                    "model_alias": "pure_ce",
                    "case_id": "case-a",
                    "variant": "baseline",
                    "center_kind": "pred",
                    "candidate_x1": 100,
                    "candidate_y1": 200,
                    "score": -1.0,
                }
            ],
            "grid_rows": [
                {
                    "model_alias": "pure_ce",
                    "case_id": "case-a",
                    "variant": "baseline",
                    "center_kind": "pred",
                    "grid_json": {"x_values": [100], "y_values": [200], "z_matrix": [[-1.0]]},
                    "figure_path": "figures/case-a.png",
                }
            ],
        },
    )

    result = run_heatmap_probe(config_path)

    run_dir = out_dir / "heatmap-run"
    assert result["run_dir"] == str(run_dir)
    assert (run_dir / "selected_cases.jsonl").exists()
    assert (run_dir / "per_cell_scores.jsonl").exists()
    assert (run_dir / "heatmaps.jsonl").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["selection"]["num_selected_cases"] == 1
    assert summary["results"]["num_grids"] == 1

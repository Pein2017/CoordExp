from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.analysis.raw_text_coord_perturbation_probe as perturbation_module
from src.analysis.raw_text_coord_perturbation_probe import (
    load_perturbation_config,
    run_perturbation_probe,
    select_perturbation_cases,
    summarize_perturbation_case_rows,
)


def test_load_perturbation_config_parses_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "perturb.yaml"
    config_path.write_text(
        """
run:
  name: perturb-demo
  output_dir: output/analysis

selection:
  bad_basin_summary_json: output/analysis/demo/bad_basin/summary.json
  cohort_jsonl: output/analysis/demo/cohort.jsonl
  max_cases: 4
  target_model_alias: pure_ce

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/demo-checkpoint

scoring:
  device: cuda:7
  candidate_radius: 8
        """.strip(),
        encoding="utf-8",
    )

    cfg = load_perturbation_config(config_path)

    assert cfg.run.name == "perturb-demo"
    assert cfg.selection.max_cases == 4
    assert cfg.selection.target_model_alias == "pure_ce"
    assert cfg.scoring.candidate_radius == 8
    assert cfg.models.base.alias == "base"


def test_select_perturbation_cases_prefers_highest_wrong_anchor_case() -> None:
    probe_rows = [
        {
            "model_alias": "pure_ce",
            "case_id": "a",
            "slot": "x1",
            "wrong_anchor_advantage_at_4": 0.2,
            "image_id": 1,
        },
        {
            "model_alias": "pure_ce",
            "case_id": "a",
            "slot": "y1",
            "wrong_anchor_advantage_at_4": 0.1,
            "image_id": 1,
        },
        {
            "model_alias": "pure_ce",
            "case_id": "b",
            "slot": "x1",
            "wrong_anchor_advantage_at_4": 0.5,
            "image_id": 2,
        },
        {
            "model_alias": "pure_ce",
            "case_id": "c",
            "slot": "x1",
            "wrong_anchor_advantage_at_4": -0.2,
            "image_id": 3,
        },
        {
            "model_alias": "base",
            "case_id": "z",
            "slot": "x1",
            "wrong_anchor_advantage_at_4": 0.9,
            "image_id": 9,
        },
    ]

    selected = select_perturbation_cases(
        probe_rows,
        max_cases=2,
        target_model_alias="pure_ce",
        slots=("x1", "y1"),
        require_positive_advantage=True,
    )

    assert [row["case_id"] for row in selected] == ["b", "a"]
    assert selected[0]["selection_wrong_anchor_advantage_at_4"] == pytest.approx(0.5)
    assert selected[1]["selection_wrong_anchor_advantage_at_4"] == pytest.approx(0.2)


def test_prepare_selected_cases_preserves_zero_line_idx_from_case_id() -> None:
    selected = [
        {
            "model_alias": "pure_ce",
            "case_id": "139:0:6->7",
            "image_id": 139,
            "slot": "x1",
            "wrong_anchor_advantage_at_4": 0.25,
            "selection_wrong_anchor_advantage_at_4": 0.25,
        }
    ]
    cohort_rows = [
        {
            "image_id": 139,
            "line_idx": 0,
            "source_gt_vs_pred_jsonl": "/tmp/source.jsonl",
        }
    ]

    prepared = perturbation_module._prepare_selected_cases(
        selection_rows=selected,
        cohort_rows=cohort_rows,
    )

    assert prepared[0]["line_idx"] == 0
    assert prepared[0]["source_object_index"] == 6
    assert prepared[0]["object_index"] == 7


def test_canonicalize_variant_prefix_objects_prefers_points_over_stale_bbox() -> None:
    prefix_objects = [{"desc": "person", "bbox_2d": [100, 200, 300, 400]}]
    gt_next = {"desc": "person", "bbox_2d": [150, 250, 360, 460]}
    variants = dict(
        perturbation_module.build_prefix_perturbation_variants(
            prefix_objects=prefix_objects,
            source_index_in_prefix=0,
            gt_next=gt_next,
        )
    )

    x1y1_prefix = perturbation_module._canonicalize_variant_prefix_objects(
        variants["source_x1y1_from_gt_next"]
    )
    interp_prefix = perturbation_module._canonicalize_variant_prefix_objects(
        variants["interp_source_to_gt_next_0p5"]
    )

    assert x1y1_prefix[0]["bbox_2d"] == [150, 250, 300, 400]
    assert interp_prefix[0]["bbox_2d"] == [125, 225, 330, 430]


def test_summarize_perturbation_case_rows_compares_against_baseline() -> None:
    rows = [
        {
            "model_alias": "pure_ce",
            "case_id": "a",
            "variant": "baseline",
            "slot": "x1",
            "pred_center_mass_at_4": 0.7,
            "gt_center_mass_at_4": 0.4,
            "wrong_anchor_advantage_at_4": 0.3,
        },
        {
            "model_alias": "pure_ce",
            "case_id": "a",
            "variant": "replace_source_with_gt_next",
            "slot": "x1",
            "pred_center_mass_at_4": 0.5,
            "gt_center_mass_at_4": 0.6,
            "wrong_anchor_advantage_at_4": -0.1,
        },
        {
            "model_alias": "pure_ce",
            "case_id": "b",
            "variant": "baseline",
            "slot": "x1",
            "pred_center_mass_at_4": 0.8,
            "gt_center_mass_at_4": 0.2,
            "wrong_anchor_advantage_at_4": 0.6,
        },
        {
            "model_alias": "pure_ce",
            "case_id": "b",
            "variant": "replace_source_with_gt_next",
            "slot": "x1",
            "pred_center_mass_at_4": 0.4,
            "gt_center_mass_at_4": 0.5,
            "wrong_anchor_advantage_at_4": -0.1,
        },
    ]

    summary = summarize_perturbation_case_rows(rows)

    assert summary["num_summary_rows"] == 4
    assert summary["variant_metrics"][0]["variant"] == "replace_source_with_gt_next"
    assert summary["variant_metrics"][0]["avg_delta_wrong_anchor_advantage_at_4"] == pytest.approx(-0.55)
    assert summary["variant_metrics"][0]["avg_delta_pred_center_mass_at_4"] == pytest.approx(-0.3)
    assert summary["variant_metrics"][0]["avg_delta_gt_center_mass_at_4"] == pytest.approx(0.25)


def test_run_perturbation_probe_materializes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "analysis"
    bad_basin_summary_path = tmp_path / "bad_basin_summary.json"
    cohort_jsonl = tmp_path / "cohort.jsonl"
    bad_basin_summary_path.write_text(
        json.dumps(
            {
                "models": {
                    "pure_ce": {
                        "probe_metrics": [
                            {
                                "model_alias": "pure_ce",
                                "case_id": "case-a",
                                "slot": "x1",
                                "wrong_anchor_advantage_at_4": 0.5,
                                "image_id": 1,
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cohort_jsonl.write_text(
        json.dumps(
            {
                "image_id": 1,
                "line_idx": 0,
                "source_gt_vs_pred_jsonl": str(tmp_path / "source.jsonl"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "perturb.yaml"
    config_path.write_text(
        f"""
run:
  name: perturb-run
  output_dir: {out_dir.as_posix()}

selection:
  bad_basin_summary_json: {bad_basin_summary_path.as_posix()}
  cohort_jsonl: {cohort_jsonl.as_posix()}
  max_cases: 1
  target_model_alias: pure_ce

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/demo-checkpoint

scoring:
  device: cuda:7
  candidate_radius: 8
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        perturbation_module,
        "_run_selected_perturbation_cases",
        lambda **kwargs: {
            "per_coord_rows": [
                {
                    "model_alias": "pure_ce",
                    "case_id": "case-a",
                    "variant": "baseline",
                    "slot": "x1",
                    "candidate_value": 100,
                    "score": -0.1,
                    "pred_value": 100,
                    "gt_value": 110,
                }
            ],
            "summary_rows": [
                {
                    "model_alias": "pure_ce",
                    "case_id": "case-a",
                    "variant": "baseline",
                    "slot": "x1",
                    "pred_center_mass_at_4": 0.7,
                    "gt_center_mass_at_4": 0.4,
                    "wrong_anchor_advantage_at_4": 0.3,
                }
            ],
        },
    )

    result = run_perturbation_probe(config_path)

    run_dir = out_dir / "perturb-run"
    assert result["run_dir"] == str(run_dir)
    assert (run_dir / "selected_cases.jsonl").exists()
    assert (run_dir / "per_coord_scores.jsonl").exists()
    assert (run_dir / "summary_rows.jsonl").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["selection"]["num_selected_cases"] == 1
    assert summary["results"]["num_summary_rows"] == 1

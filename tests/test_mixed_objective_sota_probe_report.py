from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from src.analysis.mixed_objective_sota_probe_report import build_report_summary


def test_build_report_summary_collects_eval_and_recall_inputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    summary = build_report_summary(
        target_alias="mixed_objective_sota_adapter",
        eval_summary={"bbox_AP": 0.39},
        recall_summary={"baseline_recall_loc": 0.50, "oracle_k_recall_loc": 0.62},
        basin_summary={"gt_mass_at_4": 0.71},
    )

    assert summary["target_alias"] == "mixed_objective_sota_adapter"
    assert summary["eval"]["bbox_AP"] == 0.39
    assert summary["recall"]["oracle_k_recall_loc"] == 0.62


def test_cli_writes_summary_json_and_report_md(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "report.yaml"
    output_dir = tmp_path / "output"
    base_metadata = tmp_path / "base.yaml"
    contract_summary = tmp_path / "contract.json"
    eval_summary = tmp_path / "eval.json"
    recall_summary = tmp_path / "recall.json"
    basin_summary = tmp_path / "basin.json"
    base_metadata.write_text(
        yaml.safe_dump(
            {
                "target_family": {
                    "alias": "mixed_objective_sota_adapter",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    contract_summary.write_text(json.dumps({"family_count": 1}), encoding="utf-8")
    eval_summary.write_text(json.dumps({"bbox_AP": 0.39}), encoding="utf-8")
    recall_summary.write_text(json.dumps({"oracle_k_recall_loc": 0.62}), encoding="utf-8")
    basin_summary.write_text(json.dumps({"gt_mass_at_4": 0.71}), encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": "mixed-objective-sota-probe-smoke",
                    "output_dir": str(output_dir),
                },
                "inputs": {
                    "base_metadata_yaml": str(base_metadata),
                    "contract_summary_json": str(contract_summary),
                    "eval_bundle_summary_json": str(eval_summary),
                    "recall_summary_json": str(recall_summary),
                    "basin_summary_json": str(basin_summary),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analysis"
        / "build_mixed_objective_sota_probe_report.py"
    )
    spec = importlib.util.spec_from_file_location("mixed_objective_sota_probe_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module.sys,
        "argv",
        ["build_mixed_objective_sota_probe_report.py", "--config", str(config_path)],
    )
    module.main()

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    assert summary_path.exists()
    assert report_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["target_alias"] == "mixed_objective_sota_adapter"
    assert summary["contract"]["family_count"] == 1
    assert summary["basin"]["gt_mass_at_4"] == 0.71
    assert report_path.read_text(encoding="utf-8").startswith("# Mixed-Objective SOTA Probe")

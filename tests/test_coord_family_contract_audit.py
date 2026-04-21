from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.analysis.coord_family_contract_audit import (
    FamilySpec,
    build_family_inventory,
    infer_checkpoint_runtime_mode,
    load_contract_audit_config,
    run_contract_audit,
)


def test_infer_checkpoint_runtime_mode_prefers_adapter_when_adapter_config_exists(
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "adapter_ckpt"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "model_cache/models/Qwen/Qwen3-VL-2B-Instruct"}),
        encoding="utf-8",
    )

    mode = infer_checkpoint_runtime_mode(ckpt)

    assert mode == "adapter"


def test_build_family_inventory_records_required_contract_fields(tmp_path: Path) -> None:
    merged = tmp_path / "merged_ckpt"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")

    adapter = tmp_path / "adapter_ckpt"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "model_cache/models/Qwen/Qwen3-VL-2B-Instruct"}),
        encoding="utf-8",
    )

    specs = [
        FamilySpec(
            alias="base_xyxy_merged",
            checkpoint_path=str(merged),
            checkpoint_hint="merged",
            infer_mode="coord",
            bbox_format="xyxy",
            pred_coord_mode="pixel",
            eval_compatibility_path="confidence_postop",
            is_headline_2b_family=True,
        ),
        FamilySpec(
            alias="raw_text_xyxy_pure_ce",
            checkpoint_path=str(adapter),
            checkpoint_hint="adapter",
            infer_mode="coord",
            bbox_format="xyxy",
            pred_coord_mode="norm1000",
            eval_compatibility_path="confidence_postop",
            is_headline_2b_family=True,
        ),
        FamilySpec(
            alias="mixed_objective_sota_adapter",
            checkpoint_path=str(adapter),
            checkpoint_hint="adapter",
            infer_mode="coord",
            bbox_format="xyxy",
            pred_coord_mode="auto",
            eval_compatibility_path="confidence_postop",
            is_headline_2b_family=False,
        ),
    ]

    rows = build_family_inventory(specs)

    assert rows[0]["alias"] == "base_xyxy_merged"
    assert rows[0]["checkpoint_type"] == "merged"
    assert rows[0]["runtime_load_pattern"] == "model_checkpoint only"
    assert rows[0]["bbox_format"] == "xyxy"
    assert rows[0]["pred_coord_mode"] == "pixel"
    assert rows[0]["eval_compatibility_path"] == "confidence_postop"
    assert rows[0]["is_headline_2b_family"] is True
    assert rows[1]["alias"] == "raw_text_xyxy_pure_ce"
    assert rows[1]["checkpoint_type"] == "adapter"
    assert rows[1]["runtime_load_pattern"] == "model_checkpoint + adapter_checkpoint"
    assert rows[1]["resolved_base_model_checkpoint"] == "model_cache/models/Qwen/Qwen3-VL-2B-Instruct"
    assert rows[1]["resolved_adapter_checkpoint"] == str(adapter)
    assert rows[2]["alias"] == "mixed_objective_sota_adapter"
    assert rows[2]["checkpoint_type"] == "adapter"
    assert rows[2]["pred_coord_mode"] == "auto"
    assert rows[2]["runtime_load_pattern"] == "model_checkpoint + adapter_checkpoint"


def test_load_contract_audit_config_parses_run_and_families(tmp_path: Path) -> None:
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(
        """
run:
  name: coord-family-smoke
  output_dir: output/analysis

families:
  - alias: base_xyxy_merged
    checkpoint_path: output/stage1_2b/base
    checkpoint_hint: merged
    infer_mode: coord
    bbox_format: xyxy
    pred_coord_mode: pixel
    eval_compatibility_path: confidence_postop
    is_headline_2b_family: true
        """.strip(),
        encoding="utf-8",
    )

    cfg = load_contract_audit_config(config_path)

    assert cfg.run.name == "coord-family-smoke"
    assert cfg.run.output_dir == "output/analysis"
    assert cfg.families[0].alias == "base_xyxy_merged"
    assert cfg.families[0].checkpoint_path == "output/stage1_2b/base"
    assert cfg.families[0].pred_coord_mode == "pixel"
    assert cfg.families[0].eval_compatibility_path == "confidence_postop"
    assert cfg.families[0].is_headline_2b_family is True


def test_load_contract_audit_config_rejects_invalid_pred_coord_mode(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(
        """
run:
  name: coord-family-smoke
  output_dir: output/analysis

families:
  - alias: base_xyxy_merged
    checkpoint_path: output/stage1_2b/base
    checkpoint_hint: merged
    infer_mode: coord
    bbox_format: xyxy
    pred_coord_mode: nonsense
    eval_compatibility_path: confidence_postop
    is_headline_2b_family: true
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pred_coord_mode"):
        load_contract_audit_config(config_path)


def test_run_contract_audit_materializes_inventory_bundle(tmp_path: Path) -> None:
    merged = tmp_path / "merged_ckpt"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")

    config_path = tmp_path / "audit.yaml"
    output_dir = tmp_path / "analysis"
    config_path.write_text(
        f"""
run:
  name: coord-family-smoke
  output_dir: {output_dir.as_posix()}

families:
  - alias: base_xyxy_merged
    checkpoint_path: {merged.as_posix()}
    checkpoint_hint: merged
    infer_mode: coord
    bbox_format: xyxy
    pred_coord_mode: pixel
    eval_compatibility_path: confidence_postop
    is_headline_2b_family: true
        """.strip(),
        encoding="utf-8",
    )

    result = run_contract_audit(config_path)

    run_dir = output_dir / "coord-family-smoke"
    inventory_path = run_dir / "family_inventory.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "family_contract_audit.md"
    assert result["run_dir"] == str(run_dir)
    assert inventory_path.exists()
    assert summary_path.exists()
    assert report_path.exists()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_text = report_path.read_text(encoding="utf-8")
    assert inventory[0]["alias"] == "base_xyxy_merged"
    assert inventory[0]["runtime_load_pattern"] == "model_checkpoint only"
    assert inventory[0]["pred_coord_mode"] == "pixel"
    assert inventory[0]["eval_compatibility_path"] == "confidence_postop"
    assert inventory[0]["is_headline_2b_family"] is True
    assert summary["run_name"] == "coord-family-smoke"
    assert summary["family_count"] == 1
    assert summary["families"][0]["alias"] == "base_xyxy_merged"
    assert "| Checkpoint Path |" in report_text
    assert str(merged) in report_text


def test_run_contract_audit_resolves_repo_relative_checkpoint_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    merged = tmp_path / "output/stage1_2b/base_xyxy_merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(
        """
run:
  name: coord-family-smoke
  output_dir: output/analysis

families:
  - alias: base_xyxy_merged
    checkpoint_path: output/stage1_2b/base_xyxy_merged
    checkpoint_hint: merged
    infer_mode: coord
    bbox_format: xyxy
    pred_coord_mode: pixel
    eval_compatibility_path: confidence_postop
    is_headline_2b_family: true
        """.strip(),
        encoding="utf-8",
    )
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    result = run_contract_audit(config_path, repo_root=tmp_path)

    summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    assert summary["families"][0]["checkpoint_exists"] is True
    assert summary["families"][0]["checkpoint_path"] == "output/stage1_2b/base_xyxy_merged"


def test_run_contract_audit_prefers_workspace_root_for_worktree_relative_checkpoints(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / ".worktrees" / "feature"
    repo_root.mkdir(parents=True)
    merged = workspace_root / "output/stage1_2b/base_xyxy_merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    config_path = repo_root / "audit.yaml"
    config_path.write_text(
        """
run:
  name: coord-family-smoke
  output_dir: output/analysis

families:
  - alias: base_xyxy_merged
    checkpoint_path: output/stage1_2b/base_xyxy_merged
    checkpoint_hint: merged
    infer_mode: coord
    bbox_format: xyxy
    pred_coord_mode: pixel
    eval_compatibility_path: confidence_postop
    is_headline_2b_family: true
        """.strip(),
        encoding="utf-8",
    )

    result = run_contract_audit(config_path, repo_root=repo_root)

    expected_run_dir = workspace_root / "output/analysis/coord-family-smoke"
    summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    assert result["run_dir"] == str(expected_run_dir)
    assert summary["families"][0]["checkpoint_exists"] is True
    assert summary["families"][0]["resolved_checkpoint_path"] == str(merged)


def test_bundled_configs_track_headline_2b_families() -> None:
    worktree_root = Path(__file__).resolve().parents[1]
    base_cfg = yaml.safe_load(
        (worktree_root / "configs/analysis/coord_family_comparison/base.yaml").read_text(
            encoding="utf-8"
        )
    )
    smoke_cfg = yaml.safe_load(
        (
            worktree_root
            / "configs/analysis/coord_family_comparison/smoke_inventory.yaml"
        ).read_text(encoding="utf-8")
    )

    base_aliases = [row["alias"] for row in base_cfg["families"]]
    assert base_aliases == [
        "base_xyxy_merged",
        "raw_text_xyxy_pure_ce",
        "cxcywh_pure_ce",
        "cxcy_logw_logh_pure_ce",
        "center_parameterization",
        "hard_soft_ce_2b",
    ]
    raw_text_row = next(row for row in base_cfg["families"] if row["alias"] == "raw_text_xyxy_pure_ce")
    raw_text_smoke_row = next(
        row for row in smoke_cfg["families"] if row["alias"] == "raw_text_xyxy_pure_ce"
    )
    assert raw_text_row["infer_mode"] == "text"
    assert raw_text_smoke_row["infer_mode"] == "text"
    assert raw_text_row["checkpoint_hint"] == "adapter"
    assert raw_text_smoke_row["checkpoint_hint"] == "adapter"
    assert raw_text_row["pred_coord_mode"] == "norm1000"
    assert raw_text_smoke_row["pred_coord_mode"] == "norm1000"

    smoke_paths = [row["checkpoint_path"] for row in smoke_cfg["families"]]
    assert all("some_adapter_checkpoint" not in path for path in smoke_paths)
    assert all("smoke_adapter_checkpoint" not in path for path in smoke_paths)


def test_mixed_objective_probe_configs_track_adapter_checkpoint() -> None:
    worktree_root = Path(__file__).resolve().parents[1]
    base_cfg = yaml.safe_load(
        (
            worktree_root / "configs/analysis/mixed_objective_sota_probe/base.yaml"
        ).read_text(encoding="utf-8")
    )
    eval_cfg = yaml.safe_load(
        (
            worktree_root / "configs/analysis/mixed_objective_sota_probe/eval_val200.yaml"
        ).read_text(encoding="utf-8")
    )
    audit_cfg = yaml.safe_load(
        (
            worktree_root / "configs/analysis/mixed_objective_sota_probe/contract_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    recall_cfg = yaml.safe_load(
        (
            worktree_root
            / "configs/analysis/mixed_objective_sota_probe/recall_unmatched_val64.yaml"
        ).read_text(encoding="utf-8")
    )

    family = base_cfg["target_family"]
    assert family["alias"] == "mixed_objective_sota_adapter"
    assert family["checkpoint_hint"] == "adapter"
    assert family["checkpoint_path"].endswith("checkpoint-1332")

    assert eval_cfg["infer"]["model_checkpoint"].endswith("checkpoint-1332")
    assert eval_cfg["infer"]["pred_coord_mode"] == "auto"
    assert eval_cfg["infer"]["limit"] == 200

    audit_family = audit_cfg["families"][0]
    assert audit_family["alias"] == "mixed_objective_sota_adapter"
    assert audit_family["checkpoint_path"].endswith("checkpoint-1332")
    assert audit_cfg["defaults"]["checkpoint_hint"] == "adapter"
    assert audit_cfg["defaults"]["pred_coord_mode"] == "auto"

    checkpoint = recall_cfg["checkpoints"][0]
    assert checkpoint["name"] == "mixed-objective-sota-adapter"
    assert checkpoint["path"].endswith("checkpoint-1332")
    assert checkpoint["checkpoint_hint"] == "adapter"
    assert checkpoint["pred_coord_mode"] == "auto"

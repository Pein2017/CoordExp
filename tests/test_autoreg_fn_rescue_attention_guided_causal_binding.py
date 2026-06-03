from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.analysis.autoreg_fn_rescue_attention_guided_causal_binding import (
    PHASE3_STAGES,
    build_phase3_dry_run_plan,
    load_phase3_config,
    materialize_competitor_source,
    materialize_case_linked_table,
    materialize_sink_triage,
    select_intervention_lane_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTMASTER_PATH = REPO_ROOT / "scripts/analysis/launch_autoreg_fn_rescue_attention_guided_causal_binding_tmux.sh"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_config(path: Path, *, artifact_root: Path, phase2_root: Path, fn_root: Path) -> None:
    path.write_text(
        f"""
paths:
  artifact_root: {artifact_root}
  phase2_root: {phase2_root}
  fn_rescue_root: {fn_root}
evidence_scope: unit_phase3
execution:
  fn_rescue_config_path: {path.parent / "fn_rescue.yaml"}
  target_mask_max_cases: 2
  competitor_source_max_cases: 2
case_linked:
  max_attention_rows: null
  roles: [pre_y1]
  aggregation_scopes: [union]
  region_kinds:
    - target_gt
    - same_desc_competitor_gt_object
    - wrong_control_source_region
    - far_background
""".lstrip(),
        encoding="utf-8",
    )


def _plan_rows() -> list[dict[str, object]]:
    return [
        {
            "case_id": "case-a",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "no_op_control",
        },
        {
            "case_id": "case-a",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "target_gt_mask",
            "region_kind": "target_gt",
            "bbox_xyxy": [10, 10, 40, 40],
        },
        {
            "case_id": "case-a",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "same_desc_competitor_mask",
            "region_kind": "same_desc_competitor_gt_object",
            "region_instance_id": "gt:2",
            "bbox_xyxy": [100, 100, 140, 140],
        },
        {
            "case_id": "case-a",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "same_desc_competitor_mask",
            "region_kind": "same_desc_competitor_gt_object",
            "region_instance_id": "gt:3",
            "bbox_xyxy": [160, 160, 190, 190],
        },
        {
            "case_id": "case-a",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "same_desc_rollout_prediction_mask",
            "region_kind": "same_desc_rollout_prediction",
            "bbox_xyxy": [200, 200, 240, 240],
        },
        {
            "case_id": "case-b",
            "rescue_tier": "desc_x1_wrong_control",
            "selection_kind": "desc_x1_wrong_control_failure",
            "intervention_kind": "no_op_control",
        },
        {
            "case_id": "case-b",
            "rescue_tier": "desc_x1_wrong_control",
            "selection_kind": "desc_x1_wrong_control_failure",
            "intervention_kind": "wrong_control_source_region_mask",
            "region_kind": "wrong_control_source_region",
            "bbox_xyxy": [300, 300, 340, 340],
        },
    ]


def _valid_target_summary(*, root: Path) -> None:
    path = root / "target_mask" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "row_count": 4,
                "case_count": 2,
                "kind_summaries": {
                    "no_op_control": {
                        "rows": 2,
                        "exact_tail_match_baseline": 2,
                        "invalid_parse": 0,
                        "mean_target_iou_delta": 0.0,
                        "primary_success_changed": 0,
                    },
                    "target_gt_mask": {
                        "rows": 2,
                        "exact_tail_match_baseline": 1,
                        "invalid_parse": 0,
                        "mean_target_iou_delta": -0.2,
                        "primary_success_changed": 0,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _case_linked_intervention_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    no_op = {
        "case_id": "case-a",
        "rescue_tier": "desc_x1",
        "intervention_kind": "no_op_control",
        "target_iou_delta": 0.0,
        "primary_rescue_success_changed": False,
        "primary_rescue_success": True,
        "parse_status": "ok",
        "exact_tail_match_baseline": True,
        "generated_box_xyxy": [10, 10, 40, 40],
        "generated_coord_tokens": [10, 10, 40, 40],
    }
    target_mask = {
        "case_id": "case-a",
        "rescue_tier": "desc_x1",
        "intervention_kind": "target_gt_mask",
        "target_iou_delta": -0.5,
        "primary_rescue_success_changed": True,
        "primary_rescue_success": False,
        "parse_status": "ok",
        "exact_tail_match_baseline": False,
        "generated_box_xyxy": [12, 12, 38, 38],
        "generated_coord_tokens": [12, 12, 38, 38],
    }
    competitor = {
        "case_id": "case-a",
        "rescue_tier": "desc_x1",
        "intervention_kind": "same_desc_competitor_mask",
        "target_iou_delta": 0.2,
        "primary_rescue_success_changed": False,
        "primary_rescue_success": True,
        "parse_status": "ok",
        "exact_tail_match_baseline": False,
        "generated_box_xyxy": [11, 10, 41, 40],
        "generated_coord_tokens": [11, 10, 41, 40],
    }
    return [no_op, target_mask], [competitor]


def test_load_phase3_config_and_dry_run_plan(tmp_path: Path) -> None:
    phase2_root = tmp_path / "phase2"
    fn_root = tmp_path / "fn"
    _write_jsonl(phase2_root / "intervention_plan" / "selected_interventions.jsonl", [])
    _write_jsonl(fn_root / "rescue_generation_rows.jsonl", [])
    _write_jsonl(fn_root / "rescue_attention_region_rows.jsonl", [])
    config_path = tmp_path / "phase3.yaml"
    _write_config(config_path, artifact_root=tmp_path / "out", phase2_root=phase2_root, fn_root=fn_root)

    config = load_phase3_config(config_path)
    plan = build_phase3_dry_run_plan(config, stages=("target_mask", "sink_triage", "case_linked"))

    assert config.paths.artifact_root == tmp_path / "out"
    assert config.evidence_scope == "unit_phase3"
    assert "sink_triage" in PHASE3_STAGES
    assert plan["selected_interventions_exists"] is True
    assert plan["generation_rows_exists"] is True
    assert plan["target_mask_max_cases"] == 2


def test_load_phase3_config_resolves_relative_fn_rescue_config_path(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "analysis" / "autoreg_fn_rescue_attention_guided_causal_binding"
    continuation = config_dir.parent / "autoreg_fn_rescue_continuation" / "linked.yaml"
    continuation.parent.mkdir(parents=True, exist_ok=True)
    continuation.write_text("paths: {}\n", encoding="utf-8")
    phase2_root = tmp_path / "phase2"
    fn_root = tmp_path / "fn"
    config_path = config_dir / "phase3.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
paths:
  artifact_root: {tmp_path / "out"}
  phase2_root: {phase2_root}
  fn_rescue_root: {fn_root}
evidence_scope: unit_phase3
execution:
  fn_rescue_config_path: ../autoreg_fn_rescue_continuation/linked.yaml
  target_mask_max_cases: 2
  competitor_source_max_cases: 2
""".lstrip(),
        encoding="utf-8",
    )

    config = load_phase3_config(config_path)

    assert config.execution.fn_rescue_config_path == continuation.resolve(strict=False)


def test_select_intervention_lane_rows_supports_target_and_competitor_lanes() -> None:
    target_rows = select_intervention_lane_rows(
        _plan_rows(),
        selection_kinds=("desc_x1_success_same_desc_competitor",),
        intervention_kinds_by_selection={
            "desc_x1_success_same_desc_competitor": ("no_op_control", "target_gt_mask"),
        },
        max_cases=1,
    )
    competitor_rows = select_intervention_lane_rows(
        _plan_rows(),
        selection_kinds=("desc_x1_success_same_desc_competitor", "desc_x1_wrong_control_failure"),
        intervention_kinds_by_selection={
            "desc_x1_success_same_desc_competitor": (
                "no_op_control",
                "same_desc_competitor_mask",
                "same_desc_rollout_prediction_mask",
            ),
            "desc_x1_wrong_control_failure": (
                "no_op_control",
                "wrong_control_source_region_mask",
            ),
        },
        max_cases=2,
    )

    assert [(row["case_id"], row["intervention_kind"]) for row in target_rows] == [
        ("case-a", "no_op_control"),
        ("case-a", "target_gt_mask"),
    ]
    assert [(row["case_id"], row["intervention_kind"]) for row in competitor_rows] == [
        ("case-a", "no_op_control"),
        ("case-a", "same_desc_competitor_mask"),
        ("case-a", "same_desc_competitor_mask"),
        ("case-a", "same_desc_rollout_prediction_mask"),
        ("case-b", "no_op_control"),
        ("case-b", "wrong_control_source_region_mask"),
    ]


def test_select_intervention_lane_rows_limits_each_selection_kind_independently() -> None:
    rows = _plan_rows() + [
        {
            "case_id": "case-extra",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "no_op_control",
        },
        {
            "case_id": "case-extra",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "same_desc_competitor_mask",
            "region_kind": "same_desc_competitor_gt_object",
            "bbox_xyxy": [10, 10, 20, 20],
        },
    ]

    selected = select_intervention_lane_rows(
        rows,
        selection_kinds=("desc_x1_success_same_desc_competitor", "desc_x1_wrong_control_failure"),
        intervention_kinds_by_selection={
            "desc_x1_success_same_desc_competitor": ("no_op_control", "same_desc_competitor_mask"),
            "desc_x1_wrong_control_failure": ("no_op_control", "wrong_control_source_region_mask"),
        },
        max_cases=1,
    )

    assert {row["case_id"] for row in selected} == {"case-a", "case-b"}


def test_materialize_case_linked_table_joins_intervention_and_attention(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase2_root = tmp_path / "phase2"
    fn_root = tmp_path / "fn"
    target_rows, competitor_rows = _case_linked_intervention_rows()
    _write_jsonl(artifact_root / "target_mask" / "intervention_rows.jsonl", target_rows)
    _write_jsonl(artifact_root / "competitor_source" / "intervention_rows.jsonl", competitor_rows)
    _write_jsonl(
        fn_root / "rescue_attention_region_rows.jsonl",
        [
            {
                "case_id": "case-a",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 13,
                "head": 6,
                "aggregation_scope": "union",
                "region_kind": "target_gt",
                "attention_mass_normalized": 0.7,
            },
            {
                "case_id": "case-a",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 13,
                "head": 6,
                "aggregation_scope": "union",
                "region_kind": "same_desc_competitor_gt_object",
                "attention_mass_normalized": 0.4,
            },
            {
                "case_id": "case-a",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 13,
                "head": 6,
                "aggregation_scope": "instance",
                "region_kind": "target_gt",
                "attention_mass_normalized": 0.99,
            },
        ],
    )
    _write_jsonl(phase2_root / "intervention_plan" / "selected_interventions.jsonl", _plan_rows())
    config_path = tmp_path / "phase3.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase2_root=phase2_root, fn_root=fn_root)
    config = load_phase3_config(config_path)

    summary = materialize_case_linked_table(config)

    rows = [
        json.loads(line)
        for line in (artifact_root / "case_linked" / "case_mechanism_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["row_count"] == 3
    assert rows[0]["target_attention_mass"] == 0.7
    assert rows[0]["competitor_attention_mass"] == 0.4
    assert rows[0]["target_minus_competitor_attention"] == pytest.approx(0.3)
    assert rows[1]["mechanism_bucket"] == "target_dependent"
    assert rows[1]["valid_paired_row"] is True
    assert rows[1]["paired_no_op_valid"] is True
    assert rows[1]["intervention_generated_box_xyxy"] == [12, 12, 38, 38]
    assert rows[1]["primary_success_changed"] is True
    assert rows[1]["top_attention_heads_for_case"] == [{"aggregation_scope": "union", "head": 6, "layer": 13, "role": "pre_y1"}]
    assert rows[1]["x1_logit_lens_rank_when_available"] is None


def test_materialize_case_linked_table_requires_intervention_outputs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase2_root = tmp_path / "phase2"
    fn_root = tmp_path / "fn"
    _write_jsonl(phase2_root / "intervention_plan" / "selected_interventions.jsonl", _plan_rows())
    _write_jsonl(fn_root / "rescue_attention_region_rows.jsonl", [])
    config_path = tmp_path / "phase3.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase2_root=phase2_root, fn_root=fn_root)

    with pytest.raises(FileNotFoundError, match="target_mask"):
        materialize_case_linked_table(load_phase3_config(config_path))


def test_case_linked_masks_are_uninterpretable_without_valid_paired_no_op(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase2_root = tmp_path / "phase2"
    fn_root = tmp_path / "fn"
    target_rows, competitor_rows = _case_linked_intervention_rows()
    target_rows[0]["exact_tail_match_baseline"] = False
    _write_jsonl(artifact_root / "target_mask" / "intervention_rows.jsonl", target_rows)
    _write_jsonl(artifact_root / "competitor_source" / "intervention_rows.jsonl", competitor_rows)
    _write_jsonl(phase2_root / "intervention_plan" / "selected_interventions.jsonl", _plan_rows())
    _write_jsonl(fn_root / "rescue_attention_region_rows.jsonl", [])
    config_path = tmp_path / "phase3.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase2_root=phase2_root, fn_root=fn_root)

    materialize_case_linked_table(load_phase3_config(config_path))

    rows = [
        json.loads(line)
        for line in (artifact_root / "case_linked" / "case_mechanism_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    masked_rows = [row for row in rows if row["intervention_kind"] != "no_op_control"]
    assert {row["valid_paired_row"] for row in masked_rows} == {False}
    assert {row["mechanism_bucket"] for row in masked_rows} == {"invalid_or_uninterpretable"}


def test_competitor_source_requires_target_mask_gate(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase2_root = tmp_path / "phase2"
    fn_root = tmp_path / "fn"
    _write_jsonl(phase2_root / "intervention_plan" / "selected_interventions.jsonl", _plan_rows())
    _write_jsonl(fn_root / "rescue_generation_rows.jsonl", [])
    config_path = tmp_path / "phase3.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase2_root=phase2_root, fn_root=fn_root)

    with pytest.raises(ValueError, match="target_mask gate"):
        materialize_competitor_source(load_phase3_config(config_path))


def test_competitor_source_accepts_valid_target_mask_gate_without_executing_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "out"
    phase2_root = tmp_path / "phase2"
    fn_root = tmp_path / "fn"
    _write_jsonl(phase2_root / "intervention_plan" / "selected_interventions.jsonl", _plan_rows())
    _write_jsonl(fn_root / "rescue_generation_rows.jsonl", [])
    _valid_target_summary(root=artifact_root)
    config_path = tmp_path / "phase3.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase2_root=phase2_root, fn_root=fn_root)

    import src.analysis.autoreg_fn_rescue_attention_guided_causal_binding as phase3

    captured: dict[str, object] = {}

    def fake_execute(*args: object, **kwargs: object) -> dict[str, object]:
        captured["output_subdir"] = kwargs["output_subdir"]
        captured["selected_rows"] = list(kwargs["selected_rows"])  # type: ignore[index]
        return {"stage": kwargs["output_subdir"], "row_count": len(captured["selected_rows"])}

    monkeypatch.setattr(phase3, "execute_intervention_rows", fake_execute)

    summary = materialize_competitor_source(load_phase3_config(config_path))

    assert summary["stage"] == "competitor_source"
    assert captured["output_subdir"] == "competitor_source"


def test_materialize_sink_triage_writes_candidate_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase2_root = tmp_path / "phase2"
    fn_root = tmp_path / "fn"
    _write_jsonl(phase2_root / "intervention_plan" / "selected_interventions.jsonl", _plan_rows())
    _write_jsonl(
        fn_root / "rescue_attention_region_rows.jsonl",
        [
            {
                "case_id": "case-a",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 13,
                "head": 6,
                "aggregation_scope": "union",
                "region_kind": "far_background",
                "attention_mass_normalized": 0.9,
            },
            {
                "case_id": "case-b",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 13,
                "head": 7,
                "aggregation_scope": "union",
                "region_kind": "far_background",
                "attention_mass_normalized": 0.95,
            },
            {
                "case_id": "case-a",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 13,
                "head": 6,
                "aggregation_scope": "union",
                "region_kind": "target_gt",
                "attention_mass_normalized": 0.2,
            },
        ],
    )
    config_path = tmp_path / "phase3.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase2_root=phase2_root, fn_root=fn_root)

    summary = materialize_sink_triage(load_phase3_config(config_path))

    rows = [
        json.loads(line)
        for line in (artifact_root / "sink_triage" / "sink_candidate_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["row_count"] == 2
    assert summary["max_candidates"] == 5000
    assert rows[0]["sink_candidate_status"] == "candidate_only_not_intervened"
    assert rows[0]["case_id"] == "case-b"
    assert (artifact_root / "sink_triage" / "summary.json").exists()
    assert (artifact_root / "sink_triage" / "report.md").exists()


def test_scriptmaster_dry_run_writes_worktree_command_file(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase2_root = tmp_path / "phase2"
    fn_root = tmp_path / "fn"
    config_path = tmp_path / "phase3.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase2_root=phase2_root, fn_root=fn_root)
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "CONFIG": str(config_path),
        "SESSION": "phase3_scriptmaster_test",
        "GPU_LIST": "4,5",
    }

    result = subprocess.run(
        ["bash", str(SCRIPTMASTER_PATH)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "DRY_RUN=1; not starting tmux session." in result.stdout
    assert "CUDA_VISIBLE_DEVICES=4" in result.stdout
    assert "--stages target_mask" in result.stdout
    command_file = artifact_root / "logs" / "phase3_scriptmaster_test_commands.sh"
    command_text = command_file.read_text(encoding="utf-8")
    assert f"cd {str(REPO_ROOT)!r}" in command_text or f"cd {REPO_ROOT}" in command_text
    assert str(REPO_ROOT / "scripts/analysis/run_autoreg_fn_rescue_attention_guided_causal_binding.py") in command_text
    assert "CUDA_VISIBLE_DEVICES=4" in command_text
    assert "CUDA_VISIBLE_DEVICES=5" in command_text
    assert "CUDA_VISIBLE_DEVICES= PYTHONPATH=" in command_text
    assert "--stages target_mask" in command_text
    assert "--stages competitor_source" in command_text
    assert "--stages sink_triage,case_linked,report" in command_text
    assert "analysis orchestration, not production training" in command_text


def test_scriptmaster_dry_run_with_existing_outputs_does_not_require_overwrite_or_delete(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "out"
    phase2_root = tmp_path / "phase2"
    fn_root = tmp_path / "fn"
    config_path = tmp_path / "phase3.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase2_root=phase2_root, fn_root=fn_root)
    sentinel = artifact_root / "target_mask" / "summary.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text('{"sentinel": true}\n', encoding="utf-8")
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "CONFIG": str(config_path),
        "SESSION": "phase3_scriptmaster_no_delete",
        "GPU_LIST": "0,1",
    }

    result = subprocess.run(
        ["bash", str(SCRIPTMASTER_PATH)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "existing Phase-3 outputs present; dry-run will not remove them" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == '{"sentinel": true}\n'


def test_scriptmaster_rejects_inherited_parent_repo_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase2_root = tmp_path / "phase2"
    fn_root = tmp_path / "fn"
    config_path = tmp_path / "phase3.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase2_root=phase2_root, fn_root=fn_root)
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "CONFIG": str(config_path),
        "SESSION": "phase3_scriptmaster_bad_root",
        "GPU_LIST": "0,1",
        "REPO_ROOT": "/data/CoordExp",
    }

    result = subprocess.run(
        ["bash", str(SCRIPTMASTER_PATH)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "REPO_ROOT override must equal launcher worktree root" in result.stderr

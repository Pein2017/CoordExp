from __future__ import annotations

import json
import os
import textwrap
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from src.analysis.duplication_collapse_analysis import (
    CheckpointSpec,
    HistoricalArtifactBundle,
    ResolvedStudyCheckpoint,
    _assign_local_line_indices,
    _build_report,
    _build_anchor_from_object_pair,
    _case_id,
    _classify_case,
    _detect_onset,
    _expand_replay_cases,
    _inventory_row,
    _resolve_probe_anchor,
    _subset_record_for_case,
    load_study_config,
    run_study,
)
from src.analysis.rollout_fn_factor_study import ResolvedCheckpoint

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "run_duplication_collapse_analysis.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_duplication_collapse_analysis_script",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)
_preconfigure_cuda_visible_devices = _SCRIPT_MODULE._preconfigure_cuda_visible_devices


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "study.yaml"
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return path


def test_load_study_config_enforces_authoritative_decode_contract(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        run:
          name: dup-study
          output_dir: output/analysis
          stages: [inventory, select_cases, report]

        workspace:
          root_dir: research/duplication_collapse

        subset:
          pinned_line_indices:
            stage1_case: [1040, 3079]

        execution:
          device: cuda:1

        decode:
          temperature: 0.0
          top_p: 0.9
          repetition_penalty: 1.05
          max_new_tokens: 3084
          seed: 42
          secondary_top_k: [16, 32]
          secondary_top_p: [0.95]

        probe:
          max_cases: 2

        controls:
          max_cases: 2

        report:
          write_markdown: false

        checkpoints:
          - alias: stage1_case
            path: output/stage1_2b/coco_bbox_max60-1024-lvis_proxy-center_parameterization-ckpt_1564-merged
        """,
    )

    config = load_study_config(config_path)

    assert config.run.stages == ("inventory", "select_cases", "report")
    assert config.workspace.root_dir == "research/duplication_collapse"
    assert config.subset.pinned_line_indices == {"stage1_case": (1040, 3079)}
    assert config.execution.device == "cuda:1"
    assert config.decode.temperature == 0.0
    assert config.decode.top_p == 0.9
    assert config.decode.repetition_penalty == 1.05
    assert config.decode.max_new_tokens == 3084
    assert config.decode.seed == 42
    assert config.decode.secondary_top_k == (16, 32)
    assert config.decode.secondary_top_p == (0.95,)
    assert config.run.checkpoint_name_filter == "merged"
    assert config.probe.enable_interventions is True
    assert config.probe.intervention_max_cases == 1
    assert config.subset.replay_case_aliases == ()


def test_run_study_fails_fast_for_missing_checkpoint(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        f"""
        run:
          name: dup-study-missing
          output_dir: {tmp_path.as_posix()}
          stages: [inventory]

        decode:
          temperature: 0.0
          top_p: 0.9
          repetition_penalty: 1.05
          max_new_tokens: 3084
          seed: 42

        checkpoints:
          - alias: missing_case
            path: {tmp_path.joinpath('missing-checkpoint').as_posix()}
        """,
    )

    with pytest.raises(FileNotFoundError):
        run_study(config_path)


def test_run_study_rejects_training_only_checkpoint_surface(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "training-only-merged"
    checkpoint_dir.mkdir()
    config_path = _write_config(
        tmp_path,
        f"""
        run:
          name: dup-study-training-only
          output_dir: {tmp_path.as_posix()}
          checkpoint_name_filter: merged
          stages: [inventory]

        decode:
          temperature: 0.0
          top_p: 0.9
          repetition_penalty: 1.05
          max_new_tokens: 3084
          seed: 42

        checkpoints:
          - alias: training_only
            path: {checkpoint_dir.as_posix()}
        """,
    )

    with pytest.raises(ValueError, match="training-only"):
        run_study(config_path)


def test_run_study_bootstraps_cases_for_fresh_checkpoint(tmp_path: Path) -> None:
    source_artifact_dir = tmp_path / "artifact"
    source_artifact_dir.mkdir()
    source_gt_vs_pred = source_artifact_dir / "gt_vs_pred.jsonl"
    source_gt_vs_pred.write_text(
        json.dumps(
            {
                "image": "images/sample.png",
                "width": 16,
                "height": 16,
                "gt": [{"desc": "apple", "bbox_2d": [1, 1, 8, 8]}],
                "pred": [{"desc": "apple", "bbox_2d": [1, 1, 8, 8]}],
                "image_id": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bootstrap_manifest = tmp_path / "bootstrap.jsonl"
    bootstrap_manifest.write_text(
        json.dumps(
            {
                "checkpoint_alias": "reference",
                "source_artifact_root": str(source_artifact_dir),
                "source_gt_vs_pred_jsonl": str(source_gt_vs_pred),
                "line_idx": 0,
                "pred_count": 12,
                "max_desc_count": 12,
                "same_desc_duplicate_pair_count": 6,
                "top_desc": "apple",
                "selection_reason": "same-desc duplicate family desc=apple",
                "case_id": "reference-line_00000",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        f"""
        run:
          name: dup-study-bootstrap
          output_dir: {tmp_path.as_posix()}
          stages: [inventory, select_cases]

        subset:
          max_cases_per_checkpoint: 1
          max_cases_total: 1
          bootstrap_case_manifest_jsonl: {bootstrap_manifest.as_posix()}

        decode:
          temperature: 0.0
          top_p: 0.9
          repetition_penalty: 1.05
          max_new_tokens: 3084
          seed: 42

        checkpoints:
          - alias: soft_case
            path: output/stage1/coco_bbox_max60-coco80-desc_first/epoch_4-softce_w1-coco80-ckpt_1832-merged
            prompt_variant: coco_80
            object_field_order: desc_first
            stage: stage1
            objective_family: softce_w1_4b
            coord_soft_ce_w1_state: enabled
            family_comparison_role: soft_coordinate_supervised
        """,
    )

    result = run_study(config_path)
    selected_path = (
        Path(result["run_dir"]) / "cases" / "selected_cases.jsonl"
    )
    selected_rows = [
        json.loads(line)
        for line in selected_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(selected_rows) == 1
    assert selected_rows[0]["checkpoint_alias"] == "soft_case"
    assert selected_rows[0]["selection_reason"].startswith(
        "bootstrap-manifest from reference:"
    )
    assert selected_rows[0]["source_gt_vs_pred_jsonl"] == str(source_gt_vs_pred)


def test_run_study_bootstrap_case_ids_stay_unique_across_same_line_idx_sources(
    tmp_path: Path,
) -> None:
    source_artifact_dir = tmp_path / "artifact"
    source_artifact_dir.mkdir()
    source_gt_vs_pred = source_artifact_dir / "gt_vs_pred.jsonl"
    source_gt_vs_pred.write_text(
        json.dumps(
            {
                "image": "images/sample.png",
                "width": 16,
                "height": 16,
                "gt": [{"desc": "apple", "bbox_2d": [1, 1, 8, 8]}],
                "pred": [{"desc": "apple", "bbox_2d": [1, 1, 8, 8]}],
                "image_id": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bootstrap_manifest = tmp_path / "bootstrap-dupe-line.jsonl"
    bootstrap_manifest.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "checkpoint_alias": "reference_a",
                    "source_artifact_root": str(source_artifact_dir),
                    "source_gt_vs_pred_jsonl": str(source_gt_vs_pred),
                    "line_idx": 0,
                    "pred_count": 3,
                    "max_desc_count": 3,
                    "same_desc_duplicate_pair_count": 0,
                    "top_desc": "apple",
                    "selection_reason": "control-a",
                    "case_id": "reference_a-line_00000",
                },
                {
                    "checkpoint_alias": "reference_b",
                    "source_artifact_root": str(source_artifact_dir),
                    "source_gt_vs_pred_jsonl": str(source_gt_vs_pred),
                    "line_idx": 0,
                    "pred_count": 4,
                    "max_desc_count": 4,
                    "same_desc_duplicate_pair_count": 0,
                    "top_desc": "apple",
                    "selection_reason": "control-b",
                    "case_id": "reference_b-line_00000",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        f"""
        run:
          name: dup-study-bootstrap-unique
          output_dir: {tmp_path.as_posix()}
          stages: [inventory, select_cases]

        subset:
          max_cases_per_checkpoint: 2
          max_cases_total: 2
          bootstrap_case_manifest_jsonl: {bootstrap_manifest.as_posix()}

        decode:
          temperature: 0.0
          top_p: 0.9
          repetition_penalty: 1.05
          max_new_tokens: 3084
          seed: 42

        checkpoints:
          - alias: soft_case
            path: output/stage1/coco_bbox_max60-coco80-desc_first/epoch_4-softce_w1-coco80-ckpt_1832-merged
            prompt_variant: coco_80
            object_field_order: desc_first
            stage: stage1
            objective_family: softce_w1_4b
            coord_soft_ce_w1_state: enabled
            family_comparison_role: soft_coordinate_supervised
        """,
    )

    result = run_study(config_path)
    selected_path = Path(result["run_dir"]) / "cases" / "selected_cases.jsonl"
    selected_rows = [
        json.loads(line)
        for line in selected_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    case_ids = [row["case_id"] for row in selected_rows]
    assert len(case_ids) == 2
    assert len(set(case_ids)) == 2
    assert case_ids[0] == "soft_case-from_reference_a-line_00000"
    assert case_ids[1] == "soft_case-from_reference_b-line_00000"


def test_expand_replay_cases_duplicates_source_cases_for_target_aliases(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
        run:
          name: dup-replay
          output_dir: output/analysis
          stages: [inventory, select_cases]

        subset:
          replay_case_aliases: [soft_case]

        decode:
          temperature: 0.0
          top_p: 0.9
          repetition_penalty: 1.05
          max_new_tokens: 3084
          seed: 42

        checkpoints:
          - alias: ready_case
            path: output/stage1_2b/coco_bbox_max60-1024-lvis_proxy-center_parameterization-ckpt_1564-merged
          - alias: soft_case
            path: output/stage1/coco_bbox_max60-coco80-desc_first/epoch_4-softce_w1-coco80-ckpt_1832-merged
        """,
    )
    cfg = load_study_config(config_path)
    source_case = {
        "checkpoint_alias": "ready_case",
        "case_id": "ready_case-line_00007",
        "line_idx": 7,
        "selection_reason": "same-desc duplicate family",
        "historical_row": {"image": "images/sample.png", "gt": [], "pred": []},
    }
    resolved_checkpoints = [
        ResolvedStudyCheckpoint(
            spec=CheckpointSpec(alias="ready_case", path="ready"),
            resolved=ResolvedCheckpoint(
                alias="ready_case",
                path=Path("ready"),
                resolve_source="manifest",
                artifact_kind="executable_checkpoint",
                fingerprint="ready-case",
                prompt_variant="coco_80",
                object_field_order="desc_first",
                prompt_control_source="manifest",
                provenance_sidecars={},
            ),
            bundles=(),
        ),
        ResolvedStudyCheckpoint(
            spec=CheckpointSpec(alias="soft_case", path="soft"),
            resolved=ResolvedCheckpoint(
                alias="soft_case",
                path=Path("soft"),
                resolve_source="manifest",
                artifact_kind="executable_checkpoint",
                fingerprint="soft-case",
                prompt_variant="coco_80",
                object_field_order="desc_first",
                prompt_control_source="manifest",
                provenance_sidecars={},
            ),
            bundles=(),
        ),
    ]

    expanded = _expand_replay_cases(
        [source_case],
        cfg=cfg,
        resolved_checkpoints=resolved_checkpoints,
    )

    assert len(expanded) == 2
    assert expanded[0]["checkpoint_alias"] == "ready_case"
    replay = expanded[1]
    assert replay["checkpoint_alias"] == "soft_case"
    assert replay["replay_source_checkpoint_alias"] == "ready_case"
    assert replay["replay_source_case_id"] == "ready_case-line_00007"
    assert replay["case_id"] == "soft_case-from_ready_case-line_00007"


def test_case_id_prefers_source_case_id_when_available() -> None:
    assert (
        _case_id(
            "soft_case",
            7,
            source_case_id="ready_case-line_00007",
        )
        == "soft_case-from_ready_case-line_00007"
    )


def test_assign_local_line_indices_resets_per_checkpoint_alias() -> None:
    assigned = _assign_local_line_indices(
        [
            {"checkpoint_alias": "a", "case_id": "a0"},
            {"checkpoint_alias": "a", "case_id": "a1"},
            {"checkpoint_alias": "b", "case_id": "b0"},
            {"checkpoint_alias": "a", "case_id": "a2"},
            {"checkpoint_alias": "b", "case_id": "b1"},
        ]
    )

    assert [row["local_line_idx"] for row in assigned] == [0, 1, 0, 2, 1]


def test_detect_onset_prefers_first_same_desc_local_drift_case(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        run:
          name: dup-onset
          output_dir: output/analysis
          stages: [inventory]

        subset:
          duplicate_iou_threshold: 0.6
          local_center_radius_scale: 1.0
          size_ratio_min: 0.7

        decode:
          temperature: 0.0
          top_p: 0.9
          repetition_penalty: 1.05
          max_new_tokens: 3084
          seed: 42

        checkpoints:
          - alias: stage1_case
            path: output/stage1_2b/coco_bbox_max60-1024-lvis_proxy-center_parameterization-ckpt_1564-merged
        """,
    )
    cfg = load_study_config(config_path)

    preds = [
        {"desc": "apple", "bbox_2d": [100, 100, 180, 180]},
        {"desc": "banana", "bbox_2d": [400, 400, 460, 470]},
        {"desc": "apple", "bbox_2d": [105, 103, 183, 182]},
        {"desc": "apple", "bbox_2d": [108, 107, 186, 184]},
    ]
    confidence = {
        "objects": [
            {"confidence_details": {"desc_span_token_indices": [0, 1], "matched_token_indices": [2, 3, 4, 5]}},
            {"confidence_details": {"desc_span_token_indices": [6, 7], "matched_token_indices": [8, 9, 10, 11]}},
            {"confidence_details": {"desc_span_token_indices": [12, 13], "matched_token_indices": [14, 15, 16, 17]}},
            {"confidence_details": {"desc_span_token_indices": [18, 19], "matched_token_indices": [20, 21, 22, 23]}},
        ]
    }

    onset = _detect_onset(preds, confidence_record=confidence, cfg=cfg)

    assert onset is not None
    assert onset["object_idx"] == 2
    assert onset["source_object_idx"] == 0
    assert onset["desc"] == "apple"
    assert onset["onset_generated_token_idx"] == 12
    assert onset["onset_field_phase"] == "desc"
    assert onset["duplicate_pair_count"] >= 1


def test_build_anchor_from_object_pair_supports_nonduplicate_same_desc_control(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
        run:
          name: dup-control-anchor
          output_dir: output/analysis
          stages: [inventory]

        subset:
          duplicate_iou_threshold: 0.6
          local_center_radius_scale: 1.0
          size_ratio_min: 0.7

        decode:
          temperature: 0.0
          top_p: 0.9
          repetition_penalty: 1.05
          max_new_tokens: 3084
          seed: 42

        checkpoints:
          - alias: stage1_case
            path: output/stage1_2b/coco_bbox_max60-1024-lvis_proxy-center_parameterization-ckpt_1564-merged
        """,
    )
    cfg = load_study_config(config_path)

    preds = [
        {"desc": "chair", "bbox_2d": [10, 10, 50, 80]},
        {"desc": "chair", "bbox_2d": [120, 15, 165, 90]},
    ]
    confidence = {
        "objects": [
            {"confidence_details": {"desc_span_token_indices": [7], "matched_token_indices": [16, 19, 22, 25]}},
            {"confidence_details": {"desc_span_token_indices": [31], "matched_token_indices": [40, 43, 46, 49]}},
        ]
    }

    anchor = _build_anchor_from_object_pair(
        preds,
        confidence_record=confidence,
        cfg=cfg,
        object_idx=1,
        source_object_idx=0,
        anchor_source="case_manifest_pair",
    )

    assert anchor is not None
    assert anchor["object_idx"] == 1
    assert anchor["source_object_idx"] == 0
    assert anchor["anchor_source"] == "case_manifest_pair"
    assert anchor["pair_metrics"]["duplicate_like"] is False
    assert anchor["desc_span_token_indices"] == [31]
    assert anchor["matched_token_indices"] == [40, 43, 46, 49]


def test_resolve_probe_anchor_falls_back_to_manifest_pair_when_duplicate_missing(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
        run:
          name: dup-control-fallback
          output_dir: output/analysis
          stages: [inventory]

        subset:
          duplicate_iou_threshold: 0.6
          local_center_radius_scale: 1.0
          size_ratio_min: 0.7

        decode:
          temperature: 0.0
          top_p: 0.9
          repetition_penalty: 1.05
          max_new_tokens: 3084
          seed: 42

        checkpoints:
          - alias: stage1_case
            path: output/stage1_2b/coco_bbox_max60-1024-lvis_proxy-center_parameterization-ckpt_1564-merged
        """,
    )
    cfg = load_study_config(config_path)

    preds = [
        {"desc": "wine glass", "bbox_2d": [20, 20, 50, 100]},
        {"desc": "wine glass", "bbox_2d": [140, 25, 175, 105]},
    ]
    confidence = {
        "objects": [
            {"confidence_details": {"desc_span_token_indices": [7, 8], "matched_token_indices": [16, 19, 22, 25]}},
            {"confidence_details": {"desc_span_token_indices": [31, 32], "matched_token_indices": [40, 43, 46, 49]}},
        ]
    }

    anchor = _resolve_probe_anchor(
        preds,
        confidence_record=confidence,
        case_row={"onset": {"object_idx": 1, "source_object_idx": 0}},
        cfg=cfg,
    )

    assert anchor is not None
    assert anchor["anchor_source"] == "case_manifest_pair"
    assert anchor["pair_metrics"]["duplicate_like"] is False
    assert anchor["onset_generated_token_idx"] == 31


def test_inventory_row_marks_missing_artifact_cells_explicitly() -> None:
    checkpoint = ResolvedStudyCheckpoint(
        spec=CheckpointSpec(
            alias="demo",
            path="output/stage1_2b/demo",
            coord_soft_ce_w1_state="disabled",
            parent_checkpoint="parent-demo",
            family_comparison_role="ce_proxy_disabled_continuation",
        ),
        resolved=ResolvedCheckpoint(
            alias="demo",
            path=Path("output/stage1_2b/demo"),
            resolve_source="manifest",
            artifact_kind="executable_checkpoint",
            fingerprint="demo",
            prompt_variant="coco_80",
            object_field_order="desc_first",
            prompt_control_source="manifest",
            provenance_sidecars={},
        ),
        bundles=(),
    )

    row = _inventory_row(checkpoint)

    assert row["comparison_cells"]["bundle_0:gt_vs_pred_jsonl"]["available"] is False
    assert row["comparison_cells"]["bundle_0:pred_token_trace_jsonl"]["path"] is None
    assert row["coord_soft_ce_w1_state"] == "disabled"
    assert row["parent_checkpoint"] == "parent-demo"
    assert row["probe_readiness"] == "fresh_inference_needed"
    assert row["best_probe_surface"] == "output/stage1_2b/demo"


def test_build_report_preserves_probe_surface_status(tmp_path: Path) -> None:
    report = _build_report(
        inventory={
            "inventory_dir": str(tmp_path / "inventory"),
            "checkpoints": [
                {
                    "alias": "demo",
                    "family_comparison_role": "ce_proxy_disabled_continuation",
                    "coord_soft_ce_w1_state": "disabled",
                    "parent_checkpoint": "demo-parent",
                    "probe_readiness": "ready_to_probe",
                    "has_infer_artifact": True,
                    "best_probe_surface": "output/infer/demo",
                },
                {
                    "alias": "soft",
                    "family_comparison_role": "soft_coordinate_supervised",
                    "coord_soft_ce_w1_state": "enabled",
                    "parent_checkpoint": None,
                    "probe_readiness": "fresh_inference_needed",
                    "has_infer_artifact": False,
                    "best_probe_surface": "output/stage1/soft",
                },
            ],
            "ready_to_probe": [{"alias": "demo"}],
            "fresh_inference_needed": [{"alias": "soft"}],
        },
        selected_cases=[
            {
                "case_id": "case-1",
                "checkpoint_alias": "demo",
                "pred_count": 12,
                "gt_count": 3,
                "max_desc_count": 5,
                "same_desc_duplicate_pair_count": 4,
                "top_desc": "apple",
                "selection_reason": "pinned",
                "onset": {"onset_field_phase": "desc"},
            }
        ],
        reproduction={"reproductions": [{"checkpoint_alias": "demo"}]},
        probe_rows=[
            {
                "case_id": "case-1",
                "checkpoint_alias": "demo",
                "reproduced_onset": {"onset_field_phase": "coord_x1"},
                "probe": {
                    "probe_surface_status": {
                        "raw_logits": "available",
                        "llm_attentions": "available",
                        "llm_hidden_states": "available",
                        "llm_to_visual_attention": "available",
                        "native_vision": "unavailable",
                    },
                    "step_rows": [
                        {
                            "phase": "coord_x1",
                            "layer_group_mass_summary": {
                                "overwrite_summary": {
                                    "history_overwrite_detected": True,
                                    "prior_coord_overwrite_detected": False,
                                    "final_history_minus_visual": 0.25,
                                    "final_prior_coord_minus_visual": 0.1,
                                    "visual_drop_from_peak_to_final": 0.2,
                                }
                            },
                            "coord_summary": {
                                "previous_box_neighborhood_mass": 0.6,
                            },
                            "layer_logit_lens_summary": {
                                "prev_favored_detected": True,
                                "first_prev_favored_layer": 6,
                                "final_target_minus_previous": -0.2,
                                "target_recovery_detected": False,
                            },
                            "interventions": [
                                {
                                    "intervention_id": "late_layer_visual_bias",
                                    "behavioral_outcome": "shifted_to_target",
                                    "signal_deltas": {
                                        "target_minus_previous_prob": 0.15,
                                        "final_history_minus_visual_delta": -0.2,
                                    },
                                }
                            ],
                        }
                    ],
                },
            }
        ],
        compare={"case_rows": [{"case_id": "case-1", "margins": {"gt_next_minus_duplicate": -1.0}}]},
        run_dir=tmp_path,
    )

    assert report["case_rows"][0]["probe_surface_status"] == {
        "raw_logits": "available",
        "llm_attentions": "available",
        "llm_hidden_states": "available",
        "llm_to_visual_attention": "available",
        "native_vision": "unavailable",
    }
    assert report["case_rows"][0]["evidence_layers"] == {
        "artifact_audit": True,
        "rollout_reproduction": True,
        "deterministic_reforward": True,
        "controlled_compare": True,
        "deep_onset_probe": True,
    }
    assert report["case_rows"][0]["precursor_signals"]["max_desc_count"] == 5
    assert report["readiness_split"]["ready_to_probe"][0]["alias"] == "demo"
    assert report["case_rows"][0]["mechanism_signals"]["history_overwrite_detected"] is True
    assert report["case_rows"][0]["mechanism_signals"]["prev_favored_detected"] is True
    assert report["case_rows"][0]["mechanism_signals"]["first_prev_favored_layer"] == 6
    assert report["case_rows"][0]["intervention_summary"]["available"] is True
    assert report["family_comparison"]["rows"][0]["alias"] == "demo"
    assert (tmp_path / "report" / "summary.json").exists()


def test_subset_record_for_case_resolves_historical_relative_image_path(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    source_jsonl = artifact_dir / "gt_vs_pred.jsonl"
    source_jsonl.write_text("", encoding="utf-8")
    image_path = artifact_dir / "images" / "sample.png"
    image_path.parent.mkdir()
    Image.new("RGB", (4, 4), color="white").save(image_path)

    row = {
        "case_id": "case-1",
        "source_gt_vs_pred_jsonl": str(source_jsonl),
        "line_idx": 7,
        "selection_reason": "pinned",
        "historical_row": {
            "image": "images/sample.png",
            "width": 4,
            "height": 4,
            "gt": [{"desc": "apple", "bbox_2d": [0, 0, 3, 3]}],
            "image_id": 123,
        },
    }

    subset_row = _subset_record_for_case(row)

    assert subset_row["image"] == str(image_path)
    assert subset_row["images"] == [str(image_path)]
    assert subset_row["metadata"]["source_line_idx"] == 7


def test_preconfigure_cuda_visible_devices_uses_config_when_env_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            execution:
              cuda_visible_devices: "6"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _preconfigure_cuda_visible_devices(cfg_path)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"


def test_preconfigure_cuda_visible_devices_respects_existing_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            execution:
              cuda_visible_devices: "6"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    _preconfigure_cuda_visible_devices(cfg_path)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_subset_record_for_case_uses_artifact_root_image_dir(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    source_jsonl = artifact_dir / "gt_vs_pred.jsonl"
    source_jsonl.write_text("", encoding="utf-8")
    proxy_root = tmp_path / "rescale_32_1024_bbox_max60_lvis_proxy"
    proxy_root.mkdir()
    image_path = tmp_path / "rescale_32_1024_bbox_max60" / "images" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), color="white").save(image_path)
    (artifact_dir / "resolved_config.json").write_text(
        textwrap.dedent(
            f"""
            {{
              "root_image_dir": "{proxy_root.as_posix()}"
            }}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    row = {
        "case_id": "case-2",
        "source_artifact_root": str(artifact_dir),
        "source_gt_vs_pred_jsonl": str(source_jsonl),
        "line_idx": 9,
        "selection_reason": "pinned",
        "historical_row": {
            "image": "../rescale_32_1024_bbox_max60/images/sample.png",
            "width": 4,
            "height": 4,
            "gt": [{"desc": "apple", "bbox_2d": [0, 0, 3, 3]}],
            "image_id": 456,
        },
    }

    subset_row = _subset_record_for_case(row)

    assert subset_row["image"] == str(image_path)
    assert subset_row["images"] == [str(image_path)]


def test_classify_case_marks_manifest_anchored_positive_margin_as_control_stable() -> None:
    probe_row = {
        "reproduced_onset": {
            "anchor_source": "case_manifest_pair",
            "object_idx": 1,
            "source_object_idx": 0,
        },
        "probe": {"step_rows": [{"cross_step_summary": {"final_hidden_delta_l2": 1.0}}]},
    }
    compare_row = {
        "margins": {
            "gt_next_minus_duplicate": 0.75,
            "close_minus_duplicate": 0.5,
        },
        "exact_duplicate": {},
        "gt_next": {},
    }

    assert _classify_case(probe_row=probe_row, compare_row=compare_row) == "control-stable"

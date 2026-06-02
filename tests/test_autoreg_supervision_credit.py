from __future__ import annotations

import json
from pathlib import Path

from src.analysis.autoreg_supervision_credit import run_supervision_credit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_roots(
    tmp_path: Path,
    *,
    log_rows: list[dict] | None = None,
    lane_b: str = "missing",
    lane_c: bool = False,
) -> tuple[Path, Path]:
    analysis_root = tmp_path / "analysis"
    training_run_dir = tmp_path / "train"

    _write_json(
        analysis_root / "rollout_anatomy/summary.json",
        {
            "counts": {
                "images": 2,
                "gt_objects": 5,
                "raw_predictions": 3,
                "guarded_predictions": 2,
                "under_generated_images": 1,
            },
            "scope": {
                "dataset_slice": "first_2",
                "metric_family": "guarded",
            },
        },
    )
    _write_json(
        training_run_dir / "resolved_config.json",
        {
            "schema_version": 1,
            "resolved": {
                "objective": {
                    "id": "recursive_detection_ce",
                    "variant": "random_permutation_et_rmp_ce",
                    "trie_support_weight": 2.0,
                    "trie_balance_weight": 1.0,
                    "state_weighting": "uniform_permutation",
                    "normalization": "semantic_image_bucket_balanced",
                }
            },
        },
    )
    _write_json(
        training_run_dir / "checkpoint-7/trainer_state.json",
        {"global_step": 7, "log_history": [{"step": 7, "loss": 1.0}]},
    )
    _write_jsonl(
        training_run_dir / "logging.jsonl",
        log_rows
        if log_rows is not None
        else [
            {
                "loss": 3.0,
                "recursive_detection_ce/trie_support_weight": 2.0,
                "recursive_detection_ce/trie_balance_weight": 1.0,
            },
            {
                "eval_loss": 2.0,
                "eval_recursive_detection_ce/trie_support_weight": 2.0,
            },
        ],
    )

    if lane_b == "complete":
        _write_json(analysis_root / "prefix_boundary/summary.json", {"row_count": 4})
        _write_jsonl(
            analysis_root / "prefix_boundary/per_case.jsonl",
            [{"source_line_idx": 0}, {"source_line_idx": 1}],
        )
        _write_json(
            analysis_root / "prefix_boundary/merge_summary.json",
            {"row_count": 2, "expected_shards": 2, "selected_record_count": 2},
        )
    elif lane_b == "shards_only":
        _write_json(
            analysis_root / "prefix_boundary/shards/shard_000-of-002/summary.json",
            {"row_count": 1},
        )

    if lane_c:
        _write_json(
            analysis_root / "x1_basin_attribution/summary.json",
            {"stage": "x1_basin_attribution", "row_count": 8, "case_count": 2},
        )

    return analysis_root, training_run_dir


def test_missing_lane_b_and_c_keeps_h5_inconclusive_and_recommends_none(tmp_path: Path) -> None:
    analysis_root, training_run_dir = _fixture_roots(tmp_path, lane_b="missing")

    result = run_supervision_credit(
        analysis_root=analysis_root,
        training_run_dir=training_run_dir,
        checkpoint=str(training_run_dir / "checkpoint-7"),
    )

    assert result["h5_readout"]["status"] == "inconclusive_missing_lane"
    assert result["h5_readout"]["support_level"] == "compatible_only"
    assert result["h5_readout"]["production_training_recommendation"] == "none"
    assert "lane_c_x1_basin_attribution" in result["h5_readout"]["missing_discriminators"]
    assert result["prefix_boundary"]["status"] == "missing"


def test_training_log_inventory_distinguishes_absent_target_mix_from_zero(tmp_path: Path) -> None:
    zero_rows = [
        {
            "loss": 1.0,
            "recursive_detection_ce/target_mix/eos_fraction": 0.0,
            "recursive_detection_ce/target_mix/hard_ce_fraction": 0.0,
        }
    ]
    zero_analysis_root, zero_training_run_dir = _fixture_roots(
        tmp_path / "zero",
        log_rows=zero_rows,
    )
    zero_result = run_supervision_credit(
        analysis_root=zero_analysis_root,
        training_run_dir=zero_training_run_dir,
        checkpoint=str(zero_training_run_dir / "checkpoint-7"),
    )

    absent_analysis_root, absent_training_run_dir = _fixture_roots(tmp_path / "absent")
    absent_result = run_supervision_credit(
        analysis_root=absent_analysis_root,
        training_run_dir=absent_training_run_dir,
        checkpoint=str(absent_training_run_dir / "checkpoint-7"),
    )

    assert zero_result["observed_training_logs"]["target_mix_keys_present"] is True
    assert zero_result["observed_training_logs"]["target_mix_numeric"]["last"] == {
        "recursive_detection_ce/target_mix/eos_fraction": 0.0,
        "recursive_detection_ce/target_mix/hard_ce_fraction": 0.0,
    }
    assert absent_result["observed_training_logs"]["target_mix_keys_present"] is False
    assert absent_result["observed_training_logs"]["target_mix_numeric"]["last"] == {}


def test_target_mix_keys_present_are_detected_and_summarized(tmp_path: Path) -> None:
    analysis_root, training_run_dir = _fixture_roots(
        tmp_path,
        log_rows=[
            {
                "loss": 1.0,
                "recursive_detection_ce/target_mix/eos_fraction": 0.25,
                "recursive_detection_ce/target_mix/hard_ce_fraction": 0.75,
                "recursive_detection_ce/coord_soft_ce/enabled": 1.0,
            },
            {
                "loss": 0.5,
                "recursive_detection_ce/target_mix/eos_fraction": 0.5,
                "recursive_detection_ce/target_mix/hard_ce_fraction": 0.5,
            },
        ],
    )

    result = run_supervision_credit(
        analysis_root=analysis_root,
        training_run_dir=training_run_dir,
        checkpoint=str(training_run_dir / "checkpoint-7"),
    )

    observed = result["observed_training_logs"]
    assert observed["row_counts"] == {"total": 2, "train": 2, "eval": 0}
    assert observed["target_mix_keys_present"] is True
    assert observed["target_mix_keys"] == [
        "recursive_detection_ce/target_mix/eos_fraction",
        "recursive_detection_ce/target_mix/hard_ce_fraction",
    ]
    assert observed["target_mix_numeric"]["non_null_counts"] == {
        "recursive_detection_ce/target_mix/eos_fraction": 2,
        "recursive_detection_ce/target_mix/hard_ce_fraction": 2,
    }
    assert observed["coord_soft_ce_keys_present"] is True


def test_prefix_boundary_shards_without_merged_outputs_are_incomplete(tmp_path: Path) -> None:
    analysis_root, training_run_dir = _fixture_roots(tmp_path, lane_b="shards_only")

    result = run_supervision_credit(
        analysis_root=analysis_root,
        training_run_dir=training_run_dir,
        checkpoint=str(training_run_dir / "checkpoint-7"),
    )

    assert result["inputs"]["lane_b_prefix_boundary"]["status"] == "incomplete_shards_only"
    assert result["prefix_boundary"]["status"] == "incomplete_shards_only"
    assert result["prefix_boundary"]["counts"] == {}


def test_h5_claim_bounds_do_not_call_lane_c_missing_when_lane_c_is_present(tmp_path: Path) -> None:
    analysis_root, training_run_dir = _fixture_roots(
        tmp_path,
        lane_b="complete",
        lane_c=True,
    )

    result = run_supervision_credit(
        analysis_root=analysis_root,
        training_run_dir=training_run_dir,
        checkpoint=str(training_run_dir / "checkpoint-7"),
    )

    h5 = result["h5_readout"]
    assert "lane_c_x1_basin_attribution" not in h5["missing_discriminators"]
    assert "No H5 objective recommendation until Lane C x1 attribution exists" not in h5[
        "claim_bounds"
    ]
    assert (
        "No H5 objective recommendation until missing discriminators are resolved: "
        "coordinate_locality, checkpoint_target_mix_logs"
    ) in h5["claim_bounds"]


def test_schema_has_claim_bounds_and_writes_outputs(tmp_path: Path) -> None:
    analysis_root, training_run_dir = _fixture_roots(tmp_path, lane_b="complete")

    result = run_supervision_credit(
        analysis_root=analysis_root,
        training_run_dir=training_run_dir,
        checkpoint=str(training_run_dir / "checkpoint-7"),
    )

    assert result["schema_version"] == 1
    assert result["analysis_name"] == "autoreg_supervision_credit"
    assert result["scope"]["claim_scope"] == "artifact_ledger_only_no_training_claim"
    assert result["scope"]["metric_family"] == "guarded"
    assert result["code_capability"] == {
        "current_repo_emits_target_mix_events": True,
        "capability_is_not_checkpoint_evidence": True,
    }
    assert result["prefix_boundary"]["status"] == "complete"
    assert (analysis_root / "supervision_credit/summary.json").exists()
    assert (analysis_root / "supervision_credit/report.md").exists()

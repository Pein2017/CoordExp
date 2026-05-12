from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.detection.coord_soft_targets import (
    CoordSoftTargetCandidate,
    CoordSoftTargetRuntimeConfig,
    build_iou_gibbs_coord_target,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "compute_iou_gibbs_coord_stats.py"
)
SPEC = importlib.util.spec_from_file_location("compute_iou_gibbs_coord_stats", MODULE_PATH)
assert SPEC is not None
coord_stats = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(coord_stats)


def test_one_token_iou_losses_skip_invalid_edge_moves() -> None:
    losses = coord_stats.one_token_iou_losses((0, 0, 10, 10))

    # x1 - 1 and y1 - 1 are invalid at the boundary; all other one-token
    # moves preserve xyxy order and remain in range.
    assert len(losses) == 6
    assert all(loss > 0.0 for loss in losses)


def test_summarize_losses_reports_median() -> None:
    summary = coord_stats.summarize_losses([0.1, 0.3, 0.2])

    assert summary["count"] == 3
    assert summary["median"] == pytest.approx(0.2)


def test_target_shape_audit_reports_required_slices() -> None:
    audit = coord_stats.compute_target_shape_audit([(0, 0, 10, 10)], tau=0.1)

    assert audit["overall"]["entropy"]["count"] == 4
    assert audit["slot"]["x1"]["support_bin_count"]["median"] == pytest.approx(10.0)
    assert audit["boundary_flag"]["true"]["peak_prob"]["count"] == 4
    assert audit["min_side<=32"]["true"]["effective_support_size"]["count"] == 4
    assert audit["min_side<=50"]["true"]["std"]["count"] == 4


def test_target_shape_stats_match_runtime_iou_gibbs_helper() -> None:
    bbox = (20, 30, 120, 180)
    tau = 0.1

    stats = coord_stats.target_shape_stats_for_slot(bbox, 2, tau=tau)
    dist = build_iou_gibbs_coord_target(
        (
            CoordSoftTargetCandidate(
                object_instance_id="box",
                slot_name="x2",
                bbox_xyxy=bbox,
                probability=1.0,
            ),
        ),
        CoordSoftTargetRuntimeConfig(
            target_distribution="iou_gibbs_v0",
            tau=tau,
            coord_token_start=0,
            coord_token_end=999,
        ),
    )

    assert stats["entropy"] == pytest.approx(float(dist.entropy.item()))
    assert stats["peak_prob"] == pytest.approx(float(dist.peak_prob.item()))
    assert stats["std"] == pytest.approx(float(dist.std.item()))
    assert stats["support_bin_count"] == int(dist.support_bin_count.item())


def test_target_shape_stats_can_audit_ciou_gibbs_distribution() -> None:
    bbox = (20, 30, 120, 180)
    tau = 0.1

    iou_stats = coord_stats.target_shape_stats_for_slot(
        bbox,
        0,
        tau=tau,
        target_distribution="iou_gibbs_v0",
    )
    ciou_stats = coord_stats.target_shape_stats_for_slot(
        bbox,
        0,
        tau=tau,
        target_distribution="ciou_gibbs_v0",
    )

    assert ciou_stats["support_bin_count"] == iou_stats["support_bin_count"]
    assert ciou_stats["entropy"] < iou_stats["entropy"]


def test_analyze_jsonl_rejects_known_noncanonical_bbox_chart_path(tmp_path: Path) -> None:
    jsonl = tmp_path / "cxcywh.coord.jsonl"
    jsonl.write_text(
        '{"objects":[{"bbox_2d":["<|coord_10|>","<|coord_10|>","<|coord_20|>","<|coord_20|>"]}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-canonical"):
        coord_stats.analyze_jsonl(jsonl, target_audit_sample=0)


def test_analyze_jsonl_rejects_noncanonical_prepared_bbox_metadata(
    tmp_path: Path,
) -> None:
    jsonl = tmp_path / "train.coord.jsonl"
    jsonl.write_text(
        '{"metadata":{"prepared_bbox_format":"cxcywh"},"objects":[{"bbox_2d":["<|coord_10|>","<|coord_10|>","<|coord_20|>","<|coord_20|>"]}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="offline-prepared non-canonical"):
        coord_stats.analyze_jsonl(jsonl, target_audit_sample=0)


def test_analyze_jsonl_records_surface_contract_and_provenance(tmp_path: Path) -> None:
    jsonl = tmp_path / "train.coord.jsonl"
    jsonl.write_text(
        '{"objects":[{"bbox_2d":["<|coord_10|>","<|coord_20|>","<|coord_30|>","<|coord_40|>"]}]}\n',
        encoding="utf-8",
    )

    result = coord_stats.analyze_jsonl(
        jsonl,
        target_audit_sample=1,
        target_distribution="ciou_gibbs_v0",
        command="python scripts/analysis/compute_iou_gibbs_coord_stats.py",
    )

    assert result["bbox_format"] == "xyxy"
    assert result["coordinate_surface"] == "coord_token_xyxy"
    assert result["target_distribution"] == "ciou_gibbs_v0"
    assert result["input_sha256"] == result["sha256"]
    assert result["provenance"]["input_sha256"] == result["input_sha256"]
    assert result["provenance"]["target_distribution"] == "ciou_gibbs_v0"
    assert result["provenance"]["script_sha256"]
    assert result["provenance"]["command"].startswith("python scripts/")

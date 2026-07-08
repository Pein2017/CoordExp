from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.common.errors import ArtifactContractError
from src.vis import render_gt_vs_prediction, render_prediction_comparison
from src.vis.normalization import load_visual_rows


def test_visual_normalization_converts_gt_bins_and_preserves_prediction_pixels(
    tmp_path: Path,
) -> None:
    run_dir = _write_artifacts(
        tmp_path / "run",
        image_size=(1248, 832),
        gt=[_obj("sandwich", [318, 19, 746, 527])],
        pred=[
            {
                "description": "sandwich",
                "bbox": [392, 8, 911, 435],
                "coord_bins": [314, 10, 730, 523],
                "score": 0.9,
            }
        ],
    )

    row = load_visual_rows(run_dir).rows[0]

    assert row.gt[0].bbox_pixel_xyxy == (397.0, 16.0, 931.0, 438.0)
    assert row.gt[0].source_coord_space == "norm1000"
    assert row.pred[0].bbox_pixel_xyxy == (392.0, 8.0, 911.0, 435.0)
    assert row.pred[0].source_coord_space == "pixel"
    assert row.pred[0].source_coord_bins == (314, 10, 730, 523)


def test_prediction_coord_bins_are_not_drawn_or_used_for_matching(tmp_path: Path) -> None:
    run_dir = _write_artifacts(
        tmp_path / "run",
        image_size=(1000, 1000),
        gt=[_obj("cat", [700, 700, 800, 800])],
        pred=[
            {
                "description": "cat",
                "bbox": [700, 700, 800, 800],
                "coord_bins": [10, 10, 20, 20],
                "score": 0.9,
            }
        ],
    )

    result = render_gt_vs_prediction(run_dir / "gt_vs_pred_scored.jsonl", tmp_path / "vis")
    manifest = _read_json(result.manifest_path)
    item = manifest["items"][0]

    assert manifest["coordinate_surfaces"]["pred_bbox"].startswith("pixel xyxy")
    assert item["match"]["tp"] == 1
    assert item["pred_objects"][0]["bbox_pixel_xyxy"] == [700, 700, 800, 800]
    assert item["pred_objects"][0]["source_coord_bins"] == [10, 10, 20, 20]
    assert result.image_paths[0].is_file()


def test_prediction_comparison_hides_matched_gt_and_records_surfaces(
    tmp_path: Path,
) -> None:
    left = _write_artifacts(
        tmp_path / "left",
        image_size=(100, 100),
        gt=[_obj("cat", [100, 100, 300, 300])],
        pred=[{"description": "cat", "bbox": [10, 10, 30, 30], "score": 0.9}],
    )
    right = _write_artifacts(
        tmp_path / "right",
        image_path=left / "image.png",
        image_size=(100, 100),
        gt=[_obj("cat", [100, 100, 300, 300])],
        pred=[{"description": "cat", "bbox": [10, 10, 30, 30], "score": 0.9}],
    )

    result = render_prediction_comparison(
        left,
        right,
        tmp_path / "compare",
        left_label="left",
        right_label="right",
    )
    manifest = _read_json(result.manifest_path)

    assert manifest["kind"] == "prediction_comparison"
    assert manifest["coordinate_surfaces"]["drawable_field"] == "bbox_pixel_xyxy"
    assert manifest["items"][0]["left"]["match"]["tp"] == 1
    assert manifest["items"][0]["right"]["match"]["tp"] == 1
    assert "Matched GT boxes are canceled" in result.readme_path.read_text(encoding="utf-8")
    assert result.image_paths[0].is_file()


def test_duplicate_hints_are_diagnostic_and_do_not_change_match_counts(
    tmp_path: Path,
) -> None:
    run_dir = _write_artifacts(
        tmp_path / "run",
        image_size=(100, 100),
        gt=[_obj("cat", [100, 100, 300, 300])],
        pred=[
            {"description": "cat", "bbox": [10, 10, 30, 30], "score": 0.9},
            {"description": "cat", "bbox": [11, 11, 31, 31], "score": 0.8},
        ],
    )

    result = render_gt_vs_prediction(run_dir, tmp_path / "vis")
    item = _read_json(result.manifest_path)["items"][0]

    assert item["match"]["tp"] == 1
    assert item["match"]["fp"] == 1
    assert len(item["match"]["duplicate_candidates"]) == 1
    assert item["match"]["duplicate_candidates"][0]["pred_a_status"] == "matched"
    assert item["match"]["duplicate_candidates"][0]["pred_b_status"] == "fp"


def test_gt_vs_prediction_renders_one_png_per_selected_row(tmp_path: Path) -> None:
    run_dir = _write_artifacts(
        tmp_path / "run",
        rows=[
            _row("row-a", tmp_path / "run" / "a.png", gt=[_obj("cat", [100, 100, 300, 300])], pred=[]),
            _row("row-b", tmp_path / "run" / "b.png", gt=[_obj("dog", [100, 100, 300, 300])], pred=[]),
        ],
    )

    result = render_gt_vs_prediction(run_dir, tmp_path / "vis", row_ids=["row-b"])

    assert len(result.image_paths) == 1
    assert result.image_paths[0].name == "0000_row-b_gt_vs_pred.png"
    assert _read_json(result.manifest_path)["items"][0]["row_id"] == "row-b"


def test_visualization_is_deterministic_for_crowded_labels(tmp_path: Path) -> None:
    run_dir = _write_artifacts(
        tmp_path / "run",
        image_size=(96, 96),
        gt=[_obj("cat", [100, 100, 500, 500]), _obj("dog", [120, 120, 520, 520])],
        pred=[
            {"description": "cat", "bbox": [10, 10, 48, 48], "score": 0.9},
            {"description": "wolf", "bbox": [12, 12, 50, 50], "score": 0.7},
        ],
    )

    out_a = render_gt_vs_prediction(run_dir, tmp_path / "a").image_paths[0]
    out_b = render_gt_vs_prediction(run_dir, tmp_path / "b").image_paths[0]

    assert out_a.read_bytes() == out_b.read_bytes()


def test_comparison_fails_on_gt_mismatch(tmp_path: Path) -> None:
    left = _write_artifacts(
        tmp_path / "left",
        image_size=(100, 100),
        gt=[_obj("cat", [100, 100, 300, 300])],
        pred=[],
    )
    right = _write_artifacts(
        tmp_path / "right",
        image_path=left / "image.png",
        image_size=(100, 100),
        gt=[_obj("dog", [100, 100, 300, 300])],
        pred=[],
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        render_prediction_comparison(left, right, tmp_path / "compare")

    assert exc_info.value.code == "vis.comparison_gt_mismatch"


def test_visualize_detection_cli_compare(tmp_path: Path) -> None:
    left = _write_artifacts(
        tmp_path / "left",
        image_size=(100, 100),
        gt=[_obj("cat", [100, 100, 300, 300])],
        pred=[{"description": "cat", "bbox": [10, 10, 30, 30], "score": 0.9}],
    )
    right = _write_artifacts(
        tmp_path / "right",
        image_path=left / "image.png",
        image_size=(100, 100),
        gt=[_obj("cat", [100, 100, 300, 300])],
        pred=[{"description": "cat", "bbox": [10, 10, 30, 30], "score": 0.9}],
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/visualize_detection.py",
            "compare",
            "--left-run-dir",
            str(left),
            "--right-run-dir",
            str(right),
            "--out-dir",
            str(tmp_path / "cli"),
            "--row-id",
            "row-1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "cli" / "manifest.json").is_file()
    assert "manifest:" in completed.stdout


def _write_artifacts(
    run_dir: Path,
    *,
    image_size: tuple[int, int] = (100, 100),
    image_path: Path | None = None,
    gt: list[dict[str, Any]] | None = None,
    pred: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = [_row("row-1", image_path or (run_dir / "image.png"), image_size=image_size, gt=gt or [], pred=pred or [])]
    raw_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        path = Path(row["image_path"])
        if not path.exists():
            Image.new("RGB", (row["image_width"], row["image_height"]), color=(128, 128, 128)).save(path)
        raw = {key: value for key, value in row.items() if key != "pred"}
        scored = dict(raw)
        scored["pred"] = row["pred"]
        raw_rows.append(raw)
        scored_rows.append(scored)
    _write_jsonl(run_dir / "gt_vs_pred.jsonl", raw_rows)
    _write_jsonl(run_dir / "gt_vs_pred_scored.jsonl", scored_rows)
    return run_dir


def _row(
    row_id: str,
    image_path: Path,
    *,
    image_size: tuple[int, int] = (100, 100),
    gt: list[dict[str, Any]],
    pred: list[dict[str, Any]],
) -> dict[str, Any]:
    width, height = image_size
    return {
        "row_id": row_id,
        "row_index": 0,
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "gt": gt,
        "pred": pred,
    }


def _obj(description: str, bbox: list[int]) -> dict[str, Any]:
    return {"description": description, "bbox": bbox}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

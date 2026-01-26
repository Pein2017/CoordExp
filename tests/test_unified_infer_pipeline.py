from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.infer.engine import detect_mode_from_gt
from src.infer.pipeline import resolve_artifacts
from src.infer.vis import render_vis_from_jsonl


def _write_jsonl(path: Path, records: list[object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            if isinstance(r, str):
                f.write(r + "\n")
            else:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_detect_mode_from_gt_coord_tokens(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "width": 100,
                "height": 100,
                "objects": [
                    {
                        "bbox_2d": [
                            "<|coord_10|>",
                            "<|coord_10|>",
                            "<|coord_20|>",
                            "<|coord_20|>",
                        ]
                    }
                ],
            }
        ],
    )

    mode, reason = detect_mode_from_gt(str(gt), sample_size=128)
    assert mode == "coord"
    assert reason == "coord_tokens_found"


def test_detect_mode_from_gt_points_exceed_image(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "width": 10,
                "height": 10,
                "objects": [{"bbox_2d": [0, 0, 999, 1]}],
            }
        ],
    )

    mode, reason = detect_mode_from_gt(str(gt), sample_size=128)
    assert mode == "coord"
    assert reason == "points_exceed_image"


def test_detect_mode_from_gt_within_bounds(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "width": 10,
                "height": 10,
                "objects": [{"bbox_2d": [0, 0, 9, 9]}],
            }
        ],
    )

    mode, reason = detect_mode_from_gt(str(gt), sample_size=128)
    assert mode == "text"
    assert reason == "within_image_bounds"


def test_detect_mode_from_gt_no_valid_records(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    _write_jsonl(gt, ["not json", {"width": None, "height": None, "objects": []}])

    mode, reason = detect_mode_from_gt(str(gt), sample_size=128)
    assert mode == "text"
    assert reason == "no_valid_records"


def test_pipeline_resolve_artifacts_defaults(tmp_path: Path) -> None:
    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "stages": {"infer": True, "eval": True, "vis": True},
        "infer": {
            "gt_jsonl": "x.jsonl",
            "model_checkpoint": "ckpt",
            "mode": "coord",
        },
    }
    artifacts, stages = resolve_artifacts(cfg)
    assert stages.infer and stages.eval and stages.vis
    assert artifacts.run_dir == tmp_path / "out" / "demo"
    assert artifacts.gt_vs_pred_jsonl == artifacts.run_dir / "gt_vs_pred.jsonl"
    assert artifacts.summary_json == artifacts.run_dir / "summary.json"
    assert artifacts.eval_dir == artifacts.run_dir / "eval"
    assert artifacts.vis_dir == artifacts.run_dir / "vis"


def test_vis_only_renders_without_model(tmp_path: Path, monkeypatch) -> None:
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    # Unified artifact schema expects `image` (not `images`).
    artifact = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(
        artifact,
        [
            {
                "image": "img.png",
                "width": 64,
                "height": 64,
                "mode": "text",
                "coord_mode": "pixel",
                "gt": [{"type": "bbox_2d", "points": [1, 1, 10, 10], "desc": "a", "score": 1.0}],
                "pred": [{"type": "bbox_2d", "points": [2, 2, 11, 11], "desc": "a", "score": 1.0}],
                "raw_output": "",
                "errors": [],
            }
        ],
    )

    monkeypatch.setenv("ROOT_IMAGE_DIR", str(tmp_path))

    out_dir = tmp_path / "vis"
    render_vis_from_jsonl(artifact, out_dir=out_dir, limit=1)

    assert (out_dir / "vis_0000.png").exists()

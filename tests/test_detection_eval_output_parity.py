from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pycocotools")

from src.eval.detection import EvalOptions, evaluate_and_save


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _one_record(*, image: str) -> dict:
    return {
        "image": image,
        "width": 64,
        "height": 48,
        "mode": "text",
        "coord_mode": "pixel",
        "gt": [
            {
                "type": "bbox_2d",
                "points": [0, 0, 63, 47],
                "desc": "box",
                "score": 1.0,
            }
        ],
        "pred": [
            {
                "type": "bbox_2d",
                "points": [0, 0, 63, 47],
                "desc": "box",
                "score": 1.0,
            }
        ],
        "raw_output_json": {},
        "raw_special_tokens": [],
        "raw_ends_with_im_end": True,
        "errors": [],
    }


def test_evaluate_and_save_writes_metrics_json(tmp_path: Path) -> None:
    pred_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(pred_path, [_one_record(image="img.png")])

    out_dir = tmp_path / "eval"
    options = EvalOptions(
        metrics="coco",
        strict_parse=True,
        use_segm=False,
        output_dir=out_dir,
        overlay=False,
        num_workers=0,
    )

    summary = evaluate_and_save(pred_path, options=options)
    assert summary["counters"]["invalid_json"] == 0
    assert summary["metrics"]["bbox_AP50"] > 0.9

    metrics_payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "metrics" in metrics_payload
    assert "counters" in metrics_payload
    assert metrics_payload["metrics"]["bbox_AP50"] == summary["metrics"]["bbox_AP50"]

    assert (out_dir / "coco_gt.json").exists()
    assert (out_dir / "coco_preds.json").exists()
    assert (out_dir / "per_image.json").exists()


def test_evaluate_and_save_both_includes_f1ish_metrics(tmp_path: Path) -> None:
    pred_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(pred_path, [_one_record(image="img.png")])

    out_dir = tmp_path / "eval"
    options = EvalOptions(
        metrics="both",
        strict_parse=True,
        use_segm=False,
        output_dir=out_dir,
        overlay=False,
        num_workers=0,
        # Avoid model downloads in unit tests.
        semantic_model="",
    )

    summary = evaluate_and_save(pred_path, options=options)
    assert summary["metrics"]["bbox_AP50"] > 0.9
    assert "f1ish@0.30_f1_loc_micro" in summary["metrics"]

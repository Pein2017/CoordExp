from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.proxy_views import filter_proxy_record, materialize_proxy_eval_views


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _record() -> dict:
    return {
        "images": ["images/val/000000000139.jpg"],
        "width": 1248,
        "height": 832,
        "gt": [
            {"bbox_2d": [10, 20, 40, 60], "desc": "clock"},
            {"bbox_2d": [50, 60, 90, 120], "desc": "vase"},
            {"bbox_2d": [100, 110, 180, 210], "desc": "banana"},
        ],
        "pred": [
            {"type": "bbox_2d", "points": [10, 20, 40, 60], "desc": "clock", "score": 0.8}
        ],
        "pred_score_source": "confidence_postop",
        "pred_score_version": 1,
        "metadata": {
            "coordexp_proxy_supervision": {
                "object_supervision": [
                    {"proxy_tier": "real", "source": "coco"},
                    {"proxy_tier": "strict", "source": "lvis"},
                    {"proxy_tier": "plausible", "source": "lvis"},
                ],
                "summary": {
                    "real_count": 1,
                    "strict_count": 1,
                    "plausible_count": 1,
                    "include_plausible": True,
                },
            },
            "source": "coco2017",
            "split": "val",
        },
    }


def test_filter_proxy_record_selects_expected_tiers() -> None:
    record = _record()

    coco_real = filter_proxy_record(record, view="coco_real")
    strict = filter_proxy_record(record, view="coco_real_strict")
    full = filter_proxy_record(record, view="coco_real_strict_plausible")

    assert [obj["desc"] for obj in coco_real["gt"]] == ["clock"]
    assert [obj["desc"] for obj in strict["gt"]] == ["clock", "vase"]
    assert [obj["desc"] for obj in full["gt"]] == ["clock", "vase", "banana"]
    assert full["pred"] == record["pred"]
    assert coco_real["metadata"]["coordexp_proxy_eval_view"]["view"] == "coco_real"


def test_materialize_proxy_eval_views_writes_expected_outputs(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "gt_vs_pred_scored.jsonl"
    output_dir = tmp_path / "proxy_eval_views"
    _write_jsonl(input_jsonl, [_record()])

    summary = materialize_proxy_eval_views(input_jsonl, output_dir=output_dir)

    assert summary["tier_counts_in"] == {
        "plausible": 1,
        "real": 1,
        "strict": 1,
    }
    assert set(summary["outputs"]) == {
        "coco_real",
        "coco_real_strict",
        "coco_real_strict_plausible",
    }
    assert (output_dir / "proxy_eval_views_summary.json").is_file()

    coco_real_path = output_dir / "gt_vs_pred_scored.coco_real.jsonl"
    strict_path = output_dir / "gt_vs_pred_scored.coco_real_strict.jsonl"
    full_path = output_dir / "gt_vs_pred_scored.coco_real_strict_plausible.jsonl"

    coco_real_row = json.loads(coco_real_path.read_text(encoding="utf-8").strip())
    strict_row = json.loads(strict_path.read_text(encoding="utf-8").strip())
    full_row = json.loads(full_path.read_text(encoding="utf-8").strip())

    assert len(coco_real_row["gt"]) == 1
    assert len(strict_row["gt"]) == 2
    assert len(full_row["gt"]) == 3


def test_materialize_proxy_eval_views_rejects_length_mismatch(tmp_path: Path) -> None:
    record = _record()
    record["metadata"]["coordexp_proxy_supervision"]["object_supervision"] = [
        {"proxy_tier": "real"}
    ]
    input_jsonl = tmp_path / "bad.jsonl"
    _write_jsonl(input_jsonl, [record])

    with pytest.raises(ValueError, match="length mismatch"):
        materialize_proxy_eval_views(input_jsonl, output_dir=tmp_path / "out")


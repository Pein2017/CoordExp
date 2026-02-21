from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pycocotools")

from src.eval.detection import (
    EvalCounters,
    EvalOptions,
    _prepare_all,
    _prepare_all_separate,
    _prepare_pred_objects,
    evaluate_and_save,
    preds_to_gt_records,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _one_record(*, image: str, gt_desc: str = "box", pred_desc: str = "box") -> dict:
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
                "desc": gt_desc,
                "score": 1.0,
            }
        ],
        "pred": [
            {
                "type": "bbox_2d",
                "points": [0, 0, 63, 47],
                "desc": pred_desc,
                "score": 0.9,
            }
        ],
        "pred_score_source": "confidence_postop",
        "pred_score_version": 1,
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


def test_evaluate_and_save_reports_zero_coco_metrics_when_no_predictions(
    tmp_path: Path,
) -> None:
    pred_path = tmp_path / "gt_vs_pred.jsonl"
    record = _one_record(image="img_empty.png")
    record["pred"] = []
    _write_jsonl(pred_path, [record])

    out_dir = tmp_path / "eval_empty"
    options = EvalOptions(
        metrics="coco",
        strict_parse=True,
        use_segm=False,
        output_dir=out_dir,
        overlay=False,
        num_workers=0,
    )
    summary = evaluate_and_save(pred_path, options=options)

    expected_keys = [
        "bbox_AP",
        "bbox_AP50",
        "bbox_AP75",
        "bbox_APs",
        "bbox_APm",
        "bbox_APl",
        "bbox_AR1",
        "bbox_AR10",
        "bbox_AR100",
        "bbox_ARs",
        "bbox_ARm",
        "bbox_ARl",
    ]
    for key in expected_keys:
        assert key in summary["metrics"]
        assert summary["metrics"][key] == 0.0

    metrics_payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    for key in expected_keys:
        assert metrics_payload["metrics"][key] == 0.0


def test_evaluate_and_save_both_includes_f1ish_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    pred_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(pred_path, [_one_record(image="img.png")])

    class _StubEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_norm_texts(self, texts):
            import numpy as np

            return {str(t): np.array([1.0, 0.0], dtype=np.float32) for t in texts}

    monkeypatch.setattr("src.eval.detection.SemanticDescEncoder", _StubEncoder)

    out_dir = tmp_path / "eval"
    options = EvalOptions(
        metrics="both",
        strict_parse=True,
        use_segm=False,
        output_dir=out_dir,
        overlay=False,
        num_workers=0,
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    summary = evaluate_and_save(pred_path, options=options)
    assert summary["metrics"]["bbox_AP50"] > 0.9
    assert "f1ish@0.30_f1_loc_micro" in summary["metrics"]


def test_prepare_all_and_prepare_all_separate_parity(tmp_path: Path) -> None:
    pred_records = [
        _one_record(image="img_a.png"),
        _one_record(image="img_b.png"),
    ]
    gt_records = preds_to_gt_records(pred_records)

    options = EvalOptions(
        metrics="both",
        strict_parse=True,
        use_segm=False,
        output_dir=tmp_path / "eval",
        overlay=False,
        num_workers=0,
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    counters_combined = EvalCounters()
    counters_separate = EvalCounters()

    combined = _prepare_all(
        pred_records,
        options,
        counters_combined,
        prepare_coco=True,
    )
    separate = _prepare_all_separate(
        gt_records,
        pred_records,
        options,
        counters_separate,
        prepare_coco=True,
    )

    assert combined[0] == separate[0]  # gt_samples
    assert combined[1] == separate[1]  # pred_samples
    assert combined[2] == separate[2]  # categories
    assert combined[3] == separate[3]  # coco_gt
    assert combined[4] == separate[4]  # coco_preds
    assert combined[5] == separate[5]  # run_segm
    assert combined[6] == separate[6]  # per_image


def test_evaluate_and_save_fails_when_semantic_encoder_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    pred_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(
        pred_path,
        [_one_record(image="img.png", gt_desc="cat", pred_desc="dog")],
    )

    class _BrokenEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_norm_texts(self, texts):
            raise RuntimeError("encoder unavailable")

    monkeypatch.setattr("src.eval.detection.SemanticDescEncoder", _BrokenEncoder)

    options = EvalOptions(
        metrics="coco",
        strict_parse=True,
        use_segm=False,
        output_dir=tmp_path / "eval",
        overlay=False,
        num_workers=0,
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    with pytest.raises(RuntimeError, match="Description matching requires the semantic encoder"):
        evaluate_and_save(pred_path, options=options)

def test_evaluate_and_save_f1ish_fails_when_semantic_encoder_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    pred_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(
        pred_path,
        [_one_record(image="img.png", gt_desc="cat", pred_desc="dog")],
    )

    class _BrokenEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_norm_texts(self, texts):
            raise RuntimeError("encoder unavailable")

    monkeypatch.setattr("src.eval.detection.SemanticDescEncoder", _BrokenEncoder)

    options = EvalOptions(
        metrics="f1ish",
        strict_parse=True,
        use_segm=False,
        output_dir=tmp_path / "eval",
        overlay=False,
        num_workers=0,
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    with pytest.raises(RuntimeError, match="F1-ish semantic filtering requires the semantic encoder"):
        evaluate_and_save(pred_path, options=options)

def test_eval_options_rejects_empty_semantic_model() -> None:
    with pytest.raises(ValueError, match="semantic_model must be a non-empty"):
        EvalOptions(
            metrics="coco",
            strict_parse=True,
            use_segm=False,
            output_dir=Path("eval_out"),
            semantic_model="",
        )


def test_evaluate_and_save_rejects_missing_score_provenance_for_coco(
    tmp_path: Path,
) -> None:
    record = _one_record(image="img.png")
    record.pop("pred_score_source", None)
    record.pop("pred_score_version", None)

    pred_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(pred_path, [record])

    options = EvalOptions(
        metrics="coco",
        strict_parse=True,
        use_segm=False,
        output_dir=tmp_path / "eval",
    )

    with pytest.raises(ValueError, match="pred_score_source"):
        evaluate_and_save(pred_path, options=options)


def test_evaluate_and_save_rejects_missing_pred_score_for_coco(tmp_path: Path) -> None:
    record = _one_record(image="img.png")
    record["pred"][0].pop("score", None)

    pred_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(pred_path, [record])

    options = EvalOptions(
        metrics="coco",
        strict_parse=True,
        use_segm=False,
        output_dir=tmp_path / "eval",
    )

    with pytest.raises(ValueError, match="missing `pred\\[\\*\\]\\.score`"):
        evaluate_and_save(pred_path, options=options)


def test_evaluate_and_save_rejects_out_of_range_pred_score_for_coco(
    tmp_path: Path,
) -> None:
    record = _one_record(image="img.png")
    record["pred"][0]["score"] = 1.2

    pred_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(pred_path, [record])

    options = EvalOptions(
        metrics="coco",
        strict_parse=True,
        use_segm=False,
        output_dir=tmp_path / "eval",
    )

    with pytest.raises(ValueError, match="out-of-range"):
        evaluate_and_save(pred_path, options=options)


def test_coco_export_preserves_input_order_on_score_ties(tmp_path: Path) -> None:
    record = _one_record(image="img.png")
    record["pred"] = [
        {
            "type": "bbox_2d",
            "points": [0, 0, 20, 20],
            "desc": "box",
            "score": 0.5,
        },
        {
            "type": "bbox_2d",
            "points": [10, 10, 30, 30],
            "desc": "box",
            "score": 0.5,
        },
    ]

    pred_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(pred_path, [record])

    options = EvalOptions(
        metrics="coco",
        strict_parse=True,
        use_segm=False,
        output_dir=tmp_path / "eval",
    )
    evaluate_and_save(pred_path, options=options)

    coco_preds = json.loads((options.output_dir / "coco_preds.json").read_text(encoding="utf-8"))
    assert [entry["bbox"] for entry in coco_preds] == [[0, 0, 20, 20], [10, 10, 20, 20]]


def test_prepare_pred_objects_rejects_nested_points_by_default() -> None:
    counters = EvalCounters()
    options = EvalOptions(strict_parse=False, use_segm=False, output_dir=Path("eval_out"))

    preds, invalid = _prepare_pred_objects(
        {
            "coord_mode": "pixel",
            "pred": [
                {
                    "type": "bbox_2d",
                    "points": [[1, 2], [10, 12]],
                    "desc": "nested",
                }
            ],
        },
        width=32,
        height=32,
        options=options,
        counters=counters,
    )

    assert preds == []
    assert counters.invalid_geometry == 1
    assert invalid and invalid[0]["reason"] == "geometry_points"

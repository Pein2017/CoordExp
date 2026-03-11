from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pycocotools")

from src.eval.detection import EvalOptions
from src.eval.oracle_k import (
    OracleKConfig,
    OracleKRunSpec,
    evaluate_oracle_k,
    run_oracle_k_from_config,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _box(desc: str, *, points: list[int] | None = None) -> dict:
    return {
        "type": "bbox_2d",
        "points": points or [0, 0, 31, 31],
        "desc": desc,
        "score": 0.9,
    }


def _record(
    *,
    image: str,
    gt_desc: str,
    pred: list[dict] | None = None,
    gt_points: list[int] | None = None,
) -> dict:
    return {
        "image": image,
        "width": 32,
        "height": 32,
        "mode": "text",
        "coord_mode": "pixel",
        "gt": [_box(gt_desc, points=gt_points)],
        "pred": pred or [],
        "raw_output_json": {},
        "raw_special_tokens": [],
        "raw_ends_with_im_end": True,
        "errors": [],
    }


def _stub_semantic_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_norm_texts(self, texts):
            import numpy as np

            matrix = np.eye(max(1, len(texts)), dtype=np.float32)
            return {str(text): matrix[idx] for idx, text in enumerate(texts)}

    monkeypatch.setattr("src.eval.detection.SemanticDescEncoder", _StubEncoder)


def _base_options(
    out_dir: Path,
    *,
    iou_thrs: list[float] | None = None,
    pred_scope: str = "annotated",
) -> EvalOptions:
    return EvalOptions(
        metrics="f1ish",
        strict_parse=True,
        use_segm=False,
        f1ish_iou_thrs=iou_thrs or [0.3, 0.5],
        f1ish_pred_scope=pred_scope,
        output_dir=out_dir,
        overlay=False,
        num_workers=0,
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
        semantic_threshold=0.6,
        semantic_device="cpu",
        semantic_batch_size=8,
    )


def test_oracle_k_tracks_recovery_views_and_frequencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_semantic_encoder(monkeypatch)

    baseline_path = tmp_path / "baseline.jsonl"
    oracle_a_path = tmp_path / "oracle_a.jsonl"
    oracle_b_path = tmp_path / "oracle_b.jsonl"

    baseline_records = [
        _record(image="img_loc.png", gt_desc="cat"),
        _record(image="img_full.png", gt_desc="cup"),
        _record(image="img_sys.png", gt_desc="chair"),
        _record(
            image="img_sem.png",
            gt_desc="lamp",
            pred=[_box("light")],
        ),
    ]
    oracle_a_records = [
        _record(image="img_loc.png", gt_desc="cat", pred=[_box("dog")]),
        _record(image="img_full.png", gt_desc="cup", pred=[_box("cup")]),
        _record(image="img_sys.png", gt_desc="chair"),
        _record(image="img_sem.png", gt_desc="lamp", pred=[_box("light")]),
    ]
    oracle_b_records = [
        _record(image="img_loc.png", gt_desc="cat"),
        _record(image="img_full.png", gt_desc="cup"),
        _record(image="img_sys.png", gt_desc="chair"),
        _record(image="img_sem.png", gt_desc="lamp", pred=[_box("lamp")]),
    ]
    _write_jsonl(baseline_path, baseline_records)
    _write_jsonl(oracle_a_path, oracle_a_records)
    _write_jsonl(oracle_b_path, oracle_b_records)

    out_dir = tmp_path / "oracle_k"
    config = OracleKConfig(
        out_dir=out_dir,
        eval_options=_base_options(out_dir, pred_scope="all"),
        baseline_run=OracleKRunSpec(label="baseline", pred_jsonl=baseline_path),
        oracle_runs=(
            OracleKRunSpec(label="oracle_a", pred_jsonl=oracle_a_path),
            OracleKRunSpec(label="oracle_b", pred_jsonl=oracle_b_path),
        ),
    )

    summary = evaluate_oracle_k(config)
    assert summary["oracle_run_count"] == 2
    assert summary["primary_iou_thr"] == pytest.approx(0.5)

    primary = summary["primary_recovery"]
    assert primary["baseline_fn_count_loc"] == 3
    assert primary["recoverable_fn_count_loc"] == 2
    assert primary["systematic_fn_count_loc"] == 1
    assert primary["recover_fraction_loc"] == pytest.approx(2.0 / 6.0)
    assert primary["baseline_fn_count_full"] == 4
    assert primary["recoverable_fn_count_full"] == 2
    assert primary["systematic_fn_count_full"] == 2
    assert primary["recover_fraction_full"] == pytest.approx(2.0 / 8.0)

    thr_summary = summary["iou_thresholds"]["0.50"]
    assert thr_summary["baseline"]["tp_loc"] == 1
    assert thr_summary["baseline"]["tp_full"] == 0
    assert thr_summary["oracle_k"]["tp_loc"] == 3
    assert thr_summary["oracle_k"]["tp_full"] == 2

    fn_rows = {
        row["file_name"]: row for row in _read_jsonl(out_dir / "fn_objects.jsonl")
    }
    loc_row = fn_rows["img_loc.png"]
    assert loc_row["baseline_fn_loc"] is True
    assert loc_row["baseline_fn_full"] is True
    assert loc_row["ever_recovered_loc"] is True
    assert loc_row["ever_recovered_full"] is False
    assert loc_row["recover_count_loc"] == 1
    assert loc_row["recover_count_full"] == 0
    assert loc_row["recovered_run_labels_loc"] == ["oracle_a"]
    oracle_a_row = next(run for run in loc_row["oracle_runs"] if run["label"] == "oracle_a")
    assert oracle_a_row["loc_hit"] is True
    assert oracle_a_row["full_hit"] is False

    sem_row = fn_rows["img_sem.png"]
    assert sem_row["baseline_fn_loc"] is False
    assert sem_row["baseline_fn_full"] is True
    assert sem_row["ever_recovered_loc"] is None
    assert sem_row["recover_count_loc"] is None
    assert sem_row["baseline_loc_hit"] is True
    assert sem_row["baseline_full_hit"] is False
    assert sem_row["ever_recovered_full"] is True
    assert sem_row["recover_count_full"] == 1
    assert sem_row["recovered_run_labels_full"] == ["oracle_b"]

    per_image = {
        row["file_name"]: row
        for row in json.loads((out_dir / "per_image.json").read_text(encoding="utf-8"))
    }
    assert per_image["img_sys.png"]["systematic_fn_count_loc"] == 1
    assert per_image["img_full.png"]["recoverable_fn_count_full"] == 1


def test_oracle_k_rejects_gt_content_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_semantic_encoder(monkeypatch)

    baseline_path = tmp_path / "baseline.jsonl"
    oracle_path = tmp_path / "oracle.jsonl"
    _write_jsonl(baseline_path, [_record(image="img.png", gt_desc="cat")])
    _write_jsonl(oracle_path, [_record(image="img.png", gt_desc="dog")])

    config = OracleKConfig(
        out_dir=tmp_path / "oracle_k",
        eval_options=_base_options(tmp_path / "oracle_k"),
        baseline_run=OracleKRunSpec(label="baseline", pred_jsonl=baseline_path),
        oracle_runs=(OracleKRunSpec(label="oracle", pred_jsonl=oracle_path),),
    )

    with pytest.raises(ValueError, match="GT object content mismatch"):
        evaluate_oracle_k(config)


def test_oracle_k_rejects_record_order_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_semantic_encoder(monkeypatch)

    baseline_path = tmp_path / "baseline.jsonl"
    oracle_path = tmp_path / "oracle.jsonl"
    baseline_records = [
        _record(image="img_a.png", gt_desc="cat"),
        _record(image="img_b.png", gt_desc="dog"),
    ]
    oracle_records = [
        _record(image="img_b.png", gt_desc="dog"),
        _record(image="img_a.png", gt_desc="cat"),
    ]
    _write_jsonl(baseline_path, baseline_records)
    _write_jsonl(oracle_path, oracle_records)

    config = OracleKConfig(
        out_dir=tmp_path / "oracle_k",
        eval_options=_base_options(tmp_path / "oracle_k"),
        baseline_run=OracleKRunSpec(label="baseline", pred_jsonl=baseline_path),
        oracle_runs=(OracleKRunSpec(label="oracle", pred_jsonl=oracle_path),),
    )

    with pytest.raises(ValueError, match="record-order mismatch"):
        evaluate_oracle_k(config)


@pytest.mark.parametrize(
    ("iou_thrs", "expected_primary"),
    [
        ([0.3, 0.7], 0.7),
        ([0.3, 0.5, 0.7], 0.5),
    ],
)
def test_oracle_k_primary_threshold_matches_detection_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    iou_thrs: list[float],
    expected_primary: float,
) -> None:
    _stub_semantic_encoder(monkeypatch)

    baseline_path = tmp_path / "baseline.jsonl"
    oracle_path = tmp_path / "oracle.jsonl"
    _write_jsonl(baseline_path, [_record(image="img.png", gt_desc="cat")])
    _write_jsonl(oracle_path, [_record(image="img.png", gt_desc="cat", pred=[_box("cat")])])

    config = OracleKConfig(
        out_dir=tmp_path / "oracle_k",
        eval_options=_base_options(tmp_path / "oracle_k", iou_thrs=iou_thrs),
        baseline_run=OracleKRunSpec(label="baseline", pred_jsonl=baseline_path),
        oracle_runs=(OracleKRunSpec(label="oracle", pred_jsonl=oracle_path),),
    )

    summary = evaluate_oracle_k(config)
    assert summary["primary_iou_thr"] == pytest.approx(expected_primary)


def test_oracle_k_can_materialize_pipeline_runs_from_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_semantic_encoder(monkeypatch)

    baseline_path = tmp_path / "baseline.jsonl"
    materialized_path = tmp_path / "materialized" / "oracle" / "gt_vs_pred.jsonl"
    trace_path = materialized_path.parent / "pred_token_trace.jsonl"
    resolved_config = materialized_path.parent / "resolved_config.json"
    pipeline_config = tmp_path / "pipeline.yaml"

    _write_jsonl(baseline_path, [_record(image="img.png", gt_desc="cat")])
    pipeline_config.write_text("run:\n  name: demo\n", encoding="utf-8")

    def _fake_run_pipeline(*, config_path: Path, overrides: dict) -> object:
        from src.infer.pipeline import ResolvedArtifacts

        assert config_path == pipeline_config
        assert overrides["stages.infer"] is True
        assert overrides["stages.eval"] is False
        assert overrides["stages.vis"] is False
        materialized_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            materialized_path,
            [_record(image="img.png", gt_desc="cat", pred=[_box("cat")])],
        )
        trace_path.write_text("{}\n", encoding="utf-8")
        resolved_config.write_text("{}", encoding="utf-8")
        return ResolvedArtifacts(
            run_dir=materialized_path.parent,
            gt_vs_pred_jsonl=materialized_path,
            pred_token_trace_jsonl=trace_path,
            gt_vs_pred_scored_jsonl=None,
            summary_json=materialized_path.parent / "summary.json",
            eval_dir=materialized_path.parent / "eval",
            vis_dir=materialized_path.parent / "vis",
        )

    monkeypatch.setattr("src.infer.pipeline.run_pipeline", _fake_run_pipeline)

    config_path = tmp_path / "oracle_k.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"out_dir: {tmp_path / 'oracle_k'}",
                "strict_parse: true",
                "f1ish_iou_thrs: [0.3, 0.5]",
                "f1ish_pred_scope: annotated",
                "semantic_model: sentence-transformers/all-MiniLM-L6-v2",
                "semantic_threshold: 0.6",
                "semantic_device: cpu",
                "semantic_batch_size: 8",
                "num_workers: 0",
                "baseline_run:",
                "  label: baseline",
                f"  pred_jsonl: {baseline_path}",
                "oracle_runs:",
                "  - label: oracle_materialized",
                f"    pipeline_config: {pipeline_config}",
                "    overrides:",
                "      infer.limit: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_oracle_k_from_config(config_path)
    assert summary["oracle_run_count"] == 1
    assert summary["oracle_runs"][0]["materialized"] is True
    assert summary["oracle_runs"][0]["pred_jsonl"] == str(materialized_path)

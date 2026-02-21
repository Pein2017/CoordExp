from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.eval.confidence_postop import (
    CONFIDENCE_METHOD,
    ConfidencePostOpOptions,
    PRED_SCORE_SOURCE,
    PRED_SCORE_VERSION,
    options_from_config,
    paths_from_config,
    run_confidence_postop,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _bbox_record(
    *,
    image: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    raw_output_json: dict | None,
) -> dict:
    return {
        "image": image,
        "width": 1000,
        "height": 1000,
        "mode": "coord",
        "coord_mode": "pixel",
        "gt": [],
        "pred": [
            {
                "type": "bbox_2d",
                "points": [x1, y1, x2, y2],
                "desc": "box",
            }
        ],
        "raw_output_json": raw_output_json,
        "raw_special_tokens": [],
        "raw_ends_with_im_end": True,
        "errors": [],
    }


def _bbox_raw(x1: int, y1: int, x2: int, y2: int, *, desc: str = "box") -> dict:
    return {
        "objects": [
            {
                "bbox_2d": [
                    f"<|coord_{x1}|>",
                    f"<|coord_{y1}|>",
                    f"<|coord_{x2}|>",
                    f"<|coord_{y2}|>",
                ],
                "desc": desc,
            }
        ]
    }


def test_confidence_postop_end_to_end_with_deterministic_failure_reasons(
    tmp_path: Path,
) -> None:
    gt_vs_pred_path = tmp_path / "gt_vs_pred.jsonl"
    trace_path = tmp_path / "pred_token_trace.jsonl"
    pred_confidence_path = tmp_path / "pred_confidence.jsonl"
    scored_path = tmp_path / "gt_vs_pred_scored.jsonl"
    summary_path = tmp_path / "confidence_postop_summary.json"

    good_score_lp = math.log(0.2)

    records = [
        {
            "image": "img0.png",
            "width": 1000,
            "height": 1000,
            "mode": "coord",
            "coord_mode": "pixel",
            "gt": [],
            "pred": [
                {
                    "type": "bbox_2d",
                    "points": [100, 200, 300, 400],
                    "desc": "cat",
                },
                {
                    "type": "poly",
                    "points": [10, 10, 20, 10, 20, 20, 10, 20],
                    "desc": "poly",
                },
            ],
            "raw_output_json": {
                "objects": [
                    {
                        "bbox_2d": [
                            "<|coord_100|>",
                            "<|coord_200|>",
                            "<|coord_300|>",
                            "<|coord_400|>",
                        ],
                        "desc": "cat",
                    },
                    {
                        "poly": [10, 10, 20, 10, 20, 20, 10, 20],
                        "desc": "poly",
                    },
                ]
            },
            "raw_special_tokens": [],
            "raw_ends_with_im_end": True,
            "errors": [],
        },
        _bbox_record(
            image="img1.png",
            x1=120,
            y1=220,
            x2=320,
            y2=420,
            raw_output_json=_bbox_raw(120, 220, 320, 420),
        ),
        _bbox_record(
            image="img2.png",
            x1=130,
            y1=230,
            x2=330,
            y2=430,
            raw_output_json=_bbox_raw(130, 230, 330, 430),
        ),
        _bbox_record(
            image="img3.png",
            x1=140,
            y1=240,
            x2=340,
            y2=440,
            raw_output_json=_bbox_raw(141, 241, 341, 441),
        ),
        _bbox_record(
            image="img4.png",
            x1=150,
            y1=250,
            x2=350,
            y2=450,
            raw_output_json=None,
        ),
        _bbox_record(
            image="img5.png",
            x1=160,
            y1=260,
            x2=360,
            y2=460,
            raw_output_json=_bbox_raw(160, 260, 360, 460),
        ),
        _bbox_record(
            image="img6.png",
            x1=170,
            y1=270,
            x2=370,
            y2=470,
            raw_output_json=_bbox_raw(170, 270, 370, 470),
        ),
    ]
    _write_jsonl(gt_vs_pred_path, records)

    trace_records = [
        {
            "line_idx": 0,
            "generated_token_text": [
                "<|coord_100|>",
                "<|coord_200|>",
                "<|coord_300|>",
                "<|coord_400|>",
                "<|im_end|>",
            ],
            "token_logprobs": [
                good_score_lp,
                good_score_lp,
                good_score_lp,
                good_score_lp,
                -0.1,
            ],
        },
        {
            "line_idx": 2,
            "generated_token_text": [
                "<|coord_130|>",
                "<|coord_230|>",
                "<|coord_330|>",
                "<|coord_430|>",
            ],
            "token_logprobs": [-0.1, -0.1, -0.1],
        },
        {
            "line_idx": 3,
            "generated_token_text": [
                "<|coord_141|>",
                "<|coord_241|>",
                "<|coord_341|>",
                "<|coord_441|>",
            ],
            "token_logprobs": [-0.2, -0.2, -0.2, -0.2],
        },
        {
            "line_idx": 4,
            "generated_token_text": [
                "<|coord_150|>",
                "<|coord_250|>",
                "<|coord_350|>",
                "<|coord_450|>",
            ],
            "token_logprobs": [-0.2, -0.2, -0.2, -0.2],
        },
        {
            "line_idx": 5,
            "generated_token_text": [
                "<|coord_999|>",
                "<|coord_999|>",
                "<|coord_999|>",
                "<|coord_999|>",
            ],
            "token_logprobs": [-0.2, -0.2, -0.2, -0.2],
        },
        {
            "line_idx": 6,
            "generated_token_text": [
                "<|coord_170|>",
                "<|coord_270|>",
                "<|coord_370|>",
                "<|coord_470|>",
            ],
            "token_logprobs": [-0.2, float("nan"), -0.2, -0.2],
        },
    ]
    _write_jsonl(trace_path, trace_records)

    summary = run_confidence_postop(
        paths=paths_from_config(
            {
                "artifacts": {
                    "gt_vs_pred_jsonl": str(gt_vs_pred_path),
                    "pred_token_trace_jsonl": str(trace_path),
                    "pred_confidence_jsonl": str(pred_confidence_path),
                    "gt_vs_pred_scored_jsonl": str(scored_path),
                    "confidence_postop_summary_json": str(summary_path),
                }
            }
        )
    )

    assert summary["total_samples"] == 7
    assert summary["total_pred_objects"] == 8
    assert summary["kept_pred_objects"] == 1
    assert summary["dropped_pred_objects"] == 7
    assert summary["kept_fraction"] == pytest.approx(1.0 / 8.0)
    assert summary["pred_score_source"] == PRED_SCORE_SOURCE
    assert summary["pred_score_version"] == PRED_SCORE_VERSION
    assert summary["dropped_by_reason"] == {
        "unsupported_geometry_type": 1,
        "missing_trace": 1,
        "trace_len_mismatch": 1,
        "pred_alignment_mismatch": 1,
        "missing_coord_bins": 1,
        "missing_span": 1,
        "nonfinite_logprob": 1,
    }

    confidence_records = [json.loads(line) for line in pred_confidence_path.read_text(encoding="utf-8").splitlines()]
    assert len(confidence_records) == 7

    rec0_objs = confidence_records[0]["objects"]
    assert rec0_objs[0]["kept"] is True
    assert rec0_objs[0]["confidence"] == pytest.approx(0.2)
    assert rec0_objs[0]["score"] == pytest.approx(0.2)
    assert rec0_objs[0]["confidence_details"]["failure_reason"] is None
    assert rec0_objs[1]["kept"] is False
    assert rec0_objs[1]["confidence_details"]["failure_reason"] == "unsupported_geometry_type"

    assert confidence_records[1]["objects"][0]["confidence_details"]["failure_reason"] == "missing_trace"
    assert confidence_records[2]["objects"][0]["confidence_details"]["failure_reason"] == "trace_len_mismatch"
    assert confidence_records[3]["objects"][0]["confidence_details"]["failure_reason"] == "pred_alignment_mismatch"
    assert confidence_records[4]["objects"][0]["confidence_details"]["failure_reason"] == "missing_coord_bins"
    assert confidence_records[5]["objects"][0]["confidence_details"]["failure_reason"] == "missing_span"
    assert confidence_records[6]["objects"][0]["confidence_details"]["failure_reason"] == "nonfinite_logprob"

    scored_records = [json.loads(line) for line in scored_path.read_text(encoding="utf-8").splitlines()]
    assert len(scored_records) == 7
    assert scored_records[0]["pred_score_source"] == PRED_SCORE_SOURCE
    assert scored_records[0]["pred_score_version"] == PRED_SCORE_VERSION
    assert len(scored_records[0]["pred"]) == 1
    assert scored_records[0]["pred"][0]["score"] == pytest.approx(0.2)
    for record in scored_records[1:]:
        assert record["pred"] == []


def test_confidence_postop_fuses_geom_and_desc_scores(tmp_path: Path) -> None:
    gt_vs_pred_path = tmp_path / "gt_vs_pred.jsonl"
    trace_path = tmp_path / "pred_token_trace.jsonl"
    pred_confidence_path = tmp_path / "pred_confidence.jsonl"
    scored_path = tmp_path / "gt_vs_pred_scored.jsonl"
    summary_path = tmp_path / "confidence_postop_summary.json"

    record = _bbox_record(
        image="img_latest.png",
        x1=100,
        y1=200,
        x2=300,
        y2=400,
        raw_output_json=_bbox_raw(100, 200, 300, 400, desc="wine glass"),
    )
    record["pred"][0]["desc"] = "wine glass"
    _write_jsonl(gt_vs_pred_path, [record])

    generated_token_text = [
        "{\"",
        "objects",
        "\":",
        " [{\"",
        "desc",
        "\":",
        " \"",
        "wine",
        " glass",
        "\",",
        " \"",
        "bbox",
        "_",
        "2",
        "d",
        "\":",
        " [",
        "<|coord_100|>",
        ",",
        " ",
        "<|coord_200|>",
        ",",
        " ",
        "<|coord_300|>",
        ",",
        " ",
        "<|coord_400|>",
        "]}",
        "]}",
    ]
    token_logprobs = [0.0] * len(generated_token_text)
    token_logprobs[7] = math.log(0.5)
    token_logprobs[8] = math.log(0.25)
    token_logprobs[17] = math.log(0.2)
    token_logprobs[20] = math.log(0.2)
    token_logprobs[23] = math.log(0.2)
    token_logprobs[26] = math.log(0.2)
    _write_jsonl(
        trace_path,
        [
            {
                "line_idx": 0,
                "generated_token_text": generated_token_text,
                "token_logprobs": token_logprobs,
            }
        ],
    )

    summary = run_confidence_postop(
        paths=paths_from_config(
            {
                "artifacts": {
                    "gt_vs_pred_jsonl": str(gt_vs_pred_path),
                    "pred_token_trace_jsonl": str(trace_path),
                    "pred_confidence_jsonl": str(pred_confidence_path),
                    "gt_vs_pred_scored_jsonl": str(scored_path),
                    "confidence_postop_summary_json": str(summary_path),
                }
            }
        ),
        options=ConfidencePostOpOptions(
            fusion_w_geom=0.6,
            fusion_w_desc=0.4,
        ),
    )
    assert summary["pred_score_version"] == PRED_SCORE_VERSION
    assert summary["kept_pred_objects"] == 1

    confidence_records = [json.loads(line) for line in pred_confidence_path.read_text(encoding="utf-8").splitlines()]
    obj = confidence_records[0]["objects"][0]
    expected_geom = 0.2
    expected_desc = math.exp((math.log(0.5) + math.log(0.25)) / 2.0)
    expected_fusion = 0.6 * expected_geom + 0.4 * expected_desc

    assert obj["kept"] is True
    assert obj["score_geom"] == pytest.approx(expected_geom)
    assert obj["score_desc"] == pytest.approx(expected_desc)
    assert obj["score_fusion"] == pytest.approx(expected_fusion)
    assert obj["score"] == pytest.approx(expected_fusion)
    assert obj["confidence"] == pytest.approx(expected_fusion)
    assert obj["confidence_details"]["method"] == CONFIDENCE_METHOD
    assert obj["confidence_details"]["desc_span_token_indices"] == [7, 8]
    assert obj["confidence_details"]["fusion_weights"]["w_geom"] == pytest.approx(0.6)
    assert obj["confidence_details"]["fusion_weights"]["w_desc"] == pytest.approx(0.4)

    scored_records = [json.loads(line) for line in scored_path.read_text(encoding="utf-8").splitlines()]
    assert scored_records[0]["pred_score_version"] == PRED_SCORE_VERSION
    assert scored_records[0]["pred"][0]["score"] == pytest.approx(expected_fusion)


def test_confidence_postop_strict_desc_policy_drops_on_missing_desc_span(
    tmp_path: Path,
) -> None:
    gt_vs_pred_path = tmp_path / "gt_vs_pred.jsonl"
    trace_path = tmp_path / "pred_token_trace.jsonl"
    pred_confidence_path = tmp_path / "pred_confidence.jsonl"
    scored_path = tmp_path / "gt_vs_pred_scored.jsonl"
    summary_path = tmp_path / "confidence_postop_summary.json"

    record = _bbox_record(
        image="img_latest_strict.png",
        x1=100,
        y1=200,
        x2=300,
        y2=400,
        raw_output_json=_bbox_raw(100, 200, 300, 400, desc="cat"),
    )
    record["pred"][0]["desc"] = "cat"
    _write_jsonl(gt_vs_pred_path, [record])

    _write_jsonl(
        trace_path,
        [
            {
                "line_idx": 0,
                "generated_token_text": [
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_300|>",
                    "<|coord_400|>",
                ],
                "token_logprobs": [math.log(0.3)] * 4,
            }
        ],
    )

    summary = run_confidence_postop(
        paths=paths_from_config(
            {
                "artifacts": {
                    "gt_vs_pred_jsonl": str(gt_vs_pred_path),
                    "pred_token_trace_jsonl": str(trace_path),
                    "pred_confidence_jsonl": str(pred_confidence_path),
                    "gt_vs_pred_scored_jsonl": str(scored_path),
                    "confidence_postop_summary_json": str(summary_path),
                }
            }
        ),
        options=ConfidencePostOpOptions(
            desc_span_policy="strict",
        ),
    )
    assert summary["pred_score_version"] == PRED_SCORE_VERSION
    assert summary["kept_pred_objects"] == 0
    assert summary["dropped_by_reason"] == {"missing_desc_span": 1}

    confidence_records = [json.loads(line) for line in pred_confidence_path.read_text(encoding="utf-8").splitlines()]
    obj = confidence_records[0]["objects"][0]
    assert obj["kept"] is False
    assert obj["score"] is None
    assert obj["confidence_details"]["failure_reason"] == "missing_desc_span"

    scored_records = [json.loads(line) for line in scored_path.read_text(encoding="utf-8").splitlines()]
    assert scored_records[0]["pred"] == []
    assert scored_records[0]["pred_score_version"] == PRED_SCORE_VERSION


def test_paths_from_config_uses_run_dir_defaults(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    paths = paths_from_config({"artifacts": {"run_dir": str(run_dir)}})

    assert paths.gt_vs_pred_jsonl == run_dir / "gt_vs_pred.jsonl"
    assert paths.pred_token_trace_jsonl == run_dir / "pred_token_trace.jsonl"
    assert paths.pred_confidence_jsonl == run_dir / "pred_confidence.jsonl"
    assert paths.gt_vs_pred_scored_jsonl == run_dir / "gt_vs_pred_scored.jsonl"
    assert paths.confidence_postop_summary_json == run_dir / "confidence_postop_summary.json"


def test_options_from_config_rejects_removed_version_key() -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        options_from_config({"confidence": {"version": "legacy"}})


@pytest.mark.parametrize(
    ("trace_records", "error_pattern"),
    [
        (
            [
                {
                    "line_idx": 0,
                    "generated_token_text": ["<|coord_1|>"],
                    "token_logprobs": [-0.1],
                },
                {
                    "line_idx": 0,
                    "generated_token_text": ["<|coord_2|>"],
                    "token_logprobs": [-0.2],
                },
            ],
            "duplicates line_idx",
        ),
        (
            [
                {
                    "line_idx": -1,
                    "generated_token_text": ["<|coord_1|>"],
                    "token_logprobs": [-0.1],
                }
            ],
            "line_idx must be >= 0",
        ),
        (
            [
                {
                    "line_idx": 0,
                    "generated_token_text": "not-a-list",
                    "token_logprobs": [-0.1],
                }
            ],
            "generated_token_text must be a list\\[str\\]",
        ),
        (
            [
                {
                    "line_idx": 0,
                    "generated_token_text": [123],
                    "token_logprobs": [-0.1],
                }
            ],
            "generated_token_text\\[0\\] must be a string",
        ),
        (
            [
                {
                    "line_idx": 0,
                    "generated_token_text": ["<|coord_1|>"],
                    "token_logprobs": "not-a-list",
                }
            ],
            "token_logprobs must be a list\\[number\\]",
        ),
        (
            [
                {
                    "line_idx": 0,
                    "generated_token_text": ["<|coord_1|>"],
                    "token_logprobs": ["oops"],
                }
            ],
            "token_logprobs\\[0\\] must be numeric",
        ),
    ],
)
def test_confidence_postop_rejects_invalid_trace_sidecar(
    tmp_path: Path,
    trace_records: list[dict],
    error_pattern: str,
) -> None:
    gt_vs_pred_path = tmp_path / "gt_vs_pred.jsonl"
    trace_path = tmp_path / "pred_token_trace.jsonl"
    pred_confidence_path = tmp_path / "pred_confidence.jsonl"
    scored_path = tmp_path / "gt_vs_pred_scored.jsonl"
    summary_path = tmp_path / "confidence_postop_summary.json"

    _write_jsonl(
        gt_vs_pred_path,
        [
            {
                "image": "img.png",
                "width": 100,
                "height": 100,
                "mode": "coord",
                "coord_mode": "pixel",
                "gt": [],
                "pred": [],
                "raw_output_json": {"objects": []},
                "raw_special_tokens": [],
                "raw_ends_with_im_end": True,
                "errors": [],
            }
        ],
    )
    _write_jsonl(trace_path, trace_records)

    with pytest.raises(ValueError, match=error_pattern):
        run_confidence_postop(
            paths=paths_from_config(
                {
                    "artifacts": {
                        "gt_vs_pred_jsonl": str(gt_vs_pred_path),
                        "pred_token_trace_jsonl": str(trace_path),
                        "pred_confidence_jsonl": str(pred_confidence_path),
                        "gt_vs_pred_scored_jsonl": str(scored_path),
                        "confidence_postop_summary_json": str(summary_path),
                    }
                }
            )
        )

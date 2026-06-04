from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.analysis.sorted_random_no_newline_phenotype.fn_hint_runtime import (
    _candidate_scores_for_probe,
    _hint_text,
    _parse_generated_box,
    _slot_evidence_from_decode,
    _target_coord_box_for_case,
    run_real_fn_hint_probe,
)
from src.analysis.sorted_random_no_newline_phenotype.fn_probe import (
    build_fn_candidate_score_rows,
)
from src.analysis.sorted_random_no_newline_phenotype.status import (
    CONSTRAINT_POLICY,
    DECODE_POLICY,
    REAL_FN_HINT_RUNTIME_KIND,
)


def test_empty_fn_hint_shard_materializes_json_safe_runtime_artifacts(
    tmp_path: Path,
) -> None:
    fn_root = tmp_path / "fn_probe"
    fn_root.mkdir(parents=True)
    (fn_root / "fn_cases.jsonl").write_text("", encoding="utf-8")
    config = SimpleNamespace(
        artifact_root=tmp_path,
        sampling=SimpleNamespace(num_shards=8),
    )

    result = run_real_fn_hint_probe(
        config,
        shard_id=3,
        allow_overwrite=False,
        gpu_id="cpu-smoke",
    )

    assert result["runtime_kind"] == REAL_FN_HINT_RUNTIME_KIND
    assert result["decode_policy"] == DECODE_POLICY
    assert result["constraint_policy"] == CONSTRAINT_POLICY
    assert result["fn_cases"] == 0
    summary_path = tmp_path / "fn_probe" / "fn_hint_shards" / "shard_3_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    json.dumps(summary, allow_nan=False, sort_keys=True)
    assert summary["runtime_kind"] == REAL_FN_HINT_RUNTIME_KIND
    assert summary["gpu_id"] == "cpu-smoke"
    for name in ("probe_rows", "candidate_scores", "slot_evidence", "decode_rows"):
        path = tmp_path / "fn_probe" / "fn_hint_shards" / f"shard_3_{name}.jsonl"
        assert path.read_text(encoding="utf-8") == ""


def test_candidate_scores_fan_out_to_each_probe_id() -> None:
    base_scores = [
        {
            "fn_case_id": "case-1",
            "candidate_id": "candidate:0:person",
            "candidate_gt_idx": 7,
            "desc": "person",
            "role": "residual_same_desc",
            "score": -0.2,
        },
        {
            "fn_case_id": "case-1",
            "candidate_id": "candidate:1:chair",
            "candidate_gt_idx": None,
            "desc": "chair",
            "role": "hard_competitor",
            "score": -1.0,
        },
    ]

    rows = []
    for probe_id in (
        "case-1:rollout_prefix:none",
        "case-1:rollout_prefix:desc",
        "case-1:rollout_prefix:desc_x1",
        "case-1:rollout_prefix:desc_x1_y1",
    ):
        rows.extend(_candidate_scores_for_probe(base_scores, probe_id=probe_id))

    normalized = build_fn_candidate_score_rows(rows)

    assert len(normalized) == 8
    assert {
        row["probe_id"]
        for row in normalized
        if row["candidate_id"] == "candidate:0:person"
    } == {
        "case-1:rollout_prefix:none",
        "case-1:rollout_prefix:desc",
        "case-1:rollout_prefix:desc_x1",
        "case-1:rollout_prefix:desc_x1_y1",
    }
    assert all(
        row["candidate_rank"] == 1
        for row in normalized
        if row["candidate_id"] == "candidate:0:person"
    )


def test_hint_parse_reconstructs_seeded_coordinates_and_slot_axes() -> None:
    target = [100, 200, 300, 500]
    parsed, errors = _parse_generated_box(
        "desc_x1",
        "<|coord_210|><|coord_305|><|coord_490|>",
        target_box=target,
    )
    assert errors == []
    assert parsed == [100, 210, 305, 490]

    case = {
        "fn_case_id": "case-1",
        "fn_gt_idx": 7,
        "fn_bbox": target,
    }
    evidence = _slot_evidence_from_decode(
        case,
        probe_id="case-1:empty_prefix:desc_x1",
        hint_level="desc_x1",
        generated_box=parsed,
    )

    by_slot = {row["slot"]: row for row in evidence}
    assert set(by_slot) == {"x1", "y1", "x2", "y2"}
    assert by_slot["x1"]["axis_len"] == 200
    assert by_slot["x1"]["gt_value"] == 100
    assert by_slot["x1"]["peak_value"] == 100
    assert by_slot["x1"]["hinted_control"] is True
    assert by_slot["x1"]["model_predicted"] is False
    assert by_slot["y1"]["axis_len"] == 300
    assert by_slot["x2"]["axis_len"] == 200
    assert by_slot["y2"]["axis_len"] == 300
    assert by_slot["y1"]["gt_value"] == 200
    assert by_slot["y1"]["peak_value"] == 210
    assert by_slot["y1"]["hinted_control"] is False
    assert by_slot["y1"]["model_predicted"] is True


def test_pixel_fn_bbox_is_projected_to_coord_token_surface_for_hints() -> None:
    case = {
        "fn_case_id": "case-wide-image",
        "fn_gt_idx": 0,
        "fn_desc": "person",
        "fn_bbox": [977, 151, 1247, 821],
        "width": 1248,
        "height": 832,
        "coord_mode": "pixel",
    }

    target = _target_coord_box_for_case(case)
    hint = _hint_text(case, "desc_x1_y1")
    evidence = _slot_evidence_from_decode(
        case,
        probe_id="case-wide-image:empty_prefix:desc_x1_y1",
        hint_level="desc_x1_y1",
        generated_box=target,
    )

    assert target == [783, 181, 999, 987]
    assert "<|coord_783|>" in hint
    assert "<|coord_181|>" in hint
    assert "<|coord_1247|>" not in hint
    by_slot = {row["slot"]: row for row in evidence}
    assert by_slot["x1"]["gt_value"] == 783
    assert by_slot["x2"]["gt_value"] == 999
    assert by_slot["x1"]["axis_len"] == 216

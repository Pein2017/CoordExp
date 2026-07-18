from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.analyze_image2299_horizon_branch_value import (
    ArmSpec,
    analyze_arms,
    build_relabel_objects,
    classify_call,
    match_prediction,
    paired_bootstrap_difference,
    parse_arm_spec,
    resolve_arm_specs,
)


def _record() -> dict[str, object]:
    return {
        "image_id": 2299,
        "width": 100,
        "height": 100,
        "objects": [
            {"desc": "person", "pixel_box": [0, 0, 10, 10]},
            {"desc": "person", "pixel_box": [20, 0, 30, 10]},
            {"desc": "person", "pixel_box": [40, 0, 50, 10]},
            {"desc": "person", "pixel_box": [60, 0, 70, 10]},
            {"desc": "tie", "pixel_box": [0, 20, 10, 30]},
        ],
    }


def _prediction(box: list[int], category: str = "person", order: int = 0) -> dict[str, object]:
    return {"description": category, "bbox": box, "generated_order": order}


def _bundle(predictions: list[dict[str, object]], *, drops: list[dict[str, object]] | None = None, terminal: bool = False, seed: int = 1) -> dict[str, object]:
    return {
        "image_id": "2299",
        "sampling_seed": seed,
        "runtime": {"decode_mode": "sampled"},
        "parse_result": {
            "predictions": predictions,
            "dropped_predictions": drops or [],
            "dropped_prediction_count": len(drops or []),
            "parse_status": "accepted_with_drops" if drops else "accepted",
        },
        "horizon_projection": {
            "termination_classification": "natural_termination_before_horizon" if terminal else "token_limit_truncated",
            "stop_reason": "im_end" if terminal else "length",
        },
    }


def _write_arm(tmp_path: Path, name: str, bundle: dict[str, object]) -> Path:
    bundle_path = tmp_path / f"{name}-call.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    receipt_path = tmp_path / f"{name}-receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "native_sibling_branch_replay.shard_receipt.v1",
                "image_id": "2299",
                "calls": [
                    {
                        "bundle_path": str(bundle_path),
                        "decode_mode": "sampled",
                        "sampling_seed": int(bundle["sampling_seed"]),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return receipt_path


def test_same_category_iou_match_preserves_unresolved_margin() -> None:
    objects = build_relabel_objects(_record())
    matched = match_prediction(_prediction([0, 0, 10, 10]), objects)
    assert matched["matched"] is True
    assert matched["matched_category_rank"] == 0

    ambiguous = match_prediction(
        _prediction([0, 0, 10, 10]),
        [
            {**objects[0], "pixel_box": [0, 0, 10, 10]},
            {**objects[1], "pixel_box": [0, 0, 10, 10]},
        ],
    )
    assert ambiguous["matched"] is False
    assert ambiguous["reason"] == "top_match_margin_below_threshold"


def test_classify_call_tracks_parent_branch_and_suffix_repeats() -> None:
    objects = build_relabel_objects(_record())
    bundle = _bundle(
        [
            _prediction([0, 0, 10, 10], order=0),
            _prediction([60, 0, 70, 10], order=1),
            _prediction([40, 0, 50, 10], order=2),
            _prediction([40, 0, 50, 10], order=3),
            _prediction([0, 20, 10, 30], category="tie", order=4),
        ]
    )
    result = classify_call(bundle, arm=ArmSpec(Path("receipt.json"), 3, "owner0003"), relabel_objects=objects)
    assert result["matched_person_ranks"] == [0, 3, 2, 2]
    assert result["unique_new_person_count"] == 1
    assert result["parent_repeat_count"] == 1
    assert result["branch_repeat_count"] == 1
    assert result["within_suffix_repeat_count"] == 1
    assert result["tie_count"] == 0  # the fifth row is outside the horizon
    assert result["safety_count"] == 3


def test_classify_call_reports_malformed_and_terminal_outcomes() -> None:
    objects = build_relabel_objects(_record())
    result = classify_call(
        _bundle(
            [_prediction([40, 0, 50, 10])],
            drops=[{"generated_order": 1, "reason": "malformed_object_span"}],
            terminal=True,
        ),
        arm=ArmSpec(Path("receipt.json"), 3, "owner0003"),
        relabel_objects=objects,
    )
    assert result["unique_new_person_count"] == 1
    assert result["malformed_count"] == 1
    assert result["terminal_count"] == 0  # malformed parser status cannot claim a natural terminal
    assert result["unresolved_count"] == 0
    assert result["safety_count"] == 1


def test_paired_bootstrap_is_deterministic_and_signed_sampled_minus_greedy() -> None:
    left = {"1": 2.0, "2": 0.0, "3": 2.0}
    right = {"1": 1.0, "2": 0.0, "3": 1.0}
    first = paired_bootstrap_difference(left, right, replicates=500)
    second = paired_bootstrap_difference(left, right, replicates=500)
    assert first == second
    assert first["point"] == pytest.approx(2 / 3)
    assert first["paired_unit"] == "sampling_seed"
    with pytest.raises(ValueError, match="identical seed sets"):
        paired_bootstrap_difference({"1": 1.0}, {"2": 0.0}, replicates=10)


def test_explicit_arm_mapping_and_end_to_end_paired_comparison(tmp_path: Path) -> None:
    source = tmp_path / "val.coord.jsonl"
    source.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    receipts = {
        "owner0003": _write_arm(tmp_path, "owner0003", _bundle([_prediction([40, 0, 50, 10])], seed=11)),
        "owner0004": _write_arm(tmp_path, "owner0004", _bundle([_prediction([40, 0, 50, 10])], seed=11)),
        "owner0006": _write_arm(tmp_path, "owner0006", _bundle([_prediction([40, 0, 50, 10])], seed=11)),
        "owner0012": _write_arm(tmp_path, "owner0012", _bundle([_prediction([0, 0, 10, 10])], seed=11)),
    }
    arms = resolve_arm_specs(
        [],
        [
            f"{receipts['owner0003']}:3:owner0003",
            f"{receipts['owner0004']}:2:owner0004",
            f"{receipts['owner0006']}:4:owner0006",
            f"{receipts['owner0012']}:14:owner0012:greedy",
        ],
    )
    payload = analyze_arms(arms, source_jsonl=source, bootstrap_replicates=200)
    assert payload["greedy_arm_label"] == "owner0012"
    assert set(payload["comparisons"]) == {"owner0003", "owner0004", "owner0006"}
    comparison = payload["comparisons"]["owner0003"]
    assert comparison["unique_new_person_difference"]["point"] == pytest.approx(1.0)
    assert comparison["safety_count_difference"]["point"] == pytest.approx(-1.0)
    assert all(arm["aggregate"]["sampled_seed_count"] == 1 for arm in payload["arms"])


def test_missing_explicit_arm_mapping_fails_fast(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"image_id": "2299", "calls": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit branch rank"):
        resolve_arm_specs([receipt])


def test_arm_parser_requires_rank_and_label() -> None:
    parsed = parse_arm_spec("/tmp/a.json:14:owner0012:greedy")
    assert parsed.branch_rank == 14
    assert parsed.parent_greedy_branch is True
    with pytest.raises(ValueError, match="RECEIPT:BRANCH_RANK:LABEL"):
        parse_arm_spec("/tmp/a.json")

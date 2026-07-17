from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.research.admit_native_sibling_first_rows import (
    analyze_merged_receipt,
    main,
    match_first_row_to_ledger,
)


def _prediction(*, description: str = "cup", box: list[int] | None = None, token: int = 101) -> dict:
    box = box or [0, 0, 10, 10]
    return {
        "bbox": box,
        "bbox_format": "xyxy",
        "description": description,
        "generated_order": 0,
        "raw_span_text": f"<|object_ref_start|>{description}<|object_ref_end|><row-{token}>",
        "object_span_id": f"row-{token}",
    }


def _bundle(path: Path, *, seed: int | None, token: int = 101, description: str = "cup", box: list[int] | None = None, mode: str = "sampled") -> None:
    value = {
        "schema_version": "native_sibling_first_row_test-bundle.v1",
        "image_id": "1",
        "request_id": f"request-{seed if seed is not None else 'greedy'}",
        "sampling_seed": seed,
        "runtime": {"decode_mode": mode},
        "complete_row_spans": [[0, 3]],
        "raw_generated_token_ids": [151646, token, 151649],
        "parse_result": {
            "parse_status": "accepted",
            "predictions": [_prediction(description=description, box=box, token=token)],
        },
        "decode_result": {"generated_token_ids": [151646, token, 151649]},
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def _ledger(path: Path) -> None:
    rows = [
        {
            "final_state": "accepted",
            "image_id": 1,
            "normalized_category_name": "Cup",
            "object_identifier": "owner-a",
            "source_canvas_box_xyxy": [0, 0, 10, 10],
        },
        {
            "final_state": "accepted",
            "image_id": 1,
            "normalized_category_name": "cup",
            "object_identifier": "owner-b",
            "source_canvas_box_xyxy": [20, 20, 30, 30],
        },
        {
            "final_state": "accepted",
            "image_id": 2,
            "normalized_category_name": "cup",
            "object_identifier": "wrong-image",
            "source_canvas_box_xyxy": [0, 0, 10, 10],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_category_and_image_gate_are_strict_but_case_whitespace_is_normalized() -> None:
    ledger = [
        {
            "final_state": "accepted",
            "image_id": 1,
            "normalized_category_name": "  Cup ",
            "object_identifier": "owner-a",
            "source_canvas_box_xyxy": [0, 0, 10, 10],
        },
        {
            "final_state": "accepted",
            "image_id": 2,
            "normalized_category_name": "cup",
            "object_identifier": "wrong-image",
            "source_canvas_box_xyxy": [0, 0, 10, 10],
        },
    ]
    unique = match_first_row_to_ledger(
        prediction={"description": " CUP ", "bbox": [0, 0, 10, 10]}, image_id="1", ledger=ledger
    )
    assert unique["status"] == "unique"
    assert unique["owner_id"] == "owner-a"
    no_cross_image = match_first_row_to_ledger(
        prediction={"description": "cup", "bbox": [0, 0, 10, 10]}, image_id="3", ledger=ledger
    )
    assert no_cross_image["status"] == "unmatched"
    assert no_cross_image["candidate_count"] == 0


def test_top_second_margin_at_boundary_is_ambiguous() -> None:
    ledger = [
        {
            "final_state": "accepted",
            "image_id": 1,
            "normalized_category_name": "cup",
            "object_identifier": "a",
            "source_canvas_box_xyxy": [0, 0, 10, 10],
        },
        {
            "final_state": "accepted",
            "image_id": 1,
            "normalized_category_name": "cup",
            "object_identifier": "b",
            "source_canvas_box_xyxy": [0, 0, 9.5, 10],
        },
    ]
    result = match_first_row_to_ledger(
        prediction={"description": "cup", "bbox": [0, 0, 10, 10]}, image_id="1", ledger=ledger
    )
    assert result["best_iou"] == pytest.approx(1.0)
    assert result["second_iou"] == pytest.approx(0.95)
    assert result["top_second_iou_margin"] == pytest.approx(0.05)
    assert result["status"] == "ambiguous"
    assert result["owner_id"] is None


def test_owner_gate_requires_denominator_support_and_three_exact_variants(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    _ledger(ledger_path)
    calls = []
    for index in range(32):
        bundle_path = tmp_path / f"bundle-{index}.json"
        # Three exact token variants are intentionally repeated across support.
        token = 101 + index % 3
        _bundle(bundle_path, seed=9007199254740993 + index, token=token)
        calls.append(
            {
                "decode_mode": "sampled",
                "sampling_seed": 9007199254740993 + index,
                "request_id": f"request-{index}",
                "bundle_path": str(bundle_path),
            }
        )
    # A greedy control is not eligible for support, even when it matches.
    greedy_path = tmp_path / "greedy.json"
    _bundle(greedy_path, seed=None, token=999, mode="greedy")
    calls.append({"decode_mode": "greedy", "sampling_seed": None, "request_id": "greedy", "bundle_path": str(greedy_path)})
    merged_path = tmp_path / "merged.json"
    merged_path.write_text(json.dumps({"calls": calls}), encoding="utf-8")
    manifest, variants = analyze_merged_receipt(
        merged_path=merged_path,
        ledger_path=ledger_path,
        expected_samples=32,
    )
    assert len(variants) == 33
    assert manifest["owner_admission_ready"] is True
    owner = next(item for item in manifest["owner_groups"] if item["owner_id"] == "owner-a")
    assert owner["support_count"] == 32
    assert owner["distinct_exact_row_hash_count"] == 3
    assert owner["admitted"] is True
    assert manifest["greedy_control"][0]["owner_admitted"] is True
    # Uncovered status is deliberately not claimed without explicit coverage evidence.
    assert manifest["greedy_control"][0]["owner_uncovered"] is None
    assert all(isinstance(item["sampling_seed"], int) for item in variants if item["decode_mode"] == "sampled")
    assert variants[0]["sampling_seed"] >= 9007199254740993


def test_two_sample_smoke_cannot_claim_three_of_thirty_two_admission(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    _ledger(ledger_path)
    calls = []
    for index in range(2):
        bundle_path = tmp_path / f"bundle-{index}.json"
        _bundle(bundle_path, seed=11 + index)
        calls.append({"decode_mode": "sampled", "sampling_seed": 11 + index, "request_id": str(index), "bundle_path": str(bundle_path)})
    merged_path = tmp_path / "merged.json"
    merged_path.write_text(json.dumps({"calls": calls}), encoding="utf-8")
    manifest, _ = analyze_merged_receipt(merged_path=merged_path, ledger_path=ledger_path, expected_samples=32)
    assert manifest["sample_count_matches_expected"] is False
    assert manifest["owner_admission_ready"] is False
    assert manifest["admitted_owner_ids"] == []


def test_cli_writes_jsonl_and_csv_without_float_seed_loss(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    _ledger(ledger_path)
    bundle_path = tmp_path / "bundle.json"
    exact_seed = 9007199254740993
    _bundle(bundle_path, seed=exact_seed)
    merged_path = tmp_path / "merged.json"
    merged_path.write_text(json.dumps({"calls": [{"decode_mode": "sampled", "sampling_seed": exact_seed, "request_id": "r", "bundle_path": str(bundle_path)}]}), encoding="utf-8")
    output = tmp_path / "manifest.json"
    assert main([
        "--merged-receipt", str(merged_path),
        "--ledger", str(ledger_path),
        "--output", str(output),
    ]) == 0
    jsonl = tmp_path / "manifest-variants.jsonl"
    csv_path = tmp_path / "manifest-variants.csv"
    row = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert row["sampling_seed"] == exact_seed
    assert row["sampling_seed_decimal"] == str(exact_seed)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["sampling_seed"] == str(exact_seed)
    assert csv_row["sampling_seed_decimal"] == str(exact_seed)

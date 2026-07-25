from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest
from PIL import Image

from scripts.research.build_heldout_owner_churn_review_packet import (
    ENTITY_CATEGORY_OPTIONS,
    GEOMETRY_OPTIONS,
    ReviewPacketContractError,
    build_packet,
)


REAL_LEDGER = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-24-prefix-local-and-on-policy-owner-set-training/"
    "owner-comparisons-transition-step36-transfer-max3084-matched-b4-v2/"
    "heldout-source-vs-transition-step36.json"
)
GAIN_REF = ("coco2017_train_000000003442", 1)
LOSS_REF = ("coco2017_train_000000004129", 3)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _raw_span(description: str, coord_bins: list[int]) -> str:
    coords = "".join(f"<|coord_{value}|>" for value in coord_bins)
    return (
        f"<|object_ref_start|>{description}<|object_ref_end|>"
        f"<|box_start|>{coords}<|box_end|>"
    )


def _gt(category: str, box_pixel: tuple[int, int, int, int], owner: str) -> dict[str, Any]:
    bins = [value * 10 for value in box_pixel]
    return {
        "bbox": bins,
        "description": category,
        "object_id": owner,
        "metadata": {
            "source": {
                "bbox_2d": [f"<|coord_{value}|>" for value in bins],
                "category_name": category,
                "coco_ann_id": int(owner.split("-")[-1]),
            }
        },
    }


def _row(
    *,
    row_id: str,
    row_index: int,
    image_path: Path,
    gt: list[dict[str, Any]],
    events: list[tuple[str, str, tuple[int, int, int, int]]],
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    raw_parts: list[str] = []
    char_start = 0
    for generated_order, (kind, description, bbox) in enumerate(events):
        coord_bins = [value * 10 for value in bbox]
        raw_span = _raw_span(description, coord_bins)
        common = {
            "generated_order": generated_order,
            "object_span_id": f"{row_id}:span-{generated_order}",
            "char_start": char_start,
            "char_end": char_start + len(raw_span),
            "raw_span_text": raw_span,
            "raw_span_sha256": hashlib.sha256(raw_span.encode()).hexdigest(),
        }
        if kind == "accepted":
            accepted.append(
                {
                    **common,
                    "bbox": list(bbox),
                    "coord_bins": coord_bins,
                    "description": description,
                }
            )
        elif kind == "dropped":
            dropped.append(
                {
                    **common,
                    "raw_text": raw_span,
                    "reason": "geometry_invalid",
                    "context": {"bbox": coord_bins},
                }
            )
        else:  # pragma: no cover - fixture programming error
            raise AssertionError(kind)
        raw_parts.append(raw_span)
        char_start += len(raw_span)
    return {
        "row_id": row_id,
        "example_id": row_id,
        "row_index": row_index,
        "image_path": str(image_path.resolve()),
        "image_width": 100,
        "image_height": 100,
        "gt": gt,
        "pred": accepted,
        "valid_prediction_count": len(accepted),
        "dropped_prediction_count": len(dropped),
        "dropped_predictions": dropped,
        "raw_decode_text": "".join(raw_parts) + "<|im_end|>",
        "parse_status": "accepted",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> Path:
    gain_row_id = "coco2017_train_000000000001"
    loss_row_id = "coco2017_train_000000000002"
    gain_image = tmp_path / "images" / "000000000001.jpg"
    loss_image = tmp_path / "images" / "000000000002.jpg"
    gain_image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), (230, 230, 230)).save(gain_image, format="JPEG")
    Image.new("RGB", (100, 100), (210, 220, 230)).save(loss_image, format="JPEG")

    # This is the comparator's cardinality-first counterexample: highest-IoU
    # greedy takes GT-0/pred-0 and strands GT-1, while the global assignment
    # matches GT-0/pred-1 and GT-1/pred-0.
    gain_gt = [
        _gt("person", (22, 31, 83, 72), "owner-10"),
        _gt("person", (29, 10, 79, 71), "owner-11"),
    ]
    loss_gt = [
        _gt("chair", (5, 5, 15, 15), "owner-20"),
        _gt("dog", (20, 5, 30, 15), "owner-21"),
        _gt("boat", (35, 5, 50, 15), "owner-22"),
        _gt("person", (70, 70, 90, 90), "owner-23"),
    ]
    arm_a_rows = [
        _row(
            row_id=gain_row_id,
            row_index=0,
            image_path=gain_image,
            gt=gain_gt,
            events=[("accepted", "person", (27, 37, 93, 87))],
        ),
        _row(
            row_id=loss_row_id,
            row_index=1,
            image_path=loss_image,
            gt=loss_gt,
            events=[("accepted", "person", (70, 70, 90, 90))],
        ),
    ]
    arm_b_rows = [
        _row(
            row_id=gain_row_id,
            row_index=0,
            image_path=gain_image,
            gt=gain_gt,
            events=[
                ("accepted", "person", (31, 27, 81, 88)),
                ("dropped", "person", (60, 20, 50, 30)),
                ("accepted", "person", (27, 37, 93, 87)),
            ],
        ),
        _row(
            row_id=loss_row_id,
            row_index=1,
            image_path=loss_image,
            gt=loss_gt,
            events=[],
        ),
    ]
    arm_a_path = tmp_path / "arm_a" / "gt_vs_pred.jsonl"
    arm_b_path = tmp_path / "arm_b" / "gt_vs_pred.jsonl"
    _write_jsonl(arm_a_path, arm_a_rows)
    _write_jsonl(arm_b_path, arm_b_rows)
    ledger = {
        "inputs": {
            "arm_a": {"path": str(arm_a_path.resolve()), "sha256": _sha256(arm_a_path)},
            "arm_b": {"path": str(arm_b_path.resolve()), "sha256": _sha256(arm_b_path)},
            "cohort": {"selection": "all_artifact_rows", "included_row_ids": None},
        },
        "policy": {
            "delta_convention": "arm_b_minus_arm_a",
            "match_iou_threshold": 0.5,
            "unmatched_predictions_are_not_hallucinations": True,
        },
        "common_owner_geometry": {
            "arm_a_only_owner_count": 1,
            "arm_a_only_owner_refs": [
                {"row_id": loss_row_id, "owner_index": 3}
            ],
            "arm_b_only_owner_count": 1,
            "arm_b_only_owner_refs": [
                {"row_id": gain_row_id, "owner_index": 1}
            ],
        },
    }
    ledger_path = tmp_path / "comparison.json"
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger_path


def _reviewer_files(root: Path) -> dict[str, bytes]:
    reviewer = root / "reviewer"
    return {
        str(path.relative_to(reviewer)): path.read_bytes()
        for path in sorted(reviewer.rglob("*"))
        if path.is_file()
    }


def test_builds_deterministic_blinded_packet_with_global_attribution(tmp_path: Path) -> None:
    ledger_path = _fixture(tmp_path)
    output = tmp_path / "packet"
    receipt = build_packet(ledger_path, output)
    assert receipt["selected_case_count"] == 2
    assert (output / "private" / "original_comparison_ledger.json").read_bytes() == ledger_path.read_bytes()

    manifest = _json(output / "reviewer" / "manifest.json")
    assert manifest["case_count"] == 2
    assert manifest["review_schema"]["entity_category"]["allowed_values"] == list(
        ENTITY_CATEGORY_OPTIONS
    )
    assert manifest["review_schema"]["geometry"]["allowed_values"] == list(
        GEOMETRY_OPTIONS
    )
    for item in manifest["cases"]:
        case = _json(output / "reviewer" / item["case_file"])
        assert case["blinded"] is True
        assert Image.open(output / "reviewer" / item["full_image"]).size == (1800, 830)
        assert Image.open(output / "reviewer" / item["enlarged_crop"]).size == (1800, 830)
        for view in case["views"].values():
            for pred in view["predictions"]:
                status = pred["official_attribution"]["status"]
                assert status in {"matched", "unmatched_unresolved"}

    reviewer_text = "\n".join(
        value.decode("utf-8")
        for name, value in _reviewer_files(output).items()
        if Path(name).suffix in {".json", ".jsonl", ".md"}
    )
    assert re.search(r"\b(?:source|transition|gain|loss)\b", reviewer_text, re.I) is None

    unblinding = _json(output / "private" / "unblinding.json")
    by_row = {case["row_id"]: case for case in unblinding["cases"]}
    gain = by_row["coco2017_train_000000000001"]
    assert gain["physical_owner_change_direction"] == "gain"
    assert gain["authoritative_attribution"]["arm_a"]["target"]["status"] == "unmatched_unresolved"
    assert gain["authoritative_attribution"]["arm_b"]["target"]["prediction_index"] == 0
    assert {
        (match[0], match[1])
        for match in gain["authoritative_attribution"]["arm_b"]["matches"]
    } == {(0, 1), (1, 0)}
    gain_case = _json(output / "reviewer" / "cases" / f"{gain['case_id']}.json")
    arm_b_view = next(
        view_name for view_name, arm in gain["views"].items() if arm == "arm_b"
    )
    assert [
        item["raw_output_order"] for item in gain_case["views"][arm_b_view]["predictions"]
    ] == [0, 2]
    assert [
        item["raw_output_order"]
        for item in gain_case["views"][arm_b_view]["raw_output_events"]
    ] == [0, 1, 2]

    loss = by_row["coco2017_train_000000000002"]
    assert loss["physical_owner_change_direction"] == "loss"
    assert loss["authoritative_attribution"]["arm_a"]["target"]["prediction_index"] == 0
    assert loss["authoritative_attribution"]["arm_b"]["target"]["status"] == "unmatched_unresolved"

    second_output = tmp_path / "packet_again"
    build_packet(ledger_path, second_output)
    assert _reviewer_files(output) == _reviewer_files(second_output)
    with pytest.raises(ReviewPacketContractError, match="immutable"):
        build_packet(ledger_path, output)


def _rewrite_arm_b(
    ledger_path: Path,
    mutate: Callable[[list[dict[str, Any]]], None],
    *,
    update_digest: bool = True,
) -> None:
    ledger = _json(ledger_path)
    arm_b_path = Path(ledger["inputs"]["arm_b"]["path"])
    rows = [json.loads(line) for line in arm_b_path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    _write_jsonl(arm_b_path, rows)
    if update_digest:
        ledger["inputs"]["arm_b"]["sha256"] = _sha256(arm_b_path)
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


@pytest.mark.parametrize("failure", ["row", "gt", "image", "reference_set", "prediction_index"])
def test_fails_closed_on_provenance_mismatch(tmp_path: Path, failure: str) -> None:
    ledger_path = _fixture(tmp_path)
    if failure == "row":
        _rewrite_arm_b(
            ledger_path,
            lambda rows: rows[0].__setitem__("example_id", "wrong-row"),
        )
        message = "row|example_id"
    elif failure == "gt":
        _rewrite_arm_b(
            ledger_path,
            lambda rows: rows[0]["gt"][0].__setitem__("object_id", "owner-999"),
        )
        message = "GT provenance"
    elif failure == "image":
        _rewrite_arm_b(
            ledger_path,
            lambda rows: rows[0].__setitem__("image_path", rows[1]["image_path"]),
        )
        message = "image provenance"
    elif failure == "reference_set":
        ledger = _json(ledger_path)
        ledger["common_owner_geometry"]["arm_b_only_owner_refs"][0]["owner_index"] = 0
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        message = "reference set"
    else:
        _rewrite_arm_b(
            ledger_path,
            lambda rows: rows[0]["pred"][1].__setitem__("generated_order", 0),
        )
        message = "generated_order|prediction-index"

    output = tmp_path / "packet"
    with pytest.raises(ReviewPacketContractError, match=message):
        build_packet(ledger_path, output)
    assert not output.exists()


def test_fails_closed_on_artifact_digest_mismatch(tmp_path: Path) -> None:
    ledger_path = _fixture(tmp_path)
    _rewrite_arm_b(ledger_path, lambda rows: rows[0].__setitem__("parse_status", "changed"), update_digest=False)
    with pytest.raises(ReviewPacketContractError, match="digest mismatch"):
        build_packet(ledger_path, tmp_path / "packet")


@pytest.mark.skipif(not REAL_LEDGER.is_file(), reason="Phase Zero shared artifacts are unavailable")
def test_real_two_case_preflight_attribution_and_blinding(tmp_path: Path) -> None:
    output = tmp_path / "real_packet"
    build_packet(
        REAL_LEDGER,
        output,
        case_refs=[GAIN_REF, LOSS_REF],
        expected_total_refs=85,
    )
    unblinding = _json(output / "private" / "unblinding.json")
    by_row = {case["row_id"]: case for case in unblinding["cases"]}

    gain = by_row[GAIN_REF[0]]
    assert gain["owner_index"] == 1
    assert gain["physical_owner_change_direction"] == "gain"
    assert gain["authoritative_attribution"]["arm_a"]["target"] == {
        "status": "unmatched_unresolved"
    }
    assert gain["authoritative_attribution"]["arm_b"]["target"] == {
        "status": "matched",
        "prediction_index": 2,
        "intersection_over_union": pytest.approx(0.5307125307125307),
    }

    loss = by_row[LOSS_REF[0]]
    assert loss["owner_index"] == 3
    assert loss["physical_owner_change_direction"] == "loss"
    assert loss["authoritative_attribution"]["arm_a"]["target"] == {
        "status": "matched",
        "prediction_index": 6,
        "intersection_over_union": pytest.approx(0.6967741935483871),
    }
    assert loss["authoritative_attribution"]["arm_b"]["target"] == {
        "status": "unmatched_unresolved"
    }

    reviewer_text = "\n".join(
        value.decode("utf-8")
        for name, value in _reviewer_files(output).items()
        if Path(name).suffix in {".json", ".jsonl", ".md"}
    )
    assert re.search(r"\b(?:source|transition|gain|loss)\b", reviewer_text, re.I) is None

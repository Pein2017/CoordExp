#!/usr/bin/env python3
"""Summarize paired first-action receipts for a prefix-state factorial."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-object",
        action="append",
        default=[],
        help='JSON object with "object_id", "description", and "bbox_xyxy".',
    )
    parser.add_argument("--baseline-condition", default="no-appended-row")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _receipt_rows(receipt: Mapping[str, object]) -> list[dict[str, object]]:
    forced_rows = receipt.get("first_rows")
    if isinstance(forced_rows, list):
        return [
            {
                "request_id": row["request_id"],
                "action": row["first_free_action"],
            }
            for row in forced_rows
        ]
    ordinary_rows = receipt.get("first_row_projection")
    if isinstance(ordinary_rows, list):
        return [
            {
                "request_id": row["request_id"],
                "action": row["first_action"],
            }
            for row in ordinary_rows
        ]
    raise ValueError("receipt lacks first-action rows")


def _classify_action(
    action: Mapping[str, object],
    references: Sequence[Mapping[str, object]],
    *,
    iou_threshold: float,
) -> str:
    status = str(action.get("status"))
    if status != "valid_row":
        return status
    description = str(action.get("description", ""))
    box = action.get("bbox_xyxy")
    if not isinstance(box, list) or len(box) != 4:
        return "valid_row_without_box"
    compatible = [
        reference
        for reference in references
        if str(reference.get("description")) == description
    ]
    if not compatible:
        return f"other_valid:{description}"
    best = max(
        compatible,
        key=lambda reference: _box_iou(box, reference["bbox_xyxy"]),
    )
    best_iou = _box_iou(box, best["bbox_xyxy"])
    if best_iou < iou_threshold:
        return f"other_valid:{description}"
    return str(best["object_id"])


def _action_token_ids(action: Mapping[str, object]) -> tuple[int, ...]:
    values = action.get("token_ids")
    return tuple(int(value) for value in values) if isinstance(values, list) else ()


def summarize(
    condition_root: Path,
    references: Sequence[Mapping[str, object]],
    *,
    baseline_condition: str,
    iou_threshold: float,
) -> dict[str, object]:
    receipts: dict[str, dict[str, object]] = {}
    rows: dict[str, list[dict[str, object]]] = {}
    for path in sorted(condition_root.expanduser().resolve(strict=True).glob("*/receipt.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        condition = path.parent.name
        receipts[condition] = receipt
        rows[condition] = _receipt_rows(receipt)
    if baseline_condition not in rows:
        raise ValueError(f"missing baseline condition: {baseline_condition}")
    baseline_rows = rows[baseline_condition]
    baseline_seeds = receipts[baseline_condition].get("sampling_seeds")
    conditions: dict[str, object] = {}
    for condition, condition_rows in rows.items():
        labels = [
            _classify_action(row["action"], references, iou_threshold=iou_threshold)
            for row in condition_rows
        ]
        seeds = receipts[condition].get("sampling_seeds")
        sample_equal = []
        if len(condition_rows) == len(baseline_rows):
            sample_equal = [
                _action_token_ids(condition_rows[index]["action"])
                == _action_token_ids(baseline_rows[index]["action"])
                for index in range(1, len(condition_rows))
            ]
        boundary = receipts[condition].get("boundary_evidence")
        conditions[condition] = {
            "greedy_label": labels[0],
            "sample_label_counts": dict(sorted(Counter(labels[1:]).items())),
            "paired_labels": labels,
            "sampling_seeds": seeds,
            "seeds_equal_to_baseline": seeds == baseline_seeds,
            "sample_first_row_token_equal_to_baseline": sample_equal,
            "sample_first_row_token_equal_to_baseline_count": sum(sample_equal),
            "first_actions": [row["action"] for row in condition_rows],
            "boundary_evidence": boundary,
        }
    return {
        "schema_version": "prefix_state_phrase_geometry_factorial.summary.v1",
        "condition_root": str(condition_root.expanduser().resolve()),
        "baseline_condition": baseline_condition,
        "iou_threshold": iou_threshold,
        "reference_objects": list(references),
        "condition_count": len(conditions),
        "conditions": conditions,
    }


def main() -> int:
    args = build_parser().parse_args()
    references = [json.loads(value) for value in args.reference_object]
    result = summarize(
        args.condition_root,
        references,
        baseline_condition=args.baseline_condition,
        iou_threshold=args.iou_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

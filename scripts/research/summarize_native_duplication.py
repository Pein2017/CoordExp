#!/usr/bin/env python
"""Summarize duplicate bursts in native Qwen grounding output streams."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


NEAR_DUPLICATE_IOU = 0.90
SEVERE_EXACT_EXCESS = 10
SEVERE_CONSECUTIVE_RUN = 5
JSON_DECODER = json.JSONDecoder()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()
    output = args.output or artifact_dir / "duplication_summary.json"
    rows = _read_jsonl(artifact_dir / "gt_vs_pred.jsonl")
    summary = summarize_rows(rows)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print(f"summary: {output}")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_row = [_summarize_row(row) for row in rows]
    raw_object_count = sum(row["raw_object_count"] for row in per_row)
    exact_excess = sum(row["exact_duplicate_excess"] for row in per_row)
    near_pairs = sum(row["near_duplicate_pair_count"] for row in per_row)
    affected = [row for row in per_row if row["duplicate_affected"]]
    severe = [row for row in per_row if row["severe_burst"]]
    truncated = [row for row in per_row if row["stop_reason"] == "length"]
    return {
        "policy": {
            "object_source": "complete native bbox_2d JSON objects scanned from raw_decode_text",
            "exact_identity": "same label and same integer norm1000 bbox",
            "near_duplicate": f"same label and bbox IoU >= {NEAR_DUPLICATE_IOU:.2f}",
            "severe_burst": (
                f"exact duplicate excess >= {SEVERE_EXACT_EXCESS} or consecutive "
                f"near-duplicate run >= {SEVERE_CONSECUTIVE_RUN}"
            ),
        },
        "row_count": len(per_row),
        "raw_object_count": raw_object_count,
        "exact_duplicate_excess": exact_excess,
        "exact_duplicate_excess_rate": _ratio(exact_excess, raw_object_count),
        "near_duplicate_pair_count": near_pairs,
        "duplicate_affected_row_count": len(affected),
        "duplicate_affected_row_rate": _ratio(len(affected), len(per_row)),
        "severe_burst_row_count": len(severe),
        "severe_burst_row_rate": _ratio(len(severe), len(per_row)),
        "truncated_row_count": len(truncated),
        "truncated_severe_burst_row_count": sum(row["severe_burst"] for row in truncated),
        "max_exact_multiplicity": max(
            (row["max_exact_multiplicity"] for row in per_row), default=0
        ),
        "max_consecutive_near_duplicate_run": max(
            (row["max_consecutive_near_duplicate_run"] for row in per_row), default=0
        ),
        "worst_rows": sorted(
            per_row,
            key=lambda row: (
                row["severe_burst"],
                row["exact_duplicate_excess"],
                row["max_consecutive_near_duplicate_run"],
                row["near_duplicate_pair_count"],
            ),
            reverse=True,
        )[:10],
    }


def _summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    objects = _scan_objects(str(row.get("raw_decode_text", "")))
    identities = Counter((obj["label"], tuple(obj["bbox_2d"])) for obj in objects)
    exact_excess = sum(count - 1 for count in identities.values())
    near_pairs = 0
    for index, left in enumerate(objects):
        for right in objects[index + 1 :]:
            if left["label"] == right["label"] and _iou(left["bbox_2d"], right["bbox_2d"]) >= NEAR_DUPLICATE_IOU:
                near_pairs += 1
    max_run = _max_consecutive_near_run(objects)
    severe = exact_excess >= SEVERE_EXACT_EXCESS or max_run >= SEVERE_CONSECUTIVE_RUN
    return {
        "row_id": row.get("row_id"),
        "row_index": row.get("row_index"),
        "stop_reason": row.get("decode_stop_reason"),
        "raw_object_count": len(objects),
        "exact_duplicate_excess": exact_excess,
        "near_duplicate_pair_count": near_pairs,
        "max_exact_multiplicity": max(identities.values(), default=0),
        "max_consecutive_near_duplicate_run": max_run,
        "duplicate_affected": bool(exact_excess or near_pairs),
        "severe_burst": severe,
    }


def _scan_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            value, end = JSON_DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = end
        if not isinstance(value, dict) or set(value) != {"bbox_2d", "label"}:
            continue
        bbox = value["bbox_2d"]
        label = value["label"]
        if (
            isinstance(label, str)
            and isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(coord, int) and not isinstance(coord, bool) for coord in bbox)
        ):
            objects.append({"label": label, "bbox_2d": bbox})
    return objects


def _max_consecutive_near_run(objects: list[dict[str, Any]]) -> int:
    if not objects:
        return 0
    max_run = 1
    run = 1
    for previous, current in zip(objects, objects[1:]):
        if (
            previous["label"] == current["label"]
            and _iou(previous["bbox_2d"], current["bbox_2d"]) >= NEAR_DUPLICATE_IOU
        ):
            run += 1
        else:
            run = 1
        max_run = max(max_run, run)
    return max_run


def _iou(left: list[int], right: list[int]) -> float:
    intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


if __name__ == "__main__":
    main()

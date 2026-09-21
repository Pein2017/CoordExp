"""Cold reduction for native-escape records.

Counts strict class-blind repeats once per later valid free row. Complete
geometry-invalid rows, other complete malformed rows, and incomplete fragments
remain separate. No EOS or GT-relative label is converted into success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def iou(left: list[int], right: list[int]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_left = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    area_right = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def strict_repeat_orders(record: dict[str, Any]) -> list[int]:
    partition = record["parser_partition"]
    prefix_count = partition["prefix_row_count"]
    valid = sorted(record["parsed"]["pred"], key=lambda row: row["generated_order"])
    earlier: list[list[int]] = []
    repeated: list[int] = []
    for row in valid:
        order = row["generated_order"]
        bbox = row["bbox"]
        if order >= prefix_count and any(iou(bbox, other) > 0.95 for other in earlier):
            repeated.append(order)
        earlier.append(bbox)
    return repeated


def burden(record: dict[str, Any]) -> dict[str, Any]:
    dropped = record["parser_partition"]["free_rows"]
    dropped = [row for row in dropped if row["parser_disposition"] == "dropped"]
    geometry_invalid = [
        row for row in dropped
        if row.get("reason") == "geometry_invalid"
        and str(row.get("raw_span_text", "")).endswith("<|box_end|>")
    ]
    other_complete_malformed = [
        row for row in dropped
        if row not in geometry_invalid
        and str(row.get("raw_span_text", "")).endswith("<|box_end|>")
    ]
    incomplete = [
        row for row in dropped
        if not str(row.get("raw_span_text", "")).endswith("<|box_end|>")
    ]
    return {
        "geometry_invalid_complete_count": len(geometry_invalid),
        "geometry_invalid_complete_orders": [row.get("generated_order")
                                              for row in geometry_invalid],
        "other_complete_malformed_count": len(other_complete_malformed),
        "other_complete_malformed_orders": [row.get("generated_order")
                                             for row in other_complete_malformed],
        "incomplete_fragment_count": len(incomplete),
        "incomplete_fragment_orders": [row.get("generated_order") for row in incomplete],
    }


def reduce_record(record: dict[str, Any]) -> dict[str, Any]:
    repeats = strict_repeat_orders(record)
    counts = record["parser_partition"]["counts"]
    return {
        "case_id": record["case_id"],
        "job_id": record["job_id"],
        "kind": record["kind"],
        "candidate_id": (record.get("forced_candidate") or {}).get("candidate_id"),
        "prefix_token_count": len(record["prefix_ids"]),
        "forced_candidate_row_count": record["credit_identity"]["forced_candidate_row_count"],
        "forced_candidate_in_free_counts": False,
        "free_token_count": len(record["free_ids"]),
        "stop_reason": record["stop_reason"],
        "native_eos_observed": record["native_eos_observed"],
        "eos_is_positive_outcome": False,
        "valid_complete_free_row_count": counts["valid_complete_free_rows"],
        "strict_class_blind_repeat_later_row_count": len(repeats),
        "strict_class_blind_repeat_later_orders": repeats,
        "burden": burden(record),
        "natural_owner_recovery_eligible": record["credit_identity"]
        ["natural_owner_recovery_eligible"],
        "conditional_free_suffix_review_eligible": record["credit_identity"]
        ["conditional_free_suffix_review_eligible"],
        "useful_free_continuation": "pending_root_visual_review",
        "baseline_match": record.get("baseline_match"),
    }


def read_records(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    keys = [(row["case_id"], row["job_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate case/job records")
    return rows


def reduce(paths: list[Path]) -> dict[str, Any]:
    rows = read_records(paths)
    cells = [reduce_record(row) for row in sorted(
        rows, key=lambda row: (row["case_id"], row["job_id"]))]
    return {
        "schema": "native_escape_witness.reduction.v1",
        "status": "candidate_evidence_pending_root_acceptance",
        "source_records": [
            {"path": str(path.resolve()),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths
        ],
        "cell_count": len(cells),
        "cells": cells,
        "totals": {
            "generated_tokens": sum(cell["free_token_count"] for cell in cells),
            "valid_complete_free_rows": sum(
                cell["valid_complete_free_row_count"] for cell in cells),
            "strict_class_blind_repeat_later_rows": sum(
                cell["strict_class_blind_repeat_later_row_count"] for cell in cells),
            "geometry_invalid_complete_free_rows": sum(
                cell["burden"]["geometry_invalid_complete_count"] for cell in cells),
            "other_complete_malformed_free_rows": sum(
                cell["burden"]["other_complete_malformed_count"] for cell in cells),
            "incomplete_free_fragments": sum(
                cell["burden"]["incomplete_fragment_count"] for cell in cells),
            "native_eos_cells": sum(cell["native_eos_observed"] for cell in cells),
            "forced_candidate_rows": sum(
                cell["forced_candidate_row_count"] for cell in cells),
        },
        "claim_boundary": (
            "Forced rows are excluded from free-row and owner-recovery credit. "
            "EOS is descriptive, not positive. No GT-only hallucination label is emitted."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = reduce(args.records)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

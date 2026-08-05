#!/usr/bin/env python3
"""Summarize owner recovery after forcing one opener at a model's own stop."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from analyze_individual_trajectory_union_support import (
    load_generation7_annotations,
    match_prefix,
)


SCHEMA_VERSION = "own_terminal_forced_opener_summary.v1"


def _paths(patterns: Iterable[str]) -> list[Path]:
    paths = sorted({Path(item).resolve() for pattern in patterns for item in glob.glob(pattern)})
    if not paths:
        raise ValueError("no force-panel artifacts matched")
    return paths


def _prediction(raw: Mapping[str, Any], *, image_id: str, index: int, source: str) -> dict[str, Any]:
    return {
        "prediction_id": f"{image_id}:{source}:{index}",
        "generated_row_index": index,
        "category": " ".join(str(raw["description"]).strip().lower().replace("_", " ").split()),
        "bbox": tuple(float(value) for value in raw["bbox"]),
    }


def _predictions(raw_rows: Iterable[Mapping[str, Any]], *, image_id: str, source: str, offset: int = 0) -> list[dict[str, Any]]:
    return [
        _prediction(raw, image_id=image_id, index=offset + index, source=source)
        for index, raw in enumerate(raw_rows)
    ]


def _owner_ids(predictions: list[dict[str, Any]], owners: list[dict[str, Any]]) -> set[str]:
    if not predictions:
        return set()
    return set(match_prefix(predictions, owners, len(predictions))["matched_owner_ids"])


def summarize(patterns: Iterable[str], annotations_path: str | Path) -> dict[str, Any]:
    paths = _paths(patterns)
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            image_id = str(case["image_id"])
            if image_id in seen:
                raise ValueError(f"duplicate image_id across shards: {image_id}")
            seen.add(image_id)
            cases.append(case)

    owners_by_image = load_generation7_annotations(annotations_path, image_ids=seen)
    outcome_counts: Counter[str] = Counter()
    aggregate = Counter()
    case_receipts: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: int(item["image_id"])):
        image_id = str(case["image_id"])
        owners = owners_by_image[image_id]
        native = _predictions(case["native_predictions"], image_id=image_id, source="native")
        native_ids = _owner_ids(native, owners)

        first_raw = case["first_row"].get("parsed_predictions") or []
        first = _predictions(first_raw, image_id=image_id, source="forced-first")
        first_ids = _owner_ids(first, owners)
        first_gained = first_ids - native_ids
        first_repeated = first_ids & native_ids
        if first_gained:
            first_outcome = "uncovered_owner_recovery"
        elif first_repeated:
            first_outcome = "repeat_owner"
        elif not first:
            first_outcome = "invalid_or_terminal"
        else:
            first_outcome = "valid_unresolved"
        outcome_counts[first_outcome] += 1

        released_raw = list(first_raw)
        terminal_reason = case["first_row"]["row_stop"].get("stop_reason")
        for row in case["suffix_rows"]:
            released_raw.extend(row.get("parsed_predictions") or [])
            terminal_reason = row["row_stop"].get("stop_reason")
        released = _predictions(released_raw, image_id=image_id, source="forced-release")
        final = native + _predictions(
            released_raw,
            image_id=image_id,
            source="forced-release",
            offset=len(native),
        )
        final_ids = _owner_ids(final, owners)
        gained = final_ids - native_ids
        retained = final_ids & native_ids
        lost = native_ids - final_ids
        aggregate.update(
            native=len(native_ids),
            first_gained=len(first_gained),
            final_gained=len(gained),
            retained=len(retained),
            lost=len(lost),
            released_rows=len(released),
        )
        if terminal_reason == "length":
            aggregate["length_stops"] += 1
        case_receipts.append(
            {
                "image_id": image_id,
                "native_owner_ids": sorted(native_ids),
                "first_row_outcome": first_outcome,
                "first_row_owner_ids": sorted(first_ids),
                "first_row_gained_owner_ids": sorted(first_gained),
                "final_owner_ids": sorted(final_ids),
                "final_gained_owner_ids": sorted(gained),
                "retained_owner_ids": sorted(retained),
                "lost_owner_ids": sorted(lost),
                "released_complete_row_count": len(released),
                "terminal_reason": terminal_reason,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "Automatic category-consistent one-to-one owner matching at IoU >= 0.5; "
            "ambiguous same-category matches remain uncommitted and unmatched rows are unresolved."
        ),
        "source_artifacts": [str(path) for path in paths],
        "annotations": str(Path(annotations_path).resolve()),
        "case_count": len(cases),
        "first_row_outcome_counts": dict(sorted(outcome_counts.items())),
        "aggregate": dict(aggregate),
        "cases": case_receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-artifact", action="append", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.force_artifact, args.annotations)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize the first N post-native forced-continuation predictions."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Iterator

from scripts.research.analyze_individual_trajectory_union_support import (
    load_generation7_annotations,
)
from scripts.research.materialize_forced_continue_full_trajectory_visuals import (
    _annotation_meta,
    _arm_name,
)
from scripts.research.run_iterative_forced_continue_exact_native import (
    _coverage_snapshot,
    _prediction,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input artifact path or glob")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--budget", action="append", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _row_predictions(
    *,
    image_id: str,
    row: dict[str, Any],
    row_index: int,
    event_type: str,
    force_ordinal: int | None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for local_index, raw in enumerate(row.get("parsed_predictions") or []):
        prediction = _prediction(
            raw,
            image_id=image_id,
            row_index=row_index,
            prediction_index=local_index,
        )
        yield prediction, {
            "prediction_id": prediction["prediction_id"],
            "event_type": event_type,
            "force_ordinal": force_ordinal,
            "generated_row_index": row_index,
            "raw_generated_text": row.get("raw_generated_text"),
            "raw_generated_token_ids": row.get("raw_generated_token_ids"),
            "description": str(raw["description"]),
            "bbox_pixel_xyxy": [float(value) for value in raw["bbox"]],
            "coord_bins": [int(value) for value in raw["coord_bins"]],
        }


def _case_predictions(
    case: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    image_id = str(case["image_id"])
    native: list[dict[str, Any]] = []
    post: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    observed_native = False
    for event in case["events"]:
        event_type = str(event["event_type"])
        if event_type == "natural_segment":
            segment = event["segment"]
            target = native if not observed_native else post
            phase = "native" if not observed_native else "post_force_natural"
            if bool(segment["segment_is_clean_complete_rows"]):
                rows = [
                    (row, int(row["row_index"]))
                    for row in segment.get("complete_rows", [])
                ]
            else:
                parsed = list(segment.get("parsed_predictions") or [])
                first_row_index = int(event["complete_row_count_after"]) - len(parsed)
                rows = [
                    (
                        {
                            "parsed_predictions": [raw],
                            "raw_generated_text": segment.get("raw_generated_text"),
                            "raw_generated_token_ids": segment.get("raw_generated_token_ids"),
                        },
                        first_row_index + offset,
                    )
                    for offset, raw in enumerate(parsed)
                ]
            for row, row_index in rows:
                for prediction, receipt in _row_predictions(
                    image_id=image_id,
                    row=row,
                    row_index=row_index,
                    event_type=phase,
                    force_ordinal=None,
                ):
                    target.append(prediction)
                    if observed_native:
                        provenance.append(receipt)
            observed_native = True
        elif event_type == "forced_row" and bool(event["accepted_complete_row"]):
            row = event.get("row")
            if row is None:
                continue
            for prediction, receipt in _row_predictions(
                image_id=image_id,
                row=row,
                row_index=int(event["row_index"]),
                event_type="forced_row",
                force_ordinal=int(event["force_index"]) + 1,
            ):
                post.append(prediction)
                provenance.append(receipt)
    if not observed_native:
        raise RuntimeError(f"{image_id}: no native natural segment")
    if len(native) != int(case["native_snapshot"]["prediction_count"]):
        raise RuntimeError(f"{image_id}: native prediction count mismatch")
    if len(native) + len(post) != int(case["final_snapshot"]["prediction_count"]):
        raise RuntimeError(f"{image_id}: final prediction count mismatch")
    if len(post) != len(provenance):
        raise RuntimeError(f"{image_id}: post-force provenance mismatch")
    return native, post, provenance


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    inputs = sorted(
        {
            Path(match).resolve(strict=True)
            for pattern in args.input
            for match in glob.glob(pattern)
        }
    )
    budgets = sorted(set(args.budget))
    if not inputs:
        raise SystemExit("no input artifacts matched")
    if not budgets or any(value <= 0 for value in budgets):
        raise SystemExit("budgets must be positive")
    annotation_path = args.annotations.resolve(strict=True)
    visual_meta = _annotation_meta(annotation_path)
    owners_by_image = load_generation7_annotations(
        annotation_path,
        image_ids=sorted(visual_meta),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    by_budget: dict[int, dict[str, list[dict[str, Any]]]] = {
        budget: {"gt": [], "pred": [], "detail": []} for budget in budgets
    }
    for source in inputs:
        payload = json.loads(source.read_text(encoding="utf-8"))
        arm = _arm_name(payload)
        for case in payload["cases"]:
            image_id = str(case["image_id"])
            native, post, provenance = _case_predictions(case)
            owners = owners_by_image[image_id]
            native_snapshot = _coverage_snapshot(native, owners)
            native_owner_ids = set(native_snapshot["matched_owner_ids"])
            remaining_owners = [
                owner for owner in owners if str(owner["owner_id"]) not in native_owner_ids
            ]
            meta = visual_meta[image_id]
            gt_by_owner_id = dict(zip(meta["gt_owner_ids"], meta["gt"], strict=True))
            remaining_owner_ids = [str(owner["owner_id"]) for owner in remaining_owners]

            for budget in budgets:
                selected_predictions = post[:budget]
                selected_provenance = provenance[:budget]
                snapshot = _coverage_snapshot(selected_predictions, remaining_owners)
                rendered: list[dict[str, Any]] = []
                detail_provenance: list[dict[str, Any]] = []
                for prediction, receipt in zip(
                    selected_predictions,
                    selected_provenance,
                    strict=True,
                ):
                    bbox = [float(value) for value in prediction["bbox"]]
                    x1, y1, x2, y2 = bbox
                    renderable = all(math.isfinite(value) for value in bbox) and (
                        x1 < x2 and y1 < y2
                    )
                    render_index = None
                    if renderable:
                        render_index = len(rendered)
                        rendered.append({
                            "bbox": bbox,
                            "coord_bins": receipt["coord_bins"],
                            "description": receipt["description"],
                        })
                    detail_provenance.append({
                        **receipt,
                        "renderable": renderable,
                        "render_prediction_index": render_index,
                    })
                actual = len(selected_predictions)
                row_id = (
                    f"{arm}-image-{image_id}-post-native-first-{budget}-"
                    f"actual-{actual}-rendered-{len(rendered)}"
                )
                row_index = len(by_budget[budget]["gt"])
                by_budget[budget]["gt"].append({
                    "row_id": row_id,
                    "row_index": row_index,
                    "image_path": meta["image_path"],
                    "image_width": meta["image_width"],
                    "image_height": meta["image_height"],
                    "gt": [gt_by_owner_id[owner_id] for owner_id in remaining_owner_ids],
                })
                by_budget[budget]["pred"].append({
                    "row_id": row_id,
                    "row_index": row_index,
                    "image_path": meta["image_path"],
                    "image_width": meta["image_width"],
                    "image_height": meta["image_height"],
                    "pred": rendered,
                })
                by_budget[budget]["detail"].append({
                    "row_id": row_id,
                    "arm": arm,
                    "source_artifact": str(source),
                    "image_id": image_id,
                    "requested_post_native_prediction_budget": budget,
                    "available_post_native_prediction_count": len(post),
                    "parsed_prediction_count": actual,
                    "renderable_prediction_count": len(rendered),
                    "nonrenderable_prediction_count": actual - len(rendered),
                    "invalid_or_unparsed_row_count": 0,
                    "native_snapshot": native_snapshot,
                    "final_snapshot": snapshot,
                    "gt_owner_ids": remaining_owner_ids,
                    "prediction_provenance": detail_provenance,
                })

    expected_rows = len(inputs) * len(visual_meta)
    for budget, rows in by_budget.items():
        if len(rows["detail"]) != expected_rows:
            raise RuntimeError(
                f"budget {budget}: expected {expected_rows} rows, got {len(rows['detail'])}"
            )
        budget_dir = output / f"first-{budget}-post-native-predictions"
        run_dir = budget_dir / "visual-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl", rows["gt"])
        _write_jsonl(run_dir / "gt_vs_pred_scored.jsonl", rows["pred"])
        _write_jsonl(budget_dir / "details.jsonl", rows["detail"])
        summary = {
            "requested_budget": budget,
            "row_count": len(rows["detail"]),
            "parsed_prediction_count": sum(
                int(item["parsed_prediction_count"]) for item in rows["detail"]
            ),
            "renderable_prediction_count": sum(
                int(item["renderable_prediction_count"]) for item in rows["detail"]
            ),
            "cases_with_fewer_than_budget": sum(
                int(item["available_post_native_prediction_count"]) < budget
                for item in rows["detail"]
            ),
        }
        (budget_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(output)


if __name__ == "__main__":
    main()

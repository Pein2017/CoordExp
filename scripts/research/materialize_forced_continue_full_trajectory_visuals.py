#!/usr/bin/env python3
"""Materialize full-trajectory visualization inputs for forced continuation."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import re
from typing import Any


_COORD_RE = re.compile(r"^<\|coord_(\d+)\|>$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input artifact path or glob")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _arm_name(payload: dict[str, Any]) -> str:
    config = Path(str(payload["config"]["infer_config"])).name
    if "permutation_bundle" in config:
        checkpoint = "permutation"
    elif "geo_sorted" in config:
        checkpoint = "sorted"
    else:
        checkpoint = "random"
    rp = float(payload["config"]["repetition_penalty"])
    return f"{checkpoint}-rp{rp:.1f}"


def _coord(value: str) -> int:
    match = _COORD_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid coordinate token: {value!r}")
    return int(match.group(1))


def _annotation_meta(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        image_id = str(row["image_id"])
        image_path = (path.parent / str(row["images"][0])).resolve(strict=True)
        result[image_id] = {
            "image_path": str(image_path),
            "image_width": int(row["width"]),
            "image_height": int(row["height"]),
            "gt": [
                {
                    "bbox": [_coord(value) for value in obj["bbox_2d"]],
                    "coco_ann_id": str(obj["coco_ann_id"]),
                    "description": str(obj["desc"]),
                }
                for obj in row["objects"]
            ],
            "gt_owner_ids": [
                f"{image_id}:{obj['coco_ann_id']}" for obj in row["objects"]
            ],
        }
    return result


def _complete_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in case["events"]:
        event_type = str(event["event_type"])
        if event_type == "natural_segment":
            segment = event["segment"]
            if bool(segment["segment_is_clean_complete_rows"]):
                for row in segment.get("complete_rows", []):
                    rows.append({
                        "event_type": event_type,
                        "force_ordinal": None,
                        "generated_row_index": int(row["row_index"]),
                        "row": row,
                    })
            else:
                parsed = list(segment.get("parsed_predictions") or [])
                first_row_index = int(event["complete_row_count_after"]) - len(parsed)
                for offset, prediction in enumerate(parsed):
                    rows.append({
                        "event_type": "natural_segment_terminal_parse",
                        "force_ordinal": None,
                        "generated_row_index": first_row_index + offset,
                        "row": {
                            "parsed_predictions": [prediction],
                            "raw_generated_text": segment.get("raw_generated_text"),
                            "raw_generated_token_ids": segment.get("raw_generated_token_ids"),
                        },
                    })
        elif event_type == "forced_row" and bool(event["accepted_complete_row"]):
            row = event.get("row")
            if row is not None:
                rows.append({
                    "event_type": event_type,
                    "force_ordinal": int(event["force_index"]) + 1,
                    "generated_row_index": int(event["row_index"]),
                    "row": row,
                })
    return rows


def _materialize_case(
    *,
    arm: str,
    source: Path,
    case: dict[str, Any],
    annotation: dict[str, Any],
    row_index: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    provenance: list[dict[str, Any]] = []
    unparsed_rows: list[dict[str, Any]] = []
    parsed_predictions: list[dict[str, Any]] = []
    renderable_predictions: list[dict[str, Any]] = []
    invalid_or_unparsed = 0
    for item in _complete_rows(case):
        row = item["row"]
        parsed = list(row.get("parsed_predictions") or [])
        if not parsed:
            invalid_or_unparsed += 1
            unparsed_rows.append({
                "event_type": item["event_type"],
                "force_ordinal": item["force_ordinal"],
                "generated_row_index": int(item["generated_row_index"]),
                "raw_generated_text": row.get("raw_generated_text"),
                "raw_generated_token_ids": row.get("raw_generated_token_ids"),
            })
        for local_prediction_index, prediction in enumerate(parsed):
            normalized = {
                "bbox": [float(value) for value in prediction["bbox"]],
                "coord_bins": [int(value) for value in prediction["coord_bins"]],
                "description": str(prediction["description"]),
            }
            parsed_predictions.append(normalized)
            x1, y1, x2, y2 = normalized["bbox"]
            renderable = all(math.isfinite(value) for value in normalized["bbox"]) and (
                x1 < x2 and y1 < y2
            )
            render_index = None
            if renderable:
                render_index = len(renderable_predictions)
                renderable_predictions.append(normalized)
            provenance.append({
                "prediction_index": len(parsed_predictions) - 1,
                "prediction_id": (
                    f"{case['image_id']}:row-{item['generated_row_index']}:"
                    f"prediction-{local_prediction_index}"
                ),
                "render_prediction_index": render_index,
                "renderable": renderable,
                "event_type": item["event_type"],
                "force_ordinal": item["force_ordinal"],
                "generated_row_index": int(item["generated_row_index"]),
                "description": str(prediction["description"]),
                "bbox_pixel_xyxy": [float(value) for value in prediction["bbox"]],
                "coord_bins": [int(value) for value in prediction["coord_bins"]],
                "raw_generated_text": row.get("raw_generated_text"),
                "raw_generated_token_ids": row.get("raw_generated_token_ids"),
            })

    expected = int(case["final_snapshot"]["prediction_count"])
    if len(parsed_predictions) != expected:
        raise RuntimeError(
            f"{arm}/{case['image_id']}: parsed prediction count "
            f"{len(parsed_predictions)} != {expected}"
        )
    complete_row_count = int(case["complete_row_count"])
    if len(_complete_rows(case)) != complete_row_count:
        raise RuntimeError(
            f"{arm}/{case['image_id']}: complete row count mismatch "
            f"{len(_complete_rows(case))} != {complete_row_count}"
        )

    row_id = (
        f"{arm}-image-{case['image_id']}-all-predictions-"
        f"rendered-{len(renderable_predictions)}-nonrenderable-"
        f"{len(parsed_predictions) - len(renderable_predictions)}"
    )
    gt_row = {
        "row_id": row_id,
        "row_index": row_index,
        "image_path": annotation["image_path"],
        "image_width": annotation["image_width"],
        "image_height": annotation["image_height"],
        "gt": annotation["gt"],
    }
    pred_row = {
        "row_id": row_id,
        "row_index": row_index,
        "image_path": annotation["image_path"],
        "image_width": annotation["image_width"],
        "image_height": annotation["image_height"],
        "pred": renderable_predictions,
    }
    detail = {
        "row_id": row_id,
        "arm": arm,
        "source_artifact": str(source),
        "image_id": str(case["image_id"]),
        "terminal_reason": str(case["terminal_reason"]),
        "force_count": int(case["force_count"]),
        "complete_row_count": complete_row_count,
        "parsed_prediction_count": len(parsed_predictions),
        "renderable_prediction_count": len(renderable_predictions),
        "nonrenderable_prediction_count": len(parsed_predictions) - len(renderable_predictions),
        "invalid_or_unparsed_row_count": invalid_or_unparsed,
        "final_snapshot": case["final_snapshot"],
        "gt_owner_ids": annotation["gt_owner_ids"],
        "prediction_provenance": provenance,
        "unparsed_rows": unparsed_rows,
    }
    return gt_row, pred_row, detail


def main() -> None:
    args = _parse_args()
    inputs = sorted(
        {
            Path(match).resolve(strict=True)
            for pattern in args.input
            for match in glob.glob(pattern)
        }
    )
    if not inputs:
        raise SystemExit("no input artifacts matched")
    annotations = _annotation_meta(args.annotations.resolve(strict=True))
    output = args.output.resolve()
    run_dir = output / "visual-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    gt_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for source in inputs:
        payload = json.loads(source.read_text(encoding="utf-8"))
        arm = _arm_name(payload)
        for case in payload["cases"]:
            image_id = str(case["image_id"])
            if image_id not in annotations:
                raise RuntimeError(f"missing annotation for image {image_id}")
            gt_row, pred_row, detail = _materialize_case(
                arm=arm,
                source=source,
                case=case,
                annotation=annotations[image_id],
                row_index=len(gt_rows),
            )
            gt_rows.append(gt_row)
            pred_rows.append(pred_row)
            details.append(detail)

    if len(details) != len(inputs) * len(annotations):
        raise RuntimeError(
            f"expected {len(inputs) * len(annotations)} arm/image records, got {len(details)}"
        )

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_jsonl(run_dir / "gt_vs_pred.jsonl", gt_rows)
    write_jsonl(run_dir / "gt_vs_pred_scored.jsonl", pred_rows)
    write_jsonl(output / "full_trajectory_predictions.jsonl", details)
    summary = {
        "source_artifacts": [str(path) for path in inputs],
        "arm_count": len(inputs),
        "image_count_per_arm": len(annotations),
        "render_row_count": len(details),
        "parsed_prediction_count": sum(item["parsed_prediction_count"] for item in details),
        "renderable_prediction_count": sum(
            item["renderable_prediction_count"] for item in details
        ),
        "nonrenderable_prediction_count": sum(
            item["nonrenderable_prediction_count"] for item in details
        ),
        "invalid_or_unparsed_row_count": sum(
            item["invalid_or_unparsed_row_count"] for item in details
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()

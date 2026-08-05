#!/usr/bin/env python3
"""Materialize a row-level review packet for exact-native forced continuation."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import re
from typing import Any

from scripts.research.analyze_individual_trajectory_union_support import (
    _iou,
    load_generation7_annotations,
)
from scripts.research.run_iterative_forced_continue_exact_native import (
    _coverage_snapshot,
    _normalise_category,
    _prediction,
)


_COORD_RE = re.compile(r"^<\|coord_(\d+)\|>$")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _arm_name(path: Path, payload: dict[str, Any]) -> str:
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
        owners: dict[str, dict[str, Any]] = {}
        for obj in row["objects"]:
            owner_id = f"{image_id}:{obj['coco_ann_id']}"
            owners[owner_id] = {
                "bbox": [_coord(value) for value in obj["bbox_2d"]],
                "coco_ann_id": str(obj["coco_ann_id"]),
                "description": str(obj["desc"]),
            }
        result[image_id] = {
            "image_path": str(image_path),
            "image_width": int(row["width"]),
            "image_height": int(row["height"]),
            "owners": owners,
        }
    return result


def _forced_records(
    path: Path,
    payload: dict[str, Any],
    owners_by_image: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    arm = _arm_name(path, payload)
    records: list[dict[str, Any]] = []
    for case in payload["cases"]:
        image_id = str(case["image_id"])
        owners = owners_by_image[image_id]
        cumulative: list[dict[str, Any]] = []
        prediction_index = 0
        force_meta = {int(item["force_index"]): item for item in case["forces"]}
        events = list(case["events"])
        for event_position, event in enumerate(events):
            if event["event_type"] == "natural_segment":
                for row in event["segment"].get("complete_rows", []):
                    for raw in row.get("parsed_predictions", []):
                        cumulative.append(
                            _prediction(
                                raw,
                                image_id=image_id,
                                row_index=int(row["row_index"]),
                                prediction_index=prediction_index,
                            )
                        )
                        prediction_index += 1
                continue
            if event["event_type"] != "forced_row":
                continue

            force_index = int(event["force_index"])
            meta = force_meta[force_index]
            before = _coverage_snapshot(cumulative, owners)
            if int(event["coverage_before"]) != int(before["coverage"]):
                raise RuntimeError(f"{arm}/{image_id}/force-{force_index + 1}: coverage-before mismatch")
            before_ids = set(before["matched_owner_ids"])
            row = event.get("row") or {}
            raw_predictions = list(row.get("parsed_predictions") or [])
            released_rows: list[str] = []
            if event_position + 1 < len(events) and events[event_position + 1]["event_type"] == "natural_segment":
                released_rows = [
                    str(item["raw_generated_text"])
                    for item in events[event_position + 1]["segment"].get("complete_rows", [])
                ]

            description = None
            bbox = None
            coord_bins = None
            max_same_category_iou = None
            best_owner_id = None
            classification = "invalid_or_incomplete"
            after = before
            if bool(event.get("accepted_complete_row")) and raw_predictions:
                if len(raw_predictions) != 1:
                    raise RuntimeError(f"{arm}/{image_id}/force-{force_index + 1}: expected one prediction")
                raw = raw_predictions[0]
                description = str(raw["description"])
                bbox = [float(value) for value in raw["bbox"]]
                coord_bins = [int(value) for value in raw["coord_bins"]]
                category = _normalise_category(description)
                same_category = sorted(
                    (
                        (float(_iou(bbox, owner["bbox"])), owner)
                        for owner in owners
                        if _normalise_category(owner["category"]) == category
                    ),
                    key=lambda item: item[0],
                    reverse=True,
                )
                if same_category:
                    max_same_category_iou = same_category[0][0]
                    best_owner_id = str(same_category[0][1]["owner_id"])
                strict = [item for item in same_category if item[0] >= 0.5]
                prediction = _prediction(
                    raw,
                    image_id=image_id,
                    row_index=int(event["row_index"]),
                    prediction_index=prediction_index,
                )
                after = _coverage_snapshot([*cumulative, prediction], owners)
                if int(meta["marginal_owner_gain"]) > 0:
                    classification = "new_strict_tp"
                elif strict and str(strict[0][1]["owner_id"]) in before_ids:
                    classification = "repeat_strict_owner"
                elif strict:
                    classification = "strict_match_without_net_gain"
                elif not same_category:
                    classification = "no_audited_gt_of_category"
                elif same_category[0][0] >= 0.1:
                    classification = "same_category_geometry_miss"
                else:
                    classification = "same_category_low_overlap"
                cumulative.append(prediction)
                prediction_index += 1

            if int(event["coverage_after"]) != int(after["coverage"]):
                raise RuntimeError(f"{arm}/{image_id}/force-{force_index + 1}: coverage-after mismatch")
            newly_matched = sorted(set(after["matched_owner_ids"]) - before_ids)
            record = {
                "arm": arm,
                "source_artifact": str(path),
                "image_id": image_id,
                "force_ordinal": force_index + 1,
                "classification": classification,
                "accepted_complete_row": bool(event.get("accepted_complete_row")),
                "description": description,
                "bbox_pixel_xyxy": bbox,
                "coord_bins": coord_bins,
                "raw_generated_text": row.get("raw_generated_text"),
                "raw_generated_token_ids": row.get("raw_generated_token_ids"),
                "raw_generated_token_ids_sha256": row.get("raw_generated_token_ids_sha256"),
                "coverage_before": int(before["coverage"]),
                "coverage_after_forced_row": int(after["coverage"]),
                "coverage_at_next_boundary": int(meta["coverage_at_next_boundary"]),
                "marginal_owner_gain": int(meta["marginal_owner_gain"]),
                "interval_owner_gain": int(meta["interval_owner_gain"]),
                "newly_matched_owner_ids": newly_matched,
                "covered_owner_ids_before": sorted(before_ids),
                "remaining_owner_ids_before": sorted(
                    str(owner["owner_id"])
                    for owner in owners
                    if str(owner["owner_id"]) not in before_ids
                ),
                "best_same_category_owner_id": best_owner_id,
                "max_same_category_iou": max_same_category_iou,
                "released_natural_rows": released_rows,
                "interval_termination": str(meta["interval_termination"]),
                "token_start": int(meta["token_start"]),
                "token_end": int(meta["token_end"]),
            }
            records.append(record)
    return records


def _representatives(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    arms = sorted({str(record["arm"]) for record in records})
    for arm in arms:
        arm_rows = [record for record in records if record["arm"] == arm]
        chosen_keys: set[tuple[str, int]] = set()

        def add(record: dict[str, Any]) -> None:
            key = (str(record["image_id"]), int(record["force_ordinal"]))
            if key not in chosen_keys:
                selected.append(record)
                chosen_keys.add(key)

        tp_rows = [record for record in arm_rows if record["classification"] == "new_strict_tp"]
        if tp_rows:
            add(min(tp_rows, key=lambda item: int(item["force_ordinal"])))
            add(max(tp_rows, key=lambda item: int(item["force_ordinal"])))
        interval_extra = [
            record
            for record in arm_rows
            if int(record["interval_owner_gain"]) > int(record["marginal_owner_gain"])
        ]
        if interval_extra:
            add(max(interval_extra, key=lambda item: int(item["interval_owner_gain"])))
        for classification in (
            "repeat_strict_owner",
            "same_category_geometry_miss",
            "same_category_low_overlap",
            "no_audited_gt_of_category",
            "invalid_or_incomplete",
        ):
            candidates = [record for record in arm_rows if record["classification"] == classification]
            valid_text = [record for record in candidates if record.get("raw_generated_text")]
            if valid_text:
                frequencies = Counter(str(record["raw_generated_text"]) for record in valid_text)
                add(max(valid_text, key=lambda item: (frequencies[str(item["raw_generated_text"])], -int(item["force_ordinal"]))))
            elif candidates:
                add(candidates[0])
    return selected


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in sorted({str(record["arm"]) for record in records}):
        arm_rows = [record for record in records if record["arm"] == arm]
        counts = Counter(str(record["classification"]) for record in arm_rows)
        descriptions: dict[str, list[dict[str, Any]]] = {}
        for classification in sorted(counts):
            counter = Counter(
                str(record["description"])
                for record in arm_rows
                if record["classification"] == classification and record["description"] is not None
            )
            descriptions[classification] = [
                {"description": description, "count": count}
                for description, count in counter.most_common(8)
            ]
        result[arm] = {
            "force_count": len(arm_rows),
            "classification_counts": dict(sorted(counts.items())),
            "classification_rates": {
                key: value / len(arm_rows) for key, value in sorted(counts.items())
            },
            "top_descriptions_by_classification": descriptions,
            "unique_forced_row_text_count": len(
                {record["raw_generated_text"] for record in arm_rows if record["raw_generated_text"]}
            ),
        }
    return result


def _write_review(
    path: Path,
    summary: dict[str, Any],
    selected: list[dict[str, Any]],
) -> None:
    lines = [
        "# Forced-Continue Behavior Review",
        "",
        "Red predictions in the rendered atlas mean unmatched against audited GT owners that remained before that force. They are not automatically hallucinations.",
        "",
        "`same_category_geometry_miss` uses diagnostic max IoU in [0.1, 0.5); `same_category_low_overlap` uses max IoU < 0.1.",
        "",
        "## Aggregate taxonomy",
        "",
        "| Arm | Forces | New strict TP | Repeat | Geometry miss | Low overlap | No audited GT category | Invalid/incomplete |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in summary.items():
        counts = item["classification_counts"]
        lines.append(
            f"| {arm} | {item['force_count']} | {counts.get('new_strict_tp', 0)} | "
            f"{counts.get('repeat_strict_owner', 0)} | {counts.get('same_category_geometry_miss', 0)} | "
            f"{counts.get('same_category_low_overlap', 0)} | {counts.get('no_audited_gt_of_category', 0)} | "
            f"{counts.get('invalid_or_incomplete', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Representative rows",
            "",
            "| Arm | Image | Force | Class | Description | Max same-category IoU | New owner | Raw row |",
            "|---|---:|---:|---|---|---:|---|---|",
        ]
    )
    for record in selected:
        iou = record["max_same_category_iou"]
        iou_text = "n/a" if iou is None else f"{float(iou):.3f}"
        raw = str(record.get("raw_generated_text") or "<incomplete>").replace("|", "\\|")
        lines.append(
            f"| {record['arm']} | {record['image_id']} | {record['force_ordinal']} | "
            f"{record['classification']} | {record.get('description') or 'n/a'} | {iou_text} | "
            f"{', '.join(record['newly_matched_owner_ids']) or '-'} | `{raw}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize(args: argparse.Namespace) -> Path:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    annotations = args.annotations.expanduser().resolve(strict=True)
    owners = load_generation7_annotations(annotations)
    annotation_meta = _annotation_meta(annotations)
    paths = sorted({Path(value).resolve(strict=True) for pattern in args.input for value in glob.glob(pattern)})
    if not paths:
        raise ValueError("no input artifacts matched")
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "iterative_forced_continue_exact_native.v2":
            raise ValueError(f"unsupported artifact schema: {path}")
        records.extend(_forced_records(path, payload, owners))

    summary = _summary(records)
    selected = _representatives(records)
    (output / "all_forced_rows.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "forced_continue_behavior_review.v1",
                "claim_boundary": "unmatched remaining-owner rows are review candidates, not automatic hallucinations",
                "source_artifacts": [str(path) for path in paths],
                "record_count": len(records),
                "selected_record_count": len(selected),
                "arms": summary,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "selection.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_review(output / "review.md", summary, selected)

    run_dir = output / "visual-run"
    run_dir.mkdir(exist_ok=True)
    gt_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    visual_index = 0
    for record in selected:
        if record["bbox_pixel_xyxy"] is None:
            continue
        image_id = str(record["image_id"])
        meta = annotation_meta[image_id]
        row_id = (
            f"{record['arm']}-image-{image_id}-force-{record['force_ordinal']}-"
            f"{record['classification']}"
        )
        remaining = [
            meta["owners"][owner_id]
            for owner_id in record["remaining_owner_ids_before"]
        ]
        common = {
            "image_height": meta["image_height"],
            "image_path": meta["image_path"],
            "image_width": meta["image_width"],
            "row_id": row_id,
            "row_index": visual_index,
        }
        gt_rows.append({**common, "gt": remaining})
        pred_rows.append(
            {
                **common,
                "pred": [
                    {
                        "bbox": record["bbox_pixel_xyxy"],
                        "coord_bins": record["coord_bins"],
                        "description": record["description"],
                    }
                ],
            }
        )
        visual_index += 1
    (run_dir / "gt_vs_pred.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in gt_rows),
        encoding="utf-8",
    )
    (run_dir / "gt_vs_pred_scored.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in pred_rows),
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(materialize(parse_args()))


if __name__ == "__main__":
    main()

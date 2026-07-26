#!/usr/bin/env python3
"""Materialize canonical two-run visualization inputs for forced-opener review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.summarize_source_transition_forced_opener_comparison import (  # noqa: E402
    SOURCE_UNIT_ID,
    UNIT_ID,
    _load_receipts,
)
from src.config.fingerprint import sha256_file  # noqa: E402


SCHEMA_VERSION = "source_transition_forced_opener_visual_review.v1"
_COORD_PATTERN = re.compile(r"^<\|coord_(\d+)\|>$")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _coord_value(token: Any) -> int:
    match = _COORD_PATTERN.fullmatch(str(token))
    if match is None:
        raise ValueError(f"invalid coordinate token: {token!r}")
    return int(match.group(1))


def _candidate_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    by_id = {str(row["image_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("candidate pool contains duplicate image IDs")
    return by_id


def _gt_row(
    *, boundary_id: str, row_index: int, candidate: dict[str, Any], candidate_path: Path
) -> dict[str, Any]:
    images = candidate.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError(f"{boundary_id} candidate must contain exactly one image")
    image_path = (candidate_path.parent / str(images[0])).resolve(strict=True)
    objects = candidate.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{boundary_id} candidate lacks objects")
    gt: list[dict[str, Any]] = []
    for obj in objects:
        coords = obj.get("bbox_2d")
        if not isinstance(coords, list) or len(coords) != 4:
            raise ValueError(f"{boundary_id} candidate has invalid bbox")
        gt.append(
            {
                "bbox": [_coord_value(value) for value in coords],
                "coco_ann_id": str(obj["coco_ann_id"]),
                "description": str(obj["desc"]),
            }
        )
    return {
        "gt": gt,
        "image_height": int(candidate["height"]),
        "image_path": str(image_path),
        "image_width": int(candidate["width"]),
        "row_id": boundary_id,
        "row_index": row_index,
    }


def _pred_row(
    *,
    boundary_id: str,
    row_index: int,
    candidate: dict[str, Any],
    candidate_path: Path,
    raw_case: dict[str, Any],
) -> dict[str, Any]:
    release = raw_case.get("releases", {}).get("forced_opener", {})
    predictions = release.get("parsed_predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"{boundary_id} forced release lacks parsed predictions")
    pred: list[dict[str, Any]] = []
    for prediction in predictions:
        bbox = prediction.get("bbox")
        coords = prediction.get("coord_bins")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{boundary_id} prediction has invalid pixel bbox")
        if not isinstance(coords, list) or len(coords) != 4:
            raise ValueError(f"{boundary_id} prediction has invalid coordinate bins")
        pred.append(
            {
                "bbox": [float(value) for value in bbox],
                "coord_bins": [int(value) for value in coords],
                "description": str(prediction["description"]),
            }
        )
    images = candidate["images"]
    image_path = (candidate_path.parent / str(images[0])).resolve(strict=True)
    return {
        "image_height": int(candidate["height"]),
        "image_path": str(image_path),
        "image_width": int(candidate["width"]),
        "pred": pred,
        "row_id": boundary_id,
        "row_index": row_index,
    }

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    comparison_path = args.comparison_cases.expanduser().resolve(strict=True)
    comparison_rows = _read_jsonl(comparison_path)
    comparison_by_id = {str(row["boundary_id"]): row for row in comparison_rows}
    if len(comparison_rows) != 200 or len(comparison_by_id) != 200:
        raise ValueError("comparison ledger must contain 200 unique boundaries")
    prior_selection_path = args.prior_selection.expanduser().resolve(strict=True)
    prior_selection = _read_json(prior_selection_path)
    selected: dict[str, set[str]] = {}
    for row in prior_selection.get("selection", []):
        if row.get("in_random_30") is True:
            selected.setdefault(str(row["boundary_id"]), set()).add("prior_random_30")
    for row in comparison_rows:
        if row["checkpoint_gained_owner_ids"] or row["checkpoint_lost_owner_ids"]:
            selected.setdefault(str(row["boundary_id"]), set()).add("strict_owner_change")
    if len([groups for groups in selected.values() if "prior_random_30" in groups]) != 30:
        raise ValueError("prior visual selection does not contain exactly 30 random cases")

    source_raw, _, source_receipts = _load_receipts(
        args.source_receipts_root,
        expected_unit_id=SOURCE_UNIT_ID,
        expected_role="source",
    )
    transition_raw, _, transition_receipts = _load_receipts(
        args.transition_receipts_root,
        expected_unit_id=UNIT_ID,
        expected_role="transition-step36",
    )
    candidate_path = args.candidate_pool.expanduser().resolve(strict=True)
    candidates = _candidate_rows(candidate_path)
    ordered = sorted(selected)
    source_gt: list[dict[str, Any]] = []
    source_pred: list[dict[str, Any]] = []
    transition_gt: list[dict[str, Any]] = []
    transition_pred: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for row_index, boundary_id in enumerate(ordered):
        comparison = comparison_by_id[boundary_id]
        image_id = str(comparison["image_id"])
        candidate = candidates[image_id]
        gt = _gt_row(
            boundary_id=boundary_id,
            row_index=row_index,
            candidate=candidate,
            candidate_path=candidate_path,
        )
        source_gt.append(gt)
        transition_gt.append(dict(gt))
        source_pred.append(
            _pred_row(
                boundary_id=boundary_id,
                row_index=row_index,
                candidate=candidate,
                candidate_path=candidate_path,
                raw_case=source_raw[boundary_id],
            )
        )
        transition_pred.append(
            _pred_row(
                boundary_id=boundary_id,
                row_index=row_index,
                candidate=candidate,
                candidate_path=candidate_path,
                raw_case=transition_raw[boundary_id],
            )
        )
        manifest_rows.append(
            {
                "boundary_id": boundary_id,
                "image_id": image_id,
                "selection_groups": sorted(selected[boundary_id]),
                "source_forced_outcome": comparison["source_forced"]["outcome"],
                "transition_forced_outcome": comparison["transition_forced"]["outcome"],
                "forced_raw_row_token_ids_equal": comparison[
                    "forced_raw_row_token_ids_equal"
                ],
                "checkpoint_gained_owner_ids": comparison["checkpoint_gained_owner_ids"],
                "checkpoint_lost_owner_ids": comparison["checkpoint_lost_owner_ids"],
            }
        )

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable output root: {output_root}")
    source_root = output_root / "source"
    transition_root = output_root / "transition-step36"
    source_root.mkdir(parents=True)
    transition_root.mkdir(parents=True)
    _write_jsonl(source_root / "gt_vs_pred.jsonl", source_gt)
    _write_jsonl(source_root / "gt_vs_pred_scored.jsonl", source_pred)
    _write_jsonl(transition_root / "gt_vs_pred.jsonl", transition_gt)
    _write_jsonl(transition_root / "gt_vs_pred_scored.jsonl", transition_pred)
    result = {
        "schema_version": SCHEMA_VERSION,
        "selection_count": len(ordered),
        "prior_random_30_count": sum(
            "prior_random_30" in groups for groups in selected.values()
        ),
        "strict_owner_change_count": sum(
            "strict_owner_change" in groups for groups in selected.values()
        ),
        "inputs": {
            "comparison_cases": {
                "path": str(comparison_path),
                "sha256": sha256_file(comparison_path),
            },
            "prior_selection": {
                "path": str(prior_selection_path),
                "sha256": sha256_file(prior_selection_path),
            },
            "candidate_pool": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "source_receipts": source_receipts,
            "transition_receipts": transition_receipts,
        },
        "source_run_dir": str(source_root),
        "transition_run_dir": str(transition_root),
        "selection": manifest_rows,
    }
    (output_root / "selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-cases", type=Path, required=True)
    parser.add_argument("--prior-selection", type=Path, required=True)
    parser.add_argument("--source-receipts-root", type=Path, required=True)
    parser.add_argument("--transition-receipts-root", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    result = materialize(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "selection_count": result["selection_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

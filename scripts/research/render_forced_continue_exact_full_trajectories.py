#!/usr/bin/env python3
"""Render all forced-continuation predictions with the probe's exact assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.vis.matching import MatchPair, MatchResult, match_row
from src.vis.normalization import load_visual_rows
from src.vis.rendering import render_gt_vs_prediction_png


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-." else "-" for character in value)


def main() -> None:
    args = _parse_args()
    artifacts = load_visual_rows(args.run_dir.resolve(strict=True))
    details = {
        str(item["row_id"]): item
        for item in _jsonl(args.details.resolve(strict=True))
    }
    if {row.row_id for row in artifacts.rows} != set(details):
        raise RuntimeError("visual rows and detail rows do not have identical row IDs")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_items: list[dict[str, Any]] = []

    for index, row in enumerate(artifacts.rows):
        detail = details[row.row_id]
        gt_by_owner_id = {
            str(owner_id): gt_index
            for gt_index, owner_id in enumerate(detail["gt_owner_ids"])
        }
        pred_by_prediction_id = {
            str(item["prediction_id"]): int(item["render_prediction_index"])
            for item in detail["prediction_provenance"]
            if item["render_prediction_index"] is not None
        }
        pairs: list[MatchPair] = []
        for item in detail["final_snapshot"]["matches"]:
            owner_id = str(item["owner_id"])
            prediction_id = str(item["prediction_id"])
            if owner_id not in gt_by_owner_id:
                raise RuntimeError(f"{row.row_id}: unknown matched owner {owner_id}")
            if prediction_id not in pred_by_prediction_id:
                raise RuntimeError(f"{row.row_id}: matched prediction is not renderable: {prediction_id}")
            pairs.append(
                MatchPair(
                    pred_index=pred_by_prediction_id[prediction_id],
                    gt_index=gt_by_owner_id[owner_id],
                    iou=float(item["intersection_over_union"]),
                )
            )
        matched_gt = {pair.gt_index for pair in pairs}
        matched_pred = {pair.pred_index for pair in pairs}
        if len(pairs) != int(detail["final_snapshot"]["coverage"]):
            raise RuntimeError(f"{row.row_id}: exact match count does not equal final coverage")
        duplicate_candidates = match_row(row).duplicate_candidates
        exact_match = MatchResult(
            matches=tuple(pairs),
            missing_gt_indices=tuple(
                gt.index for gt in row.gt if gt.index not in matched_gt
            ),
            fp_pred_indices=tuple(
                pred.index for pred in row.pred if pred.index not in matched_pred
            ),
            duplicate_candidates=duplicate_candidates,
        )
        output_png = output / f"{index:04d}_{_slug(row.row_id)}_exact_full_trajectory.png"
        render_gt_vs_prediction_png(
            row=row,
            match=exact_match,
            output_path=output_png,
            title=f"EXACT PROBE MATCHES | {row.row_id}",
        )
        manifest_items.append({
            "row_id": row.row_id,
            "image_path": str(row.image_path),
            "output_png": str(output_png),
            "match": exact_match.to_manifest(),
            "parsed_prediction_count": int(detail["parsed_prediction_count"]),
            "renderable_prediction_count": int(detail["renderable_prediction_count"]),
            "nonrenderable_prediction_count": int(detail["nonrenderable_prediction_count"]),
            "invalid_or_unparsed_row_count": int(detail["invalid_or_unparsed_row_count"]),
            "source_artifact": str(detail["source_artifact"]),
        })

    manifest = {
        "schema_version": 1,
        "kind": "forced_continue_exact_full_trajectory",
        "matching": "probe final_snapshot min-cost max-cardinality assignment",
        "items": manifest_items,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Exact Full-Trajectory Forced-Continuation Atlas\n\n"
        "Each PNG renders every finite positive-area prediction in one complete trajectory. "
        "Green and yellow use the probe's final min-cost maximum-cardinality assignment; "
        "red is unmatched under that assignment. Purple dashed boxes are duplicate hints. "
        "Non-renderable and unparsed rows remain counted in `manifest.json`.\n",
        encoding="utf-8",
    )
    print(output / "manifest.json")


if __name__ == "__main__":
    main()

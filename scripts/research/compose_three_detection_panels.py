#!/usr/bin/env python3
"""Compose three shared-renderer prediction panels into one row per image."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.vis.matching import match_row
from src.vis.normalization import VisualRow, load_visual_rows
from src.vis.rendering import render_comparison_png


SCHEMA_VERSION = "three_detection_panel_collage.v2"


def _items(path: Path) -> dict[str, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") not in {"gt_vs_prediction", "prediction_comparison"}:
        raise ValueError(f"unsupported source manifest kind: {manifest.get('kind')!r}")
    return {str(item["row_id"]): item for item in manifest["items"]}


def _scored_jsonl(path: Path) -> Path:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    inputs = manifest.get("inputs", {})
    raw_path = inputs.get("right_scored_jsonl") or inputs.get("scored_jsonl")
    if not raw_path:
        raise ValueError(f"source manifest has no scored JSONL: {path}")
    return Path(raw_path).resolve(strict=True)


def _filter_same_description_duplicates(row: VisualRow) -> tuple[VisualRow, tuple[int, ...]]:
    """Remove redundant unmatched duplicate predictions for presentation only.

    Matched predictions are never removed because overlapping, same-description
    boxes can represent distinct physical owners.  In an all-FP duplicate
    component, the earliest generated prediction is retained deterministically.
    """

    original_match = match_row(row)
    matched = original_match.matched_pred_indices
    adjacency: dict[int, set[int]] = {}
    for candidate in original_match.duplicate_candidates:
        adjacency.setdefault(candidate.pred_a_index, set()).add(candidate.pred_b_index)
        adjacency.setdefault(candidate.pred_b_index, set()).add(candidate.pred_a_index)

    removed: set[int] = set()
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        component: set[int] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency.get(current, ()))
        visited.update(component)
        keep = component & matched
        if not keep:
            keep = {min(component)}
        removed.update(component - keep)

    kept = [pred for pred in row.pred if pred.index not in removed]
    reindexed = tuple(replace(pred, index=index) for index, pred in enumerate(kept))
    return replace(row, pred=reindexed), tuple(sorted(removed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, help="LABEL=manifest.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(args.manifest) != 3:
        raise ValueError("exactly three LABEL=manifest inputs are required")

    sources = []
    for value in args.manifest:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"invalid manifest binding: {value!r}")
        path = Path(raw_path).resolve(strict=True)
        artifacts = load_visual_rows(_scored_jsonl(path))
        rows = {row.row_id: row for row in artifacts.rows}
        sources.append((label, path, _items(path), rows))
    row_ids = list(sources[0][2])
    if any(list(items) != row_ids for _, _, items, _ in sources[1:]):
        raise ValueError("source manifests do not have identical ordered row IDs")
    if any(list(rows) != row_ids for _, _, _, rows in sources):
        raise ValueError("scored artifacts do not have the manifest's ordered row IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=24)
    output_items = []
    with TemporaryDirectory(prefix="coordexp-three-panel-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for row_index, row_id in enumerate(row_ids):
            panels = []
            filter_receipts = []
            for source_index, (label, _, _, rows) in enumerate(sources):
                filtered_row, removed = _filter_same_description_duplicates(rows[row_id])
                filtered_match = match_row(filtered_row)
                rendered = temporary_root / f"{source_index}-{row_index}.png"
                render_comparison_png(
                    row=filtered_row,
                    right_row=filtered_row,
                    left_match=filtered_match,
                    right_match=filtered_match,
                    output_path=rendered,
                    title="",
                    left_label="",
                    right_label="",
                    show_object_labels=False,
                    show_duplicate_hints=False,
                    legend_position="upper_right",
                )
                with Image.open(rendered) as source_image:
                    source = source_image.convert("RGB")
                    panel = source.crop((source.width // 2, 48, source.width, source.height))
                framed = Image.new("RGB", (panel.width, panel.height + 38), "white")
                framed.paste(panel, (0, 38))
                draw = ImageDraw.Draw(framed)
                draw.text((12, 6), label, fill="black", font=font)
                panels.append(framed)
                filter_receipts.append(
                    {
                        "label": label,
                        "removed_original_prediction_indices": list(removed),
                        "retained_prediction_count": len(filtered_row.pred),
                        "match_after_filter": filtered_match.to_manifest(),
                    }
                )
            collage = Image.new(
                "RGB",
                (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
                "white",
            )
            offset = 0
            for panel in panels:
                collage.paste(panel, (offset, 0))
                offset += panel.width
            output = args.output_dir / f"{row_index:04d}_{row_id}_three_checkpoints.png"
            collage.save(output)
            output_items.append(
                {
                    "row_id": row_id,
                    "output_png": str(output.resolve()),
                    "same_description_duplicate_filter": filter_receipts,
                }
            )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "layout": "1x3",
                "panel_order": [label for label, _, _, _ in sources],
                "box_labels": "suppressed; color semantics are shown in the upper-right legend",
                "duplicate_filter": {
                    "scope": "presentation_only",
                    "candidate_rule": "same normalized description and IoU >= 0.30",
                    "retention_rule": "retain all GT-matched predictions; otherwise retain the earliest prediction per connected component",
                    "duplicate_hints": "suppressed",
                },
                "color_semantics": {
                    "green": "true positive / matched",
                    "yellow": "false negative / missed GT",
                    "red": "false positive / unresolved possible omission or hallucination",
                },
                "source_manifests": [str(path) for _, path, _, _ in sources],
                "items": output_items,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()

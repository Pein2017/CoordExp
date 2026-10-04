"""Public visualization APIs for coordexp-infras detection artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.common.errors import ArtifactContractError
from src.vis.matching import MatchResult, match_row
from src.vis.normalization import (
    ArtifactRows,
    VisualRow,
    load_visual_rows,
    select_rows,
)
from src.vis.rendering import (
    PANEL_LEFT_X,
    PANEL_RIGHT_X,
    prepare_render_view,
    render_comparison_png,
    render_gt_vs_prediction_png,
    render_view_manifest,
)


MATCH_IOU_THRESHOLD = 0.50


@dataclass(frozen=True)
class VisualizationResult:
    output_dir: Path
    manifest_path: Path
    summary: str
    image_paths: tuple[Path, ...]


def render_gt_vs_prediction(
    run_dir_or_scored_jsonl: str | Path,
    out_dir: str | Path,
    *,
    row_ids: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
    duplicate_iou_threshold: float = 0.30,
    input_format: str = "scored",
    labels_json: str | Path | None = None,
    crop: Sequence[float] | None = None,
    focus_region: Sequence[float] | None = None,
    focus_gt_indices: Sequence[int] | None = None,
    focus_pred_indices: Sequence[int] | None = None,
    context_alpha: float = 0.18,
) -> VisualizationResult:
    artifacts = load_visual_rows(run_dir_or_scored_jsonl, input_format=input_format, labels_json=labels_json)
    selected_rows = select_rows(artifacts.rows, row_ids=row_ids, limit=limit)
    output_dir = Path(out_dir)
    image_paths: list[Path] = []
    items: list[dict[str, Any]] = []
    prepared = []
    for row in selected_rows:
        match = match_row(
            row,
            match_iou_threshold=MATCH_IOU_THRESHOLD,
            duplicate_iou_threshold=duplicate_iou_threshold,
        )
        view = prepare_render_view(row, crop=crop, focus_region=focus_region,
                                   focus_gt_indices=focus_gt_indices, focus_pred_indices=focus_pred_indices,
                                   context_alpha=context_alpha)
        prepared.append((row, match, view))
    for index, (row, match, view) in enumerate(prepared):
        image_path = output_dir / f"{index:04d}_{_slug(row.row_id)}_gt_vs_pred.png"
        render_gt_vs_prediction_png(
            row=row,
            match=match,
            output_path=image_path,
            title=f"GT VS PREDICTION | {row.row_id}",
            view=view,
        )
        image_paths.append(image_path)
        item = _single_item(row=row, match=match, output_png=image_path)
        item["view"] = {
            "gt": render_view_manifest(row, view, panel_x=PANEL_LEFT_X),
            "prediction": render_view_manifest(row, view, panel_x=PANEL_RIGHT_X),
        }
        items.append(item)
    manifest = _base_manifest(
        kind="gt_vs_prediction",
        duplicate_iou_threshold=duplicate_iou_threshold,
        inputs={"run_dir": str(artifacts.artifact_dir),
                ("rollout_jsonl" if input_format == "rollout" else "scored_jsonl"): str(artifacts.scored_jsonl),
                "input_format": input_format, "labels_json": str(labels_json) if labels_json is not None else None},
        items=items,
    )
    return _write_outputs(
        output_dir=output_dir,
        manifest=manifest,
        image_paths=tuple(image_paths),
        readme=_readme(
            title="GT vs Prediction",
            lines=[
                "Left panel draws GT; right panel draws predictions.",
                "Green means matched, yellow means missing GT, red means unmatched prediction.",
                _coordinate_note(input_format),
            ] + _view_notes(crop=crop, focus_controls=(focus_region, focus_gt_indices, focus_pred_indices)),
            image_paths=image_paths,
        ),
    )


def render_prediction_comparison(
    left_run_dir_or_scored_jsonl: str | Path,
    right_run_dir_or_scored_jsonl: str | Path,
    out_dir: str | Path,
    *,
    left_label: str | None = None,
    right_label: str | None = None,
    row_ids: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
    duplicate_iou_threshold: float = 0.30,
    input_format: str = "scored",
    labels_json: str | Path | None = None,
    crop: Sequence[float] | None = None,
    focus_region: Sequence[float] | None = None,
    focus_gt_indices: Sequence[int] | None = None,
    left_focus_pred_indices: Sequence[int] | None = None,
    right_focus_pred_indices: Sequence[int] | None = None,
    context_alpha: float = 0.18,
) -> VisualizationResult:
    left_artifacts = load_visual_rows(left_run_dir_or_scored_jsonl, input_format=input_format, labels_json=labels_json)
    right_artifacts = load_visual_rows(right_run_dir_or_scored_jsonl, input_format=input_format, labels_json=labels_json)
    selected_left, selected_right = _comparison_rows(
        left_artifacts.rows,
        right_artifacts.rows,
        row_ids=row_ids,
        limit=limit,
    )
    output_dir = Path(out_dir)
    left_name = left_label or left_artifacts.artifact_dir.name
    right_name = right_label or right_artifacts.artifact_dir.name
    image_paths: list[Path] = []
    items: list[dict[str, Any]] = []
    prepared = []
    focus_active = any(value is not None for value in
                       (focus_region, focus_gt_indices, left_focus_pred_indices, right_focus_pred_indices))
    for left_row, right_row in zip(selected_left, selected_right, strict=True):
        _require_comparable_rows(left_row, right_row)
        left_match = match_row(
            left_row,
            match_iou_threshold=MATCH_IOU_THRESHOLD,
            duplicate_iou_threshold=duplicate_iou_threshold,
        )
        right_match = match_row(
            right_row,
            match_iou_threshold=MATCH_IOU_THRESHOLD,
            duplicate_iou_threshold=duplicate_iou_threshold,
        )
        left_view = prepare_render_view(left_row, crop=crop, focus_region=focus_region,
                                        focus_gt_indices=focus_gt_indices,
                                        focus_pred_indices=left_focus_pred_indices if left_focus_pred_indices is not None else ([] if focus_active else None),
                                        context_alpha=context_alpha)
        right_view = prepare_render_view(right_row, crop=crop, focus_region=focus_region,
                                         focus_gt_indices=focus_gt_indices,
                                         focus_pred_indices=right_focus_pred_indices if right_focus_pred_indices is not None else ([] if focus_active else None),
                                         context_alpha=context_alpha)
        prepared.append((left_row, right_row, left_match, right_match, left_view, right_view))
    for index, (left_row, right_row, left_match, right_match, left_view, right_view) in enumerate(prepared):
        image_path = output_dir / f"{index:04d}_{_slug(left_row.row_id)}_prediction_comparison.png"
        render_comparison_png(
            row=left_row,
            right_row=right_row,
            left_match=left_match,
            right_match=right_match,
            output_path=image_path,
            title=f"PREDICTION COMPARISON | {left_row.row_id} | " + ("focused GT shown; other matched GT canceled" if left_view.selected_gt_indices or right_view.selected_gt_indices else "matched GT canceled"),
            left_label=left_name,
            right_label=right_name,
            left_view=left_view,
            right_view=right_view,
        )
        image_paths.append(image_path)
        items.append(
            {
                "row_id": left_row.row_id,
                "image_path": str(left_row.image_path),
                "output_png": str(image_path),
                "gt_objects": [obj.to_manifest() for obj in left_row.gt],
                "left": _run_item(row=left_row, match=left_match),
                "right": _run_item(row=right_row, match=right_match),
                "view": {
                    "left": render_view_manifest(left_row, left_view, panel_x=PANEL_LEFT_X, comparison_match=left_match),
                    "right": render_view_manifest(right_row, right_view, panel_x=PANEL_RIGHT_X, comparison_match=right_match),
                },
            }
        )
    manifest = _base_manifest(
        kind="prediction_comparison",
        duplicate_iou_threshold=duplicate_iou_threshold,
        inputs={
            "left_run_dir": str(left_artifacts.artifact_dir),
            ("left_rollout_jsonl" if input_format == "rollout" else "left_scored_jsonl"): str(left_artifacts.scored_jsonl),
            "right_run_dir": str(right_artifacts.artifact_dir),
            ("right_rollout_jsonl" if input_format == "rollout" else "right_scored_jsonl"): str(right_artifacts.scored_jsonl),
            "left_label": left_name,
            "right_label": right_name,
            "input_format": input_format,
            "labels_json": str(labels_json) if labels_json is not None else None,
        },
        items=items,
    )
    return _write_outputs(
        output_dir=output_dir,
        manifest=manifest,
        image_paths=tuple(image_paths),
        readme=_readme(
            title="Prediction Comparison",
            lines=[
                "Focused GT boxes are shown; other matched GT boxes are canceled/hidden." if any(left_view.selected_gt_indices or right_view.selected_gt_indices for _, _, _, _, left_view, right_view in prepared) else "Matched GT boxes are canceled/hidden.",
                "Green boxes are matched predictions and focused matched GT; yellow boxes are missing GT only." if any(
                    left_view.selected_gt_indices & left_match.matched_gt_indices or right_view.selected_gt_indices & right_match.matched_gt_indices
                    for _, _, left_match, right_match, left_view, right_view in prepared
                ) else "Green boxes are prediction boxes only; yellow boxes are missing GT only.",
                "Red boxes are unmatched predictions; purple dashed boxes are duplicate hints.",
                _coordinate_note(input_format),
            ] + _view_notes(crop=crop, focus_controls=(focus_region, focus_gt_indices, left_focus_pred_indices, right_focus_pred_indices)),
            image_paths=image_paths,
        ),
    )


def _comparison_rows(
    left_rows: tuple[VisualRow, ...],
    right_rows: tuple[VisualRow, ...],
    *,
    row_ids: list[str] | tuple[str, ...] | None,
    limit: int | None,
) -> tuple[tuple[VisualRow, ...], tuple[VisualRow, ...]]:
    if row_ids is None:
        left_ids = [row.row_id for row in left_rows]
        right_ids = [row.row_id for row in right_rows]
        if left_ids != right_ids:
            raise ArtifactContractError(
                "prediction comparison requires identical ordered row ids",
                code="vis.comparison_row_order_mismatch",
                context={"left_row_ids": left_ids, "right_row_ids": right_ids},
            )
        selected_ids: list[str] | tuple[str, ...] | None = None
    else:
        selected_ids = row_ids
    return (
        select_rows(left_rows, row_ids=selected_ids, limit=limit),
        select_rows(right_rows, row_ids=selected_ids, limit=limit),
    )


def _require_comparable_rows(left: VisualRow, right: VisualRow) -> None:
    if left.gt_signature() != right.gt_signature():
        raise ArtifactContractError(
            "prediction comparison rows must share image identity, dimensions, and GT",
            code="vis.comparison_gt_mismatch",
            context={"left_row_id": left.row_id, "right_row_id": right.row_id},
        )


def _single_item(*, row: VisualRow, match: MatchResult, output_png: Path) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "image_path": str(row.image_path),
        "output_png": str(output_png),
        "gt_objects": [obj.to_manifest() for obj in row.gt],
        **_run_item(row=row, match=match),
    }


def _run_item(*, row: VisualRow, match: MatchResult) -> dict[str, Any]:
    item = {
        "match": match.to_manifest(),
        "pred_objects": [obj.to_manifest() for obj in row.pred],
    }
    if getattr(row, "source_metadata", None) is not None:
        item["source_metadata"] = row.source_metadata
    return item


def _view_notes(*, crop: Sequence[float] | None, focus_controls: tuple[Any, ...]) -> list[str]:
    if crop is None and all(value is None for value in focus_controls):
        return []
    return [
        "Counts and matching cover the full image before crop/focus; the cropped view is not rescored.",
        "Focus combines exact object indices with centers inside the source-pixel focus region.",
        "Selected boxes are solid, opaque, labeled, and drawn last; context is faint dashed without labels or duplicate hints.",
        "Purple dashed marks are duplicate diagnostics; INVALID glyphs retain directed endpoints 1->2 and do not match.",
    ]


def _coordinate_note(input_format: str) -> str:
    if input_format == "rollout":
        return "GT and native rollout prediction boxes are converted from norm1000 bins to pixels."
    return "GT boxes are converted from norm1000 bins to pixels; prediction boxes are already pixels."


def _base_manifest(
    *,
    kind: str,
    duplicate_iou_threshold: float,
    inputs: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "match_iou_threshold": MATCH_IOU_THRESHOLD,
        "duplicate_iou_threshold": duplicate_iou_threshold,
        "coordinate_surfaces": {
            "gt_bbox": "norm1000 xyxy coordinate bins converted to pixel xyxy with coord_bins_to_pixel_xyxy",
            "pred_bbox": "native rollout norm1000 xyxy converted to pixel xyxy; original endpoint order is preserved" if inputs.get("input_format") == "rollout" else "pixel xyxy; optional pred.coord_bins is preserved as metadata and never drawn",
            "drawable_field": "bbox_pixel_xyxy",
        },
        "inputs": inputs,
        "items": items,
    }


def _write_outputs(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    image_paths: tuple[Path, ...],
    readme: str,
) -> VisualizationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {**manifest, "summary": readme}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return VisualizationResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        summary=readme,
        image_paths=image_paths,
    )


def _readme(*, title: str, lines: list[str], image_paths: list[Path]) -> str:
    body = [f"# {title}", "", *lines, "", "## Files", ""]
    for path in image_paths:
        body.append(f"- `{path.name}`")
    return "\n".join(body) + "\n"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug or "row"

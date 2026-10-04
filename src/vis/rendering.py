"""PIL rendering for lightweight detection review images."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image, ImageDraw, ImageFont

from src.common.errors import ArtifactContractError
from src.vis.matching import MatchResult
from src.vis.normalization import VisualObject, VisualRow


GREEN = (0, 190, 90)
YELLOW = (255, 210, 20)
RED = (235, 45, 45)
PURPLE = (160, 60, 220)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (240, 240, 240)

CANVAS_WIDTH = 1800
CANVAS_HEIGHT = 830
PANEL_WIDTH = 876
PANEL_LEFT_X = 12
PANEL_RIGHT_X = 912
PANEL_Y = 48


@dataclass(frozen=True)
class RenderView:
    """Display controls in original source-image pixels; never a scoring filter."""

    crop: tuple[float, float, float, float]
    crop_requested: bool
    focus_region: tuple[float, float, float, float] | None
    focus_active: bool
    selected_gt_indices: frozenset[int]
    selected_pred_indices: frozenset[int]
    context_alpha: float


def prepare_render_view(
    row: VisualRow,
    *,
    crop: Sequence[float] | None = None,
    focus_region: Sequence[float] | None = None,
    focus_gt_indices: Sequence[int] | None = None,
    focus_pred_indices: Sequence[int] | None = None,
    context_alpha: float = 0.18,
) -> RenderView:
    bounds = (0.0, 0.0, float(row.image_width), float(row.image_height))
    crop_box = _view_region(crop, row=row, field="crop") if crop is not None else bounds
    region = _view_region(focus_region, row=row, field="focus_region") if focus_region is not None else None
    gt_indices = _focus_indices(focus_gt_indices, row.gt, row=row, field="focus_gt_indices")
    pred_indices = _focus_indices(focus_pred_indices, row.pred, row=row, field="focus_pred_indices")
    if isinstance(context_alpha, bool) or not isinstance(context_alpha, (int, float)) or not math.isfinite(context_alpha) or not 0 <= context_alpha <= 1:
        raise ArtifactContractError("context alpha must be finite and between 0 and 1", code="vis.invalid_context_alpha", context={"context_alpha": repr(context_alpha)})
    if region is not None:
        gt_indices.update(obj.index for obj in row.gt if _center_in_region(obj, region))
        pred_indices.update(obj.index for obj in row.pred if _center_in_region(obj, region))
    return RenderView(crop_box, crop is not None, region,
                      any(value is not None for value in (focus_region, focus_gt_indices, focus_pred_indices)),
                      frozenset(gt_indices), frozenset(pred_indices), float(context_alpha))


def render_view_manifest(row: VisualRow, view: RenderView, *, panel_x: int,
                         comparison_match: MatchResult | None = None) -> dict[str, Any]:
    crop_x1, crop_y1, crop_x2, crop_y2 = view.crop
    fit_width, fit_height = _fit_size(crop_x2 - crop_x1, crop_y2 - crop_y1,
                                    max_width=PANEL_WIDTH - 24, max_height=CANVAS_HEIGHT - 56 - 72)
    image_x = panel_x + (PANEL_WIDTH - fit_width) // 2
    image_y = PANEL_Y + 66
    payload = {
        "matching_scope": "full_image_before_crop_and_focus",
        "source_image_size": [row.image_width, row.image_height],
        "crop_requested": view.crop_requested,
        "crop_transform": {
            "source_crop_pixel_xyxy": list(view.crop),
            "translate_source_pixel_xy": [-crop_x1, -crop_y1],
            "scale_xy": [fit_width / (crop_x2 - crop_x1), fit_height / (crop_y2 - crop_y1)],
            "viewport_canvas_xyxy": [image_x, image_y, image_x + fit_width, image_y + fit_height],
            "clip_to_image_viewport": True,
        },
        "focus_active": view.focus_active,
        "focus_region_pixel_xyxy": list(view.focus_region) if view.focus_region is not None else None,
        "focus_region_selection_rule": "box_center_in_region_inclusive",
        "focus_selection_combination": "union_of_exact_indices_and_region_centers",
        "selected_gt_indices": sorted(view.selected_gt_indices),
        "selected_pred_indices": sorted(view.selected_pred_indices),
        "style": {
            "selected": {"dash": False, "alpha": 1.0, "labels": True, "draw_after_context": True},
            "context": {"dash": True, "alpha": view.context_alpha, "width": 1, "labels": False, "duplicate_hints": False},
            "context_style_applied": view.focus_active,
            "label_placement": "panel_local_offsets_with_leaders_after_annotations" if view.focus_active else "original_anchors",
            "duplicate_hint": "purple dashed diagnostic on selected predictions only when focusing",
            "invalid_prediction": "directed original endpoints 1->2, never a repaired rectangle",
        },
    }
    if comparison_match is not None:
        payload["comparison_gt_visibility"] = {
            "rule": "missing GT plus focused GT, including selected matched GT",
            "revealed_matched_gt_indices": sorted(view.selected_gt_indices & comparison_match.matched_gt_indices),
        }
    return payload


def _view_region(value: Sequence[float], *, row: VisualRow, field: str) -> tuple[float, float, float, float]:
    try:
        if isinstance(value, (str, bytes)) or len(value) != 4 or any(isinstance(item, bool) for item in value):
            raise ValueError
        box = tuple(float(item) for item in value)
        x1, y1, x2, y2 = box
        if not all(math.isfinite(item) for item in box) or not (0 <= x1 < x2 <= row.image_width and 0 <= y1 < y2 <= row.image_height):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactContractError("view region must be finite, ordered, and within the source image", code="vis.invalid_view_region", context={"row_id": row.row_id, "field": field, "value": repr(value)}) from exc
    return box


def _focus_indices(value: Sequence[int] | None, objects: tuple[VisualObject, ...], *, row: VisualRow, field: str) -> set[int]:
    available = {obj.index for obj in objects}
    try:
        indices = list(value) if value is not None else []
        if any(isinstance(index, bool) or not isinstance(index, int) or index not in available for index in indices):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("focus indices must identify existing objects in each selected row", code="vis.invalid_focus_index", context={"row_id": row.row_id, "field": field, "value": repr(value), "available_indices": sorted(available)}) from exc
    return set(indices)


def _center_in_region(obj: VisualObject, region: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = obj.bbox_pixel_xyxy
    rx1, ry1, rx2, ry2 = region
    return rx1 <= x1 / 2 + x2 / 2 <= rx2 and ry1 <= y1 / 2 + y2 / 2 <= ry2


def render_gt_vs_prediction_png(
    *,
    row: VisualRow,
    match: MatchResult,
    output_path: Path,
    title: str,
    view: RenderView | None = None,
) -> None:
    view = view or prepare_render_view(row)
    canvas = _new_canvas(title)
    _draw_panel(
        canvas,
        x=PANEL_LEFT_X,
        y=PANEL_Y,
        width=PANEL_WIDTH,
        height=CANVAS_HEIGHT - 56,
        title="GT",
        row=row,
        match=match,
        mode="gt",
        view=view,
    )
    _draw_panel(
        canvas,
        x=PANEL_RIGHT_X,
        y=PANEL_Y,
        width=PANEL_WIDTH,
        height=CANVAS_HEIGHT - 56,
        title="Prediction",
        row=row,
        match=match,
        mode="prediction",
        view=view,
    )
    _save(canvas, output_path)


def render_comparison_png(
    *,
    row: VisualRow,
    left_match: MatchResult,
    right_match: MatchResult,
    right_row: VisualRow,
    output_path: Path,
    title: str,
    left_label: str,
    right_label: str,
    show_object_labels: bool = True,
    show_duplicate_hints: bool = True,
    legend_position: Literal["left", "upper_right"] = "left",
    left_view: RenderView | None = None,
    right_view: RenderView | None = None,
) -> None:
    left_view = left_view or prepare_render_view(row)
    right_view = right_view or prepare_render_view(right_row)
    canvas = _new_canvas(title)
    _draw_panel(
        canvas,
        x=PANEL_LEFT_X,
        y=PANEL_Y,
        width=PANEL_WIDTH,
        height=CANVAS_HEIGHT - 56,
        title=left_label,
        row=row,
        match=left_match,
        mode="comparison",
        show_object_labels=show_object_labels,
        show_duplicate_hints=show_duplicate_hints,
        legend_position=legend_position,
        view=left_view,
    )
    _draw_panel(
        canvas,
        x=PANEL_RIGHT_X,
        y=PANEL_Y,
        width=PANEL_WIDTH,
        height=CANVAS_HEIGHT - 56,
        title=right_label,
        row=right_row,
        match=right_match,
        mode="comparison",
        show_object_labels=show_object_labels,
        show_duplicate_hints=show_duplicate_hints,
        legend_position=legend_position,
        view=right_view,
    )
    _save(canvas, output_path)


def _new_canvas(title: str) -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), GRAY)
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=BLACK, font=_font("bold", 24))
    return canvas


def _draw_panel(
    canvas: Image.Image,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    row: VisualRow,
    match: MatchResult,
    mode: Literal["gt", "prediction", "comparison"],
    show_object_labels: bool = True,
    show_duplicate_hints: bool = True,
    legend_position: Literal["left", "upper_right"] = "left",
    view: RenderView,
) -> None:
    draw = ImageDraw.Draw(canvas)
    stats = match.stats()
    draw.text((x, y), title, fill=BLACK, font=_font("bold", 18))
    stats_text = (
        f"TP={stats['tp']} FN={stats['fn']} FP={stats['fp']} "
        f"P={stats['precision']:.2f} R={stats['recall']:.2f} "
        f"F1={stats['f1']:.2f}"
    )
    if show_duplicate_hints:
        stats_text += f" | dup-cand={len(match.duplicate_candidates)}"
    invalid_count = sum(not getattr(pred, "geometry_valid", True) for pred in row.pred)
    if invalid_count:
        stats_text += f" | invalid={invalid_count}"
    if view.crop_requested or view.focus_active:
        stats_text += " | full-image counts"
    draw.text(
        (x, y + 24),
        stats_text,
        fill=BLACK,
        font=_font("regular", 12),
    )
    if view.focus_active or invalid_count:
        legend = "faint dashed=context (unlabeled); purple dashed=dup hint; INVALID=endpoint 1->2"
        if mode == "comparison" and view.selected_gt_indices:
            legend += "; focused GT shown"
        draw.text((x, y + 42), legend, fill=BLACK, font=_font("regular", 10))
    elif legend_position == "upper_right":
        entries = (
            ((GREEN, "TP"), (YELLOW, "FN"), (RED, "FP"))
            if mode != "gt"
            else ((GREEN, "TP"), (YELLOW, "FN"))
        )
        _draw_color_legend(draw, right=x + width, y=y + 40, entries=entries)
    else:
        legend = (
            "green=matched pred  yellow=missing GT  red=FP  purple dashed=dup hint"
            if mode != "gt" and show_duplicate_hints
            else "green=matched pred  yellow=missing GT  red=FP"
            if mode != "gt"
            else "green=matched GT  yellow=missing GT"
        )
        draw.text((x, y + 42), legend, fill=BLACK, font=_font("regular", 10))

    image = _load_image(row.image_path)
    if image.size != (row.image_width, row.image_height):
        raise ArtifactContractError("image dimensions do not match visualization artifact metadata", code="vis.image_dimension_mismatch", context={"row_id": row.row_id, "path": str(row.image_path), "actual_size": list(image.size), "artifact_size": [row.image_width, row.image_height]})
    crop_x1, crop_y1, crop_x2, crop_y2 = view.crop
    crop_width, crop_height = crop_x2 - crop_x1, crop_y2 - crop_y1
    if view.crop_requested:
        fit_size = _fit_size(crop_width, crop_height, max_width=width - 24, max_height=height - 72)
        fit = image.transform(fit_size, Image.Transform.EXTENT, view.crop, resample=Image.Resampling.BICUBIC)
    else:
        fit, _ = _fit_image(image, max_width=width - 24, max_height=height - 72)
    image_x = x + (width - fit.width) // 2
    image_y = y + 66
    scale_x = fit.width / crop_width
    scale_y = fit.height / crop_height
    # All annotations live on this viewport-sized image, so lines and labels
    # cannot paint over headings, panel gutters, or the neighboring image.
    foreground = ImageDraw.Draw(fit)
    context = Image.new("RGBA", fit.size, (0, 0, 0, 0))
    context_draw = ImageDraw.Draw(context)
    transform = {"scale_x": scale_x, "scale_y": scale_y,
                 "dx": -crop_x1 * scale_x, "dy": -crop_y1 * scale_y}

    if mode == "gt":
        _draw_gt_panel(foreground, context_draw=context_draw, row=row, match=match,
                       transform=transform, image_size=fit.size, view=view, layer="context")
    else:
        _draw_prediction_panel(
            foreground,
            context_draw=context_draw,
            row=row,
            match=match,
            transform=transform,
            image_size=fit.size,
            view=view,
            layer="context",
            show_missing_gt=(mode == "comparison"),
            show_object_labels=show_object_labels,
            show_duplicate_hints=show_duplicate_hints,
        )
    fit = Image.alpha_composite(fit.convert("RGBA"), context).convert("RGB")
    foreground = ImageDraw.Draw(fit)
    focus_labels = [] if view.focus_active else None
    if mode == "gt":
        _draw_gt_panel(foreground, context_draw=context_draw, row=row, match=match,
                       transform=transform, image_size=fit.size, view=view, layer="foreground", focus_labels=focus_labels)
    else:
        _draw_prediction_panel(foreground, context_draw=context_draw, row=row, match=match,
                               transform=transform, image_size=fit.size, view=view, layer="foreground",
                               show_missing_gt=(mode == "comparison"), show_object_labels=show_object_labels,
                               show_duplicate_hints=show_duplicate_hints, focus_labels=focus_labels)
    if focus_labels is not None:
        _draw_focus_labels(foreground, focus_labels, viewport=fit.size)
    canvas.paste(fit, (image_x, image_y))


def _draw_gt_panel(
    draw: ImageDraw.ImageDraw,
    *,
    row: VisualRow,
    match: MatchResult,
    context_draw: ImageDraw.ImageDraw,
    transform: dict[str, float],
    image_size: tuple[int, int],
    view: RenderView,
    layer: Literal["context", "foreground"],
    focus_labels: list | None = None,
) -> None:
    matched_gt = match.matched_gt_indices
    for gt in row.gt:
        selected = not view.focus_active or gt.index in view.selected_gt_indices
        if selected != (layer == "foreground"):
            continue
        color = GREEN if gt.index in matched_gt else YELLOW
        label = f"G{gt.index} {gt.description}" if gt.index in matched_gt else f"MISS G{gt.index} {gt.description}"
        box = _scale_box(gt.bbox_pixel_xyxy, **transform)
        if selected:
            draw.rectangle(box, outline=color, width=4)
            _label(draw, (box[0] + 2, max(0, box[1] + 2)), label, color, focus_labels=focus_labels)
        else:
            _dashed_rect(context_draw, box, (*color, round(255 * view.context_alpha)), width=1, viewport=image_size)


def _draw_prediction_panel(
    draw: ImageDraw.ImageDraw,
    *,
    row: VisualRow,
    match: MatchResult,
    context_draw: ImageDraw.ImageDraw,
    transform: dict[str, float],
    image_size: tuple[int, int],
    view: RenderView,
    layer: Literal["context", "foreground"],
    show_missing_gt: bool,
    show_object_labels: bool,
    show_duplicate_hints: bool,
    focus_labels: list | None = None,
) -> None:
    if show_missing_gt:
        gt_by_index = {gt.index: gt for gt in row.gt}
        visible_gt_indices = sorted(set(match.missing_gt_indices) | set(view.selected_gt_indices))
        for gt_index in visible_gt_indices:
            gt = gt_by_index[gt_index]
            selected = not view.focus_active or gt.index in view.selected_gt_indices
            if selected != (layer == "foreground"):
                continue
            box = _scale_box(gt.bbox_pixel_xyxy, **transform)
            color = GREEN if gt_index in match.matched_gt_indices else YELLOW
            text = f"G{gt.index} {gt.description}" if gt_index in match.matched_gt_indices else f"MISS G{gt.index} {gt.description}"
            if selected:
                draw.rectangle(box, outline=color, width=5)
                if show_object_labels or view.focus_active:
                    _label(draw, (box[0] + 2, max(0, box[1] + 2)), text, color, focus_labels=focus_labels)
            else:
                _dashed_rect(context_draw, box, (*color, round(255 * view.context_alpha)), width=1, viewport=image_size)

    match_by_pred = {pair.pred_index: pair for pair in match.matches}
    for pred in row.pred:
        selected = not view.focus_active or pred.index in view.selected_pred_indices
        if selected != (layer == "foreground"):
            continue
        pair = match_by_pred.get(pred.index)
        box = _scale_box(pred.bbox_pixel_xyxy, **transform)
        valid = getattr(pred, "geometry_valid", True)
        if not valid:
            color, text, width = RED, f"INVALID P{pred.index} {pred.description} (1->2)", 4
        elif pair is None:
            color = RED
            text = f"FP P{pred.index} {pred.description}"
            width = 5
        else:
            color = GREEN
            text = f"P{pred.index}->G{pair.gt_index} {pred.description} {pair.iou:.2f}"
            width = 4
        target_draw = draw if selected else context_draw
        target_color = color if selected else (*color, round(255 * view.context_alpha))
        if not valid:
            _invalid_glyph(target_draw, box, target_color, width=width if selected else 1, dashed=not selected, viewport=image_size)
        elif selected:
            draw.rectangle(box, outline=color, width=width)
        else:
            _dashed_rect(target_draw, box, target_color, width=1, viewport=image_size)
        if selected and (show_object_labels or view.focus_active):
            y = min(image_size[1] - 14, max(0, box[1] - 14))
            _label(draw, (box[0] + 2, y), text, color, focus_labels=focus_labels)

    if not show_duplicate_hints or layer != "foreground":
        return
    duplicate_pred_indices = sorted(
        {
            index
            for candidate in match.duplicate_candidates
            for index in (candidate.pred_a_index, candidate.pred_b_index)
        }
    )
    pred_by_index = {pred.index: pred for pred in row.pred}
    for pred_index in duplicate_pred_indices:
        pred = pred_by_index[pred_index]
        if not getattr(pred, "geometry_valid", True) or (view.focus_active and pred.index not in view.selected_pred_indices):
            continue
        box = _scale_box(pred.bbox_pixel_xyxy, **transform)
        _dashed_rect(draw, box, PURPLE, width=3, viewport=image_size)
        y = min(image_size[1] - 28, max(0, box[3] + 2))
        _label(draw, (box[0] + 2, y), f"DUP? P{pred.index}", PURPLE, focus_labels=focus_labels)


def _draw_color_legend(
    draw: ImageDraw.ImageDraw,
    *,
    right: int,
    y: int,
    entries: tuple[tuple[tuple[int, int, int], str], ...],
) -> None:
    """Draw a compact color-only legend away from small object boxes."""

    font = _font("regular", 11)
    swatch = 10
    gap = 10
    widths = []
    for _, label in entries:
        bbox = draw.textbbox((0, 0), label, font=font)
        widths.append(swatch + 4 + (bbox[2] - bbox[0]))
    total_width = sum(widths) + gap * (len(entries) - 1)
    x = right - total_width
    for (color, label), width in zip(entries, widths, strict=True):
        draw.rectangle((x, y + 1, x + swatch, y + 1 + swatch), outline=color, width=3)
        draw.text((x + swatch + 4, y), label, fill=BLACK, font=font)
        x += width + gap


def _load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise ArtifactContractError(
            "visualization image file is missing",
            code="vis.missing_image_file",
            context={"path": str(path)},
        )
    with Image.open(path) as image:
        return image.convert("RGB")


def _fit_image(image: Image.Image, *, max_width: int, max_height: int) -> tuple[Image.Image, float]:
    scale = min(max_width / image.width, max_height / image.height)
    return image.resize(_fit_size(image.width, image.height, max_width=max_width, max_height=max_height)), scale


def _fit_size(width: float, height: float, *, max_width: int, max_height: int) -> tuple[int, int]:
    scale = min(max_width / width, max_height / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _scale_box(
    box: tuple[float, float, float, float],
    *,
    scale_x: float,
    scale_y: float,
    dx: float,
    dy: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return dx + x1 * scale_x, dy + y1 * scale_y, dx + x2 * scale_x, dy + y2 * scale_y


def _label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    *,
    focus_labels: list | None = None,
) -> None:
    if focus_labels is not None:
        focus_labels.append((xy, text, color))
        return
    x, y = xy
    font = _font("regular", 10)
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=WHITE,
        outline=color,
        width=1,
    )
    draw.text((x, y), text, fill=BLACK, font=font)


def _draw_focus_labels(draw: ImageDraw.ImageDraw, labels: list, *, viewport: tuple[int, int]) -> None:
    """Place the small selected pool without overlapping, then draw labels last."""
    occupied = []
    placed = []
    font = _font("regular", 10)
    for preferred, text, color in labels:
        glyph = draw.textbbox((0, 0), text, font=font)
        width, height = glyph[2] - glyph[0] + 4, glyph[3] - glyph[1] + 4
        left = min(max(0, int(preferred[0] + glyph[0] - 2)), max(0, viewport[0] - width))
        top = min(max(0, int(preferred[1] + glyph[1] - 2)), max(0, viewport[1] - height))
        rows = [top, *range(top + height + 3, viewport[1] - height + 1, height + 3),
                *range(top - height - 3, -1, -height - 3)]
        candidates = ((x, y, x + width, y + height)
                      for x in [left, *range(0, viewport[0] - width + 1, width + 3)]
                      for y in rows)
        box = next((box for box in candidates if all(
            box[2] + 1 < other[0] or other[2] + 1 < box[0] or
            box[3] + 1 < other[1] or other[3] + 1 < box[1]
            for other in occupied)), (left, top, left + width, top + height))
        occupied.append(box)
        xy = (box[0] + 2 - glyph[0], box[1] + 2 - glyph[1])
        placed.append((preferred, xy, box, text, color))
    # Leaders also precede every label so later lines cannot erase earlier IDs.
    for preferred, xy, box, _, color in placed:
        if abs(xy[0] - preferred[0]) > 2 or abs(xy[1] - preferred[1]) > 2:
            target = (min(max(preferred[0], box[0]), box[2]), min(max(preferred[1], box[1]), box[3]))
            draw.line((*preferred, *target), fill=color, width=1)
    for _, xy, _, text, color in placed:
        _label(draw, xy, text, color)


def _dashed_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: tuple[int, ...],
    *,
    width: int,
    dash: int = 10,
    gap: int = 6,
    viewport: tuple[int, int] | None = None,
) -> None:
    x1, y1, x2, y2 = box
    # Start at the first visible dash while retaining its source-anchored phase.
    x_pos = x1 + max(0, math.floor(-x1 / (dash + gap))) * (dash + gap) if viewport else x1
    x_stop = min(x2, viewport[0]) if viewport else x2
    while x_pos < x_stop:
        draw.line((x_pos, y1, min(x_pos + dash, x2), y1), fill=color, width=width)
        draw.line((x_pos, y2, min(x_pos + dash, x2), y2), fill=color, width=width)
        x_pos += dash + gap
    y_pos = y1 + max(0, math.floor(-y1 / (dash + gap))) * (dash + gap) if viewport else y1
    y_stop = min(y2, viewport[1]) if viewport else y2
    while y_pos < y_stop:
        draw.line((x1, y_pos, x1, min(y_pos + dash, y2)), fill=color, width=width)
        draw.line((x2, y_pos, x2, min(y_pos + dash, y2)), fill=color, width=width)
        y_pos += dash + gap


def _invalid_glyph(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], color: tuple[int, ...], *, width: int, dashed: bool, viewport: tuple[int, int]) -> None:
    """Show the original ordered endpoints without implying valid box area."""
    x1, y1, x2, y2 = box
    delta_x, delta_y = x2 - x1, y2 - y1
    length = math.hypot(delta_x, delta_y)
    if dashed and length:
        # Clip the segment parametrically before walking the visible dashes.
        low, high = 0.0, 1.0
        for start, delta, limit in ((x1, delta_x, viewport[0]), (y1, delta_y, viewport[1])):
            if delta:
                a, b = -start / delta, (limit - start) / delta
                low, high = max(low, min(a, b)), min(high, max(a, b))
            elif not 0 <= start <= limit:
                low, high = 1.0, 0.0
        position = max(0, math.floor(low * length / 16)) * 16
        while low <= high and position < high * length:
            end = min(position + 10, length)
            draw.line((x1 + delta_x * position / length, y1 + delta_y * position / length,
                       x1 + delta_x * end / length, y1 + delta_y * end / length), fill=color, width=width)
            position += 16
    else:
        draw.line(box, fill=color, width=width)
    radius = 4 if not dashed else 2
    for px, py in ((x1, y1), (x2, y2)):
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color, width=width)
    if length:
        ux, uy = delta_x / length, delta_y / length
        arrow = 12 if not dashed else 6
        draw.polygon(((x2, y2), (x2 - arrow * ux + arrow / 2 * uy, y2 - arrow * uy - arrow / 2 * ux),
                      (x2 - arrow * ux - arrow / 2 * uy, y2 - arrow * uy + arrow / 2 * ux)), fill=color)


def _font(kind: str, size: int) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if kind == "bold" else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _save(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

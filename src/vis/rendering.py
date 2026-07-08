"""PIL rendering for lightweight detection review images."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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


def render_gt_vs_prediction_png(
    *,
    row: VisualRow,
    match: MatchResult,
    output_path: Path,
    title: str,
) -> None:
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
) -> None:
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
) -> None:
    draw = ImageDraw.Draw(canvas)
    stats = match.stats()
    draw.text((x, y), title, fill=BLACK, font=_font("bold", 18))
    draw.text(
        (x, y + 24),
        (
            f"TP={stats['tp']} FN={stats['fn']} FP={stats['fp']} "
            f"P={stats['precision']:.2f} R={stats['recall']:.2f} "
            f"F1={stats['f1']:.2f} | dup-cand={len(match.duplicate_candidates)}"
        ),
        fill=BLACK,
        font=_font("regular", 12),
    )
    legend = (
        "green=matched pred  yellow=missing GT  red=FP  purple dashed=dup hint"
        if mode != "gt"
        else "green=matched GT  yellow=missing GT"
    )
    draw.text((x, y + 42), legend, fill=BLACK, font=_font("regular", 10))

    image = _load_image(row.image_path)
    fit, scale = _fit_image(image, max_width=width - 24, max_height=height - 72)
    image_x = x + (width - fit.width) // 2
    image_y = y + 66
    canvas.paste(fit, (image_x, image_y))
    scale_x = fit.width / row.image_width
    scale_y = fit.height / row.image_height

    if mode == "gt":
        _draw_gt_panel(draw, row=row, match=match, image_x=image_x, image_y=image_y, scale_x=scale_x, scale_y=scale_y)
    else:
        _draw_prediction_panel(
            draw,
            row=row,
            match=match,
            image_x=image_x,
            image_y=image_y,
            image_h=fit.height,
            scale_x=scale_x,
            scale_y=scale_y,
            show_missing_gt=(mode == "comparison"),
        )


def _draw_gt_panel(
    draw: ImageDraw.ImageDraw,
    *,
    row: VisualRow,
    match: MatchResult,
    image_x: int,
    image_y: int,
    scale_x: float,
    scale_y: float,
) -> None:
    matched_gt = match.matched_gt_indices
    for gt in row.gt:
        color = GREEN if gt.index in matched_gt else YELLOW
        label = f"G{gt.index} {gt.description}" if gt.index in matched_gt else f"MISS G{gt.index} {gt.description}"
        box = _scale_box(gt.bbox_pixel_xyxy, scale_x=scale_x, scale_y=scale_y, dx=image_x, dy=image_y)
        draw.rectangle(box, outline=color, width=4)
        _label(draw, (box[0] + 2, max(image_y, box[1] + 2)), label, color)


def _draw_prediction_panel(
    draw: ImageDraw.ImageDraw,
    *,
    row: VisualRow,
    match: MatchResult,
    image_x: int,
    image_y: int,
    image_h: int,
    scale_x: float,
    scale_y: float,
    show_missing_gt: bool,
) -> None:
    if show_missing_gt:
        for gt_index in match.missing_gt_indices:
            gt = row.gt[gt_index]
            box = _scale_box(gt.bbox_pixel_xyxy, scale_x=scale_x, scale_y=scale_y, dx=image_x, dy=image_y)
            draw.rectangle(box, outline=YELLOW, width=5)
            _label(draw, (box[0] + 2, max(image_y, box[1] + 2)), f"MISS G{gt.index} {gt.description}", YELLOW)

    match_by_pred = {pair.pred_index: pair for pair in match.matches}
    for pred in row.pred:
        pair = match_by_pred.get(pred.index)
        box = _scale_box(pred.bbox_pixel_xyxy, scale_x=scale_x, scale_y=scale_y, dx=image_x, dy=image_y)
        if pair is None:
            color = RED
            text = f"FP P{pred.index} {pred.description}"
            width = 5
        else:
            color = GREEN
            text = f"P{pred.index}->G{pair.gt_index} {pred.description} {pair.iou:.2f}"
            width = 4
        draw.rectangle(box, outline=color, width=width)
        y = min(image_y + image_h - 14, max(image_y, box[1] - 14))
        _label(draw, (box[0] + 2, y), text, color)

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
        box = _scale_box(pred.bbox_pixel_xyxy, scale_x=scale_x, scale_y=scale_y, dx=image_x, dy=image_y)
        _dashed_rect(draw, box, PURPLE, width=3)
        y = min(image_y + image_h - 28, max(image_y, box[3] + 2))
        _label(draw, (box[0] + 2, y), f"DUP? P{pred.index}", PURPLE)


def _load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise ArtifactContractError(
            "visualization image file is missing",
            code="vis.missing_image_file",
            context={"path": str(path)},
        )
    return Image.open(path).convert("RGB")


def _fit_image(image: Image.Image, *, max_width: int, max_height: int) -> tuple[Image.Image, float]:
    scale = min(max_width / image.width, max_height / image.height)
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    return image.resize((width, height)), scale


def _scale_box(
    box: tuple[float, float, float, float],
    *,
    scale_x: float,
    scale_y: float,
    dx: int,
    dy: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return dx + x1 * scale_x, dy + y1 * scale_y, dx + x2 * scale_x, dy + y2 * scale_y


def _label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
) -> None:
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


def _dashed_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    *,
    width: int,
    dash: int = 10,
    gap: int = 6,
) -> None:
    x1, y1, x2, y2 = box
    x_pos = x1
    while x_pos < x2:
        draw.line((x_pos, y1, min(x_pos + dash, x2), y1), fill=color, width=width)
        draw.line((x_pos, y2, min(x_pos + dash, x2), y2), fill=color, width=width)
        x_pos += dash + gap
    y_pos = y1
    while y_pos < y2:
        draw.line((x1, y_pos, x1, min(y_pos + dash, y2)), fill=color, width=width)
        draw.line((x2, y_pos, x2, min(y_pos + dash, y2)), fill=color, width=width)
        y_pos += dash + gap


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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


GALLERY_LABELS = (
    "GT object",
    "Random prediction",
    "Sorted prediction",
    "FN target",
    "Emitted same-desc",
    "Residual same-desc",
    "x1 peak",
)
LEGEND_PLACEMENT = "right_panel"
LEGEND_WIDTH = 260
PLACEHOLDER_LABEL = "Missing image placeholder"

COLORS = {
    "GT object": (24, 102, 210),
    "Random prediction": (220, 47, 47),
    "Sorted prediction": (30, 150, 70),
    "FN target": (210, 42, 190),
    "Emitted same-desc": (236, 144, 28),
    "Residual same-desc": (12, 165, 180),
    "x1 peak": (120, 55, 210),
}


def build_galleries(
    artifact_root: str | Path,
    *,
    native_cases: Sequence[Mapping[str, Any]],
    fn_cases: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    root = Path(artifact_root)
    gallery_meta = build_gallery(
        root / "gallery",
        native_cases,
        title="A3.2 Native Rollout Gallery",
    )
    fn_gallery_meta = build_gallery(
        root / "fn_probe" / "gallery",
        fn_cases,
        title="A3.2 FN Probe Gallery",
    )
    return {
        "gallery": gallery_meta,
        "fn_probe_gallery": fn_gallery_meta,
    }


def build_gallery(
    gallery_root: str | Path,
    cases: Sequence[Mapping[str, Any]],
    *,
    title: str,
) -> list[dict[str, Any]]:
    root = Path(gallery_root)
    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_id = _case_id(case, index)
        image_path = image_root / f"{case_id}.jpg"
        item = render_gallery_image(case, image_path, case_id=case_id)
        metadata.append(item)
    _write_index(root / "index.md", title=title, metadata=metadata)
    _write_metadata(root / "metadata.json", metadata)
    return metadata


def render_gallery_image(
    case: Mapping[str, Any],
    output_path: str | Path,
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    Image, ImageDraw, ImageFont = _lazy_pil()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(str(case.get("image_path", "")))
    source_status = "loaded"
    if source_path.is_file():
        image = Image.open(source_path).convert("RGB")
    else:
        source_status = "missing_placeholder"
        width = max(64, int(case.get("width") or 128))
        height = max(48, int(case.get("height") or 96))
        image = Image.new("RGB", (width, height), color=(238, 238, 238))
    image_width, image_height = image.size
    legend_height = max(image_height, 28 + 24 * len(GALLERY_LABELS))
    canvas = Image.new(
        "RGB",
        (image_width + LEGEND_WIDTH, legend_height),
        color=(255, 255, 255),
    )
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if source_status == "missing_placeholder":
        draw.rectangle([0, 0, image_width - 1, image_height - 1], outline=(120, 120, 120), width=2)
        draw.text((8, 8), PLACEHOLDER_LABEL, fill=(70, 70, 70), font=font)

    scale_x = image_width / max(1, float(case.get("width") or image_width))
    scale_y = image_height / max(1, float(case.get("height") or image_height))
    _draw_box_rows(
        draw,
        case.get("gt_objects", ()),
        label="GT object",
        scale_x=scale_x,
        scale_y=scale_y,
        width=2,
    )
    _draw_box_rows(
        draw,
        case.get("random_predictions", ()),
        label="Random prediction",
        scale_x=scale_x,
        scale_y=scale_y,
        width=2,
    )
    _draw_box_rows(
        draw,
        case.get("sorted_predictions", ()),
        label="Sorted prediction",
        scale_x=scale_x,
        scale_y=scale_y,
        width=2,
    )
    fn_target = case.get("fn_target")
    if isinstance(fn_target, Mapping):
        _draw_box_rows(
            draw,
            [fn_target],
            label="FN target",
            scale_x=scale_x,
            scale_y=scale_y,
            width=4,
        )
    _draw_box_rows(
        draw,
        case.get("emitted_same_desc", ()),
        label="Emitted same-desc",
        scale_x=scale_x,
        scale_y=scale_y,
        width=3,
    )
    _draw_box_rows(
        draw,
        case.get("residual_same_desc", ()),
        label="Residual same-desc",
        scale_x=scale_x,
        scale_y=scale_y,
        width=3,
    )
    _draw_x1_peaks(draw, case.get("x1_peaks", ()), scale_x=scale_x, image_height=image_height)

    legend_bbox = [image_width, 0, image_width + LEGEND_WIDTH, legend_height]
    draw.rectangle(legend_bbox, fill=(250, 250, 250), outline=(215, 215, 215))
    draw.text((image_width + 12, 10), "Legend", fill=(20, 20, 20), font=font)
    y = 32
    for label in GALLERY_LABELS:
        color = COLORS[label]
        draw.rectangle([image_width + 12, y, image_width + 30, y + 12], fill=color)
        draw.text((image_width + 38, y - 1), label, fill=(20, 20, 20), font=font)
        y += 24

    canvas.save(output, format="JPEG", quality=92)
    image_panel_bbox = [0, 0, image_width, image_height]
    metadata = {
        "case_id": str(case_id or case.get("case_id") or case.get("image_id") or output.stem),
        "output_path": str(output),
        "relative_image_path": f"images/{output.name}",
        "image_source": str(source_path),
        "image_source_status": source_status,
        "image_panel_bbox": image_panel_bbox,
        "legend_bbox": legend_bbox,
        "legend_placement": LEGEND_PLACEMENT,
        "legend_overlaps_image": _boxes_overlap(image_panel_bbox, legend_bbox),
        "labels": list(GALLERY_LABELS),
        "placeholder_label": PLACEHOLDER_LABEL if source_status == "missing_placeholder" else None,
    }
    return _json_safe(metadata)


def _draw_box_rows(
    draw: Any,
    rows: Any,
    *,
    label: str,
    scale_x: float,
    scale_y: float,
    width: int,
) -> None:
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        return
    color = COLORS[label]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        bbox = _bbox(row)
        if bbox is None:
            continue
        x1, y1, x2, y2 = _scaled_bbox(bbox, scale_x=scale_x, scale_y=scale_y)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = _short_label(label)
        draw.text((x1 + 2, max(0, y1 + 2)), text, fill=color)


def _draw_x1_peaks(
    draw: Any,
    peaks: Any,
    *,
    scale_x: float,
    image_height: int,
) -> None:
    if not isinstance(peaks, Sequence) or isinstance(peaks, str | bytes):
        return
    color = COLORS["x1 peak"]
    for peak in peaks:
        if not isinstance(peak, Mapping):
            continue
        value = peak.get("x1", peak.get("peak_value"))
        if value is None:
            continue
        x = int(round(float(value) * scale_x))
        draw.line([x, 0, x, image_height], fill=color, width=2)
        draw.text((x + 2, 2), "x1", fill=color)


def _write_index(path: Path, *, title: str, metadata: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"Legend placement: `{LEGEND_PLACEMENT}`.",
        "",
        "Legend labels:",
    ]
    lines.extend(f"- {label}" for label in GALLERY_LABELS)
    if any(item.get("image_source_status") == "missing_placeholder" for item in metadata):
        lines.extend(["", f"Source note: {PLACEHOLDER_LABEL}."])
    lines.append("")
    lines.append("Images:")
    if not metadata:
        lines.append("- No gallery cases were provided.")
    for item in metadata:
        note = ""
        if item.get("image_source_status") == "missing_placeholder":
            note = f" - {PLACEHOLDER_LABEL}"
        lines.append(
            f"- {item['case_id']}: ![]({item['relative_image_path']}){note}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metadata(path: Path, metadata: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(metadata), allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _lazy_pil() -> tuple[Any, Any, Any]:
    from PIL import Image, ImageDraw, ImageFont

    return Image, ImageDraw, ImageFont


def _case_id(case: Mapping[str, Any], index: int) -> str:
    raw = str(case.get("case_id") or case.get("fn_case_id") or case.get("image_id") or f"case-{index}")
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in raw)
    return cleaned.strip("-") or f"case-{index}"


def _bbox(row: Mapping[str, Any]) -> list[float] | None:
    value = row.get("bbox", row.get("bbox_xyxy", row.get("fn_bbox")))
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        return None
    return [float(item) for item in value]


def _scaled_bbox(
    bbox: Sequence[float],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )


def _short_label(label: str) -> str:
    return {
        "GT object": "GT",
        "Random prediction": "R",
        "Sorted prediction": "S",
        "FN target": "FN",
        "Emitted same-desc": "E",
        "Residual same-desc": "Res",
    }.get(label, label)


def _boxes_overlap(left: Sequence[int], right: Sequence[int]) -> bool:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    return not (
        left_x2 <= right_x1
        or right_x2 <= left_x1
        or left_y2 <= right_y1
        or right_y2 <= left_y1
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "GALLERY_LABELS",
    "LEGEND_PLACEMENT",
    "PLACEHOLDER_LABEL",
    "build_galleries",
    "build_gallery",
    "render_gallery_image",
]

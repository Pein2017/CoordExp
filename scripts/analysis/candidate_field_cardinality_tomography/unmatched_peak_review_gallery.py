#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_ET_ROOT = Path(
    "/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/"
    "candidate_field_cardinality_tomography_representative8192"
)
DEFAULT_PURE_ROOT = Path(
    "/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_purece/"
    "candidate_field_cardinality_tomography_representative8192"
)
DEFAULT_PHASE_A2_ROOT = Path(
    "/data/CoordExp/outputs/analysis/autoreg_object_rollout/"
    "candidate_field_cardinality_comparisons/et_rmp_ce_vs_purece_representative8192/"
    "phase_a2_dual_checkpoint_analysis"
)
DEFAULT_IMAGE_ROOTS = (
    Path("/data/CoordExp/public_data/lvis/raw"),
    Path("/data/CoordExp/public_data/coco/raw"),
    Path("/data/CoordExp/public_data"),
    Path("/data/Qwen3-VL/public_data/lvis/raw"),
    Path("/data/Qwen3-VL/public_data"),
)

COORD_RE = re.compile(r"coord_(\d+)")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def coord_token_to_int(token: Any) -> int | None:
    if isinstance(token, int):
        return token
    if isinstance(token, float):
        return int(token)
    match = COORD_RE.search(str(token))
    if match:
        return int(match.group(1))
    return None


def parse_bbox_tokens(tokens: Any) -> list[int] | None:
    if not isinstance(tokens, list) or len(tokens) != 4:
        return None
    coords = [coord_token_to_int(token) for token in tokens]
    if any(coord is None for coord in coords):
        return None
    return [int(coord) for coord in coords]


def canonical_desc(text: Any) -> str:
    return " ".join(str(text).strip().lower().split())


def x_to_px(x_norm1000: float, width: int) -> int:
    return int(round(max(0.0, min(1000.0, x_norm1000)) / 1000.0 * width))


def bbox_to_px(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1p = x_to_px(x1, width)
    x2p = x_to_px(x2, width)
    y1p = int(round(max(0.0, min(1000.0, y1)) / 1000.0 * height))
    y2p = int(round(max(0.0, min(1000.0, y2)) / 1000.0 * height))
    return min(x1p, x2p), min(y1p, y2p), max(x1p, x2p), max(y1p, y2p)


def peak_records(row: dict[str, Any], valid_radius: float) -> list[dict[str, Any]]:
    gt_x1s = [as_float(x) for x in row.get("same_desc_gt_x1_values", [])]
    records = []
    for idx, peak in enumerate(row.get("merged_peaks", []) or []):
        x1 = as_float(peak.get("x1"))
        mass = as_float(peak.get("mass"))
        nearest = min((abs(x1 - gt_x1) for gt_x1 in gt_x1s), default=None)
        valid = nearest is not None and nearest <= valid_radius
        records.append(
            {
                "peak_index": idx,
                "x1": x1,
                "mass": mass,
                "nearest_gt_x1_distance": nearest,
                "annotated_valid_x1_radius": bool(valid),
            }
        )
    return records


def load_x1_rows(root: Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in read_jsonl(root / "x1_candidate_field_rows.jsonl")}


def resolve_image_path(image_path: str, image_roots: list[Path]) -> Path | None:
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in image_roots:
        full = root / image_path
        if full.exists():
            return full
    return None


def load_needed_dataset_rows(review_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    needed: dict[str, set[int]] = defaultdict(set)
    for row in review_rows:
        needed[row["source_dataset_jsonl"]].add(as_int(row["source_line_idx"]))
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset_path, indices in needed.items():
        wanted = set(indices)
        with Path(dataset_path).open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if idx in wanted:
                    loaded[(dataset_path, idx)] = json.loads(line)
                    if len([key for key in loaded if key[0] == dataset_path]) == len(wanted):
                        break
    return loaded


def same_desc_gt_objects(dataset_row: dict[str, Any], desc: str) -> list[dict[str, Any]]:
    objects = []
    for obj_idx, obj in enumerate(dataset_row.get("objects", []) or []):
        if canonical_desc(obj.get("desc", obj.get("category_name", ""))) != desc:
            continue
        bbox = parse_bbox_tokens(obj.get("bbox_2d"))
        if bbox is None:
            continue
        objects.append(
            {
                "object_index": obj_idx,
                "desc": canonical_desc(obj.get("desc", "")),
                "category_name": obj.get("category_name"),
                "coco_ann_id": obj.get("coco_ann_id"),
                "bbox_norm1000_xyxy": bbox,
                "x1_norm1000": bbox[0],
            }
        )
    return objects


def nearest_peak_delta(peak: dict[str, Any], other_peaks: list[dict[str, Any]]) -> float | None:
    if not other_peaks:
        return None
    x1 = as_float(peak.get("x1"))
    return min(abs(x1 - as_float(other.get("x1"))) for other in other_peaks)


def build_review_rows(
    et_rows: dict[str, dict[str, Any]],
    pure_rows: dict[str, dict[str, Any]],
    *,
    valid_radius: float,
    new_peak_radius: float,
) -> list[dict[str, Any]]:
    rows = []
    for case_id, pure in pure_rows.items():
        et = et_rows.get(case_id)
        if et is None:
            continue
        pure_peaks = peak_records(pure, valid_radius)
        et_peaks = peak_records(et, valid_radius)
        pure_unmatched = [peak for peak in pure_peaks if not peak["annotated_valid_x1_radius"]]
        et_unmatched = [peak for peak in et_peaks if not peak["annotated_valid_x1_radius"]]
        if not pure_unmatched and not et_unmatched:
            continue
        pure_new_unmatched = []
        for peak in pure_unmatched:
            nearest_et = nearest_peak_delta(peak, et_unmatched)
            item = dict(peak)
            item["nearest_et_unmatched_peak_distance"] = nearest_et
            item["is_pure_new_unmatched_peak"] = nearest_et is None or nearest_et > new_peak_radius
            pure_new_unmatched.append(item)
        same_desc_count = as_int(pure.get("same_desc_gt_count_annotated"))
        et_coverage = as_float(et.get("gt_instance_coverage_count"), math.nan)
        pure_coverage = as_float(pure.get("gt_instance_coverage_count"), math.nan)
        if math.isnan(et_coverage) or math.isnan(pure_coverage):
            et_cov_frac = coverage_from_peaks(et_peaks, pure.get("same_desc_gt_x1_values", []), same_desc_count, valid_radius)
            pure_cov_frac = coverage_from_peaks(pure_peaks, pure.get("same_desc_gt_x1_values", []), same_desc_count, valid_radius)
        else:
            et_cov_frac = et_coverage / same_desc_count if same_desc_count else 0.0
            pure_cov_frac = pure_coverage / same_desc_count if same_desc_count else 0.0
        priority = (
            1000 * sum(1 for peak in pure_new_unmatched if peak["is_pure_new_unmatched_peak"])
            + 100 * max(0.0, pure_cov_frac - et_cov_frac)
            + 10 * same_desc_count
            + len(pure_unmatched)
        )
        rows.append(
            {
                "review_status": "unreviewed",
                "manual_label": "",
                "manual_notes": "",
                "case_id": case_id,
                "split": pure.get("split"),
                "image_id": pure.get("image_id"),
                "image_path": pure.get("image_path"),
                "source_dataset_jsonl": pure.get("source_dataset_jsonl"),
                "source_line_idx": pure.get("source_line_idx"),
                "desc_text_canonical": pure.get("desc_text_canonical"),
                "same_desc_gt_count_annotated": same_desc_count,
                "x1_projection_collision": bool(pure.get("x1_projection_collision")),
                "et_peak_count": len(et_peaks),
                "pure_peak_count": len(pure_peaks),
                "et_unmatched_peak_count": len(et_unmatched),
                "pure_unmatched_peak_count": len(pure_unmatched),
                "pure_new_unmatched_peak_count": sum(
                    1 for peak in pure_new_unmatched if peak["is_pure_new_unmatched_peak"]
                ),
                "et_coverage_fraction": et_cov_frac,
                "pure_coverage_fraction": pure_cov_frac,
                "delta_coverage_fraction": pure_cov_frac - et_cov_frac,
                "pure_unmatched_peaks": pure_new_unmatched,
                "et_unmatched_peaks": et_unmatched,
                "pure_all_peaks": pure_peaks,
                "et_all_peaks": et_peaks,
                "review_priority": priority,
            }
        )
    return sorted(rows, key=lambda row: row["review_priority"], reverse=True)


def coverage_from_peaks(
    peaks: list[dict[str, Any]],
    gt_x1_values: list[Any],
    same_desc_count: int,
    valid_radius: float,
) -> float:
    if not same_desc_count:
        return 0.0
    covered = 0
    for gt_x1 in [as_float(x) for x in gt_x1_values]:
        if any(abs(as_float(peak.get("x1")) - gt_x1) <= valid_radius for peak in peaks):
            covered += 1
    return covered / same_desc_count


def summarize_review_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    desc_counts = Counter(row["desc_text_canonical"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    buckets = Counter(bucket_for_count(as_int(row["same_desc_gt_count_annotated"])) for row in rows)
    pure_new_rows = [row for row in rows if row["pure_new_unmatched_peak_count"] > 0]
    pure_unmatched_rows = [row for row in rows if row["pure_unmatched_peak_count"] > 0]
    return {
        "row_count": len(rows),
        "rows_with_pure_unmatched": len(pure_unmatched_rows),
        "rows_with_pure_new_unmatched": len(pure_new_rows),
        "total_pure_unmatched_peaks": sum(row["pure_unmatched_peak_count"] for row in rows),
        "total_pure_new_unmatched_peaks": sum(row["pure_new_unmatched_peak_count"] for row in rows),
        "total_et_unmatched_peaks": sum(row["et_unmatched_peak_count"] for row in rows),
        "split_counts": dict(split_counts),
        "same_desc_bucket_counts": dict(buckets),
        "top_desc_by_unmatched_rows": desc_counts.most_common(30),
    }


def bucket_for_count(count: int) -> str:
    if count <= 1:
        return "count1"
    if count == 2:
        return "count2"
    if count == 3:
        return "count3"
    if count <= 5:
        return "count4_5"
    return "count6_plus"


def write_manual_template(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "split",
        "image_id",
        "desc_text_canonical",
        "same_desc_gt_count_annotated",
        "pure_unmatched_peak_count",
        "pure_new_unmatched_peak_count",
        "delta_coverage_fraction",
        "gallery_image",
        "manual_label",
        "manual_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def draw_review_image(
    row: dict[str, Any],
    dataset_row: dict[str, Any],
    *,
    image_roots: list[Path],
    output_path: Path,
    max_image_width: int,
    valid_radius: float,
) -> bool:
    resolved = resolve_image_path(str(row["image_path"]), image_roots)
    if resolved is None:
        return False
    image = Image.open(resolved).convert("RGB")
    width, height = image.size
    scale = 1.0
    if max_image_width > 0 and width > max_image_width:
        scale = max_image_width / width
        image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    draw_width, draw_height = image.size
    sx = draw_width / width
    sy = draw_height / height
    font = ImageFont.load_default()

    same_desc_objects = same_desc_gt_objects(dataset_row, str(row["desc_text_canonical"]))
    pure_panel = render_model_image_panel(
        image,
        width=width,
        height=height,
        draw_width=draw_width,
        draw_height=draw_height,
        sx=sx,
        same_desc_objects=same_desc_objects,
        peaks=row["pure_all_peaks"],
        valid_radius=valid_radius,
        valid_color=(230, 160, 0),
        unmatched_color=(220, 30, 30),
        unmatched_width=5,
        valid_width=3,
        band_alpha=45,
    )
    et_panel = render_model_image_panel(
        image,
        width=width,
        height=height,
        draw_width=draw_width,
        draw_height=draw_height,
        sx=sx,
        same_desc_objects=same_desc_objects,
        peaks=row["et_all_peaks"],
        valid_radius=valid_radius,
        valid_color=(0, 150, 180),
        unmatched_color=(40, 80, 220),
        unmatched_width=4,
        valid_width=2,
        band_alpha=35,
    )

    side_w = 540
    title_h = 30
    bottom_h = 165
    pure_y = title_h
    et_title_y = pure_y + draw_height
    et_y = et_title_y + title_h
    axis_y = et_y + draw_height
    total_h = axis_y + bottom_h
    canvas = Image.new("RGBA", (draw_width + side_w, total_h), (255, 255, 255, 255))
    canvas.alpha_composite(pure_panel, (0, pure_y))
    canvas.alpha_composite(et_panel, (0, et_y))
    canvas_draw = ImageDraw.Draw(canvas)
    draw_model_title_bar(canvas_draw, 0, 0, draw_width, title_h, "Pure CE x1 candidate field", (220, 30, 30), font)
    draw_model_title_bar(canvas_draw, 0, et_title_y, draw_width, title_h, "ET-RMP-CE x1 candidate field", (40, 80, 220), font)
    draw_axis_panel(
        canvas_draw,
        row,
        image_width=width,
        draw_width=draw_width,
        image_top=0,
        image_height=draw_height,
        axis_top=axis_y,
        axis_height=bottom_h,
        sx=sx,
        font=font,
    )
    draw_side_panel(
        canvas_draw,
        row,
        same_desc_objects,
        panel_left=draw_width,
        panel_top=0,
        panel_width=side_w,
        panel_height=total_h,
        font=font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, quality=92)
    return True


def render_model_image_panel(
    image: Image.Image,
    *,
    width: int,
    height: int,
    draw_width: int,
    draw_height: int,
    sx: float,
    same_desc_objects: list[dict[str, Any]],
    peaks: list[dict[str, Any]],
    valid_radius: float,
    valid_color: tuple[int, int, int],
    unmatched_color: tuple[int, int, int],
    unmatched_width: int,
    valid_width: int,
    band_alpha: int,
) -> Image.Image:
    image_canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", image_canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    sy = draw_height / height
    for obj in same_desc_objects:
        x1, y1, x2, y2 = bbox_to_px(obj["bbox_norm1000_xyxy"], width, height)
        box = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
        draw.rectangle(box, outline=(0, 180, 0, 255), width=3)
        draw.rectangle(box, fill=(0, 180, 0, 35))
        draw_gt_edge_marker(draw, box[0], draw_height, obj["object_index"])

    draw_peak_set(
        draw,
        peaks,
        width,
        draw_height,
        sx,
        valid_radius,
        valid_color=valid_color,
        unmatched_color=unmatched_color,
        unmatched_width=unmatched_width,
        valid_width=valid_width,
        band_alpha=band_alpha,
    )
    return Image.alpha_composite(image_canvas, overlay)


def draw_model_title_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    draw.rectangle((x, y, x + width, y + height), fill=(245, 245, 245), outline=(190, 190, 190))
    draw.rectangle((x + 8, y + 8, x + 22, y + height - 8), fill=color)
    draw.text((x + 30, y + 8), title, fill=(0, 0, 0), font=font)


def draw_gt_edge_marker(draw: ImageDraw.ImageDraw, x: int, image_height: int, object_index: int) -> None:
    color = (0, 140, 0, 230)
    draw.line((x, max(0, image_height - 18), x, image_height), fill=color, width=3)
    draw.polygon([(x, image_height - 18), (x - 5, image_height - 8), (x + 5, image_height - 8)], fill=color)


def draw_peak_set(
    draw: ImageDraw.ImageDraw,
    peaks: list[dict[str, Any]],
    image_width: int,
    draw_height: int,
    sx: float,
    valid_radius: float,
    *,
    valid_color: tuple[int, int, int],
    unmatched_color: tuple[int, int, int],
    unmatched_width: int,
    valid_width: int,
    band_alpha: int,
) -> None:
    for peak in peaks:
        x1 = as_float(peak.get("x1"))
        x = int(x_to_px(x1, image_width) * sx)
        valid = bool(peak.get("annotated_valid_x1_radius"))
        color = valid_color if valid else unmatched_color
        rgba = (*color, 230)
        if not valid:
            band = max(5, int(x_to_px(valid_radius, image_width) * sx))
            draw.rectangle((max(0, x - band), 0, min(int(image_width * sx), x + band), draw_height), fill=(*color, band_alpha))
            draw.line((x, 0, x, draw_height), fill=rgba, width=unmatched_width)
            draw.ellipse((x - 7, draw_height - 18, x + 7, draw_height - 4), fill=rgba, outline=(255, 255, 255, 240), width=2)
        else:
            draw.line((x, 0, x, draw_height), fill=rgba, width=valid_width)


def draw_axis_panel(
    draw: ImageDraw.ImageDraw,
    row: dict[str, Any],
    *,
    image_width: int,
    draw_width: int,
    image_top: int,
    image_height: int,
    axis_top: int,
    axis_height: int,
    sx: float,
    font: ImageFont.ImageFont,
) -> None:
    del image_top, image_height
    draw.rectangle((0, axis_top, draw_width, axis_top + axis_height), fill=(255, 255, 255), outline=(200, 200, 200))
    axis_y = axis_top + 26
    draw.line((0, axis_y, draw_width, axis_y), fill=(80, 80, 80), width=1)
    for tick in [0, 250, 500, 750, 1000]:
        x = int(x_to_px(tick, image_width) * sx)
        draw.line((x, axis_y - 6, x, axis_y + 6), fill=(80, 80, 80), width=1)
        tick_x = min(max(2, x - 12), max(2, draw_width - 30))
        draw.text((tick_x, axis_y + 8), str(tick), fill=(60, 60, 60), font=font)
    draw.text(
        (6, axis_top + 4),
        "x1 coordinate ruler (0..1000). Colored triangles point to candidate left-boundary x1; exact x is in right table.",
        fill=(0, 0, 0),
        font=font,
    )

    marker_specs = [
        (row["pure_all_peaks"], False, (220, 30, 30), axis_top + 62, "P"),
        (row["pure_all_peaks"], True, (230, 160, 0), axis_top + 88, "P"),
        (row["et_all_peaks"], False, (40, 80, 220), axis_top + 114, "E"),
        (row["et_all_peaks"], True, (0, 150, 180), axis_top + 140, "E"),
    ]
    for peaks, want_valid, color, y, prefix in marker_specs:
        for peak in peaks:
            if bool(peak.get("annotated_valid_x1_radius")) != want_valid:
                continue
            x = int(x_to_px(as_float(peak.get("x1")), image_width) * sx)
            draw.polygon([(x, y - 10), (x - 6, y), (x + 6, y)], fill=color)
            draw.line((x, y, x, y + 8), fill=color, width=2)
            label_x = min(max(2, x + 4), max(2, draw_width - 34))
            draw.text((label_x, y - 8), f"{prefix}{peak['peak_index']}", fill=color, font=font)


def draw_side_panel(
    draw: ImageDraw.ImageDraw,
    row: dict[str, Any],
    same_desc_objects: list[dict[str, Any]],
    *,
    panel_left: int,
    panel_top: int,
    panel_width: int,
    panel_height: int,
    font: ImageFont.ImageFont,
) -> None:
    x0 = panel_left
    draw.rectangle((x0, panel_top, x0 + panel_width, panel_top + panel_height), fill=(248, 248, 248), outline=(180, 180, 180))
    y = panel_top + 10
    lines = [
        "Dual-checkpoint x1 review",
        str(row["case_id"]),
        f"desc={row['desc_text_canonical']}  GT={row['same_desc_gt_count_annotated']}",
        f"coverage ET={row['et_coverage_fraction']:.3f}  Pure={row['pure_coverage_fraction']:.3f}",
        f"Pure peaks={len(row['pure_all_peaks'])} unmatched={row['pure_unmatched_peak_count']} new={row['pure_new_unmatched_peak_count']}",
        f"ET peaks={len(row['et_all_peaks'])} unmatched={row['et_unmatched_peak_count']}",
        "Note: peaks are x1 vertical stripes, not full boxes.",
    ]
    for idx, line in enumerate(lines):
        fill = (0, 0, 0) if idx != 0 else (30, 30, 30)
        draw.text((x0 + 12, y), line[:72], fill=fill, font=font)
        y += 16
    y += 8
    legend = [
        ("same-desc GT box / x1 edge", (0, 160, 0)),
        ("Pure unmatched x1", (220, 30, 30)),
        ("Pure annotated-valid x1", (230, 160, 0)),
        ("ET unmatched x1", (40, 80, 220)),
        ("ET annotated-valid x1", (0, 150, 180)),
    ]
    for text, color in legend:
        draw.rectangle((x0 + 12, y + 3, x0 + 24, y + 13), fill=color)
        draw.text((x0 + 30, y), text, fill=(0, 0, 0), font=font)
        y += 16
    y += 8
    y = draw_peak_table(draw, x0 + 12, y, "Pure CE unmatched peaks", row["pure_all_peaks"], False, (220, 30, 30), "P", font, max_rows=14)
    y += 6
    y = draw_peak_table(draw, x0 + 12, y, "Pure CE valid peaks", row["pure_all_peaks"], True, (160, 110, 0), "P", font, max_rows=6)
    y += 6
    y = draw_peak_table(draw, x0 + 12, y, "ET-RMP-CE unmatched peaks", row["et_all_peaks"], False, (40, 80, 220), "E", font, max_rows=10)
    y += 6
    y = draw_peak_table(draw, x0 + 12, y, "ET-RMP-CE valid peaks", row["et_all_peaks"], True, (0, 120, 150), "E", font, max_rows=6)
    y += 6
    draw.text((x0 + 12, y), "Annotated same-desc GT x1:", fill=(0, 110, 0), font=font)
    y += 16
    for obj in same_desc_objects[:16]:
        draw.text((x0 + 18, y), f"GT{obj['object_index']} x1={obj['x1_norm1000']}", fill=(0, 110, 0), font=font)
        y += 14


def draw_peak_table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    peaks: list[dict[str, Any]],
    want_valid: bool,
    color: tuple[int, int, int],
    prefix: str,
    font: ImageFont.ImageFont,
    *,
    max_rows: int,
) -> int:
    rows = [peak for peak in peaks if bool(peak.get("annotated_valid_x1_radius")) == want_valid]
    draw.text((x, y), f"{title} ({len(rows)})", fill=color, font=font)
    y += 16
    for peak in rows[:max_rows]:
        dgt = peak.get("nearest_gt_x1_distance")
        dgt_text = "NA" if dgt is None else f"{as_float(dgt):.0f}"
        draw.text(
            (x + 6, y),
            f"{prefix}{peak['peak_index']}: x1={as_float(peak.get('x1')):.0f}  mass={as_float(peak.get('mass')):.4f}  dGT={dgt_text}",
            fill=color,
            font=font,
        )
        y += 14
    if len(rows) > max_rows:
        draw.text((x + 6, y), f"... {len(rows) - max_rows} more in JSONL", fill=color, font=font)
        y += 14
    return y


def render_gallery(
    rows: list[dict[str, Any]],
    *,
    output_root: Path,
    image_roots: list[Path],
    max_gallery: int,
    max_image_width: int,
    valid_radius: float,
) -> list[dict[str, Any]]:
    selected = rows if max_gallery == 0 else rows[:max_gallery]
    dataset_rows = load_needed_dataset_rows(selected)
    gallery_root = output_root / "gallery"
    image_root = gallery_root / "images"
    rendered = []
    for rank, row in enumerate(selected):
        dataset_row = dataset_rows.get((row["source_dataset_jsonl"], as_int(row["source_line_idx"])))
        if dataset_row is None:
            continue
        out_name = f"{rank:04d}_{safe_slug(row['case_id'])}.jpg"
        output_path = image_root / out_name
        ok = draw_review_image(
            row,
            dataset_row,
            image_roots=image_roots,
            output_path=output_path,
            max_image_width=max_image_width,
            valid_radius=valid_radius,
        )
        if ok:
            rendered_row = {
                "rank": rank,
                "case_id": row["case_id"],
                "desc_text_canonical": row["desc_text_canonical"],
                "same_desc_gt_count_annotated": row["same_desc_gt_count_annotated"],
                "pure_unmatched_peak_count": row["pure_unmatched_peak_count"],
                "pure_new_unmatched_peak_count": row["pure_new_unmatched_peak_count"],
                "delta_coverage_fraction": row["delta_coverage_fraction"],
                "gallery_image": str(output_path),
                "review_status": "unreviewed",
                "manual_label": "",
                "manual_notes": "",
            }
            row["gallery_image"] = str(output_path)
            rendered.append(rendered_row)
    write_jsonl(gallery_root / "gallery_rows.jsonl", rendered)
    (gallery_root / "index.md").write_text(build_gallery_index(rendered), encoding="utf-8")
    return rendered


def safe_slug(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))[:140]


def build_gallery_index(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Unmatched Peak Manual Review Gallery",
        "",
        "Labels to fill manually: `unlabeled_object`, `hallucination`, `duplication`, `ambiguous`, `other`.",
        "",
        "Layout: the upper image panel shows pure-CE peaks, the lower image panel shows ET-RMP-CE peaks, both on the same image "
        "and GT boxes.  The bottom panel is the shared 0..1000 x1 ruler, and the right panel lists exact peak coordinates.  "
        "Red means pure-CE unmatched x1, orange means pure-CE annotated-valid x1, blue means ET-RMP-CE unmatched x1, "
        "cyan means ET-RMP-CE annotated-valid x1, and green means annotated same-desc GT.",
        "",
        "Boundary: this probe only has x1, so every peak is a vertical left-boundary stripe rather than a full predicted box.",
        "",
    ]
    for row in rows:
        image_path = row["gallery_image"]
        lines.extend(
            [
                f"## {row['rank']:04d} {row['case_id']}",
                "",
                f"- desc: `{row['desc_text_canonical']}`",
                f"- same-desc GT count: `{row['same_desc_gt_count_annotated']}`",
                f"- pure unmatched peaks: `{row['pure_unmatched_peak_count']}`",
                f"- pure new unmatched peaks: `{row['pure_new_unmatched_peak_count']}`",
                f"- delta coverage: `{row['delta_coverage_fraction']:.4f}`",
                f"- manual label: `{row['manual_label']}`",
                f"- notes: `{row['manual_notes']}`",
                "",
                f"![{row['case_id']}]({image_path})",
                "",
            ]
        )
    return "\n".join(lines)


def build_report(summary: dict[str, Any], rendered_count: int, output_root: Path) -> str:
    rows = [
        ["review rows", summary["row_count"]],
        ["rows with pure unmatched", summary["rows_with_pure_unmatched"]],
        ["rows with pure new unmatched", summary["rows_with_pure_new_unmatched"]],
        ["total pure unmatched peaks", summary["total_pure_unmatched_peaks"]],
        ["total pure new unmatched peaks", summary["total_pure_new_unmatched_peaks"]],
        ["total ET unmatched peaks", summary["total_et_unmatched_peaks"]],
        ["gallery rendered images", rendered_count],
    ]
    table = "\n".join(["| Metric | Value |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in rows])
    desc_table = "\n".join(
        ["| Desc | Rows |", "| --- | ---: |"]
        + [f"| {desc} | {count} |" for desc, count in summary["top_desc_by_unmatched_rows"][:15]]
    )
    return "\n".join(
        [
            "# Unmatched Peak Review",
            "",
            "This sidecar records and visualizes x1 peaks that are not within the annotated same-desc GT x1 radius.  "
            "The label `unmatched` is an annotation-relative status, not a hallucination claim.",
            "",
            f"Artifact root: `{output_root}`",
            "",
            table,
            "",
            "## Top Descs",
            "",
            desc_table,
            "",
            "## Files",
            "",
            "- `unmatched_peak_rows.jsonl`: full row-level catalog with manual-review fields.",
            "- `manual_review_template.csv`: spreadsheet-friendly review sheet.",
            "- `gallery/gallery_rows.jsonl`: rendered subset manifest.",
            "- `gallery/index.md`: local image gallery.",
            "",
            "The rendered image layout keeps metadata and legend outside the image: the upper panel shows pure-CE peaks, "
            "the lower panel shows ET-RMP-CE peaks, the bottom panel is a shared 0..1000 x1 ruler with colored markers, "
            "and the right panel lists exact peak coordinates.",
            "",
            "Suggested manual labels: `unlabeled_object`, `hallucination`, `duplication`, `ambiguous`, `other`.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unmatched x1-peak catalog and manual review gallery.")
    parser.add_argument("--et-root", type=Path, default=DEFAULT_ET_ROOT)
    parser.add_argument("--pure-root", type=Path, default=DEFAULT_PURE_ROOT)
    parser.add_argument("--phase-a2-root", type=Path, default=DEFAULT_PHASE_A2_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--valid-radius", type=float, default=24.0)
    parser.add_argument("--new-peak-radius", type=float, default=24.0)
    parser.add_argument("--max-gallery", type=int, default=128, help="0 means render every review row.")
    parser.add_argument("--max-image-width", type=int, default=1200)
    parser.add_argument("--image-root", action="append", default=[])
    args = parser.parse_args()

    output_root = args.output_root or (args.phase_a2_root / "unmatched_review")
    image_roots = [Path(item) for item in args.image_root] + list(DEFAULT_IMAGE_ROOTS)
    et_rows = load_x1_rows(args.et_root)
    pure_rows = load_x1_rows(args.pure_root)
    review_rows = build_review_rows(
        et_rows,
        pure_rows,
        valid_radius=args.valid_radius,
        new_peak_radius=args.new_peak_radius,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize_review_rows(review_rows)
    rendered_rows = render_gallery(
        review_rows,
        output_root=output_root,
        image_roots=image_roots,
        max_gallery=args.max_gallery,
        max_image_width=args.max_image_width,
        valid_radius=args.valid_radius,
    )
    for rendered in rendered_rows:
        for row in review_rows:
            if row["case_id"] == rendered["case_id"]:
                row["gallery_image"] = rendered["gallery_image"]
                break
    write_jsonl(output_root / "unmatched_peak_rows.jsonl", review_rows)
    write_manual_template(output_root / "manual_review_template.csv", review_rows)
    summary["rendered_gallery_count"] = len(rendered_rows)
    summary["artifact_root"] = str(output_root)
    write_json(output_root / "unmatched_peak_summary.json", summary)
    (output_root / "unmatched_peak_review.md").write_text(
        build_report(summary, len(rendered_rows), output_root),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Visualize LVIS segmentation (multi-part) vs Largest Connected Component (LCC) vs convex hull.

Motivation
----------
Some LVIS instances (e.g. place_mat / tray / table) are highly non-convex and/or
multi-part due to occlusion. A convex hull enclosure can over-cover badly.

This script renders 3 panels for selected annotations:
  1) Raw LVIS segmentation polygons (all parts)
  2) LCC: union mask -> largest connected component -> external contour
  3) Convex hull of all segmentation vertices (+ bbox for reference)

We compute masks on the *smart-resized* image size so overlays match
`public_data/lvis/rescale_32_768_poly_20/images/`.

Example:
  PYTHONPATH=. conda run -n ms python public_data/scripts/visualize_lvis_seg_vs_lcc_vs_hull.py \
    --lvis-json public_data/lvis/raw/annotations/lvis_v1_val.json \
    --images-dir public_data/lvis/rescale_32_768_poly_20/images \
    --out-dir output/vis_lvis/raw_vs_lcc_vs_hull_val_table \
    --cats place_mat,tray,table \
    --num 12 --min-parts 2 --min-area 1000
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

from src.datasets.preprocessors.resize import smart_resize


Point = Tuple[float, float]


def _extract_coco_file_name(image: Dict[str, Any]) -> Optional[str]:
    file_name = image.get("coco_file_name") or image.get("file_name")
    if isinstance(file_name, str) and file_name:
        return file_name
    coco_url = image.get("coco_url")
    if isinstance(coco_url, str) and coco_url:
        parts = coco_url.split("/")
        if len(parts) >= 2:
            return "/".join(parts[-2:])
    return None


def _pair_flat(coords: Sequence[float]) -> List[Point]:
    pts: List[Point] = []
    for i in range(0, len(coords), 2):
        pts.append((float(coords[i]), float(coords[i + 1])))
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def _poly_area(poly: Sequence[Point]) -> float:
    if len(poly) < 3:
        return 0.0
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def _convex_hull(points: Sequence[Point]) -> List[Point]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return list(pts)

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _scale_flat(seg: Sequence[float], sx: float, sy: float) -> List[float]:
    out: List[float] = []
    for i in range(0, len(seg), 2):
        out.append(float(seg[i]) * sx)
        out.append(float(seg[i + 1]) * sy)
    return out


def _clamp_int_points(points: Sequence[Point], width: int, height: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for x, y in points:
        xi = max(0, min(width - 1, int(round(x))))
        yi = max(0, min(height - 1, int(round(y))))
        out.append((xi, yi))
    return out


def _draw_outline(draw: ImageDraw.ImageDraw, pts: List[Tuple[int, int]], *, color, width: int) -> None:
    if len(pts) < 2:
        return
    draw.line(pts + [pts[0]], fill=color, width=width)


def _iter_valid_parts(segmentation: Any) -> Iterable[List[float]]:
    if not isinstance(segmentation, list) or not segmentation:
        return
    for part in segmentation:
        if not isinstance(part, list) or len(part) < 6 or len(part) % 2 != 0:
            continue
        try:
            coords = [float(v) for v in part]
        except Exception:
            continue
        if len(coords) >= 6:
            yield coords


def _rle_from_scaled_polys(polys_scaled: List[List[float]], h: int, w: int):
    rles = mask_utils.frPyObjects(polys_scaled, h, w)
    return mask_utils.merge(rles)


def _largest_cc(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    import cv2

    mask_u8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8, int(mask_u8.sum())
    # stats: [x, y, w, h, area]
    areas = stats[1:, 4]
    best_idx = int(np.argmax(areas))
    best = 1 + best_idx
    out = (labels == best).astype(np.uint8)
    return out, int(areas[best_idx])


def _external_contour(mask: np.ndarray) -> List[Tuple[int, int]]:
    import cv2

    m = (mask.astype(np.uint8) * 255) if mask.max() <= 1 else mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2)
    return [(int(x), int(y)) for x, y in pts]


@dataclass(frozen=True)
class Candidate:
    ann_id: int
    image_id: int
    cat: str
    image_path: Path
    orig_w: int
    orig_h: int
    tgt_w: int
    tgt_h: int
    n_parts: int
    total_vertices: int
    hull_vertices: int
    area_gt: float
    area_hull: float
    iou_hull: float


def _safe_cat_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip().lower())
    return cleaned[:64] if cleaned else "object"


def _load_json_bytes(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        import orjson

        return orjson.loads(raw)
    except Exception:
        return json.loads(raw.decode("utf-8"))


def _collect_candidates(
    *,
    lvis_json: Path,
    images_dir: Path,
    cat_patterns: List[str],
    min_parts: int,
    min_area: float,
) -> List[Candidate]:
    data = _load_json_bytes(lvis_json)
    categories = data.get("categories") or []
    images = data.get("images") or []
    annotations = data.get("annotations") or []

    cat_id_to_name: Dict[int, str] = {}
    for c in categories:
        try:
            cat_id_to_name[int(c["id"])] = str(c["name"])
        except Exception:
            continue

    matchers: List[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in cat_patterns if p.strip()]

    image_id_to_info: Dict[int, Dict[str, Any]] = {}
    for img in images:
        try:
            img_id = int(img["id"])
        except Exception:
            continue
        file_name = _extract_coco_file_name(img)
        if not file_name:
            continue
        image_id_to_info[img_id] = {
            "file_name": file_name,
            "width": int(img["width"]),
            "height": int(img["height"]),
        }

    out: List[Candidate] = []
    for ann in annotations:
        try:
            img_id = int(ann["image_id"])
            ann_id = int(ann["id"])
            cat_name = cat_id_to_name[int(ann["category_id"])]
        except Exception:
            continue

        if matchers and not any(m.search(cat_name) for m in matchers):
            continue

        info = image_id_to_info.get(img_id)
        if not info:
            continue

        area_gt = float(ann.get("area") or 0.0)
        if area_gt < float(min_area):
            continue

        parts = list(_iter_valid_parts(ann.get("segmentation")))
        if len(parts) < int(min_parts):
            continue

        pts_all: List[Point] = []
        total_vertices = 0
        for flat in parts:
            pts = _pair_flat(flat)
            pts_all.extend(pts)
            total_vertices += len(pts)
        if len(pts_all) < 3:
            continue

        # Hull in original space; area ratios are scale-invariant.
        hull = _convex_hull(pts_all)
        if len(hull) < 3:
            continue
        area_hull = _poly_area(hull)
        if area_hull <= 0.0:
            continue
        iou_hull = area_gt / area_hull
        if iou_hull > 1.0:
            iou_hull = 1.0

        orig_w = int(info["width"])
        orig_h = int(info["height"])
        tgt_h, tgt_w = smart_resize(height=orig_h, width=orig_w)

        basename = os.path.basename(str(info["file_name"]))
        img_path = (images_dir / basename).resolve()
        if not img_path.exists():
            continue

        out.append(
            Candidate(
                ann_id=ann_id,
                image_id=img_id,
                cat=cat_name,
                image_path=img_path,
                orig_w=orig_w,
                orig_h=orig_h,
                tgt_w=int(tgt_w),
                tgt_h=int(tgt_h),
                n_parts=len(parts),
                total_vertices=int(total_vertices),
                hull_vertices=int(len(hull)),
                area_gt=float(area_gt),
                area_hull=float(area_hull),
                iou_hull=float(iou_hull),
            )
        )
    return out


def _render_candidate(
    cand: Candidate,
    *,
    lvis_json: Path,
    out_dir: Path,
    line_width: int,
    pad: int,
) -> Tuple[Path, Dict[str, Any]]:
    data = _load_json_bytes(lvis_json)
    anns = data.get("annotations") or []
    ann = None
    for a in anns:
        try:
            if int(a.get("id")) == int(cand.ann_id):
                ann = a
                break
        except Exception:
            continue
    if ann is None:
        raise RuntimeError(f"annotation id not found: {cand.ann_id}")

    sx = float(cand.tgt_w) / float(cand.orig_w)
    sy = float(cand.tgt_h) / float(cand.orig_h)

    # Scale segmentation polygons to resized resolution (for mask + drawing).
    parts_scaled_flat: List[List[float]] = []
    pts_all_scaled: List[Point] = []
    total_vertices = 0
    for part in _iter_valid_parts(ann.get("segmentation")):
        scaled = _scale_flat(part, sx, sy)
        parts_scaled_flat.append(scaled)
        pts = _pair_flat(scaled)
        pts_all_scaled.extend(pts)
        total_vertices += len(pts)
    if len(parts_scaled_flat) == 0:
        raise RuntimeError("no valid segmentation parts")

    rle = _rle_from_scaled_polys(parts_scaled_flat, cand.tgt_h, cand.tgt_w)
    union_mask = mask_utils.decode(rle).astype(np.uint8)
    union_area = int(union_mask.sum())

    lcc_mask, lcc_area = _largest_cc(union_mask)
    lcc_contour = _external_contour(lcc_mask)

    # Hull on scaled vertices.
    hull = _convex_hull(pts_all_scaled)
    hull_area = _poly_area(hull)

    # BBox (scaled) from LVIS bbox xywh.
    bx, by, bw, bh = ann.get("bbox", [0, 0, 0, 0])
    bbox_scaled = [float(bx) * sx, float(by) * sy, float(bx + bw) * sx, float(by + bh) * sy]
    bbox_area = max(0.0, float(bw) * float(bh) * sx * sy)

    # Cheap IoU proxies (area ratios) in resized scale.
    area_gt = float(ann.get("area") or 0.0) * sx * sy
    iou_bbox = (area_gt / bbox_area) if bbox_area > 0 else 0.0
    iou_hull = (area_gt / hull_area) if hull_area > 0 else 0.0
    iou_bbox = float(min(1.0, max(0.0, iou_bbox)))
    iou_hull = float(min(1.0, max(0.0, iou_hull)))
    lcc_ratio = float(lcc_area / max(1, union_area))

    img = Image.open(cand.image_path).convert("RGBA")
    # Be robust to mismatch: if actual size differs, re-scale drawn geometry once more.
    if img.size != (cand.tgt_w, cand.tgt_h):
        actual_w, actual_h = img.size
        fx = actual_w / float(cand.tgt_w)
        fy = actual_h / float(cand.tgt_h)
        vis_w, vis_h = actual_w, actual_h
    else:
        fx = fy = 1.0
        vis_w, vis_h = cand.tgt_w, cand.tgt_h

    def rescale_pts_int(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if fx == 1.0 and fy == 1.0:
            return pts
        out_pts: List[Tuple[int, int]] = []
        for x, y in pts:
            out_pts.append((int(round(x * fx)), int(round(y * fy))))
        return out_pts

    # Panels
    seg_panel = img.copy().convert("RGBA")
    lcc_panel = img.copy().convert("RGBA")
    hull_panel = img.copy().convert("RGBA")

    seg_ov = Image.new("RGBA", seg_panel.size, (0, 0, 0, 0))
    lcc_ov = Image.new("RGBA", lcc_panel.size, (0, 0, 0, 0))
    hull_ov = Image.new("RGBA", hull_panel.size, (0, 0, 0, 0))
    ds = ImageDraw.Draw(seg_ov)
    dl = ImageDraw.Draw(lcc_ov)
    dh = ImageDraw.Draw(hull_ov)

    seg_color = (64, 128, 255, 190)
    lcc_color = (64, 255, 128, 220)
    hull_color = (255, 64, 64, 220)
    bbox_color = (255, 200, 32, 220)

    # Raw segmentation (all parts)
    for part in parts_scaled_flat:
        pts = _pair_flat(part)
        pts_i = _clamp_int_points(pts, cand.tgt_w, cand.tgt_h)
        pts_i = rescale_pts_int(pts_i)
        if len(pts_i) >= 3:
            _draw_outline(ds, pts_i, color=seg_color, width=int(line_width))

    # LCC contour
    if lcc_contour:
        _draw_outline(dl, rescale_pts_int(lcc_contour), color=lcc_color, width=int(line_width + 1))

    # Convex hull outline
    if len(hull) >= 3:
        hull_i = _clamp_int_points(hull, cand.tgt_w, cand.tgt_h)
        hull_i = rescale_pts_int(hull_i)
        _draw_outline(dh, hull_i, color=hull_color, width=int(line_width + 1))

    # BBox on hull panel for reference
    x1, y1, x2, y2 = bbox_scaled
    x1i = int(round(x1 * fx))
    y1i = int(round(y1 * fy))
    x2i = int(round(x2 * fx))
    y2i = int(round(y2 * fy))
    dh.rectangle([x1i, y1i, x2i, y2i], outline=bbox_color, width=int(line_width))

    seg_panel = Image.alpha_composite(seg_panel, seg_ov).convert("RGB")
    lcc_panel = Image.alpha_composite(lcc_panel, lcc_ov).convert("RGB")
    hull_panel = Image.alpha_composite(hull_panel, hull_ov).convert("RGB")

    # Compose 3-up
    w, h = seg_panel.size
    canvas = Image.new("RGB", (w * 3 + pad * 2, h + pad * 2), (0, 0, 0))
    canvas.paste(seg_panel, (0, pad))
    canvas.paste(lcc_panel, (w + pad, pad))
    canvas.paste(hull_panel, (w * 2 + pad * 2, pad))

    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), "LVIS segmentation (all parts)", fill=(255, 255, 255))
    draw.text((w + pad + 4, 2), "Largest connected component (LCC)", fill=(255, 255, 255))
    draw.text((w * 2 + pad * 2 + 4, 2), "Convex hull (+ bbox)", fill=(255, 255, 255))

    meta = "\n".join(
        [
            (
                f"cat={cand.cat} ann_id={cand.ann_id} image_id={cand.image_id} "
                f"parts={cand.n_parts} verts={total_vertices} hull_verts={len(hull)}"
            ),
            (
                f"area(union)={union_area} area(lcc)={lcc_area} lcc_ratio={lcc_ratio:.3f} "
                f"iou_bbox~{iou_bbox:.3f} iou_hull~{iou_hull:.3f}"
            ),
            f"orig={cand.orig_w}x{cand.orig_h} resized={cand.tgt_w}x{cand.tgt_h}",
        ]
    )
    draw.text((4, h + pad + 4), meta, fill=(255, 255, 255))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ann{cand.ann_id}_img{cand.image_id}_{_safe_cat_name(cand.cat)}.png"
    canvas.save(out_path)

    info = {
        "out": str(out_path),
        "image": str(cand.image_path),
        "cat": cand.cat,
        "ann_id": cand.ann_id,
        "image_id": cand.image_id,
        "parts": cand.n_parts,
        "total_vertices": total_vertices,
        "hull_vertices": len(hull),
        "union_area": union_area,
        "lcc_area": lcc_area,
        "lcc_ratio": lcc_ratio,
        "iou_bbox": iou_bbox,
        "iou_hull": iou_hull,
    }
    return out_path, info


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize LVIS seg vs LCC vs convex hull")
    ap.add_argument("--lvis-json", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cats", type=str, default="place_mat,tray,table")
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--min-parts", type=int, default=2)
    ap.add_argument("--min-area", type=float, default=1000.0)
    ap.add_argument("--pick", type=str, default="worst", choices=["worst", "mix"])
    ap.add_argument("--line-width", type=int, default=3)
    ap.add_argument("--pad", type=int, default=8)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # Allow either literal names (comma-separated) or regex patterns.
    raw_tokens = [c.strip() for c in str(args.cats).split(",") if c.strip()]
    pats: List[str] = []
    for tok in raw_tokens:
        if any(ch in tok for ch in [".", "?", "*", "[", "]", "(", ")", "|", "\\"]):
            pats.append(tok)
        else:
            pats.append(r"^" + re.escape(tok) + r"$")

    cands = _collect_candidates(
        lvis_json=args.lvis_json,
        images_dir=args.images_dir,
        cat_patterns=pats,
        min_parts=int(args.min_parts),
        min_area=float(args.min_area),
    )
    if not cands:
        raise SystemExit("No candidates found; try lowering --min-parts/--min-area or widening --cats")

    cands_sorted = sorted(cands, key=lambda c: c.iou_hull)
    k = int(args.num)
    if str(args.pick) == "mix":
        worst = cands_sorted[: max(1, k // 2)]
        best = list(reversed(cands_sorted))[: max(1, k - len(worst))]
        picked = worst + best
    else:
        picked = cands_sorted[:k]

    index: List[Dict[str, Any]] = []
    for cand in picked:
        _, info = _render_candidate(
            cand,
            lvis_json=args.lvis_json,
            out_dir=args.out_dir,
            line_width=int(args.line_width),
            pad=int(args.pad),
        )
        index.append(info)

    idx_path = args.out_dir / "index.json"
    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(index)} visualizations to: {args.out_dir}")
    print(f"Index: {idx_path}")


if __name__ == "__main__":
    main()


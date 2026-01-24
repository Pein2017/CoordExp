#!/usr/bin/env python3
"""Find LVIS instances where bbox is likely a better *semantic* region than visible-mask polygons.

Background
----------
LVIS annotations provide:
  - segmentation: visible mask polygons (possibly multi-part)
  - bbox: axis-aligned box computed from the visible mask

For some classes (plates/trays/tables/placemats/etc.), the visible mask can be
very sparse (e.g. only a thin rim is visible), while the bbox still covers the
"semantic object region" (often because the rim surrounds occluders like food).

This script finds such edge cases via a simple proxy:
  fill_ratio = area_gt / bbox_area
where area_gt is the LVIS mask area and bbox_area is w*h.

Low fill_ratio with a large bbox suggests "thin / ring / heavily occluded"
instances where approximating with a single poly is not semantically faithful.

It can also render quick visualizations: image with bbox (yellow) and
segmentation outlines (blue).

Example:
  PYTHONPATH=. conda run -n ms python public_data/scripts/find_lvis_semantic_edge_cases.py \
    --lvis-json public_data/lvis/raw/annotations/lvis_v1_val.json \
    --images-dir public_data/lvis/rescale_32_768_poly_20/images \
    --out-dir output/vis_lvis/edge_cases_low_fill_val \
    --topk 24 --max-per-cat 3 \
    --fill-thresh 0.06 --min-bbox-frac 0.10
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

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


def _load_json_bytes(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        import orjson

        return orjson.loads(raw)
    except Exception:
        return json.loads(raw.decode("utf-8"))


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


def _scale_flat(seg: Sequence[float], sx: float, sy: float) -> List[float]:
    out: List[float] = []
    for i in range(0, len(seg), 2):
        out.append(float(seg[i]) * sx)
        out.append(float(seg[i + 1]) * sy)
    return out


def _pair_flat(coords: Sequence[float]) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for i in range(0, len(coords), 2):
        pts.append((int(round(float(coords[i]))), int(round(float(coords[i + 1])))))
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def _clamp_pts(pts: List[Tuple[int, int]], w: int, h: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for x, y in pts:
        out.append((max(0, min(w - 1, x)), max(0, min(h - 1, y))))
    return out


def _safe_name(s: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip().lower())
    return cleaned[:64] if cleaned else "obj"


@dataclass(frozen=True)
class Hit:
    ann_id: int
    image_id: int
    cat: str
    file_name: str
    orig_w: int
    orig_h: int
    bbox_xywh: List[float]
    area_gt: float
    fill_ratio: float
    n_parts: int
    total_vertices: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Find LVIS edge cases where mask is very sparse inside bbox")
    ap.add_argument("--lvis-json", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--max-per-cat", type=int, default=3)
    ap.add_argument("--fill-thresh", type=float, default=0.06, help="Keep instances with area/bbox_area <= this")
    ap.add_argument("--min-bbox-frac", type=float, default=0.10, help="Keep only if bbox_area/image_area >= this")
    ap.add_argument("--min-area", type=float, default=500.0, help="Min LVIS mask area (raw pixels)")
    ap.add_argument("--min-bbox-area", type=float, default=2000.0, help="Min bbox area (raw pixels)")
    ap.add_argument("--draw-width", type=int, default=3)
    return ap.parse_args()


def _render_hit(hit: Hit, *, images_dir: Path, out_dir: Path, lvis_json: Path, draw_width: int) -> Path:
    # Load full JSON to fetch the annotation segmentation.
    data = _load_json_bytes(lvis_json)
    anns = data.get("annotations") or []
    ann = None
    for a in anns:
        try:
            if int(a.get("id")) == int(hit.ann_id):
                ann = a
                break
        except Exception:
            continue
    if ann is None:
        raise RuntimeError(f"annotation id not found: {hit.ann_id}")

    basename = os.path.basename(hit.file_name)
    img_path = (images_dir / basename).resolve()
    img = Image.open(img_path).convert("RGBA")

    # Scale coords to resized resolution (matching images_dir).
    tgt_h, tgt_w = smart_resize(height=hit.orig_h, width=hit.orig_w)
    sx = float(tgt_w) / float(hit.orig_w)
    sy = float(tgt_h) / float(hit.orig_h)

    # Some images may not match expected size (be robust).
    if img.size != (tgt_w, tgt_h):
        actual_w, actual_h = img.size
        fx = actual_w / float(tgt_w)
        fy = actual_h / float(tgt_h)
        tgt_w = actual_w
        tgt_h = actual_h
    else:
        fx = fy = 1.0

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    seg_color = (64, 128, 255, 210)
    bbox_color = (255, 200, 32, 230)

    # Draw segmentation outlines (all parts).
    for part in _iter_valid_parts(ann.get("segmentation")):
        scaled = _scale_flat(part, sx, sy)
        pts = _pair_flat(scaled)
        pts = _clamp_pts(pts, int(tgt_w), int(tgt_h))
        if fx != 1.0 or fy != 1.0:
            pts = [(int(round(x * fx)), int(round(y * fy))) for x, y in pts]
        if len(pts) >= 3:
            draw.line(pts + [pts[0]], fill=seg_color, width=int(draw_width))

    # Draw bbox.
    bx, by, bw, bh = hit.bbox_xywh
    x1 = int(round(bx * sx * fx))
    y1 = int(round(by * sy * fy))
    x2 = int(round((bx + bw) * sx * fx))
    y2 = int(round((by + bh) * sy * fy))
    draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=int(draw_width))

    vis = Image.alpha_composite(img, overlay).convert("RGB")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ann{hit.ann_id}_img{hit.image_id}_{_safe_name(hit.cat)}.png"
    vis.save(out_path)
    return out_path


def main() -> None:
    args = parse_args()
    data = _load_json_bytes(args.lvis_json)
    categories = data.get("categories") or []
    images = data.get("images") or []
    annotations = data.get("annotations") or []

    cat_id_to_name: Dict[int, str] = {}
    for c in categories:
        try:
            cat_id_to_name[int(c["id"])] = str(c["name"])
        except Exception:
            continue

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

    hits: List[Hit] = []
    for ann in annotations:
        try:
            img_id = int(ann["image_id"])
            ann_id = int(ann["id"])
            cat = cat_id_to_name[int(ann["category_id"])]
        except Exception:
            continue

        info = image_id_to_info.get(img_id)
        if not info:
            continue
        orig_w = int(info["width"])
        orig_h = int(info["height"])
        img_area = float(orig_w) * float(orig_h)
        if img_area <= 0:
            continue

        area_gt = float(ann.get("area") or 0.0)
        if area_gt < float(args.min_area):
            continue

        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        bx, by, bw, bh = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        bbox_area = float(bw) * float(bh)
        if bbox_area < float(args.min_bbox_area):
            continue
        bbox_frac = bbox_area / img_area
        if bbox_frac < float(args.min_bbox_frac):
            continue

        if bbox_area <= 0:
            continue
        fill_ratio = float(area_gt / bbox_area)
        if fill_ratio > float(args.fill_thresh):
            continue

        # Count parts/verts for debugging.
        n_parts = 0
        total_vertices = 0
        for part in _iter_valid_parts(ann.get("segmentation")):
            n_parts += 1
            total_vertices += len(part) // 2
        if n_parts <= 0:
            continue

        hits.append(
            Hit(
                ann_id=ann_id,
                image_id=img_id,
                cat=cat,
                file_name=str(info["file_name"]),
                orig_w=orig_w,
                orig_h=orig_h,
                bbox_xywh=[bx, by, bw, bh],
                area_gt=area_gt,
                fill_ratio=fill_ratio,
                n_parts=n_parts,
                total_vertices=total_vertices,
            )
        )

    # Sort: lowest fill first; tie-break by larger bbox (more semantically relevant).
    hits.sort(key=lambda h: (h.fill_ratio, -(h.bbox_xywh[2] * h.bbox_xywh[3])))

    # Enforce max-per-cat to diversify edge cases.
    per_cat: Dict[str, int] = {}
    picked: List[Hit] = []
    for h in hits:
        if len(picked) >= int(args.topk):
            break
        c = h.cat
        if per_cat.get(c, 0) >= int(args.max_per_cat):
            continue
        per_cat[c] = per_cat.get(c, 0) + 1
        picked.append(h)

    out_index: List[Dict[str, Any]] = []
    for h in picked:
        out_path = None
        if args.out_dir is not None:
            out_path = _render_hit(
                h,
                images_dir=args.images_dir,
                out_dir=Path(args.out_dir),
                lvis_json=args.lvis_json,
                draw_width=int(args.draw_width),
            )
        out_index.append(
            {
                "cat": h.cat,
                "ann_id": h.ann_id,
                "image_id": h.image_id,
                "file_name": h.file_name,
                "fill_ratio": h.fill_ratio,
                "bbox_xywh": h.bbox_xywh,
                "area_gt": h.area_gt,
                "parts": h.n_parts,
                "total_vertices": h.total_vertices,
                "out": str(out_path) if out_path is not None else None,
            }
        )

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        idx_path = out_dir / "index.json"
        idx_path.write_text(json.dumps(out_index, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(out_index)} hits to: {out_dir}")
        print(f"Index: {idx_path}")
    else:
        print(json.dumps(out_index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Build a bbox+poly dataset JSONL from raw LVIS annotations (single-geometry per instance).

This script is designed for *dataset-fixed* ablations:
- Fix a polygon vertex budget (cap) (e.g. 10 or 20).
- For each LVIS instance, compute a convex-hull polygon from the LVIS segmentation.
  - If hull has <= cap vertices, it is a valid poly candidate.
  - Otherwise we fall back to bbox (cannot exceed cap).
- Optionally choose bbox vs poly by a simple cost-effectiveness rule based on mask IoU:
  - IoU(bbox_mask, GT_mask)  = area_gt / area_bbox
  - IoU(hull_mask, GT_mask)  = area_gt / area_hull   (GT is inside convex hull)
  - Prefer bbox when the IoU gain does not justify extra points.

Why convex hull?
- Always produces a single polygon even for multi-part masks (union of parts).
- Is robust to downstream "centroid angle sort" canonicalization because hull is convex.
- Provides a minimum-area convex enclosure (often much tighter than axis-aligned bbox).

Optional semantic guard (bbox override)
--------------------------------------
LVIS segmentations are masks of *visible* regions. For heavily occluded objects (e.g. plate under pizza),
the visible mask polygon can be a tiny fragment that does not represent the "semantic extent" of the object.
To prioritize semantic correctness, we support an optional rule-based override that forces bbox for such
edge cases based on cheap area-derived proxies (fill ratio, hull IoU, multipart-ness).

Outputs pixel-space coords on *smart-resized* images. In practice, for LVIS/COCO2017,
all images are already within the default max-pixels budget used by `smart_resize`,
so this is typically a no-op and you can point `--images-dir` directly at the raw
COCO2017 directory structure (contains `train2017/` and `val2017/`).

`--images-dir` supports two common layouts:
- Flat directory (legacy): `.../images/000000391895.jpg`
- COCO2017 structure: `.../images/train2017/000000391895.jpg` and `.../images/val2017/...`

Geometry policies (single geometry per instance)
------------------------------------------------
This script supports multiple geometry output policies via `--geometry-policy`:
- mix (default): bbox+poly mix using cost-effectiveness + optional semantic guard.
- poly_prefer_semantic: prefer poly whenever eligible under cap; only fallback to bbox
  for semantic-guard reasons or when a cap-respecting poly cannot be formed.
- bbox_only: always output bbox_2d (ignores segmentation).

Example (val, cap=20; COCO2017 layout):
  PYTHONPATH=. conda run -n ms python \\
    public_data/scripts/build_lvis_hull_mix.py \\
    --lvis-json public_data/lvis/raw/annotations/lvis_v1_val.json \\
    --images-dir public_data/lvis/raw/images \\
    --output-jsonl tmp/val.mix_hull_cap20.jsonl \\
    --poly-cap 20 \\
    --lambda-iou-per-extra-point 0.01 \\
    --drop-min-objects 50 --drop-max-unique 3 --drop-min-top1-ratio 0.95

Then convert to coord tokens:
  PYTHONPATH=. conda run -n ms python \\
    public_data/scripts/convert_to_coord_tokens.py \\
    --input public_data/lvis/rescale_32_768_poly_20/val.mix_hull_cap20.jsonl \\
    --output-tokens public_data/lvis/rescale_32_768_poly_20/val.mix_hull_cap20.coord.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import orjson

from src.datasets.geometry import clamp_points, scale_points
from src.datasets.preprocessors.resize import smart_resize
from src.datasets.utils import sort_objects_by_topleft


Point = Tuple[float, float]


def _extract_coco_file_name(image: Dict[str, Any]) -> Optional[str]:
    """Match LVISConverter logic: infer `split/filename.jpg` from coco_url when needed."""
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
    # Strip duplicate closing point if present.
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
    """Monotonic chain convex hull. Returns CCW hull without duplicate last point."""
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

    hull = lower[:-1] + upper[:-1]
    return hull


def _rotate_start_top_left(poly: List[Point]) -> List[Point]:
    """Rotate vertices so the top-most then left-most point is first (stable anchor)."""
    if not poly:
        return poly
    best = 0
    for i, (x, y) in enumerate(poly):
        bx, by = poly[best]
        if (y < by) or (y == by and x < bx):
            best = i
    return poly[best:] + poly[:best]


def _clean_int_poly(flat: List[int]) -> List[int]:
    """Remove a duplicated closing point and consecutive duplicates after rounding."""
    if len(flat) < 6 or len(flat) % 2 != 0:
        return []
    pts = [(int(flat[i]), int(flat[i + 1])) for i in range(0, len(flat), 2)]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    # Drop consecutive duplicates (can happen after scaling+rounding).
    dedup: List[Point] = []
    for p in pts:
        if not dedup or p != dedup[-1]:
            dedup.append(p)
    if len(dedup) >= 2 and dedup[0] == dedup[-1]:
        dedup = dedup[:-1]
    if len(dedup) < 3:
        return []
    out: List[int] = []
    for x, y in dedup:
        out.extend([int(x), int(y)])
    return out


def _poly_area2_flat_int(flat: List[int]) -> int:
    """Return twice the signed area (shoelace) for a flat [x0,y0,x1,y1,...] list.

    Used as a strict degeneracy check (0 => degenerate) after scaling+rounding.
    """
    if len(flat) < 6 or len(flat) % 2 != 0:
        return 0
    pts: List[Point] = [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return 0
    a2 = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a2 += x1 * y2 - x2 * y1
    return int(a2)


def _fix_bbox_xyxy(b: List[int], width: int, height: int) -> List[int]:
    """Mirror resize.py bbox fix-up: enforce x2>x1,y2>y1 after rounding/clamping."""
    if len(b) != 4:
        return b
    x1, y1, x2, y2 = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1:
        if x2 < width - 1:
            x2 = min(width - 1, x1 + 1)
        elif x1 > 0:
            x1 = max(0, x2 - 1)
    if y2 <= y1:
        if y2 < height - 1:
            y2 = min(height - 1, y1 + 1)
        elif y1 > 0:
            y1 = max(0, y2 - 1)
    return [x1, y1, x2, y2]


def _bbox_xyxy_from_coco(bbox_xywh: Sequence[float]) -> List[float]:
    if len(bbox_xywh) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(bbox_xywh)}")
    x, y, w, h = (float(bbox_xywh[0]), float(bbox_xywh[1]), float(bbox_xywh[2]), float(bbox_xywh[3]))
    return [x, y, x + w, y + h]


def _effective_classes(counts: Counter[str]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = float(c) / total
        h -= p * math.log(p + 1e-12)
    return float(math.exp(h)) if h > 0 else float(len(counts))


@dataclass
class BuildStats:
    images_seen: int = 0
    images_written: int = 0
    images_dropped_low_div: int = 0
    images_missing_resized: int = 0

    anns_seen: int = 0
    objects_written: int = 0
    objects_poly: int = 0
    objects_bbox: int = 0
    objects_poly_ineligible: int = 0  # hull > cap or invalid
    objects_bbox_preferred: int = 0  # bbox chosen despite eligible poly

    # Semantic guard overrides (bbox forced even though poly is eligible under cap).
    objects_bbox_forced: int = 0
    objects_bbox_forced_rule1: int = 0  # (hull_iou low OR fill low) under guards
    objects_bbox_forced_rule2: int = 0  # multipart + hull_iou low
    forced_by_category: Dict[str, int] = None  # lazily initialized for JSON-friendly stats

    # Debug counters
    bad_segmentation: int = 0
    bad_hull: int = 0

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        # dataclass default for dict fields needs a runtime init to avoid shared state.
        if d.get("forced_by_category") is None:
            d["forced_by_category"] = {}
        return {
            **d,
            "poly_ratio": float(self.objects_poly / max(1, self.objects_written)),
            "bbox_ratio": float(self.objects_bbox / max(1, self.objects_written)),
        }


def _count_valid_poly_parts(segmentation: Any) -> int:
    """Count polygon parts in LVIS segmentation (ignore invalid/non-polygon encodings)."""
    if not isinstance(segmentation, list) or not segmentation:
        return 0
    n = 0
    for part in segmentation:
        if not isinstance(part, list) or len(part) < 6 or len(part) % 2 != 0:
            continue
        n += 1
    return int(n)


def _poly_candidate_from_segmentation(segmentation: Any) -> Optional[List[Point]]:
    if not isinstance(segmentation, list) or not segmentation:
        return None

    pts_all: List[Point] = []
    for part in segmentation:
        if not isinstance(part, list) or len(part) < 6 or len(part) % 2 != 0:
            continue
        try:
            pts_all.extend(_pair_flat([float(v) for v in part]))
        except Exception:
            continue

    if len(pts_all) < 3:
        return None
    hull = _convex_hull(pts_all)
    if len(hull) < 3:
        return None
    hull = _rotate_start_top_left(hull)
    return hull


def _should_drop_record(
    objects: List[Dict[str, Any]],
    *,
    drop_min_objects: int,
    drop_max_unique: int,
    drop_min_top1_ratio: Optional[float],
    drop_max_effective_classes: Optional[float],
    drop_hard_max_objects: Optional[int],
) -> bool:
    n_objects = int(len(objects))
    if drop_hard_max_objects is not None and n_objects >= int(drop_hard_max_objects):
        return True
    if n_objects < int(drop_min_objects):
        return False

    counts: Counter[str] = Counter()
    for o in objects:
        desc = o.get("desc", "")
        if isinstance(desc, str):
            key = desc.strip().lower().split("/")[0]
        else:
            key = str(desc)
        counts[key] += 1

    n_unique = int(len([k for k in counts.keys() if k]))
    if n_unique > int(drop_max_unique):
        return False

    top1 = int(counts.most_common(1)[0][1]) if counts else 0
    top1_ratio = float(top1 / n_objects) if n_objects else 0.0
    if drop_min_top1_ratio is not None and top1_ratio < float(drop_min_top1_ratio):
        return False

    eff = float(_effective_classes(counts))
    if drop_max_effective_classes is not None and eff > float(drop_max_effective_classes):
        return False

    return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build LVIS bbox+poly mix JSONL (convex-hull poly)")
    ap.add_argument("--lvis-json", type=Path, required=True, help="Raw LVIS annotations JSON")
    ap.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing images. Supports either a flat images/ directory "
            "(.../images/000000391895.jpg) or COCO2017 layout "
            "(.../images/train2017/000000391895.jpg)."
        ),
    )
    ap.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL (pixel coords on resized images)")
    ap.add_argument(
        "--geometry-policy",
        type=str,
        default="mix",
        choices=("mix", "poly_prefer_semantic", "bbox_only"),
        help=(
            "Geometry output policy (single geometry per instance). "
            "'mix' uses cost-effectiveness; 'poly_prefer_semantic' ignores cost-effectiveness and only falls back "
            "to bbox for semantic-guard/cap reasons; 'bbox_only' always emits bbox_2d."
        ),
    )
    ap.add_argument("--poly-cap", type=int, required=True, help="Max polygon vertices, e.g. 10 or 20")
    ap.add_argument(
        "--lambda-iou-per-extra-point",
        type=float,
        default=0.01,
        help="Cost threshold: require delta_iou >= lambda*(poly_cap-2) to keep poly (default: 0.01)",
    )
    ap.add_argument("--max-images", type=int, default=None, help="Optional limit for quick tests")

    # Drop low-diversity dense images (whole-record drop).
    ap.add_argument("--drop-min-objects", type=int, default=50)
    ap.add_argument("--drop-max-unique", type=int, default=3)
    ap.add_argument("--drop-min-top1-ratio", type=float, default=0.95)
    ap.add_argument("--drop-max-effective-classes", type=float, default=None)
    ap.add_argument("--drop-hard-max-objects", type=int, default=None)

    # Optional semantic guard: force bbox for edge cases where visible mask is too fragmentary.
    # All thresholds are in [0,1] unless stated otherwise.
    ap.add_argument(
        "--force-bbox-hull-iou",
        type=float,
        default=None,
        help="Force bbox if hull_iou (= area_gt/area_hull) <= this value (disabled if unset)",
    )
    ap.add_argument(
        "--force-bbox-fill",
        type=float,
        default=None,
        help="Force bbox if fill_ratio (= area_gt/area_bbox) <= this value (disabled if unset)",
    )
    ap.add_argument(
        "--force-bbox-min-bbox-frac",
        type=float,
        default=0.0,
        help="Guard: only force bbox when bbox_area/image_area >= this value (default: 0)",
    )
    ap.add_argument(
        "--force-bbox-min-area",
        type=float,
        default=0.0,
        help="Guard: only force bbox when GT mask area >= this value (default: 0)",
    )
    ap.add_argument(
        "--force-bbox-max-aspect",
        type=float,
        default=None,
        help="Guard: only apply the (low-hull/low-fill) rule when bbox aspect <= this value (disabled if unset)",
    )
    ap.add_argument(
        "--force-bbox-min-parts",
        type=int,
        default=None,
        help="Multipart rule: require >= this many segmentation parts (disabled if unset)",
    )
    ap.add_argument(
        "--force-bbox-hull-iou-multipart",
        type=float,
        default=None,
        help="Multipart rule: force bbox if hull_iou <= this value when parts >= min-parts (disabled if unset)",
    )
    ap.add_argument(
        "--force-bbox-thin-aspect",
        type=float,
        default=4.0,
        help="Exception: if bbox aspect > this, do NOT force bbox when hull_iou is reasonable (default: 4.0)",
    )
    ap.add_argument(
        "--force-bbox-thin-min-hull-iou",
        type=float,
        default=0.2,
        help="Exception: if bbox aspect > thin-aspect AND hull_iou > this, keep poly (default: 0.2)",
    )

    ap.add_argument("--stats-json", type=Path, default=None, help="Optional: write build stats as JSON")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    policy = str(args.geometry_policy).strip().lower()
    poly_cap = int(args.poly_cap)
    if poly_cap < 3:
        raise SystemExit("--poly-cap must be >= 3")

    # Load LVIS JSON (orjson for speed).
    raw = args.lvis_json.read_bytes()
    data = orjson.loads(raw)

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

    anns_by_image: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        try:
            img_id = int(ann["image_id"])
        except Exception:
            continue
        anns_by_image[img_id].append(ann)

    # Optional progress bar.
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    # Stable ordering for reproducible JSONL output.
    image_ids = sorted(anns_by_image.keys())
    if tqdm is not None:
        image_ids = tqdm(image_ids, desc="Building records", unit="image")

    out_path = args.output_jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = BuildStats()

    with out_path.open("w", encoding="utf-8") as f:
        for image_id in image_ids:
            if args.max_images is not None and stats.images_seen >= int(args.max_images):
                break
            stats.images_seen += 1

            info = image_id_to_info.get(int(image_id))
            if not info:
                continue
            file_name = str(info["file_name"])
            basename = os.path.basename(file_name)

            # Prefer the COCO-style `train2017/xxx.jpg` path when present. Fall back to
            # a legacy "flat images/" layout keyed by basename.
            #
            # This keeps the script usable in both:
            # - public_data/lvis/raw/images/{train2017,val2017}/...
            # - public_data/lvis/rescale_32_768_*/images/*.jpg
            candidate_paths = [
                (args.images_dir / file_name),
                (args.images_dir / basename),
            ]
            resized_path = None
            for p in candidate_paths:
                if p.exists():
                    resized_path = p
                    break

            if resized_path is None:
                stats.images_missing_resized += 1
                continue
            # Important: do NOT resolve symlinks here. We want JSONLs to keep stable
            # `images/...`-style paths when `args.images_dir` is a symlink within the
            # dataset directory (common in our reproducible exports).
            image_rel = os.path.relpath(str(resized_path), str(out_path.parent))

            orig_w = int(info["width"])
            orig_h = int(info["height"])
            tgt_h, tgt_w = smart_resize(height=orig_h, width=orig_w)
            sx = float(tgt_w) / float(orig_w)
            sy = float(tgt_h) / float(orig_h)

            objects: List[Dict[str, Any]] = []
            for ann in anns_by_image[int(image_id)]:
                stats.anns_seen += 1
                try:
                    cat_name = cat_id_to_name[int(ann["category_id"])]
                except Exception:
                    continue

                # bbox (always available in LVIS v1)
                try:
                    bbox_xyxy = _bbox_xyxy_from_coco(ann["bbox"])
                except Exception:
                    continue
                bbox_scaled = clamp_points(scale_points(bbox_xyxy, sx, sy), tgt_w, tgt_h)
                bbox_scaled = _fix_bbox_xyxy(bbox_scaled, tgt_w, tgt_h)

                if policy == "bbox_only":
                    objects.append({"bbox_2d": bbox_scaled, "desc": cat_name})
                    stats.objects_written += 1
                    stats.objects_bbox += 1
                    continue

                # poly candidate (convex hull)
                hull = _poly_candidate_from_segmentation(ann.get("segmentation"))
                if hull is None:
                    stats.bad_segmentation += 1
                    objects.append({"bbox_2d": bbox_scaled, "desc": cat_name})
                    stats.objects_written += 1
                    stats.objects_bbox += 1
                    continue

                if len(hull) > poly_cap:
                    # Ineligible for poly under cap; must be bbox.
                    stats.objects_poly_ineligible += 1
                    objects.append({"bbox_2d": bbox_scaled, "desc": cat_name})
                    stats.objects_written += 1
                    stats.objects_bbox += 1
                    continue

                # Compute IoU gains cheaply using areas (GT inside bbox and inside hull).
                area_gt = float(ann.get("area", 0.0))
                x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
                bbox_area = float(w) * float(h)
                hull_area = _poly_area(hull)
                if bbox_area <= 0.0 or hull_area <= 0.0 or area_gt <= 0.0:
                    stats.bad_hull += 1
                    objects.append({"bbox_2d": bbox_scaled, "desc": cat_name})
                    stats.objects_written += 1
                    stats.objects_bbox += 1
                    continue

                iou_bbox = area_gt / bbox_area
                iou_poly = area_gt / hull_area
                if iou_bbox > 1.0:
                    iou_bbox = 1.0
                if iou_poly > 1.0:
                    iou_poly = 1.0

                # Optional semantic override:
                # - fill_ratio (area_gt / bbox_area) is low for thin/occluded objects.
                # - hull_iou (area_gt / hull_area) is low when the hull grossly over-encloses.
                # Both are cheap proxies that correlate with cases where "poly" hurts semantic correctness.
                fill_ratio = float(iou_bbox)
                hull_iou = float(iou_poly)
                img_area = float(orig_w) * float(orig_h)
                bbox_frac = float(bbox_area / img_area) if img_area > 0 else 0.0
                bbox_aspect = (
                    float(max(w / h, h / w)) if float(w) > 0.0 and float(h) > 0.0 else float("inf")
                )
                n_parts = _count_valid_poly_parts(ann.get("segmentation"))

                force_bbox = False
                forced_rule1 = False
                forced_rule2 = False

                # Rule 1: (hull_iou low OR fill low) under basic guards.
                low_hull = (
                    args.force_bbox_hull_iou is not None
                    and hull_iou <= float(args.force_bbox_hull_iou)
                )
                low_fill = (
                    args.force_bbox_fill is not None
                    and fill_ratio <= float(args.force_bbox_fill)
                )
                if (low_hull or low_fill) and bbox_frac >= float(args.force_bbox_min_bbox_frac) and area_gt >= float(
                    args.force_bbox_min_area
                ):
                    if args.force_bbox_max_aspect is None or bbox_aspect <= float(args.force_bbox_max_aspect):
                        force_bbox = True
                        forced_rule1 = True

                # Rule 2: multipart + hull_iou low (often tiny visible fragments).
                if (
                    not force_bbox
                    and args.force_bbox_min_parts is not None
                    and args.force_bbox_hull_iou_multipart is not None
                    and n_parts >= int(args.force_bbox_min_parts)
                    and hull_iou <= float(args.force_bbox_hull_iou_multipart)
                    and bbox_frac >= float(args.force_bbox_min_bbox_frac)
                ):
                    force_bbox = True
                    forced_rule2 = True

                # Exception: protect very thin objects from getting boxed if hull is still reasonable.
                if (
                    force_bbox
                    and bbox_aspect > float(args.force_bbox_thin_aspect)
                    and hull_iou > float(args.force_bbox_thin_min_hull_iou)
                ):
                    force_bbox = False
                    forced_rule1 = False
                    forced_rule2 = False

                if force_bbox:
                    stats.objects_bbox_forced += 1
                    if forced_rule1:
                        stats.objects_bbox_forced_rule1 += 1
                    if forced_rule2:
                        stats.objects_bbox_forced_rule2 += 1
                    if stats.forced_by_category is None:
                        stats.forced_by_category = {}
                    stats.forced_by_category[cat_name] = int(stats.forced_by_category.get(cat_name, 0)) + 1

                    objects.append({"bbox_2d": bbox_scaled, "desc": cat_name})
                    stats.objects_written += 1
                    stats.objects_bbox += 1
                    continue

                if policy == "poly_prefer_semantic":
                    # Prefer poly whenever it is eligible under cap and not semantically guarded.
                    flat_hull: List[float] = []
                    for x0, y0 in hull:
                        flat_hull.extend([float(x0), float(y0)])
                    poly_scaled = clamp_points(scale_points(flat_hull, sx, sy), tgt_w, tgt_h)
                    poly_scaled = _clean_int_poly(poly_scaled)
                    if (not poly_scaled) or (_poly_area2_flat_int(poly_scaled) == 0):
                        stats.bad_hull += 1
                        objects.append({"bbox_2d": bbox_scaled, "desc": cat_name})
                        stats.objects_written += 1
                        stats.objects_bbox += 1
                        continue

                    objects.append(
                        {"poly": poly_scaled, "poly_points": len(poly_scaled) // 2, "desc": cat_name}
                    )
                    stats.objects_written += 1
                    stats.objects_poly += 1
                    continue

                delta = iou_poly - iou_bbox
                tau = float(args.lambda_iou_per_extra_point) * float(max(0, poly_cap - 2))

                # Prefer bbox when poly's IoU gain is not worth the extra points.
                if delta < tau:
                    stats.objects_bbox_preferred += 1
                    objects.append({"bbox_2d": bbox_scaled, "desc": cat_name})
                    stats.objects_written += 1
                    stats.objects_bbox += 1
                    continue

                # Keep hull poly.
                flat_hull: List[float] = []
                for x0, y0 in hull:
                    flat_hull.extend([float(x0), float(y0)])
                poly_scaled = clamp_points(scale_points(flat_hull, sx, sy), tgt_w, tgt_h)
                poly_scaled = _clean_int_poly(poly_scaled)
                if (not poly_scaled) or (_poly_area2_flat_int(poly_scaled) == 0):
                    stats.bad_hull += 1
                    objects.append({"bbox_2d": bbox_scaled, "desc": cat_name})
                    stats.objects_written += 1
                    stats.objects_bbox += 1
                    continue

                objects.append(
                    {"poly": poly_scaled, "poly_points": len(poly_scaled) // 2, "desc": cat_name}
                )
                stats.objects_written += 1
                stats.objects_poly += 1

            if not objects:
                continue

            # Keep object ordering stable and compatible with other converters.
            objects = sort_objects_by_topleft(objects)

            if _should_drop_record(
                objects,
                drop_min_objects=int(args.drop_min_objects),
                drop_max_unique=int(args.drop_max_unique),
                drop_min_top1_ratio=float(args.drop_min_top1_ratio)
                if args.drop_min_top1_ratio is not None
                else None,
                drop_max_effective_classes=float(args.drop_max_effective_classes)
                if args.drop_max_effective_classes is not None
                else None,
                drop_hard_max_objects=int(args.drop_hard_max_objects)
                if args.drop_hard_max_objects is not None
                else None,
            ):
                stats.images_dropped_low_div += 1
                continue

            rec = {
                "images": [str(image_rel)],
                "objects": objects,
                "width": int(tgt_w),
                "height": int(tgt_h),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.images_written += 1

    print("[build_lvis_hull_mix] done")
    print(json.dumps(stats.as_dict(), indent=2, ensure_ascii=False))
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
        print(f"[build_lvis_hull_mix] wrote stats: {args.stats_json}")


if __name__ == "__main__":
    main()

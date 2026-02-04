import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple


@dataclass
class ConvertStats:
    images_total: int = 0
    images_written: int = 0
    images_skipped_empty: int = 0

    anns_total: int = 0
    anns_kept: int = 0
    anns_skipped_crowd: int = 0
    anns_skipped_missing_bbox: int = 0
    anns_skipped_missing_category: int = 0
    anns_skipped_invalid_bbox: int = 0
    anns_skipped_outside_image: int = 0


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _xywh_to_xyxy(bbox_xywh: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = map(float, bbox_xywh)
    return x, y, x + w, y + h


def _clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def _is_bbox_valid(x1: float, y1: float, x2: float, y2: float, *, eps: float = 1e-6) -> bool:
    return (x2 - x1) > eps and (y2 - y1) > eps


def _is_bbox_inside(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    *,
    eps: float = 1e-6,
) -> bool:
    return (
        x1 >= -eps
        and y1 >= -eps
        and x2 <= float(width) + eps
        and y2 <= float(height) + eps
    )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert COCO 2017 instances annotations to CoordExp JSONL contract",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output contract: docs/data/JSONL_CONTRACT.md
- images: ["images/<split>/<file_name>"] (relative to JSONL directory)
- objects: [{"bbox_2d": [x1,y1,x2,y2], "desc": "category_name", ...}]

Notes:
- COCO stores bbox as [x,y,width,height]; this converter emits bbox_2d as [x1,y1,x2,y2]
  (the CoordExp geometry convention, see src/datasets/geometry.py).
- This converter does NOT resize images or modify pixel coordinates.
- Out-of-bounds boxes are skipped by default (preserves geometry of kept instances).
""",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        required=True,
        help="Dataset split for provenance/logging only",
    )

    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to public_data/coco/raw (used for default annotations path and for stats outputs)",
    )

    parser.add_argument(
        "--annotation",
        type=str,
        default=None,
        help="Path to instances_*.json (defaults to <raw_dir>/annotations/instances_<split>2017.json)",
    )

    parser.add_argument(
        "--image_dir_name",
        type=str,
        required=True,
        help='Directory name under raw/images (e.g. "train2017" or "val2017") used to build relative paths',
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path (typically public_data/coco/raw/{train,val}.jsonl)",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of images written (deterministic prefix after sorting)",
    )

    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep images with zero kept objects (default: drop them)",
    )

    parser.add_argument(
        "--keep-crowd",
        action="store_true",
        help="Keep crowd annotations (iscrowd=1). Default: skip.",
    )

    parser.add_argument(
        "--clip-boxes",
        action="store_true",
        help="Clip boxes to image bounds instead of skipping out-of-bounds boxes.",
    )

    parser.add_argument(
        "--keep-outside",
        action="store_true",
        help="Keep out-of-bounds boxes as-is (may fail validation).",
    )

    parser.add_argument(
        "--require-80-categories",
        action="store_true",
        help="Fail conversion if the annotation file does not define exactly 80 categories.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output)

    if args.annotation is None:
        annotation_path = raw_dir / "annotations" / f"instances_{args.split}2017.json"
    else:
        annotation_path = Path(args.annotation)

    if not annotation_path.exists():
        print(f"✗ Error: annotation not found: {annotation_path}", file=sys.stderr)
        sys.exit(1)

    coco = _load_json(annotation_path)

    images: List[Dict[str, Any]] = list(coco.get("images", []))
    annotations: List[Dict[str, Any]] = list(coco.get("annotations", []))
    categories: List[Dict[str, Any]] = list(coco.get("categories", []))

    cat_id_to_name: Dict[int, str] = {}
    for c in categories:
        if "id" not in c or "name" not in c:
            continue
        cat_id_to_name[int(c["id"])] = str(c["name"])

    if args.require_80_categories and len(cat_id_to_name) != 80:
        print(
            f"✗ Error: expected exactly 80 categories, got {len(cat_id_to_name)}",
            file=sys.stderr,
        )
        sys.exit(2)

    image_id_to_image: Dict[int, Dict[str, Any]] = {}
    for img in images:
        if "id" not in img:
            continue
        image_id_to_image[int(img["id"])] = img

    anns_by_image_id: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if "image_id" not in ann:
            continue
        anns_by_image_id[int(ann["image_id"])].append(ann)

    stats = ConvertStats(images_total=len(image_id_to_image), anns_total=len(annotations))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COCO 2017 -> CoordExp JSONL converter")
    print("=" * 70)
    print(f"  split: {args.split}")
    print(f"  annotation: {annotation_path}")
    print(f"  output: {output_path}")
    print(f"  image_dir_name: {args.image_dir_name}")
    print(f"  max_samples: {args.max_samples}")
    print(f"  keep_empty: {args.keep_empty}")
    print(f"  keep_crowd: {args.keep_crowd}")
    print(f"  clip_boxes: {args.clip_boxes}")
    print(f"  keep_outside: {args.keep_outside}")
    print(f"  categories_defined: {len(cat_id_to_name)}")
    print("=" * 70)

    # Deterministic ordering: images by id; annotations by id when present.
    image_ids = sorted(image_id_to_image.keys())

    def _ann_sort_key(ann: Dict[str, Any]) -> Tuple[int, int]:
        ann_id = int(ann.get("id", 0))
        cat_id = int(ann.get("category_id", 0))
        return (ann_id, cat_id)

    images_written = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for image_id in image_ids:
            img = image_id_to_image[image_id]
            width = int(img.get("width", 0))
            height = int(img.get("height", 0))
            file_name = str(img.get("file_name", ""))

            if not file_name or width <= 0 or height <= 0:
                continue

            rel_image_path = str(Path("images") / args.image_dir_name / file_name)

            objs: List[Dict[str, Any]] = []
            for ann in sorted(anns_by_image_id.get(image_id, []), key=_ann_sort_key):
                if (not args.keep_crowd) and int(ann.get("iscrowd", 0)) == 1:
                    stats.anns_skipped_crowd += 1
                    continue

                bbox = ann.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    stats.anns_skipped_missing_bbox += 1
                    continue

                try:
                    x1, y1, x2, y2 = _xywh_to_xyxy(bbox)
                except Exception:
                    stats.anns_skipped_invalid_bbox += 1
                    continue

                if not _is_bbox_valid(x1, y1, x2, y2):
                    stats.anns_skipped_invalid_bbox += 1
                    continue

                if args.clip_boxes:
                    x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, width=width, height=height)
                    if not _is_bbox_valid(x1, y1, x2, y2):
                        stats.anns_skipped_invalid_bbox += 1
                        continue
                elif not args.keep_outside:
                    if not _is_bbox_inside(x1, y1, x2, y2, width=width, height=height):
                        stats.anns_skipped_outside_image += 1
                        continue

                cat_id = ann.get("category_id")
                if cat_id is None:
                    stats.anns_skipped_missing_category += 1
                    continue

                try:
                    cat_id_int = int(cat_id)
                except Exception:
                    stats.anns_skipped_missing_category += 1
                    continue

                cat_name = cat_id_to_name.get(cat_id_int)
                if not cat_name:
                    stats.anns_skipped_missing_category += 1
                    continue

                objs.append(
                    {
                        "bbox_2d": [x1, y1, x2, y2],
                        "desc": cat_name,
                        "category_id": cat_id_int,
                        "category_name": cat_name,
                        "coco_ann_id": int(ann.get("id", 0)),
                    }
                )
                stats.anns_kept += 1

            if not objs and not args.keep_empty:
                stats.images_skipped_empty += 1
                continue

            row: Dict[str, Any] = {
                "images": [rel_image_path],
                "objects": objs,
                "width": width,
                "height": height,
                "image_id": image_id,
                "file_name": rel_image_path,
                "metadata": {
                    "source": "coco2017",
                    "split": args.split,
                },
            }

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats.images_written += 1
            images_written += 1

            if args.max_samples is not None and images_written >= int(args.max_samples):
                break

    categories_out = output_path.parent / "categories.json"
    categories_payload = [
        {"id": cid, "name": name} for cid, name in sorted(cat_id_to_name.items(), key=lambda x: x[0])
    ]
    _write_json(categories_out, categories_payload)

    stats_out = output_path.parent / "conversion_stats.json"
    _write_json(stats_out, stats.__dict__)

    print("\n✓ Conversion complete")
    print(f"  images_written: {stats.images_written}")
    print(f"  anns_kept: {stats.anns_kept}")
    print(f"  categories.json: {categories_out}")
    print(f"  conversion_stats.json: {stats_out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise

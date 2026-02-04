import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ErrorItem:
    line_num: int
    image_id: Optional[int]
    message: str


def _load_categories(path: Path) -> Dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("categories.json must be a list of {id,name} objects")

    out: Dict[int, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "id" not in item or "name" not in item:
            continue
        out[int(item["id"])] = str(item["name"])
    return out


def _is_bbox_inside(
    bbox: List[float],
    width: int,
    height: int,
    *,
    eps: float = 1e-6,
) -> Tuple[bool, str]:
    if len(bbox) != 4:
        return False, f"bbox_2d must have 4 values, got {len(bbox)}"

    x1, y1, x2, y2 = map(float, bbox)
    if x2 <= x1 + eps:
        return False, f"invalid bbox_2d: x2 ({x2}) <= x1 ({x1})"
    if y2 <= y1 + eps:
        return False, f"invalid bbox_2d: y2 ({y2}) <= y1 ({y1})"

    if x1 < -eps or y1 < -eps:
        return False, f"bbox_2d has negative coord(s): {bbox}"
    if x2 > float(width) + eps or y2 > float(height) + eps:
        return False, f"bbox_2d exceeds image bounds (w={width}, h={height}): {bbox}"

    return True, ""


def _write_jsonl_sample(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate COCO->CoordExp JSONL (bbox inside bounds + 80-class check)",
    )

    parser.add_argument(
        "jsonl",
        type=str,
        help="Path to converted JSONL (e.g. public_data/coco/raw/train.jsonl)",
    )

    parser.add_argument(
        "--categories_json",
        type=str,
        default=None,
        help="Path to categories.json produced by the converter (recommended)",
    )

    parser.add_argument(
        "--require-80",
        action="store_true",
        help="Fail if the converter did not use exactly 80 categories.",
    )

    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum number of errors to print (default: 20)",
    )

    parser.add_argument(
        "--sample-n",
        type=int,
        default=5,
        help="Number of converted image records to write to --sample_out (default: 5)",
    )

    parser.add_argument(
        "--sample_out",
        type=str,
        default=None,
        help="If set, write the first N records to this JSONL path",
    )

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"✗ Error: missing JSONL: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    expected_categories: Optional[Dict[int, str]] = None
    if args.categories_json is not None:
        expected_categories = _load_categories(Path(args.categories_json))

    errors: List[ErrorItem] = []
    used_category_ids: Set[int] = set()
    used_category_names: Set[str] = set()

    num_lines = 0
    num_objects = 0

    sample_rows: List[Dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            num_lines += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(ErrorItem(line_num=line_num, image_id=None, message=f"invalid JSON: {exc}"))
                continue

            if len(sample_rows) < int(args.sample_n):
                if isinstance(row, dict):
                    sample_rows.append(row)

            if not isinstance(row, dict):
                errors.append(
                    ErrorItem(line_num=line_num, image_id=None, message=f"row must be dict, got {type(row).__name__}")
                )
                continue

            image_id = row.get("image_id")
            image_id_int: Optional[int] = None
            if image_id is not None:
                try:
                    image_id_int = int(image_id)
                except Exception:
                    image_id_int = None

            width = row.get("width")
            height = row.get("height")
            if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
                errors.append(
                    ErrorItem(
                        line_num=line_num,
                        image_id=image_id_int,
                        message=f"invalid width/height: width={width}, height={height}",
                    )
                )
                continue

            objects = row.get("objects")
            if not isinstance(objects, list):
                errors.append(
                    ErrorItem(
                        line_num=line_num,
                        image_id=image_id_int,
                        message=f"objects must be list, got {type(objects).__name__}",
                    )
                )
                continue

            for obj in objects:
                num_objects += 1
                if not isinstance(obj, dict):
                    errors.append(
                        ErrorItem(
                            line_num=line_num,
                            image_id=image_id_int,
                            message=f"object must be dict, got {type(obj).__name__}",
                        )
                    )
                    continue

                desc = obj.get("desc")
                if not isinstance(desc, str) or not desc.strip():
                    errors.append(
                        ErrorItem(
                            line_num=line_num,
                            image_id=image_id_int,
                            message="missing/empty desc",
                        )
                    )
                else:
                    used_category_names.add(desc)

                cat_id = obj.get("category_id")
                if cat_id is not None:
                    try:
                        used_category_ids.add(int(cat_id))
                    except Exception:
                        errors.append(
                            ErrorItem(
                                line_num=line_num,
                                image_id=image_id_int,
                                message=f"invalid category_id: {cat_id}",
                            )
                        )

                bbox = obj.get("bbox_2d")
                if not isinstance(bbox, list):
                    errors.append(
                        ErrorItem(
                            line_num=line_num,
                            image_id=image_id_int,
                            message=f"bbox_2d must be list, got {type(bbox).__name__}",
                        )
                    )
                    continue

                ok, msg = _is_bbox_inside(bbox, width=width, height=height)
                if not ok:
                    errors.append(ErrorItem(line_num=line_num, image_id=image_id_int, message=msg))

    if args.sample_out is not None:
        _write_jsonl_sample(sample_rows, Path(args.sample_out))

    if expected_categories is not None:
        expected_ids = set(expected_categories.keys())
        extra = sorted(used_category_ids - expected_ids)
        if extra:
            errors.append(
                ErrorItem(
                    line_num=0,
                    image_id=None,
                    message=f"extra category_ids not in categories.json: {extra[:20]}",
                )
            )

    if args.require_80:
        if expected_categories is None:
            errors.append(
                ErrorItem(
                    line_num=0,
                    image_id=None,
                    message="--require-80 requires --categories_json (to enforce the 80-class COCO mapping)",
                )
            )
        else:
            if len(expected_categories) != 80:
                errors.append(
                    ErrorItem(
                        line_num=0,
                        image_id=None,
                        message=f"expected categories.json to have 80 categories, got {len(expected_categories)}",
                    )
                )

    print("=" * 70)
    print("COCO JSONL validation")
    print("=" * 70)
    print(f"  file: {jsonl_path}")
    print(f"  images(lines): {num_lines}")
    print(f"  objects(total): {num_objects}")
    print(f"  used_category_names: {len(used_category_names)}")
    if expected_categories is not None:
        print(f"  categories.json: {len(expected_categories)}")
    if args.sample_out is not None:
        print(f"  sample_out: {args.sample_out} (n={args.sample_n})")
    print("=" * 70)

    if errors:
        print(f"✗ FAIL ({len(errors)} issues)")
        for e in errors[: int(args.max_errors)]:
            if e.line_num > 0:
                prefix = f"line {e.line_num}"
            else:
                prefix = "global"
            if e.image_id is not None:
                prefix += f", image_id={e.image_id}"
            print(f"  - {prefix}: {e.message}")
        sys.exit(1)

    print("✓ PASS")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise

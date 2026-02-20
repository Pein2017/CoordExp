"""CPU-only JSONL validator for CoordExp image-size contracts.

This tool is intended to validate cooked JSONL artifacts (train/val) before
launching stage_1 / stage_2 training.

Policy
- CoordExp forbids runtime image resizing because it breaks grounding geometry.
- Therefore, JSONL width/height must satisfy `width * height <= template.max_pixels`.
- For rescaled datasets, width/height are typically multiples of 32.

This script validates:
- JSONL parsing (no invalid JSON)
- width/height are positive integers
- width*height <= max_pixels
- optional: width/height multiples-of-N
- optional: spot-check that referenced images exist and match JSONL dimensions

Example:
  conda run -n ms python scripts/tools/validate_jsonl_max_pixels.py \
    --jsonl public_data/coco/rescale_32_768_bbox_max60/val.coord.jsonl \
    --max-pixels 786432 --multiple-of 32 --image-check-n 64
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


@dataclass
class OversizeExample:
    image: Optional[str]
    width: int
    height: int
    pixels: int


@dataclass
class ImageMismatchExample:
    path: str
    got: Tuple[int, int]
    expected: Tuple[int, int]


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)
        try:
            f = float(s)
        except ValueError:
            return None
        if not f.is_integer():
            return None
        return int(f)
    return None


def _resolve_image_path(*, jsonl_dir: Path, image: str) -> Path:
    p = Path(image)
    if p.is_absolute():
        return p
    return jsonl_dir / p


def scan_jsonl(
    *,
    jsonl_path: Path,
    max_pixels: int,
    multiple_of: int,
    image_check_n: int,
    image_check_mode: str,
) -> int:
    mode = str(image_check_mode).strip().lower()
    if mode not in {"none", "exists", "open"}:
        print(f"FAIL: invalid --image-check-mode={image_check_mode!r} (expected none|exists|open)")
        return 2

    Image = None
    if mode == "open":
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: PIL import failed: {exc!r}")
            return 2

    total_lines = 0
    invalid_json = 0
    invalid_size = 0
    oversize = 0
    non_multiple = 0

    max_seen_pixels = 0
    max_w = 0
    max_h = 0

    oversize_examples: List[OversizeExample] = []

    img_checked = 0
    img_missing_or_unreadable = 0
    img_missing_examples: List[str] = []
    img_size_mismatch = 0
    img_mismatch_examples: List[ImageMismatchExample] = []

    jsonl_dir = jsonl_path.parent

    # CoordExp policy: resized presets must not rely on a symlinked images/ tree.
    # A symlink can mask meta/image misalignment and can cause offline rescale jobs
    # to overwrite raw sources.
    images_dir = jsonl_dir / "images"
    if mode != "none" and images_dir.exists() and images_dir.is_symlink():
        if "rescale" in str(jsonl_dir.name).lower():
            print(f"FAIL: images/ must not be a symlink for rescale presets: {images_dir}")
            print("HINT: Materialize a real images/ directory for this preset.")
            return 2
        print(f"[warn] images/ is a symlink (non-rescale dataset): {images_dir}")

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            if not isinstance(rec, dict):
                invalid_json += 1
                continue

            w = _coerce_int(rec.get("width"))
            h = _coerce_int(rec.get("height"))
            if w is None or h is None or w <= 0 or h <= 0:
                invalid_size += 1
                continue

            max_w = max(max_w, int(w))
            max_h = max(max_h, int(h))
            pixels = int(w) * int(h)
            max_seen_pixels = max(max_seen_pixels, int(pixels))

            if pixels > max_pixels:
                oversize += 1
                if len(oversize_examples) < 5:
                    image = None
                    imgs = rec.get("images") or []
                    if isinstance(imgs, list) and imgs and isinstance(imgs[0], str):
                        image = imgs[0]
                    oversize_examples.append(
                        OversizeExample(
                            image=image,
                            width=int(w),
                            height=int(h),
                            pixels=int(pixels),
                        )
                    )

            if multiple_of > 1 and (
                (int(w) % int(multiple_of)) != 0 or (int(h) % int(multiple_of)) != 0
            ):
                non_multiple += 1

            # Image existence / size spot-check (optional).
            check_all = int(image_check_n) <= 0
            if mode != "none" and (check_all or img_checked < image_check_n):
                imgs = rec.get("images") or []
                if isinstance(imgs, list) and imgs and isinstance(imgs[0], str):
                    p = _resolve_image_path(jsonl_dir=jsonl_dir, image=imgs[0])
                    if not p.exists():
                        img_missing_or_unreadable += 1
                        if len(img_missing_examples) < 5:
                            img_missing_examples.append(f"{str(p)} (missing)")
                    elif mode == "open":
                        try:
                            with Image.open(p) as im:  # type: ignore[attr-defined]
                                iw, ih = im.size
                        except Exception:  # noqa: BLE001
                            img_missing_or_unreadable += 1
                            if len(img_missing_examples) < 5:
                                img_missing_examples.append(f"{str(p)} (unreadable)")
                        else:
                            if int(iw) != int(w) or int(ih) != int(h):
                                img_size_mismatch += 1
                                if len(img_mismatch_examples) < 5:
                                    img_mismatch_examples.append(
                                        ImageMismatchExample(
                                            path=str(p),
                                            got=(int(iw), int(ih)),
                                            expected=(int(w), int(h)),
                                        )
                                    )
                img_checked += 1

    ok = True
    if invalid_json:
        ok = False
    if invalid_size:
        ok = False
    if oversize:
        ok = False
    if non_multiple:
        ok = False
    if img_missing_or_unreadable:
        ok = False
    if img_size_mismatch:
        ok = False

    print("JSONL:", str(jsonl_path))
    print("STATUS:", "OK" if ok else "FAIL")
    print("max_pixels:", int(max_pixels))
    print("multiple_of:", int(multiple_of))
    print("total_lines:", int(total_lines))
    print("invalid_json:", int(invalid_json))
    print("invalid_size:", int(invalid_size))
    print("oversize:", int(oversize))
    print("non_multiple:", int(non_multiple))
    print("max_seen_pixels:", int(max_seen_pixels))
    print("max_width:", int(max_w))
    print("max_height:", int(max_h))

    if oversize_examples:
        print("oversize_examples:")
        for ex in oversize_examples:
            print(
                "  -",
                {
                    "image": ex.image,
                    "width": int(ex.width),
                    "height": int(ex.height),
                    "pixels": int(ex.pixels),
                },
            )

    print("image_check_mode:", mode)
    print("image_spotcheck_n:", int(img_checked))
    print("image_missing_or_unreadable:", int(img_missing_or_unreadable))
    if img_missing_examples:
        print("image_missing_examples:")
        for p in img_missing_examples:
            print("  -", p)
    print("image_size_mismatch:", int(img_size_mismatch))

    if img_mismatch_examples:
        print("image_mismatch_examples:")
        for ex in img_mismatch_examples:
            print("  -", {"path": ex.path, "got": ex.got, "expected": ex.expected})

    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--max-pixels", type=int, default=768 * 32 * 32)
    ap.add_argument("--multiple-of", type=int, default=32)
    ap.add_argument(
        "--image-check-n",
        type=int,
        default=64,
        help="How many records to check image existence/size for (<=0 means all).",
    )
    ap.add_argument(
        "--image-check-mode",
        type=str,
        default="open",
        choices=["none", "exists", "open"],
        help="Image check mode: none (skip), exists (only existence), open (exists + PIL size check).",
    )
    args = ap.parse_args()

    return scan_jsonl(
        jsonl_path=Path(args.jsonl),
        max_pixels=int(args.max_pixels),
        multiple_of=int(args.multiple_of),
        image_check_n=int(args.image_check_n),
        image_check_mode=str(args.image_check_mode),
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Materialize missing rescaled images referenced by a JSONL.

Why
- CoordExp forbids runtime image resizing (ms-swift Template rescale_image).
- Therefore, the image files on disk must already be at the resolution implied
  by the cooked JSONL (width/height).

This tool scans a JSONL and ensures every referenced image exists under a target
root. For missing images, it reads the source image and writes a resized copy
with the exact JSONL width/height.

Typical use case (LVIS stage_2):
- JSONL: public_data/lvis/rescale_32_768_bbox_max60/*coord.jsonl
- Source images: public_data/lvis/raw (contains images/train2017 + images/val2017)
- Target images: public_data/coco/rescale_32_768_bbox_max60 (shared rescaled pool)

Example:
  conda run -n ms python scripts/tools/materialize_rescaled_images_from_jsonl.py \
    --jsonl public_data/lvis/rescale_32_768_bbox_max60/train.bbox_only.max60.coord.jsonl \
    --jsonl public_data/lvis/rescale_32_768_bbox_max60/val.bbox_only.max60.coord.jsonl \
    --src-root public_data/lvis/raw \
    --dst-root public_data/coco/rescale_32_768_bbox_max60 \
    --write

Notes
- The resize kernel matches `src/datasets/preprocessors/resize.py` (LANCZOS).
- This script is CPU-only and intended for offline data preparation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple


@dataclass
class Counters:
    total_lines: int = 0
    invalid_json: int = 0
    invalid_record: int = 0
    unique_images: int = 0
    src_missing: int = 0
    dst_already_present: int = 0
    dst_created: int = 0
    dst_size_mismatch_existing: int = 0
    dst_write_failed: int = 0


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


def _iter_jsonl_paths(values: Iterable[str]) -> Iterable[Path]:
    for v in values:
        p = Path(v)
        if not p.exists():
            raise FileNotFoundError(str(p))
        yield p


def _resolve_under(root: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return root / p


def materialize(
    *,
    jsonl_paths: Iterable[Path],
    src_root: Path,
    dst_root: Path,
    write: bool,
    verify_existing_size: bool,
    limit_missing: int,
) -> int:
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: PIL import failed: {exc!r}")
        return 2

    counters = Counters()
    seen: Set[str] = set()

    # Safety: avoid silently writing into symlinked images/ trees.
    # If dst_root/images is a symlink, replace it with a real directory so we
    # materialize local copies (no symlinked datasets for resized presets).
    images_dir = dst_root / "images"
    if images_dir.exists():
        if images_dir.is_symlink():
            if write:
                images_dir.unlink()
                images_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(
                    f"FAIL: dst images path is a symlink: {images_dir} "
                    "(pass --write to materialize a real directory)"
                )
                return 2
        elif images_dir.is_file():
            print(f"FAIL: dst images path is a file: {images_dir}")
            return 2

    # Lazy tqdm import (optional).
    try:
        from tqdm import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None

    jsonl_paths_list = list(jsonl_paths)

    def _iter_lines():
        for jp in jsonl_paths_list:
            with jp.open("r", encoding="utf-8") as f:
                for line in f:
                    yield jp, line

    iterator = _iter_lines()
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Materialize", unit="line")

    for _jp, line in iterator:
        counters.total_lines += 1
        line = line.strip()
        if not line:
            continue

        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            counters.invalid_json += 1
            continue
        if not isinstance(rec, dict):
            counters.invalid_record += 1
            continue

        imgs = rec.get("images") or []
        if not isinstance(imgs, list) or not imgs or not isinstance(imgs[0], str):
            counters.invalid_record += 1
            continue
        rel_img = imgs[0]

        w = _coerce_int(rec.get("width"))
        h = _coerce_int(rec.get("height"))
        if w is None or h is None or w <= 0 or h <= 0:
            counters.invalid_record += 1
            continue

        if rel_img in seen:
            continue
        seen.add(rel_img)
        counters.unique_images = len(seen)

        src = _resolve_under(src_root, rel_img)
        dst = _resolve_under(dst_root, rel_img)

        if not src.exists():
            counters.src_missing += 1
            if counters.src_missing <= 5:
                print(f"SRC MISSING: {src}")
            continue

        if dst.exists():
            counters.dst_already_present += 1
            if verify_existing_size:
                try:
                    with Image.open(dst) as im:
                        if im.size != (int(w), int(h)):
                            counters.dst_size_mismatch_existing += 1
                            if counters.dst_size_mismatch_existing <= 5:
                                print(
                                    "DST SIZE MISMATCH:",
                                    {"path": str(dst), "got": im.size, "expected": (int(w), int(h))},
                                )
                except Exception as exc:  # noqa: BLE001
                    counters.dst_size_mismatch_existing += 1
                    if counters.dst_size_mismatch_existing <= 5:
                        print(f"DST UNREADABLE (counted as mismatch): {dst}: {exc!r}")
            continue

        # dst missing
        if not write:
            continue

        if limit_missing > 0 and counters.dst_created >= limit_missing:
            continue

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Fast path: if the source image already matches the JSONL meta size, copy
            # the file bytes instead of re-encoding via PIL.
            with Image.open(src) as im:
                src_size = im.size

            if src_size == (int(w), int(h)):
                shutil.copy2(src, dst)
            else:
                with Image.open(src) as im:
                    rgb = im.convert("RGB")
                    resized = rgb.resize(
                        (int(w), int(h)),
                        Image.Resampling.LANCZOS,
                    )
                    resized.save(dst)

            counters.dst_created += 1
            if counters.dst_created <= 3:
                print(f"WROTE: {dst}")
        except Exception as exc:  # noqa: BLE001
            counters.dst_write_failed += 1
            if counters.dst_write_failed <= 5:
                print(f"WRITE FAILED: {dst}: {exc!r}")

    ok = True
    if counters.invalid_json or counters.invalid_record:
        ok = False
    if counters.src_missing:
        ok = False
    if counters.dst_size_mismatch_existing:
        ok = False
    if counters.dst_write_failed:
        ok = False

    print("\nSUMMARY")
    print("jsonl_paths:", [str(p) for p in jsonl_paths_list])
    print("src_root:", str(src_root))
    print("dst_root:", str(dst_root))
    print("write:", bool(write))
    print("verify_existing_size:", bool(verify_existing_size))
    print("limit_missing:", int(limit_missing))
    print("total_lines:", int(counters.total_lines))
    print("invalid_json:", int(counters.invalid_json))
    print("invalid_record:", int(counters.invalid_record))
    print("unique_images:", int(counters.unique_images))
    print("src_missing:", int(counters.src_missing))
    print("dst_already_present:", int(counters.dst_already_present))
    print("dst_created:", int(counters.dst_created))
    print("dst_size_mismatch_existing:", int(counters.dst_size_mismatch_existing))
    print("dst_write_failed:", int(counters.dst_write_failed))
    print("STATUS:", "OK" if ok else "FAIL")

    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, action="append", required=True)
    ap.add_argument("--src-root", type=str, required=True)
    ap.add_argument("--dst-root", type=str, required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument(
        "--verify-existing-size",
        action="store_true",
        help="Open existing dst images and verify size matches JSONL (slow).",
    )
    ap.add_argument(
        "--limit-missing",
        type=int,
        default=0,
        help="Max number of missing images to create (0 means unlimited).",
    )
    args = ap.parse_args()

    jsonl_paths = list(_iter_jsonl_paths(args.jsonl))
    return materialize(
        jsonl_paths=jsonl_paths,
        src_root=Path(args.src_root),
        dst_root=Path(args.dst_root),
        write=bool(args.write),
        verify_existing_size=bool(args.verify_existing_size),
        limit_missing=int(args.limit_missing),
    )


if __name__ == "__main__":
    raise SystemExit(main())

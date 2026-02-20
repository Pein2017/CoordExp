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
import concurrent.futures as cf
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


def _verify_existing_size_worker(task: Tuple[str, int, int]) -> Tuple[str, Dict[str, Any]]:
    from PIL import Image

    dst_s, w, h = task
    expected = (int(w), int(h))
    dst = Path(dst_s)
    try:
        with Image.open(dst) as im:
            got = (int(im.size[0]), int(im.size[1]))
        if got != expected:
            return (
                "mismatch",
                {
                    "path": str(dst),
                    "got": got,
                    "expected": expected,
                },
            )
        return ("ok", {"path": str(dst)})
    except Exception as exc:  # noqa: BLE001
        return (
            "error",
            {
                "path": str(dst),
                "error": repr(exc),
            },
        )


def _materialize_missing_worker(task: Tuple[str, str, int, int]) -> Tuple[str, str]:
    from PIL import Image

    src_s, dst_s, w, h = task
    src = Path(src_s)
    dst = Path(dst_s)
    try:
        if dst.exists():
            return ("exists", str(dst))

        dst.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src) as im:
            src_size = (int(im.size[0]), int(im.size[1]))

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
        return ("created", str(dst))
    except Exception as exc:  # noqa: BLE001
        return ("failed", f"{dst}: {exc!r}")


def materialize(
    *,
    jsonl_paths: Iterable[Path],
    src_root: Path,
    dst_root: Path,
    write: bool,
    verify_existing_size: bool,
    limit_missing: int,
    workers: int,
) -> int:
    try:
        __import__("PIL.Image")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: PIL import failed: {exc!r}")
        return 2

    workers = max(1, int(workers))

    counters = Counters()
    seen: Set[str] = set()
    unique_specs: Dict[str, Tuple[int, int]] = {}

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

    def _with_progress(iterable, *, total: Optional[int], desc: str, unit: str):
        if tqdm is None:
            return iterable
        return tqdm(iterable, total=total, desc=desc, unit=unit)

    scan_iter = _with_progress(_iter_lines(), total=None, desc="Scan JSONL", unit="line")
    for _jp, line in scan_iter:
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
        unique_specs[rel_img] = (int(w), int(h))
        counters.unique_images = len(seen)

    verify_tasks: list[Tuple[str, int, int]] = []
    create_tasks: list[Tuple[str, str, int, int]] = []

    for rel_img, (w, h) in unique_specs.items():
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
                verify_tasks.append((str(dst), int(w), int(h)))
            continue

        if write:
            create_tasks.append((str(src), str(dst), int(w), int(h)))

    if limit_missing > 0 and len(create_tasks) > limit_missing:
        create_tasks = create_tasks[:limit_missing]

    if verify_tasks:
        verify_total = len(verify_tasks)
        if workers > 1:
            verify_chunksize = max(1, verify_total // (workers * 8))
            with cf.ProcessPoolExecutor(max_workers=workers) as executor:
                verify_iter = executor.map(
                    _verify_existing_size_worker,
                    verify_tasks,
                    chunksize=verify_chunksize,
                )
                verify_iter = _with_progress(
                    verify_iter,
                    total=verify_total,
                    desc="Verify existing",
                    unit="img",
                )
                for status, payload in verify_iter:
                    if status == "ok":
                        continue
                    counters.dst_size_mismatch_existing += 1
                    if counters.dst_size_mismatch_existing <= 5:
                        if status == "mismatch":
                            print("DST SIZE MISMATCH:", payload)
                        else:
                            print(
                                f"DST UNREADABLE (counted as mismatch): "
                                f"{payload['path']}: {payload['error']}"
                            )
        else:
            verify_iter = (_verify_existing_size_worker(task) for task in verify_tasks)
            verify_iter = _with_progress(
                verify_iter,
                total=verify_total,
                desc="Verify existing",
                unit="img",
            )
            for status, payload in verify_iter:
                if status == "ok":
                    continue
                counters.dst_size_mismatch_existing += 1
                if counters.dst_size_mismatch_existing <= 5:
                    if status == "mismatch":
                        print("DST SIZE MISMATCH:", payload)
                    else:
                        print(
                            f"DST UNREADABLE (counted as mismatch): "
                            f"{payload['path']}: {payload['error']}"
                        )

    if create_tasks:
        create_total = len(create_tasks)
        if workers > 1:
            create_chunksize = max(1, create_total // (workers * 8))
            with cf.ProcessPoolExecutor(max_workers=workers) as executor:
                create_iter = executor.map(
                    _materialize_missing_worker,
                    create_tasks,
                    chunksize=create_chunksize,
                )
                create_iter = _with_progress(
                    create_iter,
                    total=create_total,
                    desc="Materialize",
                    unit="img",
                )
                for status, payload in create_iter:
                    if status == "created":
                        counters.dst_created += 1
                        if counters.dst_created <= 3:
                            print(f"WROTE: {payload}")
                    elif status == "exists":
                        counters.dst_already_present += 1
                    else:
                        counters.dst_write_failed += 1
                        if counters.dst_write_failed <= 5:
                            print(f"WRITE FAILED: {payload}")
        else:
            create_iter = (_materialize_missing_worker(task) for task in create_tasks)
            create_iter = _with_progress(
                create_iter,
                total=create_total,
                desc="Materialize",
                unit="img",
            )
            for status, payload in create_iter:
                if status == "created":
                    counters.dst_created += 1
                    if counters.dst_created <= 3:
                        print(f"WROTE: {payload}")
                elif status == "exists":
                    counters.dst_already_present += 1
                else:
                    counters.dst_write_failed += 1
                    if counters.dst_write_failed <= 5:
                        print(f"WRITE FAILED: {payload}")

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
    print("workers:", int(workers))
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
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes for verification/materialization.",
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
        workers=max(1, int(args.workers)),
    )


if __name__ == "__main__":
    raise SystemExit(main())

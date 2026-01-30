#!/usr/bin/env python3
"""Download + convert Visual Genome (VG) to the CoordExp JSONL contract.

This script targets the HuggingFace dataset loader:
  https://huggingface.co/datasets/ranjaykrishna/visual_genome

The HF dataset repo contains only a loading script; the actual data are hosted on
the official Visual Genome mirrors (UW/Stanford). We follow the same URLs as the
HF loader and write everything under `public_data/vg/` so paths are stable and
portable within this repo.

Outputs (default):
  public_data/vg/raw/train.jsonl
  public_data/vg/raw/val.jsonl

Each JSONL record matches `docs/DATA_JSONL_CONTRACT.md`:
  {
    "images": ["images/VG_100K/1.jpg"],
    "objects": [{"bbox_2d": [x1,y1,x2,y2], "desc": "person"}],
    "width": 640,
    "height": 480,
    "metadata": {...}
  }

Notes:
- VG does not ship official train/val splits. We create a deterministic split by
  image_id modulo (configurable).
- Bounding boxes are converted from (x,y,w,h) to (x1,y1,x2,y2) and clipped into
  [0,width-1]/[0,height-1] by default to ensure `convert_to_coord_tokens.py`
  never produces the out-of-range bin 1000.
- High-confidence junk labels like articles/pronouns (e.g., "this", "the") are
  dropped by default to reduce supervision noise (opt-out: --no-filter-junk-descs).
- For quick smoke tests, pass `--max-seconds 300` to stop downloads early.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from public_data.converters.sorting import sort_objects_tlbr
from src.datasets.geometry import clamp_points, round_points

try:
    from public_data.vg.checksums import EXPECTED_SHA256 as _VG_EXPECTED_SHA256
except Exception:
    _VG_EXPECTED_SHA256 = {}

try:
    from public_data.vg.junk_descs import is_high_conf_junk_desc as _is_high_conf_junk_desc
except Exception:

    def _is_high_conf_junk_desc(desc: str) -> bool:  # type: ignore
        return False

# URLs are copied from the HF dataset loader `visual_genome.py` (snapshot).
VG_BASE = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset"
VG_IMAGE_META_URL = f"{VG_BASE}/image_data.json.zip"
VG_OBJECTS_URLS = {
    "1.0.0": f"{VG_BASE}/objects_v1.json.zip",
    "1.2.0": f"{VG_BASE}/objects_v1_2.json.zip",
}
VG_REGION_DESCRIPTIONS_URL = f"{VG_BASE}/region_descriptions.json.zip"

# Updated image URLs - Stanford sources confirmed working
VG_IMAGE_ZIPS = {
    # The zip names are historical; the extracted folders are VG_100K and VG_100K_2.
    # Using Stanford URLs which are confirmed working
    "images.zip": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "images2.zip": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
}


_URL_RE = re.compile(
    r"^https://cs\.stanford\.edu/people/rak248/(VG_100K(?:_2)?)/([0-9]+\.jpg)$"
)


def _now() -> float:
    return time.time()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path) -> bool:
    expected = _VG_EXPECTED_SHA256.get(path.name)
    if not expected:
        print(f"[warning] No expected sha256 for {path.name}; skipping checksum verification.")
        return True
    if not path.exists():
        print(f"[error] Missing file for checksum verification: {path}")
        return False
    actual = _sha256_file(path)
    if actual.lower() != expected.lower():
        print(f"[error] sha256 mismatch for {path.name}")
        print(f"  expected: {expected}")
        print(f"  actual:   {actual}")
        return False
    print(f"[checksum] OK: {path.name}")
    return True


def _download_then_verify(
    url: str,
    out_path: Path,
    *,
    timeout_s: int | None,
    no_proxy: bool,
    verify_checksums: bool,
) -> bool:
    if out_path.exists() and verify_checksums:
        if _verify_sha256(out_path):
            print(f"[skip] Already present and checksum OK: {out_path.name}")
            return True
        print(f"[warning] Existing file has wrong checksum; re-downloading: {out_path.name}")

    ok = _download_with_wget(
        url,
        out_path,
        timeout_s=timeout_s,
        no_proxy=no_proxy,
    )
    if not ok:
        return False
    if verify_checksums and not _verify_sha256(out_path):
        return False
    return True


def _run(cmd: list[str], *, timeout_s: int | None = None) -> int:
    """Run a subprocess command.

    Returns the exit code.
    On timeout, returns 124 (matching `timeout(1)` convention) and leaves any
    partial outputs on disk (important for resuming wget downloads).
    """
    try:
        proc = subprocess.run(cmd, check=False, timeout=timeout_s)
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        return 124


def _download_with_wget(
    url: str,
    out_path: Path,
    *,
    timeout_s: int | None,
    no_proxy: bool,
    max_retries: int = 3,
) -> bool:
    """Download a URL to out_path with resume support and retry logic.

    We intentionally keep partial files on timeout so a future run with
    `--continue` can resume.
    """
    _ensure_dir(out_path.parent)

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"[retry] Attempt {attempt + 1}/{max_retries} for: {url}")

        cmd = [
            "wget",
            "--continue",
            "--progress=bar:force",
            "--show-progress",
            "--tries=3",
            *(["--no-proxy"] if no_proxy else []),
            url,
            "-O",
            str(out_path),
        ]
        rc = _run(cmd, timeout_s=timeout_s)
        if rc == 124:
            print(f"[timeout] wget exceeded {timeout_s}s: {url}")
            if attempt == max_retries - 1:
                return False
            continue
        if rc != 0:
            print(f"[error] wget failed with code {rc}: {url}")
            if attempt == max_retries - 1:
                return False
            continue
        if not out_path.exists() or out_path.stat().st_size == 0:
            print(f"[error] download failed (empty file): {url}")
            if attempt == max_retries - 1:
                return False
            continue
        # Success
        print(f"[success] Downloaded: {url} ({out_path.stat().st_size} bytes)")
        return True

    return False


def _unzip(zip_path: Path, out_dir: Path, *, timeout_s: int | None) -> bool:
    _ensure_dir(out_dir)

    # First test the zip file integrity
    test_cmd = ["unzip", "-t", str(zip_path)]
    test_rc = _run(test_cmd, timeout_s=30)  # Short timeout for testing
    if test_rc != 0:
        print(f"[error] zip file corrupted (test failed): {zip_path}")
        return False

    # Extract the zip file
    cmd = ["unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)]
    rc = _run(cmd, timeout_s=timeout_s)
    if rc == 124:
        print(f"[timeout] unzip exceeded {timeout_s}s: {zip_path}")
        return False
    if rc != 0:
        print(f"[error] unzip failed with code {rc}: {zip_path}")
        return False

    print(f"[success] Extracted: {zip_path}")
    return True


def _find_one(path: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        p = path / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing expected file under {path}: {candidates}")


def _clip_bbox_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    width: int,
    height: int,
) -> Optional[List[int]]:
    """Clip + round bbox into [0,width-1]/[0,height-1] to avoid norm bin 1000.

    This uses the repo's canonical geometry helpers (`src/datasets/geometry.py`)
    to preserve coordinate semantics.
    """
    if width <= 1 or height <= 1:
        return None
    pts = clamp_points([x1, y1, x2, y2], width, height)
    x1c, y1c, x2c, y2c = pts
    if x2c <= x1c or y2c <= y1c:
        return None
    return pts


def _sanitize_desc(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    # Drop obvious multi-name delimiters to keep desc flat and JSONLinesBuilder-safe.
    s = value.strip()
    if not s:
        return None
    for delim in [",", "/", ":", ";"]:
        if delim in s:
            s = s.split(delim, 1)[0].strip()
    # Normalize whitespace (avoid disallowed patterns like double spaces).
    s = re.sub(r"\s+", " ", s)
    if not s:
        return None
    return s


def _finalize_metadata(objects: List[Dict[str, Any]]) -> None:
    """Remove non-contract fields from objects in-place.

    The global JSONL contract only allows `desc` plus exactly one geometry field
    (`bbox_2d` or `poly`, with optional `poly_points`). Any additional fields are
    dropped here so downstream validators and builders remain schema-strict.
    """

    for o in objects:
        if not isinstance(o, dict):
            continue
        for k in ("object_id", "region_id"):
            if k in o:
                o.pop(k, None)


def _pick_name(names: Any) -> Optional[str]:
    # Type guard: names should be either a list of strings or a single string
    if not isinstance(names, (list, str)):
        return None
    if isinstance(names, list):
        for n in names:
            out = _sanitize_desc(n)
            if out:
                return out
        return None
    return _sanitize_desc(names)


def _sanitize_phrase(value: Any) -> Optional[str]:
    """Sanitize free-form region phrases.

    Unlike object `names`, region phrases may include commas and other
    punctuation, so we only strip and normalize whitespace.
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s)
    return s if s else None


def _image_relpath_from_url(url: str) -> Optional[str]:
    m = _URL_RE.fullmatch(url.strip())
    if not m:
        return None
    folder, filename = m.group(1), m.group(2)
    return f"images/{folder}/{filename}"


def _iter_json_array(path: Path, *, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Stream a JSON array from disk without loading the whole file.

    The VG annotations are large (~100k items). This parser is a small streaming
    adapter around json.JSONDecoder.raw_decode.
    """
    decoder = json.JSONDecoder()
    buf = ""
    idx = 0

    with path.open("r", encoding="utf-8") as f:
        # Prime buffer.
        while True:
            if idx >= len(buf):
                chunk = f.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Unexpected EOF while reading JSON array: {path}")
                buf = buf[idx:] + chunk
                idx = 0

            # Skip whitespace.
            while idx < len(buf) and buf[idx].isspace():
                idx += 1

            if idx < len(buf):
                break

        if idx >= len(buf) or buf[idx] != "[":
            raise ValueError(f"Expected '[' at start of JSON array: {path}")
        idx += 1

        while True:
            # Ensure buffer.
            if idx >= len(buf):
                chunk = f.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Unexpected EOF in JSON array: {path}")
                buf = buf[idx:] + chunk
                idx = 0

            # Skip whitespace + commas.
            while True:
                while idx < len(buf) and buf[idx].isspace():
                    idx += 1
                if idx < len(buf) and buf[idx] == ",":
                    idx += 1
                    continue
                break

            if idx >= len(buf):
                continue

            if buf[idx] == "]":
                return

            try:
                obj, end = decoder.raw_decode(buf, idx)
            except json.JSONDecodeError:
                # Need more data.
                chunk = f.read(chunk_size)
                if not chunk:
                    raise
                buf = buf[idx:] + chunk
                idx = 0
                continue

            yield obj
            idx = end


@dataclass
class ConvertStats:
    images_total: int = 0
    images_written_train: int = 0
    images_written_val: int = 0
    images_skipped_missing_ann: int = 0
    images_skipped_bad_url: int = 0
    objects_kept: int = 0
    objects_deduped: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "images_total": self.images_total,
            "images_written_train": self.images_written_train,
            "images_written_val": self.images_written_val,
            "images_skipped_missing_ann": self.images_skipped_missing_ann,
            "images_skipped_bad_url": self.images_skipped_bad_url,
            "objects_kept": self.objects_kept,
            "objects_deduped": self.objects_deduped,
        }


def _convert_pair_to_record(
    img_meta: Dict[str, Any],
    ann: Dict[str, Any],
    *,
    min_box_area: float,
    min_box_dimension: float,
    clip_boxes: bool,
    dedupe_objects: bool,
    filter_junk_descs: bool,
) -> Tuple[Optional[Dict[str, Any]], int, int]:
    """Return (record, kept_objects, deduped_objects)."""
    image_id = img_meta.get("image_id")
    url = img_meta.get("url")
    width = img_meta.get("width")
    height = img_meta.get("height")

    if not isinstance(image_id, int) or not isinstance(url, str):
        return None, 0
    if not isinstance(width, int) or not isinstance(height, int):
        return None, 0
    if width <= 0 or height <= 0:
        return None, 0

    rel = _image_relpath_from_url(url)
    if rel is None:
        return None, 0

    objects = ann.get("objects")
    if not isinstance(objects, list) or not objects:
        return None, 0

    out_objs: List[Dict[str, Any]] = []
    kept = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_dict: Dict[str, Any] = obj  # type: ignore
        x = obj_dict.get("x")
        y = obj_dict.get("y")
        w = obj_dict.get("w")
        h = obj_dict.get("h")
        names = obj_dict.get("names")

        if x is None or y is None or w is None or h is None:
            continue

        try:
            xf = float(x)
            yf = float(y)
            wf = float(w)
            hf = float(h)
        except Exception:
            continue

        if wf <= 0 or hf <= 0:
            continue

        x1 = xf
        y1 = yf
        x2 = xf + wf
        y2 = yf + hf

        bbox: Optional[List[int]]
        if clip_boxes:
            bbox = _clip_bbox_xyxy(x1, y1, x2, y2, width=width, height=height)
        else:
            bbox = round_points([x1, y1, x2, y2])

        if bbox is None or len(bbox) != 4:
            continue

        bx1, by1, bx2, by2 = bbox
        bw = bx2 - bx1
        bh = by2 - by1
        area = bw * bh
        if (
            area < float(min_box_area)
            or bw < float(min_box_dimension)
            or bh < float(min_box_dimension)
        ):
            continue

        desc = _pick_name(names)
        if not desc:
            continue
        if filter_junk_descs and _is_high_conf_junk_desc(desc):
            continue

        out_objs.append(
            {
                "bbox_2d": [bx1, by1, bx2, by2],
                "desc": desc,
                "object_id": obj_dict.get("object_id"),
            }
        )
        kept += 1

    if not out_objs:
        return None, 0, 0

    out_objs = sort_objects_tlbr(out_objs)
    deduped = 0
    if dedupe_objects:
        before = len(out_objs)
        seen: set[tuple[str, int, int, int, int]] = set()
        deduped: List[Dict[str, Any]] = []
        for o in out_objs:
            bb = o.get("bbox_2d")
            desc = o.get("desc")
            if not (
                isinstance(desc, str)
                and isinstance(bb, list)
                and len(bb) == 4
                and all(isinstance(v, int) for v in bb)
            ):
                deduped.append(o)
                continue
            key = (desc, bb[0], bb[1], bb[2], bb[3])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(o)
        out_objs = deduped
        deduped = before - len(out_objs)

    record: Dict[str, Any] = {
        "images": [rel],
        "objects": out_objs,
        "width": int(width),
        "height": int(height),
        "metadata": {
            "dataset": "visual_genome",
            "image_id": int(image_id),
            "url": url,
            "annotation": "objects",
        },
    }
    _finalize_metadata(record["objects"])
    return record, len(out_objs), deduped


def _convert_pair_to_region_record(
    img_meta: Dict[str, Any],
    ann: Dict[str, Any],
    *,
    min_box_area: float,
    min_box_dimension: float,
    clip_boxes: bool,
    dedupe_objects: bool,
    filter_junk_descs: bool,
) -> Tuple[Optional[Dict[str, Any]], int, int]:
    """Return (record, kept_objects, deduped_objects) for region descriptions."""
    image_id = img_meta.get("image_id")
    url = img_meta.get("url")
    width = img_meta.get("width")
    height = img_meta.get("height")

    if not isinstance(image_id, int) or not isinstance(url, str):
        return None, 0
    if not isinstance(width, int) or not isinstance(height, int):
        return None, 0
    if width <= 0 or height <= 0:
        return None, 0

    rel = _image_relpath_from_url(url)
    if rel is None:
        return None, 0

    regions = ann.get("regions")
    if not isinstance(regions, list) or not regions:
        return None, 0

    out_objs: List[Dict[str, Any]] = []
    kept = 0
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_dict: Dict[str, Any] = region  # type: ignore
        x = region_dict.get("x")
        y = region_dict.get("y")
        w = region_dict.get("width")
        h = region_dict.get("height")
        phrase = region_dict.get("phrase")

        if x is None or y is None or w is None or h is None:
            continue

        try:
            xf = float(x)
            yf = float(y)
            wf = float(w)
            hf = float(h)
        except Exception:
            continue

        if wf <= 0 or hf <= 0:
            continue

        x1 = xf
        y1 = yf
        x2 = xf + wf
        y2 = yf + hf

        bbox: Optional[List[int]]
        if clip_boxes:
            bbox = _clip_bbox_xyxy(x1, y1, x2, y2, width=width, height=height)
        else:
            bbox = round_points([x1, y1, x2, y2])

        if bbox is None or len(bbox) != 4:
            continue

        bx1, by1, bx2, by2 = bbox
        bw = bx2 - bx1
        bh = by2 - by1
        area = bw * bh
        if (
            area < float(min_box_area)
            or bw < float(min_box_dimension)
            or bh < float(min_box_dimension)
        ):
            continue

        desc = _sanitize_phrase(phrase)
        if not desc:
            continue
        if filter_junk_descs and _is_high_conf_junk_desc(desc):
            continue

        out_objs.append(
            {
                "bbox_2d": [bx1, by1, bx2, by2],
                "desc": desc,
                "region_id": region_dict.get("region_id"),
            }
        )
        kept += 1

    if not out_objs:
        return None, 0, 0

    out_objs = sort_objects_tlbr(out_objs)
    deduped = 0
    if dedupe_objects:
        before = len(out_objs)
        seen: set[tuple[str, int, int, int, int]] = set()
        deduped: List[Dict[str, Any]] = []
        for o in out_objs:
            bb = o.get("bbox_2d")
            desc = o.get("desc")
            if not (
                isinstance(desc, str)
                and isinstance(bb, list)
                and len(bb) == 4
                and all(isinstance(v, int) for v in bb)
            ):
                deduped.append(o)
                continue
            key = (desc, bb[0], bb[1], bb[2], bb[3])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(o)
        out_objs = deduped
        deduped = before - len(out_objs)

    record: Dict[str, Any] = {
        "images": [rel],
        "objects": out_objs,
        "width": int(width),
        "height": int(height),
        "metadata": {
            "dataset": "visual_genome",
            "image_id": int(image_id),
            "url": url,
            "annotation": "regions",
        },
    }
    _finalize_metadata(record["objects"])
    return record, len(out_objs), deduped


def convert(
    *,
    image_meta_json: Path,
    objects_json: Path,
    out_train_jsonl: Path,
    out_val_jsonl: Path,
    max_samples: Optional[int],
    val_mod: int,
    min_box_area: float,
    min_box_dimension: float,
    clip_boxes: bool,
    dedupe_objects: bool,
    filter_junk_descs: bool,
    stats_path: Optional[Path],
) -> ConvertStats:
    stats = ConvertStats()
    _ensure_dir(out_train_jsonl.parent)

    fout_train = out_train_jsonl.open("w", encoding="utf-8")
    fout_val = out_val_jsonl.open("w", encoding="utf-8")
    try:
        it_meta = _iter_json_array(image_meta_json)
        it_ann = _iter_json_array(objects_json)

        for idx, (img_meta, ann) in enumerate(zip(it_meta, it_ann)):
            if max_samples is not None and idx >= max_samples:
                break
            stats.images_total += 1

            if not isinstance(img_meta, dict) or not isinstance(ann, dict):
                continue
            meta_id = img_meta.get("image_id")
            ann_id = ann.get("image_id")
            if meta_id != ann_id:
                raise ValueError(
                    "Misaligned VG arrays: image_data.json and objects.json are not aligned.\n"
                    f"index={idx} image_data.image_id={meta_id!r} objects.image_id={ann_id!r}"
                )

            record, kept, deduped = _convert_pair_to_record(
                img_meta,  # type: ignore
                ann,  # type: ignore
                min_box_area=min_box_area,
                min_box_dimension=min_box_dimension,
                clip_boxes=clip_boxes,
                dedupe_objects=dedupe_objects,
                filter_junk_descs=filter_junk_descs,
            )
            if record is None:
                url = img_meta.get("url") if isinstance(img_meta, dict) else None
                if isinstance(url, str) and _image_relpath_from_url(url) is None:
                    stats.images_skipped_bad_url += 1
                else:
                    stats.images_skipped_missing_ann += 1
                continue

            stats.objects_kept += kept
            stats.objects_deduped += deduped
            image_id = int(record.get("metadata", {}).get("image_id") or 0)
            if val_mod > 0 and (image_id % val_mod) == 0:
                fout_val.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.images_written_val += 1
            else:
                fout_train.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.images_written_train += 1

    finally:
        fout_train.close()
        fout_val.close()

    if stats_path is not None:
        _ensure_dir(stats_path.parent)
        stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    return stats


def convert_regions(
    *,
    image_meta_json: Path,
    regions_json: Path,
    out_train_jsonl: Path,
    out_val_jsonl: Path,
    max_samples: Optional[int],
    val_mod: int,
    min_box_area: float,
    min_box_dimension: float,
    clip_boxes: bool,
    dedupe_objects: bool,
    filter_junk_descs: bool,
    stats_path: Optional[Path],
) -> ConvertStats:
    stats = ConvertStats()
    _ensure_dir(out_train_jsonl.parent)

    fout_train = out_train_jsonl.open("w", encoding="utf-8")
    fout_val = out_val_jsonl.open("w", encoding="utf-8")
    try:
        it_meta = _iter_json_array(image_meta_json)
        it_ann = _iter_json_array(regions_json)

        for idx, (img_meta, ann) in enumerate(zip(it_meta, it_ann)):
            if max_samples is not None and idx >= max_samples:
                break
            stats.images_total += 1

            if not isinstance(img_meta, dict) or not isinstance(ann, dict):
                continue
            meta_id = img_meta.get("image_id")
            ann_id = ann.get("image_id")
            if meta_id != ann_id:
                raise ValueError(
                    "Misaligned VG arrays: image_data.json and region_descriptions.json are not aligned.\n"
                    f"index={idx} image_data.image_id={meta_id!r} regions.image_id={ann_id!r}"
                )

            record, kept, deduped = _convert_pair_to_region_record(
                img_meta,  # type: ignore
                ann,  # type: ignore
                min_box_area=min_box_area,
                min_box_dimension=min_box_dimension,
                clip_boxes=clip_boxes,
                dedupe_objects=dedupe_objects,
                filter_junk_descs=filter_junk_descs,
            )

            if record is None:
                stats.images_skipped_missing_ann += 1
                continue

            stats.objects_kept += kept
            stats.objects_deduped += deduped
            image_id = int(record.get("metadata", {}).get("image_id") or 0)
            if val_mod > 0 and (image_id % val_mod) == 0:
                fout_val.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.images_written_val += 1
            else:
                fout_train.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.images_written_train += 1

    finally:
        fout_train.close()
        fout_val.close()

    if stats_path is not None:
        _ensure_dir(stats_path.parent)
        stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["objects", "regions"],
        default="objects",
        help="Which VG annotations to convert/download (default: objects)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("public_data/vg"),
        help="Output root under repo (default: public_data/vg)",
    )
    parser.add_argument(
        "--objects-version",
        type=str,
        choices=sorted(VG_OBJECTS_URLS.keys()),
        default="1.2.0",
        help="VG objects annotation version (default: 1.2.0)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download VG zips (annotations + images) into public_data/vg/raw",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/extract raw artifacts but skip JSONL conversion",
    )
    parser.add_argument(
        "--wget-no-proxy",
        action="store_true",
        help="Pass --no-proxy to wget (useful when HF needs a proxy but VG mirrors do not)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip downloading images (annotations only)",
    )
    parser.add_argument(
        "--no-verify-checksums",
        action="store_true",
        help="Skip sha256 checksum verification for downloaded zips (not recommended)",
    )
    parser.add_argument(
        "--no-dedupe-objects",
        action="store_true",
        help="Disable per-image exact deduping of (desc, bbox_2d) pairs",
    )
    parser.add_argument(
        "--no-filter-junk-descs",
        action="store_true",
        help="Disable dropping high-confidence junk desc labels (e.g., 'this')",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Global time budget in seconds for downloads/extraction (useful for smoke tests)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of records written (for testing)",
    )
    parser.add_argument(
        "--val-mod",
        type=int,
        default=5,
        help="Deterministic val split: image_id %% val_mod == 0 (default: 5 -> ~20%%)",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=1.0,
        help="Minimum bbox area in pixels (default: 1.0)",
    )
    parser.add_argument(
        "--min-box-dim",
        type=float,
        default=1.0,
        help="Minimum bbox width/height in pixels (default: 1.0)",
    )
    parser.add_argument(
        "--no-clip-boxes",
        action="store_true",
        help="Do not clip boxes to [0,width-1]/[0,height-1] (not recommended)",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Optional path to write conversion stats JSON",
    )
    args = parser.parse_args()

    if args.download_only and not args.download:
        raise SystemExit("--download-only requires --download")

    out_root: Path = args.output_root
    raw_root = out_root / "raw"
    ann_dir = raw_root / "annotations"
    img_dir = raw_root / "images"
    _ensure_dir(ann_dir)
    _ensure_dir(img_dir)

    start = _now()

    def remaining_timeout() -> int | None:
        if args.max_seconds is None:
            return None
        elapsed = int(_now() - start)
        rem = int(args.max_seconds) - elapsed
        return rem if rem > 0 else 0

    if args.download:
        verify_checksums = not bool(args.no_verify_checksums)
        # 1) Annotation zips
        meta_zip = ann_dir / "image_data.json.zip"
        print(f"Downloading annotations into: {ann_dir}")
        t = remaining_timeout()
        if t == 0:
            print("Time budget exhausted before downloads started.")
            return
        if not _download_then_verify(
            VG_IMAGE_META_URL,
            meta_zip,
            timeout_s=t,
            no_proxy=bool(args.wget_no_proxy),
            verify_checksums=verify_checksums,
        ):
            print("Stopped during image metadata download (timeout or error).")
            return

        t = remaining_timeout()
        if t == 0:
            print("Time budget exhausted before annotation download started.")
            return

        obj_zip = None
        reg_zip = None
        if args.mode == "objects":
            obj_zip = (
                ann_dir / f"objects_v{args.objects_version.replace('.', '_')}.json.zip"
            )
            if not _download_then_verify(
                VG_OBJECTS_URLS[args.objects_version],
                obj_zip,
                timeout_s=t,
                no_proxy=bool(args.wget_no_proxy),
                verify_checksums=verify_checksums,
            ):
                print("Stopped during objects annotation download (timeout or error).")
                return
        else:
            reg_zip = ann_dir / "region_descriptions.json.zip"
            if not _download_then_verify(
                VG_REGION_DESCRIPTIONS_URL,
                reg_zip,
                timeout_s=t,
                no_proxy=bool(args.wget_no_proxy),
                verify_checksums=verify_checksums,
            ):
                print("Stopped during region_descriptions download (timeout or error).")
                return

        # Extract annotations
        t = remaining_timeout()
        if t == 0:
            print("Time budget exhausted before annotation unzip started.")
            return
        if not _unzip(meta_zip, ann_dir, timeout_s=t):
            print("Stopped during image metadata unzip (timeout or error).")
            return
        if args.mode == "objects" and obj_zip is not None:
            t = remaining_timeout()
            if t == 0:
                print("Time budget exhausted before objects unzip started.")
                return
            if not _unzip(obj_zip, ann_dir, timeout_s=t):
                print("Stopped during objects unzip (timeout or error).")
                return
        elif args.mode != "objects" and reg_zip is not None:
            t = remaining_timeout()
            if t == 0:
                print("Time budget exhausted before region_descriptions unzip started.")
                return
            if not _unzip(reg_zip, ann_dir, timeout_s=t):
                print("Stopped during region_descriptions unzip (timeout or error).")
                return

        # 2) Image zips
        if not args.skip_images:
            print(f"Downloading images into: {img_dir}")
            downloaded_images = False
            for fname, url in VG_IMAGE_ZIPS.items():
                t = remaining_timeout()
                if t == 0:
                    print("Time budget exhausted; skipping remaining image downloads.")
                    break
                zip_path = img_dir / fname
                print(f"Downloading: {fname} from {url}")
                ok = _download_then_verify(
                    url,
                    zip_path,
                    timeout_s=t,
                    no_proxy=bool(args.wget_no_proxy),
                    verify_checksums=verify_checksums,
                )
                if not ok:
                    print(
                        f"[warning] {fname} download failed; skipping this image set."
                    )
                    continue

                t = remaining_timeout()
                if t == 0:
                    print("Time budget exhausted; skipping image unzip.")
                    break

                print(f"Extracting: {fname}")
                ok = _unzip(zip_path, img_dir, timeout_s=t)
                if not ok:
                    print(
                        f"[warning] {fname} extraction failed; skipping this image set."
                    )
                    continue

                # Verify extraction was successful by checking for expected directories
                if fname == "images.zip" and not (img_dir / "VG_100K").exists():
                    print(
                        f"[warning] VG_100K directory not found after {fname} extraction"
                    )
                elif fname == "images2.zip" and not (img_dir / "VG_100K_2").exists():
                    print(
                        f"[warning] VG_100K_2 directory not found after {fname} extraction"
                    )
                else:
                    downloaded_images = True
                    print(f"[success] {fname} successfully extracted")

            if not downloaded_images:
                print("[warning] No images were successfully downloaded and extracted.")
                print(
                    "You may need to run the download again or use --skip-images for testing."
                )

    if args.download_only:
        print("Download finished (--download-only). Skipping conversion.")
        return

    # Locate extracted annotation JSONs.
    image_meta_json = _find_one(ann_dir, ["image_data.json"])

    out_train = raw_root / "train.jsonl"
    out_val = raw_root / "val.jsonl"
    dedupe_objects = not bool(args.no_dedupe_objects)
    filter_junk_descs = not bool(args.no_filter_junk_descs)
    if args.mode == "objects":
        objects_json = _find_one(
            ann_dir, ["objects.json", "objects_v1.json", "objects_v1_2.json"]
        )
        stats = convert(
            image_meta_json=image_meta_json,
            objects_json=objects_json,
            out_train_jsonl=out_train,
            out_val_jsonl=out_val,
            max_samples=args.max_samples,
            val_mod=int(args.val_mod),
            min_box_area=float(args.min_box_area),
            min_box_dimension=float(args.min_box_dim),
            clip_boxes=not bool(args.no_clip_boxes),
            dedupe_objects=dedupe_objects,
            filter_junk_descs=filter_junk_descs,
            stats_path=args.stats_json,
        )
    else:
        regions_json = _find_one(
            ann_dir, ["region_descriptions.json", "region_descriptions_v1.json"]
        )
        stats = convert_regions(
            image_meta_json=image_meta_json,
            regions_json=regions_json,
            out_train_jsonl=out_train,
            out_val_jsonl=out_val,
            max_samples=args.max_samples,
            val_mod=int(args.val_mod),
            min_box_area=float(args.min_box_area),
            min_box_dimension=float(args.min_box_dim),
            clip_boxes=not bool(args.no_clip_boxes),
            dedupe_objects=dedupe_objects,
            filter_junk_descs=filter_junk_descs,
            stats_path=args.stats_json,
        )

    print("Conversion finished.")
    print(json.dumps(stats.to_dict(), indent=2))
    print(f"train: {out_train}")
    print(f"val:   {out_val}")


if __name__ == "__main__":
    main()

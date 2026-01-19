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
- For quick smoke tests, pass `--max-seconds 300` to stop downloads early.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from public_data.converters.sorting import sort_objects_tlbr


# URLs are copied from the HF dataset loader `visual_genome.py` (snapshot).
VG_BASE = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset"
VG_IMAGE_META_URL = f"{VG_BASE}/image_data.json.zip"
VG_OBJECTS_URLS = {
    "1.0.0": f"{VG_BASE}/objects_v1.json.zip",
    "1.2.0": f"{VG_BASE}/objects_v1_2.json.zip",
}
VG_IMAGE_ZIPS = {
    # The zip names are historical; the extracted folders are VG_100K and VG_100K_2.
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


def _run(cmd: List[str], *, timeout_s: Optional[int] = None) -> int:
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
    timeout_s: Optional[int],
    no_proxy: bool,
) -> bool:
    """Download a URL to out_path with resume support.

    We intentionally keep partial files on timeout so a future run with
    `--continue` can resume.
    """
    _ensure_dir(out_path.parent)

    cmd = [
        "wget",
        "--continue",
        "--progress=bar:force",
        "--show-progress",
        *(["--no-proxy"] if no_proxy else []),
        url,
        "-O",
        str(out_path),
    ]
    rc = _run(cmd, timeout_s=timeout_s)
    if rc == 124:
        print(f"[timeout] wget exceeded {timeout_s}s: {url}")
        return False
    if rc != 0:
        print(f"[error] wget failed with code {rc}: {url}")
        return False
    if not out_path.exists() or out_path.stat().st_size == 0:
        print(f"[error] download failed: {url}")
        return False
    return True


def _unzip(zip_path: Path, out_dir: Path, *, timeout_s: Optional[int]) -> bool:
    _ensure_dir(out_dir)
    cmd = ["unzip", "-q", str(zip_path), "-d", str(out_dir)]
    rc = _run(cmd, timeout_s=timeout_s)
    if rc == 124:
        print(f"[timeout] unzip exceeded {timeout_s}s: {zip_path}")
        return False
    if rc != 0:
        print(f"[error] unzip failed with code {rc}: {zip_path}")
        return False
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
) -> Optional[List[float]]:
    """Clip bbox into [0,width-1]/[0,height-1] to avoid norm bin 1000."""
    if width <= 1 or height <= 1:
        return None
    max_x = float(width - 1)
    max_y = float(height - 1)
    x1c = max(0.0, min(float(x1), max_x))
    y1c = max(0.0, min(float(y1), max_y))
    x2c = max(0.0, min(float(x2), max_x))
    y2c = max(0.0, min(float(y2), max_y))
    if x2c <= x1c or y2c <= y1c:
        return None
    return [x1c, y1c, x2c, y2c]


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


def _pick_name(names: Any) -> Optional[str]:
    if isinstance(names, list):
        for n in names:
            out = _sanitize_desc(n)
            if out:
                return out
        return None
    return _sanitize_desc(names)


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

    def to_dict(self) -> Dict[str, int]:
        return {
            "images_total": self.images_total,
            "images_written_train": self.images_written_train,
            "images_written_val": self.images_written_val,
            "images_skipped_missing_ann": self.images_skipped_missing_ann,
            "images_skipped_bad_url": self.images_skipped_bad_url,
            "objects_kept": self.objects_kept,
        }


def _convert_pair_to_record(
    img_meta: Mapping[str, Any],
    ann: Mapping[str, Any],
    *,
    min_box_area: float,
    min_box_dimension: float,
    clip_boxes: bool,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Return (record, kept_objects)."""
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
        x = obj.get("x")
        y = obj.get("y")
        w = obj.get("w")
        h = obj.get("h")
        names = obj.get("names")

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

        bbox: Optional[List[float]]
        if clip_boxes:
            bbox = _clip_bbox_xyxy(x1, y1, x2, y2, width=width, height=height)
        else:
            bbox = [x1, y1, x2, y2]

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

        out_objs.append({"bbox_2d": [bx1, by1, bx2, by2], "desc": desc})
        kept += 1

    if not out_objs:
        return None, 0

    out_objs = sort_objects_tlbr(out_objs)

    record: Dict[str, Any] = {
        "images": [rel],
        "objects": out_objs,
        "width": int(width),
        "height": int(height),
        "metadata": {
            "dataset": "visual_genome",
            "image_id": int(image_id),
            "url": url,
        },
    }
    return record, kept


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

            record, kept = _convert_pair_to_record(
                img_meta,
                ann,
                min_box_area=min_box_area,
                min_box_dimension=min_box_dimension,
                clip_boxes=clip_boxes,
            )
            if record is None:
                url = img_meta.get("url") if isinstance(img_meta, dict) else None
                if isinstance(url, str) and _image_relpath_from_url(url) is None:
                    stats.images_skipped_bad_url += 1
                else:
                    stats.images_skipped_missing_ann += 1
                continue

            stats.objects_kept += kept
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
        stats_path.write_text(
            json.dumps(stats.to_dict(), indent=2), encoding="utf-8"
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=100,
        help="Deterministic val split: image_id % val_mod == 0 (default: 100 -> ~1%)",
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

    out_root: Path = args.output_root
    raw_root = out_root / "raw"
    ann_dir = raw_root / "annotations"
    img_dir = raw_root / "images"
    _ensure_dir(ann_dir)
    _ensure_dir(img_dir)

    start = _now()

    def remaining_timeout() -> Optional[int]:
        if args.max_seconds is None:
            return None
        elapsed = int(_now() - start)
        rem = int(args.max_seconds) - elapsed
        return rem if rem > 0 else 0

    if args.download:
        # 1) Annotation zips
        meta_zip = ann_dir / "image_data.json.zip"
        obj_zip = ann_dir / f"objects_v{args.objects_version.replace('.', '_')}.json.zip"
        print(f"Downloading annotations into: {ann_dir}")
        t = remaining_timeout()
        if t == 0:
            print("Time budget exhausted before downloads started.")
            return
        if not _download_with_wget(
            VG_IMAGE_META_URL, meta_zip, timeout_s=t, no_proxy=bool(args.wget_no_proxy)
        ):
            print("Stopped during image metadata download (timeout or error).")
            return
        t = remaining_timeout()
        if t == 0:
            print("Time budget exhausted before objects download started.")
            return
        if not _download_with_wget(
            VG_OBJECTS_URLS[args.objects_version],
            obj_zip,
            timeout_s=t,
            no_proxy=bool(args.wget_no_proxy),
        ):
            print("Stopped during objects annotation download (timeout or error).")
            return

        # Extract annotations
        t = remaining_timeout()
        if t == 0:
            print("Time budget exhausted before annotation unzip started.")
            return
        if not _unzip(meta_zip, ann_dir, timeout_s=t):
            print("Stopped during image metadata unzip (timeout or error).")
            return
        t = remaining_timeout()
        if t == 0:
            print("Time budget exhausted before objects unzip started.")
            return
        if not _unzip(obj_zip, ann_dir, timeout_s=t):
            print("Stopped during objects unzip (timeout or error).")
            return

        # 2) Image zips
        if not args.skip_images:
            print(f"Downloading images into: {img_dir}")
            for fname, url in VG_IMAGE_ZIPS.items():
                t = remaining_timeout()
                if t == 0:
                    print("Time budget exhausted; skipping remaining image downloads.")
                    break
                zip_path = img_dir / fname
                ok = _download_with_wget(
                    url, zip_path, timeout_s=t, no_proxy=bool(args.wget_no_proxy)
                )
                if not ok:
                    print("Image download incomplete (timeout or error); continuing to conversion.")
                    break
                t = remaining_timeout()
                if t == 0:
                    print("Time budget exhausted; skipping image unzip.")
                    break
                ok = _unzip(zip_path, img_dir, timeout_s=t)
                if not ok:
                    print("Image unzip incomplete (timeout or error); continuing to conversion.")
                    break

    # Locate extracted annotation JSONs.
    image_meta_json = _find_one(ann_dir, ["image_data.json"])
    objects_json = _find_one(ann_dir, ["objects.json", "objects_v1.json", "objects_v1_2.json"])

    out_train = raw_root / "train.jsonl"
    out_val = raw_root / "val.jsonl"
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
        stats_path=args.stats_json,
    )

    print("Conversion finished.")
    print(json.dumps(stats.to_dict(), indent=2))
    print(f"train: {out_train}")
    print(f"val:   {out_val}")


if __name__ == "__main__":
    main()

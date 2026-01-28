#!/usr/bin/env python3
"""End-to-end smoke test for the Visual Genome (VG) public-data pipeline.

Run:
  PYTHONPATH=. conda run -n ms python public_data/vg/smoke_test.py

What this does (no network required):
  1) Writes a tiny synthetic VG-style annotation set under temp/
  2) Runs `public_data/scripts/prepare_visual_genome.py` (convert only)
  3) Runs shared preprocessing: rescale -> coord tokens
  4) Validates all produced JSONLs with `public_data/scripts/validate_jsonl.py`

This is intentionally tiny and deterministic so it can run in < ~10 seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"cmd: {cmd}\n"
            f"exit: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_dummy_jpeg(path: Path, *, width: int, height: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (int(width), int(height)), (123, 20, 210))
    img.save(path, format="JPEG", quality=85)


def main() -> None:
    tmp_root = REPO_ROOT / "temp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Keep outputs on disk for post-run inspection (under `temp/`), since the
    # main purpose here is debuggability + reproducibility.
    work_dir = Path(tempfile.mkdtemp(dir=str(tmp_root), prefix="vg_smoke_"))
    dataset_root = work_dir / "public_data" / "vg_smoke"

    # --- 1) Minimal VG-style raw annotations (arrays are aligned by index) ---
    ann_dir = dataset_root / "raw" / "annotations"
    img_dir = dataset_root / "raw" / "images"

    image_meta = [
        {
            "image_id": 1,
            "url": "https://cs.stanford.edu/people/rak248/VG_100K/1.jpg",
            "width": 800,
            "height": 600,
        },
        {
            "image_id": 2,
            "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2.jpg",
            "width": 640,
            "height": 480,
        },
    ]
    objects = [
        {
            "image_id": 1,
            "objects": [
                # Duplicate entries (same bbox + desc) should be deduped by default.
                {"object_id": 101, "x": 10, "y": 20, "w": 200, "h": 150, "names": ["person"]},
                {"object_id": 102, "x": 10, "y": 20, "w": 200, "h": 150, "names": ["person"]},
                # Name sanitation: keep the first segment (commas/slashes/etc).
                {"object_id": 103, "x": 0, "y": 30, "w": 45, "h": 50, "names": ["dog/canine"]},
            ],
        },
        {
            "image_id": 2,
            "objects": [
                {"object_id": 201, "x": 5, "y": 6, "w": 100, "h": 80, "names": ["cat, kitten"]},
            ],
        },
    ]

    _write_json(ann_dir / "image_data.json", image_meta)
    _write_json(ann_dir / "objects.json", objects)

    # --- 2) Dummy images matching the referenced relative paths ---
    _write_dummy_jpeg(img_dir / "VG_100K" / "1.jpg", width=800, height=600)
    _write_dummy_jpeg(img_dir / "VG_100K_2" / "2.jpg", width=640, height=480)

    # --- 3) Convert (raw/train.jsonl + raw/val.jsonl) ---
    _run(
        [
            sys.executable,
            "public_data/scripts/prepare_visual_genome.py",
            "--output-root",
            str(dataset_root),
            "--mode",
            "objects",
            "--max-samples",
            "2",
            "--val-mod",
            "2",  # image_id 2 goes to val; image_id 1 goes to train
            "--stats-json",
            str(work_dir / "convert_stats.json"),
        ],
        cwd=REPO_ROOT,
    )

    raw_train = dataset_root / "raw" / "train.jsonl"
    raw_val = dataset_root / "raw" / "val.jsonl"
    if not raw_train.is_file() or not raw_val.is_file():
        raise RuntimeError("Expected raw/train.jsonl and raw/val.jsonl to be created")

    # --- 4) Validate raw JSONL (with image existence check enabled) ---
    _run([sys.executable, "public_data/scripts/validate_jsonl.py", str(raw_train)], cwd=REPO_ROOT)
    _run([sys.executable, "public_data/scripts/validate_jsonl.py", str(raw_val)], cwd=REPO_ROOT)

    # --- 5) Rescale (shared) ---
    preset_dir = dataset_root / "rescale_32_768_bbox_smoke"
    preset_train = preset_dir / "train.jsonl"
    preset_val = preset_dir / "val.jsonl"
    _run(
        [
            sys.executable,
            "public_data/scripts/rescale_jsonl.py",
            "--input-jsonl",
            str(raw_train),
            "--output-jsonl",
            str(preset_train),
            "--output-images",
            str(preset_dir),
            "--image-factor",
            "32",
            "--max-pixels",
            str(32 * 32 * 768),
            "--min-pixels",
            str(32 * 32 * 4),
            "--num-workers",
            "1",
            "--relative-images",
        ],
        cwd=REPO_ROOT,
    )
    _run(
        [
            sys.executable,
            "public_data/scripts/rescale_jsonl.py",
            "--input-jsonl",
            str(raw_val),
            "--output-jsonl",
            str(preset_val),
            "--output-images",
            str(preset_dir),
            "--image-factor",
            "32",
            "--max-pixels",
            str(32 * 32 * 768),
            "--min-pixels",
            str(32 * 32 * 4),
            "--num-workers",
            "1",
            "--relative-images",
        ],
        cwd=REPO_ROOT,
    )

    # --- 6) Coord tokens (shared) ---
    preset_train_coord = preset_dir / "train.coord.jsonl"
    preset_val_coord = preset_dir / "val.coord.jsonl"
    _run(
        [
            sys.executable,
            "public_data/scripts/convert_to_coord_tokens.py",
            "--input",
            str(preset_train),
            "--output-tokens",
            str(preset_train_coord),
            "--keys",
            "bbox_2d",
            "poly",
        ],
        cwd=REPO_ROOT,
    )
    _run(
        [
            sys.executable,
            "public_data/scripts/convert_to_coord_tokens.py",
            "--input",
            str(preset_val),
            "--output-tokens",
            str(preset_val_coord),
            "--keys",
            "bbox_2d",
            "poly",
        ],
        cwd=REPO_ROOT,
    )

    # --- 7) Validate preset JSONLs (incl. coord tokens) ---
    _run([sys.executable, "public_data/scripts/validate_jsonl.py", str(preset_train)], cwd=REPO_ROOT)
    _run([sys.executable, "public_data/scripts/validate_jsonl.py", str(preset_val)], cwd=REPO_ROOT)
    _run([sys.executable, "public_data/scripts/validate_jsonl.py", str(preset_train_coord)], cwd=REPO_ROOT)
    _run([sys.executable, "public_data/scripts/validate_jsonl.py", str(preset_val_coord)], cwd=REPO_ROOT)

    print(f"[ok] VG smoke test outputs under: {dataset_root}")


if __name__ == "__main__":
    main()

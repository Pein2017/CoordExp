from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
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
    img = Image.new("RGB", (int(width), int(height)), (1, 2, 3))
    img.save(path, format="JPEG", quality=85)


def test_vg_pipeline_smoke(tmp_path: Path) -> None:
    # Keep everything within pytest's tmp_path.
    dataset_root = tmp_path / "public_data" / "vg_smoke"
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
                # Duplicate entries should be deduped by default.
                {"object_id": 101, "x": 10, "y": 20, "w": 200, "h": 150, "names": ["person"]},
                {"object_id": 102, "x": 10, "y": 20, "w": 200, "h": 150, "names": ["person"]},
                {"object_id": 103, "x": 0, "y": 30, "w": 45, "h": 50, "names": ["dog/canine"]},
                # High-confidence junk label should be dropped by default.
                {"object_id": 104, "x": 1, "y": 2, "w": 10, "h": 10, "names": ["this"]},
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
    _write_dummy_jpeg(img_dir / "VG_100K" / "1.jpg", width=800, height=600)
    _write_dummy_jpeg(img_dir / "VG_100K_2" / "2.jpg", width=640, height=480)

    # Convert -> raw JSONL.
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
            "2",
        ]
    )

    raw_train = dataset_root / "raw" / "train.jsonl"
    raw_val = dataset_root / "raw" / "val.jsonl"
    assert raw_train.is_file()
    assert raw_val.is_file()

    # Dedup + desc sanitation sanity.
    train_row = json.loads(raw_train.read_text(encoding="utf-8").splitlines()[0])
    assert train_row["images"] == ["images/VG_100K/1.jpg"]
    descs = [o["desc"] for o in train_row["objects"]]
    # "dog/canine" -> "dog"
    assert "dog" in descs
    # Duplicate "person" bbox should be dropped (keep one).
    assert descs.count("person") == 1
    # High-confidence junk label should be filtered.
    assert "this" not in descs

    # Validate raw JSONL.
    _run([sys.executable, "public_data/scripts/validate_jsonl.py", str(raw_train)])
    _run([sys.executable, "public_data/scripts/validate_jsonl.py", str(raw_val)])

    # Rescale -> preset JSONL + images.
    preset_dir = dataset_root / "rescale_32_768_bbox_smoke"
    preset_train = preset_dir / "train.jsonl"
    preset_val = preset_dir / "val.jsonl"
    for in_jsonl, out_jsonl in [(raw_train, preset_train), (raw_val, preset_val)]:
        _run(
            [
                sys.executable,
                "public_data/scripts/rescale_jsonl.py",
                "--input-jsonl",
                str(in_jsonl),
                "--output-jsonl",
                str(out_jsonl),
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
            ]
        )

    assert preset_train.is_file()
    assert preset_val.is_file()
    assert (preset_dir / "images").is_dir()

    # Coord tokens + validation.
    preset_train_coord = preset_dir / "train.coord.jsonl"
    preset_val_coord = preset_dir / "val.coord.jsonl"
    for in_jsonl, out_jsonl in [(preset_train, preset_train_coord), (preset_val, preset_val_coord)]:
        _run(
            [
                sys.executable,
                "public_data/scripts/convert_to_coord_tokens.py",
                "--input",
                str(in_jsonl),
                "--output-tokens",
                str(out_jsonl),
                "--keys",
                "bbox_2d",
                "poly",
            ]
        )
        _run([sys.executable, "public_data/scripts/validate_jsonl.py", str(out_jsonl)])

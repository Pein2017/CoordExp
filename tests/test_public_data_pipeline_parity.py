from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from public_data.pipeline import PipelineConfig, PipelinePlanner

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise AssertionError(
            "Command failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"exit: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )


def _write_image(path: Path, width: int = 160, height: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(30, 80, 120)).save(path, format="JPEG")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _dataset_rows(split: str) -> list[dict]:
    return [
        {
            "images": [f"images/{split}2017/000000000001.jpg"],
            "objects": [
                {"bbox_2d": [10, 10, 80, 60], "desc": "person"},
                {"bbox_2d": [20, 20, 140, 110], "desc": "car"},
            ],
            "width": 160,
            "height": 120,
        },
        {
            "images": [f"images/{split}2017/000000000002.jpg"],
            "objects": [{"bbox_2d": [5, 5, 20, 30], "desc": "cat"}],
            "width": 160,
            "height": 120,
        },
    ]


def _prepare_raw_dataset(dataset_dir: Path) -> Path:
    raw_dir = dataset_dir / "raw"
    train_rows = _dataset_rows("train")
    val_rows = _dataset_rows("val")
    _write_jsonl(raw_dir / "train.jsonl", train_rows)
    _write_jsonl(raw_dir / "val.jsonl", val_rows)

    for split, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            _write_image(raw_dir / row["images"][0], width=row["width"], height=row["height"])
    return raw_dir


def _run_legacy_flow(raw_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        _run(
            [
                sys.executable,
                "public_data/scripts/rescale_jsonl.py",
                "--input-jsonl",
                str(raw_dir / f"{split}.jsonl"),
                "--output-jsonl",
                str(output_dir / f"{split}.jsonl"),
                "--output-images",
                str(output_dir),
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
        _run(
            [
                sys.executable,
                "public_data/scripts/convert_to_coord_tokens.py",
                "--input",
                str(output_dir / f"{split}.jsonl"),
                "--output-norm",
                str(output_dir / f"{split}.norm.jsonl"),
                "--output-tokens",
                str(output_dir / f"{split}.coord.jsonl"),
                "--keys",
                "bbox_2d",
                "poly",
            ]
        )


@pytest.mark.parametrize("dataset_id", ["coco", "lvis", "vg"])
def test_unified_pipeline_matches_legacy_on_synthetic_slices(tmp_path: Path, dataset_id: str) -> None:
    dataset_dir = tmp_path / "public_data" / dataset_id
    raw_dir = _prepare_raw_dataset(dataset_dir)

    legacy_dir = dataset_dir / "legacy_out"
    _run_legacy_flow(raw_dir=raw_dir, output_dir=legacy_dir)

    planner = PipelinePlanner()
    cfg = PipelineConfig(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="unified_out",
        num_workers=1,
        run_validation_stage=False,
    )
    unified = planner.run(config=cfg, mode="full")

    for split in ("train", "val"):
        unified_paths = unified.split_artifacts[split]
        expected_files = {
            "raw": legacy_dir / f"{split}.jsonl",
            "norm": legacy_dir / f"{split}.norm.jsonl",
            "coord": legacy_dir / f"{split}.coord.jsonl",
        }
        for key, legacy_path in expected_files.items():
            unified_path = getattr(unified_paths, key)
            assert _read_jsonl(unified_path) == _read_jsonl(legacy_path)

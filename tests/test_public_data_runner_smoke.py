from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_runner(*args: str) -> None:
    proc = subprocess.run(
        ["bash", "public_data/run.sh", *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "public_data/run.sh failed\n"
            f"args: {args}\n"
            f"exit: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )


def _write_dummy_images_for_jsonl(jsonl_path: Path) -> None:
    # PIL is available in the ms conda env used by this repo.
    from PIL import Image

    base_dir = jsonl_path.parent
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            images = row.get("images") or []
            assert isinstance(images, list) and images, "Expected non-empty images list"
            assert len(images) == 1, "Expected exactly 1 image per row for smoke data"
            rel = Path(str(images[0]))
            width = int(row["width"])
            height = int(row["height"])
            out_path = base_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (width, height), (120, 30, 200))
            img.save(out_path, format="JPEG")


def _write_tiny_lvis_jsonl(out_jsonl: Path) -> None:
    row = {
        "images": ["images/000000000001.jpg"],
        "objects": [
            {"bbox_2d": [10.0, 20.0, 100.0, 120.0], "desc": "cat"},
            {
                "poly": [0.0, 0.0, 50.0, 0.0, 50.0, 50.0, 0.0, 50.0],
                "poly_points": 4,
                "desc": "box",
            },
        ],
        "width": 200,
        "height": 150,
    }
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_public_data_runner_smoke() -> None:
    # Use unique dataset ids to avoid clobbering real data under public_data/.
    vg_ds = ROOT / "public_data" / "smoke_vg"
    lvis_ds = ROOT / "public_data" / "smoke_lvis"

    # Clean up from previous failed runs if needed.
    for p in [vg_ds, lvis_ds]:
        if p.exists():
            shutil.rmtree(p)

    try:
        # --- VG-like smoke dataset (fully synthetic; no bundled VG JSONL required) ---
        vg_raw = vg_ds / "raw"
        vg_raw.mkdir(parents=True, exist_ok=True)
        _write_tiny_lvis_jsonl(vg_raw / "train.jsonl")
        _write_tiny_lvis_jsonl(vg_raw / "val.jsonl")
        _write_dummy_images_for_jsonl(vg_raw / "train.jsonl")
        _write_dummy_images_for_jsonl(vg_raw / "val.jsonl")

        vg_preset = "rescale_32_768_bbox_smoke"
        _run_runner("smoke_vg", "help")
        _run_runner("smoke_vg", "rescale", "--preset", vg_preset, "--", "--num-workers", "1")
        _run_runner("smoke_vg", "coord", "--preset", vg_preset)

        assert (vg_ds / vg_preset / "train.jsonl").is_file()
        assert (vg_ds / vg_preset / "train.coord.jsonl").is_file()
        assert (vg_ds / vg_preset / "images").is_dir()

        # Validate raw + preset + coord (also runs inspect_chat_template.py on train.coord.jsonl).
        _run_runner("smoke_vg", "validate", "--preset", vg_preset, "--skip-image-check")

        # --- Tiny LVIS smoke dataset (no internet) ---
        lvis_raw = lvis_ds / "raw"
        lvis_raw.mkdir(parents=True, exist_ok=True)
        _write_tiny_lvis_jsonl(lvis_raw / "train.jsonl")
        _write_dummy_images_for_jsonl(lvis_raw / "train.jsonl")

        lvis_preset = "rescale_32_768_bbox_smoke"
        _run_runner("smoke_lvis", "rescale", "--preset", lvis_preset)
        _run_runner("smoke_lvis", "coord", "--preset", lvis_preset)

        assert (lvis_ds / lvis_preset / "train.jsonl").is_file()
        assert (lvis_ds / lvis_preset / "train.coord.jsonl").is_file()
        assert (lvis_ds / lvis_preset / "images").is_dir()
    finally:
        for p in [vg_ds, lvis_ds]:
            if p.exists():
                shutil.rmtree(p)

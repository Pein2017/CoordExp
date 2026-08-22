from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from public_data.pipeline import PipelineConfig, PipelinePlanner
from public_data.scripts.validate_jsonl import JSONLValidator


@pytest.mark.parametrize("dataset_id", ["coco", "lvis"])
def test_tiny_coco_and_lvis_transform_and_validate_in_tempdir(tmp_path: Path, dataset_id: str) -> None:
    dataset_dir = tmp_path / "public_data" / dataset_id
    raw = dataset_dir / "raw"
    image = raw / "images" / "train" / "one.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (80, 64), color=(1, 2, 3)).save(image)
    row = {
        "images": ["images/train/one.jpg"], "width": 80, "height": 64,
        "objects": [{"bbox_2d": [8, 6, 60, 50], "desc": "object"}],
    }
    (raw / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = PipelineConfig(
        dataset_id=dataset_id, dataset_dir=dataset_dir, raw_dir=raw,
        preset="tiny", image_factor=32, max_pixels=32 * 32 * 16,
        min_pixels=32 * 32 * 4, num_workers=1, relative_images=True,
        assume_normalized=False, compact_json=True, skip_image_check=False,
        run_validation_stage=True,
    )
    result = PipelinePlanner().run(config=config, mode="full", validate_raw=True, validate_preset=True)
    coord = result.preset_dir / "train.coord.jsonl"
    assert coord.is_file() and coord.stat().st_size > 0
    assert JSONLValidator(check_images=True).validate_file(str(coord))

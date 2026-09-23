from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from probes.training_set_completion.visual_detail_dependence.runtime import (
    HIGH_MAX_PIXELS,
    _fit_size,
    prepare_manifest,
)


def test_fit_size_is_grid_aligned_and_bounded() -> None:
    width, height = _fit_size(1920, 1080, max_pixels=HIGH_MAX_PIXELS, factor=32)
    assert width % 32 == height % 32 == 0
    assert width * height <= HIGH_MAX_PIXELS
    assert width <= 1920 and height <= 1080


def test_prepare_emits_high_and_same_grid_control(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.png"
    original = tmp_path / "original.png"
    Image.new("RGB", (640, 352), (1, 2, 3)).save(baseline)
    Image.new("RGB", (1920, 1056), (4, 5, 6)).save(original)
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"images": [{
        "image_id": "x",
        "image_path": str(baseline),
        "original_image_path": str(original),
    }]}))
    manifest = prepare_manifest(admission, tmp_path / "out")
    entry = manifest["entries"][0]
    assert entry["status"] == "eligible"
    assert entry["settings"]["high_detail"]["grid"] == entry["settings"]["baseline_upsampled"]["grid"]
    assert entry["settings"]["high_detail"]["pixel_sha256"] != entry["settings"]["baseline_upsampled"]["pixel_sha256"]

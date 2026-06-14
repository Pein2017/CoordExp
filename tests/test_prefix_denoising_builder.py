from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Mapping

from src.config.schema import PrefixDenoisingConfig
from src.detection.prefix_denoising.builder import (
    build_hybrid_prefix_denoising_sample,
)


_COORD_RE = re.compile(r"<\|coord_(\d+)\|>")


class FakeTokenizer:
    def encode(self, text: str, *args: object, **kwargs: object) -> list[int]:
        del args, kwargs
        if match := _COORD_RE.fullmatch(text):
            return [int(match.group(1))]
        ids: list[int] = []
        cursor = 0
        for match in _COORD_RE.finditer(text):
            ids.extend(10_000 + ord(ch) for ch in text[cursor : match.start()])
            ids.append(int(match.group(1)))
            cursor = match.end()
        ids.extend(10_000 + ord(ch) for ch in text[cursor:])
        return ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.encode(token)[0]


class FakeTemplate:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def encode(self, payload: Mapping[str, Any], *args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        content = str(payload["messages"][-1]["content"])
        input_ids = self.tokenizer.encode(content)
        return {
            "input_ids": list(input_ids),
            "labels": list(input_ids),
            "attention_mask": [1 for _ in input_ids],
        }


def _row(*, objects: list[dict[str, object]] | None = None) -> dict[str, object]:
    if objects is None:
        objects = [
            {
                "desc": "red box",
                "bbox_2d": [100, 100, 200, 220],
                "category_id": 1,
                "category_name": "red box",
                "coco_ann_id": 11,
                "object_id": "red-1",
            },
            {
                "desc": "blue box",
                "bbox_2d": [300, 320, 420, 470],
                "category_id": 2,
                "category_name": "blue box",
                "coco_ann_id": 22,
                "object_id": "blue-1",
            },
        ]
    return {
        "images": ["dummy.jpg"],
        "objects": objects,
        "width": 1000,
        "height": 1000,
        "image_id": 123,
        "file_name": "dummy.jpg",
        "metadata": {"source": "unit", "split": "train"},
    }


def _touch_image(root: Path) -> None:
    (root / "dummy.jpg").write_bytes(b"not-a-real-image")


def test_hybrid_builder_emits_two_segments_and_clean_labels(tmp_path: Path) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping(
        {
            "enabled": True,
            "current_object_kl": {"weight": 0.0, "window_radius": 8, "num_objects_per_image": 1},
        }
    )

    sample = build_hybrid_prefix_denoising_sample(
        _row(),
        base_sample_id="unit-0",
        image_root=tmp_path,
        swift_template=FakeTemplate(),
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=0,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is True
    assert sample.clean_full is not None
    assert sample.noisy_full is not None
    assert sample.clean_full.branch_id == "clean_full"
    assert sample.noisy_full.branch_id == "noisy_full"
    assert sample.clean_full.labels == sample.noisy_full.labels
    assert len(sample.clean_full.input_ids) == len(sample.noisy_full.input_ids)
    assert sample.kl_sites == ()


def test_hybrid_builder_builds_kl_sites_when_weight_positive(tmp_path: Path) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping(
        {
            "enabled": True,
            "current_object_kl": {
                "weight": 0.05,
                "window_radius": 8,
                "num_objects_per_image": 1,
            },
        }
    )

    sample = build_hybrid_prefix_denoising_sample(
        _row(),
        base_sample_id="unit-0",
        image_root=tmp_path,
        swift_template=FakeTemplate(),
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=2,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is True
    assert len(sample.kl_sites) == 4
    assert tuple(site.coord_slot for site in sample.kl_sites) == ("x1", "y1", "x2", "y2")
    for site in sample.kl_sites:
        assert site.clean_gt_bin in site.support_bins


def test_hybrid_builder_excludes_zero_object_rows_with_counter_reason(tmp_path: Path) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping({"enabled": True})

    sample = build_hybrid_prefix_denoising_sample(
        _row(objects=[]),
        base_sample_id="unit-empty",
        image_root=tmp_path,
        swift_template=FakeTemplate(),
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=0,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is False
    assert sample.skip_reason == "zero_object_hybrid_sample"
    assert sample.total_length == 0

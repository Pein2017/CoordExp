from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from public_data.scripts.build_coco_length_budget_artifacts import (
    CocoLengthBudgetBuilder,
    CoordTripletConverter,
    TokenBudgetBreakdown,
    _coord_record_to_norm,
)


class _FakeEstimator:
    def __init__(self, lengths_by_image_id: Mapping[int, int]) -> None:
        self._lengths_by_image_id = dict(lengths_by_image_id)

    def measure(self, record: Mapping[str, Any]) -> TokenBudgetBreakdown:
        object_count = len(record.get("objects") or [])
        total = int(self._lengths_by_image_id[int(record["image_id"])])
        return TokenBudgetBreakdown(
            total_tokens=total,
            text_tokens_without_image_placeholders=total - 64,
            image_patch_tokens=64,
            image_placeholders=1,
            assistant_tokens=object_count * 6,
            object_count=object_count,
        )


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(1, 2, 3)).save(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_coco_length_budget_builder_filters_aligned_triplets(tmp_path: Path) -> None:
    source_root = tmp_path / "rescale_32_1024_bbox"
    output_root = tmp_path / "rescale_32_1024_bbox_len12000"
    source_jsonl = source_root / "train.jsonl"
    rows = [
        {
            "images": ["images/train2017/000000000001.jpg"],
            "objects": [
                {"bbox_2d": [5, 5, 20, 20], "desc": "cat"},
                {"bbox_2d": [30, 30, 50, 50], "desc": "dog"},
            ],
            "width": 64,
            "height": 64,
            "image_id": 1,
            "file_name": "images/train2017/000000000001.jpg",
            "metadata": {"source": "unit", "split": "train"},
        },
        {
            "images": ["images/train2017/000000000002.jpg"],
            "objects": [{"bbox_2d": [1, 1, 60, 60], "desc": "bus"}],
            "width": 64,
            "height": 64,
            "image_id": 2,
            "file_name": "images/train2017/000000000002.jpg",
            "metadata": {"source": "unit", "split": "train"},
        },
    ]
    _write_jsonl(source_jsonl, rows)
    for row in rows:
        _write_image(source_root / row["images"][0])

    builder = CocoLengthBudgetBuilder(
        estimator=_FakeEstimator({1: 11999, 2: 12001}),
        converter=CoordTripletConverter(),
        max_total_tokens=12000,
    )

    stats = builder.build_split(
        source_jsonl=source_jsonl,
        shared_image_root=source_root,
        output_root=output_root,
        split="train",
    )

    assert stats.records_seen == 2
    assert stats.records_written == 1
    assert stats.records_dropped == 1
    assert not (output_root / "images").exists()
    assert not (output_root / rows[1]["images"][0]).exists()

    pixel_rows = _read_jsonl(output_root / "train.jsonl")
    norm_rows = _read_jsonl(output_root / "train.norm.jsonl")
    coord_rows = _read_jsonl(output_root / "train.coord.jsonl")
    assert [row["image_id"] for row in pixel_rows] == [1]
    assert [row["image_id"] for row in norm_rows] == [1]
    assert [row["image_id"] for row in coord_rows] == [1]
    assert pixel_rows[0]["images"] == [
        "../rescale_32_1024_bbox/images/train2017/000000000001.jpg"
    ]
    assert isinstance(norm_rows[0]["objects"][0]["bbox_2d"][0], int)
    assert str(coord_rows[0]["objects"][0]["bbox_2d"][0]).startswith("<|coord_")


def test_coord_record_to_norm_preserves_proxy_metadata() -> None:
    record = {
        "images": ["../base/images/train2017/000000000001.jpg"],
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                ],
                "desc": "clock",
                "proxy_source": "lvis",
                "lvis_ann_id": 99,
            }
        ],
        "width": 64,
        "height": 64,
        "image_id": 1,
        "file_name": "images/train2017/000000000001.jpg",
        "metadata": {"source": "unit", "split": "train"},
    }

    norm = _coord_record_to_norm(record)

    assert norm["objects"][0]["bbox_2d"] == [1, 2, 3, 4]
    assert norm["objects"][0]["proxy_source"] == "lvis"
    assert norm["objects"][0]["lvis_ann_id"] == 99

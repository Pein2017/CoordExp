from __future__ import annotations

import json
from pathlib import Path

from src.analysis.coord_family_text_subset import (
    convert_norm_row_to_text_pixel,
    materialize_text_pixel_subset,
)


def test_convert_norm_row_to_text_pixel_denormalizes_bbox_values() -> None:
    row = {
        "image_id": 1,
        "width": 1248,
        "height": 832,
        "objects": [
            {"desc": "kite", "bbox_2d": [389, 110, 529, 318]},
            {"desc": "person", "bbox_2d": [632, 379, 826, 949]},
        ],
    }

    converted = convert_norm_row_to_text_pixel(row)

    assert converted["objects"][0]["bbox_2d"] == [486, 92, 660, 265]
    assert converted["objects"][1]["bbox_2d"] == [789, 315, 1031, 789]
    assert row["objects"][0]["bbox_2d"] == [389, 110, 529, 318]


def test_materialize_text_pixel_subset_writes_jsonl_and_meta(tmp_path: Path) -> None:
    src = tmp_path / "sampled.norm.jsonl"
    dst = tmp_path / "sampled.text_pixel.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image_id": 1,
                        "width": 1248,
                        "height": 832,
                        "objects": [{"desc": "kite", "bbox_2d": [389, 110, 529, 318]}],
                    }
                ),
                json.dumps(
                    {
                        "image_id": 2,
                        "width": 864,
                        "height": 1152,
                        "objects": [{"desc": "person", "bbox_2d": [253, 132, 666, 880]}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = materialize_text_pixel_subset(src, dst)

    assert summary["row_count"] == 2
    rows = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["objects"][0]["bbox_2d"] == [486, 92, 660, 265]
    assert rows[1]["objects"][0]["bbox_2d"] == [219, 152, 575, 1014]
    meta = json.loads(dst.with_suffix(".jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["surface"] == "text_pixel_from_norm1000"

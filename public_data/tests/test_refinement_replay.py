from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from public_data.refinement_replay import ReplayError, replay_refinement


def _working_row(*, image_id: int, ann_id: int) -> dict[str, object]:
    locator = f"images/val2017/{image_id:012d}.jpg"
    return {
        "file_name": locator,
        "height": 1056,
        "image_id": image_id,
        "images": [locator],
        "metadata": {"source": "coco2017", "split": "val"},
        "objects": [
            {
                "bbox_2d": [10, 20, 30, 40],
                "desc": "person",
                "category_id": 1,
                "category_name": "person",
                "coco_ann_id": ann_id,
                "metadata": {"origin": "human"},
            },
            {
                "bbox_2d": [50, 60, 90, 100],
                "desc": "tie",
                "category_id": 32,
                "category_name": "tie",
                "coco_ann_id": 99,
            },
        ],
        "width": 960,
    }


def _prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    working = tmp_path / "working.norm.jsonl"
    working.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in (_working_row(image_id=2299, ann_id=-7), _working_row(image_id=2300, ann_id=9))
        ),
        encoding="utf-8",
    )
    image_root = tmp_path / "shared-images"
    (image_root / "val2017").mkdir(parents=True)
    destination = tmp_path / "replay"
    return working, image_root, destination


def test_bounded_replay_preserves_historical_jsonl_bytes_and_negative_id(tmp_path: Path) -> None:
    working, image_root, destination = _prepare(tmp_path)

    result = replay_refinement(
        working_norm=working,
        destination=destination,
        image_root=image_root,
        split="val",
        limit=1,
    )

    assert result.row_count == 1
    assert result.norm_path.read_bytes() == (
        b'{"images":["../shared-images/val2017/000000002299.jpg"],"objects":[{"bbox_2d":[10,20,30,40],"desc":"person","category_id":1,"category_name":"person","coco_ann_id":-7,"metadata":{"origin":"human"}},{"bbox_2d":[50,60,90,100],"desc":"tie","category_id":32,"category_name":"tie","coco_ann_id":99}],"width":960,"height":1056,"image_id":2299,"file_name":"images/val2017/000000002299.jpg","metadata":{"source":"coco2017","split":"val"}}\n'
    )
    assert result.coord_path.read_bytes() == (
        b'{"images":["../shared-images/val2017/000000002299.jpg"],"objects":[{"bbox_2d":["<|coord_10|>","<|coord_20|>","<|coord_30|>","<|coord_40|>"],"desc":"person","category_id":1,"category_name":"person","coco_ann_id":-7,"metadata":{"origin":"human"}},{"bbox_2d":["<|coord_50|>","<|coord_60|>","<|coord_90|>","<|coord_100|>"],"desc":"tie","category_id":32,"category_name":"tie","coco_ann_id":99}],"width":960,"height":1056,"image_id":2299,"file_name":"images/val2017/000000002299.jpg","metadata":{"source":"coco2017","split":"val"}}\n'
    )


def test_verify_existing_detects_mismatch_without_writing(tmp_path: Path) -> None:
    working, image_root, destination = _prepare(tmp_path)
    replay_refinement(working_norm=working, destination=destination, image_root=image_root, split="val")
    coord = destination / "val.coord.jsonl"
    coord.write_bytes(b"unexpected\n")
    before = coord.read_bytes()

    with pytest.raises(ReplayError, match="does not match"):
        replay_refinement(
            working_norm=working,
            destination=destination,
            image_root=image_root,
            split="val",
            verify_existing=True,
        )

    assert coord.read_bytes() == before


def test_replay_refuses_existing_outputs_by_default(tmp_path: Path) -> None:
    working, image_root, destination = _prepare(tmp_path)
    replay_refinement(working_norm=working, destination=destination, image_root=image_root, split="val")
    before = (destination / "val.norm.jsonl").read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        replay_refinement(working_norm=working, destination=destination, image_root=image_root, split="val")

    assert (destination / "val.norm.jsonl").read_bytes() == before


def test_module_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "public_data.refinement_replay", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--verify-existing" in completed.stdout
    assert "--limit" in completed.stdout

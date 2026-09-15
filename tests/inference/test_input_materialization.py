import copy
import os
from pathlib import Path

import pytest

from probes.native_owner_scale import evaluation as legacy
from src.data.examples import raw_example_from_jsonl_row
from src.inference.input_materialization import materialize_bound_single_image_case


def _case(image: Path, *, images: list[str] | None = None) -> dict:
    return {
        "image_path": str(image),
        "image_plan": {"image_content_sha256": legacy.file_hash(image)},
        "input_record": {
            "images": ["old/location.jpg"] if images is None else images,
            "file_name": image.name,
            "image_id": 1,
            "metadata": {"source": "coco2017", "split": "train"},
            "objects": [{
                "bbox_2d": [
                    "<|coord_0|>", "<|coord_0|>",
                    "<|coord_999|>", "<|coord_999|>",
                ],
                "desc": "person",
                "category_id": 1,
                "category_name": "person",
                "coco_ann_id": 1,
            }],
            "width": 32,
            "height": 32,
        },
    }


def test_public_materializer_matches_legacy_without_mutating_source(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "bound.jpg"
    image.write_bytes(b"bound image bytes")
    case = _case(image)
    original = copy.deepcopy(case)
    target = tmp_path / "target" / "fresh.jsonl"
    config = {"data": {"input_jsonl": str(target)}}

    expected = legacy._candidate_materialized_case(case, config)
    actual = materialize_bound_single_image_case(case, config)

    assert actual == expected
    assert case == original
    assert actual["input_record"]["images"] == [os.path.relpath(image, target.parent)]
    raw = raw_example_from_jsonl_row(
        actual["input_record"], jsonl_path=target, row_number=1, raw_line="fixture"
    )
    assert raw.image.path == image


def test_public_materializer_rejects_missing_changed_and_multiple_images(tmp_path):
    image = tmp_path / "bound.jpg"
    image.write_bytes(b"bound image bytes")
    config = {"data": {"input_jsonl": str(tmp_path / "fresh.jsonl")}}

    missing = _case(image)
    image.unlink()
    with pytest.raises(ValueError, match="candidate bound image missing"):
        materialize_bound_single_image_case(missing, config)

    image.write_bytes(b"bound image bytes")
    changed = _case(image)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="candidate bound image bytes changed"):
        materialize_bound_single_image_case(changed, config)

    image.write_bytes(b"bound image bytes")
    multiple = _case(image, images=["one.jpg", "two.jpg"])
    with pytest.raises(ValueError, match="candidate requires one frozen image"):
        materialize_bound_single_image_case(multiple, config)

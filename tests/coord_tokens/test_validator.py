import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coord_tokens.validator import annotate_coord_tokens  # noqa: E402


def _make_record():
    return {
        "width": 10,
        "height": 20,
        "objects": [
            {
                "bbox_2d": ["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"],
                "desc": "thing/one",
            }
        ],
    }


def test_annotate_coord_tokens_attaches_metadata():
    record = _make_record()
    found = annotate_coord_tokens(record)
    assert found is True
    obj = record["objects"][0]
    assert "_coord_tokens" in obj and "bbox_2d" in obj["_coord_tokens"]
    assert obj["_coord_token_ints"]["bbox_2d"] == [10, 20, 30, 40]
    assert record.get("_coord_tokens_enabled") is True


def test_missing_width_height_raises():
    record = _make_record()
    record.pop("width")
    with pytest.raises(ValueError):
        annotate_coord_tokens(record)

from __future__ import annotations

from typing import Any

import pytest

from src.config.schema import CoordTokensConfig
from src.datasets.dense_caption import BaseCaptionDataset


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "<decoded>"


class _FakeTemplate:
    def __init__(self) -> None:
        self.system = "BASE_SYSTEM"
        self.tokenizer = _FakeTokenizer()
        self.packing = False
        self.padding_free = False

    def encode(self, merged: dict[str, Any], return_length: bool = False) -> dict[str, Any]:
        messages = merged.get("messages", []) if isinstance(merged, dict) else []
        length = max(4, min(64, len(messages) * 4))
        return {
            "input_ids": [0] * length,
            "labels": [0] * length,
            "length": length,
        }


def _record() -> dict[str, Any]:
    return {
        "images": ["img.jpg"],
        "width": 32,
        "height": 32,
        "objects": [{"bbox_2d": [1, 1, 10, 10], "desc": "cat"}],
    }


def _unsorted_record() -> dict[str, Any]:
    return {
        "images": ["img.jpg"],
        "width": 32,
        "height": 32,
        "objects": [
            {"bbox_2d": [8, 12, 16, 20], "desc": "later"},
            {"bbox_2d": [1, 1, 6, 6], "desc": "earlier"},
        ],
    }


def test_prompt_override_restoration_no_leakage_across_sequential_encodes() -> None:
    template = _FakeTemplate()
    ds = BaseCaptionDataset(
        base_records=[_record()],
        template=template,
        user_prompt="Describe objects",
        emit_norm="none",
        json_format="standard",
        use_summary=False,
        system_prompt_dense="BASE_SYSTEM",
        coord_tokens=CoordTokensConfig(enabled=False),
        object_ordering="sorted",
    )

    builder = ds._create_builder("dense")
    rec = ds.base_records[0]

    encoded_a = ds._encode_record(record=rec, builder=builder, system_prompt="SYSTEM_A")
    assert encoded_a["messages"][0]["role"] == "system"
    assert encoded_a["messages"][0]["content"] == "SYSTEM_A"
    assert template.system == "BASE_SYSTEM"

    encoded_b = ds._encode_record(record=rec, builder=builder, system_prompt="SYSTEM_B")
    assert encoded_b["messages"][0]["role"] == "system"
    assert encoded_b["messages"][0]["content"] == "SYSTEM_B"
    assert template.system == "BASE_SYSTEM"


def test_sorted_ordering_raises_on_unsorted_objects() -> None:
    template = _FakeTemplate()
    ds = BaseCaptionDataset(
        base_records=[_unsorted_record()],
        template=template,
        user_prompt="Describe objects",
        emit_norm="none",
        json_format="standard",
        use_summary=False,
        system_prompt_dense="BASE_SYSTEM",
        coord_tokens=CoordTokensConfig(enabled=False),
        object_ordering="sorted",
    )

    with pytest.raises(ValueError, match="must already be top-left sorted"):
        ds[0]


def test_random_ordering_accepts_unsorted_objects() -> None:
    template = _FakeTemplate()
    ds = BaseCaptionDataset(
        base_records=[_unsorted_record()],
        template=template,
        user_prompt="Describe objects",
        emit_norm="none",
        json_format="standard",
        use_summary=False,
        system_prompt_dense="BASE_SYSTEM",
        coord_tokens=CoordTokensConfig(enabled=False),
        object_ordering="random",
    )

    encoded = ds[0]
    assert "input_ids" in encoded

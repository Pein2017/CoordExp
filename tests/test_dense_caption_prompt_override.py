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
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_1|>",
                    "<|coord_1|>",
                    "<|coord_10|>",
                    "<|coord_10|>",
                ],
                "desc": "cat",
            }
        ],
    }


def _unsorted_record() -> dict[str, Any]:
    return {
        "images": ["img.jpg"],
        "width": 32,
        "height": 32,
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_8|>",
                    "<|coord_12|>",
                    "<|coord_16|>",
                    "<|coord_20|>",
                ],
                "desc": "later",
            },
            {
                "bbox_2d": [
                    "<|coord_1|>",
                    "<|coord_1|>",
                    "<|coord_6|>",
                    "<|coord_6|>",
                ],
                "desc": "earlier",
            },
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
        coord_tokens=CoordTokensConfig(enabled=True),
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
        coord_tokens=CoordTokensConfig(enabled=True),
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
        coord_tokens=CoordTokensConfig(enabled=True),
        object_ordering="random",
    )

    encoded = ds[0]
    assert "input_ids" in encoded


def test_template_boundary_reflects_geometry_first_assistant_text_order() -> None:
    class _CaptureTemplate(_FakeTemplate):
        def __init__(self) -> None:
            super().__init__()
            self.last_assistant_text = ""

        def encode(
            self, merged: dict[str, Any], return_length: bool = False
        ) -> dict[str, Any]:
            messages = merged.get("messages", []) if isinstance(merged, dict) else []
            if len(messages) >= 2:
                assistant = messages[1]
                content = assistant.get("content", []) if isinstance(assistant, dict) else []
                if content and isinstance(content[0], dict):
                    self.last_assistant_text = str(content[0].get("text", ""))
            return super().encode(merged, return_length=return_length)

    template = _CaptureTemplate()
    ds = BaseCaptionDataset(
        base_records=[_record()],
        template=template,
        user_prompt="Describe objects",
        emit_norm="none",
        json_format="standard",
        use_summary=False,
        system_prompt_dense="BASE_SYSTEM",
        coord_tokens=CoordTokensConfig(enabled=True),
        object_ordering="sorted",
        object_field_order="geometry_first",
    )

    encoded = ds[0]
    obj = encoded["assistant_payload"]["objects"][0]
    assert list(obj.keys()) == ["bbox_2d", "desc"]
    assert template.last_assistant_text.index('"bbox_2d"') < template.last_assistant_text.index(
        '"desc"'
    )

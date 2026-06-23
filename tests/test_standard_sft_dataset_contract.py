from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.detection.dataset import DetectionTrainingDataset


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class _TinyTokenizer:
    eos_token = "<|endoftext|>"
    unk_token_id = 0

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|endoftext|>": 3,
            "<|object_ref_start|>": 4,
            "<|object_ref_end|>": 5,
            "<|box_start|>": 6,
            "<|box_end|>": 7,
        }
        self.eos_token_id = self._token_to_id[self.eos_token]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        encoded = self(text, return_offsets_mapping=True, add_special_tokens=False)
        return list(encoded["input_ids"])

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        assert add_generation_prompt is False
        rendered = "".join(
            f"<|im_start|>{message['role']}\n"
            f"{self._content_text(message['content'])}<|im_end|>\n"
            for message in messages
        )
        if not tokenize:
            return rendered
        encoded = self(rendered, return_offsets_mapping=True, add_special_tokens=False)
        return list(encoded["input_ids"])

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert return_offsets_mapping is True
        assert add_special_tokens is False
        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            match = _SPECIAL_TOKEN_RE.match(text, cursor)
            if match is not None:
                token_text = match.group(0)
                token_end = match.end()
            else:
                token_text = text[cursor]
                token_end = cursor + 1
            token_id = self._token_to_id.setdefault(
                token_text,
                len(self._token_to_id) + 1,
            )
            input_ids.append(token_id)
            offsets.append((cursor, token_end))
            cursor = token_end
        return {"input_ids": input_ids, "offset_mapping": offsets}

    def _content_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        assert isinstance(content, list)
        parts: list[str] = []
        for item in content:
            assert isinstance(item, dict)
            if item.get("type") == "image":
                parts.append("<image>")
            elif item.get("type") == "text":
                parts.append(str(item.get("text")))
        return "".join(parts)


class _SparseLabelSwiftTemplate:
    """Swift-like template whose labels intentionally differ from span tokenization."""

    def __init__(self) -> None:
        self.tokenizer = _TinyTokenizer()

    def encode(
        self,
        payload: Mapping[str, Any],
        *,
        return_length: bool,
    ) -> dict[str, Any]:
        assert return_length is True
        messages = [dict(message) for message in payload["messages"]]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = list(encoded["input_ids"])
        labels = [-100 for _ in input_ids]
        assistant_indices = [
            index
            for index, token_id in enumerate(input_ids)
            if token_id not in {1, 2, 3}
        ]
        for index in assistant_indices[:2]:
            labels[index] = input_ids[index]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1 for _ in input_ids],
            "length": len(input_ids),
        }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _raw_row() -> dict[str, Any]:
    return {
        "images": ["images/train2017/example.jpg"],
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
                "desc": "cat",
                "category_id": 17,
                "category_name": "cat",
                "coco_ann_id": 101,
            }
        ],
        "width": 640,
        "height": 480,
        "image_id": 9,
        "file_name": "images/train2017/example.jpg",
        "metadata": {"source": "unit", "split": "train"},
    }


def test_standard_sft_compact_dataset_does_not_require_recursive_sidecar_alignment(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [_raw_row()])
    image_path = tmp_path / "image-root/images/train2017/example.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"unit-test-image-placeholder")

    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=_SparseLabelSwiftTemplate(),
        image_root=tmp_path / "image-root",
        detection_template_id="compact_object_box_closed",
        mode="sorted_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        seed=123,
        state_weighting="none",
        normalization="token_mean",
    )

    sample = dataset[0]

    assert "recursive_detection_targets" not in sample
    assert sample["detection_metadata"]["mode"] == "sorted_sft"
    assert sample["detection_metadata"]["template_id"] == "compact_object_box_closed"

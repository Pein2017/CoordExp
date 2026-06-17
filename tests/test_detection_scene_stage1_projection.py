from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import src.detection as detection
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_KEY
from src.detection.dataset import DetectionDatasetRuntimeConfig, DetectionTrainingDataset


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class _FakeTokenizer:
    eos_token = "<|endoftext|>"
    unk_token_id = 0

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|endoftext|>": 3,
            "<|object_ref_start|>": 4,
            "<|box_start|>": 5,
        }
        self.eos_token_id = self._token_to_id[self.eos_token]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(
            self(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )["input_ids"]
        )

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
        return list(
            self(
                rendered,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )["input_ids"]
        )

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


class _FakeSwiftTemplate:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

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
        offsets = list(encoded["offset_mapping"])
        labels = [-100 for _ in input_ids]

        assistant_text = _assistant_text(messages)
        assistant_start = text.find(assistant_text)
        assert assistant_start >= 0
        assistant_end = assistant_start + len(assistant_text)
        stop_end = assistant_end
        if text.startswith("<|im_end|>", assistant_end):
            stop_end = assistant_end + len("<|im_end|>")

        for index, (start, end) in enumerate(offsets):
            if assistant_start <= start and end <= stop_end:
                labels[index] = input_ids[index]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1 for _ in input_ids],
            "length": len(input_ids),
        }


def _assistant_text(messages: list[dict[str, Any]]) -> str:
    assistant = messages[-1]
    assert assistant["role"] == "assistant"
    return _FakeTokenizer()._content_text(assistant["content"])


def _raw_row() -> dict[str, Any]:
    return {
        "images": ["images/train2017/example.jpg"],
        "objects": [
            {
                "bbox_2d": ["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"],
                "desc": "cat",
                "category_id": 17,
                "category_name": "cat",
                "coco_ann_id": 101,
                "object_id": "obj-cat",
            },
            {
                "bbox_2d": ["<|coord_50|>", "<|coord_60|>", "<|coord_70|>", "<|coord_80|>"],
                "desc": "dog",
                "category_id": 18,
                "category_name": "dog",
                "coco_ann_id": 102,
                "object_id": "obj-dog",
            },
        ],
        "width": 640,
        "height": 480,
        "image_id": 9,
        "file_name": "images/train2017/example.jpg",
        "metadata": {"source": "unit", "split": "train"},
    }


def test_stage1_teacher_forcing_dataset_projects_scene_to_rendered_sequence_and_supervision(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image-root/images/train2017/example.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"placeholder")
    dataset = DetectionTrainingDataset(
        [_raw_row()],
        swift_template=_FakeSwiftTemplate(),
        config=DetectionDatasetRuntimeConfig(
            image_root=str(tmp_path / "image-root"),
            detection_template_id="compact",
            mode="teacher_forcing",
            object_ordering="random_permutation",
            user_prompt="Detect every object.",
            system_prompt=None,
            seed=123,
            state_weighting="uniform_permutation",
            normalization="semantic_image_bucket_balanced",
            teacher_forcing_profile="pure_valid_set_marginal",
        ),
        dataset_name="unit",
    )

    item = dataset[0]

    target_ir = item[TEACHER_FORCING_TARGET_IR_KEY]
    supervision_meta = item["detection_supervision_view_metadata"]
    supervised_positions = tuple(
        index for index, label in enumerate(item["labels"]) if int(label) != -100
    )
    assert tuple(atom.target_position for atom in target_ir.atoms) == supervised_positions
    assert target_ir.metadata["token_position_origin"] == (
        "DetectionSupervisionView.adapter_to_swift_encoded"
    )
    assert target_ir.metadata["supervision_view_token_position_origin"] == (
        "DetectionSupervisionView.tokenized"
    )
    assert supervision_meta["authority"] == "DetectionSupervisionView"
    assert supervision_meta["adapter"] == (
        "DetectionTrainingDataset.teacher_forcing_swift_adapter"
    )
    assert supervision_meta["rendered_sequence_type"] == "RenderedDetectionSequence"
    assert tuple(supervision_meta["supervised_label_positions"]) == supervised_positions
    assert tuple(supervision_meta["next_token_prediction_positions"]) == tuple(
        position - 1 for position in supervised_positions
    )
    assert supervision_meta["template_id"] == "compact"
    assert len(supervision_meta["coord_token_spans"]) == 8
    assert set(supervision_meta["coord_token_positions"]).issubset(supervised_positions)
    assert all(item["labels"][position] == item["input_ids"][position] for position in supervised_positions)
    assert {atom.coord_role for atom in target_ir.atoms if atom.coord_role is not None} == {
        "x1",
        "y1",
        "x2",
        "y2",
    }
    payload_objects = item["assistant_payload"]["objects"]
    assert sorted(obj["desc"] for obj in payload_objects) == ["cat", "dog"]
    assert {
        tuple(obj["bbox_2d"])
        for obj in payload_objects
    } == {
        ("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
        ("<|coord_50|>", "<|coord_60|>", "<|coord_70|>", "<|coord_80|>"),
    }
    assert item["detection_metadata"]["template_id"] == "compact"
    assert item["detection_metadata"]["object_ordering"] == "random_permutation"
    assert item["metadata"] == {
        "source": "unit",
        "split": "train",
        "image_id": 9,
        "file_name": "images/train2017/example.jpg",
    }


def test_detection_root_exports_canonical_projection_names_only() -> None:
    assert hasattr(detection, "RenderedDetectionSequence")
    assert hasattr(detection, "DetectionSupervisionView")
    assert not hasattr(detection, "RenderedAssistantSequence")
    assert not hasattr(detection, "TokenizedDetectionExample")

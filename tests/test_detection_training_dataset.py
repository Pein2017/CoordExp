from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from src.config import ConfigLoader, LatestDetectionTrainingConfig
from src.detection.dataset import DetectionTrainingDataset
from src.detection.objective import compute_eos_trust_weight


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class FakeTokenizer:
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

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    @property
    def special_tokens_map(self) -> dict[str, str]:
        return {"im_start": "<|im_start|>", "im_end": "<|im_end|>"}

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
                token_text, len(self._token_to_id) + 1
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


class FakeSwiftTemplate:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

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


class DriftingSwiftTemplate(FakeSwiftTemplate):
    def encode(
        self,
        payload: Mapping[str, Any],
        *,
        return_length: bool,
    ) -> dict[str, Any]:
        encoded = super().encode(payload, return_length=return_length)
        encoded["input_ids"] = [*encoded["input_ids"], 999_999]
        return encoded


class ImageExpandingSwiftTemplate(FakeSwiftTemplate):
    def encode(
        self,
        payload: Mapping[str, Any],
        *,
        return_length: bool,
    ) -> dict[str, Any]:
        messages = [dict(message) for message in payload["messages"]]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        text = text.replace("<image>", "<image><image><image>", 1)
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


def _assistant_text(messages: Sequence[Mapping[str, Any]]) -> str:
    assistant_messages = [
        message for message in messages if message.get("role") == "assistant"
    ]
    assert len(assistant_messages) == 1
    content = assistant_messages[0]["content"]
    if isinstance(content, str):
        return content
    assert isinstance(content, list)
    text_parts = [
        str(item["text"])
        for item in content
        if isinstance(item, Mapping) and item.get("type") == "text"
    ]
    return "".join(text_parts)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
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
            },
            {
                "bbox_2d": [
                    "<|coord_50|>",
                    "<|coord_60|>",
                    "<|coord_70|>",
                    "<|coord_80|>",
                ],
                "desc": "dog",
                "category_id": 18,
                "category_name": "dog",
                "coco_ann_id": 102,
            },
            {
                "bbox_2d": [
                    "<|coord_90|>",
                    "<|coord_100|>",
                    "<|coord_110|>",
                    "<|coord_120|>",
                ],
                "desc": "bus",
                "category_id": 6,
                "category_name": "bus",
                "coco_ann_id": 103,
            },
        ],
        "width": 640,
        "height": 480,
        "image_id": 9,
        "file_name": "images/train2017/example.jpg",
        "metadata": {"source": "unit", "split": "train"},
    }


def _dataset(
    tmp_path: Path,
    *,
    swift_template: Any | None = None,
    eos_trust_weight_config: Mapping[str, Any] | None = None,
) -> DetectionTrainingDataset:
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [_raw_row()])
    _ensure_image(tmp_path)
    return DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=swift_template or FakeSwiftTemplate(),
        image_root=tmp_path / "image-root",
        detection_template_id="compact_full",
        mode="random_permutation_et_rmp_ce",
        object_ordering="random_permutation",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        max_objects=60,
        seed=123,
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        eos_trust_weight_config=eos_trust_weight_config,
    )


def _ensure_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "image-root/images/train2017/example.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"unit-test-image-placeholder")
    return image_path


def test_latest_detection_dataset_returns_encoded_sample_with_recursive_sidecar(
    tmp_path: Path,
) -> None:
    sample = _dataset(tmp_path)[0]

    assert "recursive_detection_targets" in sample
    assert sample["recursive_detection_targets"].token_targets
    assert sample["messages"][0]["role"] == "system"
    assert sample["messages"][1]["content"][0]["type"] == "image"
    assert sample["messages"][1]["content"][0]["image"].endswith(
        "image-root/images/train2017/example.jpg"
    )
    assert sample["messages"][2]["role"] == "assistant"
    assert sample["assistant_payload"]["objects"]
    assert sample["detection_metadata"]["template_id"] == "compact_full"
    assert sample["detection_metadata"]["mode"] == "random_permutation_et_rmp_ce"

    supervised_positions = tuple(
        index for index, label in enumerate(sample["labels"]) if label != -100
    )
    target_positions = tuple(
        target.position
        for target in sample["recursive_detection_targets"].token_targets
    )
    assert target_positions == supervised_positions
    assert all(
        sample["labels"][target.position] == target.teacher_token_id
        for target in sample["recursive_detection_targets"].token_targets
    )


def test_legacy_latest_compact_config_still_accepts_data_image_root() -> None:
    config = ConfigLoader.load_materialized_training_config(
        "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml"
    )

    assert isinstance(config, LatestDetectionTrainingConfig)
    assert config.data.train_jsonl.endswith("train.coord.jsonl")
    assert config.data.val_jsonl.endswith("val.coord.jsonl")
    assert config.data.image_root == "public_data/coco/rescale_32_1024_bbox_max60"


def test_latest_detection_dataset_applies_eos_trust_without_prefix_rollin(
    tmp_path: Path,
) -> None:
    eos_trust_weight_config = {"source": "constant_ablation", "value": 0.25}
    dataset = _dataset(tmp_path, eos_trust_weight_config=eos_trust_weight_config)

    sample = dataset[0]

    assert sample["detection_metadata"]["mode"] == "random_permutation_et_rmp_ce"
    assert "rollin_k" not in sample["detection_metadata"]
    assert sample["detection_metadata"]["eos_trust_weight"] == pytest.approx(
        compute_eos_trust_weight(3, eos_trust_weight_config)
    )

    supervised_positions = tuple(
        index for index, label in enumerate(sample["labels"]) if label != -100
    )
    targets = sample["recursive_detection_targets"].token_targets
    target_positions = tuple(target.position for target in targets)
    assert target_positions == supervised_positions

    im_end_id = dataset.tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_targets = [target for target in targets if target.teacher_token_id == im_end_id]
    non_eos_targets = [
        target for target in targets if target.teacher_token_id != im_end_id
    ]
    assert eos_targets
    assert all(target.loss_weight == pytest.approx(0.25) for target in eos_targets)
    assert all(target.loss_weight == pytest.approx(1.0) for target in non_eos_targets)


def test_latest_detection_dataset_random_order_is_epoch_deterministic(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    dataset.set_epoch(0)
    epoch0_a = tuple(dataset[0]["detection_metadata"]["realized_source_object_indices"])
    dataset.set_epoch(0)
    epoch0_b = tuple(dataset[0]["detection_metadata"]["realized_source_object_indices"])
    observed = set()
    for epoch in range(8):
        dataset.set_epoch(epoch)
        observed.add(
            tuple(dataset[0]["detection_metadata"]["realized_source_object_indices"])
        )

    assert epoch0_a == epoch0_b
    assert all(sorted(order) == [0, 1, 2] for order in observed)
    assert len(observed) > 1


def test_latest_detection_dataset_allows_swift_image_token_expansion(
    tmp_path: Path,
) -> None:
    sample = _dataset(tmp_path, swift_template=ImageExpandingSwiftTemplate())[0]

    assert len(sample["input_ids"]) > len(sample["labels"]) - 1
    assert all(
        sample["labels"][target.position] == target.teacher_token_id
        for target in sample["recursive_detection_targets"].token_targets
    )


def test_latest_detection_dataset_rejects_encode_sidecar_drift(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, swift_template=DriftingSwiftTemplate())

    with pytest.raises(ValueError, match="encoded input_ids and labels"):
        dataset[0]


def test_latest_detection_dataset_sft_mode_does_not_attach_recursive_sidecar(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [_raw_row()])
    _ensure_image(tmp_path)
    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=FakeSwiftTemplate(),
        image_root=tmp_path / "image-root",
        detection_template_id="stage1_json_pretty",
        mode="sorted_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt=None,
        max_objects=60,
        seed=123,
        state_weighting="none",
        normalization="token_mean",
    )

    sample = dataset[0]

    assert "recursive_detection_targets" not in sample
    assert sample["detection_metadata"]["template_id"] == "stage1_json_pretty"
    assert sample["detection_metadata"]["realized_source_object_indices"] == [0, 1, 2]


def test_prefix_rollin_dataset_masks_prefix_and_keeps_weighted_im_end_target(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [_raw_row()])
    _ensure_image(tmp_path)
    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=FakeSwiftTemplate(),
        image_root=tmp_path / "image-root",
        detection_template_id="compact_full",
        mode="prefix_rollin_et_rmp_ce",
        object_ordering="random_permutation",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        max_objects=60,
        seed=123,
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        eos_trust_weight_config={
            "source": "empirical_unlabeled_poisson_v0",
            "expected_unlabeled_count": {
                "intercept": -0.35,
                "slope": 0.43,
                "floor": 0.0,
            },
            "trust_mapping": {
                "type": "log_linear_missing_count_penalty",
                "penalty_per_missing": 1.0,
                "temperature": 1.0,
                "min_weight": 0.0,
                "max_weight": 1.0,
            },
        },
    )
    dataset.set_epoch(3)

    sample = dataset[0]

    assert sample["detection_metadata"]["mode"] == "prefix_rollin_et_rmp_ce"
    assert sample["detection_metadata"]["rollin_k"] == 1
    assert sample["detection_metadata"]["rollin_prefix_token_count"] > 0
    assert sample["detection_metadata"]["supervised_suffix_token_count"] > 0
    assert "recursive_detection_targets" in sample
    supervised_positions = tuple(
        index for index, label in enumerate(sample["labels"]) if label != -100
    )
    assert supervised_positions
    target_positions = tuple(
        target.position
        for target in sample["recursive_detection_targets"].token_targets
    )
    assert target_positions == supervised_positions
    assert len(supervised_positions) < (
        sample["detection_metadata"]["rollin_prefix_token_count"]
        + sample["detection_metadata"]["supervised_suffix_token_count"]
        + sample["detection_metadata"]["semantic_eos_token_count"]
    )
    im_end_id = dataset.tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_targets = [
        target
        for target in sample["recursive_detection_targets"].token_targets
        if target.teacher_token_id == im_end_id
    ]
    assert eos_targets
    assert eos_targets[-1].loss_weight == pytest.approx(
        sample["detection_metadata"]["eos_trust_weight"]
    )
    assert 0.0 <= sample["detection_metadata"]["eos_trust_weight"] <= 1.0


def test_latest_detection_dataset_rejects_missing_image_path(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [_raw_row()])
    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=FakeSwiftTemplate(),
        image_root=tmp_path / "image-root",
        detection_template_id="compact_full",
        mode="random_permutation_et_rmp_ce",
        object_ordering="random_permutation",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        max_objects=60,
        seed=123,
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
    )

    with pytest.raises(FileNotFoundError, match="image path does not exist"):
        dataset[0]


def test_latest_detection_dataset_rejects_absolute_image_outside_root(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    row = _raw_row()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    row["images"] = [str(outside)]
    _write_jsonl(jsonl_path, [row])
    (tmp_path / "image-root").mkdir(parents=True)
    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=FakeSwiftTemplate(),
        image_root=tmp_path / "image-root",
        detection_template_id="compact_full",
        mode="random_permutation_et_rmp_ce",
        object_ordering="random_permutation",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        max_objects=60,
        seed=123,
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
    )

    with pytest.raises(ValueError, match="outside image_root"):
        dataset[0]

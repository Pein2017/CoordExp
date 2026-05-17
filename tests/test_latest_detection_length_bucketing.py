from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    LengthGroupedSampler,
)

from src.detection.dataset import (
    DetectionDatasetRuntimeConfig,
    DetectionTrainingDataset,
)
from src.detection.length_bucketing import (
    LatestDetectionLengthBucketingConfig,
    LatestDetectionLengthGroupedTrainerMixin,
    LatestDetectionLengthProvider,
    build_latest_detection_length_grouped_sampler,
)


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class BoundarySensitiveTokenizer:
    eos_token = "<|endoftext|>"
    unk_token_id = 0

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|endoftext|>": 3,
            "<|object_ref_start|>": 4,
            "<|box_start|>": 5,
            "<image>": 6,
            "Detect every object.": 7,
            "striped tabby": 8,
            "black dog": 9,
            "red bus": 10,
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
        merge_tokens = sorted(self._token_to_id, key=len, reverse=True)
        while cursor < len(text):
            match = _SPECIAL_TOKEN_RE.match(text, cursor)
            if match is not None:
                token_text = match.group(0)
                token_end = match.end()
            else:
                token_text = ""
                for candidate in merge_tokens:
                    if text.startswith(candidate, cursor):
                        token_text = candidate
                        break
                if token_text:
                    token_end = cursor + len(token_text)
                else:
                    token_text = text[cursor]
                    token_end = cursor + 1
            token_id = self._token_to_id.setdefault(token_text, len(self._token_to_id) + 1)
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
            assert isinstance(item, Mapping)
            if item.get("type") == "image":
                parts.append("<image>")
            elif item.get("type") == "text":
                parts.append(str(item.get("text")))
        return "".join(parts)


class BoundarySensitiveSwiftTemplate:
    def __init__(self) -> None:
        self.tokenizer = BoundarySensitiveTokenizer()

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


class SpyDetectionTrainingDataset(DetectionTrainingDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.getitem_calls = 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.getitem_calls += 1
        return super().__getitem__(index)


def _assistant_text(messages: Sequence[Mapping[str, Any]]) -> str:
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    assert len(assistant_messages) == 1
    content = assistant_messages[0]["content"]
    if isinstance(content, str):
        return content
    assert isinstance(content, list)
    return "".join(
        str(item["text"])
        for item in content
        if isinstance(item, Mapping) and item.get("type") == "text"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _raw_row(*, image_name: str = "example.jpg") -> dict[str, Any]:
    return {
        "images": [f"images/train2017/{image_name}"],
        "objects": [
            {
                "bbox_2d": ["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"],
                "desc": "striped tabby",
                "category_id": 17,
                "category_name": "cat",
                "coco_ann_id": 101,
            },
            {
                "bbox_2d": ["<|coord_50|>", "<|coord_60|>", "<|coord_70|>", "<|coord_80|>"],
                "desc": "black dog",
                "category_id": 18,
                "category_name": "dog",
                "coco_ann_id": 102,
            },
            {
                "bbox_2d": ["<|coord_90|>", "<|coord_100|>", "<|coord_110|>", "<|coord_120|>"],
                "desc": "red bus",
                "category_id": 6,
                "category_name": "bus",
                "coco_ann_id": 103,
            },
        ],
        "width": 640,
        "height": 480,
        "image_id": 9,
        "file_name": f"images/train2017/{image_name}",
        "metadata": {"source": "unit", "split": "train"},
    }


def _norm1000_row_with_object_count(
    object_count: int,
    *,
    image_name: str = "crowded.jpg",
) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    for index in range(int(object_count)):
        coord = index % 900
        objects.append(
            {
                "object_id": f"unit:ann:{index}",
                "bbox_2d": [coord, coord, coord + 10, coord + 20],
                "desc": f"unit object {index}",
                "category_id": 1,
                "category_name": "object",
                "coco_ann_id": index,
            }
        )
    return {
        "images": [f"images/train2017/{image_name}"],
        "objects": objects,
        "width": 640,
        "height": 480,
        "image_id": 61,
        "file_name": f"images/train2017/{image_name}",
        "metadata": {"source": "unit", "split": "train"},
    }


def _make_dataset(
    tmp_path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    spy: bool = False,
    max_objects: int | None = None,
) -> DetectionTrainingDataset:
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, rows)
    image_root = tmp_path / "image-root"
    for row in rows:
        for image in row["images"]:
            image_path = image_root / str(image)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"unit-test-image-placeholder")

    dataset_cls = SpyDetectionTrainingDataset if spy else DetectionTrainingDataset
    return dataset_cls.from_jsonl(
        jsonl_path,
        swift_template=BoundarySensitiveSwiftTemplate(),
        image_root=image_root,
        detection_template_id="compact_full",
        mode="prefix_rollin_et_rmp_ce",
        object_ordering="random_permutation",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        max_objects=max_objects,
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
        dataset_name="latest_detection_train",
    )


def test_encoded_length_for_latest_compact_norm1000_row_ignores_legacy_max_objects(
    tmp_path: Path,
) -> None:
    with pytest.warns(UserWarning, match="data.max_objects is compatibility-only"):
        dataset = _make_dataset(
            tmp_path,
            [_norm1000_row_with_object_count(61)],
            max_objects=60,
        )

    assert dataset.encoded_length_for_row(0) > 0
    assert dataset[0]["detection_metadata"]["object_count"] == 61


def _padding_waste(indices: Sequence[int], lengths: Sequence[int], *, batch_size: int) -> int:
    waste = 0
    for start in range(0, len(indices), batch_size):
        batch = list(indices[start : start + batch_size])
        if not batch:
            continue
        max_len = max(lengths[index] for index in batch)
        waste += sum(max_len - lengths[index] for index in batch)
    return waste


def test_prefix_rollin_length_provider_is_invariant_to_k_and_random_order(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path, [_raw_row()])
    provider = LatestDetectionLengthProvider(dataset)

    object_count = len(dataset.rows[0]["objects"])
    lengths_by_k = {
        provider.length_for_row(base_idx=0, forced_rollin_k=k)
        for k in range(object_count + 1)
    }

    assert len(lengths_by_k) == 1
    dataset.set_epoch(0)
    length_epoch_0 = provider.length_for_row(base_idx=0, epoch=0)
    dataset.set_epoch(7)
    length_epoch_7 = provider.length_for_row(base_idx=0, epoch=7)
    assert length_epoch_0 == length_epoch_7

    for epoch in range(4):
        dataset.set_epoch(epoch)
        sample = dataset[0]
        assert len(sample["input_ids"]) == provider.length_for_row(base_idx=0, epoch=epoch)


def test_length_precompute_does_not_call_getitem(tmp_path: Path) -> None:
    rows = [_raw_row(image_name=f"example-{index}.jpg") for index in range(3)]
    dataset = _make_dataset(tmp_path, rows, spy=True)
    assert isinstance(dataset, SpyDetectionTrainingDataset)
    provider = LatestDetectionLengthProvider(dataset)

    lengths = provider.all_lengths()

    assert len(lengths) == 3
    assert dataset.getitem_calls == 0


def test_detection_dataset_rejects_out_of_range_index_for_finite_iteration(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path, [_raw_row()])

    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_length_grouped_sampler_reduces_padding_waste_and_is_epoch_stable() -> None:
    lengths = [60, 600, 70, 610, 80, 620, 90, 630, 100, 640, 110, 650]
    batch_size = 2
    plain_indices = list(range(len(lengths)))

    sampler = build_latest_detection_length_grouped_sampler(
        lengths=lengths,
        batch_size=batch_size,
        seed=17,
        drop_last=False,
    )
    grouped_indices_a = list(iter(sampler))
    grouped_indices_b = list(iter(sampler))

    assert isinstance(sampler, LengthGroupedSampler)
    assert grouped_indices_a == grouped_indices_b
    assert sorted(grouped_indices_a) == plain_indices
    assert _padding_waste(grouped_indices_a, lengths, batch_size=batch_size) < _padding_waste(
        plain_indices,
        lengths,
        batch_size=batch_size,
    )


def test_distributed_length_grouped_sampler_shards_and_reshuffles() -> None:
    lengths = [40, 400, 50, 410, 60, 420, 70, 430]
    rank0 = build_latest_detection_length_grouped_sampler(
        lengths=lengths,
        batch_size=2,
        seed=5,
        drop_last=False,
        world_size=2,
        rank=0,
    )
    rank1 = build_latest_detection_length_grouped_sampler(
        lengths=lengths,
        batch_size=2,
        seed=5,
        drop_last=False,
        world_size=2,
        rank=1,
    )

    assert isinstance(rank0, DistributedLengthGroupedSampler)
    assert isinstance(rank1, DistributedLengthGroupedSampler)
    epoch0_rank0 = list(iter(rank0))
    epoch0_rank1 = list(iter(rank1))
    assert len(epoch0_rank0) == len(epoch0_rank1)
    assert set(epoch0_rank0).isdisjoint(epoch0_rank1)
    assert sorted(epoch0_rank0 + epoch0_rank1) == list(range(len(lengths)))

    rank0.set_epoch(1)
    rank1.set_epoch(1)
    epoch1_rank0 = list(iter(rank0))
    epoch1_rank1 = list(iter(rank1))
    assert sorted(epoch1_rank0 + epoch1_rank1) == list(range(len(lengths)))
    assert epoch1_rank0 != epoch0_rank0 or epoch1_rank1 != epoch0_rank1


def test_trainer_mixin_passes_explicit_lengths_without_getitem(tmp_path: Path) -> None:
    rows = [_raw_row(image_name=f"example-{index}.jpg") for index in range(4)]
    dataset = _make_dataset(tmp_path, rows, spy=True)
    assert isinstance(dataset, SpyDetectionTrainingDataset)

    class BaseTrainer:
        def __init__(self) -> None:
            self.train_dataset = dataset
            self.args = SimpleNamespace(
                train_batch_size=2,
                gradient_accumulation_steps=1,
                seed=11,
                dataloader_drop_last=False,
            )

        def _get_train_sampler(self, train_dataset=None):
            return "fallback"

    class BucketedTrainer(LatestDetectionLengthGroupedTrainerMixin, BaseTrainer):
        pass

    trainer = BucketedTrainer()
    trainer.latest_detection_length_bucketing = LatestDetectionLengthBucketingConfig(
        enabled=True,
        seed=11,
    )

    sampler = trainer._get_train_sampler()

    assert isinstance(sampler, LengthGroupedSampler)
    assert dataset.getitem_calls == 0


def test_trainer_mixin_shards_train_sampler_by_process_rank(tmp_path: Path) -> None:
    rows = [_raw_row(image_name=f"example-{index}.jpg") for index in range(4)]
    dataset = _make_dataset(tmp_path, rows, spy=True)
    assert isinstance(dataset, SpyDetectionTrainingDataset)

    class BaseTrainer:
        def __init__(self) -> None:
            self.train_dataset = dataset
            self.args = SimpleNamespace(
                train_batch_size=2,
                gradient_accumulation_steps=1,
                seed=11,
                dataloader_drop_last=False,
                world_size=2,
                process_index=1,
            )

        def _get_train_sampler(self, train_dataset=None):
            return "fallback"

    class BucketedTrainer(LatestDetectionLengthGroupedTrainerMixin, BaseTrainer):
        pass

    trainer = BucketedTrainer()
    trainer.latest_detection_length_bucketing = LatestDetectionLengthBucketingConfig(
        enabled=True,
        seed=11,
    )

    sampler = trainer._get_train_sampler()

    assert isinstance(sampler, DistributedLengthGroupedSampler)
    assert list(iter(sampler))
    assert dataset.getitem_calls == 0


def test_trainer_mixin_eval_sampler_uses_explicit_lengths_without_getitem(
    tmp_path: Path,
) -> None:
    rows = [_raw_row(image_name=f"example-{index}.jpg") for index in range(4)]
    dataset = _make_dataset(tmp_path, rows, spy=True)
    assert isinstance(dataset, SpyDetectionTrainingDataset)

    class BaseTrainer:
        def __init__(self) -> None:
            self.args = SimpleNamespace(
                eval_batch_size=2,
                seed=11,
            )

        def _get_eval_sampler(self, eval_dataset):
            return "fallback"

    class BucketedTrainer(LatestDetectionLengthGroupedTrainerMixin, BaseTrainer):
        pass

    trainer = BucketedTrainer()
    trainer.latest_detection_length_bucketing = LatestDetectionLengthBucketingConfig(
        enabled=True,
        seed=11,
    )

    sampler = trainer._get_eval_sampler(dataset)

    assert isinstance(sampler, LengthGroupedSampler)
    assert sorted(list(iter(sampler))) == list(range(len(dataset)))
    assert dataset.getitem_calls == 0

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import src.detection.prefix_denoising.dataset as prefix_dataset_mod
from src.config.schema import PrefixDenoisingConfig
from src.detection.prefix_denoising.builder import (
    build_hybrid_prefix_denoising_sample,
    estimate_hybrid_prefix_denoising_packing,
)
from src.detection.prefix_denoising.dataset import (
    PrefixDenoisingTrainingDataset,
    build_prefix_denoising_eligibility_index,
)


_COORD_RE = re.compile(r"<\|coord_(\d+)\|>")


class FakeTokenizer:
    image_token_id = 999_999

    def encode(self, text: str, *args: object, **kwargs: object) -> list[int]:
        del args, kwargs
        if match := _COORD_RE.fullmatch(text):
            return self._coord_ids(int(match.group(1)))
        ids: list[int] = []
        cursor = 0
        for match in _COORD_RE.finditer(text):
            ids.extend(10_000 + ord(ch) for ch in text[cursor : match.start()])
            ids.extend(self._coord_ids(int(match.group(1))))
            cursor = match.end()
        ids.extend(10_000 + ord(ch) for ch in text[cursor:])
        return ids

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|image_pad|>":
            return self.image_token_id
        return self.encode(token)[0]

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: object | None = None,
    ) -> list[int] | str:
        del add_generation_prompt, return_tensors
        parts: list[str] = []
        for message in messages:
            parts.append(f"<{message['role']}>")
            content = message["content"]
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, Mapping):
                        continue
                    if item.get("type") == "image":
                        parts.append("<|image_pad|>")
                    elif item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
            else:
                parts.append(str(content))
        text = "".join(parts)
        if not tokenize:
            return text
        ids: list[int] = []
        cursor = 0
        image_token = "<|image_pad|>"
        for match in re.finditer(re.escape(image_token), text):
            ids.extend(self.encode(text[cursor : match.start()]))
            ids.append(self.image_token_id)
            cursor = match.end()
        ids.extend(self.encode(text[cursor:]))
        return ids

    def _coord_ids(self, coord_bin: int) -> list[int]:
        return [int(coord_bin)]


class MultiTokenCoordTokenizer(FakeTokenizer):
    def _coord_ids(self, coord_bin: int) -> list[int]:
        return [int(coord_bin), int(coord_bin) + 1_000]


class NoisyNonCoordDriftTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: object | None = None,
    ) -> list[int] | str:
        rendered = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
        )
        if not tokenize:
            return rendered
        assistant_text = str(messages[-1]["content"])
        if "<|coord_100|>" not in assistant_text:
            ids = list(rendered)
            for index, token_id in enumerate(ids):
                if int(token_id) >= 10_000 and int(token_id) != self.image_token_id:
                    ids[index] = int(token_id) + 1
                    break
            return ids
        return rendered


class FakeTemplate:
    tokenizer_cls = FakeTokenizer

    def __init__(self) -> None:
        self.tokenizer = self.tokenizer_cls()
        self.image_processor = SimpleNamespace(patch_size=1000, merge_size=1000)

    def encode(
        self, payload: Mapping[str, Any], *args: object, **kwargs: object
    ) -> dict[str, Any]:
        del args, kwargs
        input_ids = self.tokenizer.apply_chat_template(
            list(payload["messages"]),
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        return {
            "input_ids": list(input_ids),
            "labels": list(input_ids),
            "attention_mask": [1 for _ in input_ids],
        }


class MultiTokenCoordTemplate(FakeTemplate):
    tokenizer_cls = MultiTokenCoordTokenizer


class NoisyNonCoordDriftTemplate(FakeTemplate):
    tokenizer_cls = NoisyNonCoordDriftTokenizer


class LegacyAssistantOnlyTokenizer(FakeTokenizer):
    apply_chat_template = None


class LegacyAssistantOnlyTemplate(FakeTemplate):
    tokenizer_cls = LegacyAssistantOnlyTokenizer


class ShortAttentionMaskTemplate(FakeTemplate):
    def encode(
        self, payload: Mapping[str, Any], *args: object, **kwargs: object
    ) -> dict[str, Any]:
        encoded = super().encode(payload, *args, **kwargs)
        encoded["attention_mask"] = encoded["attention_mask"][:-1]
        return encoded


def _row(*, objects: list[dict[str, object]] | None = None) -> dict[str, object]:
    if objects is None:
        objects = [
            {
                "desc": "red box",
                "bbox_2d": [100, 100, 200, 220],
                "category_id": 1,
                "category_name": "red box",
                "coco_ann_id": 11,
                "object_id": "red-1",
            },
            {
                "desc": "blue box",
                "bbox_2d": [300, 320, 420, 470],
                "category_id": 2,
                "category_name": "blue box",
                "coco_ann_id": 22,
                "object_id": "blue-1",
            },
        ]
    return {
        "images": ["dummy.jpg"],
        "objects": objects,
        "width": 1000,
        "height": 1000,
        "image_id": 123,
        "file_name": "dummy.jpg",
        "metadata": {"source": "unit", "split": "train"},
    }


def _touch_image(root: Path) -> None:
    (root / "dummy.jpg").write_bytes(b"not-a-real-image")


def test_hybrid_builder_emits_two_segments_and_clean_labels(tmp_path: Path) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping(
        {
            "enabled": True,
            "current_object_kl": {
                "weight": 0.0,
                "window_radius": 8,
                "num_objects_per_image": 1,
            },
        }
    )

    sample = build_hybrid_prefix_denoising_sample(
        _row(),
        base_sample_id="unit-0",
        image_root=tmp_path,
        swift_template=FakeTemplate(),
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=0,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is True
    assert sample.clean_full is not None
    assert sample.noisy_full is not None
    assert sample.clean_full.branch_id == "clean_full"
    assert sample.noisy_full.branch_id == "noisy_full"
    assert sample.clean_full.labels == sample.noisy_full.labels
    assert len(sample.clean_full.input_ids) == len(sample.noisy_full.input_ids)
    diff_positions = {
        index
        for index, (clean_id, noisy_id) in enumerate(
            zip(sample.clean_full.input_ids, sample.noisy_full.input_ids, strict=True)
        )
        if clean_id != noisy_id
    }
    coord_positions = {
        index
        for index, label in enumerate(sample.clean_full.labels)
        if 0 <= int(label) <= 999
    }
    assert diff_positions == coord_positions
    assert sample.kl_sites == ()


def test_fast_packing_estimate_matches_full_hybrid_length(tmp_path: Path) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping({"enabled": True})
    template = FakeTemplate()
    row = _row()
    rng_seed = 5

    sample = build_hybrid_prefix_denoising_sample(
        row,
        base_sample_id="unit-0",
        image_root=tmp_path,
        swift_template=template,
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=0,
        rng=random.Random(rng_seed),
        max_length=12000,
    )
    estimate = estimate_hybrid_prefix_denoising_packing(
        row,
        image_root=tmp_path,
        swift_template=template,
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        rng=random.Random(rng_seed),
        max_length=12000,
    )

    assert sample.ok is True
    assert estimate.ok is True
    assert estimate.total_length == sample.total_length


def test_hybrid_builder_accepts_coord_token_bbox_rows(tmp_path: Path) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping({"enabled": True})
    row = _row(
        objects=[
            {
                "desc": "token box",
                "bbox_2d": [
                    "<|coord_100|>",
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_220|>",
                ],
                "category_id": 1,
                "category_name": "token box",
                "coco_ann_id": 11,
                "object_id": "token-1",
            }
        ]
    )

    sample = build_hybrid_prefix_denoising_sample(
        row,
        base_sample_id="unit-token-box",
        image_root=tmp_path,
        swift_template=FakeTemplate(),
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=0,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is True
    assert sample.clean_full is not None
    assert sample.noisy_full is not None
    noising = sample.metadata["noising"][0]
    assert noising["clean_bins"] == (100, 100, 200, 220)
    assert noising["noisy_bins"] != noising["clean_bins"]


def test_hybrid_builder_builds_kl_sites_when_weight_positive(tmp_path: Path) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping(
        {
            "enabled": True,
            "current_object_kl": {
                "weight": 0.05,
                "window_radius": 8,
                "num_objects_per_image": 1,
            },
        }
    )

    sample = build_hybrid_prefix_denoising_sample(
        _row(),
        base_sample_id="unit-0",
        image_root=tmp_path,
        swift_template=FakeTemplate(),
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=2,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is True
    assert len(sample.kl_sites) == 4
    assert tuple(site.coord_slot for site in sample.kl_sites) == (
        "x1",
        "y1",
        "x2",
        "y2",
    )
    for site in sample.kl_sites:
        assert site.clean_gt_bin in site.support_bins
        assert sample.clean_full is not None
        assert sample.noisy_full is not None
        assert sample.clean_full.labels[site.clean_label_position] == site.clean_gt_bin
        assert (
            sample.clean_full.input_ids[site.clean_label_position]
            == site.clean_gt_bin
        )
        assert sample.noisy_full.labels[site.noisy_label_position] == site.clean_gt_bin
        assert (
            sample.noisy_full.input_ids[site.noisy_label_position]
            != site.clean_gt_bin
        )


def test_hybrid_builder_excludes_zero_object_rows_with_counter_reason(
    tmp_path: Path,
) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping({"enabled": True})

    sample = build_hybrid_prefix_denoising_sample(
        _row(objects=[]),
        base_sample_id="unit-empty",
        image_root=tmp_path,
        swift_template=FakeTemplate(),
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=0,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is False
    assert sample.skip_reason == "zero_object_hybrid_sample"
    assert sample.total_length == 0


def test_dataset_construction_raises_on_malformed_row(tmp_path: Path) -> None:
    _touch_image(tmp_path)
    malformed = _row()
    malformed["objects"] = [
        {
            "desc": "bad box",
            "bbox_2d": [100, 100, 200],
            "category_id": 1,
            "category_name": "bad box",
            "coco_ann_id": 33,
            "object_id": "bad-1",
        }
    ]

    with pytest.raises(ValueError, match="bbox_2d|coordinate"):
        PrefixDenoisingTrainingDataset(
            [_row(), malformed],
            swift_template=FakeTemplate(),
            image_root=tmp_path,
            user_prompt="find objects",
            system_prompt=None,
            prefix_denoising=PrefixDenoisingConfig.from_mapping({"enabled": True}),
            max_length=12000,
            dataset_name="unit",
            seed=17,
        )


def test_noising_infeasible_rows_are_counted_as_sample_policy_skips(
    tmp_path: Path,
) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping(
        {
            "enabled": True,
            "noise": {"center_shift_frac": 0.0, "uniform_scale_range": [1.0, 1.0]},
        }
    )
    tiny_box_row = _row(
        objects=[
            {
                "desc": "tiny box",
                "bbox_2d": [0, 0, 1, 1],
                "category_id": 1,
                "category_name": "tiny box",
                "coco_ann_id": 44,
                "object_id": "tiny-1",
            }
        ]
    )

    eligibility = build_prefix_denoising_eligibility_index(
        [tiny_box_row],
        swift_template=FakeTemplate(),
        image_root=tmp_path,
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        max_length=12000,
        dataset_name="unit",
        seed=17,
    )

    assert eligibility["eligible_indices"] == ()
    assert eligibility["skip_counters"] == {"noise_infeasible_4coord_changed": 1}


def test_eligibility_precompute_does_not_materialize_full_hybrid_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping({"enabled": True})

    def _unexpected_full_build(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "eligibility precompute should use metadata/text length estimation, "
            "not full branch materialization"
        )

    monkeypatch.setattr(
        prefix_dataset_mod,
        "build_hybrid_prefix_denoising_sample",
        _unexpected_full_build,
    )

    eligibility = build_prefix_denoising_eligibility_index(
        [_row()],
        swift_template=FakeTemplate(),
        image_root=tmp_path,
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        max_length=12000,
        dataset_name="unit",
        seed=17,
    )

    assert eligibility["eligible_indices"] == (0,)
    assert eligibility["skip_counters"] == {}
    assert eligibility["static_lengths"][0] > 0


@pytest.mark.parametrize(
    "bbox",
    [
        [100, 100, 100, 120],
        [100, 100, 120, 100],
    ],
)
def test_degenerate_gt_boxes_are_counted_as_sample_policy_skips(
    tmp_path: Path,
    bbox: list[int],
) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping({"enabled": True})
    row = _row(
        objects=[
            {
                "desc": "degenerate box",
                "bbox_2d": bbox,
                "category_id": 1,
                "category_name": "degenerate box",
                "coco_ann_id": 55,
                "object_id": "degenerate-1",
            }
        ]
    )

    sample = build_hybrid_prefix_denoising_sample(
        row,
        base_sample_id="unit-degenerate",
        image_root=tmp_path,
        swift_template=FakeTemplate(),
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=0,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is False
    assert sample.skip_reason == "degenerate_gt_bbox"


def test_dataset_logs_visible_skip_counters_for_ineligible_rows(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _touch_image(tmp_path)
    cfg = PrefixDenoisingConfig.from_mapping({"enabled": True})

    caplog.set_level(
        logging.WARNING,
        logger="src.detection.prefix_denoising.dataset",
    )
    dataset = PrefixDenoisingTrainingDataset(
        [_row(), _row(objects=[])],
        swift_template=FakeTemplate(),
        image_root=tmp_path,
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=cfg,
        max_length=12000,
        dataset_name="unit",
        seed=17,
    )

    assert len(dataset) == 1
    assert dataset.skip_counters == {"zero_object_hybrid_sample": 1}
    assert dataset.prefix_denoising_dataset_summary() == {
        "dataset_name": "unit",
        "source_rows": 2,
        "eligible_rows": 1,
        "skipped_rows": 1,
        "skip_counters": {"zero_object_hybrid_sample": 1},
    }
    messages = [record.getMessage() for record in caplog.records]
    assert any("prefix-denoising skipped rows" in message for message in messages)
    assert any("zero_object_hybrid_sample" in message for message in messages)


def test_multitoken_coord_tokenizer_returns_alignment_skip(tmp_path: Path) -> None:
    _touch_image(tmp_path)

    sample = build_hybrid_prefix_denoising_sample(
        _row(),
        base_sample_id="unit-0",
        image_root=tmp_path,
        swift_template=MultiTokenCoordTemplate(),
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=PrefixDenoisingConfig.from_mapping({"enabled": True}),
        epoch=0,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is False
    assert sample.skip_reason == "coord_label_position_alignment_failed"


def test_noncoord_noisy_input_drift_is_rejected(tmp_path: Path) -> None:
    _touch_image(tmp_path)

    sample = build_hybrid_prefix_denoising_sample(
        _row(),
        base_sample_id="unit-0",
        image_root=tmp_path,
        swift_template=NoisyNonCoordDriftTemplate(),
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=PrefixDenoisingConfig.from_mapping({"enabled": True}),
        epoch=0,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is False
    assert sample.skip_reason == "clean_noisy_noncoord_alignment_failed"


def test_fast_packing_estimate_rejects_noncoord_noisy_input_drift(
    tmp_path: Path,
) -> None:
    _touch_image(tmp_path)

    estimate = estimate_hybrid_prefix_denoising_packing(
        _row(),
        image_root=tmp_path,
        swift_template=NoisyNonCoordDriftTemplate(),
        user_prompt="find objects",
        system_prompt=None,
        prefix_denoising=PrefixDenoisingConfig.from_mapping({"enabled": True}),
        rng=random.Random(5),
        max_length=12000,
    )

    assert estimate.ok is False
    assert estimate.skip_reason == "clean_noisy_noncoord_alignment_failed"


def test_fast_packing_estimate_ignores_coord_tokens_in_prompt(tmp_path: Path) -> None:
    _touch_image(tmp_path)

    estimate = estimate_hybrid_prefix_denoising_packing(
        _row(),
        image_root=tmp_path,
        swift_template=FakeTemplate(),
        user_prompt="format example <|coord_100|> before real answer",
        system_prompt=None,
        prefix_denoising=PrefixDenoisingConfig.from_mapping({"enabled": True}),
        rng=random.Random(5),
        max_length=12000,
    )

    assert estimate.ok is True


def test_fast_packing_estimate_requires_qwen_chat_template(tmp_path: Path) -> None:
    _touch_image(tmp_path)

    with pytest.raises(ValueError, match="apply_chat_template"):
        estimate_hybrid_prefix_denoising_packing(
            _row(),
            image_root=tmp_path,
            swift_template=LegacyAssistantOnlyTemplate(),
            user_prompt="find objects",
            system_prompt=None,
            prefix_denoising=PrefixDenoisingConfig.from_mapping({"enabled": True}),
            rng=random.Random(5),
            max_length=12000,
        )


def test_short_attention_mask_raises_template_contract_error(tmp_path: Path) -> None:
    _touch_image(tmp_path)

    with pytest.raises(ValueError, match="attention_mask.*length"):
        build_hybrid_prefix_denoising_sample(
            _row(),
            base_sample_id="unit-0",
            image_root=tmp_path,
            swift_template=ShortAttentionMaskTemplate(),
            user_prompt="find objects",
            system_prompt=None,
            prefix_denoising=PrefixDenoisingConfig.from_mapping({"enabled": True}),
            epoch=0,
            rng=random.Random(5),
            max_length=12000,
        )

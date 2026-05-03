from __future__ import annotations

import re

import pytest

from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.objective import prepare_detection_training_example
from src.detection.template import CompactFullTemplate, Stage1JsonPrettyTemplate

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class SpecialTokenAwareTokenizer:
    def __init__(self) -> None:
        self.chat_template_calls = 0
        self.encode_calls = 0
        self._token_to_id: dict[str, int] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        self.chat_template_calls += 1
        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
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
        self.encode_calls += 1

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
            token_id = self._token_to_id.setdefault(token_text, len(self._token_to_id) + 1)
            input_ids.append(token_id)
            offsets.append((cursor, token_end))
            cursor = token_end

        return {
            "input_ids": input_ids,
            "offset_mapping": offsets,
        }


def _sample(*, strategy: str = "random_permutation") -> NormalizedDetectionSample:
    ordering = (
        ObjectOrderingPlan.random_permutation(seed=19, seed_source="unit-test")
        if strategy == "random_permutation"
        else ObjectOrderingPlan.sorted(seed_source="unit-test")
    ).with_realized((7, 3))
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=7,
                object_instance_id="img-9:ann-501:src-7",
                desc="cat",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ),
                category_id=1,
                category_name="cat",
                coco_ann_id=501,
            ),
            NormalizedDetectionObject(
                normalized_object_index=1,
                source_object_index=3,
                object_instance_id="img-9:ann-502:src-3",
                desc="dog",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_300|>",
                    "<|coord_400|>",
                ),
                category_id=2,
                category_name="dog",
                coco_ann_id=502,
            ),
        ),
        width=640,
        height=480,
        image_id=9,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ordering,
    )


def test_random_permutation_et_rmp_ce_prepares_full_sequence_once_and_attaches_recursive_metadata() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    prepared = prepare_detection_training_example(
        _sample(),
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    assert tokenizer.chat_template_calls == 1
    assert tokenizer.encode_calls == 1
    assert prepared.mode == "random_permutation_et_rmp_ce"
    assert prepared.recursive_detection_targets is not None
    assert tuple(
        target.position for target in prepared.recursive_detection_targets.token_targets
    ) == tuple(
        index for index, label in enumerate(prepared.labels) if label != -100
    )
    assert prepared.labels == prepared.tokenized.labels
    assert prepared.assistant_mask == prepared.tokenized.assistant_mask


def test_recursive_mode_requires_random_permutation_sample_ordering() -> None:
    with pytest.raises(
        ValueError,
        match="random_permutation_et_rmp_ce requires .*strategy='random_permutation'",
    ):
        prepare_detection_training_example(
            _sample(strategy="sorted"),
            template=CompactFullTemplate(),
            tokenizer=SpecialTokenAwareTokenizer(),
            mode="random_permutation_et_rmp_ce",
        )


def test_sft_modes_do_not_attach_recursive_metadata() -> None:
    sorted_prepared = prepare_detection_training_example(
        _sample(strategy="sorted"),
        template=CompactFullTemplate(),
        tokenizer=SpecialTokenAwareTokenizer(),
        mode="sorted_sft",
    )
    random_prepared = prepare_detection_training_example(
        _sample(),
        template=CompactFullTemplate(),
        tokenizer=SpecialTokenAwareTokenizer(),
        mode="random_order_sft",
    )

    assert sorted_prepared.recursive_detection_targets is None
    assert random_prepared.recursive_detection_targets is None


def test_both_templates_support_random_permutation_et_rmp_ce_preparation() -> None:
    sample = _sample()

    for template in (CompactFullTemplate(), Stage1JsonPrettyTemplate()):
        prepared = prepare_detection_training_example(
            sample,
            template=template,
            tokenizer=SpecialTokenAwareTokenizer(),
            mode="random_permutation_et_rmp_ce",
        )

        assert prepared.template_id == template.template_id
        assert prepared.recursive_detection_targets is not None

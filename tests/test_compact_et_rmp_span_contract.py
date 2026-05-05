from __future__ import annotations

import re

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.objective import prepare_detection_training_example
from src.detection.template import CompactFullTemplate
from src.detection.tokenization import TokenRole

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class SpecialTokenAwareTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
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
            self._id_to_token.setdefault(token_id, token_text)
            input_ids.append(token_id)
            offsets.append((cursor, token_end))
            cursor = token_end

        return {
            "input_ids": input_ids,
            "offset_mapping": offsets,
        }

    def token_text(self, token_id: int) -> str:
        return self._id_to_token[token_id]


def _sample() -> NormalizedDetectionSample:
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
        object_ordering=ObjectOrderingPlan.random_permutation(
            seed=13,
            seed_source="unit-test",
        ).with_realized((7, 3)),
    )


def test_compact_recursive_targets_preserve_span_roles_and_separator_hard_ce() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    prepared = prepare_detection_training_example(
        _sample(),
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    assert prepared.recursive_detection_targets is not None
    targets = {
        target.position: target
        for target in prepared.recursive_detection_targets.token_targets
    }
    first_entry = prepared.tokenized.object_entries[0]

    object_ref_target = targets[first_entry.object_ref_start_span.start]
    assert object_ref_target.kind == "hard_ce"
    assert object_ref_target.token_role is TokenRole.CONTROL
    assert tuple(tokenizer.token_text(token_id) for token_id in object_ref_target.valid_token_ids) == (
        OBJECT_REF_START_TOKEN,
    )

    bbox_start_target = targets[first_entry.bbox_start_span.start]
    assert bbox_start_target.kind == "hard_ce"
    assert bbox_start_target.token_role is TokenRole.BBOX_START
    assert tuple(tokenizer.token_text(token_id) for token_id in bbox_start_target.valid_token_ids) == (
        BOX_START_TOKEN,
    )

    separator_target = targets[first_entry.separator_span.start]
    assert separator_target.kind == "hard_ce"
    assert separator_target.token_role is TokenRole.SEPARATOR
    assert separator_target.object_instance_id is None
    assert tuple(tokenizer.token_text(token_id) for token_id in separator_target.valid_token_ids) == (
        "\n",
    )


def test_compact_recursive_targets_leave_last_entry_singleton_hard_ce() -> None:
    prepared = prepare_detection_training_example(
        _sample(),
        template=CompactFullTemplate(),
        tokenizer=SpecialTokenAwareTokenizer(),
        mode="random_permutation_et_rmp_ce",
    )

    assert prepared.recursive_detection_targets is not None
    targets = {
        target.position: target
        for target in prepared.recursive_detection_targets.token_targets
    }
    second_entry = prepared.tokenized.object_entries[1]

    for position in range(second_entry.entry_span.start, second_entry.entry_span.end):
        assert targets[position].kind == "hard_ce"

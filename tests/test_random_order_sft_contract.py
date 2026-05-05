import pytest

from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    ObjectOrderingPlan,
    RawDetectionObject,
    RawDetectionRow,
    normalize_detection_row,
)
from src.detection.objective import prepare_detection_training_example
from src.detection.template import CompactFullTemplate, Stage1JsonPrettyTemplate
from src.detection.tokenization import tokenize_rendered_detection_conversation


class CharOffsetTokenizer:
    def __init__(self) -> None:
        self.chat_template_calls = 0
        self.encode_calls = 0

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
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _raw_sample() -> RawDetectionRow:
    return RawDetectionRow(
        images=("image.jpg",),
        objects=(
            RawDetectionObject(
                source_object_index=0,
                desc="traffic light",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ),
                category_id=10,
                category_name="traffic light",
                coco_ann_id=501,
            ),
            RawDetectionObject(
                source_object_index=1,
                desc="person",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_300|>",
                    "<|coord_400|>",
                ),
                category_id=1,
                category_name="person",
                coco_ann_id=502,
            ),
            RawDetectionObject(
                source_object_index=2,
                desc="bicycle",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_500|>",
                    "<|coord_600|>",
                    "<|coord_700|>",
                    "<|coord_800|>",
                ),
                category_id=2,
                category_name="bicycle",
                coco_ann_id=503,
            ),
        ),
        width=640,
        height=480,
        image_id=9,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
    )


def _sorted_sample():
    return normalize_detection_row(
        _raw_sample(),
        object_ordering=ObjectOrderingPlan.sorted(seed_source="unit:source-order"),
    )


def _random_sample(*, seed: int):
    return normalize_detection_row(
        _raw_sample(),
        object_ordering=ObjectOrderingPlan.random_permutation(
            seed=seed,
            seed_source="unit-test",
        ),
    )


def test_random_order_sft_differs_only_by_realized_object_permutation() -> None:
    template = CompactFullTemplate()
    sorted_sample = _sorted_sample()
    random_sample = _random_sample(seed=7)

    sorted_prepared = prepare_detection_training_example(
        sorted_sample,
        template=template,
        tokenizer=CharOffsetTokenizer(),
        mode="sorted_sft",
    )
    random_prepared = prepare_detection_training_example(
        random_sample,
        template=template,
        tokenizer=CharOffsetTokenizer(),
        mode="random_order_sft",
    )

    assert random_prepared.mode == "random_order_sft"
    assert random_prepared.object_ordering.strategy == "random_permutation"
    assert random_prepared.object_ordering.seed == 7
    assert random_prepared.object_ordering.seed_source == "unit-test"
    assert random_prepared.template_id == sorted_prepared.template_id
    assert random_prepared.labels == random_prepared.tokenized.labels
    assert sorted_prepared.labels == sorted_prepared.tokenized.labels
    assert sorted_prepared.realized_source_object_indices == (0, 1, 2)
    assert random_prepared.realized_source_object_indices != sorted_prepared.realized_source_object_indices
    assert sorted(random_prepared.realized_source_object_indices) == [0, 1, 2]
    assert [obj.source_object_index for obj in random_prepared.normalized_sample.objects] == list(
        random_prepared.realized_source_object_indices
    )
    assert [entry.source_object_index for entry in random_prepared.rendered_assistant.object_entries] == list(
        random_prepared.realized_source_object_indices
    )

    expected = tokenize_rendered_detection_conversation(
        template.render_assistant(random_prepared.normalized_sample),
        tokenizer=CharOffsetTokenizer(),
    )
    assert random_prepared.labels == expected.labels
    assert random_prepared.assistant_mask == expected.assistant_mask
    assert random_prepared.tokenized.token_roles == expected.token_roles


def test_random_order_sft_fixed_seed_is_deterministic_and_different_seed_can_change_order() -> None:
    template = CompactFullTemplate()

    prepared_a = prepare_detection_training_example(
        _random_sample(seed=7),
        template=template,
        tokenizer=CharOffsetTokenizer(),
        mode="random_order_sft",
    )
    prepared_b = prepare_detection_training_example(
        _random_sample(seed=7),
        template=template,
        tokenizer=CharOffsetTokenizer(),
        mode="random_order_sft",
    )
    prepared_c = prepare_detection_training_example(
        _random_sample(seed=11),
        template=template,
        tokenizer=CharOffsetTokenizer(),
        mode="random_order_sft",
    )

    assert prepared_a.realized_source_object_indices == prepared_b.realized_source_object_indices
    assert prepared_a.rendered_assistant.text == prepared_b.rendered_assistant.text
    assert prepared_c.realized_source_object_indices != prepared_a.realized_source_object_indices


def test_both_templates_can_prepare_random_order_sft_examples() -> None:
    sample = _random_sample(seed=7)

    for template in (CompactFullTemplate(), Stage1JsonPrettyTemplate()):
        prepared = prepare_detection_training_example(
            sample,
            template=template,
            tokenizer=CharOffsetTokenizer(),
            mode="random_order_sft",
        )

        assert prepared.rendered_assistant.template_id == template.template_id
        assert prepared.object_ordering.strategy == "random_permutation"
        assert sorted(prepared.realized_source_object_indices) == [0, 1, 2]


def test_sft_preparation_fails_fast_on_mode_order_mismatch() -> None:
    sorted_sample = _sorted_sample()
    random_ordered_sample = _random_sample(seed=7)

    with pytest.raises(ValueError, match="sorted_sft requires sample.object_ordering.strategy='sorted'"):
        prepare_detection_training_example(
            random_ordered_sample,
            template=CompactFullTemplate(),
            tokenizer=CharOffsetTokenizer(),
            mode="sorted_sft",
        )

    with pytest.raises(
        ValueError,
        match="random_order_sft requires .*strategy='random_permutation'",
    ):
        prepare_detection_training_example(
            sorted_sample,
            template=CompactFullTemplate(),
            tokenizer=CharOffsetTokenizer(),
            mode="random_order_sft",
        )

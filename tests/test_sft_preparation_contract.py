from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.objective import (
    PreparedDetectionExample,
    prepare_detection_training_example,
)
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


def _sample() -> NormalizedDetectionSample:
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=7,
                object_instance_id="img-9:ann-501:src-7",
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
            NormalizedDetectionObject(
                normalized_object_index=1,
                source_object_index=3,
                object_instance_id="img-9:ann-502:src-3",
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
        ),
        width=640,
        height=480,
        image_id=9,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((7, 3)),
    )


def _assert_matches_phase3_hard_ce(
    prepared: PreparedDetectionExample,
    *,
    template,
    tokenizer: CharOffsetTokenizer,
) -> None:
    expected = tokenize_rendered_detection_conversation(
        template.render_assistant(prepared.normalized_sample),
        tokenizer=tokenizer,
    )

    assert prepared.input_ids == expected.input_ids
    assert prepared.labels == expected.labels
    assert prepared.assistant_mask == expected.assistant_mask
    assert prepared.tokenized.labels == expected.labels
    assert prepared.tokenized.assistant_mask == expected.assistant_mask


def test_phase3_hard_ce_labels_supervise_assistant_tokens_and_chat_stop() -> None:
    tokenizer = CharOffsetTokenizer()
    prepared = prepare_detection_training_example(
        _sample(),
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="sorted_sft",
    )

    assert prepared.tokenized.assistant_stop_token_span is not None

    for index, token_id in enumerate(prepared.input_ids):
        is_assistant_content = (
            prepared.tokenized.assistant_token_span.start
            <= index
            < prepared.tokenized.assistant_token_span.end
        )
        is_assistant_stop = (
            prepared.tokenized.assistant_stop_token_span.start
            <= index
            < prepared.tokenized.assistant_stop_token_span.end
        )
        if is_assistant_content or is_assistant_stop:
            assert prepared.labels[index] == token_id
        else:
            assert prepared.labels[index] == -100

        if is_assistant_content:
            assert prepared.assistant_mask[index] is True
        else:
            assert prepared.assistant_mask[index] is False


def test_sorted_sft_is_hard_ce_and_preserves_source_order_metadata() -> None:
    sample = _sample()
    tokenizer = CharOffsetTokenizer()

    prepared = prepare_detection_training_example(
        sample,
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="sorted_sft",
    )

    assert tokenizer.chat_template_calls == 1
    assert tokenizer.encode_calls == 1
    assert prepared.mode == "sorted_sft"
    assert prepared.normalized_sample is sample
    assert prepared.object_ordering.strategy == "sorted"
    assert prepared.object_ordering.seed is None
    assert prepared.realized_source_object_indices == (7, 3)
    assert [obj.source_object_index for obj in prepared.normalized_sample.objects] == [7, 3]
    assert [entry.source_object_index for entry in prepared.rendered_assistant.object_entries] == [7, 3]
    _assert_matches_phase3_hard_ce(
        prepared,
        template=CompactFullTemplate(),
        tokenizer=CharOffsetTokenizer(),
    )


def test_both_templates_can_prepare_sorted_sft_examples() -> None:
    sample = _sample()

    for template in (CompactFullTemplate(), Stage1JsonPrettyTemplate()):
        tokenizer = CharOffsetTokenizer()
        prepared = prepare_detection_training_example(
            sample,
            template=template,
            tokenizer=tokenizer,
            mode="sorted_sft",
        )

        assert prepared.rendered_assistant.template_id == template.template_id
        assert prepared.template_id == template.template_id
        assert prepared.realized_source_object_indices == (7, 3)
        _assert_matches_phase3_hard_ce(
            prepared,
            template=template,
            tokenizer=CharOffsetTokenizer(),
        )


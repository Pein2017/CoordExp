from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import replace
from types import MappingProxyType
import re

import pytest

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.template import CompactFullTemplate
from src.detection.tokenization import TokenRole, TokenSpan
from src.training.encoding.view import (
    CoordinateSlot,
    EncodedDetectionView,
    EncodedObjectEntry,
)
from src.detection.tokenization import tokenize_rendered_detection_conversation
from src.training.templates.compact_full import create_compact_full_codec
from src.training.templates.codec import DetectionTemplateRenderOptions


class SnapshotTokenizer:
    """Deterministic tokenizer stub that exposes offset mappings."""

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
        token_pattern = re.compile(
            r"<\|object_ref_start\|>|<\|box_start\|>|<\|coord_\d+\|>|"
            r"<\|im_end\|>|"
            r"[A-Za-z0-9_]+|"
            r"\s|"
            r".",
            re.DOTALL,
        )
        matches = list(token_pattern.finditer(text))
        token_texts = [match.group(0) for match in matches]
        vocab = {token: index + 1 for index, token in enumerate(dict.fromkeys(token_texts))}
        return {
            "input_ids": [vocab[token] for token in token_texts],
            "offset_mapping": [(match.start(), match.end()) for match in matches],
        }


def _three_object_sample() -> NormalizedDetectionSample:
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=2,
                object_instance_id="img-42:ann-100:src-2",
                desc="traffic light",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_0|>",
                    "<|coord_1|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                ),
                category_id=10,
                category_name="traffic light",
                coco_ann_id=100,
            ),
            NormalizedDetectionObject(
                normalized_object_index=1,
                source_object_index=4,
                object_instance_id="img-42:ann-101:src-4",
                desc="red car",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_22|>",
                    "<|coord_31|>",
                    "<|coord_80|>",
                    "<|coord_90|>",
                ),
                category_id=3,
                category_name="car",
                coco_ann_id=101,
            ),
            NormalizedDetectionObject(
                normalized_object_index=2,
                source_object_index=7,
                object_instance_id="img-42:ann-102:src-7",
                desc="person",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_900|>",
                    "<|coord_910|>",
                    "<|coord_998|>",
                    "<|coord_999|>",
                ),
                category_id=1,
                category_name="person",
                coco_ann_id=102,
            ),
        ),
        width=1024,
        height=1024,
        image_id=42,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((2, 4, 7)),
    )


def _span_text(tokenized, token_span) -> str:
    start = tokenized.offset_mapping[token_span.start][0]
    end = tokenized.offset_mapping[token_span.end - 1][1]
    return tokenized.chat_text[start:end]


def _encoded_view() -> EncodedDetectionView:
    return create_compact_full_codec().encode_sample(
        _three_object_sample(),
        tokenizer=SnapshotTokenizer(),
    )


def test_compact_full_encoding_view_matches_template_and_tokenization_golden() -> None:
    sample = _three_object_sample()
    tokenizer = SnapshotTokenizer()
    rendered = CompactFullTemplate().render_assistant(sample)
    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=tokenizer,
    )

    encoded = create_compact_full_codec().encode_sample(sample, tokenizer=tokenizer)

    assert encoded.rendered_assistant_text == (
        f"{OBJECT_REF_START_TOKEN}traffic light{BOX_START_TOKEN}"
        "<|coord_0|><|coord_1|><|coord_20|><|coord_30|>\n"
        f"{OBJECT_REF_START_TOKEN}red car{BOX_START_TOKEN}"
        "<|coord_22|><|coord_31|><|coord_80|><|coord_90|>\n"
        f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
        "<|coord_900|><|coord_910|><|coord_998|><|coord_999|>"
    )
    assert encoded.rendered_assistant_text == rendered.text

    assert encoded.input_ids == tokenized.input_ids
    assert encoded.labels == tokenized.labels
    assert encoded.offset_mapping == tokenized.offset_mapping
    assert encoded.token_roles == tokenized.token_roles
    assert encoded.label_positions == tokenized.supervised_label_positions
    assert encoded.label_positions

    assert encoded.schema_spans == tokenized.structural_spans
    assert encoded.description_spans == tuple(
        entry.desc_span for entry in tokenized.object_entries
    )
    assert [_span_text(tokenized, span) for span in encoded.description_spans] == [
        "traffic light",
        "red car",
        "person",
    ]

    assert [
        (entry.object_instance_id, entry.object_index, entry.source_object_index)
        for entry in encoded.object_entries
    ] == [
        ("img-42:ann-100:src-2", 0, 2),
        ("img-42:ann-101:src-4", 1, 4),
        ("img-42:ann-102:src-7", 2, 7),
    ]

    assert [
        (slot.object_index, slot.slot_name, _span_text(tokenized, slot.token_span))
        for slot in encoded.coordinate_slots
    ] == [
        (0, "x1", "<|coord_0|>"),
        (0, "y1", "<|coord_1|>"),
        (0, "x2", "<|coord_20|>"),
        (0, "y2", "<|coord_30|>"),
        (1, "x1", "<|coord_22|>"),
        (1, "y1", "<|coord_31|>"),
        (1, "x2", "<|coord_80|>"),
        (1, "y2", "<|coord_90|>"),
        (2, "x1", "<|coord_900|>"),
        (2, "y1", "<|coord_910|>"),
        (2, "x2", "<|coord_998|>"),
        (2, "y2", "<|coord_999|>"),
    ]

    for forbidden_field in (
        "assignment_result",
        "duplicate_filter_result",
        "loss",
        "logits",
        "model_output",
        "objective",
        "target_plan",
    ):
        assert not hasattr(encoded, forbidden_field)


def test_training_import_contract_does_not_import_detection_or_backend_libraries() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        BLOCKED_FAMILIES = ("src.detection", "torch", "transformers")

        class BlockTrainingBoundaries(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in BLOCKED_FAMILIES or fullname.startswith(
                    tuple(f"{family}." for family in BLOCKED_FAMILIES)
                ):
                    raise RuntimeError(f"blocked import: {fullname}")
                return None

        sys.meta_path.insert(0, BlockTrainingBoundaries())

        import src.training.sidecars
        import src.training.encoding.model_inputs
        import src.training.templates.compact_full

        leaked = sorted(
            module_name
            for module_name in sys.modules
            if module_name in BLOCKED_FAMILIES
            or module_name.startswith(tuple(f"{family}." for family in BLOCKED_FAMILIES))
        )
        if leaked:
            raise RuntimeError(f"blocked modules leaked: {leaked}")

        print("import-ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "import-ok"


def test_detection_template_render_options_freeze_messages_and_stop_markers() -> None:
    messages = [
        {"role": "system", "content": "initial"},
        {"role": "user", "content": "<image>"},
    ]
    stop_markers = ["<|im_end|>"]

    options = DetectionTemplateRenderOptions(
        messages=messages,
        assistant_stop_markers=stop_markers,
    )
    messages.append({"role": "assistant", "content": "external"})
    messages[0]["content"] = "mutated"
    stop_markers.append("<extra>")

    assert options.assistant_stop_markers == ("<|im_end|>",)
    assert isinstance(options.messages, tuple)
    assert all(isinstance(message, MappingProxyType) for message in options.messages)
    assert tuple(dict(message) for message in options.messages) == (
        {"role": "system", "content": "initial"},
        {"role": "user", "content": "<image>"},
    )

    with pytest.raises(TypeError):
        options.messages[0]["content"] = "blocked"


@pytest.mark.parametrize(
    "kwargs,expected_error",
    [
        ({"coordinate_surface": b"coord_token"}, "coordinate_surface"),
        ({"bbox_format": b"xyxy"}, "bbox_format"),
        ({"prompt_template_id": 7}, "prompt_template_id"),
        ({"system_prompt": 7}, "system_prompt"),
        ({"user_content": 7}, "user_content"),
        ({"include_rendered_assistant_text": 1}, "include_rendered_assistant_text"),
        ({"assistant_stop_markers": "<|im_end|>"}, "assistant_stop_markers"),
        ({"assistant_stop_markers": [b"<|im_end|>"]}, "assistant_stop_markers"),
        ({"assistant_stop_markers": [""]}, "assistant_stop_markers"),
        ({"messages": {"role": "user"}}, "messages"),
        ({"messages": ["not-a-mapping"]}, "messages"),
        ({"messages": [{1: "user"}]}, "message keys"),
    ],
)
def test_detection_template_render_options_reject_malformed_scalar_and_container_types(
    kwargs,
    expected_error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected_error):
        DetectionTemplateRenderOptions(**kwargs)


@pytest.mark.parametrize(
    "label_positions,expected_error",
    [
        ((0,), "0 < pos < len"),
        ((10**9,), "0 < pos < len"),
        ((True,), "plain integers"),
        (("1",), "plain integers"),
        ({"position": 1}, "label_positions"),
    ],
)
def test_encoded_detection_view_rejects_malformed_label_positions(
    label_positions,
    expected_error: str,
) -> None:
    encoded = _encoded_view()

    with pytest.raises((TypeError, ValueError), match=expected_error):
        replace(encoded, label_positions=label_positions)


def test_encoded_detection_view_rejects_unsupervised_label_positions() -> None:
    encoded = _encoded_view()
    unsupervised_position = next(
        position
        for position, label in enumerate(encoded.labels)
        if position > 0 and label == -100
    )

    with pytest.raises(ValueError, match="supervised labels"):
        replace(encoded, label_positions=(unsupervised_position,))


def test_encoded_detection_view_rejects_malformed_coordinate_slot_counts() -> None:
    encoded = _encoded_view()

    with pytest.raises(ValueError, match="exactly four coordinate slots"):
        replace(encoded, coordinate_slots=encoded.coordinate_slots[:-1])

    with pytest.raises(ValueError, match="exactly four spans"):
        replace(
            encoded.object_entries[0],
            coordinate_spans=encoded.object_entries[0].coordinate_spans[:-1],
        )


@pytest.mark.parametrize(
    "field_name,value,expected_error",
    [
        ("input_ids", {"token": 1}, "input_ids"),
        ("labels", {"token": 1}, "labels"),
        ("schema_spans", {"span": object()}, "schema_spans"),
        ("object_entries", {"entry": object()}, "object_entries"),
        ("coordinate_slots", {"slot": object()}, "coordinate_slots"),
        ("token_roles", {"role": object()}, "token_roles"),
        ("object_entries", (object(),), "EncodedObjectEntry"),
        ("coordinate_slots", (object(),), "CoordinateSlot"),
        ("token_roles", (object(),), "TokenRole"),
        ("schema_spans", (object(),), "TokenSpan"),
        ("description_spans", (object(),), "TokenSpan"),
        ("assistant_token_span", object(), "TokenSpan"),
        ("assistant_stop_token_span", object(), "TokenSpan"),
        ("rendered_assistant_text", 123, "string or None"),
    ],
)
def test_encoded_detection_view_rejects_invalid_element_types(
    field_name: str,
    value,
    expected_error: str,
) -> None:
    encoded = _encoded_view()

    with pytest.raises(TypeError, match=expected_error):
        replace(encoded, **{field_name: value})


def test_encoded_detection_view_rejects_coordinate_slots_not_matching_entries() -> None:
    encoded = _encoded_view()
    first_slot = encoded.coordinate_slots[0]

    with pytest.raises(ValueError, match="ordered x1/y1/x2/y2"):
        replace(
            encoded,
            coordinate_slots=(
                replace(first_slot, slot_name="y1"),
                *encoded.coordinate_slots[1:],
            ),
        )

    with pytest.raises(ValueError, match="match object coordinate spans"):
        replace(
            encoded,
            coordinate_slots=(
                replace(first_slot, token_span=encoded.coordinate_slots[1].token_span),
                *encoded.coordinate_slots[1:],
            ),
        )


@pytest.mark.parametrize(
    "field_name,value,expected_error",
    [
        ("object_instance_id", 7, "plain string"),
        ("object_index", True, "plain integer"),
        ("source_object_index", -1, "nonnegative"),
        ("entry_span", object(), "TokenSpan"),
        ("description_span", object(), "TokenSpan"),
        ("coordinate_spans", (object(), object(), object(), object()), "TokenSpan"),
        ("schema_spans", (object(),), "TokenSpan"),
    ],
)
def test_encoded_object_entry_rejects_invalid_fields(
    field_name: str,
    value,
    expected_error: str,
) -> None:
    entry = _encoded_view().object_entries[0]

    with pytest.raises((TypeError, ValueError), match=expected_error):
        replace(entry, **{field_name: value})


@pytest.mark.parametrize(
    "field_name,value,expected_error",
    [
        ("object_instance_id", 7, "plain string"),
        ("object_index", True, "plain integer"),
        ("slot_index", True, "plain integer"),
        ("slot_index", -1, "nonnegative"),
        ("slot_index", 4, "x1/y1/x2/y2"),
        ("slot_name", b"x1", "plain string"),
        ("slot_name", "xmin", "x1/y1/x2/y2"),
        ("token_span", object(), "TokenSpan"),
    ],
)
def test_coordinate_slot_rejects_invalid_fields(
    field_name: str,
    value,
    expected_error: str,
) -> None:
    slot = _encoded_view().coordinate_slots[0]

    with pytest.raises((TypeError, ValueError), match=expected_error):
        replace(slot, **{field_name: value})


def test_encoded_contracts_accept_exact_detection_owner_types() -> None:
    span = TokenSpan(start=1, end=2, label="unit")
    entry = EncodedObjectEntry(
        object_instance_id="obj-1",
        object_index=0,
        source_object_index=0,
        entry_span=span,
        description_span=span,
        coordinate_spans=(span, span, span, span),
        schema_spans=(span,),
    )
    slots = tuple(
        CoordinateSlot(
            object_instance_id="obj-1",
            object_index=0,
            slot_index=index,
            slot_name=slot_name,
            token_span=span,
        )
        for index, slot_name in enumerate(("x1", "y1", "x2", "y2"))
    )

    view = EncodedDetectionView(
        template_id="unit",
        template_version=1,
        input_ids=(10, 11, 12),
        labels=(-100, 11, -100),
        offset_mapping=((0, 0), (0, 1), (1, 2)),
        assistant_token_span=span,
        assistant_stop_token_span=None,
        object_entries=(entry,),
        schema_spans=(span,),
        description_spans=(span,),
        coordinate_slots=slots,
        token_roles=(TokenRole.IGNORE, TokenRole.COORD, TokenRole.TERMINAL),
        label_positions=(1,),
        rendered_assistant_text=None,
    )

    assert view.object_entries == (entry,)

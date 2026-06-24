from __future__ import annotations

import re
from dataclasses import replace

import pytest

from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.template import get_detection_template
from src.detection.tokenization import tokenize_rendered_detection_conversation
from src.training.coverage_ledger import (
    CoverageLedgerObjectEntry,
    CoverageLedgerSidecar,
    build_coverage_ledger_sidecar,
)


class _LedgerTokenizer:
    eos_token = "<|im_end|>"

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
            r"<\|object_ref_start\|>|<\|object_ref_end\|>|"
            r"<\|box_start\|>|<\|box_end\|>|<\|coord_\d+\|>|"
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


def _sample() -> NormalizedDetectionSample:
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=7,
                object_instance_id="img-9:ann-501:src-7",
                desc="cat",
                bbox_2d=CoordinateTokenBox(10, 20, 300, 400),
                category_id=17,
                category_name="cat",
                coco_ann_id=501,
            ),
        ),
        width=640,
        height=480,
        image_id=9,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((7,)),
    )


def _tokenized(template_id: str):
    rendered = get_detection_template(template_id).render_assistant(_sample())
    return tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=_LedgerTokenizer(),
    )


def _with_coord_token_texts(coord_tokens: tuple[str, str, str, str]):
    tokenized = _tokenized("compact_object_box_closed")
    entry = tokenized.object_entries[0]
    token_texts = [
        tokenized.chat_text[start:end] for start, end in tokenized.offset_mapping
    ]
    for span, coord_token in zip(entry.coord_spans, coord_tokens, strict=True):
        token_texts[span.start] = coord_token

    cursor = 0
    offsets: list[tuple[int, int]] = []
    for token_text in token_texts:
        offsets.append((cursor, cursor + len(token_text)))
        cursor += len(token_text)

    return replace(
        tokenized,
        chat_text="".join(token_texts),
        offset_mapping=tuple(offsets),
    )


def _control_position(entry, label: str) -> int:
    matches = [span for span in entry.control_spans if span.label == label]
    assert len(matches) == 1
    assert matches[0].end - matches[0].start == 1
    return matches[0].start


def test_build_coverage_ledger_sidecar_from_tokenized_closed_template() -> None:
    tokenized = _tokenized("compact_object_box_closed")
    tokenized_entry = tokenized.object_entries[0]

    sidecar = build_coverage_ledger_sidecar(
        tokenized,
        sample_id="coco:0",
        image_grid_thw=(1, 16, 16),
        processed_width=640,
        processed_height=480,
        image_identity="image.jpg",
    )

    assert sidecar.sample_id == "coco:0"
    assert sidecar.prompt_end_position == tokenized_entry.entry_span.start - 1
    assert sidecar.image_grid_thw == (1, 16, 16)
    assert sidecar.processed_width == 640
    assert sidecar.processed_height == 480
    assert sidecar.image_identity == "image.jpg"

    assert len(sidecar.object_entries) == 1
    ledger_entry = sidecar.object_entries[0]
    assert ledger_entry.object_instance_id == "img-9:ann-501:src-7"
    assert ledger_entry.source_object_index == 7
    assert ledger_entry.emitted_order_index == 0
    assert ledger_entry.image_index == 0
    assert ledger_entry.bbox_norm1000_xyxy == (10, 20, 300, 400)
    assert ledger_entry.box_start_position == tokenized_entry.bbox_start_span.start
    assert ledger_entry.coord_label_positions == tuple(
        span.start for span in tokenized_entry.coord_spans
    )
    assert ledger_entry.object_ref_end_position == _control_position(
        tokenized_entry,
        "object_ref_end",
    )
    assert ledger_entry.box_end_position == _control_position(tokenized_entry, "box_end")


def test_build_coverage_ledger_sidecar_accepts_canonical_norm1000_edge() -> None:
    tokenized = _with_coord_token_texts(
        (
            "<|coord_0|>",
            "<|coord_0|>",
            "<|coord_999|>",
            "<|coord_999|>",
        ),
    )

    sidecar = build_coverage_ledger_sidecar(
        tokenized,
        sample_id="coco:0",
        image_grid_thw=(1, 16, 16),
        processed_width=640,
        processed_height=480,
        image_identity="image.jpg",
    )

    assert sidecar.object_entries[0].bbox_norm1000_xyxy == (0, 0, 999, 999)


@pytest.mark.parametrize(
    "coord_token",
    (
        "<|coord_1001|>",
        "<|coord_1000|>",
        "<|coord_001|>",
        "<|coord_+1|>",
        "<|coord_1.0|>",
    ),
)
def test_build_coverage_ledger_sidecar_rejects_invalid_coord_token_text(
    coord_token: str,
) -> None:
    tokenized = _with_coord_token_texts(
        ("<|coord_0|>", "<|coord_0|>", coord_token, "<|coord_999|>"),
    )

    with pytest.raises(ValueError, match="coord"):
        build_coverage_ledger_sidecar(
            tokenized,
            sample_id="coco:0",
            image_grid_thw=(1, 16, 16),
            processed_width=640,
            processed_height=480,
            image_identity="image.jpg",
        )


@pytest.mark.parametrize(
    ("template_id", "missing_label"),
    (
        ("compact", "box_end"),
        ("compact_box_closed", "object_ref_end"),
        ("compact_object_closed", "box_end"),
    ),
)
def test_coverage_ledger_sidecar_rejects_templates_without_required_control_tokens(
    template_id: str,
    missing_label: str,
) -> None:
    with pytest.raises(ValueError, match=missing_label):
        build_coverage_ledger_sidecar(
            _tokenized(template_id),
            sample_id="coco:0",
            image_grid_thw=(1, 16, 16),
            processed_width=640,
            processed_height=480,
            image_identity="image.jpg",
        )


def test_coverage_ledger_sidecar_validates_object_contract() -> None:
    entry = CoverageLedgerObjectEntry(
        object_instance_id="object-1",
        source_object_index=7,
        emitted_order_index=0,
        image_index=0,
        bbox_norm1000_xyxy=(10, 20, 300, 400),
        box_start_position=11,
        coord_label_positions=(12, 13, 14, 15),
        object_ref_end_position=10,
        box_end_position=16,
    )
    sidecar = CoverageLedgerSidecar(
        sample_id="coco:0",
        prompt_end_position=9,
        object_entries=(entry,),
        image_grid_thw=(1, 16, 16),
        processed_width=640,
        processed_height=480,
        image_identity="image.jpg",
    )
    assert sidecar.object_entries == (entry,)

    with pytest.raises(ValueError, match="object_entries.*non-empty"):
        CoverageLedgerSidecar(
            sample_id="coco:0",
            prompt_end_position=9,
            object_entries=(),
            image_grid_thw=(1, 16, 16),
            processed_width=640,
            processed_height=480,
            image_identity="image.jpg",
        )

    with pytest.raises(ValueError, match="emitted_order_index"):
        CoverageLedgerSidecar(
            sample_id="coco:0",
            prompt_end_position=9,
            object_entries=(replace(entry, emitted_order_index=1),),
            image_grid_thw=(1, 16, 16),
            processed_width=640,
            processed_height=480,
            image_identity="image.jpg",
        )

    with pytest.raises(ValueError, match="bbox_norm1000_xyxy"):
        CoverageLedgerSidecar(
            sample_id="coco:0",
            prompt_end_position=9,
            object_entries=(replace(entry, bbox_norm1000_xyxy=(10, 20, 10, 400)),),
            image_grid_thw=(1, 16, 16),
            processed_width=640,
            processed_height=480,
            image_identity="image.jpg",
        )

    with pytest.raises(ValueError, match="bbox_norm1000_xyxy"):
        CoverageLedgerSidecar(
            sample_id="coco:0",
            prompt_end_position=9,
            object_entries=(replace(entry, bbox_norm1000_xyxy=(10, 20, 999, 1000)),),
            image_grid_thw=(1, 16, 16),
            processed_width=640,
            processed_height=480,
            image_identity="image.jpg",
        )

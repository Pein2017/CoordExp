from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from src.common.errors import EncodingContractError
from src.config.loader import load_train_config
from src.data import load_raw_examples
from src.qwen.encoding import _token_indices_for_char_range, encode_rendered_example
from src.qwen.loading import QwenProcessorIdentity
from src.qwen.loading import load_qwen_components
from src.templates import render_example


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


def test_rendered_example_encodes_token_spans_and_visual_expansion() -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    raw = load_raw_examples(resolved.config.data.train)[0]
    rendered = render_example(raw, resolved.config.template)
    components = load_qwen_components(resolved.config, load_model=False)

    encoded = encode_rendered_example(
        raw,
        rendered,
        components=components,
        processor_config=resolved.config.model.processor,
        global_max_length=resolved.config.packing.global_max_length,
    )

    assert encoded.example_id == raw.example_id
    assert encoded.base_token_count == 43
    assert encoded.image_token_count == 1014
    assert encoded.input_length == 1056
    assert encoded.image_encoding.image_grid_thw == (1, 52, 78)
    assert encoded.assistant_content_start_char == 146
    assert encoded.base_to_physical_start[4] == 4
    assert encoded.base_to_physical_start[5] == 1018

    supervised = encoded.supervised_token_spans
    ignored = encoded.ignored_token_spans
    assert supervised[0].token_type == "schema"
    assert supervised[0].text == "<|object_ref_start|>"
    assert supervised[0].base_token_start == 20
    assert supervised[0].physical_token_start == 1033
    assert supervised[0].token_ids == (151646,)
    assert [span.token_type for span in supervised].count("desc_text") == 2
    assert [span.token_type for span in supervised].count("coordinate") == 8
    assert supervised[-1].token_type == "eos"
    assert supervised[-1].text == "<|im_end|>"
    assert supervised[-1].physical_token_start == 1054
    assert ignored == encoded.ignored_token_spans
    assert len(ignored) == 1
    assert ignored[0].token_type == "ignored"
    assert ignored[0].text == "\n"
    assert ignored[0].physical_token_start == 1055
    assert all(span.token_type != "ignored" for span in supervised)
    artifact = encoded.to_artifact_dict()
    assert artifact["supervised_token_count"] == 22
    assert artifact["image_pad_base_index"] == 4
    assert artifact["image_pad_physical_start"] == 4
    assert artifact["image_pad_physical_end"] == 1018
    assert artifact["base_to_physical_start"][20] == 1033
    assert len(artifact["base_input_ids_sha256"]) == 64
    assert len(artifact["input_ids_sha256"]) == 64

    with Image.open(raw.image.path) as image:
        processor_inputs = components.processor(
            text=[encoded.chat_text],
            images=[image.convert("RGB")],
            return_tensors="pt",
            do_resize=False,
        )
    assert tuple(int(token_id) for token_id in processor_inputs["input_ids"][0].tolist()) == (
        encoded.input_ids
    )
    assert tuple(int(value) for value in processor_inputs["image_grid_thw"][0].tolist()) == (
        encoded.image_encoding.image_grid_thw
    )


def test_encoded_example_accepts_exact_global_max_length() -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    raw = load_raw_examples(resolved.config.data.train)[0]
    rendered = render_example(raw, resolved.config.template)
    components = load_qwen_components(resolved.config, load_model=False)

    encoded = encode_rendered_example(
        raw,
        rendered,
        components=components,
        processor_config=resolved.config.model.processor,
        global_max_length=1056,
    )

    assert encoded.input_length == 1056


def test_encoded_example_longer_than_global_max_length_fails() -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    raw = load_raw_examples(resolved.config.data.train)[0]
    rendered = render_example(raw, resolved.config.template)
    components = load_qwen_components(resolved.config, load_model=False)

    with pytest.raises(EncodingContractError) as exc_info:
        encode_rendered_example(
            raw,
            rendered,
            components=components,
            processor_config=resolved.config.model.processor,
            global_max_length=100,
        )

    assert exc_info.value.code == "qwen.encoded_example_too_long"
    assert exc_info.value.context["input_length"] == 1056
    assert exc_info.value.context["global_max_length"] == 100


def test_missing_assistant_suffix_in_chat_template_fails() -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    raw = load_raw_examples(resolved.config.data.train)[0]
    rendered = render_example(raw, resolved.config.template)
    broken = replace(rendered, supervised_response_text=rendered.assistant_content_text)
    components = load_qwen_components(resolved.config, load_model=False)

    with pytest.raises(EncodingContractError) as exc_info:
        encode_rendered_example(
            raw,
            broken,
            components=components,
            processor_config=resolved.config.model.processor,
            global_max_length=resolved.config.packing.global_max_length,
        )

    assert exc_info.value.code == "qwen.assistant_suffix_alignment"


def test_tokenizer_offset_crossing_span_boundary_fails() -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    raw = load_raw_examples(resolved.config.data.train)[0]
    rendered = render_example(raw, resolved.config.template)
    chat_text = f"prefix <|image_pad|>\n<|im_start|>assistant\n{rendered.supervised_response_text}"
    assistant_start = chat_text.index(rendered.supervised_response_text)
    components = FakeComponents(
        processor_identity=QwenProcessorIdentity(
            processor_class="FakeQwen3VLProcessor",
            tokenizer_class="FakeTokenizer",
            image_processor_class="FakeQwen2VLImageProcessorFast",
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
        ),
        processor=FakeProcessor(chat_text=chat_text),
        tokenizer=FakeTokenizer(chat_text=chat_text, crossing_start=assistant_start),
    )

    with pytest.raises(EncodingContractError) as exc_info:
        encode_rendered_example(
            raw,
            rendered,
            components=components,
            processor_config=resolved.config.model.processor,
            global_max_length=resolved.config.packing.global_max_length,
        )

    assert exc_info.value.code == "qwen.span_token_boundary"


def test_token_span_lookup_handles_non_monotonic_zero_width_offset_without_bisect() -> None:
    """Permanent regression guard for the Wave 5 (task 5.3) rejected bisect
    candidate: `_token_indices_for_char_range` MUST stay a full linear scan.
    A bisect/binary-search shortcut over `offset_mapping` silently assumes
    monotonically non-decreasing token offsets, but a zero-width
    special-token-like `(0, 0)` entry can appear mid-sequence, breaking that
    assumption. Here, querying `[5, 15)` against
    `[(0, 5), (5, 10), (0, 0), (10, 15)]` must resolve to the two real
    overlapping tokens at indices (1, 3), correctly skipping the mid-sequence
    `(0, 0)` entry (its `token_end=0 <= char_start=5`), with no spurious
    `qwen.span_token_coverage` error. See `implementation-notes.md` "M5a" for
    the full rejected-bisect writeup."""

    offset_mapping = [(0, 5), (5, 10), (0, 0), (10, 15)]

    indices = _token_indices_for_char_range(
        offset_mapping,
        char_start=5,
        char_end=15,
        example_id="ex",
        span_text="x",
    )

    assert indices == (1, 3)


class FakeComponents:
    def __init__(
        self,
        *,
        processor_identity: QwenProcessorIdentity,
        processor: "FakeProcessor",
        tokenizer: "FakeTokenizer",
    ) -> None:
        self.processor_identity = processor_identity
        self.processor = processor
        self.tokenizer = tokenizer


class FakeProcessor:
    def __init__(self, *, chat_text: str) -> None:
        self.chat_text = chat_text
        self.image_processor = FakeImageProcessor()

    def apply_chat_template(self, *_: Any, **__: Any) -> str:
        return self.chat_text


class FakeImageProcessor:
    def __call__(self, **_: Any) -> dict[str, torch.Tensor]:
        return {
            "image_grid_thw": torch.tensor([[1, 52, 78]], dtype=torch.long),
            "pixel_values": torch.zeros((4056, 1536), dtype=torch.float32),
        }


class FakeTokenizer:
    def __init__(self, *, chat_text: str, crossing_start: int) -> None:
        self.chat_text = chat_text
        self.crossing_start = crossing_start

    def convert_tokens_to_ids(self, token: str) -> int | None:
        if token == "<|image_pad|>":
            return 151655
        return None

    def __call__(self, text: str, **_: Any) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert text == self.chat_text
        image_start = text.index("<|image_pad|>")
        return {
            "input_ids": [101, 151655, 151646],
            "offset_mapping": [
                (0, 6),
                (image_start, image_start + len("<|image_pad|>")),
                (self.crossing_start, self.crossing_start + 21),
            ],
        }

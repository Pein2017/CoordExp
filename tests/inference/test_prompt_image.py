from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from src.common.errors import EncodingContractError
from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig
from src.data import ImageRef, RawExample, RawObject, SourceProvenance
from src.qwen.loading import QwenProcessorIdentity
from src.templates import render_example


class FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


class FakeProcessor:
    def __init__(
        self,
        *,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> None:
        self.tokenizer = FakeTokenizer()
        self.image_processor = FakeImageProcessor(
            image_grid_thw=(
                torch.tensor([[1, 4, 6]], dtype=torch.long)
                if image_grid_thw is None
                else image_grid_thw
            ),
            pixel_values=(
                torch.zeros((24, 1536), dtype=torch.float32)
                if pixel_values is None
                else pixel_values
            ),
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        continue_final_message: bool = False,
        **_: Any,
    ) -> str | list[int]:
        if continue_final_message:
            assert messages[-1]["role"] == "assistant"
            assert add_generation_prompt is False
        else:
            assert messages[-1]["role"] == "user"
        pieces: list[str] = []
        for index, message in enumerate(messages):
            pieces.append(f"<|im_start|>{message['role']}\n")
            for item in message["content"]:
                if item["type"] == "image":
                    pieces.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif item["type"] == "text":
                    pieces.append(item["text"])
            is_open_final_message = (
                continue_final_message and index == len(messages) - 1
            )
            if not is_open_final_message:
                pieces.append("<|im_end|>\n")
        if add_generation_prompt:
            pieces.append("<|im_start|>assistant\n")
        text = "".join(pieces)
        if tokenize:
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return text


class FakeImageProcessor:
    def __init__(self, *, image_grid_thw: torch.Tensor, pixel_values: torch.Tensor) -> None:
        self.image_grid_thw = image_grid_thw
        self.pixel_values = pixel_values
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(dict(kwargs))
        return {"image_grid_thw": self.image_grid_thw, "pixel_values": self.pixel_values}


@dataclass(frozen=True)
class FakeComponents:
    processor_identity: QwenProcessorIdentity
    processor: FakeProcessor


def test_inference_prompt_token_ids_match_backend_prompt_ids(tmp_path: Path) -> None:
    from src.inference.prompt import build_prompt_record, verify_prompt_token_parity

    example = _raw_example(tmp_path, width=96, height=64)
    template = _template_config()
    rendered = render_example(example, template)
    processor = FakeProcessor()

    record = build_prompt_record(example, template, processor=processor, row_index=0)
    backend_prompt_ids = processor.apply_chat_template(
        list(record.messages),
        tokenize=True,
        add_generation_prompt=True,
    )

    assert record.prompt_token_ids == backend_prompt_ids
    assert record.prompt_text != rendered.supervised_response_text
    parity = verify_prompt_token_parity(
        record,
        backend_prompt_token_ids=list(backend_prompt_ids),
    )
    assert parity["prompt_token_parity"] == "verified"
    assert parity["prompt_token_count"] == len(record.prompt_token_ids)


def test_prompt_tokenization_uses_explicit_visual_input_image(tmp_path: Path) -> None:
    from src.inference.prompt import build_prompt_record

    observed_sizes: list[tuple[int, int]] = []

    class RecordingProcessor(FakeProcessor):
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            **kwargs: Any,
        ) -> str | list[int]:
            image_items = [
                item
                for message in messages
                for item in message["content"]
                if item["type"] == "image"
            ]
            assert len(image_items) == 1
            image = image_items[0]["image"]
            assert isinstance(image, Image.Image)
            observed_sizes.append(image.size)
            return super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )

    visual_input_image = Image.new("RGB", (32, 48), color=(1, 2, 3))
    example = _raw_example(tmp_path, width=96, height=64)
    try:
        record = build_prompt_record(
            example,
            _template_config(),
            processor=RecordingProcessor(),
            row_index=0,
            visual_input_image=visual_input_image,
        )
    finally:
        visual_input_image.close()

    assert observed_sizes == [(32, 48), (32, 48)]
    retained_image_item = next(
        item
        for message in record.messages
        for item in message["content"]
        if item["type"] == "image"
    )
    assert retained_image_item["image"] == str(example.image.path)


def test_prompt_token_ids_accept_real_qwen_single_batch_shape(tmp_path: Path) -> None:
    from src.inference.prompt import build_prompt_record

    class NestedTokenProcessor(FakeProcessor):
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            **kwargs: Any,
        ) -> str | list[list[int]]:
            rendered = super().apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            assert isinstance(rendered, str)
            if tokenize:
                return [[ord(char) for char in rendered]]
            return rendered

    record = build_prompt_record(
        _raw_example(tmp_path, width=96, height=64),
        _template_config(),
        processor=NestedTokenProcessor(),
        row_index=0,
    )

    assert record.prompt_token_ids
    assert all(isinstance(token_id, int) for token_id in record.prompt_token_ids)


def test_prompt_record_preserves_template_identity_and_training_fingerprint(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import build_prompt_record

    example = _raw_example(tmp_path, width=96, height=64)
    template = _template_config()

    record = build_prompt_record(example, template, processor=FakeProcessor(), row_index=3)
    rendered = render_example(example, template)

    assert record.row_index == 3
    assert record.example_id == example.example_id
    assert record.template_fingerprint == rendered.template_fingerprint
    assert record.template_id == "coordexp-swift-template-v1"
    assert record.object_ordering == "source_order"
    assert record.realized_object_order == [
        {"object_id": "object-1", "source_index": 0, "rendered_index": 0}
    ]
    assert record.to_artifact_dict()["template_fingerprint"] == rendered.template_fingerprint


def test_open_assistant_continuation_appends_without_boundary_and_records_evidence(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import AssistantContinuation, build_prompt_record

    processor = FakeProcessor()
    ordinary = build_prompt_record(
        _raw_example(tmp_path, width=96, height=64),
        _template_config(),
        processor=processor,
        row_index=0,
    )
    continuation_text = (
        "<|object_ref_start|>café<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|>"
        "<|coord_300|><|coord_400|><|box_end|>"
    )
    continued = build_prompt_record(
        _raw_example(tmp_path, width=96, height=64),
        _template_config(),
        processor=processor,
        row_index=0,
        assistant_continuation=AssistantContinuation(continuation_text),
    )

    assert continued.prompt_text == ordinary.prompt_text
    assert continued.messages == ordinary.messages
    assert continued.chat_text == ordinary.chat_text + continuation_text
    assert continued.chat_text.endswith("assistant\n" + continuation_text)
    assert "assistant\n\n" not in continued.chat_text
    assert continued.prompt_token_ids == [ord(char) for char in continued.chat_text]
    assert continued.image_placeholder_count == 1
    assert continued.open_assistant_interval_verified is True

    char_start, char_end = continued.continuation_character_span or (-1, -1)
    byte_start, byte_end = continued.continuation_byte_span or (-1, -1)
    assert continued.chat_text[char_start:char_end] == continuation_text
    assert continued.chat_text.encode("utf-8")[byte_start:byte_end] == (
        continuation_text.encode("utf-8")
    )
    assert continued.open_assistant_content_start_character == len(ordinary.chat_text)
    assert continued.open_assistant_content_start_byte == len(
        ordinary.chat_text.encode("utf-8")
    )
    assert continued.continuation_text_sha256 == hashlib.sha256(
        continuation_text.encode("utf-8")
    ).hexdigest()
    assert continued.continuation_token_impact_span == (
        len(ordinary.prompt_token_ids),
        len(continued.prompt_token_ids),
    )

    artifact = continued.to_artifact_dict()
    assert artifact["prompt_text"] == ordinary.prompt_text
    assert artifact["full_chat_text"] == continued.chat_text
    assert artifact["prompt_token_ids"] == continued.prompt_token_ids
    assert artifact["continuation_character_span"] == [char_start, char_end]
    assert artifact["continuation_byte_span"] == [byte_start, byte_end]
    assert artifact["continuation_token_impact_span"] == list(
        continued.continuation_token_impact_span
    )
    assert len(artifact["full_prompt_fingerprint"]) == 64


def test_continuation_token_impact_span_includes_boundary_retokenization(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import AssistantContinuation, build_prompt_record

    class BoundaryMergingTokenizer(FakeTokenizer):
        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool = False,
            **kwargs: Any,
        ) -> dict[str, list[int]]:
            ids = super().__call__(
                text,
                add_special_tokens=add_special_tokens,
                **kwargs,
            )["input_ids"]
            boundary = text.rfind("\nX")
            if boundary >= 0:
                ids = ids[:boundary] + [900_001] + ids[boundary + 2 :]
            return {"input_ids": ids}

    processor = FakeProcessor()
    processor.tokenizer = BoundaryMergingTokenizer()
    ordinary = build_prompt_record(
        _raw_example(tmp_path, width=96, height=64),
        _template_config(),
        processor=processor,
        row_index=0,
    )
    continued = build_prompt_record(
        _raw_example(tmp_path, width=96, height=64),
        _template_config(),
        processor=processor,
        row_index=0,
        assistant_continuation=AssistantContinuation("X"),
    )

    assert continued.continuation_token_impact_span == (
        len(ordinary.prompt_token_ids) - 1,
        len(continued.prompt_token_ids),
    )
    assert continued.prompt_token_ids[-1] == 900_001


@pytest.mark.parametrize(
    ("text", "boundary_class"),
    [
        ("prefix<|image_pad|>", "image_placeholder"),
        ("prefix<|im_end|>", "assistant_terminator"),
        ("prefix<|endoftext|>", "end_of_sequence"),
        ("prefix<|im_start|>user", "chat_turn_opener"),
    ],
)
def test_open_assistant_continuation_rejects_forbidden_controls(
    tmp_path: Path,
    text: str,
    boundary_class: str,
) -> None:
    from src.inference.prompt import AssistantContinuation, build_prompt_record

    with pytest.raises(EncodingContractError) as exc_info:
        build_prompt_record(
            _raw_example(tmp_path, width=96, height=64),
            _template_config(),
            processor=FakeProcessor(),
            row_index=0,
            assistant_continuation=AssistantContinuation(text),
        )

    assert exc_info.value.code == "inference.assistant_continuation_forbidden_control"
    assert exc_info.value.context["boundary_class"] == boundary_class


def test_open_assistant_continuation_allows_earlier_completed_turn_terminators(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import AssistantContinuation, build_prompt_record

    record = build_prompt_record(
        _raw_example(tmp_path, width=96, height=64),
        _template_config(),
        processor=FakeProcessor(),
        row_index=0,
        assistant_continuation=AssistantContinuation("accepted-row"),
    )

    assistant_start = record.chat_text.rfind("<|im_start|>assistant\n")
    assert "<|im_end|>" in record.chat_text[:assistant_start]
    assert "<|im_end|>" not in record.chat_text[assistant_start:]


def test_open_assistant_continuation_rejects_closed_or_prepopulated_assistant(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import AssistantContinuation, build_prompt_record

    class ClosedAssistantProcessor(FakeProcessor):
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            **kwargs: Any,
        ) -> str | list[int]:
            text = super().apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            assert isinstance(text, str)
            text += "existing<|im_end|>"
            return [ord(char) for char in text] if tokenize else text

    with pytest.raises(EncodingContractError) as exc_info:
        build_prompt_record(
            _raw_example(tmp_path, width=96, height=64),
            _template_config(),
            processor=ClosedAssistantProcessor(),
            row_index=0,
            assistant_continuation=AssistantContinuation("accepted-row"),
        )

    assert exc_info.value.code == "inference.open_assistant_forbidden_boundary"
    assert exc_info.value.context["boundary_class"] == "assistant_terminator"


def test_open_assistant_continuation_rejects_extra_chat_turn(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import AssistantContinuation, build_prompt_record

    class ExtraTurnProcessor(FakeProcessor):
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            **kwargs: Any,
        ) -> str | list[int]:
            text = super().apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            assert isinstance(text, str)
            text += "<|im_start|>assistant\n"
            return [ord(char) for char in text] if tokenize else text

    with pytest.raises(EncodingContractError) as exc_info:
        build_prompt_record(
            _raw_example(tmp_path, width=96, height=64),
            _template_config(),
            processor=ExtraTurnProcessor(),
            row_index=0,
            assistant_continuation=AssistantContinuation("accepted-row"),
        )

    assert exc_info.value.code == "inference.open_assistant_extra_turn"


def test_open_assistant_continuation_requires_exactly_one_image_placeholder(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import AssistantContinuation, build_prompt_record

    class MissingImagePlaceholderProcessor(FakeProcessor):
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            **kwargs: Any,
        ) -> str | list[int]:
            text = super().apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            assert isinstance(text, str)
            text = text.replace("<|image_pad|>", "")
            return [ord(char) for char in text] if tokenize else text

    with pytest.raises(EncodingContractError) as exc_info:
        build_prompt_record(
            _raw_example(tmp_path, width=96, height=64),
            _template_config(),
            processor=MissingImagePlaceholderProcessor(),
            row_index=0,
            assistant_continuation=AssistantContinuation("accepted-row"),
        )

    assert exc_info.value.code == "inference.continuation_image_placeholder_count"
    assert exc_info.value.context["image_placeholder_count"] == 0


def test_open_assistant_continuation_rejects_context_overflow(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import AssistantContinuation, build_prompt_record

    example = _raw_example(tmp_path, width=96, height=64)
    with pytest.raises(EncodingContractError) as context_exc:
        build_prompt_record(
            example,
            _template_config(),
            processor=FakeProcessor(),
            row_index=0,
            assistant_continuation=AssistantContinuation("accepted-row"),
            max_prompt_tokens=1,
        )
    assert context_exc.value.code == "inference.continuation_context_limit"


def test_open_assistant_continuation_rejects_non_native_chat_text(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import AssistantContinuation, build_prompt_record

    class ClosingContinuationProcessor(FakeProcessor):
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            continue_final_message: bool = False,
            **kwargs: Any,
        ) -> str | list[int]:
            rendered = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                **kwargs,
            )
            if not continue_final_message:
                return rendered
            suffix = "<|im_end|>\n"
            if tokenize:
                assert isinstance(rendered, list)
                return [*rendered, *(ord(char) for char in suffix)]
            assert isinstance(rendered, str)
            return rendered + suffix

    with pytest.raises(EncodingContractError) as exc_info:
        build_prompt_record(
            _raw_example(tmp_path, width=96, height=64),
            _template_config(),
            processor=ClosingContinuationProcessor(),
            row_index=0,
            assistant_continuation=AssistantContinuation("accepted-row"),
        )
    assert exc_info.value.code == "inference.continuation_native_chat_text"


def test_real_qwen_continuation_preserves_expanded_image_tokens(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import AssistantContinuation, build_prompt_record
    from src.qwen.runtime_loading import (
        QwenLoadOptions,
        load_qwen_components_from_options,
    )

    base_model = Path(
        "/data/Qwen3-VL/model_cache/models/Qwen/"
        "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
    )
    if not base_model.is_dir():
        pytest.skip("local Qwen3-VL processor fixture is unavailable")
    components = load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=str(base_model),
            dtype="bf16",
            attn_implementation="sdpa",
            load_model=False,
        )
    )
    image_pad_token_id = components.tokenizer.convert_tokens_to_ids(
        "<|image_pad|>"
    )
    example = _raw_example(tmp_path, width=96, height=64)
    with Image.open(example.image.path) as source_image:
        explicit_image = source_image.convert("RGB")
    observed_counts: list[tuple[int, int]] = []
    try:
        for visual_input_image in (None, explicit_image):
            prompt_kwargs = (
                {}
                if visual_input_image is None
                else {"visual_input_image": visual_input_image}
            )
            ordinary = build_prompt_record(
                example,
                _template_config(),
                processor=components.processor,
                row_index=0,
                **prompt_kwargs,
            )
            continued = build_prompt_record(
                example,
                _template_config(),
                processor=components.processor,
                row_index=0,
                assistant_continuation=AssistantContinuation("accepted-row"),
                **prompt_kwargs,
            )
            observed_counts.append(
                (
                    ordinary.prompt_token_ids.count(image_pad_token_id),
                    continued.prompt_token_ids.count(image_pad_token_id),
                )
            )
    finally:
        explicit_image.close()

    assert observed_counts == [(70, 70), (70, 70)]


def test_open_assistant_continuation_rejects_non_integral_context_limit(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import AssistantContinuation, build_prompt_record

    with pytest.raises(EncodingContractError) as exc_info:
        build_prompt_record(
            _raw_example(tmp_path, width=96, height=64),
            _template_config(),
            processor=FakeProcessor(),
            row_index=0,
            assistant_continuation=AssistantContinuation("accepted-row"),
            max_prompt_tokens=1.5,  # type: ignore[arg-type]
        )

    assert exc_info.value.code == "inference.continuation_context_limit_invalid"


def test_no_continuation_argument_preserves_prompt_bytes_and_tokens(tmp_path: Path) -> None:
    from src.inference.prompt import build_prompt_record

    processor = FakeProcessor()
    example = _raw_example(tmp_path, width=96, height=64)
    omitted = build_prompt_record(
        example,
        _template_config(),
        processor=processor,
        row_index=0,
    )
    explicit_none = build_prompt_record(
        example,
        _template_config(),
        processor=processor,
        row_index=0,
        assistant_continuation=None,
    )

    assert explicit_none.chat_text.encode("utf-8") == omitted.chat_text.encode("utf-8")
    assert explicit_none.prompt_token_ids == omitted.prompt_token_ids
    assert explicit_none.to_artifact_dict() == omitted.to_artifact_dict()


def test_image_plan_jsonl_records_mandatory_no_resize_fields(tmp_path: Path) -> None:
    from src.inference.image_plan import write_image_plan_jsonl

    components = FakeComponents(_processor_identity(), FakeProcessor())
    output_path = tmp_path / "image_plan.jsonl"

    rows = write_image_plan_jsonl(
        [_raw_example(tmp_path / "data", width=96, height=64)],
        components=components,
        processor_config=_processor_config(),
        output_path=output_path,
        materialize=True,
    )

    assert len(rows) == 1
    row = json.loads(output_path.read_text().strip())
    for field in {
        "row_id",
        "row_index",
        "example_id",
        "image_path",
        "declared_width",
        "declared_height",
        "decoded_width",
        "decoded_height",
        "patch_size",
        "merge_size",
        "temporal_patch_size",
        "expected_image_grid_thw",
        "observed_image_grid_thw",
        "raw_patch_rows",
        "merged_visual_tokens",
        "do_resize",
        "status",
        "error",
    }:
        assert field in row
    assert row["row_id"] == "image-96x64"
    assert row["row_index"] == 0
    assert row["expected_image_grid_thw"] == [1, 4, 6]
    assert row["observed_image_grid_thw"] == [1, 4, 6]
    assert row["raw_patch_rows"] == 24
    assert row["merged_visual_tokens"] == 6
    assert row["do_resize"] is False
    assert row["status"] == "ok"
    assert row["error"] is None


def test_image_plan_jsonl_refuses_non_materialized_public_write(tmp_path: Path) -> None:
    from src.inference.image_plan import write_image_plan_jsonl

    with pytest.raises(EncodingContractError) as exc_info:
        write_image_plan_jsonl(
            [_raw_example(tmp_path / "data", width=96, height=64)],
            components=FakeComponents(_processor_identity(), FakeProcessor()),
            processor_config=_processor_config(),
            output_path=tmp_path / "image_plan.jsonl",
            materialize=False,
        )

    assert exc_info.value.code == "inference.image_plan_materialize_required"
    assert not (tmp_path / "image_plan.jsonl").exists()


def test_image_plan_records_processor_model_vision_parity(tmp_path: Path) -> None:
    from src.inference.image_plan import verify_processor_model_vision_parity

    components = FakeComponents(_processor_identity(), FakeProcessor())
    model_config = _model_config()

    parity = verify_processor_model_vision_parity(
        processor_identity=components.processor_identity,
        model_config=model_config,
    )

    assert parity["status"] == "matched"
    assert parity["processor"]["patch_size"] == 16
    assert parity["processor"]["merge_size"] == 2
    assert parity["processor"]["temporal_patch_size"] == 2
    assert parity["model"]["patch_size"] == 16
    assert parity["model"]["spatial_merge_size"] == 2
    assert parity["model"]["temporal_patch_size"] == 2


def test_processor_model_vision_mismatch_fails_before_generation() -> None:
    from src.inference.image_plan import verify_processor_model_vision_parity

    with pytest.raises(EncodingContractError) as exc_info:
        verify_processor_model_vision_parity(
            processor_identity=_processor_identity(),
            model_config=_model_config(patch_size=14),
        )

    assert exc_info.value.code == "inference.image_vision_mismatch"
    assert exc_info.value.context["field"] == "patch_size"


def test_valid_no_resize_image_materializes_with_do_resize_false(tmp_path: Path) -> None:
    from src.inference.image_plan import materialize_image_plan_rows

    processor = FakeProcessor()
    rows = materialize_image_plan_rows(
        [_raw_example(tmp_path, width=96, height=64)],
        components=FakeComponents(_processor_identity(), processor),
        processor_config=_processor_config(),
        materialize=True,
    )

    assert rows[0].status == "ok"
    assert rows[0].do_resize is False
    assert rows[0].observed_image_grid_thw == [1, 4, 6]
    assert processor.image_processor.calls[0]["do_resize"] is False


def test_invalid_no_resize_dimensions_are_terminal_input_failure(tmp_path: Path) -> None:
    from src.inference.image_plan import materialize_image_plan_rows

    processor = FakeProcessor()

    with pytest.raises(EncodingContractError) as exc_info:
        materialize_image_plan_rows(
            [_raw_example(tmp_path, width=96, height=65)],
            components=FakeComponents(_processor_identity(), processor),
            processor_config=_processor_config(),
            materialize=True,
        )

    assert exc_info.value.code == "qwen.image_no_resize_dimensions"
    assert processor.image_processor.calls == []


def test_image_plan_row_count_matches_input_rows(tmp_path: Path) -> None:
    from src.inference.image_plan import materialize_image_plan_rows

    rows = materialize_image_plan_rows(
        [
            _raw_example(tmp_path / "a", width=96, height=64, example_id="row-a"),
            _raw_example(tmp_path / "b", width=96, height=64, example_id="row-b"),
        ],
        components=FakeComponents(
            _processor_identity(),
            FakeProcessor(
                image_grid_thw=torch.tensor([[1, 4, 6], [1, 4, 6]], dtype=torch.long),
                pixel_values=torch.zeros((48, 1536), dtype=torch.float32),
            ),
        ),
        processor_config=_processor_config(),
        materialize=True,
    )

    assert [row.row_id for row in rows] == ["row-a", "row-b"]
    assert [row.row_index for row in rows] == [0, 1]


def _template_config() -> TemplateConfig:
    return TemplateConfig(
        object_field_order="desc_first",
        object_ordering="source_order",
        assistant_format="object_box_closed",
        prompt=TemplatePromptConfig(
            system="You are a detector.",
            user="Describe each object with a closed box.",
        ),
    )


def _processor_identity() -> QwenProcessorIdentity:
    return QwenProcessorIdentity(
        processor_class="FakeQwen3VLProcessor",
        tokenizer_class="FakeTokenizer",
        image_processor_class="FakeQwen2VLImageProcessorFast",
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
    )


def _processor_config() -> ProcessorConfig:
    return ProcessorConfig(
        do_resize=False,
        max_raw_pixels=1_000_000,
        max_merged_visual_tokens=4_096,
    )


def _model_config(
    *,
    patch_size: int = 16,
    spatial_merge_size: int = 2,
    temporal_patch_size: int = 2,
) -> Any:
    return type(
        "FakeModelConfig",
        (),
        {
            "vision_config": type(
                "FakeVisionConfig",
                (),
                {
                    "patch_size": patch_size,
                    "spatial_merge_size": spatial_merge_size,
                    "temporal_patch_size": temporal_patch_size,
                },
            )()
        },
    )()


def _raw_example(
    tmp_path: Path,
    *,
    width: int,
    height: int,
    example_id: str | None = None,
) -> RawExample:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{example_id or f'image-{width}x{height}'}.jpg"
    Image.new("RGB", (width, height), color=(12, 34, 56)).save(path)
    return RawExample(
        example_id=example_id or f"image-{width}x{height}",
        image=ImageRef(
            declared_path=path.name,
            path=path,
            width=width,
            height=height,
            stat={},
        ),
        objects=(RawObject("object-1", "object", (100, 200, 300, 400), {}),),
        metadata={},
        source=SourceProvenance(tmp_path / "examples.jsonl", 1, "abc", "unit"),
    )

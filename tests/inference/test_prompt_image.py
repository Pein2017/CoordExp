from __future__ import annotations

import json
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
        self.chat_calls: list[dict[str, Any]] = []
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
        **kwargs: Any,
    ) -> str | list[int]:
        self.chat_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                **kwargs,
            }
        )
        assert messages[-1]["role"] == "user"
        pieces: list[str] = []
        for message in messages:
            pieces.append(f"<|im_start|>{message['role']}\n")
            for item in message["content"]:
                if item["type"] == "image":
                    pieces.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif item["type"] == "text":
                    pieces.append(item["text"])
            pieces.append("<|im_end|>\n")
        if add_generation_prompt:
            pieces.append("<|im_start|>assistant\n")
        text = "".join(pieces)
        if tokenize:
            return [ord(char) for char in text]
        return text


class FakeImageProcessor:
    def __init__(
        self, *, image_grid_thw: torch.Tensor, pixel_values: torch.Tensor
    ) -> None:
        self.image_grid_thw = image_grid_thw
        self.pixel_values = pixel_values
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(dict(kwargs))
        return {
            "image_grid_thw": self.image_grid_thw,
            "pixel_values": self.pixel_values,
        }


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

    record = build_prompt_record(
        example, template, processor=FakeProcessor(), row_index=3
    )
    rendered = render_example(example, template)

    assert record.row_index == 3
    assert record.example_id == example.example_id
    assert record.template_fingerprint == rendered.template_fingerprint
    assert record.template_id == "coordexp-swift-template-v1"
    assert record.object_ordering == "source_order"
    assert record.realized_object_order == [
        {"object_id": "object-1", "source_index": 0, "rendered_index": 0}
    ]
    assert (
        record.to_artifact_dict()["template_fingerprint"]
        == rendered.template_fingerprint
    )


def test_image_only_prompt_uses_current_policy_without_fake_objects() -> None:
    from src.config.fingerprint import sha256_json
    from src.inference.prompt import (
        TEMPLATE_ID,
        build_image_prompt_record,
    )

    canvas = Image.new("RGB", (96, 64), color=(3, 4, 5))
    processor = FakeProcessor()
    template = _template_config()

    record = build_image_prompt_record(
        example_id="roi-request-1",
        image=canvas,
        template_config=template,
        processor=processor,
    )

    assert record.message_roles == ("system", "user")
    tokenizing_call = next(call for call in processor.chat_calls if call["tokenize"])
    assert tokenizing_call["messages"][-1]["content"][0] == {
        "type": "image",
        "image": canvas,
    }
    assert record.prompt_token_ids
    assert record.prompt_policy_fingerprint == sha256_json(
        {
            "template": template.model_dump(mode="json"),
            "template_id": TEMPLATE_ID,
        }
    )
    assert tokenizing_call["do_resize"] is False
    assert "objects" not in record.to_artifact_dict()


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


def test_valid_no_resize_image_materializes_with_do_resize_false(
    tmp_path: Path,
) -> None:
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


def test_in_memory_canvas_reuses_no_resize_plan_and_processor_validation() -> None:
    from src.qwen.images import encode_qwen_image_canvas

    canvas = Image.new("RGB", (96, 64), color=(9, 8, 7))
    processor = FakeProcessor()

    encoding = encode_qwen_image_canvas(
        example_id="roi-request-1",
        image=canvas,
        components=FakeComponents(_processor_identity(), processor),
        processor_config=_processor_config(),
    )

    assert encoding.image_path is None
    assert encoding.image_grid_thw == (1, 4, 6)
    assert encoding.to_artifact_dict()["image_path"] is None
    call = processor.image_processor.calls[0]
    assert call["images"] == [canvas]
    assert call["return_tensors"] == "pt"
    assert call["do_resize"] is False


def test_in_memory_canvas_rejects_misalignment_before_processor_call() -> None:
    from src.qwen.images import encode_qwen_image_canvas

    processor = FakeProcessor()
    with pytest.raises(EncodingContractError) as exc_info:
        encode_qwen_image_canvas(
            example_id="roi-request-1",
            image=Image.new("RGB", (100, 64)),
            components=FakeComponents(_processor_identity(), processor),
            processor_config=_processor_config(),
        )

    assert exc_info.value.code == "qwen.image_no_resize_dimensions"
    assert processor.image_processor.calls == []


def test_invalid_no_resize_dimensions_are_terminal_input_failure(
    tmp_path: Path,
) -> None:
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

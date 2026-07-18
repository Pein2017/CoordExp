from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
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
    image_token_id = 151655

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": self.encode(text)}

    def encode(self, text: str) -> list[int]:
        marker = "<|image_pad|>"
        pieces = text.split(marker)
        ids: list[int] = []
        for index, piece in enumerate(pieces):
            ids.extend(ord(char) for char in piece)
            if index < len(pieces) - 1:
                ids.append(self.image_token_id)
        return ids

    def convert_tokens_to_ids(self, token: str) -> int | None:
        if token == "<|image_pad|>":
            return self.image_token_id
        return None


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
        **_: Any,
    ) -> str | list[int]:
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
            return self.tokenizer.encode(text)
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


def test_prompt_record_separates_unexpanded_and_executed_prompt_ids(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import build_prompt_record, verify_prompt_token_parity

    example = _raw_example(tmp_path, width=96, height=64)
    template = _template_config()
    rendered = render_example(example, template)
    processor = FakeProcessor()

    record = build_prompt_record(
        example,
        template,
        processor=processor,
        row_index=0,
        merged_visual_tokens=6,
    )
    input_prompt_ids = processor.apply_chat_template(
        list(record.messages),
        tokenize=True,
        add_generation_prompt=True,
    )

    image_token_id = processor.tokenizer.image_token_id
    assert record.chat_text == processor.apply_chat_template(
        list(record.messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    assert record.input_prompt_token_ids == input_prompt_ids
    assert record.input_prompt_token_ids.count(image_token_id) == 1
    assert record.expected_executed_prompt_token_ids.count(image_token_id) == 6
    assert len(record.expected_executed_prompt_token_ids) == len(input_prompt_ids) + 5
    assert record.prompt_token_ids == record.expected_executed_prompt_token_ids
    assert record.prompt_text != rendered.supervised_response_text
    parity = verify_prompt_token_parity(
        record,
        backend_prompt_token_ids=list(record.expected_executed_prompt_token_ids),
    )
    assert parity["prompt_token_parity"] == "verified"
    assert parity["prompt_token_count"] == len(record.prompt_token_ids)
    artifact = record.to_artifact_dict()
    assert artifact["prompt_token_ids"] == record.expected_executed_prompt_token_ids
    assert artifact["input_prompt_token_ids"] == record.input_prompt_token_ids
    assert (
        artifact["expected_executed_prompt_token_ids"]
        == record.expected_executed_prompt_token_ids
    )


def test_prompt_token_ids_are_derived_from_authoritative_chat_text(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import build_prompt_record

    class DivergentTokenProcessor(FakeProcessor):
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
                return [[999]]
            return rendered

    processor = DivergentTokenProcessor()
    record = build_prompt_record(
        _raw_example(tmp_path, width=96, height=64),
        _template_config(),
        processor=processor,
        row_index=0,
        merged_visual_tokens=6,
    )

    assert record.input_prompt_token_ids == processor.tokenizer.encode(record.chat_text)
    assert record.input_prompt_token_ids != [999]
    assert all(isinstance(token_id, int) for token_id in record.prompt_token_ids)


def test_prompt_record_preserves_template_identity_and_training_fingerprint(
    tmp_path: Path,
) -> None:
    from src.inference.prompt import build_prompt_record

    example = _raw_example(tmp_path, width=96, height=64)
    template = _template_config()

    record = build_prompt_record(
        example,
        template,
        processor=FakeProcessor(),
        row_index=3,
        merged_visual_tokens=6,
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
    assert record.to_artifact_dict()["template_fingerprint"] == rendered.template_fingerprint


def test_image_plan_jsonl_records_mandatory_no_resize_fields(tmp_path: Path) -> None:
    from src.inference.image_plan import write_image_plan_jsonl

    components = FakeComponents(_processor_identity(), FakeProcessor())
    output_path = tmp_path / "image_plan.jsonl"

    rows = write_image_plan_jsonl(
        [_raw_example(tmp_path / "data", width=96, height=64)],
        components=components,
        processor_config=_processor_config(),
        output_path=output_path,
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
        "image_content_sha256",
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
    assert row["image_content_sha256"] == hashlib.sha256(
        Path(row["image_path"]).read_bytes()
    ).hexdigest()
    assert row["expected_image_grid_thw"] == [1, 4, 6]
    assert row["observed_image_grid_thw"] is None
    assert row["raw_patch_rows"] == 24
    assert row["merged_visual_tokens"] == 6
    assert row["do_resize"] is False
    assert row["status"] == "ok"
    assert row["error"] is None


def test_image_plan_jsonl_writes_shared_plan_without_materialized_tensors(
    tmp_path: Path,
) -> None:
    from src.inference.image_plan import write_image_plan_jsonl

    processor = FakeProcessor()
    rows = write_image_plan_jsonl(
        [_raw_example(tmp_path / "data", width=96, height=64)],
        components=FakeComponents(_processor_identity(), processor),
        processor_config=_processor_config(),
        output_path=tmp_path / "image_plan.jsonl",
    )

    assert len(rows) == 1
    assert rows[0].observed_image_grid_thw is None
    assert rows[0].backend_projection_evidence_kind == "shared_reference_plan"
    assert processor.image_processor.calls == []
    assert (tmp_path / "image_plan.jsonl").exists()


def test_shared_image_plan_batch_requires_no_model_tensors(tmp_path: Path) -> None:
    from src.inference.image_plan import plan_image_batch

    processor = FakeProcessor()
    batch = plan_image_batch(
        [_raw_example(tmp_path, width=96, height=64)],
        components=FakeComponents(_processor_identity(), processor),
        processor_config=_processor_config(),
    )

    assert list(batch.plans_by_row_id) == ["image-96x64"]
    assert set(vars(batch)) == {"rows", "plans_by_row_id"}
    assert batch.rows[0].image_content_sha256
    assert batch.rows[0].observed_image_grid_thw is None
    assert processor.image_processor.calls == []


def test_private_image_materialization_rejects_path_mutation(tmp_path: Path) -> None:
    from src.inference.image_plan import plan_image_batch
    from src.qwen.images import materialize_qwen_image_plan

    processor = FakeProcessor()
    batch = plan_image_batch(
        [_raw_example(tmp_path, width=96, height=64)],
        components=FakeComponents(_processor_identity(), processor),
        processor_config=_processor_config(),
    )
    plan = batch.plans_by_row_id["image-96x64"]
    Image.new("RGB", (96, 64), color=(99, 88, 77)).save(plan.image_path)

    with pytest.raises(EncodingContractError) as exc_info:
        materialize_qwen_image_plan(
            plan,
            image_processor=processor.image_processor,
        )

    assert exc_info.value.code == "qwen.image_content_sha256_mismatch"
    assert processor.image_processor.calls == []


def test_private_image_materialization_returns_executed_evidence(
    tmp_path: Path,
) -> None:
    from src.inference.image_plan import plan_image_batch
    from src.qwen.images import (
        apply_logical_image_transform,
        materialize_qwen_image_plan,
        rgb_image_sha256,
    )

    processor = FakeProcessor()
    example = _raw_example(tmp_path, width=96, height=64)
    with Image.open(example.image.path) as source:
        asymmetric = source.convert("RGB")
    asymmetric.paste((255, 0, 0), (0, 0, 48, 64))
    asymmetric.paste((0, 0, 255), (48, 0, 96, 64))
    asymmetric.save(example.image.path)
    example = replace(
        example,
        metadata={"augmentation": {"transform_id": "hflip"}},
    )
    batch = plan_image_batch(
        [example],
        components=FakeComponents(_processor_identity(), processor),
        processor_config=_processor_config(),
    )
    plan = batch.plans_by_row_id["image-96x64"]
    with Image.open(plan.image_path) as source:
        source_rgb = source.convert("RGB")
    transformed = apply_logical_image_transform(
        source_rgb,
        "hflip",
        example_id=plan.example_id,
        image_path=plan.image_path,
    )
    expected_executed_sha256 = rgb_image_sha256(transformed)
    original_rgb_sha256 = rgb_image_sha256(source_rgb)

    encoding = materialize_qwen_image_plan(
        plan,
        image_processor=processor.image_processor,
    )

    assert encoding.pixel_values is not None
    assert encoding.image_grid_thw_tensor is not None
    assert encoding.execution_evidence is not None
    assert encoding.execution_evidence.media_sha256 == expected_executed_sha256
    assert encoding.execution_evidence.media_sha256 != original_rgb_sha256
    assert encoding.execution_evidence.media_sha256 != plan.image_content_sha256
    assert encoding.execution_evidence.observed_image_grid_thw == (1, 4, 6)
    assert encoding.execution_evidence.do_resize is False


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


def test_valid_no_resize_image_plan_avoids_backend_projection(tmp_path: Path) -> None:
    from src.inference.image_plan import plan_image_rows

    processor = FakeProcessor()
    rows = plan_image_rows(
        [_raw_example(tmp_path, width=96, height=64)],
        components=FakeComponents(_processor_identity(), processor),
        processor_config=_processor_config(),
    )

    assert rows[0].status == "ok"
    assert rows[0].do_resize is False
    assert rows[0].observed_image_grid_thw is None
    assert processor.image_processor.calls == []


def test_invalid_no_resize_dimensions_are_terminal_input_failure(tmp_path: Path) -> None:
    from src.inference.image_plan import plan_image_rows

    processor = FakeProcessor()

    with pytest.raises(EncodingContractError) as exc_info:
        plan_image_rows(
            [_raw_example(tmp_path, width=96, height=65)],
            components=FakeComponents(_processor_identity(), processor),
            processor_config=_processor_config(),
        )

    assert exc_info.value.code == "qwen.image_no_resize_dimensions"
    assert processor.image_processor.calls == []


def test_image_plan_row_count_matches_input_rows(tmp_path: Path) -> None:
    from src.inference.image_plan import plan_image_rows

    rows = plan_image_rows(
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

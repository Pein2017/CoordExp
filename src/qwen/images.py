"""No-resize Qwen image encoding contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.common.errors import EncodingContractError
from src.config.models import ProcessorConfig
from src.data import RawExample
from src.qwen.runtime_loading import QwenProcessorIdentity


@dataclass(frozen=True)
class QwenNoResizeImagePlan:
    example_id: str
    image_path: Path
    width: int
    height: int
    patch_size: int
    merge_size: int
    temporal_patch_size: int
    required_spatial_factor: int
    raw_pixels: int
    raw_patch_rows: int
    expected_pixel_values_width: int
    image_grid_thw: tuple[int, int, int]
    merged_visual_tokens: int
    max_raw_pixels: int
    max_merged_visual_tokens: int

    def to_artifact_dict(self) -> dict[str, int | str | list[int]]:
        return {
            "example_id": self.example_id,
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "temporal_patch_size": self.temporal_patch_size,
            "required_spatial_factor": self.required_spatial_factor,
            "raw_pixels": self.raw_pixels,
            "raw_patch_rows": self.raw_patch_rows,
            "expected_pixel_values_width": self.expected_pixel_values_width,
            "image_grid_thw": list(self.image_grid_thw),
            "merged_visual_tokens": self.merged_visual_tokens,
            "max_raw_pixels": self.max_raw_pixels,
            "max_merged_visual_tokens": self.max_merged_visual_tokens,
        }


@dataclass(frozen=True)
class QwenImageEncoding:
    plan: QwenNoResizeImagePlan
    pixel_values: Any | None
    image_grid_thw_tensor: Any | None
    image_processor: Any | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "pixel_values": self.pixel_values,
            "image_grid_thw_tensor": self.image_grid_thw_tensor,
            "image_processor": None,
        }

    @property
    def example_id(self) -> str:
        return self.plan.example_id

    @property
    def image_path(self) -> Path:
        return self.plan.image_path

    @property
    def width(self) -> int:
        return self.plan.width

    @property
    def height(self) -> int:
        return self.plan.height

    @property
    def required_spatial_factor(self) -> int:
        return self.plan.required_spatial_factor

    @property
    def raw_pixels(self) -> int:
        return self.plan.raw_pixels

    @property
    def raw_patch_rows(self) -> int:
        return self.plan.raw_patch_rows

    @property
    def image_grid_thw(self) -> tuple[int, int, int]:
        return self.plan.image_grid_thw

    @property
    def merged_visual_tokens(self) -> int:
        return self.plan.merged_visual_tokens

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            **self.plan.to_artifact_dict(),
            "do_resize": False,
            "pixel_values_shape": (
                None if self.pixel_values is None else list(_shape_tuple(self.pixel_values))
            ),
            "image_grid_thw_tensor_shape": (
                None
                if self.image_grid_thw_tensor is None
                else list(_shape_tuple(self.image_grid_thw_tensor))
            ),
            "pixel_values_materialized": self.pixel_values is not None,
        }


def build_no_resize_image_plan(
    raw_example: RawExample,
    *,
    processor_identity: QwenProcessorIdentity,
    processor_config: ProcessorConfig,
) -> QwenNoResizeImagePlan:
    if not isinstance(raw_example, RawExample):
        raise EncodingContractError(
            "Qwen image planning requires a validated RawExample",
            code="qwen.image_raw_example_type",
            context={"value_type": type(raw_example).__name__},
        )
    if processor_config.do_resize:
        raise EncodingContractError(
            "Qwen no-resize image encoding requires processor.do_resize=false",
            code="qwen.image_resize_enabled",
            context={"example_id": raw_example.example_id},
        )

    patch_size = processor_identity.patch_size
    merge_size = processor_identity.merge_size
    temporal_patch_size = processor_identity.temporal_patch_size
    required_factor = patch_size * merge_size
    width = raw_example.image.width
    height = raw_example.image.height
    if height % required_factor != 0 or width % required_factor != 0:
        raise EncodingContractError(
            "no-resize image dimensions must be divisible by patch_size * merge_size",
            code="qwen.image_no_resize_dimensions",
            context={
                "example_id": raw_example.example_id,
                "image_path": str(raw_example.image.path),
                "width": width,
                "height": height,
                "patch_size": patch_size,
                "merge_size": merge_size,
                "required_spatial_factor": required_factor,
            },
        )

    raw_pixels = width * height
    if raw_pixels > processor_config.max_raw_pixels:
        raise EncodingContractError(
            "no-resize raw-pixel count exceeds configured budget",
            code="qwen.image_raw_pixel_budget",
            context={
                "example_id": raw_example.example_id,
                "image_path": str(raw_example.image.path),
                "raw_pixels": raw_pixels,
                "max_raw_pixels": processor_config.max_raw_pixels,
                "width": width,
                "height": height,
            },
        )

    image_grid_thw = (1, height // patch_size, width // patch_size)
    raw_patch_rows = image_grid_thw[0] * image_grid_thw[1] * image_grid_thw[2]
    merge_area = merge_size * merge_size
    if raw_patch_rows % merge_area != 0:
        raise EncodingContractError(
            "raw patch rows must divide evenly by merge_size**2",
            code="qwen.image_merge_divisibility",
            context={
                "example_id": raw_example.example_id,
                "raw_patch_rows": raw_patch_rows,
                "merge_size": merge_size,
            },
        )
    merged_visual_tokens = raw_patch_rows // merge_area
    if merged_visual_tokens > processor_config.max_merged_visual_tokens:
        raise EncodingContractError(
            "no-resize merged visual token count exceeds configured budget",
            code="qwen.image_visual_token_budget",
            context={
                "example_id": raw_example.example_id,
                "image_path": str(raw_example.image.path),
                "merged_visual_tokens": merged_visual_tokens,
                "max_merged_visual_tokens": processor_config.max_merged_visual_tokens,
                "image_grid_thw": list(image_grid_thw),
            },
        )

    return QwenNoResizeImagePlan(
        example_id=raw_example.example_id,
        image_path=raw_example.image.path,
        width=width,
        height=height,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        required_spatial_factor=required_factor,
        raw_pixels=raw_pixels,
        raw_patch_rows=raw_patch_rows,
        expected_pixel_values_width=3 * temporal_patch_size * patch_size * patch_size,
        image_grid_thw=image_grid_thw,
        merged_visual_tokens=merged_visual_tokens,
        max_raw_pixels=processor_config.max_raw_pixels,
        max_merged_visual_tokens=processor_config.max_merged_visual_tokens,
    )


def encode_qwen_image(
    raw_example: RawExample,
    *,
    components: Any,
    processor_config: ProcessorConfig,
) -> QwenImageEncoding:
    encoding = plan_qwen_image(
        raw_example,
        components=components,
        processor_config=processor_config,
    )
    return materialize_qwen_image_encoding(encoding)


def plan_qwen_image(
    raw_example: RawExample,
    *,
    components: Any,
    processor_config: ProcessorConfig,
) -> QwenImageEncoding:
    plan = build_no_resize_image_plan(
        raw_example,
        processor_identity=components.processor_identity,
        processor_config=processor_config,
    )
    image_processor = getattr(components.processor, "image_processor", None)
    if image_processor is None:
        raise EncodingContractError(
            "Qwen processor does not expose image_processor",
            code="qwen.image_processor_missing",
            context={"example_id": raw_example.example_id},
        )
    return QwenImageEncoding(
        plan=plan,
        pixel_values=None,
        image_grid_thw_tensor=None,
        image_processor=image_processor,
    )


def materialize_qwen_image_encoding(
    encoding: QwenImageEncoding,
) -> QwenImageEncoding:
    if (
        encoding.pixel_values is not None
        and encoding.image_grid_thw_tensor is not None
    ):
        return encoding
    if encoding.image_processor is None:
        raise EncodingContractError(
            "lazy Qwen image encoding requires an image_processor",
            code="qwen.image_lazy_processor_missing",
            context={
                "example_id": encoding.example_id,
                "image_path": str(encoding.image_path),
            },
        )
    image = _load_rgb_image_from_plan(encoding.plan)
    encoded = encoding.image_processor(
        images=[image],
        return_tensors="pt",
        do_resize=False,
    )
    pixel_values = encoded.get("pixel_values")
    image_grid_thw_tensor = encoded.get("image_grid_thw")
    _validate_processor_output(
        encoding.plan,
        pixel_values=pixel_values,
        image_grid_thw_tensor=image_grid_thw_tensor,
    )
    return QwenImageEncoding(
        plan=encoding.plan,
        pixel_values=pixel_values,
        image_grid_thw_tensor=image_grid_thw_tensor,
        image_processor=encoding.image_processor,
    )


def materialize_qwen_image_encoding_batch(
    encodings: Sequence[QwenImageEncoding],
) -> tuple[torch.Tensor, torch.Tensor]:
    checked = tuple(encodings)
    if not checked:
        raise EncodingContractError(
            "Qwen image batch materialization requires at least one image",
            code="qwen.image_batch_empty",
        )
    for encoding in checked:
        if not isinstance(encoding, QwenImageEncoding):
            raise EncodingContractError(
                "Qwen image batch materialization requires QwenImageEncoding items",
                code="qwen.image_batch_encoding_type",
                context={"value_type": type(encoding).__name__},
            )
    if all(
        encoding.pixel_values is not None
        and encoding.image_grid_thw_tensor is not None
        for encoding in checked
    ):
        return _cat_materialized_image_encodings(checked)
    if not all(
        encoding.pixel_values is None
        and encoding.image_grid_thw_tensor is None
        for encoding in checked
    ):
        materialized = tuple(materialize_qwen_image_encoding(encoding) for encoding in checked)
        return _cat_materialized_image_encodings(materialized)

    image_processor = checked[0].image_processor
    if image_processor is None:
        raise EncodingContractError(
            "lazy Qwen image batch materialization requires an image_processor",
            code="qwen.image_lazy_processor_missing",
            context={
                "example_id": checked[0].example_id,
                "image_path": str(checked[0].image_path),
            },
        )
    for encoding in checked:
        if encoding.image_processor is None:
            raise EncodingContractError(
                "lazy Qwen image batch materialization requires an image_processor",
                code="qwen.image_lazy_processor_missing",
                context={
                    "example_id": encoding.example_id,
                    "image_path": str(encoding.image_path),
                },
            )
        if encoding.image_processor is not image_processor:
            materialized = tuple(
                materialize_qwen_image_encoding(encoding) for encoding in checked
            )
            return _cat_materialized_image_encodings(materialized)

    images = [_load_rgb_image_from_plan(encoding.plan) for encoding in checked]
    try:
        encoded = image_processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
        )
    finally:
        for image in images:
            image.close()
    pixel_values = encoded.get("pixel_values")
    image_grid_thw_tensor = encoded.get("image_grid_thw")
    _validate_batch_processor_output(
        checked,
        pixel_values=pixel_values,
        image_grid_thw_tensor=image_grid_thw_tensor,
    )
    return pixel_values, image_grid_thw_tensor


def attach_qwen_image_processor(
    encoding: QwenImageEncoding,
    image_processor: Any,
) -> QwenImageEncoding:
    return QwenImageEncoding(
        plan=encoding.plan,
        pixel_values=encoding.pixel_values,
        image_grid_thw_tensor=encoding.image_grid_thw_tensor,
        image_processor=image_processor,
    )


def _cat_materialized_image_encodings(
    encodings: Sequence[QwenImageEncoding],
) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_values_parts: list[torch.Tensor] = []
    image_grid_parts: list[torch.Tensor] = []
    for encoding in encodings:
        _validate_processor_output(
            encoding.plan,
            pixel_values=encoding.pixel_values,
            image_grid_thw_tensor=encoding.image_grid_thw_tensor,
        )
        pixel_values_parts.append(encoding.pixel_values)
        image_grid_parts.append(encoding.image_grid_thw_tensor)
    return (
        torch.cat(pixel_values_parts, dim=0),
        torch.cat(image_grid_parts, dim=0),
    )


def _validate_batch_processor_output(
    encodings: Sequence[QwenImageEncoding],
    *,
    pixel_values: Any,
    image_grid_thw_tensor: Any,
) -> None:
    first = encodings[0]
    if pixel_values is None:
        _validate_processor_output(
            first.plan,
            pixel_values=None,
            image_grid_thw_tensor=image_grid_thw_tensor,
        )
    if image_grid_thw_tensor is None:
        _validate_processor_output(
            first.plan,
            pixel_values=pixel_values,
            image_grid_thw_tensor=None,
        )
    expected_rows = sum(encoding.raw_patch_rows for encoding in encodings)
    expected_width = first.plan.expected_pixel_values_width
    pixel_shape = _shape_tuple(pixel_values)
    expected_pixel_shape = (expected_rows, expected_width)
    if pixel_shape != expected_pixel_shape:
        raise EncodingContractError(
            "Qwen batched pixel_values shape does not match no-resize plans",
            code="qwen.image_pixel_values_shape",
            context={
                "example_ids": [encoding.example_id for encoding in encodings],
                "observed_shape": list(pixel_shape),
                "expected_shape": list(expected_pixel_shape),
            },
        )
    grid_shape = _shape_tuple(image_grid_thw_tensor)
    expected_grid_shape = (len(encodings), 3)
    if grid_shape != expected_grid_shape:
        raise EncodingContractError(
            "Qwen batched image_grid_thw must have shape [num_images, 3]",
            code="qwen.image_grid_shape",
            context={
                "example_ids": [encoding.example_id for encoding in encodings],
                "observed_shape": list(grid_shape),
                "expected_shape": list(expected_grid_shape),
            },
        )

    offset = 0
    for index, encoding in enumerate(encodings):
        rows = encoding.raw_patch_rows
        _validate_processor_output(
            encoding.plan,
            pixel_values=pixel_values[offset : offset + rows],
            image_grid_thw_tensor=image_grid_thw_tensor[index : index + 1],
        )
        offset += rows


def _load_rgb_image(raw_example: RawExample) -> Image.Image:
    with Image.open(raw_example.image.path) as image:
        decoded_width, decoded_height = image.size
        if (decoded_width, decoded_height) != (
            raw_example.image.width,
            raw_example.image.height,
        ):
            raise EncodingContractError(
                "decoded image dimensions must match RawExample metadata",
                code="qwen.image_dimension_mismatch",
                context={
                    "example_id": raw_example.example_id,
                    "image_path": str(raw_example.image.path),
                    "declared_width": raw_example.image.width,
                    "declared_height": raw_example.image.height,
                    "decoded_width": decoded_width,
                    "decoded_height": decoded_height,
                },
            )
        return image.convert("RGB")


def _load_rgb_image_from_plan(plan: QwenNoResizeImagePlan) -> Image.Image:
    with Image.open(plan.image_path) as image:
        decoded_width, decoded_height = image.size
        if (decoded_width, decoded_height) != (plan.width, plan.height):
            raise EncodingContractError(
                "decoded image dimensions must match Qwen image plan metadata",
                code="qwen.image_dimension_mismatch",
                context={
                    "example_id": plan.example_id,
                    "image_path": str(plan.image_path),
                    "declared_width": plan.width,
                    "declared_height": plan.height,
                    "decoded_width": decoded_width,
                    "decoded_height": decoded_height,
                },
            )
        return image.convert("RGB")


def _validate_processor_output(
    plan: QwenNoResizeImagePlan,
    *,
    pixel_values: Any,
    image_grid_thw_tensor: Any,
) -> None:
    if pixel_values is None:
        raise EncodingContractError(
            "Qwen image processor output missing pixel_values",
            code="qwen.image_pixel_values_missing",
            context={"example_id": plan.example_id, "image_path": str(plan.image_path)},
        )
    if image_grid_thw_tensor is None:
        raise EncodingContractError(
            "Qwen image processor output missing image_grid_thw",
            code="qwen.image_grid_missing",
            context={"example_id": plan.example_id, "image_path": str(plan.image_path)},
        )
    image_grid_shape = _shape_tuple(image_grid_thw_tensor)
    if image_grid_shape != (1, 3):
        raise EncodingContractError(
            "Qwen image_grid_thw must have shape [1, 3] for V1 single-image examples",
            code="qwen.image_grid_shape",
            context={
                "example_id": plan.example_id,
                "image_path": str(plan.image_path),
                "observed_shape": list(image_grid_shape),
            },
        )
    observed_grid = tuple(int(value) for value in image_grid_thw_tensor[0].detach().cpu().tolist())
    if observed_grid != plan.image_grid_thw:
        raise EncodingContractError(
            "Qwen image_grid_thw does not match no-resize plan",
            code="qwen.image_grid_mismatch",
            context={
                "example_id": plan.example_id,
                "image_path": str(plan.image_path),
                "observed_image_grid_thw": list(observed_grid),
                "expected_image_grid_thw": list(plan.image_grid_thw),
            },
        )

    pixel_shape = _shape_tuple(pixel_values)
    expected_pixel_shape = (plan.raw_patch_rows, plan.expected_pixel_values_width)
    if pixel_shape != expected_pixel_shape:
        raise EncodingContractError(
            "Qwen pixel_values shape does not match no-resize plan",
            code="qwen.image_pixel_values_shape",
            context={
                "example_id": plan.example_id,
                "image_path": str(plan.image_path),
                "observed_shape": list(pixel_shape),
                "expected_shape": list(expected_pixel_shape),
            },
        )


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise EncodingContractError(
            "Qwen processor tensor-like output is missing shape",
            code="qwen.tensor_shape_missing",
            context={"value_type": type(value).__name__},
        )
    return tuple(int(item) for item in shape)


__all__ = [
    "QwenImageEncoding",
    "QwenNoResizeImagePlan",
    "build_no_resize_image_plan",
    "encode_qwen_image",
    "attach_qwen_image_processor",
    "materialize_qwen_image_encoding",
    "materialize_qwen_image_encoding_batch",
    "plan_qwen_image",
]

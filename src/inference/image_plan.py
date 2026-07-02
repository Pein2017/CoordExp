"""Inference image-plan materialization through Qwen no-resize helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.errors import EncodingContractError
from src.config.models import ProcessorConfig
from src.data import RawExample
from src.qwen.images import (
    materialize_qwen_image_encoding,
    materialize_qwen_image_encoding_batch,
    plan_qwen_image,
)
from src.qwen.runtime_loading import QwenProcessorIdentity


@dataclass(frozen=True)
class ImagePlanRow:
    row_id: str
    row_index: int
    example_id: str
    image_path: str
    declared_width: int
    declared_height: int
    decoded_width: int
    decoded_height: int
    patch_size: int
    merge_size: int
    temporal_patch_size: int
    expected_image_grid_thw: list[int]
    observed_image_grid_thw: list[int] | None
    raw_patch_rows: int
    merged_visual_tokens: int
    do_resize: bool
    status: str
    error: dict[str, Any] | None

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "row_index": self.row_index,
            "example_id": self.example_id,
            "image_path": self.image_path,
            "declared_width": self.declared_width,
            "declared_height": self.declared_height,
            "decoded_width": self.decoded_width,
            "decoded_height": self.decoded_height,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "temporal_patch_size": self.temporal_patch_size,
            "expected_image_grid_thw": self.expected_image_grid_thw,
            "observed_image_grid_thw": self.observed_image_grid_thw,
            "raw_patch_rows": self.raw_patch_rows,
            "merged_visual_tokens": self.merged_visual_tokens,
            "do_resize": self.do_resize,
            "status": self.status,
            "error": self.error,
        }


def verify_processor_model_vision_parity(
    *,
    processor_identity: QwenProcessorIdentity,
    model_config: Any,
) -> dict[str, Any]:
    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is None:
        raise EncodingContractError(
            "model config does not expose vision_config",
            code="inference.image_vision_config_missing",
        )
    checks = (
        ("patch_size", processor_identity.patch_size, getattr(vision_config, "patch_size", None)),
        (
            "spatial_merge_size",
            processor_identity.merge_size,
            getattr(vision_config, "spatial_merge_size", None),
        ),
        (
            "temporal_patch_size",
            processor_identity.temporal_patch_size,
            getattr(vision_config, "temporal_patch_size", None),
        ),
    )
    for field, processor_value, model_value in checks:
        if processor_value != model_value:
            raise EncodingContractError(
                "Qwen processor vision parameters do not match model vision config",
                code="inference.image_vision_mismatch",
                context={
                    "field": field,
                    "processor_value": processor_value,
                    "model_value": model_value,
                },
            )
    return {
        "status": "matched",
        "processor": processor_identity.to_artifact_dict(),
        "model": {
            "patch_size": getattr(vision_config, "patch_size"),
            "spatial_merge_size": getattr(vision_config, "spatial_merge_size"),
            "temporal_patch_size": getattr(vision_config, "temporal_patch_size"),
        },
    }


def materialize_image_plan_rows(
    raw_examples: list[RawExample],
    *,
    components: Any,
    processor_config: ProcessorConfig,
    materialize: bool,
) -> list[ImagePlanRow]:
    encodings = [
        plan_qwen_image(
            raw_example,
            components=components,
            processor_config=processor_config,
        )
        for raw_example in raw_examples
    ]
    if materialize and encodings:
        if len(encodings) == 1:
            encodings = [materialize_qwen_image_encoding(encodings[0])]
        else:
            pixel_values, image_grid_thw = materialize_qwen_image_encoding_batch(encodings)
            materialized = []
            pixel_offset = 0
            for index, encoding in enumerate(encodings):
                next_offset = pixel_offset + encoding.raw_patch_rows
                materialized.append(
                    type(encoding)(
                        plan=encoding.plan,
                        pixel_values=pixel_values[pixel_offset:next_offset],
                        image_grid_thw_tensor=image_grid_thw[index : index + 1],
                        image_processor=encoding.image_processor,
                    )
                )
                pixel_offset = next_offset
            encodings = materialized
    return [
        _row_from_encoding(index, encoding, materialized=materialize)
        for index, encoding in enumerate(encodings)
    ]


def write_image_plan_jsonl(
    raw_examples: list[RawExample],
    *,
    components: Any,
    processor_config: ProcessorConfig,
    output_path: Path,
    materialize: bool,
) -> list[ImagePlanRow]:
    if not materialize:
        raise EncodingContractError(
            "V1 image_plan.jsonl writing requires materialized no-resize processor evidence",
            code="inference.image_plan_materialize_required",
            context={"output_path": str(output_path)},
        )
    rows = materialize_image_plan_rows(
        raw_examples,
        components=components,
        processor_config=processor_config,
        materialize=materialize,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_artifact_dict(), sort_keys=True) + "\n")
    return rows


def _row_from_encoding(index: int, encoding: Any, *, materialized: bool) -> ImagePlanRow:
    artifact = encoding.to_artifact_dict()
    observed = artifact["image_grid_thw"] if materialized else None
    return ImagePlanRow(
        row_id=encoding.example_id,
        row_index=index,
        example_id=encoding.example_id,
        image_path=str(encoding.image_path),
        declared_width=encoding.width,
        declared_height=encoding.height,
        decoded_width=encoding.width,
        decoded_height=encoding.height,
        patch_size=encoding.plan.patch_size,
        merge_size=encoding.plan.merge_size,
        temporal_patch_size=encoding.plan.temporal_patch_size,
        expected_image_grid_thw=list(encoding.image_grid_thw),
        observed_image_grid_thw=observed,
        raw_patch_rows=encoding.raw_patch_rows,
        merged_visual_tokens=encoding.merged_visual_tokens,
        do_resize=False,
        status="ok",
        error=None,
    )

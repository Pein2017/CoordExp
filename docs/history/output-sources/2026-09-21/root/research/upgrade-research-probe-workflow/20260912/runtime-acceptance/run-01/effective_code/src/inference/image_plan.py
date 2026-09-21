"""Backend-neutral semantic image planning for inference requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.errors import EncodingContractError
from src.config.models import ProcessorConfig
from src.data import RawExample
from src.qwen.images import (
    QwenNoResizeImagePlan,
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
    image_content_sha256: str
    patch_size: int
    merge_size: int
    temporal_patch_size: int
    expected_image_grid_thw: list[int]
    observed_image_grid_thw: list[int] | None
    raw_patch_rows: int
    merged_visual_tokens: int
    logical_transform_id: str
    do_resize: bool
    backend_projection_evidence_kind: str
    executed_media_sha256: str | None
    status: str
    error: dict[str, Any] | None
    backend_prompt_token_count: int | None = None
    backend_image_placeholder_ranges: list[dict[str, int]] | None = None

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
            "image_content_sha256": self.image_content_sha256,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "temporal_patch_size": self.temporal_patch_size,
            "expected_image_grid_thw": self.expected_image_grid_thw,
            "observed_image_grid_thw": self.observed_image_grid_thw,
            "raw_patch_rows": self.raw_patch_rows,
            "merged_visual_tokens": self.merged_visual_tokens,
            "logical_transform_id": self.logical_transform_id,
            "do_resize": self.do_resize,
            "backend_projection_evidence_kind": self.backend_projection_evidence_kind,
            "executed_media_sha256": self.executed_media_sha256,
            "status": self.status,
            "error": self.error,
            "backend_prompt_token_count": self.backend_prompt_token_count,
            "backend_image_placeholder_ranges": self.backend_image_placeholder_ranges,
        }


@dataclass(frozen=True)
class ImagePlanBatch:
    rows: list[ImagePlanRow]
    plans_by_row_id: dict[str, QwenNoResizeImagePlan]


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


def plan_image_rows(
    raw_examples: list[RawExample],
    *,
    components: Any,
    processor_config: ProcessorConfig,
    row_indices: list[int] | None = None,
) -> list[ImagePlanRow]:
    return plan_image_batch(
        raw_examples,
        components=components,
        processor_config=processor_config,
        row_indices=row_indices,
    ).rows


def plan_image_batch(
    raw_examples: list[RawExample],
    *,
    components: Any,
    processor_config: ProcessorConfig,
    row_indices: list[int] | None = None,
) -> ImagePlanBatch:
    encodings = [
        plan_qwen_image(
            raw_example,
            components=components,
            processor_config=processor_config,
        )
        for raw_example in raw_examples
    ]
    row_index_values = _resolve_row_indices(
        row_indices=row_indices,
        row_count=len(encodings),
    )
    rows = [
        _row_from_encoding(row_index, encoding)
        for row_index, encoding in zip(row_index_values, encodings, strict=True)
    ]
    return ImagePlanBatch(
        rows=rows,
        plans_by_row_id={
            str(encoding.example_id): encoding.plan for encoding in encodings
        },
    )


def write_image_plan_jsonl(
    raw_examples: list[RawExample],
    *,
    components: Any,
    processor_config: ProcessorConfig,
    output_path: Path,
    row_indices: list[int] | None = None,
) -> list[ImagePlanRow]:
    rows = plan_image_rows(
        raw_examples,
        components=components,
        processor_config=processor_config,
        row_indices=row_indices,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_artifact_dict(), sort_keys=True) + "\n")
    return rows


def _row_from_encoding(index: int, encoding: Any) -> ImagePlanRow:
    decoded_width = encoding.plan.decoded_width
    decoded_height = encoding.plan.decoded_height
    image_content_sha256 = encoding.plan.image_content_sha256
    if decoded_width is None or decoded_height is None or image_content_sha256 is None:
        raise EncodingContractError(
            "semantic image plan is missing decoded media identity",
            code="inference.image_plan_media_identity_missing",
            context={"row_id": encoding.example_id},
        )
    return ImagePlanRow(
        row_id=encoding.example_id,
        row_index=index,
        example_id=encoding.example_id,
        image_path=str(encoding.image_path),
        declared_width=encoding.width,
        declared_height=encoding.height,
        decoded_width=decoded_width,
        decoded_height=decoded_height,
        image_content_sha256=image_content_sha256,
        patch_size=encoding.plan.patch_size,
        merge_size=encoding.plan.merge_size,
        temporal_patch_size=encoding.plan.temporal_patch_size,
        expected_image_grid_thw=list(encoding.image_grid_thw),
        observed_image_grid_thw=None,
        raw_patch_rows=encoding.raw_patch_rows,
        merged_visual_tokens=encoding.merged_visual_tokens,
        logical_transform_id=encoding.plan.logical_transform_id,
        do_resize=False,
        backend_projection_evidence_kind="shared_reference_plan",
        executed_media_sha256=None,
        status="ok",
        error=None,
    )


def _resolve_row_indices(
    *,
    row_indices: list[int] | None,
    row_count: int,
) -> list[int]:
    if row_indices is None:
        return list(range(row_count))
    if len(row_indices) != row_count:
        raise EncodingContractError(
            "image plan row index count must match raw examples",
            code="inference.image_plan_row_index_count_mismatch",
            context={"row_index_count": len(row_indices), "row_count": row_count},
        )
    return [int(index) for index in row_indices]

"""Map norm-1000 object boxes to post-merger Qwen visual-token regions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch

from src.training.coverage_ledger.geometry import (
    norm1000_bbox_to_pixel_bbox,
    validate_norm1000_bbox_xyxy,
)


@dataclass(frozen=True, slots=True)
class VisualTokenRegion:
    """Half-open post-merger visual-token rectangle for one object bbox."""

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    flattened_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in ("row_start", "row_end", "col_start", "col_end"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.row_start >= self.row_end:
            raise ValueError("row interval must be non-empty and half-open")
        if self.col_start >= self.col_end:
            raise ValueError("col interval must be non-empty and half-open")
        indices = _freeze_non_empty_int_tuple(
            self.flattened_indices,
            field_name="flattened_indices",
        )
        object.__setattr__(self, "flattened_indices", indices)


def map_norm1000_bbox_to_visual_token_region(
    bbox_norm1000_xyxy: Sequence[Any],
    *,
    image_grid_thw: Any,
    processed_width: Any,
    processed_height: Any,
    patch_size: int,
    spatial_merge_size: int,
) -> VisualTokenRegion:
    """Return the minimal enclosing post-merger token-cell region for a bbox."""

    bbox = validate_norm1000_bbox_xyxy(bbox_norm1000_xyxy)
    grid_t, grid_h, grid_w = _normalize_single_image_grid(image_grid_thw)
    if grid_t != 1:
        raise ValueError("coverage ledger v0 supports exactly one image and one frame")

    merge_size = _require_positive_int(
        spatial_merge_size,
        field_name="spatial_merge_size",
    )
    patch = _require_positive_int(patch_size, field_name="patch_size")
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise ValueError("image_grid_thw h/w must be divisible by spatial_merge_size")

    post_rows = grid_h // merge_size
    post_cols = grid_w // merge_size
    width = _require_positive_finite_dimension(
        processed_width,
        field_name="processed_width",
    )
    height = _require_positive_finite_dimension(
        processed_height,
        field_name="processed_height",
    )
    if not width.is_integer():
        raise ValueError("processed_width must be an integer pixel dimension")
    if not height.is_integer():
        raise ValueError("processed_height must be an integer pixel dimension")
    expected_width = grid_w * patch
    expected_height = grid_h * patch
    if int(width) != expected_width:
        raise ValueError(
            "processed_width must equal grid_w * patch_size; "
            f"got processed_width={int(width)}, grid_w={grid_w}, patch_size={patch}"
        )
    if int(height) != expected_height:
        raise ValueError(
            "processed_height must equal grid_h * patch_size; "
            f"got processed_height={int(height)}, grid_h={grid_h}, patch_size={patch}"
        )
    cell_width = width / post_cols
    cell_height = height / post_rows
    expected_cell_size = patch * merge_size
    if cell_width != expected_cell_size:
        raise ValueError(
            "cell_width must equal patch_size * spatial_merge_size; "
            f"got cell_width={cell_width}, patch_size={patch}, "
            f"spatial_merge_size={merge_size}"
        )
    if cell_height != expected_cell_size:
        raise ValueError(
            "cell_height must equal patch_size * spatial_merge_size; "
            f"got cell_height={cell_height}, patch_size={patch}, "
            f"spatial_merge_size={merge_size}"
        )
    x1_px, y1_px, x2_px, y2_px = norm1000_bbox_to_pixel_bbox(
        bbox,
        width=int(width),
        height=int(height),
    )

    col_start = _clamp(math.floor(x1_px / cell_width), 0, post_cols - 1)
    row_start = _clamp(math.floor(y1_px / cell_height), 0, post_rows - 1)
    col_end = _clamp(math.ceil(x2_px / cell_width), col_start + 1, post_cols)
    row_end = _clamp(math.ceil(y2_px / cell_height), row_start + 1, post_rows)
    indices = tuple(
        row * post_cols + col
        for row in range(row_start, row_end)
        for col in range(col_start, col_end)
    )

    return VisualTokenRegion(
        row_start=row_start,
        row_end=row_end,
        col_start=col_start,
        col_end=col_end,
        flattened_indices=indices,
    )


def pool_object_visual_embeddings(
    image_embeds: torch.Tensor,
    regions: Sequence[VisualTokenRegion],
) -> torch.Tensor:
    """Average detached post-merger image embeddings over each mapped region."""

    if not isinstance(image_embeds, torch.Tensor):
        raise TypeError("image_embeds must be a torch.Tensor")
    if image_embeds.ndim < 2:
        raise ValueError("image_embeds must have shape (visual_tokens, ...)")
    if int(image_embeds.shape[0]) <= 0:
        raise ValueError("image_embeds must contain at least one visual token")

    frozen_regions = _freeze_regions(regions)
    detached = image_embeds.detach()
    pooled: list[torch.Tensor] = []
    token_count = int(detached.shape[0])
    for object_index, region in enumerate(frozen_regions):
        max_index = max(region.flattened_indices)
        if max_index >= token_count:
            raise ValueError(
                "region flattened_indices exceed image_embeds token count "
                f"for object {object_index}: max_index={max_index}, "
                f"image_embeds={token_count}"
            )
        index_tensor = torch.tensor(
            region.flattened_indices,
            dtype=torch.long,
            device=detached.device,
        )
        pooled.append(detached.index_select(0, index_tensor).mean(dim=0))
    return torch.stack(pooled, dim=0)


def offset_visual_token_region(
    region: VisualTokenRegion,
    visual_token_start: int,
) -> VisualTokenRegion:
    """Offset flattened visual-token indices by a cumulative visual-token start."""

    if type(region) is not VisualTokenRegion:
        raise TypeError("region must be a VisualTokenRegion")
    start = _require_non_negative_int(
        visual_token_start,
        field_name="visual_token_start",
    )
    return VisualTokenRegion(
        row_start=region.row_start,
        row_end=region.row_end,
        col_start=region.col_start,
        col_end=region.col_end,
        flattened_indices=tuple(
            int(index) + start for index in region.flattened_indices
        ),
    )


def _normalize_single_image_grid(image_grid_thw: Any) -> tuple[int, int, int]:
    if isinstance(image_grid_thw, torch.Tensor):
        if image_grid_thw.ndim == 1 and int(image_grid_thw.shape[0]) == 3:
            return _freeze_positive_int_triplet(
                image_grid_thw.detach().cpu().tolist(),
                field_name="image_grid_thw",
            )
        if image_grid_thw.ndim == 2 and int(image_grid_thw.shape[-1]) == 3:
            if int(image_grid_thw.shape[0]) != 1:
                raise ValueError(
                    "coverage ledger v0 supports exactly one image and one frame"
                )
            return _freeze_positive_int_triplet(
                image_grid_thw[0].detach().cpu().tolist(),
                field_name="image_grid_thw[0]",
            )
        raise ValueError("image_grid_thw must have shape (3,) or (1, 3)")

    if isinstance(image_grid_thw, (str, bytes)) or not isinstance(
        image_grid_thw,
        Sequence,
    ):
        raise TypeError("image_grid_thw must be a sequence or torch.Tensor")
    frozen = tuple(image_grid_thw)
    if len(frozen) == 3 and not _looks_like_sequence(frozen[0]):
        return _freeze_positive_int_triplet(frozen, field_name="image_grid_thw")
    if len(frozen) != 1:
        raise ValueError("coverage ledger v0 supports exactly one image and one frame")
    row = frozen[0]
    if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
        raise TypeError("image_grid_thw[0] must be a sequence")
    return _freeze_positive_int_triplet(row, field_name="image_grid_thw[0]")


def _freeze_positive_int_triplet(
    values: Sequence[Any],
    *,
    field_name: str,
) -> tuple[int, int, int]:
    frozen = tuple(values)
    if len(frozen) != 3:
        raise ValueError(f"{field_name} must contain exactly three integers")
    return tuple(
        _require_positive_int(value, field_name=f"{field_name}[{index}]")
        for index, value in enumerate(frozen)
    )


def _require_positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _require_non_negative_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


def _require_positive_finite_dimension(value: Any, *, field_name: str) -> float:
    parsed = _require_finite_real(value, field_name=field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _require_finite_real(value: Any, *, field_name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _freeze_non_empty_int_tuple(
    values: Sequence[Any],
    *,
    field_name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    frozen = tuple(values)
    if not frozen:
        raise ValueError(f"{field_name} must be non-empty")
    for index, value in enumerate(frozen):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field_name}[{index}] must be a non-negative integer")
    return frozen


def _freeze_regions(
    regions: Sequence[VisualTokenRegion],
) -> tuple[VisualTokenRegion, ...]:
    if isinstance(regions, (str, bytes)) or not isinstance(regions, Sequence):
        raise TypeError("regions must be a sequence")
    frozen = tuple(regions)
    if not frozen:
        raise ValueError("regions must be non-empty")
    for index, region in enumerate(frozen):
        if type(region) is not VisualTokenRegion:
            raise TypeError(
                "regions must contain VisualTokenRegion values; "
                f"got {type(region).__name__} at index {index}"
            )
    return frozen


def _looks_like_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


__all__ = [
    "VisualTokenRegion",
    "map_norm1000_bbox_to_visual_token_region",
    "offset_visual_token_region",
    "pool_object_visual_embeddings",
]

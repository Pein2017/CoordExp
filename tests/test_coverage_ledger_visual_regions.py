from __future__ import annotations

import pytest
import torch

from src.training.coverage_ledger.visual_regions import (
    map_norm1000_bbox_to_visual_token_region,
    pool_object_visual_embeddings,
)


def test_full_image_bbox_maps_to_every_post_merger_visual_token_cell() -> None:
    region = map_norm1000_bbox_to_visual_token_region(
        (0, 0, 999, 999),
        image_grid_thw=(1, 8, 8),
        processed_width=80,
        processed_height=80,
        patch_size=10,
        spatial_merge_size=2,
    )

    assert (region.row_start, region.row_end) == (0, 4)
    assert (region.col_start, region.col_end) == (0, 4)
    assert region.flattened_indices == tuple(range(16))


def test_centered_bbox_maps_to_minimal_enclosing_half_open_token_rectangle() -> None:
    region = map_norm1000_bbox_to_visual_token_region(
        (250, 250, 750, 750),
        image_grid_thw=(1, 8, 8),
        processed_width=80,
        processed_height=80,
        patch_size=10,
        spatial_merge_size=2,
    )

    assert (region.row_start, region.row_end) == (1, 3)
    assert (region.col_start, region.col_end) == (1, 3)
    assert region.flattened_indices == (5, 6, 9, 10)


def test_tiny_valid_bbox_maps_to_at_least_one_cell_after_clamping() -> None:
    region = map_norm1000_bbox_to_visual_token_region(
        (1, 1, 2, 2),
        image_grid_thw=(1, 8, 8),
        processed_width=80,
        processed_height=80,
        patch_size=10,
        spatial_merge_size=2,
    )

    assert (region.row_start, region.row_end) == (0, 1)
    assert (region.col_start, region.col_end) == (0, 1)
    assert region.flattened_indices == (0,)


def test_right_bottom_edge_bbox_clamps_within_grid_bounds() -> None:
    region = map_norm1000_bbox_to_visual_token_region(
        (998, 998, 999, 999),
        image_grid_thw=(1, 8, 8),
        processed_width=80,
        processed_height=80,
        patch_size=10,
        spatial_merge_size=2,
    )

    assert (region.row_start, region.row_end) == (3, 4)
    assert (region.col_start, region.col_end) == (3, 4)
    assert region.flattened_indices == (15,)


@pytest.mark.parametrize(
    "bbox",
    (
        (10, 20, 10, 40),
        (10, 20, 30, 20),
        (-1, 20, 30, 40),
        (10, 20, 1000, 40),
        (10, 20, 1001, 40),
    ),
)
def test_degenerate_or_out_of_bounds_bbox_hard_fails(
    bbox: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError, match="bbox_norm1000_xyxy"):
        map_norm1000_bbox_to_visual_token_region(
            bbox,
            image_grid_thw=(1, 8, 8),
            processed_width=80,
            processed_height=80,
            patch_size=10,
            spatial_merge_size=2,
        )


def test_processed_dimensions_inconsistent_with_grid_and_cell_size_hard_fail() -> None:
    with pytest.raises(ValueError, match="processed_width.*grid_w.*patch_size"):
        map_norm1000_bbox_to_visual_token_region(
            (0, 0, 999, 999),
            image_grid_thw=(1, 8, 8),
            processed_width=82,
            processed_height=80,
            patch_size=10,
            spatial_merge_size=2,
        )


def test_processed_dimensions_inconsistent_with_pre_merge_patch_size_hard_fail() -> None:
    with pytest.raises(ValueError, match="processed_width.*grid_w.*patch_size"):
        map_norm1000_bbox_to_visual_token_region(
            (0, 0, 999, 999),
            image_grid_thw=(1, 8, 8),
            processed_width=84,
            processed_height=80,
            patch_size=10,
            spatial_merge_size=2,
        )


def test_mismatched_width_height_patch_geometry_hard_fails() -> None:
    with pytest.raises(ValueError, match="processed_width.*grid_w.*patch_size"):
        map_norm1000_bbox_to_visual_token_region(
            (0, 0, 999, 999),
            image_grid_thw=(1, 30, 40),
            processed_width=800,
            processed_height=480,
            patch_size=16,
            spatial_merge_size=2,
        )


@pytest.mark.parametrize(
    "image_grid_thw",
    (
        (2, 8, 8),
        ((1, 8, 8), (1, 8, 8)),
    ),
)
def test_multi_frame_or_multi_image_grid_hard_fails_in_v0(
    image_grid_thw: object,
) -> None:
    with pytest.raises(ValueError, match="exactly one image and one frame"):
        map_norm1000_bbox_to_visual_token_region(
            (0, 0, 999, 999),
            image_grid_thw=image_grid_thw,
            processed_width=80,
            processed_height=80,
            patch_size=10,
            spatial_merge_size=2,
        )


def test_pooling_detaches_gathered_embeddings_and_returns_average_over_indices() -> None:
    image_embeds = torch.arange(24, dtype=torch.float32, requires_grad=True).reshape(6, 4)
    region = map_norm1000_bbox_to_visual_token_region(
        (0, 0, 999, 500),
        image_grid_thw=torch.tensor([[1, 4, 6]], dtype=torch.long),
        processed_width=60,
        processed_height=40,
        patch_size=10,
        spatial_merge_size=2,
    )

    pooled = pool_object_visual_embeddings(image_embeds, (region,))

    assert pooled.shape == (1, 4)
    assert torch.equal(pooled[0], image_embeds.detach()[[0, 1, 2]].mean(dim=0))
    assert pooled.requires_grad is False

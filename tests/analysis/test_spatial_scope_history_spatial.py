from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
from pathlib import Path
from typing import Any

from PIL import Image
import pytest
import torch

from src.common.errors import DataContractError, EncodingContractError
from src.analysis.spatial_scope_history import (
    MaterializedVisualInput,
    SpatialGrid,
    SpatialGridSpec,
)
from src.analysis.spatial_scope_history.cohort_ledger import sha256_file
from src.config.fingerprint import sha256_json
from src.config.loader import load_train_config
from src.data import load_raw_examples
from src.qwen.loading import load_qwen_components


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


def test_primary_grid_partitions_merged_tokens_with_row_major_core_and_halo() -> None:
    grid = SpatialGrid.build(source_width=320, source_height=320)

    assert grid.spec == SpatialGridSpec()
    assert grid.merged_token_width == 10
    assert grid.merged_token_height == 10
    assert len(grid.cells) == 16
    assert [cell.index for cell in grid.cells] == list(range(16))

    cell = grid.cell(5)
    assert (cell.row_index, cell.column_index) == (1, 1)
    assert cell.core_token_xyxy == (2, 2, 5, 5)
    assert cell.halo_token_xyxy == (1, 1, 6, 6)
    assert cell.core_pixel_xyxy == (64, 64, 160, 160)
    assert cell.halo_pixel_xyxy == (32, 32, 192, 192)

    corner = grid.cell(0)
    assert corner.core_token_xyxy == (0, 0, 2, 2)
    assert corner.halo_token_xyxy == (0, 0, 3, 3)


def test_grid_rejects_non_quantized_or_empty_core_source_canvas() -> None:
    with pytest.raises(DataContractError) as quantum_exc:
        SpatialGrid.build(source_width=321, source_height=320)
    assert quantum_exc.value.code == "analysis.spatial_grid_quantum_divisibility"

    with pytest.raises(DataContractError) as core_exc:
        SpatialGrid.build(source_width=96, source_height=128)
    assert core_exc.value.code == "analysis.spatial_grid_core_extent"


def test_half_open_ownership_assigns_internal_boundaries_to_higher_index() -> None:
    grid = SpatialGrid.build(source_width=320, source_height=320)

    horizontal_boundary = grid.ownership((63, 31, 65, 33))
    assert horizontal_boundary.is_valid is True
    assert horizontal_boundary.center_xy == (64.0, 32.0)
    assert horizontal_boundary.owner_cell_index == 1

    two_dimensional_boundary = grid.ownership((63, 63, 65, 65))
    assert two_dimensional_boundary.center_xy == (64.0, 64.0)
    assert two_dimensional_boundary.owner_cell_index == 5


@pytest.mark.parametrize(
    ("bbox", "reason"),
    [
        ((0, 0, math.inf, 4), "non_finite_or_malformed_box"),
        ((4, 4, 4, 8), "empty_area"),
        ((319, 20, 321, 30), "center_outside_source_canvas"),
    ],
)
def test_invalid_or_outside_boxes_are_unowned(
    bbox: tuple[float, float, float, float],
    reason: str,
) -> None:
    ownership = SpatialGrid.build(source_width=320, source_height=320).ownership(bbox)

    assert ownership.is_valid is False
    assert ownership.owner_cell_index is None
    assert ownership.reason == reason


def test_native_tile_is_core_plus_halo_without_resize() -> None:
    source = _coordinate_image(256, 256)
    grid = SpatialGrid.build(source_width=256, source_height=256)
    plan = grid.plan(cell_index=5, variant_mode="tile_reset")

    encoding = plan.materialize(source)
    expected = source.crop((32, 32, 160, 160))
    try:
        assert plan.do_resize is False
        assert plan.output_width == 128
        assert plan.output_height == 128
        assert encoding.width == 128
        assert encoding.height == 128
        assert encoding.rgb_bytes == expected.convert("RGB").tobytes()
        assert encoding.channel_mean_rgb is None
        assert encoding.expected_image_grid_thw == (1, 8, 8)
        assert encoding.raw_patch_rows == 64
        assert encoding.total_merged_visual_tokens == 16
        assert encoding.visible_merged_visual_tokens == 16
        assert encoding.merged_visual_tokens == 16
    finally:
        expected.close()
        source.close()


@pytest.mark.parametrize(
    ("variant_mode", "expected_total", "expected_visible"),
    [
        ("tile_reset", 16, 16),
        ("mask_reset", 64, 16),
        ("mask_cumulative", 64, 16),
    ],
)
def test_total_and_visible_merged_tokens_reconcile_by_spatial_arm(
    variant_mode: Any,
    expected_total: int,
    expected_visible: int,
) -> None:
    source = _coordinate_image(256, 256)
    plan = SpatialGrid.build(source_width=256, source_height=256).plan(
        cell_index=5,
        variant_mode=variant_mode,
    )
    try:
        encoding = plan.materialize(source)
    finally:
        source.close()

    halo_left, halo_top, halo_right, halo_bottom = plan.cell.halo_token_xyxy
    halo_support = (halo_right - halo_left) * (halo_bottom - halo_top)
    assert encoding.total_merged_visual_tokens == expected_total
    assert encoding.visible_merged_visual_tokens == expected_visible
    assert encoding.visible_merged_visual_tokens == halo_support
    receipt = encoding.verify_processor_output(
        pixel_values=torch.zeros(
            (encoding.raw_patch_rows, 1536),
            dtype=torch.float32,
        ),
        image_grid_thw=torch.tensor(
            [encoding.expected_image_grid_thw],
            dtype=torch.long,
        ),
        **_processor_evidence(),
    )
    assert receipt.total_merged_visual_tokens == expected_total
    assert receipt.visible_merged_visual_tokens == expected_visible
    if variant_mode == "tile_reset":
        assert (
            encoding.total_merged_visual_tokens == encoding.visible_merged_visual_tokens
        )
    else:
        assert (
            encoding.total_merged_visual_tokens > encoding.visible_merged_visual_tokens
        )


def test_masked_full_canvas_uses_float32_ties_to_even_rgb_mean() -> None:
    source = Image.new("RGB", (128, 128), color=(0, 1, 255))
    right_half = Image.new("RGB", (64, 128), color=(1, 2, 255))
    source.paste(right_half, (64, 0))
    right_half.close()
    grid = SpatialGrid.build(source_width=128, source_height=128)
    reset = grid.plan(cell_index=5, variant_mode="mask_reset").materialize(source)
    cumulative = grid.plan(
        cell_index=5,
        variant_mode="mask_cumulative",
    ).materialize(source)

    reset_image = reset.to_pil_image()
    cumulative_image = cumulative.to_pil_image()
    try:
        assert reset.channel_mean_rgb == (0, 2, 255)
        assert reset.width == source.width
        assert reset.height == source.height
        assert reset.rgb_bytes == cumulative.rgb_bytes
        assert reset_image.getpixel((127, 127)) == (0, 2, 255)
        assert reset_image.getpixel((40, 40)) == source.getpixel((40, 40))
        assert cumulative_image.size == source.size
    finally:
        reset_image.close()
        cumulative_image.close()
        source.close()


def test_tile_local_coordinate_receipt_preserves_every_conversion_stage() -> None:
    grid = SpatialGrid.build(source_width=256, source_height=256)
    tile_plan = grid.plan(cell_index=5, variant_mode="tile_reset")

    receipt = tile_plan.coordinate_receipt((0, 0, 500, 500))

    assert receipt.original_coordinate_bins == (0, 0, 500, 500)
    assert (receipt.local_extent_width, receipt.local_extent_height) == (128, 128)
    assert receipt.local_integer_box == (0, 0, 64, 64)
    assert receipt.tile_origin_xy == (32, 32)
    assert receipt.unclipped_global_integer_box == (32, 32, 96, 96)
    assert receipt.clipped_global_integer_box == (32, 32, 96, 96)

    full_receipt = grid.plan(
        cell_index=5,
        variant_mode="mask_reset",
    ).coordinate_receipt((0, 0, 500, 500))
    assert full_receipt.local_integer_box == (0, 0, 128, 128)
    assert full_receipt.tile_origin_xy == (0, 0)
    assert full_receipt.clipped_global_integer_box == (0, 0, 128, 128)


def test_processor_shape_and_token_grid_receipt_matches_qwen_no_resize_semantics() -> (
    None
):
    source = _coordinate_image(256, 256)
    encoding = (
        SpatialGrid.build(source_width=256, source_height=256)
        .plan(
            cell_index=5,
            variant_mode="tile_reset",
        )
        .materialize(source)
    )
    source.close()

    processor_evidence = _processor_evidence()
    receipt = encoding.verify_processor_output(
        pixel_values=torch.zeros((64, 1536), dtype=torch.float32),
        image_grid_thw=torch.tensor([[1, 8, 8]], dtype=torch.long),
        **processor_evidence,
    )

    assert receipt.to_artifact_dict() == {
        "spatial_image_encoding_sha256": encoding.fingerprint,
        "do_resize": False,
        "executed_processor_config": processor_evidence["executed_processor_config"],
        "executed_processor_identity": processor_evidence[
            "executed_processor_identity"
        ],
        "executed_processor_contract_sha256": processor_evidence[
            "executed_processor_contract_sha256"
        ],
        "expected_processor_contract_sha256": processor_evidence[
            "expected_processor_contract_sha256"
        ],
        "image_grid_thw": [1, 8, 8],
        "raw_patch_rows": 64,
        "total_merged_visual_tokens": 16,
        "visible_merged_visual_tokens": 16,
        "pixel_values_shape": [64, 1536],
        "executed_visual_tensors": (receipt.executed_visual_tensors.to_artifact_dict()),
        "receipt_sha256": receipt.receipt_sha256,
    }


def test_processor_receipt_binds_exact_tensor_values_dtype_and_shape() -> None:
    source = _coordinate_image(256, 256)
    encoding = (
        SpatialGrid.build(source_width=256, source_height=256)
        .plan(
            cell_index=5,
            variant_mode="tile_reset",
        )
        .materialize(source)
    )
    source.close()
    evidence = _processor_evidence()

    zero = encoding.verify_processor_output(
        pixel_values=torch.zeros((64, 1536), dtype=torch.float32),
        image_grid_thw=torch.tensor([[1, 8, 8]], dtype=torch.int64),
        **evidence,
    )
    one = encoding.verify_processor_output(
        pixel_values=torch.ones((64, 1536), dtype=torch.float32),
        image_grid_thw=torch.tensor([[1, 8, 8]], dtype=torch.int64),
        **evidence,
    )
    float64 = encoding.verify_processor_output(
        pixel_values=torch.zeros((64, 1536), dtype=torch.float64),
        image_grid_thw=torch.tensor([[1, 8, 8]], dtype=torch.int64),
        **evidence,
    )

    assert (
        zero.executed_visual_tensors.pixel_values.canonical_content_sha256
        != one.executed_visual_tensors.pixel_values.canonical_content_sha256
    )
    assert zero.executed_visual_tensors.pixel_values.dtype == "float32"
    assert float64.executed_visual_tensors.pixel_values.dtype == "float64"
    assert (
        zero.executed_visual_tensors.receipt_sha256
        != one.executed_visual_tensors.receipt_sha256
        != float64.executed_visual_tensors.receipt_sha256
    )


def test_processor_grid_or_pixel_shape_mismatch_fails_before_use() -> None:
    source = _coordinate_image(256, 256)
    encoding = (
        SpatialGrid.build(source_width=256, source_height=256)
        .plan(
            cell_index=5,
            variant_mode="tile_reset",
        )
        .materialize(source)
    )
    source.close()
    processor_evidence = _processor_evidence()

    with pytest.raises(EncodingContractError) as grid_exc:
        encoding.verify_processor_output(
            pixel_values=torch.zeros((64, 1536), dtype=torch.float32),
            image_grid_thw=torch.tensor([[1, 7, 8]], dtype=torch.long),
            **processor_evidence,
        )
    assert grid_exc.value.code == "analysis.spatial_processor_grid_mismatch"

    with pytest.raises(EncodingContractError) as pixel_exc:
        encoding.verify_processor_output(
            pixel_values=torch.zeros((63, 1536), dtype=torch.float32),
            image_grid_thw=torch.tensor([[1, 8, 8]], dtype=torch.long),
            **processor_evidence,
        )
    assert pixel_exc.value.code == "analysis.spatial_processor_pixel_shape"


def test_spatial_plan_and_processor_receipt_fail_closed_on_resize() -> None:
    grid = SpatialGrid.build(source_width=256, source_height=256)
    plan = grid.plan(cell_index=5, variant_mode="tile_reset")

    for invalid_do_resize in (True, 0):
        with pytest.raises(EncodingContractError) as replace_exc:
            replace(plan, do_resize=invalid_do_resize)
        assert replace_exc.value.code == "analysis.spatial_resize_forbidden"

    source = _coordinate_image(256, 256)
    encoding = plan.materialize(source)
    source.close()
    object.__setattr__(plan, "do_resize", True)

    second_source = _coordinate_image(256, 256)
    with pytest.raises(EncodingContractError) as materialize_exc:
        plan.materialize(second_source)
    second_source.close()
    assert materialize_exc.value.code == "analysis.spatial_resize_forbidden"
    with pytest.raises(EncodingContractError) as receipt_exc:
        encoding.verify_processor_output(
            pixel_values=torch.zeros((64, 1536), dtype=torch.float32),
            image_grid_thw=torch.tensor([[1, 8, 8]], dtype=torch.long),
            **_processor_evidence(),
        )
    assert receipt_exc.value.code == "analysis.spatial_resize_forbidden"


@pytest.mark.parametrize(
    ("processor_config", "expected_code"),
    [
        ({"do_resize": True}, "analysis.spatial_processor_resize_executed"),
        ({"do_resize": 0}, "analysis.spatial_processor_resize_executed"),
        ({}, "analysis.spatial_processor_resize_missing"),
    ],
)
def test_executed_processor_resize_attestation_must_be_exact_false(
    processor_config: dict[str, Any],
    expected_code: str,
) -> None:
    source = _coordinate_image(256, 256)
    encoding = (
        SpatialGrid.build(source_width=256, source_height=256)
        .plan(
            cell_index=5,
            variant_mode="tile_reset",
        )
        .materialize(source)
    )
    source.close()
    evidence = _processor_evidence(processor_config=processor_config)

    with pytest.raises(EncodingContractError) as exc_info:
        encoding.verify_processor_output(
            pixel_values=torch.zeros((64, 1536), dtype=torch.float32),
            image_grid_thw=torch.tensor([[1, 8, 8]], dtype=torch.long),
            **evidence,
        )
    assert exc_info.value.code == expected_code


def test_processor_contract_digest_binding_and_expected_identity_drift_fail() -> None:
    source = _coordinate_image(256, 256)
    encoding = (
        SpatialGrid.build(source_width=256, source_height=256)
        .plan(
            cell_index=5,
            variant_mode="tile_reset",
        )
        .materialize(source)
    )
    source.close()
    evidence = _processor_evidence()

    with pytest.raises(EncodingContractError) as binding_exc:
        encoding.verify_processor_output(
            pixel_values=torch.zeros((64, 1536), dtype=torch.float32),
            image_grid_thw=torch.tensor([[1, 8, 8]], dtype=torch.long),
            **{
                **evidence,
                "executed_processor_contract_sha256": "f" * 64,
            },
        )
    assert (
        binding_exc.value.code == "analysis.spatial_processor_contract_digest_mismatch"
    )

    with pytest.raises(EncodingContractError) as drift_exc:
        encoding.verify_processor_output(
            pixel_values=torch.zeros((64, 1536), dtype=torch.float32),
            image_grid_thw=torch.tensor([[1, 8, 8]], dtype=torch.long),
            **{
                **evidence,
                "expected_processor_contract_sha256": "0" * 64,
            },
        )
    assert drift_exc.value.code == "analysis.spatial_processor_contract_drift"


def test_installed_qwen_processor_preserves_real_image_no_resize_spatial_receipt() -> (
    None
):
    resolved = load_train_config(FIXTURE_CONFIG)
    example = load_raw_examples(resolved.config.data.train)[0]
    components = load_qwen_components(resolved.config, load_model=False)
    grid = SpatialGrid.build(
        source_width=example.image.width,
        source_height=example.image.height,
    )
    plan = grid.plan(cell_index=5, variant_mode="tile_reset")
    with Image.open(example.image.path) as source:
        encoding = plan.materialize(source)
    tile_image = encoding.to_pil_image()
    try:
        processed = components.processor.image_processor(
            images=[tile_image],
            return_tensors="pt",
            do_resize=False,
        )
    finally:
        tile_image.close()
    processor_config = {
        "do_resize": False,
        "max_raw_pixels": resolved.config.model.processor.max_raw_pixels,
        "max_merged_visual_tokens": (
            resolved.config.model.processor.max_merged_visual_tokens
        ),
    }
    processor_identity = components.processor_identity.to_artifact_dict()
    processor_evidence = _processor_evidence(
        processor_config=processor_config,
        processor_identity=processor_identity,
    )

    receipt = encoding.verify_processor_output(
        pixel_values=processed["pixel_values"],
        image_grid_thw=processed["image_grid_thw"],
        **processor_evidence,
    )

    assert receipt.do_resize is False
    assert dict(receipt.executed_processor_config) == processor_config
    assert dict(receipt.executed_processor_identity) == processor_identity
    assert receipt.total_merged_visual_tokens == encoding.total_merged_visual_tokens
    assert receipt.visible_merged_visual_tokens == encoding.visible_merged_visual_tokens


def test_spatial_contract_records_are_immutable() -> None:
    grid = SpatialGrid.build(source_width=128, source_height=128)

    with pytest.raises(FrozenInstanceError):
        grid.source_width = 256  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        grid.spec.row_count = 2  # type: ignore[misc]


@pytest.mark.parametrize("input_kind", ["full_image", "tile_reset", "mask_reset"])
def test_processor_owned_materialization_rejects_same_shape_swapped_tensors(
    tmp_path: Path,
    input_kind: str,
) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.new("RGB", (128, 128), color=(10, 20, 30)).save(first_path)
    Image.new("RGB", (128, 128), color=(200, 100, 50)).save(second_path)

    def content_processor(*, images, return_tensors, do_resize):
        assert return_tensors == "pt"
        assert do_resize is False
        image = images[0]
        width, height = image.size
        content_value = float(sum(image.getpixel((0, 0))))
        return {
            "pixel_values": torch.full(
                ((height // 16) * (width // 16), 1536), content_value
            ),
            "image_grid_thw": torch.tensor([[1, height // 16, width // 16]]),
        }

    processor_digest = sha256_json({"processor": "content-sensitive-test"})
    first_sha = sha256_file(first_path)
    second_sha = sha256_file(second_path)
    if input_kind == "full_image":
        first = MaterializedVisualInput.from_full_image_path(
            source_image_path=first_path,
            expected_source_image_sha256=first_sha,
            image_processor=content_processor,
            processor_contract_sha256=processor_digest,
        )
        second = MaterializedVisualInput.from_full_image_path(
            source_image_path=second_path,
            expected_source_image_sha256=second_sha,
            image_processor=content_processor,
            processor_contract_sha256=processor_digest,
        )
    else:
        plan = SpatialGrid.build(source_width=128, source_height=128).plan(
            cell_index=5,
            variant_mode=input_kind,  # type: ignore[arg-type]
        )
        first = MaterializedVisualInput.from_spatial_plan_path(
            plan=plan,
            source_image_path=first_path,
            expected_source_image_sha256=first_sha,
            image_processor=content_processor,
            processor_contract_sha256=processor_digest,
        )
        second = MaterializedVisualInput.from_spatial_plan_path(
            plan=plan,
            source_image_path=second_path,
            expected_source_image_sha256=second_sha,
            image_processor=content_processor,
            processor_contract_sha256=processor_digest,
        )

    assert first.pixel_values.shape == second.pixel_values.shape
    assert first.receipt.receipt_sha256 != second.receipt.receipt_sha256
    with pytest.raises(EncodingContractError) as direct_exc:
        MaterializedVisualInput(
            receipt=first.receipt,
            pixel_values=first.pixel_values,
            image_grid_thw=first.image_grid_thw,
            spatial_image_encoding=first.spatial_image_encoding,
        )
    assert direct_exc.value.code == "analysis.visual_materialization_mint_authority"
    with pytest.raises(EncodingContractError) as receipt_exc:
        replace(first.receipt, _mint_capability=object())
    assert receipt_exc.value.code == "analysis.visual_materialization_mint_authority"
    with pytest.raises(EncodingContractError) as copied_receipt_exc:
        replace(first.receipt)
    assert (
        copied_receipt_exc.value.code
        == "analysis.visual_materialization_mint_authority"
    )
    with pytest.raises(EncodingContractError) as exc_info:
        first.verify_model_inputs(
            {
                "pixel_values": second.pixel_values,
                "image_grid_thw": second.image_grid_thw,
            }
        )
    assert (
        exc_info.value.code
        == "analysis.visual_materialization_model_input_mismatch"
    )


def _processor_evidence(
    *,
    processor_config: dict[str, Any] | None = None,
    processor_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_config = (
        {"do_resize": False} if processor_config is None else dict(processor_config)
    )
    resolved_identity = (
        {
            "processor_class": "TestQwen3VLProcessor",
            "image_processor_class": "TestQwen2VLImageProcessorFast",
            "patch_size": 16,
            "merge_size": 2,
            "temporal_patch_size": 2,
        }
        if processor_identity is None
        else dict(processor_identity)
    )
    digest = sha256_json(
        {
            "processor_config": resolved_config,
            "processor_identity": resolved_identity,
        }
    )
    return {
        "executed_processor_config": resolved_config,
        "executed_processor_identity": resolved_identity,
        "executed_processor_contract_sha256": digest,
        "expected_processor_contract_sha256": digest,
    }


def _coordinate_image(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    assert pixels is not None
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return image

"""CPU-only spatial construction and geometry receipts for research inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from PIL import Image
import torch

from src.common.errors import (
    ConfigContractError,
    DataContractError,
    EncodingContractError,
)
from src.config.fingerprint import sha256_json
from src.data.geometry import coord_bins_to_pixel_xyxy, validate_bbox_bins
from src.analysis.spatial_scope_history.cohort_ledger import (
    sha256_file,
    sha256_payload,
)
from src.analysis.spatial_scope_history.schedule import ResearchArmDefinition


SpatialVariantMode = Literal["tile_reset", "mask_reset", "mask_cumulative"]
PixelBox = tuple[int, int, int, int]
TokenBox = tuple[int, int, int, int]
SPATIAL_GRID_SPEC_SCHEMA_VERSION = "spatial_scope_history.grid_spec.v1"
TENSOR_CONTENT_RECEIPT_SCHEMA_VERSION = (
    "spatial_scope_history.tensor_content_receipt.v1"
)
EXECUTED_VISUAL_TENSOR_RECEIPT_SCHEMA_VERSION = (
    "spatial_scope_history.executed_visual_tensor_receipt.v1"
)
_VISUAL_MATERIALIZATION_MINT_CAPABILITY = object()
VISUAL_INPUT_MATERIALIZATION_RECEIPT_SCHEMA_VERSION = (
    "spatial_scope_history.visual_input_materialization_receipt.v1"
)


def spatial_variant_mode_for_arm(
    arm: ResearchArmDefinition,
) -> SpatialVariantMode | None:
    """Derive spatial execution semantics from the schedule-owned arm record."""

    if arm.input_policy == "complete_source_canvas":
        return None
    if arm.input_policy == "native_scale_core_plus_halo_tile":
        return "tile_reset"
    if arm.input_policy == "full_canvas_core_plus_halo_mask":
        return "mask_cumulative" if arm.cumulative_dependency else "mask_reset"
    raise DataContractError(
        "schedule arm has unsupported spatial semantics",
        code="analysis.spatial_arm_geometry",
        context={
            "arm_code": arm.arm_code,
            "input_policy": arm.input_policy,
            "cumulative_dependency": arm.cumulative_dependency,
        },
    )


@dataclass(frozen=True)
class SpatialGridSpec:
    """Merged-token grid policy; defaults are the frozen primary experiment."""

    visual_quantum_pixels: int = 32
    row_count: int = 4
    column_count: int = 4
    halo_fraction_numerator: int = 1
    halo_fraction_denominator: int = 4

    def __post_init__(self) -> None:
        for field, value in (
            ("visual_quantum_pixels", self.visual_quantum_pixels),
            ("row_count", self.row_count),
            ("column_count", self.column_count),
            ("halo_fraction_numerator", self.halo_fraction_numerator),
            ("halo_fraction_denominator", self.halo_fraction_denominator),
        ):
            _require_positive_integer(value, field=field)
        if self.halo_fraction_numerator > self.halo_fraction_denominator:
            raise DataContractError(
                "halo fraction must not exceed one whole core extent",
                code="analysis.spatial_grid_halo_fraction",
                context={
                    "numerator": self.halo_fraction_numerator,
                    "denominator": self.halo_fraction_denominator,
                },
            )

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_artifact_dict())

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "column_count": self.column_count,
            "halo_fraction_denominator": self.halo_fraction_denominator,
            "halo_fraction_numerator": self.halo_fraction_numerator,
            "row_count": self.row_count,
            "schema_version": SPATIAL_GRID_SPEC_SCHEMA_VERSION,
            "visual_quantum_pixels": self.visual_quantum_pixels,
        }


@dataclass(frozen=True)
class SpatialCell:
    """One row-major core and its clipped context halo."""

    index: int
    row_index: int
    column_index: int
    core_token_xyxy: TokenBox
    halo_token_xyxy: TokenBox
    core_pixel_xyxy: PixelBox
    halo_pixel_xyxy: PixelBox

    def __post_init__(self) -> None:
        _validate_positive_area_box(self.core_token_xyxy, field="core_token_xyxy")
        _validate_positive_area_box(self.halo_token_xyxy, field="halo_token_xyxy")
        _validate_positive_area_box(self.core_pixel_xyxy, field="core_pixel_xyxy")
        _validate_positive_area_box(self.halo_pixel_xyxy, field="halo_pixel_xyxy")


@dataclass(frozen=True)
class SpatialOwnership:
    """Unique half-open core ownership result for one global pixel box."""

    is_valid: bool
    owner_cell_index: int | None
    center_xy: tuple[float, float] | None
    reason: str


@dataclass(frozen=True)
class SpatialCoordinateReceipt:
    """Auditable local-bin to global-pixel coordinate conversion."""

    variant_mode: SpatialVariantMode
    cell_index: int
    original_coordinate_bins: tuple[int, int, int, int]
    local_extent_width: int
    local_extent_height: int
    local_integer_box: PixelBox
    tile_origin_xy: tuple[int, int]
    unclipped_global_integer_box: PixelBox
    clipped_global_integer_box: PixelBox
    source_width: int
    source_height: int


@dataclass(frozen=True)
class SpatialGrid:
    """A validated source-canvas partition over merged visual tokens."""

    spec: SpatialGridSpec
    source_width: int
    source_height: int
    merged_token_width: int
    merged_token_height: int
    cells: tuple[SpatialCell, ...]

    @classmethod
    def build(
        cls,
        *,
        source_width: int,
        source_height: int,
        spec: SpatialGridSpec | None = None,
    ) -> SpatialGrid:
        resolved_spec = spec or SpatialGridSpec()
        _require_positive_integer(source_width, field="source_width")
        _require_positive_integer(source_height, field="source_height")
        quantum = resolved_spec.visual_quantum_pixels
        if source_width % quantum != 0 or source_height % quantum != 0:
            raise DataContractError(
                "source canvas dimensions must be divisible by the visual quantum",
                code="analysis.spatial_grid_quantum_divisibility",
                context={
                    "source_width": source_width,
                    "source_height": source_height,
                    "visual_quantum_pixels": quantum,
                },
            )
        token_width = source_width // quantum
        token_height = source_height // quantum
        if (
            token_width < resolved_spec.column_count
            or token_height < resolved_spec.row_count
        ):
            raise DataContractError(
                "merged-token grid is too small to give every core positive area",
                code="analysis.spatial_grid_core_extent",
                context={
                    "merged_token_width": token_width,
                    "merged_token_height": token_height,
                    "row_count": resolved_spec.row_count,
                    "column_count": resolved_spec.column_count,
                },
            )
        cells = tuple(
            _build_cell(
                spec=resolved_spec,
                row_index=row_index,
                column_index=column_index,
                token_width=token_width,
                token_height=token_height,
            )
            for row_index in range(resolved_spec.row_count)
            for column_index in range(resolved_spec.column_count)
        )
        return cls(
            spec=resolved_spec,
            source_width=source_width,
            source_height=source_height,
            merged_token_width=token_width,
            merged_token_height=token_height,
            cells=cells,
        )

    def cell(self, index: int) -> SpatialCell:
        if isinstance(index, bool) or not isinstance(index, int):
            raise DataContractError(
                "spatial cell index must be an integer",
                code="analysis.spatial_cell_index_type",
                context={"cell_index": index},
            )
        if index < 0 or index >= len(self.cells):
            raise DataContractError(
                "spatial cell index is out of range",
                code="analysis.spatial_cell_index_range",
                context={"cell_index": index, "cell_count": len(self.cells)},
            )
        return self.cells[index]

    def plan(
        self,
        *,
        cell_index: int,
        variant_mode: SpatialVariantMode,
    ) -> SpatialVariantPlan:
        cell = self.cell(cell_index)
        if variant_mode not in {"tile_reset", "mask_reset", "mask_cumulative"}:
            raise DataContractError(
                "unsupported spatial variant mode",
                code="analysis.spatial_variant_mode",
                context={"variant_mode": variant_mode},
            )
        if variant_mode == "tile_reset":
            left, top, right, bottom = cell.halo_pixel_xyxy
            output_width = right - left
            output_height = bottom - top
        else:
            output_width = self.source_width
            output_height = self.source_height
        return SpatialVariantPlan(
            variant_mode=variant_mode,
            grid_spec=self.spec,
            cell=cell,
            source_width=self.source_width,
            source_height=self.source_height,
            output_width=output_width,
            output_height=output_height,
        )

    def ownership(self, global_pixel_box: Sequence[Any]) -> SpatialOwnership:
        parsed = _finite_box(global_pixel_box)
        if parsed is None:
            return SpatialOwnership(False, None, None, "non_finite_or_malformed_box")
        x1, y1, x2, y2 = parsed
        if x1 >= x2 or y1 >= y2:
            return SpatialOwnership(False, None, None, "empty_area")
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        center = (center_x, center_y)
        if not (
            0.0 <= center_x < self.source_width and 0.0 <= center_y < self.source_height
        ):
            return SpatialOwnership(False, None, center, "center_outside_source_canvas")
        for cell in self.cells:
            left, top, right, bottom = cell.core_pixel_xyxy
            if left <= center_x < right and top <= center_y < bottom:
                return SpatialOwnership(True, cell.index, center, "owned")
        return SpatialOwnership(False, None, center, "no_core_owner")


@dataclass(frozen=True)
class SpatialVariantPlan:
    """One native tile or full-canvas masked input for one spatial cell."""

    variant_mode: SpatialVariantMode
    grid_spec: SpatialGridSpec
    cell: SpatialCell
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    do_resize: bool = False

    def __post_init__(self) -> None:
        self._require_no_resize()

    def materialize(self, source_image: Image.Image) -> SpatialImageEncoding:
        self._require_no_resize()
        return SpatialImageEncoding.from_source_image(
            plan=self, source_image=source_image
        )

    def materialize_path(self, source_image_path: str | Path) -> SpatialImageEncoding:
        """Materialize while binding the exact source-file bytes used by execution."""

        self._require_no_resize()
        path = Path(source_image_path)
        source_image_sha256 = sha256_file(path)
        with Image.open(path) as source_image:
            return SpatialImageEncoding.from_source_image(
                plan=self,
                source_image=source_image,
                source_image_sha256=source_image_sha256,
            )

    def coordinate_receipt(
        self,
        coordinate_bins: Sequence[int],
    ) -> SpatialCoordinateReceipt:
        self._require_no_resize()
        bins = validate_bbox_bins(
            coordinate_bins,
            field="spatial_coordinate_receipt.coordinate_bins",
        )
        if self.variant_mode == "tile_reset":
            origin_x, origin_y = self.cell.halo_pixel_xyxy[:2]
            local_width = self.output_width
            local_height = self.output_height
        else:
            origin_x = origin_y = 0
            local_width = self.source_width
            local_height = self.source_height
        local_box = coord_bins_to_pixel_xyxy(
            bins,
            image_width=local_width,
            image_height=local_height,
            field="spatial_coordinate_receipt.coordinate_bins",
        )
        unclipped = (
            local_box[0] + origin_x,
            local_box[1] + origin_y,
            local_box[2] + origin_x,
            local_box[3] + origin_y,
        )
        clipped = (
            min(max(unclipped[0], 0), self.source_width),
            min(max(unclipped[1], 0), self.source_height),
            min(max(unclipped[2], 0), self.source_width),
            min(max(unclipped[3], 0), self.source_height),
        )
        return SpatialCoordinateReceipt(
            variant_mode=self.variant_mode,
            cell_index=self.cell.index,
            original_coordinate_bins=bins,
            local_extent_width=local_width,
            local_extent_height=local_height,
            local_integer_box=local_box,
            tile_origin_xy=(origin_x, origin_y),
            unclipped_global_integer_box=unclipped,
            clipped_global_integer_box=clipped,
            source_width=self.source_width,
            source_height=self.source_height,
        )

    def _require_no_resize(self) -> None:
        if self.do_resize is not False:
            raise EncodingContractError(
                "spatial variants require do_resize to be exactly false",
                code="analysis.spatial_resize_forbidden",
                context={
                    "do_resize": self.do_resize,
                    "value_type": type(self.do_resize).__name__,
                    "variant_mode": self.variant_mode,
                    "cell_index": self.cell.index,
                },
            )


@dataclass(frozen=True)
class SpatialImageEncoding:
    """Immutable RGB bytes and explicit no-resize processor expectations.

    ``total_merged_visual_tokens`` counts every merged token materialized by the
    processor. ``visible_merged_visual_tokens`` counts the exact natural-image
    support in the cell's core-plus-halo region. A native tile contains only
    that support, so its total and visible counts are equal. A full-canvas
    masked arm retains the full token grid while only the core-plus-halo token
    support remains visible.
    """

    plan: SpatialVariantPlan
    rgb_bytes: bytes
    width: int
    height: int
    channel_mean_rgb: tuple[int, int, int] | None
    expected_image_grid_thw: tuple[int, int, int]
    raw_patch_rows: int
    total_merged_visual_tokens: int
    visible_merged_visual_tokens: int
    source_image_sha256: str | None = None

    @property
    def fingerprint(self) -> str:
        """Canonical binding of source bytes, spatial plan, and rendered RGB bytes."""

        return sha256_payload(
            {
                "channel_mean_rgb": (
                    list(self.channel_mean_rgb)
                    if self.channel_mean_rgb is not None
                    else None
                ),
                "expected_image_grid_thw": list(self.expected_image_grid_thw),
                "height": self.height,
                "output_rgb_sha256": hashlib.sha256(self.rgb_bytes).hexdigest(),
                "plan": _spatial_plan_payload(self.plan),
                "raw_patch_rows": self.raw_patch_rows,
                "source_image_sha256": self.source_image_sha256,
                "total_merged_visual_tokens": self.total_merged_visual_tokens,
                "visible_merged_visual_tokens": self.visible_merged_visual_tokens,
                "width": self.width,
            }
        )

    @property
    def merged_visual_tokens(self) -> int:
        """Compatibility alias for the processor-materialized total."""

        return self.total_merged_visual_tokens

    @classmethod
    def from_source_image(
        cls,
        *,
        plan: SpatialVariantPlan,
        source_image: Image.Image,
        source_image_sha256: str | None = None,
    ) -> SpatialImageEncoding:
        plan._require_no_resize()
        if source_image_sha256 is not None:
            _require_sha256_digest(
                source_image_sha256,
                field="source_image_sha256",
            )
        if not isinstance(source_image, Image.Image):
            raise EncodingContractError(
                "spatial materialization requires a PIL image",
                code="analysis.spatial_image_type",
                context={"value_type": type(source_image).__name__},
            )
        if source_image.size != (plan.source_width, plan.source_height):
            raise EncodingContractError(
                "source image dimensions do not match the spatial plan",
                code="analysis.spatial_image_dimensions",
                context={
                    "observed_size": list(source_image.size),
                    "expected_size": [plan.source_width, plan.source_height],
                },
            )
        source_rgb = source_image.convert("RGB")
        try:
            if plan.variant_mode == "tile_reset":
                output = source_rgb.crop(plan.cell.halo_pixel_xyxy)
                channel_mean = None
            else:
                channel_mean = _float32_channel_mean_rgb(source_rgb)
                output = Image.new("RGB", source_rgb.size, color=channel_mean)
                halo = source_rgb.crop(plan.cell.halo_pixel_xyxy)
                try:
                    output.paste(halo, plan.cell.halo_pixel_xyxy[:2])
                finally:
                    halo.close()
            try:
                if output.size != (plan.output_width, plan.output_height):
                    raise EncodingContractError(
                        "spatial image output dimensions do not match the plan",
                        code="analysis.spatial_image_output_dimensions",
                        context={
                            "observed_size": list(output.size),
                            "expected_size": [plan.output_width, plan.output_height],
                        },
                    )
                rgb_bytes = output.tobytes()
            finally:
                output.close()
        finally:
            source_rgb.close()

        quantum = plan.grid_spec.visual_quantum_pixels
        if plan.output_width % quantum != 0 or plan.output_height % quantum != 0:
            raise EncodingContractError(
                "spatial output dimensions must preserve the merged-token quantum",
                code="analysis.spatial_output_quantum_divisibility",
                context={
                    "output_width": plan.output_width,
                    "output_height": plan.output_height,
                    "visual_quantum_pixels": quantum,
                },
            )
        patch_size = 16
        merge_size = quantum // patch_size
        image_grid = (
            1,
            plan.output_height // patch_size,
            plan.output_width // patch_size,
        )
        raw_patch_rows = math.prod(image_grid)
        total_merged_visual_tokens = raw_patch_rows // (merge_size * merge_size)
        visible_merged_visual_tokens = _box_area(plan.cell.halo_token_xyxy)
        if plan.variant_mode == "tile_reset":
            if total_merged_visual_tokens != visible_merged_visual_tokens:
                raise EncodingContractError(
                    "native tile total and visible merged-token counts must match",
                    code="analysis.spatial_tile_token_reconciliation",
                    context={
                        "total_merged_visual_tokens": total_merged_visual_tokens,
                        "visible_merged_visual_tokens": visible_merged_visual_tokens,
                        "cell_index": plan.cell.index,
                    },
                )
        elif visible_merged_visual_tokens > total_merged_visual_tokens:
            raise EncodingContractError(
                "masked visible support exceeds the full-canvas merged-token grid",
                code="analysis.spatial_mask_token_reconciliation",
                context={
                    "total_merged_visual_tokens": total_merged_visual_tokens,
                    "visible_merged_visual_tokens": visible_merged_visual_tokens,
                    "cell_index": plan.cell.index,
                },
            )
        return cls(
            plan=plan,
            rgb_bytes=rgb_bytes,
            width=plan.output_width,
            height=plan.output_height,
            channel_mean_rgb=channel_mean,
            expected_image_grid_thw=image_grid,
            raw_patch_rows=raw_patch_rows,
            total_merged_visual_tokens=total_merged_visual_tokens,
            visible_merged_visual_tokens=visible_merged_visual_tokens,
            source_image_sha256=source_image_sha256,
        )

    def to_pil_image(self) -> Image.Image:
        return Image.frombytes("RGB", (self.width, self.height), self.rgb_bytes)

    def verify_processor_output(
        self,
        *,
        pixel_values: Any,
        image_grid_thw: Any,
        executed_processor_config: Mapping[str, Any],
        executed_processor_identity: Mapping[str, Any],
        executed_processor_contract_sha256: str,
        expected_processor_contract_sha256: str,
    ) -> SpatialProcessorReceipt:
        self.plan._require_no_resize()
        (
            verified_processor_config,
            verified_processor_identity,
            verified_contract_sha256,
        ) = _verify_executed_processor_contract(
            executed_processor_config=executed_processor_config,
            executed_processor_identity=executed_processor_identity,
            executed_processor_contract_sha256=executed_processor_contract_sha256,
            expected_processor_contract_sha256=expected_processor_contract_sha256,
        )
        do_resize = verified_processor_config["do_resize"]
        patch_size = verified_processor_identity["patch_size"]
        merge_size = verified_processor_identity["merge_size"]
        temporal_patch_size = verified_processor_identity["temporal_patch_size"]
        for field, value in (
            ("patch_size", patch_size),
            ("merge_size", merge_size),
            ("temporal_patch_size", temporal_patch_size),
        ):
            _require_positive_integer(value, field=field)
        if patch_size * merge_size != self.plan.grid_spec.visual_quantum_pixels:
            raise EncodingContractError(
                "processor patch and merge sizes do not match the spatial visual quantum",
                code="analysis.spatial_processor_quantum_mismatch",
                context={
                    "patch_size": patch_size,
                    "merge_size": merge_size,
                    "visual_quantum_pixels": self.plan.grid_spec.visual_quantum_pixels,
                },
            )
        expected_grid = (1, self.height // patch_size, self.width // patch_size)
        observed_grid_shape = _shape_tuple(image_grid_thw)
        if observed_grid_shape != (1, 3):
            raise EncodingContractError(
                "spatial image_grid_thw must have shape [1, 3]",
                code="analysis.spatial_processor_grid_shape",
                context={"observed_shape": list(observed_grid_shape)},
            )
        observed_grid = _first_grid_row(image_grid_thw)
        if observed_grid != expected_grid:
            raise EncodingContractError(
                "processor image token grid does not match the spatial image dimensions",
                code="analysis.spatial_processor_grid_mismatch",
                context={
                    "observed_image_grid_thw": list(observed_grid),
                    "expected_image_grid_thw": list(expected_grid),
                },
            )
        expected_rows = math.prod(expected_grid)
        expected_width = 3 * temporal_patch_size * patch_size * patch_size
        observed_pixel_shape = _shape_tuple(pixel_values)
        if observed_pixel_shape != (expected_rows, expected_width):
            raise EncodingContractError(
                "processor pixel_values shape does not match no-resize spatial semantics",
                code="analysis.spatial_processor_pixel_shape",
                context={
                    "observed_shape": list(observed_pixel_shape),
                    "expected_shape": [expected_rows, expected_width],
                },
            )
        total_merged_visual_tokens = expected_rows // (merge_size * merge_size)
        if total_merged_visual_tokens != self.total_merged_visual_tokens:
            raise EncodingContractError(
                "processor merged visual-token count does not match spatial planning",
                code="analysis.spatial_processor_token_count",
                context={
                    "observed_total_merged_visual_tokens": total_merged_visual_tokens,
                    "expected_total_merged_visual_tokens": (
                        self.total_merged_visual_tokens
                    ),
                },
            )
        return SpatialProcessorReceipt.build(
            spatial_image_encoding_sha256=self.fingerprint,
            do_resize=do_resize,
            executed_processor_config=verified_processor_config,
            executed_processor_identity=verified_processor_identity,
            executed_processor_contract_sha256=verified_contract_sha256,
            expected_processor_contract_sha256=expected_processor_contract_sha256,
            image_grid_thw=observed_grid,
            raw_patch_rows=expected_rows,
            total_merged_visual_tokens=total_merged_visual_tokens,
            visible_merged_visual_tokens=self.visible_merged_visual_tokens,
            pixel_values=pixel_values,
            image_grid_thw_tensor=image_grid_thw,
        )


@dataclass(frozen=True)
class SpatialProcessorReceipt:
    """Immutable executed processor evidence bound to one spatial encoding."""

    spatial_image_encoding_sha256: str
    do_resize: bool
    executed_processor_config: Mapping[str, Any]
    executed_processor_identity: Mapping[str, Any]
    executed_processor_contract_sha256: str
    expected_processor_contract_sha256: str
    image_grid_thw: tuple[int, int, int]
    raw_patch_rows: int
    total_merged_visual_tokens: int
    visible_merged_visual_tokens: int
    pixel_values_shape: tuple[int, int]
    executed_visual_tensors: ExecutedVisualTensorReceipt
    receipt_sha256: str

    @classmethod
    def build(
        cls,
        *,
        spatial_image_encoding_sha256: str,
        do_resize: bool,
        executed_processor_config: Mapping[str, Any],
        executed_processor_identity: Mapping[str, Any],
        executed_processor_contract_sha256: str,
        expected_processor_contract_sha256: str,
        image_grid_thw: tuple[int, int, int],
        raw_patch_rows: int,
        total_merged_visual_tokens: int,
        visible_merged_visual_tokens: int,
        pixel_values: Any,
        image_grid_thw_tensor: Any,
    ) -> SpatialProcessorReceipt:
        executed_visual_tensors = ExecutedVisualTensorReceipt.from_tensors(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw_tensor,
        )
        pixel_values_shape = tuple(executed_visual_tensors.pixel_values.shape)
        payload = {
            "do_resize": do_resize,
            "executed_processor_config": dict(executed_processor_config),
            "executed_processor_contract_sha256": (executed_processor_contract_sha256),
            "executed_processor_identity": dict(executed_processor_identity),
            "expected_processor_contract_sha256": (expected_processor_contract_sha256),
            "image_grid_thw": list(image_grid_thw),
            "pixel_values_shape": list(pixel_values_shape),
            "raw_patch_rows": raw_patch_rows,
            "spatial_image_encoding_sha256": spatial_image_encoding_sha256,
            "total_merged_visual_tokens": total_merged_visual_tokens,
            "visible_merged_visual_tokens": visible_merged_visual_tokens,
            "executed_visual_tensors": executed_visual_tensors.to_artifact_dict(),
        }
        return cls(
            spatial_image_encoding_sha256=spatial_image_encoding_sha256,
            do_resize=do_resize,
            executed_processor_config=dict(executed_processor_config),
            executed_processor_identity=dict(executed_processor_identity),
            executed_processor_contract_sha256=executed_processor_contract_sha256,
            expected_processor_contract_sha256=expected_processor_contract_sha256,
            image_grid_thw=image_grid_thw,
            raw_patch_rows=raw_patch_rows,
            total_merged_visual_tokens=total_merged_visual_tokens,
            visible_merged_visual_tokens=visible_merged_visual_tokens,
            pixel_values_shape=pixel_values_shape,
            executed_visual_tensors=executed_visual_tensors,
            receipt_sha256=sha256_payload(payload),
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "executed_processor_config",
            MappingProxyType(dict(self.executed_processor_config)),
        )
        object.__setattr__(
            self,
            "executed_processor_identity",
            MappingProxyType(dict(self.executed_processor_identity)),
        )
        for field in (
            "spatial_image_encoding_sha256",
            "executed_processor_contract_sha256",
            "expected_processor_contract_sha256",
            "receipt_sha256",
        ):
            _require_sha256_digest(getattr(self, field), field=field)
        if self.do_resize is not False:
            raise EncodingContractError(
                "spatial processor receipt requires do_resize false",
                code="analysis.spatial_processor_receipt_resize",
            )
        if self.receipt_sha256 != sha256_payload(self.identity_payload()):
            raise EncodingContractError(
                "spatial processor receipt digest does not bind its evidence",
                code="analysis.spatial_processor_receipt_digest_mismatch",
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "do_resize": self.do_resize,
            "executed_processor_config": dict(self.executed_processor_config),
            "executed_processor_contract_sha256": (
                self.executed_processor_contract_sha256
            ),
            "executed_processor_identity": dict(self.executed_processor_identity),
            "expected_processor_contract_sha256": (
                self.expected_processor_contract_sha256
            ),
            "image_grid_thw": list(self.image_grid_thw),
            "pixel_values_shape": list(self.pixel_values_shape),
            "raw_patch_rows": self.raw_patch_rows,
            "spatial_image_encoding_sha256": self.spatial_image_encoding_sha256,
            "total_merged_visual_tokens": self.total_merged_visual_tokens,
            "visible_merged_visual_tokens": self.visible_merged_visual_tokens,
            "executed_visual_tensors": (
                self.executed_visual_tensors.to_artifact_dict()
            ),
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class TensorContentReceipt:
    """Device-independent exact-value identity for one executed dense tensor."""

    dtype: str
    shape: tuple[int, ...]
    canonical_content_sha256: str
    schema_version: str = TENSOR_CONTENT_RECEIPT_SCHEMA_VERSION

    @classmethod
    def from_tensor(cls, value: Any, *, field: str) -> TensorContentReceipt:
        if not isinstance(value, torch.Tensor):
            raise EncodingContractError(
                "executed tensor evidence requires a torch.Tensor",
                code="analysis.spatial_tensor_type",
                context={"field": field, "observed_type": type(value).__name__},
            )
        if value.layout != torch.strided:
            raise EncodingContractError(
                "executed tensor evidence requires a dense strided tensor",
                code="analysis.spatial_tensor_layout",
                context={"field": field, "layout": str(value.layout)},
            )
        materialized = value.detach().to(device="cpu").contiguous()
        dtype = str(materialized.dtype).removeprefix("torch.")
        shape = tuple(int(size) for size in materialized.shape)
        raw_bytes = materialized.view(torch.uint8).numpy().tobytes(order="C")
        content_sha256 = hashlib.sha256(
            b"coordexp-canonical-tensor-bytes-v1\x00"
            + dtype.encode("ascii")
            + b"\x00"
            + canonical_tensor_shape_bytes(shape)
            + b"\x00"
            + raw_bytes
        ).hexdigest()
        return cls(
            dtype=dtype,
            shape=shape,
            canonical_content_sha256=content_sha256,
        )

    def __post_init__(self) -> None:
        if self.schema_version != TENSOR_CONTENT_RECEIPT_SCHEMA_VERSION:
            raise EncodingContractError(
                "tensor content receipt schema is unsupported",
                code="analysis.spatial_tensor_schema",
            )
        if not self.dtype or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.shape
        ):
            raise EncodingContractError(
                "tensor content receipt dtype and shape must be canonical",
                code="analysis.spatial_tensor_metadata",
            )
        _require_sha256_digest(
            self.canonical_content_sha256,
            field="canonical_content_sha256",
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "canonical_content_sha256": self.canonical_content_sha256,
            "dtype": self.dtype,
            "schema_version": self.schema_version,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class ExecutedVisualTensorReceipt:
    """Exact executed pixel and image-grid tensors for one model request."""

    pixel_values: TensorContentReceipt
    image_grid_thw: TensorContentReceipt
    receipt_sha256: str
    schema_version: str = EXECUTED_VISUAL_TENSOR_RECEIPT_SCHEMA_VERSION

    @classmethod
    def from_tensors(
        cls,
        *,
        pixel_values: Any,
        image_grid_thw: Any,
    ) -> ExecutedVisualTensorReceipt:
        pixel_receipt = TensorContentReceipt.from_tensor(
            pixel_values,
            field="pixel_values",
        )
        grid_receipt = TensorContentReceipt.from_tensor(
            image_grid_thw,
            field="image_grid_thw",
        )
        payload = {
            "image_grid_thw": grid_receipt.to_artifact_dict(),
            "pixel_values": pixel_receipt.to_artifact_dict(),
            "schema_version": EXECUTED_VISUAL_TENSOR_RECEIPT_SCHEMA_VERSION,
        }
        return cls(
            pixel_values=pixel_receipt,
            image_grid_thw=grid_receipt,
            receipt_sha256=sha256_payload(payload),
        )

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTED_VISUAL_TENSOR_RECEIPT_SCHEMA_VERSION:
            raise EncodingContractError(
                "executed visual tensor receipt schema is unsupported",
                code="analysis.spatial_visual_tensor_schema",
            )
        _require_sha256_digest(self.receipt_sha256, field="receipt_sha256")
        if self.receipt_sha256 != sha256_payload(self.identity_payload()):
            raise EncodingContractError(
                "executed visual tensor receipt digest does not bind its tensors",
                code="analysis.spatial_visual_tensor_digest_mismatch",
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "image_grid_thw": self.image_grid_thw.to_artifact_dict(),
            "pixel_values": self.pixel_values.to_artifact_dict(),
            "schema_version": self.schema_version,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class VisualInputMaterializationReceipt:
    """Sealed source/RGB/processor/tensor chain for one model visual input."""

    input_kind: Literal["full_image", "spatial_variant"]
    source_image_sha256: str
    input_rgb_sha256: str
    input_width: int
    input_height: int
    spatial_image_encoding_sha256: str | None
    processor_contract_sha256: str
    executed_visual_tensors: ExecutedVisualTensorReceipt
    receipt_sha256: str
    _mint_capability: InitVar[object] = None
    schema_version: str = VISUAL_INPUT_MATERIALIZATION_RECEIPT_SCHEMA_VERSION

    @classmethod
    def _mint(
        cls,
        *,
        input_kind: Literal["full_image", "spatial_variant"],
        source_image_sha256: str,
        input_rgb_bytes: bytes,
        input_width: int,
        input_height: int,
        spatial_image_encoding_sha256: str | None,
        processor_contract_sha256: str,
        pixel_values: Any,
        image_grid_thw: Any,
        _mint_capability: object,
    ) -> VisualInputMaterializationReceipt:
        if _mint_capability is not _VISUAL_MATERIALIZATION_MINT_CAPABILITY:
            raise EncodingContractError(
                "visual-input receipts can only be minted by processor execution",
                code="analysis.visual_materialization_mint_authority",
            )
        tensors = ExecutedVisualTensorReceipt.from_tensors(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        identity = {
            "executed_visual_tensors": tensors.to_artifact_dict(),
            "input_height": input_height,
            "input_kind": input_kind,
            "input_rgb_sha256": hashlib.sha256(input_rgb_bytes).hexdigest(),
            "input_width": input_width,
            "processor_contract_sha256": processor_contract_sha256,
            "schema_version": VISUAL_INPUT_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
            "source_image_sha256": source_image_sha256,
            "spatial_image_encoding_sha256": spatial_image_encoding_sha256,
        }
        return cls(
            input_kind=input_kind,
            source_image_sha256=source_image_sha256,
            input_rgb_sha256=identity["input_rgb_sha256"],
            input_width=input_width,
            input_height=input_height,
            spatial_image_encoding_sha256=spatial_image_encoding_sha256,
            processor_contract_sha256=processor_contract_sha256,
            executed_visual_tensors=tensors,
            receipt_sha256=sha256_payload(identity),
            _mint_capability=_VISUAL_MATERIALIZATION_MINT_CAPABILITY,
        )

    def __post_init__(self, _mint_capability: object) -> None:
        if self.schema_version != VISUAL_INPUT_MATERIALIZATION_RECEIPT_SCHEMA_VERSION:
            raise EncodingContractError(
                "visual-input materialization receipt schema is unsupported",
                code="analysis.visual_materialization_schema",
            )
        if _mint_capability is not _VISUAL_MATERIALIZATION_MINT_CAPABILITY:
            raise EncodingContractError(
                "visual-input receipt lacks processor-owned mint authority",
                code="analysis.visual_materialization_mint_authority",
            )
        if self.input_kind not in {"full_image", "spatial_variant"}:
            raise EncodingContractError(
                "visual-input materialization kind is unsupported",
                code="analysis.visual_materialization_kind",
            )
        for field in (
            "source_image_sha256",
            "input_rgb_sha256",
            "processor_contract_sha256",
            "receipt_sha256",
        ):
            _require_sha256_digest(getattr(self, field), field=field)
        _require_positive_integer(self.input_width, field="input_width")
        _require_positive_integer(self.input_height, field="input_height")
        if self.input_kind == "spatial_variant":
            if self.spatial_image_encoding_sha256 is None:
                raise EncodingContractError(
                    "spatial materialization requires its encoding identity",
                    code="analysis.visual_materialization_spatial_encoding",
                )
            _require_sha256_digest(
                self.spatial_image_encoding_sha256,
                field="spatial_image_encoding_sha256",
            )
        elif self.spatial_image_encoding_sha256 is not None:
            raise EncodingContractError(
                "full-image materialization cannot name a spatial encoding",
                code="analysis.visual_materialization_full_encoding",
            )
        if self.receipt_sha256 != sha256_payload(self.identity_payload()):
            raise EncodingContractError(
                "visual-input materialization digest does not bind its evidence",
                code="analysis.visual_materialization_digest",
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "executed_visual_tensors": self.executed_visual_tensors.to_artifact_dict(),
            "input_height": self.input_height,
            "input_kind": self.input_kind,
            "input_rgb_sha256": self.input_rgb_sha256,
            "input_width": self.input_width,
            "processor_contract_sha256": self.processor_contract_sha256,
            "schema_version": self.schema_version,
            "source_image_sha256": self.source_image_sha256,
            "spatial_image_encoding_sha256": self.spatial_image_encoding_sha256,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class MaterializedVisualInput:
    """The only accepted owner of processor-produced visual model tensors.

    The owner loads or renders the exact RGB input, invokes the processor itself,
    and seals the returned tensors. Callers cannot attach independently prepared
    tensors to a source receipt.
    """

    receipt: VisualInputMaterializationReceipt
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    _mint_capability: InitVar[object] = None
    spatial_image_encoding: SpatialImageEncoding | None = None

    def __post_init__(self, _mint_capability: object) -> None:
        if _mint_capability is not _VISUAL_MATERIALIZATION_MINT_CAPABILITY:
            raise EncodingContractError(
                "materialized visual input lacks processor-owned mint authority",
                code="analysis.visual_materialization_mint_authority",
            )
        self.verify_current_tensors()

    @classmethod
    def from_full_image_path(
        cls,
        *,
        source_image_path: str | Path,
        expected_source_image_sha256: str,
        image_processor: Callable[..., Mapping[str, Any]],
        processor_contract_sha256: str,
    ) -> MaterializedVisualInput:
        path = Path(source_image_path)
        source_digest = sha256_file(path)
        if source_digest != expected_source_image_sha256:
            raise EncodingContractError(
                "full-image source bytes differ from the expected source",
                code="analysis.visual_materialization_source_mismatch",
            )
        with Image.open(path) as source:
            rgb = source.convert("RGB")
        try:
            return cls._invoke_processor(
                input_kind="full_image",
                source_image_sha256=source_digest,
                rgb_image=rgb,
                image_processor=image_processor,
                processor_contract_sha256=processor_contract_sha256,
                spatial_image_encoding=None,
            )
        finally:
            rgb.close()

    @classmethod
    def from_spatial_plan_path(
        cls,
        *,
        plan: SpatialVariantPlan,
        source_image_path: str | Path,
        expected_source_image_sha256: str,
        image_processor: Callable[..., Mapping[str, Any]],
        processor_contract_sha256: str,
    ) -> MaterializedVisualInput:
        encoding = plan.materialize_path(source_image_path)
        if encoding.source_image_sha256 != expected_source_image_sha256:
            raise EncodingContractError(
                "spatial source bytes differ from the expected source",
                code="analysis.visual_materialization_source_mismatch",
            )
        rgb = encoding.to_pil_image()
        try:
            return cls._invoke_processor(
                input_kind="spatial_variant",
                source_image_sha256=expected_source_image_sha256,
                rgb_image=rgb,
                image_processor=image_processor,
                processor_contract_sha256=processor_contract_sha256,
                spatial_image_encoding=encoding,
            )
        finally:
            rgb.close()

    @classmethod
    def _invoke_processor(
        cls,
        *,
        input_kind: Literal["full_image", "spatial_variant"],
        source_image_sha256: str,
        rgb_image: Image.Image,
        image_processor: Callable[..., Mapping[str, Any]],
        processor_contract_sha256: str,
        spatial_image_encoding: SpatialImageEncoding | None,
    ) -> MaterializedVisualInput:
        _require_sha256_digest(
            processor_contract_sha256,
            field="processor_contract_sha256",
        )
        rgb_bytes = rgb_image.tobytes()
        encoded = image_processor(
            images=[rgb_image],
            return_tensors="pt",
            do_resize=False,
        )
        if not isinstance(encoded, Mapping):
            raise EncodingContractError(
                "image processor must return a mapping",
                code="analysis.visual_materialization_processor_output",
            )
        try:
            pixel_values = encoded["pixel_values"]
            image_grid_thw = encoded["image_grid_thw"]
        except KeyError as exc:
            raise EncodingContractError(
                "image processor omitted required visual tensors",
                code="analysis.visual_materialization_processor_output",
                context={"missing_field": str(exc)},
                cause=exc,
            ) from exc
        if not isinstance(pixel_values, torch.Tensor) or not isinstance(
            image_grid_thw, torch.Tensor
        ):
            raise EncodingContractError(
                "image processor returned non-tensor visual inputs",
                code="analysis.visual_materialization_processor_output",
            )
        receipt = VisualInputMaterializationReceipt._mint(
            input_kind=input_kind,
            source_image_sha256=source_image_sha256,
            input_rgb_bytes=rgb_bytes,
            input_width=rgb_image.width,
            input_height=rgb_image.height,
            spatial_image_encoding_sha256=(
                spatial_image_encoding.fingerprint
                if spatial_image_encoding is not None
                else None
            ),
            processor_contract_sha256=processor_contract_sha256,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            _mint_capability=_VISUAL_MATERIALIZATION_MINT_CAPABILITY,
        )
        materialized = cls(
            receipt=receipt,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            spatial_image_encoding=spatial_image_encoding,
            _mint_capability=_VISUAL_MATERIALIZATION_MINT_CAPABILITY,
        )
        materialized.verify_current_tensors()
        return materialized

    def verify_current_tensors(self) -> None:
        observed = ExecutedVisualTensorReceipt.from_tensors(
            pixel_values=self.pixel_values,
            image_grid_thw=self.image_grid_thw,
        )
        if observed.receipt_sha256 != self.receipt.executed_visual_tensors.receipt_sha256:
            raise EncodingContractError(
                "materialized visual tensors changed after processor execution",
                code="analysis.visual_materialization_tensor_mutation",
            )

    def verify_model_inputs(self, model_inputs: Mapping[str, Any]) -> None:
        self.verify_current_tensors()
        try:
            observed = ExecutedVisualTensorReceipt.from_tensors(
                pixel_values=model_inputs["pixel_values"],
                image_grid_thw=model_inputs["image_grid_thw"],
            )
        except KeyError as exc:
            raise EncodingContractError(
                "decode request omitted materialized visual tensors",
                code="analysis.visual_materialization_model_inputs",
                context={"missing_field": str(exc)},
                cause=exc,
            ) from exc
        if observed.receipt_sha256 != self.receipt.executed_visual_tensors.receipt_sha256:
            raise EncodingContractError(
                "decode request visual tensors differ from processor-owned materialization",
                code="analysis.visual_materialization_model_input_mismatch",
            )


def canonical_tensor_shape_bytes(shape: tuple[int, ...]) -> bytes:
    """Serialize tensor rank and dimensions without device or Python repr drift."""

    return b",".join(str(size).encode("ascii") for size in shape)


def _spatial_plan_payload(plan: SpatialVariantPlan) -> dict[str, Any]:
    return {
        "cell": {
            "column_index": plan.cell.column_index,
            "core_pixel_xyxy": list(plan.cell.core_pixel_xyxy),
            "core_token_xyxy": list(plan.cell.core_token_xyxy),
            "halo_pixel_xyxy": list(plan.cell.halo_pixel_xyxy),
            "halo_token_xyxy": list(plan.cell.halo_token_xyxy),
            "index": plan.cell.index,
            "row_index": plan.cell.row_index,
        },
        "do_resize": plan.do_resize,
        "grid_spec": plan.grid_spec.to_artifact_dict(),
        "output_height": plan.output_height,
        "output_width": plan.output_width,
        "source_height": plan.source_height,
        "source_width": plan.source_width,
        "variant_mode": plan.variant_mode,
    }


def _verify_executed_processor_contract(
    *,
    executed_processor_config: Mapping[str, Any],
    executed_processor_identity: Mapping[str, Any],
    executed_processor_contract_sha256: str,
    expected_processor_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    processor_config = _copy_string_key_mapping(
        executed_processor_config,
        field="executed_processor_config",
    )
    processor_identity = _copy_string_key_mapping(
        executed_processor_identity,
        field="executed_processor_identity",
    )
    if "do_resize" not in processor_config:
        raise EncodingContractError(
            "executed processor config must attest do_resize",
            code="analysis.spatial_processor_resize_missing",
        )
    if processor_config["do_resize"] is not False:
        raise EncodingContractError(
            "executed spatial processor do_resize must be exactly false",
            code="analysis.spatial_processor_resize_executed",
            context={
                "do_resize": processor_config["do_resize"],
                "value_type": type(processor_config["do_resize"]).__name__,
            },
        )
    required_identity_fields = (
        "patch_size",
        "merge_size",
        "temporal_patch_size",
    )
    missing_identity_fields = [
        field for field in required_identity_fields if field not in processor_identity
    ]
    if missing_identity_fields:
        raise EncodingContractError(
            "executed processor identity is incomplete",
            code="analysis.spatial_processor_identity_missing",
            context={"missing_fields": missing_identity_fields},
        )
    _require_sha256_digest(
        executed_processor_contract_sha256,
        field="executed_processor_contract_sha256",
    )
    _require_sha256_digest(
        expected_processor_contract_sha256,
        field="expected_processor_contract_sha256",
    )
    contract = {
        "processor_config": processor_config,
        "processor_identity": processor_identity,
    }
    try:
        calculated_contract_sha256 = sha256_json(contract)
    except (ConfigContractError, TypeError) as exc:
        raise EncodingContractError(
            "executed processor contract must have a stable JSON fingerprint",
            code="analysis.spatial_processor_contract_unfingerprintable",
            cause=exc,
        ) from exc
    if executed_processor_contract_sha256 != calculated_contract_sha256:
        raise EncodingContractError(
            "executed processor contract digest does not bind the supplied evidence",
            code="analysis.spatial_processor_contract_digest_mismatch",
            context={
                "executed_processor_contract_sha256": (
                    executed_processor_contract_sha256
                ),
                "calculated_processor_contract_sha256": calculated_contract_sha256,
            },
        )
    if executed_processor_contract_sha256 != expected_processor_contract_sha256:
        raise EncodingContractError(
            "executed processor contract drifted from the expected identity",
            code="analysis.spatial_processor_contract_drift",
            context={
                "executed_processor_contract_sha256": (
                    executed_processor_contract_sha256
                ),
                "expected_processor_contract_sha256": (
                    expected_processor_contract_sha256
                ),
            },
        )
    return processor_config, processor_identity, calculated_contract_sha256


def _copy_string_key_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EncodingContractError(
            "executed processor evidence must be a mapping",
            code="analysis.spatial_processor_contract_mapping",
            context={"field": field, "value_type": type(value).__name__},
        )
    if any(not isinstance(key, str) for key in value):
        raise EncodingContractError(
            "executed processor evidence keys must be strings",
            code="analysis.spatial_processor_contract_key",
            context={"field": field},
        )
    return dict(value)


def _require_sha256_digest(value: Any, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EncodingContractError(
            "processor contract digest must be a lowercase SHA-256 hexadecimal string",
            code="analysis.spatial_processor_contract_digest_format",
            context={"field": field, "value": value},
        )


def _build_cell(
    *,
    spec: SpatialGridSpec,
    row_index: int,
    column_index: int,
    token_width: int,
    token_height: int,
) -> SpatialCell:
    core_left = column_index * token_width // spec.column_count
    core_right = (column_index + 1) * token_width // spec.column_count
    core_top = row_index * token_height // spec.row_count
    core_bottom = (row_index + 1) * token_height // spec.row_count
    core_width = core_right - core_left
    core_height = core_bottom - core_top
    halo_x = max(
        1,
        _ceiling_fraction(
            core_width,
            numerator=spec.halo_fraction_numerator,
            denominator=spec.halo_fraction_denominator,
        ),
    )
    halo_y = max(
        1,
        _ceiling_fraction(
            core_height,
            numerator=spec.halo_fraction_numerator,
            denominator=spec.halo_fraction_denominator,
        ),
    )
    core_token_box = (core_left, core_top, core_right, core_bottom)
    halo_token_box = (
        max(0, core_left - halo_x),
        max(0, core_top - halo_y),
        min(token_width, core_right + halo_x),
        min(token_height, core_bottom + halo_y),
    )
    quantum = spec.visual_quantum_pixels
    return SpatialCell(
        index=row_index * spec.column_count + column_index,
        row_index=row_index,
        column_index=column_index,
        core_token_xyxy=core_token_box,
        halo_token_xyxy=halo_token_box,
        core_pixel_xyxy=tuple(value * quantum for value in core_token_box),
        halo_pixel_xyxy=tuple(value * quantum for value in halo_token_box),
    )


def _box_area(box: Sequence[int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _float32_channel_mean_rgb(image: Image.Image) -> tuple[int, int, int]:
    values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    values = values.reshape(image.height, image.width, 3).to(dtype=torch.float32)
    rounded = torch.round(values.mean(dim=(0, 1))).clamp(0, 255).to(dtype=torch.uint8)
    return tuple(int(value) for value in rounded.tolist())


def _ceiling_fraction(value: int, *, numerator: int, denominator: int) -> int:
    return (value * numerator + denominator - 1) // denominator


def _finite_box(value: Sequence[Any]) -> tuple[float, float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        return None
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        parsed.append(number)
    return parsed[0], parsed[1], parsed[2], parsed[3]


def _first_grid_row(value: Any) -> tuple[int, int, int]:
    row = value[0]
    if hasattr(row, "detach"):
        row = row.detach().cpu()
    if hasattr(row, "tolist"):
        row = row.tolist()
    return tuple(int(item) for item in row)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise EncodingContractError(
            "processor output does not expose a shape",
            code="analysis.spatial_processor_shape_missing",
            context={"value_type": type(value).__name__},
        )
    return tuple(int(item) for item in shape)


def _require_positive_integer(value: Any, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataContractError(
            "spatial contract values must be positive integers",
            code="analysis.spatial_positive_integer",
            context={"field": field, "value": value},
        )


def _validate_positive_area_box(value: Sequence[int], *, field: str) -> None:
    if len(value) != 4 or value[0] >= value[2] or value[1] >= value[3]:
        raise DataContractError(
            "spatial box must have positive area",
            code="analysis.spatial_box_area",
            context={"field": field, "value": list(value)},
        )


__all__ = [
    "SpatialCell",
    "SpatialCoordinateReceipt",
    "SpatialGrid",
    "SpatialGridSpec",
    "SpatialImageEncoding",
    "SpatialOwnership",
    "SpatialVariantPlan",
]

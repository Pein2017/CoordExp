"""Pure, replayable discrete crop/letterbox transform for ROI inference."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from PIL import Image

from src.common.errors import DataContractError
from src.label_studio_coco_refinement.geometry import pixel_xyxy_to_norm1000


ROI_TRANSFORM_ID = "coordexp-roi-letterbox-half-up-v1"
CROP_EDGE_CONVENTION = "half-open-floor-start-ceil-end"
PIXEL_EDGE_CONVENTION = "continuous-pixel-edges"
RESAMPLER = "Pillow.Resampling.BICUBIC"
PAD_VALUE_RGB = (0, 0, 0)


class RoiTransformError(ValueError):
    """A transform input or inverse-mapped result violates the V1 contract."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class LabelStudioRoi:
    """Label Studio percentage rectangle (`x`, `y`, `width`, `height`)."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise RoiTransformError(
                "ROI percentage values must be finite", code="roi_non_finite"
            )
        normalized = tuple(float(value) for value in values)
        if normalized[2] <= 0 or normalized[3] <= 0:
            raise RoiTransformError(
                "ROI percentage width and height must be positive",
                code="roi_non_positive",
            )
        for field, value in zip(("x", "y", "width", "height"), normalized, strict=True):
            object.__setattr__(self, field, value)

    @property
    def edges(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class InverseMapping:
    """All coordinate layers for one accepted inverse-mapped parser box."""

    canvas_bbox: tuple[float, float, float, float]
    canvas_content_intersection: tuple[float, float, float, float]
    unpadded_bbox: tuple[float, float, float, float]
    crop_bbox: tuple[float, float, float, float]
    source_bbox: tuple[float, float, float, float]
    norm1000_bbox: tuple[int, int, int, int]

    def to_receipt_dict(self) -> dict[str, Any]:
        return {
            "canvas_bbox": list(self.canvas_bbox),
            "canvas_content_intersection": list(self.canvas_content_intersection),
            "unpadded_bbox": list(self.unpadded_bbox),
            "crop_bbox": list(self.crop_bbox),
            "source_bbox": list(self.source_bbox),
            "norm1000_bbox": list(self.norm1000_bbox),
        }


@dataclass(frozen=True)
class RoiLetterboxTransform:
    """One immutable transform owns both pixel preparation and box inversion."""

    source_width: int
    source_height: int
    roi_percent: LabelStudioRoi
    roi_float_edges: tuple[float, float, float, float]
    clipped_float_edges: tuple[float, float, float, float]
    crop_edges: tuple[int, int, int, int]
    canvas_width: int
    canvas_height: int
    realized_width: int
    realized_height: int
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    transform_id: str = ROI_TRANSFORM_ID
    processor_do_resize: bool = False

    @classmethod
    def from_label_studio_roi(
        cls,
        *,
        source_width: int,
        source_height: int,
        roi: LabelStudioRoi | Sequence[float],
        canvas_width: int,
        canvas_height: int,
    ) -> "RoiLetterboxTransform":
        source_width = _positive_int(source_width, field="source_width")
        source_height = _positive_int(source_height, field="source_height")
        canvas_width = _positive_int(canvas_width, field="canvas_width")
        canvas_height = _positive_int(canvas_height, field="canvas_height")
        if not isinstance(roi, LabelStudioRoi):
            roi = _coerce_label_studio_roi(roi)

        x1, y1, x2, y2 = roi.edges
        float_edges = (
            x1 * source_width / 100.0,
            y1 * source_height / 100.0,
            x2 * source_width / 100.0,
            y2 * source_height / 100.0,
        )
        clipped = (
            _clip(float_edges[0], 0.0, float(source_width)),
            _clip(float_edges[1], 0.0, float(source_height)),
            _clip(float_edges[2], 0.0, float(source_width)),
            _clip(float_edges[3], 0.0, float(source_height)),
        )
        crop_edges = (
            math.floor(clipped[0]),
            math.floor(clipped[1]),
            math.ceil(clipped[2]),
            math.ceil(clipped[3]),
        )
        crop_width = crop_edges[2] - crop_edges[0]
        crop_height = crop_edges[3] - crop_edges[1]
        if crop_width <= 0 or crop_height <= 0:
            raise RoiTransformError(
                "ROI is degenerate after clipping", code="roi_degenerate_after_clipping"
            )

        fit = min(
            Fraction(canvas_width, crop_width), Fraction(canvas_height, crop_height)
        )
        realized_width = min(canvas_width, max(1, _round_half_up(fit * crop_width)))
        realized_height = min(canvas_height, max(1, _round_half_up(fit * crop_height)))
        scale_x = realized_width / crop_width
        scale_y = realized_height / crop_height
        horizontal_pad = canvas_width - realized_width
        vertical_pad = canvas_height - realized_height
        pad_left = horizontal_pad // 2
        pad_top = vertical_pad // 2
        pad_right = horizontal_pad - pad_left
        pad_bottom = vertical_pad - pad_top

        return cls(
            source_width=source_width,
            source_height=source_height,
            roi_percent=roi,
            roi_float_edges=float_edges,
            clipped_float_edges=clipped,
            crop_edges=crop_edges,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            realized_width=realized_width,
            realized_height=realized_height,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
        )

    @classmethod
    def from_receipt_dict(cls, receipt: Mapping[str, Any]) -> "RoiLetterboxTransform":
        """Replay and strictly attest a complete serialized transform receipt."""

        if not isinstance(receipt, Mapping):
            raise RoiTransformError(
                "transform receipt must be a mapping", code="receipt_invalid"
            )
        try:
            source_size = _receipt_sequence(
                receipt.get("source_size"), length=2, field="source_size"
            )
            canvas_size = _receipt_sequence(
                receipt.get("canvas_size"), length=2, field="canvas_size"
            )
            roi_percent = _receipt_sequence(
                receipt.get("roi_percent_xywh"),
                length=4,
                field="roi_percent_xywh",
            )
            transform = cls.from_label_studio_roi(
                source_width=source_size[0],
                source_height=source_size[1],
                roi=roi_percent,
                canvas_width=canvas_size[0],
                canvas_height=canvas_size[1],
            )
            supplied = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
            replayed = json.dumps(
                transform.to_receipt_dict(), sort_keys=True, separators=(",", ":")
            )
        except (KeyError, TypeError, ValueError, RoiTransformError) as exc:
            raise RoiTransformError(
                "transform receipt cannot replay the immutable transform",
                code="receipt_invalid",
            ) from exc
        if supplied != replayed:
            raise RoiTransformError(
                "transform receipt does not match the replayed transform",
                code="receipt_mismatch",
            )
        return transform

    @property
    def crop_width(self) -> int:
        return self.crop_edges[2] - self.crop_edges[0]

    @property
    def crop_height(self) -> int:
        return self.crop_edges[3] - self.crop_edges[1]

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.canvas_width, self.canvas_height

    @property
    def content_canvas_edges(self) -> tuple[int, int, int, int]:
        return (
            self.pad_left,
            self.pad_top,
            self.pad_left + self.realized_width,
            self.pad_top + self.realized_height,
        )

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_receipt_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def prepare_image(self, image: Image.Image) -> Image.Image:
        """Crop once, resize once, and return the exact no-resize processor canvas."""

        if not isinstance(image, Image.Image):
            raise RoiTransformError("image must be a Pillow image", code="image_type")
        if image.size != (self.source_width, self.source_height):
            raise RoiTransformError(
                "decoded image dimensions do not match transform identity",
                code="image_size_mismatch",
            )
        crop = image.crop(self.crop_edges).convert("RGB")
        if crop.size != (self.realized_width, self.realized_height):
            crop = crop.resize(
                (self.realized_width, self.realized_height),
                resample=Image.Resampling.BICUBIC,
            )
        canvas = Image.new("RGB", self.canvas_size, PAD_VALUE_RGB)
        canvas.paste(crop, (self.pad_left, self.pad_top))
        return canvas

    def assert_no_resize_processor_canvas(
        self,
        *,
        do_resize: bool,
        observed_width: int,
        observed_height: int,
    ) -> None:
        """Fail closed if a processor silently changes the prepared canvas."""

        if do_resize is not False:
            raise RoiTransformError(
                "ROI processor execution must set do_resize=false",
                code="processor_resize_enabled",
            )
        if (observed_width, observed_height) != self.canvas_size:
            raise RoiTransformError(
                "processor canvas does not match the prepared canvas",
                code="processor_canvas_mismatch",
            )

    def expected_grid_thw(self, *, patch_size: int) -> tuple[int, int, int]:
        patch_size = _positive_int(patch_size, field="patch_size")
        if self.canvas_width % patch_size or self.canvas_height % patch_size:
            raise RoiTransformError(
                "canvas is not divisible by processor patch size",
                code="processor_grid_divisibility",
            )
        return (1, self.canvas_height // patch_size, self.canvas_width // patch_size)

    def inverse_canvas_bbox(self, bbox: Sequence[float]) -> InverseMapping:
        """Invert one parser-valid canvas bbox using the recorded discrete transform."""

        canvas_bbox = _strict_bbox(bbox, field="canvas_bbox")
        content = tuple(float(value) for value in self.content_canvas_edges)
        intersection = (
            max(canvas_bbox[0], content[0]),
            max(canvas_bbox[1], content[1]),
            min(canvas_bbox[2], content[2]),
            min(canvas_bbox[3], content[3]),
        )
        if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
            raise RoiTransformError(
                "canvas bbox has no positive-area intersection with ROI content",
                code="mapped_entirely_in_padding",
            )

        unpadded = (
            intersection[0] - self.pad_left,
            intersection[1] - self.pad_top,
            intersection[2] - self.pad_left,
            intersection[3] - self.pad_top,
        )
        crop_bbox = (
            _clip(unpadded[0] / self.scale_x, 0.0, float(self.crop_width)),
            _clip(unpadded[1] / self.scale_y, 0.0, float(self.crop_height)),
            _clip(unpadded[2] / self.scale_x, 0.0, float(self.crop_width)),
            _clip(unpadded[3] / self.scale_y, 0.0, float(self.crop_height)),
        )
        crop_left, crop_top, crop_right, crop_bottom = self.crop_edges
        source_bbox = (
            _clip(crop_left + crop_bbox[0], float(crop_left), float(crop_right)),
            _clip(crop_top + crop_bbox[1], float(crop_top), float(crop_bottom)),
            _clip(crop_left + crop_bbox[2], float(crop_left), float(crop_right)),
            _clip(crop_top + crop_bbox[3], float(crop_top), float(crop_bottom)),
        )
        source_bbox = (
            _clip(source_bbox[0], 0.0, float(self.source_width)),
            _clip(source_bbox[1], 0.0, float(self.source_height)),
            _clip(source_bbox[2], 0.0, float(self.source_width)),
            _clip(source_bbox[3], 0.0, float(self.source_height)),
        )
        if source_bbox[0] >= source_bbox[2] or source_bbox[1] >= source_bbox[3]:
            raise RoiTransformError(
                "inverse-mapped source bbox is degenerate",
                code="mapped_source_degenerate",
            )
        try:
            norm = pixel_xyxy_to_norm1000(
                source_bbox,
                image_width=self.source_width,
                image_height=self.source_height,
            )
        except DataContractError as exc:
            raise RoiTransformError(
                "inverse-mapped norm1000 bbox is degenerate",
                code="mapped_norm1000_degenerate",
            ) from exc
        if norm[0] >= norm[2] or norm[1] >= norm[3]:
            raise RoiTransformError(
                "inverse-mapped norm1000 bbox is degenerate",
                code="mapped_norm1000_degenerate",
            )
        return InverseMapping(
            canvas_bbox=canvas_bbox,
            canvas_content_intersection=intersection,
            unpadded_bbox=unpadded,
            crop_bbox=crop_bbox,
            source_bbox=source_bbox,
            norm1000_bbox=norm,
        )

    def map_canvas_bbox_to_norm1000(
        self, bbox: Sequence[float]
    ) -> tuple[int, int, int, int]:
        return self.inverse_canvas_bbox(bbox).norm1000_bbox

    def to_receipt_dict(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "source_size": [self.source_width, self.source_height],
            "roi_percent_xywh": [
                self.roi_percent.x,
                self.roi_percent.y,
                self.roi_percent.width,
                self.roi_percent.height,
            ],
            "roi_float_edges": list(self.roi_float_edges),
            "clipped_float_edges": list(self.clipped_float_edges),
            "crop_edges": list(self.crop_edges),
            "crop_edge_convention": CROP_EDGE_CONVENTION,
            "canvas_size": [self.canvas_width, self.canvas_height],
            "realized_size": [self.realized_width, self.realized_height],
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "padding": {
                "left": self.pad_left,
                "top": self.pad_top,
                "right": self.pad_right,
                "bottom": self.pad_bottom,
            },
            "resampler": RESAMPLER,
            "pad_value_rgb": list(PAD_VALUE_RGB),
            "pixel_edge_convention": PIXEL_EDGE_CONVENTION,
            "processor_kwargs": {"do_resize": self.processor_do_resize},
        }


# Concise alias for callers that do not need the implementation name.
RoiTransform = RoiLetterboxTransform


def _round_half_up(value: Fraction) -> int:
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _coerce_label_studio_roi(roi: Sequence[float]) -> LabelStudioRoi:
    if not isinstance(roi, Sequence) or isinstance(roi, (str, bytes)) or len(roi) != 4:
        raise RoiTransformError(
            "ROI must contain x, y, width, height percentages", code="roi_shape"
        )
    return LabelStudioRoi(roi[0], roi[1], roi[2], roi[3])


def _receipt_sequence(value: Any, *, length: int, field: str) -> tuple[Any, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise RoiTransformError(
            f"transform receipt {field} must contain {length} values",
            code="receipt_invalid",
        )
    return tuple(value)


def _strict_bbox(
    bbox: Sequence[float], *, field: str
) -> tuple[float, float, float, float]:
    try:
        raw_values = tuple(bbox)
    except TypeError as exc:
        raise RoiTransformError(
            f"{field} must contain four numbers", code="bbox_shape"
        ) from exc
    if len(raw_values) != 4:
        raise RoiTransformError(f"{field} must contain four numbers", code="bbox_shape")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw_values
    ):
        raise RoiTransformError(
            f"{field} values must be int or float", code="bbox_type"
        )
    values = tuple(float(value) for value in raw_values)
    if any(not math.isfinite(value) for value in values):
        raise RoiTransformError(f"{field} must be finite", code="bbox_non_finite")
    if values[0] >= values[2] or values[1] >= values[3]:
        raise RoiTransformError(
            f"{field} must be strictly non-degenerate", code="bbox_degenerate"
        )
    return values


def _positive_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RoiTransformError(
            f"{field} must be a positive integer", code="dimension_invalid"
        )
    return value


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))

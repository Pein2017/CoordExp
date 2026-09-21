"""Feature-space intervention primitives for one fixed-encoding Qwen probe.

The module deliberately owns no model loading, decoding, parsing, or artifact
policy.  It only validates and transforms the tuple returned by Qwen3 Vision-
Language (Qwen3-VL) ``get_image_features``.  The experiment-local script can
therefore reuse the current Hugging Face generation path without introducing a
second decoder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import types
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image


class FeatureLayoutError(ValueError):
    """Raised when a visual feature substitution would change semantics."""


FeatureMode = Literal["clean", "target", "control"]


PixelMode = Literal["clean", "target", "control"]


@dataclass(frozen=True)
class PixelBounds:
    """Half-open integer pixel bounds in ``x1, y1, x2, y2`` order."""

    x1: int
    y1: int
    x2: int
    y2: int

    def validate(self, *, width: int, height: int) -> "PixelBounds":
        values = (self.x1, self.y1, self.x2, self.y2)
        if any(not isinstance(value, int) for value in values):
            raise FeatureLayoutError("pixel bounds must contain integers")
        if self.x1 < 0 or self.y1 < 0 or self.x2 > int(width) or self.y2 > int(height):
            raise FeatureLayoutError(
                f"pixel bounds {values} exceed image dimensions {(width, height)}"
            )
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise FeatureLayoutError(f"pixel bounds must have positive area, got {values}")
        return self

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y2 - self.y1, self.x2 - self.x1)

    @property
    def count(self) -> int:
        height, width = self.shape
        return height * width

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


def _require_rgb_uint8(value: np.ndarray, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise FeatureLayoutError(f"{name} must be a numpy array")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise FeatureLayoutError(
            f"{name} must have uint8 RGB shape [height,width,3], got "
            f"dtype={value.dtype}, shape={tuple(value.shape)}"
        )
    if int(value.shape[0]) <= 0 or int(value.shape[1]) <= 0:
        raise FeatureLayoutError(f"{name} must have positive dimensions")
    return value


def rgb_array_sha256(value: np.ndarray) -> str:
    """Return the SHA-256 digest of a contiguous RGB uint8 array."""

    array = _require_rgb_uint8(value, name="RGB array")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a source image file without decoding or round-tripping it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb_uint8(path: Path) -> np.ndarray:
    """Decode a source image directly to an in-memory RGB uint8 array."""

    with Image.open(path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        array = np.asarray(image, dtype=np.uint8).copy()
    return _require_rgb_uint8(array, name=f"decoded image {path}")


def _mask_sha256(mask: np.ndarray) -> str:
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise FeatureLayoutError("pixel mask must have bool [height,width] shape")
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def _masked_rgb_sha256(value: np.ndarray, mask: np.ndarray) -> str:
    """Hash RGB pixels selected by a two-dimensional mask."""

    array = _require_rgb_uint8(value, name="RGB array")
    if mask.dtype != np.bool_ or mask.shape != array.shape[:2]:
        raise FeatureLayoutError("RGB complement mask shape or dtype is invalid")
    selected = array[mask]
    if selected.size == 0:
        raise FeatureLayoutError("RGB hash mask must select at least one pixel")
    return rgb_array_sha256(selected.reshape(-1, 1, 3))


def _pixel_diff_summary(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    _require_rgb_uint8(left, name="left RGB array")
    _require_rgb_uint8(right, name="right RGB array")
    if left.shape != right.shape or tuple(mask.shape) != tuple(left.shape[:2]):
        raise FeatureLayoutError("pixel diff arrays and mask must have compatible shapes")
    delta = right.astype(np.float32) - left.astype(np.float32)
    values = delta[mask].reshape(-1, 3)
    absolute = values.abs() if isinstance(values, torch.Tensor) else np.abs(values)
    squared = values * values
    if values.size == 0:
        raise FeatureLayoutError("pixel diff mask must select at least one pixel")
    return {
        "selected_pixel_count": int(mask.sum()),
        "mean_abs_rgb": [float(x) for x in absolute.mean(axis=0).tolist()],
        "mean_abs_all_channels": float(absolute.mean()),
        "root_mean_squared_all_channels": float(np.sqrt(squared.mean())),
        "max_abs_channel": int(absolute.max()),
        "changed_pixel_count": int(np.any(values != 0, axis=1).sum()),
        "float32": True,
    }


def compose_rgb_patch(
    recipient: np.ndarray,
    donor: np.ndarray,
    *,
    bounds: PixelBounds | None,
    mode: PixelMode,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compose a no-resize same-position donor patch in raw RGB space.

    ``mode='clean'`` is an exact copy with an empty mask.  Target and control
    modes replace exactly the half-open rectangle in ``recipient`` with the
    donor pixels at the same coordinates.  No blending, interpolation, or
    image encoding occurs here.
    """

    if mode not in {"clean", "target", "control"}:
        raise FeatureLayoutError(f"unsupported pixel composition mode: {mode}")
    recipient = _require_rgb_uint8(recipient, name="recipient")
    donor = _require_rgb_uint8(donor, name="donor")
    if recipient.shape != donor.shape:
        raise FeatureLayoutError(
            f"recipient and donor RGB shapes differ: {recipient.shape} != {donor.shape}"
        )
    height, width = (int(recipient.shape[0]), int(recipient.shape[1]))
    if mode == "clean":
        if bounds is not None:
            raise FeatureLayoutError("clean pixel composition must use an empty mask")
        mask = np.zeros((height, width), dtype=np.bool_)
        composed = recipient.copy()
    else:
        if bounds is None:
            raise FeatureLayoutError(f"{mode} pixel composition requires bounds")
        bounds.validate(width=width, height=height)
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[bounds.y1 : bounds.y2, bounds.x1 : bounds.x2] = True
        composed = recipient.copy()
        composed[bounds.y1 : bounds.y2, bounds.x1 : bounds.x2] = donor[
            bounds.y1 : bounds.y2, bounds.x1 : bounds.x2
        ]
    selected_source = donor[mask]
    selected_composed = composed[mask]
    selected_recipient = recipient[mask]
    complement = ~mask
    exact_selected = bool(np.array_equal(selected_source, selected_composed))
    exact_complement = bool(np.array_equal(recipient[complement], composed[complement]))
    if mode == "clean":
        exact_selected = True
        exact_complement = bool(np.array_equal(recipient, composed))
    if not exact_selected or not exact_complement:
        raise FeatureLayoutError("pixel compositor failed exact selected/complement preservation")
    receipt: dict[str, object] = {
        "mode": mode,
        "source_dtype": str(recipient.dtype),
        "source_shape": [int(x) for x in recipient.shape],
        "recipient_rgb_sha256": rgb_array_sha256(recipient),
        "donor_rgb_sha256": rgb_array_sha256(donor),
        "composed_rgb_sha256": rgb_array_sha256(composed),
        "mask_sha256": _mask_sha256(mask),
        "mask_shape": [height, width],
        "mask_pixel_count": int(mask.sum()),
        "bounds_xyxy_half_open": None if bounds is None else bounds.to_list(),
        "bounds_shape": None if bounds is None else list(bounds.shape),
        "bounds_pixel_count": None if bounds is None else bounds.count,
        "selected_slice_exact_equal_to_donor": exact_selected,
        "complement_exact_equal_to_recipient": exact_complement,
        "clean_exact_rgb_copy": bool(mode == "clean" and np.array_equal(recipient, composed)),
        "recipient_complement_rgb_sha256": _masked_rgb_sha256(recipient, complement),
        "composed_complement_rgb_sha256": _masked_rgb_sha256(composed, complement),
        "complement_rgb_sha256_equal": _masked_rgb_sha256(recipient, complement)
        == _masked_rgb_sha256(composed, complement),
        "selected_recipient_rgb_sha256": rgb_array_sha256(selected_recipient.reshape(-1, 1, 3))
        if selected_recipient.size
        else None,
        "selected_donor_rgb_sha256": rgb_array_sha256(selected_source.reshape(-1, 1, 3))
        if selected_source.size
        else None,
        "selected_composed_rgb_sha256": rgb_array_sha256(selected_composed.reshape(-1, 1, 3))
        if selected_composed.size
        else None,
        "float32_pixel_difference": (
            None if mode == "clean" else _pixel_diff_summary(recipient, donor, mask)
        ),
    }
    return composed, receipt


def materialize_rgb_uint8(
    image: np.ndarray,
    *,
    image_processor: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Run the standard Qwen image processor on an RGB array, with no resize."""

    image = _require_rgb_uint8(image, name="image")
    pil_image = Image.fromarray(image, mode="RGB")
    try:
        encoded = image_processor(images=[pil_image], return_tensors="pt", do_resize=False)
    finally:
        pil_image.close()
    pixel_values = encoded.get("pixel_values")
    image_grid_thw = encoded.get("image_grid_thw")
    if not isinstance(pixel_values, torch.Tensor) or not isinstance(image_grid_thw, torch.Tensor):
        raise FeatureLayoutError("Qwen image processor must return pixel_values and image_grid_thw tensors")
    processor_receipt = {
        "do_resize": False,
        "pixel_values_shape": [int(x) for x in pixel_values.shape],
        "pixel_values_dtype": str(pixel_values.dtype),
        "pixel_values_sha256": tensor_sha256(pixel_values),
        "image_grid_thw_shape": [int(x) for x in image_grid_thw.shape],
        "image_grid_thw_dtype": str(image_grid_thw.dtype),
        "image_grid_thw": [int(x) for x in image_grid_thw.reshape(-1).tolist()],
        "image_grid_thw_sha256": tensor_sha256(image_grid_thw),
    }
    processor_payload = json.dumps(processor_receipt, sort_keys=True, separators=(",", ":")).encode()
    processor_receipt["processor_output_sha256"] = hashlib.sha256(processor_payload).hexdigest()
    return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, processor_receipt


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor bytes after a deterministic CPU copy.

    Shape, dtype, and device are recorded separately by the fingerprint helper;
    this function intentionally hashes only the executed tensor bytes so a
    complement comparison can be made on exactly the same selected indices.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor_sha256 expects a torch.Tensor")
    contiguous = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_fingerprint(value: torch.Tensor) -> dict[str, object]:
    return {
        "sha256": tensor_sha256(value),
        "shape": [int(x) for x in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _as_tuple(value: Sequence[torch.Tensor], *, name: str) -> tuple[torch.Tensor, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise FeatureLayoutError(f"{name} must be a non-empty tuple or list of tensors")
    result = tuple(value)
    if any(not isinstance(item, torch.Tensor) for item in result):
        raise FeatureLayoutError(f"{name} must contain tensors only")
    return result


@dataclass(frozen=True)
class FeatureBundle:
    """One image's primary and DeepStack visual feature streams."""

    primary: tuple[torch.Tensor, ...]
    deepstack: tuple[torch.Tensor, ...]

    @classmethod
    def from_runtime(
        cls,
        image_embeds: Sequence[torch.Tensor],
        deepstack_image_embeds: Sequence[torch.Tensor],
    ) -> "FeatureBundle":
        primary = _as_tuple(image_embeds, name="image_embeds")
        deepstack = _as_tuple(deepstack_image_embeds, name="deepstack_image_embeds")
        return cls(primary=primary, deepstack=deepstack)

    def clone(self) -> "FeatureBundle":
        return FeatureBundle(
            primary=tuple(value.detach().clone() for value in self.primary),
            deepstack=tuple(value.detach().clone() for value in self.deepstack),
        )

    def runtime_value(
        self,
        *,
        primary_container: type[tuple] | type[list] = tuple,
        deepstack_container: type[tuple] | type[list] = tuple,
    ) -> tuple[tuple[torch.Tensor, ...] | list[torch.Tensor], tuple[torch.Tensor, ...] | list[torch.Tensor]]:
        """Return Qwen-compatible tuple/list stream containers.

        The replay seam accepts either container type; it does not promise to
        preserve the container class returned by the live visual tower.
        """

        primary = (
            list(self.primary) if primary_container is list else tuple(self.primary)
        )
        deepstack = (
            list(self.deepstack) if deepstack_container is list else tuple(self.deepstack)
        )
        return primary, deepstack


@dataclass(frozen=True)
class VisualFeatureLayout:
    """Validated merged-grid layout shared by recipient and donor features."""

    grid_thw: tuple[int, int, int]
    merge_size: int
    temporal: int
    merged_height: int
    merged_width: int
    primary_token_count: int
    deepstack_token_counts: tuple[int, ...]

    @property
    def merged_spatial_token_count(self) -> int:
        return self.merged_height * self.merged_width


def validate_feature_layout(
    recipient: FeatureBundle,
    donor: FeatureBundle,
    *,
    grid_thw: Sequence[int],
    merge_size: int,
) -> VisualFeatureLayout:
    """Require compatible primary and DeepStack streams before substitution."""

    if len(grid_thw) != 3:
        raise FeatureLayoutError("image_grid_thw must contain exactly three values")
    grid = tuple(int(x) for x in grid_thw)
    temporal, grid_height, grid_width = grid
    if temporal <= 0 or grid_height <= 0 or grid_width <= 0:
        raise FeatureLayoutError(f"image_grid_thw must be positive, got {grid}")
    merge = int(merge_size)
    if merge <= 0 or grid_height % merge or grid_width % merge:
        raise FeatureLayoutError(
            f"visual grid {grid} is not divisible by merge size {merge}"
        )
    if len(recipient.primary) != len(donor.primary):
        raise FeatureLayoutError("recipient and donor primary stream counts differ")
    if len(recipient.deepstack) != len(donor.deepstack):
        raise FeatureLayoutError("recipient and donor DeepStack stream counts differ")
    expected_primary_tokens = temporal * (grid_height // merge) * (grid_width // merge)
    for stream_name, left_streams, right_streams in (
        ("primary", recipient.primary, donor.primary),
        ("deepstack", recipient.deepstack, donor.deepstack),
    ):
        for index, (left, right) in enumerate(zip(left_streams, right_streams, strict=True)):
            if left.ndim != 2 or right.ndim != 2:
                raise FeatureLayoutError(
                    f"{stream_name}[{index}] must have [tokens, hidden] shape"
                )
            if tuple(left.shape) != tuple(right.shape):
                raise FeatureLayoutError(
                    f"recipient and donor {stream_name}[{index}] shapes differ: "
                    f"{tuple(left.shape)} != {tuple(right.shape)}"
                )
            if not left.dtype.is_floating_point or not right.dtype.is_floating_point:
                raise FeatureLayoutError(f"{stream_name}[{index}] must be floating point")
            if int(left.shape[0]) != expected_primary_tokens:
                raise FeatureLayoutError(
                    f"{stream_name}[{index}] token count {left.shape[0]} != "
                    f"grid-derived count {expected_primary_tokens}"
                )
    if len(recipient.primary) != 1:
        raise FeatureLayoutError(
            "this one-image probe requires exactly one split primary image stream"
        )
    return VisualFeatureLayout(
        grid_thw=grid,
        merge_size=merge,
        temporal=temporal,
        merged_height=grid_height // merge,
        merged_width=grid_width // merge,
        primary_token_count=expected_primary_tokens,
        deepstack_token_counts=tuple(int(x.shape[0]) for x in recipient.deepstack),
    )


def clone_feature_bundle(
    image_embeds: Sequence[torch.Tensor],
    deepstack_image_embeds: Sequence[torch.Tensor],
) -> FeatureBundle:
    """Clone the runtime result while retaining its stream container semantics."""

    return FeatureBundle.from_runtime(image_embeds, deepstack_image_embeds).clone()


def feature_bundle_fingerprint(bundle: FeatureBundle) -> dict[str, object]:
    return {
        "primary": [_tensor_fingerprint(value) for value in bundle.primary],
        "deepstack": [_tensor_fingerprint(value) for value in bundle.deepstack],
    }


def _normalise_bbox(bbox_xyxy: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox_xyxy) != 4:
        raise FeatureLayoutError("bbox_xyxy must contain exactly four values")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if not all(torch.isfinite(torch.tensor(value)).item() for value in (x1, y1, x2, y2)):
        raise FeatureLayoutError("bbox_xyxy must be finite")
    if x2 <= x1 or y2 <= y1:
        raise FeatureLayoutError(f"bbox_xyxy must have positive area, got {bbox_xyxy}")
    return x1, y1, x2, y2


def _dilate_merged_mask_2d(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        raise FeatureLayoutError("merged mask must be two-dimensional")
    height, width = (int(value) for value in mask.shape)
    result = mask.clone()
    for dy, dx in (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ):
        source_y1 = max(0, -dy)
        source_y2 = min(height, height - dy)
        source_x1 = max(0, -dx)
        source_x2 = min(width, width - dx)
        target_y1 = max(0, dy)
        target_y2 = min(height, height + dy)
        target_x1 = max(0, dx)
        target_x2 = min(width, width + dx)
        result[target_y1:target_y2, target_x1:target_x2] |= mask[
            source_y1:source_y2, source_x1:source_x2
        ]
    return result


def build_merged_support_mask(
    *,
    bbox_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    layout: VisualFeatureLayout,
    halo: int = 1,
) -> torch.Tensor:
    """Build a row-major merged-token mask from a pixel-space bounding box.

    Qwen3-VL visual tokens are ordered temporal-frame major, then merged rows,
    then merged columns.  The research probe currently uses one temporal frame;
    the implementation still repeats the spatial mask for every validated
    temporal frame so a non-singleton grid fails only if the runtime shape is
    otherwise incompatible.
    """

    x1, y1, x2, y2 = _normalise_bbox(bbox_xyxy)
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise FeatureLayoutError("image dimensions must be positive")
    if halo < 0:
        raise FeatureLayoutError("halo must be non-negative")
    spatial = torch.zeros(
        (layout.merged_height, layout.merged_width), dtype=torch.bool
    )
    cell_width = width / layout.merged_width
    cell_height = height / layout.merged_height
    for row in range(layout.merged_height):
        cell_y1 = row * cell_height
        cell_y2 = (row + 1) * cell_height
        for column in range(layout.merged_width):
            cell_x1 = column * cell_width
            cell_x2 = (column + 1) * cell_width
            if cell_x2 > x1 and cell_x1 < x2 and cell_y2 > y1 and cell_y1 < y2:
                spatial[row, column] = True
    if not bool(spatial.any()):
        raise FeatureLayoutError("bbox does not intersect any merged visual cell")
    for _ in range(int(halo)):
        spatial = _dilate_merged_mask_2d(spatial)
    return spatial.reshape(-1).repeat(layout.temporal)


def validate_equal_mask_shape_and_count(
    left: torch.Tensor,
    right: torch.Tensor,
) -> None:
    if left.dtype != torch.bool or right.dtype != torch.bool:
        raise FeatureLayoutError("target and control masks must be boolean")
    if tuple(left.shape) != tuple(right.shape):
        raise FeatureLayoutError(
            f"target/control mask shapes differ: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    left_count = int(left.sum().item())
    right_count = int(right.sum().item())
    if left_count != right_count:
        raise FeatureLayoutError(
            f"target/control mask counts differ: {left_count} != {right_count}"
        )
    if left_count <= 0:
        raise FeatureLayoutError("target/control masks must select at least one token")


def choose_deterministic_control_mask(
    target_mask: torch.Tensor,
    *,
    temporal: int,
    merged_height: int,
    merged_width: int,
    forbidden_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Translate one target shape in row-major order to a safe control region.

    The translated mask retains the exact selected-cell pattern and count.  A
    forbidden mask is required by the executable panel when active candidate
    regions are known; omitting it is useful only for isolated unit tests.
    """

    expected = int(temporal) * int(merged_height) * int(merged_width)
    if target_mask.ndim != 1 or int(target_mask.numel()) != expected:
        raise FeatureLayoutError("target mask length does not match merged layout")
    target = target_mask.to(dtype=torch.bool, device="cpu").clone().reshape(
        int(temporal), int(merged_height), int(merged_width)
    )
    pattern = target.any(dim=0)
    if not bool(pattern.any()):
        raise FeatureLayoutError("target mask must select at least one spatial cell")
    rows, columns = torch.where(pattern)
    min_row, max_row = int(rows.min()), int(rows.max())
    min_col, max_col = int(columns.min()), int(columns.max())
    if forbidden_mask is not None:
        if forbidden_mask.ndim != 1 or int(forbidden_mask.numel()) != expected:
            raise FeatureLayoutError("forbidden mask length does not match merged layout")
        forbidden = forbidden_mask.to(dtype=torch.bool, device="cpu").reshape(
            int(temporal), int(merged_height), int(merged_width)
        )
    else:
        forbidden = torch.zeros_like(target)
    target_count = int(target.sum().item())
    for shift_row in range(-min_row, int(merged_height) - max_row):
        for shift_col in range(-min_col, int(merged_width) - max_col):
            candidate = torch.zeros_like(target)
            for frame in range(int(temporal)):
                source = target[frame]
                destination = candidate[frame]
                destination[
                    min_row + shift_row : max_row + shift_row + 1,
                    min_col + shift_col : max_col + shift_col + 1,
                ] = source[min_row : max_row + 1, min_col : max_col + 1]
            if int(candidate.sum().item()) != target_count:
                raise FeatureLayoutError("deterministic translation changed mask count")
            if bool((candidate & forbidden).any()):
                continue
            # Never return the same mask as the target: the control must be a
            # genuinely unrelated spatial intervention.
            if torch.equal(candidate, target):
                continue
            return candidate.reshape(-1)
    raise FeatureLayoutError("could not find an equal-count control mask outside forbidden support")


def prune_symmetric_translation_overlap(
    target_mask: torch.Tensor,
    control_mask: torch.Tensor,
    *,
    temporal: int,
    merged_height: int,
    merged_width: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Remove physically overlapping relative cells from two translated masks.

    The operation is deliberately symmetric: when two equal-shaped translated
    masks overlap at one physical merged cell, the corresponding relative cell
    is removed from *both* masks.  The resulting masks remain translation
    equivalent and have equal counts, while a non-overlap receipt records the
    removed relative positions.  This is preferable to silently discarding an
    overlap from only the control region.
    """

    validate_equal_mask_shape_and_count(target_mask, control_mask)
    expected = int(temporal) * int(merged_height) * int(merged_width)
    if int(target_mask.numel()) != expected:
        raise FeatureLayoutError("translation masks do not match merged layout")
    target = target_mask.to(dtype=torch.bool, device="cpu").clone().reshape(
        int(temporal), int(merged_height), int(merged_width)
    )
    control = control_mask.to(dtype=torch.bool, device="cpu").clone().reshape(
        int(temporal), int(merged_height), int(merged_width)
    )
    target_spatial = target.any(dim=0)
    control_spatial = control.any(dim=0)
    target_rows, target_cols = torch.where(target_spatial)
    control_rows, control_cols = torch.where(control_spatial)
    if target_rows.numel() == 0 or control_rows.numel() == 0:
        raise FeatureLayoutError("translation masks must select spatial cells")
    target_origin = (int(target_rows.min()), int(target_cols.min()))
    control_origin = (int(control_rows.min()), int(control_cols.min()))
    target_rel = torch.zeros_like(target_spatial)
    control_rel = torch.zeros_like(control_spatial)
    target_rel[
        target_rows - target_origin[0], target_cols - target_origin[1]
    ] = True
    control_rel[
        control_rows - control_origin[0], control_cols - control_origin[1]
    ] = True
    if not torch.equal(target_rel, control_rel):
        raise FeatureLayoutError(
            "target/control masks are not equal-shaped translations; cannot prune symmetrically"
        )
    physical_overlap = target_spatial & control_spatial
    overlap_target_relative: list[tuple[int, int]] = []
    overlap_control_relative: list[tuple[int, int]] = []
    for row, col in zip(*torch.where(physical_overlap), strict=True):
        overlap_target_relative.append(
            (int(row) - target_origin[0], int(col) - target_origin[1])
        )
        overlap_control_relative.append(
            (int(row) - control_origin[0], int(col) - control_origin[1])
        )
    # Use the control-relative coordinates for the symmetric removal.  This
    # guarantees that the final control no longer touches the target's halo;
    # removing target-relative coordinates would remove the wrong translated
    # cells and can leave the physical overlap intact.
    removed_relative = sorted(set(overlap_control_relative))
    if removed_relative:
        for relative_row, relative_col in removed_relative:
            target[:, target_origin[0] + relative_row, target_origin[1] + relative_col] = False
            control[:, control_origin[0] + relative_row, control_origin[1] + relative_col] = False
    target_final = target.reshape(-1)
    control_final = control.reshape(-1)
    validate_equal_mask_shape_and_count(target_final, control_final)
    if bool((target_final & control_final).any()):
        raise FeatureLayoutError("symmetric overlap pruning left a physical overlap")
    return target_final, control_final, {
        "target_base_indices": [int(value) for value in torch.where(target_mask)[0].tolist()],
        "control_base_indices": [int(value) for value in torch.where(control_mask)[0].tolist()],
        "target_origin_row_col": list(target_origin),
        "control_origin_row_col": list(control_origin),
        "physical_overlap_target_relative_positions": [
            list(value) for value in sorted(set(overlap_target_relative))
        ],
        "physical_overlap_control_relative_positions": [
            list(value) for value in sorted(set(overlap_control_relative))
        ],
        "removed_relative_positions": [list(value) for value in removed_relative],
        "target_final_indices": [int(value) for value in torch.where(target_final)[0].tolist()],
        "control_final_indices": [int(value) for value in torch.where(control_final)[0].tolist()],
        "target_final_count": int(target_final.sum().item()),
        "control_final_count": int(control_final.sum().item()),
        "disjoint": True,
    }


def _selected_indices(mask: torch.Tensor, token_count: int) -> torch.Tensor:
    if mask.ndim != 1 or int(mask.numel()) != token_count:
        raise FeatureLayoutError(
            f"selected mask length {mask.numel()} does not match feature token count {token_count}"
        )
    indices = torch.where(mask.to(dtype=torch.bool, device="cpu"))[0]
    if indices.numel() == 0:
        raise FeatureLayoutError("selected mask must select at least one feature token")
    return indices


def replace_selected_indices(
    recipient: FeatureBundle,
    donor: FeatureBundle,
    selected_mask: torch.Tensor,
) -> tuple[FeatureBundle, dict[str, object]]:
    """Substitute matching selected rows in primary and every DeepStack stream."""

    if len(recipient.primary) != len(donor.primary) or len(recipient.deepstack) != len(donor.deepstack):
        raise FeatureLayoutError("recipient and donor stream counts differ")
    selected_indices = _selected_indices(selected_mask, int(recipient.primary[0].shape[0]))
    selected_list = [int(value) for value in selected_indices.tolist()]
    changed_primary: list[torch.Tensor] = []
    changed_deepstack: list[torch.Tensor] = []
    stream_receipts: list[dict[str, object]] = []
    for stream_name, left_streams, right_streams, output in (
        ("primary", recipient.primary, donor.primary, changed_primary),
        ("deepstack", recipient.deepstack, donor.deepstack, changed_deepstack),
    ):
        for stream_index, (left, right) in enumerate(zip(left_streams, right_streams, strict=True)):
            if tuple(left.shape) != tuple(right.shape):
                raise FeatureLayoutError(
                    f"recipient/donor shape mismatch at {stream_name}[{stream_index}]"
                )
            updated = left.detach().clone()
            device_indices = selected_indices.to(device=updated.device)
            donor_device = right.detach().to(device=updated.device)
            donor_selected = donor_device[device_indices]
            updated[device_indices] = donor_selected
            substituted_selected = updated[device_indices]
            selected_exact_equal = bool(torch.equal(donor_selected, substituted_selected))
            selected_float32_max_abs_diff = float(
                (
                    donor_selected.to(dtype=torch.float32)
                    - substituted_selected.to(dtype=torch.float32)
                )
                .abs()
                .max()
                .item()
            )
            donor_selected_fingerprint = _tensor_fingerprint(donor_selected)
            substituted_selected_fingerprint = _tensor_fingerprint(substituted_selected)
            if not selected_exact_equal or selected_float32_max_abs_diff != 0.0:
                raise FeatureLayoutError(
                    "donor selected slice does not exactly equal substituted selected slice "
                    f"at {stream_name}[{stream_index}]"
                )
            complement_mask = torch.ones(int(left.shape[0]), dtype=torch.bool)
            complement_mask[selected_indices] = False
            before_complement = tensor_sha256(left.detach().to(device="cpu")[complement_mask])
            after_complement = tensor_sha256(updated.detach().to(device="cpu")[complement_mask])
            if before_complement != after_complement:
                raise FeatureLayoutError(
                    f"recipient complement changed at {stream_name}[{stream_index}]"
                )
            output.append(updated)
            stream_receipts.append(
                {
                    "stream": stream_name,
                    "stream_index": stream_index,
                    "selected_count": len(selected_list),
                    "selected_indices": selected_list,
                    "recipient_before": _tensor_fingerprint(left),
                    "donor_selected_source": _tensor_fingerprint(right),
                    "substituted_after": _tensor_fingerprint(updated),
                    "donor_selected_slice": donor_selected_fingerprint,
                    "substituted_selected_slice": substituted_selected_fingerprint,
                    "donor_selected_slice_sha256": donor_selected_fingerprint["sha256"],
                    "substituted_selected_slice_sha256": substituted_selected_fingerprint["sha256"],
                    "selected_slice_exact_equal": selected_exact_equal,
                    "selected_slice_canonical_float32_max_abs_diff": selected_float32_max_abs_diff,
                    "recipient_complement_sha256": before_complement,
                    "substituted_complement_sha256": after_complement,
                }
            )
    return FeatureBundle(primary=tuple(changed_primary), deepstack=tuple(changed_deepstack)), {
        "selected_indices": selected_list,
        "selected_count": len(selected_list),
        "streams": stream_receipts,
        "complement_preserved": True,
    }


class FeatureReplayController:
    """Patch ``get_image_features`` for one generation call and restore it.

    The controller keeps the original Qwen input path and only changes the
    result returned by the visual tower.  Every call is validated against the
    cached recipient grid, so a malformed batch cannot silently use a replay
    tensor for a different encoding.
    """

    def __init__(
        self,
        *,
        model: Any,
        recipient: FeatureBundle,
        donor: FeatureBundle,
        grid_thw: Sequence[int],
        merge_size: int,
        selected_mask: torch.Tensor | None = None,
        mode: FeatureMode = "clean",
    ) -> None:
        if mode not in {"clean", "target", "control"}:
            raise FeatureLayoutError(f"unsupported feature replay mode: {mode}")
        if mode != "clean" and selected_mask is None:
            raise FeatureLayoutError(f"{mode} replay requires selected_mask")
        self.model = model
        self.recipient = recipient.clone()
        self.donor = donor.clone()
        self.layout = validate_feature_layout(
            self.recipient,
            self.donor,
            grid_thw=grid_thw,
            merge_size=merge_size,
        )
        self.mode = mode
        self.selected_mask = (
            None if selected_mask is None else selected_mask.to(dtype=torch.bool, device="cpu")
        )
        if self.selected_mask is not None:
            _selected_indices(self.selected_mask, self.layout.primary_token_count)
        self.replayed, self.application = (
            (self.recipient.clone(), {"selected_count": 0, "complement_preserved": True})
            if mode == "clean"
            else replace_selected_indices(self.recipient, self.donor, self.selected_mask)  # type: ignore[arg-type]
        )
        self._owner: Any | None = None
        self._original: Any | None = None
        self._installed = False
        self._restored = False
        self._feature_call_count = 0
        self._grid_mismatch_count = 0

    @staticmethod
    def _owner_for_model(model: Any) -> Any:
        nested = getattr(model, "model", None)
        if nested is not None and callable(getattr(nested, "get_image_features", None)):
            return nested
        if callable(getattr(model, "get_image_features", None)):
            return model
        raise FeatureLayoutError("model has no Qwen get_image_features seam")

    def install(self) -> None:
        if self._installed:
            raise FeatureLayoutError("feature replay controller is already installed")
        owner = self._owner_for_model(self.model)
        original = owner.get_image_features
        self._owner = owner
        self._original = original

        def patched(owner_self: Any, pixel_values: torch.Tensor, image_grid_thw: Any = None):
            del owner_self, pixel_values
            self._feature_call_count += 1
            if image_grid_thw is None:
                raise FeatureLayoutError("replayed visual feature call omitted image_grid_thw")
            grid_rows = image_grid_thw.reshape(-1, 3)
            observed_rows = [tuple(int(value) for value in row.tolist()) for row in grid_rows]
            if any(row != self.layout.grid_thw for row in observed_rows):
                self._grid_mismatch_count += 1
                raise FeatureLayoutError(
                    f"replayed visual grid mismatch: {observed_rows} != {self.layout.grid_thw}"
                )
            batch_size = len(observed_rows)
            if batch_size <= 0:
                raise FeatureLayoutError("replayed visual grid has no images")
            primary_streams = tuple(
                stream
                for _ in range(batch_size)
                for stream in self.replayed.primary
            )
            deepstack_streams = tuple(
                torch.cat([stream] * batch_size, dim=0)
                for stream in self.replayed.deepstack
            )
            return FeatureBundle(
                primary=primary_streams,
                deepstack=deepstack_streams,
            ).runtime_value()

        # The model seam is bound to the owner instance, so MethodType
        # preserves the normal ``self`` call convention.  Qwen accepts tuple
        # or list stream containers; replay deliberately uses tuple defaults.
        owner.get_image_features = types.MethodType(patched, owner)
        self._installed = True
        self._restored = False

    def remove(self) -> None:
        if self._installed and self._owner is not None:
            setattr(self._owner, "get_image_features", self._original)
        self._installed = False
        self._restored = True

    def __enter__(self) -> "FeatureReplayController":
        self.install()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.remove()

    def receipt(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "grid_thw": list(self.layout.grid_thw),
            "merge_size": self.layout.merge_size,
            "primary_token_count": self.layout.primary_token_count,
            "deepstack_token_counts": list(self.layout.deepstack_token_counts),
            "selected_indices": (
                []
                if self.selected_mask is None
                else [int(value) for value in torch.where(self.selected_mask)[0].tolist()]
            ),
            "selected_count": (
                0 if self.selected_mask is None else int(self.selected_mask.sum().item())
            ),
            "feature_call_count": self._feature_call_count,
            "grid_mismatch_count": self._grid_mismatch_count,
            "application": self.application,
            "recipient_features": feature_bundle_fingerprint(self.recipient),
            "donor_features": feature_bundle_fingerprint(self.donor),
            "replayed_features": feature_bundle_fingerprint(self.replayed),
            "hook_restored": self._restored,
        }

    def validate_completed(self, *, expected_feature_calls: int = 1) -> dict[str, object]:
        if self._feature_call_count != int(expected_feature_calls):
            raise FeatureLayoutError(
                f"expected exactly {expected_feature_calls} visual feature calls, "
                f"observed {self._feature_call_count}"
            )
        if self._grid_mismatch_count:
            raise FeatureLayoutError("visual feature grid mismatch occurred")
        if not self._restored:
            raise FeatureLayoutError("feature replay hook was not restored")
        receipt = self.receipt()
        if self.mode != "clean" and not bool(self.application.get("complement_preserved")):
            raise FeatureLayoutError("feature replay complement preservation was not proven")
        return receipt


class FreshFeatureCaptureController:
    """Observe fresh Qwen visual features during a normal generation call.

    Unlike :class:`FeatureReplayController`, this controller never replaces
    model outputs.  It wraps the live ``get_image_features`` method solely to
    prove that each pixel-composed condition executes a fresh primary and
    DeepStack visual computation on the condition's processor tensors.
    """

    def __init__(self, *, model: Any, expected_grid_thw: Sequence[int]) -> None:
        if len(expected_grid_thw) != 3:
            raise FeatureLayoutError("expected visual grid must contain three values")
        self.model = model
        self.expected_grid_thw = tuple(int(x) for x in expected_grid_thw)
        self._owner: Any | None = None
        self._original: Any | None = None
        self._installed = False
        self._restored = False
        self._feature_call_count = 0
        self._grid_mismatch_count = 0
        self._feature_receipts: list[dict[str, object]] = []

    def install(self) -> None:
        if self._installed:
            raise FeatureLayoutError("fresh feature capture controller is already installed")
        owner = FeatureReplayController._owner_for_model(self.model)
        self._owner = owner
        self._original = owner.get_image_features

        def captured(owner_self: Any, pixel_values: torch.Tensor, image_grid_thw: Any = None):
            del owner_self
            self._feature_call_count += 1
            if image_grid_thw is None:
                raise FeatureLayoutError("fresh visual feature call omitted image_grid_thw")
            rows = image_grid_thw.reshape(-1, 3)
            observed = [tuple(int(value) for value in row.tolist()) for row in rows]
            if any(row != self.expected_grid_thw for row in observed):
                self._grid_mismatch_count += 1
                raise FeatureLayoutError(
                    f"fresh visual feature grid mismatch: {observed} != {self.expected_grid_thw}"
                )
            result = self._original(pixel_values, image_grid_thw)
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise FeatureLayoutError("fresh get_image_features must return two feature streams")
            bundle = FeatureBundle.from_runtime(result[0], result[1]).clone()
            self._feature_receipts.append(
                {
                    "call_index": self._feature_call_count - 1,
                    "input_pixel_values_shape": [int(x) for x in pixel_values.shape],
                    "input_pixel_values_dtype": str(pixel_values.dtype),
                    "input_pixel_values_sha256": tensor_sha256(pixel_values),
                    "input_image_grid_thw": [int(x) for x in image_grid_thw.reshape(-1).tolist()],
                    "primary": feature_bundle_fingerprint(bundle)["primary"],
                    "deepstack": feature_bundle_fingerprint(bundle)["deepstack"],
                }
            )
            return result

        owner.get_image_features = types.MethodType(captured, owner)
        self._installed = True
        self._restored = False

    def remove(self) -> None:
        if self._installed and self._owner is not None:
            setattr(self._owner, "get_image_features", self._original)
        self._installed = False
        self._restored = True

    def __enter__(self) -> "FreshFeatureCaptureController":
        self.install()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.remove()

    def validate_completed(self, *, expected_feature_calls: int = 1) -> dict[str, object]:
        if self._feature_call_count != int(expected_feature_calls):
            raise FeatureLayoutError(
                f"expected exactly {expected_feature_calls} fresh visual feature calls, "
                f"observed {self._feature_call_count}"
            )
        if self._grid_mismatch_count:
            raise FeatureLayoutError("fresh visual feature grid mismatch occurred")
        if not self._restored:
            raise FeatureLayoutError("fresh feature capture hook was not restored")
        return {
            "expected_grid_thw": list(self.expected_grid_thw),
            "feature_call_count": self._feature_call_count,
            "grid_mismatch_count": self._grid_mismatch_count,
            "hook_restored": self._restored,
            "fresh_visual_recomputation": True,
            "calls": list(self._feature_receipts),
        }

    def receipt(self) -> dict[str, object]:
        """Return the current capture counters without requiring completion."""

        return {
            "expected_grid_thw": list(self.expected_grid_thw),
            "feature_call_count": self._feature_call_count,
            "grid_mismatch_count": self._grid_mismatch_count,
            "hook_restored": self._restored,
            "calls": list(self._feature_receipts),
        }


def capture_feature_bundle(
    model: Any,
    model_inputs: Mapping[str, Any],
) -> FeatureBundle:
    """Run Qwen's visual tower once for one materialized image input."""

    owner = FeatureReplayController._owner_for_model(model)
    pixel_values = model_inputs.get("pixel_values")
    image_grid_thw = model_inputs.get("image_grid_thw")
    if not isinstance(pixel_values, torch.Tensor) or not isinstance(image_grid_thw, torch.Tensor):
        raise FeatureLayoutError("model inputs require pixel_values and image_grid_thw tensors")
    try:
        device = next(owner.parameters()).device
    except (StopIteration, AttributeError):
        device = pixel_values.device
    with torch.inference_mode():
        result = owner.get_image_features(
            pixel_values.to(device=device),
            image_grid_thw.to(device=device),
        )
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise FeatureLayoutError("get_image_features must return (image_embeds, deepstack_image_embeds)")
    return clone_feature_bundle(result[0], result[1])

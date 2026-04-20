from __future__ import annotations

from dataclasses import dataclass

from src.common.geometry.bbox_parameterization import (
    cxcy_logw_logh_norm1000_to_xyxy_norm1000,
    cxcywh_norm1000_to_xyxy_norm1000,
)
from src.common.geometry.coord_utils import MAX_BIN


@dataclass(frozen=True)
class FamilyProbeSpec:
    alias: str
    native_slots: tuple[str, ...]
    requires_family_native_probe: bool
    requires_canonical_projection: bool
    infer_mode: str
    bbox_format: str
    pred_coord_mode: str
    eval_compatibility_path: str
    canonical_projection_kind: str


_REGISTRY: dict[str, FamilyProbeSpec] = {
    "base_xyxy_merged": FamilyProbeSpec(
        alias="base_xyxy_merged",
        native_slots=("x1", "y1", "x2", "y2"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
        infer_mode="coord",
        bbox_format="xyxy",
        pred_coord_mode="pixel",
        eval_compatibility_path="confidence_postop",
        canonical_projection_kind="identity_xyxy",
    ),
    "raw_text_xyxy_pure_ce": FamilyProbeSpec(
        alias="raw_text_xyxy_pure_ce",
        native_slots=("x1", "y1", "x2", "y2"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
        infer_mode="text",
        bbox_format="xyxy",
        pred_coord_mode="norm1000",
        eval_compatibility_path="confidence_postop",
        canonical_projection_kind="identity_xyxy",
    ),
    "cxcywh_pure_ce": FamilyProbeSpec(
        alias="cxcywh_pure_ce",
        native_slots=("cx", "cy", "w", "h"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
        infer_mode="coord",
        bbox_format="cxcywh",
        pred_coord_mode="norm1000",
        eval_compatibility_path="constant_score_scored_jsonl",
        canonical_projection_kind="cxcywh_to_xyxy",
    ),
    "cxcy_logw_logh_pure_ce": FamilyProbeSpec(
        alias="cxcy_logw_logh_pure_ce",
        native_slots=("cx", "cy", "logw", "logh"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
        infer_mode="coord",
        bbox_format="cxcy_logw_logh",
        pred_coord_mode="norm1000",
        eval_compatibility_path="constant_score_scored_jsonl",
        canonical_projection_kind="cxcy_logw_logh_to_xyxy",
    ),
    "center_parameterization": FamilyProbeSpec(
        alias="center_parameterization",
        native_slots=("cx", "cy", "w", "h"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
        infer_mode="coord",
        bbox_format="center_parameterization",
        pred_coord_mode="pixel",
        eval_compatibility_path="family_specific_projection_audit",
        canonical_projection_kind="family_specific",
    ),
    "hard_soft_ce_2b": FamilyProbeSpec(
        alias="hard_soft_ce_2b",
        native_slots=("x1", "y1", "x2", "y2"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
        infer_mode="coord",
        bbox_format="xyxy",
        pred_coord_mode="pixel",
        eval_compatibility_path="confidence_postop",
        canonical_projection_kind="identity_xyxy",
    ),
}


def get_family_probe_spec(alias: str) -> FamilyProbeSpec:
    return _REGISTRY[alias]


def native_slot_names(alias: str) -> tuple[str, ...]:
    return get_family_probe_spec(alias).native_slots


def _pixels_xyxy_to_norm1000(
    values: list[int] | tuple[int, ...],
    *,
    image_width: int,
    image_height: int,
) -> list[float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive for pixel projection")
    denom_x = max(1.0, float(image_width) - 1.0)
    denom_y = max(1.0, float(image_height) - 1.0)
    out: list[float] = []
    for index, value in enumerate(values):
        if index % 2 == 0:
            out.append((float(value) / denom_x) * float(MAX_BIN))
        else:
            out.append((float(value) / denom_y) * float(MAX_BIN))
    return out


def canonical_xyxy_norm1000(
    alias: str,
    values: list[int] | tuple[int, ...],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[float]:
    spec = get_family_probe_spec(alias)
    if len(values) != 4:
        raise ValueError(f"bbox values must contain 4 slots, got {len(values)}")
    if spec.canonical_projection_kind == "identity_xyxy":
        if spec.pred_coord_mode == "pixel":
            if image_width is None or image_height is None:
                raise ValueError(
                    f"family {alias!r} requires image_width and image_height for pixel projection"
                )
            return _pixels_xyxy_to_norm1000(
                values,
                image_width=image_width,
                image_height=image_height,
            )
        return [float(v) for v in values]
    if spec.canonical_projection_kind == "cxcywh_to_xyxy":
        return [float(v) for v in cxcywh_norm1000_to_xyxy_norm1000(values)]
    if spec.canonical_projection_kind == "cxcy_logw_logh_to_xyxy":
        return [float(v) for v in cxcy_logw_logh_norm1000_to_xyxy_norm1000(values)]
    raise NotImplementedError(
        f"family {alias!r} uses canonical projection kind {spec.canonical_projection_kind!r}"
    )


__all__ = [
    "FamilyProbeSpec",
    "canonical_xyxy_norm1000",
    "get_family_probe_spec",
    "native_slot_names",
]

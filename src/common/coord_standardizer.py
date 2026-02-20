"""Coordinate standardization helpers for CoordExp.

This module unifies coordinate handling across inference/evaluation by
auto-detecting the incoming representation (pixel vs. norm1000 ints vs.
``<|coord_*|>`` tokens) and emitting pixel-space coordinates alongside a
textual representation. It is intended to be the single entry point for
normalizing model outputs and ground-truth JSONL records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple, cast

from src.common.geometry import (
    MAX_BIN,
    bbox_from_points,
    coerce_point_list,
    denorm_and_clamp,
    flatten_points,
    is_degenerate_bbox,
)
from src.common.prediction_parsing import GEOM_KEYS, coords_are_pixel, parse_prediction

GeomType = Literal["bbox_2d", "poly"]
ModeType = Literal["coord", "text"]


@dataclass
class StandardizedObject:
    """Pixel-space geometry with optional text view."""

    type: GeomType
    points: List[int]
    desc: str
    score: float
    points_text: str
    coord_mode: Literal["pixel", "norm1000"]


class CoordinateStandardizer:
    """Mode-aware coordinate validation and scaling helpers.

    The standardizer accepts heterogeneous inputs:
    - Ground-truth objects with geometry keys (``bbox_2d``/``poly``)
    - Parsed prediction dicts containing ``type`` + ``points`` (as returned by
      ``parse_prediction``)
    It detects norm1000 tokens/ints vs. pixel coordinates, converts to pixel
    ints, and emits a text-friendly ``points_text`` for downstream consumers.
    """

    def __init__(
        self,
        mode: ModeType,
        *,
        emit_text: bool = True,
        pred_coord_mode: Literal["auto", "norm1000", "pixel"] = "auto",
    ) -> None:
        self.mode = mode
        self.emit_text = emit_text
        if pred_coord_mode not in {"auto", "norm1000", "pixel"}:
            raise ValueError("pred_coord_mode must be auto|norm1000|pixel")
        self.pred_coord_mode = pred_coord_mode

    @staticmethod
    def _points_to_text(points: Sequence[int]) -> str:
        return " ".join(str(int(v)) for v in points)

    @staticmethod
    def _parse_geom(obj: Dict[str, Any]) -> Tuple[GeomType, List[float], bool, str]:
        """Extract geometry type and raw points from heterogeneous objects."""

        from src.common.geometry.object_geometry import extract_single_geometry

        try:
            gtype, pts_raw = extract_single_geometry(
                obj,
                allow_type_and_points=True,
                allow_nested_points=True,
                path="object",
            )
        except ValueError as exc:
            msg = str(exc)
            if "type must be bbox_2d|poly" in msg:
                raise ValueError("geometry_kind") from exc
            if "got both" in msg or "got none" in msg:
                raise ValueError("geometry_keys") from exc
            raise ValueError("geometry_points") from exc

        points, had_tokens = coerce_point_list(pts_raw)
        if points is None:
            raise ValueError("coord_parse")
        if obj.get("_had_tokens"):
            had_tokens = True

        desc = str(obj.get("desc", ""))
        return cast(GeomType, gtype), points, had_tokens, desc

    def _choose_coord_mode_pred(
        self,
        points: Sequence[float],
        *,
        had_tokens: bool,
        width: int,
        height: int,
    ) -> Literal["norm1000", "pixel"]:
        if self.pred_coord_mode == "norm1000":
            return "norm1000"
        if self.pred_coord_mode == "pixel":
            return "pixel"
        if had_tokens:
            return "norm1000"
        if coords_are_pixel(points, width, height, had_tokens=had_tokens):
            return "pixel"
        return "norm1000"

    def _scale_points(
        self,
        *,
        points: Sequence[float],
        had_tokens: bool,
        width: int,
        height: int,
        kind: GeomType,
        is_gt: bool,
    ) -> Tuple[List[int], Literal["norm1000", "pixel"]]:
        if width <= 0 or height <= 0:
            raise ValueError("size_mismatch")

        coord_mode_override = (
            None
            if is_gt
            else (None if self.pred_coord_mode == "auto" else self.pred_coord_mode)
        )

        if self.mode == "coord":
            if is_gt:
                max_p = max(points) if points else 0
                max_img = max(width, height)
                likely_pixel = max_p <= max_img
                if max_p > MAX_BIN:
                    coord_mode = "pixel"
                elif likely_pixel and not had_tokens:
                    coord_mode = "pixel"
                else:
                    if any(p < 0 or p > MAX_BIN for p in points):
                        raise ValueError("mode_gt_mismatch")
                    coord_mode = "norm1000"
            else:
                if coord_mode_override is None:
                    coord_mode = "norm1000"
                else:
                    coord_mode = coord_mode_override  # explicit override
                if coord_mode == "norm1000" and any(
                    p < 0 or p > MAX_BIN for p in points
                ):
                    raise ValueError("coord_range")
        else:  # text mode
            if is_gt:
                if had_tokens:
                    raise ValueError("mode_gt_mismatch")
                coord_mode = "pixel"
            else:
                coord_mode = self._choose_coord_mode_pred(
                    points, had_tokens=had_tokens, width=width, height=height
                )

        pts_px = denorm_and_clamp(points, width, height, coord_mode=coord_mode)

        if kind == "bbox_2d" and len(pts_px) != 4:
            raise ValueError("bbox_points")
        if kind == "poly" and len(pts_px) < 6:
            raise ValueError("poly_points")

        x1, y1, x2, y2 = bbox_from_points(pts_px)
        if is_degenerate_bbox(x1, y1, x2, y2):
            raise ValueError("degenerate")

        return [int(v) for v in pts_px], coord_mode

    def process_objects(
        self,
        objs: Iterable[Dict[str, Any]],
        *,
        width: int,
        height: int,
        is_gt: bool,
        errors: List[str],
    ) -> List[Dict[str, Any]]:
        """Standardize a list of geometry dicts to pixel space."""

        processed: List[Dict[str, Any]] = []
        for obj in objs:
            try:
                gtype, points, had_tokens, desc = self._parse_geom(obj)
                pts_px, coord_mode = self._scale_points(
                    points=points,
                    had_tokens=had_tokens,
                    width=width,
                    height=height,
                    kind=gtype,
                    is_gt=is_gt,
                )
                score = obj.get("score", 1.0)
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 1.0

                out_obj: Dict[str, Any] = {
                    "type": gtype,
                    "points": pts_px,
                    "desc": desc,
                    "score": score,
                    "_coord_mode": coord_mode,
                }
                if self.emit_text:
                    out_obj["points_text"] = self._points_to_text(pts_px)
                processed.append(out_obj)
            except ValueError as exc:
                errors.append(str(exc))
        return processed

    def process_prediction_text(
        self,
        raw_text: str,
        *,
        width: int,
        height: int,
        errors: List[str],
    ) -> List[Dict[str, Any]]:
        """Parse model text and standardize predictions to pixel space."""

        parsed = parse_prediction(raw_text)
        if not parsed:
            errors.append("empty_pred")
            return []
        preds = self.process_objects(
            parsed, width=width, height=height, is_gt=False, errors=errors
        )
        if not preds and "empty_pred" not in errors:
            errors.append("empty_pred")
        return preds

    def process_record_gt(
        self,
        record: Dict[str, Any],
        *,
        width: int,
        height: int,
        errors: List[str],
    ) -> List[Dict[str, Any]]:
        """Standardize GT objects from a dataset record."""

        objs = record.get("objects") or record.get("gt") or []
        return self.process_objects(
            objs, width=width, height=height, is_gt=True, errors=errors
        )


__all__ = ["CoordinateStandardizer", "StandardizedObject"]

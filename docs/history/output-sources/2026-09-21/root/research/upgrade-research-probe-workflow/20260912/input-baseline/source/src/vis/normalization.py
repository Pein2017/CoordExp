"""Canonical pixel-space normalization for detection visualization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.errors import ArtifactContractError, DataContractError
from src.data.geometry import coord_bins_to_pixel_xyxy
from src.eval.detection_categories import normalize_coco_category_name
from src.inference.artifacts import RAW_NAME, SCORED_NAME


@dataclass(frozen=True)
class VisualObject:
    index: int
    description: str
    normalized_description: str
    bbox_pixel_xyxy: tuple[float, float, float, float]
    source_bbox: tuple[Any, Any, Any, Any]
    source_coord_space: str
    source_coord_bins: tuple[Any, Any, Any, Any] | None = None

    def to_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "description": self.description,
            "normalized_description": self.normalized_description,
            "bbox_pixel_xyxy": [_json_number(value) for value in self.bbox_pixel_xyxy],
            "source_bbox": list(self.source_bbox),
            "source_coord_space": self.source_coord_space,
        }
        if self.source_coord_bins is not None:
            payload["source_coord_bins"] = list(self.source_coord_bins)
        return payload


@dataclass(frozen=True)
class VisualRow:
    row_id: str
    row_index: int
    image_path: Path
    source_image_path: str
    image_width: int
    image_height: int
    gt: tuple[VisualObject, ...]
    pred: tuple[VisualObject, ...]

    def gt_signature(self) -> tuple[Any, ...]:
        return (
            self.row_id,
            self.source_image_path,
            self.image_width,
            self.image_height,
            tuple(
                (
                    obj.normalized_description,
                    obj.bbox_pixel_xyxy,
                )
                for obj in self.gt
            ),
        )


@dataclass(frozen=True)
class ArtifactRows:
    artifact_dir: Path
    scored_jsonl: Path
    raw_jsonl: Path
    rows: tuple[VisualRow, ...]


def load_visual_rows(run_dir_or_scored_jsonl: str | Path) -> ArtifactRows:
    artifact_dir, scored_path = resolve_artifact_input(run_dir_or_scored_jsonl)
    raw_path = artifact_dir / RAW_NAME
    _require_file(scored_path, code="vis.missing_scored_artifact")
    _require_file(raw_path, code="vis.missing_raw_artifact")
    raw_rows = _read_jsonl(raw_path)
    scored_rows = _read_jsonl(scored_path)
    if len(raw_rows) != len(scored_rows):
        raise ArtifactContractError(
            "raw and scored visualization artifacts must have the same row count",
            code="vis.raw_scored_row_count_mismatch",
            context={"raw_row_count": len(raw_rows), "scored_row_count": len(scored_rows)},
        )
    rows = tuple(
        _normalize_row_pair(
            raw=raw,
            scored=scored,
            row_index=index,
            artifact_dir=artifact_dir,
        )
        for index, (raw, scored) in enumerate(zip(raw_rows, scored_rows, strict=True))
    )
    return ArtifactRows(
        artifact_dir=artifact_dir,
        scored_jsonl=scored_path,
        raw_jsonl=raw_path,
        rows=rows,
    )


def resolve_artifact_input(run_dir_or_scored_jsonl: str | Path) -> tuple[Path, Path]:
    path = Path(run_dir_or_scored_jsonl)
    if path.name == SCORED_NAME:
        return path.parent, path
    if path.suffix == ".jsonl":
        raise ArtifactContractError(
            f"visualization JSONL input must be {SCORED_NAME}",
            code="vis.unsupported_jsonl_input",
            context={"path": str(path), "expected_name": SCORED_NAME},
        )
    return path, path / SCORED_NAME


def select_rows(
    rows: tuple[VisualRow, ...],
    *,
    row_ids: list[str] | tuple[str, ...] | None,
    limit: int | None,
) -> tuple[VisualRow, ...]:
    selected: tuple[VisualRow, ...]
    if row_ids is None:
        selected = rows
    else:
        by_id = _rows_by_id(rows)
        missing = [row_id for row_id in row_ids if row_id not in by_id]
        if missing:
            raise ArtifactContractError(
                "requested visualization row ids are missing",
                code="vis.row_id_missing",
                context={"missing_row_ids": missing},
            )
        selected = tuple(by_id[row_id] for row_id in row_ids)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ArtifactContractError(
                "visualization limit must be a non-negative integer",
                code="vis.invalid_limit",
                context={"limit": limit},
            )
        selected = selected[:limit]
    return selected


def _normalize_row_pair(
    *,
    raw: dict[str, Any],
    scored: dict[str, Any],
    row_index: int,
    artifact_dir: Path,
) -> VisualRow:
    raw_row_id = _row_id(raw, row_index=row_index, field="raw.row_id")
    scored_row_id = _row_id(scored, row_index=row_index, field="scored.row_id")
    if raw_row_id != scored_row_id:
        raise ArtifactContractError(
            "raw and scored visualization rows must preserve row identity",
            code="vis.raw_scored_row_id_mismatch",
            context={
                "row_index": row_index,
                "raw_row_id": raw_row_id,
                "scored_row_id": scored_row_id,
            },
        )
    if "gt" in scored and scored.get("gt") != raw.get("gt"):
        raise ArtifactContractError(
            "raw and scored visualization rows must preserve GT payload",
            code="vis.raw_scored_gt_mismatch",
            context={"row_id": raw_row_id, "row_index": row_index},
        )

    image_width = _positive_int(
        raw.get("image_width", raw.get("width", scored.get("image_width", scored.get("width")))),
        field="image_width",
        row_id=raw_row_id,
    )
    image_height = _positive_int(
        raw.get(
            "image_height",
            raw.get("height", scored.get("image_height", scored.get("height"))),
        ),
        field="image_height",
        row_id=raw_row_id,
    )
    source_image_path = _image_path(raw, scored=scored, row_id=raw_row_id)
    image_path = _resolve_image_path(source_image_path, artifact_dir=artifact_dir)
    raw_gt = _object_list(raw.get("gt"), field="gt", row_id=raw_row_id)
    scored_pred = _object_list(scored.get("pred"), field="pred", row_id=raw_row_id)
    return VisualRow(
        row_id=raw_row_id,
        row_index=_positive_int(raw.get("row_index", row_index), field="row_index", row_id=raw_row_id),
        image_path=image_path,
        source_image_path=source_image_path,
        image_width=image_width,
        image_height=image_height,
        gt=tuple(
            _gt_object(
                obj,
                index=index,
                row_id=raw_row_id,
                image_width=image_width,
                image_height=image_height,
            )
            for index, obj in enumerate(raw_gt)
        ),
        pred=tuple(
            _pred_object(obj, index=index, row_id=raw_row_id)
            for index, obj in enumerate(scored_pred)
        ),
    )


def _gt_object(
    obj: dict[str, Any],
    *,
    index: int,
    row_id: str,
    image_width: int,
    image_height: int,
) -> VisualObject:
    raw_bbox = _bbox_source(obj, field="gt.bbox", row_id=row_id, object_index=index)
    try:
        bbox_pixel = coord_bins_to_pixel_xyxy(
            raw_bbox,
            image_width=image_width,
            image_height=image_height,
            field="gt.bbox",
        )
    except DataContractError as exc:
        raise ArtifactContractError(
            "GT visualization bbox must be norm1000 xyxy coordinate bins",
            code="vis.invalid_gt_bbox",
            context={"row_id": row_id, "object_index": index, "bbox": list(raw_bbox)},
            cause=exc,
        ) from exc
    description = _description(obj)
    return VisualObject(
        index=index,
        description=description,
        normalized_description=normalize_coco_category_name(description),
        bbox_pixel_xyxy=tuple(float(value) for value in bbox_pixel),
        source_bbox=raw_bbox,
        source_coord_space="norm1000",
    )


def _pred_object(obj: dict[str, Any], *, index: int, row_id: str) -> VisualObject:
    raw_bbox = _bbox_source(obj, field="pred.bbox", row_id=row_id, object_index=index)
    bbox_pixel = _pixel_bbox(raw_bbox, row_id=row_id, object_index=index)
    description = _description(obj)
    coord_bins = obj.get("coord_bins")
    source_coord_bins = tuple(coord_bins) if isinstance(coord_bins, (list, tuple)) and len(coord_bins) == 4 else None
    return VisualObject(
        index=index,
        description=description,
        normalized_description=normalize_coco_category_name(description),
        bbox_pixel_xyxy=bbox_pixel,
        source_bbox=raw_bbox,
        source_coord_space="pixel",
        source_coord_bins=source_coord_bins,
    )


def _pixel_bbox(
    value: tuple[Any, Any, Any, Any],
    *,
    row_id: str,
    object_index: int,
) -> tuple[float, float, float, float]:
    parsed: list[float] = []
    for axis, item in enumerate(value):
        if isinstance(item, bool):
            raise _invalid_pred_bbox(row_id=row_id, object_index=object_index, bbox=value)
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise _invalid_pred_bbox(row_id=row_id, object_index=object_index, bbox=value) from exc
        if not math.isfinite(number):
            raise _invalid_pred_bbox(row_id=row_id, object_index=object_index, bbox=value)
        parsed.append(number)
    x1, y1, x2, y2 = parsed
    if x1 >= x2 or y1 >= y2:
        raise _invalid_pred_bbox(row_id=row_id, object_index=object_index, bbox=value)
    return x1, y1, x2, y2


def _invalid_pred_bbox(
    *,
    row_id: str,
    object_index: int,
    bbox: tuple[Any, Any, Any, Any],
) -> ArtifactContractError:
    return ArtifactContractError(
        "prediction visualization bbox must be finite positive-size pixel xyxy",
        code="vis.invalid_pred_bbox",
        context={"row_id": row_id, "object_index": object_index, "bbox": list(bbox)},
    )


def _bbox_source(
    obj: dict[str, Any],
    *,
    field: str,
    row_id: str,
    object_index: int,
) -> tuple[Any, Any, Any, Any]:
    raw_bbox = obj.get("bbox", obj.get("bbox_2d"))
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ArtifactContractError(
            "visualization bbox must be a four-value sequence",
            code="vis.invalid_bbox_shape",
            context={"row_id": row_id, "object_index": object_index, "field": field},
        )
    return tuple(raw_bbox)


def _description(obj: dict[str, Any]) -> str:
    return str(obj.get("description", obj.get("desc", "")))


def _object_list(value: Any, *, field: str, row_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ArtifactContractError(
            "visualization object field must be a list",
            code="vis.object_list_type",
            context={"row_id": row_id, "field": field, "value_type": type(value).__name__},
        )
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ArtifactContractError(
                "visualization object entries must be objects",
                code="vis.object_type",
                context={"row_id": row_id, "field": field, "index": index},
            )
        out.append(item)
    return out


def _image_path(raw: dict[str, Any], *, scored: dict[str, Any], row_id: str) -> str:
    value = raw.get("image_path", raw.get("image", scored.get("image_path", scored.get("image"))))
    if not isinstance(value, str) or not value.strip():
        raise ArtifactContractError(
            "visualization row requires a non-empty image path",
            code="vis.missing_image_path",
            context={"row_id": row_id},
        )
    return value.strip()


def _resolve_image_path(value: str, *, artifact_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = artifact_dir / path
    if candidate.exists():
        return candidate
    return path


def _row_id(row: dict[str, Any], *, row_index: int, field: str) -> str:
    value = row.get("row_id")
    if not isinstance(value, str) or not value.strip():
        raise ArtifactContractError(
            "visualization row requires non-empty row_id",
            code="vis.missing_row_id",
            context={"field": field, "row_index": row_index},
        )
    return value.strip()


def _positive_int(value: Any, *, field: str, row_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactContractError(
            "visualization integer field must be non-negative",
            code="vis.invalid_integer",
            context={"row_id": row_id, "field": field, "value": value},
        )
    if field in {"image_width", "image_height"} and value <= 0:
        raise ArtifactContractError(
            "visualization image dimensions must be positive",
            code="vis.invalid_image_size",
            context={"row_id": row_id, "field": field, "value": value},
        )
    return value


def _rows_by_id(rows: tuple[VisualRow, ...]) -> dict[str, VisualRow]:
    out: dict[str, VisualRow] = {}
    duplicates: list[str] = []
    for row in rows:
        if row.row_id in out:
            duplicates.append(row.row_id)
        out[row.row_id] = row
    if duplicates:
        raise ArtifactContractError(
            "visualization rows must have unique row ids",
            code="vis.duplicate_row_id",
            context={"duplicate_row_ids": sorted(set(duplicates))},
        )
    return out


def _require_file(path: Path, *, code: str) -> None:
    if not path.is_file():
        raise ArtifactContractError(
            "required visualization artifact is missing",
            code=code,
            context={"path": str(path)},
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ArtifactContractError(
                    "visualization JSONL row is malformed",
                    code="vis.json_decode",
                    context={"path": str(path), "row_index": row_index},
                    cause=exc,
                ) from exc
            if not isinstance(value, dict):
                raise ArtifactContractError(
                    "visualization JSONL row must be an object",
                    code="vis.json_row_type",
                    context={"path": str(path), "row_index": row_index},
                )
            rows.append(value)
    return rows


def _json_number(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return float(value)

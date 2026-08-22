"""Metadata contract for current COCO annotation views."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION_V1 = 1
COORDINATE_SPACE_NORM1000 = "norm1000"
COORDINATE_STORAGE_INTEGER = "integer"
COORDINATE_RANGE_NORM1000 = (0, 999)
COORDINATE_CHART_XYXY = "xyxy"
ASSISTANT_COORDINATE_RENDERING_QWEN_COORD_TOKENS = "qwen_coord_tokens"
IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE = "image_store_relative"


@dataclass(frozen=True)
class ViewMetadata:
    schema_version: int
    kind: str
    dataset: str
    view: str
    image_store: str
    path_anchor: str
    image_path_semantics: str
    coordinate_space: str
    coordinate_storage: str
    coordinate_range: tuple[int, int]
    coordinate_chart: str
    assistant_coordinate_rendering: str
    primary_jsonl: Mapping[str, str]
    summary: Mapping[str, int]
    sample_policy: str = "full"
    max_total_tokens: int | None = None


def write_view_metadata(path: Path, **values: Any) -> ViewMetadata:
    metadata = ViewMetadata(**values)
    _validate_view_metadata(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_view_metadata(path: Path) -> ViewMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["coordinate_range"] = tuple(payload["coordinate_range"])
    metadata = ViewMetadata(**payload)
    _validate_view_metadata(metadata)
    return metadata


def write_image_store_metadata(path: Path, **values: Any) -> dict[str, Any]:
    payload = {"schema_version": SCHEMA_VERSION_V1, "kind": "image_store", **values}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def resolve_view_image_root(
    meta: ViewMetadata,
    view_root: Path | None = None,
    *,
    repo_root: Path,
) -> Path:
    image_store = Path(meta.image_store)
    if image_store.is_absolute():
        if meta.path_anchor != "test_absolute":
            raise ValueError("view image_store must be repository-relative")
        return image_store.resolve()
    return (repo_root / image_store).resolve()


def resolve_image_path(value: str, image_root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("view image path must be relative and must not escape")
    return Path(image_root) / candidate


def _validate_view_metadata(meta: ViewMetadata) -> None:
    if meta.schema_version != SCHEMA_VERSION_V1 or meta.kind != "annotation_view":
        raise ValueError("unsupported annotation-view metadata schema")
    if not meta.dataset or not meta.view:
        raise ValueError("view dataset and view identity must be non-empty")
    if meta.coordinate_space != COORDINATE_SPACE_NORM1000:
        raise ValueError("view coordinate_space must be norm1000")
    if meta.coordinate_storage != COORDINATE_STORAGE_INTEGER:
        raise ValueError("view coordinate_storage must be integer")
    if tuple(meta.coordinate_range) != COORDINATE_RANGE_NORM1000:
        raise ValueError("view coordinate_range must be [0, 999]")
    if meta.coordinate_chart != COORDINATE_CHART_XYXY:
        raise ValueError("view coordinate_chart must be xyxy")
    if meta.image_path_semantics != IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE:
        raise ValueError("view image path semantics must be image-store-relative")
    if not meta.primary_jsonl:
        raise ValueError("view metadata must name at least one primary JSONL")
    if meta.sample_policy == "length_budget" and (meta.max_total_tokens or 0) <= 0:
        raise ValueError("length-budget views require max_total_tokens")

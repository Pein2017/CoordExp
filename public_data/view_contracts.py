from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ImageStoreMetadata:
    """Metadata describing a reusable image store."""

    schema_version: int
    kind: str
    dataset: str
    image_store: str
    image_path_semantics: str
    max_pixels: int
    visual_token_budget: int
    image_factor: int
    image_root: str
    splits: tuple[str, ...]


@dataclass(frozen=True)
class ViewMetadata:
    """Metadata describing an annotation view and its backing image store."""

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
    sample_policy: Mapping[str, Any] | None = None
    length_budget_scope: Mapping[str, Any] | None = None
    annotation_policy: str | None = None
    parent_view: str | None = None
    proxy_policy: Mapping[str, Any] | None = None
    summary: Mapping[str, Any] | None = None


_ABSOLUTE_PATH_ANCHORS = frozenset({"test_absolute", "debug_absolute"})


def load_view_metadata(path: Path) -> ViewMetadata:
    """Load view metadata from a JSON file.

    :param path: Metadata JSON path.
    :returns: Validated view metadata.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = ViewMetadata(
        **{
            **data,
            "coordinate_range": tuple(data["coordinate_range"]),
        }
    )
    _validate_view_metadata(metadata)
    return metadata


def write_view_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    """Write view metadata to a JSON file.

    :param path: Destination metadata path.
    :param metadata: JSON-serializable metadata mapping.
    """

    candidate = ViewMetadata(
        **{
            **metadata,
            "coordinate_range": tuple(metadata["coordinate_range"]),
        }
    )
    _validate_view_metadata(candidate)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_view_image_root(
    meta: ViewMetadata,
    view_root: Path,
    repo_root: Path | None = None,
) -> Path:
    """Resolve the image root referenced by a view metadata record.

    :param meta: View metadata containing the image store reference.
    :param view_root: Root directory for the view artifact.
    :param repo_root: Repository root for repo-root-anchored image stores.
    :returns: Absolute image-store root path.
    """

    _validate_view_metadata(meta)

    image_store = Path(meta.image_store)
    if image_store.is_absolute():
        if meta.path_anchor not in _ABSOLUTE_PATH_ANCHORS:
            raise ValueError("absolute image_store requires test/debug absolute path_anchor")
        return image_store.resolve()

    if meta.path_anchor != "repo_root":
        raise ValueError(f"unsupported path_anchor: {meta.path_anchor}")

    resolved_repo_root = repo_root if repo_root is not None else _infer_repo_root(view_root)
    return (resolved_repo_root / image_store).resolve()


def resolve_image_path(image_ref: str, image_root: Path) -> Path:
    """Resolve an image-store-relative image reference.

    :param image_ref: Relative image reference beginning with ``images/``.
    :param image_root: Absolute or relative image-store root.
    :returns: Absolute image path.
    """

    relative_ref = safe_relative_image_ref(image_ref)
    resolved_root = image_root.resolve()
    resolved_path = (resolved_root / relative_ref).resolve()

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"image_ref resolves outside image_root: {image_ref}") from exc

    return resolved_path


def safe_relative_image_ref(path: str) -> Path:
    """Validate and return an image-store-relative image path.

    :param path: Image reference from a view JSONL record.
    :returns: Safe relative path.
    """

    image_ref = Path(path)
    if image_ref.is_absolute():
        raise ValueError(f"image_ref must be relative: {path}")

    if ".." in image_ref.parts:
        raise ValueError(f"image_ref resolves outside image_root: {path}")

    if len(image_ref.parts) <= 1 or image_ref.parts[0] != "images":
        raise ValueError(f"image_ref must start with images/: {path}")

    return image_ref


def _validate_view_metadata(meta: ViewMetadata) -> None:
    """Validate the minimal view contract for Task 1."""

    if meta.image_path_semantics != "image_store_relative":
        raise ValueError("image_path_semantics must be image_store_relative")

    if meta.coordinate_space != "norm1000":
        raise ValueError("coordinate_space must be norm1000")

    if meta.coordinate_storage != "integer":
        raise ValueError("coordinate_storage must be integer")

    if tuple(meta.coordinate_range) != (0, 999):
        raise ValueError("coordinate_range must be (0, 999)")

    image_store = Path(meta.image_store)
    if image_store.is_absolute():
        if meta.path_anchor not in _ABSOLUTE_PATH_ANCHORS:
            raise ValueError("absolute image_store requires test/debug absolute path_anchor")
    elif meta.path_anchor != "repo_root":
        raise ValueError("relative image_store requires repo_root path_anchor")


def _infer_repo_root(view_root: Path) -> Path:
    """Infer the repository root from a path under ``public_data``."""

    resolved_view_root = view_root.resolve()
    try:
        public_data_index = resolved_view_root.parts.index("public_data")
    except ValueError as exc:
        raise ValueError("repo_root is required when view_root is not under public_data") from exc

    if public_data_index == 0:
        raise ValueError("repo_root is required for relative public_data view_root")

    return Path(*resolved_view_root.parts[:public_data_index])

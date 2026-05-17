from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence
from dataclasses import MISSING, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SCHEMA_VERSION_V1 = 1
IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE = "image_store_relative"
COORDINATE_SPACE_NORM1000 = "norm1000"
COORDINATE_STORAGE_INTEGER = "integer"
COORDINATE_RANGE_NORM1000 = (0, 999)
COORDINATE_CHART_XYXY = "xyxy"
ASSISTANT_COORDINATE_RENDERING_QWEN_COORD_TOKENS = "qwen_coord_tokens"
PROXY_ANNOTATION_POLICY_ALL_PROXY = "all_proxy"


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
    length_budget_template_id: str | None = None
    length_stats: Mapping[str, Mapping[str, str]] | None = None


_ABSOLUTE_PATH_ANCHORS = frozenset({"test_absolute", "debug_absolute"})


def load_view_metadata(path: Path) -> ViewMetadata:
    """Load view metadata from a JSON file.

    :param path: Metadata JSON path.
    :returns: Validated view metadata.
    """

    data = _load_metadata_mapping(path, context_label="view metadata")
    return _view_metadata_from_mapping(data, context=f"view metadata {path}")


def load_image_store_metadata(path: Path) -> ImageStoreMetadata:
    """Load image-store metadata from a JSON file.

    :param path: Metadata JSON path.
    :returns: Validated image-store metadata.
    """

    data = _load_metadata_mapping(path, context_label="image store metadata")
    return _image_store_metadata_from_mapping(
        data,
        context=f"image store metadata {path}",
    )


def write_view_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    """Write view metadata to a JSON file.

    :param path: Destination metadata path.
    :param metadata: JSON-serializable metadata mapping.
    """

    _view_metadata_from_mapping(metadata, context=f"view metadata {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_image_store_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    """Write image-store metadata to a JSON file.

    :param path: Destination metadata path.
    :param metadata: JSON-serializable metadata mapping.
    """

    _image_store_metadata_from_mapping(
        metadata,
        context=f"image store metadata {path}",
    )

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

    image_store = _validate_view_image_store(meta)
    if image_store.is_absolute():
        return image_store.resolve()

    resolved_repo_root = (
        repo_root if repo_root is not None else _infer_repo_root(view_root)
    ).resolve()
    resolved_image_root = (resolved_repo_root / image_store).resolve()

    try:
        resolved_image_root.relative_to(resolved_repo_root)
    except ValueError as exc:
        raise ValueError(
            f"image_store resolves outside repo_root: {meta.image_store}"
        ) from exc

    return resolved_image_root


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
    """Validate the annotation-view metadata contract."""

    if (
        type(meta.schema_version) is not int
        or meta.schema_version != SCHEMA_VERSION_V1
    ):
        raise ValueError(f"schema_version must be integer {SCHEMA_VERSION_V1}")

    if meta.kind != "annotation_view":
        raise ValueError("kind must be annotation_view")

    _require_non_empty_string(meta.dataset, field="dataset")
    _require_non_empty_string(meta.view, field="view")

    if meta.image_path_semantics != IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE:
        raise ValueError(
            "image_path_semantics must be "
            f"{IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE}"
        )

    if meta.coordinate_space != COORDINATE_SPACE_NORM1000:
        raise ValueError(f"coordinate_space must be {COORDINATE_SPACE_NORM1000}")

    if meta.coordinate_storage != COORDINATE_STORAGE_INTEGER:
        raise ValueError(f"coordinate_storage must be {COORDINATE_STORAGE_INTEGER}")

    if tuple(meta.coordinate_range) != COORDINATE_RANGE_NORM1000:
        raise ValueError(f"coordinate_range must be {COORDINATE_RANGE_NORM1000}")

    if meta.coordinate_chart != COORDINATE_CHART_XYXY:
        raise ValueError(f"coordinate_chart must be {COORDINATE_CHART_XYXY}")

    if (
        meta.assistant_coordinate_rendering
        != ASSISTANT_COORDINATE_RENDERING_QWEN_COORD_TOKENS
    ):
        raise ValueError(
            "assistant_coordinate_rendering must be "
            f"{ASSISTANT_COORDINATE_RENDERING_QWEN_COORD_TOKENS}"
        )

    _validate_view_image_store(meta)
    _validate_mapping_of_strings(
        meta.primary_jsonl,
        field="primary_jsonl",
        require_non_empty=True,
    )
    _require_mapping(meta.summary, field="summary")
    _validate_length_budget_view_metadata(meta)
    _validate_max_objects_view_metadata(meta)
    _validate_proxy_view_metadata(meta)


def _validate_view_image_store(meta: ViewMetadata) -> Path:
    """Validate and return the image store path reference."""

    raw_image_store = meta.image_store
    if raw_image_store == "":
        raise ValueError("image_store must not be empty")

    image_store = Path(raw_image_store)
    if image_store.is_absolute():
        if meta.path_anchor not in _ABSOLUTE_PATH_ANCHORS:
            raise ValueError("absolute image_store requires test/debug absolute path_anchor")

        return image_store

    if meta.path_anchor != "repo_root":
        raise ValueError("relative image_store requires repo_root path_anchor")

    posix_image_store = PurePosixPath(raw_image_store)
    if "\\" in raw_image_store:
        raise ValueError(
            f"image_store must be a POSIX-style relative path: {raw_image_store}"
        )

    if not posix_image_store.parts:
        raise ValueError("image_store must not be empty")

    if ".." in posix_image_store.parts:
        raise ValueError(
            f"image_store must not contain .. components: {raw_image_store}"
        )

    return image_store


def _infer_repo_root(view_root: Path) -> Path:
    """Infer the repository root from a path under ``public_data``."""

    resolved_view_root = view_root.resolve()
    try:
        public_data_index = (
            len(resolved_view_root.parts)
            - 1
            - resolved_view_root.parts[::-1].index("public_data")
        )
    except ValueError as exc:
        raise ValueError("repo_root is required when view_root is not under public_data") from exc

    if public_data_index == 0:
        raise ValueError("repo_root is required for relative public_data view_root")

    return Path(*resolved_view_root.parts[:public_data_index])


def _load_metadata_mapping(path: Path, *, context_label: str) -> Mapping[str, Any]:
    """Load a JSON object for metadata mapping."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, MappingABC):
        raise ValueError(f"{context_label} {path} must be a JSON object")

    return data


def _image_store_metadata_from_mapping(
    data: Mapping[str, Any],
    *,
    context: str,
) -> ImageStoreMetadata:
    """Create and validate image-store metadata from a mapping."""

    normalized = _validated_dataclass_kwargs(
        data,
        model=ImageStoreMetadata,
        context=context,
    )
    normalized["splits"] = _normalize_string_sequence(
        normalized["splits"],
        field="splits",
        context=context,
        require_non_empty=True,
    )

    metadata = ImageStoreMetadata(**normalized)
    try:
        _validate_image_store_metadata(metadata)
    except ValueError as exc:
        raise ValueError(f"{context} field {exc}") from exc

    return metadata


def _view_metadata_from_mapping(
    data: Mapping[str, Any],
    *,
    context: str,
) -> ViewMetadata:
    """Create and validate view metadata from a mapping."""

    normalized = _validated_dataclass_kwargs(
        data,
        model=ViewMetadata,
        context=context,
    )
    normalized["coordinate_range"] = _normalize_coordinate_range(
        normalized["coordinate_range"],
        context=context,
    )

    metadata = ViewMetadata(**normalized)
    try:
        _validate_view_metadata(metadata)
    except ValueError as exc:
        raise ValueError(f"{context} field {exc}") from exc

    return metadata


def _validated_dataclass_kwargs(
    data: Mapping[str, Any],
    *,
    model: type[ImageStoreMetadata] | type[ViewMetadata],
    context: str,
) -> dict[str, Any]:
    """Validate field coverage for a metadata dataclass."""

    if not isinstance(data, MappingABC):
        raise ValueError(f"{context} must be a mapping")

    field_specs = {field.name: field for field in fields(model)}
    unknown_fields = sorted(set(data) - set(field_specs))
    if unknown_fields:
        raise ValueError(
            f"{context} has unknown field {unknown_fields[0]}"
        )

    missing_fields = [
        field.name
        for field in field_specs.values()
        if field.default is MISSING
        and field.default_factory is MISSING
        and field.name not in data
    ]
    if missing_fields:
        raise ValueError(
            f"{context} is missing required field {missing_fields[0]}"
        )

    return {name: data[name] for name in data}


def _normalize_coordinate_range(
    value: Any,
    *,
    context: str,
) -> tuple[int, int]:
    """Normalize the coordinate range field."""

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(type(item) is int for item in value)
    ):
        return (value[0], value[1])

    raise ValueError(
        f"{context} field coordinate_range must contain exactly two integers"
    )


def _normalize_string_sequence(
    value: Any,
    *,
    field: str,
    context: str,
    require_non_empty: bool,
) -> tuple[str, ...]:
    """Normalize a string sequence field."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} field {field} must be a list of strings")

    normalized = tuple(value)
    if require_non_empty and not normalized:
        raise ValueError(f"{context} field {field} must not be empty")

    if any(not isinstance(item, str) or item == "" for item in normalized):
        raise ValueError(f"{context} field {field} must contain non-empty strings")

    return normalized


def _validate_image_store_metadata(meta: ImageStoreMetadata) -> None:
    """Validate the image-store metadata contract."""

    if (
        type(meta.schema_version) is not int
        or meta.schema_version != SCHEMA_VERSION_V1
    ):
        raise ValueError(f"schema_version must be integer {SCHEMA_VERSION_V1}")

    if meta.kind != "image_store":
        raise ValueError("kind must be image_store")

    _require_non_empty_string(meta.dataset, field="dataset")
    _require_non_empty_string(meta.image_store, field="image_store")
    _require_non_empty_string(meta.image_root, field="image_root")

    if meta.image_path_semantics != IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE:
        raise ValueError(
            "image_path_semantics must be "
            f"{IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE}"
        )

    _require_positive_int(meta.max_pixels, field="max_pixels")
    _require_positive_int(meta.visual_token_budget, field="visual_token_budget")
    _require_positive_int(meta.image_factor, field="image_factor")
    _normalize_string_sequence(
        meta.splits,
        field="splits",
        context="image store metadata",
        require_non_empty=True,
    )


def _validate_length_budget_view_metadata(meta: ViewMetadata) -> None:
    """Validate len-* view metadata when routed by view path."""

    if not _view_suffix_startswith(meta.view, prefix="len-"):
        return

    sample_policy = _require_mapping(meta.sample_policy, field="sample_policy")
    if sample_policy.get("type") != "length_budget":
        raise ValueError("sample_policy.type must be length_budget for len-* views")

    _require_positive_int(
        sample_policy.get("max_total_tokens"),
        field="sample_policy.max_total_tokens",
    )

    length_budget_scope = _require_mapping(
        meta.length_budget_scope,
        field="length_budget_scope",
    )
    _normalize_string_sequence(
        length_budget_scope.get("rendered_families"),
        field="length_budget_scope.rendered_families",
        context="view metadata",
        require_non_empty=True,
    )
    _require_non_empty_string(
        meta.length_budget_template_id,
        field="length_budget_template_id",
    )
    _validate_optional_length_stats(meta.length_stats)


def _validate_optional_length_stats(
    length_stats: Mapping[str, Mapping[str, str]] | None,
) -> None:
    """Validate optional length-stat file references."""

    if length_stats is None:
        return

    stats_by_split = _require_mapping(length_stats, field="length_stats")
    if not stats_by_split:
        raise ValueError("length_stats must not be empty when present")

    for split, split_stats in stats_by_split.items():
        if not isinstance(split, str) or split == "":
            raise ValueError("length_stats split names must be non-empty strings")

        split_mapping = _require_mapping(
            split_stats,
            field=f"length_stats.{split}",
        )
        _require_non_empty_string(
            split_mapping.get("filename"),
            field=f"length_stats.{split}.filename",
        )
        _require_non_empty_string(
            split_mapping.get("sha256"),
            field=f"length_stats.{split}.sha256",
        )


def _validate_max_objects_view_metadata(meta: ViewMetadata) -> None:
    """Validate max-* legacy object-cap metadata when routed by view path."""

    if not _view_suffix_startswith(meta.view, prefix="max-"):
        return

    sample_policy = _require_mapping(meta.sample_policy, field="sample_policy")
    if sample_policy.get("type") != "max_objects_legacy":
        raise ValueError(
            "sample_policy.type must be max_objects_legacy for max-* views"
        )

    if "max_objects" in sample_policy:
        _require_positive_int(
            sample_policy["max_objects"],
            field="sample_policy.max_objects",
        )
        return

    if "max_objects_per_image" in sample_policy:
        _require_positive_int(
            sample_policy["max_objects_per_image"],
            field="sample_policy.max_objects_per_image",
        )
        return

    raise ValueError(
        "sample_policy.max_objects or sample_policy.max_objects_per_image "
        "is required for max-* views"
    )


def _validate_proxy_view_metadata(meta: ViewMetadata) -> None:
    """Validate all-proxy view metadata when routed by view path."""

    if not _is_proxy_view(meta):
        return

    if meta.annotation_policy != PROXY_ANNOTATION_POLICY_ALL_PROXY:
        raise ValueError(
            f"annotation_policy must be {PROXY_ANNOTATION_POLICY_ALL_PROXY} "
            "for proxy views"
        )
    _require_non_empty_string(meta.parent_view, field="parent_view")

    proxy_policy = _require_mapping(meta.proxy_policy, field="proxy_policy")
    source_artifacts = proxy_policy.get("source_artifacts")
    if (
        not isinstance(source_artifacts, Sequence)
        or isinstance(source_artifacts, (str, bytes))
        or not source_artifacts
    ):
        raise ValueError(
            "proxy_policy.source_artifacts must be a non-empty list for proxy views"
        )
    _validate_proxy_source_artifacts(source_artifacts)

    summary = _require_mapping(meta.summary, field="summary")
    supervision_fields = (
        "object_supervision_count",
        "rendered_proxy_candidate_count",
        "support_sidecar_count",
    )
    if not any(field in summary for field in supervision_fields):
        raise ValueError(
            "summary must include object_supervision_count, "
            "rendered_proxy_candidate_count, or support_sidecar_count for proxy views"
        )
    for field in supervision_fields:
        if field in summary:
            _require_non_negative_int(summary[field], field=f"summary.{field}")


def _validate_proxy_source_artifacts(source_artifacts: Sequence[Any]) -> None:
    """Validate proxy source artifact references."""

    for index, source_artifact in enumerate(source_artifacts):
        field = f"proxy_policy.source_artifacts[{index}]"
        source_mapping = _require_mapping(source_artifact, field=field)
        _require_non_empty_string(source_mapping.get("kind"), field=f"{field}.kind")
        _validate_safe_artifact_path(
            source_mapping.get("path"),
            field=f"{field}.path",
        )


def _validate_safe_artifact_path(value: Any, *, field: str) -> None:
    """Validate a safe repo-relative POSIX artifact path."""

    _require_non_empty_string(value, field=field)
    if "\\" in value:
        raise ValueError(f"{field} must be a POSIX-style relative artifact path")

    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"{field} must be relative")

    if not path.parts:
        raise ValueError(f"{field} must not be empty")

    if ".." in path.parts:
        raise ValueError(f"{field} must not contain .. components")


def _validate_mapping_of_strings(
    value: Any,
    *,
    field: str,
    require_non_empty: bool,
) -> None:
    """Validate a mapping whose keys and values are strings."""

    mapping = _require_mapping(value, field=field)
    if require_non_empty and not mapping:
        raise ValueError(f"{field} must not be empty")

    for key, item in mapping.items():
        if not isinstance(key, str) or key == "":
            raise ValueError(f"{field} keys must be non-empty strings")
        if not isinstance(item, str) or item == "":
            raise ValueError(f"{field}.{key} must be a non-empty string")


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    """Require a mapping value."""

    if not isinstance(value, MappingABC):
        raise ValueError(f"{field} must be a mapping")

    return value


def _require_non_empty_string(value: Any, *, field: str) -> None:
    """Require a non-empty string value."""

    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a non-empty string")


def _require_positive_int(value: Any, *, field: str) -> None:
    """Require a positive integer value."""

    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _require_non_negative_int(value: Any, *, field: str) -> None:
    """Require a non-negative integer value."""

    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _view_suffix_startswith(view: str, *, prefix: str) -> bool:
    """Return whether the final view path part starts with a prefix."""

    if "\\" in view:
        raise ValueError(f"view must be a POSIX-style relative path: {view}")

    parts = PurePosixPath(view).parts
    return bool(parts and parts[-1].startswith(prefix))


def _is_proxy_view(meta: ViewMetadata) -> bool:
    """Return whether metadata describes a proxy view."""

    view_parts = PurePosixPath(meta.view).parts
    return (
        any("proxy" in part for part in view_parts)
        or meta.annotation_policy is not None
        or meta.parent_view is not None
        or meta.proxy_policy is not None
    )

#!/usr/bin/env python3
"""Lazy production adapter for the sealed S natural K/N/H cohort.

The cohort runner owns manifest/result validation.  This module owns the live
binding boundary: it validates the manifest, census, event, geometry and
operator identities before opening the HF session, then delegates execution to
the frozen S gate matrix.  The model is loaded once per executor instance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.research import run_s_natural_boundary_k_n_h_cohort as cohort  # noqa: E402
from scripts.research import run_s_primary_natural_boundary_gate as gate  # noqa: E402
from scripts.research import run_static_dynamic_owner_interface_experiment as legacy  # noqa: E402


class LiveExecutorError(RuntimeError):
    """Raised when an event cannot cross the live model-load boundary."""


ENV_MANIFEST = "S_NATURAL_BOUNDARY_MANIFEST"
ENV_CENSUS = "S_NATURAL_BOUNDARY_CENSUS"
ENV_CONFIG = "S_NATURAL_BOUNDARY_CONFIG"
ENV_PANEL = "S_NATURAL_BOUNDARY_PANEL"
ENV_COHORT = "S_NATURAL_BOUNDARY_COHORT"
ENV_COHORT_MANIFEST = "S_NATURAL_BOUNDARY_COHORT_MANIFEST"
ENV_H0_ROOT = "S_NATURAL_BOUNDARY_H0_ROOT"
ENV_H0_DIR = "S_NATURAL_BOUNDARY_H0_DIR"
ENV_PRE_GPU_RECEIPT = "S_NATURAL_BOUNDARY_PRE_GPU_RECEIPT"
ENV_SHARD_ID = "S_NATURAL_BOUNDARY_SHARD_ID"


def _regular_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise LiveExecutorError(f"{label} is not a regular non-symlink file: {path.resolve()}")
    return path.resolve(strict=True)


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise LiveExecutorError(f"{label} is not a regular non-symlink directory: {path.resolve()}")
    return path.resolve(strict=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveExecutorError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise LiveExecutorError(f"{label} must be a JSON object")
    return dict(value), raw


def _frozen_json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    """Detach a JSON identity from mutable caller-owned containers."""

    try:
        result = json.loads(cohort.canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LiveExecutorError(f"pre-GPU runtime identity is not canonical JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise LiveExecutorError("pre-GPU runtime identity must be an object")
    return result


def _observed_backend_identity(adapter: Any) -> dict[str, Any]:
    """Derive and enforce the frozen backend contract from the loaded adapter."""

    config = getattr(adapter, "config", None)
    backend = getattr(config, "backend", None)
    backend_type = getattr(backend, "type", None)
    model_config = getattr(config, "model", None)
    dtype = getattr(model_config, "dtype", None)
    hf = getattr(backend, "hf", None)
    attention = getattr(hf, "attn_implementation", None)
    generation = getattr(config, "generation", None)
    temperature = getattr(generation, "temperature", None)
    top_p = getattr(generation, "top_p", None)
    model = getattr(adapter, "model", None)
    session = getattr(adapter, "session", None)
    session_class = (
        f"{type(session).__module__}.{type(session).__qualname__}"
        if session is not None
        else None
    )
    model_training = getattr(model, "training", None)
    evidence = {
        "type": backend_type,
        "session_class": session_class,
        "model_dtype": dtype,
        "attention_implementation": attention,
        "generation": {
            "mode": "greedy" if temperature == 0.0 and top_p == 1.0 else "non_greedy",
            "temperature": temperature,
            "top_p": top_p,
        },
        "model_training": model_training,
    }
    if backend_type != "hf":
        raise LiveExecutorError("loaded resolved config backend is not hf")
    if session_class != "src.inference.hf_backend.HFBackendSession":
        raise LiveExecutorError("loaded adapter session is not HFBackendSession")
    if dtype != "fp32":
        raise LiveExecutorError("loaded resolved config model dtype is not fp32")
    if attention != "sdpa":
        raise LiveExecutorError("loaded resolved config attention backend is not sdpa")
    if temperature != 0.0 or top_p != 1.0:
        raise LiveExecutorError("loaded resolved config generation is not deterministic greedy")
    if model_training is not False:
        raise LiveExecutorError("loaded model is not in evaluation mode")
    return evidence


def _observed_model_identity(
    cpu_identity: Mapping[str, Any],
    pre_gpu_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check loaded H0 paths and retain exact inventory bindings."""

    h0 = cpu_identity.get("h0")
    input_paths = pre_gpu_identity.get("input_paths")
    input_hashes = pre_gpu_identity.get("input_hashes")
    if not isinstance(h0, Mapping) or not isinstance(input_paths, Mapping) or not isinstance(input_hashes, Mapping):
        raise LiveExecutorError("loaded/pre-GPU model identity is incomplete")
    base_model_path = h0.get("model_base_path")
    adapter_path = h0.get("adapter_path")
    embedding_delta_path = h0.get("embedding_delta_path")
    if base_model_path != input_paths.get("base_model_dir"):
        raise LiveExecutorError("loaded base-model path differs from pre-GPU inventory root")
    if h0.get("root") != input_paths.get("h0_dir"):
        raise LiveExecutorError("loaded H0 directory differs from pre-GPU identity")
    if not all(isinstance(value, str) and value for value in (adapter_path, embedding_delta_path)):
        raise LiveExecutorError("loaded adapter/embedding-delta paths are missing")
    resolved_config_sha256 = h0.get("resolved_config_sha256")
    if not isinstance(resolved_config_sha256, str):
        raise LiveExecutorError("loaded H0 resolved-config hash is missing")
    return {
        "checkpoint": "S",
        "step": cohort.STEP,
        "substrate": cohort.SUBSTRATE,
        "h0_dir": h0.get("root"),
        "base_model_path": base_model_path,
        "adapter_path": adapter_path,
        "embedding_delta_path": embedding_delta_path,
        "resolved_config_sha256": resolved_config_sha256,
        "h0_identity_files_sha256": input_hashes.get("h0_identity_files_sha256"),
        "base_model_files_sha256": input_hashes.get("base_model_files_sha256"),
        "base_model_inventory_sha256": input_hashes.get("base_model_inventory_sha256"),
    }


def _executor_identity(
    pre_gpu_identity: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    backend: Any,
    device: Any,
    cuda: Mapping[str, Any],
    config_sha256: Any,
    runtime_versions: Mapping[str, str],
    full_runtime_cohort_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind exact pre-GPU authority and observed post-load evidence."""

    try:
        preflight = cohort._validate_full_runtime_cohort_preflight(  # noqa: SLF001 - shared output contract
            full_runtime_cohort_preflight,
            "full_runtime_cohort_preflight",
        )
    except cohort.CohortContractError as exc:
        raise LiveExecutorError(f"full runtime cohort preflight is invalid: {exc}") from exc

    observed = {
        "model": dict(model),
        "backend": backend,
        "device": str(device),
        "cuda": dict(cuda),
        "config_sha256": config_sha256,
        "runtime_versions": dict(runtime_versions),
    }
    body = {
        "schema_version": "s_natural_boundary_k_n_h_executor_identity.v1",
        "pre_gpu": dict(pre_gpu_identity),
        "observed": observed,
        "full_runtime_cohort_preflight": preflight,
    }
    body["identity_sha256"] = cohort.sha256_json(body)
    return body


def _legacy_event_indices(event: Mapping[str, Any]) -> tuple[int | None, int | None]:
    source = event.get("source_panel_object_index")
    panel = event.get("panel_identity")
    derived = panel.get("derived_panel_object_index") if isinstance(panel, Mapping) else None
    return (
        int(source) if isinstance(source, int) and not isinstance(source, bool) else None,
        int(derived) if isinstance(derived, int) and not isinstance(derived, bool) else None,
    )


def _is_exact_legacy_s_event(candidate: Mapping[str, Any], runtime_event: Mapping[str, Any]) -> bool:
    owner_refs = runtime_event.get("owner_refs")
    if not isinstance(owner_refs, Mapping):
        return False
    source, derived = _legacy_event_indices(candidate)
    geometry = candidate.get("geometry_by_checkpoint")
    checkpoint_status = candidate.get("checkpoint_status")
    return bool(
        str(candidate.get("gt_owner_id")) == str(owner_refs.get("gt_owner_id"))
        and candidate.get("image_id") == runtime_event.get("image_id")
        and source == owner_refs.get("source_panel_object_index")
        and derived == owner_refs.get("derived_panel_object_index")
        and isinstance(geometry, Mapping)
        and isinstance(geometry.get("S"), Mapping)
        and geometry["S"].get("checkpoint") == "S"
        and isinstance(checkpoint_status, Mapping)
        and isinstance(checkpoint_status.get("S"), Mapping)
        and checkpoint_status["S"].get("disposition") == "established"
    )


def _event_id(event: Mapping[str, Any]) -> str:
    value = event.get("event_id")
    if not isinstance(value, str) or not value:
        raise LiveExecutorError("event.event_id must be a non-empty string")
    return value


def _prefix(event: Mapping[str, Any]) -> tuple[int, ...]:
    boundary = event.get("natural_boundary")
    if not isinstance(boundary, Mapping):
        raise LiveExecutorError("event.natural_boundary is missing")
    values = boundary.get("prefix_token_ids")
    if not isinstance(values, list) or not values or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise LiveExecutorError("event natural prefix_token_ids are malformed")
    return tuple(int(v) for v in values)


def _regions_from_row(row: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    candidates: Any = row.get("image_cell_regions")
    geometry = row.get("geometry")
    if candidates is None and isinstance(geometry, Mapping):
        candidates = geometry.get("image_cell_regions") or geometry.get("regions") or geometry
    if not isinstance(candidates, Mapping):
        raise LiveExecutorError("census row lacks image-cell geometry regions")
    result: dict[str, tuple[int, ...]] = {}
    for key in ("a_exclusive", "b_exclusive", "background", "shared_core"):
        values = candidates.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise LiveExecutorError(f"census geometry region {key!r} is missing")
        normalized = tuple(int(v) for v in values if not isinstance(v, bool) and isinstance(v, int) and v >= 0)
        if len(normalized) != len(values) or len(set(normalized)) != len(normalized):
            raise LiveExecutorError(f"census geometry region {key!r} is malformed")
        result[key] = normalized
    competitor = candidates.get("same_class_competitor", ())
    if not isinstance(competitor, Sequence) or isinstance(competitor, (str, bytes)):
        raise LiveExecutorError("census same_class_competitor region is malformed")
    normalized_competitor = tuple(int(v) for v in competitor if not isinstance(v, bool) and isinstance(v, int) and v >= 0)
    if len(normalized_competitor) != len(competitor) or len(set(normalized_competitor)) != len(normalized_competitor):
        raise LiveExecutorError("census same_class_competitor region is malformed")
    result["same_class_competitor"] = normalized_competitor
    return result


def _normalize_h0_image_plan(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("status") != "ok" or row.get("error") is not None:
        raise LiveExecutorError("H0 image plan row is not successful")
    observed = row.get("observed_image_grid_thw")
    expected = row.get("expected_image_grid_thw")
    if (
        not isinstance(observed, list)
        or len(observed) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in observed)
        or expected != observed
    ):
        raise LiveExecutorError("H0 image plan grid identity is invalid")
    merge_size = row.get("merge_size")
    merged_tokens = row.get("merged_visual_tokens")
    if (
        isinstance(merge_size, bool)
        or not isinstance(merge_size, int)
        or merge_size <= 0
        or observed[1] % merge_size
        or observed[2] % merge_size
    ):
        raise LiveExecutorError("H0 image plan merge identity is invalid")
    grid_rows = observed[1] // merge_size
    grid_cols = observed[2] // merge_size
    expected_tokens = observed[0] * grid_rows * grid_cols
    if merged_tokens != expected_tokens or row.get("raw_patch_rows") != math.prod(observed):
        raise LiveExecutorError("H0 image plan token/cell count is invalid")
    width = row.get("decoded_width", row.get("declared_width"))
    height = row.get("decoded_height", row.get("declared_height"))
    if (
        isinstance(width, bool)
        or not isinstance(width, (int, float))
        or not math.isfinite(float(width))
        or float(width) <= 0
        or isinstance(height, bool)
        or not isinstance(height, (int, float))
        or not math.isfinite(float(height))
        or float(height) <= 0
    ):
        raise LiveExecutorError("H0 image plan dimensions are invalid")
    return {
        "observed_image_grid_thw": list(observed),
        "grid_thw": list(observed),
        "merge_size": merge_size,
        "premerge_grid_rows": observed[1],
        "premerge_grid_cols": observed[2],
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "merged_visual_tokens": merged_tokens,
        "cell_count": expected_tokens,
        "image_width": float(width),
        "image_height": float(height),
    }


def _load_h0_image_plans(
    path: Path,
    derived_values: Sequence[Any],
) -> dict[str, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise LiveExecutorError(f"H0 image plan is not a regular file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                raise LiveExecutorError(f"H0 image plan row {line_number} is not an object")
            rows.append(dict(value))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveExecutorError(f"cannot read H0 image plan: {exc}") from exc
    if len(rows) != len(derived_values):
        raise LiveExecutorError("H0 image plan row count differs from the derived panel")
    result: dict[str, dict[str, Any]] = {}
    for index, (row, derived) in enumerate(zip(rows, derived_values, strict=True)):
        if row.get("row_index") != index:
            raise LiveExecutorError("H0 image plan row indices are not exact input order")
        image_id = legacy._raw_image_id(derived)
        if image_id in result:
            raise LiveExecutorError("H0 image plan repeats an image identity")
        result[image_id] = _normalize_h0_image_plan(row)
    return result


def _raw_owner_exact_descriptor(raw: Any, obj: Any, index: int) -> dict[str, Any]:
    """Derive an owner identity and unrounded pixel box from one raw object."""

    image_id = legacy._raw_image_id(raw)
    try:
        width, height = legacy._raw_dimensions(raw)
        bins = legacy._raw_coordinate_bins(raw, obj)
    except Exception as exc:
        raise LiveExecutorError(
            f"authoritative panel image {image_id} object {index} lacks valid dimensions/coordinate bins"
        ) from exc
    if bins is None:
        raise LiveExecutorError(
            f"authoritative panel image {image_id} object {index} lacks a valid four-coordinate box"
        )
    exact_bbox = tuple(
        float(value * (width if axis % 2 == 0 else height) / 1000.0)
        for axis, value in enumerate(bins)
    )
    if not all(math.isfinite(value) for value in exact_bbox) or exact_bbox[2] < exact_bbox[0] or exact_bbox[3] < exact_bbox[1]:
        raise LiveExecutorError(
            f"authoritative panel image {image_id} object {index} exact raw box is malformed"
        )
    category = str(
        legacy._raw_field(
            obj,
            "description",
            legacy._raw_field(
                obj,
                "desc",
                legacy._raw_field(obj, "category_name", legacy._raw_field(obj, "category", "")),
            ),
        )
    ).strip().lower()
    if not category:
        raise LiveExecutorError(f"authoritative panel image {image_id} object {index} lacks a category")
    coco_ann_id = legacy._raw_field(obj, "coco_ann_id")
    if coco_ann_id is None:
        coco_ann_id = legacy._raw_field(obj, "object_id")
    if isinstance(coco_ann_id, bool):
        raise LiveExecutorError(f"authoritative panel image {image_id} object {index} has an invalid coco_ann_id")
    return {
        "owner_id": f"gt:{image_id}:{index}",
        "source_index": index,
        "category": category,
        "coco_ann_id": coco_ann_id,
        "exact_bbox": exact_bbox,
        "rounded_bbox": tuple(int(round(value)) for value in exact_bbox),
    }


def _ephemeral_exact_raw_owner_mapping(
    source_raw: Any,
    derived_raw: Any,
    owner_mapping: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Cross-check legacy identities and add only validated raw boxes.

    The materializer binds each source owner to a derived index and then uses
    that derived object's raw coordinate-bin projection.  This helper rebuilds
    both projections independently, verifies all identity and historical
    rounded-box fields in the legacy mapping, and returns a detached mapping
    for the geometry validator.  It intentionally does not mutate the legacy
    mapping or make the rounded box usable for fractional weights.
    """

    source_image_id = legacy._raw_image_id(source_raw)
    derived_image_id = legacy._raw_image_id(derived_raw)
    if source_image_id != derived_image_id:
        raise LiveExecutorError(
            f"authoritative source/derived image identity mismatch: {source_image_id} != {derived_image_id}"
        )
    source_objects = legacy._raw_objects(source_raw)
    derived_objects = legacy._raw_objects(derived_raw)
    if len(source_objects) != len(derived_objects):
        raise LiveExecutorError(
            f"authoritative source/derived owner count mismatch for image {source_image_id}"
        )
    source_descriptors = [
        _raw_owner_exact_descriptor(source_raw, obj, index)
        for index, obj in enumerate(source_objects)
    ]
    derived_descriptors = [
        _raw_owner_exact_descriptor(derived_raw, obj, index)
        for index, obj in enumerate(derived_objects)
    ]
    expected_owner_ids = {descriptor["owner_id"] for descriptor in source_descriptors}
    if set(owner_mapping) != expected_owner_ids:
        raise LiveExecutorError("authoritative owner mapping owner-id set drifted")
    result: dict[str, dict[str, Any]] = {}
    for source in source_descriptors:
        owner_id = source["owner_id"]
        mapping = owner_mapping.get(owner_id)
        if not isinstance(mapping, Mapping):
            raise LiveExecutorError(f"authoritative owner mapping is missing {owner_id}")
        source_index = source["source_index"]
        derived_index = mapping.get("derived_index")
        if (
            mapping.get("owner_id") != owner_id
            or mapping.get("source_index") != source_index
            or isinstance(derived_index, bool)
            or not isinstance(derived_index, int)
            or not 0 <= derived_index < len(derived_descriptors)
            or mapping.get("category") != source["category"]
            or mapping.get("coco_ann_id") != source["coco_ann_id"]
            or mapping.get("source_pixel_bbox") != list(source["rounded_bbox"])
        ):
            raise LiveExecutorError(f"authoritative owner mapping identity drifted for {owner_id}")
        derived = derived_descriptors[derived_index]
        if (
            derived["owner_id"] != f"gt:{source_image_id}:{derived_index}"
            or derived["category"] != source["category"]
            or derived["coco_ann_id"] != source["coco_ann_id"]
            or mapping.get("derived_pixel_bbox") != list(derived["rounded_bbox"])
            or tuple(source["exact_bbox"]) != tuple(derived["exact_bbox"])
        ):
            raise LiveExecutorError(f"authoritative source/derived raw identity drifted for {owner_id}")
        result[owner_id] = {
            **dict(mapping),
            "source_pixel_bbox_exact_raw": list(derived["exact_bbox"]),
        }
    return result


def _mapping_cell_weights(
    mapping: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[int, float]:
    # Geometry receipts are materialized from the unrounded coordinate-bin
    # projection.  ``source_pixel_bbox`` is retained as a historical identity
    # only; using it here silently changes fractional cell weights.
    bbox = mapping.get("source_pixel_bbox_exact_raw")
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
    ):
        raise LiveExecutorError("authoritative panel mapping lacks an exact raw four-coordinate box")
    try:
        exact_bbox = tuple(float(value) for value in bbox)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LiveExecutorError("authoritative panel mapping exact raw box is malformed") from exc
    if not all(math.isfinite(value) for value in exact_bbox):
        raise LiveExecutorError("authoritative panel mapping exact raw box is not finite")
    x1, y1, x2, y2 = exact_bbox
    width = float(plan["image_width"])
    height = float(plan["image_height"])
    rows = int(plan["grid_rows"])
    cols = int(plan["grid_cols"])
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return {}
    cell_width = width / cols
    cell_height = height / rows
    cell_area = cell_width * cell_height
    result: dict[int, float] = {}
    for row in range(rows):
        overlap_y = max(0.0, min(y2, (row + 1) * cell_height) - max(y1, row * cell_height))
        if overlap_y <= 0.0:
            continue
        for col in range(cols):
            overlap_x = max(0.0, min(x2, (col + 1) * cell_width) - max(x1, col * cell_width))
            if overlap_x > 0.0:
                result[row * cols + col] = float(overlap_x * overlap_y / cell_area)
    return result


def _weight_pairs(weights: Mapping[int, float]) -> list[dict[str, Any]]:
    return [
        {"cell_index": index, "overlap_fraction": float(weight)}
        for index, weight in sorted(weights.items())
    ]


def _validate_frozen_geometry(
    geometry: Mapping[str, Any],
    *,
    h0_image_plan: Mapping[str, Any],
    owner_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the launch geometry independently before model load."""

    supplied_hash = geometry.get("geometry_sha256")
    hash_body = dict(geometry)
    hash_body.pop("geometry_sha256", None)
    if supplied_hash != cohort.sha256_json(hash_body):
        raise LiveExecutorError("runtime-cohort geometry_sha256 mismatch")
    if geometry.get("image_plan_identity") != dict(h0_image_plan):
        raise LiveExecutorError("runtime-cohort geometry differs from the bound H0 image plan")
    regions = geometry.get("image_cell_regions")
    receipts = geometry.get("image_cell_region_receipts")
    region_keys = {
        "a_exclusive", "b_exclusive", "shared_core", "background",
        "same_class_competitor",
    }
    if not isinstance(regions, Mapping) or set(regions) != region_keys:
        raise LiveExecutorError("runtime-cohort geometry region keyset drifted")
    if not isinstance(receipts, Mapping) or set(receipts) != region_keys:
        raise LiveExecutorError("runtime-cohort geometry receipt keyset drifted")
    cell_count = h0_image_plan.get("cell_count")
    if isinstance(cell_count, bool) or not isinstance(cell_count, int) or cell_count <= 0:
        raise LiveExecutorError("runtime-cohort geometry cell count is invalid")
    normalized: dict[str, list[int]] = {}
    for key in sorted(region_keys):
        values = regions[key]
        if (
            not isinstance(values, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < cell_count for value in values)
            or len(set(values)) != len(values)
            or values != sorted(values)
        ):
            raise LiveExecutorError(f"runtime-cohort geometry region {key} is invalid")
        normalized[key] = list(values)
        receipt = receipts[key]
        if not isinstance(receipt, Mapping):
            raise LiveExecutorError(f"runtime-cohort geometry receipt {key} is missing")
        weights = receipt.get("fractional_weights")
        if not isinstance(weights, list) or any(not isinstance(item, Mapping) for item in weights):
            raise LiveExecutorError(f"runtime-cohort geometry weights {key} are missing")
        weight_indices: list[int] = []
        weight_values: list[float] = []
        for item in weights:
            index = item.get("cell_index")
            weight = item.get("overlap_fraction")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or not 0.0 <= float(weight) <= 1.0
            ):
                raise LiveExecutorError(f"runtime-cohort geometry weight {key} is invalid")
            weight_indices.append(index)
            weight_values.append(float(weight))
        if (
            receipt.get("cell_indices") != values
            or receipt.get("visual_indices") != values
            or receipt.get("cell_count") != len(values)
            or weight_indices != values
            or receipt.get("cell_indices_sha256") != cohort.sha256_json(values)
            or receipt.get("weights_sha256") != cohort.sha256_json(weights)
            or not isinstance(receipt.get("weight_sum"), (int, float))
            or not math.isclose(float(receipt["weight_sum"]), sum(weight_values), rel_tol=0.0, abs_tol=1e-12)
        ):
            raise LiveExecutorError(f"runtime-cohort geometry receipt {key} drifted")
    for index, key in enumerate(sorted(region_keys)):
        for other in sorted(region_keys)[index + 1:]:
            if set(normalized[key]) & set(normalized[other]):
                raise LiveExecutorError("runtime-cohort geometry regions overlap")
    required_available = all(normalized[key] for key in ("a_exclusive", "b_exclusive", "background"))
    if geometry.get("launch_eligible") is not required_available:
        raise LiveExecutorError("runtime-cohort geometry launch eligibility drifted")
    for key in ("a_exclusive", "b_exclusive", "background"):
        if receipts[key].get("available") is not bool(normalized[key]):
            raise LiveExecutorError(f"runtime-cohort geometry availability {key} drifted")
    target_owner = str(geometry.get("target_owner_id"))
    covered = geometry.get("covered_owner_ids_at_b_boundary")
    owner_regions = geometry.get("owner_regions")
    region_owner_ids = geometry.get("owner_region_owner_ids")
    if (
        target_owner not in owner_mapping
        or not isinstance(covered, list)
        or not isinstance(owner_regions, Mapping)
        or not isinstance(region_owner_ids, list)
        or set(owner_regions) != set(region_owner_ids)
        or any(owner_id not in owner_mapping for owner_id in region_owner_ids)
    ):
        raise LiveExecutorError("runtime-cohort authoritative owner-region identities drifted")
    a_owner = next(
        (owner_id for owner_id in region_owner_ids if geometry.get("owner_region_roles", {}).get(owner_id) == "covered-A"),
        None,
    )
    if not isinstance(a_owner, str):
        raise LiveExecutorError("runtime-cohort covered-A geometry owner is missing")
    authoritative_weights = {
        owner_id: _mapping_cell_weights(owner_mapping[owner_id], h0_image_plan)
        for owner_id in region_owner_ids
    }
    expected_exclusive: dict[str, dict[int, float]] = {}
    for owner_id, weights in authoritative_weights.items():
        expected_exclusive[owner_id] = {
            index: weight
            for index, weight in weights.items()
            if all(index not in other for other_id, other in authoritative_weights.items() if other_id != owner_id)
        }
        owner_region = owner_regions[owner_id]
        expected_cells = sorted(expected_exclusive[owner_id])
        if (
            not isinstance(owner_region, Mapping)
            or owner_region.get("cells") != expected_cells
            or owner_region.get("exclusive") != expected_cells
            or owner_region.get("support") != sorted(weights)
            or owner_region.get("exclusive_available") is not bool(expected_cells)
            or owner_region.get("fractional_weights", {}).get("fractional_weights")
            != _weight_pairs(expected_exclusive[owner_id])
        ):
            raise LiveExecutorError(
                f"runtime-cohort owner geometry does not match the exact raw pixel box for {owner_id}"
            )
    a_weights = authoritative_weights[a_owner]
    b_weights = authoritative_weights[target_owner]
    expected_shared = {index: a_weights[index] for index in a_weights if index in b_weights}
    all_gt_weights = {
        owner_id: _mapping_cell_weights(mapping, h0_image_plan)
        for owner_id, mapping in owner_mapping.items()
    }
    occupied = {index for weights in all_gt_weights.values() for index in weights}
    background_count = len(expected_exclusive[target_owner])
    expected_background = [
        index for index in range(cell_count) if index not in occupied
    ][:background_count]
    expected_pair_regions = {
        "a_exclusive": sorted(expected_exclusive[a_owner]),
        "b_exclusive": sorted(expected_exclusive[target_owner]),
        "shared_core": sorted(expected_shared),
        "background": expected_background,
    }
    if any(normalized[key] != values for key, values in expected_pair_regions.items()):
        raise LiveExecutorError(
            "runtime-cohort pair/background cells do not match the exact raw pixel box"
        )
    expected_pair_weights = {
        "a_exclusive": expected_exclusive[a_owner],
        "b_exclusive": expected_exclusive[target_owner],
        "shared_core": expected_shared,
        "background": {index: 0.0 for index in expected_background},
    }
    for key, weights in expected_pair_weights.items():
        if receipts[key].get("fractional_weights") != _weight_pairs(weights):
            raise LiveExecutorError(
                f"runtime-cohort region receipt {key} does not match the exact raw pixel box"
            )
    if (
        geometry.get("status") != "available"
        or geometry.get("mechanical_disposition") != "eligible_verified_pair_regions"
        or geometry.get("reason") is not None
    ):
        raise LiveExecutorError("runtime-cohort launch geometry disposition drifted")
    competitor_owner = geometry.get("same_class_competitor_owner_id")
    competitor_cells = normalized["same_class_competitor"]
    competitor_receipt = receipts["same_class_competitor"]
    if competitor_owner is None:
        if competitor_cells or competitor_receipt.get("available") is not False or competitor_receipt.get("status") != "not_applicable":
            raise LiveExecutorError("runtime-cohort null competitor geometry drifted")
    else:
        competitor_mapping = owner_mapping.get(str(competitor_owner))
        target_mapping = owner_mapping.get(target_owner)
        competitor_region = owner_regions.get(str(competitor_owner)) if isinstance(owner_regions, Mapping) else None
        covered = geometry.get("covered_owner_ids_at_b_boundary")
        if (
            not isinstance(competitor_mapping, Mapping)
            or not isinstance(target_mapping, Mapping)
            or competitor_mapping.get("category") != target_mapping.get("category")
            or competitor_owner in (covered if isinstance(covered, list) else [])
            or not isinstance(competitor_region, Mapping)
            or competitor_region.get("cells") != competitor_cells
            or competitor_cells != sorted(expected_exclusive[str(competitor_owner)])
            or competitor_receipt.get("fractional_weights")
            != _weight_pairs(expected_exclusive[str(competitor_owner)])
            or not competitor_cells
            or competitor_receipt.get("available") is not True
        ):
            raise LiveExecutorError("runtime-cohort same-class competitor geometry drifted")
    return {
        "geometry_sha256": supplied_hash,
        "image_plan_identity": dict(h0_image_plan),
        "regions_sha256": cohort.sha256_json(normalized),
    }


def _runtime_event(event: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    """Enrich the immutable manifest event with independently bound geometry."""
    result = dict(event)
    owner_refs = event.get("owner_refs")
    owner = owner_refs.get("gt_owner_id") if isinstance(owner_refs, Mapping) else None
    if not isinstance(owner, str) or not owner:
        raise LiveExecutorError("event owner_refs.gt_owner_id is missing")
    if str(row.get("gt_owner_id")) != owner or int(row.get("image_id", -1)) != int(event.get("image_id", -2)):
        raise LiveExecutorError("event owner/image identity differs from census row")
    event_source_index = owner_refs.get("source_panel_object_index") if isinstance(owner_refs, Mapping) else None
    row_source_index = row.get("source_panel_object_index")
    if isinstance(event_source_index, bool) or not isinstance(event_source_index, int) or isinstance(row_source_index, bool) or not isinstance(row_source_index, int) or int(event_source_index) != int(row_source_index):
        raise LiveExecutorError("event source-panel object identity differs from census row")
    event_derived_index = owner_refs.get("derived_panel_object_index") if isinstance(owner_refs, Mapping) else None
    row_derived_index = row.get("derived_panel_object_index")
    if isinstance(event_derived_index, bool) or not isinstance(event_derived_index, int) or isinstance(row_derived_index, bool) or not isinstance(row_derived_index, int) or int(event_derived_index) != int(row_derived_index):
        raise LiveExecutorError("event derived-panel object identity differs from census row")
    regions = _regions_from_row(row)
    row_prefix = row.get("exact_prefix_token_ids")
    if not isinstance(row_prefix, list) or tuple(int(v) for v in row_prefix) != _prefix(event):
        raise LiveExecutorError("admitted natural prefix differs from census exact_prefix_token_ids")
    row_prefix_sha = row.get("exact_prefix_sha256")
    if not isinstance(row_prefix_sha, str) or row_prefix_sha != cohort.sha256_json(list(_prefix(event))):
        raise LiveExecutorError("admitted natural prefix hash differs from census")
    result["gt_owner_id"] = owner
    result["source_panel_object_index"] = row.get("source_panel_object_index", event.get("owner_refs", {}).get("source_panel_object_index"))
    result["image_cell_regions"] = {key: list(values) for key, values in regions.items()}
    geometry = row.get("geometry") if isinstance(row.get("geometry"), Mapping) else {}
    launch = geometry.get("launch_eligible", row.get("geometry_launch_eligible"))
    if launch is not True:
        raise LiveExecutorError("census geometry is not launch-eligible")
    plan = geometry.get("image_plan_identity") if isinstance(geometry, Mapping) else None
    if not isinstance(plan, Mapping) or not isinstance(plan.get("cell_count"), int) or int(plan["cell_count"]) <= 0:
        raise LiveExecutorError("census geometry lacks exact image_plan_identity")
    if "same_class_competitor_owner_id" not in row:
        raise LiveExecutorError("census geometry lacks same_class_competitor_owner_id disposition")
    competitor_owner = row.get("same_class_competitor_owner_id")
    if geometry.get("same_class_competitor_owner_id") != competitor_owner:
        raise LiveExecutorError("census row and geometry competitor-owner identities differ")
    if regions["same_class_competitor"] and not isinstance(competitor_owner, str):
        raise LiveExecutorError("census competitor region lacks owner identity")
    result["geometry_by_checkpoint"] = {"S": dict(geometry)}
    return result


def _validate_authoritative_event_binding(
    event: Mapping[str, Any],
    row: Mapping[str, Any],
    legacy_event: Mapping[str, Any],
    *,
    h0_records: Mapping[str, Any],
    owner_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check one admitted B/A pair against CPU-loaded H0 and panels."""

    owner_refs = event.get("owner_refs")
    if not isinstance(owner_refs, Mapping):
        raise LiveExecutorError("runtime-cohort event owner_refs are missing")
    target_owner = owner_refs.get("gt_owner_id")
    a_owner = owner_refs.get("covered_A_owner_id")
    if not isinstance(target_owner, str) or not isinstance(a_owner, str):
        raise LiveExecutorError("runtime-cohort admitted B/A owner identities are missing")
    b_record = h0_records.get(target_owner)
    a_record = h0_records.get(a_owner)
    if not isinstance(b_record, Mapping) or not isinstance(a_record, Mapping):
        raise LiveExecutorError(f"runtime-cohort authoritative H0 B/A records are missing for {target_owner}")
    prefix_ids = list(_prefix(event))
    prefix_sha = event.get("natural_boundary", {}).get("prefix_sha256")
    covered_owner_ids = list(owner_refs.get("covered_owner_ids", []))
    if (
        list(b_record.get("exact_prefix_token_ids", ())) != prefix_ids
        or b_record.get("exact_prefix_sha256") != prefix_sha
        or cohort.sha256_json(prefix_ids) != prefix_sha
        or b_record.get("natural_boundary") != row.get("natural_boundary")
        or list(b_record.get("covered_owner_ids", ())) != covered_owner_ids
        or list(row.get("covered_owner_ids", [])) != covered_owner_ids
        or b_record.get("record_index") != row.get("h0_record_index")
    ):
        raise LiveExecutorError(f"runtime-cohort authoritative target-B H0 binding drifted for {target_owner}")
    if (
        a_record.get("natural_boundary") != row.get("covered_A_natural_boundary")
        or a_record.get("exact_prefix_sha256")
        != cohort.sha256_json(list(a_record.get("exact_prefix_token_ids", ())))
        or a_record.get("strict_complete_row") is not True
        or b_record.get("latest_covered_owner_id") != a_owner
    ):
        raise LiveExecutorError(f"runtime-cohort authoritative covered-A H0 binding drifted for {target_owner}")
    mapping = owner_mapping.get(target_owner)
    a_mapping = owner_mapping.get(a_owner)
    panel_identity = legacy_event.get("panel_identity")
    pair = legacy_event.get("A_B", {}).get("S") if isinstance(legacy_event.get("A_B"), Mapping) else None
    a_context = pair.get("A_latest_covered") if isinstance(pair, Mapping) else None
    geometry = legacy_event.get("geometry_by_checkpoint", {}).get("S") if isinstance(legacy_event.get("geometry_by_checkpoint"), Mapping) else None
    owner_sources = geometry.get("owner_sources") if isinstance(geometry, Mapping) else None
    a_source = owner_sources.get(a_owner) if isinstance(owner_sources, Mapping) else None
    if (
        not isinstance(mapping, Mapping)
        or not isinstance(a_mapping, Mapping)
        or not isinstance(panel_identity, Mapping)
        or not isinstance(a_context, Mapping)
        or not isinstance(a_source, Mapping)
    ):
        raise LiveExecutorError(f"runtime-cohort authoritative target panel identity is missing for {target_owner}")
    expected_panel = {
        "source_panel_object_index": mapping.get("source_index"),
        "derived_panel_object_index": mapping.get("derived_index"),
        "coco_ann_id": mapping.get("coco_ann_id"),
    }
    if (
        owner_refs.get("source_panel_object_index") != expected_panel["source_panel_object_index"]
        or owner_refs.get("derived_panel_object_index") != expected_panel["derived_panel_object_index"]
        or panel_identity.get("source_panel_object_index") != expected_panel["source_panel_object_index"]
        or panel_identity.get("derived_panel_object_index") != expected_panel["derived_panel_object_index"]
        or str(panel_identity.get("coco_ann_id")) != str(expected_panel["coco_ann_id"])
    ):
        raise LiveExecutorError(f"runtime-cohort authoritative target panel mapping drifted for {target_owner}")
    expected_a_panel = {
        "source_panel_object_index": a_mapping.get("source_index"),
        "derived_panel_object_index": a_mapping.get("derived_index"),
        "coco_ann_id": a_mapping.get("coco_ann_id"),
    }
    if (
        a_context.get("source_panel_object_index") != expected_a_panel["source_panel_object_index"]
        or a_source.get("source_panel_object_index") != expected_a_panel["source_panel_object_index"]
        or a_source.get("derived_panel_object_index") != expected_a_panel["derived_panel_object_index"]
        or str(a_source.get("coco_ann_id")) != str(expected_a_panel["coco_ann_id"])
    ):
        raise LiveExecutorError(f"runtime-cohort authoritative covered-A panel mapping drifted for {target_owner}")
    return {
        "target_owner": target_owner,
        "covered_A_owner": a_owner,
        "target_prefix_sha256": prefix_sha,
        "covered_A_prefix_sha256": a_record["exact_prefix_sha256"],
        "covered_owner_ids": covered_owner_ids,
        "h0_record_index": b_record["record_index"],
        "panel_identity": expected_panel,
        "covered_A_panel_identity": expected_a_panel,
    }


def _ledger_exact_history_resolver(
    adapter: Any,
    event: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Resolve one B/A history from immutable H0 trace rows and ledger data.

    The cohort's natural boundary is an owner-decision boundary, not a count
    of physical generated rows.  This resolver is therefore the only live
    path that may project an admitted B history onto the physical H0 trace.
    Every projection is checked against both the flattened generated stream
    and the row-boundary provenance before it reaches the model context.
    """

    checkpoint = str(getattr(adapter, "checkpoint", ""))
    if checkpoint not in {"S", "A"}:
        raise LiveExecutorError("ledger-exact history resolver requires checkpoint S or A")
    target_owner, eligible, _pair_status, pair = legacy._validate_active_event_receipts(
        event,
        checkpoint=checkpoint,
    )
    if not eligible:
        raise LiveExecutorError(
            "ledger-exact history resolver requires an actuator-eligible verified pair"
        )
    a_owner = pair.get("_a_owner")
    if not isinstance(a_owner, str) or not a_owner:
        raise LiveExecutorError("ledger-exact history resolver lacks covered-A owner identity")
    image_id = str(event.get("image_id"))
    panel_rows = getattr(adapter, "panel_rows", {})
    runtime = panel_rows.get(image_id) if isinstance(panel_rows, Mapping) else None
    h0_rows = getattr(adapter, "h0_rows", {})
    trace = h0_rows.get(image_id) if isinstance(h0_rows, Mapping) else None
    if trace is None and runtime is not None:
        candidate_trace = getattr(runtime, "h0", None)
        if isinstance(candidate_trace, Mapping):
            trace = candidate_trace
    ledger_root = getattr(adapter, "h0_ledger_records", {})
    image_ledger = ledger_root.get(image_id) if isinstance(ledger_root, Mapping) else None
    if runtime is None or not isinstance(trace, Mapping) or not isinstance(image_ledger, Mapping):
        raise LiveExecutorError(
            f"ledger-exact history resolver lacks immutable runtime/H0 ledger binding for {image_id}/{target_owner}"
        )
    b_record = image_ledger.get(target_owner)
    a_record = image_ledger.get(a_owner)
    if not isinstance(b_record, Mapping) or not isinstance(a_record, Mapping):
        raise LiveExecutorError(
            f"ledger-exact history resolver lacks B/A H0 records for {image_id}/{target_owner}"
        )

    def token_ids(value: Any, *, label: str, allow_empty: bool = False) -> tuple[int, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise LiveExecutorError(f"{label} must be an integer token sequence")
        result: list[int] = []
        for token in value:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise LiveExecutorError(f"{label} contains an invalid token ID")
            result.append(int(token))
        if not result and not allow_empty:
            raise LiveExecutorError(f"{label} must not be empty")
        return tuple(result)

    generated = token_ids(
        trace.get("generated_token_ids"),
        label=f"H0 generated_token_ids for {image_id}",
    )
    raw_rows = trace.get("rows")
    if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence) or not raw_rows:
        raise LiveExecutorError(f"H0 physical rows are missing for {image_id}")
    rows = tuple(
        token_ids(row, label=f"H0 physical row {index} for {image_id}")
        for index, row in enumerate(raw_rows)
    )
    flattened = tuple(token for row in rows for token in row)
    if flattened != generated:
        raise LiveExecutorError(
            f"H0 physical rows are not the literal generated_token_ids stream for {image_id}"
        )
    generated_sha = trace.get("generated_token_ids_sha256")
    if generated_sha is not None and generated_sha != legacy.sha256_token_ids(generated):
        raise LiveExecutorError(f"H0 generated-token hash drifted for {image_id}")
    row_hashes = trace.get("row_token_ids_sha256")
    if row_hashes is not None:
        if not isinstance(row_hashes, Sequence) or len(row_hashes) != len(rows):
            raise LiveExecutorError(f"H0 physical row hash inventory is malformed for {image_id}")
        expected_hashes = [legacy.sha256_token_ids(row) for row in rows]
        if list(row_hashes) != expected_hashes:
            raise LiveExecutorError(f"H0 physical row hash inventory drifted for {image_id}")

    def record_prefix(
        record: Mapping[str, Any],
        label: str,
        *,
        allow_empty: bool = False,
    ) -> tuple[int, ...]:
        values = token_ids(
            record.get("exact_prefix_token_ids"),
            label=f"{label} exact history",
            allow_empty=allow_empty,
        )
        declared_sha = record.get("exact_prefix_sha256")
        if declared_sha != legacy.sha256_token_ids(values):
            raise LiveExecutorError(f"{label} exact history SHA-256 drifted")
        if generated[: len(values)] != values:
            raise LiveExecutorError(f"{label} exact history is not a literal H0 generated prefix")
        return values

    b_history = record_prefix(b_record, f"B {target_owner}")
    a_history = record_prefix(a_record, f"A {a_owner}", allow_empty=True)

    def physical_prefix_row_count(prefix: tuple[int, ...], label: str) -> int:
        consumed = 0
        for index, row in enumerate(rows):
            if consumed == len(prefix):
                return index
            next_consumed = consumed + len(row)
            if next_consumed > len(prefix):
                raise LiveExecutorError(
                    f"{label} exact history ends mid physical H0 row {index}"
                )
            consumed = next_consumed
        if consumed != len(prefix):
            raise LiveExecutorError(
                f"{label} exact history exceeds the physical H0 row stream"
            )
        return len(rows)

    a_row_count = physical_prefix_row_count(a_history, f"A {a_owner}")
    b_row_count = physical_prefix_row_count(b_history, f"B {target_owner}")
    if len(a_history) >= len(b_history) or not (
        a_history == b_history[: len(a_history)]
    ):
        raise LiveExecutorError(
            f"A {a_owner} exact history is not a strict literal prefix of B {target_owner}"
        )
    if b_row_count != a_row_count + 1:
        raise LiveExecutorError(
            f"B {target_owner} exact history must add exactly one physical row after A {a_owner}"
        )
    latest_row = tuple(b_history[len(a_history) :])
    if latest_row != rows[a_row_count] or len(latest_row) == 0:
        raise LiveExecutorError(
            f"B {target_owner} exact history suffix is not exactly one physical H0 row"
        )

    covered = b_record.get("covered_owner_ids")
    if not isinstance(covered, Sequence) or isinstance(covered, (str, bytes)):
        raise LiveExecutorError(f"B {target_owner} covered-owner order is missing")
    covered_ids = tuple(str(owner) for owner in covered)
    if not covered_ids or any(not owner for owner in covered_ids) or len(set(covered_ids)) != len(covered_ids):
        raise LiveExecutorError(f"B {target_owner} covered-owner order is malformed")
    latest_owner = b_record.get("latest_covered_owner_id")
    if latest_owner != covered_ids[-1] or latest_owner != a_owner:
        raise LiveExecutorError(
            f"B {target_owner} latest_covered_owner_id is inconsistent with covered-owner order/A"
        )
    due = b_record.get("due_boundary_evidence")
    if not isinstance(due, Mapping):
        raise LiveExecutorError(f"B {target_owner} lacks due_boundary_evidence")
    latest_order = due.get("latest_covered_generated_order")
    if isinstance(latest_order, bool) or not isinstance(latest_order, int) or latest_order < 0:
        raise LiveExecutorError(
            f"B {target_owner} due_boundary_evidence.latest_covered_generated_order is invalid"
        )
    if latest_order >= len(rows) or tuple(rows[latest_order]) != latest_row:
        raise LiveExecutorError(
            f"B {target_owner} due-boundary latest generated order does not identify the B suffix row"
        )
    due_covered = due.get("covered_owner_ids")
    if isinstance(due_covered, Sequence) and not isinstance(due_covered, (str, bytes)):
        if tuple(str(owner) for owner in due_covered) != covered_ids:
            raise LiveExecutorError(f"B {target_owner} due-boundary covered-owner order drifted")
    if due.get("latest_covered_owner_id") != a_owner:
        raise LiveExecutorError(f"B {target_owner} due-boundary latest owner differs from covered A")

    boundaries = b_record.get("generated_row_boundaries")
    if not isinstance(boundaries, Sequence) or isinstance(boundaries, (str, bytes)):
        raise LiveExecutorError(f"B {target_owner} generated-row boundaries are missing")
    matching_boundaries = [
        boundary
        for boundary in boundaries
        if isinstance(boundary, Mapping) and boundary.get("generated_order") == latest_order
    ]
    if len(matching_boundaries) != 1:
        raise LiveExecutorError(
            f"B {target_owner} has {len(matching_boundaries)} generated-row boundaries at latest covered order"
        )
    latest_boundary = matching_boundaries[0]
    if (
        latest_boundary.get("closure_step") != len(b_history) - 1
        or latest_boundary.get("row_start_step") != len(a_history)
        or latest_boundary.get("match_status") != "tp"
        or latest_boundary.get("gt_owner_id") != a_owner
        or latest_boundary.get("closure_token_id") != latest_row[-1]
    ):
        raise LiveExecutorError(
            f"B {target_owner} latest generated-row boundary does not close the covered-A suffix exactly"
        )
    contract = getattr(adapter, "wrapper_contract", None)
    closure_id = getattr(contract, "closure_token_id", None)
    if closure_id is not None:
        if latest_row.count(int(closure_id)) != 1 or latest_row[-1] != int(closure_id):
            raise LiveExecutorError(
                f"B {target_owner} latest physical row lacks exactly one terminal closure token"
            )

    return {
        "exact_history_token_ids": b_history,
        "exact_history_sha256": legacy.sha256_token_ids(b_history),
        "exact_A_history_token_ids": a_history,
        "exact_A_history_sha256": legacy.sha256_token_ids(a_history),
        "completed_row_ids": tuple(rows[:b_row_count]),
        "latest_row_ids": latest_row,
        "receipt": {
            "resolver": "checkpoint_native_h0_trace_and_ledger",
            "image_id": image_id,
            "target_B_owner_id": target_owner,
            "covered_A_owner_id": a_owner,
            "exact_history_sha256": legacy.sha256_token_ids(b_history),
            "exact_A_history_sha256": legacy.sha256_token_ids(a_history),
            "physical_prefix_row_count": b_row_count,
            "latest_covered_generated_order": latest_order,
            "latest_row_sha256": legacy.sha256_token_ids(latest_row),
            "latest_boundary": dict(latest_boundary),
        },
    }


# Keep a descriptive alias for tests and callers that use resolver terminology.
_resolve_ledger_exact_history = _ledger_exact_history_resolver


def _preflight_full_runtime_cohort(
    orchestrator: Any,
    paths: Mapping[str, Path],
    *,
    processor_adapter: Any | None = None,
) -> dict[str, Any]:
    """Prove every admitted static event has one complete CPU runtime context."""
    expected_companion = paths["cohort"].with_name(
        paths["cohort"].name.replace(".json", ".manifest.json")
    )
    if paths.get("cohort_manifest") != expected_companion:
        raise LiveExecutorError(
            "runtime-cohort companion manifest path differs from the exact legacy loader path"
        )
    manifest_info = cohort.validate_manifest(paths["manifest"])
    census, _ = _json_file(paths["census"], "census")
    rows = census.get("rows")
    if not isinstance(rows, list):
        raise LiveExecutorError("runtime-cohort preflight census rows are missing")
    legacy_events = getattr(orchestrator, "events", None)
    if not isinstance(legacy_events, list):
        raise LiveExecutorError("runtime-cohort preflight legacy events are missing")
    h0_ledger_records: Mapping[str, Any] | None = None
    derived_by_image: dict[str, Any] | None = None
    source_by_image: dict[str, Any] | None = None
    h0_image_plans: dict[str, dict[str, Any]] | None = None
    admitted_identities: list[tuple[Any, ...]] = []
    authoritative_bindings: list[dict[str, Any]] = []
    processor_context_bindings: list[dict[str, Any]] = []

    def ensure_processor_adapter() -> Any:
        nonlocal processor_adapter
        if processor_adapter is not None:
            return processor_adapter
        builder = getattr(orchestrator, "_load_ineligible_materialization_adapter", None)
        if not callable(builder):
            raise LiveExecutorError(
                "runtime-cohort preflight lacks a processor-only adapter builder"
            )
        try:
            # The builder is intentionally invoked on the already-bound
            # orchestrator.  It assembles only processor/tokenizer/panel/H0
            # state and never opens a backend or loads an executable model.
            processor_adapter = builder({})
        except Exception as exc:
            raise LiveExecutorError(
                f"runtime-cohort processor-only adapter construction failed: {exc}"
            ) from exc
        if getattr(processor_adapter, "model", None) is not None:
            raise LiveExecutorError(
                "runtime-cohort processor-only preflight unexpectedly loaded a model"
            )
        if getattr(processor_adapter, "session", None) is not None:
            raise LiveExecutorError(
                "runtime-cohort processor-only preflight unexpectedly opened a backend"
            )
        return processor_adapter

    for event in manifest_info["events"]:
        owner_refs = event["owner_refs"]
        identity = (
            "S",
            owner_refs["gt_owner_id"],
            event["image_id"],
            owner_refs.get("source_panel_object_index"),
            owner_refs.get("derived_panel_object_index"),
        )
        admitted_identities.append(identity)
        row_matches = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and (
                row.get("checkpoint"),
                row.get("gt_owner_id"),
                row.get("image_id"),
                row.get("source_panel_object_index"),
                row.get("derived_panel_object_index"),
            )
            == identity
        ]
        if len(row_matches) != 1:
            raise LiveExecutorError(f"runtime-cohort preflight found {len(row_matches)} census rows for {event['event_id']}")
        row = row_matches[0]
        runtime_event = _runtime_event(event, row)
        legacy_matches = [
            candidate
            for candidate in legacy_events
            if isinstance(candidate, Mapping) and _is_exact_legacy_s_event(candidate, runtime_event)
        ]
        if len(legacy_matches) != 1:
            raise LiveExecutorError(f"runtime-cohort preflight found {len(legacy_matches)} contexts for {event['event_id']}")
        legacy_event = legacy_matches[0]
        if h0_ledger_records is None:
            observed_h0 = getattr(orchestrator, "_h0_ledger_records", None)
            source_panel_path = getattr(orchestrator, "source_panel_path", None)
            if not isinstance(observed_h0, Mapping) or not isinstance(source_panel_path, Path):
                raise LiveExecutorError("runtime-cohort authoritative H0/panel inputs are missing")
            try:
                from src.data import load_raw_examples

                derived_values = load_raw_examples(paths["panel"])
                source_values = load_raw_examples(source_panel_path)
            except Exception as exc:
                raise LiveExecutorError(f"runtime-cohort authoritative panel load failed: {exc}") from exc
            derived_by_image = {legacy._raw_image_id(value): value for value in derived_values}
            source_by_image = {legacy._raw_image_id(value): value for value in source_values}
            if len(derived_by_image) != len(derived_values) or len(source_by_image) != len(source_values):
                raise LiveExecutorError("runtime-cohort authoritative panels repeat image identities")
            h0_image_plans = _load_h0_image_plans(
                paths["h0_dir"] / "image_plan.jsonl", derived_values
            )
            h0_ledger_records = observed_h0
        assert derived_by_image is not None and source_by_image is not None and h0_image_plans is not None
        image_key = str(event["image_id"])
        source_raw = source_by_image.get(image_key)
        derived_raw = derived_by_image.get(image_key)
        image_h0_records = h0_ledger_records.get(image_key)
        image_plan = h0_image_plans.get(image_key)
        if source_raw is None or derived_raw is None or not isinstance(image_h0_records, Mapping) or image_plan is None:
            raise LiveExecutorError(f"runtime-cohort authoritative image inputs are missing for {event['event_id']}")
        try:
            _owners, owner_mapping, _mapping_receipt = legacy._build_source_derived_owner_mapping(
                source_raw, derived_raw
            )
        except Exception as exc:
            raise LiveExecutorError(
                f"runtime-cohort authoritative panel mapping failed for {event['event_id']}: {exc}"
            ) from exc
        try:
            exact_raw_owner_mapping = _ephemeral_exact_raw_owner_mapping(
                source_raw,
                derived_raw,
                owner_mapping,
            )
        except LiveExecutorError as exc:
            raise LiveExecutorError(
                f"runtime-cohort authoritative exact raw geometry mapping failed for {event['event_id']}: {exc}"
            ) from exc
        authoritative_bindings.append(
            {
                **_validate_authoritative_event_binding(
                event,
                row,
                legacy_event,
                h0_records=image_h0_records,
                owner_mapping=owner_mapping,
                ),
                **_validate_frozen_geometry(
                    row["geometry"],
                    h0_image_plan=image_plan,
                    owner_mapping=exact_raw_owner_mapping,
                ),
            }
        )
        checkpoint = legacy_event["checkpoint_status"]["S"]
        pair = legacy_event.get("A_B", {}).get("S") if isinstance(legacy_event.get("A_B"), Mapping) else None
        a_context = pair.get("A_latest_covered") if isinstance(pair, Mapping) else None
        b_context = pair.get("B_verified_uncovered") if isinstance(pair, Mapping) else None
        h0_record_index = row.get("h0_record_index")
        if (
            not isinstance(pair, Mapping)
            or pair.get("pair_status") != "verified_pair"
            or not isinstance(a_context, Mapping)
            or a_context.get("gt_owner_id") != row.get("covered_A_owner_id")
            or a_context.get("natural_boundary") != row.get("covered_A_natural_boundary")
            or not isinstance(b_context, Mapping)
            or b_context.get("gt_owner_id") != owner_refs["gt_owner_id"]
            or b_context.get("natural_boundary") != row.get("natural_boundary")
            or b_context.get("exact_prefix_sha256") != event["natural_boundary"]["prefix_sha256"]
            or list(b_context.get("covered_owner_ids", [])) != list(row.get("covered_owner_ids", []))
            or checkpoint.get("natural_boundary") != row.get("natural_boundary")
            or isinstance(h0_record_index, bool)
            or not isinstance(h0_record_index, int)
            or checkpoint.get("boundary_record") != h0_record_index
        ):
            raise LiveExecutorError(f"runtime-cohort preflight active A/B or H0 boundary drifted for {event['event_id']}")
        if dict(legacy_event["geometry_by_checkpoint"]["S"]) != dict(row["geometry"]):
            raise LiveExecutorError(f"runtime-cohort preflight geometry drifted for {event['event_id']}")
        cpu_adapter = ensure_processor_adapter()
        try:
            seeded_context = legacy._make_event_context(
                cpu_adapter,
                legacy_event,
                history_resolver=_ledger_exact_history_resolver,
            )
        except Exception as exc:
            if isinstance(exc, LiveExecutorError):
                raise
            raise LiveExecutorError(
                f"runtime-cohort seeded context construction failed for {event['event_id']}: {exc}"
            ) from exc
        runtime = getattr(seeded_context, "runtime", None)
        if runtime is None:
            raise LiveExecutorError(
                f"runtime-cohort seeded context lacks runtime for {event['event_id']}"
            )
        binding = gate.LiveRuntimeBinding(
            adapter=cpu_adapter,
            runtime=runtime,
            event=runtime_event,
            seeded_context=seeded_context,
        )
        try:
            natural_context, natural_boundary, natural_identity = gate.build_natural_event_context(
                binding,
                event_id=str(event["event_id"]),
            )
        except Exception as exc:
            raise LiveExecutorError(
                f"runtime-cohort natural context construction failed for {event['event_id']}: {exc}"
            ) from exc
        admitted_history = _prefix(event)
        admitted_history_sha = event["natural_boundary"].get("prefix_sha256")
        exact_history = tuple(natural_context.exact_history_token_ids)
        if exact_history != admitted_history:
            raise LiveExecutorError(
                f"runtime-cohort exact history differs from admitted manifest for {event['event_id']}"
            )
        if cohort.sha256_json(list(exact_history)) != admitted_history_sha:
            raise LiveExecutorError(
                f"runtime-cohort exact history hash differs from admitted manifest for {event['event_id']}"
            )
        prompt = tuple(natural_context.prompt_token_ids)
        if tuple(natural_context.prefix_token_ids) != prompt + exact_history:
            raise LiveExecutorError(
                f"runtime-cohort natural prefix is not prompt+exact history for {event['event_id']}"
            )
        if tuple(natural_boundary.natural_prefix_token_ids) != prompt + exact_history:
            raise LiveExecutorError(
                f"runtime-cohort boundary natural prefix differs from prompt+exact history for {event['event_id']}"
            )
        latest = tuple(seeded_context.latest_row_ids)
        if latest != tuple(natural_context.latest_history_row_token_ids):
            raise LiveExecutorError(
                f"runtime-cohort latest covered row differs between seeded/natural contexts for {event['event_id']}"
            )
        closure_id = getattr(cpu_adapter.wrapper_contract, "closure_token_id", None)
        if closure_id is None:
            raise LiveExecutorError("runtime-cohort wrapper lacks a native closure token")
        terminal_positions = tuple(
            int(natural_context.history_prefix_width or 0) + index
            for index, token in enumerate(latest)
            if int(token) == int(closure_id)
        )
        if len(terminal_positions) != 1:
            raise LiveExecutorError(
                f"runtime-cohort latest row has {len(terminal_positions)} terminal key positions for {event['event_id']}"
            )
        expected_width = len(prompt) + len(
            seeded_context.prefix_receipt.get("h0", {}).get("exact_prefix_token_ids", ())
        ) - len(latest)
        if natural_context.history_prefix_width != expected_width:
            raise LiveExecutorError(
                f"runtime-cohort history_prefix_width drifted for {event['event_id']}"
            )
        processor_context_bindings.append(
            {
                "event_id": str(event["event_id"]),
                "prompt_token_count": len(prompt),
                "exact_history_sha256": cohort.sha256_json(list(exact_history)),
                "manifest_history_sha256": str(admitted_history_sha),
                "natural_prefix_sha256": cohort.sha256_json(
                    list(natural_context.prefix_token_ids)
                ),
                "latest_row_sha256": cohort.sha256_json(list(latest)),
                "history_prefix_width": int(natural_context.history_prefix_width or 0),
                "latest_terminal_key_positions": list(terminal_positions),
                "latest_terminal_key_position_count": len(terminal_positions),
                "seeded_prefix_sha256": cohort.sha256_json(
                    [int(value) for value in seeded_context.prefix_ids.reshape(-1).tolist()]
                ),
                "physical_completed_row_count": len(seeded_context.completed_row_ids),
                "natural_identity": dict(natural_identity),
            }
        )
    active_legacy_identities = [
        (
            "S",
            event.get("gt_owner_id"),
            event.get("image_id"),
            _legacy_event_indices(event)[0],
            _legacy_event_indices(event)[1],
        )
        for event in legacy_events
        if isinstance(event, Mapping)
        and isinstance(event.get("geometry_by_checkpoint"), Mapping)
        and isinstance(event["geometry_by_checkpoint"].get("S"), Mapping)
        and event["geometry_by_checkpoint"]["S"].get("launch_eligible") is True
    ]
    if active_legacy_identities != admitted_identities:
        raise LiveExecutorError("runtime-cohort preflight static context order/membership differs from manifest")
    receipt = {
        "status": "passed",
        "event_count": len(admitted_identities),
        "event_identities_sha256": cohort.sha256_json(admitted_identities),
        "authoritative_bindings_sha256": cohort.sha256_json(authoritative_bindings),
        "processor_context_bindings": processor_context_bindings,
        "processor_context_bindings_sha256": cohort.sha256_json(processor_context_bindings),
        "cohort_path": str(paths["cohort"]),
        "cohort_sha256": _sha256_file(paths["cohort"]),
        "cohort_manifest_path": str(paths["cohort_manifest"]),
        "cohort_manifest_sha256": _sha256_file(paths["cohort_manifest"]),
    }
    receipt["receipt_sha256"] = cohort.sha256_json(receipt)
    return receipt


class SNaturalBoundaryKNHLiveExecutor:
    """Lazy, singleton-model executor for one admitted event at a time."""

    def __init__(self, *, loader: Callable[..., Any] | None = None) -> None:
        self._loader = loader
        self._manifest_info: dict[str, Any] | None = None
        self._census: dict[str, Any] | None = None
        self._orchestrator: Any | None = None
        self._adapter: Any | None = None
        self._identity: dict[str, Any] = {}
        self._load_binding: dict[str, Any] | None = None
        self._pre_gpu_identity: dict[str, Any] | None = None
        self._runtime_versions: dict[str, str] | None = None

    def configure_pre_gpu_identity(
        self,
        identity: Mapping[str, Any],
        *,
        runtime_versions: Mapping[str, str],
    ) -> None:
        """Install the runner-validated shard binding before any event work."""

        frozen = _frozen_json_object(identity)
        if self._pre_gpu_identity is not None and frozen != self._pre_gpu_identity:
            raise LiveExecutorError("live executor is already configured for a different shard identity")
        if self._adapter is not None and frozen != self._load_binding:
            raise LiveExecutorError("cannot reconfigure a loaded live executor")
        expected_runtime = frozen.get("runtime")
        if not isinstance(expected_runtime, Mapping) or any(
            expected_runtime.get(key) != value for key, value in runtime_versions.items()
        ):
            raise LiveExecutorError("observed runtime versions differ from pre-GPU identity")
        self._pre_gpu_identity = frozen
        self._runtime_versions = dict(runtime_versions)

    def _require_pre_gpu_identity(self, paths: Mapping[str, Path]) -> dict[str, Any]:
        identity = self._pre_gpu_identity
        if identity is None:
            raise LiveExecutorError("live executor was not preconfigured by the guarded shard runner")
        assignment = identity.get("device_assignment")
        input_paths = identity.get("input_paths")
        if not isinstance(assignment, Mapping) or not isinstance(input_paths, Mapping):
            raise LiveExecutorError("preconfigured shard identity is structurally incomplete")
        if os.environ.get(ENV_SHARD_ID) != assignment.get("shard_id"):
            raise LiveExecutorError("runtime shard environment differs from preconfigured identity")
        observed_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        if (
            observed_cuda != assignment.get("physical_device")
            or observed_cuda != assignment.get("observed_cuda_visible_devices")
        ):
            raise LiveExecutorError("runtime CUDA_VISIBLE_DEVICES differs from preconfigured identity")
        expected_paths = {
            "manifest": paths["manifest"],
            "census": paths["census"],
            "config": paths["config"],
            "panel": paths["panel"],
            "cohort": paths["cohort"],
            "cohort_manifest": paths["cohort_manifest"],
            "h0_root": paths["h0_root"],
            "h0_dir": paths["h0_dir"],
        }
        for label, path in expected_paths.items():
            if input_paths.get(label) != str(path):
                raise LiveExecutorError(f"runtime {label} path differs from preconfigured identity")
        if identity.get("pre_gpu_receipt_path") != str(paths["pre_gpu_receipt"]):
            raise LiveExecutorError("runtime receipt path differs from preconfigured identity")
        return identity

    def _paths(self) -> dict[str, Path]:
        names = {"manifest": ENV_MANIFEST, "census": ENV_CENSUS, "config": ENV_CONFIG, "panel": ENV_PANEL, "cohort": ENV_COHORT, "cohort_manifest": ENV_COHORT_MANIFEST, "h0_root": ENV_H0_ROOT, "h0_dir": ENV_H0_DIR, "pre_gpu_receipt": ENV_PRE_GPU_RECEIPT}
        result: dict[str, Path] = {}
        for label, env in names.items():
            value = os.environ.get(env)
            if not value:
                raise LiveExecutorError(f"required identity environment {env} is missing")
            result[label] = _directory(value, env) if label in {"h0_root", "h0_dir"} else _regular_file(value, env)
        return result

    def _validate_inputs(self, event: Mapping[str, Any], arm_order: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
        if tuple(str(arm) for arm in arm_order) != cohort.ARM_ORDER:
            raise LiveExecutorError("requested arm order differs from frozen S K/N/H tuple")
        paths = self._paths()
        load_binding = self._require_pre_gpu_identity(paths)
        try:
            info = cohort.validate_manifest(paths["manifest"])
        except Exception as exc:
            raise LiveExecutorError(f"manifest validation failed: {exc}") from exc
        input_hashes = load_binding.get("input_hashes")
        if not isinstance(input_hashes, Mapping) or (
            input_hashes.get("manifest_raw_sha256") != info.get("manifest_sha256")
            or input_hashes.get("manifest_self_sha256") != info.get("manifest_self_sha256")
        ):
            raise LiveExecutorError("manifest hashes differ from preconfigured identity")
        for label in ("panel", "cohort"):
            declared = info.get("source_identities", {}).get(label, {})
            declared_path = declared.get("path") if isinstance(declared, Mapping) else None
            if declared_path is not None and Path(str(declared_path)).expanduser().resolve() != paths[label]:
                raise LiveExecutorError(f"manifest {label} identity differs from executor input")
        if info["source_census_sha256"] != _sha256_file(paths["census"]):
            raise LiveExecutorError("census bytes differ from manifest source_census sha256")
        census, raw = _json_file(paths["census"], "census")
        if raw != cohort.canonical_json_bytes(census) + b"\n":
            raise LiveExecutorError("census must be canonical JSON with one trailing newline")
        if census.get("census_revision") != "census-v3" or census.get("status") != "sealed" or not isinstance(census.get("rows"), list):
            raise LiveExecutorError("census is not a sealed census-v3 row document")
        census_self = census.get("self_sha256")
        census_body = dict(census)
        census_body.pop("self_sha256", None)
        if not isinstance(census_self, str) or census_self != cohort.sha256_json(census_body):
            raise LiveExecutorError("census self_sha256 mismatch")
        for index, row in enumerate(census["rows"]):
            if not isinstance(row, Mapping) or not isinstance(row.get("gt_owner_id"), str) or not isinstance(row.get("image_id"), int):
                raise LiveExecutorError(f"census row {index} identity is malformed")
        events = info["events"]
        target = next((item for item in events if item.get("event_id") == _event_id(event)), None)
        if not isinstance(target, Mapping) or dict(target) != dict(event):
            raise LiveExecutorError("event is not the exact admitted manifest event")
        owner_refs = target["owner_refs"]
        census_matches = [
            item
            for item in census["rows"]
            if isinstance(item, Mapping)
            and item.get("checkpoint") == "S"
            and str(item.get("gt_owner_id")) == str(owner_refs["gt_owner_id"])
            and item.get("image_id") == target.get("image_id")
            and item.get("source_panel_object_index") == owner_refs.get("source_panel_object_index")
            and item.get("derived_panel_object_index") == owner_refs.get("derived_panel_object_index")
        ]
        if len(census_matches) != 1:
            raise LiveExecutorError(
                "admitted event does not resolve to exactly one S census row with full owner/image/index identity"
            )
        row = census_matches[0]
        _runtime_event(target, row)
        return dict(target), dict(row), paths

    def _load_once(self, paths: Mapping[str, Path], load_binding: Mapping[str, Any]) -> None:
        if self._adapter is not None:
            if self._load_binding is None or dict(load_binding) != self._load_binding:
                raise LiveExecutorError(
                    "current launch paths/hashes/receipt differ from the loaded singleton identity"
                )
            return
        try:
            preload = gate._require_preload_cuda_visibility()
            self._orchestrator = legacy.OwnerInterfaceOrchestrator(checkpoint="S", stage="all", output_dir=Path("/tmp/s-natural-boundary-live-executor"), fail_collision=False, config_path=paths["config"], panel_path=paths["panel"], cohort_path=paths["cohort"], h0_root=paths["h0_root"], h0_dir=paths.get("h0_dir"))
            identity = self._orchestrator._load_cpu_contract()
            identity = {
                **dict(identity),
                "full_runtime_cohort_preflight": _preflight_full_runtime_cohort(
                    self._orchestrator, paths
                ),
            }
            if self._loader is not None:
                self._adapter = self._loader(self._orchestrator, identity)
            else:
                self._adapter = self._orchestrator._load_adapter(identity)
            self._identity = {
                **dict(identity),
                "cuda_visible_devices": preload,
                "model_identity": _observed_model_identity(identity, load_binding),
                "backend_identity": _observed_backend_identity(self._adapter),
            }
            postload = gate._attest_postload_cuda_device(
                gate.LiveRuntimeBinding(adapter=self._adapter, runtime=None, event={}, seeded_context=None)
            )
            self._identity["model_device_attestation"] = postload
            self._load_binding = dict(load_binding)
        except Exception as exc:
            raise LiveExecutorError(f"live S runtime load failed: {exc}") from exc

    def execute_event(self, event: Mapping[str, Any], *, arm_order: Sequence[str] = cohort.ARM_ORDER) -> Mapping[str, Any]:
        target, row, paths = self._validate_inputs(event, arm_order)
        runtime_event = _runtime_event(target, row)
        if self._pre_gpu_identity is None:
            raise LiveExecutorError("live executor was not preconfigured by the guarded shard runner")
        load_binding = self._pre_gpu_identity
        self._load_once(paths, load_binding)
        try:
            owner = runtime_event["owner_refs"]["gt_owner_id"]
            legacy_matches = [
                candidate
                for candidate in self._orchestrator.events
                if isinstance(candidate, Mapping) and _is_exact_legacy_s_event(candidate, runtime_event)
            ]
            if len(legacy_matches) != 1:
                raise LiveExecutorError(
                    f"event {owner} does not resolve to exactly one legacy S event with full owner/image/index identity"
                )
            legacy_event = legacy_matches[0]
            self._legacy_event = dict(legacy_event)
            seeded = legacy._make_event_context(
                self._adapter,
                self._legacy_event,
                history_resolver=_ledger_exact_history_resolver,
            )
            runtime = getattr(seeded, "runtime", None)
            if runtime is None:
                raise LiveExecutorError("bound event context lacks runtime")
            binding = gate.LiveRuntimeBinding(adapter=self._adapter, runtime=runtime, event=runtime_event, seeded_context=seeded, identity=self._identity)
            context, boundary, event_identity = gate.build_natural_event_context(
                binding, event_id=_event_id(runtime_event)
            )
            admitted_history = _prefix(target)
            admitted_history_sha = target.get("natural_boundary", {}).get("prefix_sha256")
            if tuple(context.exact_history_token_ids) != admitted_history:
                raise LiveExecutorError(
                    "gate exact history differs from admitted event natural history"
                )
            if cohort.sha256_json(list(context.exact_history_token_ids)) != admitted_history_sha:
                raise LiveExecutorError(
                    "gate exact history hash differs from admitted event natural history"
                )
            if tuple(context.prefix_token_ids) != tuple(context.prompt_token_ids) + admitted_history:
                raise LiveExecutorError(
                    "gate natural context prefix is not prompt+exact history"
                )
            seeded_ids = tuple(
                int(value)
                for value in seeded.prefix_ids.detach().cpu().reshape(-1).tolist()
            )
            expected_seeded = (
                tuple(context.prompt_token_ids)
                + admitted_history
                + (int(self._adapter.wrapper_contract.object_ref_start_token_id),)
            )
            if seeded_ids != expected_seeded:
                raise LiveExecutorError(
                    "seeded model input is not prompt+exact history+native opener"
                )
            if tuple(boundary.natural_prefix_token_ids) != tuple(context.prefix_token_ids):
                raise LiveExecutorError(
                    "gate natural boundary prefix differs from natural context prefix"
                )
            actuators = gate.build_live_attention_mask_actuators(binding, context)
            gate_runner = gate.SPrimaryNaturalBoundaryGate(binding, context, boundary, event_identity)
            gate_runner.scalar.k14_reference_positions = gate._resolve_k14_reference_positions(binding)
            matrix = gate_runner.run_matrix(arms=cohort.ARM_ORDER, residual_actuator=gate.build_live_residual_actuator(), attention_mask_actuators=actuators)
            outputs = dict(matrix.get("arms", {}))
            if tuple(outputs) != cohort.ARM_ORDER:
                raise LiveExecutorError("gate returned an arm mapping outside the frozen order")
            cuda_attestation = self._identity.get("model_device_attestation")
            if not isinstance(cuda_attestation, Mapping) or cuda_attestation.get("passed") is not True:
                raise LiveExecutorError("live model lacks a passed logical CUDA device attestation")
            preload_cuda = self._identity.get("cuda_visible_devices")
            if not isinstance(preload_cuda, Mapping):
                raise LiveExecutorError("live model lacks pre-load physical CUDA visibility evidence")
            cuda_evidence = {**dict(cuda_attestation), "cuda_visible_devices": dict(preload_cuda)}
            expected_config_sha = load_binding.get("input_hashes", {}).get("config_sha256")
            observed_config_sha = self._identity.get("config_sha256", expected_config_sha)
            if observed_config_sha != expected_config_sha:
                raise LiveExecutorError("loaded config identity differs from the pre-GPU binding")
            attestation = _executor_identity(
                load_binding,
                model=dict(self._identity.get("model_identity", {})),
                backend=self._identity.get("backend_identity"),
                device=binding.device,
                cuda=cuda_evidence,
                config_sha256=observed_config_sha,
                runtime_versions=self._runtime_versions or {},
                full_runtime_cohort_preflight=self._identity.get(
                    "full_runtime_cohort_preflight"
                ),
            )
            for arm, result in outputs.items():
                result["executor_identity"] = attestation
            return outputs
        except LiveExecutorError:
            raise
        except Exception as exc:
            raise LiveExecutorError(f"S natural gate execution failed: {exc}") from exc


_DEFAULT_EXECUTOR: SNaturalBoundaryKNHLiveExecutor | None = None


def configure_pre_gpu_identity(
    identity: Mapping[str, Any],
    *,
    runtime_versions: Mapping[str, str],
) -> None:
    """Configure the module singleton from the guarded shard runner."""

    global _DEFAULT_EXECUTOR
    if _DEFAULT_EXECUTOR is None:
        _DEFAULT_EXECUTOR = SNaturalBoundaryKNHLiveExecutor()
    _DEFAULT_EXECUTOR.configure_pre_gpu_identity(identity, runtime_versions=runtime_versions)


def execute_event(event: Mapping[str, Any], *, arm_order: Sequence[str] = cohort.ARM_ORDER) -> Mapping[str, Any]:
    global _DEFAULT_EXECUTOR
    if _DEFAULT_EXECUTOR is None:
        _DEFAULT_EXECUTOR = SNaturalBoundaryKNHLiveExecutor()
    return _DEFAULT_EXECUTOR.execute_event(event, arm_order=arm_order)


def reset_default_executor() -> None:
    global _DEFAULT_EXECUTOR
    _DEFAULT_EXECUTOR = None

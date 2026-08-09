#!/usr/bin/env python3
"""Build a strict, CPU-only owner ledger from one completed H0 run.

The H0 run is an inference substrate, not a support assay.  This builder binds
the run's immutable artifacts to the admitted source panel and its
``geo_sorted_xy`` derived view, then performs the same category-aware,
score-descending, one-to-one ``IoU >= 0.50`` bbox matching used by the current
COCO evaluator.  It never loads a model or starts inference.

The public :func:`build_h0_ledger` function is intentionally usable from tests
and from the cohort materializer.  The resulting JSON envelope has the strict
``native_h0`` fields consumed by
``materialize_static_dynamic_owner_interface_cohort.py``; a deterministic
records JSONL and a receipt are emitted when paths are supplied.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

if __package__ in {None, ""}:  # Allow ``python scripts/research/...py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.geometry import coord_bins_to_pixel_xyxy
from src.eval.detection_categories import normalize_coco_category_name


UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
SCHEMA_VERSION = "static_dynamic_native_h0_owner_ledger.v1"
RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.receipt"
IOU_THRESHOLD = 0.50
EXPECTED_IMAGE_2299 = 2299
EXPECTED_WRAPPERS = {"S": "object_box_closed", "A": "object_box_commit"}
EXPECTED_PARSERS = {
    "S": "compact_object_box_closed_only",
    "A": "compact_object_box_commit_only",
}
EXPECTED_GENERATION = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 3084,
}
COORD_RE = re.compile(r"^<\|coord_(?P<bin>[0-9]{1,3})\|>$")
IM_END_TOKEN_ID = 151645
COMMIT_TOKEN_ID = 151669
BOX_END_TOKEN_ID = 151649
TP_PREFIX_SEMANTICS = "before_queried_owner_row"
FN_PREFIX_SEMANTICS = "after_strict_covered_row_pre_stop"


class H0LedgerContractError(ValueError):
    """Raised when the H0 artifact or identity contract is not established."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise H0LedgerContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise H0LedgerContractError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise H0LedgerContractError(f"cannot read {label}: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise H0LedgerContractError(
                f"{label} row {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise H0LedgerContractError(f"{label} row {line_number} is not an object")
        rows.append(value)
    return rows


def _source_value(
    source: str | Path | dict[str, Any] | list[Any], *, label: str
) -> tuple[Any, dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve(strict=True)
        raw = path.read_bytes()
        if path.suffix.lower() == ".jsonl":
            rows: list[Any] = []
            for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise H0LedgerContractError(
                        f"{label} row {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise H0LedgerContractError(f"{label} row {line_number} is not an object")
                rows.append(value)
            return rows, {"path": str(path), "sha256": sha256_bytes(raw)}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise H0LedgerContractError(f"{label} is not valid JSON: {path}") from exc
        return value, {"path": str(path), "sha256": sha256_bytes(raw)}
    if isinstance(source, dict):
        value: Any = dict(source)
    elif isinstance(source, list):
        value = list(source)
    else:
        raise TypeError(f"{label} must be a path, mapping, or list")
    return value, {"inline": True, "sha256": sha256_json(value)}


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value.lower()
    ):
        raise H0LedgerContractError(f"{label} must be a lowercase SHA-256")
    return value.lower()


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise H0LedgerContractError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise H0LedgerContractError(f"{label} must be an integer") from exc
    if str(value).strip() != str(result) and not isinstance(value, (int, float)):
        raise H0LedgerContractError(f"{label} must be an integer")
    return result


def _category(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("category_name", "description", "desc", "category"):
            if isinstance(value.get(key), str) and value[key].strip():
                return normalize_coco_category_name(value[key])
    if isinstance(value, str) and value.strip():
        return normalize_coco_category_name(value)
    return ""


def _bbox(value: Any, *, width: int | None = None, height: int | None = None) -> tuple[float, float, float, float]:
    if isinstance(value, dict):
        for key in ("bbox", "bbox_xyxy", "bbox_2d", "pixel_bbox", "bbox_pixel_xyxy"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise H0LedgerContractError("bbox must contain exactly four values")
    if all(isinstance(item, str) and COORD_RE.fullmatch(item) for item in value):
        if width is None or height is None:
            raise H0LedgerContractError("coordinate-token bbox requires image dimensions")
        bins = [int(COORD_RE.fullmatch(item).group("bin")) for item in value]  # type: ignore[union-attr]
        try:
            result = coord_bins_to_pixel_xyxy(
                bins, image_width=width, image_height=height, field="bbox"
            )
        except Exception as exc:
            raise H0LedgerContractError("invalid coordinate-token bbox") from exc
        return tuple(float(item) for item in result)  # type: ignore[return-value]
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise H0LedgerContractError("bbox values must be numeric") from exc
    if not all(math.isfinite(item) for item in result):
        raise H0LedgerContractError("bbox values must be finite")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise H0LedgerContractError("bbox must have positive width and height")
    return result  # type: ignore[return-value]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _row_image_id(row: dict[str, Any]) -> int:
    if row.get("image_id") is not None:
        return _as_int(row["image_id"], "image_id")
    for key in ("row_id", "example_id"):
        value = row.get(key)
        if isinstance(value, str):
            match = re.search(r"(?:^|_)([0-9]{4,})$", value)
            if match:
                return int(match.group(1))
    raise H0LedgerContractError("artifact row has no resolvable image_id")


def _row_id(row: dict[str, Any]) -> str:
    value = row.get("row_id", row.get("example_id"))
    if value is None:
        raise H0LedgerContractError("artifact row has no row_id")
    return str(value)


def _panel_rows(value: Any, label: str) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else [value]
    if not rows:
        raise H0LedgerContractError(f"{label} is empty")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise H0LedgerContractError(f"{label} row {index} is not an object")
        image_id = _row_image_id(row)
        if image_id in seen:
            raise H0LedgerContractError(f"{label} contains duplicate image_id={image_id}")
        seen.add(image_id)
        objects = row.get("objects")
        if not isinstance(objects, list):
            raise H0LedgerContractError(f"{label} image {image_id} has no objects list")
        result.append(row)
    return result


def _object_ann_id(obj: dict[str, Any]) -> str | int | None:
    for key in ("coco_ann_id", "annotation_id", "ann_id"):
        if obj.get(key) is not None and not isinstance(obj[key], bool):
            return obj[key]
    return None


def _object_bbox(obj: dict[str, Any], row: dict[str, Any]) -> tuple[float, float, float, float]:
    raw = obj.get("bbox_2d", obj.get("bbox"))
    return _bbox(raw, width=_as_int(row.get("width", row.get("image_width")), "panel width"), height=_as_int(row.get("height", row.get("image_height")), "panel height"))


def _validate_derived(
    source_value: Any,
    derived_value: Any,
    receipt: dict[str, Any],
    *,
    source_sha256: str,
    derived_sha256: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    expected = {
        "unit_id": UNIT_ID,
        "ordering": "geo_sorted_xy",
        "source_sha256": source_sha256,
        "derived_sha256": derived_sha256,
        "coordinate_arity_verified": True,
        "owner_multiset_preserved": True,
        "stable_sort_verified": True,
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            raise H0LedgerContractError(
                f"derived receipt {key} mismatch: {receipt.get(key)!r} != {expected_value!r}"
            )
    if receipt.get("sort_key") != ["decoded_x1", "decoded_y1", "source_index"]:
        raise H0LedgerContractError("derived receipt sort_key is not geo_sorted_xy")
    mappings = receipt.get("source_to_derived")
    if not isinstance(mappings, list):
        raise H0LedgerContractError("derived receipt has no source_to_derived mappings")
    if receipt.get("mapping_sha256") != sha256_json(mappings):
        raise H0LedgerContractError("derived receipt mapping_sha256 mismatch")
    source_rows = _panel_rows(source_value, "source panel")
    derived_rows = _panel_rows(derived_value, "derived panel")
    source_by_image = {_row_image_id(row): row for row in source_rows}
    derived_by_image = {_row_image_id(row): row for row in derived_rows}
    if set(source_by_image) != set(derived_by_image):
        raise H0LedgerContractError("source and derived panel image sets differ")
    mapping_by_image: dict[int, dict[str, Any]] = {}
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise H0LedgerContractError("derived receipt mapping row is not an object")
        image = _row_image_id(mapping)
        if image in mapping_by_image:
            raise H0LedgerContractError(f"duplicate derived mapping image_id={image}")
        mapping_by_image[image] = mapping
    if set(mapping_by_image) != set(source_by_image):
        raise H0LedgerContractError("derived receipt does not cover every panel image")
    for image in sorted(source_by_image):
        source_objects = source_by_image[image]["objects"]
        derived_objects = derived_by_image[image]["objects"]
        mapping = mapping_by_image[image].get("mapping")
        if not isinstance(mapping, list) or len(mapping) != len(source_objects) or len(mapping) != len(derived_objects):
            raise H0LedgerContractError(f"image {image} source/derived mapping count mismatch")
        seen_source: set[int] = set()
        seen_derived: set[int] = set()
        for entry in mapping:
            if not isinstance(entry, dict):
                raise H0LedgerContractError(f"image {image} mapping entry is not an object")
            source_index = entry.get("source_index")
            derived_index = entry.get("derived_index")
            if not isinstance(source_index, int) or isinstance(source_index, bool) or not isinstance(derived_index, int) or isinstance(derived_index, bool):
                raise H0LedgerContractError(f"image {image} mapping indices are invalid")
            if source_index in seen_source or derived_index in seen_derived or not (0 <= source_index < len(source_objects)) or not (0 <= derived_index < len(derived_objects)):
                raise H0LedgerContractError(f"image {image} mapping is not bijective")
            seen_source.add(source_index)
            seen_derived.add(derived_index)
            source_obj = source_objects[source_index]
            derived_obj = derived_objects[derived_index]
            if not isinstance(source_obj, dict) or not isinstance(derived_obj, dict):
                raise H0LedgerContractError(f"image {image} contains a non-object owner")
            if canonical_json_bytes(source_obj) != canonical_json_bytes(derived_obj):
                raise H0LedgerContractError(f"image {image} object identity changed in derived panel")
            if entry.get("object_sha256") != sha256_json(source_obj):
                raise H0LedgerContractError(f"image {image} object_sha256 mismatch")
        if seen_source != set(range(len(source_objects))) or seen_derived != set(range(len(derived_objects))):
            raise H0LedgerContractError(f"image {image} mapping is not a full bijection")
    total = sum(len(row["objects"]) for row in source_rows)
    if receipt.get("row_count") != len(source_rows) or receipt.get("owner_count") != total or receipt.get("mapping_count") != total:
        raise H0LedgerContractError("derived receipt row/owner count mismatch")
    return source_by_image, derived_by_image


def _config_payload(source: str | Path | dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if isinstance(source, dict):
        payload = dict(source)
        fingerprint = payload.get("fingerprint") or payload.get("config_fingerprint")
        if isinstance(payload.get("config"), dict):
            config = dict(payload["config"])
            fingerprint = fingerprint or (payload.get("resolution") or {}).get("fingerprint")
        else:
            config = payload
        if not isinstance(fingerprint, str) or not fingerprint:
            fingerprint = sha256_json(config)
        return config, fingerprint, {"inline": True, "sha256": sha256_json(payload)}
    path = Path(source).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]

            value = yaml.safe_load(raw)
        except Exception as exc:
            raise H0LedgerContractError(f"infer config is not JSON/YAML: {path}") from exc
    if not isinstance(value, dict):
        raise H0LedgerContractError("infer config must be an object")
    config = value.get("config") if isinstance(value.get("config"), dict) else value
    if not isinstance(config, dict):
        raise H0LedgerContractError("infer config config payload must be an object")
    resolution = value.get("resolution") if isinstance(value.get("resolution"), dict) else {}
    fingerprint = resolution.get("fingerprint") or value.get("fingerprint") or value.get("config_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        try:
            from src.config.inference import load_infer_config

            fingerprint = load_infer_config(path).fingerprint
        except Exception:
            fingerprint = sha256_json(config)
    return config, fingerprint, {"path": str(path), "sha256": sha256_bytes(raw)}


def _assert_leaf_semantics(
    leaf: Any,
    resolved: Any,
    *,
    path: str = "",
) -> None:
    """Check authored leaf values against the immutable resolved config.

    This is deliberately a one-way check.  The resolved artifact owns the
    fingerprint and runtime semantics; a mutable leaf can only demonstrate
    that the fields it authored still agree.  ``extends`` is metadata and is
    not part of the semantic comparison.
    """

    if isinstance(leaf, dict):
        if not isinstance(resolved, dict):
            raise H0LedgerContractError(f"infer-config leaf/resolved mismatch at {path or '<root>'}")
        for key, value in leaf.items():
            if key == "extends":
                continue
            child_path = f"{path}.{key}" if path else key
            if key not in resolved:
                raise H0LedgerContractError(f"infer-config leaf field is absent from resolved config: {child_path}")
            _assert_leaf_semantics(value, resolved[key], path=child_path)
        return
    if isinstance(leaf, list):
        if not isinstance(resolved, list) or len(leaf) != len(resolved):
            raise H0LedgerContractError(f"infer-config leaf/resolved list mismatch at {path}")
        for index, value in enumerate(leaf):
            _assert_leaf_semantics(value, resolved[index], path=f"{path}[{index}]")
        return
    if leaf != resolved:
        raise H0LedgerContractError(
            f"infer-config leaf value differs from immutable resolved config at {path}"
        )


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _normalise_checkpoint(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower().replace("_", "-")
    if token in {"s", "plain", "primary", "step-2444", "2444", "closed"}:
        return "S"
    if token in {"a", "a3", "step-2445", "2445", "commit"}:
        return "A"
    return None


def _infer_checkpoint(config: dict[str, Any], manifest: dict[str, Any], artifact_dir: Path) -> str:
    candidates: set[str] = set()
    for value in (
        manifest.get("checkpoint"),
        manifest.get("checkpoint_id"),
        _nested(config, "checkpoint"),
        _nested(config, "model", "checkpoint"),
        _nested(config, "model", "name"),
        artifact_dir.name,
        str(_nested(config, "adapter", "path") or ""),
        str(_nested(config, "embedding_delta", "path") or ""),
    ):
        norm = _normalise_checkpoint(value)
        if norm:
            candidates.add(norm)
        text = str(value).lower()
        if "step-2444" in text or "step2444" in text:
            candidates.add("S")
        if "step-2445" in text or "step2445" in text:
            candidates.add("A")
    if len(candidates) != 1:
        raise H0LedgerContractError(f"checkpoint identity is missing or colliding: {sorted(candidates)}")
    return next(iter(candidates))


def _validate_config_and_identity(
    config: dict[str, Any],
    config_fingerprint: str,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    *,
    checkpoint: str,
    source_sha256: str,
    derived_sha256: str,
) -> None:
    declared = _nested(manifest, "resolved_config_fingerprints", "infer_config")
    if declared != config_fingerprint:
        raise H0LedgerContractError("resolved infer config fingerprint mismatch")
    if provenance.get("generation_config_fingerprint") != manifest.get("generation_config_fingerprint"):
        raise H0LedgerContractError("manifest/provenance generation identity mismatch")
    for field in ("model_identity_fingerprint", "processor_identity_fingerprint", "prompt_policy_fingerprint", "template_identity", "parser_policy"):
        if provenance.get(field) != manifest.get(field):
            raise H0LedgerContractError(f"manifest/provenance {field} mismatch")
    if manifest.get("backend") != "hf" or manifest.get("backend_mode") != "generate" or manifest.get("response_family") != "hf":
        raise H0LedgerContractError("H0 run must use the HF generate backend")
    if manifest.get("scored_artifact_materialized") is not True or manifest.get("trace_scoring_status") != "scored":
        raise H0LedgerContractError("H0 scored artifacts are incomplete")
    generation = manifest.get("generation_policy")
    config_generation = _nested(config, "generation")
    if not isinstance(generation, dict) or not isinstance(config_generation, dict):
        raise H0LedgerContractError("generation policy is missing")
    for key, expected in EXPECTED_GENERATION.items():
        observed = generation.get(key)
        if key == "do_sample" and observed is None:
            observed = config_generation.get("do_sample", False)
        if observed != expected:
            raise H0LedgerContractError(f"generation mismatch at {key}: {observed!r} != {expected!r}")
    for key in ("max_new_tokens", "repetition_penalty", "temperature", "top_p"):
        if config_generation.get(key) != EXPECTED_GENERATION[key]:
            raise H0LedgerContractError(f"infer config generation mismatch at {key}")
    if manifest.get("generation_config_fingerprint") != sha256_json(config_generation):
        raise H0LedgerContractError("generation config fingerprint does not bind infer config")
    template = manifest.get("template_identity")
    config_template = _nested(config, "template")
    if not isinstance(template, dict) or template.get("assistant_format") != EXPECTED_WRAPPERS[checkpoint] or template.get("object_ordering") != "geo_sorted_xy":
        raise H0LedgerContractError("wrapper or object ordering mismatch")
    if isinstance(config_template, dict):
        if config_template.get("assistant_format") != EXPECTED_WRAPPERS[checkpoint] or config_template.get("object_ordering") != "geo_sorted_xy":
            raise H0LedgerContractError("infer config wrapper or ordering mismatch")
        expected_prompt = sha256_json({"template": config_template, "template_id": template.get("id", "coordexp-swift-template-v1")})
        if manifest.get("prompt_policy_fingerprint") != expected_prompt:
            raise H0LedgerContractError("prompt policy fingerprint does not bind infer config")
    if manifest.get("parser_policy") != EXPECTED_PARSERS[checkpoint]:
        raise H0LedgerContractError("parser policy/wrapper mismatch")
    input_path = _nested(config, "data", "input_jsonl")
    if isinstance(input_path, str):
        try:
            if sha256_file(Path(input_path).expanduser().resolve(strict=True)) != derived_sha256:
                raise H0LedgerContractError("infer config input panel hash mismatch")
        except FileNotFoundError as exc:
            raise H0LedgerContractError("infer config input panel does not exist") from exc
    dataset_input = _nested(manifest, "dataset_identity", "input_jsonl")
    if dataset_input and input_path and str(Path(dataset_input).resolve()) != str(Path(input_path).resolve()):
        raise H0LedgerContractError("manifest/config input panel path mismatch")
    # These identity envelopes are conclusion-critical.  If a real artifact
    # supplies them, bind them back to the exact config paths; synthetic test
    # fixtures may omit the optional payloads but may not contradict them.
    identity_paths = {
        "adapter": [
            _nested(manifest, "model_identity", "adapter", "adapter_path"),
            _nested(manifest, "adapter_identity", "adapter_path"),
        ],
        "embedding_delta": [
            _nested(manifest, "model_identity", "embedding_delta", "identity", "delta_path"),
            _nested(manifest, "embedding_delta_identity", "identity", "delta_path"),
        ],
        "base": [
            _nested(manifest, "model_identity", "base", "path"),
            _nested(manifest, "model_identity", "adapter", "base_model_path"),
        ],
    }
    for key, cfg_path in (("adapter", _nested(config, "adapter", "path")), ("embedding_delta", _nested(config, "embedding_delta", "path")), ("base", _nested(config, "model", "base_model"))):
        if not isinstance(cfg_path, str):
            continue
        observed_paths = [path for path in identity_paths[key] if isinstance(path, str)]
        if observed_paths and cfg_path not in observed_paths:
            raise H0LedgerContractError(f"{key} checkpoint identity mismatch")


def _validate_run(
    artifact_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    panel_rows: list[dict[str, Any]],
    *,
    artifact_files: dict[str, Path],
) -> tuple[list[dict[str, Any]], str]:
    if manifest.get("terminal_status") != "completed" or manifest.get("failure_class") not in (None, ""):
        raise H0LedgerContractError("H0 run is incomplete or failed")
    if summary.get("terminal_status") != "completed" or summary.get("failure_class") not in (None, "") or summary.get("scored_artifact_materialized") is not True:
        raise H0LedgerContractError("H0 summary is incomplete or failed")
    expected_count = len(panel_rows)
    if summary.get("row_count") != expected_count or summary.get("raw_row_count") != expected_count or summary.get("scored_row_count") != expected_count:
        raise H0LedgerContractError("H0 row count is incomplete")
    parallelism = manifest.get("parallelism")
    if isinstance(parallelism, dict):
        if parallelism.get("merge_status") not in (None, "completed"):
            raise H0LedgerContractError("H0 merge is incomplete")
        coverage = parallelism.get("row_coverage")
        if isinstance(coverage, dict) and coverage.get("row_count") != expected_count:
            raise H0LedgerContractError("H0 row coverage count mismatch")
        shard_rows: list[str] = []
        for shard in parallelism.get("shards", []) if isinstance(parallelism.get("shards"), list) else []:
            if not isinstance(shard, dict) or shard.get("worker_status") != "completed":
                raise H0LedgerContractError("H0 contains an incomplete worker shard")
            shard_rows.extend(str(item) for item in shard.get("row_ids", []))
            for name, digest in (shard.get("artifact_hashes") or {}).items():
                shard_path = Path(str(shard.get("shard_dir", ""))) / name
                if shard_path.is_file() and digest != sha256_file(shard_path):
                    raise H0LedgerContractError(f"H0 shard artifact hash mismatch: {name}")
        if shard_rows and len(shard_rows) != len(set(shard_rows)):
            raise H0LedgerContractError("H0 shard row identity collision")
    merged = parallelism.get("merged_artifacts") if isinstance(parallelism, dict) else None
    if isinstance(merged, dict):
        for name, digest in merged.items():
            if name in artifact_files and digest != sha256_file(artifact_files[name]):
                raise H0LedgerContractError(f"H0 merged artifact hash mismatch: {name}")
    rows = _read_jsonl(artifact_files["raw"], "raw inference artifact")
    scored = _read_jsonl(artifact_files["scored"], "scored inference artifact")
    if len(rows) != expected_count or len(scored) != expected_count:
        raise H0LedgerContractError("H0 raw/scored rows are incomplete")
    panel_ids = [_row_image_id(row) for row in panel_rows]
    expected_row_ids = [_row_id(row) for row in rows]
    if len(set(expected_row_ids)) != len(expected_row_ids):
        raise H0LedgerContractError("H0 contains duplicate row IDs")
    scored_by_id = {_row_id(row): row for row in scored}
    if set(scored_by_id) != set(expected_row_ids):
        raise H0LedgerContractError("raw/scored H0 rows do not bind by row identity")
    seen_images: list[int] = []
    for row in rows:
        image = _row_image_id(row)
        seen_images.append(image)
        if image not in panel_ids:
            raise H0LedgerContractError(f"H0 row image {image} is absent from admitted panel")
        scored_row = scored_by_id[_row_id(row)]
        for field in ("row_index", "example_id", "image_path", "image_width", "image_height", "gt"):
            if field in row or field in scored_row:
                if row.get(field) != scored_row.get(field):
                    raise H0LedgerContractError(f"raw/scored H0 payload mismatch at {field}")
        if row.get("parse_status") not in {"accepted", "accepted_with_drops"} or row.get("metric_bearing") is not True:
            raise H0LedgerContractError(f"parser-invalid or non-metric H0 row: {_row_id(row)}")
        if not isinstance(scored_row.get("pred"), list):
            raise H0LedgerContractError(f"scored H0 row has no predictions list: {_row_id(row)}")
        spans: set[str] = set()
        orders: set[int] = set()
        for pred in scored_row["pred"]:
            if not isinstance(pred, dict):
                raise H0LedgerContractError("H0 prediction is not an object")
            span = pred.get("object_span_id")
            order = pred.get("generated_order")
            if span is not None and str(span) in spans:
                raise H0LedgerContractError("H0 contains duplicate prediction span identity")
            if span is not None:
                spans.add(str(span))
            if order is not None:
                if not isinstance(order, int) or order in orders:
                    raise H0LedgerContractError("H0 contains duplicate prediction order")
                orders.add(order)
    if seen_images != panel_ids:
        raise H0LedgerContractError("H0 rows are missing, reordered, or contain extra images")
    return rows, _require_sha(manifest.get("model_identity_fingerprint"), "model identity fingerprint")


def _read_tokens(path: Path, row_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    traces = _read_jsonl(path, "prediction token trace")
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        if trace.get("trace_type", "generated_token") != "generated_token":
            continue
        row_id = str(trace.get("row_id"))
        if row_id not in row_ids:
            raise H0LedgerContractError("token trace contains an unknown row")
        if not isinstance(trace.get("generated_step_index"), int) or not isinstance(trace.get("token_id"), int):
            raise H0LedgerContractError("token trace has invalid token identity")
        by_row[row_id].append(trace)
    for row_id in row_ids:
        values = sorted(by_row.get(row_id, []), key=lambda item: item["generated_step_index"])
        if not values:
            raise H0LedgerContractError(f"token trace for {row_id} is empty")
        indices = [item["generated_step_index"] for item in values]
        if indices != list(range(len(indices))):
            raise H0LedgerContractError(f"token trace for {row_id} is incomplete or colliding")
        by_row[row_id] = values
    return by_row


def _validate_image_identity(
    raw_rows: list[dict[str, Any]],
    panel_by_image: dict[int, dict[str, Any]],
    *,
    source_info: dict[str, Any],
) -> None:
    panel_parent = Path(str(source_info["path"])).parent if source_info.get("path") else None
    for row in raw_rows:
        image_id = _row_image_id(row)
        panel = panel_by_image[image_id]
        observed = row.get("image_path")
        if not isinstance(observed, str) or not observed:
            raise H0LedgerContractError(f"H0 image {image_id} has no image_path")
        observed_path = Path(observed).expanduser().resolve()
        declared = panel.get("images")
        if isinstance(declared, list) and declared:
            candidates = []
            for value in declared:
                if not isinstance(value, str) or not value:
                    raise H0LedgerContractError(f"panel image {image_id} has an invalid image reference")
                path = Path(value).expanduser()
                if not path.is_absolute() and panel_parent is not None:
                    path = panel_parent / path
                candidates.append(path.resolve())
            if observed_path not in candidates:
                raise H0LedgerContractError(f"H0 image path does not bind admitted image {image_id}")


def _validate_provenance_binding(
    provenance: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    *,
    raw_path: Path,
    scored_path: Path,
) -> None:
    for field, path in (("raw_artifact", raw_path), ("scored_artifact", scored_path)):
        artifact = provenance.get(field)
        if not isinstance(artifact, dict) or artifact.get("path") != path.name:
            raise H0LedgerContractError(f"scored provenance does not bind {path.name}")
        if artifact.get("sha256") != sha256_file(path):
            raise H0LedgerContractError(f"scored provenance hash mismatch for {path.name}")
    binding = provenance.get("row_binding")
    row_ids = [_row_id(row) for row in raw_rows]
    expected_ids_sha = hashlib.sha256(
        json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not isinstance(binding, dict) or binding.get("row_count") != len(raw_rows) or binding.get("row_ids_sha256") != expected_ids_sha:
        raise H0LedgerContractError("scored provenance row binding mismatch")


def _prediction_bbox(pred: dict[str, Any]) -> tuple[float, float, float, float]:
    return _bbox(pred.get("bbox", pred.get("bbox_xyxy")))


def _gt_bbox(value: Any, *, width: int, height: int) -> tuple[float, float, float, float]:
    """Decode canonical evaluator GT ``bbox`` values (norm1000 bins)."""

    if isinstance(value, dict):
        value = value.get("bbox", value.get("bbox_2d"))
    try:
        decoded = coord_bins_to_pixel_xyxy(
            value,
            image_width=width,
            image_height=height,
            field="gt.bbox",
        )
    except Exception as exc:
        raise H0LedgerContractError("GT bbox is not a valid norm1000 XYXY bin box") from exc
    return tuple(float(item) for item in decoded)  # type: ignore[return-value]


def _match_predictions(
    gt: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gt_items: list[dict[str, Any]] = []
    for index, obj in enumerate(gt):
        if not isinstance(obj, dict):
            raise H0LedgerContractError("artifact GT object is not an object")
        gt_items.append({"index": index, "category": _category(obj), "bbox": _gt_bbox(obj, width=width, height=height)})
    pred_items: list[dict[str, Any]] = []
    for index, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            raise H0LedgerContractError("prediction is not an object")
        score = pred.get("score", 0.0)
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise H0LedgerContractError("prediction score is invalid")
        pred_items.append({"index": index, "category": _category(pred), "bbox": _prediction_bbox(pred), "score": float(score), "raw": pred})
    matched_gt: dict[int, dict[str, Any]] = {}
    matched_pred: set[int] = set()
    unmatched: list[dict[str, Any]] = []
    # This is the per-image/category COCO matching rule: detections are sorted
    # by score (stable index tie-break), each detection can claim one GT, and a
    # GT can be claimed once.  Unknown categories naturally remain unmatched.
    for item in sorted(pred_items, key=lambda value: (-value["score"], value["index"])):
        candidates = [
            target for target in gt_items
            if target["index"] not in matched_gt
            and target["category"]
            and target["category"] == item["category"]
        ]
        scored = [(float(_iou(item["bbox"], target["bbox"])), target) for target in candidates]
        scored = [pair for pair in scored if pair[0] >= IOU_THRESHOLD]
        if scored:
            iou, target = max(scored, key=lambda pair: (pair[0], -pair[1]["index"]))
            matched_gt[target["index"]] = {
                "prediction_index": item["index"],
                "iou": round(iou, 12),
                "score": item["score"],
                "status": "tp",
                "prediction": item["raw"],
            }
            matched_pred.add(item["index"])
            continue
        duplicate_targets = [
            target for target in gt_items if target["category"] and target["category"] == item["category"]
        ]
        best_iou = max((_iou(item["bbox"], target["bbox"]) for target in duplicate_targets), default=0.0)
        status = "unmatched_duplicate_gt" if any(
            target["index"] in matched_gt and _iou(item["bbox"], target["bbox"]) >= IOU_THRESHOLD
            for target in duplicate_targets
        ) else "unmatched"
        unmatched.append({"prediction_index": item["index"], "iou": round(best_iou, 12), "score": item["score"], "status": status, "prediction": item["raw"]})
    gt_matches = []
    for target in gt_items:
        match = matched_gt.get(target["index"])
        gt_matches.append({
            "gt_index": target["index"],
            "status": "tp" if match else "fn",
            "prediction_index": match.get("prediction_index") if match else None,
            "iou": match.get("iou") if match else 0.0,
            "prediction_score": match.get("score") if match else None,
            "prediction": match.get("prediction") if match else None,
        })
    return gt_matches, unmatched


def _panel_owner_maps(source_row: dict[str, Any], derived_row: dict[str, Any], receipt: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = next((item.get("mapping") for item in receipt["source_to_derived"] if _row_image_id(item) == _row_image_id(source_row)), None)
    if not isinstance(mapping, list):
        raise H0LedgerContractError("source/derived mapping row is missing")
    by_source = {item["source_index"]: item for item in mapping}
    result = []
    for source_index, source_obj in enumerate(source_row["objects"]):
        entry = by_source.get(source_index)
        if entry is None:
            raise H0LedgerContractError("source owner has no derived index")
        derived_index = entry["derived_index"]
        result.append({
            "source_index": source_index,
            "derived_index": derived_index,
            "source_object": source_obj,
            "derived_object": derived_row["objects"][derived_index],
            "category": _category(source_obj),
            "bbox": _object_bbox(source_obj, source_row),
            "coco_ann_id": _object_ann_id(source_obj),
            "gt_owner_id": f"gt:{_row_image_id(source_row)}:{source_index}",
        })
    return result


def _is_stop_token(token: dict[str, Any]) -> bool:
    return (
        token.get("is_stop") is True
        or token.get("token_id") == IM_END_TOKEN_ID
        or token.get("token_text") == "<|im_end|>"
    )


def _first_stop_step(tokens: list[dict[str, Any]]) -> int | None:
    for token in tokens:
        if _is_stop_token(token):
            return int(token["generated_step_index"])
    return None


def _prefix_hash(
    tokens: list[dict[str, Any]],
    end_index: int | None = None,
) -> tuple[str, list[int]]:
    """Hash only generated-history suffix tokens, never the terminal stop.

    ``end_index=None`` denotes the empty/root history.  A non-empty prefix is
    always the generated suffix from step zero through ``end_index``; prompt
    tokens are deliberately not copied into the ledger because the support
    scorer extends the exact native prompt history with this suffix.
    """

    if end_index is None:
        ids: list[int] = []
    else:
        if end_index < 0 or end_index >= len(tokens):
            raise H0LedgerContractError("exact prefix end step is outside the token trace")
        selected = tokens[: end_index + 1]
        if any(_is_stop_token(item) for item in selected):
            raise H0LedgerContractError("exact prefix extends through im_end/STOP")
        ids = [int(item["token_id"]) for item in selected]
    return sha256_json(ids), ids


def _physical_rows(
    predictions: list[dict[str, Any]],
    tokens: list[dict[str, Any]],
    *,
    checkpoint: str,
) -> tuple[list[dict[str, Any]], int | None]:
    """Recover emitted row spans and native closure steps from trace metadata."""

    stop_step = _first_stop_step(tokens)
    rows: list[dict[str, Any]] = []
    for prediction_index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise H0LedgerContractError("prediction is not an object")
        order = prediction.get("generated_order")
        if not isinstance(order, int) or isinstance(order, bool) or order < 0:
            raise H0LedgerContractError("prediction generated_order is invalid")
        source = prediction.get("pred_score_source")
        selected_steps = source.get("generated_step_indices") if isinstance(source, dict) else None
        if (
            not isinstance(selected_steps, list)
            or not selected_steps
            or any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in selected_steps)
            or len(set(selected_steps)) != len(selected_steps)
        ):
            raise H0LedgerContractError("prediction span lacks generated token steps")
        start_step = min(selected_steps)
        span_end_step = max(selected_steps)
        if span_end_step >= len(tokens):
            raise H0LedgerContractError("prediction span extends beyond token trace")
        if stop_step is not None and span_end_step >= stop_step:
            raise H0LedgerContractError("prediction span extends through im_end/STOP")
        end_token = tokens[span_end_step]
        if end_token.get("token_id") != BOX_END_TOKEN_ID and end_token.get("token_text") != "<|box_end|>":
            raise H0LedgerContractError("prediction span does not close at box_end")
        closure_step = span_end_step
        closure_token = end_token
        commit_step: int | None = None
        if checkpoint == "A":
            commit_step = span_end_step + 1
            if commit_step >= len(tokens):
                raise H0LedgerContractError("A prediction span has no commit closure token")
            if stop_step is not None and commit_step >= stop_step:
                raise H0LedgerContractError("A prediction commit occurs at/after im_end/STOP")
            commit_token = tokens[commit_step]
            if commit_token.get("token_id") != COMMIT_TOKEN_ID and commit_token.get("token_text") != "<|commit|>":
                raise H0LedgerContractError("A prediction span is not followed by a commit closure")
            closure_step = commit_step
            closure_token = commit_token
        rows.append({
            "prediction_index": prediction_index,
            "generated_order": order,
            "object_span_id": prediction.get("object_span_id"),
            "row_start_step": start_step,
            "span_end_step": span_end_step,
            "closure_step": closure_step,
            "closure_token_id": closure_token.get("token_id"),
            "closure_token_text": closure_token.get("token_text"),
            "commit_step": commit_step,
        })
    rows.sort(key=lambda item: (item["generated_order"], item["prediction_index"]))
    orders = [int(row["generated_order"]) for row in rows]
    if len(orders) != len(set(orders)):
        raise H0LedgerContractError("H0 contains duplicate prediction order")
    for previous, current in zip(rows, rows[1:]):
        if current["row_start_step"] <= previous["closure_step"]:
            raise H0LedgerContractError("H0 prediction spans overlap or are not ordered")
    return rows, stop_step


def _record_rows(
    panel_source: dict[int, dict[str, Any]],
    panel_derived: dict[int, dict[str, Any]],
    receipt: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    scored_by_id: dict[str, dict[str, Any]],
    token_by_id: dict[str, list[dict[str, Any]]],
    *,
    checkpoint: str,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    unmatched_by_image: dict[int, list[dict[str, Any]]] = {}
    for raw in raw_rows:
        image_id = _row_image_id(raw)
        row_id = _row_id(raw)
        scored = scored_by_id[row_id]
        panel_row = panel_source[image_id]
        derived_row = panel_derived[image_id]
        owners = _panel_owner_maps(panel_row, derived_row, receipt)
        width = _as_int(raw.get("image_width", panel_row.get("width")), "image_width")
        height = _as_int(raw.get("image_height", panel_row.get("height")), "image_height")
        gt = raw.get("gt")
        predictions = scored.get("pred")
        if not isinstance(gt, list) or not isinstance(predictions, list) or len(gt) != len(owners):
            raise H0LedgerContractError(f"H0 GT/source owner identity mismatch for image {image_id}")
        # Bind artifact GT identities to the derived owner view by annotation
        # id first and category+pixel bbox only when that fallback is unique.
        gt_owner_for_index: dict[int, dict[str, Any]] = {}
        unused = set(range(len(owners)))
        for gt_index, gt_obj in enumerate(gt):
            ann = None
            if isinstance(gt_obj, dict):
                metadata = gt_obj.get("metadata")
                source = metadata.get("source") if isinstance(metadata, dict) else None
                ann = _object_ann_id(source) if isinstance(source, dict) else _object_ann_id(gt_obj)
            candidates = [owner for owner in owners if owner["coco_ann_id"] is not None and ann is not None and str(owner["coco_ann_id"]) == str(ann) and owner["source_index"] in unused]
            if len(candidates) != 1:
                try:
                    box = _gt_bbox(gt_obj, width=width, height=height)  # type: ignore[arg-type]
                except Exception as exc:
                    raise H0LedgerContractError(f"invalid H0 GT owner at image {image_id}") from exc
                category = _category(gt_obj)
                candidates = [owner for owner in owners if owner["source_index"] in unused and owner["category"] == category and owner["bbox"] == box]
            if len(candidates) != 1:
                raise H0LedgerContractError(f"H0 GT owner identity is missing or ambiguous at image {image_id}")
            owner = candidates[0]
            unused.remove(owner["source_index"])
            gt_owner_for_index[gt_index] = owner
        if unused:
            raise H0LedgerContractError(f"H0 GT omitted source owners at image {image_id}")
        matches, unmatched = _match_predictions(gt, predictions, width=width, height=height)
        unmatched_by_image[image_id] = unmatched
        tokens = token_by_id[row_id]
        physical_rows, stop_step = _physical_rows(predictions, tokens, checkpoint=checkpoint)
        row_by_prediction = {row["prediction_index"]: row for row in physical_rows}
        match_by_prediction = {
            match["prediction_index"]: match
            for match in matches
            if match.get("prediction_index") is not None
        }
        strict_rows = []
        for physical in physical_rows:
            match = match_by_prediction.get(physical["prediction_index"])
            if match is not None and match.get("status") == "tp":
                strict_rows.append({**physical, "owner": gt_owner_for_index[match["gt_index"]]})
        boundaries = []
        for physical in physical_rows:
            match = match_by_prediction.get(physical["prediction_index"])
            boundaries.append({
                "prediction_index": physical["prediction_index"],
                "generated_order": physical["generated_order"],
                "object_span_id": physical["object_span_id"],
                "row_start_step": physical["row_start_step"],
                "span_end_step": physical["span_end_step"],
                "closure_step": physical["closure_step"],
                "closure_token_id": physical["closure_token_id"],
                "closure_token_text": physical["closure_token_text"],
                "commit_step": physical["commit_step"],
                "match_status": match.get("status") if match is not None else "unmatched",
                "gt_owner_id": (
                    gt_owner_for_index[match["gt_index"]]["gt_owner_id"]
                    if match is not None and match.get("status") == "tp"
                    else None
                ),
            })
        for match in matches:
            owner = gt_owner_for_index[match["gt_index"]]
            pred_index = match.get("prediction_index")
            physical = row_by_prediction.get(pred_index) if pred_index is not None else None
            if match["status"] == "tp":
                if physical is None:
                    raise H0LedgerContractError("native TP has no physical prediction span")
                physical_index = next(
                    index for index, item in enumerate(physical_rows)
                    if item["prediction_index"] == physical["prediction_index"]
                )
                prior_rows = physical_rows[:physical_index]
                covered_before = [
                    item for item in strict_rows
                    if item["generated_order"] < physical["generated_order"]
                ]
                previous_emitted = prior_rows[-1] if prior_rows else None
                prefix_end = previous_emitted["closure_step"] if previous_emitted is not None else None
                natural_boundary_valid = True
                boundary_disposition = "native_tp_before_queried_row"
                prefix_semantics = TP_PREFIX_SEMANTICS
                latest_covered = covered_before[-1] if covered_before else None
                covered_ids = [item["owner"]["gt_owner_id"] for item in covered_before]
                boundary = len(covered_before)
                queried_not_covered = owner["gt_owner_id"] not in covered_ids
            else:
                # A native FN receives the earliest valid post-covered boundary:
                # the first physically matched row's closure, never im_end.
                if strict_rows:
                    latest_covered = strict_rows[0]
                    covered_ids = [latest_covered["owner"]["gt_owner_id"]]
                    prefix_end = latest_covered["closure_step"]
                    boundary = len(covered_ids)
                    natural_boundary_valid = True
                    boundary_disposition = "native_fn_after_first_covered_row"
                    prefix_semantics = FN_PREFIX_SEMANTICS
                    queried_not_covered = owner["gt_owner_id"] not in covered_ids
                else:
                    latest_covered = None
                    covered_ids = []
                    prefix_end = None
                    boundary = None
                    natural_boundary_valid = False
                    boundary_disposition = "no_valid_post_covered_boundary"
                    prefix_semantics = "no_valid_post_covered_boundary"
                    queried_not_covered = True
            prefix_hash, prefix_ids = _prefix_hash(tokens, prefix_end)
            if not natural_boundary_valid:
                prefix_hash = None
                prefix_ids = None
            generated_history_start = 0 if prefix_ids else None
            generated_history_end = prefix_end if prefix_ids else None
            due_boundary_evidence = {
                "boundary_disposition": boundary_disposition,
                "queried_owner_id": owner["gt_owner_id"],
                "queried_prediction_index": pred_index,
                "queried_generated_order": physical["generated_order"] if physical is not None else None,
                "queried_row_start_step": physical["row_start_step"] if physical is not None else None,
                "covered_owner_ids": covered_ids,
                "covered_row_count": len(covered_ids),
                "latest_covered_owner_id": latest_covered["owner"]["gt_owner_id"] if latest_covered is not None else None,
                "latest_covered_generated_order": latest_covered["generated_order"] if latest_covered is not None else None,
                "latest_covered_closure_step": latest_covered["closure_step"] if latest_covered is not None else None,
                "prefix_end_step": prefix_end,
                "stop_step": stop_step,
                "queried_owner_not_covered": queried_not_covered,
            }
            records.append({
                "unit_id": UNIT_ID,
                "checkpoint": None,
                "config_fingerprint": None,
                "history_complete": True,
                "image_id": image_id,
                "row_id": row_id,
                "row_index": raw.get("row_index"),
                "image_identity": {"image_id": image_id, "row_id": row_id, "image_path": raw.get("image_path"), "image_width": width, "image_height": height},
                "parse_status": raw.get("parse_status"),
                "parser_policy": raw.get("parser_policy"),
                "valid_prediction_count": raw.get("valid_prediction_count"),
                "dropped_prediction_count": raw.get("dropped_prediction_count"),
                "decode_stop_reason": raw.get("decode_stop_reason"),
                "source_panel_object_index": owner["source_index"],
                "derived_panel_object_index": owner["derived_index"],
                "gt_owner_id": owner["gt_owner_id"],
                "coco_ann_id": owner["coco_ann_id"],
                "category_name": owner["category"],
                "bbox_pixel_xyxy": [int(round(item)) for item in owner["bbox"]],
                "cohort_stratum": "image2299" if image_id == EXPECTED_IMAGE_2299 else "legacy12",
                "native_tp": match["status"] == "tp",
                "native_fn": match["status"] == "fn",
                "strict_complete_row": match["status"] == "tp",
                "natural_boundary_valid": natural_boundary_valid,
                "natural_boundary": boundary,
                "due_boundary_index": boundary,
                "boundary_disposition": boundary_disposition,
                "prefix_semantics": prefix_semantics,
                "generated_history_start_step": generated_history_start,
                "generated_history_end_step": generated_history_end,
                "excludes_stop": True,
                "covered_owner_ids": covered_ids,
                "latest_covered_owner_id": (
                    latest_covered["owner"]["gt_owner_id"] if latest_covered is not None else None
                ),
                "queried_owner_not_covered": queried_not_covered,
                "is_earliest_eligible_boundary": (
                    True if match["status"] == "fn" and natural_boundary_valid else False
                    if match["status"] == "fn" else None
                ),
                "due_boundary_evidence": due_boundary_evidence,
                "generated_history_stop_step": stop_step,
                "exact_prefix_sha256": prefix_hash,
                "exact_prefix_token_ids": prefix_ids,
                "generated_row_boundaries": boundaries,
                "match_evidence": {"status": match["status"], "prediction_index": pred_index, "iou": match["iou"], "iou_threshold": IOU_THRESHOLD, "prediction_score": match.get("prediction_score"), "matching_policy": "canonical_coco_bbox_category_score_desc_one_to_one"},
                "verified_support_claim": False,
                "support_status": "not_measured",
                "unmatched_predictions": unmatched,
            })
        if not matches:
            raise H0LedgerContractError(f"H0 image {image_id} has no owner records")
    return records, unmatched_by_image


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise H0LedgerContractError(f"existing output is not identical: {path}")
        return
    path.write_bytes(payload)


def build_h0_ledger(
    artifact_dir: str | Path,
    infer_config: str | Path | dict[str, Any] | None,
    source_panel: str | Path | dict[str, Any] | list[Any],
    derived_panel: str | Path | dict[str, Any] | list[Any],
    derived_receipt: str | Path | dict[str, Any],
    *,
    checkpoint: str | None = None,
    output: str | Path | None = None,
    records_output: str | Path | None = None,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one completed H0 artifact directory and build its ledger."""

    artifact_root = Path(artifact_dir).expanduser().resolve(strict=True)
    if not artifact_root.is_dir():
        raise H0LedgerContractError("artifact_dir must be a directory")
    required = {
        "manifest": artifact_root / "run_manifest.json",
        "summary": artifact_root / "summary.json",
        "provenance": artifact_root / "gt_vs_pred_scored.jsonl.provenance.json",
        "raw": artifact_root / "gt_vs_pred.jsonl",
        "scored": artifact_root / "gt_vs_pred_scored.jsonl",
        "image_plan": artifact_root / "image_plan.jsonl",
        "token_trace": artifact_root / "pred_token_trace.jsonl",
    }
    for name, path in required.items():
        if not path.is_file():
            raise H0LedgerContractError(f"H0 artifact is missing {name}: {path.name}")
    manifest = _read_json(required["manifest"], "run manifest")
    summary = _read_json(required["summary"], "run summary")
    provenance = _read_json(required["provenance"], "scored provenance")
    source_value, source_info = _source_value(source_panel, label="source panel")
    derived_value, derived_info = _source_value(derived_panel, label="derived panel")
    receipt_value, receipt_info = _source_value(derived_receipt, label="derived receipt")
    if not isinstance(receipt_value, dict):
        raise H0LedgerContractError("derived receipt must be an object")
    source_rows = _panel_rows(source_value, "source panel")
    source_by_image, derived_by_image = _validate_derived(
        source_value, derived_value, receipt_value,
        source_sha256=source_info["sha256"], derived_sha256=derived_info["sha256"],
    )
    # The run's resolved config is immutable authority.  A caller-supplied
    # leaf is accepted only as an authored-value cross-check; its current
    # loader fingerprint is never recomputed into the H0 identity.
    resolved_config_path = artifact_root / "configs" / "resolved.json"
    if not resolved_config_path.is_file():
        raise H0LedgerContractError("H0 artifact is missing immutable configs/resolved.json")
    config, config_fingerprint, resolved_info = _config_payload(resolved_config_path)
    config_info: dict[str, Any] = {"resolved": resolved_info}
    if infer_config is not None:
        leaf_config, _leaf_fingerprint, leaf_info = _config_payload(infer_config)
        _assert_leaf_semantics(leaf_config, config)
        config_info["leaf_cross_check"] = leaf_info
    inferred_checkpoint = _infer_checkpoint(config, manifest, artifact_root)
    declared_checkpoint = _normalise_checkpoint(checkpoint)
    if checkpoint is not None and declared_checkpoint is None:
        raise H0LedgerContractError(f"invalid checkpoint: {checkpoint}")
    if declared_checkpoint and declared_checkpoint != inferred_checkpoint:
        raise H0LedgerContractError("checkpoint identity mismatch")
    _validate_config_and_identity(
        config, config_fingerprint, manifest, provenance,
        checkpoint=inferred_checkpoint,
        source_sha256=source_info["sha256"], derived_sha256=derived_info["sha256"],
    )
    raw_rows, model_identity_fingerprint = _validate_run(
        artifact_root, manifest, summary, source_rows, artifact_files=required,
    )
    _validate_provenance_binding(
        provenance,
        raw_rows,
        raw_path=required["raw"],
        scored_path=required["scored"],
    )
    _validate_image_identity(raw_rows, source_by_image, source_info=source_info)
    image_plan = _read_jsonl(required["image_plan"], "image plan")
    if len(image_plan) != len(source_rows) or [_row_id(item) for item in image_plan] != [_row_id(item) for item in raw_rows]:
        raise H0LedgerContractError("image plan rows are missing or reordered")
    for item in image_plan:
        if item.get("status") != "ok" or item.get("error") not in (None, ""):
            raise H0LedgerContractError("image plan contains an invalid row")
    scored_rows = _read_jsonl(required["scored"], "scored artifact")
    scored_by_id = {_row_id(row): row for row in scored_rows}
    token_by_id = _read_tokens(required["token_trace"], [_row_id(row) for row in raw_rows])
    records, unmatched_by_image = _record_rows(
        source_by_image, derived_by_image, receipt_value, raw_rows, scored_by_id, token_by_id,
        checkpoint=inferred_checkpoint,
    )
    image_plan_by_id = {_row_image_id(item): item for item in image_plan}
    for record in records:
        record["checkpoint"] = inferred_checkpoint
        record["config_fingerprint"] = config_fingerprint
        record["run_kind"] = "native_h0"
        record["source_panel_sha256"] = source_info["sha256"]
        record["derived_panel_sha256"] = derived_info["sha256"]
        plan = image_plan_by_id[int(record["image_id"])]
        record["image_token_span"] = plan.get("backend_image_placeholder_ranges")
        record["image_plan_identity"] = {
            key: plan.get(key)
            for key in (
                "image_content_sha256",
                "executed_media_sha256",
                "expected_image_grid_thw",
                "observed_image_grid_thw",
                "merged_visual_tokens",
                "backend_prompt_token_count",
            )
            if key in plan
        }
        record["runtime_envelope"] = {
            "checkpoint": inferred_checkpoint,
            "config_fingerprint": config_fingerprint,
            "generation_config_fingerprint": manifest.get("generation_config_fingerprint"),
            "prompt_policy_fingerprint": manifest.get("prompt_policy_fingerprint"),
            "model_identity_fingerprint": manifest.get("model_identity_fingerprint"),
            "processor_identity_fingerprint": manifest.get("processor_identity_fingerprint"),
            "wrapper": EXPECTED_WRAPPERS[inferred_checkpoint],
            "parser_policy": EXPECTED_PARSERS[inferred_checkpoint],
            "backend": manifest.get("backend"),
            "backend_mode": manifest.get("backend_mode"),
        }
    records.sort(key=lambda item: (int(item["row_index"]), int(item["source_panel_object_index"])))
    legacy = [record for record in records if int(record["image_id"]) != EXPECTED_IMAGE_2299]
    image_2299 = [record for record in records if int(record["image_id"]) == EXPECTED_IMAGE_2299]
    summary_doc = {
        "record_count": len(records),
        "matched_tp_count": sum(item["native_tp"] for item in records),
        "native_fn_count": sum(item["native_fn"] for item in records),
        "unmatched_prediction_count": sum(len(items) for items in unmatched_by_image.values()),
        "legacy12": {"record_count": len(legacy), "tp_count": sum(item["native_tp"] for item in legacy), "fn_count": sum(item["native_fn"] for item in legacy)},
        "image2299": {"record_count": len(image_2299), "tp_count": sum(item["native_tp"] for item in image_2299), "fn_count": sum(item["native_fn"] for item in image_2299)},
    }
    artifact_hashes = {name: sha256_file(path) for name, path in required.items()}
    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "run_kind": "native_h0",
        "arm": "native",
        "checkpoint": inferred_checkpoint,
        "config_fingerprint": config_fingerprint,
        "source_panel_sha256": source_info["sha256"],
        "derived_panel_sha256": derived_info["sha256"],
        "history_complete": True,
        "native_outcome_only": True,
        "verified_support_claim": False,
        "matching_policy": {"evaluator": "coordexp_swift_detection_coco_bbox_v1", "category_aware": True, "iou_threshold": IOU_THRESHOLD, "one_to_one": True, "score_order": "descending_score_stable_index"},
        "checkpoint_identity": {"checkpoint": inferred_checkpoint, "model_identity_fingerprint": model_identity_fingerprint, "wrapper": EXPECTED_WRAPPERS[inferred_checkpoint], "parser_policy": EXPECTED_PARSERS[inferred_checkpoint]},
        "runtime_envelope": {
            "generation_config_fingerprint": manifest.get("generation_config_fingerprint"),
            "generation_policy": manifest.get("generation_policy"),
            "prompt_policy_fingerprint": manifest.get("prompt_policy_fingerprint"),
            "template_identity": manifest.get("template_identity"),
            "parser_policy": manifest.get("parser_policy"),
            "backend": manifest.get("backend"),
            "backend_mode": manifest.get("backend_mode"),
            "response_family": manifest.get("response_family"),
            "model_identity": manifest.get("model_identity"),
            "adapter_identity": manifest.get("adapter_identity"),
            "embedding_delta_identity": manifest.get("embedding_delta_identity"),
            "processor_identity": manifest.get("processor_identity"),
            "tokenizer_identity": manifest.get("tokenizer_identity"),
            "backend_session": manifest.get("backend_session"),
            "prompt_trace": manifest.get("prompt_trace"),
        },
        "source_panel": source_info,
        "derived_panel": derived_info,
        "derived_receipt": receipt_info,
        "infer_config": config_info,
        "artifact_dir": str(artifact_root),
        "artifact_hashes": artifact_hashes,
        "summary": summary_doc,
        "records": records,
        "unmatched_predictions": [{"image_id": image, "predictions": values} for image, values in sorted(unmatched_by_image.items())],
    }
    receipt_doc = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "complete",
        "run_kind": "native_h0",
        "checkpoint": inferred_checkpoint,
        "config_fingerprint": config_fingerprint,
        "source_panel_sha256": source_info["sha256"],
        "derived_panel_sha256": derived_info["sha256"],
        "derived_receipt_sha256": receipt_info["sha256"],
        "artifact_hashes": artifact_hashes,
        "record_count": len(records),
        "summary": summary_doc,
        "matching_policy": envelope["matching_policy"],
        "runtime_identity": envelope["runtime_envelope"],
        "verified_support_claim": False,
        "deterministic_records_sha256": sha256_json(records),
    }
    records_bytes = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    envelope["records_jsonl_sha256"] = sha256_bytes(records_bytes)
    if records_output is None and output is not None:
        output_path = Path(output).expanduser().resolve()
        records_output = output_path.with_name(output_path.stem + ".records.jsonl")
    if output is not None:
        _write_immutable(Path(output).expanduser().resolve(), canonical_json_bytes(envelope) + b"\n")
    if records_output is not None:
        _write_immutable(Path(records_output).expanduser().resolve(), records_bytes)
    if receipt_output is None and output is not None:
        receipt_output = Path(output).expanduser().resolve().with_name(Path(output).stem + ".receipt.json")
    if receipt_output is not None:
        _write_immutable(Path(receipt_output).expanduser().resolve(), canonical_json_bytes(receipt_doc) + b"\n")
    return {"ledger": envelope, "receipt": receipt_doc, "records": records}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--infer-config", type=Path, help="optional authored leaf; resolved artifact remains authoritative")
    parser.add_argument("--source-panel", required=True, type=Path)
    parser.add_argument("--derived-panel", required=True, type=Path)
    parser.add_argument("--derived-receipt", required=True, type=Path)
    parser.add_argument("--checkpoint", choices=("S", "A"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-output", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_h0_ledger(
        args.artifact_dir,
        args.infer_config,
        args.source_panel,
        args.derived_panel,
        args.derived_receipt,
        checkpoint=args.checkpoint,
        output=args.output,
        records_output=args.records_output,
        receipt_output=args.receipt,
    )
    print(json.dumps(result["receipt"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fusion helpers for multi-dataset dense-caption training.

CoordExp supports configuring multi-dataset training via a separate "fusion
config" file (YAML/JSON). The schema is intentionally compatible with the
upstream Qwen3-VL containers (`targets` + optional `sources`), but CoordExp v1
**treats targets and sources identically** (no target/source semantic split).
"""

from __future__ import annotations

import copy
import json
import random
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.common.io import load_jsonl

from .contracts import ConversationRecord
from .fusion_types import DatasetSpec, FusionDatasetSpec
from .wrappers import available_template_ids, build_dataset_spec


def _normalize_extends(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _load_fusion_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pyyaml is required to parse fusion configs; install pyyaml"
            ) from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)

    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("fusion config must be a mapping")
    normalized = dict(payload)
    return _resolve_dataset_jsonl_paths(normalized, base_dir=path.resolve().parent)


def _maybe_resolve_dataset_path(value: Any, *, base_dir: Path) -> Any:
    """Resolve a dataset path entry according to CoordExp conventions.

    - Absolute paths are kept as-is.
    - Relative paths that begin with ./ or ../ are resolved relative to the
      fusion config file directory (so extends behave intuitively).
    - Other relative paths (e.g. public_data/...) are treated as repo-root/CWD
      relative; the launcher runs from the repo root by convention.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        return value

    trimmed = value.strip()
    if not trimmed:
        return trimmed

    path = Path(trimmed)
    if path.is_absolute():
        return trimmed

    if trimmed.startswith("./") or trimmed.startswith("../"):
        return str((base_dir / path).resolve())

    return trimmed


def _resolve_dataset_jsonl_paths(payload: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Resolve dataset JSONL paths within a single fusion payload."""

    for container_key in ("targets", "sources"):
        entries_raw = payload.get(container_key)
        if entries_raw is None:
            continue
        if not isinstance(entries_raw, Sequence) or isinstance(entries_raw, (str, bytes)):
            # Leave validation to the strict parser.
            continue

        normalized_entries: list[Any] = []
        for entry_raw in entries_raw:
            if not isinstance(entry_raw, Mapping):
                normalized_entries.append(entry_raw)
                continue
            entry = dict(entry_raw)
            params = entry.get("params")
            if isinstance(params, Mapping):
                params_dict = dict(params)
                if "train_jsonl" in params_dict:
                    params_dict["train_jsonl"] = _maybe_resolve_dataset_path(
                        params_dict.get("train_jsonl"), base_dir=base_dir
                    )
                if "val_jsonl" in params_dict:
                    params_dict["val_jsonl"] = _maybe_resolve_dataset_path(
                        params_dict.get("val_jsonl"), base_dir=base_dir
                    )
                entry["params"] = params_dict
            else:
                if "train_jsonl" in entry:
                    entry["train_jsonl"] = _maybe_resolve_dataset_path(
                        entry.get("train_jsonl"), base_dir=base_dir
                    )
                if "val_jsonl" in entry:
                    entry["val_jsonl"] = _maybe_resolve_dataset_path(
                        entry.get("val_jsonl"), base_dir=base_dir
                    )
            normalized_entries.append(entry)
        payload[container_key] = normalized_entries

    return payload


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _dataset_entry_id(entry: Mapping[str, Any], *, field_name: str) -> str:
    dataset_id = entry.get("name") or entry.get("dataset")
    if dataset_id is None:
        raise ValueError(f"{field_name} entry must include 'dataset' (and optional 'name')")
    return str(dataset_id)


def _ensure_entry_list(value: object, *, field_name: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError(f"fusion {field_name} must be an iterable")
    entries = list(value)
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{field_name} entry must be a mapping")
    return cast(list[Mapping[str, Any]], entries)


def _combined_entries(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    # Treat both keys as the same logical list in CoordExp.
    targets = _ensure_entry_list(payload.get("targets"), field_name="targets")
    sources = _ensure_entry_list(payload.get("sources"), field_name="sources")
    return [*targets, *sources]


def _merge_dataset_entries(
    base_payload: Mapping[str, Any],
    override_payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    base_entries = _combined_entries(base_payload)
    override_entries = _combined_entries(override_payload)

    if not base_entries:
        return override_entries

    base_map: dict[str, Mapping[str, Any]] = {}
    for entry in base_entries:
        key = _dataset_entry_id(entry, field_name="targets/sources")
        if key in base_map:
            raise ValueError(f"Duplicate dataset ID in base fusion config: {key}")
        base_map[key] = entry

    override_map: dict[str, Mapping[str, Any]] = {}
    for entry in override_entries:
        key = _dataset_entry_id(entry, field_name="targets/sources")
        if key in override_map:
            raise ValueError(f"Duplicate dataset ID in override fusion config: {key}")
        override_map[key] = entry

    merged_entries: list[Mapping[str, Any]] = []
    for entry in base_entries:
        key = _dataset_entry_id(entry, field_name="targets/sources")
        if key in override_map:
            merged_entries.append(_deep_merge_dicts(entry, override_map[key]))
        else:
            merged_entries.append(entry)

    for entry in override_entries:
        key = _dataset_entry_id(entry, field_name="targets/sources")
        if key not in base_map:
            merged_entries.append(entry)

    return merged_entries


def _merge_fusion_payload(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)

    # Merge non-dataset keys first (deep-merge mappings; override scalars).
    for key, value in override.items():
        if key in {"targets", "sources"}:
            continue
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = value

    # Merge dataset entries (targets + sources treated uniformly).
    merged_entries = _merge_dataset_entries(base, override)
    if merged_entries:
        merged["targets"] = merged_entries
        merged.pop("sources", None)  # keep canonical form

    return merged


def _load_fusion_with_extends(
    path: Path,
    *,
    stack: set[Path] | None = None,
    memo: dict[Path, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load a fusion config, resolving extends with DAG-safe cycle detection.

    We must allow diamond-shaped DAGs (A extends B and C, both extend D)
    without treating D as a cycle. Use a recursion stack for cycle detection
    and a memo table for reuse.
    """

    abs_path = path.resolve()
    memo = {} if memo is None else memo
    if abs_path in memo:
        return copy.deepcopy(memo[abs_path])

    stack = set() if stack is None else stack
    if abs_path in stack:
        raise ValueError(f"Cyclic fusion config inheritance detected at: {abs_path}")

    stack.add(abs_path)
    try:
        payload = _load_fusion_payload(abs_path)
        extends_value = payload.pop("extends", None)
        if extends_value is None:
            extends_value = payload.pop("inherit", None)

        merged_base: dict[str, Any] = {}
        for base_ref in _normalize_extends(extends_value):
            base_path = Path(base_ref)
            if not base_path.is_absolute():
                base_path = (abs_path.parent / base_path).resolve()
            base_payload = _load_fusion_with_extends(base_path, stack=stack, memo=memo)
            merged_base = _merge_fusion_payload(merged_base, base_payload)

        merged = _merge_fusion_payload(merged_base, payload)
        memo[abs_path] = merged
        return copy.deepcopy(merged)
    finally:
        stack.remove(abs_path)


@dataclass(frozen=True)
class FusionConfig:
    """Runtime fusion config after resolving `extends` and parsing entries."""

    targets: tuple[FusionDatasetSpec, ...]
    # Accepted for schema compatibility; treated the same as `targets`.
    sources: tuple[FusionDatasetSpec, ...] = ()

    @property
    def datasets(self) -> tuple[FusionDatasetSpec, ...]:
        return self.targets + self.sources

    @classmethod
    def from_file(cls, path: str) -> "FusionConfig":
        path_obj = Path(path)
        payload = _load_fusion_with_extends(path_obj)

        if "target" in payload:
            raise ValueError("fusion config must define 'targets' (list); 'target' is not supported.")

        targets_entries = _ensure_entry_list(payload.get("targets"), field_name="targets")
        if not targets_entries:
            raise ValueError("fusion config requires at least one dataset entry in targets/sources")

        specs: list[FusionDatasetSpec] = []
        for entry in targets_entries:
            specs.append(cls._parse_dataset_entry(entry))

        cls._validate_unique_ids(specs)

        # Canonical: everything lives in targets (sources accepted at input).
        return cls(targets=tuple(specs), sources=())

    @staticmethod
    def _validate_unique_ids(specs: Sequence[FusionDatasetSpec]) -> None:
        seen: set[str] = set()
        for spec in specs:
            if spec.name in seen:
                raise ValueError(f"Duplicate dataset ID in fusion config: {spec.name}")
            seen.add(spec.name)

    @staticmethod
    def _parse_dataset_entry(entry: Mapping[str, Any]) -> FusionDatasetSpec:
        if not isinstance(entry, Mapping):
            raise ValueError("dataset entry must be a mapping")

        dataset_key_raw = entry.get("dataset")
        if dataset_key_raw is None:
            raise ValueError("dataset entry must include 'dataset'")
        dataset_key = str(dataset_key_raw)

        name_override_raw = entry.get("name")
        name_override = str(name_override_raw) if name_override_raw is not None else None

        params_raw = entry.get("params")
        if params_raw is None:
            params: dict[str, Any] = {}
        elif isinstance(params_raw, Mapping):
            params = dict(params_raw)
        else:
            raise TypeError("dataset params must be a mapping if provided")

        # Accept both "flat" fields and the upstream "params" mapping for
        # compatibility; params values win when both are present.
        for key in (
            "train_jsonl",
            "val_jsonl",
            "template",
            "poly_fallback",
            "poly_max_points",
            "augmentation_enabled",
            "curriculum_enabled",
            "max_objects_per_image",
            "user_prompt",
            "system_prompt",
            "seed",
            "mode",
            "sample_without_replacement",
        ):
            if key not in params:
                params[key] = entry.get(key)

        # Required: template must be present and known (typo guard).
        template_raw = params.get("template")
        if template_raw is None:
            raise ValueError("dataset entry must include 'template'")
        template = str(template_raw).strip()
        if not template:
            raise ValueError("dataset entry template must be a non-empty string")
        known_templates = available_template_ids()
        if template not in known_templates:
            known = ", ".join(sorted(known_templates))
            raise ValueError(f"Unknown fusion template '{template}'. Known templates: {known}")

        # Optional: ratio defaults to 1.0
        ratio_raw = entry.get("ratio", 1.0)
        if ratio_raw is None:
            ratio = 1.0
        elif isinstance(ratio_raw, str) and not ratio_raw.strip():
            ratio = 1.0
        else:
            if isinstance(ratio_raw, bool):
                raise ValueError("ratio must be numeric")
            try:
                ratio = float(ratio_raw)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError("ratio must be numeric") from exc
        if ratio < 0:
            raise ValueError("ratio must be non-negative")

        # Ensure wrapper receives the validated template.
        params = dict(params)
        params["template"] = template

        try:
            base_spec = build_dataset_spec(dataset_key, name=name_override, params=params)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

        return FusionDatasetSpec(
            key=base_spec.key,
            name=base_spec.name,
            train_jsonl=base_spec.train_jsonl,
            template=base_spec.template,
            domain=base_spec.domain,
            supports_augmentation=base_spec.supports_augmentation,
            supports_curriculum=base_spec.supports_curriculum,
            mode=base_spec.mode,
            poly_fallback=base_spec.poly_fallback,
            poly_max_points=base_spec.poly_max_points,
            val_jsonl=base_spec.val_jsonl,
            max_objects_per_image=base_spec.max_objects_per_image,
            prompt_user=base_spec.prompt_user,
            prompt_system=base_spec.prompt_system,
            seed=base_spec.seed,
            sample_without_replacement=base_spec.sample_without_replacement,
            ratio=ratio,
        )


def _annotate_record(record: ConversationRecord, spec: DatasetSpec) -> ConversationRecord:
    annotated = cast(ConversationRecord, dict(copy.deepcopy(record)))
    metadata = annotated.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["_fusion_source"] = spec.name
    metadata["_fusion_template"] = spec.template
    metadata["_fusion_domain"] = spec.domain
    annotated["metadata"] = metadata
    return annotated


def build_fused_jsonl(
    config: FusionConfig,
    output_path: str,
    *,
    seed: int = 2025,
    shuffle: bool = True,
) -> Path:
    """Offline helper: materialize a single JSONL by sampling per-dataset quotas.

    This is NOT required for training (training can read fusion configs directly)
    but can be useful for debugging or for external tools.
    """

    rng = random.Random(seed)
    fused: list[ConversationRecord] = []

    for spec in config.datasets:
        records = load_jsonl(str(spec.train_jsonl), resolve_relative=True)
        pool_len = len(records)
        quota = round(pool_len * float(spec.ratio))
        if pool_len <= 0 or quota <= 0:
            continue

        stable = zlib.crc32(spec.name.encode("utf-8")) & 0xFFFFFFFF
        rng_local = random.Random((seed ^ stable) & 0xFFFFFFFF)
        indices = list(range(pool_len))
        if pool_len > 1:
            rng_local.shuffle(indices)
        if quota <= pool_len:
            sampled = indices[:quota]
        else:
            sampled = indices[:]
            if not spec.sample_without_replacement:
                sampled.extend(
                    rng_local.randrange(pool_len) for _ in range(quota - pool_len)
                )

        for idx in sampled:
            fused.append(_annotate_record(records[idx], spec))

    if shuffle and len(fused) > 1:
        rng.shuffle(fused)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with output_path_obj.open("w", encoding="utf-8") as fout:
        for record in fused:
            fout.write(json.dumps(record, ensure_ascii=False))
            fout.write("\n")

    return output_path_obj


__all__ = [
    "DatasetSpec",
    "FusionDatasetSpec",
    "FusionConfig",
    "build_fused_jsonl",
]

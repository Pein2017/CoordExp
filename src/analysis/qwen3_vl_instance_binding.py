"""Mechanism-study helpers for Qwen3-VL coord-token instance binding.

This module intentionally starts with cheap, testable contracts: path
canonicalization, checkpoint audits, case mining, sharding, and lightweight
summary metrics. GPU-heavy hidden-state/probe/patching stages build on these
contracts instead of re-deciding path semantics at runtime.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

DEFAULT_MODEL_LAYER_READS: tuple[int, ...] = (
    0,
    1,
    12,
    13,
    24,
    25,
    26,
    27,
    -4,
    -3,
    -2,
    -1,
)

REQUIRED_POSITION_ROLES: tuple[str, ...] = (
    "desc_end",
    "desc_closing_quote",
    "field_delimiter",
    "bbox_key",
    "bbox_open_bracket",
    "pre_x1",
    "post_x1",
    "post_y1",
)

_HEAVY_RELATIVE_ROOTS = ("output_remote", "output", "public_data")
_COORD_TOKEN_PREFIX = "<|coord_"
_COORD_TOKEN_SUFFIX = "|>"
_SLOT_TO_INDEX = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
_SLOT_TO_PREDICTION_ROLE = {"x1": "pre_x1", "y1": "post_x1", "x2": "post_y1"}


@dataclass(frozen=True)
class RuntimePathConfig:
    checkpoint: str
    dataset_jsonls: tuple[str, ...]
    artifact_root: str


@dataclass(frozen=True)
class ResolvedRuntimePaths:
    checkpoint: Path
    dataset_jsonls: tuple[Path, ...]
    artifact_root: Path


@dataclass(frozen=True)
class ExecutionConfig:
    gpu_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    shard_count: int = 8
    per_gpu_generation_batch_size: int = 8
    per_gpu_teacher_forced_batch_size: int = 8


@dataclass(frozen=True)
class CaseSelectionConfig:
    priority_descs: tuple[str, ...] = (
        "book",
        "person",
        "chair",
        "baseball bat",
        "bowl",
        "sheep",
    )
    max_cases: int = 32
    min_same_desc_candidates: int = 2
    include_sparse_controls: bool = True
    max_sparse_controls: int = 4


@dataclass(frozen=True)
class MultimodalityConfig:
    coordinate_slots: tuple[str, ...] = ("x1", "x2")
    neighbor_radius: int = 3
    top_k: int = 12


@dataclass(frozen=True)
class PatchingConfig:
    spans: tuple[str, ...] = (
        "current_desc",
        "schema_context",
        "previous_geometry",
        "previous_x1_y1",
    )
    model_layers: tuple[int, ...] = (24, 25, 26, 27)
    attenuation_scale: float = 0.0
    max_cases: int = 64


@dataclass(frozen=True)
class RolloutConfig:
    max_cases: int = 64
    temperature: float = 0.0
    top_p: float = 0.9
    max_new_tokens: int = 1536
    repetition_penalty: float = 1.05
    batch_size: int = 8
    seed: int = 42


@dataclass(frozen=True)
class StudyConfig:
    config_path: Path
    paths: ResolvedRuntimePaths
    execution: ExecutionConfig
    case_selection: CaseSelectionConfig
    multimodality: MultimodalityConfig
    patching: PatchingConfig
    rollout: RolloutConfig


@dataclass(frozen=True)
class InstanceBindingCase:
    case_id: str
    source_jsonl: Path
    record_index: int
    image_id: str
    image_path: Path
    width: int
    height: int
    desc: str
    target_object_index: int
    candidate_object_indices: tuple[int, ...]
    cohort: str
    selection_reason: str
    objects: tuple[Mapping[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_jsonl": str(self.source_jsonl),
            "record_index": self.record_index,
            "image_id": self.image_id,
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
            "desc": self.desc,
            "target_object_index": self.target_object_index,
            "candidate_object_indices": list(self.candidate_object_indices),
            "cohort": self.cohort,
            "selection_reason": self.selection_reason,
            "objects": [dict(obj) for obj in self.objects],
        }


def _common_repo_root(repo_root: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return repo_root
    common_dir = Path(completed.stdout.strip())
    if common_dir.name == ".git":
        return common_dir.parent
    return common_dir


def _is_relative_heavy_path(path: Path) -> bool:
    return (
        not path.is_absolute()
        and len(path.parts) > 0
        and path.parts[0] in _HEAVY_RELATIVE_ROOTS
    )


def _resolve_heavy_path(
    raw_path: str,
    *,
    repo_root: Path,
    shared_root: Path,
    kind: str,
    must_exist: bool = False,
) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        resolved = path
    elif _is_relative_heavy_path(path):
        shared_candidate = shared_root / path
        worktree_candidate = repo_root / path
        if (
            kind == "checkpoint"
            and worktree_candidate.exists()
            and not shared_candidate.exists()
        ):
            raise ValueError(
                f"Refusing worktree-local checkpoint shadow: {worktree_candidate}. "
                f"Use the shared-root checkpoint under {shared_root}."
            )
        resolved = (
            shared_candidate
            if shared_candidate.exists() or kind in {"checkpoint", "artifact_root"}
            else worktree_candidate
        )
    else:
        candidate = repo_root / path
        resolved = candidate if candidate.exists() else shared_root / path
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{kind} path does not exist: {resolved}")
    return resolved


def resolve_runtime_paths(
    config: RuntimePathConfig,
    *,
    repo_root: Path,
    shared_root: Path | None = None,
    require_existing_inputs: bool = False,
) -> ResolvedRuntimePaths:
    """Resolve heavyweight paths against the shared CoordExp root.

    Worktrees contain tracked source/config skeletons, not the heavyweight
    checkpoint, prepared COCO data, or durable output roots. Relative paths
    beginning with output/output_remote/public_data are therefore anchored to
    the shared root, not the worktree root.
    """

    repo_root = repo_root.resolve()
    shared = (shared_root or _common_repo_root(repo_root)).resolve()
    checkpoint = _resolve_heavy_path(
        config.checkpoint,
        repo_root=repo_root,
        shared_root=shared,
        kind="checkpoint",
        must_exist=require_existing_inputs,
    )
    dataset_jsonls = tuple(
        _resolve_heavy_path(
            item,
            repo_root=repo_root,
            shared_root=shared,
            kind="dataset",
            must_exist=require_existing_inputs,
        )
        for item in config.dataset_jsonls
    )
    artifact_root = _resolve_heavy_path(
        config.artifact_root,
        repo_root=repo_root,
        shared_root=shared,
        kind="artifact_root",
        must_exist=False,
    )
    return ResolvedRuntimePaths(
        checkpoint=checkpoint,
        dataset_jsonls=dataset_jsonls,
        artifact_root=artifact_root,
    )


def audit_checkpoint_surface(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    has_coord_tokens = (checkpoint / "coord_tokens.json").is_file()
    has_config = (checkpoint / "config.json").is_file()
    has_tokenizer = (checkpoint / "tokenizer.json").is_file()
    has_model_index = (checkpoint / "model.safetensors.index.json").is_file()
    shard_count = len(list(checkpoint.glob("model-*.safetensors")))
    has_adapter_config = (checkpoint / "adapter_config.json").is_file()
    if (
        has_coord_tokens
        and has_config
        and has_tokenizer
        and (has_model_index or shard_count > 0)
    ):
        surface = "coord_tokens_full_model"
    elif has_adapter_config:
        surface = "adapter_checkpoint"
    elif has_coord_tokens:
        surface = "coord_tokens_incomplete"
    else:
        surface = "unknown"
    return {
        "checkpoint": str(checkpoint),
        "surface": surface,
        "has_coord_tokens_json": has_coord_tokens,
        "has_config_json": has_config,
        "has_tokenizer_json": has_tokenizer,
        "has_model_index": has_model_index,
        "model_shard_count": shard_count,
        "has_adapter_config": has_adapter_config,
    }


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise TypeError(f"JSONL row {idx} in {path} is not an object")
            yield idx, payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            count += 1
    return count


def _write_resolved_config(config: StudyConfig) -> None:
    payload = {
        "config_path": str(config.config_path),
        "run": {"artifact_root": str(config.paths.artifact_root)},
        "model": {"checkpoint": str(config.paths.checkpoint)},
        "data": {"dataset_jsonls": [str(path) for path in config.paths.dataset_jsonls]},
        "execution": {
            "gpu_ids": list(config.execution.gpu_ids),
            "shard_count": config.execution.shard_count,
            "per_gpu_generation_batch_size": config.execution.per_gpu_generation_batch_size,
            "per_gpu_teacher_forced_batch_size": config.execution.per_gpu_teacher_forced_batch_size,
        },
        "case_selection": {
            "priority_descs": list(config.case_selection.priority_descs),
            "max_cases": config.case_selection.max_cases,
            "min_same_desc_candidates": config.case_selection.min_same_desc_candidates,
            "include_sparse_controls": config.case_selection.include_sparse_controls,
            "max_sparse_controls": config.case_selection.max_sparse_controls,
        },
        "multimodality": {
            "coordinate_slots": list(config.multimodality.coordinate_slots),
            "neighbor_radius": config.multimodality.neighbor_radius,
            "top_k": config.multimodality.top_k,
        },
        "patching": {
            "spans": list(config.patching.spans),
            "model_layers": list(config.patching.model_layers),
            "attenuation_scale": config.patching.attenuation_scale,
            "max_cases": config.patching.max_cases,
        },
        "rollout": {
            "max_cases": config.rollout.max_cases,
            "temperature": config.rollout.temperature,
            "top_p": config.rollout.top_p,
            "max_new_tokens": config.rollout.max_new_tokens,
            "repetition_penalty": config.rollout.repetition_penalty,
            "batch_size": config.rollout.batch_size,
            "seed": config.rollout.seed,
        },
    }
    config.paths.artifact_root.mkdir(parents=True, exist_ok=True)
    (config.paths.artifact_root / "config.resolved.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _coord_token_to_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value if 0 <= value <= 999 else None
    if not isinstance(value, str):
        return None
    if not (
        value.startswith(_COORD_TOKEN_PREFIX) and value.endswith(_COORD_TOKEN_SUFFIX)
    ):
        return None
    raw = value[len(_COORD_TOKEN_PREFIX) : -len(_COORD_TOKEN_SUFFIX)]
    if not raw.isdigit():
        return None
    parsed = int(raw)
    return parsed if 0 <= parsed <= 999 else None


def _bbox_norm1000(obj: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    raw_bbox = obj.get("bbox_2d")
    if not isinstance(raw_bbox, Sequence) or isinstance(raw_bbox, (str, bytes)):
        return None
    if len(raw_bbox) != 4:
        return None
    parsed = tuple(_coord_token_to_int(item) for item in raw_bbox)
    if any(item is None for item in parsed):
        return None
    return parsed  # type: ignore[return-value]


def _resolve_record_image_path(record: Mapping[str, Any], dataset_path: Path) -> Path:
    candidates: list[str] = []
    images = record.get("images")
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
        candidates.extend(str(item) for item in images if item)
    file_name = record.get("file_name")
    if file_name:
        candidates.append(str(file_name))
    for raw in candidates:
        path = Path(raw)
        candidate = path if path.is_absolute() else dataset_path.parent / path
        if candidate.exists():
            return candidate.resolve()
        if not path.is_absolute():
            parent_candidate = dataset_path.parent.parent / path
            if parent_candidate.exists():
                return parent_candidate.resolve()
    raw_fallback = candidates[0] if candidates else ""
    return (dataset_path.parent / raw_fallback).resolve()


def _case_id(dataset_path: Path, record_index: int, desc: str, target_idx: int) -> str:
    return f"{dataset_path.stem}:{record_index}:{desc.replace(' ', '_')}:{target_idx}"


def _desc_to_indices(objects: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx, obj in enumerate(objects):
        desc = str(obj.get("desc") or "").strip()
        if not desc or _bbox_norm1000(obj) is None:
            continue
        grouped.setdefault(desc, []).append(idx)
    return grouped


def mine_repeated_desc_cases(
    dataset_jsonl: Path,
    config: CaseSelectionConfig,
) -> list[InstanceBindingCase]:
    dataset_jsonl = dataset_jsonl.resolve()
    priority_descs = tuple(str(item) for item in config.priority_descs)
    priority_set = set(priority_descs)
    cases: list[InstanceBindingCase] = []
    sparse_controls: list[InstanceBindingCase] = []
    max_cases = max(0, int(config.max_cases))
    sparse_reserve = (
        min(int(config.max_sparse_controls), max(1, max_cases // 8))
        if config.include_sparse_controls and max_cases > 1
        else 0
    )
    for record_index, record in _iter_jsonl(dataset_jsonl):
        if len(cases) >= max_cases and len(sparse_controls) >= sparse_reserve:
            break
        objects_raw = record.get("objects") or []
        if not isinstance(objects_raw, Sequence) or isinstance(
            objects_raw, (str, bytes)
        ):
            continue
        objects = tuple(obj for obj in objects_raw if isinstance(obj, Mapping))
        if not objects:
            continue
        desc_groups = _desc_to_indices(objects)
        image_path = _resolve_record_image_path(record, dataset_jsonl)
        width = int(record.get("width") or 0)
        height = int(record.get("height") or 0)
        image_id = str(
            record.get("image_id") or record.get("file_name") or record_index
        )
        for desc in priority_descs:
            indices = desc_groups.get(desc) or []
            if len(indices) >= int(config.min_same_desc_candidates):
                for ordinal, target_idx_raw in enumerate(indices):
                    if len(cases) >= max_cases:
                        break
                    target_idx = int(target_idx_raw)
                    cases.append(
                        InstanceBindingCase(
                            case_id=_case_id(
                                dataset_jsonl, record_index, desc, target_idx
                            ),
                            source_jsonl=dataset_jsonl,
                            record_index=record_index,
                            image_id=image_id,
                            image_path=image_path,
                            width=width,
                            height=height,
                            desc=desc,
                            target_object_index=target_idx,
                            candidate_object_indices=tuple(int(i) for i in indices),
                            cohort="priority_same_desc",
                            selection_reason=(
                                f"priority desc with {len(indices)} same-desc "
                                f"candidates; target ordinal {ordinal}"
                            ),
                            objects=objects,
                        )
                    )
                break
        if config.include_sparse_controls and len(sparse_controls) < sparse_reserve:
            singletons = [
                (desc, indices[0])
                for desc, indices in desc_groups.items()
                if len(indices) == 1 and desc not in priority_set
            ]
            if singletons:
                desc, target_idx = singletons[0]
                sparse_controls.append(
                    InstanceBindingCase(
                        case_id=_case_id(
                            dataset_jsonl, record_index, desc, int(target_idx)
                        ),
                        source_jsonl=dataset_jsonl,
                        record_index=record_index,
                        image_id=image_id,
                        image_path=image_path,
                        width=width,
                        height=height,
                        desc=desc,
                        target_object_index=int(target_idx),
                        candidate_object_indices=(int(target_idx),),
                        cohort="sparse_single_instance_control",
                        selection_reason="single valid object desc control",
                        objects=objects,
                    )
                )
    repeated_limit = max(0, max_cases - min(sparse_reserve, len(sparse_controls)))
    selected = [*cases[:repeated_limit], *sparse_controls[:sparse_reserve]]
    return selected[:max_cases]


def validate_position_inventory(rows: Sequence[Mapping[str, Any]]) -> None:
    roles_by_case: dict[str, dict[str, int]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or row.get("case_uid") or "")
        role = str(row.get("role") or "")
        if not case_id:
            raise ValueError("position inventory row missing case_id")
        if role not in REQUIRED_POSITION_ROLES:
            raise ValueError(f"unknown position role for {case_id}: {role}")
        case_roles = roles_by_case.setdefault(case_id, {})
        if role in case_roles:
            raise ValueError(f"duplicate position role for {case_id}: {role}")
        case_roles[role] = int(row.get("token_index", -1))
        if case_roles[role] < 0:
            raise ValueError(f"negative token index for {case_id}: {role}")
    for case_id, case_roles in roles_by_case.items():
        missing = [role for role in REQUIRED_POSITION_ROLES if role not in case_roles]
        if missing:
            raise ValueError(
                f"case {case_id} missing position roles: {', '.join(missing)}"
            )


def _entropy(probs: Sequence[float]) -> float:
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs if p > 0.0))


def candidate_alignment_from_distribution(
    *,
    slot: str,
    probs: Mapping[int, float],
    candidates: Sequence[Mapping[str, Any]],
    target_label: str,
    neighbor_radius: int = 1,
) -> dict[str, Any]:
    if slot not in _SLOT_TO_INDEX:
        raise ValueError(f"unknown coordinate slot: {slot}")
    slot_idx = _SLOT_TO_INDEX[slot]
    normalized_probs = {int(k): float(v) for k, v in probs.items() if float(v) > 0.0}
    total_mass = float(sum(normalized_probs.values()))
    if total_mass <= 0.0:
        raise ValueError("coordinate distribution has no positive probability mass")
    candidate_rows: list[dict[str, Any]] = []
    used_bins: set[int] = set()
    for candidate in candidates:
        bbox = candidate.get("bbox_norm1000")
        if (
            not isinstance(bbox, Sequence)
            or isinstance(bbox, (str, bytes))
            or len(bbox) <= slot_idx
        ):
            continue
        center_bin = int(bbox[slot_idx])
        bins = set(
            range(
                max(0, center_bin - neighbor_radius),
                min(1000, center_bin + neighbor_radius + 1),
            )
        )
        used_bins.update(bins)
        mass = float(sum(normalized_probs.get(bin_idx, 0.0) for bin_idx in bins))
        candidate_rows.append(
            {
                "label": str(candidate.get("label") or ""),
                "center_bin": center_bin,
                "neighborhood_mass": mass,
            }
        )
    target_mass = float(
        sum(
            row["neighborhood_mass"]
            for row in candidate_rows
            if row["label"] == target_label
        )
    )
    other_candidate_mass = float(
        sum(
            row["neighborhood_mass"]
            for row in candidate_rows
            if row["label"] != target_label
        )
    )
    same_desc_mass = float(target_mass + other_candidate_mass)
    non_candidate_mass = max(
        0.0,
        float(
            total_mass
            - sum(normalized_probs.get(bin_idx, 0.0) for bin_idx in used_bins)
        ),
    )
    if other_candidate_mass >= 0.2 and same_desc_mass >= 0.5:
        uncertainty_kind = "instance_multimodal"
    elif target_mass >= 0.5 and other_candidate_mass < 0.2:
        uncertainty_kind = "boundary_style_uncertainty"
    else:
        uncertainty_kind = "diffuse_or_unassigned"
    return {
        "slot": slot,
        "entropy": _entropy(
            [value / total_mass for value in normalized_probs.values()]
        ),
        "total_mass": total_mass,
        "target_neighborhood_mass": target_mass,
        "other_candidate_mass": other_candidate_mass,
        "same_desc_candidate_mass": same_desc_mass,
        "non_candidate_mass": non_candidate_mass,
        "candidate_rows": candidate_rows,
        "uncertainty_kind": uncertainty_kind,
    }


def distribute_items_by_shard(
    items: Sequence[Mapping[str, Any]],
    *,
    shard_index: int,
    num_shards: int,
) -> list[Mapping[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def select_hidden_state_layers(
    *,
    hidden_state_count: int,
    requested_model_layers: Sequence[int] = DEFAULT_MODEL_LAYER_READS,
) -> tuple[dict[str, int], ...]:
    """Map decoder-layer ids to Hugging Face hidden-state tuple indices."""

    if hidden_state_count < 2:
        raise ValueError(
            "hidden_state_count must include embeddings plus decoder layers"
        )
    decoder_layer_count = hidden_state_count - 1
    rows: list[dict[str, int]] = []
    seen: set[int] = set()
    for raw_layer in requested_model_layers:
        layer = int(raw_layer)
        model_layer = decoder_layer_count + layer if layer < 0 else layer
        if not 0 <= model_layer < decoder_layer_count:
            continue
        tuple_index = model_layer + 1
        if tuple_index in seen:
            continue
        seen.add(tuple_index)
        rows.append({"model_layer": model_layer, "hidden_state_index": tuple_index})
    return tuple(rows)


def _find_subsequence(
    haystack: Sequence[int],
    needle: Sequence[int],
    *,
    start_hint: int = 0,
) -> int | None:
    if not needle:
        return int(start_hint)
    start = max(0, int(start_hint))
    limit = len(haystack) - len(needle) + 1
    for idx in range(start, max(start, limit)):
        if list(haystack[idx : idx + len(needle)]) == list(needle):
            return int(idx)
    for idx in range(0, start):
        if list(haystack[idx : idx + len(needle)]) == list(needle):
            return int(idx)
    return None


def _find_nth_subsequence(
    haystack: Sequence[int],
    needle: Sequence[int],
    *,
    occurrence_index: int,
) -> int | None:
    start = 0
    for occurrence in range(max(0, int(occurrence_index)) + 1):
        found = _find_subsequence(haystack, needle, start_hint=start)
        if found is None:
            return None
        if occurrence == int(occurrence_index):
            return found
        start = found + max(1, len(needle))
    return None


def classify_mechanism(
    *,
    pre_x1_accuracy: float,
    after_x1_accuracy: float,
    schema_patch_delta: float,
    geometry_patch_delta: float,
) -> str:
    if pre_x1_accuracy < 0.45 and after_x1_accuracy - pre_x1_accuracy >= 0.25:
        return "no_meaningful_binding_before_x1"
    if pre_x1_accuracy >= 0.75 and schema_patch_delta >= 0.15:
        return "strong_pre_x1_binding_schema_routing"
    if pre_x1_accuracy >= 0.5 and after_x1_accuracy - pre_x1_accuracy >= 0.15:
        return "partial_pre_x1_binding_x1_y1_decisive"
    if geometry_patch_delta >= max(0.15, schema_patch_delta * 2.0):
        return "partial_pre_x1_binding_geometry_states_dominate"
    return "mixed_or_inconclusive"


def _as_tuple_ints(value: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("expected a sequence of integers")
    return tuple(int(item) for item in value)


def _as_tuple_strs(value: Any, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("expected a sequence of strings")
    return tuple(str(item) for item in value)


def load_study_config(config_path: Path) -> StudyConfig:
    config_path = config_path.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("study config root must be a mapping")
    repo_root = Path(__file__).resolve().parents[2]
    shared_root = _common_repo_root(repo_root).resolve()
    run_raw = raw.get("run") or {}
    model_raw = raw.get("model") or {}
    data_raw = raw.get("data") or {}
    execution_raw = raw.get("execution") or {}
    case_raw = raw.get("case_selection") or {}
    multimodality_raw = raw.get("multimodality") or {}
    patching_raw = raw.get("patching") or {}
    rollout_raw = raw.get("rollout") or {}
    if not all(
        isinstance(item, Mapping)
        for item in (
            run_raw,
            model_raw,
            data_raw,
            execution_raw,
            case_raw,
            multimodality_raw,
            patching_raw,
            rollout_raw,
        )
    ):
        raise TypeError(
            "run/model/data/execution/case_selection/multimodality/patching/rollout "
            "must be mappings"
        )
    paths = resolve_runtime_paths(
        RuntimePathConfig(
            checkpoint=str(model_raw.get("checkpoint") or ""),
            dataset_jsonls=_as_tuple_strs(data_raw.get("dataset_jsonls"), default=()),
            artifact_root=str(
                run_raw.get("artifact_root")
                or "output/analysis/qwen3-vl-instance-binding-mechanism-20260424"
            ),
        ),
        repo_root=repo_root,
        shared_root=shared_root,
    )
    execution = ExecutionConfig(
        gpu_ids=_as_tuple_ints(
            execution_raw.get("gpu_ids"), default=(0, 1, 2, 3, 4, 5, 6, 7)
        ),
        shard_count=int(execution_raw.get("shard_count", 8)),
        per_gpu_generation_batch_size=int(
            execution_raw.get("per_gpu_generation_batch_size", 8)
        ),
        per_gpu_teacher_forced_batch_size=int(
            execution_raw.get("per_gpu_teacher_forced_batch_size", 8)
        ),
    )
    case_selection = CaseSelectionConfig(
        priority_descs=_as_tuple_strs(
            case_raw.get("priority_descs"),
            default=CaseSelectionConfig().priority_descs,
        ),
        max_cases=int(case_raw.get("max_cases", 32)),
        min_same_desc_candidates=int(case_raw.get("min_same_desc_candidates", 2)),
        include_sparse_controls=bool(case_raw.get("include_sparse_controls", True)),
        max_sparse_controls=int(case_raw.get("max_sparse_controls", 4)),
    )
    multimodality = MultimodalityConfig(
        coordinate_slots=_as_tuple_strs(
            multimodality_raw.get("coordinate_slots"),
            default=MultimodalityConfig().coordinate_slots,
        ),
        neighbor_radius=int(multimodality_raw.get("neighbor_radius", 3)),
        top_k=int(multimodality_raw.get("top_k", 12)),
    )
    patching = PatchingConfig(
        spans=_as_tuple_strs(
            patching_raw.get("spans"),
            default=PatchingConfig().spans,
        ),
        model_layers=_as_tuple_ints(
            patching_raw.get("model_layers"),
            default=PatchingConfig().model_layers,
        ),
        attenuation_scale=float(patching_raw.get("attenuation_scale", 0.0)),
        max_cases=int(patching_raw.get("max_cases", 64)),
    )
    rollout = RolloutConfig(
        max_cases=int(rollout_raw.get("max_cases", 64)),
        temperature=float(rollout_raw.get("temperature", 0.0)),
        top_p=float(rollout_raw.get("top_p", 0.9)),
        max_new_tokens=int(rollout_raw.get("max_new_tokens", 1536)),
        repetition_penalty=float(rollout_raw.get("repetition_penalty", 1.05)),
        batch_size=int(rollout_raw.get("batch_size", 8)),
        seed=int(rollout_raw.get("seed", 42)),
    )
    return StudyConfig(
        config_path=config_path,
        paths=paths,
        execution=execution,
        case_selection=case_selection,
        multimodality=multimodality,
        patching=patching,
        rollout=rollout,
    )


def _case_summary(cases: Sequence[InstanceBindingCase]) -> dict[str, Any]:
    cohort_counts: dict[str, int] = {}
    desc_counts: dict[str, int] = {}
    candidate_count_hist: Counter[int] = Counter()
    target_ordinal_hist: Counter[int] = Counter()
    for case in cases:
        cohort_counts[case.cohort] = cohort_counts.get(case.cohort, 0) + 1
        desc_counts[case.desc] = desc_counts.get(case.desc, 0) + 1
        candidate_count_hist[len(case.candidate_object_indices)] += 1
        try:
            ordinal = list(case.candidate_object_indices).index(
                case.target_object_index
            )
        except ValueError:
            ordinal = -1
        target_ordinal_hist[int(ordinal)] += 1
    return {
        "num_cases": len(cases),
        "cohort_counts": cohort_counts,
        "desc_counts": desc_counts,
        "candidate_count_hist": {
            str(key): int(value) for key, value in sorted(candidate_count_hist.items())
        },
        "target_ordinal_hist": {
            str(key): int(value) for key, value in sorted(target_ordinal_hist.items())
        },
    }


def _read_selected_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for _, row in _iter_jsonl(path)]


def _render_object(obj: Mapping[str, Any]) -> str:
    from src.common.object_field_order import build_object_payload
    from src.utils.assistant_json import dumps_coordjson

    bbox = obj.get("bbox_2d")
    if (
        not isinstance(bbox, Sequence)
        or isinstance(bbox, (str, bytes))
        or len(bbox) != 4
    ):
        raise ValueError("object is missing bbox_2d coord tokens")
    payload = build_object_payload(
        desc=str(obj.get("desc") or ""),
        geometry_key="bbox_2d",
        geometry_value=list(bbox),
        object_field_order="desc_first",
    )
    return dumps_coordjson(payload)


def _render_assistant_text(objects: Sequence[Mapping[str, Any]]) -> str:
    from src.common.object_field_order import build_object_payload
    from src.utils.assistant_json import dumps_coordjson

    payload_objects: list[dict[str, Any]] = []
    for obj in objects:
        bbox = obj.get("bbox_2d")
        if (
            not isinstance(bbox, Sequence)
            or isinstance(bbox, (str, bytes))
            or len(bbox) != 4
        ):
            continue
        payload_objects.append(
            build_object_payload(
                desc=str(obj.get("desc") or ""),
                geometry_key="bbox_2d",
                geometry_value=list(bbox),
                object_field_order="desc_first",
            )
        )
    return dumps_coordjson({"objects": payload_objects})


def _target_position_inventory(
    *,
    tokenizer: Any,
    assistant_text: str,
    target_object: Mapping[str, Any],
    desc_occurrence_index: int = 0,
) -> dict[str, Any]:
    from src.coord_tokens.codec import get_coord_token_ids

    assistant_ids = [
        int(v) for v in tokenizer.encode(assistant_text, add_special_tokens=False)
    ]
    desc_ids = [
        int(v)
        for v in tokenizer.encode(
            str(target_object.get("desc") or ""), add_special_tokens=False
        )
    ]
    desc_start = _find_nth_subsequence(
        assistant_ids,
        desc_ids,
        occurrence_index=desc_occurrence_index,
    )
    if desc_start is None:
        raise ValueError("target desc token span not found in assistant text")
    desc_positions = list(range(desc_start, desc_start + len(desc_ids)))
    bbox = _bbox_norm1000(target_object)
    if bbox is None:
        raise ValueError("target object bbox is not coord-token norm1000")
    coord_token_ids = get_coord_token_ids(tokenizer, validate=True)
    coord_ids = [int(coord_token_ids[int(v)]) for v in bbox]
    coord_positions: list[int] = []
    search_start = desc_positions[-1] + 1
    for coord_id in coord_ids:
        found = None
        for pos in range(search_start, len(assistant_ids)):
            if int(assistant_ids[pos]) == int(coord_id):
                found = int(pos)
                break
        if found is None:
            raise ValueError("target coord token not found in assistant text")
        coord_positions.append(found)
        search_start = found + 1
    coord0 = coord_positions[0]
    roles = {
        "desc_end": desc_positions[-1],
        "desc_closing_quote": min(desc_positions[-1] + 1, coord0 - 1),
        "field_delimiter": min(desc_positions[-1] + 2, coord0 - 1),
        "bbox_key": max(desc_positions[-1] + 1, coord0 - 3),
        "bbox_open_bracket": max(desc_positions[-1] + 1, coord0 - 1),
        "pre_x1": max(desc_positions[-1] + 1, coord0 - 1),
        "post_x1": coord_positions[0],
        "post_y1": coord_positions[1],
    }
    return {
        "assistant_ids": assistant_ids,
        "roles": roles,
        "desc_positions": desc_positions,
        "coord_positions": coord_positions,
        "bbox_norm1000": list(bbox),
    }


def _candidate_rows_for_case(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    objects = case.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for idx in case.get("candidate_object_indices") or []:
        obj = objects[int(idx)]
        if not isinstance(obj, Mapping):
            continue
        bbox = _bbox_norm1000(obj)
        if bbox is None:
            continue
        rows.append(
            {
                "label": (
                    "target"
                    if int(idx) == int(case.get("target_object_index", -1))
                    else f"candidate_{idx}"
                ),
                "object_index": int(idx),
                "bbox_norm1000": list(bbox),
            }
        )
    return rows


def _top_coord_probs(
    *,
    logits: Any,
    coord_token_ids: Sequence[int],
    top_k: int = 12,
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    import torch

    coord_ids_tensor = torch.tensor(
        coord_token_ids, dtype=torch.long, device=logits.device
    )
    coord_logits = logits.index_select(dim=0, index=coord_ids_tensor).float()
    probs = torch.softmax(coord_logits, dim=-1).detach().cpu()
    values, indices = torch.topk(probs, k=min(int(top_k), int(probs.numel())))
    full_probs = {
        int(idx): float(value)
        for idx, value in enumerate(probs.tolist())
        if float(value) > 0.0
    }
    top_rows = [
        {
            "bin": int(idx.item()),
            "prob": float(value.item()),
        }
        for value, idx in zip(values, indices, strict=False)
    ]
    return full_probs, top_rows


def _run_forward_extraction_stage(
    *,
    config: StudyConfig,
    stage: str,
    shard_index: int | None,
    num_shards: int | None,
    save_hidden_vectors: bool,
) -> dict[str, Any]:
    import gc

    import numpy as np
    import torch
    from PIL import Image

    from src.analysis.unmatched_proposal_verifier import TeacherForcedScorer
    from src.coord_tokens.codec import get_coord_token_ids

    cases = _read_selected_cases(
        config.paths.artifact_root / "cases" / "selected_cases.jsonl"
    )
    if not cases:
        raise FileNotFoundError(
            "No selected cases found. Run --stage audit,select_cases before GPU stages."
        )
    active_cases = (
        distribute_items_by_shard(cases, shard_index=shard_index, num_shards=num_shards)
        if shard_index is not None and num_shards is not None
        else cases
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scorer = TeacherForcedScorer(
        checkpoint_path=config.paths.checkpoint,
        device=device,
        attn_implementation="auto",
    )
    coord_token_ids = get_coord_token_ids(scorer.tokenizer, validate=True)
    batch_size = max(1, int(config.execution.per_gpu_teacher_forced_batch_size))
    vector_rows: list[dict[str, Any]] = []
    vectors: list[Any] = []
    position_rows: list[dict[str, Any]] = []
    multimodality_rows: list[dict[str, Any]] = []
    processed = 0
    try:
        for start in range(0, len(active_cases), batch_size):
            batch_cases = active_cases[start : start + batch_size]
            images: list[Any] = []
            full_texts: list[str] = []
            inventories: list[dict[str, Any]] = []
            for case in batch_cases:
                objects_raw = case.get("objects") or []
                if not isinstance(objects_raw, Sequence) or isinstance(
                    objects_raw, (str, bytes)
                ):
                    raise ValueError(f"case {case.get('case_id')} has invalid objects")
                objects = [obj for obj in objects_raw if isinstance(obj, Mapping)]
                target_idx = int(case.get("target_object_index", 0))
                target_object = objects[target_idx]
                target_desc = str(target_object.get("desc") or "")
                desc_occurrence_index = sum(
                    1
                    for obj in objects[:target_idx]
                    if str(obj.get("desc") or "") == target_desc
                )
                assistant_text = _render_assistant_text(objects)
                inventory = _target_position_inventory(
                    tokenizer=scorer.tokenizer,
                    assistant_text=assistant_text,
                    target_object=target_object,
                    desc_occurrence_index=desc_occurrence_index,
                )
                image = Image.open(str(case["image_path"])).convert("RGB")
                _, full_messages = scorer.build_messages(
                    image=image,
                    assistant_text=assistant_text,
                    prompt_variant="coco_80",
                    object_field_order="desc_first",
                )
                images.append(image)
                full_texts.append(
                    scorer.processor.apply_chat_template(
                        full_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
                inventories.append(inventory)
            model_inputs = scorer.processor(
                text=full_texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            model_inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in model_inputs.items()
            }
            with torch.inference_mode():
                outputs = scorer.model(
                    **model_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                )
            hidden_states = getattr(outputs, "hidden_states", None)
            logits = getattr(outputs, "logits", None)
            input_ids = model_inputs.get("input_ids")
            if (
                hidden_states is None
                or logits is None
                or not isinstance(input_ids, torch.Tensor)
            ):
                raise RuntimeError(
                    "GPU extraction requires hidden_states, logits, and input_ids"
                )
            layer_rows = select_hidden_state_layers(
                hidden_state_count=len(hidden_states)
            )
            for row_idx, (case, inventory) in enumerate(
                zip(batch_cases, inventories, strict=False)
            ):
                full_ids = [int(v) for v in input_ids[row_idx].detach().cpu().tolist()]
                assistant_start = _find_subsequence(
                    full_ids,
                    inventory["assistant_ids"],
                    start_hint=max(
                        0, len(full_ids) - len(inventory["assistant_ids"]) - 64
                    ),
                )
                if assistant_start is None:
                    raise ValueError(
                        f"assistant ids not found for case {case.get('case_id')}"
                    )
                absolute_roles = {
                    role: int(assistant_start + rel_pos)
                    for role, rel_pos in dict(inventory["roles"]).items()
                }
                for role, token_index in absolute_roles.items():
                    position_rows.append(
                        {
                            "case_id": str(case["case_id"]),
                            "role": role,
                            "token_index": int(token_index),
                            "assistant_relative_token_index": int(
                                inventory["roles"][role]
                            ),
                            "shard_index": shard_index,
                            "num_shards": num_shards,
                        }
                    )
                for slot in config.multimodality.coordinate_slots:
                    prediction_role = _SLOT_TO_PREDICTION_ROLE.get(slot)
                    if prediction_role is None:
                        continue
                    full_probs, top_rows = _top_coord_probs(
                        logits=logits[row_idx, int(absolute_roles[prediction_role])],
                        coord_token_ids=coord_token_ids,
                        top_k=config.multimodality.top_k,
                    )
                    alignment = candidate_alignment_from_distribution(
                        slot=slot,
                        probs=full_probs,
                        candidates=_candidate_rows_for_case(case),
                        target_label="target",
                        neighbor_radius=config.multimodality.neighbor_radius,
                    )
                    multimodality_rows.append(
                        {
                            "case_id": str(case["case_id"]),
                            "desc": str(case.get("desc") or ""),
                            "cohort": str(case.get("cohort") or ""),
                            "slot": slot,
                            "prediction_role": prediction_role,
                            "mass_surface": "coord_family_softmax_full",
                            "top_coord_bins": top_rows,
                            **alignment,
                        }
                    )
                if save_hidden_vectors:
                    for layer in layer_rows:
                        tensor = hidden_states[int(layer["hidden_state_index"])][
                            row_idx
                        ]
                        for role, token_index in absolute_roles.items():
                            vector_rows.append(
                                {
                                    "case_id": str(case["case_id"]),
                                    "role": role,
                                    "model_layer": int(layer["model_layer"]),
                                    "hidden_state_index": int(
                                        layer["hidden_state_index"]
                                    ),
                                    "token_index": int(token_index),
                                }
                            )
                            vectors.append(
                                tensor[int(token_index)]
                                .detach()
                                .float()
                                .cpu()
                                .numpy()
                                .astype("float32")
                            )
                processed += 1
            for image in images:
                image.close()
            del outputs, hidden_states, logits, input_ids, model_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    stage_dir = config.paths.artifact_root / stage
    shard_label = (
        f"shard_{int(shard_index):03d}-of-{int(num_shards):03d}"
        if shard_index is not None and num_shards is not None
        else "shard_all"
    )
    _write_jsonl(
        stage_dir / f"token_position_inventory_{shard_label}.jsonl",
        position_rows,
    )
    _write_jsonl(
        stage_dir / f"coord_multimodality_{shard_label}.jsonl",
        multimodality_rows,
    )
    _write_jsonl(
        stage_dir / f"pre_x1_multimodality_{shard_label}.jsonl",
        [row for row in multimodality_rows if row.get("slot") == "x1"],
    )
    if save_hidden_vectors:
        vector_array = (
            np.stack(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32)
        )
        np.savez_compressed(
            stage_dir / f"hidden_vectors_{shard_label}.npz",
            vectors=vector_array,
        )
        _write_jsonl(stage_dir / f"hidden_vector_rows_{shard_label}.jsonl", vector_rows)
    summary = {
        "stage": stage,
        "status": "ok",
        "device": device,
        "num_input_cases": len(cases),
        "num_shard_cases": len(active_cases),
        "num_processed_cases": processed,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "batch_size": batch_size,
        "saved_hidden_vectors": save_hidden_vectors,
    }
    _write_json(stage_dir / f"summary_{shard_label}.json", summary)
    return summary


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for _, row in _iter_jsonl(path)]


def _read_jsonl_globs(patterns: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for pattern in patterns:
        for raw_path in sorted(glob(pattern)):
            path = Path(raw_path)
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            rows.extend(_read_jsonl_rows(path))
    return rows


def _summarize_multimodality_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_rows": 0,
            "slots": {},
            "uncertainty_counts": {},
            "status": "empty",
        }

    def _float(row: Mapping[str, Any], key: str) -> float:
        return float(row.get(key, 0.0) or 0.0)

    by_slot: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    uncertainty_counts: Counter[str] = Counter()
    for row in rows:
        by_slot[str(row.get("slot") or "")].append(row)
        uncertainty_counts[str(row.get("uncertainty_kind") or "")] += 1

    slot_summaries: dict[str, Any] = {}
    for slot, slot_rows in sorted(by_slot.items()):
        margins: list[float] = []
        target_best_or_tied = 0
        target_strict_best = 0
        target_positive = 0
        other_positive = 0
        nearest_target = 0
        nearest_other = 0
        for row in slot_rows:
            candidate_rows = row.get("candidate_rows") or []
            if not isinstance(candidate_rows, Sequence) or isinstance(
                candidate_rows, (str, bytes)
            ):
                continue
            target_rows = [
                candidate
                for candidate in candidate_rows
                if isinstance(candidate, Mapping)
                and str(candidate.get("label") or "") == "target"
            ]
            if not target_rows:
                continue
            target_mass = float(target_rows[0].get("neighborhood_mass", 0.0) or 0.0)
            other_masses = [
                float(candidate.get("neighborhood_mass", 0.0) or 0.0)
                for candidate in candidate_rows
                if isinstance(candidate, Mapping)
                and str(candidate.get("label") or "") != "target"
            ]
            best_other = max(other_masses or [0.0])
            margins.append(target_mass - best_other)
            target_best_or_tied += int(target_mass >= best_other)
            target_strict_best += int(target_mass > best_other)
            target_positive += int(target_mass > 0.0)
            other_positive += int(any(value > 0.0 for value in other_masses))
            top_bins = row.get("top_coord_bins") or []
            if (
                isinstance(top_bins, Sequence)
                and top_bins
                and isinstance(top_bins[0], Mapping)
            ):
                top_bin = int(top_bins[0].get("bin", -1))
                distances = [
                    (
                        abs(top_bin - int(candidate.get("center_bin", -10_000))),
                        str(candidate.get("label") or ""),
                    )
                    for candidate in candidate_rows
                    if isinstance(candidate, Mapping)
                ]
                if distances:
                    best_distance = min(distance for distance, _ in distances)
                    nearest_labels = [
                        label
                        for distance, label in distances
                        if distance == best_distance
                    ]
                    if "target" in nearest_labels:
                        nearest_target += 1
                    else:
                        nearest_other += 1
        count = len(slot_rows)
        slot_summaries[slot] = {
            "num_rows": count,
            "mean_entropy": sum(_float(row, "entropy") for row in slot_rows) / count,
            "mean_target_neighborhood_mass": sum(
                _float(row, "target_neighborhood_mass") for row in slot_rows
            )
            / count,
            "mean_other_candidate_mass": sum(
                _float(row, "other_candidate_mass") for row in slot_rows
            )
            / count,
            "mean_same_desc_candidate_mass": sum(
                _float(row, "same_desc_candidate_mass") for row in slot_rows
            )
            / count,
            "mean_non_candidate_mass": sum(
                _float(row, "non_candidate_mass") for row in slot_rows
            )
            / count,
            "target_best_or_tied_rate": target_best_or_tied / count,
            "target_strict_best_rate": target_strict_best / count,
            "target_positive_rate": target_positive / count,
            "other_positive_rate": other_positive / count,
            "mean_target_vs_best_other_margin": (
                sum(margins) / len(margins) if margins else 0.0
            ),
            "top_bin_nearest_target_rate": nearest_target / count,
            "top_bin_nearest_other_rate": nearest_other / count,
        }
    return {
        "num_rows": len(rows),
        "slots": slot_summaries,
        "uncertainty_counts": {
            key: int(value) for key, value in sorted(uncertainty_counts.items())
        },
        "status": "ok",
    }


def _run_merge_multimodality_stage(config: StudyConfig) -> dict[str, Any]:
    artifact_root = config.paths.artifact_root
    rows = _read_jsonl_globs(
        [str(artifact_root / "multimodality" / "coord_multimodality_*.jsonl")]
    )
    source_stage = "multimodality"
    if not rows:
        rows = _read_jsonl_globs(
            [str(artifact_root / "multimodality" / "pre_x1_multimodality_*.jsonl")]
        )
    if not rows:
        rows = _read_jsonl_globs(
            [str(artifact_root / "hidden_states" / "coord_multimodality_*.jsonl")]
        )
        source_stage = "hidden_states"
    if not rows:
        rows = _read_jsonl_globs(
            [str(artifact_root / "hidden_states" / "pre_x1_multimodality_*.jsonl")]
        )
        source_stage = "hidden_states"
    summary = {
        "stage": "merge_multimodality",
        "source_stage": source_stage,
        **_summarize_multimodality_rows(rows),
    }
    stage_dir = artifact_root / "merge_multimodality"
    _write_jsonl(stage_dir / "coord_multimodality_merged.jsonl", rows)
    _write_json(stage_dir / "summary.json", summary)
    return summary


def _case_target_bbox(case: Mapping[str, Any]) -> tuple[float, float, float, float]:
    objects = case.get("objects") or []
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError(f"case {case.get('case_id')} has invalid objects")
    target_idx = int(case.get("target_object_index", -1))
    target = objects[target_idx]
    if not isinstance(target, Mapping):
        raise ValueError(f"case {case.get('case_id')} target object is invalid")
    bbox = _bbox_norm1000(target)
    if bbox is None:
        raise ValueError(f"case {case.get('case_id')} target bbox is invalid")
    return tuple(float(value) / 1000.0 for value in bbox)


def _candidate_rank_from_prediction(
    *,
    prediction: Sequence[float],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pred = [float(value) for value in prediction]
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        bbox = candidate.get("bbox_norm1000")
        if (
            not isinstance(bbox, Sequence)
            or isinstance(bbox, (str, bytes))
            or len(bbox) != 4
        ):
            continue
        candidate_vec = [float(value) / 1000.0 for value in bbox]
        mse = sum((pred[idx] - candidate_vec[idx]) ** 2 for idx in range(4)) / 4.0
        scored.append(
            {
                "label": str(candidate.get("label") or ""),
                "object_index": int(candidate.get("object_index", -1)),
                "score": -float(mse),
            }
        )
    target_scores = [row["score"] for row in scored if row["label"] == "target"]
    if not scored or not target_scores:
        raise ValueError("candidate ranking requires a target candidate")
    target_score = float(target_scores[0])
    other_scores = [row["score"] for row in scored if row["label"] != "target"]
    rank = 1 + sum(1 for row in scored if float(row["score"]) > target_score)
    best_other = max(other_scores) if other_scores else float("-inf")
    top = max(scored, key=lambda row: float(row["score"]))
    return {
        "rank": int(rank),
        "candidate_count": len(scored),
        "top1_correct": bool(rank == 1),
        "reciprocal_rank": 1.0 / float(rank),
        "target_score": target_score,
        "best_other_score": best_other,
        "target_vs_best_other_margin": (
            target_score - best_other if other_scores else target_score
        ),
        "top_label": str(top["label"]),
        "top_object_index": int(top["object_index"]),
    }


def _assign_group_folds(groups: Sequence[str], fold_count: int) -> dict[str, int]:
    unique_groups = sorted(set(groups))
    if not unique_groups:
        return {}
    active_fold_count = max(1, min(int(fold_count), len(unique_groups)))
    return {group: idx % active_fold_count for idx, group in enumerate(unique_groups)}


def _ridge_predict_grouped(
    *,
    features: Sequence[Any],
    targets: Sequence[Sequence[float]],
    groups: Sequence[str],
    ridge_alpha: float = 25.0,
    fold_count: int = 5,
) -> list[list[float]]:
    import numpy as np

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("probe features/targets must be aligned matrices")
    if x.shape[0] < 2:
        return y.astype(float).tolist()
    fold_by_group = _assign_group_folds(groups, fold_count)
    predictions = np.zeros_like(y)
    for fold in sorted(set(fold_by_group.values())):
        test_idx = [
            idx for idx, group in enumerate(groups) if fold_by_group[group] == fold
        ]
        train_idx = [idx for idx in range(x.shape[0]) if idx not in set(test_idx)]
        if not train_idx:
            predictions[test_idx] = y[test_idx]
            continue
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        x_mean = x_train.mean(axis=0, keepdims=True)
        x_std = x_train.std(axis=0, keepdims=True)
        x_std[x_std < 1e-6] = 1.0
        y_mean = y_train.mean(axis=0, keepdims=True)
        x_train_z = (x_train - x_mean) / x_std
        x_test_z = (x_test - x_mean) / x_std
        y_train_c = y_train - y_mean
        kernel = x_train_z @ x_train_z.T
        regularizer = float(ridge_alpha) * np.eye(kernel.shape[0])
        alpha = np.linalg.solve(kernel + regularizer, y_train_c)
        predictions[test_idx] = x_test_z @ x_train_z.T @ alpha + y_mean
    return predictions.astype(float).tolist()


def _load_hidden_vector_entries(config: StudyConfig) -> list[dict[str, Any]]:
    import numpy as np

    stage_dir = config.paths.artifact_root / "hidden_states"
    entries: list[dict[str, Any]] = []
    for rows_path in sorted(stage_dir.glob("hidden_vector_rows_*.jsonl")):
        shard_label = rows_path.name.removeprefix("hidden_vector_rows_").removesuffix(
            ".jsonl"
        )
        npz_path = stage_dir / f"hidden_vectors_{shard_label}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"missing hidden vector array: {npz_path}")
        rows = _read_jsonl_rows(rows_path)
        vectors = np.load(npz_path)["vectors"]
        if int(vectors.shape[0]) != len(rows):
            raise ValueError(
                f"hidden vector row count mismatch for {shard_label}: "
                f"{vectors.shape[0]} vectors vs {len(rows)} rows"
            )
        for idx, row in enumerate(rows):
            item = dict(row)
            item["vector"] = vectors[idx]
            entries.append(item)
    return entries


def _summarize_probe_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_rows": 0, "status": "empty"}
    count = len(rows)
    return {
        "num_rows": count,
        "top1_accuracy": sum(bool(row.get("top1_correct")) for row in rows) / count,
        "mean_reciprocal_rank": sum(
            float(row.get("reciprocal_rank", 0.0) or 0.0) for row in rows
        )
        / count,
        "mean_rank": sum(float(row.get("rank", 0.0) or 0.0) for row in rows) / count,
        "mean_chance_top1": sum(
            1.0 / max(1.0, float(row.get("candidate_count", 1.0) or 1.0))
            for row in rows
        )
        / count,
        "mean_target_vs_best_other_margin": sum(
            float(row.get("target_vs_best_other_margin", 0.0) or 0.0) for row in rows
        )
        / count,
    }


def _run_binding_probe_stage(config: StudyConfig) -> dict[str, Any]:
    cases = _read_selected_cases(
        config.paths.artifact_root / "cases" / "selected_cases.jsonl"
    )
    if not cases:
        raise FileNotFoundError("binding probe requires selected cases")
    cases_by_id = {str(case["case_id"]): case for case in cases}
    entries = _load_hidden_vector_entries(config)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        key = (str(entry.get("role") or ""), int(entry.get("model_layer", -1)))
        grouped[key].append(entry)

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (role, layer), group_entries in sorted(grouped.items()):
        features: list[Any] = []
        targets: list[tuple[float, float, float, float]] = []
        groups: list[str] = []
        metadata: list[dict[str, Any]] = []
        for entry in group_entries:
            case_id = str(entry.get("case_id") or "")
            case = cases_by_id.get(case_id)
            if case is None:
                continue
            features.append(entry["vector"])
            targets.append(_case_target_bbox(case))
            groups.append(str(case.get("image_id") or case_id))
            metadata.append({"entry": entry, "case": case})
        if len(features) < 2:
            continue
        predictions = _ridge_predict_grouped(
            features=features,
            targets=targets,
            groups=groups,
        )
        role_layer_rows: list[dict[str, Any]] = []
        for pred, meta in zip(predictions, metadata, strict=False):
            case = meta["case"]
            rank = _candidate_rank_from_prediction(
                prediction=pred,
                candidates=_candidate_rows_for_case(case),
            )
            candidate_count = int(rank["candidate_count"])
            try:
                target_ordinal = list(case.get("candidate_object_indices") or []).index(
                    int(case.get("target_object_index", -1))
                )
            except ValueError:
                target_ordinal = -1
            row = {
                "case_id": str(case["case_id"]),
                "image_id": str(case.get("image_id") or ""),
                "desc": str(case.get("desc") or ""),
                "cohort": str(case.get("cohort") or ""),
                "role": role,
                "model_layer": int(layer),
                "target_ordinal": int(target_ordinal),
                "candidate_count": candidate_count,
                "chance_top1": 1.0 / max(1, candidate_count),
                "predicted_bbox_norm1000": [
                    max(0.0, min(1000.0, float(value) * 1000.0)) for value in pred
                ],
                **rank,
            }
            prediction_rows.append(row)
            role_layer_rows.append(row)
        summary = {
            "role": role,
            "model_layer": int(layer),
            **_summarize_probe_predictions(role_layer_rows),
        }
        summary["chance_normalized_lift"] = float(summary["top1_accuracy"]) - float(
            summary["mean_chance_top1"]
        )
        summary_rows.append(summary)

    stage_dir = config.paths.artifact_root / "binding_probe"
    _write_jsonl(stage_dir / "probe_predictions.jsonl", prediction_rows)
    _write_jsonl(stage_dir / "probe_summary_by_role_layer.jsonl", summary_rows)
    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_role[str(row["role"])].append(row)
    role_summary = {
        role: max(rows, key=lambda row: float(row.get("top1_accuracy", 0.0)))
        for role, rows in sorted(by_role.items())
    }
    best_pre_x1 = float(role_summary.get("pre_x1", {}).get("top1_accuracy", 0.0))
    best_after_x1 = max(
        float(role_summary.get("post_x1", {}).get("top1_accuracy", 0.0)),
        float(role_summary.get("post_y1", {}).get("top1_accuracy", 0.0)),
    )
    conclusion = classify_mechanism(
        pre_x1_accuracy=best_pre_x1,
        after_x1_accuracy=best_after_x1,
        schema_patch_delta=0.0,
        geometry_patch_delta=0.0,
    )
    summary = {
        "stage": "binding_probe",
        "status": "ok",
        "num_cases": len(cases),
        "num_prediction_rows": len(prediction_rows),
        "num_role_layer_cells": len(summary_rows),
        "role_best": role_summary,
        "best_pre_x1_accuracy": best_pre_x1,
        "best_after_x1_accuracy": best_after_x1,
        "probe_only_mechanism_signal": conclusion,
        "convergence_status": "not_converged_probe_only",
        "warning": (
            "This is a linear bbox-regression probe over hidden vectors. It is "
            "evidence of decodability, not causal proof."
        ),
    }
    _write_json(stage_dir / "summary.json", summary)
    return summary


def _run_report_stage(config: StudyConfig) -> dict[str, Any]:
    artifact_root = config.paths.artifact_root
    case_summary_path = artifact_root / "cases" / "summary.json"
    probe_summary_path = artifact_root / "binding_probe" / "summary.json"
    multimodality_summary_path = artifact_root / "merge_multimodality" / "summary.json"
    patching_summary_path = artifact_root / "merge_patching" / "summary.json"
    donor_summary_path = artifact_root / "donor_patching" / "summary.json"
    good_bad_summary_path = artifact_root / "good_bad_panels" / "summary.json"
    rollout_summary_path = artifact_root / "rollout_failure_split" / "summary.json"
    case_summary = (
        json.loads(case_summary_path.read_text(encoding="utf-8"))
        if case_summary_path.exists()
        else {}
    )
    probe_summary = (
        json.loads(probe_summary_path.read_text(encoding="utf-8"))
        if probe_summary_path.exists()
        else {}
    )
    multimodality_summary = (
        json.loads(multimodality_summary_path.read_text(encoding="utf-8"))
        if multimodality_summary_path.exists()
        else {}
    )
    patching_summary = (
        json.loads(patching_summary_path.read_text(encoding="utf-8"))
        if patching_summary_path.exists()
        else {}
    )
    donor_summary = (
        json.loads(donor_summary_path.read_text(encoding="utf-8"))
        if donor_summary_path.exists()
        else {}
    )
    good_bad_summary = (
        json.loads(good_bad_summary_path.read_text(encoding="utf-8"))
        if good_bad_summary_path.exists()
        else {}
    )
    rollout_summary = (
        json.loads(rollout_summary_path.read_text(encoding="utf-8"))
        if rollout_summary_path.exists()
        else {}
    )
    rollout_label_counts = {
        str(key): int(value)
        for key, value in (rollout_summary.get("label_counts") or {}).items()
    }
    rollout_total = sum(rollout_label_counts.values())
    rollout_healthy_count = rollout_label_counts.get(
        "healthy_same_desc_multi", 0
    ) + rollout_label_counts.get("healthy_single_same_desc", 0)
    rollout_failure_like_count = (
        rollout_label_counts.get("duplicate_collapse_like", 0)
        + rollout_label_counts.get("near_duplicate_like", 0)
        + rollout_label_counts.get("wrong_or_missing_target_desc", 0)
    )
    rollout_ready = (
        str(rollout_summary.get("status") or "") == "ok"
        and rollout_total > 0
        and rollout_healthy_count > 0
        and rollout_failure_like_count > 0
    )
    rollout_basin_crosstab: dict[str, dict[str, int]] = {}
    rollout_labels_path = Path(
        str(
            rollout_summary.get("rollout_labels")
            or (artifact_root / "rollout_failure_split" / "rollout_labels.jsonl")
        )
    )
    good_bad_panels_path = (
        artifact_root / "good_bad_panels" / "good_bad_case_panels.jsonl"
    )
    if rollout_labels_path.exists() and good_bad_panels_path.exists():
        basin_by_case = {
            str(row.get("case_id") or ""): str(row.get("basin_label") or "")
            for row in _read_jsonl_rows(good_bad_panels_path)
        }
        crosstab: dict[str, Counter[str]] = {}
        for row in _read_jsonl_rows(rollout_labels_path):
            case_id = str(row.get("case_id") or "")
            basin_label = basin_by_case.get(case_id) or "missing_basin_proxy"
            rollout_label = str(row.get("rollout_label") or "unknown_rollout_label")
            crosstab.setdefault(basin_label, Counter())[rollout_label] += 1
        rollout_basin_crosstab = {
            basin_label: {
                rollout_label: int(count)
                for rollout_label, count in sorted(label_counter.items())
            }
            for basin_label, label_counter in sorted(crosstab.items())
        }
    pre_x1_probe = float(probe_summary.get("best_pre_x1_accuracy", 0.0) or 0.0)
    after_x1_probe = float(probe_summary.get("best_after_x1_accuracy", 0.0) or 0.0)
    x1_multi = dict((multimodality_summary.get("slots") or {}).get("x1") or {})
    target_best_rate = float(x1_multi.get("target_strict_best_rate", 0.0) or 0.0)
    target_mass = float(x1_multi.get("mean_target_neighborhood_mass", 0.0) or 0.0)
    other_mass = float(x1_multi.get("mean_other_candidate_mass", 0.0) or 0.0)
    intervention_summaries = patching_summary.get("intervention_summaries") or {}
    schema_patch = dict(intervention_summaries.get("schema_context") or {})
    previous_geometry_patch = dict(
        intervention_summaries.get("previous_geometry") or {}
    )
    current_desc_patch = dict(intervention_summaries.get("current_desc") or {})
    donor_span_summaries = donor_summary.get("span_summaries") or {}
    donor_schema_summary = dict(donor_span_summaries.get("schema_context") or {})
    donor_current_desc_summary = dict(donor_span_summaries.get("current_desc") or {})
    donor_previous_geometry_summary = dict(
        donor_span_summaries.get("previous_geometry") or {}
    )
    schema_abs_delta = float(schema_patch.get("mean_abs_margin_delta", 0.0) or 0.0)
    previous_geometry_abs_delta = float(
        previous_geometry_patch.get("mean_abs_margin_delta", 0.0) or 0.0
    )
    current_desc_abs_delta = float(
        current_desc_patch.get("mean_abs_margin_delta", 0.0) or 0.0
    )
    schema_flip_rate = float(schema_patch.get("top_candidate_flip_rate", 0.0) or 0.0)
    donor_schema_mass_delta = float(
        donor_schema_summary.get("mean_donor_mass_delta", 0.0) or 0.0
    )
    donor_schema_target_delta = float(
        donor_schema_summary.get("mean_target_mass_delta", 0.0) or 0.0
    )
    donor_schema_flip_to_donor = float(
        donor_schema_summary.get("flip_to_donor_rate", 0.0) or 0.0
    )
    donor_current_desc_mass_delta = float(
        donor_current_desc_summary.get("mean_donor_mass_delta", 0.0) or 0.0
    )
    donor_previous_geometry_mass_delta = float(
        donor_previous_geometry_summary.get("mean_donor_mass_delta", 0.0) or 0.0
    )
    mixed_soft_binding_condition = (
        pre_x1_probe < 0.5
        and after_x1_probe - pre_x1_probe >= 0.15
        and schema_abs_delta >= max(0.02, previous_geometry_abs_delta * 10.0)
        and schema_flip_rate >= 0.2
    )
    next_steps = [
        "Run donor activation patching on the highest-signal pre-x1/schema roles.",
        "Expand the rollout-mined healthy vs duplicate-collapse case split.",
        "Repeat after controls if target ordinal or sparse-control balance changes.",
    ]
    if not probe_summary or not multimodality_summary:
        convergence_status = "not_converged_extraction_only"
        headline = "insufficient merged probe/multimodality evidence"
    elif not patching_summary:
        convergence_status = "not_converged_missing_causal"
        headline = "mixed probe/distribution evidence; causal patching still missing"
    elif not good_bad_summary:
        convergence_status = "not_converged_missing_good_bad_failure_split"
        headline = "causal evidence exists; good-vs-bad basin split still missing"
    elif not rollout_ready:
        convergence_status = "not_converged_missing_rollout_failure_split"
        if mixed_soft_binding_condition:
            headline = (
                "partial/weak pre-x1 binding with a clear post-x1 hardening jump; "
                "schema-context states are causally important in attenuation tests"
            )
        else:
            headline = "mechanism evidence present but rollout contrast is missing"
    elif mixed_soft_binding_condition:
        convergence_status = (
            "converged_first_pass_mixed_soft_pre_x1_coordinate_hardening"
        )
        headline = (
            "mixed view supported: weak/partial pre-x1 binding exists, but x1/y1 "
            "remains the hard instance-disambiguation boundary in difficult same-desc scenes"
        )
        if donor_summary:
            next_steps = [
                "Expand the same-desc rollout split beyond 64 cases to check stability.",
                "Add a wrong-image control for the schema-context attenuation result.",
                "Repeat donor patching with randomized donor controls to separate content transfer from positional disruption.",
            ]
        else:
            next_steps = [
                "Run donor activation patching to separate schema routing from identity-bearing state.",
                "Expand the same-desc rollout split beyond 64 cases to check stability.",
                "Add a wrong-image control for the schema-context attenuation result.",
            ]
    elif pre_x1_probe >= 0.55 and after_x1_probe - pre_x1_probe >= 0.10:
        convergence_status = "converged_first_pass_partial_pre_x1_binding"
        headline = "partial pre-x1 decodability with post-coordinate hardening"
    elif pre_x1_probe < 0.45 and after_x1_probe - pre_x1_probe >= 0.20:
        convergence_status = "converged_first_pass_weak_pre_x1_coordinate_split"
        headline = "weak pre-x1 decodability with a stronger coordinate jump"
    elif target_best_rate >= 0.65 and target_mass > other_mass:
        convergence_status = "converged_first_pass_pre_x1_mass_bias"
        headline = "pre-x1 coordinate mass often favors target but needs causal test"
    else:
        convergence_status = "not_converged_missing_rollout_failure_split"
        headline = "mixed or weak evidence; causal and control loops needed"
    summary = {
        "stage": "report",
        "status": "ok",
        "headline": headline,
        "convergence_status": convergence_status,
        "case_summary": case_summary,
        "probe_summary_path": str(probe_summary_path),
        "multimodality_summary_path": str(multimodality_summary_path),
        "patching_summary_path": str(patching_summary_path),
        "donor_summary_path": str(donor_summary_path),
        "good_bad_summary_path": str(good_bad_summary_path),
        "rollout_summary_path": str(rollout_summary_path),
        "evidence": {
            "best_pre_x1_probe_accuracy": pre_x1_probe,
            "best_after_x1_probe_accuracy": after_x1_probe,
            "x1_target_strict_best_rate": target_best_rate,
            "x1_mean_target_neighborhood_mass": target_mass,
            "x1_mean_other_candidate_mass": other_mass,
            "schema_context_mean_abs_margin_delta": schema_abs_delta,
            "schema_context_top_candidate_flip_rate": schema_flip_rate,
            "previous_geometry_mean_abs_margin_delta": previous_geometry_abs_delta,
            "current_desc_mean_abs_margin_delta": current_desc_abs_delta,
            "rollout_total": rollout_total,
            "rollout_healthy_count": rollout_healthy_count,
            "rollout_failure_like_count": rollout_failure_like_count,
            "rollout_label_counts": rollout_label_counts,
            "rollout_basin_crosstab": rollout_basin_crosstab,
            "donor_patching_span_summaries": donor_span_summaries,
            "donor_schema_mean_donor_mass_delta": donor_schema_mass_delta,
            "donor_schema_mean_target_mass_delta": donor_schema_target_delta,
            "donor_schema_flip_to_donor_rate": donor_schema_flip_to_donor,
            "donor_current_desc_mean_donor_mass_delta": donor_current_desc_mass_delta,
            "donor_previous_geometry_mean_donor_mass_delta": donor_previous_geometry_mass_delta,
        },
        "next_steps": next_steps,
    }
    if convergence_status.startswith("converged_first_pass"):
        donor_sentence = ""
        if donor_summary:
            donor_sentence = (
                " Donor patching strengthens this: schema-context copies increase "
                f"donor x1 mass by {donor_schema_mass_delta:.3f} on average and "
                f"reduce target mass by {abs(donor_schema_target_delta):.3f}, "
                "while current-desc and previous-geometry donor-mass deltas stay "
                "near zero."
            )
        interpretation = (
            "Evidence supports the mixed view rather than a pure H0 or pure H1. "
            "Instance identity is weakly decodable before x1, but the pre-x1 "
            "coordinate distribution is still multi-modal and only modestly favors "
            "the target. Decodability hardens after x1/y1, and rollout contrast "
            "contains both healthy same-desc continuations and duplicate/near-duplicate "
            "failures. Schema-context attenuation is causally high-impact, so these "
            "tokens are not inert punctuation; the remaining uncertainty is whether "
            "they carry identity directly or route/read out geometry-bearing state."
            f"{donor_sentence}"
        )
    else:
        interpretation = (
            "Probe evidence is decodability evidence, not causal evidence. "
            "The attenuation intervention is causal evidence that the tested span "
            "matters, but it is still blunt compared with donor activation patching. "
            "The conclusion remains intentionally non-converged until good-vs-bad "
            "basin comparisons or donor patching confirm the same boundary."
        )
    report_lines = [
        "# Qwen3-VL Instance Binding Mechanism Report",
        "",
        f"Artifact root: `{artifact_root}`",
        "",
        "## Current Conclusion",
        "",
        f"{headline}.",
        "",
        f"Convergence status: `{convergence_status}`.",
        "",
        "## Evidence",
        "",
        f"- Cases: `{case_summary.get('num_cases', 0)}`.",
        f"- Cohorts: `{case_summary.get('cohort_counts', {})}`.",
        f"- Target ordinal histogram: `{case_summary.get('target_ordinal_hist', {})}`.",
        f"- Best pre-x1 probe accuracy: `{pre_x1_probe:.3f}`.",
        f"- Best post-x1/post-y1 probe accuracy: `{after_x1_probe:.3f}`.",
        f"- X1 target strict-best mass rate: `{target_best_rate:.3f}`.",
        f"- X1 target/other mean mass: `{target_mass:.3f}` / `{other_mass:.3f}`.",
        f"- Schema-context patch mean absolute margin delta: `{schema_abs_delta:.3f}`.",
        f"- Previous-geometry patch mean absolute margin delta: `{previous_geometry_abs_delta:.3f}`.",
        f"- Current-desc patch mean absolute margin delta: `{current_desc_abs_delta:.3f}`.",
        f"- Schema-context top-candidate flip rate: `{schema_flip_rate:.3f}`.",
        f"- Donor schema-context mean donor/target mass delta: `{donor_schema_mass_delta:.3f}` / `{donor_schema_target_delta:.3f}`.",
        f"- Donor current-desc / previous-geometry mean donor mass delta: `{donor_current_desc_mass_delta:.6f}` / `{donor_previous_geometry_mass_delta:.6f}`.",
        f"- Donor schema-context changed-to-donor rate: `{donor_schema_flip_to_donor:.3f}`.",
        f"- Good/bad basin proxy labels: `{good_bad_summary.get('label_counts', {})}`.",
        f"- Rollout failure labels: `{rollout_label_counts}`.",
        f"- Rollout healthy/failure-like counts: `{rollout_healthy_count}` / `{rollout_failure_like_count}`.",
        f"- Rollout vs basin proxy cross-tab: `{rollout_basin_crosstab}`.",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Next Steps",
        "",
        *(f"- {step}" for step in next_steps),
    ]
    stage_dir = artifact_root / "report"
    _write_json(stage_dir / "summary.json", summary)
    (stage_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return summary


def _desc_occurrence_index(
    objects: Sequence[Mapping[str, Any]], *, object_index: int
) -> int:
    target_desc = str(objects[int(object_index)].get("desc") or "")
    return sum(
        1
        for obj in objects[: int(object_index)]
        if str(obj.get("desc") or "") == target_desc
    )


def _intervention_relative_spans(
    *,
    tokenizer: Any,
    assistant_text: str,
    objects: Sequence[Mapping[str, Any]],
    target_object_index: int,
    target_inventory: Mapping[str, Any],
) -> dict[str, list[int]]:
    target_idx = int(target_object_index)
    roles = dict(target_inventory.get("roles") or {})
    desc_positions = [int(pos) for pos in target_inventory.get("desc_positions") or []]
    pre_x1 = int(roles.get("pre_x1", -1))
    desc_end = int(roles.get("desc_end", -1))
    schema_context = (
        list(range(desc_end + 1, pre_x1))
        if desc_end >= 0 and pre_x1 > desc_end + 1
        else []
    )
    previous_idx = target_idx - 1
    previous_geometry: list[int] = []
    if 0 <= previous_idx < len(objects):
        previous_inventory = _target_position_inventory(
            tokenizer=tokenizer,
            assistant_text=assistant_text,
            target_object=objects[previous_idx],
            desc_occurrence_index=_desc_occurrence_index(
                objects, object_index=previous_idx
            ),
        )
        previous_geometry = [
            int(pos) for pos in previous_inventory.get("coord_positions") or []
        ]
    return {
        "current_desc": desc_positions,
        "schema_context": schema_context,
        "previous_geometry": previous_geometry,
        "previous_x1_y1": previous_geometry[:2],
    }


def _decoder_layers(model: Any) -> list[Any]:
    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        nested_model = getattr(model, "model", None)
        language_model = getattr(nested_model, "language_model", None)
        layers = getattr(language_model, "layers", None)
    if isinstance(layers, list):
        return list(layers)
    try:
        import torch

        if isinstance(layers, torch.nn.ModuleList):
            return list(layers)
    except Exception:
        return []
    return []


def _register_token_attenuation_hooks(
    *,
    model: Any,
    token_positions: Sequence[int],
    model_layers: Sequence[int],
    scale: float,
) -> list[Any]:
    import torch

    positions = sorted({int(pos) for pos in token_positions if int(pos) >= 0})
    if not positions:
        return []
    decoder_layers = _decoder_layers(model)
    if not decoder_layers:
        return []
    hooks: list[Any] = []

    def _pre_hook(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if not args:
            return None
        hidden_states = args[0]
        if not isinstance(hidden_states, torch.Tensor):
            return None
        seq_len = int(hidden_states.shape[1])
        active = [pos for pos in positions if pos < seq_len]
        if not active:
            return None
        modified = hidden_states.clone()
        modified[:, active, :] = modified[:, active, :] * float(scale)
        return (modified, *args[1:]), dict(kwargs)

    for raw_layer in model_layers:
        layer_idx = int(raw_layer)
        if layer_idx < 0:
            layer_idx = len(decoder_layers) + layer_idx
        if 0 <= layer_idx < len(decoder_layers):
            hooks.append(
                decoder_layers[layer_idx].register_forward_pre_hook(
                    _pre_hook, with_kwargs=True
                )
            )
    return hooks


def _register_token_copy_hooks(
    *,
    model: Any,
    token_position_pairs: Sequence[tuple[int, int]],
    model_layers: Sequence[int],
) -> list[Any]:
    import torch

    pairs = [
        (int(recipient_pos), int(donor_pos))
        for recipient_pos, donor_pos in token_position_pairs
        if int(recipient_pos) >= 0 and int(donor_pos) >= 0
    ]
    if not pairs:
        return []
    decoder_layers = _decoder_layers(model)
    if not decoder_layers:
        return []
    hooks: list[Any] = []

    def _pre_hook(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if not args:
            return None
        hidden_states = args[0]
        if not isinstance(hidden_states, torch.Tensor):
            return None
        seq_len = int(hidden_states.shape[1])
        active_pairs = [
            (recipient_pos, donor_pos)
            for recipient_pos, donor_pos in pairs
            if recipient_pos < seq_len and donor_pos < seq_len
        ]
        if not active_pairs:
            return None
        modified = hidden_states.clone()
        original = hidden_states
        for recipient_pos, donor_pos in active_pairs:
            modified[:, recipient_pos, :] = original[:, donor_pos, :]
        return (modified, *args[1:]), dict(kwargs)

    for raw_layer in model_layers:
        layer_idx = int(raw_layer)
        if layer_idx < 0:
            layer_idx = len(decoder_layers) + layer_idx
        if 0 <= layer_idx < len(decoder_layers):
            hooks.append(
                decoder_layers[layer_idx].register_forward_pre_hook(
                    _pre_hook, with_kwargs=True
                )
            )
    return hooks


def _candidate_margin_from_alignment(alignment: Mapping[str, Any]) -> dict[str, Any]:
    candidate_rows = alignment.get("candidate_rows") or []
    if not isinstance(candidate_rows, Sequence) or isinstance(
        candidate_rows, (str, bytes)
    ):
        return {
            "target_mass": 0.0,
            "best_other_mass": 0.0,
            "target_vs_best_other_margin": 0.0,
            "top_candidate_label": "",
        }
    target_mass = 0.0
    best_other_mass = 0.0
    top_label = ""
    top_mass = float("-inf")
    for candidate in candidate_rows:
        if not isinstance(candidate, Mapping):
            continue
        label = str(candidate.get("label") or "")
        mass = float(candidate.get("neighborhood_mass", 0.0) or 0.0)
        if mass > top_mass:
            top_mass = mass
            top_label = label
        if label == "target":
            target_mass = mass
        else:
            best_other_mass = max(best_other_mass, mass)
    return {
        "target_mass": target_mass,
        "best_other_mass": best_other_mass,
        "target_vs_best_other_margin": target_mass - best_other_mass,
        "top_candidate_label": top_label,
    }


def _candidate_mass_by_label(alignment: Mapping[str, Any]) -> dict[str, float]:
    candidate_rows = alignment.get("candidate_rows") or []
    if not isinstance(candidate_rows, Sequence) or isinstance(
        candidate_rows, (str, bytes)
    ):
        return {}
    mass_by_label: dict[str, float] = {}
    for candidate in candidate_rows:
        if not isinstance(candidate, Mapping):
            continue
        label = str(candidate.get("label") or "")
        if not label:
            continue
        mass_by_label[label] = float(candidate.get("neighborhood_mass", 0.0) or 0.0)
    return mass_by_label


def _best_other_candidate_from_alignment(
    alignment: Mapping[str, Any],
) -> dict[str, Any] | None:
    candidate_rows = alignment.get("candidate_rows") or []
    if not isinstance(candidate_rows, Sequence) or isinstance(
        candidate_rows, (str, bytes)
    ):
        return None
    best: dict[str, Any] | None = None
    best_mass = float("-inf")
    for candidate in candidate_rows:
        if not isinstance(candidate, Mapping):
            continue
        label = str(candidate.get("label") or "")
        if label == "target":
            continue
        object_index = candidate.get("object_index")
        if object_index is None and label.startswith("candidate_"):
            try:
                object_index = int(label.removeprefix("candidate_"))
            except ValueError:
                object_index = None
        if object_index is None:
            continue
        mass = float(candidate.get("neighborhood_mass", 0.0) or 0.0)
        if mass > best_mass:
            best_mass = mass
            best = {
                "label": label,
                "object_index": int(object_index),
                "mass": mass,
            }
    return best


def _relative_spans_for_object(
    *,
    tokenizer: Any,
    assistant_text: str,
    objects: Sequence[Mapping[str, Any]],
    object_index: int,
) -> dict[str, list[int]]:
    inventory = _target_position_inventory(
        tokenizer=tokenizer,
        assistant_text=assistant_text,
        target_object=objects[int(object_index)],
        desc_occurrence_index=_desc_occurrence_index(
            objects, object_index=int(object_index)
        ),
    )
    return _intervention_relative_spans(
        tokenizer=tokenizer,
        assistant_text=assistant_text,
        objects=objects,
        target_object_index=int(object_index),
        target_inventory=inventory,
    )


def _run_single_case_x1_alignment(
    *,
    scorer: Any,
    case: Mapping[str, Any],
    config: StudyConfig,
    intervention_positions: Sequence[int] = (),
    copy_position_pairs: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    import torch
    from PIL import Image

    from src.coord_tokens.codec import get_coord_token_ids

    objects_raw = case.get("objects") or []
    if not isinstance(objects_raw, Sequence) or isinstance(objects_raw, (str, bytes)):
        raise ValueError(f"case {case.get('case_id')} has invalid objects")
    objects = [obj for obj in objects_raw if isinstance(obj, Mapping)]
    target_idx = int(case.get("target_object_index", 0))
    target_object = objects[target_idx]
    assistant_text = _render_assistant_text(objects)
    inventory = _target_position_inventory(
        tokenizer=scorer.tokenizer,
        assistant_text=assistant_text,
        target_object=target_object,
        desc_occurrence_index=_desc_occurrence_index(objects, object_index=target_idx),
    )
    image = Image.open(str(case["image_path"])).convert("RGB")
    try:
        _, full_messages = scorer.build_messages(
            image=image,
            assistant_text=assistant_text,
            prompt_variant="coco_80",
            object_field_order="desc_first",
        )
        full_text = scorer.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        model_inputs = scorer.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )
        model_inputs = {
            key: value.to(scorer.device) if isinstance(value, torch.Tensor) else value
            for key, value in model_inputs.items()
        }
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("patching forward missing input_ids")
        full_ids = [int(v) for v in input_ids[0].detach().cpu().tolist()]
        assistant_start = _find_subsequence(
            full_ids,
            inventory["assistant_ids"],
            start_hint=max(0, len(full_ids) - len(inventory["assistant_ids"]) - 64),
        )
        if assistant_start is None:
            raise ValueError(f"assistant ids not found for case {case.get('case_id')}")
        absolute_roles = {
            role: int(assistant_start + rel_pos)
            for role, rel_pos in dict(inventory["roles"]).items()
        }
        hooks = _register_token_attenuation_hooks(
            model=scorer.model,
            token_positions=intervention_positions,
            model_layers=config.patching.model_layers,
            scale=config.patching.attenuation_scale,
        )
        hooks.extend(
            _register_token_copy_hooks(
                model=scorer.model,
                token_position_pairs=copy_position_pairs,
                model_layers=config.patching.model_layers,
            )
        )
        try:
            with torch.inference_mode():
                outputs = scorer.model(**model_inputs, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("patching forward missing logits")
        coord_token_ids = get_coord_token_ids(scorer.tokenizer, validate=True)
        full_probs, top_rows = _top_coord_probs(
            logits=logits[0, int(absolute_roles["pre_x1"])],
            coord_token_ids=coord_token_ids,
            top_k=config.multimodality.top_k,
        )
        alignment = candidate_alignment_from_distribution(
            slot="x1",
            probs=full_probs,
            candidates=_candidate_rows_for_case(case),
            target_label="target",
            neighbor_radius=config.multimodality.neighbor_radius,
        )
        return {
            "alignment": alignment,
            "margin": _candidate_margin_from_alignment(alignment),
            "top_coord_bins": top_rows,
            "assistant_start": int(assistant_start),
            "absolute_roles": absolute_roles,
            "relative_spans": _intervention_relative_spans(
                tokenizer=scorer.tokenizer,
                assistant_text=assistant_text,
                objects=objects,
                target_object_index=target_idx,
                target_inventory=inventory,
            ),
        }
    finally:
        image.close()


def _summarize_patching_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_intervention: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_intervention[str(row.get("intervention") or "")].append(row)
    summaries: dict[str, Any] = {}
    for intervention, group_rows in sorted(by_intervention.items()):
        count = len(group_rows)
        if count == 0:
            continue
        summaries[intervention] = {
            "num_rows": count,
            "mean_margin_delta": sum(
                float(row.get("margin_delta", 0.0) or 0.0) for row in group_rows
            )
            / count,
            "mean_abs_margin_delta": sum(
                abs(float(row.get("margin_delta", 0.0) or 0.0)) for row in group_rows
            )
            / count,
            "mean_target_mass_delta": sum(
                float(row.get("target_mass_delta", 0.0) or 0.0) for row in group_rows
            )
            / count,
            "top_candidate_flip_rate": sum(
                bool(row.get("top_candidate_flipped")) for row in group_rows
            )
            / count,
            "empty_span_rate": sum(
                int(row.get("num_intervention_tokens", 0) or 0) == 0
                for row in group_rows
            )
            / count,
        }
    return summaries


def _summarize_donor_patching_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_span: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_span[str(row.get("span") or "")].append(row)
    span_summaries: dict[str, Any] = {}
    for span, group_rows in sorted(by_span.items()):
        count = len(group_rows)
        if count == 0:
            continue
        span_summaries[span] = {
            "num_rows": count,
            "mean_donor_mass_delta": sum(
                float(row.get("donor_mass_delta", 0.0) or 0.0) for row in group_rows
            )
            / count,
            "mean_target_mass_delta": sum(
                float(row.get("target_mass_delta", 0.0) or 0.0) for row in group_rows
            )
            / count,
            "flip_to_donor_rate": sum(
                bool(row.get("top_candidate_flipped_to_donor")) for row in group_rows
            )
            / count,
            "patched_top_is_donor_rate": sum(
                bool(row.get("patched_top_is_donor")) for row in group_rows
            )
            / count,
            "mean_num_position_pairs": sum(
                int(row.get("num_position_pairs", 0) or 0) for row in group_rows
            )
            / count,
        }
    return {
        "num_rows": len(rows),
        "span_summaries": span_summaries,
    }


def _run_patching_stage(
    *,
    config: StudyConfig,
    shard_index: int | None,
    num_shards: int | None,
) -> dict[str, Any]:
    import gc

    import torch

    from src.analysis.unmatched_proposal_verifier import TeacherForcedScorer

    cases = _read_selected_cases(
        config.paths.artifact_root / "cases" / "selected_cases.jsonl"
    )[: max(0, int(config.patching.max_cases))]
    if not cases:
        raise FileNotFoundError("patching requires selected cases")
    active_cases = (
        distribute_items_by_shard(cases, shard_index=shard_index, num_shards=num_shards)
        if shard_index is not None and num_shards is not None
        else cases
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scorer = TeacherForcedScorer(
        checkpoint_path=config.paths.checkpoint,
        device=device,
        attn_implementation="auto",
    )
    rows: list[dict[str, Any]] = []
    processed = 0
    try:
        for case in active_cases:
            baseline = _run_single_case_x1_alignment(
                scorer=scorer,
                case=case,
                config=config,
            )
            baseline_margin = dict(baseline["margin"])
            relative_spans = dict(baseline["relative_spans"])
            assistant_start = int(baseline["assistant_start"])
            for span_name in config.patching.spans:
                rel_positions = [
                    int(pos) for pos in relative_spans.get(str(span_name), [])
                ]
                abs_positions = [assistant_start + pos for pos in rel_positions]
                patched = _run_single_case_x1_alignment(
                    scorer=scorer,
                    case=case,
                    config=config,
                    intervention_positions=abs_positions,
                )
                patched_margin = dict(patched["margin"])
                rows.append(
                    {
                        "case_id": str(case["case_id"]),
                        "desc": str(case.get("desc") or ""),
                        "cohort": str(case.get("cohort") or ""),
                        "intervention": str(span_name),
                        "attenuation_scale": config.patching.attenuation_scale,
                        "model_layers": list(config.patching.model_layers),
                        "num_intervention_tokens": len(abs_positions),
                        "baseline_target_mass": baseline_margin["target_mass"],
                        "patched_target_mass": patched_margin["target_mass"],
                        "target_mass_delta": (
                            float(patched_margin["target_mass"])
                            - float(baseline_margin["target_mass"])
                        ),
                        "baseline_margin": baseline_margin[
                            "target_vs_best_other_margin"
                        ],
                        "patched_margin": patched_margin["target_vs_best_other_margin"],
                        "margin_delta": (
                            float(patched_margin["target_vs_best_other_margin"])
                            - float(baseline_margin["target_vs_best_other_margin"])
                        ),
                        "baseline_top_candidate": baseline_margin[
                            "top_candidate_label"
                        ],
                        "patched_top_candidate": patched_margin["top_candidate_label"],
                        "top_candidate_flipped": (
                            baseline_margin["top_candidate_label"]
                            != patched_margin["top_candidate_label"]
                        ),
                    }
                )
            processed += 1
    finally:
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shard_label = (
        f"shard_{int(shard_index):03d}-of-{int(num_shards):03d}"
        if shard_index is not None and num_shards is not None
        else "shard_all"
    )
    stage_dir = config.paths.artifact_root / "patching"
    _write_jsonl(stage_dir / f"patching_results_{shard_label}.jsonl", rows)
    summary = {
        "stage": "patching",
        "status": "ok",
        "device": device,
        "num_input_cases": len(cases),
        "num_shard_cases": len(active_cases),
        "num_processed_cases": processed,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "intervention_summaries": _summarize_patching_rows(rows),
    }
    _write_json(stage_dir / f"summary_{shard_label}.json", summary)
    return summary


def _run_donor_patching_stage(
    *,
    config: StudyConfig,
    shard_index: int | None,
    num_shards: int | None,
) -> dict[str, Any]:
    import gc

    import torch

    from src.analysis.unmatched_proposal_verifier import TeacherForcedScorer

    cases = _read_selected_cases(
        config.paths.artifact_root / "cases" / "selected_cases.jsonl"
    )[: max(0, int(config.patching.max_cases))]
    if not cases:
        raise FileNotFoundError("donor_patching requires selected cases")
    active_cases = (
        distribute_items_by_shard(cases, shard_index=shard_index, num_shards=num_shards)
        if shard_index is not None and num_shards is not None
        else cases
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scorer = TeacherForcedScorer(
        checkpoint_path=config.paths.checkpoint,
        device=device,
        attn_implementation="auto",
    )
    rows: list[dict[str, Any]] = []
    processed = 0
    skipped_no_donor = 0
    try:
        for case in active_cases:
            objects_raw = case.get("objects") or []
            if not isinstance(objects_raw, Sequence) or isinstance(
                objects_raw, (str, bytes)
            ):
                skipped_no_donor += 1
                continue
            objects = [obj for obj in objects_raw if isinstance(obj, Mapping)]
            baseline = _run_single_case_x1_alignment(
                scorer=scorer,
                case=case,
                config=config,
            )
            baseline_alignment = dict(baseline["alignment"])
            baseline_margin = dict(baseline["margin"])
            donor = _best_other_candidate_from_alignment(baseline_alignment)
            if donor is None:
                skipped_no_donor += 1
                continue
            donor_object_index = int(donor["object_index"])
            if donor_object_index < 0 or donor_object_index >= len(objects):
                skipped_no_donor += 1
                continue
            assistant_text = _render_assistant_text(objects)
            target_spans = dict(baseline["relative_spans"])
            donor_spans = _relative_spans_for_object(
                tokenizer=scorer.tokenizer,
                assistant_text=assistant_text,
                objects=objects,
                object_index=donor_object_index,
            )
            assistant_start = int(baseline["assistant_start"])
            baseline_mass_by_label = _candidate_mass_by_label(baseline_alignment)
            donor_label = str(donor["label"])
            baseline_donor_mass = float(baseline_mass_by_label.get(donor_label, 0.0))
            baseline_target_mass = float(baseline_mass_by_label.get("target", 0.0))
            for span_name in config.patching.spans:
                target_positions = [
                    int(pos) for pos in target_spans.get(str(span_name), [])
                ]
                donor_positions = [
                    int(pos) for pos in donor_spans.get(str(span_name), [])
                ]
                pairs = [
                    (assistant_start + target_pos, assistant_start + donor_pos)
                    for target_pos, donor_pos in zip(target_positions, donor_positions)
                ]
                patched = _run_single_case_x1_alignment(
                    scorer=scorer,
                    case=case,
                    config=config,
                    copy_position_pairs=pairs,
                )
                patched_alignment = dict(patched["alignment"])
                patched_margin = dict(patched["margin"])
                patched_mass_by_label = _candidate_mass_by_label(patched_alignment)
                patched_donor_mass = float(patched_mass_by_label.get(donor_label, 0.0))
                patched_target_mass = float(patched_mass_by_label.get("target", 0.0))
                baseline_top_candidate = str(baseline_margin["top_candidate_label"])
                patched_top_candidate = str(patched_margin["top_candidate_label"])
                rows.append(
                    {
                        "case_id": str(case["case_id"]),
                        "desc": str(case.get("desc") or ""),
                        "cohort": str(case.get("cohort") or ""),
                        "span": str(span_name),
                        "model_layers": list(config.patching.model_layers),
                        "donor_label": donor_label,
                        "donor_object_index": donor_object_index,
                        "num_position_pairs": len(pairs),
                        "baseline_target_mass": baseline_target_mass,
                        "patched_target_mass": patched_target_mass,
                        "target_mass_delta": patched_target_mass - baseline_target_mass,
                        "baseline_donor_mass": baseline_donor_mass,
                        "patched_donor_mass": patched_donor_mass,
                        "donor_mass_delta": patched_donor_mass - baseline_donor_mass,
                        "baseline_margin": baseline_margin[
                            "target_vs_best_other_margin"
                        ],
                        "patched_margin": patched_margin["target_vs_best_other_margin"],
                        "margin_delta": float(
                            patched_margin["target_vs_best_other_margin"]
                        )
                        - float(baseline_margin["target_vs_best_other_margin"]),
                        "baseline_top_candidate": baseline_top_candidate,
                        "patched_top_candidate": patched_top_candidate,
                        "top_candidate_flipped": baseline_top_candidate
                        != patched_top_candidate,
                        "patched_top_is_donor": patched_top_candidate == donor_label,
                        "top_candidate_flipped_to_donor": (
                            baseline_top_candidate != donor_label
                            and patched_top_candidate == donor_label
                        ),
                    }
                )
            processed += 1
    finally:
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shard_label = (
        f"shard_{int(shard_index):03d}-of-{int(num_shards):03d}"
        if shard_index is not None and num_shards is not None
        else "shard_all"
    )
    stage_dir = config.paths.artifact_root / "donor_patching"
    _write_jsonl(stage_dir / f"donor_patching_results_{shard_label}.jsonl", rows)
    summary = {
        "stage": "donor_patching",
        "status": "ok",
        "device": device,
        "num_input_cases": len(cases),
        "num_shard_cases": len(active_cases),
        "num_processed_cases": processed,
        "num_skipped_no_donor": skipped_no_donor,
        "shard_index": shard_index,
        "num_shards": num_shards,
        **_summarize_donor_patching_rows(rows),
    }
    _write_json(stage_dir / f"summary_{shard_label}.json", summary)
    return summary


def _run_merge_donor_patching_stage(config: StudyConfig) -> dict[str, Any]:
    stage_dir = config.paths.artifact_root / "donor_patching"
    rows: list[dict[str, Any]] = []
    for path_str in sorted(
        str(path) for path in stage_dir.glob("donor_patching_results_*.jsonl")
    ):
        rows.extend(_read_jsonl_rows(Path(path_str)))
    if not rows:
        raise FileNotFoundError("merge_donor_patching requires donor_patching rows")
    _write_jsonl(stage_dir / "donor_patching_results_merged.jsonl", rows)
    summary = {
        "stage": "merge_donor_patching",
        "status": "ok",
        **_summarize_donor_patching_rows(rows),
    }
    _write_json(stage_dir / "summary.json", summary)
    return summary


def _run_merge_patching_stage(config: StudyConfig) -> dict[str, Any]:
    stage_dir = config.paths.artifact_root / "patching"
    rows = _read_jsonl_globs([str(stage_dir / "patching_results_*.jsonl")])
    merge_dir = config.paths.artifact_root / "merge_patching"
    summary = {
        "stage": "merge_patching",
        "status": "ok" if rows else "empty",
        "num_rows": len(rows),
        "intervention_summaries": _summarize_patching_rows(rows),
    }
    _write_jsonl(merge_dir / "patching_results_merged.jsonl", rows)
    _write_json(merge_dir / "summary.json", summary)
    return summary


def _run_good_bad_panels_stage(config: StudyConfig) -> dict[str, Any]:
    artifact_root = config.paths.artifact_root
    cases = _read_selected_cases(artifact_root / "cases" / "selected_cases.jsonl")
    cases_by_id = {str(case["case_id"]): case for case in cases}
    probe_summary_path = artifact_root / "binding_probe" / "summary.json"
    probe_summary = (
        json.loads(probe_summary_path.read_text(encoding="utf-8"))
        if probe_summary_path.exists()
        else {}
    )
    pre_x1_best = dict((probe_summary.get("role_best") or {}).get("pre_x1") or {})
    pre_x1_layer = int(pre_x1_best.get("model_layer", 25))
    probe_rows = [
        row
        for row in _read_jsonl_rows(
            artifact_root / "binding_probe" / "probe_predictions.jsonl"
        )
        if str(row.get("role") or "") == "pre_x1"
        and int(row.get("model_layer", -1)) == pre_x1_layer
    ]
    probe_by_case = {str(row["case_id"]): row for row in probe_rows}
    multimodality_rows = [
        row
        for row in _read_jsonl_rows(
            artifact_root / "merge_multimodality" / "coord_multimodality_merged.jsonl"
        )
        if str(row.get("slot") or "") == "x1"
    ]
    multimodality_by_case = {str(row["case_id"]): row for row in multimodality_rows}
    patching_rows = _read_jsonl_rows(
        artifact_root / "merge_patching" / "patching_results_merged.jsonl"
    )
    schema_patch_by_case = {
        str(row["case_id"]): row
        for row in patching_rows
        if str(row.get("intervention") or "") == "schema_context"
    }
    panel_rows: list[dict[str, Any]] = []
    for case_id, case in sorted(cases_by_id.items()):
        mm = multimodality_by_case.get(case_id)
        if not mm:
            continue
        margin = _candidate_margin_from_alignment(mm)
        probe = probe_by_case.get(case_id, {})
        patch = schema_patch_by_case.get(case_id, {})
        try:
            target_ordinal = list(case.get("candidate_object_indices") or []).index(
                int(case.get("target_object_index", -1))
            )
        except ValueError:
            target_ordinal = -1
        target_wins_x1 = bool(margin["target_vs_best_other_margin"] > 0.0)
        probe_correct = bool(probe.get("top1_correct", False))
        schema_abs_delta = abs(float(patch.get("margin_delta", 0.0) or 0.0))
        schema_flipped = bool(patch.get("top_candidate_flipped", False))
        if target_wins_x1 and probe_correct:
            basin_label = "good_basin_proxy"
        elif (not target_wins_x1) and (schema_abs_delta >= 0.02 or schema_flipped):
            basin_label = "bad_basin_proxy"
        elif not target_wins_x1:
            basin_label = "ambiguous_bad_leaning_proxy"
        else:
            basin_label = "ambiguous_good_leaning_proxy"
        panel_rows.append(
            {
                "case_id": case_id,
                "image_id": str(case.get("image_id") or ""),
                "desc": str(case.get("desc") or ""),
                "cohort": str(case.get("cohort") or ""),
                "candidate_count": len(case.get("candidate_object_indices") or []),
                "target_ordinal": int(target_ordinal),
                "basin_label": basin_label,
                "x1_target_mass": margin["target_mass"],
                "x1_best_other_mass": margin["best_other_mass"],
                "x1_target_vs_best_other_margin": margin["target_vs_best_other_margin"],
                "x1_top_candidate_label": margin["top_candidate_label"],
                "pre_x1_probe_top1_correct": probe_correct,
                "pre_x1_probe_rank": int(probe.get("rank", 0) or 0),
                "schema_margin_delta": float(patch.get("margin_delta", 0.0) or 0.0),
                "schema_abs_margin_delta": schema_abs_delta,
                "schema_top_candidate_flipped": schema_flipped,
                "image_path": str(case.get("image_path") or ""),
            }
        )
    label_counts = Counter(str(row["basin_label"]) for row in panel_rows)
    bad_rows = sorted(
        [
            row
            for row in panel_rows
            if str(row["basin_label"])
            in {"bad_basin_proxy", "ambiguous_bad_leaning_proxy"}
        ],
        key=lambda row: (
            float(row["x1_target_vs_best_other_margin"]),
            -float(row["schema_abs_margin_delta"]),
        ),
    )
    good_rows = sorted(
        [
            row
            for row in panel_rows
            if str(row["basin_label"])
            in {"good_basin_proxy", "ambiguous_good_leaning_proxy"}
        ],
        key=lambda row: (
            -float(row["x1_target_vs_best_other_margin"]),
            -float(row["schema_abs_margin_delta"]),
        ),
    )
    stage_dir = artifact_root / "good_bad_panels"
    _write_jsonl(stage_dir / "good_bad_case_panels.jsonl", panel_rows)
    _write_jsonl(stage_dir / "bad_basin_proxy_top.jsonl", bad_rows[:16])
    _write_jsonl(stage_dir / "good_basin_proxy_top.jsonl", good_rows[:16])
    summary = {
        "stage": "good_bad_panels",
        "status": "ok",
        "num_rows": len(panel_rows),
        "label_counts": {
            key: int(value) for key, value in sorted(label_counts.items())
        },
        "method": (
            "Teacher-forced basin proxy from fixed-checkpoint pre-x1 mass, "
            "pre-x1 probe rank, and schema-context attenuation. This is not a "
            "rollout-derived duplicate-collapse label."
        ),
    }
    _write_json(stage_dir / "summary.json", summary)
    return summary


def _rollout_gt_record(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "image_id": case.get("image_id"),
        "images": [str(case.get("image_path") or "")],
        "width": int(case.get("width", 0) or 0),
        "height": int(case.get("height", 0) or 0),
        "objects": list(case.get("objects") or []),
        "metadata": {
            "instance_binding_case_id": str(case.get("case_id") or ""),
            "target_object_index": int(case.get("target_object_index", -1)),
            "candidate_object_indices": list(
                case.get("candidate_object_indices") or []
            ),
            "cohort": str(case.get("cohort") or ""),
            "desc": str(case.get("desc") or ""),
        },
    }


def _run_prepare_rollout_shards_stage(config: StudyConfig) -> dict[str, Any]:
    cases = _read_selected_cases(
        config.paths.artifact_root / "cases" / "selected_cases.jsonl"
    )[: max(0, int(config.rollout.max_cases))]
    if not cases:
        raise FileNotFoundError("prepare_rollout_shards requires selected cases")
    rollout_dir = config.paths.artifact_root / "rollout"
    input_dir = rollout_dir / "inputs"
    config_dir = rollout_dir / "configs"
    run_dir = rollout_dir / "runs"
    shard_count = max(1, int(config.execution.shard_count))
    shard_manifests: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        shard_cases = distribute_items_by_shard(
            cases, shard_index=shard_index, num_shards=shard_count
        )
        shard_label = f"shard_{shard_index:03d}-of-{shard_count:03d}"
        input_path = input_dir / f"{shard_label}.jsonl"
        rows = [_rollout_gt_record(case) for case in shard_cases]
        _write_jsonl(input_path, rows)
        shard_run_dir = run_dir / shard_label
        device = f"cuda:{shard_index}"
        infer_config = {
            "run": {
                "name": shard_label,
                "output_dir": str(run_dir),
            },
            "stages": {"infer": True, "eval": False, "vis": False},
            "infer": {
                "gt_jsonl": str(input_path),
                "model_checkpoint": str(config.paths.checkpoint),
                "prompt_variant": "coco_80",
                "object_field_order": "desc_first",
                "object_ordering": "sorted",
                "mode": "coord",
                "pred_coord_mode": "auto",
                "backend": {
                    "type": "hf",
                    "attn_implementation": "auto",
                },
                "generation": {
                    "temperature": config.rollout.temperature,
                    "top_p": config.rollout.top_p,
                    "max_new_tokens": config.rollout.max_new_tokens,
                    "repetition_penalty": config.rollout.repetition_penalty,
                    "batch_size": config.rollout.batch_size,
                    "seed": config.rollout.seed,
                },
                "device": device,
                "limit": 0,
                "detect_samples": 128,
            },
        }
        config_path = config_dir / f"{shard_label}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(infer_config, sort_keys=False),
            encoding="utf-8",
        )
        shard_manifests.append(
            {
                "shard_index": shard_index,
                "num_shards": shard_count,
                "num_cases": len(shard_cases),
                "input_jsonl": str(input_path),
                "infer_config": str(config_path),
                "run_dir": str(shard_run_dir),
                "gt_vs_pred_jsonl": str(shard_run_dir / "gt_vs_pred.jsonl"),
                "case_ids": [str(case.get("case_id") or "") for case in shard_cases],
            }
        )
    summary = {
        "stage": "prepare_rollout_shards",
        "status": "ok",
        "num_cases": len(cases),
        "num_shards": shard_count,
        "batch_size": config.rollout.batch_size,
        "max_new_tokens": config.rollout.max_new_tokens,
        "shards": shard_manifests,
    }
    _write_json(rollout_dir / "shards_manifest.json", summary)
    return summary


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _center_distance_norm(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return float("inf")
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / 1000.0


def _prediction_bbox_norm1000(
    row: Mapping[str, Any], pred: Mapping[str, Any]
) -> tuple[float, float, float, float] | None:
    points = pred.get("points") or pred.get("bbox_2d")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        return None
    if len(points) != 4:
        return None
    values = [float(v) for v in points]
    coord_mode = str(row.get("coord_mode") or "").lower()
    if coord_mode == "pixel":
        width = max(float(row.get("width", 0) or 0), 1.0)
        height = max(float(row.get("height", 0) or 0), 1.0)
        return (
            values[0] / width * 1000.0,
            values[1] / height * 1000.0,
            values[2] / width * 1000.0,
            values[3] / height * 1000.0,
        )
    return tuple(values)  # type: ignore[return-value]


def _classify_rollout_row(row: Mapping[str, Any]) -> dict[str, Any]:
    pred_raw = row.get("pred") or []
    if not isinstance(pred_raw, Sequence) or isinstance(pred_raw, (str, bytes)):
        pred_raw = []
    preds = [pred for pred in pred_raw if isinstance(pred, Mapping)]
    metadata = row.get("metadata") or {}
    target_desc = ""
    if isinstance(metadata, Mapping):
        target_desc = str(metadata.get("desc") or "")
    same_desc: list[tuple[int, tuple[float, float, float, float]]] = []
    desc_counts: Counter[str] = Counter()
    for idx, pred in enumerate(preds):
        desc = str(pred.get("desc") or "").strip()
        if desc:
            desc_counts[desc] += 1
        if desc != target_desc:
            continue
        bbox = _prediction_bbox_norm1000(row, pred)
        if bbox is not None:
            same_desc.append((idx, bbox))
    duplicate_pairs = 0
    close_pairs = 0
    for idx, (_, box_a) in enumerate(same_desc):
        for _, box_b in same_desc[idx + 1 :]:
            iou = _box_iou(box_a, box_b)
            center_distance = _center_distance_norm(box_a, box_b)
            if iou >= 0.85:
                duplicate_pairs += 1
            if iou >= 0.5 or center_distance <= 0.035:
                close_pairs += 1
    errors = row.get("errors") or []
    error_count = len(errors) if isinstance(errors, Sequence) else 0
    if not preds:
        label = "empty_or_parse_failed"
    elif duplicate_pairs > 0:
        label = "duplicate_collapse_like"
    elif close_pairs > 0:
        label = "near_duplicate_like"
    elif len(same_desc) >= 2:
        label = "healthy_same_desc_multi"
    elif len(same_desc) == 1:
        label = "healthy_single_same_desc"
    else:
        label = "wrong_or_missing_target_desc"
    return {
        "rollout_label": label,
        "pred_count": len(preds),
        "target_desc": target_desc,
        "same_desc_pred_count": len(same_desc),
        "duplicate_pair_count": duplicate_pairs,
        "near_duplicate_pair_count": close_pairs,
        "error_count": error_count,
        "top_pred_desc": desc_counts.most_common(1)[0][0] if desc_counts else "",
        "top_pred_desc_count": desc_counts.most_common(1)[0][1] if desc_counts else 0,
    }


def _run_merge_rollout_shards_stage(config: StudyConfig) -> dict[str, Any]:
    rollout_dir = config.paths.artifact_root / "rollout"
    manifest_path = rollout_dir / "shards_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("merge_rollout_shards requires rollout shard manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_entries = manifest.get("shards") or []
    merged_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for shard in shard_entries:
        if not isinstance(shard, Mapping):
            continue
        path = Path(str(shard.get("gt_vs_pred_jsonl") or ""))
        if not path.exists():
            missing.append(str(path))
            continue
        for row in _read_jsonl_rows(path):
            merged_rows.append(row)
    if missing:
        summary = {
            "stage": "merge_rollout_shards",
            "status": "missing_shards",
            "missing": missing,
            "num_rows": len(merged_rows),
        }
        _write_json(rollout_dir / "merge_summary.json", summary)
        return summary
    split_rows: list[dict[str, Any]] = []
    for row in merged_rows:
        metadata = row.get("metadata") or {}
        case_id = (
            str(metadata.get("instance_binding_case_id") or "")
            if isinstance(metadata, Mapping)
            else ""
        )
        split_rows.append(
            {
                "case_id": case_id,
                "image_id": row.get("image_id"),
                "image": row.get("image"),
                **_classify_rollout_row(row),
            }
        )
    label_counts = Counter(str(row["rollout_label"]) for row in split_rows)
    _write_jsonl(rollout_dir / "gt_vs_pred_merged.jsonl", merged_rows)
    split_dir = config.paths.artifact_root / "rollout_failure_split"
    _write_jsonl(split_dir / "rollout_labels.jsonl", split_rows)
    summary = {
        "stage": "merge_rollout_shards",
        "status": "ok",
        "num_rows": len(merged_rows),
        "label_counts": {
            key: int(value) for key, value in sorted(label_counts.items())
        },
        "merged_gt_vs_pred": str(rollout_dir / "gt_vs_pred_merged.jsonl"),
        "rollout_labels": str(split_dir / "rollout_labels.jsonl"),
    }
    _write_json(rollout_dir / "merge_summary.json", summary)
    _write_json(split_dir / "summary.json", summary)
    return summary


def _write_placeholder_stage_manifest(
    *,
    config: StudyConfig,
    stage: str,
    shard_index: int | None,
    num_shards: int | None,
) -> dict[str, Any]:
    stage_dir = config.paths.artifact_root / stage
    cases = _read_selected_cases(
        config.paths.artifact_root / "cases" / "selected_cases.jsonl"
    )
    shard_cases = (
        distribute_items_by_shard(cases, shard_index=shard_index, num_shards=num_shards)
        if shard_index is not None and num_shards is not None and cases
        else cases
    )
    summary = {
        "stage": stage,
        "status": "planned_not_executed",
        "num_input_cases": len(cases),
        "num_shard_cases": len(shard_cases),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "message": (
            "GPU/model-forward implementation is intentionally separated from "
            "the CPU contract scaffolding."
        ),
    }
    _write_json(stage_dir / "summary.json", summary)
    return summary


def run_study_stage(
    *,
    config_path: Path,
    stage: str,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> dict[str, Any]:
    config = load_study_config(config_path)
    _write_resolved_config(config)
    config.paths.artifact_root.mkdir(parents=True, exist_ok=True)
    if stage == "audit":
        audit = audit_checkpoint_surface(config.paths.checkpoint)
        audit["dataset_jsonls"] = [str(path) for path in config.paths.dataset_jsonls]
        audit["artifact_root"] = str(config.paths.artifact_root)
        audit["execution"] = {
            "gpu_ids": list(config.execution.gpu_ids),
            "shard_count": config.execution.shard_count,
            "per_gpu_generation_batch_size": config.execution.per_gpu_generation_batch_size,
            "per_gpu_teacher_forced_batch_size": config.execution.per_gpu_teacher_forced_batch_size,
        }
        _write_json(
            config.paths.artifact_root / "audit" / "checkpoint_audit.json", audit
        )
        summary = {
            "stage": "audit",
            "surface": audit["surface"],
            "checkpoint": str(config.paths.checkpoint),
        }
        _write_json(config.paths.artifact_root / "audit" / "summary.json", summary)
        return summary
    if stage == "select_cases":
        all_cases: list[InstanceBindingCase] = []
        for dataset_jsonl in config.paths.dataset_jsonls:
            remaining = max(0, config.case_selection.max_cases - len(all_cases))
            if remaining <= 0:
                break
            selection_cfg = CaseSelectionConfig(
                priority_descs=config.case_selection.priority_descs,
                max_cases=remaining,
                min_same_desc_candidates=config.case_selection.min_same_desc_candidates,
                include_sparse_controls=config.case_selection.include_sparse_controls,
                max_sparse_controls=config.case_selection.max_sparse_controls,
            )
            all_cases.extend(mine_repeated_desc_cases(dataset_jsonl, selection_cfg))
        rows = [case.to_json() for case in all_cases]
        cases_dir = config.paths.artifact_root / "cases"
        _write_jsonl(cases_dir / "selected_cases.jsonl", rows)
        summary = {"stage": "select_cases", **_case_summary(all_cases)}
        _write_json(cases_dir / "summary.json", summary)
        if shard_index is not None and num_shards is not None:
            shard_rows = distribute_items_by_shard(
                rows, shard_index=shard_index, num_shards=num_shards
            )
            shard_manifest = {
                "stage": "select_cases",
                "shard_index": shard_index,
                "num_shards": num_shards,
                "num_cases": len(shard_rows),
                "case_ids": [str(row["case_id"]) for row in shard_rows],
            }
            _write_json(
                cases_dir / f"shard_{shard_index:03d}-of-{num_shards:03d}.json",
                shard_manifest,
            )
        return summary
    if stage == "hidden_states":
        return _run_forward_extraction_stage(
            config=config,
            stage=stage,
            shard_index=shard_index,
            num_shards=num_shards,
            save_hidden_vectors=True,
        )
    if stage == "multimodality":
        return _run_forward_extraction_stage(
            config=config,
            stage=stage,
            shard_index=shard_index,
            num_shards=num_shards,
            save_hidden_vectors=False,
        )
    if stage == "binding_probe":
        return _run_binding_probe_stage(config)
    if stage == "merge_multimodality":
        return _run_merge_multimodality_stage(config)
    if stage == "report":
        return _run_report_stage(config)
    if stage == "patching":
        return _run_patching_stage(
            config=config,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    if stage == "merge_patching":
        return _run_merge_patching_stage(config)
    if stage == "donor_patching":
        return _run_donor_patching_stage(
            config=config,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    if stage == "merge_donor_patching":
        return _run_merge_donor_patching_stage(config)
    if stage == "good_bad_panels":
        return _run_good_bad_panels_stage(config)
    if stage == "prepare_rollout_shards":
        return _run_prepare_rollout_shards_stage(config)
    if stage in {"merge_rollout_shards", "rollout_failure_split"}:
        return _run_merge_rollout_shards_stage(config)
    if stage in {
        "merge_hidden_states",
    }:
        return _write_placeholder_stage_manifest(
            config=config,
            stage=stage,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    raise ValueError(f"unknown qwen3-vl instance-binding study stage: {stage}")


__all__ = [
    "REQUIRED_POSITION_ROLES",
    "CaseSelectionConfig",
    "ExecutionConfig",
    "InstanceBindingCase",
    "MultimodalityConfig",
    "PatchingConfig",
    "ResolvedRuntimePaths",
    "RolloutConfig",
    "RuntimePathConfig",
    "StudyConfig",
    "audit_checkpoint_surface",
    "candidate_alignment_from_distribution",
    "classify_mechanism",
    "distribute_items_by_shard",
    "load_study_config",
    "mine_repeated_desc_cases",
    "resolve_runtime_paths",
    "run_study_stage",
    "select_hidden_state_layers",
    "validate_position_inventory",
]

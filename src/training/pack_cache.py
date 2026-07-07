"""Internal packed-micro-step cache for expensive dataset/template packing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any

from src.augmentation.geometry import GEOMETRY_FLIP_POLICY_VERSION
from src.config.models import DatasetSplitConfig, TrainConfig
from src.training.schedule import ResolvedStepSchedule
from src.training.supervised_trainer import SupervisedMicroStep


PACKING_CACHE_VERSION = "coordexp-swift-pack-cache-v1"
PACKING_CACHE_MANIFEST = "manifest.json"
PACKING_CACHE_CHUNK_DIR = "chunks"
DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS = 16
PACKING_CACHE_MATERIALIZATION_STRATEGY = "fork_process_pool"
PACKING_CACHE_CODE_IDENTITY_FILES = {
    "augmentation_factory": "src/augmentation/factory.py",
    "augmentation_geometry": "src/augmentation/geometry.py",
    "augmentation_processor": "src/augmentation/processor.py",
    "template_renderer": "src/templates/renderer.py",
    "qwen_encoding": "src/qwen/encoding.py",
    "qwen_images": "src/qwen/images.py",
    "qwen_positions": "src/qwen/positions.py",
    "qwen_fa2": "src/qwen/fa2.py",
    "qwen_forward": "src/qwen/forward.py",
    "packing_planner": "src/packing/planner.py",
    "packing_supervision": "src/packing/supervision.py",
    "supervision_tokens": "src/supervision/tokens.py",
}


def build_packing_cache_materialization(
    *,
    workers: int | None = None,
    strategy: str = PACKING_CACHE_MATERIALIZATION_STRATEGY,
) -> dict[str, Any]:
    resolved_workers = (
        DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS
        if workers is None
        else workers
    )
    if isinstance(resolved_workers, bool) or not isinstance(resolved_workers, int):
        raise ValueError("packing cache materialization workers must be an integer")
    if resolved_workers <= 0:
        raise ValueError("packing cache materialization workers must be positive")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("packing cache materialization strategy must be non-empty")
    return {"strategy": strategy, "workers": resolved_workers}


def build_packing_cache_fingerprint(
    config: TrainConfig,
    components: Any,
    *,
    dataset: DatasetSplitConfig,
    split: str,
) -> str:
    determinants = build_packing_cache_determinants(
        config,
        components,
        dataset=dataset,
        split=split,
    )
    return hashlib.sha256(_canonical_json(determinants).encode("utf-8")).hexdigest()


def build_packing_cache_determinants(
    config: TrainConfig,
    components: Any,
    *,
    dataset: DatasetSplitConfig,
    split: str,
) -> dict[str, Any]:
    dataset_path = Path(dataset.path).expanduser().resolve()
    stat = dataset_path.stat()
    processor_identity = getattr(components, "processor_identity")
    token_identity = getattr(components, "token_identity")
    base_model_path = Path(getattr(components, "base_model_path")).expanduser().resolve()
    tokenizer = getattr(components, "tokenizer", None)
    return {
        "version": PACKING_CACHE_VERSION,
        "split": split,
        "dataset": {
            "path": str(dataset_path),
            "sample_limit": dataset.sample_limit,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": _file_sha256(dataset_path),
        },
        "template": config.template.model_dump(mode="json"),
        "packing": {
            "global_max_length": config.packing.global_max_length,
        },
        "processor": config.model.processor.model_dump(mode="json"),
        "ordering": {
            "train_order": config.data.train_order,
            "runtime_seed": config.runtime.seed,
        },
        "augmentation": _augmentation_determinants(config, split=split),
        "qwen": {
            "base_model_path": str(base_model_path),
            "processor_identity": processor_identity.to_artifact_dict(),
            "token_identity": token_identity.to_artifact_dict(),
            "encoding_identity": _qwen_encoding_identity(components, tokenizer=tokenizer),
        },
        "code_identity": _packing_cache_code_identity(),
    }


def cache_dir_for_fingerprint(cache_root: str | Path, fingerprint: str) -> Path:
    return Path(cache_root).expanduser().resolve() / fingerprint


def manifest_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / PACKING_CACHE_MANIFEST


def load_cache_manifest(cache_dir: str | Path) -> dict[str, Any]:
    with manifest_path(cache_dir).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_manifest(manifest, cache_dir=Path(cache_dir))
    return manifest


def cache_is_complete(cache_dir: str | Path, *, fingerprint: str) -> bool:
    path = manifest_path(cache_dir)
    if not path.exists():
        return False
    try:
        manifest = load_cache_manifest(cache_dir)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return (
        manifest.get("version") == PACKING_CACHE_VERSION
        and manifest.get("fingerprint") == fingerprint
        and manifest.get("status") == "complete"
    )


def write_micro_step_cache(
    cache_dir: str | Path,
    micro_steps: Sequence[SupervisedMicroStep],
    *,
    fingerprint: str,
    determinants: Mapping[str, Any],
    chunk_size: int = 512,
    materialization: Mapping[str, Any] | None = None,
    augmentation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    materialization_payload = _coerce_materialization(materialization)
    root = Path(cache_dir)
    chunks_dir = root / PACKING_CACHE_CHUNK_DIR
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, len(micro_steps), chunk_size)):
        end = min(start + chunk_size, len(micro_steps))
        chunk_name = f"chunk-{chunk_index:05d}.pkl"
        chunk_path = chunks_dir / chunk_name
        with chunk_path.open("wb") as handle:
            pickle.dump(tuple(micro_steps[start:end]), handle, protocol=pickle.HIGHEST_PROTOCOL)
        chunks.append(
            {
                "path": f"{PACKING_CACHE_CHUNK_DIR}/{chunk_name}",
                "start": start,
                "end": end,
                "count": end - start,
                "sha256": _file_sha256(chunk_path),
            }
        )
    manifest = {
        "version": PACKING_CACHE_VERSION,
        "status": "complete",
        "fingerprint": fingerprint,
        "determinants": dict(determinants),
        "micro_step_count": len(micro_steps),
        "chunk_size": chunk_size,
        "chunks": chunks,
        "materialization": materialization_payload,
    }
    if augmentation is not None:
        manifest["augmentation"] = dict(augmentation)
    _atomic_write_json(manifest_path(root), manifest)
    return manifest


def load_rank_micro_steps_from_cache(
    cache_dir: str | Path,
    *,
    schedule: ResolvedStepSchedule,
    rank: int,
    world_size: int,
) -> tuple[SupervisedMicroStep, ...]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be inside world_size")
    if schedule.runtime_batch.world_size != world_size:
        raise ValueError("world_size must match schedule runtime_batch")
    manifest = load_cache_manifest(cache_dir)
    micro_step_count = int(manifest["micro_step_count"])
    indices = _rank_local_pack_indices(
        schedule,
        rank=rank,
        world_size=world_size,
        micro_step_count=micro_step_count,
    )
    required = set(indices)
    loaded: dict[int, SupervisedMicroStep] = {}
    root = Path(cache_dir)
    for chunk in manifest["chunks"]:
        start = int(chunk["start"])
        end = int(chunk["end"])
        if not any(start <= index < end for index in required):
            continue
        chunk_path = root / str(chunk["path"])
        _validate_chunk_sha256(chunk_path, expected_sha256=str(chunk["sha256"]))
        with chunk_path.open("rb") as handle:
            chunk_steps = pickle.load(handle)
        for offset, micro_step in enumerate(chunk_steps):
            absolute_index = start + offset
            if absolute_index in required:
                loaded[absolute_index] = micro_step
    return tuple(loaded[index] for index in indices)


def load_all_micro_steps_from_cache(cache_dir: str | Path) -> tuple[SupervisedMicroStep, ...]:
    manifest = load_cache_manifest(cache_dir)
    loaded: list[SupervisedMicroStep] = []
    root = Path(cache_dir)
    for chunk in manifest["chunks"]:
        chunk_path = root / str(chunk["path"])
        _validate_chunk_sha256(chunk_path, expected_sha256=str(chunk["sha256"]))
        with chunk_path.open("rb") as handle:
            chunk_steps = pickle.load(handle)
        loaded.extend(chunk_steps)
    if len(loaded) != int(manifest["micro_step_count"]):
        raise ValueError("packing cache load did not cover every micro-step")
    return tuple(loaded)


def _rank_local_pack_indices(
    schedule: ResolvedStepSchedule,
    *,
    rank: int,
    world_size: int,
    micro_step_count: int,
) -> tuple[int, ...]:
    if micro_step_count <= 0:
        raise ValueError("micro_step_count must be positive")
    local_count = (
        schedule.resolved_max_steps
        * schedule.runtime_batch.resolved_grad_accum_steps
    )
    indices: list[int] = []
    for local_index in range(local_count):
        planned_step_index = (
            local_index // schedule.runtime_batch.resolved_grad_accum_steps
        )
        local_accum_index = (
            local_index % schedule.runtime_batch.resolved_grad_accum_steps
        )
        global_micro_step_index = (
            planned_step_index * schedule.runtime_batch.effective_batch_size
            + local_accum_index * world_size
            + rank
        )
        indices.append(global_micro_step_index % micro_step_count)
    return tuple(indices)


def _validate_manifest(manifest: dict[str, Any], *, cache_dir: Path) -> None:
    if manifest.get("version") != PACKING_CACHE_VERSION:
        raise ValueError("unsupported packing cache version")
    if manifest.get("status") != "complete":
        raise ValueError("packing cache manifest is not complete")
    count = int(manifest["micro_step_count"])
    if count <= 0:
        raise ValueError("packing cache must contain at least one micro-step")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("packing cache manifest must contain chunks")
    materialization = manifest.get("materialization")
    if materialization is not None:
        _coerce_materialization(materialization)
    expected_start = 0
    for chunk in chunks:
        start = int(chunk["start"])
        end = int(chunk["end"])
        count_from_range = end - start
        if start != expected_start:
            raise ValueError("packing cache chunks must be contiguous")
        if count_from_range <= 0 or int(chunk["count"]) != count_from_range:
            raise ValueError("packing cache chunk count must match range")
        expected_start = end
        path = cache_dir / str(chunk["path"])
        if not path.exists():
            raise ValueError(f"packing cache chunk is missing: {path}")
        sha256 = chunk.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError("packing cache chunk must record sha256")
    if expected_start != count:
        raise ValueError("packing cache chunks must cover every micro-step")


def _coerce_materialization(
    materialization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if materialization is None:
        return build_packing_cache_materialization()
    if not isinstance(materialization, Mapping):
        raise ValueError("packing cache materialization must be a mapping")
    strategy = materialization.get("strategy", PACKING_CACHE_MATERIALIZATION_STRATEGY)
    workers = materialization.get("workers")
    if workers is None:
        raise ValueError("packing cache materialization must record workers")
    return build_packing_cache_materialization(
        workers=workers,
        strategy=strategy,
    )


def _validate_chunk_sha256(path: Path, *, expected_sha256: str) -> None:
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"packing cache chunk checksum mismatch: {path} "
            f"(expected {expected_sha256}, got {actual_sha256})"
        )


def _qwen_encoding_identity(components: Any, *, tokenizer: Any) -> dict[str, Any]:
    image_pad_token_id = None
    if tokenizer is not None:
        convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            image_pad_token_id = convert_tokens_to_ids("<|image_pad|>")
    return {
        "processor_class": type(getattr(components, "processor", None)).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "image_pad_token_id": image_pad_token_id,
        "chat_template_sha256": _stable_text_sha256(
            getattr(tokenizer, "chat_template", None)
            or getattr(getattr(components, "processor", None), "chat_template", None)
        ),
        "package_versions": dict(getattr(components, "package_versions", {}) or {}),
    }


def _augmentation_determinants(config: TrainConfig, *, split: str) -> dict[str, Any]:
    enabled_for_split = split == "train"
    geometry_flips = config.data.augmentation.train.geometry_flips
    return {
        "policy": "geometry_flips",
        "policy_version": GEOMETRY_FLIP_POLICY_VERSION,
        "split": split,
        "train_only": True,
        "seed": int(config.runtime.seed),
        "seed_source": "runtime.seed",
        "effective_enabled": bool(enabled_for_split and geometry_flips.enabled),
        "train": {
            "geometry_flips": geometry_flips.model_dump(mode="json"),
        },
    }


def _packing_cache_code_identity() -> dict[str, dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[2]
    identity: dict[str, dict[str, str]] = {}
    for name, relative_path in sorted(PACKING_CACHE_CODE_IDENTITY_FILES.items()):
        path = repo_root / relative_path
        identity[name] = {
            "path": relative_path,
            "sha256": _file_sha256(path),
        }
    return identity


def _stable_text_sha256(value: Any) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS",
    "PACKING_CACHE_MANIFEST",
    "PACKING_CACHE_MATERIALIZATION_STRATEGY",
    "PACKING_CACHE_VERSION",
    "build_packing_cache_materialization",
    "build_packing_cache_determinants",
    "build_packing_cache_fingerprint",
    "cache_dir_for_fingerprint",
    "cache_is_complete",
    "load_all_micro_steps_from_cache",
    "load_cache_manifest",
    "load_rank_micro_steps_from_cache",
    "manifest_path",
    "write_micro_step_cache",
]

"""Internal packed-micro-step cache for expensive dataset/template packing."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import fcntl
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import shutil
from typing import Any
import uuid

import torch

from src.augmentation.geometry import GEOMETRY_FLIP_POLICY_VERSION
from src.config.models import DatasetSplitConfig, TrainConfig
from src.training.schedule import ResolvedStepSchedule
from src.training.supervised_trainer import SupervisedMicroStep


PACKING_CACHE_VERSION = "coordexp-swift-pack-cache-v2"
PACKING_CACHE_MANIFEST = "manifest.json"
PACKING_CACHE_CHUNK_DIR = "chunks"
DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS = 16
PACKING_CACHE_MATERIALIZATION_STRATEGY = "fork_process_pool"
PACK_CACHE_VERIFICATION_LEVELS = ("manifest", "payloads")
PACKING_CACHE_CODE_IDENTITY_FILES = {
    "augmentation_factory": "src/augmentation/factory.py",
    "augmentation_geometry": "src/augmentation/geometry.py",
    "augmentation_processor": "src/augmentation/processor.py",
    "coordinate_targets": "src/coordinate_targets.py",
    "template_spans": "src/templates/spans.py",
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


class PackingCacheInvalidError(ValueError):
    """The cache root cannot be consumed under the current strict contract."""


_INVALID_CACHE_ERRORS = (
    OSError,
    json.JSONDecodeError,
    pickle.PickleError,
    ImportError,
    AttributeError,
    EOFError,
    RuntimeError,
    TypeError,
    KeyError,
    ValueError,
)

_ALLOWED_PICKLE_GLOBALS = {
    ("src.training.supervised_trainer", "SupervisedMicroStep"),
    ("src.packing.planner", "PackedSequence"),
    ("src.packing.planner", "PackedSegment"),
    ("src.qwen.encoding", "EncodedExample"),
    ("src.qwen.encoding", "EncodedTokenSpan"),
    ("src.qwen.images", "QwenImageEncoding"),
    ("src.qwen.images", "QwenNoResizeImagePlan"),
    ("pathlib", "PosixPath"),
    ("src.coordinate_targets", "CoordinateLossTarget"),
    ("src.qwen.positions", "QwenPositionInputs"),
    ("src.qwen.positions", "QwenPositionBoundaryValidation"),
    ("src.qwen.positions", "QwenPositionSegmentSummary"),
    ("src.supervision.tokens", "TokenSequence"),
    ("src.supervision.tokens", "TokenAtom"),
    ("src.supervision.tokens", "TokenSpan"),
    ("src.losses.vocab", "TokenVocabularyGroups"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("collections", "OrderedDict"),
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


def load_cache_manifest(
    cache_dir: str | Path,
    *,
    expected_fingerprint: str,
    level: str,
) -> dict[str, Any]:
    if level not in PACK_CACHE_VERIFICATION_LEVELS:
        raise ValueError(
            "packing cache verification level must be one of "
            f"{PACK_CACHE_VERIFICATION_LEVELS}"
        )
    try:
        manifest = _load_validated_manifest(
            Path(cache_dir), expected_fingerprint=expected_fingerprint
        )
        if level == "payloads":
            for _start, _chunk_steps in _iter_validated_chunks(
                Path(cache_dir), manifest
            ):
                pass
        return manifest
    except PackingCacheInvalidError:
        raise
    except _INVALID_CACHE_ERRORS as exc:
        raise PackingCacheInvalidError(f"invalid packing cache: {exc}") from exc


def cache_is_complete(cache_dir: str | Path, *, fingerprint: str) -> bool:
    path = manifest_path(cache_dir)
    if not path.exists():
        return False
    try:
        manifest = load_cache_manifest(
            cache_dir,
            expected_fingerprint=fingerprint,
            level="payloads",
        )
    except PackingCacheInvalidError:
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
    materialization: Mapping[str, Any],
    augmentation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    materialization_payload = _coerce_materialization(materialization)
    if not micro_steps:
        raise ValueError("packing cache must contain at least one micro-step")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("fingerprint must be a non-empty string")
    if not isinstance(determinants, Mapping):
        raise ValueError("determinants must be a mapping")
    if fingerprint != _determinant_fingerprint(determinants):
        raise ValueError("fingerprint must match canonical determinants")
    augmentation_payload = _validate_augmentation_receipt(augmentation)
    root = Path(cache_dir)
    root.parent.mkdir(parents=True, exist_ok=True)
    # Serialize mutations by physical cache root, not only by semantic identity.
    # The canonical pipeline uses one root per fingerprint, while this lower-level
    # API also remains safe if a direct caller reuses a root for new determinants.
    lock_path = root.parent / f".{root.name}.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if root.exists():
                try:
                    existing_manifest = load_cache_manifest(
                        root,
                        expected_fingerprint=fingerprint,
                        level="payloads",
                    )
                    if existing_manifest["determinants"] == dict(determinants):
                        return existing_manifest
                except PackingCacheInvalidError:
                    pass
            _cleanup_stale_cache_siblings(root)
            return _publish_micro_step_cache(
                root,
                micro_steps,
                fingerprint=fingerprint,
                determinants=determinants,
                chunk_size=chunk_size,
                materialization=materialization_payload,
                augmentation=augmentation_payload,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _publish_micro_step_cache(
    root: Path,
    micro_steps: Sequence[SupervisedMicroStep],
    *,
    fingerprint: str,
    determinants: Mapping[str, Any],
    chunk_size: int,
    materialization: Mapping[str, Any],
    augmentation: Mapping[str, Any],
) -> dict[str, Any]:
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    stage = root.with_name(f".{root.name}.stage-{token}")
    backup = root.with_name(f".{root.name}.backup-{token}")
    manifest = {
        "version": PACKING_CACHE_VERSION,
        "status": "complete",
        "fingerprint": fingerprint,
        "determinants": dict(determinants),
        "micro_step_count": len(micro_steps),
        "chunk_size": chunk_size,
        "chunks": [],
        "materialization": dict(materialization),
        "augmentation": dict(augmentation),
    }
    try:
        chunks_dir = stage / PACKING_CACHE_CHUNK_DIR
        chunks_dir.mkdir(parents=True)
        for chunk_index, start in enumerate(range(0, len(micro_steps), chunk_size)):
            end = min(start + chunk_size, len(micro_steps))
            chunk_name = f"chunk-{chunk_index:05d}.pkl"
            chunk_path = chunks_dir / chunk_name
            with chunk_path.open("wb") as handle:
                pickle.dump(
                    tuple(micro_steps[start:end]),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            manifest["chunks"].append(
                {
                    "path": f"{PACKING_CACHE_CHUNK_DIR}/{chunk_name}",
                    "start": start,
                    "end": end,
                    "count": end - start,
                    "sha256": _file_sha256(chunk_path),
                }
            )
        # The complete manifest is the final write inside the isolated stage.
        _atomic_write_json(manifest_path(stage), manifest)
        load_cache_manifest(
            stage,
            expected_fingerprint=fingerprint,
            level="payloads",
        )
        if root.exists():
            os.replace(root, backup)
        try:
            os.replace(stage, root)
        except Exception:
            if backup.exists() and not root.exists():
                os.replace(backup, root)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if backup.exists():
            if not root.exists():
                os.replace(backup, root)
            else:
                shutil.rmtree(backup)


def load_rank_micro_steps_from_cache(
    cache_dir: str | Path,
    *,
    expected_fingerprint: str,
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
    try:
        root = Path(cache_dir)
        manifest = _load_validated_manifest(
            root, expected_fingerprint=expected_fingerprint
        )
        micro_step_count = int(manifest["micro_step_count"])
        indices = _rank_local_pack_indices(
            schedule,
            rank=rank,
            world_size=world_size,
            micro_step_count=micro_step_count,
        )
        required = set(indices)
        selected: dict[int, SupervisedMicroStep] = {}
        for start, chunk_steps in _iter_validated_chunks(root, manifest):
            for offset, micro_step in enumerate(chunk_steps):
                absolute_index = start + offset
                if absolute_index in required:
                    selected[absolute_index] = micro_step
        return tuple(selected[index] for index in indices)
    except PackingCacheInvalidError:
        raise
    except _INVALID_CACHE_ERRORS as exc:
        raise PackingCacheInvalidError(f"invalid packing cache: {exc}") from exc


def load_all_micro_steps_from_cache(
    cache_dir: str | Path, *, expected_fingerprint: str
) -> tuple[SupervisedMicroStep, ...]:
    try:
        root = Path(cache_dir)
        manifest = _load_validated_manifest(
            root, expected_fingerprint=expected_fingerprint
        )
        loaded: list[SupervisedMicroStep] = []
        for _start, chunk_steps in _iter_validated_chunks(root, manifest):
            loaded.extend(chunk_steps)
        return tuple(loaded)
    except PackingCacheInvalidError:
        raise
    except _INVALID_CACHE_ERRORS as exc:
        raise PackingCacheInvalidError(f"invalid packing cache: {exc}") from exc


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


def _validate_manifest(
    manifest: dict[str, Any], *, cache_dir: Path, expected_fingerprint: str
) -> None:
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise ValueError("expected_fingerprint must be a non-empty string")
    if manifest.get("version") != PACKING_CACHE_VERSION:
        raise ValueError("unsupported packing cache version")
    if manifest.get("status") != "complete":
        raise ValueError("packing cache manifest is not complete")
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError("packing cache fingerprint does not match expected fingerprint")
    if not isinstance(manifest.get("determinants"), Mapping):
        raise ValueError("packing cache determinants must be a mapping")
    if manifest["fingerprint"] != _determinant_fingerprint(manifest["determinants"]):
        raise ValueError("packing cache fingerprint does not match canonical determinants")
    if not isinstance(manifest.get("materialization"), Mapping):
        raise ValueError("packing cache materialization must be a mapping")
    _coerce_materialization(manifest["materialization"])
    _validate_augmentation_receipt(manifest.get("augmentation"))
    count = _positive_int(manifest.get("micro_step_count"), field="micro_step_count")
    _positive_int(manifest.get("chunk_size"), field="chunk_size")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("packing cache manifest must contain chunks")
    expected_start = 0
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            raise ValueError("packing cache chunk declarations must be mappings")
        start = _nonnegative_int(chunk.get("start"), field="chunk start")
        end = _positive_int(chunk.get("end"), field="chunk end")
        count_from_range = end - start
        if start != expected_start:
            raise ValueError("packing cache chunks must be contiguous")
        if count_from_range <= 0 or _positive_int(
            chunk.get("count"), field="chunk count"
        ) != count_from_range:
            raise ValueError("packing cache chunk count must match range")
        expected_start = end
        path = _safe_chunk_path(cache_dir, chunk.get("path"))
        if not path.exists():
            raise ValueError(f"packing cache chunk is missing: {path}")
        sha256 = chunk.get("sha256")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError("packing cache chunk must record a valid sha256")
    if expected_start != count:
        raise ValueError("packing cache chunks must cover every micro-step")


def _load_validated_manifest(
    cache_dir: Path, *, expected_fingerprint: str
) -> dict[str, Any]:
    with manifest_path(cache_dir).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("packing cache manifest must be a mapping")
    _validate_manifest(
        manifest,
        cache_dir=cache_dir,
        expected_fingerprint=expected_fingerprint,
    )
    return manifest


def _iter_validated_chunks(
    cache_dir: Path, manifest: Mapping[str, Any]
) -> Iterator[tuple[int, tuple[SupervisedMicroStep, ...]]]:
    for chunk in manifest["chunks"]:
        chunk_path = _safe_chunk_path(cache_dir, chunk["path"])
        _validate_chunk_sha256(chunk_path, expected_sha256=chunk["sha256"])
        try:
            with chunk_path.open("rb") as handle:
                chunk_steps = _RestrictedCacheUnpickler(handle).load()
        except _INVALID_CACHE_ERRORS as exc:
            raise ValueError(f"packing cache chunk payload is unreadable: {chunk_path}") from exc
        if not isinstance(chunk_steps, tuple):
            raise ValueError("packing cache chunk payload must be a tuple")
        if len(chunk_steps) != chunk["count"]:
            raise ValueError("packing cache chunk payload length must match declared count")
        if not all(isinstance(step, SupervisedMicroStep) for step in chunk_steps):
            raise ValueError("packing cache chunk payload must contain supervised micro-steps")
        yield chunk["start"], chunk_steps


def _safe_chunk_path(cache_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("packing cache chunk path must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("packing cache chunk path must be relative")
    root = cache_dir.resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("packing cache chunk path must stay inside cache root")
    return path


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"packing cache {field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"packing cache {field} must be a non-negative integer")
    return value


def _coerce_materialization(
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(materialization, Mapping) or not materialization:
        raise ValueError("packing cache materialization must be a non-empty mapping")
    strategy = materialization.get("strategy")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("packing cache materialization must record a non-empty strategy")
    workers = _positive_int(materialization.get("workers"), field="materialization workers")
    return build_packing_cache_materialization(
        workers=workers,
        strategy=strategy,
    )


def _validate_augmentation_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("packing cache augmentation must be a non-empty mapping")
    required = {
        "split",
        "mode",
        "policy",
        "enabled",
        "seed",
        "input_example_count",
        "output_example_count",
        "presentation_count",
        "object_ordering",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(
            f"packing cache augmentation is missing fields: {sorted(missing)}"
        )
    for field in ("split", "mode", "policy", "object_ordering"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"packing cache augmentation {field} must be non-empty")
    if not isinstance(value["enabled"], bool):
        raise ValueError("packing cache augmentation enabled must be boolean")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
        raise ValueError("packing cache augmentation seed must be an integer")
    for field in ("input_example_count", "output_example_count", "presentation_count"):
        _nonnegative_int(value[field], field=f"augmentation {field}")
    if value["enabled"]:
        enabled_fields = {
            "policy_version",
            "horizontal_prob",
            "vertical_prob",
            "transform_counts",
            "random_object_order_presentations",
        }
        missing_enabled = enabled_fields.difference(value)
        if missing_enabled:
            raise ValueError(
                "packing cache enabled augmentation is missing fields: "
                f"{sorted(missing_enabled)}"
            )
        if not isinstance(value["policy_version"], str) or not value["policy_version"]:
            raise ValueError("packing cache augmentation policy_version must be non-empty")
        for field in ("horizontal_prob", "vertical_prob"):
            probability = value[field]
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise ValueError(f"packing cache augmentation {field} must be numeric")
            if probability < 0.0 or probability > 1.0:
                raise ValueError(f"packing cache augmentation {field} must be in [0, 1]")
        if not isinstance(value["transform_counts"], Mapping):
            raise ValueError("packing cache augmentation transform_counts must be a mapping")
        if not isinstance(value["random_object_order_presentations"], list):
            raise ValueError(
                "packing cache augmentation random-order receipt must be a list"
            )
    return dict(value)


def _safe_torch_load_from_bytes(payload: bytes) -> Any:
    return torch.load(io.BytesIO(payload), weights_only=True)


class _RestrictedCacheUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return _safe_torch_load_from_bytes
        if (module, name) not in _ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(
                f"packing cache pickle global is forbidden: {module}.{name}"
            )
        imported = importlib.import_module(module)
        return getattr(imported, name)

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("packing cache pickle persistent IDs are forbidden")


def _cleanup_stale_cache_siblings(root: Path) -> None:
    for prefix in (f".{root.name}.stage-", f".{root.name}.backup-"):
        for residue in root.parent.glob(f"{prefix}*"):
            if residue.is_dir() and not residue.is_symlink():
                shutil.rmtree(residue)
            else:
                residue.unlink()


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


def _determinant_fingerprint(determinants: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(determinants).encode("utf-8")).hexdigest()


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
    "PackingCacheInvalidError",
    "PACKING_CACHE_MANIFEST",
    "PACKING_CACHE_MATERIALIZATION_STRATEGY",
    "PACKING_CACHE_VERSION",
    "PACK_CACHE_VERIFICATION_LEVELS",
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

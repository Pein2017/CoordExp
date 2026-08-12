"""Internal packed-micro-step cache for expensive dataset/template packing."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import ctypes
from dataclasses import MISSING, dataclass, fields
import errno
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
import stat as stat_module
from typing import Any
import uuid

import torch

from src.augmentation.geometry import GEOMETRY_FLIP_POLICY_VERSION
from src.config.models import DatasetSplitConfig, TrainConfig
from src.data.jsonl import iter_raw_examples
from src.packing.planner import (
    PACK_PLAN_SCHEMA,
    PACK_PLAN_SCHEMA_VERSION,
    build_pack_plan_policy_identity,
)
from src.training.schedule import ResolvedStepSchedule
from src.training.supervised_trainer import SupervisedMicroStep


PACKING_CACHE_VERSION = "coordexp-swift-pack-cache-v3"
PACKING_CACHE_MANIFEST = "manifest.json"
PACKING_CACHE_CHUNK_DIR = "chunks"
PACKING_CACHE_DETERMINANT_REGISTRY_VERSION = 1
DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS = 16
PACKING_CACHE_MATERIALIZATION_STRATEGY = "fork_process_pool"
PACK_CACHE_VERIFICATION_LEVELS = ("manifest", "payloads")
_MODEL_WEIGHT_PAYLOAD_PATTERNS = (
    r"model\.safetensors",
    r"adapter_model\.safetensors",
    r"pytorch_model\.bin",
    r"(?:model|adapter_model|pytorch_model)\.(?:pt|pth|ckpt)",
    r"consolidated(?:\.\d+)?\.pth",
)
_MODEL_WEIGHT_INDEX_SUFFIXES = (".safetensors.index.json", ".bin.index.json")
_MODEL_WEIGHT_PAYLOAD_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
_UNCLASSIFIED_LARGE_MODEL_ASSET_BYTES = 1024**3
_MAX_FRONTEND_ASSET_FILES = 4096
_MAX_FRONTEND_HASHED_BYTES = 4 * 1024**3
_MAX_MODEL_ROOT_REGULAR_FILES = 8192
_MAX_MODEL_WEIGHT_INDEX_DECLARATIONS = 65536
_MAX_PACK_CACHE_CHUNK_SNAPSHOT_BYTES = 4 * 1024**3
_PACK_CACHE_CHUNK_READ_BLOCK_BYTES = 1024 * 1024

PACKING_CACHE_DETERMINANT_OWNERS = {
    "dataset_content": "src/data/examples.py",
    "template_config": "src/templates/renderer.py",
    "packing_config": "src/packing/planner.py",
    "processor_config": "src/qwen/runtime_loading.py",
    "ordering_config": "src/data/examples.py",
    "augmentation_config": "src/augmentation/geometry.py",
    "model_config_assets": "src/qwen/runtime_loading.py",
    "processor_assets": "src/qwen/runtime_loading.py",
    "tokenizer_assets": "src/qwen/runtime_loading.py",
    "token_identity": "src/qwen/tokens.py",
    "realized_vocab_groups": "src/losses/vocab.py",
    "encoding_runtime": "src/qwen/encoding.py",
    "augmentation_factory": "src/augmentation/factory.py",
    "augmentation_processor": "src/augmentation/processor.py",
    "coordinate_targets": "src/coordinate_targets.py",
    "dataset_geometry": "src/data/geometry.py",
    "dataset_image_resolver": "src/data/images.py",
    "dataset_jsonl_loader": "src/data/jsonl.py",
    "template_spans": "src/templates/spans.py",
    "renderer": "src/templates/renderer.py",
    "parser": "src/data/examples.py",
    "image_loader": "src/qwen/images.py",
    "pack_planner": "src/packing/planner.py",
    "supervision_mapper": "src/packing/supervision.py",
    "supervision_tokens": "src/supervision/tokens.py",
    "mrope_position_ids": "src/qwen/positions.py",
    "qwen_fa2_boundaries": "src/qwen/fa2.py",
    "qwen_forward_payload": "src/qwen/forward.py",
    "micro_step_runtime_config": "src/training/pipeline.py",
    "micro_step_schema": "src/training/supervised_trainer.py",
    "cache_serializer": "src/training/pack_cache.py",
}

_DETERMINANT_REASONS = {
    "dataset_content": "raw JSONL and referenced image bytes determine encoded examples",
    "template_config": "rendered prompt and target text determine cached tokens",
    "packing_config": "pack length policy determines micro-step membership",
    "processor_config": "processor geometry configuration determines visual inputs",
    "ordering_config": "resolved presentation order determines cached pack order",
    "augmentation_config": "augmentation policy and seed determine presentations",
    "model_config_assets": "model config bytes determine preparation-time identities",
    "processor_assets": "processor asset bytes determine visual preprocessing",
    "tokenizer_assets": "tokenizer asset bytes determine realized tokenization",
    "token_identity": "resolved special-token IDs determine supervision encoding",
    "realized_vocab_groups": "every realized vocabulary-group member affects supervision",
    "encoding_runtime": "resolved tokenizer runtime values affect encoded examples",
    "augmentation_factory": "augmentation construction can change cached examples",
    "augmentation_processor": "augmentation application can change cached examples",
    "coordinate_targets": "coordinate target construction affects cached supervision",
    "dataset_geometry": "dataset bbox parsing and validation affect cached examples",
    "dataset_image_resolver": "image path resolution selects bytes consumed by preparation",
    "dataset_jsonl_loader": "JSONL iteration selects validated rows consumed by preparation",
    "template_spans": "template span construction affects cached supervision",
    "renderer": "renderer source produces cached prompt and target text",
    "parser": "parser source produces raw examples consumed by preparation",
    "image_loader": "image loader source produces cached visual inputs",
    "pack_planner": "planner source produces cached pack membership and order",
    "supervision_mapper": "supervision mapper source produces cached labels",
    "supervision_tokens": "token supervision source produces cached token spans",
    "mrope_position_ids": "MRoPE owner produces cached position IDs",
    "qwen_fa2_boundaries": "packed-boundary owner affects realized packed state",
    "qwen_forward_payload": "forward payload owner defines cached input consumption",
    "micro_step_runtime_config": (
        "production constructor config determines fields serialized into each micro-step"
    ),
    "micro_step_schema": "SupervisedMicroStep schema defines the serialized payload shape",
    "cache_serializer": "serializer source defines committed payload representation",
}


class PackingCacheInvalidError(ValueError):
    """The cache root cannot be consumed under the current strict contract."""


@dataclass(frozen=True)
class EvalCacheEntry:
    """One manifest-indexed eval payload in canonical publication order."""

    canonical_ordinal: int
    micro_step: SupervisedMicroStep


@dataclass(frozen=True)
class EvalCacheShard:
    """The exact modulo-assigned eval payload retained by one rank."""

    rank: int
    world_size: int
    total_ordinal_count: int
    entries: tuple[EvalCacheEntry, ...]

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("rank must be inside world_size")
        if self.total_ordinal_count <= 0:
            raise ValueError("eval cache must contain at least one canonical ordinal")
        expected = tuple(range(self.rank, self.total_ordinal_count, self.world_size))
        observed = tuple(entry.canonical_ordinal for entry in self.entries)
        if observed != expected:
            raise ValueError(
                "eval cache shard must contain the exact canonical modulo assignment"
            )

    @property
    def canonical_ordinals(self) -> tuple[int, ...]:
        return tuple(entry.canonical_ordinal for entry in self.entries)

    @property
    def micro_steps(self) -> tuple[SupervisedMicroStep, ...]:
        return tuple(entry.micro_step for entry in self.entries)


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
        DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS if workers is None else workers
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
    vocab_groups: Any,
) -> str:
    determinants = build_packing_cache_determinants(
        config,
        components,
        dataset=dataset,
        split=split,
        vocab_groups=vocab_groups,
    )
    return packing_cache_fingerprint_from_determinants(determinants)


def packing_cache_fingerprint_from_determinants(
    determinants: Mapping[str, Any],
) -> str:
    """Return the validated fingerprint without re-resolving expensive inputs."""

    return _determinant_fingerprint(determinants)


def build_packing_cache_determinants(
    config: TrainConfig,
    components: Any,
    *,
    dataset: DatasetSplitConfig,
    split: str,
    vocab_groups: Any,
) -> dict[str, Any]:
    dataset_path = Path(dataset.path).expanduser().resolve()
    stat = dataset_path.stat()
    processor_identity = getattr(components, "processor_identity")
    token_identity = getattr(components, "token_identity")
    base_model_path = (
        Path(getattr(components, "base_model_path")).expanduser().resolve()
    )
    tokenizer = getattr(components, "tokenizer", None)
    frontend_assets = _frontend_asset_inventory(base_model_path)
    semantic_payload = {
        "version": PACKING_CACHE_VERSION,
        "split": split,
        "dataset": {
            "path": str(dataset_path),
            "sample_limit": dataset.sample_limit,
            "size_bytes": int(stat.st_size),
            "sha256": _file_sha256(dataset_path),
            "image_content_identity": _dataset_image_identity(dataset),
        },
        "template": config.template.model_dump(mode="json"),
        "packing": {
            "schema": PACK_PLAN_SCHEMA,
            "schema_version": PACK_PLAN_SCHEMA_VERSION,
            "global_max_length": config.packing.global_max_length,
            "policy_identity": build_pack_plan_policy_identity(
                policy=config.packing.policy,
                window_size=config.packing.window_size,
                lookahead=config.packing.lookahead,
                seed=config.packing.seed,
                worker_count=config.packing.worker_count,
                cursor_byte_budget=config.packing.cursor_byte_budget,
                fragment_item_budget=config.packing.fragment_item_budget,
                fragment_byte_budget=config.packing.fragment_byte_budget,
            ),
            "fragment_pack_budget": config.packing.max_packs_per_fragment,
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
            "encoding_identity": _qwen_encoding_identity(
                components, tokenizer=tokenizer
            ),
            # These semantic categories retain their separate ownership while
            # sharing one fail-closed inventory of every non-weight frontend
            # file under the resolved local model root.
            "model_config_assets": frontend_assets,
            "processor_assets": frontend_assets,
            "tokenizer_assets": frontend_assets,
        },
        "realized_vocab_groups": _realized_vocab_group_identity(vocab_groups),
        "micro_step_runtime_config": {
            "fa2_model_dtype": config.training.precision,
            "capture_fa2_branch": config.model.fa2_branch_proof == "every_forward",
            "require_fa2_branch_proof": (
                config.model.fa2_branch_proof == "every_forward"
            ),
        },
        "micro_step_schema": _supervised_micro_step_schema_identity(),
    }
    entries = _build_determinant_entries(semantic_payload)
    aggregate_fingerprint = _registry_entries_fingerprint(entries)
    determinants = {
        **semantic_payload,
        "registry_schema_version": PACKING_CACHE_DETERMINANT_REGISTRY_VERSION,
        "determinants": entries,
        "aggregate_fingerprint": aggregate_fingerprint,
        "code_identity": _registry_code_identity(entries),
    }
    _validate_determinant_registry(determinants)
    return determinants


def cache_dir_for_fingerprint(cache_root: str | Path, fingerprint: str) -> Path:
    target = _canonical_cache_target(cache_root, fingerprint=fingerprint)
    _reject_symlink_components(target)
    return target


def manifest_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / PACKING_CACHE_MANIFEST


def load_cache_manifest(
    cache_dir: str | Path,
    *,
    cache_root: str | Path,
    expected_fingerprint: str,
    level: str,
) -> dict[str, Any]:
    if level not in PACK_CACHE_VERIFICATION_LEVELS:
        raise ValueError(
            "packing cache verification level must be one of "
            f"{PACK_CACHE_VERIFICATION_LEVELS}"
        )
    try:
        root = _validate_canonical_cache_dir(
            cache_dir,
            cache_root=cache_root,
            fingerprint=expected_fingerprint,
        )
        manifest = _load_validated_manifest(
            root, expected_fingerprint=expected_fingerprint
        )
        if level == "payloads":
            for _start, _chunk_steps in _iter_validated_chunks(root, manifest):
                pass
        return manifest
    except PackingCacheInvalidError:
        raise
    except _INVALID_CACHE_ERRORS as exc:
        raise PackingCacheInvalidError(f"invalid packing cache: {exc}") from exc


def _load_cache_manifest_at_private_stage(
    cache_dir: Path,
    *,
    expected_fingerprint: str,
    level: str,
) -> dict[str, Any]:
    """Validate a writer-owned stage that is intentionally not publicly canonical."""

    if level not in PACK_CACHE_VERIFICATION_LEVELS:
        raise ValueError(
            "packing cache verification level must be one of "
            f"{PACK_CACHE_VERIFICATION_LEVELS}"
        )
    manifest = _load_validated_manifest(
        cache_dir,
        expected_fingerprint=expected_fingerprint,
    )
    if level == "payloads":
        for _start, _chunk_steps in _iter_validated_chunks(cache_dir, manifest):
            pass
    return manifest


def cache_is_complete(
    cache_dir: str | Path,
    *,
    cache_root: str | Path,
    fingerprint: str,
) -> bool:
    try:
        manifest = load_cache_manifest(
            cache_dir,
            cache_root=cache_root,
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
    cache_root: str | Path,
    fingerprint: str,
    determinants: Mapping[str, Any],
    chunk_size: int = 512,
    materialization: Mapping[str, Any],
    determinant_revalidator: Callable[[], Mapping[str, Any]],
    augmentation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    materialization_payload = _coerce_materialization(materialization)
    if not micro_steps:
        raise ValueError("packing cache must contain at least one micro-step")
    _validate_fingerprint(fingerprint)
    if not callable(determinant_revalidator):
        raise ValueError("determinant_revalidator must be callable")
    if not isinstance(determinants, Mapping):
        raise ValueError("determinants must be a mapping")
    _validate_determinant_registry(determinants)
    if fingerprint != _determinant_fingerprint(determinants):
        raise ValueError("fingerprint must match canonical determinants")
    augmentation_payload = _validate_augmentation_receipt(augmentation)
    root = _validate_canonical_cache_dir(
        cache_dir,
        cache_root=cache_root,
        fingerprint=fingerprint,
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    root = _validate_canonical_cache_dir(
        cache_dir,
        cache_root=cache_root,
        fingerprint=fingerprint,
    )
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
                        cache_root=cache_root,
                        expected_fingerprint=fingerprint,
                        level="payloads",
                    )
                    if existing_manifest["determinants"] == dict(determinants):
                        return existing_manifest
                except PackingCacheInvalidError as exc:
                    raise _immutable_collision_error(
                        root,
                        fingerprint=fingerprint,
                        validation_category="current_publication_invalid",
                        detail_type=type(exc).__name__,
                    ) from exc
                raise _immutable_collision_error(
                    root,
                    fingerprint=fingerprint,
                    validation_category="determinant_mismatch",
                    detail_type="DeterminantMismatch",
                )
            return _publish_micro_step_cache(
                root,
                micro_steps,
                fingerprint=fingerprint,
                determinants=determinants,
                chunk_size=chunk_size,
                materialization=materialization_payload,
                augmentation=augmentation_payload,
                determinant_revalidator=determinant_revalidator,
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
    determinant_revalidator: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    stage = root.with_name(f".{root.name}.stage-{token}")
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
        if _is_eval_cache_manifest(manifest):
            manifest["eval_ordinal_index"] = [
                {
                    "ordinal": ordinal,
                    "chunk_index": ordinal // chunk_size,
                    "chunk_offset": ordinal % chunk_size,
                }
                for ordinal in range(len(micro_steps))
            ]
        # The complete manifest is the final write inside the isolated stage.
        _atomic_write_json(manifest_path(stage), manifest)
        _load_cache_manifest_at_private_stage(
            stage,
            expected_fingerprint=fingerprint,
            level="payloads",
        )
        current_determinants = determinant_revalidator()
        if not isinstance(current_determinants, Mapping):
            raise ValueError("determinant revalidator must return a mapping")
        _validate_determinant_registry(current_determinants)
        current_fingerprint = packing_cache_fingerprint_from_determinants(
            current_determinants
        )
        if current_fingerprint != fingerprint or dict(current_determinants) != dict(
            determinants
        ):
            raise PackingCacheInvalidError(
                "packing cache determinants drifted during publication: "
                f"target={root} current_version={PACKING_CACHE_VERSION} "
                f"fingerprint={fingerprint} "
                "validation_category=determinant_drift "
                "automatic_recovery=unavailable"
            )
        try:
            _install_staged_cache_no_replace(stage, root)
        except FileExistsError as exc:
            raise _immutable_collision_error(
                root,
                fingerprint=fingerprint,
                validation_category="target_already_exists",
                detail_type=type(exc).__name__,
            ) from exc
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_rank_micro_steps_from_cache(
    cache_dir: str | Path,
    *,
    cache_root: str | Path,
    expected_fingerprint: str,
    schedule: ResolvedStepSchedule,
    rank: int,
    world_size: int,
    _force_full_chunk_pass: bool = False,
) -> tuple[SupervisedMicroStep, ...]:
    """Load one rank's required micro-steps, skipping unneeded chunks.

    `_force_full_chunk_pass` is an internal, test/benchmark-only control for
    the M2 paired A/B (rank-selective vs. full-pass loading on the same
    code/config/cache); it is not a public YAML/CLI compatibility surface.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be inside world_size")
    if schedule.runtime_batch.world_size != world_size:
        raise ValueError("world_size must match schedule runtime_batch")
    try:
        root = _validate_canonical_cache_dir(
            cache_dir,
            cache_root=cache_root,
            fingerprint=expected_fingerprint,
        )
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
        required = frozenset(indices)
        selected: dict[int, SupervisedMicroStep] = {}
        for start, chunk_steps in _iter_required_chunks(
            root,
            manifest,
            required=required,
            force_full_pass=_force_full_chunk_pass,
        ):
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
    cache_dir: str | Path,
    *,
    cache_root: str | Path,
    expected_fingerprint: str,
) -> tuple[SupervisedMicroStep, ...]:
    try:
        root = _validate_canonical_cache_dir(
            cache_dir,
            cache_root=cache_root,
            fingerprint=expected_fingerprint,
        )
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


def load_rank_eval_micro_steps_from_cache(
    cache_dir: str | Path,
    *,
    cache_root: str | Path,
    expected_fingerprint: str,
    rank: int,
    world_size: int,
) -> EvalCacheShard:
    """Hydrate exactly the manifest-indexed eval ordinals assigned to one rank.

    All manifest declarations are validated before any payload is decoded.
    Payload authentication and decoding are limited to chunks referenced by
    this rank's canonical `ordinal % world_size` slice. A coarse chunk can
    therefore still be decoded by multiple ranks; chunk skipping is not part
    of this semantic contract.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be inside world_size")
    try:
        root = _validate_canonical_cache_dir(
            cache_dir,
            cache_root=cache_root,
            fingerprint=expected_fingerprint,
        )
        manifest = _load_validated_manifest(
            root, expected_fingerprint=expected_fingerprint
        )
        index = _validated_eval_ordinal_index(manifest)
        required_entries = tuple(
            entry for entry in index if int(entry["ordinal"]) % world_size == rank
        )
        loaded_chunks: dict[int, tuple[SupervisedMicroStep, ...]] = {}
        entries: list[EvalCacheEntry] = []
        for entry in required_entries:
            chunk_index = int(entry["chunk_index"])
            chunk_steps = loaded_chunks.get(chunk_index)
            if chunk_steps is None:
                _start, chunk_steps = _load_validated_chunk(
                    root, manifest["chunks"][chunk_index]
                )
                loaded_chunks[chunk_index] = chunk_steps
            entries.append(
                EvalCacheEntry(
                    canonical_ordinal=int(entry["ordinal"]),
                    micro_step=chunk_steps[int(entry["chunk_offset"])],
                )
            )
        return EvalCacheShard(
            rank=rank,
            world_size=world_size,
            total_ordinal_count=int(manifest["micro_step_count"]),
            entries=tuple(entries),
        )
    except PackingCacheInvalidError:
        raise
    except _INVALID_CACHE_ERRORS as exc:
        raise PackingCacheInvalidError(f"invalid packing cache: {exc}") from exc


def _load_all_eval_micro_steps_from_cache_for_test(
    cache_dir: str | Path,
    *,
    cache_root: str | Path,
    expected_fingerprint: str,
) -> tuple[EvalCacheEntry, ...]:
    """Full-hydration compatibility oracle; tests only, never a fallback."""

    try:
        root = _validate_canonical_cache_dir(
            cache_dir,
            cache_root=cache_root,
            fingerprint=expected_fingerprint,
        )
        manifest = _load_validated_manifest(
            root, expected_fingerprint=expected_fingerprint
        )
        index = _validated_eval_ordinal_index(manifest)
        chunks = {
            chunk_index: chunk_steps
            for chunk_index, (_start, chunk_steps) in enumerate(
                _iter_validated_chunks(root, manifest)
            )
        }
        return tuple(
            EvalCacheEntry(
                canonical_ordinal=int(entry["ordinal"]),
                micro_step=chunks[int(entry["chunk_index"])][
                    int(entry["chunk_offset"])
                ],
            )
            for entry in index
        )
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
        schedule.resolved_max_steps * schedule.runtime_batch.resolved_grad_accum_steps
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
        raise ValueError(
            "unsupported packing cache version: "
            f"observed={manifest.get('version')!r}, current={PACKING_CACHE_VERSION!r}"
        )
    if manifest.get("status") != "complete":
        raise ValueError("packing cache manifest is not complete")
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "packing cache fingerprint does not match expected fingerprint"
        )
    if not isinstance(manifest.get("determinants"), Mapping):
        raise ValueError("packing cache determinants must be a mapping")
    _validate_determinant_registry(manifest["determinants"])
    if manifest["fingerprint"] != _determinant_fingerprint(manifest["determinants"]):
        raise ValueError(
            "packing cache fingerprint does not match canonical determinants"
        )
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
        if (
            count_from_range <= 0
            or _positive_int(chunk.get("count"), field="chunk count")
            != count_from_range
        ):
            raise ValueError("packing cache chunk count must match range")
        expected_start = end
        path = _safe_chunk_path(cache_dir, chunk.get("path"))
        try:
            _reject_symlink_components(path)
            chunk_stat = path.lstat()
        except FileNotFoundError:
            raise ValueError(f"packing cache chunk is missing: {path}")
        if not stat_module.S_ISREG(chunk_stat.st_mode):
            raise ValueError(f"packing cache chunk must be a regular file: {path}")
        sha256 = chunk.get("sha256")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError("packing cache chunk must record a valid sha256")
    if expected_start != count:
        raise ValueError("packing cache chunks must cover every micro-step")
    if _is_eval_cache_manifest(manifest):
        _validated_eval_ordinal_index(manifest)
    elif "eval_ordinal_index" in manifest:
        raise ValueError(
            "training cache manifest must not declare an eval ordinal index"
        )


def _is_eval_cache_manifest(manifest: Mapping[str, Any]) -> bool:
    determinants = manifest.get("determinants")
    return isinstance(determinants, Mapping) and str(
        determinants.get("split", "")
    ).startswith("eval")


def _validated_eval_ordinal_index(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Validate the complete canonical ordinal-to-chunk declaration graph."""

    count = _positive_int(manifest.get("micro_step_count"), field="micro_step_count")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("eval ordinal index requires declared chunks")
    raw_index = manifest.get("eval_ordinal_index")
    if not isinstance(raw_index, list):
        raise ValueError("eval cache manifest must declare an eval ordinal index")
    if len(raw_index) != count:
        raise ValueError("eval ordinal index must cover every canonical ordinal")
    validated: list[Mapping[str, Any]] = []
    for expected_ordinal, entry in enumerate(raw_index):
        if not isinstance(entry, Mapping):
            raise ValueError("eval ordinal index entries must be mappings")
        ordinal = _nonnegative_int(entry.get("ordinal"), field="eval ordinal")
        chunk_index = _nonnegative_int(
            entry.get("chunk_index"), field="eval ordinal chunk_index"
        )
        chunk_offset = _nonnegative_int(
            entry.get("chunk_offset"), field="eval ordinal chunk_offset"
        )
        if ordinal != expected_ordinal:
            raise ValueError("eval ordinal index must be canonical and contiguous")
        if chunk_index >= len(chunks):
            raise ValueError("eval ordinal index references an unknown chunk")
        chunk = chunks[chunk_index]
        if not isinstance(chunk, Mapping):
            raise ValueError("packing cache chunk declarations must be mappings")
        if chunk_offset >= int(chunk["count"]):
            raise ValueError("eval ordinal index chunk offset is outside its chunk")
        if int(chunk["start"]) + chunk_offset != ordinal:
            raise ValueError("eval ordinal index does not match the chunk declaration")
        validated.append(entry)
    return tuple(validated)


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
        yield _load_validated_chunk(cache_dir, chunk)


def _iter_required_chunks(
    cache_dir: Path,
    manifest: Mapping[str, Any],
    *,
    required: frozenset[int],
    force_full_pass: bool,
) -> Iterator[tuple[int, tuple[SupervisedMicroStep, ...]]]:
    """Yield only chunks whose declared `[start, end)` intersects `required`.

    Every chunk declaration was already structurally validated (contiguity,
    counts, digest syntax, path safety, file existence) by
    `_load_validated_manifest` before this runs, so skipping a chunk here
    never skips that manifest-level check. A skipped chunk's payload bytes
    are never read or digest-verified; every chunk that IS yielded still
    goes through the exact same digest-and-payload validation as a full
    pass (`_load_validated_chunk`).
    """

    for chunk in manifest["chunks"]:
        start = int(chunk["start"])
        end = int(chunk["end"])
        if not force_full_pass and not any(start <= index < end for index in required):
            continue
        yield _load_validated_chunk(cache_dir, chunk)


def _load_validated_chunk(
    cache_dir: Path, chunk: Mapping[str, Any]
) -> tuple[int, tuple[SupervisedMicroStep, ...]]:
    chunk_path = _safe_chunk_path(cache_dir, chunk["path"])
    try:
        snapshot, actual_sha256 = _read_chunk_snapshot(cache_dir, chunk_path)
    except _INVALID_CACHE_ERRORS as exc:
        raise ValueError(
            f"packing cache chunk payload is unreadable: {chunk_path}"
        ) from exc
    if actual_sha256 != chunk["sha256"]:
        snapshot.close()
        raise ValueError(
            f"packing cache chunk checksum mismatch: {chunk_path} "
            f"(expected {chunk['sha256']}, got {actual_sha256})"
        )
    try:
        with snapshot:
            chunk_steps = _RestrictedCacheUnpickler(snapshot).load()
    except _INVALID_CACHE_ERRORS as exc:
        raise ValueError(
            f"packing cache chunk payload is unreadable: {chunk_path}"
        ) from exc
    if not isinstance(chunk_steps, tuple):
        raise ValueError("packing cache chunk payload must be a tuple")
    if len(chunk_steps) != chunk["count"]:
        raise ValueError("packing cache chunk payload length must match declared count")
    if not all(isinstance(step, SupervisedMicroStep) for step in chunk_steps):
        raise ValueError(
            "packing cache chunk payload must contain supervised micro-steps"
        )
    return chunk["start"], chunk_steps


def _safe_chunk_path(cache_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("packing cache chunk path must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("packing cache chunk path must be relative")
    if ".." in relative.parts:
        raise ValueError("packing cache chunk path must stay inside cache root")
    root = Path(os.path.abspath(cache_dir))
    path = root / relative
    if path == root or root not in path.parents:
        raise ValueError("packing cache chunk path must stay inside cache root")
    return path


def _read_chunk_snapshot(cache_dir: Path, chunk_path: Path) -> tuple[io.BytesIO, str]:
    """Read one bounded, stable regular-file snapshot without following symlinks."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise OSError(
            errno.ENOTSUP,
            "no-follow packing cache chunk reads are unavailable on this platform",
        )
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    nonblocking = getattr(os, "O_NONBLOCK", 0)
    directory_flags = os.O_RDONLY | close_on_exec | directory | nofollow
    file_flags = os.O_RDONLY | close_on_exec | nonblocking | nofollow
    relative = chunk_path.relative_to(Path(os.path.abspath(cache_dir)))
    absolute_cache_dir = Path(os.path.abspath(cache_dir))
    directory_fd = os.open(absolute_cache_dir.anchor, directory_flags)
    try:
        for component in (*absolute_cache_dir.parts[1:], *relative.parts[:-1]):
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        chunk_fd = os.open(relative.parts[-1], file_flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)

    try:
        before = os.fstat(chunk_fd)
        if not stat_module.S_ISREG(before.st_mode):
            raise ValueError("packing cache chunk descriptor must be a regular file")
        size_bytes = before.st_size
        if size_bytes < 0 or size_bytes > _MAX_PACK_CACHE_CHUNK_SNAPSHOT_BYTES:
            raise ValueError("packing cache chunk snapshot exceeds the byte bound")

        digest = hashlib.sha256()
        snapshot = io.BytesIO()
        remaining = size_bytes
        while remaining:
            block = os.read(
                chunk_fd,
                min(remaining, _PACK_CACHE_CHUNK_READ_BLOCK_BYTES),
            )
            if not block:
                raise ValueError("packing cache chunk became shorter while reading")
            snapshot.write(block)
            digest.update(block)
            remaining -= len(block)
        if os.read(chunk_fd, 1):
            raise ValueError("packing cache chunk grew while reading")

        after = os.fstat(chunk_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field) for field in stable_fields
        ):
            raise ValueError("packing cache chunk changed while reading")
        snapshot.seek(0)
        return snapshot, digest.hexdigest()
    finally:
        os.close(chunk_fd)


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
        raise ValueError(
            "packing cache materialization must record a non-empty strategy"
        )
    workers = _positive_int(
        materialization.get("workers"), field="materialization workers"
    )
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
            raise ValueError(
                "packing cache augmentation policy_version must be non-empty"
            )
        for field in ("horizontal_prob", "vertical_prob"):
            probability = value[field]
            if isinstance(probability, bool) or not isinstance(
                probability, (int, float)
            ):
                raise ValueError(f"packing cache augmentation {field} must be numeric")
            if probability < 0.0 or probability > 1.0:
                raise ValueError(
                    f"packing cache augmentation {field} must be in [0, 1]"
                )
        if not isinstance(value["transform_counts"], Mapping):
            raise ValueError(
                "packing cache augmentation transform_counts must be a mapping"
            )
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
        raise pickle.UnpicklingError(
            "packing cache pickle persistent IDs are forbidden"
        )


def _install_staged_cache_no_replace(stage: Path, target: Path) -> None:
    """Atomically install one sibling directory without replacing a target."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS,
            "atomic no-replace cache publication is unavailable on this platform",
            target,
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(stage),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target)
    raise OSError(error_number, os.strerror(error_number), target)


def _qwen_encoding_identity(components: Any, *, tokenizer: Any) -> dict[str, Any]:
    image_pad_token_id = None
    if tokenizer is not None:
        convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            image_pad_token_id = convert_tokens_to_ids("<|image_pad|>")
    tokenizer_length = None
    if tokenizer is not None:
        try:
            tokenizer_length = len(tokenizer)
        except (TypeError, AttributeError):
            tokenizer_length = None
    return {
        "processor_class": type(getattr(components, "processor", None)).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_length": tokenizer_length,
        "image_pad_token_id": image_pad_token_id,
        "all_special_ids": list(getattr(tokenizer, "all_special_ids", ()) or ()),
        "control_token_ids": _json_identity_value(
            getattr(tokenizer, "control_token_ids", {}) or {}
        ),
        "special_tokens_map": _json_identity_value(
            getattr(tokenizer, "special_tokens_map", {}) or {}
        ),
        "added_vocab": _tokenizer_added_vocab(tokenizer),
        "chat_template_sha256": _stable_text_sha256(
            getattr(tokenizer, "chat_template", None)
            or getattr(getattr(components, "processor", None), "chat_template", None)
        ),
        "package_versions": dict(getattr(components, "package_versions", {}) or {}),
    }


def _tokenizer_added_vocab(tokenizer: Any) -> dict[str, Any]:
    getter = getattr(tokenizer, "get_added_vocab", None)
    if not callable(getter):
        return {}
    return _json_identity_value(getter())


def _json_identity_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_json_identity_value(item) for item in value), key=str)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_identity_value(item) for item in value]
    return str(value)


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


def _dataset_image_identity(dataset: DatasetSplitConfig) -> dict[str, Any]:
    digest = hashlib.sha256()
    image_count = 0
    total_size_bytes = 0
    for example in iter_raw_examples(dataset):
        image_path = example.image.path
        size_bytes = image_path.stat().st_size
        record = {
            "example_id": example.example_id,
            "source_row_number": example.source.row_number,
            "declared_path": example.image.declared_path,
            "path": str(image_path),
            "size_bytes": size_bytes,
            "sha256": _file_sha256(image_path),
        }
        digest.update(_canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
        image_count += 1
        total_size_bytes += size_bytes
    return {
        "algorithm": "ordered-canonical-image-content-sha256-v1",
        "image_count": image_count,
        "total_size_bytes": total_size_bytes,
        "sha256": digest.hexdigest(),
    }


def _frontend_asset_inventory(base_model_path: Path) -> dict[str, Any]:
    """SHA-bind every local frontend file except recognized model weights.

    Arbitrary files with weight-like suffixes are still included. Only the
    explicit model-weight basename patterns below are excluded; index JSON,
    custom binary assets, README files, and nested frontend assets remain
    determinants.
    """

    try:
        root = base_model_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"packing cache local model root is unavailable: {base_model_path}"
        ) from exc
    if not root.is_dir():
        raise ValueError(f"packing cache local model root must be a directory: {root}")

    discovered: list[tuple[Path, str, os.stat_result]] = []
    excluded_weight_payload_count = 0
    for current_text, dir_names, file_names in os.walk(
        root,
        followlinks=False,
        onerror=_raise_frontend_asset_walk_error,
    ):
        current = Path(current_text)
        dir_names.sort()
        file_names.sort()
        for dir_name in tuple(dir_names):
            directory = current / dir_name
            if directory.is_symlink():
                resolved = _resolve_model_asset_symlink(directory, root=root)
                raise ValueError(
                    "packing cache model-root inventory does not support symlinked "
                    f"directories: {directory} -> {resolved}"
                )
            if not directory.is_dir():
                raise ValueError(
                    f"packing cache model-root entry is not a directory: {directory}"
                )
        for file_name in file_names:
            logical_path = current / file_name
            relative_path = logical_path.relative_to(root).as_posix()
            file_stat = logical_path.lstat()
            if stat_module.S_ISLNK(file_stat.st_mode):
                resolved = _resolve_model_asset_symlink(logical_path, root=root)
                raise ValueError(
                    "packing cache model-root inventory does not support symlinked "
                    f"files: {logical_path} -> {resolved}"
                )
            if not stat_module.S_ISREG(file_stat.st_mode):
                raise ValueError(
                    "packing cache model-root inventory supports only regular files: "
                    f"{logical_path}"
                )
            if len(discovered) >= _MAX_MODEL_ROOT_REGULAR_FILES:
                raise ValueError(
                    "packing cache model-root regular file count exceeds declared "
                    f"limit {_MAX_MODEL_ROOT_REGULAR_FILES}"
                )
            discovered.append((logical_path, relative_path, file_stat))

    declared_weight_payloads = _declared_model_weight_payloads(discovered)
    assets: list[dict[str, Any]] = []
    total_hashed_bytes = 0
    for logical_path, relative_path, file_stat in discovered:
        if _is_recognized_model_weight_payload(
            relative_path,
            declared_weight_payloads=declared_weight_payloads,
        ):
            excluded_weight_payload_count += 1
            continue
        if file_stat.st_size >= _UNCLASSIFIED_LARGE_MODEL_ASSET_BYTES:
            raise ValueError(
                "packing cache model-root inventory refuses an unclassified "
                f"large file: {logical_path} ({file_stat.st_size} bytes)"
            )
        if len(assets) >= _MAX_FRONTEND_ASSET_FILES:
            raise ValueError(
                "packing cache model-root frontend asset count exceeds declared "
                f"limit {_MAX_FRONTEND_ASSET_FILES}"
            )
        total_hashed_bytes += file_stat.st_size
        if total_hashed_bytes > _MAX_FRONTEND_HASHED_BYTES:
            raise ValueError(
                "packing cache model-root frontend bytes exceed declared "
                f"limit {_MAX_FRONTEND_HASHED_BYTES}"
            )
        assets.append(
            {
                "relative_path": relative_path,
                "size_bytes": file_stat.st_size,
                "sha256": _file_sha256(logical_path),
            }
        )
    return {
        "algorithm": "recursive-non-weight-regular-file-sha256-v1",
        "base_model_path": str(root),
        "weight_payload_exclusion": {
            "policy": "recognized-model-weight-basename-v1",
            "patterns": list(_MODEL_WEIGHT_PAYLOAD_PATTERNS),
            "index_suffixes": list(_MODEL_WEIGHT_INDEX_SUFFIXES),
            "excluded_file_count": excluded_weight_payload_count,
            "content_hashed": False,
        },
        "inventory_limits": {
            "max_discovered_regular_file_count": _MAX_MODEL_ROOT_REGULAR_FILES,
            "max_weight_index_declarations": _MAX_MODEL_WEIGHT_INDEX_DECLARATIONS,
            "max_regular_file_count": _MAX_FRONTEND_ASSET_FILES,
            "max_total_hashed_bytes": _MAX_FRONTEND_HASHED_BYTES,
            "max_unclassified_file_bytes": _UNCLASSIFIED_LARGE_MODEL_ASSET_BYTES,
        },
        "asset_count": len(assets),
        "total_hashed_bytes": total_hashed_bytes,
        "assets": assets,
    }


def _raise_frontend_asset_walk_error(error: OSError) -> None:
    error_path = str(getattr(error, "filename", None) or "<unknown>")[:512]
    raise ValueError(
        "packing cache model-root inventory could not read subtree: "
        f"path={error_path!r} error_type={type(error).__name__}"
    ) from error


def _resolve_model_asset_symlink(path: Path, *, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"packing cache model-root symlink is unresolved: {path}"
        ) from exc
    if resolved == root or root not in resolved.parents:
        raise ValueError(
            f"packing cache model-root symlink escapes local root: {path} -> {resolved}"
        )
    return resolved


def _declared_model_weight_payloads(
    discovered: Sequence[tuple[Path, str, os.stat_result]],
) -> frozenset[str]:
    declared: set[str] = set()
    declaration_count = 0
    discovered_names = {relative_path for _path, relative_path, _stat in discovered}
    for index_path, relative_path, _file_stat in discovered:
        if not relative_path.endswith(_MODEL_WEIGHT_INDEX_SUFFIXES):
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"packing cache model weight index is unreadable: {index_path}"
            ) from exc
        weight_map = payload.get("weight_map") if isinstance(payload, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError(
                f"packing cache model weight index must contain weight_map: {index_path}"
            )
        for declared_name in weight_map.values():
            declaration_count += 1
            if declaration_count > _MAX_MODEL_WEIGHT_INDEX_DECLARATIONS:
                raise ValueError(
                    "packing cache model weight index declaration count exceeds "
                    f"declared limit {_MAX_MODEL_WEIGHT_INDEX_DECLARATIONS}"
                )
            if not isinstance(declared_name, str) or not declared_name:
                raise ValueError(
                    f"packing cache model weight index has invalid payload name: {index_path}"
                )
            candidate = Path(relative_path).parent / declared_name
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(
                    f"packing cache model weight index payload escapes model root: {declared_name}"
                )
            normalized = candidate.as_posix()
            if not normalized.endswith(_MODEL_WEIGHT_PAYLOAD_SUFFIXES):
                raise ValueError(
                    f"packing cache model weight index declares unsupported payload: {normalized}"
                )
            if normalized not in discovered_names:
                raise ValueError(
                    f"packing cache model weight index declares missing payload: {normalized}"
                )
            declared.add(normalized)
    return frozenset(declared)


def _is_recognized_model_weight_payload(
    relative_path: str,
    *,
    declared_weight_payloads: frozenset[str],
) -> bool:
    if relative_path in declared_weight_payloads:
        return True
    if "/" in relative_path:
        return False
    return any(
        re.fullmatch(pattern, relative_path)
        for pattern in _MODEL_WEIGHT_PAYLOAD_PATTERNS
    )


def _realized_vocab_group_identity(vocab_groups: Any) -> dict[str, Any]:
    required_fields = (
        "vocab_size",
        "desc_text",
        "schema",
        "coordinate",
        "eos",
        "blocked",
    )
    identity: dict[str, Any] = {
        "algorithm": "canonical-sorted-int-membership-sha256-v1",
        "groups": {},
    }
    for field in required_fields:
        if isinstance(vocab_groups, Mapping):
            value = vocab_groups.get(field)
        else:
            value = getattr(vocab_groups, field, None)
        if value is None:
            raise ValueError(
                f"realized vocabulary groups must expose complete field {field!r}"
            )
        if field == "vocab_size":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("realized vocabulary size must be a positive integer")
            identity[field] = value
            continue
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"realized vocabulary group {field!r} must be a sequence")
        token_ids = list(value)
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in token_ids
        ):
            raise ValueError(
                f"realized vocabulary group {field!r} must contain integer token IDs"
            )
        canonical_members = sorted(token_ids)
        if len(set(canonical_members)) != len(canonical_members):
            raise ValueError(
                f"realized vocabulary group {field!r} must not contain duplicate token IDs"
            )
        if any(
            token_id < 0 or token_id >= identity["vocab_size"]
            for token_id in canonical_members
        ):
            raise ValueError(
                f"realized vocabulary group {field!r} contains an out-of-range token ID"
            )
        identity["groups"][field] = {
            "count": len(canonical_members),
            "sha256": hashlib.sha256(
                _canonical_json({"members": canonical_members}).encode("utf-8")
            ).hexdigest(),
            "min_id": canonical_members[0] if canonical_members else None,
            "max_id": canonical_members[-1] if canonical_members else None,
        }
    return identity


def _supervised_micro_step_schema_identity() -> dict[str, Any]:
    schema_fields: list[dict[str, Any]] = []
    for schema_field in fields(SupervisedMicroStep):
        has_default = schema_field.default is not MISSING
        schema_fields.append(
            {
                "name": schema_field.name,
                "annotation": str(schema_field.type),
                "has_default": has_default,
                "default": (
                    _json_identity_value(schema_field.default) if has_default else None
                ),
            }
        )
    return {
        "class": "SupervisedMicroStep",
        "frozen": bool(SupervisedMicroStep.__dataclass_params__.frozen),
        "fields": schema_fields,
    }


def _build_determinant_entries(
    semantic_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    content_identities = _semantic_determinant_content_identities(semantic_payload)
    repo_root = Path(__file__).resolve().parents[2]
    return [
        {
            "name": name,
            "owner": owner,
            "content_identity": content_identities[name],
            "owner_source_identity": {
                "path": owner,
                "sha256": _file_sha256(repo_root / owner),
            },
            "reason": _DETERMINANT_REASONS[name],
            "schema_version": 1,
        }
        for name, owner in sorted(PACKING_CACHE_DETERMINANT_OWNERS.items())
    ]


def _semantic_determinant_content_identities(
    determinants: Mapping[str, Any],
) -> dict[str, Any]:
    qwen = determinants.get("qwen")
    if not isinstance(qwen, Mapping):
        raise ValueError("packing cache Qwen determinants must be a mapping")
    identities = {
        "dataset_content": determinants.get("dataset"),
        "template_config": determinants.get("template"),
        "packing_config": determinants.get("packing"),
        "processor_config": {
            "config": determinants.get("processor"),
            "resolved_identity": qwen.get("processor_identity"),
        },
        "ordering_config": determinants.get("ordering"),
        "augmentation_config": determinants.get("augmentation"),
        "model_config_assets": qwen.get("model_config_assets"),
        "processor_assets": qwen.get("processor_assets"),
        "tokenizer_assets": qwen.get("tokenizer_assets"),
        "token_identity": qwen.get("token_identity"),
        "realized_vocab_groups": determinants.get("realized_vocab_groups"),
        "encoding_runtime": qwen.get("encoding_identity"),
        "micro_step_runtime_config": determinants.get("micro_step_runtime_config"),
        "micro_step_schema": determinants.get("micro_step_schema"),
    }
    for name in PACKING_CACHE_DETERMINANT_OWNERS:
        identities.setdefault(name, {"semantic_contract": name})
    return identities


def _registry_entries_fingerprint(entries: Sequence[Mapping[str, Any]]) -> str:
    ordered_entries = sorted(entries, key=lambda entry: str(entry.get("name")))
    payload = {
        "cache_version": PACKING_CACHE_VERSION,
        "registry_schema_version": PACKING_CACHE_DETERMINANT_REGISTRY_VERSION,
        "determinants": ordered_entries,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_determinant_registry(determinants: Mapping[str, Any]) -> None:
    observed_version = determinants.get("registry_schema_version")
    if observed_version != PACKING_CACHE_DETERMINANT_REGISTRY_VERSION:
        raise ValueError(
            "unsupported packing cache determinant registry version: "
            f"observed={observed_version!r}, "
            f"current={PACKING_CACHE_DETERMINANT_REGISTRY_VERSION!r}"
        )
    if determinants.get("version") != PACKING_CACHE_VERSION:
        raise ValueError(
            "packing cache determinant version mismatch: "
            f"observed={determinants.get('version')!r}, "
            f"current={PACKING_CACHE_VERSION!r}"
        )
    entries = determinants.get("determinants")
    if not isinstance(entries, list):
        raise ValueError("packing cache determinant registry entries must be a list")
    observed_names: set[str] = set()
    entries_by_name: dict[str, Mapping[str, Any]] = {}
    repo_root = Path(__file__).resolve().parents[2]
    current_owner_identities: dict[str, dict[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(
                "packing cache determinant registry entries must be mappings"
            )
        name = entry.get("name")
        owner = entry.get("owner")
        if not isinstance(name, str) or not name:
            raise ValueError("packing cache determinant name must be non-empty")
        if name in observed_names:
            raise ValueError(f"duplicate packing cache determinant owner name: {name}")
        observed_names.add(name)
        entries_by_name[name] = entry
        expected_owner = PACKING_CACHE_DETERMINANT_OWNERS.get(name)
        if expected_owner is None:
            raise ValueError(
                f"unknown packing cache determinant owner: {name} ({owner})"
            )
        if owner != expected_owner:
            raise ValueError(
                f"packing cache determinant owner mismatch for {name}: "
                f"observed={owner!r}, expected={expected_owner!r}"
            )
        expected_source_identity = current_owner_identities.get(expected_owner)
        if expected_source_identity is None:
            expected_source_identity = {
                "path": expected_owner,
                "sha256": _file_sha256(repo_root / expected_owner),
            }
            current_owner_identities[expected_owner] = expected_source_identity
        observed_source_identity = entry.get("owner_source_identity")
        if observed_source_identity != expected_source_identity:
            raise ValueError(
                f"packing cache determinant owner source identity mismatch for {name}: "
                f"observed={observed_source_identity!r}, "
                f"expected={expected_source_identity!r}"
            )
        if entry.get("content_identity") in (None, "", {}, []):
            raise ValueError(
                f"packing cache determinant {name} must have a content identity"
            )
        if not isinstance(entry.get("reason"), str) or not entry["reason"]:
            raise ValueError(f"packing cache determinant {name} must have a reason")
        schema_version = entry.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != 1:
            raise ValueError(
                f"packing cache determinant {name} has unsupported schema version: "
                f"{schema_version!r}"
            )
    expected_names = set(PACKING_CACHE_DETERMINANT_OWNERS)
    if observed_names != expected_names:
        raise ValueError(
            "packing cache determinant owner inventory mismatch: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unknown={sorted(observed_names - expected_names)}"
        )
    semantic_identities = _semantic_determinant_content_identities(determinants)
    for name, expected_identity in semantic_identities.items():
        if entries_by_name[name].get("content_identity") != expected_identity:
            raise ValueError(f"packing cache determinant mirror mismatch for {name}")
    expected_code_identity = _registry_code_identity(entries)
    if determinants.get("code_identity") != expected_code_identity:
        raise ValueError("packing cache determinant code identity mirror mismatch")
    augmentation = determinants.get("augmentation")
    if not isinstance(augmentation, Mapping) or determinants.get(
        "split"
    ) != augmentation.get("split"):
        raise ValueError("packing cache determinant split mirror mismatch")
    aggregate = determinants.get("aggregate_fingerprint")
    expected_aggregate = _registry_entries_fingerprint(entries)
    if aggregate != expected_aggregate:
        raise ValueError(
            "packing cache determinant aggregate fingerprint mismatch: "
            f"observed={aggregate!r}, expected={expected_aggregate!r}"
        )


def _registry_code_identity(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    return {
        str(entry["name"]): dict(entry["owner_source_identity"]) for entry in entries
    }


def _validate_fingerprint(fingerprint: Any) -> str:
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
    ):
        raise ValueError(
            "fingerprint must be exactly 64 lowercase hexadecimal characters"
        )
    return fingerprint


def _validate_canonical_cache_dir(
    cache_dir: str | Path,
    *,
    cache_root: str | Path,
    fingerprint: str,
) -> Path:
    raw_cache_dir = Path(cache_dir).expanduser()
    if ".." in raw_cache_dir.parts:
        raise ValueError(
            "packing cache directory must not contain traversal components"
        )
    expected = _canonical_cache_target(cache_root, fingerprint=fingerprint)
    candidate = Path(os.path.abspath(raw_cache_dir))
    if candidate != expected:
        raise ValueError(
            "packing cache directory must use canonical "
            f"<cache-root>/{PACKING_CACHE_VERSION}/{fingerprint}"
        )
    _reject_symlink_components(candidate)
    return candidate


def _canonical_cache_target(
    cache_root: str | Path,
    *,
    fingerprint: str,
) -> Path:
    _validate_fingerprint(fingerprint)
    raw_root = Path(cache_root).expanduser()
    if ".." in raw_root.parts:
        raise ValueError("packing cache root must not contain traversal components")
    return Path(os.path.abspath(raw_root)) / PACKING_CACHE_VERSION / fingerprint


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            component_stat = current.lstat()
        except FileNotFoundError:
            return
        if stat_module.S_ISLNK(component_stat.st_mode):
            raise ValueError(
                "packing cache directory must not contain symlink components"
            )
        if current != absolute and not stat_module.S_ISDIR(component_stat.st_mode):
            raise ValueError("packing cache directory components must be directories")


def _immutable_collision_error(
    target: Path,
    *,
    fingerprint: str,
    validation_category: str,
    detail_type: str,
) -> PackingCacheInvalidError:
    allowed_categories = {
        "current_publication_invalid",
        "determinant_mismatch",
        "target_already_exists",
    }
    if validation_category not in allowed_categories:
        raise ValueError("unsupported immutable collision validation category")
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", detail_type) is None:
        raise ValueError("immutable collision detail type must be bounded")
    return PackingCacheInvalidError(
        "immutable packing cache collision: "
        f"target={target} current_version={PACKING_CACHE_VERSION} "
        f"fingerprint={fingerprint} validation_category={validation_category} "
        f"automatic_recovery=unavailable detail_type={detail_type}"
    )


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
    _validate_determinant_registry(determinants)
    return str(determinants["aggregate_fingerprint"])


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(
            json.dumps(
                payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
            )
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
    "EvalCacheEntry",
    "EvalCacheShard",
    "PackingCacheInvalidError",
    "PACKING_CACHE_MANIFEST",
    "PACKING_CACHE_MATERIALIZATION_STRATEGY",
    "PACKING_CACHE_DETERMINANT_OWNERS",
    "PACKING_CACHE_DETERMINANT_REGISTRY_VERSION",
    "PACKING_CACHE_VERSION",
    "PACK_CACHE_VERIFICATION_LEVELS",
    "build_packing_cache_materialization",
    "build_packing_cache_determinants",
    "build_packing_cache_fingerprint",
    "cache_dir_for_fingerprint",
    "cache_is_complete",
    "load_all_micro_steps_from_cache",
    "load_cache_manifest",
    "load_rank_eval_micro_steps_from_cache",
    "load_rank_micro_steps_from_cache",
    "manifest_path",
    "packing_cache_fingerprint_from_determinants",
    "write_micro_step_cache",
]

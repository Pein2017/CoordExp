"""Trainer checkpoint-state adapters for upstream save behavior gaps."""

from __future__ import annotations

from datetime import timedelta
import json
import logging
import os
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Type, cast

import torch
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import (
    ExportableState,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy

from src.metrics.reporter import _DISABLED_DIAGNOSTICS_ATTR
from src.utils.ddp_fail_fast import (
    ddp_any_rank_fail_fast,
    ddp_rank0_coordinated_fail_fast,
    maybe_ddp_context,
)

if TYPE_CHECKING:  # pragma: no cover
    from transformers import Trainer as _TrainerBase


logger = logging.getLogger(__name__)

COORDEXP_CHECKPOINT_STATE_NAME = "coordexp_checkpoint_state.pt"
COORDEXP_CHECKPOINT_COMPLETE_NAME = "coordexp_checkpoint_complete.json"
_COORDEXP_CHECKPOINT_SCHEMA_VERSION = 2
_COORDEXP_LEGACY_CHECKPOINT_SCHEMA_VERSIONS = {1}
_DEFAULT_CHECKPOINT_DDP_TIMEOUT_S = 300.0
_MINIMAL_ARTIFACT_STATE_FILES = (
    COORDEXP_CHECKPOINT_STATE_NAME,
    COORDEXP_CHECKPOINT_COMPLETE_NAME,
    TRAINER_STATE_NAME,
    "training_args.bin",
)
_TOKENIZER_ARTIFACT_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "image_processor_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}
_ADAPTER_WEIGHT_NAMES = {
    "adapter_model.safetensors",
    "adapter_model.safetensors.index.json",
    "adapter_model.bin",
    "adapter_model.bin.index.json",
}
_FULL_MODEL_WEIGHT_NAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


def _callback_state_key(callback: Any) -> str:
    cls = type(callback)
    return f"{cls.__module__}.{cls.__qualname__}"


def _iter_repo_callbacks(trainer: Any) -> list[Any]:
    callback_handler = getattr(trainer, "callback_handler", None)
    callbacks = getattr(callback_handler, "callbacks", None)
    if not isinstance(callbacks, list):
        return []

    repo_callbacks: list[Any] = []
    for callback in callbacks:
        if isinstance(callback, _FinalCheckpointCallback):
            continue
        module = str(type(callback).__module__ or "")
        if module.startswith("src."):
            repo_callbacks.append(callback)
    return repo_callbacks


def _collect_repo_callback_state(trainer: Any) -> dict[str, Mapping[str, Any]]:
    state: dict[str, Mapping[str, Any]] = {}
    for callback in _iter_repo_callbacks(trainer):
        state_dict_fn = getattr(callback, "state_dict", None)
        if not callable(state_dict_fn):
            continue
        payload = state_dict_fn()
        if not isinstance(payload, Mapping):
            continue
        state[_callback_state_key(callback)] = dict(payload)
    return state


def _restore_repo_callback_state(
    trainer: Any,
    callback_state: Mapping[str, Any],
) -> None:
    if not isinstance(callback_state, Mapping):
        raise ValueError(
            "Restartable checkpoint callback_state must be a mapping, "
            f"got {type(callback_state).__name__}"
        )

    callbacks_by_key = {
        _callback_state_key(callback): callback
        for callback in _iter_repo_callbacks(trainer)
    }
    for key, payload in callback_state.items():
        callback = callbacks_by_key.get(str(key))
        if callback is None:
            raise ValueError(
                f"Restartable checkpoint callback state has no matching callback: {key}"
            )
        load_state_dict_fn = getattr(callback, "load_state_dict", None)
        if not callable(load_state_dict_fn):
            raise ValueError(
                "Restartable checkpoint callback does not implement "
                f"load_state_dict: {key}"
            )
        load_state_dict_fn(dict(payload) if isinstance(payload, Mapping) else payload)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _iter_tokenizer_save_candidates(trainer: Any) -> list[tuple[str, Any]]:
    template = _safe_getattr(trainer, "template")
    data_collator = _safe_getattr(trainer, "data_collator")
    raw_candidates = [
        ("trainer.processing_class", _safe_getattr(trainer, "processing_class")),
        ("trainer.tokenizer", _safe_getattr(trainer, "tokenizer")),
        ("trainer.template.processor", _safe_getattr(template, "processor")),
        ("trainer.template.tokenizer", _safe_getattr(template, "tokenizer")),
        ("trainer.data_collator.tokenizer", _safe_getattr(data_collator, "tokenizer")),
    ]

    candidates: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for source, candidate in raw_candidates:
        if candidate is None:
            continue
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if callable(getattr(candidate, "save_pretrained", None)):
            candidates.append((source, candidate))
    return candidates


def _collect_file_inventory(
    checkpoint_dir: Path, names: set[str]
) -> dict[str, dict[str, int]]:
    inventory: dict[str, dict[str, int]] = {}
    for name in sorted(names):
        path = checkpoint_dir / name
        if not path.is_file():
            continue
        inventory[name] = {"size": int(path.stat().st_size)}
    return inventory


def _checkpoint_has_adapter_weights(checkpoint_dir: Path) -> bool:
    return any((checkpoint_dir / name).is_file() for name in _ADAPTER_WEIGHT_NAMES)


def _checkpoint_has_full_model_weights(checkpoint_dir: Path) -> bool:
    return any((checkpoint_dir / name).is_file() for name in _FULL_MODEL_WEIGHT_NAMES)


def _checkpoint_has_tokenizer_artifacts(checkpoint_dir: Path) -> bool:
    return bool(_collect_file_inventory(checkpoint_dir, _TOKENIZER_ARTIFACT_NAMES))


def _collect_rng_state_files(checkpoint_dir: Path) -> dict[str, dict[str, int]]:
    inventory: dict[str, dict[str, int]] = {}
    if not checkpoint_dir.is_dir():
        return inventory
    for entry in sorted(checkpoint_dir.iterdir(), key=lambda p: p.name):
        if (
            entry.is_file()
            and entry.name.startswith("rng_state")
            and entry.suffix == ".pth"
        ):
            inventory[entry.name] = {"size": int(entry.stat().st_size)}
    return inventory


def _trainer_world_size(trainer: Any) -> int:
    args = getattr(trainer, "args", None)
    raw_world_size = getattr(args, "world_size", None)
    if raw_world_size is None:
        raw_world_size = getattr(getattr(args, "training_args", None), "world_size", 1)
    try:
        return max(int(raw_world_size or 1), 1)
    except (TypeError, ValueError):
        return 1


def _expected_rng_state_files(world_size: int) -> set[str]:
    if int(world_size) > 1:
        return {f"rng_state_{idx}.pth" for idx in range(int(world_size))}
    return {"rng_state.pth"}


def _save_restartable_tokenizer_artifacts(
    checkpoint_dir: Path, trainer: Any
) -> dict[str, Any]:
    candidates = _iter_tokenizer_save_candidates(trainer)
    if not candidates:
        raise RuntimeError(
            "Restartable checkpoints require tokenizer or processor artifacts, "
            f"but no save_pretrained-capable tokenizer/processor was attached to the trainer for {checkpoint_dir}"
        )

    last_error: Exception | None = None
    for source, candidate in candidates:
        before = set(_collect_file_inventory(checkpoint_dir, _TOKENIZER_ARTIFACT_NAMES))
        try:
            candidate.save_pretrained(str(checkpoint_dir))
        except Exception as exc:  # pragma: no cover - exercised through integration paths
            last_error = exc
            logger.warning(
                "Failed to save tokenizer artifacts from %s into restartable checkpoint %s: %s",
                source,
                checkpoint_dir,
                exc,
            )
            continue
        inventory = _collect_file_inventory(checkpoint_dir, _TOKENIZER_ARTIFACT_NAMES)
        if inventory:
            return {
                "required": True,
                "saved": True,
                "source": source,
                "class": f"{type(candidate).__module__}.{type(candidate).__qualname__}",
                "files": inventory,
                "new_files": sorted(set(inventory) - before),
            }

    if last_error is not None:
        raise RuntimeError(
            "Unable to save tokenizer artifacts for restartable checkpoint "
            f"{checkpoint_dir}; last error: {last_error}"
        ) from last_error
    raise RuntimeError(
        "Tokenizer/processor save_pretrained completed but produced no recognized "
        f"tokenizer artifacts in restartable checkpoint {checkpoint_dir}"
    )


def _load_adapter_config(checkpoint_dir: Path) -> dict[str, Any] | None:
    config_path = checkpoint_dir / "adapter_config.json"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"adapter_config.json must contain a mapping, got {type(payload).__name__}"
        )
    return dict(payload)


def _build_model_artifact_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    adapter_config = _load_adapter_config(checkpoint_dir)
    modules_to_save = (
        adapter_config.get("modules_to_save") if isinstance(adapter_config, Mapping) else None
    )
    return {
        "full_model_weights": _collect_file_inventory(
            checkpoint_dir, _FULL_MODEL_WEIGHT_NAMES
        ),
        "adapter_weights": _collect_file_inventory(checkpoint_dir, _ADAPTER_WEIGHT_NAMES),
        "adapter_config": bool(adapter_config),
        "adapter_base_model_name_or_path": (
            adapter_config.get("base_model_name_or_path")
            if isinstance(adapter_config, Mapping)
            else None
        ),
        "adapter_modules_to_save": (
            sorted(str(item) for item in modules_to_save)
            if isinstance(modules_to_save, list)
            else []
        ),
        "token_embedding_policy": (
            "external_base_model_plus_coord_offset_adapter"
            if isinstance(modules_to_save, list)
            and "coord_offset_adapter" in {str(item) for item in modules_to_save}
            else "checkpoint_model_weights"
        ),
    }


def _build_checkpoint_artifact_manifest(
    checkpoint_dir: Path,
    *,
    trainer: Any,
    tokenizer_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": _build_model_artifact_manifest(checkpoint_dir),
        "tokenizer": dict(tokenizer_artifacts),
        "trainer_state": {
            "optimizer.pt": (checkpoint_dir / "optimizer.pt").is_file(),
            "scheduler.pt": (checkpoint_dir / "scheduler.pt").is_file(),
            TRAINER_STATE_NAME: (checkpoint_dir / TRAINER_STATE_NAME).is_file(),
            "training_args.bin": (checkpoint_dir / "training_args.bin").is_file(),
            "world_size": _trainer_world_size(trainer),
            "rng_state_files": _collect_rng_state_files(checkpoint_dir),
            "rng_state": _checkpoint_has_rng_state(checkpoint_dir),
        },
    }


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), str(tmp_path))
    os.replace(str(tmp_path), str(path))


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(tmp_path), str(path))


def _build_coordexp_checkpoint_state(
    trainer: Any,
    *,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    disabled = getattr(trainer, _DISABLED_DIAGNOSTICS_ATTR, None)
    disabled_diagnostics = (
        sorted(str(name) for name in disabled) if isinstance(disabled, set) else []
    )

    trainer_runtime_state: Mapping[str, Any] | None = None
    export_hook = getattr(trainer, "_coordexp_checkpoint_runtime_state", None)
    if callable(export_hook):
        payload = export_hook()
        if isinstance(payload, Mapping):
            trainer_runtime_state = dict(payload)

    return {
        "schema_version": _COORDEXP_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_mode": str(
            getattr(getattr(trainer, "args", None), "checkpoint_mode", "artifact_only")
            or "artifact_only"
        ),
        "global_step": int(
            getattr(getattr(trainer, "state", None), "global_step", 0) or 0
        ),
        "disabled_diagnostics": disabled_diagnostics,
        "callback_state": _collect_repo_callback_state(trainer),
        "trainer_runtime_state": dict(trainer_runtime_state or {}),
        "artifacts": dict(artifacts or {}),
    }


def _write_coordexp_checkpoint_state(checkpoint_dir: str | Path, trainer: Any) -> str:
    checkpoint_path = Path(str(checkpoint_dir))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    tokenizer_artifacts = _save_restartable_tokenizer_artifacts(checkpoint_path, trainer)
    artifacts = _build_checkpoint_artifact_manifest(
        checkpoint_path,
        trainer=trainer,
        tokenizer_artifacts=tokenizer_artifacts,
    )
    payload = _build_coordexp_checkpoint_state(trainer, artifacts=artifacts)
    sidecar_path = checkpoint_path / COORDEXP_CHECKPOINT_STATE_NAME
    _atomic_torch_save(payload, sidecar_path)
    _atomic_json_dump(
        {
            "schema_version": _COORDEXP_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_mode": payload["checkpoint_mode"],
            "global_step": payload["global_step"],
            "sidecar": COORDEXP_CHECKPOINT_STATE_NAME,
        },
        checkpoint_path / COORDEXP_CHECKPOINT_COMPLETE_NAME,
    )
    return str(sidecar_path)


def _is_restartable_checkpoint_mode(trainer: Any) -> bool:
    return (
        str(
            getattr(getattr(trainer, "args", None), "checkpoint_mode", "artifact_only")
            or "artifact_only"
        )
        == "restartable"
    )


def _uses_minimal_artifact_checkpoint(trainer: Any) -> bool:
    if _is_restartable_checkpoint_mode(trainer):
        return False
    return bool(
        getattr(getattr(trainer, "args", None), "minimal_checkpoint_artifacts", False)
    )


def _prune_minimal_artifact_checkpoint(checkpoint_dir: str | Path) -> None:
    checkpoint_path = Path(str(checkpoint_dir))
    for name in _MINIMAL_ARTIFACT_STATE_FILES:
        path = checkpoint_path / name
        if path.exists():
            path.unlink()


def load_coordexp_checkpoint_state(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(str(checkpoint_dir)) / COORDEXP_CHECKPOINT_STATE_NAME
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Restartable checkpoint sidecar is missing: {checkpoint_path}"
        )
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(
            "Restartable checkpoint sidecar must deserialize to a Mapping, "
            f"got {type(payload).__name__}"
        )
    return dict(payload)


def _checkpoint_has_model_weights(checkpoint_dir: Path) -> bool:
    if not checkpoint_dir.is_dir():
        return False
    accepted_names = _FULL_MODEL_WEIGHT_NAMES | _ADAPTER_WEIGHT_NAMES
    for entry in checkpoint_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.name in accepted_names:
            return True
    return False


def _checkpoint_has_rng_state(checkpoint_dir: Path) -> bool:
    return bool(_collect_rng_state_files(checkpoint_dir))


def _adapter_state_keys(checkpoint_dir: Path) -> set[str]:
    safe_weights = checkpoint_dir / "adapter_model.safetensors"
    if safe_weights.is_file():
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Validating adapter_model.safetensors requires safetensors"
            ) from exc
        with safe_open(str(safe_weights), framework="pt", device="cpu") as handle:
            return {str(key) for key in handle.keys()}

    bin_weights = checkpoint_dir / "adapter_model.bin"
    if bin_weights.is_file():
        payload = torch.load(str(bin_weights), map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(
                "adapter_model.bin must deserialize to a mapping, "
                f"got {type(payload).__name__}"
            )
        return {str(key) for key in payload.keys()}

    return set()


def _validate_adapter_checkpoint(checkpoint_dir: Path) -> None:
    if not _checkpoint_has_adapter_weights(checkpoint_dir):
        return
    adapter_config = _load_adapter_config(checkpoint_dir)
    if adapter_config is None:
        raise ValueError(
            "Restartable PEFT checkpoint is incomplete: adapter weights exist "
            "but adapter_config.json is missing"
        )

    modules_to_save_raw = adapter_config.get("modules_to_save") or []
    modules_to_save = (
        {str(item) for item in modules_to_save_raw}
        if isinstance(modules_to_save_raw, list)
        else set()
    )
    if "coord_offset_adapter" not in modules_to_save:
        return

    keys = _adapter_state_keys(checkpoint_dir)
    if not keys:
        raise ValueError(
            "coord_offset_adapter is declared in adapter_config.json, but no "
            "unsharded adapter_model.safetensors/bin payload was found for validation"
        )
    has_coord_ids = any(key.endswith("coord_offset_adapter.coord_ids") for key in keys)
    has_embed_offset = any(
        key.endswith("coord_offset_adapter.embed_offset") for key in keys
    )
    if not has_coord_ids or not has_embed_offset:
        raise ValueError(
            "coord_offset_adapter is declared in adapter_config.json, but adapter "
            "weights are missing coord_ids/embed_offset tensors"
        )


def _validate_schema2_artifacts(checkpoint_dir: Path, payload: Mapping[str, Any]) -> None:
    marker_path = checkpoint_dir / COORDEXP_CHECKPOINT_COMPLETE_NAME
    if not marker_path.is_file():
        raise ValueError(
            "Restartable checkpoint is incomplete for save_model_only=true. "
            f"Missing required artifacts: {COORDEXP_CHECKPOINT_COMPLETE_NAME} "
            f"in {checkpoint_dir}"
        )
    try:
        marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Restartable checkpoint completion marker is invalid JSON: {marker_path}"
        ) from exc
    if not isinstance(marker_payload, Mapping):
        raise ValueError(
            "Restartable checkpoint completion marker must contain a mapping, "
            f"got {type(marker_payload).__name__}"
        )
    for key in ("schema_version", "checkpoint_mode", "global_step"):
        if marker_payload.get(key) != payload.get(key):
            raise ValueError(
                "Restartable checkpoint completion marker does not match sidecar: "
                f"{key} marker={marker_payload.get(key)!r} sidecar={payload.get(key)!r}"
            )
    if marker_payload.get("sidecar") != COORDEXP_CHECKPOINT_STATE_NAME:
        raise ValueError(
            "Restartable checkpoint completion marker references an unexpected sidecar: "
            f"{marker_payload.get('sidecar')!r}"
        )

    artifacts = payload.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, Mapping):
        raise ValueError(
            "Restartable checkpoint sidecar artifacts must be a mapping, "
            f"got {type(artifacts).__name__}"
        )
    if not _checkpoint_has_tokenizer_artifacts(checkpoint_dir):
        raise ValueError(
            "Restartable checkpoint is incomplete for save_model_only=true. "
            f"Missing tokenizer artifacts in {checkpoint_dir}"
        )
    tokenizer_artifacts = (
        artifacts.get("tokenizer") if isinstance(artifacts, Mapping) else None
    )
    if not isinstance(tokenizer_artifacts, Mapping):
        raise ValueError(
            "Restartable checkpoint sidecar artifacts.tokenizer must be a mapping"
        )
    tokenizer_files = tokenizer_artifacts.get("files")
    if not isinstance(tokenizer_files, Mapping) or not tokenizer_files:
        raise ValueError(
            "Restartable checkpoint sidecar artifacts.tokenizer.files must list saved tokenizer files"
        )
    missing_tokenizer_files: list[str] = []
    size_mismatch_tokenizer_files: list[str] = []
    for name, metadata in tokenizer_files.items():
        path = checkpoint_dir / str(name)
        if not path.is_file():
            missing_tokenizer_files.append(str(name))
            continue
        if isinstance(metadata, Mapping) and "size" in metadata:
            try:
                expected_size = int(metadata["size"])
            except (TypeError, ValueError):
                expected_size = None
            if expected_size is not None and int(path.stat().st_size) != expected_size:
                size_mismatch_tokenizer_files.append(str(name))
    if missing_tokenizer_files:
        raise ValueError(
            "Restartable checkpoint is incomplete for save_model_only=true. "
            "Missing tokenizer artifacts: "
            + ", ".join(sorted(missing_tokenizer_files))
            + f" in {checkpoint_dir}"
        )
    if size_mismatch_tokenizer_files:
        raise ValueError(
            "Restartable checkpoint tokenizer artifact size mismatch: "
            + ", ".join(sorted(size_mismatch_tokenizer_files))
            + f" in {checkpoint_dir}"
        )

    trainer_state_artifacts = (
        artifacts.get("trainer_state") if isinstance(artifacts, Mapping) else None
    )
    if not isinstance(trainer_state_artifacts, Mapping):
        raise ValueError(
            "Restartable checkpoint sidecar artifacts.trainer_state must be a mapping"
        )
    raw_world_size = trainer_state_artifacts.get("world_size", 1)
    try:
        world_size = max(int(raw_world_size or 1), 1)
    except (TypeError, ValueError):
        raise ValueError(
            "Restartable checkpoint sidecar trainer_state.world_size must be an integer, "
            f"got {raw_world_size!r}"
        ) from None
    existing_rng_files = set(_collect_rng_state_files(checkpoint_dir))
    if world_size <= 1:
        missing_rng_files = (
            []
            if existing_rng_files & {"rng_state.pth", "rng_state_0.pth"}
            else ["rng_state.pth"]
        )
    else:
        expected_rng_files = _expected_rng_state_files(world_size)
        missing_rng_files = sorted(expected_rng_files - existing_rng_files)
    if missing_rng_files:
        raise ValueError(
            "Restartable checkpoint is incomplete for save_model_only=true. "
            "Missing required RNG state files: "
            + ", ".join(missing_rng_files)
            + f" in {checkpoint_dir}"
        )


def validate_restartable_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(str(checkpoint_dir))
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Restartable checkpoint directory does not exist: {checkpoint_path}"
        )

    missing: list[str] = []
    if not _checkpoint_has_model_weights(checkpoint_path):
        missing.append("model weights")
    if _checkpoint_has_adapter_weights(checkpoint_path) and not (
        checkpoint_path / "adapter_config.json"
    ).is_file():
        missing.append("adapter_config.json")
    if not (checkpoint_path / "optimizer.pt").is_file():
        missing.append("optimizer.pt")
    if not (checkpoint_path / "scheduler.pt").is_file():
        missing.append("scheduler.pt")
    if not (checkpoint_path / TRAINER_STATE_NAME).is_file():
        missing.append(TRAINER_STATE_NAME)
    if not (checkpoint_path / "training_args.bin").is_file():
        missing.append("training_args.bin")
    if not _checkpoint_has_rng_state(checkpoint_path):
        missing.append("rng_state*.pth")
    if not (checkpoint_path / COORDEXP_CHECKPOINT_STATE_NAME).is_file():
        missing.append(COORDEXP_CHECKPOINT_STATE_NAME)
    if missing:
        raise ValueError(
            "Restartable checkpoint is incomplete for save_model_only=true. "
            "Missing required artifacts: "
            + ", ".join(missing)
            + f" in {checkpoint_path}"
        )

    payload = load_coordexp_checkpoint_state(checkpoint_path)
    schema_version = payload.get("schema_version")
    schema_version_int = int(schema_version or 0)
    if schema_version_int not in (
        {_COORDEXP_CHECKPOINT_SCHEMA_VERSION}
        | _COORDEXP_LEGACY_CHECKPOINT_SCHEMA_VERSIONS
    ):
        raise ValueError(
            "Restartable checkpoint sidecar schema_version is incompatible. "
            f"Expected one of "
            f"{sorted({_COORDEXP_CHECKPOINT_SCHEMA_VERSION} | _COORDEXP_LEGACY_CHECKPOINT_SCHEMA_VERSIONS)}, "
            f"got {schema_version!r}"
        )
    checkpoint_mode = str(payload.get("checkpoint_mode") or "").strip().lower()
    if checkpoint_mode != "restartable":
        raise ValueError(
            "Restartable checkpoint sidecar is incompatible: expected "
            "save_model_only=true "
            f"but found {checkpoint_mode!r}"
        )
    if schema_version_int >= 2:
        _validate_schema2_artifacts(checkpoint_path, payload)
    _validate_adapter_checkpoint(checkpoint_path)
    callback_state = payload.get("callback_state")
    if callback_state is not None and not isinstance(callback_state, Mapping):
        raise ValueError(
            "Restartable checkpoint sidecar callback_state must be a mapping, "
            f"got {type(callback_state).__name__}"
        )
    trainer_runtime_state = payload.get("trainer_runtime_state")
    if trainer_runtime_state is not None and not isinstance(
        trainer_runtime_state, Mapping
    ):
        raise ValueError(
            "Restartable checkpoint sidecar trainer_runtime_state must be a mapping, "
            f"got {type(trainer_runtime_state).__name__}"
        )
    trainer_state = TrainerState.load_from_json(
        str(checkpoint_path / TRAINER_STATE_NAME)
    )
    payload_step = payload.get("global_step")
    trainer_state_step = getattr(trainer_state, "global_step", None)
    if payload_step is not None and trainer_state_step is not None:
        if int(payload_step) != int(trainer_state_step):
            raise ValueError(
                "Restartable checkpoint global_step mismatch between "
                f"{COORDEXP_CHECKPOINT_STATE_NAME} ({int(payload_step)}) and "
                f"{TRAINER_STATE_NAME} ({int(trainer_state_step)})"
            )
    return payload


def prepare_restartable_checkpoint_resume(
    trainer: Any,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    payload = validate_restartable_checkpoint(checkpoint_dir)

    disabled_diagnostics = payload.get("disabled_diagnostics")
    if isinstance(disabled_diagnostics, list):
        setattr(
            trainer,
            _DISABLED_DIAGNOSTICS_ATTR,
            {str(name) for name in disabled_diagnostics},
        )

    _restore_repo_callback_state(trainer, dict(payload.get("callback_state") or {}))

    restore_hook = getattr(trainer, "_coordexp_restore_checkpoint_runtime_state", None)
    trainer_runtime_state = dict(payload.get("trainer_runtime_state") or {})
    if trainer_runtime_state and not callable(restore_hook):
        raise ValueError(
            "Restartable checkpoint sidecar includes trainer_runtime_state, but "
            "the resumed trainer does not implement "
            "_coordexp_restore_checkpoint_runtime_state"
        )
    if callable(restore_hook):
        restore_hook(trainer_runtime_state)

    return payload


class _FinalCheckpointCallback(TrainerCallback):
    """Callback bound to a specific trainer instance to enforce the final save."""

    def __init__(self, owner: "FinalCheckpointMixin") -> None:
        self._owner_ref = weakref.ref(owner)

    def on_train_end(  # type: ignore[override]
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        owner = self._owner_ref()
        if owner is None:
            return control
        owner._maybe_save_final_checkpoint(args, state, control)
        return control


class FinalCheckpointMixin:
    """Adds checkpoint-state repairs and optional final-save fallback."""

    _final_checkpoint_callback_attr = "_final_checkpoint_callback"
    _final_checkpoint_wrapper_cache: Dict[Type, Type] = {}
    _pending_best_checkpoint_step_attr = "_pending_best_checkpoint_step"
    _checkpoint_monitor_group_attr = "_coordexp_checkpoint_monitor_group"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        setattr(self, self._pending_best_checkpoint_step_attr, None)
        if (
            not hasattr(self, self._final_checkpoint_callback_attr)
            and self._should_install_final_checkpoint_callback()
        ):
            callback = _FinalCheckpointCallback(self)
            setattr(self, self._final_checkpoint_callback_attr, callback)
            trainer = cast("_TrainerBase", self)
            trainer.add_callback(callback)

    def _determine_best_metric(self, metrics, trial):
        """Track best-checkpoint saves for `save_strategy='best'`.

        Upstream `transformers` saves the checkpoint when a new best metric is found,
        but does not populate `state.best_model_checkpoint` for `SaveStrategy.BEST`.
        Record the step here so `_save_checkpoint()` can stamp the correct path after
        the checkpoint directory exists on disk.
        """

        setattr(self, self._pending_best_checkpoint_step_attr, None)
        is_new_best_metric = super()._determine_best_metric(metrics, trial)

        if getattr(self.args, "save_strategy", SaveStrategy.NO) != SaveStrategy.BEST:
            return is_new_best_metric
        if not is_new_best_metric:
            return is_new_best_metric

        global_step = getattr(self.state, "global_step", None)
        if isinstance(global_step, int) and global_step > 0:
            setattr(self, self._pending_best_checkpoint_step_attr, global_step)
        return is_new_best_metric

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        """Preserve best-checkpoint pointers for best-only save cadence."""

        checkpoint_dir: str | None
        if self._should_use_bounded_ddp_checkpoint_save(model):
            checkpoint_dir = self._save_checkpoint_with_bounded_ddp(model, trial)
        else:
            super()._save_checkpoint(model, trial, *args, **kwargs)
            checkpoint_dir = self._get_current_checkpoint_dir(trial=trial)
        if checkpoint_dir is None:
            setattr(self, self._pending_best_checkpoint_step_attr, None)
            return

        if self._record_best_checkpoint_if_pending(checkpoint_dir):
            if getattr(self.args, "should_save", False):
                self.state.save_to_json(
                    os.path.join(checkpoint_dir, TRAINER_STATE_NAME)
                )

        if getattr(self.args, "should_save", False) and _is_restartable_checkpoint_mode(
            self
        ):
            _write_coordexp_checkpoint_state(checkpoint_dir, self)
        if getattr(
            self.args, "should_save", False
        ) and _uses_minimal_artifact_checkpoint(self):
            _prune_minimal_artifact_checkpoint(checkpoint_dir)

    def _should_use_bounded_ddp_checkpoint_save(self, model: Any) -> bool:
        """Return True when the save path is plain DDP and safe to harden locally."""

        ctx = maybe_ddp_context(model=model)
        if ctx is None:
            return False

        trainer = cast("_TrainerBase", self)
        if bool(getattr(trainer.args, "use_flash_ckpt", False)):
            return False
        if bool(getattr(trainer, "is_deepspeed_enabled", False)):
            return False
        if bool(getattr(trainer, "is_fsdp_enabled", False)):
            return False
        if (
            getattr(getattr(trainer, "accelerator", None), "parallelism_config", None)
            is not None
        ):
            return False

        tp_size = getattr(getattr(trainer, "model", None), "_tp_size", 0)
        if tp_size is not None and int(tp_size) > 1:
            return False
        return True

    def _checkpoint_ddp_timeout_s(self) -> float:
        """Return the bounded timeout for checkpoint save coordination barriers."""

        raw_timeout = getattr(getattr(self, "args", None), "ddp_timeout", None)
        if raw_timeout is None:
            return float(_DEFAULT_CHECKPOINT_DDP_TIMEOUT_S)
        try:
            timeout_s = float(raw_timeout)
        except (TypeError, ValueError):
            return float(_DEFAULT_CHECKPOINT_DDP_TIMEOUT_S)
        if timeout_s <= 0:
            return float(_DEFAULT_CHECKPOINT_DDP_TIMEOUT_S)
        return float(timeout_s)

    def _checkpoint_ddp_barrier(self, *, model: Any, phase: str) -> None:
        """Run a bounded barrier for checkpoint save coordination under plain DDP."""

        ctx = maybe_ddp_context(model=model)
        if ctx is None:
            return

        dist = ctx.dist
        if not hasattr(dist, "monitored_barrier"):
            raise RuntimeError(
                "transformers checkpoint saving under DDP requires "
                "torch.distributed.monitored_barrier for bounded coordination; "
                f"phase={str(phase)} rank={int(ctx.rank)}/{int(ctx.world_size)}."
            )

        timeout_s = float(self._checkpoint_ddp_timeout_s())
        group = getattr(self, self._checkpoint_monitor_group_attr, None)
        if group is None:
            monitor_group_timeout_s = max(
                float(timeout_s) * 2.0, float(timeout_s) + 30.0
            )
            try:
                group = dist.new_group(
                    backend="gloo",
                    timeout=timedelta(seconds=float(monitor_group_timeout_s)),
                )
            except Exception as exc:
                logger.warning(
                    "checkpoint save monitored barrier requested but gloo group init "
                    "failed; falling back to the existing DDP process-group barrier. "
                    "phase=%s rank=%s/%s timeout_s=%.1f error=%s",
                    str(phase),
                    int(ctx.rank),
                    int(ctx.world_size),
                    float(timeout_s),
                    exc,
                )
                dist.barrier()
                return
            setattr(self, self._checkpoint_monitor_group_attr, group)

        try:
            try:
                dist.monitored_barrier(
                    group=group,
                    timeout=timedelta(seconds=float(timeout_s)),
                    wait_all_ranks=True,
                )
            except TypeError:
                dist.monitored_barrier(
                    group=group,
                    timeout=timedelta(seconds=float(timeout_s)),
                )
        except Exception as exc:
            raise RuntimeError(
                "bounded DDP checkpoint barrier failed; "
                f"phase={str(phase)} rank={int(ctx.rank)}/{int(ctx.world_size)} "
                f"timeout_s={float(timeout_s):.1f}. "
                "One rank likely failed before or during checkpoint save."
            ) from exc

    def _save_checkpoint_with_bounded_ddp(self, model: Any, trial: Any) -> str:
        """Mirror the plain-DDP HF save path without the unbounded raw barrier."""

        trainer = cast("_TrainerBase", self)
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}"

        if trainer.hp_search_backend is None and trial is None:
            trainer.store_flos()

        run_dir = trainer._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        trainer.state.last_model_checkpoint = output_dir

        fix_zero3 = getattr(trainer, "_fix_zero3_gather_all_parameters", None)
        if callable(fix_zero3):
            fix_zero3()

        ddp_rank0_coordinated_fail_fast(
            where="checkpoint/save_model",
            fn_rank0_only=lambda: trainer.save_model(output_dir, _internal_call=True),
            model=model,
            barrier=lambda: self._checkpoint_ddp_barrier(
                model=model,
                phase=f"checkpoint_save:{trainer.state.global_step}",
            ),
        )

        if (
            trainer.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH]
            and trainer.state.best_global_step
        ):
            best_checkpoint_folder = (
                f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.best_global_step}"
            )
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)
            trainer.state.best_model_checkpoint = best_checkpoint_dir

        def _save_remaining_checkpoint_state() -> None:
            if not trainer.args.save_only_model:
                trainer._save_optimizer_and_scheduler(output_dir)
                trainer._save_scaler(output_dir)
                trainer._save_rng_state(output_dir)

            if getattr(trainer.args, "should_save", False):
                for cb in [
                    cb
                    for cb in trainer.callback_handler.callbacks + [trainer.control]
                    if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(trainer.state.stateful_callbacks[cb_name], list):
                        trainer.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        trainer.state.stateful_callbacks[cb_name] = cb_state
                trainer.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if trainer.args.push_to_hub:
                trainer._push_from_checkpoint(output_dir)

            if getattr(trainer.args, "should_save", False):
                trainer._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

        ddp_any_rank_fail_fast(
            where="checkpoint/post_save_state",
            fn=_save_remaining_checkpoint_state,
            model=model,
        )
        if getattr(
            trainer.args, "should_save", False
        ) and _uses_minimal_artifact_checkpoint(trainer):
            _prune_minimal_artifact_checkpoint(output_dir)
        return output_dir

    # ------------------------------------------------------------------
    # Final checkpoint helpers
    # ------------------------------------------------------------------
    def _should_install_final_checkpoint_callback(self) -> bool:
        """Return True when the repo-owned final-save fallback should be active."""

        trainer = cast("_TrainerBase", self)
        save_last_epoch = bool(getattr(trainer.args, "save_last_epoch", True))
        if not save_last_epoch:
            return False
        return self._save_strategy_requires_final_checkpoint_fallback(
            getattr(trainer.args, "save_strategy", SaveStrategy.NO)
        )

    @staticmethod
    def _save_strategy_requires_final_checkpoint_fallback(save_strategy) -> bool:
        """Return True when upstream may skip the terminal checkpoint save."""

        return save_strategy == SaveStrategy.BEST

    def _maybe_save_final_checkpoint(  # noqa: PLR0912
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        """Persist the last checkpoint if the training loop skipped it."""

        trainer = cast("_TrainerBase", self)

        save_strategy = getattr(args, "save_strategy", SaveStrategy.NO)
        if not self._save_strategy_requires_final_checkpoint_fallback(save_strategy):
            logger.debug(
                "Final checkpoint fallback skipped because save_strategy=%s is handled upstream.",
                save_strategy,
            )
            return

        should_save_rank = bool(getattr(args, "should_save", False))
        world_size = getattr(args, "world_size", 1)

        if not should_save_rank:
            if world_size <= 1:
                logger.debug(
                    "Final checkpoint skipped because no process is permitted to save checkpoints."
                )
                return
            logger.debug(
                "Final checkpoint: this rank will participate in the distributed save without writing to disk."
            )

        global_step = getattr(state, "global_step", 0)
        if not isinstance(global_step, int) or global_step <= 0:
            logger.debug("Final checkpoint skipped because global_step=%s", global_step)
            return

        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            logger.debug("Final checkpoint skipped because output_dir is undefined.")
            return

        if self._final_checkpoint_exists(output_dir, global_step):
            if should_save_rank:
                logger.debug(
                    "Final checkpoint already present for step %s", global_step
                )
            return

        checkpoint_dir = self._format_checkpoint_dir(output_dir, global_step)
        if should_save_rank:
            logger.info(
                "No checkpoint found at %s; forcing a final save.", checkpoint_dir
            )

        # Keep the forced checkpoint independent from Trainer-managed rotation so
        # save_total_limit continues to govern only the regular save cadence.
        original_limit = getattr(args, "save_total_limit", None)
        limit_suspended = (
            should_save_rank and isinstance(original_limit, int) and original_limit > 0
        )
        if limit_suspended:
            setattr(args, "save_total_limit", None)
            logger.info(
                "Temporarily disabling save_total_limit=%s while writing the final checkpoint.",
                original_limit,
            )

        try:
            try:
                trainer._save_checkpoint(trainer.model, None)  # type: ignore[misc,arg-type]
            except TypeError:
                # Some trainer overrides accept metrics; fall back to keyword form.
                trainer._save_checkpoint(trainer.model, None, metrics=None)  # type: ignore[misc,call-arg]
        finally:
            if limit_suspended:
                setattr(args, "save_total_limit", original_limit)

        # Mirror Trainer.train() behaviour so callbacks observe the save event.
        trainer.callback_handler.on_save(args, state, control)

    def _record_best_checkpoint_if_pending(self, checkpoint_dir: str) -> bool:
        """Update best-checkpoint state when the just-saved checkpoint is the new best."""

        pending_step = getattr(self, self._pending_best_checkpoint_step_attr, None)
        global_step = getattr(self.state, "global_step", None)
        setattr(self, self._pending_best_checkpoint_step_attr, None)

        if getattr(self.args, "save_strategy", SaveStrategy.NO) != SaveStrategy.BEST:
            return False
        if not isinstance(global_step, int) or global_step <= 0:
            return False
        if pending_step != global_step:
            return False
        if not os.path.isdir(checkpoint_dir):
            return False

        self.state.best_model_checkpoint = checkpoint_dir
        self.state.best_global_step = global_step
        return True

    def _get_current_checkpoint_dir(self, trial) -> str | None:
        """Return the on-disk path for the current checkpoint save."""

        trainer = cast("_TrainerBase", self)
        global_step = getattr(self.state, "global_step", None)
        if not isinstance(global_step, int) or global_step <= 0:
            return None

        run_dir = trainer._get_output_dir(trial=trial)
        return self._format_checkpoint_dir(run_dir, global_step)

    def _final_checkpoint_exists(self, output_dir: str, step: int) -> bool:
        """Return True if the checkpoint directory (or flash record) already exists."""

        trainer = cast("_TrainerBase", self)

        if getattr(trainer.args, "use_flash_ckpt", False) and hasattr(
            trainer, "_get_last_checkpoint_step"
        ):
            last_step = trainer._get_last_checkpoint_step()  # type: ignore[attr-defined]
            if isinstance(last_step, int) and last_step >= step:
                return True

        checkpoint_dir = self._format_checkpoint_dir(output_dir, step)
        if os.path.isdir(checkpoint_dir):
            return True

        last_model_checkpoint = getattr(trainer.state, "last_model_checkpoint", None)
        if isinstance(last_model_checkpoint, str) and os.path.isdir(
            last_model_checkpoint
        ):
            if os.path.basename(last_model_checkpoint) == os.path.basename(
                checkpoint_dir
            ):
                return True

        return False

    @staticmethod
    def _format_checkpoint_dir(output_dir: str, step: int) -> str:
        return os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{step}")


def with_final_checkpoint(trainer_cls: Type) -> Type:
    """Return a trainer subclass that includes :class:`FinalCheckpointMixin`."""

    if not isinstance(trainer_cls, type):
        raise TypeError("trainer_cls must be a class")

    if issubclass(trainer_cls, FinalCheckpointMixin):
        return trainer_cls

    cache = FinalCheckpointMixin._final_checkpoint_wrapper_cache
    if trainer_cls in cache:
        return cache[trainer_cls]

    wrapped = type(
        f"{trainer_cls.__name__}WithFinalCheckpoint",
        (FinalCheckpointMixin, trainer_cls),
        {},
    )
    wrapped.__module__ = trainer_cls.__module__
    cache[trainer_cls] = wrapped
    return wrapped


__all__ = [
    "COORDEXP_CHECKPOINT_STATE_NAME",
    "FinalCheckpointMixin",
    "load_coordexp_checkpoint_state",
    "prepare_restartable_checkpoint_resume",
    "validate_restartable_checkpoint",
    "with_final_checkpoint",
]

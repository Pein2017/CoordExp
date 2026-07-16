"""Provision one immutable resident ROI profile from an authored infer config."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import torch
import transformers

from src.config.inference import InferConfig, load_infer_config
from src.inference.data_parallel import (
    detect_cuda_device_count,
    resolve_visible_cuda_tokens,
)
from src.inference.parsing import PARSER_ID, PARSER_POLICY
from src.inference.runtime import assemble_runtime
from src.label_studio_coco_refinement.inference_profiles import (
    EngineProfile,
    EngineProfileStore,
    ProfileContractError,
    canonical_json,
    fingerprint_json,
)
from src.label_studio_coco_refinement.roi_launch import (
    DEFAULT_ENGINE_FACTORY_TARGET,
    ROI_LAUNCH_SCHEMA_VERSION,
    EngineFactorySpec,
    LoopbackBind,
    RoiLaunchConfig,
    load_roi_launch_config,
)
from src.label_studio_coco_refinement.roi_runtime import RESIDENT_ADAPTER_ID
from src.label_studio_coco_refinement.roi_transform import ROI_TRANSFORM_ID
from src.qwen.images import QWEN_IMAGE_PROCESSOR_KWARGS


PROVISIONING_RECEIPT_SCHEMA_VERSION = "coordexp-roi-profile-provisioning-receipt-v1"


class ProfileProvisioningError(RuntimeError):
    """The requested profile or launch publication cannot proceed safely."""


def require_single_visible_cuda(
    *,
    visible_cuda_resolver: Callable[..., tuple[str, ...]] = resolve_visible_cuda_tokens,
    cuda_device_count: Callable[[], int] = detect_cuda_device_count,
    cuda_is_available: Callable[[], bool] = torch.cuda.is_available,
) -> str:
    """Require one externally visible token and one available logical CUDA device."""

    tokens = tuple(visible_cuda_resolver(cuda_device_count=cuda_device_count))
    available_count = int(cuda_device_count())
    if len(tokens) != 1 or available_count != 1 or not cuda_is_available():
        raise ProfileProvisioningError(
            "ROI profile provisioning requires exactly one visible and available CUDA device"
        )
    return tokens[0]


def provision_roi_profile(
    *,
    infer_config_path: str | Path,
    profile_store_path: str | Path,
    launch_config_path: str | Path,
    receipt_store_path: str | Path,
    profile_name: str,
    selector: str,
    bind_host: str,
    bind_port: int,
    insertion_ack_timeout_seconds: float,
    default_width: int = 1024,
    default_height: int = 1024,
    min_axis_pixels: int,
    max_axis_pixels: int,
    max_total_pixels: int,
    deadline_seconds: float,
    runtime_loader: Callable[[InferConfig], Any] = assemble_runtime,
    gpu_preflight: Callable[[], Any] = require_single_visible_cuda,
) -> dict[str, Any]:
    """Capture and persist one profile, then publish its strict launch document."""

    profile_store = _absolute_profile_store_path(profile_store_path)
    launch_path = _resolved_output_path(launch_config_path, field="launch config")
    receipt_store = _resolved_output_path(receipt_store_path, field="receipt store")
    if len({profile_store, launch_path, receipt_store}) != 3:
        raise ProfileProvisioningError(
            "profile store, launch config, and receipt store paths must be distinct"
        )
    store = EngineProfileStore(profile_store)
    lock_paths = {
        _launch_lock_path(launch_path),
        store.lock_path,
    }
    if len(lock_paths) != 2 or lock_paths & {
        profile_store,
        launch_path,
        receipt_store,
    }:
        raise ProfileProvisioningError(
            "profile store, launch config, and receipt store must not collide "
            "with provision locks"
        )

    bind = LoopbackBind(host=bind_host, port=bind_port)
    requested_launch_payload = _validated_launch_payload(
        config_path=launch_path,
        bind=bind,
        profile_store_path=profile_store,
        receipt_store_path=receipt_store,
        insertion_ack_timeout_seconds=insertion_ack_timeout_seconds,
        profile_selectors={selector: profile_name},
    )
    _merge_launch_payload(
        launch_path,
        requested=requested_launch_payload,
        selector=selector,
        profile_name=profile_name,
    )

    resolved = load_infer_config(infer_config_path)
    artifact_paths = _artifact_paths(resolved.config)

    gpu_preflight()
    runtime = runtime_loader(resolved.config)
    runtime_identity, processor_factor = _capture_runtime_identity(runtime)
    profile = EngineProfile.capture(
        name=profile_name,
        endpoint=_loopback_endpoint(bind),
        artifact_paths=artifact_paths,
        resolved_config=resolved,
        roi_inference={
            "processor_factor": processor_factor,
            "default_width": default_width,
            "default_height": default_height,
            "min_axis_pixels": min_axis_pixels,
            "max_axis_pixels": max_axis_pixels,
            "max_total_pixels": max_total_pixels,
            "deadline_seconds": deadline_seconds,
        },
        parser_identity={"id": PARSER_ID, "policy": PARSER_POLICY},
        adapter_identity={"id": RESIDENT_ADAPTER_ID},
        transform_identity={"id": ROI_TRANSFORM_ID},
        transformers_version=transformers.__version__,
        processor_kwargs=QWEN_IMAGE_PROCESSOR_KWARGS,
        runtime_identity=runtime_identity,
    )

    # Stable order: the launch sidecar is outer; EngineProfileStore owns and
    # acquires its mutation sidecar inside save/activate.
    with _exclusive_launch_lock(launch_path):
        launch_payload, launch_fingerprint, publish_required = _merge_launch_payload(
            launch_path,
            requested=requested_launch_payload,
            selector=selector,
            profile_name=profile_name,
        )
        store.save(profile)
        saved_profiles = store.profiles()
        _require_launch_profiles_exist(
            launch_payload, saved_profile_names=set(saved_profiles)
        )
        if publish_required:
            _publish_launch_atomic(
                launch_path,
                launch_payload,
                expected_launch_fingerprint=launch_fingerprint,
            )
        loaded_launch = load_roi_launch_config(launch_path)
        _require_launch_matches(loaded_launch, expected=launch_payload)
        if not publish_required:
            _fsync_directory(launch_path.parent)
        profile_count = len(saved_profiles)
    return _safe_receipt(
        profile=profile,
        launch_payload=launch_payload,
        profile_count=profile_count,
        selector_count=len(loaded_launch.profile_selectors),
    )


def _artifact_paths(config: InferConfig) -> dict[str, Path]:
    base = Path(config.model.base_model)
    artifacts = {
        "base_weights": base,
        "model_config": base / "config.json",
        "tokenizer": base / "tokenizer.json",
        "processor": base / "preprocessor_config.json",
    }
    if config.adapter is not None:
        artifacts["adapter"] = Path(config.adapter.path)
    if config.embedding_delta is not None:
        artifacts["embedding_delta"] = Path(config.embedding_delta.path)
    return artifacts


def _capture_runtime_identity(runtime: Any) -> tuple[dict[str, Any], int]:
    qwen = getattr(runtime, "qwen", None)
    model_identity = getattr(runtime, "model_identity", None)
    processor_identity = getattr(qwen, "processor_identity", None)
    token_identity = getattr(qwen, "token_identity", None)
    processor_artifact = getattr(processor_identity, "to_artifact_dict", None)
    token_artifact = getattr(token_identity, "to_artifact_dict", None)
    if (
        not isinstance(model_identity, Mapping)
        or not callable(processor_artifact)
        or not callable(token_artifact)
    ):
        raise ProfileProvisioningError(
            "inference runtime does not expose exact model, processor, and token identities"
        )
    identity = _strict_json_copy(
        {
            "model": dict(model_identity),
            "processor": processor_artifact(),
            "tokenizer": token_artifact(),
        },
        field="runtime identity",
    )
    processor = identity["processor"]
    patch_size = _positive_int(
        processor.get("patch_size"), field="processor patch_size"
    )
    merge_size = _positive_int(
        processor.get("merge_size"), field="processor merge_size"
    )
    return identity, patch_size * merge_size


def _validated_launch_payload(
    *,
    config_path: Path,
    bind: LoopbackBind,
    profile_store_path: Path,
    receipt_store_path: Path,
    insertion_ack_timeout_seconds: float,
    profile_selectors: Mapping[str, str],
) -> dict[str, Any]:
    config = RoiLaunchConfig(
        config_path=config_path,
        bind=bind,
        profile_store_path=profile_store_path,
        receipt_store_path=receipt_store_path,
        insertion_ack_timeout_seconds=insertion_ack_timeout_seconds,
        profile_selectors=profile_selectors,
        engine_factory=EngineFactorySpec(
            target=DEFAULT_ENGINE_FACTORY_TARGET,
            config=MappingProxyType({}),
        ),
    )
    return _launch_payload_from_config(config)


def _publish_launch_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_launch_fingerprint: str | None,
) -> None:
    encoded = canonical_json(payload) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _require_launch_unchanged(
            path, expected_launch_fingerprint=expected_launch_fingerprint
        )
        if expected_launch_fingerprint is None:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise ProfileProvisioningError(
                    "launch config changed concurrently before publication"
                ) from None
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _merge_launch_payload(
    path: Path,
    *,
    requested: Mapping[str, Any],
    selector: str,
    profile_name: str,
) -> tuple[dict[str, Any], str | None, bool]:
    existing_fingerprint = _launch_file_fingerprint(path)
    if existing_fingerprint is None:
        return dict(requested), None, True
    try:
        loaded = load_roi_launch_config(path)
    except Exception:
        raise ProfileProvisioningError(
            "an incompatible launch config already exists"
        ) from None
    observed = _launch_payload_from_config(loaded)
    _require_launch_globals_match(observed, requested=requested)
    selectors = dict(loaded.profile_selectors)
    existing_profile = selectors.get(selector)
    if existing_profile is not None:
        if existing_profile != profile_name:
            raise ProfileProvisioningError(
                "launch selector is already mapped to a different profile"
            )
        return observed, existing_fingerprint, False
    bound_selector = next(
        (
            existing_selector
            for existing_selector, existing_name in selectors.items()
            if existing_name == profile_name
        ),
        None,
    )
    if bound_selector is not None:
        raise ProfileProvisioningError(
            "launch profile name is already bound to a different selector"
        )
    selectors[selector] = profile_name
    merged = RoiLaunchConfig(
        config_path=loaded.config_path,
        bind=loaded.bind,
        profile_store_path=loaded.profile_store_path,
        receipt_store_path=loaded.receipt_store_path,
        insertion_ack_timeout_seconds=loaded.insertion_ack_timeout_seconds,
        profile_selectors=selectors,
        engine_factory=loaded.engine_factory,
    )
    return _launch_payload_from_config(merged), existing_fingerprint, True


def _require_launch_globals_match(
    observed: Mapping[str, Any], *, requested: Mapping[str, Any]
) -> None:
    observed_globals = dict(observed)
    requested_globals = dict(requested)
    observed_globals.pop("profile_selectors", None)
    requested_globals.pop("profile_selectors", None)
    if canonical_json(observed_globals) != canonical_json(requested_globals):
        raise ProfileProvisioningError("an incompatible launch config already exists")


def _require_launch_profiles_exist(
    launch_payload: Mapping[str, Any], *, saved_profile_names: set[str]
) -> None:
    selectors = launch_payload.get("profile_selectors")
    if not isinstance(selectors, Mapping):
        raise ProfileProvisioningError("launch profile selectors are unavailable")
    missing_count = len(set(selectors.values()) - saved_profile_names)
    if missing_count:
        raise ProfileProvisioningError(
            f"launch references missing saved profiles: count={missing_count}"
        )


def _launch_payload_from_config(config: RoiLaunchConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "bind": config.bind.to_dict(),
        "profile_store_path": str(config.profile_store_path),
        "receipt_store_path": str(config.receipt_store_path),
        "insertion_ack_timeout_seconds": config.insertion_ack_timeout_seconds,
        "profile_selectors": dict(config.profile_selectors),
        "engine_factory": {
            "target": config.engine_factory.target,
            "config": config.engine_factory.config_copy(),
        },
    }


def _require_launch_matches(
    config: RoiLaunchConfig, *, expected: Mapping[str, Any]
) -> None:
    observed = _launch_payload_from_config(config)
    if canonical_json(observed) != canonical_json(expected):
        raise ProfileProvisioningError("an incompatible launch config already exists")


def _safe_receipt(
    *,
    profile: EngineProfile,
    launch_payload: Mapping[str, Any],
    profile_count: int,
    selector_count: int,
) -> dict[str, Any]:
    profile_receipt = profile.to_receipt_dict()
    return {
        "schema_version": PROVISIONING_RECEIPT_SCHEMA_VERSION,
        "profile": {
            "name": profile.name,
            "fingerprint": profile.fingerprint,
            "resolved_infer_config_fingerprint": profile.resolved_infer_config_fingerprint,
            "artifacts": profile_receipt["artifacts"],
            "artifact_count": len(profile.artifacts),
            "identity_fingerprints": profile.identity_fingerprints,
            "processor": profile_receipt["processor"],
            "deadline_seconds": profile.deadline_seconds,
        },
        "launch": {
            "schema_version": ROI_LAUNCH_SCHEMA_VERSION,
            "fingerprint": fingerprint_json(launch_payload),
            "selector_count": selector_count,
            "profile_count": profile_count,
        },
    }


@contextmanager
def _exclusive_launch_lock(launch_path: Path) -> Iterator[None]:
    lock_path = _launch_lock_path(launch_path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
    except OSError:
        raise ProfileProvisioningError("launch provision lock is unavailable") from None
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError:
            raise ProfileProvisioningError(
                "launch provision lock is unavailable"
            ) from None
        try:
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def _launch_lock_path(launch_path: Path) -> Path:
    return launch_path.with_name(f".{launch_path.name}.provision.lock")


def _launch_file_fingerprint(path: Path) -> str | None:
    if not os.path.lexists(path):
        return None
    try:
        payload = path.read_bytes()
    except OSError:
        raise ProfileProvisioningError(
            "an incompatible launch config already exists"
        ) from None
    return hashlib.sha256(payload).hexdigest()


def _require_launch_unchanged(
    path: Path, *, expected_launch_fingerprint: str | None
) -> None:
    if _launch_file_fingerprint(path) != expected_launch_fingerprint:
        raise ProfileProvisioningError(
            "launch config changed concurrently before publication"
        )


def _absolute_profile_store_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ProfileProvisioningError("profile store path must be absolute")
    return path.resolve(strict=False)


def _resolved_output_path(value: str | Path, *, field: str) -> Path:
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ProfileProvisioningError(f"{field} path is unavailable") from None


def _loopback_endpoint(bind: LoopbackBind) -> str:
    host = bind.host
    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{url_host}:{bind.port}/infer"


def _strict_json_copy(value: Any, *, field: str) -> dict[str, Any]:
    try:
        copied = json.loads(canonical_json(value))
    except ProfileContractError as exc:
        raise ProfileProvisioningError(f"{field} is not strict JSON") from exc
    if not isinstance(copied, dict):
        raise ProfileProvisioningError(f"{field} must be a JSON object")
    return copied


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProfileProvisioningError(f"{field} must be a positive integer")
    return value


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

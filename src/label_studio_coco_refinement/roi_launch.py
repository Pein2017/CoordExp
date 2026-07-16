"""Strict operator-owned launch boundary for resident ROI inference.

The browser selects only an allowlisted profile selector. Filesystem paths,
loopback bind, acknowledgement policy, and the exact checked-in empty-config
factory declaration come from one operator JSON file loaded before requests.
"""

from __future__ import annotations

import importlib
import ipaddress
import json
import math
import re
import socket
import threading
import time
from copy import deepcopy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

from src.common.errors import RuntimeContractError
from src.config.fingerprint import sha256_json
from src.config.inference import (
    INFER_CONFIG_LOADER_VERSION,
    InferConfig,
    ResolvedInferConfig,
)
from src.label_studio_coco_refinement.inference_profiles import (
    EngineProfile,
    EngineProfileStore,
    ProfileContractError,
    ProfileDriftError,
)
from src.label_studio_coco_refinement.resident_inference import (
    CancellationMetadata,
    ResidentInferenceCancelled,
    ResidentProfileBinding,
    ResidentRoiInferenceEngine,
)
from src.label_studio_coco_refinement.roi_runtime import (
    CurrentTargetProvider,
    InferenceReceiptStore,
    RoiInferenceService,
    build_resident_profile_binding,
)


ROI_LAUNCH_SCHEMA_VERSION = "coordexp-roi-launch-v1"
DEFAULT_ENGINE_FACTORY_TARGET = (
    "src.label_studio_coco_refinement.roi_launch:build_resident_engine_from_profile"
)
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "bind",
    "profile_store_path",
    "receipt_store_path",
    "insertion_ack_timeout_seconds",
    "profile_selectors",
    "engine_factory",
}
_BIND_FIELDS = {"host", "port"}
_FACTORY_FIELDS = {"target", "config"}
_SELECTOR_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_PROJECT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_FACTORY_TARGET_PATTERN = re.compile(
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z"
)
_NO_TARGET_FINGERPRINT = object()


class RoiLaunchConfigError(ValueError):
    """The operator launch document is unsafe, ambiguous, or malformed."""


class RoiLaunchError(RuntimeError):
    """The configured resident launch lifecycle cannot proceed safely."""


class EngineFactory(Protocol):
    def __call__(
        self,
        *,
        profile: EngineProfile,
        config: Mapping[str, Any],
    ) -> Any: ...


@dataclass(frozen=True)
class LoopbackBind:
    host: str
    port: int

    def __post_init__(self) -> None:
        _require_loopback_host(self.host, field="bind.host")
        _require_port(self.port, field="bind.port")

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port}


@dataclass(frozen=True)
class EngineFactorySpec:
    target: str
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.target != DEFAULT_ENGINE_FACTORY_TARGET:
            raise RoiLaunchConfigError(
                "engine_factory.target must be the checked-in resident factory"
            )
        if not isinstance(self.config, Mapping):
            raise RoiLaunchConfigError("engine_factory.config must be a JSON object")
        if self.config:
            raise RoiLaunchConfigError(
                "engine_factory.config must be exactly empty for the resident factory"
            )

    def config_copy(self) -> dict[str, Any]:
        return _deep_thaw(self.config)


@dataclass(frozen=True)
class RoiLaunchConfig:
    config_path: Path
    bind: LoopbackBind
    profile_store_path: Path
    receipt_store_path: Path
    insertion_ack_timeout_seconds: float | None
    profile_selectors: Mapping[str, str]
    engine_factory: EngineFactorySpec
    schema_version: str = ROI_LAUNCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROI_LAUNCH_SCHEMA_VERSION:
            raise RoiLaunchConfigError("unsupported ROI launch schema")
        if not self.config_path.is_absolute():
            raise RoiLaunchConfigError("operator config path must resolve absolutely")
        for path, field in (
            (self.profile_store_path, "profile_store_path"),
            (self.receipt_store_path, "receipt_store_path"),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise RoiLaunchConfigError(f"{field} must resolve absolutely")
            try:
                is_directory = path.exists() and path.is_dir()
            except OSError:
                raise RoiLaunchConfigError(f"{field} path is unavailable") from None
            if is_directory:
                raise RoiLaunchConfigError(f"{field} must name a file, not a directory")
        if self.profile_store_path == self.receipt_store_path:
            raise RoiLaunchConfigError(
                "profile_store_path and receipt_store_path must be distinct"
            )
        _validate_ack_timeout(self.insertion_ack_timeout_seconds)
        if (
            not isinstance(self.profile_selectors, Mapping)
            or not self.profile_selectors
        ):
            raise RoiLaunchConfigError("profile_selectors must be a non-empty object")
        normalized: dict[str, str] = {}
        for selector, profile_name in self.profile_selectors.items():
            if not isinstance(selector, str) or not _SELECTOR_PATTERN.fullmatch(
                selector
            ):
                raise RoiLaunchConfigError(
                    "profile selector must be a safe 1-64 character token"
                )
            if not isinstance(profile_name, str) or not _SELECTOR_PATTERN.fullmatch(
                profile_name
            ):
                raise RoiLaunchConfigError(
                    "saved profile name must be a safe 1-64 character token"
                )
            normalized[selector] = profile_name
        if len(set(normalized.values())) != len(normalized):
            raise RoiLaunchConfigError(
                "profile_selectors must map one-to-one onto saved profiles"
            )
        object.__setattr__(
            self,
            "profile_selectors",
            MappingProxyType(dict(sorted(normalized.items()))),
        )


@dataclass(frozen=True)
class InternalRoiProfileBinding:
    """Credential-free profile contract for trusted in-process adapters.

    This is deliberately richer than the browser projection and deliberately
    narrower than :class:`EngineProfile`: no endpoint, filesystem path,
    artifact identity, runtime payload, or saved profile name crosses the
    Django integration boundary.
    """

    fingerprint: str
    processor_factor: int
    default_width: int
    default_height: int
    min_axis_pixels: int
    max_axis_pixels: int
    max_total_pixels: int
    deadline_seconds: float


def load_roi_launch_config(path: str | Path) -> RoiLaunchConfig:
    """Load one strict operator JSON document with config-relative paths."""

    try:
        config_path = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise RoiLaunchConfigError("operator config path is unavailable") from None
    if not config_path.is_file():
        raise RoiLaunchConfigError("operator config path must name a JSON file")
    try:
        payload = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except RoiLaunchConfigError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise RoiLaunchConfigError("cannot read strict ROI launch JSON") from None
    if not isinstance(payload, dict):
        raise RoiLaunchConfigError("ROI launch config must be a JSON object")
    _validate_finite_json_numbers(payload, field="launch config")
    _require_exact_fields(payload, _TOP_LEVEL_FIELDS, field="launch config")
    if payload["schema_version"] != ROI_LAUNCH_SCHEMA_VERSION:
        raise RoiLaunchConfigError("unsupported ROI launch schema")

    raw_bind = payload["bind"]
    if not isinstance(raw_bind, dict):
        raise RoiLaunchConfigError("bind must be a JSON object")
    _require_exact_fields(raw_bind, _BIND_FIELDS, field="bind")
    bind = LoopbackBind(host=raw_bind["host"], port=raw_bind["port"])

    raw_factory = payload["engine_factory"]
    if not isinstance(raw_factory, dict):
        raise RoiLaunchConfigError("engine_factory must be a JSON object")
    _require_exact_fields(raw_factory, _FACTORY_FIELDS, field="engine_factory")
    raw_factory_config = raw_factory["config"]
    if not isinstance(raw_factory_config, dict):
        raise RoiLaunchConfigError("engine_factory.config must be a JSON object")
    if raw_factory["target"] != DEFAULT_ENGINE_FACTORY_TARGET:
        raise RoiLaunchConfigError(
            "engine_factory.target must be the checked-in resident factory"
        )
    if raw_factory_config:
        raise RoiLaunchConfigError(
            "engine_factory.config must be exactly empty for the resident factory"
        )
    factory = EngineFactorySpec(
        target=raw_factory["target"],
        config=MappingProxyType({}),
    )

    profile_store_path = _resolve_operator_path(
        payload["profile_store_path"],
        base_dir=config_path.parent,
        field="profile_store_path",
    )
    receipt_store_path = _resolve_operator_path(
        payload["receipt_store_path"],
        base_dir=config_path.parent,
        field="receipt_store_path",
    )
    timeout = _validate_ack_timeout(payload["insertion_ack_timeout_seconds"])
    selectors = payload["profile_selectors"]
    if not isinstance(selectors, dict):
        raise RoiLaunchConfigError("profile_selectors must be a JSON object")
    return RoiLaunchConfig(
        config_path=config_path,
        bind=bind,
        profile_store_path=profile_store_path,
        receipt_store_path=receipt_store_path,
        insertion_ack_timeout_seconds=timeout,
        profile_selectors=selectors,
        engine_factory=factory,
    )


class RoiLaunchManager:
    """Own one profile store, receipt authority, service, and engine lifecycle."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        current_targets: CurrentTargetProvider,
        engine_factory_loader: Callable[[str], EngineFactory] = None,
        profile_store_factory: Callable[
            [Path], EngineProfileStore
        ] = EngineProfileStore,
        receipt_store_factory: Callable[..., InferenceReceiptStore] = (
            InferenceReceiptStore
        ),
        service_factory: Callable[..., RoiInferenceService] = RoiInferenceService,
        receipt_clock: Callable[[], float] = time.time,
        service_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        try:
            current_target = getattr(current_targets, "current_target", None)
        except BaseException:
            raise RoiLaunchConfigError(
                "current_targets must provide the server-owned current_target seam"
            ) from None
        if not callable(current_target):
            raise RoiLaunchConfigError(
                "current_targets must provide the server-owned current_target seam"
            )
        self.config = load_roi_launch_config(config_path)
        try:
            self.profile_store = profile_store_factory(self.config.profile_store_path)
            saved_profiles = self.profile_store.profiles()
        except BaseException:
            raise RoiLaunchConfigError("profile store could not be loaded") from None
        _validate_saved_profile_mapping(saved_profiles)
        for profile in saved_profiles.values():
            _require_loopback_endpoint(
                profile.endpoint,
                field=f"saved profile {profile.name!r} endpoint",
            )
        missing = sorted(
            set(self.config.profile_selectors.values()) - set(saved_profiles)
        )
        if missing:
            raise RoiLaunchConfigError(
                "profile_selectors reference unknown saved profiles: "
                f"count={len(missing)}"
            )
        self._profile_snapshots = {
            name: saved_profiles[name]
            for name in sorted(set(self.config.profile_selectors.values()))
        }
        self._expected_profile_fingerprints = {
            name: profile.fingerprint
            for name, profile in self._profile_snapshots.items()
        }

        loader = (
            _import_engine_factory
            if engine_factory_loader is None
            else engine_factory_loader
        )
        try:
            engine_factory = loader(self.config.engine_factory.target)
        except BaseException:
            raise RoiLaunchConfigError(
                "engine_factory.target could not be resolved"
            ) from None
        if not callable(engine_factory):
            raise RoiLaunchConfigError("engine_factory.target is not callable")
        self._engine_factory = engine_factory

        try:
            self.receipt_store = receipt_store_factory(
                self.config.receipt_store_path,
                clock=receipt_clock,
            )
        except BaseException:
            raise RoiLaunchError("receipt store could not be initialized") from None
        self.inference_receipt_resolver = self.receipt_store
        try:
            self.inference_service = service_factory(
                profiles=self.profile_store,
                receipts=self.receipt_store,
                current_targets=current_targets,
                clock=service_clock,
                insertion_ack_timeout_seconds=(
                    self.config.insertion_ack_timeout_seconds
                ),
            )
            retained_receipts = getattr(self.inference_service, "receipts", None)
        except BaseException:
            raise RoiLaunchError(
                "ROI inference service could not be initialized"
            ) from None
        if retained_receipts is not self.receipt_store:
            raise RoiLaunchError(
                "RoiInferenceService did not retain the sole receipt-store authority"
            )

        self._state_lock = threading.Condition(threading.Lock())
        self._single_flight = threading.Lock()
        self._activation_lock = threading.RLock()
        self._closed = threading.Event()
        self._lifecycle_state = "OPEN"
        self._close_failed = False
        self._engines_by_profile: dict[str, _ManagedResidentEngine] = {}
        self._engine_failures_by_profile: set[str] = set()
        self._unavailable_projects: set[str] = set()

    def profile_options(self) -> tuple[dict[str, Any], ...]:
        """Return the only browser-safe profile projection."""

        return tuple(
            _safe_profile_projection(
                selector=selector,
                profile=self._profile_snapshots[profile_name],
            )
            for selector, profile_name in self.config.profile_selectors.items()
        )

    @staticmethod
    def new_cancellation_token() -> Any:
        """Construct the runtime-owned token without importing it from Django."""

        from src.label_studio_coco_refinement.resident_inference import CancellationToken

        return CancellationToken()

    def engine_for(self, selector: str) -> Any:
        """Resolve one allowlisted selector and load its engine at most once."""

        profile_name = self._profile_name_for_selector(selector)
        with self._state_lock:
            self._require_open()
            return self._engine_for_profile_locked(profile_name)

    def activate_profile(self, *, project_id: str, selector: str) -> dict[str, Any]:
        """Atomically load, verify, and persist one project-scoped selection."""

        self.resolve_selected_profile(project_id=project_id, selector=selector)
        profile_name = self._profile_name_for_selector(selector)
        return _safe_profile_projection(
            selector=selector,
            profile=self._profile_snapshots[profile_name],
        )

    def resolve_selected_profile(
        self,
        *,
        project_id: str,
        selector: str,
    ) -> InternalRoiProfileBinding:
        """Load, verify, and activate one selected project profile.

        Callers must invoke this before acquiring application database locks;
        engine loading and artifact verification are intentionally part of the
        activation boundary.
        """

        self._profile_name_for_selector(selector)
        project_id = self._validate_project_id(project_id)
        with self._activation_lock:
            self._require_project_available(project_id)
            _, profile = self._prepare_project_engine(
                project_id=project_id,
                selector=selector,
            )
            return _internal_profile_binding(profile)

    def current_profile(self, *, project_id: str) -> InternalRoiProfileBinding:
        """Return the verified active profile for one available project."""

        project_id = self._validate_project_id(project_id)
        with self._activation_lock:
            self._require_project_available(project_id)
            with self._state_lock:
                self._require_open()
            try:
                profile = self.profile_store.active(project_id, verify=True)
            except BaseException:
                raise RoiLaunchError(
                    "project active profile could not be verified"
                ) from None
            if not self._activation_matches_expected(profile):
                raise RoiLaunchError(
                    "project active profile could not be verified"
                ) from None
            verified = self._verified_profile(profile.name)
            if not self._same_profile(profile, verified):
                raise RoiLaunchError(
                    "project active profile could not be verified"
                ) from None
            return _internal_profile_binding(verified)

    def infer(self, *, selector: str, **kwargs: Any) -> dict[str, Any]:
        """Execute through the sole service without accepting any request path."""

        self._profile_name_for_selector(selector)
        target = kwargs.get("target")
        project_id = self._validate_project_id(getattr(target, "project_id", None))
        with self._activation_lock:
            self._require_project_available(project_id)
            engine, _ = self._prepare_project_engine(
                project_id=project_id,
                selector=selector,
                target_profile_fingerprint=getattr(
                    target,
                    "profile_fingerprint",
                    None,
                ),
            )
            try:
                return self.inference_service.infer(engine=engine, **kwargs)
            except BaseException:
                raise RoiLaunchError("ROI inference service failed") from None

    def close(self) -> None:
        """Wait for the global slot, close loaded engines in name order, once."""

        with self._activation_lock:
            with self._state_lock:
                while self._lifecycle_state == "CLOSING":
                    self._state_lock.wait()
                if self._lifecycle_state == "CLOSED":
                    self._raise_close_failure()
                    return
                if self._lifecycle_state != "OPEN":  # pragma: no cover - invariant
                    raise RoiLaunchError("invalid ROI launch lifecycle state")
                self._lifecycle_state = "CLOSING"
                self._closed.set()
                engines = tuple(sorted(self._engines_by_profile.items()))
                self._engines_by_profile.clear()
            close_failed = False
            try:
                with self._single_flight:
                    for _, engine in engines:
                        try:
                            engine._close_locked()
                        except BaseException:
                            close_failed = True
            finally:
                with self._state_lock:
                    self._close_failed = close_failed
                    self._lifecycle_state = "CLOSED"
                    self._state_lock.notify_all()
            self._raise_close_failure()

    def __enter__(self) -> "RoiLaunchManager":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()

    def _profile_name_for_selector(self, selector: Any) -> str:
        if not isinstance(selector, str) or not _SELECTOR_PATTERN.fullmatch(selector):
            raise RoiLaunchError("invalid or unknown ROI profile selector")
        profile_name = self.config.profile_selectors.get(selector)
        if profile_name is None:
            raise RoiLaunchError("invalid or unknown ROI profile selector")
        return profile_name

    @staticmethod
    def _validate_project_id(project_id: Any) -> str:
        if not isinstance(project_id, str) or not _PROJECT_ID_PATTERN.fullmatch(
            project_id
        ):
            raise RoiLaunchError("invalid ROI project identity")
        return project_id

    def _engine_for_profile_locked(self, profile_name: str) -> Any:
        existing = self._engines_by_profile.get(profile_name)
        if existing is not None:
            return existing
        if profile_name in self._engine_failures_by_profile:
            raise RoiLaunchError(
                "resident engine initialization is terminally failed"
            ) from None
        with self._single_flight:
            self._require_open()
            engine: Any | None = None
            try:
                profile = self._verified_profile(profile_name)
                engine = self._engine_factory(
                    profile=profile,
                    config=self.config.engine_factory.config_copy(),
                )
                profile = self._verified_profile(profile_name)
                _validate_loaded_engine(engine, profile=profile)
                managed = _ManagedResidentEngine(
                    engine=engine,
                    profile=build_resident_profile_binding(profile),
                    verify_profile=lambda: self._verified_profile(profile_name),
                    single_flight=self._single_flight,
                    closed=self._closed,
                )
            except BaseException:
                if engine is not None:
                    try:
                        _close_engine(engine)
                    except BaseException:
                        pass
                self._engine_failures_by_profile.add(profile_name)
                raise RoiLaunchError("resident engine initialization failed") from None
            self._engines_by_profile[profile_name] = managed
            return managed

    def _prepare_project_engine(
        self,
        *,
        project_id: str,
        selector: Any,
        target_profile_fingerprint: Any = _NO_TARGET_FINGERPRINT,
    ) -> tuple[Any, EngineProfile]:
        profile_name = self._profile_name_for_selector(selector)
        self._require_project_available(project_id)
        with self._state_lock:
            self._require_open()
            expected_fingerprint = self._expected_profile_fingerprints[profile_name]
            if (
                target_profile_fingerprint is not _NO_TARGET_FINGERPRINT
                and target_profile_fingerprint != expected_fingerprint
            ):
                raise RoiLaunchError(
                    "request target does not match the selected ROI profile"
                )
            engine = self._engine_for_profile_locked(profile_name)
            with self._single_flight:
                self._require_open()
                self._require_project_available(project_id)
                profile = self._verified_profile(profile_name)
                prior = self._capture_prior_activation(project_id)
                if not self._activate_and_verify(
                    project_id=project_id,
                    profile=profile,
                ):
                    if self._restore_prior_activation(
                        project_id=project_id,
                        prior=prior,
                    ):
                        raise RoiLaunchError(
                            "project profile activation failed; prior activation "
                            "restored"
                        ) from None
                    self._unavailable_projects.add(project_id)
                    raise RoiLaunchError(
                        "project profile activation is terminally unavailable"
                    ) from None
                return engine, profile

    def _verified_profile(self, profile_name: str) -> EngineProfile:
        try:
            current = self.profile_store.profiles().get(profile_name)
            expected_fingerprint = self._expected_profile_fingerprints[profile_name]
            if (
                not isinstance(current, EngineProfile)
                or current.fingerprint != expected_fingerprint
            ):
                raise RoiLaunchError("resident profile identity changed")
            _require_loopback_endpoint(
                current.endpoint,
                field="saved profile endpoint",
            )
            current.verify_artifacts()
        except BaseException:
            raise RoiLaunchError("resident profile verification failed") from None
        return current

    def _capture_prior_activation(self, project_id: str) -> EngineProfile | None:
        try:
            prior = self.profile_store.active(project_id, verify=True)
        except ProfileContractError as error:
            missing_message = f"project has no active profile: {project_id}"
            args = error.args if type(error) is ProfileContractError else ()
            if (
                len(args) == 1
                and isinstance(args[0], str)
                and args[0] == missing_message
            ):
                return None
            self._unavailable_projects.add(project_id)
            raise RoiLaunchError(
                "project active profile could not be verified"
            ) from None
        except BaseException:
            self._unavailable_projects.add(project_id)
            raise RoiLaunchError(
                "project active profile could not be verified"
            ) from None
        if not self._activation_matches_expected(prior):
            self._unavailable_projects.add(project_id)
            raise RoiLaunchError(
                "project active profile could not be verified"
            ) from None
        return prior

    def _activate_and_verify(
        self,
        *,
        project_id: str,
        profile: EngineProfile,
    ) -> bool:
        try:
            activated = self.profile_store.activate(project_id, profile.name)
            active = self.profile_store.active(project_id, verify=True)
            return self._same_profile(activated, profile) and self._same_profile(
                active,
                profile,
            )
        except BaseException:
            return False

    def _restore_prior_activation(
        self,
        *,
        project_id: str,
        prior: EngineProfile | None,
    ) -> bool:
        if prior is None or not self._activation_matches_expected(prior):
            return False
        try:
            restored = self.profile_store.activate(project_id, prior.name)
            active = self.profile_store.active(project_id, verify=True)
            verified = self._verified_profile(prior.name)
            return (
                self._same_profile(restored, prior)
                and self._same_profile(active, prior)
                and self._same_profile(verified, prior)
            )
        except BaseException:
            return False

    def _activation_matches_expected(self, profile: Any) -> bool:
        try:
            expected = self._expected_profile_fingerprints.get(profile.name)
            return (
                isinstance(profile, EngineProfile)
                and expected is not None
                and profile.fingerprint == expected
            )
        except BaseException:
            return False

    @staticmethod
    def _same_profile(left: Any, right: EngineProfile) -> bool:
        try:
            return (
                isinstance(left, EngineProfile)
                and left.name == right.name
                and left.fingerprint == right.fingerprint
            )
        except BaseException:
            return False

    def _require_project_available(self, project_id: str) -> None:
        if project_id in self._unavailable_projects:
            raise RoiLaunchError(
                "project profile activation is terminally unavailable"
            ) from None

    def _require_open(self) -> None:
        if self._closed.is_set() or self._lifecycle_state != "OPEN":
            raise RoiLaunchError("ROI launch manager is closed")

    def _raise_close_failure(self) -> None:
        if self._close_failed:
            raise RoiLaunchError(
                "one or more resident engines failed to close"
            ) from None


class _ManagedResidentEngine:
    """Shared-lock proxy that can release its delegate even if callers retain it."""

    def __init__(
        self,
        *,
        engine: Any,
        profile: ResidentProfileBinding,
        verify_profile: Callable[[], EngineProfile],
        single_flight: Any,
        closed: threading.Event,
    ) -> None:
        self.profile = profile
        self._engine: Any | None = engine
        self._resolved_attestation = _snapshot_resolved_attestation(engine)
        self._transformers_version_attestation = _snapshot_text_attribute(
            engine,
            "transformers_version",
        )
        self._processor_kwargs_attestation = _snapshot_mapping_attribute(
            engine,
            "processor_kwargs",
        )
        self._runtime_attestation = _snapshot_runtime_attestation(engine)
        self._verify_profile = verify_profile
        self._single_flight = single_flight
        self._manager_closed = closed

    def infer_one(self, *args: Any, **kwargs: Any) -> Any:
        with self._single_flight:
            if self._manager_closed.is_set() or self._engine is None:
                raise RoiLaunchError("ROI launch manager is closed")
            try:
                current = self._verify_profile()
                binding_matches = (
                    build_resident_profile_binding(current) == self.profile
                )
            except BaseException:
                raise RoiLaunchError("resident profile verification failed") from None
            if not binding_matches:
                raise RoiLaunchError("resident profile verification failed")
            if self._manager_closed.is_set() or self._engine is None:
                raise RoiLaunchError("ROI launch manager is closed")
            try:
                return self._engine.infer_one(*args, **kwargs)
            except ResidentInferenceCancelled as error:
                metadata = _sanitized_cancellation_metadata(error)
                if metadata is None:
                    raise RoiLaunchError("resident engine inference failed") from None
                raise ResidentInferenceCancelled(metadata) from None
            except RuntimeContractError as error:
                if error.code == "resident.cancel_synchronize_failed":
                    self._poison_locked()
                    raise RuntimeContractError(
                        "resident cancellation synchronization failed",
                        code="resident.cancel_synchronize_failed",
                    ) from None
                raise RoiLaunchError("resident engine inference failed") from None
            except BaseException:
                raise RoiLaunchError("resident engine inference failed") from None

    @property
    def resolved(self) -> ResolvedInferConfig | None:
        return _copy_resolved_attestation(self._resolved_attestation)

    @property
    def transformers_version(self) -> str | None:
        return self._transformers_version_attestation

    @property
    def processor_kwargs(self) -> dict[str, Any] | None:
        value = self._processor_kwargs_attestation
        return None if value is None else deepcopy(value)

    @property
    def runtime(self) -> "_RuntimeAttestationSnapshot | None":
        return self._runtime_attestation

    def _close_locked(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is not None:
            _close_engine(engine)

    def _poison_locked(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is not None:
            try:
                _close_engine(engine)
            except BaseException:
                pass


@dataclass(frozen=True)
class _ArtifactIdentitySnapshot:
    payload: Mapping[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.payload)


@dataclass(frozen=True)
class _QwenAttestationSnapshot:
    processor_identity: _ArtifactIdentitySnapshot
    token_identity: _ArtifactIdentitySnapshot


@dataclass(frozen=True)
class _RuntimeAttestationSnapshot:
    model_identity: Mapping[str, Any]
    qwen: _QwenAttestationSnapshot


def _snapshot_resolved_attestation(engine: Any) -> ResolvedInferConfig | None:
    return _copy_resolved_attestation(getattr(engine, "resolved", None))


def _copy_resolved_attestation(resolved: Any) -> ResolvedInferConfig | None:
    if not isinstance(resolved, ResolvedInferConfig) or not isinstance(
        resolved.config,
        InferConfig,
    ):
        return None
    try:
        strict_config = InferConfig.model_validate(
            resolved.config.model_dump(mode="json")
        )
        config_dict = deepcopy(resolved.config_dict)
    except Exception:
        return None
    return ResolvedInferConfig(
        config=strict_config,
        config_dict=config_dict,
        fingerprint=resolved.fingerprint,
        schema_version=resolved.schema_version,
        loader_version=resolved.loader_version,
        entry_config_path=Path("/__coordexp_resident_attestation__.json"),
        sources=(),
        path_origins={},
    )


def _snapshot_text_attribute(engine: Any, name: str) -> str | None:
    value = getattr(engine, name, None)
    return value if isinstance(value, str) else None


def _snapshot_mapping_attribute(engine: Any, name: str) -> dict[str, Any] | None:
    value = getattr(engine, name, None)
    if not isinstance(value, Mapping):
        return None
    try:
        return deepcopy(dict(value))
    except Exception:
        return None


def _snapshot_runtime_attestation(engine: Any) -> _RuntimeAttestationSnapshot | None:
    runtime = getattr(engine, "runtime", None)
    qwen = getattr(runtime, "qwen", None)
    model_identity = getattr(runtime, "model_identity", None)
    processor_identity = getattr(qwen, "processor_identity", None)
    token_identity = getattr(qwen, "token_identity", None)
    if (
        not isinstance(model_identity, Mapping)
        or not callable(getattr(processor_identity, "to_artifact_dict", None))
        or not callable(getattr(token_identity, "to_artifact_dict", None))
    ):
        return None
    try:
        model_payload = _deep_freeze(deepcopy(dict(model_identity)))
        processor_payload = _deep_freeze(
            deepcopy(processor_identity.to_artifact_dict())
        )
        token_payload = _deep_freeze(deepcopy(token_identity.to_artifact_dict()))
    except Exception:
        return None
    if not all(
        isinstance(payload, Mapping)
        for payload in (model_payload, processor_payload, token_payload)
    ):
        return None
    return _RuntimeAttestationSnapshot(
        model_identity=model_payload,
        qwen=_QwenAttestationSnapshot(
            processor_identity=_ArtifactIdentitySnapshot(processor_payload),
            token_identity=_ArtifactIdentitySnapshot(token_payload),
        ),
    )


def _sanitized_cancellation_metadata(
    error: ResidentInferenceCancelled,
) -> CancellationMetadata | None:
    try:
        metadata = error.metadata
        deadline_seconds = metadata.deadline_seconds
        if (
            isinstance(deadline_seconds, bool)
            or not isinstance(deadline_seconds, (int, float))
            or not math.isfinite(deadline_seconds)
            or deadline_seconds < 0
        ):
            return None
        reason = (
            metadata.reason
            if metadata.reason
            in {
                "deadline_exceeded",
                "user_cancelled",
                "user_discarded",
                "superseded",
            }
            else "client_cancelled"
        )
        return CancellationMetadata(
            requested=bool(metadata.requested),
            reason=reason,
            observed_by_backend=bool(metadata.observed_by_backend),
            backend_started=bool(metadata.backend_started),
            cuda_synchronized=bool(metadata.cuda_synchronized),
            deadline_seconds=float(deadline_seconds),
        )
    except BaseException:
        return None


def build_resident_engine_from_profile(
    *,
    profile: EngineProfile,
    config: Mapping[str, Any],
) -> ResidentRoiInferenceEngine:
    """Default real factory; model loading occurs only when manager-selected."""

    try:
        invalid_config = not isinstance(config, Mapping) or bool(set(config))
    except BaseException:
        raise RoiLaunchConfigError(
            "build_resident_engine_from_profile accepts an empty factory config"
        ) from None
    if invalid_config:
        raise RoiLaunchConfigError(
            "build_resident_engine_from_profile accepts an empty factory config"
        )
    try:
        payload = json.loads(profile.resolved_config_json)
        roi_sidecar = payload.pop("roi_inference", None)
        if not isinstance(roi_sidecar, dict):
            raise RoiLaunchError("saved profile lacks its strict ROI sidecar")
        strict = InferConfig.model_validate(payload)
        strict_payload = strict.model_dump(mode="json")
        fingerprint = sha256_json(strict_payload)
        if fingerprint != profile.resolved_infer_config_fingerprint:
            raise ProfileDriftError(
                {
                    "resolved_config": {
                        "expected_sha256": (profile.resolved_infer_config_fingerprint),
                        "observed_sha256": fingerprint,
                    }
                }
            )
        resolved = ResolvedInferConfig(
            config=strict,
            config_dict=strict_payload,
            fingerprint=fingerprint,
            schema_version=strict.schema_version,
            loader_version=INFER_CONFIG_LOADER_VERSION,
            entry_config_path=Path("/__coordexp_saved_roi_profile__.json"),
            sources=(),
            path_origins={},
        )
        return ResidentRoiInferenceEngine.load(
            resolved=resolved,
            profile=build_resident_profile_binding(profile),
        )
    except BaseException:
        raise RoiLaunchError("resident engine factory failed") from None


def _safe_profile_projection(
    *,
    selector: str,
    profile: EngineProfile,
) -> dict[str, Any]:
    return {
        "selector": selector,
        "display_label": selector,
        "default_canvas": {
            "width": profile.default_width,
            "height": profile.default_height,
        },
        "processor_factor": profile.processor_factor,
        "bounds": {
            "min_axis_pixels": profile.min_axis_pixels,
            "max_axis_pixels": profile.max_axis_pixels,
            "max_total_pixels": profile.max_total_pixels,
        },
        "generation_deadline_seconds": profile.deadline_seconds,
    }


def _internal_profile_binding(profile: EngineProfile) -> InternalRoiProfileBinding:
    return InternalRoiProfileBinding(
        fingerprint=profile.fingerprint,
        processor_factor=profile.processor_factor,
        default_width=profile.default_width,
        default_height=profile.default_height,
        min_axis_pixels=profile.min_axis_pixels,
        max_axis_pixels=profile.max_axis_pixels,
        max_total_pixels=profile.max_total_pixels,
        deadline_seconds=float(profile.deadline_seconds),
    )


def _validate_loaded_engine(engine: Any, *, profile: EngineProfile) -> None:
    expected = build_resident_profile_binding(profile)
    if not callable(getattr(engine, "infer_one", None)):
        raise RoiLaunchError("engine factory returned no infer_one callable")
    if getattr(engine, "profile", None) != expected:
        raise ProfileDriftError(
            {
                "loaded_engine": {
                    "profile_name": profile.name,
                    "expected_sha256": expected.fingerprint,
                    "observed_sha256": getattr(
                        getattr(engine, "profile", None), "fingerprint", None
                    ),
                }
            }
        )


def _close_engine(engine: Any) -> None:
    close = getattr(engine, "close", None)
    if callable(close):
        close()
        return
    runtime_close = getattr(getattr(engine, "runtime", None), "close", None)
    if callable(runtime_close):
        runtime_close()


def _import_engine_factory(target: str) -> EngineFactory:
    if not _FACTORY_TARGET_PATTERN.fullmatch(target):
        raise RoiLaunchConfigError("invalid engine factory target")
    module_name, attribute_path = target.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    if not callable(value):
        raise RoiLaunchConfigError("engine factory target is not callable")
    return value


def _validate_saved_profile_mapping(profiles: Mapping[str, EngineProfile]) -> None:
    if type(profiles) is not dict:
        raise RoiLaunchConfigError("profile store did not return a mapping")
    for stored_name, profile in profiles.items():
        if not isinstance(stored_name, str) or type(profile) is not EngineProfile:
            raise RoiLaunchConfigError(
                "profile store keys must exactly match persisted profile names"
            )
        if stored_name != profile.name:
            raise RoiLaunchConfigError(
                "profile store keys must exactly match persisted profile names"
            )
        if not _SELECTOR_PATTERN.fullmatch(stored_name):
            raise RoiLaunchConfigError(
                "persisted profile names must be safe 1-64 character tokens"
            )


def _validate_ack_timeout(value: Any) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RoiLaunchConfigError(
            "insertion_ack_timeout_seconds must be positive or explicit null"
        )
    return float(value)


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    field: str,
) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        detail: list[str] = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        unknown_count = len(observed - expected)
        if unknown_count:
            detail.append(f"unknown_count={unknown_count}")
        raise RoiLaunchConfigError(f"{field} fields are not exact: {'; '.join(detail)}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RoiLaunchConfigError("duplicate JSON key is forbidden")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    del value
    raise RoiLaunchConfigError("non-finite JSON constant is forbidden")


def _validate_finite_json_numbers(value: Any, *, field: str) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_json_numbers(item, field=field)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_finite_json_numbers(item, field=field)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise RoiLaunchConfigError(f"{field} contains a non-finite JSON number")


def _resolve_operator_path(value: Any, *, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RoiLaunchConfigError(f"{field} must be a non-empty path string")
    try:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise RoiLaunchConfigError(f"{field} path is unavailable") from None


def _is_legacy_ipv4(value: str) -> bool:
    try:
        packed = socket.inet_aton(value)
    except OSError:
        return False
    return len(packed) == 4


def _require_loopback_endpoint(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise RoiLaunchConfigError(f"{field} must be an absolute loopback URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise RoiLaunchConfigError(f"{field} is not a valid URL") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RoiLaunchConfigError(f"{field} must be credential-free absolute HTTP(S)")
    _require_loopback_host(parsed.hostname, field=field)
    if port is not None:
        _require_port(port, field=field)


def _require_loopback_host(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RoiLaunchConfigError(f"{field} must be a loopback host")
    normalized = value.casefold()
    if normalized in {"localhost", "localhost."}:
        return "localhost"
    candidate = value[1:-1] if value.startswith("[") and value.endswith("]") else value
    if "%" in candidate:
        raise RoiLaunchConfigError(
            f"{field} loopback host must not use a zone or encoded suffix"
        )
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        if _is_legacy_ipv4(candidate):
            raise RoiLaunchConfigError(
                f"{field} must not use integer, octal, or hexadecimal IPv4 encoding"
            ) from None
        raise RoiLaunchConfigError(f"{field} must be a loopback host") from None
    if not address.is_loopback:
        raise RoiLaunchConfigError(f"{field} must be loopback-only")
    if isinstance(address, ipaddress.IPv6Address) and address != ipaddress.IPv6Address(
        "::1"
    ):
        raise RoiLaunchConfigError(f"{field} must use the canonical IPv6 loopback")
    return str(address)


def _require_port(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise RoiLaunchConfigError(f"{field} must be an integer port in 1..65535")
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


__all__ = [
    "DEFAULT_ENGINE_FACTORY_TARGET",
    "EngineFactorySpec",
    "LoopbackBind",
    "ROI_LAUNCH_SCHEMA_VERSION",
    "RoiLaunchConfig",
    "RoiLaunchConfigError",
    "RoiLaunchError",
    "RoiLaunchManager",
    "build_resident_engine_from_profile",
    "load_roi_launch_config",
]

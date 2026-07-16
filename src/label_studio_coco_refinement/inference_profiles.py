"""Persisted, content-addressed profiles for resident ROI inference.

This module deliberately stops at profile identity and persistence.  It does
not load a model, contact an endpoint, or make endpoint runtime claims.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import urlsplit, urlunsplit

from src.config.fingerprint import sha256_json
from src.config.inference import (
    INFER_CONFIG_LOADER_VERSION,
    InferConfig,
    ResolvedInferConfig,
)
from src.config.models import TemplateConfig, TemplatePromptConfig
from src.inference.prompt import normalized_prompt_policy
from src.qwen.images import QWEN_IMAGE_PROCESSOR_KWARGS


PROFILE_SCHEMA_VERSION = "coordexp-roi-engine-profile-v1"
PROFILE_STORE_SCHEMA_VERSION = "coordexp-roi-engine-profile-store-v1"
ROI_CONFIG_KEY = "roi_inference"
REQUIRED_ARTIFACT_ROLES = frozenset(
    {"base_weights", "model_config", "tokenizer", "processor"}
)
CONDITIONAL_ARTIFACT_PATHS = {
    "adapter": ("adapter", "path"),
    "embedding_delta": ("embedding_delta", "path"),
}
ALLOWED_ARTIFACT_ROLES = REQUIRED_ARTIFACT_ROLES | frozenset(CONDITIONAL_ARTIFACT_PATHS)


class ProfileContractError(ValueError):
    """Raised when a profile cannot satisfy the immutable V1 contract."""


class ProfileDriftError(ProfileContractError):
    """Raised when saved content no longer matches its recorded identity."""

    def __init__(self, mismatches: Mapping[str, Mapping[str, Any]]) -> None:
        self.mismatches = {
            str(role): dict(detail) for role, detail in sorted(mismatches.items())
        }
        super().__init__(
            "inference profile content drift: " + ", ".join(sorted(self.mismatches))
        )


def canonical_json(payload: Any) -> str:
    """Return strict deterministic JSON used by every profile fingerprint."""

    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ProfileContractError(
            "profile identity payload must be finite strict JSON"
        ) from exc


def fingerprint_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArtifactFingerprint:
    """Content identity for one configured file or directory payload."""

    role: str
    path: str
    kind: str
    sha256: str
    file_count: int
    total_bytes: int

    @classmethod
    def capture(cls, role: str, path: str | Path) -> "ArtifactFingerprint":
        role = _required_text(role, field="artifact role")
        resolved = Path(path).expanduser().resolve(strict=True)
        if resolved.is_file():
            sha256, total_bytes = _sha256_file(resolved)
            return cls(
                role=role,
                path=str(resolved),
                kind="file",
                sha256=sha256,
                file_count=1,
                total_bytes=total_bytes,
            )
        if resolved.is_dir():
            sha256, file_count, total_bytes = _sha256_directory(resolved)
            return cls(
                role=role,
                path=str(resolved),
                kind="directory",
                sha256=sha256,
                file_count=file_count,
                total_bytes=total_bytes,
            )
        raise ProfileContractError(f"artifact is not a file or directory: {resolved}")

    def verify_current(self) -> None:
        try:
            observed = type(self).capture(self.role, self.path)
        except (FileNotFoundError, ProfileContractError) as exc:
            raise ProfileDriftError(
                {self.role: {"path": self.path, "error": str(exc)}}
            ) from exc
        expected = self.to_dict()
        actual = observed.to_dict()
        if expected != actual:
            raise ProfileDriftError(
                {
                    self.role: {
                        "path": self.path,
                        "expected": expected,
                        "observed": actual,
                    }
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }

    def to_receipt_dict(self) -> dict[str, Any]:
        """Project the protected artifact identity without a local path."""

        return {
            "role": self.role,
            "kind": self.kind,
            "sha256": self.sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactFingerprint":
        artifact = cls(
            role=str(payload["role"]),
            path=str(payload["path"]),
            kind=str(payload["kind"]),
            sha256=str(payload["sha256"]),
            file_count=_non_negative_int(payload["file_count"], field="file_count"),
            total_bytes=_non_negative_int(payload["total_bytes"], field="total_bytes"),
        )
        _required_text(artifact.role, field="artifact role")
        _required_text(artifact.path, field="artifact path")
        if artifact.kind not in {"file", "directory"}:
            raise ProfileContractError(f"unsupported artifact kind: {artifact.kind}")
        if len(artifact.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in artifact.sha256
        ):
            raise ProfileContractError(
                "artifact sha256 must contain 64 hexadecimal digits"
            )
        return artifact


@dataclass(frozen=True)
class EngineProfile:
    """Complete immutable identity needed to select a resident ROI engine."""

    name: str
    endpoint: str
    artifacts: tuple[ArtifactFingerprint, ...]
    resolved_config_json: str
    resolved_infer_config_fingerprint: str
    prompt_policy_json: str
    parser_identity_json: str
    adapter_identity_json: str
    transform_identity_json: str
    transformers_version: str
    processor_kwargs_json: str
    processor_factor: int
    default_width: int
    default_height: int
    min_axis_pixels: int
    max_axis_pixels: int
    max_total_pixels: int
    deadline_seconds: float
    runtime_identity_json: str
    schema_version: str = PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.name, field="profile name")
        _validate_safe_endpoint(self.endpoint)
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ProfileContractError(
                f"unsupported profile schema version: {self.schema_version}"
            )
        roles = [artifact.role for artifact in self.artifacts]
        if not roles:
            raise ProfileContractError(
                "an engine profile must fingerprint its artifacts"
            )
        if len(roles) != len(set(roles)):
            raise ProfileContractError("artifact roles must be unique")
        if tuple(sorted(roles)) != tuple(roles):
            raise ProfileContractError("artifacts must be sorted by role")
        for value, field in (
            (self.processor_factor, "processor_factor"),
            (self.default_width, "default_width"),
            (self.default_height, "default_height"),
            (self.min_axis_pixels, "min_axis_pixels"),
            (self.max_axis_pixels, "max_axis_pixels"),
            (self.max_total_pixels, "max_total_pixels"),
        ):
            _positive_int(value, field=field)
        if self.min_axis_pixels > self.max_axis_pixels:
            raise ProfileContractError("min_axis_pixels cannot exceed max_axis_pixels")
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(self.deadline_seconds)
            or self.deadline_seconds <= 0
        ):
            raise ProfileContractError("deadline_seconds must be finite and positive")
        _required_text(self.transformers_version, field="transformers_version")
        _require_sha256(
            self.resolved_infer_config_fingerprint,
            field="resolved_infer_config_fingerprint",
        )
        for serialized, field in (
            (self.resolved_config_json, "resolved_config"),
            (self.prompt_policy_json, "prompt_policy"),
            (self.parser_identity_json, "parser_identity"),
            (self.adapter_identity_json, "adapter_identity"),
            (self.transform_identity_json, "transform_identity"),
            (self.processor_kwargs_json, "processor_kwargs"),
            (self.runtime_identity_json, "runtime_identity"),
        ):
            try:
                value = json.loads(serialized)
            except json.JSONDecodeError as exc:
                raise ProfileContractError(f"{field} is not valid JSON") from exc
            if canonical_json(value) != serialized:
                raise ProfileContractError(f"{field} JSON is not canonical")
        identity_payloads = {
            "resolved_config": json.loads(self.resolved_config_json),
            "prompt_policy": json.loads(self.prompt_policy_json),
            "parser_identity": json.loads(self.parser_identity_json),
            "adapter_identity": json.loads(self.adapter_identity_json),
            "transform_identity": json.loads(self.transform_identity_json),
            "processor_kwargs": json.loads(self.processor_kwargs_json),
            "runtime_identity": json.loads(self.runtime_identity_json),
        }
        kwargs = identity_payloads["processor_kwargs"]
        if kwargs != dict(QWEN_IMAGE_PROCESSOR_KWARGS):
            raise ProfileContractError(
                "forced processor kwargs must exactly match the executed Qwen image processor"
            )
        for field, value in identity_payloads.items():
            _reject_credentials(value, field=field)
        resolved_config = _mapping(
            identity_payloads["resolved_config"], field="resolved_config"
        )
        strict_config = _validate_resolved_config_identity(
            resolved_config,
            expected_fingerprint=self.resolved_infer_config_fingerprint,
        )
        expected_prompt_policy = normalized_prompt_policy(
            _template_config(strict_config)
        )
        if identity_payloads["prompt_policy"] != expected_prompt_policy:
            raise ProfileContractError(
                "prompt policy does not match the normalized executed InferConfig template"
            )
        _validate_artifact_role_schema(self.artifacts, resolved_config=resolved_config)
        derived = _derive_profile_settings(
            self.artifacts,
            resolved_config=resolved_config,
            processor_kwargs=kwargs,
        )
        recorded = {
            "processor_factor": self.processor_factor,
            "default_width": self.default_width,
            "default_height": self.default_height,
            "min_axis_pixels": self.min_axis_pixels,
            "max_axis_pixels": self.max_axis_pixels,
            "max_total_pixels": self.max_total_pixels,
            "deadline_seconds": self.deadline_seconds,
        }
        if recorded != derived:
            raise ProfileContractError(
                "recorded processor constraints do not match captured processor/config content"
            )
        self.validate_canvas(self.default_width, self.default_height)

    @classmethod
    def capture(
        cls,
        *,
        name: str,
        endpoint: str,
        artifact_paths: Mapping[str, str | Path],
        resolved_config: ResolvedInferConfig,
        roi_inference: Mapping[str, Any],
        parser_identity: Any,
        adapter_identity: Any,
        transform_identity: Any,
        transformers_version: str,
        processor_kwargs: Mapping[str, Any],
        runtime_identity: Any | None = None,
    ) -> "EngineProfile":
        strict_config = _strict_resolved_config_payload(resolved_config)
        roi_inference = _mapping(roi_inference, field=ROI_CONFIG_KEY)
        captured_config = {**strict_config, ROI_CONFIG_KEY: dict(roi_inference)}
        _validate_declared_artifact_roles(
            artifact_paths,
            resolved_config=captured_config,
        )
        artifacts = tuple(
            ArtifactFingerprint.capture(role, path)
            for role, path in sorted(artifact_paths.items())
        )
        derived = _derive_profile_settings(
            artifacts,
            resolved_config=captured_config,
            processor_kwargs=processor_kwargs,
        )
        return cls(
            name=name,
            endpoint=endpoint,
            artifacts=artifacts,
            resolved_config_json=canonical_json(captured_config),
            resolved_infer_config_fingerprint=resolved_config.fingerprint,
            prompt_policy_json=canonical_json(
                normalized_prompt_policy(_template_config(resolved_config.config))
            ),
            parser_identity_json=canonical_json(parser_identity),
            adapter_identity_json=canonical_json(adapter_identity),
            transform_identity_json=canonical_json(transform_identity),
            transformers_version=_required_text(
                transformers_version, field="transformers_version"
            ),
            processor_kwargs_json=canonical_json(dict(processor_kwargs)),
            processor_factor=derived["processor_factor"],
            default_width=derived["default_width"],
            default_height=derived["default_height"],
            min_axis_pixels=derived["min_axis_pixels"],
            max_axis_pixels=derived["max_axis_pixels"],
            max_total_pixels=derived["max_total_pixels"],
            deadline_seconds=derived["deadline_seconds"],
            runtime_identity_json=canonical_json(runtime_identity or {}),
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.identity_payload())

    @property
    def identity_fingerprints(self) -> dict[str, str]:
        return {
            "resolved_config": fingerprint_json(json.loads(self.resolved_config_json)),
            "prompt_policy": fingerprint_json(json.loads(self.prompt_policy_json)),
            "parser": fingerprint_json(json.loads(self.parser_identity_json)),
            "adapter": fingerprint_json(json.loads(self.adapter_identity_json)),
            "transform": fingerprint_json(json.loads(self.transform_identity_json)),
            "transformers": fingerprint_json({"version": self.transformers_version}),
            "processor_kwargs": fingerprint_json(
                json.loads(self.processor_kwargs_json)
            ),
            "runtime": fingerprint_json(json.loads(self.runtime_identity_json)),
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "endpoint": self.endpoint,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "resolved_config": json.loads(self.resolved_config_json),
            "resolved_infer_config_fingerprint": self.resolved_infer_config_fingerprint,
            "prompt_policy": json.loads(self.prompt_policy_json),
            "parser_identity": json.loads(self.parser_identity_json),
            "adapter_identity": json.loads(self.adapter_identity_json),
            "transform_identity": json.loads(self.transform_identity_json),
            "transformers_version": self.transformers_version,
            "processor_kwargs": json.loads(self.processor_kwargs_json),
            "processor_factor": self.processor_factor,
            "default_canvas": [self.default_width, self.default_height],
            "axis_bounds": [self.min_axis_pixels, self.max_axis_pixels],
            "max_total_pixels": self.max_total_pixels,
            "deadline_seconds": self.deadline_seconds,
            "runtime_identity": json.loads(self.runtime_identity_json),
            "identity_fingerprints": self.identity_fingerprints,
        }

    def validate_canvas(self, width: int, height: int) -> tuple[int, int]:
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
        ):
            raise ProfileContractError("canvas dimensions must be integers")
        if not (
            self.min_axis_pixels <= width <= self.max_axis_pixels
            and self.min_axis_pixels <= height <= self.max_axis_pixels
        ):
            raise ProfileContractError(
                "canvas dimensions are outside the profile axis bounds"
            )
        if width % self.processor_factor or height % self.processor_factor:
            raise ProfileContractError(
                "canvas dimensions must be divisible by the processor-derived factor"
            )
        if width * height > self.max_total_pixels:
            raise ProfileContractError("canvas exceeds the profile total-pixel bound")
        return width, height

    def verify_artifacts(self) -> None:
        mismatches: dict[str, Mapping[str, Any]] = {}
        for artifact in self.artifacts:
            try:
                artifact.verify_current()
            except ProfileDriftError as exc:
                mismatches.update(exc.mismatches)
        if mismatches:
            raise ProfileDriftError(mismatches)

    def verify_runtime_identity(self, observed: Any) -> None:
        expected = json.loads(self.runtime_identity_json)
        if fingerprint_json(observed) != fingerprint_json(expected):
            raise ProfileDriftError(
                {
                    "runtime_identity": {
                        "expected_sha256": fingerprint_json(expected),
                        "observed_sha256": fingerprint_json(observed),
                    }
                }
            )

    def to_receipt_dict(self) -> dict[str, Any]:
        """Return a credential-free allowlisted receipt, never raw config content."""

        return {
            "schema_version": self.schema_version,
            "profile_name": self.name,
            "profile_fingerprint": self.fingerprint,
            "endpoint": _receipt_endpoint(self.endpoint),
            "artifacts": [artifact.to_receipt_dict() for artifact in self.artifacts],
            "identity_fingerprints": self.identity_fingerprints,
            "processor": {
                "factor": self.processor_factor,
                "default_canvas": [self.default_width, self.default_height],
                "axis_bounds": [self.min_axis_pixels, self.max_axis_pixels],
                "max_total_pixels": self.max_total_pixels,
                "do_resize": False,
            },
            "deadline_seconds": self.deadline_seconds,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "profile_fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EngineProfile":
        default_canvas = payload["default_canvas"]
        axis_bounds = payload["axis_bounds"]
        profile = cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            endpoint=str(payload["endpoint"]),
            artifacts=tuple(
                ArtifactFingerprint.from_dict(item) for item in payload["artifacts"]
            ),
            resolved_config_json=canonical_json(payload["resolved_config"]),
            resolved_infer_config_fingerprint=str(
                payload["resolved_infer_config_fingerprint"]
            ),
            prompt_policy_json=canonical_json(payload["prompt_policy"]),
            parser_identity_json=canonical_json(payload["parser_identity"]),
            adapter_identity_json=canonical_json(payload["adapter_identity"]),
            transform_identity_json=canonical_json(payload["transform_identity"]),
            transformers_version=str(payload["transformers_version"]),
            processor_kwargs_json=canonical_json(payload["processor_kwargs"]),
            processor_factor=payload["processor_factor"],
            default_width=default_canvas[0],
            default_height=default_canvas[1],
            min_axis_pixels=axis_bounds[0],
            max_axis_pixels=axis_bounds[1],
            max_total_pixels=payload["max_total_pixels"],
            deadline_seconds=payload["deadline_seconds"],
            runtime_identity_json=canonical_json(payload["runtime_identity"]),
        )
        recorded = payload.get("profile_fingerprint")
        if recorded is not None and recorded != profile.fingerprint:
            raise ProfileContractError(
                "persisted profile fingerprint does not match payload"
            )
        identity_fingerprints = payload.get("identity_fingerprints")
        if (
            identity_fingerprints is not None
            and identity_fingerprints != profile.identity_fingerprints
        ):
            raise ProfileContractError(
                "persisted component identity fingerprints do not match payload"
            )
        return profile


class EngineProfileStore:
    """Small JSON store with exactly one active profile binding per project."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    def profiles(self) -> dict[str, EngineProfile]:
        payload = self._read()
        return {
            name: EngineProfile.from_dict(profile)
            for name, profile in payload["profiles"].items()
        }

    def save(self, profile: EngineProfile) -> None:
        with self._exclusive_lock():
            payload = self._read()
            existing = payload["profiles"].get(profile.name)
            if existing is not None:
                persisted = EngineProfile.from_dict(existing)
                if persisted.fingerprint != profile.fingerprint:
                    raise ProfileContractError(
                        "a different immutable profile already uses this name"
                    )
                _fsync_directory(self.path.parent)
                return
            payload["profiles"][profile.name] = profile.to_dict()
            self._write(payload)

    def activate(self, project_id: str, profile_name: str) -> EngineProfile:
        project_id = _required_text(project_id, field="project_id")
        with self._exclusive_lock():
            payload = self._read()
            raw_profile = payload["profiles"].get(profile_name)
            if raw_profile is None:
                raise ProfileContractError(f"unknown inference profile: {profile_name}")
            _verify_persisted_artifacts(raw_profile)
            profile = EngineProfile.from_dict(raw_profile)
            payload["active_by_project"][project_id] = profile.name
            self._write(payload)
            return profile

    def active(self, project_id: str, *, verify: bool = True) -> EngineProfile:
        payload = self._read()
        profile_name = payload["active_by_project"].get(project_id)
        if profile_name is None:
            raise ProfileContractError(f"project has no active profile: {project_id}")
        raw_profile = payload["profiles"][profile_name]
        if verify:
            _verify_persisted_artifacts(raw_profile)
        profile = EngineProfile.from_dict(raw_profile)
        return profile

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": PROFILE_STORE_SCHEMA_VERSION,
                "profiles": {},
                "active_by_project": {},
            }
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileContractError(
                f"cannot read profile store: {self.path}"
            ) from exc
        if payload.get("schema_version") != PROFILE_STORE_SCHEMA_VERSION:
            raise ProfileContractError("unsupported profile-store schema version")
        if not isinstance(payload.get("profiles"), dict) or not isinstance(
            payload.get("active_by_project"), dict
        ):
            raise ProfileContractError("profile store has invalid mappings")
        return payload

    def _write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        encoded = canonical_json(payload) + "\n"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = self.lock_path.open("a+b")
        except OSError:
            raise ProfileContractError("profile store lock is unavailable") from None
        with handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except OSError:
                raise ProfileContractError(
                    "profile store lock is unavailable"
                ) from None
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_persisted_artifacts(payload: Mapping[str, Any]) -> None:
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ProfileContractError("persisted profile artifacts must be a list")
    mismatches: dict[str, Mapping[str, Any]] = {}
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, Mapping):
            raise ProfileContractError("persisted profile artifact must be a mapping")
        artifact = ArtifactFingerprint.from_dict(raw_artifact)
        try:
            artifact.verify_current()
        except ProfileDriftError as exc:
            mismatches.update(exc.mismatches)
    if mismatches:
        raise ProfileDriftError(mismatches)


def _strict_resolved_config_payload(resolved: ResolvedInferConfig) -> dict[str, Any]:
    if not isinstance(resolved, ResolvedInferConfig) or not isinstance(
        resolved.config, InferConfig
    ):
        raise ProfileContractError(
            "resolved_config must be a real ResolvedInferConfig with strict InferConfig"
        )
    strict_payload = resolved.config.model_dump(mode="json")
    if (
        resolved.config_dict != strict_payload
        or resolved.fingerprint != sha256_json(strict_payload)
        or resolved.schema_version != resolved.config.schema_version
        or resolved.loader_version != INFER_CONFIG_LOADER_VERSION
    ):
        raise ProfileContractError(
            "ResolvedInferConfig payload, defaults, loader, or fingerprint are inconsistent"
        )
    return strict_payload


def _validate_resolved_config_identity(
    resolved_config: Mapping[str, Any],
    *,
    expected_fingerprint: str,
) -> InferConfig:
    combined = dict(resolved_config)
    roi = combined.pop(ROI_CONFIG_KEY, None)
    if not isinstance(roi, dict):
        raise ProfileContractError(
            "captured resolved config is missing the explicit roi_inference sidecar"
        )
    try:
        strict_config = InferConfig.model_validate(combined)
    except Exception as exc:
        raise ProfileContractError(
            "captured resolved config is not a strict InferConfig"
        ) from exc
    canonical_payload = strict_config.model_dump(mode="json")
    if canonical_payload != combined:
        raise ProfileContractError(
            "captured resolved config omitted defaults or is not canonical"
        )
    if sha256_json(canonical_payload) != expected_fingerprint:
        raise ProfileContractError(
            "captured resolved config fingerprint does not match its strict payload"
        )
    return strict_config


def _template_config(config: InferConfig) -> TemplateConfig:
    return TemplateConfig(
        object_field_order=config.template.object_field_order,
        object_ordering=config.template.object_ordering,
        assistant_format=config.template.assistant_format,
        prompt=TemplatePromptConfig(
            system=config.template.prompt.system,
            user=config.template.prompt.user,
        ),
    )


def _validate_declared_artifact_roles(
    artifact_paths: Mapping[str, str | Path],
    *,
    resolved_config: Mapping[str, Any],
) -> None:
    if not isinstance(artifact_paths, Mapping):
        raise ProfileContractError("artifact_paths must be a role-to-path mapping")
    declared_roles = set(artifact_paths)
    if not all(isinstance(role, str) for role in declared_roles):
        raise ProfileContractError("artifact roles must be text")
    unknown = declared_roles - ALLOWED_ARTIFACT_ROLES
    if unknown:
        raise ProfileContractError(
            f"unknown or mislabeled artifact roles: {sorted(unknown)}"
        )
    expected = set(REQUIRED_ARTIFACT_ROLES)
    for role, path_fields in CONDITIONAL_ARTIFACT_PATHS.items():
        configured = _optional_configured_path(resolved_config, path_fields)
        if configured is not None:
            expected.add(role)
    missing = expected - declared_roles
    if missing:
        raise ProfileContractError(
            f"missing required artifact roles: {sorted(missing)}"
        )
    unexpected = declared_roles - expected
    if unexpected:
        raise ProfileContractError(
            "conditional artifact roles are present without matching resolved config: "
            f"{sorted(unexpected)}"
        )

    base_model = _required_nested_path(resolved_config, ("model", "base_model"))
    _require_same_artifact_path(
        role="base_weights",
        declared=artifact_paths["base_weights"],
        configured=base_model,
    )
    for role, path_fields in CONDITIONAL_ARTIFACT_PATHS.items():
        configured = _optional_configured_path(resolved_config, path_fields)
        if configured is not None:
            _require_same_artifact_path(
                role=role,
                declared=artifact_paths[role],
                configured=configured,
            )


def _validate_artifact_role_schema(
    artifacts: tuple[ArtifactFingerprint, ...],
    *,
    resolved_config: Mapping[str, Any],
) -> None:
    paths = {artifact.role: artifact.path for artifact in artifacts}
    _validate_declared_artifact_roles(paths, resolved_config=resolved_config)
    by_role = {artifact.role: artifact for artifact in artifacts}
    _validate_weight_artifact(by_role["base_weights"])
    _validate_tokenizer_artifact(by_role["tokenizer"])
    _load_json_artifact(by_role["model_config"], filename="config.json")
    _load_json_artifact(by_role["processor"], filename="preprocessor_config.json")


def _derive_profile_settings(
    artifacts: tuple[ArtifactFingerprint, ...],
    *,
    resolved_config: Mapping[str, Any],
    processor_kwargs: Mapping[str, Any],
) -> dict[str, int | float]:
    _validate_artifact_role_schema(artifacts, resolved_config=resolved_config)
    by_role = {artifact.role: artifact for artifact in artifacts}
    processor = _load_json_artifact(
        by_role["processor"], filename="preprocessor_config.json"
    )
    model_config = _load_json_artifact(by_role["model_config"], filename="config.json")
    patch_size = _positive_int(
        processor.get("patch_size"), field="processor.patch_size"
    )
    merge_size = _positive_int(
        processor.get("merge_size"), field="processor.merge_size"
    )
    vision = _mapping(
        model_config.get("vision_config"), field="model_config.vision_config"
    )
    model_patch = _positive_int(
        vision.get("patch_size"), field="model_config.vision_config.patch_size"
    )
    model_merge = _positive_int(
        vision.get("spatial_merge_size"),
        field="model_config.vision_config.spatial_merge_size",
    )
    if (patch_size, merge_size) != (model_patch, model_merge):
        raise ProfileContractError(
            "processor patch/merge identity does not match captured model config"
        )
    factor = patch_size * merge_size
    roi = _mapping(resolved_config.get(ROI_CONFIG_KEY), field=ROI_CONFIG_KEY)
    required_roi_fields = {
        "processor_factor",
        "default_width",
        "default_height",
        "min_axis_pixels",
        "max_axis_pixels",
        "max_total_pixels",
        "deadline_seconds",
    }
    if set(roi) != required_roi_fields:
        raise ProfileContractError(
            f"resolved {ROI_CONFIG_KEY} fields must be exact: {sorted(required_roi_fields)}"
        )
    configured_factor = _positive_int(
        roi["processor_factor"], field=f"{ROI_CONFIG_KEY}.processor_factor"
    )
    if configured_factor != factor:
        raise ProfileContractError(
            "resolved processor_factor does not match processor patch_size * merge_size"
        )
    settings: dict[str, int | float] = {
        "processor_factor": factor,
        "default_width": _positive_int(
            roi["default_width"], field=f"{ROI_CONFIG_KEY}.default_width"
        ),
        "default_height": _positive_int(
            roi["default_height"], field=f"{ROI_CONFIG_KEY}.default_height"
        ),
        "min_axis_pixels": _positive_int(
            roi["min_axis_pixels"], field=f"{ROI_CONFIG_KEY}.min_axis_pixels"
        ),
        "max_axis_pixels": _positive_int(
            roi["max_axis_pixels"], field=f"{ROI_CONFIG_KEY}.max_axis_pixels"
        ),
        "max_total_pixels": _positive_int(
            roi["max_total_pixels"], field=f"{ROI_CONFIG_KEY}.max_total_pixels"
        ),
        "deadline_seconds": _positive_finite_number(
            roi["deadline_seconds"], field=f"{ROI_CONFIG_KEY}.deadline_seconds"
        ),
    }
    if settings["min_axis_pixels"] > settings["max_axis_pixels"]:
        raise ProfileContractError("resolved ROI minimum axis exceeds maximum axis")
    for field in ("default_width", "default_height"):
        value = settings[field]
        if not isinstance(value, int):  # pragma: no cover - helper guarantees this.
            raise ProfileContractError(f"{field} must be an integer")
        if not settings["min_axis_pixels"] <= value <= settings["max_axis_pixels"]:
            raise ProfileContractError(f"resolved {field} is outside ROI axis bounds")
        if value % factor:
            raise ProfileContractError(
                f"resolved {field} is not divisible by the processor-derived factor"
            )
    if (
        settings["default_width"] * settings["default_height"]
        > settings["max_total_pixels"]
    ):
        raise ProfileContractError("resolved default canvas exceeds total-pixel bound")
    if processor_kwargs.get("do_resize") is not False:
        raise ProfileContractError("processor kwargs must force do_resize=false")
    model = _mapping(resolved_config.get("model"), field="resolved_config.model")
    processor_config = _mapping(
        model.get("processor"), field="resolved_config.model.processor"
    )
    if processor_config.get("do_resize") is not False:
        raise ProfileContractError(
            "resolved infer config must set model.processor.do_resize=false"
        )
    return settings


def _load_json_artifact(
    artifact: ArtifactFingerprint,
    *,
    filename: str,
) -> Mapping[str, Any]:
    path = Path(artifact.path)
    json_path = path / filename if artifact.kind == "directory" else path
    if not json_path.is_file():
        raise ProfileContractError(
            f"artifact role {artifact.role} does not contain required {filename}"
        )
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileContractError(
            f"artifact role {artifact.role} does not contain valid JSON"
        ) from exc
    return _mapping(payload, field=f"artifact.{artifact.role}")


def _validate_weight_artifact(artifact: ArtifactFingerprint) -> None:
    path = Path(artifact.path)
    candidates = [path] if artifact.kind == "file" else list(path.rglob("*"))
    if not any(
        candidate.is_file()
        and (
            candidate.suffix in {".bin", ".safetensors", ".pt", ".pth"}
            or candidate.name.endswith(".safetensors.index.json")
        )
        for candidate in candidates
    ):
        raise ProfileContractError(
            "base_weights artifact contains no recognized model-weight payload"
        )


def _validate_tokenizer_artifact(artifact: ArtifactFingerprint) -> None:
    path = Path(artifact.path)
    recognized = {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "spiece.model",
    }
    candidates = (
        {path.name}
        if artifact.kind == "file"
        else {item.name for item in path.rglob("*") if item.is_file()}
    )
    if not candidates & recognized:
        raise ProfileContractError(
            "tokenizer artifact contains no recognized tokenizer payload"
        )


def _optional_configured_path(
    payload: Mapping[str, Any], fields: tuple[str, str]
) -> str | None:
    section = payload.get(fields[0])
    if section is None:
        return None
    section = _mapping(section, field=f"resolved_config.{fields[0]}")
    value = section.get(fields[1])
    if not isinstance(value, str) or not value:
        raise ProfileContractError(
            f"resolved_config.{fields[0]}.{fields[1]} must be a non-empty path"
        )
    return value


def _required_nested_path(payload: Mapping[str, Any], fields: tuple[str, str]) -> str:
    section = _mapping(payload.get(fields[0]), field=f"resolved_config.{fields[0]}")
    value = section.get(fields[1])
    if not isinstance(value, str) or not value:
        raise ProfileContractError(
            f"resolved_config.{fields[0]}.{fields[1]} must be a non-empty path"
        )
    return value


def _require_same_artifact_path(
    *,
    role: str,
    declared: str | Path,
    configured: str | Path,
) -> None:
    declared_path = Path(declared).expanduser().resolve(strict=True)
    configured_path = Path(configured).expanduser().resolve(strict=True)
    if declared_path != configured_path:
        raise ProfileContractError(
            f"artifact role {role} path does not match resolved config"
        )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            total_bytes += len(chunk)
    return digest.hexdigest(), total_bytes


def _sha256_directory(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    entries = sorted(
        path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()
    )
    for entry in entries:
        relative = entry.relative_to(path).as_posix()
        if entry.is_symlink() and entry.is_dir():
            raise ProfileContractError(
                f"directory artifact contains unsupported directory symlink: {entry}"
            )
        if entry.is_dir():
            digest.update(b"D\0" + relative.encode("utf-8") + b"\0")
            continue
        if not entry.is_file():
            raise ProfileContractError(f"unsupported artifact entry: {entry}")
        file_sha, size = _sha256_file(entry)
        digest.update(
            b"F\0"
            + relative.encode("utf-8")
            + b"\0"
            + str(size).encode("ascii")
            + b"\0"
            + file_sha.encode("ascii")
            + b"\0"
        )
        file_count += 1
        total_bytes += size
    return digest.hexdigest(), file_count, total_bytes


def _required_text(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileContractError(f"{field} must be non-empty text")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProfileContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProfileContractError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProfileContractError(f"{field} must be a non-negative integer")
    return value


def _positive_finite_number(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ProfileContractError(f"{field} must be finite and positive")
    return float(value)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileContractError(f"{field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ProfileContractError(f"{field} keys must be text")
    return value


def _validate_safe_endpoint(endpoint: str) -> None:
    endpoint = _required_text(endpoint, field="endpoint")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProfileContractError("endpoint must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ProfileContractError(
            "endpoint must not contain credentials, query, or fragment"
        )


def _receipt_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


_CREDENTIAL_KEYS = {
    "authorization",
    "proxy_authorization",
    "cookie",
    "cookies",
    "set_cookie",
    "headers",
    "http_headers",
    "request_headers",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "auth_token",
    "bearer_token",
    "refresh_token",
    "credential",
    "credentials",
    "provider_credentials",
    "private_key",
    "private_key_path",
    "aws_access_key_id",
    "aws_secret_access_key",
    "access_key",
}


def _reject_credentials(payload: Any, *, field: str) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).casefold().replace("-", "_")
            if (
                normalized in _CREDENTIAL_KEYS
                or "authorization" in normalized
                or "private_key" in normalized
                or "credential" in normalized
                or normalized.endswith(
                    (
                        "_password",
                        "_secret",
                        "_token",
                        "_api_key",
                        "_headers",
                        "_cookie",
                        "_cookies",
                    )
                )
            ):
                raise ProfileContractError(
                    f"{field} contains credential-bearing field: {key}"
                )
            _reject_credentials(value, field=field)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _reject_credentials(value, field=field)
    elif isinstance(payload, str):
        stripped = payload.strip()
        upper = stripped.upper()
        if (
            "-----BEGIN PRIVATE KEY-----" in upper
            or "-----BEGIN RSA PRIVATE KEY-----" in upper
            or "-----BEGIN OPENSSH PRIVATE KEY-----" in upper
            or upper.startswith("BEARER ")
            or upper.startswith("BASIC ")
            or stripped.startswith(("sk-", "sk_proj-", "sk-proj-", "AKIA"))
        ):
            raise ProfileContractError(f"{field} contains credential-bearing content")

"""Additive reconciliation of the two frozen Human-13 execution surfaces.

The training owner is BF16/FlashAttention-2 on GPU 0 while the established
behavioral audit is fp32/SDPA on GPU 1.  This module does not replace either
surface and does not relax the Source witness contract.  It only records a
typed admission attempt when the immutable identities and the existing
teacher-forced checker agree exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from src.artifacts.json_values import json_sha256


SCHEMA_VERSION = "human13_source_surface_reconciliation.v1"
SOURCE_SURFACE = "gpu1:fp32/sdpa/batch1"
TRAINING_SURFACE = "gpu0:bfloat16/flash_attention_2"
_EXPECTED_AUDIT_RPS = (1.0, 1.1)


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 digest") from error
    return value


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _runtime_payload(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): item for key, item in value.items()}


def _runtime_contract(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "backend": value.get("backend"),
        "batch_size": value.get("batch_size"),
        "observed_model_dtype_names": value.get("observed_model_dtype_names"),
        "observed_attn_implementation": value.get(
            "observed_attn_implementation"
        ),
    }


def _runtime_identity_sha256(
    values: tuple[Mapping[str, object], ...],
) -> str:
    payload = [_runtime_payload(item) for item in values]
    try:
        return json_sha256(payload)
    except Exception:
        return json_sha256(
            [
                {
                    "unserializable_type": type(item).__name__,
                    "repr": repr(item),
                }
                for item in values
            ]
        )


@dataclass(frozen=True)
class SourceSurfaceReconciliationRequest:
    """All immutable evidence needed for one surface-admission attempt."""

    training_identity: object
    source_runtime_identities: tuple[Mapping[str, object], ...]
    source_checkpoint_payload_sha256s: tuple[str, ...]
    source_checkpoint_paths: tuple[str, ...]
    training_checkpoint_path: str
    source_adapter_sha256s: tuple[str, ...]
    source_embedding_delta_sha256s: tuple[str, ...]
    source_base_model_paths: tuple[str, ...]
    training_base_model_path: str
    source_manifest_sha256s: tuple[str, ...]
    manifest_image_sha256: str
    source_tokenizer_sha256s: tuple[str, ...]
    source_prompt_sha256s: tuple[str, ...]
    source_image_sha256s: tuple[str, ...]
    manifest_sha256: str
    image_id: int
    source_audit_sha256s: tuple[tuple[float, str], ...]
    decodes: tuple[object, ...]
    check: Callable[[], int]

    def __post_init__(self) -> None:
        if not callable(self.check):
            raise TypeError("surface reconciliation requires a checker")
        if self.image_id != 1584:
            raise ValueError("surface reconciliation is scoped to image 1584")
        _digest(self.manifest_sha256, field="manifest_sha256")
        _digest(self.manifest_image_sha256, field="manifest_image_sha256")
        runtimes = tuple(self.source_runtime_identities)
        if len(runtimes) != 2 or any(not isinstance(item, Mapping) for item in runtimes):
            raise ValueError("surface reconciliation requires two Source runtimes")
        object.__setattr__(self, "source_runtime_identities", runtimes)
        for name in (
            "source_checkpoint_payload_sha256s",
            "source_tokenizer_sha256s",
            "source_prompt_sha256s",
            "source_image_sha256s",
            "source_adapter_sha256s",
            "source_embedding_delta_sha256s",
        ):
            values = tuple(getattr(self, name))
            if len(values) != 2:
                raise ValueError(f"{name} must contain both Source audits")
            object.__setattr__(self, name, values)
        raw_checkpoint_paths = tuple(self.source_checkpoint_paths)
        if (
            len(raw_checkpoint_paths) != 2
            or any(
                not isinstance(item, str) or not item or item == "None"
                for item in raw_checkpoint_paths
            )
        ):
            raise ValueError("source checkpoint paths must bind both Source audits")
        paths = tuple(
            str(Path(item).expanduser().resolve(strict=False))
            for item in raw_checkpoint_paths
        )
        if (
            not isinstance(self.training_checkpoint_path, str)
            or not self.training_checkpoint_path
            or self.training_checkpoint_path == "None"
        ):
            raise ValueError("training checkpoint path is required")
        training_path = str(
            Path(self.training_checkpoint_path).expanduser().resolve(strict=False)
        )
        if training_path == "None":
            raise ValueError("training checkpoint path is required")
        object.__setattr__(self, "source_checkpoint_paths", paths)
        object.__setattr__(self, "training_checkpoint_path", training_path)
        raw_base_paths = tuple(self.source_base_model_paths)
        if (
            len(raw_base_paths) != 2
            or any(
                not isinstance(item, str) or not item or item == "None"
                for item in raw_base_paths
            )
        ):
            raise ValueError("source base-model paths must bind both Source audits")
        base_paths = tuple(
            str(Path(item).expanduser().resolve(strict=False))
            for item in raw_base_paths
        )
        if (
            not isinstance(self.training_base_model_path, str)
            or not self.training_base_model_path
        ):
            raise ValueError("training base-model path is required")
        object.__setattr__(self, "source_base_model_paths", base_paths)
        object.__setattr__(
            self,
            "training_base_model_path",
            str(Path(self.training_base_model_path).expanduser().resolve(strict=False)),
        )
        manifest_hashes = tuple(self.source_manifest_sha256s)
        if len(manifest_hashes) != 2:
            raise ValueError("source manifest hashes must bind both Source audits")
        for digest in manifest_hashes:
            _digest(digest, field="source_manifest_sha256")
        object.__setattr__(self, "source_manifest_sha256s", manifest_hashes)
        audits = tuple(
            (float(rp), str(digest))
            for rp, digest in self.source_audit_sha256s
            if not isinstance(rp, bool)
        )
        if tuple(sorted(rp for rp, _digest_value in audits)) != _EXPECTED_AUDIT_RPS:
            raise ValueError("surface reconciliation requires RP 1.0 and RP 1.10 audits")
        for _rp, digest in audits:
            _digest(digest, field="source_audit_sha256")
        object.__setattr__(self, "source_audit_sha256s", audits)
        decodes = tuple(self.decodes)
        if len(decodes) != 2:
            raise ValueError("surface reconciliation requires two Source decodes")
        object.__setattr__(self, "decodes", decodes)


@dataclass(frozen=True)
class SourceSurfaceReconciliationReceipt:
    """Content-addressed result of an exact two-surface reconciliation."""

    admitted: bool
    source_surface: str
    training_surface: str
    training_model_object_id: int | None
    training_checkpoint_payload_sha256: str | None
    source_checkpoint_payload_sha256s: tuple[str, ...]
    source_checkpoint_paths: tuple[str, ...]
    training_checkpoint_path: str
    source_adapter_sha256s: tuple[str, ...]
    source_embedding_delta_sha256s: tuple[str, ...]
    source_base_model_paths: tuple[str, ...]
    training_base_model_path: str
    source_manifest_sha256s: tuple[str, ...]
    manifest_image_sha256: str
    training_image_sha256: str | None
    source_image_sha256s: tuple[str, ...]
    training_parameter_state_sha256: str | None
    manifest_sha256: str
    image_id: int
    source_audit_sha256s: tuple[tuple[float, str], ...]
    source_runtime_identity_sha256: str
    checked_decode_count: int
    checked_token_count: int
    mismatch_count: int
    failure_reason: str | None

    @property
    def content_sha256(self) -> str:
        return json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "admitted": self.admitted,
            "source_surface": self.source_surface,
            "training_surface": self.training_surface,
            "training_model_object_id": self.training_model_object_id,
            "training_checkpoint_payload_sha256": self.training_checkpoint_payload_sha256,
            "source_checkpoint_payload_sha256s": list(
                self.source_checkpoint_payload_sha256s
            ),
            "source_checkpoint_paths": list(self.source_checkpoint_paths),
            "training_checkpoint_path": self.training_checkpoint_path,
            "source_adapter_sha256s": list(self.source_adapter_sha256s),
            "source_embedding_delta_sha256s": list(
                self.source_embedding_delta_sha256s
            ),
            "source_base_model_paths": list(self.source_base_model_paths),
            "training_base_model_path": self.training_base_model_path,
            "source_manifest_sha256s": list(self.source_manifest_sha256s),
            "manifest_image_sha256": self.manifest_image_sha256,
            "training_image_sha256": self.training_image_sha256,
            "source_image_sha256s": list(self.source_image_sha256s),
            "training_parameter_state_sha256": self.training_parameter_state_sha256,
            "manifest_sha256": self.manifest_sha256,
            "image_id": self.image_id,
            "source_audit_sha256s": [list(item) for item in self.source_audit_sha256s],
            "source_runtime_identity_sha256": self.source_runtime_identity_sha256,
            "checked_decode_count": self.checked_decode_count,
            "checked_token_count": self.checked_token_count,
            "mismatch_count": self.mismatch_count,
            "failure_reason": self.failure_reason,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["content_sha256"] = self.content_sha256
        return payload


def _receipt(
    request: SourceSurfaceReconciliationRequest,
    *,
    mismatch_count: int,
    failure_reason: str | None,
    admitted: bool,
) -> SourceSurfaceReconciliationReceipt:
    identity = request.training_identity
    parameter_state = _field(identity, "parameter_state_sha256")
    model_object_id = _field(identity, "model_object_id")
    checkpoint_payload = _field(identity, "checkpoint_payload_sha256")
    training_image = _field(identity, "image_sha256")
    return SourceSurfaceReconciliationReceipt(
        admitted=admitted,
        source_surface=SOURCE_SURFACE,
        training_surface=TRAINING_SURFACE,
        training_model_object_id=(
            model_object_id if isinstance(model_object_id, int) else None
        ),
        training_checkpoint_payload_sha256=(
            checkpoint_payload if isinstance(checkpoint_payload, str) else None
        ),
        source_checkpoint_payload_sha256s=request.source_checkpoint_payload_sha256s,
        source_checkpoint_paths=request.source_checkpoint_paths,
        training_checkpoint_path=request.training_checkpoint_path,
        source_adapter_sha256s=request.source_adapter_sha256s,
        source_embedding_delta_sha256s=request.source_embedding_delta_sha256s,
        source_base_model_paths=request.source_base_model_paths,
        training_base_model_path=request.training_base_model_path,
        source_manifest_sha256s=request.source_manifest_sha256s,
        manifest_image_sha256=request.manifest_image_sha256,
        training_image_sha256=(
            training_image if isinstance(training_image, str) else None
        ),
        source_image_sha256s=request.source_image_sha256s,
        training_parameter_state_sha256=(
            parameter_state if isinstance(parameter_state, str) else None
        ),
        manifest_sha256=request.manifest_sha256,
        image_id=request.image_id,
        source_audit_sha256s=request.source_audit_sha256s,
        source_runtime_identity_sha256=_runtime_identity_sha256(
            request.source_runtime_identities
        ),
        checked_decode_count=len(request.decodes),
        checked_token_count=sum(
            len(tuple(getattr(item, "generated_token_ids", ())))
            for item in request.decodes
        ),
        mismatch_count=mismatch_count,
        failure_reason=failure_reason,
    )


def reconcile_source_surface(
    request: SourceSurfaceReconciliationRequest,
) -> SourceSurfaceReconciliationReceipt:
    """Admit only exact Source/training identity and teacher-forced agreement.

    The callback is the existing witness owner.  This function never computes
    logits, changes tolerances, or substitutes another model surface.
    """

    identity = request.training_identity
    mismatches: list[str] = []
    expected_training = {
        "dtype": "bfloat16",
        "attention_backend": "flash_attention_2",
        "model_mode": "eval",
        "use_cache": False,
    }
    for name, expected in expected_training.items():
        if _field(identity, name) != expected:
            mismatches.append(f"training identity {name} differs")
    for name in (
        "checkpoint_payload_sha256",
        "tokenizer_sha256",
        "prompt_sha256",
    ):
        value = _field(identity, name)
        if not isinstance(value, str):
            mismatches.append(f"training identity {name} is absent")
        else:
            try:
                _digest(value, field=f"training identity {name}")
            except ValueError as error:
                mismatches.append(str(error))
                continue
            source_values = tuple(
                getattr(
                    request,
                    {
                        "checkpoint_payload_sha256": "source_checkpoint_payload_sha256s",
                        "tokenizer_sha256": "source_tokenizer_sha256s",
                        "prompt_sha256": "source_prompt_sha256s",
                        "image_sha256": "source_image_sha256s",
                    }[name],
                )
            )
            if name != "checkpoint_payload_sha256" and any(
                source != value for source in source_values
            ):
                mismatches.append(f"Source/training {name} differs")
            try:
                for source in source_values:
                    _digest(source, field=f"source {name}")
            except ValueError as error:
                mismatches.append(str(error))
    if any(
        path != request.training_checkpoint_path
        for path in request.source_checkpoint_paths
    ):
        mismatches.append("Source/training checkpoint path differs")
    for name, request_field in (
        ("adapter_sha256", "source_adapter_sha256s"),
        ("embedding_delta_sha256", "source_embedding_delta_sha256s"),
    ):
        value = _field(identity, name)
        source_values = tuple(getattr(request, request_field))
        if not isinstance(value, str):
            mismatches.append(f"training identity {name} is absent")
            continue
        try:
            _digest(value, field=f"training identity {name}")
            for source in source_values:
                _digest(source, field=f"source {name}")
        except ValueError as error:
            mismatches.append(str(error))
            continue
        if any(source != value for source in source_values):
            mismatches.append(f"Source/training {name} differs")
    if any(
        path
        != request.training_base_model_path
        for path in request.source_base_model_paths
    ):
        mismatches.append("Source/training base-model path differs")
    if any(
        digest != request.manifest_sha256
        for digest in request.source_manifest_sha256s
    ):
        mismatches.append("Source/manifest identity differs")
    training_image = _field(identity, "image_sha256")
    if not isinstance(training_image, str):
        mismatches.append("training identity image_sha256 is absent")
    else:
        try:
            _digest(training_image, field="training identity image_sha256")
        except ValueError as error:
            mismatches.append(str(error))
    for digest in request.source_image_sha256s:
        try:
            _digest(digest, field="source image_sha256")
        except ValueError as error:
            mismatches.append(str(error))
    if len(set(request.source_image_sha256s)) != 1:
        mismatches.append("Source image identity differs between audits")
    if any(
        digest != request.manifest_image_sha256
        for digest in request.source_image_sha256s
    ):
        mismatches.append("Source/manifest image identity differs")
    expected_runtime = {
        "backend": "hf",
        "batch_size": 1,
        "observed_model_dtype_names": ["torch.float32"],
        "observed_attn_implementation": "sdpa",
    }
    for index, runtime in enumerate(request.source_runtime_identities):
        if _runtime_contract(runtime) != expected_runtime:
            mismatches.append(f"source runtime identity {index} differs")
        try:
            json_sha256(_runtime_payload(runtime))
        except Exception:
            mismatches.append(f"source runtime identity {index} is not JSON-addressable")
    for decode in request.decodes:
        if _field(decode, "image_id") != request.image_id:
            mismatches.append("Source decode image identity differs")
        raw_rp = _field(decode, "repetition_penalty")
        if (
            isinstance(raw_rp, bool)
            or not isinstance(raw_rp, (int, float))
            or float(raw_rp) not in _EXPECTED_AUDIT_RPS
        ):
            mismatches.append("Source decode repetition penalty differs")
    if len({
        tuple(getattr(item, "prompt_token_ids", ())) for item in request.decodes
    }) != 1:
        mismatches.append("Source decode prompt identity differs")
    if mismatches:
        return _receipt(
            request,
            mismatch_count=len(mismatches),
            failure_reason="; ".join(mismatches),
            admitted=False,
        )
    try:
        changed = request.check()
        if isinstance(changed, bool) or not isinstance(changed, int) or changed < 0:
            raise TypeError("checker must return a non-negative integer")
    except BaseException as error:
        return _receipt(
            request,
            mismatch_count=1,
            failure_reason=(
                f"checker_error:{type(error).__name__}: {error}"
            ),
            admitted=False,
        )
    if changed:
        return _receipt(
            request,
            mismatch_count=changed,
            failure_reason=f"teacher_forced_greedy_changed_tokens={changed}",
            admitted=False,
        )
    return _receipt(
        request,
        mismatch_count=0,
        failure_reason=None,
        admitted=True,
    )


__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_SURFACE",
    "TRAINING_SURFACE",
    "SourceSurfaceReconciliationReceipt",
    "SourceSurfaceReconciliationRequest",
    "reconcile_source_surface",
]

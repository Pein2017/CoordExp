"""Additive reconciliation of the two frozen Human-13 execution surfaces.

The training owner is BF16/FlashAttention-2 on GPU 0 while the established
behavioral audit is fp32/SDPA on GPU 1.  This module does not replace either
surface and does not relax the Source witness contract.  It only records a
typed admission attempt when the immutable identities and the existing
teacher-forced checker agree exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import cast

from src.artifacts.json_values import json_sha256
from src.data.geometry import COORD_TOKEN_PATTERN


SCHEMA_VERSION = "human13_source_surface_reconciliation.v2"
SOURCE_SURFACE = "gpu1:fp32/sdpa/batch1"
TRAINING_SURFACE = "gpu0:bfloat16/flash_attention_2"
_EXPECTED_AUDIT_RPS = (1.0, 1.1)
_COORDINATE_TOKEN = COORD_TOKEN_PATTERN


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


def _box(value: Sequence[float], *, field: str) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError(f"{field} must contain four coordinates")
    result = cast(
        tuple[float, float, float, float],
        tuple(float(item) for item in value),
    )
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain finite coordinates")
    x1, y1, x2, y2 = result
    if not x1 < x2 or not y1 < y2:
        raise ValueError(f"{field} is not a legal rectangle")
    return result


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1, y1, x2, y2 = _box(first, field="first bbox")
    a1, b1, a2, b2 = _box(second, field="second bbox")
    intersection = max(0.0, min(x2, a2) - max(x1, a1)) * max(
        0.0, min(y2, b2) - max(y1, b1)
    )
    first_area = (x2 - x1) * (y2 - y1)
    second_area = (a2 - a1) * (b2 - b1)
    return intersection / (first_area + second_area - intersection)


@dataclass(frozen=True)
class CoordinateAliasEvidence:
    """One accepted coordinate-bin alias between the two audit surfaces."""

    token_position: int
    coordinate_role: str
    source_token: str
    training_token: str
    source_bin: int
    training_bin: int
    delta_bin: int
    source_bbox: tuple[float, float, float, float]
    training_bbox: tuple[float, float, float, float]
    owner_id: str
    source_iou: float
    training_iou: float
    disposition: str

    def to_dict(self) -> dict[str, object]:
        return {
            "token_position": self.token_position,
            "coordinate_role": self.coordinate_role,
            "source_token": self.source_token,
            "training_token": self.training_token,
            "source_bin": self.source_bin,
            "training_bin": self.training_bin,
            "delta_bin": self.delta_bin,
            "source_bbox": list(self.source_bbox),
            "training_bbox": list(self.training_bbox),
            "owner_id": self.owner_id,
            "source_iou": self.source_iou,
            "training_iou": self.training_iou,
            "disposition": self.disposition,
        }


@dataclass(frozen=True)
class CoordinateAliasReconciliation:
    """Typed result of the fixed cross-surface coordinate-only admission."""

    admitted: bool
    mismatch_count: int
    failure_reason: str | None
    evidence: tuple[CoordinateAliasEvidence, ...] = ()

    @property
    def content_sha256(self) -> str:
        return json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "human13_coordinate_alias_reconciliation.v1",
            "admitted": self.admitted,
            "mismatch_count": self.mismatch_count,
            "failure_reason": self.failure_reason,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["content_sha256"] = self.content_sha256
        return payload


def reconcile_coordinate_alias(
    *,
    source_tokens: Sequence[object],
    training_tokens: Sequence[object],
    coordinate_roles: Mapping[int, tuple[str, str]],
    source_boxes: Mapping[str, Sequence[float]],
    training_boxes: Mapping[str, Sequence[float]],
    gt_boxes: Mapping[str, Sequence[float]],
    owner_match: Mapping[str, str],
    source_owner_rows: Mapping[str, int],
    training_owner_rows: Mapping[str, int],
    source_membership: Mapping[str, str],
    training_membership: Mapping[str, str],
    source_protected_g: Collection[str],
    training_protected_g: Collection[str],
) -> CoordinateAliasReconciliation:
    """Admit only canonical owner-preserving coordinate aliases.

    ``coordinate_roles`` is produced by the canonical parser/projector and
    maps each generated token position to ``(owner_id, x1|y1|x2|y2)``.  This
    deliberately treats arbitrary token-id differences as unsafe: only token
    strings that decode to ``coord_<0..999>`` may use the fixed five-bin alias.
    """

    source = tuple(str(item) for item in source_tokens)
    training = tuple(str(item) for item in training_tokens)
    if len(source) != len(training):
        return CoordinateAliasReconciliation(
            admitted=False,
            mismatch_count=1,
            failure_reason="token_length_differs",
        )
    if dict(owner_match) != {
        owner_id: owner_id for owner_id in source_boxes
    } or set(training_boxes) != set(source_boxes):
        return CoordinateAliasReconciliation(
            admitted=False,
            mismatch_count=1,
            failure_reason="canonical owner assignment differs",
        )
    if dict(source_owner_rows) != dict(training_owner_rows):
        return CoordinateAliasReconciliation(
            admitted=False,
            mismatch_count=1,
            failure_reason="canonical owner row assignment differs",
        )
    if dict(source_membership) != dict(training_membership):
        return CoordinateAliasReconciliation(
            admitted=False,
            mismatch_count=1,
            failure_reason="G/H/M membership differs",
        )
    if set(source_protected_g) != set(training_protected_g):
        return CoordinateAliasReconciliation(
            admitted=False,
            mismatch_count=1,
            failure_reason="protected-G identity differs",
        )
    boxes: dict[str, tuple[float, float, float, float]] = {}
    training_box_values: dict[str, tuple[float, float, float, float]] = {}
    gt: dict[str, tuple[float, float, float, float]] = {}
    try:
        boxes = {
            owner_id: _box(value, field=f"source bbox {owner_id}")
            for owner_id, value in source_boxes.items()
        }
        training_box_values = {
            owner_id: _box(value, field=f"training bbox {owner_id}")
            for owner_id, value in training_boxes.items()
        }
        gt = {
            owner_id: _box(value, field=f"ground-truth bbox {owner_id}")
            for owner_id, value in gt_boxes.items()
        }
    except (TypeError, ValueError) as error:
        return CoordinateAliasReconciliation(
            admitted=False,
            mismatch_count=1,
            failure_reason=str(error),
        )
    if set(boxes) - set(gt):
        return CoordinateAliasReconciliation(
            admitted=False,
            mismatch_count=1,
            failure_reason="matched owner lacks ground-truth binding",
        )
    evidence: list[CoordinateAliasEvidence] = []
    for position, (source_token, training_token) in enumerate(zip(source, training)):
        if source_token == training_token:
            continue
        role_binding = coordinate_roles.get(position)
        if role_binding is None:
            return CoordinateAliasReconciliation(
                admitted=False,
                mismatch_count=1,
                failure_reason=f"non-coordinate token differs at position {position}",
            )
        owner_id, role = role_binding
        source_match = _COORDINATE_TOKEN.fullmatch(source_token)
        training_match = _COORDINATE_TOKEN.fullmatch(training_token)
        if source_match is None or training_match is None:
            return CoordinateAliasReconciliation(
                admitted=False,
                mismatch_count=1,
                failure_reason=f"non-coordinate token differs at position {position}",
            )
        source_bin = int(source_match.group(1))
        training_bin = int(training_match.group(1))
        delta_bin = training_bin - source_bin
        if abs(delta_bin) > 5:
            return CoordinateAliasReconciliation(
                admitted=False,
                mismatch_count=1,
                failure_reason=(
                    f"coordinate delta exceeds five at position {position}: "
                    f"{delta_bin}"
                ),
            )
        if owner_id not in boxes or owner_id not in training_box_values:
            return CoordinateAliasReconciliation(
                admitted=False,
                mismatch_count=1,
                failure_reason=f"coordinate owner {owner_id} is not matched",
            )
        evidence.append(
            CoordinateAliasEvidence(
                token_position=position,
                coordinate_role=role,
                source_token=source_token,
                training_token=training_token,
                source_bin=source_bin,
                training_bin=training_bin,
                delta_bin=delta_bin,
                source_bbox=boxes[owner_id],
                training_bbox=training_box_values[owner_id],
                owner_id=owner_id,
                source_iou=_iou(boxes[owner_id], gt[owner_id]),
                training_iou=_iou(training_box_values[owner_id], gt[owner_id]),
                disposition="metric_equivalent_coordinate_alias",
            )
        )
    return CoordinateAliasReconciliation(
        admitted=True,
        mismatch_count=0,
        failure_reason=None,
        evidence=tuple(evidence),
    )


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
    coordinate_alias_check: Callable[[], CoordinateAliasReconciliation] | None = None

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
    coordinate_alias: CoordinateAliasReconciliation | None = None

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
            "coordinate_alias": (
                self.coordinate_alias.to_dict()
                if self.coordinate_alias is not None
                else None
            ),
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
    coordinate_alias: CoordinateAliasReconciliation | None = None,
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
        coordinate_alias=coordinate_alias,
    )


def reconcile_source_surface(
    request: SourceSurfaceReconciliationRequest,
) -> SourceSurfaceReconciliationReceipt:
    """Admit immutable identities plus the fixed cross-surface checker.

    The callbacks are existing scientific owners.  This function never computes
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
    coordinate_alias: CoordinateAliasReconciliation | None = None
    if request.coordinate_alias_check is not None:
        try:
            coordinate_alias = request.coordinate_alias_check()
            if not isinstance(coordinate_alias, CoordinateAliasReconciliation):
                raise TypeError("coordinate alias checker returned an invalid receipt")
        except BaseException as error:
            return _receipt(
                request,
                mismatch_count=1,
                failure_reason=(
                    f"coordinate_alias_checker_error:{type(error).__name__}: {error}"
                ),
                admitted=False,
            )
        if not coordinate_alias.admitted:
            return _receipt(
                request,
                mismatch_count=max(1, coordinate_alias.mismatch_count),
                failure_reason=coordinate_alias.failure_reason,
                admitted=False,
                coordinate_alias=coordinate_alias,
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
            coordinate_alias=coordinate_alias,
        )
    if changed:
        return _receipt(
            request,
            mismatch_count=changed,
            failure_reason=f"teacher_forced_greedy_changed_tokens={changed}",
            admitted=False,
            coordinate_alias=coordinate_alias,
        )
    return _receipt(
        request,
        mismatch_count=0,
        failure_reason=None,
        admitted=True,
        coordinate_alias=coordinate_alias,
    )


__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_SURFACE",
    "TRAINING_SURFACE",
    "SourceSurfaceReconciliationReceipt",
    "SourceSurfaceReconciliationRequest",
    "CoordinateAliasEvidence",
    "CoordinateAliasReconciliation",
    "reconcile_coordinate_alias",
    "reconcile_source_surface",
]

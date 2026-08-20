"""Additive reconciliation of the two frozen Human-13 execution surfaces.

The training owner is BF16/FlashAttention-2 on GPU 0 while the established
behavioral audit is fp32/SDPA on GPU 1.  This module does not replace either
surface and does not relax the Source witness contract.  It only records a
typed admission attempt when the immutable identities and the existing
teacher-forced checker agree exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
import copy
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, cast

from src.artifacts.json_values import json_sha256
from src.data.geometry import COORD_TOKEN_PATTERN


LEGACY_SCHEMA_VERSION = "human13_source_surface_reconciliation.v2"
SCHEMA_VERSION = "human13_source_surface_reconciliation.v3"
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


def _coerce_int(value: object) -> int:
    return int(cast(Any, value))


def _coerce_float(value: object) -> float:
    return float(cast(Any, value))


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

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CoordinateAliasEvidence:
        expected = {
            "token_position",
            "coordinate_role",
            "source_token",
            "training_token",
            "source_bin",
            "training_bin",
            "delta_bin",
            "source_bbox",
            "training_bbox",
            "owner_id",
            "source_iou",
            "training_iou",
            "disposition",
        }
        if set(value) != expected:
            raise ValueError("coordinate alias evidence fields differ from schema")
        return cls(
            token_position=_coerce_int(value["token_position"]),
            coordinate_role=str(value["coordinate_role"]),
            source_token=str(value["source_token"]),
            training_token=str(value["training_token"]),
            source_bin=_coerce_int(value["source_bin"]),
            training_bin=_coerce_int(value["training_bin"]),
            delta_bin=_coerce_int(value["delta_bin"]),
            source_bbox=cast(
                tuple[float, float, float, float],
                tuple(
                    _coerce_float(item)
                    for item in cast(Sequence[object], value["source_bbox"])
                ),
            ),
            training_bbox=cast(
                tuple[float, float, float, float],
                tuple(
                    _coerce_float(item)
                    for item in cast(Sequence[object], value["training_bbox"])
                ),
            ),
            owner_id=str(value["owner_id"]),
            source_iou=_coerce_float(value["source_iou"]),
            training_iou=_coerce_float(value["training_iou"]),
            disposition=str(value["disposition"]),
        )


@dataclass(frozen=True)
class CoordinateAliasFailureEvidence:
    """Compact, offline-replayable evidence for a rejected alias attempt."""

    payload: Mapping[str, object]

    _EXPECTED_FIELDS = frozenset(
        {
            "schema_version",
            "repetition_penalty",
            "source_tokens",
            "training_tokens",
            "source_token_count",
            "training_token_count",
            "source_token_ids",
            "training_token_ids",
            "mismatch_positions",
            "source_owner_set",
            "training_owner_set",
            "symmetric_owner_set_difference",
            "source_owner_rows",
            "training_owner_rows",
            "source_membership",
            "training_membership",
            "source_protected_g",
            "training_protected_g",
            "token_mismatches",
            "affected_rows",
            "source_boxes",
            "training_boxes",
            "gt_boxes",
        }
    )

    def __post_init__(self) -> None:
        if set(self.payload) != self._EXPECTED_FIELDS:
            raise ValueError("coordinate alias failure evidence fields differ from schema")
        if self.payload.get("schema_version") != "human13_coordinate_alias_failure.v1":
            raise ValueError("coordinate alias failure evidence schema differs")
        # Force a JSON-addressability check at the constructor boundary so a
        # receipt can never carry an opaque object that cannot be reloaded.
        json_sha256(dict(self.payload))

    def to_dict(self) -> dict[str, object]:
        return copy.deepcopy(dict(self.payload))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CoordinateAliasFailureEvidence:
        return cls(copy.deepcopy(dict(value)))


@dataclass(frozen=True)
class CoordinateAliasReconciliation:
    """Typed result of the fixed cross-surface coordinate-only admission."""

    admitted: bool
    mismatch_count: int
    failure_reason: str | None
    evidence: tuple[CoordinateAliasEvidence, ...] = ()
    failure_evidence: CoordinateAliasFailureEvidence | None = None

    @property
    def content_sha256(self) -> str:
        return json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "human13_coordinate_alias_reconciliation.v1",
            "admitted": self.admitted,
            "mismatch_count": self.mismatch_count,
            "failure_reason": self.failure_reason,
            "evidence": [item.to_dict() for item in self.evidence],
            "failure_evidence": (
                self.failure_evidence.to_dict()
                if self.failure_evidence is not None
                else None
            ),
        }
        return payload

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["content_sha256"] = self.content_sha256
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CoordinateAliasReconciliation:
        expected = {
            "schema_version",
            "admitted",
            "mismatch_count",
            "failure_reason",
            "evidence",
            "failure_evidence",
            "content_sha256",
        }
        if set(value) != expected or value.get("schema_version") != (
            "human13_coordinate_alias_reconciliation.v1"
        ):
            raise ValueError("coordinate alias reconciliation fields differ from schema")
        raw_evidence = value.get("evidence")
        if not isinstance(raw_evidence, list) or not all(
            isinstance(item, Mapping) for item in raw_evidence
        ):
            raise ValueError("coordinate alias evidence is malformed")
        raw_failure = value.get("failure_evidence")
        failure = (
            CoordinateAliasFailureEvidence.from_dict(raw_failure)
            if isinstance(raw_failure, Mapping)
            else None
        )
        receipt = cls(
            admitted=bool(value["admitted"]),
            mismatch_count=_coerce_int(value["mismatch_count"]),
            failure_reason=(
                str(value["failure_reason"])
                if value["failure_reason"] is not None
                else None
            ),
            evidence=tuple(
                CoordinateAliasEvidence.from_dict(item) for item in raw_evidence
            ),
            failure_evidence=failure,
        )
        if value.get("content_sha256") != receipt.content_sha256:
            raise ValueError("coordinate alias reconciliation content hash differs")
        return receipt


def reconcile_coordinate_alias(
    *,
    source_tokens: Sequence[object],
    training_tokens: Sequence[object],
    source_token_ids: Sequence[object] | None = None,
    training_token_ids: Sequence[object] | None = None,
    repetition_penalty: float | None = None,
    forced_failure_reason: str | None = None,
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

    def _token_ids(
        values: Sequence[object] | None, *, length: int
    ) -> tuple[int | None, ...]:
        if values is None:
            return tuple(None for _ in range(length))
        result: list[int | None] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                result.append(None)
            else:
                result.append(value)
        return tuple(result)

    source_ids = _token_ids(source_token_ids, length=len(source))
    training_ids = _token_ids(training_token_ids, length=len(training))

    def _raw_box(value: object) -> object:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [item for item in value]
        return None

    def _safe_iou(first: object, second: object) -> float | None:
        try:
            return _iou(cast(Sequence[float], first), cast(Sequence[float], second))
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    source_owner_set = tuple(sorted(str(owner_id) for owner_id in source_boxes))
    training_owner_set = tuple(sorted(str(owner_id) for owner_id in training_boxes))
    source_owner_rows_payload = tuple(
        sorted((str(owner_id), int(row)) for owner_id, row in source_owner_rows.items())
    )
    training_owner_rows_payload = tuple(
        sorted(
            (str(owner_id), int(row)) for owner_id, row in training_owner_rows.items()
        )
    )

    def _failure_evidence() -> CoordinateAliasFailureEvidence:
        mismatches: list[dict[str, object]] = []
        for position in range(max(len(source), len(training))):
            source_token = source[position] if position < len(source) else None
            training_token = training[position] if position < len(training) else None
            if source_token == training_token:
                continue
            binding = coordinate_roles.get(position)
            source_match = (
                _COORDINATE_TOKEN.fullmatch(source_token)
                if source_token is not None
                else None
            )
            training_match = (
                _COORDINATE_TOKEN.fullmatch(training_token)
                if training_token is not None
                else None
            )
            source_bin = int(source_match.group(1)) if source_match else None
            training_bin = int(training_match.group(1)) if training_match else None
            mismatches.append(
                {
                    "token_position": position,
                    "source_token_id": source_ids[position]
                    if position < len(source_ids)
                    else None,
                    "training_token_id": training_ids[position]
                    if position < len(training_ids)
                    else None,
                    "source_token": source_token,
                    "training_token": training_token,
                    "is_coordinate": binding is not None
                    or source_match is not None
                    or training_match is not None,
                    "coordinate_role": binding[1] if binding is not None else None,
                    "owner_id": binding[0] if binding is not None else None,
                    "source_bin": source_bin,
                    "training_bin": training_bin,
                    "delta_bin": (
                        training_bin - source_bin
                        if source_bin is not None and training_bin is not None
                        else None
                    ),
                    "disposition": (
                        "coordinate_alias_candidate"
                        if (
                            source_match is not None
                            and training_match is not None
                            and binding is not None
                        )
                        else "non_coordinate_token_diff"
                    ),
                }
            )
        affected = sorted(set(source_owner_set) | set(training_owner_set))
        affected_rows: list[dict[str, object]] = []
        for owner_id in affected:
            source_box = source_boxes.get(owner_id)
            training_box = training_boxes.get(owner_id)
            candidate_ious: list[dict[str, object]] = []
            for gt_owner_id in sorted(str(item) for item in gt_boxes):
                gt_box = gt_boxes.get(gt_owner_id)
                candidate_ious.append(
                    {
                        "gt_owner_id": gt_owner_id,
                        "source_iou": _safe_iou(source_box, gt_box),
                        "training_iou": _safe_iou(training_box, gt_box),
                    }
                )
            affected_rows.append(
                {
                    "owner_id": owner_id,
                    "source_generated_order": dict(source_owner_rows).get(owner_id),
                    "training_generated_order": dict(training_owner_rows).get(owner_id),
                    "source_bbox": _raw_box(source_box),
                    "training_bbox": _raw_box(training_box),
                    "candidate_gt_ious": candidate_ious,
                }
            )
        payload = {
            "schema_version": "human13_coordinate_alias_failure.v1",
            "repetition_penalty": repetition_penalty,
            "source_tokens": list(source),
            "training_tokens": list(training),
            "source_token_count": len(source),
            "training_token_count": len(training),
            "source_token_ids": list(source_ids),
            "training_token_ids": list(training_ids),
            "mismatch_positions": [
                _coerce_int(item["token_position"]) for item in mismatches
            ],
            "source_owner_set": list(source_owner_set),
            "training_owner_set": list(training_owner_set),
            "symmetric_owner_set_difference": sorted(
                set(source_owner_set) ^ set(training_owner_set)
            ),
            "source_owner_rows": [list(item) for item in source_owner_rows_payload],
            "training_owner_rows": [
                list(item) for item in training_owner_rows_payload
            ],
            "source_membership": [
                list(item)
                for item in sorted(
                    (str(owner_id), str(member))
                    for owner_id, member in source_membership.items()
                )
            ],
            "training_membership": [
                list(item)
                for item in sorted(
                    (str(owner_id), str(member))
                    for owner_id, member in training_membership.items()
                )
            ],
            "source_protected_g": sorted(str(owner_id) for owner_id in source_protected_g),
            "training_protected_g": sorted(
                str(owner_id) for owner_id in training_protected_g
            ),
            "token_mismatches": mismatches,
            "affected_rows": affected_rows,
            "source_boxes": [
                [str(owner_id), _raw_box(value)]
                for owner_id, value in sorted(source_boxes.items())
            ],
            "training_boxes": [
                [str(owner_id), _raw_box(value)]
                for owner_id, value in sorted(training_boxes.items())
            ],
            "gt_boxes": [
                [str(owner_id), _raw_box(value)]
                for owner_id, value in sorted(gt_boxes.items())
            ],
        }
        return CoordinateAliasFailureEvidence(payload)

    def _failed(
        reason: str,
        *,
        mismatch_count: int = 1,
    ) -> CoordinateAliasReconciliation:
        return CoordinateAliasReconciliation(
            admitted=False,
            mismatch_count=max(1, mismatch_count),
            failure_reason=reason,
            failure_evidence=_failure_evidence(),
        )

    if forced_failure_reason is not None:
        return _failed(forced_failure_reason)

    if any(role not in {"x1", "y1", "x2", "y2"} for _owner, role in coordinate_roles.values()):
        return _failed("coordinate role is not canonical")

    if len(source) != len(training):
        return _failed("token_length_differs")
    if dict(owner_match) != {
        owner_id: owner_id for owner_id in source_boxes
    } or set(training_boxes) != set(source_boxes):
        return _failed("canonical owner assignment differs")
    if dict(source_owner_rows) != dict(training_owner_rows):
        return _failed("canonical owner row assignment differs")
    if dict(source_membership) != dict(training_membership):
        return _failed("G/H/M membership differs")
    if set(source_protected_g) != set(training_protected_g):
        return _failed("protected-G identity differs")
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
        return _failed(str(error))
    if set(boxes) - set(gt):
        return _failed("matched owner lacks ground-truth binding")
    evidence: list[CoordinateAliasEvidence] = []
    for position, (source_token, training_token) in enumerate(zip(source, training)):
        if source_token == training_token:
            continue
        role_binding = coordinate_roles.get(position)
        if role_binding is None:
            return _failed(f"non-coordinate token differs at position {position}")
        owner_id, role = role_binding
        source_match = _COORDINATE_TOKEN.fullmatch(source_token)
        training_match = _COORDINATE_TOKEN.fullmatch(training_token)
        if source_match is None or training_match is None:
            return _failed(f"non-coordinate token differs at position {position}")
        source_bin = int(source_match.group(1))
        training_bin = int(training_match.group(1))
        delta_bin = training_bin - source_bin
        if abs(delta_bin) > 5:
            return _failed(
                f"coordinate delta exceeds five at position {position}: {delta_bin}"
            )
        if owner_id not in boxes or owner_id not in training_box_values:
            return _failed(f"coordinate owner {owner_id} is not matched")
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
class CanonicalSourceBaselineReceipt:
    """One complete canonical Source baseline and its owner assignment."""

    surface: str
    repetition_penalty: float
    canonical_payload: Mapping[str, object]
    owner_map: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.surface not in {SOURCE_SURFACE, TRAINING_SURFACE}:
            raise ValueError("canonical Source baseline surface differs")
        if (
            isinstance(self.repetition_penalty, bool)
            or float(self.repetition_penalty) not in _EXPECTED_AUDIT_RPS
        ):
            raise ValueError("canonical Source baseline RP differs")
        if not isinstance(self.canonical_payload, Mapping):
            raise TypeError("canonical Source baseline payload must be an object")
        if not isinstance(self.owner_map, Mapping):
            raise TypeError("canonical Source baseline owner map must be an object")
        payload = copy.deepcopy(dict(self.canonical_payload))
        owner_map = copy.deepcopy(dict(self.owner_map))
        json_sha256(payload)
        json_sha256(owner_map)
        object.__setattr__(self, "repetition_penalty", float(self.repetition_penalty))
        object.__setattr__(self, "canonical_payload", payload)
        object.__setattr__(self, "owner_map", owner_map)

    @property
    def canonical_payload_sha256(self) -> str:
        return json_sha256(dict(self.canonical_payload))

    @property
    def owner_map_sha256(self) -> str:
        return json_sha256(dict(self.owner_map))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "human13_canonical_source_baseline.v1",
            "surface": self.surface,
            "repetition_penalty": self.repetition_penalty,
            "canonical_payload": copy.deepcopy(dict(self.canonical_payload)),
            "canonical_payload_sha256": self.canonical_payload_sha256,
            "owner_map": copy.deepcopy(dict(self.owner_map)),
            "owner_map_sha256": self.owner_map_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CanonicalSourceBaselineReceipt:
        expected = {
            "schema_version",
            "surface",
            "repetition_penalty",
            "canonical_payload",
            "canonical_payload_sha256",
            "owner_map",
            "owner_map_sha256",
        }
        if set(value) != expected or value.get("schema_version") != (
            "human13_canonical_source_baseline.v1"
        ):
            raise ValueError("canonical Source baseline fields differ from schema")
        payload = value["canonical_payload"]
        owner_map = value["owner_map"]
        if not isinstance(payload, Mapping) or not isinstance(owner_map, Mapping):
            raise TypeError("canonical Source baseline evidence must be objects")
        receipt = cls(
            surface=str(value["surface"]),
            repetition_penalty=_coerce_float(value["repetition_penalty"]),
            canonical_payload=payload,
            owner_map=owner_map,
        )
        if value["canonical_payload_sha256"] != receipt.canonical_payload_sha256:
            raise ValueError("canonical Source baseline payload hash differs")
        if value["owner_map_sha256"] != receipt.owner_map_sha256:
            raise ValueError("canonical Source baseline owner-map hash differs")
        return receipt


def _canonical_baseline_matrix(
    values: Sequence[CanonicalSourceBaselineReceipt],
) -> tuple[CanonicalSourceBaselineReceipt, ...]:
    baselines = tuple(values)
    expected = (
        (SOURCE_SURFACE, 1.0),
        (SOURCE_SURFACE, 1.1),
        (TRAINING_SURFACE, 1.0),
        (TRAINING_SURFACE, 1.1),
    )
    observed = tuple((item.surface, item.repetition_penalty) for item in baselines)
    if observed != expected:
        raise ValueError(
            "canonical Source baselines must bind fp32/BF16 RP1.0/RP1.1"
        )
    return baselines


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
    canonical_baselines: tuple[CanonicalSourceBaselineReceipt, ...]
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
        object.__setattr__(
            self,
            "canonical_baselines",
            _canonical_baseline_matrix(self.canonical_baselines),
        )


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
    canonical_baselines: tuple[CanonicalSourceBaselineReceipt, ...]
    coordinate_alias: CoordinateAliasReconciliation | None = None
    _schema_version: str = field(
        default=SCHEMA_VERSION,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self._schema_version == SCHEMA_VERSION:
            object.__setattr__(
                self,
                "canonical_baselines",
                _canonical_baseline_matrix(self.canonical_baselines),
            )
        elif self._schema_version == LEGACY_SCHEMA_VERSION:
            if self.canonical_baselines:
                raise ValueError("legacy v2 receipt cannot carry v3 baselines")
            object.__setattr__(self, "canonical_baselines", ())
        else:
            raise ValueError("Source reconciliation receipt schema is unsupported")

    @property
    def cross_surface_disposition(self) -> str:
        """Return the non-decision-bearing disposition of surface divergence."""

        if self.coordinate_alias is None:
            return "not_checked"
        return "admitted" if self.coordinate_alias.admitted else "diagnostic_only"

    @property
    def content_sha256(self) -> str:
        return json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self._schema_version,
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
        if self._schema_version == SCHEMA_VERSION:
            payload["canonical_baselines"] = [
                item.to_dict() for item in self.canonical_baselines
            ]
        return payload

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["content_sha256"] = self.content_sha256
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceSurfaceReconciliationReceipt:
        schema_version = value.get("schema_version")
        common_fields = {
            "schema_version",
            "admitted",
            "source_surface",
            "training_surface",
            "training_model_object_id",
            "training_checkpoint_payload_sha256",
            "source_checkpoint_payload_sha256s",
            "source_checkpoint_paths",
            "training_checkpoint_path",
            "source_adapter_sha256s",
            "source_embedding_delta_sha256s",
            "source_base_model_paths",
            "training_base_model_path",
            "source_manifest_sha256s",
            "manifest_image_sha256",
            "training_image_sha256",
            "source_image_sha256s",
            "training_parameter_state_sha256",
            "manifest_sha256",
            "image_id",
            "source_audit_sha256s",
            "source_runtime_identity_sha256",
            "checked_decode_count",
            "checked_token_count",
            "mismatch_count",
            "failure_reason",
            "coordinate_alias",
            "content_sha256",
        }
        expected = (
            common_fields | {"canonical_baselines"}
            if schema_version == SCHEMA_VERSION
            else common_fields
        )
        if schema_version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION} or set(
            value
        ) != expected:
            raise ValueError("Source reconciliation receipt fields differ from schema")
        raw_alias = value.get("coordinate_alias")
        alias = (
            CoordinateAliasReconciliation.from_dict(raw_alias)
            if isinstance(raw_alias, Mapping)
            else None
        )
        if schema_version == SCHEMA_VERSION:
            raw_baselines = value.get("canonical_baselines")
            if not isinstance(raw_baselines, Sequence):
                raise TypeError("canonical Source baselines must be a sequence")
            baselines = tuple(
                CanonicalSourceBaselineReceipt.from_dict(item)
                for item in raw_baselines
                if isinstance(item, Mapping)
            )
            if len(baselines) != len(raw_baselines):
                raise TypeError("canonical Source baseline entry must be an object")
        else:
            baselines = ()
        receipt = cls(
            admitted=bool(value["admitted"]),
            source_surface=str(value["source_surface"]),
            training_surface=str(value["training_surface"]),
            training_model_object_id=(
                _coerce_int(value["training_model_object_id"])
                if value["training_model_object_id"] is not None
                else None
            ),
            training_checkpoint_payload_sha256=(
                str(value["training_checkpoint_payload_sha256"])
                if value["training_checkpoint_payload_sha256"] is not None
                else None
            ),
            source_checkpoint_payload_sha256s=tuple(
                str(item)
                for item in cast(
                    Sequence[object], value["source_checkpoint_payload_sha256s"]
                )
            ),
            source_checkpoint_paths=tuple(
                str(item)
                for item in cast(Sequence[object], value["source_checkpoint_paths"])
            ),
            training_checkpoint_path=str(value["training_checkpoint_path"]),
            source_adapter_sha256s=tuple(
                str(item)
                for item in cast(Sequence[object], value["source_adapter_sha256s"])
            ),
            source_embedding_delta_sha256s=tuple(
                str(item)
                for item in cast(
                    Sequence[object], value["source_embedding_delta_sha256s"]
                )
            ),
            source_base_model_paths=tuple(
                str(item)
                for item in cast(Sequence[object], value["source_base_model_paths"])
            ),
            training_base_model_path=str(value["training_base_model_path"]),
            source_manifest_sha256s=tuple(
                str(item)
                for item in cast(Sequence[object], value["source_manifest_sha256s"])
            ),
            manifest_image_sha256=str(value["manifest_image_sha256"]),
            training_image_sha256=(
                str(value["training_image_sha256"])
                if value["training_image_sha256"] is not None
                else None
            ),
            source_image_sha256s=tuple(
                str(item)
                for item in cast(Sequence[object], value["source_image_sha256s"])
            ),
            training_parameter_state_sha256=(
                str(value["training_parameter_state_sha256"])
                if value["training_parameter_state_sha256"] is not None
                else None
            ),
            manifest_sha256=str(value["manifest_sha256"]),
            image_id=_coerce_int(value["image_id"]),
            source_audit_sha256s=tuple(
                (_coerce_float(item[0]), str(item[1]))
                for item in cast(
                    Sequence[Sequence[object]], value["source_audit_sha256s"]
                )
            ),
            source_runtime_identity_sha256=str(
                value["source_runtime_identity_sha256"]
            ),
            checked_decode_count=_coerce_int(value["checked_decode_count"]),
            checked_token_count=_coerce_int(value["checked_token_count"]),
            mismatch_count=_coerce_int(value["mismatch_count"]),
            failure_reason=(
                str(value["failure_reason"])
                if value["failure_reason"] is not None
                else None
            ),
            coordinate_alias=alias,
            canonical_baselines=baselines,
            _schema_version=str(schema_version),
        )
        if value.get("content_sha256") != receipt.content_sha256:
            raise ValueError("Source reconciliation receipt content hash differs")
        return receipt


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
        canonical_baselines=request.canonical_baselines,
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
        "image_sha256",
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
        # Coordinate/token/row/owner divergence is deliberately retained as
        # diagnostic evidence.  It does not gate the independent BF16 policy
        # baseline; strict identity and the BF16-native ``request.check`` below
        # remain the only admission conditions here.
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
    "LEGACY_SCHEMA_VERSION",
    "SOURCE_SURFACE",
    "TRAINING_SURFACE",
    "CanonicalSourceBaselineReceipt",
    "SourceSurfaceReconciliationReceipt",
    "SourceSurfaceReconciliationRequest",
    "CoordinateAliasEvidence",
    "CoordinateAliasFailureEvidence",
    "CoordinateAliasReconciliation",
    "reconcile_coordinate_alias",
    "reconcile_source_surface",
]

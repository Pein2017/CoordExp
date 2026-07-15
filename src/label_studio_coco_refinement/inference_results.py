"""Pure request binding, parser outcome, and replay receipt records."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

from src.common.errors import DataContractError
from src.inference.parsing import ParseRow, parse_compact_object_box_closed
from src.label_studio_coco_refinement.categories import (
    COCO80_CATEGORIES,
    COCO80_REGISTRY,
    Coco80Registry,
)
from src.label_studio_coco_refinement.roi_transform import (
    CROP_EDGE_CONVENTION,
    InverseMapping,
    PAD_VALUE_RGB,
    PIXEL_EDGE_CONVENTION,
    RESAMPLER,
    ROI_TRANSFORM_ID,
    RoiLetterboxTransform,
    RoiTransformError,
)


class InferenceResultContractError(ValueError):
    """Raised for a corrupt or contradictory inference result record."""


@dataclass(frozen=True)
class CategoryIdentity:
    """The narrow canonical category boundary required by this module."""

    name: str
    category_id: int


class CategoryResolver(Protocol):
    """Adapter seam for the parent-owned fixed COCO-80 registry."""

    def __call__(self, exact_name: str) -> CategoryIdentity | None: ...


@dataclass(frozen=True)
class CanonicalCoco80Resolver:
    """Resolver cryptographically and structurally bound to canonical COCO-80."""

    registry: Coco80Registry
    registry_fingerprint: str
    category_content: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        expected_content = tuple(
            (category.id, category.name) for category in COCO80_CATEGORIES
        )
        observed_content = tuple(
            (category.id, category.name) for category in self.registry.categories
        )
        if (
            self.registry_fingerprint != COCO80_REGISTRY.fingerprint
            or self.registry.fingerprint != COCO80_REGISTRY.fingerprint
            or self.category_content != expected_content
            or observed_content != expected_content
        ):
            raise InferenceResultContractError(
                "category resolver is not bound to the canonical sparse COCO-80 registry"
            )

    def __call__(self, exact_name: str) -> CategoryIdentity | None:
        try:
            category = self.registry.by_name(exact_name)
        except DataContractError:
            return None
        return CategoryIdentity(name=category.name, category_id=category.id)


def coco80_category_resolver(registry: Coco80Registry) -> CanonicalCoco80Resolver:
    """Bind a registry only when fingerprint and all sparse entries are canonical."""

    return CanonicalCoco80Resolver(
        registry=registry,
        registry_fingerprint=registry.fingerprint,
        category_content=tuple(
            (category.id, category.name) for category in registry.categories
        ),
    )


CANONICAL_COCO80_RESOLVER = coco80_category_resolver(COCO80_REGISTRY)


@dataclass(frozen=True)
class RequestTarget:
    """Every mutable identity frozen when the user submits one ROI request."""

    request_id: str
    project_id: str
    task_id: str
    task_epoch: str
    image_id: str
    annotation_id: str
    annotation_revision: str
    profile_fingerprint: str
    project_generation: int
    transform_fingerprint: str
    preexisting_draft_dirty: bool

    def __post_init__(self) -> None:
        for field in (
            "request_id",
            "project_id",
            "task_id",
            "task_epoch",
            "image_id",
            "annotation_id",
            "annotation_revision",
            "profile_fingerprint",
            "transform_fingerprint",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise InferenceResultContractError(f"{field} must be non-empty text")
        if (
            isinstance(self.project_generation, bool)
            or not isinstance(self.project_generation, int)
            or self.project_generation < 0
        ):
            raise InferenceResultContractError(
                "project_generation must be a non-negative integer"
            )

    def binding_payload(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "task_id": self.task_id,
            "task_epoch": self.task_epoch,
            "image_id": self.image_id,
            "annotation_id": self.annotation_id,
            "annotation_revision": self.annotation_revision,
            "profile_fingerprint": self.profile_fingerprint,
            "project_generation": self.project_generation,
        }

    def to_receipt_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            **self.binding_payload(),
            "transform_fingerprint": self.transform_fingerprint,
            "preexisting_draft_dirty": self.preexisting_draft_dirty,
        }


@dataclass(frozen=True)
class CurrentTarget:
    project_id: str
    task_id: str
    task_epoch: str
    image_id: str
    annotation_id: str
    annotation_revision: str
    profile_fingerprint: str
    project_generation: int

    def __post_init__(self) -> None:
        for field in (
            "project_id",
            "task_id",
            "task_epoch",
            "image_id",
            "annotation_id",
            "annotation_revision",
            "profile_fingerprint",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise InferenceResultContractError(f"{field} must be non-empty text")
        if (
            isinstance(self.project_generation, bool)
            or not isinstance(self.project_generation, int)
            or self.project_generation < 0
        ):
            raise InferenceResultContractError(
                "project_generation must be a non-negative integer"
            )

    def binding_payload(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "task_id": self.task_id,
            "task_epoch": self.task_epoch,
            "image_id": self.image_id,
            "annotation_id": self.annotation_id,
            "annotation_revision": self.annotation_revision,
            "profile_fingerprint": self.profile_fingerprint,
            "project_generation": self.project_generation,
        }


class Outcome(str, Enum):
    ACCEPTED = "accepted"
    ACCEPTED_WITH_DROPS = "accepted_with_drops"
    EMPTY = "empty"
    ALL_REJECTED = "all_rejected"
    ALL_SPANS_DROPPED = "all_spans_dropped"
    UNSUPPORTED_FORMAT = "unsupported_format"

    @property
    def clears_roi(self) -> bool:
        return self in {
            Outcome.ACCEPTED,
            Outcome.ACCEPTED_WITH_DROPS,
            Outcome.EMPTY,
            Outcome.ALL_REJECTED,
        }

    @property
    def response_failure(self) -> bool:
        return self in {Outcome.ALL_SPANS_DROPPED, Outcome.UNSUPPORTED_FORMAT}


@dataclass(frozen=True)
class AcceptedInferenceRegion:
    result_id: str
    category_name: str
    category_id: int
    norm1000_bbox: tuple[int, int, int, int]
    request_id: str
    parser_object_span_id: str
    region_link: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "category_name": self.category_name,
            "category_id": self.category_id,
            "bbox_2d": list(self.norm1000_bbox),
            "request_id": self.request_id,
            "parser_object_span_id": self.parser_object_span_id,
            "region_link": self.region_link,
        }


@dataclass(frozen=True)
class ResultReplayRecord:
    result_id: str
    parser_object_span_id: str
    parser_generated_order: int | None
    raw_span_text: str
    raw_span_sha256: str
    char_start: int
    char_end: int
    coord_bins: tuple[int, ...]
    parser_canvas_bbox: tuple[float, float, float, float] | None
    parsed_description: str | None
    canonical_category_name: str | None
    official_category_id: int | None
    class_decision: str
    reject_reason: str | None
    inverse_mapping: InverseMapping | None
    final_norm1000_bbox: tuple[int, int, int, int] | None
    region_link: str | None = None

    def to_receipt_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "parser_object_span_id": self.parser_object_span_id,
            "parser_generated_order": self.parser_generated_order,
            "raw_span_text": self.raw_span_text,
            "raw_span_sha256": self.raw_span_sha256,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "coord_bins": list(self.coord_bins),
            "parser_canvas_bbox": (
                None if self.parser_canvas_bbox is None else list(self.parser_canvas_bbox)
            ),
            "parsed_description": self.parsed_description,
            "canonical_category_name": self.canonical_category_name,
            "official_category_id": self.official_category_id,
            "class_decision": self.class_decision,
            "reject_reason": self.reject_reason,
            "inverse_mapping": (
                None
                if self.inverse_mapping is None
                else self.inverse_mapping.to_receipt_dict()
            ),
            "final_norm1000_bbox": (
                None
                if self.final_norm1000_bbox is None
                else list(self.final_norm1000_bbox)
            ),
            "region_link": self.region_link,
        }


@dataclass(frozen=True)
class DirectInsertionPayload:
    """One atomic append payload; it has no replacement/NMS/acceptance fields."""

    target: RequestTarget
    regions: tuple[AcceptedInferenceRegion, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_receipt_dict(),
            "mode": "append_one_undo_action",
            "regions": [region.to_dict() for region in self.regions],
        }


@dataclass(frozen=True)
class ClassifiedInferenceResult:
    target: RequestTarget
    parser_status: str
    outcome: Outcome
    raw_response_text: str
    raw_response_sha256: str
    parse_row_sha256: str
    parsed_count: int
    inserted_count: int
    rejected_count: int
    records: tuple[ResultReplayRecord, ...]
    insertion_payload: DirectInsertionPayload | None

    @property
    def clear_roi(self) -> bool:
        return self.outcome.clears_roi

    def to_receipt_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_receipt_dict(),
            "parser_status": self.parser_status,
            "outcome": self.outcome.value,
            "clear_roi": self.clear_roi,
            "raw_response_text": self.raw_response_text,
            "raw_response_sha256": self.raw_response_sha256,
            "parse_row_sha256": self.parse_row_sha256,
            "parsed_count": self.parsed_count,
            "inserted_count": self.inserted_count,
            "rejected_count": self.rejected_count,
            "results": [record.to_receipt_dict() for record in self.records],
            "insertion_payload": (
                None if self.insertion_payload is None else self.insertion_payload.to_dict()
            ),
        }


def classify_parser_result(
    *,
    parse_row: ParseRow,
    raw_response_text: str,
    target: RequestTarget,
    transform: RoiLetterboxTransform,
    resolve_category: CategoryResolver | None = None,
) -> ClassifiedInferenceResult:
    """Validate canonical classes, map boxes, and apply the exact outcome table."""

    if not isinstance(parse_row, ParseRow):
        raise InferenceResultContractError("parse_row must be a current ParseRow")
    if not isinstance(raw_response_text, str):
        raise InferenceResultContractError("raw_response_text must be text")
    if parse_row.row_id != target.request_id:
        raise InferenceResultContractError(
            "parser row identity does not match the frozen request identity"
        )
    if transform.fingerprint != target.transform_fingerprint:
        raise InferenceResultContractError(
            "ROI transform does not match the frozen request transform"
        )
    if resolve_category is None:
        resolve_category = CANONICAL_COCO80_RESOLVER
    if not isinstance(resolve_category, CanonicalCoco80Resolver):
        raise InferenceResultContractError(
            "result classification requires a canonical COCO-80 resolver binding"
        )
    # Re-run its constructor checks in case a caller supplied a deserialized object.
    resolve_category.__post_init__()
    if parse_row.parse_status not in {
        "accepted",
        "accepted_with_drops",
        "empty",
        "all_spans_dropped",
        "unsupported_format",
    }:
        raise InferenceResultContractError(
            f"unsupported current parser status: {parse_row.parse_status}"
        )
    _validate_parser_status_shape(parse_row)
    parse_row_sha256 = _bind_parse_row_to_raw(
        parse_row=parse_row,
        raw_response_text=raw_response_text,
        target=target,
        transform=transform,
    )

    records: list[ResultReplayRecord] = []
    regions: list[AcceptedInferenceRegion] = []
    for index, prediction in enumerate(parse_row.predictions):
        result_id = f"{target.request_id}:result-{index}"
        span_id = str(prediction["object_span_id"])
        raw_span = str(prediction["raw_span_text"])
        raw_span_sha = _validated_span_sha(prediction, raw_span=raw_span)
        description = str(prediction["description"])
        coord_bins = tuple(int(value) for value in prediction["coord_bins"])
        parser_bbox = _bbox_tuple(prediction["bbox"])
        category = resolve_category(description)
        if category is None or category.name != description:
            records.append(
                ResultReplayRecord(
                    result_id=result_id,
                    parser_object_span_id=span_id,
                    parser_generated_order=_optional_int(prediction.get("generated_order")),
                    raw_span_text=raw_span,
                    raw_span_sha256=raw_span_sha,
                    char_start=int(prediction["char_start"]),
                    char_end=int(prediction["char_end"]),
                    coord_bins=coord_bins,
                    parser_canvas_bbox=parser_bbox,
                    parsed_description=description,
                    canonical_category_name=None if category is None else category.name,
                    official_category_id=None if category is None else category.category_id,
                    class_decision="rejected",
                    reject_reason="unsupported_or_noncanonical_coco80_class",
                    inverse_mapping=None,
                    final_norm1000_bbox=None,
                )
            )
            continue
        if isinstance(category.category_id, bool) or category.category_id <= 0:
            raise InferenceResultContractError(
                "category resolver returned an invalid official sparse category ID"
            )
        try:
            inverse = transform.inverse_canvas_bbox(parser_bbox)
        except RoiTransformError as exc:
            records.append(
                ResultReplayRecord(
                    result_id=result_id,
                    parser_object_span_id=span_id,
                    parser_generated_order=_optional_int(prediction.get("generated_order")),
                    raw_span_text=raw_span,
                    raw_span_sha256=raw_span_sha,
                    char_start=int(prediction["char_start"]),
                    char_end=int(prediction["char_end"]),
                    coord_bins=coord_bins,
                    parser_canvas_bbox=parser_bbox,
                    parsed_description=description,
                    canonical_category_name=category.name,
                    official_category_id=category.category_id,
                    class_decision="accepted",
                    reject_reason=exc.code,
                    inverse_mapping=None,
                    final_norm1000_bbox=None,
                )
            )
            continue
        records.append(
            ResultReplayRecord(
                result_id=result_id,
                parser_object_span_id=span_id,
                parser_generated_order=_optional_int(prediction.get("generated_order")),
                raw_span_text=raw_span,
                raw_span_sha256=raw_span_sha,
                char_start=int(prediction["char_start"]),
                char_end=int(prediction["char_end"]),
                coord_bins=coord_bins,
                parser_canvas_bbox=parser_bbox,
                parsed_description=description,
                canonical_category_name=category.name,
                official_category_id=category.category_id,
                class_decision="accepted",
                reject_reason=None,
                inverse_mapping=inverse,
                final_norm1000_bbox=inverse.norm1000_bbox,
            )
        )
        regions.append(
            AcceptedInferenceRegion(
                result_id=result_id,
                category_name=category.name,
                category_id=category.category_id,
                norm1000_bbox=inverse.norm1000_bbox,
                request_id=target.request_id,
                parser_object_span_id=span_id,
            )
        )

    for index, dropped in enumerate(parse_row.dropped_predictions):
        raw_span = str(dropped.get("raw_span_text", dropped.get("raw_text", "")))
        raw_span_sha = _validated_span_sha(dropped, raw_span=raw_span)
        records.append(
            ResultReplayRecord(
                result_id=f"{target.request_id}:parser-drop-{index}",
                parser_object_span_id=str(
                    dropped.get("object_span_id", f"parser-drop-{index}")
                ),
                parser_generated_order=_optional_int(dropped.get("generated_order")),
                raw_span_text=raw_span,
                raw_span_sha256=raw_span_sha,
                char_start=int(dropped.get("char_start", 0)),
                char_end=int(dropped.get("char_end", len(raw_span))),
                coord_bins=(),
                parser_canvas_bbox=None,
                parsed_description=None,
                canonical_category_name=None,
                official_category_id=None,
                class_decision="not_parsed",
                reject_reason=f"parser:{dropped.get('reason', 'dropped')}",
                inverse_mapping=None,
                final_norm1000_bbox=None,
            )
        )

    parser_drops = len(parse_row.dropped_predictions)
    mapping_or_class_drops = len(parse_row.predictions) - len(regions)
    rejected_count = parser_drops + mapping_or_class_drops
    if parse_row.parse_status in {"unsupported_format", "all_spans_dropped"}:
        outcome = Outcome(parse_row.parse_status)
        regions = []
    elif not parse_row.predictions and not parse_row.dropped_predictions:
        outcome = Outcome.EMPTY
    elif not regions:
        outcome = Outcome.ALL_REJECTED
    elif rejected_count:
        outcome = Outcome.ACCEPTED_WITH_DROPS
    else:
        outcome = Outcome.ACCEPTED
    insertion_payload = (
        DirectInsertionPayload(target=target, regions=tuple(regions)) if regions else None
    )
    return ClassifiedInferenceResult(
        target=target,
        parser_status=parse_row.parse_status,
        outcome=outcome,
        raw_response_text=raw_response_text,
        raw_response_sha256=_sha256_text(raw_response_text),
        parse_row_sha256=parse_row_sha256,
        parsed_count=len(parse_row.predictions),
        inserted_count=len(regions),
        rejected_count=rejected_count,
        records=tuple(records),
        insertion_payload=insertion_payload,
    )


@dataclass(frozen=True)
class BindingDecision:
    status: str
    payload: DirectInsertionPayload | None
    mismatches: tuple[str, ...]


def bind_for_insertion(
    result: ClassifiedInferenceResult, current: CurrentTarget
) -> BindingDecision:
    """Revalidate the frozen target immediately before any annotation mutation."""

    expected = result.target.binding_payload()
    observed = current.binding_payload()
    mismatches = tuple(
        field for field in expected if expected[field] != observed.get(field)
    )
    if mismatches:
        return BindingDecision(
            status="abandoned_before_insertion", payload=None, mismatches=mismatches
        )
    return BindingDecision(
        status="bound",
        payload=result.insertion_payload,
        mismatches=(),
    )


def finalize_region_links(
    result: ClassifiedInferenceResult,
    current: CurrentTarget,
    *,
    region_links: Mapping[str, str],
) -> ClassifiedInferenceResult:
    """Attach exact post-insertion region links after a fresh target binding."""

    decision = bind_for_insertion(result, current)
    if decision.status != "bound" or decision.payload is None:
        raise InferenceResultContractError(
            "cannot finalize region links for an abandoned or empty insertion"
        )
    expected_ids = {region.result_id for region in decision.payload.regions}
    if set(region_links) != expected_ids:
        raise InferenceResultContractError(
            "region link keys must exactly match every inserted result"
        )
    normalized: dict[str, str] = {}
    for result_id, link in region_links.items():
        if (
            not isinstance(link, str)
            or not link
            or any(character in link for character in ("?", "#", "@"))
        ):
            raise InferenceResultContractError(
                "region links must be non-empty credential-free stable identifiers"
            )
        normalized[result_id] = link
    if len(set(normalized.values())) != len(normalized):
        raise InferenceResultContractError("inserted region links must be unique")
    regions = tuple(
        replace(region, region_link=normalized[region.result_id])
        for region in decision.payload.regions
    )
    records = tuple(
        replace(record, region_link=normalized[record.result_id])
        if record.result_id in normalized
        else record
        for record in result.records
    )
    return replace(
        result,
        records=records,
        insertion_payload=DirectInsertionPayload(target=result.target, regions=regions),
    )


class RequestState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    ACCEPTED = "accepted"
    ACCEPTED_WITH_DROPS = "accepted_with_drops"
    EMPTY = "empty"
    ALL_REJECTED = "all_rejected"
    RESPONSE_FAILURE = "response_failure"
    PROFILE_FAILURE = "profile_failure"
    TRANSPORT_FAILURE = "transport_failure"
    RUNTIME_FAILURE = "runtime_failure"
    TIMEOUT_FAILURE = "timeout_failure"
    ABANDONED_BEFORE_INSERTION = "abandoned_before_insertion"

    @property
    def terminal(self) -> bool:
        return self not in {
            RequestState.PENDING,
            RequestState.RUNNING,
            RequestState.CANCELLING,
        }


@dataclass(frozen=True)
class StateTransition:
    from_state: RequestState
    to_state: RequestState
    at_seconds: float
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_state.value,
            "to": self.to_state.value,
            "at_seconds": self.at_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RequestLifecycle:
    state: RequestState = RequestState.PENDING
    transitions: tuple[StateTransition, ...] = ()

    def transition(
        self,
        to_state: RequestState,
        *,
        at_seconds: float,
        reason: str | None = None,
    ) -> "RequestLifecycle":
        if not math.isfinite(at_seconds) or at_seconds < 0:
            raise InferenceResultContractError(
                "transition timestamp must be finite and non-negative"
            )
        if self.state.terminal:
            raise InferenceResultContractError("terminal request state cannot transition")
        allowed = _ALLOWED_TRANSITIONS[self.state]
        if to_state not in allowed:
            raise InferenceResultContractError(
                f"invalid request transition: {self.state.value} -> {to_state.value}"
            )
        if to_state in _REASON_REQUIRED_STATES and not reason:
            raise InferenceResultContractError(
                f"{to_state.value} transition requires an explicit reason"
            )
        if self.transitions and at_seconds < self.transitions[-1].at_seconds:
            raise InferenceResultContractError("transition timestamps must be monotonic")
        event = StateTransition(self.state, to_state, at_seconds, reason)
        return RequestLifecycle(
            state=to_state,
            transitions=(*self.transitions, event),
        )


_RESULT_STATES = {
    RequestState.ACCEPTED,
    RequestState.ACCEPTED_WITH_DROPS,
    RequestState.EMPTY,
    RequestState.ALL_REJECTED,
    RequestState.RESPONSE_FAILURE,
}
_FAILURE_STATES = {
    RequestState.PROFILE_FAILURE,
    RequestState.TRANSPORT_FAILURE,
    RequestState.RUNTIME_FAILURE,
    RequestState.TIMEOUT_FAILURE,
    RequestState.ABANDONED_BEFORE_INSERTION,
}
_REASON_REQUIRED_STATES = {
    RequestState.CANCELLING,
    RequestState.CANCELLED,
    *_FAILURE_STATES,
}
_ALLOWED_TRANSITIONS = {
    RequestState.PENDING: {
        RequestState.RUNNING,
        RequestState.PROFILE_FAILURE,
        RequestState.TRANSPORT_FAILURE,
        RequestState.CANCELLED,
        RequestState.ABANDONED_BEFORE_INSERTION,
    },
    RequestState.RUNNING: {
        RequestState.CANCELLING,
        *_RESULT_STATES,
        RequestState.PROFILE_FAILURE,
        RequestState.TRANSPORT_FAILURE,
        RequestState.RUNTIME_FAILURE,
        RequestState.ABANDONED_BEFORE_INSERTION,
    },
    RequestState.CANCELLING: {
        RequestState.CANCELLED,
        RequestState.TIMEOUT_FAILURE,
        RequestState.RUNTIME_FAILURE,
        RequestState.ABANDONED_BEFORE_INSERTION,
    },
}


def terminal_state_for_result(result: ClassifiedInferenceResult) -> RequestState:
    if result.outcome is Outcome.ACCEPTED:
        return RequestState.ACCEPTED
    if result.outcome is Outcome.ACCEPTED_WITH_DROPS:
        return RequestState.ACCEPTED_WITH_DROPS
    if result.outcome is Outcome.EMPTY:
        return RequestState.EMPTY
    if result.outcome is Outcome.ALL_REJECTED:
        return RequestState.ALL_REJECTED
    return RequestState.RESPONSE_FAILURE


@dataclass(frozen=True)
class InferenceAttemptReceipt:
    target: RequestTarget
    lifecycle: RequestLifecycle
    profile_receipt: Mapping[str, Any]
    transform_receipt: Mapping[str, Any]
    result: ClassifiedInferenceResult | None
    failure_stage: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if not self.lifecycle.state.terminal:
            raise InferenceResultContractError(
                "completed inference receipt requires a terminal lifecycle"
            )
        if self.result is not None and self.result.target != self.target:
            raise InferenceResultContractError(
                "receipt result target does not match receipt request target"
            )
        profile_receipt = _validate_profile_receipt(self.profile_receipt)
        if profile_receipt["profile_fingerprint"] != self.target.profile_fingerprint:
            raise InferenceResultContractError(
                "receipt profile does not match the frozen request profile"
            )
        transform_receipt = _validate_transform_receipt(self.transform_receipt)
        if _sha256_json(transform_receipt) != self.target.transform_fingerprint:
            raise InferenceResultContractError(
                "receipt transform does not match the frozen request transform"
            )
        object.__setattr__(self, "profile_receipt", _deep_freeze(profile_receipt))
        object.__setattr__(self, "transform_receipt", _deep_freeze(transform_receipt))
        if self.lifecycle.state in _RESULT_STATES:
            if (
                self.result is None
                or terminal_state_for_result(self.result) is not self.lifecycle.state
            ):
                raise InferenceResultContractError(
                    "receipt lifecycle does not match the classified parser outcome"
                )
            if any(
                value is not None
                for value in (
                    self.failure_stage,
                    self.failure_code,
                    self.failure_message,
                )
            ):
                raise InferenceResultContractError(
                    "result terminal receipt cannot contain failure fields"
                )
            _validate_result_terminal_payload(
                self.result,
                transform_receipt=transform_receipt,
            )
        if self.lifecycle.state in _FAILURE_STATES | {RequestState.CANCELLED}:
            if self.result is not None:
                raise InferenceResultContractError(
                    "failure, cancellation, or abandonment receipt cannot contain a result"
                )
            if not self.failure_stage or not self.failure_code:
                raise InferenceResultContractError(
                    "failure/cancellation receipt requires stage and code"
                )
            _safe_receipt_identifier(self.failure_stage, field="failure_stage")
            _safe_receipt_identifier(self.failure_code, field="failure_code")
        elif self.result is None:
            raise InferenceResultContractError(
                "successful terminal receipt requires a classified result"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.target.to_receipt_dict(),
            "terminal_status": self.lifecycle.state.value,
            "state_transitions": [
                transition.to_dict() for transition in self.lifecycle.transitions
            ],
            "profile": _deep_thaw(self.profile_receipt),
            "transform": _deep_thaw(self.transform_receipt),
            "result": None if self.result is None else self.result.to_receipt_dict(),
            "failure": (
                None
                if self.failure_stage is None
                else {
                    "stage": self.failure_stage,
                    "code": self.failure_code,
                    "annotation_mutated": False,
                }
            ),
        }


_PROFILE_RECEIPT_KEYS = {
    "schema_version",
    "profile_name",
    "profile_fingerprint",
    "endpoint",
    "artifacts",
    "identity_fingerprints",
    "processor",
    "deadline_seconds",
}
_ARTIFACT_RECEIPT_KEYS = {
    "role",
    "path",
    "kind",
    "sha256",
    "file_count",
    "total_bytes",
}
_ALLOWED_PROFILE_ARTIFACT_ROLES = {
    "base_weights",
    "model_config",
    "tokenizer",
    "processor",
    "adapter",
    "checkpoint",
    "embedding_delta",
}
_REQUIRED_PROFILE_ARTIFACT_ROLES = {
    "base_weights",
    "model_config",
    "tokenizer",
    "processor",
}
_IDENTITY_FINGERPRINT_KEYS = {
    "resolved_config",
    "prompt_policy",
    "parser",
    "adapter",
    "transform",
    "transformers",
    "processor_kwargs",
    "runtime",
}
_PROCESSOR_RECEIPT_KEYS = {
    "factor",
    "default_canvas",
    "axis_bounds",
    "max_total_pixels",
    "do_resize",
}
_TRANSFORM_RECEIPT_KEYS = {
    "transform_id",
    "source_size",
    "roi_percent_xywh",
    "roi_float_edges",
    "clipped_float_edges",
    "crop_edges",
    "crop_edge_convention",
    "canvas_size",
    "realized_size",
    "scale_x",
    "scale_y",
    "padding",
    "resampler",
    "pad_value_rgb",
    "pixel_edge_convention",
    "processor_kwargs",
}


def _validate_profile_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _strict_json_mapping(value, field="profile_receipt")
    if set(payload) != _PROFILE_RECEIPT_KEYS:
        raise InferenceResultContractError(
            "profile receipt fields are incomplete or non-allowlisted"
        )
    for field in ("schema_version", "profile_name"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise InferenceResultContractError(
                f"profile receipt {field} must be non-empty text"
            )
    fingerprint = payload.get("profile_fingerprint")
    _require_sha256(fingerprint, field="profile_receipt.profile_fingerprint")
    endpoint = payload["endpoint"]
    if not isinstance(endpoint, str):
        raise InferenceResultContractError("profile receipt endpoint must be text")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise InferenceResultContractError(
            "profile receipt endpoint must be credential-free HTTP(S)"
        )
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise InferenceResultContractError(
            "profile receipt artifacts must be a non-empty list"
        )
    roles: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_RECEIPT_KEYS:
            raise InferenceResultContractError("profile receipt artifacts must be a list")
        role = artifact["role"]
        if (
            not isinstance(role, str)
            or role not in _ALLOWED_PROFILE_ARTIFACT_ROLES
            or role in roles
        ):
            raise InferenceResultContractError(
                "profile receipt artifact roles must be unique non-empty text"
            )
        roles.add(role)
        _require_sha256(artifact["sha256"], field="artifact.sha256")
        path = artifact["path"]
        if not isinstance(path, str) or not path.startswith("/"):
            raise InferenceResultContractError(
                "profile receipt artifact path must be an absolute local path"
            )
    if not _REQUIRED_PROFILE_ARTIFACT_ROLES <= roles:
        raise InferenceResultContractError(
            "profile receipt is missing required artifact identities"
        )
    identities = payload["identity_fingerprints"]
    if not isinstance(identities, dict) or set(identities) != _IDENTITY_FINGERPRINT_KEYS:
        raise InferenceResultContractError(
            "profile receipt identity fingerprints are incomplete or unknown"
        )
    for field, identity_fingerprint in identities.items():
        _require_sha256(
            identity_fingerprint, field=f"identity_fingerprints.{field}"
        )
    processor = payload["processor"]
    if not isinstance(processor, dict) or set(processor) != _PROCESSOR_RECEIPT_KEYS:
        raise InferenceResultContractError(
            "profile receipt processor fields are not allowlisted"
        )
    if processor["do_resize"] is not False:
        raise InferenceResultContractError(
            "profile receipt processor must attest do_resize=false"
        )
    for field in ("factor", "max_total_pixels"):
        value = processor[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InferenceResultContractError(
                f"profile receipt processor {field} must be a positive integer"
            )
    for field in ("default_canvas", "axis_bounds"):
        value = processor[field]
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in value
            )
        ):
            raise InferenceResultContractError(
                f"profile receipt processor {field} must contain two positive integers"
            )
    deadline = payload["deadline_seconds"]
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
        or deadline <= 0
    ):
        raise InferenceResultContractError(
            "profile receipt deadline_seconds must be finite and positive"
        )
    return payload


def _validate_transform_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _strict_json_mapping(value, field="transform_receipt")
    if set(payload) != _TRANSFORM_RECEIPT_KEYS:
        raise InferenceResultContractError(
            "transform receipt fields are incomplete or not allowlisted"
        )
    padding = payload["padding"]
    if not isinstance(padding, dict) or set(padding) != {
        "left",
        "top",
        "right",
        "bottom",
    }:
        raise InferenceResultContractError("transform padding receipt is invalid")
    kwargs = payload["processor_kwargs"]
    if kwargs != {"do_resize": False}:
        raise InferenceResultContractError(
            "transform receipt must attest only do_resize=false"
        )
    expected_constants = {
        "transform_id": ROI_TRANSFORM_ID,
        "crop_edge_convention": CROP_EDGE_CONVENTION,
        "resampler": RESAMPLER,
        "pad_value_rgb": list(PAD_VALUE_RGB),
        "pixel_edge_convention": PIXEL_EDGE_CONVENTION,
    }
    for field, expected in expected_constants.items():
        if payload[field] != expected:
            raise InferenceResultContractError(
                f"transform receipt {field} does not match the fixed transform contract"
            )
    return payload


def _transform_from_receipt(
    transform_receipt: Mapping[str, Any],
) -> RoiLetterboxTransform:
    source_size = transform_receipt.get("source_size")
    canvas_size = transform_receipt.get("canvas_size")
    roi_percent = transform_receipt.get("roi_percent_xywh")
    if (
        not isinstance(source_size, list)
        or len(source_size) != 2
        or not isinstance(canvas_size, list)
        or len(canvas_size) != 2
        or not isinstance(roi_percent, list)
        or len(roi_percent) != 4
    ):
        raise InferenceResultContractError(
            "transform receipt cannot reconstruct the immutable ROI transform"
        )
    try:
        transform = RoiLetterboxTransform.from_label_studio_roi(
            source_width=source_size[0],
            source_height=source_size[1],
            roi=roi_percent,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
    except (RoiTransformError, TypeError, ValueError) as exc:
        raise InferenceResultContractError(
            "transform receipt cannot reconstruct the immutable ROI transform"
        ) from exc
    if _canonical_json(transform.to_receipt_dict()) != _canonical_json(
        transform_receipt
    ):
        raise InferenceResultContractError(
            "transform receipt does not match the reconstructed immutable ROI transform"
        )
    return transform


def _validate_result_terminal_payload(
    result: ClassifiedInferenceResult,
    *,
    transform_receipt: Mapping[str, Any],
) -> None:
    if result.parsed_count < 0 or result.inserted_count < 0 or result.rejected_count < 0:
        raise InferenceResultContractError("result terminal counts cannot be negative")
    if result.raw_response_sha256 != _sha256_text(result.raw_response_text):
        raise InferenceResultContractError(
            "result terminal raw response hash does not match raw response"
        )
    _validate_result_parser_replay(result, transform_receipt=transform_receipt)
    if result.outcome in {Outcome.ACCEPTED, Outcome.ACCEPTED_WITH_DROPS}:
        payload = result.insertion_payload
        if payload is None or result.inserted_count <= 0:
            raise InferenceResultContractError(
                "accepted receipt requires a non-empty insertion payload"
            )
        if len(payload.regions) != result.inserted_count:
            raise InferenceResultContractError(
                "accepted receipt insertion count does not match regions"
            )
        if payload.target != result.target:
            raise InferenceResultContractError(
                "accepted receipt insertion target does not match result target"
            )
        links: dict[str, str] = {}
        for region in payload.regions:
            if not isinstance(region.region_link, str) or not region.region_link:
                raise InferenceResultContractError(
                    "accepted receipt requires finalized non-null region links"
                )
            if region.result_id in links:
                raise InferenceResultContractError(
                    "accepted receipt contains duplicate result IDs"
                )
            links[region.result_id] = region.region_link
        if len(set(links.values())) != len(links):
            raise InferenceResultContractError(
                "accepted receipt contains duplicate region links"
            )
        mapped_records = {
            record.result_id: record
            for record in result.records
            if record.final_norm1000_bbox is not None
        }
        if set(mapped_records) != set(links):
            raise InferenceResultContractError(
                "accepted receipt regions do not match mapped replay records"
            )
        for record in result.records:
            if record.result_id in links:
                if record.region_link != links[record.result_id]:
                    raise InferenceResultContractError(
                        "accepted receipt replay record has no matching finalized region link"
                    )
            elif record.region_link is not None:
                raise InferenceResultContractError(
                    "rejected parser record cannot carry an inserted region link"
                )
    else:
        if result.insertion_payload is not None or result.inserted_count != 0:
            raise InferenceResultContractError(
                "non-inserting result terminal cannot contain an insertion payload"
            )
        if any(record.region_link is not None for record in result.records):
            raise InferenceResultContractError(
                "non-inserting result terminal cannot contain region links"
            )


def _strict_json_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InferenceResultContractError(f"{field} must be a mapping")
    try:
        copied = json.loads(_canonical_json(value))
    except json.JSONDecodeError as exc:  # pragma: no cover - canonical output parses.
        raise InferenceResultContractError(f"{field} is not strict JSON") from exc
    if not isinstance(copied, dict):
        raise InferenceResultContractError(f"{field} must be a mapping")
    return copied


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InferenceResultContractError(f"{field} must be a lowercase SHA-256")
    return value


def _safe_receipt_identifier(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(not (character.isalnum() or character in "_.:-") for character in value)
    ):
        raise InferenceResultContractError(
            f"{field} must be a short credential-free identifier"
        )
    return value


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _bind_parse_row_to_raw(
    *,
    parse_row: ParseRow,
    raw_response_text: str,
    target: RequestTarget,
    transform: RoiLetterboxTransform,
) -> str:
    replayed = parse_compact_object_box_closed(
        raw_response_text,
        row_id=target.request_id,
        row_index=parse_row.row_index,
        image_width=transform.canvas_width,
        image_height=transform.canvas_height,
    )
    observed = {
        "artifact": parse_row.to_artifact_dict(),
        "diagnostics": parse_row.diagnostics,
    }
    expected = {
        "artifact": replayed.to_artifact_dict(),
        "diagnostics": replayed.diagnostics,
    }
    observed_json = _canonical_json(observed)
    if observed_json != _canonical_json(expected):
        raise InferenceResultContractError(
            "ParseRow is not the exact current-parser result for raw_response_text"
        )
    for payload in (*parse_row.predictions, *parse_row.dropped_predictions):
        _validate_raw_span_binding(payload, raw_response_text=raw_response_text)
        for evidence_field in ("schema_spans", "coord_token_spans"):
            spans = payload.get(evidence_field, [])
            if not isinstance(spans, list):
                raise InferenceResultContractError(
                    f"parser {evidence_field} must be a list"
                )
            for span in spans:
                if not isinstance(span, Mapping):
                    raise InferenceResultContractError(
                        f"parser {evidence_field} entries must be mappings"
                    )
                _validate_raw_span_binding(
                    {
                        "char_start": span.get("char_start"),
                        "char_end": span.get("char_end"),
                        "raw_span_text": span.get("text"),
                        "raw_span_sha256": _sha256_text(str(span.get("text", ""))),
                    },
                    raw_response_text=raw_response_text,
                )
    return _parse_row_binding_sha(parse_row, raw_response_text=raw_response_text)


def _validate_result_parser_replay(
    result: ClassifiedInferenceResult,
    *,
    transform_receipt: Mapping[str, Any],
) -> None:
    transform = _transform_from_receipt(transform_receipt)
    replayed = parse_compact_object_box_closed(
        result.raw_response_text,
        row_id=result.target.request_id,
        row_index=0,
        image_width=transform.canvas_width,
        image_height=transform.canvas_height,
    )
    if result.parser_status != replayed.parse_status:
        raise InferenceResultContractError(
            "result terminal parser status does not match raw response replay"
        )
    expected_parse_sha = _parse_row_binding_sha(
        replayed,
        raw_response_text=result.raw_response_text,
    )
    if result.parse_row_sha256 != expected_parse_sha:
        raise InferenceResultContractError(
            "result terminal parse row hash does not match raw response replay"
        )
    if result.parsed_count != len(replayed.predictions):
        raise InferenceResultContractError(
            "result terminal parsed count does not match raw response replay"
        )

    for record in result.records:
        if not isinstance(record, ResultReplayRecord):
            raise InferenceResultContractError(
                "result terminal replay records must use the canonical record type"
            )
        _require_sha256(
            record.raw_span_sha256,
            field="result replay raw_span_sha256",
        )
        _validate_raw_span_binding(
            {
                "char_start": record.char_start,
                "char_end": record.char_end,
                "raw_span_text": record.raw_span_text,
                "raw_span_sha256": record.raw_span_sha256,
            },
            raw_response_text=result.raw_response_text,
        )
    expected = classify_parser_result(
        parse_row=replayed,
        raw_response_text=result.raw_response_text,
        target=result.target,
        transform=transform,
    )
    observed = _without_finalized_region_links(result)
    if observed != expected:
        raise InferenceResultContractError(
            "result terminal replay records and classification semantics do not match "
            "the attested raw response and ROI transform"
        )


def _parse_row_binding_sha(parse_row: ParseRow, *, raw_response_text: str) -> str:
    binding_payload = _without_row_index(
        {
            "artifact": parse_row.to_artifact_dict(),
            "diagnostics": parse_row.diagnostics,
        }
    )
    binding_json = _canonical_json(binding_payload)
    return hashlib.sha256(
        (_sha256_text(raw_response_text) + "\0" + binding_json).encode("utf-8")
    ).hexdigest()


def _without_row_index(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_row_index(item)
            for key, item in value.items()
            if key != "row_index"
        }
    if isinstance(value, (list, tuple)):
        return [_without_row_index(item) for item in value]
    return value


def _without_finalized_region_links(
    result: ClassifiedInferenceResult,
) -> ClassifiedInferenceResult:
    records = tuple(replace(record, region_link=None) for record in result.records)
    payload = result.insertion_payload
    if payload is not None:
        if not isinstance(payload, DirectInsertionPayload) or any(
            not isinstance(region, AcceptedInferenceRegion) for region in payload.regions
        ):
            raise InferenceResultContractError(
                "result terminal insertion payload must use canonical region records"
            )
        payload = replace(
            payload,
            regions=tuple(
                replace(region, region_link=None) for region in payload.regions
            ),
        )
    return replace(result, records=records, insertion_payload=payload)


def _validate_raw_span_binding(
    payload: Mapping[str, Any], *, raw_response_text: str
) -> None:
    start = payload.get("char_start")
    end = payload.get("char_end")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end < start
        or end > len(raw_response_text)
    ):
        raise InferenceResultContractError("parser raw span offsets are invalid")
    span_text = payload.get("raw_span_text", payload.get("raw_text"))
    if not isinstance(span_text, str):
        raise InferenceResultContractError("parser raw span text must be text")
    if raw_response_text[start:end] != span_text:
        raise InferenceResultContractError(
            "parser raw span offsets/content do not match raw_response_text"
        )
    _validated_span_sha(payload, raw_span=span_text)


def _canonical_json(payload: Any) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise InferenceResultContractError(
            "parser replay payload is not strict JSON"
        ) from exc


def _bbox_tuple(value: Any) -> tuple[float, float, float, float]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InferenceResultContractError("parser bbox must contain four numbers") from exc
    if len(result) != 4 or any(not math.isfinite(item) for item in result):
        raise InferenceResultContractError("parser bbox must contain four finite numbers")
    return result


def _validate_parser_status_shape(parse_row: ParseRow) -> None:
    predictions = len(parse_row.predictions)
    dropped = len(parse_row.dropped_predictions)
    expected = {
        "accepted": predictions > 0 and dropped == 0,
        "accepted_with_drops": predictions > 0 and dropped > 0,
        "empty": predictions == 0 and dropped == 0,
        "all_spans_dropped": predictions == 0 and dropped > 0,
        "unsupported_format": predictions == 0 and dropped > 0,
    }
    if not expected[parse_row.parse_status]:
        raise InferenceResultContractError(
            "parser status contradicts its accepted/dropped result counts"
        )


def _validated_span_sha(payload: Mapping[str, Any], *, raw_span: str) -> str:
    observed = _sha256_text(raw_span)
    expected = payload.get("raw_span_sha256")
    if expected is not None and expected != observed:
        raise InferenceResultContractError("parser raw span hash does not match raw span")
    return observed


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

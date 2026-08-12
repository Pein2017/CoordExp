"""No-padding pack planning for encoded examples."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from src.common.errors import PackingContractError


PACK_PLAN_SCHEMA = "coordexp-swift-pack-plan"
PACK_PLAN_SCHEMA_VERSION = 5
PACK_PLAN_CURSOR_SCHEMA = "coordexp-swift-pack-plan-cursor"
PACK_PLAN_CURSOR_SCHEMA_VERSION = 3
PACK_PLAN_AUTHENTICATION_SCHEMA = "coordexp-swift-pack-plan-authentication-v1"
PACK_PLAN_CURSOR_AUTHENTICATION_SCHEMA = (
    "coordexp-swift-pack-plan-cursor-authentication-v1"
)
PACK_PLAN_CANONICALIZATION = "json-sort-keys-ascii-no-nan-compact-v1"
PACK_PLAN_PREFIX_DIGEST_SCHEMA = "coordexp-swift-pack-plan-prefix-chain-v1"
PACK_PLAN_STREAM_RECEIPT_SCHEMA = "coordexp-swift-pack-plan-stream-receipt"
PACK_PLAN_STREAM_RECEIPT_SCHEMA_VERSION = 2
PACK_PLAN_STREAM_AUTHENTICATION_SCHEMA = (
    "coordexp-swift-pack-plan-stream-authentication-v1"
)
DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET = 65_536
DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET = 1_024
DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET = 4_194_304
DEFAULT_PACK_PLAN_CONTENT_NODE_BUDGET = 1_000_000
DEFAULT_PACK_PLAN_CONTENT_BYTE_BUDGET = 536_870_912

SOURCE_ORDER_NEXT_FIT = "source_order_next_fit"
WINDOW_BINPACK = "window_binpack"
ONLINE_WINDOW_BINPACK = "online_window_binpack"

_ALGORITHM_VERSIONS = {
    SOURCE_ORDER_NEXT_FIT: "coordexp-swift-source-order-next-fit-v2",
    WINDOW_BINPACK: "coordexp-swift-window-binpack-v2",
    ONLINE_WINDOW_BINPACK: "coordexp-swift-online-window-binpack-v3",
}
_TIE_BREAKERS = {
    SOURCE_ORDER_NEXT_FIT: "source_ordinal_v1",
    WINDOW_BINPACK: (
        "length_desc_seeded_identity_then_ordinal_best_fit_earliest_bin_v1"
    ),
    ONLINE_WINDOW_BINPACK: ("source_anchor_best_fit_seeded_identity_then_ordinal_v1"),
}
_WORKER_COUNT_DISPOSITION = "semantic_pending_upstream_materialization_equality"
_WORKER_COUNT_INTEGRATION_REQUIREMENT = (
    "canonical_concurrent_encoded_materialization_and_complete_plan_equality"
)
_COMPLETE_PLAN_SCOPE = "complete_plan"
_RESUME_FRAGMENT_SCOPE = "resume_fragment"
_INITIAL_SOURCE_PREFIX_SHA256 = hashlib.sha256(
    f"{PACK_PLAN_PREFIX_DIGEST_SCHEMA}:source".encode("ascii")
).hexdigest()
_INITIAL_EMITTED_PREFIX_SHA256 = hashlib.sha256(
    f"{PACK_PLAN_PREFIX_DIGEST_SCHEMA}:emitted".encode("ascii")
).hexdigest()
_INITIAL_FRAGMENT_CHAIN_SHA256 = hashlib.sha256(
    f"{PACK_PLAN_PREFIX_DIGEST_SCHEMA}:fragments".encode("ascii")
).hexdigest()


@dataclass(frozen=True)
class PackedSegment:
    pack_index: int
    segment_index: int
    example_index: int
    example_id: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_artifact_dict(self) -> dict[str, int | str]:
        return {
            "pack_index": self.pack_index,
            "segment_index": self.segment_index,
            "example_index": self.example_index,
            "example_id": self.example_id,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }


@dataclass(frozen=True)
class PackedSequence:
    pack_index: int
    input_ids: tuple[int, ...]
    segments: tuple[PackedSegment, ...]
    global_max_length: int

    @property
    def length(self) -> int:
        return len(self.input_ids)

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "pack_index": self.pack_index,
            "length": self.length,
            "global_max_length": self.global_max_length,
            "padding_tokens": 0,
            "segment_count": len(self.segments),
            "segments": [segment.to_artifact_dict() for segment in self.segments],
        }


def plan_packed_sequences(
    encoded_examples: list[Any] | tuple[Any, ...],
    *,
    global_max_length: int,
) -> tuple[PackedSequence, ...]:
    if global_max_length <= 0:
        raise PackingContractError(
            "packing.global_max_length must be positive",
            code="packing.global_max_length",
            context={"global_max_length": global_max_length},
        )

    packs: list[PackedSequence] = []
    current_ids: list[int] = []
    current_segments: list[PackedSegment] = []
    pack_index = 0

    for example_index, example in enumerate(encoded_examples):
        example_id = _example_id(example)
        input_ids = _input_ids(example, example_id=example_id)
        input_length = len(input_ids)
        if input_length == 0:
            raise PackingContractError(
                "encoded example must contain at least one token before packing",
                code="packing.example_empty",
                context={"example_id": example_id, "example_index": example_index},
            )
        if input_length > global_max_length:
            raise PackingContractError(
                "encoded example exceeds packing.global_max_length",
                code="packing.example_too_long",
                context={
                    "example_id": example_id,
                    "example_index": example_index,
                    "input_length": input_length,
                    "global_max_length": global_max_length,
                },
            )

        if current_ids and len(current_ids) + input_length > global_max_length:
            packs.append(
                _commit_pack(
                    pack_index=pack_index,
                    input_ids=current_ids,
                    segments=current_segments,
                    global_max_length=global_max_length,
                )
            )
            pack_index += 1
            current_ids = []
            current_segments = []

        start = len(current_ids)
        current_ids.extend(input_ids)
        current_segments.append(
            PackedSegment(
                pack_index=pack_index,
                segment_index=len(current_segments),
                example_index=example_index,
                example_id=example_id,
                start=start,
                end=start + input_length,
            )
        )

    if current_ids:
        packs.append(
            _commit_pack(
                pack_index=pack_index,
                input_ids=current_ids,
                segments=current_segments,
                global_max_length=global_max_length,
            )
        )
    return tuple(packs)


def _commit_pack(
    *,
    pack_index: int,
    input_ids: list[int],
    segments: list[PackedSegment],
    global_max_length: int,
) -> PackedSequence:
    if not segments:
        raise PackingContractError(
            "cannot commit an empty pack",
            code="packing.empty_pack",
            context={"pack_index": pack_index},
        )
    if len(input_ids) > global_max_length:
        raise PackingContractError(
            "packed sequence exceeds packing.global_max_length",
            code="packing.pack_too_long",
            context={
                "pack_index": pack_index,
                "length": len(input_ids),
                "global_max_length": global_max_length,
            },
        )
    return PackedSequence(
        pack_index=pack_index,
        input_ids=tuple(input_ids),
        segments=tuple(segments),
        global_max_length=global_max_length,
    )


def _example_id(example: Any) -> str:
    value = getattr(example, "example_id", None)
    if not isinstance(value, str) or not value:
        raise PackingContractError(
            "encoded example must expose a non-empty example_id",
            code="packing.example_id",
            context={"value_type": type(value).__name__},
        )
    return value


def _input_ids(example: Any, *, example_id: str) -> tuple[int, ...]:
    value = getattr(example, "input_ids", None)
    if not isinstance(value, tuple):
        raise PackingContractError(
            "encoded example input_ids must be a tuple",
            code="packing.input_ids_shape",
            context={"example_id": example_id, "value_type": type(value).__name__},
        )
    try:
        return tuple(int(token_id) for token_id in value)
    except (TypeError, ValueError) as exc:
        raise PackingContractError(
            "encoded example input_ids must contain integer token ids",
            code="packing.input_ids_type",
            context={"example_id": example_id},
            cause=exc,
        ) from exc


@dataclass(frozen=True)
class PackPlanInput:
    """Atomic encoded-image identity consumed by a packing policy."""

    input_ordinal: int
    example_id: str
    encoded_length: int
    input_ids_sha256: str
    encoded_example_semantics_sha256: str
    intra_image_order_identity: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "input_ordinal": self.input_ordinal,
            "example_id": self.example_id,
            "encoded_length": self.encoded_length,
            "input_ids_sha256": self.input_ids_sha256,
            "encoded_example_semantics_sha256": (self.encoded_example_semantics_sha256),
            "intra_image_order_identity": self.intra_image_order_identity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PackPlanInput:
        data = _strict_mapping(
            payload,
            keys={
                "input_ordinal",
                "example_id",
                "encoded_length",
                "input_ids_sha256",
                "encoded_example_semantics_sha256",
                "intra_image_order_identity",
            },
            field="input",
        )
        item = cls(
            input_ordinal=_strict_int(data["input_ordinal"], field="input_ordinal"),
            example_id=_strict_string(data["example_id"], field="example_id"),
            encoded_length=_strict_int(data["encoded_length"], field="encoded_length"),
            input_ids_sha256=_strict_sha256(
                data["input_ids_sha256"], field="input_ids_sha256"
            ),
            encoded_example_semantics_sha256=_strict_sha256(
                data["encoded_example_semantics_sha256"],
                field="encoded_example_semantics_sha256",
            ),
            intra_image_order_identity=_strict_string(
                data["intra_image_order_identity"],
                field="intra_image_order_identity",
            ),
        )
        _validate_plan_input(item)
        return item


@dataclass(frozen=True)
class RejectedPackPlanInput:
    """Atomic input excluded from membership with a stable reason."""

    input_ordinal: int
    example_id: str
    encoded_length: int
    input_ids_sha256: str
    encoded_example_semantics_sha256: str
    intra_image_order_identity: str
    reason: str

    @classmethod
    def from_input(
        cls,
        item: PackPlanInput,
        *,
        reason: str,
    ) -> RejectedPackPlanInput:
        return cls(
            input_ordinal=item.input_ordinal,
            example_id=item.example_id,
            encoded_length=item.encoded_length,
            input_ids_sha256=item.input_ids_sha256,
            encoded_example_semantics_sha256=(item.encoded_example_semantics_sha256),
            intra_image_order_identity=item.intra_image_order_identity,
            reason=reason,
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "input_ordinal": self.input_ordinal,
            "example_id": self.example_id,
            "encoded_length": self.encoded_length,
            "input_ids_sha256": self.input_ids_sha256,
            "encoded_example_semantics_sha256": (self.encoded_example_semantics_sha256),
            "intra_image_order_identity": self.intra_image_order_identity,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RejectedPackPlanInput:
        data = _strict_mapping(
            payload,
            keys={
                "input_ordinal",
                "example_id",
                "encoded_length",
                "input_ids_sha256",
                "encoded_example_semantics_sha256",
                "intra_image_order_identity",
                "reason",
            },
            field="rejected_example",
        )
        item = cls(
            input_ordinal=_strict_int(data["input_ordinal"], field="input_ordinal"),
            example_id=_strict_string(data["example_id"], field="example_id"),
            encoded_length=_strict_int(data["encoded_length"], field="encoded_length"),
            input_ids_sha256=_strict_sha256(
                data["input_ids_sha256"], field="input_ids_sha256"
            ),
            encoded_example_semantics_sha256=_strict_sha256(
                data["encoded_example_semantics_sha256"],
                field="encoded_example_semantics_sha256",
            ),
            intra_image_order_identity=_strict_string(
                data["intra_image_order_identity"],
                field="intra_image_order_identity",
            ),
            reason=_strict_string(data["reason"], field="reason"),
        )
        _validate_plan_input(item)
        return item


@dataclass(frozen=True)
class PackPlanPack:
    """One pack membership; tuple order is the within-pack image order."""

    pack_index: int
    input_ordinals: tuple[int, ...]
    encoded_length: int
    global_max_length: int

    @property
    def utilization(self) -> float:
        return self.encoded_length / self.global_max_length

    @property
    def tail_waste(self) -> int:
        return self.global_max_length - self.encoded_length

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_index": self.pack_index,
            "input_ordinals": list(self.input_ordinals),
            "encoded_length": self.encoded_length,
            "global_max_length": self.global_max_length,
            "utilization": self.utilization,
            "tail_waste": self.tail_waste,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PackPlanPack:
        data = _strict_mapping(
            payload,
            keys={
                "pack_index",
                "input_ordinals",
                "encoded_length",
                "global_max_length",
                "utilization",
                "tail_waste",
            },
            field="pack",
        )
        ordinals = _strict_list(data["input_ordinals"], field="input_ordinals")
        pack = cls(
            pack_index=_strict_int(data["pack_index"], field="pack_index"),
            input_ordinals=tuple(
                _strict_int(value, field="input_ordinal") for value in ordinals
            ),
            encoded_length=_strict_int(data["encoded_length"], field="encoded_length"),
            global_max_length=_strict_int(
                data["global_max_length"], field="global_max_length"
            ),
        )
        _validate_plan_pack(pack)
        utilization = _strict_float(data["utilization"], field="utilization")
        tail_waste = _strict_int(data["tail_waste"], field="tail_waste")
        if not math.isclose(utilization, pack.utilization, rel_tol=0.0, abs_tol=0.0):
            _plan_error(
                "serialized pack utilization does not match lengths",
                code="packing.pack_plan_utilization",
                context={"pack_index": pack.pack_index},
            )
        if tail_waste != pack.tail_waste:
            _plan_error(
                "serialized pack tail waste does not match lengths",
                code="packing.pack_plan_tail_waste",
                context={"pack_index": pack.pack_index},
            )
        return pack


@dataclass(frozen=True)
class PackPlanCursor:
    """Replayable bounded online state after the emitted pack prefix."""

    policy: str
    algorithm_version: str
    global_max_length: int
    window_size: int | None
    lookahead: int | None
    tie_breaker: str
    seed: int
    cursor_byte_budget: int
    next_input_ordinal: int
    next_pack_index: int
    source_prefix_sha256: str
    emitted_prefix_sha256: str
    pending_inputs: tuple[PackPlanInput, ...]
    complete: bool
    resumable: bool = True

    def to_payload_dict(self) -> dict[str, Any]:
        return {
            "schema": PACK_PLAN_CURSOR_SCHEMA,
            "schema_version": PACK_PLAN_CURSOR_SCHEMA_VERSION,
            "policy": self.policy,
            "algorithm_version": self.algorithm_version,
            "global_max_length": self.global_max_length,
            "window_size": self.window_size,
            "lookahead": self.lookahead,
            "tie_breaker": self.tie_breaker,
            "seed": self.seed,
            "cursor_byte_budget": self.cursor_byte_budget,
            "next_input_ordinal": self.next_input_ordinal,
            "next_pack_index": self.next_pack_index,
            "source_prefix_sha256": self.source_prefix_sha256,
            "emitted_prefix_sha256": self.emitted_prefix_sha256,
            "pending_inputs": [item.to_dict() for item in self.pending_inputs],
            "complete": self.complete,
            "resumable": self.resumable,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_payload_dict()
        return {
            **payload,
            "authentication": _authentication_dict(
                payload,
                schema=PACK_PLAN_CURSOR_AUTHENTICATION_SCHEMA,
            ),
        }

    def to_json(self) -> str:
        return _canonical_json_bytes(self.to_dict()).decode("ascii")

    @property
    def serialized_size_bytes(self) -> int:
        return len(self.to_json().encode("ascii"))

    @property
    def canonical_sha256(self) -> str:
        return self.to_dict()["authentication"]["sha256"]

    @classmethod
    def from_json(cls, payload: str) -> PackPlanCursor:
        return cls.from_dict(_strict_json_loads(payload, field="pack_plan_cursor"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PackPlanCursor:
        data = _strict_mapping(
            payload,
            keys={
                "schema",
                "schema_version",
                "policy",
                "algorithm_version",
                "global_max_length",
                "window_size",
                "lookahead",
                "tie_breaker",
                "seed",
                "cursor_byte_budget",
                "next_input_ordinal",
                "next_pack_index",
                "source_prefix_sha256",
                "emitted_prefix_sha256",
                "pending_inputs",
                "complete",
                "resumable",
                "authentication",
            },
            field="replay_cursor",
        )
        _verify_authentication(
            data,
            schema=PACK_PLAN_CURSOR_AUTHENTICATION_SCHEMA,
            field="replay_cursor",
        )
        if data["schema"] != PACK_PLAN_CURSOR_SCHEMA:
            _plan_error(
                "unsupported pack-plan cursor schema",
                code="packing.pack_plan_cursor_schema",
            )
        cursor_schema_version = _strict_int(
            data["schema_version"], field="schema_version"
        )
        if cursor_schema_version != PACK_PLAN_CURSOR_SCHEMA_VERSION:
            _plan_error(
                "unsupported pack-plan cursor schema version",
                code="packing.pack_plan_cursor_version",
            )
        cursor = cls(
            policy=_strict_string(data["policy"], field="policy"),
            algorithm_version=_strict_string(
                data["algorithm_version"], field="algorithm_version"
            ),
            global_max_length=_strict_int(
                data["global_max_length"], field="global_max_length"
            ),
            window_size=_optional_positive_int(
                data["window_size"], field="window_size"
            ),
            lookahead=_optional_positive_int(data["lookahead"], field="lookahead"),
            tie_breaker=_strict_string(data["tie_breaker"], field="tie_breaker"),
            seed=_strict_int(data["seed"], field="seed"),
            cursor_byte_budget=_strict_int(
                data["cursor_byte_budget"], field="cursor_byte_budget"
            ),
            next_input_ordinal=_strict_int(
                data["next_input_ordinal"], field="next_input_ordinal"
            ),
            next_pack_index=_strict_int(
                data["next_pack_index"], field="next_pack_index"
            ),
            source_prefix_sha256=_strict_sha256(
                data["source_prefix_sha256"], field="source_prefix_sha256"
            ),
            emitted_prefix_sha256=_strict_sha256(
                data["emitted_prefix_sha256"], field="emitted_prefix_sha256"
            ),
            pending_inputs=tuple(
                PackPlanInput.from_dict(item)
                for item in _strict_list(data["pending_inputs"], field="pending_inputs")
            ),
            complete=_strict_bool(data["complete"], field="complete"),
            resumable=_strict_bool(data["resumable"], field="resumable"),
        )
        _validate_cursor(cursor)
        return cursor


@dataclass(frozen=True)
class PackPlan:
    """Versioned JSON receipt for deterministic atomic-image packing."""

    policy: str
    algorithm_version: str
    global_max_length: int
    window_size: int | None
    lookahead: int | None
    tie_breaker: str
    seed: int
    cursor_byte_budget: int
    fragment_pack_budget: int | None
    fragment_item_budget: int
    fragment_byte_budget: int
    requested_worker_count: int
    worker_count_disposition: str
    worker_count_integration_requirement: str
    receipt_scope: str
    source_prefix_start_count: int
    source_prefix_start_sha256: str
    emitted_prefix_start_pack_count: int
    emitted_prefix_start_sha256: str
    predecessor_plan_sha256: str | None
    inputs: tuple[PackPlanInput, ...]
    packs: tuple[PackPlanPack, ...]
    rejected_examples: tuple[RejectedPackPlanInput, ...]
    replay_cursor: PackPlanCursor
    max_pending_items_observed: int
    max_serialized_cursor_bytes_observed: int

    def __post_init__(self) -> None:
        _validate_pack_plan(self)

    @property
    def accepted_example_count(self) -> int:
        return sum(len(pack.input_ordinals) for pack in self.packs)

    @property
    def used_tokens(self) -> int:
        return sum(pack.encoded_length for pack in self.packs)

    @property
    def capacity_tokens(self) -> int:
        return len(self.packs) * self.global_max_length

    @property
    def tail_waste(self) -> int:
        return self.capacity_tokens - self.used_tokens

    @property
    def utilization(self) -> float:
        if self.capacity_tokens == 0:
            return 0.0
        return self.used_tokens / self.capacity_tokens

    def to_payload_dict(self) -> dict[str, Any]:
        return {
            "schema": PACK_PLAN_SCHEMA,
            "schema_version": PACK_PLAN_SCHEMA_VERSION,
            "policy": self.policy,
            "algorithm_version": self.algorithm_version,
            "global_max_length": self.global_max_length,
            "parameters": {
                "window_size": self.window_size,
                "lookahead": self.lookahead,
                "tie_breaker": self.tie_breaker,
                "seed": self.seed,
                "cursor_byte_budget": self.cursor_byte_budget,
                "fragment_pack_budget": self.fragment_pack_budget,
                "fragment_item_budget": self.fragment_item_budget,
                "fragment_byte_budget": self.fragment_byte_budget,
            },
            "requested_worker_count": self.requested_worker_count,
            "worker_count_disposition": self.worker_count_disposition,
            "worker_count_integration_requirement": (
                self.worker_count_integration_requirement
            ),
            "receipt_scope": self.receipt_scope,
            "prefix_binding": {
                "digest_schema": PACK_PLAN_PREFIX_DIGEST_SCHEMA,
                "predecessor_plan_sha256": self.predecessor_plan_sha256,
                "source_prefix_start_count": self.source_prefix_start_count,
                "source_prefix_start_sha256": self.source_prefix_start_sha256,
                "emitted_prefix_start_pack_count": (
                    self.emitted_prefix_start_pack_count
                ),
                "emitted_prefix_start_sha256": self.emitted_prefix_start_sha256,
            },
            "inputs": [item.to_dict() for item in self.inputs],
            "packs": [pack.to_dict() for pack in self.packs],
            "rejected_examples": [item.to_dict() for item in self.rejected_examples],
            "metrics": {
                "accepted_example_count": self.accepted_example_count,
                "rejected_example_count": len(self.rejected_examples),
                "pack_count": len(self.packs),
                "used_tokens": self.used_tokens,
                "capacity_tokens": self.capacity_tokens,
                "tail_waste": self.tail_waste,
                "utilization": self.utilization,
                "max_pending_items_observed": self.max_pending_items_observed,
                "max_serialized_cursor_bytes_observed": (
                    self.max_serialized_cursor_bytes_observed
                ),
            },
            "replay_cursor": self.replay_cursor.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_payload_dict()
        return {
            **payload,
            "authentication": _authentication_dict(
                payload,
                schema=PACK_PLAN_AUTHENTICATION_SCHEMA,
            ),
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self) -> str:
        return _canonical_json_bytes(self.to_dict()).decode("ascii")

    @property
    def serialized_size_bytes(self) -> int:
        return len(self.to_json().encode("ascii"))

    @property
    def canonical_sha256(self) -> str:
        return self.to_dict()["authentication"]["sha256"]

    @property
    def semantic_identity_sha256(self) -> str:
        payload = self.to_payload_dict()
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()

    @classmethod
    def from_json(cls, payload: str) -> PackPlan:
        return cls.from_dict(_strict_json_loads(payload, field="pack_plan"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PackPlan:
        data = _strict_mapping(
            payload,
            keys={
                "schema",
                "schema_version",
                "policy",
                "algorithm_version",
                "global_max_length",
                "parameters",
                "requested_worker_count",
                "worker_count_disposition",
                "worker_count_integration_requirement",
                "receipt_scope",
                "prefix_binding",
                "inputs",
                "packs",
                "rejected_examples",
                "metrics",
                "replay_cursor",
                "authentication",
            },
            field="pack_plan",
        )
        _verify_authentication(
            data,
            schema=PACK_PLAN_AUTHENTICATION_SCHEMA,
            field="pack_plan",
        )
        if data["schema"] != PACK_PLAN_SCHEMA:
            _plan_error(
                "unsupported pack-plan schema",
                code="packing.pack_plan_schema",
            )
        schema_version = _strict_int(data["schema_version"], field="schema_version")
        if schema_version != PACK_PLAN_SCHEMA_VERSION:
            _plan_error(
                "unsupported pack-plan schema version",
                code="packing.pack_plan_version",
            )
        parameters = _strict_mapping(
            data["parameters"],
            keys={
                "window_size",
                "lookahead",
                "tie_breaker",
                "seed",
                "cursor_byte_budget",
                "fragment_pack_budget",
                "fragment_item_budget",
                "fragment_byte_budget",
            },
            field="parameters",
        )
        prefix_binding = _strict_mapping(
            data["prefix_binding"],
            keys={
                "digest_schema",
                "predecessor_plan_sha256",
                "source_prefix_start_count",
                "source_prefix_start_sha256",
                "emitted_prefix_start_pack_count",
                "emitted_prefix_start_sha256",
            },
            field="prefix_binding",
        )
        if prefix_binding["digest_schema"] != PACK_PLAN_PREFIX_DIGEST_SCHEMA:
            _plan_error(
                "unsupported pack-plan prefix digest schema",
                code="packing.pack_plan_prefix_schema",
            )
        metrics = _strict_mapping(
            data["metrics"],
            keys={
                "accepted_example_count",
                "rejected_example_count",
                "pack_count",
                "used_tokens",
                "capacity_tokens",
                "tail_waste",
                "utilization",
                "max_pending_items_observed",
                "max_serialized_cursor_bytes_observed",
            },
            field="metrics",
        )
        plan = cls(
            policy=_strict_string(data["policy"], field="policy"),
            algorithm_version=_strict_string(
                data["algorithm_version"], field="algorithm_version"
            ),
            global_max_length=_strict_int(
                data["global_max_length"], field="global_max_length"
            ),
            window_size=_optional_positive_int(
                parameters["window_size"], field="window_size"
            ),
            lookahead=_optional_positive_int(
                parameters["lookahead"], field="lookahead"
            ),
            tie_breaker=_strict_string(parameters["tie_breaker"], field="tie_breaker"),
            seed=_strict_int(parameters["seed"], field="seed"),
            cursor_byte_budget=_strict_int(
                parameters["cursor_byte_budget"], field="cursor_byte_budget"
            ),
            fragment_pack_budget=_optional_positive_int(
                parameters["fragment_pack_budget"], field="fragment_pack_budget"
            ),
            fragment_item_budget=_strict_int(
                parameters["fragment_item_budget"], field="fragment_item_budget"
            ),
            fragment_byte_budget=_strict_int(
                parameters["fragment_byte_budget"], field="fragment_byte_budget"
            ),
            requested_worker_count=_strict_int(
                data["requested_worker_count"], field="requested_worker_count"
            ),
            worker_count_disposition=_strict_string(
                data["worker_count_disposition"],
                field="worker_count_disposition",
            ),
            worker_count_integration_requirement=_strict_string(
                data["worker_count_integration_requirement"],
                field="worker_count_integration_requirement",
            ),
            receipt_scope=_strict_string(data["receipt_scope"], field="receipt_scope"),
            source_prefix_start_count=_strict_int(
                prefix_binding["source_prefix_start_count"],
                field="source_prefix_start_count",
            ),
            source_prefix_start_sha256=_strict_sha256(
                prefix_binding["source_prefix_start_sha256"],
                field="source_prefix_start_sha256",
            ),
            emitted_prefix_start_pack_count=_strict_int(
                prefix_binding["emitted_prefix_start_pack_count"],
                field="emitted_prefix_start_pack_count",
            ),
            emitted_prefix_start_sha256=_strict_sha256(
                prefix_binding["emitted_prefix_start_sha256"],
                field="emitted_prefix_start_sha256",
            ),
            predecessor_plan_sha256=_optional_sha256(
                prefix_binding["predecessor_plan_sha256"],
                field="predecessor_plan_sha256",
            ),
            inputs=tuple(
                PackPlanInput.from_dict(item)
                for item in _strict_list(data["inputs"], field="inputs")
            ),
            packs=tuple(
                PackPlanPack.from_dict(item)
                for item in _strict_list(data["packs"], field="packs")
            ),
            rejected_examples=tuple(
                RejectedPackPlanInput.from_dict(item)
                for item in _strict_list(
                    data["rejected_examples"], field="rejected_examples"
                )
            ),
            replay_cursor=PackPlanCursor.from_dict(
                _strict_mapping_value(data["replay_cursor"], field="replay_cursor")
            ),
            max_pending_items_observed=_strict_int(
                metrics["max_pending_items_observed"],
                field="max_pending_items_observed",
            ),
            max_serialized_cursor_bytes_observed=_strict_int(
                metrics["max_serialized_cursor_bytes_observed"],
                field="max_serialized_cursor_bytes_observed",
            ),
        )
        expected_metrics = plan.to_dict()["metrics"]
        for key, expected in expected_metrics.items():
            observed = metrics[key]
            if isinstance(expected, float):
                observed = _strict_float(observed, field=key)
            else:
                observed = _strict_int(observed, field=key)
            if observed != expected:
                _plan_error(
                    "serialized pack-plan metric does not match membership",
                    code="packing.pack_plan_metric",
                    context={"metric": key},
                )
            if isinstance(expected, float) and not math.isclose(
                observed, expected, rel_tol=0.0, abs_tol=0.0
            ):
                _plan_error(
                    "serialized pack-plan metric does not match membership",
                    code="packing.pack_plan_metric",
                    context={"metric": key},
                )
        return plan


@dataclass(frozen=True)
class PackPlanStreamReceipt:
    """Bounded terminal receipt for sink-published online fragments."""

    fragment_count: int
    source_input_count: int
    accepted_example_count: int
    rejected_example_count: int
    emitted_pack_count: int
    max_packs_per_fragment: int
    requested_worker_count: int
    worker_count_disposition: str
    worker_count_integration_requirement: str
    fragment_item_budget: int
    fragment_byte_budget: int
    fragment_chain_sha256: str
    max_fragment_items_observed: int
    max_fragment_bytes_observed: int
    max_pending_items_observed: int
    max_cursor_bytes_observed: int
    terminal_cursor: PackPlanCursor

    def __post_init__(self) -> None:
        for field, value in (
            ("fragment_count", self.fragment_count),
            ("source_input_count", self.source_input_count),
            ("accepted_example_count", self.accepted_example_count),
            ("rejected_example_count", self.rejected_example_count),
            ("emitted_pack_count", self.emitted_pack_count),
            ("max_fragment_items_observed", self.max_fragment_items_observed),
            ("max_fragment_bytes_observed", self.max_fragment_bytes_observed),
            ("max_pending_items_observed", self.max_pending_items_observed),
            ("max_cursor_bytes_observed", self.max_cursor_bytes_observed),
        ):
            _nonnegative_int(value, field=field)
        _positive_int(self.fragment_count, field="fragment_count")
        _positive_int(
            self.max_packs_per_fragment,
            field="max_packs_per_fragment",
        )
        _positive_int(self.requested_worker_count, field="requested_worker_count")
        if self.worker_count_disposition != _WORKER_COUNT_DISPOSITION:
            _plan_error(
                "unsupported worker-count disposition",
                code="packing.pack_plan_worker_disposition",
            )
        if (
            self.worker_count_integration_requirement
            != _WORKER_COUNT_INTEGRATION_REQUIREMENT
        ):
            _plan_error(
                "unsupported worker-count integration requirement",
                code="packing.pack_plan_worker_disposition",
            )
        _positive_int(self.fragment_item_budget, field="fragment_item_budget")
        _positive_int(self.fragment_byte_budget, field="fragment_byte_budget")
        _strict_sha256(
            self.fragment_chain_sha256,
            field="fragment_chain_sha256",
        )
        _validate_cursor(self.terminal_cursor)
        if not self.terminal_cursor.complete:
            _plan_error(
                "online stream receipt requires a complete terminal cursor",
                code="packing.pack_plan_stream_incomplete",
            )
        if (
            self.source_input_count != self.terminal_cursor.next_input_ordinal
            or self.emitted_pack_count != self.terminal_cursor.next_pack_index
            or self.accepted_example_count + self.rejected_example_count
            != self.source_input_count
        ):
            _plan_error(
                "online stream receipt counters do not match its terminal cursor",
                code="packing.pack_plan_stream_counters",
            )
        if (
            self.max_fragment_items_observed > self.fragment_item_budget
            or self.max_fragment_bytes_observed > self.fragment_byte_budget
            or self.max_pending_items_observed > (self.terminal_cursor.lookahead or 0)
            or self.max_cursor_bytes_observed > self.terminal_cursor.cursor_byte_budget
        ):
            _plan_error(
                "online stream receipt exceeds a declared resident-state budget",
                code="packing.pack_plan_stream_bound",
            )

    def to_payload_dict(self) -> dict[str, Any]:
        return {
            "schema": PACK_PLAN_STREAM_RECEIPT_SCHEMA,
            "schema_version": PACK_PLAN_STREAM_RECEIPT_SCHEMA_VERSION,
            "fragment_count": self.fragment_count,
            "source_input_count": self.source_input_count,
            "accepted_example_count": self.accepted_example_count,
            "rejected_example_count": self.rejected_example_count,
            "emitted_pack_count": self.emitted_pack_count,
            "max_packs_per_fragment": self.max_packs_per_fragment,
            "requested_worker_count": self.requested_worker_count,
            "worker_count_disposition": self.worker_count_disposition,
            "worker_count_integration_requirement": (
                self.worker_count_integration_requirement
            ),
            "fragment_item_budget": self.fragment_item_budget,
            "fragment_byte_budget": self.fragment_byte_budget,
            "fragment_chain_sha256": self.fragment_chain_sha256,
            "max_fragment_items_observed": self.max_fragment_items_observed,
            "max_fragment_bytes_observed": self.max_fragment_bytes_observed,
            "max_pending_items_observed": self.max_pending_items_observed,
            "max_cursor_bytes_observed": self.max_cursor_bytes_observed,
            "terminal_cursor": self.terminal_cursor.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_payload_dict()
        return {
            **payload,
            "authentication": _authentication_dict(
                payload,
                schema=PACK_PLAN_STREAM_AUTHENTICATION_SCHEMA,
            ),
        }

    def to_json(self) -> str:
        return _canonical_json_bytes(self.to_dict()).decode("ascii")

    @property
    def serialized_size_bytes(self) -> int:
        return len(self.to_json().encode("ascii"))

    @property
    def canonical_sha256(self) -> str:
        return self.to_dict()["authentication"]["sha256"]

    @classmethod
    def from_json(cls, payload: str) -> PackPlanStreamReceipt:
        return cls.from_dict(
            _strict_json_loads(payload, field="pack_plan_stream_receipt")
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PackPlanStreamReceipt:
        data = _strict_mapping(
            payload,
            keys={
                "schema",
                "schema_version",
                "fragment_count",
                "source_input_count",
                "accepted_example_count",
                "rejected_example_count",
                "emitted_pack_count",
                "max_packs_per_fragment",
                "requested_worker_count",
                "worker_count_disposition",
                "worker_count_integration_requirement",
                "fragment_item_budget",
                "fragment_byte_budget",
                "fragment_chain_sha256",
                "max_fragment_items_observed",
                "max_fragment_bytes_observed",
                "max_pending_items_observed",
                "max_cursor_bytes_observed",
                "terminal_cursor",
                "authentication",
            },
            field="pack_plan_stream_receipt",
        )
        _verify_authentication(
            data,
            schema=PACK_PLAN_STREAM_AUTHENTICATION_SCHEMA,
            field="pack_plan_stream_receipt",
        )
        if data["schema"] != PACK_PLAN_STREAM_RECEIPT_SCHEMA:
            _plan_error(
                "unsupported pack-plan stream receipt schema",
                code="packing.pack_plan_stream_schema",
            )
        schema_version = _strict_int(data["schema_version"], field="schema_version")
        if schema_version != PACK_PLAN_STREAM_RECEIPT_SCHEMA_VERSION:
            _plan_error(
                "unsupported pack-plan stream receipt schema version",
                code="packing.pack_plan_stream_version",
            )
        return cls(
            fragment_count=_strict_int(data["fragment_count"], field="fragment_count"),
            source_input_count=_strict_int(
                data["source_input_count"], field="source_input_count"
            ),
            accepted_example_count=_strict_int(
                data["accepted_example_count"], field="accepted_example_count"
            ),
            rejected_example_count=_strict_int(
                data["rejected_example_count"], field="rejected_example_count"
            ),
            emitted_pack_count=_strict_int(
                data["emitted_pack_count"], field="emitted_pack_count"
            ),
            max_packs_per_fragment=_strict_int(
                data["max_packs_per_fragment"], field="max_packs_per_fragment"
            ),
            requested_worker_count=_strict_int(
                data["requested_worker_count"], field="requested_worker_count"
            ),
            worker_count_disposition=_strict_string(
                data["worker_count_disposition"], field="worker_count_disposition"
            ),
            worker_count_integration_requirement=_strict_string(
                data["worker_count_integration_requirement"],
                field="worker_count_integration_requirement",
            ),
            fragment_item_budget=_strict_int(
                data["fragment_item_budget"], field="fragment_item_budget"
            ),
            fragment_byte_budget=_strict_int(
                data["fragment_byte_budget"], field="fragment_byte_budget"
            ),
            fragment_chain_sha256=_strict_sha256(
                data["fragment_chain_sha256"], field="fragment_chain_sha256"
            ),
            max_fragment_items_observed=_strict_int(
                data["max_fragment_items_observed"],
                field="max_fragment_items_observed",
            ),
            max_fragment_bytes_observed=_strict_int(
                data["max_fragment_bytes_observed"],
                field="max_fragment_bytes_observed",
            ),
            max_pending_items_observed=_strict_int(
                data["max_pending_items_observed"],
                field="max_pending_items_observed",
            ),
            max_cursor_bytes_observed=_strict_int(
                data["max_cursor_bytes_observed"], field="max_cursor_bytes_observed"
            ),
            terminal_cursor=PackPlanCursor.from_dict(
                _strict_mapping_value(data["terminal_cursor"], field="terminal_cursor")
            ),
        )


def _authenticate_replay_predecessor(
    *,
    replay_cursor: PackPlanCursor | None,
    replay_plan: PackPlan | None,
) -> tuple[PackPlanCursor | None, str | None]:
    if replay_cursor is None and replay_plan is None:
        return None, None
    if replay_cursor is None or replay_plan is None:
        _plan_error(
            "online resume requires both the cursor and its authenticated predecessor plan",
            code="packing.pack_plan_cursor_predecessor",
        )
    if not isinstance(replay_cursor, PackPlanCursor) or not isinstance(
        replay_plan, PackPlan
    ):
        _plan_error(
            "online resume predecessor has the wrong type",
            code="packing.pack_plan_cursor_predecessor",
        )
    _validate_pack_plan(replay_plan)
    if (
        replay_plan.policy != ONLINE_WINDOW_BINPACK
        or replay_plan.replay_cursor != replay_cursor
        or replay_plan.replay_cursor.canonical_sha256 != replay_cursor.canonical_sha256
    ):
        _plan_error(
            "online resume cursor does not match its authenticated predecessor plan",
            code="packing.pack_plan_cursor_predecessor",
        )
    return replay_cursor, replay_plan.canonical_sha256


def build_pack_plan_policy_identity(
    *,
    policy: str = SOURCE_ORDER_NEXT_FIT,
    window_size: int | None = None,
    lookahead: int | None = None,
    seed: int = 0,
    worker_count: int = 1,
    cursor_byte_budget: int = DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET,
    fragment_item_budget: int = DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET,
    fragment_byte_budget: int = DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET,
) -> dict[str, Any]:
    """Return the strict public identity consumed by config/cache receipts."""

    _validate_planner_arguments(
        global_max_length=1,
        policy=policy,
        window_size=window_size,
        lookahead=lookahead,
        seed=seed,
        worker_count=worker_count,
        cursor_byte_budget=cursor_byte_budget,
        fragment_item_budget=fragment_item_budget,
        fragment_byte_budget=fragment_byte_budget,
        replay_cursor=None,
        max_packs=1 if policy == ONLINE_WINDOW_BINPACK else None,
    )
    return {
        "policy": policy,
        "algorithm_version": _ALGORITHM_VERSIONS[policy],
        "window_size": window_size,
        "lookahead": lookahead,
        "tie_breaker": _TIE_BREAKERS[policy],
        "seed": seed,
        "requested_worker_count": worker_count,
        "worker_count_disposition": _WORKER_COUNT_DISPOSITION,
        "worker_count_integration_requirement": (_WORKER_COUNT_INTEGRATION_REQUIREMENT),
        "cursor_byte_budget": cursor_byte_budget,
        "fragment_item_budget": fragment_item_budget,
        "fragment_byte_budget": fragment_byte_budget,
    }


def create_pack_plan(
    encoded_examples: Iterable[Any],
    *,
    global_max_length: int,
    policy: str = SOURCE_ORDER_NEXT_FIT,
    window_size: int | None = None,
    lookahead: int | None = None,
    seed: int = 0,
    worker_count: int = 1,
    cursor_byte_budget: int = DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET,
    fragment_item_budget: int = DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET,
    fragment_byte_budget: int = DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET,
    replay_cursor: PackPlanCursor | None = None,
    replay_plan: PackPlan | None = None,
    max_packs: int | None = None,
    intra_image_order_identity_getter: Callable[[Any], str] | None = None,
) -> PackPlan:
    """Create a deterministic receipt without splitting encoded examples.

    Invalid zero-length and over-capacity examples are preserved in
    ``rejected_examples``. The compatibility wrapper above intentionally keeps
    its historical fail-fast behavior.
    """

    replay_cursor, predecessor_plan_sha256 = _authenticate_replay_predecessor(
        replay_cursor=replay_cursor,
        replay_plan=replay_plan,
    )
    _validate_planner_arguments(
        global_max_length=global_max_length,
        policy=policy,
        window_size=window_size,
        lookahead=lookahead,
        seed=seed,
        worker_count=worker_count,
        cursor_byte_budget=cursor_byte_budget,
        fragment_item_budget=fragment_item_budget,
        fragment_byte_budget=fragment_byte_budget,
        replay_cursor=replay_cursor,
        max_packs=max_packs,
    )
    if policy == ONLINE_WINDOW_BINPACK:
        if max_packs is None:
            _plan_error(
                "online packing must emit a finite receipt fragment",
                code="packing.pack_plan_fragment_bound",
            )
        return _plan_online(
            encoded_examples,
            global_max_length=global_max_length,
            lookahead=lookahead,
            seed=seed,
            cursor_byte_budget=cursor_byte_budget,
            fragment_item_budget=fragment_item_budget,
            fragment_byte_budget=fragment_byte_budget,
            requested_worker_count=worker_count,
            replay_cursor=replay_cursor,
            predecessor_plan_sha256=predecessor_plan_sha256,
            max_packs=max_packs,
            identity_getter=intra_image_order_identity_getter,
        )
    if replay_cursor is not None or replay_plan is not None:
        _plan_error(
            "only online_window_binpack accepts a replay cursor",
            code="packing.pack_plan_cursor_policy",
        )
    if max_packs is not None:
        _plan_error(
            "bounded partial planning is supported only for online_window_binpack",
            code="packing.pack_plan_partial_policy",
        )

    inputs, accepted, rejected = _read_all_inputs(
        encoded_examples,
        global_max_length=global_max_length,
        identity_getter=intra_image_order_identity_getter,
    )
    if policy == SOURCE_ORDER_NEXT_FIT:
        memberships = _source_order_memberships(accepted, global_max_length)
        max_pending = min(1, len(accepted))
    else:
        assert window_size is not None
        memberships = _window_binpack_memberships(
            accepted,
            global_max_length=global_max_length,
            window_size=window_size,
            seed=seed,
        )
        max_pending = min(window_size, len(accepted))
    packs = _memberships_to_packs(
        memberships,
        global_max_length=global_max_length,
        first_pack_index=0,
    )
    source_prefix_sha256 = _fold_source_prefix(
        _INITIAL_SOURCE_PREFIX_SHA256,
        inputs,
    )
    emitted_prefix_sha256 = _fold_emitted_prefix(
        _INITIAL_EMITTED_PREFIX_SHA256,
        packs,
    )
    cursor = PackPlanCursor(
        policy=policy,
        algorithm_version=_ALGORITHM_VERSIONS[policy],
        global_max_length=global_max_length,
        window_size=window_size,
        lookahead=None,
        tie_breaker=_TIE_BREAKERS[policy],
        seed=seed,
        cursor_byte_budget=cursor_byte_budget,
        next_input_ordinal=len(inputs),
        next_pack_index=len(packs),
        source_prefix_sha256=source_prefix_sha256,
        emitted_prefix_sha256=emitted_prefix_sha256,
        pending_inputs=(),
        complete=True,
    )
    cursor_size = cursor.serialized_size_bytes
    return PackPlan(
        policy=policy,
        algorithm_version=_ALGORITHM_VERSIONS[policy],
        global_max_length=global_max_length,
        window_size=window_size,
        lookahead=None,
        tie_breaker=_TIE_BREAKERS[policy],
        seed=seed,
        cursor_byte_budget=cursor_byte_budget,
        fragment_pack_budget=None,
        fragment_item_budget=fragment_item_budget,
        fragment_byte_budget=fragment_byte_budget,
        requested_worker_count=worker_count,
        worker_count_disposition=_WORKER_COUNT_DISPOSITION,
        worker_count_integration_requirement=_WORKER_COUNT_INTEGRATION_REQUIREMENT,
        receipt_scope=_COMPLETE_PLAN_SCOPE,
        source_prefix_start_count=0,
        source_prefix_start_sha256=_INITIAL_SOURCE_PREFIX_SHA256,
        emitted_prefix_start_pack_count=0,
        emitted_prefix_start_sha256=_INITIAL_EMITTED_PREFIX_SHA256,
        predecessor_plan_sha256=None,
        inputs=inputs,
        packs=packs,
        rejected_examples=rejected,
        replay_cursor=cursor,
        max_pending_items_observed=max_pending,
        max_serialized_cursor_bytes_observed=cursor_size,
    )


def stream_online_pack_plan_fragments(
    encoded_examples_factory: Callable[[], Iterable[Any]],
    *,
    fragment_sink: Callable[[PackPlan], None],
    global_max_length: int,
    lookahead: int,
    max_packs_per_fragment: int,
    seed: int = 0,
    worker_count: int = 1,
    cursor_byte_budget: int = DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET,
    fragment_item_budget: int = DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET,
    fragment_byte_budget: int = DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET,
    intra_image_order_identity_getter: Callable[[Any], str] | None = None,
) -> PackPlanStreamReceipt:
    """Publish authenticated fragments from one uninterrupted source iterator."""

    if not callable(encoded_examples_factory):
        _plan_error(
            "online fragment source factory must be callable",
            code="packing.pack_plan_fragment_source",
        )
    if not callable(fragment_sink):
        _plan_error(
            "online fragment sink must be callable",
            code="packing.pack_plan_fragment_sink",
        )
    _positive_int(max_packs_per_fragment, field="max_packs_per_fragment")
    _validate_planner_arguments(
        global_max_length=global_max_length,
        policy=ONLINE_WINDOW_BINPACK,
        window_size=None,
        lookahead=lookahead,
        seed=seed,
        worker_count=worker_count,
        cursor_byte_budget=cursor_byte_budget,
        fragment_item_budget=fragment_item_budget,
        fragment_byte_budget=fragment_byte_budget,
        replay_cursor=None,
        max_packs=1,
    )
    source = encoded_examples_factory()
    try:
        source_iterator = iter(source)
    except TypeError as exc:
        raise PackingContractError(
            "online fragment source factory must return an iterable",
            code="packing.pack_plan_fragment_source",
            cause=exc,
        ) from exc

    internal_cursor: PackPlanCursor | None = None
    internal_predecessor_sha256: str | None = None
    output_predecessor_sha256: str | None = None
    fragment_count = 0
    accepted_count = 0
    rejected_count = 0
    emitted_pack_count = 0
    fragment_chain_sha256 = _INITIAL_FRAGMENT_CHAIN_SHA256
    max_fragment_items_observed = 0
    max_fragment_bytes_observed = 0
    max_pending_items_observed = 0
    max_cursor_bytes_observed = 0

    active = False
    active_source_start_count = 0
    active_source_start_sha256 = _INITIAL_SOURCE_PREFIX_SHA256
    active_pack_start_count = 0
    active_pack_start_sha256 = _INITIAL_EMITTED_PREFIX_SHA256
    active_inputs: dict[int, PackPlanInput] = {}
    active_packs: list[PackPlanPack] = []
    active_rejected: list[RejectedPackPlanInput] = []
    active_cursor: PackPlanCursor | None = None
    active_max_pending = 0
    active_max_cursor_bytes = 0

    def build_active_fragment() -> PackPlan:
        assert active and active_cursor is not None
        return PackPlan(
            policy=ONLINE_WINDOW_BINPACK,
            algorithm_version=_ALGORITHM_VERSIONS[ONLINE_WINDOW_BINPACK],
            global_max_length=global_max_length,
            window_size=None,
            lookahead=lookahead,
            tie_breaker=_TIE_BREAKERS[ONLINE_WINDOW_BINPACK],
            seed=seed,
            cursor_byte_budget=cursor_byte_budget,
            fragment_pack_budget=max_packs_per_fragment,
            fragment_item_budget=fragment_item_budget,
            fragment_byte_budget=fragment_byte_budget,
            requested_worker_count=worker_count,
            worker_count_disposition=_WORKER_COUNT_DISPOSITION,
            worker_count_integration_requirement=(
                _WORKER_COUNT_INTEGRATION_REQUIREMENT
            ),
            receipt_scope=_RESUME_FRAGMENT_SCOPE,
            source_prefix_start_count=active_source_start_count,
            source_prefix_start_sha256=active_source_start_sha256,
            emitted_prefix_start_pack_count=active_pack_start_count,
            emitted_prefix_start_sha256=active_pack_start_sha256,
            predecessor_plan_sha256=output_predecessor_sha256,
            inputs=tuple(active_inputs[ordinal] for ordinal in sorted(active_inputs)),
            packs=tuple(active_packs),
            rejected_examples=tuple(active_rejected),
            replay_cursor=active_cursor,
            max_pending_items_observed=active_max_pending,
            max_serialized_cursor_bytes_observed=active_max_cursor_bytes,
        )

    def emit_active() -> None:
        nonlocal fragment_count, accepted_count, rejected_count
        nonlocal emitted_pack_count, fragment_chain_sha256
        nonlocal max_fragment_items_observed, max_fragment_bytes_observed
        nonlocal max_pending_items_observed, max_cursor_bytes_observed
        nonlocal output_predecessor_sha256
        fragment = build_active_fragment()
        fragment_sink(fragment)
        fragment_count += 1
        accepted_count += fragment.accepted_example_count
        rejected_count += len(fragment.rejected_examples)
        emitted_pack_count += len(fragment.packs)
        fragment_chain_sha256 = _extend_prefix_digest(
            fragment_chain_sha256,
            {
                "fragment_index": fragment_count - 1,
                "fragment_sha256": fragment.canonical_sha256,
            },
        )
        max_fragment_items_observed = max(
            max_fragment_items_observed,
            len(fragment.inputs),
        )
        max_fragment_bytes_observed = max(
            max_fragment_bytes_observed,
            fragment.serialized_size_bytes,
        )
        max_pending_items_observed = max(
            max_pending_items_observed,
            fragment.max_pending_items_observed,
        )
        max_cursor_bytes_observed = max(
            max_cursor_bytes_observed,
            fragment.max_serialized_cursor_bytes_observed,
        )
        output_predecessor_sha256 = fragment.canonical_sha256

    def start_active(part: PackPlan) -> None:
        nonlocal active, active_source_start_count, active_source_start_sha256
        nonlocal active_pack_start_count, active_pack_start_sha256
        nonlocal active_inputs, active_packs, active_rejected, active_cursor
        nonlocal active_max_pending, active_max_cursor_bytes
        active = True
        active_source_start_count = part.source_prefix_start_count
        active_source_start_sha256 = part.source_prefix_start_sha256
        active_pack_start_count = part.emitted_prefix_start_pack_count
        active_pack_start_sha256 = part.emitted_prefix_start_sha256
        active_inputs = {item.input_ordinal: item for item in part.inputs}
        active_packs = list(part.packs)
        active_rejected = list(part.rejected_examples)
        active_cursor = part.replay_cursor
        active_max_pending = part.max_pending_items_observed
        active_max_cursor_bytes = part.max_serialized_cursor_bytes_observed

    while internal_cursor is None or not internal_cursor.complete:
        prior_cursor = internal_cursor
        part = _plan_online(
            source_iterator,
            global_max_length=global_max_length,
            lookahead=lookahead,
            seed=seed,
            cursor_byte_budget=cursor_byte_budget,
            fragment_item_budget=fragment_item_budget,
            fragment_byte_budget=fragment_byte_budget,
            requested_worker_count=worker_count,
            replay_cursor=internal_cursor,
            predecessor_plan_sha256=internal_predecessor_sha256,
            max_packs=1,
            identity_getter=intra_image_order_identity_getter,
            source_positioned_at_cursor=True,
        )
        if (
            prior_cursor is not None
            and part.replay_cursor.next_input_ordinal == prior_cursor.next_input_ordinal
            and part.replay_cursor.next_pack_index == prior_cursor.next_pack_index
        ):
            _plan_error(
                "online fragment budgets cannot make forward progress",
                code="packing.pack_plan_fragment_bound",
            )
        internal_cursor = part.replay_cursor
        internal_predecessor_sha256 = part.canonical_sha256
        if not active:
            start_active(part)
        else:
            assert active_cursor is not None
            merged_inputs = dict(active_inputs)
            for item in part.inputs:
                existing = merged_inputs.get(item.input_ordinal)
                if existing is not None and existing != item:
                    _plan_error(
                        "online fragment input identity changed within one stream",
                        code="packing.pack_plan_cursor_identity",
                    )
                merged_inputs[item.input_ordinal] = item
            exceeds_count_bound = (
                len(active_packs) + len(part.packs) > max_packs_per_fragment
                or len(merged_inputs) > fragment_item_budget
            )
            if not exceeds_count_bound:
                previous_inputs = active_inputs
                previous_packs = active_packs
                previous_rejected = active_rejected
                previous_cursor = active_cursor
                previous_max_pending = active_max_pending
                previous_max_cursor_bytes = active_max_cursor_bytes
                active_inputs = merged_inputs
                active_packs = [*active_packs, *part.packs]
                active_rejected = [*active_rejected, *part.rejected_examples]
                active_cursor = part.replay_cursor
                active_max_pending = max(
                    active_max_pending, part.max_pending_items_observed
                )
                active_max_cursor_bytes = max(
                    active_max_cursor_bytes,
                    part.max_serialized_cursor_bytes_observed,
                )
                try:
                    build_active_fragment()
                except PackingContractError as exc:
                    if exc.code != "packing.pack_plan_fragment_byte_bound":
                        raise
                    exceeds_count_bound = True
                    active_inputs = previous_inputs
                    active_packs = previous_packs
                    active_rejected = previous_rejected
                    active_cursor = previous_cursor
                    active_max_pending = previous_max_pending
                    active_max_cursor_bytes = previous_max_cursor_bytes
            if exceeds_count_bound:
                emit_active()
                start_active(part)

    if active:
        emit_active()
    assert internal_cursor is not None
    return PackPlanStreamReceipt(
        fragment_count=fragment_count,
        source_input_count=internal_cursor.next_input_ordinal,
        accepted_example_count=accepted_count,
        rejected_example_count=rejected_count,
        emitted_pack_count=emitted_pack_count,
        max_packs_per_fragment=max_packs_per_fragment,
        requested_worker_count=worker_count,
        worker_count_disposition=_WORKER_COUNT_DISPOSITION,
        worker_count_integration_requirement=_WORKER_COUNT_INTEGRATION_REQUIREMENT,
        fragment_item_budget=fragment_item_budget,
        fragment_byte_budget=fragment_byte_budget,
        fragment_chain_sha256=fragment_chain_sha256,
        max_fragment_items_observed=max_fragment_items_observed,
        max_fragment_bytes_observed=max_fragment_bytes_observed,
        max_pending_items_observed=max_pending_items_observed,
        max_cursor_bytes_observed=max_cursor_bytes_observed,
        terminal_cursor=internal_cursor,
    )


def verify_pack_plan_stream_fragments(
    receipt: PackPlanStreamReceipt,
    fragments: Iterable[PackPlan],
) -> None:
    """Verify one exact ordered fragment chain against its terminal receipt."""

    if not isinstance(receipt, PackPlanStreamReceipt):
        _plan_error(
            "fragment-chain verification requires a PackPlanStreamReceipt",
            code="packing.pack_plan_stream_type",
        )
    previous: PackPlan | None = None
    fragment_chain_sha256 = _INITIAL_FRAGMENT_CHAIN_SHA256
    fragment_count = 0
    accepted_count = 0
    rejected_count = 0
    emitted_pack_count = 0
    max_fragment_items = 0
    max_fragment_bytes = 0
    max_pending_items = 0
    max_cursor_bytes = 0
    for fragment_index, fragment in enumerate(fragments):
        if fragment_index >= receipt.fragment_count:
            _plan_error(
                "fragment chain contains more fragments than its receipt",
                code="packing.pack_plan_stream_chain",
            )
        if not isinstance(fragment, PackPlan):
            _plan_error(
                "fragment chain contains a non-PackPlan value",
                code="packing.pack_plan_stream_chain",
            )
        _validate_pack_plan(fragment)
        if (
            fragment.policy != ONLINE_WINDOW_BINPACK
            or fragment.receipt_scope != _RESUME_FRAGMENT_SCOPE
            or fragment.fragment_pack_budget != receipt.max_packs_per_fragment
            or fragment.fragment_item_budget != receipt.fragment_item_budget
            or fragment.fragment_byte_budget != receipt.fragment_byte_budget
            or fragment.requested_worker_count != receipt.requested_worker_count
            or fragment.worker_count_disposition != receipt.worker_count_disposition
            or fragment.worker_count_integration_requirement
            != receipt.worker_count_integration_requirement
        ):
            _plan_error(
                "fragment identity or bound differs from its stream receipt",
                code="packing.pack_plan_stream_chain",
                context={"fragment_index": fragment_index},
            )
        if previous is None:
            if (
                fragment.predecessor_plan_sha256 is not None
                or fragment.source_prefix_start_count != 0
                or fragment.source_prefix_start_sha256 != _INITIAL_SOURCE_PREFIX_SHA256
                or fragment.emitted_prefix_start_pack_count != 0
                or fragment.emitted_prefix_start_sha256
                != _INITIAL_EMITTED_PREFIX_SHA256
            ):
                _plan_error(
                    "fragment chain does not start at the initial prefixes",
                    code="packing.pack_plan_stream_chain",
                )
        elif (
            fragment.predecessor_plan_sha256 != previous.canonical_sha256
            or fragment.source_prefix_start_count
            != previous.replay_cursor.next_input_ordinal
            or fragment.source_prefix_start_sha256
            != previous.replay_cursor.source_prefix_sha256
            or fragment.emitted_prefix_start_pack_count
            != previous.replay_cursor.next_pack_index
            or fragment.emitted_prefix_start_sha256
            != previous.replay_cursor.emitted_prefix_sha256
        ):
            _plan_error(
                "fragment chain predecessor or ordered prefix is discontinuous",
                code="packing.pack_plan_stream_chain",
                context={"fragment_index": fragment_index},
            )
        fragment_chain_sha256 = _extend_prefix_digest(
            fragment_chain_sha256,
            {
                "fragment_index": fragment_index,
                "fragment_sha256": fragment.canonical_sha256,
            },
        )
        fragment_count += 1
        accepted_count += fragment.accepted_example_count
        rejected_count += len(fragment.rejected_examples)
        emitted_pack_count += len(fragment.packs)
        max_fragment_items = max(max_fragment_items, len(fragment.inputs))
        max_fragment_bytes = max(max_fragment_bytes, fragment.serialized_size_bytes)
        max_pending_items = max(max_pending_items, fragment.max_pending_items_observed)
        max_cursor_bytes = max(
            max_cursor_bytes, fragment.max_serialized_cursor_bytes_observed
        )
        previous = fragment

    if previous is None or fragment_count != receipt.fragment_count:
        _plan_error(
            "fragment chain count differs from its stream receipt",
            code="packing.pack_plan_stream_chain",
        )
    observed = (
        previous.replay_cursor,
        fragment_chain_sha256,
        accepted_count,
        rejected_count,
        emitted_pack_count,
        max_fragment_items,
        max_fragment_bytes,
        max_pending_items,
        max_cursor_bytes,
    )
    expected = (
        receipt.terminal_cursor,
        receipt.fragment_chain_sha256,
        receipt.accepted_example_count,
        receipt.rejected_example_count,
        receipt.emitted_pack_count,
        receipt.max_fragment_items_observed,
        receipt.max_fragment_bytes_observed,
        receipt.max_pending_items_observed,
        receipt.max_cursor_bytes_observed,
    )
    if observed != expected:
        _plan_error(
            "fragment chain counters, digest, or terminal cursor differ from receipt",
            code="packing.pack_plan_stream_terminal",
        )


def replay_pack_plan(
    plan: PackPlan,
    encoded_examples: Sequence[Any],
    *,
    intra_image_order_identity_getter: Callable[[Any], str] | None = None,
) -> tuple[PackedSequence, ...]:
    """Materialize the exact membership after revalidating atomic identities."""

    if not isinstance(plan, PackPlan):
        _plan_error(
            "replay requires a PackPlan",
            code="packing.pack_plan_replay_type",
        )
    examples = tuple(encoded_examples)
    if (
        plan.replay_cursor.complete
        and len(examples) != plan.replay_cursor.next_input_ordinal
    ):
        _plan_error(
            "complete pack plan does not cover the supplied source length",
            code="packing.pack_plan_replay_coverage",
            context={
                "planned_input_count": plan.replay_cursor.next_input_ordinal,
                "supplied_input_count": len(examples),
            },
        )
    input_by_ordinal = {item.input_ordinal: item for item in plan.inputs}
    for ordinal in sorted(input_by_ordinal):
        if ordinal >= len(examples):
            _plan_error(
                "pack plan references an unavailable input ordinal",
                code="packing.pack_plan_replay_missing",
                context={"input_ordinal": ordinal},
            )
        observed = _read_plan_input(
            examples[ordinal],
            ordinal=ordinal,
            identity_getter=intra_image_order_identity_getter,
        )
        expected = input_by_ordinal[ordinal]
        if observed != expected:
            _plan_error(
                "encoded input identity changed since pack planning",
                code="packing.pack_plan_replay_identity",
                context={"input_ordinal": ordinal, "example_id": expected.example_id},
            )

    packs: list[PackedSequence] = []
    for planned_pack in plan.packs:
        token_ids: list[int] = []
        segments: list[PackedSegment] = []
        for ordinal in planned_pack.input_ordinals:
            item = input_by_ordinal[ordinal]
            example_ids = _input_ids(examples[ordinal], example_id=item.example_id)
            start = len(token_ids)
            token_ids.extend(example_ids)
            segments.append(
                PackedSegment(
                    pack_index=planned_pack.pack_index,
                    segment_index=len(segments),
                    example_index=ordinal,
                    example_id=item.example_id,
                    start=start,
                    end=len(token_ids),
                )
            )
        packs.append(
            _commit_pack(
                pack_index=planned_pack.pack_index,
                input_ids=token_ids,
                segments=segments,
                global_max_length=plan.global_max_length,
            )
        )
    return tuple(packs)


def _plan_online(
    encoded_examples: Iterable[Any],
    *,
    global_max_length: int,
    lookahead: int | None,
    seed: int,
    cursor_byte_budget: int,
    fragment_item_budget: int,
    fragment_byte_budget: int,
    requested_worker_count: int,
    replay_cursor: PackPlanCursor | None,
    predecessor_plan_sha256: str | None,
    max_packs: int | None,
    identity_getter: Callable[[Any], str] | None,
    source_positioned_at_cursor: bool = False,
) -> PackPlan:
    assert lookahead is not None
    algorithm_version = _ALGORITHM_VERSIONS[ONLINE_WINDOW_BINPACK]
    if replay_cursor is None:
        next_input_ordinal = 0
        next_pack_index = 0
        source_prefix_sha256 = _INITIAL_SOURCE_PREFIX_SHA256
        emitted_prefix_sha256 = _INITIAL_EMITTED_PREFIX_SHA256
        pending: list[PackPlanInput] = []
    else:
        if not replay_cursor.resumable:
            _plan_error(
                "online pack-plan cursor is explicitly non-resumable",
                code="packing.pack_plan_non_resumable",
            )
        if replay_cursor.complete:
            _plan_error(
                "completed online pack-plan cursor cannot be resumed",
                code="packing.pack_plan_cursor_complete",
            )
        if (
            replay_cursor.policy != ONLINE_WINDOW_BINPACK
            or replay_cursor.algorithm_version != algorithm_version
            or replay_cursor.global_max_length != global_max_length
            or replay_cursor.window_size is not None
            or replay_cursor.lookahead != lookahead
            or replay_cursor.tie_breaker != _TIE_BREAKERS[ONLINE_WINDOW_BINPACK]
            or replay_cursor.seed != seed
            or replay_cursor.cursor_byte_budget != cursor_byte_budget
        ):
            _plan_error(
                "online pack-plan cursor policy or semantic parameters do not match",
                code="packing.pack_plan_cursor_policy",
            )
        if len(replay_cursor.pending_inputs) > lookahead:
            _plan_error(
                "online pack-plan cursor exceeds configured lookahead",
                code="packing.pack_plan_cursor_bound",
            )
        next_input_ordinal = replay_cursor.next_input_ordinal
        next_pack_index = replay_cursor.next_pack_index
        source_prefix_sha256 = replay_cursor.source_prefix_sha256
        emitted_prefix_sha256 = replay_cursor.emitted_prefix_sha256
        pending = list(replay_cursor.pending_inputs)

    source_prefix_start_count = next_input_ordinal
    source_prefix_start_sha256 = source_prefix_sha256
    emitted_prefix_start_pack_count = next_pack_index
    emitted_prefix_start_sha256 = emitted_prefix_sha256
    source = iter(encoded_examples)
    pending_by_ordinal = {item.input_ordinal: item for item in pending}
    observed_source_prefix_sha256 = _INITIAL_SOURCE_PREFIX_SHA256
    replay_prefix_count = 0 if source_positioned_at_cursor else next_input_ordinal
    for ordinal in range(replay_prefix_count):
        try:
            example = next(source)
        except StopIteration:
            _plan_error(
                "source ended before the replay cursor",
                code="packing.pack_plan_cursor_source",
                context={"next_input_ordinal": next_input_ordinal},
            )
        observed = _read_plan_input(
            example,
            ordinal=ordinal,
            identity_getter=identity_getter,
        )
        observed_source_prefix_sha256 = _extend_prefix_digest(
            observed_source_prefix_sha256,
            observed.to_dict(),
        )
        expected = pending_by_ordinal.get(ordinal)
        if expected is not None:
            if observed != expected:
                _plan_error(
                    "pending input identity changed since cursor serialization",
                    code="packing.pack_plan_cursor_identity",
                    context={"input_ordinal": ordinal},
                )
    if (
        not source_positioned_at_cursor
        and observed_source_prefix_sha256 != source_prefix_sha256
    ):
        _plan_error(
            "source prefix identity changed since cursor serialization",
            code="packing.pack_plan_cursor_prefix_identity",
            context={"next_input_ordinal": next_input_ordinal},
        )

    touched = {item.input_ordinal: item for item in pending}
    rejected: list[RejectedPackPlanInput] = []
    packs: list[PackPlanPack] = []
    exhausted = False
    fragment_item_budget_reached = False
    building_membership = False
    max_pending_observed = len(pending)
    max_cursor_bytes_observed = (
        0 if replay_cursor is None else replay_cursor.serialized_size_bytes
    )

    class _FragmentItemBudgetReached(Exception):
        pass

    def current_cursor(*, complete: bool) -> PackPlanCursor:
        return PackPlanCursor(
            policy=ONLINE_WINDOW_BINPACK,
            algorithm_version=algorithm_version,
            global_max_length=global_max_length,
            window_size=None,
            lookahead=lookahead,
            tie_breaker=_TIE_BREAKERS[ONLINE_WINDOW_BINPACK],
            seed=seed,
            cursor_byte_budget=cursor_byte_budget,
            next_input_ordinal=next_input_ordinal,
            next_pack_index=next_pack_index + len(packs),
            source_prefix_sha256=source_prefix_sha256,
            emitted_prefix_sha256=emitted_prefix_sha256,
            pending_inputs=tuple(pending),
            complete=complete,
            resumable=True,
        )

    def observe_cursor_bytes() -> None:
        nonlocal max_cursor_bytes_observed
        size = current_cursor(complete=False).serialized_size_bytes
        max_cursor_bytes_observed = max(max_cursor_bytes_observed, size)
        if size > cursor_byte_budget:
            _plan_error(
                "serialized online pack-plan cursor exceeds its byte budget",
                code="packing.pack_plan_cursor_byte_bound",
                context={
                    "cursor_byte_budget": cursor_byte_budget,
                    "observed_serialized_cursor_bytes": size,
                    "pending_item_count": len(pending),
                },
            )

    def fill_pending() -> None:
        nonlocal exhausted, next_input_ordinal, max_pending_observed
        nonlocal fragment_item_budget_reached, source_prefix_sha256
        while len(pending) < lookahead and not exhausted:
            if len(touched) >= fragment_item_budget:
                if building_membership:
                    raise _FragmentItemBudgetReached
                fragment_item_budget_reached = True
                break
            try:
                example = next(source)
            except StopIteration:
                exhausted = True
                break
            item = _read_plan_input(
                example,
                ordinal=next_input_ordinal,
                identity_getter=identity_getter,
            )
            next_input_ordinal += 1
            source_prefix_sha256 = _extend_prefix_digest(
                source_prefix_sha256,
                item.to_dict(),
            )
            touched[item.input_ordinal] = item
            reason = _rejection_reason(item, global_max_length=global_max_length)
            if reason is not None:
                rejected.append(RejectedPackPlanInput.from_input(item, reason=reason))
                observe_cursor_bytes()
                continue
            pending.append(item)
            max_pending_observed = max(max_pending_observed, len(pending))
            observe_cursor_bytes()

    observe_cursor_bytes()
    fill_pending()
    while pending and (max_packs is None or len(packs) < max_packs):
        if fragment_item_budget_reached:
            break
        pending_before = list(pending)
        touched_before = dict(touched)
        rejected_count_before = len(rejected)
        next_input_ordinal_before = next_input_ordinal
        source_prefix_sha256_before = source_prefix_sha256
        exhausted_before = exhausted
        building_membership = True
        try:
            membership = _online_membership(
                pending,
                global_max_length=global_max_length,
                seed=seed,
                refill=fill_pending,
            )
        except _FragmentItemBudgetReached:
            pending[:] = pending_before
            touched.clear()
            touched.update(touched_before)
            del rejected[rejected_count_before:]
            next_input_ordinal = next_input_ordinal_before
            source_prefix_sha256 = source_prefix_sha256_before
            exhausted = exhausted_before
            fragment_item_budget_reached = True
            break
        finally:
            building_membership = False
        planned_pack = _memberships_to_packs(
            [membership],
            global_max_length=global_max_length,
            first_pack_index=next_pack_index + len(packs),
        )
        packs.extend(planned_pack)
        emitted_prefix_sha256 = _fold_emitted_prefix(
            emitted_prefix_sha256,
            planned_pack,
        )
        fill_pending()
        observe_cursor_bytes()

    complete = exhausted and not pending
    cursor = current_cursor(complete=complete)
    observe_cursor_bytes()
    return PackPlan(
        policy=ONLINE_WINDOW_BINPACK,
        algorithm_version=algorithm_version,
        global_max_length=global_max_length,
        window_size=None,
        lookahead=lookahead,
        tie_breaker=_TIE_BREAKERS[ONLINE_WINDOW_BINPACK],
        seed=seed,
        cursor_byte_budget=cursor_byte_budget,
        fragment_pack_budget=max_packs,
        fragment_item_budget=fragment_item_budget,
        fragment_byte_budget=fragment_byte_budget,
        requested_worker_count=requested_worker_count,
        worker_count_disposition=_WORKER_COUNT_DISPOSITION,
        worker_count_integration_requirement=_WORKER_COUNT_INTEGRATION_REQUIREMENT,
        receipt_scope=_RESUME_FRAGMENT_SCOPE,
        source_prefix_start_count=source_prefix_start_count,
        source_prefix_start_sha256=source_prefix_start_sha256,
        emitted_prefix_start_pack_count=emitted_prefix_start_pack_count,
        emitted_prefix_start_sha256=emitted_prefix_start_sha256,
        predecessor_plan_sha256=predecessor_plan_sha256,
        inputs=tuple(touched[ordinal] for ordinal in sorted(touched)),
        packs=tuple(packs),
        rejected_examples=tuple(rejected),
        replay_cursor=cursor,
        max_pending_items_observed=max_pending_observed,
        max_serialized_cursor_bytes_observed=max_cursor_bytes_observed,
    )


def _read_all_inputs(
    encoded_examples: Iterable[Any],
    *,
    global_max_length: int,
    identity_getter: Callable[[Any], str] | None,
) -> tuple[
    tuple[PackPlanInput, ...],
    tuple[PackPlanInput, ...],
    tuple[RejectedPackPlanInput, ...],
]:
    inputs: list[PackPlanInput] = []
    accepted: list[PackPlanInput] = []
    rejected: list[RejectedPackPlanInput] = []
    for ordinal, example in enumerate(encoded_examples):
        item = _read_plan_input(
            example,
            ordinal=ordinal,
            identity_getter=identity_getter,
        )
        inputs.append(item)
        reason = _rejection_reason(item, global_max_length=global_max_length)
        if reason is None:
            accepted.append(item)
        else:
            rejected.append(RejectedPackPlanInput.from_input(item, reason=reason))
    return tuple(inputs), tuple(accepted), tuple(rejected)


def _read_plan_input(
    example: Any,
    *,
    ordinal: int,
    identity_getter: Callable[[Any], str] | None,
) -> PackPlanInput:
    example_id = _example_id(example)
    input_ids = _input_ids(example, example_id=example_id)
    if identity_getter is not None:
        identity = identity_getter(example)
    else:
        identity = getattr(example, "intra_image_order_identity", None)
        if identity is None:
            identity = _derived_intra_image_order_identity(example, input_ids=input_ids)
    item = PackPlanInput(
        input_ordinal=ordinal,
        example_id=example_id,
        encoded_length=len(input_ids),
        input_ids_sha256=_input_ids_sha256(input_ids),
        encoded_example_semantics_sha256=(
            _encoded_example_semantics_sha256(example, input_ids=input_ids)
        ),
        intra_image_order_identity=identity,
    )
    _validate_plan_input(item)
    return item


def _derived_intra_image_order_identity(
    example: Any,
    *,
    input_ids: tuple[int, ...],
) -> str:
    spans = getattr(example, "supervised_token_spans", None)
    if isinstance(spans, tuple):
        ordered_rows = [
            {
                "object_id": getattr(span, "object_id", None),
                "field": getattr(span, "field", None),
                "source": getattr(span, "source", None),
                "physical_token_start": getattr(span, "physical_token_start", None),
                "physical_token_end": getattr(span, "physical_token_end", None),
            }
            for span in spans
        ]
        material = json.dumps(
            ordered_rows,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"encoded-row-order-sha256:{hashlib.sha256(material).hexdigest()}"
    material = json.dumps(list(input_ids), separators=(",", ":")).encode("ascii")
    return f"encoded-token-order-sha256:{hashlib.sha256(material).hexdigest()}"


def _encoded_example_semantics_sha256(
    example: Any,
    *,
    input_ids: tuple[int, ...],
) -> str:
    if is_dataclass(example) and not isinstance(example, type):
        budget = _CanonicalContentBudget()
        material: Any = _canonical_execution_value(
            example,
            budget=budget,
            path="encoded_example",
            depth=0,
        )
    else:
        material = {
            "input_ids": list(input_ids),
            "supervised_token_spans": _span_semantics_artifacts(
                getattr(example, "supervised_token_spans", ())
            ),
            "ignored_token_spans": _span_semantics_artifacts(
                getattr(example, "ignored_token_spans", ())
            ),
        }
    try:
        encoded = _canonical_json_bytes(material)
    except (TypeError, ValueError) as exc:
        raise PackingContractError(
            "encoded example semantics are not canonically serializable",
            code="packing.pack_plan_encoded_semantics",
            context={"example_id": _example_id(example)},
            cause=exc,
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class _CanonicalContentBudget:
    nodes: int = 0
    bytes_observed: int = 0

    def observe(self, *, path: str, byte_count: int = 0) -> None:
        self.nodes += 1
        self.bytes_observed += byte_count
        if (
            self.nodes > DEFAULT_PACK_PLAN_CONTENT_NODE_BUDGET
            or self.bytes_observed > DEFAULT_PACK_PLAN_CONTENT_BYTE_BUDGET
        ):
            _plan_error(
                "encoded example execution identity exceeds its canonical digest budget",
                code="packing.pack_plan_encoded_semantics_bound",
                context={
                    "path": path,
                    "node_budget": DEFAULT_PACK_PLAN_CONTENT_NODE_BUDGET,
                    "observed_nodes": self.nodes,
                    "byte_budget": DEFAULT_PACK_PLAN_CONTENT_BYTE_BUDGET,
                    "observed_bytes": self.bytes_observed,
                },
            )


def _canonical_execution_value(
    value: Any,
    *,
    budget: _CanonicalContentBudget,
    path: str,
    depth: int,
) -> Any:
    if depth > 64:
        _plan_error(
            "encoded example execution identity exceeds its nesting bound",
            code="packing.pack_plan_encoded_semantics_bound",
            context={"path": path, "max_depth": 64},
        )
    if isinstance(value, torch.Tensor):
        return _tensor_content_artifact(value, budget=budget, path=path)
    if value is None or isinstance(value, (bool, int)):
        budget.observe(path=path)
        return value
    if isinstance(value, float):
        budget.observe(path=path)
        if not math.isfinite(value):
            _plan_error(
                "encoded example execution identity contains non-finite scalar data",
                code="packing.pack_plan_encoded_semantics",
                context={"path": path},
            )
        return value
    if isinstance(value, str):
        budget.observe(path=path, byte_count=len(value.encode("utf-8")))
        return {"type": "text", "value": value}
    if isinstance(value, Path):
        text = str(value)
        budget.observe(path=path, byte_count=len(text.encode("utf-8")))
        return {"type": "path", "value": text}
    if isinstance(value, bytes):
        budget.observe(path=path, byte_count=len(value))
        return {
            "type": "bytes",
            "byte_count": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        budget.observe(path=path)
        payload: dict[str, Any] = {}
        for field in dataclass_fields(value):
            field_value = getattr(value, field.name)
            is_qwen_image_encoding = (
                type(value).__module__ == "src.qwen.images"
                and type(value).__qualname__ == "QwenImageEncoding"
            )
            if is_qwen_image_encoding and field.name == "image_processor":
                payload[field.name] = {
                    "disposition": "excluded_nonexecution_serialization_state",
                    "present": field_value is not None,
                }
                continue
            payload[field.name] = _canonical_execution_value(
                field_value,
                budget=budget,
                path=f"{path}.{field.name}",
                depth=depth + 1,
            )
        return {
            "type": "dataclass",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": payload,
        }
    if isinstance(value, Mapping):
        budget.observe(path=path)
        if not all(isinstance(key, str) for key in value):
            _plan_error(
                "encoded example execution identity mapping keys must be text",
                code="packing.pack_plan_encoded_semantics",
                context={"path": path},
            )
        return {
            "type": "mapping",
            "items": {
                key: _canonical_execution_value(
                    value[key],
                    budget=budget,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                )
                for key in sorted(value)
            },
        }
    if isinstance(value, (tuple, list)):
        budget.observe(path=path)
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [
                _canonical_execution_value(
                    item,
                    budget=budget,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                )
                for index, item in enumerate(value)
            ],
        }
    _plan_error(
        "encoded example execution identity contains an unsupported value",
        code="packing.pack_plan_encoded_semantics",
        context={"path": path, "value_type": type(value).__name__},
    )


def _tensor_content_artifact(
    tensor: torch.Tensor,
    *,
    budget: _CanonicalContentBudget,
    path: str,
) -> Mapping[str, Any]:
    if tensor.layout != torch.strided:
        _plan_error(
            "encoded example execution identity requires dense strided tensors",
            code="packing.pack_plan_encoded_semantics",
            context={"path": path, "layout": str(tensor.layout)},
        )
    byte_count = tensor.numel() * tensor.element_size()
    budget.observe(path=path, byte_count=byte_count)
    try:
        contiguous = tensor.detach().to(device="cpu").contiguous()
        byte_view = contiguous.view(torch.uint8).reshape(-1).numpy()
        digest = hashlib.sha256(memoryview(byte_view)).hexdigest()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PackingContractError(
            "encoded tensor cannot be canonicalized for pack-plan authentication",
            code="packing.pack_plan_encoded_semantics",
            context={"path": path, "dtype": str(tensor.dtype)},
            cause=exc,
        ) from exc
    return {
        "type": "tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "byte_count": byte_count,
        "sha256": digest,
    }


def _span_semantics_artifacts(spans: Any) -> list[Mapping[str, Any]]:
    if not isinstance(spans, tuple):
        _plan_error(
            "encoded supervision spans must be tuples",
            code="packing.pack_plan_encoded_semantics",
            context={"value_type": type(spans).__name__},
        )
    return [_span_semantics_artifact(span) for span in spans]


def _span_semantics_artifact(span: Any) -> Mapping[str, Any]:
    artifact_builder = getattr(span, "to_artifact_dict", None)
    if callable(artifact_builder):
        artifact = artifact_builder()
        if isinstance(artifact, Mapping):
            return artifact
        _plan_error(
            "encoded span artifact must be a mapping",
            code="packing.pack_plan_encoded_semantics",
            context={"value_type": type(artifact).__name__},
        )
    coordinate_target = getattr(span, "coordinate_target", None)
    coordinate_artifact = None
    if coordinate_target is not None:
        target_builder = getattr(coordinate_target, "to_artifact_dict", None)
        if not callable(target_builder):
            _plan_error(
                "coordinate target must expose an artifact mapping",
                code="packing.pack_plan_encoded_semantics",
            )
        coordinate_artifact = target_builder()
        if not isinstance(coordinate_artifact, Mapping):
            _plan_error(
                "coordinate target artifact must be a mapping",
                code="packing.pack_plan_encoded_semantics",
            )
    return {
        "token_type": getattr(span, "token_type", None),
        "text": getattr(span, "text", None),
        "char_start": getattr(span, "char_start", None),
        "char_end": getattr(span, "char_end", None),
        "chat_char_start": getattr(span, "chat_char_start", None),
        "chat_char_end": getattr(span, "chat_char_end", None),
        "base_token_start": getattr(span, "base_token_start", None),
        "base_token_end": getattr(span, "base_token_end", None),
        "physical_token_start": getattr(span, "physical_token_start", None),
        "physical_token_end": getattr(span, "physical_token_end", None),
        "token_ids": list(getattr(span, "token_ids", ())),
        "object_id": getattr(span, "object_id", None),
        "field": getattr(span, "field", None),
        "source": getattr(span, "source", None),
        "coordinate_target": coordinate_artifact,
    }


def _rejection_reason(item: PackPlanInput, *, global_max_length: int) -> str | None:
    if item.encoded_length == 0:
        return "encoded_example_empty"
    if item.encoded_length > global_max_length:
        return "encoded_example_exceeds_global_max_length"
    return None


def _source_order_memberships(
    inputs: Sequence[PackPlanInput],
    global_max_length: int,
) -> list[list[PackPlanInput]]:
    memberships: list[list[PackPlanInput]] = []
    current: list[PackPlanInput] = []
    current_length = 0
    for item in inputs:
        if current and current_length + item.encoded_length > global_max_length:
            memberships.append(current)
            current = []
            current_length = 0
        current.append(item)
        current_length += item.encoded_length
    if current:
        memberships.append(current)
    return memberships


def _window_binpack_memberships(
    inputs: Sequence[PackPlanInput],
    *,
    global_max_length: int,
    window_size: int,
    seed: int,
) -> list[list[PackPlanInput]]:
    memberships: list[list[PackPlanInput]] = []
    for start in range(0, len(inputs), window_size):
        window = list(inputs[start : start + window_size])
        ordered = sorted(
            window,
            key=lambda item: (
                -item.encoded_length,
                _seeded_identity(seed, item),
                item.input_ordinal,
            ),
        )
        bins: list[list[PackPlanInput]] = []
        bin_lengths: list[int] = []
        for item in ordered:
            candidates = [
                (global_max_length - (used + item.encoded_length), index)
                for index, used in enumerate(bin_lengths)
                if used + item.encoded_length <= global_max_length
            ]
            if candidates:
                _, selected = min(candidates)
                bins[selected].append(item)
                bin_lengths[selected] += item.encoded_length
            else:
                bins.append([item])
                bin_lengths.append(item.encoded_length)
        for bin_items in bins:
            memberships.append(sorted(bin_items, key=lambda item: item.input_ordinal))
    return memberships


def _online_membership(
    pending: list[PackPlanInput],
    *,
    global_max_length: int,
    seed: int,
    refill: Callable[[], None],
) -> list[PackPlanInput]:
    membership = [pending.pop(0)]
    used = membership[0].encoded_length
    refill()
    while pending:
        candidates = [
            item for item in pending if used + item.encoded_length <= global_max_length
        ]
        if not candidates:
            break
        selected = min(
            candidates,
            key=lambda item: (
                global_max_length - (used + item.encoded_length),
                _seeded_identity(seed, item),
                item.input_ordinal,
            ),
        )
        pending.remove(selected)
        membership.append(selected)
        used += selected.encoded_length
        refill()
    return sorted(membership, key=lambda item: item.input_ordinal)


def _memberships_to_packs(
    memberships: Sequence[Sequence[PackPlanInput]],
    *,
    global_max_length: int,
    first_pack_index: int,
) -> tuple[PackPlanPack, ...]:
    return tuple(
        PackPlanPack(
            pack_index=first_pack_index + offset,
            input_ordinals=tuple(item.input_ordinal for item in membership),
            encoded_length=sum(item.encoded_length for item in membership),
            global_max_length=global_max_length,
        )
        for offset, membership in enumerate(memberships)
    )


def _seeded_identity(seed: int, item: PackPlanInput) -> str:
    material = (
        f"{seed}:{item.input_ordinal}:{item.example_id}:"
        f"{item.intra_image_order_identity}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _input_ids_sha256(input_ids: tuple[int, ...]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(input_ids))).hexdigest()


def _fold_source_prefix(
    initial_sha256: str,
    inputs: Sequence[PackPlanInput],
) -> str:
    digest = initial_sha256
    for item in inputs:
        digest = _extend_prefix_digest(digest, item.to_dict())
    return digest


def _fold_emitted_prefix(
    initial_sha256: str,
    packs: Sequence[PackPlanPack],
) -> str:
    digest = initial_sha256
    for pack in packs:
        digest = _extend_prefix_digest(digest, pack.to_dict())
    return digest


def _extend_prefix_digest(previous_sha256: str, payload: Mapping[str, Any]) -> str:
    previous = bytes.fromhex(_strict_sha256(previous_sha256, field="prefix_sha256"))
    canonical = _canonical_json_bytes(payload)
    material = previous + len(canonical).to_bytes(8, byteorder="big") + canonical
    return hashlib.sha256(material).hexdigest()


def _authentication_dict(
    payload: Mapping[str, Any],
    *,
    schema: str,
) -> dict[str, str]:
    return {
        "schema": schema,
        "algorithm": "sha256",
        "canonicalization": PACK_PLAN_CANONICALIZATION,
        "sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
    }


def _verify_authentication(
    authenticated: Mapping[str, Any],
    *,
    schema: str,
    field: str,
) -> None:
    authentication = _strict_mapping(
        authenticated["authentication"],
        keys={"schema", "algorithm", "canonicalization", "sha256"},
        field=f"{field}.authentication",
    )
    if authentication["schema"] != schema:
        _plan_error(
            "unsupported pack-plan authentication schema",
            code="packing.pack_plan_authentication_schema",
            context={"field": field},
        )
    if (
        authentication["algorithm"] != "sha256"
        or authentication["canonicalization"] != PACK_PLAN_CANONICALIZATION
    ):
        _plan_error(
            "unsupported pack-plan authentication algorithm",
            code="packing.pack_plan_authentication_schema",
            context={"field": field},
        )
    observed = _strict_sha256(authentication["sha256"], field=f"{field}.sha256")
    payload = {
        key: value for key, value in authenticated.items() if key != "authentication"
    }
    expected = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if observed != expected:
        _plan_error(
            "pack-plan canonical SHA256 authentication failed",
            code="packing.pack_plan_authentication",
            context={"field": field},
        )


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PackingContractError(
            "pack-plan canonical JSON payload is invalid",
            code="packing.pack_plan_json",
            cause=exc,
        ) from exc


def _strict_json_loads(payload: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(payload, str):
        _plan_error(
            "pack-plan JSON must be text",
            code="packing.pack_plan_json_type",
            context={"field": field},
        )

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PackingContractError(
            "pack-plan JSON is invalid",
            code="packing.pack_plan_json",
            context={"field": field},
            cause=exc,
        ) from exc
    return _strict_mapping_value(decoded, field=field)


def _validate_planner_arguments(
    *,
    global_max_length: int,
    policy: str,
    window_size: int | None,
    lookahead: int | None,
    seed: int,
    worker_count: int,
    cursor_byte_budget: int,
    fragment_item_budget: int,
    fragment_byte_budget: int,
    replay_cursor: PackPlanCursor | None,
    max_packs: int | None,
) -> None:
    _positive_int(global_max_length, field="global_max_length")
    if policy not in _ALGORITHM_VERSIONS:
        _plan_error(
            "unknown packing policy",
            code="packing.pack_plan_policy",
            context={"policy": policy},
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        _plan_error("seed must be an integer", code="packing.pack_plan_seed")
    _positive_int(worker_count, field="worker_count")
    _positive_int(cursor_byte_budget, field="cursor_byte_budget")
    _positive_int(fragment_item_budget, field="fragment_item_budget")
    _positive_int(fragment_byte_budget, field="fragment_byte_budget")
    if max_packs is not None:
        _positive_int(max_packs, field="max_packs")
    if replay_cursor is not None and not isinstance(replay_cursor, PackPlanCursor):
        _plan_error(
            "replay cursor must be a PackPlanCursor",
            code="packing.pack_plan_cursor_type",
        )
    if replay_cursor is not None:
        _validate_cursor(replay_cursor)
    if policy == SOURCE_ORDER_NEXT_FIT:
        if window_size is not None or lookahead is not None:
            _plan_error(
                "source_order_next_fit does not accept window parameters",
                code="packing.pack_plan_parameters",
            )
    elif policy == WINDOW_BINPACK:
        _positive_int(window_size, field="window_size")
        if lookahead is not None:
            _plan_error(
                "window_binpack does not accept lookahead",
                code="packing.pack_plan_parameters",
            )
    else:
        _positive_int(lookahead, field="lookahead")
        if fragment_item_budget < lookahead:
            _plan_error(
                "online fragment item budget must cover the complete lookahead",
                code="packing.pack_plan_fragment_bound",
                context={
                    "fragment_item_budget": fragment_item_budget,
                    "lookahead": lookahead,
                },
            )
        if window_size is not None:
            _plan_error(
                "online_window_binpack does not accept window_size",
                code="packing.pack_plan_parameters",
            )


def _validate_pack_plan(plan: PackPlan) -> None:
    _positive_int(plan.global_max_length, field="global_max_length")
    expected_version = _ALGORITHM_VERSIONS.get(plan.policy)
    if expected_version is None or plan.algorithm_version != expected_version:
        _plan_error(
            "pack-plan policy and algorithm version do not match",
            code="packing.pack_plan_algorithm",
        )
    if plan.tie_breaker != _TIE_BREAKERS[plan.policy]:
        _plan_error(
            "pack-plan tie breaker does not match algorithm version",
            code="packing.pack_plan_tie_breaker",
        )
    _positive_int(plan.requested_worker_count, field="requested_worker_count")
    if plan.worker_count_disposition != _WORKER_COUNT_DISPOSITION:
        _plan_error(
            "unsupported worker-count disposition",
            code="packing.pack_plan_worker_disposition",
        )
    if (
        plan.worker_count_integration_requirement
        != _WORKER_COUNT_INTEGRATION_REQUIREMENT
    ):
        _plan_error(
            "unsupported worker-count integration requirement",
            code="packing.pack_plan_worker_disposition",
        )
    if plan.receipt_scope not in {_COMPLETE_PLAN_SCOPE, _RESUME_FRAGMENT_SCOPE}:
        _plan_error(
            "unsupported pack-plan receipt scope",
            code="packing.pack_plan_receipt_scope",
        )
    _validate_planner_arguments(
        global_max_length=plan.global_max_length,
        policy=plan.policy,
        window_size=plan.window_size,
        lookahead=plan.lookahead,
        seed=plan.seed,
        worker_count=1,
        cursor_byte_budget=plan.cursor_byte_budget,
        fragment_item_budget=plan.fragment_item_budget,
        fragment_byte_budget=plan.fragment_byte_budget,
        replay_cursor=None,
        max_packs=None,
    )
    _validate_cursor(plan.replay_cursor)
    if plan.policy == ONLINE_WINDOW_BINPACK:
        if plan.fragment_pack_budget is None:
            _plan_error(
                "online pack-plan receipt must declare a finite pack budget",
                code="packing.pack_plan_fragment_bound",
            )
        if len(plan.packs) > plan.fragment_pack_budget:
            _plan_error(
                "online pack-plan fragment exceeds its pack budget",
                code="packing.pack_plan_fragment_pack_bound",
            )
        if len(plan.inputs) > plan.fragment_item_budget:
            _plan_error(
                "online pack-plan fragment exceeds its item budget",
                code="packing.pack_plan_fragment_item_bound",
                context={
                    "fragment_item_budget": plan.fragment_item_budget,
                    "observed_input_count": len(plan.inputs),
                },
            )
        serialized_size = plan.serialized_size_bytes
        if serialized_size > plan.fragment_byte_budget:
            _plan_error(
                "online pack-plan fragment exceeds its byte budget",
                code="packing.pack_plan_fragment_byte_bound",
                context={
                    "fragment_byte_budget": plan.fragment_byte_budget,
                    "observed_serialized_fragment_bytes": serialized_size,
                },
            )
    elif plan.fragment_pack_budget is not None:
        _plan_error(
            "complete non-online plans cannot declare a fragment pack budget",
            code="packing.pack_plan_fragment_bound",
        )
    if (
        plan.replay_cursor.policy != plan.policy
        or plan.replay_cursor.algorithm_version != plan.algorithm_version
        or plan.replay_cursor.global_max_length != plan.global_max_length
        or plan.replay_cursor.window_size != plan.window_size
        or plan.replay_cursor.lookahead != plan.lookahead
        or plan.replay_cursor.tie_breaker != plan.tie_breaker
        or plan.replay_cursor.seed != plan.seed
        or plan.replay_cursor.cursor_byte_budget != plan.cursor_byte_budget
    ):
        _plan_error(
            "pack plan and cursor identities do not match",
            code="packing.pack_plan_cursor_policy",
        )
    if plan.policy == ONLINE_WINDOW_BINPACK:
        assert plan.lookahead is not None
        if len(plan.replay_cursor.pending_inputs) > plan.lookahead:
            _plan_error(
                "pack-plan cursor exceeds declared lookahead",
                code="packing.pack_plan_cursor_bound",
            )
        if plan.max_pending_items_observed > plan.lookahead:
            _plan_error(
                "observed pending state exceeds declared lookahead",
                code="packing.pack_plan_pending_bound",
            )
        if not plan.replay_cursor.resumable:
            _plan_error(
                "online pack plan is not resumable",
                code="packing.pack_plan_non_resumable",
            )
    if plan.max_pending_items_observed < 0:
        _plan_error(
            "observed pending state must be non-negative",
            code="packing.pack_plan_pending_bound",
        )
    _nonnegative_int(
        plan.max_serialized_cursor_bytes_observed,
        field="max_serialized_cursor_bytes_observed",
    )
    terminal_cursor_bytes = plan.replay_cursor.serialized_size_bytes
    if (
        terminal_cursor_bytes > plan.cursor_byte_budget
        or plan.max_serialized_cursor_bytes_observed > plan.cursor_byte_budget
        or plan.max_serialized_cursor_bytes_observed < terminal_cursor_bytes
    ):
        _plan_error(
            "serialized cursor byte high-water is invalid or exceeds its budget",
            code="packing.pack_plan_cursor_byte_bound",
            context={
                "cursor_byte_budget": plan.cursor_byte_budget,
                "terminal_serialized_cursor_bytes": terminal_cursor_bytes,
                "max_serialized_cursor_bytes_observed": (
                    plan.max_serialized_cursor_bytes_observed
                ),
            },
        )

    _nonnegative_int(plan.source_prefix_start_count, field="source_prefix_start_count")
    _nonnegative_int(
        plan.emitted_prefix_start_pack_count,
        field="emitted_prefix_start_pack_count",
    )
    _strict_sha256(
        plan.source_prefix_start_sha256,
        field="source_prefix_start_sha256",
    )
    _strict_sha256(
        plan.emitted_prefix_start_sha256,
        field="emitted_prefix_start_sha256",
    )
    if plan.predecessor_plan_sha256 is not None:
        _strict_sha256(
            plan.predecessor_plan_sha256,
            field="predecessor_plan_sha256",
        )
    has_prior_prefix = (
        plan.source_prefix_start_count > 0 or plan.emitted_prefix_start_pack_count > 0
    )
    if has_prior_prefix != (plan.predecessor_plan_sha256 is not None):
        _plan_error(
            "resume fragment predecessor digest does not match its prefix boundary",
            code="packing.pack_plan_cursor_predecessor",
        )
    if (
        plan.source_prefix_start_count > plan.replay_cursor.next_input_ordinal
        or plan.emitted_prefix_start_pack_count > plan.replay_cursor.next_pack_index
    ):
        _plan_error(
            "pack-plan prefix start cannot follow the terminal cursor",
            code="packing.pack_plan_prefix_coverage",
        )

    input_by_ordinal: dict[int, PackPlanInput] = {}
    for item in plan.inputs:
        _validate_plan_input(item)
        if item.input_ordinal in input_by_ordinal:
            _plan_error(
                "pack plan contains a duplicate input ordinal",
                code="packing.pack_plan_duplicate_input",
            )
        input_by_ordinal[item.input_ordinal] = item
    if tuple(input_by_ordinal) != tuple(sorted(input_by_ordinal)):
        _plan_error(
            "pack-plan inputs must be serialized in input-ordinal order",
            code="packing.pack_plan_input_order",
        )
    if any(
        ordinal >= plan.replay_cursor.next_input_ordinal for ordinal in input_by_ordinal
    ):
        _plan_error(
            "pack-plan inputs must precede the next source cursor",
            code="packing.pack_plan_cursor_ordinal",
        )
    new_source_ordinals = tuple(
        ordinal
        for ordinal in input_by_ordinal
        if ordinal >= plan.source_prefix_start_count
    )
    expected_new_source_ordinals = tuple(
        range(
            plan.source_prefix_start_count,
            plan.replay_cursor.next_input_ordinal,
        )
    )
    if new_source_ordinals != expected_new_source_ordinals:
        _plan_error(
            "pack-plan fragment does not exactly cover its newly consumed source prefix",
            code="packing.pack_plan_prefix_coverage",
        )
    observed_source_prefix = _fold_source_prefix(
        plan.source_prefix_start_sha256,
        tuple(input_by_ordinal[ordinal] for ordinal in new_source_ordinals),
    )
    if observed_source_prefix != plan.replay_cursor.source_prefix_sha256:
        _plan_error(
            "pack-plan source prefix digest does not match its terminal cursor",
            code="packing.pack_plan_prefix_identity",
        )

    disposition: dict[int, str] = {}
    expected_pack_index: int | None = None
    for pack in plan.packs:
        _validate_plan_pack(pack)
        if pack.global_max_length != plan.global_max_length:
            _plan_error(
                "pack capacity differs from plan capacity",
                code="packing.pack_plan_capacity",
            )
        if expected_pack_index is not None and pack.pack_index != expected_pack_index:
            _plan_error(
                "pack indices must be contiguous",
                code="packing.pack_plan_pack_index",
            )
        expected_pack_index = pack.pack_index + 1
        encoded_length = 0
        for ordinal in pack.input_ordinals:
            item = input_by_ordinal.get(ordinal)
            if item is None:
                _plan_error(
                    "pack membership references an unknown input ordinal",
                    code="packing.pack_plan_membership",
                )
            if ordinal in disposition:
                _plan_error(
                    "input ordinal appears more than once in pack-plan state",
                    code="packing.pack_plan_duplicate_coverage",
                    context={"input_ordinal": ordinal},
                )
            if (
                _rejection_reason(item, global_max_length=plan.global_max_length)
                is not None
            ):
                _plan_error(
                    "rejected-length input cannot appear in pack membership",
                    code="packing.pack_plan_membership",
                )
            disposition[ordinal] = "packed"
            encoded_length += item.encoded_length
        if encoded_length != pack.encoded_length:
            _plan_error(
                "pack encoded length does not match its membership",
                code="packing.pack_plan_pack_length",
            )

    if plan.packs:
        if plan.packs[0].pack_index != plan.emitted_prefix_start_pack_count:
            _plan_error(
                "pack-plan emitted fragment does not start at its bound pack prefix",
                code="packing.pack_plan_prefix_coverage",
            )
    if (
        plan.emitted_prefix_start_pack_count + len(plan.packs)
        != plan.replay_cursor.next_pack_index
    ):
        _plan_error(
            "pack-plan emitted fragment does not exactly reach its terminal cursor",
            code="packing.pack_plan_prefix_coverage",
        )
    observed_emitted_prefix = _fold_emitted_prefix(
        plan.emitted_prefix_start_sha256,
        plan.packs,
    )
    if observed_emitted_prefix != plan.replay_cursor.emitted_prefix_sha256:
        _plan_error(
            "pack-plan emitted prefix digest does not match its terminal cursor",
            code="packing.pack_plan_prefix_identity",
        )

    for rejected in plan.rejected_examples:
        item = input_by_ordinal.get(rejected.input_ordinal)
        if item is None or item.to_dict() != {
            key: value for key, value in rejected.to_dict().items() if key != "reason"
        }:
            _plan_error(
                "rejected input identity does not match plan inputs",
                code="packing.pack_plan_rejected_identity",
            )
        expected_reason = _rejection_reason(
            item, global_max_length=plan.global_max_length
        )
        if expected_reason is None or rejected.reason != expected_reason:
            _plan_error(
                "rejected input reason does not match its encoded length",
                code="packing.pack_plan_rejected_reason",
            )
        if rejected.input_ordinal in disposition:
            _plan_error(
                "input ordinal appears more than once in pack-plan state",
                code="packing.pack_plan_duplicate_coverage",
            )
        disposition[rejected.input_ordinal] = "rejected"

    for pending in plan.replay_cursor.pending_inputs:
        item = input_by_ordinal.get(pending.input_ordinal)
        if item != pending:
            _plan_error(
                "pending cursor input identity does not match plan inputs",
                code="packing.pack_plan_cursor_identity",
            )
        if (
            _rejection_reason(item, global_max_length=plan.global_max_length)
            is not None
        ):
            _plan_error(
                "rejected-length input cannot remain pending",
                code="packing.pack_plan_cursor_identity",
            )
        if pending.input_ordinal in disposition:
            _plan_error(
                "input ordinal appears more than once in pack-plan state",
                code="packing.pack_plan_duplicate_coverage",
            )
        disposition[pending.input_ordinal] = "pending"

    if set(input_by_ordinal) != set(disposition):
        _plan_error(
            "pack-plan inputs must have exactly one packed, rejected, or pending disposition",
            code="packing.pack_plan_coverage",
        )
    if plan.receipt_scope == _COMPLETE_PLAN_SCOPE:
        if (
            not plan.replay_cursor.complete
            or plan.source_prefix_start_count != 0
            or plan.source_prefix_start_sha256 != _INITIAL_SOURCE_PREFIX_SHA256
            or plan.emitted_prefix_start_pack_count != 0
            or plan.emitted_prefix_start_sha256 != _INITIAL_EMITTED_PREFIX_SHA256
            or tuple(input_by_ordinal)
            != tuple(range(plan.replay_cursor.next_input_ordinal))
        ):
            _plan_error(
                "complete pack plan must exactly cover its full source and emitted prefixes",
                code="packing.pack_plan_complete_coverage",
            )
    if plan.replay_cursor.complete and plan.replay_cursor.pending_inputs:
        _plan_error(
            "complete cursor cannot retain pending inputs",
            code="packing.pack_plan_cursor_complete",
        )
    if (
        plan.packs
        and plan.replay_cursor.next_pack_index != plan.packs[-1].pack_index + 1
    ):
        _plan_error(
            "cursor pack index does not follow emitted packs",
            code="packing.pack_plan_cursor_pack_index",
        )


def _validate_plan_input(item: Any) -> None:
    _nonnegative_int(item.input_ordinal, field="input_ordinal")
    if not isinstance(item.example_id, str) or not item.example_id:
        _plan_error(
            "pack-plan example id must be non-empty text",
            code="packing.pack_plan_example_id",
        )
    _nonnegative_int(item.encoded_length, field="encoded_length")
    _strict_sha256(item.input_ids_sha256, field="input_ids_sha256")
    _strict_sha256(
        item.encoded_example_semantics_sha256,
        field="encoded_example_semantics_sha256",
    )
    if (
        not isinstance(item.intra_image_order_identity, str)
        or not item.intra_image_order_identity
    ):
        _plan_error(
            "intra-image order identity must be non-empty text",
            code="packing.pack_plan_intra_image_identity",
        )


def _validate_plan_pack(pack: PackPlanPack) -> None:
    _nonnegative_int(pack.pack_index, field="pack_index")
    if not pack.input_ordinals:
        _plan_error(
            "pack-plan pack cannot be empty",
            code="packing.pack_plan_empty_pack",
        )
    if len(set(pack.input_ordinals)) != len(pack.input_ordinals):
        _plan_error(
            "pack-plan pack contains duplicate input ordinals",
            code="packing.pack_plan_duplicate_coverage",
        )
    for ordinal in pack.input_ordinals:
        _nonnegative_int(ordinal, field="input_ordinal")
    _positive_int(pack.global_max_length, field="global_max_length")
    _positive_int(pack.encoded_length, field="encoded_length")
    if pack.encoded_length > pack.global_max_length:
        _plan_error(
            "pack-plan pack exceeds capacity",
            code="packing.pack_plan_capacity",
        )


def _validate_cursor(cursor: PackPlanCursor) -> None:
    expected_version = _ALGORITHM_VERSIONS.get(cursor.policy)
    if expected_version is None or cursor.algorithm_version != expected_version:
        _plan_error(
            "pack-plan cursor policy and algorithm version do not match",
            code="packing.pack_plan_cursor_policy",
        )
    _validate_planner_arguments(
        global_max_length=cursor.global_max_length,
        policy=cursor.policy,
        window_size=cursor.window_size,
        lookahead=cursor.lookahead,
        seed=cursor.seed,
        worker_count=1,
        cursor_byte_budget=cursor.cursor_byte_budget,
        fragment_item_budget=max(
            DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET,
            cursor.lookahead or 1,
        ),
        fragment_byte_budget=DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET,
        replay_cursor=None,
        max_packs=None,
    )
    if cursor.tie_breaker != _TIE_BREAKERS[cursor.policy]:
        _plan_error(
            "pack-plan cursor tie breaker does not match algorithm version",
            code="packing.pack_plan_cursor_policy",
        )
    _nonnegative_int(cursor.next_input_ordinal, field="next_input_ordinal")
    _nonnegative_int(cursor.next_pack_index, field="next_pack_index")
    _strict_sha256(cursor.source_prefix_sha256, field="source_prefix_sha256")
    _strict_sha256(cursor.emitted_prefix_sha256, field="emitted_prefix_sha256")
    if (
        cursor.next_input_ordinal == 0
        and cursor.source_prefix_sha256 != _INITIAL_SOURCE_PREFIX_SHA256
    ):
        _plan_error(
            "zero-length source cursor must use the initial prefix digest",
            code="packing.pack_plan_cursor_prefix_identity",
        )
    if (
        cursor.next_pack_index == 0
        and cursor.emitted_prefix_sha256 != _INITIAL_EMITTED_PREFIX_SHA256
    ):
        _plan_error(
            "zero-length emitted cursor must use the initial prefix digest",
            code="packing.pack_plan_cursor_prefix_identity",
        )
    if not isinstance(cursor.complete, bool) or not isinstance(cursor.resumable, bool):
        _plan_error(
            "pack-plan cursor flags must be booleans",
            code="packing.pack_plan_cursor_flags",
        )
    seen: set[int] = set()
    for item in cursor.pending_inputs:
        _validate_plan_input(item)
        if item.input_ordinal >= cursor.next_input_ordinal:
            _plan_error(
                "pending input must precede the next source cursor",
                code="packing.pack_plan_cursor_ordinal",
            )
        if item.input_ordinal in seen:
            _plan_error(
                "pack-plan cursor contains duplicate pending inputs",
                code="packing.pack_plan_cursor_duplicate",
            )
        seen.add(item.input_ordinal)
    if cursor.complete and cursor.pending_inputs:
        _plan_error(
            "complete cursor cannot retain pending inputs",
            code="packing.pack_plan_cursor_complete",
        )
    cursor_size = cursor.serialized_size_bytes
    if cursor_size > cursor.cursor_byte_budget:
        _plan_error(
            "serialized online pack-plan cursor exceeds its byte budget",
            code="packing.pack_plan_cursor_byte_bound",
            context={
                "cursor_byte_budget": cursor.cursor_byte_budget,
                "observed_serialized_cursor_bytes": cursor_size,
            },
        )


def _strict_mapping(
    payload: Any,
    *,
    keys: set[str],
    field: str,
) -> Mapping[str, Any]:
    data = _strict_mapping_value(payload, field=field)
    observed = set(data)
    if observed != keys:
        _plan_error(
            "serialized pack-plan object has missing or unknown fields",
            code="packing.pack_plan_fields",
            context={
                "field": field,
                "missing": sorted(keys - observed),
                "unknown": sorted(observed - keys),
            },
        )
    return data


def _strict_mapping_value(payload: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        _plan_error(
            "serialized pack-plan field must be an object",
            code="packing.pack_plan_field_type",
            context={"field": field},
        )
    return payload


def _strict_list(payload: Any, *, field: str) -> list[Any]:
    if not isinstance(payload, list):
        _plan_error(
            "serialized pack-plan field must be an array",
            code="packing.pack_plan_field_type",
            context={"field": field},
        )
    return payload


def _strict_string(payload: Any, *, field: str) -> str:
    if not isinstance(payload, str) or not payload:
        _plan_error(
            "serialized pack-plan field must be non-empty text",
            code="packing.pack_plan_field_type",
            context={"field": field},
        )
    return payload


def _strict_sha256(payload: Any, *, field: str) -> str:
    value = _strict_string(payload, field=field)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        _plan_error(
            "serialized pack-plan field must be lowercase SHA256 hex",
            code="packing.pack_plan_field_type",
            context={"field": field},
        )
    return value


def _optional_sha256(payload: Any, *, field: str) -> str | None:
    if payload is None:
        return None
    return _strict_sha256(payload, field=field)


def _strict_int(payload: Any, *, field: str) -> int:
    if not isinstance(payload, int) or isinstance(payload, bool):
        _plan_error(
            "serialized pack-plan field must be an integer",
            code="packing.pack_plan_field_type",
            context={"field": field},
        )
    return payload


def _strict_float(payload: Any, *, field: str) -> float:
    if (
        not isinstance(payload, (int, float))
        or isinstance(payload, bool)
        or not math.isfinite(float(payload))
    ):
        _plan_error(
            "serialized pack-plan field must be finite numeric data",
            code="packing.pack_plan_field_type",
            context={"field": field},
        )
    return float(payload)


def _strict_bool(payload: Any, *, field: str) -> bool:
    if not isinstance(payload, bool):
        _plan_error(
            "serialized pack-plan field must be a boolean",
            code="packing.pack_plan_field_type",
            context={"field": field},
        )
    return payload


def _optional_positive_int(payload: Any, *, field: str) -> int | None:
    if payload is None:
        return None
    value = _strict_int(payload, field=field)
    _positive_int(value, field=field)
    return value


def _positive_int(payload: Any, *, field: str) -> None:
    if not isinstance(payload, int) or isinstance(payload, bool) or payload <= 0:
        _plan_error(
            "pack-plan parameter must be a positive integer",
            code="packing.pack_plan_parameter",
            context={"field": field},
        )


def _nonnegative_int(payload: Any, *, field: str) -> None:
    if not isinstance(payload, int) or isinstance(payload, bool) or payload < 0:
        _plan_error(
            "pack-plan field must be a non-negative integer",
            code="packing.pack_plan_field_type",
            context={"field": field},
        )


def _plan_error(
    message: str,
    *,
    code: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    raise PackingContractError(message, code=code, context=context)


__all__ = [
    "DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET",
    "DEFAULT_PACK_PLAN_CONTENT_BYTE_BUDGET",
    "DEFAULT_PACK_PLAN_CONTENT_NODE_BUDGET",
    "DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET",
    "DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET",
    "ONLINE_WINDOW_BINPACK",
    "PACK_PLAN_AUTHENTICATION_SCHEMA",
    "PACK_PLAN_CANONICALIZATION",
    "PACK_PLAN_CURSOR_SCHEMA",
    "PACK_PLAN_CURSOR_AUTHENTICATION_SCHEMA",
    "PACK_PLAN_CURSOR_SCHEMA_VERSION",
    "PACK_PLAN_PREFIX_DIGEST_SCHEMA",
    "PACK_PLAN_SCHEMA",
    "PACK_PLAN_SCHEMA_VERSION",
    "PACK_PLAN_STREAM_AUTHENTICATION_SCHEMA",
    "PACK_PLAN_STREAM_RECEIPT_SCHEMA",
    "PACK_PLAN_STREAM_RECEIPT_SCHEMA_VERSION",
    "SOURCE_ORDER_NEXT_FIT",
    "WINDOW_BINPACK",
    "PackPlan",
    "PackPlanCursor",
    "PackPlanInput",
    "PackPlanPack",
    "PackPlanStreamReceipt",
    "PackedSegment",
    "PackedSequence",
    "RejectedPackPlanInput",
    "build_pack_plan_policy_identity",
    "create_pack_plan",
    "plan_packed_sequences",
    "replay_pack_plan",
    "stream_online_pack_plan_fragments",
    "verify_pack_plan_stream_fragments",
]

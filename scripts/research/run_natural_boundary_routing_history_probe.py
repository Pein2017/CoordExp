#!/usr/bin/env python3
"""CPU-testable natural-boundary release and residual-history probe.

The old owner-interface runner starts every free row with a caller-injected
``object_ref_start`` token.  That is useful for post-opener diagnostics, but it
cannot answer whether the model naturally admits a row.  This module owns the
small, independent seam for the new question:

* an event prefix is exactly ``prompt + exact history``;
* no opener is appended at event or row boundaries;
* every next-token decision is a scalar, full-prefix recompute with
  ``use_cache=False``;
* the natural first token is classified separately from a native STOP, a
  malformed row, an over-continuation, and a budget stop; and
* residual interventions (N00/N01/N10/N20) and attention-mask interventions
  are injected through narrow callbacks rather than implementing K/H here.

The production adapter can supply multimodal model inputs and a callback that
installs a real residual hook or attention mask.  Focused tests use the same
protocol with tiny fake models; this file never loads a checkpoint or chooses
an attention-mask policy.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import torch


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
SCHEMA_VERSION = "natural_boundary_routing_history_probe.v1"
TECHNICAL_INVALIDITY = "technical_invalidity"
MAX_ROW_TOKENS = 256
MAX_ROWS = 3
NOOP_TOLERANCE = 1e-4


class TechnicalInvalid(ValueError):
    """A mechanical contract failure that must not become a scientific null."""


@runtime_checkable
class AttentionMaskActuator(Protocol):
    """Narrow callback seam for a caller-owned attention-mask intervention.

    The callback may return a tensor (used as ``attention_mask``), a mapping
    of model keyword arguments, or ``None``.  It may also return a mapping with
    ``receipt``/``mask_receipt`` metadata; those keys are persisted but are not
    forwarded to the model.  K/H construction and validation deliberately live
    outside this module so another module can provide them without a write
    conflict.
    """

    def __call__(
        self,
        context: "NaturalEventContext",
        *,
        input_ids: torch.Tensor,
        step: int,
        row_index: int,
        arm_id: str,
    ) -> Any: ...


@runtime_checkable
class ResidualActuator(Protocol):
    """Callback seam for installing one residual N-arm around a scalar call.

    A callback returns a context manager.  Its implementation may install a
    block hook, use a model-specific adapter, or simply record a test receipt.
    Returning ``None`` is equivalent to a no-op context.  N00 never invokes
    this callback; N01/N10/N20 require it (or an object exposing ``install``).
    """

    def __call__(
        self,
        model: Any,
        *,
        context: "NaturalEventContext",
        request: "ResidualRequest",
        input_ids: torch.Tensor,
        step: int,
        row_index: int,
    ) -> Any: ...


@dataclass(frozen=True)
class NativeRowContract:
    """Native compact object-row grammar for a closed or commit wrapper."""

    opener_token_id: int
    object_ref_end_token_id: int
    box_start_token_id: int
    box_end_token_id: int
    coordinate_token_start_id: int
    coordinate_bin_count: int = 1000
    coordinate_count: int = 4
    commit_token_id: int | None = None
    stop_token_id: int | None = None
    assistant_format: Literal["object_box_closed", "object_box_commit"] = "object_box_closed"
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        ints = (
            self.opener_token_id,
            self.object_ref_end_token_id,
            self.box_start_token_id,
            self.box_end_token_id,
            self.coordinate_token_start_id,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ints):
            raise TechnicalInvalid("row contract token IDs must be non-negative integers")
        if self.coordinate_count != 4:
            raise TechnicalInvalid("natural row scoring requires exactly four coordinates")
        if self.coordinate_bin_count <= 0:
            raise TechnicalInvalid("coordinate_bin_count must be positive")
        if self.assistant_format not in {"object_box_closed", "object_box_commit"}:
            raise TechnicalInvalid("assistant_format must be object_box_closed or object_box_commit")
        if self.assistant_format == "object_box_commit" and self.commit_token_id is None:
            raise TechnicalInvalid("commit wrapper requires commit_token_id")
        if self.assistant_format == "object_box_closed" and self.commit_token_id is not None:
            raise TechnicalInvalid("closed wrapper must not declare commit_token_id")
        if self.stop_token_id is not None and (
            isinstance(self.stop_token_id, bool) or not isinstance(self.stop_token_id, int) or self.stop_token_id < 0
        ):
            raise TechnicalInvalid("stop_token_id must be a non-negative integer or None")
        stops = _token_ids(self.stop_token_ids, label="stop_token_ids")
        if len(set(stops)) != len(stops):
            raise TechnicalInvalid("stop_token_ids must be unique")
        object.__setattr__(self, "stop_token_ids", stops)

    @classmethod
    def closed(
        cls,
        *,
        opener_token_id: int,
        object_ref_end_token_id: int,
        box_start_token_id: int,
        box_end_token_id: int,
        coordinate_token_start_id: int,
        coordinate_bin_count: int = 1000,
        stop_token_id: int | None = None,
        stop_token_ids: Sequence[int] = (),
    ) -> "NativeRowContract":
        return cls(
            opener_token_id,
            object_ref_end_token_id,
            box_start_token_id,
            box_end_token_id,
            coordinate_token_start_id,
            coordinate_bin_count,
            4,
            None,
            stop_token_id,
            "object_box_closed",
            tuple(int(value) for value in stop_token_ids),
        )

    @classmethod
    def commit(
        cls,
        *,
        opener_token_id: int,
        object_ref_end_token_id: int,
        box_start_token_id: int,
        box_end_token_id: int,
        coordinate_token_start_id: int,
        commit_token_id: int,
        coordinate_bin_count: int = 1000,
        stop_token_id: int | None = None,
        stop_token_ids: Sequence[int] = (),
    ) -> "NativeRowContract":
        return cls(
            opener_token_id,
            object_ref_end_token_id,
            box_start_token_id,
            box_end_token_id,
            coordinate_token_start_id,
            coordinate_bin_count,
            4,
            commit_token_id,
            stop_token_id,
            "object_box_commit",
            tuple(int(value) for value in stop_token_ids),
        )

    @property
    def closure_token_id(self) -> int:
        return self.commit_token_id if self.commit_token_id is not None else self.box_end_token_id

    @property
    def terminal_closure_token_ids(self) -> tuple[int, ...]:
        if self.commit_token_id is None:
            return (self.box_end_token_id,)
        return (self.box_end_token_id, self.commit_token_id)

    @property
    def effective_stop_token_ids(self) -> tuple[int, ...]:
        values = list(self.stop_token_ids)
        if self.stop_token_id is not None and self.stop_token_id not in values:
            values.append(self.stop_token_id)
        return tuple(values)

    def receipt(self) -> dict[str, Any]:
        return {
            "assistant_format": self.assistant_format,
            "opener_token_id": self.opener_token_id,
            "object_ref_end_token_id": self.object_ref_end_token_id,
            "box_start_token_id": self.box_start_token_id,
            "box_end_token_id": self.box_end_token_id,
            "coordinate_token_start_id": self.coordinate_token_start_id,
            "coordinate_bin_count": self.coordinate_bin_count,
            "coordinate_count": self.coordinate_count,
            "commit_token_id": self.commit_token_id,
            "closure_token_id": self.closure_token_id,
            "stop_token_id": self.stop_token_id,
            "stop_token_ids": list(self.effective_stop_token_ids),
        }

    def parse_row(self, row_token_ids: Sequence[int]) -> dict[str, Any]:
        """Parse one row including its generated opener.

        The parser is intentionally token-native.  It does not decode text and
        does not accept a second opener.  ``status=complete`` is the only
        parser result that can receive a closure endpoint score.
        """

        row = _token_ids(row_token_ids, label="row_token_ids", allow_empty=False)
        result: dict[str, Any] = {
            "status": "invalid",
            "reason": None,
            "row_token_ids": list(row),
            "description_token_ids": [],
            "coordinate_token_ids": [],
            "coordinate_bins": [],
        }
        if row[0] != self.opener_token_id:
            result["reason"] = "missing_opener"
            return result
        if any(token in self.effective_stop_token_ids for token in row):
            result["reason"] = "native_stop_inside_row"
            return result
        if row.count(self.object_ref_end_token_id) != 1:
            result["reason"] = "missing_or_repeated_object_ref_end"
            return result
        end = row.index(self.object_ref_end_token_id)
        description = row[1:end]
        result["description_token_ids"] = list(description)
        if not description:
            result["reason"] = "empty_description"
            return result
        forbidden = {
            self.opener_token_id,
            self.object_ref_end_token_id,
            self.box_start_token_id,
            self.box_end_token_id,
        }
        if self.commit_token_id is not None:
            forbidden.add(self.commit_token_id)
        if any(token in forbidden for token in description):
            result["reason"] = "wrapper_token_in_description"
            return result
        if end + 1 >= len(row) or row[end + 1] != self.box_start_token_id:
            result["reason"] = "missing_box_start"
            return result
        coord_start = end + 2
        coord_end = coord_start + self.coordinate_count
        if coord_end >= len(row):
            result["reason"] = "truncated_coordinate_span"
            return result
        coords = row[coord_start:coord_end]
        result["coordinate_token_ids"] = list(coords)
        lower = self.coordinate_token_start_id
        upper = lower + self.coordinate_bin_count
        if any(token < lower or token >= upper for token in coords):
            result["reason"] = "coordinate_token_out_of_range"
            return result
        if row[coord_end] != self.box_end_token_id:
            result["reason"] = "missing_box_end_or_extra_coordinate"
            return result
        if self.commit_token_id is None:
            if coord_end != len(row) - 1:
                result["reason"] = "over_continuation"
                return result
            closure = (coord_end,)
        else:
            if coord_end + 1 >= len(row) or row[coord_end + 1] != self.commit_token_id:
                result["reason"] = "missing_commit"
                return result
            if row.count(self.commit_token_id) != 1 or coord_end + 1 != len(row) - 1:
                result["reason"] = "over_continuation"
                return result
            closure = (coord_end, coord_end + 1)
        result.update(
            {
                "status": "complete",
                "reason": None,
                "coordinate_bins": [token - lower for token in coords],
                "indices": {
                    "row_entry": (0,),
                    "description": tuple(range(1, end)),
                    "geometry": tuple(range(end + 1, coord_end)),
                    "x1": (coord_start,),
                    "y1": (coord_start + 1,),
                    "x2": (coord_start + 2,),
                    "y2": (coord_start + 3,),
                    "closure": closure,
                    "full_row": tuple(range(len(row))),
                },
            }
        )
        return result


@dataclass(frozen=True)
class NaturalEventContext:
    """Exact event boundary for pre-opener natural release."""

    event_id: str
    prompt_token_ids: tuple[int, ...]
    exact_history_token_ids: tuple[int, ...]
    row_contract: NativeRowContract
    model_inputs: Mapping[str, Any] = field(default_factory=dict)
    max_row_tokens: int = MAX_ROW_TOKENS
    max_rows: int = MAX_ROWS
    max_new_tokens: int | None = None
    position_ids_builder: Callable[[int], Any] | None = None
    latest_history_row_token_ids: tuple[int, ...] = ()
    history_prefix_width: int | None = None

    def __post_init__(self) -> None:
        if not str(self.event_id).strip():
            raise TechnicalInvalid("event_id must be non-empty")
        prompt = _token_ids(self.prompt_token_ids, label="prompt_token_ids")
        history = _token_ids(self.exact_history_token_ids, label="exact_history_token_ids")
        object.__setattr__(self, "prompt_token_ids", prompt)
        object.__setattr__(self, "exact_history_token_ids", history)
        if not prompt and not history:
            raise TechnicalInvalid("prompt + exact history prefix must be non-empty")
        if self.prefix_token_ids[-1] == self.row_contract.opener_token_id:
            raise TechnicalInvalid("pre_opener_natural prefix must not end with an injected opener")
        if not 1 <= int(self.max_row_tokens) <= MAX_ROW_TOKENS:
            raise TechnicalInvalid(f"max_row_tokens must be in [1,{MAX_ROW_TOKENS}]")
        if not 1 <= int(self.max_rows) <= MAX_ROWS:
            raise TechnicalInvalid(f"max_rows must be in [1,{MAX_ROWS}]")
        if self.max_new_tokens is not None and int(self.max_new_tokens) <= 0:
            raise TechnicalInvalid("max_new_tokens must be positive or None")
        latest = _token_ids(self.latest_history_row_token_ids, label="latest_history_row_token_ids")
        object.__setattr__(self, "latest_history_row_token_ids", latest)
        if self.history_prefix_width is not None and int(self.history_prefix_width) < 0:
            raise TechnicalInvalid("history_prefix_width must be non-negative")
        forbidden = {"past_key_values", "key_value_cache", "cache_position"}
        present = sorted(key for key in forbidden if self.model_inputs.get(key) is not None)
        if present:
            raise TechnicalInvalid(f"event context contains forbidden cache inputs: {present}")

    @property
    def prefix_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.exact_history_token_ids

    @property
    def prefix_sha256(self) -> str:
        return hash_token_ids(self.prefix_token_ids)

    @property
    def effective_max_new_tokens(self) -> int:
        return int(self.max_new_tokens or (self.max_row_tokens * self.max_rows))

    def receipt(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "admission_mode": "pre_opener_natural",
            "prompt_token_ids": list(self.prompt_token_ids),
            "exact_history_token_ids": list(self.exact_history_token_ids),
            "prefix_token_ids": list(self.prefix_token_ids),
            "prefix_token_ids_sha256": self.prefix_sha256,
            "prefix_ends_with_opener": self.prefix_token_ids[-1] == self.row_contract.opener_token_id,
            "synthetic_opener_injections": 0,
            "row_contract": self.row_contract.receipt(),
            "max_row_tokens": int(self.max_row_tokens),
            "max_rows": int(self.max_rows),
            "max_new_tokens": self.effective_max_new_tokens,
        }


# Friendly alias used by callers that describe a row as an event boundary.
EventContext = NaturalEventContext
NaturalBoundaryEventContext = NaturalEventContext


@dataclass(frozen=True)
class ResidualRequest:
    """One N-arm request delivered to a residual actuator callback."""

    arm_id: Literal["N00", "N01", "N10", "N20"]
    positions: tuple[int, ...] = ()
    replacement: torch.Tensor | None = None
    persistent: bool = False
    operation: str = "native"

    def __post_init__(self) -> None:
        if self.arm_id not in {"N00", "N01", "N10", "N20"}:
            raise TechnicalInvalid(f"unknown residual arm {self.arm_id!r}")
        positions = tuple(int(value) for value in self.positions)
        if any(value < 0 for value in positions) or len(set(positions)) != len(positions):
            raise TechnicalInvalid("residual positions must be unique non-negative integers")
        object.__setattr__(self, "positions", positions)
        if self.arm_id == "N00" and (positions or self.replacement is not None):
            raise TechnicalInvalid("N00 cannot carry residual positions or a replacement")
        if self.arm_id != "N00" and not positions:
            raise TechnicalInvalid(f"{self.arm_id} requires explicit residual positions")
        if self.replacement is not None and not isinstance(self.replacement, torch.Tensor):
            raise TechnicalInvalid("residual replacement must be a tensor or None")


def build_residual_request(
    arm_id: Literal["N00", "N01", "N10", "N20"],
    *,
    positions: Sequence[int] = (),
    replacement: torch.Tensor | None = None,
    persistent: bool = False,
) -> ResidualRequest:
    operation = {
        "N00": "native",
        "N01": "terminal_noop_replay",
        "N10": "terminal_replacement",
        "N20": "whole_row_replacement",
    }[arm_id]
    return ResidualRequest(
        arm_id,
        tuple(int(value) for value in positions),
        replacement,
        bool(persistent),
        operation,
    )


def residual_request_from_history(
    arm_id: Literal["N00", "N01", "N10", "N20"],
    context: NaturalEventContext,
    *,
    replacement: torch.Tensor | None = None,
    persistent: bool = False,
) -> ResidualRequest:
    """Build N positions from the latest exact-history row when available."""

    if arm_id == "N00":
        return build_residual_request("N00")
    row = context.latest_history_row_token_ids
    if not row:
        raise TechnicalInvalid(f"{arm_id} requires latest_history_row_token_ids")
    prefix_width = context.history_prefix_width
    if prefix_width is None:
        prefix_width = len(context.prompt_token_ids) + len(context.exact_history_token_ids) - len(row)
    if prefix_width < 0:
        raise TechnicalInvalid("history row is longer than exact prefix")
    if arm_id in {"N01", "N10"}:
        matches = [
            index
            for index, token in enumerate(row)
            if token == context.row_contract.closure_token_id
        ]
        if len(matches) != 1:
            raise TechnicalInvalid("latest history row must have exactly one terminal carrier")
        positions = (prefix_width + matches[0],)
    else:
        positions = tuple(prefix_width + index for index in range(len(row)))
    return build_residual_request(
        arm_id,
        positions=positions,
        replacement=replacement,
        persistent=persistent,
    )


def _token_ids(values: Sequence[int] | torch.Tensor, *, label: str, allow_empty: bool = True) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TechnicalInvalid(f"{label} must be a sequence of integer token IDs")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TechnicalInvalid(f"{label} contains a non-negative integer token requirement")
        result.append(int(value))
    if not result and not allow_empty:
        raise TechnicalInvalid(f"{label} must not be empty")
    return tuple(result)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise TechnicalInvalid(f"value is not canonical finite JSON: {exc}") from exc


def hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def hash_token_ids(values: Sequence[int] | torch.Tensor) -> str:
    return hash_json(list(_token_ids(values, label="token_ids")))


def build_event_context(
    *,
    event_id: str,
    prompt_token_ids: Sequence[int],
    exact_history_token_ids: Sequence[int] = (),
    row_contract: NativeRowContract | Mapping[str, Any],
    model_inputs: Mapping[str, Any] | None = None,
    max_row_tokens: int = MAX_ROW_TOKENS,
    max_rows: int = MAX_ROWS,
    max_new_tokens: int | None = None,
    position_ids_builder: Callable[[int], Any] | None = None,
    latest_history_row_token_ids: Sequence[int] = (),
    history_prefix_width: int | None = None,
) -> NaturalEventContext:
    if not isinstance(row_contract, NativeRowContract):
        payload = dict(row_contract)
        row_contract = NativeRowContract(**payload)
    payload = dict(model_inputs or {})
    return NaturalEventContext(
        event_id=str(event_id),
        prompt_token_ids=tuple(prompt_token_ids),
        exact_history_token_ids=tuple(exact_history_token_ids),
        row_contract=row_contract,
        model_inputs=payload,
        max_row_tokens=int(max_row_tokens),
        max_rows=int(max_rows),
        max_new_tokens=None if max_new_tokens is None else int(max_new_tokens),
        position_ids_builder=position_ids_builder,
        latest_history_row_token_ids=tuple(latest_history_row_token_ids),
        history_prefix_width=history_prefix_width,
    )


build_natural_event_context = build_event_context


def validate_admission_receipt(
    receipt: Mapping[str, Any],
    *,
    initial_prefix_token_ids: Sequence[int],
    opener_token_id: int,
    generated_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Validate the admission-mode/opener cross-field contract fail-closed."""

    if not isinstance(receipt, Mapping):
        raise TechnicalInvalid("admission receipt must be a mapping")
    mode = receipt.get("admission_mode")
    if mode not in {"pre_opener_natural", "post_opener_seeded"}:
        raise TechnicalInvalid("admission_mode must be explicit pre_opener_natural or post_opener_seeded")
    prefix = _token_ids(initial_prefix_token_ids, label="initial_prefix_token_ids", allow_empty=False)
    generated = _token_ids(generated_token_ids, label="generated_token_ids")
    opener_generated = receipt.get("opener_generated_by_model")
    if type(opener_generated) is not bool:
        raise TechnicalInvalid("opener_generated_by_model must be an actual boolean")
    if mode == "pre_opener_natural":
        if prefix[-1] == int(opener_token_id):
            raise TechnicalInvalid("pre_opener_natural prefix ends with the opener")
        if receipt.get("synthetic_opener_injections", 0) != 0:
            raise TechnicalInvalid("pre_opener_natural cannot inject an opener")
        if receipt.get("opener_injected", False) is not False:
            raise TechnicalInvalid("pre_opener_natural opener_injected must be false")
        expected = bool(generated and generated[0] == int(opener_token_id))
        if opener_generated is not expected:
            raise TechnicalInvalid("opener_generated_by_model disagrees with the first generated token")
    else:
        if receipt.get("seed_provenance") not in {"caller_supplied_opener", "post_opener_seeded"}:
            raise TechnicalInvalid("post_opener_seeded requires caller opener seed provenance")
        if receipt.get("supplied_opener_token_id") != int(opener_token_id):
            raise TechnicalInvalid("post_opener_seeded supplied opener differs from contract")
        if opener_generated is not False:
            raise TechnicalInvalid("post_opener_seeded opener_generated_by_model must be false")
    return {
        "admission_mode": mode,
        "opener_generated_by_model": opener_generated,
        "prefix_ends_with_opener": prefix[-1] == int(opener_token_id),
        "synthetic_opener_injections": int(receipt.get("synthetic_opener_injections", 0)),
    }


validate_natural_admission = validate_admission_receipt


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TechnicalInvalid(f"{label} must be finite numeric")
    number = float(value)
    if not torch.isfinite(torch.tensor(number)):
        raise TechnicalInvalid(f"{label} must be finite numeric")
    return number


def _extract_logits(output: Any) -> torch.Tensor:
    raw: Any
    if isinstance(output, torch.Tensor):
        raw = output
    elif hasattr(output, "logits"):
        raw = output.logits
    elif isinstance(output, Mapping) and "logits" in output:
        raw = output["logits"]
    else:
        raise TechnicalInvalid("model output must expose logits")
    if not isinstance(raw, torch.Tensor):
        raise TechnicalInvalid("model logits must be a torch.Tensor")
    if raw.ndim == 3:
        if raw.shape[0] != 1 or raw.shape[1] <= 0:
            raise TechnicalInvalid("model logits must have shape [1,sequence,vocab]")
        logits = raw[0, -1]
    elif raw.ndim == 2:
        if raw.shape[0] == 1:
            logits = raw[0]
        else:
            logits = raw[-1]
    elif raw.ndim == 1:
        logits = raw
    else:
        raise TechnicalInvalid("model logits must have rank 1, 2, or 3")
    if logits.numel() <= 0 or not bool(torch.isfinite(logits).all().item()):
        raise TechnicalInvalid("model logits are empty or non-finite")
    return logits.detach().float().reshape(-1)


class FullLogitCapture:
    """Keep full-vocabulary scalar logits in memory for one no-op comparator.

    The vectors are never serialized into scientific artifacts.  They exist
    only long enough to compare a matching no-op arm at every scalar step.
    """

    def __init__(self) -> None:
        self.vectors: list[torch.Tensor] = []

    def __call__(self, *, logits: torch.Tensor, **_metadata: Any) -> None:
        vector = logits.detach().to(device="cpu", dtype=torch.float32).clone()
        if vector.ndim != 1 or not bool(torch.isfinite(vector).all().item()):
            raise TechnicalInvalid("full-logit capture requires one finite vocabulary vector")
        self.vectors.append(vector)


class FullLogitParityComparator:
    """Streaming full-vocabulary comparison against one captured trajectory."""

    def __init__(self, reference: Sequence[torch.Tensor], *, tolerance: float = NOOP_TOLERANCE) -> None:
        self.reference = tuple(reference)
        self.tolerance = float(tolerance)
        self.observed_count = 0
        self.max_abs_delta = 0.0
        self.shape_mismatch = False

    def __call__(self, *, logits: torch.Tensor, **_metadata: Any) -> None:
        if self.observed_count >= len(self.reference):
            raise TechnicalInvalid("no-op arm produced more scalar logit vectors than native")
        candidate = logits.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        reference = self.reference[self.observed_count]
        if candidate.shape != reference.shape:
            self.shape_mismatch = True
            raise TechnicalInvalid("no-op full-logit vocabulary shape differs from native")
        delta = torch.max(torch.abs(candidate - reference))
        if not bool(torch.isfinite(delta).item()):
            raise TechnicalInvalid("no-op full-logit delta is non-finite")
        self.max_abs_delta = max(self.max_abs_delta, float(delta.item()))
        self.observed_count += 1

    def receipt(self) -> dict[str, Any]:
        complete = self.observed_count == len(self.reference) and not self.shape_mismatch
        return {
            "status": "measured" if complete else "invalid",
            "comparison_scope": "every_full_vocabulary_logit_per_scalar_step",
            "reference_step_count": len(self.reference),
            "candidate_step_count": self.observed_count,
            "shape_mismatch": self.shape_mismatch,
            "max_abs_delta": float(self.max_abs_delta),
            "tolerance": self.tolerance,
            "passed": bool(complete and self.max_abs_delta <= self.tolerance),
        }


def _distribution(logits: torch.Tensor, token_id: int) -> dict[str, Any]:
    if token_id < 0 or token_id >= logits.numel():
        raise TechnicalInvalid(f"token ID {token_id} is outside model vocabulary")
    log_probs = torch.log_softmax(logits, dim=-1)
    selected_logit = float(logits[token_id].item())
    selected_lp = float(log_probs[token_id].item())
    rank = 1 + int((logits > logits[token_id]).sum().item())
    best_id = int(torch.argmax(logits).item())
    return {
        "token_id": int(token_id),
        "logit": selected_logit,
        "log_probability": selected_lp,
        "rank": rank,
        "best_token_id": best_id,
        "best_logit": float(logits[best_id].item()),
        "best_log_probability": float(log_probs[best_id].item()),
        "best_rank": 1,
    }


def _scalar_score(
    logits: torch.Tensor,
    *,
    selected_token_id: int,
    opener_token_id: int,
    prefix_token_ids: Sequence[int],
    step: int,
    row_index: int,
    phase: str,
) -> dict[str, Any]:
    selected = _distribution(logits, selected_token_id)
    score = {
        "step": int(step),
        "row_index": int(row_index),
        "phase": str(phase),
        "prefix_token_count": len(prefix_token_ids),
        "prefix_token_ids_sha256": hash_token_ids(prefix_token_ids),
        "selected_token_id": int(selected_token_id),
        "selected_token_logit": selected["logit"],
        "selected_token_log_probability": selected["log_probability"],
        "selected_token_rank": selected["rank"],
        "best_token_id": selected["best_token_id"],
        "best_token_logit": selected["best_logit"],
        "best_token_log_probability": selected["best_log_probability"],
        "best_token_rank": selected["best_rank"],
        "selected": selected,
        "opener": _distribution(logits, int(opener_token_id)),
        "best": {
            "token_id": selected["best_token_id"],
            "logit": selected["best_logit"],
            "log_probability": selected["best_log_probability"],
            "rank": 1,
        },
    }
    return score


def _attention_payload(
    callback: AttentionMaskActuator | Callable[..., Any] | None,
    context: NaturalEventContext,
    *,
    input_ids: torch.Tensor,
    step: int,
    row_index: int,
    arm_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if callback is None:
        return {}, {"applied": False, "receipt": None}
    # ``natural_boundary_attention_actuators`` exposes a keyword-only callback
    # (sequence_length/query_position), while the runner's local protocol uses
    # a context-first callback.  Detect the former by signature rather than
    # catching arbitrary callback TypeErrors, so actuator bugs remain visible.
    try:
        signature = inspect.signature(callback)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
    except (TypeError, ValueError):
        positional = [object()]
    if not positional:
        result = callback(
            sequence_length=int(input_ids.shape[-1]),
            query_position=int(input_ids.shape[-1]) - 1,
            device=input_ids.device,
            step=step,
            row_index=row_index,
            arm_id=arm_id,
            context=context,
        )
    else:
        result = callback(context, input_ids=input_ids, step=step, row_index=row_index, arm_id=arm_id)
    if result is None:
        return {}, {"applied": False, "receipt": None}
    if isinstance(result, torch.Tensor):
        return {"attention_mask": result}, {"applied": True, "receipt": None}
    if not isinstance(result, Mapping):
        raise TechnicalInvalid("attention-mask actuator must return tensor, mapping, or None")
    payload = dict(result.get("model_kwargs", {})) if isinstance(result.get("model_kwargs"), Mapping) else {}
    metadata: dict[str, Any] = {}
    non_model_keys = {
        "model_kwargs",
        "receipt",
        "mask_receipt",
        "score_bias",
        "score_bias_callback",
        "actuator_id",
        "protocol",
    }
    for key, value in result.items():
        if key not in non_model_keys:
            payload[key] = value
        elif key not in {
            "model_kwargs",
            "receipt",
            "mask_receipt",
            # These are executable Python handles retained only on the
            # in-memory actuator boundary.  Persist their JSON-safe actuator
            # identity/protocol and complete receipt below, never the objects.
            "score_bias",
            "score_bias_callback",
        }:
            metadata[key] = value
    if "custom_attention_mask" in payload:
        raise TechnicalInvalid("attention-mask actuator cannot use ignored custom_attention_mask")
    callback_receipt = result.get("receipt", result.get("mask_receipt"))
    if metadata:
        if isinstance(callback_receipt, Mapping):
            callback_receipt = {**dict(callback_receipt), "callback_metadata": metadata}
        else:
            callback_receipt = {"callback_metadata": metadata}
    return payload, {"applied": True, "receipt": callback_receipt}


def _position_ids(context: NaturalEventContext, length: int) -> Any:
    supplied = context.model_inputs.get("position_ids")
    if supplied is None:
        return None
    if context.position_ids_builder is not None:
        return context.position_ids_builder(int(length))
    if isinstance(supplied, torch.Tensor):
        if supplied.shape[-1] != length:
            raise TechnicalInvalid("exact natural scalar recompute needs a position_ids_builder as prefix grows")
    return supplied


def _residual_context(
    actuator: ResidualActuator | Callable[..., Any] | Any | None,
    model: Any,
    context: NaturalEventContext,
    request: ResidualRequest,
    *,
    input_ids: torch.Tensor,
    step: int,
    row_index: int,
) -> tuple[AbstractContextManager[Any], dict[str, Any]]:
    if request.arm_id == "N00":
        return nullcontext(), {
            "arm_id": "N00",
            "operation": "native",
            "applied": False,
            "positions": [],
            "persistent": False,
        }
    if actuator is None:
        raise TechnicalInvalid(f"{request.arm_id} requires a residual actuator callback")
    if hasattr(actuator, "install") and callable(getattr(actuator, "install")):
        cm = actuator.install(
            model,
            context=context,
            request=request,
            input_ids=input_ids,
            step=step,
            row_index=row_index,
        )
    else:
        cm = actuator(
            model,
            context=context,
            request=request,
            input_ids=input_ids,
            step=step,
            row_index=row_index,
        )
    if cm is None:
        cm = nullcontext()
    if not hasattr(cm, "__enter__") or not hasattr(cm, "__exit__"):
        raise TechnicalInvalid("residual actuator must return a context manager or None")
    return cm, {
        "arm_id": request.arm_id,
        "operation": request.operation,
        "applied": True,
        "positions": list(request.positions),
        "persistent": bool(request.persistent),
    }


def _forward_scalar(
    model: Any,
    context: NaturalEventContext,
    prefix_token_ids: Sequence[int],
    *,
    step: int,
    row_index: int,
    phase: str,
    residual_request: ResidualRequest,
    residual_actuator: ResidualActuator | Callable[..., Any] | Any | None,
    attention_mask_actuator: AttentionMaskActuator | Callable[..., Any] | None,
    logit_observer: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ids = _token_ids(prefix_token_ids, label="scalar_prefix_token_ids", allow_empty=False)
    input_ids = torch.tensor([list(ids)], dtype=torch.long)
    payload = dict(context.model_inputs)
    payload.pop("input_ids", None)
    for key in ("past_key_values", "key_value_cache", "cache_position"):
        if payload.get(key) is not None:
            raise TechnicalInvalid(f"scalar natural recompute received forbidden cache input {key}")
        payload.pop(key, None)
    payload["input_ids"] = input_ids
    payload["use_cache"] = False
    position_ids = _position_ids(context, len(ids))
    if position_ids is not None:
        payload["position_ids"] = position_ids
    mask_payload, mask_receipt = _attention_payload(
        attention_mask_actuator,
        context,
        input_ids=input_ids,
        step=step,
        row_index=row_index,
        arm_id=residual_request.arm_id,
    )
    payload.update(mask_payload)
    if payload.get("custom_attention_mask") is not None:
        raise TechnicalInvalid("natural scalar recompute rejects custom_attention_mask")
    cm, residual_receipt = _residual_context(
        residual_actuator,
        model,
        context,
        residual_request,
        input_ids=input_ids,
        step=step,
        row_index=row_index,
    )
    with cm:
        output = _invoke_model_schema(model, payload)
    context_receipt = getattr(cm, "receipt", None)
    if callable(context_receipt):
        observed_receipt = context_receipt()
        if not isinstance(observed_receipt, Mapping):
            raise TechnicalInvalid("residual actuator receipt must be a mapping")
        residual_receipt["actuation_receipt"] = dict(observed_receipt)
        residual_receipt["hook_applied_count"] = observed_receipt.get("hook_applied_count")
        residual_receipt["hook_removed"] = observed_receipt.get("hook_removed")
    logits = _extract_logits(output)
    if logit_observer is not None:
        logit_observer(
            logits=logits,
            step=int(step),
            row_index=int(row_index),
            phase=str(phase),
            prefix_token_ids=ids,
        )
    best_token_id = int(torch.argmax(logits).item())
    score = _scalar_score(
        logits,
        selected_token_id=best_token_id,
        opener_token_id=context.row_contract.opener_token_id,
        prefix_token_ids=ids,
        step=step,
        row_index=row_index,
        phase=phase,
    )
    receipt = {
        "step": int(step),
        "row_index": int(row_index),
        "phase": phase,
        "input_ids": list(ids),
        "input_ids_sha256": hash_token_ids(ids),
        "prefix_token_count": len(ids),
        "use_cache": False,
        "cache_forbidden_fields": [],
        "attention_mask": mask_receipt,
        "residual": residual_receipt,
    }
    return score, receipt


def score_natural_row_segments(
    row_token_ids: Sequence[int],
    selected_log_probabilities: Sequence[float],
    selected_ranks: Sequence[int] | None,
    *,
    contract: NativeRowContract,
) -> dict[str, Any]:
    """Score natural row segments with strict full-row alignment.

    Unlike the legacy helper, selected arrays include the model-generated
    opener at index zero.  Any mismatch is technical invalidity, not a
    ``not_measured`` scientific result.
    """

    row = _token_ids(row_token_ids, label="row_token_ids", allow_empty=False)
    logs = tuple(_finite_float(value, label="selected log probability") for value in selected_log_probabilities)
    if len(logs) != len(row):
        raise TechnicalInvalid(
            f"natural row score alignment mismatch: expected {len(row)} tokens, observed {len(logs)} log probabilities"
        )
    ranks = tuple(int(value) for value in selected_ranks) if selected_ranks is not None else ()
    if ranks and len(ranks) != len(row):
        raise TechnicalInvalid(
            f"natural row rank alignment mismatch: expected {len(row)} tokens, observed {len(ranks)} ranks"
        )
    if ranks and any(value <= 0 for value in ranks):
        raise TechnicalInvalid("selected token ranks must be positive")
    parsed = contract.parse_row(row)
    if parsed.get("status") != "complete":
        raise TechnicalInvalid(f"cannot score an incomplete natural row: {parsed.get('reason')}")
    indices = parsed["indices"]
    segments: dict[str, Any] = {}
    for name, raw_indices in indices.items():
        positions = tuple(int(index) for index in raw_indices)
        values = [logs[index] for index in positions]
        segment_ranks = [ranks[index] for index in positions] if ranks else None
        segments[name] = {
            "status": "measured",
            "token_indices": list(positions),
            "token_count": len(values),
            "sum_log_probability": float(sum(values)),
            "mean_log_probability": float(sum(values) / len(values)),
            "selected_token_log_probabilities": list(values),
            "selected_token_ranks": segment_ranks,
        }
    return {
        "status": "measured",
        "source": "natural_selected_token_logprobs",
        "teacher_forced": False,
        "opener_included": True,
        "row_token_count": len(row),
        "segments": segments,
        "parser": parsed,
    }


# Compatibility aliases for callers using the old helper's naming style.
score_natural_segments = score_natural_row_segments
_natural_segment_scores = score_natural_row_segments


def _score_receipt_aliases(score: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if score is None:
        return None
    return {
        "token_id": score.get("selected_token_id"),
        "logit": score.get("selected_token_logit"),
        "log_probability": score.get("selected_token_log_probability"),
        "rank": score.get("selected_token_rank"),
        "best_token_id": score.get("best_token_id"),
        "best_logit": score.get("best_token_logit"),
        "best_log_probability": score.get("best_token_log_probability"),
        "best_rank": score.get("best_token_rank"),
    }


def _row_receipt(
    *,
    context: NaturalEventContext,
    prefix_before_row: Sequence[int],
    row_index: int,
    row_token_ids: Sequence[int],
    scores: Sequence[Mapping[str, Any]],
    status: str,
    reason: str | None,
    over_continuation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = tuple(int(value) for value in row_token_ids)
    selected_logs = [float(item["selected_token_log_probability"]) for item in scores]
    selected_ranks = [int(item["selected_token_rank"]) for item in scores]
    first = scores[0] if scores else None
    opener = None
    if first is not None:
        # ``_scalar_score`` only contains the selected token and best token.
        # Reconstructing the opener score requires the raw logits, so the
        # release loop supplies this explicit field after scoring.
        opener = first.get("opener")
    terminal = None
    if scores and status in {"closure", "native_stop", "invalid", "max_budget"}:
        terminal = scores[-1]
    parsed = context.row_contract.parse_row(row) if row else {"status": "invalid", "reason": "empty_row"}
    segment_scores: dict[str, Any] | None = None
    if status == "closure" and parsed.get("status") == "complete":
        segment_scores = score_natural_row_segments(
            row,
            selected_logs,
            selected_ranks,
            contract=context.row_contract,
        )
    return {
        "row_index": int(row_index),
        "prefix_before_row_token_ids": list(prefix_before_row),
        "prefix_before_row_token_ids_sha256": hash_token_ids(prefix_before_row),
        "initial_prefix_last_token_id": int(prefix_before_row[-1]),
        "opener_token_id": int(context.row_contract.opener_token_id),
        "opener_injected": False,
        "token_ids": list(row),
        "token_ids_sha256": hash_token_ids(row),
        "status": status,
        "reason": reason,
        "stop_reason": status if status in {"closure", "native_stop", "invalid", "over_continuation", "max_budget"} else reason,
        "admission_mode": "pre_opener_natural",
        "opener_generated_by_model": bool(row and row[0] == context.row_contract.opener_token_id),
        "first_token_id": row[0] if row else None,
        "first_generated_token_id": row[0] if row else None,
        "first_token_logits": _score_receipt_aliases(first),
        "opener_logits": opener,
        "best_logits": _score_receipt_aliases(first),
        "terminal_logits": _score_receipt_aliases(terminal),
        "selected_token_log_probabilities": selected_logs,
        "selected_token_ranks": selected_ranks,
        "parser": parsed,
        "segment_scores": segment_scores,
        "over_continuation": dict(over_continuation) if over_continuation is not None else None,
        "row_token_count": len(row),
        "row_token_cap": int(context.max_row_tokens),
    }


def _trajectory_steps(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for row in result.get("rows", []):
        for score in row.get("scores", []):
            steps.append(
                {
                    "token_id": score.get("selected_token_id"),
                    "logit": score.get("selected_token_logit"),
                    "log_probability": score.get("selected_token_log_probability"),
                    "rank": score.get("selected_token_rank"),
                }
            )
    return steps


def trajectory_parity_receipt(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    tolerance: float = NOOP_TOLERANCE,
    full_logit_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare complete N01 and N00 trajectories with a per-step tolerance."""

    ref_steps = _trajectory_steps(reference)
    cand_steps = _trajectory_steps(candidate)
    deltas: list[float] = []
    for left, right in zip(ref_steps, cand_steps):
        for key in ("logit", "log_probability"):
            try:
                deltas.append(abs(float(left[key]) - float(right[key])))
            except (TypeError, ValueError):
                deltas.append(float("inf"))
    max_delta = max(deltas, default=0.0)
    token_ids_equal = [item.get("token_id") for item in ref_steps] == [item.get("token_id") for item in cand_steps]
    complete_trajectory = (
        token_ids_equal
        and len(ref_steps) == len(cand_steps)
        and reference.get("terminal_reason") == candidate.get("terminal_reason")
        and len(reference.get("rows", [])) == len(candidate.get("rows", []))
    )
    full = (
        dict(full_logit_receipt)
        if isinstance(full_logit_receipt, Mapping)
        else {
            "status": "not_measured",
            "comparison_scope": "every_full_vocabulary_logit_per_scalar_step",
            "passed": False,
            "reason": "full-logit observer was not supplied",
        }
    )
    return {
        "status": "measured",
        "complete_trajectory": bool(complete_trajectory),
        "token_ids_equal": bool(token_ids_equal),
        "reference_step_count": len(ref_steps),
        "candidate_step_count": len(cand_steps),
        "per_step_max_abs_delta": float(max_delta),
        "tolerance": float(tolerance),
        "full_logit_parity": full,
        "full_logit_max_abs_delta": full.get("max_abs_delta"),
        "passed": bool(
            complete_trajectory
            and max_delta <= float(tolerance)
            and full.get("passed") is True
        ),
    }


def _invoke_model_schema(model: Any, payload: Mapping[str, Any]) -> Any:
    """Call a fake/production model without silently dropping any field."""

    try:
        return model(**payload)
    except TypeError as exc:
        try:
            signature = inspect.signature(model)
        except (TypeError, ValueError):
            raise TechnicalInvalid(f"model call rejected explicit no-cache payload: {exc}") from exc
        accepted = sorted(signature.parameters)
        raise TechnicalInvalid(
            "model rejected the exact scalar payload; no field may be filtered "
            f"(accepted signature fields={accepted}): {exc}"
        ) from exc


def release_natural_event(
    model: Any,
    context: NaturalEventContext,
    *,
    residual_arm: Literal["N00", "N01", "N10", "N20"] = "N00",
    residual_request: ResidualRequest | None = None,
    residual_actuator: ResidualActuator | Callable[..., Any] | Any | None = None,
    attention_mask_actuator: AttentionMaskActuator | Callable[..., Any] | None = None,
    logit_observer: Callable[..., Any] | None = None,
    parity_reference: Mapping[str, Any] | None = None,
    full_logit_parity_receipt: Mapping[str, Any] | None = None,
    parity_tolerance: float = NOOP_TOLERANCE,
) -> dict[str, Any]:
    """Release up to three rows from the exact pre-opener natural boundary."""

    if not isinstance(context, NaturalEventContext):
        raise TechnicalInvalid("release requires a NaturalEventContext")
    if residual_request is None:
        residual_request = residual_request_from_history(residual_arm, context) if residual_arm != "N00" else build_residual_request("N00")
    elif residual_request.arm_id != residual_arm:
        raise TechnicalInvalid("residual_request.arm_id disagrees with residual_arm")
    if residual_arm not in {"N00", "N01", "N10", "N20"}:
        raise TechnicalInvalid(f"unknown residual arm {residual_arm!r}")
    max_rows = min(int(context.max_rows), MAX_ROWS)
    max_row_tokens = min(int(context.max_row_tokens), MAX_ROW_TOKENS)
    total_budget = min(int(context.effective_max_new_tokens), max_rows * max_row_tokens)
    generated: list[int] = []
    rows: list[dict[str, Any]] = []
    scalar_receipts: list[dict[str, Any]] = []
    scalar_scores: list[dict[str, Any]] = []
    pending: tuple[dict[str, Any], bool] | None = None
    terminal_reason = "closure"
    step = 0

    for row_index in range(max_rows):
        if len(generated) >= total_budget:
            terminal_reason = "max_budget"
            break
        row_prefix = tuple(context.prefix_token_ids + tuple(generated))
        row_tokens: list[int] = []
        row_scores: list[dict[str, Any]] = []
        row_status = "invalid"
        row_reason: str | None = None
        over_continuation: dict[str, Any] | None = None
        while True:
            if len(generated) >= total_budget:
                row_status, row_reason, terminal_reason = "max_budget", "max_new_tokens", "max_budget"
                break
            if pending is not None:
                score, already_appended = pending
                pending = None
            else:
                score, receipt = _forward_scalar(
                    model,
                    context,
                    tuple(context.prefix_token_ids + tuple(generated)),
                    step=step,
                    row_index=row_index,
                    phase="row_entry" if not row_tokens else "row_body",
                    residual_request=residual_request,
                    residual_actuator=residual_actuator,
                    attention_mask_actuator=attention_mask_actuator,
                    logit_observer=logit_observer,
                )
                scalar_receipts.append(receipt)
                scalar_scores.append(score)
                already_appended = False
                step += 1
            token_id = int(score["best_token_id"])
            score = dict(score)
            score["selected_token_id"] = token_id
            score["selected_token_logit"] = score["best_token_logit"]
            score["selected_token_log_probability"] = score["best_token_log_probability"]
            score["selected_token_rank"] = 1
            if not row_tokens:
                score["phase"] = "row_entry"
                if token_id in context.row_contract.effective_stop_token_ids:
                    row_tokens.append(token_id)
                    if not already_appended:
                        generated.append(token_id)
                    row_scores.append(score)
                    row_status, row_reason, terminal_reason = "native_stop", "native_stop", "native_stop"
                    break
                if token_id != context.row_contract.opener_token_id:
                    row_tokens.append(token_id)
                    if not already_appended:
                        generated.append(token_id)
                    row_scores.append(score)
                    row_status, row_reason, terminal_reason = "invalid", "first_token_not_opener", "invalid"
                    break
            row_tokens.append(token_id)
            if not already_appended:
                generated.append(token_id)
            row_scores.append(score)
            if token_id in context.row_contract.effective_stop_token_ids:
                row_status, row_reason, terminal_reason = "native_stop", "native_stop", "native_stop"
                break
            if token_id == context.row_contract.closure_token_id:
                parsed = context.row_contract.parse_row(row_tokens)
                if parsed.get("status") == "complete":
                    row_status, row_reason, terminal_reason = "closure", None, "closure"
                else:
                    reason = str(parsed.get("reason") or "invalid_row")
                    row_status, row_reason, terminal_reason = "invalid", reason, "invalid"
                break
            if len(row_tokens) >= max_row_tokens:
                row_status, row_reason, terminal_reason = "max_budget", "max_row_tokens", "max_budget"
                break
            if len(generated) >= total_budget:
                row_status, row_reason, terminal_reason = "max_budget", "max_new_tokens", "max_budget"
                break

        # Add opener metrics after the first scalar call.  The scalar receipt
        # may optionally carry a complete distribution from a production
        # adapter; otherwise the selected/best aliases remain available.
        row = _row_receipt(
            context=context,
            prefix_before_row=row_prefix,
            row_index=row_index,
            row_token_ids=row_tokens,
            scores=row_scores,
            status=row_status,
            reason=row_reason,
            over_continuation=over_continuation,
        )
        row["scores"] = [dict(score) for score in row_scores]
        if row_scores:
            row["first_token_logits"] = _score_receipt_aliases(row_scores[0])
            row["best_logits"] = _score_receipt_aliases(row_scores[0])
            if row_status in {"closure", "native_stop", "invalid", "max_budget"}:
                row["terminal_logits"] = _score_receipt_aliases(row_scores[-1])
        rows.append(row)
        if row_status != "closure":
            break
        if row_index + 1 >= max_rows:
            terminal_reason = "closure"
            break
        # One natural boundary lookahead decides whether the next row starts
        # with a model-generated opener.  No opener is ever appended here.
        if len(generated) >= total_budget:
            terminal_reason = "max_budget"
            break
        look_score, look_receipt = _forward_scalar(
            model,
            context,
            tuple(context.prefix_token_ids + tuple(generated)),
            step=step,
            row_index=row_index + 1,
            phase="between_rows",
            residual_request=residual_request,
            residual_actuator=residual_actuator,
            attention_mask_actuator=attention_mask_actuator,
            logit_observer=logit_observer,
        )
        scalar_receipts.append(look_receipt)
        scalar_scores.append(look_score)
        step += 1
        look_token = int(look_score["best_token_id"])
        look_score = dict(look_score)
        look_score["selected_token_id"] = look_token
        look_score["selected_token_logit"] = look_score["best_token_logit"]
        look_score["selected_token_log_probability"] = look_score["best_token_log_probability"]
        look_score["selected_token_rank"] = 1
        if look_token in context.row_contract.effective_stop_token_ids:
            generated.append(look_token)
            terminal_reason = "native_stop"
            break
        if look_token == context.row_contract.opener_token_id:
            # Keep the opener out of the next row's *prefix-before-row*
            # receipt until that row consumes it.  This preserves the exact
            # natural boundary while still reusing the already-issued scalar
            # logits for the model-generated opener.
            pending = (look_score, False)
            continue
        # A token after a complete row that is neither STOP nor a natural
        # opener is an over-continuation, not an invalid first admission.
        generated.append(look_token)
        rows[-1]["status"] = "over_continuation"
        rows[-1]["reason"] = "token_after_complete_row"
        rows[-1]["stop_reason"] = "over_continuation"
        rows[-1]["over_continuation"] = dict(look_score)
        terminal_reason = "over_continuation"
        break

    # Every row receipt is cross-field validated after the loop, including
    # native STOP/invalid rows whose first token was not an opener.
    for row in rows:
        validate_admission_receipt(
            {
                "admission_mode": row["admission_mode"],
                "opener_generated_by_model": row["opener_generated_by_model"],
                "synthetic_opener_injections": 0,
                "opener_injected": False,
            },
            initial_prefix_token_ids=row["prefix_before_row_token_ids"],
            opener_token_id=context.row_contract.opener_token_id,
            generated_token_ids=row["token_ids"],
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "event_id": context.event_id,
        "admission_mode": "pre_opener_natural",
        "initial_prefix_last_token_id": int(context.prefix_token_ids[-1]),
        "opener_token_id": int(context.row_contract.opener_token_id),
        "opener_generated_by_model": bool(rows and rows[0]["opener_generated_by_model"]),
        "opener_injected": False,
        "synthetic_opener_injections": 0,
        "first_generated_token_id": generated[0] if generated else None,
        "prefix": context.receipt(),
        "rows": rows,
        "generated_token_ids": list(generated),
        "generated_token_ids_sha256": hash_token_ids(generated),
        "terminal_reason": terminal_reason,
        "stop_reason": terminal_reason,
        "row_limit_reached": bool(rows and len(rows) >= max_rows and terminal_reason == "closure"),
        "scalar_forward_count": len(scalar_receipts),
        "scalar_receipts": scalar_receipts,
        "no_cache_scalar_recompute": all(item.get("use_cache") is False for item in scalar_receipts),
        "residual_arm": residual_arm,
        "residual_request": {
            "arm_id": residual_request.arm_id,
            "operation": residual_request.operation,
            "positions": list(residual_request.positions),
            "persistent": residual_request.persistent,
        },
        "attention_mask_actuator": attention_mask_actuator is not None,
        "max_row_tokens": max_row_tokens,
        "max_rows": max_rows,
    }
    if rows:
        result["first_token_logits"] = rows[0].get("first_token_logits")
        result["opener_logits"] = rows[0].get("opener_logits")
        result["best_logits"] = rows[0].get("best_logits")
        result["terminal_logits"] = rows[-1].get("terminal_logits")
    if residual_arm == "N01" and parity_reference is not None:
        result["n01_parity"] = trajectory_parity_receipt(
            parity_reference,
            result,
            tolerance=parity_tolerance,
            full_logit_receipt=full_logit_parity_receipt,
        )
    return result


run_natural_boundary_release = release_natural_event
release_pre_opener_natural = release_natural_event


def run_natural_residual_flow(
    model: Any,
    context: NaturalEventContext,
    *,
    residual_actuator: ResidualActuator | Callable[..., Any] | Any,
    attention_mask_actuator: AttentionMaskActuator | Callable[..., Any] | None = None,
    residual_requests: Mapping[str, ResidualRequest] | None = None,
    parity_tolerance: float = NOOP_TOLERANCE,
) -> dict[str, Any]:
    """Run N00/N01/N10/N20 and attach the complete N01 parity receipt."""

    if residual_actuator is None:
        raise TechnicalInvalid("N01/N10/N20 flow requires a residual actuator")
    outputs: dict[str, Any] = {}
    native_logits = FullLogitCapture()
    native = release_natural_event(
        model,
        context,
        residual_arm="N00",
        residual_request=(residual_requests or {}).get("N00"),
        attention_mask_actuator=attention_mask_actuator,
        logit_observer=native_logits,
    )
    outputs["N00"] = native
    for arm in ("N01", "N10", "N20"):
        request = (residual_requests or {}).get(arm)
        if request is None:
            request = residual_request_from_history(arm, context)
        full_comparator = (
            FullLogitParityComparator(native_logits.vectors, tolerance=parity_tolerance)
            if arm == "N01"
            else None
        )
        outputs[arm] = release_natural_event(
            model,
            context,
            residual_arm=arm,  # type: ignore[arg-type]
            residual_request=request,
            residual_actuator=residual_actuator,
            attention_mask_actuator=attention_mask_actuator,
            logit_observer=full_comparator,
            parity_tolerance=parity_tolerance,
        )
        if arm == "N01" and full_comparator is not None:
            outputs[arm]["n01_parity"] = trajectory_parity_receipt(
                native,
                outputs[arm],
                tolerance=parity_tolerance,
                full_logit_receipt=full_comparator.receipt(),
            )
    parity = outputs["N01"].get("n01_parity")
    return {
        "schema_version": f"{SCHEMA_VERSION}.residual_flow.v1",
        "event_id": context.event_id,
        "arms": outputs,
        "n01_parity": parity,
    }


def persist_verbatim_failure_log(
    run_root: str | Path,
    stderr: str | bytes,
    *,
    filename: str = "failure.stderr",
) -> dict[str, Any]:
    """Persist exact stderr bytes and return a hash-backed failure receipt."""

    root = Path(run_root).expanduser()
    if root.exists() and not root.is_dir():
        raise TechnicalInvalid(f"failure log root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not filename or Path(filename).name != filename:
        raise TechnicalInvalid("failure log filename must be one local filename")
    payload = stderr.encode("utf-8") if isinstance(stderr, str) else bytes(stderr)
    path = root / filename
    path.write_bytes(payload)
    observed = path.read_bytes()
    if observed != payload:
        raise TechnicalInvalid("persisted failure log is not byte-for-byte identical")
    digest = hashlib.sha256(observed).hexdigest()
    return {
        "status": "persisted",
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": len(observed),
        "verbatim_stderr": observed.decode("utf-8", errors="surrogateescape"),
    }


persist_failure_log = persist_verbatim_failure_log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-row-tokens", type=int, default=MAX_ROW_TOKENS)
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_row_tokens < 1 or args.max_row_tokens > MAX_ROW_TOKENS:
        raise SystemExit(f"--max-row-tokens must be in [1,{MAX_ROW_TOKENS}]")
    if args.max_rows < 1 or args.max_rows > MAX_ROWS:
        raise SystemExit(f"--max-rows must be in [1,{MAX_ROWS}]")
    if not args.dry_run:
        raise SystemExit("the runner requires an injected model/context; use the Python API or --dry-run")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "event_id": args.event_id,
        "max_row_tokens": args.max_row_tokens,
        "max_rows": args.max_rows,
        "status": "dry_run_schema_validated",
    }
    if args.output_root is None:
        print(json.dumps(payload, sort_keys=True))
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "schema.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

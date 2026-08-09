#!/usr/bin/env python3
"""Pure attention actuators for the natural-boundary routing unit.

The natural-boundary runner owns model loading and token generation.  This
module owns only the small, auditable tensor operations used at a boundary:

* ``K00``/``K01``/``K10``/``K11``/``K12``/``K13`` construct causal 4-D masks;
* ``K14T``/``K14B`` construct a fixed ``+2.0`` additive attention mask (plus
  a compatibility score-delta view); and
* ``H00``/``H10``/``H20`` construct the input-level history masks and attest
  that every decoder layer consumed the same tensor.

No checkpoint, tokenizer, image feature, or generation policy is imported
here.  The builders are deliberately usable with a fake model in CPU tests.
The score-bias callback is an explicit *pre-softmax* protocol: a runner gives
it scaled QK scores and receives scores with the declared entries added.  This
keeps the intervention at the intended seam instead of relying on an output
hook that cannot prove where a change was made.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import inspect
import json
from numbers import Integral
from typing import Any, Literal, Protocol, runtime_checkable

import torch


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
CROSSOVER_UNIT_ID = "2026-08-07-s-k10-h20-natural-crossover"
SCHEMA_VERSION = "natural_boundary_attention_actuators.v1"
MASK_RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.mask_receipt.v1"
BIAS_RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.bias_receipt.v1"
CONSUMPTION_RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.layer_consumption.v1"
BLOCK23_MASS_RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.block23_mass.v1"
BLOCK23_SDPA_ATTESTOR_SCHEMA_VERSION = f"{SCHEMA_VERSION}.block23_sdpa_attestor.v1"
COMPOSED_MASK_RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.composition.v1"
FIXED_BIAS = 2.0
NOOP_TOLERANCE = 1e-4

K_ARM_IDS = ("K00", "K01", "K10", "K11", "K12", "K13", "K14T", "K14B")
H_ARM_IDS = ("H00", "H10", "H20")


class ActuatorError(ValueError):
    """Raised when an actuator contract would be ambiguous or out of scope."""


class TechnicalInvalid(ActuatorError):
    """Compatibility name used by the existing research runners."""


CONSTRUCTION_CONSUMPTION_PLACEHOLDER_BASE_KEYS = frozenset(
    {
        "schema_version",
        "required",
        "status",
        "exact_same_tensor_all_layers_required",
    }
)


def construction_consumption_placeholder_kind(
    value: Any, *, expected_layer_count: int = 28
) -> str | None:
    """Name the exact pre-forward consumption placeholder ``value`` is, if any.

    Construction can only declare that every layer must later consume the same
    tensor; it never observes a forward.  This module emits exactly three such
    placeholders: the ``base`` one every built actuator carries, the
    ``declared_layer_count`` variant the additive-dose arms carry, and the
    ``composed`` one ``compose_k10_h20`` puts on a C11 receipt, which also pins
    the composed sequence length and an explicit ``passed: false``.

    Consumers use this to tell a legitimate unattested construction receipt --
    for example the K10/H20 child receipts a composed C11 mask keeps verbatim --
    apart from a failed or malformed runtime attestation.  Anything else stays
    unrecognized so a consumer fails closed instead of reading a placeholder as
    consumption evidence.
    """

    if not isinstance(value, Mapping):
        return None
    if (
        value.get("schema_version") != CONSUMPTION_RECEIPT_SCHEMA_VERSION
        or value.get("required") is not True
        or value.get("status") != "unattested"
        or value.get("exact_same_tensor_all_layers_required") is not True
    ):
        return None
    keys = set(value)
    if keys == set(CONSTRUCTION_CONSUMPTION_PLACEHOLDER_BASE_KEYS):
        return "base"
    declared_layers = value.get("declared_layer_count")
    if isinstance(declared_layers, bool) or not isinstance(declared_layers, int):
        return None
    if declared_layers != int(expected_layer_count):
        return None
    if keys == CONSTRUCTION_CONSUMPTION_PLACEHOLDER_BASE_KEYS | {"declared_layer_count"}:
        return "declared_layer_count"
    if keys != CONSTRUCTION_CONSUMPTION_PLACEHOLDER_BASE_KEYS | {
        "declared_layer_count",
        "declared_sequence_length",
        "passed",
    }:
        return None
    declared_length = value.get("declared_sequence_length")
    if isinstance(declared_length, bool) or not isinstance(declared_length, int) or declared_length <= 0:
        return None
    if value.get("passed") is not False:
        return None
    return "composed"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    """Hash dtype, shape, and values without depending on device layout."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor hash requires a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical(list(tensor.shape)))
    # Byte hashing preserves +/-inf and NaN bit patterns, which are legitimate
    # values in an additive causal mask (future entries are ``-inf``).  The
    # contiguous CPU tensor has a stable element order.
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _as_positions(
    values: Iterable[int] | torch.Tensor, *, label: str
) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ActuatorError(f"{label} must be a one-dimensional position list")
        raw = values.detach().cpu().tolist()
    else:
        raw = list(values)
    result = tuple(int(value) for value in raw)
    if len(result) != len(set(result)):
        raise ActuatorError(f"{label} must not contain duplicate positions")
    return result


def _validate_positions(
    values: Sequence[int], *, sequence_length: int, label: str, allow_empty: bool = True
) -> tuple[int, ...]:
    result = _as_positions(values, label=label)
    if not allow_empty and not result:
        raise ActuatorError(f"{label} must be non-empty")
    length = int(sequence_length)
    if any(value < 0 or value >= length for value in result):
        raise ActuatorError(
            f"{label} contains a position outside sequence length {length}"
        )
    return result


def _causal_mask(
    sequence_length: int,
    *,
    dtype: torch.dtype = torch.bool,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    length = int(sequence_length)
    if length <= 0:
        raise ActuatorError("sequence_length must be positive")
    causal = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    if dtype == torch.bool:
        return causal
    if not dtype.is_floating_point:
        raise ActuatorError("attention masks must be bool or floating point")
    zero = torch.zeros((), dtype=dtype, device=device)
    neg_inf = torch.full((), float("-inf"), dtype=dtype, device=device)
    return torch.where(causal, zero, neg_inf)


def _mask_to_4d(mask: torch.Tensor) -> torch.Tensor:
    return mask.unsqueeze(0).unsqueeze(0).contiguous()


def _mask_dtype_name(mask: torch.Tensor | None) -> str:
    return "none" if mask is None else str(mask.dtype)


def _as_float_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        zero = torch.zeros((), dtype=torch.float32, device=mask.device)
        neg_inf = torch.full((), float("-inf"), dtype=torch.float32, device=mask.device)
        return torch.where(mask, zero, neg_inf)
    if not mask.dtype.is_floating_point:
        raise ActuatorError("mask must be bool or floating point")
    return mask


def _ordered_background_positions(
    values: Sequence[Any], *, row_major_order: Mapping[int, Any] | None = None
) -> tuple[int, ...]:
    """Normalize a row-major background candidate list.

    Plain integer positions are ordered by their integer value, matching the
    flattened Qwen image-token order.  Callers that materialize a nontrivial
    order can bind it with ``row_major_order``.  ``(row, column, position)``
    tuples and mappings with ``position``/``row``/``column`` are accepted as a
    convenience for CPU materializer tests.
    """

    parsed: list[tuple[tuple[int, ...], int]] = []
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            if "position" not in item:
                raise ActuatorError("background cell mapping requires position")
            position = int(item["position"])
            order = (
                int(item.get("time", 0)),
                int(item.get("row", 0)),
                int(item.get("column", item.get("col", 0))),
            )
        elif isinstance(item, (tuple, list)) and len(item) == 4:
            # Materializer convention is (time, row, column,
            # absolute_position).
            order = (int(item[0]), int(item[1]), int(item[2]))
            position = int(item[3])
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            # Materializer convention is (row, column, absolute_position).
            order = (int(item[0]), int(item[1]))
            position = int(item[2])
        else:
            position = int(item)
            # Flattened Qwen image-token positions follow row-major materializer
            # order.  Sorting by the position keeps plain integer inputs
            # deterministic even when a caller hands us a shuffled candidate
            # bank.
            order = (position,)
        if row_major_order is not None:
            if position not in row_major_order:
                raise ActuatorError(
                    f"row_major_order has no entry for background position {position}"
                )
            bound = row_major_order[position]
            if isinstance(bound, (tuple, list)):
                order = tuple(int(part) for part in bound)
            else:
                order = (int(bound),)
        parsed.append((order, position))
    parsed.sort(key=lambda pair: (pair[0], pair[1]))
    result = tuple(position for _order, position in parsed)
    if len(result) != len(set(result)):
        raise ActuatorError("background positions must be unique")
    return result


def select_zero_overlap_background_positions(
    b_exclusive_positions: Sequence[int],
    background_positions: Sequence[Any],
    *,
    row_major_order: Mapping[int, Any] | None = None,
) -> tuple[int, ...]:
    """Select exactly ``m=|B-exclusive|`` first row-major non-overlapping cells."""

    target = _as_positions(b_exclusive_positions, label="b_exclusive_positions")
    if not target:
        raise ActuatorError("B-exclusive positions must be non-empty")
    candidates = _ordered_background_positions(
        background_positions, row_major_order=row_major_order
    )
    target_set = set(target)
    if any(position in target_set for position in candidates):
        raise ActuatorError("background candidates overlap B-exclusive positions")
    count = len(target)
    if len(candidates) < count:
        raise ActuatorError(
            f"zero-overlap background has {len(candidates)} cells but requires exactly {count}"
        )
    return candidates[:count]


def _scope_receipt(
    *,
    arm_id: str,
    sequence_length: int,
    image_key_positions: Sequence[int],
    query_positions: Sequence[int],
    selected_positions: Sequence[int],
    mask: torch.Tensor | None,
    baseline: torch.Tensor | None,
    changed_allowed: torch.Tensor | None,
    status: Literal["ready", "not_applicable"] = "ready",
    reason: str | None = None,
    mask_kind: str = "boolean_causal",
) -> dict[str, Any]:
    image = tuple(sorted(int(value) for value in image_key_positions))
    query = tuple(sorted(int(value) for value in query_positions))
    selected = tuple(sorted(int(value) for value in selected_positions))
    if mask is None or baseline is None:
        changed_count = 0
        offscope_count = 0
        future_count = 0
        non_image_count = 0
        exact = True
        mask_hash = "none"
        shape: list[int] | None = None
        dtype = "none"
    else:
        observed = mask[0, 0]
        expected = baseline[0, 0] if baseline.ndim == 4 else baseline
        changed = observed != expected
        allowed = (
            torch.zeros_like(changed)
            if changed_allowed is None
            else changed_allowed.to(device=changed.device, dtype=torch.bool)
        )
        changed_count = int(changed.sum().item())
        offscope_count = int((changed & ~allowed).sum().item())
        future_count = int(
            (changed & torch.triu(torch.ones_like(changed), diagonal=1)).sum().item()
        )
        non_image = changed.clone()
        if image:
            non_image[
                :, torch.tensor(image, dtype=torch.long, device=changed.device)
            ] = False
        non_image_count = int(non_image.sum().item())
        exact = bool(offscope_count == 0 and future_count == 0)
        mask_hash = sha256_tensor(mask)
        shape = list(mask.shape)
        dtype = str(mask.dtype)
    region = {
        "image_key_positions": list(image),
        "query_positions": list(query),
        "selected_positions": list(selected),
        "query_positions_sha256": sha256_json(list(query)),
        "selected_positions_sha256": sha256_json(list(selected)),
    }
    return {
        "schema_version": MASK_RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "arm_id": arm_id,
        "status": status,
        "reason": reason,
        "sequence_length": int(sequence_length),
        "mask_kind": mask_kind,
        "mask_shape": shape,
        "mask_dtype": dtype,
        "mask_sha256": mask_hash,
        "mask_hash": mask_hash,
        "region_sha256": sha256_json(region),
        "region_hash": sha256_json(region),
        "image_key_positions": list(image),
        "query_positions": list(query),
        "selected_positions": list(selected),
        "query_positions_sha256": sha256_json(list(query)),
        "selected_positions_sha256": sha256_json(list(selected)),
        "selected_key_count": len(selected),
        "changed_cell_count": changed_count,
        "eligible_image_cell_changes": max(changed_count - non_image_count, 0),
        "non_image_changed_cell_count": non_image_count,
        "offscope_changed_cell_count": offscope_count,
        "future_changed_cell_count": future_count,
        "mask_non_image_changed": bool(non_image_count),
        "mask_offscope_changed": bool(offscope_count),
        "mask_future_changed": bool(future_count),
        "exact_scope": exact,
    }


@dataclass(frozen=True)
class AttentionActuator:
    """One static/history arm plus its model-facing tensors and receipt."""

    arm_id: str
    attention_mask: torch.Tensor | None
    score_bias: "FixedDoseScoreBias | None"
    _receipt: Mapping[str, Any]

    @property
    def status(self) -> str:
        return str(self._receipt.get("status", "ready"))

    @property
    def applicable(self) -> bool:
        return self.status == "ready"

    @property
    def mask(self) -> torch.Tensor | None:
        """Short alias for runners that call the tensor a mask."""

        return self.attention_mask

    @property
    def bias(self) -> torch.Tensor | None:
        """Expanded [layer, head, query, key] bias tensor, if present."""

        return None if self.score_bias is None else self.score_bias.bias

    def receipt(self) -> dict[str, Any]:
        return dict(self._receipt)

    def model_inputs(self) -> dict[str, Any]:
        """Return only real model inputs; score callbacks stay explicit."""

        payload: dict[str, Any] = {}
        if self.attention_mask is not None:
            payload["attention_mask"] = self.attention_mask
        return payload

    def callback(self) -> "NaturalBoundaryActuatorCallback":
        return NaturalBoundaryActuatorCallback(self)


def _composition_json(value: Any, *, label: str) -> Any:
    """Detach a receipt value and reject tensor/callback leakage.

    Composition receipts are persisted alongside scientific results.  A
    shallow ``dict`` copy is not sufficient here because callers may hand an
    actuator a mutable nested receipt (or accidentally put a tensor in it).
    Round-tripping through the canonical JSON encoder gives us a detached,
    JSON-safe child receipt and makes the failure explicit before a model run.
    """

    try:
        return json.loads(_canonical(value).decode("utf-8"))
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        raise ActuatorError(f"{label} is not detached finite JSON: {exc}") from exc


def _composition_positions(
    receipt: Mapping[str, Any], *, key: str, label: str
) -> tuple[int, ...]:
    value = receipt.get(key)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ActuatorError(f"{label} must be a JSON list of integer positions")
    positions = tuple(int(item) for item in value)
    if len(positions) != len(set(positions)) or tuple(sorted(positions)) != positions:
        raise ActuatorError(f"{label} must be sorted and unique")
    return positions


def _composition_mask_values(mask: torch.Tensor, *, label: str) -> None:
    """Require the exact bool or canonical ``{0, -inf}`` float alphabet."""

    if mask.dtype == torch.bool:
        return
    if not mask.dtype.is_floating_point:
        raise ActuatorError(f"{label} must be bool or floating point")
    if torch.isnan(mask).any() or torch.isposinf(mask).any():
        raise ActuatorError(f"{label} contains NaN or positive infinity")
    canonical = (mask == 0) | torch.isneginf(mask)
    if not bool(torch.all(canonical).item()):
        raise ActuatorError(f"{label} contains noncanonical values; expected only 0 or -inf")


def _validate_composition_component(
    actuator: AttentionActuator,
    *,
    expected_arm_id: Literal["K10", "H20"],
    sequence_length: int | None,
) -> dict[str, Any]:
    """Validate one K10/H20 operand independently of its sibling."""

    if not isinstance(actuator, AttentionActuator):
        raise ActuatorError(f"{expected_arm_id} component is not an AttentionActuator")
    if actuator.arm_id != expected_arm_id:
        raise ActuatorError(
            f"composition requires {expected_arm_id}, got {actuator.arm_id!r}"
        )
    if actuator.status != "ready" or not actuator.applicable:
        raise ActuatorError(f"{expected_arm_id} component is not ready")
    if actuator.score_bias is not None:
        raise ActuatorError(f"{expected_arm_id} component must be mask-only")
    mask = actuator.attention_mask
    if not isinstance(mask, torch.Tensor):
        raise ActuatorError(f"{expected_arm_id} component lacks an attention mask")
    if mask.ndim != 4 or tuple(mask.shape[:2]) != (1, 1) or mask.shape[-1] != mask.shape[-2]:
        raise ActuatorError(
            f"{expected_arm_id} mask must have shape [1, 1, S, S], got {tuple(mask.shape)}"
        )
    length = int(mask.shape[-1])
    if sequence_length is not None and length != int(sequence_length):
        raise ActuatorError(f"{expected_arm_id} mask is stale for sequence length {sequence_length}")
    receipt_raw = actuator.receipt()
    receipt = _composition_json(receipt_raw, label=f"{expected_arm_id} receipt")
    if not isinstance(receipt, Mapping):  # pragma: no cover - _canonical preserves mappings
        raise ActuatorError(f"{expected_arm_id} receipt must be an object")
    if receipt.get("schema_version") != MASK_RECEIPT_SCHEMA_VERSION:
        raise ActuatorError(f"{expected_arm_id} receipt schema is stale or malformed")
    if receipt.get("unit_id") != UNIT_ID or receipt.get("arm_id") != expected_arm_id:
        raise ActuatorError(f"{expected_arm_id} receipt identity is stale or malformed")
    if receipt.get("status") != "ready":
        raise ActuatorError(f"{expected_arm_id} receipt is not ready")
    if receipt.get("sequence_length") != length:
        raise ActuatorError(f"{expected_arm_id} receipt sequence length is stale")
    if receipt.get("mask_shape") != list(mask.shape) or receipt.get("mask_dtype") != str(mask.dtype):
        raise ActuatorError(f"{expected_arm_id} receipt shape/dtype does not match its tensor")
    mask_hash = sha256_tensor(mask)
    if receipt.get("mask_sha256") != mask_hash or receipt.get("mask_hash") != mask_hash:
        raise ActuatorError(f"{expected_arm_id} receipt mask hash is stale")
    query = _composition_positions(receipt, key="query_positions", label=f"{expected_arm_id}.query_positions")
    if not query or any(position < 0 or position >= length for position in query):
        raise ActuatorError(f"{expected_arm_id}.query_positions are outside the mask")
    selected = _composition_positions(
        receipt, key="selected_positions", label=f"{expected_arm_id}.selected_positions"
    )
    image = _composition_positions(
        receipt, key="image_key_positions", label=f"{expected_arm_id}.image_key_positions"
    )
    if any(position < 0 or position >= length for position in (*selected, *image)):
        raise ActuatorError(f"{expected_arm_id} scope positions are outside the mask")
    if receipt.get("query_positions_sha256") != sha256_json(list(query)):
        raise ActuatorError(f"{expected_arm_id} query scope hash is stale")
    if receipt.get("selected_positions_sha256") != sha256_json(list(selected)):
        raise ActuatorError(f"{expected_arm_id} selected scope hash is stale")
    region = {
        "image_key_positions": list(image),
        "query_positions": list(query),
        "selected_positions": list(selected),
        "query_positions_sha256": sha256_json(list(query)),
        "selected_positions_sha256": sha256_json(list(selected)),
    }
    region_hash = sha256_json(region)
    if receipt.get("region_sha256") != region_hash or receipt.get("region_hash") != region_hash:
        raise ActuatorError(f"{expected_arm_id} region scope hash is stale")

    _composition_mask_values(mask, label=f"{expected_arm_id} mask")
    baseline2d = _causal_mask(length, dtype=mask.dtype, device=mask.device)
    baseline = _mask_to_4d(baseline2d)
    # Every component must preserve the causal future and use the same exact
    # all-allowed baseline alphabet before we reason about the changed set.
    future = torch.triu(torch.ones((length, length), dtype=torch.bool, device=mask.device), diagonal=1)
    if mask.dtype == torch.bool:
        if not bool(torch.all(mask[0, 0][future] == baseline2d[future]).item()):
            raise ActuatorError(f"{expected_arm_id} changes a causal-future cell")
    else:
        if not bool(torch.equal(mask[0, 0][future], baseline2d[future])):
            raise ActuatorError(f"{expected_arm_id} changes a causal-future cell")

    changed = mask[0, 0] != baseline2d
    if expected_arm_id == "K10":
        if not image or not selected or not set(selected).issubset(image):
            raise ActuatorError("K10 scope must contain non-empty B-exclusive keys within image keys")
        expected_pairs = {
            (query_position, key_position)
            for query_position in query
            for key_position in image
            if key_position not in set(selected) and key_position <= query_position
        }
    else:
        history = receipt.get("history_key_positions")
        if not isinstance(history, list) or tuple(history) != selected:
            raise ActuatorError("H20 history_key_positions must equal selected_positions")
        if image:
            raise ActuatorError("H20 scope must not declare image keys")
        if not selected:
            raise ActuatorError("H20 scope must contain non-empty completed-row keys")
        expected_pairs = {
            (query_position, key_position)
            for query_position in query
            for key_position in selected
            if key_position <= query_position
        }
    observed_pairs = {
        (query_position, key_position)
        for query_position in range(length)
        for key_position in range(length)
        if bool(changed[query_position, key_position].item())
    }
    if observed_pairs != expected_pairs:
        raise ActuatorError(
            f"{expected_arm_id} changed scope differs from its canonical geometry"
        )
    if receipt.get("changed_cell_count") != len(observed_pairs):
        raise ActuatorError(f"{expected_arm_id} changed_cell_count is stale")
    if receipt.get("future_changed_cell_count") != 0 or receipt.get("offscope_changed_cell_count") != 0:
        raise ActuatorError(f"{expected_arm_id} receipt reports future/offscope leakage")
    if receipt.get("exact_scope") is not True:
        raise ActuatorError(f"{expected_arm_id} receipt scope is not exact")
    return {
        "arm_id": expected_arm_id,
        "receipt": dict(receipt),
        "mask": mask,
        "baseline": baseline,
        "baseline_hash": sha256_tensor(baseline),
        "query_positions": query,
        "image_key_positions": image,
        "selected_positions": selected,
        "changed_pairs": tuple(sorted(observed_pairs)),
        "changed_mask": changed,
    }


def compose_k10_h20(
    k10: AttentionActuator,
    h20: AttentionActuator,
    *,
    cell_id: str = "C11",
) -> AttentionActuator:
    """Compose exactly the declared K10/H20 crossover cell.

    This is intentionally a two-input helper rather than a generic actuator
    factory.  It validates both independently-built operands, then performs
    boolean logical-AND or canonical float min/sum composition.  The returned
    receipt carries detached child receipts and starts with an unattested
    all-layer-consumption status; only the runner can bind actual forward
    evidence later.
    """

    if str(cell_id) != "C11":
        raise ActuatorError("K10/H20 composition is fixed to cell_id='C11'")
    left = _validate_composition_component(k10, expected_arm_id="K10", sequence_length=None)
    right = _validate_composition_component(h20, expected_arm_id="H20", sequence_length=int(left["mask"].shape[-1]))
    left_mask = left["mask"]
    right_mask = right["mask"]
    if left_mask.device != right_mask.device:
        raise ActuatorError("K10 and H20 masks must be on the same device")
    if left_mask.dtype != right_mask.dtype:
        raise ActuatorError("K10 and H20 masks must have the same dtype")
    if left["query_positions"] != right["query_positions"]:
        raise ActuatorError("K10 and H20 query scopes must match exactly")
    if left_mask.dtype == torch.bool:
        composed = torch.logical_and(left_mask, right_mask)
        expected_minimum = torch.minimum(left_mask, right_mask)
        expected_sum = left_mask & right_mask
        algebra = "logical_and"
        algebra_passed = bool(torch.equal(composed, expected_minimum) and torch.equal(composed, expected_sum))
    else:
        composed = torch.minimum(left_mask, right_mask)
        summed = left_mask + right_mask
        algebra = "minimum_and_sum"
        algebra_passed = bool(torch.equal(composed, summed))
    if not algebra_passed:
        raise ActuatorError("K10/H20 composition algebra did not produce a canonical C11 mask")

    baseline = left["baseline"]
    baseline2d = baseline[0, 0]
    composed_diff = composed[0, 0] != baseline2d
    union_diff = left["changed_mask"] | right["changed_mask"]
    if not bool(torch.equal(composed_diff, union_diff)):
        raise ActuatorError("C11 changed cells are not exactly the union of K10/H20 changes")
    length = int(composed.shape[-1])
    future = torch.triu(torch.ones((length, length), dtype=torch.bool, device=composed.device), diagonal=1)
    if composed.dtype == torch.bool:
        causal_passed = bool(torch.all(composed[0, 0][future] == baseline2d[future]).item())
    else:
        causal_passed = bool(torch.equal(composed[0, 0][future], baseline2d[future]))
    if not causal_passed:
        raise ActuatorError("C11 changes a causal-future cell")
    changed_pairs = {
        (query_position, key_position)
        for query_position in range(length)
        for key_position in range(length)
        if bool(composed_diff[query_position, key_position].item())
    }
    allowed_rows = set(left["query_positions"])
    if any(query_position not in allowed_rows or key_position > query_position for query_position, key_position in changed_pairs):
        raise ActuatorError("C11 changed cells leak outside the declared query/cause scope")
    composed_hash = sha256_tensor(composed)
    baseline_hash = sha256_tensor(baseline)
    query_positions = left["query_positions"]
    component_scopes = {
        child["arm_id"]: {
            "query_positions": list(child["query_positions"]),
            "image_key_positions": list(child["image_key_positions"]),
            "selected_positions": list(child["selected_positions"]),
            "changed_pairs": [list(pair) for pair in child["changed_pairs"]],
            "changed_cell_count": len(child["changed_pairs"]),
            "mask_sha256": sha256_tensor(child["mask"]),
            "baseline_sha256": child["baseline_hash"],
            "scope_passed": True,
        }
        for child in (left, right)
    }
    child_receipts = []
    for child in (left, right):
        detached = _composition_json(child["receipt"], label=f"{child['arm_id']} child receipt")
        child_receipts.append(
            {
                "arm_id": child["arm_id"],
                # K10/H20 are reused operators from the preceding natural
                # boundary unit; the C11 composite itself belongs only to the
                # crossover unit below.  Keeping this provenance explicit
                # prevents an old-unit receipt from masquerading as C11.
                "source_component_unit_id": child["receipt"].get("unit_id"),
                "receipt": detached,
                "receipt_sha256": sha256_json(detached),
                "mask_sha256": sha256_tensor(child["mask"]),
                "baseline_sha256": child["baseline_hash"],
            }
        )
    consumption = {
        "schema_version": CONSUMPTION_RECEIPT_SCHEMA_VERSION,
        "required": True,
        "status": "unattested",
        "passed": False,
        "exact_same_tensor_all_layers_required": True,
        "declared_sequence_length": length,
        "declared_layer_count": 28,
    }
    scope = {
        "query_positions": list(query_positions),
        "changed_pairs": [list(pair) for pair in sorted(changed_pairs)],
        "changed_cell_count": len(changed_pairs),
        "future_changed_cell_count": 0,
        "offscope_changed_cell_count": 0,
    }
    receipt = {
        "schema_version": COMPOSED_MASK_RECEIPT_SCHEMA_VERSION,
        "unit_id": CROSSOVER_UNIT_ID,
        "source_component_unit_ids": [
            child["receipt"].get("unit_id") for child in (left, right)
        ],
        "arm_id": "C11",
        "cell_id": "C11",
        "status": "ready",
        "sequence_length": length,
        "mask_kind": "composed_boolean_causal" if composed.dtype == torch.bool else "composed_float_canonical_causal",
        "mask_shape": list(composed.shape),
        "mask_dtype": str(composed.dtype),
        "mask_sha256": composed_hash,
        "mask_hash": composed_hash,
        "baseline_mask_sha256": baseline_hash,
        "baseline_mask_hash": baseline_hash,
        "baseline_mask_shape": list(baseline.shape),
        "baseline_mask_dtype": str(baseline.dtype),
        "query_positions": list(query_positions),
        "query_positions_sha256": sha256_json(list(query_positions)),
        "selected_positions": [],
        "selected_positions_sha256": sha256_json([]),
        "scope": scope,
        "component_scopes": component_scopes,
        "component_changed_cell_union": True,
        "component_changed_cell_union_count": len(changed_pairs),
        "changed_cell_count": len(changed_pairs),
        "future_changed_cell_count": 0,
        "offscope_changed_cell_count": 0,
        "exact_scope": True,
        "causal_baseline_validated": causal_passed,
        "zero_future_leakage": True,
        "offscope_leakage": False,
        "composition_algebra": algebra,
        "logical_and_passed": bool(composed.dtype == torch.bool and algebra_passed),
        "minimum_passed": bool(composed.dtype.is_floating_point and algebra_passed),
        "sum_passed": bool(composed.dtype.is_floating_point and algebra_passed),
        "children": child_receipts,
        "component_order": ["K10", "H20"],
        "layer_consumption_attestation": dict(consumption),
        "all_layer_consumption_attestation": dict(consumption),
    }
    return AttentionActuator("C11", composed, None, receipt)


def _not_applicable(
    arm_id: str,
    *,
    sequence_length: int,
    image_key_positions: Sequence[int] = (),
    query_positions: Sequence[int] = (),
    reason: str,
) -> AttentionActuator:
    receipt = _scope_receipt(
        arm_id=arm_id,
        sequence_length=sequence_length,
        image_key_positions=image_key_positions,
        query_positions=query_positions,
        selected_positions=(),
        mask=None,
        baseline=None,
        changed_allowed=None,
        status="not_applicable",
        reason=reason,
    )
    return AttentionActuator(arm_id, None, None, receipt)


def _build_hard_mask(
    arm_id: str,
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    eligible_image_positions: Sequence[int],
    query_positions: Sequence[int],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
) -> AttentionActuator:
    length = int(sequence_length)
    image = _validate_positions(
        image_key_positions, sequence_length=length, label="image_key_positions"
    )
    eligible = _validate_positions(
        eligible_image_positions,
        sequence_length=length,
        label="eligible_image_positions",
    )
    query = _validate_positions(
        query_positions,
        sequence_length=length,
        label="query_positions",
        allow_empty=False,
    )
    if not image:
        raise ActuatorError(f"{arm_id} requires non-empty image_key_positions")
    if not set(eligible).issubset(image):
        raise ActuatorError(
            f"{arm_id} eligible image keys must be a subset of image keys"
        )
    base = _causal_mask(length, dtype=dtype, device=device)
    mask = base.clone()
    blocked = tuple(sorted(set(image).difference(eligible)))
    allowed_changes = torch.zeros(
        (length, length), dtype=torch.bool, device=mask.device
    )
    for query_position in query:
        for key_position in blocked:
            # Future keys are already blocked by causality and must not be
            # rewritten by an actuator.
            if key_position <= query_position:
                allowed_changes[query_position, key_position] = True
                if dtype == torch.bool:
                    mask[query_position, key_position] = False
                else:
                    mask[query_position, key_position] = float("-inf")
    mask4d = _mask_to_4d(mask)
    baseline4d = _mask_to_4d(base)
    receipt = _scope_receipt(
        arm_id=arm_id,
        sequence_length=length,
        image_key_positions=image,
        query_positions=query,
        selected_positions=eligible,
        mask=mask4d,
        baseline=baseline4d,
        changed_allowed=allowed_changes,
        mask_kind="boolean_causal" if dtype == torch.bool else "float_additive_causal",
    )
    return AttentionActuator(arm_id, mask4d, None, receipt)


def build_k00(
    *,
    sequence_length: int | None = None,
    image_key_positions: Sequence[int] = (),
    query_positions: Sequence[int] = (),
    device: torch.device | str = "cpu",
    **_kwargs: Any,
) -> AttentionActuator:
    """Native full attention: no experiment-owned mask is supplied."""

    receipt = _scope_receipt(
        arm_id="K00",
        sequence_length=int(sequence_length or 0),
        image_key_positions=image_key_positions,
        query_positions=query_positions,
        selected_positions=(),
        mask=None,
        baseline=None,
        changed_allowed=None,
    )
    del device
    return AttentionActuator("K00", None, None, receipt)


def build_k01(
    *,
    sequence_length: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
    **_kwargs: Any,
) -> AttentionActuator:
    """Explicit all-allowed causal 4-D mask through the real model input."""

    base = _causal_mask(sequence_length, dtype=dtype, device=device)
    mask = _mask_to_4d(base)
    receipt = _scope_receipt(
        arm_id="K01",
        sequence_length=int(sequence_length),
        image_key_positions=(),
        query_positions=(),
        selected_positions=(),
        mask=mask,
        baseline=mask,
        changed_allowed=torch.zeros_like(base),
        mask_kind="explicit_all_allowed_causal"
        if dtype == torch.bool
        else "explicit_all_allowed_float_causal",
    )
    return AttentionActuator("K01", mask, None, receipt)


def build_k10(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    b_exclusive_positions: Sequence[int],
    query_positions: Sequence[int] | None = None,
    query_position: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
    **_kwargs: Any,
) -> AttentionActuator:
    """Hard B-exclusive direct query-to-image eligibility."""

    query = list(
        query_positions or ([] if query_position is None else [query_position])
    )
    b = _validate_positions(
        b_exclusive_positions,
        sequence_length=int(sequence_length),
        label="b_exclusive_positions",
        allow_empty=False,
    )
    return _build_hard_mask(
        "K10",
        sequence_length=sequence_length,
        image_key_positions=image_key_positions,
        eligible_image_positions=b,
        query_positions=query,
        device=device,
        dtype=dtype,
    )


def build_k11(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    a_exclusive_positions: Sequence[int],
    query_positions: Sequence[int] | None = None,
    query_position: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
    **_kwargs: Any,
) -> AttentionActuator:
    """Remove covered-A-exclusive image keys for the current query."""

    image = _validate_positions(
        image_key_positions,
        sequence_length=int(sequence_length),
        label="image_key_positions",
    )
    removed = _validate_positions(
        a_exclusive_positions,
        sequence_length=int(sequence_length),
        label="a_exclusive_positions",
        allow_empty=False,
    )
    if not set(removed).issubset(image):
        raise ActuatorError("K11 A-exclusive keys must be a subset of image keys")
    query = list(
        query_positions or ([] if query_position is None else [query_position])
    )
    return _build_hard_mask(
        "K11",
        sequence_length=sequence_length,
        image_key_positions=image,
        eligible_image_positions=tuple(
            position for position in image if position not in set(removed)
        ),
        query_positions=query,
        device=device,
        dtype=dtype,
    )


def _background_actuator(
    arm_id: Literal["K12", "K14B"],
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    b_exclusive_positions: Sequence[int],
    background_positions: Sequence[Any],
    query_positions: Sequence[int] | None,
    query_position: int | None,
    device: torch.device | str,
    dtype: torch.dtype,
    row_major_order: Mapping[int, Any] | None,
) -> AttentionActuator | tuple[tuple[int, ...], tuple[int, ...]]:
    b = _validate_positions(
        b_exclusive_positions,
        sequence_length=int(sequence_length),
        label="b_exclusive_positions",
        allow_empty=False,
    )
    try:
        selected = select_zero_overlap_background_positions(
            b, background_positions, row_major_order=row_major_order
        )
    except ActuatorError as exc:
        if arm_id == "K12":
            return _not_applicable(
                arm_id,
                sequence_length=sequence_length,
                image_key_positions=image_key_positions,
                query_positions=query_positions
                or (() if query_position is None else (query_position,)),
                reason=str(exc),
            )
        return _not_applicable(
            arm_id,
            sequence_length=sequence_length,
            image_key_positions=image_key_positions,
            query_positions=query_positions
            or (() if query_position is None else (query_position,)),
            reason=str(exc),
        )
    query = list(
        query_positions or ([] if query_position is None else [query_position])
    )
    if arm_id == "K12":
        return _build_hard_mask(
            arm_id,
            sequence_length=sequence_length,
            image_key_positions=image_key_positions,
            eligible_image_positions=selected,
            query_positions=query,
            device=device,
            dtype=dtype,
        )
    return selected, b


def build_k12(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    b_exclusive_positions: Sequence[int],
    background_positions: Sequence[Any] | None = None,
    zero_overlap_background_positions: Sequence[Any] | None = None,
    query_positions: Sequence[int] | None = None,
    query_position: int | None = None,
    row_major_order: Mapping[int, Any] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
    **_kwargs: Any,
) -> AttentionActuator:
    """Equal-area zero-overlap background hard eligibility (K12)."""

    candidates = (
        background_positions
        if background_positions is not None
        else zero_overlap_background_positions
    )
    if candidates is None:
        return _not_applicable(
            "K12",
            sequence_length=sequence_length,
            image_key_positions=image_key_positions,
            query_positions=query_positions
            or (() if query_position is None else (query_position,)),
            reason="zero_overlap_background_positions are unavailable",
        )
    result = _background_actuator(
        "K12",
        sequence_length=sequence_length,
        image_key_positions=image_key_positions,
        b_exclusive_positions=b_exclusive_positions,
        background_positions=candidates,
        query_positions=query_positions,
        query_position=query_position,
        device=device,
        dtype=dtype,
        row_major_order=row_major_order,
    )
    assert isinstance(result, AttentionActuator)
    return result


def build_k13(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    same_class_competitor_positions: Sequence[int] | None = None,
    competitor_positions: Sequence[int] | None = None,
    query_positions: Sequence[int] | None = None,
    query_position: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
    **_kwargs: Any,
) -> AttentionActuator:
    """Same-class competitor spotlight, or explicit ``not_applicable``."""

    selected = (
        same_class_competitor_positions
        if same_class_competitor_positions is not None
        else competitor_positions
    )
    query = query_positions or (() if query_position is None else (query_position,))
    if selected is None or not selected:
        return _not_applicable(
            "K13",
            sequence_length=sequence_length,
            image_key_positions=image_key_positions,
            query_positions=query,
            reason="same-class competitor positions are unavailable",
        )
    return _build_hard_mask(
        "K13",
        sequence_length=sequence_length,
        image_key_positions=image_key_positions,
        eligible_image_positions=selected,
        query_positions=query,
        device=device,
        dtype=dtype,
    )


def _bias_receipt(
    *,
    arm_id: str,
    sequence_length: int,
    query_positions: Sequence[int],
    selected_positions: Sequence[int],
    bias: torch.Tensor,
    layer_count: int,
    head_count: int,
    status: Literal["ready", "not_applicable"] = "ready",
    reason: str | None = None,
    target_positions: Sequence[int] = (),
    region_positions: Sequence[int] = (),
) -> dict[str, Any]:
    q = tuple(sorted(int(value) for value in query_positions))
    selected = tuple(sorted(int(value) for value in selected_positions))
    target = tuple(sorted(int(value) for value in target_positions))
    region = tuple(sorted(int(value) for value in region_positions))
    nonzero = bias != 0
    expected_count = (
        int(layer_count)
        * int(head_count)
        * sum(
            1
            for query_position in q
            for key_position in selected
            if key_position <= query_position
        )
    )
    actual_count = int(nonzero.sum().item())
    future = torch.triu(torch.ones_like(bias, dtype=torch.bool), diagonal=1)
    future_nonzero = int((nonzero & future).sum().item())
    # The tensor is [LAYER, HEAD, QUERY, KEY].  ``changed`` is expected only
    # at the current query and selected keys; all off-scope cells stay zero.
    allowed = torch.zeros_like(nonzero)
    for _layer in range(int(layer_count)):
        for _head in range(int(head_count)):
            for query_position in q:
                for key_position in selected:
                    if key_position <= query_position:
                        allowed[_layer, _head, query_position, key_position] = True
    offscope = int((nonzero & ~allowed).sum().item())
    exact_values = (
        bool(torch.all(bias[nonzero] == float(FIXED_BIAS)).item())
        if actual_count
        else False
    )
    return {
        "schema_version": BIAS_RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "arm_id": arm_id,
        "status": status,
        "reason": reason,
        "sequence_length": int(sequence_length),
        "layer_count": int(layer_count),
        "head_count": int(head_count),
        "query_positions": list(q),
        "selected_positions": list(selected),
        "target_positions": list(target),
        "region_positions": list(region),
        "query_positions_sha256": sha256_json(list(q)),
        "selected_positions_sha256": sha256_json(list(selected)),
        "selected_key_count": len(selected),
        "bias_entry_count": actual_count,
        "expected_bias_entry_count": expected_count,
        "offscope_nonzero_count": offscope,
        "future_nonzero_count": future_nonzero,
        "exact_plus_two_values": exact_values,
        "bias_shape": list(bias.shape),
        "bias_dtype": str(bias.dtype),
        "bias_sha256": sha256_tensor(bias),
        "bias_hash": sha256_tensor(bias),
        "region_sha256": sha256_json(
            {
                "query_positions": list(q),
                "selected_positions": list(selected),
                "target_positions": list(target),
                "region_positions": list(region),
            }
        ),
        "region_hash": sha256_json(
            {
                "query_positions": list(q),
                "selected_positions": list(selected),
                "target_positions": list(target),
                "region_positions": list(region),
            }
        ),
        "block23_mass_requirement": {
            "schema_version": BLOCK23_MASS_RECEIPT_SCHEMA_VERSION,
            "required_for_soft_interpretation": True,
            "status": "unattested",
            "block_idx": 23,
            "selected_key_positions": list(selected),
            "per_head_before_after_required": True,
            "applied_bias_and_mass_shift_receipts_required": True,
        },
        "scope_passed": bool(
            actual_count == expected_count
            and offscope == 0
            and future_nonzero == 0
            and exact_values
        ),
    }


class FixedDoseScoreBias:
    """A pre-softmax score-bias callback for every layer and attention head."""

    def __init__(
        self,
        bias: torch.Tensor,
        *,
        arm_id: str,
        query_positions: Sequence[int],
        selected_positions: Sequence[int],
        target_positions: Sequence[int] = (),
        region_positions: Sequence[int] = (),
        layer_count: int,
        head_count: int,
        receipt: Mapping[str, Any] | None = None,
    ) -> None:
        if bias.ndim != 4:
            raise ActuatorError(
                "score bias must have shape [layers, heads, sequence, sequence]"
            )
        if bias.shape[0] != int(layer_count) or bias.shape[1] != int(head_count):
            raise ActuatorError(
                "score bias layer/head shape disagrees with its declared scope"
            )
        self.bias = bias.detach().clone()
        self.arm_id = str(arm_id)
        self.query_positions = tuple(int(value) for value in query_positions)
        self.selected_positions = tuple(int(value) for value in selected_positions)
        self.layer_count = int(layer_count)
        self.head_count = int(head_count)
        self._receipt = dict(
            receipt
            if receipt is not None
            else _bias_receipt(
                arm_id=arm_id,
                sequence_length=int(bias.shape[-1]),
                query_positions=query_positions,
                selected_positions=selected_positions,
                bias=bias,
                layer_count=layer_count,
                head_count=head_count,
                target_positions=target_positions,
                region_positions=region_positions,
            )
        )
        self._application_count = 0
        self._applications_by_layer: dict[int, int] = {}

    @property
    def sequence_length(self) -> int:
        return int(self.bias.shape[-1])

    def receipt(self) -> dict[str, Any]:
        result = dict(self._receipt)
        result["application_count"] = self._application_count
        result["applications_by_layer"] = dict(
            sorted(self._applications_by_layer.items())
        )
        return result

    def for_layer(self, layer_idx: int, *, batch_size: int = 1) -> torch.Tensor:
        layer = int(layer_idx)
        if not 0 <= layer < self.layer_count:
            raise ActuatorError(
                f"layer_idx {layer} is outside declared layer count {self.layer_count}"
            )
        if int(batch_size) <= 0:
            raise ActuatorError("batch_size must be positive")
        return self.bias[layer].unsqueeze(0).expand(int(batch_size), -1, -1, -1)

    def apply(
        self,
        scores: torch.Tensor,
        *,
        layer_idx: int,
        query_positions: Sequence[int] | None = None,
        key_positions: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Add bias to scaled QK scores, before softmax.

        ``scores`` may be ``[B,H,Q,K]`` or ``[H,Q,K]``.  Absolute query/key
        positions are explicit when a cache or a one-token forward changes the
        local tensor dimensions.  The callback never applies to an implicit
        post-softmax probability tensor.
        """

        if not isinstance(scores, torch.Tensor) or scores.ndim not in {3, 4}:
            raise ActuatorError("scaled QK scores must have shape [B,H,Q,K] or [H,Q,K]")
        layer = int(layer_idx)
        if not 0 <= layer < self.layer_count:
            raise ActuatorError(f"layer_idx {layer} is outside declared scope")
        local_q = (
            self.query_positions
            if query_positions is None
            else tuple(int(v) for v in query_positions)
        )
        if len(local_q) != int(scores.shape[-2]):
            raise ActuatorError(
                "query_positions length must equal score query dimension"
            )
        if key_positions is None:
            local_k = tuple(range(int(scores.shape[-1])))
        else:
            local_k = tuple(int(value) for value in key_positions)
        if len(local_k) != int(scores.shape[-1]):
            raise ActuatorError("key_positions length must equal score key dimension")
        if any(
            value < 0 or value >= self.sequence_length for value in local_q + local_k
        ):
            raise ActuatorError("score positions fall outside the declared sequence")
        head_count = (
            int(scores.shape[-3]) if scores.ndim == 4 else int(scores.shape[-3])
        )
        if head_count != self.head_count:
            raise ActuatorError(
                "score head count differs from declared layer/head scope"
            )
        local_bias = torch.zeros_like(scores)
        for q_index, absolute_query in enumerate(local_q):
            for k_index, absolute_key in enumerate(local_k):
                if (
                    absolute_query >= 0
                    and absolute_key in self.selected_positions
                    and absolute_key <= absolute_query
                ):
                    if scores.ndim == 4:
                        local_bias[:, :, q_index, k_index] = float(FIXED_BIAS)
                    else:
                        local_bias[:, q_index, k_index] = float(FIXED_BIAS)
        self._application_count += 1
        self._applications_by_layer[layer] = (
            self._applications_by_layer.get(layer, 0) + 1
        )
        return scores + local_bias

    __call__ = apply


def _build_score_bias(
    arm_id: Literal["K14T", "K14B"],
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    selected_positions: Sequence[int],
    query_positions: Sequence[int] | None,
    query_position: int | None,
    layer_count: int,
    head_count: int,
    device: torch.device | str,
    target_positions: Sequence[int] = (),
    region_positions: Sequence[int] = (),
) -> AttentionActuator:
    length = int(sequence_length)
    image = _validate_positions(
        image_key_positions, sequence_length=length, label="image_key_positions"
    )
    selected = _validate_positions(
        selected_positions,
        sequence_length=length,
        label="selected_positions",
        allow_empty=False,
    )
    if not set(selected).issubset(image):
        raise ActuatorError(f"{arm_id} selected keys must be a subset of image keys")
    query = _validate_positions(
        query_positions or ([] if query_position is None else [query_position]),
        sequence_length=length,
        label="query_positions",
        allow_empty=False,
    )
    if int(layer_count) <= 0 or int(head_count) <= 0:
        raise ActuatorError("layer_count and head_count must be positive")
    # K14 must be consumable through the model's real ``attention_mask``
    # input.  The old implementation returned only ``FixedDoseScoreBias``;
    # the natural runner deliberately strips callback metadata before calling
    # the model, so that representation was a scientific no-op in production.
    # Keep the callback as a diagnostic/compatibility view, but make this
    # additive causal mask the authoritative intervention surface.
    bias = torch.zeros(
        (int(layer_count), int(head_count), length, length),
        dtype=torch.float32,
        device=device,
    )
    for layer in range(int(layer_count)):
        for head in range(int(head_count)):
            for query_pos in query:
                for key_pos in selected:
                    if key_pos <= query_pos:
                        bias[layer, head, query_pos, key_pos] = float(FIXED_BIAS)
    additive_base = _causal_mask(length, dtype=torch.float32, device=device)
    additive_mask = additive_base.clone()
    # ``attention_mask`` is [1, 1, query, key], while the receipt/controller
    # retains an all-layer/all-head delta.  Every selected edge receives the
    # same fixed dose and future edges remain ``-inf`` rather than becoming a
    # newly visible key.
    for query_pos in query:
        for key_pos in selected:
            if key_pos <= query_pos:
                additive_mask[query_pos, key_pos] = float(FIXED_BIAS)
    additive_mask4d = _mask_to_4d(additive_mask)
    additive_base4d = _mask_to_4d(additive_base)
    allowed_changes = torch.zeros(
        (length, length), dtype=torch.bool, device=additive_mask.device
    )
    for query_pos in query:
        for key_pos in selected:
            if key_pos <= query_pos:
                allowed_changes[query_pos, key_pos] = True
    mask_receipt = _scope_receipt(
        arm_id=arm_id,
        sequence_length=length,
        image_key_positions=image,
        query_positions=query,
        selected_positions=selected,
        mask=additive_mask4d,
        baseline=additive_base4d,
        changed_allowed=allowed_changes,
        mask_kind="float_additive_causal_plus_two",
    )
    receipt = _bias_receipt(
        arm_id=arm_id,
        sequence_length=length,
        query_positions=query,
        selected_positions=selected,
        bias=bias,
        layer_count=layer_count,
        head_count=head_count,
        target_positions=target_positions,
        region_positions=region_positions,
    )
    # Preserve the score-delta receipt while binding the tensor actually sent
    # to the model.  These fields make a metadata-only callback distinguishable
    # from an applied additive attention mask in downstream receipts.
    receipt.update(
        {
            "attention_mask": mask_receipt,
            "attention_mask_sha256": sha256_tensor(additive_mask4d),
            "attention_mask_hash": sha256_tensor(additive_mask4d),
            "attention_mask_shape": list(additive_mask4d.shape),
            "attention_mask_dtype": str(additive_mask4d.dtype),
            "mask_kind": "float_additive_causal_plus_two",
            "mask_consumed_via": "attention_mask",
            "mask_input_key": "attention_mask",
            "layer_consumption_attestation": {
                "schema_version": CONSUMPTION_RECEIPT_SCHEMA_VERSION,
                "required": True,
                "status": "unattested",
                "exact_same_tensor_all_layers_required": True,
                "declared_layer_count": int(layer_count),
            },
            "all_layer_consumption_attestation": {
                "schema_version": CONSUMPTION_RECEIPT_SCHEMA_VERSION,
                "required": True,
                "status": "unattested",
                "exact_same_tensor_all_layers_required": True,
                "declared_layer_count": int(layer_count),
            },
            "dose": float(FIXED_BIAS),
            "dose_applied": True,
            "selected_identity_sha256": sha256_json(list(selected)),
            "selected_identity_hash": sha256_json(list(selected)),
            "selected_count_balanced": True,
            "offscope_mask_change_count": mask_receipt["offscope_changed_cell_count"],
            "future_mask_change_count": mask_receipt["future_changed_cell_count"],
            "mask_scope_passed": mask_receipt["exact_scope"],
        }
    )
    controller = FixedDoseScoreBias(
        bias,
        arm_id=arm_id,
        query_positions=query,
        selected_positions=selected,
        target_positions=target_positions,
        region_positions=region_positions,
        layer_count=layer_count,
        head_count=head_count,
        receipt=receipt,
    )
    return AttentionActuator(arm_id, additive_mask4d, controller, receipt)


def build_k14t(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    b_exclusive_positions: Sequence[int],
    query_positions: Sequence[int] | None = None,
    query_position: int | None = None,
    layer_count: int = 1,
    head_count: int = 1,
    device: torch.device | str = "cpu",
    **_kwargs: Any,
) -> AttentionActuator:
    """Fixed ``+2.0`` bias on every B-exclusive key (K14T)."""

    b = _validate_positions(
        b_exclusive_positions,
        sequence_length=int(sequence_length),
        label="b_exclusive_positions",
        allow_empty=False,
    )
    return _build_score_bias(
        "K14T",
        sequence_length=sequence_length,
        image_key_positions=image_key_positions,
        selected_positions=b,
        query_positions=query_positions,
        query_position=query_position,
        layer_count=layer_count,
        head_count=head_count,
        device=device,
        target_positions=b,
        region_positions=b,
    )


def build_k14b(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    b_exclusive_positions: Sequence[int],
    background_positions: Sequence[Any] | None = None,
    zero_overlap_background_positions: Sequence[Any] | None = None,
    query_positions: Sequence[int] | None = None,
    query_position: int | None = None,
    row_major_order: Mapping[int, Any] | None = None,
    layer_count: int = 1,
    head_count: int = 1,
    device: torch.device | str = "cpu",
    **_kwargs: Any,
) -> AttentionActuator:
    """Fixed ``+2.0`` bias on the exact K12 background control (K14B)."""

    candidates = (
        background_positions
        if background_positions is not None
        else zero_overlap_background_positions
    )
    query = query_positions or (() if query_position is None else (query_position,))
    if candidates is None:
        return _not_applicable(
            "K14B",
            sequence_length=sequence_length,
            image_key_positions=image_key_positions,
            query_positions=query,
            reason="zero_overlap_background_positions are unavailable",
        )
    try:
        selected = select_zero_overlap_background_positions(
            b_exclusive_positions, candidates, row_major_order=row_major_order
        )
    except ActuatorError as exc:
        return _not_applicable(
            "K14B",
            sequence_length=sequence_length,
            image_key_positions=image_key_positions,
            query_positions=query,
            reason=str(exc),
        )
    return _build_score_bias(
        "K14B",
        sequence_length=sequence_length,
        image_key_positions=image_key_positions,
        selected_positions=selected,
        query_positions=query_positions,
        query_position=query_position,
        layer_count=layer_count,
        head_count=head_count,
        device=device,
        target_positions=b_exclusive_positions,
        region_positions=selected,
    )


def build_k14(arm_id: Literal["K14T", "K14B"], **kwargs: Any) -> AttentionActuator:
    return build_k14t(**kwargs) if arm_id == "K14T" else build_k14b(**kwargs)


def build_k_arm(arm_id: str, **kwargs: Any) -> AttentionActuator:
    """Dispatch helper used by runners and tests."""

    dispatch: dict[str, Callable[..., AttentionActuator]] = {
        "K00": build_k00,
        "K01": build_k01,
        "K10": build_k10,
        "K11": build_k11,
        "K12": build_k12,
        "K13": build_k13,
        "K14T": build_k14t,
        "K14B": build_k14b,
    }
    try:
        return dispatch[str(arm_id)](**kwargs)
    except KeyError as exc:
        raise ActuatorError(f"unknown static actuator arm {arm_id!r}") from exc


def _build_history_mask(
    arm_id: Literal["H00", "H10", "H20"],
    *,
    sequence_length: int,
    query_positions: Sequence[int],
    blocked_positions: Sequence[int],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
) -> AttentionActuator:
    length = int(sequence_length)
    image: tuple[int, ...] = ()
    query = _validate_positions(
        query_positions,
        sequence_length=length,
        label="query_positions",
        allow_empty=False,
    )
    blocked = _validate_positions(
        blocked_positions,
        sequence_length=length,
        label="history_key_positions",
        allow_empty=False,
    )
    base = _causal_mask(length, dtype=dtype, device=device)
    mask = base.clone()
    allowed = torch.zeros((length, length), dtype=torch.bool, device=mask.device)
    for query_position in query:
        for key_position in blocked:
            if key_position <= query_position:
                allowed[query_position, key_position] = True
                if dtype == torch.bool:
                    mask[query_position, key_position] = False
                else:
                    mask[query_position, key_position] = float("-inf")
    mask4d = _mask_to_4d(mask)
    baseline4d = _mask_to_4d(base)
    receipt = _scope_receipt(
        arm_id=arm_id,
        sequence_length=length,
        image_key_positions=image,
        query_positions=query,
        selected_positions=blocked,
        mask=mask4d,
        baseline=baseline4d,
        changed_allowed=allowed,
        mask_kind="history_boolean_causal"
        if dtype == torch.bool
        else "history_float_additive_causal",
    )
    receipt["history_key_positions"] = list(blocked)
    receipt["masked_query_key_edge_count"] = int(receipt["changed_cell_count"])
    return AttentionActuator(arm_id, mask4d, None, receipt)


def build_h00(
    *,
    sequence_length: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
    **_kwargs: Any,
) -> AttentionActuator:
    """Native/no-op history control through an explicit causal 4-D mask."""

    base = _causal_mask(sequence_length, dtype=dtype, device=device)
    mask = _mask_to_4d(base)
    receipt = _scope_receipt(
        arm_id="H00",
        sequence_length=int(sequence_length),
        image_key_positions=(),
        query_positions=tuple(range(int(sequence_length))),
        selected_positions=(),
        mask=mask,
        baseline=mask,
        changed_allowed=torch.zeros_like(base),
        mask_kind="history_explicit_all_allowed_causal"
        if dtype == torch.bool
        else "history_explicit_all_allowed_float_causal",
    )
    receipt["history_key_positions"] = []
    receipt["masked_query_key_edge_count"] = 0
    return AttentionActuator("H00", mask, None, receipt)


def build_h10(
    *,
    sequence_length: int,
    query_positions: Sequence[int],
    latest_terminal_key_position: int | None = None,
    latest_terminal_key_positions: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
    **_kwargs: Any,
) -> AttentionActuator:
    keys = latest_terminal_key_positions
    if keys is None and latest_terminal_key_position is not None:
        keys = (latest_terminal_key_position,)
    if not keys:
        return _not_applicable(
            "H10",
            sequence_length=sequence_length,
            query_positions=query_positions,
            reason="latest terminal-carrier key is unavailable",
        )
    return _build_history_mask(
        "H10",
        sequence_length=sequence_length,
        query_positions=query_positions,
        blocked_positions=keys,
        device=device,
        dtype=dtype,
    )


def build_h20(
    *,
    sequence_length: int,
    query_positions: Sequence[int],
    latest_row_key_positions: Sequence[int] | None = None,
    latest_row_positions: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
    **_kwargs: Any,
) -> AttentionActuator:
    keys = (
        latest_row_key_positions
        if latest_row_key_positions is not None
        else latest_row_positions
    )
    if not keys:
        return _not_applicable(
            "H20",
            sequence_length=sequence_length,
            query_positions=query_positions,
            reason="latest completed-row keys are unavailable",
        )
    return _build_history_mask(
        "H20",
        sequence_length=sequence_length,
        query_positions=query_positions,
        blocked_positions=keys,
        device=device,
        dtype=dtype,
    )


def build_h_arm(arm_id: str, **kwargs: Any) -> AttentionActuator:
    dispatch: dict[str, Callable[..., AttentionActuator]] = {
        "H00": build_h00,
        "H10": build_h10,
        "H20": build_h20,
    }
    try:
        return dispatch[str(arm_id)](**kwargs)
    except KeyError as exc:
        raise ActuatorError(f"unknown history actuator arm {arm_id!r}") from exc


def _absolute_positions(
    values: Iterable[int] | torch.Tensor,
    *,
    label: str,
    allow_empty: bool = True,
) -> tuple[int, ...]:
    """Validate fixed absolute identities without tying them to one step.

    A natural scalar forward grows from ``S`` to ``S+1``.  Validating image or
    history positions against the first ``S`` would either reject a legitimate
    later key or, worse, force a stale prebuilt mask to be reused.  Factories
    therefore validate non-negativity and uniqueness once, then activate only
    positions ``< current_sequence_length`` for each rebuild.
    """

    result = _as_positions(values, label=label)
    if not allow_empty and not result:
        raise ActuatorError(f"{label} must be non-empty")
    if any(value < 0 for value in result):
        raise ActuatorError(f"{label} must contain non-negative absolute positions")
    return result


class ScalarStepActuatorFactory:
    """Rebuild one actuator for every growing natural scalar prefix.

    The factory is deliberately state-light: fixed image/history identities are
    captured at construction, while ``build`` receives the current prefix
    length (and, optionally, its absolute query position) and creates a fresh
    ``[1, 1, S, S]`` tensor.  This prevents a mask made for one prefix from
    leaking stale rows/columns into a later scalar call.  The callable form is
    compatible with ``run_natural_boundary_routing_history_probe``'s
    callback path (including a context-first wrapper); length/query are
    inferred from ``input_ids`` when necessary.

    K14 is represented by a float additive causal mask with ``+2.0`` in the
    declared current-query/selected-key cells.  ``score_bias`` remains on the
    returned actuator solely for direct unit diagnostics and old consumers;
    callers must pass ``attention_mask`` to the model.
    """

    protocol = "natural_boundary_scalar_step_actuator_factory.v1"

    def __init__(
        self,
        arm_id: str,
        *,
        image_key_positions: Sequence[int] = (),
        b_exclusive_positions: Sequence[int] = (),
        a_exclusive_positions: Sequence[int] = (),
        background_positions: Sequence[Any] | None = None,
        zero_overlap_background_positions: Sequence[Any] | None = None,
        same_class_competitor_positions: Sequence[int] | None = None,
        competitor_positions: Sequence[int] | None = None,
        latest_terminal_key_positions: Sequence[int] | None = None,
        latest_terminal_key_position: int | None = None,
        latest_row_key_positions: Sequence[int] | None = None,
        latest_row_positions: Sequence[int] | None = None,
        query_position: int | None = None,
        sequence_length: int | None = None,
        row_major_order: Mapping[int, Any] | None = None,
        layer_count: int = 1,
        head_count: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_sequence_length: int | None = None,
    ) -> None:
        normalized_arm = str(arm_id)
        if normalized_arm not in K_ARM_IDS + H_ARM_IDS:
            raise ActuatorError(f"unknown scalar-step actuator arm {arm_id!r}")
        if sequence_length is not None:
            if max_sequence_length is not None and int(max_sequence_length) != int(
                sequence_length
            ):
                raise ActuatorError("sequence_length and max_sequence_length disagree")
            max_sequence_length = sequence_length
        if max_sequence_length is not None and int(max_sequence_length) <= 0:
            raise ActuatorError("max_sequence_length must be positive when supplied")
        if dtype != torch.bool and not dtype.is_floating_point:
            raise ActuatorError(
                "scalar-step attention masks must be bool or floating point"
            )
        if normalized_arm in {"K14T", "K14B"}:
            if int(layer_count) <= 0 or int(head_count) <= 0:
                raise ActuatorError("K14 layer_count and head_count must be positive")
        self.arm_id = normalized_arm
        self.image_key_positions = _absolute_positions(
            image_key_positions, label="image_key_positions"
        )
        self.b_exclusive_positions = _absolute_positions(
            b_exclusive_positions,
            label="b_exclusive_positions",
            allow_empty=normalized_arm not in {"K10", "K14T", "K14B"},
        )
        self.a_exclusive_positions = _absolute_positions(
            a_exclusive_positions, label="a_exclusive_positions"
        )
        raw_background = background_positions
        if raw_background is None:
            raw_background = zero_overlap_background_positions
        self.background_positions = (
            tuple(raw_background) if raw_background is not None else None
        )
        self.same_class_competitor_positions = _absolute_positions(
            same_class_competitor_positions
            if same_class_competitor_positions is not None
            else (competitor_positions or ()),
            label="same_class_competitor_positions",
        )
        if (
            latest_terminal_key_positions is None
            and latest_terminal_key_position is not None
        ):
            latest_terminal_key_positions = (latest_terminal_key_position,)
        self.latest_terminal_key_positions = _absolute_positions(
            latest_terminal_key_positions or (), label="latest_terminal_key_positions"
        )
        self.latest_row_key_positions = _absolute_positions(
            latest_row_key_positions
            if latest_row_key_positions is not None
            else (latest_row_positions or ()),
            label="latest_row_key_positions",
        )
        self.row_major_order = (
            dict(row_major_order) if row_major_order is not None else None
        )
        self.layer_count = int(layer_count)
        self.head_count = int(head_count)
        self.device = device
        self.dtype = dtype
        self.max_sequence_length = (
            int(max_sequence_length) if max_sequence_length is not None else None
        )
        self.default_query_position = (
            None if query_position is None else int(query_position)
        )
        if self.default_query_position is not None and self.default_query_position < 0:
            raise ActuatorError("query_position must be non-negative when supplied")
        self._build_count = 0
        self._step_receipts: list[dict[str, Any]] = []
        self._background_selection: tuple[int, ...] | None = None
        self._background_selection_error: str | None = None
        if self.background_positions is not None and self.b_exclusive_positions:
            try:
                self._background_selection = select_zero_overlap_background_positions(
                    self.b_exclusive_positions,
                    self.background_positions,
                    row_major_order=self.row_major_order,
                )
            except ActuatorError as exc:
                # The arm becomes ``not_applicable`` at build time with the
                # exact selection failure in its receipt.  Construction still
                # succeeds so a matrix can report the declared disposition.
                self._background_selection = None
                self._background_selection_error = str(exc)

    @staticmethod
    def _active(values: Sequence[int], sequence_length: int) -> tuple[int, ...]:
        return tuple(
            int(value) for value in values if int(value) < int(sequence_length)
        )

    def _validate_step(
        self, sequence_length: int, query_position: int | None
    ) -> tuple[int, int]:
        current = int(sequence_length)
        if current <= 0:
            raise ActuatorError("sequence_length must be positive")
        if self.max_sequence_length is not None and current > self.max_sequence_length:
            raise ActuatorError(
                f"sequence_length {current} exceeds factory max_sequence_length {self.max_sequence_length}"
            )
        query = (
            current - 1
            if query_position is None and self.default_query_position is None
            else self.default_query_position
            if query_position is None
            else int(query_position)
        )
        if query < 0 or query >= current:
            raise ActuatorError(
                f"query_position {query} must be within current sequence length {current}"
            )
        return current, query

    def _with_factory_receipt(
        self,
        actuator: AttentionActuator,
        *,
        sequence_length: int,
        query_position: int,
        active_image: Sequence[int],
        active_b: Sequence[int],
        active_a: Sequence[int],
        active_competitor: Sequence[int],
        fixed_history: Sequence[int],
        active_history: Sequence[int],
        active_background: Sequence[int],
    ) -> AttentionActuator:
        receipt = actuator.receipt()
        fixed = {
            "protocol": self.protocol,
            "factory_arm_id": self.arm_id,
            "factory_build_index": self._build_count,
            "current_sequence_length": int(sequence_length),
            "current_query_position": int(query_position),
            "fixed_absolute_image_key_positions": list(self.image_key_positions),
            "fixed_absolute_b_exclusive_positions": list(self.b_exclusive_positions),
            "fixed_absolute_a_exclusive_positions": list(self.a_exclusive_positions),
            "fixed_absolute_same_class_competitor_positions": list(
                self.same_class_competitor_positions
            ),
            "fixed_absolute_history_key_positions": list(fixed_history),
            "active_image_key_positions": list(active_image),
            "active_b_exclusive_positions": list(active_b),
            "active_a_exclusive_positions": list(active_a),
            "active_same_class_competitor_positions": list(active_competitor),
            "active_history_key_positions": list(active_history),
            "active_background_positions": list(active_background),
            "mask_input_key": "attention_mask",
            "fixed_image_key_positions_sha256": sha256_json(
                list(self.image_key_positions)
            ),
            "fixed_competitor_positions_sha256": sha256_json(
                list(self.same_class_competitor_positions)
            ),
            "fixed_history_key_positions_sha256": sha256_json(list(fixed_history)),
            "sequence_growth_rebuilt": True,
        }
        if self._background_selection is not None:
            fixed.update(
                {
                    "fixed_background_selection": list(self._background_selection),
                    "fixed_background_selection_sha256": sha256_json(
                        list(self._background_selection)
                    ),
                    "fixed_background_selection_count": len(self._background_selection),
                }
            )
        if self._background_selection_error is not None:
            fixed["fixed_background_selection_error"] = self._background_selection_error
        receipt.update(fixed)
        if "layer_consumption_attestation" not in receipt:
            receipt["layer_consumption_attestation"] = {
                "schema_version": CONSUMPTION_RECEIPT_SCHEMA_VERSION,
                "required": True,
                "status": "unattested",
                "exact_same_tensor_all_layers_required": True,
            }
        receipt["all_layer_consumption_attestation"] = dict(
            receipt["layer_consumption_attestation"]
        )
        if self.arm_id in {"K01", "H00"}:
            # Construction proves only mask shape/scope.  A no-op control is
            # not accepted until a live runner compares the complete
            # full-vocabulary logits and trajectory against native execution.
            receipt["no_op_parity"] = {
                "required": True,
                "status": "unassessed",
                "full_vocabulary_logits_required": True,
                "max_abs_logit_drift_tolerance": float(NOOP_TOLERANCE),
                "complete_native_trajectory_required": True,
                "selected_token_only_is_insufficient": True,
            }
        self._step_receipts.append(dict(receipt))
        self._build_count += 1
        return AttentionActuator(
            actuator.arm_id,
            actuator.attention_mask,
            actuator.score_bias,
            receipt,
        )

    def build(
        self,
        sequence_length: int,
        *,
        query_position: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> AttentionActuator:
        """Build a fresh current-prefix actuator; never reuse a prior mask."""

        current, query = self._validate_step(sequence_length, query_position)
        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        if target_dtype != torch.bool and not target_dtype.is_floating_point:
            raise ActuatorError(
                "scalar-step attention masks must be bool or floating point"
            )
        active_image = self._active(self.image_key_positions, current)
        active_b = self._active(self.b_exclusive_positions, current)
        active_a = self._active(self.a_exclusive_positions, current)
        active_terminal = self._active(self.latest_terminal_key_positions, current)
        active_row = self._active(self.latest_row_key_positions, current)
        active_competitor = self._active(self.same_class_competitor_positions, current)
        active_background = self._active(self._background_selection or (), current)
        q = (query,)

        def not_ready(reason: str) -> AttentionActuator:
            if self.arm_id == "K00":
                actuator = _not_applicable(
                    self.arm_id,
                    sequence_length=current,
                    image_key_positions=active_image,
                    query_positions=q,
                    reason=reason,
                )
            else:
                # A scalar callback still needs a correctly shaped causal
                # tensor when an intervention is not applicable at this
                # prefix (for example, a fixed absolute B key has not entered
                # the growing sequence yet).  Keep the explicit disposition
                # while returning a consumable no-op mask rather than a stale
                # prior-length tensor or ``None``.
                base = _causal_mask(current, dtype=target_dtype, device=target_device)
                base4d = _mask_to_4d(base)
                base_receipt = _scope_receipt(
                    arm_id=self.arm_id,
                    sequence_length=current,
                    image_key_positions=active_image,
                    query_positions=q,
                    selected_positions=(),
                    mask=base4d,
                    baseline=base4d,
                    changed_allowed=torch.zeros_like(base),
                    status="not_applicable",
                    reason=reason,
                    mask_kind="float_additive_causal"
                    if target_dtype != torch.bool
                    else "boolean_causal",
                )
                actuator = AttentionActuator(self.arm_id, base4d, None, base_receipt)
            return self._with_factory_receipt(
                actuator,
                sequence_length=current,
                query_position=query,
                active_image=active_image,
                active_b=active_b,
                active_a=active_a,
                active_competitor=active_competitor,
                fixed_history=self.latest_terminal_key_positions
                if self.arm_id == "H10"
                else self.latest_row_key_positions
                if self.arm_id == "H20"
                else (),
                active_history=active_terminal if self.arm_id == "H10" else active_row,
                active_background=active_background,
            )

        arm = self.arm_id
        if arm == "K00":
            actuator = build_k00(
                sequence_length=current,
                image_key_positions=active_image,
                query_positions=q,
                device=target_device,
            )
        elif arm == "K01":
            actuator = build_k01(
                sequence_length=current, device=target_device, dtype=target_dtype
            )
        elif arm == "K10":
            if not active_image or not active_b:
                return not_ready(
                    "B-exclusive image keys are not present in the current scalar prefix"
                )
            actuator = build_k10(
                sequence_length=current,
                image_key_positions=active_image,
                b_exclusive_positions=active_b,
                query_positions=q,
                device=target_device,
                dtype=target_dtype,
            )
        elif arm == "K11":
            if not active_image:
                return not_ready(
                    "image keys are not present in the current scalar prefix"
                )
            if active_a:
                actuator = build_k11(
                    sequence_length=current,
                    image_key_positions=active_image,
                    a_exclusive_positions=active_a,
                    query_positions=q,
                    device=target_device,
                    dtype=target_dtype,
                )
            else:
                actuator = _build_hard_mask(
                    "K11",
                    sequence_length=current,
                    image_key_positions=active_image,
                    eligible_image_positions=active_image,
                    query_positions=q,
                    device=target_device,
                    dtype=target_dtype,
                )
        elif arm == "K12":
            if self._background_selection is None:
                return not_ready(
                    self._background_selection_error
                    or "zero-overlap background selection is unavailable"
                )
            if not active_image or not active_background:
                return not_ready(
                    "selected background keys are not present in the current scalar prefix"
                )
            actuator = _build_hard_mask(
                "K12",
                sequence_length=current,
                image_key_positions=active_image,
                eligible_image_positions=active_background,
                query_positions=q,
                device=target_device,
                dtype=target_dtype,
            )
        elif arm == "K13":
            if not active_image or not active_competitor:
                return not_ready(
                    "same-class competitor keys are unavailable in the current scalar prefix"
                )
            actuator = build_k13(
                sequence_length=current,
                image_key_positions=active_image,
                same_class_competitor_positions=active_competitor,
                query_positions=q,
                device=target_device,
                dtype=target_dtype,
            )
        elif arm == "K14T":
            if not active_image or not active_b:
                return not_ready(
                    "B-exclusive keys are not present in the current scalar prefix"
                )
            actuator = _build_score_bias(
                "K14T",
                sequence_length=current,
                image_key_positions=active_image,
                selected_positions=active_b,
                query_positions=q,
                query_position=None,
                layer_count=self.layer_count,
                head_count=self.head_count,
                device=target_device,
                target_positions=self.b_exclusive_positions,
                region_positions=self.b_exclusive_positions,
            )
        elif arm == "K14B":
            if self._background_selection is None:
                return not_ready(
                    self._background_selection_error
                    or "zero-overlap background selection is unavailable"
                )
            if not active_image or not active_background:
                return not_ready(
                    "selected background keys are not present in the current scalar prefix"
                )
            actuator = _build_score_bias(
                "K14B",
                sequence_length=current,
                image_key_positions=active_image,
                selected_positions=active_background,
                query_positions=q,
                query_position=None,
                layer_count=self.layer_count,
                head_count=self.head_count,
                device=target_device,
                target_positions=self.b_exclusive_positions,
                region_positions=self._background_selection,
            )
        elif arm == "H00":
            actuator = build_h00(
                sequence_length=current, device=target_device, dtype=target_dtype
            )
        elif arm == "H10":
            if not active_terminal:
                return not_ready("latest terminal-carrier key is unavailable")
            actuator = _build_history_mask(
                "H10",
                sequence_length=current,
                query_positions=q,
                blocked_positions=active_terminal,
                device=target_device,
                dtype=target_dtype,
            )
        elif arm == "H20":
            if not active_row:
                return not_ready("latest completed-row keys are unavailable")
            actuator = _build_history_mask(
                "H20",
                sequence_length=current,
                query_positions=q,
                blocked_positions=active_row,
                device=target_device,
                dtype=target_dtype,
            )
        else:  # pragma: no cover - constructor validates arm IDs
            raise ActuatorError(f"unknown scalar-step actuator arm {arm!r}")
        active_history = (
            active_terminal if arm == "H10" else active_row if arm == "H20" else ()
        )
        return self._with_factory_receipt(
            actuator,
            sequence_length=current,
            query_position=query,
            active_image=active_image,
            active_b=active_b,
            active_a=active_a,
            active_competitor=active_competitor,
            fixed_history=self.latest_terminal_key_positions
            if self.arm_id == "H10"
            else self.latest_row_key_positions
            if self.arm_id == "H20"
            else (),
            active_history=active_history,
            active_background=active_background,
        )

    # Names used by different runners/tests for the same scalar-step seam.
    for_step = build
    build_step = build
    at = build
    for_scalar_step = build

    def __call__(
        self,
        context: Any | None = None,
        *,
        sequence_length: int | None = None,
        query_position: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        input_ids: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del context
        if sequence_length is None:
            if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
                raise ActuatorError(
                    "scalar-step callback requires sequence_length or a two-dimensional input_ids tensor"
                )
            sequence_length = int(input_ids.shape[-1])
        if device is None and isinstance(input_ids, torch.Tensor):
            device = input_ids.device
        if query_position is None:
            query_position = int(sequence_length) - 1
        actuator = self.build(
            sequence_length,
            query_position=query_position,
            device=device,
            dtype=dtype,
        )
        payload = actuator.model_inputs()
        # ``score_bias`` is retained only as an explicit diagnostic handle;
        # K14's attention_mask above is the consumable model input.  The
        # natural runner strips metadata fields before invocation, so these
        # fields cannot be the sole actuation path.
        payload.update(
            {
                "score_bias": actuator.score_bias,
                "score_bias_callback": actuator.score_bias,
                "actuator_id": actuator.arm_id,
                "protocol": self.protocol,
                "receipt": actuator.receipt(),
            }
        )
        return payload

    def callback(self) -> "ScalarStepActuatorFactory":
        return self

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_VERSION}.scalar_step_factory.v1",
            "unit_id": UNIT_ID,
            "protocol": self.protocol,
            "arm_id": self.arm_id,
            "build_count": self._build_count,
            "step_receipts": list(self._step_receipts),
            "fixed_absolute_image_key_positions": list(self.image_key_positions),
            "fixed_absolute_b_exclusive_positions": list(self.b_exclusive_positions),
            "fixed_absolute_a_exclusive_positions": list(self.a_exclusive_positions),
            "fixed_background_selection": list(self._background_selection or ()),
            "fixed_background_selection_sha256": sha256_json(
                list(self._background_selection or ())
            ),
        }


# Friendly aliases for callers that use the longer protocol name.
PerScalarStepActuatorFactory = ScalarStepActuatorFactory
NaturalBoundaryActuatorFactory = ScalarStepActuatorFactory
ScalarStepMaskFactory = ScalarStepActuatorFactory
NaturalBoundaryAttentionMaskFactory = ScalarStepActuatorFactory


def build_scalar_step_factory(arm_id: str, **kwargs: Any) -> ScalarStepActuatorFactory:
    return ScalarStepActuatorFactory(arm_id, **kwargs)


def build_per_scalar_step_factory(
    arm_id: str, **kwargs: Any
) -> ScalarStepActuatorFactory:
    return ScalarStepActuatorFactory(arm_id, **kwargs)


make_scalar_step_factory = build_scalar_step_factory
build_scalar_step_actuator_factory = build_scalar_step_factory
build_natural_boundary_actuator_factory = build_scalar_step_factory
build_attention_mask_factory = build_scalar_step_factory
make_attention_mask_factory = build_scalar_step_factory


@runtime_checkable
class ScoreBiasProtocol(Protocol):
    def __call__(
        self,
        scores: torch.Tensor,
        *,
        layer_idx: int,
        query_positions: Sequence[int] | None = None,
        key_positions: Sequence[int] | None = None,
    ) -> torch.Tensor: ...


class NaturalBoundaryActuatorCallback:
    """Explicit callback contract for a natural autoregressive runner.

    The runner calls the callback once per no-cache forward with the current
    sequence length and query position.  The callback returns real
    ``attention_mask`` input.  For legacy callers a K14 score callback is also
    exposed, but K14's model-facing additive mask is authoritative and remains
    consumable when callback metadata is discarded.  No opener or token is
    inserted by this object.
    """

    protocol = "natural_boundary_attention_actuator_callback.v1"

    def __init__(self, actuator: AttentionActuator) -> None:
        self.actuator = actuator

    def __call__(
        self,
        *,
        sequence_length: int | None = None,
        query_position: int | None = None,
        device: torch.device | str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        if sequence_length is not None and self.actuator.attention_mask is not None:
            current_length = int(self.actuator.attention_mask.shape[-1])
            if int(sequence_length) != current_length:
                raise ActuatorError(
                    "callback sequence_length differs from the prebuilt actuator; rebuild per natural step"
                )
        if query_position is not None:
            declared = set(self.actuator.receipt().get("query_positions", ()))
            if declared and int(query_position) not in declared:
                raise ActuatorError(
                    "callback query_position is outside the actuator scope"
                )
        mask = self.actuator.attention_mask
        if (
            mask is not None
            and device is not None
            and mask.device != torch.device(device)
        ):
            mask = mask.to(device=device)
        return {
            "attention_mask": mask,
            "score_bias": self.actuator.score_bias,
            "score_bias_callback": self.actuator.score_bias,
            "actuator_id": self.actuator.arm_id,
            "receipt": self.actuator.receipt(),
        }

    def receipt(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "arm_id": self.actuator.arm_id,
            "receipt": self.actuator.receipt(),
        }


def make_natural_runner_callback(
    actuator: AttentionActuator | ScalarStepActuatorFactory,
) -> NaturalBoundaryActuatorCallback | ScalarStepActuatorFactory:
    """Return the fixed-actuator or scalar-step callback unchanged in shape."""

    if isinstance(actuator, ScalarStepActuatorFactory):
        return actuator.callback()
    return actuator.callback()


def _find_decoder_layers(model: Any) -> tuple[Any, ...]:
    roots = (
        "model.language_model.layers",
        "model.model.language_model.layers",
        "language_model.layers",
        "layers",
    )
    found: list[Any] = []
    for path in roots:
        owner = model
        try:
            for part in path.split("."):
                owner = getattr(owner, part)
            layers = owner
            if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
                continue
            for layer in layers:
                if all(id(layer) != id(existing) for existing in found):
                    found.append(layer)
            if found:
                break
        except (AttributeError, TypeError):
            continue
    if not found:
        raise ActuatorError("could not resolve language-model decoder layers")
    return tuple(found)


def _extract_attention_mask(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> torch.Tensor | None:
    value = kwargs.get("attention_mask")
    if isinstance(value, torch.Tensor) or value is None and "attention_mask" in kwargs:
        return value
    # Qwen decoder layers receive hidden_states first and attention_mask as the
    # third positional argument only in old versions; inspect the signature for
    # fake models instead of assuming a fixed index.
    if len(args) >= 2 and isinstance(args[1], torch.Tensor) and args[1].ndim == 4:
        return args[1]
    for value in args[1:]:
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            return value
    return None


class LayerMaskConsumptionAttestor(
    AbstractContextManager["LayerMaskConsumptionAttestor"]
):
    """Prove one exact input mask was consumed by every decoder layer."""

    def __init__(
        self,
        model: Any,
        expected_mask: torch.Tensor,
        *,
        layer_indices: Sequence[int] | None = None,
    ) -> None:
        if expected_mask.ndim != 4:
            raise ActuatorError("expected attention mask must have shape [B,H,S,S]")
        self.model = model
        self.expected_mask = expected_mask.detach().clone()
        layers = _find_decoder_layers(model)
        selected = (
            tuple(range(len(layers)))
            if layer_indices is None
            else tuple(int(v) for v in layer_indices)
        )
        if not selected or any(index < 0 or index >= len(layers) for index in selected):
            raise ActuatorError("layer_indices are outside resolved decoder layers")
        self.layers = tuple(layers[index] for index in selected)
        self.layer_indices = selected
        self.expected_hash = sha256_tensor(self.expected_mask)
        self.handles: list[Any] = []
        self.call_counts: dict[int, int] = {index: 0 for index in selected}
        self.observed_hashes: dict[int, list[str]] = {index: [] for index in selected}
        self.observed_shapes: dict[int, list[list[int]]] = {
            index: [] for index in selected
        }
        self.observed_dtypes: dict[int, list[str]] = {index: [] for index in selected}
        self.errors: list[str] = []

    def record(self, layer_idx: int, attention_mask: torch.Tensor | None) -> None:
        index = int(layer_idx)
        if index not in self.call_counts:
            raise ActuatorError(
                f"unregistered decoder layer {index} consumed the actuator mask"
            )
        if not isinstance(attention_mask, torch.Tensor):
            self.errors.append(f"layer {index} received no tensor attention_mask")
            return
        self.call_counts[index] += 1
        self.observed_shapes[index].append(list(attention_mask.shape))
        self.observed_dtypes[index].append(str(attention_mask.dtype))
        observed_hash = sha256_tensor(attention_mask)
        self.observed_hashes[index].append(observed_hash)
        if tuple(attention_mask.shape) != tuple(self.expected_mask.shape):
            self.errors.append(
                f"layer {index} attention_mask shape differs from expected"
            )
        if attention_mask.dtype != self.expected_mask.dtype:
            self.errors.append(
                f"layer {index} attention_mask dtype differs from expected"
            )
        if not torch.equal(
            attention_mask.detach().cpu(), self.expected_mask.detach().cpu()
        ):
            self.errors.append(
                f"layer {index} attention_mask values differ from expected"
            )

    def _hook(self, layer_idx: int) -> Callable[..., Any]:
        def hook(
            _module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]
        ) -> None:
            self.record(layer_idx, _extract_attention_mask(args, kwargs))

        return hook

    def install(self) -> None:
        if self.handles:
            raise ActuatorError("layer mask attestor is already installed")
        for index, layer in zip(self.layer_indices, self.layers, strict=True):
            try:
                handle = layer.register_forward_pre_hook(
                    self._hook(index), with_kwargs=True
                )
            except TypeError:
                # Older torch does not expose ``with_kwargs``; retain a
                # positional fallback for fake layers.
                handle = layer.register_forward_pre_hook(
                    lambda module, args, index=index: self.record(
                        index, _extract_attention_mask(args, {})
                    )
                )
            self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def receipt(self, *, require_one_call: bool = True) -> dict[str, Any]:
        missing = [index for index, count in self.call_counts.items() if count == 0]
        repeated = [index for index, count in self.call_counts.items() if count > 1]
        one_call = all(count == 1 for count in self.call_counts.values())
        all_same = all(
            all(observed == self.expected_hash for observed in hashes)
            for hashes in self.observed_hashes.values()
        )
        result = {
            "schema_version": CONSUMPTION_RECEIPT_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "expected_mask_sha256": self.expected_hash,
            "expected_mask_shape": list(self.expected_mask.shape),
            "expected_mask_dtype": str(self.expected_mask.dtype),
            "layer_indices": list(self.layer_indices),
            "layer_count": len(self.layer_indices),
            "call_counts": {
                str(index): count for index, count in sorted(self.call_counts.items())
            },
            "observed_mask_sha256": {
                str(index): list(hashes)
                for index, hashes in sorted(self.observed_hashes.items())
            },
            "observed_shapes": {
                str(index): list(shapes)
                for index, shapes in sorted(self.observed_shapes.items())
            },
            "observed_dtypes": {
                str(index): list(dtypes)
                for index, dtypes in sorted(self.observed_dtypes.items())
            },
            "missing_layers": missing,
            "repeated_layers": repeated,
            "all_layers_identical": bool(all_same),
            "errors": list(self.errors),
            "passed": bool(
                all_same
                and not self.errors
                and (one_call if require_one_call else not missing)
            ),
        }
        return result

    def __enter__(self) -> "LayerMaskConsumptionAttestor":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()


def attest_all_layer_consumption(
    model: Any,
    expected_mask: torch.Tensor,
    forward: Callable[[], Any],
    *,
    layer_indices: Sequence[int] | None = None,
    strict: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Run one fake/real forward and return output plus layer receipt."""

    attestor = LayerMaskConsumptionAttestor(
        model, expected_mask, layer_indices=layer_indices
    )
    with attestor:
        output = forward()
    receipt = attestor.receipt()
    if strict:
        require_all_layer_consumption(receipt)
    attestation_summary = {
        "schema_version": receipt["schema_version"],
        "passed": bool(receipt["passed"]),
        "all_layers_identical": bool(receipt["all_layers_identical"]),
        "layer_indices": list(receipt["layer_indices"]),
        "missing_layers": list(receipt["missing_layers"]),
        "repeated_layers": list(receipt["repeated_layers"]),
        "errors": list(receipt["errors"]),
        "expected_mask_sha256": receipt["expected_mask_sha256"],
    }
    receipt["all_layer_consumption_attestation"] = dict(attestation_summary)
    receipt["layer_consumption_attestation"] = dict(attestation_summary)
    return output, receipt


def attest_scalar_step_consumption(
    model: Any,
    factory_or_actuator: ScalarStepActuatorFactory | AttentionActuator,
    forward: Callable[[AttentionActuator], Any] | Callable[[], Any],
    *,
    sequence_length: int | None = None,
    query_position: int | None = None,
    layer_indices: Sequence[int] | None = None,
    require_mask: bool = True,
    strict: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Attest exact same-mask consumption for one scalar prefix.

    ``forward`` may accept the built actuator (useful for a production-shaped
    test that wires ``attention_mask`` explicitly) or no arguments (when the
    caller has already closed over it).  A missing mask is a technical failure
    rather than a no-op receipt for every arm except native K00.
    """

    if isinstance(factory_or_actuator, ScalarStepActuatorFactory):
        if sequence_length is None:
            raise ActuatorError(
                "sequence_length is required when attesting a scalar-step factory"
            )
        actuator = factory_or_actuator.build(
            sequence_length, query_position=query_position
        )
    else:
        actuator = factory_or_actuator
    if actuator.attention_mask is None:
        if require_mask:
            raise TechnicalInvalid(
                f"{actuator.arm_id} scalar-step attestation requires a model-facing attention_mask"
            )
        try:
            parameters = inspect.signature(forward).parameters
        except (TypeError, ValueError) as exc:
            raise ActuatorError("cannot inspect scalar-step forward callback") from exc
        output = forward(actuator) if parameters else forward()
        return output, {
            "passed": True,
            "status": "native_no_mask",
            "arm_id": actuator.arm_id,
        }
    attestor = LayerMaskConsumptionAttestor(
        model, actuator.attention_mask, layer_indices=layer_indices
    )
    with attestor:
        try:
            parameters = inspect.signature(forward).parameters
        except (TypeError, ValueError) as exc:
            raise ActuatorError("cannot inspect scalar-step forward callback") from exc
        output = forward(actuator) if parameters else forward()
    receipt = attestor.receipt()
    attestation_summary = {
        "schema_version": receipt["schema_version"],
        "passed": bool(receipt["passed"]),
        "all_layers_identical": bool(receipt["all_layers_identical"]),
        "layer_indices": list(receipt["layer_indices"]),
        "missing_layers": list(receipt["missing_layers"]),
        "repeated_layers": list(receipt["repeated_layers"]),
        "errors": list(receipt["errors"]),
        "expected_mask_sha256": receipt["expected_mask_sha256"],
    }
    receipt["all_layer_consumption_attestation"] = dict(attestation_summary)
    receipt["layer_consumption_attestation"] = dict(attestation_summary)
    receipt.update(
        {
            "arm_id": actuator.arm_id,
            "actuator_receipt": actuator.receipt(),
            "exact_all_layer_consumption_required": True,
            "exact_all_layer_consumption_passed": bool(receipt["passed"]),
        }
    )
    if strict:
        require_all_layer_consumption(receipt)
    return output, receipt


def require_all_layer_consumption(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed when any decoder layer ignored, altered, or missed a mask."""

    if not (
        bool(receipt.get("passed"))
        and bool(receipt.get("all_layers_identical"))
        and not receipt.get("missing_layers")
        and not receipt.get("repeated_layers")
        and not receipt.get("errors")
    ):
        raise TechnicalInvalid(
            "exact all-layer attention-mask consumption failed: "
            f"missing={receipt.get('missing_layers', [])}, "
            f"errors={receipt.get('errors', [])}"
        )
    return dict(receipt)


def attach_layer_consumption_attestation(
    actuator: AttentionActuator,
    consumption_receipt: Mapping[str, Any],
    *,
    require_passed: bool = True,
) -> AttentionActuator:
    """Bind a post-forward all-layer receipt to an actuator immutably.

    The callback cannot inspect the model, so it starts ``unattested``.  A
    caller that wraps the actual forward with ``LayerMaskConsumptionAttestor``
    can attach that evidence here; strict callers fail closed on a missing or
    failed receipt.
    """

    if not isinstance(consumption_receipt, Mapping):
        raise ActuatorError("consumption_receipt must be a mapping")
    if require_passed:
        require_all_layer_consumption(consumption_receipt)
    receipt = actuator.receipt()
    bound = dict(consumption_receipt)
    receipt["layer_consumption_attestation"] = bound
    receipt["all_layer_consumption_attestation"] = dict(bound)
    receipt["layer_consumption_attestation_bound"] = bool(bound.get("passed"))
    return AttentionActuator(
        actuator.arm_id,
        actuator.attention_mask,
        actuator.score_bias,
        receipt,
    )


class Block23AttentionMassDiagnostic:
    """Capture per-head selected-key mass before and after K14 at block 23."""

    def __init__(
        self,
        selected_key_positions: Sequence[int],
        *,
        query_positions: Sequence[int] | None = None,
        block_index: int = 23,
    ) -> None:
        selected = _as_positions(selected_key_positions, label="selected_key_positions")
        if not selected:
            raise ActuatorError("block23 mass diagnostic requires selected keys")
        if int(block_index) != 23:
            raise ActuatorError(
                "the registered mass diagnostic is fixed at language-model block 23"
            )
        self.selected_key_positions = selected
        self.query_positions = (
            None
            if query_positions is None
            else _as_positions(query_positions, label="query_positions")
        )
        self.block_index = 23
        self.records: list[dict[str, Any]] = []

    @staticmethod
    def _probability_mass(
        probabilities: torch.Tensor,
        selected: Sequence[int],
        query_positions: Sequence[int] | None,
        key_positions: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if probabilities.ndim == 3:
            probabilities = probabilities.unsqueeze(0)
        if probabilities.ndim != 4:
            raise ActuatorError(
                "attention probabilities must have shape [B,H,Q,K] or [H,Q,K]"
            )
        if query_positions is None:
            query_positions = tuple(range(int(probabilities.shape[-2])))
        if len(query_positions) != int(probabilities.shape[-2]):
            raise ActuatorError(
                "query_positions length must equal probability query dimension"
            )
        local_keys = (
            tuple(range(int(probabilities.shape[-1])))
            if key_positions is None
            else tuple(int(key) for key in key_positions)
        )
        if len(local_keys) != int(probabilities.shape[-1]):
            raise ActuatorError(
                "key_positions length must equal probability key dimension"
            )
        selected_indices = [
            index
            for index, key in enumerate(local_keys)
            if key in set(int(v) for v in selected)
        ]
        if not selected_indices:
            raise ActuatorError(
                "selected key position is absent from probability key positions"
            )
        keys = torch.tensor(tuple(selected_indices), device=probabilities.device)
        return probabilities.index_select(-1, keys).sum(dim=-1)

    def observe(
        self,
        probabilities_before: torch.Tensor,
        probabilities_after: torch.Tensor,
        *,
        layer_idx: int = 23,
        query_positions: Sequence[int] | None = None,
        key_positions: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        if int(layer_idx) != self.block_index:
            raise ActuatorError("block23 mass diagnostic received a non-block23 layer")
        before = probabilities_before.detach()
        after = probabilities_after.detach()
        if before.shape != after.shape:
            raise ActuatorError(
                "before/after attention probabilities must have identical shape"
            )
        selected_queries = (
            self.query_positions
            if query_positions is None
            else _as_positions(query_positions, label="query_positions")
        )
        before_mass = self._probability_mass(
            before, self.selected_key_positions, selected_queries, key_positions
        )
        after_mass = self._probability_mass(
            after, self.selected_key_positions, selected_queries, key_positions
        )
        delta = after_mass - before_mass
        record = {
            "layer_idx": 23,
            "head_count": int(before.shape[-3]),
            "query_count": int(before.shape[-2]),
            "selected_key_positions": list(self.selected_key_positions),
            "before_mass": before_mass.squeeze(0).cpu().tolist(),
            "after_mass": after_mass.squeeze(0).cpu().tolist(),
            "delta_mass": delta.squeeze(0).cpu().tolist(),
            "before_mass_sha256": sha256_tensor(before_mass),
            "after_mass_sha256": sha256_tensor(after_mass),
        }
        self.records.append(record)
        return record

    def observe_scores(
        self,
        scores_before: torch.Tensor,
        scores_after: torch.Tensor,
        *,
        layer_idx: int = 23,
        query_positions: Sequence[int] | None = None,
        key_positions: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        if scores_before.shape != scores_after.shape:
            raise ActuatorError("before/after score tensors must have identical shape")
        before = torch.softmax(scores_before.to(torch.float32), dim=-1)
        after = torch.softmax(scores_after.to(torch.float32), dim=-1)
        return self.observe(
            before,
            after,
            layer_idx=layer_idx,
            query_positions=query_positions,
            key_positions=key_positions,
        )

    def receipt(self) -> dict[str, Any]:
        def _contains_nonzero(value: Any) -> bool:
            if isinstance(value, (list, tuple)):
                return any(_contains_nonzero(item) for item in value)
            try:
                return abs(float(value)) > 0.0
            except (TypeError, ValueError):
                return False

        return {
            "schema_version": BLOCK23_MASS_RECEIPT_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "block_idx": 23,
            "requirements": {
                "registered_block_idx": 23,
                "per_head_selected_key_mass_before_after": True,
                "selected_key_positions_required": True,
                "applied_bias_receipt_required_for_interpretation": True,
                "mass_shift_receipt_required_for_soft_null": True,
                "output_change_is_not_actuation_receipt": True,
            },
            "selected_key_positions": list(self.selected_key_positions),
            "selected_key_count": len(self.selected_key_positions),
            "record_count": len(self.records),
            "records": list(self.records),
            "passed": bool(
                self.records
                and all(record["head_count"] > 0 for record in self.records)
            ),
            "mass_shift_observed": bool(
                self.records
                and any(
                    _contains_nonzero(record["delta_mass"]) for record in self.records
                )
            ),
        }


class Block23SDPAMassAttestor(AbstractContextManager["Block23SDPAMassAttestor"]):
    """Temporarily wrap Transformers' SDPA registry for language block 23.

    Qwen3-VL's text attention computes post-RoPE Q/K and then dispatches through
    ``transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS['sdpa']``.  This
    attestor wraps that exact registry entry, computes softmax mass on the
    declared absolute keys for the declared query rows, and delegates the
    original function unchanged.  It never switches to eager attention or
    alters the returned output.  The mass path mirrors Transformers'
    ``repeat_kv`` expansion for grouped-query attention, while malformed
    geometry, masks, or registry state remain technical blockers rather than
    fabricated mass evidence.
    """

    protocol = "natural_boundary_block23_sdpa_mass_attestor.v1"

    def __init__(
        self,
        selected_key_positions: Sequence[int],
        *,
        query_positions: Sequence[int] | None = None,
        key_positions: Sequence[int] | None = None,
        absolute_query_positions: Sequence[int] | None = None,
        phase: Literal["native", "biased"] = "native",
        block_index: int = 23,
        registry_name: str = "sdpa",
        mass_tolerance: float = 0.0,
    ) -> None:
        selected = _absolute_positions(
            selected_key_positions,
            label="selected_key_positions",
            allow_empty=False,
        )
        if int(block_index) != 23:
            raise ActuatorError(
                "the registered SDPA mass attestor is fixed at language-model block 23"
            )
        if phase not in {"native", "biased"}:
            raise ActuatorError("SDPA mass attestor phase must be native or biased")
        if not registry_name:
            raise ActuatorError("registry_name must be non-empty")
        if float(mass_tolerance) < 0:
            raise ActuatorError("mass_tolerance must be non-negative")
        self.selected_key_positions = selected
        self.query_positions = (
            None
            if query_positions is None
            else _absolute_positions(
                query_positions, label="query_positions", allow_empty=False
            )
        )
        self.key_positions = (
            None
            if key_positions is None
            else _absolute_positions(
                key_positions, label="key_positions", allow_empty=False
            )
        )
        self.absolute_query_positions = (
            None
            if absolute_query_positions is None
            else _absolute_positions(
                absolute_query_positions,
                label="absolute_query_positions",
                allow_empty=False,
            )
        )
        self.phase = phase
        self.block_index = 23
        self.registry_name = str(registry_name)
        self.mass_tolerance = float(mass_tolerance)
        self.records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.blocker: str | None = None
        self.block23_call_count = 0
        self.registry_restored = False
        self._registry: Any | None = None
        self._original_delegate: Callable[..., Any] | None = None
        self._wrapper: Callable[..., Any] | None = None
        self._entered = False

    @staticmethod
    def _resolve_attention_call(
        args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        names = ("module", "query", "key", "value", "attention_mask")
        values: list[Any] = list(args[:5])
        for index, name in enumerate(names[len(values) :], start=len(values)):
            if name not in kwargs:
                raise ActuatorError(
                    f"SDPA registry call lacks required argument {name!r}"
                )
            values.append(kwargs[name])
        return values[0], values[1], values[2], values[3], values[4]

    @staticmethod
    def _expand_mask(
        attention_mask: torch.Tensor,
        *,
        batch_size: int,
        head_count: int,
        query_count: int,
        key_count: int,
    ) -> torch.Tensor:
        if attention_mask.ndim != 4:
            raise ActuatorError("block23 SDPA requires a 4-D additive attention mask")
        if not attention_mask.dtype.is_floating_point:
            raise ActuatorError(
                "block23 SDPA mass requires a floating-point additive attention mask"
            )
        if attention_mask.shape[0] not in {1, batch_size}:
            raise ActuatorError(
                "attention-mask batch dimension cannot broadcast to Q/K batch"
            )
        if attention_mask.shape[1] not in {1, head_count}:
            raise ActuatorError(
                "attention-mask head dimension cannot broadcast to Q heads"
            )
        if (
            attention_mask.shape[-2] < query_count
            or attention_mask.shape[-1] < key_count
        ):
            raise ActuatorError(
                "attention-mask query/key dimensions are smaller than Q/K"
            )
        mask = attention_mask[..., :query_count, :key_count]
        if torch.isnan(mask).any():
            raise ActuatorError("attention-mask contains NaN values")
        return mask.expand(batch_size, head_count, query_count, key_count)

    @staticmethod
    def _strict_positive_integral(value: Any, *, label: str) -> int:
        """Read a Transformers integer geometry field without coercion.

        ``int(2.5)`` and ``int("2")`` would silently manufacture a valid GQA
        ratio from malformed module state.  The production attention module
        stores a plain Python integer, so accepting only ``Integral`` values
        (and explicitly rejecting booleans) keeps the attestation fail-closed.
        """

        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ActuatorError(f"{label} must be a positive integer")
        result = int(value)
        if result <= 0:
            raise ActuatorError(f"{label} must be a positive integer")
        return result

    @staticmethod
    def _repeat_kv_for_sdpa(
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_heads: int,
        num_key_value_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        """Mirror Transformers ``repeat_kv`` for the mass-only score path.

        The registry delegate receives the original pre-expansion K/V.  This
        local expansion is only for observing the scores that SDPA evaluates;
        its head order is ``kv0, kv0, kv1, kv1, ...`` for a two-way GQA group,
        exactly matching ``repeat_kv`` rather than a broadcast over all heads.
        """

        batch_size, kv_heads, key_count, head_dim = (
            int(value) for value in key.shape
        )
        expected_query_heads = kv_heads * int(num_key_value_groups)
        if expected_query_heads != int(query_heads):
            raise ActuatorError(
                "Q/K head geometry disagrees with num_key_value_groups: "
                f"q_heads={query_heads}, kv_heads={kv_heads}, "
                f"groups={num_key_value_groups}"
            )
        if int(num_key_value_groups) == 1:
            expanded_key = key
            expanded_value = value
        else:
            expanded_key = (
                key[:, :, None, :, :]
                .expand(
                    batch_size,
                    kv_heads,
                    int(num_key_value_groups),
                    key_count,
                    head_dim,
                )
                .reshape(batch_size, expected_query_heads, key_count, head_dim)
            )
            expanded_value = (
                value[:, :, None, :, :]
                .expand(
                    batch_size,
                    kv_heads,
                    int(num_key_value_groups),
                    key_count,
                    head_dim,
                )
                .reshape(batch_size, expected_query_heads, key_count, head_dim)
            )
        head_map = tuple(
            kv_index
            for kv_index in range(kv_heads)
            for _group_index in range(int(num_key_value_groups))
        )
        return expanded_key, expanded_value, head_map

    def _capture(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        scaling: float | None,
    ) -> dict[str, Any]:
        layer_index = self._strict_positive_integral(
            getattr(module, "layer_idx", None), label="layer_idx"
        )
        if layer_index != self.block_index:
            raise ActuatorError(
                f"block23 SDPA wrapper received unexpected layer_idx={layer_index!r}"
            )
        if not all(isinstance(item, torch.Tensor) for item in (query, key, value)):
            raise ActuatorError("block23 SDPA Q/K/V must be tensors")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ActuatorError("block23 SDPA Q/K/V must have shape [B,H,S,D]")
        if query.shape[0] != key.shape[0] or key.shape != value.shape:
            raise ActuatorError(
                "block23 SDPA Q/K/V batch, key length, and value shape disagree"
            )
        if query.shape[-1] != key.shape[-1]:
            raise ActuatorError("block23 SDPA Q/K head dimensions disagree")
        batch_size, query_heads, query_count, head_dim = (int(v) for v in query.shape)
        _batch_key, key_heads, key_count, _ = (int(v) for v in key.shape)
        if min(batch_size, query_heads, query_count, key_heads, key_count, head_dim) <= 0:
            raise ActuatorError("block23 SDPA Q/K/V dimensions must be positive")
        module_groups = self._strict_positive_integral(
            getattr(module, "num_key_value_groups", 1),
            label="num_key_value_groups",
        )
        expanded_key, expanded_value, head_map = self._repeat_kv_for_sdpa(
            key,
            value,
            query_heads=query_heads,
            num_key_value_groups=module_groups,
        )
        if not isinstance(attention_mask, torch.Tensor):
            raise ActuatorError(
                "block23 SDPA mass requires an actual 4-D attention mask"
            )
        mask = self._expand_mask(
            attention_mask,
            batch_size=batch_size,
            head_count=query_heads,
            query_count=query_count,
            key_count=key_count,
        )
        full_query_positions = (
            tuple(range(query_count))
            if self.absolute_query_positions is None
            else self.absolute_query_positions
        )
        full_key_positions = (
            tuple(range(key_count))
            if self.key_positions is None
            else self.key_positions
        )
        if len(full_query_positions) != query_count:
            raise ActuatorError(
                "absolute_query_positions length differs from Q query count"
            )
        if len(full_key_positions) != key_count:
            raise ActuatorError("key_positions length differs from K key count")
        query_index = {
            position: index for index, position in enumerate(full_query_positions)
        }
        key_index = {
            position: index for index, position in enumerate(full_key_positions)
        }
        selected_queries = (
            full_query_positions
            if self.query_positions is None
            else self.query_positions
        )
        if any(position not in query_index for position in selected_queries):
            raise ActuatorError(
                "declared query position is absent from Q absolute positions"
            )
        if any(position not in key_index for position in self.selected_key_positions):
            raise ActuatorError(
                "declared selected key is absent from K absolute positions"
            )
        query_indices = tuple(query_index[position] for position in selected_queries)
        key_indices = tuple(
            key_index[position] for position in self.selected_key_positions
        )
        scale = float(scaling) if scaling is not None else float(head_dim**-0.5)
        with torch.no_grad():
            if not torch.isfinite(query).all() or not torch.isfinite(key).all():
                raise ActuatorError("block23 SDPA Q/K tensors are non-finite")
            scores = (
                torch.matmul(
                    query.detach(), expanded_key.detach().transpose(-2, -1)
                )
                * scale
            )
            scores = scores + mask.to(dtype=scores.dtype, device=scores.device)
            if torch.isnan(scores).any() or torch.isposinf(scores).any():
                raise ActuatorError("block23 SDPA Q/K/mask scores are non-finite")
            probabilities = torch.softmax(scores, dim=-1)
        query_tensor = torch.tensor(
            query_indices, dtype=torch.long, device=probabilities.device
        )
        key_tensor = torch.tensor(
            key_indices, dtype=torch.long, device=probabilities.device
        )
        selected_mass = (
            probabilities.index_select(-2, query_tensor)
            .index_select(-1, key_tensor)
            .sum(-1)
        )
        selected_mass_cpu = selected_mass.detach().cpu()
        selected_mass_per_query_head = [
            {
                "query_position": int(position),
                "head_mass": selected_mass_cpu[:, :, index].tolist(),
            }
            for index, position in enumerate(selected_queries)
        ]
        delegate = self._original_delegate
        delegate_identity = {
            "id": None if delegate is None else id(delegate),
            "module": None if delegate is None else getattr(delegate, "__module__", None),
            "qualname": None
            if delegate is None
            else getattr(delegate, "__qualname__", None),
        }
        return {
            "phase": self.phase,
            "layer_idx": self.block_index,
            "query_count": query_count,
            "key_count": key_count,
            "head_count": query_heads,
            "q_heads": query_heads,
            "kv_heads": key_heads,
            "num_key_value_groups": module_groups,
            "groups": module_groups,
            "gqa_expansion": {
                "mode": "repeat_kv",
                "groups": module_groups,
                "head_map": list(head_map),
                "source_key_shape": list(key.shape),
                "source_value_shape": list(value.shape),
                "expanded_key_shape": list(expanded_key.shape),
                "expanded_value_shape": list(expanded_value.shape),
            },
            "selected_key_positions": list(self.selected_key_positions),
            "query_positions": list(selected_queries),
            "key_positions": list(full_key_positions),
            "selected_key_count": len(self.selected_key_positions),
            "selected_query_count": len(selected_queries),
            "attention_mask_sha256": sha256_tensor(mask),
            "q_sha256": sha256_tensor(query),
            "k_sha256": sha256_tensor(key),
            "expanded_k_sha256": sha256_tensor(expanded_key),
            "selected_mass": selected_mass_cpu.tolist(),
            "selected_mass_shape": list(selected_mass.shape),
            "selected_mass_by_query_head": selected_mass_per_query_head,
            "selected_mass_per_query_head": selected_mass_per_query_head,
            "selected_mass_sha256": sha256_tensor(selected_mass),
            "native_mass": selected_mass_cpu.tolist()
            if self.phase == "native"
            else None,
            "biased_mass": selected_mass_cpu.tolist()
            if self.phase == "biased"
            else None,
            "delegate_identity": delegate_identity,
            "delegate_output_untouched": True,
        }

    def _dispatch(self, *args: Any, **kwargs: Any) -> Any:
        if self._original_delegate is None:
            raise TechnicalInvalid("block23 SDPA delegate is not installed")
        module, query, key, value, attention_mask = self._resolve_attention_call(
            args, kwargs
        )
        if getattr(module, "layer_idx", None) != self.block_index:
            return self._original_delegate(*args, **kwargs)
        self.block23_call_count += 1
        if self.block23_call_count != 1:
            self.errors.append("block23 SDPA was called more than once")
            raise ActuatorError(
                "block23 mass attestor requires exactly one block23 SDPA call"
            )
        try:
            record = self._capture(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=kwargs.get("scaling"),
            )
            self.records.append(record)
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            raise
        # The original registry delegate receives precisely the same args and
        # kwargs; no output hook, eager fallback, or output mutation is used.
        result = self._original_delegate(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            self.errors.append("SDPA delegate returned an unexpected result shape")
            raise ActuatorError(
                "block23 SDPA delegate result must be (output, weights)"
            )
        return result

    def __enter__(self) -> "Block23SDPAMassAttestor":
        if self._entered:
            raise ActuatorError("block23 SDPA mass attestor is already active")
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            registry = ALL_ATTENTION_FUNCTIONS
            original = registry[self.registry_name]
            if not callable(original):
                raise ActuatorError(
                    f"Transformers attention registry entry {self.registry_name!r} is not callable"
                )
            self._registry = registry
            self._original_delegate = original
            self._wrapper = self._dispatch
            registry[self.registry_name] = self._wrapper
            if registry[self.registry_name] is not self._wrapper:
                raise ActuatorError(
                    "Transformers attention registry rejected the SDPA wrapper"
                )
            self._entered = True
            return self
        except Exception as exc:
            self.blocker = f"{type(exc).__name__}: {exc}"
            self.errors.append(self.blocker)
            self._restore_registry()
            raise

    def _restore_registry(self) -> None:
        if self._registry is None or self._original_delegate is None:
            return
        try:
            self._registry[self.registry_name] = self._original_delegate
            self.registry_restored = (
                self._registry[self.registry_name] is self._original_delegate
            )
        except Exception as exc:
            self.errors.append(f"registry_restore_{type(exc).__name__}: {exc}")
            self.registry_restored = False

    def __exit__(self, exc_type: Any, exc: Any, _traceback: Any) -> None:
        if exc is not None:
            self.errors.append(
                f"forward_{exc_type.__name__ if exc_type else 'Exception'}: {exc}"
            )
        self._restore_registry()
        self._entered = False

    def receipt(self) -> dict[str, Any]:
        record = self.records[0] if self.records else None
        passed = bool(
            self.registry_restored
            and self.block23_call_count == 1
            and len(self.records) == 1
            and not self.errors
            and self.blocker is None
        )
        status = (
            "passed"
            if passed
            else "blocked"
            if self.blocker is not None
            else "pending"
            if self._entered
            else "failed"
        )
        return {
            "schema_version": BLOCK23_SDPA_ATTESTOR_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "protocol": self.protocol,
            "phase": self.phase,
            "status": status,
            "block_idx": self.block_index,
            "registry_name": self.registry_name,
            "selected_key_positions": list(self.selected_key_positions),
            "selected_key_count": len(self.selected_key_positions),
            "query_positions": None
            if self.query_positions is None
            else list(self.query_positions),
            "q_heads": None if record is None else record.get("q_heads"),
            "kv_heads": None if record is None else record.get("kv_heads"),
            "num_key_value_groups": (
                None if record is None else record.get("num_key_value_groups")
            ),
            "groups": None if record is None else record.get("groups"),
            "gqa_expansion": (
                None if record is None else record.get("gqa_expansion")
            ),
            "block23_call_count": self.block23_call_count,
            "exactly_one_block23_call": self.block23_call_count == 1,
            "records": list(self.records),
            "delegate_identity": {
                "id": None
                if self._original_delegate is None
                else id(self._original_delegate),
                "module": None
                if self._original_delegate is None
                else getattr(self._original_delegate, "__module__", None),
                "qualname": None
                if self._original_delegate is None
                else getattr(self._original_delegate, "__qualname__", None),
            },
            "delegate_untouched": bool(self.records)
            and all(record.get("delegate_output_untouched") for record in self.records),
            "registry_restored": self.registry_restored,
            "blocker": self.blocker,
            "errors": list(self.errors),
            "passed": passed,
        }


def compare_block23_sdpa_mass_receipts(
    native_receipt: Mapping[str, Any],
    biased_receipt: Mapping[str, Any],
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare matching native/K14 block-23 mass receipts fail-closed."""

    if native_receipt.get("selected_key_positions") != biased_receipt.get(
        "selected_key_positions"
    ) or native_receipt.get("query_positions") != biased_receipt.get("query_positions"):
        raise ActuatorError("native/biased block23 mass scopes do not match")
    if not native_receipt.get("passed") or not biased_receipt.get("passed"):
        raise TechnicalInvalid("native and biased block23 mass receipts must both pass")
    native_records = native_receipt.get("records", [])
    biased_records = biased_receipt.get("records", [])
    if len(native_records) != 1 or len(biased_records) != 1:
        raise TechnicalInvalid(
            "native and biased block23 mass require exactly one record each"
        )
    if (
        native_records[0].get("phase") != "native"
        or biased_records[0].get("phase") != "biased"
    ):
        raise TechnicalInvalid("block23 mass receipt phases must be native and biased")
    native_record = native_records[0]
    biased_record = biased_records[0]
    geometry_fields = ("q_heads", "kv_heads", "num_key_value_groups", "groups")
    if any(field not in native_record or field not in biased_record for field in geometry_fields):
        raise TechnicalInvalid("native/biased block23 receipts lack GQA geometry")
    if any(native_record[field] != biased_record[field] for field in geometry_fields):
        raise TechnicalInvalid("native/biased block23 GQA geometry differs")
    q_heads = native_record["q_heads"]
    kv_heads = native_record["kv_heads"]
    groups = native_record["num_key_value_groups"]
    if (
        isinstance(q_heads, bool)
        or isinstance(kv_heads, bool)
        or isinstance(groups, bool)
        or not all(isinstance(value, Integral) for value in (q_heads, kv_heads, groups))
        or int(q_heads) <= 0
        or int(kv_heads) <= 0
        or int(groups) <= 0
        or int(q_heads) != int(kv_heads) * int(groups)
    ):
        raise TechnicalInvalid("native/biased block23 GQA geometry is malformed")
    native_delegate = native_record.get("delegate_identity") or native_receipt.get(
        "delegate_identity"
    )
    biased_delegate = biased_record.get("delegate_identity") or biased_receipt.get(
        "delegate_identity"
    )
    same_delegate = bool(native_delegate and native_delegate == biased_delegate)
    if not same_delegate:
        raise TechnicalInvalid("native and biased block23 calls used different delegates")
    native_mass = torch.tensor(native_records[0]["selected_mass"], dtype=torch.float64)
    biased_mass = torch.tensor(biased_records[0]["selected_mass"], dtype=torch.float64)
    if native_mass.shape != biased_mass.shape:
        raise TechnicalInvalid("native/biased block23 mass shapes differ")
    if native_mass.ndim != 3 or int(native_mass.shape[1]) != int(q_heads):
        raise TechnicalInvalid("block23 selected mass is not [batch,q_heads,queries]")
    delta = biased_mass - native_mass
    max_abs_delta = float(delta.abs().max().item()) if delta.numel() else 0.0
    tolerance_value = float(tolerance)
    finite_delta = bool(torch.isfinite(delta).all().item()) if delta.numel() else False
    nonzero_shift = bool(
        finite_delta
        and delta.numel()
        and torch.all(delta.abs() > tolerance_value).item()
    )
    return {
        "schema_version": BLOCK23_SDPA_ATTESTOR_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "selected_key_positions": list(native_receipt["selected_key_positions"]),
        "query_positions": list(native_receipt["query_positions"] or ()),
        "q_heads": int(q_heads),
        "kv_heads": int(kv_heads),
        "num_key_value_groups": int(groups),
        "groups": int(groups),
        "selected_mass_shape": list(native_mass.shape),
        "native_mass": native_mass.tolist(),
        "biased_mass": biased_mass.tolist(),
        "delta_mass": delta.tolist(),
        "max_abs_delta_mass": max_abs_delta,
        "min_abs_delta_mass": (
            float(delta.abs().min().item()) if delta.numel() else 0.0
        ),
        "mass_shift_observed": bool(max_abs_delta > tolerance_value),
        "all_query_heads_nonzero_shift": nonzero_shift,
        "all_head_nonzero_shift": nonzero_shift,
        "same_delegate": same_delegate,
        "tolerance": tolerance_value,
        "passed": bool(nonzero_shift and same_delegate),
    }


def run_block23_sdpa_mass_probe(
    forward: Callable[[], Any],
    *,
    selected_key_positions: Sequence[int],
    query_positions: Sequence[int] | None = None,
    key_positions: Sequence[int] | None = None,
    absolute_query_positions: Sequence[int] | None = None,
    phase: Literal["native", "biased"] = "native",
    registry_name: str = "sdpa",
) -> tuple[Any | None, dict[str, Any]]:
    """Run one registry-wrapped forward, returning an explicit blocker receipt."""

    attestor = Block23SDPAMassAttestor(
        selected_key_positions,
        query_positions=query_positions,
        key_positions=key_positions,
        absolute_query_positions=absolute_query_positions,
        phase=phase,
        registry_name=registry_name,
    )
    try:
        with attestor:
            output = forward()
    except Exception as exc:
        attestor.blocker = attestor.blocker or f"{type(exc).__name__}: {exc}"
        receipt = attestor.receipt()
        receipt["status"] = "blocked"
        return None, receipt
    receipt = attestor.receipt()
    receipt["status"] = "passed" if receipt["passed"] else "failed"
    return output, receipt


def apply_score_bias_with_block23_diagnostic(
    scores: torch.Tensor,
    *,
    actuator: FixedDoseScoreBias,
    layer_idx: int,
    query_positions: Sequence[int] | None = None,
    key_positions: Sequence[int] | None = None,
    diagnostic: Block23AttentionMassDiagnostic | None = None,
) -> torch.Tensor:
    """Apply the pre-softmax callback and optionally record block-23 mass."""

    updated = actuator.apply(
        scores,
        layer_idx=layer_idx,
        query_positions=query_positions,
        key_positions=key_positions,
    )
    if diagnostic is not None and int(layer_idx) == 23:
        diagnostic.observe_scores(
            scores,
            updated,
            layer_idx=layer_idx,
            query_positions=query_positions,
            key_positions=key_positions,
        )
    return updated


def run_installed_qwen_cpu_probe() -> dict[str, Any]:
    """Run one tiny installed-Qwen CPU mask pass-through probe.

    This is intentionally a fixed three-layer, five-token model construction;
    it is not a model sweep.  Import/constructor failures are returned with a
    verbatim blocker so a caller can distinguish unavailable dependencies from
    an actuator null.
    """

    try:
        import transformers
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised only on stripped environments
        return {
            "status": "blocked",
            "blocker": f"{type(exc).__name__}: {exc}",
            "stage": "import_installed_qwen",
        }
    try:
        config = Qwen3VLTextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            rope_scaling={"mrope_section": [2, 2, 2], "rope_type": "default"},
        )
        config._attn_implementation = "sdpa"
        model = Qwen3VLTextModel(config).eval()
        input_ids = torch.arange(5).reshape(1, -1)
        position_ids = torch.arange(5).reshape(1, 1, -1).expand(3, 1, -1)
        actuator = build_k01(sequence_length=5, dtype=torch.float32)
        if actuator.attention_mask is None:
            raise ActuatorError("K01 probe did not produce an explicit 4-D mask")
        output, consumption = attest_all_layer_consumption(
            model,
            actuator.attention_mask,
            lambda: model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=actuator.attention_mask,
                use_cache=False,
            ),
        )
        # The pre-GPU contract also requires a real block-23 SDPA native versus
        # K14 mass probe.  Keep this separate from the three-layer pass-through
        # model above: a shorter smoke cannot attest block 23.
        block23_receipt: dict[str, Any]
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1723)
                block23_config = Qwen3VLTextConfig(
                    vocab_size=32,
                    hidden_size=32,
                    intermediate_size=32,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    num_key_value_heads=8,
                    head_dim=2,
                    max_position_embeddings=32,
                    rope_scaling={
                        "mrope_section": [2, 2, 2],
                        "rope_type": "default",
                    },
                )
                block23_config._attn_implementation = "sdpa"
                block23_model = Qwen3VLTextModel(block23_config).eval()
                block23_ids = torch.arange(4).reshape(1, -1)
                block23_positions = torch.arange(4).reshape(1, 1, -1).expand(3, 1, -1)
                native_mask = build_k01(
                    sequence_length=4, dtype=torch.float32
                ).attention_mask
                biased_mask = build_k14t(
                    sequence_length=4,
                    image_key_positions=(1, 2),
                    b_exclusive_positions=(2,),
                    query_position=3,
                    layer_count=24,
                    head_count=16,
                ).attention_mask
                if native_mask is None or biased_mask is None:
                    raise ActuatorError("block23 probe masks were not materialized")
                baseline = block23_model(
                    input_ids=block23_ids,
                    position_ids=block23_positions,
                    attention_mask=native_mask,
                    use_cache=False,
                )
                native_attestor = Block23SDPAMassAttestor(
                    (2,), query_positions=(3,), phase="native"
                )
                with native_attestor:
                    native_output = block23_model(
                        input_ids=block23_ids,
                        position_ids=block23_positions,
                        attention_mask=native_mask,
                        use_cache=False,
                    )
                biased_attestor = Block23SDPAMassAttestor(
                    (2,), query_positions=(3,), phase="biased"
                )
                with biased_attestor:
                    biased_output = block23_model(
                        input_ids=block23_ids,
                        position_ids=block23_positions,
                        attention_mask=biased_mask,
                        use_cache=False,
                    )
                native_receipt = native_attestor.receipt()
                biased_receipt = biased_attestor.receipt()
                comparison = compare_block23_sdpa_mass_receipts(
                    native_receipt, biased_receipt
                )
                native_hidden = native_output.last_hidden_state
                baseline_hidden = baseline.last_hidden_state
                biased_hidden = biased_output.last_hidden_state
                block23_passed = bool(
                    native_receipt["passed"]
                    and biased_receipt["passed"]
                    and native_receipt["registry_restored"]
                    and biased_receipt["registry_restored"]
                    and comparison["passed"]
                    and torch.allclose(native_hidden, baseline_hidden)
                    and not torch.allclose(native_hidden, biased_hidden)
                )
                block23_receipt = {
                    "status": "passed" if block23_passed else "failed",
                    "native": native_receipt,
                    "biased": biased_receipt,
                    "comparison": comparison,
                    "q_heads": comparison.get("q_heads"),
                    "kv_heads": comparison.get("kv_heads"),
                    "num_key_value_groups": comparison.get("num_key_value_groups"),
                    "groups": comparison.get("groups"),
                    "all_query_heads_nonzero_shift": comparison.get(
                        "all_query_heads_nonzero_shift", False
                    ),
                    "same_delegate": comparison.get("same_delegate", False),
                    "native_delegate_output_parity": bool(
                        torch.allclose(native_hidden, baseline_hidden)
                    ),
                    "biased_output_changed": bool(
                        not torch.allclose(native_hidden, biased_hidden)
                    ),
                    "registry_restored": bool(
                        native_receipt["registry_restored"]
                        and biased_receipt["registry_restored"]
                    ),
                }
        except Exception as exc:
            block23_passed = False
            block23_receipt = {
                "status": "blocked",
                "blocker": f"{type(exc).__name__}: {exc}",
            }
        overall_passed = bool(consumption["passed"] and block23_passed)
        return {
            "status": "passed" if overall_passed else "failed",
            "transformers_version": str(transformers.__version__),
            "torch_version": str(torch.__version__),
            "qwen_model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "mask_dtype": str(actuator.attention_mask.dtype),
            "mask_shape": list(actuator.attention_mask.shape),
            "mask_sha256": sha256_tensor(actuator.attention_mask),
            "output_shape": list(output.last_hidden_state.shape),
            "layer_consumption": consumption,
            "float_additive_4d_mask_passthrough": bool(
                consumption["passed"]
                and actuator.attention_mask.dtype.is_floating_point
            ),
            "all_layer_consumption": bool(consumption["passed"]),
            "float_4d_pass_through": bool(
                consumption["passed"] and actuator.attention_mask.dtype == torch.float32
            ),
            "silent_coercion": False,
            "ignored_kwargs": False,
            "block23_sdpa_mass_attestation": bool(block23_passed),
            "block23_sdpa_mass_receipt": block23_receipt,
        }
    except Exception as exc:  # pragma: no cover - exact blocker is environment-specific
        return {
            "status": "blocked",
            "blocker": f"{type(exc).__name__}: {exc}",
            "stage": "instantiate_or_forward_installed_qwen",
        }


# Names used by lightweight runners and hidden CPU contract tests.  Keeping
# aliases local avoids forcing every caller to know whether an arm is called a
# mask, a controller, or an actuator.
build_static_actuator = build_k_arm
build_history_actuator = build_h_arm
MaskConsumptionAttestor = LayerMaskConsumptionAttestor
Block23MassDiagnostic = Block23AttentionMassDiagnostic


def build_k_mask(arm_id: str, **kwargs: Any) -> torch.Tensor | None:
    """Return only the model-facing mask for a hard/static arm."""

    actuator = build_k_arm(arm_id, **kwargs)
    return actuator.attention_mask


def build_h_mask(arm_id: str, **kwargs: Any) -> torch.Tensor | None:
    """Return only the model-facing history mask."""

    actuator = build_h_arm(arm_id, **kwargs)
    return actuator.attention_mask


def build_k14_bias(
    arm_id: Literal["K14T", "K14B"], **kwargs: Any
) -> FixedDoseScoreBias | None:
    """Return the pre-softmax callback for a K14 arm."""

    actuator = build_k14(arm_id, **kwargs)
    return actuator.score_bias


__all__ = [
    "ActuatorError",
    "TechnicalInvalid",
    "AttentionActuator",
    "FixedDoseScoreBias",
    "NaturalBoundaryActuatorCallback",
    "LayerMaskConsumptionAttestor",
    "MaskConsumptionAttestor",
    "Block23AttentionMassDiagnostic",
    "Block23MassDiagnostic",
    "Block23SDPAMassAttestor",
    "compare_block23_sdpa_mass_receipts",
    "run_block23_sdpa_mass_probe",
    "ScoreBiasProtocol",
    "build_k00",
    "build_k01",
    "build_k10",
    "build_k11",
    "build_k12",
    "build_k13",
    "build_k14",
    "build_k14t",
    "build_k14b",
    "build_k_arm",
    "build_k_mask",
    "build_k14_bias",
    "build_h00",
    "build_h10",
    "build_h20",
    "build_h_arm",
    "build_h_mask",
    "compose_k10_h20",
    "ScalarStepActuatorFactory",
    "PerScalarStepActuatorFactory",
    "NaturalBoundaryActuatorFactory",
    "ScalarStepMaskFactory",
    "NaturalBoundaryAttentionMaskFactory",
    "build_scalar_step_factory",
    "build_per_scalar_step_factory",
    "make_scalar_step_factory",
    "build_scalar_step_actuator_factory",
    "build_natural_boundary_actuator_factory",
    "build_attention_mask_factory",
    "make_attention_mask_factory",
    "build_static_actuator",
    "build_history_actuator",
    "select_zero_overlap_background_positions",
    "make_natural_runner_callback",
    "attest_all_layer_consumption",
    "attest_scalar_step_consumption",
    "require_all_layer_consumption",
    "attach_layer_consumption_attestation",
    "construction_consumption_placeholder_kind",
    "CONSTRUCTION_CONSUMPTION_PLACEHOLDER_BASE_KEYS",
    "apply_score_bias_with_block23_diagnostic",
    "run_installed_qwen_cpu_probe",
    "sha256_json",
    "sha256_tensor",
    "CROSSOVER_UNIT_ID",
]

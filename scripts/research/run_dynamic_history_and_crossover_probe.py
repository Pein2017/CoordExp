#!/usr/bin/env python3
"""Mechanically bounded P2/P3 dynamic-history crossover probe.

The module is deliberately experiment-local.  It owns only the exact-prefix
contract, the post-block-23 residual hook, and outcome bookkeeping needed by
the frozen D/K/Y matrix.  A caller supplies a *natural* raw-token row
generator; this file never tokenizes text, swaps a donor cache, or changes
model weights.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Literal

import torch


UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
BLOCK_INDEX = 23
NOOP_TOLERANCE = 1e-4
COMMIT_TOKEN_ID = 151669
BOX_END_TOKEN_ID = 151649
IM_END_TOKEN_ID = 151645
COORDINATE_TOKEN_MIN = 151670
COORDINATE_TOKEN_MAX = 152669

DYNAMIC_ARM_IDS = ("D00", "D01", "D10", "D11", "D12", "D20", "D21")
P3_CELL_IDS = ("Y00", "Y10", "Y01", "Y11")


class TechnicalInvalid(ValueError):
    """Raised when a mechanical contract cannot support interpretation."""


class ForwardMechanicalReceipt(dict[str, Any]):
    """Live receipt issued only by an in-module model-forward helper."""


def _strict_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise TechnicalInvalid(f"{label} must be an actual JSON/Python boolean")
    return value


def _issue_forward_receipt(
    receipt: Mapping[str, Any],
    *,
    execution_adapter: Literal["production", "test_adapter"],
) -> ForwardMechanicalReceipt:
    if execution_adapter not in {"production", "test_adapter"}:
        raise TechnicalInvalid("execution_adapter must be production or test_adapter")
    issued = ForwardMechanicalReceipt(receipt)
    issued["execution_adapter"] = execution_adapter
    issued["real_forward"] = execution_adapter == "production"
    issued["test_adapter"] = execution_adapter == "test_adapter"
    validate_mechanical_receipt(issued)
    return issued


def _ints(values: Sequence[int] | torch.Tensor, *, label: str, allow_empty: bool = True) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim == 0:
            values = values.reshape(1)
        values = values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TechnicalInvalid(f"{label} must be a raw integer-token sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TechnicalInvalid(f"{label} contains a non-negative integer token requirement")
        result.append(int(value))
    if not result and not allow_empty:
        raise TechnicalInvalid(f"{label} must not be empty")
    return tuple(result)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def hash_token_ids(token_ids: Sequence[int] | torch.Tensor) -> str:
    """Hash exact token IDs; no decode/re-tokenize path exists here."""

    return _json_hash(list(_ints(token_ids, label="token_ids")))


def hash_position_ids(position_ids: torch.Tensor | Sequence[Any]) -> str:
    if isinstance(position_ids, torch.Tensor):
        value = position_ids.detach().cpu().contiguous()
        return _json_hash({"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()})
    return _json_hash(position_ids)


def _validate_qwen_position_ids(position_ids: torch.Tensor, *, sequence_length: int | None = None) -> None:
    """Require Qwen's native [3,B,S] shape (or an attested [4,B,S] variant)."""

    if not isinstance(position_ids, torch.Tensor) or position_ids.ndim != 3:
        raise TechnicalInvalid("Qwen position_ids must have rank-3 [3,B,S] or [4,B,S] shape")
    if int(position_ids.shape[0]) not in {3, 4}:
        raise TechnicalInvalid("Qwen position_ids rank-3 first axis must be 3 or an attested 4")
    if int(position_ids.shape[1]) <= 0 or int(position_ids.shape[2]) <= 0:
        raise TechnicalInvalid("Qwen position_ids batch and sequence axes must be positive")
    if sequence_length is not None and int(position_ids.shape[2]) != int(sequence_length):
        raise TechnicalInvalid("Qwen position_ids sequence axis differs from exact prefix length")
    if position_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TechnicalInvalid("Qwen position_ids must use an integer dtype")


def compute_mrope_hash(
    position_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | Sequence[Any] | None = None,
    rope_deltas: torch.Tensor | Sequence[Any] | None = None,
) -> str:
    """Compute a local, deterministic M-RoPE identity from native inputs.

    The caller's serialized hash is never used as the source of truth.  The
    hash covers the exact native position tensor and any grid/rope-delta
    metadata that affects its derivation.
    """

    _validate_qwen_position_ids(position_ids)

    def canonical(value: torch.Tensor | Sequence[Any] | None) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "values": tensor.tolist()}
        return value

    return _json_hash(
        {
            "position_ids": canonical(position_ids),
            "image_grid_thw": canonical(image_grid_thw),
            "rope_deltas": canonical(rope_deltas),
        }
    )


def hash_mask(mask: torch.Tensor | None) -> str:
    """Hash an explicit 4-D boolean attention mask, or the native sentinel."""

    if mask is None:
        return "none"
    if not isinstance(mask, torch.Tensor) or mask.ndim != 4 or mask.dtype != torch.bool:
        raise TechnicalInvalid("attention mask hash requires a boolean [B,H,S,S] tensor")
    tensor = mask.detach().cpu().contiguous()
    return _json_hash({"dtype": str(tensor.dtype), "shape": list(tensor.shape), "values": tensor.tolist()})


@dataclass(frozen=True)
class CarrierContract:
    """Explicit native carrier mapping for closed/plain versus commit/A3 rows."""

    kind: Literal["closed", "commit"]
    carrier_token_id: int
    closure_token_id: int = BOX_END_TOKEN_ID
    wrapper: str = ""
    label: str = ""

    def validate(self) -> None:
        if self.kind not in {"closed", "commit"}:
            raise TechnicalInvalid("carrier kind must be explicit: closed or commit")
        if self.carrier_token_id < 0 or self.closure_token_id < 0:
            raise TechnicalInvalid("carrier token IDs must be non-negative")
        if not self.wrapper:
            raise TechnicalInvalid("carrier contract requires the native wrapper name")
        if self.kind == "commit" and self.carrier_token_id != COMMIT_TOKEN_ID:
            raise TechnicalInvalid("commit contract must map to the declared commit token")
        if self.kind == "closed" and self.carrier_token_id != self.closure_token_id:
            raise TechnicalInvalid("closed contract must map the terminal carrier to closure token")


def build_carrier_contract(
    kind: Literal["closed", "commit"],
    *,
    carrier_token_id: int,
    closure_token_id: int = BOX_END_TOKEN_ID,
    wrapper: str,
    label: str = "",
) -> CarrierContract:
    """Build a carrier only from an explicit caller-supplied mapping."""

    contract = CarrierContract(kind, int(carrier_token_id), int(closure_token_id), str(wrapper), str(label))
    contract.validate()
    return contract


@dataclass(frozen=True)
class ExactPrefixContract:
    """Identity required for every full-prefix model call."""

    prefix_token_ids: tuple[int, ...]
    position_ids: torch.Tensor
    mrope_hash: str
    wrapper: str
    image_grid_thw: torch.Tensor | Sequence[Any] | None = None
    rope_deltas: torch.Tensor | Sequence[Any] | None = None
    natural: bool = True
    retokenized: bool = False
    forced: bool = False
    teacher_forced: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefix_token_ids", _ints(self.prefix_token_ids, label="prefix_token_ids", allow_empty=False))
        if not isinstance(self.position_ids, torch.Tensor):
            raise TechnicalInvalid("position_ids must be an explicit tensor")
        _validate_qwen_position_ids(self.position_ids, sequence_length=len(self.prefix_token_ids))
        computed_mrope_hash = compute_mrope_hash(
            self.position_ids,
            image_grid_thw=self.image_grid_thw,
            rope_deltas=self.rope_deltas,
        )
        if not self.mrope_hash or str(self.mrope_hash) != computed_mrope_hash:
            raise TechnicalInvalid("mrope_hash must equal the locally computed native M-RoPE identity")
        object.__setattr__(self, "mrope_hash", computed_mrope_hash)
        if not self.wrapper:
            raise TechnicalInvalid("native wrapper must be recorded")
        if self.retokenized or self.forced or self.teacher_forced or not self.natural:
            raise TechnicalInvalid("exact natural prefix contract rejects forced/teacher-forced/re-tokenized input")

    @property
    def prefix_hash(self) -> str:
        return hash_token_ids(self.prefix_token_ids)


def build_exact_prefix_contract(
    prefix_token_ids: Sequence[int],
    position_ids: torch.Tensor,
    *,
    wrapper: str,
    image_grid_thw: torch.Tensor | Sequence[Any] | None = None,
    rope_deltas: torch.Tensor | Sequence[Any] | None = None,
) -> ExactPrefixContract:
    """Construct a contract from local native tensors, never a caller hash."""

    computed = compute_mrope_hash(position_ids, image_grid_thw=image_grid_thw, rope_deltas=rope_deltas)
    return ExactPrefixContract(
        tuple(_ints(prefix_token_ids, label="prefix_token_ids", allow_empty=False)),
        position_ids,
        computed,
        wrapper,
        image_grid_thw,
        rope_deltas,
    )


def validate_exact_prefix(
    contract: ExactPrefixContract,
    *,
    input_ids: torch.Tensor | Sequence[int],
    position_ids: torch.Tensor,
    mrope_hash: str | None,
    use_cache: bool = False,
    extra_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed before any expensive model call."""

    actual = _ints(input_ids, label="input_ids")
    if actual != contract.prefix_token_ids:
        raise TechnicalInvalid("exact prefix token IDs differ")
    if not isinstance(position_ids, torch.Tensor):
        raise TechnicalInvalid("explicit position IDs are required")
    _validate_qwen_position_ids(position_ids, sequence_length=len(actual))
    if not torch.equal(position_ids.detach().cpu(), contract.position_ids.detach().cpu()):
        raise TechnicalInvalid("explicit position_ids differ from the exact prefix contract")
    if use_cache:
        raise TechnicalInvalid("natural probe requires full-prefix recompute; donor/recipient cache is forbidden")
    extras = dict(extra_inputs or {})
    computed_mrope_hash = compute_mrope_hash(
        position_ids,
        image_grid_thw=extras.get("image_grid_thw", contract.image_grid_thw),
        rope_deltas=extras.get("rope_deltas", contract.rope_deltas),
    )
    if computed_mrope_hash != contract.mrope_hash:
        raise TechnicalInvalid("locally computed M-RoPE identity differs from exact prefix contract")
    if mrope_hash is not None and str(mrope_hash) != computed_mrope_hash:
        raise TechnicalInvalid("caller M-RoPE string differs from the locally computed identity")
    forbidden = {
        "retokenize",
        "retokenized",
        "donor_cache",
        "past_key_values",
        "key_value_cache",
        "forced",
        "teacher_forced",
    }
    present = sorted(key for key in forbidden if extras.get(key) is not None and extras.get(key) is not False)
    if present:
        raise TechnicalInvalid(f"forbidden prefix/model inputs present: {present}")
    return {
        "passed": True,
        "prefix_token_count": len(actual),
        "prefix_token_ids_sha256": contract.prefix_hash,
        "position_ids_sha256": hash_position_ids(position_ids),
        "mrope_hash": computed_mrope_hash,
        "wrapper": contract.wrapper,
        "use_cache": False,
    }


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TechnicalInvalid("decoder block output does not expose a first tensor")


def _replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TechnicalInvalid("decoder block output cannot be replaced safely")


def resolve_decoder_layer(model: Any, layer_idx: int = BLOCK_INDEX) -> tuple[Any, dict[str, Any]]:
    """Resolve one decoder layer, accepting only an explicit Qwen-like marker."""

    wanted = int(layer_idx)
    if wanted < 0:
        raise TechnicalInvalid("layer index must be non-negative")
    candidates: list[tuple[str, Any]] = []
    for root_name in ("model.language_model", "model.model.language_model", "language_model", "model"):
        owner = model
        try:
            for part in root_name.split("."):
                owner = getattr(owner, part)
            layers = getattr(owner, "layers")
        except (AttributeError, TypeError):
            continue
        if isinstance(layers, (torch.nn.ModuleList, list, tuple)) and wanted < len(layers):
            candidates.append((f"{root_name}.layers[{wanted}]", layers[wanted]))
    if not candidates and hasattr(model, "layers"):
        layers = getattr(model, "layers")
        if wanted < len(layers):
            candidates.append((f"layers[{wanted}]", layers[wanted]))
    by_identity: dict[int, list[tuple[str, Any]]] = {}
    for name, module in candidates:
        by_identity.setdefault(id(module), []).append((name, module))
    if len(by_identity) != 1:
        raise TechnicalInvalid(f"expected exactly one decoder layer {wanted}, found {len(by_identity)}")
    aliases = next(iter(by_identity.values()))
    module = aliases[0][1]
    class_name = module.__class__.__name__
    if "Qwen3VLTextDecoderLayer" not in class_name and not getattr(module, "_coordexp_decoder_layer", False):
        raise TechnicalInvalid(f"layer {aliases[0][0]} is not an opted-in Qwen decoder block: {class_name}")
    return module, {
        "layer_idx": wanted,
        "module_path": aliases[0][0],
        "module_alias_paths": [item[0] for item in aliases],
        "module_class": class_name,
        "replacement_seam": "returned_layer_output_after_full_block_forward",
    }


def resolve_block23(model: Any) -> tuple[Any, dict[str, Any]]:
    """Resolve the only dynamic replacement layer used by P2/P3."""

    return resolve_decoder_layer(model, BLOCK_INDEX)


class ResidualSpanCapture:
    """Capture detached returned block output rows at exact absolute positions."""

    def __init__(self, module: Any, positions: Sequence[int]) -> None:
        self.module = module
        self.positions = tuple(int(value) for value in positions)
        if not self.positions or len(set(self.positions)) != len(self.positions):
            raise TechnicalInvalid("capture positions must be non-empty and unique")
        self.handle: Any = None
        self.call_count = 0
        self.state: torch.Tensor | None = None
        self.tensor_shape: tuple[int, ...] | None = None

    def _hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        tensor = _first_tensor(output)
        if tensor.ndim < 3 or any(pos < 0 or pos >= tensor.shape[1] for pos in self.positions):
            raise TechnicalInvalid("capture position outside returned block output")
        self.call_count += 1
        if self.call_count != 1:
            raise TechnicalInvalid("capture must observe exactly one full-prefix forward")
        self.tensor_shape = tuple(tensor.shape)
        self.state = tensor[0, list(self.positions), :].detach().clone()
        return output

    def install(self) -> None:
        if self.handle is not None:
            raise TechnicalInvalid("capture hook already installed")
        self.handle = self.module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self) -> "ResidualSpanCapture":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()
        if self.call_count != 1 or self.state is None:
            raise TechnicalInvalid("capture hook did not fire exactly once")


class ResidualSpanReplacement:
    """Replace one or many post-block rows, one-shot or persistently."""

    def __init__(
        self,
        module: Any,
        positions: Sequence[int],
        replacement: torch.Tensor,
        *,
        persistent: bool = False,
        tolerance: float = NOOP_TOLERANCE,
    ) -> None:
        self.module = module
        self.positions = tuple(int(value) for value in positions)
        if not self.positions or len(set(self.positions)) != len(self.positions):
            raise TechnicalInvalid("replacement positions must be non-empty and unique")
        if not isinstance(replacement, torch.Tensor) or replacement.ndim != 2:
            raise TechnicalInvalid("replacement must have [span,hidden] tensor shape")
        if replacement.shape[0] != len(self.positions):
            raise TechnicalInvalid("replacement span length differs from destination span")
        self.replacement = replacement.detach().clone()
        self.persistent = bool(persistent)
        self.tolerance = float(tolerance)
        self.handle: Any = None
        self.call_count = 0
        self.applied_count = 0
        self.last_target_max_abs_delta = 0.0
        self.max_target_max_abs_delta = 0.0
        self.max_non_target_abs_delta = 0.0
        self.tensor_shape: tuple[int, ...] | None = None

    def _hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        tensor = _first_tensor(output)
        if tensor.ndim < 3 or any(pos < 0 or pos >= tensor.shape[1] for pos in self.positions):
            raise TechnicalInvalid("replacement position outside returned block output")
        self.call_count += 1
        if not self.persistent and self.applied_count >= 1:
            raise TechnicalInvalid("one-shot replacement fired more than once")
        if self.replacement.shape[1] != tensor.shape[-1]:
            raise TechnicalInvalid("replacement hidden width differs from block output")
        updated = tensor.clone()
        replacement = self.replacement.to(device=tensor.device, dtype=tensor.dtype)
        destination = torch.as_tensor(self.positions, dtype=torch.long, device=tensor.device)
        before = tensor[0, destination, :]
        updated[0, destination, :] = replacement
        target_delta = (updated[0, destination, :] - before).detach().abs().max().item()
        delta = (updated - tensor).detach().abs()
        off_target = delta.clone()
        off_target[0, destination, :] = 0
        off_delta = off_target.max().item() if off_target.numel() else 0.0
        self.call_count = int(self.call_count)
        self.applied_count += 1
        self.last_target_max_abs_delta = float(target_delta)
        self.max_target_max_abs_delta = max(self.max_target_max_abs_delta, float(target_delta))
        self.max_non_target_abs_delta = max(self.max_non_target_abs_delta, float(off_delta))
        self.tensor_shape = tuple(tensor.shape)
        if not self.persistent:
            self.remove()
        return _replace_first_tensor(output, updated)

    def install(self) -> None:
        if self.handle is not None:
            raise TechnicalInvalid("replacement hook already installed")
        self.handle = self.module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self) -> "ResidualSpanReplacement":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()
        if self.applied_count < 1:
            raise TechnicalInvalid("replacement hook did not fire")
        if self.max_non_target_abs_delta > self.tolerance:
            raise TechnicalInvalid("replacement changed a non-target position")

    def receipt(self) -> dict[str, Any]:
        return {
            "positions": list(self.positions),
            "persistent": self.persistent,
            "call_count": self.call_count,
            "applied_count": self.applied_count,
            "hook_call_count": self.call_count,
            "hook_applied_count": self.applied_count,
            "hook_count": self.call_count,
            "hook_removed": self.handle is None,
            "target_max_abs_delta": self.max_target_max_abs_delta,
            "non_target_max_abs_delta": self.max_non_target_abs_delta,
            "hook_clean": self.handle is None,
            "tensor_shape": list(self.tensor_shape or ()),
        }


def norm_matched_mean(target: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    """Return one background mean vector with target-row average norm."""

    if target.ndim != 2 or background.ndim != 2 or target.shape[1] != background.shape[1]:
        raise TechnicalInvalid("target/background states must be [span,hidden] with equal width")
    mean = background.float().mean(dim=0)
    target_norm = target.float().norm(dim=-1).mean()
    mean_norm = mean.norm()
    if not torch.isfinite(mean_norm) or float(mean_norm) == 0.0:
        raise TechnicalInvalid("norm-matched mean is not finite/non-zero")
    return (mean * (target_norm / mean_norm)).to(dtype=target.dtype)


def norm_matched_span(target: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    replacement = norm_matched_mean(target, background)
    return replacement.unsqueeze(0).expand(target.shape[0], -1).clone()


def row_carrier_position(
    row_token_ids: Sequence[int],
    *,
    prefix_width: int,
    carrier: CarrierContract,
) -> int:
    carrier.validate()
    row = _ints(row_token_ids, label="row_token_ids", allow_empty=False)
    matches = [index for index, token_id in enumerate(row) if token_id == carrier.carrier_token_id]
    if len(matches) != 1:
        raise TechnicalInvalid("row must contain exactly one explicit carrier token")
    return int(prefix_width) + matches[0]


def last_coordinate_position(row_token_ids: Sequence[int], *, prefix_width: int) -> int:
    row = _ints(row_token_ids, label="row_token_ids", allow_empty=False)
    matches = [index for index, token_id in enumerate(row) if COORDINATE_TOKEN_MIN <= token_id <= COORDINATE_TOKEN_MAX]
    if not matches:
        raise TechnicalInvalid("row has no coordinate carrier")
    return int(prefix_width) + matches[-1]


def row_span_positions(row_token_ids: Sequence[int], *, prefix_width: int) -> tuple[int, ...]:
    row = _ints(row_token_ids, label="row_token_ids", allow_empty=False)
    return tuple(int(prefix_width) + index for index in range(len(row)))


@dataclass(frozen=True)
class DynamicArm:
    arm_id: str
    status: Literal["ready", "not_applicable"]
    positions: tuple[int, ...] = ()
    reason: str = ""
    scope: Literal["none", "single", "whole_row"] = "none"
    carrier: CarrierContract | None = None
    persistence: Literal["one_shot", "persistent"] = "one_shot"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _not_applicable(arm_id: str, reason: str, **metadata: Any) -> DynamicArm:
    return DynamicArm(arm_id, "not_applicable", reason=reason, metadata=metadata)


def build_dynamic_arm(
    arm_id: str,
    *,
    prefix_width: int,
    latest_row_token_ids: Sequence[int] | None = None,
    earlier_row_token_ids: Sequence[int] | None = None,
    carrier: CarrierContract | None = None,
    donor_row_token_ids: Sequence[int] | None = None,
    same_parent: bool | None = None,
    same_class: bool | None = None,
    replacement_persistence: Literal["one_shot", "persistent"] = "one_shot",
) -> DynamicArm:
    """Construct D00..D21 with D21 fail-closed eligibility."""

    arm = str(arm_id)
    if arm not in DYNAMIC_ARM_IDS:
        raise ValueError(f"unknown dynamic arm {arm!r}")
    if replacement_persistence not in {"one_shot", "persistent"}:
        raise ValueError("replacement_persistence must be one_shot or persistent")
    if arm == "D00":
        return DynamicArm(arm, "ready", scope="none", persistence=replacement_persistence)
    latest = None if latest_row_token_ids is None else _ints(latest_row_token_ids, label="latest row", allow_empty=False)
    earlier = None if earlier_row_token_ids is None else _ints(earlier_row_token_ids, label="earlier row", allow_empty=False)
    if arm in {"D01", "D10", "D12"} and carrier is None:
        return _not_applicable(arm, "explicit closed/commit carrier contract is required")
    if arm in {"D01", "D10", "D11", "D12", "D20", "D21"} and latest is None:
        return _not_applicable(arm, "latest complete row token IDs are required")
    if arm == "D12" and earlier is None:
        return _not_applicable(arm, "earlier equal-length row token IDs are required")
    if arm == "D01":
        assert latest is not None and carrier is not None
        return DynamicArm(arm, "ready", (row_carrier_position(latest, prefix_width=prefix_width, carrier=carrier),), scope="single", carrier=carrier, persistence=replacement_persistence)
    if arm == "D10":
        assert latest is not None and carrier is not None
        return DynamicArm(arm, "ready", (row_carrier_position(latest, prefix_width=prefix_width, carrier=carrier),), scope="single", carrier=carrier, persistence=replacement_persistence)
    if arm == "D11":
        assert latest is not None
        return DynamicArm(arm, "ready", (last_coordinate_position(latest, prefix_width=prefix_width),), scope="single", persistence=replacement_persistence)
    if arm == "D12":
        assert earlier is not None and carrier is not None
        if latest is not None and len(earlier) != len(latest):
            return _not_applicable(arm, "earlier-row terminal control requires equal token length")
        return DynamicArm(arm, "ready", (row_carrier_position(earlier, prefix_width=prefix_width, carrier=carrier),), scope="single", carrier=carrier, persistence=replacement_persistence)
    if arm == "D20":
        assert latest is not None
        return DynamicArm(arm, "ready", row_span_positions(latest, prefix_width=prefix_width), scope="whole_row", persistence=replacement_persistence)
    assert arm == "D21"
    if same_parent is not True:
        return _not_applicable(arm, "same-parent proof is required", same_parent=same_parent)
    if same_class is not True:
        return _not_applicable(arm, "same-class proof is required", same_class=same_class)
    if donor_row_token_ids is None:
        return _not_applicable(arm, "donor row token IDs are required")
    donor = _ints(donor_row_token_ids, label="donor row", allow_empty=False)
    assert latest is not None
    if len(donor) != len(latest):
        return _not_applicable(arm, "donor and recipient rows must have equal token length", donor_length=len(donor), recipient_length=len(latest))
    return DynamicArm(
        arm,
        "ready",
        row_span_positions(latest, prefix_width=prefix_width),
        scope="whole_row",
        persistence=replacement_persistence,
        metadata={"same_parent": True, "same_class": True, "equal_token_length": True, "donor_row_length": len(donor)},
    )


def validate_dynamic_arm(arm: DynamicArm) -> None:
    if arm.status != "ready":
        raise TechnicalInvalid(f"dynamic arm {arm.arm_id} is {arm.status}: {arm.reason}")
    if arm.arm_id in {"D01", "D10", "D11", "D12"} and len(arm.positions) != 1:
        raise TechnicalInvalid("single-carrier arm must have exactly one destination position")
    if arm.arm_id in {"D20", "D21"} and not arm.positions:
        raise TechnicalInvalid("whole-row arm must have a non-empty destination span")


def install_dynamic_replacement(
    model: Any,
    arm: DynamicArm,
    replacement: torch.Tensor,
    *,
    persistent: bool | None = None,
) -> ResidualSpanReplacement | None:
    """Install a block-23 replacement; caller owns the forward context."""

    if arm.arm_id in {"D00"}:
        return None
    validate_dynamic_arm(arm)
    layer, _ = resolve_decoder_layer(model, BLOCK_INDEX)
    keep = arm.persistence == "persistent" if persistent is None else bool(persistent)
    return ResidualSpanReplacement(layer, arm.positions, replacement, persistent=keep)


def forward_with_dynamic_arm(
    model: Any,
    model_inputs: Mapping[str, Any],
    *,
    prefix_contract: ExactPrefixContract,
    arm: DynamicArm,
    replacement: torch.Tensor | None = None,
    persistent: bool | None = None,
    attention_mask: torch.Tensor | None = None,
    execution_adapter: Literal["production", "test_adapter"] = "production",
) -> tuple[Any, dict[str, Any]]:
    """Run one full-prefix forward under a D-arm and return its mechanical receipt.

    ``model_inputs`` contains raw ``input_ids`` and explicit ``position_ids``;
    ``mrope_hash`` is receipt metadata and is removed before calling the model.
    The caller may retain other native multimodal inputs (pixel values, image
    grids, masks), but may not provide cache or forced/teacher-forced fields.
    """

    if not isinstance(model_inputs, Mapping):
        raise TechnicalInvalid("model_inputs must be an explicit mapping")
    if "input_ids" not in model_inputs or "position_ids" not in model_inputs:
        raise TechnicalInvalid("model_inputs require input_ids and position_ids")
    if model_inputs.get("custom_attention_mask") is not None:
        raise TechnicalInvalid("custom_attention_mask is not a model-consumed K11 seam; use attention_mask")
    mrope_hash = str(model_inputs.get("mrope_hash", prefix_contract.mrope_hash))
    use_cache = bool(model_inputs.get("use_cache", False))
    supplied_mask = attention_mask
    if supplied_mask is not None:
        if (
            supplied_mask.ndim != 4
            or supplied_mask.dtype != torch.bool
            or tuple(supplied_mask.shape[:2]) != (1, 1)
            or tuple(supplied_mask.shape[-2:]) != (len(prefix_contract.prefix_token_ids), len(prefix_contract.prefix_token_ids))
        ):
            raise TechnicalInvalid("attention mask must be boolean [1,1,S,S] for the exact prefix")
    metadata = {
        key: value
        for key, value in model_inputs.items()
        if key not in {"input_ids", "position_ids", "mrope_hash"}
    }
    identity = validate_exact_prefix(
        prefix_contract,
        input_ids=model_inputs["input_ids"],
        position_ids=model_inputs["position_ids"],
        mrope_hash=mrope_hash,
        use_cache=use_cache,
        extra_inputs=metadata,
    )
    if arm.status != "ready":
        return {"status": arm.status, "reason": arm.reason}, {
            **identity,
            "arm": arm.arm_id,
            "mask_sha256": hash_mask(supplied_mask),
            "mask_hash": hash_mask(supplied_mask),
            "mask_non_image_changed": False,
            "mask_offscope_changed": False,
            "mask_future_changed": False,
            "no_op_max_abs_delta": 0.0,
            "non_target_max_abs_delta": 0.0,
            "hook_call_count": 0,
            "hook_applied_count": 0,
            "hook_count": 0,
            "hook_removed": True,
            "hook_clean": True,
        }
    if arm.arm_id not in {"D00", "D01", "D10", "D11", "D12", "D20", "D21"}:
        raise TechnicalInvalid(f"unsupported dynamic arm {arm.arm_id!r}")
    payload = dict(model_inputs)
    payload.pop("mrope_hash", None)
    payload["use_cache"] = False
    # Qwen3-VL consumes the causal mask only through ``attention_mask``.
    # Passing a second, experiment-local keyword is silently ignored by the
    # installed SDPA path while still making a receipt look intervention-bound.
    payload.pop("custom_attention_mask", None)
    if supplied_mask is not None:
        payload["attention_mask"] = supplied_mask
    layer_receipt: dict[str, Any] = {}
    hook: ResidualSpanReplacement | None = None
    output: Any
    if arm.arm_id == "D00":
        output = model(**payload)
        hook_receipt = {
            "hook_clean": True,
            "applied_count": 0,
            "hook_call_count": 0,
            "hook_applied_count": 0,
            "hook_count": 0,
            "hook_removed": True,
            "non_target_max_abs_delta": 0.0,
            "no_op_max_abs_delta": 0.0,
        }
    else:
        if replacement is None:
            raise TechnicalInvalid(f"{arm.arm_id} requires explicit captured replacement state")
        layer, layer_receipt = resolve_block23(model)
        hook = ResidualSpanReplacement(
            layer,
            arm.positions,
            replacement,
            persistent=arm.persistence == "persistent" if persistent is None else bool(persistent),
        )
        try:
            with hook:
                output = model(**payload)
        finally:
            # ``__exit__`` already removes the hook, but this keeps cleanup
            # true even when a fake/model forward raises.
            hook.remove()
        hook_receipt = hook.receipt()
        hook_receipt["no_op_max_abs_delta"] = (
            hook.max_target_max_abs_delta if arm.arm_id == "D01" else 0.0
        )
    receipt = {
        **identity,
        **layer_receipt,
        "arm": arm.arm_id,
        "mask_input_key": "attention_mask" if supplied_mask is not None else "native_attention_mask",
        "mask_sha256": hash_mask(supplied_mask),
        "mask_hash": hash_mask(supplied_mask),
        "mask_non_image_changed": False,
        "mask_offscope_changed": False,
        "mask_future_changed": False,
        **hook_receipt,
    }
    return output, _issue_forward_receipt(receipt, execution_adapter=execution_adapter)


def build_k11_key_removal_mask(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    a_exclusive_positions: Sequence[int],
    query_positions: Sequence[int],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build K11: remove only covered-A-exclusive image keys for declared queries."""

    length = int(sequence_length)
    if length <= 0:
        raise TechnicalInvalid("sequence length must be positive")
    image = {int(value) for value in image_key_positions}
    removed = {int(value) for value in a_exclusive_positions}
    queries = {int(value) for value in query_positions}
    if not image or not queries or not removed.issubset(image):
        raise TechnicalInvalid("K11 requires non-empty image, query, and subset A-exclusive positions")
    if any(value < 0 or value >= length for value in image | queries | removed):
        raise TechnicalInvalid("K11 position outside sequence")
    mask = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    for query_pos in sorted(queries):
        for key_pos in sorted(removed):
            if key_pos <= query_pos:
                mask[query_pos, key_pos] = False
    # Non-image keys and future keys must remain exactly causal.
    baseline = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    changed = mask != baseline
    allowed = torch.zeros_like(mask)
    for query_pos in queries:
        for key_pos in removed:
            if key_pos <= query_pos:
                allowed[query_pos, key_pos] = True
    if bool((changed & ~allowed).any()):
        raise TechnicalInvalid("K11 altered a non-declared image key or future key")
    return mask.unsqueeze(0).unsqueeze(0)


@dataclass(frozen=True)
class StaticArm:
    arm_id: Literal["K00", "K11"]
    mask: torch.Tensor | None
    image_key_positions: tuple[int, ...] = ()
    removed_positions: tuple[int, ...] = ()
    query_positions: tuple[int, ...] = ()


def build_static_arm(
    arm_id: Literal["K00", "K11"],
    *,
    sequence_length: int | None = None,
    image_key_positions: Sequence[int] = (),
    a_exclusive_positions: Sequence[int] = (),
    query_positions: Sequence[int] = (),
    device: torch.device | str = "cpu",
) -> StaticArm:
    if arm_id == "K00":
        return StaticArm("K00", None)
    if sequence_length is None:
        raise TechnicalInvalid("K11 sequence length is required")
    image = tuple(sorted(set(int(value) for value in image_key_positions)))
    removed = tuple(sorted(set(int(value) for value in a_exclusive_positions)))
    queries = tuple(sorted(set(int(value) for value in query_positions)))
    mask = build_k11_key_removal_mask(
        sequence_length=sequence_length,
        image_key_positions=image,
        a_exclusive_positions=removed,
        query_positions=queries,
        device=device,
    )
    return StaticArm("K11", mask, image, removed, queries)


@dataclass(frozen=True)
class P3Cell:
    cell_id: Literal["Y00", "Y10", "Y01", "Y11"]
    static_arm: StaticArm
    dynamic_arm: DynamicArm
    persistence: Literal["one_shot", "persistent"]


def _mask_scope_receipt(static_arm: StaticArm, sequence_length: int) -> dict[str, Any]:
    if static_arm.arm_id == "K00":
        return {
            "mask_sha256": "none",
            "mask_hash": "none",
            "mask_non_image_changed": False,
            "mask_offscope_changed": False,
            "mask_future_changed": False,
        }
    mask = static_arm.mask
    if mask is None or mask.ndim != 4 or mask.dtype != torch.bool or tuple(mask.shape[:2]) != (1, 1) or tuple(mask.shape[-2:]) != (sequence_length, sequence_length):
        raise TechnicalInvalid("K11 requires one boolean [1,1,S,S] mask at the exact prefix length")
    baseline = torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=mask.device))
    observed = mask[0, 0]
    changed = observed != baseline
    allowed = torch.zeros_like(changed)
    for query_pos in static_arm.query_positions:
        for key_pos in static_arm.removed_positions:
            if key_pos <= query_pos:
                allowed[query_pos, key_pos] = True
    offscope = bool((changed & ~allowed).any())
    future = bool((changed & torch.triu(torch.ones_like(changed), diagonal=1)).any())
    return {
        "mask_sha256": hash_mask(mask),
        "mask_hash": hash_mask(mask),
        "mask_non_image_changed": offscope,
        "mask_offscope_changed": offscope,
        "mask_future_changed": future,
    }


def build_p3_cells(static_k11: StaticArm, dynamic_d10: DynamicArm, *, persistence: Literal["one_shot", "persistent"] = "one_shot") -> dict[str, P3Cell]:
    if static_k11.arm_id != "K11":
        raise TechnicalInvalid("P3 preselected static factor must be K11")
    if dynamic_d10.arm_id != "D10":
        raise TechnicalInvalid("P3 preselected dynamic factor must be D10")
    return {
        "Y00": P3Cell("Y00", build_static_arm("K00"), build_dynamic_arm("D00", prefix_width=0), persistence),
        "Y10": P3Cell("Y10", static_k11, build_dynamic_arm("D00", prefix_width=0), persistence),
        "Y01": P3Cell("Y01", build_static_arm("K00"), dynamic_d10, persistence),
        "Y11": P3Cell("Y11", static_k11, dynamic_d10, persistence),
    }


def forward_with_p3_cell(
    model: Any,
    model_inputs: Mapping[str, Any],
    *,
    prefix_contract: ExactPrefixContract,
    cell: P3Cell,
    replacement: torch.Tensor | None = None,
    persistent: bool | None = None,
    mask_input_key: str = "attention_mask",
    execution_adapter: Literal["production", "test_adapter"] = "production",
) -> tuple[Any, dict[str, Any]]:
    """Run one model forward with K11 and D10 installed together.

    This function owns the composition: it validates the exact native prefix,
    attaches the 4-D K11 mask, installs the block-23 terminal hook, executes a
    single model call, and returns one dual receipt.  A row generator may decode
    the returned output, but it cannot silently replace this combined forward.
    """

    if cell.cell_id not in P3_CELL_IDS:
        raise TechnicalInvalid(f"unknown P3 cell {cell.cell_id!r}")
    if mask_input_key != "attention_mask":
        raise TechnicalInvalid("P3 K11 must bind the model's real attention_mask input")
    if "input_ids" not in model_inputs or "position_ids" not in model_inputs:
        raise TechnicalInvalid("P3 model_inputs require input_ids and position_ids")
    if model_inputs.get("custom_attention_mask") is not None:
        raise TechnicalInvalid("P3 rejects the ignored custom_attention_mask bypass")
    mrope_hash = str(model_inputs.get("mrope_hash", prefix_contract.mrope_hash))
    supplied_mask = cell.static_arm.mask if cell.static_arm.arm_id == "K11" else None
    if supplied_mask is not None:
        if not isinstance(supplied_mask, torch.Tensor):
            raise TechnicalInvalid("K11 mask must be a tensor")
        _mask_scope_receipt(cell.static_arm, len(prefix_contract.prefix_token_ids))
    metadata = {
        key: value
        for key, value in model_inputs.items()
        if key not in {"input_ids", "position_ids", "mrope_hash", mask_input_key}
    }
    identity = validate_exact_prefix(
        prefix_contract,
        input_ids=model_inputs["input_ids"],
        position_ids=model_inputs["position_ids"],
        mrope_hash=mrope_hash,
        use_cache=bool(model_inputs.get("use_cache", False)),
        extra_inputs=metadata,
    )
    static_receipt = _mask_scope_receipt(cell.static_arm, len(prefix_contract.prefix_token_ids))
    if static_receipt["mask_non_image_changed"] or static_receipt["mask_future_changed"]:
        raise TechnicalInvalid("K11 mask changed an off-scope or future key")
    if cell.dynamic_arm.status != "ready":
        receipt = {
            **identity,
            **static_receipt,
            "cell_id": cell.cell_id,
            "static_arm": cell.static_arm.arm_id,
            "dynamic_arm": cell.dynamic_arm.arm_id,
            "dynamic_status": cell.dynamic_arm.status,
            "dynamic_reason": cell.dynamic_arm.reason,
            "no_op_max_abs_delta": 0.0,
            "non_target_max_abs_delta": 0.0,
            "hook_call_count": 0,
            "hook_applied_count": 0,
            "hook_count": 0,
            "hook_removed": True,
            "hook_clean": True,
        }
        return {"status": cell.dynamic_arm.status, "reason": cell.dynamic_arm.reason}, receipt
    if cell.dynamic_arm.arm_id != "D00" and replacement is None:
        raise TechnicalInvalid(f"P3 cell {cell.cell_id} dynamic arm requires replacement state")
    payload = dict(model_inputs)
    payload.pop("mrope_hash", None)
    payload["use_cache"] = False
    if supplied_mask is not None:
        payload["attention_mask"] = supplied_mask
    hook: ResidualSpanReplacement | None = None
    layer_receipt: dict[str, Any] = {}
    if cell.dynamic_arm.arm_id == "D00":
        output = model(**payload)
        dynamic_receipt = {
            "arm": "D00",
            "hook_call_count": 0,
            "hook_applied_count": 0,
            "hook_count": 0,
            "hook_removed": True,
            "hook_clean": True,
            "no_op_max_abs_delta": 0.0,
            "non_target_max_abs_delta": 0.0,
        }
    else:
        layer, layer_receipt = resolve_block23(model)
        hook = ResidualSpanReplacement(
            layer,
            cell.dynamic_arm.positions,
            replacement,
            persistent=cell.persistence == "persistent" if persistent is None else bool(persistent),
        )
        try:
            with hook:
                output = model(**payload)
        finally:
            hook.remove()
        dynamic_receipt = {
            **hook.receipt(),
            "arm": cell.dynamic_arm.arm_id,
            "no_op_max_abs_delta": hook.max_target_max_abs_delta if cell.dynamic_arm.arm_id == "D01" else 0.0,
        }
    receipt = {
        **identity,
        **static_receipt,
        **layer_receipt,
        **dynamic_receipt,
        "cell_id": cell.cell_id,
        "static_arm": cell.static_arm.arm_id,
        "dynamic_arm": cell.dynamic_arm.arm_id,
        "dynamic_status": "ready",
        "mask_input_key": "attention_mask" if supplied_mask is not None else "native_attention_mask",
    }
    return output, _issue_forward_receipt(receipt, execution_adapter=execution_adapter)


_ROW_EVIDENCE_SEAL = object()


@dataclass(frozen=True)
class GeneratedRow:
    """Evidence-backed row; owner identity is derived, never caller supplied."""

    token_ids: tuple[int, ...]
    parser_receipt: Mapping[str, Any]
    owner_match_receipt: Mapping[str, Any]
    forward_receipt: Mapping[str, Any]
    _evidence_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._evidence_seal is not _ROW_EVIDENCE_SEAL:
            raise TechnicalInvalid("GeneratedRow may only be issued by the evidence validation seam")
        object.__setattr__(self, "token_ids", _ints(self.token_ids, label="generated row", allow_empty=False))
        _validate_parser_receipt(self.parser_receipt, token_ids=self.token_ids)
        _validate_owner_match_receipt(self.owner_match_receipt)
        if not isinstance(self.forward_receipt, Mapping):
            raise TechnicalInvalid("generated row requires a forward mechanical receipt")

    @property
    def parse_status(self) -> str:
        return str(self.parser_receipt["parse_status"])

    @property
    def complete(self) -> bool:
        return _strict_bool(self.parser_receipt["complete"], label="parser complete")

    @property
    def stop(self) -> bool:
        return _strict_bool(self.parser_receipt["stop"], label="parser stop")

    @property
    def stop_reason(self) -> str:
        return str(self.parser_receipt["stop_reason"])

    @property
    def owner_id(self) -> str | None:
        if self.owner_match_receipt["status"] != "matched":
            return None
        return str(self.owner_match_receipt["matched_owner_id"])

    @property
    def unmatched(self) -> bool:
        return self.owner_match_receipt["status"] == "unmatched"

    @property
    def ambiguous(self) -> bool:
        return self.owner_match_receipt["status"] == "ambiguous"

    @property
    def malformed(self) -> bool:
        return self.parse_status == "malformed"

    @property
    def invalid(self) -> bool:
        return self.parse_status == "invalid"

    def as_dict(self, *, duplicate: bool) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "parser_receipt": dict(self.parser_receipt),
            "owner_match_receipt": dict(self.owner_match_receipt),
            "forward_receipt": dict(self.forward_receipt),
            "owner_id": self.owner_id,
            "parse_status": self.parse_status,
            "complete": self.complete,
            "stop": self.stop,
            "stop_reason": self.stop_reason,
            "duplicate": _strict_bool(duplicate, label="derived duplicate"),
            "unmatched": self.unmatched,
            "ambiguous": self.ambiguous,
            "malformed": self.malformed,
            "invalid": self.invalid,
        }


@dataclass(frozen=True)
class NativeRowParser:
    """Checkpoint-native closed/commit parser; syntax owns completion status."""

    wrapper: Literal["closed", "commit"]
    object_ref_start_token_id: int = 151646
    object_ref_end_token_id: int = 151647
    box_start_token_id: int = 151648
    box_end_token_id: int = BOX_END_TOKEN_ID
    coordinate_token_min: int = COORDINATE_TOKEN_MIN
    coordinate_token_max: int = COORDINATE_TOKEN_MAX
    coordinate_count: int = 4
    commit_token_id: int = COMMIT_TOKEN_ID
    im_end_token_id: int = IM_END_TOKEN_ID

    def __post_init__(self) -> None:
        if self.wrapper not in {"closed", "commit"}:
            raise TechnicalInvalid("native parser wrapper must be closed or commit")
        if self.coordinate_count != 4:
            raise TechnicalInvalid("native parser requires exactly four XYXY coordinates")
        if self.wrapper == "closed" and self.commit_token_id < 0:
            raise TechnicalInvalid("closed parser commit token configuration is invalid")

    @classmethod
    def closed(cls, **kwargs: Any) -> "NativeRowParser":
        return cls("closed", **kwargs)

    @classmethod
    def commit(cls, **kwargs: Any) -> "NativeRowParser":
        return cls("commit", **kwargs)

    def parse(self, token_ids: Sequence[int]) -> dict[str, Any]:
        row = _ints(token_ids, label="generated row", allow_empty=False)

        def receipt(
            parse_status: Literal["accepted", "malformed", "terminal", "invalid"],
            *,
            complete: bool,
            stop: bool,
            stop_reason: str,
            coordinates: Sequence[int] = (),
        ) -> dict[str, Any]:
            return {
                "parser_id": f"native_{self.wrapper}_four_coordinate_v1",
                "wrapper": self.wrapper,
                "token_ids_sha256": hash_token_ids(row),
                "parse_status": parse_status,
                "complete": _strict_bool(complete, label="parser complete"),
                "stop": _strict_bool(stop, label="parser stop"),
                "stop_reason": str(stop_reason),
                "coordinate_token_ids": [int(value) for value in coordinates],
            }

        if row == (self.im_end_token_id,):
            return receipt("terminal", complete=False, stop=True, stop_reason="terminal")
        if row[0] != self.object_ref_start_token_id:
            return receipt("malformed", complete=False, stop=False, stop_reason="missing_object_ref_start")
        if row.count(self.object_ref_start_token_id) != 1 or row.count(self.object_ref_end_token_id) != 1:
            return receipt("malformed", complete=False, stop=False, stop_reason="object_ref_marker_count")
        object_end = row.index(self.object_ref_end_token_id)
        if object_end <= 1:
            return receipt("malformed", complete=False, stop=False, stop_reason="empty_description")
        if row.count(self.box_start_token_id) != 1:
            return receipt("malformed", complete=False, stop=False, stop_reason="box_start_count")
        if row.count(self.box_end_token_id) != 1:
            return receipt("malformed", complete=False, stop=False, stop_reason="box_end_count")
        box_start = row.index(self.box_start_token_id)
        if box_start <= object_end:
            return receipt("malformed", complete=False, stop=False, stop_reason="box_start_order")
        coordinate_start = box_start + 1
        coordinate_end = coordinate_start + self.coordinate_count
        if coordinate_end >= len(row):
            return receipt("malformed", complete=False, stop=False, stop_reason="truncated_coordinates")
        coordinates = row[coordinate_start:coordinate_end]
        if any(token < self.coordinate_token_min or token > self.coordinate_token_max for token in coordinates):
            return receipt("malformed", complete=False, stop=False, stop_reason="coordinate_token_range")
        if row[coordinate_end] != self.box_end_token_id:
            return receipt("malformed", complete=False, stop=False, stop_reason="missing_box_end")
        if self.wrapper == "closed":
            if self.commit_token_id in row or coordinate_end != len(row) - 1:
                return receipt("malformed", complete=False, stop=False, stop_reason="closed_wrapper_suffix")
        else:
            if row.count(self.commit_token_id) != 1 or coordinate_end + 1 != len(row) - 1 or row[-1] != self.commit_token_id:
                return receipt("malformed", complete=False, stop=False, stop_reason="commit_wrapper_suffix")
        return receipt("accepted", complete=True, stop=False, stop_reason="complete_row", coordinates=coordinates)


def build_native_row_parser(wrapper: Literal["closed", "commit"], **kwargs: Any) -> NativeRowParser:
    """Build the explicit S (closed) or A (commit) checkpoint parser."""

    return NativeRowParser(wrapper, **kwargs)


def _validate_parser_receipt(receipt: Mapping[str, Any], *, token_ids: Sequence[int]) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise TechnicalInvalid("generated row requires a native parser receipt")
    required = (
        "parser_id",
        "wrapper",
        "token_ids_sha256",
        "parse_status",
        "complete",
        "stop",
        "stop_reason",
        "coordinate_token_ids",
    )
    missing = [name for name in required if name not in receipt]
    if missing:
        raise TechnicalInvalid(f"native parser receipt lacks fields: {missing}")
    for name in ("parser_id", "wrapper", "token_ids_sha256", "parse_status", "stop_reason"):
        if not isinstance(receipt[name], str) or not receipt[name]:
            raise TechnicalInvalid(f"native parser receipt field {name} must be a non-empty string")
    complete = _strict_bool(receipt["complete"], label="parser complete")
    stop = _strict_bool(receipt["stop"], label="parser stop")
    if receipt["token_ids_sha256"] != hash_token_ids(token_ids):
        raise TechnicalInvalid("native parser receipt token identity differs from generated row")
    status = receipt["parse_status"]
    if status not in {"accepted", "malformed", "terminal", "invalid"}:
        raise TechnicalInvalid("native parser receipt has an unknown parse status")
    if complete != (status == "accepted"):
        raise TechnicalInvalid("native parser complete flag disagrees with parse status")
    if stop != (status == "terminal"):
        raise TechnicalInvalid("native parser stop flag disagrees with parse status")
    coordinates = receipt["coordinate_token_ids"]
    if isinstance(coordinates, (str, bytes)) or not isinstance(coordinates, Sequence):
        raise TechnicalInvalid("native parser coordinate_token_ids must be a sequence")
    if complete and len(coordinates) != 4:
        raise TechnicalInvalid("complete native parser receipt must contain four coordinates")
    return dict(receipt)


def _validate_owner_match_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise TechnicalInvalid("generated row requires a physical-owner matcher receipt")
    required = (
        "matcher_id",
        "status",
        "matched_owner_id",
        "source_specific",
        "physical_match",
        "unmatched",
        "ambiguous",
    )
    missing = [name for name in required if name not in receipt]
    if missing:
        raise TechnicalInvalid(f"physical-owner matcher receipt lacks fields: {missing}")
    if not isinstance(receipt["matcher_id"], str) or not receipt["matcher_id"]:
        raise TechnicalInvalid("physical-owner matcher_id must be a non-empty string")
    status = receipt["status"]
    if status not in {"matched", "unmatched", "ambiguous"}:
        raise TechnicalInvalid("physical-owner matcher status must be matched, unmatched, or ambiguous")
    source_specific = _strict_bool(receipt["source_specific"], label="owner source_specific")
    physical_match = _strict_bool(receipt["physical_match"], label="owner physical_match")
    unmatched = _strict_bool(receipt["unmatched"], label="owner unmatched")
    ambiguous = _strict_bool(receipt["ambiguous"], label="owner ambiguous")
    if not source_specific:
        raise TechnicalInvalid("owner matcher must be source-specific")
    owner_id = receipt["matched_owner_id"]
    if status == "matched":
        if not physical_match or unmatched or ambiguous:
            raise TechnicalInvalid("matched owner receipt has inconsistent boolean evidence")
        if not isinstance(owner_id, str) or not owner_id:
            raise TechnicalInvalid("matched owner receipt requires one non-empty physical owner ID")
    else:
        if physical_match or unmatched != (status == "unmatched") or ambiguous != (status == "ambiguous"):
            raise TechnicalInvalid("neutral owner receipt has inconsistent boolean evidence")
        if owner_id is not None:
            raise TechnicalInvalid("unmatched/ambiguous owner receipt must not claim an owner ID")
    return dict(receipt)


def _validate_row_forward_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_prefix_token_ids: Sequence[int],
    allow_test_adapter: bool,
) -> Mapping[str, Any]:
    _strict_bool(allow_test_adapter, label="allow_test_adapter")
    validate_mechanical_receipt(receipt)
    if receipt["prefix_token_ids_sha256"] != hash_token_ids(expected_prefix_token_ids):
        raise TechnicalInvalid("row forward receipt prefix differs from the actual natural prefix")
    adapter = receipt["execution_adapter"]
    real_forward = _strict_bool(receipt["real_forward"], label="forward real_forward")
    test_adapter = _strict_bool(receipt["test_adapter"], label="forward test_adapter")
    if adapter == "production":
        if not isinstance(receipt, ForwardMechanicalReceipt) or not real_forward or test_adapter:
            raise TechnicalInvalid("production row requires a live in-module real-forward receipt")
    elif adapter == "test_adapter":
        if not allow_test_adapter or real_forward or not test_adapter:
            raise TechnicalInvalid("test forward receipt requires an explicitly allowed test adapter")
    else:
        raise TechnicalInvalid("unknown row forward execution adapter")
    # Preserve the issuer marker for production receipts so an evidence-backed
    # GeneratedRow can safely pass through this seam again. Serialization is
    # deliberately deferred to GeneratedRow.as_dict().
    return receipt if isinstance(receipt, ForwardMechanicalReceipt) else dict(receipt)


def _coerce_row(
    value: GeneratedRow | Mapping[str, Any] | Sequence[int],
    *,
    parser: NativeRowParser | None,
    owner_matcher: Callable[[tuple[int, ...], Mapping[str, Any]], Mapping[str, Any]] | None,
    expected_prefix_token_ids: Sequence[int],
    allow_test_adapter: bool,
    forward_receipt: Mapping[str, Any] | None = None,
) -> GeneratedRow:
    if parser is None:
        raise TechnicalInvalid("an explicit checkpoint-native closed/commit parser contract is required")
    _strict_bool(allow_test_adapter, label="allow_test_adapter")
    if owner_matcher is None:
        raise TechnicalInvalid("a source-specific physical-owner matcher callback is required")
    if isinstance(value, GeneratedRow):
        token_ids = _ints(value.token_ids, label="generated row", allow_empty=False)
        embedded_forward = value.forward_receipt
    elif isinstance(value, Mapping):
        for name in ("duplicate", "complete", "stop", "unmatched", "ambiguous", "malformed", "invalid", "accepted_complete_row"):
            if name in value:
                _strict_bool(value[name], label=f"generated row {name}")
        if "owner_id" in value or "matched_owner_id" in value:
            raise TechnicalInvalid("generated row must not provide a free caller owner_id")
        token_ids = _ints(value.get("token_ids", value.get("raw_generated_token_ids", ())), label="generated row", allow_empty=False)
        embedded_forward = value.get("forward_receipt")
    else:
        token_ids = _ints(value, label="generated row", allow_empty=False)
        embedded_forward = None
    if forward_receipt is not None and embedded_forward is not None and forward_receipt is not embedded_forward:
        raise TechnicalInvalid("generated row supplied a competing forward receipt")
    selected_forward = forward_receipt if forward_receipt is not None else embedded_forward
    if not isinstance(selected_forward, Mapping):
        raise TechnicalInvalid("generated row lacks a real-forward mechanical receipt")
    parsed = _validate_parser_receipt(parser.parse(token_ids), token_ids=token_ids)
    matched = _validate_owner_match_receipt(owner_matcher(token_ids, parsed))
    checked_forward = _validate_row_forward_receipt(
        selected_forward,
        expected_prefix_token_ids=expected_prefix_token_ids,
        allow_test_adapter=allow_test_adapter,
    )
    return GeneratedRow(token_ids, parsed, matched, checked_forward, _ROW_EVIDENCE_SEAL)


def run_native_horizon(
    prefix_token_ids: Sequence[int],
    row_generator: Callable[[tuple[int, ...], int], GeneratedRow | Mapping[str, Any] | Sequence[int]],
    *,
    horizon: int = 3,
    parser: NativeRowParser | None = None,
    owner_matcher: Callable[[tuple[int, ...], Mapping[str, Any]], Mapping[str, Any]] | None = None,
    covered_owner_ids: Sequence[str] = (),
    forced: bool = False,
    teacher_forced: bool = False,
    allow_test_adapter: bool = False,
) -> dict[str, Any]:
    """Release complete rows from a natural exact raw-token prefix."""

    _strict_bool(forced, label="forced")
    _strict_bool(teacher_forced, label="teacher_forced")
    _strict_bool(allow_test_adapter, label="allow_test_adapter")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise TechnicalInvalid("horizon must be a positive integer")
    if forced or teacher_forced:
        raise TechnicalInvalid("forced/teacher-forced row generation cannot support natural causal claims")
    prefix = list(_ints(prefix_token_ids, label="prefix_token_ids", allow_empty=False))
    rows: list[GeneratedRow] = []
    stop_reason = "horizon_exhausted"
    for row_index in range(1, horizon + 1):
        generated = row_generator(tuple(prefix), row_index)
        if isinstance(generated, str):
            raise TechnicalInvalid("row generator returned text; re-tokenization is forbidden")
        row = _coerce_row(
            generated,
            parser=parser,
            owner_matcher=owner_matcher,
            expected_prefix_token_ids=prefix,
            allow_test_adapter=allow_test_adapter,
        )
        rows.append(row)
        if row.invalid:
            stop_reason = "invalid"
            break
        if row.malformed or not row.complete:
            stop_reason = row.stop_reason or "malformed"
            break
        prefix.extend(row.token_ids)
        if row.stop_reason in {"terminal", "im_end", "stop"} or IM_END_TOKEN_ID in row.token_ids:
            stop_reason = row.stop_reason if row.stop_reason != "complete_row" else "terminal"
            break
    return bookkeep_horizon(
        rows,
        covered_owner_ids=covered_owner_ids,
        final_prefix_token_ids=prefix,
        horizon=horizon,
        stop_reason=stop_reason,
    )


def run_dynamic_matrix(
    prefix_token_ids: Sequence[int],
    arms: Mapping[str, DynamicArm],
    row_generator_factory: Callable[[str, DynamicArm, int], Callable[[tuple[int, ...], int], GeneratedRow | Mapping[str, Any] | Sequence[int]]],
    *,
    covered_owner_ids: Sequence[str] = (),
    horizons: Sequence[int] = (1, 3),
    parser: NativeRowParser | None = None,
    owner_matcher: Callable[[tuple[int, ...], Mapping[str, Any]], Mapping[str, Any]] | None = None,
    allow_test_adapter: bool = False,
) -> dict[str, Any]:
    """Run D arms at one-row and fixed-three-row natural horizons.

    ``row_generator_factory`` receives ``(arm_id, arm, horizon)`` and must
    return a generator that performs the real full-prefix forward.  This keeps
    static masks, checkpoint-native parsing, and one-shot/persistent hook
    lifetime visible to the caller while the shared bookkeeping stays exact.
    """

    result: dict[str, Any] = {"unit_id": UNIT_ID, "prefix_token_ids_sha256": hash_token_ids(prefix_token_ids), "arms": {}}
    for horizon in horizons:
        horizon_key = f"horizon_{int(horizon)}"
        result["arms"][horizon_key] = {}
        for arm_id, arm in arms.items():
            if arm_id not in DYNAMIC_ARM_IDS:
                raise ValueError(f"unknown dynamic arm {arm_id!r}")
            if arm.status != "ready":
                result["arms"][horizon_key][arm_id] = {
                    "status": arm.status,
                    "reason": arm.reason,
                    "invalid": False,
                    "not_applicable": True,
                }
                continue
            generator = row_generator_factory(arm_id, arm, int(horizon))
            try:
                receipt = run_native_horizon(
                    prefix_token_ids,
                    generator,
                    horizon=int(horizon),
                    covered_owner_ids=covered_owner_ids,
                    parser=parser,
                    owner_matcher=owner_matcher,
                    allow_test_adapter=allow_test_adapter,
                )
            except TechnicalInvalid as exc:
                receipt = {
                    "status": "invalid",
                    "invalid": True,
                    "invalid_reasons": [str(exc)],
                    "arm_id": arm_id,
                }
                result["arms"][horizon_key][arm_id] = receipt
                continue
            receipt["status"] = "valid" if not receipt["invalid"] else "invalid"
            receipt["arm_id"] = arm_id
            receipt["persistence"] = arm.persistence
            result["arms"][horizon_key][arm_id] = receipt
    return result


def run_p3_matrix(
    prefix_token_ids: Sequence[int],
    cells: Mapping[str, P3Cell],
    row_generator_factory: Callable[[str, P3Cell, int], Callable[[tuple[int, ...], int], GeneratedRow | Mapping[str, Any] | Sequence[int]]],
    *,
    covered_owner_ids: Sequence[str] = (),
    horizons: Sequence[int] = (1, 3),
    parser: NativeRowParser | None = None,
    owner_matcher: Callable[[tuple[int, ...], Mapping[str, Any]], Mapping[str, Any]] | None = None,
    allow_test_adapter: bool = False,
) -> dict[str, Any]:
    """Run Y00/Y10/Y01/Y11 with shared native-prefix and release budgets."""

    missing = [cell for cell in P3_CELL_IDS if cell not in cells]
    if missing:
        raise TechnicalInvalid(f"P3 matrix is missing cells: {missing}")
    result: dict[str, Any] = {"unit_id": UNIT_ID, "prefix_token_ids_sha256": hash_token_ids(prefix_token_ids), "cells": {}}
    for horizon in horizons:
        horizon_key = f"horizon_{int(horizon)}"
        result["cells"][horizon_key] = {}
        for cell_id in P3_CELL_IDS:
            cell = cells[cell_id]
            if cell.dynamic_arm.status != "ready":
                result["cells"][horizon_key][cell_id] = {
                    "status": cell.dynamic_arm.status,
                    "reason": cell.dynamic_arm.reason,
                    "not_applicable": True,
                }
                continue
            generator = row_generator_factory(cell_id, cell, int(horizon))
            try:
                receipt = run_native_horizon(
                    prefix_token_ids,
                    generator,
                    horizon=int(horizon),
                    covered_owner_ids=covered_owner_ids,
                    parser=parser,
                    owner_matcher=owner_matcher,
                    allow_test_adapter=allow_test_adapter,
                )
            except TechnicalInvalid as exc:
                receipt = {
                    "status": "invalid",
                    "invalid": True,
                    "invalid_reasons": [str(exc)],
                    "cell_id": cell_id,
                }
                result["cells"][horizon_key][cell_id] = receipt
                continue
            receipt["status"] = "valid" if not receipt["invalid"] else "invalid"
            receipt["cell_id"] = cell_id
            receipt["static_arm"] = cell.static_arm.arm_id
            receipt["dynamic_arm"] = cell.dynamic_arm.arm_id
            receipt["persistence"] = cell.persistence
            result["cells"][horizon_key][cell_id] = receipt
        if all(cell_id in result["cells"][horizon_key] and "net" in result["cells"][horizon_key][cell_id] for cell_id in P3_CELL_IDS):
            result[horizon_key] = factorial_deltas(result["cells"][horizon_key])
    return result


def run_p3_persistent_rows(
    model: Any,
    initial_prefix_token_ids: Sequence[int],
    *,
    input_builder: Callable[[tuple[int, ...]], Mapping[str, Any]],
    row_decoder: Callable[[Any, tuple[int, ...], int], GeneratedRow | Mapping[str, Any] | Sequence[int]],
    parser: NativeRowParser,
    owner_matcher: Callable[[tuple[int, ...], Mapping[str, Any]], Mapping[str, Any]],
    wrapper: str,
    carrier: CarrierContract,
    initial_latest_row_token_ids: Sequence[int],
    static_arm_builder: Callable[[tuple[int, ...], int], StaticArm],
    replacement_builder: Callable[[Any, tuple[int, ...], int, DynamicArm], torch.Tensor],
    covered_owner_ids: Sequence[str] = (),
    max_rows: int = 3,
    execution_adapter: Literal["production", "test_adapter"] = "production",
    allow_test_adapter: bool = False,
) -> dict[str, Any]:
    """Persistent natural horizon: rebuild exact prefix/K11/D10 once per row.

    The helper itself performs the combined K11+block23 forward.  Factories may
    build native tensors, replacements, or decode the returned output, but may
    not substitute a separate model call or a forced/teacher-forced row.
    """

    prefix = list(_ints(initial_prefix_token_ids, label="initial_prefix_token_ids", allow_empty=False))
    _strict_bool(allow_test_adapter, label="allow_test_adapter")
    latest_row = _ints(initial_latest_row_token_ids, label="initial_latest_row_token_ids", allow_empty=False)
    if len(latest_row) > len(prefix):
        raise TechnicalInvalid("initial latest row is longer than the exact prefix")
    initial_latest_start = len(prefix) - len(latest_row)
    if parser.parse(latest_row).get("complete") is not True:
        raise TechnicalInvalid("initial latest row must be a parser-accepted complete row")
    rows: list[GeneratedRow] = []
    per_row_calls: list[dict[str, Any]] = []
    stop_reason = "horizon_exhausted"
    for row_index in range(1, int(max_rows) + 1):
        current_prefix = tuple(prefix)
        try:
            raw_inputs = dict(input_builder(current_prefix))
            if "input_ids" not in raw_inputs or "position_ids" not in raw_inputs:
                raise TechnicalInvalid("input_builder must return input_ids and native position_ids")
            contract = build_exact_prefix_contract(
                current_prefix,
                raw_inputs["position_ids"],
                wrapper=wrapper,
                image_grid_thw=raw_inputs.get("image_grid_thw"),
                rope_deltas=raw_inputs.get("rope_deltas"),
            )
            static_arm = static_arm_builder(current_prefix, row_index)
            if static_arm.arm_id not in {"K00", "K11"}:
                raise TechnicalInvalid("persistent P3 static arm must be K00 or K11")
            latest_start = len(current_prefix) - len(latest_row)
            if latest_start < 0:
                raise TechnicalInvalid("latest row is outside the exact prefix")
            dynamic_arm = build_dynamic_arm(
                "D10",
                prefix_width=latest_start,
                latest_row_token_ids=latest_row,
                carrier=carrier,
                replacement_persistence="one_shot",
            )
            cell_id: Literal["Y01", "Y11"] = "Y11" if static_arm.arm_id == "K11" else "Y01"
            cell = P3Cell(cell_id, static_arm, dynamic_arm, "persistent")
            replacement = replacement_builder(model, current_prefix, row_index, dynamic_arm)
            output, receipt = forward_with_p3_cell(
                model,
                raw_inputs,
                prefix_contract=contract,
                cell=cell,
                replacement=replacement,
                persistent=False,
                execution_adapter=execution_adapter,
            )
            row = _coerce_row(
                row_decoder(output, current_prefix, row_index),
                parser=parser,
                owner_matcher=owner_matcher,
                expected_prefix_token_ids=current_prefix,
                allow_test_adapter=allow_test_adapter,
                forward_receipt=receipt,
            )
            rows.append(row)
            per_row_calls.append(
                {
                    "row_index": row_index,
                    "prefix_token_ids_sha256": contract.prefix_hash,
                    "position_ids_sha256": hash_position_ids(contract.position_ids),
                    "mrope_hash": contract.mrope_hash,
                    "mask_sha256": receipt["mask_sha256"],
                    "dynamic_arm": dynamic_arm.arm_id,
                    "hook_call_count": receipt["hook_call_count"],
                    "hook_applied_count": receipt["hook_applied_count"],
                    "hook_clean": receipt["hook_clean"],
                }
            )
        except TechnicalInvalid as exc:
            stop_reason = "invalid"
            per_row_calls.append({"row_index": row_index, "status": "invalid", "invalid_reasons": [str(exc)]})
            break
        if row.invalid:
            stop_reason = "invalid"
            break
        if row.malformed or row.parse_status == "terminal" or not row.complete:
            stop_reason = row.stop_reason
            break
        prefix.extend(row.token_ids)
        latest_row = row.token_ids
        if row.stop_reason in {"terminal", "im_end", "stop"} or IM_END_TOKEN_ID in row.token_ids:
            stop_reason = row.stop_reason if row.stop_reason != "complete_row" else "terminal"
            break
    result = bookkeep_horizon(
        rows,
        covered_owner_ids=covered_owner_ids,
        final_prefix_token_ids=prefix,
        horizon=max_rows,
        stop_reason=stop_reason,
    )
    result.update(
        {
            "status": "invalid" if stop_reason == "invalid" else "valid",
            "persistent": True,
            "persistent_reinstalled_per_row": True,
            "per_row_calls": per_row_calls,
            "initial_latest_row_start": initial_latest_start,
            "wrapper": wrapper,
        }
    )
    return result


def bookkeep_horizon(
    rows: Sequence[GeneratedRow],
    *,
    covered_owner_ids: Sequence[str] = (),
    final_prefix_token_ids: Sequence[int] | None = None,
    horizon: int = 3,
    stop_reason: str = "horizon_exhausted",
) -> dict[str, Any]:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise TechnicalInvalid("horizon must be a positive integer")
    checked = list(rows)
    if any(not isinstance(row, GeneratedRow) for row in checked):
        raise TechnicalInvalid("bookkeeping accepts only evidence-backed GeneratedRow values")
    base: set[str] = set()
    for value in covered_owner_ids:
        if not isinstance(value, str) or not value:
            raise TechnicalInvalid("covered owner IDs must be non-empty physical-owner strings")
        base.add(value)
    arm = {row.owner_id for row in checked if row.complete and row.parse_status == "accepted" and row.owner_id is not None}
    seen = set(base)
    duplicate_flags: list[bool] = []
    for row in checked:
        owner_id = row.owner_id if row.complete and row.parse_status == "accepted" else None
        duplicate = owner_id is not None and owner_id in seen
        duplicate_flags.append(duplicate)
        if owner_id is not None:
            seen.add(owner_id)
    gained = sorted(arm - base)
    retained = sorted(arm & base)
    lost = sorted(base - arm)
    repeat = {f"t+{index}": 0 for index in range(1, 4)}
    for index, row in enumerate(checked[:3], start=1):
        if row.complete and row.parse_status == "accepted" and row.owner_id is not None and row.owner_id in base:
            repeat[f"t+{index}"] += 1
    parse = {
        "valid_rows": sum(row.complete and row.parse_status == "accepted" for row in checked),
        "duplicate_rows": sum(duplicate_flags),
        "unmatched_rows": sum(row.unmatched for row in checked),
        "ambiguous_rows": sum(row.ambiguous for row in checked),
        "malformed_rows": sum(row.malformed or row.parse_status == "malformed" for row in checked),
        "invalid_rows": sum(row.invalid or row.parse_status == "invalid" for row in checked),
    }
    return {
        "rows": [row.as_dict(duplicate=duplicate) for row, duplicate in zip(checked, duplicate_flags, strict=True)],
        "horizon_requested": int(horizon),
        "horizon_rows_generated": len(checked),
        "horizon_complete": len(checked) >= int(horizon) and all(row.complete for row in checked),
        "final_prefix_token_ids": list(_ints(final_prefix_token_ids or (), label="final_prefix_token_ids")),
        "final_prefix_token_ids_sha256": hash_token_ids(final_prefix_token_ids or ()),
        "G": gained,
        "K": retained,
        "L": lost,
        "gained_owner_ids": gained,
        "retained_owner_ids": retained,
        "lost_owner_ids": lost,
        "net": len(gained) - len(lost),
        "repeat_hazard": repeat,
        "parse": parse,
        "stop": {"stopped": stop_reason != "horizon_exhausted", "stop_reason": stop_reason},
        "invalid": (parse["invalid_rows"] + parse["malformed_rows"]) > 0,
        "natural": True,
    }


def factorial_deltas(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compute P3 Y10/Y01/Y11 deltas on net owner utility."""

    missing = [cell for cell in P3_CELL_IDS if cell not in results]
    if missing:
        raise TechnicalInvalid(f"P3 result is missing cells: {missing}")
    values = {cell: int(results[cell].get("net", 0)) for cell in P3_CELL_IDS}
    return {
        "net_by_cell": values,
        "Delta_static": values["Y10"] - values["Y00"],
        "Delta_dynamic": values["Y01"] - values["Y00"],
        "tau": (values["Y11"] - values["Y10"]) - (values["Y01"] - values["Y00"]),
    }


def validate_mechanical_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate conclusion-critical identity and hook fields."""

    required = (
        "prefix_token_ids_sha256",
        "position_ids_sha256",
        "mrope_hash",
        "mask_sha256",
        "wrapper",
        "no_op_max_abs_delta",
        "non_target_max_abs_delta",
        "mask_non_image_changed",
        "mask_offscope_changed",
        "mask_future_changed",
        "hook_call_count",
        "hook_applied_count",
        "hook_clean",
        "execution_adapter",
        "real_forward",
        "test_adapter",
    )
    missing = [field for field in required if field not in receipt]
    if missing:
        raise TechnicalInvalid(f"mechanical receipt lacks fields: {missing}")
    for name in ("prefix_token_ids_sha256", "position_ids_sha256", "mrope_hash", "mask_sha256", "wrapper"):
        if not isinstance(receipt[name], str) or not receipt[name]:
            raise TechnicalInvalid(f"mechanical receipt field {name} must be a non-empty string")
    for name in ("no_op_max_abs_delta", "non_target_max_abs_delta"):
        value = receipt[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise TechnicalInvalid(f"mechanical receipt field {name} must be a finite non-negative number")
    for name in ("mask_non_image_changed", "mask_offscope_changed", "mask_future_changed", "hook_clean"):
        _strict_bool(receipt[name], label=f"mechanical receipt {name}")
    if receipt["execution_adapter"] not in {"production", "test_adapter"}:
        raise TechnicalInvalid("mechanical receipt execution_adapter is invalid")
    _strict_bool(receipt["real_forward"], label="mechanical receipt real_forward")
    _strict_bool(receipt["test_adapter"], label="mechanical receipt test_adapter")
    for name in ("hook_call_count", "hook_applied_count"):
        value = receipt[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TechnicalInvalid(f"mechanical receipt field {name} must be a non-negative integer")
    if "mask_hash" in receipt and (not isinstance(receipt["mask_hash"], str) or not receipt["mask_hash"]):
        raise TechnicalInvalid("mechanical receipt field mask_hash must be a non-empty string")
    if "hook_count" in receipt and (isinstance(receipt["hook_count"], bool) or not isinstance(receipt["hook_count"], int) or receipt["hook_count"] < 0):
        raise TechnicalInvalid("mechanical receipt field hook_count must be a non-negative integer")
    if "hook_removed" in receipt:
        _strict_bool(receipt["hook_removed"], label="mechanical receipt hook_removed")
    for name in ("retokenized", "forced", "teacher_forced", "donor_cache", "post_rope_k_swap"):
        if name in receipt:
            _strict_bool(receipt[name], label=f"mechanical receipt {name}")
    if receipt.get("retokenized") or receipt.get("forced") or receipt.get("teacher_forced"):
        raise TechnicalInvalid("receipt is not a natural exact-prefix run")
    if receipt.get("donor_cache") or receipt.get("post_rope_k_swap"):
        raise TechnicalInvalid("donor cache/post-RoPE K swap is forbidden")
    if receipt.get("hook_clean") is not True:
        raise TechnicalInvalid("intervention hook leaked")
    if receipt.get("hook_removed") is False:
        raise TechnicalInvalid("intervention hook was not removed")
    if float(receipt.get("no_op_max_abs_delta", 0.0)) > NOOP_TOLERANCE:
        raise TechnicalInvalid("self/no-op drift exceeds tolerance")
    if float(receipt.get("non_target_max_abs_delta", 0.0)) > NOOP_TOLERANCE:
        raise TechnicalInvalid("non-target residual position drift exceeds tolerance")
    if receipt.get("mask_non_image_changed") or receipt.get("mask_offscope_changed") or receipt.get("mask_future_changed"):
        raise TechnicalInvalid("K11 changed non-image or future keys")
    return {"passed": True, "technical_valid": True}


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--print-contract", action="store_true")
    args = parser.parse_args()
    if args.print_contract:
        print(json.dumps({"unit_id": UNIT_ID, "dynamic_arms": DYNAMIC_ARM_IDS, "p3_cells": P3_CELL_IDS}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Returned-residual actuators for the natural-boundary scalar probe.

The natural-boundary runner recomputes one full prefix for every next-token
decision.  This module owns the small model-facing seam used by the N matrix:
the output returned by Qwen text decoder block 23 is replaced for the
declared history positions during *one* scalar forward.  No token, position,
cache, or model input is changed here.

The hook deliberately derives every replacement from the activation returned
by the current forward.  In particular, the caller never has to materialize
or carry a donor tensor between growing prefixes.  ``N01`` writes an exact
self copy.  ``N10`` uses only the nonterminal states of the same latest row;
``N20`` uses that complete row's own mean.  Both vectors are norm-matched to
the target span, never to prompt/image states elsewhere in the prefix.

The public callback returned by :func:`make_residual_actuator_callback` is
compatible with ``run_natural_boundary_routing_history_probe``: it returns a
context manager and can therefore be entered independently for every
no-cache scalar recomputation.  All mechanical receipts are retained on the
context object and include the resolved layer, exact positions, finite-value
checks, pre/post norms, target and off-target deltas, and hook lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal, cast

import torch


BLOCK_INDEX = 23
NOOP_TOLERANCE = 1e-4
SCHEMA_VERSION = "natural_boundary_residual_actuators.v1"
RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.receipt.v1"
ARM_IDS = ("N00", "N01", "N10", "N20")
ResidualArm = Literal["N00", "N01", "N10", "N20"]


class TechnicalInvalid(ValueError):
    """Raised when the returned-residual contract is ambiguous or unsafe."""


# Existing runners use this name for mechanical failures.  Keep a descriptive
# alias so callers can catch either spelling without importing the runner.
ResidualActuatorError = TechnicalInvalid


def _as_positions(values: Sequence[int] | torch.Tensor, *, label: str) -> tuple[int, ...]:
    """Normalize and validate absolute sequence positions."""

    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise TechnicalInvalid(f"{label} must be a one-dimensional position list")
        raw = values.detach().cpu().tolist()
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TechnicalInvalid(f"{label} must be a sequence of integer positions")
    else:
        raw = list(values)
    result: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TechnicalInvalid(f"{label} must contain non-negative integer positions")
        result.append(int(value))
    if len(result) != len(set(result)):
        raise TechnicalInvalid(f"{label} must contain unique positions")
    return tuple(result)


def _validate_arm(arm_id: str, positions: Sequence[int] | torch.Tensor) -> tuple[ResidualArm, tuple[int, ...]]:
    arm = str(arm_id)
    if arm not in ARM_IDS:
        raise TechnicalInvalid(f"unknown residual arm {arm!r}")
    normalized = _as_positions(positions, label=f"{arm} positions")
    typed = cast(ResidualArm, arm)
    if typed == "N00":
        if normalized:
            raise TechnicalInvalid("N00 must not declare residual positions")
    elif typed in {"N01", "N10"}:
        if len(normalized) != 1:
            raise TechnicalInvalid(f"{typed} requires exactly one terminal position")
    else:
        if not normalized:
            raise TechnicalInvalid("N20 requires a non-empty latest-row span")
        ordered = tuple(sorted(normalized))
        expected = tuple(range(ordered[0], ordered[-1] + 1))
        if ordered != expected:
            raise TechnicalInvalid("N20 positions must be one contiguous latest-row span")
    return typed, normalized


def _validate_background_positions(
    arm_id: ResidualArm,
    target_positions: tuple[int, ...],
    background_positions: Sequence[int] | torch.Tensor | None,
) -> tuple[tuple[int, ...], str]:
    """Validate the row-local donor scope for one arm.

    N10's donor is the latest row without its terminal carrier.  It is never
    safe to infer that donor from every non-target prefix position, so direct
    contexts must declare it explicitly.  N20 uses the complete latest row as
    both target and norm-matched row-mean source.
    """

    if arm_id in {"N00", "N01"}:
        return (), "self"
    if arm_id == "N20" and background_positions is None:
        return target_positions, "target_row_default"
    if background_positions is None:
        raise TechnicalInvalid("N10 requires explicit latest-row background positions")
    normalized = _as_positions(background_positions, label=f"{arm_id} background positions")
    if not normalized:
        raise TechnicalInvalid(f"{arm_id} background positions must be non-empty")
    ordered = tuple(sorted(normalized))
    if arm_id == "N10":
        if len(target_positions) != 1:
            raise TechnicalInvalid("N10 requires one terminal target position")
        terminal = target_positions[0]
        if set(ordered) & set(target_positions):
            raise TechnicalInvalid("N10 background must exclude the terminal target")
        # A latest row is contiguous and its terminal carrier is its final
        # absolute position.  This rejects prompt/image positions even when
        # they happen to have the same width as the row.
        if ordered != tuple(range(ordered[0], terminal)):
            raise TechnicalInvalid("N10 background must be the contiguous latest-row span before terminal")
        return normalized, "latest_row_without_terminal"
    if arm_id == "N20":
        if normalized != target_positions:
            raise TechnicalInvalid("N20 background must equal the complete latest-row target span")
        return normalized, "latest_row"
    raise TechnicalInvalid(f"unsupported background scope for {arm_id}")


def _path_get(root: Any, path: str) -> Any:
    owner = root
    for part in path.split(".") if path else ():
        owner = getattr(owner, part)
    return owner


def _layer_candidates(model: Any, layer_index: int) -> list[tuple[str, Any]]:
    """Return candidate Qwen-like layer aliases without selecting one."""

    candidates: list[tuple[str, Any]] = []
    # These paths cover the HF Qwen3-VL multimodal wrapper and the compact
    # fake models used in CPU tests.  A candidate is selected only after all
    # aliases are compared by object identity.
    roots = (
        "model.language_model",
        "model.model.language_model",
        "language_model",
        "model",
        "",
    )
    for root_name in roots:
        try:
            owner = _path_get(model, root_name)
            layers = getattr(owner, "layers")
        except (AttributeError, TypeError):
            continue
        if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
            continue
        if layer_index >= len(layers):
            continue
        candidates.append((f"{root_name + '.' if root_name else ''}layers[{layer_index}]", layers[layer_index]))
    return candidates


def _is_qwen_decoder_block(module: Any) -> bool:
    if not isinstance(module, torch.nn.Module):
        return False
    if bool(getattr(module, "_coordexp_decoder_layer", False)):
        return True
    name = module.__class__.__name__
    # The production class is Qwen3VLTextDecoderLayer.  Keeping ``Qwen`` in
    # the accepted name also makes a deliberately named fake Qwen block useful
    # in CPU tests without weakening the explicit marker route above.
    return "Qwen" in name and "Decoder" in name


def resolve_qwen_block23(
    model: Any,
    layer_index: int = BLOCK_INDEX,
    *,
    layer_idx: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Resolve exactly one Qwen decoder block at absolute index 23.

    Aliased references to the same block are allowed and recorded.  Distinct
    candidate blocks, generic unmarked layers, an absent index, or a target
    that cannot register a forward hook fail closed before model execution.
    """

    if layer_idx is not None:
        if int(layer_index) != BLOCK_INDEX and int(layer_index) != int(layer_idx):
            raise TechnicalInvalid("layer_index and layer_idx disagree")
        layer_index = int(layer_idx)
    wanted = int(layer_index)
    if wanted < 0:
        raise TechnicalInvalid("decoder layer index must be non-negative")
    candidates = _layer_candidates(model, wanted)
    by_identity: dict[int, list[tuple[str, Any]]] = {}
    for path, module in candidates:
        by_identity.setdefault(id(module), []).append((path, module))
    if len(by_identity) != 1:
        raise TechnicalInvalid(
            f"expected exactly one Qwen decoder block at layer {wanted}, found {len(by_identity)}"
        )
    aliases = next(iter(by_identity.values()))
    module = aliases[0][1]
    if not _is_qwen_decoder_block(module):
        raise TechnicalInvalid(
            f"layer {aliases[0][0]} is not an opted-in Qwen decoder block: {module.__class__.__name__}"
        )
    register = getattr(module, "register_forward_hook", None)
    if not callable(register):
        raise TechnicalInvalid(f"layer {aliases[0][0]} cannot register a returned-output hook")
    receipt = {
        "layer_index": wanted,
        "layer_idx": wanted,
        "module_path": aliases[0][0],
        "module_alias_paths": [path for path, _module in aliases],
        "module_class": module.__class__.__name__,
        "replacement_seam": "returned_layer_output_after_full_block_forward",
    }
    return module, receipt


# Names used by older experiment-local code and by live adapters.
resolve_block23 = resolve_qwen_block23
resolve_decoder_layer = resolve_qwen_block23


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TechnicalInvalid("decoder block output must be a tensor or a tuple/list whose first item is a tensor")


def _replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        # A plain tuple is Qwen's native block return type.  Preserve named
        # tuple subclasses where their constructor accepts positional values.
        if type(output) is tuple:
            return (tensor, *output[1:])
        fields = getattr(output, "_fields", None)
        replace = getattr(output, "_replace", None)
        if isinstance(fields, tuple) and fields and callable(replace):
            # ``collections.namedtuple`` is common in tiny adapters and its
            # positional constructor is not always available to wrappers.
            return replace(**{str(fields[0]): tensor})
        try:
            return type(output)(tensor, *output[1:])
        except (TypeError, ValueError):
            return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TechnicalInvalid("decoder block output cannot be replaced without corrupting auxiliary outputs")


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def norm_matched_background(target: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    """Construct a repeated background mean with target-span mean norm.

    ``target`` and ``background`` are ``[span, hidden]`` activation slices
    from the *same* returned block output.  The result is one vector, not a
    donor tensor, and therefore cannot carry stale prefix positions across
    scalar recomputations.
    """

    if not isinstance(target, torch.Tensor) or not isinstance(background, torch.Tensor):
        raise TechnicalInvalid("norm matching requires tensor activations")
    if target.ndim != 2 or background.ndim != 2:
        raise TechnicalInvalid("norm matching requires [span, hidden] tensors")
    if target.shape[1] != background.shape[1] or target.shape[0] <= 0 or background.shape[0] <= 0:
        raise TechnicalInvalid("norm matching requires non-empty equal-width target/background spans")
    if not target.dtype.is_floating_point or not background.dtype.is_floating_point:
        raise TechnicalInvalid("residual activations must be floating point")
    if not _finite(target) or not _finite(background):
        raise TechnicalInvalid("norm matching requires finite activations")
    background_mean = background.float().mean(dim=0)
    target_mean_norm = target.float().norm(dim=-1).mean()
    background_mean_norm = background_mean.norm()
    if not bool(torch.isfinite(background_mean_norm).item()) or float(background_mean_norm.item()) == 0.0:
        raise TechnicalInvalid("background mean has zero or non-finite norm")
    replacement = background_mean * (target_mean_norm / background_mean_norm)
    if not _finite(replacement):
        raise TechnicalInvalid("norm-matched replacement is non-finite")
    return replacement.to(dtype=target.dtype, device=target.device).detach().clone()


# Compatibility spelling mirroring the older dynamic runner.
norm_matched_mean = norm_matched_background


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class ResidualHookContext(AbstractContextManager["ResidualHookContext"]):
    """One-shot block-23 returned-residual intervention.

    A context is intentionally one-shot: one install, one scalar forward, and
    one cleanup.  The natural runner creates a fresh context for each growing
    no-cache prefix, so runtime activations always belong to that prefix.
    """

    def __init__(
        self,
        model: Any,
        arm_id: ResidualArm | str,
        positions: Sequence[int] | torch.Tensor = (),
        *,
        input_ids: torch.Tensor | None = None,
        expected_sequence_length: int | None = None,
        expected_positions: Sequence[int] | torch.Tensor | None = None,
        background_positions: Sequence[int] | torch.Tensor | None = None,
        replacement: torch.Tensor | None = None,
        persistent: bool = False,
        tolerance: float = NOOP_TOLERANCE,
        step: int | None = None,
        row_index: int | None = None,
    ) -> None:
        if persistent:
            raise TechnicalInvalid("natural scalar residual hooks must not be persistent")
        if replacement is not None:
            raise TechnicalInvalid("runtime residual actuator does not accept a stale replacement tensor")
        self.arm_id, self.positions = _validate_arm(str(arm_id), positions)
        self.background_positions, self.background_position_source = _validate_background_positions(
            self.arm_id,
            self.positions,
            background_positions,
        )
        if expected_positions is not None:
            expected = _as_positions(expected_positions, label="expected residual positions")
            if expected != self.positions:
                raise TechnicalInvalid(
                    f"declared residual positions {self.positions} differ from expected absolute positions {expected}"
                )
        self.model = model
        if self.arm_id == "N00":
            # Native control does not touch the model.  In particular, it is
            # valid to run N00 against an adapter that deliberately does not
            # expose decoder internals.
            self.module = None
            self.layer_receipt = {
                "layer_index": BLOCK_INDEX,
                "layer_idx": BLOCK_INDEX,
                "module_path": None,
                "module_alias_paths": [],
                "module_class": None,
                "replacement_seam": "native_no_hook",
            }
        else:
            if _is_qwen_decoder_block(model) and callable(getattr(model, "register_forward_hook", None)):
                # Allow a live adapter that already resolved the exact block
                # to pass that module directly, while retaining the same
                # explicit Qwen marker and lifecycle checks.
                self.module = cast(torch.nn.Module, model)
                self.layer_receipt = {
                    "layer_index": BLOCK_INDEX,
                    "layer_idx": BLOCK_INDEX,
                    "module_path": "<provided_block>",
                    "module_alias_paths": [],
                    "module_class": model.__class__.__name__,
                    "replacement_seam": "returned_layer_output_after_full_block_forward",
                }
            else:
                self.module, self.layer_receipt = resolve_qwen_block23(model, BLOCK_INDEX)
        self.input_shape: tuple[int, ...] | None = None
        if input_ids is not None:
            if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise TechnicalInvalid("input_ids must have shape [1, sequence] for scalar residual actuation")
            self.input_shape = tuple(int(value) for value in input_ids.shape)
            actual_length = int(input_ids.shape[-1])
            if expected_sequence_length is not None and actual_length != int(expected_sequence_length):
                raise TechnicalInvalid("input_ids sequence length disagrees with expected scalar prefix length")
            expected_sequence_length = actual_length
            self.input_ids_sha256 = _json_hash([int(value) for value in input_ids[0].detach().cpu().tolist()])
        else:
            self.input_ids_sha256 = None
        if expected_sequence_length is not None:
            length = int(expected_sequence_length)
            if length <= 0:
                raise TechnicalInvalid("expected sequence length must be positive")
            if any(position >= length for position in self.positions):
                raise TechnicalInvalid("residual position lies outside expected scalar prefix length")
            if any(position >= length for position in self.background_positions):
                raise TechnicalInvalid("background position lies outside expected scalar prefix length")
            self.expected_sequence_length: int | None = length
        else:
            self.expected_sequence_length = None
        tolerance_value = float(tolerance)
        if tolerance_value < 0 or not bool(torch.isfinite(torch.tensor(tolerance_value)).item()):
            raise TechnicalInvalid("tolerance must be finite and non-negative")
        self.tolerance = tolerance_value
        self.step = None if step is None else int(step)
        self.row_index = None if row_index is None else int(row_index)
        self._handle: Any = None
        self.install_count = 0
        self.cleanup_count = 0
        self.call_count = 0
        self.applied_count = 0
        self.tensor_shape: tuple[int, ...] | None = None
        self.target_max_abs_delta = 0.0
        self.non_target_max_abs_delta = 0.0
        self.pre_target_norms: tuple[float, ...] = ()
        self.post_target_norms: tuple[float, ...] = ()
        self.pre_target_mean_norm = 0.0
        self.post_target_mean_norm = 0.0
        self.background_mean_norm = 0.0
        self.replacement_norm = 0.0
        self.pre_finite = False
        self.post_finite = False
        self.replacement_finite = False
        self.target_positions_exact = False
        self._exception_seen = False

    @property
    def handle(self) -> Any:
        return self._handle

    @property
    def hook_clean(self) -> bool:
        return self._handle is None

    def _runtime_replacement(self, tensor: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        target = tensor[0, destination, :]
        if self.arm_id == "N01":
            replacement = target.detach().clone()
            self.background_mean_norm = 0.0
        else:
            background_indices = list(self.background_positions)
            if any(index < 0 or index >= int(tensor.shape[1]) for index in background_indices):
                raise TechnicalInvalid("declared background position lies outside returned block output")
            if self.arm_id == "N10" and set(background_indices) & set(self.positions):
                raise TechnicalInvalid("N10 runtime donor overlaps terminal target")
            background = tensor[0, torch.tensor(background_indices, dtype=torch.long, device=tensor.device), :]
            replacement_vector = norm_matched_background(target, background)
            replacement = replacement_vector.unsqueeze(0).expand(target.shape[0], -1).clone()
            self.background_mean_norm = float(background.float().mean(dim=0).norm().item())
        if not isinstance(replacement, torch.Tensor) or replacement.shape != target.shape:
            raise TechnicalInvalid("runtime replacement does not match exact target span shape")
        if not _finite(replacement):
            raise TechnicalInvalid("runtime replacement contains non-finite values")
        self.replacement_norm = float(replacement.float().norm(dim=-1).mean().item())
        cast_replacement = replacement.to(device=tensor.device, dtype=tensor.dtype)
        if not _finite(cast_replacement):
            raise TechnicalInvalid("runtime replacement is non-finite after output dtype conversion")
        self.replacement_finite = True
        return cast_replacement

    def _hook(self, _module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        self.call_count += 1
        if self.call_count != 1:
            raise TechnicalInvalid("one residual context must observe exactly one block-23 forward")
        tensor = _first_tensor(output)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
            raise TechnicalInvalid("returned block output must have layout [1, sequence, hidden]")
        if not tensor.dtype.is_floating_point:
            raise TechnicalInvalid("returned residual tensor must be floating point")
        if self.expected_sequence_length is not None and int(tensor.shape[1]) != self.expected_sequence_length:
            raise TechnicalInvalid("returned block sequence length differs from exact scalar prefix length")
        if any(position >= int(tensor.shape[1]) for position in self.positions):
            raise TechnicalInvalid("declared absolute residual position lies outside returned block output")
        if not _finite(tensor):
            raise TechnicalInvalid("returned block residual contains non-finite values")
        self.pre_finite = True
        self.tensor_shape = tuple(int(value) for value in tensor.shape)
        destination = torch.tensor(self.positions, dtype=torch.long, device=tensor.device)
        # The declaration is checked against the exact tensor index set before
        # replacement.  No re-sorting or inferred position is ever applied.
        self.target_positions_exact = tuple(int(value) for value in destination.detach().cpu().tolist()) == self.positions
        if not self.target_positions_exact:
            raise TechnicalInvalid("runtime destination positions differ from declared absolute positions")
        before = tensor[0, destination, :]
        replacement = self._runtime_replacement(tensor, destination)
        updated = tensor.clone()
        updated[0, destination, :] = replacement
        delta = (updated - tensor).detach().abs()
        target_delta = delta[0, destination, :]
        off_target = delta.clone()
        off_target[0, destination, :] = 0
        self.target_max_abs_delta = float(target_delta.max().item()) if target_delta.numel() else 0.0
        self.non_target_max_abs_delta = float(off_target.max().item()) if off_target.numel() else 0.0
        self.pre_target_norms = tuple(float(value) for value in before.float().norm(dim=-1).detach().cpu().tolist())
        after = updated[0, destination, :]
        self.post_target_norms = tuple(float(value) for value in after.float().norm(dim=-1).detach().cpu().tolist())
        self.pre_target_mean_norm = float(before.float().norm(dim=-1).mean().item())
        self.post_target_mean_norm = float(after.float().norm(dim=-1).mean().item())
        self.post_finite = _finite(updated)
        if not self.post_finite:
            raise TechnicalInvalid("returned residual after replacement is non-finite")
        self.applied_count += 1
        return _replace_first_tensor(output, updated)

    def install(self) -> "ResidualHookContext":
        if self.install_count != 0 or self._handle is not None:
            raise TechnicalInvalid("residual hook may be installed exactly once")
        if self.arm_id == "N00":
            # N00 is the native no-op arm and deliberately does not install a
            # hook.  Mark the lifecycle as consumed so re-entry still fails.
            self.install_count = 1
            return self
        try:
            self._handle = self.module.register_forward_hook(self._hook)
        except Exception:
            self._handle = None
            raise
        self.install_count = 1
        return self

    def remove(self) -> None:
        if self._handle is not None:
            handle = self._handle
            self._handle = None
            try:
                handle.remove()
            finally:
                self.cleanup_count += 1
        elif self.install_count == 1 and self.cleanup_count == 0:
            # N00 has no registration handle but still has one logical cleanup
            # event, which keeps lifecycle receipts uniform across arms.
            self.cleanup_count = 1

    def __enter__(self) -> "ResidualHookContext":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._exception_seen = exc_type is not None
        # Cleanup is unconditional and happens before optional attestation so
        # even a model/layout error cannot leave a live hook on the next scalar.
        self.remove()
        if exc_type is None:
            if self.arm_id != "N00" and self.call_count != 1:
                raise TechnicalInvalid("residual hook did not observe exactly one scalar block forward")
            if self.arm_id != "N00" and self.applied_count != 1:
                raise TechnicalInvalid("residual hook did not apply exactly one runtime replacement")
            if self.non_target_max_abs_delta > self.tolerance:
                raise TechnicalInvalid("residual replacement changed a non-target position")
        return False

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "arm_id": self.arm_id,
            "operation": {
                "N00": "native",
                "N01": "terminal_noop_replay",
                "N10": "terminal_norm_matched_mute",
                "N20": "whole_row_norm_matched_mute",
            }[self.arm_id],
            "positions": list(self.positions),
            "declared_target_positions": list(self.positions),
            "target_absolute_positions": list(self.positions),
            "absolute_target_positions": list(self.positions),
            "background_positions": list(self.background_positions),
            "background_absolute_positions": list(self.background_positions),
            "background_position_source": self.background_position_source,
            "target_positions_exact": bool(self.target_positions_exact) if self.arm_id != "N00" else True,
            "layer": dict(self.layer_receipt),
            "input_shape": list(self.input_shape or ()),
            "input_ids_sha256": self.input_ids_sha256,
            "expected_sequence_length": self.expected_sequence_length,
            "tensor_shape": list(self.tensor_shape or ()),
            "install_count": self.install_count,
            "cleanup_count": self.cleanup_count,
            "hook_call_count": self.call_count,
            "hook_applied_count": self.applied_count,
            "hook_removed": self.hook_clean,
            "hook_clean": self.hook_clean,
            "exception_seen": self._exception_seen,
            "target_max_abs_delta": self.target_max_abs_delta,
            "non_target_max_abs_delta": self.non_target_max_abs_delta,
            "pre_target_norms": list(self.pre_target_norms),
            "post_target_norms": list(self.post_target_norms),
            "pre_target_row_norms": list(self.pre_target_norms),
            "post_target_row_norms": list(self.post_target_norms),
            "pre_target_mean_norm": self.pre_target_mean_norm,
            "post_target_mean_norm": self.post_target_mean_norm,
            "pre_norm": self.pre_target_mean_norm,
            "post_norm": self.post_target_mean_norm,
            "background_mean_norm": self.background_mean_norm,
            "replacement_norm": self.replacement_norm,
            "norm_match_abs_delta": abs(self.post_target_mean_norm - self.pre_target_mean_norm),
            "pre_finite": bool(self.pre_finite or self.arm_id == "N00"),
            "replacement_finite": bool(self.replacement_finite or self.arm_id == "N00"),
            "post_finite": bool(self.post_finite or self.arm_id == "N00"),
            "finite_before": bool(self.pre_finite or self.arm_id == "N00"),
            "finite_after": bool(self.post_finite or self.arm_id == "N00"),
            "step": self.step,
            "row_index": self.row_index,
        }


# Descriptive aliases for adapters that prefer the word "actuator".
Block23ResidualHook = ResidualHookContext
ResidualActuationContext = ResidualHookContext
NaturalBoundaryResidualHook = ResidualHookContext


def _context_value(context: Any, name: str, default: Any = None) -> Any:
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)


def _derive_history_scopes(
    context: Any,
    input_ids: torch.Tensor,
    arm_id: ResidualArm,
    requested_positions: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Derive and verify latest-row target/donor positions from event context."""

    row_raw = _context_value(context, "latest_history_row_token_ids", ())
    if isinstance(row_raw, torch.Tensor):
        row = tuple(int(value) for value in row_raw.detach().cpu().reshape(-1).tolist())
    else:
        row = tuple(int(value) for value in row_raw or ())
    if not row:
        raise TechnicalInvalid(f"{arm_id} requires latest_history_row_token_ids in NaturalEventContext")
    prefix_width_raw = _context_value(context, "history_prefix_width", None)
    if prefix_width_raw is None:
        prompt = _context_value(context, "prompt_token_ids", ()) or ()
        history = _context_value(context, "exact_history_token_ids", ()) or ()
        prefix_width_raw = len(prompt) + len(history) - len(row)
    prefix_width = int(prefix_width_raw)
    if prefix_width < 0:
        raise TechnicalInvalid("history row is longer than exact prefix")
    row_positions = tuple(prefix_width + index for index in range(len(row)))
    if row_positions[-1] >= int(input_ids.shape[-1]):
        raise TechnicalInvalid("latest history row lies outside current scalar prefix")
    observed_row = tuple(int(value) for value in input_ids[0, list(row_positions)].detach().cpu().tolist())
    if observed_row != row:
        raise TechnicalInvalid("latest history row token IDs disagree with current scalar prefix")
    expected_target = (row_positions[-1],) if arm_id in {"N01", "N10"} else row_positions
    if requested_positions != expected_target:
        raise TechnicalInvalid(
            f"{arm_id} request positions {requested_positions} differ from exact latest-row positions {expected_target}"
        )
    if arm_id == "N10":
        return expected_target, row_positions[:-1]
    if arm_id == "N20":
        return expected_target, row_positions
    return expected_target, ()


@dataclass
class ResidualActuatorCallback:
    """Factory/callback compatible with the natural-boundary runner."""

    tolerance: float = NOOP_TOLERANCE
    last_context: ResidualHookContext | None = field(default=None, init=False, repr=False)

    def __call__(
        self,
        model: Any,
        *,
        context: Any = None,
        request: Any,
        input_ids: torch.Tensor,
        step: int = 0,
        row_index: int = 0,
    ) -> AbstractContextManager[Any]:
        arm = str(getattr(request, "arm_id", ""))
        positions = getattr(request, "positions", ())
        if arm not in ARM_IDS:
            raise TechnicalInvalid(f"unknown residual request arm {arm!r}")
        if bool(getattr(request, "persistent", False)):
            raise TechnicalInvalid("natural scalar residual hooks must not be persistent")
        if getattr(request, "replacement", None) is not None:
            raise TechnicalInvalid("runtime residual actuator does not accept a stale replacement tensor")
        if arm == "N00":
            if positions:
                raise TechnicalInvalid("N00 request must not carry positions")
            self.last_context = None
            return nullcontext()
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise TechnicalInvalid("input_ids must have shape [1, sequence] for scalar residual actuation")
        typed_arm = cast(ResidualArm, arm)
        requested = _as_positions(positions, label=f"{arm} request positions")
        target_positions, background_positions = _derive_history_scopes(
            context,
            input_ids,
            typed_arm,
            requested,
        )
        context_manager = ResidualHookContext(
            model,
            typed_arm,
            target_positions,
            input_ids=input_ids,
            background_positions=background_positions,
            expected_positions=target_positions,
            expected_sequence_length=int(input_ids.shape[-1]),
            tolerance=self.tolerance,
            step=step,
            row_index=row_index,
        )
        self.last_context = context_manager
        return context_manager

    def install(self, model: Any, **kwargs: Any) -> AbstractContextManager[Any]:
        """Expose the runner's object-with-install callback protocol."""

        return self(model, **kwargs)

    def receipt(self) -> dict[str, Any] | None:
        """Return the most recently created context receipt, if any."""

        return None if self.last_context is None else self.last_context.receipt()


def make_residual_actuator_callback(*, tolerance: float = NOOP_TOLERANCE) -> ResidualActuatorCallback:
    """Build a fresh-runner callback; each invocation returns a new context."""

    return ResidualActuatorCallback(tolerance=float(tolerance))


def make_natural_runner_callback(*, tolerance: float = NOOP_TOLERANCE) -> ResidualActuatorCallback:
    return make_residual_actuator_callback(tolerance=tolerance)


# Short aliases keep the callback seam discoverable for lightweight live
# adapters without making them depend on one particular factory spelling.
ResidualActuator = ResidualActuatorCallback
make_residual_actuator = make_residual_actuator_callback
build_residual_actuator_callback = make_residual_actuator_callback


def install_residual_hook(
    model: Any,
    arm_id: ResidualArm | str,
    positions: Sequence[int] | torch.Tensor = (),
    *,
    input_ids: torch.Tensor | None = None,
    expected_sequence_length: int | None = None,
    tolerance: float = NOOP_TOLERANCE,
    step: int | None = None,
    row_index: int | None = None,
) -> ResidualHookContext:
    """Construct (but do not enter) one per-forward residual context."""

    return ResidualHookContext(
        model,
        arm_id,
        positions,
        input_ids=input_ids,
        expected_sequence_length=expected_sequence_length,
        tolerance=tolerance,
        step=step,
        row_index=row_index,
    )


build_residual_hook = install_residual_hook


__all__ = [
    "ARM_IDS",
    "BLOCK_INDEX",
    "NOOP_TOLERANCE",
    "SCHEMA_VERSION",
    "TechnicalInvalid",
    "ResidualActuatorError",
    "ResidualHookContext",
    "Block23ResidualHook",
    "ResidualActuationContext",
    "NaturalBoundaryResidualHook",
    "ResidualActuatorCallback",
    "ResidualActuator",
    "resolve_qwen_block23",
    "resolve_block23",
    "resolve_decoder_layer",
    "norm_matched_background",
    "norm_matched_mean",
    "install_residual_hook",
    "build_residual_hook",
    "make_residual_actuator_callback",
    "make_residual_actuator",
    "build_residual_actuator_callback",
    "make_natural_runner_callback",
]

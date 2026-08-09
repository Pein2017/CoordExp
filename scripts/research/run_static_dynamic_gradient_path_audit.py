#!/usr/bin/env python3
"""Audit the frozen-model gradient path for the static/dynamic P4 probe.

The audit is deliberately a small, runtime-agnostic harness.  A caller supplies
an explicit forward adapter that returns teacher-forced logits for three
complete rows and the four tensors at the two intervention seams:

* ``image_residual`` and ``matched_background`` (the static block-23 seam),
* ``latest_terminal_carrier`` and ``latest_row_span`` (the dynamic seam).

No optimizer, ``backward`` call, or parameter update is performed.  The two
single-path objectives and a fixed unit-weight sum are differentiated with
``torch.autograd.grad``.  The result is a JSON-serializable receipt containing
the gradients, path checks, runtime identity checks, and grammar/STOP/invalid
token mass diagnostics.  This makes a detached visual state or an LM-head-only
route a technical invalidity rather than a silent null result.

The forward adapter is the only model-specific seam::

    def forward_adapter(batch: AuditBatch) -> ForwardCapture:
        return capture_native_forward(...)

The capture's ``outputs`` mapping must return ``target_logits``,
``uncovered_b_logits``, and ``covered_a_logits``.  Each tensor is aligned to
its corresponding complete row (``[tokens, vocab]`` or ``[1, tokens, vocab]``);
this module does not infer teacher-forcing shifts.  A bare logits mapping is
rejected because it cannot prove native block-23/state provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from typing import Any
import uuid

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SCHEMA_VERSION = "static_dynamic_gradient_path_audit.v2"
REQUIRED_LOGIT_KEYS = ("target_logits", "uncovered_b_logits", "covered_a_logits")
_STATE_NAMES = (
    "image_residual",
    "matched_background",
    "latest_terminal_carrier",
    "latest_row_span",
)


_STATE_ROLES = {
    "image_residual": ("image_span_b_exclusive", False),
    "matched_background": ("background_control", False),
    "latest_terminal_carrier": ("latest_terminal_natural_history", True),
    "latest_row_span": ("latest_row_span_natural_history", True),
}


class GradientPathAuditError(ValueError):
    """Raised for an invalid runtime contract or unsupported audit input."""


def _as_int_list(value: Sequence[int] | Tensor, context: str) -> list[int]:
    """Convert a one-dimensional integer sequence/tensor to Python integers."""

    if isinstance(value, Tensor):
        if value.numel() == 0:
            return []
        if value.ndim != 1:
            raise GradientPathAuditError(f"{context} must be one-dimensional")
        if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise GradientPathAuditError(f"{context} must have an integer dtype")
        return [int(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, (str, bytes)):
        raise GradientPathAuditError(f"{context} must be an integer sequence")
    try:
        values = list(value)
    except TypeError as exc:
        raise GradientPathAuditError(f"{context} must be an integer sequence") from exc
    result: list[int] = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, int):
            raise GradientPathAuditError(f"{context}[{index}] must be an integer")
        result.append(int(item))
    return result


def _flat_int_tensor(value: Tensor, context: str) -> tuple[list[int], tuple[int, ...]]:
    if not isinstance(value, Tensor):
        raise GradientPathAuditError(f"{context} must be a tensor")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise GradientPathAuditError(f"{context} must have an integer dtype")
    return [int(item) for item in value.detach().cpu().reshape(-1).tolist()], tuple(value.shape)


def sha256_int_sequence(values: Sequence[int] | Tensor) -> str:
    """Return the stable hash used by :class:`RuntimeContract` identity checks."""

    return hashlib.sha256(
        json.dumps(_as_int_list(values, "hash values"), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_position_ids(position_ids: Tensor) -> str:
    """Hash both the shape and values of explicit position IDs."""

    values, shape = _flat_int_tensor(position_ids, "position_ids")
    payload = json.dumps(
        {
            "dtype": str(position_ids.dtype),
            "shape": list(shape),
            "values": values,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible provenance value with stable ordering."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_input_ids(input_ids: Tensor) -> str:
    """Hash input IDs in their packed sequence order (batch dimensions ignored)."""

    if not isinstance(input_ids, Tensor):
        raise GradientPathAuditError("input_ids must be a tensor")
    return sha256_int_sequence(input_ids.reshape(-1))


@dataclass(frozen=True)
class StateProvenance:
    """Binding receipt for one tensor retained at a native forward seam.

    A caller must provide this metadata from the same forward that produced the
    tensor.  The audit checks the hashes and object references; it does not
    infer that an arbitrary tensor is an image or natural-history state.
    """

    role: str
    positions: tuple[int, ...]
    positions_sha256: str
    span_provenance: str
    history_sha256: str
    natural_history: bool
    forward_id: str
    model: nn.Module
    block23_module: nn.Module
    block23_module_name: str
    input_ids_sha256: str
    position_ids_sha256: str
    mrope_sha256: str
    storage_data_ptr: int | None = None
    tensor_version: int | None = None
    grad_fn: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return identity metadata without serializing Python objects."""

        return {
            "role": self.role,
            "positions": list(self.positions),
            "positions_sha256": self.positions_sha256,
            "span_provenance": self.span_provenance,
            "history_sha256": self.history_sha256,
            "natural_history": self.natural_history,
            "forward_id": self.forward_id,
            "model_object_id": id(self.model),
            "block23_module_object_id": id(self.block23_module),
            "block23_module_name": self.block23_module_name,
            "input_ids_sha256": self.input_ids_sha256,
            "position_ids_sha256": self.position_ids_sha256,
            "mrope_sha256": self.mrope_sha256,
            "storage_data_ptr": self.storage_data_ptr,
            "tensor_version": self.tensor_version,
            "grad_fn": self.grad_fn,
        }


@dataclass
class GradientSource:
    """A forward-participating block output plus the logical positions to slice."""

    tensor: Tensor
    positions: tuple[int, ...]
    label: str

    def __post_init__(self) -> None:
        self.positions = tuple(int(position) for position in self.positions)


@dataclass
class ForwardCapture:
    """Strict result of one exact native forward used by the P4 audit.

    ``captured_states`` must contain the *same tensor objects* handed to the
    audit batch.  It is intentionally not enough to return logits and four
    arbitrary tensors: the block-23 hook output, model/module identities,
    native input references, and provenance metadata are all checked before an
    objective is differentiated.
    """

    outputs: Mapping[str, Any]
    captured_states: Mapping[str, Tensor]
    model: nn.Module
    block23_module: nn.Module
    block23_module_name: str
    block23_layer_index: int
    forward_id: str
    input_ids: Tensor
    position_ids: Tensor
    prefix_sha256: str
    position_ids_sha256: str
    mrope_sha256: str
    checkpoint_identity: str
    config_identity: str
    wrapper_identity: str
    hook_output: Any
    hook_call_count: int
    hook_cleaned: bool
    state_provenance: Mapping[str, StateProvenance]
    gradient_sources: Mapping[str, tuple[GradientSource, ...]] = field(default_factory=dict)
    expected_hook_call_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe capture identity and storage diagnostics."""

        state_storage: dict[str, dict[str, Any]] = {}
        for name, tensor in self.captured_states.items():
            state_storage[name] = _tensor_identity(tensor)
        return {
            "model_object_id": id(self.model),
            "block23_module_object_id": id(self.block23_module),
            "block23_module_name": self.block23_module_name,
            "block23_layer_index": self.block23_layer_index,
            "forward_id": self.forward_id,
            "prefix_sha256": self.prefix_sha256,
            "position_ids_sha256": self.position_ids_sha256,
            "mrope_sha256": self.mrope_sha256,
            "checkpoint_identity": self.checkpoint_identity,
            "config_identity": self.config_identity,
            "wrapper_identity": self.wrapper_identity,
            "hook_call_count": self.hook_call_count,
            "expected_hook_call_count": self.expected_hook_call_count,
            "hook_cleaned": self.hook_cleaned,
            "gradient_source_counts": {
                name: len(sources) for name, sources in self.gradient_sources.items()
            },
            "state_storage": state_storage,
            "state_provenance": {
                name: provenance.as_dict()
                for name, provenance in self.state_provenance.items()
            },
        }


def capture_native_forward(
    *,
    model: nn.Module,
    block23_module: nn.Module,
    block23_module_name: str,
    input_ids: Tensor,
    position_ids: Tensor,
    forward_call: Callable[[], Mapping[str, Any]],
    state_selector: Callable[[Any], Mapping[str, Tensor]],
    state_provenance: Mapping[str, StateProvenance],
    checkpoint_identity: str,
    config_identity: str,
    wrapper_identity: str,
    mrope_sha256: str,
    block23_layer_index: int = 23,
    forward_id: str | None = None,
    expected_hook_call_count: int = 1,
    gradient_source_selector: Callable[[tuple[Any, ...]], Mapping[str, Sequence[GradientSource]]] | None = None,
) -> ForwardCapture:
    """Run one native forward while retaining differentiable block-23 states.

    This helper owns hook installation and removal.  ``state_selector`` must
    return tensors directly from the hook output (views are allowed); cloning
    or detaching there is rejected by :func:`run_static_dynamic_gradient_path_audit`.
    The adapter still supplies semantic span/history provenance because only
    the real runtime knows its image-token and natural-history boundaries.
    """

    if not isinstance(model, nn.Module) or not isinstance(block23_module, nn.Module):
        raise GradientPathAuditError("native capture requires nn.Module model and block23 module")
    if not callable(forward_call) or not callable(state_selector):
        raise GradientPathAuditError("native capture requires callable forward and state selector")
    capture_id = forward_id or uuid.uuid4().hex
    if expected_hook_call_count <= 0:
        raise GradientPathAuditError("expected hook call count must be positive")
    holder: dict[str, Any] = {"outputs": [], "calls": 0}

    def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        holder["calls"] += 1
        holder["outputs"].append(output)
        return output

    handle = block23_module.register_forward_hook(hook)
    outputs: Mapping[str, Any]
    try:
        outputs = forward_call()
    finally:
        handle.remove()
    hook_outputs = tuple(holder["outputs"])
    selector_input: Any = hook_outputs[0] if expected_hook_call_count == 1 and hook_outputs else hook_outputs
    states = state_selector(selector_input)
    if not isinstance(states, Mapping):
        raise GradientPathAuditError("native state selector must return a mapping")
    raw_sources = gradient_source_selector(hook_outputs) if gradient_source_selector is not None else {}
    if not isinstance(raw_sources, Mapping):
        raise GradientPathAuditError("gradient source selector must return a mapping")
    gradient_sources: dict[str, tuple[GradientSource, ...]] = {}
    for name, sources in raw_sources.items():
        normalized = tuple(sources)
        if any(not isinstance(source, GradientSource) for source in normalized):
            raise GradientPathAuditError(f"gradient sources for {name} are malformed")
        gradient_sources[str(name)] = normalized
    normalized_provenance: dict[str, StateProvenance] = {}
    for name, provenance in state_provenance.items():
        normalized_provenance[name] = replace(
            provenance,
            forward_id=capture_id,
            model=model,
            block23_module=block23_module,
            block23_module_name=block23_module_name,
            input_ids_sha256=_sha256_input_ids(input_ids),
            position_ids_sha256=sha256_position_ids(position_ids),
            mrope_sha256=mrope_sha256,
            storage_data_ptr=_tensor_identity(states[name])["data_ptr"]
            if name in states and isinstance(states[name], Tensor)
            else None,
            tensor_version=_tensor_identity(states[name])["version"]
            if name in states and isinstance(states[name], Tensor)
            else None,
            grad_fn=_tensor_identity(states[name])["grad_fn"]
            if name in states and isinstance(states[name], Tensor)
            else None,
        )
    return ForwardCapture(
        outputs=outputs,
        captured_states=dict(states),
        model=model,
        block23_module=block23_module,
        block23_module_name=block23_module_name,
        block23_layer_index=block23_layer_index,
        forward_id=capture_id,
        input_ids=input_ids,
        position_ids=position_ids,
        prefix_sha256=_sha256_input_ids(input_ids),
        position_ids_sha256=sha256_position_ids(position_ids),
        mrope_sha256=mrope_sha256,
        checkpoint_identity=checkpoint_identity,
        config_identity=config_identity,
        wrapper_identity=wrapper_identity,
        hook_output=hook_outputs[0] if expected_hook_call_count == 1 and hook_outputs else hook_outputs,
        hook_call_count=int(holder["calls"]),
        hook_cleaned=True,
        state_provenance=normalized_provenance,
        gradient_sources=gradient_sources,
        expected_hook_call_count=int(expected_hook_call_count),
    )


@dataclass(frozen=True)
class RuntimeContract:
    """Exact prompt/wrapper/position identity for one audit invocation.

    ``prefix_token_ids`` and ``expected_prefix_sha256`` identify the complete
    static prefix preceding the rows.  The wrapper markers identify the row
    serialization.  In ``commit`` mode the commit token must be the final
    token after ``box_end_token_id``; in ``closed`` mode it must not occur.
    """

    wrapper_mode: str
    prefix_token_ids: tuple[int, ...]
    expected_prefix_sha256: str
    expected_position_ids_sha256: str
    object_ref_start_token_id: int
    object_ref_end_token_id: int
    box_start_token_id: int
    box_end_token_id: int
    coordinate_token_ids: tuple[int, ...] = ()
    coordinate_token_min: int | None = None
    coordinate_token_max: int | None = None
    coordinate_arity: int = 4
    commit_token_id: int | None = None
    position_ids_identity: str = "explicit"
    block23_module_identity: str = "block23.image_residual"
    terminal_token_id: int | None = None
    # The strict P4 seam identity.  Empty values intentionally fail closed;
    # the real runtime adapter must bind these to its checkpoint/config and
    # native span/history receipts rather than inheriting a default.
    checkpoint_identity: str = ""
    config_identity: str = ""
    wrapper_identity: str = ""
    expected_mrope_sha256: str = ""
    expected_image_positions_sha256: str = ""
    expected_background_positions_sha256: str = ""
    expected_terminal_positions_sha256: str = ""
    expected_row_span_positions_sha256: str = ""
    expected_natural_history_sha256: str = ""
    block23_layer_index: int = 23

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        rows: Mapping[str, Sequence[int] | Tensor],
    ) -> None:
        """Fail fast when prompt, wrapper, or position identity is not exact."""

        if self.wrapper_mode not in {"closed", "commit"}:
            raise GradientPathAuditError(
                f"wrapper_mode must be 'closed' or 'commit', got {self.wrapper_mode!r}"
            )
        if self.position_ids_identity != "explicit":
            raise GradientPathAuditError(
                "position_ids_identity must be 'explicit' for the P4 audit"
            )
        if self.block23_layer_index != 23:
            raise GradientPathAuditError("P4 requires zero-based block23_layer_index=23")
        required_identity = {
            "checkpoint_identity": self.checkpoint_identity,
            "config_identity": self.config_identity,
            "wrapper_identity": self.wrapper_identity,
            "expected_mrope_sha256": self.expected_mrope_sha256,
            "expected_image_positions_sha256": self.expected_image_positions_sha256,
            "expected_background_positions_sha256": self.expected_background_positions_sha256,
            "expected_terminal_positions_sha256": self.expected_terminal_positions_sha256,
            "expected_row_span_positions_sha256": self.expected_row_span_positions_sha256,
            "expected_natural_history_sha256": self.expected_natural_history_sha256,
        }
        missing_identity = [name for name, value in required_identity.items() if not value]
        if missing_identity:
            raise GradientPathAuditError(
                "strict P4 identity contract missing: " + ",".join(missing_identity)
            )
        input_values, input_shape = _flat_int_tensor(input_ids, "input_ids")
        position_values, position_shape = _flat_int_tensor(position_ids, "position_ids")
        # Qwen packed MRoPE positions are commonly [4, batch, sequence] while
        # input_ids are [batch, sequence]; the sequence axis is the contract
        # boundary, and the expected hash pins every MRoPE channel exactly.
        if not input_shape or not position_shape or input_shape[-1] != position_shape[-1]:
            raise GradientPathAuditError(
                "input_ids/position_ids sequence shape mismatch: "
                f"{input_shape} != {position_shape}"
            )
        prefix = _as_int_list(self.prefix_token_ids, "prefix_token_ids")
        if input_values != prefix:
            raise GradientPathAuditError("exact prefix token identity mismatch")
        if sha256_int_sequence(prefix) != self.expected_prefix_sha256:
            raise GradientPathAuditError("expected_prefix_sha256 does not match prefix_token_ids")
        if sha256_position_ids(position_ids) != self.expected_position_ids_sha256:
            raise GradientPathAuditError("exact position_ids identity mismatch")
        if not rows:
            raise GradientPathAuditError("at least one complete row is required")
        for row_name, raw_row in rows.items():
            row = _as_int_list(raw_row, f"{row_name} token IDs")
            self._validate_row(row_name, row)

    def _validate_row(self, row_name: str, row: list[int]) -> None:
        if not row:
            raise GradientPathAuditError(f"{row_name} complete row is empty")
        if row[0] != self.object_ref_start_token_id:
            raise GradientPathAuditError(f"{row_name} does not start with object-ref marker")
        if row.count(self.object_ref_start_token_id) != 1:
            raise GradientPathAuditError(f"{row_name} has duplicate object-ref start markers")
        if row.count(self.object_ref_end_token_id) != 1:
            raise GradientPathAuditError(f"{row_name} must contain one object-ref end marker")
        object_end = row.index(self.object_ref_end_token_id)
        if object_end <= 0:
            raise GradientPathAuditError(f"{row_name} has no description before object-ref end")
        if row.count(self.box_start_token_id) != 1:
            raise GradientPathAuditError(f"{row_name} must contain one box-start marker")
        box_start = row.index(self.box_start_token_id)
        if box_start <= object_end:
            raise GradientPathAuditError(f"{row_name} box begins before object-ref closes")
        if row.count(self.box_end_token_id) != 1:
            raise GradientPathAuditError(f"{row_name} must contain one box-end marker")
        box_end = row.index(self.box_end_token_id)
        if box_end <= box_start:
            raise GradientPathAuditError(f"{row_name} box-end precedes box-start")
        if self.coordinate_token_arity_invalid:
            raise GradientPathAuditError("coordinate_arity must be positive")
        if self.coordinate_token_ids:
            coords = _as_int_list(self.coordinate_token_ids, "coordinate_token_ids")
            if len(coords) != self.coordinate_arity:
                raise GradientPathAuditError(
                    f"coordinate_token_ids must contain {self.coordinate_arity} IDs"
                )
            actual = row[box_start + 1 : box_start + 1 + len(coords)]
            if actual != coords:
                raise GradientPathAuditError(
                    f"{row_name} coordinate token identity mismatch: {actual!r} != {coords!r}"
                )
            if box_end != box_start + 1 + len(coords):
                raise GradientPathAuditError(f"{row_name} has unexpected tokens inside box wrapper")
        elif self.coordinate_token_min is not None or self.coordinate_token_max is not None:
            if self.coordinate_token_min is None or self.coordinate_token_max is None:
                raise GradientPathAuditError("coordinate token range requires both min and max")
            if self.coordinate_token_min > self.coordinate_token_max:
                raise GradientPathAuditError("coordinate token range is reversed")
            actual = row[box_start + 1 : box_start + 1 + self.coordinate_arity]
            if len(actual) != self.coordinate_arity or any(
                token < self.coordinate_token_min or token > self.coordinate_token_max
                for token in actual
            ):
                raise GradientPathAuditError(f"{row_name} coordinate token range mismatch")
            if box_end != box_start + 1 + self.coordinate_arity:
                raise GradientPathAuditError(f"{row_name} has unexpected tokens inside box wrapper")
        else:
            raise GradientPathAuditError(
                "coordinate identity contract requires exact IDs or an explicit token range"
            )
        if self.wrapper_mode == "closed":
            if self.commit_token_id is not None and self.commit_token_id in row:
                raise GradientPathAuditError(f"{row_name} closed wrapper contains commit token")
            if box_end != len(row) - 1:
                raise GradientPathAuditError(f"{row_name} closed wrapper must end at box-end")
        else:
            if self.commit_token_id is None:
                raise GradientPathAuditError("commit wrapper requires commit_token_id")
            if row[-1] != self.commit_token_id or row.count(self.commit_token_id) != 1:
                raise GradientPathAuditError(
                    f"{row_name} commit wrapper must end with one commit token"
                )
            if box_end != len(row) - 2:
                raise GradientPathAuditError(f"{row_name} commit token must follow box-end")

    @property
    def coordinate_token_arity_invalid(self) -> bool:
        return isinstance(self.coordinate_arity, bool) or self.coordinate_arity <= 0


@dataclass
class AuditBatch:
    """Inputs at the two intervention seams and the three complete rows."""

    input_ids: Tensor
    position_ids: Tensor
    target_row_token_ids: Sequence[int] | Tensor
    uncovered_b_row_token_ids: Sequence[int] | Tensor
    covered_a_row_token_ids: Sequence[int] | Tensor
    image_residual: Tensor
    matched_background: Tensor
    latest_terminal_carrier: Tensor
    latest_row_span: Tensor
    grammar_token_ids: tuple[int, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    invalid_token_ids: tuple[int, ...] = ()
    # Optional same-image owner regions retained from the identical block-23
    # hook.  They are diagnostics only; omitting them is explicit
    # ``not_measured`` at the orchestration layer and never inferred from the
    # target-B gradient.
    non_target_owner_states: Mapping[str, Tensor] = field(default_factory=dict)
    non_target_owner_region_receipts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def rows(self) -> dict[str, Sequence[int] | Tensor]:
        return {
            "target_b": self.target_row_token_ids,
            "uncovered_b": self.uncovered_b_row_token_ids,
            "covered_a": self.covered_a_row_token_ids,
        }

    def state_tensors(self) -> dict[str, Tensor]:
        return {
            "image_residual": self.image_residual,
            "matched_background": self.matched_background,
            "latest_terminal_carrier": self.latest_terminal_carrier,
            "latest_row_span": self.latest_row_span,
        }

    def non_target_states(self) -> dict[str, Tensor]:
        if not isinstance(self.non_target_owner_states, Mapping):
            raise GradientPathAuditError("non_target_owner_states must be a mapping")
        return {str(name): tensor for name, tensor in self.non_target_owner_states.items()}


def _as_logits(value: Any, key: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise GradientPathAuditError(f"{key} must be a torch.Tensor")
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise GradientPathAuditError(f"{key} batch dimension must be one")
        value = value[0]
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise GradientPathAuditError(f"{key} must have shape [tokens, vocab]")
    if not value.is_floating_point():
        raise GradientPathAuditError(f"{key} must be floating point logits")
    return value


def _extract_logits(outputs: Mapping[str, Any], key: str) -> Tensor:
    value = outputs.get(key)
    if value is None and isinstance(outputs.get("logits"), Mapping):
        value = outputs["logits"].get(key)
    if value is None:
        aliases = {
            "target_logits": "target_b_logits",
            "uncovered_b_logits": "uncovered_logits",
            "covered_a_logits": "covered_logits",
        }
        value = outputs.get(aliases.get(key, ""))
    if value is None:
        raise GradientPathAuditError(f"forward output lacks {key}")
    return _as_logits(value, key)


def _row_nll(logits: Tensor, row: Sequence[int] | Tensor, context: str) -> Tensor:
    target = torch.tensor(_as_int_list(row, context), device=logits.device, dtype=torch.long)
    if logits.shape[0] != target.numel():
        raise GradientPathAuditError(
            f"{context} logits/token length mismatch: {logits.shape[0]} != {target.numel()}"
        )
    if int(target.min()) < 0 or int(target.max()) >= logits.shape[-1]:
        raise GradientPathAuditError(f"{context} token ID is outside the logits vocabulary")
    # Teacher-forced logits are already aligned to the complete row.  Sum all
    # tokens so the objective is explicitly a complete-row NLL.
    return F.cross_entropy(logits, target, reduction="sum")


def _mean_log_probability(logits: Tensor, row: Sequence[int] | Tensor, context: str) -> Tensor:
    target = torch.tensor(_as_int_list(row, context), device=logits.device, dtype=torch.long)
    if logits.shape[0] != target.numel():
        raise GradientPathAuditError(
            f"{context} logits/token length mismatch: {logits.shape[0]} != {target.numel()}"
        )
    if int(target.min()) < 0 or int(target.max()) >= logits.shape[-1]:
        raise GradientPathAuditError(f"{context} token ID is outside the logits vocabulary")
    return F.log_softmax(logits, dim=-1).gather(-1, target[:, None]).mean()


def _tensor_identity(value: Tensor) -> dict[str, Any]:
    """Capture storage/autograd identity without retaining a graph reference."""

    if not isinstance(value, Tensor):
        return {
            "data_ptr": None,
            "version": None,
            "grad_fn": None,
            "requires_grad": False,
            "is_leaf": False,
            "shape": [],
            "dtype": None,
        }
    try:
        version: int | None = int(value._version)
    except RuntimeError:
        version = None
    return {
        "data_ptr": int(value.data_ptr()) if value.numel() else int(value.untyped_storage().data_ptr()),
        "version": version,
        "grad_fn": None if value.grad_fn is None else type(value.grad_fn).__name__,
        "requires_grad": bool(value.requires_grad),
        "is_leaf": bool(value.is_leaf),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _contains_tensor_identity(container: Any, target: Tensor) -> bool:
    """Check that a hook output contains target (or a view sharing its storage)."""

    if isinstance(container, Tensor):
        if container is target:
            return True
        if container.device != target.device or container.dtype != target.dtype:
            return False
        try:
            return int(container.untyped_storage().data_ptr()) == int(
                target.untyped_storage().data_ptr()
            )
        except RuntimeError:
            return False
    if isinstance(container, Mapping):
        return any(_contains_tensor_identity(value, target) for value in container.values())
    if isinstance(container, (tuple, list)):
        return any(_contains_tensor_identity(value, target) for value in container)
    return False


def _resolve_named_module(model: nn.Module, name: str) -> nn.Module | None:
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    return None


def _validate_forward_capture(
    *,
    capture: ForwardCapture,
    model: nn.Module | None,
    runtime: RuntimeContract,
    batch: AuditBatch,
) -> list[str]:
    """Validate every model-bound/native-forward invariant before gradients."""

    reasons: list[str] = []
    if model is None or capture.model is not model:
        reasons.append("capture_model_identity_mismatch")
    if capture.block23_layer_index != runtime.block23_layer_index:
        reasons.append("block23_layer_index_mismatch")
    if capture.block23_module_name != runtime.block23_module_identity:
        reasons.append("block23_module_name_mismatch")
    expected_module = _resolve_named_module(model, runtime.block23_module_identity) if model else None
    if expected_module is None or capture.block23_module is not expected_module:
        reasons.append("block23_module_identity_mismatch")
    if not isinstance(capture.outputs, Mapping):
        reasons.append("forward_output_mapping_invalid")
    if capture.input_ids is not batch.input_ids:
        reasons.append("native_input_ids_reference_mismatch")
    if capture.position_ids is not batch.position_ids:
        reasons.append("native_position_ids_reference_mismatch")
    try:
        input_hash = _sha256_input_ids(batch.input_ids)
        position_hash = sha256_position_ids(batch.position_ids)
    except GradientPathAuditError:
        input_hash = position_hash = ""
    if capture.prefix_sha256 != runtime.expected_prefix_sha256 or capture.prefix_sha256 != input_hash:
        reasons.append("native_prefix_identity_mismatch")
    if (
        capture.position_ids_sha256 != runtime.expected_position_ids_sha256
        or capture.position_ids_sha256 != position_hash
    ):
        reasons.append("native_position_identity_mismatch")
    if capture.mrope_sha256 != runtime.expected_mrope_sha256:
        reasons.append("native_mrope_identity_mismatch")
    if capture.checkpoint_identity != runtime.checkpoint_identity:
        reasons.append("checkpoint_identity_mismatch")
    if capture.config_identity != runtime.config_identity:
        reasons.append("config_identity_mismatch")
    if capture.wrapper_identity != runtime.wrapper_identity:
        reasons.append("wrapper_identity_mismatch")
    if capture.expected_hook_call_count <= 0 or capture.hook_call_count != capture.expected_hook_call_count:
        reasons.append("block23_hook_call_count_mismatch")
    if not capture.hook_cleaned:
        reasons.append("block23_hook_cleanup_missing")
    if not isinstance(capture.captured_states, Mapping):
        reasons.append("captured_state_mapping_invalid")
        return reasons
    if not isinstance(capture.state_provenance, Mapping):
        reasons.append("state_provenance_mapping_invalid")
        return reasons
    expected_positions = {
        "image_residual": runtime.expected_image_positions_sha256,
        "matched_background": runtime.expected_background_positions_sha256,
        "latest_terminal_carrier": runtime.expected_terminal_positions_sha256,
        "latest_row_span": runtime.expected_row_span_positions_sha256,
    }
    for name in _STATE_NAMES:
        batch_state = batch.state_tensors().get(name)
        captured = capture.captured_states.get(name)
        provenance = capture.state_provenance.get(name)
        if not isinstance(captured, Tensor):
            reasons.append(f"captured_state_missing:{name}")
            continue
        if captured is not batch_state:
            reasons.append(f"captured_state_identity_mismatch:{name}")
        if not _contains_tensor_identity(capture.hook_output, captured):
            reasons.append(f"captured_state_not_in_block23_hook_output:{name}")
        identity = _tensor_identity(captured)
        if not captured.requires_grad:
            reasons.append(f"visual_state_detached:{name}")
        if captured.is_leaf or captured.grad_fn is None:
            reasons.append(f"captured_state_not_differentiable:{name}")
        if not isinstance(provenance, StateProvenance):
            reasons.append(f"state_provenance_missing:{name}")
            continue
        expected_role, expected_natural = _STATE_ROLES[name]
        if provenance.role != expected_role:
            reasons.append(f"state_role_mismatch:{name}")
        if bool(provenance.natural_history) != expected_natural:
            reasons.append(f"state_history_kind_mismatch:{name}")
        if provenance.model is not model:
            reasons.append(f"state_model_identity_mismatch:{name}")
        if provenance.block23_module is not capture.block23_module:
            reasons.append(f"state_module_identity_mismatch:{name}")
        if provenance.block23_module_name != runtime.block23_module_identity:
            reasons.append(f"state_module_name_mismatch:{name}")
        if provenance.forward_id != capture.forward_id:
            reasons.append(f"state_forward_identity_mismatch:{name}")
        if provenance.input_ids_sha256 != capture.prefix_sha256:
            reasons.append(f"state_prefix_identity_mismatch:{name}")
        if provenance.position_ids_sha256 != capture.position_ids_sha256:
            reasons.append(f"state_position_identity_mismatch:{name}")
        if provenance.mrope_sha256 != capture.mrope_sha256:
            reasons.append(f"state_mrope_identity_mismatch:{name}")
        try:
            if sha256_int_sequence(provenance.positions) != provenance.positions_sha256:
                reasons.append(f"state_positions_hash_invalid:{name}")
        except GradientPathAuditError:
            reasons.append(f"state_positions_invalid:{name}")
        if provenance.positions_sha256 != expected_positions[name]:
            reasons.append(f"state_positions_identity_mismatch:{name}")
        if provenance.storage_data_ptr != identity["data_ptr"]:
            reasons.append(f"state_storage_identity_mismatch:{name}")
        if provenance.tensor_version != identity["version"]:
            reasons.append(f"state_version_identity_mismatch:{name}")
        if provenance.grad_fn != identity["grad_fn"]:
            reasons.append(f"state_grad_fn_identity_mismatch:{name}")
        if expected_natural and provenance.history_sha256 != runtime.expected_natural_history_sha256:
            reasons.append(f"state_history_identity_mismatch:{name}")
        if not expected_natural and name == "image_residual" and provenance.span_provenance != "b-exclusive-image-span":
            reasons.append("image_span_provenance_mismatch")
        if not expected_natural and name == "matched_background" and provenance.span_provenance != "background-control":
            reasons.append("background_control_provenance_mismatch")
    # Optional non-target owner states must still be the exact tensors emitted
    # by this hook.  The owner-region metadata is validated by the runtime
    # adapter; this layer only enforces graph/object identity and finiteness.
    for owner_id, batch_state in batch.non_target_states().items():
        if not isinstance(batch_state, Tensor):
            reasons.append(f"non_target_state_invalid:{owner_id}")
            continue
        captured = capture.captured_states.get(f"non_target_owner:{owner_id}")
        if captured is None:
            captured = capture.captured_states.get(owner_id)
        if captured is not batch_state:
            reasons.append(f"non_target_state_identity_mismatch:{owner_id}")
        if not _contains_tensor_identity(capture.hook_output, batch_state):
            reasons.append(f"non_target_state_not_in_block23_hook_output:{owner_id}")
        if not batch_state.requires_grad or batch_state.is_leaf or batch_state.grad_fn is None:
            reasons.append(f"non_target_state_not_differentiable:{owner_id}")
        if not bool(torch.isfinite(batch_state).all().item()):
            reasons.append(f"nonfinite_non_target_state:{owner_id}")
    if capture.gradient_sources:
        expected_source_names = [*_STATE_NAMES, *batch.non_target_states().keys()]
        for name in expected_source_names:
            source_key = name
            sources = capture.gradient_sources.get(source_key)
            if sources is None and name not in _STATE_NAMES:
                sources = capture.gradient_sources.get(f"non_target_owner:{name}")
            if not sources:
                reasons.append(f"gradient_source_missing:{name}")
                continue
            for source in sources:
                if not isinstance(source, GradientSource) or not isinstance(source.tensor, Tensor):
                    reasons.append(f"gradient_source_invalid:{name}")
                    continue
                tensor = source.tensor
                if not _contains_tensor_identity(capture.hook_output, tensor):
                    reasons.append(f"gradient_source_not_in_block23_hook_output:{name}")
                if tensor.ndim != 3 or tensor.shape[0] != 1:
                    reasons.append(f"gradient_source_shape_invalid:{name}")
                    continue
                if not source.positions or any(
                    position < 0 or position >= int(tensor.shape[1])
                    for position in source.positions
                ):
                    reasons.append(f"gradient_source_positions_invalid:{name}")
                if not tensor.requires_grad or tensor.is_leaf or tensor.grad_fn is None:
                    reasons.append(f"gradient_source_not_differentiable:{name}")
                if not bool(torch.isfinite(tensor).all().item()):
                    reasons.append(f"nonfinite_gradient_source:{name}")
    return list(dict.fromkeys(reasons))


def _tensor_snapshot(value: Tensor) -> Tensor:
    return value.detach().cpu().clone()


def _same_tensor(value: Tensor, snapshot: Tensor) -> bool:
    return torch.equal(value.detach().cpu(), snapshot)


def _state_snapshot(model: nn.Module | None) -> dict[str, Tensor]:
    if model is None:
        return {}
    return {name: _tensor_snapshot(value) for name, value in model.state_dict().items()}


def _state_unchanged(model: nn.Module | None, snapshot: Mapping[str, Tensor]) -> bool:
    if model is None:
        return True
    current = model.state_dict()
    return set(current) == set(snapshot) and all(
        _same_tensor(value, snapshot[name]) for name, value in current.items()
    )


def _module_training_snapshot(model: nn.Module | None) -> dict[nn.Module, bool]:
    if model is None:
        return {}
    return {module: module.training for module in model.modules()}


def _restore_module_training(snapshot: Mapping[nn.Module, bool]) -> None:
    for module, training in snapshot.items():
        module.train(training)


def _parameter_gradient_snapshot(model: nn.Module | None) -> dict[int, Tensor | None]:
    if model is None:
        return {}
    return {
        id(parameter): None if parameter.grad is None else _tensor_snapshot(parameter.grad)
        for parameter in model.parameters()
    }


def _parameter_grads_unchanged(
    model: nn.Module | None, snapshot: Mapping[int, Tensor | None]
) -> bool:
    if model is None:
        return True
    for parameter in model.parameters():
        before = snapshot.get(id(parameter))
        after = parameter.grad
        if before is None:
            if after is not None:
                return False
        elif after is None or not _same_tensor(after, before):
            return False
    return True


def _gradient(
    objective: Tensor, tensors: Sequence[Tensor], *, retain_graph: bool = True
) -> tuple[list[Tensor | None], str | None]:
    if not objective.requires_grad:
        return [None] * len(tensors), "objective_does_not_require_grad"
    try:
        gradients = torch.autograd.grad(
            objective,
            tuple(tensors),
            allow_unused=True,
            retain_graph=retain_graph,
            create_graph=False,
        )
    except RuntimeError as exc:
        return [None] * len(tensors), str(exc)
    return list(gradients), None


def _gradient_record(
    name: str, gradient: Tensor | None, *, required: bool
) -> tuple[dict[str, Any], str | None]:
    if gradient is None:
        return {
            "name": name,
            "present": False,
            "finite": False,
            "norm": None,
            "max_abs": None,
        }, (f"missing_gradient:{name}" if required else None)
    finite = bool(torch.isfinite(gradient).all().item())
    norm = float(gradient.detach().norm().item()) if finite else math.nan
    max_abs = float(gradient.detach().abs().max().item()) if finite else math.nan
    reason = None if finite else f"nonfinite_gradient:{name}"
    return {
        "name": name,
        "present": True,
        "finite": finite,
        "norm": norm,
        "max_abs": max_abs,
        "shape": list(gradient.shape),
    }, reason


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _objective_receipt(
    name: str,
    objective: Tensor,
    tensors: Mapping[str, Tensor],
    required_names: Sequence[str],
    *,
    gradient_sources: Mapping[str, tuple[GradientSource, ...]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    names = list(tensors)
    if gradient_sources:
        gradients, error = _logical_gradients_from_sources(
            objective, names, gradient_sources
        )
    else:
        gradients, error = _gradient(objective, [tensors[key] for key in names])
    reasons: list[str] = []
    if error is not None and error != "objective_does_not_require_grad":
        reasons.append(f"{name}_gradient_error:{error}")
    records: dict[str, dict[str, Any]] = {}
    for key, gradient in zip(names, gradients):
        record, reason = _gradient_record(key, gradient, required=key in required_names)
        records[key] = record
        if reason is not None:
            reasons.append(f"{name}:{reason}")
    value = float(objective.detach().item()) if objective.numel() == 1 else None
    if value is not None and not math.isfinite(value):
        reasons.append(f"nonfinite_objective:{name}")
    receipt: dict[str, Any] = {
        "objective": name,
        "value": value,
        "finite": value is None or math.isfinite(value),
        "gradients": records,
        "target_control_ratio": _ratio(
            records[required_names[0]]["norm"], records[required_names[1]]["norm"]
        )
        if len(required_names) == 2
        else None,
    }
    return receipt, reasons


def _logical_gradients_from_sources(
    objective: Tensor,
    names: Sequence[str],
    gradient_sources: Mapping[str, tuple[GradientSource, ...]],
) -> tuple[list[Tensor | None], str | None]:
    """Differentiate full hook outputs, then slice logical regions afterward."""

    unique_tensors: list[Tensor] = []
    tensor_indices: dict[int, int] = {}
    for name in names:
        for source in gradient_sources.get(name, ()):
            identity = id(source.tensor)
            if identity not in tensor_indices:
                tensor_indices[identity] = len(unique_tensors)
                unique_tensors.append(source.tensor)
    if not unique_tensors:
        return [None] * len(names), "gradient_sources_absent"
    full_gradients, error = _gradient(objective, unique_tensors)
    logical: list[Tensor | None] = []
    for name in names:
        pieces: list[Tensor] = []
        for source in gradient_sources.get(name, ()):
            gradient = full_gradients[tensor_indices[id(source.tensor)]]
            if gradient is None:
                continue
            pieces.append(gradient[0, list(source.positions), :].reshape(-1))
        logical.append(torch.cat(pieces) if pieces else None)
    return logical, error


def _non_target_gradient_receipt(
    objective: Tensor,
    owner_states: Mapping[str, Tensor],
    *,
    objective_name: str,
    region_receipts: Mapping[str, Mapping[str, Any]],
    gradient_sources: Mapping[str, tuple[GradientSource, ...]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Report optional per-owner first-order effects from one native graph."""

    if not owner_states:
        return {
            "status": "not_measured",
            "reason": "no independently verified non-target owner regions were bound to this forward",
        }, []
    names = list(owner_states)
    if gradient_sources:
        owner_sources = {
            name: gradient_sources.get(name, gradient_sources.get(f"non_target_owner:{name}", ()))
            for name in names
        }
        gradients, error = _logical_gradients_from_sources(objective, names, owner_sources)
    else:
        gradients, error = _gradient(objective, [owner_states[name] for name in names])
    reasons: list[str] = []
    if error is not None and error != "objective_does_not_require_grad":
        reasons.append(f"{objective_name}_non_target_gradient_error:{error}")
    owners: dict[str, Any] = {}
    for owner_id, gradient in zip(names, gradients):
        record, reason = _gradient_record(owner_id, gradient, required=False)
        region = region_receipts.get(owner_id)
        record["region"] = dict(region) if isinstance(region, Mapping) else {
            "status": "not_measured",
            "reason": "owner-region receipt was not attached",
        }
        owners[owner_id] = record
        if reason is not None:
            reasons.append(f"{objective_name}:non_target:{reason}")
    return {"status": "measured", "owners": owners, "owner_count": len(owners)}, reasons


def _mass_receipt(logits: Tensor, token_ids: Sequence[int], name: str) -> dict[str, Any]:
    if not token_ids:
        return {"status": "not_provided", "token_count": 0}
    ids = _as_int_list(token_ids, f"{name}_token_ids")
    if min(ids) < 0 or max(ids) >= logits.shape[-1]:
        raise GradientPathAuditError(f"{name} token ID is outside the logits vocabulary")
    probabilities = F.softmax(logits.detach(), dim=-1)[..., ids].sum(dim=-1)
    finite = bool(torch.isfinite(probabilities).all().item())
    if not finite:
        return {"status": "technical_invalid", "token_count": len(ids), "finite": False}
    return {
        "status": "reported",
        "token_count": len(ids),
        "mean_probability": float(probabilities.mean().item()),
        "max_probability": float(probabilities.max().item()),
        "min_probability": float(probabilities.min().item()),
        "finite": True,
    }


def _lm_head_parameters(
    model: nn.Module | None, explicit: Iterable[Tensor] | None
) -> list[Tensor]:
    if explicit is not None:
        return [tensor for tensor in explicit if tensor.requires_grad]
    if model is None:
        return []
    return [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("lm_head" in name.lower() or "embed_out" in name.lower())
    ]


def _lm_head_receipt(
    objective: Tensor, parameters: Sequence[Tensor], objective_name: str
) -> tuple[dict[str, Any], bool]:
    if not parameters:
        return {"present": False, "finite": True, "norm": None}, False
    gradients, error = _gradient(objective, parameters)
    if error is not None:
        return {
            "present": False,
            "finite": False,
            "norm": None,
            "error": f"{objective_name}:{error}",
        }, True
    present = [gradient for gradient in gradients if gradient is not None]
    if not present:
        return {"present": False, "finite": True, "norm": None}, False
    finite = all(bool(torch.isfinite(gradient).all().item()) for gradient in present)
    norm = (
        float(torch.sqrt(sum(gradient.detach().pow(2).sum() for gradient in present)).item())
        if finite
        else math.nan
    )
    return {"present": True, "finite": finite, "norm": norm}, False


def _invalid_receipt(
    runtime: RuntimeContract, reasons: Sequence[str], *, detail: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "technical_invalid",
        "runtime": runtime.as_dict(),
        "invalid_reasons": list(dict.fromkeys(reasons)),
        "objectives": {},
        "non_target_owner_effects": {
            "status": "not_measured",
            "reason": "gradient forward was technically invalid before owner effects could be measured",
        },
        "path_checks": {
            "optimizer_used": False,
            "model_parameter_mutated": None,
            "parameter_grad_mutated": None,
            "lm_head_only_path": None,
            "visual_state_detached": [],
            "audit_input_mutated": None,
        },
    }
    if detail is not None:
        result["detail"] = detail
    return result


def run_static_dynamic_gradient_path_audit(
    *,
    model: nn.Module | None,
    runtime: RuntimeContract,
    batch: AuditBatch,
    forward_fn: Callable[[AuditBatch], Mapping[str, Any]] | None = None,
    lm_head_parameters: Iterable[Tensor] | None = None,
) -> dict[str, Any]:
    """Run the frozen P4 gradient-path audit and return a receipt.

    The fixed-sum coupled objective is exactly ``target_b_nll + margin_loss``
    with unit weights, where ``margin_loss = mean(log p(covered-A)) - mean(log
    p(uncovered-B))``.  Thus minimizing the fixed sum both lowers target-B NLL
    and increases the uncovered-B versus covered-A complete-row margin.  The
    reported ``margin`` is the opposite, higher-is-B-better quantity.
    """

    try:
        runtime.validate(batch.input_ids, batch.position_ids, batch.rows())
    except GradientPathAuditError as exc:
        return _invalid_receipt(runtime, ["runtime_contract_invalid"], detail=str(exc))

    state_tensors = batch.state_tensors()
    invalid_state_types = [
        name for name, tensor in state_tensors.items() if not isinstance(tensor, Tensor)
    ]
    if invalid_state_types:
        return _invalid_receipt(
            runtime,
            ["invalid_intervention_tensor:" + name for name in invalid_state_types],
        )
    detached = [name for name, tensor in state_tensors.items() if not tensor.requires_grad]
    if detached:
        return _invalid_receipt(
            runtime,
            ["visual_state_detached:" + name for name in detached],
            detail="all four intervention tensors must require gradients",
        )
    try:
        non_target_states = batch.non_target_states()
    except GradientPathAuditError as exc:
        return _invalid_receipt(runtime, ["non_target_owner_state_contract_invalid"], detail=str(exc))
    invalid_non_target = [name for name, tensor in non_target_states.items() if not isinstance(tensor, Tensor)]
    if invalid_non_target:
        return _invalid_receipt(
            runtime,
            ["invalid_non_target_state:" + name for name in invalid_non_target],
        )
    nonfinite_states = [
        name for name, tensor in {**state_tensors, **non_target_states}.items()
        if not bool(torch.isfinite(tensor).all().item())
    ]
    if nonfinite_states:
        return _invalid_receipt(
            runtime,
            ["nonfinite_intervention_state:" + name for name in nonfinite_states],
        )
    model_state_before = _state_snapshot(model)
    parameter_grads_before = _parameter_gradient_snapshot(model)
    input_snapshots = {
        name: _tensor_snapshot(value)
        for name, value in {
            "input_ids": batch.input_ids,
            "position_ids": batch.position_ids,
            **state_tensors,
            **{f"non_target_owner:{name}": value for name, value in non_target_states.items()},
        }.items()
    }
    training_snapshot = _module_training_snapshot(model)
    outputs: Mapping[str, Any]
    capture: ForwardCapture
    capture_invalid_reasons: list[str] = []
    try:
        if model is not None:
            model.eval()
        if forward_fn is not None:
            forward_result = forward_fn(batch)
            if not isinstance(forward_result, ForwardCapture):
                raise GradientPathAuditError(
                    "forward adapter must return ForwardCapture bound to native block23 hook"
                )
            capture = forward_result
            capture_invalid_reasons = _validate_forward_capture(
                capture=capture, model=model, runtime=runtime, batch=batch
            )
            if capture_invalid_reasons:
                raise GradientPathAuditError(
                    "forward capture contract invalid: " + ",".join(capture_invalid_reasons)
                )
            outputs = capture.outputs
        elif model is not None:
            forward_result = model(
                input_ids=batch.input_ids,
                position_ids=batch.position_ids,
                image_residual=batch.image_residual,
                matched_background=batch.matched_background,
                latest_terminal_carrier=batch.latest_terminal_carrier,
                latest_row_span=batch.latest_row_span,
            )
            if not isinstance(forward_result, ForwardCapture):
                raise GradientPathAuditError(
                    "model forward must return ForwardCapture bound to native block23 hook"
                )
            capture = forward_result
            capture_invalid_reasons = _validate_forward_capture(
                capture=capture, model=model, runtime=runtime, batch=batch
            )
            if capture_invalid_reasons:
                raise GradientPathAuditError(
                    "forward capture contract invalid: " + ",".join(capture_invalid_reasons)
                )
            outputs = capture.outputs
        else:
            raise GradientPathAuditError("model and a native ForwardCapture adapter are required")
        if not isinstance(outputs, Mapping):
            raise GradientPathAuditError("forward adapter must return a mapping")
        target_logits = _extract_logits(outputs, "target_logits")
        uncovered_logits = _extract_logits(outputs, "uncovered_b_logits")
        covered_logits = _extract_logits(outputs, "covered_a_logits")
        grammar_logits = _as_logits(outputs.get("grammar_logits", target_logits), "grammar_logits")
        target_nll = _row_nll(target_logits, batch.target_row_token_ids, "target_b")
        uncovered_logprob = _mean_log_probability(
            uncovered_logits, batch.uncovered_b_row_token_ids, "uncovered_b"
        )
        covered_logprob = _mean_log_probability(
            covered_logits, batch.covered_a_row_token_ids, "covered_a"
        )
        margin = uncovered_logprob - covered_logprob
        margin_loss = -margin
        coupled = target_nll + margin_loss
    except (GradientPathAuditError, RuntimeError, TypeError) as exc:
        reasons = capture_invalid_reasons or ["forward_or_objective_invalid"]
        if capture_invalid_reasons:
            reasons = ["forward_capture_contract_invalid", *capture_invalid_reasons]
        return _invalid_receipt(runtime, reasons, detail=str(exc))
    finally:
        _restore_module_training(training_snapshot)

    objective_tensors = {
        "image_residual": batch.image_residual,
        "matched_background": batch.matched_background,
        "latest_terminal_carrier": batch.latest_terminal_carrier,
        "latest_row_span": batch.latest_row_span,
    }
    objectives: dict[str, Any] = {}
    invalid_reasons: list[str] = []
    target_receipt, reasons = _objective_receipt(
        "target_b_complete_row_nll",
        target_nll,
        objective_tensors,
        ("image_residual", "matched_background"),
        gradient_sources=capture.gradient_sources,
    )
    objectives["target_b_complete_row_nll"] = target_receipt
    invalid_reasons.extend(reasons)
    margin_receipt, reasons = _objective_receipt(
        "uncovered_b_vs_covered_a_margin_loss",
        margin_loss,
        objective_tensors,
        ("latest_terminal_carrier", "latest_row_span"),
        gradient_sources=capture.gradient_sources,
    )
    margin_receipt["reported_margin_higher_is_better"] = float(margin.detach().item())
    margin_receipt["reported_uncovered_b_mean_logprob"] = float(uncovered_logprob.detach().item())
    margin_receipt["reported_covered_a_mean_logprob"] = float(covered_logprob.detach().item())
    objectives["uncovered_b_vs_covered_a_margin_loss"] = margin_receipt
    invalid_reasons.extend(reasons)
    coupled_receipt, reasons = _objective_receipt(
        "fixed_sum_coupled",
        coupled,
        objective_tensors,
        tuple(objective_tensors),
        gradient_sources=capture.gradient_sources,
    )
    coupled_receipt["terms"] = {
        "target_b_complete_row_nll_weight": 1.0,
        "uncovered_b_vs_covered_a_margin_loss_weight": 1.0,
    }
    objectives["fixed_sum_coupled"] = coupled_receipt
    invalid_reasons.extend(reasons)

    non_target_effects: dict[str, Any] = {}
    for objective_name, objective in (
        ("target_b_complete_row_nll", target_nll),
        ("uncovered_b_vs_covered_a_margin_loss", margin_loss),
        ("fixed_sum_coupled", coupled),
    ):
        effect_receipt, effect_reasons = _non_target_gradient_receipt(
            objective,
            non_target_states,
            objective_name=objective_name,
            region_receipts=batch.non_target_owner_region_receipts,
            gradient_sources=capture.gradient_sources,
        )
        non_target_effects[objective_name] = effect_receipt
        invalid_reasons.extend(effect_reasons)

    lm_parameters = _lm_head_parameters(model, lm_head_parameters)
    lm_head_receipts: dict[str, Any] = {}
    lm_head_only = False
    for objective_name, objective in (
        ("target_b_complete_row_nll", target_nll),
        ("uncovered_b_vs_covered_a_margin_loss", margin_loss),
        ("fixed_sum_coupled", coupled),
    ):
        receipt, failed = _lm_head_receipt(objective, lm_parameters, objective_name)
        lm_head_receipts[objective_name] = receipt
        if failed:
            invalid_reasons.append(f"lm_head_gradient_invalid:{objective_name}")
        if receipt.get("finite") is False:
            invalid_reasons.append(f"nonfinite_lm_head_gradient:{objective_name}")
        expected = objectives[objective_name]["gradients"]
        expected_names = (
            ("image_residual", "matched_background")
            if objective_name == "target_b_complete_row_nll"
            else ("latest_terminal_carrier", "latest_row_span")
            if objective_name == "uncovered_b_vs_covered_a_margin_loss"
            else tuple(objective_tensors)
        )
        if (
            receipt.get("present")
            and receipt.get("norm", 0.0) not in (None, 0.0)
            and any(not expected[name]["present"] for name in expected_names)
        ):
            lm_head_only = True
            invalid_reasons.append(f"lm_head_only_path:{objective_name}")

    model_parameter_mutated = not _state_unchanged(model, model_state_before)
    parameter_grad_mutated = not _parameter_grads_unchanged(model, parameter_grads_before)
    input_mutated = any(
        not _same_tensor(value, input_snapshots[name])
        for name, value in {
            "input_ids": batch.input_ids,
            "position_ids": batch.position_ids,
            **state_tensors,
            **{f"non_target_owner:{name}": value for name, value in non_target_states.items()},
        }.items()
    )
    if model_parameter_mutated:
        invalid_reasons.append("model_parameter_mutated")
    if parameter_grad_mutated:
        invalid_reasons.append("parameter_grad_mutated")
    if input_mutated:
        invalid_reasons.append("audit_input_mutated")

    try:
        mass = {
            "grammar": _mass_receipt(grammar_logits, batch.grammar_token_ids, "grammar"),
            "stop": _mass_receipt(grammar_logits, batch.stop_token_ids, "stop"),
            "invalid": _mass_receipt(grammar_logits, batch.invalid_token_ids, "invalid"),
        }
    except GradientPathAuditError as exc:
        mass = {"status": "technical_invalid", "detail": str(exc)}
        invalid_reasons.append("grammar_stop_invalid_mass_invalid")
    if isinstance(mass, Mapping) and mass.get("status") == "technical_invalid":
        invalid_reasons.append("grammar_stop_invalid_mass_invalid")
    elif isinstance(mass, Mapping):
        for mass_name in ("grammar", "stop", "invalid"):
            if isinstance(mass.get(mass_name), Mapping) and mass[mass_name].get("status") == "technical_invalid":
                invalid_reasons.append(f"{mass_name}_mass_nonfinite")
            elif isinstance(mass.get(mass_name), Mapping) and mass[mass_name].get("status") == "not_provided":
                invalid_reasons.append(f"missing_{mass_name}_mass_contract")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "technical_invalid" if invalid_reasons else "valid",
        "runtime": runtime.as_dict(),
        "objectives": objectives,
        "lm_head": lm_head_receipts,
        "non_target_owner_effects": non_target_effects,
        "grammar_stop_invalid_mass": mass,
        "path_checks": {
            "optimizer_used": False,
            "lm_head_only_path": lm_head_only,
            "visual_state_detached": detached,
            "model_parameter_mutated": model_parameter_mutated,
            "parameter_grad_mutated": parameter_grad_mutated,
            "audit_input_mutated": input_mutated,
            "block23_module_identity": runtime.block23_module_identity,
            "block23_layer_index": runtime.block23_layer_index,
            "position_ids_identity": runtime.position_ids_identity,
            "forward_capture": capture.as_dict(),
        },
        "invalid_reasons": list(dict.fromkeys(invalid_reasons)),
    }


def write_audit_receipt(receipt: Mapping[str, Any], path: str) -> None:
    """Write one deterministic JSON receipt for an external runtime adapter."""

    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "AuditBatch",
    "ForwardCapture",
    "GradientSource",
    "GradientPathAuditError",
    "RuntimeContract",
    "SCHEMA_VERSION",
    "StateProvenance",
    "capture_native_forward",
    "run_static_dynamic_gradient_path_audit",
    "sha256_json",
    "sha256_int_sequence",
    "sha256_position_ids",
    "write_audit_receipt",
]

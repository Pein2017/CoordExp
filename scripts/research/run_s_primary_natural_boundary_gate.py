#!/usr/bin/env python3
"""Run the S-primary natural-boundary actuator gate.

This driver is deliberately small at the live-runtime boundary.  The old
owner-interface runner remains the owner of config/H0/panel resolution and of
the HF session loader; this module only binds its S ``gt:5001:15`` event to the
new pre-opener release contract.  A scalar call is rebuilt from the complete
multimodal runtime on every step.  Attention-mask and residual implementations
are caller-owned callbacks, so this file does not choose or implement either
operator.

The public API is usable with a tiny CPU runtime in tests.  The CLI resolves
the existing S config/checkpoint/panel/cohort/H0 and executes the same path;
it is not a schema-only or dry-run command.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Literal

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_natural_boundary_routing_history_probe as natural  # noqa: E402
from scripts.research import run_static_dynamic_owner_interface_experiment as legacy  # noqa: E402


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
SCHEMA_VERSION = "s_primary_natural_boundary_gate.v1"
CHECKPOINT = "S"
EVENT_ID = "gt:5001:15"
DEFAULT_CONFIG = legacy.CHECKPOINTS[CHECKPOINT]["config"]
DEFAULT_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
    "human-refined-13.geo_sorted_xy.coord.jsonl"
)
DEFAULT_COHORT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/cohort/"
    "s-step2444-final-support.json"
)
DEFAULT_H0_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/h0"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    f"{UNIT_ID}/s-gate"
)
K_ARMS = ("K00", "K01", "K10", "K11", "K12", "K13", "K14T", "K14B")
RESIDUAL_ARM_IDS = ("N00", "N01", "N10", "N20")
HISTORY_ARM_IDS = ("H00", "H10", "H20")
# Keep the declared unit matrix visible in the public defaults.  A live CLI
# invocation still fails closed for callback-owned arms unless the caller
# supplies the corresponding actuator seams; it must never silently run them
# as native N00/K00.
DEFAULT_ARMS = K_ARMS + RESIDUAL_ARM_IDS + HISTORY_ARM_IDS
RESIDUAL_ARMS = frozenset(RESIDUAL_ARM_IDS)
HISTORY_ARMS = frozenset(HISTORY_ARM_IDS)
ATTENTION_ARMS = frozenset(K_ARMS[1:] + HISTORY_ARM_IDS)
DECLARED_ARMS = frozenset(K_ARMS + RESIDUAL_ARM_IDS + HISTORY_ARM_IDS)
FULL_LOGIT_NATIVE_ARMS = frozenset({"N00", "K00"})
K14_ARMS = frozenset({"K14T", "K14B"})
MODEL_KWARG_NAMES = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "labels",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "cache_position",
        "rope_deltas",
        "use_cache",
        "return_dict",
        "logits_to_keep",
        "num_logits_to_keep",
        "output_attentions",
        "output_hidden_states",
    }
)
FORBIDDEN_CACHE_FIELDS = frozenset({"past_key_values", "key_value_cache", "cache_position"})


class GateTechnicalInvalid(RuntimeError):
    """Raised for a mechanical gate failure; never converted to a null."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GateTechnicalInvalid(f"value is not canonical finite JSON: {exc}") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).expanduser().resolve(strict=True).read_bytes()).hexdigest()
    except OSError as exc:
        raise GateTechnicalInvalid(f"cannot hash file {path}: {exc}") from exc


def _resolve_fresh_gate_output_root(value: str | Path, *, label: str) -> Path:
    """Resolve one launch output root without following a user redirect.

    The v4 sealer binds a concrete, absolute path.  Keep the producer and
    consumer identities byte-for-byte identical and reject symlinks anywhere
    in the requested path before the model/session loader is touched.
    """

    # Normalize lexical ``.``/``..`` components without following symlinks.
    candidate = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    # ``resolve(strict=False)`` is intentionally used only after checking the
    # lexical path components: it gives us the producer's canonical absolute
    # identity while the checks below prevent a symlink redirect.
    cursor = candidate
    while True:
        if cursor.is_symlink():
            raise GateTechnicalInvalid(f"{label} must be a non-symlink path: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise GateTechnicalInvalid(f"cannot resolve {label}: {candidate}: {exc}") from exc
    if resolved != candidate:
        raise GateTechnicalInvalid(
            f"{label} resolves through a symlink and is not an exact non-symlink identity: {candidate}"
        )
    if candidate.exists() or candidate.is_symlink():
        raise GateTechnicalInvalid(f"{label} must be absent and unused at launch: {resolved}")
    return resolved


def _load_pre_gpu_identity(
    path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the immutable pre-GPU receipt before constructing a model."""

    receipt_path = Path(path).expanduser().resolve()
    if not receipt_path.is_file():
        raise GateTechnicalInvalid(f"pre-GPU receipt is not a regular file: {receipt_path}")
    try:
        document = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateTechnicalInvalid(f"cannot read pre-GPU receipt {receipt_path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise GateTechnicalInvalid("pre-GPU receipt must be a JSON object")
    # Replay the sealer's complete contract/census/support/source/test/mask
    # validator before the legacy loader is even constructed.  Repeating only
    # the receipt's copied self/code hashes would allow a semantically invalid
    # launch receipt to cross the model-load boundary.
    try:
        from scripts.research import seal_natural_boundary_pre_gpu_receipt as sealer

        sealer.validate_receipt(document, receipt_path=receipt_path)
    except Exception as exc:
        raise GateTechnicalInvalid(f"pre-GPU receipt semantic validation failed: {exc}") from exc
    if document.get("unit_id") != UNIT_ID:
        raise GateTechnicalInvalid("pre-GPU receipt unit identity differs from S gate")
    top_level_checkpoint = document.get("checkpoint")
    if top_level_checkpoint is not None and top_level_checkpoint != CHECKPOINT:
        raise GateTechnicalInvalid("pre-GPU receipt top-level checkpoint differs from S gate")
    event = document.get("event_binding")
    model_identity = document.get("model_identity")
    if not isinstance(event, Mapping) or not isinstance(model_identity, Mapping):
        raise GateTechnicalInvalid("pre-GPU receipt lacks canonical event/model identity")
    if event.get("checkpoint") != CHECKPOINT or model_identity.get("checkpoint") != CHECKPOINT:
        raise GateTechnicalInvalid("pre-GPU receipt canonical checkpoint identities differ from S gate")
    if event.get("event_id") not in {EVENT_ID, f"S/{EVENT_ID}"}:
        raise GateTechnicalInvalid("pre-GPU receipt event binding is not S gt:5001:15")
    self_hash = document.get("self_sha256")
    if self_hash is not None:
        body = dict(document)
        body.pop("self_sha256", None)
        if self_hash != sha256_json(body):
            raise GateTechnicalInvalid("pre-GPU receipt self hash mismatch")
    code_identity = document.get("code_identity")
    source_files = document.get("source_files")
    if not isinstance(code_identity, Mapping) or not isinstance(source_files, Mapping):
        raise GateTechnicalInvalid("pre-GPU receipt lacks source/code identity hashes")
    declared = code_identity.get("sha256")
    if not isinstance(declared, Mapping) or not declared:
        raise GateTechnicalInvalid("pre-GPU receipt code_identity.sha256 is empty")
    observed: dict[str, str] = {}
    for role, ref in source_files.items():
        if not isinstance(ref, Mapping) or not ref.get("path"):
            raise GateTechnicalInvalid(f"pre-GPU receipt source file {role!r} lacks a path")
        source_path = Path(str(ref["path"])).expanduser().resolve()
        digest = sha256_file(source_path)
        if ref.get("sha256") != digest:
            raise GateTechnicalInvalid(f"pre-GPU receipt source file hash drifted for {role!r}")
        observed[str(role)] = digest
    if dict(declared) != observed:
        raise GateTechnicalInvalid("pre-GPU receipt code_identity hashes drifted")
    identity: dict[str, Any] = {
        "pre_gpu_receipt_path": str(receipt_path),
        "pre_gpu_receipt_sha256": sha256_file(receipt_path),
        "pre_gpu_receipt_self_sha256": self_hash,
        "code_hashes": dict(observed),
        "source_code_hashes": dict(observed),
    }
    # v4 receipts bind the one future gate destination.  Keep this identity in
    # the runtime receipt even for a programmatic no-output call; enforce the
    # requested producer/consumer destination below when one is supplied.
    gate_binding = document.get("gate_output_binding")
    authorized = document.get("authorized_gate_output_root")
    bound_root = document.get("gate_output_root")
    has_gate_binding = isinstance(gate_binding, Mapping) and isinstance(gate_binding.get("path"), str)
    if has_gate_binding:
        if authorized != gate_binding["path"] or bound_root != gate_binding["path"]:
            raise GateTechnicalInvalid("pre-GPU receipt gate output root aliases drifted")
        receipt_root = Path(str(gate_binding["path"])).expanduser()
        if not receipt_root.is_absolute():
            raise GateTechnicalInvalid("receipt-bound gate output root must be absolute")
        try:
            receipt_root_resolved = _resolve_fresh_gate_output_root(
                receipt_root,
                label="receipt-bound gate output root",
            ) if output_root is not None else receipt_root.resolve(strict=False)
        except GateTechnicalInvalid as exc:
            # A used receipt root should report the launch collision, while a
            # symlink/redirect remains a technical identity failure.
            if receipt_root.exists() and not receipt_root.is_symlink():
                raise GateTechnicalInvalid(
                    f"receipt-bound gate output root must be absent and unused at launch: {receipt_root}"
                ) from exc
            raise
        identity.update(
            {
                "gate_output_root": str(receipt_root_resolved),
                "authorized_gate_output_root": str(receipt_root_resolved),
                "gate_output_binding": dict(gate_binding),
            }
        )
    if output_root is not None:
        if not has_gate_binding:
            raise GateTechnicalInvalid("pre-GPU receipt lacks the authorized gate output binding")
        requested = _resolve_fresh_gate_output_root(output_root, label="requested gate output root")
        if os.fsencode(str(requested)) != os.fsencode(str(receipt_root_resolved)):
            raise GateTechnicalInvalid(
                "requested gate output root does not exactly match the receipt-bound s-gt5001-live-gate-v3 root"
            )
    return identity


def sha256_tensor(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise GateTechnicalInvalid("tensor hash requires a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    return sha256_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape), "values": tensor.tolist()})


def _token_ids(values: Sequence[int] | torch.Tensor, *, label: str, allow_empty: bool = False) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        values = values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise GateTechnicalInvalid(f"{label} must be a sequence of integer token IDs")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateTechnicalInvalid(f"{label} contains an invalid token ID")
        result.append(int(value))
    if not result and not allow_empty:
        raise GateTechnicalInvalid(f"{label} must not be empty")
    return tuple(result)


@dataclass(frozen=True)
class UnseededBoundary:
    """The exact one-token subtraction from an old seeded event prefix."""

    seeded_prefix_token_ids: tuple[int, ...]
    natural_prefix_token_ids: tuple[int, ...]
    opener_token_id: int
    removed_token_count: int = 1

    def __post_init__(self) -> None:
        if self.removed_token_count != 1:
            raise GateTechnicalInvalid("natural-boundary reconstruction must remove exactly one token")
        if not self.seeded_prefix_token_ids or not self.natural_prefix_token_ids:
            raise GateTechnicalInvalid("seeded and natural prefixes must be non-empty")
        if self.seeded_prefix_token_ids[-1] != self.opener_token_id:
            raise GateTechnicalInvalid("seeded prefix does not end with object_ref_start")
        if self.natural_prefix_token_ids != self.seeded_prefix_token_ids[:-1]:
            raise GateTechnicalInvalid("natural prefix is not the exact one-token seeded-prefix subtraction")
        if self.natural_prefix_token_ids[-1] == self.opener_token_id:
            raise GateTechnicalInvalid("natural prefix still ends with object_ref_start")

    @property
    def seeded_prefix_sha256(self) -> str:
        return sha256_json(list(self.seeded_prefix_token_ids))

    @property
    def natural_prefix_sha256(self) -> str:
        return sha256_json(list(self.natural_prefix_token_ids))

    def receipt(self) -> dict[str, Any]:
        return {
            "seeded_prefix_token_ids": list(self.seeded_prefix_token_ids),
            "seeded_prefix_sha256": self.seeded_prefix_sha256,
            "natural_prefix_token_ids": list(self.natural_prefix_token_ids),
            "natural_prefix_sha256": self.natural_prefix_sha256,
            "opener_token_id": int(self.opener_token_id),
            "removed_token_count": 1,
            "removal": "exact_trailing_object_ref_start",
        }


def remove_exact_preseeded_opener(
    seeded_prefix_token_ids: Sequence[int] | torch.Tensor,
    *,
    opener_token_id: int,
) -> UnseededBoundary:
    """Remove exactly the old caller-seeded opener and nothing else."""

    seeded = _token_ids(seeded_prefix_token_ids, label="seeded_prefix_token_ids")
    if isinstance(opener_token_id, bool) or not isinstance(opener_token_id, int) or opener_token_id < 0:
        raise GateTechnicalInvalid("opener_token_id must be a non-negative integer")
    if seeded[-1] != int(opener_token_id):
        raise GateTechnicalInvalid(
            "pre-seeded event prefix must end with exactly one object_ref_start token"
        )
    natural_prefix = seeded[:-1]
    return UnseededBoundary(seeded, natural_prefix, int(opener_token_id))


# Friendly aliases for callers/tests that use the unit's terminology.
remove_seeded_opener = remove_exact_preseeded_opener
reconstruct_natural_boundary = remove_exact_preseeded_opener


def _device_of(value: Any) -> torch.device:
    if isinstance(value, torch.device):
        return value
    try:
        return torch.device(str(value))
    except (TypeError, RuntimeError) as exc:
        raise GateTechnicalInvalid(f"invalid model device {value!r}") from exc


def _model_device(model: Any) -> torch.device:
    candidate = getattr(model, "model_device", None)
    if candidate is not None:
        candidate = candidate() if callable(candidate) else candidate
        return _device_of(candidate)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise GateTechnicalInvalid("live model has no discoverable device") from exc


def _require_preload_cuda_visibility() -> dict[str, Any]:
    """Require one explicit physical CUDA token before the HF loader runs."""

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        raise GateTechnicalInvalid(
            "CUDA_VISIBLE_DEVICES must be explicitly set to one numeric device before model load"
        )
    tokens = [token.strip() for token in raw.split(",")]
    if len(tokens) != 1 or not tokens[0] or tokens[0] == "-1" or re.fullmatch(r"[0-9]+", tokens[0]) is None:
        raise GateTechnicalInvalid(
            "CUDA_VISIBLE_DEVICES must expose exactly one numeric device before model load"
        )
    return {
        "raw": raw,
        "tokens": [tokens[0]],
        "selected_physical_device": tokens[0],
    }


def _attest_postload_cuda_device(binding: LiveRuntimeBinding) -> dict[str, Any]:
    """Bind every model parameter and persistent buffer to logical ``cuda:0``."""

    model = binding.model
    try:
        parameters = list(model.parameters())
    except (AttributeError, TypeError) as exc:
        raise GateTechnicalInvalid("loaded S model parameters are not inspectable") from exc
    if not parameters:
        raise GateTechnicalInvalid("loaded S model has no parameters for CUDA device attestation")
    parameter_devices = {str(torch.device(str(parameter.device))) for parameter in parameters}
    persistent_buffers: list[Any] = []
    try:
        for _module_name, module in model.named_modules():
            non_persistent = set(getattr(module, "_non_persistent_buffers_set", set()))
            for name, value in getattr(module, "_buffers", {}).items():
                if value is not None and name not in non_persistent:
                    persistent_buffers.append(value)
    except (AttributeError, TypeError) as exc:
        raise GateTechnicalInvalid("loaded S model persistent buffers are not inspectable") from exc
    buffer_devices = {str(torch.device(str(buffer.device))) for buffer in persistent_buffers}
    all_devices = parameter_devices | buffer_devices
    floating_parameters = [parameter for parameter in parameters if parameter.is_floating_point()]
    nonfloating_parameters = [parameter for parameter in parameters if not parameter.is_floating_point()]
    floating_buffers = [buffer for buffer in persistent_buffers if buffer.is_floating_point()]
    nonfloating_buffers = [buffer for buffer in persistent_buffers if not buffer.is_floating_point()]
    floating_parameter_dtypes = {str(parameter.dtype) for parameter in floating_parameters}
    floating_buffer_dtypes = {str(buffer.dtype) for buffer in floating_buffers}
    all_floating_dtypes = floating_parameter_dtypes | floating_buffer_dtypes
    model_device = binding.device
    if model_device.type != "cuda" or model_device.index != 0:
        raise GateTechnicalInvalid(
            f"loaded S model is not on logical cuda:0: {model_device}"
        )
    if all_devices != {"cuda:0"}:
        raise GateTechnicalInvalid(
            f"loaded S model parameters/persistent buffers span forbidden devices: {sorted(all_devices)}"
        )
    if nonfloating_parameters:
        raise GateTechnicalInvalid("loaded S model contains non-floating parameters")
    if all_floating_dtypes != {"torch.float32"}:
        raise GateTechnicalInvalid(
            f"loaded S model floating parameters/persistent buffers use forbidden dtypes: {sorted(all_floating_dtypes)}"
        )
    return {
        "status": "validated",
        "passed": True,
        "logical_model_device": str(model_device),
        "logical_device_index": int(model_device.index),
        "parameter_tensor_count": len(parameters),
        "parameter_numel": sum(int(parameter.numel()) for parameter in parameters),
        "parameter_device_set": sorted(parameter_devices),
        "floating_parameter_tensor_count": len(floating_parameters),
        "floating_parameter_numel": sum(int(parameter.numel()) for parameter in floating_parameters),
        "floating_parameter_dtype_set": sorted(floating_parameter_dtypes),
        "nonfloating_parameter_tensor_count": len(nonfloating_parameters),
        "persistent_buffer_tensor_count": len(persistent_buffers),
        "persistent_buffer_numel": sum(int(buffer.numel()) for buffer in persistent_buffers),
        "persistent_buffer_device_set": sorted(buffer_devices),
        "floating_persistent_buffer_tensor_count": len(floating_buffers),
        "floating_persistent_buffer_numel": sum(int(buffer.numel()) for buffer in floating_buffers),
        "floating_persistent_buffer_dtype_set": sorted(floating_buffer_dtypes),
        "nonfloating_persistent_buffer_tensor_count": len(nonfloating_buffers),
        "nonfloating_persistent_buffer_numel": sum(int(buffer.numel()) for buffer in nonfloating_buffers),
        "nonfloating_persistent_buffer_dtype_set": sorted({str(buffer.dtype) for buffer in nonfloating_buffers}),
        "all_floating_tensor_count": len(floating_parameters) + len(floating_buffers),
        "all_floating_numel": sum(int(tensor.numel()) for tensor in floating_parameters + floating_buffers),
        "all_floating_dtype_set": sorted(all_floating_dtypes),
        "all_tensor_device_set": sorted(all_devices),
    }


def _tensor_to_device(value: Any, device: torch.device) -> Any:
    return value.to(device=device) if isinstance(value, torch.Tensor) else value


def _position_receipt(position_ids: torch.Tensor, *, sequence_length: int) -> dict[str, Any]:
    if not isinstance(position_ids, torch.Tensor) or position_ids.ndim != 3:
        raise GateTechnicalInvalid("exact MRoPE position_ids must be rank-3 [3,1,S] or [4,1,S]")
    if int(position_ids.shape[0]) not in {3, 4} or int(position_ids.shape[1]) != 1:
        raise GateTechnicalInvalid("exact MRoPE position_ids must have shape [3,1,S] or [4,1,S]")
    if int(position_ids.shape[-1]) != int(sequence_length):
        raise GateTechnicalInvalid("exact MRoPE position_ids sequence axis differs from input_ids")
    if position_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise GateTechnicalInvalid("exact MRoPE position_ids must use an integer dtype")
    return {
        "shape": list(position_ids.shape),
        "dtype": str(position_ids.dtype),
        "sha256": sha256_tensor(position_ids),
    }


def _grid_receipt(grid: torch.Tensor) -> dict[str, Any]:
    if not isinstance(grid, torch.Tensor):
        raise GateTechnicalInvalid("image_grid_thw must be a tensor")
    if grid.ndim == 1 and grid.numel() == 3:
        normalized = grid.reshape(1, 3)
    elif grid.ndim == 2 and tuple(grid.shape[1:]) == (3,):
        normalized = grid
    else:
        raise GateTechnicalInvalid("image_grid_thw must have shape [3] or [N,3]")
    if normalized.numel() == 0 or not bool((normalized > 0).all().item()):
        raise GateTechnicalInvalid("image_grid_thw must contain positive dimensions")
    return {
        "shape": list(normalized.shape),
        "dtype": str(normalized.dtype),
        "sha256": sha256_tensor(normalized),
        "values": normalized.detach().cpu().tolist(),
    }


def _strict_model_call(model: Any, payload: Mapping[str, Any]) -> Any:
    """Call the model without silently filtering a possibly ignored kwarg."""

    try:
        target = model.forward if callable(getattr(model, "forward", None)) else model
        signature = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise GateTechnicalInvalid(f"cannot inspect live model kwargs: {exc}") from exc
    parameters = signature.parameters
    has_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    unknown = set(payload) - set(parameters)
    if has_var_kwargs:
        unknown = set(payload) - MODEL_KWARG_NAMES
    if unknown:
        raise GateTechnicalInvalid(f"model kwargs would be ignored or are unsupported: {sorted(unknown)}")
    for key in FORBIDDEN_CACHE_FIELDS:
        if payload.get(key) is not None:
            raise GateTechnicalInvalid(f"natural scalar forward received forbidden cache kwarg {key}")
    try:
        return model(**dict(payload))
    except TypeError as exc:
        raise GateTechnicalInvalid(f"model rejected exact scalar kwargs (nothing was filtered): {exc}") from exc


def _forced_math_model_call(model: Any, payload: Mapping[str, Any]) -> Any:
    """Call one exact scalar forward under the frozen deterministic SDPA math backend.

    Native K00 can otherwise enter SDPA's fused causal/GQA route while an
    explicit K01 4-D mask enters a different kernel/GQA route.  Selecting the
    MATH backend is a kernel-selection repair only: it leaves the attention
    operator, mask values, token/prefix, positions, and checkpoint unchanged.
    The context is entered around every actual model call, including K14's
    native mass-baseline call, so no hidden forward escapes the receipt.
    """

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError) as exc:  # pragma: no cover - old torch runtime
        raise GateTechnicalInvalid(f"forced SDPA MATH backend is unavailable: {exc}") from exc
    with sdpa_kernel(SDPBackend.MATH):
        return _strict_model_call(model, payload)


def _sdpa_backend_receipt() -> dict[str, Any]:
    """Return the exact backend-selection receipt attached to each scalar call."""

    return {
        "status": "forced",
        "backend": "math",
        "selection": "torch.nn.attention.sdpa_kernel(SDPBackend.MATH)",
        "operator_semantics_unchanged": True,
    }


@dataclass
class LiveRuntimeBinding:
    """Runtime seam shared by the real HF adapter and CPU fake runtimes."""

    adapter: Any
    runtime: Any
    event: Mapping[str, Any]
    seeded_context: Any
    identity: Mapping[str, Any] = field(default_factory=dict)
    close_callback: Callable[[], None] | None = None

    @property
    def model(self) -> Any:
        value = getattr(self.adapter, "model", None)
        if value is None:
            value = getattr(self.adapter, "backend_model", None)
        if value is None:
            raise GateTechnicalInvalid("runtime adapter does not expose a model")
        return value

    @property
    def device(self) -> torch.device:
        return _model_device(self.adapter)

    def close(self) -> None:
        callback = self.close_callback
        if callback is not None:
            callback()
            return
        close = getattr(self.adapter, "close", None)
        if callable(close):
            close()


def _as_stop_token_ids(value: Any, *, label: str) -> tuple[int, ...]:
    """Normalize tokenizer EOS/im_end declarations without collapsing a set."""

    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    result: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise GateTechnicalInvalid(f"{label} contains an invalid token ID")
        if int(item) not in result:
            result.append(int(item))
    return tuple(result)


def _native_stop_token_ids(adapter: Any, contract: Any) -> tuple[int, ...]:
    """Return the explicit native STOP set (EOS plus ``<|im_end|>``)."""

    values: list[int] = list(_as_stop_token_ids(getattr(contract, "eos_token_id", None), label="wrapper EOS"))
    tokenizer = getattr(adapter, "tokenizer", None)
    if tokenizer is not None:
        values.extend(_as_stop_token_ids(getattr(tokenizer, "eos_token_id", None), label="tokenizer EOS"))
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            try:
                candidate = convert("<|im_end|>")
            except Exception as exc:  # pragma: no cover - tokenizer-specific
                raise GateTechnicalInvalid(f"tokenizer cannot resolve <|im_end|>: {exc}") from exc
            if candidate is not None and candidate != getattr(tokenizer, "unk_token_id", None):
                values.extend(_as_stop_token_ids(candidate, label="tokenizer <|im_end|>"))
    return tuple(dict.fromkeys(values))


def _native_contract_from_legacy(
    contract: Any,
    *,
    adapter: Any | None = None,
) -> natural.NativeRowContract:
    try:
        stop_ids = _native_stop_token_ids(adapter, contract) if adapter is not None else _as_stop_token_ids(
            getattr(contract, "eos_token_id", None), label="wrapper EOS"
        )
        return natural.NativeRowContract(
            opener_token_id=int(contract.object_ref_start_token_id),
            object_ref_end_token_id=int(contract.object_ref_end_token_id),
            box_start_token_id=int(contract.box_start_token_id),
            box_end_token_id=int(contract.box_end_token_id),
            coordinate_token_start_id=int(contract.coordinate_token_start_id),
            coordinate_bin_count=int(getattr(contract, "coordinate_bin_count", 1000)),
            coordinate_count=int(getattr(contract, "coordinate_count", 4)),
            commit_token_id=(None if contract.commit_token_id is None else int(contract.commit_token_id)),
            stop_token_id=(None if not stop_ids else int(stop_ids[0])),
            assistant_format=str(contract.assistant_format),
            stop_token_ids=stop_ids,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise GateTechnicalInvalid(f"cannot bind native row contract: {exc}") from exc


def _runtime_prompt_ids(runtime: Any) -> tuple[int, ...]:
    for name in ("prompt_ids", "prompt_token_ids"):
        value = getattr(runtime, name, None)
        if value is not None:
            return _token_ids(value, label=f"runtime.{name}")
    value = getattr(runtime, "prompt_token_ids", None)
    if value is not None:
        return _token_ids(value, label="runtime.prompt_token_ids")
    raise GateTechnicalInvalid("runtime does not expose exact native prompt token IDs")


def _event_regions(event: Mapping[str, Any]) -> Mapping[str, Sequence[int]]:
    """Resolve the frozen S geometry without inventing missing regions."""

    candidates: Any = None
    for key in ("image_cell_regions", "regions", "overlap_regions", "target_regions"):
        if isinstance(event.get(key), Mapping):
            candidates = event[key]
            break
    if candidates is None:
        raise GateTechnicalInvalid("S event lacks declared image_cell_regions")
    regions = dict(candidates)
    if "a_exclusive" not in regions and "covered_a_exclusive" in regions:
        regions["a_exclusive"] = regions["covered_a_exclusive"]
    if "b_exclusive" not in regions and "target_b_exclusive" in regions:
        regions["b_exclusive"] = regions["target_b_exclusive"]
    # A verified same-class competitor is an optional K13 scope.  The
    # materialized S event records an empty region (and a null owner identity)
    # when no qualifying competitor exists; that disposition is promoted to an
    # attested K13 ``not_applicable`` by the live factory.  All other geometry
    # regions remain hard requirements for the gate.
    required = ("a_exclusive", "b_exclusive", "background", "shared_core")
    for key in required:
        values = regions.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise GateTechnicalInvalid(f"S event image-cell region {key!r} is missing or malformed")
        normalized: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GateTechnicalInvalid(f"S event image-cell region {key!r} contains an invalid cell")
            normalized.append(int(value))
        if len(set(normalized)) != len(normalized):
            raise GateTechnicalInvalid(f"S event image-cell region {key!r} contains duplicates")
        regions[key] = tuple(normalized)
    competitor = regions.get("same_class_competitor")
    if competitor is None:
        competitor = ()
    if not isinstance(competitor, Sequence) or isinstance(competitor, (str, bytes)):
        raise GateTechnicalInvalid("S event image-cell region 'same_class_competitor' is malformed")
    normalized_competitor: list[int] = []
    for value in competitor:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateTechnicalInvalid(
                "S event image-cell region 'same_class_competitor' contains an invalid cell"
            )
        normalized_competitor.append(int(value))
    if len(set(normalized_competitor)) != len(normalized_competitor):
        raise GateTechnicalInvalid("S event image-cell region 'same_class_competitor' contains duplicates")
    regions["same_class_competitor"] = tuple(normalized_competitor)
    return regions


def _validate_frozen_event_geometry(event: Mapping[str, Any]) -> None:
    try:
        image_id = int(event.get("image_id", -1))
        source_index = int(event.get("source_panel_object_index", -1))
    except (TypeError, ValueError) as exc:
        raise GateTechnicalInvalid("resolved event image/source identity is malformed") from exc
    if str(event.get("gt_owner_id")) != EVENT_ID or image_id != 5001:
        raise GateTechnicalInvalid("resolved event is not frozen S gt:5001:15 on image 5001")
    if source_index != 15:
        raise GateTechnicalInvalid("resolved event source_panel_object_index is not 15")
    _validate_declared_event_geometry(event)


def _validate_declared_event_geometry(event: Mapping[str, Any]) -> None:
    if not isinstance(event.get("image_id"), int) or int(event["image_id"]) < 0:
        raise GateTechnicalInvalid("resolved event image identity is malformed")
    if not isinstance(event.get("source_panel_object_index"), int) or int(event["source_panel_object_index"]) < 0:
        raise GateTechnicalInvalid("resolved event source identity is malformed")
    _event_regions(event)
    geometry = event.get("geometry_by_checkpoint")
    if not isinstance(geometry, Mapping) or not isinstance(geometry.get(CHECKPOINT), Mapping):
        raise GateTechnicalInvalid("S event lacks geometry_by_checkpoint.S")
    s_geometry = geometry[CHECKPOINT]
    if s_geometry.get("launch_eligible") is not True:
        raise GateTechnicalInvalid("S event geometry is not launch-eligible")
    plan = s_geometry.get("image_plan_identity")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("cell_count"), int) or int(plan["cell_count"]) <= 0:
        raise GateTechnicalInvalid("S event lacks a valid image_plan_identity")
    competitor = _event_regions(event)["same_class_competitor"]
    competitor_owner = s_geometry.get("same_class_competitor_owner_id")
    if competitor and not isinstance(competitor_owner, str):
        raise GateTechnicalInvalid("S event lacks same_class_competitor_owner_id identity")


def _seeded_context_prefix(context: Any) -> tuple[int, ...]:
    value = getattr(context, "prefix_ids", None)
    if value is None:
        value = context.get("prefix_ids") if isinstance(context, Mapping) else None
    if value is None:
        value = getattr(context, "prefix_token_ids", None)
    if value is None:
        raise GateTechnicalInvalid("seeded event context lacks prefix_ids")
    return _token_ids(value, label="seeded event prefix")


def build_natural_event_context(
    binding: LiveRuntimeBinding,
    *,
    event_id: str | None = None,
    max_row_tokens: int = natural.MAX_ROW_TOKENS,
    max_rows: int = natural.MAX_ROWS,
    max_new_tokens: int | None = None,
) -> tuple[natural.NaturalEventContext, UnseededBoundary, dict[str, Any]]:
    """Bind the old seeded context to the exact new pre-opener context."""

    seeded_context = binding.seeded_context
    contract_source = getattr(binding.adapter, "wrapper_contract", None)
    if contract_source is None:
        contract_source = getattr(binding.runtime, "wrapper_contract", None)
    if contract_source is None:
        contract_source = getattr(seeded_context, "row_contract", None)
    if contract_source is None:
        raise GateTechnicalInvalid("runtime lacks a native wrapper contract")
    contract = (
        contract_source
        if isinstance(contract_source, natural.NativeRowContract)
        else _native_contract_from_legacy(contract_source, adapter=binding.adapter)
    )
    seeded = _seeded_context_prefix(seeded_context)
    boundary = remove_exact_preseeded_opener(seeded, opener_token_id=contract.opener_token_id)
    prompt = _runtime_prompt_ids(binding.runtime)
    if boundary.seeded_prefix_token_ids[: len(prompt)] != prompt:
        raise GateTechnicalInvalid("seeded H0 prefix does not begin with the exact native prompt")
    history = boundary.seeded_prefix_token_ids[len(prompt) : -1]
    if not history:
        raise GateTechnicalInvalid("S gate requires at least one exact completed history token")
    latest = getattr(seeded_context, "latest_row_ids", None)
    if latest is None:
        latest = getattr(seeded_context, "latest_history_row_token_ids", ())
    latest_ids = _token_ids(latest or (), label="latest covered row", allow_empty=True)
    if not latest_ids:
        runtime_h0 = getattr(binding.runtime, "h0", {})
        rows = runtime_h0.get("rows", []) if isinstance(runtime_h0, Mapping) else []
        if rows:
            latest_ids = _token_ids(rows[-1], label="latest covered row")
    if not latest_ids:
        raise GateTechnicalInvalid("S gate requires the exact latest covered row for residual seams")
    if history[-len(latest_ids) :] != latest_ids:
        raise GateTechnicalInvalid(
            "latest covered row token IDs are not the exact suffix of exact history"
        )
    history_prefix_width = len(prompt) + len(history) - len(latest_ids)
    if history_prefix_width < 0:
        raise GateTechnicalInvalid("latest covered row is longer than exact prompt+history")
    resolved_event_id = str(event_id or binding.event.get("event_id") or binding.event.get("gt_owner_id") or EVENT_ID)
    natural_context = natural.build_event_context(
        event_id=resolved_event_id,
        prompt_token_ids=prompt,
        exact_history_token_ids=history,
        row_contract=contract,
        model_inputs={},
        max_row_tokens=int(max_row_tokens),
        max_rows=int(max_rows),
        max_new_tokens=max_new_tokens,
        latest_history_row_token_ids=latest_ids,
        history_prefix_width=history_prefix_width,
    )
    if natural_context.prefix_token_ids != boundary.natural_prefix_token_ids:
        raise GateTechnicalInvalid("rebuilt natural prefix differs from exact one-token subtraction")
    identity = {
        "event_id": resolved_event_id,
        "checkpoint": CHECKPOINT,
        "admission_mode": "pre_opener_natural",
        "wrapper": contract.assistant_format,
        "prefix": boundary.receipt(),
        "prompt_token_count": len(prompt),
        "history_token_count": len(history),
        "latest_history_row_sha256": sha256_json(list(latest_ids)),
        "history_prefix_width": history_prefix_width,
        "native_stop_token_ids": list(contract.effective_stop_token_ids),
        "natural_context": natural_context.receipt(),
    }
    grid = getattr(binding.runtime, "image_grid_thw", None)
    if isinstance(grid, torch.Tensor):
        identity["image_grid"] = _grid_receipt(grid)
    return natural_context, boundary, identity


def build_live_attention_mask_actuators(
    binding: LiveRuntimeBinding,
    context: natural.NaturalEventContext,
) -> dict[str, Any]:
    """Build the declared K/H scalar-step factories from frozen S geometry.

    The factories only construct per-step model-facing masks/receipts.  Layer
    consumption remains attested around the actual scalar forward by
    ``ScalarRuntimeAdapter``; this function does not implement attention.
    """

    _validate_declared_event_geometry(binding.event)
    runtime = binding.runtime
    span = getattr(runtime, "image_span", None)
    if span is None or not hasattr(span, "absolute_positions"):
        raise GateTechnicalInvalid("live S runtime lacks the derived image_span absolute positions")
    try:
        from scripts.research import natural_boundary_attention_actuators as attention

        image_positions = tuple(int(value) for value in span.absolute_positions)
        regions = _event_regions(binding.event)
        resolve_positions = getattr(legacy.static, "resolve_image_positions", None)
        if not callable(resolve_positions):
            raise GateTechnicalInvalid("established static helper lacks resolve_image_positions")

        def absolute(name: str) -> tuple[int, ...]:
            try:
                return tuple(int(value) for value in resolve_positions(span, regions[name]))
            except Exception as exc:
                raise GateTechnicalInvalid(f"cannot resolve S {name} cells to absolute image keys: {exc}") from exc

        b_positions = absolute("b_exclusive")
        a_positions = absolute("a_exclusive")
        background_positions = absolute("background")
        competitor_positions = absolute("same_class_competitor")
        row_start = int(context.history_prefix_width or 0)
        latest = tuple(context.latest_history_row_token_ids)
        latest_positions = tuple(row_start + index for index in range(len(latest)))
        terminal_positions = tuple(
            row_start + index
            for index, token in enumerate(latest)
            if int(token) == int(context.row_contract.closure_token_id)
        )
        if not latest_positions or not terminal_positions:
            raise GateTechnicalInvalid("latest history row lacks exact terminal-carrier position identity")

        model = binding.model
        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", config)
        layer_count = getattr(text_config, "num_hidden_layers", None)
        head_count = getattr(text_config, "num_attention_heads", None)
        if layer_count is None or head_count is None:
            language_model = getattr(model, "language_model", None)
            lm_config = getattr(language_model, "config", None)
            layer_count = layer_count or getattr(lm_config, "num_hidden_layers", None)
            head_count = head_count or getattr(lm_config, "num_attention_heads", None)
        if layer_count is None or head_count is None:
            raise GateTechnicalInvalid("live model config lacks decoder layer/head counts for K14 scope")
        factory_common = {
            "image_key_positions": image_positions,
            "b_exclusive_positions": b_positions,
            "a_exclusive_positions": a_positions,
            "background_positions": background_positions,
            "same_class_competitor_positions": competitor_positions,
            "latest_terminal_key_positions": terminal_positions,
            "latest_row_key_positions": latest_positions,
            "layer_count": int(layer_count),
            "head_count": int(head_count),
            "device": binding.device,
            "dtype": torch.float32,
        }
        factories: dict[str, Any] = {}
        for arm in sorted(ATTENTION_ARMS):
            factories[arm] = attention.build_scalar_step_factory(arm, **factory_common)
        return factories
    except GateTechnicalInvalid:
        raise
    except Exception as exc:
        raise GateTechnicalInvalid(f"cannot build live S K/H actuator factories: {exc}") from exc


def _resolve_k14_reference_positions(binding: LiveRuntimeBinding) -> dict[str, tuple[int, ...]]:
    """Resolve fixed K14 target/background key identities for mass receipts."""

    span = getattr(binding.runtime, "image_span", None)
    if span is None:
        raise GateTechnicalInvalid("live S runtime lacks image_span for K14 mass scope")
    regions = _event_regions(binding.event)
    resolve_positions = getattr(legacy.static, "resolve_image_positions", None)
    if not callable(resolve_positions):
        raise GateTechnicalInvalid("established static helper lacks resolve_image_positions")
    b = tuple(int(value) for value in resolve_positions(span, regions["b_exclusive"]))
    background = tuple(int(value) for value in resolve_positions(span, regions["background"]))
    if not b:
        raise GateTechnicalInvalid("K14 requires non-empty B-exclusive image keys")
    try:
        from scripts.research import natural_boundary_attention_actuators as attention

        selected_background = attention.select_zero_overlap_background_positions(b, background)
    except Exception as exc:
        raise GateTechnicalInvalid(f"K14 background scope is not deterministically selectable: {exc}") from exc
    return {"K14T": b, "K14B": tuple(int(value) for value in selected_background)}


def build_live_residual_actuator() -> Any:
    """Return the established residual hook callback for live N arms."""

    try:
        from scripts.research import natural_boundary_residual_actuators as residual

        factory = getattr(residual, "make_residual_actuator_callback", None)
        if not callable(factory):
            raise GateTechnicalInvalid("residual actuator module lacks make_residual_actuator_callback")
        return factory(tolerance=1e-4)
    except GateTechnicalInvalid:
        raise
    except Exception as exc:
        raise GateTechnicalInvalid(f"cannot build live residual actuator callback: {exc}") from exc


# Explicit aliases for launch wrappers that use the longer unit terminology.
build_s_primary_live_attention_actuators = build_live_attention_mask_actuators
build_s_primary_live_residual_actuator = build_live_residual_actuator


class ScalarRuntimeAdapter:
    """Rebuild exact multimodal inputs/MRoPE for each scalar natural step."""

    def __init__(
        self,
        binding: LiveRuntimeBinding,
        context: natural.NaturalEventContext,
        boundary: UnseededBoundary,
    ) -> None:
        self.binding = binding
        self.context = context
        self.boundary = boundary
        self.calls: list[dict[str, Any]] = []
        self._base_prefix = boundary.natural_prefix_token_ids
        self._parity_mode = "disabled"
        self._reference_name: str | None = None
        self._native_logits_by_name: dict[str, dict[str, torch.Tensor]] = {}
        # Backwards-compatible view used by focused CPU tests and older callers.
        self._native_logits: dict[str, torch.Tensor] = {}
        self._parity_deltas: list[float] = []
        self._parity_compared = 0
        self._active_attention_callback: Any | None = None
        self.active_arm: str | None = None
        self.k14_reference_positions: dict[str, tuple[int, ...]] = {}
        self._native_mass_receipts: dict[str, dict[str, Any]] = {}

    @property
    def model(self) -> Any:
        return self.binding.model

    @property
    def device(self) -> torch.device:
        return self.binding.device

    def reset(self) -> None:
        self.calls.clear()

    def begin_parity(
        self,
        mode: Literal["native", "candidate", "disabled"],
        *,
        reference_name: str | None = None,
    ) -> None:
        if mode not in {"native", "candidate", "disabled"}:
            raise GateTechnicalInvalid(f"unknown full-logit parity mode {mode!r}")
        self._parity_mode = mode
        self._reference_name = None if reference_name is None else str(reference_name)
        self._parity_deltas.clear()
        self._parity_compared = 0
        if mode == "native":
            name = self._reference_name or "native"
            self._native_logits_by_name[name] = {}
            self._native_logits = self._native_logits_by_name[name]
        elif mode == "candidate":
            name = self._reference_name or "native"
            reference = self._native_logits_by_name.get(name)
            if not reference:
                raise GateTechnicalInvalid(
                    f"full-logit parity requires a prior native reference arm {name!r}"
                )
            self._native_logits = reference
        else:
            self._native_logits = {}

    def parity_receipt(self) -> dict[str, Any]:
        if self._parity_mode == "native":
            return {
                "status": "reference_captured",
                "reference_arm": self._reference_name,
                "reference_step_count": len(self._native_logits),
                "candidate_step_count": 0,
                "per_forward_max_abs_delta": None,
                "tolerance": 1e-4,
                "passed": True,
            }
        if self._parity_mode == "candidate":
            max_delta = max(self._parity_deltas, default=0.0)
            return {
                "status": "measured",
                "reference_arm": self._reference_name,
                "reference_step_count": len(self._native_logits),
                "candidate_step_count": self._parity_compared,
                "per_forward_max_abs_delta": float(max_delta),
                "tolerance": 1e-4,
                "passed": bool(
                    self._parity_compared == len(self._native_logits)
                    and max_delta <= 1e-4
                ),
            }
        return {
            "status": "not_measured",
            "reference_arm": self._reference_name,
            "reference_step_count": 0,
            "candidate_step_count": 0,
            "per_forward_max_abs_delta": None,
            "tolerance": 1e-4,
            "passed": False,
        }

    def _exact_inputs(
        self,
        ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> tuple[dict[str, Any], torch.Tensor, str]:
        adapter = self.binding.adapter
        runtime = self.binding.runtime
        builder = getattr(adapter, "exact_model_inputs", None)
        if not callable(builder):
            builder = getattr(adapter, "build_exact_model_inputs", None)
        if not callable(builder):
            raise GateTechnicalInvalid(
                "established runtime adapter lacks exact_model_inputs/build_exact_model_inputs"
            )
        try:
            result = builder(runtime, ids, attention_mask=attention_mask)
        except TypeError as exc:
            raise GateTechnicalInvalid(f"exact multimodal input builder rejected scalar inputs: {exc}") from exc
        if not isinstance(result, tuple) or len(result) != 3:
            raise GateTechnicalInvalid("exact multimodal input builder must return (payload, position_ids, mrope_hash)")
        payload, position_ids, mrope_hash = result
        if not isinstance(payload, Mapping):
            raise GateTechnicalInvalid("exact multimodal input payload must be a mapping")
        payload = dict(payload)
        payload_ids = payload.get("input_ids")
        if not isinstance(payload_ids, torch.Tensor) or not torch.equal(payload_ids, ids):
            raise GateTechnicalInvalid("exact multimodal payload input_ids differ from scalar prefix")
        if payload.get("use_cache") is not False:
            raise GateTechnicalInvalid("exact multimodal payload must set use_cache=False")
        if position_ids is None:
            position_ids = payload.get("position_ids")
        if not isinstance(position_ids, torch.Tensor):
            raise GateTechnicalInvalid("exact multimodal payload lacks rebuilt MRoPE position_ids")
        payload["position_ids"] = position_ids
        if not isinstance(mrope_hash, str) or not mrope_hash:
            raise GateTechnicalInvalid("exact multimodal payload lacks an MRoPE identity hash")
        device = self.device
        moved: dict[str, Any] = {}
        for key, value in payload.items():
            moved[key] = _tensor_to_device(value, device)
        payload = moved
        if payload["input_ids"].device != device or payload["position_ids"].device != device:
            raise GateTechnicalInvalid("scalar input_ids/position_ids are not on the actual model device")
        _position_receipt(payload["position_ids"], sequence_length=int(ids.shape[-1]))
        grid = payload.get("image_grid_thw")
        if grid is None:
            grid = getattr(runtime, "image_grid_thw", None)
            if isinstance(grid, torch.Tensor):
                payload["image_grid_thw"] = grid.to(device=device)
        if not isinstance(grid, torch.Tensor):
            raise GateTechnicalInvalid("exact multimodal payload lacks image_grid_thw")
        grid = grid.to(device=device)
        payload["image_grid_thw"] = grid
        _grid_receipt(grid)
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) and value.device != device:
                raise GateTechnicalInvalid(f"scalar tensor {key} is not on model device {device}")
        return payload, payload["position_ids"], mrope_hash

    def forward(self, input_ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> Any:
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise GateTechnicalInvalid("scalar input_ids must have shape [1,S]")
        ids = input_ids.detach().cpu().to(dtype=torch.long)
        values = tuple(int(value) for value in ids[0].tolist())
        if values[: len(self._base_prefix)] != self._base_prefix:
            raise GateTechnicalInvalid("growing scalar prefix no longer preserves the exact natural prefix")
        if len(values) < len(self._base_prefix):
            raise GateTechnicalInvalid("scalar prefix is shorter than the exact natural prefix")
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise GateTechnicalInvalid("attention_mask actuator must return a tensor")
            attention_mask = attention_mask.to(device=self.device)
        payload, position_ids, mrope_hash = self._exact_inputs(
            ids.to(device=self.device),
            attention_mask=attention_mask,
        )
        callback = self._active_attention_callback
        observed_attention_receipt: Mapping[str, Any] | None = None
        block23_mass_receipt: Mapping[str, Any] | None = None

        def call_with_layer_attestation(payload_to_call: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any] | None]:
            """Call once, attaching a real all-layer mask attestor when needed."""

            if callback is None or getattr(callback, "outer_arm", None) not in ATTENTION_ARMS:
                return _forced_math_model_call(self.model, payload_to_call), None
            candidate = getattr(callback, "last_attestation", None)
            # An explicit K13 ``not_applicable`` disposition has no operator
            # to consume; preserve that attested status without pretending a
            # layer hook proved a no-op.  Every applicable attention forward
            # is wrapped below, even when a callback supplies a receipt that
            # claims ``passed=True``.  Callback metadata is evidence to record,
            # never a substitute for observing the real model call.
            if isinstance(candidate, Mapping) and candidate.get("status") == "not_applicable":
                return _forced_math_model_call(self.model, payload_to_call), candidate
            try:
                from scripts.research import natural_boundary_attention_actuators as actuators

                attestor = actuators.LayerMaskConsumptionAttestor(
                    self.model, payload_to_call["attention_mask"]
                )
                with attestor:
                    observed_output = _forced_math_model_call(self.model, payload_to_call)
                observed_receipt = attestor.receipt()
            except Exception as exc:
                raise GateTechnicalInvalid(
                    f"{callback.outer_arm} all-layer attention consumption attestation failed: {exc}"
                ) from exc
            if observed_receipt.get("passed") is not True:
                raise GateTechnicalInvalid(
                    f"{callback.outer_arm} all-layer attention consumption attestation did not pass"
                )
            callback.last_attestation = dict(observed_receipt)
            previous = getattr(callback, "last_receipt", None)
            callback.last_receipt = {
                **(dict(previous) if isinstance(previous, Mapping) else {}),
                "layer_consumption_attestation": dict(observed_receipt),
                "all_layer_consumption_attestation": dict(observed_receipt),
            }
            return observed_output, observed_receipt

        active_arm = self.active_arm
        if callback is not None and getattr(callback, "outer_arm", None) in ATTENTION_ARMS:
            if "attention_mask" not in payload or not isinstance(payload.get("attention_mask"), torch.Tensor):
                raise GateTechnicalInvalid(
                    f"{callback.outer_arm} callback did not provide a model-facing attention_mask"
                )

        not_applicable_attention = bool(
            isinstance(getattr(callback, "last_mass_requirement", None), Mapping)
            and getattr(callback, "last_mass_requirement", {}).get("status") == "not_applicable"
        )
        if active_arm in K14_ARMS and not_applicable_attention:
            output, observed_attention_receipt = call_with_layer_attestation(payload)
            block23_mass_receipt = {
                "schema_version": "natural_boundary_block23_mass_requirement.v1",
                "status": "not_applicable",
                "passed": True,
                "before_after_per_head": False,
            }
        elif active_arm in K14_ARMS:
            if active_arm not in self.k14_reference_positions:
                raise GateTechnicalInvalid("K14 live mass scope is not bound to frozen event geometry")
            selected_positions = self.k14_reference_positions[active_arm]
            try:
                from scripts.research import natural_boundary_attention_actuators as actuators

                mask = payload.get("attention_mask")
                if not isinstance(mask, torch.Tensor) or not mask.dtype.is_floating_point:
                    raise GateTechnicalInvalid("K14 requires a floating additive attention mask")
                native_mask = mask.detach().clone()
                native_mask[native_mask > 0] -= 2.0
                native_payload, _native_position_ids, _native_mrope = self._exact_inputs(
                    ids.to(device=self.device), attention_mask=native_mask
                )
                cached_baseline = self._native_mass_receipts.get(sha256_json(list(values)), {}).get(active_arm)
                if isinstance(cached_baseline, Mapping):
                    native_receipt = dict(cached_baseline)
                else:
                    native_attestor = actuators.Block23SDPAMassAttestor(
                        selected_positions,
                        query_positions=(len(values) - 1,),
                        key_positions=tuple(range(len(values))),
                        absolute_query_positions=tuple(range(len(values))),
                        phase="native",
                    )
                    with native_attestor:
                        _forced_math_model_call(self.model, native_payload)
                    native_receipt = native_attestor.receipt()
                if native_receipt.get("passed") is not True:
                    raise GateTechnicalInvalid("K14 native block23 SDPA mass attestation failed")
                biased_attestor = actuators.Block23SDPAMassAttestor(
                    selected_positions,
                    query_positions=(len(values) - 1,),
                    key_positions=tuple(range(len(values))),
                    absolute_query_positions=tuple(range(len(values))),
                    phase="biased",
                )
                with biased_attestor:
                    output, observed_attention_receipt = call_with_layer_attestation(payload)
                biased_receipt = biased_attestor.receipt()
                if biased_receipt.get("passed") is not True:
                    raise GateTechnicalInvalid("K14 biased block23 SDPA mass attestation failed")
                comparison = actuators.compare_block23_sdpa_mass_receipts(
                    native_receipt, biased_receipt, tolerance=0.0
                )
                def _all_nonzero(value: Any) -> bool:
                    if isinstance(value, (list, tuple)):
                        return bool(value) and all(_all_nonzero(item) for item in value)
                    try:
                        return abs(float(value)) > 0.0
                    except (TypeError, ValueError):
                        return False

                if comparison.get("passed") is not True or not _all_nonzero(comparison.get("delta_mass")):
                    raise GateTechnicalInvalid(
                        "K14 block23 mass shift was not observed for every declared head"
                    )
                block23_mass_receipt = {
                    "schema_version": "natural_boundary_block23_mass_requirement.v1",
                    "status": "passed",
                    "passed": True,
                    "selected_key_positions": list(selected_positions),
                    "native": native_receipt,
                    "biased": biased_receipt,
                    "comparison": comparison,
                    "before_after_per_head": True,
                }
            except GateTechnicalInvalid:
                raise
            except Exception as exc:
                raise GateTechnicalInvalid(f"K14 block23 mass attestation failed: {exc}") from exc
        elif active_arm == "K01" and self.k14_reference_positions:
            try:
                from scripts.research import natural_boundary_attention_actuators as actuators

                baseline_receipts: dict[str, Any] = {}
                for reference_arm, selected_positions in sorted(self.k14_reference_positions.items()):
                    attestor = actuators.Block23SDPAMassAttestor(
                        selected_positions,
                        query_positions=(len(values) - 1,),
                        key_positions=tuple(range(len(values))),
                        absolute_query_positions=tuple(range(len(values))),
                        phase="native",
                    )
                    with attestor:
                        _forced_math_model_call(self.model, payload)
                    receipt = attestor.receipt()
                    if receipt.get("passed") is not True:
                        raise GateTechnicalInvalid(
                            f"K01 native block23 mass baseline failed for {reference_arm}"
                        )
                    baseline_receipts[reference_arm] = receipt
                self._native_mass_receipts[sha256_json(list(values))] = baseline_receipts
            except GateTechnicalInvalid:
                raise
            except Exception as exc:
                raise GateTechnicalInvalid(f"K01 block23 mass baseline failed: {exc}") from exc
            output, observed_attention_receipt = call_with_layer_attestation(payload)
        elif callback is not None and getattr(callback, "outer_arm", None) in ATTENTION_ARMS:
            output, observed_attention_receipt = call_with_layer_attestation(payload)
        else:
            output = _forced_math_model_call(self.model, payload)

        if active_arm in K14_ARMS and isinstance(block23_mass_receipt, Mapping):
            previous = getattr(callback, "last_receipt", None)
            callback.last_receipt = {
                **(dict(previous) if isinstance(previous, Mapping) else {}),
                "block23_mass_requirement": dict(block23_mass_receipt),
            }
        logits = (
            output
            if isinstance(output, torch.Tensor)
            else output.logits
            if hasattr(output, "logits")
            else output.get("logits")
            if isinstance(output, Mapping)
            else None
        )
        if not isinstance(logits, torch.Tensor):
            raise GateTechnicalInvalid("scalar model output must expose logits")
        if logits.ndim == 3:
            if logits.shape[0] != 1 or logits.shape[1] <= 0:
                raise GateTechnicalInvalid("scalar model output must expose [1,S,V] logits")
            last_logits = logits[0, -1]
        elif logits.ndim == 2:
            last_logits = logits[0] if logits.shape[0] == 1 else logits[-1]
        elif logits.ndim == 1:
            last_logits = logits
        else:
            raise GateTechnicalInvalid("scalar model output logits have unsupported rank")
        if last_logits.numel() <= 0 or not bool(torch.isfinite(last_logits).all().item()):
            raise GateTechnicalInvalid("scalar model output logits are empty or non-finite")
        last_logits = last_logits.detach().float().cpu().contiguous()
        prefix_key = sha256_json(list(values))
        if self._parity_mode == "native":
            self._native_logits[prefix_key] = last_logits
        elif self._parity_mode == "candidate":
            reference = self._native_logits.get(prefix_key)
            if reference is None:
                raise GateTechnicalInvalid(
                    "full-logit parity reference lacks the exact candidate prefix"
                )
            if reference.shape != last_logits.shape:
                raise GateTechnicalInvalid("full-logit parity vocabulary shape differs")
            delta = float((reference - last_logits).abs().max().item())
            self._parity_deltas.append(delta)
            self._parity_compared += 1
            if delta > 1e-4:
                raise GateTechnicalInvalid(
                    f"full-logit no-op parity exceeded 1e-4: observed {delta:.8g}"
                )
        call_receipt = {
            "step": len(self.calls),
            "sequence_length": len(values),
            "input_ids_sha256": sha256_json(list(values)),
            "input_ids_device": str(payload["input_ids"].device),
            "model_device": str(self.device),
            "use_cache": False,
            "sdpa_backend": _sdpa_backend_receipt(),
            "position_ids": _position_receipt(position_ids, sequence_length=len(values)),
            "mrope_hash": mrope_hash,
            "image_grid": _grid_receipt(payload["image_grid_thw"]),
            "tensor_devices": {
                key: str(value.device)
                for key, value in sorted(payload.items())
                if isinstance(value, torch.Tensor)
            },
            "kwargs": sorted(payload),
        }
        if callback is not None and getattr(callback, "outer_arm", None) in ATTENTION_ARMS:
            callback_receipt = getattr(callback, "last_receipt", None)
            if isinstance(callback_receipt, Mapping):
                call_receipt["attention_actuation_receipt"] = dict(callback_receipt)
            if isinstance(observed_attention_receipt, Mapping):
                call_receipt["layer_consumption_attestation"] = dict(observed_attention_receipt)
        if isinstance(block23_mass_receipt, Mapping):
            call_receipt["block23_mass_requirement"] = dict(block23_mass_receipt)
        if active_arm == "K01":
            baseline = self._native_mass_receipts.get(prefix_key)
            if isinstance(baseline, Mapping):
                call_receipt["block23_mass_baseline"] = dict(baseline)
        self.calls.append(call_receipt)
        return output


class NaturalModelProxy:
    """Expose the real model's module tree while rebuilding calls via binding."""

    def __init__(self, scalar: ScalarRuntimeAdapter) -> None:
        self.scalar = scalar
        self.backend_model = scalar.model

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend_model, name)

    def __call__(self, **kwargs: Any) -> Any:
        unknown = set(kwargs) - {"input_ids", "attention_mask", "use_cache"}
        if unknown:
            raise GateTechnicalInvalid(f"natural model proxy rejected ignored kwargs: {sorted(unknown)}")
        if kwargs.get("use_cache") is not False:
            raise GateTechnicalInvalid("natural model proxy requires use_cache=False")
        input_ids = kwargs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise GateTechnicalInvalid("natural model proxy requires input_ids")
        return self.scalar.forward(input_ids, attention_mask=kwargs.get("attention_mask"))


def _arm_attention_callback(callback: Any, arm_id: str) -> Any:
    if callback is None:
        return None

    def wrapped(context: Any, *, input_ids: torch.Tensor, step: int, row_index: int, arm_id: str = arm_id) -> Any:
        del arm_id  # The outer arm is bound below; natural's arm is intentionally ignored.
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
        try:
            if positional:
                result = callback(
                    context,
                    input_ids=input_ids,
                    step=step,
                    row_index=row_index,
                    arm_id=wrapped.outer_arm,
                )
            else:
                result = callback(
                    sequence_length=int(input_ids.shape[-1]),
                    query_position=int(input_ids.shape[-1]) - 1,
                    device=input_ids.device,
                    step=step,
                    row_index=row_index,
                    arm_id=wrapped.outer_arm,
                    context=context,
                )
            if wrapped.outer_arm in ATTENTION_ARMS:
                if not isinstance(result, Mapping):
                    raise GateTechnicalInvalid(
                        f"{wrapped.outer_arm} requires an attention-mask mapping with explicit attestation"
                    )
                receipt = result.get("receipt", result.get("mask_receipt"))
                receipt_status = receipt.get("status") if isinstance(receipt, Mapping) else result.get("status")
                if receipt_status == "not_applicable":
                    # An explicit deterministic not-applicable disposition
                    # (not an unattested construction) carries no operator to
                    # attest.  Keep it in the receipt and let the endpoint
                    # classify it separately from a technical failure.
                    wrapped.last_receipt = dict(receipt) if isinstance(receipt, Mapping) else {"status": receipt_status}
                    wrapped.last_attestation = {"passed": True, "status": "not_applicable"}
                    wrapped.last_mass_requirement = {"passed": True, "status": "not_applicable"}
                    return result
                attestation = result.get("layer_consumption_attestation")
                if attestation is None:
                    attestation = result.get("all_layer_consumption_attestation")
                if attestation is None and isinstance(receipt, Mapping):
                    attestation = receipt.get("layer_consumption_attestation")
                if attestation is None and isinstance(receipt, Mapping):
                    attestation = receipt.get("all_layer_consumption_attestation")
                if not isinstance(attestation, Mapping):
                    raise GateTechnicalInvalid(
                        f"{wrapped.outer_arm} lacks a layer_consumption_attestation mapping"
                    )
                mass = result.get("block23_mass_requirement")
                if mass is None and isinstance(receipt, Mapping):
                    mass = receipt.get("block23_mass_requirement")
                # K14's block-23 mass evidence is a separate required receipt;
                # construction-time ``unattested`` metadata is not promoted.
                if wrapped.outer_arm in {"K14T", "K14B"} and not isinstance(mass, Mapping):
                    raise GateTechnicalInvalid(
                        f"{wrapped.outer_arm} lacks a block23_mass_requirement mapping"
                    )
                if isinstance(receipt, Mapping):
                    normalized = dict(result)
                    normalized["receipt"] = {
                        **dict(receipt),
                        "layer_consumption_attestation": dict(attestation),
                        "all_layer_consumption_attestation": dict(attestation),
                    }
                    if isinstance(mass, Mapping):
                        normalized["receipt"]["block23_mass_requirement"] = dict(mass)
                    result = normalized
                elif isinstance(result, Mapping):
                    normalized = dict(result)
                    normalized["receipt"] = {
                        "layer_consumption_attestation": dict(attestation),
                        "all_layer_consumption_attestation": dict(attestation),
                    }
                    if isinstance(mass, Mapping):
                        normalized["receipt"]["block23_mass_requirement"] = dict(mass)
                    result = normalized
                wrapped.last_receipt = (
                    dict(result.get("receipt", {}))
                    if isinstance(result.get("receipt"), Mapping)
                    else {}
                )
                wrapped.last_attestation = dict(attestation)
                wrapped.last_mass_requirement = dict(mass) if isinstance(mass, Mapping) else None
            return result
        except TypeError as exc:
            raise GateTechnicalInvalid(f"attention-mask actuator rejected S gate callback: {exc}") from exc

    wrapped.outer_arm = arm_id  # type: ignore[attr-defined]
    wrapped.last_receipt = None  # type: ignore[attr-defined]
    wrapped.last_attestation = None  # type: ignore[attr-defined]
    wrapped.last_mass_requirement = None  # type: ignore[attr-defined]
    return wrapped


@dataclass
class SPrimaryNaturalBoundaryGate:
    """Task-specific S gate runner with injectable mask and residual seams."""

    binding: LiveRuntimeBinding
    context: natural.NaturalEventContext
    boundary: UnseededBoundary
    identity: dict[str, Any]
    max_rows: int = natural.MAX_ROWS
    max_row_tokens: int = natural.MAX_ROW_TOKENS

    def __post_init__(self) -> None:
        self.scalar = ScalarRuntimeAdapter(self.binding, self.context, self.boundary)
        try:
            self.scalar.k14_reference_positions = _resolve_k14_reference_positions(self.binding)
        except GateTechnicalInvalid:
            # CPU fake bindings do not carry image geometry; live K/H factory
            # construction will fail closed when requested, while native/N
            # focused tests remain runnable.
            self.scalar.k14_reference_positions = {}
        self.model = NaturalModelProxy(self.scalar)

    @classmethod
    def from_binding(
        cls,
        binding: LiveRuntimeBinding,
        *,
        max_rows: int = natural.MAX_ROWS,
        max_row_tokens: int = natural.MAX_ROW_TOKENS,
        max_new_tokens: int | None = None,
    ) -> "SPrimaryNaturalBoundaryGate":
        context, boundary, identity = build_natural_event_context(
            binding,
            max_rows=max_rows,
            max_row_tokens=max_row_tokens,
            max_new_tokens=max_new_tokens,
        )
        return cls(binding, context, boundary, identity, max_rows=max_rows, max_row_tokens=max_row_tokens)

    def run_arm(
        self,
        arm_id: str,
        *,
        residual_actuator: Any | None = None,
        residual_request: natural.ResidualRequest | None = None,
        attention_mask_actuator: Any | None = None,
        parity_reference: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        arm = str(arm_id)
        if arm not in DECLARED_ARMS:
            raise GateTechnicalInvalid(f"unknown S gate arm {arm!r}")
        residual_arm: Literal["N00", "N01", "N10", "N20"] = arm if arm in RESIDUAL_ARMS else "N00"  # type: ignore[assignment]
        self.scalar.reset()
        if arm == "N00":
            self.scalar.begin_parity("native", reference_name="N00")
        elif arm == "K00":
            self.scalar.begin_parity("native", reference_name="K00")
        elif arm == "N01":
            self.scalar.begin_parity("candidate", reference_name="N00")
        elif arm == "K01":
            self.scalar.begin_parity("candidate", reference_name="K00")
        elif arm == "H00":
            reference = "N00" if "N00" in self.scalar._native_logits_by_name else "K00"
            self.scalar.begin_parity("candidate", reference_name=reference)
        else:
            self.scalar.begin_parity("disabled")
        if arm in ATTENTION_ARMS and attention_mask_actuator is None:
            raise GateTechnicalInvalid(
                f"{arm} is launch-blocked without a caller-owned attention-mask actuator"
            )
        callback = _arm_attention_callback(attention_mask_actuator, arm)
        self.scalar._active_attention_callback = callback
        self.scalar.active_arm = arm
        try:
            result = natural.release_natural_event(
                self.model,
                self.context,
                residual_arm=residual_arm,
                residual_request=residual_request,
                residual_actuator=residual_actuator,
                attention_mask_actuator=callback,
                parity_reference=parity_reference,
            )
        except (natural.TechnicalInvalid, RuntimeError, TypeError, ValueError) as exc:
            raise GateTechnicalInvalid(f"S gate {arm} scalar release failed: {exc}") from exc
        if result.get("admission_mode") != "pre_opener_natural":
            raise GateTechnicalInvalid(f"S gate {arm} did not report pre_opener_natural")
        if result.get("opener_injected") is not False or result.get("synthetic_opener_injections") != 0:
            raise GateTechnicalInvalid(f"S gate {arm} injected an opener")
        if arm in ATTENTION_ARMS:
            scalar_attention = [
                item.get("layer_consumption_attestation")
                for item in self.scalar.calls
                if isinstance(item.get("layer_consumption_attestation"), Mapping)
            ]
            if not scalar_attention or any(item.get("passed") is not True for item in scalar_attention):
                raise GateTechnicalInvalid(
                    f"{arm} lacks a passed per-forward all-layer consumption attestation"
                )
            if arm == "K01" and self.scalar.k14_reference_positions:
                baselines = [item.get("block23_mass_baseline") for item in self.scalar.calls]
                if not baselines or any(
                    not isinstance(item, Mapping)
                    or set(item) != set(self.scalar.k14_reference_positions)
                    or any(receipt.get("passed") is not True for receipt in item.values() if isinstance(receipt, Mapping))
                    for item in baselines
                ):
                    raise GateTechnicalInvalid("K01 lacks matching native block23 mass baselines for K14 scopes")
            if arm in {"K14T", "K14B"}:
                mass_receipts = []
                for item in self.scalar.calls:
                    receipt = item.get("attention_actuation_receipt")
                    mass = receipt.get("block23_mass_requirement") if isinstance(receipt, Mapping) else None
                    if isinstance(mass, Mapping):
                        mass_receipts.append(mass)
                if not mass_receipts or any(item.get("passed") is not True for item in mass_receipts):
                    raise GateTechnicalInvalid(
                        f"{arm} lacks a passed block23_mass_requirement receipt"
                    )
        result = dict(result)
        result["arm_id"] = arm
        result["runtime_scalar_receipts"] = list(self.scalar.calls)
        result["runtime_scalar_forward_count"] = len(self.scalar.calls)
        result["full_logit_parity"] = self.scalar.parity_receipt()
        result["prefix_identity"] = self.identity["prefix"]
        result["native_stop_token_ids"] = list(self.context.row_contract.effective_stop_token_ids)
        first_row = result.get("rows", [{}])[0] if result.get("rows") else {}
        result["initial_prefix_last_token_id"] = self.context.prefix_token_ids[-1]
        result["opener_token_id"] = int(self.context.row_contract.opener_token_id)
        result["first_generated_token_id"] = first_row.get("first_token_id")
        result["opener_injected"] = False
        self._normalize_endpoint_receipts(result)
        endpoint_row = result.get("rows", [{}])[-1] if result.get("rows") else {}
        for key in (
            "owner_match",
            "owner_match_status",
            "native_parse",
            "native_stop",
            "invalid_token",
            "duplicate",
        ):
            if key in endpoint_row:
                result[key] = endpoint_row[key]
        return result

    def _normalize_endpoint_receipts(self, result: dict[str, Any]) -> None:
        """Attach finalizer-facing admission and owner bookkeeping fields."""

        contract = self.context.row_contract
        covered = set(self.binding.seeded_context.covered_owner_ids or ())
        adapter_parse = getattr(self.binding.adapter, "parse_row", None)
        runtime = self.binding.runtime
        # Endpoint ownership is measured from this arm's raw natural
        # trajectory.  ``covered`` is provenance for repeat classification,
        # not a baseline against which an intervention may claim gains/losses.
        seen_endpoint: set[str] = set()
        parsed_rows: list[dict[str, Any]] = []
        for row in result.get("rows", []):
            prefix = row.get("prefix_before_row_token_ids", [])
            tokens = row.get("token_ids", [])
            row["initial_prefix_last_token_id"] = prefix[-1] if prefix else None
            row["opener_token_id"] = int(contract.opener_token_id)
            row["first_generated_token_id"] = tokens[0] if tokens else None
            row["opener_injected"] = False
            owner_match: Mapping[str, Any] | None = None
            parsed: Mapping[str, Any] | None = None
            if callable(adapter_parse) and row.get("status") == "closure":
                try:
                    parsed = adapter_parse(tokens, runtime, row_index=int(row.get("row_index", 0)))
                except Exception as exc:
                    raise GateTechnicalInvalid(f"native owner parser failed: {exc}") from exc
                parsed_match = parsed.get("owner_match") if isinstance(parsed, Mapping) else None
                if isinstance(parsed_match, Mapping):
                    owner_match = dict(parsed_match)
            owner_id = owner_match.get("owner_id") if isinstance(owner_match, Mapping) else None
            owner_text = None if owner_id is None else str(owner_id)
            strict_match = bool(
                isinstance(owner_match, Mapping)
                and owner_match.get("status") in {"unique", "matched"}
                and owner_match.get("physical_match") is True
                and owner_match.get("source_specific") is True
                and owner_text is not None
            )
            # Keep provenance hazards separate: a covered owner is a
            # ``covered_repeat`` even on its first generated occurrence;
            # ``duplicate`` means the same owner was emitted again within
            # this intervention's generated endpoint trajectory.
            covered_repeat = bool(strict_match and owner_text in covered)
            duplicate = bool(strict_match and owner_text in seen_endpoint)
            if strict_match and owner_text is not None:
                seen_endpoint.add(owner_text)
            if row.get("status") == "closure":
                if owner_match is None:
                    owner_match = {
                        "status": "unmatched",
                        "owner_id": None,
                        "physical_match": False,
                        "source_specific": False,
                    }
                row["owner_match"] = dict(owner_match)
                row["owner_match_status"] = owner_match.get("status")
                if isinstance(parsed, Mapping):
                    parsed_receipt = dict(parsed)
                    parsed_receipt.setdefault("valid", row.get("status") == "closure")
                    parsed_receipt.setdefault("parse_status", "accepted")
                    row["native_parse"] = parsed_receipt
                else:
                    # The established adapter may expose only its owner
                    # matcher; retain a minimal accepted native-parse receipt
                    # without inventing an owner identity.
                    row["native_parse"] = {
                        "valid": True,
                        "parse_status": "accepted",
                        "source": "natural_row_contract",
                    }
            else:
                # A native STOP or malformed admission is not an unmatched
                # owner parse.  Omit owner_match and expose explicit endpoint
                # flags so the finalizer can classify it without charging an
                # artificial unmatched row.
                row.pop("owner_match", None)
                row.pop("native_parse", None)
                if row.get("status") == "native_stop" or row.get("stop_reason") in {"native_stop", "im_end", "eos"}:
                    row["native_stop"] = True
                elif row.get("status") in {"invalid", "over_continuation", "max_budget"}:
                    row["invalid_token"] = True
                row.pop("owner_match_status", None)
            parsed_rows.append(
                {
                    "row": row,
                    "owner_id": owner_text,
                    "strict_match": strict_match,
                    "covered_repeat": covered_repeat,
                    "duplicate": bool(duplicate),
                    "owner_match": owner_match,
                }
            )
        endpoint_owner_ids = sorted(
            {
                item["owner_id"]
                for item in parsed_rows
                if item["strict_match"] and item["owner_id"] is not None
            }
        )
        covered_repeat_owner_ids = sorted(
            {
                item["owner_id"]
                for item in parsed_rows
                if item["covered_repeat"] and item["owner_id"] is not None
            }
        )
        new_target_owner_ids = sorted(set(endpoint_owner_ids) - covered)
        parse = {
            "valid_rows": sum(item["row"].get("status") == "closure" for item in parsed_rows),
            "duplicate_rows": sum(bool(item["duplicate"]) for item in parsed_rows),
            # A closure without a strict physical/source-specific match is a
            # scientific unmatched endpoint, regardless of whether the
            # parser called it ``unique`` with a failed physical check.
            "unmatched_rows": sum(
                item["row"].get("status") == "closure" and not item["strict_match"]
                for item in parsed_rows
            ),
            "ambiguous_rows": sum(
                isinstance(item["owner_match"], Mapping)
                and item["owner_match"].get("status") == "ambiguous"
                for item in parsed_rows
            ),
            "malformed_rows": sum(item["row"].get("status") == "over_continuation" for item in parsed_rows),
            "invalid_rows": sum(item["row"].get("status") in {"invalid", "over_continuation", "max_budget"} for item in parsed_rows),
        }
        horizon_status = result.get("terminal_reason")
        horizon_stop = {
            "stopped": horizon_status in {"native_stop", "invalid", "over_continuation", "max_budget"},
            "stop_reason": horizon_status,
        }
        first_row = result.get("rows", [None])[0] if result.get("rows") else None
        if not isinstance(first_row, Mapping):
            first_row = {}
        row_entry = {
            "admission_mode": first_row.get("admission_mode", result.get("admission_mode")),
            "first_generated_token_id": first_row.get(
                "first_generated_token_id", result.get("first_generated_token_id")
            ),
            "opener_generated_by_model": first_row.get(
                "opener_generated_by_model", result.get("opener_generated_by_model")
            ),
            "opener_injected": first_row.get("opener_injected", result.get("opener_injected")),
            "row_started": (
                first_row.get("opener_generated_by_model") is True
                if first_row
                else None
            ),
            "opener_logits": first_row.get("opener_logits", result.get("opener_logits")),
            "best_logits": first_row.get("best_logits", result.get("best_logits")),
        }
        seen_before: set[str] = set(covered)
        for item in parsed_rows:
            row = item["row"]
            row["owner_bookkeeping"] = {
                "covered_owner_ids_before": sorted(covered),
                "seen_owner_ids_before": sorted(seen_before),
                "matched_owner_id": item["owner_id"],
                "strict_physical_owner_match": bool(item["strict_match"]),
                "covered_repeat": bool(item["covered_repeat"]),
                "duplicate": bool(item["duplicate"]),
                "new_target_owner": bool(item["strict_match"] and item["owner_id"] not in covered),
                "horizon_status": horizon_status,
                "row_entry": dict(row_entry),
                "parse": dict(parse),
                "stop": dict(horizon_stop),
            }
            if item["strict_match"] and item["owner_id"] is not None:
                seen_before.add(item["owner_id"])
        result["owner_bookkeeping"] = {
            "raw_endpoint_owner_ids": endpoint_owner_ids,
            "covered_repeat_owner_ids": covered_repeat_owner_ids,
            "new_target_owner_ids": new_target_owner_ids,
            "row_entry": row_entry,
            "parse": parse,
            "stop": horizon_stop,
            "horizon_status": horizon_status,
            "row_count": len(parsed_rows),
            "strict_physical_owner_match_count": sum(bool(item["strict_match"]) for item in parsed_rows),
            "duplicate_count": int(parse["duplicate_rows"]),
            "unmatched_count": int(parse["unmatched_rows"]),
        }

    def run_matrix(
        self,
        *,
        arms: Sequence[str] = DEFAULT_ARMS,
        residual_actuator: Any | None = None,
        residual_requests: Mapping[str, natural.ResidualRequest] | None = None,
        attention_mask_actuators: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected = tuple(str(arm) for arm in arms)
        if not selected:
            raise GateTechnicalInvalid("S gate requires at least one arm")
        unknown = sorted(set(selected) - DECLARED_ARMS)
        if unknown:
            raise GateTechnicalInvalid(f"unknown S gate arms: {unknown}")
        outputs: dict[str, Any] = {}
        for arm in selected:
            request = (residual_requests or {}).get(arm)
            outputs[arm] = self.run_arm(
                arm,
                residual_actuator=residual_actuator,
                residual_request=request,
                attention_mask_actuator=(attention_mask_actuators or {}).get(arm),
                parity_reference=outputs.get("N00") if arm == "N01" else None,
            )
        baseline_by_arm = {
            "K10": "K01",
            "K11": "K01",
            "K12": "K01",
            "K13": "K01",
            "K14T": "K01",
            "K14B": "K01",
            "N10": "N01",
            "N20": "N01",
            "H10": "H00",
            "H20": "H00",
        }
        matrix_contrasts = {
            arm: self._build_matrix_contrast(
                intervention_arm=arm,
                baseline_arm=baseline,
                outputs=outputs,
            )
            for arm, baseline in baseline_by_arm.items()
            if arm in outputs
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "checkpoint": CHECKPOINT,
            "event_id": EVENT_ID,
            "runtime_identity": dict(self.identity),
            "arms": outputs,
            "arm_order": list(selected),
            "baseline_by_arm": {
                arm: baseline
                for arm, baseline in baseline_by_arm.items()
                if arm in selected
            },
            "matrix_contrasts": matrix_contrasts,
            "no_training": True,
            "gpu_launch_authorized": False,
        }
        result["result_sha256"] = sha256_json(result)
        return result

    @staticmethod
    def _build_matrix_contrast(
        *,
        intervention_arm: str,
        baseline_arm: str,
        outputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compute intervention-minus-own-baseline endpoint deltas only."""

        baseline = outputs.get(baseline_arm)
        intervention = outputs.get(intervention_arm)
        baseline_bookkeeping = baseline.get("owner_bookkeeping") if isinstance(baseline, Mapping) else None
        intervention_bookkeeping = (
            intervention.get("owner_bookkeeping") if isinstance(intervention, Mapping) else None
        )
        required = {
            "raw_endpoint_owner_ids",
            "covered_repeat_owner_ids",
            "row_entry",
            "parse",
            "stop",
        }
        if (
            not isinstance(baseline_bookkeeping, Mapping)
            or not required.issubset(baseline_bookkeeping)
            or not isinstance(intervention_bookkeeping, Mapping)
            or not required.issubset(intervention_bookkeeping)
        ):
            return {
                "status": "unqualified",
                "reason": "missing or invalid own baseline endpoint bookkeeping",
                "baseline_arm": baseline_arm,
                "intervention_arm": intervention_arm,
                "endpoint_delta": None,
            }
        baseline_parse = baseline_bookkeeping.get("parse")
        intervention_parse = intervention_bookkeeping.get("parse")
        baseline_stop = baseline_bookkeeping.get("stop")
        intervention_stop = intervention_bookkeeping.get("stop")
        baseline_row_entry = baseline_bookkeeping.get("row_entry")
        intervention_row_entry = intervention_bookkeeping.get("row_entry")
        if (
            not isinstance(baseline_parse, Mapping)
            or not isinstance(intervention_parse, Mapping)
            or not isinstance(baseline_stop, Mapping)
            or not isinstance(intervention_stop, Mapping)
            or not isinstance(baseline_row_entry, Mapping)
            or not isinstance(intervention_row_entry, Mapping)
            or not isinstance(baseline_bookkeeping.get("raw_endpoint_owner_ids"), list)
            or not isinstance(intervention_bookkeeping.get("raw_endpoint_owner_ids"), list)
            or not isinstance(baseline_bookkeeping.get("covered_repeat_owner_ids"), list)
            or not isinstance(intervention_bookkeeping.get("covered_repeat_owner_ids"), list)
            or not isinstance(baseline_parse.get("unmatched_rows"), int)
            or not isinstance(intervention_parse.get("unmatched_rows"), int)
            or not isinstance(baseline_parse.get("duplicate_rows"), int)
            or not isinstance(intervention_parse.get("duplicate_rows"), int)
            or not isinstance(baseline_parse.get("invalid_rows"), int)
            or not isinstance(intervention_parse.get("invalid_rows"), int)
            or not isinstance(baseline_parse.get("malformed_rows"), int)
            or not isinstance(intervention_parse.get("malformed_rows"), int)
            or not isinstance(baseline_stop.get("stopped"), bool)
            or not isinstance(intervention_stop.get("stopped"), bool)
            or not isinstance(baseline_stop.get("stop_reason"), (str, type(None)))
            or not isinstance(intervention_stop.get("stop_reason"), (str, type(None)))
            or not isinstance(baseline_row_entry.get("admission_mode"), str)
            or not isinstance(intervention_row_entry.get("admission_mode"), str)
            or not isinstance(baseline_row_entry.get("opener_generated_by_model"), bool)
            or not isinstance(intervention_row_entry.get("opener_generated_by_model"), bool)
            or not isinstance(baseline_row_entry.get("opener_injected"), bool)
            or not isinstance(intervention_row_entry.get("opener_injected"), bool)
            or not isinstance(baseline_row_entry.get("row_started"), bool)
            or not isinstance(intervention_row_entry.get("row_started"), bool)
        ):
            return {
                "status": "unqualified",
                "reason": "baseline/intervention endpoint vector is incomplete",
                "baseline_arm": baseline_arm,
                "intervention_arm": intervention_arm,
                "endpoint_delta": None,
            }
        baseline_ids = set(str(value) for value in baseline_bookkeeping["raw_endpoint_owner_ids"])
        intervention_ids = set(str(value) for value in intervention_bookkeeping["raw_endpoint_owner_ids"])
        baseline_first_token = baseline_row_entry.get("first_generated_token_id")
        intervention_first_token = intervention_row_entry.get("first_generated_token_id")
        if (
            baseline_first_token is not None
            and (isinstance(baseline_first_token, bool) or not isinstance(baseline_first_token, int))
        ) or (
            intervention_first_token is not None
            and (isinstance(intervention_first_token, bool) or not isinstance(intervention_first_token, int))
        ):
            return {
                "status": "unqualified",
                "reason": "baseline/intervention first-generated token identity is malformed",
                "baseline_arm": baseline_arm,
                "intervention_arm": intervention_arm,
                "endpoint_delta": None,
            }
        endpoint_delta = {
            "raw_endpoint_owner_ids_added": sorted(intervention_ids - baseline_ids),
            "raw_endpoint_owner_ids_removed": sorted(baseline_ids - intervention_ids),
            "raw_endpoint_owner_count_delta": len(intervention_ids) - len(baseline_ids),
            "covered_repeat_count_delta": len(intervention_bookkeeping["covered_repeat_owner_ids"])
            - len(baseline_bookkeeping["covered_repeat_owner_ids"]),
            "unmatched_count_delta": int(intervention_parse["unmatched_rows"])
            - int(baseline_parse["unmatched_rows"]),
            "duplicate_count_delta": int(intervention_parse["duplicate_rows"])
            - int(baseline_parse["duplicate_rows"]),
            "invalid_count_delta": int(intervention_parse["invalid_rows"])
            - int(baseline_parse["invalid_rows"]),
            "malformed_count_delta": int(intervention_parse["malformed_rows"])
            - int(baseline_parse["malformed_rows"]),
            "stopped_delta": int(intervention_stop["stopped"])
            - int(baseline_stop["stopped"]),
            "stop_reason_changed": baseline_stop["stop_reason"] != intervention_stop["stop_reason"],
            "stop_reason_baseline": baseline_stop["stop_reason"],
            "stop_reason_intervention": intervention_stop["stop_reason"],
            "first_generated_token_id_baseline": baseline_first_token,
            "first_generated_token_id_intervention": intervention_first_token,
            "first_generated_token_id_changed": baseline_first_token != intervention_first_token,
            "opener_generated_by_model_delta": int(intervention_row_entry["opener_generated_by_model"])
            - int(baseline_row_entry["opener_generated_by_model"]),
            "opener_injected_delta": int(intervention_row_entry["opener_injected"])
            - int(baseline_row_entry["opener_injected"]),
            "row_started_delta": int(intervention_row_entry["row_started"])
            - int(baseline_row_entry["row_started"]),
            "admission_mode_changed": baseline_row_entry["admission_mode"]
            != intervention_row_entry["admission_mode"],
            "admission_baseline": dict(baseline_row_entry),
            "admission_intervention": dict(intervention_row_entry),
        }
        return {
            "status": "measured",
            "baseline_arm": baseline_arm,
            "intervention_arm": intervention_arm,
            "baseline_endpoint": {
                "raw_endpoint_owner_ids": sorted(baseline_ids),
                "covered_repeat_owner_ids": list(baseline_bookkeeping["covered_repeat_owner_ids"]),
                "row_entry": dict(baseline_row_entry),
                "unmatched_rows": int(baseline_parse["unmatched_rows"]),
                "invalid_rows": int(baseline_parse["invalid_rows"]),
                "malformed_rows": int(baseline_parse["malformed_rows"]),
                "duplicate_rows": int(baseline_parse["duplicate_rows"]),
                "stop": dict(baseline_stop),
            },
            "intervention_endpoint": {
                "raw_endpoint_owner_ids": sorted(intervention_ids),
                "covered_repeat_owner_ids": list(intervention_bookkeeping["covered_repeat_owner_ids"]),
                "row_entry": dict(intervention_row_entry),
                "unmatched_rows": int(intervention_parse["unmatched_rows"]),
                "invalid_rows": int(intervention_parse["invalid_rows"]),
                "malformed_rows": int(intervention_parse["malformed_rows"]),
                "duplicate_rows": int(intervention_parse["duplicate_rows"]),
                "stop": dict(intervention_stop),
            },
            "endpoint_delta": endpoint_delta,
        }


def resolve_s_primary_binding(
    *,
    config_path: Path = DEFAULT_CONFIG,
    panel_path: Path = DEFAULT_PANEL,
    cohort_path: Path = DEFAULT_COHORT,
    h0_root: Path = DEFAULT_H0_ROOT,
    h0_dir: Path | None = None,
    event_id: str = EVENT_ID,
    pre_gpu_receipt: Path | None = None,
    output_root: str | Path | None = None,
) -> LiveRuntimeBinding:
    """Resolve the frozen S event through the established old runtime loader."""

    if str(event_id) != EVENT_ID:
        raise GateTechnicalInvalid(f"S primary gate is frozen to event {EVENT_ID}, not {event_id!r}")
    if pre_gpu_receipt is None:
        raise GateTechnicalInvalid(
            "a pre-GPU receipt is required before loading the live S checkpoint"
        )
    receipt_identity = _load_pre_gpu_identity(pre_gpu_receipt, output_root=output_root)
    preload_cuda = _require_preload_cuda_visibility()
    orchestrator = legacy.OwnerInterfaceOrchestrator(
        checkpoint=CHECKPOINT,
        stage="all",
        output_dir=DEFAULT_OUTPUT_ROOT,
        config_path=Path(config_path),
        panel_path=Path(panel_path),
        cohort_path=Path(cohort_path),
        h0_root=Path(h0_root),
        h0_dir=None if h0_dir is None else Path(h0_dir),
        fail_collision=False,
    )
    try:
        identity = orchestrator._load_cpu_contract()  # noqa: SLF001 - reuse stable owner-interface loader
        event = next(
            (candidate for candidate in orchestrator.events if str(candidate.get("gt_owner_id")) == EVENT_ID),
            None,
        )
        if not isinstance(event, Mapping):
            raise GateTechnicalInvalid(f"frozen S event {EVENT_ID} is absent from the resolved cohort")
        _validate_frozen_event_geometry(event)
        adapter = orchestrator._load_adapter(identity)  # noqa: SLF001 - reuse established HF session loader
        seeded_context = legacy._make_event_context(adapter, event)  # noqa: SLF001 - bind exact old H0 context
        runtime = getattr(seeded_context, "runtime", None)
        if runtime is None:
            raise GateTechnicalInvalid("resolved S event context lacks its image runtime")
        postload_cuda = _attest_postload_cuda_device(
            LiveRuntimeBinding(
                adapter=adapter,
                runtime=runtime,
                event=event,
                seeded_context=seeded_context,
            )
        )
        bound_identity = dict(identity)
        bound_identity.update(receipt_identity)
        bound_identity["cuda_visible_devices"] = preload_cuda
        bound_identity["model_device_attestation"] = {
            **postload_cuda,
            "cuda_visible_devices": preload_cuda,
        }
        return LiveRuntimeBinding(
            adapter=adapter,
            runtime=runtime,
            event=event,
            seeded_context=seeded_context,
            identity=bound_identity,
            close_callback=lambda: _close_legacy_adapter(adapter),
        )
    except Exception:
        # The loader owns a backend context; close it before propagating a
        # technical failure so a failed gate cannot leak a model/session.
        try:
            adapter = locals().get("adapter")
            if adapter is not None:
                _close_legacy_adapter(adapter)
        finally:
            raise


def _close_legacy_adapter(adapter: Any) -> None:
    close = getattr(adapter, "close", None)
    if callable(close):
        close()
    session_context = getattr(adapter, "_session_context", None)
    if session_context is not None and callable(getattr(session_context, "__exit__", None)):
        session_context.__exit__(None, None, None)


def build_runtime_identity(gate: SPrimaryNaturalBoundaryGate) -> dict[str, Any]:
    identity = {
        "schema_version": f"{SCHEMA_VERSION}.runtime_identity.v1",
        "unit_id": UNIT_ID,
        "checkpoint": CHECKPOINT,
        "event_id": EVENT_ID,
        "execution_mode": "live_scalar_recompute",
        "no_training": True,
        "gpu_launch_authorized": False,
        "resolved": dict(gate.binding.identity),
        "event": dict(gate.identity),
        "model_device": str(gate.scalar.device),
        "sdpa_backend": _sdpa_backend_receipt(),
    }
    for key in (
        "pre_gpu_receipt_path",
        "pre_gpu_receipt_sha256",
        "pre_gpu_receipt_self_sha256",
        "code_hashes",
        "source_code_hashes",
        "cuda_visible_devices",
        "model_device_attestation",
        "gate_output_root",
        "authorized_gate_output_root",
        "gate_output_binding",
    ):
        if key in gate.binding.identity:
            identity[key] = gate.binding.identity[key]
    # Hash only the identity body; this avoids nondeterministic timestamps/PIDs.
    identity["identity_sha256"] = sha256_json(identity)
    return identity


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def persist_failure_receipt(output_root: str | Path, error: BaseException | str) -> dict[str, Any]:
    """Write deterministic failure bytes and a relative-path receipt."""

    root = Path(output_root).expanduser()
    if not root.is_absolute():
        raise GateTechnicalInvalid("failure output root must be an absolute path")
    root = root.resolve()
    # Failure evidence is immutable just like a success artifact.  Claim the
    # whole root atomically; never overwrite a success, append to a partial
    # run, or replace a different (or even identical) prior failure.
    if root.exists():
        raise FileExistsError(f"failure artifact output collision: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    stderr = str(error)
    payload = stderr.encode("utf-8", errors="surrogateescape")
    failure_path = root / "failure.stderr"
    failure_path.write_bytes(payload)
    observed = failure_path.read_bytes()
    digest = hashlib.sha256(observed).hexdigest()
    receipt = {
        "schema_version": f"{SCHEMA_VERSION}.failure.v1",
        "status": "technical_invalidity",
        "failure_file": failure_path.name,
        "sha256": digest,
        "size_bytes": len(observed),
        "verbatim_stderr": observed.decode("utf-8", errors="surrogateescape"),
    }
    _write_json(root / "failure.json", receipt)
    return receipt


def write_gate_receipts(
    output_root: str | Path,
    *,
    runtime_identity: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"gate artifact output collision: {root}")
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "runtime_identity.json", dict(runtime_identity))
    _write_json(root / "result.json", dict(result))
    terminal = {
        "schema_version": f"{SCHEMA_VERSION}.terminal.v1",
        "status": "completed",
        "event_id": EVENT_ID,
        "checkpoint": CHECKPOINT,
        "arm_order": result.get("arm_order", []),
        "result_sha256": result.get("result_sha256"),
    }
    _write_json(root / "terminal_summary.json", terminal)
    return terminal


def run_s_primary_natural_boundary_gate(
    *,
    binding: LiveRuntimeBinding | None = None,
    arms: Sequence[str] = DEFAULT_ARMS,
    residual_actuator: Any | None = None,
    residual_requests: Mapping[str, natural.ResidualRequest] | None = None,
    attention_mask_actuators: Mapping[str, Any] | None = None,
    output_root: str | Path | None = None,
    **resolve_kwargs: Any,
) -> dict[str, Any]:
    """Execute the S gate, optionally writing immutable deterministic receipts."""

    requested = tuple(str(arm) for arm in arms)
    if binding is None:
        unknown = sorted(set(requested) - DECLARED_ARMS)
        if unknown:
            raise GateTechnicalInvalid(f"unknown S gate arms: {unknown}")
        missing: list[str] = []
        for arm in requested:
            if arm in RESIDUAL_ARMS - {"N00"} and residual_actuator is None:
                missing.append(arm)
            if arm in ATTENTION_ARMS and (attention_mask_actuators or {}).get(arm) is None:
                missing.append(arm)
        if missing:
            raise GateTechnicalInvalid(
                "live S binding is blocked before model load because actuator callbacks are missing: "
                f"{sorted(set(missing))}"
            )
    owned_binding = binding is None
    resolved: LiveRuntimeBinding | None = binding
    try:
        if resolved is None:
            if output_root is not None:
                # The requested destination is part of the producer/consumer
                # binding; pass it into the pre-GPU identity check before any
                # model is loaded.
                resolve_kwargs = {**resolve_kwargs, "output_root": output_root}
            resolved = resolve_s_primary_binding(**resolve_kwargs)
        gate = SPrimaryNaturalBoundaryGate.from_binding(resolved)
        runtime_identity = build_runtime_identity(gate)
        result = gate.run_matrix(
            arms=arms,
            residual_actuator=residual_actuator,
            residual_requests=residual_requests,
            attention_mask_actuators=attention_mask_actuators,
        )
        result = {**result, "runtime_identity_sha256": runtime_identity["identity_sha256"]}
        result_body = dict(result)
        result_body.pop("result_sha256", None)
        result["result_sha256"] = sha256_json(result_body)
        if output_root is not None:
            write_gate_receipts(output_root, runtime_identity=runtime_identity, result=result)
        return {"runtime_identity": runtime_identity, "result": result}
    except Exception as exc:
        if output_root is not None:
            persist_failure_receipt(output_root, exc)
        raise
    finally:
        if owned_binding and resolved is not None:
            resolved.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--h0-root", type=Path, default=DEFAULT_H0_ROOT)
    parser.add_argument("--h0-dir", type=Path, default=None)
    parser.add_argument("--pre-gpu-receipt", type=Path, required=True)
    parser.add_argument("--event-id", default=EVENT_ID)
    parser.add_argument("--max-rows", type=int, default=natural.MAX_ROWS)
    parser.add_argument("--max-row-tokens", type=int, default=natural.MAX_ROW_TOKENS)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.event_id != EVENT_ID:
        raise SystemExit(f"--event-id is frozen to {EVENT_ID}")
    unknown_arms = sorted(set(args.arms) - DECLARED_ARMS)
    if unknown_arms:
        error = GateTechnicalInvalid(f"unknown S gate arms: {unknown_arms}")
        try:
            persist_failure_receipt(args.output_root, error)
        except Exception:
            pass
        print(json.dumps({"status": "technical_invalidity", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    try:
        binding = resolve_s_primary_binding(
            config_path=args.config,
            panel_path=args.panel,
            cohort_path=args.cohort,
            h0_root=args.h0_root,
            h0_dir=args.h0_dir,
            event_id=args.event_id,
            pre_gpu_receipt=args.pre_gpu_receipt,
            output_root=args.output_root,
        )
        gate = SPrimaryNaturalBoundaryGate.from_binding(
            binding,
            max_rows=args.max_rows,
            max_row_tokens=args.max_row_tokens,
        )
        runtime_identity = build_runtime_identity(gate)
        attention_actuators = (
            build_live_attention_mask_actuators(binding, gate.context)
            if set(args.arms) & set(ATTENTION_ARMS)
            else None
        )
        residual_actuator = (
            build_live_residual_actuator()
            if set(args.arms) & (set(RESIDUAL_ARM_IDS) - {"N00"})
            else None
        )
        result = gate.run_matrix(
            arms=args.arms,
            residual_actuator=residual_actuator,
            attention_mask_actuators=attention_actuators,
        )
        result = {**result, "runtime_identity_sha256": runtime_identity["identity_sha256"]}
        result_body = dict(result)
        result_body.pop("result_sha256", None)
        result["result_sha256"] = sha256_json(result_body)
        write_gate_receipts(args.output_root, runtime_identity=runtime_identity, result=result)
        print(json.dumps({"status": "completed", "result_sha256": result["result_sha256"]}, sort_keys=True))
        return 0
    except (GateTechnicalInvalid, legacy.OrchestrationError, FileExistsError, OSError, ValueError) as exc:
        try:
            persist_failure_receipt(args.output_root, exc)
        except Exception:
            pass
        print(json.dumps({"status": "technical_invalidity", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    finally:
        # ``binding`` is intentionally local to keep failed loader construction
        # from becoming a leaked live session.
        candidate = locals().get("binding")
        if candidate is not None:
            candidate.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Experiment-local P1 observational image-field census.

This module is a deliberately small, provisional seam for the
``2026-08-05-static-dynamic-owner-interface-crossover`` unit.  ``contract``
and ``dry-run`` validate CPU-side identities without loading a model;
``capture`` and ``live`` load and attest the owning owner-interface HF runtime;
``merge`` is CPU-only.  A live capture performs exactly one prefill for an
image/checkpoint pair, captures the image field and H0 terminal query
positions, and computes CPU-sized descriptive readouts.

The implementation keeps three concerns separate:

* :func:`capture_native_prefill` is the one-forward, fail-closed hook seam;
* :func:`compute_observational_census` is pure tensor/geometry arithmetic;
* :func:`write_shard`/:func:`merge_shards` own immutable shard receipts.

No tensor payload is written to an artifact.  Only small scalar/shape/dtype/
device/hash receipts are persisted.  The readouts are descriptive and do not
contain efficacy thresholds or causal decisions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import subprocess
from typing import Any, Literal

import torch


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
SCHEMA_VERSION = "static_dynamic_owner_observational_census.v1"
RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.receipt"
MANIFEST_SCHEMA_VERSION = f"{SCHEMA_VERSION}.manifest"
P1_CENSUS_SCHEMA_VERSION = f"{SCHEMA_VERSION}.p1"
CHECKPOINTS = ("S", "A")
LAYERS = tuple(range(28))
LAYER_NAMES = (
    "merger_output",
    "block_0_input",
    *(f"block_{index}_output" for index in LAYERS),
    "final_norm",
)
TERMINAL_TOKEN_NAMES = {"S": "box_end", "A": "commit"}
EXPECTED_WRAPPERS = {"S": "object_box_closed", "A": "object_box_commit"}
DEFAULT_SHUFFLE_SEED = 20260805
NOOP_TOLERANCE = 1.0e-4
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CensusContractError(ValueError):
    """Raised when a census identity or mechanical contract is not valid."""


class TechnicalInvalid(CensusContractError):
    """Raised for evidence that must be quarantined rather than interpreted."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CensusContractError("value is not canonical JSON serializable") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_tensor(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise CensusContractError("tensor hash requires a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    try:
        payload = tensor.numpy().tobytes()
    except TypeError as exc:
        raise CensusContractError(f"unsupported tensor dtype for hashing: {tensor.dtype}") from exc
    return sha256_bytes(
        str(tensor.dtype).encode("ascii")
        + canonical_json_bytes(list(tensor.shape))
        + payload
    )


def sha256_token_ids(token_ids: torch.Tensor | Sequence[int]) -> str:
    values = (
        token_ids.detach().reshape(-1).cpu().tolist()
        if isinstance(token_ids, torch.Tensor)
        else list(token_ids)
    )
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values):
        raise CensusContractError("token IDs must be non-negative integers")
    return sha256_json([int(item) for item in values])


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("last_hidden_state", "hidden_states"):
            candidate = output.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate
    raise TechnicalInvalid("hook output does not expose a tensor")


def _replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    if isinstance(output, Mapping):
        updated = dict(output)
        for key in ("last_hidden_state", "hidden_states"):
            if key in updated:
                updated[key] = tensor
                return updated
    raise TechnicalInvalid("hook output does not support first-tensor replacement")


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError) as exc:
        raise TechnicalInvalid("model has no attested parameter device") from exc


def validate_single_numeric_cuda_visible_devices(raw: str | None = None) -> dict[str, Any]:
    """Validate the single-physical-GPU contract used by support capture."""

    value = os.environ.get("CUDA_VISIBLE_DEVICES") if raw is None else str(raw)
    if value is None or not value.strip():
        raise CensusContractError("CUDA_VISIBLE_DEVICES must name exactly one numeric device")
    tokens = [token.strip() for token in value.split(",")]
    if len(tokens) != 1 or not tokens[0].isdigit():
        raise CensusContractError("CUDA_VISIBLE_DEVICES must contain exactly one numeric token")
    return {
        "raw": value,
        "tokens": tokens,
        "selected_physical_device": tokens[0],
        "count": 1,
    }


def _query_nvidia_smi_physical_gpu(physical_device_id: str, *, executable: str | None = None) -> dict[str, Any]:
    if re.fullmatch(r"[0-9]+", physical_device_id) is None:
        raise TechnicalInvalid("physical GPU ID must be one numeric CUDA_VISIBLE_DEVICES token")
    executable = executable or shutil.which("nvidia-smi")
    if executable is None:
        raise TechnicalInvalid("nvidia-smi is unavailable for physical GPU attestation")
    try:
        completed = subprocess.run(
            [executable, "-i", physical_device_id, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TechnicalInvalid(f"nvidia-smi physical GPU query failed: {exc}") from exc
    if completed.returncode != 0:
        raise TechnicalInvalid("nvidia-smi physical GPU query returned non-zero")
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise TechnicalInvalid("nvidia-smi query did not return exactly one physical GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 2 or fields[0] != physical_device_id or re.fullmatch(r"GPU-[0-9A-Fa-f-]+", fields[1]) is None:
        raise TechnicalInvalid("nvidia-smi physical index/UUID mapping is malformed")
    return {"physical_device_id": physical_device_id, "physical_device_index": int(physical_device_id), "physical_device_uuid": fields[1]}


def attest_live_physical_gpu(
    *,
    model: Any,
    logical_device: torch.device | str | None = None,
    cuda_visible_devices: str | None = None,
    nvidia_smi_executable: str | None = None,
) -> dict[str, Any]:
    """Return nvidia-smi↔torch UUID/UTC/PID live device attestation.

    The function intentionally accepts an optional logical device so tests can
    inject ``cuda:0`` while production callers use the model's first parameter.
    A UUID is obtained from ``torch.cuda.get_device_properties`` when the
    backend exposes one; absence is technical invalidity, not a guessed UUID.
    """

    visible = validate_single_numeric_cuda_visible_devices(cuda_visible_devices)
    device = torch.device(logical_device) if logical_device is not None else _model_device(model)
    if device.type != "cuda":
        raise TechnicalInvalid("observational census requires a CUDA model device")
    if not torch.cuda.is_available():
        raise TechnicalInvalid("CUDA is not available for live physical attestation")
    index = int(device.index or 0)
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise TechnicalInvalid(f"logical CUDA device index {index} is outside device_count={count}")
    properties = torch.cuda.get_device_properties(index)
    uuid_value = getattr(properties, "uuid", None)
    if uuid_value is None:
        raise TechnicalInvalid("CUDA device properties expose no physical UUID")
    torch_uuid_raw = str(uuid_value)
    torch_uuid_text = torch_uuid_raw.lower().removeprefix("gpu-")
    if re.fullmatch(r"[0-9a-f-]+", torch_uuid_text) is None:
        raise TechnicalInvalid("CUDA physical UUID is empty")
    physical = _query_nvidia_smi_physical_gpu(visible["selected_physical_device"], executable=nvidia_smi_executable)
    physical_uuid_text = str(physical["physical_device_uuid"]).lower().removeprefix("gpu-")
    if physical_uuid_text != torch_uuid_text:
        raise TechnicalInvalid("nvidia-smi and torch physical UUIDs disagree")
    return {
        "status": "validated",
        "passed": True,
        "device": str(device),
        "logical_device": str(device),
        "physical_device_id": visible["selected_physical_device"],
        "physical_device_index": int(visible["selected_physical_device"]),
        **physical,
        "physical_device_uuid_raw": str(physical["physical_device_uuid"]),
        "physical_device_uuid_normalized": physical_uuid_text,
        "torch_device_uuid_raw": torch_uuid_raw,
        "cuda_visible_devices": visible,
        "device_name": str(getattr(properties, "name", "")),
        "memory_total": int(getattr(properties, "total_memory", 0)),
        "current_device": int(torch.cuda.current_device()),
        "pid": os.getpid(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CensusContractError(f"{label} must be an integer >= {minimum}")
    return int(value)


def parse_shard_selector(value: str, *, default_checkpoint: str | None = None) -> tuple[str, int, int]:
    """Parse explicit ``checkpoint:i/n`` or caller-bound ``i/n`` selectors."""

    text = str(value).strip()
    match = re.fullmatch(r"(?:(S|A):)?([0-9]+)/([0-9]+)", text)
    if match is None:
        raise CensusContractError("shard selector must have form i/n or S:i/n")
    parsed_checkpoint, index_text, count_text = match.groups()
    checkpoint = parsed_checkpoint or default_checkpoint
    if checkpoint not in CHECKPOINTS:
        raise CensusContractError("unprefixed shard selector requires the checkpoint binding")
    index, count = int(index_text), int(count_text)
    if count <= 0 or index < 0 or index >= count:
        raise CensusContractError("shard selector index must satisfy 0 <= i < n")
    return checkpoint, index, count


def partition_image_ids(image_ids: Sequence[int | str], *, shard_index: int, shard_count: int) -> tuple[int, ...]:
    """Deterministically partition sorted image IDs by round-robin index."""

    count = _strict_int(shard_count, "shard_count", minimum=1)
    index = _strict_int(shard_index, "shard_index", minimum=0)
    if index >= count:
        raise CensusContractError("shard_index must be less than shard_count")
    normalized = sorted({_strict_int(int(value), "image_id", minimum=0) for value in image_ids})
    return tuple(normalized[position] for position in range(index, len(normalized), count))


@dataclass(frozen=True)
class WrapperContract:
    checkpoint: Literal["S", "A"]
    wrapper: str
    terminal_token_id: int
    terminal_token_name: str
    h0_prefix_sha256: str
    mrope_sha256: str
    prefix_token_ids_sha256: str

    def validate(self) -> None:
        if self.checkpoint not in CHECKPOINTS:
            raise CensusContractError("checkpoint must be S or A")
        if self.wrapper != EXPECTED_WRAPPERS[self.checkpoint]:
            raise CensusContractError("wrapper is not the checkpoint-native wrapper")
        if self.terminal_token_name != TERMINAL_TOKEN_NAMES[self.checkpoint]:
            raise CensusContractError("terminal token name is not checkpoint-native")
        _strict_int(self.terminal_token_id, "terminal_token_id", minimum=0)
        for name, value in (
            ("h0_prefix_sha256", self.h0_prefix_sha256),
            ("mrope_sha256", self.mrope_sha256),
            ("prefix_token_ids_sha256", self.prefix_token_ids_sha256),
        ):
            if not _is_sha256(value):
                raise CensusContractError(f"{name} must be a lowercase SHA-256")

    def receipt(self) -> dict[str, Any]:
        self.validate()
        return {
            "checkpoint": self.checkpoint,
            "wrapper": self.wrapper,
            "terminal_token_id": self.terminal_token_id,
            "terminal_token_name": self.terminal_token_name,
            "h0_prefix_sha256": self.h0_prefix_sha256,
            "mrope_sha256": self.mrope_sha256,
            "prefix_token_ids_sha256": self.prefix_token_ids_sha256,
        }

    def recipe_receipt(self) -> dict[str, Any]:
        """Return only the checkpoint-common wrapper recipe identity."""

        self.validate()
        return {
            "checkpoint": self.checkpoint,
            "wrapper": self.wrapper,
            "terminal_token_id": self.terminal_token_id,
            "terminal_token_name": self.terminal_token_name,
        }


def validate_terminal_query_binding(
    binding: Mapping[str, Any],
    *,
    input_ids: torch.Tensor,
    contract: WrapperContract,
    expected_prefix_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an exact H0-native terminal token position; synthetic queries fail."""

    contract.validate()
    if not isinstance(binding, Mapping):
        raise TechnicalInvalid("terminal query binding must be a mapping")
    source = binding.get("source", binding.get("query_source"))
    if source not in {"native_h0_complete_row", "h0_complete_row_terminal_state"}:
        raise TechnicalInvalid("terminal query state must come from an exact native H0 complete row")
    if binding.get("synthetic") is True or binding.get("teacher_forced") is True:
        raise TechnicalInvalid("synthetic or teacher-forced terminal query states are forbidden")
    position = _strict_int(binding.get("position"), "terminal query position", minimum=0)
    ids = input_ids.detach().reshape(-1).cpu()
    if position >= ids.numel():
        raise TechnicalInvalid("terminal query position is outside the exact H0 forward")
    declared_token = _strict_int(binding.get("token_id"), "terminal query token_id", minimum=0)
    observed_token = int(ids[position].item())
    if declared_token != contract.terminal_token_id or observed_token != contract.terminal_token_id:
        raise TechnicalInvalid("terminal query token is not the checkpoint-native closure token")
    prefix_hash = binding.get("prefix_sha256", binding.get("h0_prefix_sha256"))
    expected_prefix = contract.h0_prefix_sha256 if expected_prefix_sha256 is None else expected_prefix_sha256
    if prefix_hash != expected_prefix:
        raise TechnicalInvalid("terminal query H0 prefix hash disagrees with wrapper contract")
    owner_id = binding.get("owner_id", binding.get("gt_owner_id"))
    if not isinstance(owner_id, str) or not owner_id:
        raise TechnicalInvalid("terminal query binding has no physical owner ID")
    normalized = {
        "owner_id": owner_id,
        "position": position,
        "token_id": declared_token,
        "token_name": contract.terminal_token_name,
        "source": source,
        "prefix_sha256": prefix_hash,
        "synthetic": False,
    }
    # ``natural_boundary`` counts strict covered owners.  It is deliberately
    # not a physical row index because unmatched/duplicate physical rows may
    # occur between strict rows in the exact H0 history.
    if "natural_boundary" in binding:
        boundary = _strict_int(binding["natural_boundary"], "terminal query natural_boundary", minimum=1)
        terminal_row_index = _strict_int(
            binding.get("terminal_row_index"),
            "terminal query terminal_row_index",
            minimum=0,
        )
        normalized["natural_boundary"] = boundary
        normalized["terminal_row_index"] = terminal_row_index
    for key in ("latest_covered_owner_id", "physical_terminal_owner_id"):
        if key not in binding:
            continue
        owner_value = binding[key]
        if owner_value is not None and (not isinstance(owner_value, str) or not owner_value):
            raise TechnicalInvalid(f"terminal query {key} is invalid")
        normalized[key] = owner_value
    if "natural_boundary" in binding:
        strict_raw = binding.get("strict_covered_owner_ids")
        if (
            not isinstance(strict_raw, (list, tuple))
            or any(not isinstance(value, str) or not value for value in strict_raw)
            or len(set(strict_raw)) != len(strict_raw)
        ):
            raise TechnicalInvalid("terminal query strict covered-owner sequence is invalid")
        strict_owner_ids = [str(value) for value in strict_raw]
        if len(strict_owner_ids) != normalized["natural_boundary"]:
            raise TechnicalInvalid("terminal query strict covered-owner count differs from natural boundary")
        if not strict_owner_ids or strict_owner_ids[-1] != normalized.get("latest_covered_owner_id"):
            raise TechnicalInvalid("terminal query latest strict owner differs from its authoritative sequence")
        strict_sha = binding.get("strict_covered_owner_ids_sha256")
        if strict_sha != sha256_json(strict_owner_ids):
            raise TechnicalInvalid("terminal query strict covered-owner hash is invalid")
        prefix_row_count = _strict_int(
            binding.get("physical_prefix_row_count"),
            "terminal query physical_prefix_row_count",
            minimum=1,
        )
        if prefix_row_count != normalized["terminal_row_index"] + 1:
            raise TechnicalInvalid("terminal query physical prefix row count differs from terminal row index")
        closure_step = _strict_int(
            binding.get("physical_terminal_closure_step"),
            "terminal query physical_terminal_closure_step",
            minimum=0,
        )
        physical_status = binding.get("physical_terminal_match_status")
        if physical_status not in {"tp", "unmatched"}:
            raise TechnicalInvalid("terminal query physical terminal match status is invalid")
        physical_owner = normalized.get("physical_terminal_owner_id")
        if (physical_status == "tp") != isinstance(physical_owner, str):
            raise TechnicalInvalid("terminal query physical terminal owner disagrees with global match status")
        intervening_count = _strict_int(
            binding.get("intervening_unmatched_row_count"),
            "terminal query intervening_unmatched_row_count",
            minimum=0,
        )
        intervening_sha = binding.get("intervening_unmatched_rows_sha256")
        if not _is_sha256(intervening_sha):
            raise TechnicalInvalid("terminal query intervening unmatched-row hash is invalid")
        normalized.update({
            "strict_covered_owner_ids": strict_owner_ids,
            "strict_covered_owner_ids_sha256": strict_sha,
            "physical_prefix_row_count": prefix_row_count,
            "physical_terminal_closure_step": closure_step,
            "physical_terminal_match_status": physical_status,
            "intervening_unmatched_row_count": intervening_count,
            "intervening_unmatched_rows_sha256": intervening_sha,
        })
    return normalized


def _resolve_layer(model: Any, index: int) -> Any:
    candidates = (
        "model.language_model.layers",
        "model.model.language_model.layers",
        "language_model.layers",
        "layers",
    )
    found: list[Any] = []
    for path in candidates:
        owner = model
        try:
            for part in path.split("."):
                owner = getattr(owner, part)
            layers = owner
            if 0 <= index < len(layers):
                found.append(layers[index])
        except (AttributeError, TypeError, IndexError):
            continue
    unique = {id(item): item for item in found}
    if len(unique) != 1:
        raise TechnicalInvalid(f"decoder block {index} is missing or ambiguous")
    return next(iter(unique.values()))


def _resolve_optional_module(model: Any, paths: Sequence[str]) -> Any | None:
    for path in paths:
        owner = model
        try:
            for part in path.split("."):
                owner = getattr(owner, part)
        except AttributeError:
            continue
        if owner is not None and hasattr(owner, "register_forward_hook"):
            return owner
    return None


def _validate_state_tensor(tensor: torch.Tensor, *, name: str, device: torch.device) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        raise TechnicalInvalid(f"{name} did not produce a non-empty tensor")
    if tensor.device != device:
        raise TechnicalInvalid(f"{name} device {tensor.device} differs from model device {device}")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise TechnicalInvalid(f"{name} contains non-finite values")


def _select_sequence_positions(
    tensor: torch.Tensor,
    positions: Sequence[int],
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise TechnicalInvalid(f"{name} must have shape [1,S,H]")
    if not positions or max(positions) >= tensor.shape[1]:
        raise TechnicalInvalid(f"{name} positions are outside the model sequence")
    _validate_state_tensor(tensor, name=name, device=device)
    return tensor[0, list(positions), :].detach().clone()


@dataclass
class NativePrefillCapture:
    checkpoint: str
    image_id: int
    states: dict[str, torch.Tensor]
    terminal_states: dict[str, dict[str, torch.Tensor]]
    receipt: dict[str, Any]


class _HookCapture:
    def __init__(
        self,
        *,
        name: str,
        image_positions: Sequence[int],
        terminal_positions: Mapping[str, int],
        device: torch.device,
        merger: bool = False,
    ) -> None:
        self.name = name
        self.image_positions = tuple(int(value) for value in image_positions)
        self.terminal_positions = {str(key): int(value) for key, value in terminal_positions.items()}
        self.device = device
        self.merger = merger
        self.count = 0
        self.state: torch.Tensor | None = None
        self.terminal: dict[str, torch.Tensor] = {}
        self.shape: list[int] | None = None
        self.dtype: str | None = None
        self.device_name: str | None = None
        self.finite: bool | None = None

    def capture(self, output: Any) -> Any:
        tensor = _first_tensor(output)
        self.count += 1
        if self.count != 1:
            raise TechnicalInvalid(f"{self.name} hook fired more than once")
        _validate_state_tensor(tensor, name=self.name, device=self.device)
        self.shape = list(tensor.shape)
        self.dtype = str(tensor.dtype)
        self.device_name = str(tensor.device)
        self.finite = bool(torch.isfinite(tensor.detach()).all().item())
        if self.merger:
            if tensor.ndim == 2:
                if tensor.shape[0] != len(self.image_positions):
                    raise TechnicalInvalid("merger output rows do not match image span")
                selected = tensor
            elif tensor.ndim == 3 and tensor.shape[0] == 1:
                if tensor.shape[1] != len(self.image_positions):
                    raise TechnicalInvalid("merger output tokens do not match image span")
                selected = tensor[0]
            else:
                raise TechnicalInvalid("merger output must have shape [N,H] or [1,N,H]")
        else:
            selected = _select_sequence_positions(
                tensor,
                self.image_positions,
                name=self.name,
                device=self.device,
            )
        self.state = selected.detach().clone()
        if not self.merger:
            for owner_id, position in self.terminal_positions.items():
                if position >= tensor.shape[1]:
                    raise TechnicalInvalid(f"terminal position for {owner_id} is outside {self.name}")
                self.terminal[owner_id] = tensor[0, position, :].detach().clone()
        return output

    def receipt(self) -> dict[str, Any]:
        if self.state is None or self.shape is None:
            raise TechnicalInvalid(f"{self.name} hook did not capture a state")
        return {
            "shape": list(self.shape),
            "selected_shape": list(self.state.shape),
            "dtype": self.dtype,
            "device": self.device_name,
            "finite": self.finite,
            "state_sha256": sha256_tensor(self.state),
            "call_count": self.count,
            "terminal_query_count": len(self.terminal),
            "terminal_query_sha256": {
                owner: sha256_tensor(state) for owner, state in sorted(self.terminal.items())
            },
        }


def _forward_native(
    model: Any,
    *,
    input_ids: torch.Tensor,
    model_inputs: Mapping[str, Any] | None,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> Any:
    kwargs: dict[str, Any] = {
        key: value
        for key, value in (model_inputs or {}).items()
        if key not in {"input_ids", "position_ids", "attention_mask", "past_key_values", "cache_position", "use_cache"}
    }
    kwargs["input_ids"] = input_ids
    kwargs["use_cache"] = False
    kwargs["return_dict"] = True
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return model(**kwargs)


def capture_native_prefill(
    model: Any,
    *,
    checkpoint: Literal["S", "A"],
    image_id: int,
    input_ids: torch.Tensor,
    image_positions: Sequence[int],
    terminal_bindings: Sequence[Mapping[str, Any]],
    contract: WrapperContract,
    position_ids: torch.Tensor,
    model_inputs: Mapping[str, Any] | None = None,
    attention_mask: torch.Tensor | None = None,
    merger_module: Any | None = None,
    final_norm_module: Any | None = None,
    layer_indices: Sequence[int] = LAYERS,
    expected_image_span_sha256: str | None = None,
    terminal_prefix_hashes: Mapping[str, str] | None = None,
    terminal_query_statuses: Mapping[str, Mapping[str, Any]] | None = None,
) -> NativePrefillCapture:
    """Capture one native prefill and all P1 states, with exact cleanup."""

    contract.validate()
    if checkpoint != contract.checkpoint:
        raise TechnicalInvalid("capture checkpoint differs from wrapper contract")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise TechnicalInvalid("input_ids must have shape [1,S]")
    if position_ids.ndim != 3 or int(position_ids.shape[1]) != 1 or int(position_ids.shape[2]) != int(input_ids.shape[1]):
        raise TechnicalInvalid("position_ids must have native [3,1,S] or [4,1,S] shape")
    if int(position_ids.shape[0]) not in {3, 4}:
        raise TechnicalInvalid("position_ids first axis must be 3 or an attested 4")
    device = _model_device(model)
    if input_ids.device != device or position_ids.device != device:
        raise TechnicalInvalid("input_ids and position_ids must be on the model device")
    image_positions_tuple = tuple(int(value) for value in image_positions)
    if not image_positions_tuple or len(set(image_positions_tuple)) != len(image_positions_tuple):
        raise TechnicalInvalid("image positions must be unique and non-empty")
    if min(image_positions_tuple) < 0 or max(image_positions_tuple) >= input_ids.shape[1]:
        raise TechnicalInvalid("image position is outside the exact prefill sequence")
    image_span_hash = sha256_json(list(image_positions_tuple))
    if expected_image_span_sha256 is not None and image_span_hash != expected_image_span_sha256:
        raise TechnicalInvalid("image span hash differs from the bound H0 prefix")
    bindings: dict[str, dict[str, Any]] = {}
    for binding in terminal_bindings:
        binding_owner = binding.get("owner_id", binding.get("gt_owner_id")) if isinstance(binding, Mapping) else None
        expected_prefix = None if terminal_prefix_hashes is None else terminal_prefix_hashes.get(str(binding_owner))
        normalized = validate_terminal_query_binding(
            binding,
            input_ids=input_ids,
            contract=contract,
            expected_prefix_sha256=expected_prefix,
        )
        owner_id = normalized["owner_id"]
        if owner_id in bindings:
            raise TechnicalInvalid(f"duplicate terminal query binding for {owner_id}")
        bindings[owner_id] = normalized
    normalized_statuses: dict[str, dict[str, Any]] = {}
    if terminal_query_statuses is not None:
        for owner_id, value in terminal_query_statuses.items():
            if not isinstance(owner_id, str) or not owner_id or not isinstance(value, Mapping):
                raise TechnicalInvalid("terminal query statuses must map owner IDs to receipts")
            status = value.get("status")
            if status not in {"measured", "not_measured"}:
                raise TechnicalInvalid(f"terminal query status is invalid for {owner_id}")
            reason = value.get("not_measured_reason")
            if status == "not_measured" and (not isinstance(reason, str) or not reason):
                raise TechnicalInvalid(f"terminal query not_measured reason is missing for {owner_id}")
            if status == "measured" and owner_id not in bindings:
                raise TechnicalInvalid(f"measured terminal query has no binding for {owner_id}")
            if status == "not_measured" and owner_id in bindings:
                raise TechnicalInvalid(f"not_measured terminal query unexpectedly has a binding for {owner_id}")
            normalized_statuses[owner_id] = dict(value)
    for owner_id in bindings:
        normalized_statuses.setdefault(
            owner_id,
            {"status": "measured", "not_measured_reason": None},
        )
    terminal_positions = {owner_id: int(item["position"]) for owner_id, item in bindings.items()}
    modules: list[tuple[str, Any, _HookCapture, bool]] = []
    handles: list[Any] = []
    capture_by_name: dict[str, _HookCapture] = {}
    try:
        for layer_index in tuple(int(value) for value in layer_indices):
            if layer_index not in LAYERS:
                raise TechnicalInvalid("observational capture layers must be exactly block 0..27")
            module = _resolve_layer(model, layer_index)
            name = f"block_{layer_index}_output"
            capture = _HookCapture(
                name=name,
                image_positions=image_positions_tuple,
                terminal_positions=terminal_positions,
                device=device,
            )
            handles.append(module.register_forward_hook(lambda _m, _a, out, c=capture: c.capture(out)))
            modules.append((name, module, capture, False))
            capture_by_name[name] = capture
        block_zero = _resolve_layer(model, 0)
        block_zero_capture = _HookCapture(
            name="block_0_input",
            image_positions=image_positions_tuple,
            terminal_positions=terminal_positions,
            device=device,
        )
        def _capture_block_zero_input(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any] | None = None,
            capture: _HookCapture = block_zero_capture,
        ) -> None:
            hidden = args[0] if args else (kwargs or {}).get("hidden_states")
            capture.capture(hidden)

        try:
            handles.append(block_zero.register_forward_pre_hook(_capture_block_zero_input, with_kwargs=True))
        except TypeError:  # Tiny/fake modules on older torch only expose positional hooks.
            handles.append(block_zero.register_forward_pre_hook(lambda _m, args, c=block_zero_capture: c.capture(args[0] if args else None)))
        modules.append(("block_0_input", block_zero, block_zero_capture, False))
        capture_by_name["block_0_input"] = block_zero_capture

        merger_module = merger_module or _resolve_optional_module(
            model,
            ("model.visual.merger", "model.model.visual.merger", "visual.merger", "merger"),
        )
        if merger_module is None:
            raise TechnicalInvalid("merger module is missing; no synthetic image field is permitted")
        merger_capture = _HookCapture(
            name="merger_output",
            image_positions=image_positions_tuple,
            terminal_positions={},
            device=device,
            merger=True,
        )
        handles.append(merger_module.register_forward_hook(lambda _m, _a, out, c=merger_capture: c.capture(out)))
        modules.append(("merger_output", merger_module, merger_capture, True))
        capture_by_name["merger_output"] = merger_capture

        final_norm_module = final_norm_module or _resolve_optional_module(
            model,
            ("model.language_model.norm", "model.model.language_model.norm", "language_model.norm", "norm"),
        )
        if final_norm_module is None:
            raise TechnicalInvalid("final norm module is missing")
        final_norm_capture = _HookCapture(
            name="final_norm",
            image_positions=image_positions_tuple,
            terminal_positions=terminal_positions,
            device=device,
        )
        handles.append(final_norm_module.register_forward_hook(lambda _m, _a, out, c=final_norm_capture: c.capture(out)))
        modules.append(("final_norm", final_norm_module, final_norm_capture, False))
        capture_by_name["final_norm"] = final_norm_capture

        with torch.inference_mode():
            _forward_native(
                model,
                input_ids=input_ids,
                model_inputs=model_inputs,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()

    required_names = ["merger_output", "block_0_input", *(f"block_{index}_output" for index in LAYERS), "final_norm"]
    missing = [name for name in required_names if name not in capture_by_name or capture_by_name[name].count != 1 or capture_by_name[name].state is None]
    if missing:
        raise TechnicalInvalid(f"all P1 hooks must fire exactly once: {missing}")
    for name in required_names:
        capture = capture_by_name[name]
        if set(capture.terminal) != set(bindings) and name != "merger_output":
            raise TechnicalInvalid(f"{name} did not capture every H0 terminal query")
    states = {name: capture_by_name[name].state.detach().clone() for name in required_names}
    terminal_states = {
        name: {owner: state.detach().clone() for owner, state in capture_by_name[name].terminal.items()}
        for name in required_names
        if name != "merger_output"
    }
    hook_counts = {name: capture_by_name[name].count for name in required_names}
    state_receipts = {name: capture_by_name[name].receipt() for name in required_names}
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": checkpoint,
        "image_id": int(image_id),
        "forward_count": 1,
        "hook_counts": hook_counts,
        "cleanup_complete": not handles,
        "device": str(device),
        "input_ids_shape": list(input_ids.shape),
        "input_ids_dtype": str(input_ids.dtype),
        "input_ids_sha256": sha256_token_ids(input_ids),
        "position_ids_shape": list(position_ids.shape),
        "position_ids_dtype": str(position_ids.dtype),
        "mrope_sha256": contract.mrope_sha256,
        "image_span_sha256": image_span_hash,
        "wrapper_identity": contract.recipe_receipt(),
        "terminal_bindings": [bindings[key] for key in sorted(bindings)],
        "terminal_query_statuses": {
            key: normalized_statuses[key] for key in sorted(normalized_statuses)
        },
        "state_receipts": state_receipts,
        "nonfinite_state_count": sum(1 for item in state_receipts.values() if item.get("finite") is not True),
        "passed": bool(not handles and all(count == 1 for count in hook_counts.values())),
    }
    if receipt["nonfinite_state_count"]:
        raise TechnicalInvalid("non-finite state receipt cannot be interpreted")
    return NativePrefillCapture(
        checkpoint=checkpoint,
        image_id=int(image_id),
        states=states,
        terminal_states=terminal_states,
        receipt=receipt,
    )


class NativePrefillRegistry:
    """One-forward-per-(checkpoint,image) guard with collision detection."""

    def __init__(self) -> None:
        self._captures: dict[tuple[str, int], NativePrefillCapture] = {}

    def add(self, capture: NativePrefillCapture) -> None:
        key = (capture.checkpoint, int(capture.image_id))
        prior = self._captures.get(key)
        if prior is not None:
            if prior.receipt.get("input_ids_sha256") != capture.receipt.get("input_ids_sha256"):
                raise CensusContractError(f"prefill collision for {capture.checkpoint}:{capture.image_id}")
            raise CensusContractError(f"duplicate native prefill for {capture.checkpoint}:{capture.image_id}")
        self._captures[key] = capture

    def capture_once(self, checkpoint: str, image_id: int, factory: Any) -> NativePrefillCapture:
        """Invoke ``factory`` exactly once for a new checkpoint/image key."""

        key = (str(checkpoint), int(image_id))
        if key in self._captures:
            raise CensusContractError(f"duplicate native prefill for {checkpoint}:{image_id}")
        if not callable(factory):
            raise CensusContractError("capture_once factory must be callable")
        capture = factory()
        if not isinstance(capture, NativePrefillCapture):
            raise TechnicalInvalid("capture_once factory did not return NativePrefillCapture")
        self.add(capture)
        return capture

    def get(self, checkpoint: str, image_id: int) -> NativePrefillCapture:
        try:
            return self._captures[(str(checkpoint), int(image_id))]
        except KeyError as exc:
            raise CensusContractError(f"missing native prefill for {checkpoint}:{image_id}") from exc

    def keys(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._captures))


@dataclass(frozen=True)
class OwnerSpec:
    owner_id: str
    image_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    image_size: tuple[float, float]
    normalized_bbox: tuple[float, float, float, float] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OwnerSpec":
        owner_id = value.get("owner_id", value.get("gt_owner_id"))
        if not isinstance(owner_id, str) or not owner_id:
            raise CensusContractError("owner spec has no owner_id")
        image_id = _strict_int(value.get("image_id"), "owner image_id", minimum=0)
        class_name = value.get("class_name", value.get("category", value.get("category_name")))
        if not isinstance(class_name, str) or not class_name:
            raise CensusContractError(f"owner {owner_id} has no class_name")
        raw_bbox = value.get("bbox", value.get("pixel_bbox", value.get("bbox_pixel_xyxy")))
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise CensusContractError(f"owner {owner_id} bbox must have four values")
        bbox = tuple(float(item) for item in raw_bbox)
        if not all(math.isfinite(item) for item in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise CensusContractError(f"owner {owner_id} bbox must be finite and positive")
        size = value.get("image_size", (value.get("image_width"), value.get("image_height")))
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise CensusContractError(f"owner {owner_id} image_size is required")
        image_size = (float(size[0]), float(size[1]))
        if not all(math.isfinite(item) and item > 0 for item in image_size):
            raise CensusContractError(f"owner {owner_id} image_size must be positive")
        normalized_raw = value.get("normalized_bbox")
        normalized = None
        if normalized_raw is not None:
            if not isinstance(normalized_raw, (list, tuple)) or len(normalized_raw) != 4:
                raise CensusContractError(f"owner {owner_id} normalized_bbox must have four values")
            normalized = tuple(float(item) for item in normalized_raw)
        return cls(owner_id, image_id, class_name.strip().lower(), bbox, image_size, normalized)


def _grid_shape(grid_thw: Sequence[int | float], merge_size: int) -> tuple[int, int, int]:
    if len(grid_thw) != 3:
        raise CensusContractError("image grid must be [time,height,width]")
    t, height, width = (int(value) for value in grid_thw)
    if min(t, height, width) <= 0 or merge_size <= 0 or height % merge_size or width % merge_size:
        raise CensusContractError("image grid dimensions must be positive and divisible by merge_size")
    return t, height // merge_size, width // merge_size


def _norm_bbox(owner: OwnerSpec) -> tuple[float, float, float, float]:
    if owner.normalized_bbox is not None:
        values = owner.normalized_bbox
    else:
        width, height = owner.image_size
        values = (owner.bbox[0] / width, owner.bbox[1] / height, owner.bbox[2] / width, owner.bbox[3] / height)
    x1, y1, x2, y2 = values
    if not all(math.isfinite(item) for item in values) or x2 <= x1 or y2 <= y1:
        raise CensusContractError(f"owner {owner.owner_id} normalized bbox is invalid")
    return max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)), max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))


def bbox_cell_overlaps(
    owner: OwnerSpec,
    *,
    image_grid_thw: Sequence[int | float],
    merge_size: int,
) -> tuple[dict[str, Any], ...]:
    """Return fractional bbox-to-merger-cell overlap weights."""

    time, rows, columns = _grid_shape(image_grid_thw, int(merge_size))
    x1, y1, x2, y2 = _norm_bbox(owner)
    result: list[dict[str, Any]] = []
    cell_area = 1.0 / float(rows * columns)
    for t in range(time):
        for row in range(rows):
            cy1, cy2 = row / rows, (row + 1) / rows
            overlap_y = max(0.0, min(y2, cy2) - max(y1, cy1))
            if overlap_y <= 0:
                continue
            for col in range(columns):
                cx1, cx2 = col / columns, (col + 1) / columns
                overlap_x = max(0.0, min(x2, cx2) - max(x1, cx1))
                if overlap_x <= 0:
                    continue
                visual_index = t * rows * columns + row * columns + col
                result.append({
                    "visual_index": visual_index,
                    "time": t,
                    "row": row,
                    "column": col,
                    "overlap_fraction": float((overlap_x * overlap_y) / cell_area),
                })
    return tuple(result)


def build_owner_regions(
    owners: Sequence[OwnerSpec | Mapping[str, Any]],
    *,
    image_grid_thw: Sequence[int | float],
    merge_size: int,
) -> dict[str, dict[str, Any]]:
    """Split each owner support into exclusive and shared-only cells."""

    normalized = [item if isinstance(item, OwnerSpec) else OwnerSpec.from_mapping(item) for item in owners]
    support = {
        owner.owner_id: {int(cell["visual_index"]): float(cell["overlap_fraction"]) for cell in bbox_cell_overlaps(owner, image_grid_thw=image_grid_thw, merge_size=merge_size)}
        for owner in normalized
    }
    regions: dict[str, dict[str, Any]] = {}
    for owner in normalized:
        own = support[owner.owner_id]
        exclusive = {index: weight for index, weight in own.items() if not any(index in other for other_id, other in support.items() if other_id != owner.owner_id)}
        shared = {index: weight for index, weight in own.items() if index not in exclusive}
        reason = None if exclusive else "exclusive_support_vanished"
        regions[owner.owner_id] = {
            "support": dict(sorted(own.items())),
            "exclusive": dict(sorted(exclusive.items())),
            "shared_only": dict(sorted(shared.items())),
            "exclusive_available": bool(exclusive),
            "shared_available": bool(shared),
            "not_measured_reason": reason,
        }
    return regions


def _normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1:
        vector = vector.reshape(-1)
    if not bool(torch.isfinite(vector).all().item()):
        raise TechnicalInvalid("cosine vector contains non-finite values")
    norm = vector.float().norm(p=2)
    if float(norm.item()) <= 1.0e-12:
        raise TechnicalInvalid("cannot L2-normalize a zero vector")
    return vector.float() / norm


def weighted_owner_prototype(
    state: torch.Tensor,
    cells: Mapping[int, float] | Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Fractional weighted mean + L2 normalization, or explicit unavailable."""

    if state.ndim != 2:
        raise TechnicalInvalid("image field must have shape [cells,hidden]")
    if not bool(torch.isfinite(state.detach()).all().item()):
        raise TechnicalInvalid("image field contains non-finite values")
    if isinstance(cells, Mapping):
        weights = {int(index): float(weight) for index, weight in cells.items()}
    else:
        weights = {int(item["visual_index"]): float(item["overlap_fraction"]) for item in cells}
    if not weights:
        return None, {"available": False, "denominator": 0, "not_measured_reason": "exclusive_support_vanished"}
    if any(index < 0 or index >= state.shape[0] for index in weights):
        raise TechnicalInvalid("owner overlap cell is outside merger state")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise TechnicalInvalid("owner overlap weights must be finite and positive")
    indices = torch.tensor(sorted(weights), dtype=torch.long, device=state.device)
    weight_tensor = torch.tensor([weights[index] for index in sorted(weights)], dtype=torch.float32, device=state.device)
    selected = state.index_select(0, indices).float()
    pooled = (selected * weight_tensor.unsqueeze(1)).sum(dim=0) / weight_tensor.sum()
    normalized = _normalize_vector(pooled).detach()
    return normalized, {
        "available": True,
        "denominator": len(weights),
        "weight_sum": float(weight_tensor.sum().item()),
        "cell_indices": sorted(weights),
        "prototype_shape": list(normalized.shape),
        "prototype_dtype": str(normalized.dtype),
        "prototype_sha256": sha256_tensor(normalized),
    }


def _cosine(query: torch.Tensor, prototype: torch.Tensor) -> float:
    left, right = _normalize_vector(query), _normalize_vector(prototype)
    return float(torch.dot(left, right).item())


def _not_measured(reason: str, denominator: int = 0) -> dict[str, Any]:
    return {"available": False, "value": None, "denominator": int(denominator), "not_measured_reason": reason}


def _metric(value: Any, denominator: int = 1) -> dict[str, Any]:
    return {"available": True, "value": value, "denominator": int(denominator), "not_measured_reason": None}


def _retrieval_metrics(query: torch.Tensor, own: torch.Tensor | None, negatives: Sequence[tuple[str, torch.Tensor]], same_class: Sequence[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    if own is None:
        unavailable = _not_measured("exclusive_support_vanished")
        return {"own_cosine": unavailable, "hardest_same_image_negative": unavailable, "margin": unavailable, "retrieval_at_1": unavailable, "same_class_counterpart": unavailable, "own_cosine_raw": None, "hardest_negative_cosine_raw": None, "margin_raw": None, "retrieval_at_1_raw": None, "same_class_hardest_negative_cosine_raw": None, "same_class_retrieval_at_1_raw": None}
    own_value = _metric(_cosine(query, own))
    if not negatives:
        hardest = _not_measured("no_same_image_negative")
        margin = _not_measured("no_same_image_negative")
        retrieval = _not_measured("no_same_image_negative")
    else:
        values = [(owner_id, _cosine(query, vector)) for owner_id, vector in negatives]
        hardest_id, hardest_value = max(values, key=lambda item: (item[1], item[0]))
        hardest = _metric(hardest_value, len(values))
        margin = _metric(float(own_value["value"]) - hardest_value, len(values))
        retrieval = _metric(bool(float(own_value["value"]) > hardest_value), len(values))
        hardest["owner_id"] = hardest_id
    if not same_class:
        same = _not_measured("no_same_class_counterpart")
    else:
        values = [(owner_id, _cosine(query, vector)) for owner_id, vector in same_class]
        same_id, same_value = max(values, key=lambda item: (item[1], item[0]))
        same = _metric(same_value, len(values))
        same.update({"owner_id": same_id, "retrieval_at_1": bool(float(own_value["value"]) > same_value)})
    return {
        "own_cosine": own_value,
        "hardest_same_image_negative": hardest,
        "margin": margin,
        "retrieval_at_1": retrieval,
        "same_class_counterpart": same,
        "own_cosine_raw": own_value["value"],
        "hardest_negative_cosine_raw": hardest["value"],
        "margin_raw": margin["value"],
        "retrieval_at_1_raw": retrieval["value"],
        "same_class_hardest_negative_cosine_raw": same["value"],
        "same_class_retrieval_at_1_raw": same.get("retrieval_at_1"),
    }


def _geometry_vector(region: Mapping[str, float], cell_count: int) -> torch.Tensor:
    vector = torch.zeros(cell_count, dtype=torch.float32)
    for index, weight in region.items():
        vector[int(index)] = float(weight)
    if float(vector.norm().item()) <= 1.0e-12:
        raise TechnicalInvalid("empty geometry control vector")
    return vector / vector.norm(p=2)


def _scalar_similarity(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1.0e-12)
    return float(1.0 - min(1.0, abs(left - right) / denominator))


def _deterministic_derangement(owner_ids: Sequence[str], *, seed: int) -> dict[str, str] | None:
    ordered = sorted(str(owner_id) for owner_id in owner_ids)
    if len(ordered) < 2:
        return None
    offset = random.Random(int(seed)).randrange(1, len(ordered))
    mapping = {
        owner_id: ordered[(index + offset) % len(ordered)]
        for index, owner_id in enumerate(ordered)
    }
    if set(mapping) != set(mapping.values()) or any(owner_id == donor for owner_id, donor in mapping.items()):
        raise TechnicalInvalid("deterministic shuffled-owner control is not a derangement")
    return mapping


def _control_metrics(
    *,
    owner: OwnerSpec,
    owners: Sequence[OwnerSpec],
    regions: Mapping[str, Mapping[str, Any]],
    query: torch.Tensor,
    prototypes: Mapping[str, torch.Tensor | None],
    image_cell_count: int,
    next_image_state: torch.Tensor | None,
    next_image_grid_thw: Sequence[int | float] | None,
    next_image_merge_size: int | None,
    next_image_binding: Mapping[str, Any] | None,
    merge_size: int,
    layer_name: str,
    seed: int,
    missing_query_reason: str = "missing_h0_terminal_query_state",
) -> dict[str, Any]:
    def _cyclic_binding() -> dict[str, Any] | None:
        """Validate and reduce the partner receipt to scalar provenance only."""

        if next_image_state is None or next_image_grid_thw is None:
            return None
        if not isinstance(next_image_binding, Mapping):
            raise TechnicalInvalid("measured cyclic control lacks a partner binding receipt")
        partner_image_id = _strict_int(next_image_binding.get("partner_image_id"), "cyclic partner image_id", minimum=0)
        if partner_image_id == owner.image_id:
            raise TechnicalInvalid("cyclic control partner must be a different image")
        grid = [int(value) for value in next_image_grid_thw]
        if next_image_binding.get("partner_image_grid_thw") != grid:
            raise TechnicalInvalid("cyclic partner grid differs from its binding receipt")
        if next_image_binding.get("partner_image_grid_sha256") != sha256_json(grid):
            raise TechnicalInvalid("cyclic partner grid hash is invalid")
        for key in ("partner_prefix_token_ids_sha256", "partner_mrope_sha256"):
            if not _is_sha256(next_image_binding.get(key)):
                raise TechnicalInvalid(f"cyclic partner {key} is missing or invalid")
        if not isinstance(next_image_state, torch.Tensor) or next_image_state.ndim != 2:
            raise TechnicalInvalid("cyclic partner state must have shape [cells,hidden]")
        if next_image_state.shape[1] != query.numel():
            raise TechnicalInvalid("cyclic partner state hidden width differs from terminal query")
        partner_merge_size = _strict_int(
            next_image_merge_size if next_image_merge_size is not None else merge_size,
            "cyclic partner merge_size",
            minimum=1,
        )
        expected_cells = math.prod(_grid_shape(grid, partner_merge_size))
        if int(next_image_state.shape[0]) != expected_cells:
            raise TechnicalInvalid("cyclic partner state cell count differs from its image grid")
        state_sha256 = next_image_binding.get("partner_state_sha256")
        if not _is_sha256(state_sha256):
            raise TechnicalInvalid("cyclic partner state hash is missing or invalid")
        return {
            "partner_image_id": partner_image_id,
            "partner_image_grid_thw": grid,
            "partner_image_grid_sha256": str(next_image_binding["partner_image_grid_sha256"]),
            "partner_merge_size": partner_merge_size,
            "partner_prefix_token_ids_sha256": str(next_image_binding["partner_prefix_token_ids_sha256"]),
            "partner_mrope_sha256": str(next_image_binding["partner_mrope_sha256"]),
            "partner_state_sha256": str(state_sha256),
        }

    partner_binding = _cyclic_binding()
    image_owners = sorted(
        (item for item in owners if item.image_id == owner.image_id),
        key=lambda item: item.owner_id,
    )
    permutation = _deterministic_derangement(
        [item.owner_id for item in image_owners],
        seed=seed,
    )
    if not isinstance(query, torch.Tensor) or query.numel() == 0 or float(query.detach().float().norm().item()) <= 1.0e-12:
        unavailable = {key: _not_measured(missing_query_reason) for key in ("own_cosine", "hardest_negative", "margin", "retrieval_at_1")}
        cyclic_unavailable = dict(unavailable)
        if partner_binding is not None:
            cyclic_unavailable["partner_binding"] = partner_binding
        controls = {
            "coordinate_only": dict(unavailable),
            "area_density_only": dict(unavailable),
            "shuffled_owner": {
                **unavailable,
                "shuffle_seed": int(seed),
                "permutation": permutation or {},
                "derangement": permutation is not None,
            },
            "cyclic_next_image_same_normalized_geometry": cyclic_unavailable,
            "layer": layer_name,
            "shuffle_seed": int(seed),
            "target_blind": True,
        }
        for control in (controls["coordinate_only"], controls["area_density_only"], controls["shuffled_owner"], cyclic_unavailable):
            control["target_blind"] = True
        controls["coordinate_only"]["uses_layer_state"] = False
        controls["area_density_only"]["uses_layer_state"] = False
        controls["shuffled_owner"]["uses_layer_state"] = True
        cyclic_unavailable["uses_layer_state"] = True
        return controls
    geometry_values = {item.owner_id: _geometry_vector(regions[item.owner_id]["exclusive"], image_cell_count) if regions[item.owner_id]["exclusive"] else None for item in image_owners}
    own_geo = geometry_values.get(owner.owner_id)
    coordinate_negatives = [(item.owner_id, vector) for item in image_owners if item.owner_id != owner.owner_id and (vector := geometry_values.get(item.owner_id)) is not None]
    if own_geo is None:
        coordinate = {key: _not_measured("exclusive_support_vanished") for key in ("own_cosine", "hardest_negative", "margin", "retrieval_at_1")}
    else:
        own_score = _metric(float(torch.dot(own_geo, own_geo).item()))
        if coordinate_negatives:
            values = [(owner_id, float(torch.dot(own_geo, vector).item())) for owner_id, vector in coordinate_negatives]
            negative_id, negative = max(values, key=lambda item: (item[1], item[0]))
            coordinate = {"own_cosine": own_score, "hardest_negative": _metric(negative, len(values)), "margin": _metric(float(own_score["value"]) - negative, len(values)), "retrieval_at_1": _metric(float(own_score["value"]) > negative, len(values)), "hardest_negative_owner_id": negative_id}
        else:
            coordinate = {"own_cosine": own_score, "hardest_negative": _not_measured("no_same_image_negative"), "margin": _not_measured("no_same_image_negative"), "retrieval_at_1": _not_measured("no_same_image_negative")}
    own_density = float(sum(float(value) for value in regions[owner.owner_id]["support"].values()))
    density_values = {item.owner_id: float(sum(float(value) for value in regions[item.owner_id]["support"].values())) for item in image_owners}
    own_area_score = _metric(_scalar_similarity(own_density, own_density))
    area_negatives = [(item.owner_id, _scalar_similarity(own_density, density_values[item.owner_id])) for item in image_owners if item.owner_id != owner.owner_id]
    if area_negatives:
        area_id, area_negative = max(area_negatives, key=lambda item: (item[1], item[0]))
        area_control = {"own_cosine": own_area_score, "hardest_negative": _metric(area_negative, len(area_negatives)), "margin": _metric(float(own_area_score["value"]) - area_negative, len(area_negatives)), "retrieval_at_1": _metric(float(own_area_score["value"]) > area_negative, len(area_negatives)), "hardest_negative_owner_id": area_id}
    else:
        area_control = {"own_cosine": own_area_score, "hardest_negative": _not_measured("no_same_image_negative"), "margin": _not_measured("no_same_image_negative"), "retrieval_at_1": _not_measured("no_same_image_negative")}
    if permutation is None:
        shuffled = {
            key: _not_measured("derangement_requires_at_least_two_owners")
            for key in (
                "own_cosine",
                "hardest_same_image_negative",
                "margin",
                "retrieval_at_1",
                "same_class_counterpart",
            )
        }
        shuffled.update({"permutation": {}, "derangement": False, "shuffle_seed": int(seed)})
    else:
        shuffled_prototypes = {
            owner_id: prototypes.get(donor_id)
            for owner_id, donor_id in permutation.items()
        }
        shuffled_own = shuffled_prototypes.get(owner.owner_id)
        shuffled_negatives = [
            (item.owner_id, value)
            for item in image_owners
            if item.owner_id != owner.owner_id
            and (value := shuffled_prototypes.get(item.owner_id)) is not None
        ]
        shuffled = _retrieval_metrics(query, shuffled_own, shuffled_negatives, [])
        shuffled.update(
            {
                "shuffle_seed": int(seed),
                "permutation": permutation,
                "derangement": True,
            }
        )
    cyclic: dict[str, Any]
    if next_image_state is None or next_image_grid_thw is None:
        cyclic = {key: _not_measured("next_image_state_not_provided") for key in ("own_cosine", "hardest_negative", "margin", "retrieval_at_1")}
    else:
        partner_merge_size = int(next_image_merge_size if next_image_merge_size is not None else merge_size)
        cells = bbox_cell_overlaps(owner, image_grid_thw=next_image_grid_thw, merge_size=partner_merge_size)
        next_proto, _ = weighted_owner_prototype(next_image_state, cells)
        if next_proto is None:
            cyclic = {key: _not_measured("cyclic_next_image_geometry_has_no_support") for key in ("own_cosine", "hardest_negative", "margin", "retrieval_at_1")}
        else:
            cyclic = {"own_cosine": _metric(_cosine(query, next_proto)), "hardest_negative": _not_measured("single_cyclic_geometry_control"), "margin": _not_measured("single_cyclic_geometry_control"), "retrieval_at_1": _not_measured("single_cyclic_geometry_control")}
        if partner_binding is None:  # Defensive: the branch above must construct it.
            raise TechnicalInvalid("measured cyclic control lacks a partner binding receipt")
        cyclic["partner_binding"] = partner_binding
    for control in (coordinate, area_control, shuffled, cyclic):
        control["target_blind"] = True
    coordinate["uses_layer_state"] = False
    area_control["uses_layer_state"] = False
    shuffled["uses_layer_state"] = True
    cyclic["uses_layer_state"] = True
    return {"coordinate_only": coordinate, "area_density_only": area_control, "shuffled_owner": shuffled, "cyclic_next_image_same_normalized_geometry": cyclic, "layer": layer_name, "shuffle_seed": int(seed), "target_blind": True}


def _contrast_receipt(
    *,
    query: torch.Tensor | None,
    positive: tuple[str, torch.Tensor] | None,
    negative: tuple[str, torch.Tensor] | None,
    missing_query_reason: str,
    missing_positive_reason: str,
    missing_negative_reason: str,
) -> dict[str, Any]:
    def unavailable(reason: str) -> dict[str, Any]:
        return {
            **_not_measured(reason),
            "uses_layer_state": True,
            "probe_free": True,
        }

    if query is None:
        return unavailable(missing_query_reason)
    if positive is None:
        return unavailable(missing_positive_reason)
    if negative is None:
        return unavailable(missing_negative_reason)
    positive_id, positive_state = positive
    negative_id, negative_state = negative
    positive_cosine = _cosine(query, positive_state)
    negative_cosine = _cosine(query, negative_state)
    return {
        **_metric(positive_cosine - negative_cosine, 1),
        "positive_id": positive_id,
        "negative_id": negative_id,
        "positive_cosine": positive_cosine,
        "negative_cosine": negative_cosine,
        "uses_layer_state": True,
        "probe_free": True,
    }


def _best_state_candidate(
    query: torch.Tensor,
    candidates: Sequence[tuple[str, torch.Tensor]],
) -> tuple[str, torch.Tensor] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: (_cosine(query, item[1]), item[0]))


def _nearest_density_candidate(
    candidates: Sequence[tuple[str, torch.Tensor, float]],
    *,
    target_density: float,
) -> tuple[str, torch.Tensor, float] | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (abs(float(item[2]) - float(target_density)), item[0]),
    )


def _state_readouts(
    *,
    owner: OwnerSpec,
    owners: Sequence[OwnerSpec],
    query: torch.Tensor | None,
    prototypes: Mapping[str, torch.Tensor | None],
    regions: Mapping[str, Mapping[str, Any]],
    background_prototype: torch.Tensor | None,
    missing_query_reason: str,
    cell_count: int,
) -> dict[str, Any]:
    own = prototypes.get(owner.owner_id)
    foreground = _contrast_receipt(
        query=query,
        positive=(owner.owner_id, own) if own is not None else None,
        negative=("background", background_prototype) if background_prototype is not None else None,
        missing_query_reason=missing_query_reason,
        missing_positive_reason="exclusive_support_vanished",
        missing_negative_reason="no_background_cells",
    )
    if query is None:
        class_contrast = _contrast_receipt(
            query=None,
            positive=None,
            negative=None,
            missing_query_reason=missing_query_reason,
            missing_positive_reason="no_same_class_counterpart",
            missing_negative_reason="no_cross_class_counterpart",
        )
        density_conditioned = dict(class_contrast)
    else:
        same_class = [
            (other.owner_id, value)
            for other in owners
            if other.owner_id != owner.owner_id
            and other.class_name == owner.class_name
            and (value := prototypes.get(other.owner_id)) is not None
        ]
        cross_class = [
            (other.owner_id, value)
            for other in owners
            if other.class_name != owner.class_name
            and (value := prototypes.get(other.owner_id)) is not None
        ]
        class_contrast = _contrast_receipt(
            query=query,
            positive=_best_state_candidate(query, same_class),
            negative=_best_state_candidate(query, cross_class),
            missing_query_reason=missing_query_reason,
            missing_positive_reason="no_same_class_counterpart",
            missing_negative_reason="no_cross_class_counterpart",
        )
        density_by_owner = {
            item.owner_id: float(sum(float(value) for value in regions[item.owner_id]["support"].values()))
            / max(cell_count, 1)
            for item in owners
        }
        same_density_candidates = [
            (owner_id, state, density_by_owner[owner_id])
            for owner_id, state in same_class
        ]
        cross_density_candidates = [
            (owner_id, state, density_by_owner[owner_id])
            for owner_id, state in cross_class
        ]
        own_density = density_by_owner[owner.owner_id]
        same_density = _nearest_density_candidate(
            same_density_candidates,
            target_density=own_density,
        )
        cross_density = _nearest_density_candidate(
            cross_density_candidates,
            target_density=own_density,
        )
        density_conditioned = _contrast_receipt(
            query=query,
            positive=(same_density[0], same_density[1]) if same_density is not None else None,
            negative=(cross_density[0], cross_density[1]) if cross_density is not None else None,
            missing_query_reason=missing_query_reason,
            missing_positive_reason="no_same_class_counterpart",
            missing_negative_reason="no_cross_class_counterpart",
        )
        if density_conditioned.get("available"):
            density_conditioned.update(
                {
                    "target_fractional_occupancy": own_density,
                    "positive_fractional_occupancy": same_density[2],
                    "negative_fractional_occupancy": cross_density[2],
                    "positive_density_gap": abs(same_density[2] - own_density),
                    "negative_density_gap": abs(cross_density[2] - own_density),
                    "conditioning": "nearest_fractional_occupancy_per_class_role",
                }
            )
    return {
        "foreground_state_contrast": foreground,
        "class_state_contrast": class_contrast,
        "density_conditioned_class_state_contrast": density_conditioned,
    }


def compute_observational_census(
    capture: NativePrefillCapture,
    owners: Sequence[OwnerSpec | Mapping[str, Any]],
    *,
    image_grid_thw: Sequence[int | float],
    merge_size: int,
    next_image_state_by_layer: Mapping[str, torch.Tensor] | None = None,
    next_image_grid_thw: Sequence[int | float] | None = None,
    next_image_merge_size: int | None = None,
    next_image_binding_by_layer: Mapping[str, Mapping[str, Any]] | None = None,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> dict[str, Any]:
    """Compute descriptive P1 readouts without retaining tensor payloads."""

    normalized = [item if isinstance(item, OwnerSpec) else OwnerSpec.from_mapping(item) for item in owners]
    if not normalized:
        raise CensusContractError("observational census owner cohort must be non-empty")
    if any(item.image_id != capture.image_id for item in normalized):
        raise CensusContractError("one native prefill must serve exactly one image cohort")
    regions = build_owner_regions(normalized, image_grid_thw=image_grid_thw, merge_size=merge_size)
    cell_count = int(capture.states["merger_output"].shape[0])
    occupied_cell_indices = {
        int(index)
        for region in regions.values()
        for index in region["support"]
    }
    background_cells = {
        index: 1.0 for index in range(cell_count) if index not in occupied_cell_indices
    }
    prototypes_by_layer: dict[str, dict[str, torch.Tensor | None]] = {}
    prototype_receipts: dict[str, dict[str, dict[str, Any]]] = {}
    background_by_layer: dict[str, torch.Tensor | None] = {}
    background_receipts: dict[str, dict[str, Any]] = {}
    for layer_name, state in capture.states.items():
        layer_prototypes: dict[str, torch.Tensor | None] = {}
        layer_receipts: dict[str, dict[str, Any]] = {}
        for owner in normalized:
            prototype, receipt = weighted_owner_prototype(state, regions[owner.owner_id]["exclusive"])
            layer_prototypes[owner.owner_id] = prototype
            layer_receipts[owner.owner_id] = receipt
        prototypes_by_layer[layer_name] = layer_prototypes
        prototype_receipts[layer_name] = layer_receipts
        if background_cells:
            background, background_receipt = weighted_owner_prototype(state, background_cells)
        else:
            background = None
            background_receipt = {
                "available": False,
                "denominator": 0,
                "not_measured_reason": "no_background_cells",
            }
        background_by_layer[layer_name] = background
        background_receipts[layer_name] = background_receipt
    rows: list[dict[str, Any]] = []
    for layer_name in capture.states:
        state = capture.states[layer_name]
        prototypes = prototypes_by_layer[layer_name]
        for owner in normalized:
            query_state = capture.terminal_states.get(layer_name, {}).get(owner.owner_id)
            status_receipts = capture.receipt.get("terminal_query_statuses")
            owner_status = status_receipts.get(owner.owner_id) if isinstance(status_receipts, Mapping) else None
            missing_query_reason = (
                str(owner_status["not_measured_reason"])
                if isinstance(owner_status, Mapping)
                and owner_status.get("status") == "not_measured"
                and isinstance(owner_status.get("not_measured_reason"), str)
                else "missing_h0_terminal_query_state"
            )
            region = regions[owner.owner_id]
            own = prototypes.get(owner.owner_id)
            if query_state is None:
                retrieval = {key: _not_measured(missing_query_reason) for key in ("own_cosine", "hardest_same_image_negative", "margin", "retrieval_at_1", "same_class_counterpart")}
            else:
                negatives = [(other.owner_id, prototypes[other.owner_id]) for other in normalized if other.image_id == owner.image_id and other.owner_id != owner.owner_id and prototypes[other.owner_id] is not None]
                same_class = [(other.owner_id, prototypes[other.owner_id]) for other in normalized if other.image_id == owner.image_id and other.owner_id != owner.owner_id and other.class_name == owner.class_name and prototypes[other.owner_id] is not None]
                retrieval = _retrieval_metrics(query_state, own, negatives, same_class)
            support = region["support"]
            occupied = len(support)
            fractional = float(sum(float(value) for value in support.values()))
            foreground = _metric(bool(occupied > 0))
            foreground_fraction = _metric(float(occupied / max(cell_count, 1)), cell_count) if occupied else _not_measured("no_foreground_overlap")
            fractional_occupancy = _metric(fractional / max(cell_count, 1), cell_count)
            density = _metric(fractional / occupied, occupied) if occupied else _not_measured("no_foreground_overlap")
            log_density = _metric(math.log1p(float(density["value"])), density["denominator"]) if density["available"] else _not_measured("no_foreground_overlap")
            class_peers = [other for other in normalized if other.class_name == owner.class_name and other.owner_id != owner.owner_id and regions[other.owner_id]["support"]]
            if class_peers:
                class_density_values = [sum(float(value) for value in regions[item.owner_id]["support"].values()) / len(regions[item.owner_id]["support"]) for item in class_peers]
                class_readout = {"available": True, "class_name": owner.class_name, "denominator": len(class_density_values), "mean_density": float(sum(class_density_values) / len(class_density_values)), "owner_density": density["value"]}
            else:
                class_readout = {"available": False, "class_name": owner.class_name, "denominator": 0, "mean_density": None, "owner_density": density["value"], "not_measured_reason": "no_same_class_counterpart"}
            state_readouts = _state_readouts(
                owner=owner,
                owners=normalized,
                query=query_state,
                prototypes=prototypes,
                regions=regions,
                background_prototype=background_by_layer[layer_name],
                missing_query_reason=missing_query_reason,
                cell_count=cell_count,
            )
            controls = _control_metrics(
                owner=owner,
                owners=normalized,
                regions=regions,
                query=query_state if query_state is not None else torch.zeros(state.shape[1], device=state.device),
                prototypes=prototypes,
                image_cell_count=cell_count,
                next_image_state=(next_image_state_by_layer or {}).get(layer_name),
                next_image_grid_thw=next_image_grid_thw,
                next_image_merge_size=next_image_merge_size,
                next_image_binding=(next_image_binding_by_layer or {}).get(layer_name),
                merge_size=merge_size,
                layer_name=layer_name,
                seed=shuffle_seed,
                missing_query_reason=missing_query_reason,
            )
            rows.append({
                "owner_id": owner.owner_id,
                "image_id": owner.image_id,
                "class_name": owner.class_name,
                "layer": layer_name,
                "region": {"exclusive_cell_indices": sorted(region["exclusive"]), "shared_only_cell_indices": sorted(region["shared_only"]), "exclusive_available": region["exclusive_available"], "shared_available": region["shared_available"], "not_measured_reason": region["not_measured_reason"]},
                "retrieval": retrieval,
                "state_readouts": state_readouts,
                "geometry_labels": {
                    "foreground": foreground,
                    "foreground_fraction": foreground_fraction,
                    "fractional_occupancy": fractional_occupancy,
                    "density": density,
                    "log1p_density": log_density,
                    "same_class_density_baseline": class_readout,
                    "uses_layer_state": False,
                    "interpretation": "labels_and_geometry_baselines_only",
                },
                "controls": controls,
            })
    return {
        "schema_version": P1_CENSUS_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": capture.checkpoint,
        "image_id": capture.image_id,
        "layer_names": list(capture.states),
        "owner_count": len(normalized),
        "cell_count": cell_count,
        "target_blind_controls": True,
        "no_efficacy_thresholds": True,
        "prototype_receipts": prototype_receipts,
        "background_prototype_receipts": background_receipts,
        "regions": regions,
        "rows": rows,
        "capture_receipt": capture.receipt,
    }


def _immutable_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise CensusContractError(f"fail-collision: existing artifact differs: {path}")
        return
    path.write_bytes(content)


def _identity_hashes(identity: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    required = ("config", "panel", "cohort", "h0", "runtime", "prefix", "wrapper", "mrope")
    normalized: dict[str, Any] = {}
    for key in required:
        value = identity.get(key)
        if not isinstance(value, Mapping):
            raise CensusContractError(f"{label}.{key} identity is missing")
        digest = value.get("sha256", value.get("hash"))
        if not _is_sha256(digest):
            raise CensusContractError(f"{label}.{key} identity must carry a SHA-256")
        normalized[key] = {**dict(value), "sha256": digest}
    return normalized


def _validate_runtime_attestation(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CensusContractError(f"{label} runtime attestation is missing")
    normalized = dict(value)
    if normalized.get("status") != "validated" or normalized.get("passed") is not True:
        raise CensusContractError(f"{label} runtime attestation must be validated and passed")
    physical_id = normalized.get("physical_device_id")
    if not isinstance(physical_id, str) or re.fullmatch(r"[0-9]+", physical_id) is None:
        raise CensusContractError(f"{label} runtime attestation physical_device_id is invalid")
    physical_uuid = normalized.get("physical_device_uuid")
    if not isinstance(physical_uuid, str) or re.fullmatch(r"GPU-[0-9A-Fa-f-]+", physical_uuid) is None:
        raise CensusContractError(f"{label} runtime attestation physical_device_uuid is invalid")
    pid = normalized.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise CensusContractError(f"{label} runtime attestation pid is invalid")
    return normalized


def build_shard_manifest(
    *,
    checkpoint: Literal["S", "A"],
    shard_index: int,
    shard_count: int,
    image_ids: Sequence[int | str],
    identity: Mapping[str, Any],
    runtime_attestation: Mapping[str, Any],
    contract: WrapperContract,
) -> dict[str, Any]:
    """Build an immutable manifest with explicit checkpoint/image partition."""

    if checkpoint not in CHECKPOINTS:
        raise CensusContractError("manifest checkpoint must be S or A")
    expected = partition_image_ids(image_ids, shard_index=shard_index, shard_count=shard_count)
    contract.validate()
    if contract.checkpoint != checkpoint:
        raise CensusContractError("manifest checkpoint differs from wrapper contract")
    normalized_attestation = _validate_runtime_attestation(runtime_attestation, label="manifest")
    normalized_identity = _identity_hashes(identity, label="identity")
    wrapper_identity = contract.recipe_receipt()
    if normalized_identity["wrapper"]["sha256"] != sha256_json(wrapper_identity):
        raise CensusContractError("manifest wrapper identity differs from wrapper recipe")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": checkpoint,
        "shard": {"index": int(shard_index), "count": int(shard_count), "selector": f"{checkpoint}:{int(shard_index)}/{int(shard_count)}"},
        "image_ids": list(expected),
        "identity": normalized_identity,
        "runtime_attestation": normalized_attestation,
        "wrapper_identity": wrapper_identity,
        "partition_sha256": sha256_json(list(expected)),
        "immutable": True,
        "technical_invalid_policy": "quarantine_and_exclude_from_cpu_merge",
        "collision_policy": "fail_closed",
    }


def validate_shard_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or manifest.get("unit_id") != UNIT_ID:
        raise CensusContractError("unsupported observational census shard manifest")
    checkpoint = manifest.get("checkpoint")
    if checkpoint not in CHECKPOINTS:
        raise CensusContractError("manifest checkpoint is invalid")
    shard = manifest.get("shard")
    if not isinstance(shard, Mapping):
        raise CensusContractError("manifest shard selector is missing")
    index, count = _strict_int(shard.get("index"), "manifest shard index", minimum=0), _strict_int(shard.get("count"), "manifest shard count", minimum=1)
    if index >= count or shard.get("selector") != f"{checkpoint}:{index}/{count}":
        raise CensusContractError("manifest shard selector is inconsistent")
    image_ids = manifest.get("image_ids")
    if not isinstance(image_ids, list) or image_ids != sorted(set(image_ids)) or any(not isinstance(item, int) for item in image_ids):
        raise CensusContractError("manifest image_ids must be sorted unique integers")
    if manifest.get("partition_sha256") != sha256_json(image_ids):
        raise CensusContractError("manifest partition hash mismatch")
    identity = _identity_hashes(manifest.get("identity", {}), label="manifest.identity")
    _validate_runtime_attestation(manifest.get("runtime_attestation"), label="manifest")
    wrapper_identity = manifest.get("wrapper_identity")
    if not isinstance(wrapper_identity, Mapping) or identity["wrapper"]["sha256"] != sha256_json(wrapper_identity):
        raise CensusContractError("manifest wrapper recipe hash mismatch")
    if manifest.get("immutable") is not True:
        raise CensusContractError("manifest must attest immutable=true")
    return dict(manifest)


def _validate_valid_census_row(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = manifest["checkpoint"]
    image_id = row.get("image_id")
    if row.get("schema_version") != P1_CENSUS_SCHEMA_VERSION or row.get("status") != "valid":
        raise CensusContractError("valid shard row has an invalid schema/status")
    if row.get("checkpoint") != checkpoint or not isinstance(image_id, int) or image_id not in manifest["image_ids"]:
        raise CensusContractError("valid shard row checkpoint/image identity is invalid")
    if row.get("runtime_attestation") != manifest.get("runtime_attestation"):
        raise CensusContractError("valid shard row runtime attestation differs from manifest")
    row_identity = _identity_hashes(row.get("identity", {}), label=f"row[{image_id}].identity")
    manifest_identity = manifest["identity"]
    for key in ("config", "panel", "cohort", "h0", "runtime", "wrapper"):
        if row_identity[key] != manifest_identity[key]:
            raise CensusContractError(f"valid shard row common identity differs: {key}")
    capture = row.get("capture_receipt")
    if not isinstance(capture, Mapping):
        raise CensusContractError("valid shard row capture receipt is missing")
    if (
        capture.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or capture.get("unit_id") != UNIT_ID
        or capture.get("checkpoint") != checkpoint
        or capture.get("image_id") != image_id
        or capture.get("forward_count") != 1
        or capture.get("cleanup_complete") is not True
        or capture.get("passed") is not True
        or capture.get("nonfinite_state_count") != 0
    ):
        raise CensusContractError("valid shard row capture receipt is not mechanically valid")
    hook_counts = capture.get("hook_counts")
    if not isinstance(hook_counts, Mapping) or set(hook_counts) != set(LAYER_NAMES) or any(value != 1 for value in hook_counts.values()):
        raise CensusContractError("valid shard row capture hook coverage is incomplete")
    if capture.get("input_ids_sha256") != row_identity["prefix"]["sha256"]:
        raise CensusContractError("valid shard row prefix identity differs from capture")
    if capture.get("mrope_sha256") != row_identity["mrope"]["sha256"]:
        raise CensusContractError("valid shard row MRoPE identity differs from capture")
    p1 = row.get("p1_census")
    if not isinstance(p1, Mapping):
        raise CensusContractError("valid shard row P1 census is missing")
    if (
        p1.get("schema_version") != P1_CENSUS_SCHEMA_VERSION
        or p1.get("unit_id") != UNIT_ID
        or p1.get("checkpoint") != checkpoint
        or p1.get("image_id") != image_id
        or p1.get("layer_names") != list(LAYER_NAMES)
        or p1.get("capture_receipt") != capture
    ):
        raise CensusContractError("valid shard row P1 census identity/capture binding is invalid")
    owner_count = p1.get("owner_count")
    p1_rows = p1.get("rows")
    if isinstance(owner_count, bool) or not isinstance(owner_count, int) or owner_count <= 0:
        raise CensusContractError("valid shard row P1 owner count is invalid")
    if not isinstance(p1_rows, list) or len(p1_rows) != owner_count * len(LAYER_NAMES):
        raise CensusContractError("valid shard row P1 readout matrix is incomplete")
    return row_identity


def _validate_valid_rows_against_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
) -> None:
    identities = [
        _validate_valid_census_row(row, manifest=manifest)
        for row in rows
    ]
    for key in ("prefix", "mrope"):
        values = sorted(identity[key]["sha256"] for identity in identities)
        if manifest["identity"][key]["sha256"] != sha256_json(values):
            raise CensusContractError(f"valid shard image-local {key} aggregate differs from manifest")


def write_shard(
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    census_rows: Sequence[Mapping[str, Any]],
    status: Literal["valid", "technical_invalid", "quarantined"] = "valid",
    invalid_reason: str | None = None,
) -> dict[str, Any]:
    """Write one shard's small JSON artifacts with fail-collision semantics."""

    normalized_manifest = validate_shard_manifest(manifest)
    if status != "valid" and not invalid_reason:
        raise CensusContractError("technical_invalid/quarantined shard requires invalid_reason")
    rows = [dict(row) for row in census_rows]
    expected_images = set(normalized_manifest["image_ids"])
    observed_images = [row.get("image_id") for row in rows]
    if len(observed_images) != len(set(observed_images)):
        raise CensusContractError("shard census contains duplicate image IDs")
    if not set(observed_images).issubset(expected_images):
        raise CensusContractError("shard census contains an image outside its partition")
    if status == "valid" and set(observed_images) != expected_images:
        raise CensusContractError("valid shard census does not cover its exact image partition")
    if status == "valid":
        _validate_valid_rows_against_manifest(rows, manifest=normalized_manifest)
    rows_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_bytes = canonical_json_bytes(normalized_manifest) + b"\n"
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": normalized_manifest["checkpoint"],
        "shard": normalized_manifest["shard"],
        "status": status,
        "technical_invalid": status != "valid",
        "quarantined": status == "quarantined",
        "invalid_reason": invalid_reason,
        "row_count": len(rows),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "census_rows_sha256": sha256_bytes(rows_bytes),
        "partition_sha256": normalized_manifest["partition_sha256"],
    }
    _immutable_write(root / "manifest.json", manifest_bytes)
    _immutable_write(root / "p1-census.jsonl", rows_bytes)
    _immutable_write(root / "receipt.json", canonical_json_bytes(receipt) + b"\n")
    if status == "quarantined":
        _immutable_write(root / "quarantine.json", canonical_json_bytes({"reason": invalid_reason, "receipt_sha256": sha256_json(receipt)}) + b"\n")
    return {"output_dir": str(root), "manifest": normalized_manifest, "receipt": receipt, "rows": rows}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CensusContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise CensusContractError(f"JSON artifact must be an object: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CensusContractError(f"cannot read JSONL artifact: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CensusContractError(f"invalid JSONL row {line_number}: {path}") from exc
        if not isinstance(value, Mapping):
            raise CensusContractError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(dict(value))
    return rows


def merge_shards(
    shard_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    expected_checkpoint: Literal["S", "A"] | None = None,
    expected_image_ids: Sequence[int | str] | None = None,
) -> dict[str, Any]:
    """CPU-only exact partition merge; emits P1 census and a receipt."""

    if not shard_dirs:
        raise CensusContractError("merge requires at least one shard")
    loaded: list[tuple[dict[str, Any], list[dict[str, Any]], Path]] = []
    for source in shard_dirs:
        root = Path(source).expanduser().resolve()
        manifest_path = root / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = validate_shard_manifest(_read_json(manifest_path))
        rows_path = root / "p1-census.jsonl"
        rows = _read_jsonl(rows_path)
        receipt = _read_json(root / "receipt.json")
        rows_bytes = rows_path.read_bytes()
        if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION or receipt.get("unit_id") != UNIT_ID:
            raise CensusContractError(f"shard receipt schema/identity mismatch: {root}")
        if receipt.get("manifest_sha256") != sha256_bytes(manifest_bytes):
            raise CensusContractError(f"shard manifest hash mismatch: {root}")
        if receipt.get("census_rows_sha256") != sha256_bytes(rows_bytes):
            raise CensusContractError(f"shard census hash mismatch: {root}")
        if (
            receipt.get("checkpoint") != manifest["checkpoint"]
            or receipt.get("shard") != manifest["shard"]
            or receipt.get("partition_sha256") != manifest["partition_sha256"]
            or receipt.get("row_count") != len(rows)
        ):
            raise CensusContractError(f"shard receipt does not bind manifest/partition/rows: {root}")
        if (
            receipt.get("status") != "valid"
            or receipt.get("technical_invalid") is not False
            or receipt.get("quarantined") is not False
            or receipt.get("invalid_reason") is not None
        ):
            raise CensusContractError(f"technical-invalid/quarantined shard cannot enter merge: {root}")
        _validate_valid_rows_against_manifest(rows, manifest=manifest)
        loaded.append((manifest, rows, root))
    checkpoints = {manifest["checkpoint"] for manifest, _, _ in loaded}
    if len(checkpoints) != 1 or (expected_checkpoint is not None and next(iter(checkpoints)) != expected_checkpoint):
        raise CensusContractError("merge shards must share one checkpoint")
    counts = {int(manifest["shard"]["count"]) for manifest, _, _ in loaded}
    if len(counts) != 1:
        raise CensusContractError("merge shards must share one shard_count")
    shard_count = next(iter(counts))
    indices = [int(manifest["shard"]["index"]) for manifest, _, _ in loaded]
    if sorted(indices) != list(range(shard_count)):
        raise CensusContractError("merge requires an exact 0..n-1 shard partition")
    identities = [manifest["identity"] for manifest, _, _ in loaded]
    common_identity_keys = ("config", "panel", "cohort", "h0", "runtime", "wrapper")
    for key in common_identity_keys:
        if any(identity.get(key) != identities[0].get(key) for identity in identities[1:]):
            raise CensusContractError(f"merge shard common identity differs: {key}")
    partitions = [tuple(int(value) for value in manifest["image_ids"]) for manifest, _, _ in loaded]
    flattened_partitions = sum((list(partition) for partition in partitions), [])
    if len(flattened_partitions) != len(set(flattened_partitions)):
        raise CensusContractError("merge shard partitions contain duplicate image IDs")
    all_image_ids = sorted(set(flattened_partitions))
    if expected_image_ids is not None and all_image_ids != sorted({_strict_int(int(value), "expected_image_id", minimum=0) for value in expected_image_ids}):
        raise CensusContractError("merge image partition differs from expected cohort images")
    for manifest, _rows, _root in loaded:
        expected_partition = partition_image_ids(all_image_ids, shard_index=int(manifest["shard"]["index"]), shard_count=shard_count)
        if tuple(int(value) for value in manifest["image_ids"]) != expected_partition:
            raise CensusContractError("merge shard does not match deterministic image partition")
    rows: list[dict[str, Any]] = []
    seen_images: set[int] = set()
    for manifest, shard_rows, _root in loaded:
        expected = set(int(value) for value in manifest["image_ids"])
        for row in shard_rows:
            image_id = row.get("image_id")
            if not isinstance(image_id, int) or image_id not in expected or image_id in seen_images:
                raise CensusContractError("merge row violates exact image partition")
            seen_images.add(image_id)
            rows.append(row)
    if seen_images != set(all_image_ids):
        raise CensusContractError("merge rows do not cover the declared image partition")
    rows.sort(key=lambda row: (int(row["image_id"]), str(row.get("checkpoint", ""))))
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    census_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "merged",
        "checkpoint": next(iter(checkpoints)),
        "shard_count": shard_count,
        "shards": [{"index": int(manifest["shard"]["index"]), "path": str(root), "manifest_sha256": sha256_json(manifest), "runtime_attestation": manifest.get("runtime_attestation")} for manifest, _rows, root in loaded],
        "image_ids": all_image_ids,
        "image_count": len(all_image_ids),
        "p1_row_count": len(rows),
        "p1_census_sha256": sha256_bytes(census_bytes),
        "partition_sha256": sha256_json(all_image_ids),
        "identity": {key: identities[0][key] for key in common_identity_keys},
        "per_shard_runtime_attestations": [{"index": int(manifest["shard"]["index"]), "attestation": manifest.get("runtime_attestation")} for manifest, _rows, _root in loaded],
        "technical_invalid_excluded": True,
    }
    _immutable_write(output / "p1-census.jsonl", census_bytes)
    _immutable_write(output / "p1-receipt.json", canonical_json_bytes(receipt) + b"\n")
    return {"output_dir": str(output), "rows": rows, "receipt": receipt}


def _read_json_mapping(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CensusContractError(f"invalid JSON input: {resolved}") from exc
    if not isinstance(value, Mapping):
        raise CensusContractError(f"JSON input must be an object: {resolved}")
    return dict(value)


def _validate_optional_ledger_binding(cohort: Mapping[str, Any], ledger: Path | None) -> dict[str, Any] | None:
    if ledger is None:
        return None
    resolved = ledger.expanduser().resolve(strict=True)
    digest = sha256_bytes(resolved.read_bytes())
    sources = cohort.get("sources")
    declared: list[Mapping[str, Any]] = []
    if isinstance(sources, Mapping):
        values = sources.get("h0_ledgers")
        if isinstance(values, list):
            declared = [item for item in values if isinstance(item, Mapping)]
    if declared and not any(item.get("path") and Path(str(item["path"])).expanduser().resolve() == resolved and item.get("sha256") == digest for item in declared):
        raise CensusContractError("--ledger does not match any immutable cohort H0 ledger source")
    return {"path": str(resolved), "sha256": digest}


def _cohort_image_owner_ids(cohort: Mapping[str, Any], *, checkpoint: str) -> dict[int, tuple[str, ...]]:
    """Select final-cohort owner IDs without re-ranking or substituting rows."""

    events = cohort.get("events")
    if not isinstance(events, list):
        raise CensusContractError("cohort.events must be a list")
    by_image: dict[int, list[str]] = {}
    for event in events:
        if not isinstance(event, Mapping):
            raise CensusContractError("cohort event must be an object")
        try:
            image_id = int(event["image_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CensusContractError("cohort event image_id is invalid") from exc
        owner_id = event.get("gt_owner_id", event.get("owner_id"))
        if not isinstance(owner_id, str) or not owner_id:
            continue
        status = event.get("checkpoint_status")
        branch = status.get(checkpoint) if isinstance(status, Mapping) else None
        disposition = branch.get("disposition") if isinstance(branch, Mapping) else "established"
        if disposition not in {None, "established"}:
            continue
        by_image.setdefault(image_id, []).append(owner_id)
    return {image: tuple(dict.fromkeys(values)) for image, values in sorted(by_image.items())}


def _native_contract_to_census(
    native_contract: Any,
    *,
    checkpoint: str,
    h0_prefix_sha256: str,
    mrope_sha256: str,
    prefix_token_ids_sha256: str,
) -> WrapperContract:
    """Convert the existing experiment adapter's native wrapper contract."""

    assistant_format = str(getattr(native_contract, "assistant_format", ""))
    terminal_id = getattr(native_contract, "commit_token_id", None)
    terminal_name = "commit"
    if terminal_id is None:
        terminal_id = getattr(native_contract, "box_end_token_id", None)
        terminal_name = "box_end"
    if terminal_id is None:
        raise TechnicalInvalid("native adapter wrapper has no terminal token ID")
    return WrapperContract(
        checkpoint=checkpoint,  # type: ignore[arg-type]
        wrapper=assistant_format,
        terminal_token_id=int(terminal_id),
        terminal_token_name=terminal_name,
        h0_prefix_sha256=h0_prefix_sha256,
        mrope_sha256=mrope_sha256,
        prefix_token_ids_sha256=prefix_token_ids_sha256,
    )


def _authoritative_h0_row_boundaries(
    record: Mapping[str, Any],
    *,
    physical_rows: Sequence[Mapping[str, Any]],
    expected_terminal_token_id: int,
    image_id: int,
) -> list[dict[str, Any]]:
    """Cross-bind global one-to-one H0 matches to exact physical token rows."""

    raw_boundaries = record.get("generated_row_boundaries")
    if not isinstance(raw_boundaries, list) or not raw_boundaries:
        raise TechnicalInvalid(f"H0 generated-row boundaries are missing for image {image_id}")
    declared_valid_count = record.get("valid_prediction_count")
    if declared_valid_count is not None:
        valid_count = _strict_int(
            declared_valid_count,
            f"H0 valid_prediction_count for image {image_id}",
            minimum=0,
        )
        if valid_count != len(raw_boundaries):
            raise TechnicalInvalid(
                f"H0 generated-row boundary count differs from its valid prediction count for image {image_id}"
            )
    normalized_by_row: dict[int, dict[str, Any]] = {}
    strict_owner_ids: set[str] = set()
    previous_generated_order = -1
    for boundary_index, raw in enumerate(raw_boundaries):
        if not isinstance(raw, Mapping):
            raise TechnicalInvalid(f"H0 generated-row boundary {boundary_index} for image {image_id} is malformed")
        generated_order = _strict_int(
            raw.get("generated_order"),
            f"H0 generated-row boundary {boundary_index} generated_order",
            minimum=0,
        )
        if generated_order <= previous_generated_order or generated_order >= len(physical_rows):
            raise TechnicalInvalid(
                f"H0 generated-row boundary {boundary_index} does not name one ordered exact physical row"
            )
        physical = physical_rows[generated_order]
        row_start_step = _strict_int(
            raw.get("row_start_step"),
            f"H0 generated-row boundary {boundary_index} row_start_step",
            minimum=0,
        )
        closure_step = _strict_int(
            raw.get("closure_step"),
            f"H0 generated-row boundary {boundary_index} closure_step",
            minimum=0,
        )
        cumulative_tokens = physical.get("cumulative_token_ids")
        row_tokens = physical.get("row")
        if not isinstance(cumulative_tokens, tuple) or not isinstance(row_tokens, list) or not row_tokens:
            raise TechnicalInvalid(f"exact H0 physical row {generated_order} for image {image_id} is malformed")
        expected_start_step = len(cumulative_tokens) - len(row_tokens)
        expected_closure_step = len(cumulative_tokens) - 1
        if row_start_step != expected_start_step or closure_step != expected_closure_step:
            raise TechnicalInvalid(
                f"H0 generated-row boundary {boundary_index} does not bind the exact token-row span"
            )
        closure_token_id = _strict_int(
            raw.get("closure_token_id"),
            f"H0 generated-row boundary {boundary_index} closure_token_id",
            minimum=0,
        )
        if row_tokens[-1] != expected_terminal_token_id or closure_token_id != row_tokens[-1]:
            raise TechnicalInvalid(
                f"H0 generated-row boundary {boundary_index} does not bind the checkpoint-native closure token"
            )
        match_status = raw.get("match_status")
        owner_id = raw.get("gt_owner_id")
        if match_status == "tp":
            if not isinstance(owner_id, str) or not owner_id or owner_id in strict_owner_ids:
                raise TechnicalInvalid(
                    f"H0 generated-row boundary {boundary_index} violates global one-to-one owner matching"
                )
            strict_owner_ids.add(owner_id)
        elif match_status == "unmatched":
            if owner_id is not None:
                raise TechnicalInvalid(
                    f"H0 unmatched generated-row boundary {boundary_index} unexpectedly names a strict owner"
                )
        else:
            raise TechnicalInvalid(f"H0 generated-row boundary {boundary_index} has an invalid global match status")
        normalized_by_row[generated_order] = {
            "row_index": generated_order,
            "generated_order": generated_order,
            "row_start_step": row_start_step,
            "closure_step": closure_step,
            "closure_token_id": closure_token_id,
            "match_status": match_status,
            "gt_owner_id": owner_id,
            "ledger_boundary_present": True,
            "row_token_ids_sha256": sha256_token_ids(row_tokens),
            "cumulative_token_ids_sha256": str(physical["cumulative_sha256"]),
        }
        previous_generated_order = generated_order
    normalized: list[dict[str, Any]] = []
    for row_index, physical in enumerate(physical_rows):
        authoritative = normalized_by_row.get(row_index)
        if authoritative is not None:
            normalized.append(authoritative)
            continue
        row_tokens = physical.get("row")
        cumulative_tokens = physical.get("cumulative_token_ids")
        if not isinstance(row_tokens, list) or not row_tokens or not isinstance(cumulative_tokens, tuple):
            raise TechnicalInvalid(f"exact H0 physical row {row_index} for image {image_id} is malformed")
        if row_tokens[-1] != expected_terminal_token_id:
            raise TechnicalInvalid(
                f"exact H0 physical row {row_index} lacks the checkpoint-native closure token"
            )
        normalized.append({
            "row_index": row_index,
            "generated_order": row_index,
            "row_start_step": len(cumulative_tokens) - len(row_tokens),
            "closure_step": len(cumulative_tokens) - 1,
            "closure_token_id": int(row_tokens[-1]),
            "match_status": "unmatched",
            "gt_owner_id": None,
            "ledger_boundary_present": False,
            "row_token_ids_sha256": sha256_token_ids(row_tokens),
            "cumulative_token_ids_sha256": str(physical["cumulative_sha256"]),
        })
    return normalized


def _existing_experiment_payload(
    adapter: Any,
    *,
    image_id: int,
    owner_ids: Sequence[str],
) -> dict[str, Any]:
    """Bind one existing ``ExperimentRuntimeAdapter`` image to a full H0 forward.

    The adapter already owns processor/model-input identity, H0 row tokens,
    source/derived owner mapping, and native parsing.  This bridge only turns
    those exact artifacts into the observational capture contract; it never
    reconstructs prompts or imports a second runtime.
    """

    runtime = getattr(adapter, "panel_rows", {}).get(str(image_id))
    if runtime is None:
        raise TechnicalInvalid(f"existing runtime adapter has no image {image_id}")
    rows = runtime.h0.get("rows") if isinstance(runtime.h0, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise TechnicalInvalid(f"H0 has no generated rows for image {image_id}")
    ledger = getattr(adapter, "h0_ledger_records", {}).get(str(image_id), {})
    if not isinstance(ledger, Mapping):
        raise TechnicalInvalid(f"H0 ledger has no image record for {image_id}")
    selected = set(str(value) for value in owner_ids)
    prompt_ids = runtime.prompt_ids.detach().clone()
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    device = getattr(adapter, "model_device", prompt_ids.device)
    prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
    generated: list[int] = []
    physical_rows: list[dict[str, Any]] = []
    terminal_bindings: list[dict[str, Any]] = []
    terminal_prefix_hashes: dict[str, str] = {}
    terminal_query_statuses: dict[str, dict[str, Any]] = {}
    for row_index, row_value in enumerate(rows):
        if not isinstance(row_value, Sequence) or isinstance(row_value, (str, bytes)):
            raise TechnicalInvalid(f"H0 row {row_index} for image {image_id} is malformed")
        row = [int(value) for value in row_value]
        if not row:
            raise TechnicalInvalid(f"H0 row {row_index} for image {image_id} is empty")
        generated.extend(row)
        parsed = adapter.parse_row(row, runtime, row_index=row_index)
        match = parsed.get("owner_match") if isinstance(parsed, Mapping) else None
        local_owner_id = (
            str(match["owner_id"])
            if isinstance(match, Mapping)
            and match.get("status") == "unique"
            and isinstance(match.get("owner_id"), str)
            and match.get("owner_id")
            else None
        )
        physical_rows.append(
            {
                "row_index": row_index,
                "row": row,
                "cumulative_token_ids": tuple(generated),
                "cumulative_sha256": sha256_token_ids(generated),
                "local_parse_match_status": match.get("status") if isinstance(match, Mapping) else None,
                "local_parse_owner_id": local_owner_id,
            }
        )
    declared_generated = runtime.h0.get("generated_token_ids") if isinstance(runtime.h0, Mapping) else None
    if not isinstance(declared_generated, list) or [int(value) for value in declared_generated] != generated:
        raise TechnicalInvalid(f"H0 generated-token stream does not exactly equal complete rows for image {image_id}")
    expected_row_terminal = int(getattr(adapter.wrapper_contract, "closure_token_id"))
    for owner_id in sorted(selected):
        record = ledger.get(owner_id)
        if not isinstance(record, Mapping):
            raise TechnicalInvalid(f"H0 ledger lacks final-cohort owner {owner_id}")
        boundary = record.get("natural_boundary")
        if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
            raise TechnicalInvalid(f"H0 owner {owner_id} has an invalid natural boundary")
        prefix_tokens_raw = record.get("exact_prefix_token_ids")
        if not isinstance(prefix_tokens_raw, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in prefix_tokens_raw
        ):
            raise TechnicalInvalid(f"H0 owner {owner_id} has no exact generated-history prefix")
        prefix_tokens = [int(value) for value in prefix_tokens_raw]
        prefix_hash = record.get("exact_prefix_sha256")
        if not _is_sha256(prefix_hash) or sha256_token_ids(prefix_tokens) != prefix_hash:
            raise TechnicalInvalid(f"H0 owner {owner_id} exact terminal prefix hash does not bind its natural boundary")
        covered_raw = record.get("covered_owner_ids", ())
        if not isinstance(covered_raw, (list, tuple)) or any(
            not isinstance(value, str) or not value for value in covered_raw
        ):
            raise TechnicalInvalid(f"H0 owner {owner_id} has invalid strict covered-owner identity")
        covered_owner_ids = tuple(str(value) for value in covered_raw)
        expected_terminal_owner = record.get("latest_covered_owner_id")
        if boundary != len(covered_owner_ids):
            raise TechnicalInvalid(f"H0 owner {owner_id} natural boundary differs from strict covered-owner count")
        if boundary == 0:
            if prefix_tokens or covered_owner_ids or expected_terminal_owner is not None:
                raise TechnicalInvalid(f"H0 root boundary for {owner_id} is not an empty strict history")
            terminal_query_statuses[owner_id] = {
                "status": "not_measured",
                "natural_boundary": 0,
                "not_measured_reason": "root_history_has_no_preceding_terminal",
            }
            continue
        matches = [item for item in physical_rows if list(item["cumulative_token_ids"]) == prefix_tokens]
        if len(matches) != 1:
            raise TechnicalInvalid(f"H0 owner {owner_id} exact prefix does not end at one physical row closure")
        terminal_physical = matches[0]
        terminal_row_index = int(terminal_physical["row_index"])
        terminal_row = terminal_physical["row"]
        if terminal_row[-1] != expected_row_terminal:
            raise TechnicalInvalid(f"H0 boundary row for {owner_id} lacks the checkpoint-native terminal token")
        authoritative_boundaries = _authoritative_h0_row_boundaries(
            record,
            physical_rows=physical_rows,
            expected_terminal_token_id=expected_row_terminal,
            image_id=image_id,
        )
        prefix_boundaries = authoritative_boundaries[: terminal_row_index + 1]
        observed_strict = [
            str(item["gt_owner_id"])
            for item in prefix_boundaries
            if item["match_status"] == "tp"
        ]
        if tuple(observed_strict) != covered_owner_ids:
            raise TechnicalInvalid(f"H0 boundary for {owner_id} does not reproduce its strict covered-owner sequence")
        if not isinstance(expected_terminal_owner, str) or not expected_terminal_owner or observed_strict[-1] != expected_terminal_owner:
            raise TechnicalInvalid(f"H0 boundary for {owner_id} does not match its ledger latest strict owner")
        latest_strict_row_index = max(
            index
            for index, item in enumerate(prefix_boundaries)
            if item["match_status"] == "tp"
        )
        intervening_unmatched = prefix_boundaries[latest_strict_row_index + 1 :]
        if any(item["match_status"] != "unmatched" for item in intervening_unmatched):
            raise TechnicalInvalid(f"H0 boundary for {owner_id} has a strict row after its ledger latest owner")
        terminal_authority = prefix_boundaries[-1]
        physical_terminal_status = str(terminal_authority["match_status"])
        physical_terminal_owner = terminal_authority["gt_owner_id"]
        intervening_unmatched_sha = sha256_json(intervening_unmatched)
        strict_owner_sha = sha256_json(observed_strict)
        binding_receipt = {
            "natural_boundary": boundary,
            "terminal_row_index": terminal_row_index,
            "physical_prefix_row_count": terminal_row_index + 1,
            "physical_terminal_closure_step": int(terminal_authority["closure_step"]),
            "strict_covered_owner_ids": observed_strict,
            "strict_covered_owner_ids_sha256": strict_owner_sha,
            "latest_covered_owner_id": expected_terminal_owner,
            "physical_terminal_match_status": physical_terminal_status,
            "physical_terminal_owner_id": physical_terminal_owner,
            "intervening_unmatched_row_count": len(intervening_unmatched),
            "intervening_unmatched_rows_sha256": intervening_unmatched_sha,
        }
        terminal_bindings.append({
            "owner_id": owner_id,
            "position": int(prompt_ids.shape[1]) + len(prefix_tokens) - 1,
            "token_id": terminal_row[-1],
            "source": "native_h0_complete_row",
            "prefix_sha256": prefix_hash,
            **binding_receipt,
        })
        terminal_prefix_hashes[owner_id] = str(prefix_hash)
        terminal_query_statuses[owner_id] = {
            "status": "measured",
            **binding_receipt,
            "not_measured_reason": None,
        }
    full_ids = torch.cat((prompt_ids, torch.tensor([generated], dtype=torch.long, device=device)), dim=1)
    model_inputs, position_ids, mrope_hash = adapter.exact_model_inputs(runtime, full_ids)
    mrope_hash = str(mrope_hash)
    owners = []
    for owner in runtime.owners:
        if str(owner.get("owner_id")) not in selected:
            continue
        owners.append({
            "owner_id": str(owner["owner_id"]),
            "image_id": int(image_id),
            "class_name": str(owner.get("category", owner.get("category_name", ""))),
            "bbox": list(owner.get("pixel_bbox", owner.get("bbox", []))),
            "image_size": [int(runtime.width), int(runtime.height)],
        })
    if not owners:
        raise TechnicalInvalid(f"final cohort image {image_id} has no owner geometry")
    first_prefix_hash = (
        str(terminal_bindings[0]["prefix_sha256"])
        if terminal_bindings
        else sha256_token_ids([])
    )
    contract = _native_contract_to_census(
        adapter.wrapper_contract,
        checkpoint=str(adapter.checkpoint),
        h0_prefix_sha256=first_prefix_hash,
        mrope_sha256=mrope_hash if _is_sha256(mrope_hash) else sha256_json(mrope_hash),
        prefix_token_ids_sha256=sha256_token_ids(full_ids),
    )
    return {
        "checkpoint": str(adapter.checkpoint),
        "image_id": int(image_id),
        "input_ids": full_ids,
        "position_ids": position_ids,
        "model_inputs": model_inputs,
        "image_positions": list(runtime.image_span.absolute_positions),
        "image_grid_thw": runtime.image_grid_thw.detach().cpu().reshape(-1).tolist(),
        "merge_size": int(runtime.merge_size),
        "terminal_bindings": terminal_bindings,
        "terminal_prefix_hashes": terminal_prefix_hashes,
        "terminal_query_statuses": terminal_query_statuses,
        "contract": contract,
        "owners": owners,
        "prefix_identity": {"full_h0_input_ids_sha256": sha256_token_ids(full_ids), "mrope_sha256": contract.mrope_sha256},
    }


def _adapter_payload(adapter: Any, *, image_id: int, owner_ids: Sequence[str]) -> dict[str, Any]:
    method = getattr(adapter, "build_observational_image", None)
    if callable(method):
        payload = method(image_id=int(image_id), owner_ids=tuple(owner_ids))
        if not isinstance(payload, Mapping):
            raise TechnicalInvalid("injected adapter build_observational_image must return a mapping")
        return dict(payload)
    return _existing_experiment_payload(adapter, image_id=image_id, owner_ids=owner_ids)


def _adapter_identity(adapter: Any, *, checkpoint: str, image_id: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    supplied = getattr(adapter, "identity", None)
    if callable(supplied):
        supplied = supplied()
    if isinstance(supplied, Mapping):
        return dict(supplied)
    existing = getattr(adapter, "_census_identity", None)
    if isinstance(existing, Mapping):
        config_sha = existing.get("config_sha256")
        panel_sha = existing.get("panel_sha256")
        cohort_sha = existing.get("cohort_sha256")
        h0_value = existing.get("h0")
        h0_sha = h0_value.get("run_manifest_sha256") if isinstance(h0_value, Mapping) else None
        resolved = existing.get("resolved_config_fingerprint")
        return {
            "config": {"sha256": config_sha if _is_sha256(config_sha) else sha256_json({"checkpoint": checkpoint, "config": config_sha})},
            "panel": {"sha256": panel_sha if _is_sha256(panel_sha) else sha256_json(existing.get("panel_identity", {}))},
            "cohort": {"sha256": cohort_sha if _is_sha256(cohort_sha) else sha256_json({"cohort": cohort_sha})},
            "h0": {"sha256": h0_sha if _is_sha256(h0_sha) else sha256_json(h0_value or {})},
            "runtime": {"sha256": resolved if _is_sha256(resolved) else sha256_json({"checkpoint": checkpoint, "resolved": resolved})},
            "prefix": {"sha256": str(payload.get("prefix_identity", {}).get("full_h0_input_ids_sha256", sha256_json({"image_id": image_id})))},
            "wrapper": {"sha256": sha256_json(payload.get("contract").recipe_receipt() if isinstance(payload.get("contract"), WrapperContract) else payload.get("contract", {}))},
            "mrope": {"sha256": str(payload.get("contract").mrope_sha256) if isinstance(payload.get("contract"), WrapperContract) else sha256_json(payload.get("prefix_identity", {}))},
        }
    runtime_identity = getattr(adapter, "runtime_identity", None)
    if callable(runtime_identity):
        runtime_identity = runtime_identity()
    return {
        "config": {"sha256": sha256_json({"checkpoint": checkpoint, "config": str(getattr(adapter, "config", "unknown"))})},
        "panel": {"sha256": sha256_json({"panel": str(getattr(adapter, "panel_path", "unknown"))})},
        "cohort": {"sha256": sha256_json({"cohort": str(getattr(adapter, "cohort_path", "unknown"))})},
        "h0": {"sha256": sha256_json({"h0": str(getattr(adapter, "h0_root", "unknown"))})},
        "runtime": {"sha256": sha256_json(runtime_identity or {"checkpoint": checkpoint})},
        "prefix": {"sha256": str(payload.get("prefix_identity", {}).get("full_h0_input_ids_sha256", sha256_json({"image_id": image_id})))},
        "wrapper": {"sha256": sha256_json(payload.get("contract").recipe_receipt() if isinstance(payload.get("contract"), WrapperContract) else payload.get("contract", {}))},
        "mrope": {"sha256": str(payload.get("contract").mrope_sha256) if isinstance(payload.get("contract"), WrapperContract) else sha256_json(payload.get("prefix_identity", {}))},
    }


def _require_cyclic_partner_capacity(assigned: Sequence[int]) -> None:
    """Hold a shard that cannot materialize the declared cross-image control."""

    if len(assigned) < 2:
        raise CensusContractError(
            "preflight hold: cyclic cross-image geometry control requires at least two assigned final-cohort images"
        )


def _is_explicit_test_only_adapter(adapter: Any, *, explicitly_injected: bool) -> bool:
    return bool(explicitly_injected and getattr(adapter, "test_only", False) is True)


def _attest_census_runtime(
    adapter: Any,
    *,
    explicitly_injected: bool,
) -> dict[str, Any]:
    """Attest the loaded runtime, with a narrow test-only injection escape hatch."""

    if _is_explicit_test_only_adapter(adapter, explicitly_injected=explicitly_injected):
        supplied = getattr(adapter, "attest_runtime", None)
        attestation = supplied() if callable(supplied) else getattr(adapter, "runtime_attestation", None)
        source = "explicit_test_only_adapter"
    else:
        # Never accept an adapter field as an attestation for a production
        # adapter.  The owning runtime module validates the live HF session,
        # CUDA visibility, and nvidia-smi/torch UUID mapping together.
        try:
            from scripts.research.run_static_dynamic_owner_interface_experiment import _attest_live_runtime

            attestation = _attest_live_runtime(adapter)
        except Exception as exc:
            raise TechnicalInvalid(f"live runtime attestation failed closed: {exc}") from exc
        source = "owner_interface_live_runtime"
    if not isinstance(attestation, Mapping):
        raise TechnicalInvalid("runtime attestation did not return a mapping")
    normalized = dict(attestation)
    if normalized.get("status") != "validated" or normalized.get("passed") is not True:
        raise TechnicalInvalid("runtime attestation is not validated and passed")
    normalized["census_attestation_source"] = source
    return normalized


def _validate_observational_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint: str,
    image_id: int,
    owner_ids: Sequence[str],
) -> dict[str, Any]:
    """Reject incomplete adapter payloads before any per-image forward begins."""

    normalized = dict(payload)
    if normalized.get("checkpoint", checkpoint) != checkpoint:
        raise TechnicalInvalid("adapter payload checkpoint differs from census checkpoint")
    if normalized.get("image_id", image_id) != image_id:
        raise TechnicalInvalid("adapter payload image_id differs from assigned image")
    required = (
        "input_ids", "position_ids", "image_positions", "image_grid_thw", "merge_size",
        "terminal_bindings", "terminal_prefix_hashes", "terminal_query_statuses", "contract", "owners",
    )
    missing = [key for key in required if key not in normalized]
    if missing:
        raise TechnicalInvalid(f"adapter payload is incomplete: {','.join(missing)}")
    if not isinstance(normalized["contract"], WrapperContract):
        raise TechnicalInvalid("adapter payload contract must be census WrapperContract")
    if normalized["contract"].checkpoint != checkpoint:
        raise TechnicalInvalid("adapter payload wrapper contract checkpoint differs from census checkpoint")
    if not isinstance(normalized["owners"], Sequence) or isinstance(normalized["owners"], (str, bytes)):
        raise TechnicalInvalid("adapter payload owners must be a sequence")
    expected_owners = set(str(value) for value in owner_ids)
    owner_values = [item.get("owner_id", item.get("gt_owner_id")) for item in normalized["owners"] if isinstance(item, Mapping)]
    if len(owner_values) != len(normalized["owners"]) or set(owner_values) != expected_owners or len(owner_values) != len(set(owner_values)):
        raise TechnicalInvalid("adapter payload owners do not exactly cover the assigned final-cohort owners")
    bindings = normalized["terminal_bindings"]
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
        raise TechnicalInvalid("adapter payload terminal_bindings must be a sequence")
    bound_owners = [item.get("owner_id", item.get("gt_owner_id")) for item in bindings if isinstance(item, Mapping)]
    if len(bound_owners) != len(bindings) or not set(bound_owners).issubset(expected_owners) or len(bound_owners) != len(set(bound_owners)):
        raise TechnicalInvalid("adapter payload terminal bindings are not a unique subset of final-cohort owners")
    hashes = normalized["terminal_prefix_hashes"]
    if not isinstance(hashes, Mapping) or set(str(key) for key in hashes) != set(bound_owners) or any(not _is_sha256(value) for value in hashes.values()):
        raise TechnicalInvalid("adapter payload terminal prefix hashes do not exactly cover measured terminal owners")
    statuses = normalized["terminal_query_statuses"]
    if not isinstance(statuses, Mapping) or set(str(key) for key in statuses) != expected_owners:
        raise TechnicalInvalid("adapter payload terminal query statuses do not exactly cover final-cohort owners")
    measured_statuses = {
        str(owner_id)
        for owner_id, value in statuses.items()
        if isinstance(value, Mapping) and value.get("status") == "measured"
    }
    not_measured_statuses = {
        str(owner_id)
        for owner_id, value in statuses.items()
        if isinstance(value, Mapping)
        and value.get("status") == "not_measured"
        and isinstance(value.get("not_measured_reason"), str)
        and value.get("not_measured_reason")
    }
    if measured_statuses != set(bound_owners) or measured_statuses | not_measured_statuses != expected_owners:
        raise TechnicalInvalid("adapter payload terminal query statuses are incomplete or inconsistent")
    return normalized


def _build_manifest_identity(
    adapter: Any,
    *,
    checkpoint: str,
    payloads: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Bind common recipe identities and retain image-local prefix receipts."""

    identities = {
        image_id: _adapter_identity(adapter, checkpoint=checkpoint, image_id=image_id, payload=payload)
        for image_id, payload in sorted(payloads.items())
    }
    if not identities:
        raise TechnicalInvalid("cannot build a shard identity without image payloads")
    first_image_id = min(identities)
    first = identities[first_image_id]
    for key in ("config", "panel", "cohort", "h0", "runtime", "wrapper"):
        if any(identity.get(key) != first.get(key) for identity in identities.values()):
            raise TechnicalInvalid(f"assigned images disagree on common recipe identity: {key}")
    manifest_identity = {key: dict(value) if isinstance(value, Mapping) else value for key, value in first.items()}
    for key in ("prefix", "mrope"):
        values = sorted(str(identity.get(key, {}).get("sha256")) for identity in identities.values())
        if any(not _is_sha256(value) for value in values):
            raise TechnicalInvalid(f"image-local {key} identity is missing or invalid")
        manifest_identity[key] = {"sha256": sha256_json(values)}
    return manifest_identity, identities


def _cyclic_partner_binding(
    *,
    target_image_id: int,
    partner_image_id: int,
    partner_payload: Mapping[str, Any],
    partner_capture: NativePrefillCapture,
) -> dict[str, Mapping[str, Any]]:
    """Return per-layer scalar receipts for a deterministic partner capture."""

    if target_image_id == partner_image_id:
        raise TechnicalInvalid("cyclic partner selection must never select the target image")
    grid = [int(value) for value in partner_payload["image_grid_thw"]]
    prefix_identity = partner_payload.get("prefix_identity")
    prefix_hash = prefix_identity.get("full_h0_input_ids_sha256") if isinstance(prefix_identity, Mapping) else None
    if not _is_sha256(prefix_hash):
        prefix_hash = partner_capture.receipt.get("input_ids_sha256")
    if not _is_sha256(prefix_hash):
        raise TechnicalInvalid("cyclic partner has no full H0 prefix hash")
    mrope_hash = partner_capture.receipt.get("mrope_sha256")
    if not _is_sha256(mrope_hash):
        raise TechnicalInvalid("cyclic partner has no MRoPE hash")
    states = partner_capture.receipt.get("state_receipts")
    if not isinstance(states, Mapping):
        raise TechnicalInvalid("cyclic partner capture has no state receipts")
    bindings: dict[str, Mapping[str, Any]] = {}
    for layer_name in partner_capture.states:
        state_receipt = states.get(layer_name)
        state_hash = state_receipt.get("state_sha256") if isinstance(state_receipt, Mapping) else None
        if not _is_sha256(state_hash):
            raise TechnicalInvalid(f"cyclic partner state receipt is invalid for {layer_name}")
        bindings[layer_name] = {
            "partner_image_id": int(partner_image_id),
            "partner_image_grid_thw": grid,
            "partner_image_grid_sha256": sha256_json(grid),
            "partner_merge_size": int(partner_payload["merge_size"]),
            "partner_prefix_token_ids_sha256": prefix_hash,
            "partner_mrope_sha256": mrope_hash,
            "partner_state_sha256": state_hash,
        }
    return bindings


def _write_quarantined_shard(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    assigned: Sequence[int],
    checkpoint: str,
    runtime_attestation: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Persist only quarantine rows so no partial valid census can escape."""

    rows = [
        {
            "schema_version": P1_CENSUS_SCHEMA_VERSION,
            "status": "technical_invalid",
            "checkpoint": checkpoint,
            "image_id": int(image_id),
            "reason": reason,
            "runtime_attestation": dict(runtime_attestation),
        }
        for image_id in assigned
    ]
    return write_shard(
        output_dir,
        manifest=manifest,
        census_rows=rows,
        status="quarantined",
        invalid_reason=reason,
    )


def _load_existing_experiment_adapter(
    *,
    checkpoint: str,
    infer_config: Path,
    panel: Path,
    cohort: Path,
    h0_dir: Path | None,
    output_dir: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    """Use the existing ExperimentRuntimeAdapter/H0 loader as authority."""

    try:
        from scripts.research.run_static_dynamic_owner_interface_experiment import OwnerInterfaceOrchestrator
    except Exception as exc:  # pragma: no cover - import boundary
        raise CensusContractError(f"existing experiment runtime import failed: {exc}") from exc
    bootstrap = output_dir / "runtime-bootstrap"
    orchestrator = OwnerInterfaceOrchestrator(
        checkpoint=checkpoint, stage="p1", output_dir=bootstrap, config_path=infer_config,
        panel_path=panel, cohort_path=cohort, h0_dir=h0_dir, dry_run=False, fail_collision=False,
    )
    identity = orchestrator._load_cpu_contract()  # noqa: SLF001 - existing authority seam
    adapter = orchestrator._load_adapter(identity)  # noqa: SLF001 - existing authority seam
    setattr(adapter, "_census_identity", identity)
    return adapter, orchestrator, identity


def run_observational_census(
    mode: Literal["contract", "dry-run", "capture", "live", "merge"],
    *,
    checkpoint: Literal["S", "A"],
    infer_config: Path,
    panel: Path,
    cohort: Path,
    h0_dir: Path | None,
    ledger: Path | None,
    image_shard: str,
    output_dir: Path,
    adapter: Any | None = None,
    adapter_loader: Any | None = None,
    merge_shards_input: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Run one CPU contract/dry-run or injected/production capture."""

    shard_checkpoint, shard_index, shard_count = parse_shard_selector(image_shard, default_checkpoint=checkpoint)
    if shard_checkpoint != checkpoint:
        raise CensusContractError("image-shard checkpoint differs from --checkpoint")
    if ledger is not None and not ledger.expanduser().resolve(strict=True).is_file():
        raise CensusContractError(f"--ledger does not resolve to a file: {ledger}")
    if mode == "merge":
        if merge_shards_input is None:
            raise CensusContractError("merge mode requires merge_shards_input")
        return merge_shards(merge_shards_input, output_dir=output_dir, expected_checkpoint=checkpoint)
    adapter_was_explicitly_injected = adapter is not None
    orchestrator = None
    identity: dict[str, Any] | None = None
    if adapter is None:
        if mode in {"contract", "dry-run"}:
            try:
                from scripts.research.run_static_dynamic_owner_interface_experiment import OwnerInterfaceOrchestrator
                orchestrator = OwnerInterfaceOrchestrator(checkpoint=checkpoint, stage="p1", output_dir=output_dir / "runtime-bootstrap", config_path=infer_config, panel_path=panel, cohort_path=cohort, h0_dir=h0_dir, dry_run=True, fail_collision=False)
                identity = orchestrator._load_cpu_contract()  # noqa: SLF001
            except Exception as exc:
                raise CensusContractError(f"contract validation failed: {exc}") from exc
            contract_cohort = _read_json_mapping(cohort)
            contract_ledger = _validate_optional_ledger_binding(contract_cohort, ledger)
            image_ids = sorted({int(event["image_id"]) for event in contract_cohort.get("events", []) if isinstance(event, Mapping) and "image_id" in event})
            assigned = partition_image_ids(image_ids, shard_index=shard_index, shard_count=shard_count)
            _require_cyclic_partner_capacity(assigned)
            receipt = {"schema_version": RECEIPT_SCHEMA_VERSION, "unit_id": UNIT_ID, "mode": mode, "status": "contract_validated" if mode == "contract" else "dry_run_validated", "checkpoint": checkpoint, "image_shard": f"{checkpoint}:{shard_index}/{shard_count}", "assigned_image_ids": list(assigned), "ledger": contract_ledger, "identity": identity}
            _immutable_write(output_dir / "contract-receipt.json", canonical_json_bytes(receipt) + b"\n")
            return receipt
        loader = adapter_loader or _load_existing_experiment_adapter
        loaded = loader(checkpoint=checkpoint, infer_config=infer_config, panel=panel, cohort=cohort, h0_dir=h0_dir, output_dir=output_dir)
        if isinstance(loaded, tuple) and len(loaded) == 3:
            adapter, orchestrator, identity = loaded
        else:
            adapter, orchestrator, identity = loaded, None, None
    cohort_value = _read_json_mapping(cohort)
    ledger_identity = _validate_optional_ledger_binding(cohort_value, ledger)
    if mode in {"contract", "dry-run"}:
        by_image = _cohort_image_owner_ids(cohort_value, checkpoint=checkpoint)
        assigned = partition_image_ids(sorted(by_image), shard_index=shard_index, shard_count=shard_count)
        _require_cyclic_partner_capacity(assigned)
        receipt = {"schema_version": RECEIPT_SCHEMA_VERSION, "unit_id": UNIT_ID, "mode": mode, "status": "contract_validated" if mode == "contract" else "dry_run_validated", "checkpoint": checkpoint, "image_shard": f"{checkpoint}:{shard_index}/{shard_count}", "assigned_image_ids": list(assigned), "ledger": ledger_identity, "identity": identity if identity is not None else _adapter_identity(adapter, checkpoint=checkpoint, image_id=assigned[0] if assigned else 0, payload={})}
        _immutable_write(output_dir / "contract-receipt.json", canonical_json_bytes(receipt) + b"\n")
        return receipt
    by_image = _cohort_image_owner_ids(cohort_value, checkpoint=checkpoint)
    assigned = partition_image_ids(sorted(by_image), shard_index=shard_index, shard_count=shard_count)
    if not assigned:
        raise CensusContractError("image shard has no assigned final-cohort images")
    _require_cyclic_partner_capacity(assigned)
    try:
        runtime_attestation = _attest_census_runtime(
            adapter,
            explicitly_injected=adapter_was_explicitly_injected,
        )
        # Phase 0: make every per-image input explicit and complete before a
        # forward.  This prevents a missing later image from producing a
        # deceptively valid partial shard.
        image_payloads = {
            image_id: _validate_observational_payload(
                _adapter_payload(adapter, image_id=image_id, owner_ids=by_image[image_id]),
                checkpoint=checkpoint,
                image_id=image_id,
                owner_ids=by_image[image_id],
            )
            for image_id in assigned
        }
        manifest_identity, row_identities = _build_manifest_identity(
            adapter,
            checkpoint=checkpoint,
            payloads=image_payloads,
        )
        contract_for_manifest = image_payloads[assigned[0]]["contract"]
        manifest = build_shard_manifest(
            checkpoint=checkpoint,
            shard_index=shard_index,
            shard_count=shard_count,
            image_ids=sorted(by_image),
            identity=manifest_identity,
            runtime_attestation=runtime_attestation,
            contract=contract_for_manifest,
        )
        prefill_registry = NativePrefillRegistry()
        try:
            # Phase 1: exactly one native prefill for every assigned image.
            for image_id in assigned:
                payload = image_payloads[image_id]
                contract = payload["contract"]
                prefill_registry.capture_once(
                    checkpoint,
                    image_id,
                    lambda payload=payload, image_id=image_id, contract=contract: capture_native_prefill(
                        adapter.model,
                        checkpoint=checkpoint,
                        image_id=image_id,
                        input_ids=payload["input_ids"],
                        image_positions=payload["image_positions"],
                        terminal_bindings=payload["terminal_bindings"],
                        terminal_prefix_hashes=payload["terminal_prefix_hashes"],
                        terminal_query_statuses=payload["terminal_query_statuses"],
                        contract=contract,
                        position_ids=payload["position_ids"],
                        model_inputs=payload.get("model_inputs"),
                        attention_mask=payload.get("attention_mask"),
                    ),
                )
            expected_keys = tuple((checkpoint, image_id) for image_id in assigned)
            if prefill_registry.keys() != expected_keys:
                raise TechnicalInvalid("native prefill registry does not exactly cover the assigned image shard")

            # Phase 2: only after every image is captured, pair each image with
            # the next sorted image in the shard.  This is deterministic and
            # target-blind, and it cannot fall back to an unavailable control.
            rows: list[dict[str, Any]] = []
            for offset, image_id in enumerate(assigned):
                partner_image_id = assigned[(offset + 1) % len(assigned)]
                capture = prefill_registry.get(checkpoint, image_id)
                partner_capture = prefill_registry.get(checkpoint, partner_image_id)
                payload = image_payloads[image_id]
                partner_payload = image_payloads[partner_image_id]
                partner_bindings = _cyclic_partner_binding(
                    target_image_id=image_id,
                    partner_image_id=partner_image_id,
                    partner_payload=partner_payload,
                    partner_capture=partner_capture,
                )
                census = compute_observational_census(
                    capture,
                    payload["owners"],
                    image_grid_thw=payload["image_grid_thw"],
                    merge_size=int(payload["merge_size"]),
                    next_image_state_by_layer=partner_capture.states,
                    next_image_grid_thw=partner_payload["image_grid_thw"],
                    next_image_merge_size=int(partner_payload["merge_size"]),
                    next_image_binding_by_layer=partner_bindings,
                )
                rows.append({
                    "schema_version": P1_CENSUS_SCHEMA_VERSION,
                    "status": "valid",
                    "checkpoint": checkpoint,
                    "image_id": image_id,
                    "identity": row_identities[image_id],
                    "runtime_attestation": runtime_attestation,
                    "capture_receipt": capture.receipt,
                    "p1_census": census,
                })
        except Exception as exc:
            reason = f"shard_quarantined_after_capture_or_readout_failure:{type(exc).__name__}:{exc}"
            return _write_quarantined_shard(
                output_dir,
                manifest=manifest,
                assigned=assigned,
                checkpoint=checkpoint,
                runtime_attestation=runtime_attestation,
                reason=reason,
            )
        return write_shard(output_dir, manifest=manifest, census_rows=rows, status="valid")
    finally:
        if orchestrator is not None and adapter is not None:
            try:
                adapter.close()
            finally:
                context = getattr(adapter, "_session_context", None)
                if context is not None and callable(getattr(context, "__exit__", None)):
                    context.__exit__(None, None, None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "dry-run", "capture", "live", "merge"), default="contract")
    parser.add_argument("--merge", nargs="+", type=Path, help="CPU-merge shard directories (legacy alias for --mode merge)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--infer-config", "--config", dest="infer_config", type=Path, required=False)
    parser.add_argument("--panel", type=Path, required=False)
    parser.add_argument("--cohort", type=Path, required=False)
    parser.add_argument("--h0-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--image-shard", default=None, help="explicit checkpoint:i/n image partition")
    parser.add_argument("--shard", dest="image_shard_alias", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-image-id", action="append", type=int, default=[])
    parser.add_argument("--cuda-visible-devices", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = "merge" if args.merge else args.mode
    image_shard = args.image_shard or args.image_shard_alias
    try:
        if args.cuda_visible_devices is not None:
            validate_single_numeric_cuda_visible_devices(args.cuda_visible_devices)
        if mode == "merge":
            if not args.merge:
                raise SystemExit("merge mode requires --merge shard directories")
            result = merge_shards(args.merge, output_dir=args.output_dir, expected_checkpoint=args.checkpoint, expected_image_ids=args.expected_image_id or None)
        else:
            if args.infer_config is None or args.panel is None or args.cohort is None:
                raise SystemExit("contract/dry-run/capture/live require --infer-config, --panel, and --cohort")
            if image_shard is None:
                raise SystemExit("contract/dry-run/capture/live require --image-shard i/n")
            result = run_observational_census(mode, checkpoint=args.checkpoint, infer_config=args.infer_config, panel=args.panel, cohort=args.cohort, h0_dir=args.h0_dir, ledger=args.ledger, image_shard=image_shard, output_dir=args.output_dir)
    except (CensusContractError, FileExistsError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

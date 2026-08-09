#!/usr/bin/env python3
"""Static post-LLM image-field probe primitives.

This file is intentionally experiment-local.  It contains the mechanical
seams needed by the static/dynamic owner-interface unit without changing the
shared model or inference runtime:

* exact image-span and wrapper contracts;
* all-layer observational image residual capture;
* row-query-only image-key masks for full-prefix greedy release; and
* bounded multi-position residual capture/replacement at blocks 13 and 23.

The module does not load a checkpoint, train, or interpret an experiment.  A
caller supplies the already materialised native model inputs and an explicit
wrapper contract.  Hidden tests exercise the helpers with a tiny fake Qwen
stack; the production runner can use the same functions after its H0 receipt
has selected an exact prefix.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Literal

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_fixed_encoding_query_scoped_object_centered_spatial_eligibility as query  # noqa: E402


UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
SCHEMA_VERSION = "static_post_llm_image_field_probe.v1"
TOLERANCE = 1e-4
DEFAULT_LAYER_INDICES = tuple(range(28))
CAUSAL_LAYER_INDICES = frozenset({13, 23, 27})
SOURCE_SPECIFIC_OWNER_MATCH_FIELD = "source_specific_physical_owner_match"


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sha256_token_ids(value: torch.Tensor | Sequence[int]) -> str:
    if isinstance(value, torch.Tensor):
        ids = [int(item) for item in value.detach().reshape(-1).cpu().tolist()]
    else:
        ids = [int(item) for item in value]
    return sha256_json(ids)


def _model_device(model: Any, fallback: torch.device | str = "cpu") -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device(fallback)


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("last_hidden_state", "hidden_states"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError("module output does not expose a tensor as its first value")


def _replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    if isinstance(output, Mapping):
        updated = dict(output)
        if "last_hidden_state" in updated:
            updated["last_hidden_state"] = tensor
            return updated
        if "hidden_states" in updated:
            updated["hidden_states"] = tensor
            return updated
    raise TypeError("module output does not support first-tensor replacement")


def _logits(output: Any) -> torch.Tensor:
    value = output.logits if hasattr(output, "logits") else output.get("logits")
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise TypeError("model output must expose logits with shape [batch, sequence, vocab]")
    return value


@dataclass(frozen=True)
class WrapperContract:
    """Explicit tokenizer/wrapper contract for one checkpoint stratum.

    ``generated_suffix`` excludes the already present object-reference start
    token.  A closed row ends at ``box_end_token_id``.  A commit row ends at
    ``commit_token_id`` immediately after ``box_end_token_id``.
    """

    assistant_format: Literal["object_box_closed", "object_box_commit"]
    object_ref_start_token_id: int
    object_ref_end_token_id: int
    box_start_token_id: int
    box_end_token_id: int
    coordinate_token_start_id: int
    coordinate_bin_count: int = 1000
    coordinate_count: int = 4
    commit_token_id: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.assistant_format not in {"object_box_closed", "object_box_commit"}:
            raise ValueError(f"unsupported assistant format {self.assistant_format!r}")
        if self.coordinate_count != 4:
            raise ValueError("the static probe requires four XYXY coordinate tokens")
        if self.assistant_format == "object_box_commit" and self.commit_token_id is None:
            raise ValueError("object_box_commit requires an explicit commit_token_id")
        if self.assistant_format == "object_box_closed" and self.commit_token_id is not None:
            raise ValueError("object_box_closed must not declare a commit token")

    @property
    def closure_token_id(self) -> int:
        return self.commit_token_id if self.commit_token_id is not None else self.box_end_token_id

    def validate_row_start(self, prefix_ids: torch.Tensor | Sequence[int]) -> None:
        values = [int(item) for item in prefix_ids.detach().reshape(-1).cpu().tolist()] if isinstance(prefix_ids, torch.Tensor) else [int(item) for item in prefix_ids]
        if not values or values[-1] != self.object_ref_start_token_id:
            raise ValueError("exact row prefix must end at object_ref_start_token_id")

    def parse_generated_suffix(self, generated_suffix: Sequence[int], *, tokenizer: Any | None = None) -> dict[str, Any]:
        suffix = [int(item) for item in generated_suffix]
        result: dict[str, Any] = {
            "valid": False,
            "reason": None,
            "assistant_format": self.assistant_format,
            "generated_token_ids": suffix,
            "row_token_ids": [self.object_ref_start_token_id, *suffix],
            "description_token_ids": [],
            "coordinate_token_ids": [],
            "coordinate_bins": [],
            "commit_token_id": self.commit_token_id,
            "decoded_description": None,
        }

        def decode(values: Sequence[int]) -> str | None:
            if tokenizer is None or not callable(getattr(tokenizer, "decode", None)):
                return None
            try:
                return str(tokenizer.decode(list(values), skip_special_tokens=False))
            except TypeError:
                return str(tokenizer.decode(list(values)))

        if not suffix:
            result["reason"] = "empty_suffix"
            return result
        if suffix.count(self.object_ref_end_token_id) != 1:
            result["reason"] = "missing_or_repeated_object_ref_end"
            return result
        end = suffix.index(self.object_ref_end_token_id)
        description = suffix[:end]
        result["description_token_ids"] = description
        result["decoded_description"] = decode(description)
        if not description:
            result["reason"] = "empty_description"
            return result
        forbidden_description_tokens = {
            self.object_ref_start_token_id,
            self.box_start_token_id,
            self.box_end_token_id,
        }
        if self.commit_token_id is not None and self.commit_token_id in description:
            result["reason"] = "premature_commit"
            return result
        if any(token in forbidden_description_tokens for token in description):
            result["reason"] = "description_contains_wrapper_token"
            return result
        if end + 1 >= len(suffix) or suffix[end + 1] != self.box_start_token_id:
            result["reason"] = "missing_box_start"
            return result
        coord_start = end + 2
        coord_end = coord_start + self.coordinate_count
        if len(suffix) <= coord_end:
            result["reason"] = "truncated_coordinate_span"
            return result
        coords = suffix[coord_start:coord_end]
        result["coordinate_token_ids"] = coords
        if any(not self.coordinate_token_start_id <= token < self.coordinate_token_start_id + self.coordinate_bin_count for token in coords):
            result["reason"] = "coordinate_token_out_of_range"
            return result
        if suffix[coord_end] != self.box_end_token_id:
            result["reason"] = "missing_box_end_or_extra_coordinate"
            return result
        if self.assistant_format == "object_box_closed":
            if coord_end != len(suffix) - 1:
                result["reason"] = "box_end_not_final"
                return result
        else:
            if self.commit_token_id in suffix[: coord_end + 1]:
                result["reason"] = "premature_commit"
                return result
            if coord_end + 1 >= len(suffix) or suffix[coord_end + 1] != self.commit_token_id:
                result["reason"] = "missing_commit"
                return result
            if suffix.count(self.commit_token_id) != 1 or coord_end + 1 != len(suffix) - 1:
                result["reason"] = "repeated_or_nonfinal_commit"
                return result
        bins = [token - self.coordinate_token_start_id for token in coords]
        result["coordinate_bins"] = bins
        result["parsed_box"] = bins
        result["valid"] = True
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "assistant_format": self.assistant_format,
            "object_ref_start_token_id": self.object_ref_start_token_id,
            "object_ref_end_token_id": self.object_ref_end_token_id,
            "box_start_token_id": self.box_start_token_id,
            "box_end_token_id": self.box_end_token_id,
            "coordinate_token_start_id": self.coordinate_token_start_id,
            "coordinate_count": self.coordinate_count,
            "commit_token_id": self.commit_token_id,
            "closure_token_id": self.closure_token_id,
            "eos_token_id": self.eos_token_id,
        }


@dataclass(frozen=True)
class ImageSpan:
    image_token_id: int
    grid_thw: tuple[int, int, int]
    merge_size: int
    absolute_positions: tuple[int, ...]
    grid_indices: tuple[tuple[int, int, int], ...]

    @property
    def token_count(self) -> int:
        return len(self.absolute_positions)

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.receipt())

    def receipt(self) -> dict[str, Any]:
        return {
            "image_token_id": self.image_token_id,
            "grid_thw": list(self.grid_thw),
            "merge_size": self.merge_size,
            "token_count": self.token_count,
            "absolute_positions": list(self.absolute_positions),
            "grid_indices": [list(item) for item in self.grid_indices],
            "fingerprint": sha256_json({
                "image_token_id": self.image_token_id,
                "grid_thw": list(self.grid_thw),
                "merge_size": self.merge_size,
                "absolute_positions": list(self.absolute_positions),
                "grid_indices": [list(item) for item in self.grid_indices],
            }),
        }


def _one_grid(grid_thw: torch.Tensor | Sequence[int]) -> tuple[int, int, int]:
    tensor = torch.as_tensor(grid_thw, dtype=torch.long).detach().cpu()
    if tensor.ndim == 1 and tensor.numel() == 3:
        values = tensor.tolist()
    elif tensor.ndim == 2 and tuple(tensor.shape) == (1, 3):
        values = tensor[0].tolist()
    else:
        raise ValueError("P1 requires exactly one image grid with shape [3] or [1,3]")
    if any(int(value) <= 0 for value in values):
        raise ValueError("image grid dimensions must be positive")
    return tuple(int(value) for value in values)


def derive_image_span(
    input_ids: torch.Tensor | Sequence[int],
    *,
    image_token_id: int,
    image_grid_thw: torch.Tensor | Sequence[int],
    merge_size: int,
) -> ImageSpan:
    """Map image placeholders to flattened temporal/row/column merger cells."""

    if int(merge_size) <= 0:
        raise ValueError("merge_size must be positive")
    ids = input_ids.detach().cpu() if isinstance(input_ids, torch.Tensor) else torch.as_tensor(input_ids, dtype=torch.long)
    if ids.ndim == 2:
        if tuple(ids.shape[:1]) != (1,):
            raise ValueError("image-span derivation accepts one batch element")
        ids = ids[0]
    if ids.ndim != 1:
        raise ValueError("input_ids must have shape [S] or [1,S]")
    grid = _one_grid(image_grid_thw)
    t, h, w = grid
    if h % int(merge_size) or w % int(merge_size):
        raise ValueError("image grid height/width must be divisible by merge_size")
    expected = t * (h // int(merge_size)) * (w // int(merge_size))
    positions = [int(value) for value in torch.where(ids == int(image_token_id))[0].tolist()]
    if len(positions) != expected:
        raise ValueError(
            f"image placeholder count {len(positions)} does not match merger grid count {expected}"
        )
    cells = tuple(
        (time, row, column)
        for time in range(t)
        for row in range(h // int(merge_size))
        for column in range(w // int(merge_size))
    )
    return ImageSpan(
        image_token_id=int(image_token_id),
        grid_thw=grid,
        merge_size=int(merge_size),
        absolute_positions=tuple(positions),
        grid_indices=cells,
    )


def resolve_image_positions(span: ImageSpan, indices: Sequence[int]) -> list[int]:
    values = [int(index) for index in indices]
    if len(values) != len(set(values)):
        raise ValueError("image-cell indices must be unique")
    if any(index < 0 or index >= span.token_count for index in values):
        raise ValueError("image-cell index is outside the image span")
    return [span.absolute_positions[index] for index in values]


def _relative_image_indices(span: ImageSpan, absolute_positions: Sequence[int]) -> list[int]:
    by_position = {position: index for index, position in enumerate(span.absolute_positions)}
    values = [int(position) for position in absolute_positions]
    if any(position not in by_position for position in values):
        raise ValueError("absolute residual position is outside the image span")
    return [by_position[position] for position in values]


def build_row_query_image_key_mask(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    eligible_image_positions: Sequence[int],
    query_position: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a causal mask changed only at one current row query."""

    length = int(sequence_length)
    if length <= 0 or not 0 <= int(query_position) < length:
        raise ValueError("query position is outside the sequence")
    image = {int(value) for value in image_key_positions}
    eligible = {int(value) for value in eligible_image_positions}
    if any(value < 0 or value >= length for value in image | eligible):
        raise ValueError("image key position is outside the sequence")
    if not eligible.issubset(image):
        raise ValueError("eligible image keys must be a subset of image keys")
    mask = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    blocked = image - eligible
    if blocked:
        blocked_tensor = torch.tensor(sorted(blocked), dtype=torch.long, device=device)
        mask[int(query_position), blocked_tensor] = False
    return mask.unsqueeze(0).unsqueeze(0).contiguous()


def inspect_row_query_image_key_mask(
    mask: torch.Tensor,
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    eligible_image_positions: Sequence[int],
    query_position: int,
) -> dict[str, Any]:
    """Return exact mask-structure evidence; no result is interpreted."""

    length = int(sequence_length)
    if tuple(mask.shape) != (1, 1, length, length) or mask.dtype is not torch.bool:
        raise ValueError("row-query mask must be boolean [1,1,S,S]")
    expected = build_row_query_image_key_mask(
        sequence_length=length,
        image_key_positions=image_key_positions,
        eligible_image_positions=eligible_image_positions,
        query_position=query_position,
        device=mask.device,
    )[0, 0]
    observed = mask[0, 0]
    baseline = torch.tril(torch.ones_like(observed))
    changed = observed != baseline
    image = {int(value) for value in image_key_positions}
    non_image = changed.clone()
    if image:
        image_tensor = torch.tensor(sorted(image), dtype=torch.long, device=mask.device)
        non_image[:, image_tensor] = False
    off_query = changed.clone()
    off_query[int(query_position), :] = False
    eligible = {int(value) for value in eligible_image_positions}
    eligible_changed = 0
    if eligible:
        eligible_tensor = torch.tensor(sorted(eligible), dtype=torch.long, device=mask.device)
        eligible_changed = int(changed[int(query_position), eligible_tensor].sum().item())
    receipt = {
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        "query_position": int(query_position),
        "changed_cell_count": int(changed.sum().item()),
        "changed_non_image_cell_count": int(non_image.sum().item()),
        "changed_off_query_cell_count": int(off_query.sum().item()),
        "eligible_image_cell_changes": eligible_changed,
        "exact_expected_mask_match": bool(torch.equal(observed, expected)),
        "future_key_blocking_unchanged": bool(torch.equal(torch.triu(observed, diagonal=1), torch.triu(baseline, diagonal=1))),
    }
    receipt["passed"] = bool(
        receipt["exact_expected_mask_match"]
        and receipt["changed_non_image_cell_count"] == 0
        and receipt["changed_off_query_cell_count"] == 0
        and receipt["eligible_image_cell_changes"] == 0
        and receipt["future_key_blocking_unchanged"]
    )
    return receipt


def _layer_candidates(model: Any, layer_idx: int) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    for root_name in ("model.language_model", "model.model.language_model", "language_model"):
        owner = model
        try:
            for part in root_name.split("."):
                owner = getattr(owner, part)
            layers = getattr(owner, "layers")
        except (AttributeError, TypeError):
            continue
        if isinstance(layers, (torch.nn.ModuleList, list, tuple)) and 0 <= int(layer_idx) < len(layers):
            candidates.append((f"{root_name}.layers[{int(layer_idx)}]", layers[int(layer_idx)]))
    if not candidates and hasattr(model, "layers"):
        layers = getattr(model, "layers")
        if 0 <= int(layer_idx) < len(layers):
            candidates.append((f"layers[{int(layer_idx)}]", layers[int(layer_idx)]))
    return candidates


def resolve_decoder_layer(model: Any, layer_idx: int) -> tuple[Any, dict[str, Any]]:
    candidates = _layer_candidates(model, layer_idx)
    by_identity: dict[int, list[tuple[str, Any]]] = {}
    for path, module in candidates:
        by_identity.setdefault(id(module), []).append((path, module))
    if len(by_identity) != 1:
        raise ValueError(f"expected one decoder layer {layer_idx}, found {[path for path, _ in candidates]}")
    aliases = next(iter(by_identity.values()))
    module = aliases[0][1]
    class_name = module.__class__.__name__
    if "Qwen3VLTextDecoderLayer" not in class_name and not getattr(module, "_coordexp_decoder_layer", False):
        raise TypeError(f"resolved module is not Qwen3VLTextDecoderLayer: {class_name}")
    return module, {
        "layer_idx": int(layer_idx),
        "module_path": aliases[0][0],
        "module_alias_paths": [path for path, _ in aliases],
        "module_class": class_name,
        "seam": "returned_layer_output_after_full_block",
    }


def validate_causal_layer(layer_idx: int) -> int:
    """Reject accidental broad layer sweeps on the causal seam."""

    value = int(layer_idx)
    if value not in CAUSAL_LAYER_INDICES:
        raise ValueError("causal residual hooks are limited to preregistered blocks 13, 23, and sentinel 27")
    return value


def _resolve_module_path(model: Any, paths: Sequence[str]) -> tuple[Any | None, str | None]:
    for path in paths:
        owner = model
        try:
            for part in path.split("."):
                owner = getattr(owner, part)
            return owner, path
        except AttributeError:
            continue
    return None, None


class ImageResidualCensus:
    """One-forward all-layer image-position observational capture."""

    def __init__(
        self,
        model: Any,
        *,
        span: ImageSpan,
        layer_indices: Sequence[int] = DEFAULT_LAYER_INDICES,
        final_norm_module: Any | None = None,
        merger_module: Any | None = None,
    ) -> None:
        self.model = model
        self.span = span
        self.layer_indices = tuple(int(item) for item in layer_indices)
        if not self.layer_indices:
            raise ValueError("layer_indices must not be empty")
        self.final_norm_module = (
            final_norm_module
            if final_norm_module is not None
            else _resolve_module_path(
                model,
                ("model.language_model.norm", "model.model.language_model.norm", "language_model.norm", "norm"),
            )[0]
        )
        self.merger_module = (
            merger_module
            if merger_module is not None
            else _resolve_module_path(
                model,
                ("model.visual.merger", "visual.merger", "model.visual.merger.linear_fc2"),
            )[0]
        )
        self.handles: list[Any] = []
        self.call_counts: dict[str, int] = {}
        self.states: dict[str, torch.Tensor] = {}
        self.shapes: dict[str, list[int]] = {}

    def _capture(self, name: str, output: Any) -> Any:
        tensor = _first_tensor(output)
        if name == "merger_output" and tensor.ndim == 2:
            if int(tensor.shape[0]) != self.span.token_count:
                raise ValueError(f"{name} output row count does not match the image span")
            selected = tensor
        elif name == "merger_output" and tensor.ndim == 3 and tensor.shape[0] == 1:
            if int(tensor.shape[1]) != self.span.token_count:
                raise ValueError(f"{name} output row count does not match the image span")
            selected = tensor[0]
        else:
            if tensor.ndim < 3 or tensor.shape[0] != 1:
                raise ValueError(f"{name} output must have shape [1,S,H]")
            if int(tensor.shape[1]) <= max(self.span.absolute_positions):
                raise ValueError(f"{name} output is shorter than the image span")
            selected = tensor[0, list(self.span.absolute_positions), :]
        count = self.call_counts.get(name, 0) + 1
        self.call_counts[name] = count
        if count != 1:
            raise RuntimeError(f"{name} hook fired more than once")
        self.states[name] = selected.detach().clone()
        self.shapes[name] = list(tensor.shape)
        return output

    def _pre_hook(self, name: str) -> Callable[..., Any]:
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                raise ValueError(f"{name} pre-hook received no hidden-state tensor")
            self._capture(name, args[0])

        return hook

    def _forward_hook(self, name: str) -> Callable[..., Any]:
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            return self._capture(name, output)

        return hook

    def install(self) -> None:
        if self.handles:
            raise RuntimeError("image residual census hooks are already installed")
        layers: dict[int, Any] = {}
        for index in self.layer_indices:
            module, _receipt = resolve_decoder_layer(self.model, index)
            layers[index] = module
            self.handles.append(module.register_forward_hook(self._forward_hook(f"block_{index}_output")))
        block_zero = layers.get(0)
        if block_zero is None:
            block_zero, _receipt = resolve_decoder_layer(self.model, 0)
        self.handles.append(block_zero.register_forward_pre_hook(self._pre_hook("block_0_input")))
        if self.final_norm_module is not None:
            self.handles.append(self.final_norm_module.register_forward_hook(self._forward_hook("final_norm")))
        if self.merger_module is not None:
            self.handles.append(self.merger_module.register_forward_hook(self._forward_hook("merger_output")))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def validate(self, *, require_final_norm: bool = True) -> None:
        required = ["block_0_input", *(f"block_{index}_output" for index in self.layer_indices)]
        if require_final_norm:
            required.append("final_norm")
        missing = [name for name in required if self.call_counts.get(name) != 1]
        if missing:
            raise RuntimeError(f"all-layer image residual census did not capture exactly once: {missing}")

    def receipt(self, *, require_final_norm: bool = True) -> dict[str, Any]:
        self.validate(require_final_norm=require_final_norm)
        return {
            "schema_version": SCHEMA_VERSION,
            "seam": "post_llm_image_residual_observation",
            "image_span": self.span.receipt(),
            "layer_indices": list(self.layer_indices),
            "call_counts": dict(sorted(self.call_counts.items())),
            "state_shapes": dict(sorted(self.shapes.items())),
            "state_sha256": {name: sha256_tensor(value) for name, value in sorted(self.states.items())},
            "merger_output_layout": "flat image-cell order [N,H] when merger_output is present",
            "deepstack_semantics": "block_0_input_after_scatter; block_0..2 outputs precede later deepstack additions; block_13/23/27 outputs are post-full-block",
            "final_norm_consumption": "per_position_final_norm",
        }

    def __enter__(self) -> "ImageResidualCensus":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()
        self.validate(require_final_norm=self.final_norm_module is not None)


def capture_image_residual_census(
    model: Any,
    *,
    input_ids: torch.Tensor,
    image_token_id: int,
    image_grid_thw: torch.Tensor,
    merge_size: int,
    model_inputs: Mapping[str, Any] | None = None,
    position_ids: torch.Tensor | None = None,
    custom_mask: torch.Tensor | None = None,
    layer_indices: Sequence[int] = DEFAULT_LAYER_INDICES,
    final_norm_module: Any | None = None,
    merger_module: Any | None = None,
) -> dict[str, Any]:
    """Run one native prefill and return image-only all-layer states/receipt."""

    span = derive_image_span(
        input_ids,
        image_token_id=image_token_id,
        image_grid_thw=image_grid_thw,
        merge_size=merge_size,
    )
    census = ImageResidualCensus(
        model,
        span=span,
        layer_indices=layer_indices,
        final_norm_module=final_norm_module,
        merger_module=merger_module,
    )
    with census:
        with torch.inference_mode():
            output = _forward_model(
                model,
                model_inputs=model_inputs,
                input_ids=input_ids,
                position_ids=position_ids,
                custom_mask=custom_mask,
            )
    return {
        "output": output,
        "states": {name: value.detach().clone() for name, value in census.states.items()},
        "receipt": census.receipt(require_final_norm=census.final_norm_module is not None),
    }


def _hidden_states_from_pre_hook(args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None) -> torch.Tensor:
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    if kwargs is not None and isinstance(kwargs.get("hidden_states"), torch.Tensor):
        return kwargs["hidden_states"]
    raise ValueError("decoder block pre-hook did not receive hidden_states")


class MultiPositionResidualCapture:
    """Capture all selected image positions from one returned block output."""

    def __init__(self, module: Any, *, absolute_positions: Sequence[int]) -> None:
        self.module = module
        self.absolute_positions = tuple(int(item) for item in absolute_positions)
        if not self.absolute_positions:
            raise ValueError("residual capture requires at least one image position")
        self.handle: Any | None = None
        self.call_count = 0
        self.state: torch.Tensor | None = None
        self.full_state: torch.Tensor | None = None

    def _hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        tensor = _first_tensor(output)
        if tensor.ndim < 3 or tensor.shape[0] != 1:
            raise ValueError("decoder output must have shape [1,S,H]")
        if max(self.absolute_positions) >= tensor.shape[1]:
            raise ValueError("residual position is outside decoder output")
        self.call_count += 1
        if self.call_count != 1:
            raise RuntimeError("multi-position residual capture fired more than once")
        self.state = tensor[0, list(self.absolute_positions), :].detach().clone()
        self.full_state = tensor.detach().clone()
        return output

    def install(self) -> None:
        if self.handle is not None:
            raise RuntimeError("residual capture hook already installed")
        self.handle = self.module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self) -> "MultiPositionResidualCapture":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()
        if self.call_count != 1 or self.state is None:
            raise RuntimeError("multi-position residual capture did not fire exactly once")


class MultiPositionResidualReplacement:
    """Replace selected image positions and prove complement preservation."""

    def __init__(
        self,
        module: Any,
        *,
        absolute_positions: Sequence[int],
        replacement: torch.Tensor,
        remove_after_first: bool = True,
        operator_arm: Literal["R00", "R10", "R11", "R12"] | None = None,
    ) -> None:
        self.module = module
        self.absolute_positions = tuple(int(item) for item in absolute_positions)
        self.replacement = replacement.detach().clone()
        if not self.absolute_positions:
            raise ValueError("residual replacement requires at least one image position")
        if self.replacement.ndim != 2 or self.replacement.shape[0] != len(self.absolute_positions):
            raise ValueError("replacement must have shape [selected_image_positions, hidden_size]")
        self.remove_after_first = bool(remove_after_first)
        self.operator_arm = operator_arm
        self.handle: Any | None = None
        self.call_count = 0
        self.replacement_count = 0
        self.other_position_max_abs_delta = 0.0
        self.selected_position_max_abs_delta = 0.0
        self.hook_removed_inside_hook = False

    def _hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        tensor = _first_tensor(output)
        if tensor.ndim < 3 or tensor.shape[0] != 1:
            raise ValueError("decoder output must have shape [1,S,H]")
        if max(self.absolute_positions) >= tensor.shape[1]:
            raise ValueError("residual position is outside decoder output")
        self.call_count += 1
        if self.remove_after_first and self.call_count != 1:
            raise RuntimeError("residual replacement fired after its one-shot contract")
        updated = tensor.clone()
        selected = torch.tensor(self.absolute_positions, dtype=torch.long, device=tensor.device)
        replacement = self.replacement.to(device=tensor.device, dtype=tensor.dtype)
        before = tensor[0, selected, :].detach().clone()
        updated[0, selected, :] = replacement
        delta = (updated - tensor).abs()
        selected_delta = (updated[0, selected, :] - before).abs()
        outside = torch.ones_like(delta, dtype=torch.bool)
        outside[0, selected, :] = False
        self.other_position_max_abs_delta = max(
            self.other_position_max_abs_delta,
            float(delta[outside].max().item()) if bool(outside.any()) else 0.0,
        )
        self.selected_position_max_abs_delta = max(
            self.selected_position_max_abs_delta,
            float(selected_delta.max().item()) if bool(selected_delta.any()) else 0.0,
        )
        self.replacement_count += 1
        if self.remove_after_first and self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.hook_removed_inside_hook = True
        return _replace_first_tensor(output, updated)

    def install(self) -> None:
        if self.handle is not None:
            raise RuntimeError("residual replacement hook already installed")
        self.handle = self.module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def receipt(self) -> dict[str, Any]:
        return {
            "absolute_positions": list(self.absolute_positions),
            "replacement_shape": list(self.replacement.shape),
            "call_count": self.call_count,
            "replacement_count": self.replacement_count,
            "other_position_max_abs_delta": self.other_position_max_abs_delta,
            "selected_position_max_abs_delta": self.selected_position_max_abs_delta,
            "hook_removed_inside_hook": self.hook_removed_inside_hook,
            "cleanup_complete": self.handle is None,
            "declared_operator_arm": self.operator_arm,
            "target_tensor_max_abs_delta": self.selected_position_max_abs_delta,
            "non_target_max_abs_delta": self.other_position_max_abs_delta,
            "passed": (
                (self.call_count == 1 and self.replacement_count == 1)
                if self.remove_after_first
                else (self.call_count >= 1 and self.replacement_count == self.call_count)
            ) and self.other_position_max_abs_delta == 0.0,
        }

    def __enter__(self) -> "MultiPositionResidualReplacement":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()
        if self.remove_after_first:
            valid_calls = self.call_count == 1 and self.replacement_count == 1
        else:
            valid_calls = self.call_count >= 1 and self.replacement_count == self.call_count
        if not valid_calls:
            raise RuntimeError("multi-position residual replacement did not satisfy its call contract")
        if self.other_position_max_abs_delta != 0.0:
            raise RuntimeError("residual replacement changed a non-target position")


def capture_post_block_image_field(
    model: Any,
    *,
    layer_idx: int,
    span: ImageSpan,
    input_ids: torch.Tensor,
    model_inputs: Mapping[str, Any] | None = None,
    position_ids: torch.Tensor | None = None,
    custom_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Capture one returned post-full-block image field at a declared layer."""

    layer_idx = validate_causal_layer(layer_idx)
    module, resolution = resolve_decoder_layer(model, layer_idx)
    capture = MultiPositionResidualCapture(module, absolute_positions=span.absolute_positions)
    with capture:
        with torch.inference_mode():
            output = _forward_model(
                model,
                model_inputs=model_inputs,
                input_ids=input_ids,
                position_ids=position_ids,
                custom_mask=custom_mask,
            )
    if capture.state is None:
        raise RuntimeError("post-block image field capture returned no state")
    return {
        "output": output,
        "state": capture.state.detach().clone(),
        "layer_resolution": resolution,
        "receipt": {
            "layer_idx": int(layer_idx),
            "seam": "returned_layer_output_after_full_block",
            "absolute_positions": list(span.absolute_positions),
            "state_shape": list(capture.state.shape),
            "state_sha256": sha256_tensor(capture.state),
            "call_count": capture.call_count,
        },
    }


def replace_post_block_image_field(
    model: Any,
    *,
    layer_idx: int,
    absolute_positions: Sequence[int],
    replacement: torch.Tensor,
    input_ids: torch.Tensor,
    model_inputs: Mapping[str, Any] | None = None,
    position_ids: torch.Tensor | None = None,
    custom_mask: torch.Tensor | None = None,
    remove_after_first: bool = True,
    operator_arm: Literal["R00", "R10", "R11", "R12"] | None = None,
) -> dict[str, Any]:
    """Apply one bounded post-block image-field replacement and receipt."""

    layer_idx = validate_causal_layer(layer_idx)
    module, resolution = resolve_decoder_layer(model, layer_idx)
    hook = MultiPositionResidualReplacement(
        module,
        absolute_positions=absolute_positions,
        replacement=replacement,
        remove_after_first=remove_after_first,
        operator_arm=operator_arm,
    )
    with hook:
        with torch.inference_mode():
            output = _forward_model(
                model,
                model_inputs=model_inputs,
                input_ids=input_ids,
                position_ids=position_ids,
                custom_mask=custom_mask,
            )
    receipt = hook.receipt()
    receipt.update({"layer_idx": int(layer_idx), "layer_resolution": resolution})
    return {"output": output, "receipt": receipt}


def norm_matched_replacement(
    source: torch.Tensor,
    background: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Scale each background vector to the corresponding source L2 norm."""

    if source.ndim != 2 or background.ndim != 2 or source.shape != background.shape:
        raise ValueError("source and background must have identical [N,H] shapes")
    if float(epsilon) <= 0:
        raise ValueError("epsilon must be positive")
    source_norm = source.detach().float().norm(dim=-1)
    background_norm = background.detach().float().norm(dim=-1)
    if bool(((source_norm > epsilon) & (background_norm <= epsilon)).any()):
        raise ValueError("cannot norm-match a nonzero source to a zero background vector")
    scale = torch.where(background_norm > epsilon, source_norm / background_norm, torch.ones_like(source_norm))
    replacement = background.detach().clone() * scale.to(device=background.device, dtype=background.dtype).unsqueeze(-1)
    replacement_norm = replacement.detach().float().norm(dim=-1)
    receipt = {
        "epsilon": float(epsilon),
        "source_norms": source_norm.tolist(),
        "background_norms": background_norm.tolist(),
        "replacement_norms": replacement_norm.tolist(),
        "max_norm_abs_error": float((replacement_norm - source_norm).abs().max().item()) if source.numel() else 0.0,
        "passed": bool(torch.allclose(replacement_norm, source_norm, atol=1e-5, rtol=1e-5)),
    }
    return replacement, receipt


def build_residual_replacement(
    *,
    arm: Literal["R00", "R10", "R11", "R12"],
    target_state: torch.Tensor,
    background_state: torch.Tensor | None = None,
    donor_state: torch.Tensor | None = None,
    shared_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Construct one declared R-arm replacement without touching model code."""

    if arm == "R00":
        replacement = target_state.detach().clone()
        return replacement, {"arm": arm, "operation": "byte_identical_self", "passed": bool(torch.equal(replacement, target_state))}
    if arm == "R10":
        if background_state is None:
            raise ValueError("R10 requires equal-count background_state")
        replacement, receipt = norm_matched_replacement(target_state, background_state)
        return replacement, {"arm": arm, "operation": "norm_matched_background", **receipt}
    if arm == "R11":
        if donor_state is None or donor_state.shape != target_state.shape:
            raise ValueError("R11 requires equal-count donor_state")
        replacement = donor_state.detach().clone()
        return replacement, {"arm": arm, "operation": "equal_count_ab_swap", "passed": True}
    if arm == "R12":
        if shared_state is None:
            raise ValueError("R12 requires shared_state")
        if shared_state.shape != target_state.shape:
            raise ValueError("R12 shared_state must have equal selected-cell count")
        replacement = torch.zeros_like(target_state)
        return replacement, {"arm": arm, "operation": "shared_core_knockout", "interpretation": "density_region_only", "passed": True}
    raise ValueError(f"unknown residual arm {arm!r}")


def _arm_relative_indices(
    arm: str,
    *,
    span: ImageSpan,
    regions: Mapping[str, Sequence[int]] | None,
) -> list[int] | None:
    if arm == "K00":
        return None
    if regions is None:
        raise ValueError(f"{arm} requires explicit image-cell regions")
    all_indices = set(range(span.token_count))
    if arm == "K01":
        return sorted(all_indices)
    if arm == "K10":
        key = "b_exclusive"
    elif arm == "K11":
        key = "covered_a_exclusive"
    elif arm == "K12":
        key = "background"
    elif arm == "K13":
        key = "same_class_competitor"
        if key not in regions or not regions.get(key):
            return None
    else:
        raise ValueError(f"unknown key arm {arm!r}")
    selected = {int(index) for index in regions.get(key, ())}
    if not selected.issubset(all_indices):
        raise ValueError(f"{arm} region contains an image-cell index outside the span")
    if arm == "K11":
        selected = all_indices - selected
    return sorted(selected)


def _forward_model(
    model: Any,
    *,
    model_inputs: Mapping[str, Any] | None,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor | None,
    custom_mask: torch.Tensor | None,
    use_cache: bool = False,
) -> Any:
    device = input_ids.device
    kwargs = {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in (model_inputs or {}).items()
        if key not in {"input_ids", "attention_mask", "position_ids", "past_key_values", "cache_position"}
    }
    kwargs.update({
        "input_ids": input_ids,
        "attention_mask": custom_mask if custom_mask is not None else torch.ones_like(input_ids),
        "use_cache": bool(use_cache),
        "return_dict": True,
    })
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return model(**kwargs)


def _position_ids(
    model: Any,
    *,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    position_ids_builder: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    if not isinstance(image_grid_thw, torch.Tensor):
        raise ValueError("image_grid_thw must be a tensor")
    if image_grid_thw.ndim == 1:
        if image_grid_thw.numel() != 3:
            raise ValueError("image_grid_thw must have shape [3] or [N,3]")
        normalized_grid = image_grid_thw.reshape(1, 3)
    elif image_grid_thw.ndim == 2 and int(image_grid_thw.shape[1]) == 3:
        normalized_grid = image_grid_thw
    else:
        raise ValueError("image_grid_thw must have shape [3] or [N,3]")
    if normalized_grid.shape[0] <= 0 or not bool((normalized_grid > 0).all().item()):
        raise ValueError("image_grid_thw must contain positive dimensions")
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("position_ids require one input batch element")

    config = getattr(model, "config", None)
    if config is None:
        config = getattr(getattr(model, "model", None), "config", None)
    image_token_id = getattr(config, "image_token_id", None)
    vision_config = getattr(config, "vision_config", None)
    merge_size = getattr(vision_config, "spatial_merge_size", None)
    if image_token_id is None or isinstance(merge_size, bool) or not isinstance(merge_size, int) or merge_size <= 0:
        raise ValueError(
            "position_ids require Qwen image-token and spatial-merge identity"
        )
    ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    run_lengths: list[int] = []
    index = 0
    while index < len(ids):
        if ids[index] != int(image_token_id):
            index += 1
            continue
        end = index + 1
        while end < len(ids) and ids[end] == int(image_token_id):
            end += 1
        run_lengths.append(end - index)
        index = end
    expected_lengths: list[int] = []
    for temporal, height, width in normalized_grid.detach().cpu().tolist():
        if int(height) % merge_size or int(width) % merge_size:
            raise ValueError("image grid height/width must be divisible by spatial_merge_size")
        expected_lengths.append(
            int(temporal) * (int(height) // merge_size) * (int(width) // merge_size)
        )
    if run_lengths != expected_lengths:
        raise ValueError(
            "image token run cardinality does not match normalized image_grid_thw"
        )
    normalized_grid = normalized_grid.to(device=input_ids.device, dtype=torch.long)
    if position_ids_builder is not None:
        result = position_ids_builder(
            model=model,
            input_ids=input_ids,
            image_grid_thw=normalized_grid,
        )
    else:
        result = query.derive_explicit_position_ids(
            model,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            image_grid_thw=normalized_grid,
        )
    if not isinstance(result, torch.Tensor) or result.ndim != 3:
        raise ValueError("position_ids builder must return [3,1,S] MRoPE positions")
    if tuple(result.shape[:2]) != (3, 1) or int(result.shape[-1]) != int(input_ids.shape[-1]):
        raise ValueError("position_ids must have shape [3,1,S] matching input_ids")
    return result.to(device=input_ids.device, dtype=torch.long)


def _residual_context(value: Any) -> Any:
    if value is None:
        return nullcontext()
    if hasattr(value, "__enter__") and hasattr(value, "__exit__"):
        return value
    raise TypeError("residual_factory must return a context manager or None")


def _build_operator_receipt(
    *,
    key_arm: str,
    residual_arm: str | None,
    forward_calls: int,
    residual_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if residual_arm is None:
        kind = "native" if key_arm == "K00" else "query_to_image_key_mask"
        return {
            "arm": key_arm,
            "kind": kind,
            "hook_expected": False,
            "hook_installed_count": 0,
            "hook_fired_count": 0,
            "hook_applied_count": 0,
            "cleanup_complete": True,
            "target_tensor_max_abs_delta": 0.0,
            "non_target_max_abs_delta": 0.0,
            "residual_declarations_match": True,
        }
    kind_by_arm = {
        "R00": "residual_self_replacement",
        "R10": "residual_norm_matched_background",
        "R11": "residual_owner_exclusive_swap",
        "R12": "residual_shared_core_knockout",
    }
    installed = len(residual_receipts)
    fired = sum(int(receipt.get("call_count", 0)) for receipt in residual_receipts)
    applied = sum(int(receipt.get("replacement_count", 0)) for receipt in residual_receipts)
    cleanup = all(bool(receipt.get("cleanup_complete")) for receipt in residual_receipts)
    target_drift = max(
        (float(receipt.get("target_tensor_max_abs_delta", float("inf"))) for receipt in residual_receipts),
        default=float("inf"),
    )
    non_target_drift = max(
        (float(receipt.get("non_target_max_abs_delta", float("inf"))) for receipt in residual_receipts),
        default=float("inf"),
    )
    return {
        "arm": residual_arm,
        "kind": kind_by_arm.get(residual_arm, "unknown_residual_operator"),
        "key_arm": key_arm,
        "hook_expected": True,
        "hook_installed_count": installed,
        "hook_fired_count": fired,
        "hook_applied_count": applied,
        "expected_hook_count": int(forward_calls),
        "cleanup_complete": cleanup,
        "target_tensor_max_abs_delta": target_drift,
        "non_target_max_abs_delta": non_target_drift,
        "residual_declarations_match": bool(
            residual_receipts
            and all(receipt.get("declared_operator_arm") == residual_arm for receipt in residual_receipts)
        ),
    }


def generate_complete_row(
    model: Any,
    *,
    prefix_ids: torch.Tensor,
    image_token_id: int,
    image_grid_thw: torch.Tensor,
    merge_size: int,
    runtime_contract: WrapperContract,
    arm: Literal["K00", "K01", "K10", "K11", "K12", "K13"] = "K00",
    regions: Mapping[str, Sequence[int]] | None = None,
    model_inputs: Mapping[str, Any] | None = None,
    max_new_tokens: int = 256,
    tokenizer: Any | None = None,
    position_ids_builder: Callable[..., torch.Tensor] | None = None,
    residual_factory: Callable[..., Any] | None = None,
    residual_arm: Literal["R00", "R10", "R11", "R12"] | None = None,
) -> dict[str, Any]:
    """Greedily release one complete row with a query-to-image-key arm.

    Every step recomputes the exact prefix with explicit MRoPE positions.  No
    teacher-forced likelihood is returned as a release result, and no cached
    post-RoPE K/V is accepted.  ``K13`` returns ``not_applicable`` when the
    checkpoint has no declared same-class competitor region.
    """

    if prefix_ids.ndim != 2 or tuple(prefix_ids.shape[:1]) != (1,):
        raise ValueError("prefix_ids must have shape [1,S]")
    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    if (residual_factory is None) != (residual_arm is None):
        raise ValueError("residual_factory and residual_arm must be declared together")
    runtime_contract.validate_row_start(prefix_ids)
    device = _model_device(model, fallback=prefix_ids.device)
    current_ids = prefix_ids.detach().clone().to(device=device, dtype=torch.long)
    base_span = derive_image_span(
        current_ids,
        image_token_id=image_token_id,
        image_grid_thw=image_grid_thw,
        merge_size=merge_size,
    )
    relative = _arm_relative_indices(arm, span=base_span, regions=regions)
    if arm == "K13" and relative is None:
        return {
            "arm": arm,
            "status": "not_applicable",
            "reason": "same_class_competitor_region_not_declared",
            "runtime_contract": runtime_contract.receipt(),
            "image_span": base_span.receipt(),
        }
    generated: list[int] = []
    selected_log_probs: list[float] = []
    selected_ranks: list[int] = []
    top_prediction_token_ids: list[int] = []
    position_hashes: list[str] = []
    span_hashes: list[str] = []
    mask_receipts: list[dict[str, Any]] = []
    residual_receipts: list[dict[str, Any]] = []
    stop_reason = "max_new_tokens"
    for step in range(int(max_new_tokens)):
        span = derive_image_span(
            current_ids,
            image_token_id=image_token_id,
            image_grid_thw=image_grid_thw,
            merge_size=merge_size,
        )
        if span.fingerprint != base_span.fingerprint:
            raise RuntimeError("image span changed while releasing a row")
        span_hashes.append(span.fingerprint)
        position_ids = _position_ids(
            model,
            input_ids=current_ids,
            image_grid_thw=image_grid_thw.to(device=device),
            position_ids_builder=position_ids_builder,
        )
        position_hashes.append(sha256_tensor(position_ids))
        custom_mask: torch.Tensor | None = None
        if arm != "K00":
            eligible_positions = resolve_image_positions(span, relative or [])
            custom_mask = build_row_query_image_key_mask(
                sequence_length=current_ids.shape[1],
                image_key_positions=span.absolute_positions,
                eligible_image_positions=eligible_positions,
                query_position=current_ids.shape[1] - 1,
                device=device,
            )
            mask_receipts.append(
                inspect_row_query_image_key_mask(
                    custom_mask,
                    sequence_length=current_ids.shape[1],
                    image_key_positions=span.absolute_positions,
                    eligible_image_positions=eligible_positions,
                    query_position=current_ids.shape[1] - 1,
                )
            )
        context_value = None
        if residual_factory is not None:
            context_value = residual_factory(
                step=step,
                input_ids=current_ids,
                position_ids=position_ids,
                span=span,
            )
        with _residual_context(context_value):
            with torch.inference_mode():
                output = _forward_model(
                    model,
                    model_inputs=model_inputs,
                    input_ids=current_ids,
                    position_ids=position_ids,
                    custom_mask=custom_mask,
                )
        if context_value is not None and hasattr(context_value, "receipt"):
            residual_receipts.append(context_value.receipt())
        next_logits = _logits(output)[0, -1].detach().float()
        token = int(torch.argmax(next_logits).item())
        log_probs = torch.log_softmax(next_logits, dim=-1)
        selected_log_probs.append(float(log_probs[token].item()))
        selected_ranks.append(int(1 + (next_logits > next_logits[token]).sum().item()))
        top_prediction_token_ids.append(token)
        generated.append(token)
        current_ids = torch.cat((current_ids, torch.tensor([[token]], dtype=torch.long, device=device)), dim=1)
        if token == runtime_contract.closure_token_id:
            stop_reason = "commit" if runtime_contract.commit_token_id is not None else "box_end"
            break
        if runtime_contract.eos_token_id is not None and token == runtime_contract.eos_token_id:
            stop_reason = "eos"
            break
    parsed = runtime_contract.parse_generated_suffix(generated, tokenizer=tokenizer)
    complete = bool(parsed["valid"] and stop_reason in {"box_end", "commit"})
    residual_call_count = sum(
        int(receipt.get("call_count", 0))
        for receipt in residual_receipts
        if isinstance(receipt, Mapping)
    )
    non_target_drift = max(
        (
            float(receipt.get("other_position_max_abs_delta", 0.0))
            for receipt in residual_receipts
            if isinstance(receipt, Mapping)
        ),
        default=0.0,
    )
    operator_receipt = _build_operator_receipt(
        key_arm=arm,
        residual_arm=residual_arm,
        forward_calls=len(generated),
        residual_receipts=[receipt for receipt in residual_receipts if isinstance(receipt, Mapping)],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "status": "complete" if complete else "incomplete",
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "selected_token_log_probabilities": selected_log_probs,
        "selected_token_ranks": selected_ranks,
        "top_prediction_token_ids": top_prediction_token_ids,
        "stop_reason": stop_reason,
        "closed_at_wrapper": complete,
        "complete_row": complete,
        "parsed": parsed,
        "runtime_contract": runtime_contract.receipt(),
        "exact_prefix_token_ids_sha256": sha256_token_ids(prefix_ids),
        "position_ids_sha256": position_hashes,
        "image_span_fingerprint": base_span.fingerprint,
        "image_span_fingerprints": span_hashes,
        "mask_receipts": mask_receipts,
        "residual_receipts": residual_receipts,
        "hook_counts": {
            "forward_calls": len(generated),
            "residual_hook_calls": residual_call_count,
            "residual_receipt_count": len(residual_receipts),
        },
        "non_target_drift": non_target_drift,
        "operator_receipt": operator_receipt,
        "recomputed_call_count": len(generated) if stop_reason != "max_new_tokens" else int(max_new_tokens),
        "cache_used": False,
    }


def _validate_noop_receipt(
    receipt: Mapping[str, Any],
    *,
    label: str,
    allowed_operator_arms: Sequence[str],
    tolerance: float,
) -> dict[str, Any]:
    """Validate the complete identity/mechanics contract before parity checks."""

    errors: list[str] = []
    required = (
        "generated_token_ids",
        "selected_token_log_probabilities",
        "selected_token_ranks",
        "exact_prefix_token_ids_sha256",
        "position_ids_sha256",
        "image_span_fingerprint",
        "runtime_contract",
        "hook_counts",
        "non_target_drift",
        "operator_receipt",
        "residual_receipts",
    )
    if not isinstance(receipt, Mapping):
        return {"label": label, "valid": False, "errors": ["receipt_not_mapping"]}
    missing = [key for key in required if key not in receipt]
    errors.extend(f"missing:{key}" for key in missing)

    ids = receipt.get("generated_token_ids")
    if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids):
        errors.append("generated_token_ids_must_be_nonempty_int_list")
    probs = receipt.get("selected_token_log_probabilities")
    if not isinstance(probs, list) or not probs or any(
        (type(value) not in {int, float}) or not math.isfinite(float(value)) for value in probs
    ):
        errors.append("selected_token_log_probabilities_must_be_finite_numeric_list")
    ranks = receipt.get("selected_token_ranks")
    if not isinstance(ranks, list) or not ranks or any(type(value) is not int or value < 1 for value in ranks):
        errors.append("selected_token_ranks_must_be_positive_int_list")
    if isinstance(ids, list) and isinstance(probs, list) and len(ids) != len(probs):
        errors.append("token_and_logprob_lengths_differ")
    if isinstance(ids, list) and isinstance(ranks, list) and len(ids) != len(ranks):
        errors.append("token_and_rank_lengths_differ")

    prefix_hash = receipt.get("exact_prefix_token_ids_sha256")
    if not isinstance(prefix_hash, str) or not prefix_hash:
        errors.append("exact_prefix_token_ids_sha256_must_be_nonempty_string")
    position_hashes = receipt.get("position_ids_sha256")
    if not isinstance(position_hashes, list) or not position_hashes or any(
        not isinstance(value, str) or not value for value in position_hashes
    ):
        errors.append("position_ids_sha256_must_be_nonempty_string_list")
    elif isinstance(ids, list) and len(position_hashes) != len(ids):
        errors.append("position_hash_and_token_lengths_differ")
    span_fingerprint = receipt.get("image_span_fingerprint")
    if not isinstance(span_fingerprint, str) or not span_fingerprint:
        errors.append("image_span_fingerprint_must_be_nonempty_string")
    span_fingerprints = receipt.get("image_span_fingerprints")
    if span_fingerprints is not None:
        if not isinstance(span_fingerprints, list) or not span_fingerprints or any(
            not isinstance(value, str) or not value for value in span_fingerprints
        ):
            errors.append("image_span_fingerprints_must_be_nonempty_string_list")
        elif isinstance(ids, list) and len(span_fingerprints) != len(ids):
            errors.append("image_span_fingerprint_and_token_lengths_differ")

    runtime_contract = receipt.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping) or not runtime_contract:
        errors.append("runtime_contract_must_be_nonempty_mapping")
    else:
        if not isinstance(runtime_contract.get("assistant_format"), str) or not runtime_contract.get("assistant_format"):
            errors.append("runtime_contract_assistant_format_missing")
        if type(runtime_contract.get("closure_token_id")) is not int:
            errors.append("runtime_contract_closure_token_id_missing")

    hook_counts = receipt.get("hook_counts")
    if not isinstance(hook_counts, Mapping):
        errors.append("hook_counts_must_be_mapping")
    else:
        for key, minimum in (("forward_calls", 1), ("residual_hook_calls", 0), ("residual_receipt_count", 0)):
            value = hook_counts.get(key)
            if type(value) is not int or value < minimum:
                errors.append(f"hook_counts_{key}_must_be_int_at_least_{minimum}")
        if isinstance(ids, list) and type(hook_counts.get("forward_calls")) is int:
            if hook_counts["forward_calls"] != len(ids):
                errors.append("hook_counts_forward_calls_must_equal_generated_token_count")
    non_target_drift = receipt.get("non_target_drift")
    if type(non_target_drift) not in {int, float} or not math.isfinite(float(non_target_drift)) or float(non_target_drift) < 0:
        errors.append("non_target_drift_must_be_finite_nonnegative_number")
    residual_receipts = receipt.get("residual_receipts")
    if not isinstance(residual_receipts, list):
        errors.append("residual_receipts_must_be_list")
    operator = receipt.get("operator_receipt")
    if not isinstance(operator, Mapping):
        errors.append("operator_receipt_must_be_mapping")
    else:
        arm = operator.get("arm")
        if not isinstance(arm, str) or arm not in set(allowed_operator_arms):
            errors.append(f"operator_arm_not_allowed:{arm}")
        expected_kinds = {
            "K00": "native",
            "K01": "query_to_image_key_mask",
            "R00": "residual_self_replacement",
            "R10": "residual_norm_matched_background",
        }
        if operator.get("kind") != expected_kinds.get(arm):
            errors.append("operator_kind_does_not_match_arm")
        for key in ("hook_installed_count", "hook_fired_count", "hook_applied_count"):
            if type(operator.get(key)) is not int or int(operator[key]) < 0:
                errors.append(f"operator_{key}_must_be_nonnegative_int")
        if type(operator.get("hook_expected")) is not bool:
            errors.append("operator_hook_expected_must_be_bool")
        if type(operator.get("cleanup_complete")) is not bool:
            errors.append("operator_cleanup_complete_must_be_bool")
        for key in ("target_tensor_max_abs_delta", "non_target_max_abs_delta"):
            value = operator.get(key)
            if type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) < 0:
                errors.append(f"operator_{key}_must_be_finite_nonnegative_number")
        if arm in {"K00", "K01"}:
            if operator.get("hook_expected") is not False:
                errors.append("native_or_key_noop_must_declare_no_residual_hook")
            if any(operator.get(key) != 0 for key in ("hook_installed_count", "hook_fired_count", "hook_applied_count")):
                errors.append("native_or_key_noop_hook_counts_must_be_zero")
            if isinstance(hook_counts, Mapping) and (
                hook_counts.get("residual_hook_calls") != 0 or hook_counts.get("residual_receipt_count") != 0
            ):
                errors.append("native_or_key_noop_top_level_residual_counts_must_be_zero")
            if residual_receipts != []:
                errors.append("native_or_key_noop_residual_receipts_must_be_empty")
            if operator.get("cleanup_complete") is not True:
                errors.append("native_or_key_noop_cleanup_must_be_complete")
            if type(operator.get("target_tensor_max_abs_delta")) in {int, float} and float(operator["target_tensor_max_abs_delta"]) > float(tolerance):
                errors.append("native_or_key_noop_target_tensor_drift")
        elif arm in {"R00", "R10"}:
            forward_calls = hook_counts.get("forward_calls") if isinstance(hook_counts, Mapping) else None
            expected_count = operator.get("expected_hook_count")
            if type(expected_count) is not int or expected_count < 1 or expected_count != forward_calls:
                errors.append("residual_expected_hook_count_must_equal_forward_calls")
            if operator.get("hook_expected") is not True:
                errors.append("residual_operator_must_expect_hook")
            for key in ("hook_installed_count", "hook_fired_count", "hook_applied_count"):
                if operator.get(key) != expected_count:
                    errors.append(f"residual_{key}_must_equal_expected_hook_count")
            if isinstance(hook_counts, Mapping) and (
                hook_counts.get("residual_hook_calls") != expected_count
                or hook_counts.get("residual_receipt_count") != expected_count
            ):
                errors.append("residual_top_level_counts_must_equal_expected_hook_count")
            if operator.get("cleanup_complete") is not True:
                errors.append("residual_hook_cleanup_incomplete")
            if operator.get("residual_declarations_match") is not True:
                errors.append("residual_hook_declaration_mismatch")
            if operator.get("key_arm") != "K00":
                errors.append("residual_noop_sentinel_must_preserve_native_key_arm")
            if not isinstance(residual_receipts, list) or len(residual_receipts) != expected_count:
                errors.append("residual_receipt_count_must_equal_expected_hook_count")
            else:
                nested_target_drifts: list[float] = []
                nested_non_target_drifts: list[float] = []
                for index, nested in enumerate(residual_receipts):
                    if not isinstance(nested, Mapping):
                        errors.append(f"residual_receipt_{index}_must_be_mapping")
                        continue
                    if nested.get("declared_operator_arm") != arm:
                        errors.append(f"residual_receipt_{index}_operator_declaration_mismatch")
                    if nested.get("call_count") != 1 or nested.get("replacement_count") != 1:
                        errors.append(f"residual_receipt_{index}_hook_not_fired_and_applied_once")
                    if nested.get("cleanup_complete") is not True:
                        errors.append(f"residual_receipt_{index}_cleanup_incomplete")
                    nested_target = nested.get("target_tensor_max_abs_delta")
                    nested_non_target = nested.get("non_target_max_abs_delta")
                    if type(nested_target) not in {int, float} or not math.isfinite(float(nested_target)):
                        errors.append(f"residual_receipt_{index}_target_drift_invalid")
                    else:
                        nested_target_drifts.append(float(nested_target))
                    if type(nested_non_target) not in {int, float} or not math.isfinite(float(nested_non_target)):
                        errors.append(f"residual_receipt_{index}_non_target_drift_invalid")
                    else:
                        nested_non_target_drifts.append(float(nested_non_target))
                if nested_target_drifts and type(operator.get("target_tensor_max_abs_delta")) in {int, float}:
                    if not math.isclose(max(nested_target_drifts), float(operator["target_tensor_max_abs_delta"]), abs_tol=1e-12, rel_tol=0.0):
                        errors.append("operator_target_drift_does_not_match_nested_receipts")
                if nested_non_target_drifts and type(operator.get("non_target_max_abs_delta")) in {int, float}:
                    if not math.isclose(max(nested_non_target_drifts), float(operator["non_target_max_abs_delta"]), abs_tol=1e-12, rel_tol=0.0):
                        errors.append("operator_non_target_drift_does_not_match_nested_receipts")
            target_drift = operator.get("target_tensor_max_abs_delta")
            if type(target_drift) in {int, float}:
                if arm == "R00" and float(target_drift) > float(tolerance):
                    errors.append("r00_target_tensor_replacement_not_exact_self")
                if arm == "R10" and float(target_drift) <= float(tolerance):
                    errors.append("r10_target_tensor_replacement_did_not_change_target")
        if type(operator.get("non_target_max_abs_delta")) in {int, float} and float(operator["non_target_max_abs_delta"]) > float(tolerance):
            errors.append("operator_non_target_drift_exceeds_tolerance")
        if type(non_target_drift) in {int, float} and float(non_target_drift) > float(tolerance):
            errors.append("top_level_non_target_drift_exceeds_tolerance")
    return {
        "label": label,
        "valid": not errors,
        "errors": errors,
        "operator_arm": operator.get("arm") if isinstance(operator, Mapping) else None,
    }


def compare_noop_receipts(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    tolerance: float = TOLERANCE,
    candidate_operator_arms: Sequence[str] = ("K01", "R00"),
) -> dict[str, Any]:
    """Check K00/K01 or R00 self/no-op parity, failing closed on bad receipts."""

    baseline_validation = _validate_noop_receipt(
        baseline,
        label="baseline",
        allowed_operator_arms=("K00",),
        tolerance=tolerance,
    )
    candidate_validation = _validate_noop_receipt(
        candidate,
        label="candidate",
        allowed_operator_arms=candidate_operator_arms,
        tolerance=tolerance,
    )
    if not baseline_validation["valid"] or not candidate_validation["valid"]:
        return {
            "passed": False,
            "status": "invalid",
            "receipt_validation": {
                "baseline": baseline_validation,
                "candidate": candidate_validation,
            },
            "invalid_reason": "incomplete_or_malformed_noop_receipt",
        }

    left_ids = baseline["generated_token_ids"]
    right_ids = candidate["generated_token_ids"]
    left_probs = baseline["selected_token_log_probabilities"]
    right_probs = candidate["selected_token_log_probabilities"]
    left_ranks = baseline["selected_token_ranks"]
    right_ranks = candidate["selected_token_ranks"]
    max_drift = max((abs(float(a) - float(b)) for a, b in zip(left_probs, right_probs, strict=True)), default=float("inf"))
    ids_equal = left_ids == right_ids
    ranks_equal = left_ranks == right_ranks
    prefix_equal = baseline["exact_prefix_token_ids_sha256"] == candidate["exact_prefix_token_ids_sha256"]
    position_equal = baseline["position_ids_sha256"] == candidate["position_ids_sha256"]
    span_equal = (
        baseline["image_span_fingerprint"] == candidate["image_span_fingerprint"]
        and baseline.get("image_span_fingerprints") == candidate.get("image_span_fingerprints")
    )
    runtime_equal = baseline["runtime_contract"] == candidate["runtime_contract"]
    non_target_drift_equal = (
        float(baseline["non_target_drift"]) <= float(tolerance)
        and float(candidate["non_target_drift"]) <= float(tolerance)
    )
    result = {
        "status": "valid",
        "receipt_validation": {
            "baseline": baseline_validation,
            "candidate": candidate_validation,
        },
        "generated_token_ids_equal": ids_equal,
        "selected_token_ranks_equal": ranks_equal,
        "max_abs_selected_logprob_drift": max_drift,
        "tolerance": float(tolerance),
        "prefix_hash_equal": bool(prefix_equal),
        "position_ids_equal": bool(position_equal),
        "image_span_equal": bool(span_equal),
        "runtime_contract_equal": bool(runtime_equal),
        "operator_lifecycles_valid": True,
        "non_target_drift_within_tolerance": bool(non_target_drift_equal),
    }
    result["passed"] = bool(
        ids_equal
        and ranks_equal
        and max_drift <= float(tolerance)
        and prefix_equal
        and position_equal
        and span_equal
        and runtime_equal
        and non_target_drift_equal
    )
    return result


def qualify_native_tp_actuator(
    *,
    baseline: Mapping[str, Any],
    intervention: Mapping[str, Any],
    target_matcher: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Mechanically qualify a target-support removal/replacement operator."""

    def resolve_match(receipt: Mapping[str, Any], *, label: str) -> tuple[bool | None, str | None, str | None]:
        if not isinstance(receipt, Mapping):
            return None, "invalid", f"{label}_receipt_must_be_mapping"
        if target_matcher is not None:
            try:
                value = target_matcher(receipt)
            except Exception as exc:  # pragma: no cover - defensive contract boundary
                return None, "invalid", f"{label}_target_matcher_error:{type(exc).__name__}"
            if type(value) is not bool:
                return None, "invalid", f"{label}_target_matcher_must_return_bool"
            return value, None, None
        if SOURCE_SPECIFIC_OWNER_MATCH_FIELD not in receipt:
            return None, "indeterminate", f"{label}_missing_{SOURCE_SPECIFIC_OWNER_MATCH_FIELD}"
        value = receipt[SOURCE_SPECIFIC_OWNER_MATCH_FIELD]
        if type(value) is not bool:
            return None, "invalid", f"{label}_{SOURCE_SPECIFIC_OWNER_MATCH_FIELD}_must_be_bool"
        return value, None, None

    baseline_target, baseline_status, baseline_error = resolve_match(baseline, label="baseline")
    intervention_target, intervention_status, intervention_error = resolve_match(intervention, label="intervention")
    errors = [error for error in (baseline_error, intervention_error) if error is not None]
    if errors:
        status = "invalid" if "invalid" in {baseline_status, intervention_status} else "indeterminate"
        return {
            "status": status,
            "operator_qualified": False,
            "target_moved_in_expected_direction": False,
            "baseline_target_match": baseline_target,
            "intervention_target_match": intervention_target,
            "invalid_reason": ";".join(errors),
            "match_evidence": "explicit_callback" if target_matcher is not None else SOURCE_SPECIFIC_OWNER_MATCH_FIELD,
        }
    assert baseline_target is not None and intervention_target is not None
    generated_changed = baseline.get("generated_token_ids") != intervention.get("generated_token_ids")
    moved = bool(baseline_target and not intervention_target)
    result = {
        "status": "qualified" if moved else "unqualified",
        "baseline_target_match": baseline_target,
        "intervention_target_match": intervention_target,
        "generated_token_ids_changed": bool(generated_changed),
        "baseline_complete_row": bool(baseline.get("complete_row")),
        "intervention_complete_row": bool(intervention.get("complete_row")),
        "target_moved_in_expected_direction": moved,
        "operator_qualified": moved,
        "match_evidence": "explicit_callback" if target_matcher is not None else SOURCE_SPECIFIC_OWNER_MATCH_FIELD,
    }
    return result


def assess_block27_sentinel(
    baseline: Mapping[str, Any],
    intervention: Mapping[str, Any],
    *,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    """Block 27 must be mechanically inert; any effect invalidates the seam."""

    parity = compare_noop_receipts(
        baseline,
        intervention,
        tolerance=tolerance,
        candidate_operator_arms=("R00", "R10"),
    )
    effect_detected = False
    if parity.get("status") == "valid":
        effect_detected = not bool(parity.get("passed"))
    return {
        **parity,
        "sentinel_layer": 27,
        "mechanical_operator_valid": bool(
            parity.get("receipt_validation", {}).get("candidate", {}).get("valid")
        ),
        "behavioral_effect_detected": effect_detected,
        "instrumentation_valid": bool(parity["passed"]),
        "effect_detected": effect_detected,
        "technical_invalid": not bool(parity["passed"]),
        "invalid_reason": None if parity["passed"] else "block27_effect_or_receipt_drift",
    }


def build_exact_runtime_receipt(
    *,
    prefix_ids: torch.Tensor,
    position_ids: torch.Tensor,
    span: ImageSpan,
    runtime_contract: WrapperContract,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Create the identity receipt shared by all K/R event artifacts."""

    if attention_mask is not None:
        valid_shapes = {
            tuple(prefix_ids.shape[:2]),
            (int(prefix_ids.shape[0]), 1, int(prefix_ids.shape[1]), int(prefix_ids.shape[1])),
        }
        if tuple(attention_mask.shape) not in valid_shapes:
            raise ValueError("attention_mask must be [batch,sequence] or [batch,1,sequence,sequence]")
    return {
        "schema_version": SCHEMA_VERSION,
        "exact_prefix_token_ids": [int(value) for value in prefix_ids.detach().reshape(-1).cpu().tolist()],
        "exact_prefix_token_ids_sha256": sha256_token_ids(prefix_ids),
        "position_ids_sha256": sha256_tensor(position_ids),
        "position_ids_shape": list(position_ids.shape),
        "image_span": span.receipt(),
        "runtime_contract": runtime_contract.receipt(),
        "attention_mask_sha256": None if attention_mask is None else sha256_tensor(attention_mask),
        "cache_policy": "no post_rope_kv_swap; full-prefix recompute for K arms",
    }


def validate_disjoint_regions(
    span: ImageSpan,
    *,
    exclusive_regions: Mapping[str, Sequence[int]],
    shared_region: Sequence[int] = (),
) -> dict[str, Any]:
    names = list(exclusive_regions)
    sets = {name: set(int(index) for index in values) for name, values in exclusive_regions.items()}
    shared = set(int(index) for index in shared_region)
    for name, values in {**sets, "shared": shared}.items():
        if not values.issubset(set(range(span.token_count))):
            raise ValueError(f"region {name!r} has an image-cell index outside the span")
    overlap_pairs: list[list[str]] = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            if sets[left_name] & sets[right_name]:
                overlap_pairs.append([left_name, right_name])
    shared_overlap = sorted({name for name, values in sets.items() if values & shared})
    return {
        "exclusive_region_names": names,
        "exclusive_overlap_pairs": overlap_pairs,
        "shared_overlap_names": shared_overlap,
        "passed": not overlap_pairs and not shared_overlap,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        raise SystemExit("This experiment-local module requires an H0-owned exact prefix; no model-loading CLI is provided.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "prepared_only",
        "message": "Provide an H0 runtime contract and invoke generate_complete_row directly; no model was loaded.",
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

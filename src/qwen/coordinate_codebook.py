"""Live visual-to-coordinate output codebook for Qwen3-VL.

Output rows are resolved on every forward pass from the live output wrapper
and its independent delta; no detached initial codebook is cached.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import weakref
from contextlib import contextmanager
from collections.abc import Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from src.qwen.tokens import DEFAULT_WRAPPER_TOKENS


COORDINATE_COUNT = 1000
NORMALIZATION_EPS = 1e-6
CODEBOOK_SCHEMA = "coordexp-live-coordinate-codebook-v1"
EARLY_CODEBOOK_SCHEMA = "coordexp-live-coordinate-codebook-v2"
TENSOR_FILE = "coordinate_codebook.safetensors"
METADATA_FILE = "coordinate_codebook.json"
LATE_MASKED_CENTER = "late_masked_center"
EARLY_PATCH_EDGES = "early_patch_edges"
EDGE_ORDER = ("x_left", "x_right", "y_top", "y_bottom")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        save_file(tensors, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _as_coordinate_ids(coordinate_ids: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(coordinate_ids, torch.Tensor):
        if coordinate_ids.ndim != 1:
            raise ValueError("coordinate_ids must be one-dimensional")
        values = coordinate_ids.detach().cpu().tolist()
    else:
        values = list(coordinate_ids)
    result = tuple(int(value) for value in values)
    if len(result) != COORDINATE_COUNT or len(set(result)) != COORDINATE_COUNT:
        raise ValueError("coordinate_ids must contain 1,000 unique IDs")
    if any(value < 0 for value in result):
        raise ValueError("coordinate_ids must be non-negative")
    return result


def _visual_module(model: nn.Module) -> nn.Module:
    candidates: dict[int, nn.Module] = {}
    for owner in model.modules():
        visual = getattr(owner, "visual", None)
        if isinstance(visual, nn.Module) and isinstance(getattr(visual, "merger", None), nn.Module):
            candidates[id(visual)] = visual
    if len(candidates) != 1:
        raise TypeError("model must expose a maintained visual.merger module")
    return next(iter(candidates.values()))


def _output_rows(model: nn.Module, coordinate_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Resolve live base output rows plus the independent output delta."""

    output = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    base = getattr(output, "base", None)
    delta = getattr(output, "shared_embed_delta", None)
    selected = getattr(output, "selected_token_ids", None)
    if not isinstance(base, nn.Module) or not isinstance(delta, nn.Parameter):
        raise TypeError(
            "coordinate codebook requires a selected output wrapper with an independent delta"
        )
    if selected is None:
        selection = getattr(output, "selection", None)
        selected = getattr(selection, "token_ids", None)
    if selected is None:
        raise TypeError("selected output wrapper is missing token-id mapping")
    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=coordinate_ids.device).reshape(-1)
    if selected_tensor.numel() != delta.shape[0] or selected_tensor.numel() != selected_tensor.unique().numel():
        raise ValueError("output wrapper token mapping and delta shape disagree")
    if delta.ndim != 2:
        raise ValueError("output coordinate delta must remain an independent parameter")
    selection = getattr(output, "selection", None)
    token_strings = getattr(selection, "token_strings", None)
    if token_strings is not None:
        wrapper_ids = {
            int(token)
            for token, text in zip(selected_tensor.tolist(), token_strings, strict=True)
            if str(text) in DEFAULT_WRAPPER_TOKENS
        }
        if wrapper_ids.intersection(int(token) for token in coordinate_ids.tolist()):
            raise ValueError("coordinate_ids must exclude selected wrapper token IDs")
    input_wrapper = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    input_delta = getattr(input_wrapper, "shared_embed_delta", None)
    if input_delta is delta:
        raise ValueError("coordinate codebook requires independent input and output deltas")
    weight = getattr(base, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.requires_grad:
        raise ValueError("output wrapper base weight must be a frozen two-dimensional tensor")
    locations = {int(token): index for index, token in enumerate(selected_tensor.tolist())}
    try:
        delta_rows = torch.tensor(
            [locations[int(token)] for token in coordinate_ids.tolist()],
            dtype=torch.long,
            device=delta.device,
        )
    except KeyError as exc:
        raise ValueError("output wrapper does not contain every coordinate row") from exc
    if weight.shape[0] <= int(coordinate_ids.max()):
        raise ValueError("coordinate IDs exceed the output base vocabulary")
    if weight.shape[1] != delta.shape[1]:
        raise ValueError("output base and independent delta hidden sizes disagree")
    base_rows = weight.index_select(0, coordinate_ids.to(weight.device)).to(delta.device)
    return base_rows + delta.index_select(0, delta_rows), int(weight.shape[1])


def _grid_thw(value: Any, spatial_merge_size: int = 2) -> torch.Tensor:
    grid = torch.as_tensor(value, dtype=torch.long)
    if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0:
        raise ValueError("image_grid_thw must have shape (images, 3)")
    if bool((grid[:, 0] != 1).any()) or bool((grid[:, 1:] <= 0).any()):
        raise ValueError("coordinate codebook requires positive still-image grids")
    if spatial_merge_size <= 0 or bool((grid[:, 1:] % spatial_merge_size).any()):
        raise ValueError("image grids must be divisible by the loaded spatial merge size")
    return grid


def _normalized_addresses(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    rows = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    columns = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def _interpolate(rows: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    rows = rows.float()
    t = (u.float() * COORDINATE_COUNT).clamp(0.0, COORDINATE_COUNT - 1.0)
    lower = t.floor().to(torch.long)
    upper = (lower + 1).clamp_max(COORDINATE_COUNT - 1)
    fraction = (t - lower.to(t.dtype)).unsqueeze(-1)
    return rows.index_select(0, lower) * (1.0 - fraction) + rows.index_select(0, upper) * fraction


def _rms_normalize(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return value / torch.sqrt(value.square().mean(dim=-1, keepdim=True) + NORMALIZATION_EPS)


def _patch_edge_coordinates(
    grid: torch.Tensor,
    spatial_merge_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Processed-image patch footprints in the processor's merge-block order."""

    footprints: list[torch.Tensor] = []
    for temporal, height, width in _grid_thw(grid, spatial_merge_size).tolist():
        if temporal != 1:
            raise ValueError("early patch-edge codebook only supports still-image grids")
        merged_h, merged_w = height // spatial_merge_size, width // spatial_merge_size
        rows = torch.arange(height, device=device, dtype=torch.long).view(merged_h, spatial_merge_size)
        columns = torch.arange(width, device=device, dtype=torch.long).view(merged_w, spatial_merge_size)
        patch_y = rows[:, None, :, None].expand(merged_h, merged_w, spatial_merge_size, spatial_merge_size)
        patch_x = columns[None, :, None, :].expand(merged_h, merged_w, spatial_merge_size, spatial_merge_size)
        patch_y = patch_y.reshape(-1)
        patch_x = patch_x.reshape(-1)
        footprints.append(
            torch.stack(
                (
                    patch_x.float() / width,
                    (patch_x + 1).float() / width,
                    patch_y.float() / height,
                    (patch_y + 1).float() / height,
                ),
                dim=-1,
            )
        )
    return torch.cat(footprints, dim=0)


def _visual_dimensions(visual: nn.Module) -> tuple[int, int, int]:
    config = getattr(visual, "config", None)
    merge = getattr(visual, "spatial_merge_size", getattr(config, "spatial_merge_size", None))
    patch = getattr(visual, "patch_size", getattr(config, "patch_size", None))
    patch_embed = getattr(visual, "patch_embed", None)
    vision_width = getattr(patch_embed, "embed_dim", getattr(config, "hidden_size", None))
    if any(value is None for value in (merge, patch, vision_width)):
        raise TypeError("early patch-edge mode requires loaded vision patch, merge, and width settings")
    merge, patch, vision_width = int(merge), int(patch), int(vision_width)
    if min(merge, patch, vision_width) <= 0:
        raise ValueError("loaded vision patch, merge, and width settings must be positive")
    return merge, patch, vision_width


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _seeded_projection(
    input_size: int,
    output_size: int,
    *,
    seed: int,
    device: torch.device,
) -> nn.Linear:
    # Linear's constructor initializes weights, so fork all RNG streams it can touch.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        projection = nn.Linear(input_size, output_size, bias=False, device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(seed)
    nn.init.xavier_uniform_(projection.weight, generator=generator)
    return projection


class CoordinateCodebook(nn.Module):
    """Live output-row address residual at one configured vision location."""

    def __init__(
        self,
        model: nn.Module,
        coordinate_ids: Sequence[int] | torch.Tensor,
        *,
        initial_gain: float = 0.05,
        mode: str = LATE_MASKED_CENTER,
        projection_seed: int = 1729,
    ) -> None:
        super().__init__()
        ids = _as_coordinate_ids(coordinate_ids)
        if mode not in (LATE_MASKED_CENTER, EARLY_PATCH_EDGES):
            raise ValueError(f"unsupported coordinate codebook mode: {mode}")
        if not math.isfinite(float(initial_gain)) or initial_gain <= 0:
            raise ValueError("initial_gain must be a positive finite number")
        if isinstance(projection_seed, bool) or not isinstance(projection_seed, int) or projection_seed < 0:
            raise ValueError("projection_seed must be a non-negative integer")
        self._model_ref = weakref.ref(model)
        visual = _visual_module(model)
        self._visual_ref = weakref.ref(visual)
        self._active_grid: ContextVar[torch.Tensor | None] = ContextVar(
            f"coordinate_codebook_grid_{id(self)}", default=None
        )
        self._active_edge_coordinates: ContextVar[torch.Tensor | None] = ContextVar(
            f"coordinate_codebook_edges_{id(self)}", default=None
        )
        self._checkpoint_original: Any = None
        self._checkpoint_wrapper: Any = None
        self.register_buffer("coordinate_ids", torch.tensor(ids, dtype=torch.long))
        self.mode = mode
        self.projection_seed = projection_seed
        self.patch_size: int | None = None
        self.vision_width: int | None = None
        if mode == EARLY_PATCH_EDGES:
            self.spatial_merge_size, self.patch_size, self.vision_width = _visual_dimensions(visual)
        else:
            # The legacy route historically supported merger-only test doubles.
            self.spatial_merge_size = int(getattr(visual, "spatial_merge_size", 2))
        self.projection: nn.Linear | None = None
        if mode == EARLY_PATCH_EDGES:
            assert self.vision_width is not None
            rows, output_width = _output_rows(model, self.coordinate_ids)
            if rows.shape[0] != COORDINATE_COUNT:
                raise ValueError("output coordinate row count must be exactly 1,000")
            self.projection = _seeded_projection(
                4 * output_width,
                self.vision_width,
                seed=projection_seed,
                device=_module_device(visual.patch_embed),
            )
        self.raw_gain = nn.Parameter(
            torch.tensor(
                math.log(math.expm1(float(initial_gain))),
                dtype=torch.float32,
                device=_module_device(visual),
            )
        )
        self.enabled = True
        self._grid: torch.Tensor | None = None
        self._edge_coordinates_override: torch.Tensor | None = None
        self.capture_injection_stats = False
        self.last_injection_stats: dict[str, float] | None = None
        self._hook_handles: list[Any] = []

    @property
    def gain(self) -> torch.Tensor:
        return F.softplus(self.raw_gain)

    def _effective_rows(self) -> tuple[torch.Tensor, int]:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("coordinate codebook owner was collected")
        return _output_rows(model, self.coordinate_ids)

    def inject(
        self,
        visual_tokens: torch.Tensor,
        grid: torch.Tensor,
        *,
        edge_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return visual_tokens
        if self.mode == EARLY_PATCH_EDGES:
            return self.inject_early(visual_tokens, grid, edge_coordinates=edge_coordinates)
        if visual_tokens.ndim != 2:
            raise ValueError("main merger output must have shape (tokens, hidden)")
        rows, hidden_size = self._effective_rows()
        if visual_tokens.shape[1] != hidden_size:
            raise ValueError("visual merger and output coordinate rows have different hidden sizes")
        addresses: list[torch.Tensor] = []
        offset = 0
        for temporal, height, width in _grid_thw(grid, self.spatial_merge_size).tolist():
            count = (int(height) // self.spatial_merge_size) * (int(width) // self.spatial_merge_size)
            if offset + count > visual_tokens.shape[0]:
                raise ValueError("visual merger output is shorter than image-grid boundaries")
            addresses.append(
                _normalized_addresses(
                    int(height) // self.spatial_merge_size,
                    int(width) // self.spatial_merge_size,
                    device=visual_tokens.device,
                )
            )
            offset += count
        if offset != visual_tokens.shape[0]:
            raise ValueError("visual merger output and image-grid boundaries disagree")
        address = torch.cat(addresses, dim=0)
        x_rows = _interpolate(rows, address[:, 0])
        y_rows = _interpolate(rows, address[:, 1])
        even = torch.arange(hidden_size, device=visual_tokens.device) % 2 == 0
        odd = ~even
        normalized_x = _rms_normalize(x_rows)
        normalized_y = _rms_normalize(y_rows)
        axis_basis = normalized_x * even.to(normalized_x.dtype) + normalized_y * odd.to(normalized_y.dtype)
        normalized_basis = axis_basis / torch.sqrt(axis_basis.square().mean(-1, keepdim=True) + NORMALIZATION_EPS)
        visual_fp32 = visual_tokens.float()
        visual_rms = torch.sqrt(visual_fp32.square().mean(-1, keepdim=True)).detach()
        residual = self.gain.float() * visual_rms * normalized_basis
        return visual_tokens + residual.to(dtype=visual_tokens.dtype)

    def patch_edge_coordinates(self, grid: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
        """Return left, right, top, bottom coordinates for raw patches in processor order."""

        target_device = device or torch.as_tensor(grid).device
        return _patch_edge_coordinates(grid, self.spatial_merge_size, device=target_device)

    @contextmanager
    def override_patch_edges(self, edge_coordinates: torch.Tensor) -> Iterator[None]:
        """Use supplied whole edge tuples in the next installed vision forwards."""

        if self.mode != EARLY_PATCH_EDGES:
            raise RuntimeError("patch-edge overrides require early_patch_edges mode")
        if self._edge_coordinates_override is not None:
            raise RuntimeError("a patch-edge override is already active")
        self._edge_coordinates_override = torch.as_tensor(edge_coordinates).detach()
        try:
            yield
        finally:
            self._edge_coordinates_override = None

    def inject_early(
        self,
        hidden_states: torch.Tensor,
        grid: torch.Tensor,
        *,
        edge_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            self.last_injection_stats = None
            return hidden_states
        self.last_injection_stats = None
        if self.mode != EARLY_PATCH_EDGES or self.projection is None or self.vision_width is None:
            raise RuntimeError("early injection requires early_patch_edges mode")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.vision_width:
            raise ValueError("first vision block input must have shape (raw patches, loaded vision width)")
        footprints = (
            self.patch_edge_coordinates(grid, device=hidden_states.device)
            if edge_coordinates is None
            else torch.as_tensor(edge_coordinates, dtype=torch.float32, device=hidden_states.device)
        )
        if footprints.shape != (hidden_states.shape[0], 4) or not bool(torch.isfinite(footprints).all()):
            raise ValueError("edge coordinates must be finite [raw_patches, 4] tuples")
        if bool(((footprints < 0.0) | (footprints > 1.0)).any()):
            raise ValueError("edge coordinates must lie in processed-image [0, 1]")
        rows, _ = self._effective_rows()
        edge_rows = _interpolate(rows.to(hidden_states.device), footprints.reshape(-1))
        slots = _rms_normalize(edge_rows).reshape(hidden_states.shape[0], 4, -1).flatten(1)
        projected = F.linear(slots, self.projection.weight.float())
        address_residual = _rms_normalize(projected)
        visual_fp32 = hidden_states.float()
        visual_rms = torch.sqrt(visual_fp32.square().mean(-1, keepdim=True)).detach()
        gain = self.gain.to(hidden_states.device).float()
        residual = gain * visual_rms * address_residual
        if self.capture_injection_stats:
            self.last_injection_stats = {
                "gain": float(gain.detach().mean().cpu()),
                "input_rms": float(visual_rms.detach().mean().cpu()),
                "normalized_address_rms": float(
                    address_residual.detach().square().mean(-1).sqrt().mean().cpu()
                ),
                "residual_rms": float(residual.detach().square().mean(-1).sqrt().mean().cpu()),
            }
        return hidden_states + residual.to(dtype=hidden_states.dtype)

    def forward(self, visual_tokens: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        return self.inject(visual_tokens, _grid_thw(grid_thw, self.spatial_merge_size))

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        visual = self._visual_ref()
        blocks = getattr(visual, "blocks", None) if visual is not None else None
        checkpoint_module = blocks[0] if blocks is not None and len(blocks) else None
        if (
            checkpoint_module is not None
            and self._checkpoint_wrapper is not None
            and getattr(checkpoint_module, "_gradient_checkpointing_func", None) is self._checkpoint_wrapper
        ):
            checkpoint_module._gradient_checkpointing_func = self._checkpoint_original
        self._checkpoint_original = None
        self._checkpoint_wrapper = None

    def architecture_metadata(self) -> dict[str, Any]:
        _, output_width = self._effective_rows()
        if self.mode == LATE_MASKED_CENTER:
            return {
                "schema": CODEBOOK_SCHEMA,
                "coordinate_ids": self.coordinate_ids.detach().cpu().tolist(),
                "coordinate_count": COORDINATE_COUNT,
                "hidden_size": output_width,
                "normalization_eps": NORMALIZATION_EPS,
                "axis_masks": {"x": "even_hidden_channels", "y": "odd_hidden_channels"},
                "interpolation": "clip(1000*u,0,999), adjacent linear rows",
                "visual_injection": "main_vision_merger_output_only",
                "enabled": bool(self.enabled),
            }
        if self.projection is None or self.patch_size is None or self.vision_width is None:
            raise RuntimeError("early codebook architecture is incomplete")
        return {
            "schema": EARLY_CODEBOOK_SCHEMA,
            "mode": self.mode,
            "coordinate_ids": self.coordinate_ids.detach().cpu().tolist(),
            "coordinate_count": COORDINATE_COUNT,
            "output_width": output_width,
            "vision_width": self.vision_width,
            "patch_size": self.patch_size,
            "spatial_merge_size": self.spatial_merge_size,
            "projection_seed": self.projection_seed,
            "projection_shape": list(self.projection.weight.shape),
            "normalization_eps": NORMALIZATION_EPS,
            "edge_order": list(EDGE_ORDER),
            "interpolation": "clip(1000*u,0,999), adjacent linear rows",
            "visual_injection": "after_learned_vision_positions_before_first_vision_block",
            "enabled": bool(self.enabled),
        }


def install_coordinate_codebook(
    model: nn.Module,
    coordinate_ids: Sequence[int] | torch.Tensor,
    *,
    initial_gain: float = 0.05,
    mode: str = LATE_MASKED_CENTER,
    projection_seed: int = 1729,
) -> CoordinateCodebook:
    """Attach the codebook at the selected legacy or early vision location."""

    if hasattr(model, "coordinate_codebook"):
        raise ValueError("model already has a coordinate_codebook")
    visual = _visual_module(model)
    codebook = CoordinateCodebook(
        model,
        coordinate_ids,
        initial_gain=initial_gain,
        mode=mode,
        projection_seed=projection_seed,
    )

    def capture_grid(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        value = kwargs.get("grid_thw") if "grid_thw" in kwargs else (args[1] if len(args) > 1 else None)
        if value is None:
            raise ValueError("visual caller must provide image_grid_thw/grid_thw")
        codebook._grid = _grid_thw(value, codebook.spatial_merge_size).detach()
        blocks = getattr(visual, "blocks", None) if mode == EARLY_PATCH_EDGES else None
        checkpoint_module = blocks[0] if blocks is not None and len(blocks) else None
        if mode == EARLY_PATCH_EDGES and checkpoint_module is not None and bool(
            getattr(checkpoint_module, "gradient_checkpointing", False)
        ):
            original = getattr(checkpoint_module, "_gradient_checkpointing_func", None)
            if callable(original) and original is not codebook._checkpoint_wrapper:
                def checkpoint_with_grid(function: Any, *checkpoint_args: Any, **checkpoint_kwargs: Any) -> Any:
                    request_grid = codebook._grid
                    if request_grid is None:
                        request_grid = codebook._active_grid.get()
                    if request_grid is None:
                        raise RuntimeError("checkpointed vision block has no request-local image grid")
                    request_edges = codebook._edge_coordinates_override

                    def run_with_grid(*run_args: Any, **run_kwargs: Any) -> Any:
                        grid_token = codebook._active_grid.set(request_grid)
                        edge_token = codebook._active_edge_coordinates.set(request_edges)
                        try:
                            return function(*run_args, **run_kwargs)
                        finally:
                            codebook._active_edge_coordinates.reset(edge_token)
                            codebook._active_grid.reset(grid_token)

                    return original(run_with_grid, *checkpoint_args, **checkpoint_kwargs)

                codebook._checkpoint_original = original
                codebook._checkpoint_wrapper = checkpoint_with_grid
                checkpoint_module._gradient_checkpointing_func = checkpoint_with_grid

    def apply_merger(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        grid = codebook._grid
        codebook._grid = None
        if grid is None:
            raise RuntimeError("main vision merger ran without a captured image grid")
        return codebook.inject(output, grid.to(output.device))

    codebook._hook_handles = [visual.register_forward_pre_hook(capture_grid, with_kwargs=True)]
    if mode == LATE_MASKED_CENTER:
        codebook._hook_handles.append(visual.merger.register_forward_hook(apply_merger))
    else:
        blocks = getattr(visual, "blocks", None)
        if blocks is None or len(blocks) == 0:
            codebook.remove_hooks()
            raise TypeError("early patch-edge mode requires a first vision block")

        def apply_first_block(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            grid = codebook._active_grid.get()
            if grid is None:
                grid = codebook._grid
            if grid is None:
                raise RuntimeError("first vision block ran without a captured image grid")
            edge_coordinates = codebook._active_edge_coordinates.get()
            if edge_coordinates is None:
                edge_coordinates = codebook._edge_coordinates_override
            if args:
                hidden_states = args[0]
                return (
                    codebook.inject_early(
                        hidden_states,
                        grid.to(hidden_states.device),
                        edge_coordinates=edge_coordinates,
                    ),
                    *args[1:],
                ), kwargs
            hidden_states = kwargs.get("hidden_states")
            if not isinstance(hidden_states, torch.Tensor):
                raise TypeError("first vision block is missing its hidden_states argument")
            changed_kwargs = dict(kwargs)
            changed_kwargs["hidden_states"] = codebook.inject_early(
                hidden_states,
                grid.to(hidden_states.device),
                edge_coordinates=edge_coordinates,
            )
            return args, changed_kwargs

        codebook._hook_handles.append(blocks[0].register_forward_pre_hook(apply_first_block, with_kwargs=True))

        def clear_request_grid(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            codebook._grid = None

        codebook._hook_handles.append(
            visual.register_forward_hook(clear_request_grid, with_kwargs=True, always_call=True)
        )
    setattr(model, "coordinate_codebook", codebook)
    return codebook


def save_coordinate_codebook(model: nn.Module, directory: Path | str) -> dict[str, Any]:
    codebook = getattr(model, "coordinate_codebook", None)
    if not isinstance(codebook, CoordinateCodebook):
        raise ValueError("model has no installed coordinate codebook")
    directory = Path(directory)
    metadata = codebook.architecture_metadata()
    tensors = {"raw_gain": codebook.raw_gain.detach().cpu().contiguous()}
    if codebook.mode == EARLY_PATCH_EDGES:
        if codebook.projection is None:
            raise RuntimeError("early codebook projection is missing")
        tensors["projection.weight"] = codebook.projection.weight.detach().cpu().contiguous()
    _atomic_safetensors(directory / TENSOR_FILE, tensors)
    _atomic_json(directory / METADATA_FILE, metadata)
    return {"tensor_path": str(directory / TENSOR_FILE), "metadata_path": str(directory / METADATA_FILE), **metadata}


def load_coordinate_codebook(model: nn.Module, directory: Path | str) -> CoordinateCodebook:
    codebook = getattr(model, "coordinate_codebook", None)
    if not isinstance(codebook, CoordinateCodebook):
        raise ValueError("model must have an installed coordinate codebook before loading")
    directory = Path(directory)
    metadata_path = directory / METADATA_FILE
    tensor_path = directory / TENSOR_FILE
    if not metadata_path.is_file() or not tensor_path.is_file():
        raise FileNotFoundError("coordinate codebook metadata and tensor payload are both required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = codebook.architecture_metadata()
    if set(metadata) != set(expected):
        raise ValueError("coordinate codebook architecture metadata keys are not exact")
    for key in set(expected) - {"enabled"}:
        if metadata.get(key) != expected.get(key):
            raise ValueError(f"coordinate codebook architecture metadata mismatch: {key}")
    tensors = load_file(str(tensor_path), device=str(codebook.raw_gain.device))
    expected_tensors = {"raw_gain"}
    if codebook.mode == EARLY_PATCH_EDGES:
        expected_tensors.add("projection.weight")
    if set(tensors) != expected_tensors or tensors["raw_gain"].shape != torch.Size([]):
        raise ValueError("coordinate codebook tensor payload does not match its architecture")
    if not bool(torch.isfinite(tensors["raw_gain"]).item()):
        raise ValueError("coordinate codebook raw_gain must be finite")
    if codebook.mode == EARLY_PATCH_EDGES:
        assert codebook.projection is not None
        projection = tensors["projection.weight"]
        if projection.shape != codebook.projection.weight.shape or not bool(torch.isfinite(projection).all()):
            raise ValueError("coordinate codebook projection shape or values are invalid")
    with torch.no_grad():
        codebook.raw_gain.copy_(tensors["raw_gain"].to(dtype=codebook.raw_gain.dtype))
        if codebook.mode == EARLY_PATCH_EDGES:
            assert codebook.projection is not None
            codebook.projection.weight.copy_(tensors["projection.weight"].to(codebook.projection.weight))
    codebook.enabled = bool(metadata.get("enabled", True))
    return codebook


__all__ = [
    "CODEBOOK_SCHEMA",
    "COORDINATE_COUNT",
    "CoordinateCodebook",
    "EARLY_CODEBOOK_SCHEMA",
    "EARLY_PATCH_EDGES",
    "LATE_MASKED_CENTER",
    "install_coordinate_codebook",
    "load_coordinate_codebook",
    "save_coordinate_codebook",
]

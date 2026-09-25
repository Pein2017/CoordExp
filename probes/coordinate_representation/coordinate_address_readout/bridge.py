"""Small frozen-backbone coordinate readout sidecar.

The module only transforms the 1,000 canonical coordinate logits.  The caller
owns the model hook and supplies the effective head state and primary visual
tokens in their actual row-major order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


COORDINATE_COUNT = 1000
HIDDEN_SIZE = 2048
KEY_SIZE = 64
ROLE_COUNT = 4
ROLE_NAMES = ("x1", "y1", "x2", "y2")
READOUT_SLOT_CHUNK = 8


def grid_cell_addresses(
    height: int,
    width: int,
    *,
    frame: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return row-major ``(x, y)`` cell centers in the supplied frame."""

    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ValueError("grid dimensions must be positive integers")
    if len(frame) != 4 or not all(torch.isfinite(torch.tensor(value)) for value in frame):
        raise ValueError("grid frame must contain four finite numbers")
    x0, x1, y0, y1 = (float(value) for value in frame)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("grid frame must have positive width and height")
    columns = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    rows = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack(
        (x0 + (x1 - x0) * xx.reshape(-1), y0 + (y1 - y0) * yy.reshape(-1)),
        dim=-1,
    )


def address_permutation(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the fixed wrong-address control permutation for one grid shape."""

    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ValueError("grid dimensions must be positive integers")
    count = height * width
    if count < 2:
        raise ValueError("a wrong-address control needs at least two visual tokens")
    # A one-cell cyclic shift is deterministic, shape-specific in length, and
    # never identity for a valid multi-cell grid.
    return torch.roll(torch.arange(count, device=device), shifts=1, dims=0)


def coordinate_role_from_prefix(
    prefix_token_ids: Sequence[int] | torch.Tensor,
    *,
    coordinate_token_ids: Sequence[int] | torch.Tensor,
    object_ref_start_id: int,
    object_ref_end_id: int,
    box_start_id: int,
    box_end_id: int,
    eos_token_id: int | None = None,
    commit_token_id: int | None = None,
) -> int:
    """Return the admitted ``x1,y1,x2,y2`` role, or ``-1``.

    This is a prefix parser for the maintained closed object-box grammar.  The
    supplied continuation must start at an object row; wrappers and the four
    coordinate positions are strict.  A malformed prefix is unadmitted.
    """

    coords = tuple(int(value) for value in coordinate_token_ids)
    if len(coords) != COORDINATE_COUNT or len(set(coords)) != COORDINATE_COUNT:
        raise ValueError("coordinate_token_ids must contain 1,000 unique IDs")
    coord_set = set(coords)
    if isinstance(prefix_token_ids, torch.Tensor):
        if prefix_token_ids.ndim != 1:
            raise ValueError("prefix_token_ids must be one-dimensional")
        prefix = [int(value) for value in prefix_token_ids.detach().cpu().tolist()]
    else:
        prefix = [int(value) for value in prefix_token_ids]

    DESCRIPTION, AFTER_DESCRIPTION, COORDS, AFTER_COORDS, OUTSIDE, TERMINAL = range(6)
    state = OUTSIDE
    seen_row = False
    coord_count = 0
    description_nonempty = False
    for token in prefix:
        if state == OUTSIDE:
            if token == object_ref_start_id:
                state, seen_row, coord_count = DESCRIPTION, True, 0
                description_nonempty = False
            elif seen_row:
                if token in (eos_token_id, commit_token_id):
                    state = TERMINAL
                    continue
                # A completed row may only be followed by another row opener.
                return -1
            else:
                # The caller supplies the assistant continuation, not the prompt.
                return -1
            continue
        if state == TERMINAL:
            return -1
        if state == DESCRIPTION:
            if token == object_ref_end_id:
                if not description_nonempty:
                    return -1
                state = AFTER_DESCRIPTION
            elif token in {
                object_ref_start_id,
                box_start_id,
                box_end_id,
                eos_token_id,
                commit_token_id,
            } or token in coord_set:
                return -1
            else:
                description_nonempty = True
            continue
        if state == AFTER_DESCRIPTION:
            if token != box_start_id:
                return -1
            state = COORDS
            coord_count = 0
            continue
        if state == COORDS:
            if token in coord_set and coord_count < 4:
                coord_count += 1
                if coord_count == 4:
                    state = AFTER_COORDS
                continue
            if token == box_end_id and coord_count == 4:
                state = OUTSIDE
                continue
            return -1
        if state == AFTER_COORDS:
            if token != box_end_id:
                return -1
            state = OUTSIDE
            continue

    if state == COORDS and coord_count < 4:
        return coord_count
    return -1


def roles_from_prefixes(
    prefixes: Sequence[Sequence[int] | torch.Tensor],
    **grammar: Any,
) -> torch.Tensor:
    """Parse several causal prefixes into ``-1`` or the four role indices."""

    return torch.tensor(
        [coordinate_role_from_prefix(prefix, **grammar) for prefix in prefixes],
        dtype=torch.long,
    )


class CoordinateAddressReadout(nn.Module):
    """Trainable coarse address assistance for a frozen coordinate head."""

    def __init__(
        self,
        coordinate_ids: Sequence[int] | torch.Tensor,
        *,
        permuted: bool = False,
        hidden_size: int = HIDDEN_SIZE,
        key_size: int = KEY_SIZE,
    ) -> None:
        super().__init__()
        ids = tuple(int(value) for value in coordinate_ids)
        if len(ids) != COORDINATE_COUNT or len(set(ids)) != COORDINATE_COUNT:
            raise ValueError("coordinate_ids must contain 1,000 unique IDs")
        if hidden_size != HIDDEN_SIZE or key_size != KEY_SIZE:
            raise ValueError("the pilot sidecar is fixed at 2048 -> 64")
        self.register_buffer("coordinate_ids", torch.tensor(ids, dtype=torch.long))
        self.permuted = bool(permuted)
        self.q_proj = nn.Linear(hidden_size, key_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, key_size, bias=False)
        self.role_embeddings = nn.Embedding(ROLE_COUNT, key_size)
        self.gain = nn.Parameter(torch.zeros((), dtype=torch.float32))
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=hidden_size**-0.5)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=hidden_size**-0.5)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=key_size**-0.5)

    def role_from_prefix(self, prefix_token_ids: Sequence[int] | torch.Tensor, **grammar: Any) -> int:
        return coordinate_role_from_prefix(
            prefix_token_ids,
            coordinate_token_ids=self.coordinate_ids,
            **grammar,
        )

    def parse_roles(
        self,
        prefixes: Sequence[Sequence[int] | torch.Tensor],
        **grammar: Any,
    ) -> torch.Tensor:
        return roles_from_prefixes(prefixes, coordinate_token_ids=self.coordinate_ids, **grammar)

    def forward(
        self,
        native_logits: torch.Tensor,
        hidden: torch.Tensor,
        visual_tokens: torch.Tensor,
        roles: torch.Tensor,
        grid_height: int,
        grid_width: int,
        *,
        addresses: torch.Tensor | None = None,
        axis_cell_widths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the sidecar to ``native_logits`` at admitted coordinate slots."""

        if native_logits.ndim != 2 or hidden.ndim != 2 or visual_tokens.ndim != 2:
            raise ValueError("native_logits, hidden, and visual_tokens must be two-dimensional")
        slots, vocab = native_logits.shape
        if hidden.shape != (slots, HIDDEN_SIZE):
            raise ValueError(f"hidden must have shape ({slots}, {HIDDEN_SIZE})")
        if visual_tokens.shape != (grid_height * grid_width, HIDDEN_SIZE):
            raise ValueError("visual_tokens must be row-major and match the rectangular grid")
        if roles.ndim != 1 or roles.shape[0] != slots or roles.dtype not in (torch.int32, torch.int64):
            raise ValueError("roles must be an integer vector aligned with native_logits")
        if bool((roles < -1).any()) or bool((roles >= ROLE_COUNT).any()):
            raise ValueError("roles must be -1 or one of the four coordinate roles")
        if int(self.coordinate_ids.max()) >= vocab:
            raise ValueError("coordinate IDs exceed native vocabulary")
        active = roles >= 0
        if not bool(active.any()):
            return native_logits.clone()

        coordinate_logits = native_logits.index_select(-1, self.coordinate_ids)
        conserved = self.adjust_coordinate_logits(
            coordinate_logits,
            hidden,
            visual_tokens,
            roles,
            grid_height,
            grid_width,
            addresses=addresses,
            axis_cell_widths=axis_cell_widths,
        )
        output = native_logits.clone()
        return output.scatter(
            -1,
            self.coordinate_ids.expand(slots, -1),
            conserved.to(native_logits.dtype),
        )

    def adjust_coordinate_logits(
        self,
        native_coordinate_logits: torch.Tensor,
        hidden: torch.Tensor,
        visual_tokens: torch.Tensor,
        roles: torch.Tensor,
        grid_height: int,
        grid_width: int,
        *,
        addresses: torch.Tensor | None = None,
        axis_cell_widths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return conserved coordinate-family logits without a full-vocab tensor.

        The returned family has the same logsumexp as ``native_coordinate_logits``;
        a caller holding the native full-vocabulary logsumexp can therefore place
        this family back into a full-vocabulary CE calculation without caching the
        full output-head matrix.
        """

        if native_coordinate_logits.ndim != 2 or native_coordinate_logits.shape[1] != COORDINATE_COUNT:
            raise ValueError("native_coordinate_logits must have shape (slots, 1000)")
        slots = native_coordinate_logits.shape[0]
        if hidden.shape != (slots, HIDDEN_SIZE):
            raise ValueError(f"hidden must have shape ({slots}, {HIDDEN_SIZE})")
        if visual_tokens.shape != (grid_height * grid_width, HIDDEN_SIZE):
            raise ValueError("visual_tokens must be row-major and match the rectangular grid")
        if roles.ndim != 1 or roles.shape[0] != slots or roles.dtype not in (torch.int32, torch.int64):
            raise ValueError("roles must be an integer vector aligned with native_coordinate_logits")
        if bool((roles < -1).any()) or bool((roles >= ROLE_COUNT).any()):
            raise ValueError("roles must be -1 or one of the four coordinate roles")
        active = roles >= 0
        if not bool(active.any()):
            return native_coordinate_logits.clone()

        work_dtype = self.q_proj.weight.dtype
        hidden_work = hidden.to(dtype=work_dtype)
        visual_work = visual_tokens.to(dtype=work_dtype)
        q = self.q_proj(hidden_work).float()
        k = self.k_proj(visual_work).float()
        role = self.role_embeddings(roles.clamp_min(0)).float()
        scores = torch.einsum("nd,md->nm", q + role, k) / (KEY_SIZE**0.5)
        log_attention = torch.log_softmax(scores, dim=-1)

        if addresses is None:
            addr = grid_cell_addresses(
                grid_height,
                grid_width,
                device=visual_tokens.device,
                dtype=torch.float32,
            )
        else:
            if addresses.shape != (visual_tokens.shape[0], 2):
                raise ValueError("addresses must have shape (visual_tokens, 2)")
            addr = addresses.to(device=visual_tokens.device, dtype=torch.float32)
            if not bool(torch.isfinite(addr).all()):
                raise ValueError("addresses must be finite")
        if self.permuted:
            addr = addr.index_select(0, address_permutation(grid_height, grid_width, device=addr.device))

        if axis_cell_widths is None:
            widths = torch.tensor((1.0 / grid_width, 1.0 / grid_height), device=addr.device)
        else:
            widths = axis_cell_widths.to(device=addr.device, dtype=torch.float32).reshape(-1)
            if widths.numel() != 2 or not bool(torch.isfinite(widths).all()) or bool((widths <= 0).any()):
                raise ValueError("axis_cell_widths must contain two positive finite values")
        bins = torch.arange(COORDINATE_COUNT, device=addr.device, dtype=torch.float32) / COORDINATE_COUNT
        distances = (bins[None, :, None] - addr[:, None, :]) / (0.5 * widths[None, None, :])
        log_kernel = torch.log_softmax(-0.5 * distances.square(), dim=1)
        axis = torch.tensor((0, 1, 0, 1), device=roles.device)[roles.clamp_min(0)]
        axis = axis.to(device=log_kernel.device)
        # Bound each temporary selected-kernel tensor at 8*M*1000 elements;
        # autograd may retain chunk outputs until backward.
        chunks = []
        for start in range(0, slots, READOUT_SLOT_CHUNK):
            stop = min(start + READOUT_SLOT_CHUNK, slots)
            selected_kernel = log_kernel[:, :, axis[start:stop]].permute(2, 0, 1)
            chunks.append(torch.logsumexp(log_attention[start:stop, :, None] + selected_kernel, dim=1))
        log_readout = torch.cat(chunks, dim=0)

        native = native_coordinate_logits.float()
        delta = self.gain.float() * log_readout
        native_lse = torch.logsumexp(native, dim=-1, keepdim=True)
        correction = delta - (torch.logsumexp(native + delta, dim=-1, keepdim=True) - native_lse)
        conserved = native + correction
        return torch.where(active[:, None], conserved, native)


__all__ = [
    "COORDINATE_COUNT",
    "ROLE_NAMES",
    "CoordinateAddressReadout",
    "address_permutation",
    "coordinate_role_from_prefix",
    "grid_cell_addresses",
    "roles_from_prefixes",
]

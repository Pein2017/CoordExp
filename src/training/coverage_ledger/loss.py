"""Coverage-ledger auxiliary target construction and loss math."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.metrics.events import MetricEvent
from src.training.coverage_ledger.head import CoverageLedgerHead
from src.training.coverage_ledger.sidecars import CoverageLedgerSidecar
from src.training.objectives.types import DEFAULT_PRECISION_POLICY


@dataclass(frozen=True, slots=True)
class CoverageLedgerLossConfig:
    """Scalar controls for coverage-ledger auxiliary loss math."""

    coverage_weight: float
    region_anchor_weight: float
    temperature: float
    pos_weight: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coverage_weight",
            _finite_float(
                self.coverage_weight,
                field_name="coverage_weight",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "region_anchor_weight",
            _finite_float(
                self.region_anchor_weight,
                field_name="region_anchor_weight",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "temperature",
            _finite_float(self.temperature, field_name="temperature", minimum=0.0),
        )
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        object.__setattr__(
            self,
            "pos_weight",
            _finite_float(self.pos_weight, field_name="pos_weight", minimum=0.0),
        )
        if self.pos_weight <= 0.0:
            raise ValueError("pos_weight must be positive")


@dataclass(frozen=True, slots=True)
class CoverageLedgerTargets:
    """Exact token-position targets derived from one coverage-ledger sidecar."""

    coverage_state_positions: tuple[int, ...]
    region_anchor_positions: tuple[int, ...]
    region_anchor_object_indices: tuple[int, ...]
    coverage_targets: torch.Tensor


@dataclass(frozen=True, slots=True)
class CoverageLedgerDebugRows:
    """Intermediate tensors and counts for tests and diagnostic inspection."""

    coverage_state_positions: tuple[int, ...]
    region_anchor_positions: tuple[int, ...]
    region_anchor_object_indices: tuple[int, ...]
    coverage_targets: torch.Tensor
    coverage_logits: torch.Tensor
    region_anchor_targets: torch.Tensor
    region_anchor_logits: torch.Tensor
    object_count: int
    coverage_state_count: int
    coverage_pair_count: int
    region_anchor_pair_count: int


@dataclass(frozen=True, slots=True)
class CoverageLedgerLossResult:
    """Typed result for coverage-ledger auxiliary objective math."""

    total_loss: torch.Tensor
    coverage_loss: torch.Tensor
    region_anchor_loss: torch.Tensor
    weighted_loss: torch.Tensor
    coverage_weight: float
    region_anchor_weight: float
    metric_events: tuple[MetricEvent, ...]
    debug_rows: CoverageLedgerDebugRows


def build_coverage_ledger_targets(
    sidecar: CoverageLedgerSidecar,
    *,
    device: torch.device | str,
) -> CoverageLedgerTargets:
    """Build coverage and region-anchor targets without causal row shifting."""

    _validate_sidecar(sidecar)
    object_count = len(sidecar.object_entries)
    coverage_positions = (sidecar.prompt_end_position,) + tuple(
        entry.box_end_position for entry in sidecar.object_entries
    )
    anchor_positions = tuple(
        entry.box_start_position for entry in sidecar.object_entries
    )
    anchor_indices = tuple(range(object_count))

    rows: list[list[float]] = [[0.0 for _ in range(object_count)]]
    for row_index in range(object_count):
        rows.append(
            [
                1.0 if object_index <= row_index else 0.0
                for object_index in range(object_count)
            ]
        )

    targets = torch.tensor(rows, dtype=torch.float32, device=device)
    return CoverageLedgerTargets(
        coverage_state_positions=coverage_positions,
        region_anchor_positions=anchor_positions,
        region_anchor_object_indices=anchor_indices,
        coverage_targets=targets,
    )


def compute_coverage_ledger_loss(
    *,
    head: CoverageLedgerHead,
    final_hidden_states: torch.Tensor,
    pooled_visual_object_embeddings: torch.Tensor,
    sidecar: CoverageLedgerSidecar,
    config: CoverageLedgerLossConfig,
    sample_id_to_batch_index: Mapping[str, int] | None = None,
) -> CoverageLedgerLossResult:
    """Compute cumulative coverage BCE and one-vs-all row-object binding loss."""

    if not isinstance(head, CoverageLedgerHead):
        raise TypeError("head must be a CoverageLedgerHead")
    if not isinstance(config, CoverageLedgerLossConfig):
        raise TypeError("config must be a CoverageLedgerLossConfig")
    if not isinstance(final_hidden_states, torch.Tensor):
        raise TypeError("final_hidden_states must be a torch.Tensor")
    if not isinstance(pooled_visual_object_embeddings, torch.Tensor):
        raise TypeError("pooled_visual_object_embeddings must be a torch.Tensor")
    _validate_sidecar(sidecar)

    targets = build_coverage_ledger_targets(
        sidecar,
        device=final_hidden_states.device,
    )
    object_count = len(sidecar.object_entries)
    _validate_shapes(
        head=head,
        final_hidden_states=final_hidden_states,
        pooled_visual_object_embeddings=pooled_visual_object_embeddings,
        object_count=object_count,
    )
    _validate_positions(
        final_hidden_states=final_hidden_states,
        sidecar=sidecar,
        targets=targets,
        sample_id_to_batch_index=sample_id_to_batch_index,
    )
    normalize_eps = _positive_finite_float(
        float(head.normalize_eps),
        field_name="head.normalize_eps",
    )

    with DEFAULT_PRECISION_POLICY.context(final_hidden_states):
        coverage_hidden = _gather_hidden_states(
            final_hidden_states,
            targets.coverage_state_positions,
            sidecar=sidecar,
            sample_id_to_batch_index=sample_id_to_batch_index,
        )
        anchor_hidden = _gather_hidden_states(
            final_hidden_states,
            targets.region_anchor_positions,
            sidecar=sidecar,
            sample_id_to_batch_index=sample_id_to_batch_index,
        )
        object_embeddings = pooled_visual_object_embeddings.detach()

        coverage_state_proj = _linear_float32(
            coverage_hidden,
            head.state_projection,
        )
        anchor_state_proj = _linear_float32(
            anchor_hidden,
            head.region_anchor_state_projection,
        )
        object_proj = _linear_float32(object_embeddings, head.object_projection)

        coverage_state_norm = F.normalize(
            coverage_state_proj,
            p=2.0,
            dim=-1,
            eps=normalize_eps,
        )
        anchor_state_norm = F.normalize(
            anchor_state_proj,
            p=2.0,
            dim=-1,
            eps=normalize_eps,
        )
        object_norm = F.normalize(object_proj, p=2.0, dim=-1, eps=normalize_eps)

        coverage_logits = coverage_state_norm @ object_norm.transpose(0, 1)
        coverage_logits = coverage_logits / float(config.temperature)
        _raise_non_finite(coverage_logits, name="coverage ledger coverage logits")

        coverage_targets = targets.coverage_targets.to(
            device=coverage_logits.device,
            dtype=torch.float32,
        )
        pos_weight = coverage_logits.new_tensor(float(config.pos_weight))
        coverage_loss = F.binary_cross_entropy_with_logits(
            coverage_logits.reshape(-1),
            coverage_targets.reshape(-1),
            pos_weight=pos_weight,
        ).to(dtype=torch.float32)
        _raise_non_finite(coverage_loss, name="coverage ledger coverage loss")

        anchor_indices = torch.tensor(
            targets.region_anchor_object_indices,
            dtype=torch.long,
            device=object_norm.device,
        )
        region_anchor_logits = anchor_state_norm @ object_norm.transpose(0, 1)
        region_anchor_logits = region_anchor_logits / float(config.temperature)
        _raise_non_finite(
            region_anchor_logits,
            name="coverage ledger region-anchor logits",
        )
        region_anchor_targets = torch.zeros(
            (anchor_state_norm.shape[0], object_count),
            dtype=torch.float32,
            device=region_anchor_logits.device,
        )
        row_indices = torch.arange(
            anchor_state_norm.shape[0],
            dtype=torch.long,
            device=region_anchor_logits.device,
        )
        region_anchor_targets[row_indices, anchor_indices] = 1.0
        region_anchor_loss = F.binary_cross_entropy_with_logits(
            region_anchor_logits.reshape(-1),
            region_anchor_targets.reshape(-1),
        )
        region_anchor_loss = region_anchor_loss.to(dtype=torch.float32)
        _raise_non_finite(
            region_anchor_loss,
            name="coverage ledger region-anchor loss",
        )

    weighted_loss = (
        coverage_loss * float(config.coverage_weight)
        + region_anchor_loss * float(config.region_anchor_weight)
    ).to(dtype=torch.float32)
    _raise_non_finite(weighted_loss, name="coverage ledger weighted loss")
    total_loss = weighted_loss

    debug_rows = CoverageLedgerDebugRows(
        coverage_state_positions=targets.coverage_state_positions,
        region_anchor_positions=targets.region_anchor_positions,
        region_anchor_object_indices=targets.region_anchor_object_indices,
        coverage_targets=coverage_targets.detach(),
        coverage_logits=coverage_logits.detach(),
        region_anchor_targets=region_anchor_targets.detach(),
        region_anchor_logits=region_anchor_logits.detach(),
        object_count=object_count,
        coverage_state_count=int(coverage_targets.shape[0]),
        coverage_pair_count=int(coverage_targets.numel()),
        region_anchor_pair_count=int(region_anchor_targets.numel()),
    )

    return CoverageLedgerLossResult(
        total_loss=total_loss,
        coverage_loss=coverage_loss,
        region_anchor_loss=region_anchor_loss,
        weighted_loss=weighted_loss,
        coverage_weight=float(config.coverage_weight),
        region_anchor_weight=float(config.region_anchor_weight),
        metric_events=(),
        debug_rows=debug_rows,
    )


def _linear_float32(tensor: torch.Tensor, module: torch.nn.Linear) -> torch.Tensor:
    bias = module.bias.float() if module.bias is not None else None
    return F.linear(tensor.float(), module.weight.float(), bias)


def _gather_hidden_states(
    final_hidden_states: torch.Tensor,
    positions: tuple[int, ...],
    *,
    sidecar: CoverageLedgerSidecar,
    sample_id_to_batch_index: Mapping[str, int] | None,
) -> torch.Tensor:
    indices = torch.tensor(
        positions,
        dtype=torch.long,
        device=final_hidden_states.device,
    )
    if final_hidden_states.ndim == 2:
        return final_hidden_states.index_select(dim=0, index=indices)

    batch_index = _resolve_batch_index(
        final_hidden_states,
        sidecar=sidecar,
        sample_id_to_batch_index=sample_id_to_batch_index,
    )
    return final_hidden_states[batch_index].index_select(dim=0, index=indices)


def _resolve_batch_index(
    final_hidden_states: torch.Tensor,
    *,
    sidecar: CoverageLedgerSidecar,
    sample_id_to_batch_index: Mapping[str, int] | None,
) -> int:
    batch_size = int(final_hidden_states.shape[0])
    if sample_id_to_batch_index is None:
        if batch_size == 1:
            return 0
        raise ValueError(
            "sample_id_to_batch_index is required for batched final_hidden_states"
        )
    if not isinstance(sample_id_to_batch_index, Mapping):
        raise TypeError("sample_id_to_batch_index must be a mapping")
    try:
        batch_index = sample_id_to_batch_index[sidecar.sample_id]
    except KeyError as exc:
        raise ValueError(
            f"sample_id {sidecar.sample_id!r} has no hidden-state batch index"
        ) from exc
    if type(batch_index) is not int:
        raise TypeError("batch indices must be integers")
    if batch_index < 0 or batch_index >= batch_size:
        raise ValueError(
            f"batch index {batch_index} for sample_id {sidecar.sample_id!r} "
            f"is outside batch size {batch_size}"
        )
    return int(batch_index)


def _validate_sidecar(sidecar: CoverageLedgerSidecar) -> None:
    if type(sidecar) is not CoverageLedgerSidecar:
        raise TypeError("sidecar must be a CoverageLedgerSidecar")
    if len(sidecar.object_entries) == 0:
        raise ValueError("coverage ledger requires at least one object")


def _validate_shapes(
    *,
    head: CoverageLedgerHead,
    final_hidden_states: torch.Tensor,
    pooled_visual_object_embeddings: torch.Tensor,
    object_count: int,
) -> None:
    if final_hidden_states.ndim not in {2, 3}:
        raise ValueError(
            "final_hidden_states must have shape [time, hidden] or "
            "[batch, time, hidden]"
        )
    if int(final_hidden_states.shape[-1]) != int(head.state_projection.in_features):
        raise ValueError("final_hidden_states hidden size does not match ledger head")
    if int(head.region_anchor_state_projection.in_features) != int(
        head.state_projection.in_features
    ):
        raise ValueError("coverage ledger state projection dimensions do not match")
    if int(head.region_anchor_state_projection.out_features) != int(
        head.state_projection.out_features
    ):
        raise ValueError("coverage ledger projection dimensions do not match")
    if int(head.object_projection.out_features) != int(
        head.state_projection.out_features
    ):
        raise ValueError("coverage ledger object projection dimension does not match")

    if pooled_visual_object_embeddings.ndim != 2:
        raise ValueError(
            "pooled_visual_object_embeddings must have shape [objects, visual_dim]"
        )
    if int(pooled_visual_object_embeddings.shape[0]) != object_count:
        raise ValueError(
            "pooled_visual_object_embeddings object count does not match sidecar"
        )
    if int(pooled_visual_object_embeddings.shape[1]) != int(
        head.object_projection.in_features
    ):
        raise ValueError(
            "pooled_visual_object_embeddings visual_dim does not match ledger head"
        )


def _validate_positions(
    *,
    final_hidden_states: torch.Tensor,
    sidecar: CoverageLedgerSidecar,
    targets: CoverageLedgerTargets,
    sample_id_to_batch_index: Mapping[str, int] | None,
) -> None:
    if final_hidden_states.ndim == 3:
        _resolve_batch_index(
            final_hidden_states,
            sidecar=sidecar,
            sample_id_to_batch_index=sample_id_to_batch_index,
        )
    time_steps = int(final_hidden_states.shape[-2])
    for position in (
        targets.coverage_state_positions + targets.region_anchor_positions
    ):
        if position >= time_steps:
            raise ValueError(
                "coverage ledger hidden-state position is outside final_hidden_states: "
                f"position={position}, time_steps={time_steps}"
            )


def _finite_float(value: object, *, field_name: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite numeric scalar")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    if parsed < float(minimum):
        raise ValueError(f"{field_name} must be >= {minimum}")
    return parsed


def _positive_finite_float(value: object, *, field_name: str) -> float:
    parsed = _finite_float(value, field_name=field_name, minimum=0.0)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _raise_non_finite(tensor: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
        raise FloatingPointError(f"{name} contains non-finite values")


__all__ = [
    "CoverageLedgerDebugRows",
    "CoverageLedgerLossConfig",
    "CoverageLedgerLossResult",
    "CoverageLedgerTargets",
    "build_coverage_ledger_targets",
    "compute_coverage_ledger_loss",
]

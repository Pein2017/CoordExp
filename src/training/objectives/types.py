"""Typed objective runner contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from src.metrics.events import MetricEvent
from src.training.supervision.distributions import (
    TargetObjectiveId,
    validate_target_objective_id,
)
from src.training.supervision.spans import SupervisionSpan


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """Requested objective and its local configuration.

    :param objective_id: Closed semantic objective identifier.
    :param weight: Objective-local contribution weight in the runner sum.
    :param config: Objective-local scalar or typed configuration values.
    """

    objective_id: TargetObjectiveId
    weight: float = 1.0
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze the objective request."""

        # validate the closed objective identity.
        object.__setattr__(
            self,
            "objective_id",
            validate_target_objective_id(self.objective_id),
        )

        # normalize the objective contribution weight.
        weight = _coerce_non_negative_float(self.weight, field_name="objective weight")
        object.__setattr__(self, "weight", weight)

        # freeze caller-provided configuration without interpreting it globally.
        if not isinstance(self.config, Mapping):
            raise TypeError("objective config must be a mapping")
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))


@dataclass(frozen=True, slots=True)
class ObjectivePrecisionPolicy:
    """Precision policy for sensitive objective math.

    :param math_dtype: Floating dtype used inside objective reductions.
    :param disable_autocast: Whether objective math disables active autocast.
    """

    math_dtype: torch.dtype = torch.float32
    disable_autocast: bool = True

    def context(self, tensor: torch.Tensor):
        """Return an autocast-disabled context for the tensor device."""

        # preserve non-autocast devices while disabling CPU/CUDA autocast.
        if self.disable_autocast and tensor.device.type in {"cpu", "cuda"}:
            return torch.autocast(device_type=tensor.device.type, enabled=False)

        return nullcontext()

    def cast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the tensor in objective math precision."""

        return tensor.to(dtype=self.math_dtype)


DEFAULT_PRECISION_POLICY = ObjectivePrecisionPolicy()


@dataclass(frozen=True, slots=True)
class LabelLogitRow:
    """Resolved causal logit row for one target-token label position.

    :param sample_id: Stable sample identifier from the supervision span.
    :param label_position: Target-token label position.
    :param row_index: Causal logit row that predicts the label position.
    :param batch_index: Optional batch index for 3D logits.
    """

    sample_id: str
    label_position: int
    row_index: int
    batch_index: int | None = None


@dataclass(frozen=True, slots=True)
class LabelLogitRowMap:
    """Default causal label-position to logit-row mapper."""

    time_steps: int
    vocab_size: int
    batch_size: int | None = None
    sample_id_to_batch_index: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze direct row-map construction."""

        # normalize static logits dimensions.
        time_steps = _coerce_positive_int(self.time_steps, field_name="time_steps")
        vocab_size = _coerce_positive_int(self.vocab_size, field_name="vocab_size")
        object.__setattr__(self, "time_steps", time_steps)
        object.__setattr__(self, "vocab_size", vocab_size)

        # normalize optional batch ownership exactly as 3D logits require it.
        if self.batch_size is None:
            if len(self.sample_id_to_batch_index) != 0:
                raise ValueError(
                    "2D label row maps cannot define sample_id_to_batch_index"
                )
            object.__setattr__(self, "sample_id_to_batch_index", MappingProxyType({}))
            return

        batch_size = _coerce_positive_int(self.batch_size, field_name="batch_size")
        if len(self.sample_id_to_batch_index) == 0:
            raise ValueError("sample_id_to_batch_index is required for 3D row maps")
        mapping = _normalize_sample_id_to_batch_index(
            self.sample_id_to_batch_index,
            batch_size=batch_size,
        )
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "sample_id_to_batch_index", mapping)

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        *,
        sample_id_to_batch_index: Mapping[str, int] | None = None,
    ) -> "LabelLogitRowMap":
        """Create a row map for 2D or 3D training logits."""

        # identify the supported logits layout.
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if logits.ndim == 2:
            return cls(
                time_steps=int(logits.shape[0]),
                vocab_size=int(logits.shape[1]),
            )
        if logits.ndim != 3:
            raise ValueError(
                "logits must have shape [time, vocab] or [batch, time, vocab]; "
                f"got {tuple(logits.shape)}"
            )

        # validate explicit sample-to-batch ownership for batched logits.
        if sample_id_to_batch_index is None:
            raise ValueError(
                "sample_id_to_batch_index is required for 3D logits "
                "[batch, time, vocab]"
            )
        batch_size = int(logits.shape[0])
        mapping = _normalize_sample_id_to_batch_index(
            sample_id_to_batch_index,
            batch_size=batch_size,
        )

        return cls(
            time_steps=int(logits.shape[1]),
            vocab_size=int(logits.shape[2]),
            batch_size=batch_size,
            sample_id_to_batch_index=mapping,
        )

    def resolve(
        self,
        span: SupervisionSpan,
        label_position: int,
    ) -> LabelLogitRow:
        """Resolve one target-token label position to a causal logit row."""

        # map target-token position p to causal logit row p - 1.
        if type(label_position) is not int:
            raise TypeError("label position must be an integer")
        if label_position <= 0:
            raise ValueError(
                "label position 0 is not valid for causal objective rows; "
                "expected p > 0"
            )
        row_index = int(label_position) - 1
        if row_index < 0 or row_index >= int(self.time_steps):
            raise ValueError(
                "label position maps outside logits rows: "
                f"label_position={label_position}, row={row_index}, "
                f"time_steps={self.time_steps}"
            )

        # resolve sample ownership for batched logits.
        batch_index: int | None = None
        if self.batch_size is not None:
            try:
                batch_index = self.sample_id_to_batch_index[span.sample_id]
            except KeyError as exc:
                raise ValueError(
                    f"sample_id {span.sample_id!r} has no batch index"
                ) from exc

        return LabelLogitRow(
            sample_id=span.sample_id,
            label_position=label_position,
            row_index=row_index,
            batch_index=batch_index,
        )

    def resolve_span(self, span: SupervisionSpan) -> tuple[LabelLogitRow, ...]:
        """Resolve all label positions carried by a supervision span."""

        return tuple(
            self.resolve(span, label_position)
            for label_position in span.label_positions
        )

    def gather(
        self,
        logits: torch.Tensor,
        rows: Sequence[LabelLogitRow],
    ) -> torch.Tensor:
        """Gather resolved rows from 2D or 3D logits."""

        # materialize an empty row tensor with the original dtype/device.
        if len(rows) == 0:
            return logits.new_zeros((0, int(self.vocab_size)))

        # gather rows without reinterpreting label positions in objective modules.
        if logits.ndim == 2:
            row_indices = torch.tensor(
                [row.row_index for row in rows],
                device=logits.device,
                dtype=torch.long,
            )
            return logits.index_select(dim=0, index=row_indices)

        gathered = []
        for row in rows:
            if row.batch_index is None:
                raise ValueError("3D logits require resolved batch indices")
            gathered.append(logits[row.batch_index, row.row_index])

        return torch.stack(gathered, dim=0)

    def validate_logits(self, logits: torch.Tensor) -> None:
        """Validate that this row map still matches the provided logits."""

        # reject stale maps before any objective gathers rows.
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if logits.ndim == 2:
            if self.batch_size is not None:
                raise ValueError("3D label row map cannot be used with 2D logits")
            if int(logits.shape[0]) != self.time_steps:
                raise ValueError("label row map time_steps do not match logits")
            if int(logits.shape[1]) != self.vocab_size:
                raise ValueError("label row map vocab_size does not match logits")
            return
        if logits.ndim == 3:
            if self.batch_size is None:
                raise ValueError("2D label row map cannot be used with 3D logits")
            if int(logits.shape[0]) != self.batch_size:
                raise ValueError("label row map batch_size does not match logits")
            if int(logits.shape[1]) != self.time_steps:
                raise ValueError("label row map time_steps do not match logits")
            if int(logits.shape[2]) != self.vocab_size:
                raise ValueError("label row map vocab_size does not match logits")
            return

        raise ValueError(
            "logits must have shape [time, vocab] or [batch, time, vocab]; "
            f"got {tuple(logits.shape)}"
        )


@dataclass(frozen=True, slots=True)
class ResolvedObjectiveSpan:
    """Supervision span with resolved rows and gathered logits."""

    span: SupervisionSpan
    rows: tuple[LabelLogitRow, ...]
    logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    """Objective-local normalized loss and diagnostics."""

    objective_id: TargetObjectiveId
    loss: torch.Tensor
    weighted_loss: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor
    span_count: int
    weight: float
    precision_policy: ObjectivePrecisionPolicy
    metric_events: tuple[MetricEvent, ...] = ()
    state: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def zero(
        cls,
        *,
        objective_id: TargetObjectiveId,
        weight: float,
        logits: torch.Tensor,
    ) -> "ObjectiveResult":
        """Return a differentiable zero result for an objective with no spans."""

        # anchor the zero in the logits graph while keeping objective dtype fp32.
        zero = logits.float().sum() * 0.0
        zero = zero.to(dtype=torch.float32)

        return cls(
            objective_id=objective_id,
            loss=zero,
            weighted_loss=zero * float(weight),
            numerator=zero,
            denominator=zero.detach(),
            span_count=0,
            weight=float(weight),
            precision_policy=DEFAULT_PRECISION_POLICY,
            metric_events=(),
            state=MappingProxyType({}),
        )


@dataclass(frozen=True, slots=True)
class ObjectiveRunResult:
    """Aggregated objective runner output."""

    loss: torch.Tensor
    objectives: Mapping[TargetObjectiveId, ObjectiveResult]
    metric_events: tuple[MetricEvent, ...] = ()
    state: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CoordinateVocabulary:
    """Typed coordinate-token vocabulary for geometry objectives."""

    token_ids: tuple[int, ...]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        *,
        require_1000_bins: bool = True,
    ) -> "CoordinateVocabulary":
        """Create a coordinate vocabulary from objective configuration."""

        # read the explicit coordinate token sequence.
        raw_token_ids = config.get("coord_token_ids")
        if isinstance(raw_token_ids, (str, bytes)) or not isinstance(
            raw_token_ids,
            Sequence,
        ):
            raise ValueError("objective requires config['coord_token_ids']")

        token_ids: list[int] = []
        for token_id in raw_token_ids:
            if type(token_id) is not int:
                raise TypeError("coord_token_ids must contain integer token ids")
            if token_id < 0:
                raise ValueError("coord_token_ids must be non-negative")
            token_ids.append(token_id)
        if require_1000_bins and len(token_ids) != 1000:
            raise ValueError(
                "box_regression coord_token_ids must contain exactly 1000 bins"
            )
        if len(token_ids) == 0:
            raise ValueError("coord_token_ids must contain at least one token id")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("coord_token_ids must be unique")

        return cls(token_ids=tuple(token_ids))

    def as_tensor(self, *, device: torch.device) -> torch.Tensor:
        """Return coordinate token ids as a long tensor."""

        return torch.tensor(self.token_ids, device=device, dtype=torch.long)


def precision_context(tensor: torch.Tensor):
    """Return the default objective precision context for a tensor."""

    return DEFAULT_PRECISION_POLICY.context(tensor)


def loss_float(tensor: torch.Tensor) -> torch.Tensor:
    """Return the tensor in objective math precision."""

    return DEFAULT_PRECISION_POLICY.cast(tensor)


def metadata_float(
    metadata: Mapping[str, object],
    key: str,
    *,
    default: float,
    minimum: float = 0.0,
) -> float:
    """Return a validated finite float from scalar metadata."""

    # use the objective default when metadata omits the scalar.
    value = metadata.get(key, default)
    return _coerce_non_negative_float(value, field_name=key, minimum=minimum)


def config_float(
    config: Mapping[str, object],
    key: str,
    *,
    default: float,
    minimum: float | None = 0.0,
) -> float:
    """Return a validated finite float from objective config."""

    # use the objective default when config omits the scalar.
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a finite numeric scalar")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite")
    if minimum is not None and parsed < float(minimum):
        raise ValueError(f"{key} must be >= {minimum}")

    return parsed


def config_tensor(
    config: Mapping[str, object],
    key: str,
    *,
    ndim: int | None = None,
) -> torch.Tensor:
    """Return a required tensor from objective config."""

    try:
        value = config[key]
    except KeyError as exc:
        raise ValueError(f"objective requires config[{key!r}]") from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"config[{key!r}] must be a torch.Tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"config[{key!r}] must have rank {ndim}")

    return value


def make_weighted_mean_event(
    *,
    key: str,
    value: torch.Tensor,
    weight: torch.Tensor,
    objective_id: str,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Return a canonical weighted-mean metric event for an objective."""

    from src.metrics.events import weighted_mean_event

    return weighted_mean_event(
        key,
        float(value.detach().cpu().item()),
        float(weight.detach().cpu().item()),
        unit="span",
        semantic_role="training_objective",
        metric_surface="training_logits",
        object_scope=objective_id,
        diagnostic_only=diagnostic_only,
    )


def make_sum_event(
    *,
    key: str,
    value: float | int,
    objective_id: str,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Return a canonical sum metric event for an objective."""

    from src.metrics.events import sum_event

    return sum_event(
        key,
        float(value),
        unit="span",
        semantic_role="training_objective",
        metric_surface="training_logits",
        object_scope=objective_id,
        diagnostic_only=diagnostic_only,
    )


def _coerce_non_negative_float(
    value: object,
    *,
    field_name: str,
    minimum: float = 0.0,
) -> float:
    """Return a finite float not smaller than ``minimum``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite numeric scalar")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    if parsed < float(minimum):
        raise ValueError(f"{field_name} must be >= {minimum}")

    return parsed


def _coerce_positive_int(value: object, *, field_name: str) -> int:
    """Return a positive integer field value."""

    if type(value) is not int:
        raise TypeError(f"{field_name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")

    return value


def _normalize_sample_id_to_batch_index(
    mapping: Mapping[str, int],
    *,
    batch_size: int,
) -> Mapping[str, int]:
    """Return a validated sample-id to batch-index mapping."""

    if not isinstance(mapping, Mapping):
        raise TypeError("sample_id_to_batch_index must be a mapping")

    normalized: dict[str, int] = {}
    for sample_id, batch_index in mapping.items():
        if type(sample_id) is not str or sample_id == "":
            raise TypeError("sample ids in sample_id_to_batch_index must be strings")
        if type(batch_index) is not int:
            raise TypeError("batch indices must be integers")
        if batch_index < 0 or batch_index >= batch_size:
            raise ValueError(
                f"batch index {batch_index} for sample_id {sample_id!r} "
                f"is outside batch size {batch_size}"
            )
        normalized[sample_id] = batch_index

    return MappingProxyType(normalized)

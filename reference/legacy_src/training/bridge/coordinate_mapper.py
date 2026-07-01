"""Bridge-level prediction-coordinate mapping abstractions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from src.training.objectives.types import LabelLogitRow, LabelLogitRowMap
from src.training.supervision.batch import SupervisionBatch


@dataclass(frozen=True, slots=True)
class PredictionCoordinateMapper:
    """Bridge-owned wrapper around causal label-to-logit row mapping.

    :param logits_shape: Full logits tensor shape observed at the bridge.
    :param time_steps: Number of prediction time steps available for objectives.
    :param vocab_size: Vocabulary dimension preserved from full logits.
    :param batch_size: Optional batch dimension for 3D logits.
    :param label_rows: Underlying objective-runner row map, when row ownership is
        resolvable.
    :param resolved_rows: Rows resolved eagerly for bridge-level validation.
    :param constructed: Explicit construction marker for tests and diagnostics.
    """

    logits_shape: tuple[int, ...]
    time_steps: int
    vocab_size: int
    batch_size: int | None
    label_rows: LabelLogitRowMap | None
    resolved_rows: tuple[LabelLogitRow, ...]
    constructed: bool = True

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        *,
        supervision: SupervisionBatch,
        sample_id_to_batch_index: Mapping[str, int] | None = None,
    ) -> "PredictionCoordinateMapper":
        """Create a mapper for 2D or 3D logits and validate supervised spans."""

        # validate the bridge inputs before delegating causal math.
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if type(supervision) is not SupervisionBatch:
            raise TypeError("supervision must be a SupervisionBatch")

        # preserve full shape metadata for downstream diagnostics.
        if logits.ndim == 2:
            label_rows = LabelLogitRowMap.from_logits(logits)
            return cls._from_label_rows(
                logits=logits,
                supervision=supervision,
                label_rows=label_rows,
            )
        if logits.ndim != 3:
            raise ValueError(
                "logits must have shape [time, vocab] or [batch, time, vocab]; "
                f"got {tuple(logits.shape)}"
            )

        # allow empty 3D batches to form graph-anchored zero losses without row ownership.
        cls._validate_positive_3d_shape(logits)
        if len(supervision.spans) == 0 and sample_id_to_batch_index is None:
            return cls(
                logits_shape=tuple(int(dim) for dim in logits.shape),
                time_steps=int(logits.shape[1]),
                vocab_size=int(logits.shape[2]),
                batch_size=int(logits.shape[0]),
                label_rows=None,
                resolved_rows=(),
            )

        # require explicit sample ownership before resolving any non-empty 3D row.
        label_rows = LabelLogitRowMap.from_logits(
            logits,
            sample_id_to_batch_index=sample_id_to_batch_index,
        )
        return cls._from_label_rows(
            logits=logits,
            supervision=supervision,
            label_rows=label_rows,
        )

    @classmethod
    def _from_label_rows(
        cls,
        *,
        logits: torch.Tensor,
        supervision: SupervisionBatch,
        label_rows: LabelLogitRowMap,
    ) -> "PredictionCoordinateMapper":
        """Create a mapper after the underlying row map has been established."""

        # validate map freshness and every supervised causal row eagerly.
        label_rows.validate_logits(logits)
        resolved: list[LabelLogitRow] = []
        for span in supervision.spans:
            resolved.extend(label_rows.resolve_span(span))

        # expose immutable bridge-level shape and row state.
        return cls(
            logits_shape=tuple(int(dim) for dim in logits.shape),
            time_steps=label_rows.time_steps,
            vocab_size=label_rows.vocab_size,
            batch_size=label_rows.batch_size,
            label_rows=label_rows,
            resolved_rows=tuple(resolved),
        )

    @staticmethod
    def _validate_positive_3d_shape(logits: torch.Tensor) -> None:
        """Reject empty dimensions before constructing an empty 3D mapper."""

        if int(logits.shape[0]) <= 0:
            raise ValueError("batch_size must be positive")
        if int(logits.shape[1]) <= 0:
            raise ValueError("time_steps must be positive")
        if int(logits.shape[2]) <= 0:
            raise ValueError("vocab_size must be positive")

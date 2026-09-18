"""Expected-coordinate raw-axis validity hinge for complete xyxy boxes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.common.errors import LossContractError
from src.losses.context import LossContext


DEFAULT_COORDINATE_BINS = 1000
DEFAULT_RAW_AXIS_MARGIN = 1.0 / 999.0


@dataclass(frozen=True)
class RawAxisValidityHingeResult:
    segment_losses: torch.Tensor
    eligible_segment_count: int
    skipped_segment_count: int
    complete_box_count: int
    incomplete_box_count: int
    coordinate_atom_count: int
    zero_box_segment_count: int

    def __post_init__(self) -> None:
        if self.segment_losses.ndim != 1:
            raise LossContractError(
                "raw-axis validity segment losses must be one-dimensional",
                code="loss.raw_axis_validity_hinge_segment_shape",
                context={"shape": list(self.segment_losses.shape)},
            )
        if int(self.segment_losses.numel()) != int(self.eligible_segment_count):
            raise LossContractError(
                "raw-axis validity segment losses must align with eligible segments",
                code="loss.raw_axis_validity_hinge_segment_count",
                context={
                    "loss_count": int(self.segment_losses.numel()),
                    "eligible_segment_count": int(self.eligible_segment_count),
                },
            )

    def diagnostics(self) -> dict[str, int]:
        return {
            "eligible_segment_count": self.eligible_segment_count,
            "skipped_segment_count": self.skipped_segment_count,
            "complete_box_count": self.complete_box_count,
            "incomplete_box_count": self.incomplete_box_count,
            "coordinate_atom_count": self.coordinate_atom_count,
            "zero_box_segment_count": self.zero_box_segment_count,
        }


@dataclass(frozen=True)
class RawAxisValidityHingeLoss:
    margin: float = DEFAULT_RAW_AXIS_MARGIN
    name: str = "raw_axis_validity_hinge"

    def __post_init__(self) -> None:
        if (
            isinstance(self.margin, bool)
            or not isinstance(self.margin, (int, float))
            or float(self.margin) < 0.0
            or not math.isfinite(float(self.margin))
        ):
            raise LossContractError(
                "raw-axis validity hinge margin must be finite and non-negative",
                code="loss.raw_axis_validity_hinge_margin",
                context={"margin": self.margin},
            )

    def per_atom_loss(self, context: LossContext) -> torch.Tensor:
        # Repeat each segment's box mean over its atoms. The existing reducer
        # averages those atoms back to exactly one box mean per segment, retaining
        # its global denominator and DDP compensation (including no-box zeros).
        result = self.per_segment_loss(context)
        object.__setattr__(self, "last_diagnostics", result.diagnostics())
        present = {atom.segment_index for atom in context.atoms}
        row_by_segment = {
            segment_index: row
            for row, segment_index in enumerate(
                segment.segment_index for segment in context.token_sequence.segments
                if segment.segment_index in present
            )
        }
        rows = torch.tensor(
            [row_by_segment[atom.segment_index] for atom in context.atoms],
            dtype=torch.long, device=context.logits.device,
        )
        return result.segment_losses.index_select(0, rows)

    def per_segment_loss(self, context: LossContext) -> RawAxisValidityHingeResult:
        if not isinstance(context, LossContext):
            raise LossContractError(
                "raw-axis validity hinge requires a LossContext",
                code="loss.raw_axis_validity_hinge_context",
                context={"value_type": type(context).__name__},
            )
        coordinate_ids = tuple(
            int(token_id) for token_id in context.vocab_groups.coordinate
        )
        if len(coordinate_ids) != DEFAULT_COORDINATE_BINS:
            raise LossContractError(
                "raw-axis validity hinge requires the canonical 1000-bin coordinate vocabulary",
                code="loss.raw_axis_validity_hinge_vocab",
                context={"coordinate_count": len(coordinate_ids)},
            )

        eligible_segment_indices = tuple(
            segment.segment_index
            for segment in context.token_sequence.segments
            if any(
                atom.segment_index == segment.segment_index for atom in context.atoms
            )
        )
        skipped_segment_count = len(context.token_sequence.segments) - len(
            eligible_segment_indices
        )
        coordinate_logits, coordinate_target_ids, coordinate_atoms = (
            context.select_logits_fp32(token_types=("coordinate",))
        )
        zero = (
            coordinate_logits.sum() * 0.0
            if coordinate_logits.numel() > 0
            else context.logits[0, 0, 0].float() * 0.0
        )
        if not eligible_segment_indices:
            return RawAxisValidityHingeResult(
                segment_losses=zero.unsqueeze(0)[:0],
                eligible_segment_count=0,
                skipped_segment_count=skipped_segment_count,
                complete_box_count=0,
                incomplete_box_count=0,
                coordinate_atom_count=len(coordinate_atoms),
                zero_box_segment_count=0,
            )

        coordinate_bin_by_token_id = {
            token_id: bin_value for bin_value, token_id in enumerate(coordinate_ids)
        }
        groups: dict[
            tuple[int, int, int, str, str],
            dict[str, Any],
        ] = {}
        for coordinate_index, (target_id, atom) in enumerate(
            zip(coordinate_target_ids.tolist(), coordinate_atoms, strict=True)
        ):
            target = atom.coordinate_target
            if target is None:
                raise LossContractError(
                    "supervised coordinate atom is missing CoordinateLossTarget metadata",
                    code="loss.raw_axis_validity_hinge_target_missing",
                    context={"atom": atom.to_artifact_dict()},
                )
            if not isinstance(atom.object_id, str) or not atom.object_id:
                raise LossContractError(
                    "supervised coordinate atom is missing object identity",
                    code="loss.raw_axis_validity_hinge_object_missing",
                    context={"atom": atom.to_artifact_dict()},
                )
            slot = int(target.slot_index)
            expected_bin = int(target.bbox[slot])
            observed_bin = coordinate_bin_by_token_id[int(target_id)]
            if observed_bin != expected_bin:
                raise LossContractError(
                    "coordinate target token does not match its declared bbox slot",
                    code="loss.raw_axis_validity_hinge_target_mismatch",
                    context={
                        "atom": atom.to_artifact_dict(),
                        "expected_bin": expected_bin,
                        "observed_bin": observed_bin,
                    },
                )
            key = (
                int(atom.pack_index),
                int(atom.segment_index),
                int(atom.example_index),
                str(atom.example_id),
                atom.object_id,
            )
            group = groups.setdefault(
                key,
                {"bbox": target.bbox, "positions": {}},
            )
            if tuple(group["bbox"]) != tuple(target.bbox):
                raise LossContractError(
                    "coordinate slots for one object must declare one bbox",
                    code="loss.raw_axis_validity_hinge_bbox_mismatch",
                    context={"identity": list(key)},
                )
            positions = group["positions"]
            if slot in positions:
                raise LossContractError(
                    "coordinate slots for one object must be unique",
                    code="loss.raw_axis_validity_hinge_duplicate_slot",
                    context={"identity": list(key), "slot_index": slot},
                )
            positions[slot] = coordinate_index

        complete_groups: list[tuple[int, Mapping[str, int]]] = []
        incomplete_box_count = 0
        complete_count_by_segment = {
            segment_index: 0 for segment_index in eligible_segment_indices
        }
        for key, group in groups.items():
            positions = group["positions"]
            if set(positions) != {0, 1, 2, 3}:
                incomplete_box_count += 1
                continue
            segment_index = int(key[1])
            complete_count_by_segment[segment_index] += 1
            complete_groups.append(
                (
                    segment_index,
                    {
                        "x1_position": int(positions[0]),
                        "y1_position": int(positions[1]),
                        "x2_position": int(positions[2]),
                        "y2_position": int(positions[3]),
                    },
                )
            )

        if complete_groups:
            box_losses = raw_axis_validity_hinge_per_box(
                coordinate_logits,
                tuple(box for _, box in complete_groups),
                coordinate_token_ids=coordinate_ids,
                coordinate_bin_values=tuple(range(DEFAULT_COORDINATE_BINS)),
                margin=float(self.margin),
            )
        else:
            box_losses = zero.unsqueeze(0)[:0]
        losses_by_segment: dict[int, list[torch.Tensor]] = {
            segment_index: [] for segment_index in eligible_segment_indices
        }
        for (segment_index, _), box_loss in zip(
            complete_groups, box_losses, strict=True
        ):
            losses_by_segment[segment_index].append(box_loss)
        segment_losses = torch.stack(
            tuple(
                torch.stack(losses_by_segment[segment_index]).mean()
                if losses_by_segment[segment_index]
                else zero
                for segment_index in eligible_segment_indices
            )
        )
        if not torch.isfinite(segment_losses).all():
            raise LossContractError(
                "raw-axis validity hinge produced non-finite segment losses",
                code="loss.raw_axis_validity_hinge_non_finite",
                context={"pack_index": context.token_sequence.pack_index},
            )
        return RawAxisValidityHingeResult(
            segment_losses=segment_losses,
            eligible_segment_count=len(eligible_segment_indices),
            skipped_segment_count=skipped_segment_count,
            complete_box_count=len(complete_groups),
            incomplete_box_count=incomplete_box_count,
            coordinate_atom_count=len(coordinate_atoms),
            zero_box_segment_count=sum(
                count == 0 for count in complete_count_by_segment.values()
            ),
        )


def raw_axis_validity_hinge(
    logits: torch.Tensor,
    boxes: Sequence[Mapping[str, Any]],
    *,
    coordinate_token_ids: Sequence[int],
    coordinate_bin_values: Sequence[int],
    margin: float,
) -> torch.Tensor:
    per_box = raw_axis_validity_hinge_per_box(
        logits,
        boxes,
        coordinate_token_ids=coordinate_token_ids,
        coordinate_bin_values=coordinate_bin_values,
        margin=margin,
    )
    if per_box.numel() == 0:
        return logits.new_zeros(())
    return per_box.mean()


def raw_axis_validity_hinge_per_box(
    logits: torch.Tensor,
    boxes: Sequence[Mapping[str, Any]],
    *,
    coordinate_token_ids: Sequence[int],
    coordinate_bin_values: Sequence[int],
    margin: float,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise LossContractError(
            "raw-axis validity hinge logits must have shape [positions, vocab]",
            code="loss.raw_axis_validity_hinge_logits_shape",
            context={"shape": getattr(logits, "shape", None)},
        )
    token_ids = tuple(int(value) for value in coordinate_token_ids)
    bin_values = tuple(int(value) for value in coordinate_bin_values)
    if not token_ids or len(token_ids) != len(bin_values):
        raise LossContractError(
            "coordinate token ids and bin values must be non-empty and aligned",
            code="loss.raw_axis_validity_hinge_coordinate_table",
            context={
                "token_id_count": len(token_ids),
                "bin_value_count": len(bin_values),
            },
        )
    if len(set(token_ids)) != len(token_ids):
        raise LossContractError(
            "coordinate token ids must be unique",
            code="loss.raw_axis_validity_hinge_coordinate_table",
            context={"token_id_count": len(token_ids)},
        )
    if any(token_id < 0 or token_id >= int(logits.shape[1]) for token_id in token_ids):
        raise LossContractError(
            "coordinate token id is outside the logits vocabulary",
            code="loss.raw_axis_validity_hinge_coordinate_table",
            context={"vocab_size": int(logits.shape[1])},
        )
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or float(margin) < 0.0
        or not math.isfinite(float(margin))
    ):
        raise LossContractError(
            "raw-axis validity hinge margin must be finite and non-negative",
            code="loss.raw_axis_validity_hinge_margin",
            context={"margin": margin},
        )
    if not boxes:
        return logits.new_zeros((0,), dtype=torch.float32)

    columns = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    values = torch.tensor(bin_values, dtype=torch.float32, device=logits.device) / 999.0
    probabilities = logits.float().index_select(1, columns).softmax(dim=-1)
    # Elementwise FP32 reduction stays FP32 inside the training BF16 autocast.
    expectations = (probabilities * values).sum(dim=-1)
    per_box: list[torch.Tensor] = []
    for box_index, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = (
                int(box[key])
                for key in (
                    "x1_position",
                    "y1_position",
                    "x2_position",
                    "y2_position",
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LossContractError(
                "raw-axis validity hinge box positions must be integer xyxy mappings",
                code="loss.raw_axis_validity_hinge_box_positions",
                context={"box_index": box_index},
                cause=exc,
            ) from exc
        positions = (x1, y1, x2, y2)
        if any(
            position < 0 or position >= int(logits.shape[0]) for position in positions
        ):
            raise LossContractError(
                "raw-axis validity hinge box position is outside logits rows",
                code="loss.raw_axis_validity_hinge_box_positions",
                context={
                    "box_index": box_index,
                    "positions": list(positions),
                    "logits_rows": int(logits.shape[0]),
                },
            )
        x_penalty = F.relu(float(margin) - (expectations[x2] - expectations[x1]))
        y_penalty = F.relu(float(margin) - (expectations[y2] - expectations[y1]))
        per_box.append((x_penalty + y_penalty) / 2.0)
    result = torch.stack(per_box)
    if not torch.isfinite(result).all():
        raise LossContractError(
            "raw-axis validity hinge produced non-finite box losses",
            code="loss.raw_axis_validity_hinge_non_finite",
            context={"box_count": len(per_box)},
        )
    return result


__all__ = [
    "DEFAULT_COORDINATE_BINS",
    "DEFAULT_RAW_AXIS_MARGIN",
    "RawAxisValidityHingeLoss",
    "RawAxisValidityHingeResult",
    "raw_axis_validity_hinge",
    "raw_axis_validity_hinge_per_box",
]

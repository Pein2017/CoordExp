"""Conditional legal-coordinate mass for the supplied xyxy teacher prefix."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.common.errors import LossContractError
from src.losses.context import LossContext
from src.supervision import TokenAtom, TokenSequence


@dataclass(frozen=True)
class ConditionalOrderGateResult:
    segment_losses: torch.Tensor
    eligible_segment_count: int
    skipped_segment_count: int
    complete_box_count: int
    incomplete_box_count: int
    coordinate_atom_count: int
    zero_box_segment_count: int
    participating_slot_count: int

    def diagnostics(self) -> dict[str, int]:
        return {
            "eligible_segment_count": self.eligible_segment_count,
            "skipped_segment_count": self.skipped_segment_count,
            "complete_box_count": self.complete_box_count,
            "incomplete_box_count": self.incomplete_box_count,
            "coordinate_atom_count": self.coordinate_atom_count,
            "zero_box_segment_count": self.zero_box_segment_count,
            "participating_slot_count": self.participating_slot_count,
        }


def grouped_coordinate_atoms(
    sequence: TokenSequence,
) -> dict[tuple[int, int, int, str, str], dict[int, TokenAtom]]:
    """Validate identity, causal order and actual supplied prefix tokens."""
    groups: dict[tuple[int, int, int, str, str], dict[int, TokenAtom]] = {}
    bboxes: dict[tuple[int, int, int, str, str], tuple[int, ...]] = {}
    for atom in sequence.atoms:
        if atom.token_type != "coordinate":
            continue
        target = atom.coordinate_target
        if target is None or not atom.object_id:
            raise LossContractError(
                "order gate requires coordinate target and object identity",
                code="loss.conditional_order_gate_target_missing",
                context={"atom": atom.to_artifact_dict()},
            )
        if sequence.input_ids[atom.target_position] != atom.token_id:
            raise LossContractError(
                "order gate target differs from the token actually supplied in its prefix",
                code="loss.conditional_order_gate_prefix_mismatch",
                context={"atom": atom.to_artifact_dict()},
            )
        key = (atom.pack_index, atom.segment_index, atom.example_index,
               atom.example_id, atom.object_id)
        bbox = tuple(int(value) for value in target.bbox)
        if key in bboxes and bboxes[key] != bbox:
            raise LossContractError(
                "order gate object declares inconsistent boxes",
                code="loss.conditional_order_gate_bbox_mismatch",
                context={"identity": list(key)},
            )
        bboxes[key] = bbox
        slots = groups.setdefault(key, {})
        slot = int(target.slot_index)
        if slot in slots:
            raise LossContractError(
                "order gate object repeats a coordinate role",
                code="loss.conditional_order_gate_duplicate_slot",
                context={"identity": list(key), "slot": slot},
            )
        slots[slot] = atom
    for key, slots in groups.items():
        if 2 in slots and 0 not in slots or 3 in slots and 1 not in slots:
            raise LossContractError(
                "order gate constrained slot lacks its same-object prefix slot",
                code="loss.conditional_order_gate_missing_predecessor",
                context={"identity": list(key)},
            )
        if set(slots) == {0, 1, 2, 3} and not (
            slots[0].target_position < slots[1].target_position
            < slots[2].target_position < slots[3].target_position
        ):
            raise LossContractError(
                "order gate roles are not in causal xyxy order",
                code="loss.conditional_order_gate_role_order",
                context={"identity": list(key)},
            )
    return groups


@dataclass(frozen=True)
class ConditionalOrderGateLoss:
    name: str = "conditional_order_gate"

    def per_segment_loss(self, context: LossContext) -> ConditionalOrderGateResult:
        coordinate_ids = tuple(int(value) for value in context.vocab_groups.coordinate)
        if len(coordinate_ids) != 1000 or len(set(coordinate_ids)) != 1000:
            raise LossContractError(
                "order gate requires 1000 unique numeric-bin coordinate IDs",
                code="loss.conditional_order_gate_vocab",
                context={"count": len(coordinate_ids)},
            )
        selected, targets, atoms = context.select_logits_fp32(token_types=("coordinate",))
        groups = grouped_coordinate_atoms(context.token_sequence)
        row_by_atom = {id(atom): row for row, atom in enumerate(atoms)}
        bin_by_id = {token_id: bin_value for bin_value, token_id in enumerate(coordinate_ids)}
        for token_id, atom in zip(targets.tolist(), atoms, strict=True):
            target = atom.coordinate_target
            assert target is not None
            if bin_by_id.get(int(token_id)) != int(target.bbox[target.slot_index]):
                raise LossContractError(
                    "order gate target ID disagrees with declared numeric bin",
                    code="loss.conditional_order_gate_target_mismatch",
                    context={"atom": atom.to_artifact_dict()},
                )
        zero = selected.sum() * 0.0 if selected.numel() else context.logits[0, 0, 0].float() * 0.0
        eligible = tuple(segment.segment_index for segment in context.token_sequence.segments
                         if any(atom.segment_index == segment.segment_index for atom in context.atoms))
        losses: dict[int, list[torch.Tensor]] = {index: [] for index in eligible}
        complete = 0
        incomplete = 0
        columns = torch.tensor(coordinate_ids, dtype=torch.long, device=selected.device)
        coord_logits = selected.index_select(1, columns)
        for key, slots in groups.items():
            if set(slots) != {0, 1, 2, 3}:
                incomplete += 1
                continue
            complete += 1
            slot_losses = []
            for predecessor, constrained in ((0, 2), (1, 3)):
                preceding_atom = slots[predecessor]
                query_atom = slots[constrained]
                if preceding_atom.target_position >= query_atom.target_position:
                    raise LossContractError(
                        "order gate predecessor is not in the causal prefix",
                        code="loss.conditional_order_gate_role_order",
                        context={"identity": list(key)},
                    )
                threshold = bin_by_id[context.token_sequence.input_ids[preceding_atom.target_position]]
                if threshold == 999:
                    raise LossContractError(
                        "order gate prefix leaves no legal coordinate",
                        code="loss.conditional_order_gate_empty_valid",
                        context={"identity": list(key)},
                    )
                row = coord_logits[row_by_atom[id(query_atom)]]
                slot_losses.append(torch.logsumexp(row, 0) - torch.logsumexp(row[threshold + 1:], 0))
            losses[key[1]].append(torch.stack(slot_losses).mean())
        segment_losses = torch.stack(tuple(torch.stack(losses[index]).mean() if losses[index] else zero
                                           for index in eligible)) if eligible else zero.unsqueeze(0)[:0]
        if not torch.isfinite(segment_losses).all():
            raise LossContractError(
                "order gate produced non-finite loss",
                code="loss.conditional_order_gate_non_finite",
                context={"pack_index": context.token_sequence.pack_index},
            )
        return ConditionalOrderGateResult(
            segment_losses=segment_losses,
            eligible_segment_count=len(eligible),
            skipped_segment_count=len(context.token_sequence.segments) - len(eligible),
            complete_box_count=complete,
            incomplete_box_count=incomplete,
            coordinate_atom_count=len(atoms),
            zero_box_segment_count=sum(not losses[index] for index in eligible),
            participating_slot_count=2 * complete,
        )

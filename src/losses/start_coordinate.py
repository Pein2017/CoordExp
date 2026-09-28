"""Three GT-prefix onset objectives for the bounded fourth-loss benchmark."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

from src.common.errors import LossContractError
from src.losses.conditional_order_gate import grouped_coordinate_atoms
from src.losses.context import LossContext


@dataclass(frozen=True)
class StartCoordinateLoss:
    mode: str
    margin: float = 0.2
    radius_fraction: float = 0.02
    radius_cap: int = 4
    calibrate: bool = False
    name: str = "start_coordinate"

    def __post_init__(self):
        if self.mode not in {"ce", "local_mass", "instance_margin"}:
            raise ValueError("unsupported start-coordinate objective")
        if not 0 < self.margin or not 0 <= self.radius_fraction <= 1 or not 0 <= self.radius_cap <= 999:
            raise ValueError("invalid start-coordinate margin or radius")

    def per_atom_loss(self, context: LossContext) -> torch.Tensor:
        coord_ids = tuple(context.vocab_groups.coordinate)
        if len(coord_ids) != 1000 or len(set(coord_ids)) != 1000:
            raise LossContractError("start loss requires numeric 0..999 coordinate IDs", code="loss.start_vocab")
        bins = {token: value for value, token in enumerate(coord_ids)}
        groups = grouped_coordinate_atoms(context.token_sequence)
        descriptions: dict[tuple, set[str]] = {}
        for atom in context.atoms:
            if atom.token_type == "desc_text" and atom.object_id:
                key = (atom.pack_index, atom.segment_index, atom.example_index, atom.example_id, atom.object_id)
                descriptions.setdefault(key, set()).add(atom.text)
        complete = {key: slots for key, slots in groups.items() if set(slots) == {0, 1, 2, 3}}
        for key, slots in complete.items():
            if len(descriptions.get(key, ())) != 1:
                raise LossContractError("start loss needs one complete object description", code="loss.start_description")
            for role, atom in slots.items():
                if bins.get(atom.token_id) != atom.coordinate_target.bbox[role]:
                    raise LossContractError("start coordinate metadata differs from supplied token", code="loss.start_target")

        starts = [(key, role, slots[role]) for key, slots in complete.items() for role in (0, 1)]
        kept = None if context.logits_position_ids is None else {position: i for i, position in enumerate(context.logits_position_ids)}
        positions = torch.tensor([
            atom.causal_logits_position if kept is None else kept[atom.causal_logits_position]
            for _, _, atom in starts
        ], dtype=torch.long, device=context.logits.device)
        if starts:
            # Select the coordinate columns before FP32 conversion; no second full-vocabulary copy.
            rows = context.logits[0].index_select(0, positions).index_select(
                1, torch.tensor(coord_ids, device=context.logits.device)
            ).float()
        else:
            rows = context.logits.new_empty((0, 1000), dtype=torch.float32)
        zero = context.logits[0, 0, 0].float() * 0
        by_segment: dict[int, list[torch.Tensor]] = {}
        segment_slots: dict[int, int] = {}
        for key, _, _ in starts:
            segment_slots[key[1]] = segment_slots.get(key[1], 0) + 1
        grad_sq = {mode: 0.0 for mode in ("ce", "local_mass", "instance_margin")}
        eligible_pairs = 0
        active_margins = []

        def interval(slots, role):
            box = slots[role].coordinate_target.bbox
            radius = min(self.radius_cap, int((box[role + 2] - box[role]) * self.radius_fraction))
            return max(0, box[role] - radius), min(999, box[role] + radius)

        for row, (key, role, atom) in zip(rows, starts, strict=True):
            lo, hi = interval(complete[key], role)
            target = int(atom.coordinate_target.bbox[role])
            terms = {}
            modes = grad_sq if self.calibrate else (self.mode,)
            if "ce" in modes:
                terms["ce"] = torch.logsumexp(row, 0) - row[target]
            if "local_mass" in modes:
                terms["local_mass"] = torch.logsumexp(row, 0) - torch.logsumexp(row[lo:hi + 1], 0)
            negative_ids: list[int] = []
            if "instance_margin" in modes:
                for other_key, other_slots in complete.items():
                    if other_key[:4] != key[:4] or other_key == key or descriptions[other_key] != descriptions[key]:
                        continue
                    if role == 1:
                        xlo, xhi = interval(other_slots, 0)
                        actual_x = bins[context.token_sequence.input_ids[complete[key][0].target_position]]
                        if not xlo <= actual_x <= xhi:
                            continue
                    nlo, nhi = interval(other_slots, role)
                    if nlo <= hi and lo <= nhi:
                        continue  # Shared coordinate bins cannot identify a different instance.
                    negative_ids.extend(range(nlo, nhi + 1))
                if negative_ids:
                    negative_ids = sorted(set(negative_ids))
                    value = self.margin + row[negative_ids].max() - row[lo:hi + 1].max()
                    terms["instance_margin"] = value.relu()
                    eligible_pairs += 1
                    active_margins.append(value.detach() > 0)
                else:
                    terms["instance_margin"] = row.sum() * 0
            by_segment.setdefault(key[1], []).append(terms[self.mode])
            if self.calibrate:
                # Exact squared logit-gradient norms of the sum of segment means.
                # The common global denominator cancels in the calibration ratio.
                p = row.detach().softmax(0)
                ce_grad = p.clone()
                ce_grad[target] -= 1
                local_grad = p.clone()
                local_grad[lo:hi + 1] -= row[lo:hi + 1].detach().softmax(0)
                scale = segment_slots[key[1]] ** 2
                grad_sq["ce"] += float(ce_grad.square().sum()) / scale
                grad_sq["local_mass"] += float(local_grad.square().sum()) / scale
                if negative_ids and float(terms["instance_margin"].detach()) > 0:
                    positive = row[lo:hi + 1].detach()
                    negative = row[negative_ids].detach()
                    # torch.max() splits its subgradient across equal maxima.
                    grad_sq["instance_margin"] += (
                        1.0 / int((positive == positive.max()).sum())
                        + 1.0 / int((negative == negative.max()).sum())
                    ) / scale
        means = {segment: torch.stack(values).mean() for segment, values in by_segment.items()}
        object.__setattr__(self, "last_diagnostics", {
            "mode": self.mode, "start_slot_count": len(starts),
            "margin_eligible_slot_count": eligible_pairs,
            "margin_active_slot_count": int(torch.stack(active_margins).sum()) if active_margins else 0,
            "complete_box_count": len(complete), "incomplete_box_count": len(groups) - len(complete),
        })
        if self.calibrate:
            print("START_LOSS_CALIBRATION " + json.dumps({
                "rank": int(os.environ.get("RANK", "0")), "grad_sq": grad_sq,
                **self.last_diagnostics,
            }, sort_keys=True) + "\n", end="", flush=True)
        # Reuse the existing segment-balanced/global-DDP reducer unchanged.
        return torch.stack([means.get(atom.segment_index, zero) for atom in context.atoms])

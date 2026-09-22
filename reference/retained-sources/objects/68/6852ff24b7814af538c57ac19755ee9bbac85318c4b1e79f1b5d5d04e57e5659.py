"""Token-type group-mass gate loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.common.errors import LossContractError
from src.losses.context import LossContext


@dataclass(frozen=True)
class TokenTypeGateLoss:
    name: str = "token_type_gate"

    def per_atom_loss(self, context: LossContext) -> torch.Tensor:
        logits_fp32, _target_ids, atoms = context.select_logits_fp32()
        if not atoms:
            raise LossContractError(
                "TokenTypeGateLoss requires at least one supervised atom",
                code="loss.token_type_gate_empty",
                context={"term": self.name},
            )
        losses = torch.empty(
            (len(atoms),),
            dtype=torch.float32,
            device=logits_fp32.device,
        )
        all_logsumexp = torch.logsumexp(logits_fp32, dim=1)
        group_tensors: dict[str, torch.Tensor] = {}
        row_indices_by_type: dict[str, list[int]] = {}
        for row_index, atom in enumerate(atoms):
            row_indices_by_type.setdefault(atom.token_type, []).append(row_index)
        for token_type, row_indices in row_indices_by_type.items():
            group = group_tensors.get(token_type)
            if group is None:
                group = torch.tensor(
                    context.vocab_groups.allowed_ids(token_type),
                    dtype=torch.long,
                    device=logits_fp32.device,
                )
                group_tensors[token_type] = group
            row_index_tensor = torch.tensor(
                row_indices,
                dtype=torch.long,
                device=logits_fp32.device,
            )
            rows = logits_fp32.index_select(0, row_index_tensor)
            group_logsumexp = torch.logsumexp(rows.index_select(1, group), dim=1)
            losses.index_copy_(
                0,
                row_index_tensor,
                all_logsumexp.index_select(0, row_index_tensor) - group_logsumexp,
            )
        return losses


__all__ = ["TokenTypeGateLoss"]

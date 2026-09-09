"""Full-vocabulary base cross-entropy term."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.common.errors import LossContractError
from src.losses.context import LossContext
from src.losses.token_scores import aligned_token_logprobs


@dataclass(frozen=True)
class BaseTokenCE:
    name: str = "base_ce"

    def per_atom_loss(self, context: LossContext) -> torch.Tensor:
        logits_fp32, target_ids, atoms = context.select_logits_fp32()
        if not atoms:
            raise LossContractError(
                "BaseTokenCE requires at least one supervised atom",
                code="loss.base_ce_empty",
                context={"term": self.name},
            )
        return -aligned_token_logprobs(logits_fp32, target_ids)


__all__ = ["BaseTokenCE"]

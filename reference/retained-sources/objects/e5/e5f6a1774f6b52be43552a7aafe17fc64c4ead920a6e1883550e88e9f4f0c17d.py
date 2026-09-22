"""Distinct retained self-rollout coordinate/full-action credit objectives.

The frozen profile is 16 images x 4 branches. Credits and coordinate masks are
caller-owned evidence; these are not Source256 CE or RLOO normalization.
"""

import math

import torch

from src.losses import aligned_token_logprobs
from src.qwen.native import prepare_replay

COORDINATE_SCOPE = "coordinate_action_tokens"
FULL_ACTION_SCOPE = "full_sampled_action_tokens"
FULL_ACTION_OBJECTIVE = "-mean_16_images(mean_4_branches(credit*sum_all_sampled_action_raw_chosen_logprob))"


def coordinate_loss(chosen, positions, credit, world_size=8):
    """DDP averaging yields -mean_16(mean_4(credit * sum_4(log p)))."""
    return -chosen[positions].sum() * float(credit) * (world_size / (16 * 4))


def action_loss(chosen, positions, credit, scope=COORDINATE_SCOPE, world_size=8):
    if scope not in (COORDINATE_SCOPE, FULL_ACTION_SCOPE):
        raise ValueError("unsupported direct likelihood scope")
    if scope == FULL_ACTION_SCOPE:
        return -chosen.sum() * float(credit) * (world_size / (16 * 4))
    return coordinate_loss(chosen, positions, credit, world_size)


class OwnerOutcomeScorer(torch.nn.Module):
    """Replay a retained generated prefix and consume its declared credit scope."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, native_inputs, prompt_ids, cell, *, credit, scope, world_size=8):
        action = cell["action_token_ids"]
        positions = cell["coordinate_action_positions"]
        actual = [index for index, token in enumerate(action) if 151670 <= token <= 152669]
        if positions != actual or len(positions) != 4 or any(type(index) is not int for index in positions):
            raise ValueError("coordinate mask must select exactly all four action coordinates")
        if not math.isfinite(float(credit)):
            raise ValueError("nonfinite credit")
        replay = prepare_replay(
            self.model, native_inputs,
            prompt_token_ids=[*prompt_ids, *cell["prefix_token_ids"]],
            continuation_token_ids=action,
        )
        logits = replay.aligned_logits(self.model(**replay.inputs).logits)
        chosen = aligned_token_logprobs(logits, replay.target_ids)
        return action_loss(chosen, positions, credit, scope, world_size)

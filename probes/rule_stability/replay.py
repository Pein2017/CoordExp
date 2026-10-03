"""Keep image placeholders inside the validated original prompt during replay."""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch

from src.qwen.native import _checked_token_ids, resolve_rope_index


@contextmanager
def prompt_only_placeholder_masks(
    model: Any, prompt_token_ids: Sequence[int], *, record_masks: bool = False,
):
    """Scope only ``get_placeholder_mask`` to the original singleton prompt.

    Original validation receives unchanged feature tensors and graph-connected
    prompt slices. False suffix masks leave scatter, visual positions and
    DeepStack computation unchanged. The exact prior instance/class method
    binding is restored on success or failure. Optional receipts retain scalars
    and shapes only; count synchronization is limited to diagnostic callers.
    """
    prompt = _checked_token_ids(prompt_token_ids)
    if not prompt or not isinstance(record_masks, bool):
        raise ValueError("placeholder scope requires a nonempty literal prompt")
    owner = getattr(resolve_rope_index(model), "__self__", None)
    original = getattr(owner, "get_placeholder_mask", None)
    if owner is None or not callable(original):
        raise ValueError("placeholder scope requires the actual bound Qwen owner")
    had_instance_attribute = "get_placeholder_mask" in vars(owner)
    saved_attribute = vars(owner).get("get_placeholder_mask")
    width = len(prompt)
    receipt = {"prompt_length": width, "calls": []}

    def scoped(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1
                or not isinstance(inputs_embeds, torch.Tensor) or inputs_embeds.ndim != 3
                or inputs_embeds.shape[:2] != input_ids.shape or input_ids.shape[1] < width):
            raise ValueError("placeholder replay requires one full literal history and aligned embeddings")
        expected = torch.tensor(prompt, dtype=input_ids.dtype, device=input_ids.device)
        if not torch.equal(input_ids[0, :width], expected):
            raise ValueError("placeholder replay history differs from the original prompt prefix")
        masks = original(input_ids[:, :width], inputs_embeds[:, :width],
                         image_features=image_features, video_features=video_features)
        if len(masks) != 2 or any(mask is not None and
                (mask.shape != (1, width, 1) or mask.dtype != torch.bool) for mask in masks):
            raise ValueError("original Qwen placeholder mask shape/dtype changed")
        suffix_width = input_ids.shape[1] - width
        padded = tuple(None if mask is None else torch.cat((mask,
            torch.zeros((1, suffix_width, 1), dtype=torch.bool, device=mask.device)), dim=1) for mask in masks)
        entry = {"history_width": input_ids.shape[1],
                 "original_image_mask_shape": None if masks[0] is None else list(masks[0].shape),
                 "original_video_mask_shape": None if masks[1] is None else list(masks[1].shape),
                 "padded_image_mask_shape": None if padded[0] is None else list(padded[0].shape),
                 "padded_video_mask_shape": None if padded[1] is None else list(padded[1].shape),
                 "suffix_image_true": 0, "suffix_video_true": 0}
        if record_masks:
            entry.update(prompt_image_positions=0 if masks[0] is None else int(masks[0].sum().item()),
                         prompt_video_positions=0 if masks[1] is None else int(masks[1].sum().item()))
        receipt["calls"].append(entry)
        return padded

    owner.get_placeholder_mask = MethodType(scoped, owner)
    try:
        yield receipt
    finally:
        if had_instance_attribute:
            owner.get_placeholder_mask = saved_attribute
        else:
            delattr(owner, "get_placeholder_mask")

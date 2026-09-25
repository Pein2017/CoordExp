"""Saved native-prefix construction with the original companion EOS policy.

This does not perform model work or regenerate positions. Its all-visible
extension mask is distinct from exact-history row scoring's companion policy.
"""
from __future__ import annotations
from typing import Any
import torch
from src.qwen.native import _STALE_HISTORY_FIELDS
EOS = 151645


def prefix_tokens(raw: list[dict[str, Any]], offset: int, pad: int) -> list[list[int]]:
    """Make equal-length prefixes, ending short companions with EOS then pad."""
    result: list[list[int]] = []
    for row in raw:
        tokens = list(row["token_ids"][:offset])
        if len(tokens) < offset:
            if EOS not in tokens:
                tokens.append(EOS)
            else:
                tokens = tokens[: tokens.index(EOS) + 1]
            tokens.extend([pad] * (offset - len(tokens)))
        result.append(tokens[:offset])
    if any(len(row) != offset for row in result):
        raise ValueError("prefix materialization did not produce equal lengths")
    return result

def full_prefix(
    batch: Any,
    raw: list[dict[str, Any]],
    offset: int,
    pad: int,
    device: torch.device | str,
    mutation: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Build a same-width native prefix while retaining the image tensors."""
    suffix = prefix_tokens(raw, offset, pad)
    if mutation is not None:
        batch_index, position, old, new = mutation
        if not 0 <= batch_index < len(suffix) or not 0 <= position < offset:
            raise ValueError("mutation position is outside the native prefix")
        if suffix[batch_index][position] != old:
            raise ValueError("mutation does not name the original prefix token")
        suffix[batch_index][position] = new
    inputs = {
        key: value
        for key, value in batch.inputs.items()
        if key not in _STALE_HISTORY_FIELDS
        and key not in ("use_cache", "return_dict", "logits_to_keep")
    }
    ext = torch.tensor(suffix, device=device, dtype=torch.long)
    inputs["input_ids"] = torch.cat((batch.inputs["input_ids"], ext), dim=1)
    inputs["attention_mask"] = torch.cat(
        (
            batch.inputs["attention_mask"],
            torch.ones(
                len(raw), offset, device=device, dtype=batch.inputs["attention_mask"].dtype
            ),
        ),
        dim=1,
    )
    return inputs

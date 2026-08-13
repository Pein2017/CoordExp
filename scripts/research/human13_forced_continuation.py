"""Exact forced-row then natural-continuation projection for Human-13.

The selected native row is an intervention.  Everything after that row is one
ordinary greedy model continuation; no token from the released suffix is
teacher-forced or re-tokenized before generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

from scripts.research.run_local_branch_causal_value import (
    _append_exact_prefix,
    hash_prefix_token_ids,
)
from src.inference.parsing import parse_compact_object_box_closed


@dataclass(frozen=True)
class ForcedContinuationResult:
    forced_row_token_ids: tuple[int, ...]
    released_token_ids: tuple[int, ...]
    termination_status: str
    cap_hit: bool
    generated_text: str
    forced_row_parse_evidence: Mapping[str, Any]
    parse_evidence: Mapping[str, Any]
    natural_prefix_token_ids_sha256: str
    forced_row_token_ids_sha256: str
    forced_context_sha256: str
    released_token_ids_sha256: str
    requested_continuation_cap: int
    minimum_continuation_cap: int
    repetition_penalty: float
    current_checkpoint_payload_sha256: str


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def source_continuation_cap(*, source_row_count: int, source_token_count: int) -> int:
    """Return the declared per-image continuation-cap lower bound."""

    for label, value in (
        ("source_row_count", source_row_count),
        ("source_token_count", source_token_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    return max(2 * source_row_count, source_token_count + 512)


def _token_tuple(
    values: Sequence[int], *, label: str, allow_empty: bool
) -> tuple[int, ...]:
    token_ids = tuple(int(value) for value in values)
    if not allow_empty and not token_ids:
        raise ValueError(f"{label} must be non-empty")
    if any(value < 0 for value in token_ids):
        raise ValueError(f"{label} must contain non-negative token ids")
    return token_ids


def forced_complete_row_then_natural_continuation(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    natural_prefix_token_ids: Sequence[int],
    forced_row_token_ids: Sequence[int],
    tokenizer: Any,
    image_width: int,
    image_height: int,
    repetition_penalty: float,
    continuation_cap: int,
    source_row_count: int,
    source_token_count: int,
    current_checkpoint_payload_sha256: str,
) -> ForcedContinuationResult:
    """Force one complete native row, then release exactly one greedy suffix."""

    import torch

    natural_prefix = _token_tuple(
        natural_prefix_token_ids,
        label="natural_prefix_token_ids",
        allow_empty=True,
    )
    forced_row = _token_tuple(
        forced_row_token_ids,
        label="forced_row_token_ids",
        allow_empty=False,
    )
    minimum_cap = source_continuation_cap(
        source_row_count=source_row_count,
        source_token_count=source_token_count,
    )
    if (
        isinstance(continuation_cap, bool)
        or not isinstance(continuation_cap, int)
        or continuation_cap < minimum_cap
    ):
        raise ValueError(
            f"continuation_cap is below the Source-derived minimum {minimum_cap}"
        )
    if (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, (int, float))
        or not math.isfinite(float(repetition_penalty))
        or float(repetition_penalty) <= 0
    ):
        raise ValueError("repetition_penalty must be finite and positive")
    if (
        not isinstance(current_checkpoint_payload_sha256, str)
        or _SHA256_RE.fullmatch(current_checkpoint_payload_sha256) is None
    ):
        raise ValueError("current checkpoint payload SHA-256 is invalid")
    if "input_ids" not in native_inputs:
        raise ValueError("native_inputs must contain input_ids")

    forced_context = natural_prefix + forced_row
    model_inputs, model_input_width = _append_exact_prefix(
        native_inputs, forced_context
    )
    kwargs: dict[str, Any] = {
        **model_inputs,
        "max_new_tokens": int(continuation_cap),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": False,
        "eos_token_id": session._im_end_token_id(),  # noqa: SLF001
        "pad_token_id": session._pad_token_id(),  # noqa: SLF001
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    with torch.inference_mode():
        output = session._model.generate(**kwargs)  # noqa: SLF001
    sequences = getattr(output, "sequences", None)
    if sequences is None or int(sequences.ndim) != 2 or int(sequences.shape[0]) != 1:
        raise RuntimeError("forced continuation did not return exactly one sequence")
    if int(sequences.shape[1]) < model_input_width:
        raise RuntimeError(
            "forced continuation returned a sequence shorter than its input"
        )

    released = tuple(int(value) for value in sequences[0, model_input_width:].tolist())
    cap_hit = len(released) >= int(continuation_cap)
    im_end = int(session._im_end_token_id())  # noqa: SLF001
    if cap_hit:
        termination_status = "cap_hit"
    elif released and released[-1] == im_end:
        termination_status = "natural_im_end"
    else:
        termination_status = "nonterminal_return"

    generated_text = tokenizer.decode(list(released), skip_special_tokens=False)
    forced_row_text = tokenizer.decode(list(forced_row), skip_special_tokens=False)
    forced_parsed = parse_compact_object_box_closed(
        forced_row_text,
        row_id="human13:forced-row:intervention",
        row_index=0,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    if len(forced_parsed.predictions) != 1 or forced_parsed.dropped_predictions:
        raise ValueError(
            "forced native row did not parse as exactly one clean prediction"
        )
    parsed = parse_compact_object_box_closed(
        generated_text,
        row_id="human13:forced-row:natural-continuation",
        row_index=0,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    return ForcedContinuationResult(
        forced_row_token_ids=forced_row,
        released_token_ids=released,
        termination_status=termination_status,
        cap_hit=cap_hit,
        generated_text=generated_text,
        forced_row_parse_evidence=forced_parsed.to_artifact_dict(),
        parse_evidence=parsed.to_artifact_dict(),
        natural_prefix_token_ids_sha256=hash_prefix_token_ids(natural_prefix),
        forced_row_token_ids_sha256=hash_prefix_token_ids(forced_row),
        forced_context_sha256=hash_prefix_token_ids(forced_context),
        released_token_ids_sha256=hash_prefix_token_ids(released),
        requested_continuation_cap=continuation_cap,
        minimum_continuation_cap=minimum_cap,
        repetition_penalty=float(repetition_penalty),
        current_checkpoint_payload_sha256=current_checkpoint_payload_sha256,
    )


__all__ = [
    "ForcedContinuationResult",
    "forced_complete_row_then_natural_continuation",
    "source_continuation_cap",
]

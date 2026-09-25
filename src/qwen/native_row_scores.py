"""Exact diagnostic row scoring and saved-trace comparison.

Scoring is inference-only. Companion padding intentionally does not insert EOS;
the separate saved-prefix operation has a different frozen conditioning rule.
Row roles retain the offset-four compact-template convention, not a general
variable-length category parser. Tolerance is a technical comparison only.
"""
from __future__ import annotations
import hashlib
import json
from typing import Any
import torch
from src.qwen.native import exact_history_inputs
REF_END = 151647
BOX_START = 151648
COORD_BASE = 151670
COORD_COUNT = 1000
ROW_END = 151649
ATOL = 2e-4


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def fixed_template_row_role(tokens: list[int], offset: int) -> str:
    if offset == 0:
        return "entry"
    if tokens[offset] == REF_END:
        return "description_end"
    if tokens[offset] == BOX_START:
        return "box_start"
    if tokens[offset] == ROW_END:
        return "terminator"
    if COORD_BASE <= tokens[offset] < COORD_BASE + COORD_COUNT:
        return ("x1", "y1", "x2", "y2")[max(0, min(3, offset - 4))]
    return "description"

def _trace_top2(step: dict[str, Any], batch_index: int) -> tuple[list[int], list[float]]:
    values = step["raw_top2"][batch_index]
    if values and isinstance(values[0], (list, tuple)):
        token_ids = [int(item[0]) for item in values]
        logits = [float(item[1]) for item in values]
    else:
        token_ids = [int(step["raw_winners"][batch_index]), int(step["raw_runnerups"][batch_index])]
        logits = [float(item) for item in values]
    _require(len(token_ids) == 2 and len(logits) == 2, "native source trace top2 is not length two")
    return token_ids, logits

def compare_saved_trace(
    *,
    logits: torch.Tensor,
    trace: dict[str, Any],
    batch_index: int,
    absolute_offset: int,
    token_id: int,
    role: str,
    atol: float = ATOL,
) -> dict[str, Any]:
    steps = trace.get("steps", [])
    _require(0 <= absolute_offset < len(steps), f"source trace lacks offset {absolute_offset}")
    step = steps[absolute_offset]
    chosen = int(step["chosen"][batch_index])
    chosen_raw = float(step["chosen_raw_logits"][batch_index])
    saved_lse = float(step["logsumexp"][batch_index])
    current = logits.detach().float()
    current_lse = float(torch.logsumexp(current, dim=-1).item())
    current_chosen = float(current[int(token_id)].item())
    current_logprob = float(torch.log_softmax(current, dim=-1)[int(token_id)].item())
    source_logprob = chosen_raw - saved_lse
    values, indices = torch.topk(current, 2)
    current_top_ids = [int(item) for item in indices.tolist()]
    current_top_logits = [float(item) for item in values.tolist()]
    source_top_ids, source_top_logits = _trace_top2(step, batch_index)
    logprob_error = abs(current_logprob - source_logprob)
    chosen_logit_error = abs(current_chosen - chosen_raw)
    lse_error = abs(current_lse - saved_lse)
    top2_error = max(abs(a - b) for a, b in zip(current_top_logits, source_top_logits, strict=True))
    return {
        "absolute_offset": absolute_offset,
        "role": role,
        "token_id": int(token_id),
        "source_chosen_token_id": chosen,
        "source_top2_token_ids": source_top_ids,
        "current_top2_token_ids": current_top_ids,
        "source_chosen_raw_logit": chosen_raw,
        "source_logsumexp": saved_lse,
        "source_logprob": source_logprob,
        "current_chosen_logit": current_chosen,
        "current_logsumexp": current_lse,
        "current_logprob": current_logprob,
        "chosen_logit_abs_error": chosen_logit_error,
        "logsumexp_abs_error": lse_error,
        "logprob_abs_error": logprob_error,
        "top2_max_abs_error": top2_error,
        "winner_match": current_top_ids[0] == source_top_ids[0],
        "runnerup_match": current_top_ids[1] == source_top_ids[1],
        "chosen_token_match": chosen == int(token_id),
        "passed": bool(
            chosen == int(token_id)
            and current_top_ids == source_top_ids
            and max(chosen_logit_error, lse_error, logprob_error, top2_error) <= atol
        ),
        "tolerance": atol,
    }

def candidate_histories(batch: Any, raw: list[dict[str, Any]], target: int, actions: list[int], pad: int) -> list[list[int]]:
    histories: list[list[int]] = []
    for index, prompt in enumerate(batch.prompt_token_ids):
        if index == target:
            suffix = list(actions)
        else:
            suffix = [int(token) for token in raw[index]["token_ids"][: len(actions)]]
        suffix.extend([pad] * (len(actions) - len(suffix)))
        histories.append(list(prompt) + suffix)
    return histories

def score_candidate(
    *,
    model: Any,
    batch: Any,
    raw: list[dict[str, Any]],
    target: int,
    prefix: list[int],
    tokens: list[int],
    pad: int,
    device: torch.device,
) -> dict[str, Any]:
    actions = prefix + list(tokens)
    histories = candidate_histories(batch, raw, target, actions, pad)
    inputs = exact_history_inputs(
        model,
        batch.inputs,
        histories,
        pad_token_id=pad,
        logits_to_keep=len(tokens) + 1,
    )
    with torch.inference_mode():
        output = model(**inputs)
    logits = output.logits[target].detach().float()
    _require(logits.ndim == 2 and logits.shape[0] == len(tokens) + 1, "compact exact replay shape changed")
    action_logits = logits[:-1]
    logprobs = torch.log_softmax(action_logits, dim=-1)
    selected = logprobs[torch.arange(len(tokens), device=logprobs.device), torch.tensor(tokens, device=logprobs.device)]
    input_width = int(inputs["input_ids"].shape[1])
    position_start = input_width - len(tokens) - 1
    positions = inputs["position_ids"][..., target, position_start : input_width - 1].detach().cpu().transpose(0, 1).tolist()
    _require(len(positions) == len(tokens), "compact replay position count changed")
    return {
        "token_ids": list(tokens),
        "token_logprobs": [float(value) for value in selected.detach().cpu().tolist()],
        "row_sum_logprob": float(selected.sum().item()),
        "positions": positions,
        "positions_sha256": _digest(positions),
        "vocabulary_size": int(action_logits.shape[-1]),
        "boundary_logits": logits[0].detach().cpu(),
        "action_logits": action_logits.detach().cpu(),
        "input_ids_sha256": _digest(inputs["input_ids"][target].detach().cpu().tolist()),
        "prefix_sha256": _digest(prefix),
        "device": str(device),
    }

"""Exact-history Qwen continuation with explicit budgets and optional likelihoods.

This is an operation on caller-owned models and prepared inputs. It does not
load models, choose cohorts, parse detections, or manage a training lifecycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

import torch
from torch.nn import functional as F

from src.common.errors import RuntimeContractError
from src.qwen.native import (
    NativeBatch,
    _STALE_HISTORY_FIELDS,
    _checked_token_ids,
    _require_rank_two_tensor,
    model_device,
    move_to_device,
    padded_histories,
)

_POLICY_SCORE_STEP_CHUNK_SIZE = 32


@dataclass(frozen=True)
class NativeGenerationPolicy:
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    top_k: int | None = None
    use_model_defaults: bool = True

    def __post_init__(self) -> None:
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float))
            for v in (self.temperature, self.top_p, self.repetition_penalty)
        ):
            raise ValueError("generation policy numbers must be real numeric values")
        if not isinstance(self.use_model_defaults, bool):
            raise ValueError("use_model_defaults must be boolean")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive")
        if self.top_k is not None and (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 0
        ):
            raise ValueError("top_k must be a nonnegative integer or None")
        if self.temperature == 0 and (self.top_p != 1 or self.top_k not in (None, 0)):
            raise ValueError("greedy policy cannot apply sampling filters")


@dataclass(frozen=True)
class ContinuationTrace:
    """Full emitted steps, including batch padding required by strict HF traces."""

    token_ids: tuple[int, ...]
    policy_logprobs: tuple[float, ...]
    raw_logprobs: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ContinuationResult:
    request_id: str
    token_ids: tuple[int, ...]
    stop_reason: Literal["im_end", "length", "forced_eos"]
    trace: ContinuationTrace | None = None

    @property
    def policy_logprobs(self) -> tuple[float, ...] | None:
        return (
            None
            if self.trace is None
            else self.trace.policy_logprobs[: len(self.token_ids)]
        )

    @property
    def raw_logprobs(self) -> tuple[float, ...] | None:
        return (
            None
            if self.trace is None or self.trace.raw_logprobs is None
            else self.trace.raw_logprobs[: len(self.token_ids)]
        )


class _SuffixError(ValueError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def trim_suffix(
    token_ids: Sequence[int],
    *,
    budget: int,
    eos_token_id: int,
    pad_token_id: int,
    allow_pad_tokens: bool = False,
) -> tuple[tuple[int, ...], Literal["im_end", "length"]]:
    """Remove batch-only padding without accepting a truncated or malformed row."""
    ids = _checked_token_ids(token_ids)
    if eos_token_id in ids[:budget]:
        end = ids.index(eos_token_id) + 1
        if end > budget:
            raise _SuffixError(
                "generation exceeds its budget", "hf_backend.sequence_shape"
            )
        if any(i != pad_token_id for i in ids[end:]):
            raise _SuffixError(
                "generation contains tokens after EOS", "hf_backend.post_stop_content"
            )
        if (
            not allow_pad_tokens
            and pad_token_id != eos_token_id
            and pad_token_id in ids[: end - 1]
        ):
            raise _SuffixError(
                "generation contains padding before EOS",
                "hf_backend.unexpected_pad_token",
            )
        return ids[:end], "im_end"
    if len(ids) < budget or any(i != pad_token_id for i in ids[budget:]):
        raise ValueError(
            "generation ended before its budget or contains non-padding after it"
        )
    kept = ids[:budget]
    if not allow_pad_tokens and pad_token_id != eos_token_id and pad_token_id in kept:
        raise _SuffixError(
            "generation contains padding before its budget",
            "hf_backend.unexpected_pad_token",
        )
    return kept, "length"


def _select_inputs(batch: NativeBatch, indices: Sequence[int]) -> dict[str, Any]:
    """Select processor rows and their flattened Qwen image patches together."""
    if tuple(indices) == tuple(range(len(batch.request_ids))):
        return dict(batch.inputs)
    values = batch.inputs
    if "video_grid_thw" in values or "pixel_values_videos" in values:
        raise ValueError("sub-batched video continuation is unsupported")
    grid = values.get("image_grid_thw")
    pixels = values.get("pixel_values")
    selected: dict[str, Any] = {}
    if pixels is not None:
        if not isinstance(grid, torch.Tensor) or grid.shape != (
            len(batch.request_ids),
            3,
        ):
            raise ValueError("image patches require one image grid per request")
        sizes = grid.prod(dim=1).tolist()
        if not isinstance(pixels, torch.Tensor) or pixels.shape[0] != sum(sizes):
            raise ValueError("flattened image patch rows do not match their grids")
        parts = pixels.split([int(size) for size in sizes], dim=0)
        selected["pixel_values"] = torch.cat([parts[i] for i in indices], dim=0)
    for key, value in values.items():
        if key == "pixel_values" or key in _STALE_HISTORY_FIELDS:
            continue
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.shape[0] != len(batch.request_ids):
                raise ValueError(
                    f"cannot safely associate native tensor {key} with requests"
                )
            selected[key] = value[list(indices)]
        else:
            selected[key] = value
    return selected


def generate_continuations(
    model: Any,
    batch: NativeBatch,
    *,
    extensions: Sequence[Sequence[int]],
    budgets: Sequence[int],
    eos_token_id: int,
    pad_token_id: int,
    policy: NativeGenerationPolicy = NativeGenerationPolicy(),
    trace: Literal["none", "policy", "raw_and_policy"] = "none",
    seed: int | None = None,
    allow_pad_tokens: bool = False,
) -> tuple[ContinuationResult, ...]:
    """Continue literal histories; fixed seed and fixed grouping define sampling.

    The caller supplies the maximum *new suffix* count for every request. A
    terminal extension or zero budget performs no model work for that request.
    Repetition-penalty batches use equal history widths so left-padding tokens
    cannot become unintended penalty conditioning.
    """
    if trace not in ("none", "policy", "raw_and_policy"):
        raise ValueError("unsupported generation trace")
    _checked_token_ids((eos_token_id, pad_token_id))
    count = len(batch.request_ids)
    if (
        len(set(batch.request_ids)) != count
        or len(extensions) != count
        or len(budgets) != count
    ):
        raise ValueError(
            "continuations require unique IDs and one extension/budget per request"
        )
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in budgets):
        raise ValueError("continuation budgets must be nonnegative integers")
    if policy.temperature > 0 and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError("sampled continuation requires an explicit nonnegative seed")
    extras = tuple(_checked_token_ids(row) for row in extensions)
    prompts = batch.prompt_token_ids
    if len(prompts) != count:
        raise ValueError("prepared input rows do not match request IDs")
    histories = tuple(
        (*prompt, *extra) for prompt, extra in zip(prompts, extras, strict=True)
    )
    results: dict[int, ContinuationResult] = {}
    active: list[int] = []
    for index, extra in enumerate(extras):
        if eos_token_id in extra[:-1]:
            raise ValueError("literal extension contains tokens after EOS")
        if extra and extra[-1] == eos_token_id:
            results[index] = ContinuationResult(
                batch.request_ids[index], (), "forced_eos"
            )
        elif budgets[index] == 0:
            results[index] = ContinuationResult(batch.request_ids[index], (), "length")
        else:
            active.append(index)
    if not active:
        return tuple(results[i] for i in range(count))
    groups: dict[int, list[int]] = {}
    for index in active:
        key = len(histories[index]) if policy.repetition_penalty != 1 else 0
        groups.setdefault(key, []).append(index)
    if policy.temperature > 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    for indices in groups.values():
        inputs = _select_inputs(batch, indices)
        inputs = move_to_device(
            {
                k: v
                for k, v in inputs.items()
                if k not in _STALE_HISTORY_FIELDS
                and k not in {"return_dict", "use_cache", "logits_to_keep"}
            },
            device=model_device(model),
        )
        ids, mask = padded_histories(
            [histories[i] for i in indices],
            pad_token_id=pad_token_id,
            device=model_device(model),
        )
        inputs.update(input_ids=ids, attention_mask=mask)
        group_budgets = tuple(budgets[i] for i in indices)
        kwargs: dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max(group_budgets),
            do_sample=policy.temperature > 0,
            repetition_penalty=policy.repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            return_dict_in_generate=trace != "none",
            output_scores=trace != "none",
            output_logits=trace == "raw_and_policy",
            output_hidden_states=False,
            output_attentions=False,
        )
        if policy.temperature > 0:
            kwargs.update(temperature=policy.temperature, top_p=policy.top_p)
            if policy.top_k is not None:
                kwargs["top_k"] = policy.top_k
        if not policy.use_model_defaults:
            from transformers import GenerationConfig

            kwargs["generation_config"] = GenerationConfig(
                **{k: v for k, v in kwargs.items() if k not in inputs}
            )
            kwargs["use_model_defaults"] = False
        if len(set(group_budgets)) > 1:
            from transformers import StoppingCriteria, StoppingCriteriaList

            width = ids.shape[1]
            budget_tensor = torch.tensor(group_budgets, device=ids.device)

            class PerRequestBudget(StoppingCriteria):
                def __call__(
                    self, input_ids: torch.Tensor, scores: Any, **_: Any
                ) -> torch.Tensor:
                    return input_ids.shape[1] - width >= budget_tensor

            kwargs["stopping_criteria"] = StoppingCriteriaList([PerRequestBudget()])
        with torch.inference_mode():
            output = model.generate(**kwargs)
        sequences = _require_rank_two_tensor(
            output
            if isinstance(output, torch.Tensor)
            else getattr(output, "sequences", None),
            field="sequences",
        )
        if (
            sequences.shape[0] != len(indices)
            or sequences.shape[1] <= ids.shape[1]
            or not torch.equal(sequences[:, : ids.shape[1]], ids)
        ):
            raise RuntimeContractError(
                "generated sequences do not preserve exact conditioning",
                code="hf_backend.sequence_shape",
            )
        generated = sequences[:, ids.shape[1] :]
        policy_values = raw_values = None
        if trace != "none":
            scores = _require_step_tensors(
                getattr(output, "scores", None), batch_size=len(indices), field="scores"
            )
            if len(scores) != generated.shape[1]:
                raise RuntimeContractError(
                    "generation scores do not align with suffix",
                    code="hf_backend.sequence_shape",
                )
            policy_values = policy_chosen_token_logprobs(
                model=model, sequences=sequences, scores=scores, generated=generated
            )
            if trace == "raw_and_policy":
                raw = _require_step_tensors(
                    getattr(output, "logits", None),
                    batch_size=len(indices),
                    field="logits",
                )
                if len(raw) != len(scores):
                    raise RuntimeContractError(
                        "raw logits differ from policy steps",
                        code="hf_backend.raw_logit_alignment",
                    )
                raw_values = chosen_token_logprobs(raw, generated)
        for row, index in enumerate(indices):
            emitted = tuple(int(token) for token in generated[row].tolist())
            try:
                tokens, stop = trim_suffix(
                    generated[row].tolist(),
                    budget=budgets[index],
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    allow_pad_tokens=allow_pad_tokens,
                )
            except ValueError as exc:
                raise RuntimeContractError(
                    str(exc),
                    code=getattr(exc, "code", "hf_backend.sequence_shape"),
                    context={"request_id": batch.request_ids[index]},
                    cause=exc,
                ) from exc
            length = len(tokens)
            for channel, values in (("policy", policy_values), ("raw", raw_values)):
                if values is not None and not bool(
                    torch.isfinite(values[row, :length]).all()
                ):
                    raise RuntimeContractError(
                        "generation returned a nonfinite selected likelihood",
                        code="qwen.generation_nonfinite_likelihood",
                        context={
                            "request_id": batch.request_ids[index],
                            "channel": channel,
                        },
                    )
            captured_trace = (
                None
                if policy_values is None
                else ContinuationTrace(
                    emitted,
                    tuple(float(x) for x in policy_values[row].tolist()),
                    None
                    if raw_values is None
                    else tuple(float(x) for x in raw_values[row].tolist()),
                )
            )
            results[index] = ContinuationResult(
                batch.request_ids[index], tokens, stop, captured_trace
            )

    return tuple(results[i] for i in range(count))


def _require_step_tensors(
    values: Any,
    *,
    batch_size: int,
    field: str,
) -> tuple[torch.Tensor, ...]:
    if values is None:
        raise RuntimeContractError(
            f"HF generation returned no {field}",
            code=f"hf_backend.missing_{field}",
        )
    tensors = tuple(torch.as_tensor(value) for value in values)
    if not tensors:
        raise RuntimeContractError(
            f"HF generation returned empty {field}",
            code=f"hf_backend.missing_{field}",
        )
    for step_index, tensor in enumerate(tensors):
        if tensor.ndim != 2 or tensor.shape[0] != batch_size:
            raise RuntimeContractError(
                f"HF generation {field} shape does not match the native batch",
                code="hf_backend.step_shape",
                context={
                    "field": field,
                    "step_index": step_index,
                    "shape": tuple(tensor.shape),
                    "batch_size": batch_size,
                },
            )
    return tensors


def chosen_token_logprobs(
    step_logits: Sequence[torch.Tensor] | None,
    generated: torch.Tensor,
) -> torch.Tensor:
    if step_logits is None:
        raise AssertionError("step logits are required")
    columns = []
    for step_index, logits in enumerate(step_logits):
        chosen = generated[:, step_index : step_index + 1].to(logits.device)
        columns.append(F.log_softmax(logits.float(), dim=-1).gather(1, chosen))
    return torch.cat(columns, dim=1)


def policy_chosen_token_logprobs(
    *,
    model: Any,
    sequences: torch.Tensor,
    scores: Sequence[torch.Tensor],
    generated: torch.Tensor,
) -> torch.Tensor:
    compute_transition_scores = getattr(model, "compute_transition_scores", None)
    if callable(compute_transition_scores):
        prompt_width = sequences.shape[1] - len(scores)
        chunks: list[torch.Tensor] = []
        for start in range(0, len(scores), _POLICY_SCORE_STEP_CHUNK_SIZE):
            end = min(start + _POLICY_SCORE_STEP_CHUNK_SIZE, len(scores))
            chunk_scores = tuple(scores[start:end])
            chunk_sequences = sequences[:, : prompt_width + end]
            try:
                values = compute_transition_scores(
                    chunk_sequences,
                    chunk_scores,
                    normalize_logits=True,
                )
            except Exception as exc:
                raise RuntimeContractError(
                    "HF policy likelihood extraction failed",
                    code="hf_backend.policy_logprob_extraction",
                    context={
                        "sequence_shape": tuple(sequences.shape),
                        "score_steps": len(scores),
                        "chunk_start": start,
                        "chunk_end": end,
                    },
                    cause=exc,
                ) from exc
            chunk = torch.as_tensor(values)
            expected_chunk_shape = (generated.shape[0], end - start)
            if chunk.shape != expected_chunk_shape:
                raise RuntimeContractError(
                    "HF policy likelihood chunk does not align with generated ids",
                    code="hf_backend.policy_logprob_alignment",
                    context={
                        "likelihood_shape": tuple(chunk.shape),
                        "expected_shape": expected_chunk_shape,
                        "chunk_start": start,
                        "chunk_end": end,
                    },
                )
            chunks.append(chunk)
        result = torch.cat(chunks, dim=1)
        if result.shape != generated.shape:
            raise RuntimeContractError(
                "HF policy likelihood shape does not align with generated ids",
                code="hf_backend.policy_logprob_alignment",
                context={
                    "likelihood_shape": tuple(result.shape),
                    "generated_shape": tuple(generated.shape),
                },
            )
        return result
    return chosen_token_logprobs(scores, generated)

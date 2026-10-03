"""Opt-in paired selected-action evidence at the native vLLM sampler seam.

Only one pre-policy logits batch is transiently retained. Each sampler call
reduces it on device to the sampler's literal selected actions. Blocking
finalization transfers compact integer IDs and FP32 probability pairs once;
request association uses the live V2 request-slot metadata. Compact rows
include incomplete prefill and scheduler overshoot until finalization; retained
storage scales with finite sampler work, and returned traces with emitted actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from src.qwen.generation import chosen_token_logprobs


_METADATA_KEY = "coordexp_paired_trace"


def assert_trace_idle(model: Any) -> None:
    """Protect direct refresh calls as well as the engine's refresh RPC seam."""
    if getattr(model, "_coordexp_paired_trace_state", None) is not None:
        raise RuntimeError("cannot refresh while a paired trace is active")


def capture_raw_logits(model: Any, logits: torch.Tensor) -> None:
    """Called after output deltas, before any median policy transformation."""
    state = getattr(model, "_coordexp_paired_trace_state", None)
    if state is not None:
        state.capture(logits, getattr(model, "coordexp_dora_identity", None))


def _descriptor(value: dict[str, Any], snapshot_id: str) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or set(value) != {"request_id", "snapshot_id", "max_new_tokens"}
            or not isinstance(value["request_id"], str) or not value["request_id"]
            or value["snapshot_id"] != snapshot_id
            or type(value["max_new_tokens"]) is not int or value["max_new_tokens"] <= 0):
        raise RuntimeError("invalid paired trace request metadata or snapshot identity")
    return value.copy()


def _validate_sampling(params: Any, budget: int) -> None:
    neutral = dict(n=1, top_p=1.0, min_p=0.0, frequency_penalty=0.0,
                   presence_penalty=0.0, repetition_penalty=1.0, min_tokens=0,
                   ignore_eos=False)
    if any(getattr(params, name, None) != value for name, value in neutral.items()):
        raise RuntimeError("unsupported paired trace sampling controls")
    temperature = getattr(params, "temperature", None)
    if (temperature not in (0.0, 1.0) or getattr(params, "top_k", None) not in (-1, 0)
            or getattr(params, "max_tokens", None) != budget
            or type(getattr(params, "logprobs", None)) is not int
            or params.logprobs not in (0, 1)
            or (temperature == 1.0 and type(getattr(params, "seed", None)) is not int)):
        raise RuntimeError("paired trace sampling requires seeded full support, temperature 0/1, and compact logprobs")
    unsupported = ("stop", "bad_words", "_bad_words_token_ids", "allowed_token_ids",
                   "logit_bias", "structured_outputs", "logits_processors",
                   "thinking_token_budget", "repetition_detection", "logprob_token_ids",
                   "prompt_logprobs")
    if any(getattr(params, name, None) for name in unsupported):
        raise RuntimeError("unsupported paired trace sampling processors or stopping controls")
    if any(getattr(params, name, None) is not None for name in (
        "allowed_token_ids", "logit_bias", "structured_outputs", "thinking_token_budget",
        "repetition_detection", "prompt_logprobs",
    )):
        raise RuntimeError("unsupported paired trace sampling processors or stopping controls")


@dataclass
class _TraceBatch:
    request_ids: tuple[str, ...]
    positions: torch.Tensor
    token_ids: torch.Tensor
    policy_token_ids: torch.Tensor
    num_sampled: torch.Tensor
    raw_logprobs: torch.Tensor
    policy_logprobs: torch.Tensor
    vocab_size: int


class _TraceState:
    def __init__(self, runner: Any, model: Any, snapshot_id: str,
                 requests: list[dict[str, Any]]):
        self.runner = runner
        self.model = model
        self.original_sampler = runner.sampler
        self.snapshot_id = snapshot_id
        self.requests: dict[str, dict[str, Any]] = {}
        for value in requests:
            metadata = _descriptor(value, snapshot_id)
            request_id = metadata["request_id"]
            if request_id in self.requests:
                raise RuntimeError("duplicate paired trace request identity")
            self.requests[request_id] = metadata
        if not self.requests:
            raise RuntimeError("paired trace requests must be nonempty")
        self.bindings: dict[int, tuple[str, dict[str, Any]]] = {}
        self.internal_by_caller: dict[str, str] = {}
        self.pending_raw_logits: torch.Tensor | None = None
        self.batches: list[_TraceBatch] = []
        self.failed: str | None = None

    def capture(self, logits: torch.Tensor, snapshot_id: str | None) -> None:
        try:
            if self.failed:
                raise RuntimeError(f"paired trace failed: {self.failed}")
            if snapshot_id != self.snapshot_id:
                raise RuntimeError("raw logits model snapshot identity differs")
            if self.pending_raw_logits is not None:
                raise RuntimeError("paired trace already has a pending raw logits batch")
            if logits.ndim != 2 or not logits.is_floating_point():
                raise RuntimeError("raw logits must be a floating matrix")
            self.pending_raw_logits = logits.detach().clone()
        except Exception as exc:
            self.failed = str(exc)
            self.pending_raw_logits = None
            raise

    def bind(self, req_idx: int, sampling_params: Any) -> tuple[str, dict[str, Any]]:
        if self.failed:
            raise RuntimeError(f"paired trace failed: {self.failed}")
        metadata = _descriptor((getattr(sampling_params, "extra_args", None) or {}).get(_METADATA_KEY),
                               self.snapshot_id)
        request_id = metadata["request_id"]
        if self.requests.get(request_id) != metadata:
            raise RuntimeError("admitted request metadata differs from paired trace identity")
        _validate_sampling(sampling_params, metadata["max_new_tokens"])
        internal_id = self.runner.req_states.index_to_req_id.get(req_idx)
        if not isinstance(internal_id, str) or not internal_id:
            raise RuntimeError("paired trace request slot has no current internal identity")
        previous = self.internal_by_caller.get(request_id)
        if previous is not None and previous != internal_id:
            raise RuntimeError("caller request identity changed its internal request")
        return internal_id, metadata

    def prepare(self, input_batch: Any) -> tuple[tuple[str, ...], torch.Tensor]:
        if self.failed:
            raise RuntimeError(f"paired trace failed: {self.failed}")
        if getattr(self.model, "coordexp_dora_identity", None) != self.snapshot_id:
            raise RuntimeError("sampler model snapshot identity differs")
        count = input_batch.num_reqs
        raw = self.pending_raw_logits
        if raw is None:
            raise RuntimeError("sampler has no pending paired raw logits")
        if (input_batch.num_draft_tokens != 0 or count <= 0
                or raw.shape[0] != count or len(input_batch.req_ids) != count
                or len(input_batch.idx_mapping_np) != count
                or input_batch.idx_mapping.shape != (count,)
                or input_batch.logits_indices.shape != (count,)
                or len(input_batch.cu_num_logits_np) != count + 1
                or any(int(input_batch.cu_num_logits_np[i]) != i for i in range(count + 1))):
            raise RuntimeError("unsupported speculative paired trace batch layout")
        request_ids = []
        for internal_id, slot in zip(input_batch.req_ids, input_batch.idx_mapping_np, strict=True):
            slot = int(slot)
            binding = self.bindings.get(slot)
            if (binding is None or binding[0] != internal_id
                    or self.runner.req_states.index_to_req_id.get(slot) != internal_id):
                raise RuntimeError("paired trace request slot or internal identity is stale")
            request_ids.append(binding[1]["request_id"])
        # These advanced indices are existing GPU tensors. Arithmetic creates
        # fresh storage now, before runner input buffers can be reused.
        positions = (input_batch.positions[input_batch.logits_indices] + 1
                     - self.original_sampler.req_states.prompt_len.gpu[input_batch.idx_mapping])
        return tuple(request_ids), positions.detach()

    def consume(self, request_ids: tuple[str, ...], positions: torch.Tensor, output: Any) -> None:
        raw = self.pending_raw_logits
        count = len(request_ids)
        sampled = output.sampled_token_ids
        compact = output.logprobs_tensors
        num_sampled = output.num_sampled
        if (sampled.shape != (count, 1) or sampled.dtype not in (torch.int32, torch.int64)
                or compact is None or compact.logprob_token_ids.ndim != 2
                or compact.logprob_token_ids.shape[0] != count
                or compact.logprob_token_ids.shape[1] not in (1, 2)
                or compact.logprob_token_ids.dtype not in (torch.int32, torch.int64)
                or compact.logprobs.shape != compact.logprob_token_ids.shape
                or not compact.logprobs.is_floating_point()
                or num_sampled is None or num_sampled.shape != (count,)
                or num_sampled.dtype not in (torch.int32, torch.int64)):
            raise RuntimeError("paired trace sampler tensors have unsupported sampled layout")
        tokens = sampled[:, 0].long().detach().clone()
        # A discarded prefill sample may contain irrelevant IDs/scores. Use a
        # valid scoring ID on device, preserving literal IDs for final evidence.
        # Clamping also defers malformed selected-ID rejection to finalization.
        scoring_ids = torch.where(num_sampled == 0, torch.zeros_like(tokens), tokens).clamp(0, raw.shape[1] - 1)
        raw_scores = chosen_token_logprobs((raw,), scoring_ids[:, None])[:, 0].detach()
        self.batches.append(_TraceBatch(
            request_ids, positions, tokens,
            compact.logprob_token_ids[:, 0].long().detach().clone(),
            num_sampled.long().detach().clone(), raw_scores,
            compact.logprobs[:, 0].float().detach().clone(), raw.shape[1],
        ))

    def finalize(self, snapshot_id: str, outputs: list[dict[str, Any]],
                 eos_token_ids: list[int]) -> dict[str, dict[str, Any]]:
        if self.failed:
            raise RuntimeError(f"paired trace failed: {self.failed}")
        if (snapshot_id != self.snapshot_id
                or getattr(self.model, "coordexp_dora_identity", None) != snapshot_id):
            raise RuntimeError("paired trace final snapshot identity differs")
        if self.pending_raw_logits is not None:
            raise RuntimeError("paired trace finalization has pending raw logits")
        if any(type(token) is not int or token < 0 for token in eos_token_ids):
            raise RuntimeError("invalid paired trace EOS token IDs")
        emitted = {}
        for value in outputs:
            request_id = value["request_id"]
            ids = value["token_ids"]
            if request_id in emitted or request_id not in self.requests:
                raise RuntimeError("final request identity differs or is duplicate")
            if (not isinstance(ids, (list, tuple)) or not ids
                    or len(ids) > self.requests[request_id]["max_new_tokens"]
                    or any(type(token) is not int or token < 0 for token in ids)):
                raise RuntimeError("invalid emitted paired trace token IDs")
            if any(token in eos_token_ids for token in ids[:-1]):
                raise RuntimeError("emitted paired trace tokens continue past EOS")
            emitted[request_id] = list(ids)
        if emitted.keys() != self.requests.keys():
            raise RuntimeError("final paired trace request identities are incomplete")
        records: dict[str, list[tuple[int, int, float, float]]] = {key: [] for key in emitted}
        captured = dict.fromkeys(emitted, 0)
        discarded = dict.fromkeys(emitted, 0)
        overshoot = dict.fromkeys(emitted, 0)
        if self.batches:
            # IDs and positions remain integer, independently of FP32 scores.
            ids = torch.cat([torch.stack((b.token_ids, b.policy_token_ids, b.positions, b.num_sampled), dim=-1)
                             for b in self.batches]).cpu().tolist()
            scores = torch.cat([torch.stack((b.raw_logprobs, b.policy_logprobs), dim=-1)
                                for b in self.batches]).cpu().tolist()
            associations = [(request_id, b.vocab_size) for b in self.batches for request_id in b.request_ids]
            for (request_id, vocab), (token, policy_token, position, num_sampled), (raw, policy) in zip(associations, ids, scores, strict=True):
                captured[request_id] += 1
                # Native num_sampled defines validity; optimistic CPU counters
                # and dummy prefill IDs/probabilities must never override it.
                if num_sampled == 0:
                    discarded[request_id] += 1
                    continue
                if num_sampled != 1:
                    raise RuntimeError("unsupported speculative num_sampled layout")
                if position >= self.requests[request_id]["max_new_tokens"]:
                    overshoot[request_id] += 1
                    continue
                if position < 0:
                    raise RuntimeError("undiscarded paired trace has negative action position")
                if not 0 <= token < vocab or token != policy_token:
                    raise RuntimeError("sampler selected token differs from compact probability token or vocabulary")
                if not math.isfinite(raw) or not math.isfinite(policy):
                    raise RuntimeError("paired trace probability pairs must be finite")
                records[request_id].append((position, token, raw, policy))
        result = {}
        for request_id, tokens in emitted.items():
            rows = sorted(records[request_id])
            if [row[0] for row in rows] != list(range(len(rows))) or len(rows) < len(tokens):
                raise RuntimeError("paired trace action positions are incomplete or duplicate")
            if [row[1] for row in rows[:len(tokens)]] != tokens:
                raise RuntimeError("paired trace selected token IDs differ from emitted tokens")
            excluded = len(rows) - len(tokens)
            if excluded and (not self.runner.scheduler_config.async_scheduling or tokens[-1] not in eos_token_ids):
                raise RuntimeError("extra actions require verified async post-EOS scheduler work")
            result[request_id] = dict(token_ids=tokens,
                raw_logprobs=[row[2] for row in rows[:len(tokens)]],
                policy_logprobs=[row[3] for row in rows[:len(tokens)]],
                excluded_async_suffix=excluded, dropped_budget_actions=overshoot[request_id],
                discarded_prefill_actions=discarded[request_id], captured_rows=captured[request_id])
        return result

    def clear(self) -> None:
        self.pending_raw_logits = None
        self.batches.clear()
        self.bindings.clear()
        self.internal_by_caller.clear()
        self.requests.clear()


class _SamplerTraceAdapter:
    """Scoped observation around V2's plain callable; native output is untouched."""
    def __init__(self, state: _TraceState):
        self._state = state
        self._original = state.original_sampler

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_state", "_original"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._original, name, value)

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> Any:
        state = self._state
        try:
            binding = state.bind(req_idx, sampling_params)
            result = self._original.add_request(req_idx, prompt_len, sampling_params)
            state.bindings[req_idx] = binding
            state.internal_by_caller[binding[1]["request_id"]] = binding[0]
            return result
        except Exception as exc:
            state.failed = str(exc)
            state.pending_raw_logits = None
            raise

    def __call__(self, logits: torch.Tensor, input_batch: Any) -> Any:
        state = self._state
        try:
            request_ids, positions = state.prepare(input_batch)
            output = self._original(logits, input_batch)
            state.consume(request_ids, positions, output)
            return output
        except Exception as exc:
            state.failed = str(exc)
            raise
        finally:
            state.pending_raw_logits = None


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


class CoordExpTraceWorkerExtension:
    """Unique worker RPCs; install the V2 adapter only for an active acquisition."""
    def coordexp_trace_assert_idle(self) -> bool:
        if getattr(self, "_coordexp_trace_state", None) is not None:
            raise RuntimeError("paired trace is active")
        assert_trace_idle(self.get_model())
        return True

    def coordexp_trace_begin(self, snapshot_id: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
        self.coordexp_trace_assert_idle()
        model = self.get_model()
        runner = self.model_runner
        if not isinstance(snapshot_id, str) or not snapshot_id or getattr(model, "coordexp_dora_identity", None) != snapshot_id:
            raise RuntimeError("paired trace begin snapshot identity differs")
        # V2 passes total decode_query_len in this misleadingly named field;
        # ordinary ModelState produces one token with zero speculative steps.
        if runner.speculative_config is not None or runner.sampler.num_speculative_tokens != 1:
            raise RuntimeError("paired traces require nonspeculative single-token decode width")
        if runner.batch_sharder is not None:
            raise RuntimeError("paired traces do not support batch sharded sampling")
        if runner.parallel_config.tensor_parallel_size != 1 or runner.parallel_config.pipeline_parallel_size != 1:
            raise RuntimeError("paired traces require TP=1 and PP=1")
        if runner.sampler.req_states is not runner.req_states:
            raise RuntimeError("paired trace sampler request state differs from runner")
        if runner.sampler.logprobs_mode not in ("raw_logprobs", "processed_logprobs"):
            raise RuntimeError("paired trace sampling requires compact probability output")
        state = _TraceState(runner, model, snapshot_id, requests)
        ack = dict(snapshot_id=snapshot_id, request_ids=list(state.requests),
                   runner_class=_class_name(runner), sampler_class=_class_name(runner.sampler))
        runner.sampler = _SamplerTraceAdapter(state)
        self._coordexp_trace_state = state
        model._coordexp_paired_trace_state = state
        return ack

    def coordexp_trace_abort(self) -> bool:
        state = getattr(self, "_coordexp_trace_state", None)
        if state is not None:
            state.runner.sampler = state.original_sampler
            state.clear()
            if getattr(state.model, "_coordexp_paired_trace_state", None) is state:
                state.model._coordexp_paired_trace_state = None
            self._coordexp_trace_state = None
        return True

    def coordexp_trace_finalize(self, snapshot_id: str, outputs: list[dict[str, Any]],
                               eos_token_ids: list[int]) -> dict[str, dict[str, Any]]:
        state = getattr(self, "_coordexp_trace_state", None)
        if state is None:
            raise RuntimeError("paired trace finalization has no active acquisition")
        try:
            return state.finalize(snapshot_id, outputs, eos_token_ids)
        finally:
            self.coordexp_trace_abort()

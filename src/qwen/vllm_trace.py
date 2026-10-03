"""Opt-in paired selected-action evidence at the native vLLM sampler seam.

Only one pre-policy logits batch is transiently retained. Each sampler call
reduces it on device to the sampler's literal selected actions. Blocking
finalization transfers compact integer IDs and FP32 probability pairs once;
request association always comes from the runner's current CPU bookkeeping.
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
    rows: tuple[tuple[str, int], ...]
    token_ids: torch.Tensor
    policy_token_ids: torch.Tensor
    raw_logprobs: torch.Tensor
    policy_logprobs: torch.Tensor


class _TraceState:
    def __init__(self, runner: Any, model: Any, snapshot_id: str,
                 requests: list[dict[str, Any]]):
        self.runner = runner
        self.model = model
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
        self.pending_raw_logits: torch.Tensor | None = None
        self.batches: list[_TraceBatch] = []
        self.positions = {key: set() for key in self.requests}
        self.discarded_prefill = dict.fromkeys(self.requests, 0)
        self.dropped_budget = dict.fromkeys(self.requests, 0)
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

    def consume(self, sampler: Any, kwargs: dict[str, Any], output: Any) -> None:
        try:
            if output is None:
                self.failed = "native sampler failed"
                return
            if self.failed:
                raise RuntimeError(f"paired trace failed: {self.failed}")
            if getattr(self.model, "coordexp_dora_identity", None) != self.snapshot_id:
                raise RuntimeError("sampler model snapshot identity differs")
            raw = self.pending_raw_logits
            if raw is None:
                raise RuntimeError("sampler has no pending paired raw logits")
            mode = kwargs.get("logprobs_mode_override") or sampler.logprobs_mode
            sampling = kwargs.get("sampling_metadata")
            if mode not in ("raw_logprobs", "processed_logprobs") or getattr(sampling, "max_num_logprobs", None) not in (0, 1):
                raise RuntimeError("paired trace sampling requires compact probability output")
            runner = self.runner
            batch = runner.input_batch
            count = batch.num_reqs
            sampled = output.sampled_token_ids
            compact = output.logprobs_tensors
            if (raw.shape[0] != count or sampled.shape != (count, 1)
                    or sampled.dtype not in (torch.int32, torch.int64) or compact is None
                    or compact.logprob_token_ids.ndim != 2
                    or compact.logprob_token_ids.shape[0] != count
                    or compact.logprob_token_ids.shape[1] not in (1, 2)
                    or compact.logprob_token_ids.dtype not in (torch.int32, torch.int64)
                    or compact.logprobs.shape != compact.logprob_token_ids.shape
                    or not compact.logprobs.is_floating_point()):
                raise RuntimeError("paired trace sampler tensors are not compact, aligned selected actions")
            rows: list[tuple[str, int]] = []
            selected_rows: list[int] = []
            for row, internal_id in enumerate(batch.req_ids):
                request = runner.requests[internal_id]
                params = request.sampling_params
                metadata = (getattr(params, "extra_args", None) or {}).get(_METADATA_KEY)
                metadata = _descriptor(metadata, self.snapshot_id)
                request_id = metadata["request_id"]
                if self.requests.get(request_id) != metadata:
                    raise RuntimeError("live request metadata differs from paired trace identity")
                _validate_sampling(params, metadata["max_new_tokens"])
                if runner.discard_request_mask.np[row]:
                    self.discarded_prefill[request_id] += 1
                    continue
                if request.prompt_token_ids is None:
                    raise RuntimeError("paired trace needs literal prompt token IDs")
                position = (int(batch.num_computed_tokens_cpu[row])
                            + int(runner.num_scheduled_tokens.np[row])
                            - len(request.prompt_token_ids))
                if position >= metadata["max_new_tokens"]:
                    self.dropped_budget[request_id] += 1
                    continue
                if position < 0:
                    raise RuntimeError("undiscarded paired trace has negative action position")
                if position in self.positions[request_id]:
                    raise RuntimeError("duplicate paired trace action position")
                self.positions[request_id].add(position)
                rows.append((request_id, position))
                selected_rows.append(row)
            if not rows:
                return
            # Slice tensors already on the sampling device. Constructing an
            # index tensor from this Python list would block on H2D per step.
            tokens = torch.cat([sampled[row:row + 1, 0] for row in selected_rows]).long().detach()
            raw_scores = chosen_token_logprobs(
                (torch.cat([raw[row:row + 1] for row in selected_rows]),), tokens[:, None]
            )[:, 0].detach()
            self.batches.append(_TraceBatch(
                tuple(rows), tokens,
                torch.cat([compact.logprob_token_ids[row:row + 1, 0]
                           for row in selected_rows]).long().detach(),
                raw_scores,
                torch.cat([compact.logprobs[row:row + 1, 0]
                           for row in selected_rows]).float().detach(),
            ))
        except Exception as exc:
            self.failed = str(exc)
            raise
        finally:
            self.pending_raw_logits = None

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
        if self.batches:
            # IDs stay integer; a large vocabulary ID must never round through FP32.
            ids = torch.cat([torch.stack((b.token_ids, b.policy_token_ids), dim=-1)
                             for b in self.batches]).cpu().tolist()
            scores = torch.cat([torch.stack((b.raw_logprobs, b.policy_logprobs), dim=-1)
                                for b in self.batches]).cpu().tolist()
            associations = [row for b in self.batches for row in b.rows]
            for (request_id, position), (token, policy_token), (raw, policy) in zip(associations, ids, scores, strict=True):
                if token != policy_token:
                    raise RuntimeError("sampler selected token differs from compact probability token")
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
            if excluded and (not self.runner.use_async_scheduling or tokens[-1] not in eos_token_ids):
                raise RuntimeError("extra actions require verified async post-EOS scheduler work")
            result[request_id] = dict(
                token_ids=tokens,
                raw_logprobs=[row[2] for row in rows[:len(tokens)]],
                policy_logprobs=[row[3] for row in rows[:len(tokens)]],
                excluded_async_suffix=excluded,
                dropped_budget_actions=self.dropped_budget[request_id],
                discarded_prefill_actions=self.discarded_prefill[request_id],
            )
        return result

    def clear(self) -> None:
        self.pending_raw_logits = None
        self.batches.clear()
        self.positions.clear()


class CoordExpTraceWorkerExtension:
    """Unique RPC names injected by vLLM's supported worker extension option.

    No Worker method or sampler implementation is replaced. The hook is lazily
    installed after startup/warmup and remains inert when there is no trace.
    """
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
        if getattr(runner, "speculative_config", None) is not None:
            raise RuntimeError("paired traces do not support speculative decoding")
        state = _TraceState(runner, model, snapshot_id, requests)
        if getattr(self, "_coordexp_trace_hook", None) is None:
            def collect(sampler, args, kwargs, output):
                active = getattr(self, "_coordexp_trace_state", None)
                if active is not None:
                    active.consume(sampler, kwargs, output)
            self._coordexp_trace_hook = runner.sampler.register_forward_hook(
                collect, with_kwargs=True, always_call=True,
            )
        self._coordexp_trace_state = state
        model._coordexp_paired_trace_state = state
        return dict(snapshot_id=snapshot_id, request_ids=list(state.requests))

    def coordexp_trace_abort(self) -> bool:
        state = getattr(self, "_coordexp_trace_state", None)
        if state is not None:
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

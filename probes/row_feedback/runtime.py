"""Native Qwen execution for the bounded row-end feedback pilot.

The data contract contains visible tokens only.  This module inserts one
continuous internal slot after every assistant ``BOX_END`` and keeps that slot
out of the visible token stream.  Scientific schedules and loss weights stay
with the pilot owner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
from typing import Any, Iterable, Literal, Mapping, Sequence, cast

import torch

from src.qwen.native import derive_position_ids, model_device, move_to_device


Arm = Literal["S", "F"]
BOX_END = 151649
EOS = 151645
PAD = 0
VISIBLE_BUDGET = 3084
EXPECTED_DORA_TENSORS = 588
EXPECTED_DORA_SCALARS = 18_006_016
ANCHOR_ADAPTER = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-12-native-owner-scale-and-state/scale/training/"
    "full-fixedP-N16-v2/adapter"
)
ACCEPTED_SOURCE_GATE_ROOT = Path("/data/CoordExp/.worktrees/research-probes")


@dataclass(frozen=True)
class FeedbackSourceOverride:
    """Replace one F source while checking its visible boundary identity."""

    source: torch.Tensor
    visible_boundary_index: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ids(value: Sequence[int], *, label: str, nonempty: bool = False) -> tuple[int, ...]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)), label)
    result = tuple(value)
    _require(not nonempty or bool(result), f"{label} must be nonempty")
    _require(
        all(not isinstance(token, bool) and isinstance(token, int) and token >= 0 for token in result),
        f"{label} must contain nonnegative integer token IDs",
    )
    return result


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def prepared_inputs_sha256(inputs: Mapping[str, Any]) -> str:
    """Hash native input keys, tensor metadata, and exact bytes."""

    digest = hashlib.sha256()
    for key in sorted(inputs):
        digest.update(key.encode())
        value = inputs[key]
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().cpu()
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean(dim=-1, keepdim=True).sqrt().to(tensor.dtype)


def feedback_slot_embedding(box_end_embedding: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Apply the frozen coefficient-one, projection-free F slot rule."""

    _require(
        box_end_embedding.ndim == source.ndim == 3
        and box_end_embedding.shape == source.shape
        and box_end_embedding.shape[0:2] == (1, 1),
        "feedback slot requires matching [1, 1, hidden] tensors",
    )
    _require(box_end_embedding.dtype == source.dtype, "feedback source dtype changed")
    source_rms = _rms(source)
    embedding_rms = _rms(box_end_embedding)
    _require(bool(torch.isfinite(source_rms).all()) and bool((source_rms > 0).all()),
             "feedback source has zero or nonfinite RMS")
    _require(bool(torch.isfinite(embedding_rms).all()) and bool((embedding_rms > 0).all()),
             "box-end embedding has zero or nonfinite RMS")
    return box_end_embedding + embedding_rms * source / source_rms


def _native_inputs(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, Any], value))
    inputs = getattr(value, "inputs", None)
    _require(isinstance(inputs, Mapping), "inputs must be a native mapping or NativeBatch")
    return dict(cast(Mapping[str, Any], inputs))


def _final_norm(model: torch.nn.Module) -> torch.nn.Module:
    matches = [module for name, module in model.named_modules() if name.endswith("language_model.norm")]
    unique = {id(module): module for module in matches}
    _require(len(unique) == 1, "model must expose exactly one final language_model.norm")
    return next(iter(unique.values()))


def _first_language_k_projection(model: torch.nn.Module) -> torch.nn.Module:
    suffix = "language_model.layers.0.self_attn.k_proj"
    matches = [module for name, module in model.named_modules() if name.endswith(suffix)]
    unique = {id(module): module for module in matches}
    _require(len(unique) == 1, "model must expose exactly one first-language-layer k_proj")
    return next(iter(unique.values()))


@contextmanager
def _capture_postnorm(model: torch.nn.Module):
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
        _require(isinstance(output, torch.Tensor) and output.ndim == 3,
                 "final post-norm output must be [batch, sequence, hidden]")
        captured.append(output)

    handle = _final_norm(model).register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def _capture_native_history_projection(model: torch.nn.Module, *, enabled: bool):
    captured: list[torch.Tensor] = []
    handle = None
    if enabled:
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            _require(isinstance(output, torch.Tensor) and output.ndim == 3,
                     "first-layer K projection must be [batch, sequence, hidden]")
            captured.append(output)

        handle = _first_language_k_projection(model).register_forward_hook(hook)
    try:
        yield captured
    finally:
        if handle is not None:
            handle.remove()


@contextmanager
def _count_image_forwards(model: torch.nn.Module, *, required: bool):
    count = [0]
    matches = [module for name, module in model.named_modules() if name.endswith("visual")]
    unique = {id(module): module for module in matches}
    if required:
        _require(len(unique) == 1, "native multimodal input requires exactly one visual module")
    handles = [module.register_forward_pre_hook(lambda *_: count.__setitem__(0, count[0] + 1))
               for module in unique.values()]
    try:
        yield count
    finally:
        for handle in handles:
            handle.remove()


def _position_row(position_ids: torch.Tensor, index: int) -> list[int]:
    return [int(value) for value in position_ids[:, 0, index].tolist()]


def _spans_ending_at_box_end(token_ids: Sequence[int]) -> Iterable[tuple[int, ...]]:
    start = 0
    for index, token in enumerate(token_ids):
        if token == BOX_END:
            yield tuple(token_ids[start:index + 1])
            start = index + 1
    if start != len(token_ids):
        yield tuple(token_ids[start:])


def _normalize_override(
    value: torch.Tensor | FeedbackSourceOverride,
    *,
    source: torch.Tensor,
    visible_boundary_index: int,
) -> torch.Tensor:
    if isinstance(value, FeedbackSourceOverride):
        _require(value.visible_boundary_index == visible_boundary_index,
                 "feedback override visible boundary mismatch")
        value = value.source
    _require(isinstance(value, torch.Tensor), "feedback override must be a tensor")
    candidate = value.to(device=source.device, dtype=source.dtype)
    if candidate.shape == source.shape[-1:]:
        candidate = candidate.view(1, 1, -1)
    elif candidate.shape == source.shape[1:]:
        candidate = candidate.unsqueeze(0)
    _require(candidate.shape == source.shape, "feedback override hidden shape mismatch")
    _require(bool(torch.isfinite(candidate).all()), "feedback override is nonfinite")
    return candidate


class _CausalRunner:
    """One single-row graph-retaining cached execution."""

    def __init__(
        self,
        qwen: Any,
        inputs: Any,
        *,
        prompt_ids: Sequence[int],
        physical_suffix_capacity: int,
        arm: Arm,
        feedback_source_overrides: Mapping[int, torch.Tensor | FeedbackSourceOverride] | None,
        capture_feedback_sources: bool,
        detach_feedback_source_boundaries: Iterable[int],
        capture_native_history_activation: bool,
    ) -> None:
        _require(arm in ("S", "F"), "arm must be S or F")
        self.qwen = qwen
        self.model = qwen.model
        self.arm = arm
        self.prompt_ids = _ids(prompt_ids, label="prompt_ids", nonempty=True)
        self.inputs = move_to_device(_native_inputs(inputs), device=model_device(self.model))
        self.overrides = dict(feedback_source_overrides or {})
        self.detach_boundaries = frozenset(detach_feedback_source_boundaries)
        _require(all(not isinstance(key, bool) and isinstance(key, int) and key >= 0
                     for key in self.overrides), "feedback override keys must be nonnegative integers")
        _require(all(not isinstance(key, bool) and isinstance(key, int) and key >= 0
                     for key in self.detach_boundaries), "detach boundary keys must be nonnegative integers")
        _require(not set(self.overrides).intersection(self.detach_boundaries),
                 "one feedback boundary cannot be overridden and detached")
        if arm == "S":
            _require(not self.overrides, "S arm rejects feedback source overrides")
            _require(not self.detach_boundaries, "S arm has no feedback source to detach")
        self.capture_feedback_sources = capture_feedback_sources
        self.capture_native_history_activation = capture_native_history_activation
        self.native_history_activation: torch.Tensor | None = None
        self.feedback_sources: dict[int, torch.Tensor] = {}
        self.feedback_boundaries: list[dict[str, Any]] = []
        self.physical_trace: list[dict[str, Any]] = []
        self.slot_work = {"prefill": 0, "history": 0, "target": 0, "generated": 0, "total": 0}
        self.used_overrides: set[int] = set()
        self.used_detaches: set[int] = set()
        self.model_forwards = 0
        self.visible_offset = 0
        self.boundary_index = 0
        self.physical_position = len(self.prompt_ids)
        self.past_key_values: Any = None
        self.current_logits: torch.Tensor | None = None

        ids = self.inputs.get("input_ids")
        mask = self.inputs.get("attention_mask")
        grid = self.inputs.get("image_grid_thw")
        _require(isinstance(ids, torch.Tensor) and ids.ndim == 2 and ids.shape[0] == 1,
                 "native inputs require one input_ids row")
        _require(isinstance(mask, torch.Tensor) and mask.shape == ids.shape
                 and bool((mask == 1).all()), "native input must be one unpadded row")
        _require(ids[0].tolist() == list(self.prompt_ids), "native input IDs differ from prompt_ids")
        _require(isinstance(grid, torch.Tensor) and grid.ndim == 2 and grid.shape[1] == 3,
                 "native inputs require image_grid_thw")
        _require(physical_suffix_capacity >= 0, "physical suffix capacity must be nonnegative")
        filler = [BOX_END] * physical_suffix_capacity
        physical_ids = torch.tensor([[*self.prompt_ids, *filler]], dtype=torch.long,
                                    device=model_device(self.model))
        physical_mask = torch.ones_like(physical_ids)
        self.position_ids = derive_position_ids(
            model=self.model,
            input_ids=physical_ids,
            attention_mask=physical_mask,
            image_grid_thw=grid,
            video_grid_thw=self.inputs.get("video_grid_thw"),
        )
        _require(self.position_ids.shape == (3, 1, physical_ids.shape[1]),
                 "Qwen MRoPE plan shape changed")
        self.full_attention_mask = physical_mask
        self.embedding = self.model.get_input_embeddings()

    def _forward(
        self,
        captured: list[torch.Tensor],
        native_captured: list[torch.Tensor],
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        start: int,
        end: int,
        prefill: bool = False,
    ) -> tuple[Any, torch.Tensor, torch.Tensor | None]:
        _require((input_ids is None) != (inputs_embeds is None), "exactly one model input form")
        before = len(captured)
        native_before = len(native_captured)
        if prefill:
            kwargs: dict[str, Any] = {key: value for key, value in self.inputs.items() if key not in {
                "position_ids", "cache_position", "past_key_values", "inputs_embeds",
                "return_dict", "use_cache", "logits_to_keep",
            }}
            kwargs["input_ids"] = input_ids
        else:
            kwargs = {"input_ids": input_ids} if input_ids is not None else {"inputs_embeds": inputs_embeds}
        kwargs.update(
            attention_mask=self.full_attention_mask[:, :end],
            position_ids=self.position_ids[:, :, start:end],
            cache_position=torch.arange(start, end, device=model_device(self.model)),
            past_key_values=None if prefill else self.past_key_values,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1 if prefill or end - start == 1 else 0,
        )
        output = self.model(**kwargs)
        self.model_forwards += 1
        _require(len(captured) == before + 1, "final post-norm hook did not fire exactly once")
        hidden = captured.pop()
        native_hidden = None
        if self.capture_native_history_activation:
            _require(len(native_captured) == native_before + 1,
                     "first-layer K projection hook did not fire exactly once")
            native_hidden = native_captured.pop()
        expected_logits = 1 if kwargs["logits_to_keep"] == 1 else end - start
        _require(isinstance(output.logits, torch.Tensor)
                 and output.logits.shape[0:2] == (1, expected_logits),
                 "model logits do not match physical forward rows")
        self.past_key_values = output.past_key_values
        _require(self.past_key_values is not None
                 and self.past_key_values.get_seq_length() == end,
                 "cache length differs from physical token count")
        return output, hidden[:, -1:, :], native_hidden

    def prefill(self, captured: list[torch.Tensor], native_captured: list[torch.Tensor]) -> None:
        ids = self.inputs["input_ids"]
        output, _, _ = self._forward(
            captured, native_captured, input_ids=ids,
            start=0, end=len(self.prompt_ids), prefill=True,
        )
        self.current_logits = output.logits[0, -1]

    def _slot(
        self,
        captured: list[torch.Tensor],
        native_captured: list[torch.Tensor],
        source: torch.Tensor,
        *,
        phase: str,
    ) -> None:
        boundary = self.boundary_index
        visible_boundary = self.visible_offset - 1
        override_applied = boundary in self.overrides
        detach_applied = boundary in self.detach_boundaries
        effective_source = source
        if override_applied:
            effective_source = _normalize_override(
                self.overrides[boundary], source=source,
                visible_boundary_index=visible_boundary,
            )
            self.used_overrides.add(boundary)
        if detach_applied:
            effective_source = effective_source.detach()
            self.used_detaches.add(boundary)
        token = torch.tensor([[BOX_END]], dtype=torch.long, device=model_device(self.model))
        box_end_embedding = self.embedding(token)
        if self.arm == "F":
            if self.capture_feedback_sources and source.requires_grad:
                source.retain_grad()
            if self.capture_feedback_sources:
                self.feedback_sources[boundary] = source
            slot = feedback_slot_embedding(box_end_embedding, effective_source)
            source_meta = effective_source
        else:
            slot = box_end_embedding
            source_meta = None
        position = self.physical_position
        output, _, _ = self._forward(
            captured, native_captured, inputs_embeds=slot,
            start=position, end=position + 1,
        )
        self.current_logits = output.logits[0, -1]
        metadata = {
            "boundary_index": boundary,
            "visible_boundary_index": visible_boundary,
            "physical_slot_position": position,
            "source_shape": None if source_meta is None else list(source_meta.shape[-1:]),
            "source_dtype": None if source_meta is None else str(source_meta.dtype),
            "source_sha256": None if source_meta is None else _tensor_sha256(source_meta),
            "source_rms": None if source_meta is None else float(_rms(source_meta).detach()),
            "native_source_sha256": None if self.arm == "S" else _tensor_sha256(source),
            "override_applied": override_applied,
            "detach_applied": detach_applied,
        }
        self.feedback_boundaries.append(metadata)
        self.physical_trace.append({
            "kind": "internal_slot", "phase": phase,
            "physical_position": position,
            "mrope_position": _position_row(self.position_ids, position),
            "boundary_index": boundary,
        })
        self.physical_position += 1
        self.boundary_index += 1
        self.slot_work[phase] += 1
        self.slot_work["total"] += 1

    def consume_visible(
        self,
        captured: list[torch.Tensor],
        native_captured: list[torch.Tensor],
        token_ids: Sequence[int],
        *,
        phase: Literal["history", "target", "generated"],
        collect_aligned_logits: bool,
    ) -> list[torch.Tensor]:
        tokens = _ids(token_ids, label=f"{phase}_ids")
        aligned: list[torch.Tensor] = []
        for span in _spans_ending_at_box_end(tokens):
            current_logits = self.current_logits
            if current_logits is None:
                raise ValueError("prefill must precede visible replay")
            if collect_aligned_logits:
                aligned.append(current_logits)
            start = self.physical_position
            end = start + len(span)
            ids = torch.tensor([span], dtype=torch.long, device=model_device(self.model))
            output, source, native_hidden = self._forward(
                captured, native_captured, input_ids=ids, start=start, end=end,
            )
            if phase == "history" and native_hidden is not None:
                self.native_history_activation = native_hidden
            if collect_aligned_logits and len(span) > 1:
                aligned.extend(output.logits[0, :-1].unbind(0))
            self.current_logits = output.logits[0, -1]
            for offset, token in enumerate(span):
                position = start + offset
                self.physical_trace.append({
                    "kind": "visible", "phase": phase, "token_id": token,
                    "visible_index": self.visible_offset + offset,
                    "physical_position": position,
                    "mrope_position": _position_row(self.position_ids, position),
                })
            self.visible_offset += len(span)
            self.physical_position = end
            if span and span[-1] == BOX_END:
                self._slot(captured, native_captured, source, phase=phase)
        return aligned

    def finish(self) -> None:
        _require(self.used_overrides == set(self.overrides),
                 "requested feedback override boundary was not reached")
        _require(self.used_detaches == set(self.detach_boundaries),
                 "requested detach boundary was not reached")
        _require(self.boundary_index == self.slot_work["total"], "slot counter mismatch")


def replay_visible(
    qwen: Any,
    inputs: Any,
    *,
    prompt_ids: Sequence[int],
    history_ids: Sequence[int],
    target_ids: Sequence[int],
    arm: Arm,
    feedback_source_overrides: Mapping[int, torch.Tensor | FeedbackSourceOverride] | None = None,
    capture_feedback_sources: bool = False,
    detach_feedback_source_boundaries: Iterable[int] = (),
    capture_native_history_activation: bool = False,
) -> dict[str, Any]:
    """Replay visible history/targets with causally constructed internal slots.

    Returned logits are aligned one-for-one with ``target_ids``.  The cache and
    every native-history and F-source path remain attached to the current graph.
    """

    history = _ids(history_ids, label="history_ids")
    targets = _ids(target_ids, label="target_ids", nonempty=True)
    physical = len(history) + history.count(BOX_END) + len(targets) + targets.count(BOX_END)
    started = time.monotonic()
    runner = _CausalRunner(
        qwen, inputs, prompt_ids=prompt_ids, physical_suffix_capacity=physical, arm=arm,
        feedback_source_overrides=feedback_source_overrides,
        capture_feedback_sources=capture_feedback_sources,
        detach_feedback_source_boundaries=detach_feedback_source_boundaries,
        capture_native_history_activation=capture_native_history_activation,
    )
    has_pixels = any(key in runner.inputs for key in ("pixel_values", "pixel_values_videos"))
    with _capture_postnorm(runner.model) as captured, _capture_native_history_projection(
        runner.model, enabled=capture_native_history_activation,
    ) as native_captured, _count_image_forwards(
        runner.model, required=has_pixels,
    ) as image_forwards:
        runner.prefill(captured, native_captured)
        runner.consume_visible(
            captured, native_captured, history,
            phase="history", collect_aligned_logits=False,
        )
        if capture_native_history_activation:
            native_history_activation = runner.native_history_activation
            _require(native_history_activation is not None and bool(history),
                     "native history activation capture requires nonempty history")
            if native_history_activation is None:
                raise AssertionError("native history activation vanished after validation")
            if native_history_activation.requires_grad:
                native_history_activation.retain_grad()
        aligned = runner.consume_visible(
            captured, native_captured, targets,
            phase="target", collect_aligned_logits=True,
        )
    runner.finish()
    _require(len(aligned) == len(targets), "visible target/logit alignment changed")
    logits = torch.stack(aligned)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=logits.device)
    return {
        "arm": arm,
        "logits": logits,
        "target_ids": target_tensor,
        "visible_target_tokens": len(targets),
        "internal_slot_count": runner.slot_work["total"],
        "physical_token_count": runner.physical_position,
        "model_forwards": runner.model_forwards,
        "image_forwards": image_forwards[0],
        "slot_work": dict(runner.slot_work),
        "physical_trace": runner.physical_trace,
        "feedback_boundaries": runner.feedback_boundaries,
        "feedback_sources": runner.feedback_sources,
        "native_history_activation": runner.native_history_activation,
        "timing": {"wall_seconds": time.monotonic() - started},
    }


def generate_visible(
    qwen: Any,
    inputs: Any,
    *,
    prompt_ids: Sequence[int],
    history_ids: Sequence[int] = (),
    arm: Arm,
    max_visible_tokens: int = VISIBLE_BUDGET,
    eos_token_id: int = EOS,
    feedback_source_overrides: Mapping[int, torch.Tensor | FeedbackSourceOverride] | None = None,
    capture_feedback_sources: bool = False,
) -> dict[str, Any]:
    """Greedily generate visible tokens while counting internal physical work."""

    _require(not isinstance(max_visible_tokens, bool) and isinstance(max_visible_tokens, int)
             and max_visible_tokens >= 0, "max_visible_tokens must be nonnegative")
    _require(not isinstance(eos_token_id, bool) and isinstance(eos_token_id, int)
             and eos_token_id >= 0, "eos_token_id must be a token ID")
    history = _ids(history_ids, label="history_ids")
    capacity = len(history) + history.count(BOX_END) + 2 * max_visible_tokens
    started = time.monotonic()
    runner = _CausalRunner(
        qwen, inputs, prompt_ids=prompt_ids, physical_suffix_capacity=capacity, arm=arm,
        feedback_source_overrides=feedback_source_overrides,
        capture_feedback_sources=capture_feedback_sources,
        detach_feedback_source_boundaries=(),
        capture_native_history_activation=False,
    )
    has_pixels = any(key in runner.inputs for key in ("pixel_values", "pixel_values_videos"))
    visible: list[int] = []
    with torch.inference_mode(), _capture_postnorm(runner.model) as captured, _count_image_forwards(
        runner.model, required=has_pixels,
    ) as image_forwards:
        runner.prefill(captured, [])
        runner.consume_visible(captured, [], history, phase="history", collect_aligned_logits=False)
        for _ in range(max_visible_tokens):
            current_logits = runner.current_logits
            if current_logits is None:
                raise ValueError("generation lost predictive logits")
            token = int(current_logits.argmax(dim=-1))
            visible.append(token)
            runner.consume_visible(
                captured, [], (token,), phase="generated", collect_aligned_logits=False,
            )
            if token == eos_token_id:
                break
    runner.finish()
    eos = bool(visible and visible[-1] == eos_token_id)
    finish_reason = "eos" if eos else "length"
    text = qwen.tokenizer.decode(visible, skip_special_tokens=False)
    return {
        "arm": arm,
        "visible_token_ids": visible,
        "text": text,
        "finish_reason": finish_reason,
        "eos": eos,
        "cap": not eos,
        "visible_generated_tokens": len(visible),
        "internal_slot_count": runner.slot_work["total"],
        "physical_token_count": runner.physical_position,
        "model_forwards": runner.model_forwards,
        "image_forwards": image_forwards[0],
        "slot_work": {
            "prefill": runner.slot_work["prefill"],
            "history": runner.slot_work["history"],
            "generated": runner.slot_work["generated"],
            "total": runner.slot_work["total"],
        },
        "physical_trace": runner.physical_trace,
        "feedback_boundaries": runner.feedback_boundaries,
        "feedback_sources": runner.feedback_sources,
        "timing": {"wall_seconds": time.monotonic() - started},
        "decode_contract": {
            "max_visible_tokens": max_visible_tokens,
            "eos_token_id": eos_token_id,
            "do_sample": False,
            "temperature": 0,
            "top_p": 1,
            "repetition_penalty": 1,
        },
    }


def visible_nll_sum(replay: Mapping[str, Any]) -> torch.Tensor:
    """Return the arm-neutral sum of visible next-token negative log likelihoods."""

    from src.losses import aligned_token_logprobs

    logits_value, targets_value = replay.get("logits"), replay.get("target_ids")
    _require(isinstance(logits_value, torch.Tensor) and logits_value.ndim == 2
             and isinstance(targets_value, torch.Tensor) and targets_value.ndim == 1
             and logits_value.shape[0] == targets_value.numel(),
             "replay target/logit shape mismatch")
    logits = cast(torch.Tensor, logits_value)
    targets = cast(torch.Tensor, targets_value)
    return -aligned_token_logprobs(logits, targets).sum()


def mapped_teacher_kl(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    reduction: Literal["sum", "mean"] = "sum",
) -> torch.Tensor:
    """KL over corresponding visible target ordinals across two protocols."""

    _require(student_logits.ndim == teacher_log_probs.ndim == 2
             and student_logits.shape == teacher_log_probs.shape,
             "teacher/student visible distributions must have identical [target, vocab] shape")
    _require(student_logits.dtype == teacher_log_probs.dtype == torch.float32,
             "teacher protection requires FP32 visible distributions")
    _require(reduction in ("sum", "mean"), "unsupported teacher KL reduction")
    _require(not teacher_log_probs.requires_grad, "native teacher distribution must be detached")
    teacher_probs = teacher_log_probs.exp()
    row_mass = teacher_probs.sum(dim=-1)
    _require(bool(torch.isfinite(teacher_probs).all())
             and bool(torch.allclose(row_mass, torch.ones_like(row_mass), atol=1e-5, rtol=1e-5)),
             "teacher log probabilities are not normalized")
    values = (teacher_probs * (teacher_log_probs - student_logits.log_softmax(dim=-1))).sum(dim=-1)
    return values.sum() if reduction == "sum" else values.mean()


def bind_language_dora(model: torch.nn.Module) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], tuple[tuple[str, torch.nn.Parameter], ...]]:
    """Select the exact inherited N16 language-only DoRA surface."""

    from probes.dora_owner_learning.runtime import bind_source256_language_dora

    return bind_source256_language_dora(
        model,
        expected_tensor_count=EXPECTED_DORA_TENSORS,
        expected_scalar_count=EXPECTED_DORA_SCALARS,
    )


def embedding_source_gate_receipt(
    root: str | Path = ACCEPTED_SOURCE_GATE_ROOT,
) -> Mapping[str, Any]:
    """Bind the existing passed gate; never synthesize or bypass its evidence."""

    from src.qwen.special_token_embeddings import (
        DEFAULT_SPECIAL_TOKEN_EMBEDDING_PROBE_RECEIPT_PATH,
        DEFAULT_SPECIAL_TOKEN_EMBEDDING_SOURCE_STUDY_PATH,
        load_default_special_token_embedding_source_gate_evidence,
    )

    resolved = Path(root).resolve()
    evidence = load_default_special_token_embedding_source_gate_evidence(resolved)
    _require(evidence.source_study_passed and evidence.roundtrip_probe_passed
             and evidence.probe_receipt is not None
             and evidence.probe_receipt.get("ok") is True,
             "explicit embedding source gate has not passed")
    study = resolved / DEFAULT_SPECIAL_TOKEN_EMBEDDING_SOURCE_STUDY_PATH
    probe = resolved / DEFAULT_SPECIAL_TOKEN_EMBEDDING_PROBE_RECEIPT_PATH
    _require(study.is_file() and probe.is_file(), "embedding source-gate files are missing")
    return {
        "root": str(resolved),
        "source_study": {"path": str(study), "sha256": _file_sha256(study)},
        "roundtrip_probe": {"path": str(probe), "sha256": _file_sha256(probe)},
        "source_study_passed": True,
        "roundtrip_probe_passed": True,
        "probe_receipt_ok": True,
    }


def load_feedback_policy(
    *, adapter_path: str | Path, device: torch.device,
    source_gate_root: str | Path = ACCEPTED_SOURCE_GATE_ROOT,
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    """Compose the existing FP32/SDPA policy with an explicit adapter."""

    from src.config.fingerprint import sha256_json
    from src.config.inference import load_research_infer_config
    from src.inference.runtime import assemble_frontend
    from probes.dora_owner_learning.route_access import CONFIG, checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy

    base = load_research_infer_config(CONFIG).config
    config = checkpoint_config(base, str(Path(adapter_path).resolve()))
    gate = Path(source_gate_root).resolve()
    _require(gate.is_dir(), "embedding source-gate root is missing")
    embedding_delta_config = config.embedding_delta
    _require(embedding_delta_config is not None, "feedback policy requires embedding delta")
    if embedding_delta_config is None:
        raise AssertionError("embedding delta vanished after validation")
    embedding_delta = embedding_delta_config.model_copy(
        update={"source_gate_root": str(gate)}, deep=True,
    )
    config = config.model_copy(update={"embedding_delta": embedding_delta}, deep=True)
    backend_hf = getattr(config.backend, "hf", None)
    _require(config.model.dtype == "fp32"
             and backend_hf is not None
             and backend_hf.attn_implementation == "sdpa"
             and config.generation.max_new_tokens == VISIBLE_BUDGET
             and config.generation.temperature == 0
             and config.generation.top_p == 1
             and config.generation.repetition_penalty == 1,
             "feedback policy requires the frozen native FP32/SDPA decode contract")
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
    )
    qwen, identity = load_policy(config, device=device)
    return qwen, frontend, config, identity


def _materialization_raw_sources(
    bank: Mapping[str, Any], config: Any,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Load the exact N16 train/dev union once and prove full bank coverage."""

    from probes.parallel_owner_research.training import _load_materialization_raw

    reference = bank["sources"]["native_n16_training_input"]
    _require(_file_sha256(reference["path"]) == reference["sha256"],
             "bank materialization source packet changed")
    packet = json.loads(Path(reference["path"]).read_text())
    raw = _load_materialization_raw(packet, config_input=config.data.input_jsonl)
    expected = {str(record["example_id"]) for record in bank["records"]}
    _require(expected.issubset(raw), "N16 train/dev union does not cover the feedback bank")
    return reference, raw


def materialize_record(
    qwen: Any,
    frontend: Any,
    config: Any,
    record: Mapping[str, Any],
    *,
    raw_by_example: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare one bound bank record through the existing native frontend."""

    from src.data import load_raw_examples
    from src.qwen.native import prepare_native_inputs
    from probes.dora_owner_learning.runtime import build_request

    expected_id = str(record["example_id"])
    if raw_by_example is None:
        matches = [raw for raw in load_raw_examples(config.data.input_jsonl)
                   if str(raw.example_id) == expected_id]
        _require(len(matches) == 1, "bound record is absent or duplicated in configured raw data")
        raw = matches[0]
    else:
        _require(expected_id in raw_by_example, "bound record is absent from explicit raw source union")
        raw = raw_by_example[expected_id]
    request, image, _ = build_request(
        raw, config=config, qwen=frontend.qwen, row_index=int(record["image"]["row_index"]),
    )
    batch = prepare_native_inputs(
        qwen.processor, (request,), device="cpu", record_media_identity=True,
    )
    prompt_ids = list(batch.prompt_token_ids[0])
    expected_prompt = list(record["prompt_token_ids"])
    _require(prompt_ids == expected_prompt
             and _json_sha256(prompt_ids) == record["prompt_token_ids_sha256"],
             "live prompt token identity differs from record")
    planned_path = Path(image.image_path).resolve()
    expected_path = Path(record["image"]["image_path"]).resolve()
    _require(planned_path == expected_path
             and image.image_content_sha256 == record["image"]["image_sha256"],
             "live image identity differs from record")
    observed_grid = batch.image_grids[0]
    observed_media = None if batch.media_sha256 is None else batch.media_sha256[0]
    _require(observed_grid is not None
             and list(observed_grid) == list(record["image"]["observed_image_grid_thw"])
             and observed_media == record["image"]["executed_media_sha256"],
             "live image grid/media identity differs from record")
    inputs = dict(batch.inputs)
    return {
        "inputs": inputs,
        "prompt_ids": prompt_ids,
        "prepared_inputs_sha256": prepared_inputs_sha256(inputs),
        "materialization": {
            "example_id": expected_id,
            "row_index": int(record["image"]["row_index"]),
            "prompt_token_ids_sha256": _json_sha256(prompt_ids),
            "image_path": str(planned_path),
            "image_sha256": image.image_content_sha256,
            "executed_media_sha256": observed_media,
            "observed_image_grid_thw": list(cast(tuple[int, int, int], observed_grid)),
        },
    }


def materialize_bank_records(
    qwen: Any, frontend: Any, config: Any, bank: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Materialize every feedback record once from its exact bound train/dev union."""

    source_reference, raw = _materialization_raw_sources(bank, config)
    result = {
        str(record["record_id"]): materialize_record(
            qwen, frontend, config, record, raw_by_example=raw,
        )
        for record in bank["records"]
    }
    _require(len(result) == len(bank["records"]), "duplicate bank record ID during materialization")
    for value in result.values():
        value["materialization"]["raw_source_packet"] = dict(source_reference)
    return result


def _bound_endpoint_raw(
    selection_record: Mapping[str, Any], native_source: Mapping[str, Any],
) -> Any:
    """Recover one embedded native row from its exact bound JSONL ordinal."""

    from src.data.examples import raw_example_from_jsonl_row

    path = Path(native_source["path"]).resolve()
    _require(path.is_file() and _file_sha256(path) == native_source["sha256"],
             "endpoint native source binding changed")
    row_number_value = selection_record.get("native_row_number")
    _require(not isinstance(row_number_value, bool) and isinstance(row_number_value, int)
             and row_number_value > 0, "endpoint native row number is invalid")
    row_number = cast(int, row_number_value)
    raw_line = None
    with path.open() as stream:
        for index, line in enumerate(stream, start=1):
            if index == row_number:
                raw_line = line.rstrip("\n")
                break
    _require(raw_line is not None and bool(raw_line.strip()), "endpoint native row is missing")
    if raw_line is None:
        raise AssertionError("endpoint native row vanished after validation")
    _require(hashlib.sha256(raw_line.encode()).hexdigest()
             == selection_record["native_record_sha256"],
             "endpoint native row bytes changed")
    payload = json.loads(raw_line)
    _require(payload == selection_record["native_record"], "embedded endpoint native row changed")
    return raw_example_from_jsonl_row(
        payload, jsonl_path=path, row_number=row_number, raw_line=raw_line,
    )


def materialize_endpoint_record(
    qwen: Any,
    frontend: Any,
    config: Any,
    selection_record: Mapping[str, Any],
    *,
    native_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare one frozen original/native endpoint row with empty visible history."""

    from src.qwen.native import prepare_native_inputs
    from probes.dora_owner_learning.runtime import build_request

    raw = _bound_endpoint_raw(selection_record, native_source)
    _require(str(raw.example_id) == str(selection_record["example_id"]),
             "endpoint raw/example identity changed")
    request, image, _ = build_request(raw, config=config, qwen=frontend.qwen, row_index=0)
    batch = prepare_native_inputs(
        qwen.processor, (request,), device="cpu", record_media_identity=True,
    )
    prompt_ids = list(batch.prompt_token_ids[0])
    image_path = Path(image.image_path).resolve()
    _require(image_path == Path(selection_record["image_path"]).resolve()
             and image.image_content_sha256 == selection_record["image_sha256"],
             "endpoint planned image identity changed")
    observed_grid = batch.image_grids[0]
    observed_media = None if batch.media_sha256 is None else batch.media_sha256[0]
    _require(observed_grid is not None and isinstance(observed_media, str) and bool(observed_media),
             "endpoint executed media/grid identity changed")
    inputs = dict(batch.inputs)
    return {
        "inputs": inputs,
        "prompt_ids": prompt_ids,
        "prepared_inputs_sha256": prepared_inputs_sha256(inputs),
        "materialization": {
            "example_id": str(raw.example_id),
            "native_source": dict(native_source),
            "native_row_number": selection_record["native_row_number"],
            "native_record_sha256": selection_record["native_record_sha256"],
            "prompt_token_ids_sha256": _json_sha256(prompt_ids),
            "image_path": str(image_path),
            "image_sha256": image.image_content_sha256,
            "executed_media_sha256": observed_media,
            "observed_image_grid_thw": list(cast(tuple[int, int, int], observed_grid)),
            "visible_history_token_ids": [],
        },
    }


def save_feedback_adapter(qwen: Any, *, source_adapter: str | Path, output: str | Path) -> Mapping[str, Any]:
    """Save the unchanged adapter payload class through the existing owner."""

    from probes.dora_owner_learning.train import _save_adapter_only

    return _save_adapter_only(
        qwen.model, source_root=Path(source_adapter).resolve(), output=Path(output).resolve(),
    )


def _grad_norm(named: Sequence[tuple[str, torch.nn.Parameter]]) -> float:
    values = [parameter.grad for _, parameter in named]
    _require(all(value is not None and bool(torch.isfinite(value).all()) for value in values),
             "selected DoRA gradients are missing or nonfinite")
    return math.sqrt(sum(float(value.detach().double().square().sum()) for value in values if value is not None))


def _load_smoke(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from probes.row_feedback.data import validate_bank

    smoke = json.loads(path.read_text())
    digest = smoke.pop("smoke_sha256")
    _require(digest == _json_sha256(smoke), "technical smoke hash changed")
    smoke["smoke_sha256"] = digest
    bank_binding = smoke["bank"]
    _require(_file_sha256(bank_binding["path"]) == bank_binding["sha256"],
             "technical smoke bank binding changed")
    bank = json.loads(Path(bank_binding["path"]).read_text())
    validate_bank(bank)
    expected = next((row for row in bank["records"]
                     if row["record_id"] == smoke["record"]["record_id"]), None)
    _require(expected == smoke["record"], "technical smoke record differs from bank")
    return smoke, bank


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _jsonable_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items()
            if key not in {"logits", "target_ids", "feedback_sources", "native_history_activation"}}


def technical_smoke_train(*, input_path: Path, output: Path) -> Mapping[str, Any]:
    """Run one disposable F update after the graph-detach falsifier."""

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "technical smoke owns physical GPU0 only")
    _require(not output.exists(), "technical smoke output already exists")
    output.mkdir(parents=True)
    started = time.monotonic()
    smoke, bank = _load_smoke(input_path)
    record = smoke["record"]
    source_adapter = Path(record["teacher"]["source_adapter"]["root"])
    _require(source_adapter.resolve() == ANCHOR_ADAPTER.resolve(), "technical smoke anchor changed")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    phase = "load"
    try:
        source_gate_receipt = embedding_source_gate_receipt()
        qwen, frontend, config, identity = load_feedback_policy(
            adapter_path=source_adapter, device=device,
        )
        phase = "materialize"
        bank_materialized = materialize_bank_records(qwen, frontend, config, bank)
        materialized = bank_materialized[record["record_id"]]
        named, frozen = bind_language_dora(qwen.model)
        frozen_versions = [(parameter, parameter._version) for _, parameter in frozen]
        train_input_ref = bank["sources"]["native_n16_training_input"]
        _require(_file_sha256(train_input_ref["path"]) == train_input_ref["sha256"],
                 "source optimizer input changed")
        optimizer_options = json.loads(Path(train_input_ref["path"]).read_text())["optimizer"]
        _require(optimizer_options == {
            "lr": 1e-5, "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": 0, "foreach": False,
        }, "source optimizer settings changed")
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            lr=optimizer_options["lr"], betas=tuple(optimizer_options["betas"]),
            eps=optimizer_options["eps"], weight_decay=optimizer_options["weight_decay"],
            foreach=optimizer_options["foreach"],
        )
        history = record["visible_history_token_ids"]
        targets = record["visible_target_token_ids"]
        boundary = history.count(BOX_END) - 1
        _require(boundary >= 0 and targets, "technical source boundary/target missing")

        phase = "S_native_history_falsifier"
        optimizer.zero_grad(set_to_none=True)
        ordinary = replay_visible(
            qwen, materialized["inputs"], prompt_ids=materialized["prompt_ids"],
            history_ids=history, target_ids=targets, arm="S",
            capture_native_history_activation=True,
        )
        ordinary_loss = visible_nll_sum(ordinary)
        ordinary_loss.backward()
        ordinary_history = ordinary["native_history_activation"]
        ordinary_history_grad = None if ordinary_history.grad is None else ordinary_history.grad[:, -1]
        _require(ordinary_history.grad is not None
                 and ordinary_history_grad is not None
                 and bool(torch.isfinite(ordinary_history_grad).all())
                 and float(ordinary_history_grad.norm()) > 0,
                 "S cached native-history activation lacks later-target gradient")
        if ordinary_history_grad is None:
            raise AssertionError("ordinary history gradient vanished after validation")
        ordinary_parameter_grad_norm = _grad_norm(named)
        ordinary_receipt = {
            "history_activation": "first_language_layer_k_projection_last_visible_history_token",
            "history_activation_grad_norm": float(ordinary_history_grad.norm()),
            "parameter_grad_norm": ordinary_parameter_grad_norm,
            "loss": float(ordinary_loss.detach()),
            "execution": _jsonable_result(ordinary),
        }
        del ordinary_loss, ordinary, ordinary_history

        phase = "detach_falsifier"
        optimizer.zero_grad(set_to_none=True)
        detached = replay_visible(
            qwen, materialized["inputs"], prompt_ids=materialized["prompt_ids"],
            history_ids=history, target_ids=targets, arm="F",
            capture_feedback_sources=True,
            detach_feedback_source_boundaries=(boundary,),
            capture_native_history_activation=True,
        )
        detached_loss = visible_nll_sum(detached)
        detached_loss.backward()
        detached_source = detached["feedback_sources"][boundary]
        detached_source_grad = detached_source.grad
        _require(detached_source_grad is None or float(detached_source_grad.norm()) == 0,
                 "detached final feedback source still receives later-target gradient")
        detached_parameter_grad_norm = _grad_norm(named)
        detached_history = detached["native_history_activation"]
        detached_history_grad = None if detached_history.grad is None else detached_history.grad[:, -1]
        _require(detached_history_grad is not None
                 and bool(torch.isfinite(detached_history_grad).all())
                 and float(detached_history_grad.norm()) > 0,
                 "F cached native-history gradient disappeared with source detach")
        if detached_history_grad is None:
            raise AssertionError("detached history gradient vanished after validation")
        detach_receipt = {
            "boundary_index": boundary,
            "visible_boundary_index": len(history) - 1,
            "source_grad_norm": 0.0 if detached_source_grad is None else float(detached_source_grad.norm()),
            "history_activation": "first_language_layer_k_projection_last_visible_history_token",
            "history_activation_grad_norm": float(detached_history_grad.norm()),
            "parameter_grad_norm": detached_parameter_grad_norm,
            "loss": float(detached_loss.detach()),
            "execution": _jsonable_result(detached),
        }
        del detached_loss, detached, detached_source, detached_source_grad, detached_history

        phase = "optimizer_update"
        optimizer.zero_grad(set_to_none=True)
        before = [parameter.detach().clone() for _, parameter in named]
        live = replay_visible(
            qwen, materialized["inputs"], prompt_ids=materialized["prompt_ids"],
            history_ids=history, target_ids=targets, arm="F",
            capture_feedback_sources=True,
        )
        loss = visible_nll_sum(live)
        loss.backward()
        live_source = live["feedback_sources"][boundary]
        _require(live_source.grad is not None and bool(torch.isfinite(live_source.grad).all())
                 and float(live_source.grad.norm()) > 0,
                 "live final feedback source lacks later-target gradient")
        raw_grad_norm = _grad_norm(named)
        clipped = float(torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named], 1.0,
            error_if_nonfinite=True, foreach=False,
        ))
        optimizer.step()
        movement = math.sqrt(sum(float((parameter.detach() - old).double().square().sum())
                                 for (_, parameter), old in zip(named, before, strict=True)))
        _require(movement > 0 and math.isfinite(movement), "optimizer made no finite adapter update")
        _require(all(parameter._version == version for parameter, version in frozen_versions),
                 "optimizer changed a frozen parameter")
        live_receipt = {
            "loss": float(loss.detach()),
            "source_grad_norm": float(live_source.grad.norm()),
            "raw_dora_grad_norm": raw_grad_norm,
            "clip_returned_norm": clipped,
            "adapter_movement_l2": movement,
            "optimizer_steps": 1,
            "execution": _jsonable_result(live),
        }
        phase = "adapter_save"
        saved = save_feedback_adapter(
            qwen, source_adapter=source_adapter, output=output / "adapter",
        )
        receipt = {
            "schema": "row_feedback.technical_train_receipt.v1",
            "status": "technical_candidate_one_disposable_update_saved",
            "claim_boundary": "Execution/gradient/save evidence only; zero scientific fit updates.",
            "input": {"path": str(input_path.resolve()), "sha256": _file_sha256(input_path)},
            "record_id": record["record_id"],
            "arm": "F",
            "source_adapter": record["teacher"]["source_adapter"],
            "loaded_identity": identity,
            "source_gate": source_gate_receipt,
            "trainable_surface": {
                "tensor_count": len(named),
                "scalar_count": sum(parameter.numel() for _, parameter in named),
            },
            "full_bank_materialization": {
                "record_count": len(bank_materialized),
                "record_ids": sorted(bank_materialized),
                "prepared_inputs_sha256": {
                    key: value["prepared_inputs_sha256"] for key, value in bank_materialized.items()
                },
            },
            "optimizer": optimizer_options,
            "S_native_history_falsifier": ordinary_receipt,
            "detach_falsifier": detach_receipt,
            "update": live_receipt,
            "saved_adapter": saved,
            "resources": {
                "wall_seconds": time.monotonic() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "peak_rss_bytes": _rss_bytes(),
            },
        }
        (output / "train-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt
    except BaseException as exc:
        failure = {
            "schema": "row_feedback.technical_train_failure.v1",
            "status": "failed", "phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.monotonic() - started,
        }
        (output / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise


def technical_smoke_cold(
    *, input_path: Path, train_output: Path, output: Path, max_visible_tokens: int,
) -> Mapping[str, Any]:
    """Fresh-process cold load and natural F continuation from the saved adapter."""

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "cold smoke owns physical GPU0 only")
    _require(not output.exists(), "cold smoke output already exists")
    output.mkdir(parents=True)
    started = time.monotonic()
    smoke, bank = _load_smoke(input_path)
    receipt_path = train_output / "train-receipt.json"
    _require(receipt_path.is_file(), "training receipt is missing")
    train_receipt = json.loads(receipt_path.read_text())
    _require(train_receipt["status"] == "technical_candidate_one_disposable_update_saved"
             and train_receipt["input"]["sha256"] == _file_sha256(input_path),
             "cold load training receipt/input mismatch")
    adapter = Path(train_receipt["saved_adapter"]["root"])
    _require(adapter.resolve() == (train_output / "adapter").resolve(),
             "cold load adapter path differs from training output")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    phase = "load"
    try:
        source_gate_receipt = embedding_source_gate_receipt()
        qwen, frontend, config, identity = load_feedback_policy(
            adapter_path=adapter, device=device,
        )
        phase = "materialize"
        record = smoke["record"]
        bank_materialized = materialize_bank_records(qwen, frontend, config, bank)
        materialized = bank_materialized[record["record_id"]]
        phase = "natural_continuation"
        result = generate_visible(
            qwen, materialized["inputs"], prompt_ids=materialized["prompt_ids"],
            history_ids=record["visible_history_token_ids"], arm="F",
            max_visible_tokens=max_visible_tokens,
        )
        _require(result["visible_generated_tokens"] == len(result["visible_token_ids"])
                 and result["visible_generated_tokens"] <= max_visible_tokens,
                 "cold visible budget/accounting mismatch")
        receipt = {
            "schema": "row_feedback.technical_cold_receipt.v1",
            "status": "cold_adapter_loaded_and_natural_F_continuation_completed",
            "claim_boundary": "Cold persistence/consumer evidence only; generated content is not a quality result.",
            "input": {"path": str(input_path.resolve()), "sha256": _file_sha256(input_path)},
            "training_receipt": {"path": str(receipt_path.resolve()), "sha256": _file_sha256(receipt_path)},
            "loaded_identity": identity,
            "source_gate": source_gate_receipt,
            "full_bank_materialization": {
                "record_count": len(bank_materialized),
                "record_ids": sorted(bank_materialized),
                "prepared_inputs_sha256": {
                    key: value["prepared_inputs_sha256"] for key, value in bank_materialized.items()
                },
            },
            "generation": _jsonable_result(result),
            "resources": {
                "wall_seconds": time.monotonic() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "peak_rss_bytes": _rss_bytes(),
            },
        }
        (output / "cold-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt
    except BaseException as exc:
        failure = {
            "schema": "row_feedback.technical_cold_failure.v1",
            "status": "failed", "phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.monotonic() - started,
        }
        (output / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("technical-train")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    cold = subparsers.add_parser("technical-cold")
    cold.add_argument("--input", type=Path, required=True)
    cold.add_argument("--train-output", type=Path, required=True)
    cold.add_argument("--output", type=Path, required=True)
    cold.add_argument("--max-visible-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.command == "technical-train":
        result = technical_smoke_train(input_path=args.input, output=args.output)
    else:
        result = technical_smoke_cold(
            input_path=args.input, train_output=args.train_output,
            output=args.output, max_visible_tokens=args.max_visible_tokens,
        )
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

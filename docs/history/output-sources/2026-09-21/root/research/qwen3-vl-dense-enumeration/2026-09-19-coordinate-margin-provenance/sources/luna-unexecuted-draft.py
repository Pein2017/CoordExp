"""Exact incremental provenance replay for the frozen coordinate-margin states.

The source prefix is materialised with the accepted native batch builder.  The
saved continuation is then consumed through the model's ordinary incremental
``generate`` path, with a processor that forces only the recorded tokens before
the requested score.  The processor returns the unmodified score at the target
step; it never patches a model output or a hidden state.

This producer deliberately keeps reduction out of the runtime.  It writes one
``capture.pt`` per state with the tensors consumed by
``coordinate_margin/reduce.py`` and a small JSON receipt carrying all source
bindings and replay parity checks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from transformers import (
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from probes.training_set_completion.numerical_feedback.runtime import full_prefix
from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity
from probes.training_set_completion.untied_shared import load_model
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs


EOS = 151645
COORD = 151670
VOCAB_PREFIX = 4
PARITY_ATOL = 2e-4


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def token_hash(tokens: list[int]) -> str:
    return canonical_hash(tokens)


def tensor_hash(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous()
    return hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()


def _path_binding(path: Path) -> dict[str, Any]:
    return _binding(path)


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"nonfinite tensor: {name}")


def _target_row(value: Any, target_index: int) -> torch.Tensor:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise TypeError("hook output is not a tensor with a sequence dimension")
    if target_index >= value.shape[0]:
        raise IndexError("target index is outside hook batch")
    return value[target_index, -1]


def _source_sha(path: Path, expected: dict[str, Any] | None) -> dict[str, Any]:
    observed = _path_binding(path)
    if expected is not None and observed.get("sha256") != expected.get("sha256"):
        raise ValueError(f"source binding changed: {path}")
    return observed


def _state_source(state: dict[str, Any]) -> tuple[Path, Path, Path]:
    return (
        Path(state["source_release"]["path"]),
        Path(state["source_trajectory"]["path"]),
        Path(state["effective_readout"]["path"]),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _continuation_from_companion(
    row: dict[str, Any], source_end: int, count: int, pad: int
) -> list[int]:
    """Reproduce the source generator's suffix for a non-target companion."""
    tokens = [int(x) for x in row["token_ids"]]
    suffix: list[int] = []
    ended = False
    for index in range(count):
        absolute = source_end + index
        if ended or absolute >= len(tokens):
            suffix.append(pad)
            continue
        value = tokens[absolute]
        suffix.append(value)
        if value == EOS:
            ended = True
    return suffix


def _module_value(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError("module hook output is not a tensor")
    return output


class _ReplayCapture:
    """Hooks and score capture for exactly one target generation step."""

    def __init__(
        self,
        runtime: "MarginRuntime",
        target_index: int,
        target_step: int,
        instrument: bool,
    ) -> None:
        self.runtime = runtime
        self.target_index = target_index
        self.target_step = target_step
        self.instrument = instrument
        self.forward_count = 0
        self.active_step = -1
        self.last_head_input: torch.Tensor | None = None
        self.last_head_logits: torch.Tensor | None = None
        self.target_head_input: torch.Tensor | None = None
        self.target_logits: torch.Tensor | None = None
        self.target_logit_hook: torch.Tensor | None = None
        self.layers: dict[str, list[torch.Tensor | None]] = {
            "layer_inputs": [],
            "attention": [],
            "post_attention": [],
            "mlp": [],
            "layer_outputs": [],
        }
        self.pre_final: torch.Tensor | None = None
        self.handles: list[Any] = []

    def _clone(self, value: torch.Tensor) -> torch.Tensor:
        # The clone happens before the text model applies any DeepStack
        # post-layer in-place update to its caller-visible hidden state.
        return value.detach().clone()[self.target_index, -1].cpu()

    def model_pre(self, _module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        self.active_step = self.forward_count
        self.forward_count += 1
        self.runtime.forward_count += 1
        if self.runtime.forward_count > self.runtime.args.max_forwards:
            raise RuntimeError("forward budget exhausted")
        if time.monotonic() - self.runtime.started > self.runtime.args.max_seconds:
            raise RuntimeError("wall-time budget exhausted")
        self.last_head_input = None
        self.last_head_logits = None

    def head_pre(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs:
            raise RuntimeError("lm head prehook received no input")
        value = inputs[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("lm head input is not a tensor")
        self.last_head_input = self._clone(value)
        if self.active_step == self.target_step:
            self.target_head_input = self.last_head_input.clone()

    def head_post(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        value = _module_value(output)
        self.last_head_logits = self._clone(value)
        if self.active_step == self.target_step:
            self.target_logit_hook = self.last_head_logits.clone()

    def layer_pre(self, index: int, _module: Any, inputs: tuple[Any, ...]) -> None:
        if self.active_step != self.target_step or not inputs:
            return
        value = inputs[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("decoder layer input is not a tensor")
        self.layers["layer_inputs"][index] = self._clone(value)

    def attention_post(self, index: int, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if self.active_step != self.target_step:
            return
        value = _module_value(output)
        self.layers["attention"][index] = self._clone(value)

    def post_attention_pre(self, index: int, _module: Any, inputs: tuple[Any, ...]) -> None:
        if self.active_step != self.target_step or not inputs:
            return
        value = inputs[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("post-attention residual is not a tensor")
        self.layers["post_attention"][index] = self._clone(value)

    def mlp_post(self, index: int, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if self.active_step != self.target_step:
            return
        value = _module_value(output)
        self.layers["mlp"][index] = self._clone(value)

    def layer_post(self, index: int, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if self.active_step != self.target_step:
            return
        value = _module_value(output)
        self.layers["layer_outputs"][index] = self._clone(value)

    def norm_pre(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if self.active_step != self.target_step or not inputs:
            return
        value = inputs[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("final norm input is not a tensor")
        self.pre_final = self._clone(value)

    def install(self) -> None:
        model = self.runtime.model
        self.handles.append(model.register_forward_pre_hook(self.model_pre, with_kwargs=True))
        self.handles.append(self.runtime.head.register_forward_pre_hook(self.head_pre))
        self.handles.append(self.runtime.head.register_forward_hook(self.head_post))
        if not self.instrument:
            return
        for index, layer in enumerate(self.runtime.layers):
            self.layers["layer_inputs"].append(None)
            self.layers["attention"].append(None)
            self.layers["post_attention"].append(None)
            self.layers["mlp"].append(None)
            self.layers["layer_outputs"].append(None)
            self.handles.append(
                layer.register_forward_pre_hook(
                    lambda module, inputs, i=index: self.layer_pre(i, module, inputs)
                )
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    lambda module, inputs, output, i=index: self.attention_post(i, module, inputs, output)
                )
            )
            self.handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    lambda module, inputs, i=index: self.post_attention_pre(i, module, inputs)
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    lambda module, inputs, output, i=index: self.mlp_post(i, module, inputs, output)
                )
            )
            self.handles.append(
                layer.register_forward_hook(
                    lambda module, inputs, output, i=index: self.layer_post(i, module, inputs, output)
                )
            )
        self.handles.append(self.runtime.final_norm.register_forward_pre_hook(self.norm_pre))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def validate(self) -> None:
        if self.target_head_input is None or self.target_logit_hook is None:
            raise RuntimeError("target head capture was not observed")
        if not self.instrument:
            return
        for name, values in self.layers.items():
            if len(values) != len(self.runtime.layers) or any(value is None for value in values):
                raise RuntimeError(f"missing target layer capture: {name}")
        if self.pre_final is None:
            raise RuntimeError("missing target pre-final residual")


class _FixedContinuation(LogitsProcessor):
    def __init__(
        self,
        runtime: "MarginRuntime",
        expected: torch.Tensor,
        initial_width: int,
        target_step: int,
        capture: _ReplayCapture,
    ) -> None:
        self.runtime = runtime
        self.expected = expected
        self.initial_width = initial_width
        self.target_step = target_step
        self.capture = capture
        self.captured = False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        step = int(input_ids.shape[1] - self.initial_width)
        if step < 0:
            raise RuntimeError("generation prefix shrank unexpectedly")
        if step == self.target_step:
            self.capture.target_logits = scores[self.runtime.target_index].detach().clone().cpu()
            self.captured = True
            return scores
        if step > self.target_step:
            return scores
        if step >= self.expected.shape[1]:
            raise RuntimeError("fixed continuation matrix is too short")
        values = torch.full_like(scores, -torch.inf)
        choices = self.expected[:, step].to(device=scores.device)
        values.scatter_(1, choices[:, None], 0.0)
        return values


class _StopAfterTarget(StoppingCriteria):
    def __init__(self, processor: _FixedContinuation) -> None:
        self.processor = processor

    def __call__(self, _input_ids: torch.Tensor, _scores: Any, **_kwargs: Any) -> bool:
        return bool(self.processor.captured)


class MarginRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started = time.monotonic()
        self.forward_count = 0
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.selected_path = args.selected_states
        self.selected_binding = _path_binding(self.selected_path)
        selected = _read_json(self.selected_path)
        if selected.get("status") != "root-frozen":
            raise ValueError("selected states are not root-frozen")
        self.states = {str(item["id"]): item for item in selected["states"]}
        if args.state_id not in self.states:
            raise KeyError(f"unknown frozen state: {args.state_id}")
        self.state = self.states[args.state_id]
        self.target_index = int(self.state.get("batch_index", 3))
        self.cell_dir = self.output / args.state_id
        if self.cell_dir.exists() and any(self.cell_dir.iterdir()):
            raise FileExistsError(f"state output already contains artifacts: {self.cell_dir}")
        self.cell_dir.mkdir(parents=True, exist_ok=True)
        self.receipt: dict[str, Any] = {
            "schema": "coordinate_margin.cell_receipt.v1",
            "status": "running",
            "state_id": args.state_id,
            "model": self.state["model"],
            "source_policy": self.state["source_policy"],
            "offset": int(self.state["offset"]),
            "selected_states": self.selected_binding,
            "producer": _path_binding(Path(__file__)),
            "pid": os.getpid(),
            "model_forwards": 0,
            "artifacts": [],
        }
        write_json(self.cell_dir / "receipt.json", self.receipt)

        self.model_name = str(self.state["model"])
        self.q, self.identity = load_model(self.model_name, args.device)
        self.model = self.q.model
        self.head = self.model.get_output_embeddings()
        self.language_model = self.model.model.language_model
        self.layers = list(self.language_model.layers)
        self.final_norm = self.language_model.norm
        self.pad = int(self.q.tokenizer.pad_token_id)
        self._load_source()
        self._prepare_native_group()
        self._prepare_readout()
        self.versions = {
            name: parameter._version for name, parameter in self.model.named_parameters()
        }
        self.receipt["identity"] = self.identity
        self.receipt["source"] = self.source_bindings
        self.receipt["input_identity"] = _input_identity(self.batch)
        self.receipt["batch_size"] = len(self.raw)
        self.receipt["layer_count"] = len(self.layers)
        self.receipt["target_index"] = self.target_index
        self._persist()

    def _persist(self) -> None:
        self.receipt["model_forwards"] = self.forward_count
        self.receipt["elapsed_seconds"] = time.monotonic() - self.started
        write_json(self.cell_dir / "receipt.json", self.receipt)

    def _load_source(self) -> None:
        release_path, trajectory_path, effective_path = _state_source(self.state)
        release_binding = _source_sha(release_path, self.state["source_release"])
        trajectory_binding = _source_sha(trajectory_path, self.state["source_trajectory"])
        effective_binding = _source_sha(effective_path, self.state["effective_readout"])
        release = _read_json(release_path)
        if release.get("status") != "candidate_complete":
            raise ValueError("source release is not complete")
        if release.get("model") != self.model_name or release.get("policy") != self.state["source_policy"]:
            raise ValueError("source release identity changed")
        source = release.get("source") or {}
        source_end = int(source.get("source_end_exclusive", -1))
        offset = int(self.state["offset"])
        target_tokens = [int(x) for x in release["target"]["token_ids"]]
        if source_end <= 0 or offset < 0 or offset >= len(target_tokens):
            raise ValueError("source continuation does not cover selected offset")
        expected_prefix = [int(x) for x in self.state["actual_prefix_token_ids"]]
        if target_tokens[:offset] != expected_prefix:
            raise ValueError("saved state prefix differs from source release continuation")
        receipt_path = Path(release["source_receipt"]["path"])
        receipt_binding = _source_sha(receipt_path, release["source_receipt"])
        source_receipt = _read_json(receipt_path)
        if source_receipt.get("status") != "candidate_complete":
            raise ValueError("natural source receipt is not complete")
        if source_receipt.get("condition") != f"{self.model_name}-original":
            raise ValueError("natural source condition changed")
        raw_path = Path(source_receipt["raw"]["path"])
        trace_path = Path(source_receipt["trace"]["path"])
        raw_binding = _source_sha(raw_path, source_receipt["raw"])
        trace_binding = _source_sha(trace_path, source_receipt["trace"])
        raw_doc = _read_json(raw_path)
        self.raw = raw_doc.get("rows")
        if not isinstance(self.raw, list) or len(self.raw) != 4:
            raise ValueError("source batch is not the required heterogeneous bs4 group")
        if not 0 <= self.target_index < len(self.raw):
            raise ValueError("target batch index is outside source batch")
        native = [int(x) for x in self.raw[self.target_index]["token_ids"]]
        if native[:source_end] != [int(x) for x in self.raw[self.target_index]["token_ids"][:source_end]]:
            raise AssertionError("native source prefix changed while loading")
        if int(self.raw[self.target_index]["image_id"]) != int(self.state["image_id"]):
            raise ValueError("source image identity changed")
        trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=False)
        if trajectory.get("target_tokens") is not None:
            saved_tokens = [int(x) for x in trajectory["target_tokens"].tolist()]
            if saved_tokens != target_tokens:
                raise ValueError("source trajectory target tokens differ from release")
        if trajectory["head_inputs"].shape[0] <= offset:
            raise ValueError("source trajectory does not cover selected offset")
        trace = _read_json(trace_path)
        steps = trace.get("steps")
        if not isinstance(steps, list) or offset >= len(steps):
            raise ValueError("source trace does not cover selected offset")
        self.source_release = release
        self.source_receipt = source_receipt
        self.source_end = source_end
        self.offset = offset
        self.target_tokens = target_tokens
        self.source_trajectory = trajectory
        self.source_trace = trace
        self.source_step = steps[offset]
        self.source_head = trajectory["head_inputs"][offset].detach().cpu().float()
        self.source_coordinate = trajectory["raw_coordinate_logits"][offset].detach().cpu().float()
        self.source_top2 = self.source_step["raw_top2"]
        self.native_tokens = native
        self.source_bindings = {
            "release": release_binding,
            "trajectory": trajectory_binding,
            "effective_readout": effective_binding,
            "natural_receipt": receipt_binding,
            "raw": raw_binding,
            "trace": trace_binding,
        }
        self.receipt["source_end_exclusive"] = source_end
        self.receipt["target_token_hash"] = token_hash(target_tokens)
        self.receipt["source_prefix_token_hash"] = token_hash(native[:source_end])

    def _prepare_native_group(self) -> None:
        panel_path = Path(self.source_receipt["panel"]["path"])
        panel = _read_json(panel_path)
        group_key = str(self.source_release["group"])
        groups = [item for item in panel["groups"] if item["key"] == group_key]
        if len(groups) != 1:
            raise ValueError("source panel group is not unique")
        group = groups[0]
        config = dict(panel["configs"][self.model_name])
        config["data"] = dict(input_jsonl=group["input_jsonl"])
        requests, _ = build_bound_native_requests(self.q, config, group["cases"])
        self.batch = prepare_native_inputs(
            self.q.processor,
            requests,
            device=self.args.device,
            record_media_identity=True,
        )
        if len(self.batch.request_ids) != 4:
            raise ValueError("native replay requires exactly four source companions")
        expected_identity = self.source_receipt.get("input_identity")
        observed_identity = _input_identity(self.batch)
        if expected_identity is not None and observed_identity != expected_identity:
            raise ValueError("source heterogeneous batch input identity changed")
        self.panel_binding = _path_binding(panel_path)
        self.receipt["panel"] = self.panel_binding
        self.receipt["group"] = group_key

    def _prepare_readout(self) -> None:
        selected = self.head.selected_token_ids
        expected = list(range(COORD, COORD + 1000))
        if selected[VOCAB_PREFIX:].tolist() != expected:
            raise ValueError("unexpected coordinate token IDs")
        self.coordinate_ids = selected[VOCAB_PREFIX:].detach().cpu()
        self.effective_W = (
            self.head.base.weight[selected] + self.head.shared_embed_delta
        ).detach()[VOCAB_PREFIX:].cpu().float()
        self.receipt["coordinate_ids_sha256"] = tensor_hash(self.coordinate_ids)
        self.receipt["effective_W_sha256"] = tensor_hash(self.effective_W)

    def _expected_suffix(self) -> torch.Tensor:
        count = self.offset + 1
        rows: list[list[int]] = []
        for index, row in enumerate(self.raw):
            if index == self.target_index:
                values = self.target_tokens[:count]
            else:
                values = _continuation_from_companion(row, self.source_end, count, self.pad)
            if len(values) != count:
                raise RuntimeError("fixed continuation row has wrong length")
            rows.append(values)
        return torch.tensor(rows, dtype=torch.long)

    def _inputs(self) -> dict[str, Any]:
        inputs = full_prefix(
            self.batch,
            self.raw,
            self.source_end,
            self.pad,
            self.args.device,
        )
        target_suffix = inputs["input_ids"][self.target_index, -self.source_end :]
        expected = torch.tensor(
            self.native_tokens[: self.source_end],
            device=target_suffix.device,
            dtype=target_suffix.dtype,
        )
        if not torch.equal(target_suffix, expected):
            raise ValueError("source prefix was not appended byte-for-byte")
        if self.pad in target_suffix.tolist():
            raise ValueError("target source prefix contains padding")
        return inputs

    def _run_replay(self, instrument: bool) -> tuple[_ReplayCapture, dict[str, Any]]:
        # Qwen caches rope deltas on the text decoder.  Reinitialise this
        # derived cache so the paired runs start from identical state.
        self.language_model.rope_deltas = None
        inputs = self._inputs()
        initial_width = int(inputs["input_ids"].shape[1])
        expected = self._expected_suffix().to(self.args.device)
        capture = _ReplayCapture(self, self.target_index, self.offset, instrument)
        processor = _FixedContinuation(
            self, expected, initial_width, self.offset, capture
        )
        capture.install()
        started_forwards = self.forward_count
        try:
            config = GenerationConfig(
                max_new_tokens=self.offset + 1,
                do_sample=False,
                repetition_penalty=1,
                eos_token_id=None,
                pad_token_id=self.pad,
            )
            with torch.inference_mode():
                self.model.generate(
                    **inputs,
                    generation_config=config,
                    use_model_defaults=False,
                    logits_processor=LogitsProcessorList([processor]),
                    stopping_criteria=StoppingCriteriaList([_StopAfterTarget(processor)]),
                )
        finally:
            capture.remove()
        capture.validate()
        if not processor.captured:
            raise RuntimeError("target score was not reached by fixed replay")
        if capture.target_logits is None:
            raise RuntimeError("processor did not retain target logits")
        metadata = {
            "instrumented": instrument,
            "initial_width": initial_width,
            "offset": self.offset,
            "expected_prefix_hash": token_hash(self.target_tokens[: self.offset]),
            "expected_suffix_hash": token_hash(self.target_tokens[: self.offset + 1]),
            "model_forwards": self.forward_count - started_forwards,
            "forward_count_total": self.forward_count,
            "generation_path": "native_model.generate.incremental_fixed_continuation",
            "source_seed": "full_prefix_through_source_end_only",
            "forced_target_steps": self.offset,
            "score_step_unforced": True,
        }
        return capture, metadata

    @staticmethod
    def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left.double() - right.double()).abs().max().item())

    def _source_parity(self, capture: _ReplayCapture) -> dict[str, Any]:
        if capture.target_head_input is None or capture.target_logits is None:
            raise RuntimeError("missing target score capture")
        source_head_error = self._max_abs(capture.target_head_input, self.source_head)
        observed_coord = capture.target_logits[self.coordinate_ids.long()]
        source_coord_error = self._max_abs(observed_coord, self.source_coordinate)
        top = torch.topk(capture.target_logits, 2)
        source_top = self.source_top2["top2"]
        top_error = max(
            abs(float(top.values[index]) - float(source_top[index]["logit"]))
            for index in range(2)
        )
        winner = int(top.indices[0])
        source_winner = int(self.source_step["raw_winner_token"])
        return {
            "head_input_max_abs": source_head_error,
            "coordinate_logits_max_abs": source_coord_error,
            "full_top2_max_abs": top_error,
            "raw_winner": winner,
            "source_raw_winner": source_winner,
            "raw_winner_exact": winner == source_winner,
            "atol": PARITY_ATOL,
            "passed": bool(
                source_head_error <= PARITY_ATOL
                and source_coord_error <= PARITY_ATOL
                and top_error <= PARITY_ATOL
                and winner == source_winner
            ),
        }

    def _norm_scale(self, pre_final: torch.Tensor) -> tuple[float, torch.Tensor, float]:
        module_eps = getattr(self.final_norm, "variance_epsilon", None)
        if module_eps is None:
            module_eps = getattr(self.final_norm, "eps", None)
        if module_eps is None:
            raise RuntimeError("final norm epsilon is unavailable")
        value = pre_final.to(torch.float32)
        variance = value.pow(2).mean().item()
        scale = float((variance + float(module_eps)) ** -0.5)
        gain = self.final_norm.weight.detach().cpu().float() * scale
        return scale, gain, float(module_eps)

    def _capture_payload(
        self,
        no_hook: _ReplayCapture,
        hooked: _ReplayCapture,
        no_meta: dict[str, Any],
        hook_meta: dict[str, Any],
    ) -> dict[str, Any]:
        if no_hook.target_logits is None or hooked.target_logits is None:
            raise RuntimeError("paired replay has no target logits")
        if no_hook.target_head_input is None or hooked.target_head_input is None:
            raise RuntimeError("paired replay has no target head input")
        if hooked.pre_final is None:
            raise RuntimeError("paired replay has no final residual")
        hook_logits = hooked.target_logits.float()
        no_logits = no_hook.target_logits.float()
        paired_error = self._max_abs(no_logits, hook_logits)
        paired_bitwise = bool(torch.equal(no_hook.target_logits, hooked.target_logits))
        if not paired_bitwise:
            raise RuntimeError(f"no-hook/hook logits are not bitwise equal (max={paired_error})")
        parity = self._source_parity(hooked)
        if not parity["passed"]:
            raise RuntimeError(f"source replay parity failed: {parity}")
        layers = hooked.layers
        layer_inputs = torch.stack([value for value in layers["layer_inputs"] if value is not None]).float()
        attention = torch.stack([value for value in layers["attention"] if value is not None]).float()
        post_attention = torch.stack([value for value in layers["post_attention"] if value is not None]).float()
        mlp = torch.stack([value for value in layers["mlp"] if value is not None]).float()
        layer_outputs = torch.stack([value for value in layers["layer_outputs"] if value is not None]).float()
        pre_final = hooked.pre_final.float()
        norm_scale, norm_gain, norm_eps = self._norm_scale(pre_final)
        inter_layer_extra = layer_inputs[1:] - layer_outputs[:-1]
        payload: dict[str, Any] = {
            "schema": "coordinate_margin.capture.v1",
            "state_id": self.state["id"],
            "model": self.model_name,
            "source_policy": self.state["source_policy"],
            "group": self.source_release["group"],
            "boundary_id": self.source_release["boundary_id"],
            "batch_index": self.target_index,
            "batch_size": len(self.raw),
            "source_end_exclusive": self.source_end,
            "offset": self.offset,
            "target_token_id": int(self.target_tokens[self.offset]),
            "target_prefix_token_ids": self.target_tokens[: self.offset],
            "target_prefix_hash": token_hash(self.target_tokens[: self.offset]),
            "input_residual": layer_inputs[0],
            "layer_inputs": layer_inputs,
            "attention": attention,
            "post_attention": post_attention,
            "mlp": mlp,
            "layer_outputs": layer_outputs,
            "inter_layer_extra": inter_layer_extra,
            "pre_final": pre_final,
            "norm_weight": self.final_norm.weight.detach().cpu().float(),
            "norm_gain": norm_gain,
            "norm_scale": norm_scale,
            "norm_scale_fp64": float(torch.rsqrt(pre_final.double().square().mean() + norm_eps)),
            "norm_eps": norm_eps,
            "head_input": hooked.target_head_input.float(),
            "no_hook_head_input": no_hook.target_head_input.float(),
            "hook_head_input": hooked.target_head_input.float(),
            "logits": hook_logits,
            "no_hook_logits": no_logits,
            "hook_logits": hook_logits,
            "coordinate_ids": self.coordinate_ids,
            "effective_W": self.effective_W,
            "source_head": self.source_head,
            "source_coordinate_logits": self.source_coordinate,
            "target_input_token_id": int(self.target_tokens[self.offset - 1])
            if self.offset > 0
            else int(self.native_tokens[self.source_end - 1]),
            "source_top2": self.source_top2,
            "paired_replay": {
                "bitwise_equal": paired_bitwise,
                "max_abs": paired_error,
                "no_hook": no_meta,
                "hooked": hook_meta,
            },
            "source_parity": parity,
            "source_bindings": self.source_bindings,
        }
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                _finite(value, key)
        return payload

    def run(self) -> dict[str, Any]:
        self.receipt["status"] = "running_pair"
        self._persist()
        no_hook, no_meta = self._run_replay(False)
        hooked, hook_meta = self._run_replay(True)
        payload = self._capture_payload(no_hook, hooked, no_meta, hook_meta)
        capture_path = self.cell_dir / "capture.pt"
        torch.save(payload, capture_path)
        readback = torch.load(capture_path, map_location="cpu", weights_only=False)
        if readback.get("schema") != payload["schema"]:
            raise RuntimeError("capture readback schema mismatch")
        self.receipt["capture"] = _path_binding(capture_path)
        self.receipt["capture_keys"] = sorted(payload.keys())
        self.receipt["paired_replay"] = payload["paired_replay"]
        self.receipt["source_parity"] = payload["source_parity"]
        self.receipt["status"] = "candidate_complete"
        self.receipt["ended_pid"] = os.getpid()
        self._persist()
        return payload

    def close(self, error: BaseException | None = None) -> None:
        current = {
            name: parameter._version for name, parameter in self.model.named_parameters()
        }
        if current != self.versions:
            error = error or RuntimeError("parameter mutation detected")
        if error is not None:
            self.receipt["status"] = "partial_candidate"
            self.receipt["error"] = repr(error)
        self.receipt["ended_pid"] = os.getpid()
        self._persist()


def selfcheck() -> None:
    # CPU-only contract checks for the fixed continuation and the algebraic
    # tensor names.  Model loading is intentionally excluded.
    runtime = object.__new__(MarginRuntime)
    runtime.offset = 2
    runtime.source_end = 2
    runtime.target_index = 1
    runtime.pad = 0
    runtime.target_tokens = [10, 11, 12]
    runtime.raw = [
        {"token_ids": [1, 2, 3, 4, 5]},
        {"token_ids": [7, 8, 9, 10, 11]},
    ]
    got = runtime._expected_suffix()
    assert got.tolist() == [[3, 4, 5], [10, 11, 12]]
    assert token_hash([1, 2]) == canonical_hash([1, 2])
    print(json.dumps({"status": "pass", "checks": ["fixed_suffix", "hash"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-states", type=Path, required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-forwards", type=int, default=10000)
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    runtime: MarginRuntime | None = None
    error: BaseException | None = None
    try:
        runtime = MarginRuntime(args)
        runtime.run()
    except BaseException as exc:
        error = exc
        raise
    finally:
        if runtime is not None:
            runtime.close(error)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired Image2299 HF logit-lens probe with DeepStack seam checks.

This is deliberately experiment-local.  It generates one native greedy RP1.0
trajectory from each frozen adapter, then teacher-forced replays both exact
trajectories on both adapters.  Layer hooks are observational only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import torch
from src.qwen.inspection import CaptureHiddenRows, resolve_text_stack as qwen_text_stack



CONFIG = Path(__file__).with_name("configs") / "qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml"
PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
    "human-refined-13.geo_sorted_xy.coord.jsonl"
)
SOURCE_CHECKPOINT = Path(
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444"
)
SOURCE_ADAPTER = SOURCE_CHECKPOINT / "adapter"
SOURCE_DELTA = SOURCE_CHECKPOINT / "special_token_embeddings"
SOURCE_GATE_STUDY = Path(
    "docs/history/architecture/proposals/2026-06-27-coordexp-swift/"
    "source-studies/special-token-embeddings.md"
)
SOURCE_GATE_STUDY_SHA256 = "e024f8f9754475cfa6ed81136eae6c72becd2aca53b7b03b9fa3047e8da7d193"
SOURCE_GATE_RECEIPT_SHA256 = "b46af47a270e7e616577cf07719ce4c533ef671c648c6878ace1cf0941c938d3"
SOURCE_TRAIN_CONFIG = SOURCE_CHECKPOINT.parents[1] / "resolved_config.json"
OVERFIT_ADAPTER = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-05-human13-pure-ce-replay/run-v2/adapter"
)
DEFAULT_OUTPUT = Path("/tmp/coordexp-logit-lens-base-new")

SOURCE_ADAPTER_SHA256 = "49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da"
OVERFIT_ADAPTER_SHA256 = "8a5ebfcacfa92be4b873fea4439fc25570a9c415be2245e20fd1da94c9ff4070"
SOURCE_DELTA_SHA256 = "a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2"
OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
IM_END = 151645
COORD_START = 151670
COORD_END = 152670
EXPECTED_LAYER_COUNT = 28
DEEPSTACK_LAYER_COUNT = 3
MAX_NEW_TOKENS = 768
MAX_SITES = 24
TOP_K = 5
ATOL = 2e-4
RTOL = 2e-4
GPU_BUDGET_SECONDS = 30 * 60


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    os.replace(temp, path)


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("decoder output does not expose its hidden-state tensor")


def _resolve_path(owner: Any, dotted: str) -> Any | None:
    try:
        for part in dotted.split("."):
            owner = getattr(owner, part)
        return owner
    except AttributeError:
        return None


def resolve_text_stack(model: Any) -> tuple[Sequence[Any], Any, Any, dict[str, Any]]:
    """Resolve one distinct decoder stack, final norm, and output head."""

    stack = qwen_text_stack(model)
    owner, layers = stack.language_model, stack.layers
    aliases = [(path, owner) for path in ("model.language_model", "model.model.language_model", "language_model")
               if _resolve_path(model, path) is owner]
    require(bool(aliases), "Qwen stack has no registered logit-lens alias")
    stack_path = aliases[0][0]
    require(len(layers) == EXPECTED_LAYER_COUNT, f"expected {EXPECTED_LAYER_COUNT} decoder layers, found {len(layers)}")
    require(
        all("Qwen3VLTextDecoderLayer" in layer.__class__.__name__ for layer in layers),
        "resolved stack contains a non-Qwen3VLTextDecoderLayer module",
    )
    norm = owner.norm
    head = model.get_output_embeddings()
    require(isinstance(norm, torch.nn.Module) and isinstance(head, torch.nn.Module), "missing norm or lm_head")
    return layers, norm, head, {
        "stack_path": stack_path,
        "stack_aliases": [path for path, _ in aliases],
        "layer_count": len(layers),
        "layer_class": layers[0].__class__.__name__,
        "norm_class": norm.__class__.__name__,
        "head_class": head.__class__.__name__,
    }


def select_sites(
    *, prompt_token_count: int, generated_token_ids: Sequence[int]
) -> list[dict[str, Any]]:
    """Choose <=24 absolute hidden-state sites; state t predicts token t+1."""

    generated = [int(value) for value in generated_token_ids]
    require(prompt_token_count > 0 and generated, "site selection requires prompt and generated tokens")
    full_count = prompt_token_count + len(generated)
    labels: dict[int, list[str]] = {}

    def add(position: int, label: str) -> None:
        require(0 <= position < full_count, f"selected position {position} is out of range")
        labels.setdefault(position, []).append(label)

    add(prompt_token_count - 1, "prompt_end")
    ends = [index for index, token in enumerate(generated) if token == BOX_END]
    picked: dict[int, list[str]] = {}
    if ends:
        raw = [("first", 0), ("middle", len(ends) // 2), ("last", len(ends) - 1)]
        for name, index in raw:
            picked.setdefault(ends[index], []).append(name)
    for end, row_names in picked.items():
        starts = [index for index in range(end + 1) if generated[index] == OBJECT_REF_START]
        require(starts, f"selected completed row at {end} lacks an opener")
        start = starts[-1]
        opener_predictor = prompt_token_count + start - 1
        for row_name in row_names:
            add(opener_predictor, f"{row_name}_row_opener_decision")
            add(prompt_token_count + start, f"{row_name}_row_description_decision")
        coords = [
            index
            for index in range(start, end + 1)
            if COORD_START <= generated[index] < COORD_END
        ][:4]
        for coord_index, generated_index in enumerate(coords):
            for row_name in row_names:
                add(prompt_token_count + generated_index - 1, f"{row_name}_row_coord_{coord_index + 1}_decision")
        for row_name in row_names:
            add(prompt_token_count + end, f"{row_name}_row_boundary")
    eos_indices = [index for index, token in enumerate(generated) if token == IM_END]
    if eos_indices:
        add(prompt_token_count + eos_indices[0] - 1, "terminal_eos_decision")
    require(len(labels) <= MAX_SITES, f"site selection produced {len(labels)} > {MAX_SITES} sites")
    result = []
    for position, site_labels in labels.items():
        next_position = position + 1
        actual_next = None
        if next_position >= prompt_token_count:
            offset = next_position - prompt_token_count
            if offset < len(generated):
                actual_next = generated[offset]
        result.append(
            {
                "position": position,
                "labels": site_labels,
                "actual_next_token_id": actual_next,
            }
        )
    return result


class DeepStackLensCapture:
    """Clone decoder returns before in-place DeepStack and inspect next inputs."""

    def __init__(
        self,
        *,
        layers: Sequence[Any],
        norm: Any,
        selected_positions: Sequence[int],
        visual_mask: torch.Tensor,
    ) -> None:
        self.layers = layers
        self.norm = norm
        self.positions = tuple(int(value) for value in selected_positions)
        self.visual_mask = visual_mask.bool()
        self.handles: list[Any] = []
        self.capture_stack = ExitStack()
        self.selected_captures: dict[int, Any] = {}
        self.pending: dict[int, torch.Tensor] = {}
        self.residuals: dict[int, torch.Tensor] = {}
        self.boundaries: dict[int, dict[str, Any]] = {}

    def _layer_hook(self, index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            hidden = _first_tensor(output)
            require(hidden.ndim == 3 and hidden.shape[0] == 1, "decoder hidden shape changed")
            require(index not in self.pending, f"layer {index} hook fired twice")
            pre = hidden.detach().clone()  # Required: DeepStack mutates this return in place.
            self.pending[index] = pre
            return output

        return hook

    def _consume(self, index: int, hidden: torch.Tensor) -> None:
        require(index in self.pending, f"missing pre-injection clone for layer {index}")
        pre = self.pending.pop(index)
        post = hidden.detach()
        require(pre.shape == post.shape, f"layer {index} pre/post shape drift")
        require(self.visual_mask.shape == pre.shape[:2], "visual mask shape drift")
        delta = post - pre
        text = ~self.visual_mask
        text_delta = delta[text]
        visual_delta = delta[self.visual_mask]
        per_visual = visual_delta.float().norm(dim=-1)
        selected_pre = pre[0, list(self.positions), :]
        selected_post = post[0, list(self.positions), :]
        self.boundaries[index] = {
            "layer_index": index,
            "expected_deepstack_injection": index < DEEPSTACK_LAYER_COUNT,
            "text_position_count": int(text.sum().item()),
            "visual_position_count": int(self.visual_mask.sum().item()),
            "text_exact_equal": bool(torch.equal(text_delta, torch.zeros_like(text_delta))),
            "text_max_abs_delta": float(text_delta.abs().max().item()) if text_delta.numel() else 0.0,
            "selected_text_exact_equal": bool(torch.equal(selected_pre, selected_post)),
            "selected_text_max_abs_delta": float((selected_post - selected_pre).abs().max().item()),
            "visual_delta_l2": float(visual_delta.float().norm().item()),
            "visual_delta_max_row_l2": float(per_visual.max().item()) if per_visual.numel() else 0.0,
            "visual_changed_position_count": int((per_visual > 0).sum().item()),
        }

    def _next_pre_hook(self, prior: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            require(args and isinstance(args[0], torch.Tensor), "next layer prehook lacks hidden states")
            self._consume(prior, args[0])

        return hook

    def _norm_pre_hook(self, _module: Any, args: tuple[Any, ...]) -> None:
        require(args and isinstance(args[0], torch.Tensor), "norm prehook lacks hidden states")
        self._consume(len(self.layers) - 1, args[0])

    def __enter__(self) -> "DeepStackLensCapture":
        for index, layer in enumerate(self.layers):
            self.selected_captures[index] = self.capture_stack.enter_context(
                CaptureHiddenRows(layer, self.positions, boundary="output")
            )
            self.handles.append(layer.register_forward_hook(self._layer_hook(index)))
            if index:
                self.handles.append(layer.register_forward_pre_hook(self._next_pre_hook(index - 1)))
        self.handles.append(self.norm.register_forward_pre_hook(self._norm_pre_hook))
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.capture_stack.__exit__(*_exc)
        if not _exc or _exc[0] is None:
            self.residuals = {index: capture.hidden for index, capture in self.selected_captures.items()}

    def validate(self) -> None:
        require(not self.pending, f"unconsumed layer clones: {sorted(self.pending)}")
        require(sorted(self.residuals) == list(range(len(self.layers))), "incomplete residual capture")
        require(sorted(self.boundaries) == list(range(len(self.layers))), "incomplete boundary capture")
        for index, item in self.boundaries.items():
            require(item["text_exact_equal"], f"DeepStack changed text positions at layer {index}")
            require(item["selected_text_exact_equal"], f"DeepStack changed selected text at layer {index}")
            if index < DEEPSTACK_LAYER_COUNT:
                require(item["visual_delta_l2"] > 0.0, f"expected nonzero visual injection at layer {index}")
                require(
                    item["visual_changed_position_count"] == item["visual_position_count"],
                    f"not every visual position changed at layer {index}",
                )
            else:
                require(item["visual_delta_l2"] == 0.0, f"unexpected visual injection at layer {index}")


def _open_session(adapter: Path, *, source_gate_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    from src.config.fingerprint import sha256_json as config_sha256
    from src.config.inference import load_research_infer_config
    from src.inference.hf_backend import open_hf_backend_session
    from src.inference.runtime import assemble_frontend
    from probes.logit_lens.runtime import load_source_components

    resolved = load_research_infer_config(CONFIG.resolve())
    config = resolved.config
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=config_sha256(config.generation.model_dump(mode="json")),
    )
    launch = replace(
        frontend.launch,
        adapter={"type": "dora", "path": str(adapter.resolve()), "name": "default"},
        embedding_delta={
            "path": str(SOURCE_DELTA.resolve()),
            "source_gate_root": str(source_gate_root.resolve()),
        },
    )
    loaded = load_source_components(launch)
    opened = open_hf_backend_session(launch, components_loader=lambda _: loaded)
    return opened, frontend, config, resolved, loaded.qwen


def _request_and_inputs(
    *, components: Any, frontend: Any, config: Any, request_id: str
) -> tuple[Any, dict[str, Any], list[int], dict[str, Any]]:
    from src.data import load_raw_examples
    from src.inference.backend import DecodeRequest, GenerationPolicy
    from src.inference.image_plan import plan_image_batch
    from probes.logit_lens.runtime import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record

    raw = next(item for item in load_raw_examples(PANEL) if str(item.example_id).endswith("000000002299"))
    require(len(raw.objects) == 46, "Image2299 annotated-owner count changed")
    require(sum(obj.description == "person" for obj in raw.objects) == 38, "Image2299 person count changed")
    image = plan_image_batch(
        [raw], components=frontend.qwen, processor_config=_processor_config(config), row_indices=[0]
    ).rows[0]
    prompt = build_prompt_record(
        raw,
        _template_config(config),
        processor=frontend.qwen.processor,
        row_index=0,
        merged_visual_tokens=image.merged_visual_tokens,
        object_order_seed=config.template.object_order_seed,
    )
    request = DecodeRequest(
        request_id=request_id,
        chat_text=prompt.chat_text,
        input_prompt_token_ids=tuple(prompt.input_prompt_token_ids),
        expected_executed_prompt_token_ids=tuple(prompt.expected_executed_prompt_token_ids),
        image_path=image.image_path,
        declared_image_width=image.declared_width,
        declared_image_height=image.declared_height,
        decoded_image_width=image.decoded_width,
        decoded_image_height=image.decoded_height,
        image_sha256=image.image_content_sha256,
        expected_image_grid_thw=tuple(image.expected_image_grid_thw),
        logical_transform_id=image.logical_transform_id,
        generation_policy=GenerationPolicy(max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.0),
    )
    from probes.logit_lens.runtime import materialize_request
    native_inputs, executed_ids, grids, media_sha = materialize_request(components, request)
    prompt_ids = list(executed_ids[0])
    require(tuple(prompt_ids) == tuple(prompt.expected_executed_prompt_token_ids), "executed prompt drift")
    data = {
        "example_id": raw.example_id,
        "image_id": 2299,
        "annotated_owner_count_provenance_only": 46,
        "annotated_person_count_provenance_only": 38,
        "image_path": image.image_path,
        "image_file_sha256": image.image_content_sha256,
        "executed_rgb_sha256": media_sha[0],
        "image_grid_thw": list(grids[0]) if grids[0] is not None else None,
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": sha256_json(prompt_ids),
        "chat_text_sha256": hashlib.sha256(prompt.chat_text.encode()).hexdigest(),
    }
    return request, native_inputs, prompt_ids, data


def _trajectory(result: Any, tokenizer: Any, *, origin: str) -> dict[str, Any]:
    ids = [int(value) for value in result.generated_token_ids]
    eos_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    require(result.stop_reason in {"length", "im_end"}, f"unexpected stop reason {result.stop_reason}")
    require((result.stop_reason == "im_end") == (bool(ids) and ids[-1] == eos_id), "EOS/stop mismatch")
    require((result.stop_reason != "length") or len(ids) == MAX_NEW_TOKENS, "length stop before cap")
    return {
        "schema_version": "image2299_logit_lens_trajectory.v1",
        "prefix_origin": origin,
        "conditioning": "checkpoint_native_greedy_self_prefix",
        "generation_policy": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "score_channel": "HF_generation_policy_scores_not_logit_lens_raw_logits",
        },
        "token_ids": ids,
        "token_ids_sha256": sha256_json(ids),
        "token_count": len(ids),
        "completed_row_count": ids.count(BOX_END),
        "stop_reason": result.stop_reason,
        "hit_cap": result.stop_reason == "length",
        "emitted_eos": result.stop_reason == "im_end",
        "eos_token_id": eos_id,
        "raw_generated_text": result.raw_generated_text,
        "policy_chosen_logprobs": [
            None if row.policy_logprob is None else float(row.policy_logprob)
            for row in result.token_trace
            if not row.is_pad
        ],
    }


def _forward_inputs(native_inputs: Mapping[str, Any], prompt_ids: Sequence[int], generated: Sequence[int], model: Any) -> dict[str, Any]:
    from src.qwen.native import exact_history_inputs

    inputs = exact_history_inputs(model, native_inputs, [[*prompt_ids, *generated]], pad_token_id=0)
    inputs.pop("logits_to_keep")  # Each lens call selects its scientific sites explicitly.
    return inputs


def _sample_parameter_identity(module: Any, *, sample_count: int = 4096) -> dict[str, Any]:
    parameters = list(module.parameters())
    require(parameters, f"{module.__class__.__name__} has no parameters")
    total = sum(parameter.numel() for parameter in parameters)
    sampled: list[torch.Tensor] = []
    remaining = sample_count
    for parameter in parameters:
        flat = parameter.detach().reshape(-1)
        if flat.numel() <= remaining:
            sampled.append(flat.to(device="cpu", dtype=torch.float32))
            remaining -= flat.numel()
        else:
            indices = torch.linspace(0, flat.numel() - 1, steps=remaining, device=flat.device).long()
            sampled.append(flat[indices].to(device="cpu", dtype=torch.float32))
            remaining = 0
        if remaining == 0:
            break
    vector = torch.cat(sampled)
    return {
        "module_class": module.__class__.__name__,
        "parameter_count": len(parameters),
        "scalar_count": total,
        "sample_count": int(vector.numel()),
        "deterministic_sample_sha256": tensor_sha256(vector),
        "sample_is_not_full_parameter_hash": int(vector.numel()) != total,
    }


def _decode(tokenizer: Any, token_id: int | None) -> str | None:
    if token_id is None:
        return None
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def _summarize_layer(
    *,
    residual: torch.Tensor,
    norm: Any,
    head: Any,
    sites: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    eos_id: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    device = next(norm.parameters()).device
    with torch.inference_mode():
        normalized = norm(residual.to(device=device))  # final norm exactly once
        logits = head(normalized).detach().to(device="cpu", dtype=torch.float32).contiguous()
    require(logits.ndim == 2 and logits.shape[0] == len(sites), "lens logit shape drift")
    require(torch.isfinite(logits).all().item(), "nonfinite lens logits")
    rows: list[dict[str, Any]] = []
    for index, site in enumerate(sites):
        vector = logits[index]
        log_denom = torch.logsumexp(vector, dim=0)
        top_values, top_ids = torch.topk(vector, k=TOP_K)
        actual_id = site["actual_next_token_id"]
        actual = None
        if actual_id is not None:
            value = vector[int(actual_id)]
            actual = {
                "token_id": int(actual_id),
                "decoded": _decode(tokenizer, int(actual_id)),
                "raw_logit": float(value.item()),
                "probability": float(torch.exp(value - log_denom).item()),
                "rank_strict_greater_plus_one": int((vector > value).sum().item()) + 1,
                "is_coordinate": COORD_START <= int(actual_id) < COORD_END,
            }
        coordinate = vector[COORD_START:COORD_END]
        coordinate_value, coordinate_offset = coordinate.max(dim=0)
        rows.append(
            {
                "position": int(site["position"]),
                "labels": list(site["labels"]),
                "actual_next": actual,
                "top_k": [
                    {
                        "token_id": int(token_id),
                        "decoded": _decode(tokenizer, int(token_id)),
                        "raw_logit": float(value),
                    }
                    for token_id, value in zip(top_ids.tolist(), top_values.tolist(), strict=True)
                ],
                "opener_minus_eos_raw_logit": float((vector[OBJECT_REF_START] - vector[eos_id]).item()),
                "coordinate_family": {
                    "probability_mass": float(torch.exp(torch.logsumexp(coordinate, dim=0) - log_denom).item()),
                    "max_token_id": COORD_START + int(coordinate_offset.item()),
                    "max_decoded": _decode(tokenizer, COORD_START + int(coordinate_offset.item())),
                    "max_raw_logit": float(coordinate_value.item()),
                    "max_minus_eos_raw_logit": float((coordinate_value - vector[eos_id]).item()),
                },
                "eos": {
                    "token_id": eos_id,
                    "raw_logit": float(vector[eos_id].item()),
                    "probability": float(torch.exp(vector[eos_id] - log_denom).item()),
                    "rank_strict_greater_plus_one": int((vector > vector[eos_id]).sum().item()) + 1,
                },
            }
        )
    return {"sites": rows}, logits


def replay(
    *,
    checkpoint: str,
    prefix_origin: str,
    components: Any,
    native_inputs: Mapping[str, Any],
    prompt_ids: Sequence[int],
    trajectory: Mapping[str, Any],
    lens_handle: Any,
) -> tuple[list[dict[str, Any]], torch.Tensor, dict[str, Any]]:
    model = components.model
    tokenizer = components.tokenizer  # noqa: SLF001
    layers, norm, head, seam = resolve_text_stack(model)
    generated = [int(value) for value in trajectory["token_ids"]]
    sites = select_sites(prompt_token_count=len(prompt_ids), generated_token_ids=generated)
    positions = [int(site["position"]) for site in sites]
    eos_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    image_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    full_ids = torch.tensor([[*prompt_ids, *generated]], dtype=torch.long, device=next(model.parameters()).device)
    visual_mask = full_ids == image_id
    require(int(visual_mask.sum().item()) > 0, "replay lacks image placeholder positions")
    inputs = _forward_inputs(native_inputs, prompt_ids, generated, model)
    selected = torch.tensor(positions, dtype=torch.long, device=full_ids.device)

    with torch.inference_mode():
        reference_output = model(**inputs, logits_to_keep=selected)
    reference_logits = reference_output.logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    require(reference_logits.shape[:2] == (1, len(sites)), "unhooked selected-logit shape drift")

    with DeepStackLensCapture(
        layers=layers,
        norm=norm,
        selected_positions=positions,
        visual_mask=visual_mask,
    ) as capture:
        with torch.inference_mode():
            hooked_output = model(**inputs, logits_to_keep=selected)
    capture.validate()
    hooked_logits = hooked_output.logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    hook_max_abs = float((hooked_logits - reference_logits).abs().max().item())
    require(
        torch.allclose(hooked_logits, reference_logits, atol=ATOL, rtol=RTOL),
        f"hooks changed raw logits for {checkpoint}/{prefix_origin}: max_abs={hook_max_abs}",
    )

    records: list[dict[str, Any]] = []
    final_reconstruction: dict[str, Any] | None = None
    for layer_index in range(len(layers)):
        summary, logits = _summarize_layer(
            residual=capture.residuals[layer_index],
            norm=norm,
            head=head,
            sites=sites,
            tokenizer=tokenizer,
            eos_id=eos_id,
        )
        if layer_index == len(layers) - 1:
            delta = logits.unsqueeze(0) - reference_logits
            maximum = float(delta.abs().max().item())
            passed = bool(torch.allclose(logits.unsqueeze(0), reference_logits, atol=ATOL, rtol=RTOL))
            final_reconstruction = {
                "passed": passed,
                "max_absolute_difference": maximum,
                "atol": ATOL,
                "rtol": RTOL,
                "semantics": "captured_pre_norm_last_decoder_output_then_final_norm_once_then_lm_head",
            }
            require(passed, f"final layer reconstruction mismatch: max_abs={maximum}")
        records.append(
            {
                "schema_version": "image2299_logit_lens_layer.v1",
                "checkpoint": checkpoint,
                "prefix_origin": prefix_origin,
                "conditioning": "teacher_forced_exact_checkpoint_native_self_prefix_not_gt",
                "layer_index": layer_index,
                "prefix_token_ids_sha256": trajectory["token_ids_sha256"],
                "site_count": len(sites),
                **summary,
            }
        )
    require(final_reconstruction is not None, "missing final reconstruction")
    stacked = torch.stack([capture.residuals[index] for index in range(len(layers))])
    metadata = {
        "checkpoint": checkpoint,
        "prefix_origin": prefix_origin,
        "prefix_token_ids_sha256": trajectory["token_ids_sha256"],
        "sites": sites,
        "site_count": len(sites),
        "full_token_count": len(prompt_ids) + len(generated),
        "visual_position_count": int(visual_mask.sum().item()),
        "seam": seam,
        "hooked_vs_unhooked_raw_logits": {
            "passed": True,
            "max_absolute_difference": hook_max_abs,
            "atol": ATOL,
            "rtol": RTOL,
        },
        "final_layer_reconstruction": final_reconstruction,
        "deepstack_boundaries": [capture.boundaries[index] for index in range(len(layers))],
        "residual_tensor_shape": list(stacked.shape),
        "residual_tensor_sha256": tensor_sha256(stacked),
    }
    lens_handle.flush()
    return records, stacked, metadata


def _write_jsonl_rows(handle: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _runtime_identity() -> dict[str, Any]:
    import peft
    import transformers

    return {
        "pid": os.getpid(),
        "python": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _identity_receipt(resolved: Any, *, source_gate: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    source_adapter_file = SOURCE_ADAPTER / "adapter_model.safetensors"
    overfit_adapter_file = OVERFIT_ADAPTER / "adapter_model.safetensors"
    delta_file = SOURCE_DELTA / "special_token_embeddings.safetensors"
    require(sha256_file(source_adapter_file) == SOURCE_ADAPTER_SHA256, "Source adapter hash mismatch")
    require(sha256_file(overfit_adapter_file) == OVERFIT_ADAPTER_SHA256, "overfit adapter hash mismatch")
    require(sha256_file(delta_file) == SOURCE_DELTA_SHA256, "embedding delta hash mismatch")
    return {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256_at_launch": sha256_file(Path(__file__)),
        "config": resolved.to_artifact_dict(),
        "config_entry_sha256": sha256_file(CONFIG),
        "source_training_resolved_config_path": str(SOURCE_TRAIN_CONFIG),
        "source_training_resolved_config_sha256": sha256_file(SOURCE_TRAIN_CONFIG),
        "panel_path": str(PANEL),
        "panel_sha256": sha256_file(PANEL),
        "source_adapter": {"path": str(SOURCE_ADAPTER), "tensor_sha256": SOURCE_ADAPTER_SHA256},
        "overfit_adapter": {"path": str(OVERFIT_ADAPTER), "tensor_sha256": OVERFIT_ADAPTER_SHA256},
        "embedding_delta": {"path": str(SOURCE_DELTA), "tensor_sha256": SOURCE_DELTA_SHA256},
        "embedding_source_gate": dict(source_gate),
    }


def _stage_source_gate(output: Path) -> tuple[Path, dict[str, Any]]:
    """Copy the archived accepted evidence into this immutable run root."""

    root = output / "source-gate"
    destinations = {
        SOURCE_GATE_STUDY: root / SOURCE_GATE_STUDY,
        Path("outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json"): (
            root / "outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json"
        ),
    }
    sources = {
        SOURCE_GATE_STUDY: (Path(__file__).with_name("configs") / "source-gate-study.md").resolve(strict=True),
        Path("outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json"): (
            (Path(__file__).with_name("configs") / "source-gate-receipt.json").resolve(strict=True)
        ),
    }
    expected = {
        SOURCE_GATE_STUDY: SOURCE_GATE_STUDY_SHA256,
        Path("outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json"): (
            SOURCE_GATE_RECEIPT_SHA256
        ),
    }
    files: dict[str, Any] = {}
    for relative, destination in destinations.items():
        source = sources[relative]
        digest = sha256_file(source)
        require(digest == expected[relative], f"source-gate evidence hash changed: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        require(sha256_file(destination) == digest, f"source-gate copy changed: {destination}")
        files[str(relative)] = {"source": str(source), "staged": str(destination), "sha256": digest}
    return root, {"root": str(root), "files": files}


def _load_and_generate(
    *, checkpoint: str, adapter: Path, retain: bool, source_gate_root: Path
) -> tuple[dict[str, Any], Any | None, Any | None, list[int] | None, dict[str, Any], dict[str, Any], Any, Any | None]:
    opened, frontend, config, resolved, components = _open_session(adapter, source_gate_root=source_gate_root)
    request, native_inputs, prompt_ids, input_receipt = _request_and_inputs(
        components=components, frontend=frontend, config=config, request_id=f"image2299-logit-lens-{checkpoint}"
    )
    started = time.perf_counter()
    result = opened.decode((request,))[0]
    elapsed = time.perf_counter() - started
    trajectory = _trajectory(result, components.tokenizer, origin=checkpoint)  # noqa: SLF001
    trajectory["generation_elapsed_seconds"] = elapsed
    session_receipt = opened.receipt.to_artifact_dict()
    if retain:
        return trajectory, opened, native_inputs, prompt_ids, input_receipt, session_receipt, resolved, components
    opened.close()
    return trajectory, None, None, None, input_receipt, session_receipt, resolved, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started_wall = time.time()
    started_mono = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    inprogress = {
        "schema_version": "image2299_logit_lens_receipt.v1",
        "status": "running",
        "started_unix": started_wall,
        "output_root": str(output),
        "runtime": _runtime_identity(),
        "contract": {
            "image_id": 2299,
            "checkpoints": ["source", "overfit"],
            "prefix_origins": ["source", "overfit"],
            "generation_count": 2,
            "replay_count": 4,
            "replay_forward_count": 8,
            "max_new_tokens": MAX_NEW_TOKENS,
            "repetition_penalty": 1.0,
            "max_sites_per_prefix": MAX_SITES,
            "atol": ATOL,
            "rtol": RTOL,
            "gpu_budget_seconds": GPU_BUDGET_SECONDS,
        },
    }
    atomic_json(output / "receipt.inprogress.json", inprogress)
    opened: Any | None = None
    try:
        require(torch.cuda.is_available(), "CUDA is required for the live probe")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "GPU-8d43cb78-19ca-2f59-3179-7ea166cb1a4e"}, "probe is not bound to physical GPU0")
        source_gate_root, source_gate = _stage_source_gate(output)

        # Generate Source, unload it, then retain the overfit session for its two replays.
        source_traj, _, _, _, source_input, source_session, resolved, _ = _load_and_generate(
            checkpoint="source", adapter=SOURCE_ADAPTER, retain=False, source_gate_root=source_gate_root
        )
        atomic_json(output / "trajectory-source.json", source_traj)
        require(time.perf_counter() - started_mono < GPU_BUDGET_SECONDS, "GPU budget exhausted after Source generation")

        overfit_traj, opened, native_inputs, prompt_ids, overfit_input, overfit_session, resolved2, components = _load_and_generate(
            checkpoint="overfit", adapter=OVERFIT_ADAPTER, retain=True, source_gate_root=source_gate_root
        )
        require(native_inputs is not None and prompt_ids is not None, "retained overfit session is incomplete")
        atomic_json(output / "trajectory-overfit.json", overfit_traj)
        require(resolved.fingerprint == resolved2.fingerprint, "config resolution changed between sessions")
        require(source_input == overfit_input, "input/prompt identity changed between checkpoints")

        identities = _identity_receipt(resolved, source_gate=source_gate)
        layers, norm, head, _ = resolve_text_stack(components.model)  # noqa: SLF001
        module_identities = {
            "overfit_norm": _sample_parameter_identity(norm, sample_count=4096),
            "overfit_head": _sample_parameter_identity(head, sample_count=4096),
        }
        replay_receipts: list[dict[str, Any]] = []
        residual_payload: dict[str, Any] = {
            "schema_version": "image2299_logit_lens_selected_residuals.v1",
            "tensors": {},
            "sites": {},
        }
        with (output / "lens.jsonl").open("x") as lens_handle:
            for origin, trajectory in (("source", source_traj), ("overfit", overfit_traj)):
                require(time.perf_counter() - started_mono < GPU_BUDGET_SECONDS, "GPU budget exhausted before overfit replay")
                records, residuals, metadata = replay(
                    checkpoint="overfit",
                    prefix_origin=origin,
                    components=components,
                    native_inputs=native_inputs,
                    prompt_ids=prompt_ids,
                    trajectory=trajectory,
                    lens_handle=lens_handle,
                )
                _write_jsonl_rows(lens_handle, records)
                key = f"overfit__{origin}"
                residual_payload["tensors"][key] = residuals
                residual_payload["sites"][key] = metadata["sites"]
                replay_receipts.append(metadata)
        opened.close()
        opened = None
        components = None

        require(time.perf_counter() - started_mono < GPU_BUDGET_SECONDS, "GPU budget exhausted before Source replay")
        opened, frontend, config, resolved3, components = _open_session(SOURCE_ADAPTER, source_gate_root=source_gate_root)
        _request, native_inputs, prompt_ids, source_input2 = _request_and_inputs(
            components=components, frontend=frontend, config=config, request_id="image2299-logit-lens-source-replay"
        )
        require(source_input2 == source_input, "Source replay input identity changed")
        require(resolved3.fingerprint == resolved.fingerprint, "Source replay config drift")
        layers, norm, head, _ = resolve_text_stack(components.model)  # noqa: SLF001
        module_identities.update(
            {
                "source_norm": _sample_parameter_identity(norm, sample_count=4096),
                "source_head": _sample_parameter_identity(head, sample_count=4096),
            }
        )
        require(
            module_identities["source_norm"]["deterministic_sample_sha256"]
            == module_identities["overfit_norm"]["deterministic_sample_sha256"],
            "final norm differs across checkpoints",
        )
        require(
            module_identities["source_head"]["deterministic_sample_sha256"]
            == module_identities["overfit_head"]["deterministic_sample_sha256"],
            "output-head sample differs across checkpoints",
        )
        with (output / "lens.jsonl").open("a") as lens_handle:
            for origin, trajectory in (("source", source_traj), ("overfit", overfit_traj)):
                require(time.perf_counter() - started_mono < GPU_BUDGET_SECONDS, "GPU budget exhausted before Source replay")
                records, residuals, metadata = replay(
                    checkpoint="source",
                    prefix_origin=origin,
                    components=components,
                    native_inputs=native_inputs,
                    prompt_ids=prompt_ids,
                    trajectory=trajectory,
                    lens_handle=lens_handle,
                )
                _write_jsonl_rows(lens_handle, records)
                key = f"source__{origin}"
                residual_payload["tensors"][key] = residuals
                residual_payload["sites"][key] = metadata["sites"]
                replay_receipts.append(metadata)
        opened.close()
        opened = None
        components = None
        torch.save(residual_payload, output / "selected-residuals.pt")

        lens_lines = (output / "lens.jsonl").read_text().splitlines()
        parsed_lines = [json.loads(line) for line in lens_lines]
        require(len(parsed_lines) == 4 * EXPECTED_LAYER_COUNT, "lens JSONL row count mismatch")
        require(all(len(row["sites"]) == row["site_count"] for row in parsed_lines), "lens JSONL site count mismatch")
        elapsed = time.perf_counter() - started_mono
        require(elapsed <= GPU_BUDGET_SECONDS, f"probe exceeded GPU budget: {elapsed:.1f}s")
        artifacts = {}
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in {"receipt.inprogress.json", "receipt.json"}:
                relative = str(path.relative_to(output))
                artifacts[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        receipt = {
            **inprogress,
            "status": "mechanics_candidate",
            "completed_unix": time.time(),
            "elapsed_seconds": elapsed,
            "identity": identities,
            "input": source_input,
            "sessions": {"source_generation": source_session, "overfit_generation_and_replay": overfit_session},
            "trajectories": {
                "source": {key: source_traj[key] for key in ("token_ids_sha256", "token_count", "completed_row_count", "stop_reason", "hit_cap", "emitted_eos", "generation_elapsed_seconds")},
                "overfit": {key: overfit_traj[key] for key in ("token_ids_sha256", "token_count", "completed_row_count", "stop_reason", "hit_cap", "emitted_eos", "generation_elapsed_seconds")},
            },
            "prefixes_distinct": source_traj["token_ids_sha256"] != overfit_traj["token_ids_sha256"],
            "module_identities": module_identities,
            "replays": replay_receipts,
            "counts": {
                "native_generations": 2,
                "teacher_forced_replays": 4,
                "replay_forwards": 8,
                "lens_jsonl_rows": len(parsed_lines),
                "selected_residual_tensors": len(residual_payload["tensors"]),
            },
            "checks": {
                "all_hook_parity_passed": all(item["hooked_vs_unhooked_raw_logits"]["passed"] for item in replay_receipts),
                "all_final_reconstruction_passed": all(item["final_layer_reconstruction"]["passed"] for item in replay_receipts),
                "all_text_prepost_exact": all(all(layer["text_exact_equal"] for layer in item["deepstack_boundaries"]) for item in replay_receipts),
                "all_expected_visual_injections_nonzero": all(
                    all((layer["visual_delta_l2"] > 0) if layer["expected_deepstack_injection"] else (layer["visual_delta_l2"] == 0) for layer in item["deepstack_boundaries"])
                    for item in replay_receipts
                ),
                "paired_sites_same_within_prefix_origin": all(
                    next(item for item in replay_receipts if item["checkpoint"] == "source" and item["prefix_origin"] == origin)["sites"]
                    == next(item for item in replay_receipts if item["checkpoint"] == "overfit" and item["prefix_origin"] == origin)["sites"]
                    for origin in ("source", "overfit")
                ),
                "lens_jsonl_readback_passed": True,
            },
            "resource": {
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "artifact_payload_bytes_excluding_receipt": sum(item["bytes"] for item in artifacts.values()),
            },
            "artifacts": artifacts,
            "claim_boundary": {
                "descriptive_only": True,
                "trained_image_only": True,
                "teacher_forced_replay_not_gt": True,
                "no_causality_or_quality_claim": True,
                "raw_lens_logits_distinct_from_generation_policy_scores": True,
                "immediate_text_prepost_equality_is_architecture_expected": True,
            },
        }
        require(all(receipt["checks"].values()), "one or more terminal checks failed")
        atomic_json(output / "receipt.json", receipt)
        (output / "receipt.inprogress.json").unlink()
        print(json.dumps({"status": receipt["status"], "output_root": str(output), "elapsed_seconds": elapsed}, sort_keys=True))
        return 0
    except BaseException as error:
        if opened is not None:
            opened.close()
        failure = {
            **inprogress,
            "status": "failed",
            "failed_unix": time.time(),
            "elapsed_seconds": time.perf_counter() - started_mono,
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
            "resource": {
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
        }
        atomic_json(output / "receipt.failed.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

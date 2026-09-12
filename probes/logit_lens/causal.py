#!/usr/bin/env python3
"""Bounded Image2299 residual-state causal-transfer probe.

The maintained base helper is imported directly; predecessor inputs stay hash-bound.
This runner replays its exact overfit trajectory, captures six decoder block
outputs, and performs the preregistered bidirectional current-token,
full-causal-prefix, and equal-norm random-delta interventions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import torch
from src.qwen.inspection import CaptureInputs, CaptureHiddenRows


ROOT = Path(__file__).resolve().parents[2]
PARENT_OUTPUT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-08-image2299-logit-lens/run-v2"
)
TRAJECTORY = PARENT_OUTPUT / "trajectory-overfit.json"
TRAJECTORY_SHA256 = "537079135d3b12a3bfd72778ea63352112cde60775405e702abbbfa71efce063"
PARENT_RECEIPT = PARENT_OUTPUT / "receipt.json"
PARENT_RECEIPT_SHA256 = "0fc018e26e42d97a08b8a4a6d1e1f56dfd52f279d0637dcbb0c5e49890aa3e58"
DEFAULT_OUTPUT = Path("/tmp/coordexp-logit-lens-causal-new")

BLOCKS = (8, 16, 24, 26, 27, 28)  # 1-based decoder block outputs
NONFINAL_BLOCKS = BLOCKS[:-1]
STAGE_B_BLOCKS = (16, 24, 27, 28)
STAGE_B_NONFINAL_BLOCKS = STAGE_B_BLOCKS[:-1]
STAGE_B_IMAGE_IDS = (1584, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228)
RANDOM_SEEDS = (104729, 104759, 104761, 104773)
EXPECTED_SITE_COUNT = 12
ATOL = 2e-4
RTOL = 2e-4
TOP_K = 5
GPU_BUDGET_SECONDS = 20 * 60
STAGE_B_GPU_BUDGET_SECONDS = 40 * 60
MAX_DEVICE_BYTES = 48 * 1024**3




from probes.logit_lens import base as parent

PARENT_HELPER = Path(parent.__file__)
PARENT_HELPER_SHA256 = parent.sha256_file(PARENT_HELPER)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TypeError("decoder output does not support first-tensor replacement")


def _jsonl_append(handle: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _tensor_digest(value: torch.Tensor) -> str:
    return parent.tensor_sha256(value.detach().to(device="cpu").contiguous())


def _nested_tensor_identity(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _tensor_digest(value)}
    if isinstance(value, (tuple, list)):
        return [_nested_tensor_identity(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(type(value))


def coordinate_sites(prompt_ids: Sequence[int], trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = parent.select_sites(
        prompt_token_count=len(prompt_ids), generated_token_ids=trajectory["token_ids"]
    )
    sites = [site for site in selected if any("_coord_" in label for label in site["labels"])]
    require(len(sites) == EXPECTED_SITE_COUNT, f"expected 12 coordinate sites, found {len(sites)}")
    require(len({int(site["position"]) for site in sites}) == len(sites), "coordinate sites are not unique")
    return sites


def middle_coordinate_sites(
    prompt_ids: Sequence[int], trajectory: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str | None]:
    selected = parent.select_sites(
        prompt_token_count=len(prompt_ids), generated_token_ids=trajectory["token_ids"]
    )
    sites = [site for site in selected if any("middle_row_coord_" in label for label in site["labels"])]
    if not sites:
        return [], "no_completed_middle_row"
    if len(sites) != 4:
        return sites, f"middle_row_coordinate_site_count_{len(sites)}_not_4"
    if len({int(site["position"]) for site in sites}) != 4:
        return sites, "middle_row_coordinate_sites_not_unique"
    return sites, None


def request_and_inputs_for_image(
    *, components: Any, frontend: Any, config: Any, image_id: int, request_id: str
) -> tuple[Any, dict[str, Any], list[int], dict[str, Any]]:
    """Materialize one exact Human13 image through the frozen parent path."""

    from src.data import load_raw_examples
    from src.inference.backend import DecodeRequest, GenerationPolicy
    from src.inference.inputs import plan_examples

    suffix = f"{int(image_id):012d}"
    matches = [item for item in load_raw_examples(parent.PANEL) if str(item.example_id).endswith(suffix)]
    require(len(matches) == 1, f"expected one Human13 row for image {image_id}, found {len(matches)}")
    raw = matches[0]
    planned = plan_examples([raw], config=config, components=frontend.qwen)[0]
    image, prompt = planned.image, planned.prompt
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
        generation_policy=GenerationPolicy(
            max_new_tokens=parent.MAX_NEW_TOKENS, repetition_penalty=1.0
        ),
    )
    from probes.logit_lens.runtime import materialize_request
    native_inputs, executed_ids, grids, media_sha = materialize_request(components, request)
    prompt_ids = list(executed_ids[0])
    require(
        tuple(prompt_ids) == tuple(prompt.expected_executed_prompt_token_ids),
        f"executed prompt drift for image {image_id}",
    )
    receipt = {
        "example_id": raw.example_id,
        "image_id": int(image_id),
        "annotated_owner_count_provenance_only": len(raw.objects),
        "annotated_person_count_provenance_only": sum(obj.description == "person" for obj in raw.objects),
        "image_path": image.image_path,
        "image_file_sha256": image.image_content_sha256,
        "executed_rgb_sha256": media_sha[0],
        "image_grid_thw": list(grids[0]) if grids[0] is not None else None,
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": parent.sha256_json(prompt_ids),
        "chat_text_sha256": hashlib.sha256(prompt.chat_text.encode()).hexdigest(),
    }
    return request, native_inputs, prompt_ids, receipt




def _language_inputs(capture: CaptureInputs) -> dict[str, Any]:
    require(not capture.args, "text model unexpectedly received positional inputs")
    require(isinstance(capture.kwargs.get("inputs_embeds"), torch.Tensor), "missing text inputs")
    # Preserve this experiment's exact post-vision, cache-free projection.
    return {key: capture.kwargs.get(key) for key in (
        "attention_mask", "position_ids", "inputs_embeds", "cache_position",
        "visual_pos_masks", "deepstack_visual_embeds",
    )}


class ModuleCallCounter:
    def __init__(self, module: Any) -> None:
        self.module = module
        self.calls = 0
        self.handle: Any = None

    def _hook(self, _module: Any, _args: tuple[Any, ...], _output: Any) -> None:
        self.calls += 1

    def __enter__(self) -> "ModuleCallCounter":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def resolve_visual_module(model: Any) -> tuple[Any, list[str]]:
    candidates: list[tuple[str, Any]] = []
    for path in ("model.visual", "model.model.visual", "visual"):
        module = parent._resolve_path(model, path)
        if isinstance(module, torch.nn.Module):
            candidates.append((path, module))
    distinct: dict[int, list[tuple[str, Any]]] = {}
    for path, module in candidates:
        distinct.setdefault(id(module), []).append((path, module))
    require(len(distinct) == 1, f"expected one visual module, found {[path for path, _ in candidates]}")
    aliases = next(iter(distinct.values()))
    return aliases[0][1], [path for path, _ in aliases]




class ResidualPatch:
    """One-shot exact-position decoder-output replacement with corruption checks."""

    def __init__(
        self,
        module: Any,
        *,
        positions: Sequence[int],
        replacement: torch.Tensor,
        expected_before: torch.Tensor,
    ) -> None:
        self.module = module
        self.positions = tuple(int(position) for position in positions)
        self.replacement = replacement.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self.expected_before = expected_before.detach().to(device="cpu", dtype=torch.float32).contiguous()
        require(self.positions and len(set(self.positions)) == len(self.positions), "patch positions must be unique")
        require(self.replacement.ndim == 2, "replacement must have [positions, hidden] shape")
        require(self.replacement.shape == self.expected_before.shape, "replacement/before shape mismatch")
        require(self.replacement.shape[0] == len(self.positions), "patch span length mismatch")
        self.handle: Any = None
        self.calls = 0
        self.before_exact = False
        self.non_target_exact = False
        self.target_max_abs_delta = 0.0

    def _hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = parent._first_tensor(output)
        self.calls += 1
        require(self.calls == 1, "patch hook fired more than once")
        require(hidden.ndim == 3 and hidden.shape[0] == 1, "patch hidden shape changed")
        require(all(0 <= position < hidden.shape[1] for position in self.positions), "patch position out of range")
        positions = torch.tensor(self.positions, dtype=torch.long, device=hidden.device)
        observed = hidden[0, positions, :].detach().to(device="cpu", dtype=torch.float32)
        self.before_exact = bool(torch.equal(observed, self.expected_before))
        require(self.before_exact, "recipient state identity mismatch before patch")
        updated = hidden.clone()
        replacement = self.replacement.to(device=hidden.device, dtype=hidden.dtype)
        updated[0, positions, :] = replacement
        self.target_max_abs_delta = float((updated[0, positions, :] - hidden[0, positions, :]).abs().max().item())
        mask = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        mask[positions] = False
        self.non_target_exact = bool(torch.equal(updated[:, mask, :], hidden[:, mask, :]))
        require(self.non_target_exact, "patch changed a non-target residual position")
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return _replace_first_tensor(output, updated)

    def __enter__(self) -> "ResidualPatch":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        require(self.calls == 1, "patch did not fire exactly once")

    def receipt(self) -> dict[str, Any]:
        return {
            "position_count": len(self.positions),
            "first_position": self.positions[0],
            "last_position": self.positions[-1],
            "hook_calls": self.calls,
            "before_exact": self.before_exact,
            "non_target_residual_exact": self.non_target_exact,
            "target_max_abs_delta": self.target_max_abs_delta,
        }


@dataclass
class ModelBundle:
    name: str
    opened: Any
    tokenizer: Any
    model: Any
    text_model: Any
    layers: Sequence[Any]
    head: Any
    language_inputs: dict[str, Any]
    baseline_logits: torch.Tensor
    states: dict[int, torch.Tensor]
    input_receipt: dict[str, Any]
    session_receipt: dict[str, Any]
    baseline_checks: dict[str, Any]


def _text_forward(bundle: ModelBundle, selected: torch.Tensor) -> torch.Tensor:
    kwargs = {key: value for key, value in bundle.language_inputs.items() if value is not None}
    kwargs.update({"use_cache": False, "return_dict": True})
    with torch.inference_mode():
        output = bundle.text_model(**kwargs)
        logits = bundle.head(output.last_hidden_state[:, selected.to(output.last_hidden_state.device), :])
    result = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    require(torch.isfinite(result).all().item(), "nonfinite intervention logits")
    return result


def _open_and_capture(
    *, name: str, adapter: Path, source_gate_root: Path, trajectory: Mapping[str, Any]
) -> ModelBundle:
    opened, frontend, config, resolved, components = parent._open_session(adapter, source_gate_root=source_gate_root)
    _request, native_inputs, prompt_ids, input_receipt = parent._request_and_inputs(
        components=components, frontend=frontend, config=config, request_id=f"logit-lens-causal-transfer-{name}"
    )
    require(parent.sha256_json(list(prompt_ids)) == input_receipt["prompt_token_ids_sha256"], "prompt hash drift")
    sites = coordinate_sites(prompt_ids, trajectory)
    selected = torch.tensor([int(site["position"]) for site in sites], dtype=torch.long)
    model = components.model  # noqa: SLF001
    layers, _norm, head, seam = parent.resolve_text_stack(model)
    text_model = parent._resolve_path(model, seam["stack_path"])
    require(text_model is not None, "cannot resolve text model owner")
    forwarded = parent._forward_inputs(native_inputs, prompt_ids, trajectory["token_ids"], model)

    # Hooks-off native baseline (the pre-hook only records immutable text-stack inputs).
    with CaptureInputs(text_model) as input_capture:
        with torch.inference_mode():
            native = model(**forwarded, logits_to_keep=selected.to(next(model.parameters()).device))
    native_logits = native.logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    require(native_logits.shape[:2] == (1, EXPECTED_SITE_COUNT), "baseline selected-logit shape drift")

    provisional = ModelBundle(
        name=name,
        opened=opened,
        tokenizer=components.tokenizer,
        model=model,
        text_model=text_model,
        layers=layers,
        head=head,
        language_inputs=_language_inputs(input_capture),
        baseline_logits=native_logits,
        states={},
        input_receipt=input_receipt,
        session_receipt=opened.receipt.to_artifact_dict(),
        baseline_checks={},
    )
    with ExitStack() as capture:
        captured = {block: capture.enter_context(CaptureHiddenRows(
            layers[block - 1], range(len(prompt_ids) + len(trajectory["token_ids"])), boundary="output"
        )) for block in BLOCKS}
        hooked_logits = _text_forward(provisional, selected)
    maximum = float((hooked_logits - native_logits).abs().max().item())
    require(
        torch.allclose(hooked_logits, native_logits, atol=ATOL, rtol=RTOL),
        f"native/direct hooked baseline mismatch for {name}: max_abs={maximum}",
    )
    provisional.states = {block: item.hidden.unsqueeze(0) for block, item in captured.items()}
    provisional.baseline_checks = {
        "native_hooks_off_vs_direct_capture_passed": True,
        "max_absolute_difference": maximum,
        "atol": ATOL,
        "rtol": RTOL,
        "image_encoding_passes": 1,
        "decoder_forward_count": 2,
        "text_input_identity": _nested_tensor_identity(_language_inputs(input_capture)),
        "resolved_stack": seam,
        "resolved_config": resolved.to_artifact_dict(),
    }
    return provisional


def _capture_existing_session(
    *,
    name: str,
    opened: Any,
    components: Any,
    native_inputs: Mapping[str, Any],
    prompt_ids: Sequence[int],
    trajectory: Mapping[str, Any],
    sites: Sequence[Mapping[str, Any]],
    input_receipt: Mapping[str, Any],
    blocks: Sequence[int],
) -> ModelBundle:
    """Capture a no-KV-cache baseline from an already-open Stage B session."""

    selected = torch.tensor([int(site["position"]) for site in sites], dtype=torch.long)
    model = components.model  # noqa: SLF001
    layers, _norm, head, seam = parent.resolve_text_stack(model)
    text_model = parent._resolve_path(model, seam["stack_path"])
    require(text_model is not None, "cannot resolve text model owner")
    forwarded = parent._forward_inputs(native_inputs, prompt_ids, trajectory["token_ids"], model)
    with CaptureInputs(text_model) as input_capture:
        with torch.inference_mode():
            native = model(**forwarded, logits_to_keep=selected.to(next(model.parameters()).device))
    native_logits = native.logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    require(native_logits.shape[:2] == (1, len(sites)), "Stage B selected-logit shape drift")
    provisional = ModelBundle(
        name=name,
        opened=opened,
        tokenizer=components.tokenizer,
        model=model,
        text_model=text_model,
        layers=layers,
        head=head,
        language_inputs=_language_inputs(input_capture),
        baseline_logits=native_logits,
        states={},
        input_receipt=dict(input_receipt),
        session_receipt=opened.receipt.to_artifact_dict(),
        baseline_checks={},
    )
    with ExitStack() as capture:
        captured = {block: capture.enter_context(CaptureHiddenRows(
            layers[block - 1], range(len(prompt_ids) + len(trajectory["token_ids"])), boundary="output"
        )) for block in blocks}
        hooked_logits = _text_forward(provisional, selected)
    maximum = float((hooked_logits - native_logits).abs().max().item())
    require(
        torch.allclose(hooked_logits, native_logits, atol=ATOL, rtol=RTOL),
        f"Stage B native/direct hooked baseline mismatch {name}: max_abs={maximum}",
    )
    provisional.states = {block: item.hidden.unsqueeze(0) for block, item in captured.items()}
    provisional.baseline_checks = {
        "native_hooks_off_vs_direct_capture_passed": True,
        "max_absolute_difference": maximum,
        "atol": ATOL,
        "rtol": RTOL,
        "image_encoding_passes": 1,
        "decoder_forward_count": 2,
        "text_input_identity": _nested_tensor_identity(_language_inputs(input_capture)),
        "resolved_stack": seam,
        "blocks_1based": list(blocks),
    }
    return provisional


def _decode(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def _topk(vector: torch.Tensor, tokenizer: Any) -> list[dict[str, Any]]:
    values, ids = torch.topk(vector, k=TOP_K)
    return [
        {"token_id": int(token), "decoded": _decode(tokenizer, int(token)), "raw_logit": float(value)}
        for token, value in zip(ids.tolist(), values.tolist(), strict=True)
    ]


def _endpoint_summary(vector: torch.Tensor, *, a_id: int, s_id: int, tokenizer: Any) -> dict[str, Any]:
    top_id = int(vector.argmax().item())
    log_z = torch.logsumexp(vector, dim=0)
    return {
        "a_raw_logit": float(vector[a_id].item()),
        "s_raw_logit": float(vector[s_id].item()),
        "margin_a_minus_s": float((vector[a_id] - vector[s_id]).item()),
        "a_probability": float(torch.exp(vector[a_id] - log_z).item()),
        "s_probability": float(torch.exp(vector[s_id] - log_z).item()),
        "top1_token_id": top_id,
        "top1_decoded": _decode(tokenizer, top_id),
        "top_k": _topk(vector, tokenizer),
        "logsumexp": float(log_z.item()),
    }


def _distribution_diagnostics(patch: torch.Tensor, receiver: torch.Tensor, donor: torch.Tensor) -> dict[str, float]:
    return {
        "logit_l2_to_receiver": float((patch - receiver).norm().item()),
        "logit_l2_to_donor": float((patch - donor).norm().item()),
        "logit_max_abs_to_receiver": float((patch - receiver).abs().max().item()),
        "logit_max_abs_to_donor": float((patch - donor).abs().max().item()),
    }


def _random_replacement(
    recipient: torch.Tensor, donor: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    recipient = recipient.detach().to(device="cpu", dtype=torch.float32)
    donor = donor.detach().to(device="cpu", dtype=torch.float32)
    target_delta = donor - recipient
    target_norm = target_delta.norm()
    require(torch.isfinite(target_norm).item(), "nonfinite donor-recipient norm")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    direction = torch.randn(recipient.shape, generator=generator, dtype=torch.float32)
    unit = direction / direction.norm()
    scaled = unit * target_norm
    replacement = recipient + scaled
    actual_norm = (replacement - recipient).norm()
    require(
        torch.allclose(actual_norm, target_norm, atol=1e-5, rtol=1e-5),
        f"random delta norm mismatch seed={seed}",
    )
    return replacement, {
        "seed": int(seed),
        "unit_direction_sha256": _tensor_digest(unit),
        "donor_minus_recipient_l2": float(target_norm.item()),
        "random_delta_l2": float(actual_norm.item()),
        "norm_absolute_difference": float(abs(actual_norm.item() - target_norm.item())),
        "norm_match_atol": 1e-5,
        "norm_match_rtol": 1e-5,
        "norm_match_passed": True,
    }


def _eligible_endpoint(a_id: int, s_id: int) -> tuple[bool, str | None]:
    if not (parent.COORD_START <= a_id < parent.COORD_END):
        return False, "overfit_endpoint_noncoordinate"
    if not (parent.COORD_START <= s_id < parent.COORD_END):
        return False, "source_endpoint_noncoordinate"
    if a_id == s_id:
        return False, "equal_endpoint_tokens"
    return True, None


def build_endpoints(
    *,
    sites: Sequence[Mapping[str, Any]],
    source: ModelBundle,
    overfit: ModelBundle,
    expected_site_count: int = EXPECTED_SITE_COUNT,
) -> list[dict[str, Any]]:
    tokenizer = overfit.tokenizer  # noqa: SLF001
    rows: list[dict[str, Any]] = []
    for index, site in enumerate(sites):
        source_vector = source.baseline_logits[0, index]
        overfit_vector = overfit.baseline_logits[0, index]
        s_id = int(source_vector.argmax().item())
        a_id = int(overfit_vector.argmax().item())
        eligible, exclusion = _eligible_endpoint(a_id, s_id)
        rows.append(
            {
                "site_index": index,
                "position": int(site["position"]),
                "labels": list(site["labels"]),
                "trajectory_actual_next_token_id": int(site["actual_next_token_id"]),
                "a_overfit_top1_token_id": a_id,
                "a_decoded": _decode(tokenizer, a_id),
                "s_source_top1_token_id": s_id,
                "s_decoded": _decode(tokenizer, s_id),
                "a_is_coordinate": parent.COORD_START <= a_id < parent.COORD_END,
                "s_is_coordinate": parent.COORD_START <= s_id < parent.COORD_END,
                "equal_endpoint_tokens": a_id == s_id,
                "eligible_for_R": eligible,
                "exclusion_reason": exclusion,
                "source": _endpoint_summary(source_vector, a_id=a_id, s_id=s_id, tokenizer=tokenizer),
                "overfit": _endpoint_summary(overfit_vector, a_id=a_id, s_id=s_id, tokenizer=tokenizer),
            }
        )
    require(len(rows) == expected_site_count, "endpoint population changed")
    return rows


def _row_for_patch(
    *,
    direction: str,
    donor: ModelBundle,
    recipient: ModelBundle,
    endpoint: Mapping[str, Any],
    block: int,
    scope: str,
    patch_vector: torch.Tensor,
    patch_receipt: Mapping[str, Any],
    random_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    index = int(endpoint["site_index"])
    a_id = int(endpoint["a_overfit_top1_token_id"])
    s_id = int(endpoint["s_source_top1_token_id"])
    tokenizer = recipient.tokenizer  # noqa: SLF001
    receiver_vector = recipient.baseline_logits[0, index]
    donor_vector = donor.baseline_logits[0, index]
    summary = _endpoint_summary(patch_vector, a_id=a_id, s_id=s_id, tokenizer=tokenizer)
    m_receiver = float(receiver_vector[a_id] - receiver_vector[s_id])
    m_donor = float(donor_vector[a_id] - donor_vector[s_id])
    m_patch = float(patch_vector[a_id] - patch_vector[s_id])
    denominator = m_donor - m_receiver
    donorward = ((m_patch - m_receiver) * denominator > 0.0) if endpoint["eligible_for_R"] else None
    return {
        "schema_version": "logit_lens_causal_transfer_site.v1",
        "direction": direction,
        "donor": donor.name,
        "recipient": recipient.name,
        "block_1based": int(block),
        "scope": scope,
        "control_seed": None if random_receipt is None else int(random_receipt["seed"]),
        "site_index": index,
        "position": int(endpoint["position"]),
        "labels": list(endpoint["labels"]),
        "a_overfit_top1_token_id": a_id,
        "s_source_top1_token_id": s_id,
        "eligible_for_R": bool(endpoint["eligible_for_R"]),
        "exclusion_reason": endpoint["exclusion_reason"],
        "m_receiver": m_receiver,
        "m_donor": m_donor,
        "m_patch": m_patch,
        "patch_minus_receiver": m_patch - m_receiver,
        "donor_minus_receiver": denominator,
        "donorward": donorward,
        "patch_endpoint": summary,
        "distribution_diagnostics": _distribution_diagnostics(patch_vector, receiver_vector, donor_vector),
        "residual_patch": dict(patch_receipt),
        "random_delta": None if random_receipt is None else dict(random_receipt),
    }


def reduce_trace(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute unclipped grouped R values from raw rows."""

    keys = sorted(
        {
            (str(row["direction"]), int(row["block_1based"]), str(row["scope"]), row["control_seed"])
            for row in rows
        },
        key=lambda key: (key[0], key[1], key[2], -1 if key[3] is None else int(key[3])),
    )
    groups: list[dict[str, Any]] = []
    for direction, block, scope, seed in keys:
        group_rows = [
            row
            for row in rows
            if row["direction"] == direction
            and int(row["block_1based"]) == block
            and row["scope"] == scope
            and row["control_seed"] == seed
        ]
        eligible = [row for row in group_rows if row["eligible_for_R"]]
        numerator = sum(float(row["patch_minus_receiver"]) for row in eligible)
        denominator = sum(float(row["donor_minus_receiver"]) for row in eligible)
        r_value = None if denominator == 0.0 else numerator / denominator
        groups.append(
            {
                "direction": direction,
                "block_1based": block,
                "scope": scope,
                "control_seed": seed,
                "population_site_count": len(group_rows),
                "eligible_site_count": len(eligible),
                "excluded_site_count": len(group_rows) - len(eligible),
                "raw_numerator_sum": numerator,
                "raw_denominator_sum": denominator,
                "R_unclipped": r_value,
                "denominator_small_abs_lt_1e-6": abs(denominator) < 1e-6,
                "donorward_site_count": sum(row["donorward"] is True for row in eligible),
            }
        )

    random_means: list[dict[str, Any]] = []
    full_vs_current: list[dict[str, Any]] = []
    for direction in sorted({str(row["direction"]) for row in rows}):
        for block in sorted({int(row["block_1based"]) for row in rows if row["direction"] == direction}):
            random_groups = [
                group
                for group in groups
                if group["direction"] == direction and group["block_1based"] == block and group["scope"] == "random_delta"
            ]
            if random_groups:
                values = [float(group["R_unclipped"]) for group in random_groups if group["R_unclipped"] is not None]
                random_means.append(
                    {
                        "direction": direction,
                        "block_1based": block,
                        "seed_count": len(random_groups),
                        "R_seed_mean": None if not values else sum(values) / len(values),
                        "R_seed_min": None if not values else min(values),
                        "R_seed_max": None if not values else max(values),
                        "donorward_site_count_mean": sum(group["donorward_site_count"] for group in random_groups)
                        / len(random_groups),
                    }
                )
            current = next(
                (group for group in groups if group["direction"] == direction and group["block_1based"] == block and group["scope"] == "current"),
                None,
            )
            full = next(
                (group for group in groups if group["direction"] == direction and group["block_1based"] == block and group["scope"] == "full_prefix"),
                None,
            )
            if current is not None and full is not None:
                r_difference = None
                if current["R_unclipped"] is not None and full["R_unclipped"] is not None:
                    r_difference = float(full["R_unclipped"]) - float(current["R_unclipped"])
                full_vs_current.append(
                    {
                        "direction": direction,
                        "block_1based": block,
                        "R_full_minus_current": r_difference,
                        "raw_numerator_full_minus_current": float(full["raw_numerator_sum"])
                        - float(current["raw_numerator_sum"]),
                    }
                )
    return {"groups": groups, "random_controls": random_means, "full_vs_current": full_vs_current}


def _run_patch(
    *,
    recipient: ModelBundle,
    block: int,
    positions: Sequence[int],
    replacement: torch.Tensor,
    expected_before: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    patch = ResidualPatch(
        recipient.layers[block - 1],
        positions=positions,
        replacement=replacement,
        expected_before=expected_before,
    )
    with patch:
        logits = _text_forward(recipient, selected)
    return logits, patch.receipt()


def _self_patch_checks(bundle: ModelBundle, selected: torch.Tensor) -> list[dict[str, Any]]:
    sequence_length = int(next(iter(bundle.states.values())).shape[1])
    positions = list(range(sequence_length))
    checks: list[dict[str, Any]] = []
    for block in BLOCKS:
        logits, patch_receipt = _run_patch(
            recipient=bundle,
            block=block,
            positions=positions,
            replacement=bundle.states[block][0],
            expected_before=bundle.states[block][0],
            selected=selected,
        )
        exact = bool(torch.equal(logits, bundle.baseline_logits))
        require(exact, f"self patch changed selected logits for {bundle.name} block {block}")
        checks.append({"checkpoint": bundle.name, "block_1based": block, "selected_logits_exact": exact, **patch_receipt})
    return checks


def _self_patch_checks_for_blocks(
    bundle: ModelBundle, selected: torch.Tensor, *, blocks: Sequence[int]
) -> list[dict[str, Any]]:
    sequence_length = int(next(iter(bundle.states.values())).shape[1])
    positions = list(range(sequence_length))
    checks: list[dict[str, Any]] = []
    for block in blocks:
        logits, patch_receipt = _run_patch(
            recipient=bundle,
            block=int(block),
            positions=positions,
            replacement=bundle.states[int(block)][0],
            expected_before=bundle.states[int(block)][0],
            selected=selected,
        )
        exact = bool(torch.equal(logits, bundle.baseline_logits))
        require(exact, f"self patch changed selected logits for {bundle.name} block {block}")
        checks.append(
            {
                "checkpoint": bundle.name,
                "block_1based": int(block),
                "selected_logits_exact": exact,
                **patch_receipt,
            }
        )
    return checks


def _artifact_manifest(output: Path) -> dict[str, Any]:
    result = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"receipt.inprogress.json", "receipt.json"}:
            result[str(path.relative_to(output))] = {"bytes": path.stat().st_size, "sha256": parent.sha256_file(path)}
    return result


def _stage_b_image(
    *,
    image_id: int,
    stage_root: Path,
    source_opened: Any,
    source_components: Any,
    source_frontend: Any,
    source_config: Any,
    overfit_opened: Any,
    overfit_components: Any,
    overfit_frontend: Any,
    overfit_config: Any,
    stage_started_mono: float,
) -> dict[str, Any]:
    image_root = stage_root / "images" / f"image-{image_id:012d}"
    image_root.mkdir(parents=True, exist_ok=False)
    image_started = time.perf_counter()
    image_progress: dict[str, Any] = {
        "schema_version": "logit_lens_causal_transfer_stage_b_image_receipt.v1",
        "status": "running",
        "image_id": int(image_id),
        "blocks_1based": list(STAGE_B_BLOCKS),
        "nonfinal_blocks_1based": list(STAGE_B_NONFINAL_BLOCKS),
        "random_seeds": list(RANDOM_SEEDS),
    }
    parent.atomic_json(image_root / "receipt.inprogress.json", image_progress)
    counts = {
        "generation_decode_requests": 0,
        "generation_model_forward_calls": 0,
        "generation_tokens": 0,
        "full_model_teacher_forced": 0,
        "decoder_direct_baseline_capture": 0,
        "self_patch": 0,
        "current": 0,
        "full_prefix": 0,
        "random_delta": 0,
        "block28_current_identity": 0,
        "vision_encoder_forward_calls": 0,
        "trace_rows": 0,
    }
    try:
        require(
            time.perf_counter() - stage_started_mono < STAGE_B_GPU_BUDGET_SECONDS,
            "Stage B GPU budget exhausted before image",
        )
        overfit_request, overfit_native, overfit_prompt, overfit_input = request_and_inputs_for_image(
            components=overfit_components,
            frontend=overfit_frontend,
            config=overfit_config,
            image_id=image_id,
            request_id=f"logit-lens-causal-transfer-stage-b-overfit-{image_id}",
        )
        overfit_visual, overfit_visual_aliases = resolve_visual_module(overfit_components.model)  # noqa: SLF001
        generation_started = time.perf_counter()
        with (
            ModuleCallCounter(overfit_components.model) as generation_model_counter,  # noqa: SLF001
            ModuleCallCounter(overfit_visual) as generation_visual_counter,
        ):
            result = overfit_opened.decode((overfit_request,))[0]
        generation_elapsed = time.perf_counter() - generation_started
        trajectory = parent._trajectory(
            result, overfit_components.tokenizer, origin="overfit"  # noqa: SLF001
        )
        trajectory.update(
            {
                "image_id": int(image_id),
                "generation_elapsed_seconds": generation_elapsed,
                "generation_model_forward_calls": generation_model_counter.calls,
                "vision_encoder_forward_calls": generation_visual_counter.calls,
            }
        )
        require(generation_visual_counter.calls == 1, f"generation vision pass count changed for {image_id}")
        parent.atomic_json(image_root / "trajectory-overfit.json", trajectory)
        counts["generation_decode_requests"] = 1
        counts["generation_model_forward_calls"] = generation_model_counter.calls
        counts["generation_tokens"] = trajectory["token_count"]
        counts["vision_encoder_forward_calls"] += generation_visual_counter.calls

        source_request, source_native, source_prompt, source_input = request_and_inputs_for_image(
            components=source_components,
            frontend=source_frontend,
            config=source_config,
            image_id=image_id,
            request_id=f"logit-lens-causal-transfer-stage-b-source-{image_id}",
        )
        del source_request
        require(source_input == overfit_input, f"prompt/image identity differs for image {image_id}")
        parent.atomic_json(
            image_root / "input.json",
            {
                "schema_version": "logit_lens_causal_transfer_stage_b_input.v1",
                "image_id": int(image_id),
                "source": source_input,
                "overfit": overfit_input,
                "identical": True,
                "overfit_generation_request_id": overfit_request.request_id,
            },
        )
        sites, site_exclusion = middle_coordinate_sites(overfit_prompt, trajectory)
        if site_exclusion is not None:
            excluded = {
                **image_progress,
                "status": "scientifically_excluded",
                "exclusion_reason": site_exclusion,
                "trajectory": {
                    key: trajectory[key]
                    for key in (
                        "token_ids_sha256",
                        "token_count",
                        "completed_row_count",
                        "stop_reason",
                        "hit_cap",
                        "emitted_eos",
                    )
                },
                "counts": counts,
                "elapsed_seconds": time.perf_counter() - image_started,
            }
            parent.atomic_json(image_root / "receipt.json", excluded)
            (image_root / "receipt.inprogress.json").unlink()
            return excluded

        source_visual, source_visual_aliases = resolve_visual_module(source_components.model)  # noqa: SLF001
        with ModuleCallCounter(source_visual) as source_baseline_visual:
            source = _capture_existing_session(
                name="source",
                opened=source_opened, components=source_components,
                native_inputs=source_native,
                prompt_ids=source_prompt,
                trajectory=trajectory,
                sites=sites,
                input_receipt=source_input,
                blocks=STAGE_B_BLOCKS,
            )
        with ModuleCallCounter(overfit_visual) as overfit_baseline_visual:
            overfit = _capture_existing_session(
                name="overfit",
                opened=overfit_opened, components=overfit_components,
                native_inputs=overfit_native,
                prompt_ids=overfit_prompt,
                trajectory=trajectory,
                sites=sites,
                input_receipt=overfit_input,
                blocks=STAGE_B_BLOCKS,
            )
        require(
            source_baseline_visual.calls == 1 and overfit_baseline_visual.calls == 1,
            f"teacher-forced vision pass count changed for image {image_id}",
        )
        counts["full_model_teacher_forced"] = 2
        counts["decoder_direct_baseline_capture"] = 2
        counts["vision_encoder_forward_calls"] += source_baseline_visual.calls + overfit_baseline_visual.calls
        endpoints = build_endpoints(
            sites=sites, source=source, overfit=overfit, expected_site_count=4
        )
        require(len(endpoints) == 4, "Stage B endpoint count changed")
        for endpoint in endpoints:
            endpoint["image_id"] = int(image_id)
        baseline = {
            "schema_version": "logit_lens_causal_transfer_stage_b_baseline.v1",
            "image_id": int(image_id),
            "trajectory_token_ids_sha256": trajectory["token_ids_sha256"],
            "input": source_input,
            "sites": endpoints,
            "checkpoint_checks": {
                "source": source.baseline_checks,
                "overfit": overfit.baseline_checks,
            },
            "visual_module_aliases": {
                "source": source_visual_aliases,
                "overfit": overfit_visual_aliases,
            },
        }
        parent.atomic_json(image_root / "baseline.json", baseline)
        selected = torch.tensor([int(site["position"]) for site in sites], dtype=torch.long)
        compact = {
            "schema_version": "logit_lens_causal_transfer_stage_b_compact_tensors.v1",
            "image_id": int(image_id),
            "blocks_1based": list(STAGE_B_BLOCKS),
            "sites": [{"image_id": int(image_id), "site_index": i, **dict(site)} for i, site in enumerate(sites)],
            "baseline_logits": {
                "source": source.baseline_logits,
                "overfit": overfit.baseline_logits,
            },
            "selected_residuals": {
                checkpoint.name: torch.stack(
                    [checkpoint.states[block][0, selected, :] for block in STAGE_B_BLOCKS]
                )
                for checkpoint in (source, overfit)
            },
        }
        torch.save(compact, image_root / "compact-selected.pt")
        image_progress["status"] = "baseline_complete"
        image_progress["counts"] = dict(counts)
        parent.atomic_json(image_root / "receipt.inprogress.json", image_progress)

        self_checks = [
            *_self_patch_checks_for_blocks(source, selected, blocks=(28,)),
            *_self_patch_checks_for_blocks(overfit, selected, blocks=(28,)),
        ]
        for check in self_checks:
            check["image_id"] = int(image_id)
        counts["self_patch"] = len(self_checks)
        trace_rows: list[dict[str, Any]] = []
        with (image_root / "patch-traces.jsonl").open("x") as trace_handle:
            for direction, donor, recipient in (
                ("overfit_to_source", overfit, source),
                ("source_to_overfit", source, overfit),
            ):
                for block in STAGE_B_BLOCKS:
                    require(
                        time.perf_counter() - stage_started_mono < STAGE_B_GPU_BUDGET_SECONDS,
                        "Stage B GPU budget exhausted before patch group",
                    )
                    for endpoint in endpoints:
                        index = int(endpoint["site_index"])
                        position = int(endpoint["position"])
                        logits, patch_receipt = _run_patch(
                            recipient=recipient,
                            block=block,
                            positions=[position],
                            replacement=donor.states[block][0, position, :].unsqueeze(0),
                            expected_before=recipient.states[block][0, position, :].unsqueeze(0),
                            selected=selected,
                        )
                        counts["block28_current_identity" if block == 28 else "current"] += 1
                        earlier = selected < position
                        earlier_exact = bool(
                            torch.equal(logits[:, earlier, :], recipient.baseline_logits[:, earlier, :])
                        )
                        require(earlier_exact, "Stage B future patch contaminated earlier selected logits")
                        if block == 28:
                            other = torch.ones(4, dtype=torch.bool)
                            other[index] = False
                            unpatched_exact = bool(
                                torch.equal(logits[:, other, :], recipient.baseline_logits[:, other, :])
                            )
                            parity = bool(
                                torch.allclose(
                                    logits[0, index],
                                    donor.baseline_logits[0, index],
                                    atol=ATOL,
                                    rtol=RTOL,
                                )
                            )
                            parity_max = float(
                                (logits[0, index] - donor.baseline_logits[0, index]).abs().max().item()
                            )
                            require(parity and unpatched_exact, "Stage B block28 identity failed")
                            patch_receipt = {
                                **patch_receipt,
                                "all_unpatched_selected_logits_exact": unpatched_exact,
                                "donor_full_vocab_parity_passed": parity,
                                "donor_full_vocab_max_abs": parity_max,
                                "atol": ATOL,
                                "rtol": RTOL,
                            }
                        else:
                            patch_receipt = {
                                **patch_receipt,
                                "all_earlier_selected_logits_exact": earlier_exact,
                                "later_selected_logits_are_not_used_by_this_site_isolated_arm": True,
                            }
                        row = _row_for_patch(
                            direction=direction,
                            donor=donor,
                            recipient=recipient,
                            endpoint=endpoint,
                            block=block,
                            scope="current",
                            patch_vector=logits[0, index],
                            patch_receipt=patch_receipt,
                            random_receipt=None,
                        )
                        row["image_id"] = int(image_id)
                        trace_rows.append(row)
                        _jsonl_append(trace_handle, [row])
                    if block == 28:
                        continue

                    sequence_length = recipient.states[block].shape[1]
                    logits, patch_receipt = _run_patch(
                        recipient=recipient,
                        block=block,
                        positions=list(range(sequence_length)),
                        replacement=donor.states[block][0],
                        expected_before=recipient.states[block][0],
                        selected=selected,
                    )
                    counts["full_prefix"] += 1
                    patch_receipt = {
                        **patch_receipt,
                        "scope_semantics": "all_visual_prompt_generated_history_and_current_positions",
                        "causal_equivalence": "whole_sequence_once_read_all_sites_equals_each_prefix_through_t",
                    }
                    full_rows = []
                    for endpoint in endpoints:
                        row = _row_for_patch(
                            direction=direction,
                            donor=donor,
                            recipient=recipient,
                            endpoint=endpoint,
                            block=block,
                            scope="full_prefix",
                            patch_vector=logits[0, int(endpoint["site_index"])],
                            patch_receipt=patch_receipt,
                            random_receipt=None,
                        )
                        row["image_id"] = int(image_id)
                        full_rows.append(row)
                    trace_rows.extend(full_rows)
                    _jsonl_append(trace_handle, full_rows)

                    for endpoint in endpoints:
                        index = int(endpoint["site_index"])
                        position = int(endpoint["position"])
                        recipient_state = recipient.states[block][0, position, :]
                        donor_state = donor.states[block][0, position, :]
                        for seed in RANDOM_SEEDS:
                            replacement, random_receipt = _random_replacement(
                                recipient_state, donor_state, seed=seed
                            )
                            logits, patch_receipt = _run_patch(
                                recipient=recipient,
                                block=block,
                                positions=[position],
                                replacement=replacement.unsqueeze(0),
                                expected_before=recipient_state.unsqueeze(0),
                                selected=selected,
                            )
                            counts["random_delta"] += 1
                            earlier = selected < position
                            earlier_exact = bool(
                                torch.equal(logits[:, earlier, :], recipient.baseline_logits[:, earlier, :])
                            )
                            require(earlier_exact, "Stage B random patch contaminated earlier logits")
                            patch_receipt = {
                                **patch_receipt,
                                "all_earlier_selected_logits_exact": earlier_exact,
                                "later_selected_logits_are_not_used_by_this_site_isolated_arm": True,
                            }
                            row = _row_for_patch(
                                direction=direction,
                                donor=donor,
                                recipient=recipient,
                                endpoint=endpoint,
                                block=block,
                                scope="random_delta",
                                patch_vector=logits[0, index],
                                patch_receipt=patch_receipt,
                                random_receipt=random_receipt,
                            )
                            row["image_id"] = int(image_id)
                            trace_rows.append(row)
                            _jsonl_append(trace_handle, [row])
                    image_progress["status"] = f"patch_complete_{direction}_block_{block}"
                    counts["trace_rows"] = len(trace_rows)
                    image_progress["counts"] = dict(counts)
                    parent.atomic_json(image_root / "receipt.inprogress.json", image_progress)

        require(len(trace_rows) == 152, f"Stage B trace count changed for image {image_id}")
        counts["trace_rows"] = len(trace_rows)
        reduction = reduce_trace(trace_rows)
        for collection in reduction.values():
            for item in collection:
                item["image_id"] = int(image_id)
        summary = {
            "schema_version": "logit_lens_causal_transfer_stage_b_summary.v1",
            "image_id": int(image_id),
            "population_site_count": len(endpoints),
            "eligible_site_count": sum(bool(endpoint["eligible_for_R"]) for endpoint in endpoints),
            "excluded_sites": [
                {
                    "image_id": int(image_id),
                    "site_index": endpoint["site_index"],
                    "reason": endpoint["exclusion_reason"],
                }
                for endpoint in endpoints
                if not endpoint["eligible_for_R"]
            ],
            **reduction,
        }
        parent.atomic_json(image_root / "summary.json", summary)
        cold_rows = [json.loads(line) for line in (image_root / "patch-traces.jsonl").read_text().splitlines()]
        require(cold_rows == trace_rows, f"Stage B cold trace readback failed for image {image_id}")
        cold_compact = torch.load(image_root / "compact-selected.pt", map_location="cpu", weights_only=False)
        require(cold_compact["image_id"] == image_id, "Stage B compact readback image mismatch")
        require(
            all(row["random_delta"] is None or row["random_delta"]["norm_match_passed"] for row in trace_rows),
            "Stage B random norm predicate failed",
        )
        artifacts = _artifact_manifest(image_root)
        elapsed = time.perf_counter() - image_started
        teacher_forced_total = (
            counts["full_model_teacher_forced"]
            + counts["decoder_direct_baseline_capture"]
            + counts["self_patch"]
            + counts["current"]
            + counts["full_prefix"]
            + counts["random_delta"]
            + counts["block28_current_identity"]
        )
        receipt = {
            **image_progress,
            "status": "mechanics_candidate",
            "elapsed_seconds": elapsed,
            "trajectory": {
                key: trajectory[key]
                for key in (
                    "token_ids_sha256",
                    "token_count",
                    "completed_row_count",
                    "stop_reason",
                    "hit_cap",
                    "emitted_eos",
                    "generation_elapsed_seconds",
                    "generation_model_forward_calls",
                    "vision_encoder_forward_calls",
                )
            },
            "input": source_input,
            "counts": {
                **counts,
                "teacher_forced_decoder_forward_total": teacher_forced_total,
                "model_forward_calls_including_generation": teacher_forced_total
                + counts["generation_model_forward_calls"],
                "population_sites": len(endpoints),
                "eligible_sites": summary["eligible_site_count"],
            },
            "checks": {
                "input_identity_across_models": True,
                "native_vs_text_capture_parity": all(
                    bundle.baseline_checks["native_hooks_off_vs_direct_capture_passed"]
                    for bundle in (source, overfit)
                ),
                "all_self_patch_exact": all(check["selected_logits_exact"] for check in self_checks),
                "all_patch_before_identity_exact": all(row["residual_patch"]["before_exact"] for row in trace_rows),
                "all_non_target_residual_exact": all(
                    row["residual_patch"]["non_target_residual_exact"] for row in trace_rows
                ),
                "all_random_norm_actual_allclose_passed": all(
                    row["random_delta"] is None or row["random_delta"]["norm_match_passed"]
                    for row in trace_rows
                ),
                "all_block28_donor_full_vocab_parity": all(
                    row["residual_patch"].get("donor_full_vocab_parity_passed", False)
                    for row in trace_rows
                    if row["block_1based"] == 28
                ),
                "all_block28_unpatched_exact": all(
                    row["residual_patch"].get("all_unpatched_selected_logits_exact", False)
                    for row in trace_rows
                    if row["block_1based"] == 28
                ),
                "trace_and_compact_cold_readback": True,
            },
            "self_patch_checks": self_checks,
            "resource": {
                "peak_cuda_allocated_bytes_stage_so_far": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes_stage_so_far": int(torch.cuda.max_memory_reserved()),
                "peak_host_rss_kib_stage_so_far": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "artifact_payload_bytes_excluding_receipt": sum(item["bytes"] for item in artifacts.values()),
            },
            "artifacts": artifacts,
            "claim_boundary": {
                "Human13_training_image": True,
                "overfit_native_prefix_middle_completed_row": True,
                "four_local_coordinate_decisions_not_accuracy": True,
                "no_heldout_or_owner_recovery_claim": True,
            },
        }
        require(all(receipt["checks"].values()), f"Stage B image checks failed {image_id}")
        parent.atomic_json(image_root / "receipt.json", receipt)
        (image_root / "receipt.inprogress.json").unlink()
        return receipt
    except BaseException as error:
        failure = {
            **image_progress,
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - image_started,
            "counts": counts,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        parent.atomic_json(image_root / "receipt.failed.json", failure)
        raise


def run_stage_b(output: Path) -> int:
    from probes.logit_lens.runtime import input_source_hashes

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started_unix = time.time()
    started_mono = time.perf_counter()
    runner_hash = parent.sha256_file(Path(__file__))
    executed_input_sources = input_source_hashes()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    progress: dict[str, Any] = {
        "schema_version": "logit_lens_causal_transfer_stage_b_receipt.v1",
        "status": "running",
        "stage": "B_remaining_Human13_fixed_replication",
        "started_unix": started_unix,
        "output_root": str(output),
        "intended_image_ids": list(STAGE_B_IMAGE_IDS),
        "excluded_anchor_image_id": 2299,
        "runner_sha256_at_launch": runner_hash,
        "executed_input_sources": executed_input_sources,
        "contract": {
            "blocks_1based": list(STAGE_B_BLOCKS),
            "nonfinal_blocks_1based": list(STAGE_B_NONFINAL_BLOCKS),
            "sites_per_eligible_image": 4,
            "site_selector": "middle_completed_row_four_coordinate_decisions",
            "directions": ["overfit_to_source", "source_to_overfit"],
            "random_seeds": list(RANDOM_SEEDS),
            "atol": ATOL,
            "rtol": RTOL,
            "gpu_budget_seconds": STAGE_B_GPU_BUDGET_SECONDS,
            "no_kv_cache_for_teacher_forced_forwards": True,
            "native_generation_may_use_its_standard_decode_cache": True,
        },
    }
    parent.atomic_json(output / "receipt.inprogress.json", progress)
    source_opened: Any | None = None
    overfit_opened: Any | None = None
    try:
        require(torch.cuda.is_available(), "CUDA is required for Stage B")
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "GPU-8d43cb78-19ca-2f59-3179-7ea166cb1a4e"},
            "Stage B is not bound to physical GPU0",
        )
        require(parent.sha256_file(PARENT_HELPER) == PARENT_HELPER_SHA256, "parent helper hash mismatch")
        panel_rows = [json.loads(line) for line in parent.PANEL.read_text().splitlines()]
        panel_ids = tuple(int(row["image_id"]) for row in panel_rows)
        require(
            tuple(image_id for image_id in panel_ids if image_id != 2299) == STAGE_B_IMAGE_IDS,
            f"frozen Stage B image IDs differ from Human13 panel: {panel_ids}",
        )
        source_gate_root, source_gate = parent._stage_source_gate(output)
        source_opened, source_frontend, source_config, source_resolved, source_components = parent._open_session(
            parent.SOURCE_ADAPTER, source_gate_root=source_gate_root
        )
        overfit_opened, overfit_frontend, overfit_config, overfit_resolved, overfit_components = parent._open_session(
            parent.OVERFIT_ADAPTER, source_gate_root=source_gate_root
        )
        require(source_resolved.fingerprint == overfit_resolved.fingerprint, "Stage B config drift")
        resident_after_two = int(torch.cuda.memory_allocated())
        require(resident_after_two < MAX_DEVICE_BYTES, "Stage B two-model resident allocation exceeds 48 GiB")
        image_receipts: list[dict[str, Any]] = []
        for image_id in STAGE_B_IMAGE_IDS:
            receipt = _stage_b_image(
                image_id=image_id,
                stage_root=output,
                source_opened=source_opened,
            source_components=source_components,
                source_frontend=source_frontend,
                source_config=source_config,
                overfit_opened=overfit_opened,
            overfit_components=overfit_components,
                overfit_frontend=overfit_frontend,
                overfit_config=overfit_config,
                stage_started_mono=started_mono,
            )
            image_receipts.append(receipt)
            progress["status"] = f"image_{image_id}_{receipt['status']}"
            progress["completed_image_ids"] = [int(item["image_id"]) for item in image_receipts]
            parent.atomic_json(output / "receipt.inprogress.json", progress)
        elapsed = time.perf_counter() - started_mono
        require(elapsed <= STAGE_B_GPU_BUDGET_SECONDS, f"Stage B exceeded GPU budget: {elapsed:.1f}s")
        candidate_images = [item for item in image_receipts if item["status"] == "mechanics_candidate"]
        excluded_images = [item for item in image_receipts if item["status"] == "scientifically_excluded"]
        aggregate_counts: dict[str, int] = {}
        for item in image_receipts:
            for key, value in item["counts"].items():
                if isinstance(value, int):
                    aggregate_counts[key] = aggregate_counts.get(key, 0) + value
        artifacts = _artifact_manifest(output)
        receipt = {
            **progress,
            "status": "mechanics_candidate",
            "completed_unix": time.time(),
            "elapsed_seconds": elapsed,
            "identity": {
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256_at_launch": runner_hash,
                "executed_input_sources": executed_input_sources,
                "runner_sha256_at_completion": parent.sha256_file(Path(__file__)),
                "helper_binding_scope": "maintained_package_sources_at_launch",
                "parent_helper_path": str(PARENT_HELPER),
                "parent_helper_sha256": PARENT_HELPER_SHA256,
                "panel_path": str(parent.PANEL),
                "panel_sha256": parent.sha256_file(parent.PANEL),
                "source_adapter": {"path": str(parent.SOURCE_ADAPTER), "sha256": parent.SOURCE_ADAPTER_SHA256},
                "overfit_adapter": {"path": str(parent.OVERFIT_ADAPTER), "sha256": parent.OVERFIT_ADAPTER_SHA256},
                "embedding_delta": {"path": str(parent.SOURCE_DELTA), "sha256": parent.SOURCE_DELTA_SHA256},
                "source_gate": source_gate,
            },
            "runtime": parent._runtime_identity(),
            "sessions": {
                "source": source_opened.receipt.to_artifact_dict(),
                "overfit": overfit_opened.receipt.to_artifact_dict(),
            },
            "counts": {
                **aggregate_counts,
                "intended_images": len(STAGE_B_IMAGE_IDS),
                "mechanics_candidate_images": len(candidate_images),
                "scientifically_excluded_images": len(excluded_images),
                "resident_models": 2,
            },
            "image_receipts": [
                {
                    "image_id": int(item["image_id"]),
                    "status": item["status"],
                    "path": str(output / "images" / f"image-{int(item['image_id']):012d}" / "receipt.json"),
                    "sha256": parent.sha256_file(
                        output / "images" / f"image-{int(item['image_id']):012d}" / "receipt.json"
                    ),
                }
                for item in image_receipts
            ],
            "checks": {
                "runner_unchanged": parent.sha256_file(Path(__file__)) == runner_hash,
                "all_intended_images_accounted": len(image_receipts) == len(STAGE_B_IMAGE_IDS),
                "image_ids_exact_and_ordered": tuple(int(item["image_id"]) for item in image_receipts)
                == STAGE_B_IMAGE_IDS,
                "all_nonexcluded_image_mechanics_passed": all(
                    all(item["checks"].values()) for item in candidate_images
                ),
                "all_image_receipts_cold_readable": all(
                    json.loads(
                        (output / "images" / f"image-{int(item['image_id']):012d}" / "receipt.json").read_text()
                    )["status"]
                    == item["status"]
                    for item in image_receipts
                ),
                "gpu_budget_passed": elapsed <= STAGE_B_GPU_BUDGET_SECONDS,
                "device_allocation_cap_passed": int(torch.cuda.max_memory_allocated()) < MAX_DEVICE_BYTES,
            },
            "resource": {
                "two_model_resident_allocated_bytes": resident_after_two,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "artifact_payload_bytes_excluding_receipt": sum(item["bytes"] for item in artifacts.values()),
            },
            "artifacts": artifacts,
            "claim_boundary": {
                "remaining_12_Human13_training_images": True,
                "fixed_middle_row_four_coordinate_sites": True,
                "per_image_reduction_only_lead_owns_image_equal_synthesis": True,
                "no_heldout_generalization_or_owner_recovery_claim": True,
            },
            "stop": "stage_B_candidate_complete_no_extra_ablations",
        }
        require(all(receipt["checks"].values()), "Stage B terminal checks failed")
        parent.atomic_json(output / "receipt.json", receipt)
        (output / "receipt.inprogress.json").unlink()
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "output_root": str(output),
                    "elapsed_seconds": elapsed,
                    "candidate_images": len(candidate_images),
                    "excluded_images": len(excluded_images),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        failure = {
            **progress,
            "status": "failed",
            "failed_unix": time.time(),
            "elapsed_seconds": time.perf_counter() - started_mono,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "resource": {
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
        }
        parent.atomic_json(output / "receipt.failed.json", failure)
        raise
    finally:
        for opened in (overfit_opened, source_opened):
            if opened is not None:
                try:
                    opened.close()
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()




def main() -> int:
    from probes.logit_lens.runtime import input_source_hashes

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage-b-output-root", type=Path)
    args = parser.parse_args()
    if args.stage_b_output_root is not None:
        return run_stage_b(args.stage_b_output_root)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner_sha256_at_launch = parent.sha256_file(Path(__file__))
    executed_input_sources = input_source_hashes()
    started_unix = time.time()
    started_mono = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inprogress: dict[str, Any] = {
        "schema_version": "logit_lens_causal_transfer_receipt.v1",
        "status": "running",
        "stage": "A_image2299_pilot",
        "started_unix": started_unix,
        "output_root": str(output),
        "launch_identity": {
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256_at_launch": runner_sha256_at_launch,
            "executed_input_sources": executed_input_sources,
            "helper_binding_scope": "maintained_package_sources_at_launch",
                "parent_helper_path": str(PARENT_HELPER),
            "parent_helper_sha256": PARENT_HELPER_SHA256,
            "trajectory_path": str(TRAJECTORY),
            "trajectory_file_sha256": TRAJECTORY_SHA256,
            "parent_receipt_path": str(PARENT_RECEIPT),
            "parent_receipt_sha256": PARENT_RECEIPT_SHA256,
        },
        "contract": {
            "trajectory": str(TRAJECTORY),
            "site_count": EXPECTED_SITE_COUNT,
            "blocks_1based": list(BLOCKS),
            "nonfinal_blocks_1based": list(NONFINAL_BLOCKS),
            "directions": ["overfit_to_source", "source_to_overfit"],
            "random_seeds": list(RANDOM_SEEDS),
            "atol": ATOL,
            "rtol": RTOL,
            "gpu_budget_seconds": GPU_BUDGET_SECONDS,
            "no_kv_cache": True,
            "full_prefix_equivalence": "one_whole_sequence_post_block_graft_read_at_all_selected_sites_under_causal_mask",
        },
    }
    parent.atomic_json(output / "receipt.inprogress.json", inprogress)
    bundles: list[ModelBundle] = []
    forward_counts = {
        "full_model_with_image_encoding": 0,
        "decoder_baseline_capture": 0,
        "self_patch": 0,
        "current": 0,
        "full_prefix": 0,
        "random_delta": 0,
        "block28_current_identity": 0,
    }
    try:
        require(torch.cuda.is_available(), "CUDA is required")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "GPU-8d43cb78-19ca-2f59-3179-7ea166cb1a4e"}, "not bound to physical GPU0")
        require(parent.sha256_file(PARENT_HELPER) == PARENT_HELPER_SHA256, "parent helper hash mismatch")
        require(parent.sha256_file(TRAJECTORY) == TRAJECTORY_SHA256, "frozen trajectory file hash mismatch")
        require(parent.sha256_file(PARENT_RECEIPT) == PARENT_RECEIPT_SHA256, "parent receipt hash mismatch")
        trajectory = json.loads(TRAJECTORY.read_text())
        require(trajectory["token_count"] == 415 and len(trajectory["token_ids"]) == 415, "trajectory length drift")
        require(trajectory["token_ids_sha256"] == parent.sha256_json(trajectory["token_ids"]), "trajectory token hash drift")

        source_gate_root, source_gate = parent._stage_source_gate(output)
        source = _open_and_capture(
            name="source", adapter=parent.SOURCE_ADAPTER, source_gate_root=source_gate_root, trajectory=trajectory
        )
        bundles.append(source)
        forward_counts["full_model_with_image_encoding"] += 1
        forward_counts["decoder_baseline_capture"] += 1
        require(time.perf_counter() - started_mono < GPU_BUDGET_SECONDS, "GPU budget exhausted after Source capture")
        overfit = _open_and_capture(
            name="overfit", adapter=parent.OVERFIT_ADAPTER, source_gate_root=source_gate_root, trajectory=trajectory
        )
        bundles.append(overfit)
        forward_counts["full_model_with_image_encoding"] += 1
        forward_counts["decoder_baseline_capture"] += 1
        require(source.input_receipt == overfit.input_receipt, "prompt/image identity differs across checkpoints")
        resident_after_two = int(torch.cuda.memory_allocated())
        require(resident_after_two < MAX_DEVICE_BYTES, f"two-model resident allocation exceeds 48 GiB: {resident_after_two}")

        prompt_count = int(source.input_receipt["prompt_token_count"])
        sites = coordinate_sites([0] * prompt_count, trajectory)
        selected = torch.tensor([int(site["position"]) for site in sites], dtype=torch.long)
        endpoints = build_endpoints(sites=sites, source=source, overfit=overfit)
        baseline = {
            "schema_version": "logit_lens_causal_transfer_baseline.v1",
            "trajectory_path": str(TRAJECTORY),
            "trajectory_file_sha256": TRAJECTORY_SHA256,
            "trajectory_token_ids_sha256": trajectory["token_ids_sha256"],
            "input": source.input_receipt,
            "sites": endpoints,
            "checkpoint_checks": {"source": source.baseline_checks, "overfit": overfit.baseline_checks},
        }
        parent.atomic_json(output / "baseline.json", baseline)
        compact = {
            "schema_version": "logit_lens_causal_transfer_compact_tensors.v1",
            "blocks_1based": list(BLOCKS),
            "sites": [{"site_index": i, **dict(site)} for i, site in enumerate(sites)],
            "baseline_logits": {
                "source": source.baseline_logits,
                "overfit": overfit.baseline_logits,
            },
            "selected_residuals": {
                checkpoint.name: torch.stack(
                    [checkpoint.states[block][0, selected, :] for block in BLOCKS]
                )
                for checkpoint in (source, overfit)
            },
        }
        torch.save(compact, output / "compact-selected.pt")
        inprogress["status"] = "baseline_complete"
        inprogress["counts"] = {**forward_counts, "trace_rows": 0}
        parent.atomic_json(output / "receipt.inprogress.json", inprogress)

        self_checks = [*_self_patch_checks(source, selected), *_self_patch_checks(overfit, selected)]
        forward_counts["self_patch"] += len(self_checks)
        trace_rows: list[dict[str, Any]] = []
        with (output / "patch-traces.jsonl").open("x") as trace_handle:
            for direction, donor, recipient in (
                ("overfit_to_source", overfit, source),
                ("source_to_overfit", source, overfit),
            ):
                for block in BLOCKS:
                    require(time.perf_counter() - started_mono < GPU_BUDGET_SECONDS, "GPU budget exhausted before patch group")
                    # Current-state donor transplant: one selected site per decoder forward.
                    for endpoint in endpoints:
                        index = int(endpoint["site_index"])
                        position = int(endpoint["position"])
                        logits, patch_receipt = _run_patch(
                            recipient=recipient,
                            block=block,
                            positions=[position],
                            replacement=donor.states[block][0, position, :].unsqueeze(0),
                            expected_before=recipient.states[block][0, position, :].unsqueeze(0),
                            selected=selected,
                        )
                        forward_counts["block28_current_identity" if block == 28 else "current"] += 1
                        earlier = selected < position
                        earlier_exact = bool(torch.equal(logits[:, earlier, :], recipient.baseline_logits[:, earlier, :]))
                        require(earlier_exact, f"future-position patch contaminated earlier logits block={block} site={index}")
                        if block == 28:
                            other = torch.ones(EXPECTED_SITE_COUNT, dtype=torch.bool)
                            other[index] = False
                            unpatched_exact = bool(torch.equal(logits[:, other, :], recipient.baseline_logits[:, other, :]))
                            require(unpatched_exact, f"block28 changed an unpatched selected logit site={index}")
                            parity = bool(
                                torch.allclose(
                                    logits[0, index], donor.baseline_logits[0, index], atol=ATOL, rtol=RTOL
                                )
                            )
                            parity_max = float((logits[0, index] - donor.baseline_logits[0, index]).abs().max().item())
                            require(parity, f"block28 donor full-vocab parity failed site={index}: {parity_max}")
                            patch_receipt = {
                                **patch_receipt,
                                "all_unpatched_selected_logits_exact": unpatched_exact,
                                "donor_full_vocab_parity_passed": parity,
                                "donor_full_vocab_max_abs": parity_max,
                                "atol": ATOL,
                                "rtol": RTOL,
                            }
                        else:
                            patch_receipt = {
                                **patch_receipt,
                                "all_earlier_selected_logits_exact": earlier_exact,
                                "later_selected_logits_are_not_used_by_this_site_isolated_arm": True,
                            }
                        row = _row_for_patch(
                            direction=direction,
                            donor=donor,
                            recipient=recipient,
                            endpoint=endpoint,
                            block=block,
                            scope="current",
                            patch_vector=logits[0, index],
                            patch_receipt=patch_receipt,
                            random_receipt=None,
                        )
                        trace_rows.append(row)
                        _jsonl_append(trace_handle, [row])

                    if block == 28:
                        continue

                    # Full causal-prefix splice: one whole-sequence graft is
                    # causally equivalent to per-site <=t grafts for selected
                    # readouts because later positions cannot influence earlier logits.
                    sequence_length = recipient.states[block].shape[1]
                    logits, patch_receipt = _run_patch(
                        recipient=recipient,
                        block=block,
                        positions=list(range(sequence_length)),
                        replacement=donor.states[block][0],
                        expected_before=recipient.states[block][0],
                        selected=selected,
                    )
                    forward_counts["full_prefix"] += 1
                    patch_receipt = {
                        **patch_receipt,
                        "scope_semantics": "all_visual_prompt_generated_history_and_current_positions",
                        "causal_equivalence": "whole_sequence_once_read_all_sites_equals_each_prefix_through_t",
                    }
                    rows = [
                        _row_for_patch(
                            direction=direction,
                            donor=donor,
                            recipient=recipient,
                            endpoint=endpoint,
                            block=block,
                            scope="full_prefix",
                            patch_vector=logits[0, int(endpoint["site_index"])],
                            patch_receipt=patch_receipt,
                            random_receipt=None,
                        )
                        for endpoint in endpoints
                    ]
                    trace_rows.extend(rows)
                    _jsonl_append(trace_handle, rows)

                    for endpoint in endpoints:
                        index = int(endpoint["site_index"])
                        position = int(endpoint["position"])
                        recipient_state = recipient.states[block][0, position, :]
                        donor_state = donor.states[block][0, position, :]
                        for seed in RANDOM_SEEDS:
                            replacement, random_receipt = _random_replacement(
                                recipient_state, donor_state, seed=seed
                            )
                            logits, patch_receipt = _run_patch(
                                recipient=recipient,
                                block=block,
                                positions=[position],
                                replacement=replacement.unsqueeze(0),
                                expected_before=recipient_state.unsqueeze(0),
                                selected=selected,
                            )
                            forward_counts["random_delta"] += 1
                            earlier = selected < position
                            earlier_exact = bool(torch.equal(logits[:, earlier, :], recipient.baseline_logits[:, earlier, :]))
                            require(earlier_exact, "random future-position patch contaminated earlier logits")
                            patch_receipt = {
                                **patch_receipt,
                                "all_earlier_selected_logits_exact": earlier_exact,
                                "later_selected_logits_are_not_used_by_this_site_isolated_arm": True,
                            }
                            row = _row_for_patch(
                                direction=direction,
                                donor=donor,
                                recipient=recipient,
                                endpoint=endpoint,
                                block=block,
                                scope="random_delta",
                                patch_vector=logits[0, index],
                                patch_receipt=patch_receipt,
                                random_receipt=random_receipt,
                            )
                            trace_rows.append(row)
                            _jsonl_append(trace_handle, [row])

                    inprogress["status"] = f"patch_complete_{direction}_block_{block}"
                    inprogress["counts"] = {**forward_counts, "trace_rows": len(trace_rows)}
                    parent.atomic_json(output / "receipt.inprogress.json", inprogress)

        elapsed = time.perf_counter() - started_mono
        require(elapsed <= GPU_BUDGET_SECONDS, f"Stage A exceeded GPU budget: {elapsed:.1f}s")
        summary = {
            "schema_version": "logit_lens_causal_transfer_summary.v1",
            "population_site_count": len(endpoints),
            "eligible_site_count": sum(bool(endpoint["eligible_for_R"]) for endpoint in endpoints),
            "excluded_sites": [
                {"site_index": endpoint["site_index"], "reason": endpoint["exclusion_reason"]}
                for endpoint in endpoints
                if not endpoint["eligible_for_R"]
            ],
            **reduce_trace(trace_rows),
        }
        parent.atomic_json(output / "summary.json", summary)

        # Cold readback and exact row/accounting validation before receipt.
        cold_rows = [json.loads(line) for line in (output / "patch-traces.jsonl").read_text().splitlines()]
        require(cold_rows == trace_rows, "cold trace readback differs from in-memory trace")
        cold_compact = torch.load(output / "compact-selected.pt", map_location="cpu", weights_only=False)
        require(cold_compact["schema_version"] == compact["schema_version"], "compact tensor cold readback failed")
        expected_trace_rows = 2 * (
            len(BLOCKS) * EXPECTED_SITE_COUNT
            + len(NONFINAL_BLOCKS) * EXPECTED_SITE_COUNT
            + len(NONFINAL_BLOCKS) * EXPECTED_SITE_COUNT * len(RANDOM_SEEDS)
        )
        require(len(trace_rows) == expected_trace_rows, f"trace row count mismatch: {len(trace_rows)} != {expected_trace_rows}")
        artifacts = _artifact_manifest(output)
        receipt = {
            **inprogress,
            "status": "mechanics_candidate",
            "completed_unix": time.time(),
            "elapsed_seconds": elapsed,
            "identity": {
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256_at_launch": runner_sha256_at_launch,
                "executed_input_sources": executed_input_sources,
                "runner_sha256_at_completion": parent.sha256_file(Path(__file__)),
                "helper_binding_scope": "maintained_package_sources_at_launch",
                "parent_helper_path": str(PARENT_HELPER),
                "parent_helper_sha256": PARENT_HELPER_SHA256,
                "parent_receipt_path": str(PARENT_RECEIPT),
                "parent_receipt_sha256": PARENT_RECEIPT_SHA256,
                "trajectory_path": str(TRAJECTORY),
                "trajectory_file_sha256": TRAJECTORY_SHA256,
                "trajectory_token_ids_sha256": trajectory["token_ids_sha256"],
                "source_gate": source_gate,
                "source_adapter": {"path": str(parent.SOURCE_ADAPTER), "sha256": parent.SOURCE_ADAPTER_SHA256},
                "overfit_adapter": {"path": str(parent.OVERFIT_ADAPTER), "sha256": parent.OVERFIT_ADAPTER_SHA256},
                "embedding_delta": {"path": str(parent.SOURCE_DELTA), "sha256": parent.SOURCE_DELTA_SHA256},
            },
            "runtime": parent._runtime_identity(),
            "sessions": {"source": source.session_receipt, "overfit": overfit.session_receipt},
            "input": source.input_receipt,
            "counts": {
                **forward_counts,
                "decoder_forward_total_including_full_model": sum(forward_counts.values()),
                "image_encoding_passes": forward_counts["full_model_with_image_encoding"],
                "trace_rows": len(trace_rows),
                "population_sites": len(endpoints),
                "eligible_sites": summary["eligible_site_count"],
                "self_patch_checks": len(self_checks),
                "resident_models": 2,
            },
            "checks": {
                "parent_hashes_passed": True,
                "same_prompt_and_image_across_models": True,
                "all_native_vs_capture_parity": all(bundle.baseline_checks["native_hooks_off_vs_direct_capture_passed"] for bundle in bundles),
                "all_self_patch_exact": all(check["selected_logits_exact"] for check in self_checks),
                "all_patch_before_identity_exact": all(row["residual_patch"]["before_exact"] for row in trace_rows),
                "all_patch_non_target_residual_exact": all(row["residual_patch"]["non_target_residual_exact"] for row in trace_rows),
                "all_current_earlier_logits_exact": all(
                    row["residual_patch"].get("all_earlier_selected_logits_exact", True)
                    for row in trace_rows
                    if row["scope"] in {"current", "random_delta"} and row["block_1based"] != 28
                ),
                "all_block28_donor_full_vocab_parity": all(
                    row["residual_patch"].get("donor_full_vocab_parity_passed", False)
                    for row in trace_rows
                    if row["block_1based"] == 28
                ),
                "all_block28_unpatched_selected_exact": all(
                    row["residual_patch"].get("all_unpatched_selected_logits_exact", False)
                    for row in trace_rows
                    if row["block_1based"] == 28
                ),
                "all_random_norms_matched": all(
                    row["random_delta"] is None or row["random_delta"]["norm_match_passed"]
                    for row in trace_rows
                ),
                "cold_readback_passed": True,
            },
            "self_patch_checks": self_checks,
            "resource": {
                "two_model_resident_allocated_bytes": resident_after_two,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "artifact_payload_bytes_excluding_receipt": sum(item["bytes"] for item in artifacts.values()),
            },
            "artifacts": artifacts,
            "claim_boundary": {
                "one_training_image": True,
                "teacher_forced_exact_overfit_prefix": True,
                "coordinate_endpoint_preference_not_accuracy": True,
                "no_owner_coverage_or_natural_generation_claim": True,
                "block28_is_algebraic_positive_control": True,
                "full_prefix_splice_may_be_out_of_distribution": True,
            },
            "stop": "stage_A_candidate_complete_stop_GPU_for_lead_decision",
        }
        require(
            receipt["identity"]["runner_sha256_at_completion"] == runner_sha256_at_launch,
            "runner changed during execution",
        )
        require(all(receipt["checks"].values()), "terminal mechanics checks failed")
        parent.atomic_json(output / "receipt.json", receipt)
        (output / "receipt.inprogress.json").unlink()
        print(json.dumps({"status": receipt["status"], "output_root": str(output), "elapsed_seconds": elapsed}, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {
            **inprogress,
            "status": "failed",
            "failed_unix": time.time(),
            "elapsed_seconds": time.perf_counter() - started_mono,
            "counts": {**forward_counts, "trace_rows": inprogress.get("counts", {}).get("trace_rows", 0)},
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
            "resource": {
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
        }
        parent.atomic_json(output / "receipt.failed.json", failure)
        raise
    finally:
        for bundle in reversed(bundles):
            try:
                bundle.opened.close()
            except Exception:
                pass
        bundles.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""One-shot block-27 overfit-direction graft followed by natural Source decode."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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
from src.qwen.generation import NativeGenerationPolicy, generate_continuations
from src.qwen.native import NativeBatch


ROOT = Path(__file__).resolve().parents[2]
CAUSAL_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-08-logit-lens-causal-transfer"
)
STAGE_A_ROOT = CAUSAL_ROOT / "run-v1"
STAGE_A_RECEIPT = STAGE_A_ROOT / "evaluation-v2/receipt.json"
STAGE_A_RECEIPT_SHA256 = "082ffdb2ba027869168dac268764559f90e156ed5ed5a84ee94560781d167db1"
STAGE_B_ROOT = CAUSAL_ROOT / "stage-b-v1"
STAGE_B_RECEIPT = STAGE_B_ROOT / "receipt.json"
STAGE_B_RECEIPT_SHA256 = "974980e16baaa45fd292c503449ca8676f84bd2e3436f0d42a90ff743b44231a"
RADIUS_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-08-logit-lens-radius-direction/final-v1"
)
RADIUS_RECEIPT = RADIUS_ROOT / "receipt.json"
RADIUS_RECEIPT_SHA256 = "d6262dcab25b6dbe53e975e6d661b56c8b6f3adca031e8821b363a242693e3ed"
IMAGE2299_TRAJECTORY = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-08-image2299-logit-lens/run-v2/trajectory-overfit.json"
)
IMAGE2299_TRAJECTORY_SHA256 = "537079135d3b12a3bfd72778ea63352112cde60775405e702abbbfa71efce063"
DEFAULT_OUTPUT = Path("/tmp/coordexp-logit-lens-natural-new")

IMAGE_IDS = (2299, 1584, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228)
SMOKE_IMAGE_ID = 2299
BLOCK = 27
TOTAL_GENERATED_CAP = 768
ATOL = 2e-4
RTOL = 2e-4
GPU_BUDGET_SECONDS = 20 * 60
MAX_DEVICE_BYTES = 48 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


from probes.logit_lens import causal, base as parent

PARENT_HELPER = Path(parent.__file__)
PARENT_HELPER_SHA256 = parent.sha256_file(PARENT_HELPER)
CAUSAL_HELPER = Path(causal.__file__)
CAUSAL_HELPER_SHA256 = parent.sha256_file(CAUSAL_HELPER)


def atomic_json(path: Path, value: Any) -> None:
    parent.atomic_json(path, value)


def direction_only_state(source_state: torch.Tensor, donor_state: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    source = source_state.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    donor = donor_state.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    require(source.shape == donor.shape and source.numel() > 0, "state shape mismatch")
    source_radius = source.norm()
    donor_radius = donor.norm()
    require(torch.isfinite(source_radius).item() and float(source_radius) > 0.0, "zero/nonfinite source radius")
    require(torch.isfinite(donor_radius).item() and float(donor_radius) > 0.0, "zero/nonfinite donor radius")
    source_unit = source / source_radius
    donor_unit = donor / donor_radius
    replacement = source_radius * donor_unit
    actual_radius = replacement.norm()
    actual_unit = replacement / actual_radius
    radius_passed = bool(torch.allclose(actual_radius, source_radius, atol=ATOL, rtol=RTOL))
    direction_passed = bool(torch.allclose(actual_unit, donor_unit, atol=ATOL, rtol=RTOL))
    require(radius_passed, "direction-only state changed Source radius")
    require(direction_passed, "direction-only state failed donor direction")
    return replacement, {
        "source_state_sha256": parent.tensor_sha256(source),
        "donor_state_sha256": parent.tensor_sha256(donor),
        "replacement_state_sha256": parent.tensor_sha256(replacement),
        "source_radius": float(source_radius.item()),
        "donor_radius": float(donor_radius.item()),
        "replacement_radius": float(actual_radius.item()),
        "radius_absolute_difference": float(abs(actual_radius.item() - source_radius.item())),
        "unit_direction_max_absolute_difference": float((actual_unit - donor_unit).abs().max().item()),
        "source_donor_unit_cosine": float(torch.dot(source_unit, donor_unit).item()),
        "radius_passed": radius_passed,
        "direction_passed": direction_passed,
        "atol": ATOL,
        "rtol": RTOL,
    }


def one_shot_patch(
    module: Any,
    *,
    position: int,
    replacement: torch.Tensor,
    expected_before: torch.Tensor,
) -> Any:
    return causal.ResidualPatch(
        module,
        positions=[int(position)],
        replacement=replacement.reshape(1, -1),
        expected_before=expected_before.reshape(1, -1),
    )




class GenerationCallTrace:
    """Bounded evidence that generate used one full prefill then its KV cache."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.handle: Any = None
        self.calls: list[dict[str, Any]] = []

    def _hook(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        tensor = input_ids if isinstance(input_ids, torch.Tensor) else inputs_embeds
        require(isinstance(tensor, torch.Tensor) and tensor.ndim >= 2, "generation forward lacks token-width tensor")
        past = kwargs.get("past_key_values")
        past_length = None
        if past is not None and callable(getattr(past, "get_seq_length", None)):
            past_length = int(past.get_seq_length())
        self.calls.append(
            {
                "input_width": int(tensor.shape[1]),
                "used_input_ids": isinstance(input_ids, torch.Tensor),
                "past_key_values_present": past is not None,
                "past_sequence_length": past_length,
                "pixel_values_present": isinstance(kwargs.get("pixel_values"), torch.Tensor),
                "use_cache": kwargs.get("use_cache"),
            }
        )

    def __enter__(self) -> "GenerationCallTrace":
        self.handle = self.model.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        require(self.calls, "generation model forward was not observed")

    def validate(self, *, full_input_width: int, generated_count: int) -> dict[str, Any]:
        require(len(self.calls) == generated_count, "generation forward/token count mismatch")
        prefill = self.calls[0]
        prefill_passed = prefill["input_width"] == full_input_width and prefill["pixel_values_present"]
        require(prefill_passed, "generation prefill did not consume full multimodal prefix")
        cached = self.calls[1:]
        cached_passed = all(
            item["input_width"] == 1
            and item["past_key_values_present"]
            and (item["past_sequence_length"] is None or item["past_sequence_length"] > 0)
            for item in cached
        )
        require(cached_passed, "generation follow-up did not use one-token KV-cache steps")
        return {
            "model_forward_count": len(self.calls),
            "prefill_full_width_and_image_passed": prefill_passed,
            "cached_followup_count": len(cached),
            "cached_followups_passed": cached_passed,
            "cache_followup_observed": bool(cached),
            "calls": self.calls,
        }


def _finish_reason(new_ids: Sequence[int], *, cap: int, im_end_id: int) -> str:
    if len(new_ids) >= cap:
        return "length"
    if new_ids and int(new_ids[-1]) == im_end_id:
        return "im_end"
    return "nonterminal_return"


def _generate_source_branch(
    *,
    components: Any,
    native_inputs: Mapping[str, Any],
    generated_prefix_ids: Sequence[int],
    suffix_cap: int,
    tokenizer: Any,
    visual_module: Any,
    patch: Any | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    model = components.model
    batch = NativeBatch(inputs=native_inputs, request_ids=("source-branch",))
    input_width = len(batch.prompt_token_ids[0]) + len(generated_prefix_ids)
    im_end = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    pad_id = tokenizer.pad_token_id
    require(isinstance(pad_id, int) and pad_id >= 0, "tokenizer has no pad token ID")
    require(bool(getattr(model.generation_config, "use_cache", True)), "natural profile requires KV cache")
    def generate():
        return generate_continuations(
            model, batch, extensions=[generated_prefix_ids], budgets=[int(suffix_cap)],
            eos_token_id=im_end, pad_token_id=pad_id,
            policy=NativeGenerationPolicy(temperature=0., top_p=1., repetition_penalty=1.),
            trace="none",
        )[0]
    started = time.perf_counter()
    with GenerationCallTrace(model) as call_trace, causal.ModuleCallCounter(visual_module) as visual_counter:
        if patch is None:
            output = generate()
            patch_receipt = None
        else:
            with patch:
                output = generate()
            patch_receipt = patch.receipt()
    elapsed = time.perf_counter() - started
    require(visual_counter.calls == 1, "generation branch encoded the image other than once")
    new_ids = list(output.token_ids)
    require(new_ids, "generate returned an empty continuation")
    cache = call_trace.validate(full_input_width=input_width, generated_count=len(new_ids))
    full_generated = [*map(int, generated_prefix_ids), *new_ids]
    require(len(full_generated) <= TOTAL_GENERATED_CAP, "full generated output exceeded frozen cap")
    branch = {
        "new_token_ids": new_ids,
        "new_token_ids_sha256": parent.sha256_json(new_ids),
        "new_text": str(tokenizer.decode(new_ids, skip_special_tokens=False)),
        "full_generated_token_ids": full_generated,
        "full_generated_token_ids_sha256": parent.sha256_json(full_generated),
        "full_generated_text": str(tokenizer.decode(full_generated, skip_special_tokens=False)),
        "finish_reason": _finish_reason(new_ids, cap=suffix_cap, im_end_id=im_end),
        "cap_hit": len(new_ids) >= suffix_cap,
        "emitted_im_end": bool(new_ids and new_ids[-1] == im_end),
        "suffix_cap": int(suffix_cap),
        "new_token_count": len(new_ids),
        "full_generated_token_count": len(full_generated),
        "elapsed_seconds": elapsed,
        "vision_encoder_forward_calls": visual_counter.calls,
        "generation_cache": cache,
    }
    return branch, patch_receipt


def _prior_paths(image_id: int) -> dict[str, Path]:
    if image_id == 2299:
        return {
            "trajectory": IMAGE2299_TRAJECTORY,
            "baseline": STAGE_A_ROOT / "baseline.json",
            "radius_trace": RADIUS_ROOT / "images/image-000000002299/trace.jsonl",
        }
    base = STAGE_B_ROOT / "images" / f"image-{image_id:012d}"
    return {
        "trajectory": base / "trajectory-overfit.json",
        "baseline": base / "baseline.json",
        "radius_trace": RADIUS_ROOT / "images" / f"image-{image_id:012d}" / "trace.jsonl",
    }


def _verify_prior_hash(path: Path) -> str:
    digest = sha256_file(path)
    if path == IMAGE2299_TRAJECTORY:
        require(digest == IMAGE2299_TRAJECTORY_SHA256, "Image2299 trajectory hash mismatch")
        return digest
    if STAGE_B_ROOT in path.parents:
        receipt = json.loads(STAGE_B_RECEIPT.read_text())
        key = str(path.relative_to(STAGE_B_ROOT))
        require(receipt["artifacts"][key]["sha256"] == digest, f"Stage B artifact hash mismatch {key}")
        return digest
    if STAGE_A_ROOT in path.parents:
        receipt = json.loads(STAGE_A_RECEIPT.read_text())
        key = str(path.relative_to(CAUSAL_ROOT))
        require(receipt["raw_artifacts"][key]["sha256"] == digest, f"Stage A artifact hash mismatch {key}")
        return digest
    if RADIUS_ROOT in path.parents:
        receipt = json.loads(RADIUS_RECEIPT.read_text())
        key = str(path.relative_to(RADIUS_ROOT))
        require(receipt["artifacts"][key]["sha256"] == digest, f"radius artifact hash mismatch {key}")
        return digest
    raise RuntimeError(f"unowned prior artifact {path}")


def _load_anchor(image_id: int) -> dict[str, Any]:
    paths = _prior_paths(image_id)
    hashes = {key: _verify_prior_hash(path) for key, path in paths.items()}
    trajectory = json.loads(paths["trajectory"].read_text())
    baseline = json.loads(paths["baseline"].read_text())
    candidates = [
        site
        for site in baseline["sites"]
        if "middle_row_coord_1_decision" in site["labels"]
    ]
    require(len(candidates) == 1, f"middle coord1 anchor count changed image={image_id}")
    site = dict(candidates[0])
    prompt_count = int(baseline["input"]["prompt_token_count"])
    prefix_count = int(site["position"]) - prompt_count + 1
    require(0 < prefix_count < TOTAL_GENERATED_CAP, "generated prefix length outside frozen cap")
    prefix_ids = [int(value) for value in trajectory["token_ids"][:prefix_count]]
    require(len(prefix_ids) == prefix_count, "trajectory is shorter than selected prefix")
    actual = int(trajectory["token_ids"][prefix_count])
    require(actual == int(site["trajectory_actual_next_token_id"]), "selected token alignment changed")
    radius_rows = [json.loads(line) for line in paths["radius_trace"].read_text().splitlines()]
    anchors = [
        row
        for row in radius_rows
        if row["direction"] == "overfit_to_source"
        and int(row["block_1based"]) == BLOCK
        and int(row["site_index"]) == int(site["site_index"])
        and row["scope"] == "direction_only"
    ]
    require(len(anchors) == 1, "prior direction-only anchor count changed")
    return {
        "image_id": image_id,
        "paths": paths,
        "hashes": hashes,
        "trajectory": trajectory,
        "baseline": baseline,
        "site": site,
        "generated_prefix_ids": prefix_ids,
        "generated_prefix_count": prefix_count,
        "suffix_cap": TOTAL_GENERATED_CAP - prefix_count,
        "prior_direction_anchor": anchors[0],
    }


def _artifact_manifest(root: Path) -> dict[str, Any]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"receipt.json", "receipt.inprogress.json"}:
            result[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return result


def _run_image(
    *,
    image_id: int,
    output_root: Path,
    source_opened: Any,
    source_components: Any,
    source_frontend: Any,
    source_config: Any,
    overfit_opened: Any,
    overfit_components: Any,
    overfit_frontend: Any,
    overfit_config: Any,
    stage_started: float,
) -> dict[str, Any]:
    image_root = output_root / "images" / f"image-{image_id:012d}"
    image_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    progress: dict[str, Any] = {
        "schema_version": "logit_lens_natural_continuation_image_receipt.v1",
        "status": "running",
        "image_id": image_id,
    }
    atomic_json(image_root / "receipt.inprogress.json", progress)
    counts = {
        "full_model_no_cache_capture": 0,
        "direct_text_no_cache_capture": 0,
        "natural_generation_branches": 0,
        "generation_model_forward_calls": 0,
        "generated_new_tokens": 0,
        "vision_encoder_forward_calls": 0,
        "direction_graft_hook_calls": 0,
        "self_graft_hook_calls": 0,
    }
    try:
        require(time.perf_counter() - stage_started < GPU_BUDGET_SECONDS, "GPU budget exhausted before image")
        anchor = _load_anchor(image_id)
        source_request, source_native, source_prompt, source_input = causal.request_and_inputs_for_image(
            components=source_components,
            frontend=source_frontend,
            config=source_config,
            image_id=image_id,
            request_id=f"logit-lens-natural-source-{image_id}",
        )
        overfit_request, overfit_native, overfit_prompt, overfit_input = causal.request_and_inputs_for_image(
            components=overfit_components,
            frontend=overfit_frontend,
            config=overfit_config,
            image_id=image_id,
            request_id=f"logit-lens-natural-overfit-{image_id}",
        )
        require(source_input == overfit_input == anchor["baseline"]["input"], "input identity differs from frozen prior")
        require(source_prompt == overfit_prompt, "prompt token IDs differ across checkpoints")
        prefix_ids = anchor["generated_prefix_ids"]
        site = anchor["site"]
        require(int(site["position"]) == len(source_prompt) + len(prefix_ids) - 1, "predictor position mismatch")
        trajectory_prefix = {"token_ids": prefix_ids}
        selected_site = {
            "position": int(site["position"]),
            "labels": list(site["labels"]),
            "actual_next_token_id": int(site["trajectory_actual_next_token_id"]),
        }
        source_visual, source_visual_aliases = causal.resolve_visual_module(source_components.model)  # noqa: SLF001
        overfit_visual, overfit_visual_aliases = causal.resolve_visual_module(overfit_components.model)  # noqa: SLF001
        with causal.ModuleCallCounter(source_visual) as source_capture_visual:
            source = causal._capture_existing_session(
                name="source",
                opened=source_opened, components=source_components,
                native_inputs=source_native,
                prompt_ids=source_prompt,
                trajectory=trajectory_prefix,
                sites=[selected_site],
                input_receipt=source_input,
                blocks=(BLOCK,),
            )
        with causal.ModuleCallCounter(overfit_visual) as overfit_capture_visual:
            overfit = causal._capture_existing_session(
                name="overfit",
                opened=overfit_opened, components=overfit_components,
                native_inputs=overfit_native,
                prompt_ids=overfit_prompt,
                trajectory=trajectory_prefix,
                sites=[selected_site],
                input_receipt=overfit_input,
                blocks=(BLOCK,),
            )
        require(source_capture_visual.calls == 1 and overfit_capture_visual.calls == 1, "capture vision count changed")
        counts["full_model_no_cache_capture"] = 2
        counts["direct_text_no_cache_capture"] = 2
        counts["vision_encoder_forward_calls"] = 2
        source_state = source.states[BLOCK][0, int(site["position"]), :]
        donor_state = overfit.states[BLOCK][0, int(site["position"]), :]
        replacement, state_receipt = direction_only_state(source_state, donor_state)
        no_cache_source_top1 = int(source.baseline_logits[0, 0].argmax().item())
        no_cache_donor_top1 = int(overfit.baseline_logits[0, 0].argmax().item())
        require(
            no_cache_source_top1 == int(site["s_source_top1_token_id"]),
            "fresh Source next-token identity differs from frozen anchor",
        )
        require(
            no_cache_donor_top1 == int(site["a_overfit_top1_token_id"]),
            "fresh overfit next-token identity differs from frozen anchor",
        )
        input_record = {
            "schema_version": "logit_lens_natural_continuation_input.v1",
            "image_id": image_id,
            "prompt_token_ids": [int(value) for value in source_prompt],
            "prompt_token_ids_sha256": parent.sha256_json(source_prompt),
            "prompt_chat_text": source_request.chat_text,
            "prompt_chat_text_sha256": hashlib.sha256(source_request.chat_text.encode()).hexdigest(),
            "input_identity": source_input,
            "generated_prefix_token_ids": prefix_ids,
            "generated_prefix_token_ids_sha256": parent.sha256_json(prefix_ids),
            "generated_prefix_text": str(source_components.tokenizer.decode(prefix_ids, skip_special_tokens=False)),  # noqa: SLF001
            "generated_prefix_token_count": len(prefix_ids),
            "suffix_cap": anchor["suffix_cap"],
            "total_generated_cap": TOTAL_GENERATED_CAP,
            "selected_site": site,
            "prior_artifacts": {
                key: {"path": str(anchor["paths"][key]), "sha256": value}
                for key, value in anchor["hashes"].items()
            },
            "visual_module_aliases": {
                "source": source_visual_aliases,
                "overfit": overfit_visual_aliases,
            },
        }
        atomic_json(image_root / "input.json", input_record)
        torch.save(
            {
                "schema_version": "logit_lens_natural_continuation_states.v1",
                "image_id": image_id,
                "block_1based": BLOCK,
                "position": int(site["position"]),
                "source_state": source_state,
                "overfit_donor_state": donor_state,
                "direction_only_state": replacement,
                "source_selected_logits": source.baseline_logits,
                "overfit_selected_logits": overfit.baseline_logits,
            },
            image_root / "states.pt",
        )

        baseline, baseline_patch = _generate_source_branch(
            components=source_components,
            native_inputs=source_native,
            generated_prefix_ids=prefix_ids,
            suffix_cap=anchor["suffix_cap"],
            tokenizer=source_components.tokenizer,  # noqa: SLF001
            visual_module=source_visual,
            patch=None,
        )
        require(baseline_patch is None, "baseline unexpectedly has patch receipt")
        counts["natural_generation_branches"] += 1
        counts["generation_model_forward_calls"] += baseline["generation_cache"]["model_forward_count"]
        counts["generated_new_tokens"] += baseline["new_token_count"]
        counts["vision_encoder_forward_calls"] += baseline["vision_encoder_forward_calls"]
        require(baseline["new_token_ids"][0] == no_cache_source_top1, "Source native prefill/cache first token mismatch")

        smoke_self = None
        smoke_self_patch = None
        if image_id == SMOKE_IMAGE_ID:
            self_patch = one_shot_patch(
                source.layers[BLOCK - 1],
                position=int(site["position"]),
                replacement=source_state,
                expected_before=source_state,
            )
            smoke_self, smoke_self_patch = _generate_source_branch(
                components=source_components,
                native_inputs=source_native,
                generated_prefix_ids=prefix_ids,
                suffix_cap=anchor["suffix_cap"],
                tokenizer=source_components.tokenizer,  # noqa: SLF001
                visual_module=source_visual,
                patch=self_patch,
            )
            counts["natural_generation_branches"] += 1
            counts["generation_model_forward_calls"] += smoke_self["generation_cache"]["model_forward_count"]
            counts["generated_new_tokens"] += smoke_self["new_token_count"]
            counts["vision_encoder_forward_calls"] += smoke_self["vision_encoder_forward_calls"]
            counts["self_graft_hook_calls"] = int(smoke_self_patch["hook_calls"])
            require(smoke_self["new_token_ids"] == baseline["new_token_ids"], "self-graft continuation differs from native")
            require(smoke_self["finish_reason"] == baseline["finish_reason"], "self-graft finish differs from native")
            require(smoke_self["generation_cache"]["cache_followup_observed"], "smoke self branch lacked cached follow-up")

        patch = one_shot_patch(
            source.layers[BLOCK - 1],
            position=int(site["position"]),
            replacement=replacement,
            expected_before=source_state,
        )
        direction, direction_patch = _generate_source_branch(
            components=source_components,
            native_inputs=source_native,
            generated_prefix_ids=prefix_ids,
            suffix_cap=anchor["suffix_cap"],
            tokenizer=source_components.tokenizer,  # noqa: SLF001
            visual_module=source_visual,
            patch=patch,
        )
        counts["natural_generation_branches"] += 1
        counts["generation_model_forward_calls"] += direction["generation_cache"]["model_forward_count"]
        counts["generated_new_tokens"] += direction["new_token_count"]
        counts["vision_encoder_forward_calls"] += direction["vision_encoder_forward_calls"]
        counts["direction_graft_hook_calls"] = int(direction_patch["hook_calls"])
        prior_first = int(anchor["prior_direction_anchor"]["patch_endpoint"]["top1_token_id"])
        require(direction["new_token_ids"][0] == prior_first, "direction graft prefill/cache first token differs from prior no-cache anchor")

        trajectory_record = {
            "schema_version": "logit_lens_natural_continuation_trajectory.v1",
            "image_id": image_id,
            "prefix_token_ids": prefix_ids,
            "prefix_token_ids_sha256": parent.sha256_json(prefix_ids),
            "prefix_text": input_record["generated_prefix_text"],
            "prompt_token_ids": input_record["prompt_token_ids"],
            "prompt_token_ids_sha256": input_record["prompt_token_ids_sha256"],
            "selected_site": site,
            "total_generated_cap": TOTAL_GENERATED_CAP,
            "suffix_cap": anchor["suffix_cap"],
            "arms": {
                "baseline": baseline,
                "direction_only": direction,
            },
            "mechanics_self_arm": smoke_self,
        }
        atomic_json(image_root / "trajectory.json", trajectory_record)
        mechanics = {
            "schema_version": "logit_lens_natural_continuation_mechanics.v1",
            "image_id": image_id,
            "state": state_receipt,
            "source_capture": source.baseline_checks,
            "overfit_capture": overfit.baseline_checks,
            "no_cache_source_top1_token_id": no_cache_source_top1,
            "no_cache_overfit_top1_token_id": no_cache_donor_top1,
            "prior_direction_only_top1_token_id": prior_first,
            "baseline_first_token_cache_parity": baseline["new_token_ids"][0] == no_cache_source_top1,
            "direction_first_token_cache_parity": direction["new_token_ids"][0] == prior_first,
            "direction_patch": direction_patch,
            "smoke_self_patch": smoke_self_patch,
            "smoke_self_exact_continuation_parity": None
            if smoke_self is None
            else smoke_self["new_token_ids"] == baseline["new_token_ids"],
            "same_prefix_all_branches": True,
            "donor_prefix_contains_no_selected_or_future_tokens": True,
        }
        atomic_json(image_root / "mechanics.json", mechanics)
        cold = json.loads((image_root / "trajectory.json").read_text())
        require(cold == trajectory_record, "trajectory cold readback mismatch")
        cold_states = torch.load(image_root / "states.pt", map_location="cpu", weights_only=False)
        require(cold_states["image_id"] == image_id, "state cold readback mismatch")
        checks = {
            "input_matches_frozen_prior": True,
            "selected_site_is_middle_coord1_by_position": True,
            "same_prefix_all_branches": True,
            "source_and_donor_capture_native_text_parity": all(
                bundle.baseline_checks["native_hooks_off_vs_direct_capture_passed"]
                for bundle in (source, overfit)
            ),
            "direction_state_preserves_source_radius": state_receipt["radius_passed"],
            "direction_state_matches_donor_unit": state_receipt["direction_passed"],
            "direction_hook_fired_once": direction_patch["hook_calls"] == 1,
            "direction_hook_before_state_exact": direction_patch["before_exact"],
            "direction_hook_non_target_exact": direction_patch["non_target_residual_exact"],
            "baseline_native_cache_first_token_parity": mechanics["baseline_first_token_cache_parity"],
            "direction_prior_no_cache_first_token_parity": mechanics["direction_first_token_cache_parity"],
            "normal_kv_cache_path": baseline["generation_cache"]["cached_followups_passed"]
            and direction["generation_cache"]["cached_followups_passed"],
            "terminal_or_cap_baseline": baseline["finish_reason"] in {"im_end", "length"},
            "terminal_or_cap_direction": direction["finish_reason"] in {"im_end", "length"},
            "full_outputs_within_total_cap": baseline["full_generated_token_count"] <= TOTAL_GENERATED_CAP
            and direction["full_generated_token_count"] <= TOTAL_GENERATED_CAP,
            "cold_readback": True,
        }
        if image_id == SMOKE_IMAGE_ID:
            checks.update(
                {
                    "smoke_self_hook_once": smoke_self_patch["hook_calls"] == 1,
                    "smoke_self_state_exact": smoke_self_patch["before_exact"],
                    "smoke_self_continuation_exact": mechanics["smoke_self_exact_continuation_parity"],
                    "smoke_real_cache_followup_observed": baseline["generation_cache"]["cache_followup_observed"]
                    and direction["generation_cache"]["cache_followup_observed"]
                    and smoke_self["generation_cache"]["cache_followup_observed"],
                }
            )
        require(all(checks.values()), f"image mechanics failed {image_id}: {[key for key, value in checks.items() if not value]}")
        artifacts = _artifact_manifest(image_root)
        receipt = {
            **progress,
            "status": "mechanics_candidate",
            "elapsed_seconds": time.perf_counter() - started,
            "input": source_input,
            "prefix": {
                "token_count": len(prefix_ids),
                "token_ids_sha256": parent.sha256_json(prefix_ids),
                "selected_position": int(site["position"]),
                "suffix_cap": anchor["suffix_cap"],
                "total_generated_cap": TOTAL_GENERATED_CAP,
            },
            "arms": {
                name: {
                    key: arm[key]
                    for key in (
                        "new_token_ids_sha256",
                        "full_generated_token_ids_sha256",
                        "new_token_count",
                        "full_generated_token_count",
                        "finish_reason",
                        "cap_hit",
                        "emitted_im_end",
                        "elapsed_seconds",
                    )
                }
                for name, arm in (("baseline", baseline), ("direction_only", direction))
            },
            "counts": counts,
            "checks": checks,
            "mechanics_path": str(image_root / "mechanics.json"),
            "mechanics_sha256": sha256_file(image_root / "mechanics.json"),
            "trajectory_path": str(image_root / "trajectory.json"),
            "trajectory_sha256": sha256_file(image_root / "trajectory.json"),
            "artifacts": artifacts,
            "resource": {
                "peak_cuda_allocated_bytes_stage_so_far": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes_stage_so_far": int(torch.cuda.max_memory_reserved()),
                "peak_host_rss_kib_stage_so_far": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "artifact_payload_bytes_excluding_receipt": sum(item["bytes"] for item in artifacts.values()),
            },
            "claim_boundary": {
                "paired_Source_natural_continuations": True,
                "one_block27_direction_graft_on_prefill_only": True,
                "same_frozen_overfit_prefix_training_image": True,
                "raw_outputs_not_owner_scoring": True,
            },
        }
        atomic_json(image_root / "receipt.json", receipt)
        (image_root / "receipt.inprogress.json").unlink()
        return receipt
    except BaseException as error:
        failure = {
            **progress,
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        atomic_json(image_root / "receipt.failed.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started_unix = time.time()
    started = time.perf_counter()
    runner_hash = sha256_file(Path(__file__))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    progress: dict[str, Any] = {
        "schema_version": "logit_lens_natural_continuation_receipt.v1",
        "status": "running",
        "started_unix": started_unix,
        "output_root": str(output),
        "image_ids": list(IMAGE_IDS),
        "smoke_image_id": SMOKE_IMAGE_ID,
        "runner_sha256_at_launch": runner_hash,
        "contract": {
            "source_checkpoint_only_for_both_natural_arms": True,
            "donor_checkpoint": "overfit_state_capture_only",
            "block_1based": BLOCK,
            "intervention": "one_prefill_current_position_direction_only_preserve_source_radius",
            "total_generated_cap": TOTAL_GENERATED_CAP,
            "repetition_penalty": 1.0,
            "greedy": True,
            "normal_kv_cache_after_prefill": True,
            "gpu_budget_seconds": GPU_BUDGET_SECONDS,
            "atol": ATOL,
            "rtol": RTOL,
        },
    }
    atomic_json(output / "receipt.inprogress.json", progress)
    source_opened: Any | None = None
    overfit_opened: Any | None = None
    try:
        require(torch.cuda.is_available(), "CUDA is required")
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "GPU-8d43cb78-19ca-2f59-3179-7ea166cb1a4e"},
            "not bound to physical GPU0",
        )
        require(sha256_file(PARENT_HELPER) == PARENT_HELPER_SHA256, "parent helper hash mismatch")
        require(sha256_file(CAUSAL_HELPER) == CAUSAL_HELPER_SHA256, "causal helper hash mismatch")
        require(sha256_file(STAGE_A_RECEIPT) == STAGE_A_RECEIPT_SHA256, "Stage A receipt hash mismatch")
        require(sha256_file(STAGE_B_RECEIPT) == STAGE_B_RECEIPT_SHA256, "Stage B receipt hash mismatch")
        require(sha256_file(RADIUS_RECEIPT) == RADIUS_RECEIPT_SHA256, "radius receipt hash mismatch")
        source_gate_root, source_gate = parent._stage_source_gate(output)
        source_opened, source_frontend, source_config, source_resolved, source_components = parent._open_session(
            parent.SOURCE_ADAPTER, source_gate_root=source_gate_root
        )
        overfit_opened, overfit_frontend, overfit_config, overfit_resolved, overfit_components = parent._open_session(
            parent.OVERFIT_ADAPTER, source_gate_root=source_gate_root
        )
        require(source_resolved.fingerprint == overfit_resolved.fingerprint, "config drift")
        resident_after_two = int(torch.cuda.memory_allocated())
        require(resident_after_two < MAX_DEVICE_BYTES, "two-model allocation exceeds 48 GiB")
        receipts = []
        for index, image_id in enumerate(IMAGE_IDS):
            receipt = _run_image(
                image_id=image_id,
                output_root=output,
                source_opened=source_opened,
            source_components=source_components,
                source_frontend=source_frontend,
                source_config=source_config,
                overfit_opened=overfit_opened,
            overfit_components=overfit_components,
                overfit_frontend=overfit_frontend,
                overfit_config=overfit_config,
                stage_started=started,
            )
            receipts.append(receipt)
            if index == 0:
                require(image_id == SMOKE_IMAGE_ID, "first image is not frozen smoke")
                require(receipt["checks"]["smoke_self_continuation_exact"], "production-shaped smoke failed")
            progress["status"] = f"image_{image_id}_complete"
            progress["completed_image_ids"] = [int(item["image_id"]) for item in receipts]
            atomic_json(output / "receipt.inprogress.json", progress)
        elapsed = time.perf_counter() - started
        require(elapsed <= GPU_BUDGET_SECONDS, f"GPU budget exceeded: {elapsed:.1f}s")
        aggregate: dict[str, int] = {}
        for receipt in receipts:
            for key, value in receipt["counts"].items():
                aggregate[key] = aggregate.get(key, 0) + int(value)
        artifacts = _artifact_manifest(output)
        terminal = {
            **progress,
            "status": "mechanics_candidate",
            "completed_unix": time.time(),
            "elapsed_seconds": elapsed,
            "identity": {
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256_at_launch": runner_hash,
                "runner_sha256_at_completion": sha256_file(Path(__file__)),
                "causal_helper_path": str(CAUSAL_HELPER),
                "causal_helper_sha256": CAUSAL_HELPER_SHA256,
                "helper_binding_scope": "maintained_package_sources_at_launch",
                "parent_helper_path": str(PARENT_HELPER),
                "parent_helper_sha256": PARENT_HELPER_SHA256,
                "stage_a_receipt": {"path": str(STAGE_A_RECEIPT), "sha256": STAGE_A_RECEIPT_SHA256},
                "stage_b_receipt": {"path": str(STAGE_B_RECEIPT), "sha256": STAGE_B_RECEIPT_SHA256},
                "radius_receipt": {"path": str(RADIUS_RECEIPT), "sha256": RADIUS_RECEIPT_SHA256},
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
                **aggregate,
                "images": len(receipts),
                "paired_natural_trajectories": 13,
                "scientific_generation_arms": 26,
                "smoke_self_generation_arms": 1,
                "resident_models": 2,
            },
            "image_receipts": [
                {
                    "image_id": int(receipt["image_id"]),
                    "path": str(output / "images" / f"image-{int(receipt['image_id']):012d}" / "receipt.json"),
                    "sha256": sha256_file(
                        output / "images" / f"image-{int(receipt['image_id']):012d}" / "receipt.json"
                    ),
                    "trajectory_path": receipt["trajectory_path"],
                    "trajectory_sha256": receipt["trajectory_sha256"],
                }
                for receipt in receipts
            ],
            "checks": {
                "runner_unchanged": sha256_file(Path(__file__)) == runner_hash,
                "all_13_pairs_complete": len(receipts) == 13,
                "image_ids_exact": tuple(int(receipt["image_id"]) for receipt in receipts) == IMAGE_IDS,
                "smoke_first_and_reused": int(receipts[0]["image_id"]) == SMOKE_IMAGE_ID,
                "all_image_checks_passed": all(all(receipt["checks"].values()) for receipt in receipts),
                "all_direction_hooks_once": aggregate["direction_graft_hook_calls"] == 13,
                "one_self_smoke_hook": aggregate["self_graft_hook_calls"] == 1,
                "terminal_26_scientific_arms": sum(
                    arm["finish_reason"] in {"im_end", "length"}
                    for receipt in receipts
                    for arm in receipt["arms"].values()
                )
                == 26,
                "gpu_budget_passed": elapsed <= GPU_BUDGET_SECONDS,
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
                "raw_natural_trajectories_only_scoring_owned_by_lead": True,
                "13_Human13_training_images": True,
                "one_source_prefill_direction_graft": True,
                "no_owner_benefit_or_generalization_claim": True,
            },
            "stop": "bounded_natural_continuation_candidate_complete_no_successor",
        }
        require(all(terminal["checks"].values()), "terminal checks failed")
        atomic_json(output / "receipt.json", terminal)
        (output / "receipt.inprogress.json").unlink()
        print(
            json.dumps(
                {
                    "status": terminal["status"],
                    "output_root": str(output),
                    "elapsed_seconds": elapsed,
                    "pairs": 13,
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
            "elapsed_seconds": time.perf_counter() - started,
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
        atomic_json(output / "receipt.failed.json", failure)
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


if __name__ == "__main__":
    raise SystemExit(main())

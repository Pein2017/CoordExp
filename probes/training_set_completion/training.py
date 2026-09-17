"""Single-GPU masked coherent-route CE for the training-set-completion unit.

The route manifest is the label authority.  This module only checks and consumes
literal prompt/continuation IDs, explicit CE masks, and explicit trusted box
positions; it never parses text to infer a target or repairs a supplied box.
"""
from __future__ import annotations

from probes.training_set_completion.artifacts import canonical, digest, file_hash

import argparse
import copy
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.adapters.dora import (
    inspect_dora_adapter_payload,
    save_dora_adapter_payload,
    select_dora_parameters,
)
from src.losses import (
    aligned_token_logprobs,
    raw_axis_validity_hinge as _shared_raw_axis_validity_hinge,
)
from src.qwen.native import prepare_native_inputs, prepare_replay


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/runtime-preparation-v1")
SCHEMA = "training_set_completion.coherent_route_training.v1"
DEFAULT_OPTIMIZER = {"lr": 1e-5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "foreach": False}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_file(), f"bound path is not a file: {resolved}")
    return {"path": str(resolved), "sha256": file_hash(resolved), "size_bytes": resolved.stat().st_size}


def publish(path: str | Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    with path.open("xb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    require(path.read_bytes() == data, f"publication readback differs: {path}")


def _ids(value: Any, name: str, *, allow_empty: bool = False) -> list[int]:
    require(isinstance(value, list) and (bool(value) or allow_empty), f"{name} must be a token-ID list")
    require(all(type(token) is int and token >= 0 for token in value), f"{name} has invalid token ID")
    return list(value)


def _positions(value: Any, *, length: int, name: str, nonempty: bool = True) -> list[int]:
    require(isinstance(value, list) and (bool(value) or not nonempty), f"{name} must be a position list")
    require(all(type(pos) is int and 0 <= pos < length for pos in value), f"{name} out of range")
    require(len(set(value)) == len(value), f"{name} has duplicate positions")
    return list(value)


def _verify_reference(value: Mapping[str, Any], name: str) -> None:
    require(set(value) == {"path", "sha256", "size_bytes"}, f"{name} binding fields")
    require(binding(value["path"]) == dict(value), f"{name} bytes changed")


def validate_route(record: Mapping[str, Any], *, eos_token_id: int, coordinate_token_ids: Sequence[int] | None = None) -> dict[str, Any]:
    required = {"route_id", "image_id", "example_id", "case", "image_identity", "prompt_token_ids", "continuation_token_ids", "ce_weights", "trusted_boxes", "provenance"}
    require(required <= set(record), "route fields")
    require(isinstance(record["route_id"], str) and record["route_id"], "route ID")
    require(type(record["image_id"]) is int and record["image_id"] >= 0, "route image ID")
    prompt = _ids(record["prompt_token_ids"], "prompt")
    continuation = _ids(record["continuation_token_ids"], "continuation")
    require(isinstance(record["case"], Mapping) and isinstance(record["provenance"], Mapping), "route case/provenance")
    image = record["image_identity"]
    plan = record["case"].get("image_plan", {}) if isinstance(record["case"], Mapping) else {}
    require(isinstance(image, Mapping) and set(image) == {"image_path", "image_content_sha256", "executed_media_sha256", "observed_image_grid_thw"}, "route image identity")
    require(image["image_path"] == record["case"].get("image_path") and image["image_content_sha256"] == plan.get("image_content_sha256") and image["executed_media_sha256"] == plan.get("executed_media_sha256") and image["observed_image_grid_thw"] == plan.get("observed_image_grid_thw"), "route image identity differs from acquisition")
    weights = record["ce_weights"]
    require(isinstance(weights, list) and len(weights) == len(continuation)
            and all(type(weight) is int and weight in (0, 1) for weight in weights), "CE weights must be 0/1")
    require(any(weights), "route has no active CE target")
    endpoint = record.get("trusted_complete_support_endpoint", False)
    require(type(endpoint) is bool, "trusted complete endpoint must be bool")
    for position, token in enumerate(continuation):
        if token == eos_token_id and weights[position]:
            require(endpoint and position == len(continuation) - 1, "premature/untrusted EOS CE positive")
    boxes = record["trusted_boxes"]
    require(isinstance(boxes, list), "trusted boxes list")
    for index, box in enumerate(boxes):
        require(set(box) == {"x1_position", "y1_position", "x2_position", "y2_position", "expected_bins"}, "trusted box fields")
        positions = _positions([box["x1_position"], box["y1_position"], box["x2_position"], box["y2_position"]], length=len(continuation), name=f"box {index} positions")
        require(all(weights[position] == 1 for position in positions), "trusted coordinates must be active CE targets")
        bins = box["expected_bins"]
        require(isinstance(bins, list) and len(bins) == 4 and all(type(value) is int and value >= 0 for value in bins), "trusted box expected bins")
        require(bins[0] < bins[2] and bins[1] < bins[3], "trusted box has invalid raw axes")
        if coordinate_token_ids is not None:
            require(all(value < len(coordinate_token_ids) for value in bins), "trusted box bin outside bound tokenizer")
            require([continuation[position] for position in positions] == [coordinate_token_ids[value] for value in bins], "trusted box positions do not match literal coordinate tokens")
    return dict(record)


def validate_manifest(value: Mapping[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    required = {"schema", "status", "sources", "acquisition_manifest", "source_adapter", "model_config", "routes", "optimizer", "runtime", "validity_hinge", "content_sha256"}
    require(required <= set(value), "training manifest fields")
    content = {key: val for key, val in value.items() if key != "content_sha256"}
    require(value["schema"] == SCHEMA and value["content_sha256"] == digest(content), "manifest schema/content")
    require(value["status"] == "candidate_ready", "manifest is not candidate-ready")
    sources = value["sources"]
    require(isinstance(sources, Mapping) and set(sources) == {"reviewed_routes", "producer"}, "route/code source bindings")
    if verify_sources:
        producer = sources["producer"]
        require(
            isinstance(producer, Mapping)
            and set(producer) == {"path", "sha256", "size_bytes"}
            and isinstance(producer["path"], str)
            and isinstance(producer["sha256"], str)
            and len(producer["sha256"]) == 64
            and type(producer["size_bytes"]) is int
            and producer["size_bytes"] >= 0,
            "training producer provenance binding",
        )
        _verify_reference(value["acquisition_manifest"], "acquisition manifest")
        _verify_reference(sources["reviewed_routes"], "reviewed routes")
    adapter = value["source_adapter"]
    require(isinstance(adapter, Mapping) and isinstance(adapter.get("root"), str) and isinstance(adapter.get("fingerprint"), str), "source adapter identity")
    if verify_sources:
        observed = inspect_dora_adapter_payload(adapter["root"], value["model_config"]["model"]["base_model"])
        require(observed["fingerprint"] == adapter["fingerprint"], "source adapter changed")
    config = value["model_config"]
    require(isinstance(config, Mapping) and config.get("backend", {}).get("type") == "hf" and config.get("model", {}).get("dtype") == "fp32", "FP32 HF model config")
    eos = int(value["runtime"].get("eos_token_id", 151645))
    hinge = value["validity_hinge"]
    require(set(hinge) == {"weight", "margin", "coordinate_token_ids", "coordinate_bin_values", "coordinate_token_spellings", "coordinate_units"}, "validity hinge fields")
    require(type(hinge["weight"]) in (int, float) and hinge["weight"] >= 0 and type(hinge["margin"]) in (int, float) and hinge["margin"] > 0, "validity hinge weight/margin")
    tokens, bins, spellings = _ids(hinge["coordinate_token_ids"], "coordinate token IDs"), hinge["coordinate_bin_values"], hinge["coordinate_token_spellings"]
    require(len(tokens) == len(set(tokens)) == 1000 and bins == list(range(1000)) and spellings == [f"<|coord_{index}|>" for index in range(1000)], "bound coordinate vocabulary")
    require(hinge["coordinate_units"] == "normalized_0_1_from_raw_bins_0_999" and hinge["margin"] == 1 / 999, "normalized coordinate hinge units")
    if verify_sources:
        observed = coordinate_token_table(value["model_config"]["model"]["base_model"])
        require(tokens == observed["ids"] and spellings == observed["spellings"], "bound tokenizer coordinate vocabulary changed")
    routes = value["routes"]
    require(isinstance(routes, list) and routes, "routes")
    checked = [validate_route(route, eos_token_id=eos, coordinate_token_ids=tokens) for route in routes]
    require(len({route["route_id"] for route in checked}) == len(checked), "duplicate route")
    require(len({route["image_id"] for route in checked}) == len(checked), "one coherent route per image")
    optimizer = value["optimizer"]
    require(set(optimizer) == set(DEFAULT_OPTIMIZER), "explicit AdamW settings")
    require(type(optimizer["lr"]) in (int, float) and optimizer["lr"] > 0 and math.isfinite(optimizer["lr"]), "AdamW lr")
    require(isinstance(optimizer["betas"], list) and len(optimizer["betas"]) == 2 and all(type(item) in (int, float) and 0 <= item < 1 for item in optimizer["betas"]), "AdamW betas")
    require(type(optimizer["eps"]) in (int, float) and optimizer["eps"] > 0 and optimizer["foreach"] is False, "AdamW eps/foreach")
    runtime = value["runtime"]
    require(set(runtime) >= {"updates", "checkpoint_steps", "wall_seconds", "max_model_forwards", "eos_token_id"}, "runtime budget")
    require(type(runtime["updates"]) is int and runtime["updates"] > 0 and type(runtime["wall_seconds"]) is int and 0 < runtime["wall_seconds"] < 5 * 3600, "finite update/wall budget")
    require(isinstance(runtime["checkpoint_steps"], list) and all(type(step) is int and 0 < step <= runtime["updates"] for step in runtime["checkpoint_steps"]), "checkpoint schedule")
    require(type(runtime["max_model_forwards"]) is int and runtime["max_model_forwards"] >= runtime["updates"] * len(routes), "forward budget")
    return dict(value)


def validate_training_execution_producer(
    manifest: Mapping[str, Any], *, producer_path: str | Path = Path(__file__)
) -> dict[str, Any]:
    """Bind a mutating training launch to the exact producer bytes executing it.

    ``validate_manifest`` deliberately remains able to read historical manifests
    whose recorded producer path has since changed. Mutation is stricter: the
    manifest must have been prepared for this exact executable producer.
    """

    expected = binding(producer_path)
    require(
        manifest.get("sources", {}).get("producer") == expected,
        "training execution producer differs from manifest binding",
    )
    return expected


def coordinate_token_table(base_model: str | Path) -> dict[str, list[Any]]:
    """Load only the bound tokenizer and prove the literal coordinate vocabulary."""
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options

    qwen = load_qwen_components_from_options(QwenLoadOptions(base_model=str(base_model), dtype="fp32", attn_implementation="sdpa", load_model=False))
    ids, spellings = [], []
    for value in range(1000):
        spelling = f"<|coord_{value}|>"
        token_id = qwen.tokenizer.convert_tokens_to_ids(spelling)
        require(type(token_id) is int and token_id >= 0 and qwen.tokenizer.convert_ids_to_tokens(token_id) == spelling and qwen.tokenizer.decode([token_id], skip_special_tokens=False) == spelling, f"bound tokenizer lacks exact coordinate spelling: {spelling}")
        ids.append(token_id); spellings.append(spelling)
    require(len(set(ids)) == 1000, "bound tokenizer coordinate IDs are not unique")
    return {"ids": ids, "spellings": spellings}


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: Sequence[int]) -> tuple[torch.Tensor, dict[str, float]]:
    require(logits.ndim == 2 and logits.shape[0] == targets.numel() == len(weights), "aligned CE shape")
    mask = torch.tensor(weights, dtype=logits.dtype, device=logits.device)
    active = int(mask.sum().item())
    require(active > 0, "CE has no active positions")
    nll = -aligned_token_logprobs(logits.float(), targets)
    return (nll * mask).sum() / active, {"active_tokens": active, "masked_nll_sum": float((nll * mask).detach().sum())}


def raw_axis_validity_hinge(logits: torch.Tensor, boxes: Sequence[Mapping[str, Any]], *, coordinate_token_ids: Sequence[int], coordinate_bin_values: Sequence[int], margin: float) -> torch.Tensor:
    return _shared_raw_axis_validity_hinge(
        logits,
        boxes,
        coordinate_token_ids=coordinate_token_ids,
        coordinate_bin_values=coordinate_bin_values,
        margin=margin,
    )


def route_objective(model: torch.nn.Module, native_inputs: Mapping[str, Any], route: Mapping[str, Any], hinge: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    replay = prepare_replay(model, native_inputs, prompt_token_ids=route["prompt_token_ids"], continuation_token_ids=route["continuation_token_ids"])
    logits = replay.aligned_logits(model(**replay.inputs).logits)
    require(replay.target_ids.tolist() == route["continuation_token_ids"], "literal continuation changed during replay")
    ce, metrics = masked_ce_loss(logits, replay.target_ids, route["ce_weights"])
    raw_hinge = raw_axis_validity_hinge(logits, route["trusted_boxes"], coordinate_token_ids=hinge["coordinate_token_ids"], coordinate_bin_values=hinge["coordinate_bin_values"], margin=hinge["margin"])
    total = ce + float(hinge["weight"]) * raw_hinge
    require(bool(torch.isfinite(total)), "nonfinite route objective")
    return total, {**metrics, "ce": float(ce.detach()), "raw_axis_validity_hinge": float(raw_hinge.detach()), "total": float(total.detach())}


def bind_language_dora(model: torch.nn.Module, *, source_adapter: Mapping[str, Any]) -> tuple[tuple[str, torch.nn.Parameter], tuple[tuple[str, torch.nn.Parameter], ...]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named = select_dora_parameters(model, towers=("language",), adapter_name="default")
    declared = source_adapter.get("semantic_identity", {})
    require(isinstance(declared, Mapping) and type(declared.get("tensor_key_count")) is int, "source adapter tensor-count identity")
    # The adapter payload is the published source of the N16 588-tensor count;
    # record the live scalar count instead of maintaining a guessed duplicate.
    require(len(named) == declared["tensor_key_count"], "loaded DoRA tensor count differs from source adapter")
    require(sum(parameter.numel() for _, parameter in named) > 0, "empty loaded DoRA scalar surface")
    require(all("language_model" in name and not any(item in name for item in ("visual", "merger", "embed_tokens", "lm_head")) for name, _ in named), "language-only DoRA surface")
    for _, parameter in named:
        parameter.requires_grad_(True)
    selected = {id(parameter) for _, parameter in named}
    frozen = tuple((name, parameter) for name, parameter in model.named_parameters() if id(parameter) not in selected)
    return named, frozen


def _layout(named: Sequence[tuple[str, torch.nn.Parameter]]) -> list[dict[str, Any]]:
    return [{"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype), "numel": parameter.numel()} for name, parameter in named]


def _checkpoint(output: Path, *, manifest_path: Path, manifest: Mapping[str, Any], model: Any, optimizer: torch.optim.Optimizer, named: Sequence[tuple[str, torch.nn.Parameter]], step: int) -> dict[str, Any]:
    root = output / "checkpoints" / f"step-{step:05d}"
    require(not root.exists(), "checkpoint collision")
    root.mkdir(parents=True)
    adapter_dir = root / "adapter"
    identity = save_dora_adapter_payload(
        model,
        source_root=Path(manifest["source_adapter"]["root"]),
        output=adapter_dir,
        expected_base_model_path=manifest["model_config"]["model"]["base_model"],
        expected_tensor_count=manifest["source_adapter"]["semantic_identity"]["tensor_key_count"],
    )
    state = {"schema": f"{SCHEMA}.checkpoint.v1", "manifest": binding(manifest_path), "source_adapter": manifest["source_adapter"], "saved_adapter": identity, "step": step, "parameter_layout": _layout(named), "optimizer": manifest["optimizer"], "optimizer_state_dict": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(), "python_random_state": random.getstate()}
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, root / "state.pt")
    return {"root": str(root), "adapter": identity, "state": binding(root / "state.pt"), "step": step}


def _restore(resume: Path, *, manifest_path: Path, manifest: Mapping[str, Any], optimizer: torch.optim.Optimizer, named: Sequence[tuple[str, torch.nn.Parameter]]) -> int:
    state_path = resume / "state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    require(state["schema"] == f"{SCHEMA}.checkpoint.v1" and state["manifest"] == binding(manifest_path), "resume manifest identity")
    require(state["source_adapter"] == manifest["source_adapter"] and state["optimizer"] == manifest["optimizer"] and state["parameter_layout"] == _layout(named), "resume model/surface/optimizer identity")
    observed = inspect_dora_adapter_payload(resume / "adapter", manifest["model_config"]["model"]["base_model"])
    require(observed == state["saved_adapter"], "resume adapter changed")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    torch.set_rng_state(state["torch_rng_state"])
    random.setstate(state["python_random_state"])
    if "cuda_rng_state_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return int(state["step"])


def _native_entries(qwen: Any, manifest: Mapping[str, Any], device: torch.device) -> list[dict[str, Any]]:
    from src.inference.bound_requests import build_bound_native_requests as build_requests
    entries = []
    for route in manifest["routes"]:
        requests, _ = build_requests(qwen, manifest["model_config"], [route["case"]])
        batch = prepare_native_inputs(qwen.processor, requests, device=device, record_media_identity=True)
        prompt = list(batch.prompt_token_ids[0])
        require(prompt == route["prompt_token_ids"], "live native prompt differs from bound route")
        require(batch.media_sha256[0] == route["image_identity"]["executed_media_sha256"] and list(batch.image_grids[0]) == route["image_identity"]["observed_image_grid_thw"], "live native image identity differs from bound route")
        entries.append({"route": route, "inputs": dict(batch.inputs)})
    return entries


def run(manifest_path: Path, *, output: Path, device: str, resume: Path | None = None) -> dict[str, Any]:
    """Run one explicitly bounded single-GPU attempt.  Caller owns tmux/launch."""
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig
    from src.qwen.checkpointing import (
        install_language_decoder_checkpointing,
        language_decoder_checkpointing_receipt as checkpointing_receipt,
    )

    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    execution_producer = validate_training_execution_producer(manifest)
    require(device.startswith("cuda") and torch.cuda.is_available(), "single-GPU CUDA device required")
    require(not output.exists(), "attempt output already exists")
    output.mkdir(parents=True)
    started = time.monotonic()
    forwards = 0
    checkpoints = []
    phase = "model_setup"
    start_step = 0
    old_alarm = signal.getsignal(signal.SIGALRM)

    def expired(*_: Any) -> None:
        raise TimeoutError("training wall budget")

    try:
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(math.ceil(manifest["runtime"]["wall_seconds"]))
        config = checkpoint_config(InferConfig.model_validate(manifest["model_config"]), str((resume / "adapter") if resume else manifest["source_adapter"]["root"]))
        qwen, loaded = load_policy(config, device=torch.device(device))
        model = qwen.model
        model.eval()
        named, frozen = bind_language_dora(model, source_adapter=manifest["source_adapter"])
        checkpointing = install_language_decoder_checkpointing(model, expected_layer_count=28)
        checkpointing.update(enabled=True, phase="train")
        optimizer = torch.optim.AdamW([parameter for _, parameter in named], **{**manifest["optimizer"], "betas": tuple(manifest["optimizer"]["betas"])})
        require(resume is not None or not optimizer.state, "fresh attempt requires a fresh AdamW optimizer")
        start_step = _restore(resume, manifest_path=manifest_path, manifest=manifest, optimizer=optimizer, named=named) if resume else 0
        require(start_step < manifest["runtime"]["updates"], "resume already reaches terminal step")
        phase = "native_input_setup"
        entries = _native_entries(qwen, manifest, torch.device(device))
        for step in range(start_step + 1, manifest["runtime"]["updates"] + 1):
            phase = f"train_update_{step}"
            optimizer.zero_grad(set_to_none=True)
            losses, cards = [], []
            # One complete, fixed route per image is accumulated before exactly one update.
            for entry in entries:
                loss, card = route_objective(model, entry["inputs"], entry["route"], manifest["validity_hinge"])
                (loss / len(entries)).backward()
                losses.append(loss)
                cards.append({"route_id": entry["route"]["route_id"], **card})
                forwards += 1
            require(forwards <= manifest["runtime"]["max_model_forwards"], "model-forward budget")
            require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for _, parameter in named), "missing/nonfinite DoRA gradients")
            require(all(parameter.grad is None for _, parameter in frozen), "frozen parameter received gradient")
            raw_norm = float(torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named], 1.0, error_if_nonfinite=True, foreach=False))
            optimizer.step()
            update = {"schema": f"{SCHEMA}.update.v1", "step": step, "image_count": len(entries), "objective_mean_over_images": float(torch.stack(losses).mean().detach()), "routes": cards, "gradient_norm_before_clip": raw_norm, "forwards": forwards}
            publish(output / "updates" / f"step-{step:05d}.json", update)
            if step in manifest["runtime"]["checkpoint_steps"] or step == manifest["runtime"]["updates"]:
                checkpoints.append(_checkpoint(output, manifest_path=manifest_path, manifest=manifest, model=model, optimizer=optimizer, named=named, step=step))
        terminal = {"schema": f"{SCHEMA}.terminal.v1", "status": "completed", "manifest": binding(manifest_path), "execution_producer": execution_producer, "loaded_model": loaded, "optimizer_mode": "resume" if resume else "fresh", "trainable_surface": _layout(named), "updates": manifest["runtime"]["updates"], "model_forwards": forwards, "checkpoints": checkpoints, "activation_checkpointing": checkpointing_receipt(model, checkpointing), "elapsed_seconds": time.monotonic() - started}
    except Exception as exc:
        terminal = {"schema": f"{SCHEMA}.terminal.v1", "status": "failed", "manifest": binding(manifest_path), "execution_producer": execution_producer, "phase": phase, "step": start_step, "model_forwards": forwards, "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - started}
        publish(output / "terminal.json", terminal)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
    publish(output / "terminal.json", terminal)
    return terminal


def prepare(*, acquisition_manifest: Path, routes_path: Path, output: Path, updates: int, checkpoint_steps: Sequence[int], wall_seconds: int) -> dict[str, Any]:
    """CPU-only: bind acquisition + explicit reviewed routes into a launch manifest."""
    from probes.training_set_completion.acquisition import validate_manifest as validate_acquisition

    acquisition = validate_acquisition(acquisition_manifest)
    routes = json.loads(routes_path.read_text())
    require(isinstance(routes, list), "route source must be a JSON list")
    by_image = {int(record["image_id"]): record for record in acquisition["records"]}
    bound_routes = []
    for item in routes:
        require(type(item.get("image_id")) is int and item["image_id"] in by_image, "route image is outside acquisition cohort")
        source = by_image[item["image_id"]]
        require(item.get("example_id") == source["example_id"], "route example differs from acquisition")
        plan = source["case"]["image_plan"]
        bound_routes.append({**item, "case": source["case"], "prompt_token_ids": source["prompt_token_ids"], "image_identity": {"image_path": source["case"]["image_path"], "image_content_sha256": plan["image_content_sha256"], "executed_media_sha256": plan["executed_media_sha256"], "observed_image_grid_thw": plan["observed_image_grid_thw"]}})
    coordinate_table = coordinate_token_table(acquisition["model"]["config"]["model"]["base_model"])
    value = {"schema": SCHEMA, "status": "candidate_ready", "sources": {"reviewed_routes": binding(routes_path), "producer": binding(Path(__file__))}, "acquisition_manifest": binding(acquisition_manifest), "source_adapter": acquisition["model"]["adapter"], "model_config": acquisition["model"]["config"], "routes": bound_routes, "optimizer": copy.deepcopy(DEFAULT_OPTIMIZER), "runtime": {"updates": updates, "checkpoint_steps": sorted(set(checkpoint_steps)), "wall_seconds": wall_seconds, "max_model_forwards": updates * len(bound_routes), "eos_token_id": 151645}, "validity_hinge": {"weight": 0.01, "margin": 1 / 999, "coordinate_token_ids": coordinate_table["ids"], "coordinate_bin_values": list(range(1000)), "coordinate_token_spellings": coordinate_table["spellings"], "coordinate_units": "normalized_0_1_from_raw_bins_0_999"}}
    value["content_sha256"] = digest(value)
    validate_manifest(value)
    require(not output.exists(), "manifest output exists")
    publish(output, value)
    return value


def readback(manifest_path: Path, *, adapter: Path, output: Path, device: str) -> dict[str, Any]:
    """Native original-image, empty-prefix greedy readback for one exported adapter.

    This is deliberately an evidence producer only: it writes literal generated
    IDs and terminal reasons, leaving physical-owner scoring to its owner.
    """
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.inference.bound_requests import build_bound_native_requests as build_requests
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations

    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    require(device.startswith("cuda") and torch.cuda.is_available(), "single-GPU CUDA device required")
    require(not output.exists(), "readback output already exists")
    adapter_identity = inspect_dora_adapter_payload(adapter, manifest["model_config"]["model"]["base_model"])
    config = checkpoint_config(InferConfig.model_validate(manifest["model_config"]), str(adapter))
    qwen, loaded = load_policy(config, device=torch.device(device))
    qwen.model.eval()
    cap = int(config.generation.max_new_tokens)
    policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, use_model_defaults=False)
    rows = []
    for route in manifest["routes"]:
        requests, _ = build_requests(qwen, manifest["model_config"], [route["case"]])
        batch = prepare_native_inputs(qwen.processor, requests, device=device, record_media_identity=True)
        require(list(batch.prompt_token_ids[0]) == route["prompt_token_ids"], "readback original prompt changed")
        require(batch.media_sha256[0] == route["image_identity"]["executed_media_sha256"] and list(batch.image_grids[0]) == route["image_identity"]["observed_image_grid_thw"], "readback original image identity changed")
        with torch.inference_mode():
            generated, = generate_continuations(qwen.model, batch, extensions=[[]], budgets=[cap], eos_token_id=manifest["runtime"]["eos_token_id"], pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace="none", seed=None)
        ids = list(generated.token_ids)
        rows.append({"route_id": route["route_id"], "image_id": route["image_id"], "empty_assistant_prefix": True, "prompt_token_ids": route["prompt_token_ids"], "generated_token_ids": ids, "generated_token_ids_sha256": digest(ids), "decode_stop_reason": generated.stop_reason, "raw_decode_text": qwen.tokenizer.decode(ids, skip_special_tokens=False), "executed_media_sha256": batch.media_sha256[0], "observed_image_grid_thw": list(batch.image_grids[0])})
    receipt = {"schema": f"{SCHEMA}.native_readback.v1", "status": "completed_unscored", "manifest": binding(manifest_path), "execution_producer": binding(Path(__file__)), "adapter": adapter_identity, "loaded_model": loaded, "policy": {"empty_assistant_prefix": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "assistant_token_cap": cap}, "rows": rows}
    publish(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="CPU-only manifest preparation")
    prepare_parser.add_argument("--acquisition-manifest", type=Path, required=True)
    prepare_parser.add_argument("--routes", type=Path, required=True, help="reviewed literal route JSON list")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--updates", type=int, required=True)
    prepare_parser.add_argument("--checkpoint-step", type=int, action="append", required=True)
    prepare_parser.add_argument("--wall-seconds", type=int, required=True)
    verify_parser = sub.add_parser("verify", help="CPU-only manifest validation")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    run_parser = sub.add_parser("run", help="single-GPU bounded training")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--resume", type=Path)
    readback_parser = sub.add_parser("readback", help="native original-image empty-prefix greedy readback")
    readback_parser.add_argument("--manifest", type=Path, required=True)
    readback_parser.add_argument("--adapter", type=Path, required=True)
    readback_parser.add_argument("--output", type=Path, required=True)
    readback_parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(acquisition_manifest=args.acquisition_manifest, routes_path=args.routes, output=args.output, updates=args.updates, checkpoint_steps=args.checkpoint_step, wall_seconds=args.wall_seconds)
    elif args.command == "verify":
        validate_manifest(json.loads(args.manifest.read_text()))
    elif args.command == "run":
        run(args.manifest, output=args.output, device=args.device, resume=args.resume)
    else:
        readback(args.manifest, adapter=args.adapter, output=args.output, device=args.device)


if __name__ == "__main__":
    main()

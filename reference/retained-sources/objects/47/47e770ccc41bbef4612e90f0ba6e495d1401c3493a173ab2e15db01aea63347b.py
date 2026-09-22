"""CPU-only immutable acceptance replay for fourth-fit training and readbacks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open


WORKTREE = Path("/data/CoordExp/.worktrees/research-probes").resolve()
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from probes.source_rweak_row_cross.run import native_record
from probes.training_set_completion.acquisition import binding, digest, read, require
import probes.training_set_completion.recover_readback as recovery
from probes.training_set_completion.training import inspect_dora_adapter_payload, validate_manifest
from src.qwen.special_token_embeddings import inspect_special_token_embedding_delta_payload


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum").resolve()
ROOT = BASE / "root/fourth-fit-execution-acceptance-v1"
RUN_ROOT = BASE / "fourth-fit-v1"
TRAINING_ROOT = RUN_ROOT / "training"
READBACK_ROOT = RUN_ROOT / "readback-recovery"
MANIFEST = BASE / "fourth-fit-preparation-v1/manifest.json"
PREDECESSOR_MANIFEST = BASE / "third-fit-preparation-v1/manifest.json"
PREDECESSOR_CHECKPOINT = BASE / "third-fit-v1/training/checkpoints/step-00064"
ACCEPTANCE = ROOT / "receipt.json"
VERIFY = Path(__file__).resolve()
RUN_WRAPPER = RUN_ROOT / "run.py"
CONTINUATION_SOURCE = WORKTREE / "probes/training_set_completion/continue_training.py"
RECOVERY_SOURCE = WORKTREE / "probes/training_set_completion/recover_readback.py"
TRAINING_SOURCE = WORKTREE / "probes/training_set_completion/training.py"

STEPS = (128, 192, 256)
UPDATE_STEPS = tuple(range(65, 257))
CURVE_STEPS = (65, 128, 192, 256)
IMAGES = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
EXPECTED_TENSORS = 588
EXPECTED_SEGMENT_UPDATES = 192
EXPECTED_SEGMENT_FORWARDS = 2112
CAP = 3084
EOS = 151645
ALLOWED_RUNTIME_FIELDS = {"checkpoint_steps", "max_model_forwards", "updates", "wall_seconds"}


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"accepted receipt collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == payload, "receipt write/read mismatch")


def _finite(value: Any, name: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"non-finite {name}")
    return result


def _rng_identity(state: Mapping[str, Any]) -> dict[str, Any]:
    torch_state = state["torch_rng_state"]
    cuda_states = state["cuda_rng_state_all"]
    require(isinstance(torch_state, torch.Tensor) and torch_state.ndim == 1 and torch_state.numel() > 0, "torch RNG state")
    require(isinstance(cuda_states, list) and cuda_states, "CUDA RNG states")
    require(all(isinstance(item, torch.Tensor) and item.ndim == 1 and item.numel() > 0 for item in cuda_states), "CUDA RNG state payloads")
    return {
        "torch_rng_state_sha256": hashlib.sha256(bytes(torch_state.tolist())).hexdigest(),
        "cuda_rng_state_sha256": [hashlib.sha256(bytes(item.tolist())).hexdigest() for item in cuda_states],
        "python_random_state_sha256": hashlib.sha256(pickle.dumps(state["python_random_state"], protocol=4)).hexdigest(),
    }


def _compact_adapter(identity: Mapping[str, Any]) -> dict[str, Any]:
    semantic = identity["semantic_identity"]
    return {
        "root": identity["root"],
        "fingerprint": identity["fingerprint"],
        "files": identity["files"],
        "semantic_identity": {
            key: semantic[key]
            for key in ("base_model_name_or_path", "lora_A_count", "lora_B_count", "lora_magnitude_vector_count", "tensor_key_count", "use_dora", "r")
        },
    }


def _compact_embedding(identity: Mapping[str, Any]) -> dict[str, Any]:
    semantic = identity["semantic_identity"]
    return {
        "root": identity["root"],
        "fingerprint": identity["fingerprint"],
        "files": identity["files"],
        "semantic_identity": {
            "base_model_path": semantic["base_model_path"],
            "base_config_sha256": semantic["base_config_sha256"],
            "tokenizer_sha256": semantic["tokenizer_sha256"],
            "tensor_shape": semantic["tensor_shape"],
            "tensor_dtype": semantic["tensor_dtype"],
            "token_count": len(semantic["token_ids"]),
            "semantics": semantic["semantics"],
        },
    }


def _manifest_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    old_raw = read(PREDECESSOR_MANIFEST)
    new_raw = read(MANIFEST)
    old = validate_manifest(old_raw)
    new = validate_manifest(new_raw)
    require(old == old_raw and new == new_raw, "manifest canonical validation")
    require(set(new) == set(old) | {"continuation"}, "fourth-fit top-level extension")

    old_core = copy.deepcopy(old)
    new_core = copy.deepcopy(new)
    old_core.pop("content_sha256")
    new_core.pop("content_sha256")
    continuation = new_core.pop("continuation")
    require(set(old_core) == set(new_core), "frozen manifest key set")
    changed = {key for key in old_core if old_core[key] != new_core[key]}
    require(changed == {"runtime"}, "only runtime may change outside continuation provenance")
    old_runtime = old_core["runtime"]
    new_runtime = new_core["runtime"]
    require(set(old_runtime) == set(new_runtime), "runtime key set")
    runtime_changes = {key for key in old_runtime if old_runtime[key] != new_runtime[key]}
    require(runtime_changes == ALLOWED_RUNTIME_FIELDS, "exact permitted runtime extensions")
    require(new_runtime == {"checkpoint_steps": [128, 192, 256], "eos_token_id": EOS, "max_model_forwards": 2816, "updates": 256, "wall_seconds": 4500}, "fourth-fit runtime contract")
    require(continuation == {
        "schema": "training_set_completion.continuation.v1",
        "predecessor_manifest": binding(PREDECESSOR_MANIFEST),
        "predecessor_checkpoint": str(PREDECESSOR_CHECKPOINT),
        "predecessor_state": binding(PREDECESSOR_CHECKPOINT / "state.pt"),
        "wrapper": binding(CONTINUATION_SOURCE),
        "allowed_runtime_fields": sorted(ALLOWED_RUNTIME_FIELDS),
        "remaining_logical_image_forwards": EXPECTED_SEGMENT_FORWARDS,
        "cumulative_runtime_forward_bound": 2816,
    }, "continuation provenance")
    require(new["sources"] == old["sources"], "frozen source identities")
    require(new["sources"]["producer"] == binding(TRAINING_SOURCE), "training source binding")
    require(new["content_sha256"] == digest({key: value for key, value in new.items() if key != "content_sha256"}), "manifest content digest")
    return old, new


def _adapter_finite(adapter_root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = inspect_dora_adapter_payload(adapter_root, expected["semantic_identity"]["base_model_name_or_path"])
    require(observed == expected, f"saved adapter identity: {adapter_root}")
    tensor_path = adapter_root / "adapter_model.safetensors"
    finite = 0
    with safe_open(str(tensor_path), framework="pt", device="cpu") as payload:
        keys = list(payload.keys())
        require(len(keys) == EXPECTED_TENSORS, f"adapter tensor count: {adapter_root}")
        for key in keys:
            require(bool(torch.isfinite(payload.get_tensor(key)).all()), f"non-finite adapter tensor: {key}")
            finite += 1
    return {"identity": observed, "finite_tensor_count": finite}


def _checkpoint_summary(
    root: Path,
    step: int,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    expected_layout: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state_path = root / "state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    require(state["schema"].endswith(".checkpoint.v1") and int(state["step"]) == step, f"checkpoint {step} schema/step")
    require(state["manifest"] == binding(manifest_path), f"checkpoint {step} manifest binding")
    require(state["source_adapter"] == manifest["source_adapter"], f"checkpoint {step} source adapter")
    require(state["optimizer"] == manifest["optimizer"], f"checkpoint {step} optimizer identity")
    layout = state["parameter_layout"]
    require(len(layout) == EXPECTED_TENSORS, f"checkpoint {step} parameter layout count")
    if expected_layout is not None:
        require(layout == expected_layout, f"checkpoint {step} parameter layout continuity")

    saved = _adapter_finite(root / "adapter", state["saved_adapter"])
    optimizer = state["optimizer_state_dict"]
    require(len(optimizer["param_groups"]) == 1, f"checkpoint {step} optimizer group count")
    group = optimizer["param_groups"][0]
    require(len(group["params"]) == EXPECTED_TENSORS and {int(item) for item in group["params"]} == set(range(EXPECTED_TENSORS)), f"checkpoint {step} optimizer parameter IDs")
    require(group["lr"] == manifest["optimizer"]["lr"] and list(group["betas"]) == manifest["optimizer"]["betas"] and group["eps"] == manifest["optimizer"]["eps"] and group["weight_decay"] == manifest["optimizer"]["weight_decay"] and group["foreach"] == manifest["optimizer"]["foreach"], f"checkpoint {step} AdamW hyperparameters")
    parameter_state = optimizer["state"]
    require(len(parameter_state) == EXPECTED_TENSORS, f"checkpoint {step} optimizer state count")
    step_values: set[int] = set()
    finite_optimizer_tensors = 0
    for parameter_id, value in parameter_state.items():
        require(int(parameter_id) in range(EXPECTED_TENSORS), f"checkpoint {step} optimizer parameter outside layout")
        require({"step", "exp_avg", "exp_avg_sq"} <= set(value), f"checkpoint {step} AdamW state fields")
        counter = value["step"]
        step_values.add(int(counter.item() if isinstance(counter, torch.Tensor) else counter))
        for name in ("exp_avg", "exp_avg_sq"):
            require(bool(torch.isfinite(value[name]).all()), f"checkpoint {step} non-finite optimizer {name}")
            finite_optimizer_tensors += 1
    require(step_values == {step}, f"checkpoint {step} optimizer counters")
    summary = {
        "step": step,
        "state": binding(state_path),
        "adapter_fingerprint": saved["identity"]["fingerprint"],
        "adapter_finite_tensor_count": saved["finite_tensor_count"],
        "optimizer_state_count": len(parameter_state),
        "optimizer_finite_tensor_count": finite_optimizer_tensors,
        "optimizer_step_values": sorted(step_values),
        "rng": _rng_identity(state),
    }
    del state, optimizer, parameter_state
    return summary, layout


def _update_summary(path: Path, step: int) -> dict[str, Any]:
    update = read(path)
    require(update["schema"].endswith(".update.v1") and int(update["step"]) == step, f"update {step} schema/step")
    require(int(update["image_count"]) == len(IMAGES) and int(update["forwards"]) == (step - 64) * len(IMAGES), f"update {step} segment counters")
    require(len(update["routes"]) == len(IMAGES), f"update {step} route count")
    objective = _finite(update["objective_mean_over_images"], f"update {step} objective")
    gradient = _finite(update["gradient_norm_before_clip"], f"update {step} gradient")
    totals = [_finite(row["total"], f"update {step} route total") for row in update["routes"]]
    ce = [_finite(row["ce"], f"update {step} route CE") for row in update["routes"]]
    hinge = [_finite(row["raw_axis_validity_hinge"], f"update {step} route hinge") for row in update["routes"]]
    require(abs(sum(totals) / len(IMAGES) - objective) < 2e-6, f"update {step} objective reduction")
    return {
        "step": step,
        "segment_logical_image_forwards": int(update["forwards"]),
        "cumulative_logical_image_forwards": step * len(IMAGES),
        "objective_mean_over_images": objective,
        "mean_masked_ce": sum(ce) / len(IMAGES),
        "mean_raw_axis_validity_hinge": sum(hinge) / len(IMAGES),
        "gradient_norm_before_clip": gradient,
    }


def _training_audit(old: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected_update_files = {f"step-{step:05d}.json" for step in UPDATE_STEPS}
    actual_update_files = {path.name for path in (TRAINING_ROOT / "updates").iterdir() if path.is_file()}
    require(actual_update_files == expected_update_files, "exact fourth-fit update file set 65..256")
    checkpoint_dirs = {path.name for path in (TRAINING_ROOT / "checkpoints").iterdir() if path.is_dir()}
    require(checkpoint_dirs == {f"step-{step:05d}" for step in STEPS}, "exact fourth-fit checkpoint directory set")

    predecessor, layout = _checkpoint_summary(PREDECESSOR_CHECKPOINT, 64, PREDECESSOR_MANIFEST, old, None)
    require(predecessor["state"] == manifest["continuation"]["predecessor_state"], "predecessor state continuation binding")
    checkpoints: list[dict[str, Any]] = []
    for step in STEPS:
        summary, observed_layout = _checkpoint_summary(TRAINING_ROOT / "checkpoints" / f"step-{step:05d}", step, MANIFEST, manifest, layout)
        require(observed_layout == layout, f"checkpoint {step} DoRA layout")
        checkpoints.append(summary)

    updates = [_update_summary(TRAINING_ROOT / "updates" / f"step-{step:05d}.json", step) for step in UPDATE_STEPS]
    terminal_path = TRAINING_ROOT / "terminal.json"
    terminal = read(terminal_path)
    require(terminal["status"] == "completed" and terminal["optimizer_mode"] == "resume", "training completion/resume mode")
    require(int(terminal["updates"]) == 256 and int(terminal["model_forwards"]) == EXPECTED_SEGMENT_FORWARDS, "training terminal counters")
    require(_finite(terminal["elapsed_seconds"], "training elapsed seconds") == 3474.636510401964, "frozen training elapsed receipt")
    require(terminal["elapsed_seconds"] < manifest["runtime"]["wall_seconds"] == 4500, "training wall limit")
    require(terminal["manifest"] == binding(MANIFEST), "training terminal manifest")
    require(terminal["trainable_surface"] == layout and len(layout) == EXPECTED_TENSORS, "training terminal DoRA surface")
    require([int(item["step"]) for item in terminal["checkpoints"]] == list(STEPS), "training terminal checkpoint schedule")
    for actual, recorded in zip(checkpoints, terminal["checkpoints"], strict=True):
        require(actual["state"] == recorded["state"] and actual["adapter_fingerprint"] == recorded["adapter"]["fingerprint"], "training terminal checkpoint bindings")

    loaded = terminal["loaded_model"]
    model_identity = loaded["model_identity"]
    require(loaded["effective_settings"]["observed_attn_implementation"] == "sdpa", "training attention implementation")
    require(loaded["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"], "training dtype")
    require(model_identity["base"]["path"] == manifest["model_config"]["model"]["base_model"], "training base model")
    require(model_identity["adapter"]["adapter_path"] == str(PREDECESSOR_CHECKPOINT / "adapter"), "training loaded predecessor adapter")
    adapter_evidence = model_identity["adapter"]
    require(adapter_evidence["adapter_type"] == "dora" and adapter_evidence["status"] == "validated" and adapter_evidence["enabled"] is True, "training loaded DoRA adapter")
    require(adapter_evidence["adapter_payload_evidence"]["key_count"] == EXPECTED_TENSORS and adapter_evidence["adapter_state_evidence"] == {"normalized_materialized_key_count": EXPECTED_TENSORS, "normalized_saved_key_count": EXPECTED_TENSORS, "state_checked": True}, "training loaded adapter state")
    embedding = inspect_special_token_embedding_delta_payload(manifest["model_config"]["embedding_delta"]["path"], manifest["model_config"]["model"]["base_model"])
    require(model_identity["embedding_delta"]["identity"]["delta_path"] == embedding["root"], "training embedding path")
    require(model_identity["embedding_delta"]["identity"]["metadata"] == embedding["semantic_identity"], "training embedding identity")

    curve = [next(item for item in updates if item["step"] == step) for step in CURVE_STEPS]
    return {
        "terminal": binding(terminal_path),
        "runtime": {
            "predecessor_step": 64,
            "updates": [65, 256],
            "segment_update_count": EXPECTED_SEGMENT_UPDATES,
            "segment_logical_image_forwards": EXPECTED_SEGMENT_FORWARDS,
            "cumulative_update": 256,
            "cumulative_logical_image_forwards": 2816,
            "optimizer_mode": "resume",
            "checkpoint_steps": list(STEPS),
            "wall_seconds_limit": 4500,
            "elapsed_seconds": terminal["elapsed_seconds"],
        },
        "initialization": {
            "predecessor": predecessor,
            "source_adapter": _compact_adapter(manifest["source_adapter"]),
            "frozen_base_path": manifest["model_config"]["model"]["base_model"],
            "embedding_delta": _compact_embedding(embedding),
            "loaded_predecessor_model_identity": True,
        },
        "checkpoints": checkpoints,
        "optimization_curve": {
            "selected_steps": curve,
            "objective_change_step65_to_step256": curve[-1]["objective_mean_over_images"] - curve[0]["objective_mean_over_images"],
            "objective_change_step192_to_step256": curve[-1]["objective_mean_over_images"] - curve[-2]["objective_mean_over_images"],
            "gradient_norm_range_all_updates": {
                "min": min(item["gradient_norm_before_clip"] for item in updates),
                "max": max(item["gradient_norm_before_clip"] for item in updates),
                "all_finite": True,
            },
        },
    }


def _configure_recovery(manifest_path: Path) -> None:
    recovery.MANIFEST = manifest_path.resolve()
    recovery.TRAIN = TRAINING_ROOT.resolve()
    recovery.ROOT = READBACK_ROOT.resolve()
    recovery.STEPS = STEPS


def _readback_missing() -> list[str]:
    required = [
        READBACK_ROOT / "manifest.json",
        READBACK_ROOT / "exits.json",
        READBACK_ROOT / "result.json",
        RUN_ROOT / "terminal.json",
        *[READBACK_ROOT / "terminals" / f"shard-{shard}.json" for shard in range(8)],
        *[READBACK_ROOT / f"readback-step-{step}.json" for step in STEPS],
    ]
    return [str(path) for path in required if not path.is_file()]


def _stop_kind(ids: list[int], reason: str) -> str:
    require(ids and len(ids) <= CAP, "generated token count/cap")
    if reason == "im_end":
        require(ids[-1] == EOS and EOS not in ids[:-1], "invalid natural EOS termination")
        return "eos"
    require(reason == "length" and len(ids) == CAP and EOS not in ids, "invalid length-cap termination")
    return "cap"


def _compare_envelope_row(envelope: Mapping[str, Any], durable: Mapping[str, Any]) -> None:
    fields = (
        "route_id", "image_id", "empty_assistant_prefix", "prompt_token_ids", "generated_token_ids",
        "generated_token_ids_sha256", "decode_stop_reason", "raw_decode_text",
        "executed_media_sha256", "observed_image_grid_thw",
    )
    require(all(envelope.get(key) == durable.get(key) for key in fields), "envelope/durable row mismatch")


def _model_receipt_summary(path: Path, step: int, manifest: Mapping[str, Any], embedding_identity: Mapping[str, Any]) -> dict[str, Any]:
    value = read(path)
    require(value["schema_version"] == "source256_loaded_policy.v1" and value["scope"] == "loaded-model-identity-only", "cold model receipt schema/scope")
    require(value["effective_settings"]["observed_attn_implementation"] == "sdpa", "cold model attention")
    require(value["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"], "cold model dtype")
    identity = value["model_identity"]
    require(identity["base"]["path"] == manifest["model_config"]["model"]["base_model"], "cold model base identity")
    require(identity["embedding_delta"] == embedding_identity, "cold model embedding identity")
    adapter = identity["adapter"]
    expected_root = TRAINING_ROOT / "checkpoints" / f"step-{step:05d}" / "adapter"
    require(adapter["adapter_path"] == str(expected_root), "cold model checkpoint adapter path")
    require(adapter["adapter_type"] == "dora" and adapter["status"] == "validated" and adapter["enabled"] is True, "cold model DoRA status")
    require(adapter["active_adapters"] == ["default"] and adapter["missing_keys"] == [] and adapter["unexpected_keys"] == [], "cold model active adapter")
    payload = adapter["adapter_payload_evidence"]
    require(payload["key_count"] == EXPECTED_TENSORS and payload["lora_A_count"] == payload["lora_B_count"] == payload["lora_magnitude_vector_count"] == 196, "cold model DoRA payload")
    require(adapter["adapter_state_evidence"] == {"normalized_materialized_key_count": EXPECTED_TENSORS, "normalized_saved_key_count": EXPECTED_TENSORS, "state_checked": True}, "cold model materialized state")
    return {"receipt": binding(path), "step": step, "adapter_path": str(expected_root)}


def _readback_audit(manifest: Mapping[str, Any], training: Mapping[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    readback_manifest_path = READBACK_ROOT / "manifest.json"
    _configure_recovery(readback_manifest_path)
    readback_manifest = read(readback_manifest_path)
    recovery.validate(readback_manifest)
    require(readback_manifest["training_manifest"] == binding(MANIFEST), "readback training manifest binding")
    require(readback_manifest["training_terminal"] == binding(TRAINING_ROOT / "terminal.json"), "readback training terminal binding")
    require(readback_manifest["bound_roots"] == {"manifest": str(MANIFEST), "training": str(TRAINING_ROOT), "readback": str(READBACK_ROOT)}, "readback bound roots")
    require(readback_manifest["producer"] == binding(RUN_WRAPPER), "readback wrapper binding")
    require(readback_manifest["reused_producer"] == binding(RECOVERY_SOURCE), "readback recovery source binding")
    require(readback_manifest["runtime"] == {"cap": CAP, "eos": EOS, "worker_seconds": 3600, "policy": {"empty_assistant_prefix": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}}, "readback decode policy")
    require(readback_manifest["step_mapping"] == {"original_recovery_steps": [16, 32, 64], "continuation_steps": [128, 192, 256], "mapping": {"16": 128, "32": 192, "64": 256}}, "readback step mapping")
    require([int(route["image_id"]) for route in readback_manifest["routes"]] == list(IMAGES), "readback route cohort")

    outer = read(RUN_ROOT / "terminal.json")
    require(outer["status"] == "completed_unscored" and outer["phase"] == "completed", "outer completion")
    require(outer["manifest"] == binding(MANIFEST) and outer["training_terminal"] == binding(TRAINING_ROOT / "terminal.json"), "outer training bindings")
    require(outer["readback_manifest"] == binding(readback_manifest_path) and outer["result"] == binding(READBACK_ROOT / "result.json"), "outer readback bindings")
    require(outer["wrapper"] == binding(RUN_WRAPPER) and outer["reused_producer"] == binding(RECOVERY_SOURCE), "outer source bindings")
    require(outer["training_updates"] == 256 and outer["segment_model_forwards"] == EXPECTED_SEGMENT_FORWARDS and outer["readback_requests"] == 33, "outer counters")

    checkpoint_fingerprints: dict[str, str] = {}
    for step in STEPS:
        observed = recovery.adapter_identity(step, manifest["model_config"])
        require(observed == readback_manifest["checkpoint_adapters"][str(step)], f"readback checkpoint {step} adapter identity")
        checkpoint_fingerprints[str(step)] = observed["fingerprint"]

    partitions = readback_manifest["partitions"]
    require(len(partitions) == 8 and sum(len(group) for group in partitions) == 33 and max(map(len, partitions)) <= 5, "readback partition shape")
    expected_loads = sum(len({int(job["step"]) for job in group}) for group in partitions)
    require(expected_loads == 9, "readback expected model loads")
    require(sorted({int(job["step"]) for job in partitions[2]}) == [128, 256], "GPU2 mixed checkpoint reload")
    job_to_shard = {(int(job["step"]), int(job["image_id"])): shard for shard, group in enumerate(partitions) for job in group}
    require(len(job_to_shard) == 33, "readback unique jobs")

    exits = read(READBACK_ROOT / "exits.json")["exits"]
    require(len(exits) == 8 and all(int(item["exit_code"]) == 0 for item in exits), "readback worker exits")
    terminal_summaries: list[dict[str, Any]] = []
    model_receipts: dict[tuple[int, int], dict[str, Any]] = {}
    embedding_identity = training["loaded_model"]["model_identity"]["embedding_delta"]
    for shard, (group, exit_receipt) in enumerate(zip(partitions, exits, strict=True)):
        command = exit_receipt["command"]
        require(command[1] == str(RUN_WRAPPER), "readback child wrapper")
        require(command[command.index("--manifest") + 1] == str(readback_manifest_path), "readback child manifest")
        require(command[command.index("--training-root") + 1] == str(TRAINING_ROOT), "readback child training root")
        require(command[command.index("--readback-root") + 1] == str(READBACK_ROOT), "readback child readback root")
        require(command[command.index("--shard") + 1] == str(shard) and command[command.index("--gpu") + 1] == str(shard), "readback child shard/GPU")
        require(command[0].startswith("/root/miniconda3/envs/ms/bin/"), "readback child interpreter")
        terminal = read(READBACK_ROOT / "terminals" / f"shard-{shard}.json")
        require(terminal["status"] == "completed" and terminal["shard"] == terminal["gpu"] == shard, "readback terminal identity")
        require(terminal["expected"] == terminal["completed"] == len(group), "readback terminal jobs")
        steps = sorted({int(job["step"]) for job in group})
        require(terminal["model_loads"] == len(steps), "readback terminal model loads")
        require(terminal["manifest"] == binding(readback_manifest_path), "readback terminal manifest")
        terminal_summaries.append({"shard": shard, "gpu": shard, "jobs": len(group), "model_loads": terminal["model_loads"], "steps": steps, "generated_tokens": terminal["generated_tokens"]})
        for step in steps:
            model_path = READBACK_ROOT / "model" / f"shard-{shard}-step-{step:05d}.json"
            model_receipts[(shard, step)] = _model_receipt_summary(model_path, step, manifest, embedding_identity)
    require(len(model_receipts) == 9 and sum(item["model_loads"] for item in terminal_summaries) == 9, "nine cold checkpoint loads")
    require({key for key in model_receipts if key[0] == 2} == {(2, 128), (2, 256)}, "GPU2 cold reload receipts")

    acquisition = read(manifest["acquisition_manifest"]["path"])
    acquisition_by_image = {int(record["image_id"]): record for record in acquisition["records"]}
    routes = {int(route["image_id"]): route for route in readback_manifest["routes"]}
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_config"]["model"]["base_model"], local_files_only=True)
    parser_counts = {str(step): {"rows": 0, "valid_predictions": 0, "dropped_predictions": 0} for step in STEPS}
    stop_counts = {str(step): {"eos": 0, "cap": 0} for step in STEPS}
    generated_tokens = 0
    durable_rows: list[dict[str, Any]] = []
    for step in STEPS:
        envelope_path = READBACK_ROOT / f"readback-step-{step}.json"
        envelope = read(envelope_path)
        require(envelope["status"] == "completed_unscored" and envelope["manifest"] == binding(MANIFEST), f"readback envelope {step}")
        require(envelope["adapter"] == readback_manifest["checkpoint_adapters"][str(step)], f"readback envelope {step} adapter")
        require(len(envelope["rows"]) == len(IMAGES) and [int(row["image_id"]) for row in envelope["rows"]] == sorted(IMAGES), f"readback envelope {step} cohort")
        bindings = envelope["recovery"]["per_image_rows"]
        require(len(bindings) == len(IMAGES), f"readback envelope {step} row bindings")
        seen: set[int] = set()
        for envelope_row, durable_binding in zip(envelope["rows"], bindings, strict=True):
            image_id = int(envelope_row["image_id"])
            require(image_id in routes and image_id not in seen, f"readback {step} image identity")
            seen.add(image_id)
            row_path = READBACK_ROOT / "rows" / f"step-{step:05d}-image-{image_id:012d}.json"
            require(durable_binding == binding(row_path), f"readback {step}:{image_id} durable row binding")
            durable = read(row_path)
            recovery.validate_row_checkpoint(durable, {"step": step, "image_id": image_id}, readback_manifest)
            _compare_envelope_row(envelope_row, durable)
            route = routes[image_id]
            require(durable["route_id"] == route["route_id"] and durable["image_id"] == image_id, "readback route identity")
            require(durable["checkpoint_step"] == step and durable["checkpoint_adapter"] == readback_manifest["checkpoint_adapters"][str(step)], "readback checkpoint identity")
            require(durable["empty_assistant_prefix"] is True and durable["prompt_token_ids"] == route["prompt_token_ids"], "readback empty prefix/prompt")
            require(digest(durable["prompt_token_ids"]) == durable["prompt_token_ids_sha256"], "readback prompt hash")
            image_path = Path(route["case"]["image_path"])
            require(_hash(image_path) == route["image_identity"]["image_content_sha256"], "readback original media bytes")
            require(durable["executed_media_sha256"] == route["image_identity"]["executed_media_sha256"], "readback executed media")
            require(durable["observed_image_grid_thw"] == route["image_identity"]["observed_image_grid_thw"], "readback image grid")
            ids = durable["generated_token_ids"]
            require(isinstance(ids, list) and all(type(token) is int and token >= 0 for token in ids), "readback raw token IDs")
            require(digest(ids) == durable["generated_token_ids_sha256"], "readback generated-token hash")
            text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            require(text == durable["raw_decode_text"], "readback raw token decode")
            stop = _stop_kind(ids, durable["decode_stop_reason"])
            stop_counts[str(step)][stop] += 1
            generated_tokens += len(ids)
            parsed = native_record(text, route["case"], acquisition_by_image[image_id]["golden"], durable["decode_stop_reason"])
            require(parsed["raw_decode_text"] == text and parsed["decode_stop_reason"] == durable["decode_stop_reason"], "native parser replay identity")
            parser_counts[str(step)]["rows"] += 1
            parser_counts[str(step)]["valid_predictions"] += int(parsed["valid_prediction_count"])
            parser_counts[str(step)]["dropped_predictions"] += int(parsed["dropped_prediction_count"])
            shard = job_to_shard[(step, image_id)]
            require(durable["model_receipt"] == model_receipts[(shard, step)]["receipt"], "row/cold-model receipt binding")
            durable_rows.append(binding(row_path))
        require(seen == set(IMAGES), f"readback step {step} complete image set")

    result = read(READBACK_ROOT / "result.json")
    require(result["status"] == "candidate_ready" and result["request_count"] == 33, "readback result status/count")
    require(result["manifest"] == binding(readback_manifest_path), "readback result manifest")
    require(result["stop_counts"] == {"im_end": sum(item["eos"] for item in stop_counts.values()), "length": sum(item["cap"] for item in stop_counts.values())}, "readback result stop counts")
    require(result["generated_tokens"] == generated_tokens, "readback result generated token count")
    require(result["envelopes"] == [binding(READBACK_ROOT / f"readback-step-{step}.json") for step in STEPS], "readback result envelope bindings")

    return {
        "sources": {
            "manifest": binding(readback_manifest_path),
            "outer_terminal": binding(RUN_ROOT / "terminal.json"),
            "result": binding(READBACK_ROOT / "result.json"),
            "wrapper": binding(RUN_WRAPPER),
            "recovery_source": binding(RECOVERY_SOURCE),
        },
        "coverage": {"steps": list(STEPS), "rows": len(durable_rows), "images_per_step": len(IMAGES), "worker_count": 8, "cold_model_loads": len(model_receipts), "gpu2_reload_steps": [128, 256]},
        "worker_summary": terminal_summaries,
        "stop_counts": stop_counts,
        "parser_replay": parser_counts,
        "generated_tokens": generated_tokens,
        "checkpoint_fingerprints": checkpoint_fingerprints,
    }


def audit() -> dict[str, Any]:
    old, manifest = _manifest_contract()
    training = _training_audit(old, manifest)
    missing = _readback_missing()
    if missing:
        return {
            "schema": "training_set_completion.fourth_fit_execution_acceptance.v1",
            "status": "pending_readback",
            "scope": "CPU-only fourth-fit execution audit; training accepted, readback incomplete on this observation.",
            "manifest": binding(MANIFEST),
            "training": training,
            "missing_readback_artifacts": missing,
            "claim_boundary": "No native readback or physical-owner claim is made.",
        }
    readback = _readback_audit(manifest, read(TRAINING_ROOT / "terminal.json"))
    return {
        "schema": "training_set_completion.fourth_fit_execution_acceptance.v1",
        "status": "candidate_ready",
        "scope": "CPU-only replay of completed fourth-fit continuation training and native readbacks; no physical-owner or model-quality claim.",
        "manifest": binding(MANIFEST),
        "continuation": {
            "predecessor_manifest": binding(PREDECESSOR_MANIFEST),
            "predecessor_state": binding(PREDECESSOR_CHECKPOINT / "state.pt"),
            "allowed_runtime_fields": sorted(ALLOWED_RUNTIME_FIELDS),
            "source": binding(CONTINUATION_SOURCE),
        },
        "training": training,
        "readback": readback,
        "checks": {
            "exact_manifest_extension_and_frozen_sources": True,
            "predecessor_dora_adamw_rng_identity": True,
            "all_192_update_files_and_2112_forwards": True,
            "finite_588_tensor_checkpoints_and_optimizer_states": True,
            "resume_counters_schedule_and_wall_limit": True,
            "cold_checkpoint_model_identity": True,
            "all_33_prompt_media_grid_and_raw_decode_rows": True,
            "atomic_row_envelope_result_bindings": True,
            "native_parser_replay_and_stop_contract": True,
            "eight_workers_nine_loads_gpu2_reload": True,
        },
        "claim_boundary": "The artifacts are accepted as complete execution evidence with raw unscored native readbacks. Physical ownership, unmatched/repeat/false semantics, geometry/class correctness, and model-quality selection remain outside this receipt.",
        "verifier": binding(VERIFY),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", action="store_true", help="replay and compare receipt.json without writing")
    args = parser.parse_args()
    candidate = audit()
    if candidate["status"] == "pending_readback":
        require(not args.verify_existing, "cannot verify an existing receipt while readback is pending")
        print(json.dumps(candidate, sort_keys=True))
        return
    if args.verify_existing:
        require(ACCEPTANCE.is_file(), "accepted receipt is absent")
        require(read(ACCEPTANCE) == candidate, "existing receipt differs from fresh CPU replay")
        print(json.dumps({"status": "verified_existing", "receipt": binding(ACCEPTANCE)}, sort_keys=True))
        return
    require(not ACCEPTANCE.exists(), "accepted receipt already exists; use --verify-existing")
    _publish(ACCEPTANCE, candidate)
    print(json.dumps({"status": candidate["status"], "receipt": binding(ACCEPTANCE)}, sort_keys=True))


if __name__ == "__main__":
    main()

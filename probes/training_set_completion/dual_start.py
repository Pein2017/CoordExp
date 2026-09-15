"""Prepare and run the bounded dual-start COCO-80 comparison.

The outer trial owns the two initializations and the matched new-update labels.
The distributed training backend remains owned by
:mod:`dual_start_distributed`; this module only composes immutable arm
manifests, launches owned processes after an explicit release, and writes
durable per-row natural readbacks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shlex
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

from probes.training_set_completion import training

BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
PREPARATION = BASE / "dual-start-coco80-preparation-v1/preparation.json"
TEACHER = BASE / "dual-start-coco80-teacher-v1/bank.json"
TRIAL_ROOT = BASE / "dual-start-v3"
TMUX_SESSION = "coordexp-dual-start-v3"
REPO = Path("/data/CoordExp/.worktrees/research-probes")
SCHEMA = "training_set_completion.dual_start.v1"
ARMS = ("A", "B")
IMAGE_COUNT = 11
OWNER_COUNT = 218
UPDATES = 256
CHECKPOINT_STEPS = (64, 128, 256)
READBACK_STEPS = (0, 64, 128, 256)
SEED = 42
TRAIN_WALL_SECONDS = 7_200
READBACK_WALL_SECONDS = 7_200
READBACK_TOKEN_CAP = 3_084
EOS = 151_645
TRAINING_GPU_GROUPS = {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}
READBACK_GPUS = list(range(8))
DISTRIBUTED_SOURCE = REPO / "probes/training_set_completion/dual_start_distributed.py"
DEFAULT_SHARED_SOURCES = (
    REPO / "src/losses/runner.py",
    REPO / "src/losses/context.py",
    REPO / "src/losses/raw_axis_validity_hinge.py",
    REPO / "src/losses/__init__.py",
    REPO / "src/coordinate_targets.py",
    REPO / "src/config/models.py",
    DISTRIBUTED_SOURCE,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_file(), f"bound source is not a file: {resolved}")
    return {"path": str(resolved), "sha256": file_hash(resolved), "size_bytes": resolved.stat().st_size}


def publish(path: str | Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == data, f"publication readback differs: {path}")


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def controller_argv(*, trial_path: Path, output: Path, release_path: Path) -> list[str]:
    return [
        "python", "-m", "probes.training_set_completion.dual_start", "controller",
        "--trial", str(trial_path), "--output", str(output), "--release", str(release_path),
    ]


def tmux_launch_command(*, trial_path: Path, output: Path, release_path: Path) -> str:
    return f"tmux new-session -d -s {TMUX_SESSION} {shlex.quote(shlex.join(controller_argv(trial_path=trial_path, output=output, release_path=release_path)))}"


def validate_launch_command(value: Mapping[str, Any], *, trial_path: Path, output: Path, release_path: Path) -> None:
    expected_argv = controller_argv(trial_path=trial_path, output=output, release_path=release_path)
    require(value.get("controller_argv") == expected_argv, "launch controller argv")
    require(value.get("command") == tmux_launch_command(trial_path=trial_path, output=output, release_path=release_path), "launch command")
    shell_tokens = shlex.split(value["command"])
    require(shell_tokens[:5] == ["tmux", "new-session", "-d", "-s", TMUX_SESSION], "tmux command prefix")
    require(shlex.split(shell_tokens[-1]) == expected_argv, "launch command argv parse")
    require(value.get("training_gpu_groups") == TRAINING_GPU_GROUPS, "training GPU groups")
    require(value.get("readback_gpus") == READBACK_GPUS, "readback GPU assignment")


def _verify(value: Mapping[str, Any], label: str) -> None:
    require(set(value) == {"path", "sha256", "size_bytes"}, f"{label} binding fields")
    require(binding(value["path"]) == dict(value), f"{label} changed")


def _verify_with_metadata(value: Mapping[str, Any], label: str, *, metadata: set[str]) -> None:
    require(set(value) == {"path", "sha256", "size_bytes"} | metadata, f"{label} binding fields")
    base = {key: value[key] for key in ("path", "sha256", "size_bytes")}
    require(binding(base["path"]) == base, f"{label} changed")


def validate_preparation(value: Mapping[str, Any]) -> dict[str, Any]:
    require(value.get("schema") == "training_set_completion.dual_start_preparation.v1", "preparation schema")
    require(value.get("status") == "prepared_inputs_not_training_admission", "preparation status")
    require(value.get("target_version") == "coco80-source232-trusted-description-subset-v1", "target version")
    require(value.get("target_counts") == {"included": OWNER_COUNT, "pending_category_or_scope": 13, "confirmed_out_of_scope": 1}, "COCO-80 counts")
    require(value.get("source_population") == 232 and value.get("physical_history_population") == 248, "historical populations")
    image_ids = value.get("image_ids")
    require(isinstance(image_ids, list) and len(image_ids) == IMAGE_COUNT and len(set(image_ids)) == IMAGE_COUNT, "11 image cohort")
    require(set(value.get("arms", {})) == set(ARMS), "dual arm identities")
    for arm in ARMS:
        info = value["arms"][arm]
        require(info.get("adapter", {}).get("path") and info.get("adapter_config", {}).get("path"), f"{arm} adapter bindings")
        _verify(info["adapter"], f"{arm} adapter")
        _verify(info["adapter_config"], f"{arm} adapter config")
        require(info.get("optimizer", "").startswith("fresh AdamW"), f"{arm} fresh optimizer")
    delta = value.get("shared_embedding_delta", {})
    require(delta.get("semantics") == "additive_delta" and delta.get("trainable") is False, "shared embedding delta semantics")
    _verify(delta["tensor"], "shared embedding tensor")
    _verify(delta["metadata"], "shared embedding metadata")
    recipe = value.get("common_recipe", {})
    require(recipe.get("backend") == "HF fp32 SDPA", "common HF FP32 SDPA recipe")
    require(recipe.get("trainable_surface") == "language DoRA rank16; base, vision, embedding and lm_head frozen", "DoRA surface")
    require(recipe.get("optimizer") == {"betas": [0.9, 0.999], "eps": 1e-8, "foreach": False, "lr": 1e-5, "weight_decay": 0}, "optimizer recipe")
    require(recipe.get("geometry_weight") == 0.01 and math.isclose(recipe.get("geometry_margin"), 1 / 999), "geometry recipe")
    require(recipe.get("teacher_order") == "preserve retained source relative order identically for both arms" and recipe.get("refresh") == "none in first comparison", "teacher order/refresh")
    previous = value.get("source_bindings", {}).get("previous_manifest")
    require(isinstance(previous, Mapping), "previous manifest binding")
    _verify(previous, "previous manifest")
    return dict(value)


def _route_owner_ids(route: Mapping[str, Any]) -> list[str]:
    direct = route.get("owner_ids")
    if direct is None:
        provenance = route.get("provenance", {})
        direct = provenance.get("fixed_owner_ids")
        if direct is None:
            direct = provenance.get("selected_owner_ids")
        if direct is None:
            direct = provenance.get("source_owner_ids")
    require(isinstance(direct, list) and all(isinstance(item, str) and item for item in direct), "teacher route owner IDs")
    return list(direct)


def validate_teacher_bank(value: Mapping[str, Any], *, image_ids: Sequence[int]) -> dict[str, Any]:
    require(isinstance(value.get("routes"), list) and len(value["routes"]) == IMAGE_COUNT, "teacher route denominator")
    routes = value["routes"]
    require([int(route.get("image_id", -1)) for route in routes] == [int(item) for item in image_ids], "teacher retained image order")
    owners: list[str] = []
    for route in routes:
        training.validate_route(route, eos_token_id=EOS)
        route_owners = _route_owner_ids(route)
        require(len(route_owners) == len(route["trusted_boxes"]), "teacher owner/box cardinality")
        require(len(route_owners) == len(set(route_owners)), "teacher route owner duplication")
        owners.extend(route_owners)
    require(len(owners) == OWNER_COUNT and len(set(owners)) == OWNER_COUNT, "teacher global owner denominator")
    declared = value.get("fixed_owner_count", value.get("owner_count", OWNER_COUNT))
    require(declared == OWNER_COUNT, "teacher declared owner denominator")
    require(value.get("status") in (None, "candidate_ready", "lead_accepted", "prepared"), "teacher status")
    return dict(value)


def _source_adapter(path: Path, base_model: str) -> dict[str, Any]:
    root = path if path.is_dir() else path.parent
    return training.inspect_dora_adapter_payload(root, base_model)


def _training_manifest(*, arm: str, source: Mapping[str, Any], teacher: Mapping[str, Any], previous: Mapping[str, Any], output: Path, embedding_delta_path: Path) -> dict[str, Any]:
    previous_config = copy.deepcopy(previous["model_config"])
    base_model = str(previous_config["model"]["base_model"])
    adapter_path = Path(source["adapter"]["path"]).resolve().parent
    config = copy.deepcopy(previous_config)
    config["adapter"]["path"] = str(adapter_path)
    require(Path(config["embedding_delta"]["path"]).resolve() == embedding_delta_path.resolve(), "shared embedding delta path changed")
    config["generation"].update(batch_size=1, max_new_tokens=READBACK_TOKEN_CAP, n=1, repetition_penalty=1.0, temperature=0.0, top_p=1.0)
    # The real entry admits this frozen batch-size-one probe only through its
    # explicit smoke flag; it does not alter the objective or exposure.
    config["debug"]["smoke"] = True
    config["run"].update(name=f"dual-start-v3-{arm}", artifact_root=str(output / arm / "training"), output_dir=None)
    source_adapter = _source_adapter(adapter_path, base_model)
    teacher_routes = copy.deepcopy(teacher["routes"])
    # The existing runner validates this exact route schema and remains the
    # single owner of replay/aligned CE and the shared geometry call.
    manifest: dict[str, Any] = {
        "schema": training.SCHEMA,
        "status": "candidate_ready",
        "sources": {
            "reviewed_routes": binding(teacher["_path"]),
            # The distributed backend is the real producer for this manifest;
            # training.py remains bound in the outer trial as the objective
            # and checkpoint-schema dependency.
            "producer": binding(DISTRIBUTED_SOURCE),
        },
        "acquisition_manifest": binding(previous["_path"]),
        "source_adapter": source_adapter,
        "model_config": config,
        "routes": teacher_routes,
        "optimizer": copy.deepcopy(training.DEFAULT_OPTIMIZER),
        "runtime": {"updates": UPDATES, "checkpoint_steps": list(CHECKPOINT_STEPS), "wall_seconds": TRAIN_WALL_SECONDS, "max_model_forwards": UPDATES * IMAGE_COUNT, "eos_token_id": EOS, "seed": SEED, "new_step_labels": list(READBACK_STEPS), "initial_step": 0, "optimizer_mode": "fresh"},
        "validity_hinge": {"weight": 0.01, "margin": 1 / 999, "coordinate_token_ids": list(previous["validity_hinge"]["coordinate_token_ids"]), "coordinate_bin_values": list(range(1000)), "coordinate_token_spellings": [f"<|coord_{index}|>" for index in range(1000)], "coordinate_units": "normalized_0_1_from_raw_bins_0_999"},
    }
    manifest["content_sha256"] = digest(manifest)
    training.validate_manifest(manifest, verify_sources=False)
    return manifest


def _implementation_bindings(shared_sources: Sequence[Path]) -> list[dict[str, Any]]:
    paths = [Path(training.__file__).resolve(), Path(__file__).resolve(), *(Path(path).resolve() for path in shared_sources)]
    unique = []
    seen = set()
    for path in paths:
        require(path.is_file(), f"missing implementation source: {path}")
        if str(path) not in seen:
            unique.append(binding(path)); seen.add(str(path))
    return unique


def prepare_distributed_qualification_manifests(*, output: Path, preparation_path: Path = PREPARATION, teacher_path: Path = TEACHER) -> dict[str, Path]:
    """Publish the bounded two-update manifests consumed by rank qualification."""
    require(not output.exists(), f"qualification output already exists: {output}")
    preparation = validate_preparation(read(preparation_path))
    teacher = read(teacher_path)
    validate_teacher_bank(teacher, image_ids=preparation["image_ids"])
    previous_path = Path(preparation["source_bindings"]["previous_manifest"]["path"])
    previous = read(previous_path)
    preparation["_path"], teacher["_path"], previous["_path"] = str(preparation_path), str(teacher_path), str(previous_path)
    embedding_delta_path = Path(preparation["shared_embedding_delta"]["tensor"]["path"]).resolve().parent
    paths: dict[str, Path] = {}
    for arm in ARMS:
        manifest = _training_manifest(arm=arm, source=preparation["arms"][arm], teacher=teacher, previous=previous, output=output, embedding_delta_path=embedding_delta_path)
        # The bounded qualification uses the real entry's explicit smoke
        # admission for batch_size=1; full trial manifests retain production
        # settings and are unchanged.
        manifest["model_config"]["debug"]["smoke"] = True
        manifest["runtime"].update(updates=2, checkpoint_steps=[2], wall_seconds=600, max_model_forwards=22)
        manifest["content_sha256"] = digest({key: item for key, item in manifest.items() if key != "content_sha256"})
        training.validate_manifest(manifest, verify_sources=False)
        path = output / arm / "training-manifest.json"
        publish(path, manifest)
        paths[arm] = path
    return paths


def prepare(*, preparation_path: Path = PREPARATION, teacher_path: Path = TEACHER, output: Path = TRIAL_ROOT, shared_sources: Sequence[Path] = DEFAULT_SHARED_SOURCES) -> dict[str, Any]:
    require(not output.exists(), f"trial output already exists: {output}")
    preparation = validate_preparation(read(preparation_path))
    teacher = read(teacher_path)
    validate_teacher_bank(teacher, image_ids=preparation["image_ids"])
    previous_path = Path(preparation["source_bindings"]["previous_manifest"]["path"])
    previous = read(previous_path)
    require(previous.get("schema") == training.SCHEMA and previous.get("status") == "candidate_ready", "previous training manifest")
    embedding_delta_path = Path(preparation["shared_embedding_delta"]["tensor"]["path"]).resolve().parent
    preparation["_path"] = str(preparation_path)
    teacher["_path"] = str(teacher_path)
    previous["_path"] = str(previous_path)
    arms: dict[str, Any] = {}
    arm_manifests: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_manifest = _training_manifest(arm=arm, source=preparation["arms"][arm], teacher=teacher, previous=previous, output=output, embedding_delta_path=embedding_delta_path)
        arm_path = output / arm / "training-manifest.json"
        publish(arm_path, arm_manifest)
        arm_manifests[arm] = arm_manifest
        arms[arm] = {"role": preparation["arms"][arm]["role"], "source_adapter": arm_manifest["source_adapter"], "training_manifest": binding(arm_path), "optimizer_mode": "fresh_adamw", "initial_optimizer_state": "empty", "initial_step": 0, "new_step_labels": list(READBACK_STEPS)}
    trial_path = output / "trial.json"
    release_path = output / "release.json"
    trial: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "candidate_ready_release_pending",
        "contract": {"question": "At matched new exposure, does full-teacher step256 or original geo_sorted_xy step2444 initialization improve final natural-greedy FN/F1, incumbent retention and complete-output quality?", "target_version": preparation["target_version"], "teacher_owner_count": OWNER_COUNT, "teacher_route_count": IMAGE_COUNT, "same_seed": SEED, "no_refresh": True, "no_new_training_images": True},
        "preparation": binding(preparation_path),
        "teacher_bank": {**binding(teacher_path), "route_count": IMAGE_COUNT, "owner_count": OWNER_COUNT},
        "shared_embedding_delta": preparation["shared_embedding_delta"],
        "implementation_bindings": _implementation_bindings(shared_sources),
        "arms": arms,
        "execution_topology": {"backend": "torch.distributed.run", "ranks_per_arm": 4, "training_gpu_groups": copy.deepcopy(TRAINING_GPU_GROUPS), "global_update_images": IMAGE_COUNT, "route_partition": "contiguous source-order 3/3/3/2; rank0 publishes one shared terminal/checkpoint set", "readback_workers": 8, "readback_gpu_assignment": "one worker per arm/new-step on GPUs 0..7"},
        "training": {"updates": UPDATES, "images_per_update": IMAGE_COUNT, "image_forwards_per_arm": UPDATES * IMAGE_COUNT, "checkpoint_steps": list(CHECKPOINT_STEPS), "wall_seconds_per_arm": TRAIN_WALL_SECONDS, "optimizer": copy.deepcopy(training.DEFAULT_OPTIMIZER), "clip_norm": 1.0, "segment_reduction": "mean over 11 per-image active-token-normalized CE plus mean geometry per image", "new_step_labels": list(READBACK_STEPS)},
        "readback": {"steps": list(READBACK_STEPS), "request_count": len(ARMS) * len(READBACK_STEPS) * IMAGE_COUNT, "empty_assistant_prefix": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "max_new_tokens": READBACK_TOKEN_CAP, "eos_token_id": EOS, "wall_seconds_per_worker": READBACK_WALL_SECONDS, "max_image_forwards_per_worker": IMAGE_COUNT, "max_generated_tokens_per_worker": IMAGE_COUNT * READBACK_TOKEN_CAP},
        "launch": {"repository": str(REPO), "python": "python", "tmux_session": TMUX_SESSION, "trial_path": str(trial_path), "output_root": str(output), "release_path": str(release_path), "controller_argv": controller_argv(trial_path=trial_path, output=output, release_path=release_path), "command": tmux_launch_command(trial_path=trial_path, output=output, release_path=release_path), "training_gpu_groups": copy.deepcopy(TRAINING_GPU_GROUPS), "readback_gpus": list(READBACK_GPUS), "release_required": True, "release_status": "pending"},
    }
    trial["content_sha256"] = digest(trial)
    publish(trial_path, trial)
    # Make the exact shell command reviewable without introducing a circular hash.
    launch = {"schema": f"{SCHEMA}.launch", "trial": binding(trial_path), "controller_argv": controller_argv(trial_path=trial_path, output=output, release_path=output / "release.json"), "command": tmux_launch_command(trial_path=trial_path, output=output, release_path=output / "release.json"), "training_gpu_groups": copy.deepcopy(TRAINING_GPU_GROUPS), "readback_gpus": list(READBACK_GPUS)}
    validate_launch_command(launch, trial_path=trial_path, output=output, release_path=output / "release.json")
    publish(output / "launch-command.json", launch)
    return trial


def validate_trial(value: Mapping[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    require(value.get("schema") == SCHEMA and value.get("status") == "candidate_ready_release_pending", "trial schema/status")
    content = {key: item for key, item in value.items() if key != "content_sha256"}
    require(value.get("content_sha256") == digest(content), "trial content hash")
    if verify_sources:
        _verify(value["preparation"], "preparation")
        _verify_with_metadata(value["teacher_bank"], "teacher bank", metadata={"route_count", "owner_count"})
        preparation = validate_preparation(read(value["preparation"]["path"]))
        teacher = read(value["teacher_bank"]["path"])
        validate_teacher_bank(teacher, image_ids=preparation["image_ids"])
        require(value["shared_embedding_delta"] == preparation["shared_embedding_delta"], "shared embedding delta identity")
        for item in value["implementation_bindings"]:
            _verify(item, "implementation source")
        launch = value.get("launch", {})
        validate_launch_command(launch, trial_path=Path(launch["trial_path"]), output=Path(launch["output_root"]), release_path=Path(launch["release_path"]))
    require(value.get("execution_topology") == {"backend": "torch.distributed.run", "ranks_per_arm": 4, "training_gpu_groups": copy.deepcopy(TRAINING_GPU_GROUPS), "global_update_images": IMAGE_COUNT, "route_partition": "contiguous source-order 3/3/3/2; rank0 publishes one shared terminal/checkpoint set", "readback_workers": 8, "readback_gpu_assignment": "one worker per arm/new-step on GPUs 0..7"}, "execution topology")
    require(value["training"] == {"updates": UPDATES, "images_per_update": IMAGE_COUNT, "image_forwards_per_arm": UPDATES * IMAGE_COUNT, "checkpoint_steps": list(CHECKPOINT_STEPS), "wall_seconds_per_arm": TRAIN_WALL_SECONDS, "optimizer": copy.deepcopy(training.DEFAULT_OPTIMIZER), "clip_norm": 1.0, "segment_reduction": "mean over 11 per-image active-token-normalized CE plus mean geometry per image", "new_step_labels": list(READBACK_STEPS)}, "training contract")
    require(value["readback"]["request_count"] == 88 and value["readback"]["steps"] == list(READBACK_STEPS), "readback denominator")
    require(set(value["arms"]) == set(ARMS), "arm set")
    for arm in ARMS:
        info = value["arms"][arm]
        require(info["optimizer_mode"] == "fresh_adamw" and info.get("initial_optimizer_state") == "empty" and info.get("initial_step") == 0 and info["new_step_labels"] == list(READBACK_STEPS), f"{arm} optimizer/step mapping")
        if verify_sources:
            _verify(info["training_manifest"], f"{arm} training manifest")
            manifest = read(info["training_manifest"]["path"])
            expected_adapter_root = str(Path(preparation["arms"][arm]["adapter"]["path"]).resolve().parent)
            require(manifest.get("source_adapter", {}).get("root") == expected_adapter_root, f"{arm} wrong source adapter")
            require(info.get("source_adapter") == manifest.get("source_adapter"), f"{arm} adapter identity mismatch")
            require(Path(manifest.get("model_config", {}).get("embedding_delta", {}).get("path", "")).resolve() == Path(preparation["shared_embedding_delta"]["tensor"]["path"]).resolve().parent, f"{arm} embedding delta path")
            require(manifest.get("runtime", {}).get("seed") == SEED and manifest.get("runtime", {}).get("new_step_labels") == list(READBACK_STEPS) and manifest.get("runtime", {}).get("initial_step") == 0 and manifest.get("runtime", {}).get("optimizer_mode") == "fresh", f"{arm} runtime mapping")
            require(manifest.get("optimizer") == training.DEFAULT_OPTIMIZER, f"{arm} optimizer")
    return dict(value)


def readback_jobs(trial: Mapping[str, Any], *, arm: str, step: int) -> list[dict[str, Any]]:
    require(arm in ARMS and step in READBACK_STEPS, "readback arm/step")
    manifest = read(trial["arms"][arm]["training_manifest"]["path"])
    jobs = []
    for route in manifest["routes"]:
        jobs.append({"arm": arm, "step": step, "image_id": int(route["image_id"]), "route_id": route["route_id"], "request_id": f"dual-start:{arm}:new-step-{step}:image-{int(route['image_id']):012d}"})
    require(len(jobs) == IMAGE_COUNT and len({row["image_id"] for row in jobs}) == IMAGE_COUNT, "readback jobs")
    return jobs


def validate_readback_rows(rows: Sequence[Mapping[str, Any]], jobs: Sequence[Mapping[str, Any]], *, trial_path: Path | None = None, routes: Mapping[int, Mapping[str, Any]] | None = None, manifest_path: Path | None = None, expected_adapter: Mapping[str, Any] | None = None) -> None:
    expected = {(str(job["arm"]), int(job["step"]), int(job["image_id"])) for job in jobs}
    actual = {(str(row.get("arm")), int(row.get("step", -1)), int(row.get("image_id", -1))) for row in rows}
    require(actual == expected and len(rows) == len(expected), "readback request completeness")
    for row in rows:
        require(row.get("empty_assistant_prefix") is True and row.get("temperature") == 0.0 and row.get("top_p") == 1.0 and row.get("top_k") == 0 and row.get("repetition_penalty") == 1.0, "readback decode contract")
        require(row.get("max_new_tokens") == READBACK_TOKEN_CAP, "readback decode token cap")
        require(row.get("checkpoint_step") == row.get("step"), "mixed checkpoint row")
        require(row.get("generated_token_ids_sha256") == digest(row.get("generated_token_ids")), "generated token hash")
        job = next(job for job in jobs if int(job["image_id"]) == int(row["image_id"]))
        require(row.get("request_id") == job["request_id"] and row.get("route_id") == job["route_id"], "readback route identity")
        if routes is not None:
            route = routes[int(row["image_id"])]
            require(row.get("prompt_token_ids") == route["prompt_token_ids"] and row.get("executed_media_sha256") == route["image_identity"]["executed_media_sha256"] and row.get("observed_image_grid_thw") == route["image_identity"]["observed_image_grid_thw"], "readback prompt/media binding")
        if manifest_path is not None:
            require(row.get("training_manifest") == binding(manifest_path), "readback manifest binding")
        if trial_path is not None:
            require(row.get("trial") == binding(trial_path), "readback trial binding")
        if expected_adapter is not None:
            require(row.get("adapter") == dict(expected_adapter), "readback adapter fingerprint")
            loaded = row.get("loaded_model")
            require(isinstance(loaded, Mapping), "readback loaded model identity")
            loaded_adapter = loaded.get("model_identity", {}).get("adapter", {})
            require(isinstance(loaded_adapter, Mapping) and loaded_adapter.get("adapter_path") == expected_adapter.get("root") and loaded_adapter.get("merged_adapters", []) == [], "readback loaded adapter identity")
        ids = row.get("generated_token_ids")
        require(isinstance(ids, list) and ids and len(ids) <= READBACK_TOKEN_CAP and all(type(item) is int and item >= 0 for item in ids), "readback token budget")
        stop = row.get("decode_stop_reason")
        require((stop == "im_end" and ids[-1] == EOS and EOS not in ids[:-1]) or (stop == "length" and len(ids) == READBACK_TOKEN_CAP and EOS not in ids), "readback stop contract")


def _adapter_for_step(trial: Mapping[str, Any], arm: str, step: int, output: Path) -> Path:
    if step == 0:
        manifest = read(trial["arms"][arm]["training_manifest"]["path"])
        return Path(manifest["source_adapter"]["root"])
    return output / arm / "training" / "checkpoints" / f"step-{step:05d}" / "adapter"


def pending_readback_jobs(*, trial_path: Path, output: Path, arm: str, step: int, adapter_identity: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (all, missing), retaining and validating rows already on disk."""
    trial = validate_trial(read(trial_path))
    require(Path(trial["launch"]["output_root"]).resolve() == output.resolve(), "readback recovery must use the same trial output root")
    jobs = readback_jobs(trial, arm=arm, step=step)
    manifest_path = Path(trial["arms"][arm]["training_manifest"]["path"])
    manifest = read(manifest_path)
    routes = {int(route["image_id"]): route for route in manifest["routes"]}
    root = output / "readback" / arm / f"new-step-{step:03d}" / "rows"
    retained = []
    missing = []
    for job in jobs:
        path = root / f"image-{job['image_id']:012d}.json"
        if path.is_file():
            row = read(path)
            validate_readback_rows([row], [job], trial_path=trial_path, routes=routes, manifest_path=manifest_path, expected_adapter=adapter_identity)
            retained.append(row)
        else:
            require(not path.exists(), "readback row path is not a regular file")
            missing.append(job)
    terminal_path = output / "readback" / arm / f"new-step-{step:03d}" / "terminal.json"
    if terminal_path.is_file():
        terminal = read(terminal_path)
        require(terminal.get("status") == "completed" and not missing, "completed readback terminal has missing rows")
    return jobs, missing


def readback_worker(*, trial_path: Path, output: Path, arm: str, step: int, gpu: int) -> None:
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from probes.source_rweak_row_cross.run import build_requests
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    trial = validate_trial(read(trial_path))
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu) and torch.cuda.device_count() == 1, "GPU isolation")
    manifest = read(trial["arms"][arm]["training_manifest"]["path"])
    adapter = _adapter_for_step(trial, arm, step, output)
    training.validate_manifest(manifest)
    adapter_identity = training.inspect_dora_adapter_payload(adapter, manifest["model_config"]["model"]["base_model"])
    if step == 0:
        require(adapter_identity == manifest["source_adapter"], "step0 source adapter identity")
    else:
        terminal = read(output / arm / "training" / "terminal.json")
        checkpoint = next((item for item in terminal.get("checkpoints", []) if int(item.get("step", -1)) == step), None)
        require(checkpoint is not None and checkpoint.get("adapter") == adapter_identity, "readback checkpoint identity")
    jobs, pending = pending_readback_jobs(trial_path=trial_path, output=output, arm=arm, step=step, adapter_identity=adapter_identity)
    terminal_path = output / "readback" / arm / f"new-step-{step:03d}" / "terminal.json"
    if not pending:
        if not terminal_path.is_file():
            publish(terminal_path, {"schema": f"{SCHEMA}.readback_terminal", "status": "completed", "arm": arm, "step": step, "gpu": gpu, "pid": os.getpid(), "request_count": len(jobs), "image_forwards": 0, "generated_tokens": 0, "recovered_without_generation": True})
        return
    config = checkpoint_config(InferConfig.model_validate(manifest["model_config"]), str(adapter))
    qwen, loaded = load_policy(config, device=torch.device("cuda:0"))
    qwen.model.eval()
    policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, use_model_defaults=False)
    rows = []
    route_by_image = {int(route["image_id"]): route for route in manifest["routes"]}
    row_root = output / "readback" / arm / f"new-step-{step:03d}" / "rows"
    started = time.monotonic()
    for job in pending:
        route = route_by_image[job["image_id"]]
        requests, _ = build_requests(qwen, manifest["model_config"], [route["case"]])
        batch = prepare_native_inputs(qwen.processor, requests, device="cuda:0", record_media_identity=True)
        require(list(batch.prompt_token_ids[0]) == route["prompt_token_ids"], "readback prompt identity")
        require(batch.media_sha256[0] == route["image_identity"]["executed_media_sha256"] and list(batch.image_grids[0]) == route["image_identity"]["observed_image_grid_thw"], "readback media identity")
        with torch.inference_mode():
            generated, = generate_continuations(qwen.model, batch, extensions=[[]], budgets=[READBACK_TOKEN_CAP], eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace="none", seed=None)
        ids = list(generated.token_ids)
        require((generated.stop_reason == "im_end" and ids and ids[-1] == EOS and EOS not in ids[:-1]) or (generated.stop_reason == "length" and len(ids) == READBACK_TOKEN_CAP and EOS not in ids), "readback terminal")
        row = {"schema": f"{SCHEMA}.readback_row", "arm": arm, "step": step, "checkpoint_step": step, "route_id": route["route_id"], "image_id": route["image_id"], "request_id": job["request_id"], "empty_assistant_prefix": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "max_new_tokens": READBACK_TOKEN_CAP, "prompt_token_ids": route["prompt_token_ids"], "generated_token_ids": ids, "generated_token_ids_sha256": digest(ids), "decode_stop_reason": generated.stop_reason, "raw_decode_text": qwen.tokenizer.decode(ids, skip_special_tokens=False), "executed_media_sha256": batch.media_sha256[0], "observed_image_grid_thw": list(batch.image_grids[0]), "adapter": adapter_identity, "loaded_model": loaded, "training_manifest": binding(trial["arms"][arm]["training_manifest"]["path"]), "trial": binding(trial_path)}
        publish(row_root / f"image-{job['image_id']:012d}.json", row)
        rows.append(row)
    validate_readback_rows(rows, pending, trial_path=trial_path, routes=route_by_image, manifest_path=Path(trial["arms"][arm]["training_manifest"]["path"]), expected_adapter=adapter_identity)
    publish(output / "readback" / arm / f"new-step-{step:03d}" / "terminal.json", {"schema": f"{SCHEMA}.readback_terminal", "status": "completed", "arm": arm, "step": step, "gpu": gpu, "pid": os.getpid(), "request_count": len(jobs), "generated_request_count": len(rows), "retained_request_count": len(jobs) - len(pending), "image_forwards": len(rows), "generated_tokens": sum(len(row["generated_token_ids"]) for row in rows), "elapsed_seconds": time.monotonic() - started})


def collect_readbacks(trial_path: Path, output: Path) -> dict[str, Any]:
    trial = validate_trial(read(trial_path))
    all_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for step in READBACK_STEPS:
            jobs = readback_jobs(trial, arm=arm, step=step)
            rows = [read(output / "readback" / arm / f"new-step-{step:03d}" / "rows" / f"image-{job['image_id']:012d}.json") for job in jobs]
            manifest_path = Path(trial["arms"][arm]["training_manifest"]["path"])
            manifest = read(manifest_path)
            adapter = _adapter_for_step(trial, arm, step, output)
            adapter_identity = training.inspect_dora_adapter_payload(adapter, manifest["model_config"]["model"]["base_model"])
            validate_readback_rows(rows, jobs, trial_path=trial_path, routes={int(route["image_id"]): route for route in manifest["routes"]}, manifest_path=manifest_path, expected_adapter=adapter_identity)
            require(all(int(row["step"]) == step and row["arm"] == arm for row in rows), "mixed checkpoint collection")
            all_rows.extend(rows)
    result = {"schema": f"{SCHEMA}.readback_result", "status": "completed_unscored", "trial": binding(trial_path), "request_count": len(all_rows), "rows": [{"arm": row["arm"], "step": row["step"], "image_id": row["image_id"], "generated_token_ids_sha256": row["generated_token_ids_sha256"], "decode_stop_reason": row["decode_stop_reason"]} for row in all_rows]}
    publish(output / "readback-result.json", result)
    return result


def load_admitted_readback_rows(result_path: Path, *, arm: str, step: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one complete, provenance-checked 11-row endpoint for evaluation.

    The collection receipt is the admission boundary.  Rows are revalidated
    from their durable files so a score consumer cannot rely on summary hashes
    or caller-supplied arm/step labels alone.
    """
    result = read(result_path)
    require(result.get("schema") == f"{SCHEMA}.readback_result" and result.get("status") == "completed_unscored", "readback collection receipt")
    trial_binding = result.get("trial")
    require(isinstance(trial_binding, Mapping), "readback collection trial binding")
    _verify(trial_binding, "readback collection trial")
    trial_path = Path(trial_binding["path"])
    trial = validate_trial(read(trial_path))
    output = Path(result_path).resolve().parent
    require(Path(trial["launch"]["output_root"]).resolve() == output, "readback collection output root")
    jobs = readback_jobs(trial, arm=arm, step=step)
    manifest_path = Path(trial["arms"][arm]["training_manifest"]["path"])
    manifest = read(manifest_path)
    adapter = _adapter_for_step(trial, arm, step, output)
    adapter_identity = training.inspect_dora_adapter_payload(adapter, manifest["model_config"]["model"]["base_model"])
    rows = [read(output / "readback" / arm / f"new-step-{step:03d}" / "rows" / f"image-{job['image_id']:012d}.json") for job in jobs]
    validate_readback_rows(rows, jobs, trial_path=trial_path, routes={int(route["image_id"]): route for route in manifest["routes"]}, manifest_path=manifest_path, expected_adapter=adapter_identity)
    admitted = [item for item in result.get("rows", []) if item.get("arm") == arm and int(item.get("step", -1)) == step]
    require(len(admitted) == IMAGE_COUNT and {int(item["image_id"]) for item in admitted} == {int(job["image_id"]) for job in jobs}, "readback collection endpoint")
    summary = {(int(item["image_id"]), item["generated_token_ids_sha256"]) for item in admitted}
    require(summary == {(int(row["image_id"]), row["generated_token_ids_sha256"]) for row in rows}, "readback collection row hashes")
    return rows, result


def _owned_kill(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _stop_live(processes: Sequence[tuple[subprocess.Popen[Any], Any, list[str], str, Any, float]]) -> None:
    for process, _stream, _command, _name, _gpu, _spawned_at in processes:
        if process.poll() is None:
            _owned_kill(process)


def distributed_training_command(*, manifest_path: Path, output: Path) -> list[str]:
    """Return the exact four-rank backend invocation used by the controller."""
    require(DISTRIBUTED_SOURCE.is_file(), f"missing distributed backend: {DISTRIBUTED_SOURCE}")
    return [
        "python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
        "-m", "probes.training_set_completion.dual_start_distributed",
        "--manifest", str(manifest_path), "--output", str(output),
    ]


def validate_training_terminal(output: Path, *, arm: str) -> dict[str, Any]:
    """Admission check before any readback process is allowed to start."""
    terminal = read(output / "terminal.json")
    require(terminal.get("status") == "completed" and terminal.get("optimizer_mode") == "fresh", f"{arm} training terminal")
    require(terminal.get("updates") == UPDATES and terminal.get("model_forwards") == UPDATES * IMAGE_COUNT, f"{arm} training exposure")
    require([int(item["step"]) for item in terminal.get("checkpoints", [])] == list(CHECKPOINT_STEPS), f"{arm} checkpoint schedule")
    seed = read(output.parent / "training-seed.json")
    require(seed.get("seed") == SEED and seed.get("mode") == "fresh_adamw", f"{arm} seed receipt")
    import torch
    for checkpoint in terminal["checkpoints"]:
        state = torch.load(checkpoint["state"]["path"], map_location="cpu", weights_only=False)
        observed = {int(value["step"].item()) for value in state["optimizer_state_dict"]["state"].values() if isinstance(value, Mapping) and "step" in value}
        require(observed == {int(checkpoint["step"])}, f"{arm} optimizer step state")
    return terminal


def _spawn(command: list[str], *, visible_devices: str, log_path: Path) -> tuple[subprocess.Popen[Any], Any, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("x")
    spawned_at = time.monotonic()
    process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": visible_devices, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}, start_new_session=True)
    return process, stream, spawned_at


def wait_owned_processes(processes: Sequence[tuple[subprocess.Popen[Any], Any, list[str], str, Any, float]], *, wall_seconds: float) -> list[dict[str, Any]]:
    """Wait sequentially while preserving each child's own spawn deadline."""
    result = []
    for process, stream, command, name, gpu, spawned_at in processes:
        deadline = spawned_at + float(wall_seconds)
        try:
            # A child may have exited while an earlier sibling consumed this
            # waiter's wall budget.  Completed work is successful even when
            # observed after its deadline; only a live child is terminated.
            code = process.poll()
            if code is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, wall_seconds)
                code = process.wait(timeout=remaining)
            result.append({"name": name, "pid": process.pid, "gpu": gpu, "exit_code": code, "command": command, "spawned_at_monotonic": spawned_at, "deadline_monotonic": deadline})
        except subprocess.TimeoutExpired:
            _owned_kill(process)
            result.append({"name": name, "pid": process.pid, "gpu": gpu, "exit_code": "timeout", "command": command, "spawned_at_monotonic": spawned_at, "deadline_monotonic": deadline})
            raise
        finally:
            stream.close()
    return result


def controller(*, trial_path: Path, output: Path, release_path: Path) -> None:
    trial = validate_trial(read(trial_path))
    release = read(release_path)
    require(release.get("status") == "released" and release.get("trial_sha256") == file_hash(trial_path), "explicit trial release")
    require(not (output / "controller-terminal.json").exists(), "controller terminal collision")
    output.mkdir(parents=True, exist_ok=True)
    controller_identity = {"schema": f"{SCHEMA}.controller_identity", "pid": os.getpid(), "trial": binding(trial_path), "release": binding(release_path), "started_at": time.time()}
    publish(output / "controller-identity.json", controller_identity)
    log_path = output / "controller.log"
    log = log_path.open("x")
    def emit(line: str) -> None:
        log.write(line + "\n"); log.flush(); os.fsync(log.fileno())
    emit(f"DUAL_START_CONTROLLER pid={os.getpid()} trial_sha256={controller_identity['trial']['sha256']}")
    processes: list[tuple[subprocess.Popen[Any], Any, list[str], str, Any, float]] = []
    exits: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for arm in ARMS:
            gpu_group = TRAINING_GPU_GROUPS[arm]
            seed_path = output / arm / "training-seed.json"
            publish(seed_path, {"schema": f"{SCHEMA}.training_seed", "seed": SEED, "pid": os.getpid(), "manifest": binding(output / arm / "training-manifest.json"), "mode": "fresh_adamw", "world_size": 4, "gpu_group": gpu_group})
            command = distributed_training_command(manifest_path=output / arm / "training-manifest.json", output=output / arm / "training")
            visible_devices = ",".join(str(gpu) for gpu in gpu_group)
            process, stream, spawned_at = _spawn(command, visible_devices=visible_devices, log_path=output / "logs" / f"train-{arm}.log")
            processes.append((process, stream, command, f"train-{arm}", gpu_group, spawned_at))
        exits.extend(wait_owned_processes(processes, wall_seconds=TRAIN_WALL_SECONDS))
        require(all(item["exit_code"] == 0 for item in exits), "training arm failure")
        for arm in ARMS:
            validate_training_terminal(output / arm / "training", arm=arm)
        publish(output / "training-exits.json", {"schema": f"{SCHEMA}.training_exits", "exits": exits})
        processes = []
        jobs = [(arm, step, gpu) for gpu, (arm, step) in enumerate(((arm, step) for arm in ARMS for step in READBACK_STEPS))]
        for arm, step, gpu in jobs:
            command = ["python", "-m", "probes.training_set_completion.dual_start", "readback-worker", "--trial", str(trial_path), "--output", str(output), "--arm", arm, "--step", str(step), "--gpu", str(gpu)]
            process, stream, spawned_at = _spawn(command, visible_devices=str(gpu), log_path=output / "logs" / f"readback-{arm}-step-{step:03d}.log")
            processes.append((process, stream, command, f"readback-{arm}-step-{step}", gpu, spawned_at))
        exits.extend(wait_owned_processes(processes, wall_seconds=READBACK_WALL_SECONDS))
        publish(output / "readback-exits.json", {"schema": f"{SCHEMA}.readback_exits", "exits": exits})
        require(all(item["exit_code"] == 0 for item in exits), "readback worker failure")
        collect_readbacks(trial_path, output)
        status = "completed_unscored"
        emit("DUAL_START_COMPLETED_UNSCORED")
        terminal = {"schema": f"{SCHEMA}.controller_terminal", "status": status, "pid": os.getpid(), "trial": controller_identity["trial"], "release": controller_identity["release"], "exits": exits, "elapsed_seconds": time.monotonic() - started, "result": binding(output / "readback-result.json")}
    except BaseException as error:
        _stop_live(processes)
        emit(f"DUAL_START_FAILED error={type(error).__name__}: {error}")
        terminal = {"schema": f"{SCHEMA}.controller_terminal", "status": "failed", "pid": os.getpid(), "trial": controller_identity["trial"], "release": controller_identity["release"], "exits": exits, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started}
    finally:
        publish(output / "controller-terminal.json", terminal)
        log.close()
    if terminal["status"] != "completed_unscored":
        raise RuntimeError(terminal["error"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--preparation", type=Path, default=PREPARATION); p.add_argument("--teacher", type=Path, default=TEACHER); p.add_argument("--output", type=Path, default=TRIAL_ROOT); p.add_argument("--shared-source", type=Path, action="append", default=list(DEFAULT_SHARED_SOURCES))
    p = sub.add_parser("verify"); p.add_argument("--trial", type=Path, required=True)
    p = sub.add_parser("controller"); p.add_argument("--trial", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--release", type=Path, required=True)
    p = sub.add_parser("readback-worker"); p.add_argument("--trial", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--arm", choices=ARMS, required=True); p.add_argument("--step", type=int, required=True); p.add_argument("--gpu", type=int, required=True)
    p = sub.add_parser("collect"); p.add_argument("--trial", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(preparation_path=args.preparation, teacher_path=args.teacher, output=args.output, shared_sources=args.shared_source)
    elif args.command == "verify":
        validate_trial(read(args.trial))
    elif args.command == "controller":
        controller(trial_path=args.trial, output=args.output, release_path=args.release)
    elif args.command == "readback-worker":
        readback_worker(trial_path=args.trial, output=args.output, arm=args.arm, step=args.step, gpu=args.gpu)
    else:
        collect_readbacks(args.trial, args.output)


if __name__ == "__main__":
    main()

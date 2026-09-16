"""New-only batch-4 readback for the Source256 CE-normalization successor.

Source/A/original-B have already been accepted under the predecessor packet and
are never generated here.  This worker owns only B-normalized step16/64 shards
and uses the predecessor's frozen cohort, decode plan and batch-4 qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from probes.training_set_completion import source256_evaluation as predecessor_evaluation
from probes.training_set_completion import source256_readback as predecessor_readback
from probes.training_set_completion import source256_normalized_evaluation as controls
from probes.training_set_completion import training


SCHEMA = "training_set_completion.source256_completion_ce_normalization_readback.v1"
ARM = "B-normalized"
ENDPOINTS = (("Bnormalized16", 16), ("Bnormalized64", 64))
QUALIFICATION_ENDPOINT = {"label": "BnormalizedQualification2", "arm": ARM, "step": 2}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _runtime() -> Any:
    # Imported only when the training owner has supplied the successor runtime.
    from probes.training_set_completion import source256_normalized_training

    return source256_normalized_training


def _endpoint(label: str, step: int) -> dict[str, Any]:
    require((label, step) in ENDPOINTS, "new-only normalized endpoint")
    return {"label": label, "arm": ARM, "step": step}


def validate_static_inputs(
    *,
    control_reuse_path: Path,
    plan_path: Path,
    qualification_path: Path,
    training_manifest_path: Path,
    allow_qualification: bool = False,
) -> dict[str, Any]:
    """Validate the immutable CPU/readback surface shared by all new shards."""

    control_reuse_path = control_reuse_path.resolve(strict=True)
    control = controls.validate_control_reuse_identity(read(control_reuse_path))
    plan_path = plan_path.resolve(strict=True)
    qualification_path = qualification_path.resolve(strict=True)
    require(
        training.binding(plan_path) == control["readback_plan"],
        "successor must reuse the frozen predecessor readback plan",
    )
    require(
        training.binding(qualification_path) == control["batch4_qualification"],
        "successor must reuse the frozen batch4 qualification",
    )
    plan = predecessor_readback.validate_plan(read(plan_path))
    require(
        predecessor_readback._qualified_batch(qualification_path) == 4,
        "frozen batch4 readback",
    )
    runtime = _runtime()
    manifest_path = training_manifest_path.resolve(strict=True)
    manifest = runtime.validate_training_manifest(read(manifest_path))
    require(manifest.get("arm") == ARM, "B-normalized training manifest arm")
    modes = ("qualification", "main") if allow_qualification else ("main",)
    require(manifest.get("mode") in modes, "readback manifest mode")
    require(
        manifest.get("preparation") == control["predecessor"]["preparation"],
        "successor must preserve the predecessor known-owner bank",
    )
    predecessor_readback._model_config(manifest)
    return {
        "control_reuse": control,
        "control_reuse_binding": training.binding(control_reuse_path),
        "plan": plan,
        "plan_path": plan_path,
        "qualification_path": qualification_path,
        "training_manifest": manifest,
        "training_manifest_path": manifest_path,
    }


def validate_training_terminal(
    *, training_manifest_path: Path, terminal_path: Path
) -> dict[str, Any]:
    """Admit a completed new-arm checkpoint without accepting a stale run."""

    runtime = _runtime()
    manifest_path = training_manifest_path.resolve(strict=True)
    manifest = runtime.validate_training_manifest(read(manifest_path))
    terminal_path = terminal_path.resolve(strict=True)
    terminal = read(terminal_path)
    expected_steps = list(manifest["runtime"]["checkpoint_steps"])
    require(
        terminal.get("schema") == f"{runtime.SCHEMA}.terminal.v1"
        and terminal.get("status") == "completed"
        and terminal.get("arm") == ARM
        and terminal.get("mode") == manifest["mode"]
        and terminal.get("manifest") == training.binding(manifest_path)
        and terminal.get("preparation") == manifest["preparation"]
        and terminal.get("optimizer_mode") == "fresh"
        and terminal.get("updates") == manifest["runtime"]["updates"]
        and terminal.get("logical_model_forwards") == manifest["runtime"]["max_model_forwards"]
        and terminal.get("model_calls") == manifest["runtime"]["max_model_calls"],
        "B-normalized completed training terminal",
    )
    checkpoints = terminal.get("checkpoints")
    require(
        isinstance(checkpoints, list)
        and [item.get("step") for item in checkpoints] == expected_steps,
        "B-normalized checkpoint steps",
    )
    for checkpoint in checkpoints:
        adapter = checkpoint.get("adapter")
        adapter_root = Path(str(adapter.get("root", ""))).resolve(strict=True)
        observed = training.inspect_dora_adapter_payload(
            adapter_root, manifest["model_config"]["model"]["base_model"]
        )
        require(observed == adapter, "B-normalized checkpoint adapter payload")
    return {"manifest": manifest, "terminal": terminal}


def _adapter(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    terminal_path: Path,
    step: int,
) -> tuple[Path, Mapping[str, Any]]:
    require(step in {item[1] for item in ENDPOINTS}, "new normalized checkpoint step")
    return _checkpoint_adapter(
        manifest_path=manifest_path,
        manifest=manifest,
        terminal_path=terminal_path,
        step=step,
    )


def _checkpoint_adapter(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    terminal_path: Path,
    step: int,
) -> tuple[Path, Mapping[str, Any]]:
    checked = validate_training_terminal(
        training_manifest_path=manifest_path, terminal_path=terminal_path
    )
    terminal = checked["terminal"]
    matches = [item for item in terminal["checkpoints"] if item.get("step") == step]
    require(len(matches) == 1, "one new normalized checkpoint")
    adapter = matches[0]["adapter"]
    path = Path(adapter["root"]).resolve(strict=True)
    observed = training.inspect_dora_adapter_payload(
        path, manifest["model_config"]["model"]["base_model"]
    )
    require(observed == adapter, "new normalized checkpoint adapter binding")
    return path, observed


def _train_items(prepared: Mapping[str, Any], image_ids: Sequence[int]) -> list[dict[str, Any]]:
    records = _runtime().hydrate_bound_cases(prepared)
    by_image = {int(record["image_id"]): record for record in records}
    require(len(by_image) == 256 and set(image_ids) <= set(by_image), "normalized train item IDs")
    return [
        {
            "split": "train",
            "row_index": int(by_image[image_id]["canonical_route"]["case"]["row_index"]),
            "image_id": image_id,
            "example_id": by_image[image_id]["example_id"],
            "case": by_image[image_id]["canonical_route"]["case"],
            "expected_prompt_token_ids": by_image[image_id]["canonical_route"]["prompt_token_ids"],
            "expected_media_sha256": by_image[image_id]["canonical_route"]["image_identity"]["executed_media_sha256"],
            "expected_grid": by_image[image_id]["canonical_route"]["image_identity"]["observed_image_grid_thw"],
        }
        for image_id in image_ids
    ]


def endpoint_worker(
    *,
    control_reuse_path: Path,
    plan_path: Path,
    qualification_path: Path,
    training_manifest_path: Path,
    terminal_path: Path,
    label: str,
    step: int,
    split: str,
    shard: int,
    output: Path,
    device: str,
) -> dict[str, Any]:
    """Generate exactly one missing B-normalized endpoint shard at batch four."""

    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig

    endpoint = _endpoint(label, step)
    require(split in predecessor_readback.SPLIT_COUNTS, "normalized readback split")
    require(0 <= shard < predecessor_readback.ENDPOINT_SHARDS, "normalized readback shard")
    require(device.startswith("cuda") and torch.cuda.is_available(), "CUDA normalized endpoint readback")
    require(not output.exists() and not output.is_symlink(), "normalized endpoint output collision")
    checked = validate_static_inputs(
        control_reuse_path=control_reuse_path,
        plan_path=plan_path,
        qualification_path=qualification_path,
        training_manifest_path=training_manifest_path,
    )
    manifest_path = checked["training_manifest_path"]
    manifest = checked["training_manifest"]
    adapter_path, adapter = _adapter(
        manifest_path=manifest_path,
        manifest=manifest,
        terminal_path=terminal_path,
        step=step,
    )
    image_ids = checked["plan"]["plan"]["endpoint_shards"][split][shard]
    items = (
        _train_items(checked["plan"]["prepared"], image_ids)
        if split == "train"
        else predecessor_readback._dev_items(checked["plan"]["prepared"], image_ids)
    )
    require(len(items) % 4 == 0, "full batch4 normalized endpoint shard")
    config_data = predecessor_readback._model_config(manifest)
    config = checkpoint_config(InferConfig.model_validate(config_data), str(adapter_path))
    torch.cuda.set_device(device)
    qwen, loaded = load_policy(config, device=torch.device(device))
    predecessor_readback._loaded_identity(loaded, adapter_path=adapter_path)
    generated = predecessor_readback._generate(
        qwen,
        config,
        config_data,
        items,
        batch_size=4,
        formal=True,
    )
    value = {
        "schema": f"{SCHEMA}.endpoint_shard",
        "status": "completed_unscored",
        "endpoint": endpoint,
        "split": split,
        "shard": shard,
        "shard_count": predecessor_readback.ENDPOINT_SHARDS,
        "policy": dict(predecessor_readback.POLICY),
        "batch_size": 4,
        "control_reuse": checked["control_reuse_binding"],
        "plan": training.binding(plan_path),
        "qualification": training.binding(qualification_path),
        "training_manifest": training.binding(manifest_path),
        "training_terminal": training.binding(terminal_path),
        "checkpoint_adapter": adapter,
        "loaded_model": loaded,
        "generation": generated,
    }
    validate_endpoint_shard(
        value=value,
        control_reuse_path=control_reuse_path,
        plan_path=plan_path,
        qualification_path=qualification_path,
        training_manifest_path=manifest_path,
        terminal_path=terminal_path,
        label=label,
        step=step,
        split=split,
        shard=shard,
    )
    training.publish(output, value)
    return value


def qualification_checkpoint_worker(
    *,
    control_reuse_path: Path,
    plan_path: Path,
    qualification_path: Path,
    training_manifest_path: Path,
    terminal_path: Path,
    output: Path,
    device: str,
) -> dict[str, Any]:
    """Cold-read exactly the already selected four batch-4 qualification images.

    This is an execution qualification for the successor consumer.  It neither
    scores a new endpoint nor changes the frozen comparison cohort.
    """

    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig

    require(device.startswith("cuda") and torch.cuda.is_available(), "CUDA normalized qualification readback")
    require(not output.exists() and not output.is_symlink(), "normalized qualification readback collision")
    checked = validate_static_inputs(
        control_reuse_path=control_reuse_path,
        plan_path=plan_path,
        qualification_path=qualification_path,
        training_manifest_path=training_manifest_path,
        allow_qualification=True,
    )
    manifest_path = checked["training_manifest_path"]
    manifest = checked["training_manifest"]
    require(manifest.get("mode") == "qualification", "qualification checkpoint requires qualification manifest")
    adapter_path, adapter = _checkpoint_adapter(
        manifest_path=manifest_path,
        manifest=manifest,
        terminal_path=terminal_path,
        step=2,
    )
    image_ids = list(checked["plan"]["plan"]["qualification"]["image_ids"])
    require(len(image_ids) == 4 and len(set(image_ids)) == 4, "four frozen qualification images")
    items = _train_items(checked["plan"]["prepared"], image_ids)
    require(len(items) == 4, "one formal qualification batch")
    config_data = predecessor_readback._model_config(manifest)
    config = checkpoint_config(InferConfig.model_validate(config_data), str(adapter_path))
    torch.cuda.set_device(device)
    load_started = time.monotonic()
    qwen, loaded = load_policy(config, device=torch.device(device))
    model_load_seconds = time.monotonic() - load_started
    predecessor_readback._loaded_identity(loaded, adapter_path=adapter_path)
    generated = predecessor_readback._generate(
        qwen,
        config,
        config_data,
        items,
        batch_size=4,
        formal=True,
    )
    value = {
        "schema": f"{SCHEMA}.qualification_checkpoint.v1",
        "status": "completed_batch4_cold_checkpoint_readback",
        "endpoint": dict(QUALIFICATION_ENDPOINT),
        "selection": "the frozen predecessor batch4 qualification image_ids; one successor step2 batch only",
        "qualification_image_ids": image_ids,
        "policy": dict(predecessor_readback.POLICY),
        "batch_size": 4,
        "control_reuse": checked["control_reuse_binding"],
        "plan": training.binding(plan_path),
        "qualification": training.binding(qualification_path),
        "training_manifest": training.binding(manifest_path),
        "training_terminal": training.binding(terminal_path),
        "checkpoint_adapter": adapter,
        "loaded_model": loaded,
        "model_load_seconds": model_load_seconds,
        "generation": generated,
        "producer": training.binding(Path(__file__)),
    }
    validate_qualification_checkpoint(
        value=value,
        control_reuse_path=control_reuse_path,
        plan_path=plan_path,
        qualification_path=qualification_path,
        training_manifest_path=training_manifest_path,
        terminal_path=terminal_path,
    )
    training.publish(output, value)
    return value


def _validate_generation_identity(
    *, generation: Mapping[str, Any], items: Sequence[Mapping[str, Any]]
) -> None:
    """Require a retained shard to stay bound to the exact requested rows."""

    rows = generation.get("rows")
    batches = generation.get("batches")
    require(
        generation.get("status") == "completed"
        and generation.get("configured_batch_size") == 4
        and generation.get("request_count") == len(items)
        and isinstance(rows, list)
        and isinstance(batches, list)
        and len(rows) == len(items)
        and len(items) % 4 == 0
        and len(batches) == len(items) // 4,
        "normalized generation shape",
    )
    require(
        [batch.get("batch_index") for batch in batches] == list(range(len(batches)))
        and all(batch.get("actual_batch_size") == 4 for batch in batches)
        and [batch.get("generated_tokens") for batch in batches]
        == [
            sum(len(row.get("generated_token_ids", [])) for row in rows[index * 4 : (index + 1) * 4])
            for index in range(len(batches))
        ]
        and generation.get("generated_tokens")
        == sum(len(row.get("generated_token_ids", [])) for row in rows),
        "normalized generation batch accounting",
    )
    for index, (row, item) in enumerate(zip(rows, items, strict=True)):
        ids = row.get("generated_token_ids")
        prompt = row.get("prompt_token_ids")
        require(
            row.get("split") == item["split"]
            and row.get("row_index") == item["row_index"]
            and row.get("image_id") == item["image_id"]
            and row.get("example_id") == item["example_id"]
            and row.get("batch_index") == index // 4
            and row.get("actual_batch_size") == 4
            and isinstance(prompt, list)
            and all(type(token) is int and token >= 0 for token in prompt)
            and row.get("prompt_token_ids_sha256") == training.digest(prompt)
            and isinstance(ids, list)
            and all(type(token) is int and token >= 0 for token in ids)
            and row.get("generated_token_ids_sha256") == training.digest(ids)
            and isinstance(row.get("raw_decode_text"), str)
            and isinstance(row.get("executed_media_sha256"), str)
            and len(row["executed_media_sha256"]) == 64
            and isinstance(row.get("observed_image_grid_thw"), list)
            and all(type(value) is int and value > 0 for value in row["observed_image_grid_thw"]),
            "normalized generation row identity",
        )
        predecessor_readback._validate_stop(ids, str(row.get("decode_stop_reason")))
        if item["split"] == "train":
            require(
                prompt == item["expected_prompt_token_ids"]
                and row["executed_media_sha256"] == item["expected_media_sha256"]
                and row["observed_image_grid_thw"] == item["expected_grid"],
                "normalized generation train request identity",
            )


def validate_endpoint_shard(
    *,
    value: Mapping[str, Any],
    control_reuse_path: Path,
    plan_path: Path,
    qualification_path: Path,
    training_manifest_path: Path,
    terminal_path: Path,
    label: str,
    step: int,
    split: str,
    shard: int,
) -> dict[str, Any]:
    """Validate a retained shard before recovery skips it or evaluation scores it."""

    endpoint = _endpoint(label, step)
    checked = validate_static_inputs(
        control_reuse_path=control_reuse_path,
        plan_path=plan_path,
        qualification_path=qualification_path,
        training_manifest_path=training_manifest_path,
    )
    adapter_path, adapter = _adapter(
        manifest_path=checked["training_manifest_path"],
        manifest=checked["training_manifest"],
        terminal_path=terminal_path,
        step=step,
    )
    expected_ids = checked["plan"]["plan"]["endpoint_shards"][split][shard]
    generation = value.get("generation", {})
    rows = generation.get("rows")
    require(
        value.get("schema") == f"{SCHEMA}.endpoint_shard"
        and value.get("status") == "completed_unscored"
        and value.get("endpoint") == endpoint
        and value.get("split") == split
        and value.get("shard") == shard
        and value.get("shard_count") == predecessor_readback.ENDPOINT_SHARDS
        and value.get("policy") == predecessor_readback.POLICY
        and value.get("batch_size") == 4
        and value.get("control_reuse") == checked["control_reuse_binding"]
        and value.get("plan") == training.binding(plan_path)
        and value.get("qualification") == training.binding(qualification_path)
        and value.get("training_manifest") == training.binding(training_manifest_path)
        and value.get("training_terminal") == training.binding(terminal_path)
        and value.get("checkpoint_adapter") == adapter
        and generation.get("configured_batch_size") == 4
        and generation.get("request_count") == len(expected_ids)
        and isinstance(rows, list)
        and [row.get("image_id") for row in rows] == expected_ids
        and all(batch.get("actual_batch_size") == 4 for batch in generation.get("batches", []))
        and all(row.get("actual_batch_size") == 4 for row in rows),
        "B-normalized endpoint shard identity",
    )
    items = (
        _train_items(checked["plan"]["prepared"], expected_ids)
        if split == "train"
        else predecessor_readback._dev_items(checked["plan"]["prepared"], expected_ids)
    )
    _validate_generation_identity(generation=generation, items=items)
    require(adapter_path.is_dir(), "B-normalized adapter root")
    return dict(value)


def validate_qualification_checkpoint(
    *,
    value: Mapping[str, Any],
    control_reuse_path: Path,
    plan_path: Path,
    qualification_path: Path,
    training_manifest_path: Path,
    terminal_path: Path,
) -> dict[str, Any]:
    """Admit the one bounded cold readback before main training is released."""

    checked = validate_static_inputs(
        control_reuse_path=control_reuse_path,
        plan_path=plan_path,
        qualification_path=qualification_path,
        training_manifest_path=training_manifest_path,
        allow_qualification=True,
    )
    manifest_path = checked["training_manifest_path"]
    manifest = checked["training_manifest"]
    require(manifest.get("mode") == "qualification", "qualification receipt manifest mode")
    adapter_path, adapter = _checkpoint_adapter(
        manifest_path=manifest_path,
        manifest=manifest,
        terminal_path=terminal_path,
        step=2,
    )
    image_ids = list(checked["plan"]["plan"]["qualification"]["image_ids"])
    items = _train_items(checked["plan"]["prepared"], image_ids)
    generation = value.get("generation", {})
    require(
        value.get("schema") == f"{SCHEMA}.qualification_checkpoint.v1"
        and value.get("status") == "completed_batch4_cold_checkpoint_readback"
        and value.get("endpoint") == QUALIFICATION_ENDPOINT
        and value.get("selection")
        == "the frozen predecessor batch4 qualification image_ids; one successor step2 batch only"
        and value.get("qualification_image_ids") == image_ids
        and value.get("policy") == predecessor_readback.POLICY
        and value.get("batch_size") == 4
        and value.get("control_reuse") == checked["control_reuse_binding"]
        and value.get("plan") == training.binding(plan_path)
        and value.get("qualification") == training.binding(qualification_path)
        and value.get("training_manifest") == training.binding(manifest_path)
        and value.get("training_terminal") == training.binding(terminal_path)
        and value.get("checkpoint_adapter") == adapter
        and value.get("producer") == training.binding(Path(__file__))
        and isinstance(value.get("model_load_seconds"), float)
        and value["model_load_seconds"] >= 0,
        "B-normalized qualification checkpoint identity",
    )
    predecessor_readback._loaded_identity(value.get("loaded_model", {}), adapter_path=adapter_path)
    _validate_generation_identity(generation=generation, items=items)
    require(adapter_path.is_dir(), "B-normalized qualification adapter root")
    return dict(value)


def endpoint_command(
    *,
    control_reuse_path: Path,
    plan_path: Path,
    qualification_path: Path,
    training_manifest_path: Path,
    terminal_path: Path,
    label: str,
    step: int,
    split: str,
    shard: int,
    output: Path,
    device: str,
) -> list[str]:
    _endpoint(label, step)
    return [
        sys.executable,
        "-m",
        "probes.training_set_completion.source256_normalized_readback",
        "endpoint-worker",
        "--control-reuse",
        str(control_reuse_path),
        "--plan",
        str(plan_path),
        "--qualification",
        str(qualification_path),
        "--training-manifest",
        str(training_manifest_path),
        "--terminal",
        str(terminal_path),
        "--label",
        label,
        "--step",
        str(step),
        "--split",
        split,
        "--shard",
        str(shard),
        "--output",
        str(output),
        "--device",
        device,
    ]


def qualification_checkpoint_command(
    *,
    control_reuse_path: Path,
    plan_path: Path,
    qualification_path: Path,
    training_manifest_path: Path,
    terminal_path: Path,
    output: Path,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "probes.training_set_completion.source256_normalized_readback",
        "qualification-checkpoint-worker",
        "--control-reuse",
        str(control_reuse_path),
        "--plan",
        str(plan_path),
        "--qualification",
        str(qualification_path),
        "--training-manifest",
        str(training_manifest_path),
        "--terminal",
        str(terminal_path),
        "--output",
        str(output),
        "--device",
        device,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("endpoint-worker")
    worker.add_argument("--control-reuse", type=Path, required=True)
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--qualification", type=Path, required=True)
    worker.add_argument("--training-manifest", type=Path, required=True)
    worker.add_argument("--terminal", type=Path, required=True)
    worker.add_argument("--label", choices=tuple(label for label, _ in ENDPOINTS), required=True)
    worker.add_argument("--step", type=int, required=True)
    worker.add_argument("--split", choices=tuple(predecessor_readback.SPLIT_COUNTS), required=True)
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--device", default="cuda:0")
    qualification = sub.add_parser("qualification-checkpoint-worker")
    qualification.add_argument("--control-reuse", type=Path, required=True)
    qualification.add_argument("--plan", type=Path, required=True)
    qualification.add_argument("--qualification", type=Path, required=True)
    qualification.add_argument("--training-manifest", type=Path, required=True)
    qualification.add_argument("--terminal", type=Path, required=True)
    qualification.add_argument("--output", type=Path, required=True)
    qualification.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "endpoint-worker":
        result = endpoint_worker(
            control_reuse_path=args.control_reuse,
            plan_path=args.plan,
            qualification_path=args.qualification,
            training_manifest_path=args.training_manifest,
            terminal_path=args.terminal,
            label=args.label,
            step=args.step,
            split=args.split,
            shard=args.shard,
            output=args.output,
            device=args.device,
        )
    else:
        result = qualification_checkpoint_worker(
            control_reuse_path=args.control_reuse,
            plan_path=args.plan,
            qualification_path=args.qualification,
            training_manifest_path=args.training_manifest,
            terminal_path=args.terminal,
            output=args.output,
            device=args.device,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

"""Batched, resumable natural readback for the COCO227 CE-normalization trial.

Qualification is a mechanical serial-versus-microbatch comparison on the
common source checkpoint.  Scientific endpoint workers use only the selected
qualified batch size and publish one immutable row per image before collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence


REPO = Path("/data/CoordExp/.worktrees/research-probes")
ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-15-coco227-ce-normalization"
)
PRIOR = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-14-training-set-completion-curriculum/dual-start-v3"
)
SOURCE_MANIFEST = PRIOR / "A/training-manifest.json"
SOURCE_TERMINAL = PRIOR / "A/training/terminal.json"
SOURCE_ADAPTER = PRIOR / "A/training/checkpoints/step-00256/adapter"
SOURCE_ROWS = PRIOR / "readback/A/new-step-256/rows"
SOURCE_COLLECTION = PRIOR / "readback-result.json"
QUALIFICATION_ROOT = ROOT / "readback-qualification"
QUALIFICATION_TMUX_SESSION = "coordexp-coco227-readback-qualification"
SCHEMA = "training_set_completion.coco227_readback.v1"
IMAGE_COUNT = 11
CAP = 3_084
EOS = 151_645
ENDPOINT_STEPS = (8, 16, 32, 64, 128, 256)
ARMS = ("S", "T")
QUALIFICATION_CONFIGS = {
    "serial": {"batch_size": 1, "physical_gpu": 4},
    "batch2": {"batch_size": 2, "physical_gpu": 5},
    "batch3": {"batch_size": 3, "physical_gpu": 6},
}
QUALIFICATION_SECONDS = 1_200
ENDPOINT_SECONDS = 7_200
POLICY = {
    "empty_assistant_prefix": True,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "assistant_token_cap": CAP,
    "eos_token_id": EOS,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_file() and not resolved.is_symlink(), f"bound source is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": file_hash(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def verify_binding(value: Mapping[str, Any], name: str) -> Path:
    require(set(value) == {"path", "sha256", "size_bytes"}, f"{name} binding fields")
    path = Path(str(value["path"]))
    require(binding(path) == dict(value), f"{name} binding changed")
    return path


def publish(path: str | Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"publication collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == data, f"publication readback differs: {path}")


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _routes(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    routes = manifest.get("routes")
    require(isinstance(routes, list) and len(routes) == IMAGE_COUNT, "eleven readback routes")
    require(
        len({int(route["image_id"]) for route in routes}) == IMAGE_COUNT,
        "unique readback image IDs",
    )
    for route in routes:
        require(isinstance(route.get("case"), Mapping), "route case")
        require(
            isinstance(route.get("prompt_token_ids"), list)
            and bool(route["prompt_token_ids"])
            and all(type(token) is int and token >= 0 for token in route["prompt_token_ids"]),
            "route prompt token IDs",
        )
        identity = route.get("image_identity")
        require(isinstance(identity, Mapping), "route image identity")
        require(
            isinstance(identity.get("executed_media_sha256"), str)
            and len(identity["executed_media_sha256"]) == 64,
            "route media identity",
        )
        grid = identity.get("observed_image_grid_thw")
        require(
            isinstance(grid, list)
            and len(grid) == 3
            and all(type(item) is int and item > 0 for item in grid),
            "route image grid",
        )
    return [dict(route) for route in routes]


def _validate_model_config(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    config = manifest.get("model_config")
    require(isinstance(config, Mapping), "training manifest model config")
    require(config.get("model", {}).get("dtype") == "fp32", "readback FP32 model")
    require(config.get("backend", {}).get("type") == "hf", "readback HF backend")
    require(
        config.get("backend", {}).get("hf", {}).get("attn_implementation") == "sdpa",
        "readback SDPA attention",
    )
    generation = config.get("generation", {})
    require(
        generation.get("max_new_tokens") == CAP
        and generation.get("temperature") == 0
        and generation.get("top_p") == 1
        and generation.get("repetition_penalty") == 1,
        "readback generation config",
    )
    require(config.get("embedding_delta", {}).get("path"), "bound embedding delta")
    return config


def _route_descriptor(route: Mapping[str, Any]) -> dict[str, Any]:
    identity = route["image_identity"]
    return {
        "image_id": int(route["image_id"]),
        "route_id": str(route["route_id"]),
        "example_id": str(route["example_id"]),
        "image_path": str(route["case"]["image_path"]),
        "prompt_token_ids_sha256": digest(route["prompt_token_ids"]),
        "prompt_token_count": len(route["prompt_token_ids"]),
        "executed_media_sha256": identity["executed_media_sha256"],
        "observed_image_grid_thw": list(identity["observed_image_grid_thw"]),
    }


def _inspect_adapter(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    from src.adapters.dora import inspect_dora_adapter_payload

    config = _validate_model_config(manifest)
    return inspect_dora_adapter_payload(path, config["model"]["base_model"])


def _checkpoint_from_terminal(
    terminal: Mapping[str, Any], *, step: int, adapter: Mapping[str, Any]
) -> None:
    require(terminal.get("status") == "completed", "training terminal status")
    matches = [
        item
        for item in terminal.get("checkpoints", [])
        if int(item.get("step", -1)) == step
    ]
    require(len(matches) == 1 and matches[0].get("adapter") == dict(adapter), "checkpoint adapter receipt")


def old_row_path(image_id: int) -> Path:
    return SOURCE_ROWS / f"image-{image_id:012d}.json"


def prepare_qualification(output: Path = QUALIFICATION_ROOT) -> dict[str, Any]:
    manifest = read(SOURCE_MANIFEST)
    routes = _routes(manifest)
    config = _validate_model_config(manifest)
    adapter = _inspect_adapter(SOURCE_ADAPTER, manifest)
    terminal = read(SOURCE_TERMINAL)
    _checkpoint_from_terminal(terminal, step=256, adapter=adapter)
    collection = read(SOURCE_COLLECTION)
    require(
        collection.get("schema") == "training_set_completion.dual_start.v1.readback_result"
        and collection.get("status") == "completed_unscored",
        "prior collection receipt",
    )
    prior_rows = []
    for route in routes:
        image_id = int(route["image_id"])
        path = old_row_path(image_id)
        row = read(path)
        require(row.get("arm") == "A" and row.get("step") == 256, "prior source provenance")
        require(row.get("adapter") == adapter, "prior source adapter")
        prior_rows.append({"image_id": image_id, "row": binding(path)})
    value = {
        "schema": f"{SCHEMA}.qualification_manifest",
        "status": "candidate_ready",
        "sources": {
            "training_manifest": binding(SOURCE_MANIFEST),
            "training_terminal": binding(SOURCE_TERMINAL),
            "prior_collection": binding(SOURCE_COLLECTION),
            "prior_rows": prior_rows,
            "producer": binding(Path(__file__)),
        },
        "source_endpoint": {
            "original_provenance": {"arm": "A", "new_step": 256},
            "adapter": adapter,
            "model_config_sha256": digest(config),
            "routes": [_route_descriptor(route) for route in routes],
        },
        "policy": dict(POLICY),
        "configurations": {
            name: dict(settings) for name, settings in QUALIFICATION_CONFIGS.items()
        },
        "bounds": {
            "configuration_count": 3,
            "requests_per_configuration": IMAGE_COUNT,
            "maximum_total_image_requests": 36,
            "planned_total_image_requests": 33,
            "wall_seconds_per_configuration": QUALIFICATION_SECONDS,
        },
        "selection": {
            "baseline": "serial",
            "criterion": "fastest completed configuration with detection-level parity to live serial and retained source; otherwise serial",
            "exact_token_identity": "diagnostic",
        },
        "content_sha256": None,
    }
    value["content_sha256"] = digest(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    publish(output / "manifest.json", value)
    return value


def validate_qualification_manifest(value: Mapping[str, Any]) -> tuple[Path, Path]:
    require(value.get("schema") == f"{SCHEMA}.qualification_manifest", "qualification schema")
    require(
        value.get("content_sha256")
        == digest({key: item for key, item in value.items() if key != "content_sha256"}),
        "qualification manifest digest",
    )
    require(value.get("policy") == POLICY, "qualification decode policy")
    require(value.get("configurations") == QUALIFICATION_CONFIGS, "qualification configurations")
    require(value.get("bounds", {}).get("planned_total_image_requests") == 33, "qualification request bound")
    sources = value.get("sources", {})
    manifest_path = verify_binding(sources["training_manifest"], "qualification training manifest")
    terminal_path = verify_binding(sources["training_terminal"], "qualification training terminal")
    verify_binding(sources["prior_collection"], "qualification prior collection")
    verify_binding(sources["producer"], "qualification producer")
    manifest = read(manifest_path)
    routes = _routes(manifest)
    config = _validate_model_config(manifest)
    require(digest(config) == value["source_endpoint"]["model_config_sha256"], "qualification model config")
    require([_route_descriptor(route) for route in routes] == value["source_endpoint"]["routes"], "qualification routes")
    adapter = _inspect_adapter(value["source_endpoint"]["adapter"]["root"], manifest)
    require(adapter == value["source_endpoint"]["adapter"], "qualification adapter")
    _checkpoint_from_terminal(read(terminal_path), step=256, adapter=adapter)
    require(len(sources["prior_rows"]) == IMAGE_COUNT, "qualification prior row count")
    for descriptor in sources["prior_rows"]:
        path = verify_binding(descriptor["row"], "qualification prior row")
        row = read(path)
        require(
            int(row.get("image_id", -1)) == int(descriptor["image_id"])
            and row.get("arm") == "A"
            and row.get("step") == 256
            and row.get("adapter") == adapter,
            "qualification prior row provenance",
        )
    return manifest_path, terminal_path


def batches(items: Sequence[Any], batch_size: int) -> list[list[Any]]:
    require(type(batch_size) is int and batch_size > 0, "positive batch size")
    return [list(items[start : start + batch_size]) for start in range(0, len(items), batch_size)]


def _row_path(root: Path, image_id: int) -> Path:
    return root / "rows" / f"image-{image_id:012d}.json"


def _tokenizer(manifest: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    config = _validate_model_config(manifest)
    return AutoTokenizer.from_pretrained(
        str(config["model"]["base_model"]), local_files_only=True
    )


def _validate_stop(ids: Sequence[int], stop: Any) -> None:
    require(
        (stop == "im_end" and bool(ids) and ids[-1] == EOS and EOS not in ids[:-1])
        or (stop == "length" and len(ids) == CAP and EOS not in ids),
        "per-request EOS/cap contract",
    )


def validate_row(
    row: Mapping[str, Any],
    *,
    route: Mapping[str, Any],
    arm: str,
    step: int,
    batch_size: int,
    adapter: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    training_terminal: Mapping[str, Any],
    tokenizer: Any,
    source_kind: str,
    trial: Mapping[str, Any] | None = None,
) -> None:
    require(row.get("schema") == f"{SCHEMA}.row", "readback row schema")
    require(
        row.get("arm") == arm
        and row.get("checkpoint_step") == step
        and int(row.get("image_id", -1)) == int(route["image_id"]),
        "readback row endpoint identity",
    )
    require(row.get("step") == step, "readback row step alias")
    require(row.get("source_kind") == source_kind, "readback source kind")
    require(row.get("policy") == POLICY, "readback row policy")
    require(
        row.get("empty_assistant_prefix") is True
        and row.get("temperature") == 0.0
        and row.get("top_p") == 1.0
        and row.get("top_k") == 0
        and row.get("repetition_penalty") == 1.0
        and row.get("max_new_tokens") == CAP,
        "readback flat decode policy",
    )
    require(row.get("batch_size") == batch_size, "readback row batch size")
    require(row.get("route_id") == route["route_id"], "readback route identity")
    require(row.get("prompt_token_ids") == route["prompt_token_ids"], "readback prompt IDs")
    require(row.get("prompt_token_ids_sha256") == digest(route["prompt_token_ids"]), "readback prompt digest")
    require(
        row.get("executed_media_sha256")
        == route["image_identity"]["executed_media_sha256"]
        and row.get("observed_image_grid_thw")
        == route["image_identity"]["observed_image_grid_thw"],
        "readback media/grid identity",
    )
    require(row.get("checkpoint_adapter") == dict(adapter), "readback checkpoint adapter")
    require(row.get("adapter") == dict(adapter), "readback adapter alias")
    require(row.get("training_manifest") == dict(training_manifest), "readback training manifest binding")
    require(row.get("training_terminal") == dict(training_terminal), "readback training terminal binding")
    if trial is None:
        require(row.get("trial") is None, "qualification row trial boundary")
    else:
        require(row.get("trial") == dict(trial), "readback trial binding")
    model_receipt_path = verify_binding(row["model_receipt"], "readback model receipt")
    raw_batch_path = verify_binding(row["raw_batch_receipt"], "readback raw batch receipt")
    ids = row.get("generated_token_ids")
    require(
        isinstance(ids, list)
        and bool(ids)
        and len(ids) <= CAP
        and all(type(token) is int and token >= 0 for token in ids),
        "readback generated IDs",
    )
    require(row.get("generated_token_ids_sha256") == digest(ids), "readback generated digest")
    text = row.get("raw_decode_text")
    require(isinstance(text, str), "readback raw decode text")
    require(
        tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        == text,
        "readback token/text identity",
    )
    _validate_stop(ids, row.get("decode_stop_reason"))
    model_receipt = read(model_receipt_path)
    require(
        model_receipt.get("schema") == f"{SCHEMA}.model_receipt"
        and model_receipt.get("checkpoint_adapter") == dict(adapter)
        and model_receipt.get("training_manifest") == dict(training_manifest)
        and model_receipt.get("training_terminal") == dict(training_terminal)
        and model_receipt.get("trial") == (None if trial is None else dict(trial)),
        "readback model receipt identity",
    )
    raw_batch = read(raw_batch_path)
    require(
        raw_batch.get("schema") == f"{SCHEMA}.raw_batch"
        and raw_batch.get("configured_batch_size") == batch_size
        and 0 <= int(row.get("batch_position", -1)) < int(row.get("actual_batch_size", 0))
        <= batch_size
        and raw_batch.get("actual_batch_size") == row.get("actual_batch_size")
        and raw_batch.get("image_ids", [])[row["batch_position"]] == int(route["image_id"]),
        "readback raw batch identity",
    )
    raw_rows = [
        item
        for item in raw_batch.get("rows", [])
        if int(item.get("image_id", -1)) == int(route["image_id"])
    ]
    require(
        len(raw_rows) == 1
        and raw_rows[0].get("generated_token_ids") == ids
        and raw_rows[0].get("generated_token_ids_sha256") == row["generated_token_ids_sha256"]
        and raw_rows[0].get("decode_stop_reason") == row["decode_stop_reason"],
        "readback raw batch row identity",
    )


def pending_routes(
    *,
    row_root: Path,
    routes: Sequence[Mapping[str, Any]],
    arm: str,
    step: int,
    batch_size: int,
    adapter: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    training_terminal: Mapping[str, Any],
    tokenizer: Any,
    source_kind: str,
    trial: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for route in routes:
        path = _row_path(row_root, int(route["image_id"]))
        if path.is_file():
            row = read(path)
            validate_row(
                row,
                route=route,
                arm=arm,
                step=step,
                batch_size=batch_size,
                adapter=adapter,
                training_manifest=training_manifest,
                training_terminal=training_terminal,
                tokenizer=tokenizer,
                source_kind=source_kind,
                trial=trial,
            )
            retained.append(row)
        else:
            require(not path.exists(), "readback row path is not a regular file")
            missing.append(dict(route))
    return retained, missing


def _validate_loaded(
    loaded: Mapping[str, Any], *, adapter: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    from probes.dora_owner_learning.repeat_recovery_train import loaded_composition_evidence
    from src.qwen.special_token_embeddings import inspect_special_token_embedding_delta_payload

    effective = loaded.get("effective_settings", {})
    require(
        effective.get("observed_model_dtype", {}).get("parameter_dtype_names")
        == ["torch.float32"],
        "loaded FP32 identity",
    )
    require(effective.get("observed_attn_implementation") == "sdpa", "loaded SDPA identity")
    model_identity = loaded.get("model_identity", {})
    require(
        Path(model_identity.get("base", {}).get("path", "")).resolve()
        == Path(config["model"]["base_model"]).resolve(),
        "loaded base-model identity",
    )
    loaded_adapter = model_identity.get("adapter", {})
    require(
        loaded_adapter.get("adapter_path") == adapter["root"]
        and loaded_adapter.get("merged_adapters", []) == [],
        "loaded adapter identity",
    )
    require(
        model_identity.get("embedding_delta", {}).get("identity", {}).get("delta_path")
        == config["embedding_delta"]["path"],
        "loaded special-token delta identity",
    )
    inspected_embedding = inspect_special_token_embedding_delta_payload(
        config["embedding_delta"]["path"], config["model"]["base_model"]
    )
    composition = loaded_composition_evidence(
        loaded_identity=loaded,
        expected_base=config["model"]["base_model"],
        expected_adapter=adapter["root"],
        expected_embedding=inspected_embedding,
        inspected_embedding=inspected_embedding,
    )
    require(composition.get("passed") is True, "loaded model composition")
    return {
        "inspected_embedding_delta": inspected_embedding,
        "composition": composition,
    }


def _hooks(model: Any, counters: dict[str, int]) -> list[Any]:
    handles = [
        model.register_forward_pre_hook(
            lambda *_: counters.__setitem__("model_forwards", counters["model_forwards"] + 1)
        )
    ]
    visual = [module for name, module in model.named_modules() if name.endswith("visual")]
    require(len(visual) == 1, "single visual module")
    handles.append(
        visual[0].register_forward_pre_hook(
            lambda *_: counters.__setitem__("image_forwards", counters["image_forwards"] + 1)
        )
    )
    return handles


def _run_worker(
    *,
    manifest_path: Path,
    terminal_path: Path,
    adapter_path: Path,
    row_root: Path,
    arm: str,
    step: int,
    source_kind: str,
    gpu: int,
    batch_size: int,
    attempt: str,
    wall_seconds: int,
    trial_path: Path | None = None,
) -> None:
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from probes.source_rweak_row_cross.run import build_requests
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
        and torch.cuda.device_count() == 1,
        "explicit single-GPU isolation",
    )
    require(attempt and "/" not in attempt, "readback attempt ID")
    manifest = read(manifest_path)
    routes = _routes(manifest)
    config_data = _validate_model_config(manifest)
    adapter = _inspect_adapter(adapter_path, manifest)
    terminal = read(terminal_path)
    _checkpoint_from_terminal(terminal, step=step, adapter=adapter)
    manifest_receipt = binding(manifest_path)
    training_terminal_receipt = binding(terminal_path)
    trial_receipt = binding(trial_path) if trial_path is not None else None
    tokenizer = _tokenizer(manifest)
    retained, missing = pending_routes(
        row_root=row_root,
        routes=routes,
        arm=arm,
        step=step,
        batch_size=batch_size,
        adapter=adapter,
        training_manifest=manifest_receipt,
        training_terminal=training_terminal_receipt,
        tokenizer=tokenizer,
        source_kind=source_kind,
        trial=trial_receipt,
    )
    attempt_root = row_root / "attempts" / attempt
    attempt_terminal = attempt_root / "terminal.json"
    require(not attempt_terminal.exists(), "readback attempt terminal collision")
    result: dict[str, Any] = {
        "schema": f"{SCHEMA}.worker_terminal",
        "status": "running",
        "attempt": attempt,
        "arm": arm,
        "checkpoint_step": step,
        "source_kind": source_kind,
        "physical_gpu": gpu,
        "pid": os.getpid(),
        "batch_size": batch_size,
        "expected_request_count": IMAGE_COUNT,
        "retained_request_count": len(retained),
        "generated_request_count": 0,
        "model_loads": 0,
        "model_forwards": 0,
        "image_forwards": 0,
        "generated_tokens": 0,
        "raw_batch_receipts": [],
    }
    started = time.monotonic()
    old_handler = signal.getsignal(signal.SIGALRM)
    handles: list[Any] = []
    try:
        signal.signal(
            signal.SIGALRM,
            lambda *_: (_ for _ in ()).throw(TimeoutError("readback worker wall")),
        )
        signal.alarm(wall_seconds)
        torch.cuda.set_device("cuda:0")
        if missing:
            torch.cuda.reset_peak_memory_stats()
            load_started = time.monotonic()
            config = checkpoint_config(
                InferConfig.model_validate(config_data), str(adapter_path.resolve(strict=True))
            )
            qwen, loaded = load_policy(config, device=torch.device("cuda:0"))
            result["model_load_seconds"] = time.monotonic() - load_started
            result["model_load_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            result["model_load_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            result["model_loads"] = 1
            composition = _validate_loaded(loaded, adapter=adapter, config=config_data)
            require(qwen.tokenizer.pad_token_id is not None, "tokenizer pad identity")
            qwen.model.eval()
            for parameter in qwen.model.parameters():
                parameter.requires_grad_(False)
            model_receipt_path = attempt_root / "model.json"
            publish(
                model_receipt_path,
                {
                    "schema": f"{SCHEMA}.model_receipt",
                    "status": "loaded_before_generation",
                    "attempt": attempt,
                    "arm": arm,
                    "checkpoint_step": step,
                    "source_kind": source_kind,
                    "model_config_sha256": digest(config_data),
                    "checkpoint_adapter": adapter,
                    **composition,
                    "training_manifest": manifest_receipt,
                    "training_terminal": training_terminal_receipt,
                    "trial": trial_receipt,
                    "loaded": loaded,
                },
            )
            model_receipt = binding(model_receipt_path)
            handles = _hooks(qwen.model, result)
            policy = NativeGenerationPolicy(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                use_model_defaults=False,
            )
            torch.cuda.reset_peak_memory_stats()
            generation_started = time.monotonic()
            for batch_index, route_batch in enumerate(batches(missing, batch_size)):
                requests, _ = build_requests(qwen, config_data, [route["case"] for route in route_batch])
                native = prepare_native_inputs(
                    qwen.processor,
                    requests,
                    device=torch.device("cuda:0"),
                    record_media_identity=True,
                )
                require(
                    [list(ids) for ids in native.prompt_token_ids]
                    == [route["prompt_token_ids"] for route in route_batch],
                    "batched prompt identity",
                )
                require(
                    list(native.media_sha256 or ())
                    == [route["image_identity"]["executed_media_sha256"] for route in route_batch],
                    "batched media identity",
                )
                require(
                    [list(grid) if grid is not None else None for grid in native.image_grids]
                    == [route["image_identity"]["observed_image_grid_thw"] for route in route_batch],
                    "batched image grid identity",
                )
                tick = time.monotonic()
                with torch.inference_mode():
                    generated = generate_continuations(
                        qwen.model,
                        native,
                        extensions=[[] for _ in route_batch],
                        budgets=[CAP for _ in route_batch],
                        eos_token_id=EOS,
                        pad_token_id=qwen.tokenizer.pad_token_id,
                        policy=policy,
                        trace="none",
                        seed=None,
                    )
                elapsed = time.monotonic() - tick
                require(len(generated) == len(route_batch), "generated batch result count")
                raw_rows = []
                for route, generated_row in zip(route_batch, generated, strict=True):
                    require(
                        generated_row.request_id == str(route["case"]["row_id"]),
                        "generated request order",
                    )
                    ids = list(generated_row.token_ids)
                    _validate_stop(ids, generated_row.stop_reason)
                    raw_rows.append(
                        {
                            "image_id": int(route["image_id"]),
                            "request_id": generated_row.request_id,
                            "generated_token_ids": ids,
                            "generated_token_ids_sha256": digest(ids),
                            "decode_stop_reason": generated_row.stop_reason,
                        }
                    )
                raw_batch_path = attempt_root / "batches" / f"batch-{batch_index:03d}.json"
                publish(
                    raw_batch_path,
                    {
                        "schema": f"{SCHEMA}.raw_batch",
                        "status": "generated_before_decode",
                        "attempt": attempt,
                        "batch_index": batch_index,
                        "configured_batch_size": batch_size,
                        "actual_batch_size": len(route_batch),
                        "image_ids": [int(route["image_id"]) for route in route_batch],
                        "generation_seconds": elapsed,
                        "rows": raw_rows,
                    },
                )
                raw_batch = binding(raw_batch_path)
                result["raw_batch_receipts"].append(raw_batch)
                for route, raw in zip(route_batch, raw_rows, strict=True):
                    ids = raw["generated_token_ids"]
                    row = {
                        "schema": f"{SCHEMA}.row",
                        "source_kind": source_kind,
                        "arm": arm,
                        "step": step,
                        "checkpoint_step": step,
                        "route_id": route["route_id"],
                        "example_id": route["example_id"],
                        "image_id": int(route["image_id"]),
                        "request_id": raw["request_id"],
                        "policy": dict(POLICY),
                        "empty_assistant_prefix": True,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 0,
                        "repetition_penalty": 1.0,
                        "max_new_tokens": CAP,
                        "batch_size": batch_size,
                        "batch_index": batch_index,
                        "batch_position": route_batch.index(route),
                        "actual_batch_size": len(route_batch),
                        "prompt_token_ids": route["prompt_token_ids"],
                        "prompt_token_ids_sha256": digest(route["prompt_token_ids"]),
                        "generated_token_ids": ids,
                        "generated_token_ids_sha256": raw["generated_token_ids_sha256"],
                        "raw_decode_text": qwen.tokenizer.decode(
                            ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                        "decode_stop_reason": raw["decode_stop_reason"],
                        "executed_media_sha256": route["image_identity"]["executed_media_sha256"],
                        "observed_image_grid_thw": route["image_identity"]["observed_image_grid_thw"],
                        "checkpoint_adapter": adapter,
                        "adapter": adapter,
                        "model_receipt": model_receipt,
                        "raw_batch_receipt": raw_batch,
                        "training_manifest": manifest_receipt,
                        "training_terminal": training_terminal_receipt,
                        "trial": trial_receipt,
                    }
                    validate_row(
                        row,
                        route=route,
                        arm=arm,
                        step=step,
                        batch_size=batch_size,
                        adapter=adapter,
                        training_manifest=manifest_receipt,
                        training_terminal=training_terminal_receipt,
                        tokenizer=qwen.tokenizer,
                        source_kind=source_kind,
                        trial=trial_receipt,
                    )
                    publish(_row_path(row_root, int(route["image_id"])), row)
                    result["generated_request_count"] += 1
                    result["generated_tokens"] += len(ids)
            result["generation_seconds"] = time.monotonic() - generation_started
            result["generation_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            result["generation_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        else:
            result.update(
                recovered_without_model_load=True,
                model_load_seconds=0.0,
                generation_seconds=0.0,
                model_load_peak_allocated_bytes=0,
                model_load_peak_reserved_bytes=0,
                generation_peak_allocated_bytes=0,
                generation_peak_reserved_bytes=0,
            )
        retained_after, missing_after = pending_routes(
            row_root=row_root,
            routes=routes,
            arm=arm,
            step=step,
            batch_size=batch_size,
            adapter=adapter,
            training_manifest=manifest_receipt,
            training_terminal=training_terminal_receipt,
            tokenizer=tokenizer,
            source_kind=source_kind,
            trial=trial_receipt,
        )
        require(len(retained_after) == IMAGE_COUNT and not missing_after, "readback endpoint completion")
        result.update(status="completed", exit_code=0, completed_request_count=IMAGE_COUNT)
    except BaseException as error:
        result.update(
            status="failed",
            exit_code=1,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        for handle in handles:
            handle.remove()
        result["elapsed_seconds"] = time.monotonic() - started
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        publish(attempt_terminal, result)


def qualification_worker(*, manifest_path: Path, config_name: str, attempt: str) -> None:
    qualification = read(manifest_path)
    source_manifest, source_terminal = validate_qualification_manifest(qualification)
    require(config_name in QUALIFICATION_CONFIGS, "qualification config")
    settings = QUALIFICATION_CONFIGS[config_name]
    _run_worker(
        manifest_path=source_manifest,
        terminal_path=source_terminal,
        adapter_path=Path(qualification["source_endpoint"]["adapter"]["root"]),
        row_root=manifest_path.parent / "configs" / config_name,
        arm="source-A256-live",
        step=256,
        source_kind="qualification_live_common_source",
        gpu=settings["physical_gpu"],
        batch_size=settings["batch_size"],
        attempt=attempt,
        wall_seconds=QUALIFICATION_SECONDS,
        trial_path=None,
    )


def _accepted_batch_size(path: Path) -> int:
    value = read(path)
    require(
        value.get("schema") == f"{SCHEMA}.qualification_result"
        and value.get("status") == "completed",
        "qualification result admission",
    )
    verify_binding(value["manifest"], "qualification result manifest")
    selected = value.get("selection", {}).get("batch_size")
    require(selected in (1, 2, 3), "selected readback batch size")
    return int(selected)


def endpoint_worker(
    *,
    manifest_path: Path,
    terminal_path: Path,
    adapter_path: Path,
    arm: str,
    step: int,
    output: Path,
    gpu: int,
    batch_size: int,
    qualification_result: Path,
    attempt: str,
    trial_path: Path,
) -> None:
    require(arm in ARMS and step in ENDPOINT_STEPS, "scientific endpoint arm/step")
    require(batch_size == _accepted_batch_size(qualification_result), "endpoint batch size differs from qualification")
    _run_worker(
        manifest_path=manifest_path,
        terminal_path=terminal_path,
        adapter_path=adapter_path,
        row_root=output / arm / f"step-{step:05d}",
        arm=arm,
        step=step,
        source_kind="scientific_checkpoint_readback",
        gpu=gpu,
        batch_size=batch_size,
        attempt=attempt,
        wall_seconds=ENDPOINT_SECONDS,
        trial_path=trial_path,
    )


def _golden(route: Mapping[str, Any]) -> dict[str, Any]:
    case = route["case"]
    record = case["input_record"]
    return {
        "example_id": route["example_id"],
        "gt": record["objects"],
        "image_height": case["image_height"],
        "image_path": case["image_path"],
        "image_width": case["image_width"],
        "row_id": case["row_id"],
        "row_index": case["row_index"],
    }


def _parse_row(row: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
    from probes.source_rweak_row_cross.run import native_record

    parsed = native_record(
        row["raw_decode_text"],
        route["case"],
        _golden(route),
        row["decode_stop_reason"],
    )
    predictions = [
        {
            "description": item["description"],
            "coord_bins": list(item["coord_bins"]),
            "generated_order": int(item["generated_order"]),
        }
        for item in parsed.get("pred", [])
    ]
    return {
        "parse_status": parsed.get("parse_status"),
        "metric_bearing": parsed.get("metric_bearing"),
        "valid_prediction_count": len(predictions),
        "dropped_prediction_count": len(parsed.get("dropped_predictions", [])),
        "dropped_predictions": parsed.get("dropped_predictions", []),
        "predictions": predictions,
    }


def detection_consistency(
    reference_row: Mapping[str, Any], candidate_row: Mapping[str, Any], route: Mapping[str, Any]
) -> dict[str, Any]:
    from src.eval.assignment import global_matches

    reference = _parse_row(reference_row, route)
    candidate = _parse_row(candidate_row, route)
    left = [("object", tuple(item["coord_bins"])) for item in reference["predictions"]]
    right = [("object", tuple(item["coord_bins"])) for item in candidate["predictions"]]
    matches = global_matches(left, right, 0.5)
    descriptions_equal = all(
        reference["predictions"][left_index]["description"]
        == candidate["predictions"][right_index]["description"]
        for left_index, right_index, _ in matches
    )
    max_bin_delta = max(
        (
            max(
                abs(a - b)
                for a, b in zip(
                    reference["predictions"][left_index]["coord_bins"],
                    candidate["predictions"][right_index]["coord_bins"],
                    strict=True,
                )
            )
            for left_index, right_index, _ in matches
        ),
        default=0,
    )
    exact_tokens = (
        reference_row.get("generated_token_ids")
        == candidate_row.get("generated_token_ids")
    )
    parity = (
        reference["metric_bearing"] is True
        and candidate["metric_bearing"] is True
        and reference["parse_status"] == candidate["parse_status"]
        and reference["valid_prediction_count"]
        == candidate["valid_prediction_count"]
        == len(matches)
        and reference["dropped_prediction_count"]
        == candidate["dropped_prediction_count"]
        == 0
        and descriptions_equal
        and reference_row.get("decode_stop_reason")
        == candidate_row.get("decode_stop_reason")
        == "im_end"
    )
    return {
        "image_id": int(route["image_id"]),
        "parity": parity,
        "exact_token_identity": exact_tokens,
        "reference_valid_rows": reference["valid_prediction_count"],
        "candidate_valid_rows": candidate["valid_prediction_count"],
        "matched_iou_0_5": len(matches),
        "descriptions_equal": descriptions_equal,
        "reference_dropped_rows": reference["dropped_prediction_count"],
        "candidate_dropped_rows": candidate["dropped_prediction_count"],
        "reference_stop": reference_row.get("decode_stop_reason"),
        "candidate_stop": candidate_row.get("decode_stop_reason"),
        "minimum_iou": min((overlap for _, _, overlap in matches), default=None),
        "maximum_coordinate_bin_delta": max_bin_delta,
    }


def _latest_completed_terminal(root: Path, config_name: str) -> tuple[Path, dict[str, Any]] | None:
    candidates = sorted((root / "configs" / config_name / "attempts").glob("*/terminal.json"))
    completed = [(path, read(path)) for path in candidates if read(path).get("status") == "completed"]
    require(len(completed) <= 1, f"multiple completed qualification attempts: {config_name}")
    return completed[0] if completed else None


def _qualification_generated_request_count(root: Path) -> int:
    count = 0
    for path in sorted(root.glob("attempts/*/batches/*.json")):
        receipt = read(path)
        require(receipt.get("schema") == f"{SCHEMA}.raw_batch", "qualification raw batch schema")
        size = receipt.get("actual_batch_size")
        require(type(size) is int and 0 < size <= 3, "qualification raw batch size")
        count += size
    return count


def _old_rows(qualification: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(item["image_id"]): read(verify_binding(item["row"], "retained source row"))
        for item in qualification["sources"]["prior_rows"]
    }


def _validate_prior_source_row(
    row: Mapping[str, Any],
    *,
    route: Mapping[str, Any],
    adapter: Mapping[str, Any],
    manifest_path: Path,
    tokenizer: Any,
) -> None:
    require(
        row.get("schema") == "training_set_completion.dual_start.v1.readback_row"
        and row.get("arm") == "A"
        and row.get("step") == row.get("checkpoint_step") == 256,
        "prior source row provenance",
    )
    require(
        row.get("empty_assistant_prefix") is True
        and row.get("temperature") == 0
        and row.get("top_p") == 1
        and row.get("top_k") == 0
        and row.get("repetition_penalty") == 1
        and row.get("max_new_tokens") == CAP,
        "prior source decode policy",
    )
    require(
        int(row.get("image_id", -1)) == int(route["image_id"])
        and row.get("route_id") == route["route_id"]
        and row.get("prompt_token_ids") == route["prompt_token_ids"],
        "prior source prompt/route identity",
    )
    require(
        row.get("executed_media_sha256")
        == route["image_identity"]["executed_media_sha256"]
        and row.get("observed_image_grid_thw")
        == route["image_identity"]["observed_image_grid_thw"],
        "prior source media/grid identity",
    )
    require(row.get("adapter") == dict(adapter), "prior source adapter identity")
    require(row.get("training_manifest") == binding(manifest_path), "prior source manifest binding")
    loaded = row.get("loaded_model", {})
    require(
        loaded.get("model_identity", {}).get("adapter", {}).get("adapter_path")
        == adapter["root"]
        and loaded.get("model_identity", {}).get("adapter", {}).get("merged_adapters")
        == []
        and loaded.get("effective_settings", {})
        .get("observed_model_dtype", {})
        .get("parameter_dtype_names")
        == ["torch.float32"]
        and loaded.get("effective_settings", {}).get("observed_attn_implementation")
        == "sdpa",
        "prior source loaded model identity",
    )
    ids = row.get("generated_token_ids")
    require(
        isinstance(ids, list)
        and row.get("generated_token_ids_sha256") == digest(ids),
        "prior source token identity",
    )
    require(
        tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        == row.get("raw_decode_text"),
        "prior source token/text identity",
    )
    _validate_stop(ids, row.get("decode_stop_reason"))


def _evaluation_signature(score: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = score.get("aggregate", {})
    return {
        "ledgers_iou_0_5": aggregate.get("ledgers_iou_0_5"),
        "annotation_relative_micro_scoped227": aggregate.get(
            "annotation_relative_micro_scoped227"
        ),
        "raw_and_physical": aggregate.get("raw_and_physical"),
        "clean_completion": score.get("clean_completion"),
        "per_image": [
            {
                "image_id": row.get("image_id"),
                "raw": row.get("raw"),
                "ledgers_iou_0_5": row.get("ledgers_iou_0_5"),
                "duplicate_candidates_iou_gt_0_95": row.get(
                    "duplicate_candidates_iou_gt_0_95"
                ),
                "physical": row.get("physical"),
                "outside_literal_coco80_protocol_violations": row.get(
                    "outside_literal_coco80_protocol_violations"
                ),
            }
            for row in score.get("per_image", [])
        ],
    }


def collect_qualification(
    manifest_path: Path, *, attempt: str, evaluation_preparation: Path
) -> dict[str, Any]:
    from probes.training_set_completion import coco227_evaluation

    qualification = read(manifest_path)
    source_manifest_path, _ = validate_qualification_manifest(qualification)
    manifest = read(source_manifest_path)
    routes = _routes(manifest)
    route_by_image = {int(route["image_id"]): route for route in routes}
    old = _old_rows(qualification)
    tokenizer = _tokenizer(manifest)
    for image_id, row in old.items():
        _validate_prior_source_row(
            row,
            route=route_by_image[image_id],
            adapter=qualification["source_endpoint"]["adapter"],
            manifest_path=source_manifest_path,
            tokenizer=tokenizer,
        )
    evaluation_preparation = evaluation_preparation.resolve(strict=True)
    coco227_evaluation.validate_preparation(read(evaluation_preparation))
    configs: dict[str, Any] = {}
    live_rows: dict[str, dict[int, dict[str, Any]]] = {}
    for name, settings in QUALIFICATION_CONFIGS.items():
        generated_request_count = _qualification_generated_request_count(
            manifest_path.parent / "configs" / name
        )
        found = _latest_completed_terminal(manifest_path.parent, name)
        if found is None:
            configs[name] = {
                "status": "failed_or_incomplete",
                "batch_size": settings["batch_size"],
                "generated_request_count": generated_request_count,
            }
            continue
        terminal_path, terminal = found
        rows = {
            int(route["image_id"]): read(
                _row_path(manifest_path.parent / "configs" / name, int(route["image_id"]))
            )
            for route in routes
        }
        live_rows[name] = rows
        configs[name] = {
            "status": "completed",
            "batch_size": settings["batch_size"],
            "terminal": binding(terminal_path),
            "timing": {
                key: terminal.get(key)
                for key in (
                    "elapsed_seconds",
                    "model_load_seconds",
                    "generation_seconds",
                )
            },
            "counts": {
                key: terminal.get(key)
                for key in (
                    "model_loads",
                    "model_forwards",
                    "image_forwards",
                    "generated_request_count",
                    "generated_tokens",
                )
            },
            "memory": {
                key: terminal.get(key)
                for key in (
                    "peak_rss_bytes",
                    "model_load_peak_allocated_bytes",
                    "model_load_peak_reserved_bytes",
                    "generation_peak_allocated_bytes",
                    "generation_peak_reserved_bytes",
                )
            },
            "generated_request_count": generated_request_count,
        }
    require(configs["serial"]["status"] == "completed", "live serial qualification failed")
    comparisons: dict[str, Any] = {}
    for name, rows in live_rows.items():
        to_serial = [
            detection_consistency(live_rows["serial"][image_id], rows[image_id], route_by_image[image_id])
            for image_id in sorted(route_by_image)
        ]
        to_retained = [
            detection_consistency(old[image_id], rows[image_id], route_by_image[image_id])
            for image_id in sorted(route_by_image)
        ]
        comparisons[name] = {
            "to_live_serial": to_serial,
            "to_retained_source": to_retained,
            "detection_parity": all(row["parity"] for row in to_serial + to_retained),
            "exact_tokens_to_serial": sum(row["exact_token_identity"] for row in to_serial),
            "exact_tokens_to_retained": sum(row["exact_token_identity"] for row in to_retained),
        }
        configs[name]["correct"] = comparisons[name]["detection_parity"]
    retained_admission, retained_rows = coco227_evaluation.admit_source0()
    retained_score = coco227_evaluation.score_admitted_rows(
        preparation_path=evaluation_preparation,
        label="qualification-retained-prior-source",
        rows=retained_rows,
        readback_admission=retained_admission,
    )
    retained_score_path = manifest_path.parent / "evaluation" / "retained-source0.json"
    publish(retained_score_path, retained_score)
    retained_signature = _evaluation_signature(retained_score)
    serial_signature: dict[str, Any] | None = None
    for name, rows in live_rows.items():
        admission = {
            "schema": f"{SCHEMA}.qualification_readback_admission",
            "status": "mechanically_validated_for_qualification_scoring",
            "arm": "qualification",
            "step": 0,
            "source_kind": "qualification_live_common_source",
            "configuration": name,
            "batch_size": QUALIFICATION_CONFIGS[name]["batch_size"],
            "qualification_manifest": binding(manifest_path),
            "checkpoint_adapter": qualification["source_endpoint"]["adapter"],
            "row_bindings": [
                binding(
                    _row_path(
                        manifest_path.parent / "configs" / name,
                        image_id,
                    )
                )
                for image_id in sorted(rows)
            ],
        }
        score = coco227_evaluation.score_qualification_rows(
            preparation_path=evaluation_preparation,
            label=f"qualification-live-{name}",
            rows=[rows[image_id] for image_id in sorted(rows)],
            readback_admission=admission,
        )
        score_path = manifest_path.parent / "evaluation" / f"{name}.json"
        publish(score_path, score)
        signature = _evaluation_signature(score)
        if name == "serial":
            serial_signature = signature
        configs[name]["evaluation"] = binding(score_path)
        configs[name]["owner_error_parity_to_retained"] = (
            signature == retained_signature
        )
    require(serial_signature is not None, "serial evaluation signature")
    for name in live_rows:
        signature = _evaluation_signature(read(configs[name]["evaluation"]["path"]))
        configs[name]["owner_error_parity_to_serial"] = signature == serial_signature
        configs[name]["correct"] = (
            configs[name]["correct"]
            and configs[name]["owner_error_parity_to_retained"]
            and configs[name]["owner_error_parity_to_serial"]
        )
    serial_seconds = configs["serial"]["timing"]["generation_seconds"]
    eligible = [
        name
        for name in QUALIFICATION_CONFIGS
        if configs[name].get("correct")
        and isinstance(configs[name]["timing"].get("generation_seconds"), (int, float))
        and configs[name]["timing"]["generation_seconds"]
        <= serial_seconds
    ]
    selected_name = min(
        eligible,
        key=lambda name: configs[name]["timing"]["generation_seconds"],
        default="serial",
    )
    value = {
        "schema": f"{SCHEMA}.qualification_result",
        "status": "completed",
        "manifest": binding(manifest_path),
        "attempt": attempt,
        "configs": configs,
        "comparisons": comparisons,
        "evaluation": {
            "preparation": binding(evaluation_preparation),
            "retained_source0": binding(retained_score_path),
        },
        "selection": {
            "name": selected_name,
            "batch_size": QUALIFICATION_CONFIGS[selected_name]["batch_size"],
            "generation_seconds": configs[selected_name]["timing"]["generation_seconds"],
            "serial_generation_seconds": serial_seconds,
            "criterion": qualification["selection"]["criterion"],
        },
        "request_count": sum(
            int(config["generated_request_count"]) for config in configs.values()
        ),
    }
    require(value["request_count"] <= 36, "qualification image request bound")
    result_path = manifest_path.parent / "results" / f"{attempt}.json"
    publish(result_path, value)
    final_path = manifest_path.parent / "result.json"
    if not final_path.exists():
        publish(final_path, value)
    else:
        require(read(final_path) == value, "qualification final result collision")
    return value


def _qualification_controller_inner(
    manifest_path: Path, *, attempt: str, evaluation_preparation: Path
) -> dict[str, Any]:
    qualification = read(manifest_path)
    validate_qualification_manifest(qualification)
    launch_path = manifest_path.parent / "controller" / f"{attempt}-launch.json"
    commands = []
    processes = []
    for name, settings in QUALIFICATION_CONFIGS.items():
        command = [
            "python",
            "-m",
            "probes.training_set_completion.coco227_readback",
            "qualification-worker",
            "--qualification-manifest",
            str(manifest_path),
            "--config-name",
            name,
            "--attempt",
            attempt,
        ]
        log = manifest_path.parent / "logs" / f"{attempt}-{name}.log"
        require(not log.exists(), f"qualification log collision: {log}")
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("x")
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(settings["physical_gpu"]),
                "OMP_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false",
            },
            start_new_session=True,
        )
        commands.append(
            {
                "name": name,
                "batch_size": settings["batch_size"],
                "physical_gpu": settings["physical_gpu"],
                "command": command,
                "pid": process.pid,
                "log": str(log),
            }
        )
        processes.append((process, stream, name, time.monotonic()))
    publish(
        launch_path,
        {
            "schema": f"{SCHEMA}.qualification_launch",
            "status": "spawned",
            "attempt": attempt,
            "manifest": binding(manifest_path),
            "tmux_session": os.environ.get("TMUX"),
            "commands": commands,
        },
    )
    exits = []
    for process, stream, name, spawned in processes:
        deadline = spawned + QUALIFICATION_SECONDS + 30
        try:
            exit_code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait(timeout=30)
        finally:
            stream.close()
        exits.append({"name": name, "pid": process.pid, "exit_code": exit_code})
    publish(
        manifest_path.parent / "controller" / f"{attempt}-exits.json",
        {
            "schema": f"{SCHEMA}.qualification_exits",
            "attempt": attempt,
            "exits": exits,
        },
    )
    return collect_qualification(
        manifest_path,
        attempt=attempt,
        evaluation_preparation=evaluation_preparation,
    )


def qualification_controller(
    manifest_path: Path, *, attempt: str, evaluation_preparation: Path
) -> None:
    terminal_path = manifest_path.parent / "controller" / f"{attempt}-terminal.json"
    require(not terminal_path.exists(), "qualification controller terminal collision")
    started = time.monotonic()
    value: dict[str, Any] = {
        "schema": f"{SCHEMA}.qualification_controller_terminal",
        "status": "running",
        "attempt": attempt,
        "pid": os.getpid(),
        "manifest": binding(manifest_path),
        "evaluation_preparation": binding(evaluation_preparation),
    }
    try:
        tmux_session = subprocess.check_output(
            ["tmux", "display-message", "-p", "#S"], text=True
        ).strip()
        require(
            tmux_session == QUALIFICATION_TMUX_SESSION,
            "qualification controller must run in its named tmux session",
        )
        value["tmux_session"] = tmux_session
        result = _qualification_controller_inner(
            manifest_path,
            attempt=attempt,
            evaluation_preparation=evaluation_preparation,
        )
        value.update(
            status="completed", exit_code=0, result=binding(manifest_path.parent / "result.json")
        )
        require(result == read(value["result"]["path"]), "qualification result publication")
    except BaseException as error:
        value.update(
            status="failed",
            exit_code=1,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        value["elapsed_seconds"] = time.monotonic() - started
        publish(terminal_path, value)


def collect_endpoint(
    *,
    manifest_path: Path,
    terminal_path: Path,
    adapter_path: Path,
    arm: str,
    step: int,
    output: Path,
    qualification_result: Path,
    trial_path: Path,
    teacher_bank_path: Path,
) -> dict[str, Any]:
    from probes.training_set_completion import coco227_evaluation

    require(arm in ARMS and step in ENDPOINT_STEPS, "scientific endpoint arm/step")
    batch_size = _accepted_batch_size(qualification_result)
    manifest = read(manifest_path)
    routes = _routes(manifest)
    adapter = _inspect_adapter(adapter_path, manifest)
    terminal = read(terminal_path)
    _checkpoint_from_terminal(terminal, step=step, adapter=adapter)
    tokenizer = _tokenizer(manifest)
    endpoint_root = output / arm / f"step-{step:05d}"
    manifest_receipt = binding(manifest_path)
    terminal_receipt = binding(terminal_path)
    trial_receipt = binding(trial_path)
    rows = []
    for route in routes:
        row_path = _row_path(endpoint_root, int(route["image_id"]))
        row = read(row_path)
        validate_row(
            row,
            route=route,
            arm=arm,
            step=step,
            batch_size=batch_size,
            adapter=adapter,
            training_manifest=manifest_receipt,
            training_terminal=terminal_receipt,
            tokenizer=tokenizer,
            source_kind="scientific_checkpoint_readback",
            trial=trial_receipt,
        )
        rows.append(row)
    value = {
        "schema": f"{SCHEMA}.endpoint",
        "status": "completed_unscored",
        "source_kind": "scientific_checkpoint_readback",
        "arm": arm,
        "checkpoint_step": step,
        "policy": dict(POLICY),
        "batch_size": batch_size,
        "qualification_result": binding(qualification_result),
        "training_manifest": manifest_receipt,
        "training_terminal": terminal_receipt,
        "trial": trial_receipt,
        "checkpoint_adapter": adapter,
        "row_bindings": [binding(_row_path(endpoint_root, int(row["image_id"]))) for row in rows],
        "rows": rows,
    }
    endpoint_path = endpoint_root / "endpoint.json"
    if endpoint_path.is_file():
        require(read(endpoint_path) == value, "endpoint collection changed")
    else:
        require(not endpoint_path.exists(), "endpoint collection path")
        publish(endpoint_path, value)
    admission = {
        "schema": coco227_evaluation.READBACK_ADMISSION_SCHEMA,
        "status": "admitted_natural_readback",
        "arm": arm,
        "step": step,
        "source_kind": "scientific_checkpoint_readback",
        "rows": binding(endpoint_path),
        "teacher_bank": binding(teacher_bank_path),
        "trial": trial_receipt,
        "training_manifest": manifest_receipt,
        "training_terminal": terminal_receipt,
        "adapter": adapter,
        "qualification_result": binding(qualification_result),
        "producer": binding(Path(__file__)),
    }
    admission_path = endpoint_root / "admission.json"
    if admission_path.is_file():
        require(read(admission_path) == admission, "endpoint admission changed")
    else:
        require(not admission_path.exists(), "endpoint admission path")
        publish(admission_path, admission)
    return {"endpoint": value, "admission": admission}


def admit_source0(
    *, output: Path = ROOT / "readback", qualification_manifest: Path = QUALIFICATION_ROOT / "manifest.json"
) -> dict[str, Any]:
    from probes.training_set_completion import coco227_evaluation

    qualification = read(qualification_manifest)
    source_manifest_path, _ = validate_qualification_manifest(qualification)
    manifest = read(source_manifest_path)
    routes = _routes(manifest)
    tokenizer = _tokenizer(manifest)
    checked_rows = []
    for route in routes:
        image_id = int(route["image_id"])
        path = old_row_path(image_id)
        row = read(path)
        ids = row.get("generated_token_ids")
        _validate_prior_source_row(
            row,
            route=route,
            adapter=qualification["source_endpoint"]["adapter"],
            manifest_path=source_manifest_path,
            tokenizer=tokenizer,
        )
        checked_rows.append(row)
    value, admitted_rows = coco227_evaluation.admit_source0(
        runtime_result_path=SOURCE_COLLECTION
    )
    require(admitted_rows == checked_rows, "source0 evaluator/readback admission rows")
    path = output / "source0-admission.json"
    if path.is_file():
        require(read(path) == value, "source0 admission changed")
    else:
        require(not path.exists(), "source0 admission path")
        publish(path, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare-qualification",
            "qualification-worker",
            "qualification-controller",
            "collect-qualification",
            "endpoint-worker",
            "collect-endpoint",
            "admit-source0",
        ),
    )
    parser.add_argument("--qualification-manifest", type=Path, default=QUALIFICATION_ROOT / "manifest.json")
    parser.add_argument("--qualification-result", type=Path, default=QUALIFICATION_ROOT / "result.json")
    parser.add_argument(
        "--evaluation-preparation",
        type=Path,
        default=ROOT / "evaluation-v1/preparation-v3.json",
    )
    parser.add_argument("--config-name", choices=tuple(QUALIFICATION_CONFIGS))
    parser.add_argument("--attempt", default="attempt-001")
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--training-terminal", type=Path)
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--teacher-bank", type=Path, default=ROOT / "data-v1/bank.json")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--step", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.command == "prepare-qualification":
        print(json.dumps(prepare_qualification(args.qualification_manifest.parent)))
    elif args.command == "qualification-worker":
        require(args.config_name is not None, "qualification worker config")
        qualification_worker(
            manifest_path=args.qualification_manifest,
            config_name=args.config_name,
            attempt=args.attempt,
        )
    elif args.command == "qualification-controller":
        qualification_controller(
            args.qualification_manifest,
            attempt=args.attempt,
            evaluation_preparation=args.evaluation_preparation,
        )
    elif args.command == "collect-qualification":
        print(
            json.dumps(
                collect_qualification(
                    args.qualification_manifest,
                    attempt=args.attempt,
                    evaluation_preparation=args.evaluation_preparation,
                )
            )
        )
    elif args.command == "admit-source0":
        print(
            json.dumps(
                admit_source0(
                    output=args.output or ROOT / "readback",
                    qualification_manifest=args.qualification_manifest,
                )
            )
        )
    else:
        require(
            args.training_manifest is not None
            and args.training_terminal is not None
            and args.adapter is not None
            and args.arm is not None
            and args.step is not None
            and args.output is not None,
            "scientific endpoint arguments",
        )
        require(args.trial is not None, "scientific endpoint trial")
        if args.command == "endpoint-worker":
            require(args.gpu is not None and args.batch_size is not None, "endpoint worker GPU/batch")
            endpoint_worker(
                manifest_path=args.training_manifest,
                terminal_path=args.training_terminal,
                adapter_path=args.adapter,
                arm=args.arm,
                step=args.step,
                output=args.output,
                gpu=args.gpu,
                batch_size=args.batch_size,
                qualification_result=args.qualification_result,
                attempt=args.attempt,
                trial_path=args.trial,
            )
        else:
            print(
                json.dumps(
                    collect_endpoint(
                        manifest_path=args.training_manifest,
                        terminal_path=args.training_terminal,
                        adapter_path=args.adapter,
                        arm=args.arm,
                        step=args.step,
                        output=args.output,
                        qualification_result=args.qualification_result,
                        trial_path=args.trial,
                        teacher_bank_path=args.teacher_bank,
                    )
                )
            )


if __name__ == "__main__":
    main()

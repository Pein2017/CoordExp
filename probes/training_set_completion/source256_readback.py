"""Bounded native readback for the Source256 paired trial.

The qualification probe uses four singleton requests only as a small parity
reference.  Formal qualification and endpoint reads require batch >= 4.  The
full train256/dev128 workers consume one fixed eighth of a split and never keep
prepared image tensors beyond the current generation batch.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import time
from typing import Any, Mapping, Sequence

from probes.training_set_completion import source256_training as runtime
from probes.training_set_completion import training


SCHEMA = "training_set_completion.source256_readback.v1"
CAP = 3_084
EOS = 151_645
MIN_FORMAL_BATCH = 4
QUALIFICATION_CANDIDATES = (4,)
QUALIFICATION_IMAGE_COUNT = 4
SERIAL_REFERENCE_COUNT = 4
ENDPOINT_SHARDS = 8
SPLIT_COUNTS = {"train": 256, "dev": 128}
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


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _verify_binding(value: Mapping[str, Any], name: str) -> Path:
    require(set(value) == {"path", "sha256", "size_bytes"}, f"{name} binding fields")
    path = Path(str(value["path"])).resolve(strict=True)
    require(training.binding(path) == dict(value), f"{name} binding changed")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    require(all(isinstance(row, dict) for row in rows), "JSONL object rows")
    return rows


def _image_ids(rows: Sequence[Mapping[str, Any]], *, expected: int, name: str) -> list[int]:
    ids = [row.get("image_id") for row in rows]
    require(
        len(ids) == expected
        and len(set(ids)) == expected
        and all(type(image_id) is int and image_id >= 0 for image_id in ids),
        f"{name} image cohort",
    )
    return list(ids)


def qualification_image_ids(prepared: Mapping[str, Any]) -> list[int]:
    """Choose deterministic high-prefill cases for the bounded batch probe."""

    ranked = sorted(
        prepared["records"],
        key=lambda record: (
            math.prod(
                record["canonical_route"]["image_identity"]["observed_image_grid_thw"]
            ),
            len(record["canonical_route"]["prompt_token_ids"]),
            int(record["image_id"]),
        ),
        reverse=True,
    )
    ids = [int(record["image_id"]) for record in ranked[:QUALIFICATION_IMAGE_COUNT]]
    require(len(ids) == len(set(ids)) == QUALIFICATION_IMAGE_COUNT, "qualification cases")
    return ids


def _round_robin_shards(ids: Sequence[int], count: int = ENDPOINT_SHARDS) -> list[list[int]]:
    require(type(count) is int and count > 0 and len(ids) % count == 0, "even shard count")
    shards = [list(ids[index::count]) for index in range(count)]
    require(
        sorted(image_id for shard in shards for image_id in shard) == sorted(ids)
        and len({len(shard) for shard in shards}) == 1,
        "complete balanced shards",
    )
    return shards


def build_plan(*, preparation_path: Path, output: Path) -> dict[str, Any]:
    """Freeze CPU-only batching/sharding identity; this never launches a model."""

    require(not output.exists(), f"readback plan collision: {output}")
    prepared = runtime.validate_preparation(read(preparation_path))
    raw = prepared["preparation"]
    input_manifest_path = _verify_binding(raw["sources"]["input_manifest"], "input manifest")
    input_manifest = read(input_manifest_path)
    train_path = _verify_binding(raw["sources"]["train_jsonl"], "train JSONL")
    dev_path = _verify_binding(raw["sources"]["dev_jsonl"], "dev JSONL")
    cohorts = {
        "train": _image_ids(_read_jsonl(train_path), expected=256, name="train"),
        "dev": _image_ids(_read_jsonl(dev_path), expected=128, name="dev"),
    }
    require(input_manifest.get("cohort_ids") == cohorts, "input manifest cohort identity")
    require(
        cohorts["train"] == [record["image_id"] for record in prepared["records"]],
        "preparation/train order",
    )
    value: dict[str, Any] = {
        "schema": f"{SCHEMA}.plan",
        "status": "candidate_ready_for_batch_qualification",
        "preparation": training.binding(preparation_path),
        "sources": {
            "input_manifest": training.binding(input_manifest_path),
            "train_jsonl": training.binding(train_path),
            "dev_jsonl": training.binding(dev_path),
            "producer": training.binding(Path(__file__)),
        },
        "policy": dict(POLICY),
        "qualification": {
            "image_ids": qualification_image_ids(prepared),
            "singleton_reference_count": SERIAL_REFERENCE_COUNT,
            "formal_candidates": list(QUALIFICATION_CANDIDATES),
            "minimum_formal_batch": MIN_FORMAL_BATCH,
            "selection": "user-frozen batch4 with exact token/stop parity; failure blocks",
        },
        "endpoint_shards": {
            split: _round_robin_shards(ids) for split, ids in cohorts.items()
        },
        "bounds": {
            "qualification_requests": SERIAL_REFERENCE_COUNT
            + QUALIFICATION_IMAGE_COUNT * len(QUALIFICATION_CANDIDATES),
            "endpoint_shard_count": ENDPOINT_SHARDS,
            "train_images_per_shard": 32,
            "dev_images_per_shard": 16,
            "full_endpoint_images": 384,
            "materialization": "one generation batch at a time; no full-cohort image tensors",
        },
        "content_sha256": None,
    }
    value["content_sha256"] = training.digest(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    training.publish(output, value)
    return value


def validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    require(value.get("schema") == f"{SCHEMA}.plan", "readback plan schema")
    require(
        value.get("content_sha256")
        == training.digest({key: item for key, item in value.items() if key != "content_sha256"}),
        "readback plan digest",
    )
    require(value.get("policy") == POLICY, "readback policy")
    preparation_path = _verify_binding(value["preparation"], "preparation")
    prepared = runtime.validate_preparation(read(preparation_path))
    for name, source in value["sources"].items():
        _verify_binding(source, name)
    qualification = value.get("qualification", {})
    require(
        qualification.get("image_ids") == qualification_image_ids(prepared)
        and qualification.get("singleton_reference_count") == SERIAL_REFERENCE_COUNT
        and qualification.get("formal_candidates") == list(QUALIFICATION_CANDIDATES)
        and qualification.get("minimum_formal_batch") == MIN_FORMAL_BATCH,
        "readback qualification contract",
    )
    input_manifest = read(value["sources"]["input_manifest"]["path"])
    cohorts = input_manifest["cohort_ids"]
    require(
        value.get("endpoint_shards")
        == {split: _round_robin_shards(cohorts[split]) for split in SPLIT_COUNTS},
        "readback endpoint shards",
    )
    return {"plan": dict(value), "prepared": prepared, "cohorts": cohorts}


def select_qualified_batch(configurations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the fail-closed batch floor to measured qualification receipts."""

    serial = configurations.get("serial")
    batch4 = configurations.get("batch4")
    require(
        isinstance(serial, Mapping)
        and serial.get("status") == "completed"
        and serial.get("configured_batch_size") == 1
        and serial.get("request_count") == SERIAL_REFERENCE_COUNT,
        "tiny singleton reference",
    )
    if not (
        isinstance(batch4, Mapping)
        and batch4.get("status") == "completed"
        and batch4.get("configured_batch_size") == 4
        and batch4.get("request_count") == QUALIFICATION_IMAGE_COUNT
        and batch4.get("exact_reference_parity") is True
    ):
        return {
            "status": "blocked_batch_below_4",
            "selected_batch_size": None,
            "reason": "batch4 did not complete with exact singleton token/stop parity",
        }
    return {
        "status": "qualified",
        "selected_batch_size": 4,
        "selection_scope": list(QUALIFICATION_CANDIDATES),
        "criterion": "user-frozen batch4 after exact singleton parity",
    }


def _chunks(items: Sequence[Any], batch_size: int, *, formal: bool) -> list[list[Any]]:
    require(type(batch_size) is int and batch_size > 0, "positive batch size")
    if formal:
        require(batch_size >= MIN_FORMAL_BATCH, "formal readback batch must be >=4")
    chunks = [list(items[start : start + batch_size]) for start in range(0, len(items), batch_size)]
    require(bool(chunks), "nonempty readback work")
    if formal:
        require(all(len(chunk) >= MIN_FORMAL_BATCH for chunk in chunks), "formal tail batch below4")
    return chunks


def _model_config(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    config = manifest["model_config"]
    generation = config.get("generation", {})
    require(
        config.get("model", {}).get("dtype") == "fp32"
        and config.get("backend", {}).get("type") == "hf"
        and config.get("backend", {}).get("hf", {}).get("attn_implementation") == "sdpa"
        and generation.get("max_new_tokens") == CAP
        and generation.get("temperature") == 0
        and generation.get("top_p") == 1
        and generation.get("repetition_penalty") == 1,
        "Source256 native readback config",
    )
    return config


def _adapter(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    step: int,
    terminal_path: Path | None,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any] | None]:
    base = manifest["model_config"]["model"]["base_model"]
    if step == 0:
        require(terminal_path is None, "Source0 has no training terminal")
        path = Path(manifest["source_adapter"]["root"]).resolve(strict=True)
        observed = training.inspect_dora_adapter_payload(path, base)
        require(observed == manifest["source_adapter"], "Source0 adapter identity")
        return path, observed, None
    require(step in (2, 16, 64) and terminal_path is not None, "qualified checkpoint step")
    terminal = read(terminal_path)
    require(
        terminal.get("status") == "completed"
        and terminal.get("manifest") == training.binding(manifest_path),
        "training terminal identity",
    )
    matches = [item for item in terminal.get("checkpoints", []) if item.get("step") == step]
    require(len(matches) == 1, "one requested checkpoint")
    path = Path(matches[0]["root"]) / "adapter"
    observed = training.inspect_dora_adapter_payload(path, base)
    require(observed == matches[0]["adapter"], "checkpoint adapter receipt")
    return path.resolve(strict=True), observed, terminal


def _train_items(prepared: Mapping[str, Any], image_ids: Sequence[int]) -> list[dict[str, Any]]:
    records = runtime.hydrate_bound_cases(prepared)
    by_image = {int(record["image_id"]): record for record in records}
    require(len(by_image) == 256 and set(image_ids) <= set(by_image), "train item IDs")
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


def _dev_items(prepared: Mapping[str, Any], image_ids: Sequence[int]) -> list[dict[str, Any]]:
    from src.data import load_raw_examples

    path = Path(prepared["preparation"]["sources"]["dev_jsonl"]["path"])
    raws = load_raw_examples(path)
    rows = _read_jsonl(path)
    require(len(raws) == len(rows) == 128, "dev128 rows")
    by_image = {
        int(row["image_id"]): (index, raw)
        for index, (row, raw) in enumerate(zip(rows, raws, strict=True))
    }
    require(len(by_image) == 128 and set(image_ids) <= set(by_image), "dev item IDs")
    return [
        {
            "split": "dev",
            "row_index": by_image[image_id][0],
            "image_id": image_id,
            "example_id": str(by_image[image_id][1].example_id),
            "raw": by_image[image_id][1],
        }
        for image_id in image_ids
    ]


def _requests(qwen: Any, config: Any, config_data: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> tuple[list[Any], list[dict[str, Any]]]:
    from probes.dora_owner_learning.runtime import build_request
    from src.inference.bound_requests import build_bound_native_requests

    if all(item["split"] == "train" for item in items):
        requests, _ = build_bound_native_requests(qwen, config_data, [item["case"] for item in items])
        expected = [
            {
                "prompt": item["expected_prompt_token_ids"],
                "media": item["expected_media_sha256"],
                "grid": item["expected_grid"],
            }
            for item in items
        ]
        return requests, expected
    require(all(item["split"] == "dev" for item in items), "one split per native batch")
    requests, expected = [], []
    for item in items:
        request, image, prompt = build_request(
            item["raw"], config=config, qwen=qwen, row_index=item["row_index"]
        )
        image_record = image.to_artifact_dict()
        requests.append(request)
        expected.append(
            {
                "prompt": list(prompt.expected_executed_prompt_token_ids),
                # Dev has no retained executed-media receipt.  The bound input
                # bytes and NativeRequest constrain the projection; capture the
                # live executed identity below.
                "media": None,
                "grid": image_record["expected_image_grid_thw"],
            }
        )
    return requests, expected


def _validate_stop(ids: Sequence[int], stop: str) -> None:
    require(
        (stop == "im_end" and bool(ids) and ids[-1] == EOS and EOS not in ids[:-1])
        or (stop == "length" and len(ids) == CAP and EOS not in ids),
        "native EOS/cap stop",
    )


def _generate(
    qwen: Any,
    config: Any,
    config_data: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    formal: bool,
) -> dict[str, Any]:
    import torch
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    policy = NativeGenerationPolicy(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        use_model_defaults=False,
    )
    rows: list[dict[str, Any]] = []
    batch_receipts = []
    started = time.monotonic()
    generation_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    for batch_index, selected in enumerate(_chunks(items, batch_size, formal=formal)):
        requests, expected = _requests(qwen, config, config_data, selected)
        native = prepare_native_inputs(
            qwen.processor,
            requests,
            device=next(qwen.model.parameters()).device,
            record_media_identity=True,
        )
        require(
            [list(ids) for ids in native.prompt_token_ids]
            == [item["prompt"] for item in expected],
            "native prompt identity",
        )
        observed_media = list(native.media_sha256 or ())
        require(
            len(observed_media) == len(expected)
            and all(
                item["media"] is None or item["media"] == observed
                for item, observed in zip(expected, observed_media, strict=True)
            )
            and all(isinstance(observed, str) and len(observed) == 64 for observed in observed_media),
            "native media identity",
        )
        observed_grids = [
            list(grid) if grid is not None else None for grid in native.image_grids
        ]
        require(
            observed_grids == [item["grid"] for item in expected],
            "native image-grid identity",
        )
        tick = time.monotonic()
        generated = generate_continuations(
            qwen.model,
            native,
            extensions=[[] for _ in selected],
            budgets=[CAP for _ in selected],
            eos_token_id=EOS,
            pad_token_id=qwen.tokenizer.pad_token_id,
            policy=policy,
            trace="none",
            seed=None,
        )
        elapsed = time.monotonic() - tick
        generation_seconds += elapsed
        require(len(generated) == len(selected), "native generated count")
        token_count = 0
        for position, (item, expected_item, result) in enumerate(
            zip(selected, expected, generated, strict=True)
        ):
            require(result.request_id == item["example_id"], "native request order")
            ids = list(result.token_ids)
            _validate_stop(ids, result.stop_reason)
            token_count += len(ids)
            rows.append(
                {
                    "split": item["split"],
                    "row_index": item["row_index"],
                    "image_id": item["image_id"],
                    "example_id": item["example_id"],
                    "prompt_token_ids": expected_item["prompt"],
                    "prompt_token_ids_sha256": training.digest(expected_item["prompt"]),
                    "generated_token_ids": ids,
                    "generated_token_ids_sha256": training.digest(ids),
                    "decode_stop_reason": result.stop_reason,
                    "raw_decode_text": qwen.tokenizer.decode(
                        ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "executed_media_sha256": observed_media[position],
                    "observed_image_grid_thw": observed_grids[position],
                    "batch_index": batch_index,
                    "actual_batch_size": len(selected),
                }
            )
        batch_receipts.append(
            {
                "batch_index": batch_index,
                "actual_batch_size": len(selected),
                "generation_seconds": elapsed,
                "generated_tokens": token_count,
            }
        )
    elapsed = time.monotonic() - started
    generated_tokens = sum(len(row["generated_token_ids"]) for row in rows)
    return {
        "status": "completed",
        "configured_batch_size": batch_size,
        "request_count": len(rows),
        "elapsed_seconds": elapsed,
        "generation_seconds": generation_seconds,
        "images_per_second": len(rows) / elapsed,
        "generated_tokens_per_second": generated_tokens / generation_seconds,
        "generated_tokens": generated_tokens,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "batches": batch_receipts,
        "rows": rows,
    }


def _loaded_identity(loaded: Mapping[str, Any], *, adapter_path: Path) -> None:
    require(
        loaded.get("effective_settings", {}).get("observed_model_dtype", {}).get("parameter_dtype_names")
        == ["torch.float32"]
        and loaded.get("effective_settings", {}).get("observed_attn_implementation") == "sdpa",
        "loaded FP32/SDPA identity",
    )
    observed = loaded.get("model_identity", {}).get("adapter", {})
    require(
        Path(observed.get("adapter_path", "")).resolve() == adapter_path.resolve()
        and observed.get("merged_adapters", []) == [],
        "loaded unmerged adapter identity",
    )


def qualify_batching(
    *,
    plan_path: Path,
    training_manifest_path: Path,
    output: Path,
    device: str,
) -> dict[str, Any]:
    """Run one bounded Source-adapter parity/throughput batch qualification."""

    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig

    require(device.startswith("cuda") and torch.cuda.is_available(), "CUDA readback qualification")
    require(not output.exists(), "qualification output collision")
    checked = validate_plan(read(plan_path))
    manifest = runtime.validate_training_manifest(read(training_manifest_path))
    require(manifest["preparation"] == checked["plan"]["preparation"], "plan/training preparation")
    config_data = _model_config(manifest)
    adapter_path, adapter, _ = _adapter(
        training_manifest_path, manifest, step=0, terminal_path=None
    )
    config = checkpoint_config(InferConfig.model_validate(config_data), str(adapter_path))
    torch.cuda.set_device(device)
    load_started = time.monotonic()
    qwen, loaded = load_policy(config, device=torch.device(device))
    _loaded_identity(loaded, adapter_path=adapter_path)
    load_seconds = time.monotonic() - load_started
    ids = checked["plan"]["qualification"]["image_ids"]
    items = _train_items(checked["prepared"], ids)
    configurations: dict[str, dict[str, Any]] = {}
    serial = _generate(
        qwen,
        config,
        config_data,
        items[:SERIAL_REFERENCE_COUNT],
        batch_size=1,
        formal=False,
    )
    serial["exact_reference_parity"] = True
    configurations["serial"] = serial
    reference = {
        row["image_id"]: (row["generated_token_ids"], row["decode_stop_reason"])
        for row in serial["rows"]
    }
    for candidate in QUALIFICATION_CANDIDATES:
        name = f"batch{candidate}"
        try:
            receipt = _generate(
                qwen,
                config,
                config_data,
                items,
                batch_size=candidate,
                formal=True,
            )
            if candidate == 4:
                compared = reference
            else:
                compared = {
                    row["image_id"]: (row["generated_token_ids"], row["decode_stop_reason"])
                    for row in configurations["batch4"]["rows"]
                }
            receipt["exact_reference_parity"] = all(
                compared[row["image_id"]]
                == (row["generated_token_ids"], row["decode_stop_reason"])
                for row in receipt["rows"]
                if row["image_id"] in compared
            ) and set(compared) <= {row["image_id"] for row in receipt["rows"]}
            configurations[name] = receipt
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            configurations[name] = {
                "status": "rejected_oom",
                "configured_batch_size": candidate,
                "request_count": 0,
                "exact_reference_parity": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if candidate == 4:
                break
    selection = select_qualified_batch(configurations)
    value = {
        "schema": f"{SCHEMA}.qualification",
        "status": selection["status"],
        "plan": training.binding(plan_path),
        "training_manifest": training.binding(training_manifest_path),
        "adapter": adapter,
        "policy": dict(POLICY),
        "loaded_model": loaded,
        "model_load_seconds": load_seconds,
        "configurations": configurations,
        "selection": selection,
        "elapsed_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    training.publish(output, value)
    return value


def _qualified_batch(path: Path) -> int:
    value = read(path)
    require(
        value.get("schema") == f"{SCHEMA}.qualification"
        and value.get("status") == "qualified"
        and value.get("selection", {}).get("selected_batch_size") in QUALIFICATION_CANDIDATES,
        "qualified readback batch",
    )
    observed = select_qualified_batch(value["configurations"])
    require(observed == value["selection"], "qualification selection changed")
    return int(observed["selected_batch_size"])


def qualification_checkpoint_worker(
    *,
    plan_path: Path,
    qualification_path: Path,
    training_manifest_path: Path,
    terminal_path: Path,
    output: Path,
    device: str,
) -> dict[str, Any]:
    """Cold-reload one step-2 adapter/AdamW/scheduler and read four images."""

    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import (
        bind_source256_language_dora,
        load_policy,
    )
    from src.config.inference import InferConfig

    require(device.startswith("cuda") and torch.cuda.is_available(), "CUDA checkpoint qualification")
    require(not output.exists(), "checkpoint qualification output collision")
    require(_qualified_batch(qualification_path) == 4, "user-frozen batch4 qualification")
    checked = validate_plan(read(plan_path))
    manifest_path = training_manifest_path.resolve(strict=True)
    manifest = runtime.validate_training_manifest(read(manifest_path))
    require(
        manifest["mode"] == "qualification"
        and manifest["runtime"]["updates"] == 2
        and manifest["runtime"]["checkpoint_steps"] == [2]
        and manifest["preparation"] == checked["plan"]["preparation"],
        "two-step qualification manifest",
    )
    config_data = _model_config(manifest)
    adapter_path, adapter, terminal = _adapter(
        manifest_path,
        manifest,
        step=2,
        terminal_path=terminal_path,
    )
    require(terminal is not None, "qualification terminal")
    checkpoint, = [item for item in terminal["checkpoints"] if item["step"] == 2]
    state_path = _verify_binding(checkpoint["state"], "checkpoint state")
    scheduler_path = _verify_binding(checkpoint["scheduler"], "checkpoint scheduler")
    config = checkpoint_config(InferConfig.model_validate(config_data), str(adapter_path))
    torch.cuda.set_device(device)
    load_started = time.monotonic()
    qwen, loaded = load_policy(config, device=torch.device(device))
    _loaded_identity(loaded, adapter_path=adapter_path)
    scalar_count = runtime.source_adapter_scalar_count(manifest["source_adapter"])
    named, frozen = bind_source256_language_dora(
        qwen.model,
        expected_tensor_count=manifest["source_adapter"]["semantic_identity"]["tensor_key_count"],
        expected_scalar_count=scalar_count,
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        **{**manifest["optimizer"], "betas": tuple(manifest["optimizer"]["betas"])},
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=manifest["scheduler"]["total_updates"],
        eta_min=manifest["optimizer"]["lr"] * manifest["scheduler"]["min_lr_ratio"],
    )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    require(
        state.get("schema") == f"{training.SCHEMA}.checkpoint.v1"
        and state.get("manifest") == training.binding(manifest_path)
        and state.get("source_adapter") == manifest["source_adapter"]
        and state.get("saved_adapter") == adapter
        and state.get("optimizer") == manifest["optimizer"]
        and state.get("parameter_layout") == training._layout(named)
        and state.get("step") == 2,
        "saved checkpoint state identity",
    )
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
    scheduler.load_state_dict(scheduler_state)
    require(scheduler.state_dict() == scheduler_state, "scheduler reload identity")
    restored = runtime.distributed.state_fingerprint(named, optimizer)
    consensus, = [
        item
        for item in terminal["distributed"]["checkpoint_consensus"]
        if item["step"] == 2
    ]
    require(
        consensus["rank_count"] == runtime.REQUIRED_WORLD_SIZE
        and restored == consensus["state"]
        and restored["optimizer_steps"] == [2]
        and scheduler.last_epoch == 2,
        "cold checkpoint/optimizer/scheduler state",
    )
    require(
        all(parameter.grad is None for _, parameter in named)
        and all(parameter.grad is None for _, parameter in frozen),
        "cold reload has no stale gradients",
    )
    qwen.model.eval()
    state_load_seconds = time.monotonic() - load_started
    ids = checked["plan"]["qualification"]["image_ids"]
    generated = _generate(
        qwen,
        config,
        config_data,
        _train_items(checked["prepared"], ids),
        batch_size=4,
        formal=True,
    )
    value = {
        "schema": f"{SCHEMA}.qualification_checkpoint",
        "status": "completed_unscored",
        "arm": manifest["arm"],
        "step": 2,
        "policy": dict(POLICY),
        "batch_size": 4,
        "plan": training.binding(plan_path),
        "batch_qualification": training.binding(qualification_path),
        "training_manifest": training.binding(manifest_path),
        "training_terminal": training.binding(terminal_path),
        "checkpoint_adapter": adapter,
        "loaded_model": loaded,
        "model_and_state_load_seconds": state_load_seconds,
        "reload": {
            "checkpoint_state": checkpoint["state"],
            "scheduler_state": checkpoint["scheduler"],
            "state_fingerprint": restored,
            "scheduler_last_epoch": scheduler.last_epoch,
            "optimizer_lr": float(optimizer.param_groups[0]["lr"]),
            "saved_rng_state_count": len(state.get("cuda_rng_state_all", [])),
        },
        "generation": generated,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    training.publish(output, value)
    return value


def endpoint_worker(
    *,
    plan_path: Path,
    qualification_path: Path,
    training_manifest_path: Path,
    terminal_path: Path | None,
    step: int,
    split: str,
    shard: int,
    output: Path,
    device: str,
) -> dict[str, Any]:
    """Read one immutable train/dev shard from Source0, step16 or step64."""

    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig

    require(split in SPLIT_COUNTS and 0 <= shard < ENDPOINT_SHARDS, "endpoint split/shard")
    require(device.startswith("cuda") and torch.cuda.is_available(), "CUDA endpoint readback")
    require(not output.exists(), "endpoint output collision")
    batch_size = _qualified_batch(qualification_path)
    checked = validate_plan(read(plan_path))
    manifest = runtime.validate_training_manifest(read(training_manifest_path))
    require(manifest["preparation"] == checked["plan"]["preparation"], "plan/training preparation")
    config_data = _model_config(manifest)
    adapter_path, adapter, terminal = _adapter(
        training_manifest_path,
        manifest,
        step=step,
        terminal_path=terminal_path,
    )
    ids = checked["plan"]["endpoint_shards"][split][shard]
    items = (
        _train_items(checked["prepared"], ids)
        if split == "train"
        else _dev_items(checked["prepared"], ids)
    )
    require(len(items) % batch_size == 0, "endpoint shard must use full qualified batches")
    config = checkpoint_config(InferConfig.model_validate(config_data), str(adapter_path))
    torch.cuda.set_device(device)
    load_started = time.monotonic()
    qwen, loaded = load_policy(config, device=torch.device(device))
    _loaded_identity(loaded, adapter_path=adapter_path)
    load_seconds = time.monotonic() - load_started
    generated = _generate(
        qwen,
        config,
        config_data,
        items,
        batch_size=batch_size,
        formal=True,
    )
    value = {
        "schema": f"{SCHEMA}.endpoint_shard",
        "status": "completed_unscored",
        "endpoint": {"arm": "Source" if step == 0 else manifest["arm"], "step": step},
        "split": split,
        "shard": shard,
        "shard_count": ENDPOINT_SHARDS,
        "policy": dict(POLICY),
        "batch_size": batch_size,
        "plan": training.binding(plan_path),
        "qualification": training.binding(qualification_path),
        "training_manifest": training.binding(training_manifest_path),
        "training_terminal": None if terminal_path is None else training.binding(terminal_path),
        "checkpoint_adapter": adapter,
        "loaded_model": loaded,
        "model_load_seconds": load_seconds,
        "generation": generated,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    require(terminal is None if step == 0 else terminal is not None, "endpoint terminal mode")
    training.publish(output, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--preparation", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    qualify = sub.add_parser("qualify-batching")
    qualify.add_argument("--plan", type=Path, required=True)
    qualify.add_argument("--training-manifest", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--device", default="cuda:0")
    checkpoint = sub.add_parser("qualification-checkpoint")
    checkpoint.add_argument("--plan", type=Path, required=True)
    checkpoint.add_argument("--qualification", type=Path, required=True)
    checkpoint.add_argument("--training-manifest", type=Path, required=True)
    checkpoint.add_argument("--terminal", type=Path, required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    checkpoint.add_argument("--device", default="cuda:0")
    endpoint = sub.add_parser("endpoint-worker")
    endpoint.add_argument("--plan", type=Path, required=True)
    endpoint.add_argument("--qualification", type=Path, required=True)
    endpoint.add_argument("--training-manifest", type=Path, required=True)
    endpoint.add_argument("--terminal", type=Path)
    endpoint.add_argument("--step", type=int, required=True)
    endpoint.add_argument("--split", choices=tuple(SPLIT_COUNTS), required=True)
    endpoint.add_argument("--shard", type=int, required=True)
    endpoint.add_argument("--output", type=Path, required=True)
    endpoint.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "prepare":
        result = build_plan(preparation_path=args.preparation, output=args.output)
    elif args.command == "qualify-batching":
        result = qualify_batching(
            plan_path=args.plan,
            training_manifest_path=args.training_manifest,
            output=args.output,
            device=args.device,
        )
    elif args.command == "qualification-checkpoint":
        result = qualification_checkpoint_worker(
            plan_path=args.plan,
            qualification_path=args.qualification,
            training_manifest_path=args.training_manifest,
            terminal_path=args.terminal,
            output=args.output,
            device=args.device,
        )
    else:
        result = endpoint_worker(
            plan_path=args.plan,
            qualification_path=args.qualification,
            training_manifest_path=args.training_manifest,
            terminal_path=args.terminal,
            step=args.step,
            split=args.split,
            shard=args.shard,
            output=args.output,
            device=args.device,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

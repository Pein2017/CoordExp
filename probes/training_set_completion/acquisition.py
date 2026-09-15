"""Stage-01 native HF empty-prefix acquisition for the fixed N16 train11 cohort."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from probes.training_set_completion import artifacts as artifact_primitives
from src.inference import input_materialization


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-14-training-set-completion-curriculum" / "stage01-acquisition-v1"
SOURCE_PACKET = BASE / "2026-09-11-positive-progress-matched-control" / "endpoint-preparation" / "packet.json"
TRAINING_INPUT = BASE / "2026-09-12-native-owner-scale-and-state" / "scale/training/preparation/inputs-v2.json"
TRAINING_RECEIPT = BASE / "2026-09-12-native-owner-scale-and-state" / "scale/training/full-fixedP-N16-v2/receipt.json"
N16_ADAPTER = BASE / "2026-09-12-native-owner-scale-and-state" / "scale/training/full-fixedP-N16-v2/adapter"
IMAGE_IDS = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
TEMPERATURES = (0.1, 0.3, 0.7)
SEED_ROOT = 2026091401
CAP, EOS = 3084, 151645
GPUS = tuple(range(8))
MAX_PHASE_SECONDS = 3600
SCHEMA = "training_set_completion.stage01_acquisition.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_canonical = artifact_primitives.canonical
digest = artifact_primitives.digest
file_hash = artifact_primitives.file_hash


def binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f"bound path is not a file: {path}")
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def tree_binding(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve(strict=True)
    require(root.is_dir(), f"bound path is not a directory: {root}")
    files = [
        {"relative_path": str(item.relative_to(root)), "sha256": file_hash(item), "size_bytes": item.stat().st_size}
        for item in sorted(root.rglob("*")) if item.is_file()
    ]
    require(files, f"bound directory is empty: {root}")
    return {"root": str(root), "file_count": len(files), "files": files, "fingerprint": digest(files)}


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def publish(path: str | Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(value)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == data, f"publication readback differs: {path}")


def sample_seed(image_id: int, temperature: float) -> int:
    value = int(digest({"root": SEED_ROOT, "image_id": image_id, "temperature": temperature})[:8], 16)
    return value & 0x7FFF_FFFF


def request_id(image_id: int, temperature: float | None) -> str:
    suffix = "greedy" if temperature is None else f"sample-t{str(temperature).replace('.', 'p')}"
    return f"stage01:image-{image_id:012d}:{suffix}"


def request_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_id in IMAGE_IDS:
        rows.append({"request_id": request_id(image_id, None), "image_id": image_id, "kind": "greedy",
                     "temperature": 0.0, "seed": None, "grouping": "one_request_per_generate_call"})
        for temperature in TEMPERATURES:
            rows.append({"request_id": request_id(image_id, temperature), "image_id": image_id, "kind": "sample",
                         "temperature": temperature, "seed": sample_seed(image_id, temperature),
                         "grouping": "one_request_per_generate_call"})
    validate_request_plan(rows)
    return rows


def validate_request_plan(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == 44 and len({r["request_id"] for r in rows}) == 44, "request denominator/identity")
    require({int(r["image_id"]) for r in rows} == set(IMAGE_IDS), "request image cohort")
    require(all(sum(int(x["image_id"]) == image for x in rows) == 4 for image in IMAGE_IDS), "four policies/image")
    seeds = [r["seed"] for r in rows if r["kind"] == "sample"]
    require(len(seeds) == len(set(seeds)) == 33 and all(type(seed) is int and seed >= 0 for seed in seeds), "sample seeds")
    for row in rows:
        require(row["grouping"] == "one_request_per_generate_call", "generation grouping")
        require((row["kind"] == "greedy" and row["temperature"] == 0.0 and row["seed"] is None)
                or (row["kind"] == "sample" and row["temperature"] in TEMPERATURES and type(row["seed"]) is int),
                "request policy")


def _verify_tree(value: Mapping[str, Any]) -> None:
    root = Path(value["root"]).resolve(strict=True)
    observed = tree_binding(root)
    require(observed == value, f"bound directory changed: {root}")


def prepare(output: Path = ROOT) -> dict[str, Any]:
    from src.adapters.dora import inspect_dora_adapter_payload

    require(not output.exists(), f"stage root already exists: {output}")
    source = read(SOURCE_PACKET)
    train = read(TRAINING_INPUT)
    receipt = read(TRAINING_RECEIPT)
    require(source.get("schema") == "positive_progress_matched_endpoint.packet.v1", "source packet schema")
    require(train.get("schema") == "parallel_owner_training.inputs.v1", "training packet schema")
    require(receipt.get("status") == "technically_completed_cold_pending", "N16 training receipt status")
    require(Path(receipt["saved_adapter"]["root"]).resolve() == N16_ADAPTER.resolve(), "N16 adapter path")
    positive_images = {int(row["image"]["image_id"]) for row in train["positive_records"]}
    require(positive_images == set(IMAGE_IDS), "N16 positive image cohort")
    by_image = {int(row["image_id"]): row for row in source["eval_records"]}
    require(set(IMAGE_IDS) <= set(by_image), "N16 cohort missing from source packet")
    config = copy.deepcopy(source["config"])
    require(config["backend"] == {"type": "hf", "hf": {"attn_implementation": "sdpa", "patch_embed_linearization": "enabled"}}, "HF SDPA config")
    require(config["model"]["dtype"] == "fp32" and config["embedding_delta"] is not None, "FP32 embedding composition")
    config["adapter"]["path"] = str(N16_ADAPTER)
    config["generation"].update(batch_size=2, max_new_tokens=CAP, n=1, repetition_penalty=1.0,
                                temperature=0.0, top_p=1.0)
    requests = request_plan()
    records = []
    for image_id in IMAGE_IDS:
        frozen = by_image[image_id]
        path = Path(frozen["case"]["image_path"]).resolve(strict=True)
        require(file_hash(path) == frozen["case"]["image_plan"]["image_content_sha256"], "source image bytes")
        records.append({"image_id": image_id, "example_id": frozen["example_id"],
                        "prompt_token_ids": frozen["prompt_token_ids"], "prompt_token_ids_sha256": digest(frozen["prompt_token_ids"]),
                        "case": frozen["case"], "golden": frozen["golden"], "source_split": frozen["split"],
                        "image_file": binding(path)})
    adapter_identity = inspect_dora_adapter_payload(N16_ADAPTER, config["model"]["base_model"])
    manifest = {"schema": SCHEMA, "status": "frozen_ready_for_smoke", "request_count": 44,
                "cohort": {"lineage": "N16", "image_ids": list(IMAGE_IDS), "image_count": 11},
                "policy": {"empty_assistant_prefix": True, "assistant_token_cap": CAP, "eos_token_id": EOS,
                           "repetition_penalty": 1.0, "top_p": 1.0, "top_k": 0,
                           "temperatures": [0.0, *TEMPERATURES], "seed_root": SEED_ROOT,
                           "fixed_grouping": "one_request_per_generate_call"},
                "model": {"config": config, "base_model": tree_binding(config["model"]["base_model"]),
                          "adapter": adapter_identity, "adapter_files": tree_binding(N16_ADAPTER),
                          "embedding_delta": tree_binding(config["embedding_delta"]["path"])},
                "sources": {"source_packet": binding(SOURCE_PACKET), "training_input": binding(TRAINING_INPUT),
                            "training_receipt": binding(TRAINING_RECEIPT), "producer": binding(Path(__file__)),
                            "artifact_primitives": binding(Path(artifact_primitives.__file__)),
                            "input_materialization": binding(Path(input_materialization.__file__))},
                "records": records, "requests": requests,
                "execution": {"physical_gpus": list(GPUS), "max_phase_seconds": MAX_PHASE_SECONDS,
                              "smoke_request_ids": [requests[0]["request_id"], requests[1]["request_id"]],
                              "long_jobs_require_named_tmux": True}}
    predecessor = ROOT / "terminals/smoke-shard-0.json"
    if output.resolve() != ROOT.resolve() and predecessor.is_file():
        manifest["diagnosed_predecessor"] = {
            "classification": "pre_model_config_admission_failure_batch_size_one",
            "reuse": "none; zero requests, model loads, forwards, image forwards, and generated tokens",
            "terminal": binding(predecessor),
            "outer_exits": binding(ROOT / "smoke-exits.json"),
        }
    manifest["content_sha256"] = digest(manifest)
    publish(output / "manifest.json", manifest)
    validate_manifest(output / "manifest.json")
    return manifest


def validate_manifest(path: str | Path = ROOT / "manifest.json", *, verify_large_trees: bool = False) -> dict[str, Any]:
    from src.config.inference import InferConfig

    manifest = read(path)
    claimed = manifest.get("content_sha256")
    content = {k: v for k, v in manifest.items() if k != "content_sha256"}
    require(manifest.get("schema") == SCHEMA and claimed == digest(content), "manifest schema/content")
    for name, source in manifest["sources"].items():
        observed = binding(source["path"])
        if observed != source and name == "producer":
            snapshot = Path(path).resolve().parent / "producer-snapshot-executed.py"
            require(snapshot.is_file() and file_hash(snapshot) == source["sha256"]
                    and snapshot.stat().st_size == source["size_bytes"], "executed producer snapshot")
        else:
            require(observed == source, f"bound source changed: {source['path']}")
    validate_request_plan(manifest["requests"])
    require(manifest["request_count"] == 44 and manifest["cohort"]["image_ids"] == list(IMAGE_IDS), "manifest cohort")
    require(manifest["policy"] == {"empty_assistant_prefix": True, "assistant_token_cap": CAP, "eos_token_id": EOS,
            "repetition_penalty": 1.0, "top_p": 1.0, "top_k": 0, "temperatures": [0.0, *TEMPERATURES],
            "seed_root": SEED_ROOT, "fixed_grouping": "one_request_per_generate_call"}, "manifest policy")
    config = InferConfig.model_validate(manifest["model"]["config"])
    require(config.generation.batch_size == 2 and config.model.dtype == "fp32"
            and config.backend.type == "hf" and config.backend.hf.attn_implementation == "sdpa", "validated HF config")
    require(len(manifest["records"]) == 11 and len({r["example_id"] for r in manifest["records"]}) == 11, "record identities")
    for record in manifest["records"]:
        require(digest(record["prompt_token_ids"]) == record["prompt_token_ids_sha256"], "prompt IDs")
        require(binding(record["image_file"]["path"]) == record["image_file"], "media bytes")
        require(record["case"]["image_plan"]["image_content_sha256"] == record["image_file"]["sha256"], "media plan")
    require(Path(manifest["model"]["adapter"]["root"]).resolve() == N16_ADAPTER.resolve(), "adapter identity")
    require(manifest["model"]["adapter_files"]["fingerprint"] == tree_binding(N16_ADAPTER)["fingerprint"], "adapter files")
    if "diagnosed_predecessor" in manifest:
        for name in ("terminal", "outer_exits"):
            value = manifest["diagnosed_predecessor"][name]
            require(binding(value["path"]) == value, f"diagnosed predecessor {name}")
    if verify_large_trees:
        for tree in manifest["model"].values():
            if isinstance(tree, Mapping) and set(tree) == {"root", "file_count", "files", "fingerprint"}:
                _verify_tree(tree)
    return manifest


def _phase_requests(manifest: Mapping[str, Any], phase: str, shard: int, world_size: int) -> list[dict[str, Any]]:
    require(phase in ("smoke", "remaining") and 0 <= shard < world_size, "phase/shard")
    smoke = set(manifest["execution"]["smoke_request_ids"])
    selected = [r for r in manifest["requests"] if (r["request_id"] in smoke) == (phase == "smoke")]
    require(len(selected) == (2 if phase == "smoke" else 42), "phase request denominator")
    return selected[shard::world_size]


def _checked_terminal(ids: Sequence[int], stop: str) -> None:
    require(ids and len(ids) <= CAP and all(type(token) is int and token >= 0 for token in ids), "generated token IDs")
    require((stop == "im_end" and ids[-1] == EOS and EOS not in ids[:-1])
            or (stop == "length" and len(ids) == CAP and EOS not in ids), "generated terminal")


def worker(*, manifest_path: Path, output: Path, phase: str, shard: int, world_size: int, physical_gpu: int) -> None:
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.inference.bound_requests import build_bound_native_requests as build_requests
    from src.eval.native_rows import native_detection_record as native_record
    from src.adapters.dora import inspect_dora_adapter_payload
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    manifest = validate_manifest(manifest_path)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu) and torch.cuda.device_count() == 1,
            "worker requires exact one visible physical GPU")
    jobs = _phase_requests(manifest, phase, shard, world_size)
    output.mkdir(parents=True, exist_ok=True)
    terminal_path = output / "terminals" / f"{phase}-shard-{shard}.json"
    require(not terminal_path.exists(), "worker terminal collision")
    terminal: dict[str, Any] = {"schema": f"{SCHEMA}.terminal", "status": "running", "phase": phase,
        "shard": shard, "world_size": world_size, "physical_gpu": physical_gpu, "pid": os.getpid(),
        "manifest": binding(manifest_path), "expected_requests": len(jobs), "completed_requests": 0,
        "model_loads": 0, "model_forwards": 0, "image_forwards": 0, "new_tokens": 0}
    started = time.monotonic()
    handles = []
    qwen = None
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("acquisition phase wall bound")))
        signal.alarm(MAX_PHASE_SECONDS)
        config = checkpoint_config(InferConfig.model_validate(manifest["model"]["config"]), N16_ADAPTER)
        observed_adapter = inspect_dora_adapter_payload(N16_ADAPTER, config.model.base_model)
        require(observed_adapter["fingerprint"] == manifest["model"]["adapter"]["fingerprint"], "adapter fingerprint")
        torch.cuda.set_device(torch.device("cuda:0"))
        torch.cuda.reset_peak_memory_stats()
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        require(identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"]
                and identity["effective_settings"]["observed_attn_implementation"] == "sdpa", "live FP32/SDPA")
        require(identity["model_identity"]["adapter"]["adapter_path"] == str(N16_ADAPTER), "live N16 adapter")
        publish(output / "model" / f"{phase}-shard-{shard}.json", identity)
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)
        handles.append(qwen.model.register_forward_pre_hook(lambda *_: terminal.__setitem__("model_forwards", terminal["model_forwards"] + 1)))
        visual = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visual) == 1, "single visual module")
        handles.append(visual[0].register_forward_pre_hook(lambda *_: terminal.__setitem__("image_forwards", terminal["image_forwards"] + 1)))
        by_image = {int(r["image_id"]): r for r in manifest["records"]}
        for job in jobs:
            record = by_image[int(job["image_id"])]
            case = input_materialization.materialize_bound_single_image_case(
                record["case"], manifest["model"]["config"]
            )
            requests, _ = build_requests(qwen, manifest["model"]["config"], [case])
            batch = prepare_native_inputs(qwen.processor, requests, device="cuda:0", record_media_identity=True)
            plan = record["case"]["image_plan"]
            require(list(batch.prompt_token_ids[0]) == record["prompt_token_ids"], "live prompt token identity")
            require(batch.media_sha256[0] == plan["executed_media_sha256"]
                    and list(batch.image_grids[0]) == plan["observed_image_grid_thw"], "live media/grid identity")
            policy = NativeGenerationPolicy(temperature=float(job["temperature"]), top_p=1.0, top_k=0,
                                            repetition_penalty=1.0, use_model_defaults=False)
            tick = time.monotonic()
            with torch.inference_mode():
                generated, = generate_continuations(qwen.model, batch, extensions=[[]], budgets=[CAP],
                    eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=policy,
                    trace="none", seed=job["seed"])
            ids = list(generated.token_ids)
            _checked_terminal(ids, generated.stop_reason)
            text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
            parsed = native_record(text, record["case"], record["golden"], generated.stop_reason)
            payload = {"schema": f"{SCHEMA}.row", "request": dict(job), "manifest_sha256": manifest["content_sha256"],
                "example_id": record["example_id"], "image_id": record["image_id"], "source_split": record["source_split"],
                "empty_assistant_prefix": True, "assistant_token_cap": CAP,
                "prompt_token_ids": record["prompt_token_ids"], "prompt_token_ids_sha256": record["prompt_token_ids_sha256"],
                "executed_media_sha256": batch.media_sha256[0], "observed_image_grid_thw": list(batch.image_grids[0]),
                "generated_token_ids": ids, "generated_token_ids_sha256": digest(ids), "raw_decode_text": text,
                "decode_stop_reason": generated.stop_reason, "generated_token_count": len(ids), "parsed": parsed,
                "timing": {"generation_seconds": time.monotonic() - tick},
                "model_receipt": binding(output / "model" / f"{phase}-shard-{shard}.json")}
            publish(output / "rows" / (hashlib.sha256(job["request_id"].encode()).hexdigest() + ".json"), payload)
            terminal["completed_requests"] += 1
            terminal["new_tokens"] += len(ids)
        require(terminal["completed_requests"] == terminal["expected_requests"], "worker request denominator")
        terminal["status"] = "completed"
        terminal["exit_code"] = 0
    except BaseException as exc:
        terminal.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        for handle in handles:
            handle.remove()
        terminal.update(elapsed_seconds=time.monotonic() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        publish(terminal_path, terminal)


def launch(*, manifest_path: Path, output: Path, phase: str) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path)
    require(output.resolve() == ROOT.resolve() or output.name.startswith(ROOT.name + "-retry"),
            "launch output must be the frozen stage root or a diagnosed retry root")
    if phase == "remaining":
        smoke = read(output / "smoke-readback.json")
        require(smoke.get("status") == "passed" and smoke.get("request_count") == 2, "remaining requires passing smoke")
    world_size = 1 if phase == "smoke" else 8
    launch_path = output / f"{phase}-launch.json"
    require(not launch_path.exists(), "phase launch collision")
    processes = []
    commands = []
    for shard in range(world_size):
        gpu = GPUS[shard]
        command = [sys.executable, "-m", "probes.training_set_completion.acquisition", "worker",
                   "--manifest", str(manifest_path.resolve()), "--output", str(output.resolve()), "--phase", phase,
                   "--shard", str(shard), "--world-size", str(world_size), "--physical-gpu", str(gpu)]
        log_path = output / "logs" / f"{phase}-shard-{shard}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("x")
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2",
                                        "TOKENIZERS_PARALLELISM": "false"}, stdout=log, stderr=subprocess.STDOUT)
        commands.append({"shard": shard, "physical_gpu": gpu, "pid": process.pid, "command": command,
                         "log": str(log_path), "expected_requests": len(_phase_requests(manifest, phase, shard, world_size))})
        processes.append((process, log))
    publish(launch_path, {"schema": f"{SCHEMA}.launch", "status": "launched", "phase": phase,
                          "manifest": binding(manifest_path), "commands": commands, "max_phase_seconds": MAX_PHASE_SECONDS})
    exits = []
    for spec, (process, log) in zip(commands, processes, strict=True):
        exits.append({"shard": spec["shard"], "pid": spec["pid"], "exit_code": process.wait()})
        log.close()
    publish(output / f"{phase}-exits.json", {"schema": f"{SCHEMA}.exits", "phase": phase, "exits": exits})
    require(all(item["exit_code"] == 0 for item in exits), f"{phase} worker failed; partial evidence preserved")
    return {"status": "completed", "phase": phase, "request_count": sum(x["expected_requests"] for x in commands)}


def _row_path(output: Path, request: Mapping[str, Any]) -> Path:
    return output / "rows" / (hashlib.sha256(request["request_id"].encode()).hexdigest() + ".json")


def validate_result_payload(payload: Mapping[str, Any], request: Mapping[str, Any], record: Mapping[str, Any], *, decode) -> None:
    require(payload.get("schema") == f"{SCHEMA}.row" and payload.get("request") == dict(request), "result request identity")
    require(payload.get("example_id") == record["example_id"] and payload.get("image_id") == record["image_id"], "result image identity")
    require(payload.get("empty_assistant_prefix") is True and payload.get("assistant_token_cap") == CAP, "result prefix/budget")
    require(payload.get("prompt_token_ids") == record["prompt_token_ids"]
            and digest(payload["prompt_token_ids"]) == payload.get("prompt_token_ids_sha256"), "result prompt IDs")
    ids = payload.get("generated_token_ids")
    require(isinstance(ids, list) and digest(ids) == payload.get("generated_token_ids_sha256"), "result generated IDs")
    _checked_terminal(ids, payload.get("decode_stop_reason"))
    require(payload.get("generated_token_count") == len(ids) and decode(ids) == payload.get("raw_decode_text"), "result token/text")
    require(payload.get("executed_media_sha256") == record["case"]["image_plan"]["executed_media_sha256"], "result media")
    require(payload.get("observed_image_grid_thw") == record["case"]["image_plan"]["observed_image_grid_thw"], "result grid")


def geometry_invalid_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        dropped.get("reason") == "geometry_invalid"
        for row in rows
        for dropped in row["parsed"]["dropped_predictions"]
    )


def readback(*, manifest_path: Path, output: Path, phase: str, receipt_name: str | None = None) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from src.eval.native_rows import native_detection_record as native_record

    manifest = validate_manifest(manifest_path)
    require(phase in ("smoke", "final"), "readback phase")
    selected = [r for r in manifest["requests"] if phase == "final" or r["request_id"] in manifest["execution"]["smoke_request_ids"]]
    require(len(selected) == (2 if phase == "smoke" else 44), "readback denominator")
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"]["base_model"]["root"], local_files_only=True)
    by_image = {int(r["image_id"]): r for r in manifest["records"]}
    rows = []
    for request in selected:
        payload = read(_row_path(output, request))
        record = by_image[int(request["image_id"])]
        validate_result_payload(payload, request, record,
                                decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False))
        reparsed = native_record(payload["raw_decode_text"], record["case"], record["golden"], payload["decode_stop_reason"])
        require(reparsed == payload["parsed"], "cold parser readback")
        rows.append(payload)
    request_ids = [row["request"]["request_id"] for row in rows]
    require(len(request_ids) == len(set(request_ids)), "cold duplicate request")
    receipt = {"schema": f"{SCHEMA}.{phase}_readback", "status": "passed", "manifest": binding(manifest_path),
               "request_count": len(rows), "request_ids_sha256": digest(request_ids),
               "generated_tokens": sum(row["generated_token_count"] for row in rows),
               "stop_counts": {stop: sum(row["decode_stop_reason"] == stop for row in rows) for stop in ("im_end", "length")},
               "parser": {"valid_predictions": sum(row["parsed"]["valid_prediction_count"] for row in rows),
                          "dropped_predictions": sum(row["parsed"]["dropped_prediction_count"] for row in rows)},
               "geometry_invalid": geometry_invalid_count(rows),
               "cold_token_text_and_parser_replay": True}
    name = receipt_name or ("smoke-readback.json" if phase == "smoke" else "result.json")
    require(Path(name).name == name and name.endswith(".json"), "readback receipt name")
    if receipt_name is not None:
        receipt["consumer_repair"] = {
            "scope": "geometry-invalid aggregation only; generated rows and original parser payloads unchanged",
            "supersedes": binding(output / "result.json"),
            "executed_producer_snapshot": binding(output / "producer-snapshot-executed.py"),
        }
    publish(output / name, receipt)
    if phase == "final":
        rows_path = output / "rows.jsonl"
        if rows_path.exists():
            existing = [json.loads(line) for line in rows_path.read_text().splitlines()]
            require(existing == rows, "existing merged rows differ from immutable request rows")
        else:
            with rows_path.open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        require(sum(1 for _ in rows_path.open()) == 44, "merged row denominator")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "launch", "worker", "readback"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("smoke", "remaining", "final"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--verify-large-trees", action="store_true")
    parser.add_argument("--receipt-name")
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output)
    elif args.command == "verify":
        value = validate_manifest(args.manifest, verify_large_trees=args.verify_large_trees)
    elif args.command == "launch":
        require(args.phase in ("smoke", "remaining"), "launch phase required")
        value = launch(manifest_path=args.manifest, output=args.output, phase=args.phase)
    elif args.command == "worker":
        require(args.phase in ("smoke", "remaining") and args.shard is not None
                and args.world_size is not None and args.physical_gpu is not None, "worker arguments")
        worker(manifest_path=args.manifest, output=args.output, phase=args.phase, shard=args.shard,
               world_size=args.world_size, physical_gpu=args.physical_gpu)
        return
    else:
        require(args.phase in ("smoke", "final"), "readback phase required")
        value = readback(manifest_path=args.manifest, output=args.output, phase=args.phase,
                         receipt_name=args.receipt_name)
    print(json.dumps({"schema": value.get("schema"), "status": value.get("status"),
                      "request_count": value.get("request_count")}, sort_keys=True))


if __name__ == "__main__":
    main()

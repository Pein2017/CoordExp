"""Review-only COCO22 new-image discovery using the frozen HF native decode."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import resource
import shlex
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

from probes.training_set_completion import acquisition as frozen
from src.inference import input_materialization


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-15-coco22-cumulative-expansion" / "discovery-v1"
SOURCE_PACKET = frozen.SOURCE_PACKET
SOURCE_ADAPTER = BASE / "2026-09-15-coco227-ce-normalization/trial-v1/S/training/checkpoints/step-00256/adapter"
EMBEDDING_DELTA = Path("/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
                       "2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/"
                       "step-2444/special_token_embeddings")
ADAPTER_FINGERPRINT = "0a2d96f2f4a8c89c8e3db5cc743ebf4e4a22e7656d5a70437e68c66b59441a87"
OLD_IMAGE_IDS = frozenset(frozen.IMAGE_IDS)
SCHEMA = "training_set_completion.coco22_discovery.v1"
CAP, EOS = frozen.CAP, frozen.EOS
TEMPERATURES = frozen.TEMPERATURES
GPUS = tuple(range(8))
TMUX_FIRST = "coordexp-coco22-discovery-first"
TMUX_REMAINING = "coordexp-coco22-discovery-remaining"

require = frozen.require
binding = frozen.binding
tree_binding = frozen.tree_binding
digest = frozen.digest
file_hash = frozen.file_hash
read = frozen.read
publish = frozen.publish
geometry_invalid_count = frozen.geometry_invalid_count


def new_image_ids(cohort: Mapping[str, Any]) -> list[int]:
    ids = cohort.get("new_image_ids")
    if ids is None and isinstance(cohort.get("new_images"), list):
        ids = [record["image_id"] for record in cohort["new_images"]]
    require(isinstance(ids, list) and len(ids) == 11
            and all(type(item) is int and item > 0 for item in ids)
            and len(set(ids)) == 11 and not set(ids) & OLD_IMAGE_IDS,
            "11 distinct new image IDs outside frozen old cohort")
    return ids


def request_plan(image_ids: Sequence[int]) -> list[dict[str, Any]]:
    require(len(image_ids) == 11 and len(set(image_ids)) == 11
            and not set(image_ids) & OLD_IMAGE_IDS, "request image cohort")
    rows = []
    for image_id in image_ids:
        for temperature in (None, *TEMPERATURES):
            greedy = temperature is None
            rows.append({
                "request_id": f"coco22:new-image-{image_id:012d}:"
                              + ("greedy" if greedy else f"sample-t{str(temperature).replace('.', 'p')}"),
                "image_id": image_id, "kind": "greedy" if greedy else "sample",
                "temperature": 0.0 if greedy else temperature,
                "seed": None if greedy else frozen.sample_seed(image_id, temperature),
                "grouping": "one_request_per_generate_call",
            })
    validate_request_plan(rows, image_ids)
    return rows


def validate_request_plan(rows: Sequence[Mapping[str, Any]], image_ids: Sequence[int]) -> None:
    require(len(rows) == 44 and len({r["request_id"] for r in rows}) == 44,
            "request denominator/identity")
    require({r["image_id"] for r in rows} == set(image_ids)
            and all(sum(r["image_id"] == image_id for r in rows) == 4 for image_id in image_ids),
            "four policies/new image")
    seeds = [r["seed"] for r in rows if r["kind"] == "sample"]
    require(len(seeds) == len(set(seeds)) == 33
            and all(type(seed) is int and seed >= 0 for seed in seeds), "sample seeds")
    for row in rows:
        image_id = row["image_id"]
        temperature = None if row["kind"] == "greedy" else row["temperature"]
        require(temperature is None or temperature in TEMPERATURES, "request temperature")
        require(row == {
            "request_id": f"coco22:new-image-{image_id:012d}:"
                          + ("greedy" if temperature is None else f"sample-t{str(temperature).replace('.', 'p')}"),
            "image_id": image_id,
            "kind": "greedy" if temperature is None else "sample",
            "temperature": 0.0 if temperature is None else temperature,
            "seed": None if temperature is None else frozen.sample_seed(image_id, temperature),
            "grouping": "one_request_per_generate_call",
        }, "request policy/seed")


def _case_records(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = packet.get("eval_records", packet.get("records"))
    require(isinstance(records, list), "case packet requires eval_records or records")
    return records


def prepare_cases(*, cohort_path: Path, original_jsonl: Path, output: Path,
                  exclude_image_ids: Sequence[int] = (),
                  reuse_case_packet_path: Path | None = None) -> dict[str, Any]:
    """Freeze CPU-executed processor, prompt, and media cases without loading the model."""
    from src.config.inference import InferConfig
    from src.data.examples import raw_example_from_jsonl_row
    from src.inference.inputs import plan_examples
    from src.qwen.native import prepare_native_inputs
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options

    require(not output.exists(), "case packet already exists")
    cohort = read(cohort_path)
    ids = new_image_ids(cohort)
    excluded = set(exclude_image_ids)
    require(excluded <= set(ids), "excluded image is outside selected cohort")
    originals = cohort.get("new_originals")
    if isinstance(originals, Mapping):
        require(binding(original_jsonl) == {"path": originals["path"], "sha256": originals["sha256"],
                                            "size_bytes": originals["size_bytes"]},
                "source originals differ from selected cohort receipt")
    source = read(SOURCE_PACKET)
    require(source["schema"] == "positive_progress_matched_endpoint.packet.v1", "frozen source packet")
    config = InferConfig.model_validate(source["config"])
    require(config.model.dtype == "fp32" and config.backend.hf.attn_implementation == "sdpa"
            and Path(config.embedding_delta.path).resolve() == EMBEDDING_DELTA.resolve(),
            "case route frozen FP32/SDPA/embedding composition")
    components = load_qwen_components_from_options(QwenLoadOptions(
        base_model=config.model.base_model, dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    reusable: dict[int, dict[str, Any]] = {}
    if reuse_case_packet_path is not None:
        prior = read(reuse_case_packet_path)
        require(prior.get("schema") == f"{SCHEMA}.case_packet"
                and prior.get("content_sha256") == digest({key: value for key, value in prior.items()
                                                            if key != "content_sha256"})
                and prior.get("model_loaded") is False and prior.get("gpu_used") is False
                and prior.get("source_packet") == binding(SOURCE_PACKET)
                and prior["processor_identity"]["base_config_sha256"] == components.base_config_sha256
                and prior["processor_identity"]["tokenizer_sha256"] == components.tokenizer_sha256
                and prior["processor_identity"]["processor"] == components.processor_identity.to_artifact_dict(),
                "prior processor-only case packet identity")
        reusable = {record["image_id"]: record for record in prior["records"]}
    records = {}
    reused_ids = []
    planned_ids = []
    with original_jsonl.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            image_id = record["image_id"]
            require(image_id in ids and image_id not in records, "selected original cohort/unique IDs")
            if image_id in excluded:
                continue
            raw = raw_example_from_jsonl_row(record, jsonl_path=original_jsonl,
                                             row_number=index + 1, raw_line=line.rstrip("\n"))
            if image_id in reusable:
                prior_record = reusable[image_id]
                case = prior_record["case"]
                require(prior_record["image_id"] == image_id
                        and prior_record["example_id"] == raw.example_id
                        and case["row_index"] == index and case["input_record"] == record
                        and case["image_path"] == str(raw.image.path)
                        and case["image_plan"]["image_content_sha256"] == file_hash(raw.image.path)
                        and case["image_plan"]["status"] == "ok"
                        and case["image_plan"]["observed_image_grid_thw"]
                        == case["image_plan"]["expected_image_grid_thw"]
                        and prior_record["golden"]["gt"] == [obj.to_artifact_dict() for obj in raw.objects]
                        and len(prior_record["prompt_token_ids"])
                        == case["image_plan"]["backend_prompt_token_count"],
                        "reused case must match new cohort original row/media/GT")
                records[image_id] = prior_record
                reused_ids.append(image_id)
                continue
            planned, = plan_examples([raw], config=config, components=components, row_indices=[index])
            batch = prepare_native_inputs(components.processor, (planned.request,), device="cpu",
                                          record_media_identity=True)
            plan = planned.image.to_artifact_dict()
            require(list(batch.prompt_token_ids[0]) == planned.prompt.expected_executed_prompt_token_ids
                    and list(batch.image_grids[0]) == plan["expected_image_grid_thw"],
                    "CPU processor prompt/grid identity")
            plan.update(observed_image_grid_thw=list(batch.image_grids[0]),
                        executed_media_sha256=batch.media_sha256[0],
                        backend_prompt_token_count=len(batch.prompt_token_ids[0]),
                        backend_projection_evidence_kind="native_processor_executed_tensors")
            case = {"row_id": raw.example_id, "row_index": index, "input_record": record,
                    "image_path": str(raw.image.path), "image_width": raw.image.width,
                    "image_height": raw.image.height, "image_plan": plan}
            golden = {"row_id": raw.example_id, "row_index": index,
                      "example_id": raw.example_id, "image_path": str(raw.image.path),
                      "image_width": raw.image.width, "image_height": raw.image.height,
                      "gt": [obj.to_artifact_dict() for obj in raw.objects]}
            records[image_id] = {"image_id": image_id, "example_id": raw.example_id,
                                 "prompt_token_ids": list(batch.prompt_token_ids[0]),
                                 "case": case, "golden": golden, "split": "coco22_train_original_gt"}
            planned_ids.append(image_id)
    require(set(records) == set(ids) - excluded, "CPU case record denominator")
    packet = {"schema": f"{SCHEMA}.case_packet", "status": "candidate_prepared_pending_cohort_admission"
              if excluded else "complete_processor_identity_pending_lead_admission",
              "model_loaded": False, "gpu_used": False,
              "cohort": binding(cohort_path), "original_jsonl": binding(original_jsonl),
              "source_packet": binding(SOURCE_PACKET), "producer": binding(Path(__file__)),
              "processor_identity": components.to_artifact_dict(),
              "selected_image_ids": ids, "excluded_image_ids": sorted(excluded),
              "reuse_case_packet": binding(reuse_case_packet_path) if reuse_case_packet_path is not None else None,
              "reused_image_ids": reused_ids, "newly_planned_image_ids": planned_ids,
              "record_count": len(records),
              "records": [records[image_id] for image_id in ids if image_id not in excluded]}
    packet["content_sha256"] = digest(packet)
    publish(output, packet)
    require(read(output) == packet, "cold case packet readback")
    return packet


def prepare(*, cohort_path: Path, case_packet_path: Path, output: Path = ROOT) -> dict[str, Any]:
    from src.adapters.dora import inspect_dora_adapter_payload

    require(not output.exists(), "discovery root already exists")
    cohort = read(cohort_path)
    ids = new_image_ids(cohort)
    packet = read(case_packet_path)
    if packet.get("schema") == f"{SCHEMA}.case_packet":
        require(packet.get("record_count") == 11 and packet.get("excluded_image_ids") == []
                and packet.get("selected_image_ids") == ids and packet.get("model_loaded") is False
                and packet.get("content_sha256") == digest({key: value for key, value in packet.items()
                                                             if key != "content_sha256"}),
                "complete selected processor-only case packet")
        for name in ("cohort", "original_jsonl", "source_packet", "producer"):
            require(binding(packet[name]["path"]) == packet[name], "case packet source drift")
    by_image = {int(record["image_id"]): record for record in _case_records(packet)}
    require(set(ids) <= set(by_image), "case packet lacks a selected new image")
    source = read(SOURCE_PACKET)
    require(source["schema"] == "positive_progress_matched_endpoint.packet.v1", "frozen source packet schema")
    config = copy.deepcopy(source["config"])
    require(config["backend"] == {"type": "hf", "hf": {"attn_implementation": "sdpa",
                                                         "patch_embed_linearization": "enabled"}}, "HF SDPA")
    require(config["model"]["dtype"] == "fp32"
            and Path(config["embedding_delta"]["path"]).resolve() == EMBEDDING_DELTA.resolve(),
            "frozen FP32/embedding delta")
    config["adapter"]["path"] = str(SOURCE_ADAPTER)
    config["generation"].update(batch_size=2, max_new_tokens=CAP, n=1,
                                 repetition_penalty=1.0, temperature=0.0, top_p=1.0)
    records = []
    for image_id in ids:
        frozen_case = by_image[image_id]
        case = frozen_case["case"]
        plan = case["image_plan"]
        path = Path(case["image_path"]).resolve(strict=True)
        require(file_hash(path) == plan["image_content_sha256"], "bound source image bytes")
        require(case["input_record"]["image_id"] == image_id, "case image ID")
        prompt_ids = frozen_case["prompt_token_ids"]
        require(isinstance(prompt_ids, list) and prompt_ids
                and all(type(token) is int for token in prompt_ids)
                and len(prompt_ids) == plan["backend_prompt_token_count"], "case prompt token IDs")
        require(plan["status"] == "ok" and plan["backend_projection_evidence_kind"]
                in ("hf_executed_tensors", "native_processor_executed_tensors")
                and plan["logical_transform_id"] == "identity", "native case image plan")
        records.append({"image_id": image_id, "example_id": frozen_case["example_id"],
                        "prompt_token_ids": prompt_ids, "prompt_token_ids_sha256": digest(prompt_ids),
                        "case": case, "golden": frozen_case["golden"], "source_split": frozen_case["split"],
                        "image_file": binding(path)})
    requests = request_plan(ids)
    identity = inspect_dora_adapter_payload(SOURCE_ADAPTER, config["model"]["base_model"])
    require(identity["fingerprint"] == ADAPTER_FINGERPRINT, "frozen S-final adapter fingerprint")
    manifest = {
        "schema": SCHEMA, "status": "frozen_ready_for_first_request",
        "scope": "new11 teacher discovery candidates; visual review and lead admission required",
        "cohort": {"new_image_ids": ids, "new_image_count": 11,
                   "old_image_ids": sorted(OLD_IMAGE_IDS), "total_image_count": 22},
        "policy": {"empty_assistant_prefix": True, "assistant_token_cap": CAP, "eos_token_id": EOS,
                   "repetition_penalty": 1.0, "top_p": 1.0, "top_k": 0,
                   "temperatures": [0.0, *TEMPERATURES], "seed_root": frozen.SEED_ROOT,
                   "fixed_grouping": "one_request_per_generate_call"},
        "model": {"config": config, "base_model": tree_binding(config["model"]["base_model"]),
                  "adapter": identity, "adapter_files": tree_binding(SOURCE_ADAPTER),
                  "embedding_delta": tree_binding(EMBEDDING_DELTA)},
        "sources": {"source_packet": binding(SOURCE_PACKET), "cohort": binding(cohort_path),
                    "case_packet": binding(case_packet_path), "producer": binding(Path(__file__)),
                    "frozen_acquisition": binding(Path(frozen.__file__)),
                    "input_materialization": binding(Path(input_materialization.__file__))},
        "records": records, "requests": requests, "request_count": 44,
        "execution": {"physical_gpus": list(GPUS), "first_request_id": requests[0]["request_id"],
                      "remaining_request_count": 43, "long_jobs_require_named_tmux": True,
                      "tmux_sessions": {"first": TMUX_FIRST, "remaining": TMUX_REMAINING}},
    }
    manifest["content_sha256"] = digest(manifest)
    publish(output / "manifest.json", manifest)
    validate_manifest(output / "manifest.json")
    return manifest


def validate_manifest(path: Path, *, verify_large_trees: bool = False) -> dict[str, Any]:
    from src.config.inference import InferConfig

    manifest = read(path)
    claimed = manifest.get("content_sha256")
    content = {key: value for key, value in manifest.items() if key != "content_sha256"}
    require(manifest.get("schema") == SCHEMA and claimed == digest(content), "manifest schema/content")
    require(manifest["status"] == "frozen_ready_for_first_request"
            and manifest["scope"] == "new11 teacher discovery candidates; visual review and lead admission required",
            "discovery-only scope")
    for source in manifest["sources"].values():
        require(binding(source["path"]) == source, "frozen input/producer changed")
    ids = new_image_ids(manifest["cohort"])
    require(manifest["cohort"] == {"new_image_ids": ids, "new_image_count": 11,
                                     "old_image_ids": sorted(OLD_IMAGE_IDS), "total_image_count": 22},
            "manifest cohort")
    validate_request_plan(manifest["requests"], ids)
    require(manifest["request_count"] == 44
            and manifest["execution"] == {"physical_gpus": list(GPUS),
                                          "first_request_id": manifest["requests"][0]["request_id"],
                                          "remaining_request_count": 43,
                                          "long_jobs_require_named_tmux": True,
                                          "tmux_sessions": {"first": TMUX_FIRST,
                                                            "remaining": TMUX_REMAINING}}, "manifest execution")
    require(manifest["requests"][0]["kind"] == "greedy", "first request greedy")
    require(manifest["policy"] == {"empty_assistant_prefix": True, "assistant_token_cap": CAP,
                                  "eos_token_id": EOS, "repetition_penalty": 1.0,
                                  "top_p": 1.0, "top_k": 0, "temperatures": [0.0, *TEMPERATURES],
                                  "seed_root": frozen.SEED_ROOT,
                                  "fixed_grouping": "one_request_per_generate_call"}, "decode policy")
    config = InferConfig.model_validate(manifest["model"]["config"])
    require(config.generation.batch_size == 2 and config.generation.max_new_tokens == CAP
            and config.model.dtype == "fp32" and config.backend.type == "hf"
            and config.backend.hf.attn_implementation == "sdpa"
            and Path(config.adapter.path).resolve() == SOURCE_ADAPTER.resolve()
            and Path(config.embedding_delta.path).resolve() == EMBEDDING_DELTA.resolve(), "frozen HF config")
    require(manifest["model"]["adapter"]["fingerprint"] == ADAPTER_FINGERPRINT
            and manifest["model"]["adapter_files"]["fingerprint"] == tree_binding(SOURCE_ADAPTER)["fingerprint"],
            "frozen adapter identity/files")
    require(len(manifest["records"]) == 11
            and [record["image_id"] for record in manifest["records"]] == ids
            and len({record["example_id"] for record in manifest["records"]}) == 11,
            "manifest record identities")
    for record in manifest["records"]:
        require(digest(record["prompt_token_ids"]) == record["prompt_token_ids_sha256"], "prompt IDs")
        require(binding(record["image_file"]["path"]) == record["image_file"]
                and record["case"]["image_plan"]["image_content_sha256"] == record["image_file"]["sha256"],
                "frozen media bytes")
    if verify_large_trees:
        for name in ("base_model", "adapter_files", "embedding_delta"):
            require(tree_binding(manifest["model"][name]["root"]) == manifest["model"][name],
                    f"model tree changed: {name}")
    return manifest


def _phase_requests(manifest: Mapping[str, Any], phase: str, shard: int, world_size: int) -> list[dict[str, Any]]:
    require(phase in ("first", "remaining") and world_size == (1 if phase == "first" else 8)
            and 0 <= shard < world_size, "phase/shard")
    first_id = manifest["execution"]["first_request_id"]
    selected = [request for request in manifest["requests"]
                if (request["request_id"] == first_id) == (phase == "first")]
    require(len(selected) == (1 if phase == "first" else 43), "phase request denominator")
    return selected[shard::world_size]


def _row_path(output: Path, request: Mapping[str, Any]) -> Path:
    return output / "rows" / (hashlib.sha256(request["request_id"].encode()).hexdigest() + ".json")


def worker(*, manifest_path: Path, output: Path, phase: str, shard: int,
           world_size: int, physical_gpu: int) -> None:
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from probes.source_rweak_row_cross.run import build_requests, native_record
    from src.adapters.dora import inspect_dora_adapter_payload
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    manifest = validate_manifest(manifest_path)
    require(output.resolve() == manifest_path.resolve().parent, "worker output/manifest identity")
    require(physical_gpu == GPUS[shard] and os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu)
            and torch.cuda.device_count() == 1, "worker exact one physical GPU")
    jobs = _phase_requests(manifest, phase, shard, world_size)
    terminal_path = output / "terminals" / f"{phase}-shard-{shard}.json"
    require(not terminal_path.exists(), "worker terminal collision")
    terminal: dict[str, Any] = {"schema": f"{SCHEMA}.terminal", "status": "running", "phase": phase,
                                "shard": shard, "world_size": world_size, "physical_gpu": physical_gpu,
                                "pid": os.getpid(), "manifest": binding(manifest_path),
                                "expected_requests": len(jobs), "completed_requests": 0,
                                "model_loads": 0, "model_forwards": 0, "image_forwards": 0,
                                "new_tokens": 0}
    started = time.monotonic()
    handles = []
    try:
        config = checkpoint_config(InferConfig.model_validate(manifest["model"]["config"]), SOURCE_ADAPTER)
        observed = inspect_dora_adapter_payload(SOURCE_ADAPTER, config.model.base_model)
        require(observed["fingerprint"] == manifest["model"]["adapter"]["fingerprint"], "live adapter")
        torch.cuda.set_device(torch.device("cuda:0"))
        torch.cuda.reset_peak_memory_stats()
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        require(identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"]
                and identity["effective_settings"]["observed_attn_implementation"] == "sdpa", "live FP32/SDPA")
        require(identity["model_identity"]["adapter"]["adapter_path"] == str(SOURCE_ADAPTER), "live S adapter")
        require(identity["model_identity"]["base"]["path"] == config.model.base_model
                and identity["model_identity"]["embedding_delta"]["status"] == "loaded"
                and identity["model_identity"]["embedding_delta"]["identity"]["delta_path"] == str(EMBEDDING_DELTA),
                "live base/2444 embedding delta")
        publish(output / "model" / f"{phase}-shard-{shard}.json", identity)
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)
        handles.append(qwen.model.register_forward_pre_hook(
            lambda *_: terminal.__setitem__("model_forwards", terminal["model_forwards"] + 1)))
        visual = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visual) == 1, "single visual module")
        handles.append(visual[0].register_forward_pre_hook(
            lambda *_: terminal.__setitem__("image_forwards", terminal["image_forwards"] + 1)))
        records = {record["image_id"]: record for record in manifest["records"]}
        for job in jobs:
            record = records[job["image_id"]]
            case = input_materialization.materialize_bound_single_image_case(
                record["case"], manifest["model"]["config"])
            requests, _ = build_requests(qwen, manifest["model"]["config"], [case])
            batch = prepare_native_inputs(qwen.processor, requests, device="cuda:0", record_media_identity=True)
            plan = record["case"]["image_plan"]
            require(list(batch.prompt_token_ids[0]) == record["prompt_token_ids"], "live prompt tokens")
            require(batch.media_sha256[0] == plan["executed_media_sha256"]
                    and list(batch.image_grids[0]) == plan["observed_image_grid_thw"], "live media/grid")
            policy = NativeGenerationPolicy(temperature=float(job["temperature"]), top_p=1.0,
                                            top_k=0, repetition_penalty=1.0,
                                            use_model_defaults=False)
            tick = time.monotonic()
            with torch.inference_mode():
                generated, = generate_continuations(
                    qwen.model, batch, extensions=[[]], budgets=[CAP], eos_token_id=EOS,
                    pad_token_id=qwen.tokenizer.pad_token_id, policy=policy,
                    trace="none", seed=job["seed"])
            ids = list(generated.token_ids)
            frozen._checked_terminal(ids, generated.stop_reason)
            decoded = qwen.tokenizer.decode(ids, skip_special_tokens=False)
            parsed = native_record(decoded, record["case"], record["golden"], generated.stop_reason)
            payload = {"schema": f"{SCHEMA}.row", "request": dict(job),
                       "manifest_sha256": manifest["content_sha256"], "example_id": record["example_id"],
                       "image_id": record["image_id"], "source_split": record["source_split"],
                       "empty_assistant_prefix": True, "assistant_token_cap": CAP,
                       "prompt_token_ids": record["prompt_token_ids"],
                       "prompt_token_ids_sha256": record["prompt_token_ids_sha256"],
                       "executed_media_sha256": batch.media_sha256[0],
                       "observed_image_grid_thw": list(batch.image_grids[0]),
                       "generated_token_ids": ids, "generated_token_ids_sha256": digest(ids),
                       "raw_decode_text": decoded, "decode_stop_reason": generated.stop_reason,
                       "generated_token_count": len(ids), "parsed": parsed,
                       "timing": {"generation_seconds": time.monotonic() - tick},
                       "model_receipt": binding(output / "model" / f"{phase}-shard-{shard}.json")}
            publish(_row_path(output, job), payload)
            terminal["completed_requests"] += 1
            terminal["new_tokens"] += len(ids)
        require(terminal["completed_requests"] == terminal["expected_requests"], "worker request denominator")
        terminal.update(status="completed", exit_code=0)
    except BaseException as exc:
        terminal.update(status="failed", exit_code=1,
                        error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        terminal.update(elapsed_seconds=time.monotonic() - started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        publish(terminal_path, terminal)


def validate_result_payload(payload: Mapping[str, Any], request: Mapping[str, Any],
                            record: Mapping[str, Any], *, decode) -> None:
    require(payload.get("schema") == f"{SCHEMA}.row" and payload.get("request") == dict(request), "result request")
    require(payload.get("manifest_sha256") and payload.get("example_id") == record["example_id"]
            and payload.get("image_id") == record["image_id"], "result image/manifest")
    require(payload.get("empty_assistant_prefix") is True and payload.get("assistant_token_cap") == CAP,
            "result prefix/budget")
    require(payload.get("prompt_token_ids") == record["prompt_token_ids"]
            and digest(payload["prompt_token_ids"]) == payload.get("prompt_token_ids_sha256"), "result prompt IDs")
    ids = payload.get("generated_token_ids")
    require(isinstance(ids, list) and digest(ids) == payload.get("generated_token_ids_sha256"), "result generated IDs")
    frozen._checked_terminal(ids, payload.get("decode_stop_reason"))
    require(payload.get("generated_token_count") == len(ids) and decode(ids) == payload.get("raw_decode_text"),
            "result token/text")
    require(payload.get("executed_media_sha256") == record["case"]["image_plan"]["executed_media_sha256"],
            "result media")
    require(payload.get("observed_image_grid_thw") == record["case"]["image_plan"]["observed_image_grid_thw"],
            "result grid")


def readback(*, manifest_path: Path, output: Path, phase: str) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from probes.source_rweak_row_cross.run import native_record

    manifest = validate_manifest(manifest_path)
    require(output.resolve() == manifest_path.resolve().parent and phase in ("first", "final"), "readback phase/root")
    selected = [request for request in manifest["requests"]
                if phase == "final" or request["request_id"] == manifest["execution"]["first_request_id"]]
    require(len(selected) == (1 if phase == "first" else 44), "readback denominator")
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"]["base_model"]["root"], local_files_only=True)
    records = {record["image_id"]: record for record in manifest["records"]}
    rows = []
    for request in selected:
        payload = read(_row_path(output, request))
        record = records[request["image_id"]]
        require(payload["manifest_sha256"] == manifest["content_sha256"], "row manifest identity")
        validate_result_payload(payload, request, record,
                                decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False))
        require(native_record(payload["raw_decode_text"], record["case"], record["golden"],
                              payload["decode_stop_reason"]) == payload["parsed"], "cold native parser replay")
        rows.append(payload)
    require(len({row["request"]["request_id"] for row in rows}) == len(selected), "cold unique requests")
    receipt = {"schema": f"{SCHEMA}.{phase}_readback", "status": "passed",
               "scope": "review-only candidate discovery; no automatic teacher/evaluation admission",
               "manifest": binding(manifest_path), "request_count": len(rows),
               "request_ids_sha256": digest([row["request"]["request_id"] for row in rows]),
               "generated_tokens": sum(row["generated_token_count"] for row in rows),
               "stop_counts": {stop: sum(row["decode_stop_reason"] == stop for row in rows)
                               for stop in ("im_end", "length")},
               "parser": {"valid_predictions": sum(row["parsed"]["valid_prediction_count"] for row in rows),
                          "dropped_predictions": sum(row["parsed"]["dropped_prediction_count"] for row in rows)},
               "geometry_invalid": geometry_invalid_count(rows),
               "cold_token_text_and_parser_replay": True}
    if phase == "final":
        merged = output / "rows.jsonl"
        require(not merged.exists(), "merged discovery rows collision")
        with merged.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        require(sum(1 for _ in merged.open()) == 44, "merged discovery denominator")
    publish(output / ("first-readback.json" if phase == "first" else "result.json"), receipt)
    return receipt


def build_review_packets(*, manifest_path: Path, output: Path, gt_index_path: Path) -> dict[str, Any]:
    """Render every raw candidate as a per-proposal overlay/crop for human review."""
    from PIL import Image
    from probes.training_set_completion import review_packets as visuals

    manifest = validate_manifest(manifest_path)
    require(output.resolve() == manifest_path.resolve().parent, "review packet root/manifest")
    complete = read(output / "result.json")
    require(complete.get("status") == "passed" and complete.get("request_count") == 44
            and complete.get("manifest") == binding(manifest_path), "44 cold-readback rows required")
    rows_path = output / "rows.jsonl"
    rows = visuals._load_rows(rows_path)
    ids = manifest["cohort"]["new_image_ids"]
    by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in ids}
    for row in rows:
        require(row["request"] in manifest["requests"]
                and row["manifest_sha256"] == manifest["content_sha256"], "review source request/manifest")
        by_image[row["image_id"]].append(row)
    require(all(len(by_image[image_id]) == 4 for image_id in ids), "four review policies/new image")
    refs: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in ids}
    with gt_index_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("image_id") in refs:
                refs[record["image_id"]].append(record)
    require(all(refs.values()), "GT visual review references/new image")
    destination = output / "review-packets-v1"
    require(not destination.exists(), "review packet root exists")
    destination.mkdir(parents=True)
    records = {record["image_id"]: record for record in manifest["records"]}
    index = []
    total_raw = total_crops = 0
    for image_id in ids:
        record = records[image_id]
        image_path = Path(record["image_file"]["path"])
        require(binding(image_path) == record["image_file"], "review source image bytes")
        golden = record["golden"]
        gt = []
        for owner in golden["gt"]:
            source = owner["metadata"].get("source", {})
            gt.append({"owner_id": owner["object_id"], "category_name": owner["description"],
                       "is_crowd": bool(source.get("iscrowd", False)),
                       "coord_bins_1000": owner["bbox"],
                       "bbox_pixel_xyxy": visuals._pixel_from_bins(owner["bbox"],
                                                                      golden["image_width"],
                                                                      golden["image_height"])})
        folder = destination / f"image-{image_id:012d}"
        folder.mkdir()
        policies = []
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        require([image.width, image.height] == [golden["image_width"], golden["image_height"]],
                "visual image dimensions")
        for row in sorted(by_image[image_id], key=visuals._request_order):
            parsed = row["parsed"]
            require(parsed["image_path"] == str(image_path)
                    and [parsed["image_width"], parsed["image_height"]] == [image.width, image.height],
                    "parser image identity")
            proposals = [visuals._proposal_from_valid(value, row=row, index=index,
                                                       gt=gt)
                         for index, value in enumerate(parsed["pred"])]
            proposals.extend(visuals._proposal_from_drop(value, row=row, gt=gt)
                             for value in parsed.get("dropped_predictions", []))
            # The frozen renderer assumes a numeric second-nearest value even
            # when a parser drop has no drawable coordinates.
            for proposal in proposals:
                if proposal["second_nearest_gt_iou"] is None:
                    require(proposal["bbox_pixel_xyxy"] is None, "missing IoU on drawable candidate")
                    proposal["second_nearest_gt_iou"] = 0.0
            visuals._flags(proposals)
            policy = visuals._policy(row)
            overlay = folder / f"pred-overlay-{policy}.png"
            visuals._render_policy(image, gt, proposals, overlay, image_id, policy)
            for proposal in proposals:
                box = proposal["bbox_pixel_xyxy"]
                if box is not None:
                    # Every drawable candidate is given its own crop, including
                    # those overlapping GT; similarity is a viewing aid only.
                    name = hashlib.sha256(proposal["proposal_id"].encode()).hexdigest()[:16]
                    crop = folder / "crops" / f"{policy}-{name}.png"
                    crop.parent.mkdir(exist_ok=True)
                    visuals._crop(image, box, crop)
                    proposal["crop_path"] = str(crop.resolve())
                    total_crops += 1
                proposal["source_row"] = {
                    "request_id": row["request"]["request_id"],
                    "line_number_1_based": row["_source_line_number"],
                    "line_sha256": row["_source_line_sha256"]}
            total_raw += len(proposals)
            policies.append({"policy": policy, "request": row["request"],
                             "overlay_path": str(overlay.resolve()), "proposals": proposals,
                             "raw_visible_count": len(proposals),
                             "decode_stop_reason": row["decode_stop_reason"]})
        packet = {"schema": f"{SCHEMA}.review_packet", "status": "candidate_pending_native_visual_review",
                  "image_id": image_id, "original_image": record["image_file"],
                  "gt_reference_source": binding(gt_index_path), "gt_visual_references": refs[image_id],
                  "gt": gt, "policies": policies,
                  "boundary": "Every prediction and parser-dropped raw proposal is a candidate."
                              " Human original/overlay/crop review and lead admission decide physical owner/class."}
        publish(folder / "packet.json", packet)
        index.append({"image_id": image_id, "packet": binding(folder / "packet.json"),
                      "policy_count": len(policies),
                      "raw_candidate_count": sum(policy["raw_visible_count"] for policy in policies)})
    receipt = {"schema": f"{SCHEMA}.review_receipt", "status": "candidate_packets_ready_pending_visual_review",
               "manifest": binding(manifest_path), "rows": binding(rows_path),
               "cold_readback": binding(output / "result.json"),
               "cohort_gt_visual_index": binding(gt_index_path),
               "render_helpers": binding(Path(visuals.__file__)), "producer": binding(Path(__file__)),
               "image_count": len(index), "request_count": 44,
               "raw_candidate_count": total_raw, "crop_count": total_crops,
               "packets": index, "no_truth_or_teacher_admissions": True}
    publish(destination / "receipt.json", receipt)
    return receipt


def _wait_child(process: subprocess.Popen[Any], completions: queue.Queue[dict[str, Any]]) -> None:
    try:
        code: int | str = process.wait()
        error = None
    except BaseException as exc:
        code = "wait_error"
        error = f"{type(exc).__name__}: {exc}"
    completions.put({"pid": process.pid, "exit_code": code, "wait_error": error,
                     "completed_at": time.time()})


def worker_commands(*, manifest_path: Path, output: Path, phase: str) -> list[dict[str, Any]]:
    manifest = validate_manifest(manifest_path)
    require(output.resolve() == manifest_path.resolve().parent, "launch output/manifest identity")
    world_size = 1 if phase == "first" else 8
    require(phase in ("first", "remaining"), "launch phase")
    specs = []
    for shard in range(world_size):
        gpu = GPUS[shard]
        specs.append({"shard": shard, "physical_gpu": gpu,
                      "command": [sys.executable, "-m", "probes.training_set_completion.coco22_acquisition",
                                  "worker", "--manifest", str(manifest_path.resolve()),
                                  "--output", str(output.resolve()), "--phase", phase,
                                  "--shard", str(shard), "--world-size", str(world_size),
                                  "--physical-gpu", str(gpu)],
                      "log": str(output / "logs" / f"{phase}-shard-{shard}.log"),
                      "expected_requests": len(_phase_requests(manifest, phase, shard, world_size))})
    return specs


def tmux_command(*, manifest_path: Path, output: Path, phase: str) -> list[str]:
    require(phase in ("first", "remaining"), "tmux phase")
    worker_commands(manifest_path=manifest_path, output=output, phase=phase)
    session = TMUX_FIRST if phase == "first" else TMUX_REMAINING
    controller = [sys.executable, "-m", "probes.training_set_completion.coco22_acquisition",
                  "launch", "--manifest", str(manifest_path.resolve()),
                  "--output", str(output.resolve()), "--phase", phase]
    return ["tmux", "new-session", "-d", "-s", session, shlex.join(controller)]


def launch(*, manifest_path: Path, output: Path, phase: str) -> dict[str, Any]:
    specs = worker_commands(manifest_path=manifest_path, output=output, phase=phase)
    if phase == "remaining":
        first = read(output / "first-readback.json")
        require(first["status"] == "passed" and first["request_count"] == 1
                and first["manifest"] == binding(manifest_path), "remaining requires first real request readback")
    require(not (output / f"{phase}-launch.json").exists()
            and not (output / f"{phase}-exits.json").exists()
            and all(not Path(spec["log"]).exists() and not (output / "terminals" /
                         f"{phase}-shard-{spec['shard']}.json").exists() for spec in specs),
            "existing launch/log/terminal: reconcile before relaunch")
    # Immutable command packet exists before the first child can be started.
    publish(output / f"{phase}-launch.json", {"schema": f"{SCHEMA}.launch",
                                              "status": "commands_frozen",
                                              "phase": phase, "manifest": binding(manifest_path),
                                              "commands": specs})
    completions: queue.Queue[dict[str, Any]] = queue.Queue()
    processes: dict[int, tuple[subprocess.Popen[Any], Any]] = {}
    started = []
    start_error = None
    try:
        for spec in specs:
            path = Path(spec["log"])
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open("x")
            try:
                process = subprocess.Popen(
                    spec["command"], cwd=Path(__file__).resolve().parents[2],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(spec["physical_gpu"]),
                         "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"},
                    stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            processes[process.pid] = (process, log)
            started.append({"shard": spec["shard"], "physical_gpu": spec["physical_gpu"],
                            "pid": process.pid, "command": spec["command"], "log": spec["log"],
                            "expected_requests": spec["expected_requests"]})
            threading.Thread(target=_wait_child, args=(process, completions),
                             name=f"coco22-discovery-wait-{process.pid}", daemon=True).start()
    except BaseException as exc:
        start_error = f"{type(exc).__name__}: {exc}"
    publish(output / f"{phase}-started.json", {"schema": f"{SCHEMA}.started",
                                               "phase": phase, "commands_started": started,
                                               "start_error": start_error})
    exits = []
    for _ in started:
        completion = completions.get()
        _, log = processes[completion["pid"]]
        log.close()
        exits.append(completion)
    publish(output / f"{phase}-exits.json", {"schema": f"{SCHEMA}.exits", "phase": phase,
                                             "start_error": start_error,
                                             "exits": sorted(exits, key=lambda row: row["pid"])})
    require(start_error is None and len(started) == len(specs)
            and all(item["exit_code"] == 0 and item["wait_error"] is None for item in exits),
            "discovery workers failed; inspect preserved exits/terminals/logs")
    return {"schema": f"{SCHEMA}.launch_result", "status": "completed", "phase": phase,
            "request_count": sum(spec["expected_requests"] for spec in specs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-cases", "prepare", "verify", "commands",
                                           "launch", "worker", "readback", "review-packets"))
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--original-jsonl", type=Path)
    parser.add_argument("--case-output", type=Path)
    parser.add_argument("--reuse-case-packet", type=Path)
    parser.add_argument("--exclude-image-id", action="append", type=int, default=[])
    parser.add_argument("--case-packet", type=Path)
    parser.add_argument("--gt-review-index", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("first", "remaining", "final"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--verify-large-trees", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-cases":
        require(args.cohort is not None and args.original_jsonl is not None and args.case_output is not None,
                "case preparation requires cohort/original-jsonl/case-output")
        value = prepare_cases(cohort_path=args.cohort, original_jsonl=args.original_jsonl,
                              output=args.case_output, exclude_image_ids=args.exclude_image_id,
                              reuse_case_packet_path=args.reuse_case_packet)
    elif args.command == "prepare":
        require(args.cohort is not None and args.case_packet is not None, "cohort/case packet required")
        value = prepare(cohort_path=args.cohort, case_packet_path=args.case_packet, output=args.output)
    elif args.command == "verify":
        value = validate_manifest(args.manifest, verify_large_trees=args.verify_large_trees)
    elif args.command == "commands":
        require(args.phase in ("first", "remaining"), "commands phase required")
        value = {"tmux_command": tmux_command(manifest_path=args.manifest, output=args.output,
                                                phase=args.phase),
                 "worker_commands": worker_commands(manifest_path=args.manifest,
                                                     output=args.output, phase=args.phase)}
    elif args.command == "launch":
        require(args.phase in ("first", "remaining"), "launch phase required")
        value = launch(manifest_path=args.manifest, output=args.output, phase=args.phase)
    elif args.command == "worker":
        require(args.phase in ("first", "remaining") and args.shard is not None
                and args.world_size is not None and args.physical_gpu is not None, "worker arguments")
        worker(manifest_path=args.manifest, output=args.output, phase=args.phase, shard=args.shard,
               world_size=args.world_size, physical_gpu=args.physical_gpu)
        return
    elif args.command == "review-packets":
        require(args.gt_review_index is not None, "review packet GT index required")
        value = build_review_packets(manifest_path=args.manifest, output=args.output,
                                     gt_index_path=args.gt_review_index)
    else:
        require(args.phase in ("first", "final"), "readback phase required")
        value = readback(manifest_path=args.manifest, output=args.output, phase=args.phase)
    print(json.dumps(value if args.command == "commands" else
                     {"schema": value.get("schema"), "status": value.get("status"),
                      "request_count": value.get("request_count")}, sort_keys=True))


if __name__ == "__main__":
    main()

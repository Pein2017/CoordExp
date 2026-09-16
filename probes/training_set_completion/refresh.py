"""Bounded stage-02 native self-rollout acquisition from the first-fit step-16 adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from probes.training_set_completion.acquisition import IMAGE_IDS, binding, digest, file_hash, publish, read, require

BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
ROOT = BASE / "stage02-refresh-v1"
ACQUISITION = BASE / "stage01-acquisition-v1-retry1-config-batch2"
PARENT = BASE / "first-fit-v1" / "training" / "checkpoints" / "step-00016" / "adapter"
PARENT_STATE = BASE / "first-fit-v1" / "training" / "checkpoints" / "step-00016" / "state.pt"
PARENT_READBACK = BASE / "first-fit-v1" / "readback-step-16.json"
CAP, EOS, SEED_ROOT, WORKER_SECONDS = 3084, 151645, 2026091402, 1800
TEMPERATURES, GPUS = (0.1, 0.3, 0.7), tuple(range(8))
SCHEMA = "training_set_completion.stage02_refresh.v1"


def sample_seed(image_id: int, temperature: float) -> int:
    return int(digest({"root": SEED_ROOT, "image_id": image_id, "temperature": temperature})[:8], 16) & 0x7FFF_FFFF


def request_id(image_id: int, temperature: float | None) -> str:
    suffix = "greedy" if temperature is None else f"sample-t{str(temperature).replace('.', 'p')}"
    return f"stage02:image-{image_id:012d}:{suffix}"


def request_plan() -> list[dict[str, Any]]:
    rows = []
    for image_id in IMAGE_IDS:
        rows.append({"request_id": request_id(image_id, None), "image_id": image_id, "kind": "greedy", "temperature": 0.0, "seed": SEED_ROOT, "grouping": "one_request_per_generate_call"})
        rows.extend({"request_id": request_id(image_id, temp), "image_id": image_id, "kind": "sample", "temperature": temp, "seed": sample_seed(image_id, temp), "grouping": "one_request_per_generate_call"} for temp in TEMPERATURES)
    validate_request_plan(rows)
    return rows


def validate_request_plan(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == 44 and len({row["request_id"] for row in rows}) == 44, "refresh request denominator")
    require({int(row["image_id"]) for row in rows} == set(IMAGE_IDS) and all(sum(row["image_id"] == image for row in rows) == 4 for image in IMAGE_IDS), "refresh image/policy denominator")
    require(all(row["grouping"] == "one_request_per_generate_call" for row in rows), "coherent route grouping")
    greedy = [row for row in rows if row["kind"] == "greedy"]
    samples = [row for row in rows if row["kind"] == "sample"]
    require(len(greedy) == 11 and all(row["temperature"] == 0.0 and row["seed"] == SEED_ROOT for row in greedy), "greedy policy/seed")
    require(len(samples) == 33 and len({row["seed"] for row in samples}) == 33 and all(row["temperature"] in TEMPERATURES for row in samples), "sample policy/seeds")


def _source_manifest() -> tuple[dict[str, Any], Path]:
    path = ACQUISITION / "manifest.json"
    source = read(path)
    require(source.get("schema") == "training_set_completion.stage01_acquisition.v1" and source.get("status") == "frozen_ready_for_smoke", "bound acquisition manifest")
    require([int(row["image_id"]) for row in source["records"]] == list(IMAGE_IDS), "bound acquisition cohort")
    return source, path


def prepare(output: Path = ROOT) -> dict[str, Any]:
    """Freeze parent, original source image/prompt records, and the 44-route plan."""
    from src.adapters.dora import inspect_dora_adapter_payload

    require(not output.exists(), "refresh root already exists")
    source, source_path = _source_manifest()
    rows_path = ACQUISITION / "rows.jsonl"
    require(binding(rows_path) == read(BASE / "stage01-review-extraction-v2" / "source-manifest.json")["acquisition"], "acquisition rows changed")
    parent = inspect_dora_adapter_payload(PARENT, source["model"]["config"]["model"]["base_model"])
    saved_readback = read(PARENT_READBACK)
    expected = {int(row["image_id"]): row for row in saved_readback["rows"]}
    require(set(expected) == set(IMAGE_IDS) and saved_readback["policy"] == {"empty_assistant_prefix": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "assistant_token_cap": CAP}, "parent greedy readback denominator/policy")
    manifest = {"schema": SCHEMA, "status": "candidate_ready", "seed_root": SEED_ROOT, "parent": {"adapter": parent, "state": binding(PARENT_STATE), "saved_greedy_readback": binding(PARENT_READBACK), "greedy_reference_identity": "readback is all-image greedy by its bound NativeGenerationPolicy; each route_id remains the inherited training-source route identity and is compared by image_id"}, "sources": {"acquisition_manifest": binding(source_path), "acquisition_rows": binding(rows_path), "producer": binding(Path(__file__))}, "model_config": source["model"]["config"], "records": source["records"], "requests": request_plan(), "runtime": {"assistant_token_cap": CAP, "eos_token_id": EOS, "per_worker_wall_seconds": WORKER_SECONDS, "physical_gpus": list(GPUS), "empty_assistant_prefix": True, "precision": "fp32", "attention": "sdpa", "repetition_penalty": 1.0, "top_p": 1.0, "top_k": 0}, "content_sha256": None}
    manifest["content_sha256"] = digest({key: value for key, value in manifest.items() if key != "content_sha256"})
    validate_manifest(manifest)
    publish(output / "manifest.json", manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    require(manifest.get("schema") == SCHEMA and manifest.get("content_sha256") == digest({key: value for key, value in manifest.items() if key != "content_sha256"}), "refresh manifest identity")
    require(manifest.get("seed_root") == SEED_ROOT and manifest.get("status") == "candidate_ready", "refresh status/seed")
    validate_request_plan(manifest.get("requests", []))
    runtime = manifest.get("runtime", {})
    require(runtime == {"assistant_token_cap": CAP, "eos_token_id": EOS, "per_worker_wall_seconds": WORKER_SECONDS, "physical_gpus": list(GPUS), "empty_assistant_prefix": True, "precision": "fp32", "attention": "sdpa", "repetition_penalty": 1.0, "top_p": 1.0, "top_k": 0}, "refresh runtime identity")
    for label in ("acquisition_manifest", "acquisition_rows", "producer"):
        require(binding(manifest["sources"][label]["path"]) == manifest["sources"][label], f"{label} changed")
    require(binding(manifest["parent"]["state"]["path"]) == manifest["parent"]["state"] and binding(manifest["parent"]["saved_greedy_readback"]["path"]) == manifest["parent"]["saved_greedy_readback"], "parent evidence changed")
    require(manifest["parent"]["adapter"].get("root") == str(PARENT), "parent adapter identity")


def _row_path(output: Path, request: Mapping[str, Any]) -> Path:
    return output / "rows" / (hashlib.sha256(str(request["request_id"]).encode()).hexdigest() + ".json")


def _jobs(manifest: Mapping[str, Any], *, phase: str, shard: int = 0, world_size: int = 1) -> list[dict[str, Any]]:
    rows = list(manifest["requests"])
    require(phase in ("preflight", "main"), "refresh phase")
    if phase == "preflight":
        require(shard == 0 and world_size == 1, "preflight worker identity")
        return [rows[0]]
    remainder = rows[1:]
    require(0 <= shard < world_size == 8, "main worker identity")
    return remainder[shard::world_size]


def _checked_terminal(ids: Sequence[int], stop: str) -> None:
    require(ids and len(ids) <= CAP and all(type(token) is int and token >= 0 for token in ids), "generated token IDs")
    require((stop == "im_end" and ids[-1] == EOS and EOS not in ids[:-1]) or (stop == "length" and len(ids) == CAP and EOS not in ids), "generated terminal")


def worker(*, manifest_path: Path, output: Path, phase: str, shard: int, world_size: int, physical_gpu: int) -> None:
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from probes.native_owner_scale.evaluation import _candidate_materialized_case
    from src.inference.bound_requests import build_bound_native_requests as build_requests
    from src.eval.native_rows import native_detection_record as native_record
    from src.adapters.dora import inspect_dora_adapter_payload
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    manifest = read(manifest_path); validate_manifest(manifest)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu) and torch.cuda.device_count() == 1, "worker GPU isolation")
    jobs = _jobs(manifest, phase=phase, shard=shard, world_size=world_size)
    terminal_path = output / "terminals" / f"{phase}-shard-{shard}.json"
    require(not terminal_path.exists(), "terminal collision")
    terminal: dict[str, Any] = {"schema": f"{SCHEMA}.terminal", "status": "running", "phase": phase, "shard": shard, "world_size": world_size, "physical_gpu": physical_gpu, "manifest": binding(manifest_path), "expected_requests": len(jobs), "completed_requests": 0, "model_loads": 0, "model_forwards": 0, "image_forwards": 0, "new_tokens": 0}
    started, handles = time.monotonic(), []
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("refresh worker wall bound")))
        signal.alarm(WORKER_SECONDS)
        config = checkpoint_config(InferConfig.model_validate(manifest["model_config"]), PARENT)
        require(inspect_dora_adapter_payload(PARENT, config.model.base_model) == manifest["parent"]["adapter"], "live parent adapter differs")
        torch.cuda.set_device(torch.device("cuda:0")); torch.cuda.reset_peak_memory_stats()
        qwen, identity = load_policy(config, device=torch.device("cuda:0")); terminal["model_loads"] = 1
        require(identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"] and identity["effective_settings"]["observed_attn_implementation"] == "sdpa", "live FP32/SDPA")
        publish(output / "model" / f"{phase}-shard-{shard}.json", identity)
        qwen.model.eval()
        handles.append(qwen.model.register_forward_pre_hook(lambda *_: terminal.__setitem__("model_forwards", terminal["model_forwards"] + 1)))
        visual = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]; require(len(visual) == 1, "single visual module")
        handles.append(visual[0].register_forward_pre_hook(lambda *_: terminal.__setitem__("image_forwards", terminal["image_forwards"] + 1)))
        by_image = {int(row["image_id"]): row for row in manifest["records"]}
        for job in jobs:
            record = by_image[int(job["image_id"])]
            case = _candidate_materialized_case(record["case"], manifest["model_config"])
            requests, _ = build_requests(qwen, manifest["model_config"], [case])
            batch = prepare_native_inputs(qwen.processor, requests, device="cuda:0", record_media_identity=True)
            plan = record["case"]["image_plan"]
            require(list(batch.prompt_token_ids[0]) == record["prompt_token_ids"] and batch.media_sha256[0] == plan["executed_media_sha256"] and list(batch.image_grids[0]) == plan["observed_image_grid_thw"], "bound prompt/media/grid differs")
            policy = NativeGenerationPolicy(temperature=float(job["temperature"]), top_p=1.0, top_k=0, repetition_penalty=1.0, use_model_defaults=False)
            tick = time.monotonic()
            with torch.inference_mode():
                generated, = generate_continuations(qwen.model, batch, extensions=[[]], budgets=[CAP], eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace="none", seed=job["seed"])
            ids = list(generated.token_ids); _checked_terminal(ids, generated.stop_reason)
            text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
            payload = {"schema": f"{SCHEMA}.row", "request": dict(job), "manifest_sha256": manifest["content_sha256"], "example_id": record["example_id"], "image_id": record["image_id"], "source_split": record["source_split"], "empty_assistant_prefix": True, "assistant_token_cap": CAP, "prompt_token_ids": record["prompt_token_ids"], "prompt_token_ids_sha256": digest(record["prompt_token_ids"]), "executed_media_sha256": batch.media_sha256[0], "observed_image_grid_thw": list(batch.image_grids[0]), "generated_token_ids": ids, "generated_token_ids_sha256": digest(ids), "raw_decode_text": text, "decode_stop_reason": generated.stop_reason, "generated_token_count": len(ids), "parsed": native_record(text, record["case"], record["golden"], generated.stop_reason), "timing": {"generation_seconds": time.monotonic() - tick}, "model_receipt": binding(output / "model" / f"{phase}-shard-{shard}.json")}
            publish(_row_path(output, job), payload); terminal["completed_requests"] += 1; terminal["new_tokens"] += len(ids)
        require(terminal["completed_requests"] == terminal["expected_requests"], "worker denominator")
        terminal.update(status="completed", exit_code=0)
    except BaseException as exc:
        terminal.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm)
        for handle in handles: handle.remove()
        terminal.update(elapsed_seconds=time.monotonic() - started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0, peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        publish(terminal_path, terminal)


def _compare_greedy(row: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    same = row["generated_token_ids"] == expected["generated_token_ids"]
    return {"request_id": row["request"]["request_id"], "image_id": row["image_id"], "expected_route_id": expected["route_id"], "same_token_ids": same, "observed_generated_token_ids_sha256": row["generated_token_ids_sha256"], "expected_generated_token_ids_sha256": expected["generated_token_ids_sha256"]}


def _collect(manifest_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from src.eval.native_rows import native_detection_record as native_record

    manifest = read(manifest_path); validate_manifest(manifest)
    by_image = {int(row["image_id"]): row for row in manifest["records"]}
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_config"]["model"]["base_model"], local_files_only=True)
    rows = []
    for request in manifest["requests"]:
        row = read(_row_path(output, request)); record = by_image[int(request["image_id"])]
        require(row["request"] == request and row["manifest_sha256"] == manifest["content_sha256"] and row["prompt_token_ids"] == record["prompt_token_ids"], "row/request identity")
        _checked_terminal(row["generated_token_ids"], row["decode_stop_reason"])
        require(digest(row["generated_token_ids"]) == row["generated_token_ids_sha256"] and tokenizer.decode(row["generated_token_ids"], skip_special_tokens=False) == row["raw_decode_text"], "literal token/text identity")
        require(row["parsed"] == native_record(row["raw_decode_text"], record["case"], record["golden"], row["decode_stop_reason"]), "cold parser identity")
        rows.append(row)
    expected = {row["image_id"]: row for row in read(PARENT_READBACK)["rows"]}
    greedy = [_compare_greedy(row, expected[row["image_id"]]) for row in rows if row["request"]["kind"] == "greedy"]
    require(len(greedy) == 11, "greedy comparison denominator")
    value = {"schema": f"{SCHEMA}.result", "status": "candidate_ready" if all(row["same_token_ids"] for row in greedy) else "technical_greedy_mismatch", "manifest": binding(manifest_path), "request_count": len(rows), "request_ids_sha256": digest([row["request"]["request_id"] for row in rows]), "generated_tokens": sum(row["generated_token_count"] for row in rows), "stop_counts": {stop: sum(row["decode_stop_reason"] == stop for row in rows) for stop in ("im_end", "length")}, "parser": {"valid_predictions": sum(row["parsed"]["valid_prediction_count"] for row in rows), "dropped_predictions": sum(row["parsed"]["dropped_prediction_count"] for row in rows), "geometry_invalid": sum(drop.get("reason") == "geometry_invalid" for row in rows for drop in row["parsed"]["dropped_predictions"])}, "greedy_parent_comparison": greedy, "literal_token_text_and_parser_replay": True}
    publish(output / "result.json", value)
    with (output / "rows.jsonl").open("x") as stream:
        for row in rows: stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return value


def _stop_owned_process(process: subprocess.Popen[Any]) -> int | str | None:
    """Best-effort terminate/reap without letting cleanup mask the caller failure."""
    code = process.poll()
    if code is not None:
        return code
    try:
        process.terminate()
    except ProcessLookupError:
        return process.poll()
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            return process.poll()
        try:
            return process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            return "kill_timeout"


def controller(*, manifest_path: Path, output: Path) -> None:
    """Run preflight once, then the remaining 43 routes on eight isolated workers."""
    manifest = read(manifest_path); validate_manifest(manifest)
    terminal = {"schema": f"{SCHEMA}.controller", "status": "running", "manifest": binding(manifest_path), "started_unix": time.time()}
    publish(output / "controller-start.json", terminal)
    started = time.monotonic()
    owned: list[tuple[subprocess.Popen[Any], Any, dict[str, Any]]] = []
    try:
        def spawn(phase: str, shard: int, world: int, gpu: int) -> tuple[subprocess.Popen[Any], Any, dict[str, Any]]:
            command = [sys.executable, "-m", "probes.training_set_completion.refresh", "worker", "--manifest", str(manifest_path), "--output", str(output), "--phase", phase, "--shard", str(shard), "--world-size", str(world), "--physical-gpu", str(gpu)]
            log_path = output / "logs" / f"{phase}-shard-{shard}.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("x")
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"})
            item = (process, log, {"phase": phase, "shard": shard, "gpu": gpu, "pid": process.pid, "command": command, "log": str(log_path)})
            owned.append(item)
            return item
        preflight, log, spec = spawn("preflight", 0, 1, 0)
        try:
            code = preflight.wait(timeout=WORKER_SECONDS)
        except subprocess.TimeoutExpired:
            _stop_owned_process(preflight)
            raise
        finally:
            log.close()
        require(code == 0, "native preflight worker failed")
        preflight_row = read(_row_path(output, manifest["requests"][0])); expected = read(PARENT_READBACK)["rows"][0]
        comparison = _compare_greedy(preflight_row, expected); publish(output / "preflight.json", {"schema": f"{SCHEMA}.preflight", "status": "passed" if comparison["same_token_ids"] else "technical_greedy_mismatch", "worker": spec, "comparison": comparison})
        require(comparison["same_token_ids"], "native preflight differs from saved step-16 greedy")
        workers = [spawn("main", shard, 8, gpu) for shard, gpu in enumerate(GPUS)]
        exits = []
        for process, worker_log, worker_spec in workers:
            try:
                code = process.wait(timeout=WORKER_SECONDS)
            except subprocess.TimeoutExpired:
                code = _stop_owned_process(process)
            finally:
                worker_log.close()
            exits.append({**worker_spec, "exit_code": code})
        publish(output / "main-exits.json", {"schema": f"{SCHEMA}.exits", "exits": exits})
        require(all(item["exit_code"] == 0 for item in exits), "refresh main worker failed")
        result = _collect(manifest_path, output); require(result["status"] == "candidate_ready", "saved step-16 greedy mismatch")
        terminal.update(status="completed", result=binding(output / "result.json"))
    except BaseException as exc:
        cleanup_errors = []
        for process, worker_log, worker_spec in owned:
            try:
                if process.poll() is None:
                    code = _stop_owned_process(process)
                    if code == "kill_timeout":
                        cleanup_errors.append({"pid": process.pid, "error": "kill_timeout"})
            except BaseException as cleanup_exc:
                cleanup_errors.append({"pid": process.pid, "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}"})
            finally:
                worker_log.close()
        terminal.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), cleanup_errors=cleanup_errors)
    finally:
        terminal["elapsed_seconds"] = time.monotonic() - started
        publish(output / "terminal.json", terminal)
    if terminal["status"] != "completed": raise RuntimeError(terminal["error"])


def launch(*, manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = read(manifest_path); validate_manifest(manifest)
    require(not (output / "launch.json").exists(), "launch collision")
    session = "coordexp-stage02-refresh-v1"
    command = f"cd {Path(__file__).resolve().parents[2]} && exec {sys.executable} -m probes.training_set_completion.refresh controller --manifest {manifest_path} --output {output}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
    receipt = {"schema": f"{SCHEMA}.launch", "status": "launched", "tmux_session": session, "manifest": binding(manifest_path), "controller_command": command, "physical_gpus": list(GPUS), "per_worker_wall_seconds": WORKER_SECONDS, "expected_requests": 44, "preflight_request": manifest["requests"][0]["request_id"]}
    publish(output / "launch.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker", "controller", "launch"))
    parser.add_argument("--output", type=Path, default=ROOT); parser.add_argument("--manifest", type=Path)
    parser.add_argument("--phase", choices=("preflight", "main")); parser.add_argument("--shard", type=int); parser.add_argument("--world-size", type=int); parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    if args.command == "prepare": value = prepare(args.output)
    elif args.command == "launch": value = launch(manifest_path=args.manifest or args.output / "manifest.json", output=args.output)
    elif args.command == "controller": controller(manifest_path=args.manifest or args.output / "manifest.json", output=args.output); return
    else:
        require(args.manifest is not None and args.phase is not None and args.shard is not None and args.world_size is not None and args.physical_gpu is not None, "worker arguments")
        worker(manifest_path=args.manifest, output=args.output, phase=args.phase, shard=args.shard, world_size=args.world_size, physical_gpu=args.physical_gpu); return
    print(json.dumps({"schema": value.get("schema"), "status": value.get("status")}, sort_keys=True))


if __name__ == "__main__":
    main()

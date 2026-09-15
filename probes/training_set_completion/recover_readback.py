"""Durable per-image recovery of interrupted second-fit native readbacks.

The original training readback writer published only after an entire 11-image
checkpoint readback.  This recovery keeps the same native input and greedy
policy but atomically publishes every completed checkpoint/image result first.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from probes.training_set_completion.acquisition import binding, digest, publish, read, require


B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
ROOT = B / "second-fit-readback-recovery-v1"
MANIFEST = B / "second-fit-preparation-v1/manifest.json"
TRAIN = B / "second-fit-v1/training"
STEPS = (16, 32, 64)
CAP = 3084
EOS = 151645
WORKER_SECONDS = 3600
SCHEMA = "training_set_completion.second_fit_readback_recovery.v1"


def row_path(root: Path, step: int, image_id: int) -> Path:
    return root / "rows" / f"step-{step:05d}-image-{image_id:012d}.json"


def jobs(manifest: Mapping[str, Any]) -> list[dict[str, int]]:
    images = [int(route["image_id"]) for route in manifest["routes"]]
    require(len(images) == 11 and len(set(images)) == 11, "route denominator")
    return [{"step": step, "image_id": image_id} for step in STEPS for image_id in images]


def partition(manifest: Mapping[str, Any]) -> list[list[dict[str, int]]]:
    """Fixed 8-GPU schedule: one step-16/64 worker, nine model loads total."""
    by_step = {step: [job for job in jobs(manifest) if job["step"] == step] for step in STEPS}
    groups = [
        by_step[16][:4],
        by_step[16][4:8],
        by_step[16][8:] + by_step[64][10:],  # GPU 2 reloads 16 -> 64.
        by_step[32][:4],
        by_step[32][4:8],
        by_step[32][8:],
        by_step[64][:5],
        by_step[64][5:10],
    ]
    mixed = [group for group in groups if len({job["step"] for job in group}) > 1]
    require(
        len(groups) == 8
        and sum(map(len, groups)) == 33
        and all(len(group) <= 5 and len({job["step"] for job in group}) <= 2 for group in groups)
        and len(mixed) == 1
        and [job["step"] for job in mixed[0]] == [16, 16, 16, 64],
        "worker partition",
    )
    return groups


def adapter_path(step: int) -> Path:
    return TRAIN / "checkpoints" / f"step-{step:05d}" / "adapter"


def adapter_identity(step: int, model_config: Mapping[str, Any]) -> dict[str, Any]:
    from src.adapters.dora import inspect_dora_adapter_payload

    return inspect_dora_adapter_payload(adapter_path(step), model_config["model"]["base_model"])


def prepare(output: Path = ROOT) -> dict[str, Any]:
    from probes.training_set_completion.training import validate_manifest

    require(not (output / "manifest.json").exists(), "recovery collision")
    training_manifest = read(MANIFEST)
    validate_manifest(training_manifest)
    adapters = {str(step): adapter_identity(step, training_manifest["model_config"]) for step in STEPS}
    value = {
        "schema": SCHEMA,
        "status": "candidate_ready",
        "training_manifest": binding(MANIFEST),
        "training_terminal": binding(TRAIN / "terminal.json"),
        "checkpoint_adapters": adapters,
        "routes": training_manifest["routes"],
        "model_config": training_manifest["model_config"],
        "runtime": {
            "cap": CAP,
            "eos": EOS,
            "worker_seconds": WORKER_SECONDS,
            "policy": {
                "empty_assistant_prefix": True,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
            },
        },
        "jobs": jobs(training_manifest),
        "partitions": partition(training_manifest),
        "producer": binding(Path(__file__)),
        "content_sha256": None,
    }
    value["content_sha256"] = digest({key: item for key, item in value.items() if key != "content_sha256"})
    publish(output / "manifest.json", value)
    return value


def validate(value: Mapping[str, Any]) -> None:
    require(value.get("schema") == SCHEMA, "recovery schema")
    require(value.get("content_sha256") == digest({key: item for key, item in value.items() if key != "content_sha256"}), "recovery manifest hash")
    require(len(value["jobs"]) == 33 and len(value["partitions"]) == 8, "recovery job denominator")
    for name in ("training_manifest", "training_terminal", "producer"):
        require(binding(value[name]["path"]) == value[name], f"{name} changed")
    for step in STEPS:
        observed = adapter_identity(step, value["model_config"])
        require(observed == value["checkpoint_adapters"][str(step)], f"checkpoint {step} adapter changed")


def validate_row_checkpoint(row: Mapping[str, Any], job: Mapping[str, int], manifest: Mapping[str, Any]) -> None:
    """Consumer-side guard: an image may only be paired with its requested adapter."""
    step = int(job["step"])
    require(row.get("checkpoint_step") == step and row.get("image_id") == job["image_id"], "durable row identity")
    require(row.get("checkpoint_adapter") == manifest["checkpoint_adapters"][str(step)], "row checkpoint adapter mismatch")
    require(binding(row["model_receipt"]["path"]) == row["model_receipt"], "row model receipt changed")


def _load_checkpoint(*, step: int, manifest: Mapping[str, Any], output: Path, shard: int, torch: Any, InferConfig: Any, checkpoint_config: Any, load_policy: Any) -> tuple[Any, dict[str, Any], Path]:
    adapter = adapter_path(step)
    config = checkpoint_config(InferConfig.model_validate(manifest["model_config"]), str(adapter))
    qwen, identity = load_policy(config, device=torch.device("cuda:0"))
    require(identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"], "FP32")
    require(identity["effective_settings"]["observed_attn_implementation"] == "sdpa", "SDPA")
    require(identity["model_identity"]["adapter"]["adapter_path"] == str(adapter), "loaded adapter path")
    require(adapter_identity(step, manifest["model_config"]) == manifest["checkpoint_adapters"][str(step)], "adapter fingerprint")
    receipt = output / "model" / f"shard-{shard}-step-{step:05d}.json"
    publish(receipt, identity)
    qwen.model.eval()
    return qwen, identity, receipt


def _hooks(qwen: Any, terminal: dict[str, Any]) -> list[Any]:
    handles = [qwen.model.register_forward_pre_hook(lambda *_: terminal.__setitem__("model_forwards", terminal["model_forwards"] + 1))]
    visuals = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
    require(len(visuals) == 1, "visual")
    handles.append(visuals[0].register_forward_pre_hook(lambda *_: terminal.__setitem__("image_forwards", terminal["image_forwards"] + 1)))
    return handles


def worker(*, manifest_path: Path, output: Path, shard: int, gpu: int) -> None:
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from probes.source_rweak_row_cross.run import build_requests
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    manifest = read(manifest_path)
    validate(manifest)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu) and torch.cuda.device_count() == 1, "GPU isolation")
    work = manifest["partitions"][shard]
    terminal = {
        "schema": SCHEMA + ".terminal",
        "status": "running",
        "shard": shard,
        "gpu": gpu,
        "expected": len(work),
        "completed": 0,
        "model_loads": 0,
        "model_forwards": 0,
        "image_forwards": 0,
        "generated_tokens": 0,
        "manifest": binding(manifest_path),
    }
    terminal_path = output / "terminals" / f"shard-{shard}.json"
    require(not terminal_path.exists(), "terminal collision")
    started = time.monotonic()
    handles: list[Any] = []
    qwen: Any = None
    active_step: int | None = None
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("readback worker wall")))
        signal.alarm(WORKER_SECONDS)
        torch.cuda.set_device("cuda:0")
        routes = {int(route["image_id"]): route for route in manifest["routes"]}
        policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, use_model_defaults=False)
        for job in work:
            step = int(job["step"])
            if step != active_step:
                for handle in handles:
                    handle.remove()
                handles = []
                if qwen is not None:
                    del qwen
                    torch.cuda.empty_cache()
                qwen, _, model_receipt = _load_checkpoint(
                    step=step, manifest=manifest, output=output, shard=shard, torch=torch,
                    InferConfig=InferConfig, checkpoint_config=checkpoint_config, load_policy=load_policy,
                )
                active_step = step
                terminal["model_loads"] += 1
                handles = _hooks(qwen, terminal)
            route = routes[int(job["image_id"])]
            requests, _ = build_requests(qwen, manifest["model_config"], [route["case"]])
            batch = prepare_native_inputs(qwen.processor, requests, device="cuda:0", record_media_identity=True)
            require(list(batch.prompt_token_ids[0]) == route["prompt_token_ids"], "original prompt changed")
            require(batch.media_sha256[0] == route["image_identity"]["executed_media_sha256"], "original media changed")
            require(list(batch.image_grids[0]) == route["image_identity"]["observed_image_grid_thw"], "original image grid changed")
            tick = time.monotonic()
            with torch.inference_mode():
                generated, = generate_continuations(
                    qwen.model, batch, extensions=[[]], budgets=[CAP], eos_token_id=EOS,
                    pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace="none", seed=None,
                )
            ids = list(generated.token_ids)
            require(ids and len(ids) <= CAP, "generated token cap")
            require(
                (generated.stop_reason == "im_end" and ids[-1] == EOS and EOS not in ids[:-1])
                or (generated.stop_reason == "length" and len(ids) == CAP and EOS not in ids),
                "terminal/cap",
            )
            elapsed = time.monotonic() - tick
            row = {
                "schema": SCHEMA + ".row",
                "checkpoint_step": step,
                "route_id": route["route_id"],
                "image_id": route["image_id"],
                "empty_assistant_prefix": True,
                "prompt_token_ids": route["prompt_token_ids"],
                "prompt_token_ids_sha256": digest(route["prompt_token_ids"]),
                "generated_token_ids": ids,
                "generated_token_ids_sha256": digest(ids),
                "decode_stop_reason": generated.stop_reason,
                "raw_decode_text": qwen.tokenizer.decode(ids, skip_special_tokens=False),
                "executed_media_sha256": batch.media_sha256[0],
                "observed_image_grid_thw": list(batch.image_grids[0]),
                "elapsed_seconds": elapsed,
                "token_rate_per_second": len(ids) / elapsed if elapsed else None,
                "checkpoint_adapter": manifest["checkpoint_adapters"][str(step)],
                "model_receipt": binding(model_receipt),
                "manifest": binding(manifest_path),
            }
            publish(row_path(output, step, int(route["image_id"])), row)
            terminal["completed"] += 1
            terminal["generated_tokens"] += len(ids)
        terminal.update(status="completed", exit_code=0)
    except BaseException as error:
        terminal.update(status="failed", exit_code=1, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        for handle in handles:
            handle.remove()
        terminal.update(elapsed_seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        publish(terminal_path, terminal)


def collect(manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = read(manifest_path)
    validate(manifest)
    by_step: dict[int, list[dict[str, Any]]] = {step: [] for step in STEPS}
    for job in manifest["jobs"]:
        row = read(row_path(output, int(job["step"]), int(job["image_id"])))
        validate_row_checkpoint(row, job, manifest)
        require(len(row["generated_token_ids"]) <= CAP, "durable cap")
        by_step[int(job["step"])].append(row)
    envelopes = []
    for step, rows in by_step.items():
        rows.sort(key=lambda row: row["image_id"])
        loaded_model = read(rows[0]["model_receipt"]["path"])
        envelope = {
            "schema": "training_set_completion.masked_coherent_route.native_readback.v1",
            "status": "completed_unscored",
            "manifest": manifest["training_manifest"],
            "adapter": manifest["checkpoint_adapters"][str(step)],
            "loaded_model": loaded_model,
            "policy": {**manifest["runtime"]["policy"], "assistant_token_cap": CAP},
            "rows": [
                {key: row[key] for key in (
                    "route_id", "image_id", "empty_assistant_prefix", "prompt_token_ids",
                    "generated_token_ids", "generated_token_ids_sha256", "decode_stop_reason",
                    "raw_decode_text", "executed_media_sha256", "observed_image_grid_thw",
                )}
                for row in rows
            ],
            "recovery": {"checkpoint_step": step, "per_image_rows": [binding(row_path(output, step, row["image_id"])) for row in rows]},
        }
        path = output / f"readback-step-{step}.json"
        publish(path, envelope)
        envelopes.append(binding(path))
    result = {
        "schema": SCHEMA + ".result",
        "status": "candidate_ready",
        "manifest": binding(manifest_path),
        "envelopes": envelopes,
        "request_count": 33,
        "generated_tokens": sum(len(row["generated_token_ids"]) for rows in by_step.values() for row in rows),
        "stop_counts": {reason: sum(row["decode_stop_reason"] == reason for rows in by_step.values() for row in rows) for reason in ("im_end", "length")},
        "per_image_timing": [
            {"step": row["checkpoint_step"], "image_id": row["image_id"], "elapsed_seconds": row["elapsed_seconds"], "token_rate_per_second": row["token_rate_per_second"]}
            for rows in by_step.values() for row in rows
        ],
    }
    publish(output / "result.json", result)
    return result


def controller(manifest_path: Path, output: Path) -> None:
    manifest = read(manifest_path)
    validate(manifest)
    terminal: dict[str, Any] = {"schema": SCHEMA + ".controller", "status": "running", "manifest": binding(manifest_path)}
    started = time.monotonic()
    try:
        processes = []
        for shard, gpu in enumerate(range(8)):
            log = output / "logs" / f"shard-{shard}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("x")
            command = ["python", "-m", "probes.training_set_completion.recover_readback", "worker", "--manifest", str(manifest_path), "--output", str(output), "--shard", str(shard), "--gpu", str(gpu)]
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"})
            processes.append((process, stream, command))
        exits = []
        for process, stream, command in processes:
            exits.append({"pid": process.pid, "exit_code": process.wait(timeout=WORKER_SECONDS), "command": command})
            stream.close()
        publish(output / "exits.json", {"schema": SCHEMA + ".exits", "exits": exits})
        require(all(entry["exit_code"] == 0 for entry in exits), "worker failure")
        collect(manifest_path, output)
        terminal.update(status="completed", result=binding(output / "result.json"))
    except BaseException as error:
        terminal.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        terminal["elapsed_seconds"] = time.monotonic() - started
        publish(output / "terminal.json", terminal)
    if terminal["status"] != "completed":
        raise RuntimeError(terminal["error"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker", "controller"))
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output)))
    elif args.command == "controller":
        controller(args.manifest or args.output / "manifest.json", args.output)
    else:
        require(args.manifest is not None and args.shard is not None and args.gpu is not None, "worker args")
        worker(manifest_path=args.manifest, output=args.output, shard=args.shard, gpu=args.gpu)


if __name__ == "__main__":
    main()

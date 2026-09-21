"""Bounded fourth-fit continuation and durable readback controller.

The continuation producer owns the step-64 restore and cumulative training
manifest.  This wrapper owns only new output roots and remaps the unchanged
readback worker's fixed 16/32/64 schedule to 128/192/256 per child process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

WORKTREE = Path("/data/CoordExp/.worktrees/research-probes").resolve()
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from probes.training_set_completion.acquisition import binding, digest, publish, read, require
import probes.training_set_completion.recover_readback as recovery


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum").resolve()
ROOT = Path(__file__).resolve().parent
MANIFEST = BASE / "fourth-fit-preparation-v1/manifest.json"
TRAINING = ROOT / "training"
READBACK = ROOT / "readback-recovery"
RECOVERY_SOURCE = Path(recovery.__file__).resolve()
THIRD_WRAPPER = BASE / "third-fit-v1/run.py"
MS_PYTHON = Path("/root/miniconda3/envs/ms/bin/python").resolve()
STEPS = (128, 192, 256)
IMAGE_IDS = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
CAP = 3084
EOS = 151645
TRAINING_GPU = 0
TRAINING_TIMEOUT_SECONDS = 4560
WORKER_TIMEOUT_SECONDS = 3660
SCHEMA = "training_set_completion.fourth_fit.v1"


def partition(manifest: Mapping[str, Any]) -> list[list[dict[str, int]]]:
    """Accepted 8-worker/9-load layout with continuation step labels."""
    images = [int(route["image_id"]) for route in manifest["routes"]]
    require(len(images) == 11 and set(images) == set(IMAGE_IDS), "route denominator/cohort")
    by_step = {step: [{"step": step, "image_id": image} for image in images] for step in STEPS}
    groups = [
        by_step[128][:4],
        by_step[128][4:8],
        by_step[128][8:] + by_step[256][10:],
        by_step[192][:4],
        by_step[192][4:8],
        by_step[192][8:],
        by_step[256][:5],
        by_step[256][5:10],
    ]
    mixed = [group for group in groups if len({job["step"] for job in group}) > 1]
    require(len(groups) == 8 and sum(map(len, groups)) == 33 and all(len(group) <= 5 for group in groups), "worker partition")
    require(len(mixed) == 1 and [job["step"] for job in mixed[0]] == [128, 128, 128, 256], "mixed continuation partition")
    require(sum(len({job["step"] for job in group}) for group in groups) == 9, "nine model loads")
    return groups


def _configure(*, manifest: Path, training_root: Path, readback_root: Path) -> None:
    recovery.MANIFEST = manifest.resolve()
    recovery.TRAIN = training_root.resolve()
    recovery.ROOT = readback_root.resolve()
    recovery.STEPS = STEPS
    recovery.partition = partition


def _env(gpu: int | None = None) -> dict[str, str]:
    require(MS_PYTHON.is_file(), f"selected ms interpreter missing: {MS_PYTHON}")
    env = dict(os.environ)
    env["CONDA_PREFIX"] = str(MS_PYTHON.parent.parent)
    env["PATH"] = f"{MS_PYTHON.parent}:{env.get('PATH', '')}"
    env["OMP_NUM_THREADS"] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"receipt collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _training_command() -> list[str]:
    return [str(MS_PYTHON), "-m", "probes.training_set_completion.continue_training", "run",
            "--manifest", str(MANIFEST), "--output", str(TRAINING), "--device", "cuda:0"]


def _check_training() -> dict[str, Any]:
    terminal = read(TRAINING / "terminal.json")
    require(terminal["status"] == "completed" and terminal["updates"] == 256, "continuation training terminal")
    require(terminal["model_forwards"] == 2112, "continuation segment forward count")
    require(terminal["manifest"] == binding(MANIFEST), "continuation manifest binding")
    require([int(item["step"]) for item in terminal["checkpoints"]] == list(STEPS), "continuation checkpoint schedule")
    for step in STEPS:
        root = TRAINING / "checkpoints" / f"step-{step:05d}"
        require((root / "adapter").is_dir() and (root / "state.pt").is_file(), f"checkpoint {step} files")
    return terminal


def train() -> dict[str, Any]:
    require(MANIFEST.is_file(), f"continuation manifest missing: {MANIFEST}")
    require(not TRAINING.exists(), f"training output collision: {TRAINING}")
    command = _training_command()
    TRAINING.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _write_json(ROOT / "training-launch.json", {"schema": SCHEMA + ".training_launch.v1", "status": "running",
               "manifest": binding(MANIFEST), "output": str(TRAINING), "command": command,
               "selected_python": str(MS_PYTHON), "cuda_visible_devices": str(TRAINING_GPU),
               "timeout_seconds": TRAINING_TIMEOUT_SECONDS, "resume_start_step": 64})
    with (ROOT / "training.log").open("x") as log:
        process = subprocess.Popen(command, cwd=WORKTREE, env=_env(TRAINING_GPU), stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=TRAINING_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _write_json(ROOT / "training-exit.json", {"schema": SCHEMA + ".training_exit.v1", "exit_code": process.returncode,
                       "timed_out": True, "elapsed_seconds": time.monotonic() - started})
            raise TimeoutError("fourth-fit continuation exceeded outer wall bound")
    _write_json(ROOT / "training-exit.json", {"schema": SCHEMA + ".training_exit.v1", "exit_code": code,
               "timed_out": False, "elapsed_seconds": time.monotonic() - started})
    require(code == 0, f"continuation training exit {code}")
    return _check_training()


def prepare_readback_manifest() -> dict[str, Any]:
    """Bind the continuation checkpoints to the remapped unchanged worker."""
    from probes.training_set_completion.training import validate_manifest

    require(not (READBACK / "manifest.json").exists(), "readback manifest collision")
    training_manifest = read(MANIFEST)
    validate_manifest(training_manifest)
    training_terminal = TRAINING / "terminal.json"
    require(training_terminal.is_file() and read(training_terminal).get("status") == "completed", "training not complete")
    _configure(manifest=MANIFEST, training_root=TRAINING, readback_root=READBACK)
    adapters = {str(step): recovery.adapter_identity(step, training_manifest["model_config"]) for step in STEPS}
    jobs = recovery.jobs(training_manifest)
    groups = partition(training_manifest)
    require(len(jobs) == 33 and len(groups) == 8 and sum(map(len, groups)) == 33, "readback denominator")
    value: dict[str, Any] = {
        "schema": recovery.SCHEMA,
        "status": "candidate_ready",
        "training_manifest": binding(MANIFEST),
        "training_terminal": binding(training_terminal),
        "checkpoint_adapters": adapters,
        "routes": training_manifest["routes"],
        "model_config": training_manifest["model_config"],
        "runtime": {"cap": CAP, "eos": EOS, "worker_seconds": 3600,
                    "policy": {"empty_assistant_prefix": True, "temperature": 0.0,
                                "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}},
        "jobs": jobs,
        "partitions": groups,
        "producer": binding(Path(__file__).resolve()),
        "reused_producer": binding(RECOVERY_SOURCE),
        "bound_roots": {"manifest": str(MANIFEST), "training": str(TRAINING), "readback": str(READBACK)},
        "step_mapping": {"original_recovery_steps": [16, 32, 64], "continuation_steps": list(STEPS), "mapping": {"16": 128, "32": 192, "64": 256}}
    }
    value["content_sha256"] = digest(value)
    READBACK.mkdir(parents=True, exist_ok=True)
    publish(READBACK / "manifest.json", value)
    return value


def validate_readback_manifest(path: Path) -> dict[str, Any]:
    value = read(path)
    _configure(manifest=path, training_root=TRAINING, readback_root=READBACK)
    recovery.validate(value)
    require(value["producer"] == binding(Path(__file__).resolve()), "wrapper binding changed")
    require(value["reused_producer"] == binding(RECOVERY_SOURCE), "recovery binding changed")
    require(value["bound_roots"] == {"manifest": str(MANIFEST), "training": str(TRAINING), "readback": str(READBACK)}, "stale roots")
    require(value["training_manifest"] == binding(MANIFEST) and value["training_terminal"] == binding(TRAINING / "terminal.json"), "training binding")
    require(value["step_mapping"] == {"original_recovery_steps": [16, 32, 64], "continuation_steps": [128, 192, 256], "mapping": {"16": 128, "32": 192, "64": 256}}, "step mapping")
    return value


def _worker_command(manifest: Path, shard: int) -> list[str]:
    return [str(MS_PYTHON), str(Path(__file__).resolve()), "worker", "--manifest", str(manifest),
            "--training-root", str(TRAINING), "--readback-root", str(READBACK), "--shard", str(shard), "--gpu", str(shard)]


def readback_controller() -> dict[str, Any]:
    manifest_path = READBACK / "manifest.json"
    value = validate_readback_manifest(manifest_path)
    require(not (READBACK / "controller-start.json").exists(), "controller collision")
    groups = value["partitions"]
    (READBACK / "logs").mkdir(parents=True, exist_ok=True)
    _write_json(READBACK / "controller-start.json", {"schema": SCHEMA + ".readback_controller_start.v1", "status": "running",
               "manifest": binding(manifest_path), "training_root": str(TRAINING), "readback_root": str(READBACK),
               "selected_python": str(MS_PYTHON), "gpus": list(range(8)), "job_count": 33, "worker_count": 8,
               "max_jobs_per_worker": 5, "expected_model_loads": 9, "step_mapping": value["step_mapping"]})
    processes: list[tuple[subprocess.Popen[Any], Any, list[str]]] = []
    exits: list[dict[str, Any]] = []
    try:
        for shard in range(8):
            command = _worker_command(manifest_path, shard)
            stream = (READBACK / "logs" / f"shard-{shard}.log").open("x")
            process = subprocess.Popen(command, cwd=WORKTREE, env=_env(shard), stdout=stream, stderr=subprocess.STDOUT)
            processes.append((process, stream, command))
        for process, stream, command in processes:
            try:
                code = process.wait(timeout=WORKER_TIMEOUT_SECONDS)
            finally:
                stream.close()
            exits.append({"pid": process.pid, "exit_code": code, "command": command})
        _write_json(READBACK / "exits.json", {"schema": recovery.SCHEMA + ".exits", "exits": exits})
        require(all(item["exit_code"] == 0 for item in exits), "readback worker failure")
        result = recovery.collect(manifest_path, READBACK)
        require(result.get("status") == "candidate_ready" and result.get("request_count") == 33, "readback collection")
        return result
    except BaseException:
        if not (READBACK / "exits.json").exists():
            _write_json(READBACK / "exits.json", {"schema": recovery.SCHEMA + ".exits", "exits": exits, "status": "failed"})
        raise


def controller() -> None:
    require(not (ROOT / "terminal.json").exists(), "terminal collision")
    started = time.monotonic()
    terminal: dict[str, Any] = {"schema": SCHEMA + ".terminal.v1", "status": "running", "manifest": binding(MANIFEST),
                                "training_root": str(TRAINING), "readback_root": str(READBACK),
                                "wrapper": binding(Path(__file__).resolve()), "reused_producer": binding(RECOVERY_SOURCE)}
    try:
        train_terminal = train()
        prepare_readback_manifest()
        result = readback_controller()
        terminal.update(status="completed_unscored", phase="completed", training_terminal=binding(TRAINING / "terminal.json"),
                        readback_manifest=binding(READBACK / "manifest.json"), result=binding(READBACK / "result.json"),
                        training_updates=train_terminal["updates"], segment_model_forwards=train_terminal["model_forwards"],
                        readback_requests=result["request_count"])
    except BaseException as error:
        terminal.update(status="failed", phase="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        terminal["elapsed_seconds"] = time.monotonic() - started
        _write_json(ROOT / "terminal.json", terminal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("controller")
    worker = sub.add_parser("worker")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--training-root", type=Path, required=True)
    worker.add_argument("--readback-root", type=Path, required=True)
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.command == "controller":
        controller()
    else:
        _configure(manifest=args.manifest, training_root=args.training_root, readback_root=args.readback_root)
        require(args.training_root.resolve() == TRAINING.resolve() and args.readback_root.resolve() == READBACK.resolve(), "child roots")
        require(args.shard == args.gpu and 0 <= args.shard < 8, "child shard/GPU")
        recovery.worker(manifest_path=args.manifest, output=args.readback_root, shard=args.shard, gpu=args.gpu)


if __name__ == "__main__":
    main()

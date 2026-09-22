"""Bounded third-fit controller and wrapper around durable readback recovery.

This file is a launch preparation artifact.  It is intentionally the child
entrypoint as well as the controller entrypoint: every child receives the
new manifest, training root, and readback root, then patches the unchanged
recovery module's historical globals before calling its worker.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

# Absolute-script child invocations do not guarantee the worktree is on
# sys.path; bind it explicitly before importing the unchanged producers.
WORKTREE = Path("/data/CoordExp/.worktrees/research-probes").resolve()
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from probes.training_set_completion.acquisition import binding, digest, publish, read, require
import probes.training_set_completion.recover_readback as recovery


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT.parent / "third-fit-preparation-v1" / "manifest.json"
TRAINING = ROOT / "training"
READBACK = ROOT / "readback-recovery"
RECOVERY_SOURCE = Path(recovery.__file__).resolve()
MS_PYTHON = Path("/root/miniconda3/envs/ms/bin/python").resolve()
STEPS = (16, 32, 64)
TRAINING_GPU = 0
TRAINING_TIMEOUT_SECONDS = 3660
WORKER_TIMEOUT_SECONDS = 3660
CAP = 3084
EOS = 151645
SCHEMA = "training_set_completion.third_fit.v1"


def _configure(*, manifest: Path, training_root: Path, readback_root: Path) -> None:
    """Bind all historical recovery globals to this invocation's roots."""
    recovery.MANIFEST = manifest.resolve()
    recovery.TRAIN = training_root.resolve()
    recovery.ROOT = readback_root.resolve()


def _child_env(gpu: int | None = None) -> dict[str, str]:
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


def _training_command(manifest: Path, output: Path) -> list[str]:
    return [str(MS_PYTHON), "-m", "probes.training_set_completion.training", "run",
            "--manifest", str(manifest), "--output", str(output), "--device", "cuda:0"]


def _check_training_terminal(manifest: Path, output: Path) -> dict[str, Any]:
    terminal_path = output / "terminal.json"
    require(terminal_path.is_file(), "training terminal missing")
    terminal = read(terminal_path)
    require(terminal.get("status") == "completed", "training did not complete")
    require(terminal.get("manifest") == binding(manifest), "training manifest binding changed")
    require(terminal.get("updates") == 64 and terminal.get("model_forwards") == 704, "training counters")
    checkpoint_steps = [int(item["step"]) for item in terminal.get("checkpoints", [])]
    require(checkpoint_steps == list(STEPS), "training checkpoint schedule")
    for step in STEPS:
        checkpoint = output / "checkpoints" / f"step-{step:05d}"
        require((checkpoint / "adapter").is_dir() and (checkpoint / "state.pt").is_file(), f"checkpoint {step} incomplete")
    return terminal


def train(manifest: Path, output: Path) -> dict[str, Any]:
    require(manifest.is_file(), f"final training manifest missing: {manifest}")
    require(not output.exists(), f"training output collision: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = _training_command(manifest, output)
    started = time.monotonic()
    launch = {"schema": SCHEMA + ".training_launch.v1", "status": "running", "manifest": binding(manifest),
              "output": str(output), "command": command, "selected_python": str(MS_PYTHON),
              "cuda_visible_devices": str(TRAINING_GPU), "timeout_seconds": TRAINING_TIMEOUT_SECONDS}
    _write_json(output.parent / "training-launch.json", launch)
    with (output.parent / "training.log").open("x") as log:
        process = subprocess.Popen(command, cwd=WORKTREE, env=_child_env(TRAINING_GPU), stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=TRAINING_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _write_json(output.parent / "training-exit.json", {"schema": SCHEMA + ".training_exit.v1", "exit_code": process.returncode,
                       "timed_out": True, "elapsed_seconds": time.monotonic() - started})
            raise TimeoutError("third-fit training exceeded outer wall bound")
    _write_json(output.parent / "training-exit.json", {"schema": SCHEMA + ".training_exit.v1", "exit_code": code,
               "timed_out": False, "elapsed_seconds": time.monotonic() - started})
    require(code == 0, f"training exit {code}")
    return _check_training_terminal(manifest, output)


def _readback_manifest(*, manifest: Path, training_root: Path, output: Path) -> dict[str, Any]:
    """Create a new-root recovery manifest while retaining recovery semantics."""
    from probes.training_set_completion.training import validate_manifest

    require(not (output / "manifest.json").exists(), "readback manifest collision")
    training_value = read(manifest)
    validate_manifest(training_value)
    training_terminal = training_root / "terminal.json"
    require(training_terminal.is_file(), "completed training terminal missing")
    terminal = read(training_terminal)
    require(terminal.get("status") == "completed", "readback requires completed training")
    _configure(manifest=manifest, training_root=training_root, readback_root=output)
    adapters = {str(step): recovery.adapter_identity(step, training_value["model_config"]) for step in STEPS}
    jobs = recovery.jobs(training_value)
    partitions = recovery.partition(training_value)
    require(len(jobs) == 33 and len(partitions) == 8 and sum(len(group) for group in partitions) == 33, "readback schedule")
    require(sum(len({job["step"] for job in group}) for group in partitions) == 9, "readback model-load schedule")
    value: dict[str, Any] = {
        "schema": recovery.SCHEMA,
        "status": "candidate_ready",
        "training_manifest": binding(manifest),
        "training_terminal": binding(training_terminal),
        "checkpoint_adapters": adapters,
        "routes": training_value["routes"],
        "model_config": training_value["model_config"],
        "runtime": {"cap": CAP, "eos": EOS, "worker_seconds": recovery.WORKER_SECONDS,
                    "policy": {"empty_assistant_prefix": True, "temperature": 0.0, "top_p": 1.0,
                                "top_k": 0, "repetition_penalty": 1.0}},
        "jobs": jobs,
        "partitions": partitions,
        "producer": binding(Path(__file__).resolve()),
        "reused_producer": binding(RECOVERY_SOURCE),
        "bound_roots": {"manifest": str(manifest.resolve()), "training": str(training_root.resolve()), "readback": str(output.resolve())},
    }
    value["content_sha256"] = digest(value)
    publish(output / "manifest.json", value)
    return value


def validate_readback_manifest(manifest_path: Path, *, training_root: Path, readback_root: Path) -> dict[str, Any]:
    value = read(manifest_path)
    recovery.validate(value)
    require(value.get("producer") == binding(Path(__file__).resolve()), "wrapper binding changed")
    require(value.get("reused_producer") == binding(RECOVERY_SOURCE), "recovery producer binding changed")
    require(value.get("bound_roots") == {"manifest": str(value["training_manifest"]["path"]),
            "training": str(training_root.resolve()), "readback": str(readback_root.resolve())}, "hidden or stale roots")
    require(value["training_manifest"] == binding(value["training_manifest"]["path"]), "training manifest changed")
    require(value["training_terminal"] == binding(value["training_terminal"]["path"]), "training terminal changed")
    return value


def worker(*, manifest: Path, training_root: Path, readback_root: Path, shard: int, gpu: int) -> None:
    _configure(manifest=manifest, training_root=training_root, readback_root=readback_root)
    value = validate_readback_manifest(manifest, training_root=training_root, readback_root=readback_root)
    require(0 <= shard < 8 and gpu == shard, "fixed shard/GPU identity")
    recovery.worker(manifest_path=manifest, output=readback_root, shard=shard, gpu=gpu)


def _worker_command(manifest: Path, training_root: Path, readback_root: Path, shard: int, gpu: int) -> list[str]:
    return [str(MS_PYTHON), str(Path(__file__).resolve()), "worker", "--manifest", str(manifest),
            "--training-root", str(training_root), "--readback-root", str(readback_root),
            "--shard", str(shard), "--gpu", str(gpu)]


def readback_controller(*, manifest: Path, training_root: Path, output: Path) -> dict[str, Any]:
    _configure(manifest=manifest, training_root=training_root, readback_root=output)
    value = validate_readback_manifest(manifest, training_root=training_root, readback_root=output)
    require(output.is_dir(), "readback root missing")
    groups = value["partitions"]
    require(len(groups) == 8 and sum(len(group) for group in groups) == 33 and max(map(len, groups)) <= 5, "readback partition")
    require(sum(len({job["step"] for job in group}) for group in groups) == 9, "readback expected nine model loads")
    (output / "logs").mkdir(parents=True, exist_ok=True)
    start = {"schema": SCHEMA + ".readback_controller_start.v1", "status": "running", "manifest": binding(manifest),
             "training_root": str(training_root.resolve()), "readback_root": str(output.resolve()),
             "selected_python": str(MS_PYTHON), "gpus": list(range(8)), "job_count": 33,
             "worker_count": 8, "max_jobs_per_worker": 5, "expected_model_loads": 9}
    _write_json(output / "controller-start.json", start)
    processes: list[tuple[subprocess.Popen[Any], Any, list[str]]] = []
    exits: list[dict[str, Any]] = []
    try:
        for shard, gpu in enumerate(range(8)):
            log_path = output / "logs" / f"shard-{shard}.log"
            command = _worker_command(manifest, training_root, output, shard, gpu)
            stream = log_path.open("x")
            process = subprocess.Popen(command, cwd=WORKTREE, env=_child_env(gpu), stdout=stream, stderr=subprocess.STDOUT)
            processes.append((process, stream, command))
        for process, stream, command in processes:
            try:
                code = process.wait(timeout=WORKER_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code = process.returncode
                raise TimeoutError("third-fit readback worker exceeded wall bound")
            finally:
                stream.close()
            exits.append({"pid": process.pid, "exit_code": code, "command": command})
        _write_json(output / "exits.json", {"schema": recovery.SCHEMA + ".exits", "exits": exits})
        require(all(item["exit_code"] == 0 for item in exits), "readback worker failure")
        result = recovery.collect(manifest, output)
        require(result.get("status") == "candidate_ready" and result.get("request_count") == 33, "readback collection incomplete")
        return result
    except BaseException:
        if not (output / "exits.json").exists():
            _write_json(output / "exits.json", {"schema": recovery.SCHEMA + ".exits", "exits": exits, "status": "failed"})
        raise


def controller(manifest: Path = MANIFEST) -> None:
    require(manifest.is_file(), f"final training manifest missing: {manifest}")
    require(not (ROOT / "terminal.json").exists(), "third-fit terminal collision")
    started = time.monotonic()
    terminal: dict[str, Any] = {"schema": SCHEMA + ".terminal.v1", "status": "running", "manifest": binding(manifest),
                                "training_root": str(TRAINING.resolve()), "readback_root": str(READBACK.resolve()),
                                "wrapper": binding(Path(__file__).resolve()), "reused_producer": binding(RECOVERY_SOURCE)}
    try:
        train_terminal = train(manifest, TRAINING)
        recovery_manifest = _readback_manifest(manifest=manifest, training_root=TRAINING, output=READBACK)
        result = readback_controller(manifest=READBACK / "manifest.json", training_root=TRAINING, output=READBACK)
        terminal.update(status="completed_unscored", phase="completed", training_terminal=binding(TRAINING / "terminal.json"),
                        readback_manifest=binding(READBACK / "manifest.json"), result=binding(READBACK / "result.json"),
                        training_updates=train_terminal["updates"], readback_requests=result["request_count"])
    except BaseException as error:
        terminal.update(status="failed", phase="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        terminal["elapsed_seconds"] = time.monotonic() - started
        _write_json(ROOT / "terminal.json", terminal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    controller_parser = sub.add_parser("controller")
    controller_parser.add_argument("--manifest", type=Path, default=MANIFEST)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--training-root", type=Path, required=True)
    worker_parser.add_argument("--readback-root", type=Path, required=True)
    worker_parser.add_argument("--shard", type=int, required=True)
    worker_parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.command == "controller":
        controller(args.manifest)
    else:
        worker(manifest=args.manifest, training_root=args.training_root, readback_root=args.readback_root,
               shard=args.shard, gpu=args.gpu)


if __name__ == "__main__":
    main()

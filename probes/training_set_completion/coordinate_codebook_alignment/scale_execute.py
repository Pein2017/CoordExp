"""Run the fixed scale queues with the maintained per-process cost supervisor."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import time

from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.coordinate_codebook_alignment import execute


def remaining(queue: Path) -> int:
    total = len(json.loads(queue.read_text())["specs"])
    state = queue.with_name("queue-state.json")
    claimed = json.loads(state.read_text())["next_index"] if state.exists() else 0
    return max(0, total - claimed)


def select_condition(launch: dict, training_done: bool) -> str | None:
    order = launch.get("condition_order_after" if training_done else "condition_order_during")
    if order is None:
        order = ("epoch32", "source", "epoch16", "epoch4") if training_done else ("epoch4", "epoch16", "source")
    for condition in order:
        item = launch["evaluation"][condition]
        checkpoint = item["checkpoint"]
        if checkpoint != "source" and not Path(checkpoint).is_dir():
            continue
        if remaining(Path(item["queue"])):
            return condition
    return None


def run(launch_path: Path, *, continue_existing: bool = False) -> None:
    launch = json.loads(launch_path.read_text())
    root = Path(launch["root"])
    if (root / "cost.json").exists() and not continue_existing:
        raise ValueError("cost ledger already exists; reconcile producers instead of relaunching")
    if continue_existing:
        prior = json.loads((root / "cost.json").read_text())
        busy = {g for j in prior["jobs"] if "terminal_time" not in j for g in j["gpus"]}
        if busy.intersection(range(4)):
            raise ValueError("training GPUs still have an owned producer")
    for item in launch["bindings"]:
        if binding(item["path"])["sha256"] != item["sha256"]:
            raise ValueError(f"launch binding changed: {item['path']}")
    queues = [Path(x["queue"]) for x in launch["evaluation"].values()]
    if len({p.parent for p in queues}) != len(queues):
        raise ValueError("condition queues must have isolated claim directories")
    if not continue_existing and any(p.with_name("queue-state.json").exists() for p in queues):
        raise ValueError("production queues already claimed")
    active = {}
    free = set(range(4, 8))
    training_done = False
    hold = False
    serial = 0
    owned_names = set()
    attempt = launch.get("attempt", "v1")
    events = root / "supervision-events.jsonl"

    def event(value):
        with events.open("a") as out:
            out.write(json.dumps({"time": time.time(), **value}, sort_keys=True) + "\n")

    def submit(pool, name, gpus, argv, condition):
        future = pool.submit(execute.run, name, gpus, argv, root,
                             cache_root=Path(launch["cache_root"]), reserve_seconds=900)
        active[future] = (name, gpus, condition)
        owned_names.add(name)
        event({"event": "submitted", "name": name, "gpus": gpus, "condition": condition})

    halfway = False
    with ThreadPoolExecutor(max_workers=8) as pool:
        submit(pool, launch.get("training_job_name", "fit-seed1729"), [0, 1, 2, 3], launch["training_argv"], "training")
        while active or free:
            external_jobs = []
            cost_path = root / "cost.json"
            if cost_path.exists():
                cost = json.loads(cost_path.read_text())
                external_jobs = [j for j in cost['jobs'] if 'terminal_time' not in j and j['name'] not in owned_names]
                external_gpus = {g for j in external_jobs for g in j['gpus']}
                active_gpus = {g for _, gpus, _ in active.values() for g in gpus}
                free = set(range(8)) - external_gpus - active_gpus
                used = execute.allocated_seconds(cost, time.time())
                elapsed = time.time() - cost["wall_start"]
                if elapsed >= 27900 or used >= 230400 - 7200:
                    hold = True
                if not halfway and (elapsed >= 14400 or used >= 115200):
                    event({"event": "halfway", "wall_seconds": elapsed,
                           "allocated_gpu_seconds": used,
                           "unclaimed": {p.parent.name: remaining(p) for p in queues}})
                    halfway = True
            if not hold:
                for gpu in sorted(tuple(free)):
                    condition = select_condition(launch, training_done)
                    if condition is None:
                        break
                    serial += 1
                    item = launch["evaluation"][condition]
                    argv = ["python", "-B", "-m",
                            "probes.training_set_completion.coordinate_codebook_alignment.evaluation",
                            "--admission", launch["evaluation_admission"],
                            "--checkpoint", item["checkpoint"], "--queue", item["queue"],
                            "--output", item["output"], "--device", "cuda:0", "--teacher"]
                    submit(pool, f"{condition}-gpu{gpu}-{attempt}-job{serial}", [gpu], argv, condition)
                    free.remove(gpu)
            if not active:
                if external_jobs:
                    time.sleep(30)
                    continue
                break
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                name, gpus, condition = active.pop(future)
                free.update(gpus)
                try:
                    code = future.result()
                    error = None
                except Exception as exc:
                    code, error = 1, repr(exc)
                event({"event": "terminal", "name": name, "exit_code": code, "error": error})
                if condition == "training":
                    training_done = True
                if code:
                    hold = True  # Preserve affected evidence; owner decides the bounded repair.
    event({"event": "all_jobs_joined", "hold": hold,
           "unclaimed": {p.parent.name: remaining(p) for p in queues}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--continue-existing", action="store_true")
    args = parser.parse_args()
    run(args.launch, continue_existing=args.continue_existing)

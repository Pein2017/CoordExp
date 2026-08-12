#!/usr/bin/env python3
"""Plan or explicitly execute isolated Human-13 world-size-one arm jobs.

Dry-run is the default.  ``--execute`` additionally requires the explicit
``--user-model-gpu-authority`` acknowledgement and a separately verified
``human13_runner_cli.v1`` entry that consumes each resolved plan.  None of
these flags grants execution authority by itself.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


MAX_JOBS = 8
ZERO_LAUNCH_ACTIONS = {"process_starts": 0, "gpu_jobs": 0, "retries": 0}


class LaunchContractError(ValueError):
    """Raised when independent-arm launch topology is not fail-closed."""


def _load_plans(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raw = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise LaunchContractError("materialized plans must be a JSON object")
    return raw


def plan_launches(
    plans: Mapping[str, Any] | str | Path,
    *,
    gpu_ids: Sequence[int],
    runner_entry: str | Path = "scripts/research/run_human13_k_union_overfit.py",
    runner_entry_contract: str | None = None,
) -> dict[str, Any]:
    payload = _load_plans(plans)
    all_plans = payload.get("plans")
    if not isinstance(all_plans, list) or not all_plans:
        raise LaunchContractError("materialized receipt has no arm plans")
    omitted_no_update = [
        str(plan.get("arm_id", ""))
        for plan in all_plans
        if plan.get("updates") is False
    ]
    if any(arm != "frozen_source" for arm in omitted_no_update):
        raise LaunchContractError("only frozen_source may be a no-update arm")
    raw_plans = [plan for plan in all_plans if plan.get("updates") is True]
    if not raw_plans:
        raise LaunchContractError("materialized receipt has no updated arm plans")
    if len(raw_plans) > MAX_JOBS:
        raise LaunchContractError("Human-13 launcher supports at most eight jobs")
    checked_gpus = tuple(gpu_ids)
    if len(checked_gpus) != len(raw_plans):
        raise LaunchContractError("one unique GPU ID is required for every arm job")
    if len(set(checked_gpus)) != len(checked_gpus):
        raise LaunchContractError("GPU IDs must be unique")
    if any(
        isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0
        for gpu in checked_gpus
    ):
        raise LaunchContractError("GPU IDs must be non-negative integers")

    arms = [str(plan.get("arm_id", "")) for plan in raw_plans]
    outputs = [str(plan.get("output_root", "")) for plan in raw_plans]
    states = [
        str(plan.get("optimizer_state_root", ""))
        for plan in raw_plans
        if plan.get("updates") is True
    ]
    if not all(arms) or len(set(arms)) != len(arms):
        raise LaunchContractError("arm identities must be nonempty and unique")
    if not all(outputs) or len(set(outputs)) != len(outputs):
        raise LaunchContractError("output roots must be nonempty and unique")
    if not all(states) or len(set(states)) != len(states):
        raise LaunchContractError("optimizer state roots must be nonempty and unique")
    if any(Path(root).exists() for root in outputs):
        raise LaunchContractError("refusing to overwrite an existing arm output root")

    runner_path = Path(runner_entry).resolve()
    contract_bound = runner_entry_contract == "human13_runner_cli.v1"
    jobs: list[dict[str, Any]] = []
    for plan, gpu_id in zip(raw_plans, checked_gpus, strict=True):
        plan_path = str(plan.get("resolved_plan_path", ""))
        if not plan_path:
            raise LaunchContractError("each arm requires one resolved_plan_path")
        command = [
            "conda",
            "run",
            "-n",
            "ms",
            "accelerate",
            "launch",
            "--num_processes",
            "1",
            "--num_machines",
            "1",
            str(runner_path),
            "--resolved-plan",
            plan_path,
        ]
        jobs.append(
            {
                "arm_id": str(plan["arm_id"]),
                "gpu_id": gpu_id,
                "world_size": 1,
                "output_root": str(plan["output_root"]),
                "optimizer_state_root": plan.get("optimizer_state_root"),
                "command": command,
                "retry_policy": "none",
                "runner_entry_contract": runner_entry_contract,
                "execution_ready": bool(
                    contract_bound
                    and runner_path.is_file()
                    and Path(plan_path).is_file()
                ),
            }
        )
    return {
        "schema_version": "human13_launch_plan.v1",
        "mode": "dry_run",
        "actions": dict(ZERO_LAUNCH_ACTIONS),
        "max_concurrent_jobs": len(jobs),
        "omitted_no_update_arms": omitted_no_update,
        "jobs": jobs,
    }


def execute_launches(
    launch_plan: Mapping[str, Any],
    *,
    execution_authorized: bool,
) -> dict[str, Any]:
    if execution_authorized is not True:
        raise LaunchContractError(
            "execute mode requires separate user model/GPU launch authority"
        )
    jobs = launch_plan.get("jobs")
    if not isinstance(jobs, list) or not jobs or len(jobs) > MAX_JOBS:
        raise LaunchContractError("execute requires one bounded dry-run launch plan")
    if any(Path(str(job["output_root"])).exists() for job in jobs):
        raise LaunchContractError("refusing to overwrite an existing arm output root")
    if any(
        job.get("runner_entry_contract") != "human13_runner_cli.v1"
        or job.get("execution_ready") is not True
        for job in jobs
    ):
        raise LaunchContractError(
            "execute requires an explicit bound runnable runner entry contract"
        )

    processes: list[tuple[Mapping[str, Any], subprocess.Popen[Any]]] = []
    for job in jobs:
        if job.get("world_size") != 1 or job.get("retry_policy") != "none":
            raise LaunchContractError("every job must be world-size one with no retry")
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(job["gpu_id"]),
                "WORLD_SIZE": "1",
                "LOCAL_WORLD_SIZE": "1",
                "RANK": "0",
                "LOCAL_RANK": "0",
            }
        )
        processes.append((job, subprocess.Popen(list(job["command"]), env=env)))

    statuses = [
        {"arm_id": str(job["arm_id"]), "returncode": process.wait()}
        for job, process in processes
    ]
    failures = [item for item in statuses if item["returncode"] != 0]
    receipt = {
        "schema_version": "human13_launch_execution.v1",
        "mode": "execute",
        "process_start_count": len(processes),
        "retry_count": 0,
        "statuses": statuses,
    }
    if failures:
        raise RuntimeError(f"Human-13 arm jobs failed without retry: {failures}")
    return receipt


def _gpu_ids(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "gpus must be comma-separated integers"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--gpus", type=_gpu_ids, required=True)
    parser.add_argument(
        "--runner-entry",
        type=Path,
        default=Path("scripts/research/run_human13_k_union_overfit.py"),
    )
    parser.add_argument(
        "--runner-entry-contract",
        choices=("human13_runner_cli.v1",),
        help="Bind a separately verified CLI adapter that consumes --resolved-plan.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--user-model-gpu-authority",
        action="store_true",
        help="Acknowledge separately documented user execution authority.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = plan_launches(
        args.plans,
        gpu_ids=args.gpus,
        runner_entry=args.runner_entry,
        runner_entry_contract=args.runner_entry_contract,
    )
    receipt = (
        execute_launches(plan, execution_authorized=args.user_model_gpu_authority)
        if args.execute
        else plan
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

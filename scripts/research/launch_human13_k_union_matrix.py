#!/usr/bin/env python3
"""Plan or explicitly execute isolated Human-13 world-size-one arm jobs.

Dry-run is the default. ``--execute`` additionally requires the explicit
``--user-model-gpu-authority`` acknowledgement. Every planned command is bound
to the content-bearing plans receipt and the sealed manifest consumed by the
real Human-13 live trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


MAX_JOBS = 8
ZERO_LAUNCH_ACTIONS = {"process_starts": 0, "gpu_jobs": 0, "retries": 0}
RUNNER_ENTRY_CONTRACT = "human13_runner_cli.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER_ENTRY = REPO_ROOT / "scripts/research/train_human13_live_arm.py"


class LaunchContractError(ValueError):
    """Raised when independent-arm launch topology is not fail-closed."""


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchContractError(f"cannot load {label}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise LaunchContractError(f"{label} must be a JSON object")
    return raw


def _load_plans(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value).expanduser().resolve(strict=True)
    return _read_json_object(path, label="materialized plans")


def _bound_plans_path(
    plans: Mapping[str, Any] | str | Path,
    payload: Mapping[str, Any],
    plans_receipt_path: str | Path | None,
) -> Path:
    if isinstance(plans, Mapping):
        if plans_receipt_path is None:
            raise LaunchContractError(
                "mapping plans require an existing plans_receipt_path"
            )
        path = Path(plans_receipt_path).expanduser().resolve(strict=True)
        if _read_json_object(path, label="plans receipt") != payload:
            raise LaunchContractError(
                "plans_receipt_path content differs from the planned payload"
            )
        return path
    path = Path(plans).expanduser().resolve(strict=True)
    if plans_receipt_path is not None:
        requested = Path(plans_receipt_path).expanduser().resolve(strict=True)
        if requested != path:
            raise LaunchContractError("plans receipt paths disagree")
    return path


def _selected_updated_plans(
    all_plans: list[Mapping[str, Any]],
    selected_arm_ids: Sequence[str] | None,
    omitted_arms: Any,
) -> tuple[list[Mapping[str, Any]], list[str], list[str]]:
    by_arm = {str(plan.get("arm_id", "")): plan for plan in all_plans}
    if "" in by_arm or len(by_arm) != len(all_plans):
        raise LaunchContractError("arm identities must be nonempty and unique")

    omitted_no_update = [
        arm_id for arm_id, plan in by_arm.items() if plan.get("updates") is False
    ]
    if any(arm != "frozen_source" for arm in omitted_no_update):
        raise LaunchContractError("only frozen_source may be a no-update arm")
    updating = [plan for plan in all_plans if plan.get("updates") is True]
    if not updating:
        raise LaunchContractError("materialized receipt has no updated arm plans")
    if any(plan.get("updates") not in (True, False) for plan in all_plans):
        raise LaunchContractError("every arm must declare a boolean updates value")

    if selected_arm_ids is None:
        selected = updating
    else:
        requested = tuple(selected_arm_ids)
        if not requested or any(not arm for arm in requested):
            raise LaunchContractError("selected arm IDs must be nonempty")
        if len(set(requested)) != len(requested):
            raise LaunchContractError("selected arm IDs must be unique")
        omitted_ids = {
            str(item.get("arm_id", ""))
            for item in omitted_arms or ()
            if isinstance(item, Mapping)
        }
        selected = []
        for arm_id in requested:
            if arm_id not in by_arm:
                qualifier = "omitted" if arm_id in omitted_ids else "absent"
                raise LaunchContractError(
                    f"selected arm {arm_id} is {qualifier} from the plans receipt"
                )
            if by_arm[arm_id].get("updates") is not True:
                raise LaunchContractError(f"selected arm {arm_id} does not update")
            selected.append(by_arm[arm_id])

    selected_ids = {str(plan["arm_id"]) for plan in selected}
    unselected = [
        str(plan["arm_id"])
        for plan in updating
        if str(plan["arm_id"]) not in selected_ids
    ]
    return selected, omitted_no_update, unselected


def plan_launches(
    plans: Mapping[str, Any] | str | Path,
    *,
    gpu_ids: Sequence[int],
    manifest_path: str | Path,
    selected_arm_ids: Sequence[str] | None = None,
    plans_receipt_path: str | Path | None = None,
    runner_entry: str | Path = DEFAULT_RUNNER_ENTRY,
    runner_entry_contract: str = RUNNER_ENTRY_CONTRACT,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    payload = _load_plans(plans)
    if payload.get("schema_version") != "human13_materialized_plans.v1":
        raise LaunchContractError("materialized plans schema differs")
    raw_all_plans = payload.get("plans")
    if not isinstance(raw_all_plans, list) or not raw_all_plans:
        raise LaunchContractError("materialized receipt has no arm plans")
    if not all(isinstance(plan, Mapping) for plan in raw_all_plans):
        raise LaunchContractError("every materialized arm plan must be an object")
    all_plans = list(raw_all_plans)
    raw_plans, omitted_no_update, unselected = _selected_updated_plans(
        all_plans,
        selected_arm_ids,
        payload.get("omitted_arms"),
    )
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

    outputs = [str(plan.get("output_root", "")) for plan in raw_plans]
    states = [str(plan.get("optimizer_state_root", "")) for plan in raw_plans]
    if not all(outputs) or len(set(outputs)) != len(outputs):
        raise LaunchContractError("output roots must be nonempty and unique")
    if not all(states) or len(set(states)) != len(states):
        raise LaunchContractError("optimizer state roots must be nonempty and unique")
    if any(Path(root).exists() for root in outputs):
        raise LaunchContractError("refusing to overwrite an existing arm output root")

    plans_path = _bound_plans_path(plans, payload, plans_receipt_path)
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    root = Path(repo_root).expanduser().resolve(strict=True)
    runner = Path(runner_entry).expanduser().resolve(strict=True)
    if runner != DEFAULT_RUNNER_ENTRY.resolve(strict=True):
        raise LaunchContractError("runner entry must be the real Human-13 live CLI")
    if runner_entry_contract != RUNNER_ENTRY_CONTRACT:
        raise LaunchContractError("runner entry contract differs")

    jobs: list[dict[str, Any]] = []
    for plan, gpu_id in zip(raw_plans, checked_gpus, strict=True):
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
            str(runner),
            "--plans-receipt",
            str(plans_path),
            "--arm-id",
            str(plan["arm_id"]),
            "--manifest",
            str(manifest),
            "--repo-root",
            str(root),
            "--max-updates",
            "16",
            "--execute",
            "--user-model-gpu-authority",
        ]
        jobs.append(
            {
                "arm_id": str(plan["arm_id"]),
                "gpu_id": gpu_id,
                "world_size": 1,
                "output_root": str(plan["output_root"]),
                "optimizer_state_root": plan["optimizer_state_root"],
                "command": command,
                "environment": {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
                "retry_policy": "none",
                "runner_entry_contract": RUNNER_ENTRY_CONTRACT,
                "execution_ready": True,
                "execution_block_reason": None,
            }
        )
    return {
        "schema_version": "human13_launch_plan.v1",
        "mode": "dry_run",
        "actions": dict(ZERO_LAUNCH_ACTIONS),
        "repo_root": str(root),
        "plans_receipt_path": str(plans_path),
        "manifest_path": str(manifest),
        "max_concurrent_jobs": len(jobs),
        "omitted_no_update_arms": omitted_no_update,
        "unselected_updated_arms": unselected,
        "jobs": jobs,
    }


def execute_launches(
    launch_plan: Mapping[str, Any],
    *,
    execution_authorized: bool,
    process_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    if execution_authorized is not True:
        raise LaunchContractError(
            "execute mode requires separate user model/GPU launch authority"
        )
    if launch_plan.get("schema_version") != "human13_launch_plan.v1":
        raise LaunchContractError("launch plan schema differs")
    jobs = launch_plan.get("jobs")
    if not isinstance(jobs, list) or not jobs or len(jobs) > MAX_JOBS:
        raise LaunchContractError("execute requires between one and eight arm jobs")
    if launch_plan.get("actions") != ZERO_LAUNCH_ACTIONS:
        raise LaunchContractError("execute requires an unconsumed dry-run plan")

    processes: list[tuple[Mapping[str, Any], Any]] = []
    root = str(launch_plan["repo_root"])
    for job in jobs:
        if (
            not isinstance(job, Mapping)
            or job.get("world_size") != 1
            or job.get("retry_policy") != "none"
            or job.get("execution_ready") is not True
            or job.get("runner_entry_contract") != RUNNER_ENTRY_CONTRACT
        ):
            raise LaunchContractError("arm job does not satisfy the live CLI contract")
        command = job.get("command")
        environment = job.get("environment")
        if not isinstance(command, list) or not isinstance(environment, Mapping):
            raise LaunchContractError("arm command/environment is malformed")
        required_flags = (
            "--plans-receipt",
            "--arm-id",
            "--manifest",
            "--execute",
            "--user-model-gpu-authority",
        )
        if any(flag not in command for flag in required_flags):
            raise LaunchContractError("arm command is not content/authority bound")
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in environment.items()})
        processes.append((job, process_factory(command, env=env, cwd=root)))

    completed = []
    failures = []
    for job, process in processes:
        returncode = int(process.wait())
        item = {
            "arm_id": str(job["arm_id"]),
            "gpu_id": int(job["gpu_id"]),
            "returncode": returncode,
        }
        completed.append(item)
        if returncode != 0:
            failures.append(item)
    if failures:
        raise LaunchContractError(
            "arm job(s) failed without retry: "
            + ", ".join(f"{item['arm_id']}={item['returncode']}" for item in failures)
        )
    return {
        "schema_version": "human13_launch_execution.v1",
        "mode": "execute",
        "actions": {
            "process_starts": len(processes),
            "gpu_jobs": len(processes),
            "retries": 0,
        },
        "jobs": completed,
    }


def _gpu_ids(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "gpus must be comma-separated integers"
        ) from exc


def _arm_ids(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arms", type=_arm_ids)
    parser.add_argument("--gpus", type=_gpu_ids, required=True)
    parser.add_argument(
        "--runner-entry",
        type=Path,
        default=DEFAULT_RUNNER_ENTRY,
    )
    parser.add_argument(
        "--runner-entry-contract",
        choices=(RUNNER_ENTRY_CONTRACT,),
        default=RUNNER_ENTRY_CONTRACT,
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
        manifest_path=args.manifest,
        selected_arm_ids=args.arms,
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

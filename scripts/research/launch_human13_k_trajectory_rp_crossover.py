#!/usr/bin/env python3
"""Static dry-run-first launcher for the Human-13 K-Trajectory RP-Crossover screen.

This module owns two CPU-only, no-model, no-GPU, no-subprocess concerns:

* ``materialize`` -- load the six frozen leaf configs and build the exact
  static DAG of six matrix acquisitions (feeding eighteen A/B/C cells), the
  two qualification acquisitions (feeding two excluded C-only cells), and the
  two immutable evaluation-RP Source baselines that every cell depends on but
  which are never themselves cells.
* ``launch`` -- assign at most eight explicit unique GPU ids to the eight
  live acquisition-level nodes and build their static, no-retry launch
  commands.  Dry-run (the default) never writes, loads a model, allocates a
  GPU, or starts a subprocess; ``--execute`` additionally requires the
  explicit ``--user-model-gpu-authority`` acknowledgement.

Each live node bundles the sampling for one ``(training RP, seed group)``
acquisition together with its nested A/B/C (or qualification-only C) cells,
which share one evidence identity, fresh independent Source/optimizer state,
and exactly one update.  A node's own terminal receipt records that whole
job's outcome; deeper per-cell evidence is the concern of the not-yet-built
runtime and analyzer.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any

import yaml

from scripts.research.human13_rp_crossover_matrix_contracts import (
    ARM_IDS,
    DRY_RUN_COUNTER_KEYS,
    EVALUATION_RPS,
    MATRIX_SEED_GROUPS,
    PHASE_MATRIX,
    PHASE_QUALIFICATION,
    QUALIFICATION_SEED_GROUP,
    TRAINING_RPS,
    AcquisitionKey,
    CellKey,
)


UNIT_ID = "2026-08-14-human13-k-trajectory-rp-crossover-screen"
LEAF_CONFIG_SCHEMA = "human13_rp_crossover_leaf_config.v1"
DAG_PLAN_SCHEMA = "human13_rp_crossover_dag_plan.v1"
LAUNCH_PLAN_SCHEMA = "human13_rp_crossover_launch_plan.v1"
EXECUTION_SCHEMA = "human13_rp_crossover_launch_execution.v1"
RUNNER_ENTRY_CONTRACT = "human13_rp_crossover_runner_cli.v1"
MAX_LIVE_NODES = 8

CONFIG_ROOT = Path("configs/coordexp_swift/research/human13_k_trajectory_rp_crossover")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER_ENTRY = REPO_ROOT / "scripts/research/train_human13_rp_crossover_cell.py"
ARTIFACT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-14-human13-k-trajectory-rp-crossover-screen"
)
_ZERO_ACTIONS: Mapping[str, int] = MappingProxyType(
    dict.fromkeys(DRY_RUN_COUNTER_KEYS, 0)
)

_OBJECTIVE_COMPONENTS_BY_ARM: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "A": ("trajectory",),
        "B": ("trajectory", "compiler"),
        "C": ("trajectory", "compiler", "preservation"),
    }
)
_COMPILER_COEFFICIENT = 1.0
_FROZEN_OPTIMIZER: Mapping[str, Any] = MappingProxyType(
    {
        "name": "adamw_torch",
        "learning_rate": 3.0e-6,
        "betas": (0.9, 0.999),
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
    }
)
_LEAF_FIELDS = frozenset(
    {
        "schema_version",
        "unit_id",
        "training_rp",
        "arm_id",
        "arm_name",
        "objective_components",
        "compiler_coefficient",
        "evaluation_rps",
        "optimizer",
        "max_updates",
        "retry_policy",
    }
)
_EXPECTED_LEAF_IDENTITY: Mapping[str, tuple[float, str]] = MappingProxyType(
    {
        "01_rp100_trajectory.yaml": (1.0, "A"),
        "02_rp100_trajectory_compiler.yaml": (1.0, "B"),
        "03_rp100_trajectory_compiler_preservation.yaml": (1.0, "C"),
        "04_rp110_trajectory.yaml": (1.10, "A"),
        "05_rp110_trajectory_compiler.yaml": (1.10, "B"),
        "06_rp110_trajectory_compiler_preservation.yaml": (1.10, "C"),
    }
)
_EXPECTED_LEAF_FILENAMES: tuple[str, ...] = tuple(_EXPECTED_LEAF_IDENTITY)


class LaunchContractError(ValueError):
    """Raised when the RP-crossover DAG or launch topology is not fail-closed."""


class LeafConfigError(LaunchContractError):
    """Raised when a leaf config drifts from the frozen per-arm contract."""


@dataclass(frozen=True)
class LeafConfig:
    """One ``(training RP, arm)`` descriptive leaf: frozen differences only."""

    training_rp: float
    arm_id: str
    arm_name: str
    objective_components: tuple[str, ...]
    compiler_coefficient: float | None
    evaluation_rps: tuple[float, ...]
    optimizer: Mapping[str, Any]
    max_updates: int
    retry_policy: str


def _leaf_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LeafConfigError(f"{label} must be an object")
    return value


def load_leaf_config(path: str | Path) -> LeafConfig:
    config_path = Path(path)
    raw = _leaf_mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), config_path.name
    )
    unknown = set(raw) - _LEAF_FIELDS
    missing = _LEAF_FIELDS - set(raw)
    if unknown:
        raise LeafConfigError(
            f"{config_path.name} has unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise LeafConfigError(
            f"{config_path.name} is missing fields: {sorted(missing)}"
        )
    if raw["schema_version"] != LEAF_CONFIG_SCHEMA:
        raise LeafConfigError(f"{config_path.name} schema_version differs")
    if raw["unit_id"] != UNIT_ID:
        raise LeafConfigError(f"{config_path.name} unit_id differs")

    raw_rp = raw["training_rp"]
    if isinstance(raw_rp, bool):
        raise LeafConfigError(f"{config_path.name} training_rp must be numeric")
    training_rp = float(raw_rp)
    if training_rp not in TRAINING_RPS:
        raise LeafConfigError(
            f"{config_path.name} training_rp must be exactly 1.0 or 1.10"
        )

    arm_id = raw["arm_id"]
    if arm_id not in ARM_IDS:
        raise LeafConfigError(f"{config_path.name} arm_id must be exactly A, B, or C")

    components = tuple(raw["objective_components"])
    if components != _OBJECTIVE_COMPONENTS_BY_ARM[arm_id]:
        raise LeafConfigError(
            f"{config_path.name} objective_components differ from the canonical "
            "arm-nested surface"
        )

    expected_coefficient = None if arm_id == "A" else _COMPILER_COEFFICIENT
    coefficient = raw["compiler_coefficient"]
    if coefficient is not None:
        if isinstance(coefficient, bool):
            raise LeafConfigError(
                f"{config_path.name} compiler_coefficient must be numeric"
            )
        coefficient = float(coefficient)
    if coefficient != expected_coefficient:
        raise LeafConfigError(
            f"{config_path.name} compiler_coefficient differs from the frozen contract"
        )

    evaluation_rps = tuple(float(item) for item in raw["evaluation_rps"])
    if evaluation_rps != EVALUATION_RPS:
        raise LeafConfigError(
            f"{config_path.name} evaluation_rps must declare exactly the two canonical RPs"
        )

    optimizer = _leaf_mapping(raw["optimizer"], f"{config_path.name} optimizer")
    if set(optimizer) != set(_FROZEN_OPTIMIZER):
        raise LeafConfigError(f"{config_path.name} optimizer fields differ")
    betas = optimizer["betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise LeafConfigError(
            f"{config_path.name} optimizer.betas must contain two values"
        )
    normalized_optimizer = {
        "name": str(optimizer["name"]),
        "learning_rate": float(optimizer["learning_rate"]),
        "betas": (float(betas[0]), float(betas[1])),
        "epsilon": float(optimizer["epsilon"]),
        "weight_decay": float(optimizer["weight_decay"]),
    }
    if normalized_optimizer != dict(_FROZEN_OPTIMIZER):
        raise LeafConfigError(
            f"{config_path.name} optimizer differs from the frozen AdamW contract"
        )

    if raw["max_updates"] != 1:
        raise LeafConfigError(f"{config_path.name} max_updates must be exactly 1")
    if raw["retry_policy"] != "none":
        raise LeafConfigError(f"{config_path.name} retry_policy must be exactly 'none'")

    return LeafConfig(
        training_rp=training_rp,
        arm_id=arm_id,
        arm_name=str(raw["arm_name"]),
        objective_components=components,
        compiler_coefficient=coefficient,
        evaluation_rps=evaluation_rps,
        optimizer=MappingProxyType(normalized_optimizer),
        max_updates=1,
        retry_policy="none",
    )


def load_leaf_configs(config_root: str | Path = CONFIG_ROOT) -> tuple[LeafConfig, ...]:
    root = Path(config_root)
    paths = sorted(root.glob("*.yaml"))
    if (
        len(paths) != 6
        or tuple(path.name for path in paths) != _EXPECTED_LEAF_FILENAMES
    ):
        raise LeafConfigError(
            "Human-13 RP-crossover requires exactly six canonically named leaf configs"
        )
    configs = tuple(load_leaf_config(path) for path in paths)
    for path, config in zip(paths, configs, strict=True):
        expected = _EXPECTED_LEAF_IDENTITY[path.name]
        if (config.training_rp, config.arm_id) != expected:
            raise LeafConfigError(
                f"{path.name} filename disagrees with its declared training_rp/arm_id"
            )
    pairs = {(config.training_rp, config.arm_id) for config in configs}
    expected_pairs = {(rp, arm) for rp in TRAINING_RPS for arm in ARM_IDS}
    if pairs != expected_pairs or len(pairs) != len(configs):
        raise LeafConfigError(
            "Human-13 RP-crossover leaf configs must cover the 2x3 RP/arm surface exactly once"
        )
    return configs


def _rp_slug(training_rp: float) -> str:
    return "rp100" if training_rp == 1.0 else "rp110"


def _node_id(training_rp: float, seed_group_id: str) -> str:
    return f"{_rp_slug(training_rp)}:{seed_group_id}"


def build_dag_plan(
    leaf_configs: Sequence[LeafConfig],
    *,
    run_id: str,
    output_root: str | Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Build the exact static DAG: six matrix acquisitions, two qualification

    acquisitions, and the two immutable Source-baseline prerequisites.  Pure
    and CPU-only: touches no filesystem path and starts nothing.
    """

    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise LaunchContractError("run_id must be one nonempty path-safe component")
    configs = tuple(leaf_configs)
    if len(configs) != 6:
        raise LaunchContractError(
            "Human-13 RP-crossover requires exactly six leaf configs"
        )
    by_key = {(config.training_rp, config.arm_id): config for config in configs}
    expected_pairs = {(rp, arm) for rp in TRAINING_RPS for arm in ARM_IDS}
    if set(by_key) != expected_pairs or len(by_key) != 6:
        raise LaunchContractError(
            "leaf configs must cover the 2x3 RP/arm surface exactly once"
        )

    root = Path(output_root)
    acquisitions: list[dict[str, Any]] = []
    matrix_cell_keys: list[str] = []
    qualification_cell_keys: list[str] = []

    def _acquisition_entry(
        training_rp: float, seed_group_id: str, phase: str, arm_ids: Sequence[str]
    ) -> dict[str, Any]:
        acquisition_key = AcquisitionKey(
            training_rp=training_rp, seed_group_id=seed_group_id, phase=phase
        )
        acquisition_sha = acquisition_key.content_sha256
        node_root = root / run_id / _rp_slug(training_rp) / seed_group_id
        cells: list[dict[str, Any]] = []
        for arm_id in arm_ids:
            leaf_config = by_key[(training_rp, arm_id)]
            cell_key = CellKey(acquisition_key=acquisition_key, arm_id=arm_id)
            cell_sha = cell_key.content_sha256
            proposal_id = hashlib.sha256(
                f"{run_id}\0{cell_sha}\0fresh-adamw-proposal".encode("utf-8")
            ).hexdigest()
            cells.append(
                {
                    "cell_key": cell_key.to_dict(),
                    "cell_key_sha256": cell_sha,
                    "arm_name": leaf_config.arm_name,
                    "objective_components": list(leaf_config.objective_components),
                    "compiler_coefficient": leaf_config.compiler_coefficient,
                    "shared_evidence_group_sha256": acquisition_sha,
                    "output_root": str(node_root / arm_id.lower()),
                    "proposal_id": proposal_id,
                    "source": "fresh",
                    "optimizer": "fresh_adamw",
                    "world_size": 1,
                    "max_updates": leaf_config.max_updates,
                    "retry_policy": leaf_config.retry_policy,
                    "evaluation_rps": list(leaf_config.evaluation_rps),
                }
            )
            (
                matrix_cell_keys if phase == PHASE_MATRIX else qualification_cell_keys
            ).append(cell_sha)
        return {
            "acquisition_key": acquisition_key.to_dict(),
            "acquisition_key_sha256": acquisition_sha,
            "node_id": _node_id(training_rp, seed_group_id),
            "phase": phase,
            "shared_evidence_group_sha256": acquisition_sha,
            "receipt_path": str(node_root / "receipt.json"),
            "cells": cells,
        }

    for training_rp in TRAINING_RPS:
        for seed_group_id in MATRIX_SEED_GROUPS:
            acquisitions.append(
                _acquisition_entry(training_rp, seed_group_id, PHASE_MATRIX, ARM_IDS)
            )
    for training_rp in TRAINING_RPS:
        acquisitions.append(
            _acquisition_entry(
                training_rp, QUALIFICATION_SEED_GROUP, PHASE_QUALIFICATION, ("C",)
            )
        )

    if len(acquisitions) != MAX_LIVE_NODES:
        raise LaunchContractError(
            "Human-13 RP-crossover DAG must bind exactly eight live acquisition nodes"
        )
    output_roots = [cell["output_root"] for a in acquisitions for cell in a["cells"]]
    proposal_ids = [cell["proposal_id"] for a in acquisitions for cell in a["cells"]]
    if len(output_roots) != 20 or len(set(output_roots)) != 20:
        raise LaunchContractError("every cell output_root must be unique")
    if len(proposal_ids) != 20 or len(set(proposal_ids)) != 20:
        raise LaunchContractError(
            "every cell must have an independent fresh AdamW proposal identity"
        )
    if len(matrix_cell_keys) != 18 or len(set(matrix_cell_keys)) != 18:
        raise LaunchContractError("matrix disposition must bind exactly eighteen cells")
    if len(qualification_cell_keys) != 2 or len(set(qualification_cell_keys)) != 2:
        raise LaunchContractError(
            "qualification disposition must bind exactly two cells"
        )
    receipt_paths = [a["receipt_path"] for a in acquisitions]
    if len(receipt_paths) != len(set(receipt_paths)):
        raise LaunchContractError(
            "every acquisition node must have a unique terminal receipt path"
        )

    source_baselines = [
        {
            "evaluation_rp": rp,
            "role": "immutable_prerequisite",
            "baseline_id": f"source_baseline:{rp}",
        }
        for rp in EVALUATION_RPS
    ]
    all_cell_keys = [*matrix_cell_keys, *qualification_cell_keys]
    dependency_edges = [
        [a["acquisition_key_sha256"], cell["cell_key_sha256"]]
        for a in acquisitions
        for cell in a["cells"]
    ]
    dependency_edges.extend(
        [baseline["baseline_id"], cell_sha]
        for baseline in source_baselines
        for cell_sha in all_cell_keys
    )

    return {
        "schema_version": DAG_PLAN_SCHEMA,
        "unit_id": UNIT_ID,
        "run_id": run_id,
        "mode": "dry_run",
        "actions": dict(_ZERO_ACTIONS),
        "concurrency_cap": MAX_LIVE_NODES,
        "source_baselines": source_baselines,
        "acquisitions": acquisitions,
        "matrix_cell_keys": matrix_cell_keys,
        "qualification_cell_keys": qualification_cell_keys,
        "dependency_edges": dependency_edges,
    }


def _load_dag_plan_mapping(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value).expanduser().resolve(strict=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise LaunchContractError("dag plan file must contain a JSON object")
    return raw


def _dag_plan_reference(
    dag_plan: Mapping[str, Any] | str | Path,
    payload: Mapping[str, Any],
    dag_plan_path: str | Path | None,
) -> str:
    if isinstance(dag_plan, Mapping):
        if dag_plan_path is None:
            raise LaunchContractError(
                "an in-memory dag plan requires an explicit dag_plan_path bound to an "
                "existing file before it can be turned into live-node commands"
            )
        path = Path(dag_plan_path).expanduser().resolve(strict=True)
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise LaunchContractError(
                "dag_plan_path content differs from the planned payload"
            )
        return str(path)
    path = Path(dag_plan).expanduser().resolve(strict=True)
    if dag_plan_path is not None:
        requested = Path(dag_plan_path).expanduser().resolve(strict=True)
        if requested != path:
            raise LaunchContractError("dag plan paths disagree")
    return str(path)


def plan_launches(
    dag_plan: Mapping[str, Any] | str | Path,
    *,
    gpu_ids: Sequence[int],
    node_ids: Sequence[str] | None = None,
    dag_plan_path: str | Path | None = None,
    runner_entry: str | Path = DEFAULT_RUNNER_ENTRY,
    runner_entry_contract: str = RUNNER_ENTRY_CONTRACT,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    payload = _load_dag_plan_mapping(dag_plan)
    if payload.get("schema_version") != DAG_PLAN_SCHEMA:
        raise LaunchContractError("dag plan schema_version differs")
    if payload.get("mode") != "dry_run" or dict(payload.get("actions", {})) != dict(
        _ZERO_ACTIONS
    ):
        raise LaunchContractError("dag plan must be an unconsumed dry-run plan")
    acquisitions = payload.get("acquisitions")
    if not isinstance(acquisitions, list) or len(acquisitions) != MAX_LIVE_NODES:
        raise LaunchContractError(
            "dag plan must bind exactly eight live acquisition nodes"
        )

    by_node = {str(item.get("node_id", "")): item for item in acquisitions}
    if "" in by_node or len(by_node) != len(acquisitions):
        raise LaunchContractError(
            "acquisition node identities must be nonempty and unique"
        )

    if node_ids is None:
        selected_ids = tuple(item["node_id"] for item in acquisitions)
    else:
        selected_ids = tuple(node_ids)
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise LaunchContractError("selected node IDs must be nonempty and unique")
        for node_id in selected_ids:
            if node_id not in by_node:
                raise LaunchContractError(
                    f"selected node is absent from the dag plan: {node_id}"
                )
    selected = [by_node[node_id] for node_id in selected_ids]
    if len(selected) > MAX_LIVE_NODES:
        raise LaunchContractError(
            "Human-13 RP-crossover launcher supports at most eight live nodes"
        )

    checked_gpus = tuple(gpu_ids)
    if len(checked_gpus) != len(selected):
        raise LaunchContractError("one unique GPU ID is required for every live node")
    if len(set(checked_gpus)) != len(checked_gpus):
        raise LaunchContractError("GPU IDs must be unique")
    if any(
        isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0
        for gpu in checked_gpus
    ):
        raise LaunchContractError("GPU IDs must be non-negative integers")

    dag_path = _dag_plan_reference(dag_plan, payload, dag_plan_path)
    root = Path(repo_root).expanduser().resolve(strict=True)
    runner = Path(runner_entry).expanduser().resolve(strict=True)
    if runner_entry_contract != RUNNER_ENTRY_CONTRACT:
        raise LaunchContractError("runner entry contract differs")

    if any(Path(item["receipt_path"]).exists() for item in selected):
        raise LaunchContractError("refusing to overwrite an existing node receipt")
    if any(
        Path(cell["output_root"]).exists()
        for item in selected
        for cell in item["cells"]
    ):
        raise LaunchContractError("refusing to overwrite an existing cell output root")

    jobs: list[dict[str, Any]] = []
    for acquisition, gpu_id in zip(selected, checked_gpus, strict=True):
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
            "--dag-plan",
            dag_path,
            "--node-id",
            acquisition["node_id"],
            "--receipt-path",
            acquisition["receipt_path"],
            "--repo-root",
            str(root),
            "--max-updates",
            "1",
            "--execute",
            "--user-model-gpu-authority",
        ]
        jobs.append(
            {
                "node_id": acquisition["node_id"],
                "phase": acquisition["phase"],
                "gpu_id": gpu_id,
                "world_size": 1,
                "receipt_path": acquisition["receipt_path"],
                "cell_count": len(acquisition["cells"]),
                "command": command,
                "environment": {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
                "retry_policy": "none",
                "runner_entry_contract": RUNNER_ENTRY_CONTRACT,
                "execution_ready": True,
            }
        )

    unselected = [node_id for node_id in by_node if node_id not in set(selected_ids)]
    return {
        "schema_version": LAUNCH_PLAN_SCHEMA,
        "mode": "dry_run",
        "actions": dict(_ZERO_ACTIONS),
        "repo_root": str(root),
        "dag_plan_path": dag_path,
        "runner_entry": str(runner),
        "max_concurrent_jobs": len(jobs),
        "unselected_node_ids": unselected,
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
    if launch_plan.get("schema_version") != LAUNCH_PLAN_SCHEMA:
        raise LaunchContractError("launch plan schema differs")
    jobs = launch_plan.get("jobs")
    if not isinstance(jobs, list) or not jobs or len(jobs) > MAX_LIVE_NODES:
        raise LaunchContractError(
            "execute requires between one and eight live node jobs"
        )
    if dict(launch_plan.get("actions", {})) != dict(_ZERO_ACTIONS):
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
            raise LaunchContractError(
                "live node job does not satisfy the launch contract"
            )
        command = job.get("command")
        environment = job.get("environment")
        if not isinstance(command, list) or not isinstance(environment, Mapping):
            raise LaunchContractError("node command/environment is malformed")
        required_flags = (
            "--dag-plan",
            "--node-id",
            "--receipt-path",
            "--execute",
            "--user-model-gpu-authority",
        )
        if any(flag not in command for flag in required_flags):
            raise LaunchContractError("node command is not content/authority bound")
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in environment.items()})
        processes.append((job, process_factory(command, env=env, cwd=root)))

    completed = []
    failures = []
    for job, process in processes:
        returncode = int(process.wait())
        item = {
            "node_id": str(job["node_id"]),
            "gpu_id": int(job["gpu_id"]),
            "returncode": returncode,
        }
        completed.append(item)
        if returncode != 0:
            failures.append(item)
    if failures:
        raise LaunchContractError(
            "live node job(s) failed without retry: "
            + ", ".join(f"{item['node_id']}={item['returncode']}" for item in failures)
        )
    return {
        "schema_version": EXECUTION_SCHEMA,
        "mode": "execute",
        "actions": {
            "model_loads": 0,
            "gpu_allocations": len(processes),
            "subprocess_launches": len(processes),
            "output_roots_created": 0,
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


def _node_ids(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser(
        "materialize",
        help="Build the static dry-run DAG plan; touches no filesystem path.",
    )
    materialize.add_argument("--config-root", type=Path, default=CONFIG_ROOT)
    materialize.add_argument("--output-root", type=Path, default=ARTIFACT_ROOT)
    materialize.add_argument("--run-id", required=True)
    materialize.add_argument(
        "--write-dag-plan",
        type=Path,
        help="Explicitly write the DAG plan; default emits JSON to stdout only.",
    )

    launch = subparsers.add_parser(
        "launch", help="Assign GPUs to live nodes and plan or execute their jobs."
    )
    launch.add_argument("--dag-plan", type=Path, required=True)
    launch.add_argument("--nodes", type=_node_ids)
    launch.add_argument("--gpus", type=_gpu_ids, required=True)
    launch.add_argument("--runner-entry", type=Path, default=DEFAULT_RUNNER_ENTRY)
    launch.add_argument(
        "--runner-entry-contract",
        choices=(RUNNER_ENTRY_CONTRACT,),
        default=RUNNER_ENTRY_CONTRACT,
    )
    launch.add_argument("--execute", action="store_true")
    launch.add_argument(
        "--user-model-gpu-authority",
        action="store_true",
        help="Acknowledge separately documented user execution authority.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize":
        configs = load_leaf_configs(args.config_root)
        plan = build_dag_plan(configs, run_id=args.run_id, output_root=args.output_root)
        encoded = json.dumps(plan, sort_keys=True, indent=2) + "\n"
        if args.write_dag_plan is not None:
            destination = args.write_dag_plan.resolve()
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite dag plan: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0

    launch_plan = plan_launches(
        args.dag_plan,
        gpu_ids=args.gpus,
        node_ids=args.nodes,
        runner_entry=args.runner_entry,
        runner_entry_contract=args.runner_entry_contract,
    )
    receipt = (
        execute_launches(
            launch_plan, execution_authorized=args.user_model_gpu_authority
        )
        if args.execute
        else launch_plan
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

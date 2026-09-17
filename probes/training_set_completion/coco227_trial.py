"""Prepare, qualify, and control the frozen COCO227 S/T training pair."""

from __future__ import annotations


from src.runtime.owned_process import spawn_logged_process, terminate_owned_process

import argparse
import copy
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

import torch

from probes.training_set_completion import coco227_training as backend
from probes.training_set_completion import dual_start
from probes.training_set_completion import training
from src.runtime.process_completion import next_process_completion, start_process_waiter


SCHEMA = "training_set_completion.coco227_ce_normalization.v1"
REPO = Path(__file__).resolve().parents[2]
BASE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization"
)
OLD_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3"
)
SOURCE_MANIFEST = OLD_ROOT / "A/training-manifest.json"
SOURCE_ADAPTER = OLD_ROOT / "A/training/checkpoints/step-00256/adapter"
TEACHER = BASE / "data-v1/bank.json"
ARMS = ("S", "T")
ARM_REDUCTIONS = {"S": "sample_equal", "T": "global_active_token_equal"}
CHECKPOINT_STEPS = (8, 16, 32, 64, 128, 256)
TRAINING_GPU_GROUPS = {"S": [0, 1, 2, 3], "T": [4, 5, 6, 7]}
TRAIN_WALL_SECONDS = 7_200
READBACK_PHASE_SECONDS = 7_200
QUALIFICATION_WALL_SECONDS = 600
TRAINING_QUALIFICATION_TMUX_SESSION = "coordexp-coco227-training-qualification"
TRIAL_TMUX_SESSION = "coordexp-coco227-ce-normalization-trial"
QUALIFICATION_CONFIGS = tuple(
    {
        "id": f"{arm}-mb{microbatch}-checkpointed",
        "arm": arm,
        "microbatch_size": microbatch,
        "activation_checkpointing": True,
    }
    for arm in ARMS
    for microbatch in (1, 2, 3)
)

# Declared before reading any real batched difference.
PARITY_TOLERANCES = {
    "loss_max_abs": 5e-5,
    "gradient_max_abs": 5e-4,
    "gradient_relative_l2": 1e-4,
    "parameter_max_abs": 5e-5,
    "parameter_relative_l2": 1e-5,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _verify(binding: Mapping[str, Any], label: str) -> None:
    require(training.binding(binding["path"]) == dict(binding), f"{label} changed")


def validate_teacher(
    value: Mapping[str, Any], *, source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    require(
        value.get("schema")
        == "training_set_completion.coco227_ce_normalization_teacher.v1",
        "teacher schema",
    )
    require(
        value.get("status") == "candidate_ready"
        and value.get("fixed_owner_count") == 227,
        "teacher status/count",
    )
    require(
        value.get("old218_owner_count") == 218 and value.get("new9_owner_count") == 9,
        "teacher old/new partition",
    )
    require(
        len(value.get("old218_owner_ids", [])) == 218
        and len(value.get("new9_owner_ids", [])) == 9,
        "teacher owner ledgers",
    )
    routes = value.get("routes")
    require(isinstance(routes, list) and len(routes) == 11, "teacher route denominator")
    checked = [
        training.validate_route(
            route,
            eos_token_id=source_manifest["runtime"]["eos_token_id"],
            coordinate_token_ids=source_manifest["validity_hinge"][
                "coordinate_token_ids"
            ],
        )
        for route in routes
    ]
    require(
        sum(len(route["trusted_boxes"]) for route in checked) == 227,
        "teacher box count",
    )
    require(
        sum(sum(route["ce_weights"]) for route in checked) == 2176,
        "teacher active-token count",
    )
    require(
        [route["image_id"] for route in checked]
        == [route["image_id"] for route in source_manifest["routes"]],
        "teacher image order",
    )
    content = {key: item for key, item in value.items() if key != "content_sha256"}
    require(
        value.get("content_sha256") == training.digest(content), "teacher content hash"
    )
    return dict(value)


def build_training_manifest(
    *,
    teacher_path: Path,
    output: Path,
    training_output: Path,
    arm: str,
    updates: int,
    checkpoint_steps: Sequence[int],
    wall_seconds: int,
    microbatch_size: int,
    activation_checkpointing: bool,
    capture_gradient_step1: bool,
) -> dict[str, Any]:
    from probes.dora_owner_learning.route_access import checkpoint_config
    from src.config.inference import InferConfig

    require(arm in ARMS, "training arm")
    source = training.validate_manifest(read(SOURCE_MANIFEST))
    teacher = validate_teacher(read(teacher_path), source_manifest=source)
    adapter = training.inspect_dora_adapter_payload(
        SOURCE_ADAPTER, source["model_config"]["model"]["base_model"]
    )
    require(
        adapter["fingerprint"] == backend.SOURCE_ADAPTER_FINGERPRINT,
        "common source adapter",
    )
    model_config = copy.deepcopy(source["model_config"])
    model_config["adapter"] = {
        "name": "default",
        "path": str(SOURCE_ADAPTER),
        "type": "dora",
    }
    model_config["run"].update(
        name=f"coco227-{arm}",
        artifact_root=str(training_output),
        output_dir=None,
        collision_policy="fail",
    )
    checkpoint_config(InferConfig.model_validate(model_config), str(SOURCE_ADAPTER))
    active = sum(sum(route["ce_weights"]) for route in teacher["routes"])
    model_calls_per_update = sum(
        math.ceil(count / microbatch_size) for count in (3, 3, 3, 2)
    )
    value = {
        "schema": training.SCHEMA,
        "status": "candidate_ready",
        "sources": {
            "reviewed_routes": training.binding(teacher_path),
            "producer": training.binding(Path(backend.__file__)),
        },
        "acquisition_manifest": source["acquisition_manifest"],
        "source_adapter": adapter,
        "model_config": model_config,
        "routes": copy.deepcopy(teacher["routes"]),
        "optimizer": copy.deepcopy(training.DEFAULT_OPTIMIZER),
        "runtime": {
            "updates": updates,
            "checkpoint_steps": list(checkpoint_steps),
            "wall_seconds": wall_seconds,
            "max_model_forwards": updates * 11,
            "max_model_calls": updates * model_calls_per_update,
            "eos_token_id": 151645,
            "seed": 42,
            "initial_step": 0,
            "optimizer_mode": "fresh",
            "microbatch_size": microbatch_size,
            "activation_checkpointing": activation_checkpointing,
            "global_ce_eligible_images": 11,
            "global_active_tokens": active,
            "capture_gradient_step1": capture_gradient_step1,
        },
        "validity_hinge": copy.deepcopy(source["validity_hinge"]),
        "objective": {
            "ce_reduction": ARM_REDUCTIONS[arm],
            "geometry_reduction": "global_equal_image_mean",
        },
        "teacher_partition": {
            "old218_owner_count": 218,
            "new9_owner_count": 9,
            "old218_owner_ids_sha256": training.digest(teacher["old218_owner_ids"]),
            "new9_owner_ids_sha256": training.digest(teacher["new9_owner_ids"]),
        },
    }
    value["content_sha256"] = training.digest(value)
    backend.validate_manifest(value)
    training.publish(output, value)
    return value


def prepare_qualification(*, teacher_path: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), "qualification output exists")
    output.mkdir(parents=True)
    manifests = []
    for config in QUALIFICATION_CONFIGS:
        path = output / "manifests" / f"{config['id']}.json"
        run_output = output / "runs" / config["id"]
        build_training_manifest(
            teacher_path=teacher_path,
            output=path,
            training_output=run_output,
            arm=config["arm"],
            updates=2,
            checkpoint_steps=[2],
            wall_seconds=QUALIFICATION_WALL_SECONDS,
            microbatch_size=config["microbatch_size"],
            activation_checkpointing=config["activation_checkpointing"],
            capture_gradient_step1=True,
        )
        manifests.append(
            {
                **config,
                "manifest": training.binding(path),
                "output": str(run_output),
                "command": distributed_training_command(
                    manifest_path=path, output=run_output
                ),
            }
        )
    plan = {
        "schema": f"{SCHEMA}.training_qualification_plan.v1",
        "status": "candidate_ready",
        "teacher": training.binding(teacher_path),
        "producer": training.binding(Path(backend.__file__)),
        "controller": training.binding(Path(__file__)),
        "configurations": manifests,
        "limits": {
            "configuration_count": 6,
            "updates_per_configuration": 2,
            "wall_seconds_per_configuration": 600,
            "logical_image_exposures": 132,
            "physical_gpus": [0, 1, 2, 3],
        },
        "parity_tolerances": dict(PARITY_TOLERANCES),
    }
    plan["content_sha256"] = training.digest(plan)
    training.publish(output / "plan.json", plan)
    return plan


def validate_qualification_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    content = {key: item for key, item in value.items() if key != "content_sha256"}
    require(
        value.get("schema") == f"{SCHEMA}.training_qualification_plan.v1"
        and value.get("status") == "candidate_ready"
        and value.get("content_sha256") == training.digest(content),
        "qualification plan",
    )
    require(
        value.get("parity_tolerances") == PARITY_TOLERANCES,
        "qualification parity tolerances",
    )
    require(
        value.get("limits")
        == {
            "configuration_count": 6,
            "updates_per_configuration": 2,
            "wall_seconds_per_configuration": 600,
            "logical_image_exposures": 132,
            "physical_gpus": [0, 1, 2, 3],
        },
        "qualification limits",
    )
    configs = value.get("configurations")
    require(
        isinstance(configs, list)
        and [
            {
                key: item[key]
                for key in ("id", "arm", "microbatch_size", "activation_checkpointing")
            }
            for item in configs
        ]
        == list(QUALIFICATION_CONFIGS),
        "qualification configurations",
    )
    _verify(value["teacher"], "qualification teacher")
    _verify(value["producer"], "qualification producer")
    _verify(value["controller"], "qualification controller")
    for item in configs:
        _verify(item["manifest"], f"{item['id']} manifest")
        backend.validate_manifest(read(item["manifest"]["path"]))
        require(
            item["command"]
            == distributed_training_command(
                manifest_path=Path(item["manifest"]["path"]),
                output=Path(item["output"]),
            ),
            "qualification command",
        )
    return dict(value)


def distributed_training_command(*, manifest_path: Path, output: Path) -> list[str]:
    return [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        "probes.training_set_completion.coco227_training",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    ]


def validate_training_terminal(
    output: Path, *, manifest_path: Path, require_fresh: bool = True
) -> dict[str, Any]:
    manifest = backend.validate_manifest(read(manifest_path))
    terminal = read(output / "terminal.json")
    require(terminal.get("status") == "completed", "training terminal status")
    require(
        not require_fresh or terminal.get("optimizer_mode") == "fresh",
        "training optimizer mode",
    )
    require(
        terminal.get("manifest") == training.binding(manifest_path),
        "training terminal manifest",
    )
    require(
        terminal.get("updates") == manifest["runtime"]["updates"]
        and terminal.get("logical_model_forwards")
        == manifest["runtime"]["max_model_forwards"]
        and terminal.get("model_calls") == manifest["runtime"]["max_model_calls"],
        "training counters",
    )
    require(
        [item["step"] for item in terminal.get("checkpoints", [])]
        == manifest["runtime"]["checkpoint_steps"],
        "checkpoint schedule",
    )
    for entry, consensus in zip(
        terminal["checkpoints"],
        terminal["distributed"]["checkpoint_consensus"],
        strict=True,
    ):
        state = torch.load(
            entry["state"]["path"], map_location="cpu", weights_only=False
        )
        steps = {
            int(item["step"].item())
            for item in state["optimizer_state_dict"]["state"].values()
        }
        require(
            steps == {entry["step"]}
            and consensus["step"] == entry["step"]
            and consensus["state"]["optimizer_steps"] == [entry["step"]]
            and consensus["rank_count"] == 4,
            "optimizer/rank consensus",
        )
    if manifest["runtime"].get("capture_gradient_step1"):
        _verify(terminal["gradient_snapshot_step1"], "qualification gradient snapshot")
    return terminal


def _tensor_difference(
    reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    require(set(reference) == set(candidate), "tensor comparison keys")
    diff_sq = 0.0
    reference_sq = 0.0
    max_abs = 0.0
    for key in sorted(reference):
        left = reference[key].double()
        right = candidate[key].double()
        require(
            left.shape == right.shape
            and bool(torch.isfinite(left).all())
            and bool(torch.isfinite(right).all()),
            "tensor comparison shape/finite",
        )
        delta = left - right
        diff_sq += float(torch.sum(delta * delta))
        reference_sq += float(torch.sum(left * left))
        max_abs = max(max_abs, float(torch.max(torch.abs(delta))))
    return {
        "max_abs": max_abs,
        "relative_l2": math.sqrt(diff_sq / max(reference_sq, 1e-300)),
    }


def _adapter_tensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path / "adapter_model.safetensors"), device="cpu")


def compare_qualification_runs(
    reference_output: Path, candidate_output: Path
) -> dict[str, Any]:
    reference_terminal = read(reference_output / "terminal.json")
    candidate_terminal = read(candidate_output / "terminal.json")
    reference_grad = torch.load(
        reference_terminal["gradient_snapshot_step1"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    candidate_grad = torch.load(
        candidate_terminal["gradient_snapshot_step1"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    gradient = _tensor_difference(reference_grad, candidate_grad)
    reference_cp = Path(reference_terminal["checkpoints"][-1]["adapter"]["root"])
    candidate_cp = Path(candidate_terminal["checkpoints"][-1]["adapter"]["root"])
    parameters = _tensor_difference(
        _adapter_tensors(reference_cp), _adapter_tensors(candidate_cp)
    )
    loss_max = 0.0
    for step in (1, 2):
        left = read(reference_output / "updates" / f"step-{step:05d}.json")
        right = read(candidate_output / "updates" / f"step-{step:05d}.json")
        require(
            [row["route_id"] for row in left["routes"]]
            == [row["route_id"] for row in right["routes"]],
            "qualification route order",
        )
        for field in (
            "global_ce",
            "global_geometry_mean",
            "objective_total",
            "gradient_norm_before_clip",
        ):
            loss_max = max(loss_max, abs(float(left[field]) - float(right[field])))
        for lrow, rrow in zip(left["routes"], right["routes"], strict=True):
            for field in (
                "masked_nll_sum",
                "active_token_mean_ce",
                "raw_axis_validity_hinge",
            ):
                loss_max = max(loss_max, abs(float(lrow[field]) - float(rrow[field])))
    passed = (
        loss_max <= PARITY_TOLERANCES["loss_max_abs"]
        and gradient["max_abs"] <= PARITY_TOLERANCES["gradient_max_abs"]
        and gradient["relative_l2"] <= PARITY_TOLERANCES["gradient_relative_l2"]
        and parameters["max_abs"] <= PARITY_TOLERANCES["parameter_max_abs"]
        and parameters["relative_l2"] <= PARITY_TOLERANCES["parameter_relative_l2"]
    )
    return {
        "passed": passed,
        "loss_max_abs": loss_max,
        "gradient": gradient,
        "step2_parameters": parameters,
        "tolerances": dict(PARITY_TOLERANCES),
    }


def _timing(output: Path) -> dict[str, Any]:
    terminal = read(output / "terminal.json")
    updates = [read(output / "updates" / f"step-{step:05d}.json") for step in (1, 2)]
    forward = [
        max(
            row["forward_backward_seconds"]
            for row in update["distributed"]["rank_timings"]
        )
        for update in updates
    ]
    compute = [
        max(
            row["compute_seconds"] - row["gradient_snapshot_io_seconds"]
            for row in update["distributed"]["rank_timings"]
        )
        for update in updates
    ]
    setup = [
        row["native_preparation"] for row in terminal["distributed"]["rank_receipts"]
    ]
    checkpoint_io = max(
        row["checkpoint_io_seconds"]
        for row in terminal["checkpoint_timings"][0]["rank_timings"]
    )
    resources = [row["resources"] for row in terminal["distributed"]["rank_receipts"]]
    return {
        "max_rank_forward_backward_seconds_per_update": forward,
        "mean_max_rank_forward_backward_seconds": sum(forward) / len(forward),
        "max_rank_compute_seconds_per_update_excluding_snapshot_io": compute,
        "max_rank_model_setup_seconds": max(
            row["model_setup_seconds"] for row in setup
        ),
        "max_rank_native_input_setup_seconds": max(
            row["native_input_setup_seconds"] for row in setup
        ),
        "checkpoint_io_seconds": checkpoint_io,
        "max_rank_peak_rss_kib": max(row["peak_rss_kib"] for row in resources),
        "max_rank_peak_cuda_allocated_bytes": max(
            row["peak_cuda_allocated_bytes"] for row in resources
        ),
        "max_rank_peak_cuda_reserved_bytes": max(
            row["peak_cuda_reserved_bytes"] for row in resources
        ),
        "rank_prompt_padding_tokens": [row["prompt_padding_tokens"] for row in setup],
        "rank_microbatch_image_counts": [
            row["microbatch_image_counts"] for row in setup
        ],
    }


def qualification_controller(*, plan_path: Path, output: Path) -> dict[str, Any]:
    plan = validate_qualification_plan(read(plan_path))
    require(Path(plan_path).parent == output, "qualification plan/output")
    require(not (output / "terminal.json").exists(), "qualification terminal exists")
    tmux_session = subprocess.check_output(
        ["tmux", "display-message", "-p", "#S"], text=True
    ).strip()
    require(
        tmux_session == TRAINING_QUALIFICATION_TMUX_SESSION,
        "qualification controller must run in its named tmux session",
    )
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    exits = []
    started = time.monotonic()
    training.publish(
        output / "launch.json",
        {
            "schema": f"{SCHEMA}.training_qualification_launch.v1",
            "status": "running",
            "pid": os.getpid(),
            "tmux_session": tmux_session,
            "plan": training.binding(plan_path),
            "started_at": time.time(),
        },
    )
    try:
        for config in plan["configurations"]:
            log_path = logs / f"{config['id']}.log"
            stream = log_path.open("x")
            spawned = time.monotonic()
            process = subprocess.Popen(
                config["command"],
                cwd=REPO,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "OMP_NUM_THREADS": "2",
                    "TOKENIZERS_PARALLELISM": "false",
                },
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=QUALIFICATION_WALL_SECONDS)
            except subprocess.TimeoutExpired:
                terminate_owned_process(process)
                code = "timeout"
            finally:
                stream.close()
            exits.append(
                {
                    "id": config["id"],
                    "pid": process.pid,
                    "exit_code": code,
                    "elapsed_seconds": time.monotonic() - spawned,
                    "log": training.binding(log_path),
                    "command": config["command"],
                }
            )
            require(code == 0, f"qualification failed: {config['id']}")
            validate_training_terminal(
                Path(config["output"]), manifest_path=Path(config["manifest"]["path"])
            )
        rows = []
        by = {
            (row["arm"], row["microbatch_size"]): row for row in plan["configurations"]
        }
        for config in plan["configurations"]:
            parity = (
                {"passed": True, "reference": "self"}
                if config["microbatch_size"] == 1
                else compare_qualification_runs(
                    Path(by[(config["arm"], 1)]["output"]), Path(config["output"])
                )
            )
            rows.append(
                {
                    "id": config["id"],
                    "arm": config["arm"],
                    "microbatch_size": config["microbatch_size"],
                    "activation_checkpointing": config["activation_checkpointing"],
                    "manifest": config["manifest"],
                    "terminal": training.binding(
                        Path(config["output"]) / "terminal.json"
                    ),
                    "timing": _timing(Path(config["output"])),
                    "parity_to_same_arm_microbatch1": parity,
                }
            )
        eligible = []
        for microbatch in (1, 2, 3):
            pair = [row for row in rows if row["microbatch_size"] == microbatch]
            if all(row["parity_to_same_arm_microbatch1"]["passed"] for row in pair):
                eligible.append(
                    {
                        "microbatch_size": microbatch,
                        "mean_arm_forward_backward_seconds": sum(
                            row["timing"]["mean_max_rank_forward_backward_seconds"]
                            for row in pair
                        )
                        / 2,
                    }
                )
        require(
            any(item["microbatch_size"] == 1 for item in eligible),
            "serial fallback missing",
        )
        selected = min(
            eligible, key=lambda item: item["mean_arm_forward_backward_seconds"]
        )
        baseline = next(item for item in eligible if item["microbatch_size"] == 1)
        if (
            selected["microbatch_size"] != 1
            and selected["mean_arm_forward_backward_seconds"]
            >= baseline["mean_arm_forward_backward_seconds"]
        ):
            selected = baseline
        receipt = {
            "schema": f"{SCHEMA}.training_qualification.v1",
            "status": "passed",
            "plan": training.binding(plan_path),
            "producer": plan["producer"],
            "controller": plan["controller"],
            "controller_pid": os.getpid(),
            "tmux_session": tmux_session,
            "exits": exits,
            "rows": rows,
            "selected": {
                "microbatch_size": selected["microbatch_size"],
                "activation_checkpointing": True,
                "selection_metric": "mean across S/T of max-rank forward+backward seconds per update",
                "measured_seconds": selected["mean_arm_forward_backward_seconds"],
                "serial_baseline_seconds": baseline[
                    "mean_arm_forward_backward_seconds"
                ],
                "fallback_serial": selected["microbatch_size"] == 1,
            },
            "elapsed_seconds": time.monotonic() - started,
            "observed_logical_image_exposures": sum(
                read(Path(config["output"]) / "terminal.json")["logical_model_forwards"]
                for config in plan["configurations"]
            ),
        }
        require(
            receipt["observed_logical_image_exposures"] == 132,
            "qualification exposure count",
        )
        training.publish(output / "receipt.json", receipt)
        terminal = {
            "schema": f"{SCHEMA}.training_qualification_terminal.v1",
            "status": "completed",
            "exit_code": 0,
            "pid": os.getpid(),
            "tmux_session": tmux_session,
            "receipt": training.binding(output / "receipt.json"),
            "elapsed_seconds": time.monotonic() - started,
        }
    except BaseException as exc:
        terminal = {
            "schema": f"{SCHEMA}.training_qualification_terminal.v1",
            "status": "failed",
            "exit_code": 1,
            "pid": os.getpid(),
            "tmux_session": tmux_session,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "exits": exits,
            "elapsed_seconds": time.monotonic() - started,
        }
    training.publish(output / "terminal.json", terminal)
    if terminal["status"] != "completed":
        raise RuntimeError(terminal["error"])
    return terminal


def prepare_trial(
    *,
    teacher_path: Path,
    qualification_receipt: Path,
    output: Path,
    readback_qualification_result: Path,
    source0_admission: Path,
) -> dict[str, Any]:
    require(not output.exists(), "trial output exists")
    qualification = read(qualification_receipt)
    require(qualification.get("status") == "passed", "training qualification")
    selected = qualification["selected"]
    readback_qualification = read(readback_qualification_result)
    require(
        readback_qualification.get("schema")
        == "training_set_completion.coco227_readback.v1.qualification_result"
        and readback_qualification.get("status") == "completed"
        and readback_qualification.get("selection", {}).get("batch_size") in (1, 2, 3)
        and int(readback_qualification.get("request_count", 10**9)) <= 36,
        "readback qualification",
    )
    source0 = read(source0_admission)
    require(
        source0.get("schema")
        == "training_set_completion.coco227_ce_normalization_evaluation.v1.source0_admission"
        and source0.get("status") == "source0_admitted_common_prior_A_final256"
        and source0.get("source_adapter", {}).get("fingerprint")
        == backend.SOURCE_ADAPTER_FINGERPRINT,
        "source0 admission",
    )
    output.mkdir(parents=True)
    arms = {}
    for arm in ARMS:
        path = output / arm / "training-manifest.json"
        manifest = build_training_manifest(
            teacher_path=teacher_path,
            output=path,
            training_output=output / arm / "training",
            arm=arm,
            updates=256,
            checkpoint_steps=CHECKPOINT_STEPS,
            wall_seconds=TRAIN_WALL_SECONDS,
            microbatch_size=selected["microbatch_size"],
            activation_checkpointing=selected["activation_checkpointing"],
            capture_gradient_step1=False,
        )
        arms[arm] = {
            "ce_reduction": ARM_REDUCTIONS[arm],
            "training_manifest": training.binding(path),
            "source_adapter": manifest["source_adapter"],
            "optimizer_mode": "fresh",
            "initial_step": 0,
        }
    implementation = [
        training.binding(Path(module))
        for module in (backend.__file__, __file__, training.__file__)
    ]
    readback_module = REPO / "probes/training_set_completion/coco227_readback.py"
    require(readback_module.is_file(), "readback module unavailable")
    implementation.append(training.binding(readback_module))
    trial = {
        "schema": SCHEMA,
        "status": "candidate_ready_release_pending",
        "arms": arms,
        "teacher": training.binding(teacher_path),
        "training": {
            "updates": 256,
            "global_image_count": 11,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "fresh_seed": 42,
            "qualification": training.binding(qualification_receipt),
            "selected_layout": selected,
        },
        "readback": {
            "steps": list(CHECKPOINT_STEPS),
            "request_count": 132,
            "qualification_result": training.binding(readback_qualification_result),
            "source0": training.binding(source0_admission),
            "phase_wall_seconds": READBACK_PHASE_SECONDS,
            "worker_module": "probes.training_set_completion.coco227_readback",
        },
        "implementation_bindings": implementation,
        "execution": {
            "training_gpu_groups": TRAINING_GPU_GROUPS,
            "readback_gpus": list(range(8)),
        },
    }
    trial["content_sha256"] = training.digest(trial)
    training.publish(output / "trial.json", trial)
    return trial


def validate_trial(
    value: Mapping[str, Any], *, verify_sources: bool = True
) -> dict[str, Any]:
    content = {key: item for key, item in value.items() if key != "content_sha256"}
    require(
        value.get("schema") == SCHEMA
        and value.get("status") == "candidate_ready_release_pending"
        and value.get("content_sha256") == training.digest(content),
        "trial schema/content",
    )
    require(set(value.get("arms", {})) == set(ARMS), "trial arms")
    require(
        value.get("training", {}).get("checkpoint_steps") == list(CHECKPOINT_STEPS)
        and value["training"].get("updates") == 256,
        "trial training dose",
    )
    require(
        value.get("readback", {}).get("request_count") == 132
        and value["readback"].get("steps") == list(CHECKPOINT_STEPS),
        "trial readback dose",
    )
    for arm in ARMS:
        if verify_sources:
            _verify(value["arms"][arm]["training_manifest"], f"{arm} manifest")
        manifest = backend.validate_manifest(
            read(value["arms"][arm]["training_manifest"]["path"]),
            verify_sources=verify_sources,
        )
        require(
            manifest["objective"]["ce_reduction"] == ARM_REDUCTIONS[arm],
            f"{arm} reduction",
        )
    if verify_sources:
        _verify(value["teacher"], "trial teacher")
        _verify(value["training"]["qualification"], "training qualification")
        _verify(
            value["readback"]["qualification_result"],
            "readback qualification result",
        )
        _verify(value["readback"]["source0"], "source0 readback")
        for item in value["implementation_bindings"]:
            _verify(item, "implementation")
    return dict(value)


def _spawn(command: list[str], *, visible_devices: str, log_path: Path) -> tuple[subprocess.Popen[Any], Any, float]:
    return spawn_logged_process(
        command, cwd=REPO, log_path=log_path,
        env={"CUDA_VISIBLE_DEVICES": visible_devices, "OMP_NUM_THREADS": "2",
             "TOKENIZERS_PARALLELISM": "false"},
    )


def _readback_command(
    *,
    trial: Mapping[str, Any],
    trial_path: Path,
    training_manifest: Path,
    training_terminal: Path,
    adapter: Path,
    arm: str,
    step: int,
    output: Path,
    gpu: int,
    qualification_result: Path,
    attempt: str,
) -> list[str]:
    return [
        "python",
        "-m",
        trial["readback"]["worker_module"],
        "endpoint-worker",
        "--training-manifest",
        str(training_manifest),
        "--training-terminal",
        str(training_terminal),
        "--adapter",
        str(adapter),
        "--arm",
        arm,
        "--step",
        str(step),
        "--output",
        str(output),
        "--gpu",
        str(gpu),
        "--batch-size",
        str(read(qualification_result)["selection"]["batch_size"]),
        "--qualification-result",
        str(qualification_result),
        "--trial",
        str(trial_path),
        "--attempt",
        attempt,
    ]


def _collect_endpoint_if_complete(
    *,
    readback_module: Any,
    trial: Mapping[str, Any],
    trial_path: Path,
    output: Path,
    arm: str,
    step: int,
    adapter: Path,
) -> dict[str, Any] | None:
    endpoint_root = output / "readback" / arm / f"step-{step:05d}"
    endpoint_path = endpoint_root / "endpoint.json"
    rows = list((endpoint_root / "rows").glob("image-*.json"))
    if not endpoint_path.is_file() and len(rows) < 11:
        return None
    require(endpoint_path.is_file() or len(rows) == 11, "partial readback endpoint")
    return readback_module.collect_endpoint(
        manifest_path=Path(trial["arms"][arm]["training_manifest"]["path"]),
        terminal_path=output / arm / "training/terminal.json",
        adapter_path=adapter,
        arm=arm,
        step=step,
        output=output / "readback",
        qualification_result=Path(trial["readback"]["qualification_result"]["path"]),
        trial_path=trial_path,
        teacher_bank_path=Path(trial["teacher"]["path"]),
    )


def collect_readbacks(
    *, trial_path: Path, output: Path, collections: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    trial = validate_trial(read(trial_path))
    require(len(collections) == 12, "twelve readback endpoints")
    endpoints = [row["endpoint"] for row in collections]
    admissions = [row["admission"] for row in collections]
    keys = {(row["arm"], int(row["checkpoint_step"])) for row in endpoints}
    require(
        keys == {(arm, step) for step in CHECKPOINT_STEPS for arm in ARMS},
        "readback endpoint cross product",
    )
    require(
        all(row.get("status") == "completed_unscored" for row in endpoints),
        "readback endpoint status",
    )
    require(
        {(row["arm"], int(row["step"])) for row in admissions} == keys
        and all(row.get("status") == "admitted_natural_readback" for row in admissions),
        "readback admission cross product/status",
    )
    result = {
        "schema": f"{SCHEMA}.readback_result.v1",
        "status": "completed_unscored",
        "trial": training.binding(trial_path),
        "source0": trial["readback"]["source0"],
        "request_count": 132,
        "endpoint_count": 12,
        "endpoints": [
            {
                "arm": endpoint["arm"],
                "step": endpoint["checkpoint_step"],
                "rows": training.binding(
                    output
                    / "readback"
                    / endpoint["arm"]
                    / f"step-{int(endpoint['checkpoint_step']):05d}"
                    / "endpoint.json"
                ),
                "admission": training.binding(
                    output
                    / "readback"
                    / endpoint["arm"]
                    / f"step-{int(endpoint['checkpoint_step']):05d}"
                    / "admission.json"
                ),
            }
            for endpoint in endpoints
        ],
        "rows": [
            {
                "arm": endpoint["arm"],
                "step": endpoint["checkpoint_step"],
                "image_id": row["image_id"],
                "row": endpoint["row_bindings"][index],
                "generated_token_ids_sha256": row["generated_token_ids_sha256"],
                "decode_stop_reason": row["decode_stop_reason"],
            }
            for endpoint in endpoints
            for index, row in enumerate(endpoint["rows"])
        ],
    }
    require(len(result["rows"]) == 132, "readback row count")
    result_path = output / "readback/result.json"
    if result_path.is_file():
        require(read(result_path) == result, "readback result changed on recovery")
    else:
        training.publish(result_path, result)
    return result


def controller(*, trial_path: Path, output: Path, release_path: Path) -> None:
    trial = validate_trial(read(trial_path))
    release = read(release_path)
    require(
        release.get("status") == "released"
        and release.get("trial_sha256") == training.file_hash(trial_path),
        "explicit release",
    )
    require(
        not (output / "controller-terminal.json").exists(),
        "controller already terminal",
    )
    tmux_session = subprocess.check_output(
        ["tmux", "display-message", "-p", "#S"], text=True
    ).strip()
    require(
        tmux_session == TRIAL_TMUX_SESSION,
        "trial controller must run in its named tmux session",
    )
    identities = output / "controller-identities"
    identities.mkdir(parents=True, exist_ok=True)
    attempt = len(list(identities.glob("attempt-*.json"))) + 1
    identity_path = identities / f"attempt-{attempt:03d}.json"
    identity = {
        "schema": f"{SCHEMA}.controller_identity.v1",
        "pid": os.getpid(),
        "attempt": attempt,
        "trial": training.binding(trial_path),
        "release": training.binding(release_path),
        "tmux_session": tmux_session,
        "started_at": time.time(),
    }
    training.publish(identity_path, identity)
    log_path = output / "logs" / f"controller-attempt-{attempt:03d}.log"
    log_path.parent.mkdir(exist_ok=True)
    log = log_path.open("x")

    def emit(text: str) -> None:
        log.write(text + "\n")
        log.flush()
        os.fsync(log.fileno())

    emit(f"COCO227_CONTROLLER pid={os.getpid()} attempt={attempt}")
    started = time.monotonic()
    exits = []
    live: list[tuple[Any, ...]] = []
    active: dict[int, dict[str, Any]] = {}
    try:
        for arm in ARMS:
            manifest_path = Path(trial["arms"][arm]["training_manifest"]["path"])
            train_output = output / arm / "training"
            if (train_output / "terminal.json").is_file():
                validate_training_terminal(train_output, manifest_path=manifest_path)
                exits.append(
                    {
                        "name": f"train-{arm}",
                        "status": "recovered_completed",
                        "exit_code": 0,
                    }
                )
                continue
            require(
                not train_output.exists(),
                f"{arm} incomplete training output requires explicit recovery",
            )
            seed_path = output / arm / "training-seed.json"
            training.publish(
                seed_path,
                {
                    "schema": f"{SCHEMA}.training_seed.v1",
                    "seed": 42,
                    "controller_pid": os.getpid(),
                    "controller_identity": training.binding(identity_path),
                    "manifest": training.binding(manifest_path),
                    "mode": "fresh_adamw",
                    "world_size": 4,
                    "gpu_group": TRAINING_GPU_GROUPS[arm],
                },
            )
            command = distributed_training_command(
                manifest_path=manifest_path, output=train_output
            )
            process, stream, spawned = _spawn(
                command,
                visible_devices=",".join(map(str, TRAINING_GPU_GROUPS[arm])),
                log_path=output / "logs" / f"train-{arm}.log",
            )
            live.append(
                (
                    process,
                    stream,
                    command,
                    f"train-{arm}",
                    TRAINING_GPU_GROUPS[arm],
                    spawned,
                )
            )
        if live:
            exits.extend(
                dual_start.wait_owned_processes(live, wall_seconds=TRAIN_WALL_SECONDS)
            )
        require(all(item["exit_code"] == 0 for item in exits), "training failure")
        for arm in ARMS:
            validate_training_terminal(
                output / arm / "training",
                manifest_path=Path(trial["arms"][arm]["training_manifest"]["path"]),
            )
        attempt_root = output / "controller-attempts" / f"attempt-{attempt:03d}"
        training.publish(
            attempt_root / "training-exits.json",
            {"schema": f"{SCHEMA}.training_exits.v1", "exits": list(exits)},
        )
        # The readback package owns endpoint-level missing-row recovery and collection.
        from probes.training_set_completion import coco227_readback

        jobs = [(arm, step) for step in CHECKPOINT_STEPS for arm in ARMS]
        pending = []
        collections = []
        for arm, step in jobs:
            terminal = read(output / arm / "training/terminal.json")
            entry = next(
                item for item in terminal["checkpoints"] if item["step"] == step
            )
            adapter = Path(entry["adapter"]["root"])
            collection = _collect_endpoint_if_complete(
                readback_module=coco227_readback,
                trial=trial,
                trial_path=trial_path,
                output=output,
                arm=arm,
                step=step,
                adapter=adapter,
            )
            if collection is not None:
                collections.append(collection)
                exits.append(
                    {
                        "name": f"readback-{arm}-{step}",
                        "status": "recovered_completed",
                        "exit_code": 0,
                    }
                )
                continue
            pending.append((arm, step, adapter))
        phase_deadline = time.monotonic() + READBACK_PHASE_SECONDS
        available = list(range(8))
        wave = 0
        completions: queue.Queue[dict[str, Any]] = queue.Queue()
        while pending or active:
            while pending and available:
                require(
                    time.monotonic() < phase_deadline,
                    "global readback phase timeout",
                )
                arm, step, adapter = pending.pop(0)
                gpu = available.pop(0)
                wave += 1
                attempt_id = f"controller-{attempt:03d}-job-{wave:03d}"
                command = _readback_command(
                    trial=trial,
                    trial_path=trial_path,
                    training_manifest=Path(
                        trial["arms"][arm]["training_manifest"]["path"]
                    ),
                    training_terminal=output / arm / "training/terminal.json",
                    adapter=adapter,
                    arm=arm,
                    step=step,
                    output=output / "readback",
                    gpu=gpu,
                    qualification_result=Path(
                        trial["readback"]["qualification_result"]["path"]
                    ),
                    attempt=attempt_id,
                )
                process, stream, spawned = _spawn(
                    command,
                    visible_devices=str(gpu),
                    log_path=output
                    / "logs"
                    / f"readback-{arm}-{step:05d}-{attempt_id}.log",
                )
                worker_deadline = min(
                    spawned + READBACK_PHASE_SECONDS, phase_deadline
                )
                active[process.pid] = {
                    "process": process,
                    "stream": stream,
                    "command": command,
                    "arm": arm,
                    "step": step,
                    "adapter": adapter,
                    "gpu": gpu,
                    "spawned": spawned,
                    "deadline": worker_deadline,
                }
                start_process_waiter(
                    process,
                    completions,
                    thread_name_prefix="coco227-readback-wait",
                )
            completion = next_process_completion(
                completions,
                deadline=phase_deadline,
                timeout_message="global readback phase timeout",
            )
            item = active.pop(int(completion["pid"]))
            item["stream"].close()
            exits.append(
                {
                    "name": f"readback-{item['arm']}-{item['step']}",
                    "pid": completion["pid"],
                    "gpu": item["gpu"],
                    "exit_code": completion["exit_code"],
                    "wait_error": completion["wait_error"],
                    "command": item["command"],
                    "spawned_at_monotonic": item["spawned"],
                    "completed_at_monotonic": completion["completed_at_monotonic"],
                    "deadline_monotonic": item["deadline"],
                }
            )
            available.append(int(item["gpu"]))
            available.sort()
            require(
                completion["exit_code"] == 0,
                f"readback {item['arm']}/{item['step']} failed",
            )
            collection = _collect_endpoint_if_complete(
                readback_module=coco227_readback,
                trial=trial,
                trial_path=trial_path,
                output=output,
                arm=item["arm"],
                step=item["step"],
                adapter=item["adapter"],
            )
            require(collection is not None, "worker left incomplete endpoint")
            collections.append(collection)
        result = collect_readbacks(
            trial_path=trial_path, output=output, collections=collections
        )
        require(result.get("request_count") == 132, "readback result count")
        training.publish(
            attempt_root / "readback-exits.json",
            {"schema": f"{SCHEMA}.readback_exits.v1", "exits": list(exits)},
        )
        terminal = {
            "schema": f"{SCHEMA}.controller_terminal.v1",
            "status": "completed_unscored",
            "pid": os.getpid(),
            "identity": training.binding(identity_path),
            "trial": identity["trial"],
            "release": identity["release"],
            "tmux_session": tmux_session,
            "exits": exits,
            "result": training.binding(output / "readback/result.json"),
            "elapsed_seconds": time.monotonic() - started,
        }
        emit("COCO227_COMPLETED_UNSCORED")
    except BaseException as exc:
        cleanup_errors: list[dict[str, Any]] = []

        def cleanup_owned(process: Any, stream: Any, name: str) -> None:
            try:
                if process.poll() is None:
                    terminate_owned_process(process)
            except BaseException as cleanup_exc:
                cleanup_errors.append(
                    {"name": name, "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}"}
                )
            finally:
                try:
                    stream.close()
                except BaseException as cleanup_exc:
                    cleanup_errors.append(
                        {"name": name, "error": f"stream close: {type(cleanup_exc).__name__}: {cleanup_exc}"}
                    )

        for item in live:
            cleanup_owned(item[0], item[1], str(item[3]))
        for item in active.values():
            cleanup_owned(
                item["process"],
                item["stream"],
                f"readback-{item['arm']}-{item['step']}",
            )
        failure = {
            "schema": f"{SCHEMA}.controller_failure.v1",
            "status": "failed",
            "pid": os.getpid(),
            "identity": training.binding(identity_path),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "exits": exits,
            "cleanup_errors": cleanup_errors,
            "elapsed_seconds": time.monotonic() - started,
        }
        failure_path = output / "controller-failures" / f"attempt-{attempt:03d}.json"
        training.publish(failure_path, failure)
        emit(f"COCO227_FAILED {failure['error']}")
        log.close()
        raise
    training.publish(output / "controller-terminal.json", terminal)
    log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-qualification")
    p.add_argument("--teacher", type=Path, default=TEACHER)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("qualification-controller")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare-trial")
    p.add_argument("--teacher", type=Path, default=TEACHER)
    p.add_argument("--qualification-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--readback-qualification-result", type=Path, required=True)
    p.add_argument("--source0-admission", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--trial", type=Path, required=True)
    p = sub.add_parser("controller")
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-qualification":
        prepare_qualification(teacher_path=args.teacher, output=args.output)
    elif args.command == "qualification-controller":
        qualification_controller(plan_path=args.plan, output=args.output)
    elif args.command == "prepare-trial":
        prepare_trial(
            teacher_path=args.teacher,
            qualification_receipt=args.qualification_receipt,
            output=args.output,
            readback_qualification_result=args.readback_qualification_result,
            source0_admission=args.source0_admission,
        )
    elif args.command == "verify":
        validate_trial(read(args.trial))
    else:
        controller(trial_path=args.trial, output=args.output, release_path=args.release)


if __name__ == "__main__":
    main()

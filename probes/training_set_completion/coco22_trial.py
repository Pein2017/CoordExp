"""Prepare and run the bounded COCO22 full-replay S arm or conditional Source control."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

from probes.training_set_completion import coco22_training as backend
from probes.training_set_completion import coco227_training as prior_backend
from probes.training_set_completion import coco22_readback as readback
from probes.training_set_completion import training

ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion")
REPO = Path(__file__).resolve().parents[2]
SCHEMA = "training_set_completion.coco22_cumulative_trial.v1"
PRIOR_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization")
PRIOR_MANIFEST = PRIOR_ROOT / "trial-v1/S/training-manifest.json"
PRIOR_BANK = PRIOR_ROOT / "data-v1/bank.json"
S_ADAPTER = PRIOR_ROOT / "trial-v1/S/training/checkpoints/step-00256/adapter"
SOURCE_ADAPTER = Path(
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444/adapter"
)
SOURCE_BINDING = {
    "S": (S_ADAPTER, backend.SOURCE_ADAPTER_FINGERPRINT),
    "Source": (SOURCE_ADAPTER, backend.CONTROL_ADAPTER_FINGERPRINT),
}
STEPS = (8, 16, 32, 64, 128, 256)
TRAINING_QUALIFICATION_TMUX = "coordexp-coco22-training-qualification"
TRIAL_TMUX = readback.TRIAL_TMUX
GPU_GROUP = list(range(8))
RANK_COUNTS = list(backend.RANK_COUNTS)
PREDECLARED = ROOT / "execution-preparation-v1/predeclared-qualification.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _verify(binding: Mapping[str, Any], label: str) -> None:
    require(training.binding(binding["path"]) == dict(binding), f"{label} source changed")


def _predeclared_tolerances() -> dict[str, Any]:
    value = read(PREDECLARED)
    require(
        value.get("schema") == "training_set_completion.coco22_execution_preparation.v1"
        and value.get("status") == "predeclared_before_coco22_model_measurement",
        "predeclared qualification receipt",
    )
    return dict(value["tolerances"])


def validate_teacher(value: Mapping[str, Any], *, prior: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze exact old routes; admit new obligations only from the bank producer."""
    require(value.get("status") == "candidate_ready", "teacher bank candidate")
    require(
        value.get("content_sha256")
        == training.digest({k: v for k, v in value.items() if k != "content_sha256"}),
        "teacher bank content hash",
    )
    routes = value.get("routes")
    require(isinstance(routes, list) and len(routes) == 22, "teacher has exactly 22 image routes")
    require(routes[:11] == prior["routes"], "old 11 teacher routes changed")
    require(
        [row["image_id"] for row in routes[:11]]
        == [row["image_id"] for row in prior["routes"]],
        "old image order changed",
    )
    require(sum(len(row["trusted_boxes"]) for row in routes[:11]) == 227
            and sum(sum(row["ce_weights"]) for row in routes[:11]) == 2176,
            "old 227-owner teacher exposure changed")
    require(sum(len(row["trusted_boxes"]) for row in routes[11:]) > 0,
            "new image teacher has no trusted owners")
    for route in routes:
        training.validate_route(
            route, eos_token_id=151645,
            coordinate_token_ids=prior["validity_hinge"]["coordinate_token_ids"],
        )
        ids = route["continuation_token_ids"]
        require(
            ids[-1] == 151645 and route["ce_weights"][-1] == 1
            and len(ids) <= 3084,
            "every teacher has supervised EOS within natural cap",
        )
    require(len({route["image_id"] for route in routes}) == 22
            and len({route["route_id"] for route in routes}) == 22,
            "one teacher route per distinct image")
    return dict(value)


def build_training_manifest(
    *, teacher_path: Path, output: Path, training_output: Path,
    arm: str, updates: int, microbatch_size: int,
    capture_gradient_step1: bool = False,
) -> dict[str, Any]:
    from probes.dora_owner_learning.route_access import checkpoint_config
    from src.config.inference import InferConfig

    require(arm in SOURCE_BINDING and updates in (2, 256), "training arm/dose")
    require(microbatch_size in (1, 2, 3), "training native microbatch")
    prior = prior_backend.validate_manifest(read(PRIOR_MANIFEST))
    teacher = validate_teacher(read(teacher_path), prior=prior)
    source_path, fingerprint = SOURCE_BINDING[arm]
    adapter = training.inspect_dora_adapter_payload(
        source_path, prior["model_config"]["model"]["base_model"]
    )
    require(adapter["fingerprint"] == fingerprint, f"{arm} frozen source adapter")
    model_config = copy.deepcopy(prior["model_config"])
    model_config["adapter"] = {"name": "default", "path": str(source_path), "type": "dora"}
    model_config["run"].update(
        name=f"coco22-{arm}", artifact_root=str(training_output),
        output_dir=None, collision_policy="fail",
    )
    checkpoint_config(InferConfig.model_validate(model_config), str(source_path))
    active = sum(sum(route["ce_weights"]) for route in teacher["routes"])
    calls_per_update = sum(math.ceil(n / microbatch_size) for n in RANK_COUNTS)
    value = {
        "schema": training.SCHEMA, "status": "candidate_ready",
        "sources": {
            "reviewed_routes": training.binding(teacher_path),
            "producer": training.binding(Path(backend.__file__)),
        },
        # The shared manifest field is a binding to the complete reviewed
        # 22-image route source, rather than the old-only acquisition packet.
        "acquisition_manifest": training.binding(teacher_path),
        "source_adapter": adapter, "model_config": model_config,
        "routes": copy.deepcopy(teacher["routes"]),
        "optimizer": copy.deepcopy(training.DEFAULT_OPTIMIZER),
        "runtime": {
            "updates": updates,
            "checkpoint_steps": [2] if updates == 2 else list(STEPS),
            "wall_seconds": None, "wall_budget_enforced": False,
            "max_model_forwards": updates * 22,
            "max_model_calls": updates * calls_per_update,
            "eos_token_id": 151645, "seed": 42, "initial_step": 0,
            "optimizer_mode": "fresh", "microbatch_size": microbatch_size,
            "activation_checkpointing": True,
            "global_ce_eligible_images": 22, "global_active_tokens": active,
            "capture_gradient_step1": capture_gradient_step1,
        },
        "validity_hinge": copy.deepcopy(prior["validity_hinge"]),
        "objective": {
            "ce_reduction": "sample_equal",
            "geometry_reduction": "global_equal_image_mean",
        },
        "teacher_partition": {
            "old227_owner_count": 227,
            "new_trusted_owner_count": sum(len(r["trusted_boxes"]) for r in teacher["routes"][11:]),
            "old_route_ids_sha256": training.digest([r["route_id"] for r in teacher["routes"][:11]]),
            "new_route_ids_sha256": training.digest([r["route_id"] for r in teacher["routes"][11:]]),
        },
        "content_sha256": None,
    }
    value["content_sha256"] = training.digest(
        {k: v for k, v in value.items() if k != "content_sha256"}
    )
    backend.validate_manifest(value)
    training.publish(output, value)
    return value


def distributed_training_command(*, manifest_path: Path, output: Path) -> list[str]:
    return [
        "python", "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=8", "-m",
        "probes.training_set_completion.coco22_training",
        "--manifest", str(manifest_path), "--output", str(output),
    ]


def validate_training_terminal(
    output: Path, *, manifest_path: Path, require_fresh: bool = True
) -> dict[str, Any]:
    manifest = backend.validate_manifest(read(manifest_path))
    terminal = read(output / "terminal.json")
    require(terminal.get("status") == "completed"
            and terminal.get("manifest") == training.binding(manifest_path),
            "completed training terminal identity")
    require(not require_fresh or terminal.get("optimizer_mode") == "fresh",
            "fresh AdamW start")
    require(terminal.get("updates") == manifest["runtime"]["updates"]
            and terminal.get("logical_model_forwards") == manifest["runtime"]["max_model_forwards"]
            and terminal.get("model_calls") == manifest["runtime"]["max_model_calls"],
            "full-cohort/model-call accounting")
    require(
        terminal.get("distributed", {}).get("world_size") == 8
        and terminal["distributed"].get("rank_image_counts") == RANK_COUNTS
        and terminal["distributed"].get("normalization", {}).get("gradient_collective") == "SUM"
        and terminal["distributed"]["normalization"].get("post_collective_divisor") == 1,
        "eight-rank global SUM receipt",
    )
    require(
        [item["step"] for item in terminal.get("checkpoints", [])]
        == manifest["runtime"]["checkpoint_steps"],
        "checkpoint schedule",
    )
    for checkpoint, consensus in zip(
        terminal["checkpoints"],
        terminal["distributed"]["checkpoint_consensus"],
        strict=True,
    ):
        state = torch.load(
            checkpoint["state"]["path"], map_location="cpu", weights_only=False
        )
        require(
            checkpoint["step"] == consensus["step"]
            and consensus["rank_count"] == 8
            and state["manifest"] == training.binding(manifest_path)
            and state["source_adapter"] == manifest["source_adapter"]
            and state["optimizer"] == manifest["optimizer"],
            "cold checkpoint source/manifest/rank identity",
        )
        observed_adapter = training.inspect_dora_adapter_payload(
            checkpoint["adapter"]["root"],
            manifest["model_config"]["model"]["base_model"],
        )
        require(
            observed_adapter == checkpoint["adapter"]
            == state["saved_adapter"],
            "cold checkpoint adapter payload identity",
        )
        optimizer_steps = {
            int(entry["step"].item())
            for entry in state["optimizer_state_dict"]["state"].values()
        }
        require(optimizer_steps == {checkpoint["step"]}
                and consensus["state"]["optimizer_steps"] == [checkpoint["step"]],
                "optimizer checkpoint counter",
        )
    if manifest["runtime"]["capture_gradient_step1"]:
        _verify(terminal["gradient_snapshot_step1"], "qualification gradient")
    require(
        len(terminal["distributed"]["rank_receipts"]) == 8
        and [r["local_image_count"] for r in terminal["distributed"]["rank_receipts"]]
        == RANK_COUNTS,
        "eight full-cohort rank receipts",
    )
    return terminal


def _tensor_difference(
    reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    require(set(reference) == set(candidate), "tensor comparison keys")
    diff_sq = reference_sq = max_abs = 0.0
    for key in sorted(reference):
        left, right = reference[key].double(), candidate[key].double()
        require(
            left.shape == right.shape and bool(torch.isfinite(left).all())
            and bool(torch.isfinite(right).all()), "finite matching tensors",
        )
        delta = left - right
        diff_sq += float(torch.sum(delta * delta))
        reference_sq += float(torch.sum(left * left))
        max_abs = max(max_abs, float(torch.max(torch.abs(delta))))
    return {"max_abs": max_abs,
            "relative_l2": math.sqrt(diff_sq / max(reference_sq, 1e-300))}


def compare_qualification_runs(
    reference_output: Path, candidate_output: Path
) -> dict[str, Any]:
    from safetensors.torch import load_file

    tolerances = _predeclared_tolerances()
    left_terminal = read(reference_output / "terminal.json")
    right_terminal = read(candidate_output / "terminal.json")
    left_grad = torch.load(
        left_terminal["gradient_snapshot_step1"]["path"],
        map_location="cpu", weights_only=True,
    )
    right_grad = torch.load(
        right_terminal["gradient_snapshot_step1"]["path"],
        map_location="cpu", weights_only=True,
    )
    gradients = _tensor_difference(left_grad, right_grad)
    left_adapter = Path(left_terminal["checkpoints"][-1]["adapter"]["root"])
    right_adapter = Path(right_terminal["checkpoints"][-1]["adapter"]["root"])
    parameters = _tensor_difference(
        load_file(str(left_adapter / "adapter_model.safetensors"), device="cpu"),
        load_file(str(right_adapter / "adapter_model.safetensors"), device="cpu"),
    )
    differences: dict[str, float] = {
        "normalized": 0.0, "masked_nll_ratio": 0.0,
        "gradient_norm_before_clip": 0.0,
    }
    nll_pass = True
    for step in (1, 2):
        left = read(reference_output / "updates" / f"step-{step:05d}.json")
        right = read(candidate_output / "updates" / f"step-{step:05d}.json")
        require(
            [r["route_id"] for r in left["routes"]]
            == [r["route_id"] for r in right["routes"]]
            and [r["active_tokens"] for r in left["routes"]]
            == [r["active_tokens"] for r in right["routes"]],
            "exact teacher route and active-token accounting",
        )
        for key in ("global_ce", "global_geometry_mean", "objective_total"):
            differences["normalized"] = max(
                differences["normalized"], abs(float(left[key]) - float(right[key]))
            )
        differences["gradient_norm_before_clip"] = max(
            differences["gradient_norm_before_clip"],
            abs(float(left["gradient_norm_before_clip"]) -
                float(right["gradient_norm_before_clip"])),
        )
        for lrow, rrow in zip(left["routes"], right["routes"], strict=True):
            for key in ("active_token_mean_ce", "raw_axis_validity_hinge"):
                differences["normalized"] = max(
                    differences["normalized"],
                    abs(float(lrow[key]) - float(rrow[key])),
                )
            nll_diff = abs(float(lrow["masked_nll_sum"]) - float(rrow["masked_nll_sum"]))
            nll_bound = max(0.002, 1e-5 * abs(float(lrow["masked_nll_sum"])))
            differences["masked_nll_ratio"] = max(
                differences["masked_nll_ratio"], nll_diff / nll_bound,
            )
            nll_pass &= nll_diff <= nll_bound
    passed = (
        differences["normalized"] <= 5e-5
        and nll_pass
        and differences["gradient_norm_before_clip"] <= 5e-4
        and gradients["max_abs"] <= 5e-4
        and gradients["relative_l2"] <= 1e-4
        and parameters["max_abs"] <= 5e-5
        and parameters["relative_l2"] <= 1e-5
    )
    require(
        tolerances["normalized_route_active_token_mean_ce_max_abs"] == 5e-5
        and tolerances["masked_nll_sum_max_abs"] == "max(0.002, 1e-5 * abs(reference_sum))"
        and tolerances["gradient_relative_l2"] == 1e-4,
        "frozen quantity-aware tolerances",
    )
    return {
        "passed": passed, "differences": differences,
        "gradient": gradients, "step2_parameters": parameters,
        "tolerances": tolerances,
    }


def _rank_throughput(output: Path) -> dict[str, Any]:
    updates = [
        read(output / "updates" / f"step-{step:05d}.json") for step in (1, 2)
    ]
    maxima = [
        max(row["forward_backward_seconds"]
            for row in update["distributed"]["rank_timings"])
        for update in updates
    ]
    terminal = read(output / "terminal.json")
    return {
        "max_rank_forward_backward_seconds": maxima,
        "mean_max_rank_forward_backward_seconds": sum(maxima) / 2,
        "rank_prompt_padding_tokens": [
            row["native_preparation"]["prompt_padding_tokens"]
            for row in terminal["distributed"]["rank_receipts"]
        ],
        "rank_microbatch_image_counts": [
            row["native_preparation"]["microbatch_image_counts"]
            for row in terminal["distributed"]["rank_receipts"]
        ],
        "rank_peak_cuda_allocated_bytes": [
            row["resources"]["peak_cuda_allocated_bytes"]
            for row in terminal["distributed"]["rank_receipts"]
        ],
    }


def prepare_qualification(
    *, teacher_path: Path, output: Path, arm: str = "S"
) -> dict[str, Any]:
    require(arm in SOURCE_BINDING, "qualification arm")
    require(not output.exists(), "qualification output collision")
    require(PREDECLARED.is_file(), "predeclared tolerances missing")
    output.mkdir(parents=True)
    configs = []
    for size in (1, 2, 3):
        manifest_path = output / "manifests" / f"mb{size}.json"
        run_output = output / "runs" / f"mb{size}"
        build_training_manifest(
            teacher_path=teacher_path, output=manifest_path,
            training_output=run_output, arm=arm, updates=2,
            microbatch_size=size, capture_gradient_step1=True,
        )
        configs.append({
            "id": f"mb{size}", "microbatch_size": size,
            "manifest": training.binding(manifest_path),
            "output": str(run_output),
            "command": distributed_training_command(
                manifest_path=manifest_path, output=run_output
            ),
        })
    plan = {
        "schema": f"{SCHEMA}.qualification_plan",
        "status": "candidate_ready", "arm": arm,
        "teacher": training.binding(teacher_path),
        "producer": training.binding(Path(backend.__file__)),
        "controller": training.binding(Path(__file__)),
        "predeclared": training.binding(PREDECLARED),
        "configurations": configs,
        "bounds": {
            "updates_per_config": 2, "logical_image_exposures": 132,
            "maximum_model_calls": 88, "physical_gpus": GPU_GROUP,
            "wall_kill": None,
        },
    }
    plan["content_sha256"] = training.digest(plan)
    training.publish(output / "plan.json", plan)
    return plan


def validate_qualification_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    require(
        value.get("schema") == f"{SCHEMA}.qualification_plan"
        and value.get("status") == "candidate_ready"
        and value.get("content_sha256")
        == training.digest({k: v for k, v in value.items() if k != "content_sha256"}),
        "qualification plan integrity",
    )
    require(
        value.get("arm") in SOURCE_BINDING,
        "qualification matched source arm",
    )
    require(
        value.get("bounds") == {
            "updates_per_config": 2, "logical_image_exposures": 132,
            "maximum_model_calls": 88, "physical_gpus": GPU_GROUP,
            "wall_kill": None,
        }, "bounded eight-rank qualification",
    )
    _verify(value["teacher"], "teacher")
    _verify(value["predeclared"], "predeclared qualification")
    _verify(value["producer"], "training backend")
    _verify(value["controller"], "trial controller")
    configs = value["configurations"]
    require([row["microbatch_size"] for row in configs] == [1, 2, 3],
            "serial/mb2/mb3 qualification order")
    for row in configs:
        _verify(row["manifest"], f"{row['id']} manifest")
        backend.validate_manifest(read(row["manifest"]["path"]))
        require(
            read(row["manifest"]["path"])["source_adapter"]["fingerprint"]
            == SOURCE_BINDING[value["arm"]][1],
            "qualification source adapter identity",
        )
        require(
            row["command"] == distributed_training_command(
                manifest_path=Path(row["manifest"]["path"]),
                output=Path(row["output"]),
            ), "qualification command",
        )
    return dict(value)


def _require_named_tmux(session: str) -> None:
    observed = subprocess.check_output(
        ["tmux", "display-message", "-p", "#S"], text=True
    ).strip()
    require(observed == session, f"controller must run in named tmux {session}")


def qualification_controller(*, plan_path: Path, output: Path) -> dict[str, Any]:
    _require_named_tmux(TRAINING_QUALIFICATION_TMUX)
    require(not (output / "terminal.json").exists(), "qualification already terminal")
    plan = validate_qualification_plan(read(plan_path))
    started = time.monotonic()
    exits = []
    training.publish(output / "launch.json", {
        "schema": f"{SCHEMA}.qualification_launch",
        "status": "running", "pid": os.getpid(),
        "tmux_session": TRAINING_QUALIFICATION_TMUX,
        "plan": training.binding(plan_path),
    })
    try:
        rows = []
        for config in plan["configurations"]:
            log = output / "logs" / f"{config['id']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("x")
            process = subprocess.Popen(
                config["command"], cwd=REPO, stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                     "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"},
                start_new_session=True,
            )
            code = process.wait()
            stream.close()
            exits.append({
                "id": config["id"], "pid": process.pid, "exit_code": code,
                "command": config["command"], "log": training.binding(log),
            })
            require(code == 0, f"qualification {config['id']} failed")
            validate_training_terminal(
                Path(config["output"]),
                manifest_path=Path(config["manifest"]["path"]),
            )
            parity = (
                {"passed": True, "reference": "self"}
                if config["microbatch_size"] == 1
                else compare_qualification_runs(
                    Path(plan["configurations"][0]["output"]),
                    Path(config["output"]),
                )
            )
            rows.append({
                "id": config["id"], "microbatch_size": config["microbatch_size"],
                "manifest": config["manifest"],
                "terminal": training.binding(Path(config["output"]) / "terminal.json"),
                "throughput": _rank_throughput(Path(config["output"])),
                "parity_to_mb1": parity,
            })
        eligible = [row for row in rows if row["parity_to_mb1"]["passed"]]
        selected = min(
            eligible,
            key=lambda row: row["throughput"]["mean_max_rank_forward_backward_seconds"],
        )
        serial = rows[0]
        if selected["throughput"]["mean_max_rank_forward_backward_seconds"] >= serial["throughput"]["mean_max_rank_forward_backward_seconds"]:
            selected = serial
        receipt = {
            "schema": f"{SCHEMA}.qualification_receipt",
            "status": "passed", "arm": plan["arm"],
            "plan": training.binding(plan_path),
            "predeclared": plan["predeclared"],
            "rows": rows, "exits": exits,
            "selected": {
                "microbatch_size": selected["microbatch_size"],
                "activation_checkpointing": True,
                "measured_max_rank_seconds": selected["throughput"]["mean_max_rank_forward_backward_seconds"],
                "serial_baseline_seconds": serial["throughput"]["mean_max_rank_forward_backward_seconds"],
            },
            "observed_logical_exposures": sum(
                read(Path(c["output"]) / "terminal.json")["logical_model_forwards"]
                for c in plan["configurations"]
            ),
        }
        require(receipt["observed_logical_exposures"] == 132,
                "qualification full-cohort exposure count")
        training.publish(output / "receipt.json", receipt)
        terminal = {
            "schema": f"{SCHEMA}.qualification_terminal",
            "status": "completed", "pid": os.getpid(),
            "tmux_session": TRAINING_QUALIFICATION_TMUX,
            "receipt": training.binding(output / "receipt.json"),
            "elapsed_seconds": time.monotonic() - started,
        }
    except BaseException as exc:
        terminal = {
            "schema": f"{SCHEMA}.qualification_terminal",
            "status": "failed", "pid": os.getpid(),
            "tmux_session": TRAINING_QUALIFICATION_TMUX,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), "exits": exits,
            "elapsed_seconds": time.monotonic() - started,
        }
        raise
    finally:
        training.publish(output / "terminal.json", terminal)
    return terminal


def prepare_trial(
    *, teacher_path: Path, qualification_receipt: Path,
    readback_qualification_result: Path, source0_admission: Path,
    arm: str, output: Path,
) -> dict[str, Any]:
    require(arm in SOURCE_BINDING and not output.exists(), "trial arm/output")
    qualification = read(qualification_receipt)
    require(
        qualification.get("schema") == f"{SCHEMA}.qualification_receipt"
        and qualification.get("status") == "passed"
        and qualification.get("arm") == arm
        and qualification.get("selected", {}).get("microbatch_size") in (1, 2, 3),
        "admitted 22-image training qualification",
    )
    _verify(qualification["plan"], "training qualification plan")
    selected = qualification["selected"]
    rbq = read(readback_qualification_result)
    require(
        rbq.get("schema") == f"{readback.SCHEMA}.qualification_result"
        and rbq.get("status") == "completed"
        and rbq.get("arm") == arm
        and rbq.get("selected", {}).get("batch_size") in (1, 2, 3)
        and rbq.get("requests") == 66, "admitted 22-image native readback batch",
    )
    source0 = read(source0_admission)
    require(
        source0.get("schema") == "training_set_completion.coco22_readback.v1.admission"
        and source0.get("status") == "admitted_natural_readback"
        and source0.get("arm") == arm
        and source0.get("step") == 0
        and source0.get("teacher_bank") == training.binding(teacher_path),
        "cold 22-image source0 readback admission",
    )
    output.mkdir(parents=True)
    path = output / arm / "training-manifest.json"
    manifest = build_training_manifest(
        teacher_path=teacher_path, output=path,
        training_output=output / arm / "training",
        arm=arm, updates=256,
        microbatch_size=selected["microbatch_size"],
    )
    trial = {
        "schema": SCHEMA, "status": "candidate_ready_release_pending",
        "arm": arm, "teacher": training.binding(teacher_path),
        "training_manifest": training.binding(path),
        "source_adapter": manifest["source_adapter"],
        "optimizer_mode": "fresh",
        "training": {
            "updates": 256, "global_image_count": 22,
            "rank_image_counts": RANK_COUNTS, "checkpoint_steps": list(STEPS),
            "logical_exposures": 5632, "physical_gpus": GPU_GROUP,
            "wall_kill": None,
            "qualification": training.binding(qualification_receipt),
            "selected_layout": selected,
        },
        "readback": {
            "steps": list(STEPS), "primary_requests_including_step0": 154,
            "qualification_result": training.binding(readback_qualification_result),
            "source0": training.binding(source0_admission),
            "worker_module": "probes.training_set_completion.coco22_readback",
            "batch_size": rbq["selected"]["batch_size"],
        },
        "implementation_bindings": [
            training.binding(Path(backend.__file__)),
            training.binding(Path(__file__)),
            training.binding(Path(readback.__file__)),
            training.binding(Path(training.__file__)),
        ],
        "content_sha256": None,
    }
    trial["content_sha256"] = training.digest(
        {k: v for k, v in trial.items() if k != "content_sha256"}
    )
    training.publish(output / "trial.json", trial)
    return trial


def validate_trial(value: Mapping[str, Any]) -> dict[str, Any]:
    require(
        value.get("schema") == SCHEMA
        and value.get("status") == "candidate_ready_release_pending"
        and value.get("content_sha256")
        == training.digest({k: v for k, v in value.items() if k != "content_sha256"}),
        "trial identity",
    )
    arm = value.get("arm")
    require(arm in SOURCE_BINDING, "trial arm")
    for name in ("teacher", "training_manifest"):
        _verify(value[name], f"trial {name}")
    for item in value["implementation_bindings"]:
        _verify(item, "trial implementation")
    manifest = backend.validate_manifest(read(value["training_manifest"]["path"]))
    require(manifest["source_adapter"] == value["source_adapter"]
            and manifest["runtime"]["updates"] == 256
            and manifest["runtime"]["wall_seconds"] is None
            and value["training"]["logical_exposures"] == 5632,
            "trial fixed dose/source/no-wall")
    _verify(value["training"]["qualification"], "training qualification")
    _verify(value["readback"]["qualification_result"], "readback qualification")
    _verify(value["readback"]["source0"], "cold source0 admission")
    return dict(value)


def _live_matching_process(output: Path) -> list[int]:
    """Reject a duplicate controller while any exact trial producer is live."""
    needles = (
        "probes.training_set_completion.coco22_training",
        "probes.training_set_completion.coco22_readback",
    )
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = [x.decode(errors="replace") for x in path.read_bytes().split(b"\0") if x]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(needle in args for needle in needles) and any(
            arg == str(output) or arg.startswith(str(output) + "/") for arg in args
        ):
            found.append(int(path.parent.name))
    return found


def _readback_command(
    *, trial_path: Path, trial: Mapping[str, Any], terminal_path: Path,
    adapter_path: Path, step: int, gpu: int, output: Path, attempt: str,
) -> list[str]:
    return [
        "python", "-m", trial["readback"]["worker_module"], "worker",
        "--training-manifest", trial["training_manifest"]["path"],
        "--training-terminal", str(terminal_path),
        "--adapter", str(adapter_path), "--arm", trial["arm"],
        "--step", str(step), "--output",
        str(output / "readback" / trial["arm"] / f"step-{step:05d}"),
        "--gpu", str(gpu), "--batch-size", str(trial["readback"]["batch_size"]),
        "--attempt", attempt, "--source-kind", "scientific_checkpoint_readback",
        "--trial", str(trial_path),
    ]


def collect_readbacks(*, trial_path: Path, output: Path) -> dict[str, Any]:
    trial = validate_trial(read(trial_path))
    arm = trial["arm"]
    rows = []
    endpoints = []
    for step in STEPS:
        endpoint_path = output / "readback" / arm / f"step-{step:05d}" / "endpoint.json"
        admission_path = endpoint_path.parent / "admission.json"
        endpoint, admission = read(endpoint_path), read(admission_path)
        require(
            endpoint["status"] == "completed_unscored"
            and endpoint["arm"] == arm and endpoint["step"] == step
            and admission["status"] == "admitted_natural_readback"
            and admission["rows"] == training.binding(endpoint_path),
            "saved endpoint/admission",
        )
        require(len(endpoint["row_bindings"]) == 22, "22 immutable endpoint rows")
        endpoints.append({
            "step": step, "endpoint": training.binding(endpoint_path),
            "admission": training.binding(admission_path),
        })
        for image_row, binding in zip(
            endpoint["rows"], endpoint["row_bindings"], strict=True,
        ):
            require(training.binding(binding["path"]) == binding,
                    "readback row binding")
            rows.append({
                "arm": arm, "step": step, "image_id": image_row["image_id"],
                "row": binding, "generated_token_ids_sha256":
                    image_row["generated_token_ids_sha256"],
                "decode_stop_reason": image_row["decode_stop_reason"],
            })
    require(len(rows) == 132, "132 saved-checkpoint readback rows")
    value = {
        "schema": f"{SCHEMA}.readback_result", "status": "completed_unscored",
        "trial": training.binding(trial_path), "source0": trial["readback"]["source0"],
        "endpoint_count": 6, "saved_request_count": 132,
        "primary_request_count_including_source0": 154,
        "endpoints": endpoints, "rows": rows,
    }
    path = output / "readback" / "result.json"
    if path.is_file():
        require(read(path) == value, "readback collection changed")
    else:
        training.publish(path, value)
    return value


def controller(*, trial_path: Path, output: Path, release_path: Path) -> None:
    trial = validate_trial(read(trial_path))
    release = read(release_path)
    require(
        release.get("status") == "released"
        and release.get("trial_sha256") == training.file_hash(trial_path),
        "explicit lead release receipt",
    )
    _require_named_tmux(TRIAL_TMUX)
    require(not (output / "controller-terminal.json").exists(),
            "trial already completed")
    require(not _live_matching_process(output), "matching COCO22 producer already live")
    attempts = output / "controller-identities"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = len(list(attempts.glob("attempt-*.json"))) + 1
    identity_path = attempts / f"attempt-{attempt:03d}.json"
    training.publish(identity_path, {
        "schema": f"{SCHEMA}.controller_identity",
        "pid": os.getpid(), "attempt": attempt,
        "trial": training.binding(trial_path),
        "release": training.binding(release_path),
        "tmux_session": TRIAL_TMUX, "started_at": time.time(),
    })
    arm = trial["arm"]
    manifest_path = Path(trial["training_manifest"]["path"])
    training_output = output / arm / "training"
    started = time.monotonic()
    exits = []
    active: dict[int, tuple[Any, ...]] = {}
    terminal: dict[str, Any] = {
        "schema": f"{SCHEMA}.controller_terminal",
        "status": "running", "arm": arm, "pid": os.getpid(),
        "identity": training.binding(identity_path),
    }
    try:
        if (training_output / "terminal.json").is_file():
            validate_training_terminal(training_output, manifest_path=manifest_path)
            exits.append({"name": "train", "status": "recovered_completed", "exit_code": 0})
        else:
            require(not training_output.exists(),
                    "incomplete training output requires explicit checkpoint recovery")
            log_path = output / "logs" / f"train-{arm}-attempt-{attempt:03d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("x")
            command = distributed_training_command(
                manifest_path=manifest_path, output=training_output
            )
            process = subprocess.Popen(
                command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                     "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"},
                start_new_session=True,
            )
            training.publish(output / arm / "training-launch.json", {
                "schema": f"{SCHEMA}.training_launch", "status": "spawned",
                "controller_identity": training.binding(identity_path),
                "pid": process.pid, "command": command,
                "world_size": 8, "gpu_group": GPU_GROUP,
                "fresh_optimizer": True, "wall_kill": None,
            })
            code = process.wait()
            stream.close()
            exits.append({
                "name": "train", "pid": process.pid, "exit_code": code,
                "log": training.binding(log_path), "command": command,
            })
            require(code == 0, "COCO22 training failed")
            validate_training_terminal(training_output, manifest_path=manifest_path)
        train_terminal = read(training_output / "terminal.json")
        jobs = []
        for step in STEPS:
            checkpoint = next(
                row for row in train_terminal["checkpoints"] if row["step"] == step
            )
            adapter = Path(checkpoint["adapter"]["root"])
            endpoint_root = output / "readback" / arm / f"step-{step:05d}"
            endpoint_path = endpoint_root / "endpoint.json"
            if endpoint_path.is_file():
                readback.collect_endpoint(
                    manifest_path=manifest_path,
                    terminal_path=training_output / "terminal.json",
                    adapter_path=adapter, arm=arm, step=step,
                    output=output / "readback",
                    qualification_result=Path(
                        trial["readback"]["qualification_result"]["path"]
                    ),
                    teacher_bank_path=Path(trial["teacher"]["path"]),
                    trial_path=trial_path,
                )
                exits.append({
                    "name": f"readback-{step}", "status": "recovered_completed",
                    "exit_code": 0,
                })
            else:
                jobs.append((step, adapter))
        events: queue.Queue[dict[str, Any]] = queue.Queue()
        available = list(GPU_GROUP)
        failure = False
        job_serial = 0
        while jobs or active:
            if failure and not active:
                break
            while jobs and available and not failure:
                step, adapter = jobs.pop(0)
                gpu = available.pop(0)
                job_serial += 1
                name = f"readback-{step}"
                attempt_id = f"controller-{attempt:03d}-job-{job_serial:03d}"
                command = _readback_command(
                    trial_path=trial_path, trial=trial,
                    terminal_path=training_output / "terminal.json",
                    adapter_path=adapter, step=step, gpu=gpu,
                    output=output, attempt=attempt_id,
                )
                log = output / "logs" / f"{name}-{attempt_id}.log"
                stream = log.open("x")
                process = subprocess.Popen(
                    command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                         "OMP_NUM_THREADS": "2",
                         "TOKENIZERS_PARALLELISM": "false"},
                    start_new_session=True,
                )
                active[process.pid] = (
                    process, stream, name, step, adapter, gpu, command, log
                )
                readback.start_waiter(process, events)
            event = events.get()
            process, stream, name, step, adapter, gpu, command, log = active.pop(event["pid"])
            stream.close()
            available.append(gpu)
            available.sort()
            exits.append({
                "name": name, "pid": process.pid, "exit_code": event["exit_code"],
                "wait_error": event["wait_error"], "command": command,
                "log": training.binding(log),
            })
            if event["exit_code"] == 0:
                readback.collect_endpoint(
                    manifest_path=manifest_path,
                    terminal_path=training_output / "terminal.json",
                    adapter_path=adapter, arm=arm, step=step,
                    output=output / "readback",
                    qualification_result=Path(
                        trial["readback"]["qualification_result"]["path"]
                    ),
                    teacher_bank_path=Path(trial["teacher"]["path"]),
                    trial_path=trial_path,
                )
            else:
                failure = True
        require(not failure and not jobs, "saved readback worker failed")
        result = collect_readbacks(trial_path=trial_path, output=output)
        terminal.update(
            status="completed_unscored", exits=exits,
            result=training.binding(output / "readback" / "result.json"),
        )
        require(result == read(terminal["result"]["path"]),
                "readback collection publication")
    except BaseException as exc:
        # Keep ownership of spawned workers through controller failure; their
        # durable rows and attempt terminals remain available for recovery.
        for process, stream, *_ in active.values():
            process.wait()
            stream.close()
        terminal.update(
            status="failed", exits=exits,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        terminal["elapsed_seconds"] = time.monotonic() - started
        training.publish(
            output / "controller-attempts" / f"attempt-{attempt:03d}" / "terminal.json",
            terminal,
        )
        if terminal["status"] == "completed_unscored":
            training.publish(output / "controller-terminal.json", terminal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-manifest")
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-output", type=Path, required=True)
    p.add_argument("--arm", choices=tuple(SOURCE_BINDING), default="S")
    p.add_argument("--updates", type=int, choices=(2, 256), default=2)
    p.add_argument("--microbatch-size", type=int, choices=(1, 2, 3), default=1)
    p = sub.add_parser("prepare-qualification")
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--arm", choices=tuple(SOURCE_BINDING), default="S")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("qualification-controller")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare-trial")
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--qualification-receipt", type=Path, required=True)
    p.add_argument("--readback-qualification-result", type=Path, required=True)
    p.add_argument("--source0-admission", type=Path, required=True)
    p.add_argument("--arm", choices=tuple(SOURCE_BINDING), default="S")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--trial", type=Path, required=True)
    p = sub.add_parser("controller")
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-manifest":
        build_training_manifest(
            teacher_path=args.teacher, output=args.output,
            training_output=args.training_output,
            arm=args.arm, updates=args.updates,
            microbatch_size=args.microbatch_size,
        )
    elif args.command == "prepare-qualification":
        prepare_qualification(teacher_path=args.teacher, output=args.output,
                              arm=args.arm)
    elif args.command == "qualification-controller":
        qualification_controller(plan_path=args.plan, output=args.output)
    elif args.command == "prepare-trial":
        prepare_trial(
            teacher_path=args.teacher,
            qualification_receipt=args.qualification_receipt,
            readback_qualification_result=args.readback_qualification_result,
            source0_admission=args.source0_admission,
            arm=args.arm, output=args.output,
        )
    elif args.command == "verify":
        validate_trial(read(args.trial))
    else:
        controller(trial_path=args.trial, output=args.output,
                   release_path=args.release)


if __name__ == "__main__":
    main()

"""COCO22 full-replay sample-mean trainer; all eight ranks sum gradients once/update.

The prior COCO227 replay and checkpoint helpers are reused without changing their
hash-bound producer. No wall-clock signal or experiment kill cap is installed.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import random
import socket
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
import torch.distributed as dist
from probes.training_set_completion import coco227_training as previous
from probes.training_set_completion import dual_start_distributed as distributed
from probes.training_set_completion import training

SCHEMA = "training_set_completion.coco22_training.v1"
GLOBAL_IMAGE_COUNT = 22
REQUIRED_WORLD_SIZE = 8
SOURCE_ADAPTER_FINGERPRINT = "0a2d96f2f4a8c89c8e3db5cc743ebf4e4a22e7656d5a70437e68c66b59441a87"
CONTROL_ADAPTER_FINGERPRINT = "b8ca2461c93bf32e886c9e42f7ab0e52495ef2d439a605207c86af7258408815"
COLLECTIVE_TIMEOUT_SECONDS = 600
RANK_COUNTS = (3, 3, 3, 3, 3, 3, 2, 2)
OLD_BANK = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-15-coco227-ce-normalization/data-v1/bank.json"
)
OLD_BANK_SHA256 = "d65126ec827a9cd070b6a06c4557f44e2bb6c5a37b3fb3fc1b1fb6e94d172aac"
_batched_aligned_logits = previous._batched_aligned_logits
_route_terms = previous._route_terms
_prepare_microbatches = previous._prepare_microbatches

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def dependency_bindings() -> dict[str, dict[str, Any]]:
    return previous.dependency_bindings()

def partition_route_indices(total: int, *, rank: int, world_size: int) -> list[int]:
    require(total == GLOBAL_IMAGE_COUNT and world_size == REQUIRED_WORLD_SIZE and 0 <= rank < world_size,
            "COCO22 requires 22 routes across eight ranks")
    quotient, remainder = divmod(total, world_size)
    start = rank * quotient + min(rank, remainder)
    return list(range(start, start + quotient + int(rank < remainder)))

def objective_from_route_terms(
    ce_means: Sequence[torch.Tensor], active_tokens: Sequence[int], raw_hinges: Sequence[torch.Tensor],
    *, ce_reduction: str, global_ce_eligible_images: int, global_active_tokens: int,
    geometry_weight: float = 0.01,
) -> torch.Tensor:
    require(ce_reduction == "sample_equal" and global_ce_eligible_images == GLOBAL_IMAGE_COUNT,
            "COCO22 sample-equal objective")
    return previous.objective_from_route_terms(
        ce_means, active_tokens, raw_hinges, ce_reduction=ce_reduction,
        global_ce_eligible_images=GLOBAL_IMAGE_COUNT, global_active_tokens=global_active_tokens,
        global_image_count=GLOBAL_IMAGE_COUNT, geometry_weight=geometry_weight,
    )

def validate_manifest(value: Mapping[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    """Reuse strict shared admission with an inert legacy wall-schema projection.

    The original manifest remains hash-bound with wall_seconds=null. A detached
    projection is passed only through the shared schema validator; the trainer
    and controller never install or enforce that projected value.
    """
    require(value.get("schema") == training.SCHEMA and value.get("status") == "candidate_ready",
            "COCO22 training manifest schema/status")
    require(value.get("content_sha256") == training.digest({k: v for k, v in value.items() if k != "content_sha256"}),
            "COCO22 training manifest digest")
    runtime = value.get("runtime", {})
    require(runtime.get("wall_seconds") is None and runtime.get("wall_budget_enforced") is False,
            "COCO22 cannot inherit a training wall kill")
    projected = copy.deepcopy(dict(value))
    projected["runtime"]["wall_seconds"] = 7_200
    projected["content_sha256"] = training.digest({k: v for k, v in projected.items() if k != "content_sha256"})
    training.validate_manifest(projected, verify_sources=verify_sources)
    require(
        Path(value["sources"]["producer"]["path"]).resolve()
        == Path(__file__).resolve(),
        "COCO22 manifest producer",
    )
    require(
        value["acquisition_manifest"] == value["sources"]["reviewed_routes"],
        "COCO22 complete cohort acquisition binding",
    )
    require(
        training.file_hash(OLD_BANK) == OLD_BANK_SHA256,
        "prior 227-owner teacher file changed",
    )
    old_routes = json.loads(OLD_BANK.read_text())["routes"]
    require(value["routes"][:11] == old_routes, "old teacher routes changed")
    if verify_sources:
        reviewed = json.loads(
            Path(value["sources"]["reviewed_routes"]["path"]).read_text()
        )
        require(
            value["routes"] == reviewed.get("routes"),
            "manifest routes differ from bound reviewed teacher",
        )
    require(value["source_adapter"]["fingerprint"] in (
        SOURCE_ADAPTER_FINGERPRINT,
        CONTROL_ADAPTER_FINGERPRINT,
    ), "COCO22 source adapter identity")
    require(len(value["routes"]) == GLOBAL_IMAGE_COUNT and
            sum(len(route["trusted_boxes"]) for route in value["routes"]) > 227,
            "COCO22 frozen teacher denominator")
    require(value.get("objective") == {"ce_reduction": "sample_equal", "geometry_reduction": "global_equal_image_mean"},
            "COCO22 objective")
    active = [sum(route["ce_weights"]) for route in value["routes"]]
    require(all(count > 0 for count in active), "every image has active CE targets")
    require(runtime.get("seed") == 42 and runtime.get("initial_step") == 0 and runtime.get("optimizer_mode") == "fresh",
            "fresh optimizer seed42")
    require(runtime.get("microbatch_size") in (1, 2, 3) and type(runtime.get("activation_checkpointing")) is bool,
            "microbatch/checkpointing")
    require(runtime.get("global_ce_eligible_images") == GLOBAL_IMAGE_COUNT and
            runtime.get("global_active_tokens") == sum(active), "22-image denominators")
    require(runtime.get("max_model_forwards") == runtime.get("updates") * GLOBAL_IMAGE_COUNT,
            "22-image exposure accounting")
    calls_per_update = sum(math.ceil(n / runtime["microbatch_size"]) for n in RANK_COUNTS)
    require(runtime.get("max_model_calls") == runtime.get("updates") * calls_per_update,
            "eight-rank model-call accounting")
    require(runtime.get("updates") in (2, 256), "fixed qualification or production dose")
    expected_steps = [2] if runtime["updates"] == 2 else [8, 16, 32, 64, 128, 256]
    require(runtime.get("checkpoint_steps") == expected_steps, "frozen checkpoint schedule")
    require(value["optimizer"] == training.DEFAULT_OPTIMIZER and
            value["validity_hinge"]["weight"] == 0.01 and
            value["validity_hinge"]["margin"] == 1 / 999, "unchanged AdamW/geometry recipe")
    return dict(value)


def resume_adapter_identity(
    *, manifest_path: Path, manifest: Mapping[str, Any], resume: Path
) -> tuple[Path, dict[str, Any]]:
    """Validate a completed distributed checkpoint before cold model loading.

    The manifest binds the original starting adapter. A resumed model instead
    loads the saved adapter, whose own inspected payload is checked against the
    checkpoint state and eight-rank consensus. Optimizer/layout/RNG restore
    remains with the shared _restore caller after the model is bound.
    """
    root = Path(resume).resolve(strict=True)
    state = torch.load(root / "state.pt", map_location="cpu", weights_only=False)
    require(
        state.get("schema") == f"{training.SCHEMA}.checkpoint.v1"
        and state.get("manifest") == training.binding(manifest_path)
        and state.get("source_adapter") == manifest["source_adapter"]
        and state.get("optimizer") == manifest["optimizer"],
        "resume checkpoint original manifest/optimizer lineage",
    )
    step = state.get("step")
    require(
        type(step) is int and step in manifest["runtime"]["checkpoint_steps"]
        and 0 < step < manifest["runtime"]["updates"],
        "resume saved step outside frozen dose",
    )
    consensus = json.loads((root / "consensus.json").read_text())
    require(
        consensus.get("step") == step
        and consensus.get("rank_count") == REQUIRED_WORLD_SIZE
        and consensus.get("state", {}).get("optimizer_steps") == [step],
        "resume eight-rank checkpoint consensus",
    )
    adapter_path = root / "adapter"
    observed = training.inspect_dora_adapter_payload(
        adapter_path, manifest["model_config"]["model"]["base_model"]
    )
    require(
        observed == state.get("saved_adapter")
        and Path(observed["root"]).resolve(strict=True) == adapter_path,
        "resume saved adapter payload identity",
    )
    return adapter_path, observed


def _rank_receipt(
    *,
    rank: int,
    local_rank: int,
    route_indices: Sequence[int],
    routes: Sequence[Mapping[str, Any]],
    start_step: int,
    terminal_step: int,
    local_logical_forwards: int,
    local_model_calls: int,
    started: float,
    device: torch.device,
    preparation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}.rank.v1",
        "status": "completed",
        "rank": rank,
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "device": str(device),
        "route_indices": list(route_indices),
        "route_ids": [route["route_id"] for route in routes],
        "local_image_count": len(routes),
        "start_step": start_step,
        "terminal_step": terminal_step,
        "local_logical_forwards": local_logical_forwards,
        "local_model_calls": local_model_calls,
        "elapsed_seconds": time.monotonic() - started,
        "native_preparation": dict(preparation),
        "resources": distributed._resource_receipt(device),
    }


def run(
    manifest_path: Path, *, output: Path, resume: Path | None = None
) -> dict[str, Any] | None:
    """Execute a full 22-image, eight-rank update under torchrun."""
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig
    from src.qwen.checkpointing import (
        install_language_decoder_checkpointing,
        language_decoder_checkpointing_receipt as checkpointing_receipt,
    )

    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    dependencies = dependency_bindings()
    require(
        training.binding(manifest["sources"]["producer"]["path"])
        == manifest["sources"]["producer"],
        "producer binding changed",
    )
    require(
        Path(manifest["sources"]["producer"]["path"]).resolve()
        == Path(__file__).resolve(),
        "manifest producer is not COCO22 backend",
    )
    require(torch.cuda.is_available(), "COCO22 distributed training requires CUDA")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    require(
        world_size == REQUIRED_WORLD_SIZE
        and 0 <= rank < world_size
        and 0 <= local_rank < world_size,
        "launch with eight torchrun ranks",
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=COLLECTIVE_TIMEOUT_SECONDS)
    )
    started = time.monotonic()
    phase = "output_setup"
    start_step = 0
    local_logical_forwards = 0
    local_model_calls = 0
    checkpoints: list[dict[str, Any]] = []
    checkpoint_consensus: list[dict[str, Any]] = []
    checkpoint_timings: list[dict[str, Any]] = []
    gradient_snapshot: dict[str, Any] | None = None
    try:

        def prepare_output() -> None:
            if rank == 0:
                require(not output.exists(), "attempt output already exists")
                output.mkdir(parents=True)

        distributed._coordinated_call(prepare_output, phase=phase)
        dist.barrier()
        phase = "model_setup"
        setup_started = time.monotonic()

        def setup_model() -> tuple[
            Any,
            Mapping[str, Any],
            torch.nn.Module,
            tuple[tuple[str, torch.nn.Parameter], ...],
            tuple[tuple[str, torch.nn.Parameter], ...],
            Mapping[str, Any],
            torch.optim.Optimizer,
            int,
            Mapping[str, Any],
        ]:
            random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            if resume:
                adapter_path, cold_adapter = resume_adapter_identity(
                    manifest_path=manifest_path, manifest=manifest, resume=resume
                )
            else:
                adapter_path = Path(manifest["source_adapter"]["root"])
                cold_adapter = manifest["source_adapter"]
            source = str(adapter_path)
            config = checkpoint_config(
                InferConfig.model_validate(manifest["model_config"]), source
            )
            qwen, loaded = load_policy(config, device=device)
            from probes.training_set_completion.coco227_readback import _validate_loaded

            _validate_loaded(
                loaded, adapter=cold_adapter,
                config=manifest["model_config"],
            )
            model = qwen.model
            model.eval()
            named, frozen = training.bind_language_dora(
                model, source_adapter=manifest["source_adapter"]
            )
            checkpointing = install_language_decoder_checkpointing(
                model, expected_layer_count=28
            )
            enabled = manifest["runtime"]["activation_checkpointing"]
            checkpointing.update(
                enabled=enabled, phase="train" if enabled else "train_disabled"
            )
            optimizer = torch.optim.AdamW(
                [parameter for _, parameter in named],
                **{
                    **manifest["optimizer"],
                    "betas": tuple(manifest["optimizer"]["betas"]),
                },
            )
            require(
                resume is not None or not optimizer.state,
                "fresh attempt requires fresh AdamW",
            )
            restored = (
                training._restore(
                    resume,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    optimizer=optimizer,
                    named=named,
                )
                if resume
                else 0
            )
            require(
                restored < manifest["runtime"]["updates"], "resume already terminal"
            )
            return (
                qwen,
                loaded,
                model,
                named,
                frozen,
                checkpointing,
                optimizer,
                restored,
                cold_adapter,
            )

        qwen, loaded, model, named, frozen, checkpointing, optimizer, start_step, cold_adapter = (
            distributed._coordinated_call(setup_model, phase=phase)
        )
        setup_seconds = time.monotonic() - setup_started
        distributed._require_consensus(
            distributed.state_fingerprint(named, optimizer), label="initial state"
        )
        route_indices = partition_route_indices(
            GLOBAL_IMAGE_COUNT, rank=rank, world_size=world_size
        )
        local_routes = [manifest["routes"][index] for index in route_indices]
        phase = "native_input_setup"
        native_started = time.monotonic()
        microbatches, preparation = distributed._coordinated_call(
            lambda: _prepare_microbatches(
                qwen,
                manifest,
                local_routes,
                device=device,
                microbatch_size=manifest["runtime"]["microbatch_size"],
            ),
            phase=phase,
        )
        native_seconds = time.monotonic() - native_started
        pad_token_id = qwen.tokenizer.pad_token_id
        require(type(pad_token_id) is int and pad_token_id >= 0, "runtime pad token")
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(start_step + 1, manifest["runtime"]["updates"] + 1):
            optimizer.zero_grad(set_to_none=True)
            update_started = time.monotonic()
            cards = []
            local_ce_numerator = 0.0
            local_hinge_sum = 0.0
            local_step_calls = 0
            history_padding = 0
            prompt_padding = preparation["prompt_padding_tokens"]
            phase = f"update_{step}_forward_backward"
            forward_started = time.monotonic()

            def forward_backward() -> None:
                nonlocal \
                    local_logical_forwards, \
                    local_model_calls, \
                    local_ce_numerator, \
                    local_hinge_sum, \
                    local_step_calls, \
                    history_padding
                for group in microbatches:
                    logits_rows, padding = _batched_aligned_logits(
                        model,
                        group["inputs"],
                        group["routes"],
                        pad_token_id=pad_token_id,
                    )
                    ce_means = []
                    hinges = []
                    active = []
                    for logits, route in zip(logits_rows, group["routes"], strict=True):
                        ce, raw_hinge, count, card = _route_terms(
                            logits, route, manifest["validity_hinge"]
                        )
                        ce_means.append(ce)
                        hinges.append(raw_hinge)
                        active.append(count)
                        cards.append(card)
                        local_ce_numerator += (
                            float(ce.detach())
                            if manifest["objective"]["ce_reduction"] == "sample_equal"
                            else float(ce.detach()) * count
                        )
                        local_hinge_sum += float(raw_hinge.detach())
                    loss = objective_from_route_terms(
                        ce_means,
                        active,
                        hinges,
                        ce_reduction=manifest["objective"]["ce_reduction"],
                        global_ce_eligible_images=manifest["runtime"][
                            "global_ce_eligible_images"
                        ],
                        global_active_tokens=manifest["runtime"][
                            "global_active_tokens"
                        ],
                        geometry_weight=manifest["validity_hinge"]["weight"],
                    )
                    loss.backward()
                    local_logical_forwards += len(group["routes"])
                    local_model_calls += 1
                    local_step_calls += 1
                    history_padding += padding["history_padding_tokens"]
                require(
                    all(
                        parameter.grad is not None
                        and bool(torch.isfinite(parameter.grad).all())
                        for _, parameter in named
                    ),
                    "missing/nonfinite local DoRA gradients",
                )
                require(
                    all(parameter.grad is None for _, parameter in frozen),
                    "frozen parameter received gradient",
                )

            distributed._coordinated_call(forward_backward, phase=phase)
            forward_seconds = time.monotonic() - forward_started
            phase = f"update_{step}_gradient_sum"
            collective_started = time.monotonic()
            distributed.sum_gradients_(named)
            collective_seconds = time.monotonic() - collective_started
            if step == 1 and manifest["runtime"].get("capture_gradient_step1", False):
                snapshot_started = time.monotonic()

                def save_snapshot() -> None:
                    nonlocal gradient_snapshot
                    if rank == 0:
                        path = output / "qualification-gradient-step-00001.pt"
                        torch.save(
                            {
                                name: parameter.grad.detach().cpu()
                                for name, parameter in named
                            },
                            path,
                        )
                        gradient_snapshot = training.binding(path)

                distributed._coordinated_call(
                    save_snapshot, phase="qualification_gradient_snapshot"
                )
                snapshot_seconds = time.monotonic() - snapshot_started
            else:
                snapshot_seconds = 0.0
            raw_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in named],
                    1.0,
                    error_if_nonfinite=True,
                    foreach=False,
                )
            )
            optimizer_started = time.monotonic()
            optimizer.step()
            optimizer_seconds = time.monotonic() - optimizer_started
            metric = torch.tensor(
                [local_ce_numerator, local_hinge_sum],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
            global_ce = float(
                metric[0].item()
                / (
                    manifest["runtime"]["global_ce_eligible_images"]
                    if manifest["objective"]["ce_reduction"] == "sample_equal"
                    else manifest["runtime"]["global_active_tokens"]
                )
            )
            global_hinge = float(metric[1].item() / GLOBAL_IMAGE_COUNT)
            gathered_cards = distributed._gather_objects(cards)
            ordered_cards = [card for rows in gathered_cards for card in rows]
            require(
                [card["route_id"] for card in ordered_cards]
                == [route["route_id"] for route in manifest["routes"]],
                "gathered route order",
            )
            local_timing = {
                "rank": rank,
                "forward_backward_seconds": forward_seconds,
                "gradient_collective_seconds": collective_seconds,
                "optimizer_seconds": optimizer_seconds,
                "gradient_snapshot_io_seconds": snapshot_seconds,
                "compute_seconds": time.monotonic() - update_started,
                "local_model_calls": local_step_calls,
                "local_logical_forwards": len(local_routes),
                "history_padding_tokens": history_padding,
                "prompt_padding_tokens": prompt_padding,
                "resources": distributed._resource_receipt(device),
            }
            rank_timings = distributed._gather_objects(local_timing)
            update = {
                "schema": f"{SCHEMA}.update.v1",
                "step": step,
                "ce_reduction": manifest["objective"]["ce_reduction"],
                "image_count": GLOBAL_IMAGE_COUNT,
                "active_tokens": manifest["runtime"]["global_active_tokens"],
                "global_ce_eligible_images": manifest["runtime"][
                    "global_ce_eligible_images"
                ],
                "global_ce": global_ce,
                "global_geometry_mean": global_hinge,
                "geometry_weight": manifest["validity_hinge"]["weight"],
                "objective_total": global_ce
                + manifest["validity_hinge"]["weight"] * global_hinge,
                "routes": ordered_cards,
                "gradient_norm_before_clip": raw_norm,
                "start_step": start_step,
                "logical_model_forwards": (step - start_step) * GLOBAL_IMAGE_COUNT,
                "model_calls": (step - start_step)
                * sum(
                    math.ceil(count / manifest["runtime"]["microbatch_size"])
                    for count in (3, 3, 3, 3, 3, 3, 2, 2)
                ),
                "distributed": {
                    "schema": f"{SCHEMA}.update_distributed.v1",
                    "world_size": world_size,
                    "rank_image_counts": [3, 3, 3, 3, 3, 3, 2, 2],
                    "rank_timings": rank_timings,
                    "normalization": {
                        "ce_reduction": manifest["objective"]["ce_reduction"],
                        "ce_denominator": manifest["runtime"][
                            "global_ce_eligible_images"
                        ]
                        if manifest["objective"]["ce_reduction"] == "sample_equal"
                        else manifest["runtime"]["global_active_tokens"],
                        "geometry_denominator": GLOBAL_IMAGE_COUNT,
                        "gradient_collective": "SUM",
                        "post_collective_divisor": 1,
                    },
                },
            }
            distributed._coordinated_call(
                lambda: training.publish(
                    output / "updates" / f"step-{step:05d}.json", update
                )
                if rank == 0
                else None,
                phase=f"update_{step}_receipt",
            )
            if step in manifest["runtime"]["checkpoint_steps"]:
                phase = f"checkpoint_{step}"
                state = distributed.state_fingerprint(named, optimizer)
                states = distributed._require_consensus(
                    state, label=f"checkpoint {step}"
                )
                require(state["optimizer_steps"] == [step], "optimizer counter")
                save_started = time.monotonic()
                holder: dict[str, Any] = {}

                def save_checkpoint() -> None:
                    if rank == 0:
                        holder["receipt"] = training._checkpoint(
                            output,
                            manifest_path=manifest_path,
                            manifest=manifest,
                            model=model,
                            optimizer=optimizer,
                            named=named,
                            step=step,
                        )

                distributed._coordinated_call(
                    save_checkpoint, phase=f"checkpoint_{step}_save"
                )
                dist.barrier()
                save_seconds = time.monotonic() - save_started
                consensus = {"step": step, "state": state, "rank_count": len(states)}
                distributed._coordinated_call(
                    lambda: training.publish(
                        output / "checkpoints" / f"step-{step:05d}" / "consensus.json",
                        consensus,
                    )
                    if rank == 0
                    else None,
                    phase=f"checkpoint_{step}_consensus_receipt",
                )
                gathered_io = distributed._gather_objects(
                    {"rank": rank, "checkpoint_io_seconds": save_seconds}
                )
                checkpoint_consensus.append(consensus)
                checkpoint_timings.append({"step": step, "rank_timings": gathered_io})
                if rank == 0:
                    checkpoints.append(holder["receipt"])
        receipt = _rank_receipt(
            rank=rank,
            local_rank=local_rank,
            route_indices=route_indices,
            routes=local_routes,
            start_step=start_step,
            terminal_step=manifest["runtime"]["updates"],
            local_logical_forwards=local_logical_forwards,
            local_model_calls=local_model_calls,
            started=started,
            device=device,
            preparation=preparation
            | {
                "model_setup_seconds": setup_seconds,
                "native_input_setup_seconds": native_seconds,
            },
        )
        receipts = distributed._gather_objects(receipt)
        segment_steps = manifest["runtime"]["updates"] - start_step
        segment_calls = segment_steps * sum(
            math.ceil(count / manifest["runtime"]["microbatch_size"])
            for count in RANK_COUNTS
        )
        require(
            sum(row["local_logical_forwards"] for row in receipts)
            == segment_steps * GLOBAL_IMAGE_COUNT
            and sum(row["local_model_calls"] for row in receipts) == segment_calls,
            "executed segment/full-cohort receipt counters",
        )
        distributed._coordinated_call(
            lambda: training.publish(
                output / "ranks" / f"rank-{rank:03d}.json", receipt
            ),
            phase="rank_receipt",
        )
        terminal = {
            "schema": f"{SCHEMA}.terminal.v1",
            "status": "completed_segment" if resume else "completed",
            "manifest": training.binding(manifest_path),
            "loaded_model": loaded,
            "optimizer_mode": "resume" if resume else "fresh",
            "cold_loaded_adapter": cold_adapter,
            "resume_origin": {
                "root": str(Path(resume).resolve(strict=True)),
                "state": training.binding(resume / "state.pt"),
                "consensus": training.binding(resume / "consensus.json"),
                "saved_adapter": cold_adapter,
            } if resume else None,
            "trainable_surface": training._layout(named),
            "declared_target_updates": manifest["runtime"]["updates"],
            "start_step": start_step,
            "updates": segment_steps,
            "logical_model_forwards": segment_steps
            * GLOBAL_IMAGE_COUNT,
            "model_calls": segment_calls,
            "checkpoints": checkpoints,
            "checkpoint_timings": checkpoint_timings,
            "gradient_snapshot_step1": gradient_snapshot,
            "activation_checkpointing": checkpointing_receipt(model, checkpointing),
            "elapsed_seconds": time.monotonic() - started,
            "distributed": {
                "schema": SCHEMA,
                "backend": dist.get_backend(),
                "world_size": world_size,
                "rank_image_counts": [3, 3, 3, 3, 3, 3, 2, 2],
                "normalization": {
                    "ce_reduction": manifest["objective"]["ce_reduction"],
                    "ce_denominator": manifest["runtime"]["global_ce_eligible_images"]
                    if manifest["objective"]["ce_reduction"] == "sample_equal"
                    else manifest["runtime"]["global_active_tokens"],
                    "geometry_denominator": GLOBAL_IMAGE_COUNT,
                    "gradient_collective": "SUM",
                    "post_collective_divisor": 1,
                },
                "rank_receipts": receipts,
                "checkpoint_consensus": checkpoint_consensus,
                "source_bindings": {
                    "training_backend": training.binding(Path(__file__)),
                    **dependencies,
                },
            },
        }
        distributed._coordinated_call(
            lambda: training.publish(output / "terminal.json", terminal)
            if rank == 0
            else None,
            phase="terminal",
        )
        dist.barrier()
        return terminal if rank == 0 else None
    except Exception as exc:
        if rank == 0 and output.exists() and not (output / "terminal.json").exists():
            try:
                training.publish(
                    output / "terminal.json",
                    {
                        "schema": f"{SCHEMA}.terminal.v1",
                        "status": "failed",
                        "manifest": training.binding(manifest_path),
                        "phase": phase,
                        "step": start_step,
                        "local_logical_forwards": local_logical_forwards,
                        "local_model_calls": local_model_calls,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.monotonic() - started,
                        "distributed": {"rank": rank, "world_size": world_size},
                    },
                )
            except Exception:
                pass
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    terminal = run(args.manifest, output=args.output, resume=args.resume)
    if terminal is not None:
        print(json.dumps(terminal, sort_keys=True))


if __name__ == "__main__":
    main()

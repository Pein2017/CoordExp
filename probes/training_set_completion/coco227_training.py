"""Four-rank batched training for the frozen COCO227 CE-normalization trial.

Each rank owns a contiguous 3/3/3/2 slice of the eleven image routes.  A
microbatch changes only how many images share a model call.  The declared CE
numerator and geometry contribution are accumulated with global denominators
before a single SUM gradient collective, so neither uneven rank work nor
padding can introduce a local mean-of-means.
"""

from __future__ import annotations

from probes.training_set_completion import replay

import argparse
import json
import math
import os
import random
import signal
import socket
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from probes.training_set_completion import distributed
from probes.training_set_completion import training


SCHEMA = "training_set_completion.coco227_training.v1"
GLOBAL_IMAGE_COUNT = 11
OWNER_COUNT = 227
REQUIRED_WORLD_SIZE = 4
COLLECTIVE_TIMEOUT_SECONDS = 600
SOURCE_ADAPTER_FINGERPRINT = (
    "8eb3d6c99692c744d534958fa419fedcd20b833549618e104bff58e1bc15deff"
)
CE_REDUCTIONS = ("sample_equal", "global_active_token_equal")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def dependency_bindings() -> dict[str, dict[str, Any]]:
    """Record current helper identities; historical runs retain their own frozen hashes."""

    root = Path(__file__).resolve().parents[2]
    return {
        "training_helpers": training.binding(Path(training.__file__)),
        "artifact_primitives": training.binding(root / "probes/training_set_completion/artifacts.py"),
        "native_replay_helpers": training.binding(root / "src/qwen/native.py"),
        "batched_replay_helpers": training.binding(Path(replay.__file__)),
        "distributed_helpers": training.binding(Path(distributed.__file__)),
        "partition_recipe": training.binding(root / "probes/training_set_completion/dual_start_distributed.py"),
        "shared_raw_axis_validity_hinge": training.binding(
            root / "src/losses/raw_axis_validity_hinge.py"
        ),
    }


def partition_route_indices(total: int, *, rank: int, world_size: int) -> list[int]:
    from probes.training_set_completion.dual_start_distributed import partition_route_indices as fixed_partition
    return fixed_partition(total, rank=rank, world_size=world_size)


def objective_from_route_terms(
    ce_means: Sequence[torch.Tensor],
    active_tokens: Sequence[int],
    raw_hinges: Sequence[torch.Tensor],
    *,
    ce_reduction: str,
    global_ce_eligible_images: int,
    global_active_tokens: int,
    global_image_count: int = GLOBAL_IMAGE_COUNT,
    geometry_weight: float = 0.01,
) -> torch.Tensor:
    """Return a local contribution whose cross-rank SUM is the global objective."""
    require(
        len(ce_means) == len(active_tokens) == len(raw_hinges), "route term cardinality"
    )
    require(ce_reduction in CE_REDUCTIONS, "unsupported CE reduction")
    require(
        global_ce_eligible_images > 0 and global_active_tokens > 0,
        "global CE denominator",
    )
    require(
        global_image_count > 0 and geometry_weight >= 0, "geometry denominator/weight"
    )
    require(
        all(type(value) is int and value > 0 for value in active_tokens),
        "active-token counts",
    )
    if ce_means:
        zero = ce_means[0].new_zeros(())
    elif raw_hinges:
        zero = raw_hinges[0].new_zeros(())
    else:
        raise ValueError("empty local route terms")
    if ce_reduction == "sample_equal":
        ce = sum(ce_means, zero) / global_ce_eligible_images
    else:
        ce = (
            sum(
                (
                    mean * count
                    for mean, count in zip(ce_means, active_tokens, strict=True)
                ),
                zero,
            )
            / global_active_tokens
        )
    geometry = geometry_weight * sum(raw_hinges, zero) / global_image_count
    return ce + geometry


def validate_manifest(
    value: Mapping[str, Any], *, verify_sources: bool = True
) -> dict[str, Any]:
    value = training.validate_manifest(value, verify_sources=verify_sources)
    require(
        value["source_adapter"]["fingerprint"] == SOURCE_ADAPTER_FINGERPRINT,
        "wrong common source adapter",
    )
    require(
        len(value["routes"]) == GLOBAL_IMAGE_COUNT, "COCO227 requires 11 image routes"
    )
    require(
        sum(len(route["trusted_boxes"]) for route in value["routes"]) == OWNER_COUNT,
        "teacher must contain exactly 227 boxes",
    )
    objective = value.get("objective")
    require(
        isinstance(objective, Mapping)
        and set(objective) == {"ce_reduction", "geometry_reduction"},
        "objective fields",
    )
    require(objective["ce_reduction"] in CE_REDUCTIONS, "CE reduction")
    require(
        objective["geometry_reduction"] == "global_equal_image_mean",
        "geometry reduction changed",
    )
    active = [sum(route["ce_weights"]) for route in value["routes"]]
    require(
        all(count > 0 for count in active), "actual teacher has no all-masked image"
    )
    runtime = value["runtime"]
    require(
        runtime.get("seed") == 42
        and runtime.get("initial_step") == 0
        and runtime.get("optimizer_mode") == "fresh",
        "fresh seed42 recipe",
    )
    require(runtime.get("microbatch_size") in (1, 2, 3), "runtime microbatch size")
    require(
        type(runtime.get("activation_checkpointing")) is bool,
        "activation checkpointing switch",
    )
    require(
        runtime.get("global_ce_eligible_images") == GLOBAL_IMAGE_COUNT,
        "sample-equal denominator",
    )
    require(
        runtime.get("global_active_tokens") == sum(active), "token-equal denominator"
    )
    require(
        runtime.get("max_model_forwards") == runtime["updates"] * GLOBAL_IMAGE_COUNT,
        "logical exposure budget",
    )
    model_calls = sum(
        math.ceil(count / runtime["microbatch_size"]) for count in (3, 3, 3, 2)
    )
    require(
        runtime.get("max_model_calls") == runtime["updates"] * model_calls,
        "physical model-call budget",
    )
    if runtime["updates"] == 2:
        require(
            runtime["checkpoint_steps"] == [2] and runtime["wall_seconds"] <= 600,
            "qualification budget",
        )
    else:
        require(
            runtime["updates"] == 256
            and runtime["checkpoint_steps"] == [8, 16, 32, 64, 128, 256]
            and runtime["wall_seconds"] == 7200,
            "full training budget",
        )
    require(value["optimizer"] == training.DEFAULT_OPTIMIZER, "fresh AdamW recipe")
    require(
        value["validity_hinge"]["weight"] == 0.01
        and value["validity_hinge"]["margin"] == 1 / 999,
        "geometry recipe",
    )
    return dict(value)


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
        "resources": distributed.resource_receipt(device),
    }


def run(
    manifest_path: Path, *, output: Path, resume: Path | None = None
) -> dict[str, Any] | None:
    """Execute one S or T arm under ``python -m torch.distributed.run``."""
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
        "manifest producer is not COCO227 backend",
    )
    require(torch.cuda.is_available(), "COCO227 distributed training requires CUDA")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    require(
        world_size == REQUIRED_WORLD_SIZE
        and 0 <= rank < world_size
        and 0 <= local_rank < world_size,
        "launch with four torchrun ranks",
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
    old_alarm = signal.getsignal(signal.SIGALRM)

    def expired(*_: Any) -> None:
        raise TimeoutError("COCO227 distributed training wall budget")

    try:

        def prepare_output() -> None:
            if rank == 0:
                require(not output.exists(), "attempt output already exists")
                output.mkdir(parents=True)

        distributed.coordinated_call(prepare_output, phase=phase)
        dist.barrier()
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(math.ceil(manifest["runtime"]["wall_seconds"]))
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
        ]:
            random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            source = str(
                (resume / "adapter") if resume else manifest["source_adapter"]["root"]
            )
            config = checkpoint_config(
                InferConfig.model_validate(manifest["model_config"]), source
            )
            qwen, loaded = load_policy(config, device=device)
            model = qwen.model
            model.eval()
            named, frozen = training.bind_language_dora(
                model, source_adapter=manifest["source_adapter"]
            )
            checkpointing = install_language_decoder_checkpointing(model, expected_layer_count=28)
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
            )

        qwen, loaded, model, named, frozen, checkpointing, optimizer, start_step = (
            distributed.coordinated_call(setup_model, phase=phase)
        )
        setup_seconds = time.monotonic() - setup_started
        distributed.require_consensus(
            distributed.state_fingerprint(named, optimizer), label="initial state"
        )
        route_indices = partition_route_indices(
            GLOBAL_IMAGE_COUNT, rank=rank, world_size=world_size
        )
        local_routes = [manifest["routes"][index] for index in route_indices]
        phase = "native_input_setup"
        native_started = time.monotonic()
        microbatches, preparation = distributed.coordinated_call(
            lambda: replay.prepare_microbatches(
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
                    logits_rows, padding = replay.batched_aligned_logits(
                        model,
                        group["inputs"],
                        group["routes"],
                        pad_token_id=pad_token_id,
                    )
                    ce_means = []
                    hinges = []
                    active = []
                    for logits, route in zip(logits_rows, group["routes"], strict=True):
                        ce, raw_hinge, count, card = replay.route_terms(
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

            distributed.coordinated_call(forward_backward, phase=phase)
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

                distributed.coordinated_call(
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
            gathered_cards = distributed.gather_objects(cards)
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
                "resources": distributed.resource_receipt(device),
            }
            rank_timings = distributed.gather_objects(local_timing)
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
                "logical_model_forwards": step * GLOBAL_IMAGE_COUNT,
                "model_calls": step
                * sum(
                    math.ceil(count / manifest["runtime"]["microbatch_size"])
                    for count in (3, 3, 3, 2)
                ),
                "distributed": {
                    "schema": f"{SCHEMA}.update_distributed.v1",
                    "world_size": world_size,
                    "rank_image_counts": [3, 3, 3, 2],
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
            distributed.coordinated_call(
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
                states = distributed.require_consensus(
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

                distributed.coordinated_call(
                    save_checkpoint, phase=f"checkpoint_{step}_save"
                )
                dist.barrier()
                save_seconds = time.monotonic() - save_started
                consensus = {"step": step, "state": state, "rank_count": len(states)}
                distributed.coordinated_call(
                    lambda: training.publish(
                        output / "checkpoints" / f"step-{step:05d}" / "consensus.json",
                        consensus,
                    )
                    if rank == 0
                    else None,
                    phase=f"checkpoint_{step}_consensus_receipt",
                )
                gathered_io = distributed.gather_objects(
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
        receipts = distributed.gather_objects(receipt)
        distributed.coordinated_call(
            lambda: training.publish(
                output / "ranks" / f"rank-{rank:03d}.json", receipt
            ),
            phase="rank_receipt",
        )
        terminal = {
            "schema": f"{SCHEMA}.terminal.v1",
            "status": "completed",
            "manifest": training.binding(manifest_path),
            "loaded_model": loaded,
            "optimizer_mode": "resume" if resume else "fresh",
            "trainable_surface": training._layout(named),
            "updates": manifest["runtime"]["updates"],
            "logical_model_forwards": manifest["runtime"]["updates"]
            * GLOBAL_IMAGE_COUNT,
            "model_calls": manifest["runtime"]["max_model_calls"],
            "checkpoints": checkpoints,
            "checkpoint_timings": checkpoint_timings,
            "gradient_snapshot_step1": gradient_snapshot,
            "activation_checkpointing": checkpointing_receipt(model, checkpointing),
            "elapsed_seconds": time.monotonic() - started,
            "distributed": {
                "schema": SCHEMA,
                "backend": dist.get_backend(),
                "world_size": world_size,
                "rank_image_counts": [3, 3, 3, 2],
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
        distributed.coordinated_call(
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
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
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

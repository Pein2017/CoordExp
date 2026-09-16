"""Four-rank backend for the frozen dual-start training comparison.

Each rank owns a contiguous subset of the eleven source-ordered image routes.
It backpropagates each per-image objective divided by the global denominator
eleven, then gradients are summed exactly once.  There is no DDP wrapper and
therefore no collective inside the unequal 3/3/3/2 forward/backward loops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import signal
import socket
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import torch
import torch.distributed as dist

from probes.training_set_completion import training


REQUIRED_WORLD_SIZE = 4
GLOBAL_IMAGE_COUNT = 11
EXPECTED_ACTIVE_TOKENS = 2_088
EXPECTED_IMAGE_IDS = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
EXPECTED_TEACHER_SHA256 = "ef79536316453b274dbe6b45d47d769c64d77805c2fb080f1b977eab80d1bc25"
ALLOWED_SOURCE_ADAPTERS = {
    "d7563a96275cced00b34a2078cdfa558018a380328e221c270fff0b23234fc61",
    "b8ca2461c93bf32e886c9e42f7ab0e52495ef2d439a605207c86af7258408815",
}
COLLECTIVE_TIMEOUT_SECONDS = 600
SCHEMA = "training_set_completion.dual_start_distributed.v1"
T = TypeVar("T")


class DistributedTrainingError(RuntimeError):
    """A failure that every live rank has observed and accepted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def dependency_bindings() -> dict[str, dict[str, Any]]:
    """Record the executed helper sources without pinning today's tree to an old hash."""

    root = Path(__file__).resolve().parents[2]
    return {
        "training_helpers": training.binding(Path(training.__file__)),
        "shared_raw_axis_validity_hinge": training.binding(root / "src/losses/raw_axis_validity_hinge.py"),
    }


def partition_route_indices(total: int, *, rank: int, world_size: int) -> list[int]:
    """Return the frozen contiguous 3/3/3/2 source-order partition."""
    require(
        total == GLOBAL_IMAGE_COUNT and world_size == REQUIRED_WORLD_SIZE,
        "distributed dual-start requires exactly 11 images across 4 ranks",
    )
    require(0 <= rank < world_size, "rank outside distributed world")
    quotient, remainder = divmod(total, world_size)
    count = quotient + int(rank < remainder)
    start = rank * quotient + min(rank, remainder)
    return list(range(start, start + count))


def validate_distributed_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the manifest is one of the two frozen execution modes."""
    routes = manifest.get("routes")
    require(isinstance(routes, list) and len(routes) == GLOBAL_IMAGE_COUNT, "distributed route denominator must be 11")
    if all(isinstance(route, Mapping) and "image_id" in route for route in routes):
        require(tuple(int(route["image_id"]) for route in routes) == EXPECTED_IMAGE_IDS, "teacher source image order changed")
    if all(isinstance(route, Mapping) and "ce_weights" in route for route in routes):
        active_tokens = sum(sum(int(weight) for weight in route["ce_weights"]) for route in routes)
        require(active_tokens == EXPECTED_ACTIVE_TOKENS, "teacher active-token denominator changed")

    require(manifest.get("optimizer") == training.DEFAULT_OPTIMIZER, "frozen fresh AdamW recipe changed")
    hinge = manifest.get("validity_hinge", {})
    require(hinge.get("weight") == 0.01 and hinge.get("margin") == 1 / 999, "frozen geometry recipe changed")
    adapter = manifest.get("source_adapter")
    if isinstance(adapter, Mapping) and "fingerprint" in adapter:
        require(adapter["fingerprint"] in ALLOWED_SOURCE_ADAPTERS, "source adapter is outside frozen A/B starts")
    sources = manifest.get("sources")
    if isinstance(sources, Mapping) and isinstance(sources.get("reviewed_routes"), Mapping):
        require(sources["reviewed_routes"].get("sha256") == EXPECTED_TEACHER_SHA256, "teacher bank bytes changed")

    runtime = manifest.get("runtime")
    require(isinstance(runtime, Mapping), "distributed runtime contract")
    require(runtime.get("seed") == 42, "distributed dual-start requires seed42")
    require(runtime.get("initial_step") == 0 and runtime.get("optimizer_mode") == "fresh", "fresh new-dose numbering changed")
    updates = runtime.get("updates")
    require(updates in (2, 256), "only the 2-update qualification and frozen 256-update modes are admitted")
    expected = {
        2: {"checkpoint_steps": [2], "max_model_forwards": 22},
        256: {"checkpoint_steps": [64, 128, 256], "max_model_forwards": 2_816},
    }[int(updates)]
    require(runtime.get("checkpoint_steps") == expected["checkpoint_steps"], "checkpoint schedule changed")
    require(runtime.get("max_model_forwards") == expected["max_model_forwards"], "global image-forward budget changed")
    if updates == 2:
        require(type(runtime.get("wall_seconds")) is int and 0 < runtime["wall_seconds"] <= 600, "qualification wall must be at most 600 seconds")
    else:
        require(runtime.get("wall_seconds") == 7_200, "full training wall must be 7200 seconds")
    return dict(manifest)


def _coordination_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")


def raise_if_rank_failed(error: BaseException | None, *, phase: str) -> None:
    """Propagate an ordinary local failure before starting the next collective phase."""
    local_failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=_coordination_device())
    dist.all_reduce(local_failed, op=dist.ReduceOp.MAX)
    if not int(local_failed.item()):
        return
    local_message = None if error is None else f"{type(error).__name__}: {error}"
    messages: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(messages, local_message)
    failures = "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(messages) if message is not None)
    raise DistributedTrainingError(f"{phase} failed collectively: {failures}")


def _coordinated_call(function: Callable[[], T], *, phase: str) -> T:
    value: T | None = None
    error: BaseException | None = None
    try:
        value = function()
    except Exception as exc:
        error = exc
    raise_if_rank_failed(error, phase=phase)
    return value  # type: ignore[return-value]


def sum_gradients_(named: Sequence[tuple[str, torch.nn.Parameter]]) -> None:
    """SUM already-global-normalized gradients in a fixed parameter order."""
    preflight_error: BaseException | None = None
    try:
        for name, parameter in named:
            if parameter.grad is None:
                raise ValueError(f"missing local gradient before SUM: {name}")
            if not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError(f"nonfinite local gradient before SUM: {name}")
    except Exception as exc:
        preflight_error = exc
    raise_if_rank_failed(preflight_error, phase="gradient_sum_preflight")
    for _, parameter in named:
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    postflight_error: BaseException | None = None
    try:
        for name, parameter in named:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError(f"nonfinite global gradient after SUM: {name}")
    except Exception as exc:
        postflight_error = exc
    raise_if_rank_failed(postflight_error, phase="gradient_sum_postflight")


def _hash_tensor(hasher: Any, *, name: str, tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous()
    hasher.update(training.canonical({"name": name, "shape": list(cpu.shape), "dtype": str(cpu.dtype)}))
    hasher.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())


def state_fingerprint(
    named: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Hash trainable parameters and Adam state without persisting per-rank copies."""
    parameters = hashlib.sha256()
    optimizer_state = hashlib.sha256()
    steps: set[int] = set()
    for name, parameter in named:
        _hash_tensor(parameters, name=name, tensor=parameter)
        state = optimizer.state.get(parameter, {})
        optimizer_state.update(training.canonical({"parameter": name, "keys": sorted(state)}))
        for key in sorted(state):
            value = state[key]
            if isinstance(value, torch.Tensor):
                _hash_tensor(optimizer_state, name=f"{name}:{key}", tensor=value)
                if key == "step" and value.numel() == 1:
                    steps.add(int(value.item()))
            else:
                optimizer_state.update(training.canonical({"name": f"{name}:{key}", "value": value}))
                if key == "step":
                    steps.add(int(value))
    return {
        "parameter_sha256": parameters.hexdigest(),
        "optimizer_sha256": optimizer_state.hexdigest(),
        "optimizer_steps": sorted(steps),
        "parameter_count": len(named),
        "scalar_count": sum(parameter.numel() for _, parameter in named),
    }


def _gather_objects(value: T) -> list[T]:
    values: list[T | None] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return [item for item in values if item is not None]


def _require_consensus(value: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    values = _gather_objects(dict(value))
    require(len(values) == dist.get_world_size(), f"{label} rank receipt missing")
    require(len({training.digest(item) for item in values}) == 1, f"{label} differs across ranks")
    return values


def _resource_receipt(device: torch.device) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    if device.type == "cuda":
        receipt.update(
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
    return receipt


def _rank_receipt(
    *,
    rank: int,
    local_rank: int,
    route_indices: Sequence[int],
    routes: Sequence[Mapping[str, Any]],
    device: torch.device,
    start_step: int,
    terminal_step: int,
    local_forwards: int,
    started: float,
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
        "local_image_count": len(route_indices),
        "start_step": start_step,
        "terminal_step": terminal_step,
        "local_forwards": local_forwards,
        "elapsed_seconds": time.monotonic() - started,
        "resources": _resource_receipt(device),
    }


def run(manifest_path: Path, *, output: Path, resume: Path | None = None) -> dict[str, Any] | None:
    """Execute the frozen four-rank backend under torchrun."""
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig
    from src.qwen.checkpointing import (
        install_language_decoder_checkpointing,
        language_decoder_checkpointing_receipt as checkpointing_receipt,
    )

    manifest = validate_distributed_contract(training.validate_manifest(json.loads(manifest_path.read_text())))
    executed_dependency_bindings = dependency_bindings()
    require(training.binding(manifest["sources"]["producer"]["path"]) == manifest["sources"]["producer"], "distributed producer binding changed")
    require(Path(manifest["sources"]["producer"]["path"]).resolve() == Path(__file__).resolve(), "manifest producer is not distributed backend")
    require(torch.cuda.is_available(), "distributed dual-start requires CUDA")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    require(world_size == REQUIRED_WORLD_SIZE and 0 <= rank < world_size and 0 <= local_rank < world_size, "launch with torchrun nproc-per-node=4")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=COLLECTIVE_TIMEOUT_SECONDS))

    started = time.monotonic()
    local_forwards = 0
    start_step = 0
    phase = "output_setup"
    checkpoints: list[dict[str, Any]] = []
    checkpoint_consensus: list[dict[str, Any]] = []
    old_alarm = signal.getsignal(signal.SIGALRM)

    def expired(*_: Any) -> None:
        raise TimeoutError("distributed training wall budget")

    try:
        def prepare_output() -> None:
            if rank == 0:
                require(not output.exists(), "attempt output already exists")
                output.mkdir(parents=True)

        _coordinated_call(prepare_output, phase=phase)
        dist.barrier()
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(math.ceil(manifest["runtime"]["wall_seconds"]))

        phase = "model_setup"

        def setup_model() -> tuple[Any, Mapping[str, Any], torch.nn.Module, tuple[tuple[str, torch.nn.Parameter], ...], tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any], torch.optim.Optimizer, int]:
            random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            source = str((resume / "adapter") if resume else manifest["source_adapter"]["root"])
            config = checkpoint_config(InferConfig.model_validate(manifest["model_config"]), source)
            qwen, loaded = load_policy(config, device=device)
            model = qwen.model
            model.eval()
            named, frozen = training.bind_language_dora(model, source_adapter=manifest["source_adapter"])
            checkpointing = install_language_decoder_checkpointing(model, expected_layer_count=28)
            checkpointing.update(enabled=True, phase="train")
            optimizer = torch.optim.AdamW(
                [parameter for _, parameter in named],
                **{**manifest["optimizer"], "betas": tuple(manifest["optimizer"]["betas"])},
            )
            require(resume is not None or not optimizer.state, "fresh attempt requires fresh AdamW")
            restored = training._restore(
                resume,
                manifest_path=manifest_path,
                manifest=manifest,
                optimizer=optimizer,
                named=named,
            ) if resume else 0
            require(restored < manifest["runtime"]["updates"], "resume already reaches terminal step")
            return qwen, loaded, model, named, frozen, checkpointing, optimizer, restored

        qwen, loaded, model, named, frozen, checkpointing, optimizer, start_step = _coordinated_call(setup_model, phase=phase)
        initial_state = state_fingerprint(named, optimizer)
        _require_consensus(initial_state, label="initial trainable/optimizer state")

        route_indices = partition_route_indices(GLOBAL_IMAGE_COUNT, rank=rank, world_size=world_size)
        local_routes = [manifest["routes"][index] for index in route_indices]
        phase = "native_input_setup"
        entries = _coordinated_call(
            lambda: training._native_entries(qwen, {**manifest, "routes": local_routes}, device),
            phase=phase,
        )
        require(len(entries) == len(route_indices), "local native-entry denominator changed")
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)

        for step in range(start_step + 1, manifest["runtime"]["updates"] + 1):
            phase = f"train_update_{step}_local_objectives"
            update_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            local_losses: list[torch.Tensor] = []
            local_cards: list[dict[str, Any]] = []

            def local_objectives() -> None:
                nonlocal local_forwards
                for entry in entries:
                    loss, card = training.route_objective(model, entry["inputs"], entry["route"], manifest["validity_hinge"])
                    (loss / GLOBAL_IMAGE_COUNT).backward()
                    local_losses.append(loss.detach())
                    local_cards.append({"route_id": entry["route"]["route_id"], **card})
                    local_forwards += 1
                require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for _, parameter in named), "missing/nonfinite local DoRA gradients")
                require(all(parameter.grad is None for _, parameter in frozen), "frozen parameter received gradient")

            _coordinated_call(local_objectives, phase=phase)
            phase = f"train_update_{step}_gradient_sum"
            sum_gradients_(named)

            local_loss_sum = torch.tensor(
                [sum(float(loss.item()) for loss in local_losses)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
            global_objective_mean = float(local_loss_sum.item() / GLOBAL_IMAGE_COUNT)
            gathered_cards = _gather_objects(local_cards)
            cards = [card for rank_cards in gathered_cards for card in rank_cards]
            require([card["route_id"] for card in cards] == [route["route_id"] for route in manifest["routes"]], "gathered route order changed")

            phase = f"train_update_{step}_optimizer"
            raw_norm = float(torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named],
                1.0,
                error_if_nonfinite=True,
                foreach=False,
            ))
            optimizer.step()

            local_update_receipt = {
                "rank": rank,
                "local_image_count": len(entries),
                "local_forwards": local_forwards,
                "local_objective_sum": sum(float(loss.item()) for loss in local_losses),
                "elapsed_seconds": time.monotonic() - update_started,
                "resources": _resource_receipt(device),
            }
            rank_updates = _gather_objects(local_update_receipt)
            update = {
                "schema": f"{training.SCHEMA}.update.v1",
                "step": step,
                "image_count": GLOBAL_IMAGE_COUNT,
                "objective_mean_over_images": global_objective_mean,
                "routes": cards,
                "gradient_norm_before_clip": raw_norm,
                "forwards": step * GLOBAL_IMAGE_COUNT,
                "distributed": {
                    "schema": f"{SCHEMA}.update.v1",
                    "world_size": world_size,
                    "rank_image_counts": [len(item) for item in gathered_cards],
                    "rank_updates": rank_updates,
                    "normalization": {
                        "per_image_objective": "active-token-mean CE + 0.01 * per-image mean raw-axis hinge",
                        "local_backward_divisor": GLOBAL_IMAGE_COUNT,
                        "gradient_collective": "SUM",
                        "post_collective_divisor": 1,
                        "effective_global_objective": "mean over 11 per-image objectives",
                    },
                },
            }
            _coordinated_call(
                lambda: training.publish(output / "updates" / f"step-{step:05d}.json", update) if rank == 0 else None,
                phase=f"train_update_{step}_receipt",
            )

            if step in manifest["runtime"]["checkpoint_steps"] or step == manifest["runtime"]["updates"]:
                phase = f"checkpoint_{step}_consensus"
                state = state_fingerprint(named, optimizer)
                states = _require_consensus(state, label=f"checkpoint {step} trainable/optimizer state")
                require(state["optimizer_steps"] == [step], f"checkpoint {step} optimizer counter")
                checkpoint_consensus.append({"step": step, "state": state, "rank_count": len(states)})

                def save_checkpoint() -> None:
                    if rank == 0:
                        checkpoints.append(training._checkpoint(
                            output,
                            manifest_path=manifest_path,
                            manifest=manifest,
                            model=model,
                            optimizer=optimizer,
                            named=named,
                            step=step,
                        ))

                _coordinated_call(save_checkpoint, phase=f"checkpoint_{step}_save")
                dist.barrier()

        phase = "rank_receipts"
        rank_receipt = _rank_receipt(
            rank=rank,
            local_rank=local_rank,
            route_indices=route_indices,
            routes=local_routes,
            device=device,
            start_step=start_step,
            terminal_step=manifest["runtime"]["updates"],
            local_forwards=local_forwards,
            started=started,
        )
        rank_receipts = _gather_objects(rank_receipt)
        _coordinated_call(
            lambda: training.publish(output / "ranks" / f"rank-{rank:03d}.json", rank_receipt),
            phase=phase,
        )

        terminal = {
            "schema": f"{training.SCHEMA}.terminal.v1",
            "status": "completed",
            "manifest": training.binding(manifest_path),
            "loaded_model": loaded,
            "optimizer_mode": "resume" if resume else "fresh",
            "trainable_surface": training._layout(named),
            "updates": manifest["runtime"]["updates"],
            "model_forwards": manifest["runtime"]["updates"] * GLOBAL_IMAGE_COUNT,
            "checkpoints": checkpoints,
            "activation_checkpointing": checkpointing_receipt(model, checkpointing),
            "elapsed_seconds": time.monotonic() - started,
            "distributed": {
                "schema": SCHEMA,
                "backend": dist.get_backend(),
                "world_size": world_size,
                "global_image_count": GLOBAL_IMAGE_COUNT,
                "rank_image_counts": [len(partition_route_indices(GLOBAL_IMAGE_COUNT, rank=item, world_size=world_size)) for item in range(world_size)],
                "normalization": {
                    "local_backward_divisor": GLOBAL_IMAGE_COUNT,
                    "gradient_collective": "SUM",
                    "post_collective_divisor": 1,
                },
                "logical_model_forwards": manifest["runtime"]["updates"] * GLOBAL_IMAGE_COUNT,
                "invocation_model_forwards": (manifest["runtime"]["updates"] - start_step) * GLOBAL_IMAGE_COUNT,
                "invocation_updates": manifest["runtime"]["updates"] - start_step,
                "rank_receipts": rank_receipts,
                "checkpoint_consensus": checkpoint_consensus,
                "source_bindings": {
                    "distributed_backend": training.binding(Path(__file__)),
                    **executed_dependency_bindings,
                },
            },
        }
        _coordinated_call(
            lambda: training.publish(output / "terminal.json", terminal) if rank == 0 else None,
            phase="terminal_publication",
        )
        dist.barrier()
        return terminal if rank == 0 else None
    except Exception as exc:
        if rank == 0 and output.exists() and not (output / "terminal.json").exists():
            failed = {
                "schema": f"{training.SCHEMA}.terminal.v1",
                "status": "failed",
                "manifest": training.binding(manifest_path),
                "phase": phase,
                "step": start_step,
                "model_forwards": local_forwards,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.monotonic() - started,
                "distributed": {"schema": SCHEMA, "rank": rank, "world_size": world_size},
            }
            try:
                training.publish(output / "terminal.json", failed)
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

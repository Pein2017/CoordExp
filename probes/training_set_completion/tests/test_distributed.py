from __future__ import annotations

from probes.training_set_completion import distributed as shared_distributed

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from probes.training_set_completion import dual_start_distributed as distributed


def _model() -> torch.nn.Module:
    model = torch.nn.Linear(3, 2, bias=True, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.3, 0.5], [-0.7, 0.1, 0.4]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.05, -0.15], dtype=torch.float64))
    return model


def _samples() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.tensor([float(index + 1), float((index % 4) - 1), float(index % 3)], dtype=torch.float64),
            torch.tensor([float(index % 5) / 4.0, -float(index + 2) / 9.0], dtype=torch.float64),
        )
        for index in range(distributed.GLOBAL_IMAGE_COUNT)
    ]


def _optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )


def _distributed_worker(rank: int, world_size: int, init_path: str, output: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _model()
        optimizer = _optimizer(model)
        named = tuple(model.named_parameters())
        local_indices = distributed.partition_route_indices(
            distributed.GLOBAL_IMAGE_COUNT,
            rank=rank,
            world_size=world_size,
        )
        optimizer.zero_grad(set_to_none=True)
        for index in local_indices:
            features, target = _samples()[index]
            per_image_loss = torch.nn.functional.mse_loss(model(features), target)
            (per_image_loss / distributed.GLOBAL_IMAGE_COUNT).backward()

        shared_distributed.sum_gradients_(named)
        gradients = {name: parameter.grad.detach().tolist() for name, parameter in named}
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named],
            1.0,
            error_if_nonfinite=True,
            foreach=False,
        )
        optimizer.step()
        state = shared_distributed.state_fingerprint(named, optimizer)
        parameters = {name: parameter.detach().tolist() for name, parameter in named}
        states = [None] * world_size
        dist.all_gather_object(states, state)

        for _, parameter in named:
            parameter.grad = torch.ones_like(parameter)
        if rank == 2:
            named[-1][1].grad = None
        coordinated_error = None
        try:
            shared_distributed.sum_gradients_(named)
        except shared_distributed.DistributedTrainingError as exc:
            coordinated_error = str(exc)

        Path(output, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "local_indices": local_indices,
                    "gradients": gradients,
                    "parameters": parameters,
                    "state": state,
                    "all_states": states,
                    "coordinated_error": coordinated_error,
                },
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        dist.destroy_process_group()


def test_real_multiprocess_sum_matches_serial_global_image_mean_and_fails_closed(tmp_path: Path):
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")

    init_path = tmp_path / "gloo-init"
    mp.spawn(
        _distributed_worker,
        args=(distributed.REQUIRED_WORLD_SIZE, str(init_path), str(tmp_path)),
        nprocs=distributed.REQUIRED_WORLD_SIZE,
        join=True,
    )
    rank_results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(4)]

    assert [len(item["local_indices"]) for item in rank_results] == [3, 3, 3, 2]
    assert [index for item in rank_results for index in item["local_indices"]] == list(range(11))

    serial = _model()
    serial_optimizer = _optimizer(serial)
    serial_optimizer.zero_grad(set_to_none=True)
    serial_loss = torch.stack(
        [torch.nn.functional.mse_loss(serial(features), target) for features, target in _samples()]
    ).mean()
    serial_loss.backward()
    serial_gradients = {name: parameter.grad.detach() for name, parameter in serial.named_parameters()}

    gradient_max_abs_diff = 0.0
    for rank_result in rank_results:
        for name, expected in serial_gradients.items():
            observed = torch.tensor(rank_result["gradients"][name], dtype=expected.dtype)
            torch.testing.assert_close(observed, expected)
            gradient_max_abs_diff = max(gradient_max_abs_diff, float((observed - expected).abs().max()))

    torch.nn.utils.clip_grad_norm_(serial.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
    serial_optimizer.step()
    serial_state = shared_distributed.state_fingerprint(tuple(serial.named_parameters()), serial_optimizer)
    parameter_max_abs_diff = 0.0
    for rank_result in rank_results:
        for name, expected in serial.named_parameters():
            observed = torch.tensor(rank_result["parameters"][name], dtype=expected.dtype)
            torch.testing.assert_close(observed, expected)
            parameter_max_abs_diff = max(parameter_max_abs_diff, float((observed - expected).abs().max()))
    distributed_state = rank_results[0]["state"]
    assert all(item["state"] == distributed_state for item in rank_results)
    assert all(item["all_states"] == [distributed_state] * 4 for item in rank_results)
    assert distributed_state["optimizer_steps"] == serial_state["optimizer_steps"] == [1]
    assert gradient_max_abs_diff <= 1e-12
    assert parameter_max_abs_diff <= 1e-12
    assert all("gradient_sum_preflight" in item["coordinated_error"] for item in rank_results)
    assert all("rank 2: ValueError: missing local gradient" in item["coordinated_error"] for item in rank_results)
    print(json.dumps({"gradient_max_abs_diff": gradient_max_abs_diff, "parameter_max_abs_diff": parameter_max_abs_diff}, sort_keys=True))


def test_frozen_partition_and_manifest_modes_fail_closed():
    dependencies = distributed.dependency_bindings()
    assert dependencies["training_helpers"] == distributed.training.binding(
        Path(distributed.training.__file__)
    )
    assert dependencies["shared_raw_axis_validity_hinge"] == distributed.training.binding(
        Path(distributed.__file__).resolve().parents[2] / "src/losses/raw_axis_validity_hinge.py"
    )
    assert distributed.partition_route_indices(11, rank=0, world_size=4) == [0, 1, 2]
    assert distributed.partition_route_indices(11, rank=3, world_size=4) == [9, 10]
    with pytest.raises(ValueError, match="exactly 11 images across 4 ranks"):
        distributed.partition_route_indices(10, rank=0, world_size=4)

    mechanics = {
        "routes": [{} for _ in range(11)],
        "optimizer": dict(distributed.training.DEFAULT_OPTIMIZER),
        "validity_hinge": {"weight": 0.01, "margin": 1 / 999},
        "runtime": {
            "updates": 2,
            "checkpoint_steps": [2],
            "wall_seconds": 600,
            "max_model_forwards": 22,
            "seed": 42,
            "initial_step": 0,
            "optimizer_mode": "fresh",
        },
    }
    assert distributed.validate_distributed_contract(mechanics)["runtime"]["updates"] == 2
    full = json.loads(json.dumps(mechanics))
    full["runtime"].update(
        updates=256,
        checkpoint_steps=[64, 128, 256],
        wall_seconds=7200,
        max_model_forwards=2816,
    )
    distributed.validate_distributed_contract(full)
    full["runtime"]["seed"] = 43
    with pytest.raises(ValueError, match="seed42"):
        distributed.validate_distributed_contract(full)

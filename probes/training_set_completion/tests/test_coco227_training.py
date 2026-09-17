from __future__ import annotations

from probes.training_set_completion import replay

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from probes.training_set_completion import coco227_training as training227
from src.qwen.native import select_compact_replay_logits


ACTIVE = [1, 7, 2, 11, 3, 13, 5, 17, 19, 23, 29]


def test_execution_bindings_include_the_actual_partition_recipe():
    from probes.training_set_completion import dual_start_distributed

    expected = training227.training.binding(Path(dual_start_distributed.__file__))
    assert expected in training227.dependency_bindings().values()


def _model() -> torch.nn.Module:
    model = torch.nn.Linear(3, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.3, 0.5]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.05], dtype=torch.float64))
    return model


def _route_terms(
    model: torch.nn.Module, index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [float(index + 1), float(index % 4 - 1), float(index % 3)], dtype=torch.float64
    )
    value = model(features).squeeze()
    ce_mean = (value - (index + 1) / 9) ** 2
    raw_hinge = (value + (index % 5) / 7) ** 2
    return ce_mean, raw_hinge


def _worker(rank: int, mode: str, init_path: str, output: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_path}", rank=rank, world_size=4
    )
    try:
        model = _model()
        named = tuple(model.named_parameters())
        indices = training227.partition_route_indices(11, rank=rank, world_size=4)
        ce_means, hinges = zip(
            *[_route_terms(model, index) for index in indices], strict=True
        )
        objective = training227.objective_from_route_terms(
            ce_means,
            [ACTIVE[index] for index in indices],
            hinges,
            ce_reduction=mode,
            global_ce_eligible_images=11,
            global_active_tokens=sum(ACTIVE),
        )
        objective.backward()
        training227.distributed.sum_gradients_(named)
        Path(output, f"{mode}-{rank}.json").write_text(
            json.dumps(
                {
                    "indices": indices,
                    "gradients": {
                        name: parameter.grad.tolist() for name, parameter in named
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", training227.CE_REDUCTIONS)
def test_unequal_four_rank_reduction_matches_serial_objective(
    tmp_path: Path, mode: str
):
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    mp.spawn(
        _worker,
        args=(mode, str(tmp_path / f"init-{mode}"), str(tmp_path)),
        nprocs=4,
        join=True,
    )
    rows = [
        json.loads((tmp_path / f"{mode}-{rank}.json").read_text()) for rank in range(4)
    ]
    assert [len(row["indices"]) for row in rows] == [3, 3, 3, 2]
    assert [index for row in rows for index in row["indices"]] == list(range(11))

    serial = _model()
    ce_means, hinges = zip(
        *[_route_terms(serial, index) for index in range(11)], strict=True
    )
    objective = training227.objective_from_route_terms(
        ce_means,
        ACTIVE,
        hinges,
        ce_reduction=mode,
        global_ce_eligible_images=11,
        global_active_tokens=sum(ACTIVE),
    )
    objective.backward()
    expected = dict(serial.named_parameters())
    for row in rows:
        for name, parameter in expected.items():
            torch.testing.assert_close(
                torch.tensor(row["gradients"][name], dtype=torch.float64),
                parameter.grad,
            )


def test_sample_and_token_equal_have_distinct_ce_but_identical_geometry_weighting():
    parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    ce = [parameter * scale for scale in (1.0, 2.0, 8.0)]
    hinges = [parameter.square(), 2 * parameter.square(), 4 * parameter.square()]
    sample = training227.objective_from_route_terms(
        ce,
        [1, 3, 12],
        hinges,
        ce_reduction="sample_equal",
        global_ce_eligible_images=3,
        global_active_tokens=16,
        global_image_count=3,
    )
    token = training227.objective_from_route_terms(
        ce,
        [1, 3, 12],
        hinges,
        ce_reduction="global_active_token_equal",
        global_ce_eligible_images=3,
        global_active_tokens=16,
        global_image_count=3,
    )
    geometry = 0.01 * sum(hinges) / 3
    torch.testing.assert_close(sample - sum(ce) / 3, geometry)
    torch.testing.assert_close(
        token
        - sum(value * count for value, count in zip(ce, [1, 3, 12], strict=True)) / 16,
        geometry,
    )
    assert not torch.isclose(sample, token)


def test_left_padded_compact_logits_select_exact_causal_rows_and_gradients():
    lengths = [2, 5, 3]
    width = max(lengths) + 1
    logits = torch.arange(3 * width * 7, dtype=torch.float64).reshape(3, width, 7)
    logits.requires_grad_(True)
    observed = select_compact_replay_logits(logits, lengths)
    expected = [
        logits[index, width - count - 1 : width - 1].float()
        for index, count in enumerate(lengths)
    ]
    for left, right in zip(observed, expected, strict=True):
        torch.testing.assert_close(left, right)

    observed_loss = sum(row.square().mean() for row in observed)
    observed_loss.backward()
    observed_gradient = logits.grad.detach().clone()
    logits.grad = None
    expected_loss = sum(row.square().mean() for row in expected)
    expected_loss.backward()
    torch.testing.assert_close(logits.grad, observed_gradient)


class _ExactHistoryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.03, dtype=torch.float64))

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        video_grid_thw: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del image_grid_thw, video_grid_thw
        positions = attention_mask.cumsum(dim=1).sub(1).clamp_min(0)
        return positions.unsqueeze(0).expand(3, -1, -1), torch.zeros(
            input_ids.shape[0], 1, dtype=torch.long
        )

    def forward(self, **inputs: torch.Tensor | int | bool) -> SimpleNamespace:
        ids = inputs["input_ids"]
        assert isinstance(ids, torch.Tensor)
        vocab = torch.arange(13, dtype=torch.float64).view(1, 1, -1)
        logits = self.scale * ids.double().unsqueeze(-1) * vocab
        keep = inputs["logits_to_keep"]
        assert isinstance(keep, int)
        return SimpleNamespace(logits=logits[:, -keep:])


def test_exact_history_batch_padding_and_supervision_gradient_match_serial():
    routes = [
        {"prompt_token_ids": [1, 2], "continuation_token_ids": [3, 4]},
        {"prompt_token_ids": [5, 6, 7, 8], "continuation_token_ids": [9]},
        {"prompt_token_ids": [2], "continuation_token_ids": [6, 5, 4]},
    ]
    native = {
        "input_ids": torch.ones((3, 2), dtype=torch.long),
        "image_grid_thw": torch.ones((3, 3), dtype=torch.long),
    }
    batched_model = _ExactHistoryModel()
    batched, padding = replay.batched_aligned_logits(
        batched_model, native, routes, pad_token_id=0
    )
    batched_loss = sum(
        training227.training.masked_ce_loss(
            logits,
            torch.tensor(route["continuation_token_ids"]),
            [1] * len(route["continuation_token_ids"]),
        )[0]
        for logits, route in zip(batched, routes, strict=True)
    )
    batched_loss.backward()

    serial_model = _ExactHistoryModel()
    serial_loss = serial_model.scale.new_zeros(())
    for index, route in enumerate(routes):
        logits, _ = replay.batched_aligned_logits(
            serial_model,
            {
                "input_ids": native["input_ids"][index : index + 1],
                "image_grid_thw": native["image_grid_thw"][index : index + 1],
            },
            [route],
            pad_token_id=0,
        )
        serial_loss = (
            serial_loss
            + training227.training.masked_ce_loss(
                logits[0],
                torch.tensor(route["continuation_token_ids"]),
                [1] * len(route["continuation_token_ids"]),
            )[0]
        )
    serial_loss.backward()

    assert padding["history_padding_tokens"] > 0
    torch.testing.assert_close(batched_loss.double(), serial_loss)
    torch.testing.assert_close(batched_model.scale.grad, serial_model.scale.grad)


def test_microbatch_slices_preserve_local_order_without_empty_batches():
    assert replay.microbatch_slices(3, 1) == [[0], [1], [2]]
    assert replay.microbatch_slices(3, 2) == [[0, 1], [2]]
    assert replay.microbatch_slices(3, 3) == [[0, 1, 2]]
    assert replay.microbatch_slices(2, 3) == [[0, 1]]

from __future__ import annotations

import math

import pytest
import torch

from scripts.research.human13_gradient_preservation import (
    GradientProjectionError,
    project_and_write_gradients,
)


def _parameters():
    return (
        ("b", torch.nn.Parameter(torch.tensor([0.0]))),
        ("a", torch.nn.Parameter(torch.tensor([0.0, 0.0]))),
    )


def test_negative_dot_is_projected_and_written_before_optimizer() -> None:
    parameters = _parameters()
    receipt = project_and_write_gradients(
        parameters,
        r1_gradients={"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([2.0])},
        watch_gradients={"a": torch.tensor([-1.0, 1.0]), "b": torch.tensor([-1.0])},
        epsilon=1e-12,
        tolerance=1e-6,
        world_size=1,
    )

    assert receipt.applied
    assert receipt.pre_dot == -3.0
    assert receipt.post_dot >= -receipt.tolerance
    assert receipt.written_post_dot >= -receipt.tolerance
    assert receipt.projection_coefficient < 0
    written = {name: parameter.grad.float() for name, parameter in parameters}
    actual = sum(
        (written[name] * watch).sum()
        for name, watch in {
            "a": torch.tensor([-1.0, 1.0]),
            "b": torch.tensor([-1.0]),
        }.items()
    )
    assert float(actual) >= -1e-6


def test_nonconflicting_gradient_is_unchanged() -> None:
    parameters = _parameters()
    source = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    receipt = project_and_write_gradients(
        parameters,
        r1_gradients=source,
        watch_gradients={"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([1.0])},
        epsilon=1e-12,
        tolerance=1e-6,
        world_size=1,
    )
    assert not receipt.applied
    assert receipt.projection_coefficient == 0.0
    assert torch.equal(dict(parameters)["a"].grad, source["a"])
    assert torch.equal(dict(parameters)["b"].grad, source["b"])


def test_parameter_order_does_not_change_projection() -> None:
    first = _parameters()
    second = tuple(reversed(_parameters()))
    kwargs = {
        "r1_gradients": {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([2.0])},
        "watch_gradients": {"a": torch.tensor([-1.0, 1.0]), "b": torch.tensor([-1.0])},
        "epsilon": 1e-12,
        "tolerance": 1e-6,
        "world_size": 1,
    }
    left = project_and_write_gradients(first, **kwargs)
    right = project_and_write_gradients(second, **kwargs)
    assert left == right
    assert all(
        torch.equal(dict(first)[name].grad, dict(second)[name].grad)
        for name in ("a", "b")
    )


def test_projected_direction_is_first_order_nonadverse_to_watch_loss() -> None:
    parameters = (("p", torch.nn.Parameter(torch.tensor([0.0, 0.0]))),)
    receipt = project_and_write_gradients(
        parameters,
        r1_gradients={"p": torch.tensor([1.0, 0.0])},
        watch_gradients={"p": torch.tensor([-1.0, 1.0])},
        epsilon=1e-12,
        tolerance=1e-6,
        world_size=1,
    )
    step = -dict(parameters)["p"].grad
    directional_watch_change = torch.dot(torch.tensor([-1.0, 1.0]), step)
    assert receipt.applied
    assert float(directional_watch_change) <= 1e-6


@pytest.mark.parametrize(
    ("r1", "watch", "world_size", "match"),
    [
        ({"a": torch.ones(2)}, {"a": torch.ones(2)}, 2, "world-size one"),
        ({"a": torch.ones(2)}, {"a": torch.ones(2)}, 1, "gradient names"),
        (
            {"a": torch.ones(2), "b": torch.ones(1)},
            {"a": torch.zeros(2), "b": torch.zeros(1)},
            1,
            "watch norm",
        ),
        (
            {"a": torch.tensor([math.nan, 1.0]), "b": torch.ones(1)},
            {"a": torch.ones(2), "b": torch.ones(1)},
            1,
            "finite",
        ),
    ],
)
def test_projection_fails_closed(r1, watch, world_size: int, match: str) -> None:
    with pytest.raises(GradientProjectionError, match=match):
        project_and_write_gradients(
            _parameters(),
            r1_gradients=r1,
            watch_gradients=watch,
            epsilon=1e-12,
            tolerance=1e-6,
            world_size=world_size,
        )

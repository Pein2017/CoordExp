"""World-size-one accumulated-gradient preservation for the Human-13 successor."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import torch


@dataclass(frozen=True)
class GradientProjectionReceipt:
    schema_version: str
    parameter_count: int
    pre_dot: float
    post_dot: float
    written_post_dot: float
    r1_norm: float
    watch_norm: float
    projected_norm: float
    projection_coefficient: float
    applied: bool
    epsilon: float
    tolerance: float
    all_finite: bool
    world_size: int


class GradientProjectionError(ValueError):
    """Raised before optimizer mutation when projection evidence is unusable."""


def project_and_write_gradients(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    r1_gradients: Mapping[str, torch.Tensor],
    watch_gradients: Mapping[str, torch.Tensor],
    epsilon: float,
    tolerance: float,
    world_size: int,
) -> GradientProjectionReceipt:
    """Project one complete R1 gradient and write it into ``parameter.grad``.

    Inputs are explicit buffers so the caller cannot accidentally project one
    pack at a time.  The helper owns no backward, clipping, or optimizer step.
    """

    if world_size != 1:
        raise GradientProjectionError("gradient preservation requires world-size one")
    checked_epsilon = float(epsilon)
    checked_tolerance = float(tolerance)
    if (
        not math.isfinite(checked_epsilon)
        or checked_epsilon <= 0
        or not math.isfinite(checked_tolerance)
        or checked_tolerance < 0
    ):
        raise GradientProjectionError("epsilon/tolerance must be finite and valid")

    parameter_entries = tuple(named_parameters)
    parameters = dict(parameter_entries)
    if not parameters or len(parameters) != len(parameter_entries):
        raise GradientProjectionError("projection requires unique named parameters")
    expected = set(parameters)
    if set(r1_gradients) != expected or set(watch_gradients) != expected:
        raise GradientProjectionError("parameter and gradient names must match exactly")
    devices = {parameter.device for parameter in parameters.values()}
    if len(devices) != 1:
        raise GradientProjectionError("projection requires one parameter device")
    device = next(iter(devices))

    r1: dict[str, torch.Tensor] = {}
    watch: dict[str, torch.Tensor] = {}
    for name in sorted(parameters):
        parameter = parameters[name]
        first = r1_gradients[name]
        second = watch_gradients[name]
        if (
            not isinstance(first, torch.Tensor)
            or not isinstance(second, torch.Tensor)
            or first.shape != parameter.shape
            or second.shape != parameter.shape
        ):
            raise GradientProjectionError(
                f"gradient shape for {name} does not match its parameter"
            )
        first = first.detach().to(device=device, dtype=torch.float32)
        second = second.detach().to(device=device, dtype=torch.float32)
        if not bool(torch.isfinite(first).all().item()) or not bool(
            torch.isfinite(second).all().item()
        ):
            raise GradientProjectionError("all projection gradients must be finite")
        r1[name] = first
        watch[name] = second

    pre_dot_tensor = sum((r1[name] * watch[name]).sum() for name in sorted(parameters))
    r1_sq = sum(r1[name].square().sum() for name in sorted(parameters))
    watch_sq = sum(watch[name].square().sum() for name in sorted(parameters))
    if not bool(torch.isfinite(watch_sq).item()) or float(watch_sq) <= checked_epsilon:
        raise GradientProjectionError("watch norm is zero, tiny, or non-finite")

    pre_dot = float(pre_dot_tensor.item())
    applied = pre_dot < 0.0
    coefficient = pre_dot / float(watch_sq.item()) if applied else 0.0
    projected = {
        name: (
            r1[name] - r1[name].new_tensor(coefficient) * watch[name]
            if applied
            else r1[name]
        )
        for name in sorted(parameters)
    }
    post_dot_tensor = sum(
        (projected[name] * watch[name]).sum() for name in sorted(parameters)
    )
    projected_sq = sum(projected[name].square().sum() for name in sorted(parameters))
    written = {
        name: projected[name].to(dtype=parameters[name].dtype)
        for name in sorted(parameters)
    }
    written_post_dot_tensor = sum(
        (written[name].float() * watch[name]).sum() for name in sorted(parameters)
    )
    post_dot = float(post_dot_tensor.item())
    written_post_dot = float(written_post_dot_tensor.item())
    scalar_values = (
        pre_dot,
        post_dot,
        written_post_dot,
        float(r1_sq.item()),
        float(watch_sq.item()),
        float(projected_sq.item()),
        coefficient,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise GradientProjectionError("projection result is not finite")
    if post_dot < -checked_tolerance or written_post_dot < -checked_tolerance:
        raise GradientProjectionError(
            "projected gradient remains adverse beyond numerical tolerance"
        )

    with torch.no_grad():
        for name in sorted(parameters):
            parameters[name].grad = written[name].detach().clone()

    return GradientProjectionReceipt(
        schema_version="human13_gradient_projection.v1",
        parameter_count=len(parameters),
        pre_dot=pre_dot,
        post_dot=post_dot,
        written_post_dot=written_post_dot,
        r1_norm=math.sqrt(float(r1_sq.item())),
        watch_norm=math.sqrt(float(watch_sq.item())),
        projected_norm=math.sqrt(float(projected_sq.item())),
        projection_coefficient=coefficient,
        applied=applied,
        epsilon=checked_epsilon,
        tolerance=checked_tolerance,
        all_finite=True,
        world_size=world_size,
    )


__all__ = [
    "GradientProjectionError",
    "GradientProjectionReceipt",
    "project_and_write_gradients",
]

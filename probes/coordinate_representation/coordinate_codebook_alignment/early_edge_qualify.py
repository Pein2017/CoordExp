"""Exercise the early hook through the maintained tiny training caller."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import torch

from probes.coordinate_representation.coordinate_codebook_alignment import qualify
from src.qwen.coordinate_codebook import _visual_module


class EarlyEdgeProbe(qualify._LossQualificationProbe):
    def __init__(self, output: Path) -> None:
        super().__init__(output)
        self.early_grid: torch.Tensor | None = None
        self.early_hidden: torch.Tensor | None = None

    def capture_forward_hooks(self, model: object) -> list[object]:
        handles = super().capture_forward_hooks(model)
        visual = _visual_module(qualify._unwrap_model(model))

        def capture_grid(_module: object, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
            if self.early_grid is not None:
                return
            value = kwargs.get("grid_thw") if "grid_thw" in kwargs else args[1]
            self.early_grid = torch.as_tensor(value).detach().clone()

        def capture_input(_module: object, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
            if self.early_hidden is not None:
                return
            hidden = args[0] if args else kwargs.get("hidden_states")
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("first vision block has no tensor hidden state")
            self.early_hidden = hidden.detach().clone()

        handles.append(visual.register_forward_pre_hook(capture_grid, with_kwargs=True, prepend=True))
        handles.append(visual.blocks[0].register_forward_pre_hook(capture_input, with_kwargs=True, prepend=True))
        return handles

    def _address_checks(self, device: torch.device) -> None:
        model = qualify._unwrap_model(self.runtime.model) if self.runtime is not None else None
        codebook = getattr(model, "coordinate_codebook", None)
        if codebook is None or self.early_grid is None or self.early_hidden is None:
            self.address_checks = {"status": "unavailable"}
            return
        hidden = self.early_hidden.to(device)
        grid = self.early_grid.to(device)
        edges = codebook.patch_edge_coordinates(grid, device=device)
        live = codebook.inject_early(hidden, grid)
        _, output_ids = qualify._embedding_delta_ids(model)
        output_delta = next(parameter for parameter in model.parameters() if id(parameter) in output_ids)
        weights = (1 + torch.arange(live.numel(), device=device, dtype=torch.float32) / live.numel()).reshape_as(live)
        objective = (live.float() * weights).sum()
        output_grad, projection_grad = torch.autograd.grad(
            objective, (output_delta, codebook.projection.weight), retain_graph=True,
            allow_unused=True,
        )
        wrong_order = codebook.inject_early(hidden, grid, edge_coordinates=edges.roll(1, 0))
        edge_swap = codebook.inject_early(hidden, grid, edge_coordinates=edges[:, [1, 0, 2, 3]])
        rows, width = codebook._effective_rows()
        with patch.object(codebook, "_effective_rows", return_value=(rows.detach(), width)):
            detached = codebook.inject_early(hidden, grid)
            detached_grad = torch.autograd.grad(
                (detached.float() * weights).sum(), output_delta,
                retain_graph=True, allow_unused=True,
            )[0]
        self.address_checks = {
            "status": "checked",
            "live_output_delta_gradient_nonzero": bool(output_grad is not None and output_grad.abs().max() > 0),
            "projection_gradient_nonzero": bool(projection_grad is not None and projection_grad.abs().max() > 0),
            "stale_detached_output_delta_gradient_none": detached_grad is None,
            "wrong_grid_detected": bool((live.float() - wrong_order.float()).abs().max() > 0),
            "wrong_order_detected": bool((live.float() - wrong_order.float()).abs().max() > 0),
            "edge_swap_detected": bool((live.float() - edge_swap.float()).abs().max() > 0),
            "raw_patches": int(edges.shape[0]),
            "edge_corner_min": float(edges.min()),
            "edge_corner_max": float(edges.max()),
        }

    def validate(self) -> None:
        super().validate()
        if not all(self.address_checks.get(key, False) for key in (
            "projection_gradient_nonzero", "wrong_order_detected", "edge_swap_detected",
        )):
            raise AssertionError("early edge-slot or projection mutation check failed")
        name = next((name for name in self.post_backward_updates[0]["all"]
                     if name.endswith("coordinate_codebook.projection.weight")), None)
        if name is None or not any(update["all"][name]["nonzero"] for update in self.post_backward_updates[:2]):
            raise AssertionError("early projection had no actual training gradient")
        if not any(update.get(name, 0.0) > 0 for update in self.update_deltas_by_step[:2]):
            raise AssertionError("early projection did not move at positive LR")


def run(config: Path, output: Path) -> dict[str, object]:
    with patch.object(qualify, "_LossQualificationProbe", EarlyEdgeProbe):
        return qualify.run(config, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)

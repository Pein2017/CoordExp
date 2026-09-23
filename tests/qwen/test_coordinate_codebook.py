from __future__ import annotations

from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

from src.qwen.coordinate_codebook import (
    COORDINATE_COUNT,
    EARLY_PATCH_EDGES,
    LATE_MASKED_CENTER,
    install_coordinate_codebook,
    load_coordinate_codebook,
    save_coordinate_codebook,
    _interpolate,
    _rms_normalize,
)


HIDDEN = 8
COORDINATE_IDS = tuple(range(10, 10 + COORDINATE_COUNT))
SELECTED_IDS = tuple(range(4)) + COORDINATE_IDS


class _OutputWrapper(nn.Module):
    def __init__(self, *, independent: bool = True, stale: bool = False) -> None:
        super().__init__()
        self.base = nn.Linear(HIDDEN, max(SELECTED_IDS) + 1, bias=False)
        self.base.weight.requires_grad_(False)
        delta = nn.Parameter(torch.linspace(-0.2, 0.2, len(SELECTED_IDS) * HIDDEN).reshape(len(SELECTED_IDS), HIDDEN))
        self.shared_embed_delta = delta if independent else nn.Parameter(delta.detach().clone())
        self.register_buffer("selected_token_ids", torch.tensor(SELECTED_IDS if not stale else SELECTED_IDS[:-1]))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.base(hidden) + hidden @ self.shared_embed_delta.new_zeros((HIDDEN, self.base.out_features))


class _InputWrapper(nn.Module):
    def __init__(self, delta: nn.Parameter) -> None:
        super().__init__()
        self.shared_embed_delta = delta


class _Merger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.linear.weight.requires_grad_(False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


class _PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = HIDDEN


class _FirstVisionBlock(GradientCheckpointingLayer):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.seen_inputs.append(hidden_states.detach().clone())
        return self.linear(hidden_states)


class _EarlyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_merge_size = 2
        self.patch_size = 16
        self.config = SimpleNamespace(hidden_size=HIDDEN)
        self.patch_embed = _PatchEmbed()
        self.blocks = nn.ModuleList([_FirstVisionBlock()])
        self.merger = _Merger()
        self.last_pre_block: torch.Tensor | None = None
        self.last_post_block: torch.Tensor | None = None

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del grid_thw
        positional = torch.arange(pixel_values.numel(), dtype=pixel_values.dtype, device=pixel_values.device)
        positional = positional.reshape_as(pixel_values) / 1000
        hidden_states = pixel_values + positional
        self.last_pre_block = hidden_states.detach().clone()
        hidden_states = self.blocks[0](hidden_states)
        self.last_post_block = hidden_states.detach().clone()
        return self.merger(hidden_states), [hidden_states[:1]]


class _Visual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.merger = _Merger()

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del pixel_values
        values = torch.arange(int(grid_thw[:, 1].mul(grid_thw[:, 2]).sum().item()) // 4 * HIDDEN, dtype=torch.float32).reshape(-1, HIDDEN) / 100
        return self.merger(values), [values[:1]]


class _Model(nn.Module):
    def __init__(
        self,
        *,
        independent: bool = True,
        stale: bool = False,
        visual: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self._output = _OutputWrapper(independent=independent, stale=stale)
        self.model = nn.Module()
        self.model.add_module("visual", visual if visual is not None else _Visual())
        input_delta = (
            self._output.shared_embed_delta
            if not independent
            else nn.Parameter(self._output.shared_embed_delta.detach().clone())
        )
        self._input = _InputWrapper(input_delta)

    def get_output_embeddings(self) -> nn.Module:
        return self._output

    def get_input_embeddings(self) -> nn.Module:
        return self._input

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        return self.model.visual(pixel_values, grid_thw=image_grid_thw)[0]


class _NestedModel(nn.Module):
    """Peft-like wrapper where the Qwen model is below an extra model layer."""

    def __init__(self, inner: _Model) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.add_module("base_model", inner)

    def get_output_embeddings(self) -> nn.Module:
        return self.model.base_model.get_output_embeddings()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.base_model.get_input_embeddings()


def _grid() -> torch.Tensor:
    return torch.tensor([[1, 4, 6], [1, 2, 4]], dtype=torch.long)


def test_rectangular_multiimage_hook_preserves_boundaries_and_sends_live_delta_gradient() -> None:
    model = _Model()
    codebook = install_coordinate_codebook(model, COORDINATE_IDS)
    grid = _grid()
    output = model(torch.empty(1), image_grid_thw=grid)
    assert output.shape == (6 + 2, HIDDEN)
    assert float(codebook.gain.detach()) == pytest.approx(0.05, rel=1e-5)
    loss = (output * torch.arange(1, output.numel() + 1, dtype=output.dtype).reshape_as(output)).sum()
    loss.backward()
    assert codebook.raw_gain.grad is not None and float(codebook.raw_gain.grad.abs()) > 0
    assert model.get_output_embeddings().shared_embed_delta.grad is not None


def test_nested_peft_like_model_resolves_the_unique_visual_merger() -> None:
    model = _NestedModel(_Model())
    codebook = install_coordinate_codebook(model, COORDINATE_IDS)
    output = codebook(torch.ones((8, HIDDEN)), _grid())
    assert output.shape == (8, HIDDEN)


def test_off_switch_is_exact_identity() -> None:
    model = _Model()
    baseline = model(torch.empty(1), image_grid_thw=_grid())
    codebook = install_coordinate_codebook(model, COORDINATE_IDS)
    grid = _grid()
    codebook.enabled = False
    output = model(torch.empty(1), image_grid_thw=grid)
    assert torch.equal(output, baseline)


def test_device_move_uses_nn_module_apply_and_frozen_inference_delta_is_valid() -> None:
    model = _Model()
    codebook = install_coordinate_codebook(model, COORDINATE_IDS)
    model.requires_grad_(False)
    model.to("cpu")
    output = model(torch.empty(1), image_grid_thw=_grid())
    assert output.shape == (8, HIDDEN)
    assert not codebook.raw_gain.requires_grad


def test_effective_output_rows_are_live_and_not_wrapper_weight_alone() -> None:
    model = _Model()
    codebook = install_coordinate_codebook(model, COORDINATE_IDS)
    with torch.no_grad():
        before = model(torch.empty(1), image_grid_thw=_grid()).clone()
        model.get_output_embeddings().shared_embed_delta.add_(1.0)
        after = model(torch.empty(1), image_grid_thw=_grid())
    assert not torch.equal(before, after)


@pytest.mark.parametrize("kwargs", [{"stale": True}, {"independent": False}])
def test_stale_or_nonindependent_output_payload_fails_closed(kwargs: dict[str, bool]) -> None:
    model = _Model(**kwargs)
    codebook = install_coordinate_codebook(model, COORDINATE_IDS)
    with pytest.raises((TypeError, ValueError)):
        model(torch.empty(1), image_grid_thw=_grid())


def test_wrong_grid_and_missing_grid_fail_closed() -> None:
    model = _Model()
    install_coordinate_codebook(model, COORDINATE_IDS)
    with pytest.raises(ValueError):
        model(torch.empty(1), image_grid_thw=torch.tensor([[1, 3, 4]]))
    with pytest.raises(ValueError):
        model.model.visual(torch.empty(1), grid_thw=None)


def test_save_reload_is_strict_and_restores_gain(tmp_path: Path) -> None:
    model = _Model()
    codebook = install_coordinate_codebook(model, COORDINATE_IDS)
    with torch.no_grad():
        codebook.raw_gain.add_(0.7)
    saved = save_coordinate_codebook(model, tmp_path)
    expected = codebook.raw_gain.detach().clone()
    with torch.no_grad():
        codebook.raw_gain.zero_()
    load_coordinate_codebook(model, tmp_path)
    torch.testing.assert_close(codebook.raw_gain, expected)

    (tmp_path / "coordinate_codebook.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        load_coordinate_codebook(model, tmp_path)
    assert saved["coordinate_count"] == COORDINATE_COUNT


def _set_monotone_coordinate_rows(model: _Model) -> None:
    values = torch.arange(1, COORDINATE_COUNT + 1, dtype=torch.float32) / COORDINATE_COUNT
    rows = torch.stack(
        (values, values.square() + 0.1, values.sqrt(), values * 2, values + 0.5, values * 3, values + 1, values * 4),
        dim=-1,
    )
    output = model.get_output_embeddings()
    with torch.no_grad():
        output.shared_embed_delta.zero_()
        output.base.weight[torch.tensor(COORDINATE_IDS)] = rows


def _expected_processor_positions(height: int, width: int) -> list[tuple[int, int]]:
    if (height, width) == (4, 6):
        return [
            (0, 0), (0, 1), (1, 0), (1, 1),
            (0, 2), (0, 3), (1, 2), (1, 3),
            (0, 4), (0, 5), (1, 4), (1, 5),
            (2, 0), (2, 1), (3, 0), (3, 1),
            (2, 2), (2, 3), (3, 2), (3, 3),
            (2, 4), (2, 5), (3, 4), (3, 5),
        ]
    if (height, width) == (2, 4):
        return [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    raise AssertionError("unexpected test grid")


def _expected_edges(grid: tuple[int, int], positions: list[tuple[int, int]]) -> torch.Tensor:
    height, width = grid
    return torch.tensor(
        [(x / width, (x + 1) / width, y / height, (y + 1) / height) for y, x in positions],
        dtype=torch.float32,
    )


def _early_model() -> _Model:
    model = _Model(visual=_EarlyVisual())
    _set_monotone_coordinate_rows(model)
    return model


def test_early_hook_uses_processor_patch_order_and_full_ordered_edge_slots() -> None:
    model = _early_model()
    codebook = install_coordinate_codebook(
        model, COORDINATE_IDS, mode=EARLY_PATCH_EDGES, projection_seed=77
    )
    assert codebook.projection is not None
    assert codebook.projection.in_features == 4 * HIDDEN
    assert codebook.projection.out_features == HIDDEN
    grid = torch.tensor([[1, 4, 6], [1, 2, 4]])
    expected = torch.cat(
        (_expected_edges((4, 6), _expected_processor_positions(4, 6)),
         _expected_edges((2, 4), _expected_processor_positions(2, 4))),
        dim=0,
    )
    actual = codebook.patch_edge_coordinates(grid)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bool((expected[:, [0, 2]] == 0).any())
    assert float(actual[:, 1].max()) == 1.0
    assert float(actual[:, 3].max()) == 1.0

    with torch.no_grad():
        codebook.projection.weight.zero_()
        for edge_slot in range(4):
            codebook.projection.weight[edge_slot, edge_slot * HIDDEN] = 1.0
    hidden = torch.ones((expected.shape[0], HIDDEN))
    injected = codebook.inject_early(hidden, grid, edge_coordinates=expected)
    rows, _ = codebook._effective_rows()
    edge_rows = _interpolate(rows, expected.reshape(-1))
    slots = _rms_normalize(edge_rows).reshape(hidden.shape[0], 4, HIDDEN).flatten(1)
    projected = slots @ codebook.projection.weight.detach().T
    expected_residual = 0.05 * torch.sqrt(hidden.square().mean(-1, keepdim=True)) * _rms_normalize(projected)
    torch.testing.assert_close(injected, hidden + expected_residual)
    swapped = expected[:, [1, 0, 2, 3]]
    swapped_injected = codebook.inject_early(hidden, grid, edge_coordinates=swapped)
    assert not torch.equal(injected, swapped_injected)
    assert codebook.architecture_metadata()["edge_order"] == ["x_left", "x_right", "y_top", "y_bottom"]
    pixels = torch.linspace(-1, 1, expected.shape[0] * HIDDEN).reshape_as(hidden)
    correct_forward = model(pixels, image_grid_thw=grid)
    with codebook.override_patch_edges(expected):
        identity_replay = model(pixels, image_grid_thw=grid)
    with codebook.override_patch_edges(swapped):
        permuted_forward = model(pixels, image_grid_thw=grid)
    torch.testing.assert_close(identity_replay, correct_forward, atol=2e-4, rtol=0)
    assert not torch.equal(permuted_forward, correct_forward)


def test_norm1000_interpolation_clips_inclusive_edges_to_rows_zero_and_999() -> None:
    rows = torch.arange(COORDINATE_COUNT, dtype=torch.float32).unsqueeze(-1)
    got = _interpolate(rows, torch.tensor([0.0, 0.5, 0.5005, 1.0]))[:, 0]
    torch.testing.assert_close(got, torch.tensor([0.0, 500.0, 500.5, 999.0]))


def test_early_projection_computes_in_fp32_and_casts_only_the_residual_for_bf16() -> None:
    model = _early_model()
    codebook = install_coordinate_codebook(model, COORDINATE_IDS, mode=EARLY_PATCH_EDGES)
    assert codebook.projection is not None and codebook.projection.weight.dtype == torch.float32
    hidden = torch.ones((32, HIDDEN), dtype=torch.bfloat16)
    grid = torch.tensor([[1, 4, 6], [1, 2, 4]])
    got = codebook.inject_early(hidden, grid)
    assert got.dtype == torch.bfloat16
    assert bool(torch.isfinite(got).all())


def test_early_hook_runs_after_positions_and_keeps_live_output_delta_gradients() -> None:
    model = _early_model()
    codebook = install_coordinate_codebook(model, COORDINATE_IDS, mode=EARLY_PATCH_EDGES)
    grid = torch.tensor([[1, 4, 6], [1, 2, 4]])
    pixels = torch.linspace(-1, 1, 32 * HIDDEN).reshape(32, HIDDEN)
    model(pixels, image_grid_thw=grid)
    visual = model.model.visual
    assert isinstance(visual, _EarlyVisual)
    first_block = visual.blocks[0]
    expected = codebook.inject_early(visual.last_pre_block, grid)
    torch.testing.assert_close(first_block.seen_inputs[-1], expected)
    assert not torch.equal(first_block.seen_inputs[-1], visual.last_pre_block)

    codebook.capture_injection_stats = True
    codebook.raw_gain.grad = None
    output = model(pixels, image_grid_thw=grid)
    weights = torch.linspace(0.1, 1.0, output.numel()).reshape_as(output)
    (output * weights).sum().backward()
    assert model.get_output_embeddings().shared_embed_delta.grad is not None
    assert float(model.get_output_embeddings().shared_embed_delta.grad.abs().sum()) > 0
    assert codebook.projection is not None and codebook.projection.weight.grad is not None
    assert float(codebook.projection.weight.grad.abs().sum()) > 0
    assert codebook.raw_gain.grad is not None and float(codebook.raw_gain.grad.abs()) > 0
    assert visual.merger.linear.weight.grad is None
    assert codebook.last_injection_stats is not None
    assert codebook.last_injection_stats["gain"] == pytest.approx(0.05, rel=1e-5)
    assert codebook.last_injection_stats["normalized_address_rms"] == pytest.approx(1.0, rel=1e-4)
    assert codebook.last_injection_stats["residual_rms"] == pytest.approx(
        codebook.last_injection_stats["gain"] * codebook.last_injection_stats["input_rms"], rel=1e-4
    )


def test_early_disabled_route_is_exact_and_projection_seed_preserves_torch_rng() -> None:
    model = _early_model()
    torch.manual_seed(701)
    rng_state = torch.random.get_rng_state()
    codebook = install_coordinate_codebook(
        model, COORDINATE_IDS, mode=EARLY_PATCH_EDGES, projection_seed=91
    )
    after_install = torch.rand(5)
    torch.random.set_rng_state(rng_state)
    expected_after_install = torch.rand(5)
    torch.testing.assert_close(after_install, expected_after_install)
    second = _early_model()
    second_codebook = install_coordinate_codebook(
        second, COORDINATE_IDS, mode=EARLY_PATCH_EDGES, projection_seed=91
    )
    assert codebook.projection is not None and second_codebook.projection is not None
    torch.testing.assert_close(codebook.projection.weight, second_codebook.projection.weight)

    pixels = torch.ones((32, HIDDEN))
    grid = torch.tensor([[1, 4, 6], [1, 2, 4]])
    baseline = model(pixels, image_grid_thw=grid)
    codebook.enabled = False
    disabled = model(pixels, image_grid_thw=grid)
    assert not torch.equal(disabled, baseline)
    visual = model.model.visual
    assert isinstance(visual, _EarlyVisual)
    codebook.enabled = True
    enabled = model(pixels, image_grid_thw=grid)
    codebook.enabled = False
    disabled = model(pixels, image_grid_thw=grid)
    assert visual.last_post_block is not None
    assert torch.equal(disabled, visual.merger(visual.last_post_block))
    assert not torch.equal(enabled, disabled)


def test_checkpoint_recompute_keeps_each_forward_grid_without_full_activation_cache() -> None:
    model = _early_model()
    codebook = install_coordinate_codebook(model, COORDINATE_IDS, mode=EARLY_PATCH_EDGES)
    first_block = model.model.visual.blocks[0]
    original_checkpoint = partial(checkpoint, use_reentrant=False)
    first_block.gradient_checkpointing = True
    first_block._gradient_checkpointing_func = original_checkpoint
    grid_a = torch.tensor([[1, 4, 6], [1, 2, 4]])
    grid_b = torch.tensor([[1, 8, 4]])
    pixels_a = torch.linspace(-0.5, 0.5, 32 * HIDDEN).reshape(32, HIDDEN).requires_grad_()
    pixels_b = torch.linspace(0.7, 1.2, 32 * HIDDEN).reshape(32, HIDDEN).requires_grad_()
    output_a = model(pixels_a, image_grid_thw=grid_a)
    visual = model.model.visual
    assert isinstance(visual, _EarlyVisual) and visual.last_pre_block is not None
    expected_a = codebook.inject_early(visual.last_pre_block, grid_a).detach()
    output_b = model(pixels_b, image_grid_thw=grid_b)
    assert visual.last_pre_block is not None
    expected_b = codebook.inject_early(visual.last_pre_block, grid_b).detach()
    assert codebook._grid is None
    (output_a.square().sum() + output_b.square().sum()).backward()
    replayed = first_block.seen_inputs[2:]
    assert len(replayed) >= 2
    assert any(torch.allclose(value, expected_a) for value in replayed)
    assert any(torch.allclose(value, expected_b) for value in replayed)
    assert first_block._gradient_checkpointing_func is codebook._checkpoint_wrapper
    codebook.remove_hooks()
    assert first_block._gradient_checkpointing_func is original_checkpoint


def test_old_late_checkpoint_schema_and_loader_meaning_remain_v1(tmp_path: Path) -> None:
    source = _Model()
    legacy = install_coordinate_codebook(source, COORDINATE_IDS)
    saved = save_coordinate_codebook(source, tmp_path)
    metadata = json.loads((tmp_path / "coordinate_codebook.json").read_text())
    assert metadata["schema"] == "coordexp-live-coordinate-codebook-v1"
    assert "mode" not in metadata and "projection_seed" not in metadata
    assert metadata["visual_injection"] == "main_vision_merger_output_only"
    from safetensors.torch import load_file
    assert set(load_file(str(tmp_path / "coordinate_codebook.safetensors"))) == {"raw_gain"}

    target = _Model()
    restored = install_coordinate_codebook(target, COORDINATE_IDS)
    load_coordinate_codebook(target, tmp_path)
    assert restored.mode == LATE_MASKED_CENTER
    torch.testing.assert_close(restored.raw_gain, legacy.raw_gain)
    assert saved["schema"] == metadata["schema"]

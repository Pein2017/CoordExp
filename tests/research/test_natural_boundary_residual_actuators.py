"""CPU contract tests for returned block-23 natural-boundary actuators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.research import natural_boundary_residual_actuators as residual
from scripts.research.run_natural_boundary_routing_history_probe import (
    NativeRowContract,
    build_event_context,
    build_residual_request,
    release_natural_event,
)


class FakeQwenDecoderBlock(torch.nn.Module):
    _coordexp_decoder_layer = True

    def __init__(self, width: int, index: int, *, return_tuple: bool = True) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.arange(width, dtype=torch.float32) + index + 1.0)
        self.index = index
        self.return_tuple = return_tuple
        self.marker = object()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, object]:
        value = hidden + self.bias
        if self.return_tuple:
            return value, self.marker
        return value


class FakeQwen(torch.nn.Module):
    def __init__(self, *, width: int = 4, layer_count: int = 24, return_tuple: bool = True) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList(
            FakeQwenDecoderBlock(width, index, return_tuple=return_tuple) for index in range(layer_count)
        )
        self.last_block_extra: object | None = None

    def forward(self, *, input_ids: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4).clone()
        for layer in self.model.language_model.layers:
            output = layer(hidden)
            if isinstance(output, tuple):
                hidden, self.last_block_extra = output
            else:
                hidden = output
        return hidden


class RaisingQwen(FakeQwen):
    def forward(self, *, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4).clone()
        for index, layer in enumerate(self.model.language_model.layers):
            output = layer(hidden)
            hidden = output[0] if isinstance(output, tuple) else output
            if index == residual.BLOCK_INDEX:
                raise RuntimeError("synthetic scalar failure")
        return hidden


class FakeQwenLogits(FakeQwen):
    """Fake multimodal-shaped Qwen whose scalar logits choose native STOP."""

    def __init__(self) -> None:
        super().__init__(width=64)

    def forward(self, *, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 64).clone()
        for layer in self.model.language_model.layers:
            output = layer(hidden)
            if isinstance(output, tuple):
                hidden, self.last_block_extra = output
            else:
                hidden = output
        logits = torch.full((1, input_ids.shape[1], 64), -100.0)
        logits[..., 0] = 100.0
        return SimpleNamespace(logits=logits)


def _input_ids(length: int = 6) -> torch.Tensor:
    return torch.arange(10, 10 + length, dtype=torch.long).reshape(1, -1)


def _run(model: FakeQwen, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids, use_cache=False)


def test_resolver_finds_unique_qwen_block_23_and_records_alias() -> None:
    model = FakeQwen()
    block, receipt = residual.resolve_qwen_block23(model)
    assert block is model.model.language_model.layers[23]
    assert receipt["layer_index"] == 23
    assert receipt["module_path"] == "model.language_model.layers[23]"


def test_n01_identity_parity_and_tuple_auxiliary_preservation() -> None:
    ids = _input_ids()
    native_model = FakeQwen()
    native = _run(native_model, ids)

    model = FakeQwen()
    with residual.ResidualHookContext(model, "N01", positions=(2,), input_ids=ids) as hook:
        replay = _run(model, ids)
    receipt = hook.receipt()
    assert torch.equal(native, replay)
    assert receipt["target_max_abs_delta"] == 0.0
    assert receipt["non_target_max_abs_delta"] == 0.0
    assert receipt["target_positions_exact"] is True
    assert receipt["hook_clean"] is True
    assert receipt["install_count"] == receipt["cleanup_count"] == 1
    assert receipt["hook_call_count"] == receipt["hook_applied_count"] == 1
    assert model.last_block_extra is model.model.language_model.layers[-1].marker


def test_n10_changes_only_terminal_target_and_matches_mean_norm() -> None:
    ids = _input_ids()
    model = FakeQwen()
    native = _run(model, ids)

    model = FakeQwen()
    with residual.ResidualHookContext(
        model,
        "N10",
        positions=(3,),
        background_positions=(1, 2),
        input_ids=ids,
    ) as hook:
        muted = _run(model, ids)
    receipt = hook.receipt()
    delta = (muted - native).abs()
    assert float(delta[:, 3, :].max()) > 0.0
    assert torch.count_nonzero(delta[:, [0, 1, 2, 4, 5], :]).item() == 0
    assert receipt["target_positions_exact"] is True
    assert receipt["non_target_max_abs_delta"] == 0.0
    assert receipt["replacement_finite"] is True
    assert abs(receipt["post_target_mean_norm"] - receipt["pre_target_mean_norm"]) <= 1e-4
    assert receipt["norm_match_abs_delta"] <= 1e-4


def test_n20_changes_only_contiguous_latest_row_scope() -> None:
    ids = _input_ids()
    native_model = FakeQwen()
    native = _run(native_model, ids)
    model = FakeQwen()
    with residual.ResidualHookContext(model, "N20", positions=(1, 2, 3), input_ids=ids) as hook:
        muted = _run(model, ids)
    receipt = hook.receipt()
    delta = (muted - native).abs()
    assert torch.count_nonzero(delta[:, [1, 2, 3], :]).item() > 0
    assert torch.count_nonzero(delta[:, [0, 4, 5], :]).item() == 0
    assert receipt["positions"] == [1, 2, 3]
    assert receipt["target_positions_exact"] is True
    assert receipt["non_target_max_abs_delta"] == 0.0
    assert len(receipt["pre_target_norms"]) == 3
    assert len(receipt["post_target_norms"]) == 3
    assert abs(receipt["post_target_mean_norm"] - receipt["pre_target_mean_norm"]) <= 1e-4


def test_callback_returns_fresh_per_scalar_context_and_rejects_stale_replacement() -> None:
    ids = _input_ids()
    callback = residual.make_residual_actuator_callback()
    request = SimpleNamespace(arm_id="N10", positions=(3,), persistent=False, replacement=None)
    event_context = SimpleNamespace(
        prompt_token_ids=(10,),
        exact_history_token_ids=(11, 12, 13),
        latest_history_row_token_ids=(11, 12, 13),
        history_prefix_width=1,
    )
    model = FakeQwen()
    context = callback(model, context=event_context, request=request, input_ids=ids, step=7, row_index=2)
    assert isinstance(context, residual.ResidualHookContext)
    with context:
        context.model(input_ids=ids, use_cache=False)
    assert context.receipt()["step"] == 7
    assert context.receipt()["row_index"] == 2
    assert context.receipt()["background_positions"] == [1, 2]
    assert context.receipt()["background_position_source"] == "latest_row_without_terminal"
    assert callback.receipt() == context.receipt()

    stale = SimpleNamespace(
        arm_id="N10",
        positions=(3,),
        persistent=False,
        replacement=torch.zeros((1, 4)),
    )
    with pytest.raises(residual.TechnicalInvalid, match="stale replacement"):
        callback(FakeQwen(), context=event_context, request=stale, input_ids=ids)


def test_exception_path_removes_hook_exactly_once() -> None:
    ids = _input_ids()
    model = RaisingQwen()
    hook = residual.ResidualHookContext(model, "N20", positions=(1, 2), input_ids=ids)
    with pytest.raises(RuntimeError, match="synthetic scalar failure"):
        with hook:
            model(input_ids=ids, use_cache=False)
    receipt = hook.receipt()
    assert receipt["hook_clean"] is True
    assert receipt["install_count"] == 1
    assert receipt["cleanup_count"] == 1
    assert receipt["exception_seen"] is True


def test_layout_and_position_mismatch_fail_closed_and_cleanup() -> None:
    ids = _input_ids()
    model = FakeQwen()
    with pytest.raises(residual.TechnicalInvalid, match="outside expected"):
        residual.ResidualHookContext(model, "N10", positions=(7,), background_positions=(5, 6), input_ids=ids)

    with pytest.raises(residual.TechnicalInvalid, match="contiguous"):
        residual.ResidualHookContext(FakeQwen(), "N20", positions=(1, 3), input_ids=ids)


def test_native_n00_is_a_clean_no_hook_context() -> None:
    hook = residual.ResidualHookContext(object(), "N00")
    with hook:
        pass
    receipt = hook.receipt()
    assert receipt["hook_call_count"] == 0
    assert receipt["hook_clean"] is True
    assert receipt["install_count"] == receipt["cleanup_count"] == 1


def test_natural_runner_persists_runtime_actuation_receipt_after_context_exit() -> None:
    contract = NativeRowContract.closed(
        opener_token_id=1,
        object_ref_end_token_id=2,
        box_start_token_id=3,
        box_end_token_id=9,
        coordinate_token_start_id=10,
        coordinate_bin_count=100,
        stop_token_id=0,
    )
    context = build_event_context(
        event_id="fake-runtime-event",
        prompt_token_ids=(10,),
        exact_history_token_ids=(11, 12, 13),
        latest_history_row_token_ids=(11, 12, 13),
        history_prefix_width=1,
        row_contract=contract,
        max_rows=1,
    )
    request = build_residual_request("N10", positions=(3,))
    callback = residual.make_residual_actuator_callback()
    result = release_natural_event(
        FakeQwenLogits(),
        context,
        residual_arm="N10",
        residual_request=request,
        residual_actuator=callback,
    )
    assert result["terminal_reason"] == "native_stop"
    scalar = result["scalar_receipts"][0]
    actuation = scalar["residual"]["actuation_receipt"]
    assert actuation["layer"]["layer_index"] == residual.BLOCK_INDEX
    assert actuation["background_positions"] == [1, 2]
    assert actuation["hook_applied_count"] == 1
    assert actuation["hook_removed"] is True
    assert actuation["non_target_max_abs_delta"] == 0.0
    assert actuation["pre_finite"] is True
    assert actuation["post_finite"] is True
    # The fake uses a 64-wide residual with norm around 9k; one float32 ULP
    # is expected after the runtime replacement cast.
    assert actuation["norm_match_abs_delta"] <= 1e-3

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from scripts.research.run_static_dynamic_gradient_path_audit import (
    AuditBatch,
    ForwardCapture,
    GradientSource,
    RuntimeContract,
    StateProvenance,
    capture_native_forward,
    run_static_dynamic_gradient_path_audit,
    sha256_int_sequence,
    sha256_json,
    sha256_position_ids,
)


VOCAB_SIZE = 128
OBJECT_START = 100
OBJECT_END = 101
BOX_START = 102
BOX_END = 103
COMMIT = 104
COORDINATES = (10, 11, 12, 13)
PREFIX = (7, 8)
IMAGE_POSITIONS = (2, 3)
BACKGROUND_POSITIONS = (4, 5)
TERMINAL_POSITIONS = (6,)
ROW_SPAN_POSITIONS = (7, 8)
NATURAL_HISTORY_HASH = sha256_json({"history": "native-natural"})
STATE_NAMES = (
    "image_residual",
    "matched_background",
    "latest_terminal_carrier",
    "latest_row_span",
)


def _row(*, commit: bool) -> list[int]:
    row = [OBJECT_START, 41, OBJECT_END, BOX_START, *COORDINATES, BOX_END]
    return row + ([COMMIT] if commit else [])


def _runtime(
    mode: str = "closed", position_ids: Tensor | None = None
) -> RuntimeContract:
    position_ids = position_ids if position_ids is not None else torch.tensor([[0, 1]], dtype=torch.long)
    return RuntimeContract(
        wrapper_mode=mode,
        prefix_token_ids=PREFIX,
        expected_prefix_sha256=sha256_int_sequence(PREFIX),
        expected_position_ids_sha256=sha256_position_ids(position_ids),
        object_ref_start_token_id=OBJECT_START,
        object_ref_end_token_id=OBJECT_END,
        box_start_token_id=BOX_START,
        box_end_token_id=BOX_END,
        coordinate_token_ids=COORDINATES,
        commit_token_id=COMMIT,
        block23_module_identity="tiny.block23",
        checkpoint_identity="tiny-checkpoint",
        config_identity="tiny-config",
        wrapper_identity=f"native-{mode}",
        expected_mrope_sha256=sha256_position_ids(position_ids),
        expected_image_positions_sha256=sha256_int_sequence(IMAGE_POSITIONS),
        expected_background_positions_sha256=sha256_int_sequence(BACKGROUND_POSITIONS),
        expected_terminal_positions_sha256=sha256_int_sequence(TERMINAL_POSITIONS),
        expected_row_span_positions_sha256=sha256_int_sequence(ROW_SPAN_POSITIONS),
        expected_natural_history_sha256=NATURAL_HISTORY_HASH,
    )


class TinyBlock23(nn.Module):
    """A real module boundary whose hook returns the four retained states."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, states: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        # Keep exact state object identity while executing a real module
        # parameterized boundary.  ``_block23_gate`` makes this a genuine
        # module call without replacing the retained tensors with clones.
        return {**states, "_block23_gate": self.scale * 0.0}


class TinyNativeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tiny = nn.Module()
        self.tiny.block23 = TinyBlock23()
        self.lm_head = nn.Linear(1, VOCAB_SIZE, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.linspace(-1.0, 1.0, VOCAB_SIZE).reshape(-1, 1))

    def _row_logits(self, scalar: Tensor, length: int) -> Tensor:
        return self.lm_head(scalar.reshape(1, 1).expand(length, 1))

    def outputs_for_batch(self, batch: AuditBatch) -> Mapping[str, Tensor]:
        hooked = self.tiny.block23(
            {
                "image_residual": batch.image_residual,
                "matched_background": batch.matched_background,
                "latest_terminal_carrier": batch.latest_terminal_carrier,
                "latest_row_span": batch.latest_row_span,
            }
        )
        states = {name: hooked[name] for name in STATE_NAMES}
        static = states["image_residual"].mean() + states["matched_background"].mean()
        dynamic = states["latest_terminal_carrier"].mean() + states["latest_row_span"].mean()
        return {
            "target_logits": self._row_logits(static, len(batch.target_row_token_ids)),
            "uncovered_b_logits": self._row_logits(
                dynamic + 0.2, len(batch.uncovered_b_row_token_ids)
            ),
            "covered_a_logits": self._row_logits(
                dynamic - 0.2, len(batch.covered_a_row_token_ids)
            ),
        }


class HeadOnlyModel(TinyNativeModel):
    def outputs_for_batch(self, batch: AuditBatch) -> Mapping[str, Tensor]:
        # Still executes block23, but the logits intentionally ignore its
        # outputs.  The audit must identify this as an LM-head-only route.
        self.tiny.block23(
            {
                "image_residual": batch.image_residual,
                "matched_background": batch.matched_background,
                "latest_terminal_carrier": batch.latest_terminal_carrier,
                "latest_row_span": batch.latest_row_span,
            }
        )
        constant = torch.ones(1, 1)
        return {
            "target_logits": self._row_logits(constant, len(batch.target_row_token_ids)),
            "uncovered_b_logits": self._row_logits(
                constant, len(batch.uncovered_b_row_token_ids)
            ),
            "covered_a_logits": self._row_logits(constant, len(batch.covered_a_row_token_ids)),
        }


class MutatingModel(TinyNativeModel):
    def outputs_for_batch(self, batch: AuditBatch) -> Mapping[str, Tensor]:
        with torch.no_grad():
            self.lm_head.weight.add_(1.0)
        return super().outputs_for_batch(batch)


def _batch(*, commit: bool = False, requires_grad: bool = True) -> AuditBatch:
    def state(value: list[list[float]]) -> Tensor:
        # The extra multiply makes these non-leaf tensors while keeping a real
        # differentiable path for the captured block output.
        return torch.tensor(value, dtype=torch.float32, requires_grad=requires_grad) * 1.0

    row = _row(commit=commit)
    return AuditBatch(
        input_ids=torch.tensor([PREFIX], dtype=torch.long),
        position_ids=torch.tensor([[0, 1]], dtype=torch.long),
        target_row_token_ids=row,
        uncovered_b_row_token_ids=row,
        covered_a_row_token_ids=row,
        image_residual=state([[0.2, 0.3]]),
        matched_background=state([[0.4, 0.1]]),
        latest_terminal_carrier=state([[0.5, 0.7]]),
        latest_row_span=state([[0.9, 0.6]]),
        grammar_token_ids=(1, 2),
        stop_token_ids=(BOX_END,),
        invalid_token_ids=(127,),
    )


def _provenance_template() -> dict[str, StateProvenance]:
    return {
        "image_residual": StateProvenance(
            role="image_span_b_exclusive",
            positions=IMAGE_POSITIONS,
            positions_sha256=sha256_int_sequence(IMAGE_POSITIONS),
            span_provenance="b-exclusive-image-span",
            history_sha256=sha256_json({"history": "static"}),
            natural_history=False,
            forward_id="pending",
            model=nn.Identity(),
            block23_module=nn.Identity(),
            block23_module_name="pending",
            input_ids_sha256="pending",
            position_ids_sha256="pending",
            mrope_sha256="pending",
        ),
        "matched_background": StateProvenance(
            role="background_control",
            positions=BACKGROUND_POSITIONS,
            positions_sha256=sha256_int_sequence(BACKGROUND_POSITIONS),
            span_provenance="background-control",
            history_sha256=sha256_json({"history": "static"}),
            natural_history=False,
            forward_id="pending",
            model=nn.Identity(),
            block23_module=nn.Identity(),
            block23_module_name="pending",
            input_ids_sha256="pending",
            position_ids_sha256="pending",
            mrope_sha256="pending",
        ),
        "latest_terminal_carrier": StateProvenance(
            role="latest_terminal_natural_history",
            positions=TERMINAL_POSITIONS,
            positions_sha256=sha256_int_sequence(TERMINAL_POSITIONS),
            span_provenance="natural-history-terminal",
            history_sha256=NATURAL_HISTORY_HASH,
            natural_history=True,
            forward_id="pending",
            model=nn.Identity(),
            block23_module=nn.Identity(),
            block23_module_name="pending",
            input_ids_sha256="pending",
            position_ids_sha256="pending",
            mrope_sha256="pending",
        ),
        "latest_row_span": StateProvenance(
            role="latest_row_span_natural_history",
            positions=ROW_SPAN_POSITIONS,
            positions_sha256=sha256_int_sequence(ROW_SPAN_POSITIONS),
            span_provenance="natural-history-row-span",
            history_sha256=NATURAL_HISTORY_HASH,
            natural_history=True,
            forward_id="pending",
            model=nn.Identity(),
            block23_module=nn.Identity(),
            block23_module_name="pending",
            input_ids_sha256="pending",
            position_ids_sha256="pending",
            mrope_sha256="pending",
        ),
    }


def _capture(model: TinyNativeModel, batch: AuditBatch, runtime: RuntimeContract) -> ForwardCapture:
    return capture_native_forward(
        model=model,
        block23_module=model.tiny.block23,
        block23_module_name="tiny.block23",
        input_ids=batch.input_ids,
        position_ids=batch.position_ids,
        forward_call=lambda: model.outputs_for_batch(batch),
        state_selector=lambda output: output,
        state_provenance=_provenance_template(),
        checkpoint_identity=runtime.checkpoint_identity,
        config_identity=runtime.config_identity,
        wrapper_identity=runtime.wrapper_identity,
        mrope_sha256=runtime.expected_mrope_sha256,
    )


def _run(
    *,
    model: TinyNativeModel | None = None,
    runtime: RuntimeContract | None = None,
    batch: AuditBatch | None = None,
    capture_transform=None,
) -> dict[str, object]:
    model = model or TinyNativeModel()
    runtime = runtime or _runtime()
    batch = batch or _batch(commit=runtime.wrapper_mode == "commit")

    def forward(current: AuditBatch) -> ForwardCapture:
        capture = _capture(model, current, runtime)
        return capture_transform(capture) if capture_transform else capture

    return run_static_dynamic_gradient_path_audit(
        model=model,
        runtime=runtime,
        batch=batch,
        forward_fn=forward,
    )


@pytest.mark.parametrize("mode", ["closed", "commit"])
def test_frozen_gradient_audit_reports_all_three_objectives(mode: str) -> None:
    model = TinyNativeModel()
    batch = _batch(commit=mode == "commit")
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    receipt = _run(model=model, runtime=_runtime(mode), batch=batch)

    assert receipt["status"] == "valid"
    assert receipt["path_checks"]["optimizer_used"] is False
    assert receipt["path_checks"]["lm_head_only_path"] is False
    assert receipt["path_checks"]["block23_layer_index"] == 23
    capture = receipt["path_checks"]["forward_capture"]
    assert capture["block23_module_name"] == "tiny.block23"
    assert capture["hook_call_count"] == 1
    assert capture["hook_cleaned"] is True
    objectives = receipt["objectives"]
    assert set(objectives) == {
        "target_b_complete_row_nll",
        "uncovered_b_vs_covered_a_margin_loss",
        "fixed_sum_coupled",
    }
    assert objectives["fixed_sum_coupled"]["terms"] == {
        "target_b_complete_row_nll_weight": 1.0,
        "uncovered_b_vs_covered_a_margin_loss_weight": 1.0,
    }
    required_by_objective = {
        "target_b_complete_row_nll": {"image_residual", "matched_background"},
        "uncovered_b_vs_covered_a_margin_loss": {
            "latest_terminal_carrier",
            "latest_row_span",
        },
        "fixed_sum_coupled": {
            "image_residual",
            "matched_background",
            "latest_terminal_carrier",
            "latest_row_span",
        },
    }
    for objective_name, objective in objectives.items():
        for name in required_by_objective[objective_name]:
            gradient = objective["gradients"][name]
            assert gradient["present"] is True
            assert gradient["finite"] is True
        assert gradient["norm"] > 0.0
    assert receipt["grammar_stop_invalid_mass"]["grammar"]["status"] == "reported"
    assert receipt["grammar_stop_invalid_mass"]["stop"]["status"] == "reported"
    assert receipt["grammar_stop_invalid_mass"]["invalid"]["status"] == "reported"
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_post_forward_views_use_forward_participating_full_block_gradient_source() -> None:
    class FullBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, hidden: Tensor) -> Tensor:
            return hidden * self.scale

    class FullOutputModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tiny = nn.Module()
            self.tiny.block23 = FullBlock()
            self.seed = nn.Parameter(torch.arange(18, dtype=torch.float32).reshape(1, 9, 2) / 10.0)
            self.lm_head = nn.Linear(1, VOCAB_SIZE, bias=False)

        def outputs(self, row_length: int) -> Mapping[str, Tensor]:
            hidden = self.tiny.block23(self.seed * 1.0)

            def logits(value: Tensor) -> Tensor:
                return self.lm_head(value.reshape(1, 1).expand(row_length, 1))

            static_value = hidden[:, 2:4].mean() + hidden[:, 4:6].mean()
            dynamic_value = hidden[:, 6:7].mean() + hidden[:, 7:9].mean()
            return {
                "target_logits": logits(static_value),
                "uncovered_b_logits": logits(dynamic_value + 0.2),
                "covered_a_logits": logits(dynamic_value - 0.2),
            }

    model = FullOutputModel()
    row = _row(commit=False)
    input_ids = torch.tensor([PREFIX], dtype=torch.long)
    position_ids = torch.tensor([[0, 1]], dtype=torch.long)

    def select_states(hidden: Tensor) -> Mapping[str, Tensor]:
        # These views are deliberately created after the forward completed.
        return {
            "image_residual": hidden[0, 2:4, :],
            "matched_background": hidden[0, 4:6, :],
            "latest_terminal_carrier": hidden[0, 6:7, :],
            "latest_row_span": hidden[0, 7:9, :],
        }

    position_map = {
        "image_residual": IMAGE_POSITIONS,
        "matched_background": BACKGROUND_POSITIONS,
        "latest_terminal_carrier": TERMINAL_POSITIONS,
        "latest_row_span": ROW_SPAN_POSITIONS,
    }
    capture = capture_native_forward(
        model=model,
        block23_module=model.tiny.block23,
        block23_module_name="tiny.block23",
        input_ids=input_ids,
        position_ids=position_ids,
        forward_call=lambda: model.outputs(len(row)),
        state_selector=select_states,
        state_provenance=_provenance_template(),
        checkpoint_identity="tiny-checkpoint",
        config_identity="tiny-config",
        wrapper_identity="native-closed",
        mrope_sha256=sha256_position_ids(position_ids),
        gradient_source_selector=lambda outputs: {
            name: (GradientSource(outputs[0], positions, "candidate"),)
            for name, positions in position_map.items()
        },
    )
    posthoc_gradient = torch.autograd.grad(
        capture.outputs["target_logits"].sum(),
        capture.captured_states["image_residual"],
        allow_unused=True,
        retain_graph=True,
    )[0]
    assert posthoc_gradient is None
    batch = AuditBatch(
        input_ids=input_ids,
        position_ids=position_ids,
        target_row_token_ids=row,
        uncovered_b_row_token_ids=row,
        covered_a_row_token_ids=row,
        image_residual=capture.captured_states["image_residual"],
        matched_background=capture.captured_states["matched_background"],
        latest_terminal_carrier=capture.captured_states["latest_terminal_carrier"],
        latest_row_span=capture.captured_states["latest_row_span"],
        grammar_token_ids=(1, 2),
        stop_token_ids=(BOX_END,),
        invalid_token_ids=(127,),
    )
    receipt = run_static_dynamic_gradient_path_audit(
        model=model,
        runtime=_runtime(),
        batch=batch,
        forward_fn=lambda _batch: capture,
    )
    assert receipt["status"] == "valid"
    for objective in receipt["objectives"].values():
        for gradient_record in objective["gradients"].values():
            assert gradient_record["present"] is True
            assert gradient_record["finite"] is True


def test_arbitrary_leaf_state_is_technical_invalid() -> None:
    receipt = _run(batch=_batch(requires_grad=True), capture_transform=None)
    # Replace the capture's retained tensors by leaves while leaving the hook
    # output untouched.  Both state identity and non-leaf checks must fail.
    batch = _batch()
    batch.image_residual = batch.image_residual.detach().requires_grad_()
    receipt = _run(batch=batch)
    assert receipt["status"] == "technical_invalid"
    assert any("captured_state_not_differentiable:image_residual" in reason for reason in receipt["invalid_reasons"])


def test_lm_head_only_path_is_technical_invalid() -> None:
    receipt = _run(model=HeadOnlyModel())
    assert receipt["status"] == "technical_invalid"
    assert receipt["path_checks"]["lm_head_only_path"] is True
    assert any(reason.startswith("lm_head_only_path:") for reason in receipt["invalid_reasons"])


def test_wrong_model_and_module_fail_closed() -> None:
    model = TinyNativeModel()
    other = TinyNativeModel()
    batch = _batch()
    stale = _capture(model, batch, _runtime())
    receipt = run_static_dynamic_gradient_path_audit(
        model=other,
        runtime=_runtime(),
        batch=batch,
        forward_fn=lambda _batch: stale,
    )
    assert receipt["status"] == "technical_invalid"
    assert "capture_model_identity_mismatch" in receipt["invalid_reasons"]
    assert "block23_module_identity_mismatch" in receipt["invalid_reasons"]


def test_different_forward_and_stale_prefix_fail_closed() -> None:
    model = TinyNativeModel()
    batch = _batch()
    stale_batch = _batch()
    stale_batch.input_ids = torch.tensor([[9, 8]], dtype=torch.long)
    stale_capture = _capture(model, stale_batch, _runtime())
    receipt = run_static_dynamic_gradient_path_audit(
        model=model,
        runtime=_runtime(),
        batch=batch,
        forward_fn=lambda _batch: stale_capture,
    )
    assert receipt["status"] == "technical_invalid"
    assert "native_input_ids_reference_mismatch" in receipt["invalid_reasons"]
    assert "native_prefix_identity_mismatch" in receipt["invalid_reasons"]


def test_detached_clone_external_state_fails_closed() -> None:
    model = TinyNativeModel()
    batch = _batch()

    def clone(capture: ForwardCapture) -> ForwardCapture:
        cloned = dict(capture.captured_states)
        cloned["image_residual"] = cloned["image_residual"].detach().clone().requires_grad_()
        return replace(capture, captured_states=cloned)

    receipt = _run(model=model, batch=batch, capture_transform=clone)
    assert receipt["status"] == "technical_invalid"
    assert "captured_state_identity_mismatch:image_residual" in receipt["invalid_reasons"]
    assert "captured_state_not_in_block23_hook_output:image_residual" in receipt["invalid_reasons"]


def test_hook_not_fired_or_not_cleaned_fail_closed() -> None:
    receipt = _run(capture_transform=lambda capture: replace(capture, hook_call_count=0))
    assert receipt["status"] == "technical_invalid"
    assert "block23_hook_call_count_mismatch" in receipt["invalid_reasons"]

    receipt = _run(capture_transform=lambda capture: replace(capture, hook_cleaned=False))
    assert receipt["status"] == "technical_invalid"
    assert "block23_hook_cleanup_missing" in receipt["invalid_reasons"]


def test_parameter_mutation_is_technical_invalid() -> None:
    receipt = _run(model=MutatingModel())
    assert receipt["status"] == "technical_invalid"
    assert "model_parameter_mutated" in receipt["invalid_reasons"]


def test_wrapper_and_position_identity_fail_closed() -> None:
    commit_rows = _batch(commit=True)
    closed_receipt = _run(batch=commit_rows)
    assert closed_receipt["status"] == "technical_invalid"
    assert "runtime_contract_invalid" in closed_receipt["invalid_reasons"]

    bad_position = _batch()
    bad_position.position_ids = torch.tensor([[1, 0]], dtype=torch.long)
    position_receipt = _run(batch=bad_position)
    assert position_receipt["status"] == "technical_invalid"
    assert "runtime_contract_invalid" in position_receipt["invalid_reasons"]


def test_qwen_mrope_position_shape_and_hash_are_bound() -> None:
    batch = _batch()
    batch.position_ids = torch.tensor(
        [[[0, 1]], [[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long
    )
    runtime = _runtime(position_ids=batch.position_ids)
    receipt = _run(batch=batch, runtime=runtime)
    assert receipt["status"] == "valid"


def test_invalid_and_missing_mass_contracts_are_not_silent() -> None:
    batch = _batch()
    batch.grammar_token_ids = (VOCAB_SIZE,)
    receipt = _run(batch=batch)
    assert receipt["status"] == "technical_invalid"
    assert "grammar_stop_invalid_mass_invalid" in receipt["invalid_reasons"]

    batch = _batch()
    batch.stop_token_ids = ()
    receipt = _run(batch=batch)
    assert receipt["status"] == "technical_invalid"
    assert "missing_stop_mass_contract" in receipt["invalid_reasons"]


def test_multiple_non_target_owner_effects_are_reported_from_same_hook_forward() -> None:
    model = TinyNativeModel()
    batch = _batch()
    # These are graph-preserving views into the exact image state retained by
    # the native block hook; no detached target-gradient inference is allowed.
    owner_one = batch.image_residual[:, :1]
    owner_two = batch.image_residual[:, 1:]
    batch.non_target_owner_states = {
        "gt:img:3": owner_one,
        "gt:img:4": owner_two,
    }
    batch.non_target_owner_region_receipts = {
        "gt:img:3": {"absolute_positions": [9], "shared_core_excluded": True},
        "gt:img:4": {"absolute_positions": [10], "shared_core_excluded": True},
    }

    def capture_with_owners(current: AuditBatch) -> ForwardCapture:
        capture = _capture(model, current, _runtime())
        states = dict(capture.captured_states)
        states.update(
            {
                "non_target_owner:gt:img:3": current.non_target_owner_states["gt:img:3"],
                "non_target_owner:gt:img:4": current.non_target_owner_states["gt:img:4"],
            }
        )
        return replace(capture, captured_states=states)

    receipt = run_static_dynamic_gradient_path_audit(
        model=model,
        runtime=_runtime(),
        batch=batch,
        forward_fn=capture_with_owners,
    )
    assert receipt["status"] == "valid"
    effects = receipt["non_target_owner_effects"]
    assert effects["target_b_complete_row_nll"]["status"] == "measured"
    assert set(effects["target_b_complete_row_nll"]["owners"]) == {"gt:img:3", "gt:img:4"}
    assert effects["fixed_sum_coupled"]["owners"]["gt:img:3"]["region"]["shared_core_excluded"] is True


def test_non_target_state_missing_from_capture_is_technical_invalid() -> None:
    model = TinyNativeModel()
    batch = _batch()
    batch.non_target_owner_states = {"gt:img:3": batch.image_residual[:, :1]}
    batch.non_target_owner_region_receipts = {"gt:img:3": {"absolute_positions": [9]}}
    receipt = _run(model=model, batch=batch)
    assert receipt["status"] == "technical_invalid"
    assert any("non_target_state_identity_mismatch" in reason for reason in receipt["invalid_reasons"])

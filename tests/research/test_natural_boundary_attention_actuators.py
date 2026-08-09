"""CPU contract tests for natural-boundary attention actuators."""

from __future__ import annotations

import pytest
import torch

from scripts.research import natural_boundary_attention_actuators as actuators


def test_k00_k01_are_native_and_explicit_real_four_d_masks() -> None:
    k00 = actuators.build_k00(sequence_length=6)
    assert k00.attention_mask is None
    assert k00.receipt()["status"] == "ready"

    k01 = actuators.build_k01(sequence_length=6)
    assert k01.attention_mask is not None
    assert tuple(k01.attention_mask.shape) == (1, 1, 6, 6)
    assert k01.attention_mask.dtype == torch.bool
    assert bool(k01.attention_mask[0, 0, 4, 4])
    assert not bool(k01.attention_mask[0, 0, 3, 4])
    assert k01.receipt()["changed_cell_count"] == 0


def test_hard_k10_k11_k12_k13_scope_and_not_applicable() -> None:
    kwargs = {
        "sequence_length": 10,
        "image_key_positions": (1, 2, 3, 4, 5, 6, 7),
        "query_position": 9,
    }
    k10 = actuators.build_k10(b_exclusive_positions=(4, 5), **kwargs)
    assert k10.receipt()["selected_positions"] == [4, 5]
    assert not bool(k10.attention_mask[0, 0, 9, 1])
    assert bool(k10.attention_mask[0, 0, 9, 4])
    assert k10.receipt()["offscope_changed_cell_count"] == 0
    assert k10.receipt()["future_changed_cell_count"] == 0

    k11 = actuators.build_k11(a_exclusive_positions=(1, 2), **kwargs)
    assert bool(k11.attention_mask[0, 0, 9, 3])
    assert not bool(k11.attention_mask[0, 0, 9, 1])

    k12 = actuators.build_k12(
        b_exclusive_positions=(4, 5),
        background_positions=(9, 8, 7, 6),
        **kwargs,
    )
    assert k12.applicable
    # The flattened candidate order is deterministic, so 6 and 7 win.
    assert k12.receipt()["selected_positions"] == [6, 7]
    assert bool(k12.attention_mask[0, 0, 9, 6])
    assert not bool(k12.attention_mask[0, 0, 9, 4])

    unavailable = actuators.build_k13(
        sequence_length=10,
        image_key_positions=(1, 2),
        query_position=9,
    )
    assert unavailable.status == "not_applicable"
    assert unavailable.receipt()["reason"]


def test_k14_fixed_plus_two_all_heads_and_future_zero() -> None:
    k14 = actuators.build_k14t(
        sequence_length=8,
        image_key_positions=(1, 2, 3, 4),
        b_exclusive_positions=(3, 4),
        query_position=6,
        layer_count=24,
        head_count=2,
    )
    bias = k14.score_bias
    assert bias is not None
    assert tuple(bias.bias.shape) == (24, 2, 8, 8)
    assert torch.all(bias.bias[:, :, 6, (3, 4)] == 2.0)
    assert torch.count_nonzero(bias.bias).item() == 24 * 2 * 2
    assert torch.count_nonzero(bias.bias[:, :, :6, 6:]).item() == 0
    receipt = k14.receipt()
    assert receipt["exact_plus_two_values"]
    assert receipt["offscope_nonzero_count"] == 0
    assert receipt["future_nonzero_count"] == 0

    scores = torch.zeros((1, 2, 1, 8))
    shifted = bias.apply(
        scores, layer_idx=23, query_positions=(6,), key_positions=tuple(range(8))
    )
    assert torch.all(shifted[:, :, 0, (3, 4)] == 2.0)
    assert torch.all(shifted[:, :, 0, (0, 1, 2, 5, 6, 7)] == 0.0)
    assert bias.receipt()["applications_by_layer"] == {23: 1}


def test_k14b_reuses_exact_equal_count_row_major_selection() -> None:
    k12 = actuators.build_k12(
        sequence_length=12,
        image_key_positions=tuple(range(1, 9)),
        b_exclusive_positions=(7, 8),
        background_positions=(6, 5, 4, 3),
        query_position=11,
    )
    k14b = actuators.build_k14b(
        sequence_length=12,
        image_key_positions=tuple(range(1, 9)),
        b_exclusive_positions=(7, 8),
        background_positions=(6, 5, 4, 3),
        query_position=11,
        layer_count=1,
        head_count=1,
    )
    assert (
        k12.receipt()["selected_positions"]
        == k14b.receipt()["selected_positions"]
        == [3, 4]
    )
    assert k14b.score_bias is not None


class _FakeLayer(torch.nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, *, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert attention_mask is not None
        return hidden_states + attention_mask[..., :1, :1].reshape(1, 1, 1).to(
            hidden_states
        )


class _FakeModel(torch.nn.Module):
    def __init__(self, layer_count: int = 3) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_FakeLayer() for _ in range(layer_count))

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


def test_h00_h10_h20_and_all_layer_consumption_attestation() -> None:
    h00 = actuators.build_h00(sequence_length=7)
    assert h00.receipt()["changed_cell_count"] == 0
    h10 = actuators.build_h10(
        sequence_length=7, query_positions=(6,), latest_terminal_key_position=4
    )
    h20 = actuators.build_h20(
        sequence_length=7, query_positions=(6,), latest_row_key_positions=(2, 3, 4)
    )
    assert h10.receipt()["masked_query_key_edge_count"] == 1
    assert h20.receipt()["masked_query_key_edge_count"] == 3

    model = _FakeModel()
    hidden = torch.zeros((1, 7, 4))
    output, receipt = actuators.attest_all_layer_consumption(
        model,
        h20.attention_mask,
        lambda: model(hidden, attention_mask=h20.attention_mask),
    )
    assert output.shape == hidden.shape
    assert receipt["passed"]
    assert receipt["all_layers_identical"]
    assert receipt["call_counts"] == {"0": 1, "1": 1, "2": 1}
    bound = actuators.attach_layer_consumption_attestation(h20, receipt)
    assert bound.receipt()["all_layer_consumption_attestation"]["passed"]


def test_block23_mass_receipt_is_per_head_and_score_callback_is_pre_softmax() -> None:
    k14 = actuators.build_k14t(
        sequence_length=6,
        image_key_positions=(1, 2, 3),
        b_exclusive_positions=(2,),
        query_position=5,
        layer_count=24,
        head_count=2,
    )
    assert k14.score_bias is not None
    diag = actuators.Block23MassDiagnostic((2,), query_positions=(5,))
    scores = torch.zeros((1, 2, 1, 6))
    shifted = actuators.apply_score_bias_with_block23_diagnostic(
        scores,
        actuator=k14.score_bias,
        layer_idx=23,
        query_positions=(5,),
        key_positions=tuple(range(6)),
        diagnostic=diag,
    )
    assert shifted[0, 0, 0, 0].item() == 0.0
    assert shifted[0, 1, 0, 2].item() == 2.0
    record = diag.receipt()["records"][0]
    assert record["head_count"] == 2
    assert record["after_mass"][0][0] > record["before_mass"][0][0]


class _FakeBlock23SDPAModule(torch.nn.Module):
    layer_idx = 23
    num_key_value_groups = 1


def test_block23_sdpa_attestor_restores_registry_and_reproduces_gqa_expansion() -> None:
    pytest.importorskip("transformers", reason="transformers is not installed")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    module = _FakeBlock23SDPAModule()
    query = torch.zeros((1, 2, 4, 8))
    key = torch.zeros((1, 2, 4, 8))
    value = torch.zeros((1, 2, 4, 8))
    mask = torch.triu(torch.full((1, 1, 4, 4), float("-inf")), diagonal=1)
    attestor = actuators.Block23SDPAMassAttestor((2,), query_positions=(3,))
    with attestor:
        output, _weights = ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, mask, scaling=8**-0.5
        )
    assert output.shape == (1, 4, 2, 8)
    receipt = attestor.receipt()
    assert receipt["passed"]
    assert receipt["exactly_one_block23_call"]
    assert receipt["delegate_untouched"]
    assert receipt["registry_restored"]
    assert receipt["delegate_identity"]["id"] == id(original)
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original

    class _GQAModule(_FakeBlock23SDPAModule):
        num_key_value_groups = 2

    gqa_query = torch.zeros((1, 4, 4, 8))
    gqa_key = torch.zeros((1, 2, 4, 8))
    gqa_value = torch.zeros((1, 2, 4, 8))
    gqa = actuators.Block23SDPAMassAttestor((2,), query_positions=(3,))
    with gqa:
        output, _weights = ALL_ATTENTION_FUNCTIONS["sdpa"](
            _GQAModule(), gqa_query, gqa_key, gqa_value, mask, scaling=8**-0.5
        )
    assert output.shape == (1, 4, 4, 8)
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    gqa_receipt = gqa.receipt()
    assert gqa_receipt["passed"]
    gqa_record = gqa_receipt["records"][0]
    assert gqa_record["q_heads"] == 4
    assert gqa_record["kv_heads"] == 2
    assert gqa_record["num_key_value_groups"] == 2
    assert gqa_record["gqa_expansion"]["head_map"] == [0, 0, 1, 1]
    assert gqa_record["selected_mass_shape"] == [1, 4, 1]
    assert len(gqa_record["selected_mass_per_query_head"][0]["head_mass"][0]) == 4

    class _NonIntegralGQAModule(_FakeBlock23SDPAModule):
        num_key_value_groups = 2.5

    nonintegral = actuators.Block23SDPAMassAttestor((2,), query_positions=(3,))
    with pytest.raises(actuators.ActuatorError):
        with nonintegral:
            ALL_ATTENTION_FUNCTIONS["sdpa"](
                _NonIntegralGQAModule(),
                gqa_query,
                gqa_key,
                gqa_value,
                mask,
                scaling=8**-0.5,
            )
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    assert not nonintegral.receipt()["passed"]

    shape_drift = actuators.Block23SDPAMassAttestor((2,), query_positions=(3,))
    with pytest.raises(actuators.ActuatorError):
        with shape_drift:
            ALL_ATTENTION_FUNCTIONS["sdpa"](
                _GQAModule(),
                gqa_query[:, :3],
                key[:, :1],
                value[:, :1],
                mask,
                scaling=8**-0.5,
            )
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    assert not shape_drift.receipt()["passed"]

    malformed = actuators.Block23SDPAMassAttestor((2,), query_positions=(3,))
    with pytest.raises(actuators.ActuatorError):
        with malformed:
            ALL_ATTENTION_FUNCTIONS["sdpa"](
                module, query, key, value, mask.bool(), scaling=8**-0.5
            )
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    assert not malformed.receipt()["passed"]


def test_installed_tiny_qwen_block23_sdpa_native_biased_mass_and_registry_restore() -> (
    None
):
    pytest.importorskip("transformers", reason="transformers is not installed")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(4321)
    config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=2,
        max_position_embeddings=32,
        rope_scaling={"mrope_section": [2, 2, 2], "rope_type": "default"},
    )
    config._attn_implementation = "sdpa"
    model = Qwen3VLTextModel(config).eval()
    input_ids = torch.arange(4).reshape(1, -1)
    position_ids = torch.arange(4).reshape(1, 1, -1).expand(3, 1, -1)
    k01 = actuators.build_k01(sequence_length=4, dtype=torch.float32)
    k14 = actuators.build_k14t(
        sequence_length=4,
        image_key_positions=(1, 2),
        b_exclusive_positions=(2,),
        query_position=3,
        layer_count=24,
        head_count=16,
    )
    assert k01.attention_mask is not None and k14.attention_mask is not None
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    native_reference = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=k01.attention_mask,
        use_cache=False,
    )
    native_attestor = actuators.Block23SDPAMassAttestor(
        (2,), query_positions=(3,), phase="native"
    )
    with native_attestor:
        native = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=k01.attention_mask,
            use_cache=False,
        )
    biased_attestor = actuators.Block23SDPAMassAttestor(
        (2,), query_positions=(3,), phase="biased"
    )
    with biased_attestor:
        biased = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=k14.attention_mask,
            use_cache=False,
        )
    native_receipt = native_attestor.receipt()
    biased_receipt = biased_attestor.receipt()
    assert native_receipt["passed"] and biased_receipt["passed"]
    assert native_receipt["registry_restored"] and biased_receipt["registry_restored"]
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    assert torch.allclose(native.last_hidden_state, native_reference.last_hidden_state)
    assert not torch.allclose(native.last_hidden_state, biased.last_hidden_state)
    comparison = actuators.compare_block23_sdpa_mass_receipts(
        native_receipt, biased_receipt
    )
    assert comparison["mass_shift_observed"]
    assert comparison["all_query_heads_nonzero_shift"]
    assert comparison["same_delegate"]
    assert comparison["q_heads"] == 16
    assert comparison["kv_heads"] == 8
    assert comparison["num_key_value_groups"] == 2
    assert comparison["selected_mass_shape"] == [1, 16, 1]
    assert len(comparison["delta_mass"][0]) == 16
    assert comparison["passed"]
    assert biased_receipt["records"][0]["selected_key_positions"] == [2]
    assert biased_receipt["records"][0]["query_positions"] == [3]


def test_natural_callback_exposes_real_attention_input_and_score_protocol() -> None:
    arm = actuators.build_k10(
        sequence_length=5,
        image_key_positions=(1, 2),
        b_exclusive_positions=(2,),
        query_position=4,
    )
    callback = actuators.make_natural_runner_callback(arm)
    payload = callback(sequence_length=5, query_position=4)
    assert torch.equal(payload["attention_mask"], arm.attention_mask)
    assert payload["score_bias"] is None
    assert payload["actuator_id"] == "K10"


def test_scalar_step_factory_rebuilds_growing_prefix_with_absolute_keys() -> None:
    factory = actuators.build_scalar_step_factory(
        "K10",
        image_key_positions=(1, 2, 5),
        b_exclusive_positions=(2, 5),
        dtype=torch.float32,
        max_sequence_length=8,
    )
    first = factory.build(3)
    assert first.attention_mask is not None
    assert tuple(first.attention_mask.shape) == (1, 1, 3, 3)
    assert first.receipt()["current_query_position"] == 2
    # The absolute key at position 5 is retained by the factory but cannot
    # appear in a length-3 tensor yet.
    assert first.receipt()["fixed_absolute_b_exclusive_positions"] == [2, 5]
    assert first.receipt()["active_b_exclusive_positions"] == [2]
    assert first.attention_mask[0, 0, 2, 2].item() == 0.0

    second = factory(sequence_length=6, query_position=5)
    assert second["attention_mask"] is not None
    assert tuple(second["attention_mask"].shape) == (1, 1, 6, 6)
    assert second["attention_mask"][0, 0, 5, 5].item() == 0.0
    assert second["receipt"]["active_b_exclusive_positions"] == [2, 5]
    # A fresh tensor is built; the old scalar mask remains length-3 and is not
    # silently reused for the growing prefix.
    assert tuple(first.attention_mask.shape) == (1, 1, 3, 3)
    assert second["receipt"]["sequence_growth_rebuilt"]

    # The S gate adapter uses a context-first wrapper.  The factory accepts
    # that shape and still rebuilds from the input prefix length.
    wrapped = factory(
        object(),
        input_ids=torch.zeros((1, 7), dtype=torch.long),
        step=0,
        row_index=0,
        arm_id="K10",
    )
    assert wrapped["attention_mask"] is not None
    assert tuple(wrapped["attention_mask"].shape) == (1, 1, 7, 7)


@pytest.mark.parametrize(
    "arm_id", ["K01", "K10", "K11", "K12", "K13", "H00", "H10", "H20"]
)
def test_scalar_step_masks_keep_future_keys_causal_and_scope_local(arm_id: str) -> None:
    kwargs: dict[str, object] = {
        "image_key_positions": (1, 2, 3, 4),
        "b_exclusive_positions": (3,),
        "a_exclusive_positions": (1,),
        "background_positions": (2, 4),
        "same_class_competitor_positions": (2,),
        "latest_terminal_key_positions": (2,),
        "latest_row_key_positions": (1, 2),
        "dtype": torch.float32,
    }
    factory = actuators.build_scalar_step_factory(arm_id, **kwargs)
    step = factory.build(6, query_position=4)
    if not step.applicable:
        pytest.skip(step.receipt().get("reason", "arm not applicable for fixture"))
    mask = step.attention_mask
    assert mask is not None
    # Every row remains causal, including rows outside the current-query
    # intervention scope.
    future = mask[0, 0].triu(diagonal=1)
    assert torch.all(torch.isneginf(future[future != 0]))
    receipt = step.receipt()
    assert receipt["future_changed_cell_count"] == 0
    assert receipt["offscope_changed_cell_count"] == 0
    # Only the declared query row may differ from the all-allowed causal base.
    changed = (
        mask[0, 0]
        != actuators.build_k01(sequence_length=6, dtype=torch.float32).attention_mask[
            0, 0
        ]
    )
    assert torch.all(~changed[:4])
    assert torch.all(~changed[5:])


def test_k14_factory_is_consumable_additive_mask_and_balanced() -> None:
    common = {
        "sequence_length": 7,
        "image_key_positions": (1, 2, 3, 4, 5),
        "b_exclusive_positions": (3, 4),
        "background_positions": (5, 2, 1),
        "query_position": 6,
        "layer_count": 2,
        "head_count": 2,
        "dtype": torch.float32,
    }
    target = actuators.build_scalar_step_factory("K14T", **common)
    background = actuators.build_scalar_step_factory("K14B", **common)
    target_step = target.build(7, query_position=6)
    background_step = background.build(7, query_position=6)
    assert target_step.attention_mask is not None
    assert background_step.attention_mask is not None
    target_mask = target_step.attention_mask[0, 0]
    background_mask = background_step.attention_mask[0, 0]
    assert target_mask.dtype.is_floating_point
    assert target_mask[6, 3].item() == pytest.approx(2.0)
    assert target_mask[6, 4].item() == pytest.approx(2.0)
    assert target_mask[5, 3].item() == pytest.approx(0.0)
    assert target_mask[6, 6].item() == pytest.approx(0.0)
    assert torch.isneginf(target_mask[4, 6])
    assert background_step.receipt()["selected_positions"] == [1, 2]
    assert (
        background_step.receipt()["selected_key_count"]
        == target_step.receipt()["selected_key_count"]
        == 2
    )
    assert background_step.receipt()["mask_consumed_via"] == "attention_mask"
    assert background_step.receipt()["dose"] == pytest.approx(2.0)
    assert background_step.receipt()["selected_identity_sha256"]
    assert background_mask[6, 1].item() == pytest.approx(2.0)
    assert background_mask[6, 2].item() == pytest.approx(2.0)


def test_k14_factory_returns_current_length_causal_noop_until_absolute_key_enters() -> (
    None
):
    factory = actuators.build_scalar_step_factory(
        "K14T",
        image_key_positions=(1, 4),
        b_exclusive_positions=(4,),
        dtype=torch.float32,
    )
    before = factory.build(3, query_position=2)
    assert before.status == "not_applicable"
    assert before.attention_mask is not None
    assert tuple(before.attention_mask.shape) == (1, 1, 3, 3)
    future = before.attention_mask[0, 0].triu(diagonal=1)
    assert torch.all(future[future != 0] == float("-inf"))
    after = factory.build(5, query_position=4)
    assert after.applicable
    assert after.attention_mask is not None
    assert after.attention_mask[0, 0, 4, 4].item() == pytest.approx(2.0)


class _DeterministicTinyMaskedLM(torch.nn.Module):
    """Small deterministic attention/logit surface for the K14 contrast."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer()])
        self.register_buffer("values", torch.tensor([0.0, 1.0, 2.0, 4.0]))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        del input_ids
        scores = self.values[: attention_mask.shape[-1]].reshape(1, 1, 1, -1)
        scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)
        return probs


def test_k14_changes_deterministic_attention_relative_to_k01() -> None:
    model = _DeterministicTinyMaskedLM().eval()
    k01 = actuators.build_scalar_step_factory("K01", dtype=torch.float32).build(
        4, query_position=3
    )
    k14 = actuators.build_scalar_step_factory(
        "K14T",
        image_key_positions=(1, 2),
        b_exclusive_positions=(2,),
        dtype=torch.float32,
        layer_count=1,
        head_count=1,
    ).build(4, query_position=3)
    assert k01.attention_mask is not None and k14.attention_mask is not None
    baseline = model(torch.arange(4).reshape(1, -1), k01.attention_mask)
    shifted = model(torch.arange(4).reshape(1, -1), k14.attention_mask)
    assert not torch.allclose(baseline, shifted)
    assert shifted[0, 0, 3, 2] > baseline[0, 0, 3, 2]
    assert k14.receipt()["attention_mask"]["exact_scope"]


class _IgnoredMaskLayer(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return hidden_states


def test_malformed_or_ignored_masks_fail_closed_attestation() -> None:
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([_IgnoredMaskLayer(), _IgnoredMaskLayer()])
    mask = actuators.build_k01(sequence_length=3, dtype=torch.float32).attention_mask
    assert mask is not None
    _output, receipt = actuators.attest_all_layer_consumption(
        model,
        mask,
        lambda: [
            layer(torch.zeros((1, 3, 2)), attention_mask=None) for layer in model.layers
        ],
    )
    assert not receipt["passed"]
    with pytest.raises(actuators.TechnicalInvalid):
        actuators.require_all_layer_consumption(receipt)


def test_block23_receipt_states_interpretation_requirements() -> None:
    diag = actuators.Block23MassDiagnostic((2,), query_positions=(3,))
    scores = torch.zeros((1, 2, 1, 4))
    diag.observe_scores(
        scores,
        scores + torch.tensor([[[[0.0, 0.0, 2.0, 0.0]], [[0.0, 0.0, 2.0, 0.0]]]]),
        layer_idx=23,
        query_positions=(3,),
        key_positions=(0, 1, 2, 3),
    )
    receipt = diag.receipt()
    assert receipt["requirements"]["applied_bias_receipt_required_for_interpretation"]
    assert receipt["requirements"]["mass_shift_receipt_required_for_soft_null"]
    assert receipt["mass_shift_observed"]


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
def test_installed_qwen_cpu_accepts_float_or_bool_4d_mask_and_consumes_all_layers(
    dtype: torch.dtype,
) -> None:
    pytest.importorskip("transformers", reason="transformers is not installed")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_scaling={"mrope_section": [2, 2, 2], "rope_type": "default"},
    )
    config._attn_implementation = "sdpa"
    model = Qwen3VLTextModel(config).eval()
    input_ids = torch.arange(5).reshape(1, -1)
    position_ids = torch.arange(5).reshape(1, 1, -1).expand(3, 1, -1)
    mask = actuators.build_k01(sequence_length=5, dtype=dtype).attention_mask
    assert mask is not None
    if dtype.is_floating_point:
        assert mask.dtype == dtype
    consumed: list[torch.Tensor] = []
    handles = [
        layer.register_forward_pre_hook(
            lambda _module, _args, kwargs: consumed.append(kwargs["attention_mask"]),
            with_kwargs=True,
        )
        for layer in model.layers
    ]
    try:
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=mask,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert len(consumed) == len(model.layers)
    assert all(torch.equal(item, mask) for item in consumed)


def test_installed_qwen_cpu_probe_emits_canonical_pre_gpu_identity_fields() -> None:
    pytest.importorskip("transformers", reason="transformers is not installed")
    probe = actuators.run_installed_qwen_cpu_probe()
    assert probe["status"] == "passed"
    assert probe["torch_version"]
    assert probe["transformers_version"]
    assert "qwen" in probe["qwen_model_class"].lower()
    assert probe["float_additive_4d_mask_passthrough"] is True
    assert probe["all_layer_consumption"] is True
    assert probe["silent_coercion"] is False
    assert probe["ignored_kwargs"] is False
    assert probe["block23_sdpa_mass_attestation"] is True
    assert probe["block23_sdpa_mass_receipt"]["status"] == "passed"
    block23 = probe["block23_sdpa_mass_receipt"]
    assert block23["q_heads"] == 16
    assert block23["kv_heads"] == 8
    assert block23["num_key_value_groups"] == 2
    assert block23["all_query_heads_nonzero_shift"] is True
    assert block23["same_delegate"] is True


def test_installed_tiny_qwen_consumes_k14_additive_mask_and_changes_hidden_state() -> (
    None
):
    pytest.importorskip("transformers", reason="transformers is not installed")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(1234)
    config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_scaling={"mrope_section": [2, 2, 2], "rope_type": "default"},
    )
    config._attn_implementation = "sdpa"
    model = Qwen3VLTextModel(config).eval()
    input_ids = torch.arange(6).reshape(1, -1)
    position_ids = torch.arange(6).reshape(1, 1, -1).expand(3, 1, -1)
    k01 = actuators.build_scalar_step_factory("K01", dtype=torch.float32).build(
        6, query_position=5
    )
    k14 = actuators.build_scalar_step_factory(
        "K14T",
        image_key_positions=(1, 2, 3),
        b_exclusive_positions=(2,),
        dtype=torch.float32,
        layer_count=3,
        head_count=2,
    ).build(6, query_position=5)
    assert k01.attention_mask is not None and k14.attention_mask is not None
    native, native_receipt = actuators.attest_all_layer_consumption(
        model,
        k01.attention_mask,
        lambda: model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=k01.attention_mask,
            use_cache=False,
        ),
    )
    shifted, shifted_receipt = actuators.attest_all_layer_consumption(
        model,
        k14.attention_mask,
        lambda: model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=k14.attention_mask,
            use_cache=False,
        ),
    )
    assert native_receipt["passed"] and shifted_receipt["passed"]
    assert not torch.allclose(native.last_hidden_state, shifted.last_hidden_state)
    assert k14.attention_mask[0, 0, 5, 2].item() == pytest.approx(2.0)

from __future__ import annotations

import pytest
import torch

import scripts.research.run_dynamic_history_and_crossover_probe as probe


class Qwen3VLTextDecoderLayerFake(torch.nn.Module):
    _coordexp_decoder_layer = True

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.ones(width))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.bias


class FakeModel(torch.nn.Module):
    def __init__(self, width: int = 4, layer_count: int = 24) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList(
            Qwen3VLTextDecoderLayerFake(width) for _ in range(layer_count)
        )
        self.calls = 0

    def forward(self, *, input_ids: torch.Tensor, position_ids: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        self.calls += 1
        self.last_attention_mask = _kwargs.get("attention_mask")
        hidden = input_ids.float().unsqueeze(-1).expand(*input_ids.shape, 4).clone()
        for layer in self.model.language_model.layers:
            hidden = layer(hidden)
        return hidden


def _inputs(contract: probe.ExactPrefixContract) -> dict[str, object]:
    return {
        "input_ids": torch.tensor([list(contract.prefix_token_ids)], dtype=torch.long),
        "position_ids": contract.position_ids,
        "mrope_hash": contract.mrope_hash,
        "use_cache": False,
    }


def _contract() -> probe.ExactPrefixContract:
    positions = torch.arange(9, dtype=torch.long).reshape(3, 1, 3)
    return probe.build_exact_prefix_contract((10, 11, 12), positions, wrapper="object_box_closed")


def _parser() -> probe.NativeRowParser:
    return probe.NativeRowParser.closed()


def _row_tokens(description_token: int = 100) -> tuple[int, ...]:
    return (
        151646,
        description_token,
        151647,
        151648,
        151670,
        151671,
        151672,
        151673,
        probe.BOX_END_TOKEN_ID,
    )


def _test_forward_receipt(prefix: tuple[int, ...]) -> dict[str, object]:
    return {
        "prefix_token_ids_sha256": probe.hash_token_ids(prefix),
        "position_ids_sha256": "test-position-hash",
        "mrope_hash": "test-mrope-hash",
        "mask_sha256": "none",
        "wrapper": "object_box_closed",
        "no_op_max_abs_delta": 0.0,
        "non_target_max_abs_delta": 0.0,
        "mask_non_image_changed": False,
        "mask_offscope_changed": False,
        "mask_future_changed": False,
        "hook_call_count": 1,
        "hook_applied_count": 0,
        "hook_clean": True,
        "execution_adapter": "test_adapter",
        "real_forward": False,
        "test_adapter": True,
    }


def _owner_receipt(status: str, owner_id: str | None = None) -> dict[str, object]:
    return {
        "matcher_id": "strict-physical-owner-test-matcher",
        "status": status,
        "matched_owner_id": owner_id,
        "source_specific": True,
        "physical_match": status == "matched",
        "unmatched": status == "unmatched",
        "ambiguous": status == "ambiguous",
    }


def _description_owner_matcher(
    owners: dict[int, str | None], *, neutral_status: str = "unmatched"
):
    def match(token_ids: tuple[int, ...], _parse_receipt: object) -> dict[str, object]:
        owner_id = owners.get(token_ids[1]) if len(token_ids) > 1 else None
        return _owner_receipt("matched", owner_id) if owner_id is not None else _owner_receipt(neutral_status)

    return match


def _generated_fixture(prefix: tuple[int, ...], row: tuple[int, ...], **flags: object) -> dict[str, object]:
    return {"token_ids": row, "forward_receipt": _test_forward_receipt(prefix), **flags}


def _closed_carrier() -> probe.CarrierContract:
    return probe.build_carrier_contract(
        "closed", carrier_token_id=probe.BOX_END_TOKEN_ID, wrapper="object_box_closed"
    )


def test_block23_capture_and_single_position_noop_replacement_is_exact() -> None:
    model = FakeModel()
    contract = _contract()
    layer, receipt = probe.resolve_decoder_layer(model)
    assert receipt["layer_idx"] == 23
    with probe.ResidualSpanCapture(layer, (1, 2)) as capture:
        native = model(**_inputs(contract))
    assert capture.state is not None
    assert capture.call_count == 1
    with probe.ResidualSpanReplacement(layer, (1, 2), capture.state.clone()) as replacement:
        replay = model(**_inputs(contract))
    assert torch.equal(native, replay)
    assert replacement.receipt()["hook_clean"] is True
    assert replacement.receipt()["non_target_max_abs_delta"] == 0.0


def test_block23_whole_row_replacement_is_position_local_and_persistent() -> None:
    model = FakeModel()
    contract = _contract()
    layer, _ = probe.resolve_decoder_layer(model)
    with probe.ResidualSpanCapture(layer, (0, 1, 2)) as capture:
        model(**_inputs(contract))
    assert capture.state is not None
    replacement_state = capture.state + 3.0
    hook = probe.ResidualSpanReplacement(layer, (0, 1, 2), replacement_state, persistent=True)
    hook.install()
    model(**_inputs(contract))
    model(**_inputs(contract))
    hook.remove()
    assert hook.applied_count == 2
    assert hook.call_count == 2
    assert hook.max_non_target_abs_delta == 0.0
    assert hook.receipt()["hook_clean"] is True


def test_forward_with_dynamic_arm_preserves_prefix_and_marks_hook_receipt() -> None:
    model = FakeModel()
    contract = _contract()
    layer, _ = probe.resolve_block23(model)
    with probe.ResidualSpanCapture(layer, (2,)) as capture:
        model(**_inputs(contract))
    assert capture.state is not None
    arm = probe.build_dynamic_arm(
        "D10",
        prefix_width=2,
        latest_row_token_ids=[probe.BOX_END_TOKEN_ID],
        carrier=_closed_carrier(),
    )
    output, receipt = probe.forward_with_dynamic_arm(
        model,
        _inputs(contract),
        prefix_contract=contract,
        arm=arm,
        replacement=capture.state,
    )
    assert isinstance(output, torch.Tensor)
    assert receipt["arm"] == "D10"
    assert receipt["hook_clean"] is True
    assert receipt["prefix_token_ids_sha256"] == contract.prefix_hash


def test_dynamic_and_p3_matrix_use_natural_generators_and_report_horizons() -> None:
    row = list(_row_tokens())
    closed = _closed_carrier()
    arms = {
        arm_id: probe.build_dynamic_arm(
            arm_id,
            prefix_width=0,
            latest_row_token_ids=row,
            earlier_row_token_ids=row,
            carrier=closed,
            donor_row_token_ids=row,
            same_parent=True,
            same_class=True,
        )
        for arm_id in probe.DYNAMIC_ARM_IDS
    }

    def dynamic_factory(_arm_id: str, _arm: probe.DynamicArm, _horizon: int):
        return lambda prefix, _index: _generated_fixture(prefix, tuple(row))

    matcher = _description_owner_matcher({100: "new"})
    dynamic = probe.run_dynamic_matrix(
        (9,),
        arms,
        dynamic_factory,
        covered_owner_ids=(),
        parser=_parser(),
        owner_matcher=matcher,
        allow_test_adapter=True,
    )
    assert set(dynamic["arms"]) == {"horizon_1", "horizon_3"}
    assert dynamic["arms"]["horizon_1"]["D10"]["net"] == 1

    static = probe.build_static_arm(
        "K11", sequence_length=4, image_key_positions=(1,), a_exclusive_positions=(1,), query_positions=(2, 3)
    )
    d10 = arms["D10"]
    cells = probe.build_p3_cells(static, d10)

    def p3_factory(_cell_id: str, _cell: probe.P3Cell, _horizon: int):
        return lambda prefix, _index: _generated_fixture(prefix, tuple(row))

    p3 = probe.run_p3_matrix(
        (9,),
        cells,
        p3_factory,
        covered_owner_ids=(),
        parser=_parser(),
        owner_matcher=matcher,
        allow_test_adapter=True,
    )
    assert set(p3["cells"]["horizon_1"]) == set(probe.P3_CELL_IDS)
    assert p3["horizon_1"]["tau"] == 0


def test_dynamic_arm_matrix_and_d21_fail_closed() -> None:
    row = [101, 151670, 151671, 151672, 151673, probe.BOX_END_TOKEN_ID]
    earlier = [201, 151674, 151675, 151676, 151677, probe.BOX_END_TOKEN_ID]
    carrier = _closed_carrier()
    arms = {
        arm_id: probe.build_dynamic_arm(
            arm_id,
            prefix_width=7,
            latest_row_token_ids=row,
            earlier_row_token_ids=earlier,
            carrier=carrier,
            donor_row_token_ids=row,
            same_parent=True,
            same_class=True,
        )
        for arm_id in probe.DYNAMIC_ARM_IDS
    }
    assert all(arms[arm_id].status == "ready" for arm_id in probe.DYNAMIC_ARM_IDS)
    assert arms["D01"].positions == arms["D10"].positions
    assert arms["D10"].positions == (12,)
    assert arms["D11"].positions == (11,)
    assert len(arms["D20"].positions) == len(row)
    invalid = probe.build_dynamic_arm(
        "D21",
        prefix_width=7,
        latest_row_token_ids=row,
        donor_row_token_ids=row,
        same_parent=False,
        same_class=True,
    )
    assert invalid.status == "not_applicable"
    assert "same-parent" in invalid.reason


def test_d21_requires_equal_length_and_same_class() -> None:
    row = [101, 151670, 151671, 151672, 151673, probe.BOX_END_TOKEN_ID]
    carrier = _closed_carrier()
    for kwargs, needle in (
        ({"same_parent": True, "same_class": False, "donor_row_token_ids": row}, "same-class"),
        ({"same_parent": True, "same_class": True, "donor_row_token_ids": row[:-1]}, "equal token length"),
    ):
        arm = probe.build_dynamic_arm(
            "D21", prefix_width=0, latest_row_token_ids=row, carrier=carrier, **kwargs
        )
        assert arm.status == "not_applicable"
        assert needle in arm.reason


def test_k11_changes_only_declared_image_keys_for_declared_queries() -> None:
    mask = probe.build_k11_key_removal_mask(
        sequence_length=6,
        image_key_positions=(1, 2, 3),
        a_exclusive_positions=(2,),
        query_positions=(3, 4, 5),
    )
    assert mask.shape == (1, 1, 6, 6)
    assert bool(mask[0, 0, 3, 2]) is False
    assert bool(mask[0, 0, 4, 2]) is False
    assert bool(mask[0, 0, 2, 1]) is True
    assert bool(mask[0, 0, 5, 1]) is True
    with pytest.raises(probe.TechnicalInvalid, match="subset"):
        probe.build_k11_key_removal_mask(
            sequence_length=4,
            image_key_positions=(1,),
            a_exclusive_positions=(2,),
            query_positions=(3,),
        )


def test_p3_cells_and_factorial_deltas() -> None:
    static = probe.build_static_arm(
        "K11", sequence_length=5, image_key_positions=(1, 2), a_exclusive_positions=(1,), query_positions=(3, 4)
    )
    dynamic = probe.build_dynamic_arm(
        "D10", prefix_width=0, latest_row_token_ids=[1, probe.BOX_END_TOKEN_ID], carrier=_closed_carrier()
    )
    cells = probe.build_p3_cells(static, dynamic, persistence="persistent")
    assert tuple(cells) == probe.P3_CELL_IDS
    assert cells["Y10"].static_arm.arm_id == "K11"
    assert cells["Y01"].dynamic_arm.arm_id == "D10"
    result = probe.factorial_deltas({cell: {"net": index} for index, cell in enumerate(probe.P3_CELL_IDS)})
    assert result["Delta_static"] == 1
    assert result["Delta_dynamic"] == 2
    assert result["tau"] == 0
    with pytest.raises(probe.TechnicalInvalid, match="missing cells"):
        probe.factorial_deltas({"Y00": {"net": 0}})


def test_natural_horizon_bookkeeping_g_k_l_repeat_parse_stop() -> None:
    parser = _parser()
    rows = iter((_row_tokens(100), _row_tokens(101), _row_tokens(100)))

    def generate(prefix: tuple[int, ...], _row_index: int) -> dict[str, object]:
        return _generated_fixture(prefix, next(rows))

    result = probe.run_native_horizon(
        (9,),
        generate,
        covered_owner_ids=("A",),
        horizon=3,
        parser=parser,
        owner_matcher=_description_owner_matcher({100: "A", 101: "B"}),
        allow_test_adapter=True,
    )
    assert result["G"] == ["B"]
    assert result["K"] == ["A"]
    assert result["L"] == []
    assert result["net"] == 1
    assert result["repeat_hazard"] == {"t+1": 1, "t+2": 0, "t+3": 1}
    assert result["parse"]["duplicate_rows"] == 2
    assert result["stop"]["stopped"] is False


def test_natural_horizon_rejects_text_forced_and_teacher_forced_paths() -> None:
    with pytest.raises(probe.TechnicalInvalid, match="text"):
        probe.run_native_horizon((1,), lambda _prefix, _index: "<row>", horizon=1)
    with pytest.raises(probe.TechnicalInvalid, match="forced"):
        probe.run_native_horizon((1,), lambda _prefix, _index: [probe.BOX_END_TOKEN_ID], forced=True)
    with pytest.raises(probe.TechnicalInvalid, match="teacher"):
        probe.run_native_horizon((1,), lambda _prefix, _index: [probe.BOX_END_TOKEN_ID], teacher_forced=True)


def test_matrix_retains_generator_contract_failure_as_invalid_receipt() -> None:
    arms = {"D00": probe.build_dynamic_arm("D00", prefix_width=0)}

    def broken(_arm_id: str, _arm: probe.DynamicArm, _horizon: int):
        return lambda _prefix, _index: "text-is-not-a-row"

    result = probe.run_dynamic_matrix((1,), arms, broken, horizons=(1,))
    receipt = result["arms"]["horizon_1"]["D00"]
    assert receipt["status"] == "invalid"
    assert receipt["invalid"] is True


def test_exact_prefix_and_receipt_validation_fail_closed() -> None:
    contract = _contract()
    passed = probe.validate_exact_prefix(
        contract,
        input_ids=torch.tensor([[10, 11, 12]]),
        position_ids=contract.position_ids,
        mrope_hash=contract.mrope_hash,
    )
    assert passed["passed"] is True
    with pytest.raises(probe.TechnicalInvalid, match="token IDs"):
        probe.validate_exact_prefix(
            contract,
            input_ids=torch.tensor([[10, 11, 13]]),
            position_ids=contract.position_ids,
            mrope_hash="mrope-test",
        )
    with pytest.raises(probe.TechnicalInvalid, match="donor cache"):
        probe.validate_mechanical_receipt(
            {
                "prefix_token_ids_sha256": "p",
                "position_ids_sha256": "q",
                "mrope_hash": "m",
                "wrapper": "object_box_closed",
                "mask_sha256": "none",
                "no_op_max_abs_delta": 0.0,
                "non_target_max_abs_delta": 0.0,
                "mask_non_image_changed": False,
                "mask_offscope_changed": False,
                "mask_future_changed": False,
                "hook_call_count": 0,
                "hook_applied_count": 0,
                "hook_clean": True,
                "execution_adapter": "test_adapter",
                "real_forward": False,
                "test_adapter": True,
                "donor_cache": True,
            }
        )
    with pytest.raises(probe.TechnicalInvalid, match="lacks fields"):
        probe.validate_mechanical_receipt({"hook_clean": True})


def test_norm_matched_mean_preserves_target_average_norm() -> None:
    target = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    background = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    replacement = probe.norm_matched_mean(target, background)
    assert torch.isclose(replacement.norm(), target.norm(dim=-1).mean())


def test_native_parser_owns_completion_and_rejects_box_end_only_or_mapping_override() -> None:
    parser = _parser()
    accepted = parser.parse(_row_tokens())
    assert accepted["complete"] is True
    assert parser.parse((probe.BOX_END_TOKEN_ID,))["complete"] is False
    malformed = probe._coerce_row(
        {
            "raw_generated_token_ids": [probe.BOX_END_TOKEN_ID],
            "complete": True,
            "forward_receipt": _test_forward_receipt((9,)),
        },
        parser=parser,
        owner_matcher=_description_owner_matcher({}),
        expected_prefix_token_ids=(9,),
        allow_test_adapter=True,
    )
    assert malformed.complete is False
    assert malformed.parse_status == "malformed"
    commit_parser = probe.NativeRowParser.commit()
    commit_row = _row_tokens() + (probe.COMMIT_TOKEN_ID,)
    assert commit_parser.parse(commit_row)["complete"] is True
    assert parser.parse(commit_row)["complete"] is False


def test_position_shape_and_local_mrope_identity_are_hard_contracts() -> None:
    positions = torch.arange(12, dtype=torch.long).reshape(4, 1, 3)
    contract = probe.build_exact_prefix_contract((10, 11, 12), positions, wrapper="object_box_closed")
    assert probe.validate_exact_prefix(
        contract,
        input_ids=torch.tensor([[10, 11, 12]]),
        position_ids=positions,
        mrope_hash=contract.mrope_hash,
    )["passed"] is True
    with pytest.raises(probe.TechnicalInvalid, match="position_ids"):
        probe.validate_exact_prefix(
            contract,
            input_ids=torch.tensor([[10, 11, 12]]),
            position_ids=positions + 1,
            mrope_hash=contract.mrope_hash,
        )
    with pytest.raises(probe.TechnicalInvalid, match="rank-3"):
        probe.build_exact_prefix_contract((10, 11, 12), torch.arange(3).reshape(1, 1, 3), wrapper="object_box_closed")


def test_p3_forward_combines_k11_mask_and_d10_hook_in_one_call() -> None:
    model = FakeModel()
    contract = _contract()
    static = probe.build_static_arm(
        "K11", sequence_length=3, image_key_positions=(1,), a_exclusive_positions=(1,), query_positions=(2,)
    )
    dynamic = probe.build_dynamic_arm(
        "D10", prefix_width=2, latest_row_token_ids=[probe.BOX_END_TOKEN_ID], carrier=_closed_carrier()
    )
    layer, _ = probe.resolve_block23(model)
    with probe.ResidualSpanCapture(layer, (2,)) as capture:
        model(**_inputs(contract))
    assert capture.state is not None
    cell = probe.P3Cell("Y11", static, dynamic, "one_shot")
    output, receipt = probe.forward_with_p3_cell(
        model,
        _inputs(contract),
        prefix_contract=contract,
        cell=cell,
        replacement=capture.state,
    )
    assert isinstance(output, torch.Tensor)
    assert model.calls == 2
    assert isinstance(model.last_attention_mask, torch.Tensor)
    assert receipt["cell_id"] == "Y11"
    assert receipt["static_arm"] == "K11"
    assert receipt["dynamic_arm"] == "D10"
    assert probe.validate_mechanical_receipt(receipt)["passed"] is True


def test_installed_qwen_sdpa_only_consumes_k11_through_attention_mask() -> None:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rope_scaling={
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 0],
            "rope_type": "default",
        },
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3VLTextModel(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    position_ids = torch.arange(4).reshape(1, 1, 4).expand(3, -1, -1)
    baseline_mask = torch.tril(torch.ones((1, 1, 4, 4), dtype=torch.bool))
    k11_mask = baseline_mask.clone()
    k11_mask[:, :, 3, 1] = False
    with torch.no_grad():
        baseline = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=baseline_mask,
            use_cache=False,
        ).last_hidden_state
        ignored_custom_kwarg = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=baseline_mask,
            custom_attention_mask=k11_mask,
            use_cache=False,
        ).last_hidden_state
        real_attention_mask = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=k11_mask,
            use_cache=False,
        ).last_hidden_state
    assert torch.equal(baseline, ignored_custom_kwarg)
    assert float((baseline - real_attention_mask).abs().max().item()) > 0.0

    contract = _contract()
    bypass_inputs = _inputs(contract)
    bypass_inputs["custom_attention_mask"] = probe.build_k11_key_removal_mask(
        sequence_length=3,
        image_key_positions=(1,),
        a_exclusive_positions=(1,),
        query_positions=(2,),
    )
    with pytest.raises(probe.TechnicalInvalid, match="custom_attention_mask"):
        probe.forward_with_dynamic_arm(
            FakeModel(),
            bypass_inputs,
            prefix_contract=contract,
            arm=probe.build_dynamic_arm("D00", prefix_width=0),
        )


def test_persistent_helper_rebuilds_prefix_positions_mask_and_terminal_hook_per_row() -> None:
    model = FakeModel()
    parser = _parser()
    row = _row_tokens()
    initial_prefix = (10, 11, 12, *row)

    def input_builder(prefix: tuple[int, ...]) -> dict[str, object]:
        positions = torch.arange(3 * len(prefix), dtype=torch.long).reshape(3, 1, len(prefix))
        return {
            "input_ids": torch.tensor([list(prefix)], dtype=torch.long),
            "position_ids": positions,
            "mrope_hash": probe.compute_mrope_hash(positions),
            "use_cache": False,
        }

    def static_builder(prefix: tuple[int, ...], _row_index: int) -> probe.StaticArm:
        return probe.build_static_arm(
            "K11",
            sequence_length=len(prefix),
            image_key_positions=(1,),
            a_exclusive_positions=(1,),
            query_positions=tuple(range(2, len(prefix))),
        )

    def replacement_builder(_model: object, _prefix: tuple[int, ...], _row_index: int, _arm: probe.DynamicArm) -> torch.Tensor:
        return torch.zeros((1, 4))

    def row_decoder(_output: object, _prefix: tuple[int, ...], row_index: int) -> dict[str, object]:
        return {"token_ids": _row_tokens(99 + row_index)}

    result = probe.run_p3_persistent_rows(
        model,
        initial_prefix,
        input_builder=input_builder,
        row_decoder=row_decoder,
        parser=parser,
        owner_matcher=_description_owner_matcher({100: "owner-1", 101: "owner-2", 102: "owner-3"}),
        wrapper="object_box_closed",
        carrier=_closed_carrier(),
        initial_latest_row_token_ids=row,
        static_arm_builder=static_builder,
        replacement_builder=replacement_builder,
        max_rows=3,
        execution_adapter="test_adapter",
        allow_test_adapter=True,
    )
    assert result["status"] == "valid"
    assert result["persistent_reinstalled_per_row"] is True
    assert len(result["per_row_calls"]) == 3
    assert all(call["hook_call_count"] == 1 for call in result["per_row_calls"])
    assert len({call["prefix_token_ids_sha256"] for call in result["per_row_calls"]}) == 3
    assert len({call["mask_sha256"] for call in result["per_row_calls"]}) == 3


@pytest.mark.parametrize("field", ["duplicate", "complete", "stop", "unmatched", "invalid"])
def test_generated_outcome_boolean_strings_are_rejected(field: str) -> None:
    prefix = (9,)
    candidate = _generated_fixture(prefix, _row_tokens(), **{field: "false"})
    with pytest.raises(probe.TechnicalInvalid, match="actual JSON/Python boolean"):
        probe._coerce_row(
            candidate,
            parser=_parser(),
            owner_matcher=_description_owner_matcher({100: "owner"}),
            expected_prefix_token_ids=prefix,
            allow_test_adapter=True,
        )


def test_fabricated_owner_id_and_missing_evidence_fail_closed() -> None:
    prefix = (9,)
    valid = _generated_fixture(prefix, _row_tokens())
    with pytest.raises(probe.TechnicalInvalid, match="free caller owner_id"):
        probe._coerce_row(
            {**valid, "owner_id": "fabricated"},
            parser=_parser(),
            owner_matcher=_description_owner_matcher({100: "real-owner"}),
            expected_prefix_token_ids=prefix,
            allow_test_adapter=True,
        )
    with pytest.raises(probe.TechnicalInvalid, match="parser contract"):
        probe._coerce_row(
            valid,
            parser=None,
            owner_matcher=_description_owner_matcher({100: "owner"}),
            expected_prefix_token_ids=prefix,
            allow_test_adapter=True,
        )
    with pytest.raises(probe.TechnicalInvalid, match="matcher callback"):
        probe._coerce_row(
            valid,
            parser=_parser(),
            owner_matcher=None,
            expected_prefix_token_ids=prefix,
            allow_test_adapter=True,
        )
    with pytest.raises(probe.TechnicalInvalid, match="lacks a real-forward"):
        probe._coerce_row(
            {"token_ids": _row_tokens()},
            parser=_parser(),
            owner_matcher=_description_owner_matcher({100: "owner"}),
            expected_prefix_token_ids=prefix,
            allow_test_adapter=True,
        )


def test_test_adapter_requires_explicit_opt_in_and_production_dict_is_not_live() -> None:
    prefix = (9,)
    candidate = _generated_fixture(prefix, _row_tokens())
    with pytest.raises(probe.TechnicalInvalid, match="explicitly allowed"):
        probe._coerce_row(
            candidate,
            parser=_parser(),
            owner_matcher=_description_owner_matcher({100: "owner"}),
            expected_prefix_token_ids=prefix,
            allow_test_adapter=False,
        )
    fabricated_production = dict(_test_forward_receipt(prefix))
    fabricated_production.update(
        {"execution_adapter": "production", "real_forward": True, "test_adapter": False}
    )
    with pytest.raises(probe.TechnicalInvalid, match="live in-module"):
        probe._coerce_row(
            {"token_ids": _row_tokens(), "forward_receipt": fabricated_production},
            parser=_parser(),
            owner_matcher=_description_owner_matcher({100: "owner"}),
            expected_prefix_token_ids=prefix,
            allow_test_adapter=False,
        )


def test_receipt_boolean_strings_are_rejected() -> None:
    prefix = (9,)
    forward_receipt = _test_forward_receipt(prefix)
    forward_receipt["real_forward"] = "false"
    with pytest.raises(probe.TechnicalInvalid, match="actual JSON/Python boolean"):
        probe._coerce_row(
            {"token_ids": _row_tokens(), "forward_receipt": forward_receipt},
            parser=_parser(),
            owner_matcher=_description_owner_matcher({100: "owner"}),
            expected_prefix_token_ids=prefix,
            allow_test_adapter=True,
        )

    def string_boolean_matcher(_token_ids: object, _receipt: object) -> dict[str, object]:
        receipt = _owner_receipt("matched", "owner")
        receipt["source_specific"] = "false"
        return receipt

    with pytest.raises(probe.TechnicalInvalid, match="actual JSON/Python boolean"):
        probe._coerce_row(
            _generated_fixture(prefix, _row_tokens()),
            parser=_parser(),
            owner_matcher=string_boolean_matcher,
            expected_prefix_token_ids=prefix,
            allow_test_adapter=True,
        )


def test_unmatched_and_ambiguous_rows_are_neutral_and_duplicates_are_derived() -> None:
    rows = iter((_row_tokens(100), _row_tokens(101), _row_tokens(100)))

    def generate(prefix: tuple[int, ...], _row_index: int) -> dict[str, object]:
        return _generated_fixture(prefix, next(rows), duplicate=False)

    def matcher(token_ids: tuple[int, ...], _receipt: object) -> dict[str, object]:
        if token_ids[1] == 100:
            return _owner_receipt("matched", "A")
        return _owner_receipt("ambiguous")

    result = probe.run_native_horizon(
        (9,),
        generate,
        horizon=3,
        parser=_parser(),
        owner_matcher=matcher,
        covered_owner_ids=("A",),
        allow_test_adapter=True,
    )
    assert result["G"] == []
    assert result["K"] == ["A"]
    assert result["parse"]["ambiguous_rows"] == 1
    assert result["parse"]["duplicate_rows"] == 2


def test_positive_strict_fixture_uses_live_forward_and_derived_physical_owner() -> None:
    model = FakeModel()
    contract = _contract()
    native = probe.build_dynamic_arm("D00", prefix_width=0)
    _output, forward_receipt = probe.forward_with_dynamic_arm(
        model,
        _inputs(contract),
        prefix_contract=contract,
        arm=native,
    )

    def generate(prefix: tuple[int, ...], _row_index: int) -> dict[str, object]:
        return {"token_ids": _row_tokens(), "forward_receipt": forward_receipt}

    result = probe.run_native_horizon(
        contract.prefix_token_ids,
        generate,
        horizon=1,
        parser=_parser(),
        owner_matcher=_description_owner_matcher({100: "physical-owner"}),
    )
    assert result["G"] == ["physical-owner"]
    assert result["rows"][0]["owner_id"] == "physical-owner"
    assert result["rows"][0]["forward_receipt"]["real_forward"] is True


def test_evidence_backed_production_row_can_reenter_validation_seam() -> None:
    model = FakeModel()
    contract = _contract()
    _output, forward_receipt = probe.forward_with_dynamic_arm(
        model,
        _inputs(contract),
        prefix_contract=contract,
        arm=probe.build_dynamic_arm("D00", prefix_width=0),
    )
    matcher = _description_owner_matcher({100: "physical-owner"})
    row = probe._coerce_row(
        {"token_ids": _row_tokens(), "forward_receipt": forward_receipt},
        parser=_parser(),
        owner_matcher=matcher,
        expected_prefix_token_ids=contract.prefix_token_ids,
        allow_test_adapter=False,
    )
    checked_again = probe._coerce_row(
        row,
        parser=_parser(),
        owner_matcher=matcher,
        expected_prefix_token_ids=contract.prefix_token_ids,
        allow_test_adapter=False,
    )
    assert checked_again.owner_id == "physical-owner"

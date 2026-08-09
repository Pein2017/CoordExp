"""Mechanical CPU tests for the experiment-local static image-field probe."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.research import run_static_post_llm_image_field_probe as probe


IMAGE_TOKEN = 5
OBJECT_START = 100
OBJECT_END = 101
BOX_START = 102
BOX_END = 103
COMMIT = 104
COORD_START = 200


class Qwen3VLTextDecoderLayer(torch.nn.Module):
    _coordexp_decoder_layer = True

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        torch.nn.init.eye_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return hidden_states + self.proj(hidden_states)


class TinyLanguageModel(torch.nn.Module):
    def __init__(self, layer_count: int = 3) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Qwen3VLTextDecoderLayer() for _ in range(layer_count)]
        )
        self.norm = torch.nn.LayerNorm(4)


class TinyInner(torch.nn.Module):
    def __init__(self, layer_count: int = 3) -> None:
        super().__init__()
        self.language_model = TinyLanguageModel(layer_count)


class TinyQwen(torch.nn.Module):
    def __init__(self, layer_count: int = 3, vocab_size: int = 256) -> None:
        super().__init__()
        self.model = TinyInner(layer_count)
        self.embed = torch.nn.Embedding(vocab_size, 4)
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(
            image_token_id=IMAGE_TOKEN,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
        self.rope_grid_shapes: list[tuple[int, ...]] = []
        self.model.get_rope_index = self.get_rope_index

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        _video_grid_thw: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        self.rope_grid_shapes.append(tuple(image_grid_thw.shape))
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).reshape(1, -1)
        return positions.repeat(3, 1).unsqueeze(1), None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        hidden = self.embed(input_ids)
        for layer in self.model.language_model.layers:
            hidden = layer(hidden)
        hidden = self.model.language_model.norm(hidden)
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        suffix = input_ids[0, input_ids[0].tolist().index(OBJECT_START) + 1 :].tolist()
        sequence = [10, OBJECT_END, BOX_START, COORD_START, COORD_START + 1, COORD_START + 2, COORD_START + 3, BOX_END]
        if suffix and suffix[-1] == BOX_END:
            next_token = COMMIT
        else:
            next_token = sequence[len(suffix)] if len(suffix) < len(sequence) else COMMIT
        logits[:, -1, next_token] = 100.0
        return SimpleNamespace(logits=logits, last_hidden_state=hidden)


def _closed_contract() -> probe.WrapperContract:
    return probe.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=OBJECT_START,
        object_ref_end_token_id=OBJECT_END,
        box_start_token_id=BOX_START,
        box_end_token_id=BOX_END,
        coordinate_token_start_id=COORD_START,
    )


def _commit_contract() -> probe.WrapperContract:
    return probe.WrapperContract(
        assistant_format="object_box_commit",
        object_ref_start_token_id=OBJECT_START,
        object_ref_end_token_id=OBJECT_END,
        box_start_token_id=BOX_START,
        box_end_token_id=BOX_END,
        coordinate_token_start_id=COORD_START,
        commit_token_id=COMMIT,
    )


def _prefix() -> torch.Tensor:
    return torch.tensor([[9, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, OBJECT_START]])


def _span() -> probe.ImageSpan:
    return probe.derive_image_span(
        _prefix(), image_token_id=IMAGE_TOKEN, image_grid_thw=torch.tensor([[1, 4, 4]]), merge_size=2
    )


def _position_builder(*, model: object, input_ids: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
    del model, image_grid_thw
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).reshape(1, -1)
    return positions.repeat(3, 1).unsqueeze(1)


def _r00_factory(model: TinyQwen, *, layer_idx: int = 27):
    def factory(
        *,
        step: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        span: probe.ImageSpan,
    ) -> probe.MultiPositionResidualReplacement:
        del step
        capture = probe.capture_post_block_image_field(
            model,
            layer_idx=layer_idx,
            span=span,
            input_ids=input_ids,
            position_ids=position_ids,
        )
        module, _receipt = probe.resolve_decoder_layer(model, layer_idx)
        return probe.MultiPositionResidualReplacement(
            module,
            absolute_positions=span.absolute_positions,
            replacement=capture["state"],
            operator_arm="R00",
        )

    return factory


def _r10_factory(model: TinyQwen, *, layer_idx: int = 27):
    def factory(
        *,
        step: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        span: probe.ImageSpan,
    ) -> probe.MultiPositionResidualReplacement:
        del step
        capture = probe.capture_post_block_image_field(
            model,
            layer_idx=layer_idx,
            span=span,
            input_ids=input_ids,
            position_ids=position_ids,
        )
        module, _receipt = probe.resolve_decoder_layer(model, layer_idx)
        return probe.MultiPositionResidualReplacement(
            module,
            absolute_positions=span.absolute_positions,
            replacement=-capture["state"],
            operator_arm="R10",
        )

    return factory


def test_image_span_and_regions_are_exact() -> None:
    span = _span()
    assert span.absolute_positions == (1, 2, 3, 4)
    assert span.grid_indices == ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1))
    with pytest.raises(ValueError, match="placeholder count"):
        probe.derive_image_span(
            _prefix()[:, :-2], image_token_id=IMAGE_TOKEN, image_grid_thw=torch.tensor([[1, 4, 4]]), merge_size=2
        )
    assert probe.validate_disjoint_regions(
        span, exclusive_regions={"a": [0], "b": [1]}, shared_region=[2]
    )["passed"]


def test_wrapper_contracts_require_the_declared_closure() -> None:
    closed = _closed_contract()
    suffix = [10, OBJECT_END, BOX_START, COORD_START, COORD_START + 1, COORD_START + 2, COORD_START + 3, BOX_END]
    assert closed.parse_generated_suffix(suffix)["valid"]
    assert not closed.parse_generated_suffix([*suffix, COMMIT])["valid"]
    commit = _commit_contract()
    assert commit.parse_generated_suffix([*suffix, COMMIT])["valid"]
    assert commit.parse_generated_suffix(suffix)["reason"] == "missing_commit"
    assert commit.parse_generated_suffix([10, COMMIT, *suffix[1:], COMMIT])["reason"] == "premature_commit"


def test_row_query_mask_changes_only_declared_image_keys() -> None:
    span = _span()
    mask = probe.build_row_query_image_key_mask(
        sequence_length=8,
        image_key_positions=span.absolute_positions,
        eligible_image_positions=[span.absolute_positions[0]],
        query_position=7,
    )
    receipt = probe.inspect_row_query_image_key_mask(
        mask,
        sequence_length=8,
        image_key_positions=span.absolute_positions,
        eligible_image_positions=[span.absolute_positions[0]],
        query_position=7,
    )
    assert receipt["passed"]
    assert receipt["changed_off_query_cell_count"] == 0
    assert receipt["changed_non_image_cell_count"] == 0


def test_runtime_receipt_accepts_native_and_four_d_masks() -> None:
    ids = _prefix()
    positions = _position_builder(model=None, input_ids=ids, image_grid_thw=torch.tensor([[1, 4, 4]]))
    span = _span()
    native = probe.build_exact_runtime_receipt(
        prefix_ids=ids,
        position_ids=positions,
        span=span,
        runtime_contract=_closed_contract(),
        attention_mask=torch.ones_like(ids),
    )
    four_d = probe.build_exact_runtime_receipt(
        prefix_ids=ids,
        position_ids=positions,
        span=span,
        runtime_contract=_closed_contract(),
        attention_mask=torch.ones((1, 1, ids.shape[1], ids.shape[1]), dtype=torch.bool),
    )
    assert native["exact_prefix_token_ids_sha256"] == four_d["exact_prefix_token_ids_sha256"]


def test_all_layer_census_captures_image_positions_once() -> None:
    model = TinyQwen(layer_count=3)
    span = _span()
    with probe.ImageResidualCensus(
        model,
        span=span,
        layer_indices=(0, 1, 2),
        final_norm_module=model.model.language_model.norm,
    ) as census:
        model(input_ids=_prefix())
    receipt = census.receipt()
    assert receipt["call_counts"]["block_0_input"] == 1
    assert receipt["call_counts"]["block_2_output"] == 1
    assert receipt["call_counts"]["final_norm"] == 1
    assert receipt["image_span"]["token_count"] == 4


def test_post_block_helpers_cover_preregistered_layers_and_block27_sentinel() -> None:
    model = TinyQwen(layer_count=28)
    span = _span()
    input_ids = _prefix()
    native = probe.capture_post_block_image_field(
        model,
        layer_idx=13,
        span=span,
        input_ids=input_ids,
    )
    donor = probe.capture_post_block_image_field(
        model,
        layer_idx=23,
        span=span,
        input_ids=input_ids,
    )
    assert native["receipt"]["call_count"] == donor["receipt"]["call_count"] == 1
    replacement = probe.replace_post_block_image_field(
        model,
        layer_idx=23,
        absolute_positions=span.absolute_positions,
        replacement=donor["state"],
        input_ids=input_ids,
    )
    assert replacement["receipt"]["passed"]
    block27_native = probe.capture_post_block_image_field(
        model,
        layer_idx=27,
        span=span,
        input_ids=input_ids,
    )
    block27_replaced = probe.replace_post_block_image_field(
        model,
        layer_idx=27,
        absolute_positions=span.absolute_positions,
        replacement=block27_native["state"],
        input_ids=input_ids,
    )
    assert block27_replaced["receipt"]["passed"]
    assert torch.equal(block27_native["output"].logits, block27_replaced["output"].logits)
    with pytest.raises(ValueError, match="limited to preregistered"):
        probe.capture_post_block_image_field(model, layer_idx=12, span=span, input_ids=input_ids)


def test_multi_position_replacement_preserves_non_target_and_self_removes() -> None:
    model = TinyQwen(layer_count=1)
    layer = model.model.language_model.layers[0]
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    replacement = torch.full((2, 4), 7.0)
    hook = probe.MultiPositionResidualReplacement(
        layer, absolute_positions=(1, 4), replacement=replacement
    )
    with hook:
        output = layer(hidden)
    assert torch.equal(output[0, 1], replacement[0])
    assert hook.receipt()["passed"]
    assert hook.hook_removed_inside_hook
    assert hook.other_position_max_abs_delta == 0.0


def test_norm_matching_and_r_arm_construction() -> None:
    source = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    background = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    replacement, receipt = probe.norm_matched_replacement(source, background)
    assert receipt["passed"]
    assert torch.allclose(replacement[0], torch.tensor([5.0, 0.0]))
    with pytest.raises(ValueError, match="zero background"):
        probe.norm_matched_replacement(torch.tensor([[1.0, 0.0]]), torch.zeros((1, 2)))
    noop, noop_receipt = probe.build_residual_replacement(arm="R00", target_state=source)
    assert noop_receipt["passed"] and torch.equal(noop, source)


@pytest.mark.parametrize("assistant_format", ["object_box_closed", "object_box_commit"])
def test_complete_row_generation_is_full_prefix_and_not_teacher_forced(assistant_format: str) -> None:
    model = TinyQwen(layer_count=3)
    contract = _closed_contract() if assistant_format == "object_box_closed" else _commit_contract()
    result = probe.generate_complete_row(
        model,
        prefix_ids=_prefix(),
        image_token_id=IMAGE_TOKEN,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        merge_size=2,
        runtime_contract=contract,
        arm="K00",
        max_new_tokens=16,
        position_ids_builder=_position_builder,
    )
    assert result["complete_row"]
    assert result["cache_used"] is False
    assert result["recomputed_call_count"] == len(result["generated_token_ids"])
    assert result["parsed"]["valid"]


def test_complete_row_generation_normalizes_stored_single_image_grid_for_qwen_rope() -> None:
    model = TinyQwen(layer_count=3)
    stored_grid = torch.tensor([1, 4, 4])
    result = probe.generate_complete_row(
        model,
        prefix_ids=_prefix(),
        image_token_id=IMAGE_TOKEN,
        image_grid_thw=stored_grid,
        merge_size=2,
        runtime_contract=_closed_contract(),
        arm="K00",
        max_new_tokens=16,
    )
    assert result["complete_row"]
    assert model.rope_grid_shapes
    assert set(model.rope_grid_shapes) == {(1, 3)}
    assert stored_grid.shape == (3,)


def test_position_ids_accept_exact_multi_image_runs_and_reject_mismatch() -> None:
    model = TinyQwen(layer_count=1)
    ids = torch.tensor([[IMAGE_TOKEN, 7, IMAGE_TOKEN]], dtype=torch.long)
    grids = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)
    positions = probe._position_ids(
        model,
        input_ids=ids,
        image_grid_thw=grids,
        position_ids_builder=None,
    )
    assert positions.shape == (3, 1, 3)
    assert model.rope_grid_shapes == [(2, 3)]
    with pytest.raises(ValueError, match="image token run cardinality"):
        probe._position_ids(
            model,
            input_ids=torch.tensor([[IMAGE_TOKEN, IMAGE_TOKEN]], dtype=torch.long),
            image_grid_thw=grids,
            position_ids_builder=None,
        )


def test_k_matrix_and_noop_receipt() -> None:
    model = TinyQwen(layer_count=3)
    kwargs = {
        "prefix_ids": _prefix(),
        "image_token_id": IMAGE_TOKEN,
        "image_grid_thw": torch.tensor([1, 4, 4]),
        "merge_size": 2,
        "runtime_contract": _closed_contract(),
        "max_new_tokens": 16,
        "regions": {
            "b_exclusive": [0],
            "covered_a_exclusive": [1],
            "background": [2],
            "same_class_competitor": [3],
        },
    }
    arms = {
        arm: probe.generate_complete_row(model, arm=arm, **kwargs)
        for arm in ("K00", "K01", "K10", "K11", "K12", "K13")
    }
    assert all(result["complete_row"] for result in arms.values())
    assert model.rope_grid_shapes and set(model.rope_grid_shapes) == {(1, 3)}
    assert probe.compare_noop_receipts(arms["K00"], arms["K01"])["passed"]
    assert arms["K10"]["mask_receipts"][-1]["passed"]


def test_k13_not_applicable_and_block27_sentinel() -> None:
    model = TinyQwen(layer_count=28)
    calls = 0

    def count_forward(_module: torch.nn.Module, _args: tuple[object, ...], _kwargs: dict[str, object]) -> None:
        nonlocal calls
        calls += 1

    handle = model.register_forward_pre_hook(count_forward, with_kwargs=True)
    result = probe.generate_complete_row(
        model,
        prefix_ids=_prefix(),
        image_token_id=IMAGE_TOKEN,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        merge_size=2,
        runtime_contract=_closed_contract(),
        arm="K13",
        regions={"same_class_competitor": []},
        position_ids_builder=_position_builder,
    )
    handle.remove()
    assert result["status"] == "not_applicable"
    assert calls == 0
    baseline = probe.generate_complete_row(
        model,
        prefix_ids=_prefix(),
        image_token_id=IMAGE_TOKEN,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        merge_size=2,
        runtime_contract=_closed_contract(),
        arm="K00",
        position_ids_builder=_position_builder,
    )
    exact_self = probe.generate_complete_row(
        model,
        prefix_ids=_prefix(),
        image_token_id=IMAGE_TOKEN,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        merge_size=2,
        runtime_contract=_closed_contract(),
        arm="K00",
        position_ids_builder=_position_builder,
        residual_factory=_r00_factory(model),
        residual_arm="R00",
    )
    assert probe.compare_noop_receipts(baseline, exact_self)["passed"]
    sentinel = probe.assess_block27_sentinel(baseline, exact_self)
    assert sentinel["instrumentation_valid"]
    assert sentinel["mechanical_operator_valid"]
    assert not sentinel["behavioral_effect_detected"]
    changed_target = probe.generate_complete_row(
        model,
        prefix_ids=_prefix(),
        image_token_id=IMAGE_TOKEN,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        merge_size=2,
        runtime_contract=_closed_contract(),
        arm="K00",
        position_ids_builder=_position_builder,
        residual_factory=_r10_factory(model),
        residual_arm="R10",
    )
    r10_sentinel = probe.assess_block27_sentinel(baseline, changed_target)
    assert r10_sentinel["instrumentation_valid"]
    assert r10_sentinel["mechanical_operator_valid"]
    assert not r10_sentinel["behavioral_effect_detected"]
    changed_ids = list(exact_self["generated_token_ids"])
    changed_ids[0] += 1
    changed = dict(exact_self, generated_token_ids=changed_ids)
    changed["selected_token_log_probabilities"] = [
        value + 1.0 for value in exact_self["selected_token_log_probabilities"]
    ]
    assert not probe.assess_block27_sentinel(baseline, changed)["instrumentation_valid"]
    assert probe.assess_block27_sentinel(baseline, changed)["technical_invalid"]
    assert probe.assess_block27_sentinel(baseline, changed)["effect_detected"]


def test_native_tp_qualification_requires_target_movement() -> None:
    baseline = {
        "complete_row": True,
        "source_specific_physical_owner_match": True,
        "generated_token_ids": [1],
    }
    removed = {
        "complete_row": False,
        "source_specific_physical_owner_match": False,
        "generated_token_ids": [2],
    }
    receipt = probe.qualify_native_tp_actuator(baseline=baseline, intervention=removed)
    assert receipt["operator_qualified"]


def test_noop_comparison_fails_closed_on_empty_or_incomplete_receipts() -> None:
    empty = probe.compare_noop_receipts({}, {})
    assert not empty["passed"]
    assert empty["status"] == "invalid"
    model = TinyQwen(layer_count=3)
    valid = probe.generate_complete_row(
        model,
        prefix_ids=_prefix(),
        image_token_id=IMAGE_TOKEN,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        merge_size=2,
        runtime_contract=_closed_contract(),
        arm="K00",
        position_ids_builder=_position_builder,
    )
    missing_runtime = dict(valid)
    del missing_runtime["runtime_contract"]
    assert not probe.compare_noop_receipts(valid, missing_runtime)["passed"]
    key_noop = probe.generate_complete_row(
        model,
        prefix_ids=_prefix(),
        image_token_id=IMAGE_TOKEN,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        merge_size=2,
        runtime_contract=_closed_contract(),
        arm="K01",
        regions={},
        position_ids_builder=_position_builder,
    )
    malformed_hook = dict(
        key_noop,
        hook_counts={
            "forward_calls": "one",
            "residual_hook_calls": 0,
            "residual_receipt_count": 0,
        },
    )
    assert not probe.compare_noop_receipts(valid, malformed_hook)["passed"]


def test_residual_noop_lifecycle_fails_closed() -> None:
    model = TinyQwen(layer_count=28)
    common = {
        "prefix_ids": _prefix(),
        "image_token_id": IMAGE_TOKEN,
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "merge_size": 2,
        "runtime_contract": _closed_contract(),
        "arm": "K00",
        "position_ids_builder": _position_builder,
    }
    baseline = probe.generate_complete_row(model, **common)
    exact_self = probe.generate_complete_row(
        model,
        **common,
        residual_factory=_r00_factory(model),
        residual_arm="R00",
    )
    assert probe.compare_noop_receipts(baseline, exact_self)["passed"]

    def operator_mutation(**updates: object) -> dict[str, object]:
        return {
            **exact_self,
            "operator_receipt": {**exact_self["operator_receipt"], **updates},
        }

    wrong_arm = operator_mutation(arm="R11", kind="residual_owner_exclusive_swap")
    not_fired = operator_mutation(hook_fired_count=0)
    wrong_count = operator_mutation(hook_installed_count=exact_self["operator_receipt"]["expected_hook_count"] + 1)
    leaked = operator_mutation(cleanup_complete=False)
    target_drift = operator_mutation(target_tensor_max_abs_delta=1e-2)
    non_target_drift = {
        **operator_mutation(non_target_max_abs_delta=1e-2),
        "non_target_drift": 1e-2,
    }
    for invalid in (wrong_arm, not_fired, wrong_count, leaked, target_drift, non_target_drift):
        comparison = probe.compare_noop_receipts(baseline, invalid)
        assert not comparison["passed"]
        assert comparison["status"] == "invalid"


def test_native_tp_requires_explicit_source_specific_owner_evidence() -> None:
    baseline = {"complete_row": True, "owner_match": True}
    intervention = {"complete_row": False, "owner_match": False}
    indeterminate = probe.qualify_native_tp_actuator(baseline=baseline, intervention=intervention)
    assert indeterminate["status"] == "indeterminate"
    assert not indeterminate["operator_qualified"]
    malformed = probe.qualify_native_tp_actuator(
        baseline={"source_specific_physical_owner_match": "yes"},
        intervention={"source_specific_physical_owner_match": False},
    )
    assert malformed["status"] == "invalid"
    callback = probe.qualify_native_tp_actuator(
        baseline={"complete_row": False},
        intervention={"complete_row": True},
        target_matcher=lambda result: bool(result.get("physical_owner")),
    )
    assert callback["status"] == "unqualified"

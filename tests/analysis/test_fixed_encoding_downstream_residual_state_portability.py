from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.research.run_fixed_encoding_downstream_residual_state_portability import (
    BOX_START,
    OBJECT_REF_START,
    OBJECT_REF_END,
    RETURNED_LAYER_OUTPUT_SEAM,
    ResidualStateCapture,
    ResidualStateReplacement,
    assess_donor_eligibility,
    assess_portability_release,
    build_decoder_layer_resolution_receipt,
    build_request_identity,
    classify_portability_case,
    derive_boundary_position,
    derive_boundary_positions,
    greedy_cached_one_row_continuation,
    parse_generated_suffix,
    half_positive_persistent_release,
    resolve_decoder_layer,
    validate_parent_arm_mapping,
)


class FakeDecoderLayer(torch.nn.Module):
    _coordexp_decoder_layer = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 1.0


class FakeModel(torch.nn.Module):
    def __init__(self, count: int = 3) -> None:
        super().__init__()
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=torch.nn.ModuleList(FakeDecoderLayer() for _ in range(count))))


def test_resolve_decoder_layer_deduplicates_aliases_by_module_identity() -> None:
    model = FakeModel()
    layer, receipt = resolve_decoder_layer(model, 2)
    assert isinstance(layer, FakeDecoderLayer)
    assert receipt["layer_idx"] == 2
    assert receipt["module_path"] == "model.language_model.layers[2]"

    model.language_model = SimpleNamespace(layers=model.model.language_model.layers)
    layer, aliased_receipt = resolve_decoder_layer(model, 2)
    assert layer is model.model.language_model.layers[2]
    assert aliased_receipt["module_path"] == "model.language_model.layers[2]"
    assert aliased_receipt["module_alias_count"] == 2
    assert set(aliased_receipt["module_alias_paths"]) == {
        "model.language_model.layers[2]",
        "language_model.layers[2]",
    }


def test_decoder_layer_resolution_receipt_attests_returned_output_seam() -> None:
    receipt = build_decoder_layer_resolution_receipt({
        "layer_idx": 23,
        "module_path": "model.language_model.layers[23]",
        "module_alias_paths": ["model.language_model.layers[23]"],
        "module_alias_count": 1,
        "module_class": "Qwen3VLTextDecoderLayer",
    })
    assert receipt["layer_idx"] == 23
    assert receipt["module_path"] == "model.language_model.layers[23]"
    assert receipt["module_alias_paths"] == ["model.language_model.layers[23]"]
    assert receipt["module_alias_count"] == 1
    assert receipt["module_class"] == "Qwen3VLTextDecoderLayer"
    assert receipt["replacement_seam"] == RETURNED_LAYER_OUTPUT_SEAM
    assert receipt["seam_semantics"] == {
        "hook_kind": "forward_hook",
        "capture_tensor": "first_tensor_returned_by_decoder_block",
        "replacement_tensor": "first_tensor_returned_by_decoder_block",
        "timing": "after_full_decoder_block_forward",
        "position": "batch_index_0_boundary_position",
        "scope": "one_boundary_position_only",
    }


def test_request_identity_receipt_records_frozen_recipient_and_donors() -> None:
    identity = build_request_identity()
    assert identity == {
        "image_id": "139",
        "recipient": {"object_name": "vase", "annotation_id": "1669970"},
        "donors": [
            {"donor_name": "vase", "object_name": "vase", "annotation_id": "1669970"},
            {"donor_name": "clock", "object_name": "clock", "annotation_id": "1666628"},
        ],
    }


def test_resolve_decoder_layer_rejects_distinct_modules_at_same_index() -> None:
    model = FakeModel()
    model.language_model = SimpleNamespace(
        layers=torch.nn.ModuleList(FakeDecoderLayer() for _ in range(3))
    )
    with pytest.raises(ValueError, match="distinct decoder layer"):
        resolve_decoder_layer(model, 2)


def test_capture_hook_captures_one_returned_boundary_state_and_removes() -> None:
    layer = FakeDecoderLayer()
    capture = ResidualStateCapture(layer, boundary_pos=1)
    x = torch.zeros(1, 3, 4)
    with capture:
        out = layer(x)
    assert torch.equal(capture.state, torch.ones(4))
    assert capture.capture_count == 1
    assert out.shape == x.shape
    assert capture.handle is None


def test_capture_hook_rejects_second_forward() -> None:
    layer = FakeDecoderLayer()
    capture = ResidualStateCapture(layer, boundary_pos=0)
    capture.install()
    layer(torch.zeros(1, 2, 3))
    with pytest.raises(RuntimeError, match="more than once"):
        layer(torch.zeros(1, 2, 3))
    capture.remove()


def test_replacement_changes_only_one_position_and_removes_after_first_forward() -> None:
    layer = FakeDecoderLayer()
    replacement = ResidualStateReplacement(layer, boundary_pos=1, replacement=torch.full((4,), 9.0))
    x = torch.zeros(1, 3, 4)
    with replacement:
        out = layer(x)
        # The hook removed itself; later cached-token-like calls are untouched.
        second = layer(x)
    assert torch.equal(out[0, 1], torch.full((4,), 9.0))
    assert torch.equal(out[0, 0], torch.ones(4))
    assert torch.equal(out[0, 2], torch.ones(4))
    assert torch.equal(second, torch.ones_like(second))
    assert replacement.replacement_count == 1
    assert replacement.forward_count == 1
    assert replacement.other_position_max_abs_delta == 0.0
    assert replacement.handle is None
    assert replacement.hook_removed_inside_hook is True


def test_replacement_fails_if_boundary_is_invalid() -> None:
    layer = FakeDecoderLayer()
    replacement = ResidualStateReplacement(layer, boundary_pos=9, replacement=torch.ones(3))
    replacement.install()
    with pytest.raises(ValueError, match="outside"):
        layer(torch.zeros(1, 2, 3))
    replacement.remove()


def test_boundary_positions_are_absolute_and_roles_are_strict() -> None:
    first_prefix = [7, OBJECT_REF_START, 44, OBJECT_REF_START]
    pre_x1_prefix = [BOX_START, OBJECT_REF_START, 44, OBJECT_REF_END, BOX_START]
    assert derive_boundary_position(first_prefix, boundary_token_id=OBJECT_REF_START, role="first_description") == 3
    assert derive_boundary_position(pre_x1_prefix, boundary_token_id=BOX_START, role="pre_x1") == 4
    assert derive_boundary_positions(
        torch.tensor([first_prefix]), pre_x1_input_ids=torch.tensor([pre_x1_prefix])
    ) == {"first_description": 3, "pre_x1": 4}
    with pytest.raises(ValueError, match="final token"):
        derive_boundary_position([OBJECT_REF_START, 44, BOX_START], boundary_token_id=OBJECT_REF_START, role="first_description")
    with pytest.raises(ValueError, match="final token"):
        derive_boundary_position([7, 8], boundary_token_id=OBJECT_REF_START, role="first_description")
    with pytest.raises(ValueError, match="unsupported"):
        derive_boundary_position(first_prefix, boundary_token_id=OBJECT_REF_START, role="row_entry")


def test_donor_eligibility_requires_valid_owner_and_non_destructive_release() -> None:
    assert assess_donor_eligibility(persistent_release=0.2, valid_path=True, owner_match=True)["passed"]
    assert not assess_donor_eligibility(persistent_release=-0.06, valid_path=True, owner_match=True)["passed"]
    assert not assess_donor_eligibility(persistent_release=0.2, valid_path=True, owner_match=False)["passed"]
    assert not assess_donor_eligibility(persistent_release=0.2, valid_path=False, owner_match=True)["passed"]


def test_one_sided_half_positive_release_and_portability_gate() -> None:
    assert half_positive_persistent_release(-2.0) == 0.0
    assert half_positive_persistent_release(0.8) == pytest.approx(0.4)
    passed = assess_portability_release(
        persistent_release=0.8,
        replacement_release=0.5,
        no_op_drift=1e-4,
        valid_continuation=True,
        owner_path_match=True,
        path_phase="description",
    )
    assert passed["half_positive_persistent_release"] == pytest.approx(0.4)
    assert passed["owner_path_match"] is True
    assert passed["path_phase"] == "description"
    assert passed["passed"] is True
    assert not assess_portability_release(
        persistent_release=0.8,
        replacement_release=0.05,
        no_op_drift=1e-4,
        valid_continuation=True,
        owner_path_match=True,
        path_phase="geometry",
    )["passed"]


def test_parent_arm_mapping_is_frozen_and_does_not_substitute_parent_hard_endpoint() -> None:
    all_query = {
        "results": [{
            "image_id": "139",
            "target_annotation_id": "1669970",
            "competitor_annotation_id": "1666628",
            "arms": {"target_eligibility": {"target": {}}, "competitor_eligibility": {"competitor": {}}},
        }]
    }
    row_query = {
        "results": [{
            "image_id": "139",
            "target_annotation_id": "1669970",
            "competitor_annotation_id": "1666628",
            "arms": {},
        }]
    }
    mapping = validate_parent_arm_mapping(image_id="139", all_query=all_query, row_query=row_query)
    assert mapping["mapping"]["target"] == {"receipt": "all_query", "arm_name": "target_eligibility"}
    assert mapping["mapping"]["competitor"] == {"receipt": "all_query", "arm_name": "competitor_eligibility"}
    assert mapping["arms"]["target"] == all_query["results"][0]["arms"]["target_eligibility"]
    assert mapping["arms"]["competitor"] == all_query["results"][0]["arms"]["competitor_eligibility"]


def test_parent_arm_mapping_rejects_live_arm_without_nested_owner_score() -> None:
    with pytest.raises(ValueError, match="lacks nested target score"):
        validate_parent_arm_mapping(
            image_id="139",
            all_query={
                "results": [{
                    "image_id": "139",
                    "target_annotation_id": "1669970",
                    "competitor_annotation_id": "1666628",
                    "arms": {
                        "target_eligibility": {"competitor": {}},
                        "competitor_eligibility": {"competitor": {}},
                    },
                }],
            },
            row_query={"results": []},
        )


def test_parent_arm_mapping_fails_closed_when_arm_missing() -> None:
    with pytest.raises(ValueError, match="missing frozen arm"):
        validate_parent_arm_mapping(
            image_id="139",
            all_query={"results": [{"image_id": "139", "target_annotation_id": "1669970", "competitor_annotation_id": "1666628", "arms": {}}]},
            row_query={"results": []},
        )


def test_classification_and_stop_rules_fail_closed_before_interpretation() -> None:
    invalid = classify_portability_case(
        trust_gate_passed=False,
        semantic={"passed": True},
        geometry={"passed": True},
        eligible_count=1,
    )
    assert invalid == {"classification": "invalid_execution_trust_gate", "interpreted": False}
    closed = classify_portability_case(
        trust_gate_passed=True,
        semantic={"passed": False},
        geometry={"passed": False},
        eligible_count=1,
    )
    assert closed["classification"] == "close_one_site_conditional_downstream_portability"
    positive = classify_portability_case(
        trust_gate_passed=True,
        semantic={"passed": True},
        geometry={"passed": False},
        eligible_count=1,
    )
    assert positive["classification"] == "promote_bounded_one_sided_semantic_portability"
    geometry_only = classify_portability_case(
        trust_gate_passed=True,
        semantic={"passed": False},
        geometry={"passed": True},
        eligible_count=1,
    )
    assert geometry_only["classification"] == "promote_bounded_one_sided_geometry_portability"
    both = classify_portability_case(
        trust_gate_passed=True,
        semantic={"passed": True},
        geometry={"passed": True},
        eligible_count=1,
    )
    assert both["classification"] == "promote_phase_specific_semantic_and_geometry_portability"


class FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class FakeCachedModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.rope_deltas = torch.tensor([100, 200, 300])

    def prepare_inputs_for_generation(self, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("cached probe must bypass mutable generation preparation")

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        self.rope_deltas = torch.tensor([999, 998, 997])
        assert "rope_deltas" not in kwargs
        cache = kwargs["past_key_values"]
        assert isinstance(cache, FakeCache)
        positions = kwargs["position_ids"]
        assert isinstance(positions, torch.Tensor)
        assert tuple(positions.shape) == (3, 1, 1)
        cache.length += 1
        return SimpleNamespace(logits=torch.tensor([[[0.0, 10.0]]]), past_key_values=cache)


def test_cached_continuation_uses_recipient_cache_position_and_growing_mask() -> None:
    model = FakeCachedModel()
    output = SimpleNamespace(
        logits=torch.tensor([[[10.0, 0.0]]]),
        past_key_values=FakeCache(3),
    )
    result = greedy_cached_one_row_continuation(
        model,
        prefill_output=output,
        input_ids=torch.tensor([[1, 2, 3]]),
        prefill_position_ids=torch.tensor([[[0, 1, 2]], [[10, 11, 12]], [[20, 21, 22]]]),
        box_end_token_id=1,
        max_new_tokens=4,
    )
    assert result["generated_token_ids"] == [0, 1]
    assert result["cache_positions"] == [3]
    assert result["cached_call_count"] == 1
    assert model.calls[0]["cache_position"].tolist() == [3]
    assert model.calls[0]["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert model.calls[0]["position_ids"].tolist() == [[[3]], [[13]], [[23]]]
    assert result["prepare_inputs_for_generation_bypassed"] is True
    assert result["explicit_position_shapes"] == [[3, 1, 1]]


def test_cached_continuation_rejects_wrong_prefill_mrope_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \[3,1,S\]"):
        greedy_cached_one_row_continuation(
            FakeCachedModel(),
            prefill_output=SimpleNamespace(logits=torch.tensor([[[10.0, 0.0]]]), past_key_values=FakeCache(3)),
            input_ids=torch.tensor([[1, 2, 3]]),
            prefill_position_ids=torch.zeros(1, 3, dtype=torch.long),
            box_end_token_id=1,
        )


def test_strict_generated_suffix_parser_rejects_truncated_and_extra_rows() -> None:
    valid = parse_generated_suffix([44, 45, 151647, 151648, 151670, 151671, 151672, 151673, 151649])
    assert valid["valid"] is True
    assert valid["coordinate_bins"] == [0, 1, 2, 3]
    assert parse_generated_suffix([44, 151647])["reason"] == "missing_box_start"
    assert parse_generated_suffix([44, 151647, 151648, 151670])["reason"] == "truncated_coordinate_span"
    assert parse_generated_suffix([44, 151647, 151648, 151670, 151671, 151672, 151673, 151649, 12])["reason"] == "box_end_not_final"

from __future__ import annotations

import pytest
import torch
import numpy as np
from types import SimpleNamespace

from src.analysis.visual_support_counterfactual import (
    FeatureBundle,
    FeatureLayoutError,
    FeatureReplayController,
    PixelBounds,
    compose_rgb_patch,
    FreshFeatureCaptureController,
    materialize_rgb_uint8,
    build_merged_support_mask,
    choose_deterministic_control_mask,
    clone_feature_bundle,
    replace_selected_indices,
    validate_equal_mask_shape_and_count,
    validate_feature_layout,
    prune_symmetric_translation_overlap,
    rgb_array_sha256,
)
from src.inference.backend import DecodeResult, TokenTrace
from scripts.research.run_visual_support_counterfactual_commit import (
    _assert_frozen_pilot_geometry,
    _first_action_label,
    _require_noop_parity,
    _result_record,
)
from scripts.research.run_prevision_raw_pixel_visual_support_counterfactual_commit import (
    FROZEN_COHERENT_ROW_TOKEN_HASH,
    FROZEN_SEEDS,
    FROZEN_PROMPT_TOKEN_HASH,
    _assert_frozen_cli,
    _pixel_bounds_disjoint,
    summarize_paired_first_action_labels,
)


def _bundle(*, offset: float = 0.0, primary_container: type = tuple, deepstack_container: type = tuple):
    primary = [torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset]
    deepstack = [torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset + 100.0,
                 torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset + 200.0]
    return FeatureBundle.from_runtime(
        primary_container(primary) if primary_container is list else tuple(primary),
        deepstack_container(deepstack) if deepstack_container is list else tuple(deepstack),
    )


def _layout(recipient: FeatureBundle, donor: FeatureBundle):
    return validate_feature_layout(recipient, donor, grid_thw=(1, 4, 4), merge_size=2)


def test_bbox_mask_maps_merged_cells_and_halo() -> None:
    layout = _layout(_bundle(), _bundle(offset=1.0))
    mask = build_merged_support_mask(
        bbox_xyxy=(0.0, 0.0, 32.0, 32.0),
        image_width=64,
        image_height=64,
        layout=layout,
        halo=1,
    )
    # The upper-left merged cell plus one halo covers the complete 2x2 grid.
    assert mask.tolist() == [True, True, True, True]
    assert int(mask.sum()) == 4


def test_control_mask_is_equal_count_and_avoids_forbidden_region() -> None:
    layout = _layout(_bundle(), _bundle(offset=1.0))
    target = build_merged_support_mask(
        bbox_xyxy=(0.0, 0.0, 32.0, 32.0),
        image_width=64,
        image_height=64,
        layout=layout,
        halo=0,
    )
    forbidden = target.clone()
    control = choose_deterministic_control_mask(
        target,
        temporal=layout.temporal,
        merged_height=layout.merged_height,
        merged_width=layout.merged_width,
        forbidden_mask=forbidden,
    )
    validate_equal_mask_shape_and_count(target, control)
    assert not bool((target & control).any())


def test_control_mask_fails_when_no_translation_exists() -> None:
    layout = _layout(_bundle(), _bundle(offset=1.0))
    target = torch.ones(layout.primary_token_count, dtype=torch.bool)
    with pytest.raises(FeatureLayoutError, match="control mask"):
        choose_deterministic_control_mask(
            target,
            temporal=layout.temporal,
            merged_height=layout.merged_height,
            merged_width=layout.merged_width,
            forbidden_mask=target,
        )


def test_symmetric_overlap_pruning_removes_same_relative_cells() -> None:
    target = torch.zeros(1 * 4 * 6, dtype=torch.bool)
    control = torch.zeros_like(target)
    # A 2x2 translated shape overlaps at one physical cell.
    target.reshape(1, 4, 6)[0, 1:3, 1:3] = True
    control.reshape(1, 4, 6)[0, 0:2, 0:2] = True
    target_final, control_final, receipt = prune_symmetric_translation_overlap(
        target,
        control,
        temporal=1,
        merged_height=4,
        merged_width=6,
    )
    assert receipt["physical_overlap_target_relative_positions"] == [[0, 0]]
    assert receipt["physical_overlap_control_relative_positions"] == [[1, 1]]
    assert receipt["removed_relative_positions"] == [[1, 1]]
    assert int(target_final.sum()) == int(control_final.sum()) == 3
    assert not bool((target_final & control_final).any())


def test_frozen_pilot_geometry_guard_accepts_reviewed_layout() -> None:
    recipient = FeatureBundle(
        primary=(torch.zeros(972, 1),),
        deepstack=tuple(torch.zeros(972, 1) for _ in range(3)),
    )
    donor = FeatureBundle(
        primary=(torch.ones(972, 1),),
        deepstack=tuple(torch.ones(972, 1) for _ in range(3)),
    )
    layout = validate_feature_layout(
        recipient,
        donor,
        grid_thw=(1, 72, 54),
        merge_size=2,
    )
    target = torch.zeros(972, dtype=torch.bool).reshape(1, 36, 27)
    control = torch.zeros_like(target)
    target[0, 10:21, 6:14] = True
    control[0, 0:11, 0:8] = True
    target_final, control_final, pruning = prune_symmetric_translation_overlap(
        target.reshape(-1),
        control.reshape(-1),
        temporal=1,
        merged_height=36,
        merged_width=27,
    )
    receipt = _assert_frozen_pilot_geometry(
        layout=layout,
        recipient_features=recipient,
        donor_features=donor,
        target_mask=target_final,
        control_mask=control_final,
        control_pruning=pruning,
    )
    assert receipt["verified"] is True
    assert receipt["base_mask_count"] == 88
    assert receipt["final_mask_count"] == 86
    assert receipt["removed_control_relative_positions"] == [[10, 6], [10, 7]]


def test_frozen_pilot_geometry_guard_rejects_drift() -> None:
    recipient = FeatureBundle(
        primary=(torch.zeros(972, 1),),
        deepstack=tuple(torch.zeros(972, 1) for _ in range(3)),
    )
    donor = recipient.clone()
    layout = validate_feature_layout(
        recipient,
        donor,
        grid_thw=(1, 72, 54),
        merge_size=2,
    )
    mask = torch.zeros(972, dtype=torch.bool)
    with pytest.raises(SystemExit, match="frozen visual-support pilot geometry contract failed"):
        _assert_frozen_pilot_geometry(
            layout=layout,
            recipient_features=recipient,
            donor_features=donor,
            target_mask=mask,
            control_mask=mask.clone(),
            control_pruning={
                "target_base_indices": list(range(88)),
                "control_base_indices": list(range(88)),
                "removed_relative_positions": [[10, 6], [10, 7]],
                "disjoint": True,
            },
        )


def test_action_label_requires_phrase_geometry_owner_consistency() -> None:
    raw = SimpleNamespace(
        image=SimpleNamespace(width=100, height=100),
        objects=(
            SimpleNamespace(object_id="678923", description="cup", bbox=(0, 0, 999, 999)),
            SimpleNamespace(object_id="678023", description="cup", bbox=(0, 0, 500, 500)),
            SimpleNamespace(object_id="1571077", description="pizza", bbox=(500, 500, 999, 999)),
        ),
    )
    assert _first_action_label(
        {"status": "valid_row", "description": "cup", "bbox_xyxy": [0, 0, 100, 100]},
        raw,
    ) == "right_cup"
    assert _first_action_label(
        {"status": "valid_row", "description": "pizza", "bbox_xyxy": [0, 0, 50, 50]},
        raw,
    ) == "chimera_phrase_geometry"
    assert _first_action_label(
        {"status": "valid_row", "description": "cup", "bbox_xyxy": [0, 0, 50, 50]},
        raw,
    ) == "left_cup_revisit"


def test_result_record_uses_decode_result_serializer_for_token_trace() -> None:
    class _Tokenizer:
        def decode(self, _ids, **_kwargs):
            return ""

    result = DecodeResult(
        request_id="serializer-regression",
        backend="huggingface",
        backend_mode="greedy",
        response_family="qwen3_vl",
        prompt_token_ids=[1],
        generated_token_ids=[2],
        raw_generated_text="",
        parser_text="",
        strip_policy="none",
        stop_reason="max_new_tokens",
        model_identity={},
        tokenizer_identity={},
        generation_config_fingerprint="generation-fingerprint",
        token_trace=[
            TokenTrace(
                step_index=0,
                token_id=2,
                token_text="x",
                logprob=-0.5,
                is_stop=False,
                is_pad=False,
                backend="huggingface",
                backend_mode="greedy",
                response_family="qwen3_vl",
            )
        ],
        execution_receipt=None,
    )
    raw = SimpleNamespace(
        image=SimpleNamespace(width=100, height=100),
        objects=(),
    )
    record = _result_record(
        result,
        _Tokenizer(),
        raw,
        None,
        "serializer-regression",
        right_cup_object_id="right",
        target_object_id="target",
        target_pizza_object_id="pizza",
    )
    assert record["token_trace"] == [
        {
            "step_index": 0,
            "token_id": 2,
            "token_text": "x",
            "logprob": -0.5,
            "is_stop": False,
            "is_pad": False,
            "backend": "huggingface",
            "backend_mode": "greedy",
            "response_family": "qwen3_vl",
        }
    ]


def test_noop_parity_returns_durable_token_and_float32_hash_receipt() -> None:
    score_receipt = SimpleNamespace(canonical_float32_score_trace_hash="score-hash")
    standard = SimpleNamespace(
        generated_token_ids=[11, 12, 13],
        execution_receipt=score_receipt,
    )
    replay = SimpleNamespace(
        generated_token_ids=[11, 12, 13],
        execution_receipt=SimpleNamespace(canonical_float32_score_trace_hash="score-hash"),
    )
    parity = _require_noop_parity(standard, replay, scope="test-greedy")
    assert parity["verified"] is True
    assert parity["scope"] == "test-greedy"
    assert parity["generated_token_count"] == 3
    assert parity["generated_token_ids_equal"] is True
    assert parity["canonical_float32_score_trace_hash_equal"] is True
    assert parity["standard_generated_token_ids_sha256"] == parity["replay_generated_token_ids_sha256"]
    assert parity["standard_canonical_float32_score_trace_hash"] == "score-hash"


def test_replacement_updates_primary_and_every_deepstack_and_preserves_complement() -> None:
    recipient = _bundle()
    donor = _bundle(offset=1000.0)
    selected = torch.tensor([False, True, False, True])
    replaced, receipt = replace_selected_indices(recipient, donor, selected)
    assert torch.equal(replaced.primary[0][selected], donor.primary[0][selected])
    for left, right in zip(replaced.deepstack, donor.deepstack, strict=True):
        assert torch.equal(left[selected], right[selected])
    complement = ~selected
    assert torch.equal(replaced.primary[0][complement], recipient.primary[0][complement])
    assert receipt["complement_preserved"] is True
    assert receipt["selected_count"] == 2
    stream_receipts = receipt["streams"]
    assert isinstance(stream_receipts, list)
    assert len(stream_receipts) == 3
    for stream_receipt in stream_receipts:
        assert stream_receipt["selected_slice_exact_equal"] is True
        assert stream_receipt["selected_slice_canonical_float32_max_abs_diff"] == 0.0
        assert (
            stream_receipt["donor_selected_slice_sha256"]
            == stream_receipt["substituted_selected_slice_sha256"]
        )
        assert stream_receipt["donor_selected_slice"]["sha256"] == stream_receipt[
            "donor_selected_slice_sha256"
        ]
        assert stream_receipt["substituted_selected_slice"]["sha256"] == stream_receipt[
            "substituted_selected_slice_sha256"
        ]


def test_replacement_rejects_unequal_mask_shape_or_count() -> None:
    recipient = _bundle()
    donor = _bundle(offset=1000.0)
    with pytest.raises(FeatureLayoutError, match="shape"):
        validate_equal_mask_shape_and_count(
            torch.tensor([True, False]), torch.tensor([True, False, False])
        )
    with pytest.raises(FeatureLayoutError, match="count"):
        validate_equal_mask_shape_and_count(
            torch.tensor([True, False, False]), torch.tensor([True, True, False])
        )
    with pytest.raises(FeatureLayoutError, match="shape"):
        validate_feature_layout(
            recipient,
            FeatureBundle.from_runtime(
                (torch.zeros(5, 4),),
                (torch.zeros(4, 4), torch.zeros(4, 4)),
            ),
            grid_thw=(1, 4, 4),
            merge_size=2,
        )


def test_clone_preserves_values_and_supports_list_inputs() -> None:
    original = _bundle(primary_container=list, deepstack_container=list)
    cloned = clone_feature_bundle(list(original.primary), list(original.deepstack))
    assert isinstance(cloned.primary, tuple)
    assert isinstance(cloned.deepstack, tuple)
    assert all(torch.equal(left, right) for left, right in zip(original.primary, cloned.primary, strict=True))
    assert all(torch.equal(left, right) for left, right in zip(original.deepstack, cloned.deepstack, strict=True))
    cloned.primary[0][0, 0] = -1
    assert original.primary[0][0, 0] != -1


class _FakeVisionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = 0

    def get_image_features(self, pixel_values, image_grid_thw):
        self.calls += 1
        del pixel_values, image_grid_thw
        return [torch.zeros(4, 4)], [torch.ones(4, 4), torch.ones(4, 4) * 2]


class _FakeQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeVisionModel()


def test_replay_controller_restores_method_and_validates_call() -> None:
    qwen = _FakeQwen()
    recipient = _bundle()
    donor = _bundle(offset=1000.0)
    selected = torch.tensor([True, False, False, False])
    original_method = qwen.model.get_image_features
    controller = FeatureReplayController(
        model=qwen,
        recipient=recipient,
        donor=donor,
        grid_thw=(1, 4, 4),
        merge_size=2,
        selected_mask=selected,
        mode="target",
    )
    with controller:
        primary, deepstack = qwen.model.get_image_features(
            torch.zeros(1), torch.tensor([[1, 4, 4]])
        )
        # Replay uses Qwen-compatible tuple containers; it does not promise
        # to preserve the fake tower's list return type.
        assert isinstance(primary, tuple)
        assert isinstance(deepstack, tuple)
        assert torch.equal(primary[0][0], donor.primary[0][0])
        assert torch.equal(deepstack[0][0], donor.deepstack[0][0])
    receipt = controller.validate_completed(expected_feature_calls=1)
    assert receipt["hook_restored"] is True
    assert qwen.model.get_image_features == original_method


def test_replay_controller_restores_method_on_exception() -> None:
    qwen = _FakeQwen()
    original_method = qwen.model.get_image_features
    controller = FeatureReplayController(
        model=qwen,
        recipient=_bundle(),
        donor=_bundle(offset=1000.0),
        grid_thw=(1, 4, 4),
        merge_size=2,
        mode="clean",
    )
    with pytest.raises(RuntimeError):
        with controller:
            raise RuntimeError("boom")
    assert qwen.model.get_image_features == original_method


def test_raw_pixel_compositor_replaces_exact_half_open_slice_and_preserves_complement() -> None:
    recipient = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    donor = np.flip(recipient, axis=1).copy()
    bounds = PixelBounds(2, 1, 6, 5)
    composed, receipt = compose_rgb_patch(
        recipient,
        donor,
        bounds=bounds,
        mode="target",
    )
    expected = recipient.copy()
    expected[1:5, 2:6] = donor[1:5, 2:6]
    assert np.array_equal(composed, expected)
    assert receipt["selected_slice_exact_equal_to_donor"] is True
    assert receipt["complement_exact_equal_to_recipient"] is True
    assert receipt["clean_exact_rgb_copy"] is False
    assert receipt["complement_rgb_sha256_equal"] is True
    assert receipt["recipient_complement_rgb_sha256"] == receipt["composed_complement_rgb_sha256"]
    assert receipt["bounds_xyxy_half_open"] == [2, 1, 6, 5]
    assert receipt["bounds_shape"] == [4, 4]
    assert receipt["bounds_pixel_count"] == 16
    assert receipt["float32_pixel_difference"]["float32"] is True


def test_raw_pixel_clean_compositor_is_exact_copy_and_empty_mask() -> None:
    recipient = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    donor = np.zeros_like(recipient)
    composed, receipt = compose_rgb_patch(recipient, donor, bounds=None, mode="clean")
    assert np.array_equal(composed, recipient)
    assert receipt["mask_pixel_count"] == 0
    assert receipt["selected_slice_exact_equal_to_donor"] is True
    assert receipt["complement_exact_equal_to_recipient"] is True
    assert receipt["clean_exact_rgb_copy"] is True
    assert receipt["complement_rgb_sha256_equal"] is True
    assert receipt["float32_pixel_difference"] is None


def test_raw_pixel_compositor_rejects_resize_or_bad_bounds() -> None:
    recipient = np.zeros((4, 5, 3), dtype=np.uint8)
    donor = np.zeros((4, 5, 3), dtype=np.uint8)
    with pytest.raises(FeatureLayoutError, match="exceed image dimensions"):
        compose_rgb_patch(recipient, donor, bounds=PixelBounds(0, 0, 6, 2), mode="target")
    with pytest.raises(FeatureLayoutError, match="shapes differ"):
        compose_rgb_patch(recipient, np.zeros((5, 5, 3), dtype=np.uint8), bounds=PixelBounds(0, 0, 1, 1), mode="target")


def test_fresh_feature_capture_controller_observes_live_outputs_and_restores() -> None:
    qwen = _FakeQwen()
    original_method = qwen.model.get_image_features
    controller = FreshFeatureCaptureController(model=qwen, expected_grid_thw=(1, 4, 4))
    with controller:
        primary, deepstack = qwen.model.get_image_features(
            torch.zeros(4, 4), torch.tensor([[1, 4, 4]])
        )
        assert torch.equal(primary[0], torch.zeros(4, 4))
        assert len(deepstack) == 2
    receipt = controller.validate_completed(expected_feature_calls=1)
    assert receipt["fresh_visual_recomputation"] is True
    assert receipt["calls"][0]["primary"][0]["shape"] == [4, 4]
    assert qwen.model.get_image_features == original_method
    assert controller.receipt()["hook_restored"] is True


def test_pixel_bounds_disjointness_uses_half_open_geometry() -> None:
    assert _pixel_bounds_disjoint(PixelBounds(0, 0, 2, 2), PixelBounds(2, 0, 4, 2)) is True
    assert _pixel_bounds_disjoint(PixelBounds(0, 0, 3, 2), PixelBounds(2, 0, 4, 2)) is False


def test_frozen_cli_rejects_research_identity_overrides() -> None:
    args = SimpleNamespace(
        recipient_prefix_token_count=56,
        max_new_tokens=512,
        sampling_temperature=0.4,
        donor_image_id="17436",
        target_object_id="678023",
        right_cup_object_id="678923",
        target_pizza_object_id="1571077",
    )
    _assert_frozen_cli(args)
    for field, value in (
        ("recipient_prefix_token_count", 55),
        ("max_new_tokens", 511),
        ("sampling_temperature", 0.5),
        ("target_object_id", "other"),
    ):
        broken = SimpleNamespace(**vars(args))
        setattr(broken, field, value)
        with pytest.raises(SystemExit, match="frozen pre-vision panel arguments mismatch"):
            _assert_frozen_cli(broken)


def test_frozen_identity_hash_constants_are_explicit() -> None:
    assert len(FROZEN_PROMPT_TOKEN_HASH) == 64
    assert len(FROZEN_COHERENT_ROW_TOKEN_HASH) == 64


def _gate(**extra):
    inputs = {
        "frozen_cli_arguments": True,
        "target_control_disjoint": True,
        "pixel_selected_and_complement_exact": True,
        "processor_no_resize_grid": True,
        "processor_hashes_distinct": True,
        "clean_processor_parity": True,
        "greedy_noop_parity": True,
        "sampled_noop_parity": True,
        "fresh_primary_and_three_deepstack_recomputation": True,
    }
    inputs.update(extra)
    return {"status": "passed", "input": inputs}


def test_paired_summary_positive_precedes_null_and_counts_supported_alternatives() -> None:
    clean = ["right_cup"] * 8
    control = ["right_cup"] * 8
    target = ["left_cup_revisit", "target_pizza_fallback", "another_supported_object", "left_cup_revisit", "target_pizza_fallback", "another_supported_object", "right_cup", "right_cup"]
    summary = summarize_paired_first_action_labels(clean, target, control, seeds=FROZEN_SEEDS, trust_gate=_gate())
    assert summary["verdict"] == "positive"
    assert summary["target_selective_count"] == 6
    assert summary["reverse_selective_count"] == 0
    assert summary["supported_valid_alternative_count"] == 6
    assert len(summary["per_seed"]) == 8
    assert summary["condition_right_cup_fractions"]["target"] == 0.25


def test_paired_summary_strong_null_is_selected_after_positive_fails() -> None:
    labels = ["right_cup"] * 8
    summary = summarize_paired_first_action_labels(labels, labels, labels, seeds=FROZEN_SEEDS, trust_gate=_gate())
    assert summary["verdict"] == "strong_null"
    assert summary["target_selective_count"] == 0
    assert summary["reverse_selective_count"] == 0


def test_paired_summary_trust_failure_precedes_all_other_verdicts() -> None:
    summary = summarize_paired_first_action_labels(
        ["right_cup"] * 8,
        ["left_cup_revisit"] * 8,
        ["right_cup"] * 8,
        seeds=FROZEN_SEEDS,
        trust_gate={"status": "failed", "input": {"processor_no_resize_grid": False}},
    )
    assert summary["verdict"] == "inconclusive"
    assert summary["trust_gate"]["status"] == "failed"


def test_paired_summary_greedy_only_is_not_evaluated() -> None:
    summary = summarize_paired_first_action_labels(
        ["right_cup"], ["right_cup"], ["right_cup"], trust_gate=_gate()
    )
    assert summary["verdict"] == "not_evaluated"


def test_paired_summary_inconclusive_when_target_losses_are_not_supported_alternatives() -> None:
    clean = ["right_cup"] * 8
    control = ["right_cup"] * 8
    target = ["invalid", "terminal", "chimera_phrase_geometry", "valid_row_unmatched_geometry", "invalid", "terminal", "right_cup", "right_cup"]
    summary = summarize_paired_first_action_labels(clean, target, control, seeds=FROZEN_SEEDS, trust_gate=_gate())
    assert summary["verdict"] == "inconclusive"

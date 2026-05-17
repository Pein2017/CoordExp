from __future__ import annotations

from src.bootstrap.stage2_policy_provenance import build_stage2_policy_provenance
from src.config.loader import ConfigLoader


def test_stage2_policy_provenance_records_effective_legacy_threshold() -> None:
    provenance = build_stage2_policy_provenance(
        {
            "custom": {
                "trainer_variant": "stage2_two_channel",
                "object_ordering": "sorted",
            },
            "stage2_ab": {
                "channel_b": {
                    "assignment": {"strategy": "legacy_hungarian_mask_iou"},
                    "duplicate_control": {
                        "iou_threshold": 0.91,
                        "center_radius_scale": 0.75,
                    },
                    "insertion_order": "tail_append",
                    "rollout_template_family": "compact_full",
                    "rollout_decode_policy": "unconstrained",
                    "invalid_rollout_policy": "fallback_to_gt_fn_append",
                }
            },
            "rollout_matching": {"maskiou_gate": 0.55},
        }
    )

    assert provenance is not None
    assert provenance["assignment_strategy"] == "legacy_hungarian_mask_iou"
    assert provenance["assignment_iou_threshold"] is None
    assert provenance["assignment_iou_threshold_effective"] == 0.55
    assert provenance["assignment_iou_threshold_source"] == "rollout_matching.maskiou_gate"
    assert provenance["duplicate_filter_strategy"] == "legacy_channel_b_duplicate_control"
    assert provenance["duplicate_iou_threshold"] == 0.91
    assert provenance["duplicate_center_radius_scale"] == 0.75
    assert provenance["object_ordering_policy"] == "tail_append"
    assert provenance["object_ordering_strategy_id"] == "legacy_tail_append"
    assert provenance["sample_object_ordering"] == "sorted"
    assert provenance["rollout_template_family"] == "compact_full"
    assert provenance["rollout_decode_policy"] == "unconstrained"
    assert provenance["invalid_rollout_policy"] == "fallback_to_gt_fn_append"


def test_stage2_policy_provenance_prefers_explicit_assignment_threshold() -> None:
    provenance = build_stage2_policy_provenance(
        {
            "custom": {"trainer_variant": "stage2_two_channel"},
            "stage2_ab": {
                "channel_b": {
                    "assignment": {
                        "strategy": "greedy_iou",
                        "iou_threshold": 0.61,
                    },
                    "insertion_order": "sorted",
                }
            },
            "rollout_matching": {"maskiou_gate": 0.31},
        }
    )

    assert provenance is not None
    assert provenance["assignment_strategy"] == "greedy_iou"
    assert provenance["assignment_iou_threshold"] == 0.61
    assert provenance["assignment_iou_threshold_effective"] == 0.61
    assert (
        provenance["assignment_iou_threshold_source"]
        == "stage2_ab.channel_b.assignment.iou_threshold"
    )
    assert provenance["object_ordering_policy"] == "sorted"
    assert provenance["object_ordering_strategy_id"] == "top_left_spatial"


def test_stage2_policy_provenance_omits_non_stage2_two_channel_runs() -> None:
    provenance = build_stage2_policy_provenance(
        {"custom": {"trainer_variant": "sft"}},
        trainer_variant="sft",
    )

    assert provenance is None


def test_stage2_policy_provenance_reads_real_typed_stage2_config() -> None:
    cfg = ConfigLoader.load_materialized_training_config(
        "configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_gate2_16sample.yaml"
    )

    provenance = build_stage2_policy_provenance(
        cfg,
        trainer_variant="stage2_two_channel",
    )

    assert provenance is not None
    assert provenance["assignment_strategy"] == "legacy_hungarian_mask_iou"
    assert provenance["assignment_iou_threshold"] is None
    assert provenance["assignment_iou_threshold_effective"] == 0.5
    assert provenance["assignment_iou_threshold_source"] == "rollout_matching.maskiou_gate"
    assert provenance["duplicate_filter_strategy"] == "legacy_channel_b_duplicate_control"
    assert provenance["duplicate_iou_threshold"] == 0.95
    assert provenance["duplicate_center_radius_scale"] == 0.80
    assert provenance["object_ordering_policy"] == "tail_append"
    assert provenance["object_ordering_strategy_id"] == "legacy_tail_append"
    assert provenance["sample_object_ordering"] == "random"
    assert provenance["rollout_template_family"] == "compact_full"
    assert provenance["rollout_decode_policy"] == "unconstrained"
    assert provenance["invalid_rollout_policy"] == "fallback_gt_fn_append_only"

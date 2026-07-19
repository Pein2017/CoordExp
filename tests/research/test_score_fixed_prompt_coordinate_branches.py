"""Pure validation tests for fixed-prompt coordinate branch scoring."""

from __future__ import annotations

import copy

import pytest

from scripts.research.score_fixed_prompt_coordinate_branches import (
    COORDINATE_SLOT_NAMES,
    pair_coordinate_branch_reports,
    summarize_coordinate_logits,
    validate_bundle_pair,
)


def _bundle(*, image_id: str = "12670", category: str = "person", prompt_fingerprint: str = "prompt", role: str = "clean") -> dict:
    clean = [151700, 151760, 151820, 151900]
    bad = [151700, 151760, 151820, 151920]
    selected = clean if role == "clean" else bad
    generated = [151646, 123, 151647, 151648, *selected, 151649]
    return {
        "request_id": f"request-{role}",
        "scheduled_request": {
            "image_id": int(image_id),
            "arm": {
                "arm_code": "FULL_BAG_K",
                "history_policy": "fresh_base_prompt_per_call",
            },
        },
        "execution_evidence": {
            "image_id": int(image_id),
            "source_image_sha256": "rgb-sha",
            "arm": {
                "arm_code": "FULL_BAG_K",
                "history_policy": "fresh_base_prompt_per_call",
            },
            "executed_prompt_evidence": {
                "full_prompt_fingerprint": prompt_fingerprint,
                "prompt_token_ids": [1, 2, 3],
            },
        },
        "decode_result": {
            "prompt_token_ids": [1, 2, 3],
            "generated_token_ids": generated,
            "model_identity": {
                "family": "base-plus-adapter-plus-delta",
                "base": {"path": "/models/base"},
                "adapter": {
                    "adapter_path": "/models/adapter",
                    "adapter_type": "dora",
                    "adapter_name": "default",
                },
                "embedding_delta": {"identity": {"delta_path": "/models/delta"}},
            },
        },
        "parse_score_receipts": [
            {
                "parse_row_index": 0,
                "generated_row_index": 0,
                "normalized_category_name": category,
                "coordinate_bins": [100, 200, 300, 400],
                "selected_token_ids": [151646, 151647, 151648, *selected, 151649],
                "selected_generated_step_indices": [0, 2, 3, 4, 5, 6, 7, 8],
            }
        ],
    }


def _case() -> dict:
    return {
        "name": "image-12670-person",
        "image_id": 12670,
        "category": "person",
        "role": "primary",
        "reference_box_xyxy": [10, 20, 30, 40],
    }


def test_validate_bundle_pair_accepts_same_prompt_and_first_coordinate_difference() -> None:
    result = validate_bundle_pair(
        _case(),
        _bundle(role="clean"),
        _bundle(role="bad"),
        expected_model_composition={
            "base_model_path": "/models/base",
            "adapter_path": "/models/adapter",
            "adapter_type": "dora",
            "adapter_name": "default",
            "embedding_delta_path": "/models/delta",
        },
    )
    assert result["first_differing_coordinate_slot"] == "y2"
    assert result["shared_generated_prefix_token_ids"][-1] == 151820
    assert result["source_model_composition"]["adapter_path"] == "/models/adapter"
    assert result["executed_model_composition"]["adapter_path"] == "/models/adapter"
    assert result["counterfactual_model_composition"] == {
        "enabled": False,
        "source_and_executed_composition_differ": False,
        "purpose": None,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda b: b["execution_evidence"]["arm"].update({"arm_code": "FULL_SINGLE"}), "Full-Image"),
        (lambda b: b["execution_evidence"]["executed_prompt_evidence"].update({"full_prompt_fingerprint": "other"}), "full_prompt"),
        (lambda b: b["parse_score_receipts"][0].update({"generated_row_index": 1}), "parse row"),
    ],
)
def test_validate_bundle_pair_rejects_non_comparable_pairs(change, message: str) -> None:
    clean = _bundle(role="clean")
    bad = _bundle(role="bad")
    change(bad)
    with pytest.raises(ValueError, match=message):
        validate_bundle_pair(
            _case(),
            clean,
            bad,
            expected_model_composition={
                "base_model_path": "/models/base",
                "adapter_path": "/models/adapter",
                "adapter_type": "dora",
                "adapter_name": "default",
                "embedding_delta_path": "/models/delta",
            },
        )


def test_validate_bundle_pair_rejects_different_model_without_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="does not match the current inference configuration"):
        validate_bundle_pair(
            _case(),
            _bundle(role="clean"),
            _bundle(role="bad"),
            expected_model_composition={
                "base_model_path": "/models/base",
                "adapter_path": "/models/pure-ce-adapter",
                "adapter_type": "dora",
                "adapter_name": "default",
                "embedding_delta_path": "/models/pure-ce-delta",
            },
        )


def test_validate_bundle_pair_accepts_declared_counterfactual_and_records_metadata() -> None:
    result = validate_bundle_pair(
        _case(),
        _bundle(role="clean"),
        _bundle(role="bad"),
        expected_model_composition={
            "base_model_path": "/models/base",
            "adapter_path": "/models/pure-ce-adapter",
            "adapter_type": "dora",
            "adapter_name": "default",
            "embedding_delta_path": "/models/pure-ce-delta",
        },
        allow_counterfactual_model_composition=True,
        counterfactual_purpose="Compare a pure cross-entropy checkpoint against Gaussian coordinate-loss source trajectories.",
    )
    assert result["source_model_composition"]["adapter_path"] == "/models/adapter"
    assert result["executed_model_composition"]["adapter_path"] == "/models/pure-ce-adapter"
    assert result["counterfactual_model_composition"] == {
        "enabled": True,
        "source_and_executed_composition_differ": True,
        "purpose": "Compare a pure cross-entropy checkpoint against Gaussian coordinate-loss source trajectories.",
    }


def test_counterfactual_mode_does_not_allow_clean_bad_source_mismatch() -> None:
    clean = _bundle(role="clean")
    bad = _bundle(role="bad")
    bad["decode_result"]["model_identity"]["adapter"]["adapter_path"] = "/models/other-source-adapter"
    with pytest.raises(ValueError, match="different model composition"):
        validate_bundle_pair(
            _case(),
            clean,
            bad,
            expected_model_composition={
                "base_model_path": "/models/base",
                "adapter_path": "/models/pure-ce-adapter",
                "adapter_type": "dora",
                "adapter_name": "default",
                "embedding_delta_path": "/models/pure-ce-delta",
            },
            allow_counterfactual_model_composition=True,
            counterfactual_purpose="Compare a pure cross-entropy checkpoint against Gaussian coordinate-loss source trajectories.",
        )


def test_validate_bundle_pair_requires_purpose_for_counterfactual_mode() -> None:
    with pytest.raises(ValueError, match="purpose"):
        validate_bundle_pair(
            _case(),
            _bundle(role="clean"),
            _bundle(role="bad"),
            expected_model_composition={
                "base_model_path": "/models/base",
                "adapter_path": "/models/pure-ce-adapter",
                "adapter_type": "dora",
                "adapter_name": "default",
                "embedding_delta_path": "/models/pure-ce-delta",
            },
            allow_counterfactual_model_composition=True,
        )


def test_coordinate_summary_reports_full_and_conditional_mass() -> None:
    token_map = {index: index for index in range(10)}
    logits = [0.0] * 10
    logits[4] = 4.0
    logits[5] = 3.0
    summary = summarize_coordinate_logits(
        logits,
        token_map,
        clean_coordinate=4,
        bad_coordinate=5,
        reference_coordinate=4,
        window_radii=(1, 2),
    )
    assert summary["clean"]["coordinate_token_rank"] == 1
    assert summary["bad"]["coordinate_token_rank"] == 2
    assert summary["coordinate_argmax_bin"] == 4
    assert summary["reference_windows"]["plus_or_minus_1_bins"]["greedy_coordinate_argmax_inside"] is True
    assert summary["coordinate_vocabulary_probability_mass"] == pytest.approx(1.0)


def test_sampling_policy_is_explicit_and_top_p_only_removes_tokens() -> None:
    token_map = {index: index for index in range(10)}
    logits = [0.0] * 10
    logits[0] = 10.0
    logits[1] = 9.0
    logits[2] = 0.0
    summary = summarize_coordinate_logits(
        logits,
        token_map,
        clean_coordinate=0,
        bad_coordinate=1,
        reference_coordinate=0,
    )
    assert "raw_temperature_1" in summary["distribution_semantics"]
    assert "source_sampling_policy_temperature_0_4_top_p_0_95" in summary["distribution_semantics"]
    raw = summary["clean"]["raw_temperature_1"]
    policy = summary["clean"]["source_sampling_policy_temperature_0_4_top_p_0_95"]
    assert raw["full_vocabulary_rank"] == 1
    assert policy["included_by_top_p"] is True
    assert summary["bad"]["raw_temperature_1"]["full_vocabulary_rank"] == 2
    assert summary["bad"]["source_sampling_policy_temperature_0_4_top_p_0_95"]["included_by_top_p"] is True
    assert summary["reference_windows"]["plus_or_minus_4_bins"]["source_sampling_policy_probability_mass_over_full_vocabulary_after_top_p"] > 0


def test_pairing_keeps_y2_reports_when_x1_diverges_first() -> None:
    clean = _bundle(role="clean")
    degraded = _bundle(role="bad")
    # Make x1 differ first, while y2 also differs later.  The first-divergence
    # report must remain x1, but the complete paired artifact must retain y2.
    degraded["decode_result"]["generated_token_ids"][4] += 1
    degraded["decode_result"]["generated_token_ids"][7] += 1
    degraded["parse_score_receipts"][0]["selected_token_ids"][3] += 1
    degraded["parse_score_receipts"][0]["selected_token_ids"][6] += 1
    identity = validate_bundle_pair(_case(), clean, degraded)
    assert identity["first_differing_coordinate_slot"] == "x1"

    token_map = {index: index for index in range(10)}

    def trajectory(branch_coordinate_offset: int) -> list[dict[str, object]]:
        reports = []
        for slot_index, slot_name in enumerate(COORDINATE_SLOT_NAMES):
            report = summarize_coordinate_logits(
                [float(index) for index in range(10)],
                token_map,
                clean_coordinate=1,
                bad_coordinate=2,
                reference_coordinate=1,
            )
            reports.append(
                {
                    "slot_index": slot_index,
                    "slot": slot_name,
                    "branch_coordinate_bin": branch_coordinate_offset + slot_index,
                    "coordinate_report": report,
                }
            )
        return reports

    paired = pair_coordinate_branch_reports(trajectory(0), trajectory(10))
    y2 = paired[3]
    assert y2["slot"] == "y2"
    assert "coordinate_report" in y2["clean_prefix"]
    assert "coordinate_report" in y2["degraded_prefix"]
    assert y2["clean_prefix"]["coordinate_report"]["clean"]["raw_temperature_1"]
    assert y2["degraded_prefix"]["coordinate_report"]["bad"]["raw_temperature_1"]


def test_validation_does_not_mutate_bundle_inputs() -> None:
    clean = _bundle(role="clean")
    bad = _bundle(role="bad")
    before = copy.deepcopy((clean, bad))
    validate_bundle_pair(_case(), clean, bad)
    assert (clean, bad) == before

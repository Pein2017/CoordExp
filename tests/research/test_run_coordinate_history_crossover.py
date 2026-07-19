from pathlib import Path

from scripts.research import run_coordinate_history_crossover as crossover


ADMISSION = Path(
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-07-19-sampled-history-target-reachability-and-complete-row-value/"
    "stage5-coordinate-history-crossover-admission.json"
)


def test_frozen_crossover_prefixes_and_budget() -> None:
    frozen = crossover.load_stage_five_source(ADMISSION)
    contract = frozen["admission"]["execution_contract"]
    assert contract["forced_prefix_token_count"] == 45
    assert contract["post_prefix_generated_token_budget"] == 467
    assert contract["total_trajectory_generated_token_budget"] == 512
    assert set(frozen["prefixes"]) == {
        "native_row_zero__native_row_four",
        "sampled_row_zero__native_row_four",
        "native_row_zero__sampled_row_four",
        "sampled_row_zero__sampled_row_four",
    }
    assert frozen["row_zero_structure"]["passed"] is True
    assert frozen["row_four_structure"]["passed"] is True


def test_crossover_classifications() -> None:
    cases = {
        ("target", "duplicate"): "persistent_row_zero_effect",
        ("duplicate", "target"): "row_four_sufficient_or_mediating",
        ("target", "target"): "either_coordinate_state_sufficient",
        ("duplicate", "duplicate"): "joint_interaction",
        ("other", "duplicate"): "other_or_unresolved",
    }
    for (left, right), expected in cases.items():
        result = crossover.classify_crossover(
            sampled_zero_native_four_owner=left,
            native_zero_sampled_four_owner=right,
            target_owner_id="target",
            duplicate_owner_id="duplicate",
            gates_passed=True,
        )
        assert result["classification"] == expected
        assert result["claim_allowed"] is (expected != "other_or_unresolved")


def test_failed_gate_refuses_interpretation() -> None:
    result = crossover.classify_crossover(
        sampled_zero_native_four_owner="target",
        native_zero_sampled_four_owner="duplicate",
        target_owner_id="target",
        duplicate_owner_id="duplicate",
        gates_passed=False,
    )
    assert result["classification"] == "other_or_unresolved"
    assert result["claim_allowed"] is False

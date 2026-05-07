from __future__ import annotations

import math

import pytest

from src.detection.objective import (
    compute_eos_trust_weight,
    eos_trust_weight_from_expected_unlabeled_count,
    expected_unlabeled_count,
)


def _empirical_cfg(**trust_updates: float) -> dict[str, object]:
    trust_mapping = {
        "type": "log_linear_missing_count_penalty",
        "penalty_per_missing": 1.0,
        "temperature": 1.0,
        "min_weight": 0.0,
        "max_weight": 1.0,
    }
    trust_mapping.update(trust_updates)
    return {
        "source": "empirical_unlabeled_poisson_v0",
        "expected_unlabeled_count": {
            "intercept": -0.35,
            "slope": 0.43,
            "floor": 0.0,
        },
        "trust_mapping": trust_mapping,
    }


def test_expected_unlabeled_count_matches_user_linear_formula_with_floor() -> None:
    assert expected_unlabeled_count(0) == pytest.approx(0.0)
    assert expected_unlabeled_count(1) == pytest.approx(0.08)
    assert expected_unlabeled_count(2) == pytest.approx(0.51)
    assert expected_unlabeled_count(10) == pytest.approx(3.95)


def test_empirical_unlabeled_poisson_eos_trust_weight_values() -> None:
    cfg = _empirical_cfg()

    assert compute_eos_trust_weight(0, cfg) == pytest.approx(1.0)
    assert compute_eos_trust_weight(1, cfg) == pytest.approx(math.exp(-0.08))
    assert compute_eos_trust_weight(2, cfg) == pytest.approx(math.exp(-0.51))
    assert compute_eos_trust_weight(10, cfg) == pytest.approx(math.exp(-3.95))


def test_empirical_eos_trust_weight_clamps_to_configured_bounds() -> None:
    cfg = _empirical_cfg(min_weight=0.1, max_weight=0.9)

    assert compute_eos_trust_weight(0, cfg) == pytest.approx(0.9)
    assert compute_eos_trust_weight(20, cfg) == pytest.approx(0.1)


def test_disabled_ablation_returns_exact_zero_eos_trust_weight() -> None:
    cfg = {"source": "disabled_ablation"}

    assert compute_eos_trust_weight(0, cfg) == pytest.approx(0.0)
    assert compute_eos_trust_weight(50, cfg) == pytest.approx(0.0)


def test_constant_ablation_returns_configured_eos_trust_weight() -> None:
    cfg = {"source": "constant_ablation", "value": 0.25}

    assert compute_eos_trust_weight(0, cfg) == pytest.approx(0.25)
    assert compute_eos_trust_weight(50, cfg) == pytest.approx(0.25)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_constant_ablation_rejects_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError, match="constant_ablation"):
        compute_eos_trust_weight(1, {"source": "constant_ablation", "value": value})


def test_log_linear_mapping_rejects_invalid_clamp() -> None:
    with pytest.raises(ValueError, match="0 <= min_weight <= max_weight <= 1"):
        eos_trust_weight_from_expected_unlabeled_count(
            1.0,
            min_weight=0.0,
            max_weight=1.5,
        )


def test_calibrated_formula_ref_fails_fast_until_runtime_formula_exists() -> None:
    with pytest.raises(ValueError, match="calibrated_formula_ref"):
        compute_eos_trust_weight(
            5,
            {
                "source": "calibrated_formula_ref",
                "calibration_artifact_ref": "artifacts/eos_calibration_formula.v1.json",
            },
        )

"""Current supported training-config inventory for the strict loss contract.

Every current supported CoordExp-Swift training config MUST strict-resolve to
the canonical protected/auxiliary loss contract. Historical/archived configs
remain provenance and MUST NOT be accepted as current inputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.errors import ConfigContractError
from src.config.loader import load_train_config
from src.config.models import LossesConfig


SUPPORTED_TRAIN_CONFIG_ROOTS = (
    Path("configs/coordexp_swift/prod"),
    Path("configs/coordexp_swift/smoke"),
)
EXPECTED_SUPPORTED_CONFIG_COUNT = 25
CANONICAL_GROUPS = ("desc_text", "schema", "coordinate", "eos")
HISTORICAL_CONFIG = Path(
    "configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/"
    "prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p2.yaml"
)


def _supported_config_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for root in SUPPORTED_TRAIN_CONFIG_ROOTS:
        paths.extend(sorted(root.rglob("*.yaml")))
    return tuple(paths)


def test_supported_config_inventory_is_complete() -> None:
    paths = _supported_config_paths()

    assert len(paths) == EXPECTED_SUPPORTED_CONFIG_COUNT
    assert len(set(paths)) == len(paths)


@pytest.mark.parametrize(
    "config_path",
    _supported_config_paths(),
    ids=lambda path: str(path),
)
def test_supported_config_resolves_to_the_strict_loss_contract(
    config_path: Path,
) -> None:
    resolved = load_train_config(config_path)
    losses = resolved.config.losses

    assert losses.normalizer == "segment_balanced"
    assert losses.protected.base_ce.weight == 1.0

    gate = losses.protected.token_type_gate
    assert gate.groups == CANONICAL_GROUPS
    if gate.mode == "enabled":
        assert gate.weight == pytest.approx(0.1)
    else:
        assert gate.mode == "zero_weight_ablation"
        assert gate.weight == 0.0

    assert not hasattr(losses.protected, "coord_gaussian_rps")

    dumped = resolved.config_dict["losses"]
    assert "coord_gaussian_rps" not in dumped["protected"]
    assert dumped["protected"]["token_type_gate"]["mode"] in (
        "enabled",
        "zero_weight_ablation",
    )


def test_supported_inventory_covers_both_gate_modes_and_the_auxiliary_term() -> None:
    modes: set[str] = set()
    auxiliary_weights: list[float] = []
    for config_path in _supported_config_paths():
        losses = load_train_config(config_path).config.losses
        modes.add(losses.protected.token_type_gate.mode)
        auxiliary = losses.auxiliary
        if auxiliary is not None and auxiliary.coord_gaussian_rps is not None:
            auxiliary_weights.append(auxiliary.coord_gaussian_rps.weight)

    assert modes == {"enabled", "zero_weight_ablation"}
    assert auxiliary_weights == [1.0, 1.0]


def test_historical_config_is_provenance_not_a_current_input() -> None:
    assert HISTORICAL_CONFIG.exists()

    with pytest.raises(ConfigContractError):
        load_train_config(HISTORICAL_CONFIG)


def test_historical_protected_coordinate_layout_is_rejected() -> None:
    """The pre-migration supported layout must fail with a migration message."""
    legacy_payload = {
        "normalizer": "segment_balanced",
        "protected": {
            "base_ce": {"weight": 1.0},
            "token_type_gate": {
                "weight": 0.2,
                "groups": ["desc_text", "schema", "coordinate", "eos"],
            },
            "coord_gaussian_rps": {
                "weight": 1.0,
                "gaussian_weight": 0.5,
                "rps_weight": 0.2,
                "temperature": 1.0,
                "gaussian_r95_axis_fraction": 0.04,
                "gaussian_r95_cap_bins": 8,
                "gaussian_r95_min_bins": 1,
                "gaussian_r95_fallback_bins": 8,
            },
        },
    }

    with pytest.raises(ValueError) as exc_info:
        LossesConfig.model_validate(legacy_payload)

    assert "losses.auxiliary.coord_gaussian_rps" in str(exc_info.value)

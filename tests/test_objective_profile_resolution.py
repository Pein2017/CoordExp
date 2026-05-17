from __future__ import annotations

import pytest

from src.training.surfaces import TrainingSurfaceResolver


def _removed_key(*parts: str) -> str:
    return "_".join(parts)


def test_keyed_objectives_resolve_to_deterministic_order() -> None:
    profile = TrainingSurfaceResolver().resolve_objectives(
        {
            "box_regression": {"enabled": True, "weight": 0.25},
            "trie_ce": {"enabled": True, "weight": 0.5},
            "token_ce": {"enabled": True, "weight": 1.0},
            "coord_soft_ce": {"enabled": True, "weight": 0.75},
        }
    )

    assert [entry.objective_id for entry in profile.objectives] == [
        "token_ce",
        "trie_ce",
        "coord_soft_ce",
        "box_regression",
    ]
    assert [entry.weight for entry in profile.enabled_objectives] == [
        1.0,
        0.5,
        0.75,
        0.25,
    ]


def test_disabled_objective_stays_explicit_without_dropping_siblings() -> None:
    profile = TrainingSurfaceResolver().resolve_objectives(
        {
            "token_ce": {"enabled": True, "weight": 1.0},
            "trie_ce": {"enabled": False, "weight": 0.5},
            "box_regression": {"enabled": True, "weight": 0.25},
        }
    )

    assert [entry.objective_id for entry in profile.objectives] == [
        "token_ce",
        "trie_ce",
        "box_regression",
    ]
    assert [entry.objective_id for entry in profile.enabled_objectives] == [
        "token_ce",
        "box_regression",
    ]
    assert profile.objectives[1].enabled is False


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -0.1])
def test_objective_weight_must_be_finite_and_non_negative(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and >= 0"):
        TrainingSurfaceResolver().resolve_objectives(
            {"token_ce": {"enabled": True, "weight": weight}}
        )


def test_objective_config_metadata_must_be_finite() -> None:
    with pytest.raises(ValueError, match=r"objectives\.token_ce\.config\.temperature"):
        TrainingSurfaceResolver().resolve_objectives(
            {
                "token_ce": {
                    "enabled": True,
                    "config": {"temperature": float("nan")},
                }
            }
        )


@pytest.mark.parametrize(
    "objective_id",
    [
        _removed_key("loss", "duplicate", "burst", "unlikelihood"),
        _removed_key("adjacent", "repulsion"),
        _removed_key("eos", "loosen"),
        _removed_key(
            "forced",
            "continuation",
        ),
        _removed_key("stop", "gate"),
    ],
)
def test_removed_objective_keys_fail_fast(objective_id: str) -> None:
    with pytest.raises(ValueError, match=objective_id):
        TrainingSurfaceResolver().resolve_objectives(
            {
                "token_ce": {"enabled": True},
                objective_id: {"enabled": True},
            }
        )


@pytest.mark.parametrize(
    "removed_key",
    [
        _removed_key("adjacent", "repulsion", "copy", "margin"),
        _removed_key("adjacent", "repulsion", "filter", "mode"),
        _removed_key("adjacent", "repulsion", "margin", "ratio"),
        _removed_key("adjacent", "repulsion", "weight"),
        _removed_key("continue", "over", "eos", "weight"),
        _removed_key("eos", "stop", "weight"),
        _removed_key("eos", "trust", "weight"),
        _removed_key(
            "force",
            "continuation",
        ),
        _removed_key("missing", "label", "prior", "weighted", "ce"),
        _removed_key("stop", "signal", "ce"),
    ],
)
def test_removed_nested_objective_config_keys_fail_fast(removed_key: str) -> None:
    with pytest.raises(ValueError, match=removed_key):
        TrainingSurfaceResolver().resolve_objectives(
            {
                "token_ce": {
                    "enabled": True,
                    "config": {
                        "nested": [
                            {"safe_key": 1},
                            {removed_key: 0.5},
                        ]
                    },
                },
            }
        )


def test_unknown_objective_keys_fail_fast() -> None:
    with pytest.raises(ValueError, match="unknown objective"):
        TrainingSurfaceResolver().resolve_objectives(
            {
                "token_ce": {"enabled": True},
                "mystery_loss": {"enabled": True},
            }
        )

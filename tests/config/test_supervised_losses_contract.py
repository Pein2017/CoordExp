"""Strict supervised-loss configuration contract.

Covers the protected/auxiliary loss surface of
`openspec/changes/standardize-coordexp-swift-supervised-losses`:

- `losses.protected.base_ce.weight` is exactly `1.0`;
- `losses.protected.token_type_gate.mode` discriminates the canonical enabled
  weight `0.1` from the named `zero_weight_ablation` weight `0.0`;
- gate `groups` equal the canonical ordered V1 token-type tuple exactly;
- coordinate Gaussian/RPS lives only under `losses.auxiliary`;
- no import paths, callables, or unknown loss names anywhere under `losses`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.common.errors import ConfigContractError
from src.config.loader import load_train_config

from test_train_config import _minimal_config, _set_nested, _write_yaml


CANONICAL_GROUPS = ["desc_text", "schema", "coordinate", "eos"]


def _losses_payload(
    *,
    base_ce: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    auxiliary: dict[str, Any] | None = None,
    omit_auxiliary: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "normalizer": "segment_balanced",
        "protected": {
            "base_ce": {"weight": 1.0} if base_ce is None else base_ce,
            "token_type_gate": (
                {"mode": "enabled", "weight": 0.1, "groups": list(CANONICAL_GROUPS)}
                if gate is None
                else gate
            ),
        },
    }
    if auxiliary is not None:
        payload["auxiliary"] = auxiliary
    elif not omit_auxiliary:
        payload["auxiliary"] = {}
    return payload


def _config_with_losses(losses: dict[str, Any]) -> dict[str, Any]:
    payload = _minimal_config()
    payload["losses"] = losses
    return payload


def _load(tmp_path: Path, payload: dict[str, Any], name: str = "config.yaml"):
    config_path = tmp_path / name
    _write_yaml(config_path, payload)
    return load_train_config(config_path)


def _expect_rejection(tmp_path: Path, payload: dict[str, Any]) -> ConfigContractError:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, payload)
    with pytest.raises(ConfigContractError) as exc_info:
        load_train_config(config_path)
    return exc_info.value


def _diagnostic(error: ConfigContractError) -> str:
    context = error.context
    return f"{context.get('field', '')}: {context.get('message', '')}"


# --------------------------------------------------------------------------
# Canonical accepted shapes
# --------------------------------------------------------------------------


def test_canonical_enabled_gate_resolves_with_exact_constants(tmp_path: Path) -> None:
    resolved = _load(tmp_path, _config_with_losses(_losses_payload()))

    losses = resolved.config.losses
    assert losses.normalizer == "segment_balanced"
    assert losses.protected.base_ce.weight == pytest.approx(1.0)
    assert losses.protected.token_type_gate.mode == "enabled"
    assert losses.protected.token_type_gate.weight == pytest.approx(0.1)
    assert losses.protected.token_type_gate.groups == tuple(CANONICAL_GROUPS)
    assert losses.auxiliary is None

    dumped = resolved.config_dict["losses"]
    assert dumped["protected"]["token_type_gate"]["mode"] == "enabled"
    assert dumped["protected"]["token_type_gate"]["weight"] == pytest.approx(0.1)
    assert "coord_gaussian_rps" not in dumped["protected"]


def test_named_zero_weight_ablation_resolves_and_keeps_identity(
    tmp_path: Path,
) -> None:
    losses = _losses_payload(
        gate={
            "mode": "zero_weight_ablation",
            "weight": 0.0,
            "groups": list(CANONICAL_GROUPS),
        }
    )

    resolved = _load(tmp_path, _config_with_losses(losses))

    gate = resolved.config.losses.protected.token_type_gate
    assert gate.mode == "zero_weight_ablation"
    assert gate.weight == 0.0
    assert resolved.config_dict["losses"]["protected"]["token_type_gate"]["mode"] == (
        "zero_weight_ablation"
    )


def test_models_canonical_groups_equal_loss_vocabulary_groups() -> None:
    from src.config.models import (
        CANONICAL_TOKEN_TYPE_GROUPS,
        TOKEN_TYPE_GATE_MODE_WEIGHTS,
    )
    from src.losses.vocab import V1_TOKEN_TYPES

    assert CANONICAL_TOKEN_TYPE_GROUPS == tuple(V1_TOKEN_TYPES)
    assert TOKEN_TYPE_GATE_MODE_WEIGHTS == {
        "enabled": 0.1,
        "zero_weight_ablation": 0.0,
    }


# --------------------------------------------------------------------------
# Base CE
# --------------------------------------------------------------------------


def test_base_ce_omission_fails_before_loss_construction(tmp_path: Path) -> None:
    losses = _losses_payload()
    del losses["protected"]["base_ce"]

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "base_ce" in error.context["field"]


@pytest.mark.parametrize("weight", [0.0, 0.5, 0.9, 1.5, 2.0])
def test_base_ce_reweighting_fails(tmp_path: Path, weight: float) -> None:
    losses = _losses_payload(base_ce={"weight": weight})

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "base_ce" in error.context["field"]
    assert "1.0" in error.context["message"]


# --------------------------------------------------------------------------
# Token-type gate mode / weight pairing
# --------------------------------------------------------------------------


def test_gate_without_mode_fails_with_migration_message(tmp_path: Path) -> None:
    losses = _losses_payload(
        gate={"weight": 0.1, "groups": list(CANONICAL_GROUPS)},
    )

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "token_type_gate" in error.context["field"]
    assert "mode" in error.context["message"]
    assert "zero_weight_ablation" in error.context["message"]


@pytest.mark.parametrize(
    ("mode", "weight"),
    [
        ("enabled", 0.0),
        ("enabled", 0.05),
        ("enabled", 0.2),
        ("enabled", 0.25),
        ("enabled", 1.0),
        ("zero_weight_ablation", 0.1),
        ("zero_weight_ablation", 0.2),
        ("zero_weight_ablation", 1.0),
    ],
)
def test_incompatible_gate_mode_and_weight_pairs_fail(
    tmp_path: Path,
    mode: str,
    weight: float,
) -> None:
    losses = _losses_payload(
        gate={"mode": mode, "weight": weight, "groups": list(CANONICAL_GROUPS)},
    )

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "token_type_gate" in error.context["field"]
    assert mode in error.context["message"]


def test_gate_weight_without_mode_identity_is_not_inferred(tmp_path: Path) -> None:
    losses = _losses_payload(gate={"weight": 0.0, "groups": list(CANONICAL_GROUPS)})

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "token_type_gate" in error.context["field"]


@pytest.mark.parametrize("mode", ["disabled", "ablation", "ENABLED", "", None])
def test_unknown_gate_modes_fail(tmp_path: Path, mode: Any) -> None:
    losses = _losses_payload(
        gate={"mode": mode, "weight": 0.1, "groups": list(CANONICAL_GROUPS)},
    )

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "token_type_gate" in error.context["field"]


def test_gate_mode_may_not_be_authored_outside_the_gate(tmp_path: Path) -> None:
    losses = _losses_payload()
    losses["protected"]["mode"] = "enabled"

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "Extra inputs are not permitted" in error.context["message"]


# --------------------------------------------------------------------------
# Token-type gate groups
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "groups",
    [
        [],
        ["desc_text", "schema", "coordinate"],
        ["schema", "coordinate", "eos"],
        ["desc_text", "schema", "coordinate", "eos", "eos"],
        ["desc_text", "desc_text", "schema", "coordinate", "eos"],
        ["desc_text", "schema", "coordinate", "eos", "bbox"],
        ["eos", "coordinate", "schema", "desc_text"],
        ["desc_text", "coordinate", "schema", "eos"],
        ["desc_text", "schema", "eos", "coordinate"],
    ],
)
def test_non_canonical_gate_group_tuples_fail(
    tmp_path: Path,
    groups: list[str],
) -> None:
    losses = _losses_payload(
        gate={"mode": "enabled", "weight": 0.1, "groups": groups},
    )

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "token_type_gate" in error.context["field"]


def test_gate_groups_are_required(tmp_path: Path) -> None:
    losses = _losses_payload(gate={"mode": "enabled", "weight": 0.1})

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "groups" in error.context["field"]


# --------------------------------------------------------------------------
# Coordinate auxiliary placement
# --------------------------------------------------------------------------


def test_legacy_protected_coordinate_placement_fails_with_migration_message(
    tmp_path: Path,
) -> None:
    losses = _losses_payload()
    losses["protected"]["coord_gaussian_rps"] = {
        "weight": 1.0,
        "gaussian_weight": 0.5,
        "rps_weight": 0.2,
        "temperature": 1.0,
        "gaussian_r95_axis_fraction": 0.04,
        "gaussian_r95_cap_bins": 8,
        "gaussian_r95_min_bins": 1,
        "gaussian_r95_fallback_bins": 8,
    }

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "losses.auxiliary.coord_gaussian_rps" in error.context["message"]


def test_typed_auxiliary_coordinate_placement_resolves(tmp_path: Path) -> None:
    losses = _losses_payload(
        auxiliary={
            "coord_gaussian_rps": {
                "weight": 1.0,
                "gaussian_weight": 0.5,
                "rps_weight": 0.2,
                "temperature": 1.0,
                "gaussian_r95_axis_fraction": 0.04,
                "gaussian_r95_cap_bins": 8,
                "gaussian_r95_min_bins": 1,
                "gaussian_r95_fallback_bins": 8,
            }
        }
    )

    resolved = _load(tmp_path, _config_with_losses(losses))

    auxiliary = resolved.config.losses.auxiliary
    assert auxiliary is not None
    coord = auxiliary.coord_gaussian_rps
    assert coord is not None
    assert coord.weight == pytest.approx(1.0)
    assert coord.gaussian_weight == pytest.approx(0.5)
    assert coord.rps_weight == pytest.approx(0.2)
    assert coord.temperature == pytest.approx(1.0)
    assert coord.gaussian_r95_axis_fraction == pytest.approx(0.04)
    assert coord.gaussian_r95_cap_bins == 8
    assert coord.gaussian_r95_min_bins == 1
    assert coord.gaussian_r95_fallback_bins == 8


def test_auxiliary_zero_weight_is_a_valid_omitted_term(tmp_path: Path) -> None:
    losses = _losses_payload(auxiliary={"coord_gaussian_rps": {"weight": 0.0}})

    resolved = _load(tmp_path, _config_with_losses(losses))

    auxiliary = resolved.config.losses.auxiliary
    assert auxiliary is not None
    assert auxiliary.coord_gaussian_rps is not None
    assert auxiliary.coord_gaussian_rps.weight == 0.0


def test_absent_auxiliary_section_is_valid(tmp_path: Path) -> None:
    resolved = _load(tmp_path, _config_with_losses(_losses_payload()))

    assert resolved.config.losses.auxiliary is None


def test_empty_auxiliary_section_omits_the_coordinate_term(tmp_path: Path) -> None:
    resolved = _load(tmp_path, _config_with_losses(_losses_payload(auxiliary={})))

    auxiliary = resolved.config.losses.auxiliary
    assert auxiliary is not None
    assert auxiliary.coord_gaussian_rps is None


def test_auxiliary_coordinate_term_keeps_its_typed_field_validation(
    tmp_path: Path,
) -> None:
    losses = _losses_payload(
        auxiliary={"coord_gaussian_rps": {"weight": 1.0, "temperature": 1e-45}}
    )

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "coord_gaussian_rps" in error.context["field"]


# --------------------------------------------------------------------------
# Unknown names and implementation hooks
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        ("losses.registry", {"coord_soft_ce": "src.losses.legacy:CoordSoftCE"}),
        ("losses.plugins", ["src.losses.legacy:CoordSoftCE"]),
        ("losses.protected.coord_soft_ce", {"weight": 1.0}),
        ("losses.protected.object_balanced", {"weight": 1.0}),
        ("losses.protected.base_ce.impl", "src.losses.base_ce:BaseTokenCE"),
        ("losses.protected.base_ce.class_path", "src.losses.base_ce.BaseTokenCE"),
        ("losses.protected.token_type_gate.factory", "src.losses:make_gate"),
        ("losses.protected.token_type_gate.options", {"alpha": 1.0}),
        ("losses.auxiliary.coord_soft_ce", {"weight": 1.0}),
        ("losses.auxiliary.custom", {"import_path": "src.losses:Custom"}),
    ],
)
def test_unknown_loss_names_and_implementation_hooks_are_rejected(
    tmp_path: Path,
    field_path: str,
    value: Any,
) -> None:
    payload = _config_with_losses(_losses_payload())
    _set_nested(payload, field_path, value)

    error = _expect_rejection(tmp_path, payload)

    assert "Extra inputs are not permitted" in error.context["message"]


def test_auxiliary_coordinate_term_rejects_unknown_fields(tmp_path: Path) -> None:
    losses = _losses_payload(
        auxiliary={
            "coord_gaussian_rps": {
                "weight": 1.0,
                "loss_class": "src.losses.coord_gaussian_rps:CoordGaussianRPSLoss",
            }
        }
    )

    error = _expect_rejection(tmp_path, _config_with_losses(losses))

    assert "Extra inputs are not permitted" in error.context["message"]

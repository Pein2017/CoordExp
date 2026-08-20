"""Wave-2 task 2.1 interface tests for the closed token-loss binding inventory.

These are interface-level contract tests: they pin the exact closed set of
implemented token losses, their frozen metadata (role, normalizer, zero
policy), and the fact that `LossRunner` composition has exactly one source of
truth for "which terms exist". They also fail closed on any re-introduction of
a registry, import-by-name, callable config, or pass-through factory surface.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

import src.losses as losses_package
from src.common.errors import LossContractError
from src.config.models import (
    AuxiliaryLossesConfig,
    BaseCELossConfig,
    CoordGaussianRPSLossConfig,
    LossesConfig,
    ProtectedLossesConfig,
    TokenTypeGateLossConfig,
)
from src.coordinate_targets import CoordinateLossTarget
from src.losses import LossContext, LossRunner, TokenVocabularyGroups
from src.losses.bindings import (
    BASE_CE_BINDING,
    COORD_GAUSSIAN_RPS_BINDING,
    COORDINATE_TOKEN_TYPES,
    PROTECTED_BASE_CE_WEIGHT,
    TOKEN_LOSS_BINDINGS,
    TOKEN_TYPE_GATE_BINDING,
    TokenLossBinding,
    binding_for,
)
from src.packing.planner import PackedSegment
from src.supervision import TokenAtom, TokenSequence

CANONICAL_GROUPS = ("desc_text", "schema", "coordinate", "eos")


def _coordinate_context() -> LossContext:
    """One pack, one supervised coordinate atom -- eligible for all three
    bindings, so term presence/absence is decided by composition alone."""

    features = torch.tensor((((1.0, -0.5), (0.25, 2.0)),), dtype=torch.float32)
    weight = torch.linspace(-0.4, 0.6, steps=16).reshape(2, 8).requires_grad_()
    bias = torch.linspace(0.3, -0.2, steps=8).requires_grad_()
    logits = features @ weight + bias
    return LossContext(
        logits=logits,
        token_sequence=TokenSequence(
            pack_index=0,
            input_ids=(0, 0),
            segments=(
                PackedSegment(
                    pack_index=0,
                    segment_index=0,
                    example_index=0,
                    example_id="ex-0",
                    start=0,
                    end=2,
                ),
            ),
            atoms=(
                TokenAtom(
                    pack_index=0,
                    segment_index=0,
                    example_index=0,
                    example_id="ex-0",
                    target_position=1,
                    token_id=3,
                    token_type="coordinate",
                    text="x",
                    logical_target_position=1,
                    object_id="obj-1",
                    field="bbox[0]",
                    source="unit",
                    coordinate_target=CoordinateLossTarget(
                        bbox=(2, 3, 8, 13), slot_index=0
                    ),
                ),
            ),
            spans=(),
        ),
        vocab_groups=TokenVocabularyGroups(
            vocab_size=8,
            desc_text=(7,),
            schema=(1, 2),
            coordinate=(3, 4),
            eos=(5,),
            blocked=(0, 6),
        ),
        logits_position_ids=None,
    )


def _losses_config(
    *,
    gate_mode: str = "enabled",
    gate_weight: float = 0.1,
    coord_weight: float | None = None,
) -> LossesConfig:
    auxiliary = (
        None
        if coord_weight is None
        else AuxiliaryLossesConfig(
            coord_gaussian_rps=CoordGaussianRPSLossConfig(weight=coord_weight)
        )
    )
    return LossesConfig(
        normalizer="segment_balanced",
        protected=ProtectedLossesConfig(
            base_ce=BaseCELossConfig(weight=1.0),
            token_type_gate=TokenTypeGateLossConfig(
                mode=gate_mode,
                weight=gate_weight,
                groups=CANONICAL_GROUPS,
            ),
        ),
        auxiliary=auxiliary,
    )


def test_binding_inventory_is_exactly_the_three_implemented_token_losses() -> None:
    assert isinstance(TOKEN_LOSS_BINDINGS, tuple)
    assert [binding.name for binding in TOKEN_LOSS_BINDINGS] == [
        "base_ce",
        "token_type_gate",
        "coord_gaussian_rps",
    ]
    assert len(TOKEN_LOSS_BINDINGS) == 3
    assert TOKEN_LOSS_BINDINGS == (
        BASE_CE_BINDING,
        TOKEN_TYPE_GATE_BINDING,
        COORD_GAUSSIAN_RPS_BINDING,
    )


@pytest.mark.parametrize(
    ("binding", "name", "role", "zero_policy"),
    (
        (BASE_CE_BINDING, "base_ce", "protected", "forbid"),
        (
            TOKEN_TYPE_GATE_BINDING,
            "token_type_gate",
            "protected",
            "detached_diagnostic",
        ),
        (COORD_GAUSSIAN_RPS_BINDING, "coord_gaussian_rps", "auxiliary", "omit"),
    ),
)
def test_binding_metadata_is_exact(
    binding: TokenLossBinding,
    name: str,
    role: str,
    zero_policy: str,
) -> None:
    assert binding.name == name
    assert binding.role == role
    assert binding.normalizer == "segment_balanced"
    assert binding.zero_policy == zero_policy


def test_every_binding_uses_the_segment_balanced_normalizer() -> None:
    assert {binding.normalizer for binding in TOKEN_LOSS_BINDINGS} == {
        "segment_balanced"
    }


def test_binding_records_are_frozen_and_reject_new_attributes() -> None:
    with pytest.raises(FrozenInstanceError):
        BASE_CE_BINDING.name = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        TOKEN_TYPE_GATE_BINDING.zero_policy = "omit"  # type: ignore[misc]
    with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
        COORD_GAUSSIAN_RPS_BINDING.extra_hook = object()  # type: ignore[attr-defined]
    # Slotted: no per-instance `__dict__`, so no hook can be attached at all.
    assert TokenLossBinding.__slots__ == (
        "name",
        "role",
        "normalizer",
        "zero_policy",
    )
    assert not hasattr(BASE_CE_BINDING, "__dict__")


def test_binding_inventory_is_an_immutable_sequence() -> None:
    with pytest.raises(TypeError):
        TOKEN_LOSS_BINDINGS[0] = BASE_CE_BINDING  # type: ignore[index]
    with pytest.raises(AttributeError):
        TOKEN_LOSS_BINDINGS.append(BASE_CE_BINDING)  # type: ignore[attr-defined]


def test_binding_lookup_is_closed_and_rejects_unknown_names() -> None:
    for binding in TOKEN_LOSS_BINDINGS:
        assert binding_for(binding.name) is binding
    for unknown in ("coord_soft_ce", "kl", "reward", "base_ce ", "BASE_CE"):
        with pytest.raises(LossContractError) as excinfo:
            binding_for(unknown)
        assert excinfo.value.code == "loss.unknown_token_loss_binding"


def test_protected_base_ce_weight_constant_is_exactly_one() -> None:
    assert PROTECTED_BASE_CE_WEIGHT == 1.0
    assert COORDINATE_TOKEN_TYPES == ("coordinate",)


def test_runner_composition_selects_binding_objects_from_the_inventory() -> None:
    baseline = LossRunner.from_config(_losses_config())
    assert baseline.active_bindings() == (BASE_CE_BINDING, TOKEN_TYPE_GATE_BINDING)
    for binding in baseline.active_bindings():
        assert any(binding is known for known in TOKEN_LOSS_BINDINGS)

    ablation = LossRunner.from_config(
        _losses_config(gate_mode="zero_weight_ablation", gate_weight=0.0)
    )
    assert ablation.active_bindings() == (BASE_CE_BINDING, TOKEN_TYPE_GATE_BINDING)

    with_auxiliary = LossRunner.from_config(_losses_config(coord_weight=0.5))
    assert with_auxiliary.active_bindings() == (
        BASE_CE_BINDING,
        TOKEN_TYPE_GATE_BINDING,
        COORD_GAUSSIAN_RPS_BINDING,
    )

    omitted_auxiliary = LossRunner.from_config(_losses_config(coord_weight=0.0))
    assert omitted_auxiliary.active_bindings() == (
        BASE_CE_BINDING,
        TOKEN_TYPE_GATE_BINDING,
    )


@pytest.mark.parametrize("coord_weight", (None, 0.0, 0.5))
def test_active_bindings_are_the_single_source_of_runtime_term_structure(
    coord_weight: float | None,
) -> None:
    """Denominators, bundle terms, finite status and metrics all derive from
    the same closed composition: nothing may appear that the inventory-driven
    `active_bindings()` did not select, and nothing it selects may be absent.
    """

    runner = LossRunner.from_config(_losses_config(coord_weight=coord_weight))
    expected = tuple(binding.name for binding in runner.active_bindings())
    # Canonical inventory order is preserved by the composition path.
    assert expected == tuple(
        binding.name for binding in TOKEN_LOSS_BINDINGS if binding.name in set(expected)
    )

    context = _coordinate_context()
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    finalized = runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)

    assert tuple(plan.denominators) == expected
    assert tuple(term.name for term in bundle.terms) == expected
    assert tuple(bundle.finite_status["terms"]) == expected
    assert tuple(str(term["name"]) for term in finalized["terms"]) == expected
    assert tuple(finalized["diagnostics"]["term_order"]) == expected
    for absent in {
        binding.name for binding in TOKEN_LOSS_BINDINGS
    } - set(expected):
        assert absent not in plan.denominators
        assert not [key for key in bundle.metrics if absent in key]
        assert not [key for key in finalized["metrics"] if absent in key]


def test_losses_package_exposes_no_registry_or_dynamic_selection_surface() -> None:
    forbidden = (
        "register",
        "registry",
        "discover",
        "plugin",
        "entry_point",
        "factory",
        "from_import_path",
        "import_path",
    )
    exported = tuple(losses_package.__all__)
    for name in exported:
        lowered = name.lower()
        assert not any(token in lowered for token in forbidden), name
    # The binding inventory is private metadata for the one deep runner
    # implementation: it is deliberately NOT part of the package's public
    # surface, so it can never become a third-party extension point.
    assert "TokenLossBinding" not in exported
    assert "TOKEN_LOSS_BINDINGS" not in exported


def test_loss_package_source_contains_no_dynamic_import_machinery() -> None:
    package_root = Path(losses_package.__file__).resolve().parent
    forbidden_tokens = (
        "importlib",
        "__import__",
        "pkgutil",
        "pkg_resources",
        "entry_points",
        "eval(",
        "exec(",
    )
    offenders: list[str] = []
    for module_path in sorted(package_root.glob("*.py")):
        source = module_path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in source:
                offenders.append(f"{module_path.name}:{token}")
    assert offenders == []


@pytest.mark.parametrize(
    "hook",
    (
        {"registry": "src.losses.base_ce:BaseTokenCE"},
        {"target": "src.losses.base_ce.BaseTokenCE"},
        {"class_path": "src.losses.base_ce.BaseTokenCE"},
        {"impl": lambda context: None},
    ),
)
def test_public_loss_config_rejects_import_path_or_callable_selection(
    hook: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "normalizer": "segment_balanced",
        "protected": {
            "base_ce": {"weight": 1.0},
            "token_type_gate": {
                "mode": "enabled",
                "weight": 0.1,
                "groups": list(CANONICAL_GROUPS),
            },
        },
    }
    payload.update(hook)
    with pytest.raises(Exception) as excinfo:
        LossesConfig(**payload)  # type: ignore[arg-type]
    assert "extra" in str(excinfo.value).lower() or "unexpected" in str(
        excinfo.value
    ).lower()


def test_public_loss_config_rejects_unrecognized_auxiliary_loss_name() -> None:
    with pytest.raises(Exception):
        AuxiliaryLossesConfig(rollout_kl={"weight": 1.0})  # type: ignore[call-arg]

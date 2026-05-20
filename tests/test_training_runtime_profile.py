from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.bootstrap.trainer_setup import compose_trainer_class
from src.trainers.metrics.mixins import (
    AggregateTokenTypeMetricsMixin,
    CoordSoftCEW1LossMixin,
    GradAccumLossScaleMixin,
    InstabilityMonitorMixin,
    RecursiveDetectionCEMixin,
    SFTStructuralCloseLossMixin,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "src" / "training_runtime" / "profile.py"


def _profile_module():
    return importlib.import_module("src.training_runtime.profile")


def test_profile_module_is_import_safe() -> None:
    assert PROFILE_PATH.exists()

    tree = ast.parse(PROFILE_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots <= {"__future__", "dataclasses", "src", "typing"}

    banned_modules = [
        "torch",
        "transformers",
        "swift",
        "vllm",
        "datasets",
        "src.sft",
        "src.config",
        "src.data_collators",
        "src.trainers",
    ]
    code = (
        "import sys\n"
        "import src.training_runtime.profile\n"
        f"banned = {banned_modules!r}\n"
        "loaded = [name for name in banned if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
    )


def test_default_profile_derives_generic_stage1_policy_from_plan() -> None:
    profile_mod = _profile_module()

    profile = profile_mod.resolve_training_runtime_profile(None)

    assert profile.variant == ""
    assert profile.runtime_stage == "stage1"
    assert profile.rollout_runtime_owner is None
    assert profile.rollout_runtime_owned is False
    assert profile.explicit_pipeline_required is False
    assert profile.required_pipeline_namespace is None
    assert profile.ordinary_stage1_mixins_allowed is True
    assert profile.ordinary_stage1_mixins_excluded is False
    assert profile.dataset_static_packing_allowed is True
    assert profile.dataset_static_packing_owner == "dataset"
    assert profile.post_rollout_packing_owner is None
    assert profile.encoded_cache_policy == "dataset_static_cache_allowed"
    assert profile.collator_family == "default"
    assert profile.manifest_family == "stage1"


@pytest.mark.parametrize(
    ("variant", "pipeline_namespace", "manifest_family"),
    [
        ("stage2_two_channel", "stage2_ab.pipeline", "stage2_ab"),
    ],
)
def test_stage2_two_channel_profile_derives_rollout_policy_from_plan(
    variant: str,
    pipeline_namespace: str,
    manifest_family: str,
) -> None:
    profile_mod = _profile_module()

    profile = profile_mod.resolve_training_runtime_profile(variant)

    assert profile.variant == variant
    assert profile.runtime_stage == "stage2"
    assert profile.rollout_runtime_owner == "trainer"
    assert profile.rollout_runtime_owned is True
    assert profile.explicit_pipeline_required is True
    assert profile.required_pipeline_namespace == pipeline_namespace
    assert profile.ordinary_stage1_mixins_allowed is False
    assert profile.ordinary_stage1_mixins_excluded is True
    assert profile.dataset_static_packing_allowed is False
    assert profile.dataset_static_packing_owner is None
    assert profile.post_rollout_packing_owner == "trainer"
    assert profile.encoded_cache_policy == "trainer_post_rollout_dynamic"
    assert profile.collator_family == "identity"
    assert profile.manifest_family == manifest_family


@pytest.mark.parametrize(
    ("variant", "replacement"),
    [
        ("stage2_ab_training", "stage2_two_channel"),
        ("rollout_matching_sft", "stage2_two_channel"),
        ("stage2_rollout_aligned", "stage2_two_channel"),
        ("stage2_rollout_runtime", "stage2_two_channel"),
        ("stage1_set_continuation", "prefix_rollin_et_rmp_ce"),
    ],
)
def test_profile_removed_variants_fail_fast_with_replacement_guidance(
    variant: str,
    replacement: str,
) -> None:
    profile_mod = _profile_module()

    with pytest.raises(ValueError) as exc_info:
        profile_mod.resolve_training_runtime_profile(variant)

    message = str(exc_info.value)
    assert f"custom.trainer_variant={variant} has been removed" in message
    assert f"use {replacement}" in message


def test_training_runtime_profile_is_frozen() -> None:
    profile_mod = _profile_module()

    profile = profile_mod.resolve_training_runtime_profile("stage2_two_channel")

    with pytest.raises(FrozenInstanceError):
        profile.collator_family = "default"


class _BaseTrainer:
    pass


def _compose_for_variant(variant: str) -> type:
    return compose_trainer_class(
        trainer_cls=_BaseTrainer,
        trainer_variant=variant,
        instability_monitor_cfg={"enabled": True},
        token_type_cfg=SimpleNamespace(enabled=True),
        bbox_geo_cfg=SimpleNamespace(enabled=True),
        bbox_size_aux_cfg=SimpleNamespace(enabled=True),
        coord_soft_ce_w1_cfg=SimpleNamespace(enabled=True),
        sft_structural_close_cfg=SimpleNamespace(enabled=True),
        recursive_detection_ce_cfg=None,
    )


def test_compose_trainer_class_keeps_ordinary_stage1_mixins_for_default_variant() -> (
    None
):
    trainer_cls = _compose_for_variant("")

    assert issubclass(trainer_cls, GradAccumLossScaleMixin)
    assert issubclass(trainer_cls, InstabilityMonitorMixin)
    assert issubclass(trainer_cls, AggregateTokenTypeMetricsMixin)
    assert issubclass(trainer_cls, CoordSoftCEW1LossMixin)
    assert issubclass(trainer_cls, SFTStructuralCloseLossMixin)
    assert issubclass(trainer_cls, _BaseTrainer)


def test_compose_trainer_class_adds_recursive_detection_ce_mixin_when_enabled() -> None:
    trainer_cls = compose_trainer_class(
        trainer_cls=_BaseTrainer,
        trainer_variant="",
        instability_monitor_cfg=None,
        token_type_cfg=None,
        bbox_geo_cfg=None,
        bbox_size_aux_cfg=None,
        coord_soft_ce_w1_cfg=None,
        sft_structural_close_cfg=None,
        recursive_detection_ce_cfg=SimpleNamespace(enabled=True),
    )

    assert issubclass(trainer_cls, RecursiveDetectionCEMixin)
    assert issubclass(trainer_cls, GradAccumLossScaleMixin)
    assert issubclass(trainer_cls, _BaseTrainer)


@pytest.mark.parametrize(
    ("field_name", "cfg_kwargs"),
    [
        ("coord_soft_ce_w1", {"coord_soft_ce_w1_cfg": SimpleNamespace(enabled=True)}),
        (
            "sft_structural_close",
            {"sft_structural_close_cfg": SimpleNamespace(enabled=True)},
        ),
    ],
)
def test_compose_trainer_class_rejects_auxiliary_losses_with_recursive_ce(
    field_name: str,
    cfg_kwargs: dict[str, object],
) -> None:
    kwargs = {
        "trainer_cls": _BaseTrainer,
        "trainer_variant": "",
        "instability_monitor_cfg": None,
        "token_type_cfg": None,
        "bbox_geo_cfg": None,
        "bbox_size_aux_cfg": None,
        "coord_soft_ce_w1_cfg": None,
        "sft_structural_close_cfg": None,
        "recursive_detection_ce_cfg": SimpleNamespace(enabled=True),
    }
    kwargs.update(cfg_kwargs)

    with pytest.raises(ValueError, match=rf"teacher-forced target sidecars.*{field_name}"):
        compose_trainer_class(**kwargs)


@pytest.mark.parametrize(
    "variant",
    [
        "stage2_two_channel",
    ],
)
def test_compose_trainer_class_excludes_ordinary_stage1_mixins_via_profile(
    variant: str,
) -> None:
    trainer_cls = _compose_for_variant(variant)

    assert trainer_cls is _BaseTrainer
    assert not issubclass(trainer_cls, GradAccumLossScaleMixin)
    assert not issubclass(trainer_cls, CoordSoftCEW1LossMixin)
    assert not issubclass(trainer_cls, RecursiveDetectionCEMixin)


@pytest.mark.parametrize(
    ("variant", "replacement"),
    [
        ("stage2_ab_training", "stage2_two_channel"),
        ("rollout_matching_sft", "stage2_two_channel"),
        ("stage2_rollout_aligned", "stage2_two_channel"),
        ("stage2_rollout_runtime", "stage2_two_channel"),
        ("stage1_set_continuation", "prefix_rollin_et_rmp_ce"),
    ],
)
def test_compose_trainer_class_removed_variants_fail_through_profile(
    variant: str,
    replacement: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _compose_for_variant(variant)

    message = str(exc_info.value)
    assert f"custom.trainer_variant={variant} has been removed" in message
    assert f"use {replacement}" in message

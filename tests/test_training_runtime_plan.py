from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "src" / "training_runtime" / "plan.py"


def _plan_module():
    return importlib.import_module("src.training_runtime.plan")


def test_plan_module_is_import_safe() -> None:
    assert PLAN_PATH.exists()

    tree = ast.parse(PLAN_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots <= {"__future__", "dataclasses", "typing"}

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
        "import src.training_runtime.plan\n"
        f"banned = {banned_modules!r}\n"
        "loaded = [name for name in banned if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
    )


def test_training_runtime_package_root_is_import_safe() -> None:
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
        "import src.training_runtime\n"
        f"banned = {banned_modules!r}\n"
        "loaded = [name for name in banned if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
    )


def test_default_empty_and_unknown_variants_keep_generic_stage1_policy() -> None:
    plan_mod = _plan_module()

    for variant in (None, "", "legacy_custom_trainer"):
        plan = plan_mod.resolve_training_runtime_plan(variant)

        assert plan.variant == str(variant or "")
        assert plan.preserve_raw_sample_metadata is False
        assert plan.dataset_static_packing_allowed is True
        assert plan.dataset_static_packing_owner == "dataset"
        assert plan.post_rollout_packing_owner is None
        assert plan.collator_family == "default"
        assert plan.ordinary_stage1_mixins_allowed is True
        assert plan.required_pipeline_namespace is None
        assert plan.requires_top_level_rollout_matching is False


def test_stage1_set_continuation_policy_preserves_branch_owned_metadata() -> None:
    plan_mod = _plan_module()

    plan = plan_mod.resolve_training_runtime_plan("stage1_set_continuation")

    assert plan.preserve_raw_sample_metadata is True
    assert plan.dataset_static_packing_allowed is False
    assert plan.dataset_static_packing_owner is None
    assert plan.post_rollout_packing_owner is None
    assert plan.collator_family == "stage1_set_continuation"
    assert plan.ordinary_stage1_mixins_allowed is False
    assert plan.required_pipeline_namespace is None
    assert plan.requires_top_level_rollout_matching is False


@pytest.mark.parametrize(
    ("variant", "pipeline_namespace"),
    [
        ("stage2_two_channel", "stage2_ab.pipeline"),
        ("stage2_rollout_aligned", "rollout_matching.pipeline"),
    ],
)
def test_stage2_variants_share_rollout_setup_policy(
    variant: str,
    pipeline_namespace: str,
) -> None:
    plan_mod = _plan_module()

    plan = plan_mod.resolve_training_runtime_plan(variant)

    assert plan.preserve_raw_sample_metadata is True
    assert plan.dataset_static_packing_allowed is False
    assert plan.dataset_static_packing_owner is None
    assert plan.post_rollout_packing_owner == "trainer"
    assert plan.collator_family == "identity"
    assert plan.ordinary_stage1_mixins_allowed is False
    assert plan.required_pipeline_namespace == pipeline_namespace
    assert plan.requires_top_level_rollout_matching is True


@pytest.mark.parametrize(
    ("variant", "replacement"),
    [
        ("stage2_ab_training", "stage2_two_channel"),
        ("rollout_matching_sft", "stage2_rollout_aligned"),
    ],
)
def test_removed_variants_fail_fast_with_replacement_guidance(
    variant: str,
    replacement: str,
) -> None:
    plan_mod = _plan_module()

    with pytest.raises(ValueError) as exc_info:
        plan_mod.resolve_training_runtime_plan(variant)

    message = str(exc_info.value)
    assert f"custom.trainer_variant={variant} has been removed" in message
    assert f"use {replacement}" in message


def test_training_runtime_plan_is_frozen() -> None:
    plan_mod = _plan_module()

    plan = plan_mod.resolve_training_runtime_plan("stage2_two_channel")

    with pytest.raises(FrozenInstanceError):
        plan.collator_family = "default"


def test_resolved_plans_use_known_policy_vocabularies() -> None:
    plan_mod = _plan_module()

    collator_families = {"default", "stage1_set_continuation", "identity"}
    packing_owners = {"dataset", "trainer", None}
    pipeline_namespaces = {"stage2_ab.pipeline", "rollout_matching.pipeline", None}

    for variant in (
        None,
        "",
        "legacy_custom_trainer",
        "stage1_set_continuation",
        "stage2_two_channel",
        "stage2_rollout_aligned",
    ):
        plan = plan_mod.resolve_training_runtime_plan(variant)
        assert plan.collator_family in collator_families
        assert plan.dataset_static_packing_owner in packing_owners
        assert plan.post_rollout_packing_owner in packing_owners
        assert plan.required_pipeline_namespace in pipeline_namespaces

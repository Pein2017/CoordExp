"""Contract for the frozen model-free ``TrainingExecutionPlan``.

Wave 2 of ``decompose-coordexp-swift-training-orchestration`` (tasks 3.1-3.2,
design decision 2).  The plan is the immutable *value* the facade resolves
before any live runtime exists: strict config, repository root, model-free
launcher identity, a copied bounded measurement context, and entry evidence.

Construction must perform no collective, no filesystem publication, no cache
materialization, no Accelerator construction, no tokenizer/model load, and no
callback registration; the plan must not become a service locator carrying a
live model, runtime, cache, writer, or mutable lifecycle state.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
import dataclasses
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch.distributed as dist

from src.common.errors import RuntimeContractError
import src.training.execution_plan as execution_plan
from src.training.execution_plan import (
    TrainingExecutionPlan,
    build_training_execution_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = REPO_ROOT / "tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml"
OWNER_MODULE = REPO_ROOT / "src/training/execution_plan.py"

#: The exact field set design decision 2 freezes for the plan.
EXPECTED_PLAN_FIELDS = (
    "resolved_config",
    "repo_root",
    "launch_rank",
    "launch_world_size",
    "measurement_context",
    "entry_started_at",
    "entry_started_monotonic",
    "entry_resources",
)

#: Owners the plan may never reach: any of these would make construction do
#: live runtime work or turn the plan into a service locator.
FORBIDDEN_OWNER_IMPORTS = (
    "accelerate",
    "src.artifacts.run_writer",
    "src.artifacts.checkpoints",
    "src.data",
    "src.eval",
    "src.optim",
    "src.qwen",
    "src.templates",
    "src.training.control_plane",
    "src.training.forward_input_provider",
    "src.training.pack_cache",
    "src.training.pipeline",
    "src.training.session",
    "src.training.supervised_trainer",
    "torch",
)

#: Field names that would smuggle a live runtime surface into the frozen value.
FORBIDDEN_PLAN_FIELD_SUBSTRINGS = (
    "accelerator",
    "cache",
    "callback",
    "gatherer",
    "handler",
    "lifecycle",
    "model",
    "runtime",
    "tokenizer",
    "trainer",
    "writer",
)


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return tuple(sorted(names))


@pytest.fixture(autouse=True)
def _absent_launch_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


# ---------------------------------------------------------------------------
# Frozen shape
# ---------------------------------------------------------------------------


def test_plan_is_a_frozen_dataclass_with_the_exact_declared_fields() -> None:
    assert dataclasses.is_dataclass(TrainingExecutionPlan)
    assert TrainingExecutionPlan.__dataclass_params__.frozen is True
    assert tuple(
        field.name for field in dataclasses.fields(TrainingExecutionPlan)
    ) == EXPECTED_PLAN_FIELDS


def test_plan_carries_no_live_model_runtime_or_cache_field() -> None:
    observed = {field.name for field in dataclasses.fields(TrainingExecutionPlan)}

    assert not {
        name
        for name in observed
        if any(token in name for token in FORBIDDEN_PLAN_FIELD_SUBSTRINGS)
    }


def test_plan_owner_never_imports_a_live_runtime_owner() -> None:
    observed = _imported_modules(OWNER_MODULE)

    assert not [
        module
        for module in observed
        if module in FORBIDDEN_OWNER_IMPORTS
        or any(module.startswith(f"{owner}.") for owner in FORBIDDEN_OWNER_IMPORTS)
    ], observed


def test_plan_instances_reject_attribute_mutation() -> None:
    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.launch_rank = 3  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Static resolution
# ---------------------------------------------------------------------------


def test_plan_resolves_the_exact_strict_config_and_fingerprint() -> None:
    from src.config.loader import load_train_config

    expected = load_train_config(FIXTURE_CONFIG)
    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    assert plan.resolved_config.fingerprint == expected.fingerprint
    assert plan.resolved_config.schema_version == expected.schema_version
    assert plan.resolved_config.loader_version == expected.loader_version
    assert plan.resolved_config.config_dict == expected.config_dict
    assert plan.resolved_config.entry_config_path == expected.entry_config_path


def test_plan_resolves_the_repository_root_from_the_working_directory() -> None:
    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    assert plan.repo_root == Path.cwd().resolve()
    assert plan.repo_root.is_absolute()


def test_plan_defaults_launch_identity_only_when_both_fields_are_absent() -> None:
    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    assert (plan.launch_rank, plan.launch_world_size) == (0, 1)


def test_plan_admits_a_strict_decimal_launcher_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    assert (plan.launch_rank, plan.launch_world_size) == (1, 2)


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [("0", None), (None, "2"), ("01", "2"), ("0", "0"), ("2", "2"), ("-1", "2")],
)
def test_plan_rejects_an_invalid_launcher_identity(
    monkeypatch: pytest.MonkeyPatch,
    rank: str | None,
    world_size: str | None,
) -> None:
    if rank is not None:
        monkeypatch.setenv("RANK", rank)
    if world_size is not None:
        monkeypatch.setenv("WORLD_SIZE", world_size)

    with pytest.raises(RuntimeContractError) as exc_info:
        build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    assert exc_info.value.code == "runtime.preflight_launch_identity_invalid"


def test_plan_copies_the_measurement_context_and_never_aliases_the_caller() -> None:
    source: dict[str, Any] = {"run": "unit", "nested": {"arm": "a"}}

    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=source)

    assert dict(plan.measurement_context) == {"run": "unit", "nested": {"arm": "a"}}
    source["run"] = "mutated"
    assert plan.measurement_context["run"] == "unit"
    assert isinstance(plan.measurement_context, Mapping)
    with pytest.raises(TypeError):
        plan.measurement_context["run"] = "mutated"  # type: ignore[index]


def test_plan_treats_an_absent_measurement_context_as_empty() -> None:
    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    assert dict(plan.measurement_context) == {}


def test_plan_captures_bounded_entry_evidence() -> None:
    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    assert plan.entry_started_at.endswith("+00:00")
    assert isinstance(plan.entry_started_monotonic, float)
    assert isinstance(plan.entry_resources, Mapping)
    assert plan.entry_resources["schema_version"] == 1
    assert "cpu" in plan.entry_resources
    with pytest.raises(TypeError):
        plan.entry_resources["cpu"] = {}  # type: ignore[index]


def test_plan_entry_evidence_precedes_config_resolution() -> None:
    """Entry evidence must bracket the whole entry, config load included."""

    order: list[str] = []
    real_loader = execution_plan.load_train_config

    def recording_loader(path: Any) -> Any:
        order.append("load_train_config")
        return real_loader(path)

    def recording_snapshot() -> Mapping[str, Any]:
        order.append("collect_resource_snapshot")
        return {"schema_version": 1, "cpu": {}}

    original_snapshot = execution_plan.collect_resource_snapshot
    execution_plan.load_train_config = recording_loader  # type: ignore[assignment]
    execution_plan.collect_resource_snapshot = recording_snapshot  # type: ignore[assignment]
    try:
        build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)
    finally:
        execution_plan.load_train_config = real_loader  # type: ignore[assignment]
        execution_plan.collect_resource_snapshot = original_snapshot  # type: ignore[assignment]

    assert order == ["collect_resource_snapshot", "load_train_config"]


# ---------------------------------------------------------------------------
# Model-free construction
# ---------------------------------------------------------------------------


def test_plan_construction_performs_no_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("plan construction must not perform a collective")

    for name in (
        "all_gather",
        "all_gather_object",
        "all_reduce",
        "broadcast",
        "broadcast_object_list",
        "barrier",
        "init_process_group",
        "new_group",
    ):
        monkeypatch.setattr(dist, name, forbidden, raising=False)

    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    assert plan.launch_world_size == 1
    assert not dist.is_initialized()


def test_plan_construction_loads_no_model_and_builds_no_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.qwen as qwen
    import src.runtime as runtime

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("plan construction must stay model-free")

    monkeypatch.setattr(qwen, "load_qwen_components", forbidden, raising=False)
    monkeypatch.setattr(
        runtime, "seed_training_runtime", forbidden, raising=False
    )
    try:
        import accelerate

        monkeypatch.setattr(accelerate, "Accelerator", forbidden, raising=False)
    except ImportError:  # pragma: no cover - stripped environments only
        pass

    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    assert plan.resolved_config is not None


def test_plan_construction_publishes_no_artifact_and_materializes_no_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.artifacts.run_writer as run_writer
    import src.training.pack_cache as pack_cache

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("plan construction must publish nothing")

    monkeypatch.setattr(run_writer.RunWriter, "initialize", forbidden, raising=False)
    monkeypatch.setattr(pack_cache, "write_micro_step_cache", forbidden)
    monkeypatch.setattr(pack_cache, "load_rank_micro_steps_from_cache", forbidden)
    monkeypatch.chdir(tmp_path)

    before = sorted(os.listdir(tmp_path))
    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    assert sorted(os.listdir(tmp_path)) == before
    assert plan.repo_root == tmp_path.resolve()


def test_plan_construction_registers_no_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registrations: list[str] = []
    real_loader = execution_plan.load_train_config

    def recording_loader(path: Any) -> Any:
        resolved = real_loader(path)
        return SimpleNamespace(
            config=resolved.config,
            config_dict=resolved.config_dict,
            fingerprint=resolved.fingerprint,
            schema_version=resolved.schema_version,
            loader_version=resolved.loader_version,
            entry_config_path=resolved.entry_config_path,
            sources=resolved.sources,
            path_origins=resolved.path_origins,
            register=lambda *a, **k: registrations.append("register"),
        )

    monkeypatch.setattr(execution_plan, "load_train_config", recording_loader)

    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    assert registrations == []
    assert not [
        name for name in dir(plan) if name.startswith("register") or name == "on_event"
    ]


def test_plan_is_a_value_and_not_a_service_locator() -> None:
    plan = build_training_execution_plan(FIXTURE_CONFIG, measurement_context=None)

    public = {
        name
        for name in dir(plan)
        if not name.startswith("_") and callable(getattr(plan, name, None))
    }

    assert public == set()
    assert isinstance(plan.measurement_context, MappingProxyType)
    assert isinstance(plan.entry_resources, MappingProxyType)

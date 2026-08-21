"""Intended one-way training import graph and production-import inventory.

Wave 0 of ``decompose-coordexp-swift-training-orchestration`` freezes the target
import direction from design decision 1 *before* any owner module exists.  The
whole module is therefore expected RED at the Wave-0 baseline: the intended
owner modules are absent and production still imports generic identity helpers
from the historical ``src.qwen.parity`` owner.

Every boundary obligation is one parametrized node carrying the wave that is
allowed to clear it, so later wave gates diff failure sets mechanically instead
of reading prose.  A node that fails after its declared wave, or an obligation
that disappears from the observed inventory without its wave landing, is a
boundary regression rather than a snapshot-update request.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = "src"


# ---------------------------------------------------------------------------
# Declared boundary inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryObligation:
    """One declared import-boundary obligation and the wave that clears it."""

    edge_id: str
    kind: str
    subject: str
    detail: str
    clears_at_wave: int


#: Owner modules introduced by this change, with the wave that creates them.
INTENDED_OWNER_MODULES: tuple[BoundaryObligation, ...] = (
    BoundaryObligation(
        edge_id="owner-module-src-artifacts-identity",
        kind="owner_module",
        subject="src/artifacts/identity.py",
        detail="domain-neutral owner for generic identity helpers",
        clears_at_wave=1,
    ),
    BoundaryObligation(
        edge_id="owner-module-src-training-micro-steps",
        kind="owner_module",
        subject="src/training/micro_steps.py",
        detail="canonical SupervisedMicroStep and its schema identity",
        clears_at_wave=1,
    ),
    BoundaryObligation(
        edge_id="owner-module-src-training-execution-plan",
        kind="owner_module",
        subject="src/training/execution_plan.py",
        detail="immutable model-free TrainingExecutionPlan",
        clears_at_wave=2,
    ),
    BoundaryObligation(
        edge_id="owner-module-src-training-control-plane",
        kind="owner_module",
        subject="src/training/control_plane.py",
        detail="bounded RankControlPlane rank convergence",
        clears_at_wave=2,
    ),
    BoundaryObligation(
        edge_id="owner-module-src-training-cache-contract",
        kind="owner_module",
        subject="src/training/cache_contract.py",
        detail="narrow cached micro-step determinant owners",
        clears_at_wave=3,
    ),
    BoundaryObligation(
        edge_id="owner-module-src-training-cache-workflow",
        kind="owner_module",
        subject="src/training/cache_workflow.py",
        detail="cache preparation, admission, and hydration orchestration",
        clears_at_wave=3,
    ),
    BoundaryObligation(
        edge_id="owner-module-src-training-reporting",
        kind="owner_module",
        subject="src/training/reporting.py",
        detail="CompletedStepReporter for the existing logging callback",
        clears_at_wave=4,
    ),
    BoundaryObligation(
        edge_id="owner-module-src-artifacts-run-schema",
        kind="owner_module",
        subject="src/artifacts/run_schema.py",
        detail="pure RunWriter normalization and serialization",
        clears_at_wave=4,
    ),
    BoundaryObligation(
        edge_id="owner-module-src-artifacts-run-state",
        kind="owner_module",
        subject="src/artifacts/run_state.py",
        detail="pure run-state transitions and resume admission",
        clears_at_wave=4,
    ),
    BoundaryObligation(
        edge_id="owner-module-src-training-session",
        kind="owner_module",
        subject="src/training/session.py",
        detail="TrainingSession model/runtime lifetime and choreography",
        clears_at_wave=5,
    ),
)


#: Production modules that still import the historical generic identity owner.
PRODUCTION_PARITY_IMPORTS: tuple[BoundaryObligation, ...] = (
    BoundaryObligation(
        edge_id="parity-import-src-prepare-train-cache",
        kind="production_parity_import",
        subject="src/prepare_train_cache.py",
        detail="assert_absent_artifact_target, write_strict_json_atomic",
        clears_at_wave=1,
    ),
    BoundaryObligation(
        edge_id="parity-import-src-training-input-attestation",
        kind="production_parity_import",
        subject="src/training/input_attestation.py",
        detail=(
            "MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA, "
            "assert_model_weight_identity_equal, "
            "base_model_weight_identity_with_execution_policy, "
            "canonical_json_bytes, validate_model_weight_identity"
        ),
        clears_at_wave=1,
    ),
    BoundaryObligation(
        edge_id="parity-import-src-training-pipeline",
        kind="production_parity_import",
        subject="src/training/pipeline.py",
        detail=(
            "base_model_weight_identity: training assembly moves to "
            "src/training/session.py, which takes this import with it"
        ),
        clears_at_wave=5,
    ),
)


#: Owners that may never import the facade or the session.
LEAF_AND_DOMAIN_OWNERS: tuple[str, ...] = (
    "src/artifacts/identity.py",
    "src/artifacts/run_schema.py",
    "src/artifacts/run_state.py",
    "src/artifacts/run_writer.py",
    "src/losses/vocab.py",
    "src/supervision/tokens.py",
    "src/training/cache_contract.py",
    "src/training/control_plane.py",
    "src/training/execution_plan.py",
    "src/training/exact_resume.py",
    "src/training/forward_input_provider.py",
    "src/training/micro_steps.py",
    "src/training/pack_cache.py",
    "src/training/reporting.py",
    "src/training/supervised_trainer.py",
)

FORBIDDEN_UPWARD_TARGETS: tuple[str, ...] = (
    "src.training.pipeline",
    "src.training.session",
)

#: `cache_workflow.py` orchestrates caches without owning a model or a session.
CACHE_WORKFLOW_MODULE = "src/training/cache_workflow.py"
CACHE_WORKFLOW_FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "src.training.session",
    "src.training.pipeline",
)

BOUNDARY_INVENTORY: tuple[BoundaryObligation, ...] = (
    INTENDED_OWNER_MODULES + PRODUCTION_PARITY_IMPORTS
)


# ---------------------------------------------------------------------------
# Import parsing
# ---------------------------------------------------------------------------


def _module_name(path: Path, *, repo_root: Path) -> str:
    relative = path.relative_to(repo_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _owner_package_parts(path: Path, *, repo_root: Path) -> list[str]:
    """Return the package a relative import inside ``path`` resolves against.

    A package ``__init__.py`` *is* its own package, so ``from . import x`` there
    resolves inside that package. Every other module resolves against its parent
    package. Getting this wrong silently drops or misattributes reverse edges.
    """

    owner_parts = _module_name(path, repo_root=repo_root).split(".")
    if path.name == "__init__.py":
        return owner_parts
    return owner_parts[:-1]


def imported_modules(path: Path, *, repo_root: Path) -> tuple[str, ...]:
    """Return every absolute module this production file imports, ast-parsed."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package_parts = _owner_package_parts(path, repo_root=repo_root)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[
                    : max(0, len(package_parts) - (node.level - 1))
                ]
                module = ".".join([*base, node.module] if node.module else base)
            else:
                module = node.module or ""
            if module:
                names.add(module)
                # `from <pkg> import <member>` may import a SUBMODULE under a
                # plain-name alias; record the dotted form too so exact-match
                # forbidden-target sets cannot be evaded (P2-4). Non-module
                # members are a harmless over-approximation.
                for alias in node.names:
                    if alias.name != "*":
                        names.add(f"{module}.{alias.name}")
    return tuple(sorted(names))


def iter_production_modules(repo_root: Path) -> Iterator[Path]:
    for path in sorted((repo_root / PRODUCTION_ROOT).rglob("*.py")):
        if path.is_file():
            yield path


def observed_production_parity_importers(repo_root: Path) -> tuple[str, ...]:
    """Return every production file that imports ``src.qwen.parity``."""

    importers: list[str] = []
    for path in iter_production_modules(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        if relative.startswith("src/qwen/"):
            continue
        if any(
            module == "src.qwen.parity" or module.startswith("src.qwen.parity.")
            for module in imported_modules(path, repo_root=repo_root)
        ):
            importers.append(relative)
    return tuple(importers)


def observed_forbidden_upward_edges(repo_root: Path) -> tuple[str, ...]:
    """Return each ``owner -> forbidden target`` edge that currently exists."""

    edges: list[str] = []
    for relative in LEAF_AND_DOMAIN_OWNERS:
        path = repo_root / relative
        if not path.is_file():
            continue
        for module in imported_modules(path, repo_root=repo_root):
            if module in FORBIDDEN_UPWARD_TARGETS:
                edges.append(f"{relative} -> {module}")
    return tuple(sorted(edges))


def observed_cache_workflow_violations(repo_root: Path) -> tuple[str, ...]:
    """Return forbidden imports and model-loading calls in the cache workflow."""

    path = repo_root / CACHE_WORKFLOW_MODULE
    if not path.is_file():
        return ()
    violations: list[str] = []
    for module in imported_modules(path, repo_root=repo_root):
        if module in CACHE_WORKFLOW_FORBIDDEN_IMPORTS:
            violations.append(f"{CACHE_WORKFLOW_MODULE} imports {module}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "load_model"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                violations.append(
                    f"{CACHE_WORKFLOW_MODULE} calls with load_model=True at line "
                    f"{node.lineno}"
                )
    return tuple(sorted(violations))


def missing_owner_modules(repo_root: Path) -> tuple[str, ...]:
    return tuple(
        obligation.subject
        for obligation in INTENDED_OWNER_MODULES
        if not (repo_root / obligation.subject).is_file()
    )


def assert_allowed_training_imports(repo_root: Path) -> None:
    """Assert the complete intended one-way training import graph.

    Fails while any intended owner module is absent, any leaf/domain owner
    imports the facade or the session, the cache workflow imports the session or
    loads a model, or production still imports ``src.qwen.parity``.
    """

    repo_root = Path(repo_root).resolve()
    problems: list[str] = []
    for subject in missing_owner_modules(repo_root):
        problems.append(f"intended owner module is absent: {subject}")
    for edge in observed_forbidden_upward_edges(repo_root):
        problems.append(f"forbidden reverse import: {edge}")
    for violation in observed_cache_workflow_violations(repo_root):
        problems.append(f"cache workflow boundary violation: {violation}")
    for importer in observed_production_parity_importers(repo_root):
        problems.append(f"production imports src.qwen.parity: {importer}")
    if problems:
        raise AssertionError(
            "training import graph is not the intended one-way graph:\n  "
            + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _ids(obligations: Sequence[BoundaryObligation]) -> list[str]:
    return [obligation.edge_id for obligation in obligations]


def test_training_import_graph_has_no_reverse_edges() -> None:
    assert_allowed_training_imports(REPO_ROOT)


@pytest.mark.parametrize(
    "obligation", INTENDED_OWNER_MODULES, ids=_ids(INTENDED_OWNER_MODULES)
)
def test_intended_owner_module_exists(obligation: BoundaryObligation) -> None:
    assert (REPO_ROOT / obligation.subject).is_file(), (
        f"{obligation.subject} must exist by wave {obligation.clears_at_wave} "
        f"({obligation.detail})"
    )


@pytest.mark.parametrize(
    "obligation", PRODUCTION_PARITY_IMPORTS, ids=_ids(PRODUCTION_PARITY_IMPORTS)
)
def test_production_parity_import_is_cleared(obligation: BoundaryObligation) -> None:
    observed = observed_production_parity_importers(REPO_ROOT)

    assert obligation.subject not in observed, (
        f"{obligation.subject} must stop importing src.qwen.parity by wave "
        f"{obligation.clears_at_wave} ({obligation.detail})"
    )


def test_production_parity_import_inventory_is_complete() -> None:
    observed = set(observed_production_parity_importers(REPO_ROOT))
    declared = {
        obligation.subject for obligation in PRODUCTION_PARITY_IMPORTS
    }

    assert observed <= declared, (
        "undeclared production imports of src.qwen.parity: "
        f"{sorted(observed - declared)}"
    )


def test_leaf_and_domain_owners_never_import_the_facade_or_session() -> None:
    assert observed_forbidden_upward_edges(REPO_ROOT) == ()


def test_cache_workflow_owns_no_session_or_model_loading() -> None:
    assert observed_cache_workflow_violations(REPO_ROOT) == ()


def test_boundary_inventory_edge_ids_are_unique_and_waved() -> None:
    edge_ids = [obligation.edge_id for obligation in BOUNDARY_INVENTORY]

    assert len(edge_ids) == len(set(edge_ids))
    assert all(1 <= obligation.clears_at_wave <= 6 for obligation in BOUNDARY_INVENTORY)


def test_import_parser_reads_absolute_and_relative_imports(tmp_path: Path) -> None:
    package = tmp_path / "src" / "training"
    package.mkdir(parents=True)
    (package / "probe.py").write_text(
        "import src.training.pipeline\n"
        "from src.qwen.parity import canonical_json_bytes\n"
        "from . import pack_cache\n"
        "from ..artifacts import run_writer\n",
        encoding="utf-8",
    )

    observed = imported_modules(package / "probe.py", repo_root=tmp_path)

    # P2-4 (2026-08-21 review): `from <pkg> import <member>` must also record
    # the dotted member form — `from . import pack_cache` previously collapsed
    # to just "src.training", letting a forbidden submodule import evade the
    # exact-match target sets. Members that are plain names (functions) are a
    # harmless over-approximation.
    assert observed == (
        "src.artifacts",
        "src.artifacts.run_writer",
        "src.qwen.parity",
        "src.qwen.parity.canonical_json_bytes",
        "src.training",
        "src.training.pack_cache",
        "src.training.pipeline",
    )


def test_import_parser_resolves_relative_imports_inside_a_package_init(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "training"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from .pipeline import run_training_pipeline\n"
        "from . import pack_cache\n"
        "from .. import artifacts\n",
        encoding="utf-8",
    )

    observed = imported_modules(package / "__init__.py", repo_root=tmp_path)

    # A package `__init__.py` is its own package.  Resolving it as a plain module
    # would report `src.pipeline` for the first edge (hiding a forbidden reverse
    # import of `src.training.pipeline`), `src` for the second, and would drop the
    # third entirely.  Member forms are recorded too (P2-4), so
    # `from . import pack_cache` here surfaces `src.training.pack_cache`.
    assert observed == (
        "src",
        "src.artifacts",
        "src.training",
        "src.training.pack_cache",
        "src.training.pipeline",
        "src.training.pipeline.run_training_pipeline",
    )

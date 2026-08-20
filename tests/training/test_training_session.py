"""Wave-5 contract for ``src/training/session.py`` and the reduced facade.

Design decision 8 of ``decompose-coordexp-swift-training-orchestration``:
``TrainingSession`` owns the mutable, model-bearing lifetime that begins once
the control plane has admitted the model-free inputs.  These nodes pin the
fixed phase order, the owned lifecycle and resource-close behavior, the
model/runtime assembly boundary, the cache hydration inputs, the trainer
handler wiring, primary-exception preservation, best-effort failure
publication, success finalization, the exact facade result mapping, and
idempotent close.

The frozen two-rank ordered evidence lives in
``tests/training/test_orchestration_compatibility.py``; this module owns the
single-process session/facade contract that evidence cannot express.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from src.common.errors import RuntimeContractError
from src.config.models import RunDirectory
from src.training import cache_workflow, control_plane, execution_plan, pipeline, session


REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_SOURCE = REPO_ROOT / "src" / "training" / "session.py"
PIPELINE_SOURCE = REPO_ROOT / "src" / "training" / "pipeline.py"

WAVE0_FACADE_RESULT_KEYS = (
    "completed_steps",
    "consumed_micro_steps",
    "resolved_config_fingerprint",
    "run_dir",
    "run_id",
    "scheduled_event_counts",
)


# ---------------------------------------------------------------------------
# Scripted model-free entry
# ---------------------------------------------------------------------------


def _config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(
            name="session-run",
            artifact_root=str(root / "artifacts"),
            output_dir="run",
            collision_policy="fail",
        ),
        runtime=SimpleNamespace(seed=17, determinism=SimpleNamespace(mode="legacy")),
        training=SimpleNamespace(precision="no"),
        model=SimpleNamespace(attn_implementation="flash_attention_2"),
        data=SimpleNamespace(train=object(), eval=None),
    )


def _runtime_baseline_receipt() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "baseline_sha256": "a" * 64,
        "attention_backend": "flash_attention_2",
        "admitted": True,
        "mismatches": [],
        "reference_only": {},
    }


def _scripted_preflight(root: Path) -> dict[str, Any]:
    """The exact model-free bundle the facade admits and hands to the session."""

    return {
        "rank": 0,
        "world_size": 1,
        "cache_root": str(root / "cache-root"),
        "cache_root_receipt": {
            "resolved_root": str(root / "cache-root"),
            "source": "default",
        },
        "components": object(),
        "vocab_groups": object(),
        "schedule": SimpleNamespace(resolved_max_steps=1),
        "train_cache": {"fingerprint": "f" * 64},
        "eval_cache": None,
        "eval_reduction": {"effective_mode": "replicated", "pack_count": None},
        "train_micro_steps": (),
        "phase_trace": {},
    }


class _ScriptedEntry:
    """Drive one single-rank model-free entry and record its ordered events."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = tmp_path
        self.events: list[str] = []
        self.sessions: list[session.TrainingSession] = []
        self.fail_calls: list[BaseException] = []
        self.close_calls: list[int] = []
        self.run_kwargs: list[dict[str, Any]] = []
        self.preflight = _scripted_preflight(tmp_path)
        self.config_path = (tmp_path / "config.yaml").resolve()
        self.config_path.write_text("session: true\n", encoding="utf-8")
        self._install(monkeypatch)

    def _install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(self.root)
        resolved = SimpleNamespace(
            config=config,
            fingerprint="config-fingerprint",
            entry_config_path=self.config_path,
            to_artifact_dict=lambda: {"session": True},
        )
        events = self.events

        real_build_plan = execution_plan.build_training_execution_plan

        def build_plan(path: Any, **kwargs: Any) -> Any:
            events.append("plan")
            return real_build_plan(path, **kwargs)

        monkeypatch.setattr(execution_plan, "load_train_config", lambda path: resolved)
        monkeypatch.setattr(
            execution_plan, "build_training_execution_plan", build_plan
        )

        real_open = control_plane.RankControlPlane.open

        @classmethod  # type: ignore[misc]
        def opening(cls: Any, **kwargs: Any) -> Any:
            events.append("control.open")
            return real_open(**kwargs)

        monkeypatch.setattr(control_plane.RankControlPlane, "open", opening)

        real_close = control_plane.RankControlPlane.close

        def closing(plane: Any) -> None:
            events.append("control.close")
            real_close(plane)

        monkeypatch.setattr(control_plane.RankControlPlane, "close", closing)

        real_run_owner = session._initialize_model_free_run_owner

        def run_owner(**kwargs: Any) -> Any:
            events.append("writer")
            return real_run_owner(**kwargs)

        monkeypatch.setattr(
            session, "_initialize_model_free_run_owner", run_owner
        )
        monkeypatch.setattr(
            session,
            "collect_execution_provenance",
            lambda **kwargs: {"schema_version": 1},
        )
        monkeypatch.setattr(
            session,
            "require_pinned_runtime_baseline",
            lambda **kwargs: _runtime_baseline_receipt(),
        )
        monkeypatch.setattr(
            cache_workflow,
            "require_pinned_runtime_baseline",
            lambda **kwargs: _runtime_baseline_receipt(),
        )

        def admit(**kwargs: Any) -> Any:
            events.append("cache.admit")
            return self.preflight

        monkeypatch.setattr(
            cache_workflow, "_resolve_model_free_training_preflight", admit
        )
        monkeypatch.setattr(
            session,
            "_build_accelerator",
            lambda precision: SimpleNamespace(
                process_index=0, num_processes=1, is_main_process=True
            ),
        )
        monkeypatch.setattr(
            session, "validate_accelerator_runtime", lambda *a, **k: None
        )

        entry = self

        class RecordingSession(session.TrainingSession):
            def __init__(self, **kwargs: Any) -> None:
                events.append("session.init")
                super().__init__(**kwargs)
                entry.sessions.append(self)

            def run(self) -> dict[str, Any]:
                events.append("session.run")
                return entry.on_run(self)

            def fail(self, error: BaseException) -> None:
                events.append("session.fail")
                entry.fail_calls.append(error)
                super().fail(error)

            def close(self) -> None:
                events.append("session.close")
                entry.close_calls.append(1)
                super().close()

        monkeypatch.setattr(session, "TrainingSession", RecordingSession)

    def on_run(self, live: session.TrainingSession) -> dict[str, Any]:
        self.run_kwargs.append(
            {
                "accelerator": live.control_plane.accelerator,
                "cache_preflight": live.cache_preflight,
                "lifecycle": live.lifecycle,
                "writer": live.writer,
            }
        )
        if live.writer is not None:
            live.writer.finalize(
                status="completed",
                updated_at="2026-01-01T00:00:00Z",
                completed_steps=1,
                consumed_packs=2,
                checkpoint_event_count=0,
                optimizer_update_status="applied",
                finite_status="finite",
            )
        return {
            "run_dir": str(live.run_identity.run_directory.run_dir),
            "run_id": live.run_identity.run_id,
            "resolved_config_fingerprint": live.plan.resolved_config.fingerprint,
            "completed_steps": 1,
            "consumed_micro_steps": 2,
            "scheduled_event_counts": {"checkpoint": 0, "eval": 0},
        }

    def run(self) -> dict[str, Any]:
        return pipeline.run_training_pipeline(self.config_path)

    def run_state(self) -> dict[str, Any]:
        return json.loads(
            (self.root / "artifacts" / "run" / "run.json").read_text(encoding="utf-8")
        )


@pytest.fixture
def scripted_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _ScriptedEntry:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(tmp_path / "cache-root"))
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    return _ScriptedEntry(tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# Facade lifetime: exactly one session, returned result, close in finally
# ---------------------------------------------------------------------------


def test_pipeline_builds_exactly_one_session_and_closes_in_finally(
    scripted_entry: _ScriptedEntry,
) -> None:
    result = scripted_entry.run()

    assert scripted_entry.events == [
        "plan",
        "control.open",
        "writer",
        "cache.admit",
        "control.close",
        "session.init",
        "session.run",
        "session.close",
        "control.close",
    ]
    assert len(scripted_entry.sessions) == 1
    assert tuple(sorted(result)) == WAVE0_FACADE_RESULT_KEYS
    assert result["scheduled_event_counts"] == {"checkpoint": 0, "eval": 0}
    assert result["resolved_config_fingerprint"] == "config-fingerprint"
    assert scripted_entry.fail_calls == []


def test_pipeline_session_receives_the_bound_accelerator_and_admitted_preflight(
    scripted_entry: _ScriptedEntry,
) -> None:
    scripted_entry.run()

    (observed,) = scripted_entry.run_kwargs
    live = scripted_entry.sessions[0]
    # The facade binds the Accelerator through the control plane; the session
    # reads it back from the plane rather than from a seventh constructor slot.
    assert observed["accelerator"] is live.control_plane.accelerator
    assert observed["accelerator"] is not None
    # Cache hydration inputs reach the session as the exact admitted bundle.
    assert observed["cache_preflight"] is scripted_entry.preflight


def test_pipeline_success_finalization_is_the_sessions_own(
    scripted_entry: _ScriptedEntry,
) -> None:
    scripted_entry.run()

    run_state = scripted_entry.run_state()

    assert run_state["status"] == "completed"
    # Same relative order as the frozen two-rank `phase_order.json` evidence:
    # `cache_admission` completes before the two not-run cache phases are
    # recorded.  The preflight-owned phases are absent because this scripted
    # entry admits a preflight bundle without exercising the cache workflow.
    assert run_state["measurement"]["phase_order"] == [
        "config_provenance_resolution",
        "cache_admission",
        "cache_preparation",
        "cache_publication",
    ]


def test_training_session_phase_order_is_fixed_and_literal(
    scripted_entry: _ScriptedEntry,
) -> None:
    """The pre-model phases the facade owns keep their literal Wave-0 order."""

    scripted_entry.run()

    phases = scripted_entry.run_state()["measurement"]["phases"]

    assert phases["config_provenance_resolution"]["status"] == "completed"
    assert phases["cache_admission"]["status"] == "completed"
    assert phases["cache_preparation"]["status"] == "not_run"
    assert phases["cache_publication"]["status"] == "not_run"


# ---------------------------------------------------------------------------
# Failure: primary exception preserved, best-effort publication, close
# ---------------------------------------------------------------------------


def test_pipeline_failing_session_is_failed_then_closed_without_swallowing(
    scripted_entry: _ScriptedEntry,
) -> None:
    primary = RuntimeError("initialized training failed")

    def raising(_live: session.TrainingSession) -> dict[str, Any]:
        raise primary

    scripted_entry.on_run = raising  # type: ignore[method-assign]

    with pytest.raises(RuntimeError) as excinfo:
        scripted_entry.run()

    assert excinfo.value is primary
    assert scripted_entry.fail_calls == [primary]
    assert scripted_entry.close_calls == [1]
    assert scripted_entry.events[-4:] == [
        "session.run",
        "session.fail",
        "session.close",
        "control.close",
    ]
    run_state = scripted_entry.run_state()
    assert run_state["status"] == "failed"
    assert run_state["terminal_error"] == "RuntimeError: initialized training failed"


def test_pipeline_pre_session_failure_publishes_without_constructing_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(tmp_path / "cache-root"))
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    entry = _ScriptedEntry(tmp_path, monkeypatch)
    primary = RuntimeContractError("cache admission failed", code="training.scripted")

    def failing(**kwargs: Any) -> Any:
        entry.events.append("cache.admit")
        raise primary

    monkeypatch.setattr(
        cache_workflow, "_resolve_model_free_training_preflight", failing
    )

    with pytest.raises(RuntimeContractError) as excinfo:
        entry.run()

    assert excinfo.value is primary
    assert entry.sessions == []
    assert "session.init" not in entry.events
    run_state = entry.run_state()
    assert run_state["status"] == "failed"
    assert run_state["terminal_error"].startswith("RuntimeContractError:")


# ---------------------------------------------------------------------------
# Owned lifecycle, resource close, idempotence
# ---------------------------------------------------------------------------


def _bare_session(
    tmp_path: Path,
    *,
    writer: Any = None,
    preflight: Any = None,
) -> session.TrainingSession:
    plan = execution_plan.TrainingExecutionPlan(
        resolved_config=SimpleNamespace(  # type: ignore[arg-type]
            config=_config(tmp_path),
            fingerprint="fingerprint",
            entry_config_path=tmp_path / "config.yaml",
        ),
        repo_root=tmp_path,
        launch_rank=0,
        launch_world_size=1,
        measurement_context=MappingProxyType({}),
        entry_started_at="2026-01-01T00:00:00Z",
        entry_started_monotonic=0.0,
        entry_resources=MappingProxyType({}),
    )
    run_identity = session.RunIdentity(
        run_directory=RunDirectory(
            run_name="session-run",
            artifact_root=tmp_path / "artifacts",
            run_dir=tmp_path / "artifacts" / "run",
            collision_policy="fail",
        ),
        run_id="run-000000000000",
        run_segment_id="segment-" + "0" * 32,
        writer=writer,
        measurement_warmup_steps=1,
        provenance={"schema_version": 1},
        continuation_lineage=None,
    )
    plane = control_plane.RankControlPlane(rank=0, world_size=1, gatherer=None)
    plane.accelerator = SimpleNamespace(process_index=0, num_processes=1)
    return session.TrainingSession(
        plan=plan,
        control_plane=plane,
        writer=writer,
        run_identity=run_identity,
        cache_preflight=preflight,
        admitted_policies={
            "pinned_runtime_baseline": _runtime_baseline_receipt(),
            "profile_sync_timings": {"enabled": False, "source": "default"},
            "forward_input_provider": SimpleNamespace(resolved_mode="synchronous"),
        },
        lifecycle=session.new_training_lifecycle(plan=plan, run_identity=run_identity),
    )


def test_training_session_run_forwards_every_admitted_owner_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = _scripted_preflight(tmp_path)
    live = _bare_session(tmp_path, preflight=preflight)
    captured: list[dict[str, Any]] = []

    def initialized(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {"run_dir": "sentinel"}

    monkeypatch.setattr(session, "_run_initialized_training", initialized)

    assert live.run() == {"run_dir": "sentinel"}

    (observed,) = captured
    assert observed["repo_root"] is live.plan.repo_root
    assert observed["resolved_config"] is live.plan.resolved_config
    assert observed["config"] is live.plan.resolved_config.config
    assert observed["accelerator"] is live.control_plane.accelerator
    assert observed["run_directory"] is live.run_identity.run_directory
    assert observed["run_id"] == live.run_identity.run_id
    assert observed["run_segment_id"] == live.run_identity.run_segment_id
    assert observed["writer"] is live.writer
    assert observed["lifecycle"] is live.lifecycle
    assert observed["rank_report_gatherer"] is live.control_plane.gatherer
    assert observed["preflight"] is preflight
    assert observed["provenance"] is live.run_identity.provenance
    assert observed["continuation_lineage"] is live.run_identity.continuation_lineage
    assert (
        observed["pinned_runtime_baseline"]
        is live.admitted_policies["pinned_runtime_baseline"]
    )
    assert (
        observed["profile_sync_timings"]
        is live.admitted_policies["profile_sync_timings"]
    )
    assert (
        observed["resolved_forward_input_provider"]
        is live.admitted_policies["forward_input_provider"]
    )


def test_training_session_admitted_policies_are_a_read_only_copy(
    tmp_path: Path,
) -> None:
    source = {
        "pinned_runtime_baseline": {},
        "profile_sync_timings": {},
        "forward_input_provider": object(),
    }
    live = _bare_session(tmp_path)
    live.admitted_policies = MappingProxyType(dict(source))

    source["forward_input_provider"] = "replaced"

    assert live.admitted_policies["forward_input_provider"] != "replaced"
    with pytest.raises(TypeError):
        live.admitted_policies["forward_input_provider"] = "mutated"  # type: ignore[index]


def test_training_session_owns_the_lifecycle_counters_at_entry_values(
    tmp_path: Path,
) -> None:
    live = _bare_session(tmp_path)

    assert live.lifecycle["completed_steps"] == 0
    assert live.lifecycle["consumed_packs"] == 0
    assert live.lifecycle["checkpoint_event_count"] == 0
    assert live.lifecycle["active_phase"] is None
    assert live.lifecycle["measurement_warmup_steps"] == 1
    assert live.lifecycle["world_size"] == 1
    assert live.lifecycle["entry_started_at"] == "2026-01-01T00:00:00Z"
    assert live.lifecycle["phase_rank_receipts"] == {}


def test_training_session_close_is_idempotent_and_releases_the_sync_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    released: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        session,
        "set_qwen_profile_sync_timing_policy",
        lambda value: released.append(("qwen", value)),
    )
    monkeypatch.setattr(
        session,
        "set_trainer_profile_sync_timing_policy",
        lambda value: released.append(("trainer", value)),
    )
    live = _bare_session(tmp_path)

    live.close()
    live.close()
    live.close()

    assert released == [("qwen", None), ("trainer", None)]


def test_training_session_close_never_performs_a_success_finalization(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    writer = SimpleNamespace(
        finalize=lambda **kwargs: calls.append("finalize"),
        read_run=lambda: calls.append("read_run"),
    )
    live = _bare_session(tmp_path, writer=writer)

    live.close()

    assert calls == []


def test_training_session_fail_never_replaces_the_primary_exception(
    tmp_path: Path,
) -> None:
    class _HostileWriter:
        def __getattr__(self, name: str) -> Any:
            def explode(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(f"writer.{name} is unavailable")

            return explode

    live = _bare_session(tmp_path, writer=_HostileWriter())

    # Every publication step is individually guarded; `fail` returns normally so
    # the caller's `raise` re-raises the primary error untouched.
    assert live.fail(ValueError("primary")) is None


def test_training_session_fail_is_a_no_op_without_a_rank_zero_writer(
    tmp_path: Path,
) -> None:
    live = _bare_session(tmp_path, writer=None)

    assert live.fail(ValueError("primary")) is None


# ---------------------------------------------------------------------------
# Ownership boundary: assembly, handlers, and the reduced facade
# ---------------------------------------------------------------------------


SESSION_OWNED_SURFACE = (
    "_run_initialized_training",
    "_checkpoint_handler",
    "_eval_forward_handler",
    "_final_handler",
    "_build_accelerator",
    "_initialize_artifact_owner",
    "_initialize_model_free_run_owner",
    "_begin_run_phase",
    "_finish_run_phase",
    "enable_training_memory_savers",
    "build_optimizer_and_scheduler",
    "load_qwen_components",
    "setup_dora_adapter",
    "install_special_token_embedding_deltas",
    "TrainRuntime",
    "SupervisedTrainer",
    "validate_accelerator_runtime",
    "base_model_weight_identity",
)


@pytest.mark.parametrize("name", SESSION_OWNED_SURFACE)
def test_session_owns_the_model_and_runtime_assembly_surface(name: str) -> None:
    assert hasattr(session, name), f"src/training/session.py must own {name}"
    assert not hasattr(pipeline, name), (
        f"wave 5 moved {name} to src/training/session.py; the facade must delete "
        "it rather than forward through it"
    )


def test_facade_defines_only_its_public_surface() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text(encoding="utf-8"))
    defined = tuple(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )

    assert defined == ("run_training_pipeline", "prepare_training_pack_caches")


def test_facade_holds_no_pass_through_helper_layer() -> None:
    """No module-level assignment may re-publish a moved owner's symbol."""

    tree = ast.parse(PIPELINE_SOURCE.read_text(encoding="utf-8"))
    republished = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.ImportFrom))
        and not (
            isinstance(node, ast.ImportFrom)
            and node.module
            in ("__future__", "collections.abc", "pathlib", "typing", "src.training")
        )
    ]

    assert republished == []


def test_session_takes_the_parity_import_repointed_to_the_identity_owner() -> None:
    tree = ast.parse(SESSION_SOURCE.read_text(encoding="utf-8"))
    sources = {
        alias.name: node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert sources["base_model_weight_identity"] == "src.artifacts.identity"
    assert not any(
        module == "src.qwen.parity"
        for module in sources.values()
        if module is not None
    )


def test_session_introduces_no_phase_subclass_registry_or_backend() -> None:
    tree = ast.parse(SESSION_SOURCE.read_text(encoding="utf-8"))
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    session_subclasses = [
        node.name
        for node in classes
        if any(
            isinstance(base, ast.Name) and base.id == "TrainingSession"
            for base in node.bases
        )
    ]

    assert session_subclasses == []
    (training_session,) = [node for node in classes if node.name == "TrainingSession"]
    assert training_session.bases == []
    assert [
        node.name
        for node in training_session.body
        if isinstance(node, ast.FunctionDef)
    ] == ["__init__", "run", "fail", "close"]


def _initialized_training_trainer_call() -> ast.Call:
    tree = ast.parse(SESSION_SOURCE.read_text(encoding="utf-8"))
    (initialized,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_initialized_training"
    ]
    calls = [
        node
        for node in ast.walk(initialized)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SupervisedTrainer"
    ]
    assert len(calls) == 1
    return calls[0]


def test_session_wires_every_trainer_handler_to_its_owner() -> None:
    call = _initialized_training_trainer_call()
    producers = {}
    for keyword in call.keywords:
        value = keyword.value
        if isinstance(value, ast.Call):
            producers[keyword.arg] = ast.unparse(value.func)
        elif isinstance(value, ast.Name):
            producers[keyword.arg] = value.id

    assert producers["on_completed_step"] == "reporting.CompletedStepReporter"
    assert producers["on_eval"] == "_eval_forward_handler"
    assert producers["on_final"] == "_final_handler"
    assert producers["on_checkpoint"] == "checkpoint_handler"
    assert producers["forward_input_provider"] == "forward_input_provider"


def test_session_owns_the_forward_input_provider_lifetime() -> None:
    tree = ast.parse(SESSION_SOURCE.read_text(encoding="utf-8"))
    (initialized,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_initialized_training"
    ]
    closes = [
        node
        for node in ast.walk(initialized)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "close"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "forward_input_provider"
    ]

    assert len(closes) == 1
    # The provider is closed in the trainer `finally`, not left to the facade.
    assert any(
        isinstance(node, ast.Try)
        and any(
            close is inner
            for inner in ast.walk(ast.Module(body=node.finalbody, type_ignores=[]))
            for close in closes
        )
        for node in ast.walk(initialized)
    )


def test_session_owns_cache_hydration_and_the_facade_does_not() -> None:
    facade = PIPELINE_SOURCE.read_text(encoding="utf-8")
    owner = SESSION_SOURCE.read_text(encoding="utf-8")

    for hydration in (
        "_hydrate_eval_micro_steps_from_cache",
        "load_rank_micro_steps_from_cache",
        "_attach_image_processors_to_micro_steps",
        "_resolve_eval_pack_cache",
    ):
        assert hydration in owner, hydration
        assert hydration not in facade, hydration


# ---------------------------------------------------------------------------
# Behavioral terminal optimizer-boundary seam (Wave-3 carried obligation W3-3,
# discharged by add-coordexp-swift-training-observability task 4.1)
#
# The Wave-3 receipts proved the reporting seam behaviorally and the SESSION
# seam only by source inspection.  These nodes drive the real
# `_run_initialized_training` body with a stub trainer whose `run()` raises the
# converged `OptimizerBoundaryTerminal`, a REAL `RunWriter`, and a real
# `ObservationPublisher`, and read the resulting artifacts.
# ---------------------------------------------------------------------------


def _terminal_seam_config(root: Path, *, observability_steps: int = 1) -> SimpleNamespace:
    dataset = SimpleNamespace(path=root / "train.jsonl", sample_limit=2)
    dataset.path.write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        run=SimpleNamespace(
            name="terminal-run",
            artifact_root=str(root / "artifacts"),
            output_dir="run",
            collision_policy="fail",
        ),
        runtime=SimpleNamespace(seed=7, determinism=SimpleNamespace(mode="legacy")),
        training=SimpleNamespace(precision="no", max_grad_norm=1.0),
        model=SimpleNamespace(
            special_token_embeddings=object(),
            attn_implementation="flash_attention_2",
            fa2_branch_proof="first_micro_step",
        ),
        adapter=object(),
        packing=SimpleNamespace(
            global_max_length=7,
            policy="source_order_next_fit",
            window_size=None,
            lookahead=None,
            seed=17,
            worker_count=1,
            cursor_byte_budget=65_536,
            max_packs_per_fragment=None,
            fragment_item_budget=1_024,
            fragment_byte_budget=4_194_304,
        ),
        template=SimpleNamespace(object_ordering="geo_sorted"),
        losses=object(),
        optimizer=object(),
        data=SimpleNamespace(train=dataset, eval=None, train_order="source_order"),
        checkpoint=SimpleNamespace(save_final=False),
        observability=SimpleNamespace(steps=observability_steps),
    )


class _TerminalSeamHarness:
    """Drive the real `_run_initialized_training` body up to `trainer.run()`."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        is_main: bool = True,
    ) -> None:
        from src.artifacts.run_writer import RunWriter
        from src.runtime.optimizer_boundary import (
            AppliedUpdateReceipt,
            OptimizerBoundaryTerminal,
        )

        self.root = tmp_path
        self.is_main = is_main
        self.config = _terminal_seam_config(tmp_path)
        self.eval_dispatches: list[Any] = []
        self.checkpoint_dispatches: list[Any] = []
        self.final_dispatches: list[Any] = []
        self.console = _RecordingConsole()
        self.receipt = AppliedUpdateReceipt.terminal_not_attempted(
            42, 1, "pre_wrapper_mixed_scaler_overflow", unscale_completed=True
        )
        self.terminal = OptimizerBoundaryTerminal(self.receipt)
        self.writer = (
            RunWriter.initialize(
                run_dir=tmp_path / "run",
                run_id="run",
                run_name="terminal-run",
                artifact_root=tmp_path,
                collision_outcome="created",
                created_at="now",
                config_fingerprint="fp",
                resolved_config={},
                world_size=1,
                resolved_max_steps=None,
            )
            if is_main
            else None
        )
        self.accelerator = SimpleNamespace(
            is_main_process=is_main,
            num_processes=1,
            process_index=0,
            device="cpu",
        )
        self.runtime = SimpleNamespace(
            model=object(),
            accelerator=self.accelerator,
            is_main_process=is_main,
            world_size=1,
            optimizer_step_count=5,
            scheduler_step_count=5,
            validate_eval_reduction_consensus=lambda **kwargs: None,
        )
        self._install(monkeypatch)

    def _install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        components = SimpleNamespace(
            model=SimpleNamespace(to=lambda device: None),
            token_identity=SimpleNamespace(
                tokenizer_vocab_size=10, to_artifact_dict=lambda: {"tokens": 1}
            ),
            processor_identity=SimpleNamespace(
                to_artifact_dict=lambda: {"processor": 1}
            ),
            tokenizer=object(),
            base_model_path=self.root / "model",
            base_config_sha256="base",
            tokenizer_sha256="tokenizer",
            processor=SimpleNamespace(image_processor=object()),
        )
        self.components = components
        model = SimpleNamespace(name="model")
        monkeypatch.setattr(
            cache_workflow, "seed_training_runtime", lambda *a, **k: None, raising=False
        )
        for module in (session, cache_workflow):
            monkeypatch.setattr(
                module, "load_qwen_components", lambda *a, **k: components,
                raising=False,
            )
        monkeypatch.setattr(
            session, "load_default_adapter_source_gate_evidence", lambda root: object()
        )
        monkeypatch.setattr(
            session, "build_adapter_setup_plan",
            lambda *a, **k: SimpleNamespace(mode="fresh"),
        )
        monkeypatch.setattr(
            session, "setup_dora_adapter",
            lambda m, plan: SimpleNamespace(
                model=model, receipt=SimpleNamespace(adapter_name="default")
            ),
        )
        monkeypatch.setattr(
            session, "build_default_special_token_selection", lambda *a: object()
        )
        monkeypatch.setattr(
            session,
            "load_default_special_token_embedding_source_gate_evidence",
            lambda root: object(),
        )
        monkeypatch.setattr(
            session, "install_special_token_embedding_deltas",
            lambda m, selection, source_gate: SimpleNamespace(
                model=model, receipt=object()
            ),
        )
        monkeypatch.setattr(session, "enable_training_memory_savers", lambda m: None)
        monkeypatch.setattr(
            session,
            "_move_model_and_resolve_mapped_native_execution",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            cache_workflow,
            "_attach_image_processors_to_micro_steps",
            lambda steps, **kwargs: tuple(steps),
        )
        monkeypatch.setattr(
            cache_workflow,
            "_apply_fa2_branch_proof_policy",
            lambda steps, config: tuple(steps),
        )
        monkeypatch.setattr(session.LossRunner, "from_config", lambda config: object())
        monkeypatch.setattr(
            session, "build_optimizer_group_plan", lambda *a, **k: object()
        )
        monkeypatch.setattr(session, "build_scheduler_plan", lambda *a, **k: object())
        monkeypatch.setattr(
            session, "build_optimizer_and_scheduler",
            lambda *a, **k: (object(), object()),
        )
        monkeypatch.setattr(
            session, "build_trainable_surface_receipt", lambda *a, **k: object()
        )
        monkeypatch.setattr(session, "TrainRuntime", lambda **kwargs: self.runtime)
        monkeypatch.setattr(session, "CheckpointWriter", lambda run_dir: object())
        monkeypatch.setattr(
            session,
            "_checkpoint_handler",
            lambda *a, **k: lambda *args, **kwargs: self.checkpoint_dispatches.append(
                args
            ),
        )
        monkeypatch.setattr(
            session,
            "_eval_forward_handler",
            lambda **kwargs: lambda *args: self.eval_dispatches.append(args),
        )
        monkeypatch.setattr(
            session,
            "_final_handler",
            lambda **kwargs: lambda *args: self.final_dispatches.append(args),
        )

        harness = self

        def trainer(**kwargs: Any) -> Any:
            harness.trainer_kwargs = kwargs

            def run() -> Any:
                raise harness.terminal

            return SimpleNamespace(run=run)

        monkeypatch.setattr(session, "SupervisedTrainer", trainer)

    @property
    def preflight(self) -> dict[str, Any]:
        return {
            "rank": 0,
            "world_size": 1,
            "cache_root": str(self.root / "cache-root"),
            "cache_root_receipt": {
                "resolved_root": str(self.root / "cache-root"),
                "source": "default",
            },
            "components": self.components,
            "vocab_groups": object(),
            "schedule": SimpleNamespace(
                resolved_max_steps=100, runtime_batch=object()
            ),
            "train_cache": {
                "cache_dir": self.root / "cache",
                "micro_step_count": 1,
                "format_version": "v1",
                "fingerprint": "f" * 64,
                "determinants_sha256": "d" * 64,
                "phase_receipt": {},
            },
            "eval_cache": None,
            "eval_reduction": {"effective_mode": "replicated", "pack_count": None},
            "train_micro_steps": (SimpleNamespace(split="train"),),
            "phase_trace": {},
        }

    def run(self) -> dict[str, Any]:
        gatherer = lambda report: (report,)  # noqa: E731
        gatherer.close = lambda: None  # type: ignore[attr-defined]
        return session._run_initialized_training(
            repo_root=self.root,
            resolved_config=SimpleNamespace(
                entry_config_path=self.root / "config.yaml", fingerprint="config-fp"
            ),
            config=self.config,
            accelerator=self.accelerator,
            run_directory=RunDirectory("run", self.root, self.root / "run", "created"),
            run_id="run",
            run_segment_id="segment-run",
            writer=self.writer,
            lifecycle={
                "completed_steps": 0,
                "consumed_packs": 0,
                "checkpoint_event_count": 0,
                "optimizer_update_status": None,
                "finite_status": None,
            },
            rank_report_gatherer=gatherer,
            resolved_forward_input_provider=session.resolve_forward_input_provider_mode(
                "synchronous"
            ),
            provenance={"schema_version": 3},
            preflight=self.preflight,
        )

    def logging_rows(self) -> list[dict[str, Any]]:
        path = self.root / "run" / "logging.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


class _RecordingConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: str) -> int:
        if text.strip():
            self.lines.append(text)
        return len(text)

    def flush(self) -> None:
        return None


def test_session_publishes_one_terminal_row_before_the_boundary_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.runtime.optimizer_boundary import OptimizerBoundaryTerminal

    harness = _TerminalSeamHarness(tmp_path, monkeypatch)

    with pytest.raises(OptimizerBoundaryTerminal) as excinfo:
        harness.run()

    # The PRIMARY optimizer-boundary failure is what leaves the session.
    assert excinfo.value is harness.terminal
    assert excinfo.value.receipt.terminal_reason == "pre_wrapper_mixed_scaler_overflow"

    (row,) = harness.logging_rows()
    assert row["step"] == 42
    assert row["split"] == "train"
    assert row["optimizer_boundary_terminal"] is True
    assert row["optimizer_terminal_reason"] == "pre_wrapper_mixed_scaler_overflow"
    assert row["optimizer_update_applied"] is False
    assert row["optimizer_step_count"] == 5
    assert row["scheduler_step_count"] == 5

    # No scheduled handler ran, and no success finalization was published.
    assert harness.eval_dispatches == []
    assert harness.checkpoint_dispatches == []
    assert harness.final_dispatches == []
    assert harness.writer.read_run()["status"] != "completed"
    assert not (harness.writer.run_dir / "final.json").exists()


def test_session_terminal_row_publication_failure_keeps_the_boundary_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.artifacts.run_writer import RunWriter
    from src.runtime.optimizer_boundary import OptimizerBoundaryTerminal

    harness = _TerminalSeamHarness(tmp_path, monkeypatch)
    attempts: list[dict[str, Any]] = []

    def failing_append(self: Any, row: Any) -> Any:
        attempts.append(dict(row))
        raise OSError("disk full")

    monkeypatch.setattr(RunWriter, "append_logging_row", failing_append)

    with pytest.raises(OptimizerBoundaryTerminal) as excinfo:
        harness.run()

    # The bounded primary optimizer-boundary failure remains the terminal cause
    # ahead of the row-publication failure, and every rank still converges it.
    assert excinfo.value is harness.terminal
    # Publication was attempted exactly once for the terminal planned step.
    assert [row["step"] for row in attempts] == [42]
    assert harness.logging_rows() == []
    assert harness.eval_dispatches == []
    assert harness.checkpoint_dispatches == []
    assert harness.final_dispatches == []
    assert harness.writer.read_run()["status"] != "completed"


def test_session_presents_the_terminal_row_only_after_it_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The publisher is wired from the session, and presentation is derived."""

    from src.runtime.optimizer_boundary import OptimizerBoundaryTerminal

    harness = _TerminalSeamHarness(tmp_path, monkeypatch)

    with pytest.raises(OptimizerBoundaryTerminal):
        harness.run()

    presented = capsys.readouterr().err
    assert "42/100" in presented
    assert len(harness.logging_rows()) == 1


def test_session_presents_nothing_when_the_terminal_row_cannot_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.artifacts.run_writer import RunWriter
    from src.runtime.optimizer_boundary import OptimizerBoundaryTerminal

    harness = _TerminalSeamHarness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        RunWriter,
        "append_logging_row",
        lambda self, row: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OptimizerBoundaryTerminal):
        harness.run()

    assert "42/100" not in capsys.readouterr().err
    assert harness.logging_rows() == []


def test_a_non_main_rank_writes_no_progress_and_no_event_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.runtime.optimizer_boundary import OptimizerBoundaryTerminal

    harness = _TerminalSeamHarness(tmp_path, monkeypatch, is_main=False)

    with pytest.raises(OptimizerBoundaryTerminal):
        harness.run()

    assert capsys.readouterr().err.strip() == ""
    assert harness.logging_rows() == []
    assert not (tmp_path / "run" / "tensorboard").exists()

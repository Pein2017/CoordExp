from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.training.pipeline as pipeline
from src.artifacts.run_writer import RunWriter
from src.common.errors import RuntimeContractError
from src.config.models import RunDirectory
from src.training.supervised_trainer import CompletedStepObservation


class _Accelerator:
    is_main_process = True
    num_processes = 1
    process_index = 0


class _Runtime:
    is_main_process = True
    world_size = 1
    accelerator = _Accelerator()

    def gather_metrics(self, metrics: object, **kwargs: object) -> object:
        return {"metrics": dict(metrics)}


def _writer(tmp_path: Path) -> RunWriter:
    return RunWriter.initialize(
        run_dir=tmp_path / "run", run_id="run", run_name="run",
        artifact_root=tmp_path, collision_outcome="created", created_at="now",
        config_fingerprint="fp", resolved_config={}, world_size=1,
        resolved_max_steps=5,
    )


def _observation(step: int) -> CompletedStepObservation:
    return CompletedStepObservation(
        planned_step_id=step,
        micro_step_count=2,
        loss_bundle_artifact={"metrics": {"loss/total": 1.0, "acc_top1": 0.5, "acc_top5": 1.0}},
        optimizer_update_status="applied",
        finite_status="finite",
        scheduler_artifact={"learning_rates": [{"group_index": 0, "lr": 1e-5}]},
    )


def test_five_train_and_two_eval_callbacks_write_exact_wide_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {}
    runtime = _Runtime()
    train = pipeline._train_logging_handler(writer, lifecycle, runtime)

    class FakeEvalRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, *, planned_step_id: int, trigger_reasons: object) -> object:
            return SimpleNamespace(to_logging_row=lambda: {
                "step": planned_step_id, "split": "eval", "trigger_reasons": list(trigger_reasons),
                "example_count": 2, "pack_count": 1, "acc_top1": 0.6,
            })

    monkeypatch.setattr(pipeline, "ForwardEvalRunner", FakeEvalRunner)
    eval_handler = pipeline._eval_forward_handler(
        model=object(), runtime=runtime, eval_micro_steps=(), loss_runner=object(),
        writer=writer, eval_source={"path": "eval.jsonl"}, eval_by_step={},
    )
    for step in range(1, 6):
        train(_observation(step))
        if step in {2, 4}:
            eval_handler(SimpleNamespace(planned_step_id=step, trigger_reasons=("scheduled",)), _observation(step))

    rows = [json.loads(line) for line in writer.logging_path.read_text().splitlines()]
    assert [row["split"] for row in rows].count("train") == 5
    assert [row["split"] for row in rows].count("eval") == 2
    assert lifecycle["completed_steps"] == 5
    assert lifecycle["consumed_packs"] == 10
    assert all("non_finite_fields" in row for row in rows)


def test_peer_artifact_initialization_returns_no_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accelerator = SimpleNamespace(is_main_process=False, num_processes=2)
    monkeypatch.setattr(
        pipeline, "broadcast_object_list",
        lambda values, from_process=0: values.__setitem__(0, {"ok": True}),
    )
    resolved = SimpleNamespace(fingerprint="fp", to_artifact_dict=lambda: {})
    result = pipeline._initialize_artifact_owner(
        accelerator=accelerator,
        run_directory=RunDirectory("run", tmp_path, tmp_path / "run", "created"),
        run_id="run", created_at="now", resolved_config=resolved,
    )
    assert result is None
    assert not (tmp_path / "run").exists()


def test_rank_zero_writer_initialization_failure_is_shared_and_preserves_collision_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    existing = run_dir / "existing.txt"
    existing.write_text("keep")
    resolved = SimpleNamespace(fingerprint="fp", to_artifact_dict=lambda: {})
    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline._initialize_artifact_owner(
            accelerator=_Accelerator(),
            run_directory=RunDirectory("run", tmp_path, run_dir, "fail"),
            run_id="run", created_at="now", resolved_config=resolved,
        )
    assert exc_info.value.code == "runtime.artifact_initialization_failed"
    assert existing.read_text() == "keep"


def test_failed_lifecycle_state_keeps_progress_before_original_error(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {"checkpoint_event_count": 1}
    callback = pipeline._train_logging_handler(writer, lifecycle, _Runtime())
    callback(_observation(1))
    callback(_observation(2))
    original = RuntimeError("injected after two steps")
    writer.finalize(
        status="failed", updated_at="later",
        completed_steps=int(lifecycle["completed_steps"]),
        consumed_packs=int(lifecycle["consumed_packs"]),
        checkpoint_event_count=int(lifecycle["checkpoint_event_count"]),
        optimizer_update_status=str(lifecycle["optimizer_update_status"]),
        finite_status=str(lifecycle["finite_status"]), terminal_error=str(original),
    )
    state = writer.read_run()
    assert state["completed_steps"] == 2 and state["consumed_packs"] == 4
    assert state["checkpoint_event_count"] == 1
    assert state["terminal_error"] == "injected after two steps"


def test_train_logging_uses_all_rank_reduced_scalars_and_preserves_nonfinite(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    calls: list[dict[str, object]] = []

    class Runtime(_Runtime):
        world_size = 2

        def gather_metrics(self, metrics: object, **kwargs: object) -> object:
            calls.append(dict(metrics))
            return {"metrics": {"loss/total": 2.0, "acc_top1": float("inf")}}

    pipeline._train_logging_handler(writer, {}, Runtime())(_observation(1))
    row = json.loads(writer.logging_path.read_text())
    assert calls == [{"lr/group_0": 1e-5, "loss/total": 1.0, "acc_top1": 0.5, "acc_top5": 1.0}]
    assert row["loss/total"] == 2.0
    assert row["acc_top1"] is None
    assert row["non_finite_fields"] == ["acc_top1"]


def test_rank_zero_logging_failure_is_broadcast_as_shared_named_error(
    tmp_path: Path,
) -> None:
    shared: dict[str, object] = {}

    class Collective:
        num_processes = 2
        is_main_process = True

        def broadcast_object_list(self, values: list[object], from_process: int = 0) -> None:
            if self.is_main_process:
                shared["status"] = values[0]
            else:
                values[0] = shared["status"]

    accelerator = Collective()
    main_runtime = SimpleNamespace(
        accelerator=accelerator, is_main_process=True, world_size=2,
    )
    peer_accelerator = Collective()
    peer_accelerator.is_main_process = False
    peer_runtime = SimpleNamespace(
        accelerator=peer_accelerator, is_main_process=False, world_size=2,
    )
    failing_writer = SimpleNamespace(
        append_logging_row=lambda row: (_ for _ in ()).throw(OSError("disk full"))
    )
    for runtime, writer in ((main_runtime, failing_writer), (peer_runtime, None)):
        with pytest.raises(RuntimeContractError) as exc_info:
            pipeline._append_logging_row_shared(
                writer=writer, row={"step": 1, "split": "train"}, runtime=runtime
            )
        assert exc_info.value.code == "runtime.logging_append_failed"
        assert "OSError: disk full" in str(exc_info.value)


@pytest.mark.parametrize(("save_final", "expected_calls"), [(False, 0), (True, 1)])
def test_final_handler_respects_save_final(
    save_final: bool, expected_calls: int
) -> None:
    calls: list[int] = []
    handler = pipeline._final_handler(
        checkpoint_handler=lambda event, observation: calls.append(event.planned_step_id),
        committed_steps=set(),
        save_final=save_final,
    )
    handler(SimpleNamespace(planned_step_id=5), _observation(5))
    assert len(calls) == expected_calls


def test_final_handler_deduplicates_same_step_explicit_checkpoint() -> None:
    calls: list[int] = []
    handler = pipeline._final_handler(
        checkpoint_handler=lambda event, observation: calls.append(event.planned_step_id),
        committed_steps={5},
        save_final=True,
    )
    handler(SimpleNamespace(planned_step_id=5), _observation(5))
    assert calls == []


def test_pre_trainer_failure_finalizes_initialized_run_without_masking_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config = SimpleNamespace(
        training=SimpleNamespace(precision="no"),
        runtime=SimpleNamespace(seed=7),
        run=SimpleNamespace(name="run"),
    )
    resolved = SimpleNamespace(config=config, fingerprint="fp")
    accelerator = _Accelerator()
    holder: dict[str, RunWriter] = {}
    monkeypatch.setattr(pipeline, "load_train_config", lambda path: resolved)
    monkeypatch.setattr(pipeline, "_build_accelerator", lambda precision: accelerator)
    monkeypatch.setattr(pipeline, "validate_accelerator_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline, "_resolve_shared_run_directory",
        lambda *args, **kwargs: RunDirectory("run", tmp_path, run_dir, "created"),
    )

    def initialize(**kwargs: object) -> RunWriter:
        writer = RunWriter.initialize(
            run_dir=run_dir, run_id="run", run_name="run", artifact_root=tmp_path,
            collision_outcome="created", created_at="now", config_fingerprint="fp",
            resolved_config={}, world_size=1,
        )
        holder["writer"] = writer
        return writer

    monkeypatch.setattr(pipeline, "_initialize_artifact_owner", initialize)
    monkeypatch.setattr(
        pipeline, "_run_initialized_training",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("pre-trainer failure")),
    )
    with pytest.raises(ValueError, match="pre-trainer failure"):
        pipeline.run_training_pipeline(tmp_path / "config.yaml")
    state = holder["writer"].read_run()
    assert state["status"] == "failed"
    assert state["completed_steps"] == 0 and state["consumed_packs"] == 0
    assert state["terminal_error"] == "ValueError: pre-trainer failure"


def test_prepare_training_pack_caches_is_model_free_and_covers_train_and_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(
        runtime=SimpleNamespace(seed=17),
        data=SimpleNamespace(train=object(), eval=object()),
    )
    resolved = SimpleNamespace(
        config=config,
        fingerprint="config-fingerprint",
        entry_config_path=tmp_path / "config.yaml",
    )
    components = SimpleNamespace(token_identity=object(), tokenizer=object())
    load_model_values: list[bool] = []
    seed_phases: list[str] = []
    train_verification_levels: list[str] = []
    train_cache = {
        "status": "complete",
        "build_status": "built",
        "cache_dir": tmp_path / "train-cache",
        "format_version": "v2",
        "fingerprint": "train-fingerprint",
        "manifest_path": tmp_path / "train-cache" / "manifest.json",
        "manifest_sha256": "train-manifest",
        "micro_step_count": 11,
    }
    eval_cache = {
        **train_cache,
        "cache_dir": tmp_path / "eval-cache",
        "fingerprint": "eval-fingerprint",
        "manifest_path": tmp_path / "eval-cache" / "manifest.json",
        "manifest_sha256": "eval-manifest",
        "micro_step_count": 3,
    }

    monkeypatch.setattr(pipeline, "load_train_config", lambda path: resolved)
    monkeypatch.setattr(
        pipeline,
        "seed_training_runtime",
        lambda seed, deterministic, phase: seed_phases.append(phase),
    )

    def load_components(config: object, *, load_model: bool) -> object:
        load_model_values.append(load_model)
        return components

    monkeypatch.setattr(pipeline, "load_qwen_components", load_components)
    monkeypatch.setattr(
        pipeline, "build_token_vocabulary_groups", lambda *args, **kwargs: object()
    )

    def resolve_train_cache(*args: object, **kwargs: object) -> dict[str, object]:
        train_verification_levels.append(str(kwargs["verification_level"]))
        return train_cache

    monkeypatch.setattr(
        pipeline, "_resolve_or_build_train_pack_cache", resolve_train_cache
    )
    monkeypatch.setattr(
        pipeline, "_resolve_eval_pack_cache", lambda *args, **kwargs: eval_cache
    )

    result = pipeline.prepare_training_pack_caches(tmp_path / "config.yaml")

    assert load_model_values == [False]
    assert seed_phases == ["pack_cache_preparation"]
    assert train_verification_levels == ["payloads"]
    assert result["model_loaded"] is False
    assert result["train"]["fingerprint"] == "train-fingerprint"
    assert result["eval"]["fingerprint"] == "eval-fingerprint"
    assert result["train"]["micro_step_count"] == 11


def test_same_dataset_eval_resolves_distinct_full_cache_and_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = SimpleNamespace(path=tmp_path / "shared.jsonl", sample_limit=2)
    dataset.path.write_text("{}\n")
    config = SimpleNamespace(
        runtime=SimpleNamespace(seed=7),
        training=SimpleNamespace(precision="no", max_grad_norm=1.0),
        model=SimpleNamespace(special_token_embeddings=object()),
        adapter=object(), template=object(), losses=object(), optimizer=object(),
        data=SimpleNamespace(train=dataset, eval=SimpleNamespace(path=dataset.path, sample_limit=2, model_dump=lambda **kwargs: {"path": str(dataset.path), "sample_limit": 2})),
        checkpoint=SimpleNamespace(save_final=False),
    )
    components = SimpleNamespace(
        model=object(), token_identity=SimpleNamespace(tokenizer_vocab_size=10),
        tokenizer=object(), base_model_path=tmp_path / "model",
        base_config_sha256="base", tokenizer_sha256="tokenizer",
        processor=SimpleNamespace(image_processor=object()),
    )
    train_shard = (SimpleNamespace(),)
    full_cache = (SimpleNamespace(), SimpleNamespace())
    cache = {
        "cache_dir": tmp_path / "cache", "micro_step_count": 2,
        "format_version": "v1", "fingerprint": "fp",
        "determinants_sha256": "digest",
    }
    eval_cache = {
        **cache,
        "cache_dir": tmp_path / "eval-cache",
        "fingerprint": "eval-fp",
        "determinants_sha256": "eval-digest",
    }
    schedule = SimpleNamespace(resolved_max_steps=1, runtime_batch=object())
    bindings: list[tuple[str, dict[str, object]]] = []
    writer = SimpleNamespace(
        bind_schedule=lambda **kwargs: None,
        bind_materialization=lambda split, **kwargs: bindings.append((split, kwargs)),
        finalize=lambda **kwargs: None,
    )
    loaded_full: list[tuple[Path, str]] = []
    train_verification_levels: list[str] = []

    monkeypatch.setattr(pipeline, "seed_training_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "load_qwen_components", lambda *args, **kwargs: components)
    monkeypatch.setattr(pipeline, "resolve_qwen_runtime_controls", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "load_default_adapter_source_gate_evidence", lambda root: object())
    monkeypatch.setattr(pipeline, "build_adapter_setup_plan", lambda *args, **kwargs: SimpleNamespace(mode="fresh"))
    monkeypatch.setattr(pipeline, "setup_dora_adapter", lambda model, plan: SimpleNamespace(model=model, receipt=SimpleNamespace(adapter_name="default")))
    monkeypatch.setattr(pipeline, "build_default_special_token_selection", lambda *args: object())
    monkeypatch.setattr(pipeline, "load_default_special_token_embedding_source_gate_evidence", lambda root: object())
    monkeypatch.setattr(pipeline, "install_special_token_embedding_deltas", lambda model, selection, source_gate: SimpleNamespace(model=model, receipt=object()))
    monkeypatch.setattr(pipeline, "enable_training_memory_savers", lambda model: None)
    monkeypatch.setattr(pipeline, "build_token_vocabulary_groups", lambda *args, **kwargs: object())

    def resolve_train_cache(*args: object, **kwargs: object) -> dict[str, object]:
        train_verification_levels.append(str(kwargs["verification_level"]))
        return cache

    monkeypatch.setattr(
        pipeline, "_resolve_or_build_train_pack_cache", resolve_train_cache
    )
    monkeypatch.setattr(pipeline, "resolve_planned_step_schedule", lambda *args, **kwargs: schedule)
    monkeypatch.setattr(pipeline, "load_rank_micro_steps_from_cache", lambda *args, **kwargs: train_shard)
    monkeypatch.setattr(pipeline, "_attach_image_processors_to_micro_steps", lambda steps, **kwargs: tuple(steps))
    monkeypatch.setattr(pipeline, "_apply_fa2_branch_proof_policy", lambda steps, config: tuple(steps))
    monkeypatch.setattr(pipeline.LossRunner, "from_config", lambda config: object())
    monkeypatch.setattr(pipeline, "build_optimizer_group_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "build_scheduler_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "build_optimizer_and_scheduler", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(pipeline, "build_trainable_surface_receipt", lambda *args, **kwargs: object())
    closed: list[str] = []
    runtime = SimpleNamespace(
        model=object(),
        accelerator=_Accelerator(),
        is_main_process=True,
        world_size=1,
        rank_report_gatherer=SimpleNamespace(
            close=lambda: closed.append("rank-report")
        ),
    )
    monkeypatch.setattr(pipeline, "TrainRuntime", lambda **kwargs: runtime)
    monkeypatch.setattr(
        pipeline, "_resolve_eval_pack_cache", lambda *args, **kwargs: eval_cache
    )

    def load_full(path: Path, *, expected_fingerprint: str) -> tuple[object, ...]:
        loaded_full.append((path, expected_fingerprint))
        return full_cache

    monkeypatch.setattr(pipeline, "load_all_micro_steps_from_cache", load_full)
    monkeypatch.setattr(pipeline, "CheckpointWriter", lambda run_dir: object())
    result = SimpleNamespace(
        completed_steps=1, consumed_micro_steps=1,
        scheduled_event_counts={}, latest_observation=None,
    )
    monkeypatch.setattr(pipeline, "SupervisedTrainer", lambda **kwargs: SimpleNamespace(run=lambda: result))

    pipeline._run_initialized_training(
        repo_root=tmp_path,
        resolved_config=SimpleNamespace(entry_config_path=tmp_path / "config.yaml", fingerprint="config-fp"),
        config=config,
        accelerator=_Accelerator(),
        run_directory=RunDirectory("run", tmp_path, tmp_path / "run", "created"),
        run_id="run",
        writer=writer,
        lifecycle={"completed_steps": 0, "consumed_packs": 0, "checkpoint_event_count": 0, "optimizer_update_status": None, "finite_status": None},
    )

    assert loaded_full == [(eval_cache["cache_dir"], "eval-fp")]
    assert train_verification_levels == ["manifest"]
    assert full_cache != train_shard
    assert [split for split, _ in bindings] == ["train", "eval"]
    assert bindings[0][1]["semantic_fingerprint"] == "fp"
    assert bindings[1][1]["semantic_fingerprint"] == "eval-fp"
    assert closed == ["rank-report"]
    assert set(bindings[1][1]) == {
        "cache_format_version", "semantic_fingerprint", "determinant_digest"
    }

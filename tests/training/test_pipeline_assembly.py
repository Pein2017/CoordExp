from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

import src.training.pipeline as pipeline_mod
from src.artifacts.manager import RunArtifactManager
from src.artifacts.metric_stream import MetricStreamEvent
from src.common.errors import RuntimeContractError
from src.config.loader import load_train_config
from src.config.models import RuntimeBatchResolution
from src.config.paths import RunDirectory
from src.qwen.images import QwenImageEncoding, QwenNoResizeImagePlan
from src.runtime import GateDecision
from src.training.pipeline import (
    BestEvalMetricStore,
    TrainingArtifactBridge,
    _apply_fa2_branch_proof_policy,
    _attach_image_processors_to_micro_steps,
    build_repeating_micro_step_stream,
    enable_training_memory_savers,
    _checkpoint_handler,
    _pack_plan_artifact,
    _resolve_rank_local_run_directory,
    run_training_pipeline,
)
from src.training.pack_cache import (
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    build_packing_cache_determinants,
    build_packing_cache_fingerprint,
    cache_dir_for_fingerprint,
    load_cache_manifest,
    write_micro_step_cache,
)
from src.training.schedule import ResolvedStepSchedule
from src.training.supervised_trainer import (
    PlannedStepResult,
    ScheduledTrainerEvent,
    SupervisedTrainingResult,
    SupervisedMicroStep,
    SupervisedTrainerEvent,
    StepScheduleEvent,
)


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


def test_run_training_pipeline_writes_core_artifacts_with_fake_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = yaml.safe_load(FIXTURE_CONFIG.read_text())
    _point_dataset_paths_at_fixture(payload)
    payload["run"]["artifact_root"] = str(tmp_path / "artifacts")
    payload["run"]["name"] = "fake-smoke"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    log: list[str] = []
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(tmp_path / "pack-cache"))

    _install_fake_training_pipeline_boundaries(monkeypatch, log)

    summary = run_training_pipeline(config_path)

    run_dir = Path(summary["run_dir"])
    assert summary["completed_steps"] == 5
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "configs" / "resolved.json").exists()
    assert (run_dir / "resolved_step_schedule.json").exists()
    manifest = __import__("json").loads((run_dir / "run_manifest.json").read_text())
    assert manifest["receipts"]["qwen"]["setup"] == "receipts/qwen/setup.json"
    assert manifest["receipts"]["packing"]["pack_plan"] == "receipts/packing/pack_plan.json"
    assert manifest["receipts"]["losses"]["loss_plan"] == "receipts/losses/loss_plan.json"
    assert manifest["receipts"]["optimizer"]["optimizer_groups"] == (
        "receipts/optimizer/optimizer_groups.json"
    )
    assert manifest["backend_status"]["deepspeed"] == [
        "schema_accepted",
        "conflict_validation_implemented",
    ]
    assert "systems_smoke_verified" not in manifest["backend_status"]["deepspeed"]
    assert manifest["status"] == "completed"
    assert manifest["reports"]["token_type_vocab"] == "reports/token_type_vocab.json"
    assert (run_dir / "reports" / "token_type_vocab.json").exists()
    qwen_setup = __import__("json").loads(
        (run_dir / "receipts" / "qwen" / "setup.json").read_text()
    )
    patch_receipt = qwen_setup["components"]["runtime_patches"][
        "qwen3_vl_patch_embed_linearization"
    ]
    assert patch_receipt["policy"] == "enabled"
    assert patch_receipt["applied"] is True
    assert patch_receipt["owner_path"] == "model.visual.patch_embed"
    loss_plan = __import__("json").loads(
        (run_dir / "receipts" / "losses" / "loss_plan.json").read_text()
    )
    assert loss_plan["objective_dtype"] == "float32_selected_logits"
    assert loss_plan["metric_definitions"]["top_level"] == ["acc_top1", "acc_top5"]
    assert loss_plan["vocabulary_groups"]["coordinate_count"] == 1000
    assert log == ["trainer.run"]


def test_run_training_pipeline_wires_accelerate_runtime_for_multirank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = yaml.safe_load(FIXTURE_CONFIG.read_text())
    _point_dataset_paths_at_fixture(payload)
    payload["run"]["artifact_root"] = str(tmp_path / "artifacts")
    payload["run"]["name"] = "fake-accelerate-smoke"
    payload["runtime"] = {
        "backend": "accelerate",
        "seed": 17,
        "accelerate": {
            "mixed_precision": "bf16",
            "gradient_accumulation_steps": None,
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    log: list[str] = []
    FakePipelineRuntime.last_kwargs = None

    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(tmp_path / "pack-cache"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        "src.training.pipeline.Accelerator",
        FakePipelineAccelerator,
        raising=False,
    )
    _install_fake_training_pipeline_boundaries(monkeypatch, log)
    monkeypatch.setattr(
        "src.training.pipeline._resolve_or_build_train_pack_cache",
        lambda config, components, vocab_groups, repo_root: _fake_train_cache(tmp_path),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.load_rank_micro_steps_from_cache",
        lambda cache_dir, schedule, rank, world_size: (_micro_step(0),),
        raising=False,
    )

    summary = run_training_pipeline(config_path)

    assert summary["completed_steps"] == 5
    assert FakePipelineRuntime.last_kwargs is not None
    assert isinstance(
        FakePipelineRuntime.last_kwargs["accelerator"],
        FakePipelineAccelerator,
    )
    assert callable(FakePipelineRuntime.last_kwargs["rank_report_gatherer"])
    accelerator = FakePipelineRuntime.last_kwargs["accelerator"]
    assert accelerator.mixed_precision == "bf16"
    assert accelerator.gradient_accumulation_steps == 1
    assert FakePipelineRuntime.last_kwargs["world_size"] == 2
    assert FakePipelineRuntime.last_kwargs["runtime_batch"].world_size == 2


def test_run_training_pipeline_wires_deepspeed_runtime_with_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    deepspeed_config_path = tmp_path / "ds_config.json"
    deepspeed_config_path.write_text(
        '{"zero_optimization": {"stage": 2}}\n',
        encoding="utf-8",
    )
    payload = yaml.safe_load(FIXTURE_CONFIG.read_text())
    _point_dataset_paths_at_fixture(payload)
    payload["run"]["artifact_root"] = str(tmp_path / "artifacts")
    payload["run"]["name"] = "fake-deepspeed-smoke"
    payload["training"]["effective_batch_size"] = 2
    payload["runtime"] = {
        "backend": "deepspeed",
        "seed": 17,
        "deepspeed": {
            "config_path": str(deepspeed_config_path),
            "gradient_accumulation_steps": None,
            "train_batch_size": None,
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    log: list[str] = []
    FakePipelineRuntime.last_kwargs = None
    FakePipelineDeepSpeedPlugin.last_kwargs = None

    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(tmp_path / "pack-cache"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        "src.training.pipeline.Accelerator",
        FakePipelineAccelerator,
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.DeepSpeedPlugin",
        FakePipelineDeepSpeedPlugin,
        raising=False,
    )
    _install_fake_training_pipeline_boundaries(monkeypatch, log)
    monkeypatch.setattr(
        "src.training.pipeline._resolve_or_build_train_pack_cache",
        lambda config, components, vocab_groups, repo_root: _fake_train_cache(tmp_path),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.load_rank_micro_steps_from_cache",
        lambda cache_dir, schedule, rank, world_size: (_micro_step(0),),
        raising=False,
    )

    summary = run_training_pipeline(config_path)

    assert summary["completed_steps"] == 5
    assert FakePipelineRuntime.last_kwargs is not None
    accelerator = FakePipelineRuntime.last_kwargs["accelerator"]
    assert isinstance(accelerator, FakePipelineAccelerator)
    assert FakePipelineRuntime.last_kwargs["runtime_config"].backend == "deepspeed"
    assert FakePipelineRuntime.last_kwargs["runtime_batch"].resolved_grad_accum_steps == 1
    assert FakePipelineRuntime.last_kwargs["runtime_batch"].world_size == 2
    assert FakePipelineDeepSpeedPlugin.last_kwargs == {
        "hf_ds_config": str(deepspeed_config_path),
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "zero_stage": None,
    }
    assert accelerator.deepspeed_plugin is not None


def test_run_training_pipeline_uses_rank_local_artifacts_for_nonzero_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = yaml.safe_load(FIXTURE_CONFIG.read_text())
    _point_dataset_paths_at_fixture(payload)
    payload["run"]["artifact_root"] = str(tmp_path / "artifacts")
    payload["run"]["name"] = "fake-rank-local-smoke"
    payload["runtime"] = {
        "backend": "accelerate",
        "seed": 17,
        "accelerate": {
            "mixed_precision": "bf16",
            "gradient_accumulation_steps": None,
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    log: list[str] = []

    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(tmp_path / "pack-cache"))
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        "src.training.pipeline.Accelerator",
        FakePipelineAccelerator,
        raising=False,
    )
    _install_fake_training_pipeline_boundaries(monkeypatch, log)
    monkeypatch.setattr(
        "src.training.pipeline._resolve_or_build_train_pack_cache",
        lambda config, components, vocab_groups, repo_root: _fake_train_cache(tmp_path),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.load_rank_micro_steps_from_cache",
        lambda cache_dir, schedule, rank, world_size: (_micro_step(0),),
        raising=False,
    )

    summary = run_training_pipeline(config_path)

    assert Path(summary["run_dir"]).name.endswith("-rank1")
    assert Path(summary["run_dir"]).exists()


def test_multi_rank_run_suffix_keeps_rank_dirs_under_one_launch_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = yaml.safe_load(FIXTURE_CONFIG.read_text())
    _point_dataset_paths_at_fixture(payload)
    payload["run"]["artifact_root"] = str(tmp_path / "artifacts")
    payload["run"]["name"] = "fake-prod"
    payload["run"]["collision_policy"] = "timestamp"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = __import__(
        "src.config.loader",
        fromlist=["load_train_config"],
    ).load_train_config(config_path).config
    (tmp_path / "artifacts" / "fake-prod").mkdir(parents=True)

    monkeypatch.setenv("COORDEXP_SWIFT_RUN_SUFFIX", "launch-123")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    (tmp_path / "artifacts" / "fake-prod-launch-123").mkdir()

    run_directory = _resolve_rank_local_run_directory(config, cwd=tmp_path)

    assert run_directory.run_dir.name == "fake-prod-launch-123-rank1"


def test_repeating_micro_step_stream_fills_planned_rank_local_window() -> None:
    schedule = _schedule(resolved_max_steps=3, grad_accum_steps=2)
    base_steps = (_micro_step(0), _micro_step(1))

    stream = build_repeating_micro_step_stream(base_steps, schedule)

    assert [next(stream).metadata["pack_id"] for _ in range(6)] == [0, 1, 0, 1, 0, 1]


def test_repeating_micro_step_stream_shards_effective_batch_across_ranks() -> None:
    schedule = _schedule(resolved_max_steps=2, grad_accum_steps=2, world_size=2)
    base_steps = tuple(_micro_step(index) for index in range(8))

    rank0 = build_repeating_micro_step_stream(
        base_steps,
        schedule,
        rank=0,
        world_size=2,
    )
    rank1 = build_repeating_micro_step_stream(
        base_steps,
        schedule,
        rank=1,
        world_size=2,
    )

    assert [next(rank0).metadata["pack_id"] for _ in range(4)] == [0, 2, 4, 6]
    assert [next(rank1).metadata["pack_id"] for _ in range(4)] == [1, 3, 5, 7]


def test_repeating_micro_step_stream_rejects_empty_base_stream() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_repeating_micro_step_stream((), _schedule(resolved_max_steps=1))


def test_cached_micro_steps_reattach_qwen_image_processor() -> None:
    image_processor = object()
    cached_encoding = QwenImageEncoding(
        plan=_image_plan(),
        pixel_values=None,
        image_grid_thw_tensor=None,
        image_processor=None,
    )
    cached_step = SupervisedMicroStep(
        pack="pack-0",
        encoded_examples=(FakeCachedEncodedExample("ex-0", cached_encoding),),
        position_inputs="positions",
        token_sequence="tokens",
        vocab_groups="vocab",
        metadata={"pack_id": 0},
    )

    (attached_step,) = _attach_image_processors_to_micro_steps(
        (cached_step,),
        image_processor=image_processor,
    )

    assert attached_step is not cached_step
    attached_example = attached_step.encoded_examples[0]
    assert attached_example.image_encoding.image_processor is image_processor
    assert cached_step.encoded_examples[0].image_encoding.image_processor is None


def test_fa2_branch_proof_policy_overrides_stale_cached_micro_steps() -> None:
    cached_steps = tuple(
        SupervisedMicroStep(
            pack=f"pack-{index}",
            encoded_examples=(f"example-{index}",),
            position_inputs=f"positions-{index}",
            token_sequence=f"tokens-{index}",
            vocab_groups=f"vocab-{index}",
            metadata={"pack_id": index},
            fa2_branch_evidence={"observed_branch": "stale"},
            capture_fa2_branch=True,
            require_fa2_branch_proof=True,
        )
        for index in range(3)
    )

    disabled_steps = _apply_fa2_branch_proof_policy(
        cached_steps,
        SimpleNamespace(model=SimpleNamespace(fa2_branch_proof="disabled")),
    )
    first_only_steps = _apply_fa2_branch_proof_policy(
        cached_steps,
        SimpleNamespace(model=SimpleNamespace(fa2_branch_proof="first_micro_step")),
    )

    assert [step.capture_fa2_branch for step in disabled_steps] == [False, False, False]
    assert [step.require_fa2_branch_proof for step in disabled_steps] == [
        False,
        False,
        False,
    ]
    assert [step.fa2_branch_evidence for step in disabled_steps] == [
        None,
        None,
        None,
    ]
    assert [step.fa2_branch_proof_policy for step in disabled_steps] == [
        "disabled",
        "disabled",
        "disabled",
    ]
    assert [step.capture_fa2_branch for step in first_only_steps] == [
        True,
        False,
        False,
    ]
    assert [step.require_fa2_branch_proof for step in first_only_steps] == [
        True,
        False,
        False,
    ]
    assert [step.fa2_branch_evidence for step in first_only_steps] == [
        None,
        None,
        None,
    ]
    assert [step.fa2_branch_proof_policy for step in first_only_steps] == [
        "first_micro_step",
        "first_micro_step",
        "first_micro_step",
    ]


def test_pack_plan_artifact_separates_global_cache_and_rank_local_counts() -> None:
    schedule = _schedule(resolved_max_steps=2, grad_accum_steps=2, world_size=2)
    rank_local_steps = (_micro_step(0), _micro_step(2), _micro_step(4), _micro_step(6))

    artifact = _pack_plan_artifact(
        rank_local_steps,
        schedule=schedule,
        cache={
            "cache_dir": Path("/tmp/coordexp-pack-cache/fingerprint"),
            "fingerprint": "fingerprint",
            "micro_step_count": 10,
            "chunk_count": 2,
            "chunk_size": 8,
            "status": "complete",
            "build_status": "hit",
            "manifest_path": Path("/tmp/coordexp-pack-cache/fingerprint/manifest.json"),
            "manifest_sha256": "a" * 64,
            "determinants_sha256": "b" * 64,
            "chunk_sha256s": ["c" * 64, "d" * 64],
        },
    )

    assert artifact["packs_per_epoch"] == 10
    assert artifact["rank_local_micro_step_count"] == 4
    assert artifact["cache"]["global_micro_step_count"] == 10
    assert artifact["cache"]["manifest_sha256"] == "a" * 64
    assert artifact["cache"]["chunk_sha256s"] == ["c" * 64, "d" * 64]


def test_pack_plan_artifact_records_cache_materialization() -> None:
    schedule = _schedule(resolved_max_steps=1)

    artifact = _pack_plan_artifact(
        (_micro_step(0),),
        schedule=schedule,
        cache={
            "cache_dir": Path("/tmp/coordexp-pack-cache/fingerprint"),
            "fingerprint": "fingerprint",
            "micro_step_count": 1,
            "chunk_count": 1,
            "chunk_size": 8,
            "status": "complete",
            "build_status": "built",
            "manifest_path": Path("/tmp/coordexp-pack-cache/fingerprint/manifest.json"),
            "manifest_sha256": "a" * 64,
            "determinants_sha256": "b" * 64,
            "chunk_sha256s": ["c" * 64],
            "materialization": {
                "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
                "workers": 16,
            },
        },
    )

    assert artifact["cache"]["materialization"] == {
        "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
        "workers": 16,
    }


def test_pack_cache_miss_receipt_records_materialization_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_with_temp_dataset(tmp_path)
    components = FakeComponents()
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(tmp_path / "pack-cache"))

    receipt = pipeline_mod._resolve_or_build_pack_cache(
        config,
        components,
        FakeVocabGroups(),
        repo_root=tmp_path,
        dataset=config.data.train,
        split="train",
        build_micro_steps=lambda materialization_workers: (_micro_step(materialization_workers),),
        materialization_workers=4,
    )
    manifest = load_cache_manifest(receipt["cache_dir"])

    assert receipt["build_status"] == "built"
    assert receipt["materialization"] == {
        "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
        "workers": 4,
    }
    assert manifest["materialization"] == receipt["materialization"]


def test_pack_cache_hit_does_not_rebuild_or_construct_process_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_with_temp_dataset(tmp_path)
    components = FakeComponents()
    cache_root = tmp_path / "pack-cache"
    fingerprint = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
    )
    determinants = build_packing_cache_determinants(
        config,
        components,
        dataset=config.data.train,
        split="train",
    )
    write_micro_step_cache(
        cache_dir_for_fingerprint(cache_root, fingerprint),
        (_micro_step(0),),
        fingerprint=fingerprint,
        determinants=determinants,
    )
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(cache_root))

    def _unexpected_pool(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("cache hit must not construct a process pool")

    monkeypatch.setattr(
        pipeline_mod.concurrent.futures,
        "ProcessPoolExecutor",
        _unexpected_pool,
        raising=False,
    )

    receipt = pipeline_mod._resolve_or_build_pack_cache(
        config,
        components,
        FakeVocabGroups(),
        repo_root=tmp_path,
        dataset=config.data.train,
        split="train",
        build_micro_steps=lambda materialization_workers: (_raise_rebuild(),),
    )

    assert receipt["build_status"] == "hit"
    assert receipt["micro_step_count"] == 1


def test_pack_cache_materialization_uses_default_pool_and_restores_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_examples = tuple(
        SimpleNamespace(example_id=f"ex-{index}")
        for index in range(4)
    )
    config = _minimal_pack_materialization_config()
    fake_context = object()
    _FakeProcessPoolExecutor.instances.clear()
    monkeypatch.setattr(
        pipeline_mod.multiprocessing,
        "get_all_start_methods",
        lambda: ["fork"],
    )
    monkeypatch.setattr(
        pipeline_mod.multiprocessing,
        "get_context",
        lambda method: fake_context,
    )
    monkeypatch.setattr(
        pipeline_mod.concurrent.futures,
        "ProcessPoolExecutor",
        _FakeProcessPoolExecutor,
    )
    monkeypatch.setattr(
        pipeline_mod.concurrent.futures,
        "as_completed",
        lambda futures: tuple(reversed(tuple(futures))),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "render_example",
        lambda raw_example, template, object_order_seed: f"rendered-{raw_example.example_id}",
    )
    monkeypatch.setattr(
        pipeline_mod,
        "encode_rendered_example",
        lambda raw_example, rendered, **kwargs: SimpleNamespace(
            example_id=raw_example.example_id,
            input_ids=(int(raw_example.example_id.removeprefix("ex-")),),
            rendered=rendered,
            supervised_token_spans=(),
        ),
    )

    encoded = pipeline_mod._build_encoded_examples_for_dataset(
        config,
        object(),
        raw_examples,
    )

    assert [example.example_id for example in encoded] == ["ex-0", "ex-1", "ex-2", "ex-3"]
    assert [example.rendered for example in encoded] == [
        "rendered-ex-0",
        "rendered-ex-1",
        "rendered-ex-2",
        "rendered-ex-3",
    ]
    assert len(_FakeProcessPoolExecutor.instances) == 1
    executor = _FakeProcessPoolExecutor.instances[0]
    assert executor.max_workers == 16
    assert executor.mp_context is fake_context
    assert executor.submitted_indices == [0, 1, 2, 3]
    assert pipeline_mod._PACK_CACHE_WORKER_CONTEXT is None


def test_pack_cache_worker_override_preserves_encoded_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_examples = tuple(
        SimpleNamespace(example_id=f"ex-{index}")
        for index in range(4)
    )
    config = _minimal_pack_materialization_config()
    monkeypatch.setattr(
        pipeline_mod,
        "render_example",
        lambda raw_example, template, object_order_seed: f"rendered-{raw_example.example_id}",
    )
    monkeypatch.setattr(
        pipeline_mod,
        "encode_rendered_example",
        lambda raw_example, rendered, **kwargs: SimpleNamespace(
            example_id=raw_example.example_id,
            input_ids=(int(raw_example.example_id.removeprefix("ex-")),),
            rendered=rendered,
            supervised_token_spans=(),
        ),
    )
    serial = pipeline_mod._build_encoded_examples_for_dataset(
        config,
        object(),
        raw_examples,
        materialization_workers=1,
    )
    _FakeProcessPoolExecutor.instances.clear()
    monkeypatch.setattr(
        pipeline_mod.multiprocessing,
        "get_all_start_methods",
        lambda: ["fork"],
    )
    monkeypatch.setattr(
        pipeline_mod.concurrent.futures,
        "ProcessPoolExecutor",
        _FakeProcessPoolExecutor,
    )
    monkeypatch.setattr(
        pipeline_mod.concurrent.futures,
        "as_completed",
        lambda futures: tuple(reversed(tuple(futures))),
    )

    parallel = pipeline_mod._build_encoded_examples_for_dataset(
        config,
        object(),
        raw_examples,
        materialization_workers=3,
    )

    assert [example.example_id for example in parallel] == [
        example.example_id for example in serial
    ]
    assert [example.rendered for example in parallel] == [
        example.rendered for example in serial
    ]
    serial_packs = pipeline_mod.plan_packed_sequences(serial, global_max_length=2)
    parallel_packs = pipeline_mod.plan_packed_sequences(parallel, global_max_length=2)
    assert [
        (pack.pack_index, tuple(segment.example_id for segment in pack.segments))
        for pack in parallel_packs
    ] == [
        (pack.pack_index, tuple(segment.example_id for segment in pack.segments))
        for pack in serial_packs
    ]


def test_pack_cache_workers_require_fork_when_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_mod.multiprocessing,
        "get_all_start_methods",
        lambda: ["spawn"],
    )

    with pytest.raises(RuntimeContractError, match="training.pack_cache_workers_unavailable"):
        pipeline_mod._build_encoded_examples_for_dataset(
            _minimal_pack_materialization_config(),
            object(),
            (SimpleNamespace(example_id="ex-0"),),
            materialization_workers=2,
        )


def test_training_artifact_bridge_writes_train_metrics_and_forward_receipt(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    bridge = TrainingArtifactBridge(manager, rank=0, world_size=1)
    step_result = PlannedStepResult(
        planned_step_id=1,
        micro_step_count=2,
        loss_bundle_artifact={
            "metrics": {
                "loss/total": 1.25,
                "loss/base_ce": 1.0,
                "loss/token_type_gate": 0.25,
                "acc_top1": 0.5,
                "acc_top5": 1.0,
            },
            "finite_status": "finite",
        },
        pre_backward_decision=_gate(1),
        post_backward_decision=_gate(1),
        qwen_forward_receipts=(
            {"pack_index": 0, "segment_count": 2},
            {"pack_index": 0, "segment_count": 2},
        ),
        optimizer_update_status="applied",
        finite_status="finite",
    )

    bridge(
        SupervisedTrainerEvent(
            event_type="planned_step.completed",
            planned_step_id=1,
            payload=step_result.to_artifact_dict(),
        )
    )

    train_records = [
        __import__("json").loads(line)
        for line in (manager.run_dir / "metrics" / "train.jsonl").read_text().splitlines()
    ]
    assert {(record["split"], record["name"]) for record in train_records} >= {
        ("train", "loss/total"),
        ("train", "acc_top1"),
        ("train", "acc_top5"),
    }
    assert all(record["optimizer_update_status"] == "applied" for record in train_records)
    qwen_receipt = __import__("json").loads(
        (manager.run_dir / "receipts" / "qwen" / "forward_step_1.json").read_text()
    )
    assert qwen_receipt["qwen_forward_receipts"][0]["segment_count"] == 2


def test_training_artifact_bridge_progress_stays_outside_metric_stream(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    bridge = TrainingArtifactBridge(manager, rank=1, world_size=2)

    bridge(
        SupervisedTrainerEvent(
            event_type="micro_step.forward",
            planned_step_id=3,
            payload={
                "local_micro_step_index": 4,
                "sync_gradients": False,
                "timings_ns": {"qwen_forward_total_ns": 333},
                "receipt": {
                    "pack_index": 9,
                    "pack_length": 12000,
                    "segment_count": 11,
                    "pixel_values_shape": [4096, 1536],
                    "output_logits_shape": [1, 256, 151936],
                    "placeholder_token_count": 4096,
                    "expected_visual_token_count": 4096,
                    "timings_ns": {
                        "model_forward_ns": 111,
                        "total_build_inputs_ns": 222,
                    },
                    "fa2_varlen": {
                        "max_length_q": 1100,
                        "max_length_k": 1100,
                        "segment_boundaries": [0, 100, 12000],
                    },
                },
            },
        )
    )
    bridge(
        SupervisedTrainerEvent(
            event_type="micro_step.pre_backward_gate",
            planned_step_id=3,
            payload={
                "local_micro_step_index": 4,
                "sync_gradients": False,
                "stage": "pre_backward_scalar",
                "optimizer_update_status": "pending_backward",
                "finite_status": "finite",
            },
        )
    )

    progress_path = manager.run_dir / "diagnostics" / "progress.rank-1.jsonl"
    records = [
        __import__("json").loads(line)
        for line in progress_path.read_text().splitlines()
    ]
    assert records == [
        {
            "event_type": "micro_step.forward",
            "local_micro_step_index": 4,
            "expected_visual_token_count": 4096,
            "fa2_max_length_k": 1100,
            "fa2_max_length_q": 1100,
            "fa2_segment_boundaries": [0, 100, 12000],
            "monotonic_ns": records[0]["monotonic_ns"],
            "output_logits_shape": [1, 256, 151936],
            "pack_index": 9,
            "pack_length": 12000,
            "pixel_values_shape": [4096, 1536],
            "placeholder_token_count": 4096,
            "planned_step_id": 3,
            "rank": 1,
            "segment_count": 11,
            "sync_gradients": False,
            "timings_ns": {
                "model_forward_ns": 111,
                "qwen_forward_total_ns": 333,
                "total_build_inputs_ns": 222,
            },
            "world_size": 2,
        },
        {
            "event_type": "micro_step.pre_backward_gate",
            "finite_status": "finite",
            "local_micro_step_index": 4,
            "monotonic_ns": records[1]["monotonic_ns"],
            "optimizer_update_status": "pending_backward",
            "planned_step_id": 3,
            "rank": 1,
            "stage": "pre_backward_scalar",
            "sync_gradients": False,
            "world_size": 2,
        },
    ]
    assert not (manager.run_dir / "metrics" / "train.jsonl").exists()
    manifest = __import__("json").loads((manager.run_dir / "run_manifest.json").read_text())
    assert manifest["metrics"]["streams"] == {}


def test_enable_training_memory_savers_records_gradient_checkpointing_and_cache_disable() -> None:
    model = FakeMemorySaverModel()

    receipt = enable_training_memory_savers(model)

    assert model.training is True
    assert model.gradient_checkpointing_enabled is True
    assert model.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert model.input_require_grads_enabled is True
    assert model.config.use_cache is False
    assert receipt["train_mode_enabled"] is True
    assert receipt["model_training"] is True
    assert receipt["gradient_checkpointing_enabled"] is True
    assert receipt["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert receipt["input_require_grads_enabled"] is True
    assert receipt["use_cache_disabled"] == ["model.config"]


def test_enable_training_memory_savers_disables_nested_qwen_text_cache() -> None:
    model = FakeNestedQwenMemorySaverModel()

    receipt = enable_training_memory_savers(model)

    assert model.config.text_config.use_cache is False
    assert receipt["use_cache_disabled"] == ["model.config.text_config"]


def test_checkpoint_handler_uses_same_step_eval_acc_top1_for_best_selection(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    writer = __import__("src.artifacts", fromlist=["CheckpointWriter"]).CheckpointWriter(manager)
    best_metrics = BestEvalMetricStore()
    best_metrics.record(
        MetricStreamEvent(
            event_type="metric",
            planned_step_id=2,
            split="eval.forward",
            name="acc_top1",
            value=0.42,
            trigger_reasons=("explicit_step",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        )
    )
    handler = _checkpoint_handler(
        writer,
        model=None,
        adapter_receipt=None,
        special_token_result=None,
        trainable_surface={"trainable_towers": []},
        processor_identity={"processor": "fake"},
        resolved_config_fingerprint="fingerprint",
        schedule=_schedule(resolved_max_steps=2),
        base_model_path=Path("/tmp/fake-qwen"),
        best_eval_metrics=best_metrics,
    )

    handler(
        ScheduledTrainerEvent(
            scheduled_event=StepScheduleEvent(
                planned_step_id=2,
                event="checkpoint",
                trigger_reasons=("every_fraction:1.0",),
                source_config_path=None,
                deduped_from=(),
                required=False,
            ),
            step_result=PlannedStepResult(
                planned_step_id=2,
                micro_step_count=1,
                loss_bundle_artifact={"metrics": {"acc_top1": 0.1}},
                pre_backward_decision=_gate(2),
                post_backward_decision=_gate(2),
                qwen_forward_receipts=(),
                optimizer_update_status="applied",
                finite_status="finite",
            ),
        )
    )

    best_alias = __import__("json").loads(
        (manager.run_dir / "checkpoints" / "best_acc_top1.json").read_text()
    )
    assert best_alias["checkpoint_id"] == "step-2"
    assert best_alias["metric"]["value"] == 0.42


def test_checkpoint_handler_unwraps_accelerate_model_and_saves_adapter_only(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    writer = __import__("src.artifacts", fromlist=["CheckpointWriter"]).CheckpointWriter(manager)
    unwrapped = FakeAdapterOnlyModel()
    wrapped = FakeWrappedModel(unwrapped)
    runtime = FakeCheckpointRuntime(
        is_main_process=True,
        accelerator=FakeUnwrapAccelerator(),
    )
    handler = _checkpoint_handler(
        writer,
        model=wrapped,
        runtime=runtime,
        adapter_receipt=FakeReceipt({"adapter_type": "dora"}),
        special_token_result=None,
        trainable_surface={"trainable_towers": ["adapter.language"]},
        processor_identity={"processor": "fake"},
        resolved_config_fingerprint="fingerprint",
        schedule=_schedule(resolved_max_steps=1),
        base_model_path=Path("/tmp/fake-qwen"),
    )

    handler(
        ScheduledTrainerEvent(
            scheduled_event=StepScheduleEvent(
                planned_step_id=1,
                event="checkpoint",
                trigger_reasons=("checkpoint.final",),
                source_config_path=None,
                deduped_from=(),
                required=False,
            ),
            step_result=PlannedStepResult(
                planned_step_id=1,
                micro_step_count=1,
                loss_bundle_artifact={"metrics": {"loss/total": 1.0}},
                pre_backward_decision=_gate(1),
                post_backward_decision=_gate(1),
                qwen_forward_receipts=(),
                optimizer_update_status="applied",
                finite_status="finite",
            ),
        )
    )

    adapter_dir = manager.run_dir / "checkpoints" / "step-1" / "adapter"
    assert (adapter_dir / "adapter_config.json").exists()
    assert (adapter_dir / "adapter_model.safetensors").exists()
    assert not (adapter_dir / "config.json").exists()
    assert not (adapter_dir / "model.safetensors").exists()
    assert unwrapped.save_pretrained_calls == 1


def test_checkpoint_handler_skips_non_main_process_checkpoint_side_effects(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    writer = __import__("src.artifacts", fromlist=["CheckpointWriter"]).CheckpointWriter(manager)
    runtime = FakeCheckpointRuntime(is_main_process=False)
    handler = _checkpoint_handler(
        writer,
        model=FakeAdapterOnlyModel(),
        runtime=runtime,
        adapter_receipt=FakeReceipt({"adapter_type": "dora"}),
        special_token_result=None,
        trainable_surface={"trainable_towers": ["adapter.language"]},
        processor_identity={"processor": "fake"},
        resolved_config_fingerprint="fingerprint",
        schedule=_schedule(resolved_max_steps=1),
        base_model_path=Path("/tmp/fake-qwen"),
    )

    handler(
        ScheduledTrainerEvent(
            scheduled_event=StepScheduleEvent(
                planned_step_id=1,
                event="checkpoint",
                trigger_reasons=("checkpoint.final",),
                source_config_path=None,
                deduped_from=(),
                required=False,
            ),
            step_result=PlannedStepResult(
                planned_step_id=1,
                micro_step_count=1,
                loss_bundle_artifact={"metrics": {"loss/total": 1.0}},
                pre_backward_decision=_gate(1),
                post_backward_decision=_gate(1),
                qwen_forward_receipts=(),
                optimizer_update_status="applied",
                finite_status="finite",
            ),
        )
    )

    assert not (manager.run_dir / "checkpoints").exists()


def test_best_eval_metric_store_only_returns_same_step_acc_top1() -> None:
    store = BestEvalMetricStore()
    store.record(
        MetricStreamEvent(
            event_type="metric",
            planned_step_id=4,
            split="eval.forward",
            name="acc_top5",
            value=0.9,
            trigger_reasons=("explicit_step",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        )
    )
    store.record(
        MetricStreamEvent(
            event_type="metric",
            planned_step_id=4,
            split="eval.forward",
            name="acc_top1",
            value=0.5,
            trigger_reasons=("explicit_step",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        )
    )

    assert store.best_for_step(4).value == 0.5
    assert store.best_for_step(5) is None


def _micro_step(index: int) -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=f"pack-{index}",
        encoded_examples=(f"example-{index}",),
        position_inputs=f"positions-{index}",
        token_sequence=f"tokens-{index}",
        vocab_groups=f"vocab-{index}",
        metadata={"pack_id": index},
    )


def _image_plan() -> QwenNoResizeImagePlan:
    return QwenNoResizeImagePlan(
        example_id="ex-0",
        image_path=Path("/tmp/image.jpg"),
        width=64,
        height=64,
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        required_spatial_factor=32,
        raw_pixels=4096,
        raw_patch_rows=16,
        expected_pixel_values_width=1536,
        image_grid_thw=(1, 4, 4),
        merged_visual_tokens=4,
        max_raw_pixels=1_000_000,
        max_merged_visual_tokens=4096,
    )


def _schedule(
    resolved_max_steps: int,
    grad_accum_steps: int = 1,
    *,
    world_size: int = 1,
) -> ResolvedStepSchedule:
    return ResolvedStepSchedule(
        resolved_max_steps=resolved_max_steps,
        packs_per_epoch=1,
        requested_pack_presentations=resolved_max_steps * grad_accum_steps * world_size,
        actual_pack_presentations=resolved_max_steps * grad_accum_steps * world_size,
        tail_fill_pack_count=0,
        runtime_batch=RuntimeBatchResolution(
            world_size=world_size,
            effective_batch_size=grad_accum_steps * world_size,
            resolved_grad_accum_steps=grad_accum_steps,
        ),
        events={"checkpoint": (), "eval.forward": (), "training.logging": (), "final": ()},
    )


def _manager(tmp_path: Path) -> RunArtifactManager:
    return RunArtifactManager.initialize(
        run_directory=RunDirectory(
            run_name="pipeline-test",
            artifact_root=tmp_path,
            run_dir=tmp_path / "run",
            collision_policy="fail",
        ),
        run_id="run-pipeline-test",
        created_at="2026-06-30T00:00:00+00:00",
        runtime_identity={"backend": "single"},
        backend_status={"deepspeed": ["schema_accepted", "conflict_validation_implemented"]},
    )


def _point_dataset_paths_at_fixture(payload: dict[str, Any]) -> None:
    dataset_path = str((FIXTURE_CONFIG.parent / "examples.jsonl").resolve())
    payload["data"]["train"]["path"] = dataset_path
    if payload["data"].get("eval") is not None:
        payload["data"]["eval"]["path"] = dataset_path


def _config_with_temp_dataset(tmp_path: Path) -> Any:
    dataset = tmp_path / "train.coord.jsonl"
    dataset.write_text('{"example_id":"ex-0"}\n', encoding="utf-8")
    config = load_train_config(FIXTURE_CONFIG).config
    return config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(
                        update={"path": str(dataset)}
                    )
                }
            )
        }
    )


def _minimal_pack_materialization_config() -> Any:
    return SimpleNamespace(
        template=SimpleNamespace(object_ordering="geo_sorted"),
        model=SimpleNamespace(processor=SimpleNamespace()),
        packing=SimpleNamespace(global_max_length=128),
        runtime=SimpleNamespace(seed=17),
    )


def _raise_rebuild() -> Any:
    raise AssertionError("cache hit must not rebuild micro-steps")


def _fake_train_cache(tmp_path: Path) -> dict[str, Any]:
    return {
        "cache_dir": tmp_path / "pack-cache" / "fake",
        "fingerprint": "fake",
        "micro_step_count": 1,
        "chunk_count": 1,
        "chunk_size": 1,
        "status": "complete",
        "build_status": "hit",
        "manifest_path": tmp_path / "pack-cache" / "fake" / "manifest.json",
        "manifest_sha256": "a" * 64,
        "determinants_sha256": "b" * 64,
        "chunk_sha256s": ["c" * 64],
        "determinants": {"purpose": "unit-test"},
        "materialization": {
            "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
            "workers": 16,
        },
    }


def _gate(planned_step_id: int) -> GateDecision:
    return GateDecision(
        planned_step_id=planned_step_id,
        stage="post_backward_gradient",
        world_size=1,
        ranks=(0,),
        all_ranks_safe=True,
        optimizer_update_status="ready_to_step",
        finite_status="finite",
        should_call_backward=False,
        should_call_optimizer_step=True,
        should_clear_gradients=False,
        reason_codes=(),
        rank_diagnostics=(),
        diagnostics={},
    )


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


class FakeTokenIdentity:
    tokenizer_vocab_size = 150000

    def to_artifact_dict(self) -> dict[str, Any]:
        return {"tokenizer_vocab_size": self.tokenizer_vocab_size}


@dataclass(frozen=True)
class FakeIdentity:
    payload: dict[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class FakeCachedEncodedExample:
    example_id: str
    image_encoding: Any


class FakeComponents:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.base_model_path = Path("/tmp/fake-qwen")
        self.token_identity = FakeTokenIdentity()
        self.tokenizer = object()
        self.processor_identity = FakeIdentity({"processor": "fake"})
        self.processor = FakePipelineProcessor()

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "base_model_path": str(self.base_model_path),
            "processor": self.processor_identity.to_artifact_dict(),
            "tokens": self.token_identity.to_artifact_dict(),
            "load_model": True,
            "runtime_patches": {
                "qwen3_vl_patch_embed_linearization": {
                    "policy": "enabled",
                    "applied": True,
                    "owner_path": "model.visual.patch_embed",
                }
            },
        }


class FakePipelineProcessor:
    image_processor = object()


class FakePlan:
    def to_artifact_dict(self) -> dict[str, Any]:
        return {"adapter_type": "dora"}


class FakeReceipt:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_artifact_dict(self) -> dict[str, Any]:
        return dict(self.payload)


class FakeAdapterResult:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.receipt = FakeReceipt({"adapter_type": "dora"})


class FakeSpecialTokenResult:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.receipt = FakeReceipt({"enabled": True, "selected_token_count": 1004})


class FakeVocabGroups:
    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": 150000,
            "desc_text_count": 148000,
            "schema": [10, 11, 12, 13],
            "coordinate_count": 1000,
            "coordinate_min": 140000,
            "coordinate_max": 140999,
            "eos": [2],
            "blocked_count": 996,
        }


class FakeOptimizerPlan:
    def to_artifact_dict(self) -> dict[str, Any]:
        return {"groups": [{"group_name": "adapter.language"}]}


class FakeOptimizer:
    pass


class FakeScheduler:
    pass


class FakeTrainableSurface:
    def to_artifact_dict(self) -> dict[str, Any]:
        return {"trainable_towers": ["adapter.language"]}


class FakeMemorySaverConfig:
    use_cache = True


class FakeMemorySaverModel:
    def __init__(self) -> None:
        self.config = FakeMemorySaverConfig()
        self.training = False
        self.gradient_checkpointing_enabled = False
        self.gradient_checkpointing_kwargs = None
        self.input_require_grads_enabled = False

    def train(self) -> None:
        self.training = True

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs: dict[str, Any]) -> None:
        self.gradient_checkpointing_enabled = True
        self.gradient_checkpointing_kwargs = dict(gradient_checkpointing_kwargs)

    def enable_input_require_grads(self) -> None:
        self.input_require_grads_enabled = True


class FakeNestedQwenTextConfig:
    use_cache = True


class FakeNestedQwenConfig:
    def __init__(self) -> None:
        self.text_config = FakeNestedQwenTextConfig()


class FakeNestedQwenMemorySaverModel(FakeMemorySaverModel):
    def __init__(self) -> None:
        super().__init__()
        self.config = FakeNestedQwenConfig()


def _install_fake_training_pipeline_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    log: list[str],
) -> None:
    monkeypatch.setattr(
        "src.training.pipeline.load_qwen_components",
        lambda config, load_model: FakeComponents(),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.load_default_adapter_source_gate_evidence",
        lambda repo_root: object(),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.build_adapter_setup_plan",
        lambda adapter_config, evidence, base_model_path: FakePlan(),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.setup_dora_adapter",
        lambda model, plan: FakeAdapterResult(model=model),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.build_default_special_token_selection",
        lambda config, token_identity: object(),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.load_default_special_token_embedding_source_gate_evidence",
        lambda repo_root: object(),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.install_special_token_embedding_deltas",
        lambda model, selection, source_gate: FakeSpecialTokenResult(model=model),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.build_base_micro_steps",
        lambda config, components, vocab_groups, materialization_workers=None: (_micro_step(0),),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.build_token_vocabulary_groups",
        lambda token_identity, tokenizer: FakeVocabGroups(),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.build_optimizer_group_plan",
        lambda model, optimizer_config, adapter_receipt, special_token_receipt: FakeOptimizerPlan(),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.build_optimizer_and_scheduler",
        lambda optimizer_config, group_plan, total_training_steps: (FakeOptimizer(), FakeScheduler()),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.build_trainable_surface_receipt",
        lambda model, adapter_receipt, special_token_receipt, optimizer_group_plan: FakeTrainableSurface(),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.TrainRuntime",
        lambda **kwargs: FakePipelineRuntime(**kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        "src.training.pipeline.SupervisedTrainer",
        lambda **kwargs: FakePipelineTrainer(log=log, **kwargs),
        raising=False,
    )


class FakePipelineAccelerator:
    def __init__(
        self,
        *,
        mixed_precision: str | None = None,
        gradient_accumulation_steps: int | None = None,
        deepspeed_plugin: Any | None = None,
    ) -> None:
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.deepspeed_plugin = deepspeed_plugin


class FakePipelineDeepSpeedPlugin:
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        FakePipelineDeepSpeedPlugin.last_kwargs = dict(kwargs)


class _FakeFuture:
    def __init__(self, result: Any) -> None:
        self._result = result

    def result(self) -> Any:
        return self._result


class _FakeProcessPoolExecutor:
    instances: list["_FakeProcessPoolExecutor"] = []

    def __init__(self, *, max_workers: int, mp_context: Any) -> None:
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.submitted_indices: list[int] = []
        _FakeProcessPoolExecutor.instances.append(self)

    def __enter__(self) -> "_FakeProcessPoolExecutor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def submit(self, fn: Any, index: int) -> _FakeFuture:
        self.submitted_indices.append(index)
        return _FakeFuture(fn(index))


class FakeAdapterOnlyModel:
    def __init__(self) -> None:
        self.save_pretrained_calls = 0

    def save_pretrained(self, output_dir: str | Path) -> None:
        self.save_pretrained_calls += 1
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(
            '{"peft_type": "LORA", "use_dora": true}\n',
            encoding="utf-8",
        )
        (path / "adapter_model.safetensors").write_bytes(b"adapter")


class FakeWrappedModel:
    def __init__(self, unwrapped: FakeAdapterOnlyModel) -> None:
        self.unwrapped = unwrapped


class FakeUnwrapAccelerator:
    def unwrap_model(self, model: Any) -> Any:
        return model.unwrapped


class FakeCheckpointRuntime:
    def __init__(
        self,
        *,
        is_main_process: bool,
        accelerator: Any | None = None,
    ) -> None:
        self.is_main_process = is_main_process
        self.accelerator = accelerator


class FakePipelineRuntime:
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        FakePipelineRuntime.last_kwargs = dict(kwargs)
        self.model = kwargs["model"]
        self.rank = kwargs["rank"]
        self.world_size = kwargs["world_size"]
        self.setup_receipt = FakeReceipt(
            {
                "backend": kwargs["runtime_config"].backend,
                "world_size": self.world_size,
                "runtime_batch": kwargs["runtime_batch"].to_artifact_dict(),
                "backend_status": {
                    "single": ["active"],
                    "accelerate": [],
                    "deepspeed": [],
                },
            }
        )


class FakePipelineTrainer:
    last_pack_ids: list[int] | None = None

    def __init__(self, *, log: list[str], event_sink: Any, **kwargs: Any) -> None:
        del event_sink
        self.log = log
        FakePipelineTrainer.last_pack_ids = [
            step.metadata["pack_id"] for step in kwargs["pack_stream"]
        ]

    def run(self) -> SupervisedTrainingResult:
        self.log.append("trainer.run")
        return SupervisedTrainingResult(
            completed_steps=5,
            consumed_micro_steps=10,
            step_results=(),
            scheduled_event_counts={
                "checkpoint": 0,
                "eval.forward": 0,
                "final": 0,
                "training.logging": 0,
            },
        )

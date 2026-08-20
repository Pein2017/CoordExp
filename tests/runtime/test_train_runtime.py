from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import math
import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.common.errors import RuntimeContractError
from src.config.models import RuntimeBatchResolution, RuntimeConfig
from src.runtime.metrics import (
    REDUCER_BOOL_ALL as BOOL_ALL,
)
from src.runtime.metrics import (
    REDUCER_IDENTICAL as IDENTICAL,
)
from src.runtime.metrics import (
    REDUCER_MAX as MAX,
)
from src.runtime.metrics import (
    REDUCER_SUM as SUM,
)
from src.runtime.metrics import (
    AccuracySufficientStats,
    MetricBatch,
    RatioSample,
    ScalarSample,
    checked_accuracy_stats,
    loss_telemetry_batch,
)
from src.runtime.seeding import seed_training_runtime
from src.runtime.train_runtime import TrainRuntime, validate_accelerator_runtime
from src.training import SupervisedMicroStep


def _config(*, seed: int = 17, determinism_mode: str = "legacy") -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {"seed": seed, "determinism": {"mode": determinism_mode}}
    )


def _batch(*, world_size: int = 1, accumulation: int = 1) -> RuntimeBatchResolution:
    return RuntimeBatchResolution(
        world_size=world_size,
        effective_batch_size=world_size * accumulation,
        resolved_grad_accum_steps=accumulation,
    )


def _runtime(
    *,
    accelerator: FakeAccelerator | None = None,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    world_size: int = 1,
    accumulation: int = 1,
    gatherer: Any | None = None,
    max_grad_norm: float | None = None,
    expected_mixed_precision: str = "bf16",
    determinism_mode: str = "legacy",
) -> TrainRuntime:
    accelerator = accelerator or FakeAccelerator(num_processes=world_size)
    model = model or torch.nn.Linear(1, 1)
    return TrainRuntime(
        runtime_config=_config(determinism_mode=determinism_mode),
        runtime_batch=_batch(world_size=world_size, accumulation=accumulation),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_mixed_precision=expected_mixed_precision,
        max_grad_norm=max_grad_norm,
        accelerator=accelerator,
        rank_report_gatherer=gatherer,
    )


def test_seed_training_runtime_delegates_to_transformers_set_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        "src.runtime.seeding.set_seed",
        lambda seed, deterministic=False: calls.append((seed, deterministic)),
    )
    receipt = seed_training_runtime(17, phase="pipeline_assembly").to_artifact_dict()
    assert calls == [(17, False)]
    assert receipt["phase"] == "pipeline_assembly"


def test_strict_determinism_requires_launcher_environment_and_sets_torch_cudnn_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = {
        key: os.environ[key]
        for key in ("FLASH_ATTENTION_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG")
    }
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    deterministic_calls: list[tuple[bool, bool]] = []
    seed_calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only=False: deterministic_calls.append(
            (enabled, warn_only)
        ),
    )
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
    )
    monkeypatch.setattr(
        "src.runtime.seeding.set_seed",
        lambda seed, deterministic=False: seed_calls.append((seed, deterministic)),
    )
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)

    receipt = seed_training_runtime(
        17,
        determinism_mode="strict_cuda_replay_v1",
        phase="pipeline_entry",
    ).to_artifact_dict()

    assert os.environ["FLASH_ATTENTION_DETERMINISTIC"] == "1"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert deterministic_calls == [(True, False)]
    assert seed_calls == [(17, False)]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert {
        key: os.environ[key]
        for key in ("FLASH_ATTENTION_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG")
    } == before
    assert receipt["mode"] == "strict_cuda_replay_v1"
    assert receipt["deterministic_algorithms"] == {
        "enabled": True,
        "managed": True,
        "warn_only": False,
    }
    assert receipt["cudnn"] == {
        "benchmark": False,
        "deterministic": True,
        "managed": True,
    }
    assert receipt["required_environment"] == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "FLASH_ATTENTION_DETERMINISTIC": "1",
    }
    assert receipt["observed_environment"] == receipt["required_environment"]


def test_initial_strict_determinism_attests_applied_torch_policy_before_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda _enabled, *, warn_only=False: None,
    )
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    monkeypatch.setattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
    )
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    seed_calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        "src.runtime.seeding.set_seed",
        lambda seed, deterministic=False: seed_calls.append((seed, deterministic)),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        seed_training_runtime(
            17,
            determinism_mode="strict_cuda_replay_v1",
            phase="pipeline_entry",
        )

    assert exc_info.value.code == "runtime.determinism_policy_drift"
    assert seed_calls == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FLASH_ATTENTION_DETERMINISTIC", None),
        ("CUBLAS_WORKSPACE_CONFIG", None),
        ("FLASH_ATTENTION_DETERMINISTIC", "0"),
        ("CUBLAS_WORKSPACE_CONFIG", ":16:8"),
    ],
)
def test_strict_determinism_rejects_any_environment_conflict_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    if value is not None:
        monkeypatch.setenv(name, value)
    other = (
        "CUBLAS_WORKSPACE_CONFIG"
        if name == "FLASH_ATTENTION_DETERMINISTIC"
        else "FLASH_ATTENTION_DETERMINISTIC"
    )
    monkeypatch.setenv(
        other,
        ":4096:8" if other == "CUBLAS_WORKSPACE_CONFIG" else "1",
    )
    before = {
        key: os.environ.get(key)
        for key in ("FLASH_ATTENTION_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG")
    }
    deterministic_calls: list[object] = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda *args, **kwargs: deterministic_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        seed_training_runtime(
            17,
            determinism_mode="strict_cuda_replay_v1",
            phase="pipeline_entry",
        )

    assert exc_info.value.code == "runtime.determinism_environment_conflict"
    assert {
        key: os.environ.get(key)
        for key in ("FLASH_ATTENTION_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG")
    } == before
    assert deterministic_calls == []


def test_initial_strict_determinism_rejects_initialized_cuda_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(RuntimeContractError) as exc_info:
        seed_training_runtime(
            17,
            determinism_mode="strict_cuda_replay_v1",
            phase="pipeline_entry",
        )

    assert exc_info.value.code == "runtime.determinism_cuda_already_initialized"
    assert os.environ["FLASH_ATTENTION_DETERMINISTIC"] == "1"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_runtime_strict_reapplication_only_accepts_exact_existing_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
    )
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    seed_calls: list[tuple[int, bool]] = []
    policy_mutations: list[object] = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda *args, **kwargs: policy_mutations.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "src.runtime.seeding.set_seed",
        lambda seed, deterministic=False: seed_calls.append((seed, deterministic)),
    )

    receipt = seed_training_runtime(
        17,
        determinism_mode="strict_cuda_replay_v1",
        phase="runtime_setup_reapplied",
    ).to_artifact_dict()

    assert policy_mutations == []
    assert seed_calls == [(17, False)]
    assert receipt["cuda_initialized"] is True


def test_runtime_strict_reapplication_rejects_drift_before_seed_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", lambda: True
    )
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    seed_calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        "src.runtime.seeding.set_seed",
        lambda seed, deterministic=False: seed_calls.append((seed, deterministic)),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        seed_training_runtime(
            17,
            determinism_mode="strict_cuda_replay_v1",
            phase="runtime_setup_reapplied",
        )

    assert exc_info.value.code == "runtime.determinism_policy_drift"
    assert seed_calls == []


def test_strict_deterministic_algorithm_enablement_errors_propagate_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    unsupported = RuntimeError(
        "nondeterministic operation has no deterministic implementation"
    )
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda *args, **kwargs: (_ for _ in ()).throw(unsupported),
    )

    with pytest.raises(RuntimeError) as exc_info:
        seed_training_runtime(
            17,
            determinism_mode="strict_cuda_replay_v1",
            phase="pipeline_entry",
        )

    assert exc_info.value is unsupported


def test_runtime_identity_comes_from_constructed_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool, str]] = []
    monkeypatch.setattr(
        "src.runtime.train_runtime.seed_training_runtime",
        lambda seed, determinism_mode="legacy", phase="": calls.append(
            (seed, determinism_mode, phase)
        ),
    )
    accelerator = FakeAccelerator(
        process_index=1,
        num_processes=2,
        device="cpu",
        is_main_process=False,
        distributed_type="MULTI_GPU",
    )
    runtime = _runtime(
        accelerator=accelerator,
        world_size=2,
        gatherer=FakeReportGatherer(),
    )
    assert (runtime.rank, runtime.world_size, runtime.device) == (
        1,
        2,
        torch.device("cpu"),
    )
    assert runtime.is_main_process is False
    assert calls == [(17, "legacy", "runtime_setup_reapplied")]


@pytest.mark.parametrize("distributed_type", ["FSDP", "DEEPSPEED", "TP", "MULTI_CPU"])
def test_unsupported_distributed_type_is_rejected_before_prepare(
    distributed_type: str,
) -> None:
    accelerator = FakeAccelerator(distributed_type=distributed_type)
    with pytest.raises(RuntimeContractError) as exc_info:
        _runtime(accelerator=accelerator)
    assert exc_info.value.code == "runtime.distributed_type_unsupported"
    assert exc_info.value.context["distributed_type"] == distributed_type
    assert accelerator.prepare_calls == 0


def test_accelerator_accumulation_must_be_neutral() -> None:
    accelerator = FakeAccelerator(gradient_accumulation_steps=2)
    with pytest.raises(RuntimeContractError) as exc_info:
        _runtime(accelerator=accelerator, accumulation=2)
    assert exc_info.value.code == "runtime.accelerator_accumulation_non_neutral"
    assert accelerator.prepare_calls == 0


def test_accelerator_mixed_precision_matches_resolved_training_precision() -> None:
    accelerator = FakeAccelerator(mixed_precision="bf16")
    validate_accelerator_runtime(
        accelerator,
        expected_mixed_precision="BF16",
    )
    _runtime(accelerator=accelerator, expected_mixed_precision="bf16")
    assert accelerator.prepare_calls == 1


@pytest.mark.parametrize("observed_precision", ["fp16", "no", None])
def test_accelerator_mixed_precision_mismatch_is_rejected_before_prepare(
    observed_precision: str | None,
) -> None:
    accelerator = FakeAccelerator(mixed_precision=observed_precision)
    with pytest.raises(RuntimeContractError) as exc_info:
        _runtime(accelerator=accelerator, expected_mixed_precision="bf16")
    assert exc_info.value.code == "runtime.mixed_precision_mismatch"
    assert exc_info.value.context == {
        "expected_mixed_precision": "bf16",
        "observed_mixed_precision": "no"
        if observed_precision is None
        else observed_precision,
    }
    assert accelerator.prepare_calls == 0


def test_prepare_owns_model_and_optimizer_but_not_scheduler() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    accelerator = FakeAccelerator()
    runtime = _runtime(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert accelerator.prepared_objects == (model, optimizer)
    assert runtime.scheduler is scheduler


def test_runtime_preserves_accumulation_backward_clip_optimizer_scheduler_order() -> (
    None
):
    model = torch.nn.Linear(1, 1)
    optimizer = RecordingOptimizer(model.parameters())
    scheduler = RecordingScheduler(optimizer)
    accelerator = FakeAccelerator(events=optimizer.events)
    runtime = _runtime(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulation=2,
        max_grad_norm=0.5,
    )
    scheduler.events = optimizer.events
    with runtime.accumulation_context(sync_gradients=False):
        runtime.backward(runtime.model(torch.tensor([[1.0]])).sum(), planned_step_id=1)
    with runtime.accumulation_context(sync_gradients=True):
        runtime.backward(runtime.model(torch.tensor([[2.0]])).sum(), planned_step_id=1)
    decision = runtime.post_backward(planned_step_id=1)
    runtime.execute_optimizer_boundary(decision, planned_step_id=1)
    runtime.scheduler_step(planned_step_id=1)
    runtime.zero_gradients(planned_step_id=1)
    assert decision.should_call_optimizer_step
    assert accelerator.no_sync_models == [runtime.model]
    assert optimizer.events == [
        "backward",
        "backward",
        "clip",
        "optimizer_step",
        "scheduler_step",
        "zero_grad",
    ]


def test_all_rank_scalar_and_gradient_decisions_are_preserved() -> None:
    gatherer = FakeReportGatherer(peer_scalar_total=float("nan"))
    runtime = _runtime(world_size=2, gatherer=gatherer)
    pre = runtime.pre_backward(
        FakeLossBundle(total_loss=torch.tensor(1.0)), planned_step_id=6
    )
    assert not pre.should_call_backward
    assert pre.reason_codes == ("rank1:non_finite_scalar",)

    gatherer.peer_scalar_total = 1.0
    gatherer.peer_backend_overflow = True
    for parameter in runtime.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    post = runtime.post_backward(planned_step_id=7)
    assert not post.should_call_optimizer_step
    assert post.reason_codes == ("rank1:backend_overflow",)


def test_all_rank_loss_denominators_are_preserved() -> None:
    runtime = _runtime(world_size=2, gatherer=DenominatorGatherer())
    gathered = runtime.gather_loss_denominators(
        {"base_ce": {"eligible_segment_count": 1}}, planned_step_id=9
    )
    assert gathered[0]["base_ce"]["eligible_segment_count"] == 1
    assert gathered[1]["base_ce"]["eligible_segment_count"] == 3


def test_multirank_runtime_requires_report_gatherer() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _runtime(world_size=2)
    assert exc_info.value.code == "runtime.report_gather_unavailable"


def _sample(
    name: str,
    reducer: str,
    value: float | None,
    *,
    required: bool = True,
    integral: bool = False,
) -> ScalarSample:
    return ScalarSample(
        name=name, reducer=reducer, value=value, required=required, integral=integral
    )


def _metric_batch(
    *samples: ScalarSample | RatioSample,
    split: str = "train",
    planned_step_id: int = 4,
    accuracy: AccuracySufficientStats | None = None,
    reduction_mode: str | None = None,
) -> MetricBatch:
    return MetricBatch(
        planned_step_id=planned_step_id,
        split=split,
        samples=samples,
        accuracy=accuracy,
        reduction_mode=reduction_mode,
    )


def test_move_and_single_rank_metrics() -> None:
    runtime = _runtime()
    moved = runtime.move_micro_step(
        SupervisedMicroStep(
            pack="pack",
            encoded_examples=(),
            position_inputs="positions",
            token_sequence="tokens",
            vocab_groups="vocab",
        ),
        planned_step_id=1,
        local_micro_step_index=0,
    )
    assert moved.forward_device == torch.device("cpu")
    gathered = runtime.gather_metrics(
        _metric_batch(_sample("loss/total", SUM, 1.5), planned_step_id=1)
    )
    assert gathered["reduction"] == "single_rank"
    assert gathered["metrics"] == {"loss/total": 1.5}
    assert gathered["per_rank_metrics"] == {"0": {"loss/total": 1.5}}


def test_gather_metrics_rejects_an_untyped_metric_mapping() -> None:
    # Wave 2: an untyped mapping cannot declare reduction semantics, so the
    # collective boundary refuses it instead of guessing from key names.
    runtime = _runtime()
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.gather_metrics({"loss/total": 1.0})  # type: ignore[arg-type]
    assert exc_info.value.code == "runtime.metric_batch_untyped"


def test_multirank_declared_reducers_all_ride_one_gather() -> None:
    gatherer = TypedMetricGatherer(
        peer_values={
            "count/packs": 3.0,
            "step_duration_seconds": 0.9,
            "lr/group_0": 1e-5,
            "finite/total_loss": 0.0,
            "loss/base_ce/token_weighted_diag": (4.0 * 3, 3.0),
        }
    )
    runtime = _runtime(world_size=2, gatherer=gatherer)
    reduced = runtime.gather_metrics(
        _metric_batch(
            _sample("count/packs", SUM, 1.0, integral=True),
            _sample("step_duration_seconds", MAX, 0.4),
            _sample("lr/group_0", IDENTICAL, 1e-5),
            _sample("finite/total_loss", BOOL_ALL, 1.0),
            RatioSample("loss/base_ce/token_weighted_diag", 1.0 * 1, 1.0),
        )
    )
    assert reduced["reduction"] == "all_rank_mixed"
    assert reduced["metrics"] == {
        "count/packs": 4.0,
        "finite/total_loss": 0.0,
        "loss/base_ce/token_weighted_diag": 13.0 / 4.0,
        "lr/group_0": 1e-5,
        "step_duration_seconds": 0.9,
    }
    # The reduction rides the existing gather payload: exactly one collective.
    assert gatherer.calls == 1


def test_multirank_accuracy_reduction_sums_integer_stats_for_unequal_atom_counts() -> (
    None
):
    # Rank 0 (local): 1/3 correct; rank 1 (peer): 1/6 correct. Unequal
    # per-rank atom counts (3 vs 6): the pooled ratio MUST be the sum of
    # integer correct counts over the sum of integer atom counts, not the
    # plain mean of the two ratios.
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(
            peer_accuracy={
                "reducer": SUM,
                "top1_correct": 1,
                "top5_correct": 2,
                "atom_count": 6,
                "top1_metric_name": "acc_top1",
                "top5_metric_name": "acc_top5",
            }
        ),
    )
    result = runtime.gather_metrics(
        _metric_batch(
            accuracy=AccuracySufficientStats(
                reducer=SUM, top1_correct=1, top5_correct=2, atom_count=3
            )
        )
    )
    reduced = result["metrics"]
    assert reduced["acc_top1"] == pytest.approx((1 + 1) / (3 + 6))
    assert reduced["acc_top5"] == pytest.approx((2 + 2) / (3 + 6))
    assert result["accuracy_stats"] == {
        "top1_correct": 2,
        "top5_correct": 4,
        "atom_count": 9,
    }

    # Anti-pattern (rejected): the plain mean of the two rank-local ratios.
    plain_mean_acc_top1 = ((1 / 3) + (1 / 6)) / 2
    assert reduced["acc_top1"] != pytest.approx(plain_mean_acc_top1)


def test_eight_rank_accuracy_reduction_returns_authoritative_global_integer_stats() -> (
    None
):
    class EightRankAccuracyGatherer:
        def __call__(self, local_report: dict[str, Any]) -> tuple[dict[str, Any], ...]:
            reports = [local_report]
            for rank in range(1, 8):
                atom_count = rank + 2
                reports.append(
                    {
                        **deepcopy(local_report),
                        "rank": rank,
                        "accuracy_stats": {
                            "reducer": SUM,
                            "top1_correct": rank,
                            "top5_correct": rank + 1,
                            "atom_count": atom_count,
                            "top1_metric_name": "acc_top1",
                            "top5_metric_name": "acc_top5",
                        },
                    }
                )
            return tuple(reports)

    runtime = _runtime(world_size=8, gatherer=EightRankAccuracyGatherer())
    result = runtime.gather_metrics(
        _metric_batch(
            accuracy=AccuracySufficientStats(
                reducer=SUM, top1_correct=0, top5_correct=1, atom_count=2
            ),
            planned_step_id=3,
        )
    )

    assert result["accuracy_stats"] == {
        "top1_correct": 28,
        "top5_correct": 36,
        "atom_count": 44,
    }
    assert result["metrics"]["acc_top1"] == pytest.approx(28 / 44)
    assert result["metrics"]["acc_top5"] == pytest.approx(36 / 44)


def test_multirank_accuracy_reduction_rejects_rounded_ratio_reconstruction() -> None:
    # Rank-local exact stats: rank 0 has 333 atoms with 111 correct (ratio
    # exactly 1/3); rank 1 has 3 atoms with 1 correct (ratio exactly 1/3).
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(
            peer_accuracy={
                "reducer": SUM,
                "top1_correct": 1,
                "top5_correct": 1,
                "atom_count": 3,
                "top1_metric_name": "acc_top1",
                "top5_metric_name": "acc_top5",
            }
        ),
    )
    result = runtime.gather_metrics(
        _metric_batch(
            accuracy=AccuracySufficientStats(
                reducer=SUM, top1_correct=111, top5_correct=111, atom_count=333
            ),
            planned_step_id=9,
        )
    )
    reduced_acc_top1 = result["metrics"]["acc_top1"]
    assert reduced_acc_top1 == pytest.approx(1.0 / 3.0)

    # Anti-pattern (rejected): report each rank's ratio at 2-decimal
    # precision and reconstruct an integer correct-count from it before
    # pooling. Rounding rank 0's ratio (0.3333...) to "0.33" and multiplying
    # back by its 333 atoms reconstructs 110 correct, not the true 111.
    reconstructed_rank0 = round(round(111 / 333, 2) * 333)
    reconstructed_rank1 = round(round(1 / 3, 2) * 3)
    assert reconstructed_rank0 == 110
    anti_pattern_value = (reconstructed_rank0 + reconstructed_rank1) / (333 + 3)
    assert anti_pattern_value != pytest.approx(reduced_acc_top1)


def test_single_rank_accuracy_stats_degenerate_to_rank_local_value() -> None:
    runtime = _runtime()
    result = runtime.gather_metrics(
        _metric_batch(
            accuracy=AccuracySufficientStats(
                reducer=SUM, top1_correct=5, top5_correct=7, atom_count=10
            ),
            planned_step_id=1,
        )
    )
    assert result["metrics"] == {"acc_top1": 0.5, "acc_top5": 0.7}
    assert result["accuracy_stats"] == {
        "top1_correct": 5,
        "top5_correct": 7,
        "atom_count": 10,
    }
    assert result["reduction"] == "single_rank"


def test_producer_accuracy_ratio_mismatch_with_integer_stats_fails_closed() -> None:
    # The producer-side check that a published accuracy ratio agrees with the
    # exact integer statistics it claims to summarize, before anything is
    # gathered (the reduced ratio is always derived from the integers).
    with pytest.raises(RuntimeContractError) as exc_info:
        checked_accuracy_stats(
            {"top1_correct": 2, "top5_correct": 3, "atom_count": 4},
            reducer=SUM,
            reported_metrics={"acc_top1": 0.0, "acc_top5": 0.0},
        )
    assert exc_info.value.code == "runtime.accuracy_metric_stats_mismatch"


def test_producer_accuracy_metric_without_stats_fails_closed() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        checked_accuracy_stats(None, reducer=SUM, reported_metrics={"acc_top1": 0.5})
    assert exc_info.value.code == "runtime.accuracy_stats_missing"


def test_multirank_accuracy_metric_without_peer_accuracy_stats_fails_closed() -> None:
    # Local rank supplies accuracy_stats but a peer omits them entirely: this
    # MUST fail closed, never silently drop to a plain mean.
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(peer_accuracy=None),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.gather_metrics(
            _metric_batch(
                accuracy=AccuracySufficientStats(
                    reducer=SUM, top1_correct=5, top5_correct=5, atom_count=10
                ),
                split="eval",
            )
        )
    assert exc_info.value.code == "runtime.accuracy_stats_missing"


def test_multirank_metrics_without_accuracy_keys_never_require_accuracy_stats() -> None:
    # A caller genuinely gathering a metric set with no accuracy statistics
    # (e.g. timing-only) is unaffected by the accuracy requirement.
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(peer_values={"loss/total": 3.0}),
    )
    result = runtime.gather_metrics(
        _metric_batch(_sample("loss/total", SUM, 1.0), split="eval")
    )
    assert result["metrics"] == {"loss/total": 4.0}
    assert "accuracy_stats" not in result


def test_multirank_timing_fields_reduce_to_all_rank_maximum_not_mean() -> None:
    # The slowest rank owns the distributed critical path: unequal per-rank
    # timings MUST reduce to the maximum, never the mean.
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(
            peer_values={
                "eval_duration_seconds": 1.4,
                "step_duration_seconds": 0.9,
                "input_build_seconds": 0.05,
                "input_wait_seconds": 0.0,
            }
        ),
    )
    result = runtime.gather_metrics(
        _metric_batch(
            _sample("eval_duration_seconds", MAX, 0.8),
            _sample("step_duration_seconds", MAX, 0.4),
            _sample("input_build_seconds", MAX, 0.2),
            _sample("input_wait_seconds", MAX, 0.1),
        )
    )
    reduced = result["metrics"]
    assert reduced == {
        "eval_duration_seconds": 1.4,
        "input_build_seconds": 0.2,
        "input_wait_seconds": 0.1,
        "step_duration_seconds": 0.9,
    }
    assert reduced["step_duration_seconds"] != pytest.approx((0.4 + 0.9) / 2)


def test_multirank_timing_receipt_preserves_each_rank_value() -> None:
    # The per-rank projection stays available to bounded lifecycle resource
    # accounting even though it is no longer serialized into a normal row.
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(
            peer_values={"step_duration_seconds": 0.9, "input_wait_seconds": 0.3}
        ),
    )
    result = runtime.gather_metrics(
        _metric_batch(
            _sample("step_duration_seconds", MAX, 0.4),
            _sample("input_wait_seconds", MAX, 0.1),
        )
    )
    assert result["per_rank_metrics"] == {
        "0": {"input_wait_seconds": 0.1, "step_duration_seconds": 0.4},
        "1": {"input_wait_seconds": 0.3, "step_duration_seconds": 0.9},
    }


def test_multirank_resource_high_water_fields_reduce_to_max_and_preserve_ranks() -> (
    None
):
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(
            peer_values={
                "resource/cpu_max_rss_bytes": 2048.0,
                "resource/gpu_max_memory_allocated_bytes": 128.0,
            }
        ),
    )
    result = runtime.gather_metrics(
        _metric_batch(
            _sample("resource/cpu_max_rss_bytes", MAX, 1024.0),
            _sample("resource/gpu_max_memory_allocated_bytes", MAX, 512.0),
        )
    )
    assert result["metrics"] == {
        "resource/cpu_max_rss_bytes": 2048.0,
        "resource/gpu_max_memory_allocated_bytes": 512.0,
    }
    assert result["per_rank_metrics"]["1"]["resource/cpu_max_rss_bytes"] == 2048.0


# ---------------------------------------------------------------------------
# eval.forward reduction declarations (task 2.4)
# ---------------------------------------------------------------------------


def _eval_loss_artifact() -> dict[str, Any]:
    return {
        "terms": [
            {"name": "base_ce", "selected_count": 2},
            {"name": "coordinate_gaussian", "selected_count": 0},
        ]
    }


def _eval_batch(
    metrics: dict[str, Any],
    *,
    sharded: bool,
    extra_samples: tuple[ScalarSample, ...] = (),
) -> MetricBatch:
    return loss_telemetry_batch(
        planned_step_id=4,
        split="eval",
        loss_metrics=metrics,
        loss_artifact=_eval_loss_artifact(),
        partial_rank_contributions=sharded,
        extra_samples=extra_samples,
        reduction_mode="disjoint_shard" if sharded else None,
    )


def test_eval_disjoint_shard_sum_keys_sum_rank_local_counts_not_mean() -> None:
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(
            peer_values={
                "count/packs": 1.0,
                "count/examples": 2.0,
                "example_count": 2.0,
                "pack_count": 1.0,
                "loss/total": 3.0,
            }
        ),
    )
    reduced = runtime.gather_metrics(
        _eval_batch(
            {
                "count/packs": 2.0,
                "count/examples": 3.0,
                "loss/total": 1.0,
            },
            sharded=True,
            extra_samples=(
                _sample("example_count", SUM, 3.0, integral=True),
                _sample("pack_count", SUM, 2.0, integral=True),
            ),
        )
    )["metrics"]
    assert reduced["count/packs"] == 3.0
    assert reduced["count/examples"] == 5.0
    assert reduced["example_count"] == 5.0
    assert reduced["pack_count"] == 3.0
    assert reduced["loss/total"] == 4.0


def test_eval_disjoint_shard_identical_keys_require_exact_cross_rank_agreement() -> (
    None
):
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(peer_values={}),
    )
    reduced = runtime.gather_metrics(
        _eval_batch(
            {
                "count/supervised_atoms": 7.0,
                "count/eligible_segments": 3.0,
                "count/skipped_segments": 1.0,
                "loss/base_ce/segment_count": 3.0,
            },
            sharded=True,
        )
    )["metrics"]
    assert reduced == {
        "count/eligible_segments": 3.0,
        "count/skipped_segments": 1.0,
        "count/supervised_atoms": 7.0,
        "loss/base_ce/segment_count": 3.0,
    }


def test_eval_disjoint_shard_identical_key_mismatch_fails_closed() -> None:
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(peer_values={"count/supervised_atoms": 9.0}),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.gather_metrics(
            _eval_batch({"count/supervised_atoms": 7.0}, sharded=True)
        )
    assert exc_info.value.code == "runtime.metric_identical_mismatch"


def test_eval_disjoint_shard_sum_key_rejects_non_integer_count() -> None:
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(peer_values={"count/packs": 1.5}),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.gather_metrics(_eval_batch({"count/packs": 2.0}, sharded=True))
    assert exc_info.value.code == "runtime.metric_count_value_invalid"


def test_eval_disjoint_shard_token_weighted_diag_ratio_propagates_nonfinite() -> None:
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(
            peer_values={"loss/base_ce/token_weighted_diag": (float("nan"), 2.0)}
        ),
    )
    reduced = runtime.gather_metrics(
        _eval_batch({"loss/base_ce/token_weighted_diag": 1.0}, sharded=True)
    )["metrics"]
    assert math.isnan(reduced["loss/base_ce/token_weighted_diag"])


def test_eval_disjoint_shard_reduction_mode_mismatch_across_ranks_fails_closed() -> (
    None
):
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(peer_payload={"reduction_mode": None}),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.gather_metrics(_eval_batch({"loss/total": 1.0}, sharded=True))
    assert exc_info.value.code == "runtime.metric_gather_reduction_mode"


def test_eval_finite_keys_reduce_by_conjunction_not_mean() -> None:
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(
            peer_values={"finite/total_loss": 0.0, "finite/base_ce": 1.0}
        ),
    )
    reduced = runtime.gather_metrics(
        _eval_batch(
            {"finite/total_loss": 1.0, "finite/base_ce": 1.0},
            sharded=True,
        )
    )["metrics"]
    assert reduced == {"finite/base_ce": 1.0, "finite/total_loss": 0.0}
    assert reduced["finite/total_loss"] != pytest.approx(0.5)


def test_eval_finite_key_rejects_malformed_value() -> None:
    runtime = _runtime(world_size=2, gatherer=TypedMetricGatherer(peer_values={}))
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.gather_metrics(_eval_batch({"finite/total_loss": 0.5}, sharded=True))
    assert exc_info.value.code == "runtime.metric_bool_all_value_invalid"


def test_replicated_eval_declares_identical_not_summed_objective_values() -> None:
    runtime = _runtime(world_size=2, gatherer=TypedMetricGatherer(peer_values={}))
    reduced = runtime.gather_metrics(
        _eval_batch(
            {
                "loss/total": 2.0,
                "loss/base_ce/raw": 2.0,
                "count/packs": 3.0,
                "count/supervised_atoms": 7.0,
                "loss/base_ce/token_weighted_diag": 1.25,
                "finite/total_loss": 1.0,
            },
            sharded=False,
        )
    )["metrics"]
    # Every rank already holds the whole eval set: the global value is that
    # identical value, neither summed nor averaged.
    assert reduced == {
        "count/packs": 3.0,
        "count/supervised_atoms": 7.0,
        "finite/total_loss": 1.0,
        "loss/base_ce/raw": 2.0,
        "loss/base_ce/token_weighted_diag": 1.25,
        "loss/total": 2.0,
    }


def test_replicated_eval_objective_divergence_fails_closed() -> None:
    runtime = _runtime(
        world_size=2, gatherer=TypedMetricGatherer(peer_values={"loss/total": 2.0})
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.gather_metrics(_eval_batch({"loss/total": 1.0}, sharded=False))
    assert exc_info.value.code == "runtime.metric_identical_mismatch"


def test_planned_step_telemetry_with_no_declared_reducer_is_rejected() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _eval_batch({"unclassified/scientific_metric": 1.0}, sharded=True)
    assert exc_info.value.code == "runtime.metric_reducer_undeclared"


def test_eval_reduction_consensus_passes_when_ranks_agree() -> None:
    runtime = _runtime(
        world_size=2,
        gatherer=ConsensusReportGatherer(
            peer_mutation={"reduction_mode": "disjoint_shard", "pack_count": 5},
        ),
    )
    # No exception: the real bounded gather path is exercised end to end.
    runtime.validate_eval_reduction_consensus(
        reduction_mode="disjoint_shard", pack_count=5
    )


def test_eval_reduction_consensus_allows_matching_none_pack_count() -> None:
    # The eval-cache-is-None fallback branch: no canonical cross-rank
    # pack_count exists, so both ranks pass None and that must be accepted.
    runtime = _runtime(
        world_size=2,
        gatherer=ConsensusReportGatherer(
            peer_mutation={"reduction_mode": "replicated", "pack_count": None},
        ),
    )
    runtime.validate_eval_reduction_consensus(
        reduction_mode="replicated", pack_count=None
    )


def test_eval_reduction_consensus_fails_closed_on_mode_mismatch() -> None:
    # Opus HOLD P2-B: this must be caught HERE, before either rank could
    # otherwise reach the mode-dependent denominator-gather collective that
    # only a disjoint_shard rank calls -- a divergence there would hang a
    # replicated peer forever instead of failing closed.
    runtime = _runtime(
        world_size=2,
        gatherer=ConsensusReportGatherer(
            peer_mutation={"reduction_mode": "replicated", "pack_count": 5},
        ),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.validate_eval_reduction_consensus(
            reduction_mode="disjoint_shard", pack_count=5
        )
    assert exc_info.value.code == "runtime.eval_reduction_consensus_mode_mismatch"


def test_eval_reduction_consensus_fails_closed_on_pack_count_mismatch() -> None:
    runtime = _runtime(
        world_size=2,
        gatherer=ConsensusReportGatherer(
            peer_mutation={"reduction_mode": "disjoint_shard", "pack_count": 9},
        ),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.validate_eval_reduction_consensus(
            reduction_mode="disjoint_shard", pack_count=5
        )
    assert exc_info.value.code == "runtime.eval_reduction_consensus_pack_count_mismatch"


def test_eval_reduction_consensus_is_noop_at_world_size_one() -> None:
    runtime = _runtime()  # world_size=1, no gatherer configured
    # Must not raise and must not require a rank_report_gatherer.
    runtime.validate_eval_reduction_consensus(
        reduction_mode="disjoint_shard", pack_count=1
    )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"planned_step_id": 5}, "runtime.metric_gather_step"),
        ({"split": "train"}, "runtime.metric_gather_split"),
    ],
)
def test_multirank_metric_identity_mismatch_is_rejected(
    mutation: dict[str, Any],
    expected_code: str,
) -> None:
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(peer_payload=mutation),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.gather_metrics(_metric_batch(_sample("loss/total", SUM, 1.0), split="eval"))
    assert exc_info.value.code == expected_code


def test_multirank_metric_key_mismatch_is_rejected() -> None:
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(peer_rename={"loss/total": "other"}),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.gather_metrics(_metric_batch(_sample("loss/total", SUM, 1.0), split="eval"))
    assert exc_info.value.code == "runtime.metric_gather_keys"


def test_multirank_metric_report_count_mismatch_is_rejected() -> None:
    runtime = _runtime(
        world_size=2,
        gatherer=lambda local_report: (local_report,),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.gather_metrics(_metric_batch(_sample("loss/total", SUM, 1.0), split="eval"))
    assert exc_info.value.code == "runtime.metric_gather_count"


@pytest.mark.parametrize("peer_value", [float("nan"), float("inf"), float("-inf")])
def test_multirank_metric_reduction_preserves_nonfinite_values(
    peer_value: float,
) -> None:
    runtime = _runtime(
        world_size=2,
        gatherer=TypedMetricGatherer(peer_values={"loss/total": peer_value}),
    )
    reduced = runtime.gather_metrics(
        _metric_batch(_sample("loss/total", SUM, 1.0), split="eval")
    )["metrics"]["loss/total"]
    if math.isnan(peer_value):
        assert math.isnan(reduced)
    else:
        assert reduced == peer_value



class FakeLossBundle:
    def __init__(self, *, total_loss: torch.Tensor) -> None:
        self.total_loss = total_loss
        self.terms = ()


class FakeReportGatherer:
    def __init__(
        self,
        *,
        peer_scalar_total: float = 1.0,
        peer_backend_overflow: bool = False,
    ) -> None:
        self.peer_scalar_total = peer_scalar_total
        self.peer_backend_overflow = peer_backend_overflow

    def __call__(self, local_report: Any) -> tuple[Any, Any]:
        from src.runtime import RankGradientFiniteReport, RankScalarFiniteReport

        if isinstance(local_report, RankScalarFiniteReport):
            peer = RankScalarFiniteReport.from_loss_bundle(
                FakeLossBundle(total_loss=torch.tensor(self.peer_scalar_total)),
                planned_step_id=local_report.planned_step_id,
                rank=1,
                world_size=2,
            )
            return local_report, peer
        if isinstance(local_report, RankGradientFiniteReport):
            peer = RankGradientFiniteReport(
                planned_step_id=local_report.planned_step_id,
                rank=1,
                world_size=2,
                gradients_finite=True,
                backend_overflow=self.peer_backend_overflow,
                grad_norm=1.0,
            )
            return local_report, peer
        raise AssertionError(type(local_report).__name__)


class DenominatorGatherer:
    def __call__(self, payload: dict[str, Any]) -> tuple[Any, Any]:
        peer = {
            **payload,
            "rank": 1,
            "denominators": {
                name: {**value, "eligible_segment_count": 3}
                for name, value in payload["denominators"].items()
            },
        }
        return payload, peer


class ConsensusReportGatherer:
    """Two-rank gatherer for the non-metric consensus payload."""

    def __init__(self, *, peer_mutation: Mapping[str, Any]) -> None:
        self.peer_mutation = dict(peer_mutation)

    def __call__(self, local_report: dict[str, Any]) -> tuple[Any, Any]:
        return local_report, {**deepcopy(local_report), "rank": 1, **self.peer_mutation}


class TypedMetricGatherer:
    """Two-rank typed-payload gatherer.

    Rank 1's payload is a deep copy of rank 0's with the requested per-sample
    values, renames, accuracy statistics, or identity fields applied, so a
    test states exactly one asymmetry at a time.
    """

    _UNSET = object()

    def __init__(
        self,
        *,
        peer_values: Mapping[str, Any] | None = None,
        peer_rename: Mapping[str, str] | None = None,
        peer_accuracy: Any = _UNSET,
        peer_payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.peer_values = dict(peer_values or {})
        self.peer_rename = dict(peer_rename or {})
        self.peer_accuracy = peer_accuracy
        self.peer_payload = dict(peer_payload or {})
        self.calls = 0

    def __call__(self, local_report: dict[str, Any]) -> tuple[Any, Any]:
        self.calls += 1
        peer = deepcopy(local_report)
        peer["rank"] = 1
        for sample in peer["samples"]:
            name = str(sample["name"])
            if name in self.peer_rename:
                sample["name"] = self.peer_rename[name]
            if name not in self.peer_values:
                continue
            value = self.peer_values[name]
            if sample["form"] == "ratio":
                sample["numerator"], sample["denominator"] = value
            else:
                sample["value"] = value
        if self.peer_accuracy is not TypedMetricGatherer._UNSET:
            peer["accuracy_stats"] = self.peer_accuracy
        peer.update(self.peer_payload)
        return local_report, peer


class FakeAccelerator:
    def __init__(
        self,
        *,
        process_index: int = 0,
        num_processes: int = 1,
        device: str = "cpu",
        is_main_process: bool = True,
        distributed_type: str = "NO",
        gradient_accumulation_steps: int = 1,
        mixed_precision: str | None = "bf16",
        events: list[str] | None = None,
    ) -> None:
        self.process_index = process_index
        self.num_processes = num_processes
        self.device = torch.device(device)
        self.is_main_process = is_main_process
        self.distributed_type = SimpleNamespace(name=distributed_type)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.events = events if events is not None else []
        self.prepare_calls = 0
        self.prepared_objects: tuple[Any, ...] = ()
        self.no_sync_models: list[Any] = []
        self.no_sync_depth = 0
        self.scaler = None

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        self.prepare_calls += 1
        self.prepared_objects = tuple(objects)
        return self.prepared_objects

    @contextmanager
    def no_sync(self, model: Any) -> Any:
        self.no_sync_models.append(model)
        self.no_sync_depth += 1
        try:
            yield
        finally:
            self.no_sync_depth -= 1

    def backward(self, loss: torch.Tensor) -> None:
        self.events.append("backward")
        loss.backward()

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> None:
        self.events.append("clip")
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class RecordingOptimizer(torch.optim.SGD):
    def __init__(self, parameters: Any) -> None:
        super().__init__(parameters, lr=0.1)
        self.events: list[str] = []

    def step(self, closure: Any | None = None) -> Any:
        self.events.append("optimizer_step")
        return super().step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.events.append("zero_grad")
        super().zero_grad(set_to_none=set_to_none)


class RecordingScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.events: list[str] = []
        self.last_epoch = 0

    def step(self) -> None:
        self.events.append("scheduler_step")
        self.last_epoch += 1

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

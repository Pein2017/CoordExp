import pytest


def test_stage2_rollout_correction_rollout_correction_ddp_phase_timeout_zero_rejected_under_ddp(monkeypatch):
    """`ddp_phase_timeout_s: 0` is invalid when running with DDP."""

    import torch.distributed as dist

    from src.trainers.rollout_correction.executors import RolloutCorrectionExecutorsMixin

    class DummyModel:
        def train(self):
            return self

    class DummyTrainer(RolloutCorrectionExecutorsMixin):
        def __init__(self):
            self.model = DummyModel()

        def _packing_enabled(self):
            return True

        def _packing_drop_last(self):
            return True

        def _packing_buffer_cap(self):
            return 1

        def _packing_length(self):
            return 16

        def _packing_min_fill_ratio(self):
            return 0.5

        def _rollout_backend(self):
            return "hf"

        def _vllm_mode(self):
            return ""

        def _stage2_rollout_correction_pipeline_enabled(self, *, backend: str, mode: str):
            return False

        def _rollout_decode_batch_size_per_rank(self):
            return 1

        def _ab_rollout_correction_get(self, key: str, default=None):
            if key == "ddp_phase_timeout_s":
                return 0
            return default

        def _prepare_rollout_correction_inputs(self, inputs, *, _segments_only: bool):
            assert _segments_only is True
            return [({}, {}, 1)], {}

        def _stage2_append_post_rollout_segments(self, *, channel: str, segments):
            return None

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    barrier_calls = {"n": 0}

    def _barrier(*_args, **_kwargs):
        barrier_calls["n"] += 1

    monkeypatch.setattr(dist, "barrier", _barrier)

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "dist.new_group/monitored_barrier should not be called when ddp_phase_timeout_s <= 0"
        )

    monkeypatch.setattr(dist, "new_group", _boom)
    monkeypatch.setattr(dist, "monitored_barrier", _boom)

    t = DummyTrainer()
    with pytest.raises(ValueError, match=r"ddp_phase_timeout_s must be > 0 under DDP"):
        t._stage2_rollout_correction_step_budgeted_train(
            t.model,
            raw_samples=[{"messages": []}],
            global_step=1,
        )
    assert barrier_calls["n"] == 0


def test_stage2_rollout_correction_nonpipeline_prepare_barrier_uses_rollout_wait_budget(monkeypatch):
    import contextlib

    import torch
    import torch.distributed as dist

    import src.trainers.rollout_correction.executors as executors_mod
    from src.trainers.rollout_correction.executors import RolloutCorrectionExecutorsMixin

    class DummyModel:
        def train(self):
            return self

    class DummyTrainer(RolloutCorrectionExecutorsMixin):
        def __init__(self):
            self.model = DummyModel()

        def _packing_enabled(self):
            return True

        def _packing_length(self):
            return 16

        def _packing_min_fill_ratio(self):
            return 0.5

        def _rollout_backend(self):
            return "vllm"

        def _vllm_mode(self):
            return "server"

        def _vllm_server_timeouts(self):
            return 240.0, 240.0

        def _ab_rollout_correction_get(self, key: str, default=None):
            return default

        def _stage2_rollout_correction_pipeline_enabled(self, *, backend: str, mode: str):
            return False

        def _rollout_decode_batch_size_per_rank(self):
            return 1

        def _stage2_stage_wallclock_ctx(self, _stage: str):
            return contextlib.nullcontext()

        def _stage2_reset_train_monitor_dump(self, *, global_step: int):
            return None

        def _stage2_post_rollout_buffer(self, *, channel: str):
            return []

        def _ab_rollout_correction_get(self, key: str, default=None):
            if key == "ddp_phase_timeout_s":
                return 120.0
            return default

        def _prepare_rollout_correction_inputs(self, inputs, *, _segments_only: bool):
            assert _segments_only is True
            return [({}, {}, 1)], {}

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "monitored_barrier", lambda *_args, **_kwargs: None)

    group_timeouts = []

    def _new_group(*_args, **kwargs):
        group_timeouts.append(kwargs["timeout"].total_seconds())
        return object()

    monkeypatch.setattr(dist, "new_group", _new_group)

    captured = {}

    def _fake_nonpipeline_loop(**kwargs):
        captured.update(kwargs)
        return torch.tensor(1.0)

    monkeypatch.setattr(
        executors_mod,
        "run_rollout_correction_nonpipeline_learning_loop",
        _fake_nonpipeline_loop,
    )

    t = DummyTrainer()
    loss = t._stage2_rollout_correction_step_budgeted_train(
        t.model,
        raw_samples=[{"messages": []}],
        global_step=1,
    )

    assert isinstance(loss, torch.Tensor)
    assert captured["ddp_phase_prepare_timeout_s"] == pytest.approx(480.0)
    assert captured["ddp_phase_final_sync_timeout_s"] == pytest.approx(120.0)
    assert group_timeouts == [480.0]


def test_rollout_correction_nonpipeline_prepare_and_final_sync_barriers_use_separate_timeouts(
    monkeypatch,
):
    import contextlib

    import torch

    import src.trainers.rollout_correction.coordination as coordination_mod

    class DummyOwner:
        def __init__(self):
            self._buffer = []

        def _stage2_flush_train_monitor_dump(self, *, global_step: int):
            return None

        def _stage2_append_post_rollout_segments(self, *, channel: str, segments):
            assert channel == "rollout_correction"
            self._buffer.extend(segments)

        def _stage2_post_rollout_buffer(self, *, channel: str):
            assert channel == "rollout_correction"
            return self._buffer

        def _stage2_stage_wallclock_ctx(self, _stage: str):
            return contextlib.nullcontext()

        def _stage2_pop_post_rollout_pack(self, *, channel: str):
            assert channel == "rollout_correction"
            selected = list(self._buffer)
            self._buffer.clear()
            return selected, {}

    barrier_calls = []

    def _barrier(phase: str, *, timeout_s=None):
        barrier_calls.append((phase, timeout_s))

    monkeypatch.setattr(
        coordination_mod,
        "run_rollout_correction_train_one_pack",
        lambda **_kwargs: torch.tensor(1.0),
    )

    loss = coordination_mod.run_rollout_correction_nonpipeline_learning_loop(
        owner=DummyOwner(),
        model=object(),
        segments=[({}, {}, 1)],
        batch_metrics={},
        target_log_step=2,
        total_segments_target=1,
        ddp_phase_prepare_timeout_s=480.0,
        ddp_phase_final_sync_timeout_s=120.0,
        ddp_phase_barrier_fn=_barrier,
        dist=None,
        ddp_rank=0,
        ddp_world_size=2,
    )

    assert isinstance(loss, torch.Tensor)
    assert barrier_calls == [
        ("rollout_correction_non_pipeline_after_prepare", 480.0),
        ("rollout_correction_non_pipeline_before_final_sync_backward", 120.0),
    ]


def test_rollout_correction_nonpipeline_left_pads_shadow_slots_before_final_sync(
    monkeypatch,
):
    import contextlib

    import torch

    import src.trainers.rollout_correction.coordination as coordination_mod

    class DummyOwner:
        def __init__(self):
            self._buffer = []
            self.traces = []

        def _stage2_record_ddp_phase_trace(
            self, *, global_step, phase, rank, world_size, payload
        ):
            self.traces.append(
                {
                    "global_step": global_step,
                    "phase": phase,
                    "rank": rank,
                    "world_size": world_size,
                    "payload": dict(payload),
                }
            )

        def _stage2_flush_train_monitor_dump(self, *, global_step: int):
            return None

        def _stage2_append_post_rollout_segments(self, *, channel: str, segments):
            assert channel == "rollout_correction"
            self._buffer.extend(segments)

        def _stage2_post_rollout_buffer(self, *, channel: str):
            assert channel == "rollout_correction"
            return self._buffer

        def _stage2_stage_wallclock_ctx(self, _stage: str):
            return contextlib.nullcontext()

        def _stage2_pop_post_rollout_pack(self, *, channel: str):
            assert channel == "rollout_correction"
            selected = list(self._buffer)
            self._buffer.clear()
            return selected, {"packing/post_rollout_segments": float(len(selected))}

    barrier_calls = []
    train_calls = []

    def _barrier(phase: str, *, timeout_s=None):
        barrier_calls.append((phase, timeout_s))

    monkeypatch.setattr(
        coordination_mod,
        "gather_rollout_correction_local_pack_counts",
        lambda **_kwargs: [1, 3],
    )

    def _train_one_pack(**kwargs):
        train_calls.append(
            {
                "selected": list(kwargs["selected"]),
                "sync_gradients": bool(kwargs["sync_gradients"]),
                "shadow_zero_loss": bool(kwargs.get("shadow_zero_loss", False)),
                "pack_metrics": dict(kwargs["pack_metrics"]),
            }
        )
        return torch.tensor(1.0)

    monkeypatch.setattr(
        coordination_mod,
        "run_rollout_correction_train_one_pack",
        _train_one_pack,
    )

    owner = DummyOwner()
    loss = coordination_mod.run_rollout_correction_nonpipeline_learning_loop(
        owner=owner,
        model=object(),
        segments=[({"input_ids": [1]}, {"sample_id": "s0"}, 1)],
        batch_metrics={},
        target_log_step=2,
        total_segments_target=1,
        ddp_phase_prepare_timeout_s=480.0,
        ddp_phase_final_sync_timeout_s=120.0,
        ddp_phase_barrier_fn=_barrier,
        dist=None,
        ddp_rank=0,
        ddp_world_size=2,
    )

    assert isinstance(loss, torch.Tensor)
    assert barrier_calls == [
        ("rollout_correction_non_pipeline_after_prepare", 480.0),
        ("rollout_correction_non_pipeline_before_final_sync_backward", 120.0),
        ("rollout_correction_non_pipeline_before_final_sync_backward", 120.0),
        ("rollout_correction_non_pipeline_before_final_sync_backward", 120.0),
    ]
    assert len(train_calls) == 3
    assert [call["sync_gradients"] for call in train_calls] == [True, True, True]
    assert [call["shadow_zero_loss"] for call in train_calls] == [True, True, False]
    assert train_calls[-1]["pack_metrics"][
        "packing/post_rollout_global_slot_count"
    ] == pytest.approx(3.0)
    assert train_calls[-1]["pack_metrics"][
        "packing/post_rollout_empty_slot_count"
    ] == pytest.approx(2.0)
    assert train_calls[-1]["pack_metrics"][
        "packing/post_rollout_slot_index"
    ] == pytest.approx(2.0)
    assert train_calls[-1]["pack_metrics"][
        "packing/post_rollout_local_pack_count"
    ] == pytest.approx(1.0)
    assert train_calls[-1]["pack_metrics"][
        "packing/post_rollout_slot_is_final_sync"
    ] == pytest.approx(1.0)
    empty_traces = [
        item
        for item in owner.traces
        if item["phase"] == "rollout_correction_non_pipeline_empty_pack_slot"
    ]
    assert [item["payload"]["slot_index"] for item in empty_traces] == [0, 1]
    assert all(
        item["payload"]["sync_gradients"] == pytest.approx(1.0)
        for item in empty_traces
    )


def test_rollout_correction_nonpipeline_zero_local_packs_fail_after_count_gather(
    monkeypatch,
):
    import contextlib

    import pytest
    import torch

    import src.trainers.rollout_correction.coordination as coordination_mod

    class DummyOwner:
        def __init__(self):
            self._buffer = []

        def _stage2_flush_train_monitor_dump(self, *, global_step: int):
            return None

        def _stage2_append_post_rollout_segments(self, *, channel: str, segments):
            assert channel == "rollout_correction"
            self._buffer.extend(segments)

        def _stage2_post_rollout_buffer(self, *, channel: str):
            assert channel == "rollout_correction"
            return self._buffer

        def _stage2_stage_wallclock_ctx(self, _stage: str):
            return contextlib.nullcontext()

    barrier_calls = []
    train_calls = []

    def _barrier(phase: str, *, timeout_s=None):
        barrier_calls.append((phase, timeout_s))

    monkeypatch.setattr(
        coordination_mod,
        "gather_rollout_correction_local_pack_counts",
        lambda **_kwargs: [0, 2],
    )
    monkeypatch.setattr(
        coordination_mod,
        "run_rollout_correction_train_one_pack",
        lambda **kwargs: train_calls.append(kwargs) or torch.tensor(1.0),
    )

    with pytest.raises(RuntimeError, match="zero local packs"):
        coordination_mod.run_rollout_correction_nonpipeline_learning_loop(
            owner=DummyOwner(),
            model=object(),
            segments=[],
            batch_metrics={},
            target_log_step=2,
            total_segments_target=1,
            ddp_phase_prepare_timeout_s=480.0,
            ddp_phase_final_sync_timeout_s=120.0,
            ddp_phase_barrier_fn=_barrier,
            dist=None,
            ddp_rank=0,
            ddp_world_size=2,
        )

    assert barrier_calls == [("rollout_correction_non_pipeline_after_prepare", 480.0)]
    assert train_calls == []


def test_gather_rollout_correction_local_pack_counts_uses_object_gather_when_available():
    import src.trainers.rollout_correction.coordination as coordination_mod

    class FakeDist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def all_gather_object(self, gathered, local_value):
            assert local_value == 2
            gathered[:] = [1, 2, 3]

        def all_gather(self, *_args, **_kwargs):
            raise AssertionError("tensor all_gather should not be used")

    counts = coordination_mod.gather_rollout_correction_local_pack_counts(
        owner=object(),
        model=object(),
        local_pack_count=2,
        dist=FakeDist(),
        ddp_world_size=3,
    )

    assert counts == [1, 2, 3]


def test_gather_rollout_correction_local_pack_counts_gloo_tensor_fallback_uses_cpu():
    import torch

    import src.trainers.rollout_correction.coordination as coordination_mod

    class FakeDist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_backend(self):
            return "gloo"

        def all_gather(self, gathered, local):
            assert local.device.type == "cpu"
            gathered[0].fill_(4)
            gathered[1].fill_(5)

    class FakeModel:
        device = torch.device("cuda:0")

    counts = coordination_mod.gather_rollout_correction_local_pack_counts(
        owner=object(),
        model=FakeModel(),
        local_pack_count=4,
        dist=FakeDist(),
        ddp_world_size=2,
    )

    assert counts == [4, 5]


def test_rollout_correction_nonpipeline_uses_global_slot_count_for_sync_sequence(
    monkeypatch,
):
    import contextlib

    import torch

    import src.trainers.rollout_correction.coordination as coordination_mod

    class DummyOwner:
        def __init__(self):
            self._buffer = []

        def _stage2_flush_train_monitor_dump(self, *, global_step: int):
            return None

        def _stage2_append_post_rollout_segments(self, *, channel: str, segments):
            assert channel == "rollout_correction"
            self._buffer.extend(segments)

        def _stage2_post_rollout_buffer(self, *, channel: str):
            assert channel == "rollout_correction"
            return self._buffer

        def _stage2_stage_wallclock_ctx(self, _stage: str):
            return contextlib.nullcontext()

        def _stage2_pop_post_rollout_pack(self, *, channel: str):
            assert channel == "rollout_correction"
            selected = [self._buffer.pop(0)]
            return selected, {"packing/post_rollout_segments": 1.0}

    barrier_calls = []
    train_calls = []

    def _barrier(phase: str, *, timeout_s=None):
        barrier_calls.append((phase, timeout_s))

    monkeypatch.setattr(
        coordination_mod,
        "gather_rollout_correction_local_pack_counts",
        lambda **_kwargs: [1, 2],
    )

    def _train_one_pack(**kwargs):
        train_calls.append(
            {
                "sync_gradients": bool(kwargs["sync_gradients"]),
                "pack_metrics": dict(kwargs["pack_metrics"]),
            }
        )
        return torch.tensor(1.0)

    monkeypatch.setattr(
        coordination_mod,
        "run_rollout_correction_train_one_pack",
        _train_one_pack,
    )

    loss = coordination_mod.run_rollout_correction_nonpipeline_learning_loop(
        owner=DummyOwner(),
        model=object(),
        segments=[
            ({"input_ids": [1]}, {"sample_id": "s0"}, 1),
            ({"input_ids": [2]}, {"sample_id": "s1"}, 1),
        ],
        batch_metrics={},
        target_log_step=2,
        total_segments_target=2,
        ddp_phase_prepare_timeout_s=480.0,
        ddp_phase_final_sync_timeout_s=120.0,
        ddp_phase_barrier_fn=_barrier,
        dist=None,
        ddp_rank=1,
        ddp_world_size=2,
    )

    assert isinstance(loss, torch.Tensor)
    assert [call["sync_gradients"] for call in train_calls] == [True, True]
    assert [
        call["pack_metrics"]["packing/post_rollout_slot_index"]
        for call in train_calls
    ] == [pytest.approx(0.0), pytest.approx(1.0)]
    assert barrier_calls == [
        ("rollout_correction_non_pipeline_after_prepare", 480.0),
        ("rollout_correction_non_pipeline_before_final_sync_backward", 120.0),
        ("rollout_correction_non_pipeline_before_final_sync_backward", 120.0),
    ]


def test_stage2_shared_metric_collectives_use_bounded_entry_barrier() -> None:
    import torch

    from src.trainers.stage2_coordination import (
        reduce_metric_payload_global,
        resolve_stage2_rollout_correction_metric_spec,
    )

    barrier_calls = []
    all_reduce_calls = []
    all_gather_calls = []

    class _FakeReduceOp:
        SUM = "sum"
        MAX = "max"

    class _FakeDist:
        ReduceOp = _FakeReduceOp

        def new_group(self, **kwargs):
            return {"timeout_s": kwargs["timeout"].total_seconds()}

        def monitored_barrier(self, *, group=None, timeout=None, wait_all_ranks=None):
            barrier_calls.append(
                {
                    "group": group,
                    "timeout_s": timeout.total_seconds() if timeout is not None else None,
                    "wait_all_ranks": wait_all_ranks,
                }
            )

        def all_gather_object(self, gathered: list[object], obj: object) -> None:
            all_gather_calls.append(list(obj))
            for i in range(len(gathered)):
                gathered[i] = list(obj) if isinstance(obj, list) else obj

        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            all_reduce_calls.append({"op": op, "values": tensor.tolist()})

    class _Owner:
        def _dist_info(self):
            return 0, 2, _FakeDist()

        def _ab_rollout_correction_get(self, key: str, default=None):
            if key == "ddp_phase_timeout_s":
                return 75.0
            return default

    out = reduce_metric_payload_global(
        _Owner(),
        {
            "stage2/_log_weight_total": 1.0,
            "loss/coord/bbox_smoothl1": 2.0,
        },
        resolver=resolve_stage2_rollout_correction_metric_spec,
        error_prefix="stage2-ab metric",
    )

    assert out["loss/coord/bbox_smoothl1"] == pytest.approx(2.0)
    assert len(barrier_calls) == 2
    assert all(call["timeout_s"] == pytest.approx(75.0) for call in barrier_calls)
    assert len(all_gather_calls) == 1
    assert len(all_reduce_calls) == 1


def test_stage2_rollout_correction_rollout_correction_and_b_share_final_sync_barrier_policy(monkeypatch):
    import contextlib

    import torch
    import torch.distributed as dist

    import src.trainers.rollout_correction.coordination as coordination_mod
    import src.trainers.rollout_correction.executors as executors_mod
    from src.trainers.stage2_coordination import Stage2DDPPhaseConfig
    from src.trainers.rollout_correction.executors import RolloutCorrectionExecutorsMixin

    class DummyModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def train(self):
            return self

        def no_sync(self):
            return contextlib.nullcontext()

    class DummyTemplate:
        def data_collator(self, encoded_batch):
            return {"input_ids": encoded_batch}

    class DummyTrainer(RolloutCorrectionExecutorsMixin):
        def __init__(self):
            self.model = DummyModel()
            self.template = DummyTemplate()
            self._buffers = {"A": [], "B": []}

        def _packing_enabled(self):
            return True

        def _packing_length(self):
            return 16

        def _packing_min_fill_ratio(self):
            return 0.5

        def _rollout_backend(self):
            return "vllm"

        def _vllm_mode(self):
            return "server"

        def _vllm_server_timeouts(self):
            return 240.0, 240.0

        def _ab_rollout_correction_get(self, key: str, default=None):
            return default

        def _stage2_rollout_correction_pipeline_enabled(self, *, backend: str, mode: str):
            return False

        def _rollout_decode_batch_size_per_rank(self):
            return 1

        def _stage2_stage_wallclock_ctx(self, _stage: str):
            return contextlib.nullcontext()

        def _stage2_reset_train_monitor_dump(self, *, global_step: int):
            return None

        def _stage2_flush_train_monitor_dump(self, *, global_step: int):
            return None

        def _template_packing_enabled(self):
            return contextlib.nullcontext()

        def _assert_single_packed_forward(self, batch, *, where: str):
            return None

        def _merge_rollout_matching_batch_metrics(self, batch, metrics):
            batch["_merged_metrics"] = dict(metrics)

        def compute_loss_context_manager(self):
            return contextlib.nullcontext()

        def compute_loss(self, model, batch):
            return torch.tensor(1.0, requires_grad=True)

        def _stage2_rollout_correction_disable_average_tokens_across_devices_for_packed_step(
            self, **_kwargs
        ):
            return contextlib.nullcontext()

        def _prepare_rollout_correction_inputs(self, inputs, *, _segments_only: bool):
            assert _segments_only is True
            return [({"input_ids": [1]}, {}, 1)], {}

        def _prepare_rollout_correction_inputs(self, inputs, *, _segments_only: bool):
            assert _segments_only is True
            return [({"input_ids": [1]}, {}, 1)], {}

        def _stage2_append_post_rollout_segments(self, *, channel: str, segments):
            self._buffers[str(channel)].extend(list(segments))

        def _stage2_post_rollout_buffer(self, *, channel: str):
            return self._buffers[str(channel)]

        def _stage2_pop_post_rollout_pack(self, *, channel: str):
            selected = list(self._buffers[str(channel)])
            self._buffers[str(channel)].clear()
            return selected, {}

    shared_config = Stage2DDPPhaseConfig(
        monitor_enabled=True,
        final_sync_timeout_s=210.0,
        monitor_group_timeout_s=210.0,
    )
    prime_calls = []

    monkeypatch.setattr(
        executors_mod,
        "resolve_stage2_rollout_correction_ddp_phase_config",
        lambda owner, *, ddp_world_size: shared_config,
    )
    monkeypatch.setattr(
        coordination_mod,
        "resolve_stage2_rollout_correction_ddp_phase_config",
        lambda owner, *, ddp_world_size: shared_config,
    )
    monkeypatch.setattr(
        executors_mod,
        "prime_stage2_ddp_monitor_group",
        lambda owner, *, dist, rank, world_size, config, logger=None: prime_calls.append(
            (rank, world_size, config.final_sync_timeout_s, config.monitor_group_timeout_s)
        ),
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "monitored_barrier", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dist, "new_group", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda gathered, local: gathered.__setitem__(slice(None), [local, local]),
    )

    a_barrier_calls = []

    trainer = DummyTrainer()
    trainer._stage2_rollout_correction_ddp_monitored_barrier = (  # type: ignore[method-assign]
        lambda **kwargs: a_barrier_calls.append(dict(kwargs))
    )

    loss_a = trainer._stage2_a_step_budgeted_train(
        trainer.model,
        raw_samples=[{"messages": []}],
        global_step=1,
    )

    captured_b = {}
    monkeypatch.setattr(
        executors_mod,
        "run_rollout_correction_nonpipeline_learning_loop",
        lambda **kwargs: captured_b.update(kwargs) or torch.tensor(1.0),
    )

    loss_b = trainer._stage2_rollout_correction_step_budgeted_train(
        trainer.model,
        raw_samples=[{"messages": []}],
        global_step=1,
    )

    assert isinstance(loss_a, torch.Tensor)
    assert isinstance(loss_b, torch.Tensor)
    assert a_barrier_calls == [
        {
            "dist": dist,
            "phase": "stage2-ab rollout-correction after-prepare",
            "rank": 0,
            "world_size": 2,
            "timeout_s": pytest.approx(210.0),
            "monitor_group_timeout_s": pytest.approx(210.0),
        },
        {
            "dist": dist,
            "phase": "stage2-ab rollout-correction final-sync backward",
            "rank": 0,
            "world_size": 2,
            "timeout_s": pytest.approx(210.0),
            "monitor_group_timeout_s": pytest.approx(210.0),
        }
    ]
    assert captured_b["ddp_phase_final_sync_timeout_s"] == pytest.approx(210.0)
    assert captured_b["ddp_phase_prepare_timeout_s"] == pytest.approx(480.0)
    assert prime_calls == [
        (0, 2, 210.0, 210.0),
        (0, 2, 210.0, 480.0),
    ]

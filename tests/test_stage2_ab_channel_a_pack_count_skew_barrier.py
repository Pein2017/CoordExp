import contextlib

import pytest


def test_stage2_rollout_correction_rollout_correction_calls_barrier_on_final_pack(monkeypatch):
    """rollout-correction step-budgeted packing must align ranks on the final (sync) backward.

    Without a barrier, differing per-rank pack counts can deadlock DDP because one rank
    enters the synchronized backward (allreduce) while another rank is still in a
    `no_sync()` micro-pack.
    """

    import torch
    import torch.distributed as dist

    import src.trainers.rollout_correction.executors as executors_mod
    from src.trainers.rollout_correction.executors import RolloutCorrectionExecutorsMixin

    class DummyModel:
        device = torch.device("cpu")

        def train(self):
            return self

        def no_sync(self):
            return contextlib.nullcontext()

    class DummyTemplate:
        def data_collator(self, _batch):
            return {"loss": torch.tensor(1.0, requires_grad=True)}

    class DummyTrainer(RolloutCorrectionExecutorsMixin):
        def __init__(self):
            self.model = DummyModel()
            self.template = DummyTemplate()

        def _packing_enabled(self):
            return True

        def _packing_buffer_cap(self):
            return 256

        def _packing_length(self):
            return 4

        def _packing_min_fill_ratio(self):
            return 0.0

        def _rollout_backend(self):
            return "hf"

        def _vllm_mode(self):
            return ""

        def _stage2_rollout_correction_pipeline_enabled(self, *, backend: str, mode: str):
            return False

        def _rollout_decode_batch_size_per_rank(self):
            return 1

        def _rollout_correction_cfg_get(self, _key: str, default=None):
            return default

        def _vllm_server_timeouts(self):
            return 1.0, 1.0

        def _stage2_reset_train_monitor_dump(self, *, global_step: int):
            self._stage2_train_monitor_dump_written_step = global_step

        def _stage2_flush_train_monitor_dump(self, *, global_step: int):
            self._stage2_train_monitor_dump_written_step = global_step

        def _stage2_stage_wallclock_ctx(self, _stage: str):
            return contextlib.nullcontext()

        def _stage2_record_ddp_phase_trace(self, **_kwargs):
            return None

        def _template_packing_enabled(self):
            return contextlib.nullcontext()

        def _assert_single_packed_forward(self, _batch, *, where: str):
            return None

        def _merge_rollout_matching_batch_metrics(self, _batch, _metrics):
            return None

        def compute_loss(self, _model, batch):
            return batch["loss"]

        def _prepare_rollout_correction_inputs(self, inputs, *, _segments_only: bool):
            assert _segments_only is True
            segs = [({"input_ids": [1]}, {}, 1) for _ in inputs]
            return segs, {}

        def _select_post_rollout_segment_indices(
            self, _encoded_lens, _packing_length, *, min_fill_ratio=None
        ):
            # Always pick just the oldest segment so we get multiple packs.
            return [0]

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda gathered, local: gathered.__setitem__(slice(None), [local, local]),
    )

    barrier_calls = {"n": 0}

    def _monitored_barrier(**_kwargs):
        barrier_calls["n"] += 1

    monkeypatch.setattr(
        executors_mod,
        "run_rollout_correction_ddp_monitored_barrier",
        _monitored_barrier,
    )

    t = DummyTrainer()
    loss = t._stage2_rollout_correction_step_budgeted_train(
        t.model,
        raw_samples=[{}, {}, {}],
        global_step=1,
    )

    assert isinstance(loss, torch.Tensor)
    assert barrier_calls["n"] == 2


def test_stage2_rollout_correction_rollout_correction_uses_shadow_slots_for_pack_count_skew(monkeypatch):
    import torch
    import torch.distributed as dist

    import src.trainers.rollout_correction.executors as executors_mod
    from src.trainers.rollout_correction.executors import RolloutCorrectionExecutorsMixin

    class DummyModel:
        device = torch.device("cpu")

        def train(self):
            return self

        def no_sync(self):
            return contextlib.nullcontext()

    class DummyTemplate:
        def data_collator(self, _batch):
            return {"loss": torch.tensor(1.0, requires_grad=True)}

    class DummyTrainer(RolloutCorrectionExecutorsMixin):
        def __init__(self):
            self.model = DummyModel()
            self.template = DummyTemplate()
            self.shadow_flags = []
            self.sync_flags = []

        def _packing_enabled(self):
            return True

        def _packing_buffer_cap(self):
            return 256

        def _packing_length(self):
            return 4

        def _packing_min_fill_ratio(self):
            return 0.0

        def _rollout_backend(self):
            return "hf"

        def _vllm_mode(self):
            return ""

        def _stage2_rollout_correction_pipeline_enabled(self, *, backend: str, mode: str):
            return False

        def _rollout_decode_batch_size_per_rank(self):
            return 1

        def _rollout_correction_cfg_get(self, _key: str, default=None):
            return default

        def _vllm_server_timeouts(self):
            return 1.0, 1.0

        def _stage2_reset_train_monitor_dump(self, *, global_step: int):
            self._stage2_train_monitor_dump_written_step = global_step

        def _stage2_flush_train_monitor_dump(self, *, global_step: int):
            self._stage2_train_monitor_dump_written_step = global_step

        def _stage2_stage_wallclock_ctx(self, _stage: str):
            return contextlib.nullcontext()

        def _stage2_record_ddp_phase_trace(self, **_kwargs):
            return None

        def _template_packing_enabled(self):
            return contextlib.nullcontext()

        def _assert_single_packed_forward(self, _batch, *, where: str):
            return None

        def _merge_rollout_matching_batch_metrics(self, _batch, _metrics):
            return None

        def compute_loss(self, _model, batch):
            self.shadow_flags.append(bool(batch.get("_stage2_rollout_correction_shadow_pack", False)))
            self.sync_flags.append(
                bool(getattr(self, "_loss_gradient_monitor_sync_gradients", False))
            )
            return batch["loss"]

        def _prepare_rollout_correction_inputs(self, inputs, *, _segments_only: bool):
            assert _segments_only is True
            return [({"input_ids": [1]}, {}, 1) for _ in inputs], {}

        def _select_post_rollout_segment_indices(
            self, _encoded_lens, _packing_length, *, min_fill_ratio=None
        ):
            return [0]

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda gathered, local: gathered.__setitem__(slice(None), [1, 3]),
    )

    barrier_calls = {"n": 0}

    def _monitored_barrier(**_kwargs):
        barrier_calls["n"] += 1

    monkeypatch.setattr(
        executors_mod,
        "run_rollout_correction_ddp_monitored_barrier",
        _monitored_barrier,
    )

    t = DummyTrainer()
    loss = t._stage2_rollout_correction_step_budgeted_train(
        t.model,
        raw_samples=[{}],
        global_step=1,
    )

    assert isinstance(loss, torch.Tensor)
    assert t.shadow_flags == [True, True, False]
    assert t.sync_flags == [True, True, True]
    assert barrier_calls["n"] == 4

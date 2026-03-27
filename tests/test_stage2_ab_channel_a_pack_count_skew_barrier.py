import contextlib

import pytest


def test_stage2_ab_channel_a_calls_barrier_on_final_pack(monkeypatch):
    """Channel-A step-budgeted packing must align ranks on the final (sync) backward.

    Without a barrier, differing per-rank pack counts can deadlock DDP because one rank
    enters the synchronized backward (allreduce) while another rank is still in a
    `no_sync()` micro-pack.
    """

    import torch
    import torch.distributed as dist

    from src.trainers.stage2_two_channel.executors import Stage2ABChannelExecutorsMixin

    class DummyModel:
        device = torch.device("cpu")

        def train(self):
            return self

        def no_sync(self):
            return contextlib.nullcontext()

    class DummyTemplate:
        def data_collator(self, _batch):
            return {"loss": torch.tensor(1.0, requires_grad=True)}

    class DummyTrainer(Stage2ABChannelExecutorsMixin):
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

        def _template_packing_enabled(self):
            return contextlib.nullcontext()

        def _assert_single_packed_forward(self, _batch, *, where: str):
            return None

        def _merge_rollout_matching_batch_metrics(self, _batch, _metrics):
            return None

        def compute_loss(self, _model, batch):
            return batch["loss"]

        def _prepare_batch_inputs_a(self, inputs, *, _segments_only: bool):
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

    barrier_calls = {"n": 0}

    def _monitored_barrier(self, **_kwargs):
        barrier_calls["n"] += 1

    monkeypatch.setattr(
        Stage2ABChannelExecutorsMixin,
        "_stage2_ab_ddp_monitored_barrier",
        _monitored_barrier,
    )

    t = DummyTrainer()
    loss = t._stage2_a_step_budgeted_train(
        t.model,
        raw_samples=[{}, {}, {}],
        global_step=1,
    )

    assert isinstance(loss, torch.Tensor)
    assert barrier_calls["n"] == 1

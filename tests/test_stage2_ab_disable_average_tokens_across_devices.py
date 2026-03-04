import contextlib
from types import SimpleNamespace


def test_stage2_ab_step_budgeted_disables_average_tokens_across_devices(monkeypatch):
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
            self.args = SimpleNamespace(average_tokens_across_devices=True)

        def _packing_enabled(self):
            return True

        def _packing_buffer_cap(self):
            return 256

        def _packing_length(self):
            return 4

        def _packing_min_fill_ratio(self):
            return 0.0

        def _packing_min_fill_ratio_a(self):
            return 0.0

        def _template_packing_enabled(self):
            return contextlib.nullcontext()

        def _assert_single_packed_forward(self, _batch, *, where: str):
            return None

        def _merge_rollout_matching_batch_metrics(self, _batch, _metrics):
            return None

        def compute_loss(self, _model, batch):
            # Must be forced off for DDP safety inside per-pack loops.
            assert self.args.average_tokens_across_devices is False
            return batch["loss"]

        def _prepare_batch_inputs_a(self, inputs, *, _segments_only: bool):
            assert _segments_only is True
            segs = [({"input_ids": [1]}, {}, 1) for _ in inputs]
            return segs, {}

        def _prepare_batch_inputs_b(self, inputs, *, _segments_only: bool):
            assert _segments_only is True
            segs = [({"input_ids": [1]}, {}, 1) for _ in inputs]
            return segs, {}

        def _select_post_rollout_segment_indices(
            self, _encoded_lens, _packing_length, *, min_fill_ratio=None
        ):
            # Always pick just the oldest segment so we get multiple packs.
            return [0]

        def _rollout_backend(self):
            return "hf"

        def _vllm_mode(self):
            return "local"

        def _rollout_decode_batch_size_per_rank(self):
            return 1

        def _ab_channel_b_get(self, key: str, default=None):
            if key == "ddp_phase_timeout_s":
                # Avoid needing dist.monitored_barrier in unit tests.
                return 0.0
            return default

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "barrier", lambda *_args, **_kwargs: None)

    t = DummyTrainer()

    loss_a = t._stage2_a_step_budgeted_train(
        t.model,
        raw_samples=[{}, {}, {}],
        global_step=1,
    )
    assert isinstance(loss_a, torch.Tensor)
    assert t.args.average_tokens_across_devices is True

    loss_b = t._stage2_b_step_budgeted_train(
        t.model,
        raw_samples=[{}, {}, {}],
        global_step=1,
    )
    assert isinstance(loss_b, torch.Tensor)
    assert t.args.average_tokens_across_devices is True

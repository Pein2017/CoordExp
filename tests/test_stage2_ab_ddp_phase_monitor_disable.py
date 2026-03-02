import pytest


def test_stage2_ab_channel_b_ddp_phase_timeout_zero_disables_monitor(monkeypatch):
    """`ddp_phase_timeout_s: 0` should disable the optional monitored barriers.

    This is a production-safety knob: monitored barriers are useful for debugging deadlocks,
    but they can also introduce large idle windows when Channel-B pack counts differ across ranks.
    """

    import torch.distributed as dist

    from src.trainers.stage2_two_channel.executors import Stage2ABChannelExecutorsMixin

    class DummyModel:
        def train(self):
            return self

    class DummyTrainer(Stage2ABChannelExecutorsMixin):
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

        def _stage2_channel_b_pipeline_enabled(self, *, backend: str, mode: str):
            return False

        def _rollout_decode_batch_size_per_rank(self):
            return 1

        def _ab_channel_b_get(self, key: str, default=None):
            if key == "ddp_phase_timeout_s":
                return 0
            return default

        def _prepare_batch_inputs_b(self, inputs, *, _segments_only: bool):
            assert _segments_only is True
            return [({}, {}, 1)], {}

        def _stage2_append_post_rollout_segments(self, *, channel: str, segments):
            # Intentionally skip appending so the step exits before any forward/backward.
            return None

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "dist.new_group/monitored_barrier should not be called when ddp_phase_timeout_s <= 0"
        )

    monkeypatch.setattr(dist, "new_group", _boom)
    monkeypatch.setattr(dist, "monitored_barrier", _boom)

    t = DummyTrainer()
    with pytest.raises(AssertionError, match="produced no packs"):
        t._stage2_b_step_budgeted_train(
            t.model,
            raw_samples=[{"messages": []}],
            global_step=1,
        )

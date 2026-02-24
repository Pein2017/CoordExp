import os
import types

from src.trainers.stage2_rollout_aligned import RolloutMatchingSFTTrainer


def _make_trainer(tmp_path) -> RolloutMatchingSFTTrainer:
    t = object.__new__(RolloutMatchingSFTTrainer)
    t.args = types.SimpleNamespace(
        output_dir=str(tmp_path), logging_steps=1, logging_first_step=True
    )
    t.is_world_process_zero = True
    t._vllm_server_debug_dump_count = 0
    t._vllm_server_debug_last_step = None
    t.rollout_matching_cfg = {
        "vllm": {
            "server": {
                "debug_dump": {
                    "enabled": True,
                    "every_steps": 1,
                    "dump_first_step": True,
                    "only_world_process_zero": True,
                    "max_events": 1,
                    "max_samples": 1,
                    "max_chars": 0,
                    "async_write": False,
                    "min_free_gb": 0.0,
                }
            }
        }
    }
    return t


def test_vllm_server_debug_dump_does_not_raise_on_makedirs_failure(
    monkeypatch, tmp_path
) -> None:
    t = _make_trainer(tmp_path)

    def _boom(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(os, "makedirs", _boom)

    # Best-effort debug dumps: should not crash training if dump dir cannot be created.
    t._maybe_debug_dump_vllm_server_rollouts(
        global_step=0,
        seed_base=0,
        infer_requests=[{"prompt": "x"}],
        outputs=[([1, 2], "resp", "prefix", [3, 4])],
        samples=[{"messages": [{"role": "assistant", "content": "gt"}]}],
    )

import types

from src.trainers.stage2_rollout_aligned import RolloutMatchingSFTTrainer


def test_monitor_dump_does_not_raise_on_json_io_error(monkeypatch, tmp_path) -> None:
    t = object.__new__(RolloutMatchingSFTTrainer)
    t.args = types.SimpleNamespace(output_dir=str(tmp_path))
    t.rollout_matching_cfg = {
        "monitor_dump": {"out_dir": str(tmp_path / "dumps"), "write_markdown": False}
    }

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _boom)

    # Best-effort I/O: should not crash training.
    t._write_monitor_dump(global_step=0, payload={"global_step": 0})


def test_monitor_dump_does_not_raise_on_markdown_format_error(tmp_path) -> None:
    t = object.__new__(RolloutMatchingSFTTrainer)
    t.args = types.SimpleNamespace(output_dir=str(tmp_path))
    t.rollout_matching_cfg = {
        "monitor_dump": {"out_dir": str(tmp_path / "dumps"), "write_markdown": True}
    }

    def _fmt(_payload):
        raise RuntimeError("formatting failed")

    # Monkeypatch instance method to simulate markdown formatter failure.
    t._format_monitor_dump_markdown = _fmt  # type: ignore[assignment]

    t._write_monitor_dump(global_step=0, payload={"global_step": 0})


def test_train_monitor_dump_markdown_keeps_full_text(tmp_path) -> None:
    t = object.__new__(RolloutMatchingSFTTrainer)
    t.args = types.SimpleNamespace(output_dir=str(tmp_path))
    t.rollout_matching_cfg = {"train_monitor_dump": {"max_text_chars": 8}}

    full_text = "very long rollout text that should stay intact"
    md = t._format_monitor_dump_markdown(
        {
            "kind": "train_monitor_dump",
            "global_step": 3,
            "samples": [
                {
                    "sample_id": "sample-1",
                    "base_idx": 1,
                    "image_id": 99,
                    "images": ["img.jpg"],
                    "messages": [],
                    "rollout_text": full_text,
                    "prefix_text": full_text,
                    "train_text": full_text,
                    "gt": [{"desc": "person", "bbox_2d": [1, 2, 3, 4]}],
                    "pred": [{"desc": "person", "bbox_2d": [1, 2, 3, 4]}],
                    "match": {},
                    "stats": {},
                }
            ],
        }
    )

    assert full_text in md
    assert "...<truncated>" not in md


def test_eval_monitor_dump_uses_every_evals_cadence() -> None:
    t = object.__new__(RolloutMatchingSFTTrainer)
    t.rollout_matching_cfg = {
        "eval_monitor_dump": {"enabled": True, "every_evals": 2}
    }
    t.is_world_process_zero = True
    t._eval_monitor_dump_count = 0
    t._eval_monitor_dump_last_eval = None

    assert t._should_eval_monitor_dump(global_step=300, eval_index=1) is False
    assert t._should_eval_monitor_dump(global_step=600, eval_index=2) is True

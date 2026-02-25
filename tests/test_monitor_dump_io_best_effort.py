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

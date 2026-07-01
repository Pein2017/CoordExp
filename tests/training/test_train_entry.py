from __future__ import annotations

import json
from pathlib import Path

from src.config.loader import load_train_config
from src.train import main


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


def test_train_entry_is_config_first_and_delegates_to_runner(capsys) -> None:
    calls: list[tuple[str, str]] = []

    def fake_runner(config_path: Path) -> dict[str, object]:
        resolved = load_train_config(config_path)
        calls.append((str(config_path), resolved.fingerprint))
        return {
            "run_dir": "/tmp/coordexp-swift-smoke",
            "resolved_config_fingerprint": resolved.fingerprint,
            "completed_steps": 5,
        }

    exit_code = main(["--config", str(FIXTURE_CONFIG)], runner=fake_runner)

    assert exit_code == 0
    assert calls and calls[0][0] == str(FIXTURE_CONFIG)
    summary = json.loads(capsys.readouterr().out)
    assert summary["entry_config_path"] == str(FIXTURE_CONFIG)
    assert summary["run_dir"] == "/tmp/coordexp-swift-smoke"
    assert summary["completed_steps"] == 5
    assert summary["resolved_config_fingerprint"] == calls[0][1]

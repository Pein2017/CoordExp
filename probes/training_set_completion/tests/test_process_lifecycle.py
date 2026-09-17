from __future__ import annotations

from pathlib import Path
import sys
import time

import pytest

from probes.training_set_completion import coco227_trial, dual_start
from probes.training_set_completion import source256_normalized_controller as controller
from src.runtime.owned_process import spawn_logged_process, terminate_owned_process


@pytest.mark.parametrize("profile", [dual_start, coco227_trial, controller])
def test_profile_keeps_resource_choices_outside_shared_spawn(
    profile, tmp_path, monkeypatch
):
    observed = []

    def spawn(command, **kwargs):
        observed.append((command, kwargs))
        return ("owned-child", "log", 1.0)

    monkeypatch.setattr(profile, "spawn_logged_process", spawn)
    result = profile._spawn(
        ["test-only"], visible_devices="3,5", log_path=tmp_path / "log"
    )
    assert result == ("owned-child", "log", 1.0)
    command, options = observed.pop()
    assert command == ["test-only"]
    assert options["env"] == {
        "CUDA_VISIBLE_DEVICES": "3,5",
        "OMP_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    }
    assert options["cwd"] == Path(profile.__file__).resolve().parents[2]


def test_controller_pre_wait_timeout_reaps_child_and_closes_log(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    process, stream, _ = spawn_logged_process(
        command,
        cwd=tmp_path,
        log_path=tmp_path / "child.log",
        env={},
    )
    try:
        with pytest.raises(TimeoutError, match="training wall deadline"):
            controller._wait_one(
                process=process,
                stream=stream,
                command=command,
                name="training",
                gpu=None,
                started=time.monotonic() - 10,
                wall_seconds=1,
            )
        assert process.poll() is not None and stream.closed
    finally:
        terminate_owned_process(process)
        stream.close()

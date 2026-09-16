import json
from pathlib import Path
import subprocess

import pytest

from probes.training_set_completion import refresh as r


def test_refresh_plan_is_exactly_one_seeded_route_per_policy_per_image():
    rows = r.request_plan()
    assert len(rows) == 44
    assert [row for row in rows if row["kind"] == "greedy"][0]["seed"] == r.SEED_ROOT
    assert len({row["seed"] for row in rows if row["kind"] == "sample"}) == 33
    assert {row["image_id"] for row in rows} == set(r.IMAGE_IDS)


def test_refresh_plan_rejects_a_missing_policy_route():
    rows = r.request_plan()
    with pytest.raises(ValueError, match="denominator"):
        r.validate_request_plan(rows[:-1])


def test_greedy_comparison_requires_literal_token_identity():
    expected = {"route_id": "stage01:image-000000000001:greedy", "generated_token_ids": [1, 2], "generated_token_ids_sha256": r.digest([1, 2])}
    observed = {"request": {"request_id": "stage02:image-000000000001:greedy"}, "image_id": 1, "generated_token_ids": [1, 3], "generated_token_ids_sha256": r.digest([1, 3])}
    assert r._compare_greedy(observed, expected)["same_token_ids"] is False


def test_stop_owned_process_escalates_to_kill_after_terminate_timeout():
    class Process:
        pid = 9001

        def __init__(self):
            self.terminated = False
            self.killed = False

        def poll(self):
            return None if not self.killed else -9

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired(["fake-worker"], timeout)
            return -9

    process = Process()
    assert r._stop_owned_process(process) == -9
    assert process.terminated is True
    assert process.killed is True


def test_refresh_controller_preflight_timeout_reaps_worker_and_writes_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    output = tmp_path / "output"
    monkeypatch.setattr(r, "validate_manifest", lambda value: None)

    class Process:
        pid = 9002

        def __init__(self, stdout):
            self.stdout_target = stdout
            self.terminated = False
            self.killed = False

        def poll(self):
            if self.killed:
                return -9
            if self.terminated:
                return -15
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            if self.terminated:
                return -15
            if self.killed:
                return -9
            raise subprocess.TimeoutExpired(["fake-preflight"], timeout)

    processes = []

    def popen(*args, **kwargs):
        process = Process(kwargs["stdout"])
        processes.append(process)
        return process

    monkeypatch.setattr(r.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="TimeoutExpired"):
        r.controller(manifest_path=manifest_path, output=output)

    assert len(processes) == 1
    assert processes[0].terminated is True
    assert processes[0].stdout_target.closed is True
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "failed"
    assert terminal["cleanup_errors"] == []
    assert terminal["error"].startswith("TimeoutExpired:")

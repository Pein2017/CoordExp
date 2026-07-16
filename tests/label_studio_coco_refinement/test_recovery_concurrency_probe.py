from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.probes.label_studio_coco_refinement import recovery_concurrency as probe


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/status").exists(),
    reason="the executable probe requires POSIX signals and Linux /proc",
)


@pytest.mark.parametrize("cut", probe.RECOVERY_CUTS, ids=lambda cut: cut.name)
def test_recovery_cut_emits_complete_nondestructive_receipt(
    tmp_path: Path,
    cut: probe.RecoveryCut,
) -> None:
    receipt = probe.run_cut(tmp_path, cut)

    assert receipt["cut"] == cut.name
    assert receipt["boundary"] == cut.boundary
    assert receipt["worker"]["stopped_state"] == "T"
    assert receipt["worker"]["terminal_signal"] == "SIGKILL"
    for surface in ("supported_reader", "status_query"):
        observation = receipt["barrier_observations"][surface]
        assert observation["outcome"] == "error"
        assert observation["type"] == "StoreBusyError"
    admission = receipt["barrier_observations"]["second_enqueue"]
    if cut.queue_lock_held:
        assert admission["outcome"] == "timeout"
    else:
        assert admission == {
            "outcome": "ok",
            "batch_id": "os-kill-batch",
            "status": "running",
        }

    recovery = receipt["recovery"]
    assert recovery["status"] == cut.expected_status.value
    assert recovery["generation"] == cut.expected_generation
    assert recovery["changed_row_indices"] == (
        [0, 1] if cut.expected_status is probe.BatchStatus.SUCCEEDED else []
    )
    assert recovery["untouched_row_preserved"] is True
    assert recovery["prepared_records"] == 1
    assert recovery["journal_terminals"] == 1
    assert recovery["queue_terminals"] == 1
    assert recovery["repeated_recovery_byte_identical"] is True
    assert recovery["orphan_candidates"] == 0
    assert receipt["immutable_fixture"]["unchanged"] is True


def test_cli_writes_same_receipt_that_it_prints(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(probe.__file__).resolve()),
            "--cut",
            "before-working-rename",
            "--receipt",
            str(receipt_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=probe.WAIT_SECONDS * 3,
    )

    printed = json.loads(completed.stdout)
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert printed == persisted
    assert printed["code_identity"] == {
        "platform": printed["code_identity"]["platform"],
        "probe_sha256": probe.sha256_file(Path(probe.__file__).resolve()),
        "python_version": probe.platform.python_version(),
        "store_sha256": probe.sha256_file(
            probe.REPO_ROOT / "src/label_studio_coco_refinement/store.py"
        ),
    }
    assert printed["summary"] == {
        "all_barriers_fail_closed": True,
        "all_fixtures_unchanged": True,
        "all_recovered_exactly_once": True,
        "cut_count": 1,
        "runtime_seconds": printed["summary"]["runtime_seconds"],
    }

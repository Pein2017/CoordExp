import json
from pathlib import Path

import pytest

from probes.coordinate_representation.coordinate_codebook_alignment.execute import run


def test_new_budget_is_pinned_to_one_ledger_and_completed_job(tmp_path: Path):
    root = tmp_path / "budget"
    assert run(
        "cpu-entry", [0], ["python", "-c", "pass"], root,
        wall_seconds=4 * 3600, gpu_seconds=32 * 3600, reserve_seconds=900,
    ) == 0
    receipt = json.loads((root / "cost.json").read_text())
    assert receipt["limits"] == {"wall_seconds": 4 * 3600, "gpu_seconds": 32 * 3600}
    assert receipt["jobs"][0]["state"] == "terminal"
    assert receipt["jobs"][0]["terminal_time"] >= receipt["wall_start"]

    with pytest.raises(ValueError, match="ledger limits differ"):
        run(
            "wrong-limit", [0], ["python", "-c", "pass"], root,
            wall_seconds=8 * 3600, gpu_seconds=64 * 3600, reserve_seconds=900,
        )
    assert len(json.loads((root / "cost.json").read_text())["jobs"]) == 1

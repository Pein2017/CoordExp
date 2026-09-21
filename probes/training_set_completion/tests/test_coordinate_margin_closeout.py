"""Exercise closeout storage with synthetic accounting, without model execution."""

import json

from probes.training_set_completion.coordinate_margin import closeout
from src.artifacts.output_layout import scan_output_root


def test_closeout_keeps_reconstruction_map_in_json(tmp_path, monkeypatch):
    root, unit = tmp_path / "outputs", tmp_path / "unit"
    root.mkdir()
    unit.mkdir()
    monkeypatch.setattr(closeout, "ROOT", root)
    monkeypatch.setattr(closeout, "UNIT", unit)
    (unit / "unit.md").write_text("Synthetic storage test\n")
    closeout.write(unit / "state.json", {})
    contributions = {"input": 0, "attention": [0], "mlp": [0]}
    pairs = [dict(competitor=n, raw_margin=0, equal_norm_margin=0,
                  symmetric_direction_term=0, symmetric_length_term=0,
                  contributions=contributions, equal_norm_contributions=contributions)
             for n in (1, 30)]
    ids = [f"{model}-{policy}-offset15" for model in ("tied", "untied")
           for policy in ("original", "pair01", "only0", "pair0")]
    states = [dict(id=name, pairs=pairs, raw_winner_bin=0, equal_norm_winner_bin=0,
                   checks={"residual_sum_max_abs": 0, "pair_margin_max_abs": 0},
                   corruption_detected=True) for name in ids]
    for name in ids:
        directory = root / "runtime" / name
        directory.mkdir(parents=True)
        closeout.write(directory / "receipt.json", dict(
            status="candidate_complete", pid="nonexistent-storage-test", state_id=name,
            model_forwards=0, gpu_seconds=0))
    for name in ("reduction.json", "reduction-recheck.json"):
        closeout.write(root / name, dict(all_pass=True, state_count=8, states=states))
    closeout.write(root / "verification.json", dict(status="pass", reports=[
        dict(head_source_error=0, coordinate_source_error=0)]))
    (root / "launch").mkdir()
    closeout.write(root / "launch/scaleout-settled.json", {"jobs": []})
    (root / "launch/pilot.exit").write_text("0\n")
    (root / "coordination").mkdir()

    closeout.main()

    assert scan_output_root(root)["passed"]
    artifact_map = json.loads((root / "artifact-map.json").read_text())
    assert "# Reconstruction map" in artifact_map["summary"]
    assert "coordinate_margin/verify.py" in artifact_map["summary"]
    assert "artifact-map.json" in (unit / "results.md").read_text()
    terminal = json.loads((root / "integrated-terminal.json").read_text())
    assert closeout.bind(root / "artifact-map.json") in terminal["bindings"]

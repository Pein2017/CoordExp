import hashlib
import json
from pathlib import Path

import pytest

from probes.unmatched_judge.codetr_profile import runner


FIXTURE = Path(__file__).with_name("fixture-candidate.jsonl")


@pytest.fixture(autouse=True)
def skip_large_model_payload_hashes_in_cpu_wiring_tests(monkeypatch):
    # The real consumer verifies these bytes. Unit wiring tests still hash all
    # small code, configuration and prompt files, but do not reread20GB per case.
    real_sha = runner.sha
    monkeypatch.setattr(runner, "sha", lambda path: runner.FROZEN_HASHES[path]
                        if path in runner.FROZEN_HASHES and path.suffix in {".pth", ".safetensors"}
                        else real_sha(path))


@pytest.mark.parametrize("change", ["declared_image_hash", "relative_case_path", "absolute_case_path"])
def test_bad_image_identity_or_case_path_never_launches(tmp_path, monkeypatch, change):
    row = json.loads(FIXTURE.read_text())
    if change == "declared_image_hash":
        row["image_sha256"] = "0" * 64
    else:
        row["case_id"] = "../escape" if change == "relative_case_path" else "/tmp/escape"
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(runner, "_run_stage", lambda *args: pytest.fail("stage launched"))
    with pytest.raises(ValueError):
        runner.run(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_validate_rejects_duplicate_and_out_of_bounds(tmp_path):
    row = json.loads(FIXTURE.read_text())
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate case_id"):
        runner._validate(duplicate)
    row["bbox"][2] = row["width"] + 1
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="out of bounds"):
        runner._validate(invalid)


def test_distinct_case_ids_must_not_share_a_rendered_filename(tmp_path):
    row = json.loads(FIXTURE.read_text())
    source = tmp_path / "collision.jsonl"
    source.write_text(json.dumps(dict(row, case_id="sample:one")) + "\n"
                      + json.dumps(dict(row, case_id="sample_one")) + "\n")
    with pytest.raises(ValueError, match="filename collision"):
        runner._validate(source)


def test_collision_fails_before_any_stage(tmp_path, monkeypatch):
    out = tmp_path / "existing"
    out.mkdir()
    prior = b'{"status":"prior_success"}\n'
    (out / "run-receipt.json").write_bytes(prior)
    monkeypatch.setattr(runner, "_run_stage", lambda *args: pytest.fail("stage launched"))
    with pytest.raises(FileExistsError):
        runner.run(FIXTURE, out)
    assert (out / "run-receipt.json").read_bytes() == prior
    assert sorted(p.name for p in out.iterdir()) == ["run-receipt.json"]


def test_nonzero_stage_is_recorded_and_raised(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    def failed(*args, **kwargs):
        return type("Result", (), {"returncode": 17})()
    monkeypatch.setattr(runner.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="failed with exit 17"):
        runner._run_stage("detector", ["conda"], {}, out, [])


def test_source_identity_mismatch_fails_closed(tmp_path, monkeypatch):
    out = tmp_path / "run"
    calls = []
    def fake_stage(name, command, env, parent, receipts):
        calls.append(name)
        if name == "detector":
            stage = parent / "detector"
            stage.mkdir()
            (stage / "config.json").write_text(json.dumps({"source_sha256": "wrong", "code_sha256": "wrong"}))
    monkeypatch.setattr(runner, "_run_stage", fake_stage)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        runner.run(FIXTURE, out)
    assert calls == ["detector"]
    assert json.loads((out / "run-receipt.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("model_error", [None, "unresolved", "invalid_schema", "not_stopped"])
def test_sequence_uses_only_detector_accepts_and_joins_forms(tmp_path, monkeypatch, model_error):
    out = tmp_path / "run"
    source_hash = runner.sha(FIXTURE)
    stage_env = {}
    monkeypatch.setenv("CODETR_ADMIT_IOU", "0.01")
    def fake_stage(name, command, env, parent, receipts):
        stage_env[name] = (command, env)
        stage = parent / ("detector" if name == "detector" else "semantic")
        stage.mkdir()
        if name == "detector":
            decisions = [{"case_id": "237954:p11", "decision": "accept", "selected": {"score": .8}, "candidate_iou": .9}]
            config = {"source_sha256": source_hash, "code_sha256": runner.FROZEN_HASHES[runner.CODETR]}
            (stage / "decisions.json").write_text(json.dumps(decisions))
        else:
            semantic_source = parent / "semantic-eligible.jsonl"
            config = {"source_sha256": runner.sha(semantic_source), "code_sha256": runner.FROZEN_HASHES[runner.SEMANTIC]}
            rec = {"case_id": "237954:p11", "form": "full", "error": None, "finish_reason": "stop", "parsed": {"category": "dog"}}
            if model_error:
                rec.update(error=model_error, parsed=None)
                if model_error == "not_stopped":
                    rec["finish_reason"] = "length"
            rec2 = dict(rec, form="fixed_context")
            (stage / "responses.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(rec2) + "\n")
        (stage / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(runner, "_run_stage", fake_stage)
    result = runner.run(FIXTURE, out)
    assert set(result) == {"summary", "decisions"}
    assert stage_env["detector"][0][0:4] == ["conda", "run", "-n", "mmdet"]
    assert stage_env["detector"][1]["CUDA_VISIBLE_DEVICES"] == "0"
    assert float(stage_env["detector"][1]["CODETR_ADMIT_IOU"]) == .75
    assert stage_env["semantic"][0][0:4] == ["conda", "run", "-n", "ms"]
    assert stage_env["semantic"][1]["CUDA_VISIBLE_DEVICES"] == "1"
    assert stage_env["semantic"][1]["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert stage_env["semantic"][1]["HF_HUB_OFFLINE"] == "1"
    final = json.loads((out / "decisions.jsonl").read_text().splitlines()[0])
    assert final["decision"] == ("unknown" if model_error else "accept_candidate")
    assert json.loads((out / "summary.json").read_text())["detector_eligible"] == 1

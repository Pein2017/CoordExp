from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.research import materialize_natural_boundary_pre_gpu_evidence as evidence


def test_default_output_revision_is_v4() -> None:
    assert evidence.DEFAULT_OUTPUT_ROOT.name == "pre-gpu-evidence-v4"


def test_run_focused_tests_binds_exact_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = tmp_path / "test_first.py"
    second = tmp_path / "test_second.py"
    first.write_text("def test_first(): pass\n", encoding="utf-8")
    second.write_text("def test_second(): pass\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="2 passed in 0.01s\n", stderr="")

    monkeypatch.setattr(evidence.subprocess, "run", fake_run)
    receipt = evidence.run_focused_tests((first, second))
    assert receipt["status"] == "passed"
    assert receipt["passed_test_count"] == 2
    assert [item["path"] for item in receipt["focused_tests"]] == [
        str(first.resolve()),
        str(second.resolve()),
    ]
    assert captured["command"] == evidence._test_command((first.resolve(), second.resolve()))
    assert captured["kwargs"] == {
        "cwd": evidence.REPO_ROOT,
        "check": False,
        "capture_output": True,
        "text": True,
    }


def test_write_once_rejects_different_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    first = evidence._write_json_once(target, {"status": "passed"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "passed"}
    assert evidence._write_json_once(target, {"status": "passed"}) == first
    with pytest.raises(evidence.EvidenceError, match="different bytes"):
        evidence._write_json_once(target, {"status": "failed"})


def test_materialize_writes_passing_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        evidence,
        "run_focused_tests",
        lambda: {"status": "passed", "command": "pytest", "focused_tests": ["x"]},
    )
    monkeypatch.setattr(
        evidence.attention,
        "run_installed_qwen_cpu_probe",
        lambda: {"status": "passed", "block23_sdpa_mass_attestation": True},
    )
    result = evidence.materialize(
        test_receipt_path=tmp_path / "tests.json",
        mask_probe_path=tmp_path / "probe.json",
    )
    assert result["status"] == "passed"
    assert Path(result["focused_tests"]["path"]).is_file()
    assert Path(result["mask_probe"]["path"]).is_file()

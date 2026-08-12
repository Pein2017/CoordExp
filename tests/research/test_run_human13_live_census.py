from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    manifest = _write(tmp_path / "manifest.json", b'{"manifest":"sealed"}\n')
    source = tmp_path / "source"
    k_root = tmp_path / "k"
    for root, label in ((source, "source"), (k_root, "k")):
        _write(root / "receipt.json", (f'{{"mode":"{label}"}}\n').encode())
        _write(root / "trajectories.jsonl", (f'{{"trajectory":"{label}"}}\n').encode())
    config = _write(tmp_path / "a1.yaml", b"arm_id: A1\n")
    return {
        "manifest": manifest,
        "source": source,
        "k": k_root,
        "config": config,
        "census": tmp_path / "out" / "census.json",
        "receipt": tmp_path / "out" / "receipt.json",
    }


def _install_cpu_fakes(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str]]:
    from scripts.research import run_human13_live_census as cli

    calls: list[str] = []
    census_plan = SimpleNamespace(
        schema_version="human13_no_update_census_plan.v1",
        manifest_sha256=hashlib.sha256(b'{"manifest":"sealed"}\n').hexdigest(),
        discovery_binding_sha256="d" * 64,
        plan_sha256="p" * 64,
    )
    model_plan = SimpleNamespace(
        arm_id="A1",
        source=SimpleNamespace(
            checkpoint_path="/source/step-2444",
            base_model_path="/source/base",
            adapter_path="/source/adapter",
            special_embedding_path="/source/special",
            adapter_sha256="a" * 64,
            special_embedding_sha256="e" * 64,
        ),
        to_artifact_dict=lambda: {"arm_id": "A1"},
    )
    validation = SimpleNamespace(
        to_artifact_dict=lambda: {
            "arm_id": "A1",
            "adapter_tensor_sha256": "a" * 64,
        }
    )
    manifest = SimpleNamespace(images=(SimpleNamespace(image_id=1),))
    components = SimpleNamespace(tokenizer="tokenizer")
    skeletons = {1: "skeleton"}
    prepared = SimpleNamespace(packed_plan=SimpleNamespace(packs=("pack-0", "pack-1")))
    assembly = SimpleNamespace(
        components=components,
        model="packed-model",
        runtime=SimpleNamespace(optimizer_step_count=0),
        accelerator=SimpleNamespace(device="cpu"),
    )
    live_result = SimpleNamespace(
        evidence=("evidence",),
        receipt={
            "physical_pack_count": 2,
            "packed_forward_count": 2,
            "hf_forward_count": 1,
            "evidence_count": 3,
        },
    )
    canonical_census = {
        "schema_version": "human13_k_union_no_update_census.v1",
        "trie": {"original_row_count": 73},
    }

    monkeypatch.setattr(
        cli,
        "_load_census_plan",
        lambda **_: calls.append("plan") or census_plan,
    )
    monkeypatch.setattr(
        cli,
        "_build_model_plan",
        lambda _: calls.append("model_plan") or model_plan,
    )
    monkeypatch.setattr(
        cli,
        "_validate_model_plan",
        lambda _: calls.append("validation") or validation,
    )
    monkeypatch.setattr(
        cli,
        "_load_manifest",
        lambda _: calls.append("manifest") or manifest,
    )
    monkeypatch.setattr(
        cli,
        "_load_processor_components",
        lambda _: calls.append("processor") or components,
    )
    monkeypatch.setattr(
        cli,
        "_build_skeletons",
        lambda *_: calls.append("skeletons") or skeletons,
    )
    monkeypatch.setattr(
        cli,
        "_prepare_capture",
        lambda **_: calls.append("prepare") or prepared,
    )
    monkeypatch.setattr(
        cli,
        "_assemble_model",
        lambda *_, **__: calls.append("assembly") or assembly,
    )

    @contextmanager
    def hf_context(_: Path):
        calls.append("hf_open")
        yield "hf-scorer"
        calls.append("hf_close")

    monkeypatch.setattr(cli, "_open_hf_scorer", hf_context)
    monkeypatch.setattr(
        cli,
        "_capture",
        lambda **_: calls.append("capture") or live_result,
    )
    monkeypatch.setattr(
        cli,
        "_run_census",
        lambda **_: calls.append("census") or canonical_census,
    )
    monkeypatch.setattr(cli, "_peak_memory_bytes", lambda _: 1234)
    return SimpleNamespace(
        census_plan=census_plan,
        model_plan=model_plan,
        validation=validation,
        canonical_census=canonical_census,
    ), calls


def _kwargs(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "manifest_path": paths["manifest"],
        "source_root": paths["source"],
        "k_root": paths["k"],
        "a1_config_path": paths["config"],
        "repo_root": paths["manifest"].parent,
        "census_output": paths["census"],
        "receipt_output": paths["receipt"],
    }


def test_dry_run_validates_identities_without_crossing_live_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research import run_human13_live_census as cli

    paths = _fixture_paths(tmp_path)
    _fakes, calls = _install_cpu_fakes(monkeypatch)

    result = cli.execute_cli(**_kwargs(paths), execute=False, authority=False)

    assert result["mode"] == "dry_run"
    assert result["execution_ready"] is False
    assert result["actions"] == {
        "model_loads": 0,
        "forwards": 0,
        "backwards": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "artifact_writes": 0,
        "gpu_allocations": 0,
    }
    assert calls == ["plan", "model_plan", "validation"]
    assert not paths["census"].exists()
    assert not paths["receipt"].exists()


def test_execute_requires_explicit_model_gpu_authority_before_live_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research import run_human13_live_census as cli

    paths = _fixture_paths(tmp_path)
    _fakes, calls = _install_cpu_fakes(monkeypatch)

    with pytest.raises(cli.LiveCensusError, match="explicit model/GPU authority"):
        cli.execute_cli(**_kwargs(paths), execute=True, authority=False)

    assert calls == []


def test_execute_composes_source_census_and_publishes_bound_immutable_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research import run_human13_live_census as cli

    paths = _fixture_paths(tmp_path)
    fakes, calls = _install_cpu_fakes(monkeypatch)

    receipt = cli.execute_cli(**_kwargs(paths), execute=True, authority=True)

    assert calls == [
        "plan",
        "model_plan",
        "validation",
        "manifest",
        "processor",
        "skeletons",
        "prepare",
        "assembly",
        "hf_open",
        "capture",
        "hf_close",
        "census",
    ]
    assert json.loads(paths["census"].read_text()) == fakes.canonical_census
    assert json.loads(paths["receipt"].read_text()) == receipt
    assert receipt["schema_version"] == "human13_live_census_execution_receipt.v1"
    assert receipt["census"]["schema_version"] == (
        "human13_k_union_no_update_census.v1"
    )
    assert receipt["census"]["sha256"] == hashlib.sha256(
        paths["census"].read_bytes()
    ).hexdigest()
    assert receipt["census_plan"] == {
        "schema_version": "human13_no_update_census_plan.v1",
        "sha256": "p" * 64,
        "discovery_binding_sha256": "d" * 64,
    }
    assert receipt["manifest"]["sha256"] == fakes.census_plan.manifest_sha256
    assert receipt["source_discovery"]["receipt_sha256"] == hashlib.sha256(
        (paths["source"] / "receipt.json").read_bytes()
    ).hexdigest()
    assert receipt["k_discovery"]["records_sha256"] == hashlib.sha256(
        (paths["k"] / "trajectories.jsonl").read_bytes()
    ).hexdigest()
    assert receipt["runtime"]["packed_forward_count"] == 2
    assert receipt["runtime"]["hf_forward_count"] == 1
    assert receipt["runtime"]["total_forward_count"] == 3
    assert receipt["runtime"]["optimizer_steps"] == 0
    assert receipt["runtime"]["peak_memory_bytes"] == 1234
    assert receipt["runtime"]["elapsed_seconds"] >= 0.0

    with pytest.raises(cli.LiveCensusError, match="already exists"):
        cli.execute_cli(**_kwargs(paths), execute=True, authority=True)
    assert calls.count("processor") == 1


def test_execute_rejects_same_output_path_before_live_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research import run_human13_live_census as cli

    paths = _fixture_paths(tmp_path)
    _fakes, calls = _install_cpu_fakes(monkeypatch)
    kwargs = _kwargs(paths)
    kwargs["receipt_output"] = kwargs["census_output"]

    with pytest.raises(cli.LiveCensusError, match="distinct"):
        cli.execute_cli(**kwargs, execute=True, authority=True)

    assert calls == []


def test_direct_script_entry_bootstraps_repository_imports(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    script = Path(__file__).resolve().parents[2] / "scripts/research/run_human13_live_census.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(paths["manifest"]),
            "--source-root",
            str(paths["source"]),
            "--k-root",
            str(paths["k"]),
            "--a1-config",
            str(paths["config"]),
            "--census-output",
            str(paths["census"]),
            "--receipt-output",
            str(paths["receipt"]),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "No module named 'scripts.research'" not in completed.stderr

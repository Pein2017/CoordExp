"""CPU contract tests for the post-execution finalization successor sealer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research import finalize_s_k10_h20_crossover as finalizer
from scripts.research import seal_s_k10_h20_crossover_finalization_receipt as sealer


EVENT_IDS = finalizer.EVENT_IDS
EVENT_INDICES = finalizer.EVENT_INDICES
DEVICE_PLAN = finalizer.DEVICE_PLAN


def _write(path: Path, value: Any, *, field: str = "self_sha256") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {key: item for key, item in value.items() if key != field}
    document = dict(value)
    document[field] = finalizer.sha256_json(body)
    path.write_bytes(finalizer.canonical_json_bytes(document) + b"\n")
    return path


def _fixture(tmp_path: Path) -> dict[str, Any]:
    """A minimal but structurally exact parent/plan/execution set."""

    unit = tmp_path / "unit"
    execution_root = unit / "execution-v4"
    evidence_root = unit / "evidence-v4"

    authority = tmp_path / "authority.md"
    authority.write_text("# successor authority fixture\n")

    tools = tmp_path / "tools"
    tools.mkdir()
    finalizer_file = tools / "finalize_fixture.py"
    finalizer_file.write_text("# repaired finalizer fixture\n")
    finalizer_test_file = tools / "test_finalize_fixture.py"
    finalizer_test_file.write_text("# repaired finalizer test fixture\n")
    sealer_file = tools / "seal_fixture.py"
    sealer_file.write_text("# successor sealer fixture\n")
    sealer_test_file = tools / "test_seal_fixture.py"
    sealer_test_file.write_text("# successor sealer test fixture\n")

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    census = inputs / "census.json"
    _write(census, {"census": "fixture"})
    model_dir = inputs / "model_dir"
    (model_dir / "nested").mkdir(parents=True)
    (model_dir / "config.json").write_bytes(b'{"model": "fixture"}')
    (model_dir / "nested" / "weights.bin").write_bytes(b"weights-fixture")
    entries = [
        {
            "relative_path": relative,
            "sha256": finalizer.sha256_bytes((model_dir / relative).read_bytes()),
            "size_bytes": (model_dir / relative).stat().st_size,
        }
        for relative in ("config.json", "nested/weights.bin")
    ]
    runner_file = tools / "runner_fixture.py"
    runner_file.write_text("# runner fixture\n")

    plan_path = unit / "plan-v1" / "plan.json"
    plan = _write(
        plan_path,
        {
            "schema_version": finalizer.PLAN_SCHEMA_VERSION,
            "unit_id": finalizer.UNIT_ID,
            "status": "planned",
            "events": [
                {"event_id": event_id, "event_index": index, "image_id": 100 + position}
                for position, (event_id, index) in enumerate(zip(EVENT_IDS, EVENT_INDICES))
            ],
        },
    )
    plan_document = json.loads(plan.read_text())

    parent_body = {
        "schema_version": finalizer.PRE_GPU_SCHEMA_VERSION,
        "unit_id": finalizer.UNIT_ID,
        "status": "sealed_pre_gpu",
        "plan_self_sha256": plan_document["self_sha256"],
        "roots": {
            "execution_root": {"path": str(execution_root)},
            "final_root": {"path": str(evidence_root)},
        },
        "input_bindings": {
            "census": {
                "path": str(census),
                "sha256": finalizer.sha256_file(census),
                "size_bytes": census.stat().st_size,
                "kind": "file",
            },
            "base_model_dir": {
                "path": str(model_dir),
                "sha256": finalizer.sha256_json(entries),
                "size_bytes": sum(item["size_bytes"] for item in entries),
                "kind": "directory",
            },
        },
        "source_files": {
            "crossover_runner": {
                "path": str(runner_file),
                "sha256": finalizer.sha256_file(runner_file),
                "kind": "file",
            },
            "crossover_finalizer": {
                "path": str(finalizer_file),
                "sha256": "a" * 64,
                "kind": "file",
            },
        },
        "test_files": {
            "crossover_finalizer_test": {
                "path": str(finalizer_test_file),
                "sha256": "b" * 64,
                "kind": "file",
            },
        },
    }
    parent_path = unit / "pre-gpu-receipt-v5" / "pre-gpu-receipt.json"
    parent = _write(parent_path, parent_body)
    parent_document = json.loads(parent.read_text())

    for index, shard_id in enumerate(sealer.SHARD_IDS):
        root = execution_root / shard_id
        result_body = {
            "schema_version": finalizer.EVENT_SCHEMA_VERSION,
            "unit_id": finalizer.UNIT_ID,
            "status": "completed",
            "shard_id": shard_id,
            "event_id": EVENT_IDS[index],
            "event_index": EVENT_INDICES[index],
            "image_id": 100 + index,
            "plan_self_sha256": plan_document["self_sha256"],
            "pre_gpu_receipt_self_sha256": parent_document["self_sha256"],
        }
        _write(root / "result.json", result_body, field="result_sha256")
        _write(
            root / "runtime_identity.json",
            {
                "schema_version": finalizer.RUNTIME_SCHEMA_VERSION,
                "shard_id": shard_id,
                "device": {"physical_device": DEVICE_PLAN[shard_id], "logical_device": "cuda:0"},
            },
        )
        _write(root / "terminal_summary.json", {"shard_id": shard_id, "status": "completed"})
        _write(root / "aggregate.receipt.json", {"shard_id": shard_id, "event_id": EVENT_IDS[index]})

    return {
        "parent_receipt": parent,
        "plan": plan,
        "execution_root": execution_root,
        "evidence_root": evidence_root,
        "authority": authority,
        "finalizer_path": finalizer_file,
        "finalizer_test_path": finalizer_test_file,
        "sealer_path": sealer_file,
        "sealer_test_path": sealer_test_file,
        "census": census,
        "model_dir": model_dir,
        "runner_file": runner_file,
    }


def _selfed(document: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in document.items() if key != "self_sha256"}
    result = dict(body)
    result["self_sha256"] = finalizer.sha256_json(body)
    return result


def _rewrite_parent(fixture: dict[str, Any], update: Any, *, rebind_shards: bool = True) -> dict[str, Any]:
    """Rewrite the parent receipt and re-bind the shards that pin its self hash."""

    path = Path(fixture["parent_receipt"])
    parent = _selfed(json.loads(path.read_text()))
    update(parent)
    parent = _selfed(parent)
    path.write_bytes(finalizer.canonical_json_bytes(parent) + b"\n")
    if rebind_shards:
        for index, shard_id in enumerate(sealer.SHARD_IDS):
            result_path = fixture["execution_root"] / shard_id / "result.json"
            body = json.loads(result_path.read_text())
            body.pop("result_sha256", None)
            body["pre_gpu_receipt_self_sha256"] = parent["self_sha256"]
            _write(result_path, body, field="result_sha256")
    return parent


def _build(fixture: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    kwargs = {
        "parent_receipt": fixture["parent_receipt"],
        "plan": fixture["plan"],
        "execution_root": fixture["execution_root"],
        "evidence_root": fixture["evidence_root"],
        "authority": fixture["authority"],
        "finalizer_path": fixture["finalizer_path"],
        "finalizer_test_path": fixture["finalizer_test_path"],
        "sealer_path": fixture["sealer_path"],
        "sealer_test_path": fixture["sealer_test_path"],
    }
    kwargs.update(overrides)
    return sealer.build_finalization_receipt(**kwargs)


def test_sealed_receipt_binds_every_required_identity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt = _build(fixture)

    assert receipt["schema_version"] == sealer.SCHEMA_VERSION
    assert receipt["status"] == sealer.STATUS
    assert receipt["unit_id"] == finalizer.UNIT_ID
    assert receipt["authorization"]["cpu_only"] is True
    assert receipt["authorization"]["gpu_used"] is False
    assert receipt["authorization"]["model_loaded"] is False
    assert receipt["authorization"]["no_training"] is True
    assert receipt["authorization"]["endpoint_semantics_unchanged"] is True
    assert receipt["authorization"]["authority"]["sha256"] == finalizer.sha256_file(fixture["authority"])

    parent_document = json.loads(Path(fixture["parent_receipt"]).read_text())
    assert receipt["parent"]["receipt"]["raw_sha256"] == finalizer.sha256_file(fixture["parent_receipt"])
    assert receipt["parent"]["receipt"]["self_sha256"] == parent_document["self_sha256"]
    assert receipt["parent"]["document"] == parent_document
    assert receipt["parent"]["plan"]["raw_sha256"] == finalizer.sha256_file(fixture["plan"])

    assert [shard["shard_id"] for shard in receipt["execution"]["shards"]] == list(sealer.SHARD_IDS)
    assert [shard["event_id"] for shard in receipt["execution"]["shards"]] == list(EVENT_IDS)
    assert [shard["physical_device"] for shard in receipt["execution"]["shards"]] == ["0", "1", "7"]
    assert receipt["execution"]["root"]["status"] == "complete_immutable"
    for shard in receipt["execution"]["shards"]:
        assert set(shard["artifacts"]) == set(sealer.SHARD_ARTIFACTS)
        for name, ref in shard["artifacts"].items():
            assert ref["raw_sha256"] == finalizer.sha256_file(Path(shard["root"]) / name)

    assert receipt["evidence_root"]["state"] == "absent_at_seal"
    assert receipt["evidence_root"]["symlink"] is False
    assert not any(key in receipt["evidence_root"] for key in ("sha256", "raw_sha256", "self_sha256"))

    assert set(receipt["authorized_drift"]) == set(sealer.AUTHORIZED_SLOTS)
    finalizer_slot = receipt["authorized_drift"]["source_files.crossover_finalizer"]
    assert finalizer_slot["old_sha256"] == "a" * 64
    assert finalizer_slot["new_sha256"] == finalizer.sha256_file(fixture["finalizer_path"])
    assert finalizer_slot["cause"]
    test_slot = receipt["authorized_drift"]["test_files.crossover_finalizer_test"]
    assert test_slot["old_sha256"] == "b" * 64
    assert test_slot["new_sha256"] == finalizer.sha256_file(fixture["finalizer_test_path"])

    assert set(receipt["unchanged_parent_bindings"]["source_files"]) == {"crossover_runner"}
    assert receipt["unchanged_parent_bindings"]["test_files"] == {}
    assert set(receipt["unchanged_parent_bindings"]["input_bindings"]) == {"census", "base_model_dir"}
    assert set(receipt["finalization_tools"]) == set(sealer.FINALIZATION_TOOL_ROLES)
    assert receipt["runtime_input_pins"] == {
        "pre_gpu_receipt": receipt["parent"]["receipt"]["raw_sha256"],
        "pre_gpu_receipt_sha256": receipt["parent"]["receipt"]["raw_sha256"],
        "pre_gpu_receipt_self_sha256": receipt["parent"]["receipt"]["self_sha256"],
    }
    assert receipt["self_sha256"] == finalizer.document_self_sha256(receipt)


def test_seal_is_canonical_and_write_once(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / sealer.RECEIPT_DIR_NAME / sealer.RECEIPT_FILE_NAME
    first = sealer.seal(
        parent_receipt=fixture["parent_receipt"],
        plan=fixture["plan"],
        execution_root=fixture["execution_root"],
        evidence_root=fixture["evidence_root"],
        authority=fixture["authority"],
        output=output,
        finalizer_path=fixture["finalizer_path"],
        finalizer_test_path=fixture["finalizer_test_path"],
        sealer_path=fixture["sealer_path"],
        sealer_test_path=fixture["sealer_test_path"],
    )
    assert first["write"]["byte_identical"] is False
    payload = output.read_bytes()
    assert payload == finalizer.canonical_json_bytes(first["receipt"]) + b"\n"
    assert json.loads(payload)["self_sha256"] == first["self_sha256"]

    second = sealer.seal(
        parent_receipt=fixture["parent_receipt"],
        plan=fixture["plan"],
        execution_root=fixture["execution_root"],
        evidence_root=fixture["evidence_root"],
        authority=fixture["authority"],
        output=output,
        finalizer_path=fixture["finalizer_path"],
        finalizer_test_path=fixture["finalizer_test_path"],
        sealer_path=fixture["sealer_path"],
        sealer_test_path=fixture["sealer_test_path"],
    )
    assert second["write"]["byte_identical"] is True

    fixture["finalizer_path"].write_text("# a different repaired finalizer\n")
    with pytest.raises(FileExistsError, match=r"immutable finalization receipt collision"):
        sealer.seal(
            parent_receipt=fixture["parent_receipt"],
            plan=fixture["plan"],
            execution_root=fixture["execution_root"],
            evidence_root=fixture["evidence_root"],
            authority=fixture["authority"],
            output=output,
            finalizer_path=fixture["finalizer_path"],
            finalizer_test_path=fixture["finalizer_test_path"],
            sealer_path=fixture["sealer_path"],
            sealer_test_path=fixture["sealer_test_path"],
        )


@pytest.mark.parametrize("name", ["wrong_file", "wrong_directory"])
def test_seal_output_must_be_the_canonical_receipt_path(tmp_path: Path, name: str) -> None:
    fixture = _fixture(tmp_path)
    output = (
        tmp_path / sealer.RECEIPT_DIR_NAME / "receipt.json"
        if name == "wrong_file"
        else tmp_path / "receipts" / sealer.RECEIPT_FILE_NAME
    )
    with pytest.raises(sealer.FinalizationReceiptError, match=r"finalization-receipt-v1/finalization-receipt.json"):
        sealer.seal(
            parent_receipt=fixture["parent_receipt"],
            plan=fixture["plan"],
            execution_root=fixture["execution_root"],
            evidence_root=fixture["evidence_root"],
            authority=fixture["authority"],
            output=output,
            finalizer_path=fixture["finalizer_path"],
            finalizer_test_path=fixture["finalizer_test_path"],
            sealer_path=fixture["sealer_path"],
            sealer_test_path=fixture["sealer_test_path"],
        )


def test_seal_rejects_an_existing_or_symlinked_evidence_root(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["evidence_root"].mkdir(parents=True)
    with pytest.raises(sealer.FinalizationReceiptError, match=r"evidence root must be absent"):
        _build(fixture)
    fixture["evidence_root"].rmdir()
    fixture["evidence_root"].symlink_to(fixture["execution_root"], target_is_directory=True)
    with pytest.raises(sealer.FinalizationReceiptError, match=r"must not traverse a symlink"):
        _build(fixture)


def test_seal_rejects_a_no_op_slot(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    live = finalizer.sha256_file(fixture["finalizer_path"])
    _rewrite_parent(fixture, lambda doc: doc["source_files"]["crossover_finalizer"].__setitem__("sha256", live))
    with pytest.raises(sealer.FinalizationReceiptError, match=r"has not drifted"):
        _build(fixture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("parent_unit", r"parent receipt unit/status identity drifted"),
        ("parent_status", r"parent receipt unit/status identity drifted"),
        ("parent_self_tamper", r"parent pre-GPU receipt.self_sha256 mismatch"),
        ("wrong_plan", r"parent receipt binds a different plan"),
        ("execution_root_mismatch", r"execution root differs from the parent-sealed"),
        ("evidence_root_mismatch", r"evidence root differs from the parent-sealed"),
        ("foreign_shard_file", r"partial, duplicated, or carries a foreign file"),
        ("missing_shard", r"not exactly the three sealed shard roots"),
        ("shard_result_event", r"does not bind its planned event/parent identity"),
        ("shard_result_parent", r"does not bind its planned event/parent identity"),
        ("shard_device", r"physical device differs from the sealed device plan"),
        ("shard_result_self_tamper", r"result_sha256 mismatch"),
        ("unchanged_source_drift", r"parent source_files.crossover_runner raw SHA-256 drifted"),
        ("unchanged_input_drift", r"parent input_bindings.census raw SHA-256 drifted"),
        ("unchanged_directory_drift", r"parent input_bindings.base_model_dir raw SHA-256 drifted"),
        ("foreign_slot_path", r"live path differs from the finalization tool path"),
        ("missing_authority", r"successor authority document must be an existing regular non-symlink file"),
        ("missing_sealer_test", r"finalization tool successor_sealer_test"),
        ("symlink_execution_root", r"must not traverse a symlink"),
        ("nonregular_parent", r"must not traverse a symlink"),
    ],
)
def test_seal_fails_closed(tmp_path: Path, mutation: str, message: str) -> None:
    fixture = _fixture(tmp_path)
    overrides: dict[str, Any] = {}
    parent_path = Path(fixture["parent_receipt"])

    def rewrite_parent(update: Any) -> None:
        _rewrite_parent(fixture, update)

    if mutation == "parent_unit":
        rewrite_parent(lambda doc: doc.__setitem__("unit_id", "other-unit"))
    elif mutation == "parent_status":
        rewrite_parent(lambda doc: doc.__setitem__("status", "draft"))
    elif mutation == "parent_self_tamper":
        tampered = json.loads(parent_path.read_text())
        tampered["self_sha256"] = "c" * 64
        parent_path.write_bytes(finalizer.canonical_json_bytes(tampered) + b"\n")
    elif mutation == "wrong_plan":
        other = _write(tmp_path / "other-plan.json", {"schema_version": finalizer.PLAN_SCHEMA_VERSION, "unit_id": finalizer.UNIT_ID, "events": []})
        overrides["plan"] = other
    elif mutation == "execution_root_mismatch":
        other = tmp_path / "unit" / "execution-v3"
        (other / "shard-000").mkdir(parents=True)
        overrides["execution_root"] = other
    elif mutation == "evidence_root_mismatch":
        overrides["evidence_root"] = tmp_path / "unit" / "evidence-v5"
    elif mutation == "foreign_shard_file":
        (fixture["execution_root"] / "shard-000" / "extra.json").write_bytes(b"{}\n")
    elif mutation == "missing_shard":
        for child in (fixture["execution_root"] / "shard-002").iterdir():
            child.unlink()
        (fixture["execution_root"] / "shard-002").rmdir()
    elif mutation == "shard_result_event":
        path = fixture["execution_root"] / "shard-000" / "result.json"
        body = json.loads(path.read_text())
        body["image_id"] = 999
        _write(path, {k: v for k, v in body.items() if k != "result_sha256"}, field="result_sha256")
    elif mutation == "shard_result_parent":
        path = fixture["execution_root"] / "shard-001" / "result.json"
        body = json.loads(path.read_text())
        body["pre_gpu_receipt_self_sha256"] = "d" * 64
        _write(path, {k: v for k, v in body.items() if k != "result_sha256"}, field="result_sha256")
    elif mutation == "shard_device":
        path = fixture["execution_root"] / "shard-002" / "runtime_identity.json"
        body = json.loads(path.read_text())
        body["device"]["physical_device"] = "3"
        _write(path, {k: v for k, v in body.items() if k != "self_sha256"})
    elif mutation == "shard_result_self_tamper":
        path = fixture["execution_root"] / "shard-000" / "result.json"
        body = json.loads(path.read_text())
        body["result_sha256"] = "e" * 64
        path.write_bytes(finalizer.canonical_json_bytes(body) + b"\n")
    elif mutation == "unchanged_source_drift":
        fixture["runner_file"].write_text("# drifted runner\n")
    elif mutation == "unchanged_input_drift":
        fixture["census"].write_bytes(finalizer.canonical_json_bytes({"census": "drifted"}) + b"\n")
    elif mutation == "unchanged_directory_drift":
        (fixture["model_dir"] / "extra.bin").write_bytes(b"extra")
    elif mutation == "foreign_slot_path":
        foreign = tmp_path / "tools" / "foreign_finalizer.py"
        foreign.write_text("# foreign\n")
        overrides["finalizer_path"] = foreign
    elif mutation == "missing_authority":
        overrides["authority"] = tmp_path / "absent-authority.md"
    elif mutation == "missing_sealer_test":
        overrides["sealer_test_path"] = tmp_path / "tools" / "absent_test.py"
    elif mutation == "symlink_execution_root":
        link = tmp_path / "linked-execution"
        link.symlink_to(fixture["execution_root"], target_is_directory=True)
        overrides["execution_root"] = link
    else:
        link = tmp_path / "linked-parent.json"
        link.symlink_to(parent_path)
        overrides["parent_receipt"] = link

    with pytest.raises(sealer.FinalizationReceiptError, match=message):
        _build(fixture, **overrides)


def _repoint_to_real_tools(fixture: dict[str, Any]) -> None:
    real_test = Path(finalizer.__file__).resolve().parents[2] / "tests" / "research" / "test_finalize_s_k10_h20_crossover.py"

    def repoint(doc: dict[str, Any]) -> None:
        doc["source_files"]["crossover_finalizer"]["path"] = str(Path(finalizer.__file__).resolve())
        doc["test_files"]["crossover_finalizer_test"]["path"] = str(real_test)

    _rewrite_parent(fixture, repoint)


def test_cli_dry_run_reports_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _fixture(tmp_path)
    _repoint_to_real_tools(fixture)
    output = tmp_path / sealer.RECEIPT_DIR_NAME / sealer.RECEIPT_FILE_NAME
    code = sealer.main(
        [
            "--parent-receipt", str(fixture["parent_receipt"]),
            "--plan", str(fixture["plan"]),
            "--execution-root", str(fixture["execution_root"]),
            "--evidence-root", str(fixture["evidence_root"]),
            "--authority", str(fixture["authority"]),
            "--output", str(output),
            "--dry-run",
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema_version"] == sealer.SCHEMA_VERSION
    assert summary["status"] == sealer.STATUS
    assert summary["authorized_drift"] == sorted(sealer.AUTHORIZED_SLOTS)
    assert summary["write"] is None
    assert not output.exists()


def test_cli_requires_an_output_unless_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _fixture(tmp_path)
    code = sealer.main(
        [
            "--parent-receipt", str(fixture["parent_receipt"]),
            "--plan", str(fixture["plan"]),
            "--execution-root", str(fixture["execution_root"]),
            "--evidence-root", str(fixture["evidence_root"]),
            "--authority", str(fixture["authority"]),
        ]
    )
    assert code == 2
    assert "--output is required" in capsys.readouterr().err


def test_cli_blocks_on_a_drifted_boundary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _fixture(tmp_path)
    _repoint_to_real_tools(fixture)
    fixture["evidence_root"].mkdir(parents=True)
    code = sealer.main(
        [
            "--parent-receipt", str(fixture["parent_receipt"]),
            "--plan", str(fixture["plan"]),
            "--execution-root", str(fixture["execution_root"]),
            "--evidence-root", str(fixture["evidence_root"]),
            "--authority", str(fixture["authority"]),
            "--dry-run",
        ]
    )
    assert code == 2
    assert "evidence root must be absent" in capsys.readouterr().err


def test_real_repository_tools_are_bound_by_default(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    parent_path = Path(fixture["parent_receipt"])
    _repoint_to_real_tools(fixture)

    receipt = sealer.build_finalization_receipt(
        parent_receipt=parent_path,
        plan=fixture["plan"],
        execution_root=fixture["execution_root"],
        evidence_root=fixture["evidence_root"],
        authority=fixture["authority"],
    )
    tools = receipt["finalization_tools"]
    assert tools["finalizer"]["path"] == str(Path(finalizer.__file__).resolve())
    assert tools["successor_sealer"]["path"] == str(Path(sealer.__file__).resolve())
    assert tools["successor_sealer"]["sha256"] == finalizer.sha256_file(Path(sealer.__file__).resolve())
    assert receipt["authorized_drift"]["source_files.crossover_finalizer"]["new_sha256"] == tools["finalizer"]["sha256"]


def test_shard_result_must_match_the_rewritten_parent_self_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _rewrite_parent(fixture, lambda doc: doc.__setitem__("no_training", True), rebind_shards=False)
    # The shards still bind the previous parent self hash.
    with pytest.raises(sealer.FinalizationReceiptError, match=r"does not bind its planned event/parent identity"):
        _build(fixture)


def test_copy_of_parent_is_never_written(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before = finalizer.sha256_file(fixture["parent_receipt"])
    receipt = _build(fixture)
    assert finalizer.sha256_file(fixture["parent_receipt"]) == before
    assert receipt["parent"]["receipt"]["raw_sha256"] == before
    assert copy.deepcopy(receipt["parent"]["document"]) == json.loads(Path(fixture["parent_receipt"]).read_text())

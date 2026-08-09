from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research import seal_natural_boundary_routing_history_contract as sealer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _real_kwargs() -> dict[str, object]:
    return {
        "unit_path": sealer.UNIT_PATH,
        "tasks_path": sealer.TASKS_PATH,
        "launch_gate_path": sealer.LAUNCH_GATE_PATH,
        "prior_output_root": sealer.PRIOR_OUTPUT_ROOT,
        "sealer_path": Path(sealer.__file__).resolve(),
        "tests_path": Path(__file__).resolve(),
    }


def test_build_contract_binds_boundaries_seven_upstreams_and_source_identities() -> None:
    document = sealer.build_contract(**_real_kwargs())

    assert document["unit_id"] == sealer.UNIT_ID
    assert document["boundary_contract"] == sealer.BOUNDARY_CONTRACT
    assert document["boundary_contract"]["primary"]["checkpoint"] == "S"
    assert document["boundary_contract"]["secondary"]["checkpoint"] == "A3"
    assert document["boundary_contract"]["training"]["authorized"] is False
    assert len(document["upstream_artifacts"]) == 7
    assert all(
        item["declared_sha256"] == item["observed_sha256"]
        for item in document["upstream_artifacts"]
    )
    assert document["contract_files"]["unit"]["sha256"] == _sha256(sealer.UNIT_PATH)
    assert document["source_files"]["sealer"]["sha256"] == _sha256(Path(sealer.__file__).resolve())
    assert document["source_files"]["focused_tests"]["sha256"] == _sha256(Path(__file__).resolve())
    assert document["self_sha256"] == sealer.document_self_sha256(document)
    sealer.validate_contract(document)


def test_caller_review_dispositions_are_bound_without_claiming_fixed_defaults() -> None:
    reviews = {
        "contract_audit": {
            "status": "passed",
            "reviewer": "test-fixture",
            "receipt_sha256": "a" * 64,
        }
    }
    document = sealer.build_contract(**_real_kwargs(), review_dispositions=reviews)
    assert document["review_disposition_source"] == "caller_supplied"
    assert document["review_dispositions"] == reviews
    sealer.validate_contract(document)


def test_upstream_hash_mutation_fails_closed(tmp_path: Path) -> None:
    unit_copy = tmp_path / "unit.md"
    original = sealer.UNIT_PATH.read_text(encoding="utf-8")
    # Keep the table shape but point one declared artifact at bytes absent from
    # the prior immutable output root.
    mutated = original.replace(
        "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23",
        "0" * 64,
        1,
    )
    unit_copy.write_text(mutated, encoding="utf-8")
    with pytest.raises(sealer.ContractSealError, match="must resolve to exactly one file"):
        sealer.build_contract(
            **{
                **_real_kwargs(),
                "unit_path": unit_copy,
            }
        )


def test_boundary_mutation_fails_closed(tmp_path: Path) -> None:
    unit_copy = tmp_path / "unit.md"
    original = sealer.UNIT_PATH.read_text(encoding="utf-8")
    unit_copy.write_text(
        original.replace(
            "unit_id: 2026-08-06-natural-boundary-routing-history-replication",
            "unit_id: wrong-unit",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(sealer.ContractSealError, match="unit_id mismatch"):
        sealer.build_contract(**{**_real_kwargs(), "unit_path": unit_copy})


def test_task_and_launch_gate_snapshot_drift_is_tolerated_but_unit_drift_is_not(
    tmp_path: Path,
) -> None:
    unit_copy = tmp_path / "unit.md"
    tasks_copy = tmp_path / "tasks.md"
    launch_gate_copy = tmp_path / "launch-gate.md"
    unit_copy.write_bytes(sealer.UNIT_PATH.read_bytes())
    tasks_copy.write_bytes(sealer.TASKS_PATH.read_bytes())
    launch_gate_copy.write_bytes(sealer.LAUNCH_GATE_PATH.read_bytes())
    document = sealer.build_contract(
        **{
            **_real_kwargs(),
            "unit_path": unit_copy,
            "tasks_path": tasks_copy,
            "launch_gate_path": launch_gate_copy,
        }
    )

    tasks_copy.write_bytes(tasks_copy.read_bytes() + b"\n- [x] follow-up snapshot item\n")
    launch_gate_copy.write_bytes(launch_gate_copy.read_bytes() + b"\nSnapshot advanced.\n")
    sealer.validate_contract(document)

    unit_copy.write_bytes(unit_copy.read_bytes() + b"\nContract drift is invalid.\n")
    with pytest.raises(sealer.ContractSealError, match="contract_files.unit bytes differ"):
        sealer.validate_contract(document)


def test_immutable_output_accepts_identical_bytes_and_rejects_drift(tmp_path: Path) -> None:
    output = tmp_path / "contract.json"
    kwargs = _real_kwargs()
    first = sealer.seal(output, **kwargs)
    first_bytes = output.read_bytes()
    payload = json.loads(first_bytes)
    assert payload["self_sha256"] == sealer.document_self_sha256(payload)
    assert first["byte_identical"] is False

    second = sealer.seal(output, **kwargs)
    assert second["byte_identical"] is True
    assert output.read_bytes() == first_bytes

    output.write_bytes(first_bytes + b"\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sealer.seal(output, **kwargs)


def test_cli_accepts_explicit_paths_and_emits_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "explicit-contract.json"
    argv = [
        "--unit",
        str(sealer.UNIT_PATH),
        "--tasks",
        str(sealer.TASKS_PATH),
        "--launch-gate",
        str(sealer.LAUNCH_GATE_PATH),
        "--prior-output-root",
        str(sealer.PRIOR_OUTPUT_ROOT),
        "--sealer",
        str(Path(sealer.__file__).resolve()),
        "--tests",
        str(Path(__file__).resolve()),
        "--output",
        str(output),
    ]
    assert sealer.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["path"] == str(output.resolve())
    assert output.is_file()

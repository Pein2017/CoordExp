from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from public_data.provenance import (
    ProvenanceError,
    validate_manifest,
    validate_refinement_authority,
)


def _manifest(repo: Path, *, content: bytes, materialized: bool = True) -> Path:
    schema_source = Path("manifests/public_data_provenance/schema.json")
    schema_target = repo / schema_source
    schema_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(schema_source, schema_target)
    producer = repo / "public_data/scripts/build.py"
    producer.parent.mkdir(parents=True, exist_ok=True)
    producer.write_text("# fixture\n", encoding="utf-8")
    dependency = repo / "public_data/geometry.py"
    dependency.write_text("# fixture\n", encoding="utf-8")
    relative = "public_data/coco/fixture"
    path = repo / relative / "train.jsonl"
    if materialized:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    records = sum(1 for line in content.splitlines() if line.strip())
    aggregate_line = f"{relative}/train.jsonl {digest} {len(content)} {records}\n"
    payload = {
        "schema_version": 1,
        "artifact_type": "processed_directory",
        "relative_path": relative,
        "producer_script": "public_data/scripts/build.py",
        "regeneration_status": "ready",
        "validator_module": "public_data.provenance",
        "working_dir": ".",
        "command": "python -m public_data.scripts.build --output public_data/coco/fixture",
        "dependencies": [{"kind": "coordinate_contract", "path": "public_data/geometry.py", "required": True}],
        "inputs": [{"kind": "raw_dataset", "path": "public_data/coco/raw", "notes": "optional fixture input"}],
        "key_params": {"dataset": "coco"},
        "checksums": {
            "scope": "jsonl_training_samples_only", "algorithm": "sha256",
            "aggregate_sha256": hashlib.sha256(aggregate_line.encode()).hexdigest(),
            "aggregate_source": "sorted path sha256 size_bytes records lines",
            "files": [{"path": f"{relative}/train.jsonl", "sha256": digest, "size_bytes": len(content), "records": records}],
        },
        "code_ref": None, "generated_at_utc": None, "notes": "fixture",
    }
    manifest = repo / "manifests/public_data_provenance/coco/fixture.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_manifest_detects_wrong_jsonl_content(tmp_path: Path) -> None:
    content = b'{"images":["one.jpg"],"objects":[]}\n'
    manifest = _manifest(tmp_path, content=content)
    assert validate_manifest(manifest, repo_root=tmp_path).materialization == "present_validated"
    (tmp_path / "public_data/coco/fixture/train.jsonl").write_bytes(content + b"{}\n")
    with pytest.raises(ProvenanceError, match="content mismatch"):
        validate_manifest(manifest, repo_root=tmp_path)


def test_absent_materialization_is_reported_as_absent(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, content=b"{}\n", materialized=False)
    result = validate_manifest(manifest, repo_root=tmp_path)
    assert result.materialization == "absent"
    assert result.checked_files == 0


def test_manifest_on_hold_cannot_report_validation_success(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, content=b"{}\n")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["regeneration_status"] = "hold_unresolved_source_parity"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="regeneration closure is on hold"):
        validate_manifest(manifest, repo_root=tmp_path)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _journal_record(payload: dict[str, object], previous: str | None) -> dict[str, object]:
    record = {**payload, "prev_record_hash": previous}
    record["record_hash"] = _canonical_sha256(record)
    return record


def _write_refinement_authority(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    runtime_root = tmp_path / "runtime"
    split_root = runtime_root / "train"
    output_root = tmp_path / "public_data/coco/refined"
    split_root.mkdir(parents=True)
    output_root.mkdir(parents=True)

    working = split_root / "working.norm.jsonl"
    norm = output_root / "train.norm.jsonl"
    coord = output_root / "train.coord.jsonl"
    task_index = split_root / "task_index.json"
    config = tmp_path / "config.yaml"
    tokenizer = tmp_path / "tokenizer.json"
    row = {
        "image_id": 9,
        "images": ["../base/images/train2017/000000000009.jpg"],
        "objects": [
            {
                "bbox_2d": [1, 2, 3, 4],
                "category_id": 55,
                "category_name": "orange",
                "coco_ann_id": -1,
                "desc": "orange",
            }
        ],
    }
    working.write_text(json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8")
    norm.write_text(json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8")
    coord_row = json.loads(json.dumps(row))
    coord_row["objects"][0]["bbox_2d"] = [
        "<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>",
    ]
    coord.write_text(json.dumps(coord_row, separators=(",", ":")) + "\n", encoding="utf-8")
    task_index.write_text("{}\n", encoding="utf-8")
    config.write_text("packing:\n  global_max_length: 12000\n", encoding="utf-8")
    tokenizer.write_text("{}\n", encoding="utf-8")

    working_sha = hashlib.sha256(working.read_bytes()).hexdigest()
    task_index_sha = hashlib.sha256(task_index.read_bytes()).hexdigest()
    project_payload = {
        "schema_version": 2,
        "split": "train",
        "generation": 1,
        "task_count": 1,
        "working_line_count": 1,
        "working_sha256": working_sha,
        "task_index_sha256": task_index_sha,
    }
    project = split_root / "project.json"
    project.write_text(json.dumps(project_payload, separators=(",", ":")), encoding="utf-8")
    canonical_manifest_sha = _canonical_sha256(project_payload)

    reservation = _journal_record(
        {
            "kind": "reservation",
            "split": "train",
            "batch_id": "fixture",
            "image_id": 9,
            "coco_ann_id": -1,
        },
        None,
    )
    prepared = _journal_record(
        {
            "kind": "batch_prepared",
            "batch_id": "fixture",
            "candidate_generation": 1,
            "candidate_working_sha256": working_sha,
            "candidate_manifest": project_payload,
            "candidate_manifest_hash": canonical_manifest_sha,
        },
        str(reservation["record_hash"]),
    )
    terminal = _journal_record(
        {
            "kind": "batch_terminal",
            "batch_id": "fixture",
            "generation": 1,
            "status": "succeeded",
            "working_sha256": working_sha,
            "prepared_record_hash": prepared["record_hash"],
        },
        str(prepared["record_hash"]),
    )
    journal = split_root / "journal.jsonl"
    journal.write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in (reservation, prepared, terminal)),
        encoding="utf-8",
    )

    receipt = split_root / "training.publish.receipt.json"
    receipt_payload = {
        "schema_version": 2,
        "code": "coco_refinement.committed_generation_published",
        "publisher_version": "coco-refinement-dataset-publisher-v1",
        "split": "train",
        "generation": 1,
        "published_at_utc": "2026-07-20T08:25:45Z",
        "runtime_root": str(runtime_root),
        "working": {"path": str(working), "sha256": working_sha},
        "outputs": {
            "norm": {
                "path": str(norm),
                "sha256": hashlib.sha256(norm.read_bytes()).hexdigest(),
                "size_bytes": norm.stat().st_size,
            },
            "coord": {
                "path": str(coord),
                "sha256": hashlib.sha256(coord.read_bytes()).hexdigest(),
                "size_bytes": coord.stat().st_size,
            },
        },
        "row_count": 1,
        "object_count": 1,
        "identity_authority": {
            "journal_path": str(journal),
            "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
            "negative_object_count": 1,
            "status": "passed",
        },
        "loader_attestation": {"coord_row_count": 1, "status": "passed"},
        "norm_schema_attestation": {"row_count": 1, "status": "passed"},
        "token_budget": {
            "row_count": 1,
            "max_total_tokens": 12000,
            "training_config_path": str(config),
            "training_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            "status": "passed",
        },
    }
    receipt.write_text(json.dumps(receipt_payload, separators=(",", ":")), encoding="utf-8")

    files = {
        "receipt": receipt,
        "project": project,
        "journal": journal,
        "working": working,
        "task_index": task_index,
        "norm": norm,
        "coord": coord,
        "config": config,
        "tokenizer": tokenizer,
    }
    authority: dict[str, object] = {
        "historical_producer": {
            "git_commit": "93517eadf6679cad0a0e87288df4097ed95e066e",
            "publisher_version": "coco-refinement-dataset-publisher-v1",
        },
        "runtime_root": str(runtime_root),
        "historical_training_config": {
            "identity_mode": "recorded_receipt_only",
            "recorded_path": str(config),
            "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        },
        "tokenizer": {"path": str(tokenizer), "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest()},
        "splits": {
            "train": {
                "generation": 1,
                "published_at_utc": "2026-07-20T08:25:45Z",
                "row_count": 1,
                "object_count": 1,
                "negative_object_count": 1,
                "canonical_manifest_sha256": canonical_manifest_sha,
                "terminal_record_hash": terminal["record_hash"],
                "files": {
                    name: {"path": str(files[name]), "sha256": hashlib.sha256(files[name].read_bytes()).hexdigest()}
                    for name in ("receipt", "project", "journal", "working", "task_index")
                },
                "outputs": {
                    name: {
                        "path": str(files[name]),
                        "sha256": hashlib.sha256(files[name].read_bytes()).hexdigest(),
                        "size_bytes": files[name].stat().st_size,
                        "records": 1,
                    }
                    for name in ("norm", "coord")
                },
            }
        },
    }
    return authority, files


def _refresh_authority_file(authority: dict[str, object], name: str, path: Path) -> None:
    split = authority["splits"]["train"]
    split["files"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_refinement_authority_accepts_terminal_fixture(tmp_path: Path) -> None:
    authority, files = _write_refinement_authority(tmp_path)
    files["config"].unlink()
    validate_refinement_authority(authority)


def test_refinement_authority_rejects_recorded_config_path_mismatch(tmp_path: Path) -> None:
    authority, _ = _write_refinement_authority(tmp_path)
    authority["historical_training_config"]["recorded_path"] = str(tmp_path / "other-config.yaml")
    with pytest.raises(ProvenanceError, match="token/config authority mismatch"):
        validate_refinement_authority(authority)


def test_refinement_schema_requires_recorded_config_identity_mode() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repo_root / "manifests/public_data_provenance/schema.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            repo_root
            / "manifests/public_data_provenance/coco/rescale_32_1024_bbox_len12000.json"
        ).read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    assert list(validator.iter_errors(manifest)) == []

    del manifest["refinement_authority"]["historical_training_config"]["identity_mode"]
    errors = list(validator.iter_errors(manifest))
    assert any("'identity_mode' is a required property" in error.message for error in errors)


def test_refinement_authority_rejects_wrong_receipt_hash(tmp_path: Path) -> None:
    authority, _ = _write_refinement_authority(tmp_path)
    authority["splits"]["train"]["files"]["receipt"]["sha256"] = "0" * 64
    with pytest.raises(ProvenanceError, match="receipt.*hash mismatch"):
        validate_refinement_authority(authority)


def test_refinement_authority_rejects_nonterminal_receipt(tmp_path: Path) -> None:
    authority, files = _write_refinement_authority(tmp_path)
    payload = json.loads(files["receipt"].read_text(encoding="utf-8"))
    payload["code"] = "coco_refinement.publication_incomplete"
    files["receipt"].write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    _refresh_authority_file(authority, "receipt", files["receipt"])
    with pytest.raises(ProvenanceError, match="terminal publication"):
        validate_refinement_authority(authority)


def test_refinement_authority_rejects_working_hash_mismatch(tmp_path: Path) -> None:
    authority, files = _write_refinement_authority(tmp_path)
    files["working"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="working.*hash mismatch"):
        validate_refinement_authority(authority)


def test_refinement_authority_rejects_broken_journal_chain(tmp_path: Path) -> None:
    authority, files = _write_refinement_authority(tmp_path)
    records = [json.loads(line) for line in files["journal"].read_text(encoding="utf-8").splitlines()]
    records[1]["prev_record_hash"] = "0" * 64
    files["journal"].write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in records),
        encoding="utf-8",
    )
    receipt = json.loads(files["receipt"].read_text(encoding="utf-8"))
    receipt["identity_authority"]["journal_sha256"] = hashlib.sha256(files["journal"].read_bytes()).hexdigest()
    files["receipt"].write_text(json.dumps(receipt, separators=(",", ":")), encoding="utf-8")
    _refresh_authority_file(authority, "journal", files["journal"])
    _refresh_authority_file(authority, "receipt", files["receipt"])
    with pytest.raises(ProvenanceError, match="broken journal chain"):
        validate_refinement_authority(authority)


def test_refinement_authority_rejects_unmatched_negative_ownership(tmp_path: Path) -> None:
    authority, files = _write_refinement_authority(tmp_path)
    records = [json.loads(line) for line in files["journal"].read_text(encoding="utf-8").splitlines()][1:]
    previous = None
    rebuilt = []
    for record in records:
        record.pop("record_hash")
        if record["kind"] == "batch_terminal":
            record["prepared_record_hash"] = rebuilt[-1]["record_hash"]
        rebuilt_record = _journal_record(record, previous)
        rebuilt.append(rebuilt_record)
        previous = str(rebuilt_record["record_hash"])
    files["journal"].write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in rebuilt),
        encoding="utf-8",
    )
    receipt = json.loads(files["receipt"].read_text(encoding="utf-8"))
    receipt["identity_authority"]["journal_sha256"] = hashlib.sha256(files["journal"].read_bytes()).hexdigest()
    files["receipt"].write_text(json.dumps(receipt, separators=(",", ":")), encoding="utf-8")
    _refresh_authority_file(authority, "journal", files["journal"])
    _refresh_authority_file(authority, "receipt", files["receipt"])
    authority["splits"]["train"]["terminal_record_hash"] = rebuilt[-1]["record_hash"]
    with pytest.raises(ProvenanceError, match="negative-ID ownership"):
        validate_refinement_authority(authority)


def test_refinement_authority_rejects_output_hash_mismatch(tmp_path: Path) -> None:
    authority, files = _write_refinement_authority(tmp_path)
    files["coord"].write_bytes(files["coord"].read_bytes() + b"{}\n")
    with pytest.raises(ProvenanceError, match="coord.*hash mismatch"):
        validate_refinement_authority(authority)

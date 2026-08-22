"""Validate tracked public-data provenance manifests and local JSONL content."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jsonschema


class ProvenanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidationResult:
    manifest: Path
    materialization: str
    checked_files: int


_TERMINAL_PUBLICATION_CODE = "coco_refinement.committed_generation_published"
_REFINEMENT_PUBLISHER_VERSION = "coco-refinement-dataset-publisher-v1"


def validate_manifest(
    manifest_path: Path,
    *,
    repo_root: Path,
    data_root: Path | None = None,
) -> ValidationResult:
    manifest_path = Path(manifest_path)
    repo_root = Path(repo_root).resolve()
    data_root = Path(data_root).resolve() if data_root is not None else repo_root
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_path = repo_root / "manifests/public_data_provenance/schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(payload), key=lambda err: list(err.path))
    if errors:
        raise ProvenanceError(f"schema validation failed: {errors[0].message}")
    _validate_executable_closure(payload, repo_root=repo_root)
    if "refinement_authority" in payload:
        validate_refinement_authority(payload["refinement_authority"])
    artifact_root = data_root / payload["relative_path"]
    if not artifact_root.exists():
        return ValidationResult(manifest_path, "absent", 0)
    checksums = payload.get("checksums")
    if checksums is None:
        return ValidationResult(manifest_path, "present_unhashed_image_store", 0)
    declared = {entry["path"] for entry in checksums["files"]}
    excluded = {entry["path"] for entry in payload.get("excluded_local_jsonl", [])}
    if declared & excluded:
        raise ProvenanceError("a JSONL path cannot be both checksummed and excluded")
    actual = {
        path.relative_to(data_root).as_posix()
        for path in artifact_root.glob("*.jsonl")
        if path.is_file()
    }
    if actual != declared | excluded:
        raise ProvenanceError(
            "top-level JSONL set mismatch: "
            f"declared={sorted(declared)} excluded={sorted(excluded)} actual={sorted(actual)}"
        )
    aggregate_rows: list[str] = []
    for entry in checksums["files"]:
        path = data_root / entry["path"]
        if not path.is_file():
            raise ProvenanceError(f"declared JSONL is missing: {entry['path']}")
        digest = _sha256(path)
        size = path.stat().st_size
        records = _count_records(path)
        if digest != entry["sha256"] or size != entry["size_bytes"] or records != entry["records"]:
            raise ProvenanceError(f"JSONL content mismatch: {entry['path']}")
        aggregate_rows.append(f"{entry['path']} {digest} {size} {records}\n")
    aggregate = hashlib.sha256("".join(sorted(aggregate_rows)).encode()).hexdigest()
    if aggregate != checksums["aggregate_sha256"]:
        raise ProvenanceError("aggregate JSONL checksum mismatch")
    for section in (payload.get("metadata") or {}).values():
        values = section.values() if isinstance(section, dict) and "path" not in section else [section]
        for entry in values:
            if not isinstance(entry, dict) or "path" not in entry:
                continue
            sidecar = data_root / entry["path"]
            if not sidecar.is_file() or _sha256(sidecar) != entry["sha256"] or sidecar.stat().st_size != entry["size_bytes"]:
                raise ProvenanceError(f"sidecar content mismatch: {entry['path']}")
    return ValidationResult(manifest_path, "present_validated", len(declared))


def _validate_executable_closure(payload: dict[str, Any], *, repo_root: Path) -> None:
    if payload["regeneration_status"] != "ready":
        raise ProvenanceError(
            "regeneration closure is on hold: "
            f"{payload['regeneration_status']}"
        )
    producer = repo_root / payload["producer_script"]
    if not producer.is_file():
        raise ProvenanceError(f"producer script is missing: {payload['producer_script']}")
    validator = payload["validator_module"]
    if validator != "public_data.provenance" or importlib.util.find_spec(validator) is None:
        raise ProvenanceError(f"validator module is not importable: {validator}")
    command = payload["command"]
    if "python -m public_data." not in command:
        raise ProvenanceError("generation command must use a retained public_data module")
    for dependency in payload.get("dependencies", []):
        if dependency.get("required", True) and not (repo_root / dependency["path"]).exists():
            raise ProvenanceError(f"required executable dependency is missing: {dependency['path']}")


def validate_refinement_authority(authority: dict[str, Any]) -> None:
    """Validate immutable terminal refinement authority without mutating it."""

    if not isinstance(authority, dict):
        raise ProvenanceError("refinement authority must be an object")
    producer = authority.get("historical_producer")
    if not isinstance(producer, dict) or producer.get("publisher_version") != _REFINEMENT_PUBLISHER_VERSION:
        raise ProvenanceError("refinement authority has an invalid historical publisher")
    commit = producer.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ProvenanceError("refinement authority has an invalid producer commit")
    runtime_root = _authority_path(authority.get("runtime_root"), label="runtime root")
    config_recorded_path, config_sha = _validate_recorded_identity(
        authority.get("historical_training_config"),
        label="historical training config",
    )
    tokenizer_path, tokenizer_sha = _validate_named_identity(authority.get("tokenizer"), label="tokenizer")
    splits = authority.get("splits")
    if not isinstance(splits, dict) or not splits or set(splits) - {"train", "val"}:
        raise ProvenanceError("refinement authority must name train and/or val splits")
    for split, split_authority in splits.items():
        _validate_refinement_split(
            split,
            split_authority,
            runtime_root=runtime_root,
            config_recorded_path=config_recorded_path,
            config_sha=config_sha,
            tokenizer_sha=tokenizer_sha,
        )


def _validate_refinement_split(
    split: str,
    authority: Any,
    *,
    runtime_root: Path,
    config_recorded_path: str,
    config_sha: str,
    tokenizer_sha: str,
) -> None:
    if not isinstance(authority, dict):
        raise ProvenanceError(f"{split} refinement authority must be an object")
    generation = _positive_int(authority.get("generation"), label=f"{split} generation")
    row_count = _positive_int(authority.get("row_count"), label=f"{split} row count")
    object_count = _nonnegative_int(authority.get("object_count"), label=f"{split} object count")
    negative_count = _nonnegative_int(
        authority.get("negative_object_count"), label=f"{split} negative object count"
    )
    files = authority.get("files")
    outputs = authority.get("outputs")
    if not isinstance(files, dict) or set(files) != {"receipt", "project", "journal", "working", "task_index"}:
        raise ProvenanceError(f"{split} authority has an invalid file inventory")
    if not isinstance(outputs, dict) or set(outputs) != {"norm", "coord"}:
        raise ProvenanceError(f"{split} authority has an invalid output inventory")
    identities = {
        name: _validate_named_identity(files[name], label=f"{split} {name}")
        for name in ("receipt", "project", "journal", "working", "task_index")
    }
    output_identities = {
        name: _validate_output_identity(outputs[name], label=f"{split} {name}")
        for name in ("norm", "coord")
    }
    receipt_path, _ = identities["receipt"]
    project_path, _ = identities["project"]
    journal_path, journal_sha = identities["journal"]
    working_path, working_sha = identities["working"]
    task_index_path, task_index_sha = identities["task_index"]
    receipt = _read_strict_json_object(receipt_path, label=f"{split} receipt")
    project = _read_strict_json_object(project_path, label=f"{split} project")

    expected_split_root = runtime_root / split
    for label, path in (
        ("receipt", receipt_path),
        ("project", project_path),
        ("journal", journal_path),
        ("working", working_path),
        ("task index", task_index_path),
    ):
        if path.parent != expected_split_root:
            raise ProvenanceError(f"{split} {label} is outside the declared runtime split")

    if (
        receipt.get("schema_version") != 2
        or receipt.get("code") != _TERMINAL_PUBLICATION_CODE
        or receipt.get("publisher_version") != _REFINEMENT_PUBLISHER_VERSION
    ):
        raise ProvenanceError(f"{split} receipt is not a terminal publication receipt")
    if (
        receipt.get("split") != split
        or receipt.get("generation") != generation
        or receipt.get("published_at_utc") != authority.get("published_at_utc")
    ):
        raise ProvenanceError(f"{split} receipt generation identity mismatch")
    if Path(str(receipt.get("runtime_root", ""))).resolve() != runtime_root:
        raise ProvenanceError(f"{split} receipt runtime root mismatch")
    receipt_working = receipt.get("working")
    if not _receipt_identity_matches(receipt_working, working_path, working_sha):
        raise ProvenanceError(f"{split} receipt working hash mismatch")
    receipt_outputs = receipt.get("outputs")
    if not isinstance(receipt_outputs, dict):
        raise ProvenanceError(f"{split} receipt output inventory is missing")
    for name, (path, digest, size, records) in output_identities.items():
        entry = receipt_outputs.get(name)
        if (
            not isinstance(entry, dict)
            or Path(str(entry.get("path", ""))).resolve() != path
            or entry.get("sha256") != digest
            or entry.get("size_bytes") != size
            or records != row_count
        ):
            raise ProvenanceError(f"{split} receipt {name} output binding mismatch")
    if receipt.get("row_count") != row_count or receipt.get("object_count") != object_count:
        raise ProvenanceError(f"{split} receipt count mismatch")
    identity = receipt.get("identity_authority")
    if (
        not isinstance(identity, dict)
        or Path(str(identity.get("journal_path", ""))).resolve() != journal_path
        or identity.get("journal_sha256") != journal_sha
        or identity.get("negative_object_count") != negative_count
        or identity.get("status") != "passed"
    ):
        raise ProvenanceError(f"{split} receipt journal authority mismatch")
    loader = receipt.get("loader_attestation")
    norm_schema = receipt.get("norm_schema_attestation")
    token_budget = receipt.get("token_budget")
    if not isinstance(loader, dict) or loader.get("status") != "passed" or loader.get("coord_row_count") != row_count:
        raise ProvenanceError(f"{split} loader attestation mismatch")
    if not isinstance(norm_schema, dict) or norm_schema.get("status") != "passed" or norm_schema.get("row_count") != row_count:
        raise ProvenanceError(f"{split} norm-schema attestation mismatch")
    if (
        not isinstance(token_budget, dict)
        or token_budget.get("status") != "passed"
        or token_budget.get("row_count") != row_count
        or token_budget.get("max_total_tokens") != 12000
        or token_budget.get("training_config_path") != config_recorded_path
        or token_budget.get("training_config_sha256") != config_sha
        or token_budget.get("tokenizer_sha256") != tokenizer_sha
    ):
        raise ProvenanceError(f"{split} token/config authority mismatch")

    if (
        project.get("schema_version") != 2
        or project.get("split") != split
        or project.get("generation") != generation
        or project.get("task_count") != row_count
        or project.get("working_line_count") != row_count
        or project.get("working_sha256") != working_sha
        or project.get("task_index_sha256") != task_index_sha
    ):
        raise ProvenanceError(f"{split} project authority mismatch")
    canonical_manifest_sha = authority.get("canonical_manifest_sha256")
    if canonical_manifest_sha != _canonical_json_sha256(project):
        raise ProvenanceError(f"{split} canonical project manifest hash mismatch")

    records = _read_and_validate_journal(journal_path, split=split)
    terminals = [
        record
        for record in records
        if record.get("kind") == "batch_terminal" and record.get("status") == "succeeded"
    ]
    if not terminals:
        raise ProvenanceError(f"{split} journal has no succeeded terminal generation")
    terminal = max(terminals, key=lambda record: _nonnegative_int(record.get("generation"), label="journal generation"))
    if (
        terminal.get("record_hash") != authority.get("terminal_record_hash")
        or
        terminal.get("generation") != generation
        or terminal.get("working_sha256") != working_sha
        or terminal.get("error") not in (None, "")
    ):
        raise ProvenanceError(f"{split} terminal journal generation mismatch")
    prepared_hash = terminal.get("prepared_record_hash")
    prepared = next(
        (
            record
            for record in records
            if record.get("kind") == "batch_prepared" and record.get("record_hash") == prepared_hash
        ),
        None,
    )
    if (
        prepared is None
        or prepared.get("candidate_generation") != generation
        or prepared.get("candidate_working_sha256") != working_sha
        or prepared.get("candidate_manifest_hash") != canonical_manifest_sha
        or prepared.get("candidate_manifest") != project
    ):
        raise ProvenanceError(f"{split} prepared/project authority mismatch")

    working_rows, working_objects, working_negative_pairs = _working_inventory(working_path, split=split)
    if working_rows != row_count or working_objects != object_count or len(working_negative_pairs) != negative_count:
        raise ProvenanceError(f"{split} working inventory count mismatch")
    reservation_pairs: set[tuple[int, int]] = set()
    for record in records:
        if record.get("kind") != "reservation":
            continue
        image_id = record.get("image_id")
        coco_ann_id = record.get("coco_ann_id")
        if not isinstance(image_id, int) or isinstance(image_id, bool) or not isinstance(coco_ann_id, int) or isinstance(coco_ann_id, bool) or coco_ann_id >= 0:
            raise ProvenanceError(f"{split} journal has an invalid negative-ID reservation")
        pair = (image_id, coco_ann_id)
        if pair in reservation_pairs:
            raise ProvenanceError(f"{split} journal has duplicate negative-ID ownership")
        reservation_pairs.add(pair)
    if reservation_pairs != working_negative_pairs:
        raise ProvenanceError(f"{split} negative-ID ownership mismatch")


def _validate_named_identity(value: Any, *, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise ProvenanceError(f"{label} identity is missing")
    path = _authority_path(value.get("path"), label=label)
    expected = value.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ProvenanceError(f"{label} identity has an invalid sha256")
    if not path.is_file():
        raise ProvenanceError(f"{label} authority file is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ProvenanceError(f"{label} file hash mismatch: expected={expected} actual={actual}")
    return path, expected


def _validate_recorded_identity(value: Any, *, label: str) -> tuple[str, str]:
    """Validate a receipt-bound identity without treating its path as a live dependency."""

    if not isinstance(value, dict):
        raise ProvenanceError(f"{label} identity is missing")
    if value.get("identity_mode") != "recorded_receipt_only":
        raise ProvenanceError(f"{label} identity mode is invalid")
    recorded_path = value.get("recorded_path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ProvenanceError(f"{label} recorded path is missing")
    expected = value.get("sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdef" for char in expected)
    ):
        raise ProvenanceError(f"{label} identity has an invalid sha256")
    return recorded_path, expected


def _validate_output_identity(value: Any, *, label: str) -> tuple[Path, str, int, int]:
    path, digest = _validate_named_identity(value, label=label)
    if not isinstance(value, dict):
        raise ProvenanceError(f"{label} identity is missing")
    size = _nonnegative_int(value.get("size_bytes"), label=f"{label} size")
    records = _nonnegative_int(value.get("records"), label=f"{label} records")
    if path.stat().st_size != size:
        raise ProvenanceError(f"{label} size mismatch")
    actual_records = _count_records(path)
    if actual_records != records:
        raise ProvenanceError(f"{label} record-count mismatch")
    return path, digest, size, records


def _read_and_validate_journal(path: Path, *, split: str) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ProvenanceError(f"{split} journal has an incomplete final record")
    previous: str | None = None
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(data.splitlines(), start=1):
        try:
            record = json.loads(raw, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise ProvenanceError(f"{split} journal has invalid JSON at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ProvenanceError(f"{split} journal record {line_number} is not an object")
        if record.get("prev_record_hash") != previous:
            raise ProvenanceError(f"{split} broken journal chain at line {line_number}")
        recorded = record.get("record_hash")
        check = dict(record)
        check.pop("record_hash", None)
        if recorded != _canonical_json_sha256(check):
            raise ProvenanceError(f"{split} journal hash disagreement at line {line_number}")
        previous = str(recorded)
        records.append(record)
    return records


def _working_inventory(path: Path, *, split: str) -> tuple[int, int, set[tuple[int, int]]]:
    rows = 0
    objects = 0
    negative_pairs: set[tuple[int, int]] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            try:
                payload = json.loads(raw, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                raise ProvenanceError(f"{split} working JSONL is invalid at line {line_number}") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
                raise ProvenanceError(f"{split} working row {line_number} has an invalid object inventory")
            image_id = payload.get("image_id")
            if not isinstance(image_id, int) or isinstance(image_id, bool):
                raise ProvenanceError(f"{split} working row {line_number} has an invalid image_id")
            rows += 1
            objects += len(payload["objects"])
            for obj in payload["objects"]:
                if not isinstance(obj, dict):
                    raise ProvenanceError(f"{split} working row {line_number} has an invalid object")
                coco_ann_id = obj.get("coco_ann_id")
                if isinstance(coco_ann_id, int) and not isinstance(coco_ann_id, bool) and coco_ann_id < 0:
                    pair = (image_id, coco_ann_id)
                    if pair in negative_pairs:
                        raise ProvenanceError(f"{split} working data has duplicate negative-ID ownership")
                    negative_pairs.add(pair)
    return rows, objects, negative_pairs


def _receipt_identity_matches(value: Any, path: Path, digest: str) -> bool:
    return (
        isinstance(value, dict)
        and Path(str(value.get("path", ""))).resolve() == path
        and value.get("sha256") == digest
    )


def _read_strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes(), parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ProvenanceError(f"{label} is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise ProvenanceError(f"{label} is not a JSON object")
    return payload


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _authority_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProvenanceError(f"{label} path is missing")
    return Path(value).expanduser().resolve()


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProvenanceError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProvenanceError(f"{label} must be a nonnegative integer")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for manifest in args.manifest:
        result = validate_manifest(manifest, repo_root=args.repo_root, data_root=args.data_root)
        print(f"{manifest}: {result.materialization} files={result.checked_files}")


if __name__ == "__main__":
    main()

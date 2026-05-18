from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "manifests" / "public_data_provenance"

REQUIRED_FIELDS = {
    "schema_version",
    "artifact_type",
    "relative_path",
    "producer_script",
    "working_dir",
    "command",
    "inputs",
    "key_params",
    "checksums",
    "code_ref",
    "generated_at_utc",
    "notes",
}

OPTIONAL_FIELDS = {
    "metadata",
}

REQUESTED_PUBLIC_DATA_MANIFEST_TARGETS = {
    "public_data/coco/images/res-1024",
    "public_data/coco/rescale_32_1024_bbox",
    "public_data/coco/rescale_32_1024_bbox_max60",
    "public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy",
    "public_data/coco/rescale_32_1024_bbox_len12000",
    "public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000",
    "public_data/coco/views/coco80/full",
    "public_data/coco/views/coco80/len-12000",
    "public_data/coco/views/coco80/max-60",
    "public_data/coco/views/coco80-lvis-proxy/len-12000",
}


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict), f"manifest must be a JSON object: {path}"
    return payload


def _manifest_path_for(relative_path: str) -> Path:
    rel = Path(relative_path)
    assert rel.parts[0] == "public_data"
    return MANIFEST_ROOT / Path(*rel.parts[1:]).with_suffix(".json")


def _iter_manifest_paths() -> list[Path]:
    return sorted(
        path
        for path in MANIFEST_ROOT.rglob("*.json")
        if path.name != "schema.json"
    )


def _assert_repo_relative_path(value: str, *, field: str) -> None:
    path = Path(value)
    assert not path.is_absolute(), f"{field} must stay repo-relative: {value}"
    assert ".." not in path.parts, f"{field} must not escape repo root: {value}"


def test_public_data_provenance_manifest_schema_allows_jsonl_only_checksums() -> None:
    schema = _load_manifest(MANIFEST_ROOT / "schema.json")

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == REQUIRED_FIELDS
    assert schema["properties"]["artifact_type"]["enum"] == [
        "processed_directory",
        "image_store",
        "annotation_view",
    ]
    checksum_schema = schema["properties"]["checksums"]
    assert checksum_schema["properties"]["scope"]["const"] == "jsonl_training_samples_only"
    assert checksum_schema["properties"]["algorithm"]["const"] == "sha256"
    path_schema = checksum_schema["properties"]["files"]["items"]["properties"]["path"]
    assert path_schema["pattern"].endswith("\\.jsonl$")
    assert "not" in path_schema
    assert checksum_schema["properties"]["files"]["minItems"] == 1
    assert "not" in schema["properties"]["relative_path"]
    assert "not" in schema["properties"]["inputs"]["items"]["properties"]["path"]
    assert "allOf" in schema


def test_public_data_provenance_manifests_validate_against_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_manifest(MANIFEST_ROOT / "schema.json")
    validator = jsonschema.Draft202012Validator(schema)

    for manifest_path in _iter_manifest_paths():
        payload = _load_manifest(manifest_path)
        errors = sorted(
            validator.iter_errors(payload),
            key=lambda error: tuple(error.absolute_path),
        )
        assert not errors, (
            f"manifest does not validate against schema: {manifest_path}\n"
            + "\n".join(
                f"{list(error.absolute_path)}: {error.message}" for error in errors
            )
        )


def test_public_data_provenance_manifests_follow_path_and_shape_contract() -> None:
    manifest_paths = _iter_manifest_paths()
    assert manifest_paths, "expected public_data provenance manifests"

    for manifest_path in manifest_paths:
        payload = _load_manifest(manifest_path)

        assert REQUIRED_FIELDS <= set(payload.keys())
        assert set(payload.keys()) <= REQUIRED_FIELDS | OPTIONAL_FIELDS
        assert payload["schema_version"] == 1
        assert manifest_path == _manifest_path_for(payload["relative_path"])

        relative_path = str(payload["relative_path"])
        assert relative_path.startswith("public_data/")
        assert (
            "/raw" not in relative_path
        ), "raw datasets are external recovery inputs, not processed provenance targets"

        _assert_repo_relative_path(relative_path, field="relative_path")
        _assert_repo_relative_path(str(payload["producer_script"]), field="producer_script")
        _assert_repo_relative_path(str(payload["working_dir"]), field="working_dir")

        inputs = payload["inputs"]
        assert isinstance(inputs, list) and inputs
        for input_entry in inputs:
            assert set(input_entry.keys()) == {"kind", "path", "notes"}
            assert str(input_entry["kind"]).strip()
            assert str(input_entry["path"]).strip()
            _assert_repo_relative_path(str(input_entry["path"]), field="inputs[].path")

        assert isinstance(payload["key_params"], dict)
        assert "checksums" in payload
        assert str(payload["command"]).strip()
        assert str(payload["notes"]).strip()
        assert payload["artifact_type"] == _expected_artifact_type(relative_path)
        _assert_checksum_contract(payload)
        _assert_sidecar_metadata_contract(payload)


def test_requested_public_data_targets_have_regeneration_manifests() -> None:
    for relative_path in sorted(REQUESTED_PUBLIC_DATA_MANIFEST_TARGETS):
        manifest_path = _manifest_path_for(relative_path)
        assert manifest_path.is_file(), f"missing provenance manifest for {relative_path}"
        assert _load_manifest(manifest_path)["relative_path"] == relative_path


def test_phase1_image_store_manifest_does_not_hash_image_files() -> None:
    payload = _load_manifest(_manifest_path_for("public_data/coco/images/res-1024"))

    assert payload["artifact_type"] == "image_store"
    assert payload["checksums"] is None
    assert payload["key_params"]["image_file_hashing"] == "not_materialized_by_default"
    assert "--image-store-mode hardlink" in payload["command"]


def test_annotation_view_manifests_have_jsonl_checksums_and_sidecar_metadata() -> None:
    for manifest_path in _iter_manifest_paths():
        payload = _load_manifest(manifest_path)
        relative_path = str(payload["relative_path"])
        if payload["artifact_type"] != "annotation_view":
            continue

        assert isinstance(payload["checksums"], dict)

        key_params = payload["key_params"]
        view_summary = key_params["view_summary"]
        assert isinstance(view_summary["rendered_object_count"], int)
        assert view_summary["rendered_object_count"] >= 0
        assert isinstance(view_summary["support_sidecar_count"], int)
        assert view_summary["support_sidecar_count"] >= 0
        if relative_path.endswith("len-12000"):
            assert key_params["length_budget_scope"] == {
                "rendered_families": ["objects"],
                "excluded_sidecars": ["metadata.supervision.support_objects"],
            }

        metadata = payload["metadata"]
        assert metadata["view_metadata"]["path"] == f"{relative_path}/meta.json"
        assert metadata["source_comparison"]["path"] == f"{relative_path}/source_comparison.json"
        if relative_path.endswith("len-12000"):
            assert set(metadata["length_stats"]) == {"train", "val"}


def test_public_data_manifests_encode_regeneration_not_baidu_sync() -> None:
    forbidden_command_fragments = {
        "BaiduPCS-Go",
        "baidu",
        "rsync",
        "scp ",
        "tar ",
        "sha256sum public_data",
    }

    for manifest_path in _iter_manifest_paths():
        payload = _load_manifest(manifest_path)
        command = str(payload["command"])
        policy = str(payload["key_params"].get("routine_sync_policy", ""))

        assert "regenerate_from_raw_plus_manifest" in policy
        assert not any(fragment in command for fragment in forbidden_command_fragments), manifest_path


def test_materialized_jsonl_checksums_match_local_training_samples() -> None:
    for manifest_path in _iter_manifest_paths():
        payload = _load_manifest(manifest_path)
        _assert_materialized_jsonl_checksums_match(payload, repo_root=ROOT)


def test_materialized_sidecar_metadata_checksums_match_local_files() -> None:
    for manifest_path in _iter_manifest_paths():
        payload = _load_manifest(manifest_path)
        metadata = payload.get("metadata")
        if metadata is None:
            continue

        _assert_materialized_sidecar_checksums_match(payload, repo_root=ROOT)


def test_local_checksum_verification_rejects_partial_materialization(
    tmp_path: Path,
) -> None:
    aggregate_line = (
        "public_data/unit/views/example/train.jsonl "
        f"{'0' * 64} 0 0\n"
    )
    payload = {
        "relative_path": "public_data/unit/views/example",
        "checksums": {
            "scope": "jsonl_training_samples_only",
            "algorithm": "sha256",
            "aggregate_sha256": hashlib.sha256(
                aggregate_line.encode("utf-8")
            ).hexdigest(),
            "aggregate_source": "sorted path sha256 size_bytes records lines",
            "files": [
                {
                    "path": "public_data/unit/views/example/train.jsonl",
                    "sha256": "0" * 64,
                    "size_bytes": 0,
                    "records": 0,
                }
            ],
        },
    }
    (tmp_path / "public_data/unit/views/example").mkdir(parents=True)

    with pytest.raises(AssertionError, match="unexpected local JSONL set"):
        _assert_materialized_jsonl_checksums_match(payload, repo_root=tmp_path)


def test_local_checksum_verification_rejects_extra_jsonl(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "public_data/unit/views/example"
    artifact_root.mkdir(parents=True)
    expected = artifact_root / "train.jsonl"
    expected.write_text("{}\n", encoding="utf-8")
    extra = artifact_root / "extra.jsonl"
    extra.write_text("{}\n", encoding="utf-8")
    expected_sha = _sha256_file(expected)
    aggregate_line = (
        "public_data/unit/views/example/train.jsonl "
        f"{expected_sha} {expected.stat().st_size} 1\n"
    )
    payload = {
        "relative_path": "public_data/unit/views/example",
        "checksums": {
            "scope": "jsonl_training_samples_only",
            "algorithm": "sha256",
            "aggregate_sha256": hashlib.sha256(
                aggregate_line.encode("utf-8")
            ).hexdigest(),
            "aggregate_source": "sorted path sha256 size_bytes records lines",
            "files": [
                {
                    "path": "public_data/unit/views/example/train.jsonl",
                    "sha256": expected_sha,
                    "size_bytes": expected.stat().st_size,
                    "records": 1,
                }
            ],
        },
    }

    with pytest.raises(AssertionError, match="unexpected local JSONL set"):
        _assert_materialized_jsonl_checksums_match(payload, repo_root=tmp_path)


def test_manifest_only_checksum_verification_rejects_wrong_root_or_aggregate(
    tmp_path: Path,
) -> None:
    payload = {
        "relative_path": "public_data/unit/views/example",
        "checksums": {
            "scope": "jsonl_training_samples_only",
            "algorithm": "sha256",
            "aggregate_sha256": "0" * 64,
            "aggregate_source": "sorted path sha256 size_bytes records lines",
            "files": [
                {
                    "path": "public_data/unit/views/other/train.jsonl",
                    "sha256": "1" * 64,
                    "size_bytes": 10,
                    "records": 1,
                }
            ],
        },
    }

    with pytest.raises(AssertionError, match="must live under artifact root"):
        _assert_materialized_jsonl_checksums_match(payload, repo_root=tmp_path)

    payload["checksums"]["files"][0]["path"] = (
        "public_data/unit/views/example/train.jsonl"
    )
    with pytest.raises(AssertionError, match="aggregate_sha256"):
        _assert_materialized_jsonl_checksums_match(payload, repo_root=tmp_path)


def test_local_sidecar_verification_rejects_missing_declared_sidecar(
    tmp_path: Path,
) -> None:
    payload = {
        "relative_path": "public_data/unit/views/example",
        "metadata": {
            "view_metadata": {
                "path": "public_data/unit/views/example/meta.json",
                "sha256": "0" * 64,
                "size_bytes": 0,
            }
        },
    }
    (tmp_path / "public_data/unit/views/example").mkdir(parents=True)

    with pytest.raises(AssertionError, match="missing locally"):
        _assert_materialized_sidecar_checksums_match(payload, repo_root=tmp_path)


def test_local_sidecar_verification_rejects_extra_sidecar(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "public_data/unit/views/example"
    artifact_root.mkdir(parents=True)
    meta_path = artifact_root / "meta.json"
    meta_path.write_text("{}\n", encoding="utf-8")
    source_comparison_path = artifact_root / "source_comparison.json"
    source_comparison_path.write_text("{}\n", encoding="utf-8")
    payload = {
        "relative_path": "public_data/unit/views/example",
        "metadata": {
            "view_metadata": {
                "path": "public_data/unit/views/example/meta.json",
                "sha256": _sha256_file(meta_path),
                "size_bytes": meta_path.stat().st_size,
            }
        },
    }

    with pytest.raises(AssertionError, match="unexpected local sidecar set"):
        _assert_materialized_sidecar_checksums_match(payload, repo_root=tmp_path)


def test_sidecar_metadata_contract_rejects_unknown_or_escaping_metadata() -> None:
    payload = {
        "artifact_type": "annotation_view",
        "relative_path": "public_data/unit/views/example",
        "metadata": {
            "view_metadata": {
                "path": "public_data/unit/views/example/meta.json",
                "sha256": "0" * 64,
                "size_bytes": 0,
            },
            "source_comparison": {
                "path": "public_data/unit/views/example/source_comparison.json",
                "sha256": "0" * 64,
                "size_bytes": 0,
            },
            "unknown": {
                "path": "public_data/unit/views/example/unknown.json",
                "sha256": "0" * 64,
                "size_bytes": 0,
            },
        },
    }
    with pytest.raises(AssertionError, match="unknown metadata"):
        _assert_sidecar_metadata_contract(payload)

    payload["metadata"].pop("unknown")
    payload["metadata"]["view_metadata"]["path"] = "../meta.json"
    with pytest.raises(AssertionError, match="must not escape repo root"):
        _assert_sidecar_metadata_contract(payload)


def test_checksum_contract_rejects_empty_non_image_checksum_files() -> None:
    payload = {
        "artifact_type": "annotation_view",
        "checksums": {
            "scope": "jsonl_training_samples_only",
            "algorithm": "sha256",
            "aggregate_sha256": "0" * 64,
            "aggregate_source": "sorted path sha256 size_bytes records lines",
            "files": [],
        },
    }

    with pytest.raises(AssertionError, match="missing JSONL checksum entries"):
        _assert_checksum_contract(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _expected_artifact_type(relative_path: str) -> str:
    path = Path(relative_path)
    parts = path.parts
    if "images" in parts:
        return "image_store"
    if "views" in parts:
        return "annotation_view"
    return "processed_directory"


def _assert_checksum_contract(payload: dict[str, Any]) -> None:
    artifact_type = payload["artifact_type"]
    checksums = payload["checksums"]
    if artifact_type == "image_store":
        assert checksums is None, "image-store manifests must not hash image files"
        return

    assert isinstance(
        checksums,
        dict,
    ), f"{artifact_type} manifests require JSONL checksums"
    file_entries = checksums.get("files")
    assert isinstance(file_entries, list)
    assert file_entries, f"missing JSONL checksum entries for {artifact_type}"


def _assert_sidecar_metadata_contract(payload: dict[str, Any]) -> None:
    artifact_type = payload["artifact_type"]
    relative_path = str(payload["relative_path"])
    metadata = payload.get("metadata")

    if artifact_type == "processed_directory":
        if metadata is None:
            return
    else:
        assert isinstance(metadata, dict), f"{artifact_type} manifests require metadata"

    if metadata is None:
        return

    allowed_keys = {
        "image_store_metadata",
        "view_metadata",
        "source_comparison",
        "length_stats",
    }
    unknown_keys = sorted(set(metadata) - allowed_keys)
    assert set(metadata) <= allowed_keys, f"unknown metadata keys: {unknown_keys}"

    if artifact_type == "image_store":
        assert "image_store_metadata" in metadata
    if artifact_type == "annotation_view":
        assert "view_metadata" in metadata
        assert "source_comparison" in metadata

    for sidecar_entry in _iter_sidecar_entries(metadata):
        path = str(sidecar_entry["path"])
        _assert_repo_relative_path(path, field="metadata sidecar path")
        assert path.startswith(
            f"{relative_path}/"
        ), f"sidecar path must live under artifact root: {path}"


def _assert_materialized_jsonl_checksums_match(
    payload: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    checksum_payload = payload["checksums"]
    if checksum_payload is None:
        return

    relative_path = str(payload["relative_path"])
    assert checksum_payload["scope"] == "jsonl_training_samples_only"
    assert checksum_payload["algorithm"] == "sha256"
    assert (
        checksum_payload["aggregate_source"]
        == "sorted path sha256 size_bytes records lines"
    )

    file_entries = checksum_payload["files"]
    assert file_entries == sorted(file_entries, key=lambda entry: entry["path"])
    assert file_entries, f"missing JSONL checksum entries for {relative_path}"

    aggregate_lines: list[str] = []
    for file_entry in file_entries:
        path = str(file_entry["path"])
        assert path.startswith(
            f"{relative_path}/"
        ), f"checksum path must live under artifact root: {path}"
        assert path.endswith(".jsonl")

        aggregate_lines.append(
            f"{path} {file_entry['sha256']} {file_entry['size_bytes']} {file_entry['records']}\n"
        )

    observed_aggregate = hashlib.sha256(
        "".join(aggregate_lines).encode("utf-8")
    ).hexdigest()
    assert checksum_payload["aggregate_sha256"] == observed_aggregate, (
        "aggregate_sha256 must match manifest checksum entries"
    )

    artifact_root = repo_root / relative_path
    if not artifact_root.exists():
        return

    listed_paths = {repo_root / str(entry["path"]) for entry in file_entries}
    local_jsonl_paths = set(artifact_root.glob("*.jsonl"))
    assert local_jsonl_paths == listed_paths, (
        "unexpected local JSONL set for "
        f"{relative_path}: expected={sorted(str(p) for p in listed_paths)} "
        f"observed={sorted(str(p) for p in local_jsonl_paths)}"
    )

    for file_entry in file_entries:
        path = str(file_entry["path"])
        local_path = repo_root / path
        assert local_path.is_file(), f"manifest checksum path is missing locally: {path}"
        assert file_entry["size_bytes"] == local_path.stat().st_size
        assert file_entry["records"] == _count_nonempty_lines(local_path)
        assert file_entry["sha256"] == _sha256_file(local_path)


def _assert_materialized_sidecar_checksums_match(
    payload: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    metadata = payload.get("metadata")
    if metadata is None:
        return

    artifact_root = repo_root / str(payload["relative_path"])
    if not artifact_root.exists():
        return

    for sidecar_entry in _iter_sidecar_entries(metadata):
        path = sidecar_entry["path"]
        local_path = repo_root / path
        assert local_path.is_file(), f"manifest sidecar path is missing locally: {path}"
        assert sidecar_entry["sha256"] == _sha256_file(local_path)
        assert sidecar_entry["size_bytes"] == local_path.stat().st_size

    declared_paths = {repo_root / entry["path"] for entry in _iter_sidecar_entries(metadata)}
    local_sidecar_paths = _local_sidecar_paths(payload, repo_root=repo_root)
    assert declared_paths == local_sidecar_paths, (
        "unexpected local sidecar set for "
        f"{payload['relative_path']}: "
        f"expected={sorted(str(p) for p in declared_paths)} "
        f"observed={sorted(str(p) for p in local_sidecar_paths)}"
    )


def _iter_sidecar_entries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for key in ("image_store_metadata", "view_metadata", "source_comparison"):
        value = metadata.get(key)
        if isinstance(value, dict) and "path" in value:
            entries.append(value)

    length_stats = metadata.get("length_stats")
    if isinstance(length_stats, dict):
        for value in length_stats.values():
            if isinstance(value, dict) and "path" in value:
                entries.append(value)

    return entries


def _local_sidecar_paths(
    payload: dict[str, Any],
    *,
    repo_root: Path,
) -> set[Path]:
    artifact_root = repo_root / str(payload["relative_path"])
    artifact_type = payload.get("artifact_type")
    if artifact_type == "image_store" or "/images/" in str(payload["relative_path"]):
        return {path for path in [artifact_root / "meta.json"] if path.exists()}
    if artifact_type == "annotation_view" or "/views/" in str(payload["relative_path"]):
        sidecars = {
            path
            for path in [
                artifact_root / "meta.json",
                artifact_root / "source_comparison.json",
            ]
            if path.exists()
        }
        sidecars.update(artifact_root.glob("*.length_stats.json"))
        return sidecars
    return set()

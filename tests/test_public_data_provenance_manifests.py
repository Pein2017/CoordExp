from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "manifests" / "public_data_provenance"

REQUIRED_FIELDS = {
    "schema_version",
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

REQUESTED_COCO1024_TRAINING_VARIANTS = {
    "public_data/coco/rescale_32_1024_bbox",
    "public_data/coco/rescale_32_1024_bbox_max60",
    "public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy",
    "public_data/coco/rescale_32_1024_bbox_len12000",
    "public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000",
}

MATERIALIZED_COCO1024_VARIANTS = {
    "public_data/coco/rescale_32_1024_bbox",
    "public_data/coco/rescale_32_1024_bbox_max60",
    "public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy",
    "public_data/coco/rescale_32_1024_bbox_len12000",
    "public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000",
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


def _assert_repo_relative_path(value: str, *, field: str) -> None:
    path = Path(value)
    assert not path.is_absolute(), f"{field} must stay repo-relative: {value}"
    assert ".." not in path.parts, f"{field} must not escape repo root: {value}"


def test_public_data_provenance_manifest_schema_allows_jsonl_only_checksums() -> None:
    schema = _load_manifest(MANIFEST_ROOT / "schema.json")

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == REQUIRED_FIELDS
    checksum_schema = schema["properties"]["checksums"]
    assert checksum_schema["properties"]["scope"]["const"] == "jsonl_training_samples_only"
    assert checksum_schema["properties"]["algorithm"]["const"] == "sha256"
    assert checksum_schema["properties"]["files"]["items"]["properties"]["path"][
        "pattern"
    ].endswith("\\.jsonl$")


def test_public_data_provenance_manifests_follow_path_and_shape_contract() -> None:
    manifest_paths = sorted(MANIFEST_ROOT.glob("*/*.json"))
    assert manifest_paths, "expected public_data provenance manifests"

    for manifest_path in manifest_paths:
        payload = _load_manifest(manifest_path)

        assert set(payload.keys()) == REQUIRED_FIELDS
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


def test_requested_coco1024_training_variants_have_regeneration_manifests() -> None:
    for relative_path in sorted(REQUESTED_COCO1024_TRAINING_VARIANTS):
        manifest_path = _manifest_path_for(relative_path)
        assert manifest_path.is_file(), f"missing provenance manifest for {relative_path}"
        assert _load_manifest(manifest_path)["relative_path"] == relative_path


def test_materialized_coco1024_processed_dirs_have_matching_manifests() -> None:
    for relative_path in sorted(MATERIALIZED_COCO1024_VARIANTS):
        assert (
            ROOT / relative_path
        ).is_dir(), f"expected materialized processed data dir: {relative_path}"

        payload = _load_manifest(_manifest_path_for(relative_path))
        assert payload["relative_path"] == relative_path
        assert isinstance(
            payload["checksums"], dict
        ), f"materialized data needs JSONL checksums: {relative_path}"


def test_public_data_manifests_encode_regeneration_not_baidu_sync() -> None:
    forbidden_command_fragments = {
        "BaiduPCS-Go",
        "baidu",
        "rsync",
        "scp ",
        "tar ",
        "sha256sum public_data",
    }

    for manifest_path in sorted(MANIFEST_ROOT.glob("*/*.json")):
        payload = _load_manifest(manifest_path)
        command = str(payload["command"])
        policy = str(payload["key_params"].get("routine_sync_policy", ""))

        assert "regenerate_from_raw_plus_manifest" in policy
        assert not any(fragment in command for fragment in forbidden_command_fragments), manifest_path


def test_materialized_jsonl_checksums_match_local_training_samples() -> None:
    for relative_path in sorted(MATERIALIZED_COCO1024_VARIANTS):
        payload = _load_manifest(_manifest_path_for(relative_path))
        checksum_payload = payload["checksums"]
        assert checksum_payload["scope"] == "jsonl_training_samples_only"
        assert checksum_payload["algorithm"] == "sha256"

        file_entries = checksum_payload["files"]
        assert file_entries == sorted(file_entries, key=lambda entry: entry["path"])
        assert file_entries, f"missing JSONL checksum entries for {relative_path}"

        aggregate_lines: list[str] = []
        for file_entry in file_entries:
            path = str(file_entry["path"])
            assert path.startswith(f"{relative_path}/")
            assert path.endswith(".jsonl")

            local_path = ROOT / path
            assert local_path.is_file(), f"manifest checksum path is missing locally: {path}"
            assert file_entry["size_bytes"] == local_path.stat().st_size
            assert file_entry["records"] == _count_nonempty_lines(local_path)
            assert file_entry["sha256"] == _sha256_file(local_path)

            aggregate_lines.append(
                f"{path} {file_entry['sha256']} {file_entry['size_bytes']} {file_entry['records']}\n"
            )

        observed_aggregate = hashlib.sha256(
            "".join(aggregate_lines).encode("utf-8")
        ).hexdigest()
        assert checksum_payload["aggregate_sha256"] == observed_aggregate


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

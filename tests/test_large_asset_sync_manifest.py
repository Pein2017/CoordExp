from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.large_asset_sync import (
    LargeAssetManifest,
    classify_local_against_manifest,
    load_manifest,
    load_policy,
    remote_path_for_relative_path,
    scan_local_root,
    should_ignore_relative_path,
    write_manifest,
)


def test_load_policy_reads_seed_contract_and_preserves_relative_roots() -> None:
    policy = load_policy(Path("manifests/large_assets/policy.yaml"))

    assert policy.schema_version == 1
    assert policy.remote_root == "/CoordExp"
    assert policy.report_dir == "temp/large_asset_sync"
    assert policy.ignore_file == "manifests/large_assets/ignore.txt"

    managed_roots = {root.name: root for root in policy.managed_roots}
    assert sorted(managed_roots) == ["model_cache", "output", "public_data"]
    assert managed_roots["public_data"].relative_root == "public_data"
    assert managed_roots["public_data"].hash_policy == "full"
    assert managed_roots["output"].relative_root == "output"
    assert managed_roots["output"].hash_policy == "sampled"
    assert (
        managed_roots["output"].manifest_path
        == "manifests/large_assets/output.manifest.json"
    )


def test_should_ignore_relative_path_matches_globs() -> None:
    patterns = ["**/*.tmp", "**/tmp/**", "**/cache/**"]

    assert should_ignore_relative_path("output/run-a/model.tmp", patterns) is True
    assert should_ignore_relative_path("output/run-a/cache/model.bin", patterns) is True
    assert should_ignore_relative_path("output/tmp/model.bin", patterns) is True
    assert (
        should_ignore_relative_path(
            "output/run-a/checkpoint-100/adapter_model.safetensors",
            patterns,
        )
        is False
    )


def test_remote_path_for_relative_path_preserves_repo_layout() -> None:
    assert remote_path_for_relative_path("/CoordExp", "output/stage1/run-a/file.bin") == (
        "/CoordExp/output/stage1/run-a/file.bin"
    )


def test_path_helpers_reject_absolute_and_parent_traversal_inputs() -> None:
    patterns = ["**/*.tmp"]

    with pytest.raises(ValueError, match="relative"):
        should_ignore_relative_path("/tmp/output/run-a/model.tmp", patterns)
    with pytest.raises(ValueError, match="parent traversal"):
        remote_path_for_relative_path("/CoordExp", "../output/run-a/file.bin")


def test_empty_manifest_roundtrip_preserves_seed_shape(tmp_path: Path) -> None:
    manifest = LargeAssetManifest(
        schema_version=1,
        remote_root="/CoordExp",
        managed_root="output",
        relative_root="output",
        hash_policy="sampled",
        generated_at_utc="",
        files=[],
    )
    manifest_path = tmp_path / "output.manifest.json"

    write_manifest(manifest_path, manifest)
    loaded = load_manifest(manifest_path)

    assert loaded == manifest


def test_classify_local_against_manifest_marks_missing_synced_and_metadata_drift() -> None:
    manifest_files = {
        "output/run-a/adapter_model.safetensors": {"size_bytes": 12, "sha256": "abc"},
        "output/run-a/summary.json": {"size_bytes": 2, "sha256": "def"},
        "output/run-a/stale.bin": {"size_bytes": 7, "sha256": "old"},
    }
    local_files = {
        "output/run-a/adapter_model.safetensors": {"size_bytes": 12, "sha256": "abc"},
        "output/run-a/stale.bin": {"size_bytes": 8, "sha256": "new"},
        "output/run-a/new.bin": {"size_bytes": 1, "sha256": "ghi"},
    }

    diff = classify_local_against_manifest(manifest_files, local_files)

    assert diff["synced"] == ["output/run-a/adapter_model.safetensors"]
    assert diff["missing_local"] == ["output/run-a/summary.json"]
    assert diff["missing_remote"] == ["output/run-a/new.bin"]
    assert diff["metadata_drift"] == ["output/run-a/stale.bin"]


def test_scan_local_root_classification_ignores_manifest_only_remote_path(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    managed_root = repo_root / "output" / "run-a"
    managed_root.mkdir(parents=True)

    synced_path = managed_root / "adapter_model.safetensors"
    synced_path.write_bytes(b"weights-1234")

    drift_path = managed_root / "stale.bin"
    drift_path.write_bytes(b"changed")

    ignored_path = managed_root / "cache" / "ignored.bin"
    ignored_path.parent.mkdir(parents=True)
    ignored_path.write_bytes(b"ignore-me")

    local_files = scan_local_root(repo_root, "output", ["**/cache/**"])
    manifest_files = {
        "output/run-a/adapter_model.safetensors": {
            "size_bytes": synced_path.stat().st_size,
            "sha256": local_files["output/run-a/adapter_model.safetensors"]["sha256"],
            "remote_path": "/CoordExp/output/run-a/adapter_model.safetensors",
        },
        "output/run-a/stale.bin": {
            "size_bytes": 99,
            "sha256": "stale",
            "remote_path": "/CoordExp/output/run-a/stale.bin",
        },
        "output/run-a/missing_remote.bin": {
            "size_bytes": 5,
            "sha256": "abcde",
            "remote_path": "/CoordExp/output/run-a/missing_remote.bin",
        },
    }

    diff = classify_local_against_manifest(manifest_files, local_files)

    assert "output/run-a/cache/ignored.bin" not in local_files
    assert diff["synced"] == ["output/run-a/adapter_model.safetensors"]
    assert diff["missing_local"] == ["output/run-a/missing_remote.bin"]
    assert diff["missing_remote"] == []
    assert diff["metadata_drift"] == ["output/run-a/stale.bin"]


def test_scan_local_root_rejects_invalid_relative_root_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative"):
        scan_local_root(tmp_path, "/tmp/outside", [])
    with pytest.raises(ValueError, match="parent traversal"):
        scan_local_root(tmp_path, "../outside", [])

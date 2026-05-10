from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import yaml


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "large_asset_sync.py"
    spec = spec_from_file_location("large_asset_sync_cli_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_policy(repo_root: Path) -> Path:
    policy_path = repo_root / "manifests" / "large_assets" / "policy.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    (policy_path.parent / "ignore.txt").write_text("", encoding="utf-8")
    policy_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "repo_root": ".",
                "remote_root": "/CoordExp",
                "report_dir": "temp/large_asset_sync",
                "ignore_file": "manifests/large_assets/ignore.txt",
                "managed_roots": [
                    {
                        "name": "output",
                        "relative_root": "output",
                        "manifest_path": "manifests/large_assets/output.manifest.json",
                        "hash_policy": "sampled",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return policy_path


def _write_manifest(repo_root: Path, files: list[dict[str, object]]) -> Path:
    manifest_path = repo_root / "manifests" / "large_assets" / "output.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "remote_root": "/CoordExp",
                "managed_root": "output",
                "relative_root": "output",
                "hash_policy": "sampled",
                "generated_at_utc": "",
                "files": files,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_scan_local_writes_report(tmp_path: Path) -> None:
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    (repo_root / "output" / "run-a").mkdir(parents=True)
    (repo_root / "output" / "run-a" / "summary.json").write_text("{}", encoding="utf-8")
    policy_path = _write_policy(repo_root)
    report_path = tmp_path / "local_scan.json"

    exit_code = module.main(
        [
            "scan-local",
            "--repo-root",
            str(repo_root),
            "--policy",
            str(policy_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert "output/run-a/summary.json" in payload["local_files"]


def test_scan_remote_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    policy_path = _write_policy(repo_root)
    _write_manifest(
        repo_root,
        [
            {
                "relative_path": "output/run-a/summary.json",
                "size_bytes": 2,
                "sha256": "abc",
                "remote_path": "/CoordExp/output/run-a/summary.json",
            }
        ],
    )
    report_path = tmp_path / "remote_scan.json"

    def fake_scan_remote_manifest_index(
        bin_path: str,
        manifest_files: dict[str, dict[str, object]],
        *,
        remote_root: str,
    ) -> dict[str, object]:
        assert bin_path == "BaiduPCS-Go"
        assert remote_root == "/CoordExp"
        assert list(manifest_files) == ["output/run-a/summary.json"]
        return {
            "remote_files": {
                "output/run-a/summary.json": {
                    "remote_path": "/CoordExp/output/run-a/summary.json",
                    "size_bytes": 2,
                }
            },
            "unknown_remote_state": [],
        }

    monkeypatch.setattr(module, "scan_remote_manifest_index", fake_scan_remote_manifest_index)

    exit_code = module.main(
        [
            "scan-remote",
            "--repo-root",
            str(repo_root),
            "--policy",
            str(policy_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["remote_files"]["output/run-a/summary.json"]["size_bytes"] == 2


def test_publish_plan_reports_changed_files_without_mutating_manifest(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    (repo_root / "output" / "run-a").mkdir(parents=True)
    local_path = repo_root / "output" / "run-a" / "summary.json"
    local_path.write_text('{"updated": true}', encoding="utf-8")
    policy_path = _write_policy(repo_root)
    manifest_path = _write_manifest(
        repo_root,
        [
            {
                "relative_path": "output/run-a/summary.json",
                "size_bytes": 2,
                "sha256": "old-hash",
                "remote_path": "/CoordExp/output/run-a/summary.json",
            }
        ],
    )
    manifest_before = manifest_path.read_text(encoding="utf-8")
    report_path = tmp_path / "publish_report.json"

    exit_code = module.main(
        [
            "publish",
            "--repo-root",
            str(repo_root),
            "--policy",
            str(policy_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["plan"]["changed"] == ["output/run-a/summary.json"]
    assert payload["plan"]["new"] == []
    assert manifest_path.read_text(encoding="utf-8") == manifest_before


def test_publish_execute_rewrites_manifest_with_timestamp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    (repo_root / "output" / "run-a").mkdir(parents=True)
    (repo_root / "output" / "run-a" / "summary.json").write_text(
        '{"updated": true}',
        encoding="utf-8",
    )
    policy_path = _write_policy(repo_root)
    manifest_path = _write_manifest(repo_root, [])
    report_path = tmp_path / "publish_report.json"

    upload_calls: list[tuple[str, str, str]] = []
    mkdir_calls: list[tuple[str, str]] = []

    def fake_mkdir_remote_path(bin_path: str, remote_path: str) -> None:
        mkdir_calls.append((bin_path, remote_path))

    def fake_upload_file(bin_path: str, local_path: Path, remote_parent: str) -> None:
        upload_calls.append((bin_path, str(local_path), remote_parent))

    def fake_scan_remote_manifest_index(
        bin_path: str,
        manifest_files: dict[str, dict[str, object]],
        *,
        remote_root: str,
    ) -> dict[str, object]:
        remote_files = {
            relative_path: {
                "remote_path": str(file_info["remote_path"]),
                "size_bytes": int(file_info["size_bytes"]),
            }
            for relative_path, file_info in manifest_files.items()
        }
        return {"remote_files": remote_files, "unknown_remote_state": []}

    monkeypatch.setattr(module, "mkdir_remote_path", fake_mkdir_remote_path)
    monkeypatch.setattr(module, "upload_file", fake_upload_file)
    monkeypatch.setattr(module, "scan_remote_manifest_index", fake_scan_remote_manifest_index)

    exit_code = module.main(
        [
            "publish",
            "--repo-root",
            str(repo_root),
            "--policy",
            str(policy_path),
            "--report",
            str(report_path),
            "--execute",
        ]
    )

    assert exit_code == 0
    assert mkdir_calls == [("BaiduPCS-Go", "/CoordExp/output/run-a")]
    assert upload_calls == [
        (
            "BaiduPCS-Go",
            str(repo_root / "output" / "run-a" / "summary.json"),
            "/CoordExp/output/run-a",
        )
    ]
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["generated_at_utc"] != ""
    assert manifest_payload["files"] == [
        {
            "relative_path": "output/run-a/summary.json",
            "size_bytes": len('{"updated": true}'),
            "sha256": manifest_payload["files"][0]["sha256"],
            "remote_path": "/CoordExp/output/run-a/summary.json",
        }
    ]


def test_align_local_plan_reports_missing_local_without_downloading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    policy_path = _write_policy(repo_root)
    _write_manifest(
        repo_root,
        [
            {
                "relative_path": "output/run-a/summary.json",
                "size_bytes": 2,
                "sha256": "abc",
                "remote_path": "/CoordExp/output/run-a/summary.json",
            }
        ],
    )
    report_path = tmp_path / "align_plan.json"

    def fake_scan_remote_manifest_index(
        bin_path: str,
        manifest_files: dict[str, dict[str, object]],
        *,
        remote_root: str,
    ) -> dict[str, object]:
        return {
            "remote_files": {
                "output/run-a/summary.json": {
                    "remote_path": "/CoordExp/output/run-a/summary.json",
                    "size_bytes": 2,
                }
            },
            "unknown_remote_state": [],
        }

    def fail_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("download should not run during dry-run align-local")

    monkeypatch.setattr(module, "scan_remote_manifest_index", fake_scan_remote_manifest_index)
    monkeypatch.setattr(module, "download_file_to_local_path", fail_download)

    exit_code = module.main(
        [
            "align-local",
            "--repo-root",
            str(repo_root),
            "--policy",
            str(policy_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["plan"]["missing_local"] == ["output/run-a/summary.json"]
    assert payload["plan"]["metadata_drift_local"] == []


def test_align_local_drifted_remote_is_not_restorable_or_downloaded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    policy_path = _write_policy(repo_root)
    _write_manifest(
        repo_root,
        [
            {
                "relative_path": "output/run-a/summary.json",
                "size_bytes": 2,
                "sha256": "abc",
                "remote_path": "/CoordExp/output/run-a/summary.json",
            }
        ],
    )
    report_path = tmp_path / "align_plan.json"

    def fake_scan_remote_manifest_index(
        bin_path: str,
        manifest_files: dict[str, dict[str, object]],
        *,
        remote_root: str,
    ) -> dict[str, object]:
        return {
            "remote_files": {
                "output/run-a/summary.json": {
                    "remote_path": "/CoordExp/output/run-a/summary.json",
                    "size_bytes": 99,
                }
            },
            "unknown_remote_state": [],
        }

    def fail_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("drifted remote file should not be downloaded")

    monkeypatch.setattr(module, "scan_remote_manifest_index", fake_scan_remote_manifest_index)
    monkeypatch.setattr(module, "download_file_to_local_path", fail_download)

    exit_code = module.main(
        [
            "align-local",
            "--repo-root",
            str(repo_root),
            "--policy",
            str(policy_path),
            "--report",
            str(report_path),
            "--execute",
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["plan"]["missing_local"] == []
    assert payload["plan"]["metadata_drift_local"] == []
    assert payload["plan"]["unavailable_remote"] == ["output/run-a/summary.json"]

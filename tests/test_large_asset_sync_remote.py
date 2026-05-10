from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.utils.large_asset_sync_baidupcs import (
    download_path,
    parse_baidupcs_ls_output,
    remote_rows_to_manifest_index,
    scan_remote_manifest_index,
    upload_file,
)


def test_parse_baidupcs_ls_output_extracts_dirs_and_files() -> None:
    text = """
当前目录: /CoordExp/output
----
  #  文件大小       修改日期               文件(目录)
  0     12.00MB  2026-05-09 12:00:00  adapter_model.safetensors
  1          -  2026-05-09 12:00:00  stage1_2b/
----
"""

    rows = parse_baidupcs_ls_output(text, current_dir="/CoordExp/output")

    assert rows[0]["path"] == "/CoordExp/output/adapter_model.safetensors"
    assert rows[0]["is_dir"] is False
    assert rows[0]["size_bytes"] == 12 * 1024 * 1024
    assert rows[1]["path"] == "/CoordExp/output/stage1_2b"
    assert rows[1]["is_dir"] is True
    assert rows[1]["size_bytes"] == 0


def test_remote_rows_to_manifest_index_drops_dirs() -> None:
    rows = [
        {"path": "/CoordExp/output/run-a", "is_dir": True, "size_bytes": 0},
        {"path": "/CoordExp/output/run-a/summary.json", "is_dir": False, "size_bytes": 42},
    ]

    index = remote_rows_to_manifest_index(rows, remote_root="/CoordExp")

    assert list(index) == ["output/run-a/summary.json"]
    assert index["output/run-a/summary.json"]["remote_path"] == "/CoordExp/output/run-a/summary.json"
    assert index["output/run-a/summary.json"]["size_bytes"] == 42


def test_remote_rows_to_manifest_index_rejects_non_absolute_remote_root() -> None:
    rows = [
        {"path": "/CoordExp/output/run-a/summary.json", "is_dir": False, "size_bytes": 42},
    ]

    with pytest.raises(ValueError, match="absolute"):
        remote_rows_to_manifest_index(rows, remote_root="CoordExp")


def test_parse_baidupcs_ls_output_accepts_size_without_space_before_unit() -> None:
    text = """
当前目录: /CoordExp/output
----
  #  文件大小       修改日期               文件(目录)
  0        512B  2026-05-09 12:00:00  metrics.json
----
"""

    rows = parse_baidupcs_ls_output(text, current_dir="/CoordExp/output")

    assert rows == [
        {
            "path": "/CoordExp/output/metrics.json",
            "is_dir": False,
            "size_bytes": 512,
        }
    ]


def test_scan_remote_manifest_index_marks_failed_parent_as_unknown_remote_state(
    monkeypatch,
) -> None:
    manifest_files = {
        "output/run-a/ok.bin": {"remote_path": "/CoordExp/output/run-a/ok.bin"},
        "output/run-b/missing-state.bin": {"remote_path": "/CoordExp/output/run-b/missing-state.bin"},
    }
    outputs = {
        "/CoordExp/output/run-a": """
当前目录: /CoordExp/output/run-a
----
  #  文件大小       修改日期               文件(目录)
  0      4.00KB  2026-05-09 12:00:00  ok.bin
----
""",
    }

    def fake_run(
        args: list[str],
        *,
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert text is True
        assert capture_output is True
        remote_path = args[-1]
        if remote_path == "/CoordExp/output/run-b":
            raise subprocess.CalledProcessError(
                2,
                args,
                stderr="no such directory",
            )
        return subprocess.CompletedProcess(args, 0, stdout=outputs[remote_path], stderr="")

    monkeypatch.setattr("src.utils.large_asset_sync_baidupcs.subprocess.run", fake_run)

    report = scan_remote_manifest_index(
        "/usr/local/bin/BaiduPCS-Go",
        manifest_files,
        remote_root="/CoordExp",
    )

    assert report["remote_files"] == {
        "output/run-a/ok.bin": {
            "remote_path": "/CoordExp/output/run-a/ok.bin",
            "size_bytes": 4 * 1024,
        }
    }
    assert report["unknown_remote_state"] == ["output/run-b/missing-state.bin"]


def test_scan_remote_manifest_index_marks_parser_failure_as_unknown_remote_state(
    monkeypatch,
) -> None:
    manifest_files = {
        "output/run-a/ok.bin": {"remote_path": "/CoordExp/output/run-a/ok.bin"},
        "output/run-a/bad.bin": {"remote_path": "/CoordExp/output/run-a/bad.bin"},
    }

    def fake_list_remote_path(bin_path: str, remote_path: str) -> str:
        assert bin_path == "/usr/local/bin/BaiduPCS-Go"
        assert remote_path == "/CoordExp/output/run-a"
        return "current output format drifted"

    def fake_parse_baidupcs_ls_output(text: str, current_dir: str) -> list[dict[str, object]]:
        raise ValueError(f"unable to parse listing for {current_dir}: {text}")

    monkeypatch.setattr(
        "src.utils.large_asset_sync_baidupcs.list_remote_path",
        fake_list_remote_path,
    )
    monkeypatch.setattr(
        "src.utils.large_asset_sync_baidupcs.parse_baidupcs_ls_output",
        fake_parse_baidupcs_ls_output,
    )

    report = scan_remote_manifest_index(
        "/usr/local/bin/BaiduPCS-Go",
        manifest_files,
        remote_root="/CoordExp",
    )

    assert report["remote_files"] == {}
    assert report["unknown_remote_state"] == [
        "output/run-a/bad.bin",
        "output/run-a/ok.bin",
    ]


def test_upload_and_download_commands_are_explicit_and_mockable(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert text is True
        assert capture_output is True
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("src.utils.large_asset_sync_baidupcs.subprocess.run", fake_run)

    local_path = tmp_path / "output" / "run-a" / "summary.json"
    local_path.parent.mkdir(parents=True)
    local_path.write_text("{}", encoding="utf-8")
    save_dir = tmp_path / "downloads"

    upload_file("/usr/local/bin/BaiduPCS-Go", local_path, "/CoordExp/output/run-a")
    download_path(
        "/usr/local/bin/BaiduPCS-Go",
        "/CoordExp/output/run-a/summary.json",
        save_dir,
    )

    assert calls == [
        [
            "/usr/local/bin/BaiduPCS-Go",
            "upload",
            str(local_path),
            "/CoordExp/output/run-a",
            "--policy",
            "overwrite",
            "-p",
            "1",
            "-l",
            "1",
            "--retry",
            "8",
            "--norapid",
        ],
        [
            "/usr/local/bin/BaiduPCS-Go",
            "config",
            "set",
            "-savedir",
            str(save_dir),
        ],
        [
            "/usr/local/bin/BaiduPCS-Go",
            "download",
            "/CoordExp/output/run-a/summary.json",
            "--fullpath",
            "--mode",
            "locate",
            "-p",
            "8",
            "-l",
            "4",
            "--retry",
            "8",
            "--ow",
            "--mtime",
        ],
    ]

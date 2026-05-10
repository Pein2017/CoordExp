from __future__ import annotations

import re
import subprocess
import tempfile
from shutil import move
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


_LS_ROW_PATTERN = re.compile(
    r"^\s*\d+\s+(?P<size>.+?)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(?P<name>.+?)\s*$"
)
_NUMBERED_ROW_PATTERN = re.compile(r"^\s*\d+\s+")
_SIZE_PATTERN = re.compile(r"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>B|KB|MB|GB|TB)$")
_SIZE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}


def run_baidupcs(bin_path: str | Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(bin_path), *args],
        check=True,
        text=True,
        capture_output=True,
    )


def _normalize_remote_path(remote_path: str | Path) -> str:
    return str(PurePosixPath(str(remote_path).replace("\\", "/")))


def _normalize_remote_root(remote_root: str | Path) -> str:
    normalized = _normalize_remote_path(remote_root)
    if not PurePosixPath(normalized).is_absolute():
        raise ValueError(f"Expected an absolute POSIX remote_root, got: {remote_root!r}")
    return normalized


def _parse_size_bytes(size_text: str) -> int:
    normalized = size_text.strip()
    if normalized == "-":
        return 0
    match = _SIZE_PATTERN.match(normalized)
    if match is None:
        raise ValueError(f"Unsupported BaiduPCS-Go size field: {size_text!r}")
    value = float(match.group("value"))
    unit = match.group("unit")
    return int(value * _SIZE_UNITS[unit])


def parse_baidupcs_ls_output(text: str, current_dir: str) -> list[dict[str, Any]]:
    current_dir_normalized = _normalize_remote_path(current_dir).rstrip("/")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if _NUMBERED_ROW_PATTERN.match(line) is None:
            continue
        match = _LS_ROW_PATTERN.match(line)
        if match is None:
            raise ValueError(f"Unable to parse BaiduPCS-Go ls row: {line!r}")
        size_text = match.group("size")
        raw_name = match.group("name").strip()
        is_dir = raw_name.endswith("/")
        clean_name = raw_name.rstrip("/")
        rows.append(
            {
                "path": f"{current_dir_normalized}/{clean_name}",
                "is_dir": is_dir,
                "size_bytes": 0 if is_dir else _parse_size_bytes(size_text),
            }
        )
    return rows


def remote_rows_to_manifest_index(
    rows: list[dict[str, Any]],
    remote_root: str,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    prefix = _normalize_remote_root(remote_root).rstrip("/") + "/"
    for row in rows:
        if row["is_dir"]:
            continue
        remote_path = _normalize_remote_path(row["path"])
        if not remote_path.startswith(prefix):
            continue
        relative_path = remote_path[len(prefix) :]
        index[relative_path] = {
            "size_bytes": int(row["size_bytes"]),
            "remote_path": remote_path,
        }
    return index


def list_remote_path(bin_path: str | Path, remote_path: str) -> str:
    return run_baidupcs(bin_path, "ls", remote_path).stdout


def mkdir_remote_path(bin_path: str | Path, remote_path: str) -> None:
    run_baidupcs(bin_path, "mkdir", remote_path)


def upload_file(bin_path: str | Path, local_path: Path, remote_parent: str) -> None:
    run_baidupcs(
        bin_path,
        "upload",
        str(local_path),
        remote_parent,
        "--policy",
        "overwrite",
        "-p",
        "1",
        "-l",
        "1",
        "--retry",
        "8",
        "--norapid",
    )


def download_path(bin_path: str | Path, remote_path: str, save_dir: Path) -> None:
    run_baidupcs(bin_path, "config", "set", "-savedir", str(save_dir))
    run_baidupcs(
        bin_path,
        "download",
        remote_path,
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
    )


def download_file_to_local_path(
    bin_path: str | Path,
    remote_path: str,
    *,
    remote_root: str,
    local_path: Path,
) -> None:
    normalized_remote_path = _normalize_remote_path(remote_path)
    normalized_remote_root = _normalize_remote_root(remote_root).rstrip("/")
    if not normalized_remote_path.startswith(f"{normalized_remote_root}/"):
        raise ValueError(
            f"Expected remote_path under remote_root, got remote_path={remote_path!r}, remote_root={remote_root!r}"
        )

    relative_remote_path = normalized_remote_path[len(normalized_remote_root) + 1 :]
    with tempfile.TemporaryDirectory(prefix="large-asset-sync-") as temp_dir:
        temp_root = Path(temp_dir)
        download_path(bin_path, normalized_remote_path, temp_root)
        downloaded_path = temp_root / normalized_remote_root.lstrip("/") / relative_remote_path
        if not downloaded_path.exists():
            raise FileNotFoundError(
                f"Downloaded file not found at expected path: {downloaded_path}"
            )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        move(str(downloaded_path), str(local_path))


def scan_remote_manifest_index(
    bin_path: str | Path,
    manifest_files: Mapping[str, Mapping[str, Any]],
    *,
    remote_root: str,
) -> dict[str, Any]:
    normalized_remote_root = _normalize_remote_root(remote_root)
    manifest_by_parent: dict[str, list[str]] = {}
    for relative_path, file_info in manifest_files.items():
        remote_path_raw = file_info.get("remote_path")
        remote_path = _normalize_remote_path(
            remote_path_raw if remote_path_raw else f"{normalized_remote_root.rstrip('/')}/{relative_path}"
        )
        remote_parent = str(PurePosixPath(remote_path).parent)
        manifest_by_parent.setdefault(remote_parent, []).append(relative_path)

    remote_files: dict[str, dict[str, Any]] = {}
    unknown_remote_state: list[str] = []
    for remote_parent in sorted(manifest_by_parent):
        relative_paths = sorted(manifest_by_parent[remote_parent])
        try:
            rows = parse_baidupcs_ls_output(
                list_remote_path(bin_path, remote_parent),
                current_dir=remote_parent,
            )
        except (subprocess.CalledProcessError, ValueError):
            unknown_remote_state.extend(relative_paths)
            continue

        listed_index = remote_rows_to_manifest_index(rows, remote_root=normalized_remote_root)
        for relative_path in relative_paths:
            listed_file = listed_index.get(relative_path)
            if listed_file is not None:
                remote_files[relative_path] = listed_file

    return {
        "remote_files": remote_files,
        "unknown_remote_state": sorted(unknown_remote_state),
    }

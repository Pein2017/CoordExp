#!/usr/bin/env python3
"""Union-sync local asset trees with Baidu Netdisk through BaiduPCS-Go.

The script implements conservative Git-like large-asset sync semantics:
additions can move both ways, while deletes and overwrites are never automated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


UNSAFE_NAME_CHARS = set('<>:"|?*')


@dataclass(frozen=True)
class RootSpec:
    """Resolved sync root configuration."""

    name: str
    local: Path
    remote: str
    manifest_remote: str
    denylist: Path | None
    exclude: tuple[str, ...]
    settle_seconds: int
    hash_mode: str
    reject_symlinks: bool
    reject_special_files: bool
    reject_unsafe_filenames: bool


@dataclass(frozen=True)
class ScanResult:
    """Local tree scan result."""

    records: list[dict[str, Any]]
    skipped_unsettled: list[str]
    denied: list[str]
    errors: list[str]
    manifest_path: Path


def utc_stamp() -> str:
    """Timestamp string safe for filenames."""

    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_config(path: Path) -> dict[str, Any]:
    """Parsed JSON configuration."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def state_dir(config: dict[str, Any]) -> Path:
    """Absolute state directory."""

    base = Path(config.get("local_base", ".")).resolve()
    return (base / config.get("state_dir", "temp/baidudisk-union-sync")).resolve()


def node_id(config: dict[str, Any]) -> str:
    """Stable local node identifier."""

    base = Path(config.get("local_base", ".")).resolve()
    node_file = (base / config.get("node_id_file", ".baidudisk-union-sync-node-id")).resolve()

    if node_file.exists():
        value = node_file.read_text(encoding="utf-8").strip()
        if value:
            return value

    value = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    node_file.write_text(value + "\n", encoding="utf-8")
    return value


def baidupcs_bin(config: dict[str, Any]) -> str:
    """BaiduPCS-Go executable path."""

    return str(config.get("baidupcs_bin", os.environ.get("BAIDUPCS_BIN", "BaiduPCS-Go")))


def root_spec(config: dict[str, Any], name: str) -> RootSpec:
    """Resolved root specification."""

    roots = config.get("roots", {})
    if name not in roots:
        raise SystemExit(f"Unknown root '{name}'. Available roots: {', '.join(sorted(roots))}")

    base = Path(config.get("local_base", ".")).resolve()
    defaults = config.get("defaults", {})
    raw = roots[name]

    denylist = raw.get("denylist")
    return RootSpec(
        name=name,
        local=(base / raw["local"]).resolve(),
        remote=raw["remote"],
        manifest_remote=raw["manifest_remote"],
        denylist=(base / denylist).resolve() if denylist else None,
        exclude=tuple(raw.get("exclude", [])),
        settle_seconds=int(raw.get("settle_seconds", defaults.get("settle_seconds", 600))),
        hash_mode=str(raw.get("hash_mode", defaults.get("hash_mode", "size"))),
        reject_symlinks=bool(raw.get("reject_symlinks", defaults.get("reject_symlinks", True))),
        reject_special_files=bool(
            raw.get("reject_special_files", defaults.get("reject_special_files", True))
        ),
        reject_unsafe_filenames=bool(
            raw.get("reject_unsafe_filenames", defaults.get("reject_unsafe_filenames", True))
        ),
    )


def run(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess while preserving readable output."""

    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, env=env)


def run_capture(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture output for diagnostics."""

    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def mkdir_chain(bin_path: str, remote_path: str) -> None:
    """Create a remote directory chain, ignoring already-exists failures."""

    current = ""
    for part in remote_path.split("/"):
        if not part:
            continue
        current += "/" + part
        run([bin_path, "mkdir", current], check=False)


def is_excluded(relpath: str, patterns: Iterable[str]) -> bool:
    """Whether a relative path matches any exclude pattern."""

    return any(fnmatch.fnmatch(relpath, pattern) for pattern in patterns)


def has_unsafe_name(relpath: str) -> bool:
    """Whether any path component contains unsafe filename characters."""

    return any(char in UNSAFE_NAME_CHARS for char in relpath)


def load_denylist(path: Path | None) -> list[str]:
    """Denylist glob patterns."""

    if path is None or not path.exists():
        return []

    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped)
    return patterns


def sha256_file(path: Path) -> str:
    """SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_local(config: dict[str, Any], spec: RootSpec) -> ScanResult:
    """Scan a local tree and write a local manifest."""

    if not spec.local.exists():
        raise SystemExit(f"Local root does not exist: {spec.local}")

    current_node = node_id(config)
    now_ns = dt.datetime.now(dt.timezone.utc).timestamp() * 1_000_000_000
    settle_ns = spec.settle_seconds * 1_000_000_000
    deny_patterns = load_denylist(spec.denylist)

    records: list[dict[str, Any]] = []
    skipped_unsettled: list[str] = []
    denied: list[str] = []
    errors: list[str] = []

    for dirpath, dirnames, filenames in os.walk(spec.local, followlinks=False):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not is_excluded(
                (Path(dirpath) / dirname).relative_to(spec.local).as_posix() + "/",
                spec.exclude,
            )
        ]

        for filename in filenames:
            path = Path(dirpath) / filename
            relpath = path.relative_to(spec.local).as_posix()

            if is_excluded(relpath, spec.exclude):
                continue
            if is_excluded(relpath, deny_patterns):
                denied.append(relpath)
                continue
            if spec.reject_unsafe_filenames and has_unsafe_name(relpath):
                errors.append(f"unsafe filename: {relpath}")
                continue
            if path.is_symlink() and spec.reject_symlinks:
                errors.append(f"symlink rejected: {relpath}")
                continue

            stat = path.lstat()
            if not path.is_file():
                if spec.reject_special_files:
                    errors.append(f"special file rejected: {relpath}")
                continue
            if now_ns - stat.st_mtime_ns < settle_ns:
                skipped_unsettled.append(relpath)
                continue

            record: dict[str, Any] = {
                "root": spec.name,
                "relpath": relpath,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "node_id": current_node,
                "scanned_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            if spec.hash_mode == "sha256":
                record["sha256"] = sha256_file(path)
            elif spec.hash_mode != "size":
                errors.append(f"unsupported hash_mode for {relpath}: {spec.hash_mode}")
                continue

            records.append(record)

    manifest_dir = state_dir(config) / "manifests" / "local" / spec.name / current_node
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{utc_stamp()}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item["relpath"]):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    return ScanResult(records, skipped_unsettled, denied, errors, manifest_path)


def read_manifest_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL manifest records."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def signature(record: dict[str, Any]) -> tuple[Any, ...]:
    """Conflict comparison signature."""

    if record.get("sha256"):
        return (record.get("size"), record.get("sha256"))
    return (record.get("size"),)


def records_by_path(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Latest record per relative path."""

    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        by_path[record["relpath"]] = record
    return by_path


def download_remote_manifests(config: dict[str, Any], spec: RootSpec) -> Path:
    """Download remote manifests into local state, if present."""

    bin_path = baidupcs_bin(config)
    target = state_dir(config) / "remote-manifests" / spec.name
    target.mkdir(parents=True, exist_ok=True)

    completed = run_capture(
        [
            bin_path,
            "download",
            spec.manifest_remote,
            "--fullpath",
            "--mode",
            str(config.get("defaults", {}).get("download_mode", "locate")),
            "-p",
            "4",
            "-l",
            "2",
            "--retry",
            str(config.get("defaults", {}).get("retry", 8)),
            "--mtime",
            "--saveto",
            str(target),
        ],
        check=False,
    )
    if completed.returncode != 0:
        print(f"warning: could not download remote manifests yet: {spec.manifest_remote}", file=sys.stderr)
        print(completed.stdout[-2000:], file=sys.stderr)

    return target


def remote_manifest_records(config: dict[str, Any], spec: RootSpec) -> list[dict[str, Any]]:
    """Merged records from locally cached remote manifests."""

    root = download_remote_manifests(config, spec)
    records: list[dict[str, Any]] = []
    for path in root.rglob("*.jsonl"):
        try:
            records.extend(read_manifest_records(path))
        except Exception as exc:  # noqa: BLE001 - keep damaged manifests visible.
            print(f"warning: failed to read manifest {path}: {exc}", file=sys.stderr)
    return records


def classify(
    local_records: list[dict[str, Any]],
    remote_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify local/remote differences from manifests."""

    local = records_by_path(local_records)
    remote = records_by_path(remote_records)

    upload_new: list[str] = []
    download_new: list[str] = []
    same: list[str] = []
    conflicts: list[dict[str, Any]] = []

    for relpath, local_record in local.items():
        remote_record = remote.get(relpath)
        if remote_record is None:
            upload_new.append(relpath)
        elif signature(local_record) == signature(remote_record):
            same.append(relpath)
        else:
            conflicts.append({"relpath": relpath, "local": local_record, "remote": remote_record})

    for relpath in remote:
        if relpath not in local:
            download_new.append(relpath)

    return {
        "local_count": len(local),
        "remote_manifest_count": len(remote),
        "upload_new": sorted(upload_new),
        "download_new": sorted(download_new),
        "same": sorted(same),
        "conflicts": conflicts,
    }


def print_summary(summary: dict[str, Any], scan: ScanResult | None = None) -> None:
    """Emit a compact JSON summary."""

    compact = {
        "local_count": summary.get("local_count"),
        "remote_manifest_count": summary.get("remote_manifest_count"),
        "upload_new_count": len(summary.get("upload_new", [])),
        "download_new_count": len(summary.get("download_new", [])),
        "same_count": len(summary.get("same", [])),
        "conflict_count": len(summary.get("conflicts", [])),
    }
    if scan is not None:
        compact.update(
            {
                "manifest_path": str(scan.manifest_path),
                "skipped_unsettled_count": len(scan.skipped_unsettled),
                "denied_count": len(scan.denied),
                "error_count": len(scan.errors),
            }
        )

    print(json.dumps(compact, indent=2, sort_keys=True))

    for conflict in summary.get("conflicts", [])[:20]:
        print(f"CONFLICT {conflict['relpath']}", file=sys.stderr)


def upload_manifest(config: dict[str, Any], spec: RootSpec, manifest_path: Path) -> None:
    """Upload a unique manifest file."""

    bin_path = baidupcs_bin(config)
    current_node = node_id(config)
    remote_dir = f"{spec.manifest_remote.rstrip('/')}/{current_node}"
    mkdir_chain(bin_path, remote_dir)
    run([bin_path, "upload", str(manifest_path), remote_dir, "--policy", "skip"])


def upload_root(config: dict[str, Any], spec: RootSpec) -> None:
    """Upload local root with skip-existing semantics."""

    defaults = config.get("defaults", {})
    bin_path = baidupcs_bin(config)
    mkdir_chain(bin_path, spec.remote)
    parent = str(Path(spec.remote).parent).replace("\\", "/")
    if parent == ".":
        parent = "/"

    cmd = [
        bin_path,
        "upload",
        str(spec.local),
        parent,
        "--policy",
        "skip",
        "-p",
        str(defaults.get("upload_file_threads", 1)),
        "-l",
        str(defaults.get("upload_parallel_files", 8)),
        "--retry",
        str(defaults.get("retry", 8)),
    ]
    if bool(defaults.get("no_rapid", True)):
        cmd.append("--norapid")
    run(cmd)


def remote_tail_path(staging: Path, remote: str) -> Path | None:
    """Find the staged path corresponding to a remote absolute path."""

    parts = [part for part in remote.split("/") if part]
    if not parts:
        return staging

    leaf = parts[-1]
    for candidate in staging.rglob(leaf):
        if not candidate.is_dir():
            continue
        candidate_parts = candidate.parts
        if len(candidate_parts) >= len(parts) and list(candidate_parts[-len(parts) :]) == parts:
            return candidate
    return None


def download_root_to_staging(config: dict[str, Any], spec: RootSpec) -> Path:
    """Download remote root to reusable local staging without overwrite."""

    defaults = config.get("defaults", {})
    bin_path = baidupcs_bin(config)
    staging = state_dir(config) / "download-staging" / spec.name
    staging.mkdir(parents=True, exist_ok=True)

    run(
        [
            bin_path,
            "download",
            spec.remote,
            "--fullpath",
            "--mode",
            str(defaults.get("download_mode", "locate")),
            "-p",
            str(defaults.get("download_threads", 8)),
            "-l",
            str(defaults.get("download_parallel_files", 4)),
            "--retry",
            str(defaults.get("retry", 8)),
            "--mtime",
            "--saveto",
            str(staging),
        ]
    )

    staged = remote_tail_path(staging, spec.remote)
    if staged is None:
        raise SystemExit(f"Could not locate downloaded remote path {spec.remote} under {staging}")
    return staged


def merge_staging_into_local(staged: Path, spec: RootSpec) -> None:
    """Merge staged files locally without overwriting existing files."""

    spec.local.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", "--ignore-existing", f"{staged}/", f"{spec.local}/"])


def command_doctor(config: dict[str, Any]) -> None:
    """Check tool availability and BaiduPCS-Go account access."""

    bin_path = baidupcs_bin(config)
    if shutil.which(bin_path) is None and not Path(bin_path).exists():
        raise SystemExit(f"BaiduPCS-Go binary not found: {bin_path}")
    if shutil.which("rsync") is None:
        raise SystemExit("rsync not found")

    run([bin_path, "quota"], check=False)
    run([bin_path, "pwd"], check=False)
    print(f"node_id={node_id(config)}")
    print(f"state_dir={state_dir(config)}")


def command_scan(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Run local scan."""

    spec = root_spec(config, args.root)
    scan = scan_local(config, spec)
    if scan.errors:
        for error in scan.errors[:50]:
            print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(f"scan failed with {len(scan.errors)} error(s)")

    print(
        json.dumps(
            {
                "records": len(scan.records),
                "manifest_path": str(scan.manifest_path),
                "skipped_unsettled": len(scan.skipped_unsettled),
                "denied": len(scan.denied),
            },
            indent=2,
            sort_keys=True,
        )
    )


def status_for(config: dict[str, Any], root: str) -> tuple[RootSpec, ScanResult, dict[str, Any]]:
    """Scan local and compare with remote manifests."""

    spec = root_spec(config, root)
    remote_records = remote_manifest_records(config, spec)
    scan = scan_local(config, spec)
    if scan.errors:
        for error in scan.errors[:50]:
            print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(f"scan failed with {len(scan.errors)} error(s)")
    return spec, scan, classify(scan.records, remote_records)


def command_status(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Show sync status."""

    _spec, scan, summary = status_for(config, args.root)
    print_summary(summary, scan)


def command_push(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Push local additions to remote."""

    spec, scan, summary = status_for(config, args.root)
    print_summary(summary, scan)
    if summary["conflicts"]:
        raise SystemExit("push stopped because conflicts exist")
    if not args.apply:
        print("dry-run: add --apply to upload")
        return

    upload_root(config, spec)
    upload_manifest(config, spec, scan.manifest_path)


def command_pull(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Pull remote additions into local tree."""

    spec, scan, summary = status_for(config, args.root)
    print_summary(summary, scan)
    if summary["conflicts"]:
        raise SystemExit("pull stopped because conflicts exist")
    if not args.apply:
        print("dry-run: add --apply to download")
        return

    staged = download_root_to_staging(config, spec)
    merge_staging_into_local(staged, spec)


def command_sync(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Pull then push."""

    command_pull(config, args)
    command_push(config, args)


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON config path")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")

    scan = subparsers.add_parser("scan")
    scan.add_argument("root")

    status = subparsers.add_parser("status")
    status.add_argument("root")

    push = subparsers.add_parser("push")
    push.add_argument("root")
    push.add_argument("--apply", action="store_true")

    pull = subparsers.add_parser("pull")
    pull.add_argument("root")
    pull.add_argument("--apply", action="store_true")

    sync = subparsers.add_parser("sync")
    sync.add_argument("root")
    sync.add_argument("--apply", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "doctor":
        command_doctor(config)
    elif args.command == "scan":
        command_scan(config, args)
    elif args.command == "status":
        command_status(config, args)
    elif args.command == "push":
        command_push(config, args)
    elif args.command == "pull":
        command_pull(config, args)
    elif args.command == "sync":
        command_sync(config, args)
    else:
        parser.error(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

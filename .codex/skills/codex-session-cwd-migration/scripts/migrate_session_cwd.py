#!/usr/bin/env python3
"""Safely rebind one existing Codex session to a new cwd."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

try:
    import websockets
except ImportError as exc:  # pragma: no cover - environment error
    raise SystemExit("websockets is required; run this script in the Codex Python environment") from exc


THREAD_ID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
PROTECTED_COLUMNS = (
    "git_sha",
    "git_branch",
    "git_origin_url",
    "model_provider",
    "model",
    "reasoning_effort",
    "approval_mode",
    "sandbox_policy",
    "rollout_path",
    "title",
    "project_id",
    "source",
    "thread_source",
    "cli_version",
    "history_mode",
    "name",
    "agent_nickname",
    "agent_role",
    "memory_mode",
)


class MigrationError(RuntimeError):
    pass


def normalize_thread_id(value: str) -> str:
    value = value.strip()
    prefix = "codex://threads/"
    if value.startswith(prefix):
        value = value[len(prefix) :]
    if not THREAD_ID_RE.fullmatch(value):
        raise MigrationError(f"invalid Codex thread id: {value!r}")
    return value


def absolute_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise MigrationError(f"target cwd is not an existing directory: {path}")
    return path.resolve()


def load_thread(db: Path, thread_id: str) -> dict[str, Any]:
    if not db.is_file():
        raise MigrationError(f"state database does not exist: {db}")
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
    if row is None:
        raise MigrationError(f"thread not found in state database: {thread_id}")
    return dict(row)


def quick_check(db: Path) -> str:
    with sqlite3.connect(db) as connection:
        result = connection.execute("PRAGMA quick_check").fetchone()[0]
    if result != "ok":
        raise MigrationError(f"SQLite quick_check failed: {result}")
    return result


def protected_diffs(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        column: [before.get(column), after.get(column)]
        for column in PROTECTED_COLUMNS
        if column in before and before.get(column) != after.get(column)
    }


def backup_dir(root: Path, thread_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(100):
        extra = "" if suffix == 0 else f"-{suffix}"
        path = root / f"{stamp}-session-{thread_id}-cwd-rebind{extra}"
        try:
            path.mkdir()
            (path / "rollouts").mkdir()
            return path
        except FileExistsError:
            continue
    raise MigrationError(f"could not allocate backup directory under {root}")


def copy_locked(source: Path, destination: Path) -> None:
    with source.open("rb") as source_file:
        with contextlib.suppress(OSError):
            fcntl.flock(source_file.fileno(), fcntl.LOCK_SH)
        data = source_file.read()
        with contextlib.suppress(OSError):
            fcntl.flock(source_file.fileno(), fcntl.LOCK_UN)
    with destination.open("wb") as destination_file:
        destination_file.write(data)
        destination_file.flush()
        os.fsync(destination_file.fileno())
    shutil.copystat(source, destination)


def create_backup(
    codex_home: Path,
    db: Path,
    thread_id: str,
    row: dict[str, Any],
    target: Path,
    backup_root: Path | None = None,
) -> Path:
    backup = backup_dir(backup_root or codex_home / "migration-backups", thread_id)
    source_db = sqlite3.connect(db)
    destination_db = sqlite3.connect(backup / "state_5.sqlite")
    try:
        source_db.backup(destination_db)
    finally:
        destination_db.close()
        source_db.close()
    quick_check(backup / "state_5.sqlite")

    session_files = sorted((codex_home / "sessions").rglob(f"*{thread_id}*.jsonl"))
    if not session_files:
        raise MigrationError(f"no rollout files found for {thread_id}")
    for rollout in session_files:
        copy_locked(rollout, backup / "rollouts" / rollout.name)

    git_status = backup / "target-git-status.txt"
    try:
        result = subprocess.run(
            ["git", "-C", str(target), "status", "--short", "--branch"],
            check=False,
            capture_output=True,
            text=True,
        )
        git_status.write_text(result.stdout or result.stderr, encoding="utf-8")
    except OSError as exc:
        git_status.write_text(f"git status unavailable: {exc}\n", encoding="utf-8")

    manifest = {
        "thread_id": thread_id,
        "source_cwd": row.get("cwd"),
        "target_cwd": str(target),
        "rollout_path": row.get("rollout_path"),
        "rollout_files": [str(path) for path in session_files],
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protected_metadata": {column: row.get(column) for column in PROTECTED_COLUMNS if column in row},
    }
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return backup


def status_type(thread: dict[str, Any]) -> str | None:
    status = thread.get("status")
    return status.get("type") if isinstance(status, dict) else status


def live_summary(result: dict[str, Any]) -> dict[str, Any]:
    thread = result.get("thread") or {}
    return {
        "cwd": result.get("cwd") or thread.get("cwd"),
        "runtime_first_root": (result.get("runtimeWorkspaceRoots") or [None])[0],
        "status": status_type(thread),
        "git_info": thread.get("gitInfo"),
        "model": result.get("model"),
        "approval_policy": result.get("approvalPolicy"),
        "reasoning_effort": result.get("reasoningEffort"),
    }


async def live_migrate(socket: Path, thread_id: str, target: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not socket.exists():
        raise MigrationError(f"app-server control socket does not exist: {socket}")

    async with websockets.unix_connect(str(socket), max_size=None, compression=None) as raw_websocket:
        websocket: Any = raw_websocket

        async def call(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
            await websocket.send(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}))
            while True:
                message = json.loads(await websocket.recv())
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    raise MigrationError(f"RPC {method} failed: {message['error']}")
                return message.get("result") or {}

        await call(
            1,
            "initialize",
            {"clientInfo": {"name": "codex-session-cwd-migration", "version": "1.0"}, "capabilities": {"experimentalApi": True}},
        )
        await websocket.send(json.dumps({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
        before = cast(dict[str, Any], await call(2, "thread/resume", {"threadId": thread_id, "excludeTurns": True}))
        if live_summary(before).get("cwd") == str(target):
            update: dict[str, Any] = {"skipped": "already_at_target"}
        else:
            update = cast(dict[str, Any], await call(3, "thread/settings/update", {"threadId": thread_id, "cwd": str(target)}))
        after = cast(dict[str, Any], await call(4, "thread/resume", {"threadId": thread_id, "excludeTurns": True}))
    return live_summary(before), update, live_summary(after)


def event_types(data: bytes) -> list[str]:
    result = []
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MigrationError("rollout has an appended non-JSON line") from exc
        result.append(value.get("payload", {}).get("type") or value.get("type") or "unknown")
    return result


def replace_json_string_token(line: bytes, old: str, new: str) -> bytes:
    for ensure_ascii in (False, True):
        old_token = json.dumps(old, ensure_ascii=ensure_ascii).encode()
        if line.count(old_token):
            if line.count(old_token) != 1:
                raise MigrationError("rollout first line has an unexpected cwd token count")
            return line.replace(old_token, json.dumps(new, ensure_ascii=ensure_ascii).encode(), 1)
    raise MigrationError("rollout first line has no cwd token")


def reconcile_rollout(path: Path, backup_path: Path, thread_id: str, target: Path) -> dict[str, Any]:
    with path.open("r+b") as current_file:
        fcntl.flock(current_file.fileno(), fcntl.LOCK_EX)
        current = current_file.read()
        backup = backup_path.read_bytes()
        separator = current.find(b"\n")
        backup_separator = backup.find(b"\n")
        if separator < 0 or backup_separator < 0:
            raise MigrationError(f"rollout has no first-line separator: {path}")
        current_first, current_suffix = current[:separator], current[separator + 1 :]
        backup_first, backup_suffix = backup[:backup_separator], backup[backup_separator + 1 :]
        try:
            current_obj = json.loads(current_first)
            backup_obj = json.loads(backup_first)
        except json.JSONDecodeError as exc:
            raise MigrationError(f"rollout first line is not JSON: {path}") from exc
        current_payload = current_obj.get("payload", {})
        backup_payload = backup_obj.get("payload", {})
        if current_obj.get("type") != "session_meta" or backup_obj.get("type") != "session_meta":
            raise MigrationError(f"unexpected rollout first-line type: {path}")
        if current_payload.get("id") != thread_id or backup_payload.get("id") != thread_id:
            raise MigrationError(f"rollout session id mismatch: {path}")
        original_cwd = backup_payload.get("cwd")
        current_cwd = current_payload.get("cwd")
        if not isinstance(original_cwd, str) or not isinstance(current_cwd, str):
            raise MigrationError(f"rollout session_meta has no cwd: {path}")
        normalized = json.loads(json.dumps(current_obj))
        normalized["payload"]["cwd"] = original_cwd
        if normalized != backup_obj:
            raise MigrationError("rollout first-line metadata drifted beyond cwd")
        if not current_suffix.startswith(backup_suffix):
            raise MigrationError("rollout history is not an exact backup prefix")
        appended = current_suffix[len(backup_suffix) :]
        appended_types = event_types(appended)
        changed = current_cwd != str(target)
        if changed:
            new_first = replace_json_string_token(current_first, current_cwd, str(target))
            if json.loads(new_first).get("payload", {}).get("cwd") != str(target):
                raise MigrationError("rollout cwd replacement failed")
            current_file.seek(0)
            current_file.write(new_first + b"\n" + current_suffix)
            current_file.truncate()
            current_file.flush()
            os.fsync(current_file.fileno())
        fcntl.flock(current_file.fileno(), fcntl.LOCK_UN)
    return {"path": str(path), "changed": changed, "history_prefix_exact": True, "appended_event_types": appended_types}


def reconcile_sqlite(db: Path, thread_id: str, target: Path, allowed_old_cwds: set[str]) -> None:
    with sqlite3.connect(db) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT cwd FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if row is None:
            raise MigrationError(f"thread disappeared from state database: {thread_id}")
        current_cwd = row[0]
        if current_cwd != str(target):
            if current_cwd not in allowed_old_cwds:
                raise MigrationError(f"SQLite cwd changed unexpectedly: {current_cwd!r}")
            updated = connection.execute(
                "UPDATE threads SET cwd = ? WHERE id = ? AND cwd = ?",
                (str(target), thread_id, current_cwd),
            ).rowcount
            if updated != 1:
                raise MigrationError("SQLite cwd update was lost to a concurrent change")
        connection.commit()


def final_rollout_check(path: Path, backup_path: Path, thread_id: str, target: Path) -> dict[str, Any]:
    current = path.read_bytes()
    backup = backup_path.read_bytes()
    separator = current.find(b"\n")
    backup_separator = backup.find(b"\n")
    if separator < 0 or backup_separator < 0:
        raise MigrationError(f"rollout has no first-line separator: {path}")
    current_obj = json.loads(current[:separator])
    backup_obj = json.loads(backup[:backup_separator])
    if current_obj.get("payload", {}).get("id") != thread_id:
        raise MigrationError("final rollout session id mismatch")
    if current_obj.get("payload", {}).get("cwd") != str(target):
        raise MigrationError("final rollout cwd mismatch")
    for key, value in backup_obj.get("payload", {}).items():
        if key != "cwd" and current_obj.get("payload", {}).get(key) != value:
            raise MigrationError(f"final rollout metadata changed beyond cwd: {key}")
    suffix = current[separator + 1 :]
    backup_suffix = backup[backup_separator + 1 :]
    if not suffix.startswith(backup_suffix):
        raise MigrationError("final rollout history prefix mismatch")
    return {
        "path": str(path),
        "history_prefix_exact": True,
        "appended_event_types": event_types(suffix[len(backup_suffix) :]),
    }


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", required=True, help="UUID or codex://threads/<UUID>")
    parser.add_argument("--cwd", required=True, help="existing absolute target directory")
    parser.add_argument("--codex-home", default=os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    parser.add_argument("--state-db")
    parser.add_argument("--control-socket")
    parser.add_argument("--backup-root")
    parser.add_argument("--timeout", type=float, default=30.0, help="RPC timeout in seconds (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="validate persisted state without mutating anything")
    parser.add_argument("--json", action="store_true", help="emit one JSON receipt")
    return vars(parser.parse_args())


def main() -> int:
    args = parse_args()
    try:
        thread_id = normalize_thread_id(args["thread_id"])
        target = absolute_dir(args["cwd"])
        codex_home = Path(args["codex_home"]).expanduser().resolve()
        db = Path(args["state_db"] or codex_home / "state_5.sqlite").expanduser().resolve()
        socket = Path(args["control_socket"] or codex_home / "app-server-control" / "app-server-control.sock").expanduser().resolve()
        row_before = load_thread(db, thread_id)
        authoritative = Path(row_before["rollout_path"]).expanduser().resolve()
        if not authoritative.is_file():
            raise MigrationError(f"authoritative rollout does not exist: {authoritative}")
        if args["dry_run"]:
            result = {
                "status": "dry_run",
                "thread_id": thread_id,
                "source_cwd": row_before.get("cwd"),
                "target_cwd": str(target),
                "authoritative_rollout": str(authoritative),
                "state_db": str(db),
                "target_is_git": (target / ".git").exists(),
                "sqlite_quick_check": quick_check(db),
            }
        else:
            backup_root = Path(args["backup_root"]).expanduser().resolve() if args["backup_root"] else None
            backup = create_backup(codex_home, db, thread_id, row_before, target, backup_root)
            live_before, update_result, live_after = asyncio.run(
                asyncio.wait_for(live_migrate(socket, thread_id, target), timeout=args["timeout"])
            )
            allowed_old_cwds = {
                value
                for value in (row_before.get("cwd"), live_before.get("cwd"))
                if isinstance(value, str)
            }
            reconcile_sqlite(db, thread_id, target, allowed_old_cwds)
            row_after = load_thread(db, thread_id)
            diffs = protected_diffs(row_before, row_after)
            if diffs:
                raise MigrationError(f"protected SQLite metadata changed: {diffs}")
            rollout_result = reconcile_rollout(
                authoritative,
                backup / "rollouts" / authoritative.name,
                thread_id,
                target,
            )
            row_final = load_thread(db, thread_id)
            final_rollout = final_rollout_check(
                authoritative,
                backup / "rollouts" / authoritative.name,
                thread_id,
                target,
            )
            if row_final.get("cwd") != str(target):
                raise MigrationError("final SQLite cwd mismatch")
            if live_after.get("cwd") != str(target) or live_after.get("runtime_first_root") != str(target):
                raise MigrationError(f"final live cwd/runtime root mismatch: {live_after}")
            for key in ("git_info", "model", "approval_policy", "reasoning_effort"):
                if live_before.get(key) != live_after.get(key):
                    raise MigrationError(f"live metadata changed beyond cwd: {key}")
            result = {
                "status": "migrated",
                "thread_id": thread_id,
                "source_cwd": live_before.get("cwd") or row_before.get("cwd"),
                "target_cwd": str(target),
                "backup": str(backup),
                "live_before": live_before,
                "live_after": live_after,
                "rpc_update_result": update_result,
                "sqlite_quick_check": quick_check(db),
                "protected_metadata_unchanged": True,
                "rollout": rollout_result,
                "final_rollout": final_rollout,
            }
        output = json.dumps(result, ensure_ascii=False, indent=None if args["json"] else 2, sort_keys=True)
        print(output)
        return 0
    except (MigrationError, OSError, sqlite3.Error, json.JSONDecodeError, asyncio.TimeoutError, websockets.WebSocketException) as exc:
        print(f"migration not accepted: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

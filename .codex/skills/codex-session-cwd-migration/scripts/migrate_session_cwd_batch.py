#!/usr/bin/env python3
"""Rebind many sessions with missing worktree cwds in one guarded batch."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import importlib.util
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SINGLE_PATH = Path(__file__).with_name("migrate_session_cwd.py")
SPEC = importlib.util.spec_from_file_location("codex_session_cwd_migration", SINGLE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - packaging failure
    raise SystemExit(f"cannot load single-session helper: {SINGLE_PATH}")
SINGLE: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SINGLE)

TARGET = "/data/CoordExp/.worktrees/research-probes"
PROTECTED_COLUMNS = SINGLE.PROTECTED_COLUMNS


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", default=TARGET, help="target cwd for every selected session")
    parser.add_argument("--root", action="append", required=True, help="missing .worktrees root to include; repeatable")
    parser.add_argument("--codex-home", default=os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    parser.add_argument("--state-db")
    parser.add_argument("--control-socket")
    parser.add_argument("--backup-root")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return vars(parser.parse_args())


def load_candidates(db: Path, roots: set[str], target: Path) -> tuple[list[dict[str, Any]], int, int]:
    if not target.is_dir():
        raise SINGLE.MigrationError(f"target cwd is not an existing directory: {target}")
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM threads WHERE cwd LIKE '/data/CoordExp/.worktrees/%' ORDER BY updated_at, id"
        ).fetchall()
    candidates = []
    skipped_existing = 0
    skipped_subagents = 0
    for row in rows:
        item = dict(row)
        cwd = Path(str(item["cwd"]))
        parts = cwd.parts
        if ".worktrees" not in parts:
            continue
        root = parts[parts.index(".worktrees") + 1] if len(parts) > parts.index(".worktrees") + 1 else ""
        if root not in roots:
            continue
        if row["agent_role"] or row["agent_path"]:
            skipped_subagents += 1
            continue
        if cwd.is_dir():
            skipped_existing += 1
            continue
        if item.get("cwd") == str(target):
            raise SINGLE.MigrationError(f"target unexpectedly selected as missing cwd: {target}")
        candidates.append(item)
    if not candidates:
        raise SINGLE.MigrationError("no missing-cwd sessions matched the requested roots")
    return candidates, skipped_existing, skipped_subagents


def allocate_backup(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(100):
        extra = "" if suffix == 0 else f"-{suffix}"
        path = root / f"{stamp}-batch-research-cwd-rebind{extra}"
        try:
            path.mkdir()
            (path / "rollouts").mkdir()
            return path
        except FileExistsError:
            continue
    raise SINGLE.MigrationError(f"could not allocate batch backup under {root}")


def backup_state(
    codex_home: Path,
    db: Path,
    candidates: list[dict[str, Any]],
    target: Path,
    backup_root: Path | None,
) -> Path:
    backup = allocate_backup(backup_root or codex_home / "migration-backups")
    source_db = sqlite3.connect(db)
    destination_db = sqlite3.connect(backup / "state_5.sqlite")
    try:
        source_db.backup(destination_db)
    finally:
        destination_db.close()
        source_db.close()
    SINGLE.quick_check(backup / "state_5.sqlite")

    ids = {str(item["id"]) for item in candidates}
    files_by_id: dict[str, list[Path]] = defaultdict(list)
    rollout_roots = [codex_home / "sessions", codex_home / "archived_sessions"]
    for item in candidates:
        thread_id = str(item["id"])
        authoritative = Path(str(item["rollout_path"])).expanduser().resolve()
        if authoritative.is_file():
            files_by_id[thread_id].append(authoritative)
    for rollout_root in rollout_roots:
        if not rollout_root.is_dir():
            continue
        for path in rollout_root.rglob("*.jsonl"):
            matched = [thread_id for thread_id in ids if thread_id in path.name]
            for thread_id in matched:
                if path not in files_by_id[thread_id]:
                    files_by_id[thread_id].append(path)
    missing = sorted(ids - files_by_id.keys())
    if missing:
        raise SINGLE.MigrationError(f"no rollout files for {len(missing)} selected sessions; first={missing[0]}")
    copied: set[str] = set()
    for paths in files_by_id.values():
        for path in paths:
            destination = backup / "rollouts" / path.name
            if str(path) in copied:
                continue
            SINGLE.copy_locked(path, destination)
            copied.add(str(path))

    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_cwd": str(target),
        "session_count": len(candidates),
        "rollout_file_count": len(copied),
        "sessions": [
            {
                key: item.get(key)
                for key in ("id", "cwd", "rollout_path", "git_sha", "git_branch", "model", "reasoning_effort", "approval_mode", "sandbox_policy", "updated_at")
            }
            for item in candidates
        ],
    }
    (backup / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return backup


def identity_diffs(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[Any]]:
    return SINGLE.protected_diffs(before, after)


async def migrate_live(
    socket: Path,
    candidates: list[dict[str, Any]],
    target: Path,
    receipt_path: Path,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import websockets

    success: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not socket.exists():
        raise SINGLE.MigrationError(f"app-server control socket does not exist: {socket}")
    async with websockets.unix_connect(str(socket), max_size=None, compression=None) as raw_websocket:
        websocket: Any = raw_websocket
        async def call(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
            await websocket.send(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}))
            while True:
                message = json.loads(await websocket.recv())
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    raise SINGLE.MigrationError(f"RPC {method} failed: {message['error']}")
                return message.get("result") or {}

        await call(1, "initialize", {"clientInfo": {"name": "codex-session-cwd-migration-batch", "version": "1.0"}, "capabilities": {"experimentalApi": True}})
        await websocket.send(json.dumps({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
        request_id = 2
        with receipt_path.open("w", encoding="utf-8") as receipts:
            for item in candidates:
                thread_id = str(item["id"])
                record: dict[str, Any] = {"thread_id": thread_id, "source_cwd": item.get("cwd")}
                try:
                    before_result = await call(request_id, "thread/resume", {"threadId": thread_id, "excludeTurns": True})
                    request_id += 1
                    before = SINGLE.live_summary(before_result)
                    record["live_before"] = before
                    if before.get("cwd") == str(target):
                        record["rpc_update"] = {"skipped": "already_at_target"}
                    else:
                        record["rpc_update"] = await call(request_id, "thread/settings/update", {"threadId": thread_id, "cwd": str(target)})
                        request_id += 1
                    after_result = await call(request_id, "thread/resume", {"threadId": thread_id, "excludeTurns": True})
                    request_id += 1
                    after = SINGLE.live_summary(after_result)
                    record["live_after"] = after
                    if after.get("cwd") != str(target) or after.get("runtime_first_root") != str(target):
                        raise SINGLE.MigrationError(f"live cwd/runtime root mismatch: {after}")
                    for key in ("git_info", "model", "approval_policy", "reasoning_effort"):
                        if before.get(key) != after.get(key):
                            raise SINGLE.MigrationError(f"live metadata changed beyond cwd: {key}")
                    record["status"] = "live_ok"
                    success.append(record)
                except Exception as exc:  # keep independent sessions moving; receipt marks the failure
                    record["status"] = "live_failed"
                    record["error"] = str(exc)
                    failures.append(record)
                receipts.write(json.dumps(record, ensure_ascii=False) + "\n")
                receipts.flush()
    return success, failures


def reconcile_persistence(
    db: Path,
    candidates_by_id: dict[str, dict[str, Any]],
    live_success: list[dict[str, Any]],
    backup: Path,
    target: Path,
    receipt_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    persisted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        for record in live_success:
            thread_id = record["thread_id"]
            before = candidates_by_id[thread_id]
            row = connection.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
            if row is None:
                failures.append({"thread_id": thread_id, "status": "sqlite_failed", "error": "thread disappeared"})
                continue
            after = dict(row)
            diffs = identity_diffs(before, after)
            if diffs:
                failures.append({"thread_id": thread_id, "status": "sqlite_failed", "error": f"protected metadata changed: {diffs}"})
                continue
            current_cwd = str(after.get("cwd"))
            allowed = {str(before.get("cwd")), str(record.get("live_before", {}).get("cwd"))}
            if current_cwd != str(target):
                if current_cwd not in allowed:
                    failures.append({"thread_id": thread_id, "status": "sqlite_failed", "error": f"unexpected cwd: {current_cwd}"})
                    continue
                if connection.execute("UPDATE threads SET cwd = ? WHERE id = ? AND cwd = ?", (str(target), thread_id, current_cwd)).rowcount != 1:
                    failures.append({"thread_id": thread_id, "status": "sqlite_failed", "error": "cwd update lost to concurrent change"})
                    continue
            persisted.append(record)
        connection.commit()

    with receipt_path.open("a", encoding="utf-8") as receipts:
        for record in list(persisted):
            thread_id = record["thread_id"]
            before = candidates_by_id[thread_id]
            authoritative = Path(str(before["rollout_path"])).resolve()
            backup_rollout = backup / "rollouts" / authoritative.name
            try:
                rollout = SINGLE.reconcile_rollout(authoritative, backup_rollout, thread_id, target)
                final = SINGLE.final_rollout_check(authoritative, backup_rollout, thread_id, target)
                record["rollout"] = rollout
                record["final_rollout"] = final
                record["status"] = "migrated"
                receipts.write(json.dumps(record, ensure_ascii=False) + "\n")
                receipts.flush()
            except Exception as exc:
                record["status"] = "rollout_failed"
                record["error"] = str(exc)
                failures.append(record)
                receipts.write(json.dumps(record, ensure_ascii=False) + "\n")
                receipts.flush()
    return persisted, failures


def main() -> int:
    args = parse_args()
    try:
        target = Path(args["cwd"]).expanduser().resolve()
        codex_home = Path(args["codex_home"]).expanduser().resolve()
        db = Path(args["state_db"] or codex_home / "state_5.sqlite").expanduser().resolve()
        socket = Path(args["control_socket"] or codex_home / "app-server-control" / "app-server-control.sock").expanduser().resolve()
        roots = set(args["root"])
        candidates, skipped_existing, skipped_subagents = load_candidates(db, roots, target)
        result: dict[str, Any] = {
            "status": "dry_run" if args["dry_run"] else "prepared",
            "target_cwd": str(target),
            "roots": sorted(roots),
            "selected_missing_cwd_sessions": len(candidates),
            "skipped_existing_cwd_sessions": skipped_existing,
            "skipped_subagent_sessions": skipped_subagents,
        }
        if args["dry_run"]:
            print(json.dumps(result, ensure_ascii=False, indent=None if args["json"] else 2, sort_keys=True))
            return 0

        backup_root = Path(args["backup_root"]).expanduser().resolve() if args["backup_root"] else None
        backup = backup_state(codex_home, db, candidates, target, backup_root)
        live_receipts = backup / "live-receipts.jsonl"
        live_success, live_failures = asyncio.run(
            asyncio.wait_for(migrate_live(socket, candidates, target, live_receipts, args["timeout"]), timeout=args["timeout"] * max(2, len(candidates) * 4))
        )
        by_id = {str(item["id"]): item for item in candidates}
        persisted, persistence_failures = reconcile_persistence(db, by_id, live_success, backup, target, live_receipts)
        result.update({
            "status": "migrated" if not live_failures and not persistence_failures else "partial_hold",
            "backup": str(backup),
            "live_ok": len(live_success),
            "live_failed": len(live_failures),
            "persisted": len(persisted),
            "persistence_failed": len(persistence_failures),
            "receipt": str(live_receipts),
            "sqlite_quick_check": SINGLE.quick_check(db),
            "failure_ids": [r["thread_id"] for r in live_failures + persistence_failures],
        })
        (backup / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=None if args["json"] else 2, sort_keys=True))
        return 0 if result["status"] == "migrated" else 2
    except (SINGLE.MigrationError, OSError, sqlite3.Error, asyncio.TimeoutError, ValueError) as exc:
        print(f"batch migration not accepted: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

---
name: codex-session-cwd-migration
description: Rebind an existing Codex session to an existing directory in place while preserving its ID, history, Git identity, model, permissions, and other metadata.
---

# Codex Session Cwd Migration

Use this skill only when the user explicitly asks to change an existing session's
cwd/worktree. Do not create a replacement session, infer a branch change, or
rewrite old rollout contexts.

## Command

Run the bundled script:

```bash
python /data/CoordExp/.agents/skills/codex-session-cwd-migration/scripts/migrate_session_cwd.py \
  --thread-id codex://threads/<session-id> \
  --cwd /absolute/target/directory
```

Use `--codex-home`, `--state-db`, `--control-socket`, or `--backup-root` only
when the session belongs to a non-default Codex installation or backup policy.
Use `--dry-run` to validate the session, target, and persisted records without
changing anything.

## Safety contract

The script:

1. validates the target directory and the exact session row;
2. creates a rollback backup of the SQLite database and every rollout file
   containing the session id;
3. sends only `thread/settings/update` with `{threadId, cwd}` over the local
   app-server control socket, then reads `thread/resume`;
4. reconciles only the SQLite `cwd` and the authoritative rollout's first-line
   `session_meta.payload.cwd`; historical rollout bytes remain an exact prefix;
5. verifies the target is the first `runtimeWorkspaceRoots` entry, checks
   `PRAGMA quick_check`, and confirms protected Git/model/permission metadata.

It never sends a synthetic prompt or changes Git identity. An active session is
not interrupted; let its next normal turn provide any user-facing `pwd` smoke.
If the script exits non-zero, treat the migration as unaccepted and inspect the
reported backup before retrying.

The success receipt reports the backup path, live cwd/runtime root, status,
protected metadata, rollout prefix preservation, and appended event types.
If the app UI is available, optionally confirm the same cwd with
`codex_app__read_thread` after the script succeeds.

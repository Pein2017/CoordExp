---
name: safe-runtime-maintenance
description: Audit and, only with explicit authorization, safely stop confirmed orphaned runtime processes or remove exact disposable caches without disrupting active work.
---

# Safe Runtime Maintenance

Use for a requested cleanup of stale headless processes or one-time caches in
the CoordExp environment. It protects active Codex, research, GPU, test,
service, and user-data surfaces.

Do not use it for broad `/tmp` or `.codex` cleanup, ambiguous workers, model or
research data, or a deletion the user has not explicitly authorized. A request
to "clean up" authorizes an audit, not a signal or deletion.

## Classify before acting

1. Freeze the exact target list and mutation boundary. Inventory process
   ancestry, sockets, open descriptors, current working directories, locks or
   heartbeats, recent activity, and GPU use. For a directory, also inspect its
   provenance, mount status, writers, and process references.
2. Return each target as `preserve`, `eligible with authorization`, or
   `needs user direction`, with the evidence behind it. PPID 1, age, size,
   deleted cwd, or no GPU use alone never establishes staleness.
3. Preserve active or ambiguous Codex, research, training, pytest, service,
   daemon, SQLite/WAL/SHM, receipt, and user-data surfaces. A zombie child of
   an active parent is not an individual kill target.

## Execute only after authorization

- Revalidate PID identity immediately before action. For an eligible process,
  send only `SIGTERM` to the exact PID; do not escalate or broaden the target
  set automatically.
- For an explicitly authorized, eligible directory, delete only the exact,
  non-mounted path with `find "$target" -xdev -depth -delete`. Never use a
  broad root, glob, or `rm -rf`.
- If a lock, heartbeat, socket, open descriptor, or recent writer contradicts
  the classification, stop and preserve the target.

## Verify and report

Recheck the targeted PID/listener or path, plus the protected-service
inventory. Report the exact targets, evidence, action (or preservation), and
any unresolved lock or ownership discrepancy. Treat prior cleanup targets as
history, not a standing deletion list.

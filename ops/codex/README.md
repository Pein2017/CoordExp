# Codex Ops

Codex operational helpers live here instead of under `scripts/` so training,
inference, and evaluation entrypoints stay focused on CoordExp itself.

- `commit_codex_memories.sh`: commits only `.codex/memories/**/*.md` changes
  with the default message `refresh memories`.
- `watch_codex_memories.sh`: watches `.codex/memories` and invokes the commit
  helper after a debounce interval.
- `install_codex_memory_watcher.sh`: installs the watcher as a user-level
  `systemd` service.

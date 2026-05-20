# Codex Ops

Tracked Codex operations are intentionally minimal.

Policy:

- `.codex/skills/` is the tracked repo-local agent capability surface.
- `.codex/memories/`, sessions, logs, plugin caches, auth, local config, and
  app runtime state are local-only workspace state.
- Do not add memory auto-commit watchers here. They conflict with the current
  local-only memory policy and can silently mix agent state with source changes.

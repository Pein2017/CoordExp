# Ops Tools

This directory contains repository-tracked operational tooling that supports
the local development environment but is not part of CoordExp training,
inference, evaluation, or research artifact production.

Use `scripts/` for CoordExp pipeline commands. Use `ops/` for IT, system,
agent-runtime, and workstation automation helpers.

## Current Layout

- `ops/codex/`: tracked policy note for repo-local Codex state. Agent runtime
  state stays local-only; there is no memory auto-commit helper.
- `ops/workspace/`: local workstation and repository hygiene helpers that are
  not training, inference, evaluation, or artifact-production entrypoints.

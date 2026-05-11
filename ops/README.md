# Ops Tools

This directory contains repository-tracked operational tooling that supports
the local development environment but is not part of CoordExp training,
inference, evaluation, or research artifact production.

Use `scripts/` for CoordExp pipeline commands. Use `ops/` for IT, system,
agent-runtime, and workstation automation helpers.

## Current Layout

- `ops/codex/`: Codex agent/runtime helpers, including memory auto-commit
  tooling.
- `ops/workspace/`: local workstation and repository hygiene helpers that are
  not training, inference, evaluation, or artifact-production entrypoints.

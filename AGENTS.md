# CoordExp main

Main owns stable source integration, annotation/data operations and shared
ownership policy. Sibling worktrees are separate versioned capabilities, not an
automatic release ladder.

If the user-wide contract is not already present in context, read
`/data/CoordExp/.codex/AGENTS.md` once. This entry adds only local routing.

Read the relevant owner when the task crosses its boundary:

- Checkout selection or integration: `docs/checkout-ownership.md`.
- Run/report paths, shared assets or retirement: `docs/storage-ownership.md`.
  Root `outputs/` is selected shared retention, not a branch-run destination.
- Agent configuration/loading or local Python environment: `docs/agent-integration.md`.
- Scientific records: canonical research-probes `research/conventions.md`;
  ordinary documentation recovery uses `docs/documentation.md`.

Runtime behavior belongs to this checkout's source, typed configs and stable
`openspec/specs/`, not a sibling's current implementation or historical receipt.

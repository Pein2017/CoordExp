# CoordExp main

Main owns stable source integration, annotation/data operations and shared
ownership policy. Sibling worktrees are separate versioned capabilities, not an
automatic release ladder.

If the user-wide contract is not already present in context, read
`/data/CoordExp/.codex/AGENTS.md` once. This entry adds only local routing.

Read the relevant owner when the task crosses its boundary:

- Checkout selection or integration: `docs/BRANCH_AND_WORKTREE_POLICY.md`.
- Run/report paths, shared assets or retirement: `docs/OUTPUT_STORAGE_POLICY.md`.
  Root `outputs/` is selected shared retention, not a branch-run destination.
- Agent configuration/loading or local Python environment: `docs/AGENT_INDEX.md`.
- Scientific records: canonical research-probes `research/CONVENTIONS.md`;
  ordinary documentation recovery uses `docs/RETENTION.md`.

Runtime behavior belongs to this checkout's source, typed configs and stable
`openspec/specs/`, not a sibling's current implementation or historical receipt.

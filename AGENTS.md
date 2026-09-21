# Research-probes entry

The canonical research checkout is `/data/CoordExp/.worktrees/research-probes`.
Verify this checkout, current Git state, and task authority before changing it.
Preserve unrelated dirty work; publication and experiments need their own grants.

Read only the owners needed for the task:

- `docs/OUTPUT_STORAGE_POLICY.md`: maintained source, output artifacts, source
  captures, archival recovery and the read-only output-layout check.
- `docs/RESEARCH_PROBE_INFRA_BASE.md`: reusable mechanisms and direction entries.
- `research/CONVENTIONS.md` and `research/index.md`: research placement and frontier.
- `docs/BRANCH_AND_WORKTREE_POLICY.md`: checkout and worktree lifecycle.

Use ordinary maintained imports. Do not execute code from `outputs/`, archives,
scratch directories or another worktree. Reuse proven operations, not scientific
assumptions. Keep new source/config identity distinct from frozen historical
evidence. Do not turn old run instructions or archive records into launch authority.

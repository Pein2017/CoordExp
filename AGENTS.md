# Research-probes entry

The canonical research checkout is `/data/CoordExp/.worktrees/research-probes`.
Verify this checkout, current Git state, and task authority before changing it.
Preserve unrelated dirty work; publication and experiments need their own grants.

Read only the owners needed for the task:

- `docs/OUTPUT_STORAGE_POLICY.md`: maintained source, output artifacts and recovery.
- `probes/README.md`: retained research operators and direction entries.
- `docs/SYSTEM_OVERVIEW.md`: dependency boundaries when integrating operators.
- `openspec/specs/coordexp-infras-research-probe-infra-base/spec.md`: stable
  reusable execution and integrity mechanics.
- `research/CONVENTIONS.md` and `research/index.md`: research placement and frontier.
- `docs/BRANCH_AND_WORKTREE_POLICY.md`: checkout and worktree lifecycle.

Use ordinary maintained imports. Do not execute code from `outputs/`, archives,
scratch directories or another worktree. Reuse proven operations, not scientific
assumptions. Keep new source/config identity distinct from frozen historical
evidence. Do not turn old run instructions or archive records into launch authority.

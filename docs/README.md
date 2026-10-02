# Long-lived documentation

This directory keeps stable reasoning and a small map, not a second description
of the source tree. Use the selected checkout's source, CodeGraph, CLI help,
typed configs and tests for current implementation details.

| Need | Owner |
|---|---|
| Documentation admission and historical recovery | [Asset policy](RETENTION.md) |
| Dependency boundaries | [Overview](SYSTEM_OVERVIEW.md) |
| Checkout authority | [Worktrees](BRANCH_AND_WORKTREE_POLICY.md) |
| Source/data/output ownership | [Storage](OUTPUT_STORAGE_POLICY.md) |
| Geometry and data identity | [Data](data/CONTRACT.md) |
| Evaluation meaning | [Evaluation](eval/CONTRACT.md) |
| Engineering design principles | [Style](standards/CODE_STYLE.md) |

Stable compatibility requirements belong to local `openspec/specs/`.
Scientific interpretation belongs to the owning research question, not docs.
[Task navigation](AGENT_INDEX.md) is optional, not a required reading chain.

Main is the stable integration checkout. Current research is routed through
[the research index](../research/index.md), not copied here from another branch.

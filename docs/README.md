# Long-lived documentation

This directory keeps stable reasoning and a small map, not a second description
of the source tree. Use the selected checkout's source, CodeGraph, CLI help,
typed configs and tests for current implementation details.

| Need | Owner |
|---|---|
| Documentation admission and historical recovery | [Asset policy](documentation.md) |
| Dependency boundaries | [Overview](architecture.md) |
| Checkout authority | [Worktrees](checkout-ownership.md) |
| Source/data/output ownership | [Storage](storage-ownership.md) |
| User-facing cross-environment artifact transfer | [Transfer guide](artifact-transfer.md) |
| Geometry and data identity | [Data](data/contract.md) |
| Evaluation meaning | [Evaluation](eval/contract.md) |
| Engineering design principles | [Engineering principles](engineering-principles.md) |

Stable compatibility requirements belong to local `openspec/specs/`.
Scientific interpretation belongs to the owning research question, not docs.
[Task navigation](agent-integration.md) is optional, not a required reading chain.

The [existing local documentation seal](../manifests/documentation/retirement-20261003.json)
locates this checkout's earlier manuals; it is not a new archive or a recovery
claim for every historical research or untracked file.

Main is the stable integration checkout. Current research is routed through
[the research index](../research/index.md), not copied here from another branch.

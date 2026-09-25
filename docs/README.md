# Engineering interfaces

For scientific state use [research](../research/index.md); for executable methods
use [probes](../probes/README.md). This directory owns current engineering contracts,
not operation diaries or a parallel research status system.

| Need | Owner |
|---|---|
| Architecture and dependencies | [System overview](SYSTEM_OVERVIEW.md) |
| Agent navigation | [Task entry](AGENT_INDEX.md) |
| Checkout and integration | [Worktree policy](BRANCH_AND_WORKTREE_POLICY.md) |
| Core behavior | [Core contract map](coordexp_infras.md) |
| Input/geometry and packing | [Data](data/README.md) |
| Evaluation semantics | [Evaluation](eval/README.md) |
| Training and continuation | [Training](training/README.md) |
| Artifact and source identity | [Artifacts](ARTIFACTS.md) |
| Storage boundaries | [Storage](OUTPUT_STORAGE_POLICY.md) |
| Current adapter qualification | [DoRA](adapters/dora-qualification.md), [selected embeddings](adapters/selected-embedding-qualification.md) |
| Coding standards | [Standards](standards/README.md) |

Stable behavior belongs in `openspec/specs`. The current bounded cleanup remains
in the existing change until validated integration; absorbed old changes are in
Git rather than a second in-tree archive.

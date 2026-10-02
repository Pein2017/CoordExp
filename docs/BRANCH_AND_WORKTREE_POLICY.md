# Checkout ownership

| Checkout | Physical root | Responsibility |
|---|---|---|
| main | `/data/CoordExp` | Stable source integration and shared ownership policy |
| coordexp-infras | `/data/CoordExp/.worktrees/coordexp-infras` | Infrastructure development and its runs |
| research-probes | `/data/CoordExp/.worktrees/research-probes` | Canonical research source, knowledge and runs |
| research-probes-web-codex | `/data/CoordExp/.worktrees/research-probes-web-codex` | Separate research development checkout |

These branches may differ. A document describes its local checkout, not whatever
happens to be newest in a sibling. Validate the registered Project, canonical
physical root, HEAD and dirty scope before modifying or running anything.
Source integration and moving artifacts are separate decisions; neither rewrites
the original producer nor authorizes a new model run.

Preserve unrelated dirty work and active consumers. Do not reset, clean, stash,
force-update or recreate a temporary worktree merely to simplify a task. Commit,
push, merge, rebase and worktree lifecycle changes require the user's scope.
Validate the exact descendant before authorized integration and recheck canonical
HEAD, dirty state and holders immediately before adoption.

Runs write to their owning physical worktree. Root `outputs/` is not a fallback
run directory. See [storage](OUTPUT_STORAGE_POLICY.md). Historical sessions and
receipts retain their original provenance; stale UI metadata is not Git truth.

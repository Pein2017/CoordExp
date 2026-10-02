# Research checkout boundary

Canonical research is `/data/CoordExp/.worktrees/research-probes`.
Main and infrastructure are separate versioned checkouts; do not substitute their
source or docs for this branch's behavior. The cross-checkout owner map is
`/data/CoordExp/docs/BRANCH_AND_WORKTREE_POLICY.md`.

Verify the registered Project, physical root, HEAD, dirty scope and active holders
before effects. Preserve parallel work; no automatic reset, clean, stash,
worktree creation or broad process stop. Explicitly authorized integration must
validate the exact descendant and recheck canonical HEAD/dirty state immediately
before adoption. A clean tree alone does not establish that no consumer is live.

[Storage ownership](OUTPUT_STORAGE_POLICY.md) and scientific evidence identity
are distinct from source integration. Historical records retain their original
producer and do not authorize a new run or Git publication. Temporary task state
must not become a permanent second research lane.

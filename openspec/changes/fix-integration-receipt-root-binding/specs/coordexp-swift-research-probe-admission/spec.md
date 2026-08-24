## ADDED Requirements

### Requirement: CPU compatibility receipts bind an explicit execution root

The CPU compatibility adapter SHALL select a sealed support-and-crossover
receipt pair only for its exact resolved execution worktree. The fixed
`research-probes` execution root SHALL use the pair captured from that root,
and the fixed `research-probe-infras` execution root SHALL use the pair
captured from that root. Receipt validation MUST continue to require the full
resolved path, byte count, and SHA-256 for every declared source and artifact
input; equal bytes at another absolute path MUST NOT substitute for the bound
identity.

Canonical selection MUST NOT require the integration-lane worktree to remain
readable or present after a separately approved merge; an unavailable
non-selected fixed root MUST NOT cause a fallback or prevent canonical receipt
selection.

Any execution root outside those two declared fixed paths MUST fail before a
receipt is accepted. The capability MUST NOT choose a receipt by filename,
content hash alone, output-directory discovery, environment fallback, or
automatic receipt regeneration. This selection changes no consumer-owned plan,
projection, finalizer, scientific field, model/GPU boundary, or target-tree
revalidation behavior.

#### Scenario: Integration lane accepts its own sealed CPU receipt pair
- **WHEN** the adapter executes from the clean fixed
  `research-probe-infras` worktree and its declared files match the sealed
  integration-lane receipt pair
- **THEN** both CPU compatibility receipts validate through their existing
  support and crossover checks without weakening any path-or-byte identity

#### Scenario: Canonical target accepts its merged-target CPU receipt pair
- **WHEN** the adapter executes from the clean fixed `research-probes` worktree
  and its declared files match the sealed merged-target receipt pair
- **THEN** both CPU compatibility receipts validate through their existing
  support and crossover checks without reading the integration-lane pair

#### Scenario: Integration lane is unavailable after canonical integration
- **WHEN** the adapter executes from the fixed `research-probes` worktree after
  the `research-probe-infras` worktree no longer resolves
- **THEN** it still selects the canonical merged-target receipt pair and does
  not substitute a receipt or fail because the unavailable integration root was
  not selected

#### Scenario: Unknown execution root is proposed
- **WHEN** the adapter executes from any root other than the two declared fixed
  worktrees
- **THEN** it rejects before accepting a CPU compatibility receipt and does not
  substitute a same-name, same-content, or discovered receipt

#### Scenario: Receipt from the other fixed root is supplied
- **WHEN** an adapter is given a receipt whose stored execution-tree path is
  the other fixed worktree, even if its bytes match the local file
- **THEN** exact receipt validation rejects the path identity mismatch before
  publishing or accepting stage evidence

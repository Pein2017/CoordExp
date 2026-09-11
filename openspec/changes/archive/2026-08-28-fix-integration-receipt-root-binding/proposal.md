## Why

After the integration lane fast-forwarded to the canonical research baseline,
its CPU compatibility suite rejected the canonical receipt even though the
bound adapter and generator bytes matched. The receipt correctly includes each
file's absolute path, but two execution-tree-local bindings were compared to a
receipt captured from a different fixed worktree; the lane therefore cannot
validate its own exact execution surface before a later merge.

## What Changes

- Select the existing CPU compatibility receipt pair by the exact fixed
  execution worktree: `research-probes` uses its merged-target receipts and
  `research-probe-infras` uses its integration-lane receipts; canonical
  selection does not require the integration lane to remain present.
- Reject every other execution root before accepting a compatibility receipt;
  do not fall back by filename, content hash, or a shared output directory.
- Preserve full resolved-path, byte-count, and SHA-256 receipt identity, and
  add focused tests for both allowed roots, unknown-root rejection, and
  canonical selection after the integration lane becomes unavailable.
- Keep CPU mechanics-only semantics, external receipt payloads, target-tree
  binding, GPU boundaries, and consumer scientific fields unchanged.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `coordexp-infras-research-probe-admission`: CPU compatibility receipt
  acceptance gains an explicit fixed execution-root selection rule while
  retaining exact path-and-byte binding and fail-closed behavior.

## Impact

- Affected code: `scripts/research/research_probe_admission_consumers.py`.
- Affected tests: `tests/research/test_research_probe_admission_consumers.py`.
- Affected external inputs: the two already-existing CPU receipt roots under
  `/data/CoordExp/outputs/research-probe-infras/`; this change neither rewrites
  nor regenerates them.
- No new dependency, GPU/model execution, worktree move, branch rewrite, or
  public workflow abstraction is introduced.

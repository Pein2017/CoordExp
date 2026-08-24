## Context

See [proposal.md](proposal.md) for motivation and the delta spec for the
observable contract. CPU receipt validation already binds a full resolved path,
byte count, and SHA-256. Most consumer inputs are pinned to the canonical
target tree, while the support adapter and receipt generator resolve from the
executing worktree. After the integration lane was fast-forwarded, those two
execution-local fields correctly rejected a receipt captured from the canonical
tree despite byte equality.

## Goals / Non-Goals

**Goals:**

- Make the two permitted execution-root/receipt-pair associations explicit and
  testable.
- Preserve strict path-and-byte validation so an unmerged integration-lane edit
  cannot be treated as canonical validation.
- Keep a clean integration lane locally verifiable with its own sealed receipt
  pair, and keep canonical validation bound to its merged-target pair.
- Keep canonical receipt selection operable after the integration lane has been
  retired.

**Non-Goals:**

- Rewriting, regenerating, copying, or relaxing either external receipt.
- Discovering arbitrary worktrees, adding configuration/environment fallbacks,
  packaging, a generic runner, or a new artifact ledger.
- Changing target-tree binding, consumer scientific semantics, GPU execution,
  worktree paths/locks, or baseline tags.

## Decisions

### Select from a closed two-root map

The consumer adapter will own one narrow mapping from each exact fixed execution
root to its pre-existing support/crossover receipt pair. Selection is made
before receipt validation, with an explicit rejection for an unknown root.
This matches the two fixed paths already protected by project policy and avoids
turning a receipt locator into a discovery protocol.

Candidate roots are resolved independently: an unavailable non-selected peer is
skipped, while the selected execution root must still resolve exactly. This
lets a merged canonical tree survive later integration-lane retirement without
turning absence into a fallback or accepting a third root.

Using one canonical receipt for both roots is rejected: it would make an
integration-lane adapter/generator modification appear validated against bytes
from a different source path. Dropping paths from identity is rejected because
the current contract deliberately binds the exact local execution surface.
Generating a receipt automatically is rejected because a fresh receipt is an
explicit mechanics operation with its own external-root and replay boundary.

### Test selection independently, then replay the consumer contract

A focused test will exercise both allowed roots, the unknown-root rejection,
and canonical selection with an unavailable integration root without relying on
source-tree import accidents. The existing consumer
compatibility suite will then run from each fixed worktree, demonstrating that
the selected receipt matches its exact local adapter/generator identities while
the existing cross-root rejection remains intact.

## Risks / Trade-offs

- [A future third execution root needs validation] → It fails closed until a
  separately scoped change introduces an exact receipt and an explicit mapping.
- [A receipt is stale after a local source change] → Existing full identity
  validation rejects it; this change provides no fallback or regeneration.
- [The two fixed roots converge in bytes] → Their absolute-path identities
  remain distinct by design, preserving merge-time provenance.
- [The integration lane is retired after merge] → Canonical selection skips the
  unavailable non-selected root; it does not substitute an alternate root or
  receipt.

## Migration Plan

1. Add a focused RED test for the closed two-root selection and unknown-root
   rejection.
2. Implement the smallest explicit selector and route the existing default
   receipt constants through it.
3. Replay the focused selector tests plus the expanded 44-test target-binding
   contract suite from the integration lane, and replay the canonical target's
   existing 41-test suite CPU-only. The selector regression directly covers the
   canonical pair; after any separately approved merge, re-run the expanded
   suite from the actual canonical target as a fresh integration gate.
4. Validate the change strictly and obtain one independent read-only review;
   leave external receipts, tags, locks, and GPU state unchanged.

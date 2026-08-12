# Disposition: Superseded While Incomplete

## Status

This change was archived on 2026-08-12 as **superseded-incomplete** by the
user's explicit decision. It is not a completed-change record.

At disposition time, `tasks.md` contained 64 completed tasks out of 78. The 14
unchecked tasks remain unchecked. No missing gate, measurement, audit, cache
publication, or compatibility result is converted into a pass by this
archive.

## Why This Change Was Superseded

The change accumulated correctness hardening, measurement campaigns,
experimental input and packing policies, exact-resume work, upstream
compatibility, cache publication, production convergence, and final cleanup in
one authority surface. The next phase separates these concerns so that
operational telemetry, scientific loss semantics, cache identity, and
behavior-neutral architecture work have independent acceptance and rollback
boundaries.

The user also selected these cleanup decisions for the successor work:

- keep `synchronous` as the production input-provider reference;
- retire `legacy_fused` and its deprecated environment override;
- retain `overlapped` only as an explicit experimental, tested mode;
- defer RL objective composition to a later dedicated change;
- permit one intentional absent-target cache fingerprint transition after the
  narrow cached-payload producer becomes the determinant owner.

## Preserved Evidence And Claim Boundary

Checked tasks, immutable receipts, implementation notes, measurement plans,
and review records remain historical evidence for the exact slices they name.
In particular, an executed correctness or plumbing receipt does not become an
efficiency promotion, a future-policy decision, or whole-change completion.

The following work remains explicitly unresolved at archive time:

- Wave 6 tasks 7.7 through 7.9: matched training comparison and promotion gate
  for changed-order packing policies. Task 7.10 already records that this work
  was deferred to future design and that `source_order_next_fit` remained the
  production default.
- Wave 7 tasks 8.8 and 8.9: interruption/incomplete-rank publication behavior
  and the complete exact-resume gate.
- Wave 8 task 9.5: the final pinned compatibility gate.
- Wave 9 tasks 10.1, 10.2, 10.4, and 10.6 through 10.10: convergence
  preconditions, transition packet, production cache publication, final
  comparison, cleanup, full verification, independent audits, and user-facing
  closure.

## Delta-Spec Disposition

All six delta-spec directories are preserved here, but this archive deliberately
uses `--skip-specs`. These deltas are not synchronized wholesale into stable
specs because their acceptance gates were not all completed and because the
exact-resume delta needs reconciliation with the stable minimal inference
checkpoint contract.

Successor changes must restate only the live, accepted behavior they own. They
must not cite this archive as authority for an unchecked requirement. Exact
training state, when retained, must be specified as an opt-in typed sibling of
the minimal inference payload; inference loading must ignore it, and disabled
exact resume must not write it.

## Successor Boundaries

The successor sequence is intentionally split:

1. reconcile the current accepted training contracts and eliminate conflicts
   between live behavior, stable specs, and operator documentation;
2. decompose training orchestration and narrow cache/probe identity ownership
   without changing protected training behavior;
3. standardize typed SFT loss bindings and protected-versus-optional zero-weight
   behavior against the final orchestration owners, leaving RL composition out
   of scope;
4. add typed training telemetry and rank-zero presentation sinks on top of the
   standardized loss-row schema while keeping the canonical per-step JSONL
   receipt authoritative.

This disposition file is the authoritative explanation for why the archived
task list is intentionally incomplete.

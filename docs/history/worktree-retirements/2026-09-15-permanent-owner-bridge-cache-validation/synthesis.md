---
title: Permanent Owner Bridge historical intake
role: historical-worktree-synthesis
authority: archived-source-summary
status: retired
source_commit: 4f6fca72e91ae8b752a2a20edb69f772b3f69196
---

# Permanent Owner Bridge: historical intake and claim boundary

## What this capture preserves

The historical Stage1 design specified a persistent bridge made of addressable
visual-owner atoms, a global image-level assignment, set routing, row-local
commitment, and a later row write. Its exact intended contract is preserved in
the [design](sources/openspec/changes/add-permanent-owner-bridge/design.md),
[proposal](sources/openspec/changes/add-permanent-owner-bridge/proposal.md),
and [Stage1 guide](sources/docs/training/PERMANENT_OWNER_BRIDGE_STAGE1.md).
Those documents describe a historical architecture and frozen execution
contracts; they are not a compatibility requirement for current training.

The execution records establish several narrower facts. The original production
activation failed before model load because ranks waited behind a cache scan,
not because the bridge was evaluated
([failure record](sources/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-10-permanent-owner-bridge-final-readiness/production-activation-attempt-1-failure.md)).
A later repair demonstrated its declared pre-model choreography, a bounded W8
smoke with finite/applied updates, checkpoint composition, and two-row fresh-HF
lifecycle behavior
([repair acceptance](sources/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-10-permanent-owner-bridge-premodel-repair-acceptance/results.md)).
The earlier W8 collective split is retained as a technical failure: it showed
rank-divergent collective order, while its proposed real-versus-shadow-forward
cause remained a hypothesis
([failure record](sources/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-10-permanent-owner-bridge-w8-collective-choreography-failure/results.md)).

## Absorbed interpretation

These records support implementation and bounded runtime claims only on their
named identities. They do not establish model quality, dense-object recall,
generalization, a useful owner ledger, or a reason to promote the architecture.
The later, separately preserved
[step-611 native-greedy probe](../../research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-12-owner-bridge-step611-recall-probe/results.md)
is the behavioral boundary: the complete bridge composition loaded and its
route/latch/write lifecycle executed, yet its 13-image primary screen recovered
only 7 of 392 class-and-IoU50 matches and image 2299 recovered 0 of 46 owners.
It therefore held further training and architecture promotion.

The present research frontier treats an explicit slot, ledger, commit, or bridge
as an optional hypothesis. Reopening requires a separately authorized,
decision-bearing comparison with a stated behavioral or cost advantage over an
ordinary native-state solution. A repaired process, a finite update, or a
working route lifecycle cannot substitute for that comparator.

## Reading order

1. Start with the step-611 behavioral result above for the decisive model
   limitation.
2. Use the [final smoke acceptance](sources/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-10-permanent-owner-bridge-final-smoke-acceptance/results.md)
   and [pre-model repair acceptance](sources/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-10-permanent-owner-bridge-premodel-repair-acceptance/results.md)
   for the accepted mechanical scope.
3. Use the [launch acceptance](sources/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-11-permanent-owner-bridge-production-launch-acceptance/results.md)
   and [one-time recovery successor](sources/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-11-permanent-owner-bridge-recovery-successor/results.md)
   only as historical lifecycle records. They do not reopen their original
   launch authority.

The [manifest](manifest.json) and the annotated archive tag are the source
recovery route for any deeper code, config, receipt, or document question.

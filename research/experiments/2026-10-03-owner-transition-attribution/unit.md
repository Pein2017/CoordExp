# Saved-output owner eligibility and assignment

## Question and authority

From the accepted round04 anchor and two endpoints, where do the114 observed
category-owner transition instances fall in the unchanged candidate-eligibility
and one-to-one-assignment pipeline?

The autonomous research grant remains active. A new GPT-6.1-Sol/high worker
owns this finite CPU-only package; the lead selects the question and accepts
the result. There is no training, model forward, GPU call, new generation,
annotation change, or new threshold. This is a retrospective evaluator
diagnostic, not a causal explanation or physical-owner disappearance audit.

The [accepted predecessor](../2026-10-03-prefix-exposure-ranking/lead-ruling-02.md)
has single31 gains/29 losses and multiple24 gains/30 losses. Both learned arms
repair every frozen illegal context, but neither natural endpoint visits those
exact contexts. Early coordinate changes and saturated late margins leave at
least two alternatives: actual changed output coverage, and geometry/category/
assignment changes at the evaluator boundary. Saved outputs can separate parts
of these alternatives before another dose or objective experiment.

## Frozen inputs and denominator

Consume only the54 accepted natural records in the execution-local round04
`package-01/native-{anchor,R-single,R-multiple}/natural-<image>.json`, their
original18-image/570-annotation snapshot, released contract, terminal candidate,
and lead acceptance. Preserve original image/reference/row order.

Round04 root:
`/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/`.

- `native-terminal-candidate-01.json` SHA256
  `78583948601bdb86b94e3d08d6086b82d452d0c8f3fd834cac235af375304080`.
- `lead-acceptance-01.json` SHA256
  `c93f0863496fde0f14404ca4af8daa2733f7c88f37bc405553131b4b21cba931`.
- `released-contract-01.json` SHA256
  `0fe3deac8f3eff7ec542686bdda8c7673aeee91b993bd3ca1628ac2233bf441b`.

Resolve the annotation identity from that contract's evidence bindings. Verify
only consumed input hashes once and reuse those bindings; no checkpoint/model
tree hashing or rerun of the five-phase readback is required. Keep the saved
execution source identity distinct from this new CPU analysis source revision.

The114 instances are55 gains plus59 losses, keyed by arm/image/annotation ID/
direction; they are not114 distinct physical owners. Reconcile exact category
coverage274/276/268 for anchor/single/multiple and exact transition ID sets.
Annotations are used only to explain the already fixed evaluation; no selector,
training target, corrected truth or acquisition feedback is produced.

## Reuse the actual evaluator

The maintained path is `prefix_exposure_ranking.natural_measure` ->
`rollout_row_credit.assess_outputs` -> `hidden_human_recovery.candidates`
(aliased through `iterative_positive`) -> `saved_rows.one_to_one_matches` ->
`assignment.global_matches`. Do not substitute `online_row_credit.observations`
as the matching pool. It may supply token-position diagnostics only.

Reconstruct all54 saved measurements through the maintained parser/evaluator
and require equality with their accepted records before reporting attribution.
The accepted raw text/token identity may be reused; do not reload a model merely
to redecode text already checked by the accepted round04 readback.

Every strict-parser-valid row participates, including duplicates. Matching is
class-agnostic maximum cardinality then maximum total quantized IoU at inclusive
IoU>=0.5. Category credit uses exact description equality after assignment;
there is no category-aware rematch. Preserve full-image normalized coordinate
mapping without pixel rounding. Invalid/malformed rows remain burdens only.

Candidate `generated_order` is a compressed valid-row index. The prediction-ID
suffix uses raw parser row order; keep both. Do not align physical objects by
equal row numbers across trajectories. Reuse maintained parsing/span helpers
for a witness's raw row and token positions; do not invent another parser.

## Descriptive ledger

For a gain examine the uncovered anchor side; for a loss examine the uncovered
endpoint side. For that reference owner, let S be valid same-description rows
with IoU>=0.5, A all valid rows with IoU>=0.5, and s the maximum same-description
IoU, null if no same-description candidate exists. Assign exactly one class:

1. S nonempty: `eligible_same_category_not_credited`.
2. S empty, A nonempty: `only_other_categories_at_threshold`.
3. A empty and0<s<0.5: `same_category_overlap_below_threshold`.
4. Otherwise: `no_positive_same_category_overlap`; distinguish absent same-
   description candidates from present candidates with zero overlap.

Keep assignment status separately: unassigned, or assigned to a different
description, with the actual assigned row. In class1 retain every qualifying
same-description row and the owner, if any, that received it under the existing
assignment. For both sides retain best same-description and any-description
IoU with witnesses, eligible candidate counts, actual assignment, coverage,
reference box/description and original raw locators/row positions. Keep null
and zero distinct. These overlapping descriptive facts are not disjoint causes.

Produce one `attribution.json` containing provenance, frozen semantics, all114
entries, and aggregate classes separately by arm and gain/loss. Reconcile every
entry to the accepted transition ID set. Do not add near-repeat attribution,
new similarity thresholds, image judging, a gallery, or a second matching rule.
Existing near-repeat totals remain in the predecessor's results.

## Implementation, checks and stop

Use the canonical research checkout. One small `probes/owner_transition_attribution.py`
and a focused check file may implement build/readback. Reuse installed matching
and parser helpers; no shared runtime/source changes or new dependency. Record
Git revision/diff and explicit input identities; this pure CPU derivative does
not need a new clean native execution checkout.

The final consumer must recompute the114-entry projection from the bound raw
records and reject fabricated IDs, assignments, IoUs/classification/positions,
or omitted transitions even when output hashes are refreshed. Use a small
number of focused fixtures through this caller: class-agnostic matching then
category check versus category-aware rematching; maximum cardinality versus
greedy matching; an exact0.5 overlap; a parser-drop row-order gap; null versus
zero overlap. Reuse existing assignment fixtures where possible. These are
consumer falsifiers, not a new universal testing framework.

The worker may implement, run the CPU analysis and consumer, repair within
this package, and return the terminal candidate without another release gate.
The first actual54-record readback is the production-shaped seam. Bound work
to these54 records/114 instances; no acquisition/optimizer/model requests.
Expected CPU time is minutes, not a background research job. Measure elapsed
time, peak RSS and output bytes; report if unexpectedly over300s or8GiB rather
than starting a broader analysis. Stop after full reconciliation and targeted
checks pass. Missing/changed inputs or unreconciled evaluator semantics stop
dependent analysis; do not rewrite historical receipts or silently substitute.

Canonical output root:
`outputs/research/physical-fn-recovery/2026-10-03/owner-transition-attribution-05/`.
Worker owns only the new probe/check, this unit's state/results and its output/
scratch paths. Lead owns this protocol, index/catalog, interpretation and the
next decision. No worker self-acceptance or next-unit scheduling is authorized.

Supported result: the eligibility/assignment conditions under which annotation
credit changes in these saved outputs. Unsupported: physical absence, semantic
forgetting, causal interference, reliable track identity, or policy promotion.

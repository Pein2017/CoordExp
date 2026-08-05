---
title: Sorted Supported False-Negative Native-Prefix Reachability Prevalence
description: CPU-only reanalysis of whether calibrated owner-local support for 114 native false negatives coincides with favorable gate, category-route, same-category-owner, and scan-frontier conditions on native greedy prefixes.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-08-03
---

# Sorted Supported False-Negative Native-Prefix Reachability Prevalence

## Execution closure

The CPU analysis, visualization, and independent review are complete. The
authoritative interpretation is [results.md](results.md); review provenance is
preserved in [review.md](review.md). The final immutable run is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence/
  20260803T231934Z/
```

The original protocol below remains frozen provenance. The result demotes STOP
and a pure covered-owner ledger failure, then narrows the live successor to
the 26-owner exact crossing-boundary cohort. No training or architecture was
promoted.

## Decision and outcome

This unit returns the investigation to the majority false-negative regime from
the completed [owner accessibility phenotype census](../2026-08-03-sorted-owner-accessibility-phenotype-census/results.md):
`114/202` eligible native false-negative owners already have calibrated
category-conditioned localization support at one or more tested native greedy
prefixes.

The decision is which simple upstream bottleneck should own the next
prospective probe:

- natural continue-versus-stop gating;
- category-description routing;
- same-category physical-owner competition;
- support that becomes favorable only after the geometry-sorted route has
  passed the owner; or
- a residual policy/row-realization failure even when all observed channels
  are jointly favorable.

The outcome is owner-level prevalence over all `114` supported-but-native-
missed owners. This is a post-hoc descriptive reanalysis of a spent panel. It
does not create a training admission rule, estimate a population effect, or
turn separately scored channels into a model proposal probability.

## Question

For each of the `114` owners, does any calibrated support context on the native
greedy trajectory simultaneously have:

1. continuation favored over stop;
2. the owner's category ranked first or within the top three tested
   categories;
3. the target physical owner ranked first among same-category owner-assigned
   candidates; and
4. a root, ahead-of-frontier, or at-frontier state rather than a state after
   the geometry-sorted frontier has passed the owner?

How frequently does each missing channel prevent this conjunction, and when a
different same-category owner wins, is that owner already covered by the
prefix or still uncovered?

## Strongest alternative

Every localization score is obtained after the category description is
teacher-forced, and the gate, category route, and coordinate bank are separate
readouts. Their conjunction is therefore only a **favorable observed surface**.
It is not the joint probability of a naturally generated row, and absence of
the conjunction does not prove which decoder token caused the miss.

To bound usefulness, the same channels are reported for all `141` native true
positives at their exact native due boundary. This is a positive reference,
not a matched causal control. If the conjunction is uncommon even there, it is
not a useful reachability descriptor and the unit stops without routing a
successor.

## Source boundary

| Source | Frozen role |
| --- | --- |
| Predecessor result | `../2026-08-03-sorted-owner-accessibility-phenotype-census/results.md` |
| Immutable run | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-owner-accessibility-phenotype-census/20260803T065743Z/` |
| Owner summaries | `phases/presentation/owner-summaries.jsonl` |
| Owner-context features | `phases/presentation/owner-context-features.jsonl` |
| Native contexts | `plan/context-registry.jsonl` |
| Native due owners | `plan/native-sidecar-registry.jsonl` |
| Semantic implementation owner | `scripts/research/merge_sorted_owner_accessibility_census_shards.py` |

No model, GPU, new score, new candidate, new threshold, new context, or free
decode is permitted.

## Operational definitions

- **Supported false negative**: an owner whose frozen presentation summary has
  disposition `resolved_tested_localization_support`. The denominator is
  exactly `114`.
- **Usable support context**: a context identifier in
  `upper_bound_u.usable_support_context_ids`. Lower-bound results are a
  required sensitivity, never a replacement denominator.
- **Native prefix**: root, a boundary after complete native greedy rows, or the
  native terminal boundary. Forced-continue rows are absent by construction.
- **Gate open**: `continue_vs_stop_logprob_margin > 0` at that context.
- **Category rank one / top three**: the category-route rank equals `1` or is
  at most `3` within the context's tested category population.
- **Owner rank one**: `owner_competition_u.rank == 1`, using same-context,
  same-category physical-owner competition.
- **Before-or-at frontier**: `passed_state` is `root_no_frontier`,
  `ahead_of_frontier`, or `at_frontier`.
- **Covered competitor**: the best competing physical owner at a support
  context is present among that context's prefix rows with strict matched
  status. Otherwise a non-target best owner is an uncovered competitor.
- **Favorable top-three surface**: usable support, gate open, category rank at
  most three, owner rank one, and before-or-at frontier at the same context.
- **Favorable rank-one surface**: the stricter version requiring category rank
  one.

`Rank within the local candidate bank`, `same-category owner rank`, and
`category-route rank` are distinct fields and must never be substituted for
one another. Rank or margin never creates localization support.

## Primary observations

Report owner-level numerators, denominators, Wilson `95%` intervals, per-image
counts, per-category counts, and full rank distributions for:

1. any usable support at root, row boundary, and terminal;
2. any support before-or-at versus after the sorted frontier;
3. any support context with gate open;
4. any support context with category rank one and category top three;
5. any support context with owner rank one;
6. favorable top-three and favorable rank-one surfaces, with and without the
   before-or-at-frontier restriction; and
7. target, covered-other, and uncovered-other best-owner status, especially
   among owners that never attain owner rank one.

Report the exact same field ladder for native true positives at their due
boundary. Because the false-negative arm uses an optimistic `any context`
operator while the reference uses one exact due context, differences remain
descriptive and cannot be read as causal effect sizes.

No binary phenotype threshold is fitted. Multi-channel conditions remain a
transparent conjunction ladder, and each owner record retains the underlying
continuous ranks and margins.

## Interpretation map

| Observation | Bounded next-route implication |
| --- | --- |
| Gate open for nearly all supported false negatives | Demote STOP as the main cause for this cohort |
| Category route rarely reaches rank one despite local support | Prioritize description/row-selection accessibility over vision or geometry repair |
| Target owner rarely wins same-category competition | Prioritize owner selection and competition; inspect whether the winner is covered or uncovered |
| Covered competitors dominate losses | A simple coverage-state intervention becomes credible |
| Uncovered competitors dominate losses | The issue is broader remaining-owner ranking, not merely forgetting covered owners |
| Many favorable pre-frontier surfaces still never emit | Select those owners for a prospective natural-prefix row-release or short-horizon policy probe |
| Favorable conjunction uncommon at native-TP due boundaries | Stop; the conjunction is not a useful descriptor |

These implications select a discriminator only. They do not authorize
training.

## Controls and safeguards

- Validate the predecessor plan, presentation files, schema identities,
  content hashes, image set, and exact `114`/`141` denominators before analysis.
- Exclude loop-tail contexts exactly as the predecessor did for usable
  non-loop support.
- Treat image `4134`, image `14038`, `person`, owner scale, and unequal context
  counts as explicit stratification/confound diagnostics.
- Aggregate to one owner before every headline count; an owner with 55 support
  contexts must not weigh 55 times more than one with a single context.
- Preserve both ambiguity bounds. The optimistic upper bound owns the primary
  denominator; the lower bound is sensitivity only.
- Do not reuse `margin_to_best_owner_in_group` as a support predicate; it was
  highly correlated with `peak_lift` in discovery and would partly re-derive
  the outcome.
- Every emitted owner row must retain source context IDs and exact input file
  digests.

## Visualization

Produce one score-independent owner matrix and small-multiple summaries:

- rows are the `114` owners, grouped by image and category;
- columns show support-context count, context role, frontier state, gate,
  category rank, owner rank, favorable conjunctions, and competitor coverage;
- a separate native-TP due-boundary reference panel uses the same columns; and
- no color encodes cross-image raw log probability.

The lead must inspect representative images with `view_image`, including
jointly favorable misses, uncovered-competitor misses, covered-competitor
misses, and late-only support.

## Stop rule

This unit stops after one complete CPU analysis and independent audit.

- If native-TP due boundaries do not commonly satisfy the channel ladder,
  close it as a weak descriptor.
- If one simple missing channel dominates across images, route one prospective
  discriminator to that channel.
- If several channels remain comparably prevalent, select a small factorial or
  short-horizon probe; do not create a complex latent-state architecture.
- If favorable pre-frontier misses form a material cross-image cohort, prefer
  them over a single anecdotal owner for the next natural-rollout-adjacent
  probe.
- The separate image-`4134` repetition probe remains available for its
  loop-degenerate stratum, but it cannot replace the all-owner prevalence
  result.

## Artifact handle

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence/
  <run-id>/
```

The compact product is one JSON report, one Markdown report, one owner-level
JSONL table, one exact-input receipt, and a visualization manifest. A path
without validated input identities is not evidence.

## Not claimed

- No naturally generated owner probability or row probability.
- No causal decomposition of description, owner identity, or coordinate
  tokens.
- No claim that the prefix implements a set ledger.
- No claim that a passed frontier prevents recovery.
- No new population estimate beyond the frozen twelve-image panel.
- No training objective, commit token, typed binding, contrastive loss, slot,
  detector, or architecture promotion.

## Originating-intent alignment

| Condition | Source | Class | Decision effect | Disposition |
| --- | --- | --- | --- | --- |
| Analyze all owners data-first rather than return to one special owner | User direction | scientific invariant | denominator and case selection | inherited |
| Keep the Sorted step-4887 checkpoint and frozen twelve-image panel | User direction and predecessor | scientific invariant | evidence scope | inherited |
| Distinguish visually unavailable owners from training-adjustable misses | User direction | scientific invariant | interpretation | inherited |
| Use native self-prefixes and exclude forced-continue rows | User direction and predecessor | scientific invariant | context meaning | inherited |
| Prefer the simplest discriminator and avoid premature commit/binding machinery | User direction | conservative design choice | successor and stop rule | inherited |
| Treat the parallel owner-commit training session only as a complexity ceiling | User direction | conservative design choice | architecture boundary | inherited |
| Add the native-TP due-boundary reference | Lead design | conservative design choice | descriptor validation | authorized within the active goal |

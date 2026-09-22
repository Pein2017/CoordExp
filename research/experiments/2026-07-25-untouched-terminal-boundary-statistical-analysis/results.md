---
title: Untouched Natural-Terminal Boundary Statistical Analysis Results
description: Completed read-only characterization of 200 Source natural-stop boundaries with 780 verified remaining owners before forced-opener release.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-25-untouched-terminal-boundary-statistical-analysis
topic: qwen3-vl-dense-enumeration
status: complete_ready_for_user_discussion
evidence_status: verified_bounded_terminal_panel_statistics
updated: 2026-07-25
---

# Untouched Natural-Terminal Boundary Statistical Analysis Results

## Decision

The fixed 200-boundary panel is suitable for a paired forced-opener release
study, but the data analysis changes both the mechanism interpretation and the
correct primary outcome.

The checkpoint effects mostly move the **same boundaries by increasing
strength**. Positive two-token margin sets are perfectly nested:

`Source 2 ⊂ transition step 36 45 ⊂ pairwise step 90 94 ⊂ owner-conditioned step 90 145`.

The checkpoint-minus-Source changes are strongly rank-correlated across arms,
and their magnitude is strongly associated with Source already being closer
to the row-opener-versus-terminal boundary. In contrast, the change is almost
uncorrelated with the number of verified owners still missing. This supports a
common continuation-confidence shift of different strength, not evidence that
the treatments measure the size of the unfinished owner set.

The panel contains 780 verified remaining physical owners. Only 55/200
boundaries have one remaining owner; 145/200 have multiple remaining owners.
A future release study therefore must use **recovery of any verified uncovered
owner** as its primary positive outcome. Choosing one arbitrary `intended
owner` would silently replace the user's set-completion question with a
single-target routing question.

Finally, the historical observed action and the FP32 two-token score are not
interchangeable. All 200 historical Source rollouts emitted the terminal token,
but two have a small positive FP32 row-opener-minus-terminal diagnostic margin.
Any causal release comparison must include a native arm and a forced-opener arm
under the same current runtime rather than comparing a new forced run only to
the historical action.

No inference or training was run in this unit.

## Verified Evidence Scope

The reduction joins the immutable manifest, all eight locality shards, and the
frozen candidate annotations. It contains:

- 200 rows, 200 unique boundary IDs, and 200 unique images;
- 780 unique verified remaining physical-owner IDs;
- observed Source action `terminal` and Source `natural_end=true` for every
  row; and
- identical boundary coverage for Source, transition step 36, pairwise step
  90, and owner-conditioned step 90.

Final artifacts are:

- `reduction-v2/summary.json`, SHA-256
  `d84130c6d68033f67e33e4dba6546270febd6bd8713b8b9ef072fa3289a2e991`;
- `reduction-v2/cases.jsonl`, SHA-256
  `3a43f787346aa5c5e44a43b6e8ae784c002a752fb533a7ae837411fdffdc63c2`.

The absolute root is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-untouched-terminal-boundary-statistical-analysis/`

The materializer selected boundaries deterministically by prefix depth and
then a stable hash. This is not a random sample of all Source stops, so Wilson
and bootstrap intervals are descriptive uncertainty summaries within the
frozen panel, not population confidence intervals.

## Panel Composition

| Property | Value |
| --- | ---: |
| Boundaries / unique images | 200 / 200 |
| Verified remaining owners | 780 |
| Remaining owners per boundary | mean 3.90, median 3, range 1--17 |
| One remaining owner | 55 boundaries |
| Multiple remaining owners | 145 boundaries |
| Prefix depth | mean 5.95, median 5, range 1--14 |
| Sparse / medium / dense / very dense images | 41 / 74 / 74 / 11 |
| Distinct remaining-owner categories | 71 |

The owner distribution is not balanced. `person` contributes 148/780 owner
occurrences, followed by `carrot` 68, `chair` 44, and `book` 41. The top four
categories contribute 38.59% of all remaining-owner occurrences; the top ten
contribute 58.59%. A later study should report per-boundary macro results and
category/owner composition, not only a pooled owner count that can be dominated
by repeated instances in a few images.

## Absolute Two-Token Margins

The margin is
`log P(canonical new-row opener) - log P(terminal token)` under the frozen FP32
diagnostic. `> 0` means the opener outranks the terminal token in that
two-token comparison only.

| Checkpoint | Mean margin | Median | Margin > 0 | Margin > +0.5 | Margin > +1.0 | Within ±0.5 | Terminal ahead by at least 2.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Source | -2.378 | -2.109 | 2/200 | 0/200 | 0/200 | 28/200 | 105/200 |
| Transition step 36 | -1.729 | -1.372 | 45/200 | 25/200 | 9/200 | 42/200 | 79/200 |
| Pairwise step 90 | -0.481 | -0.198 | 94/200 | 79/200 | 56/200 | 31/200 | 53/200 |
| Owner-conditioned step 90 | +2.826 | +2.575 | 145/200 | 137/200 | 130/200 | 17/200 | 18/200 |

The step-36 `45/200` crossing count is not uniformly strong: only 25 exceed
`+0.5`, and only 9 exceed `+1.0`. This is another reason not to predict release
behavior from the sign alone.

The descriptive Wilson 95% intervals for positive-margin fractions are
`[17.26%, 28.77%]` for transition, `[40.20%, 53.91%]` for pairwise, and
`[65.93%, 78.22%]` for owner-conditioned. These intervals do not repair the
panel's deterministic selection and must not be called population estimates.

## Paired Checkpoint Changes

| Checkpoint minus Source | Mean | Median | 10%-trimmed mean | Positive / negative changes | Bootstrap 95% interval for mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transition step 36 | +0.649 | +0.604 | +0.615 | 184 / 16 | `[+0.582, +0.720]` |
| Pairwise step 90 | +1.897 | +1.777 | +1.804 | 200 / 0 | `[+1.794, +2.008]` |
| Owner-conditioned step 90 | +5.204 | +4.342 | +4.666 | 200 / 0 | `[+4.785, +5.652]` |

The means are not single-outlier artifacts. Removing any one boundary changes
the transition mean only within `[0.637, 0.653]`, pairwise within
`[1.876, 1.902]`, and owner-conditioned within `[5.115, 5.222]`. The gap
between owner-conditioned mean and trimmed mean shows a real upper tail, but no
single boundary owns the conclusion.

The 16 negative transition changes are small and occur at strong Source stops:
their mean Source margin is `-4.959`, and the most negative change is only
`-0.256`. Transition does not create a second competing behavior here; it
mostly leaves already strong stops strong.

## One Shared Boundary Ordering, Not Three Specialized Sets

Positive-margin membership is exactly nested:

- all 45 transition-positive boundaries are pairwise-positive;
- all 94 pairwise-positive boundaries are owner-conditioned-positive; and
- neither larger arm loses a smaller arm's positive boundary.

Checkpoint-minus-Source deltas also have high Spearman rank correlations:

| Delta pair | Spearman coefficient |
| --- | ---: |
| Transition vs pairwise | 0.821 |
| Transition vs owner-conditioned | 0.771 |
| Pairwise vs owner-conditioned | 0.952 |

The evidence is most naturally described as a common ordering of boundary
susceptibility plus increasing continuation strength. It does not show that
the complete-row objectives discovered a distinct set-aware class of terminal
states.

## What Predicts the Shift?

Spearman correlations between checkpoint-minus-Source delta and panel fields:

| Field | Transition | Pairwise | Owner-conditioned |
| --- | ---: | ---: | ---: |
| Source pairwise margin | +0.700 | +0.682 | +0.666 |
| Prefix depth | -0.303 | -0.383 | -0.343 |
| Annotation object count | -0.091 | -0.164 | -0.126 |
| Verified remaining-owner count | +0.054 | +0.003 | +0.038 |

The strongest association is with the pre-existing Source decision state:
boundaries already closer to continuation receive larger changes. Earlier
prefixes also tend to receive larger shifts. Scene object count is weak, and
the number of owners still missing is effectively unrelated to the shift.

This is evidence against interpreting the continuation margin as a count-like
unfinished-set readout. It remains possible that the model represents owner
content in a way not summarized by the census count, but these objectives do
not scale the margin with the observable amount of remaining work.

## Source-Stop Strength Strata

| Source diagnostic state | Boundaries | Remaining owners | Transition positive | Pairwise positive | Owner-conditioned positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Positive despite historical stop | 2 | 4 | 2 | 2 | 2 |
| Terminal ahead by <0.5 | 26 | 133 | 26 | 26 | 26 |
| Terminal ahead by 0.5--2.0 | 67 | 290 | 17 | 58 | 67 |
| Terminal ahead by at least 2.0 | 105 | 353 | 0 | 8 | 50 |

Threshold crossing is therefore governed primarily by original stop strength.
Transition crosses every near-boundary case but no strong-stop case. The
owner-conditioned arm is strong enough to cross 50/105 strong Source stops,
consistent with its known excessive continuation and output expansion.

The effect is not confined to one depth or density stratum. Transition-positive
fractions are 12/52, 13/53, 11/51, and 9/44 over depth bins `1--2`, `3--5`,
`6--9`, and `10+`. All object-density bands contain positive crossings. The
aggregate is broad within this panel even though earlier prefixes receive
larger changes on average.

## Diagnostic Inconsistencies

Two historical Source terminal states have positive FP32 two-token margins:

- image 235809, depth 13, margin `+0.0089`, with two remaining cars;
- image 484369, depth 2, margin `+0.1836`, with two remaining potted plants.

Both are below `+0.5`. This small discrepancy is sufficient to enforce the
claim boundary: the diagnostic score does not reconstruct the historical
generation decision. Numerical precision, other-token competition, and
generation-time processing are not controlled by a two-token comparison.

## Consequence for the Later Release Study

The statistically aligned experiment is a **paired current-runtime replay on
all 200 boundaries**, not a release only on the 45 transition-positive cases:

1. Source native one-row release under the same runtime used by the forced arm;
2. Source with only the canonical row opener forced, then one-row release;
3. primary outcome: whether the forced arm produces any verified uncovered
   physical owner that the paired native arm does not;
4. secondary outcomes: another uncovered owner already produced natively,
   covered-owner repeat, unmatched or ambiguous valid row, invalid/incomplete
   row, and terminal behavior;
5. report per-boundary macro results, owner/category composition, Source-margin
   strata, remaining-owner-count strata, and paired gain/loss transitions.

Filtering to diagnostic-positive states would condition on the strongest
predictor of treatment response and overstate the general forced-continue
effect. An `intended owner` primary would be undefined for 145 multi-owner
boundaries and would change the research target. Transition can be added as a
checkpoint comparison, but it is not required to answer the primary Source
forced-continuation question.

## Supported, Ruled Out, Unresolved, and Not Claimed

### Supported

- The checkpoint treatments move a shared ordering of terminal boundaries at
  increasing strength.
- Step 36 creates 43 new positive two-token margins beyond the two Source
  diagnostic inconsistencies, but only 9/200 boundaries exceed `+1.0`.
- The shift is broad across panel strata and is not driven by one outlier.
- Observable remaining-owner count does not explain change magnitude.
- A later release study must use paired current-runtime arms and any-new-owner
  recovery as its primary outcome.

### Ruled out or materially weakened

- The three trained checkpoints do not select three disjoint or specialized
  terminal-state populations in this panel.
- The continuation change is not proportional to the number of verified
  remaining owners.
- Positive two-token margin cannot be called a decode flip.
- An arbitrary single intended owner is not an aligned primary outcome for this
  multi-owner panel.

### Unresolved

- How often forced continuation produces any new verified owner rather than a
  repeat, another already-produced behavior, invalid output, or unmatched
  geometry.
- Whether transition changes that forced-release distribution relative to
  Source.
- Whether the deterministic 200-boundary panel represents other Source stops.

### Not claimed

- No owner recovery, free-rollout set gain, loss recipe, architecture, prompt
  change, or production decoding rule is supported by this statistics-only
  unit.

## Stop

The read-only analysis is complete. No release, training, or architecture work
follows automatically; the paired release protocol above awaits user
discussion and approval.

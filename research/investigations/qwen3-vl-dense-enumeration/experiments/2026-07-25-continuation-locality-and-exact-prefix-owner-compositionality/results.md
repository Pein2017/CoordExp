---
title: Continuation-Shift Locality and Exact-Prefix Remaining-Owner Compositionality Results
description: Completed no-training evidence on whether continuation changes are specific to trained prefixes and whether sampled-reachable remaining owners survive exact self-prefix composition.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality
topic: qwen3-vl-dense-enumeration
status: complete_ready_for_user_discussion
evidence_status: verified_bounded_global_continuation_and_conditional_composition
updated: 2026-07-25
---

# Continuation-Shift Locality and Exact-Prefix Remaining-Owner Compositionality Results

## Decision

The two authorized no-training measurements are complete.

First, the checkpoint continuation change is **not concentrated at the exact
prefixes used by training**. On nonterminal complete-row boundaries, the
checkpoint-minus-Source change in new-row-opener versus terminal log
probability is almost the same at trained prefixes, different prefixes from
the same images, and matched prefixes from untouched images. Natural terminal
states with a verified remaining owner move less, but still move toward
continuation. The evidence therefore supports a broad, state-dependent
continuation shift rather than a training-prefix-local correction.

Second, the 400-case atlas shows that **sampled-reachable remaining-owner
evidence is usually already greedily realizable at the admitted exact
prefixes**. Source recovers the intended physical owner in 362/400 native
releases. Forcing only the new-row opener changes that to 363/400; the sole
gain is also the atlas's sole actual Source terminal case. Forcing the complete
verified description before releasing geometry changes recovery to 387/400.
This separates two facts: forced continuation can expose a missed true
positive, but in this selected atlas the larger remaining bottleneck is which
owner description is chosen, not whether a supplied owner description can be
grounded geometrically.

The strict two-owner subset provides bounded compositional evidence. After one
verified new-owner row is appended, the fixed candidate likelihood and
continuation margin for a second owner usually decrease, yet second-owner
released recovery is mostly preserved and sometimes improves. This is real
short-horizon sequential compatibility, but it is heterogeneous and does not
support an additive owner score, an explicit covered-owner ledger, or a new
architecture.

The final task remains one `list all objects` prompt and one free
autoregressive completion. No optimizer update, prompt change, state carrier,
one-row-at-a-time task, or production forced-decoding rule was run or promoted.

## Frozen Evidence Scope

The immutable manifest contains:

- 1,040 locality-control boundaries per checkpoint: 480 different
  same-image nonterminal boundaries, 360 matched untouched nonterminal
  boundaries, and 200 untouched natural terminal boundaries with at least one
  verified remaining owner;
- one unique image per locality-control boundary within each cohort;
- 1,440 previously scored trained exact-prefix events per checkpoint;
- 400 prefix-by-owner atlas cases spanning 311 images and 382 exact prefix
  states, including 285 same-category multi-instance cases; and
- only 17 exact prefix states from 16 images with multiple admitted remaining
  owners, yielding 38 ordered owner pairs.

Manifest SHA-256 is
`3f0fb0a56463e66ace4fab89fc670d4113f0a727ed841a292f265e3f67523e87`.
The final reduction is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality/reduction-v4/summary.json`
with SHA-256
`2096e502cbd77f46f7acb8b7fffa3cf91940ca1811e02fe0b2dae2f6fa8ca71a`.

All scored states preserve literal prompt and prefix token IDs and hashes.
Every prefix ends at a complete row boundary. Source and compared checkpoints
use identical images, prompts, prefixes, candidate rows, FP32 runtime, and
physical batch size one. All 16 full receipts passed manifest-hash, shard,
count, and unique-ID checks.

Atlas admission is intentionally selective: each target owner already had a
verified positive sampled row at that exact prefix in the frozen trajectory
evidence. The recovery rates therefore measure greedy recovery of
sampled-reachable owner evidence. They are not prevalence estimates over all
annotated remaining owners or over arbitrary model stops.

## Result One: Continuation Change Is Broad, Not Prefix-Local

The estimand is the checkpoint-minus-Source change in
`log P(new-row opener) - log P(terminal)`. The last two columns subtract the
matched control change from the trained-prefix change.

| Checkpoint | Trained exact, n=1,440 | Same-image nonterminal, n=480 | Untouched nonterminal, n=360 | Untouched terminal, n=200 | Trained minus same-image, 95% bootstrap interval | Trained minus untouched, 95% bootstrap interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transition step 36 | +1.774 | +1.785 | +1.788 | +0.649 | +0.029 `[+0.006, +0.055]` | -0.053 `[-0.158, +0.052]` |
| Pairwise step 90 | +8.056 | +8.014 | +7.671 | +1.897 | +0.047 `[+0.012, +0.085]` | +0.238 `[+0.053, +0.424]` |
| Owner-conditioned step 90 | +13.457 | +13.406 | +13.138 | +5.204 | +0.078 `[+0.026, +0.134]` | +0.067 `[-0.219, +0.366]` |

Every checkpoint raises the continuation margin at every nonterminal control
boundary. The small matched differences are negligible relative to the
checkpoint shifts themselves. Pairwise and owner-conditioned checkpoints also
raise all 200 untouched terminal controls. Transition raises 184/200 and
lowers 16/200, with a positive mean. This rules out the interpretation that
the known continuation expansion is mainly a memorized response to the exact
training prefixes.

Terminal attenuation is still meaningful. The checkpoint transformation is
not a perfectly constant logit offset: boundaries where Source naturally
stops retain structure that reduces the shift. The evidence identifies a
broad continuation tendency conditioned by boundary state, not a universally
identical change.

## Result Two: Remaining-Owner Recovery Ladder

Each row below reports a physical-owner match, not category text alone.

| Checkpoint | Native release | Force only new-row opener | Force complete verified description, then release geometry | Per-image macro native | Per-image macro description-forced |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 362/400 (90.50%) | 363/400 (90.75%) | 387/400 (96.75%) | 93.03% over 311 images | 97.41% over 311 images |
| Transition step 36 | 369/400 (92.25%) | 369/400 (92.25%) | 391/400 (97.75%) | 94.18% over 311 images | 97.91% over 311 images |

The paired intervention ladder is more informative than the marginal rates:

- Source opener forcing has 1 failure-to-success change, 0
  success-to-failure changes, and 37 cases that remain failures. Complete
  description forcing after native release has 25 failure-to-success changes,
  0 success-to-failure changes, and 13 remaining failures.
- Transition opener forcing changes no case because every admitted native
  release already opens a row. Complete description forcing after native has
  22 failure-to-success changes, 0 success-to-failure changes, and 9 remaining
  failures.
- Same-category multi-instance recovery after complete description forcing is
  275/285 for Source and 277/285 for transition. The supplied description can
  contain more information than the COCO category, so this is evidence for
  physical-instance grounding under that full description, not category-only
  instance selection.

### The one actual stop case

At image 243909, depth two, Source emits the terminal token while a verified
`bench` owner remains. Forcing only the canonical new-row opener then produces
a complete `bench` row matched to that owner. At the identical prefix,
transition step 36 natively produces and matches the same bench.

This is a clean existence proof that forcing continuation can recover a missed
true positive. It is not a rate estimate: the sampled-reachable atlas contains
only one Source-native terminal case by construction. The 200 terminal
locality controls were scored only for opener-versus-terminal probability and
were not released in this unit.

## Transition Step 36 Changes Scores More Than Released Recovery

Across the same 400 atlas cases, transition minus Source changes:

- continuation margin by +1.940 on all 400 cases;
- verified complete-row log probability sum by +0.857 on average, positive in
  358/400 cases;
- verified-description mean log probability by +0.026, positive in 372/400;
- verified-geometry mean log probability by +0.163, positive in 355/400; and
- native intended-owner recovery by 12 gains and 5 losses, for +7 net.

The corresponding opener-forced comparison is 11 gains and 5 losses (+6
net), and the description-forced comparison is 7 gains and 3 losses (+4 net).
Thus transition broadly improves the verified row's likelihood while released
physical-owner recovery moves only modestly and with owner churn. This agrees
with the earlier conclusion that a likelihood improvement is not equivalent
to final set-level gain.

## Strict One-Step Composition

For each ordered pair A then B, B is first evaluated at the original exact
prefix and then after appending a verified A row. The oracle branch appends the
previously sampled verified A row. The model-realized branch forces A's full
description and appends the actual row whose geometry and closure the model
generates; it is not a fully free-generated A row.

| Checkpoint and first-row branch | Ordered pairs | B complete-row log-probability change | B continuation-margin change | B opener-forced recovery before -> after | B description-forced recovery before -> after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source, sampled verified A | 38 | -2.988 | -0.926 | 15/38 -> 22/38 | 31/38 -> 32/38 |
| Transition, sampled verified A | 38 | -3.339 | -0.882 | 18/38 -> 22/38 | 33/38 -> 32/38 |
| Source, model-realized A after forced description | 31 | -1.959 | -0.469 | 11/31 -> 20/31 | 24/31 -> 27/31 |
| Transition, model-realized A after forced description | 33 | -2.701 | -1.090 | 13/33 -> 19/33 | 28/33 -> 29/33 |

The means hide real exchange. In the sampled-A branch, Source opener-forced B
recovery has 11 gains, 4 losses, 11 retained successes, and 12 retained
failures. Transition has 11 gains, 7 losses, 11 retained successes, and 9
retained failures. Description-forced recovery remains high but also churns.

The bounded interpretation is:

1. appending A does not erase B's usable physical-owner support;
2. sequential context can make B easier to release even while the fixed
   teacher-forced B row becomes less likely in absolute terms;
3. composition is not monotone for every owner pair; and
4. candidate likelihood and released owner reachability are distinct
   measurements and should not be substituted for one another.

Representative reviewed gains include `person -> car`, `dining table -> fork`,
and one `backpack -> another backpack` direction. Reviewed losses include
`person -> cup` and the reverse direction of that two-backpack state. These
examples confirm both real sequential compatibility and order-sensitive
failure; they do not justify an explicit set-memory claim.

## Supported, Ruled Out, and Unresolved

### Supported within this evidence scope

- The completed checkpoint treatments create a broad continuation shift at
  untrained as well as trained complete-row boundaries.
- At sampled-reachable exact prefixes, most remaining-owner rows are already
  greedily accessible; supplying the complete owner description closes most
  of the remaining recovery gap.
- Forced continuation can recover a missed true positive in at least one clean
  natural-stop case.
- One verified owner can be appended without generally destroying released
  reachability for another admitted owner, although likelihood and recovery
  move differently.

### Ruled out or materially weakened

- The continuation effect is not primarily specific to the exact training
  prefixes.
- Opener forcing is not a large general recovery mechanism on an atlas where
  native decoding already continues.
- Higher candidate-row likelihood cannot be treated as a sufficient proxy for
  released physical-owner recovery or one-step compositional success.
- The evidence does not require a new input prompt, output contract, or state
  carrier.

### Still unresolved

- The probability that forced continuation recovers a true positive over a
  representative population of actual Source stops. This unit has one released
  stop, not 200.
- Whether the same composition pattern holds for arbitrary annotated remaining
  owners rather than owners admitted by prior positive sampling.
- Whether the 38 ordered pairs from 17 exact states generalize beyond this
  small, selected composition subset.
- Whether a future training objective can improve final unique-owner coverage
  without converting the broad continuation shift into low-yield output
  expansion.

## Stop and Discussion Boundary

The unit's stop condition is met. No training or architecture change follows
automatically.

The most direct discussion candidate is a bounded release study on the 200
already-scored untouched natural terminal boundaries: force only the opener,
then classify intended-owner recovery, another uncovered owner, covered-owner
repeat, invalid row, and unmatched geometry. That would answer the user's
forced-continue probability question on an actual-stop cohort. It is a
candidate next unit, not authorized continuation of this completed one.

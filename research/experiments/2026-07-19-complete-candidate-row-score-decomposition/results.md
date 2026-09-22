---
title: Complete Candidate-Row Score Decomposition under Paired Prefix States Results
description: Full-model float32 evidence that same-covered-set initial order and downstream route state can redirect candidate geometry without a uniform terminal, description, or covered-set effect.
type: investigation
role: research-results
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: not_promoted
unit_id: 2026-07-19-complete-candidate-row-score-decomposition
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded_panel
updated: 2026-07-19
---

# Complete Candidate-Row Score Decomposition under Paired Prefix States Results

## Verdict

At the two initial-boundary controls, changing the order of the same complete
prefix rows can redirect a particular physical continuation before free
decoding. At later boundaries, the compared states also contain different
generated histories, so they measure downstream route-state effects rather
than order alone. Across the four frozen cases, neither comparison supports one
universal failure such as terminal-token dominance, description uncertainty,
or absence of all covered-object suppression.

The clearest effects occur at local branch tokens and inside the coordinate
continuation:

- image `18380` first flips the description choice from `person` to `cup`.
  Conditional on the shared `cup` description, it then raises the uncovered
  cup row by `+1.1640` summed natural-log units, almost entirely through `x1`
  (`+1.1075`), while the already covered cup becomes slightly worse
  (`-0.2483`);
- image `19109` shows a downstream route-state switch among same-description
  motorcycle geometries. The current route gives every coordinate token of
  its generated unmatched row vocabulary rank `1`; the alternative route
  instead gives every coordinate token of an already covered `gt_0021` row
  rank `1` along that row's own teacher-forced coordinate history;
- image `9590` changes neither the terminal margin nor any complete candidate
  row materially at the sampled terminal branch; and
- image `9400` mostly changes the shared `cup` description score for both an
  uncovered and a covered cup, while the covered row remains geometrically
  much worse.

The bounded mechanism conclusion is:

> The native prefix is a route-conditioned scoring state. It can preserve
> useful local covered-owner suppression, but it can also redirect
> same-category physical geometry toward a previously covered object. Its
> effect is not reducible to one terminal logit, one description choice, one
> coordinate, or a stable covered set.

No training arm or explicit state architecture is promoted from four cases.
The result rejects terminal-only calibration and description-only contrastive
training as the next default. It also shows that complete-row sums alone are
not aligned with the first greedy branch: later easy tokens can compensate for
the token at which two candidate paths diverge. If a training screen is later
authorized, its primary signal should compare the local margin at the earliest
candidate-distinguishing token under a real own-prefix state. Complete-row
consistency can remain a secondary signal, with uncovered-versus-covered
identity and per-coordinate diagnostics retained.

## Execution Evidence

The scorer used the geometry-sorted pure-cross-entropy plus token-type-gate
Weight-Decomposed Low-Rank Adaptation checkpoint at step `4,887`. Each image
ran in a separate Hugging Face process. The model contained
`2,149,097,472` parameters and every reported parameter was loaded as
`torch.float32`. Score accumulation and selected-token ranking also used
32-bit floating point. Processor-versus-model vision parameters, expanded
prompt tokens, image hashes, and no-resize image grids passed runtime checks.

No decoding policy transformed the logits: repetition penalty, top-p
filtering, terminal suppression, and coordinate smoothing were absent.

| Image | Immutable receipt | SHA-256 |
|---|---|---|
| `18380` | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-complete-candidate-row-score-decomposition/smoke-image18380-fp32-v1/receipt.json` | `630fbb672a2c73874059d69925b0c59cb50992da3e707ae7911dde58c0c1b00f` |
| `9400` | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-complete-candidate-row-score-decomposition/matrix-image9400-fp32-v1/receipt.json` | `e9757aac38f9361e59aa198ef229f32920bdcd4902984dca18b7f3a5afd3cead` |
| `9590` | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-complete-candidate-row-score-decomposition/matrix-image9590-fp32-v1/receipt.json` | `aaeb193e1434a744353e286b1b6f566c71130f4e071184abb3fb13dc57e58564` |
| `19109` | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-complete-candidate-row-score-decomposition/matrix-image19109-fp32-v1/receipt.json` | `85f75f1b412bc69a16c46d8405d51760e9c05bebb9e59867018db448130461e3` |

The implementation is
[`run_complete_candidate_row_scoring.py`](../../../../../scripts/research/run_complete_candidate_row_scoring.py).
It resolves exact Stage-2 token selectors, validates declared token hashes,
uses the current native inference frontend and Hugging Face backend session,
and writes an immutable receipt.

## Paired Score Changes

Every value below is:

```text
alternative prefix score - current prefix score
```

Larger values mean that the exact frozen candidate becomes more likely under
the alternative prefix. Complete-row values compare the same candidate with
itself, so row length is held fixed. The terminal column is the change in:

```text
log p(object-row-start) - log p(terminal)
```

It is not compared numerically with a complete-row sum.

| Image | Candidate | Role | Terminal-margin change | Complete row | Description | `x1` | `y1` | `x2` | `y2` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `18380` | person `gt_0019` | uncovered | `-0.1250` | `+0.0257` | `-0.0700` | `+0.1260` | `-0.0053` | `-0.0190` | `-0.0061` |
| `18380` | cup `gt_0018` | uncovered | `-0.1250` | `+1.1640` | `+0.0568` | `+1.1075` | `-0.0050` | `+0.0008` | `+0.0041` |
| `18380` | cup `gt_0017` | covered | `-0.1250` | `-0.2483` | `+0.0568` | `-0.3369` | `-0.0063` | `+0.0245` | `+0.0137` |
| `9400` | generated laptop | diagnostic | `+0.0172` | `-0.0941` | `-0.2100` | `+0.0864` | `+0.0405` | `-0.0033` | `-0.0078` |
| `9400` | cup `gt_0021` | uncovered | `+0.0172` | `+0.9106` | `+0.8026` | `+0.0853` | `+0.0079` | `-0.0049` | `+0.0196` |
| `9400` | mouse `gt_0018` | uncovered | `+0.0172` | `+0.0180` | `+0.0759` | `-0.0423` | `-0.0040` | `-0.0035` | `-0.0081` |
| `9400` | cup `gt_0016` | covered | `+0.0172` | `+1.0814` | `+0.8026` | `+0.6331` | `-0.4896` | `+0.0448` | `+0.0904` |
| `9590` | spoon `gt_0026` | uncovered | `-0.0390` | `-0.0601` | `-0.0261` | `-0.0194` | `+0.0035` | `+0.0031` | `-0.0075` |
| `9590` | cup `gt_0022` | uncovered | `-0.0390` | `-0.1044` | `-0.0192` | `-0.0053` | `-0.0555` | `-0.0148` | `+0.0040` |
| `9590` | cup `gt_0024` | recently emitted | `-0.0390` | `-0.2016` | `-0.0192` | `-0.0685` | `-0.0785` | `+0.0026` | `-0.0245` |
| `9590` | generated spoon | recently emitted | `-0.0390` | `-0.0236` | `-0.0261` | `+0.0015` | `+0.0147` | `+0.0032` | `-0.0032` |
| `19109` | motorcycle `gt_0015` | uncovered | `+1.9703` | `+2.8454` | `-0.1677` | `+0.2473` | `+2.7725` | `+0.0600` | `-0.0671` |
| `19109` | generated unmatched motorcycle | diagnostic | `+1.9703` | `-0.9949` | `-0.1677` | `-0.2267` | `-0.5477` | `+0.0642` | `-0.1173` |
| `19109` | motorcycle `gt_0021` | covered | `+1.9703` | `+2.6803` | `-0.1677` | `+0.1428` | `+1.7290` | `+0.7851` | `+0.1908` |
| `19109` | motorcycle `gt_0017` | covered | `+1.9703` | `+4.8222` | `-0.1677` | `+0.3358` | `+4.2753` | `+0.3198` | `+0.0587` |

Wrapper and closure tokens are nearly deterministic throughout and do not
explain the paired changes.

## Case Interpretation

### Image 18380: clean selective geometric redistribution

The object-row-start versus terminal margin stays very high under both
prefixes (`11.3693` and `11.2443`). At the first description token, the
current prefix ranks `person` first and `cup` second, while the alternative
prefix ranks `cup` first and `person` second. This local rank flip, not the
complete-row sum, predicts the actual greedy category. Conditional on `cup`,
the alternative order makes the uncovered cup's `x1` token move from
vocabulary rank `26` to rank `1`, while the covered cup's `x1` becomes worse,
rank `68` to rank `82`.

This is the cleanest evidence in the panel that the same covered set can first
redirect category choice and then, within the chosen category, selectively
redirect one physical instance. It does not merely encourage another row or
all cup geometries. It also agrees with the short-horizon observation that both
routes can remain valid and reconverge.

### Image 9400: category and geometry are not the same state

The terminal margin is unchanged. Reversing earlier rows raises the
description score of both cup candidates by the same `+0.8026`, but does not
make their coordinates equivalent. The covered cup remains much worse than
the uncovered cup, especially at `x1`, despite its total improvement.

This case shows that a prefix can alter category-level readiness without
proving a physical-instance coverage update. The oversized laptop row is a
diagnostic control only and remains nearly unchanged. No training conclusion
is drawn from it.

### Image 9590: sampled terminal output is not a raw argmax reversal

Both route states still favor object-row-start over terminal by approximately
`0.6` natural-log units. Every same-candidate complete-row change is at most
`0.2016` in absolute value. Among the two equal-length spoon rows, the
remaining spoon stays above the recently generated spoon in both states.

The alternative trajectory's sampled terminal token therefore does not show
that terminal had the highest unmodified boundary score. It is consistent
with a low-margin stochastic branch. This case rejects using one sampled stop
as evidence for an intrinsic terminal-prior failure.

### Image 19109: downstream route state, same description, different geometry

Every candidate says `motorcycle`, so the description score is identical
within each prefix and cannot identify an instance. The compared row-3 states
descend from different earlier orders, but they also contain different
generated rows and newly covered owners. Their difference is therefore a
downstream route-state effect, not an isolated order-only effect.

The alternative state raises the continuation margin by `+1.9703` and strongly
reorganizes coordinate logits. Under the current route, the diagnostic
generated row has rank `1` at all four coordinates. Under the alternative
route, the already covered `gt_0021` row has rank `1` at all four coordinates
along its own teacher-forced path. These conditional ranks explain why each
free route is locally greedy-compatible; they do not make the covered row the
highest-scoring supplied complete row. At the alternative boundary, the
diagnostic unmatched row still has a better geometry sum (`-14.558`) than the
covered `gt_0021` row (`-16.284`).

The uncovered `gt_0015` row improves by `+2.8454`, mostly at `y1`, but remains
below several alternatives. Another covered row, `gt_0017`, improves even more
by `+4.8222`, also mostly at `y1`. This is incompatible with a clean rule that
order only redistributes probability from covered to uncovered instances.

Because the scene is dense and the unmatched generated row has imperfect
geometry, the result is case-level evidence about route-conditioned,
same-class coordinate routing. It is not an order-only causal estimate, a
population estimate, or evidence that turns the unmatched row into ground
truth.

## Hypothesis Decisions

### Rejected as a universal explanation

1. **Terminal-token dominance.** Three cases change the continuation margin by
   at most `0.1250`, while candidate geometry still changes. Image `9590`
   samples terminal even though row start has the higher raw score.
2. **Description-only competition.** Image `19109` changes physical geometry
   while all candidates have the same description. Image `18380` changes the
   two cup geometries in opposite directions despite their shared description.
3. **One-coordinate ownership.** The main clean change is at `x1` on image
   `18380`, but it is at `y1` on image `19109`. No one coordinate is a general
   completed instance identity.
4. **A stable covered-set ledger.** The image-`19109` alternative free route
   returns to a covered motorcycle and makes its local coordinate path
   greedy-compatible. Because the compared downstream states contain
   different generated histories, this unit supports route-sensitive
   recurrence but does not isolate an order-only ledger failure.

### Bounded support

1. Complete prefix rows create an executable, order-sensitive state before
   generation begins.
2. That state can alter a specific physical continuation through coordinate
   logits rather than only through row length or terminal probability.
3. Useful covered-owner suppression exists in some states, especially image
   `18380`, but it is local and route-dependent rather than a reliable set
   operation.
4. Sampling can expose a low-margin terminal or object branch that is not the
   raw greedy choice; sampled output must not be mistaken for the boundary
   argmax.

## Training and Architecture Decision

The panel has no single stable defect that authorizes a general 256-image
training screen. In particular:

- suppressing terminal would not select the correct physical object;
- training only the description would miss same-category coordinate routing;
- imitating one canonical order would penalize benign alternatives such as
  image `18380`; and
- a covered-region mask is not justified by one dense same-category case.

The most plausible future loss family is branch-aware own-prefix ranking. At a
real self-rollout boundary, its primary term would compare the earliest token
that distinguishes a verified uncovered path from the actual covered,
duplicate, malformed, or terminal competitor. A smaller complete-row term can
encourage coherent phrase-plus-geometry transcription after that branch is
chosen. This is a candidate treatment, not an authorized implementation: the
panel contains only one strong covered-versus-uncovered recurrence case and
the safe dense label cohort is still incomplete.

The next evidence-changing unit should test whether this exact defect recurs
on `4` to `8` crop-reviewed, same-category dense scenes under the pure
cross-entropy checkpoint. This is a selected mechanism gate, not a population
frequency estimate. It should use natural own-prefix states and ask:

```text
At the first token where two real rollout paths differ, does a verified
uncovered path lose to a covered or duplicate path, and does that local loss
remain compatible with the complete-row scores?
```

If the local branch failure recurs, authorize a short matched-budget
branch-aware ranking screen with complete-row consistency as a secondary
term. If local uncovered branches are already ranked safely but free rollout
still misses them, move the treatment later to rollout-state training rather
than adding an explicit covered-set carrier. If no recurrent signature
appears, close this loss route and return to data completeness or
representation-side intervention.

## Limitations

- Four deliberately selected images are mechanism cases, not prevalence
  evidence.
- Some supplied rows are diagnostic generated geometry rather than trusted
  physical-object targets.
- Complete-row sums are not a normalized distribution over candidates and are
  length-sensitive. Only the same candidate across prefixes, or candidates of
  identical frozen length, support direct numerical comparison.
- Teacher-forced complete-row scoring measures local accessibility, not the
  probability that free decoding will traverse the entire row without drift.
- This unit uses only the geometry-sorted pure-cross-entropy checkpoint. A
  matched random-order checkpoint remains a separate replication question.

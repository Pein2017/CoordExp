---
title: Individual-Trajectory versus Sampled-Union Object-Support Audit Results
description: Fixed-row evidence that safer, higher-coverage sampled routes exist while useful object support also remains distributed across multiple trajectories.
type: investigation-results
role: evidence-record
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: bounded_successor_recommended
unit_id: 2026-07-21-individual-trajectory-versus-union-support-audit
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: route_local_conservative_certificates
updated: 2026-07-21
---

# Individual-Trajectory versus Sampled-Union Object-Support Audit Results

## Verdict

Two mechanisms coexist in this twelve-image development cohort.

1. Greedy decoding sometimes chooses a strictly worse complete route. Three
   crop-reviewed images contain a sampled trajectory whose verified coverage
   lower bound exceeds greedy's most favorable coverage upper bound at the
   same row budget, while the sampled route's worst-case harmful-row count does
   not exceed greedy's confirmed harmful-row count.
2. Useful support is also distributed across trajectories. At the four-row
   budget, five images have a verified sampled-union lower bound greater than
   four. No single four-row trajectory can contain more than four unique
   owners, so complementarity is proven without knowing the exact best route.

These are conservative route-local certificates, not a claim that the
analyzer formally selected a program branch or that every exact
best-trajectory identity is audited. They provide enough evidence to recommend
a bounded next screen: positive-only, span-masked weighted self-imitation of a
certified better sampled trajectory against a within-image shuffled-reward
control. Whole-trajectory imitation is not claimed to recover the sampled
union or solve dense enumeration.

## Observed execution

The frozen panel executed one true greedy trajectory and sixteen sampled
trajectories for each of twelve human-refined images:

```text
12 images * 17 trajectories = 204 trajectories
```

- 202 trajectories ended with the native image-end action.
- 2 reached the 512-new-token limit, but both contained at least 32 complete
  rows; no branch-bearing budget was right-censored.
- 192 parser receipts were accepted directly.
- 12 were accepted with one dropped malformed fragment each.
- Matching was recomputed independently at row budgets 4, 8, 16, and 32.
- Entity ownership and box geometry were reviewed separately. Low-pixel cases
  were cropped with context and conventionally enlarged. Interpolation was
  used only to make existing pixels easier to inspect; it was not treated as
  new evidence.

Primary reviewed output:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-individual-trajectory-versus-union-support-audit/
support-audit-step4887-20260721a/analysis/reviewed-support-v3.json
```

Review decisions:

```text
research/investigations/qwen3-vl-dense-enumeration/experiments/
2026-07-21-individual-trajectory-versus-union-support-audit/
review-decisions.json
```

Independent audit and accepted corrections:

```text
research/investigations/qwen3-vl-dense-enumeration/experiments/
2026-07-21-individual-trajectory-versus-union-support-audit/audit.md
```

## Branch-bearing complete routes

The table reports lower-to-upper bounds at the same maximum number of complete
generated rows. An `uncertain` row receives no verified owner credit, remains
visible as unresolved potential, and contributes one possible owner and one
possible harmful row to the upper bounds.

| Image | Rows | Greedy coverage lower-upper | Greedy harm lower-upper | Sampled route | Sampled coverage lower-upper | Sampled harm lower-upper | Conservative gain lower bound |
|---|---:|---:|---:|---|---:|---:|---:|
| 5001 | 4 | 3-3 | 1-1 | seed 21015 | 4-4 | 0-0 | +1 |
| 5001 | 8 | 6-7 | 1-2 | seed 21015 | 8-8 | 0-0 | +1 |
| 7511 | 8 | 5-6 | 2-3 | seed 21012 | 8-8 | 0-0 | +2 |
| 7511 | 16 | 6-7 | 3-4 | seed 21012 | 8-8 | 0-0 | +1 |
| 2685 | 16 | 8-10 | 6-8 | seed 21008 | 13-14 | 2-3 | +3 |

The sampled route on image 5001 retains a conservative gain and safety at
adjacent budgets 4 and 8. The route on image 7511 does so at adjacent budgets
8 and 16. Image 2685 supplies the third distinct conservative route. The
certificate uses the sampled coverage lower bound against the greedy coverage
upper bound and the sampled harmful-row upper bound against the greedy
harmful-row lower bound; unresolved rows therefore cannot reverse its sign.

The difficult cases strengthen rather than weaken the interpretation:

- Image 5001 greedy row 3 is a shifted repeat of an already covered person.
  Greedy row 6 spans two people and remains unassigned instead of being forced
  into either owner.
- Image 7511 sampled row 4 is a real, distinct low-pixel person, but its box is
  loose and bottom-truncated. The corresponding greedy low-pixel row remains
  uncertain.
- Image 2685 contains a dense glass-and-bottle cluster. The sampled route finds
  additional real bottles while still producing imperfect extents; its main
  advantage is lower duplicate emission, not perfect geometry.

## Support distributed across trajectories

Five images provide a hard complementarity witness at four rows:

| Image | Verified sampled-union lower bound | Maximum owners in any one four-row trajectory |
|---|---:|---:|
| 1584 | 6 | 4 |
| 7511 | 5 | 4 |
| 13348 | 5 | 4 |
| 14439 | 7 | 4 |
| 16228 | 5 | 4 |

This proof does not depend on the provisional `C_best` field: one complete row
can introduce at most one physical owner. The larger-budget automatic summary
also has sampled-union lower bounds above current best-single lower bounds on
11 of 12 images at 32 rows, but exact best identities and magnitudes remain
provisional because many non-branch-bearing low-pixel rows were not reviewed.
That exploratory pattern is not used to choose the next treatment.

## Supported

- Useful object support is already present in the sampled policy. The current
  bottleneck is not simply that these objects are inaccessible to the model.
- Greedy decoding can commit to a weaker route even when a safer,
  higher-coverage complete route exists in the same model distribution.
- Sampling gain is not explained only by longer output: the certified gains
  survive equal complete-row budgets.
- Sampling gain is not explained only by duplicate spray: the certified
  sampled routes have fewer harmful rows than their greedy counterparts.
- Entity discovery can be substantially better than physical-extent recovery.
  A real owner may be found while one or more box boundaries remain shifted,
  loose, truncated, or contaminated by neighbors.
- Route quality and route complementarity are simultaneous properties. A
  treatment may improve greedy route choice without recovering every owner
  exposed by multi-trajectory bagging.

## Ruled out for this cohort

- The claim that all bagging-only owners exist only in an artificial union and
  no better complete sampled trajectory exists.
- The claim that sampled gains are merely a consequence of using a larger row
  budget.
- The claim that extra unmatched rows can safely be equated with entity
  hallucinations.
- The claim that a geometry error necessarily means the entity was not
  discovered.
- Returning immediately to a new vision backbone or detector because the
  current policy has no support for missed owners.

## Unresolved

- This selected twelve-image cohort does not estimate population-level
  frequency or effect size.
- The experiment does not identify which hidden state, layer, or visual route
  makes seed 21015, seed 21012, or seed 21008 preferable.
- It does not establish whether self-imitation generalizes beyond the sampled
  prefixes or merely memorizes route style.
- It does not determine whether a compact covered-set carrier is necessary.
- It does not solve inaccurate coordinate extent, neighboring-instance
  contamination, or same-class owner ambiguity.
- Exact whole-panel best-trajectory identities and 32-row owner counts remain
  provisional until every contender that could alter them is adjudicated.
- The analyzer deliberately emits route-local conservative certificates but no
  program-level branch recommendation. The next treatment is a bounded
  research recommendation, not a mechanically finalized Rule 1 verdict.

## Not claimed

- No final architecture is selected.
- No training efficacy, Average Precision improvement, or blind-test gain is
  claimed.
- The twelve images are development and validation evidence and must not enter
  training gradients.
- Sampling is not proposed as the final inference algorithm.
- The result does not say that one canonical object order should replace all
  other valid orders.

## Next discriminator

Run exactly one successor screen: positive-only, span-masked weighted
self-imitation on model-generated complete trajectories whose verified unique
owner coverage and safety exceed a paired baseline. Compare it with a
within-image shuffled-reward control that uses the same trajectories, token
mask, optimizer, and update budget but deliberately breaks the association
between trajectory quality and weight.

The primary question is whether the correctly weighted treatment moves native
greedy decoding toward higher unique-owner coverage without increasing
duplicates, unsupported entities, malformed rows, or geometry contamination.
If the shuffled control improves equally, trajectory quality was not the
causal training signal. If both fail while fixed-prefix likelihood improves,
the missing ingredient is not merely trajectory-level credit and the next
unit should move to local remaining-object completion or an explicit state
carrier.

## Reproducibility boundary

Source Git commit before experiment-local changes:

```text
c429041dbccac01ca53042fed598f65db398ac45
```

Experiment-local implementation hashes:

```text
9413a0d891daa59040b99043f719d5d7ee8b1007c260c00df030a98c589861c3  scripts/research/run_current_seeded_sampled_rollouts.py
4efab7f7f390176f8ddff4e72a1ff1a9e3d877b2797bed49ab701ddd69b8448a  scripts/research/analyze_individual_trajectory_union_support.py
00adb71200e42560fa5e4c59954cb0864d7ef95ff15bf87d884db584d7661f90  tests/research/test_run_current_seeded_sampled_rollouts.py
2cc26f3bf56a2fdf4396b56d3b37457df8a053763caa206c5df38493fc4e5824  tests/research/test_analyze_individual_trajectory_union_support.py
679ad454de1c4d204a1229a9f10b838e48f79fa2987d584996cd1a2cf89f0892  review-decisions.json
```

The reviewed output records immutable Secure Hash Algorithm 256-bit digests
for all nine rollout files, the annotation file, and the applied review
overlay.

Reviewed analysis output digest:

```text
6e09f27845ff9ccd73dba4c224b55e05de85cd594b9c92e3886cbadfc41d4721  reviewed-support-v3.json
```

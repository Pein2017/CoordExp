---
title: Sorted Supported False-Negative Native-Prefix Reachability Prevalence - Independent Review
description: Contract PASS and scientific narrowing for the verified 114-owner native-prefix prevalence result.
type: investigation
role: research-review
authority: non_normative_research
unit_id: 2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence
topic: qwen3-vl-dense-enumeration
status: complete
updated: 2026-08-03
---

# Independent Review

## Review boundary

The final contract audit examined the frozen unit, implementation, tests, and
immutable run:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence/
  20260803T231934Z/
```

The independent scientific review was performed by Fable-5 at `xhigh`, agent
`/root/audit_reachability_prevalence_science`. Its conclusion-bearing message
was emitted at `2026-08-03T23:10:12.247Z` with message identifier
`65828922-71ea-4a13-b5b7-93f55bd372e7`; a follow-up correction was emitted at
`2026-08-03T23:11:04.817Z` with identifier
`be267bec-4fac-40b0-bef7-b7b990c943b9`.

Both reviews were read-only. The lead reconciled them against current source
and immutable bytes before closing [results.md](results.md).

## Contract audit

The final audit returned **PASS** with no residual P0/P1/P2:

- exact `114` FN and `141` TP admission reproduced;
- all-114 and never-owner-rank-one competitor Wilson intervals and
  image/category strata verified;
- TP due roles and owner-weighted rank distributions verified;
- FN context-row-weighted histograms are explicitly distinguished from
  owner-level metrics;
- analyzer, report JSON/Markdown, owner records, visualizer, all seven local
  products, and all four external references pass their seals;
- `49` focused tests passed; Ruff and `git diff --check` passed; and
- all three local PNGs and the three unique referenced owner maps were
  visually inspected and readable.

The audit allowed only the frozen-panel descriptive channel prevalence and
selection of a prospective discriminator.

## Scientific audit

The scientific audit accepted the artifact and narrowed its interpretation.

### Due-timing asynchrony

The 49 owners with a favorable prefrontier surface do not have 49 favorable
due contexts. Their 184 favorable contexts are 178 ahead-of-frontier and six
root contexts, with zero at-frontier contexts. Only 26 owners remain favorable
at the exact crossing boundary.

### Matched-row selection bias

Exactly 12 of the 26 crossing cases have an exact next native row that
strict-matches a physical owner. Running those 12 alone would discard the 14
cases in which an unmatched row consumes the crossing opportunity. The 12
remain a stratum, not
the cohort.

### Exposure and singleton cautions

Any-context favorable prevalence increases with observed context count. Also,
some owner-rank-one events have no same-category competitor. These facts do
not invalidate the owner-level report, but they prevent reading the counts as
equal-exposure competition effects.

### Measurement-channel confounding

Among matched crossing rows, same-description versus different-description
cases determine where the target and emitted paths first diverge. They are
measurement channels, not identifiable levels of a factorial treatment.
Description release and coordinate realization must therefore be reported as
separate within-owner ladders, not pooled raw log probabilities.

## Review-owned successor constraints

The next probe must:

1. use all 26 U-bound crossing owners as primary, with L=`25` and exact U/L
   same-context=`24` as sensitivity;
2. retain both the 12 matched-row and 14 unmatched-row strata;
3. compare within-owner, within-context margins only;
4. separate first-divergent description-token release from target-conditioned
   coordinate realization;
5. treat same-description matched cases as coordinate-only observations;
6. keep starvation as a trajectory tag, not claim it was distinguished;
7. use native argmax replay as an alignment gate and stop after more than two
   quarantined cases; and
8. route one successor only if at least two thirds of interpretable crossing
   owners land in one failure branch. Otherwise close the favorable-surface
   route as a weak descriptor.

No review finding supports adding the parallel owner-commit mechanism now.

---
title: Scalable Annotated-Owner Shared DoRA Pilot, G0.4 v3 correction
description: A semantics-only successor that treats natural row order as a monitor rather than an admission invariant.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-09-01-scalable-annotated-owner-shared-dora-pilot-g0-v3
topic: qwen3-vl-dense-enumeration
status: planned
evidence_status: none
updated: 2026-09-01
---

# G0.4 v3 natural-order monitor correction

## Authority and unchanged contract

This successor incorporates the predecessor
[`unit.md`](unit.md), SHA-256
`72608aa1343584e0d7786a633047f0e152c2449b0d5dee4671f530b86ba2ea6b`,
except for the ordering-admission semantics replaced below. Its active
question remains:

> From the immutable four-coordinate `geo_sorted_xy` step-2444 Source, does
> actual-prefix annotated-positive training with fixed preservation in one
> shared, unmerged DoRA adapter improve image-disjoint screen-dev natural-greedy
> annotated-owner coverage over censored teacher-forced transcript learning on
> the identical adapter surface and dose, after a semantic-over-permutation
> cross-image gradient gate and under registered retention and termination
> gates?

The Source, `256 / 128` cohort, execution media, partial-label boundary,
primary estimand, min-64 signal gates, K1 whole-bundle null, objectives,
resources, G1 contrast, claim boundary, and scientific stop rules are
unchanged. This is a mechanical semantics correction, not a new arm or new
scientific evidence.

## Corrected ordering semantics

- `geo_sorted_xy=(x1,y1)` binds transcript construction and the declared
  quantized ordering key. It is a training preference, not a promise that a
  natural decode is monotone.
- A complete event-eligible Source row must contain four non-bool integer
  `coord_bins` in `[0,999]`. A missing or inconsistent key is mechanical
  identity drift; a valid natural `(x1,y1)` inversion is ordinary model
  behavior.
- Natural inversions never exclude an image, reject an exact Source
  preservation row, make a run mechanically invalid, or fire a current
  scientific stop. Preservation reuses the exact natural prefix and complete
  Source row regardless of its relation to earlier Source anchors.
- The ordering admission applies only to proposed positive rows: a positive
  key earlier than any exact-prefix committed key is rejected; equality is
  allowed.

For eligible images, define emitted anchors `s_i=(x1_i,y1_i)` in emission
order. Row `i` is a natural-order-violation row exactly when
`s_i < max_{h<i} s_h` lexicographically. The receipt reports the number of
such rows and the number of eligible images containing at least one; it does
not report or gate on inversion-pair count.

## First-crossing non-expansion lemma

For one grouped positive alternative with key `k`, choose the prefix before
`j=min{i: s_i >= k}` in emission order, or the terminal prefix when no such row
exists. Every row before `j` is `<k`, so the inserted candidate is not a
violation. If `j` exists, any later row `<k` was already a violation because
`s_j>=k`; if it does not, `k` is appended as a new maximum. Therefore this
choice preserves the set of violation rows. It does not claim that the number
of inverted row pairs is unchanged, and grouped candidates remain alternatives
rather than simultaneous insertions.

## Provenance and next gate

The v2 full attempt remains immutable `MECHANICAL_INVALID`: its StateBank
incorrectly applied positive-row ordering admission to a nonmonotone exact
preservation row and also used stricter full-XYXY/strict-greater semantics.
The later unlaunched hard-admission v3 draft is superseded and supplies no
evidence. The preserved Image102420 diagnostic capture is the mechanics
witness: it is event-eligible and has exactly one violation row under the
definition above.

No G0.3/G0.4 scientific result is established here. A fresh Image102420
mechanics smoke must show eligible image `1`, violation image/row counts `1/1`,
successful StateBank assembly, and admitted preservation before one immutable
full v3 G0.4 attempt may start.

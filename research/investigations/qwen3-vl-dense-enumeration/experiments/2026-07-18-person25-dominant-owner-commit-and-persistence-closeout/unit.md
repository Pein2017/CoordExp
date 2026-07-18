---
title: Person 25 Dominant-Owner Commit and Persistence Closeout
description: Tests whether the historical random-order adapter suppresses its previously dominant person after that same physical person is committed, where released outcomes go, and whether any suppression persists beyond one successor row.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-18-person25-dominant-owner-commit-and-persistence-closeout
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-18
---

# Person 25 Dominant-Owner Commit and Persistence Closeout

Execution is complete. See [results](results.md).

## Question

On image `2299`, the historical random-order pure cross-entropy adapter selects
person-only rank `25` in all 96 prior one-row samples after four different
non-25 histories. Does the same adapter still select person 25 after a complete,
naturally generated row for person 25 is appended to the prefix?

Person-only rank `25` is a zero-based rank among the 38 manually relabeled
people in image `2299`; it is not a category identifier or a global object
index.

## Competing explanations

1. **Static dominant candidate.** Person 25 remains dominant even after its own
   row is committed.
2. **Immediate physical-owner suppression.** Committing person 25 suppresses
   that physical person and transfers output to valid, not-yet-emitted people.
3. **Literal-row suppression.** Only an exact emitted token string is avoided;
   natural coordinate variants for the same physical person do not share the
   effect.
4. **Last-row spatial routing.** Person 25 is avoided immediately, but earlier
   committed people no longer affect the successor after the final row is held
   fixed.
5. **Persistent multi-row state.** With the same final person-25 row, changing
   an earlier committed person selectively changes later successor ownership.
6. **Row-depth effect.** Appending any fourth row changes the next-row behavior,
   regardless of whether that row belongs to person 25.
7. **Unhealthy suppression.** Person 25 falls, but the released behavior goes
   to terminal, malformed, unresolved, or immediately rebounding output rather
   than reliable uncovered people.

## Evidence boundary entering the unit

The verified historical screen used the matched checkpoint-3668 random-order
and geometry-sorted adapters in full-model 32-bit floating point. Four prefixes
ending in person ranks `2`, `3`, `4`, or `14` all led the random adapter to
person 25 in `24/24` paired samples. Person 25 itself was never a treatment
owner. The prior result therefore establishes a dominant uncommitted successor,
not failure of self-owner suppression.

## Primary panel: immediate dominant-owner commit

Use the historical random-order checkpoint-3668 adapter and the exact compact
prompt, image bytes, coordinate-token adapter, full-model 32-bit floating point,
eager attention, and candidate matcher from the verified historical screen.

1. Select the naturally sampled person-25 continuations at seeds `21`, `0`, and
   `9` from the existing random-owner-person-2 arm. They represent low-, middle-,
   and high-overlap person-25 geometry within the existing samples. Every donor
   must match physical person 25 under the frozen relabel.
2. Append each natural row to an exact historical parent state that previously
   generated person 25.
3. Run greedy decoding and the paired seeds `0..23` at temperature `0.4`,
   top-p threshold `0.95`, repetition penalty `1.0`, and one complete-row
   structural stopping.
4. Preserve the donor's seven raw generated token identifiers and append them
   directly. Do not decode and retokenize the donor. The receipt must verify
   token equality, the full prefix hash, source artifact path and seed, and the
   prompt, checkpoint, and image hashes.
5. Run a same-depth control that appends the canonical person-rank-3 row after
   `[0, 1, 2]`. This is a forced but valid random-permutation history; it owns
   only the row-depth control.
6. Preserve raw output before matching. Classify person-25 repeat, earlier
   committed-person repeat, other uniquely matched person, `tie`, unresolved
   geometry, malformed output, and terminal output. Retain generated `x1`,
   continuous Intersection over Union to person 25, and nearest-owner identity.

The existing uncommitted `96/96` person-25 result is context. The same-depth
non-25 final-row control is the primary baseline for owner-specific suppression.
It must retain at least `20/24` strict person-25 matches; otherwise the treatment
contrast is narrowed to an unresolved row-depth effect. The three donor variants
are robustness treatments, not independent experimental replicates, and remain
separate in reporting.

## Conditional panel: persistence and earlier-history influence

This panel runs only if the immediate person-25 recurrence rate materially
collapses without an increase in terminal, malformed, or unresolved output.

1. Continue three complete successor rows and record the first lag at which
   person 25 returns. This is a descriptive rebound phenotype, not sufficient
   evidence for persistent covered-state memory because intervening rows diverge.
2. Where one exact natural person-25 row is supported after each prior treatment
   history, hold that final row fixed and compare histories whose earlier
   treatment person is rank `2`, `3`, `4`, or `14`.
3. Test whether an earlier person is less likely to reappear specifically in
   the history that already contains that person after the final person-25 row
   has been made identical.

This separates one-row self-suppression and last-row routing from evidence that
an earlier committed owner remains causally active.

## Executed adaptive extensions

The immediate count gate passed, but coordinate inspection showed that the
rows assigned to person 25 after its donor were vertically merged boxes rather
than clean person-25 instances. The authorized research loop therefore added
three bounded extensions inside the same question:

1. a 96-seed, token-identical-final-row comparison that placed dominant
   successor person 18 in the earlier history, with nearby and distant earlier
   person controls;
2. a three-row descriptive return-time panel;
3. one fixed-prefix full-model 32-bit floating-point next-token probe at the
   final `y2` decision, comparing the single `coord_999` spike with aggregate
   probability over the person-18 boundary neighborhood.

These extensions do not estimate prevalence or authorize training. They close
the person-25 interpretation boundary exposed by the primary result.

## Interpretation rules

- Avoiding person 25 for one row is not sufficient evidence for a covered set.
- Immediate suppression is promoted only if all three donor arms reduce strict
  person-25 recurrence by at least `12/24` relative to the paired same-depth
  control, while at least `22/24` outputs in every arm remain valid rows and the
  continuous-overlap diagnostics agree with the strict-match result.
- Released outcomes count as healthy redistribution only when they are valid,
  uniquely assigned, not-yet-emitted physical people.
- Variant sensitivity is reported directly; exact-string avoidance is not
  promoted as physical-owner memory.
- A three-row rebound is reported as short-lived suppression, not persistent
  commit.
- Earlier-history effects own a persistent-state claim only when the final
  person-25 row is token-identical across histories.
- Teacher-forced complete-row scores cannot substitute for free autoregressive
  successor behavior.

## Non-goals and stop rule

This is one image, one historical training seed, and one checkpoint pair. The
person-25 donors are naturally sampled continuations conditional on a forced
canonical prefix, not end-to-end natural rollout states. This unit
does not estimate prevalence, authorize training, prove an explicit ledger, or
promote architecture. Stop after the immediate panel if person 25 remains
dominant in at least `20/24` samples for every donor while the same-depth control
passes. Stop as unresolved if the same-depth control fails, donor variants
disagree materially, or released outcomes are mainly unresolved or invalid.
Stop after the conditional panel if suppression is only transient or
the common-final-row comparison is unsupported. In either case, proceed to the
separate same-prefix phrase-and-box recoverability question rather than
expanding this image into a broad campaign.

## Reused implementation and artifact handle

Reuse model loading, prompt reconstruction, coordinate-token restoration,
row parsing, and human-relabel matching from:

```text
scripts/research/run_historical_random_sorted_image2299_screen.py
```

Durable execution artifacts resolve under:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-18-person25-dominant-owner-commit-and-persistence-closeout/<run-id>/
```

Each run retains checkpoint and adapter hashes, prompt and image identity, exact
prefix token identifiers, person-25 donor provenance, seeds, decode policy, raw
rows, parsed outcomes, matching results, and terminal status.

---
title: Sorted Crossing Matched-Length Neutral-Row Insertion Control - Results
description: Verified twelve-image score-only control showing that the selected covered-row insertion is itself non-neutral and that its damage is strongly associated post hoc with sorted-route regression distance.
type: investigation
role: research-results
authority: non_normative_research
unit_id: 2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-08-04
---

# Results

## Verdict

The unit is complete and independently audited. Its frozen route is:

`neutral_row_not_neutral_inconclusive`

The deterministic covered-row control `N` is material on `8/12` frozen
specificity owners, exceeding the absolute failure threshold of four. The
specificity gate therefore blocks both scientific routes before the voting
contrast is interpreted. Material `N` is not evidence that generic row
insertion explains the earlier `P+C -> E` tail, and nonmaterial `N` in three
voting cases is not enough to establish a `C`-specific mechanism.

This is a failure of the selected neutral-control design, not a failed runtime
or a null model result. The strongest remaining explanation is that repeating
an already covered row perturbs the model's learned sorted trajectory, with
larger backward jumps producing larger downstream exact-coordinate damage.
That explanation is post-hoc and descriptive; it was not a frozen route.

## Evidence boundary

| Item | Executed evidence |
| --- | --- |
| Checkpoint and panel | Geometry-sorted step `4887`; frozen human-refined twelve-image panel only |
| Final run | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control/20260804T093553Z/` |
| CPU plan | `9` voting owners, `12` specificity owners, `5` structurally infeasible owners |
| Score capture | `21` `P+N -> E`, `21` same-run `P+C -> E`, and `12` benign replay pairs; `54/54` complete |
| Runtime | Hugging Face full-model float32; one admitted real-HF smoke and twelve immutable image shards |
| Quarantine | `0` owners and no benign-blocked image |
| Readout | Exact teacher-forced likelihood of the same four sealed coordinate tokens of downstream row `E` |
| Excluded | Image `2299`, free decoding, sampling, owner matching, final-set coverage, training, and architecture |

The plan, shard, merge, and analysis artifacts use create-or-identical
publication and bind the exact source, checkpoint, tokenizer, prompt, image,
prefix, inserted-row, scored-row, and runtime identities.

## Frozen gates

The gates were evaluated in the preregistered order.

| # | Gate | Result |
| ---: | --- | --- |
| 1 | Input and token identity | pass |
| 2 | Runtime replay | pass; `21/21`, maximum selected-logit difference `0` |
| 3 | Same-run positive controls | pass; `21/21`, including both sentinels |
| 4 | Benign-reference replay | pass; `12/12` |
| 5 | Cached-versus-uncached parity | pass; representative smoke maximum difference `3.170967102050781e-05` |
| 6 | Neutral-row specificity | **fail**; `8/12` material at cutoff `<= -1.0` nat |

All four non-voting sensitivities preserve the same route: cutoffs `-0.75`
and `-1.25` nat, excluding row-length delta two, and excluding the two
support/extent-uncertain rows.

## Voting contrast, reported but not interpreted

Within the nine frozen `C`-material owners:

- `N` is nonmaterial for `3/9` owners across three images, including `2/4`
  clearly separated owners across two images;
- `N` is material for `6/9` owners across five images, including `2/4`
  clearly separated owners in one image; and
- no owner is quarantined or denominator-ineligible.

Neither frozen scientific route would have fired even if the specificity gate
were ignored. Gate six has precedence, so these counts do not adjudicate
content-specific versus generic insertion sensitivity.

## Post-hoc route-regression diagnostic

The independent scientific audit found a strong relationship between the
selected neutral row's `rows_back_distance` and its materiality. Reanalysis of
the frozen selection registry and owner rows reproduces:

- all `7` N-nonmaterial owners have `rows_back_distance <= 3`;
- `13/14` N-material owners have `rows_back_distance >= 5`;
- the sole exception is `gt:4134:29`, material at `rows_back_distance = 2`;
- the rule `rows_back_distance >= 4` predicts materiality for `20/21` owners;
  and
- Spearman correlation between `rows_back_distance` and the relative neutral
  coordinate delta is `-0.7510843921476228`.

The pattern is not merely inherited from the clean-`C` cohort. N materiality
has the same `67%` rate in both strata: `8/12` specificity owners and `6/9`
voting owners. The `N`/`C` crosstab is `4`, `3`, `8`, `6` for
`N-nonmaterial/C-nonmaterial`, `N-nonmaterial/C-material`,
`N-material/C-nonmaterial`, and `N-material/C-material` respectively.

This diagnostic suggests that the intervention disrupts a monotone sorted
route in proportion to how far it jumps backward. It does not establish that
route regression is the causal variable: row identity, duplication, position,
recency, and regression distance remain joined.

## Feasibility stop for a lower-regression sibling

A CPU-only scan of the already sealed candidate ledgers asked whether the same
predicates could support a new `N'` with `rows_back_distance <= 3`. Such a
candidate exists for only `5/12` specificity owners and `5/9` voting owners.
That is below the independently proposed minimum of approximately `8/12`
specificity owners needed to re-establish a useful neutrality gate.

No lower-regression GPU sibling is launched. Relaxing geometry, description,
covered-before-boundary, length, or score-blind selection after seeing this
result would change the estimand and invite adaptive control selection.

## Supported

- The chosen matched-length covered real row is not a valid neutral insertion
  control on this frozen panel.
- Exact downstream coordinate likelihood is highly sensitive to some duplicate
  sorted-route regressions.
- The implementation and runtime controls are comparable enough to trust the
  failed specificity result.
- The current repeated-covered-row insertion line should stop.

## Not supported

- No claim that generic row insertion is sufficient.
- No claim that `C`-specific interference survived or was refuted.
- No claim that every backward jump, duplicate, or sort violation has the same
  effect.
- No owner-emission, strict-match, final-set coverage, natural-stop,
  population-prevalence, training, architecture, or production claim.

## Program consequence

The broader false-negative program already has an owner-level prevalence
owner: the completed `114`-owner native-prefix reachability unit. This result
does not justify duplicating that census. Together, the current evidence says:

1. many native false negatives retain tested localization support;
2. STOP is not the dominant simple explanation in that supported cohort;
3. owner release and downstream row realization are path-sensitive at the
   crossing boundary; and
4. interventions that move backward through the learned sorted trajectory can
   themselves damage later exact-row accessibility.

The next research decision should therefore concern a deployment-compatible
training or causal contrast that preserves downstream suffix owners while
teaching recovery of a skipped owner. It should not be another selected
covered-row insertion control, and it must retain unique-owner gained,
retained, and lost accounting before any treatment or architecture is
promoted.

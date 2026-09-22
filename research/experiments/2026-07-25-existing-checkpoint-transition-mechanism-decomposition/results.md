---
title: Existing-Checkpoint Transition Mechanism Decomposition and Robust Evaluation Results
description: Completed Phase Zero evidence separating robust free-rollout owner changes, continuation versus stopping, conditional row ranking, row realization, full-row normalization and output expansion, and held-out owner churn.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-25-existing-checkpoint-transition-mechanism-decomposition
topic: qwen3-vl-dense-enumeration
status: complete_ready_for_user_discussion
evidence_status: verified_bounded_mixed_mechanism
updated: 2026-07-25
---

# Existing-Checkpoint Transition Mechanism Decomposition and Robust Evaluation Results

## Decision

Transition step 36 shows a **mixed checkpoint-level effect**:

1. it raises the model's preference for opening another row rather than
   stopping;
2. after the row opener is fixed, it usually raises the best verified
   uncovered-owner row relative to a plausible covered-owner row in the seven-
   case diagnostic panel; and
3. it does not improve the tested model's ability to realize the selected
   physical owner as a valid matched row after generation is released.

The unit disposition is therefore `post-continue-improvement`, with an
important qualification: the post-continue evidence is a change in relative
row likelihood on five of six eligible selected cases, not reliable released
owner recovery. The checkpoint also has a continuation-gate shift. It is not
`gate-dominant`, because the relative uncovered-versus-covered row movement
survives after continuation is fixed. It is not evidence for an explicit
covered-set representation, a production forced-continuation policy, an
objective recipe, or an architecture.

The original task remains unchanged: one `list all objects` prompt, one free
autoregressive completion, and final unique trusted physical-owner coverage.
No new training was run in this unit.

## Evidence at a Glance

| Lane | Observation | Decision |
|---|---|---|
| Robust free rollout | Heldout remains `46 gained / 39 lost / +7 net`, with 118 fewer predictions and 13 fewer strict duplicate candidates; the direction survives token cutoffs, natural-stop restriction, leave-one-image-out analysis, and matching-threshold sensitivity | Directional checkpoint benefit is not a decode-tail or single-image artifact; it remains too small and heterogeneous for promotion |
| Held-out owner review | The unchanged official geometry ledger is `+7`; blind entity and geometry review yields `22 genuine gains / 16 genuine losses / +6` | Most of the positive direction survives physical-owner review, but only 38 of 85 churn cases meet the strict human-refined rule |
| Same Source prefix | Continue-minus-stop increases on all 7 cases; best uncovered-minus-covered row score improves on 5 of 6 eligible cases; neither of 2 actual Source terminal states flips to continue | Both gate movement and post-continue conditional row-ranking evidence exist on the selected panel |
| Released row | Source recovers an uncovered owner after an opener-only force in 1 of 2 terminal cases; transition recovers 0 of 2. With a complete one-token description, Source realizes any uncovered owner in 4 of 6 cases and the intended owner in 3; transition obtains 3 of 6 and 2 of 6 | Forced continuation can expose a missed true positive, but no frequency estimate or transition realization advantage is supported |
| Full-row score normalization | On 1,440 common exact events, sequence sum, token mean, equal-group mean, and every 9/10/11-token stratum preserve the same checkpoint ordering | Sum-versus-mean normalization is not the main explanation |
| Full-row free rollout | Pairwise and owner-conditioned arms gain owners by 256 tokens, then add many predictions without further net gain; owner yield collapses and length stops concentrate the worst tails | Owner signal is present, but continuation/output expansion is the dominant failure mode of these arms |

## Lane One: Robust Free-Rollout Evaluation

The primary policy is unchanged Hugging Face greedy decoding with batch size 4,
3,084 generated tokens, repetition penalty 1.0, one prompt, and one completion.
Owner matching uses Intersection over Union 0.50 as primary.

| Cohort | Images | Source owners | Transition owners | Gained / lost / net | Source / transition predictions | Source / transition strict duplicate candidates | Source / transition owner yield |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development, all | 256 | 1,452 | 1,520 | 128 / 60 / +68 | 3,529 / 3,879 | 29 / 15 | 0.4114 / 0.3919 |
| Development, paired natural stop | 253 | 1,429 | 1,498 | 128 / 59 / +69 | 2,998 / 2,920 | 29 / 15 | 0.4767 / 0.5130 |
| Heldout, all | 128 | 729 | 736 | 46 / 39 / +7 | 1,591 / 1,473 | 30 / 17 | 0.4582 / 0.4997 |
| Heldout, paired natural stop | 127 | 724 | 731 | 46 / 39 / +7 | 1,439 / 1,273 | — | — |

The held-out result is not controlled by one decode tail:

- fixed generated-token cutoffs 256, 512, 1,024, 2,048, and 3,084 give net
  owner deltas `+12`, `+9`, `+7`, `+7`, and `+7`;
- matching thresholds 0.30, 0.50, and 0.75 give `+27`, `+7`, and `+14`;
- leaving out any one image keeps the net between `+4` and `+10`;
- the per-image owner-net median remains zero, so the aggregate gain is spread
  over a minority of changed images rather than a typical-image shift; and
- excluding the one treatment length stop leaves the primary owner net
  unchanged.

This strengthens the earlier statement from "promising aggregate" to
"robustly directional within this analyzed held-out cohort." It does not make
the held-out cohort a population confirmation set: the magnitude is small,
owner exchange remains, and the same cohort was used for mechanism case
selection and human review.

## Lane Two: Same-Prefix Mechanism Decomposition

### Frozen panel

The panel contains seven deliberately selected Source-produced exact prefixes
and 22 complete candidate rows. Each prefix, prompt payload, token sequence,
image identity, and candidate token sequence is hash-bound. The cases cover
two Source terminal states, clean gain, gain/loss exchange, pure loss, a multi-
car choice, an unchanged-owner control, and the treatment length-stop case.
This is a mechanism panel, not a random prevalence sample.

The continue-versus-stop value is
`log p(canonical row opener) - log p(end of assistant turn)`. The conditional
row-ranking value is the complete candidate suffix log likelihood after the
same canonical row opener has already been supplied. Candidate rows are scored
independently; these values are not a normalized probability distribution over
all possible next rows.

| Image | Source continue minus stop | Transition continue minus stop | Change | Change in best uncovered minus best covered suffix score |
|---:|---:|---:|---:|---:|
| 15,379 | -1.458 | -0.471 | +0.987 | +1.586 |
| 28,058 | 11.404 | 13.083 | +1.680 | +1.127 |
| 3,442 | -3.438 | -3.185 | +0.252 | -0.164 |
| 355,385 | 1.880 | 3.712 | +1.832 | unavailable: no verified owner-bearing uncovered comparator |
| 4,129 | 11.195 | 12.196 | +1.001 | +3.661 |
| 65,891 | 11.433 | 12.784 | +1.351 | +5.690 |
| 70,558 | 10.763 | 12.801 | +2.038 | +7.178 |

All seven continuation margins increase. Neither Source terminal state changes
sign, so this panel does not show that step 36 alone makes the model continue
at an actual stopping point. Five of the six comparable cases improve the best
verified uncovered-owner row relative to the best covered-owner row. That is
the bounded evidence for a conditional effect after continuation has been
fixed.

This relative movement cannot establish that the model computes a covered set.
Action identity and geometry differ across candidates, the candidate list is
incomplete, and independently scored rows do not compete through a shared
normalizer. A future matched training control must determine whether the effect
comes from prefix-relative owner status or from correlated row content.

### Forced and released generation

Two terminal cases were tested by supplying only the canonical row opener and
then releasing greedy generation. On image 3,442, Source generated a legal
`bottle` row matching previously uncovered owner `2095731`; it was neither a
repeat nor malformed. Transition instead generated an unmatched `wine glass`.
On image 15,379, both checkpoints generated unmatched `motorcycle` rows.

This proves existence: forcing "continue" can reveal a missed true-positive
owner rather than only a duplicate or malformed row. Two terminal cases cannot
estimate how often that happens, and the tested transition checkpoint does not
improve it.

Six cases also supplied the canonical opener plus a complete one-token object
description before coordinates were released:

| Checkpoint | Runs | Any uncovered owner realized | Intended owner realized | Covered-owner repeats |
|---|---:|---:|---:|---:|
| Source | 6 | 4 | 3 | 0 |
| Transition step 36 | 6 | 3 | 2 | 0 |

Both checkpoints realize the intended person on image 28,058 and the future
bird on image 70,558. Both select a different uncovered person than the
specified gain owner on image 4,129. Source alone realizes the intended car on
image 65,891. Neither realizes the intended chair on image 3,442 or car on
image 15,379. The observed transition advantage is therefore in conditional
row scoring, not released row realization.

## Lane Three: Full-Row Normalization and Fixed-Budget Reanalysis

The common projection contains 1,440 exact prompt, prefix, and owner actions.
The owner-conditioned bank contains 309 additional aliases; they are recorded
but excluded from the common comparison. Higher log probability is better.

| Checkpoint | Continue minus stop | Row sequence sum | Target-token mean | Schema and description mean | Coordinate mean |
|---|---:|---:|---:|---:|---:|
| Source | 10.082 | -9.751 | -1.044 | -0.0382 | -2.3866 |
| Transition step 36 | 11.856 | -8.766 | -0.938 | -0.0312 | -2.1493 |
| Pairwise, learning rate 3e-6, step 90 | 18.138 | -9.507 | -1.018 | -0.0394 | -2.3239 |
| Owner-conditioned, learning rate 1e-5, step 90 | 23.539 | -9.436 | -1.010 | -0.0429 | -2.3016 |

Sequence sum, target-token mean, equal weighting of the schema/description and
coordinate groups, and each 9-, 10-, and 11-token row stratum preserve the same
ordering: transition step 36, owner-conditioned, pairwise, then Source. The
full-row arms improve coordinate-token likelihood modestly and do not improve
aggregate schema/description likelihood. Normalization changes scale, not the
qualitative conclusion.

The 64-image free-rollout comparison shows why the full-row arms are not usable
despite containing owner signal:

| Checkpoint | Owners / predictions / owner yield at 256 tokens | Owners / predictions / owner yield at 3,084 tokens | Gained / lost / net at full horizon | Length stops |
|---|---:|---:|---:|---:|
| Source | 381 / 588 / 0.648 | 389 / 786 / 0.495 | reference | 0 |
| Transition step 36 | 385 / 572 / 0.673 | 392 / 631 / 0.621 | 21 / 18 / +3 | 0 |
| Pairwise, learning rate 3e-6, step 90 | 397 / 694 / 0.572 | 405 / 1,396 / 0.290 | 24 / 8 / +16 | 2 |
| Owner-conditioned, learning rate 1e-5, step 90 | 401 / 819 / 0.490 | 408 / 3,235 / 0.126 | 35 / 16 / +19 | 10 |

Pairwise already has `+16` net owners at 256 tokens and remains `+16` after
roughly doubling predictions. Owner-conditioned has `+20` at 256 and `+19` at
the full horizon while predictions grow from 819 to 3,235. Length-stopped cases
are the worst extremes, but paired natural-stop images also expand. The primary
failure is excessive continuation and output expansion, not merely the maximum
generation length and not merely sequence-sum normalization.

## Lane Four: Held-Out Owner Churn Review

The official geometry-derived ledger remains unchanged at 46 gains and 39
losses. Five arm-blinded 17-case review shards cover all 85 cases exactly once.
Entity/category and geometry were reviewed separately before unblinding.

| Human-refined interpretation | Gains | Losses | Net | Total |
|---|---:|---:|---:|---:|
| Genuine real-owner change with acceptable geometry | 22 | 16 | +6 | 38 |
| Real-owner change with geometry error | 11 | 6 | +5 | 17 |
| Duplicate | 3 | 6 | -3 | 9 |
| Category alias or disagreement | 3 | 2 | +1 | 5 |
| Uncertain | 7 | 9 | -2 | 16 |
| Unsupported | 0 | 0 | 0 | 0 |

The strict human-refined owner direction is therefore `+6`, close to the
official `+7`. The review supports a real but small physical-owner shift; it
does not convert reviewer judgment into ground truth or classify every
unmatched prediction as hallucination.

## Integrated Mechanism Judgment

The evidence rules out three overly simple explanations:

- **Pure stopping-gate explanation:** rejected on this selected panel because
  uncovered rows improve relative to covered rows after the opener is fixed.
- **Pure normalization artifact:** rejected because every score normalization
  and length stratum preserves the ordering and owner gains exist at the
  256-token cutoff.
- **Reliable owner-selection mechanism:** not supported because the candidate
  panel is selected and incomplete, actual terminal choices do not flip, and
  released owner realization does not improve.

The best current account is that step 36 makes continuation somewhat more
likely and changes conditional row ranking in a direction that can favor an
uncovered owner, while preserving much better stopping and prediction
efficiency than the full-row arms. The full-row objectives contain learnable
owner signal, but their much larger continuation shift dominates free rollout
and destroys efficiency. Row realization remains a separate unsolved stage.

## Recommendation for the Next Training Discussion

A mixed 256-image cohort is scientifically justified; requiring every image to
satisfy the old strict composite predicate is not. Images or exact prefix
events with verified uncovered-versus-covered comparisons can carry the
conditional owner-selection signal. Other images can carry Source
preservation, ordinary row realization, verified stopping controls, and
matched exposure. These roles must remain explicit in the data ledger and in
claims; background images are not declared equivalent set-expansion signal.

If the user authorizes training after discussion, the next compact matched
screen should separate four ordinary loss contributions rather than repeat the
full-row objective unchanged:

1. a modest binary preference between the row opener and stopping at verified
   continuation or stop states;
2. a pairwise preference, after the row opener is fixed, for a verified
   uncovered-owner row over a plausible already-covered-owner row;
3. token-level cross-entropy for realizing a selected valid row, reported
   separately for description/schema and coordinates; and
4. Source-policy preservation on ordinary and control prefixes.

The matched arms should include gate-only, conditional-owner-only, combined
with Source preservation, and an equal-exposure ordinary-row control. Learning
rate, update count, token exposure, and event population must be recorded, but
compute efficiency is not a precondition for testing each paradigm. The first
promotion gate should use fixed 256- and 512-token free rollouts and require:

- positive gained-minus-lost trusted-owner direction;
- no new length stops;
- retained Source owners reported alongside gains;
- owner gain per prediction and strict duplicate counts that do not collapse;
- no invalid-row or common-owner geometry regression; and
- final confirmation with the original one-prompt, one-completion clean greedy
  policy.

Owner exchange is reported, not automatically converted into negative
supervision. The next screen should not add a state carrier, change the prompt,
use one-row-at-a-time inference, or promote a final architecture.

## Claim Boundary

Supported:

- a small positive held-out checkpoint direction survives robust evaluation
  and strict human-refined owner review;
- forcing continuation can recover a missed true-positive owner in at least
  one current held-out terminal case;
- step 36 shifts both continuation and conditional relative row scores on a
  seven-case selected Source-prefix panel; and
- full-row arms primarily fail through excessive continuation and output
  expansion, not score normalization alone.

Not supported:

- a prevalence or probability estimate for forced-continuation recovery;
- an explicit covered-set state;
- reliable row realization under transition step 36;
- objective-recipe causality from one checkpoint per treatment;
- a generalization, scale, or architecture promotion claim; or
- a production change to the original prompt or decoding contract.

## Evidence and Receipts

All durable artifacts are under:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-25-existing-checkpoint-transition-mechanism-decomposition/
```

Decision-bearing subtrees are `robust-evaluation-v2`,
`heldout-owner-churn-review-v1/audit-summary-v1`,
`candidate-scoring-common-v1`, `candidate-score-full-v1`,
`fullrow-free-budget-v1`, and `fixed-prefix-panel-v1`.

Key artifact hashes:

- held-out robust summary: `ee31f0d3bb63d60f0bfa3fd3b7ca5f91d5c7df81b60aea022515f3b827d85cbd`;
- owner-churn audit summary: `407dee09cb833c3f580fabeb36d0c1c5ab2c3cd6ef81974cdb80629f62c68467`;
- common candidate-score reduction: `7cfd40ae275b40f09b891f56d7b40d7c4c1ccf195a51b1b61d463b223bf349f1`;
- fixed-prefix manifest: `07b760737652d75fa6926f6f76181d9a698178a0c21d799ba17afe155ea14226`;
- fixed-prefix summary: `42f0764caa826ae07f79400a445f4ce76d1c91e8b2b7dc09fe2f8ffc62b884f8`.

Independent audits found no correctness issue at any reviewed severity in
either the 85-case churn reducer or the fixed-prefix panel. The churn reducer
replay is byte-identical and its focused audit tests pass 11 of 11. The fixed-
prefix audit independently verifies every prefix, all 22 candidate rows, both score
receipts, all 16 releases, checkpoint/config identities, and the persisted
reducer output; its focused replay tests pass 7 of 7. The remaining provenance
limitation is that the transition-step-36 receipts bind its resolved adapter
path and loaded-state evidence but do not embed a tensor-byte 256-bit Secure
Hash Algorithm digest. No live identity inconsistency was found.

The unit stops here for user discussion before any optimizer update or new
training launch.

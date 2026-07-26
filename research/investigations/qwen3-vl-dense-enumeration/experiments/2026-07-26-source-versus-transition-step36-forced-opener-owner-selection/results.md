---
title: Source versus Transition Step 36 Forced-Opener Owner Selection Results
description: Paired 200-boundary checkpoint result separating natural continuation from conditional one-row owner selection.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-26-source-versus-transition-step36-forced-opener-owner-selection
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: executed_and_verified
updated: 2026-07-26
---

# Source versus Transition Step 36 Forced-Opener Owner Selection Results

## Decision

Recent transition step 36 did **not** create the Source checkpoint's substantial
latent one-row recovery after the continue decision is fixed. On the same 200
Source natural-terminal boundaries, forced Source and forced transition each
recover a strict uncovered owner on `45/200` boundaries. Transition gains two
strict owners, loses two, retains 43, and has strict owner net zero.

Transition step 36 does materially expose the latent behavior during natural
decoding: transition native release produces 16 strict uncovered owners versus
Source native release's one and terminates immediately on 155 rather than 198
boundaries. That is evidence for a learned continuation-gate change. It is not
evidence for aggregate improvement in owner selection conditional on already
continuing.

The strongest alternative in the frozen unit is therefore retained. No prompt,
inference policy, training objective, or architecture is promoted.

## Executed Contract

The execution used the frozen 200-boundary panel, literal Source-produced
assistant histories, and the canonical new-row opener token `151646`. Source
receipts came from the prior current-runtime production execution. Transition
step 36 was executed twice per boundary: native one-row release and forced-opener
one-row release.

Eight boundary-ID shards ran on GPUs 0 through 7 and produced exactly 200 unique
results. Runtime receipts prove the declared step-36 adapter and special-token
embedding identities, full-model 32-bit floating point, scaled dot-product
attention, physical batch size one, greedy decoding, 64-token one-row horizon,
and shared base model, tokenizer, processor, prompt, image frontend, parser, and
owner matcher. The only decision-bearing model differences were the transition
adapter and embedding delta frozen in the unit.

Two representative smoke cases passed attribution and a repeated strict-recovery
smoke produced identical native and forced token sequences.

## Primary Paired Result

| Outcome | Source forced | Transition step 36 forced |
| --- | ---: | ---: |
| Strict uncovered owner | 45 | 45 |
| Covered-owner repeat | 19 | 17 |
| Valid unmatched or ambiguous row | 134 | 137 |
| Invalid or incomplete row | 2 | 1 |

The paired owner transition is:

| Source has strict uncovered owner | Transition has strict uncovered owner | Boundaries |
| --- | --- | ---: |
| no | no | 153 |
| no | yes | 2 |
| yes | no | 2 |
| yes | yes | 43 |

This yields:

- checkpoint owner gains: `2`;
- checkpoint owner losses: `2`;
- retained strict owners: `43`;
- owner exchanges: `0`;
- strict owner net: `0`;
- exact two-sided paired sign-test `p = 1.0`; and
- deterministic 5,000-replicate bootstrap interval for mean owner net:
  `[-0.02, +0.02]` owners per boundary.

The per-boundary median, 10-percent trimmed mean, and aggregate mean are all
zero. Only four boundaries have nonzero owner net. No prefix-depth,
scene-density, remaining-owner-count, or earlier diagnostic-margin stratum
shows a stable targeted advantage; the small stratum deviations cancel in the
aggregate.

## Natural Continuation versus Conditional Selection

| Outcome | Source native | Transition step 36 native |
| --- | ---: | ---: |
| Immediate terminal | 198 | 155 |
| Strict uncovered owner | 1 | 16 |
| Covered-owner repeat | 1 | 2 |
| Valid unmatched or ambiguous row | 0 | 27 |

The transition checkpoint therefore converts many Source terminal states into
actual rows and exposes 15 additional strict owners natively. But once the row
opener is supplied to both checkpoints, the strict conditional capacity remains
`45` versus `45`. The clean interpretation is:

```text
recent transition training
  -> substantially changes whether another row starts
  -> does not improve aggregate strict owner selection after that start is fixed
```

The forced raw row token sequence is exactly equal on only `62/200`
boundaries, so this null is not caused by identical decoding everywhere.
Transition perturbs many descriptions or coordinates while leaving strict
owner yield unchanged.

## Entity-First and Geometry-Second Review

All four strict-discordant cases were inspected in the standard paired
ground-truth-versus-prediction renderer.

| Image | Strict change | Entity judgment | Geometry judgment |
| --- | --- | --- | --- |
| `154435` | transition gain | same intended carrot category; transition produces a valid new carrot owner | Source box is degenerate; transition IoU is `0.826` |
| `301714` | transition gain | genuine switch from a covered carrot to a different uncovered carrot | transition IoU is `0.715` |
| `497558` | transition loss | both arms select the same visible potted plant | IoU degrades from `0.560` to `0.456` after a small left-edge shift |
| `534112` | transition loss | both arms select the same visible baseball bat | IoU degrades from `0.635` to `0.491` after a small top-edge shift |

Thus the reproducible strict result remains net zero, while the arm-aware
entity-selection interpretation of the four discordant cases is two genuine
new-owner choices and zero entity-choice losses, paid for by two geometry
degradations. This is useful mechanism evidence, not a replacement metric: it
was not a blinded full-200 human review and does not adjudicate every changed
unmatched row.

The visual packet also reuses the prior fixed-seed random 30 Source-unmatched
sample. Within that sample, parsed predictions change in 21 cases, pixel boxes
change in 20, but descriptions change in only two. A targeted inspection of
the eight previously identified likely raw-COCO omissions plus the largest or
most consequential additional changes finds no broad transition-created
recovery pattern. Most selected entities remain the same with small coordinate
movement; one bag changes only from `backpack` to `handbag`, one prior
`cell phone` becomes a partial `person`, and image `57071` develops a conspicuous
near-full-image `car` box. The earlier Source-side estimate of visually grounded
unmatched rows therefore cannot be attributed to transition step 36.

## Claim Boundary

Supported:

- premature stopping is a real Source bottleneck;
- recent transition step 36 substantially changes the natural continue/stop
  gate on this panel;
- Source already contains the aggregate latent one-row conditional recovery
  exposed by the forced opener;
- transition causes broad row-level perturbations but no aggregate strict
  conditional-owner gain; and
- its four strict owner changes decompose into two entity-selection gains and
  two geometry losses.

Not supported:

- that transition step 36 created the roughly 40-percent human-refined latent
  recovery suggested by the earlier Source unmatched audit;
- that forcing continuation is a safe free-decoding policy;
- that one-row gains survive a longer trajectory or improve the final unique
  owner set;
- that transition improves every geometry or owner-selection case; or
- that this comparison selects a new architecture or training objective.

## Artifacts

Immutable artifact root:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-26-source-versus-transition-step36-forced-opener-owner-selection/`

Decision-bearing artifacts:

- `comparison-reduction-v1/summary.json`, SHA-256
  `9789d5ca1d1c8c17d0b714bf6eeef6074120910c87d1273c33b6fd0826fec2e9`;
- `comparison-reduction-v1/cases.jsonl`, SHA-256
  `ce87648dfc697ce4e616412bc11cec06a10c10fe0ffd5183a0c55c8b30ac49cf`;
- `comparison-reduction-v1/review-cases.jsonl`, SHA-256
  `de4547293c9a8550aa1b02ec5f61afa260b5ece8e482382dcc1f78db991788e1`;
- `transition-reduction-v1/summary.json`, SHA-256
  `e12d186c7b8007be878ac064fc1d03f661575f2606c9df12cbc12a56417fe3cc`;
- `visual-review-v1/selection.json`, SHA-256
  `960c75b31d430f7cdfed724cacd730345d94046eb1867dfb0e66a97112db58bb`;
- `visual-review-v1/rendered/manifest.json`, SHA-256
  `089f92a2dba482cbeb3e053cbaecd80731705c1e82dd41803ced9b83a703c36c`;
  and
- 34 paired rendered PNGs for the random-30 and four strict-discordant cases.

## Stop and Next Discussion

The unit stop condition is met. Do not automatically launch another checkpoint,
longer forced trajectory, training run, prompt change, or architecture branch.
The next discussion should decide whether the most valuable successor is:

1. a geometry-preserving continuation treatment, because step 36 already moves
   the gate but trades some localization quality; or
2. a longer-trajectory value test that asks whether the 45 latent one-row
   Source recoveries can produce net final-set gain without repetition and
   downstream loss.

That choice changes research meaning and remains with the user.

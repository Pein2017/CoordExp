# Physical-Owner Duplication: Interim Results

## Evidence Scope

This note records the completed training-free causal probe, validated
one-burst StateBank assembly, one-update optimizer smoke, and ordinary clean
greedy replay of the resulting checkpoints. The trained replay is an in-sample
treatment-efficacy result, not a generalization result.

- Source checkpoint: geometry-sorted pure cross-entropy Qwen3-VL 2B DoRA
  adapter at step 4,887.
- Decode repetition penalty: `1.0`.
- Reviewed image: COCO validation image `455649`.
- Reviewed duplicate owner: physical bottle owner `455649:89490`.
- Reviewed recovery owner: distinct bottle owner `455649:81761`.
- Corrected prefix-counterfactual receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-physical-owner-duplication-causality-and-training-treatment/prefix-counterfactual-455649-v3/receipt.json`
- Validated StateBanks:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-physical-owner-duplication-causality-and-training-treatment/smoke-state-banks-455649-v3/state-banks/`
- One-update training runs:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-physical-owner-duplication-causality-and-training-treatment/smoke-training-v1/runs/`
- Reloaded-checkpoint clean greedy inference:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-physical-owner-duplication-causality-and-training-treatment/smoke-checkpoint-inference-v2/`

Official ground-truth mismatch was not used as evidence of hallucination.
Training roles were assigned only after crop-enlarged physical-owner review.

## Reviewed Trajectory

The Source greedy trajectory contains:

1. a usable first occurrence of bottle owner `455649:89490`;
2. a distinct bottle owner `455649:85960`;
3. a near-exact duplicate of `455649:89490`;
4. a distinct, geometrically trusted bottle owner `455649:81761`.

The duplicate is not a useful localization correction: its box has 0.957
intersection over union with the first prediction and does not improve the
owner localization. The recovery row has 0.930 intersection over union with
its distinct reviewed owner.

## Training-Free Prefix Counterfactual

At the exact self prefix immediately before the duplicate:

- at the first differing coordinate token, the recovery-row minus
  duplicate-row log probability is `-0.25` natural-log units;
- the coordinate-field mean favors the recovery row by `+0.6154`;
- the complete-row mean favors the recovery row by `+0.3077`;
- greedy continuation emits the duplicate before the new owner;
- sampled seed 11 also emits the duplicate before the new owner;
- sampled seeds 12 and 13 emit the new owner first and do not emit the
  duplicate in the bounded continuation.

This is direct evidence for a local-versus-complete-row mismatch. The distinct
physical owner and its complete bounding box are already supported by the
Source model, and the complete row is scored better on average, while the
myopic first coordinate decision still selects the duplicate under greedy
decoding.

Changing prefix history produced the following complete-row recovery-minus-
duplicate margins:

| Prefix condition | Complete-row margin | First differing coordinate margin |
|---|---:|---:|
| Exact self prefix | +0.3077 | -0.25 |
| Same covered owners in a different order | +0.1845 | not used as the primary claim |
| Earlier repeated-owner row removed | -0.3680 | -4.000 |
| Length-matched covered-owner replacement | -0.3408 | -3.875 |
| Same description with corrupted coordinates | -0.3642 | -4.125 |

Removing or spatially corrupting the earlier owner record strongly restores
preference for the formerly duplicated owner. Merely preserving the word
`bottle`, the prefix length, or a generic covered row does not preserve the
suppression. The prefix therefore carries causally usable spatial owner
history, not only a description count or a generic continuation signal.

Changing the order of the same covered owners changes the strength but does
not erase the effect. This supports a qualified conclusion: Qwen3-VL has a
native, spatially specific commit signal, but it is sensitive to trajectory
order and does not guarantee that greedy decoding chooses the best complete
uncovered-object row.

## What This Supports

The strongest current treatment target is complete-row decision calibration:
under an actual self-rollout prefix, rank a verified uncovered-owner row above
a confirmed covered-owner duplicate row while preserving the native Source
route. This is more directly supported than suppressing end-of-sequence,
penalizing one token, or assuming the model has no commit state.

The result also supports retaining both treatment families:

- local duplicate rejection and recovery changes the decision at the observed
  self-prefix branch;
- duplicate-cleaned imitation teaches continued valid behavior after deleting
  only confirmed complete duplicate rows.

The cleaned trajectory remains explicitly counterfactual. It must not be
described as a naturally sampled model trajectory.

## One-Update Optimizer Smoke

Five one-update runs completed with finite nonzero gradients, an applied
optimizer step, and reloadable adapter plus special-token-embedding
checkpoints:

| Arm | Admitted event families | Total loss | Gradient norm | Pairwise margin before -> after |
|---|---|---:|---:|---:|
| Source-preservation control | 1 Source row | 1.3789 | 11.9501 | not applicable |
| Recovery-positive imitation | 1 Source + 1 recovery row | 2.4018 | 18.6633 | not applicable |
| Local duplicate rejection and recovery | 1 Source + 1 local pair | 2.4521 | 21.2963 | 0.3452 -> 0.8472 |
| Duplicate-cleaned imitation | 1 Source + 2 cleaned rows | 2.3528 | 15.2092 | not applicable |
| Combined treatment | 1 Source + 1 local pair + 2 cleaned rows | 2.4024 | 16.6611 | 0.3452 -> 0.7125 |

The local and combined arms moved the explicit uncovered-owner versus
covered-owner complete-row margin in the intended direction. The executed
profiles also retained the declared Source-preservation dose. The corrected
reducer preserves one total credit unit when a burst contains one or several
duplicate rows, while the combined arm preserves its declared local-versus-
cleaned allocation.

## In-Sample Clean Greedy Replay

All five step-1 checkpoints were reloaded through the ordinary Hugging Face
inference path with repetition penalty `1.0`. Every run completed without
truncation, parser failure, malformed output, or dropped prediction.

The Source-preservation control emitted ten rows. Its third row repeated the
leftmost bottle owner already emitted by its first row, then it continued to
the remaining bottles, sandwiches, and books.

Each of the other four arms emitted nine rows and did not repeat that physical
bottle owner. They retained five distinct bottle predictions followed by the
same two sandwiches and two books. This establishes that every treatment
family can alter ordinary greedy behavior after one update on the treated
trajectory. It also shows why a Source-only trained control is necessary: the
duplicate survived a matched ordinary update that did not contain the
treatment signal.

The recovery-positive arm also removed this trained duplicate, so this single
case does not identify the pairwise penalty as uniquely necessary. A broader,
never-trained comparison is required to distinguish general positive recovery
supervision, direct duplicate rejection, cleaned-trajectory imitation, and the
combined treatment.

## What This Does Not Establish

- One image does not establish population-level treatment efficacy.
- The probe does not show that every duplication burst has the same cause.
- Geometry refinements and owner-ambiguous dense clusters remain excluded.
- The corrected receipt now contains finite numeric description, coordinate,
  complete-row, closure, and terminal scores, but description is identical for
  the same-category bottle pair and therefore is not the discriminating field
  in this case.
- The optimizer and clean-replay result is only one trained image and one
  update. It does not establish never-trained transfer, population duplicate
  reduction, or a favorable precision-recall tradeoff.
- The cleaned-row training loss was finite, but a separate before-versus-after
  teacher-forced cleaned-row loss replay was not run; ordinary clean greedy
  behavior is the stronger completed behavioral check.

## Current Execution Status

Five immutable, load-validated smoke StateBanks were materialized:

- Source preservation only: 1 record;
- recovery positive only: 2 records;
- local duplicate rejection and recovery: 2 records;
- duplicate-cleaned imitation only: 3 records;
- combined local and cleaned treatment: 4 records.

An independent audit correctly held the first optimizer attempt because the
initial trainer profiles filtered out Source-preservation rows, an extra
reducer denominator changed the declared one-unit burst credit, and receipt
metadata overwrote a numeric description score. All three findings were fixed
before the completed optimizer smoke. The focused verification suite passed
190 tests, the corrected causal receipt passed strict finite-score validation,
and the five reloaded checkpoints completed ordinary clean inference.

The next decision-grade step is a matched multi-image screen with a strict
physical-owner ledger. Official unmatched predictions remain neutral unless
crop-assisted review establishes a semantic error or entity hallucination.
Confirmed false negatives, confirmed duplicate owners, and trusted valid
recovery suffixes are the actionable evidence. Never-trained images must be
held out so that the next result measures transfer rather than memorization.

## Multi-Trajectory Review Queue

A fresh deterministic join covered 256 images, one greedy trajectory per
image, and sixteen sampled trajectories per image. The final parser and
provenance checks retained 4,352 exact trajectories. Partial suffixes,
non-positive-area boxes, malformed rows, and official-unmatched rows were
recorded as neutral evidence rather than silently promoted to training labels.

The strict prediction-overlap queue contains 37 candidate bursts from 17
images. The expanded queue contains 90 candidate bursts from 32 images, of
which 62 later return to a trusted, previously uncovered annotated owner.
Eight crop-enlarged image reviews found seven clear examples of the intended
physical-owner pattern; one dense book scene was judged owner-ambiguous and is
excluded from primary evidence.

The first training screen uses 24 one-row bursts from 14 duplicate-mechanism
images. Each
selected trajectory changes directly from one confirmed repeated-owner row to
one trusted uncovered-owner row. This restriction avoids teaching across an
unresolved intermediate row and lets the local and cleaned treatments use the
same observed recovery. Every recovery row has annotation intersection over
union at least 0.8. Only two of the nine otherwise eligible image `65530`
trajectories are retained, and one extra image `305573` trajectory is removed;
the final reviewed set contains exactly 24 cleaned rows, one per admitted
burst. The materialized arm sizes are 24 Source-only records, 48
Source-plus-recovery records, 48 Source-plus-local-rejection records, 48
Source-plus-cleaned-imitation records, and 72 combined records. The Source
family touches 11 images, the duplicate-mechanism families touch 14 images,
and their union touches 16 images. This is a treatment screen, not a population
estimate; broader multi-row bursts are reserved for expansion only if the
first screen is promising.

## Completed Multi-Image Treatment Screen

Five eight-graphics-processing-unit runs completed from the same Source
checkpoint with a learning rate of `1e-5` and effective batch size 8. The
initial record counts produced 3 Source-only updates, 6 updates for each
single treatment, and 9 updates for the combined treatment. Because those
update counts were not matched, two additional Source-only controls repeated
the same 24 Source events to exactly 6 and 9 updates. These controls isolate
optimizer-update count, but they do not match event composition or Source
exposure: the controls contain only repeated Source rows, while each treatment
replaces part of that exposure with treatment rows. The result therefore
supports the complete training recipe, not a claim that cleaned-trajectory
semantics have been isolated as the sole cause.

The local duplicate-rejection profile failed at step zero in two preliminary
runs because an optimizer window could contain no eligible positive event.
The canonical third run used deterministic Source-plus-local family
stratification and completed all 6 updates. The same fail-fast requirement now
applies to every multi-family profile. The failed attempts are runtime
evidence, not treatment results.

### Train-256 Clean Greedy Replay

The following table compares each treatment with the Source-only control that
received the same number of optimizer updates. It is an equal-update control,
not a fully dose- and composition-matched control. `Owner change` is the change in
unique category-and-box matched annotated physical owners. `Strict duplicate
change` is a geometry-derived review candidate count, not a substitute for
human physical-owner review.

| Treatment minus equal-update Source control | Prediction rows | Owner change | Strict duplicate change | Mean matched intersection-over-union change |
|---|---:|---:|---:|---:|
| Recovery-positive imitation, 6 updates | -450 | -16 | -65 | +0.00214 |
| Local duplicate rejection and recovery, 6 updates | -28 | -21 | -56 | +0.00273 |
| Duplicate-cleaned imitation, 6 updates | -281 | **+4** | **-51** | +0.00156 |
| Combined local and cleaned treatment, 9 updates | -662 | -22 | +39 | +0.00535 |

On the 240 images that contributed no training event, duplicate-cleaned
imitation remained `+3` owners and `-48` strict duplicate candidates relative
to the equal-update Source control. This is a transfer result rather than
memorization of the 14 duplicate-mechanism images. Relative to the completely
frozen Source checkpoint, however, it remained 15 owners lower on train-256;
the treatment is therefore promising rather than solved.

The net train-256 owner gain hides substantial exchange: 81 owners appear only
in the equal-update Source control and 85 only in cleaned imitation. The net
gain is therefore small relative to trajectory reshuffling, and one checkpoint
per arm provides no run-to-run uncertainty estimate.

### Human-Refined 12-Image Replay

The 12-image validation panel has manually refined, nearly exhaustive COCO-80
boxes and is fully disjoint from the treatment events. Against the equal-update
6-update Source control, duplicate-cleaned imitation changed:

- prediction rows: `245 -> 218`;
- unique matched owners: `139 -> 142` (`+3`);
- strict duplicate candidates: `11 -> 2` (`-9`);
- mean matched intersection over union: `+0.00284`.

The human-refined panel likewise contains 8 control-only and 11
treatment-only owners. Its `+3` net gain is independent directional support,
not evidence that owner exchange has been eliminated.

Against the completely frozen Source checkpoint, duplicate-cleaned imitation
was `+4` owners and `-3` strict duplicate candidates. It therefore did not
merely obtain a better duplicate count by stopping earlier: on this higher
quality panel it emitted 98 fewer rows while matching four more physical
owners. Visual review also found a counterexample, image `10707`, where the
treatment lost one matched owner and produced a new repeated `remote` row.
The effect is real but not uniformly safe.

### Treatment Decision

Duplicate-cleaned imitation is the only complete training recipe that improves
unique-owner coverage and duplicate behavior against its equal-update
Source-only control on both the train-256 aggregate and the human-refined
panel. Recovery-positive imitation
and local duplicate rejection suppress repeats but usually pay for them with
owner recall. The combined profile is rejected: it loses owners and increases
strict duplicates against its 9-update control on train-256.

Ordinary predictions unmatched to official ground truth remain neutral during
this decision. They are not called hallucinations unless crop-assisted review
shows that no physical entity exists or that the COCO-80 category is wrong.
Confirmed false negatives and physical-owner duplicates remain the primary
negative evidence.

Primary analysis artifacts:

- equal-update train-256 comparisons:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-physical-owner-duplication-causality-and-training-treatment/final-checkpoint-inference-v2-dose-controls/owner-coverage-analysis-v1/train-256/`;
- equal-update human-refined comparisons:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-physical-owner-duplication-causality-and-training-treatment/final-checkpoint-inference-v2-dose-controls/owner-coverage-analysis-v1/human-refined-12/`;
- Source-versus-cleaned visual review:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-physical-owner-duplication-causality-and-training-treatment/final-checkpoint-inference-v2-batch4/visual-review/source-vs-cleaned-human-refined-12/`.

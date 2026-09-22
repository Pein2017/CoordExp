---
title: SFT256 improves training fit but not development owner recall
description: Complete five-point natural learning curve from original Source with full language DoRA and frozen embedding/readout.
type: investigation
role: research-results
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-05-sft256-dev128-baseline
status: complete
evidence_status: complete
updated: 2026-09-05
---

# Result: train gains do not transfer at the registered milestones

**Scientific disposition:** `NO_DEV_IOU50_PROMOTION_AT_REGISTERED_MILESTONES`.
All four trained checkpoints improve annotated-owner recall on train256, but
none improves the primary development IoU50 owner count over original Source.
Do not promote the final checkpoint or automatically use it as an improved RL
anchor. The registered 256-step course is closed; no schedule extension or
seed sweep was run.

**Technical disposition:** complete. The native eight-rank run made 256 finite,
applied updates; every saved checkpoint contains all 588 updated DoRA tensors
and a selected-token embedding delta byte-identical to Source. Ten independent
native inference arms completed 1,920 rows, including all 80 worker receipts.
The lead verified the 90 merged/worker model-identity receipts and replayed all
24 Source-to-milestone comparisons through the unchanged matcher.

## Full registered curve

Counts are unique annotated owners at IoU50 / IoU60 / IoU80, not detections
summed across thresholds. Denominators are 1,955 train owners and 891 development
owners. Forward CE is the native segment-balanced development diagnostic.

| Checkpoint | Train256 owners, 50 / 60 / 80 | Dev128 owners, 50 / 60 / 80 | Dev forward CE |
|---|---:|---:|---:|
| Source | 1259 / 1190 / 908 | 614 / 585 / 451 | not measured in this course |
| step 16 | 1275 / 1205 / 920 | 605 / 579 / 457 | 1.397071 |
| step 64 | 1330 / 1255 / 987 | 609 / 578 / 444 | 1.435613 |
| step 128 | 1387 / 1307 / 1036 | 605 / 571 / 423 | 1.502298 |
| step 256 | 1394 / 1317 / 1046 | 596 / 563 / 418 | 1.547337 |

At step 256, train IoU50 increases by 135 owners (64.40% to 71.30%), whereas
development loses 18 (68.91% to 66.89%; 23 gains versus 41 losses). Development
IoU60/80 changes are -22 / -33. The early step-16 IoU80 gain of six is real but
does not reverse the negative primary IoU50 result. No best-only reporting.

All four development density bands lose IoU50 owners at step 256:
1-3 / 4-7 / 8-15 / 16+ annotated objects have net changes -2 / -5 / -4 / -7.
The 16+ band has only ten images; its temporary step-64 gain of four is not
evidence of a general dense-scene improvement.

## Output behavior and limits of interpretation

| Checkpoint | Train length caps | Train dropped predictions | Dev dropped predictions |
|---|---:|---:|---:|
| Source | 4 | 794 | 58 |
| step 16 | 3 | 481 | 5 |
| step 64 | 2 | 341 | 6 |
| step 128 | 1 | 11 | 18 |
| step 256 | 1 | 33 | 17 |

Every development arm ends with native `im_end` on all 128 images, with no
length caps. Native parser, scoring and image-validation failures are zero
across the matrix. Matcher-invalid predictions are one on train at steps 64
and 128 and zero elsewhere. Dropped-prediction diagnostics are retained; zero
parser failures do not mean zero malformed or dropped output debt.

From Source to step 256, development predictions fall from 1,103 to 908, strict
physical-owner duplicate candidates from nine to two, and IoU50-unmatched
predictions from 489 to 312. Those unmatched rows are **unknown**, not verified
false positives; their reduction cannot establish better precision. Fewer
outputs and less diagnostic debt coexist with lost annotated owners.

Localization also changes: among the 573 development owners matched by both
Source and step 256 at IoU50, mean IoU falls by 0.010653. The analogous selected
common-owner train mean rises by 0.025552. These are conditional common-owner
summaries, not averages over all GT. A pure output-count/stop explanation is
therefore insufficient to describe the whole observed shift.

The combined pattern supports small-panel overfitting and negative transfer at
this full-DoRA dose. It does **not** identify its unique cause. Partial-label
full-transcript/EOS supervision, parameter interference, and natural prefix or
grounding changes remain competing mechanisms. There was no matched EOS-only,
RLOO, GSPO, or QP intervention in this unit. Historical censored-CE results have
different objectives/doses and are not a controlled replacement for that test.

This development panel was previously used and is not a virgin final test.
The native `benchmark_eligible=false` on 128 rows is expected; the 200-row size
threshold is unchanged. The train256 flag being true likewise does not make
training-set performance a held-out benchmark. Claims concern these fixed
annotated-owner panels, not complete-scene recall or precision.

## Runtime recovery and measured cost

Native training plus four forward evaluations took 1,522.54 seconds (25.4 min),
excluding launcher overhead. The ten cold commands took 5,064 seconds (84.4 min)
in aggregate, measured from per-arm controller UTC start/completion markers;
this is not the optimistic maximum-rank decode estimate. Qualification and
agent/CPU preparation are excluded from those two durations.

The batch's terminal exit was initially 1 only because the final direct-script
CPU reducer imported an unrelated `/data/verl/scripts` package before the repo
root was on `sys.path`. All ten inference arms had already exited successfully.
A fresh narrow Luna/medium repair added early direct-entry root binding and a
subprocess regression; lead checks passed nine tests and the exact direct
reducer command then completed. No model run or inference row was repeated,
and the failure log remains preserved.

## Evidence and continuation

- Frozen scientific contract: [unit](unit.md).
- Overall lead verdict and hashes: [lead acceptance (preserved from archive ref `archive/research-restructure-20260909/dora-prox-linear-n2` at `ba801de51`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-sft256-dev128-baseline/lead-acceptance-v1.json).
- Training counters and checkpoint payloads: [training receipt (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-sft256-dev128-baseline/training-lead-acceptance-v1.json).
- Ten-arm identity/debt/timing evidence: [native inference receipt (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-sft256-dev128-baseline/native-evaluation-lead-acceptance-v1.json).
- Complete machine-readable curves, density strata and 24 comparison receipts:
  [aggregate](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/natural-eval-v1/reduction-v1/aggregate.json).
- Aggregate SHA-256: `0f6c005b5b0926e0001a604778c5405ee7c6ef693fbdcbdb64b0d9c404801624`.
- Separate orchestration pilot: [scope-limited routing/cost audit (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-sft256-dev128-baseline/agent-routing-audit-v1-report.md).
  It includes earlier work in this task and lacks Astra price coverage; neither
  SFT-only total cost nor overall lead-cost reduction is established.

Next research should distinguish preservation/grounding interference from
partial-label stopping pressure before selecting a common anchor and reward
for continued CE versus refreshed RLOO. This is a proposed successor question,
not an executed causal result or a reward-policy authorization. In particular,
do not carry forward the historical unmatched-negative or clamped-EOS credit
rules merely because their implementation already exists. QP/hidden-state and
self-prefix credit remain retained possibilities, not promoted mechanisms.

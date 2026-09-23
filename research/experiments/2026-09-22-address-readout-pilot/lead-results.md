# Coarse address-readout pilot: accepted execution, no promotion

Date: 2026-09-22. Status: **lead-accepted and closed**, with the evidence limits
below. This is a negative result for the fixed intervention, not a rejection of
all coordinate-address mechanisms. No successor is authorized.

## Decision and evidence boundary

The correctly associated coarse address sidecar learned on its training data,
but both seeds worsened independent calibration and fresh native annotation
coverage. Do not scale this recipe or select a checkpoint/seed from its favorable
old failure cases. Better coordinate calibration was a necessary intermediate
prediction of the proposed remedy; it was not achieved. Therefore the experiment
does not establish what would happen to bursts if coordinate readout were
actually improved. The broader ruler hypothesis remains unidentified.

Lane B remains source-inapplicable HOLD with zero cells; H_B is untested.
The optional recurrence-prefix diagnostic was not executed. The source is the
mature tied step-2444; no untied comparison or backbone update occurred.

[Lead acceptance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-address-readout-pilot/production/lead-acceptance-v1.json) binds the candidate, independent replay,
checks, costs and per-image comparisons. The immutable [worker handoff](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-address-readout-pilot/production/candidate-v1/handoff.json)
and [model-evidence manifest](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-address-readout-pilot/production/candidate-v1/manifest.json) retain the original candidate status.
[Unit](unit.md), [ruling 01](lead-ruling-01.md) and [ruling 02](lead-ruling-02.md)
own the frozen scope and staged execution authority.

All four fresh fits reached update 256: 1024 updates and 230144 supervised
coordinate-token targets. Native evaluation completed 190/190 cells: all 38
five-condition blocks, split into 32 fresh and six development images. Calibration
completed 160/160 cells: all 32 blocks. No missing/partial Lane A blocks or
production retries. Original is evaluated once; each trained arm has two seeds.

## Calibration

All conditions score the same 808 positive teacher coordinate targets. Geometry
uses bin/1000. Free-box MAE is conditional on the displayed scored cases; the
missing first boxes remain visible in the full denominator, with parser/cap
outcomes retained. There is no successful-only whole-cohort accuracy claim.

| Condition | CE/token | Teacher coordinate MAE | Free-box MAE (scored cases) | Missing first box |
|---|---:|---:|---:|---:|
| original | 3.409525 | 0.015389 | 0.125070 (32) | 0/32 |
| aligned-1729 | 3.553782 | 0.020423 | 0.140363 (31) | 1/32 |
| permuted-1729 | 3.452764 | 0.015775 | 0.151558 (30) | 2/32 |
| aligned-2718 | 3.524034 | 0.021116 | 0.153695 (32) | 0/32 |
| permuted-2718 | 3.436619 | 0.019126 | 0.132081 (31) | 1/32 |

Online training CE declined for all four fits; the acceptance receipt binds the
curves and first/last epoch summaries. These online epoch means are not fixed
checkpoint evaluations. Final aligned gains are positive, while both permuted
gains are negative. The latter is a learned response to the frozen roll-1
control, not evidence that it is information-free.

## Natural enumeration

These are **class-agnostic annotation IoU>=0.5 proxies**, not newly adjudicated
physical owners. Coverage uses one-to-one matching; recurrence uses the retained
proxy assignment semantics. Unmatched predictions remain UNKNOWN. Invalid counts
are emitted box events, not counts of distinct affected objects or images.
The maximum run is a per-image maximum, not a cohort average.

Fresh 32 images:

| Condition | Covered | Gained/lost vs original | Revisits | Max contiguous run | Invalid geometry | Capped images |
|---|---:|---:|---:|---:|---:|---:|
| original | 175 | 0/0 | 11 | 6 | 9 | 0 |
| aligned-1729 | 138 | 7/44 | 199 | 195 | 665 | 4 |
| permuted-1729 | 156 | 9/28 | 268 | 256 | 359 | 2 |
| aligned-2718 | 154 | 12/33 | 12 | 6 | 268 | 2 |
| permuted-2718 | 164 | 11/22 | 352 | 339 | 624 | 3 |

Fresh per-image coverage improves/is unchanged/worsens in 3/18/11 images for
aligned-1729, and 5/17/10 for aligned-2718. This is not only a comparison of totals
dominated by long trajectories. The panel's frozen annotation-density strata are
2 images with one object and 10 each with 2-4, 5-9 and 10+ objects; no claim of
exhaustive physical annotation or pretraining/SFT holdout follows.

Six development images, kept separate:

| Condition | Covered | Gained/lost vs original | Revisits | Max contiguous run | Invalid geometry | Capped images |
|---|---:|---:|---:|---:|---:|---:|
| original | 26 | 0/0 | 28 | 12 | 210 | 3 |
| aligned-1729 | 42 | 23/7 | 6 | 2 | 271 | 1 |
| permuted-1729 | 29 | 10/7 | 26 | 9 | 328 | 2 |
| aligned-2718 | 32 | 8/2 | 20 | 10 | 566 | 2 |
| permuted-2718 | 23 | 3/6 | 15 | 5 | 731 | 3 |

The old failure pool contains favorable local routes, including aligned-1729's
16 gained/1 lost annotation proxies on donut417044. Its aggregate improvement
does not transfer to the fresh panel; this pilot does not select that seed or
image as evidence of a general repair.

## Interpretation and next decision

Observation: learned sidecar training loss falls while held-out teacher CE/error
and fresh coverage worsen. Inference: this recipe fails to generalize coordinate
corrections; the regression is already visible under supplied correct histories,
so errors introduced only during free autoregressive rollout cannot be its sole
explanation. Free rollout can still amplify those errors. None of this identifies
the original model's burst cause.

This result does not separate training-panel overfit from an inadequate coarse
address interface. The strongest remaining alternative in the original contract
is that improved calibration could still leave history-dependent instance
selection unresolved. This experiment did not reach that discriminating outcome.
A future proposal must establish a reliable out-of-sample geometric benefit
before treating its burst outcome as a test of that alternative; simply adding
dose, seeds or old escape examples is not released.

## Acceptance, limits and closure

The lead independently replayed the saved reducer byte-for-byte (SHA256
`dbdac13637fcb7d6974d24b8a66f9e68418d6afa68273843fb44a6c6493d1ab2`),
verified 572 binding entries / 513 unique binding identities with no mismatch,
and ran all 15 CPU tests. One existing tensor-to-scalar warning was nonblocking.
A bounded independent review checked all 350 cell identities/traces and disjoint
image sets, reproduced all fresh coverage via independent maximum-cardinality
matching, checked original calibration caches/alignment, and reproduced free-box
errors/denominators. No decision-changing scoring defect was found.

Two disclosed limits are accepted without changing the primary evidence: the
six development images lack a materialized density/class stratum field (no
post-outcome stratum inference is made); superseded intermediate reducer telemetry
has no separately captured exact intermediate source bytes. Final producer and
shared imports are captured, final reduction replays exactly, and raw model cells
are preserved. Neither limit rehabilitates the intermediate telemetry.

All 18 owned producers are terminal and their PIDs absent. Independently summed
costs include both failed qualification attempts: 91315 model forwards, 538 vision
forwards, 8270.141018 GPU-seconds (2.297261 allocated GPU-hours), and 4509.113175
seconds (75.151886 minutes) from the original wall start to final model terminal.
Both original ceilings were respected. Eight GPUs were used; the fixed image
assignment left a tail worker. No rerun is warranted merely to improve utilization.

Close this pilot. Preserve all evidence, qualifications, failures and controls.
No further model work, broader training, new physical review, or successor launch.

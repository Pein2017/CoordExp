# Selective owner learning: bounded IoU50 improvement after targeted preservation

Final scientific disposition: **lead-accepted bounded IoU50 development
candidate `positive7-support50-81`, with explicit localization, old-owner and
subpopulation debts**. Technical disposition: **all six completed candidates
and the final384 cold read verified**. The first five candidates are not
promoted; the dense48 technical-invalid attempt remains separate from its
valid retry. All7 trained targets recover, the3 newly capped cases return to
EOS without cap migration, and the full384 and outside327 IoU50 owner/F1
directions improve. No held-out or deployment promotion is claimed.
This investigation is closed under the [owning protocol](unit.md); use the
[handoff](handoff.md) for the current decision and reopening boundary.

## First candidate: soft-preservation-10

From original Source step2444, train the same588 language DoRA A/B/m tensors
for exactly23 updates. Two entry x1 labels receive CE; Source full-vocabulary
distributions at134/41 surrounding healthy-trajectory positions receive
mean-state forward KL with coefficient10, equally weighted by image. The
remaining supplied owner-row coordinates/boundary are excluded; EOS and the
shared header are included. This is soft reference preservation, not new
suffix truth, a hard replay constraint, or a proof of local parameter effects.

Initial KL is exactly0. Frozen tensor bytes remain unchanged; adapter movement
L2 is.42343 (pointCE23:.46283). Training takes124.75 seconds, one GPU load,
96 model forwards and106869000 reference-cache bytes. Final isolated margins
are+.295244 on image368 and−.063457 on image7116. The latter does not invalidate
the fixed23 run or prevent natural target recovery. Last logged training KL
is before update23, not terminal-checkpoint KL.

## Frozen natural screen

All18 outputs use empty forced prefixes, original full images/prompts,
FP32/SDPA, T0/top-p1/RP1 and cap3084. All finish naturally. Annotation-relative
matching uses the existing global category-consistent pixel-IoU consumer.

| Outcome | Train2 Source | Train2 pointCE23 | Train2 KL10 | Guard16 Source | Guard16 pointCE23 | Guard16 KL10 |
|---|---:|---:|---:|---:|---:|---:|
| TP50 |16|19|18|56|55|56|
| TP60 |16|16|18|54|52|55|
| TP80 |9|11|11|43|39|43|
| FP50 |2|34|2|146|155|135|
| FN50 |3|0|1|33|34|33|
| F1@50 |.864865|.527778|.923077|.384880|.367893|.400000|
| Valid predictions |18|53|20|202|210|191|
| Strict repeats |0|21|0|44|44|43|
| Parser drops |0|1|0|7|12|8|
| Complete tokens |167|491|185|2000|2122|1908|
| Caps |0|0|0|0|0|0|

Train2 gains two owners and loses none relative to Source at50/60/80. These
are the intended targets: person2022537 in image368 and boat181378 in image7116.
The person bbox bins[455,184,489,229] have IoU.6415; the boat
[290,460,371,562] has IoU.8946. Actual boat x1 is290, not trained291.
Both histories diverge before the training entrance (indices32 and6), so
recovery is not exact teacher-route replay. Root viewed both GT/prediction
images; the prior occlusion/extent qualifications still apply.

The boat's additional person1759775, recovered by defective pointCE23, is not
recovered by KL10. Thus KL10 loses one pointCE owner while removing its large
annotation-relative excess/repetition burden; it is not a superset of pointCE.

Guard16 owner changes versus Source are2 gained/2 lost at50,2/1 at60 and2/2
at80. At50: image59571 gains2094968; image388795 gains1989344 and loses1987783;
image561545 loses313455. The unchanged TP total does not mean unchanged owners.
Parser drops increase by one; the favorable aggregate tradeoff is qualified,
not a claim of uniform improvement. Unmatched predictions remain annotation-
relative and are not automatically called hallucinations.

## Interpretation and next action

Unlike pointCE23, this update realizes both intended owners with no new
training-image repeats, no prior Source-owner loss and improved joint F1.
It is a positive selective-feasibility observation. The comparison does not
isolate reference preservation from a changed effective update dose or prove
an owner-ledger/hidden-state mechanism. It also does not establish performance
on untouched images or identify the population prevalence of recoverable owners.

The sole candidate was locked before any dev112 generation. We evaluated all
remaining historical dev128 images, excluding exactly the selected guard16,
with no retuning or outcome-filtered selection. Train2, guard16 and dev112
must remain separately visible. The existing confirmation512 stays untouched.
The separate Source/pointCE boat cross is explanatory, not an acceptance gate.

## Locked dev112: owner gains coexist with a wider precision cost

The exact128-minus-guard16 complement was frozen before generation, with no
training-image overlap, substitutions or retuning. Four independent28-image
shards produced all112 unique natural outputs under the same checkpoint and
native policy; root freshly replayed the full merge/native-consumer verifier.

| Outcome | Source | KL10 |
|---|---:|---:|
| TP50 |558|576|
| TP60 |531|533|
| TP80 |408|412|
| FP50 |343|466|
| FN50 |244|226|
| F1@50 |.655314|.624729|
| F1@60 |.623605|.578091|
| F1@80 |.479154|.446855|
| Valid predictions |901|1042|
| Strict repeats |68|109|
| Parser drops |51|4|
| Complete tokens |9070|10019|
| EOS/caps |112/0|112/0|

Owner gains/losses versus Source are28/10 at50,18/16 at60 and18/14 at80.
Therefore the first screen did not establish broad joint improvement. It also
did not produce a total scientific null: new owners occur outside the two
training images, at a substantial annotation-relative excess/repetition cost.
Improved parsing changes what reaches the scorer and must remain visible;
FP is not a physical hallucination adjudication. The candidate is not promoted.

Four model loads generate10019 new tokens with880.85 allocated GPU-seconds;
the longest shard takes270.57 seconds. All exit0. Artifacts remain immutable.
The remaining-dev panel is now exposed to this investigation; future reads on
it are adaptive development evidence, not newly untouched validation.

The next decision is to distinguish concentrated repeat/parse effects from
widespread excess predictions and assess broader Source preservation support
from train256, disjoint from dev128. This does not authorize unchanged training
extension or automatic lambda sweeps; the next objective requires a new packet.

The completed [concentration diagnosis](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/dev112-diagnosis/README.md)
decomposes FP+123 into repeated-FP+41 and non-repeat-FP+82. FP rises on32/112
images; the top5 supply65.7% of gross-positive FP increases, not65.7% of net
increase. Excluding only the largest spoon-burst image still leaves F1−.01566;
this is descriptive, not a corrected evaluation. Four parser-drop-improved
images contribute only net FP+3. All10 lost IoU50 owners lack candidate direct
same-category support at.5; one gained owner already had Source direct support.
Three inspected overlays include genuinely ambiguous carrot-instance extents
and a dense crowd whose13-person GT is visibly incomplete, so the numerical
FP cost is not uniformly a physically false-object verdict. Root independently
replayed all224 native parses/scores and aggregate decompositions.

[Locked dev112 manifest](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-10/dev112/manifest.json)
and [native reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-10/dev112/reduction.json).

```bash
PYTHONPATH=/data/CoordExp/.worktrees/research-probes python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-10/dev112/run.py verify
```

## Second candidate: wide31 preserves local targets but not overall quality

Restart Source, retain the exact original two CE/10-KL terms, and add10 times
the equally image-weighted mean KL on31 additional Source train trajectories.
The initial salted-rank32 inventory explicitly excludes the sole capped
image417044 without backfill. All31 retained trajectories end naturally, have
zero parser drops and are disjoint from dev128. No additional positive owner
labels are introduced. This jointly changes support and regularization mass;
it is not a pure support-size causal ablation.

The exact23-step checkpoint is frozen before all130 natural outputs. Training
takes918.43 seconds,840 forwards, one load and a1064415240-byte static reference
cache; first-update33-trajectory smoke passes. Frozen bytes remain unchanged,
and cold terminal entry vectors match. Adapter movement L2 is.421904.

| Panel/outcome | Source | First KL10 | Wide31 |
|---|---:|---:|---:|
| Train2 TP50/FP50 |16/2|18/2|18/2|
| Train2 F1@50 |.864865|.923077|.923077|
| Train2 strict repeats |0|0|0|
| Guard16 TP50/FP50 |56/146|56/135|54/182|
| Guard16 F1@50 |.384880|.400000|.332308|
| Guard16 strict repeats |44|43|85|
| Dev112 TP50/FP50 |558/343|576/466|557/363|
| Dev112 F1@50 |.655314|.624729|.646922|
| Dev112 strict repeats |68|109|78|
| Dev128 descriptive TP50/FP50 |614/489|632/601|611/545|
| Dev128 descriptive F1@50 |.615848|.595104|.596971|
| Dev128 descriptive strict repeats |112|152|163|

Wide31 retains both intended targets at IoU50/60 and the boat at80, with no
training Source-owner loss. Outside train2, the new31 reference images do not
resolve the joint tradeoff. Combined dev128 has17 gains/20 losses at50; TP60
falls585 to578 and TP80 falls451 to441. F1@60 falls.586760 to.564729 and F1@80
.452357 to.430874. Parser drops58 to5 and tokens11070 to11107 remain visible.
All130 generations reach EOS without caps. Neither the unchanged local
success nor better dev112 FP relative to first KL10 justifies promotion when
guard deterioration and total owner loss are included.

Four independent cold loads, two entry parity reads and11292 generated tokens
take996.77 allocated GPU-seconds (maximum shard348.86 seconds). Root freshly
passes10 focused tests and both CPU verification CLIs, replaying exact130
native rows, complete shard/candidate identity and split reductions.

The remaining decision is whether stronger preservation, broader reference
coverage or a changed learning dose can retain useful gains; no next arm is
implicitly launched by these observations. Keep both failed broader candidates
and their positive local evidence. No blanket claim against scalar objectives,
KL preservation, or the model's ability to learn owners follows.

- [Wide31 training receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-wide31/training/receipt.json).
- [Wide31 independent130 reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-wide31/evaluation/reduction.json).

```bash
python -m probes.dora_owner_learning.selective_preservation_wide verify
python -m probes.dora_owner_learning.selective_preservation_wide_eval verify
```

## Dense48: coverage contrast and corrected numerical admission

The third scientific recipe keeps support-KL mass10 and23 steps but expands
the31 references by17 Source-train clean-EOS trajectories with at least20
valid predictions. Its48 references have5848 action states (maximum325) and
remain dev-disjoint. Eight data-parallel replicas execute the same globally
normalized objective; this does not change the model architecture or labels.

The initial invocation is **technical-invalid** after update1, with no quality
evaluation or saved candidate. Its output-only comparison to the differently
supported wide31 first update exceeds1e-5 (max1.08838e-4/3.71933e-5). Eight
failed terminals, launcher exit1,104 forwards and482.12 rank-seconds remain
preserved. Live cross-rank checks passed, but their values were not saved before
that gate; no durable hash evidence is invented for the failed attempt.

A bounded same-objective serial mechanical control shows the wide31 discrepancy
also occurs without DDP. Root independently verifies54 model forwards,89.54
seconds, exact initial Source vectors, all50 initial KL values0, and saved raw/
clipped gradients and Adam/adapter states. The same dense48 serial/DDP vectors
differ by at most5.14984e-5/3.43323e-5 while entry top1/ranks remain unchanged.
CPU fixtures also show FP32 exact-zero KL need not have an exactly zero
backward residual. These facts invalidate the original cross-population1e-5
output-equivalence assumption; they do not alone establish correct weighting.

Before any natural result, root registers one fresh retry with a stronger
direct objective check: full raw and clipped gradients must match the serial
control within1e-5 relative L2, computed in FP64. Same-objective entry vectors
must agree within1e-4 with identical top1/ranks. All eight ranks must agree
exactly on reduced gradients, adapter and Adam state. Evidence is persisted
before comparison failures. This replaces the invalid gate, not the objective.

The retry's integrated two-update slice passes: raw-gradient relative error
4.54387e-8 and clipped-gradient error4.67509e-8; all rank states agree exactly.
Step1 vector differences match the observed serial/DDP range; step2 exercises
nonzero KL on every rank and frozen bytes stay unchanged. The recipe completes
its fixed23 checkpoint. Training/evaluation are in separate retry1 roots;
the serial one-step adapter is mechanical-only and cannot enter natural eval.

Root freshly recomputes both gradient errors and all23 steps' exact rank-state
agreement,50 unique loss identities per step,46 entry-label occurrences and
1104 support-trajectory uses. An unused `support_KL_trajectories` telemetry
counter remained0; the retained per-step identities establish the true1104
count. The verification discloses this without changing the sealed receipt.

Training completes1248 forwards in302.92 maximum-rank seconds (2412.37 summed
rank-seconds;315.28 launcher seconds), with31.02GB maximum allocated CUDA and
adapter movement.422813. All8 rank terminals and launcher exit0 precede sealing.
The independent130 read uses12015 new tokens,1054.52 allocated GPU-seconds and
four complete cold processes; all outputs reach EOS without caps. Root freshly
passes15 focused tests and both verification CLIs, including all130 consumers.

| Outcome | Source | Wide31 | Dense48 |
|---|---:|---:|---:|
| Train2 TP50/FP50 |16/2|18/2|18/2|
| Train2 F1@50 |.864865|.923077|.923077|
| Guard16 TP50/FP50 |56/146|54/182|54/222|
| Guard16 F1@50 |.384880|.332308|.295890|
| Guard16 repeats |44|85|108|
| Dev112 TP50/FP50 |558/343|557/363|559/395|
| Dev112 F1@50 |.655314|.646922|.636674|
| Dev112 repeats |68|78|106|
| Dev128 descriptive TP50/FP50 |614/489|611/545|613/617|
| Dev128 descriptive F1@50 |.615848|.596971|.578029|
| Dev128 descriptive repeats |112|163|214|

Both targets remain recovered with the same boxes as wide31 and no new
training-image repeats. Combined dev128 gains18/loses19 owners at50 relative
to Source; TP60 is580 versus585, TP80 is450 versus451. F1@60 is.546912 versus
.586760; F1@80 is.424328 versus.452357. Parser drops58 to11 and complete tokens
11070 to11830 remain visible. The proposed coverage addition fails to resolve
the broader tradeoff; it is not promoted and is not evidence that longer
history coverage alone solves the problem.

The next frozen contrast keeps these48 references and23 steps but raises only
the additional KL coefficient10 to100. It tests weak preservation rather than
expanding reference count automatically; no parameter sweep is authorized.

- [Failed initial invocation](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48/training/failure_receipt.json).
- [Serial mechanical control](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/dense48-parity-serial/mechanical_receipt.json).
- [Retry direct-gradient parity](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48/training-retry1/step1-gradient-parity.json).
- [Retry two-update slice](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48/training-retry1/two-step-smoke.json).
- [Sealed dense48 checkpoint receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48/training-retry1/receipt.json).
- [Dense48 complete130 reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48/evaluation-retry1/reduction.json).

## Strong100: first joint development improvement

This fourth arm changes only dense48's additional support KL coefficient10
to100. Source start,48 reference images/5848 action states, original two
positive entry labels and local coefficient10, optimizer and fixed23 steps
remain unchanged. The accepted eight-rank DDP path is reused, with a fresh
analytic tenfold-support-gradient fixture and a current two-update real-entry
slice; there is no cross-coefficient first-step equality gate or new serial run.

All8 ranks and the launcher complete before sealing. Root independently
replays the training and natural consumer verification CLIs and11 focused
tests. Current50 initial KLs are exactly0; step2 exercises nonzero KL and all23
updates preserve exact cross-rank gradient/adapter/optimizer identity. Actual
support uses are1104 and positive-token uses46. Training takes1248 forwards,
8 loads,308.69 maximum rank seconds/2463.27 summed rank seconds and the same
3678125640 cached bytes; maximum CUDA allocation31.018GB. Adapter movement L2
is.411330. Final isolated margins are+.311623/-.060745, not acceptance gates.
Last pre-update23 mean local/support KL is.00350550/.00212465; these are not
terminal KL measurements and smaller KL alone does not establish selectivity.

| Frozen natural outcome | Source | Strong100 |
|---|---:|---:|
| Train2 TP50 / FP50 / F1 |16 /2 /.864865|18 /2 /.923077|
| Train2 strict repeats / parser drops |0 /0|0 /0|
| Guard16 TP50 / FP50 / F1 |56 /146 /.384880|54 /121 /.409091|
| Guard16 repeats / parser drops |44 /7|38 /22|
| Dev112 TP50 / FP50 / F1 |558 /343 /.655314|564 /322 /.668246|
| Dev112 repeats / parser drops |68 /51|58 /2|
| Dev128 descriptive TP50 / FP50 / F1 |614 /489 /.615848|618 /443 /.633197|
| Dev128 repeats / parser drops / tokens |112 /58 /11070|96 /24 /10358|

Dev128 at50 has21 gained/17 lost/597 retained owners, not universal
preservation. At60 TP585 to589 and F1.586760 to.603484; at80 TP451 to448 while
F1.452357 to.459016. Both trained targets remain recovered at50/60; boat also
at80. The person's actual box is[455,184,489,231], IoU.667720; boat remains
[290,460,371,562], IoU.894584. This is natural changed-history behavior, not
exact-prefix replay. All130 outputs reach EOS with no caps,4 cold loads and
2 isolated scores,10543 new tokens and944.57 allocated GPU-seconds.

Inference: stronger support preservation can contain the previous broad
precision/repetition damage without abandoning the two local recoveries on
this exposed development panel. This does not establish held-out improvement,
zero old-owner loss, universal localization improvement, or a unique mechanism.
The guard parser-drop increase receives one bounded raw-output diagnosis.
Because this is the first promising joint result, freeze the unchanged
checkpoint for the remaining254 Source-train images, separating48 reference
images from206 images outside this update's positive/reference support. This
is data expansion for evaluation, not new training or confirmation512 access.

- [Sealed strong100 training receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48-strong100/training/receipt.json), SHA256 `037f1af3561e5522837364c41266af135e20c3d80579767092c63b761d6821ba`.
- [Strong100 natural130 reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48-strong100/evaluation/reduction.json), SHA256 `a7d42b1d72547e034751668e2fd7a4f8ccd018b6cd36935fc2de0e1a427c14e4`.
- [Complete consumer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48-strong100/evaluation/consumer.json), SHA256 `8b8b1d967d8b058f3fc35d03b0e9245a44da33647f0b41926178049221ba776b`.

### Parser qualification of the positive development screen

The bounded diagnosis finds no parser/scorer defect. Root replays the six
decisive Source/candidate parser/scorer checks and aggregate arithmetic.
Only6/128 images change drop counts, all `geometry_invalid`. On122 images
with equal drop counts, F1 still rises.677864 to.683283 but TP565 to562:
outside repaired cases this is a precision/volume tradeoff, not recall gain.
Four drop-improved images supply TP+8, two worsened images-1, the remaining
images-3. In particular39654 removes46 reversed/zero-width banana drops and
gains5 owners without losses. This is a real malformed-sequence reduction,
not a new parser excluding valid predictions; no one-to-one causal conversion
of invalid spans to recovered owners is established.

Conversely59571 develops14 additional zero-width bottle spans and loses an
owner. Its valid predictions fall80 to58, but total valid+dropped spans only
fall81 to73:14 of the22 fewer valid predictions reflect extra malformed
output. Do not call guard16 clean preservation. A labeled sensitivity adding
one error unit per dropped span keeps the global positive direction
(.598441 to.625506) but leaves guard16 nearly neutral(.375839 to.377622).
This sensitivity does not replace official F1 or classify physical objects.
The larger read retains all capped/dropped Source cases without filtering.

- [Named parser diagnosis and exact groups](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/strong100-parser-diagnosis/README.md).
- [CPU reproduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/strong100-parser-diagnosis/reproduce.md).

### Expanded254 reverses the proposed strong100 promotion

Root replays the frozen task-local consumer and all four partition/admission
tests. All254 new outputs and8 process exits are complete, including the4
capped Source cases, without outcome filtering. Strata remain48 preservation
references versus206 outside this update's positive/reference support, with
all128 development images disjoint. The same sealed checkpoint is used and
no original130 image is generated again. Eight cold loads, zero entry scores,
29645 new tokens and2713.63 allocated GPU-seconds produce250 EOS/4 capped
outputs; the maximum shard takes700.05 seconds. Four output caps are a measured
model outcome, not a missing-run or technical-invalid status.

| Frozen stratum | Source TP50 / FP50 / F1 | Strong100 TP50 / FP50 / F1 | Repeats Source to strong |
|---|---:|---:|---:|
| Outside206, primary |869 /817 /.569836|863 /879 /.555699|463 to501|
| Reference48 |374 /245 /.628044|376 /245 /.630344|4 to6|
| Remaining254 |1243 /1062 /.586183|1239 /1124 /.576413|467 to507|
| Train256 descriptive |1259 /1064 /.588593|1257 /1126 /.579530|467 to507|
| Union384 descriptive |1873 /1553 /.597258|1875 /1569 /.596184|579 to603|

Outside206 gains13/loses19 owners at50, gains13/loses17 at60 and gains17/
loses16 at80. F1 decreases at all three thresholds; tokens23671 to23778,
parser drops794 to750 and caps4 to4 remain visible. Reference48 gains12/
loses10 owners at50: teacher preservation is not exact owner preservation.
The384-image union has48 gained/46 lost owners, only+2 net, with worse F1 and
more repeats. The dev128 positive is real on that panel but does not persist
on the broader acquisition pool. **Strong100 is not promoted.** No claim of
held-out generalization or stable joint benefit survives this read.

The next hypothesis concerns positive-signal supply rather than another KL
weight change. All four arms learned from the same two entrance labels; more
preservation data/strength alone has not yielded a robust joint gain. An
existing inventory contains six further annotation-backed Source-native first-
row fork candidates, but none is yet a valid positive training entrance. The
next packet checks their eight retained single-token variants through Source
greedy release, with no new training until admission and a new frozen recipe.

- [Expanded254 frozen packet](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48-strong100/expanded-train254/manifest.json), SHA256 `a61616a8f74bbf8793bbc457a6a9805c93e591d62863949ec7f48ef8a299d3ca`.
- [Expanded254/384 reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48-strong100/expanded-train254/reduction.json), SHA256 `2619cd49695dbb2c14d3fbded81741f02787ab861b98ce5e46d770eddc8871b7`.
- [Native254 consumer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48-strong100/expanded-train254/consumer.json), SHA256 `90310d5b816f7bbb7ae5f92abcbafe8a6850572937e19b858eb30d80e5b40e21`.

## Positive supply: five additional useful conditional entrances

The frozen8 single-token Source releases complete in one load/35.36seconds,
with8 image forwards,334 model forwards/new tokens and no training. Root
replays the full reduction and8 boundary/eligibility tests. Five actual first
rows, not merely sampled rows, qualify; all three table variants fail and
are retained without a fallback.

| New target image / owner | Actual first-row IoU | Conditional outcome |
|---|---:|---|
|252411 /1676022 hair drier|.607201|Target50/60; old owners retained; no FP/repeats/drops|
|465695 /1612233 bed|.987554|Target50/60/80; suitcase/cat retained; no FP/repeats/drops|
|529411 /2147505 teddy bear|.926241|Target50/60/80; dog retained; no FP/repeats/drops|
|538814 /1121312 oven|.911548|Target50/60/80; FP1 to0; localization repair|
|540567 /1491504 bottle|.581158|Target50 plus untrained1488726; old owners retained; FP1 unchanged|
|496747 /1614837 table, three variants|.391/.394/.375|All fail target matching; original suffix owners retained|

All8 reach EOS; no Source-owner loss occurs at50/60/80. The five selected
controls gain6 IoU50 owners in total because bottle's continuation adds an
unsupervised second bottle. This is conditional feasibility, not autonomous
retrieval or training success. The bed is heavily occluded with full-frame
annotation extent; the plush is tiger-shaped under the existing `teddy bear`
category; no GT or category semantics have been changed. The table failure
shows that selecting a sampled first coordinate does not suffice for every
sampled correct row; no complete-row injection was added to rescue it.

Root admits exactly these5 labels for the next frozen package, giving7 total
positives. Preserve original2 local trajectories/masks, use actual Source
single-token-release trajectories for the new5, remove529411's conflicting
full-Source reference and retain the other47. The new fixed81-step package
uses `mean7 CE + 10mean7 localKL + 100mean47 supportKL`; dose81 roughly
maintains prior per-positive coefficient-time under the larger mean but is
not optimizer-equivalent. It is a joint positive-pool/dose test, not an
isolated diversity mechanism experiment. Its sole natural384 read will keep
positive7/reference47/train202/dev128 separate. At that boundary the fifth-arm
quality read had not yet run; its completed result follows below. See the
[frozen protocol](unit.md#fifth-frozen-iteration-seven-positive-entrances-with-support47).

- [Source release inputs](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive-branch-expansion/inputs.json), SHA256 `710c21601156a6e04a3ad195d71f824203c0a2f7eb3cf080dd0b7da79180eb4f`.
- [Eight conditional reductions](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive-branch-expansion/reduction.json), SHA256 `4bc95c3d736dab4d2614d087c67c24a36477f621ea12a14605fb180d6c712728`.

## Seven81: local expansion succeeds, three new capped outputs prevent promotion

Root replays the producer/consumer verification CLIs,12 focused CPU tests and
all81 update records. Training completes with the registered54 references,
4374 differentiable forwards and14 isolated scores:4442 total,567 positive
uses/3807 support uses,8 loads and3731865480 reference-cache bytes. All54
initial KLs are0; step2 exercises nonzero KL; all81 updates have exactly one
cross-rank gradient, adapter and optimizer hash. Frozen bytes remain unchanged.
Maximum rank time782.02s, summed rank time6245.61s; adapter movement L2 is
1.121352. Last pre-update81 mean local/support KL is.000951822/.000454061,
not a terminal KL measurement. Terminal isolated margins1.94–6.37 are not
quality acceptance gates.

Eight cold replicas produce all384 outputs, seven isolated entry readbacks
and49211 natural tokens in4341.30 allocated GPU-seconds. All8 processes exit0;
7 capped generations are valid recorded outcomes, not missing executions.

| Frozen panel | Source TP50 / FP50 / F1 | Seven81 TP50 / FP50 / F1 | Repeats | Drops | Caps |
|---|---:|---:|---:|---:|---:|
| Positive7 |23 /4 /.754098|31 /3 /.911765|0 to0|0 to0|0 to0|
| Reference47 |373 /245 /.627946|377 /239 /.635750|4 to5|0 to0|0 to0|
| Remaining202 |863 /815 /.569825|865 /988 /.539950|463 to594|794 to1645|4 to7|
| Dev128 |614 /489 /.615848|611 /407 /.640126|112 to80|58 to17|0 to0|
| Outside330, primary |1477 /1304 /.588095|1476 /1395 /.577352|575 to674|852 to1662|4 to7|
| Union384 descriptive |1873 /1553 /.597258|1884 /1637 /.591801|579 to679|852 to1662|4 to7|

All7 trained owners recover naturally at50,6 at60 and4 at80, with no old-owner
loss or new repeats/drops on those7 images. Positive7 gains8 total owners,
including the untrained second bottle. This extends the local result beyond
the original two coordinate entrances. Outside330, however,35 gained/36 lost
owners and higher FP/repetition/invalid-output burden fail the primary joint
criterion. **Seven81 is not promoted.** Lower in-reference KL and stronger
entry margins have not guaranteed native rollout stability outside support.

### Decision-bearing cap concentration, not a filtered success score

Root projects the retained per-image reduction:7 images are capped under
either model. The other323, a post-outcome diagnostic subset only, improve
TP1459 to1466, FP787 to746, F1.665450 to.673868, repeats124 to100 and drops
64 to22. On the7 cap-associated images, TP18 to10, FP517 to649, repeats451
to574 and drops788 to1640 reverse the aggregate direction. These7 remain
inside every official panel and no filtered number replaces the primary score.

Three previously clean-EOS Source cases newly cap at3084:

| Train image | Source tokens | Seven81 tokens | TP50 | FP50 | Parser drops |
|---|---:|---:|---:|---:|---:|
|73843|10|3084|1 to1|0 to83|0 to259|
|360573|95|3084|6 to1|4 to27|0 to315|
|545632|81|3084|4 to0|4 to3|0 to306|

They were outside both the positive and reference supports. Existing Source
caps351017/417044/477415/502725 remain capped; their burden also changes, so
the3 new cases are not the sole possible failure. Non-cap cases are not
uniformly improved either. The next bounded contrast adds exactly these3
unchanged, clean Source full trajectories(including EOS) to reference47,
keeping positive7, coefficient masses and fixed81 dose unchanged. This tests
targeted reference coverage, not an EOS-only mechanism, a parser fix or a
general stability guarantee. All384 cases remain in the next read.

- [Seven81 sealed training receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive7-support47-81/training/receipt.json), SHA256 `7eee80e9684bd7e094e04371b11fcc00306c12e42476cc6a1876459d51fe7652`.
- [Seven81 natural384 reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive7-support47-81/evaluation/reduction.json), SHA256 `6050ae10132129430cbf3dcb1ae7e282f7d7f8182fb60a364d92f084b6bfacc5`.
- [Native384 consumer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive7-support47-81/evaluation/consumer.json), SHA256 `616470a47a0b785b79b39a31085b42b6be3c3f0f7a2700a3648342d2e990dc9a`.

## Stable50 closeout: a bounded IoU50 positive with explicit debts

The sixth arm adds only the3 frozen clean Source trajectories to reference47,
keeping positive7/local masks/81 steps and component masses unchanged.
Root independently replays producer and consumer verification,11 focused
tests and all81 update records. Exactly57 references +4617 differentiable
forwards +14 isolated scores =4688 model forwards,567 positive uses and4050
support uses complete on8 ranks. All57 initial KLs are0; step2 exercises
nonzero KL; all81 rank gradient/adapter/optimizer identities agree and frozen
bytes remain unchanged. Sealing follows all8 terminal receipts and launcher
exit0, not rank0 alone.

Training uses3845451960 cached bytes,878.66 maximum rank seconds/7017.23
summed rank seconds and31.178GB maximum allocated CUDA memory. Adapter
movement L2 is1.208333. Last pre-update81 mean local/support KL is
.000699561/.000364415; these are not terminal KL measurements. The cold
consumer independently checks all7 isolated entry readouts, then scores all
384 unforced natural generations:8 loads,40200 tokens,3765.70 allocated
GPU-seconds and887.20 maximum shard seconds. All8 processes complete;
380 EOS/4 caps is the model outcome, not a technical failure.

| Panel, IoU50 | Source TP / FP / F1 | Stable50 TP / FP / F1 | Gained / lost owners |
|---|---:|---:|---:|
| Positive7 |23 /4 /.754098|31 /3 /.911765|8 /0|
| Reference50 |384 /253 /.620857|389 /242 /.632006|9 /4|
| Remaining train199 |852 /807 /.571812|865 /861 /.567772|25 /12|
| Exposed dev128 |614 /489 /.615848|614 /416 /.639250|17 /17|
| Outside327, primary |1466 /1296 /.589465|1479 /1277 /.595411|42 /29|
| Previous outside330, unchanged IDs |1477 /1304 /.588095|1489 /1284 /.593819|42 /30|
| Full384 descriptive |1873 /1553 /.597258|1899 /1522 /.606032|59 /33|

All7 directly trained targets recover at50,6 at60 and4 at80. Their seven
images retain every old50/60/80 owner and have no repeats or drops; the extra
bottle remains an untrained gained owner. The full384 net+26 owners are not
26 loss-free additions:59 are gained and33 are lost. Existing outside330
improves without excluding the3 newly trained reference cases, while the
current outside327 is disjoint from all7 positive and50 reference images.
These are exposed development/acquisition panels, not untouched held-out data.

| Full384 burden / stricter threshold | Source | Stable50 |
|---|---:|---:|
| TP60 / F1@60 |1775 /.566008|1787 /.570289|
| TP80 / F1@80 |1359 /.433355|1354 /.432105|
| Valid predictions |3426|3421|
| Strict repeats |579|582|
| Parser drops |852|792|
| Complete tokens |40756|40200|
| Cap count |4|4|

The current cap identities exactly match Source:
351017/417044/477415/502725. No new image caps. The3 targeted cases all return
to EOS:73843 returns to10 tokens/TP1/FP0;545632 returns to81 tokens/TP4/FP4;
360573 ends at85 tokens/TP5/FP3 but loses Source owner1789916 and has one
parser drop. Thus targeted containment is not exact Source replay. Whole
outside327 repeats575 to577 and drops852 to791 remain visible. IoU80 outside
support worsens(TP1094 to1084,F1.439887 to.436393), and remainingtrain199 F1
still falls despite13 net more owners. Dev128's benefit is precision rather
than net50 recall and it loses11 TP80. Do not call the result universally
selective, uniformly better localization, or clean preservation in every group.

**Lead decision and stop.** Accept the final saved candidate for this bounded
IoU50 outcome: both complete384 and outside-support327 gain owners and F1,
annotation-relative FP and invalid-output burden decrease overall, and the
catastrophic new-cap failure is contained without cap migration. The small
repeat increase and the strict-IoU/subpopulation/owner-exchange regressions
are accepted and explicit, not waived by a hidden score change. This closes
the autonomous search at a useful but modest tradeoff, not a proof of a
unique mechanism or a robust population-wide improvement. No seventh arm or
confirmation read follows automatically; all workers and study GPU jobs have
settled. Keep the existing Source/default configs unchanged.

The strongest remaining scientific alternative is adaptation to an exposed
finite pool rather than transferable selectivity. A genuinely untouched
confirmation evaluation would discriminate that claim, but requires a new
contract and explicit reopening of the reserved confirmation axis. Another
possible future objective is better IoU80/remaining199 preservation; neither
branch is launched. Unlimited compute was not treated as a requirement to
continue after the bounded question had a decision-bearing result.

- [Final training receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive7-support50-81/training/receipt.json), SHA256 `5e4a6130d28d6ec65cc03ceadf70088fea9bb6f6b2a757db8e5ce0c1688b8833`.
- [Final native384 reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive7-support50-81/evaluation/reduction.json), SHA256 `8c1a31ea9457d97bce575b76b53a9fec11621d22eb958e106a8b7b1538f4a16a`.
- [Final native384 consumer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive7-support50-81/evaluation/consumer.json), SHA256 `27850345b25c2e1eb26e11fae9e58e4fe2be35e8294889a6f615f3a518c1cde3`.
- [Lead acceptance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive7-support50-81/lead-acceptance.json).
- [Final adapter](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/positive7-support50-81/training/adapter). Compose it with the unchanged Source special-token embeddings and exact base recorded in the receipt; do not substitute the base tokenizer/model family.

## Boat cross: changed history is not required for old CE's repetition

Two new crossed suffixes complete the Source/old-pointCE23 by H_good/H_new
table, reusing both retained diagonals. H_good has27 tokens/3 complete rows;
H_new has45 tokens/5 rows through the naturally generated target boat. Their
IoU50 covered-owner sets happen to agree, but their geometry, ordering, length
and higher-IoU coverage differ. Only within-history checkpoint contrasts are
controlled; matching owner sets do not establish semantic-state equivalence.

| Fixed history | Checkpoint | TP/FP/FN @50 | F1@50 | New suffix predictions/repeats/drops |
|---|---|---:|---:|---:|
| H_good | Source, retained |5/0/1|.909091|2/0/0|
| H_good | old CE23, new cross |6/26/0|.315789|29/20/1|
| H_new | Source, new cross |5/2/1|.769231|2/0/0|
| H_new | old CE23, retained |6/30/0|.285714|31/21/1|

All four end naturally. Old CE gains only person1759775 at50 under either
history, without owner loss. Its altered natural history is therefore not
required for the repetition burden: old CE repeats even at H_good. Conversely,
H_new alone does not force Source to repeat. This establishes a checkpoint-
dependent continuation effect at these two prefixes, not an internal mediator,
general ledger failure or the specific responsibility of the boat training
example. The new KL10 candidate is not part of this table.

Root freshly replayed the CPU verifier and exact four-cell reduction. Total
new work: two cold loads, two generations,290 new tokens/forwards and69.54
seconds. The first process completed Source(H_new), then exited1 at an overly
strict zero-CUDA-allocation cleanup assertion. Preserve that failed terminal
and completed raw row. A separately authorized fresh process ran only the
untouched CE(H_good) cell and exited0; no cell was rerun. Overall diagnostic
execution is complete and lead-accepted, with the failed invocation disclosed.

[Boat-cross receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/boat-cross/receipt.json)
and [four-cell consumer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/boat-cross/consumer.json).

## Reproduction and evidence

Root freshly replayed both verification CLIs and10 focused tests, confirming
training identity/counters, cold full-vocabulary parity, all18 raw native
consumers/reductions and rendered-image coverage. Natural evaluation takes
173.12 seconds, one independent GPU load, two cold score checks and2093 new
tokens; model work totals297.87 seconds, excluding preparation/review.

```bash
python -m pytest -q probes/dora_owner_learning/tests/test_selective_preservation.py probes/dora_owner_learning/tests/test_selective_preservation_eval.py
python -m probes.dora_owner_learning.selective_preservation verify
python -m probes.dora_owner_learning.selective_preservation_eval verify
```

- [Training receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-10/training/receipt.json).
- [Frozen screen and Source controls](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-10/evaluation/manifest.json).
- [Natural raw trajectories](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-10/evaluation/execution/rows.jsonl).
- [Native reduction and owner changes](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-10/evaluation/execution/reduction.json).
- [GT/prediction visual manifest](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-10/evaluation/execution/visualizations/manifest.json).
- [Bounded DeepSeek versus Astra worker trial](worker-trial.md).

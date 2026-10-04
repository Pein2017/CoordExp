# Conditional localization after supplied x1; native bottle entry remains unresolved

Lead-accepted and closed. Supplying a missed bottle's x1 produces an approximately
localized complete box under both tested first-person histories. A different
bottle x1 produces a different, approximately localized box. Without a coordinate
cue, both histories still select x1=0 and miss all supplied bottle annotations.
This separates a usable conditional completion route from its natural selection
at these prefixes; it does not establish physical recovery or a unique cause.
The same-hidden-state FP32 readout changes some local winners but leaves bottle
x1 at0. No training or successor is authorized by this result.

## Scope and technical acceptance

The [unit](unit.md) owns the frozen question, ten cells, B16 checkpoint, two
images, annotation identities and stop rule. Denominators remain four x1-cued
rows, three uncued rows (including two controls), and three teacher paths.
All cells receive literal prior history and a category/box header; "uncued"
means no supplied coordinate in the selected row, not empty-history generation.
The original persistent Worker `01a101ac-9ec8-7862-bdb6-38b9cd154673` executed
the single release. Implementation commit was
`3f5725abc61444cfac61ccbbc2f95ac1fa92cb77`; clean execution commit was its
records-only descendant `530d2c3fec7232aa25b2b5fa1db88e2e655d6ade`.

All10 requests completed841 actions:807 forced and34 free, one B16 load/GPU0,
10 prefill predictions and831 cached suffix predictions. There was no training,
backward, optimizer, replay, warmup, export, retry, extra acquisition or decoder
pass. Both native controls exactly reproduced every saved token and incoming
median winner (255 actions total); raw chosen log-probability differences were0.
Maximum median differences were5.3644e-7 and7.1526e-7, retained descriptively.
All selected rows were complete, valid, coordinate-aligned and closed. Every
request ended at its literal budget without EOS; this does not test full-scene
stopping or continued enumeration. The supplied person prefixes retain their
pre-existing duplicate event; it is not selected-box recovery evidence.

External and producer exits were0. External wall was100.4598s, producer98.7780s,
including9.0797s loading,86.9027s generation and0.3830s CPU shadow computation.
Peak RSS was12.8264GiB, allocated CUDA4.7351GiB and retained payload29.4022MiB,
within900s/32GiB/12GiB/64MiB. Both owned PIDs and their process group settled;
the GPU settlement receipt confirms the native PID absent. Source, release and
input bindings remained unchanged through terminal verification.

Lead reused14 distinct passing CPU checks and preserved their failed fixture
attempts, verified the saved native artifact identities, literal forced actions,
row/annotation accounting, denominators, companion summaries, exits and cleanup.
The automatic consumer was not reinvoked. A narrow independent read-only review
found no head/prefix/factor binding or shadow-arithmetic blocker. Lead acceptance
covers the completed diagnostic and its bounded interpretation; user scientific
acceptance and checkpoint promotion are separate and are not asserted.

## Complete-row observations

Coordinates use0..999 bins. IoU is an annotation-relative observation, not a
physical identity claim or a technical gate. Target bottle annotation
-4947389372712316 is `[186,30,207,106]`; distinct bottle1489041 is
`[495,396,543,614]`; person191150 is `[546,633,556,682]`.

| Cell | Selected box | Designated IoU | Supplied selected coordinates |
|---|---|---:|---|
| bottle-native | `[0,0,23,86]` |0|none|
| person-native | `[544,632,559,682]` |0.653333|none|
| bottle-native-target-x1 | `[186,30,210,123]` |0.715054|x1|
| bottle-corrected-history | `[0,0,21,86]` |0|none|
| bottle-corrected-target-x1 | `[186,30,210,123]` |0.715054|x1|
| bottle-corrected-control-x1 | `[495,377,552,648]` |0.677413|x1|
| bottle-target-teacher | `[186,30,207,106]` |1|all four|
| bottle-control-teacher | `[495,396,543,614]` |1|all four|
| person-target-x1 | `[546,632,558,682]` |0.816667|x1|
| person-target-teacher | `[546,633,556,682]` |1|all four|

For the focal bottle, both cued histories freely produce y1=30, x2=210 and
y2=123: errors0,+3,+17 relative to its annotation. Its overlap is largest with
the designated annotation; the adjacent bottle overlap is0.212523. For the
distinct cue, free-coordinate errors are-19,+9,+34 and only the designated bottle
has positive same-category overlap. All37 bottle annotation overlaps and all7
person overlaps remain in the saved consumer. Both uncued bottle rows have0
overlap with every same-category annotation.

The distinct cue changes y1, width and height, weakening a fixed-box translation
explanation. It does not exclude a spatial-cue-to-box mapping, memorized scene
geometry or other cue-dependent completion. No image intervention identifies
visual causal use or an internal owner representation. Correcting the preceding
person changes uncued x2 from23 to21 but does not recover bottle entry; this is
a bounded negative for that specific history repair, not history invariance.

## Exact teacher tokens versus approximate localization

The target x1=186 has unforced median full-vocabulary rank141 and target-minus-
best-other gap-5.690680 under native history, versus rank130/gap-5.444818 under
corrected history. Distinct x1=495 has full rank461/gap-10.674442 under the same
corrected prefix. Forcing a later x1 does not change these preceding scores.

| Teacher path | x1 rank / gap | y1 rank / gap | x2 rank / gap | y2 rank / gap |
|---|---|---|---|---|
| Focal bottle, corrected history |130 / -5.444818|1 / +0.235558|14 / -0.842432|10 / -0.285748|
| Distinct bottle, corrected history |461 / -10.674442|55 / -2.647823|26 / -1.896584|25 / -0.315420|
| Normal person, native history |5 / -0.854153|2 / -0.031563|3 / -0.291960|1 / +0.017899|

These are median full-vocabulary ranks/gaps at each path's actual supplied
prefix. Raw scores, coordinate-family ranks/ties, full normalizers, competitors,
header/closure likelihoods and actual selected-action likelihoods are retained.
The normal native box already has0.653333 annotation overlap despite three
negative exact-teacher gaps. Thus exact-token deficits do not equal complete
localization failure, and the results do not mean that only x1 is wrong or that
the remaining corners are fully learned. Teacher IoU1 is wholly supplied geometry.

## Same-state coordinate-only FP32 companion

All40 original coordinate slots were captured and paired with the actual native
head/raw processor, request, action and prefix. After acquisition closed, CPU
arithmetic projected the same BF16 h through separate runtime BF16 base and FP32
delta rows in FP32, added the results, then applied the captured FP64 median
factors with FP64 multiplication and FP32 cast. No decoder/native-head rerun or
feedback occurred. Upcasting cannot restore bits already lost in h/base values.

| Channel | Coordinate winners changed | Conditional TV median / max | Conditional W1 median / max (bins) |
|---|---:|---:|---:|
| Raw |14/40|0.017972 / 0.041597|0.102257 / 0.633290|
| Median |10/40|0.018101 / 0.039230|0.114019 / 0.630303|

All14 raw changes begin at native winner ties (15 raw-tie slots total). Changed
median sites have native top-two gaps0.004677..0.095161; the largest winner shift
is12bins, distinct teacher y2=638 to626. Focal teacher y2 changes107 to99.
These40 records include repeated literal prefixes and are not independent
samples. They describe one-step readout alternatives, not generated FP32 boxes.

Bottle x1 stays0 in both histories. Focal186's coordinate-family rank/gap is
140/-5.625921 under native history and143/-5.573881 under corrected history.
The tested final-head arithmetic therefore affects local competition without
resolving this entry deficit. It does not rule out other precision effects or
identify the training cause. No FP32 full-vocabulary winner, normalizer or mass
was computed. Native full-normalizer-derived coordinate masses include tiny
rounding excesses above1 (raw max1.000000658; median max1.000001912); these remain
unclamped numerical observations, not probabilities greater than1.

## Decision and evidence locators

This unit supports conditional approximate localization once an informative x1
is supplied, while the tested uncued entry remains poor. It weakens both total
conditional-localization incapacity and a final-readout-tie-only explanation of
the entry failure. Entry is not established as the sole obstacle. Natural
physical-FN reduction, visual causal use, sustained enumeration, incumbent
preservation, generalization and a training-objective remedy remain unmeasured.
B16 already uses hard target-token CE; no soft/hard objective contrast occurred.
The frozen package is complete, all jobs settled, and no successor is scheduled.

Artifact root: `outputs/research/physical-fn-recovery/2026-10-04/owner-entry-localization/`.
`prepared-01/input-packet.json` and `native-release-01.json` bind source, protocol,
runtime, checkpoint and original image/label/policy inputs. `native-01/readback.json`
owns all rows, conditions and primary metrics; each named condition JSON retains
literal actions, likelihoods and score/capture vectors. `native-01/companion.json`
and `companion-runtime-rows.safetensors` retain the auxiliary arithmetic evidence.
Costs and cleanup are in `native-01/terminal.json`, `resource-01..10.json`,
`native-01-owner-invocation.json`, `native-01-settlement.json` and
`native-01-gpu-settlement.json`. `native-01-summary.json` is the Worker projection;
`lead-consumer-check-01.json` records Lead acceptance without inference.
Readback SHA256 is `d65a4e8f6dd855085fcd06b532067d783e859e257755ed9c7f8a8eeb04f5bf01`.
CPU preparation and failed/successful checks remain under `cpu-qualification-01/`.

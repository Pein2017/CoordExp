# Coordinate-cued completion depends on the perturbed image region

Lead accepts and closes the frozen12-cell native contrast. All requests and
four clean anchors passed; external and producer exits are0, and the source
holder is released. This establishes conditional region dependence in the two
selected histories. It does not establish natural owner discovery or a unique
internal mechanism. User scientific acceptance and checkpoint promotion are
not asserted.

## Primary result

Supplying the same x1 does not make the completion independent of the image.
For the bottle, target-region masking changes the first free y1 distribution
substantially and produces a wider box; background masking has a much smaller
effect on that first decision. The person control also retains substantial
target-region dependence after x1 is supplied.

| State / supplied x1 | Input | Actual complete box | Designated annotation IoU |
|---|---|---|---:|
| Bottle /186 |clean |`[186,30,210,123]` |0.715054|
| Bottle /186 |target mask |`[186,28,226,115]` |0.458621|
| Bottle /186 |background mask |`[186,30,208,99]` |0.870270|
| Person /546 |clean |`[546,632,558,682]` |0.816667|
| Person /546 |target mask |`[546,656,552,680]` |0.293878|
| Person /546 |background mask |`[546,632,559,682]` |0.753846|

The supplied x1 is privileged information. IoU uses the original annotations
as fixed response references after masking, not ground truth about an erased
object. For the target-masked bottle, the best annotation overlap is the adjacent
box `[200,27,229,133]` at0.527027, exceeding the designated0.458621. This does
not identify a physical-owner switch. All37 bottle/7 person overlaps per cell,
including zeros, remain available.

The uncued bottle still chooses x1=0 on all three inputs: clean/target boxes
are `[0,0,23,86]`, background `[0,0,22,86]`, with zero overlap with every bottle
reference. The uncued person reproduces the prior contrast: clean
`[544,632,559,682]`, target `[684,607,701,638]`, background `[545,632,558,682]`.
Its designated overlaps are0.653333,0 and0.753846 respectively. These uncued
cells still receive the original native history and selected category/header;
they are not empty-history detections.

## First free decision before suffix feedback

At cued bottle action15 and person action232, all three image conditions have
the exact same literal prefix. Let `q=s(y_GT)-s(0)` in the actual unforced
median channel, with y_GT30/633. Endpoint0 is a fixed score reference, not a
claimed natural competitor for the person. TV and W1 use coordinate-conditional
distributions; full-vocabulary family mass is retained separately.

| State | Target−clean q | Background−clean q | Target−background q | TV target/clean | TV background/clean | W1 target/clean | W1 background/clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bottle |−4.629335 |+0.006495 |−4.635830 |0.584700 |0.033009 |20.220876 |0.167691|
| Person |−9.148609 |−0.180567 |−8.968042 |0.876617 |0.084723 |25.479558 |0.406531|

W1 is in coordinate bins. Clean q is4.308880 for the bottle and20.085034 for
the person. Median y1 winners are bottle30/28/30 and person632/656/632 under
clean/target/background. In the bottle target cell, raw winner0 differs from
the actual median-selected28; raw selected-action likelihood remains associated
with28. Raw/median channels and forced policy traces are not interchangeable.

Target-mask cued trajectories first diverge at these y1 actions. Background
trajectories first diverge at x2. Later comparisons can therefore include token
feedback. In uncued person cells, even y1 follows different x1 values. The saved
consumer labels actual prefix equality and semantic coordinate roles rather
than treating every nominal slot as a fixed-prefix image contrast.

## Secondary comparison: cue dependence at the same y1 role

The already acquired bottle data permit a useful additional comparison. Its
uncued x1 is0 under all three images, so its y1 prefixes also match within that
triplet. Between uncued and cued triplets the text prefix differs only at the
immediately preceding x1 token:0 versus186. This avoids comparing x1 sensitivity
with y1 sensitivity. The following is a secondary derived observation, not a
replacement for the frozen primary result.

| Bottle y1 context | Target−clean q | Background−clean q | Target−background q | Target/clean TV | Target/clean W1 |
|---|---:|---:|---:|---:|---:|
| Native x1=0 |−0.240755 |+0.006496 |−0.247252 |0.062930 |1.777729|
| Supplied x1=186 |−4.629335 |+0.006495 |−4.635830 |0.584700 |20.220876|

The paired score interaction is−4.388578. The same target-region intervention
has a much larger effect on this decision after the changed x1, while the
background q effect is nearly unchanged. A purely image-independent lookup
from the coordinate cue to a box cannot explain these measured region effects.
An image-conditioned spatial prior or distributed image-conditioned state can
still participate; fresh attention to a selected object has not been identified.

## Interpretation and next unresolved question

The supported computation-level statement is that changing the preceding
coordinate token changes the model's conditional y1 response to the same
regional pixel intervention. A plausible hypothesis is that x1 acts as a spatial
query that changes which visual evidence can affect continuation. Alternatives
include a coordinate-conditioned readout of already formed visual state,
nonlinear mixing with spatial priors, and wider contextual effects of the mask.
The effect could also reflect a general change in visual susceptibility rather
than selection of this particular region.

The nearest discriminating question is cue/region specificity: whether another
valid bottle cue changes which of two fixed image-region interventions affects
completion. That is a new contrast, not an extension of this consumed release.
The earlier495 completion used a corrected first-person history; its behavior
under the current native history remains a new observation. No architecture,
training remedy, layer/cache attribution or population claim follows this unit.

## Technical acceptance, cost and evidence

The single invocation used execution source
`fb05481a95278e3ac9eaec8ee0d754163a673199`, implementation
`98afc295f1bb76debed8be23d1d006c8f7ead3fc`, the frozen B16 checkpoint, six existing
RGB inputs and the current maintained runtime. All12 requests completed1530
actions:1476 forced/54 free,12 prefills/1518 subsequent singleton predictions,
one checkpoint load and zero training. All four anchors reproduce every output
ID and pre-force median winner. Every selected box is complete and valid; all
requests end at the literal budget with closure and no EOS. Full-scene stopping
and continued enumeration remain censored.

Native selected likelihoods are unchanged. All96 saved FP32 score vectors pass
the canonical one-thread CPU reconstruction and consumer. Four descriptive
native-device/CPU normalizers differ by at most1.90735e-6. Seven family-mass
diagnostics slightly exceed1, with maximum1.00000191246, because their FP64
coordinate reduction is paired with the frozen FP32 full normalizer. These
numerical limitations are retained without clipping or tolerance changes.

External wall is173.082018s, producer169.721059s; load/session construction
5.333271s and acquisition156.457820s. Peak RSS13,843,193,856B, allocated
CUDA5,084,301,312B and retained payload67,616,760B meet the frozen limits.
The automatic saved consumer ran once. No extra acquisition, retry, warmup,
model/head pass or duplicate consumer occurred. Matching PID589487,
supervisor589484 and process group are absent; no owned GPU process remains.
Source/release/input identities are unchanged and the source holder is released.

Lead reused the accepted19 CPU checks, reviewed the native saved consumer and
artifact identities, verified counts/forcing/roles/overlaps and independently
reconstructed the primary and secondary score differences using Python only.
No test, consumer or model call was repeated. The previous native HOLD and
failed CPU receipts remain immutable.

Artifacts are under
`outputs/research/physical-fn-recovery/2026-10-04/cued-visual-use/`.
`native-01/readback.json` owns the complete matrix, roles, likelihoods and
contrasts; its SHA256 is
`be1e858879cdecc761755235c6c9e48c3d49c504a8982d94bec6155af8ddfae0`.
`native-01-summary.json` and `native-01-report.md` retain the detailed projection;
raw per-cell JSON/safetensors remain under `native-01/`.
`lead-native-acceptance-01.json` binds the consumer, summary, terminal, owner,
settlement and separate secondary calculation. `native-01-settlement.json`
owns cleanup/source release. The [unit](unit.md) and [release ruling](lead-ruling-01.md)
retain the scientific contract and CPU/native boundary.

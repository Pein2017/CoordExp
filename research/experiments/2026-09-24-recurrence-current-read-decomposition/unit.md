# Current-query historical contribution versus redistribution

2026-09-24; finite probe. Lead 01a0c831-b332-7e12-b931-6ebe2359c99f (gpt-6-astra/xhigh); worker 01a0ce4b-9b55-7392-8a25-6a76f9e12c3a (gpt-6-sol/xhigh). Frozen scientific scope; exact source/reference bindings and execution grant are in [lead-admission-v1.json](lead-admission-v1.json).

## Question and predecessor boundary

From the native one-record A anchor, does removing the historical weighted-value contribution without redistribution approach the current-query mask x2 distribution more closely than redistributing remaining-key weight while retaining that historical contribution?

The [accepted query partition](../2026-09-24-recurrence-coordinate-query-routing/results.md) favors current-query removal relative to earlier-query removal. Earlier-only still has TV0.796782 to native. Reciprocal x1/y1 complete-row routing predictions failed. The older [live-query mass/profile experiment](../2026-09-22-recurrence-attention-mass/results.md) rejected a shared mass-following winner rule at other conditions. This unit resolves a specific confound of the accepted current-query deletion; it does not reopen those negatives, extend an image cohort, or seek a general mass law.

[CPU feasibility acceptance](../2026-09-24-recurrence-coordinate-query-routing/supporting/lead-read-decomposition-cpu-acceptance-v1.json) rehashed26 bindings and independently checked local arithmetic. It corrects a scalar label in the immutable CPU candidate: native→current-only TV is0.9697091740894589;0.9734647803791534 names native→both. Neither scalar substitutes for this experiment's fresh baseline. Fresh actual Q/K/V and pre-o_proj captures remain necessary.

## Fixed source and operation

Original refined-03 four-request batch, target2 train351017, untied step2444 FP32 SDPA. Original source/model/checkpoint/loader, images, companions, left pads, tokens, causal mask, positions and request order are fixed. Historical A raw[:9] is [151646,8987,151647,151648,151670,151683,152206,152669,151649]. Every call replays current [151646,8987,151647,151648,151670,151670], giving width1377. This is conditioning, zero generation. The y1-position query C at1376 predicts x2; historical keys are[1362,1371). Earlier x1 at1375 and all other query positions remain native. No cache split or changed phase.

At each of28 text layers and each of16 query heads, compute from that arm's own incoming Q/K/V, original RoPE and native mask:

- H: historical weighted values using the native full denominator.
- R: nonhistorical weighted values using the same denominator.
- O_R: nonhistorical weighted values normalized over readable nonhistorical keys alone.

Native output is H+R; contribution removal is R; redistribution with H retained is H+O_R; full current-query history mask is O_R. Each actuation replaces only target2 C's merged16×128 pre-o_proj input, casting the declared FP64 result to the model's FP32 dtype once. Q/K/V are not clamped or replaced. Subsequent residual/MLP/query computation remains adaptive. These expressions are locally exact in real arithmetic; whole-model effects need not add. No cross-arm native Q/K/V may be substituted into a treatment.

Observe post-q_norm Q, post-k_norm K, projected V and actual rotary/mask inputs. Reconstruct the model's FP32 RoPE operations before promoting Q/K/V for FP64 score/reduction arithmetic, preserving16-to8 GQA grouping and scale1/sqrt(128), dropout0. Save selective target tensors for independent cold reconstruction. The actual pre-o_proj native SDPA output must match its native or masked reconstruction before any selected output write. Observe post-write input at the actual o_proj consumer separately from the actuator.

Compute separate FP64 logZ_H/logZ_R, full logZ=logaddexp(logZ_H,logZ_R), and positive representable remaining mass=exp(logZ_R−logZ). Compute O_R directly via softmax on readable nonhistorical scores and R=remaining_mass·O_R. Never divide by rounded1−m, clip/floor mass, change temperature or switch model precision. Nonfinite inputs/outputs/logZ or zero remaining mass stop technically. Record masses descriptively across all heads/layers; do not select a subset.

## Six calls and technical gate

| Order | Cell | Actual attention mask | Selected pre-o_proj operation |
|---|---|---|---|
|1|native_anchor|native|observe only|
|2|current_mask_anchor|only C→historical keys blocked|observe only|
|3|identity_reconstruction|native|H+R|
|4|full_mask_bridge|native|O_R|
|5|remove_H|native|R|
|6|redistribute_R|native|H+O_R|

Calls1–4 must pass before either partial. Native and current-mask anchors reproduce the accepted all-four full vectors and exact saved inputs. Identity matches fresh and saved native; bridge matches fresh and saved current-only. Maximum absolute full-vector tolerance2e-4 applies to all four requests. No endpoint reference exists for calls5–6. Both partials execute after qualification regardless of scientific signs.

Per layer, reconstruct the actual pre-write head output from same-call inputs. Freeze max-absolute error ≤5e-5×max(1,max|actual|,max|oracle|), using the accepted observer's scale rule. This is a local arithmetic gate, not proof of complete-model parity; the full-vector gates remain mandatory. Selected post-write input must equal the declared FP64 formula cast to FP32 exactly. All unselected pre-o_proj inputs must be unchanged by the writer. Earlier x1/header/history, target prehistory and companions remain equal to the same-source native within2e-4 (retain exact hashes when equal); all companion full vectors stay native. Actual masks and all28 layer occurrences, positions, cache/no-split, media and token inputs must be recorded. Target source trace applies only to native/identity; original companions retain trace checks everywhere. Same-source gates never invent a target trace for a counterfactual.

Before model load, exercise the actual writer, observer and serialized-reader boundary with an asymmetric CPU SDPA fixture, including wrong query/target/history/GQA/mask/RoPE/formula, unrelated output changes, missing/duplicate capture, reference alias and next-cell ordering mutations. Test normal and forced-exception hook removal, no stale same-call capture reuse, and common-device serialization comparisons. Reuse maintained operations and existing dtype/device fixes; do not edit accepted producers/dependencies or build a framework. Freeze producer/direct-source captures, exact commands and shape/artifact estimate. If preflight cannot meet this fixed route, return the concrete conflict before model work.

Persist all raw vectors/inputs, selective tensors, consumed masks/positions, pre/post head outputs/complement proof and terminal/cost records before reduction. Separate-process CPU readback rehashes source/captures, reconstructs source inputs and each local transform, checks identity/bridge/complements/counts, and computes full-vocabulary metrics. One model job, no automatic retry or repair. Any technical failure stops immediately; failed scientific predictions are valid outcomes, not reasons to stop the two fixed partials or add cells.

## Frozen scientific endpoint

FP64 full152670-way softmax, no masking, fitted coefficient or temperature. B=TV(fresh native_anchor,current_mask_anchor). Let r_H=TV(remove_H,current_mask_anchor)/B and r_R=TV(redistribute_R,current_mask_anchor)/B. Also report both partial-to-native TVs and the identity/bridge errors.

- Primary historical-contribution selectivity: B>1e-6, r_H<0.5, r_R−r_H>0.1.
- Symmetric redistribution comparator: B>1e-6, r_R<0.5, r_H−r_R>0.1.
- B≤1e-6 is uninformative. Within1e-6 of a ratio0.5 or signed difference0.1 decision boundary is numerical HOLD; otherwise neither rule means mixed_or_neither. All distances and both partials remain visible.

Fixed x2 IDs151703/152206/152669 and global winner/runner are descriptive only. This outcome concerns loss of H versus redistribution at one replayed x2 endpoint. Removing H also changes magnitude; a positive does not prove semantic content over attenuation, coordinate copying, a unique head/circuit, a mediation percentage, image-specific retrieval, natural recurrence or physical recovery. R includes prompt/image/current nonhistory positions; F ownership remains HOLD. Stop this unit after candidate and lead acceptance; no automatic head/layer/query, magnitude-control, token or image scan.

## Resources and ownership

Exactly6 model/6 vision/0 generated/0 reused, at most one model attempt, zero non-model CUDA diagnostics. User removed elapsed-time/GPU-hour ceilings; count/technical/ownership stops remain. Forecast55.93–83.90 parent seconds, same source pixels24,502,272, full width1377; selective evidence approximately2.49GB with allowance, within3GiB planning envelope. Fresh preflight must confirm this; no saving full attention matrices. Report actual parent-monotonic outer time including setup, failed-call charge, peak GPU/RSS and final artifact bytes. Prior sequence0.9083011121599776 GPUh.

Worker owns only the new maintained producer package, this unit's candidate/supporting runtime artifacts, and the exact new raw root. Lead owns protocol/admission/state/synthesis and acceptance. No Git/index, peer/shared-runtime, accepted-file mutation or model/effort changes. A stable self-contained candidate or decision-bearing failure returns directly to the paired lead, with no successor.

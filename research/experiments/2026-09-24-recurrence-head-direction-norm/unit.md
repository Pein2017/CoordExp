# Head-output direction versus norm at the fixed current coordinate query

Lead-frozen finite probe, 2026-09-24. From the accepted native N and historical-contribution-removal R anchors, does exchanging per-head direction and norm make the same replayed x2 distribution follow direction in both controls, rather than norm, under fresh source/reference/consumer gates?

## Predecessor and decision

The [accepted decomposition](../2026-09-24-recurrence-current-read-decomposition/results.md) gives removal-to-mask ratio0.216373 versus0.903136 for redistribution with H retained. It favors loss of historical weighted contribution over redistribution for this endpoint, but removing H changes direction and magnitude. N=H+R is the native head output and R retains the native denominator. R and the full-mask O_R share a per-head direction at identical incoming tensors while differing in norm; their qualified full-model outputs motivate the direction-retention primary. The strongest competing explanation is a per-head gain response, not semantic content. The old [cache-amplitude](../2026-09-12-parallel-owner-research/instance-state/amplitude-control/results.md) and [output-readout direction](../2026-09-19-readout-direction-control/results.md) counterexamples discourage a semantic-specificity inference, but use different interventions/populations.

[CPU feasibility](../2026-09-24-recurrence-current-read-decomposition/supporting/direction-norm-feasibility.md) and its [lead acceptance](../2026-09-24-recurrence-current-read-decomposition/supporting/lead-direction-norm-cpu-acceptance-v1.json) checked all2688 saved heads without exclusions. Future adaptive tensors still require online qualification. No prior G/D output exists.

## Fixed source and two controls

Original refined-03 four-request batch, target2 train351017, untied step2444 FP32 SDPA, original images/checkpoint/companions/prompts/positions. One native A history raw[:9], physical[1362,1371). Replay current six IDs [151646,8987,151647,151648,151670,151670], giving width1377. Query y1 at physical1376 predicts x2; earlier x1 is1375. All six calls have original native masks, native three-axis rotary and no cache split. Zero generation or token selection.

For every one of28 layers and16 query heads, use that arm's own incoming Q/K/V, accepted FP32 RoPE then FP64 attention partition, N=H+R and R. Let n=||N||2 and r=||R||2 over that head's128 dimensions. Define:

- G=(r/n)N: native direction, R norm.
- D=(n/r)R: R direction, native norm.

These are analytical per-head gains, not fitted or selected coefficients. Ratios above1 are amplification. No global norm, epsilon, clipping, floors, alternate norm placement, donor mixing, changed phase, anchor-QKV clamp or head/layer selection. Require finite strictly positive n/r, finite gains, positive representable remaining attention mass, finite/nonzero FP32 controls; otherwise technical stop. Compute in FP64 and cast the selected output once to FP32. Later-layer Q/K/V, residual and MLP computation remain live: these are local crossed formulas, not a matched whole-model mediation factorial. Head-output norms are matched within each local calculation, not across already diverged arms. Perturbation-delta norms and post-o_proj/residual norms are not matched.

Only target2 query1376 merged16x128 pre-o_proj input may be written. Observe actual incoming Q/K/V, rotary, mask, pre-write SDPA headout and separate post-write o_proj consumer. Every other position and companion stays untouched; earlier header/x1/history states and companion vectors retain native gates.

## Finite ordered cells and qualification

| Order | Cell | Selected operation | Reference |
| --- | --- | --- | --- |
|1|native_anchor|Observe only|Accepted native_anchor|
|2|remove_anchor|Write R|Accepted remove_H|
|3|native_identity|Write N|Fresh and accepted native|
|4|remove_identity|Independent write R|Fresh and accepted remove_H|
|5|gain_control|Write G|No prior endpoint|
|6|direction_control|Write D|No prior endpoint|

R is not the prior current_mask_anchor. Calls1–4 must all qualify before5–6. All-four full-vector maximum absolute tolerance remains2e-4, with exact saved source inputs. Each pre-write actual native SDPA headout matches same-call N within5e-5*max(1,maxabs(actual),maxabs(oracle)). Runtime selected consumer equals the declared FP64 formula cast once to FP32 exactly; unselected actuator complement equality is exact. State/full-companion error remains at most2e-4, retaining exact hashes when equal. No counterfactual target source-trace claim; original companions are checked everywhere and target only in native/native_identity.

For each G/D head after FP32 cast, relative norm error to its declared r/n target must be at most2e-6, and max absolute difference of normalized directions from N/n or R/r must be at most2e-6. Record every norm/gain/error, including amplification. Fail closed on zero/nonfinite quantities; no numerical fallback. CPU cold reconstruction follows the already qualified CPU mask normalization and cross-backend rules: FP64 error/scale discrepancies at most1e-12*max(1,recorded_scale,recomputed_scale); mass differences at most1e-12 with finite positive remaining mass; FP32 selected-formula error at most2*eps32*max(1,maxabs(expected),maxabs(saved)). These allowances never replace the exact runtime consumer/complement checks or all-four vector gates.

Before model work, freeze current producer/direct-import captures, exact commands, original full-batch preparation, six aliases and a shape/artifact estimate. Actual new caller/writer/serialized-reader CPU checks must reject wrong arm/order/reference, global gain, swapped G/D, zero/nonfinite inputs, wrong target/query/history/GQA/phase, other-output changes, stale captures, wrong mask/source/companion/position and corrupted norm/direction evidence. Exercise normal and forced-exception restoration. Do not run an old synthetic helper instead of the new actual caller.

One explicitly bounded non-model CUDA mixed-device qualification is authorized before model load: reconstructed inputs/mask truly CUDA via a device-bearing ConfigOnlyRope, saved Q/K/V payload CPU, all6 actual serialized paths. Freeze its code/command before invocation and record its outer charge. No language model or vision forward in that fixture. Existing production/cold RED evidence suffices; do not rerun RED. Any unexpected fixture failure returns immediately; no repeat. Put fixture stdout under output storage, not research.

One fresh six-call model job after qualification; save complete raw/input/consumer evidence before reduction. Separate-process CPU cold readback must reconstruct source, all168 local transforms, 6 vectors, identities and finite counts. Execute both controls regardless of scientific sign after technical gates; technical failure stops immediately, with no retry or repair permission.

## Frozen scientific decision

Use full152670-way FP64 softmax/TV, no coordinate normalization or temperature. B=TV(fresh native_anchor,remove_anchor). For X in{G,D}, d_N(X)=TV(X,native_anchor)/B and d_R(X)=TV(X,remove_anchor)/B.

Primary direction-following pair requires B>1e-6 and all four strict clauses:
1. d_N(G)<0.5; d_R(G)-d_N(G)>0.1.
2. d_R(D)<0.5; d_N(D)-d_R(D)>0.1.

Symmetric norm-following comparator requires B>1e-6 and all four reversed clauses:
1. d_R(G)<0.5; d_N(G)-d_R(G)>0.1.
2. d_N(D)<0.5; d_R(D)-d_N(D)>0.1.

B<=1e-6 is uninformative. Any tested normalized distance within1e-6 of0.5 or either signed difference within1e-6 of0.1 gives numerical HOLD. Otherwise classify direction_following_pair, norm_following_pair, or mixed_or_other. Retain both controls and all distances, including TV(G,D), identity errors and per-control clause results. Fixed tokens151703/152206/152669 probabilities/ranks/winners are descriptive, never rescue a failed TV criterion.

A direction pair weakens a simple local head-output-norm account; a norm pair weakens the proposed direction-retention account. Mixed results close this discriminator without a shared law. None identifies semantic coordinate content, unique attention heads/layers, physical F identity, a natural mediation share, complete-row recovery or a generic recurrence cure. No gain/dose/norm-placement/head/layer scan follows automatically.

## Resources, ownership and stop

Lead01a0c831-b332-7e12-b931-6ebe2359c99f gpt-6-astra/xhigh; worker01a0ce4b-9b55-7392-8a25-6a76f9e12c3a gpt-6-sol/xhigh, unchanged. User permits autonomous research and removed elapsed-time ceilings; all charges remain explicit. Six model/six vision/zero generated/zero reused; at most one non-model CUDA fixture. Same width1377, four requests,24502272 pixel elements. Twice the measured predecessor is101.342369 outer seconds planning only; measured raw1,996,222,285 bytes gives25% allowance2,495,277,857 below3GiB. Fresh qualification measures real peaks. Prior accepted sequence0.9321684536531304GPUh; include fixture/startup/failures in charge. No elapsed or sequence-hour ceiling is introduced.

Worker owns only the new recurrence_head_direction_norm probe, this unit's supporting/candidate files and its bound output root. Accepted predecessors, lead protocol/admission/state, shared routers/catalog, peers and Git/index remain read-only. Ordinary maintained imports only. New source identity is captured separately; no executing retained source captures. Direct self-contained candidate or failure to the paired lead, then stop. Only lead accepts; no self-admitted successor.

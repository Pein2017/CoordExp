# Does older-record K/V influence survive fixed current-header attention states?

2026-09-24. Lead01a0c831-b332-7e12-b931-6ebe2359c99f;923-worker01a0ce4b-9b55-7392-8a25-6a76f9e12c3a, gpt-6-sol/xhigh. This new finite contract uses the user's autonomous research authorization and removal of elapsed-time ceilings. Lead owns scientific interpretation and acceptance; worker owns implementation and evidence production. It does not reopen any closed predecessor result.

## Question and predecessor boundary

From the accepted AF/FF common-header anchors, does fixing current-header Q/K/V to each base's native tensors preserve the older-record joint K/V endpoint and its component pattern, under exact source, cache, identity-clamp and actual-consumer gates?

The [accepted older split](../2026-09-24-recurrence-older-kv-split/results.md) rejects shared V-only and K-only selectivity. AF component-to-joint ratios are0.144726/0.255575; FF ratios0.976821/0.986966, with joint TV0.530136 but single-component-to-native TVs0.055732/0.073177. One candidate explanation is jointly usable A-origin address/content: removing either component disrupts it, adding either alone cannot restore it. The strongest alternative here is amplification through adaptive current-header attention states. Residual/MLP nonlinearities remain another explanation even if the proposed clamp retains the effect.

The [CPU acceptance](../2026-09-24-recurrence-older-kv-split/lead-header-clamp-cpu-acceptance-v1.json) verifies the finite route and missing current-header tensors, not its execution. The old [phase-clamp result](../2026-09-22-recurrence-key-phase/results.md) concerns another state and supplies an operation pattern only. Do not import its scientific direction, six-token tensors or launch settings. No population, physical-owner, free-row, onset/exit, literal-copying or natural-mediation claim is at issue. F/F2 physical identity remains HOLD and the two-record shared prediction remains NONPASS.

## Fixed source, historical factorial and actuator

Original refined-03 four requests in original order; target2 train351017; untied step2444 FP32 SDPA; original images, prompt/media, positions, native masks and companions. All source/model/input hashes and accepted references are bound in lead-admission-v1.json. AF versus FF differs only in older target raw5/6/7. Physical older[1362,1371), latest[1371,1380), current header[1380,1384); three-axis rotary starts387/396/405. The common naturally emitted header[151646,8987,151647,151648] is replayed conditioning, not free generation. The next distribution is x1. AF target has original trace parity; FF target uses its accepted written-history vector, with original companion traces.

For each base, keep latest contextual K/V fixed and retain all four older corners: native, joint K+V from the other base, V-only, K-only. Patch all nine older positions at all28 layers exactly as the accepted producer; post-RoPE keys stay at their original phase. Different bases have different latest contextual caches and different native current-header captures.

Capture target current-header Q from q_norm output[4,4,16,128], K from k_norm output[4,4,8,128], V from v_proj output[4,4,1024] at each native cached anchor before any historical edit. Target-only saved shapes are[4,16,128], [4,8,128], [4,1024], FP32. Fresh AF/FF native captures are required: saved older/latest cache tensors are not a substitute.

Clamped cells replace target2's normalized pre-RoPE Q/K and projected V at ALL four current-header positions and ALL28 layers with their OWN base's native capture. Use all three axes together; do not clamp Q alone. Keep residual stream, MLP, o_proj and final head computations live. Actuate before observing the same module output and before rotary/cache consumption. No opposite-base capture, phase edit, current token change, layer/head selection or companion edit. Restore hooks and historical cache in finally under inference_mode, including exceptions.

At fixed Q and current-header K/V, one attention read has o(K,V)=softmax(QK^T+mask)V. Its older K-by-V cross difference is the changed older-slot weights multiplied by the changed older-slot values. This identity motivates a possible direct joint effect; it does not identify its contribution to final logits, which remain nonlinear through residual/MLP computation.

## Exact22-call order and qualification barriers

Calls1–14 freshly reproduce the accepted older-split cells in their original order: full_AF,full_FF,prefill_AF,prefill_FF,anchor_AF,anchor_FF,sham_AF,sham_FF,joint_AF_from_FF,joint_FF_from_AF,V_AF_from_FF,V_FF_from_AF,K_AF_from_FF,K_FF_from_AF. Native anchors additionally capture current-header Q/K/V. The first four calls include the original images; all suffix calls are image-free.

Every adaptive cell's all-four vector must reproduce its corresponding accepted reference within2e-4 maximum absolute logits. Full references also match the original saved full-prefix references; source traces, cached/full parity, explicit older-write shams and complementary joint references retain their original meaning. Calls1–10 qualify before component calls11–14, preserving predecessor qualification. All14 must qualify before call15.

Calls15–16 separately execute clamp_native_AF and clamp_native_FF, with no older edit. Each must match its fresh same-base native anchor for ALL four vocabulary vectors within2e-4, and verify actual capture consumption. Both must pass before any clamped treatment.

Calls17–22: clamp_joint_AF_from_FF,clamp_joint_FF_from_AF,clamp_V_AF_from_FF,clamp_V_FF_from_AF,clamp_K_AF_from_FF,clamp_K_FF_from_AF. All six run regardless of a technically valid scientific nonpass. No reuse, extra diagnostic forward, retry or automatic repair. Total22model/4vision/0generated, one GPU job. A technical gate failure stops dependent cells and returns evidence to lead without changing tolerances.

## Frozen scientific decision

Use FP64 softmax over the entire152670 vocabulary, without masking, fitting or temperature. TV is half the L1 probability difference. Let P_b and J_b be the fresh adaptive native and joint distributions for base b; D_b=TV(P_b,J_b). Qualification reproduces the prior endpoints but all new metrics use fresh vectors. D_b<=1e-6 makes that comparison scientifically uninformative, not a technical failure.

Let C_N,C_J,C_V,C_K denote clamped native, joint, V-only and K-only for the same base. Define:

- endpoint discrepancy e_b=TV(C_J,J_b)/D_b;
- remaining joint displacement t_b=TV(C_J,C_N)/D_b.

**Primary: bidirectional endpoint retention**, requiring e_AF<0.5 AND e_FF<0.5. This says each clamped joint result remains closer to its original joint than half the original native-to-joint separation. **Comparator: bidirectional displacement collapse**, requiring t_AF<0.2 AND t_FF<0.2. Report per-base retention/collapse/changed and both continuous values. If neither shared conjunction passes, report mixed_or_changed, retaining each base; failure of retention alone is not proof of collapse or a feedback circuit. Any tested cutoff within1e-6 is numerical HOLD for that predicate. These operational bands are not confidence intervals or natural mediation percentages.

**Secondary, declared before new results:** determine whether the remove/add component pattern survives too. Let D_C=TV(C_N,C_J), r_V^C=TV(C_V,C_J)/D_C, r_K^C=TV(C_K,C_J)/D_C, n_V^C=TV(C_V,C_N)/D_C and n_K^C=TV(C_K,C_N)/D_C. If D_C<=1e-6 these ratios are undefined and the pattern is unresolved. Otherwise the strict pattern requires AF r_V^C<0.5 AND r_K^C<0.5; FF r_V^C>0.5 AND r_K^C>0.5 AND n_V^C<0.2 AND n_K^C<0.2. Use the same1e-6 guard. Call it retained component pattern only when the shared primary also passes. Report its raw predicates even when that gate fails; do not use it to rescue the primary.

For every clamped corner preserve TV to same-base native, joint and corresponding adaptive corner, all denominators, global winner/runner/gap, and fixed z(151671)-z(151670) plus both tokens' full-vocabulary P/rank. These secondary outputs are descriptive and cannot rescue a failed distributional decision. Do not rerun the previous shared V-selectivity prediction as the new primary.

Retention would show current-header Q/K/V adaptation unnecessary for this bounded endpoint transfer under the clamp; it would not isolate attention from downstream nonlinearities. Collapse would support dependence on that adaptation under a joint intervention, not identify a unique layer or a natural causal share. Preserved joint endpoints with a changed component pattern limits the conjunctive account separately. Stop this package after the finite readout; no contingent layer/head/position scan.

## Actual execution and persistence gates

Reuse ordinary maintained source/cache/observer operations, minimally adapting the old clamp pattern for this actual four-token route. Preserve all accepted producers and raw bytes. CPU preparation binds original four requests, AF/FF inputs, checkpoint independent input/output deltas, maintained-loader crosswalk, config, exact call order, commands, source captures, masks/rotary/physical slots, and artifact/resource forecast before model load.

The actual CPU caller, clamp hooks, cache scope and serialized reader must reject wrong base donor, target, layer count, dropped header position, Q-only actuation, omitted older component, changed companion, wrong source/mask/phase/cache slot, and changed cell order/container. Exercise identity plus both historical donor directions and K/V components, normal exit and forced-body exception with hook removal and cache crop/restore. The smallest real all-four identity clamp is the production gate before clamped treatments; mocks do not substitute for it.

At every relevant actual attention call, verify all28 layers: intended older K and V separately, untouched latest/prehistory/companion cache; top input and current native rotary/mask/cache slots; observed post-hook Q/K/V exactly matching the same-base capture in clamped cells; companions unmodified. At o_proj entry, require appended current K to equal the rotary-transformed captured K within2e-5 max absolute error, appended V exactly equal captured V, and companion suffix K/V to match the corresponding anchor. For adaptive cells, capture and observe their actual current tensors without clamping.

Reconstruct the actual target pre-o_proj head output from observed post-RoPE Q, actual full historical/current K/V, native bool mask, model scaling and correct16-query/8-KV head mapping in FP64. For each layer, require maximum absolute reconstruction error <=5e-5*max(1,maxabs(actual_headout),maxabs(FP64_reconstruction)). Freeze this scale-aware numerical gate now; save absolute error, scale and bound. Exact Q/K/V consumer hashes, source masks and donor checks remain mandatory even if this numerical gate passes. An out-of-bound result is technical-invalid, not permission to relax the gate. No alternate attention implementation should replace native SDPA.

Save each actual and reconstructed target head output, native target captures with base identity, actual post-hook/rotary/appended-cache hashes, selected older/latest donor blocks, whole historical cache/restoration and companion receipts, raw all-four vectors and per-cell inputs before scientific reduction. Full-cache dumps are unnecessary. Keep runtime numerical reconstruction distinct from cold recomputation: separate-process CPU readback can compare saved headouts/consumer hashes and reconstruct source/donor/capture binding without claiming an independent full SDPA replay when its full inputs were not retained. It must independently recompute all22 vector parities and scientific metrics from raw tensors, exact cell/order/counts, identity and restoration records.

## Bounds, ownership and return

Shapes remain full[4,1384], prefill[4,1380], suffix[4,4], bool mask[4,1,4,1384];24,502,272 media elements; two historical FP32 caches plus3,670,016 target capture bytes. Accepted peak allocated/reserved12,476,571,136/13,254,000,640B and RSS11,093,772KiB are measured references. Planning outer time365.305860311s, no elapsed ceiling. Raw plan256MiB; the new observer may add selective evidence, so preflight must project actual saved tensors and report a material capacity conflict before launch. Charge actual parent-monotonic outer time including imports/setup, internal time separately, all calls/failures, RSS/VRAM, artifacts and terminal PID. Prior accepted sequence0.5924270202297751GPUh is accounting, not a time cap. Shared stress occupancy is not a reason to wait for idle GPUs.

Worker owns new `probes/training_set_completion/recurrence_older_kv_header_clamp/`, this unit's supporting/candidate files and the new output root in admission. Lead owns unit/admission/state/acceptance/router. No Git/index, accepted predecessor, peer/shared runtime, model-setting or lead-state edit. Return a self-contained candidate or decision-bearing failure directly to lead through worker_turn.py. No self-acceptance or successor.

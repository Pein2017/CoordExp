# Query adaptation versus current-header K/V adaptation

2026-09-24. Lead01a0c831-b332-7e12-b931-6ebe2359c99f; existing923-worker01a0ce4b-9b55-7392-8a25-6a76f9e12c3a, gpt-6-sol/xhigh unchanged. User-authorized autonomous finite follow-up; no elapsed-time ceiling. Lead owns science and acceptance, worker implementation/evidence. No self-admitted successor.

## Question and reason to run

From the accepted AF/FF anchors, does fixing current-header Q alone collapse the older-record joint K/V effect while fixing current-header K/V alone retains it, in BOTH directions under matched native identity and actual-consumer gates?

The [accepted whole-header clamp](../2026-09-24-recurrence-older-kv-header-clamp/results.md) leaves only0.038752/0.102511 of the adaptive native-to-joint displacement. This supports dependence on current-header Q/K/V adaptation under the joint clamp, but does not distinguish the two paths. The leading prediction here is query adaptation: history changes what current positions query at later layers. The symmetric alternative is current-token K/V adaptation: history changes the information current positions make available to themselves and other causal current positions. Both may be required, or either adaptive path may suffice. These are within-forward computations, not an already identified multi-token repetition mechanism.

This is a global path distinction, not a layer/head search. The older-record K-only/V-only factorial is closed and is not expanded here. Use only its joint older replacement in each direction. Residual/MLP adaptation remains live. F/F2 physical HOLD and prior shared scientific nonpasses remain unchanged.

## Fixed source and two new actuator modes

Inherit the exact source/model/geometry/cache split and technical semantics of the [predecessor contract](../2026-09-24-recurrence-older-kv-header-clamp/unit.md), bound separately by the new admission: refined-03 four original requests, target2 train351017, untied step2444 FP32 SDPA; same image/media/prompt/companions; AF/FF historical difference only older raw5/6/7; older physical[1362,1371), latest[1371,1380), current header[1380,1384). Same three-axis positions and native masks. The naturally observed four-token header is replayed; zero generated tokens. AF target uses source trace, FF target accepted counterfactual vector; companion traces stay original.

Latest contextual cache remains fixed within each base. Older joint treatment replaces all nine older K AND V at all28 layers with the other base's native donor; no rephasing. Keep native older-write shams. Fresh native cached anchors capture all28/all4 current-header Q/K/V and must reproduce the accepted same-base captures exactly as well as accepted all-four output vectors.

Three explicit current-header hook modes:

- `all`: replace Q,K,V with own-base native capture; already accepted, fresh bridge only.
- `Q_only`: replace only normalized pre-RoPE Q at q_norm output. Current K and V are computed normally and may adapt; they must not be changed by any hook.
- `KV_only`: replace normalized pre-RoPE K at k_norm and projected V at v_proj. Current Q is computed normally and may adapt; it must not be changed by any hook.

Every mode covers target2, ALL four current-header positions, ALL28 layers and uses the same base's native capture. Preserve actual native dtype/device, phase, physical slots and companions. Do not clamp residual, MLP, o_proj, final head or any historical component beyond the prescribed older joint edit. This new contract explicitly admits partial clamps; the predecessor's prohibition of Q-only remains in force for that closed unit.

## Twenty-two calls and qualification

1–10: full_AF,full_FF,prefill_AF,prefill_FF,anchor_AF,anchor_FF,sham_AF,sham_FF,joint_AF_from_FF,joint_FF_from_AF. These reproduce the accepted corresponding all-four vectors/inputs and capture native current Q/K/V. Only calls1–4 encode the original images.

11–14: all_native_AF,all_native_FF,all_joint_AF_from_FF,all_joint_FF_from_AF. Reproduce the accepted whole-clamp native and joint all-four vectors; these are qualification bridges, not new scientific evidence. Both native-all identities precede both all-joint references.

15–18: Q_native_AF,Q_native_FF,KV_native_AF,KV_native_FF. Separately execute the two new actuator modes without an older edit. All four must reproduce the same-base native all-four vector before any new treatment.

19–22: Q_joint_AF_from_FF,Q_joint_FF_from_AF,KV_joint_AF_from_FF,KV_joint_FF_from_AF. Run all four after qualification regardless of scientific nonpass. No component is inferred from another. Exactly22model/4vision/0generated, one GPU job; no reuse, retry or extra model diagnostic. Technical failure stops dependent calls and returns to lead; scientific nonpass does not skip a later fixed treatment.

## Frozen full-vocabulary decision

Use FP64 softmax over all152670 vocabulary entries and TV=half L1, no fitting/temperature/coordinate restriction. For base b, let N_b,J_b be fresh adaptive native/joint; D_b=TV(N_b,J_b). For X in {Q,KV}, let N_X and J_X be its fresh partial-clamp native and joint. Define t_X=TV(J_X,N_X)/D_b and e_X=TV(J_X,J_b)/D_b. Always retain both continuous values and all raw TVs. D_b<=1e-6 leaves that base uninformative, not technical-invalid.

**Primary query-adaptation signature:** in BOTH bases, t_Q<0.2 AND e_KV<0.5. **Symmetric current-K/V-adaptation comparator:** in BOTH bases, t_KV<0.2 AND e_Q<0.5. These test loss under one clamp plus retention under its complement, not merely two sensitivities.

For each base also distinguish `both_collapse` (t_Q<0.2 AND t_KV<0.2), `both_retain` (e_Q<0.5 AND e_KV<0.5), and mixed/changed when neither path signature nor these categories applies. A shared label requires the same predicate in both bases; do not average opposite directions. Any decision cutoff within1e-6 is numerical HOLD. Thresholds are operational bands, not significance tests or natural mediation shares. The already accepted all-clamp collapse is a reproduced reference, not an alternative success criterion for a failed new primary.

Report descriptive winner/runner/gap, z(151671)-z(151670), full-vocabulary P/rank for both fixed tokens, and partial-joint TV to the same-base all-clamped joint. These do not rescue a failed distributional decision. No new coordinate/class/owner or complete-row claim.

A primary pass would favor adaptive queries over adaptive current-token K/V as the necessary route for this conditional older-joint effect under the matched clamps. The comparator reverses that local preference. Both-collapse leaves coupled dependence; both-retain with known all-collapse is consistent with alternative adaptive routes. None establishes a unique circuit or direct image-versus-history attention destination. Stop this global decomposition after the finite package; do not subdivide layers, heads or header positions automatically.

## Actuator qualification and saved evidence

Reuse the maintained predecessor source preparation, older joint cache scope, native capture, actual attention observer and cold reader patterns without editing accepted source bytes. Add only a bounded explicit axis selection to a new local producer, not a generic hook framework. Before model load freeze exact commands, source/import captures, source/media/checkpoint/input identities, both native captures' saved references, finite cells, resource and storage forecasts.

CPU fixture must exercise the ACTUAL partial-clamp caller and serialized reader: all three modes, both bases, all layers/positions, normal and forced-exception hook/cache restoration. Reject wrong mode/axis mask, wrong base donor/target, missing selected axis, accidental free-axis edit, missing position, wrong historical span/component, changed companion, wrong rotary/mask/cache/input and changed cell order/container. In particular, Q_only must leave computed K/V untouched and KV_only must leave computed Q untouched; a hook merely claiming its mode is insufficient.

At every suffix attention consumer retain predecessor gates: declared older K/V, base latest/prehistory/companions, top input/native mask/rotary/cache slots, full cache crop/restoration. Record each current axis before actuation and after all hooks. A selected axis must equal own-base native capture exactly; an unselected axis must equal its actual computed pre-actuation tensor exactly; companions stay exact for every axis. Later unselected target states are allowed to differ from native.

Retain actual post-RoPE Q and appended K/V consumption checks for ALL modes: appended K versus observed pre-K rotated with actual native phase <=2e-5 max absolute, appended V exactly observed projected V. Reconstruct pre-o_proj head output from actually consumed Q/full K/V/mask/scaling in FP64; each layer maxabs error <=5e-5*max(1,maxabs(actual),maxabs(reconstruction)). These are unchanged technical bounds. All-four saved-reference, cached/full, whole-clamp bridge and partial-native identity gates remain2e-4 maximum absolute logits; companions/restoration hashes exact.

Save raw all-four vectors, per-cell inputs, actual per-axis before/after/selected-mode/capture hashes, native tensors, headouts, historical donor/complement/companion/restoration records and terminal receipt before science reduction. Separate-process CPU cold readback reconstructs source/history/header, binding and mode application, raw-vector parities and metrics, and saved headout error arithmetic. Do not claim a new full SDPA replay without its full saved inputs. Save complete selective evidence without full-cache dumps.

## Resources, ownership and stop

Same full[4,1384], prefill[4,1380], suffix[4,4], bool mask[4,1,4,1384],24502272 media elements, two historical caches and two native target captures. Same22/4 call shape as the accepted predecessor: measured186.280167371 outer seconds,119878951B artifacts, peak allocated/reserved12476571136/13266583552B, RSS11076928KiB. Planning2x time372.560334742s and2x payload239757902B fit a256MiB envelope; these are forecasts, not elapsed caps. New before/after hash evidence must be included in preflight sizing. Report a material capacity conflict before expansion. Charge parent-monotonic outer time including setup, internal separately, exact calls/failures/RSS/VRAM/artifacts/terminal PID. Prior sequence0.6441715111661737GPUh is accounting only.

Worker owns new `probes/training_set_completion/recurrence_header_feedback_components/`, unit supporting/candidate records and new output root in admission. Lead owns unit/admission/state/acceptance/router. No Git/index, model settings, accepted producer/raw, peer/shared runtime or lead-state edits. Return candidate or decision-bearing failure directly to the paired lead via worker_turn.py, preserving model/effort. No self-acceptance or successor.

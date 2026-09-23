# Matched incoming state and coherent downstream query-phase transfer

2026-09-22. User renews autonomous research. Root owns science/acceptance;
existing gpt-5.6-luna/max worker executes. No peer codebook surfaces are in scope.

## Frozen decision and predecessor

From the original val7511 row42 x2 query with layer0 whole-head output fixed to
actual row89, does advancing ONLY layers1..27 current-query/self-key rotary
phase to row89 produce the row89 global winner999, under qualified native and
held-state/native-phase controls?

Accepted reverse transfer retained38 (d38-9991.901579→1.780115). Thus the late
first-layer vector alone was insufficient. Its incoming state and downstream
phase were not jointly matched. Earlier global position/history interventions
also changed first-layer processing. This unit matches the actual layer1 input
and downstream current-position phase while retaining the earlier visible
history, separating that remaining sufficiency claim.

Root's advance primary prediction is retained global38. The strongest alternative
is999: the matched incoming state plus query position suffices without the47
extra repeated rows. A third winner falsifies both categorical predictions and
still fails to reproduce the native late decision. Full vocabulary argmax is
primary; movement of z38-z999 is secondary and cannot rescue a failed prediction.
If not999 with qualified matching, earlier visible historical memory is not
interchangeable with late memory even after both controls. This does not isolate
memory length, anchor phases, contextual K/V content, a particular layer or owner,
or predict natural onset/exit time. Numerical999 is not physical recovery.

## Source mapping and three cells

Mature untied+axis step2444, original FP32 SDPA native batch4,width2127,target2.
Recipient row42 x2: input1703, predicted raw384, input token152241, native38.
Donor row89 x2: input2126, predicted raw807, input token152241, native999.
Use saved ACTUAL groups/native.pt head_output[0/1] [16,128], query_cos/sin[0/1]
[128]; do not infer MRoPE phases from physical indices. Native trajectory supplies
layer1 residual_input[1][15/62] [2048] for the exact incoming-state gate.
Selection.json binds original oracle, acceptances, hashes and source indices.

Every cell uses original full inputs, fresh DynamicCache and selected LM indices
[1703,2126]. Save logits [4,2,152670]; primary is [2,0,:]. The late logits in
TREATED cells are not a scientific endpoint because the earlier intervention
can affect future computation. Native late logits are an endpoint/parity control.

1. native: no layer0 or phase replacement; capture both selected rows.
2. held_native_phase: layer0 o_proj INPUT [2,1703,:] replaced by actual late
   head_output[1]. In layers1..27 re-create only current query Q and current
   self-key K using LIVE normalized pre-Q/pre-K and original EARLY phases.
3. held_late_phase: same layer0 substitution, but layers1..27 current Q and
   self-key K use saved LATE phases. This is the sole new scientific treatment.

Co-rotate Q and its self-key: changing Q alone would also change its self-score.
The boundary is actual attention-consumer Q[2,:,1703,:] and K[2,:,1703,:], AFTER
normalization/RoPE and cache update, BEFORE SDPA. Q shape[4,16,2127,128], K/V shape
[4,8,2127,128]. Registered q_norm/k_norm outputs have layout[B,S,H,D], not[B,H,S,D].
Reconstruct from live normalized vectors in every layer, allowing downstream
adaptation; never substitute saved later-layer Q/K content. V, causal masks,
cache slots, all historical K positions<1703 and all other Q/K entries at the
same consumer remain unchanged. The original current cache tensor may stay
native: K replacement is consumer-local; no generation/cache reuse is allowed.
No position_ids, history tokens, hidden residuals or layer0 Q/K phases are edited.
Layer0 output substitution reuses the accepted actual-consumer hook helper.

## Actual-consumer and numerical qualification

Reuse installed SDPA route and existing trajectory attention_attestors. A scoped
registry wrapper may patch text-attention arguments; a separate observer at
ACTUAL torch.nn.functional.scaled_dot_product_attention must verify those consumed
arguments after GQA handling. Filter exact text module IDs; vision calls pass
through unchanged. Restore registry/function entries and hooks in finally.
Do not double-run full attention just to attest consumption. CPU use the same
wrapper/observer with real SDPA and a small nonconstant tensor, unequal B/S/H/D,
causal mask and GQA. A wrong-query patch and reversed sine sign must be rejected;
coherent correct patch must equal independently constructed SDPA output.

Per treated layer, save live pre-Q/pre-K, pre-intervention and actual-consumed
post-Q/post-K target vectors, actual phase references and consumer count1. Show
exact equality of off-target Q/K, unchanged V/mask, and observed head expansion.
Check rotation independently with FP64 real/imaginary halves (absolute2e-4), not
only the producer rotate helper. Current self-scores before/after must agree
within2e-4; query/key norms within2e-4. Phase treatment must be nonzero somewhere.
Retain all27 layers, no selected-head verdict. Hash actual strictly historical
K/V [target2,:,0:1703,:] at the SDPA boundary and require exact native equality
across cells (account explicitly for GQA head repetition). Only hashes/metrics
are saved for full historical tensors; do not dump them.

Record ACTUAL layer1 input at1703. Both held cells must match accepted native
late layer1 input<=2e-4; native must match early. This gate is needed to claim
matched incoming state. Native both-query target full vectors must match groups
capture<=2e-4 with global winners38,999. Held_native_phase early target full
vector must match accepted reverse-transfer late.pt target<=2e-4 and winner38.
Every cell's companions0/1/3 at both selected queries must match fresh native
<=2e-4. All28 actual masks/cache slots and original input identities remain native;
early query visibility ends1703, late query ends2126. No tolerance tuning.
Save raw tensors and observed numeric errors BEFORE gates raise. Native/control
failure stops remaining model calls and leaves the scientific question unanswered.

## Ownership, resource and stop

Worker owns only probes/training_set_completion/recurrence_downstream_query_phase.py
and this unit output attempt-001. Root owns selection, records and independent
readback. Import proven mechanics without altering accepted producers. The source
and newly used installed SDPA dependency, runtime/model/media identities and
protocol must be frozen before the first model call. No duplicate generic manager.
Exactly3 model/3 vision forwards, GPU4,10minutes; total tensor payload32MiB.
Native and held-native-phase are the production-shaped admission; no extra model
smoke. Execute CPU qualification then all three automatically; preserve failure
and return to root, no automatic retry. No head/layer/position/strength scan,
training, generation, peer work or user permission round. Stop after candidate.

The worker already returned a read-only mapping in a separate short turn. Root
adds COHERENT SELF-KEY rotation to avoid changing self-score, and chooses an
actual SDPA observer rather than duplicate attention evaluation. These are lead
rulings. Root retains scientific choice and independently accepts raw evidence.

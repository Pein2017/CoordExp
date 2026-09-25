# Which coordinate query carries the continuing history effect?

Lead-frozen 2026-09-24. One fixed original source prefix; six image-bearing full forwards, zero generated tokens. The user authorizes autonomous bounded continuation. `lead-admission-v1.json` owns the exact one-attempt grant; the worker returns candidate evidence and the lead accepts independently.

## One decision

At the native A-history prefix ending in current x1=0,y1=0, does blocking **the current y1-position query** from the prior record reproduce the already qualified both-coordinate-query block more strongly than blocking **the earlier x1-position query** alone?

The [accepted reciprocal y1 unit](../2026-09-24-recurrence-y1-row-routing/results.md) rejects both the shared exchange and persistence signatures. Native y1→13 yields a near-image-wide box, not restored localization; blocked y1→0 remains broad. Its saved step6 vectors compare identical complete inputs: native versus coordinate-block TV is0.973464780379 at y1=0, and0.907858494468 at y1=13. This means selected y1 alone does not account for the read-mode effect. It does not identify where the remaining influence enters.

Leading hypothesis: the current prediction query's own access to history is the stronger channel at this fixed endpoint. Strongest comparator: history-conditioned state at the earlier x1 position is the stronger channel into later attention, even when the current query retains its own historical access. The two partial masks separate these computational paths with equal nine-edge deletions per layer. Both, neither, or interacting contributions remain possible. No unique attention head, copied value, physical owner, natural mediation share or whole-row result follows. This is an endpoint discriminator, not another coordinate-token scan or a claim that masking preserves attention normalization.

## Bound source and conditioning

Original refined-03 four-request batch, target2 train351017, untied step2444 FP32 SDPA. Original checkpoint/input-delta/output-delta identities, images, prompts, companions, causal mask and three-axis positions remain bound to predecessor source receipts. Full padded width is1377, with24,502,272 pixel elements. All four source rows are active at raw offset15.

Prior record A is raw[0,9), physical[1362,1371):
`[151646,8987,151647,151648,151670,151683,152206,152669,151649]`.

The current **replayed native** prefix is raw[9,15), physical[1371,1377):
`[151646,8987,151647,151648,151670,151670]` (person header, x1=0,y1=0). These six IDs are conditioning, not newly generated tokens. The last y1-position query predicts x2. No current-token policy, free generation, cache split, phase or K/V write is admitted.

The accepted `native_A-step6` and `coordinate_flip-step6` full input JSONs are byte-identical (SHA256 `fcf4a094187ef351cc8ff8d93a180471c1ec6ad44d1f48325add7fb65e0be4b5`). Their existing all-four full vectors qualify fresh native and both-block calls respectively. Only the native target has original source-trace parity; both-block target has an accepted counterfactual vector. Companion source traces apply to every arm.

Earlier coordinate query **E** is the x1 token at physical[1375,1376), rotary400 on each axis. Current query **C** is the y1 token at physical[1376,1377), rotary401. Both read the same nine prior keys **H**=[1362,1371). At all28 text layers, the only deletable edges are target2 E→H and C→H. All headers, current-to-current edges, other target edges, companions and positions remain native. Causal order prevents C changes from reaching earlier E; masking E may change what later C receives through current-token context. Current states are otherwise adaptive.

## Fixed six calls

| Order | Cell | Deleted edges per layer | Qualification |
|---|---|---:|---|
|1|native|0|Accepted native_A-step6 all-four vector and original traces|
|2|both|18, E→H and C→H|Accepted coordinate_flip-step6 all-four vector; companion traces|
|3|native_sham|0, independent explicit identity-mask path|Fresh native vector/state parity|
|4|both_sham|18, independent explicit same-mask path|Fresh both vector/state parity|
|5|earlier_only|9, E→H|Unknown prospective partial endpoint|
|6|current_only|9, C→H|Unknown prospective partial endpoint|

Every call executes independently with the same full source input and images; zero reuse. Calls1–4 must qualify before either partial. No result-conditioned skipping of call6. The same checked mask-edit path must serve shams and partials. Normalize complement comparisons over each cell's **own** selected rectangle; never compare different rectangles' complement hashes.

## Frozen primary and comparator

Use FP64 softmax over the entire152670-way vocabulary, without coordinate restriction, fitted coefficients or temperature. Fresh anchors define B=TV(native,both). For each partial j, r_j=TV(j,both)/B. B≤1e-6 is an uninformative scientific endpoint, not a technical failure.

The **current-query primary** passes iff B>1e-6, r_current<0.5 and r_earlier−r_current>0.1. The **earlier-query comparator** reverses current/earlier in those two strict conditions. Any distance≤1e-6 from a numerical boundary is HOLD. Otherwise retain `mixed_or_neither` if neither selective rule passes, plus both continuous ratios and all TVs to native/both. A qualified failed predicate is NONPASS; a failed technical gate leaves the endpoint unanswered.

Winners, runners, probabilities/ranks and fixed x2 token logits33/536/999 are descriptive only. The full-vocabulary rule owns the decision; no categorical winner, complete box, physical A/F attribution or exact-token secondary is a success criterion. Both partials execute even if the first appears to settle a signature.

## Minimal implementation and acceptance

Reuse maintained original source preparation, full-prefix native SDPA mask construction, model loader, input hashing, attention consumers and source-capture operations. Implement a small **fixed-endpoint** caller and cold reader under the new package. Do not copy the free-row generation/policy driver or its parser-test corpus: there is no rollout in this unit. Do not edit accepted producers or execute captures.

CPU preflight must freshly reconstruct the original four-request source, exact replayed prefix, physical/rotary spans and accepted reference bindings. Freeze commands and new/directly imported source captures before model load. Exercise the actual caller and serialized consumer for all six masks; reject wrong target, either query/key endpoint, E/C swap, wrong scope/complement/current-to-current edge, companion/prefix/position drift, reference alias and cell order. Carry forward the qualified CPU/CUDA normalization for serialized comparisons. Use a focused falsification check, not a required test count or new framework.

All-four fresh/reference and sham logit gates use max absolute2e-4 and exact inputs. Native traces and all companions' chosen/top2/log-normalizer checks use2e-4; never impose the native target trace on a partial or both-block cell. Record actual top-level inputs, images, positions and native or edited mask at all28 attention entries, exact selected counts and same-rectangle complements. Prior target and companion states/vectors remain unchanged. Observe the six current-prefix states per layer: header states must stay equal; current_only's earlier x1 state must equal native, while the intervened/current downstream states may differ. No Q/K/V or residual clamp is added.

Save all-four full vectors, source inputs and actual consumer/state evidence before reduction. Separate-process CPU readback rehashes sources, reconstructs exact input/masks, recomputes reference/sham gates and FP64 TVs, validates counts and terminal outer receipt. Raw captures support replay of checks; no cold model forward. Return a stable candidate with all ratios and both scientific verdicts. No overlay or owner adjudication is needed for a single conditional x2 distribution.

## Finite resources and stop

Exactly **6 model/6 vision forwards/0 generated/0 reused**, one model attempt. Optional one non-model CUDA serialized-boundary check is allowed if the new checker needs it; record actual placement and charge, with failure returning before model work. No other diagnostic model call or retry.

The accepted same-shape54-call job cost155.09143260866404 outer seconds and728,197,475 bytes. Two-times linear six-call planning is34.465s, before any extra state recording allowance; use **50s** as a nonbinding planning estimate. A **256MiB** artifact envelope covers the doubled call-scaled payload plus small current-state captures; fresh preflight must check this. Prior measured GPU allocated/reserved peaks are9,964,165,632/10,779,361,280B and RSS11,506,048KiB. No elapsed-time or cumulative-hour ceiling. Charge all outer setup/fixtures/failures from prior accepted sequence **0.9005328468261231GPUh**.

Any source, capacity, reference or consumer failure stops and returns; preserve incomplete raw evidence and cost. No automatic repair/retry or new arm. A scientific nonpass never skips the other fixed partial. Worker `01a0ce4b-9b55-7392-8a25-6a76f9e12c3a` remains gpt-6-sol/xhigh, paired to lead `01a0c831-b332-7e12-b931-6ebe2359c99f` gpt-6-astra/xhigh. Worker owns new `recurrence_coordinate_query_routing/` code, this unit's supporting/candidate files and its named raw root; lead owns protocol/admission/state/acceptance/synthesis. No Git/index, predecessor, peer or shared-runtime write. Stop at candidate; no self-acceptance or successor.

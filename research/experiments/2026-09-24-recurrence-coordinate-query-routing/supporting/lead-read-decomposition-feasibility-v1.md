# CPU feasibility: historical contribution versus attention redistribution

Lead-owned assignment, 2026-09-24. Worker 01a0ce4b-9b55-7392-8a25-6a76f9e12c3a remains gpt-6-sol/xhigh; lead 01a0c831-b332-7e12-b931-6ebe2359c99f remains gpt-6-astra/xhigh. Cwd is /data/CoordExp/.worktrees/research-probes. This grants one CPU/source feasibility task only, not a producer implementation or model/GPU attempt.

## Accepted boundary and decision

[Lead acceptance](../lead-acceptance-v1.json), SHA256 a8f580804daab8a53783d2495df80051ec63b2a9a3e7bd57a31ecfbedc6542d1, closes the six-call query partition. Its frozen current-query relative-selectivity primary passes; earlier-only is also far from native. Read the [compact result](../results.md) and exact current unit admission/receipts rather than older conversations. All prior protocols, producers, raw bytes, candidates and acceptance remain immutable.

Fixed source: original refined-03 four-request target2 train351017, native one-row A raw[:9], identical replayed six current tokens [151646,8987,151647,151648,151670,151670], full width1377. Current y1 query at physical1376 predicts x2; selected historical keys [1362,1371), all28 text layers. Earlier x1 at1375, all header/history/companion positions and every current-to-current edge remain native. No new source, token, query, head or layer selection.

Question: can a small maintained route separate removal of the selected historical weighted-value contribution from the accompanying redistribution of attention to the remaining keys? Masking changes both. An interpretable positive must distinguish these computations; do not call mask selectivity evidence of literal copying or of an exclusive historical-value channel.

For one head at its actual incoming Q/K/V and native mask, define native probabilities p over all readable keys, H=sum(history p_i V_i), R=sum(other p_i V_i), and m=sum(history p_i). Then:

- Native output: H+R.
- Historical-contribution removal with the original denominator: R.
- Redistribution while retaining historical contribution: H+R/(1-m).
- Full selected-edge mask: R/(1-m).

This is exact local attention arithmetic when the remaining mass is positive; it is not a claim that multi-layer output effects add. Each proposed arm would use its own adaptive incoming Q/K/V at each layer, and only the selected target current-query head output would be transformed. Do not clamp queries, change historical/cache values, or silently use native-anchor probabilities in treatments. The residual/MLP and later layers remain live. Loss of H may also alter output magnitude, so even a selective result would not establish semantic-content specificity over generic attenuation. R includes image, prompt and current/history-external positions; no image-attention claim follows without another contrast.

The provisional leading prediction is that removing H without redistribution approaches the accepted current-only masked endpoint more than redistribution with H retained. The strongest alternative reverses this, or requires their combination. No threshold, pass category or model admission is frozen here. The accepted earlier-only result and the failed reciprocal token bridges must remain alongside any future positive.

## Deliverable and permitted work

Write exactly these two new supporting artifacts:

1. `read-decomposition-feasibility.md`: short feasibility conclusion, decisive missing gates, minimum call route and interpretation limits.
2. `read-decomposition-bindings.json`: exact source/accepted-vector/consumer/code bindings, dimensions and proposed cell ledger/cost estimate.

Use read-only maintained-source inspection, saved CPU tensor loading, processor/config reconstruction with load_model=False if needed, and small CPU arithmetic fixtures. No model load, model/vision forward, CUDA operation, GPU job, generation, producer implementation or existing-file edit. Do not execute captured historical code. No Git/index, peer/shared-runtime, lead state, catalog or synthesis writes.

Answer concretely:

- What needed actual tensors are saved now, and what must be captured afresh? The current six-call evidence should not be presumed to contain Q/K/V or attention probabilities.
- Can existing maintained Q/K/V/rotary/head-output observers support this full-prefix current-query operation? Check the actual SDPA/GQA scaling, masks, head arrangement, RoPE and o_proj input. Existing header-clamp reconstruction is a lead, not proof that its cached route fits this full-prefix call. Prefer one local adaptation over a new instrumentation framework.
- Specify actual-consumer and independent CPU arithmetic checks for the selected query, untouched complement, original source/media/companions, current-only's unchanged earlier x1, finally/hook restoration and failure paths. A future reconstruction identity must match native, and a reconstructed full-mask bridge must match accepted current-only all-four vectors before scientific partials. No reference is invented for the two partial outputs.
- Explain numerical conditioning and remaining-mass checks. Compute the remaining-key normalized contribution stably (rather than relying on subtraction of a rounded m from1); no silent clipping, denominator floor, altered temperature, precision switch or tolerance relaxation. CPU synthetic algebra can prove semantics, not actual model qualification.
- Propose the shortest finite ledger, likely fresh native/current-mask anchors, independent reconstruction identity, reconstruction mask bridge, then the two partials. Determine whether more qualification calls are actually necessary; justify any change. Zero generation and no extra ablations. Bind measured same-shape costs, state forecast as planning, and estimate selective evidence size without saving every full attention matrix.

Stop after these two artifacts and direct candidate return. If the operation cannot be qualified narrowly, identify the specific blocker instead of expanding scope. Lead decides any prospective metric and GPU admission later. Prior accepted sequence charge is0.9083011121599776 GPUh; this assignment adds zero GPU time. User removed elapsed-time ceilings; finite scope and technical gates remain.

# Downstream readout of fixed first-layer whole-head counterfactuals

2026-09-22. User explicitly renews autonomous research and asks root to improve
Luna-max prompting. Peer synchronization changes neither the research owner nor
the active direction. Root owns this question and acceptance; Luna-max executes.

## Frozen question and decision

From mature untied+axis step2444 at the original val7511 row89 x2 query, does
replacing ONLY layer0's whole-head attention output with the accepted old-anchor/
old-pool vector restore the previously repeated coordinate38, under qualified
native and reconstruction-sham controls?

The accepted predecessor separates anchor aging and pool growth in layer0
head-output displacement (signed projections0.566/0.434). It has not shown that
this geometric change drives the final vocabulary decision. This unit changes
the evidence surface to a local model intervention with downstream recomputation.
Root's advance primary prediction is global winner38 in cell00. The strongest
alternative is that late-layer context/relative phase preserves999 despite
undoing the first-layer change. A third winner also rejects the prediction.
Native or sham failure makes the scientific question unanswered, not negative.

Exact anchor: original val batch4, target index2, native full width2127; zero-
based physical query input2126 predicts raw action807, row89 x2. Current query
input token152241; native next token152669 (bin999), repeated token151708 (bin38).
Original image/prompt/history/positions, mature checkpoint and FP32 SDPA route
remain the accepted ones. No generation, image/checkpoint transfer or teacher
prefix substitution. The parallel codebook source/outputs/jobs remain untouched.

## Intervention and five cells

Use accepted predecessor
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-first-layer-groups/lead-checks/factorial.pt
with its lead-acceptance.json. Source tensors `cells[ab]` have shape[16,128] and
dtypefloat64. The first digit is ANCHOR age, the second REPEATED-POOL state;
0=old row42 (15 repeats), 1=new row89 (62 repeats). Current-prefix scores are
already fixed to the new state in all saved cells. Do not reconstruct or refit
these counterfactuals in the model producer.

| Run order | Cell | Layer0 o_proj INPUT at batch2,query2126 |
|---:|---|---|
|1|native|Unmodified actual16-head output|
|2|11|Saved new anchor/new pool, reconstruction sham|
|3|01|Saved old anchor/new pool, undo anchor aging only|
|4|10|Saved new anchor/old pool, undo pool growth only|
|5|00|Saved old anchor/old pool, undo both (PRIMARY)|

Real consumer tensor is[4,2127,2048]. Cast each saved cell to that tensor's actual
dtype/device and flatten row-major [head,dim] to2048. Patch only[2,2126,:] in
`text.layers[0].self_attn.o_proj`'s first positional argument. Preserve all other
arguments and all off-target tensor elements exactly. Patch BEFORE o_proj, not
its output, not a cache row and not all current-S positions. Do not select head13.
Downstream o_proj, residual, MLP, all later layers and final vocabulary head run
normally. Each cell gets a fresh empty DynamicCache and original full inputs.
Do not reuse a modified cache or batch several conditions into altered batches.

Register the patch pre-hook before a separate actual-consumer pre-hook. The
second hook must verify the consumed target equals the cast saved cell, and
all off-target values equal the pre-patch input. Native has no replacement but
still records the actual consumer. Store before/consumed target vectors, hashes,
off-target comparisons and exactly-one call counts. Remove hooks in finally.
Input target heads before every patch must match the accepted native capture
within2e-4. Quantization is defined by the actual dtype; sham qualification tests
its model-level effect. There is no unresolved dtype policy for the worker.

## Evidence and qualification

Root supplies immutable selection.json with source bindings and independently
computed cast/flatten target hashes. Bind the original source oracle and accepted
model/media/native inputs; reuse recurrence_native_trajectory.source_inputs
and attention_attestors, and the prior first-layer capture entry pattern.
Preserve producer and dependency sources plus manifest BEFORE the first forward.
Do not bind mutable state.json as immutable runtime source.

Use selected-logit physical index[2126], verified through actual LM-head input.
Save all4 batch members' full152670-vocabulary vectors per cell (shape[4,152670]),
all28 native mask/cache-slot/position attestations, unchanged input hashes,
actual intervention readback and forward/vision counts. Save raw tensors and
observed errors BEFORE qualification/postprocessing. Do not dump full hidden
histories, all-layer caches or attention matrices.

Native target full-vector must match accepted predecessor's last-query vector
within2e-4 and have the same global winner999. Sham11 must match fresh native
full vectors within2e-4, with identical target winner. In every intervention,
the three companion batch members' selected vectors must match native<=2e-4;
off-target o_proj input equality is exact. Input IDs/mask/positions and all28
actual consumed mask/cache-slot hashes must match native. No tolerance tuning.
Stop before the remaining cells if native/sham fails; no automatic retry.

CPU check must exercise the ACTUAL patch+second-observer helper on a small
nonconstant tensor and real torch.nn.Linear consumer with unequal batch/sequence/
hidden sizes. Encode distinct batch, query, head and feature values. Verify the
consumer output, not just the hook's returned tensor; deliberately shifting the
query by one or swapping head/feature flatten order must be rejected by the
source-aligned observer. Use the saved native query positions/token oracle to
verify807->2126, not arithmetic constants alone. Existing selector checks can
be reused. No new test framework or generic intervention manager.

Primary outcome is global argmax for00. Also retain full-vocabulary top5,
top1-top2 gap, FP64 P38/P999, fixed d=z38-z999, each vector's difference from
native and the2x2 d interaction. Mixed cells explain specificity but cannot
rescue a failed primary prediction. A binary margin never substitutes for the
global winner; EOS/other tokens are explicit outcomes.

## Scope, prompting and stop

Local whole-head substitution can identify sensitivity at this one fixed query.
Even restored38 would not predict natural onset/exit time, establish a learned
counter, identify physical-owner recovery, or prove the natural source of these
states. No automatic layer/depth/head scan or altered replacement strength.

Execution owner /root/luna_temporal, gpt-5.6-luna/max. Worker owns only
probes/training_set_completion/recurrence_first_layer_readout.py and raw output
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-first-layer-readout/attempt-001/.
Root owns records, source selection and independent readback. Reuse the same
worker; do not spawn children or change any accepted producer.

Budget exactly5 model/5 vision forwards,10minutes, GPU4; tensor payload64MiB
total. Expected GPU stress occupancy is not a reason to wait. Report concrete
OOM/conflict and preserve failed attempts. No extra smoke model pass is needed:
native+sham are the production-shaped qualification at the head of this package.

Improved brief uses a source/shape/index table, explicit hook-consumer boundary,
nonconstant index mutations, and a short understanding checkpoint before coding.
Luna already distinguished01/10 and raw807/physical2126 correctly. After root
dispatches this packet, execute CPU qualification and the five cells without
another permission round. Report source/CPU-qualified milestone, then proceed;
failed attempts return to root. Assess prompting from observed corrections and
acceptance in results; do not create a separate model benchmark or infer cost
savings without usage evidence. Stop after this package's candidate; only root
accepts or chooses a new scientific contract.

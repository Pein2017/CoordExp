# Native exit: normalized key source crossed with final rotary phase

2026-09-22. Lead-owned probe under the user's explicit authorization to collect
intermediate states and continue autonomously, with all GPUs available.
Fixed mature untied+axis step-2444, val7511, original refined-01 batch4/target2,
FP32 and unchanged SDPA. No checkpoint, image, donor, head or layer scan.

## Question and frozen contrast

At the native row89 x2 decision, replacing row88's post-RoPE keys with row87's
keys reverses the global winner from999 to38, with native values held fixed.
Does this effect travel with the final rotary phase or with normalized pre-RoPE
key vectors? The strongest alternative to phase transport is dependence on
contextual pre-key content; interaction is a third admissible outcome.

Predecessor protocol and identities:
research/experiments/2026-09-22-recurrence-history-location/native-row-mass.md.
Accepted anchors are native-row-mass-002 N and K, not B. Preserve all exact
checkpoint, untied-delta, input, image, processor and dependency bindings.
The loader path may differ from historical source only with the already
qualified identical bytes and size; record the current path and SHA.

Native historical prefill ends before raw801, physical width2121. Current
six-token S is raw801:807 / physical2121:2127 and recomputed for every condition
for all four actual companions. Row87 source raw783:792 / physical2103:2112;
row88 destination raw792:801 / physical2112:2121. Both have the same nine token
IDs. Target batch2 only; all28 layers. Source, all other history, companions,
native destination V, current positions and visibility stay unchanged.

Let k be normalized pre-final-RoPE K, with old=row87 and new=row88. At the
destination slots construct:

| Cell | Pre-key source | Final phase | Consumed K |
|---|---|---|---|
| NN | new | new | exact native row88 post-K |
| OO | old | old | exact row87 post-K |
| ON | old | new | R_new inverse(R_old) K_old |
| NO | new | old | R_old inverse(R_new) K_new |

NN and OO use exact cached bytes. Capture actual k_norm output and the actual
runtime rotary cos/sin for both rows in the single historical prefill. Invert
the installed rotation using (K*cos-rotate_half(K)*sin)/(cos^2+sin^2), and apply
the other captured phase. Verify duplicated half coefficients, identity,
roundtrip and reconstruction against observed normalized K. Do not assume an
unverified MRoPE packing or omit a nonunit rotation scale. Pre-key content still
contains upstream positional effects; this is not all position versus content.

## Primary acceptance and readout

NN must match the predecessor N full-vocabulary vector and OO predecessor K
within max absolute2e-4, with identical global argmax. Save all four full-vocab
vectors; independently reduce d=z38-z999, global top two and gap, FP64 softmax
P/logP38 and999. A winner gap<=0.001 is numerically inconclusive for binary
transport; keep the continuous margins. Report both phase effects, both pre-key
effects and factorial interaction, without treating four interventions as
population replication.

ON=999 and NO=38 supports bidirectional phase transport locally; ON=38 and
NO=999 supports pre-key source transport. Both crossed cells repeating, both
exiting, or near ties indicate conditional effects or inconclusive binary
transport; inspect the prespecified margins instead of inventing a new scan.
These statements concern this exact surgical construction and endpoint only.

Verify actual consumed cache/mask/positions at all layers and all four cells,
native V, other batches and source blocks, and native cache restoration after
cropping S, including finally paths. Wrong source/destination, axis, companion,
rotation and post-append restoration must fail the CPU selfcheck. Preserve
failed attempt receipts; no automatic retry or source changes mid-attempt.

## Intermediate evidence, separate from endpoint qualification

For target2/all six S positions/all28 layers capture current residual stream,
attention and MLP outputs, q_norm output, actual rotary phases, source/destination
pre/post K and native V, and actual per-head output from o_proj's prehook.
Reconstruct post-Q and attention probabilities from observed Q/K/V, installed
GQA mapping (16 query heads,8 KV heads), scale1/sqrt128 and actual boolean mask
(True=allowed). Do not switch SDPA or claim direct capture of its internal
softmax. Save target-query probabilities and only source/destination block
scores; no full historical cache archive. Intermediate payload cap256 MiB.

Reconstructed head outputs must agree with actual o_proj inputs at maxabs2e-4;
record errors by layer/cell. If this auxiliary gate fails, mark intermediate
interpretation HOLD while retaining independently qualified primary endpoints.
Do not silently relax tolerances or rerun the model for metadata.

For each cell C and layer/head, hold NN query and all NN K/V fixed except the
nine historical destination keys. Compute the resulting head output from NN
probabilities, native V and changed K, with FP64/expm1 stabilization and a
brute softmax qualification. Save fixed-state response versus NN and the
actual-C minus fixed-state remainder, norms and alignment for every layer.
The final S query is the prespecified primary query; retain the other five.
The remainder includes changed queries AND changed current-S K/V. Neither a
large remainder nor a large layer activation establishes a causal circuit.
Do not sum layer differences as independent effects or select a peak layer.

## Budget, ownership and stop

One native historical prefill plus four S scores:5 model and1 vision calls;
cap8 model calls and15 minutes execution, one GPU initially. Existing helper
source is imported normally; old sealed producers are not modified or rerun.
Source owner /root/trace_dynamics; root owns protocol, interpretation and
acceptance. Read-only technical adviser /root/cache_route; scientific adviser
/root/model_falsifier. Root independently checks original tensors and bindings.

Converge after the crossed endpoints and qualified intermediate decomposition.
One contingent whole-stack NN/OO Q+current-S K/V clamp may be frozen separately
if it discriminates feedback necessity; it is not part of this launch. No
automatic layer/head subdivision, new cohort/checkpoint, rollout, training,
publication or unrelated runtime cleanup follows. Numerical exit still changes
a bad water-person box into a bad multi-owner box; physical recovery is untested.

Producer: probes/training_set_completion/recurrence_key_phase.py.
Artifacts: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-key-phase/attempt-001/.

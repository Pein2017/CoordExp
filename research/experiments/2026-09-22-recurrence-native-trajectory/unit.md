# Native within-burst cache and readout trajectory

2026-09-22. User explicitly renews autonomous research and selects Luna-max for
new delegated work, without Astra workers. Root retains scientific decisions and
independent acceptance. The parallel task01a0c1f3-dbef-7b63-b2da-8dc7072cea8d is
running coordinate-codebook scale training/evaluation; this assay keeps the
original mature untied+axis step2444 composition fixed and does not touch that
owner's code, outputs or jobs.

## Current question and predecessor boundary

From the two accepted native histories, does contextual cache movement largely
settle soon after entry into exact repetition, or remain distributed across the
established repeated segment, under exact causal native-replay gates?
This is a DESCRIPTIVE temporal discriminator, not a causal accumulator assay.

Accepted first-versus-recent substitutions establish contextual source-state
noninterchangeability at two late exits. The first donor fails even when only
the last row is replaced; that result does not identify distributed replacement
extent or distinguish entry transient from later evolution. Relative final-key
phase affects attention, but static common-shift invariance forbids calling it
an absolute-position clock. The mass-only global-winner rule is rejected.
No old donor, layer or head scan is reopened.

Use accepted exact case preparation and source identities from
recurrence_attention_mass.prepare_case and the closed fixed-template unit.
Bind their lead receipts, original raw/trace/media/model/batch and native inputs
before forwarding. Source selection remains
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-fixed-template/selection.json.

| Case | Complete identical rows for K/V | Coordinate queries including exit | Target | Native full width | Action offset rule | Query input index |
|---|---|---|---:|---:|---|---|
| val7511 |27..88,62 rows|27..89,63 queries,x2|2|2127|9*r+6|1320+9*r+5|
| train269858 |16..19,4 rows|16..20,5 queries,x1|1|1546|9*r+4|1362+9*r+3|

All row indices are zero-based. The action offset is the NEXT token to predict;
its input query is one position earlier. The final queries must reproduce the
accepted native NN full vectors; every earlier query is evaluated against the
original per-token generation trace. These are original native greedy histories,
replayed causally, not newly generated or modified trajectories. Later supplied
tokens cannot enter an earlier query under the actual causal mask.

## Collection through the real consumer

One full exact-history forward per case. Use the installed model's supported
selected-logit index route if verified; do not materialize full T*vocabulary
logits. Preserve per-query full vocabulary vectors (63+5) and exact index/token
crosswalk. Capture all28 layers' normalized PRE-RoPE K and actual V for each
complete repeated row, Q at each declared query, and query residual input/output
where available from ordinary hooks. Capture actual native row/query cos/sin.
Do not change model forward, mask, positions, attention, cache or any activation.
No full attention matrices are required.

At actual self-attention consumption, attest causal/left-padding visibility for
every selected query, original positions/cache slots, source post-K replay and
actual cached V, with all28 layers covered. Match first/penultimate/last source
states against accepted packets where available. Preserve exact input/batch/
checkpoint identities and actual query selection by the LM-head input consumer,
not merely the requested logits_to_keep argument. Full final-logit parity<=2e-4
and identical global winner in both cases are required. Each earlier position
must match original chosen-token identity and saved top-two logits<=2e-4; inspect
the original trace schema rather than assuming contiguous list indices.
If trace evidence is absent or incomparable, stop before interpretation and
report the exact gap; do not silently relabel native parity as unchecked.

CPU qualification must exercise the installed selected-logit consumer or its
nearest real entry, including a one-token index shift that fails source-aligned
next-token matching. Native GPU anchors close the production-shaped boundary.
Save receipts and observed errors before qualification gates; preserve failures.

## Frozen descriptive readout

Primary long-run case is val; four-row train is a short trajectory comparison,
not a long-plateau replication. Report every layer separately; layers are not
independent examples. For X=pre-K or V and each layer, flatten each9token row:
Delta_i=||X_(i+1)-X_i||_F; T=sum_i Delta_i. Val has61 transitions. Burn-in is
first15 transitions (one quarter, floor(61/4)); late transitions start at
row42->43. Late movement fraction=sum_{i>=15} Delta_i/T.

The prespecified rapid-settling description requires late fraction<=0.10 in at
least90% of active layers, separately for BOTH pre-K and V. An active layer has
T>64*eps(float32)*max_row||X_row||_F. Retain every layer, denominator and threshold;
zero/near-zero layers are reported as inactive, not dropped silently. Failure
rejects this quantitative90/10 description, not all onset-transient mechanisms.
Success indicates geometric settling only: tiny remaining changes may still
matter near a decision boundary. Do not equate Euclidean drift with causal
importance or identify a scalar accumulating variable.

Also report full-vocabulary argmax/top-five/gap and absolute FP64 probabilities
at all68 queries, fixed val z38-z999 and train z350-z348/z591-z348. Plot the native
margin and contextual-change trajectories with position and coordinate identity
explicit. Preserve the complete tensors for independent interpretation.
A first-to-last chord projection and residual distance may be supplementary;
no fitted decay model, threshold tuning or new layer selection is authorized.

If late movement is substantial, the next scientific decision is whether it
has decision-relevant temporal structure beyond entry effects; it does not
immediately authorize another intervention. If states settle but logits evolve,
positional readout/history-pool composition becomes the stronger next contrast.
Neither outcome claims burst origin, held-out exit timing or physical recovery.

## Owner, cost and stop

Execution owner /root/luna_temporal, model gpt-5.6-luna, effort max, as selected
by user. Root owns records and acceptance. The worker may implement and run
within this contract once root dispatches it; no child agents or new scientific
arms. Preserve unrelated dirty work and the parallel codebook run.

Own only probes/training_set_completion/recurrence_native_trajectory.py and
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-native-trajectory/attempt-001/.
Reuse maintained preparation/capture helpers; do not edit accepted producers.

Budget2 model/2 vision forwards, cap4 model calls and15minutes, one GPU4;
intermediate artifact cap256MiB per case, no full attention maps/cache dump.
Do not wait for expected GPU stress occupancy; report concrete OOM/conflict.
No automatic retry. Return qualified candidate plus exact counters/source hashes,
CPU check, full-vector/trace parity, artifact sizes and terminal job status.
Root independently accepts the original tensors and interpretation. Record any
Luna misunderstanding/correction in this unit's result; one package does not
establish a model capability ceiling or a measured cost advantage.

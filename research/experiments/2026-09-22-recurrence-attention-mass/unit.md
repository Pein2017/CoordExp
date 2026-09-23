# Live-query row log-mass crossed with within-row score profile

2026-09-22. Root's next bounded loop under the user's explicit instruction to
continue autonomously. Two fixed cases, same mature untied+axis checkpoint.
The val7511 key/phase experiment found phase-following choices and a large
contrast lost under a joint current-S Q/K/V clamp. The train269858 directional
transfer prediction failed: old phase chooses591, not the predicted350, while
new phase chooses348. This new experiment does not repair that failed forecast.

## Question, sources and prediction

At each actual live query, does the final-phase-dependent row log-normalizer
carry the two observed choice contrasts, or is within-row allocation required?
An accepted-data CPU factorization motivated this question: val7511 pure-phase
NO has larger fixed-state mass-only than allocation-only response in355/448
layer-heads, all above the reconstruction-noise guard. That is descriptive,
not already a causal contribution to final logits.

Use only the two already fixed states and native pre-key source=new throughout:

* val7511 row89 x2: target2, prefix2121, full2127, S6,
  destination2112:2121. New-phase NN predicts999, old-phase NO predicts38.
  Source: 2026-09-22-recurrence-key-phase/attempt-003.
* train269858 row20 x1: target1, prefix1542, full1546, S4,
  destination1533:1542. New-phase NN predicts348, old-phase NO predicts591.
  Source: 2026-09-22-recurrence-phase-transfer/attempt-002.

Both source roots are under
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/.
Bind original model/batch/media/input/producer identities, exact native histories
and original current prefixes, phases and masks. Original companion EOS/padding
stays fixed. Require lead acceptance of both source contrasts before launching.

For layer l, head h and each current query q, form scores s_N=q K_N^T/sqrt128
and s_O=q K_O^T/sqrt128 for the nine destination keys. K_N is exact native new
post-K, K_O is the accepted new-pre-key/old-final-phase construction. Let
L_N=logsumexp(s_N), L_O=logsumexp(s_O). All use the SAME LIVE q in that arm,
not stored NN queries. Cached key phase chooses conditional profile P; a uniform
bias b=L_M-L_P to those nine attention logits chooses row log-mass rule M.

| Cell | Row log-mass rule M | Cached key/profile phase P | val prediction | train prediction |
|---|---|---|---:|---:|
| NN | new | new |999|348|
| OO | old | old |38|591|
| ON | old | new |38|591|
| NO | new | old |999|348|

Names here are MASS then PROFILE, distinct from the predecessor pre-key/phase
names. In particular this OO anchor equals the predecessor NO vector. State
this mapping in machine-readable manifests and do not silently substitute old
pre-key content. Both mixed winners following mass phase in both cases is the
advance strong prediction. Either decisive profile-following or third winner
rejects it; near ties<=0.001 remain inconclusive. No successful case can erase
a failure in the other. Preserve full logits and actual global competitors.

## Actuator and invariants

At all28 layers and all S positions, use unchanged SDPA with an independent
float mask[B,16,S,full]. Original allowed entries become0, blocked entries−inf.
Clone after expansion. Change only the target batch's nine historical row
entries, equally across the nine keys for each head/query. Native historical
V, other history/companions, current positions and visibility stay unchanged;
current S is recomputed freely for every cell. No query, current-K/V or residual
clamp. Restore native historical cache exactly after S append/crop.

The attention prehook may retain the actual rotary coefficients and replace
the mask kwargs; q_norm hook uses actual normalized Q to fill the same mask
before SDPA. Do not duplicate Q projection or use a stale query. Instrument the
actual consumer path and retain per-layer post-Q, both phase-key blocks, scalar
bias, before/after row scores or logsumexp, mask identity and head-output check.
No full attention-mask/cache archive is needed; intermediate cap256MiB per case.

On actual mixed-cell queries at every layer/head/query require:
1. post-bias row logsumexp equals selected L_M within2e-4;
2. centered nine-key scores equal selected profile within2e-4;
3. outside-row mask entries and blocked visibility are exactly unchanged,
   target historical V/source/companions unchanged;
4. reconstruct attention from the consumed Q/K/V and float mask and match
   actual o_proj input within2e-4. Record errors before gates, preserve failures.

Float-mask diagonal NN and OO full-vocabulary vectors must reproduce the
accepted predecessor NN and NO vectors, respectively, within2e-4 and identical
global argmax. This is the production-shaped no-op/mask-path qualification.
CPU caller tests through installed Qwen attention must show boolean/float-zero
equivalence, hook ordering, uniform-bias equivalence to a direct mask call, and
rejection of wrong destination/head or mass/profile assignment. A passing leaf
formula alone cannot admit the actual actuator.

The identity controls a head/query-wise attention rule. It does NOT match
normalized attention mass between separate arms whose queries and outside
states differ, isolate one scalar gate, or estimate natural mediation fractions.
Earlier-layer changes can alter later Q and current-S K/V. Keep that adaptive
computation active: it is the estimand.

## Readout, cost and stop

Primary fixed margins are old-phase winner minus new-phase winner:
val d=z38-z999; train d=z591-z348. The train350 value remains a reported
diagnostic because it was the failed predecessor prediction; never redefine
that earlier success criterion. Report all four full-vocabulary argmaxes/gaps,
FP64 P/logP of declared competitors, both mass effects, both profile effects
and interaction. Matching a winner is not matching an effect size or vector.

Execution owner /root/trace_dynamics; root owns decisions/acceptance/records.
Read-only adviser /root/model_falsifier; technical adviser /root/cache_route.
Reuse maintained operations and a small two-case producer, without editing
sealed producers or introducing a general intervention framework.

One historical prefill plus four S scores per case:10 model/2 vision calls,
cap12 model/20 minutes, initially one GPU4. All input/source/phase/readback
records precede scientific interpretation; root independently checks tensors
and consumer evidence. No automatic retry or extra case. Stop this unit after
the two-case verdict; root may select the next distinct discriminator under
the user's ongoing loop grant, without treating this unit as an unbounded scan.
No training, publication, physical-recovery claim or unrelated runtime cleanup.

Producer: probes/training_set_completion/recurrence_attention_mass.py.
Artifacts: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-attention-mass/attempt-001/.

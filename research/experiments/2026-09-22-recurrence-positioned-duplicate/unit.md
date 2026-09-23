# Position-correct preceding-row K/V substitution

2026-09-22. Root freezes one further discriminator under the user's continuing
instruction to loop autonomously. The preceding mass/profile experiment is
accepted and closed with its global-winner prediction rejected; it is not
reopened. No new image/checkpoint, dose scan or query/K/V component partition.

## Question and advance prediction

Can the preceding row's contextual K/V, placed at the new row's correct final
rotary phase, replace freshly computed new-row K/V while preserving the native
global winner? This is conditional replacement sufficiency, not context-free
copying or a count-only account. Old K/V already contain contextual information.

Static interleaved MRoPE is invariant to a common shift of all text query/key
axes with pre-Q/pre-K fixed. Natural new query/newest-row motion preserves their
relative phase; the earlier one-row phase intervention changed relative age.
The proposed experiment closes the remaining joint-cache substitution gap:
old pre-K/new phase with native V and the specified old-V/native-K substitution
were separately sufficient on val, but their joint effect has not been tested.

Use the same mature untied+axis step2444 checkpoint and exact native batches,
media, histories, masks, current-S positions and source packets as the accepted
[phase crossing](../2026-09-22-recurrence-key-phase/results.md) and
[fixed train transfer](../2026-09-22-recurrence-phase-transfer/results.md).

| Case | Target batch | Historical width / full | Old source | New destination | S | Predicted NN and D winner |
|---|---:|---|---|---|---:|---:|
| val7511 row89 x2 |2|2121 /2127|2103:2112|2112:2121|6|999|
| train269858 row20 x1 |1|1542 /1546|1524:1533|1533:1542|4|348|

Two cells per case:
- NN: unchanged native history and S.
- D: at every28 layer, replace only destination K by the accepted old pre-K
  rotated to the actual new-row phase (predecessor ON key), and destination V
  by the actual preceding/source row's V. Keep source row, other history and
  all companions fixed. Current S recomputes freely; no mask/score/Q/K/V clamp.

Prespecified success is native global winner in D with top-two gap>0.001 in
both cases, plus all technical invariants below. A decisive different winner
rejects joint two-case sufficiency; a gap<=0.001 is inconclusive. Report cases
separately and retain full vectors/all competitors. Winner retention is not
vector equality. Compare D with the accepted key-only ON and native NN vectors;
report z38-z999 for val, z350-z348 and z591-z348 for train, absolute FP64 P/logP,
full-vector distance, and global top-five. Do not reinterpret a third winner.

The accepted val masked-row M and old-phase whole-K/V B controls remain fixed
references: M and B choose38. If D retains999, this distinguishes correctly
positioned substitution from literal post-RoPE K/V duplication, and supports an
addition effect versus M in this late context. Train lacks a matched masked-row
control; only replacement sufficiency is claimed there. No generic burst-origin,
physical progress, population transfer or absolute-position clock claim.

## Execution and qualification

Root owns scientific decisions and records; /root/trace_dynamics owns one
producer and its attempt001 directory. Reuse maintained case preparation,
identity, source capture, exact-history input and cache helpers. Do not edit
sealed producers or introduce a reusable intervention framework.

Bind accepted source/lead receipts before any forward, including mass/profile
closure. Original input/model/media identities must match. One historical
prefill per case verifies live old/new post-K and native destination V against
accepted packets; current phase must be native. Persist source V, native V,
old/new pre/post-K provenance and actual consumed destination K/V hashes.
Accepted predecessor ON is the old-pre/new-phase key; it must not be confused
with mass/profile ON (old mass/new profile).

On all28 actual self-attention consumers, qualify historical length, target
source/destination K/V, masks, slots, unchanged outside history and companions.
Native mask and current positions/S must exactly match accepted NN. After S
append/crop, restore the native cache by layer reference and assert its digest.
The fresh NN full-vocabulary vector must match the accepted NN within2e-4 with
identical argmax. CPU installed-attention caller or actual-cache selfcheck must
reject wrong phase/V destination and restoration leakage; GPU NN is the real
entry anchor. Persist failure evidence before stopping. Root independently
recomputes phase construction and inspects consumed K/V and full vectors.

Budget: two cases x (one prefill + NN S + D S) =6 model/2 vision forwards;
cap8 model calls,15 minutes, one GPU4 initially. Intermediate cap128MiB/case.
No generation or new trajectories, no automatic retry or extra arm. Stop the
unit after the two-case sufficiency verdict. Root synthesizes the full mechanism
loop and only selects another distinct contrast if it can change that conclusion.

Producer: probes/training_set_completion/recurrence_positioned_duplicate.py.
Artifacts: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-positioned-duplicate/attempt-001/.

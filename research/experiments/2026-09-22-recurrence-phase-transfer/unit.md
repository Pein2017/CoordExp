# Cross-case directional transport of historical-key phase

2026-09-22. User requests continued autonomous analysis loops. Root selects
train:269858 row20 x1 before new interventions, following the already accepted
val7511 phase/pre-key crossing and whole-stack current-S Q/K/V clamp.
Source: research/experiments/2026-09-22-recurrence-key-phase/results.md.
Native exit and four-row run were exposed in the earlier transition readback;
this is retrospective exit-selected causal transfer, not prospective forecasting
of an unseen natural trajectory or an estimate of prevalence.

## Question and advance prediction

Does the same directional phase transport hold for this different image,
coordinate role, repeated value and much shorter exact run? The strongest
alternative is specimen-specific phase sensitivity: contextual pre-key content
or interaction may dominate this second exit. A failed directional pattern
rejects the proposed transfer while leaving the local val7511 result intact.

Native rows16–19 repeat person [350,181,369,230]. Row20 first changes at x1,
350 to348. Target original untied new-14 batch index1/source_row57. Source
row18 raw162:171 and destination row19 raw171:180 have identical nine IDs:
[151646,8987,151647,151648,152020,151851,152039,151900,151649].
Current S is row20 raw180:184, four IDs[151646,8987,151647,151648]; target
raw184 is token152018 (348); old token152020 is350. Derive physical slots from
exact native batch reconstruction and verify raw/physical alignment before any
forward. Keep every original batch companion and its actual prefix.

Exact reconstructed full width1546, historical width1542; source1524:1533,
destination1533:1542, S1542:1546. Original batch image IDs are
266124,269858,271580,277496. Companions have already emitted EOS by this
offset; retain their original EOS followed by padding through184 and the
original left padding. The saved target top-two logits are348=16.6262626648,
350=16.5974292755. No source full-vocabulary vector exists at this offset, so
the six-call plan below is required.

Same mature untied+axis step2444 adapter, effective input/output deltas,
FP32 SDPA, original image/processor/native-policy identity. Bound source paths,
hashes, batch identity and exposure are supplied by the selection packet at:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-phase-transfer/selection/.
Selection file selected-case-metadata-v1.json has SHA256
7e5d4b6fa308e1a10a47592e2c88d4f1a3c31fe5effb5dc3e512f059115e96aa.
No training/checkpoint change or annotation/physical-owner claim.

Native historical prefill ends at raw180; freshly recompute four S positions
for every cell. At target1/all28 layers change only destination historical K;
source, all other history, native destination V, all companions, masks and
current positions remain fixed. Exact old/new K and actual normalized pre-K
and cos/sin are captured from one native prefill. Four constructions use the
qualified predecessor rotation/inverse and actual interleaved coefficients:

| Cell | Normalized pre-key source | Final rotary phase | Advance global winner |
|---|---|---|---:|
| NN | row19/new | row19/new |348|
| OO | row18/old | row18/old |350|
| ON | row18/old | row19/new |348|
| NO | row19/new | row18/old |350|

NN/OO use exact cached bytes; ON/NO invert the corresponding observed post-K
then rephase. Preserve the already qualified FP32 scale-aware rotation checks
and independent FP64 complex-pair oracle. Removing final rotation does not
remove upstream positional effects from pre-keys.

## Decisive evidence and technical acceptance

Primary margin d=z350-z348. All four predicted global argmaxes and absolute
signed margins>0.001 are required for full directional transfer. Near ties,
partial rescue, other global winners and contextual/interaction effects are
reported as observed, not promoted to successful transfer. Save full-vocabulary
logits and independently reduce global top two, d, and FP64 P/logP348/350.
Report both phase effects, both pre-key effects and factorial interaction.

Obtain one full-native-history reference vector at raw184 unless an existing
bound full vector is available. Qualify its actual argmax and available raw
top-two logits against the original trace (maxabs2e-4); trace top-two alone is
not full-vector parity. Cached NN must then match the fresh full-history
reference at maxabs2e-4 over the entire vocabulary and same global winner.
Missing original trace parity or identity invalidates the affected contrast.

Observe actual embeddings, position/cos-sin consumption, all28 historical
destination/source K/V blocks, actual mask and cache length. Check other
batches and history remain unchanged and exact restoration after S append/crop,
including exception paths. Save original normalized pre-K, post-K, phases,
constructed blocks and error components before numeric gates. Reject wrong
batch/block/phase and restoration failures in a focused CPU check.
All bindings/producer/dependencies are captured before model execution. No
model rerun merely for metadata. No new all-layer attention decomposition or
clamp is part of this transfer launch.

## Ownership, cost and next decision

Root owns interpretation, next loop and acceptance. Native execution owner
/root/trace_dynamics owns producer and attempt; /root/evidence_map owns only
CPU selection/exposure packet; /root/model_falsifier is read-only scientific
adviser. Reuse helper operations via maintained imports, not source captures or
outputs execution. Avoid a generic runner or duplicated large producer.

Budget: one full-history qualification +one historical prefill +four S scores,
6 model/2 vision calls planned, cap8 model and15 minutes, GPU4 initially. If an
existing qualified full vector is available, planned cost drops to5/1. Preserve
failed attempts within the cap; no automatic retries or worker-run successor.

Stop this unit after the one fixed case and independent acceptance. No extra
case can repair its transfer verdict. Root then chooses a distinct next question
within the user's continued-loop grant: successful transfer motivates genuine
withheld-outcome prediction; failure motivates explaining the concrete boundary
between cases. That choice must have its own frozen discriminator, not a layer,
head, phase-dose or favorable-case search. No user permission pause is needed
for such bounded continuation under the current grant.

Producer: probes/training_set_completion/recurrence_phase_transfer.py.
Artifacts: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-phase-transfer/attempt-001/.

# Conditional continuation: whole-stack current-S Q/K/V clamp

2026-09-22, root-frozen after independent acceptance of the four-cell crossing
and intermediate decomposition. Primary receipt:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-key-phase/lead-checks/phase-readback.json.

Both crossed choices follow final rotary phase. For the NN-to-OO contrast,
the final-query feedback remainder exceeds the fixed-native direct component
in25/28 layers. This descriptive norm result cannot show that feedback causes
the margin change. One global clamp distinguishes whether the historical-key
contrast survives when current-prefix attention states cannot adapt.

## Two cells and actual consumption

Keep the exact accepted inputs/model/positions/masks/SDPA, native historical V,
target2, all28 layers, source/destination blocks and six-token current S.
One native historical prefill supplies NN and OO historical K constructions.
For both conditions replace target2's normalized pre-RoPE Q and K and projected
V at all six current-S positions in every layer with the accepted NN values
from attempt003/NN-intermediates.pt. Current rotary phases are the same as NN.
Historical keys are NN (native destination K) or OO (exact old-row post-K).
All remaining history, other batches, residual streams and MLP computations
are free and unchanged by the actuator. Recompute all six S for all four
companions each time; crop and restore native cache between cells.

Do not clamp Q alone: changed current-S K/V would leave another feedback route.
Clamp hooks must precede observation hooks. Observe the actual normalized Q
consumed by attention and actual appended S K/V at o_proj input time; require
exact pre-Q/pre-K/V and position/mask identities and post-K parity<=2e-5 with
the captured NN phase. Check all28 layers and all six positions. Verify native
history/source/destination-V and companions, plus cache restoration. Persist
actuation/readback records and full output logits; no full-cache dump needed.

Clamped NN must match accepted NN full-vocabulary logits within2e-4 and retain
global999. Failure invalidates the contrast. The old unclamped OO is an accepted
reference, not a new model call. Full logits for both new endpoints support an
independent FP64 probability reduction and global argmax/gap check.

## Frozen decision

Let d=z38-z999. The unclamped contrast is0.7541065216064453. Report
r=(d_OO_clamp-d_NN_clamp)/0.7541065216064453.
Operational descriptive bands, not confidence intervals or mediation fractions:
0.8<=r<=1.2 substantial retention; abs(r)<=0.2 substantial collapse;
r< -0.2 reversal; all other values partial retention or enhancement.
Always report continuous values and actual global winners separately.

If clamped OO still selects38 with gap>0.001, adaptive current-S Q/K/V feedback
is unnecessary for this categorical reversal under the clamp, even if r is
small. If magnitude collapses, feedback contributes to the size of the
unclamped contrast under this joint intervention; it does not establish that
feedback is necessary to cross a near-tied greedy boundary. Residual and MLP
computation remains active. No additive natural-mediation or unique-layer claim.

## Authority, budget and stop

The user's autonomous intermediate-state authorization covers this previously
declared contingent discriminator. This is a separate three-call continuation,
not a retry or expansion of the phase crossing's8-call cap. One shared native
prefill plus two S scores:3 model/1 vision, cap3 calls and10 minutes, initially
GPU4. Total planned unit cost including two retained failures:10 model/4 vision.
CPU hook sensitivity and baseline replay qualify the actuator. No automatic
retry. Root owns acceptance and closes regardless of scientific outcome, with
no layer/head subdivision, more phase doses, new images/checkpoints or rollout.

Source owner: /root/trace_dynamics. Root owns this record and lead-checks.
Producer: probes/training_set_completion/recurrence_key_phase_feedback.py.
Artifacts: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-key-phase/feedback-clamp-001/.

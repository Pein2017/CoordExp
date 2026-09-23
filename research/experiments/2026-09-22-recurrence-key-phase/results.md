# Key-source / final-phase crossing

Crossing, intermediates and whole-stack feedback clamp are lead-accepted.
The unit is closed at its prespecified local computational boundary.

## Execution rulings

Attempt001 stopped after1 model/1 vision prefill at layer0's combined RoPE
qualification, before any S score. No endpoint exists. Its receipt reports
15.5444126725 seconds elapsed; model time and peak memory were not recorded.
The producer raised before saving subcheck errors or observed tensors, so the
specific failed subcheck cannot be inferred from this attempt.

Root authorizes attempt002 as a capture-first instrumentation repair: save all
28 layers' actual phase, normalized pre-K, cached post-K, constructed keys and
per-subcheck errors before applying the same numeric gates. Preserve attempt001
and bind its receipt/source. Do not loosen rotation, endpoint or auxiliary
thresholds. If any gate fails, return immediately with inspectable tensors;
otherwise finish the frozen four S scores. No automatic retry. The cumulative
cap remains8 model calls, including attempt001's1. A later repair requires a
concrete diagnosis and explicit lead ruling under the existing user grant.

Attempt002 again stopped after1 model/1 vision prefill and no S endpoints,
now with complete CPU-inspectable tensors. All28 native pre-K to cached post-K
replays have exactly zero error. Inversion and cross-phase roundtrip maximum
errors are6.103515625e-5, exceeding the producer's absolute2e-5 gate. Layer0
pre-key maximum is424.0510559 and post-key maximum424.0556641. The lead's
independent FP64 complex-pair oracle finds inverse versus captured pre-K error
2.1132189e-5 and crossed construction error at most4.1249853e-5; normalized
by float32 epsilon times each layer's key maximum, worst inverse error is0.789
and worst crossed error1.595. These measurements support ordinary FP32
rounding at the observed scale, with exact native reapplication, rather than
a phase/source mismatch. See lead-checks/failed-rotation-diagnosis.json in the
unit's external artifact root.

Root authorizes attempt003 with unchanged intervention arithmetic and native
replay gate2e-5. Replace inversion/roundtrip/norm absolute gates with
8*epsilon32*max(1, reference magnitude), using respectively maximum absolute
pre-K, maximum absolute post-K, and maximum vector norm. Additionally compare
crossed K against the independent FP64 complex-pair oracle under the post-K
bound, recording all errors, scales and ratios. A wrong-phase sensitivity
check must fail. This is an explicit producer numerical-qualification repair;
NN/OO full-vocabulary tolerance2e-4 and auxiliary attention tolerance2e-4
remain unchanged. Bind both failed attempts. Prior2 plus planned5 calls gives
7 cumulative model calls, within the original8-call cap. No automatic retry.

## Accepted phase crossing

Full-vocabulary anchors NN=N and OO=K match exactly. All46 bound artifacts
pass independent SHA/size checks; model/batch/native-input identities match
the accepted predecessor. All28 source/destination cache blocks, native V,
S masks/positions, restoration and residual identities pass.

| Cell | Pre-key | Final phase | d=z38-z999 | Winner | P38 | P999 |
|---|---|---|---:|---:|---:|---:|
| NN | new | new | -0.006084442 | 999 | 0.041014349 | 0.041264659 |
| OO | old | old | +0.748022079 | 38 | 0.047636520 | 0.022546450 |
| ON | old | new | -0.017925262 | 999 | 0.040817183 | 0.041555439 |
| NO | new | old | +0.756042480 | 38 | 0.047779792 | 0.022433610 |

The new-minus-old phase effect is-0.765947342
with old pre-keys and-0.762126923 with new pre-keys.
Old-minus-new pre-key effects are-0.008020401
under old phase and-0.011840820 under new phase.
Interaction is-0.003820419. Both crossed winners follow
final phase; the pre-key-source effects are small in this factorial contrast.

This identifies a local computational carrier: final rotation of the repeated
historical row changes attention-address matching and can carry the38/999
decision across the greedy boundary with native V fixed. It does not show
that rotation is the origin of the bad detection, a universal recurrence
mechanism, or that upstream positional effects are absent from pre-keys.
The entire repeated block is intervened at all28 layers, so no unique head,
layer or token has been identified.

## Accepted intermediate evidence

Target-only traces cover all28 layers and all six current-S queries. Reconstructed
SDPA attention outputs match actual o_proj inputs within7.6293945e-5; the
fixed-native-state brute-softmax qualification is1.3386e-5. Root independently
recomputed the saved decomposition with FP64 key subtraction and obtained
maximum error1.8944e-6. These are reconstructed and validated probabilities,
not directly captured internal SDPA softmax buffers.

At the final query, OO feedback-remainder norms exceed fixed-native direct
response norms in25/28 layers. Layer0 direct norm is0.095210 and remainder
7.34e-7; layer27 direct8.49699, remainder19.86739, actual difference20.32726.
Layer norms are descriptive, not additive causal effects on the logit margin.
The remainder includes changed queries and current-S K/V. This motivates the
single prespecified [whole-stack clamp](feedback-clamp.md), rather than a
peak-layer search.

Two identical token rows have exactly equal normalized pre-keys at layer0.
Across layers, relative pre-key difference is0–0.01670, whereas post-rotation
key difference is0.08679–0.49152 (whole-block L2 relative to native new keys).
This is descriptive consistency with the phase intervention, not an independent
causal test or proof that small contextual differences cannot matter.

Primary acceptance: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-key-phase/lead-checks/phase-readback.json.
Attempt003 result: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-key-phase/attempt-003/result.json.
Attempt003 used5 model/1 vision calls,33.4538s measured execution,40.5161s
elapsed, peak reserved12,918,456,320 bytes,183.8MB artifacts. Including the two
failed prefills:7 model/3 vision calls; failed-attempt model time/peak were
unrecorded, so cumulative model-time/peak claims exclude those unknowns.

## Accepted whole-stack feedback clamp

Both conditions hold target2 current-S normalized Q/K and projected V to
accepted NN values at all six positions/all28 layers. Actual consumed Q/K/V
hashes match; appended post-K and V match exactly; masks and companion output
hashes are unchanged. The lead checked36 bound artifacts and the actual
consumer path. Clamped NN reproduces its accepted full-vocabulary vector
exactly. Attention reconstruction remains below2e-4.

| Historical keys | d=z38-z999 | Global winner | P38 | P999 |
|---|---:|---:|---:|---:|
| NN | -0.006084442 | 999 | 0.041014349 | 0.041264659 |
| OO | -0.023942947 | 999 | 0.039559117 | 0.040517709 |

The clamped contrast is-0.017858505, versus unclamped
+0.754106522; r=-0.023681675. This falls in the frozen
substantial-collapse band. The residual magnitude is2.37% of the unclamped
contrast, with opposite sign, and OO no longer restores the38 winner.

**Lead interpretation:** the final historical-key rotation carries the local
decision contrast, and its large effect requires current-row Q/K/V adaptation
under this joint clamp. A supported computational account is: changed phase
changes qK scores and historical attention output; the current residual state
then changes, later layers recompute Q/K/V and reread history, and the resulting
logit competition crosses the38/999 boundary. The direct fixed-state route
alone does not preserve this particular categorical reversal. This identifies
a local dependency through successive layers; the clamp does not distinguish
Q from K or V feedback, identify individual heads, or estimate a natural
mediation fraction. The small reversed residual does not meet the registered
substantial-reversal band.

This is **within one forward computation**, with the six current prefix tokens
fixed. It is not by itself proof of a recurrent dynamical mechanism over newly
generated tokens, the onset of the hallucination, or the origin of an entire
burst. Native999 remains a bad multi-owner box;38 remains the bad water-person
box. No physical recovery was evaluated.

The next decision-changing scientific question is whether this phase-conditioned
state propagation predicts an unmodified natural exit before its winner is
observed, in a separately frozen held-out transition. That would test transport
and explanatory prediction; a phase-dose/head/layer sweep on this same exposed
exit would supply sensitivity rather than that evidence. No successor or GPU
job is active.

## Closeout and artifacts

The clamp used3 model/1 vision calls,22.8098s measured execution,29.6613s elapsed,
and12,916,359,168 bytes peak reserved. Across this unit:10 model/4 vision calls
including two failed diagnostic prefills. Valid-attempt measured execution is
56.2636s; all-attempt elapsed is103.0930s. Maximum recorded peak is12,918,456,320
bytes; failed-prefill model time and peaks remain unknown. All four jobs are
terminal. Root reran both CPU selfchecks and independently reduced original
logits, rotary constructions, masks/cache consumption and intermediate tensors.

All-layer descriptive figure (fixed-native response versus feedback remainder):
![All-layer attention decomposition](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-key-phase/lead-checks/all-layer-attention-decomposition.png)

Clamp acceptance: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-key-phase/lead-checks/feedback-clamp-readback.json.
Final receipt: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-key-phase/lead-acceptance.json.

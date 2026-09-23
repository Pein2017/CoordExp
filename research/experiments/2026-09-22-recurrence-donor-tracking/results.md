# Written values change broad support without localized donor transfer

Date: 2026-09-22. **Lead-accepted; five-condition assay closed.**
The [prospective protocol](unit.md) owns the contrast and numerical criteria.
[Acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-donor-tracking/lead-acceptance.json)
binds the independent readback and model artifacts.

At mature untied+axis step2444, val:7511 row88 x2, replacing one preceding-row
coordinate with640 or832 produced **0/4 localized absolute amplification cells**.
Neither source axis passed matched two-donor transfer. These results lower the
priority of a strong literal identity-echo account under this assay. They do not
exclude smooth or masked token contributions, box copying, or history effects.

## Numerical decision and absolute probabilities

Only previous row87 x2 or y2 changes. Current S, sequence length, positions,
causal mask, image and batch companions are fixed. All probabilities below use
FP64 softmax over the full vocabulary, not renormalization over a candidate pair.
The row and token offsets are zero-based.

| History condition | P(current x2=38) | P(current x2=999) | z38-z999 | Global winner |
|---|---:|---:|---:|---:|
| Native R | 0.041947 | 0.034114 | +0.206699 | 38 |
| X640: previous x2=640 | 0.002547 | 0.035096 | -2.623060 | 999 |
| X832: previous x2=832 | 0.012665 | 0.040097 | -1.152481 | 999 |
| Y640: previous y2=640 | 0.040231 | 0.018878 | +0.756647 | 38 |
| Y832: previous y2=832 | 0.023571 | 0.018439 | +0.245550 | 38 |

Both x2 edits switch the winner before the natural exit. In particular, X640
lowers P38 from4.19% to0.25%, while P999 moves only from3.41% to3.51%.
Thus the switch accompanies strong depletion of the repeated value and broad
redistribution; it is not evidence that640 was selectively copied into the next
coordinate. The two y2 edits retain38 and lower P999. Their starting values and
numeric displacements differ from the x2 edits, so this is not a calibrated
comparison of intrinsic axis strength.

The supplementary bound readback of the earlier row89 experiment is also
decision-relevant: writing999 in past y2 improves999/38 odds while P999 falls
from0.041264 to0.036163. The earlier relative-support claim remains valid, but
an interpretation as absolute amplification of999 would be incorrect. That
readback uses existing vectors and consumes no additional model forward.

## Frozen identity-tracking criteria

F(v) is the change in log probability at donor v; C is local curvature and M is
the smaller difference against either immediate neighbor. Localized absolute
amplification requires F>0.001 and M>0.001. The guard is numerical, not a
confidence interval.

| Cell | F(v) | C(v) | M(v) | Localized absolute amplification |
|---|---:|---:|---:|---|
| X640 | +4.315330 | +0.063747 | -0.094855 | no |
| X832 | +3.063017 | -0.069067 | -0.168450 | no |
| Y640 | -0.068887 | +0.153498 | +0.035619 | no: local feature, net decrease |
| Y832 | +1.997652 | -0.036041 | -0.142333 | no |

Matching-donor differences must also both exceed0.001. X gives
(D640,D832)=(+1.557972,-0.632138), and Y gives(-1.256681,+1.927656).
Neither passes; positive sums T alone would conceal a failing recipient.
This does not turn failure of the strict peak test into proof of no identity
component.

Descriptive inspection of the saved1,000-bin responses strengthens the broad
redistribution interpretation. X640 and X832 response correlation is0.942946
(affine R-squared0.889147); their largest response peaks are703 and750.
Y832 exceeds Y640 at920/1,000 coordinate bins. These are posthoc shape
descriptions, not replacement acceptance criteria or independent observations.
No smoothing window, shifted peak, donor or threshold was selected afterward
to convert the frozen negative result into a positive one.

## What changed in the research judgment

The combined evidence now supports three distinct observations: position and
added history have opposing local effects; written coordinates affect the next
decision; and that effect need not selectively amplify the written identity.
The simpler narrative "a new coordinate is written, then the model copies it"
is insufficient for these interventions.

A more promising working hypothesis is conditional maintenance of the existing
coordinate: recent same-role content helps sustain38, and altering that content
weakens its support enough for an already competitive999 to win. This is a
hypothesis, not an identified attention pathway. Changing the preceding box's
geometry and contextual compatibility is a strong alternative explanation.
The observations do not establish that recent history is sufficient, that it
is the sole memory timescale, or that a scalar accumulator governs recurrence.

A genuinely different next discriminator would preserve the complete multiset
of row contents while exchanging the altered row's location with an older
identical row. A location-sensitive response would weaken a position-insensitive
content-count account; equal responses would weaken a strongly local anchoring
account. Relative position, intervening contextual computation and recency
would still need separation before identifying a circuit. This is a research
direction under the continuing user authorization, not another launch in the
present assay. The present result does not justify another donor or layer scan.

The repeated water-person and first long multi-owner box are both user-adjudicated
bad predictions. Numerical winner changes do not establish physical recovery;
later valid kite/person outputs remain outside this endpoint. There is one
exposed image/state, no population estimate, training-origin claim or remedy.

## Evidence and cost

The lead rebuilt the native and four edited input tensors independently from
literal prompt/raw tokens, verified their hashes against the actual embedding
hooks, and checked fixed rotary positions and first/last masks/cache slots.
Raw edit offsets789/790 map to physical[2,2109]/[2,2110]; current S792:798 is
unchanged. All35 inspected source/vector/capture bindings match.

Native R is bit-identical to the accepted E_E full-vocabulary vector. Saved
native top-two error is7.6293945e-06. The lead independently recomputed all
probabilities, donor metrics, transfer decisions and vocabulary winners.
The producer selfcheck passes for wrong prefixes, edit slots, broad versus
localized responses and relative odds versus absolute probability.

Five model and five vision forwards used physical GPU4:14.712 seconds in the
execution phase,21.028 seconds including load/preparation, and peak reserved
11,230,248,960 bytes. A shell log redirection initially failed because its parent
directory did not exist; it produced no process, attempt or model call. The
single actual attempt completed, and its PID is terminal. There was no training,
free continuation, additional donor or GPU replay for redundant metadata.

Runnable producer: `python -B -m probes.training_set_completion.recurrence_donor_tracking --selfcheck`.
The GPU entry refuses to overwrite its completed attempt. Scientific readback
uses the saved full tensors and the maintained producer's `response`,
`probability_readback` and `donor_metrics` functions; it needs no model reload.

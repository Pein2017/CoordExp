# C-Anchored Tail-Boundary Two-Step DDP Result

## Decision

`SCIENTIFIC_STOP_C_ANCHORED_TAIL_BOUNDARY`

The run is mechanically valid, but neither registered exposure recovers any
of the eleven selected targets at IoU50.  Both exposures retain every one of
C's 137 IoU50 owners, including all three protected owners.  Tail displacement
therefore removes the observed direct replacement failure on this panel, but
does not produce useful owner uptake at the frozen dose.

Authoritative reduction:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-tail-boundary-ddp-two-step/analysis-v3.json`

SHA-256:
`34f2865dc9857d9ab60b25da47419304af92dabaece14055664fe922ddccf3b1`.

`analysis-v1.json` and `analysis-v2.json` are preliminary reductions of the
same immutable run.  V3 adds the frozen visual and tail-ordering monitors and
is the sole authoritative interpretation.

## Materialization and execution

The CPU materializer extracted exact generated IDs from the frozen C trace,
required one final `<|im_end|>`, removed only that token, and placed the same
canonical annotated row after the remaining transcript.

- StateBank ID:
  `e3531e231ccb2c3bd3cd27a060971eb29fd9443301b62887ddfc8f6e340ddd04`;
- manifest SHA-256:
  `7ed2eac493f8656719f1559457b1e596771bf82a238374f1f2edff57a8091e57`;
- records SHA-256:
  `86377fadc327a46d308891e2c58f630339a49876a22203c4525cadacdf48c6d3`;
- receipt SHA-256:
  `ae330c3488f504c5068e0ed966b077c06370a8a7e551269846c134ae6d31ead3`;
- 24 records in two lexical windows, with exact `1/11` credit per image per
  update;
- longest `prompt + C tail + target` sequence: 1,598/12,000 tokens;
- image `398214` retained its registered one parser-dropped C row monitor;
- deterministic rematerialization reproduced manifest and records byte for
  byte.

The canonical two-rank BF16 run completed both updates with six microsteps per
rank and denominator 12 at each step.  Both gradients were finite and both
optimizer updates were applied.

| step | total loss | annotated-row loss | site gate | grad norm | LR |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.420952 | 3.405733 | 0.015219 | 12.880745 | 2.5e-6 |
| 2 | 3.373128 | 3.358960 | 0.014168 | 13.028287 | 2.5e-6 |

Run receipt SHA-256:
`378d46194c80ccf979aa45882dcd038a2399ad779261f9dee47f71e2576c9cae`.
Resolved-config fingerprint:
`bc8ebed8703ec77c7999de6733ea57764f48edf56a46db5abc959022deee676d`.

The checkpoints contain only the unmerged language-DoRA adapter plus an exact
identity-copy of the already-required frozen selected-token embedding delta:

- step 1 adapter fingerprint:
  `762574a47084141b157a7f3cea59291bc94837004f5a074300c73cf3e6146b41`;
- step 2 adapter fingerprint:
  `6e67c1d151173134b77b167352a76bba6c51824ca47cd2dd480905c8e57f5a64`;
- each adapter has 588 tensors and 18,006,016 trainable scalars;
- no base-weight merge or merged export occurred.

## Cold natural behavior

Both checkpoints were loaded in fresh inference processes with exact
saved-to-materialized adapter equality, `merged_adapters=[]`, frozen embedding
identity, HF greedy decoding, RP1, and the same twelve-image panel.

| read | IoU50 | IoU60 | IoU80 | selected IoU50 | C IoU50 retained | protected | predictions | duplicate candidates | parser drops | natural EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 137 | 127 | 77 | 0/11 | 137/137 | 3/3 | 204 | 1 | 1 | 12/12 |
| step 1 | 137 | 127 | 77 | 0/11 | 137/137 | 3/3 | 205 | 2 | 1 | 12/12 |
| step 2 | 137 | 126 | 78 | 0/11 | 137/137 | 3/3 | 207 | 3 | 0 | 12/12 |

There are no invalid, malformed, or capped rows.  At IoU60, step 2 loses one
non-target owner.  At IoU80, both steps exchange non-target owners; these are
reported monitors and do not alter the frozen IoU50 decision.

Natural ordering remains monitor-only.  The C/step1/step2 reads have
respectively 14/14/13 adjacent violations over 6/6/5 rows.  No transcript was
excluded or rewritten for ordering.

The selected-target result is zero in every annotation-quality and visual
difficulty stratum.  Hence neither the dense repeated book/produce/cup GT
exception nor the separate small/occluded-object monitor can rescue or weaken
the decision.

## Mechanism localization

### 1. Boundary displacement removed direct owner replacement

The first-crossing predecessor recovered the umbrella by replacing the
protected handbag at the same prefix.  Here both updated reads retain all 137
C owners and all three protected owners.  This is consistent with the boundary
displacement lemma: at the supervised tail state the target directly competes
with EOS, not an incumbent owner.

This is evidence for the local competition explanation, not a general
preservation theorem.  Shared DoRA still changes earlier tokens and can affect
stricter IoU thresholds.

### 2. Exact-state reach does not explain the whole failure

Step 1 exactly reproduces the complete supervised C tail on 6/11 selected
images; step 2 does so on 3/11.  Every one of those reachable states still
chooses EOS (`151645`) immediately, and no canonical target row is realized.
Thus natural prefix drift is one obstacle, but it is not a sufficient account:
the two target-CE updates also fail to cross the target-versus-EOS greedy
boundary at exact reachable tails.

For the remaining images, earlier token changes prevent exact visitation of
the teacher-forced tail state.  Tail supervision therefore also has a
long-horizon off-policy problem: its training state depends on reproducing the
entire C token transcript.

### 3. Tail placement conflicts with the learned ordering preference

Appending the canonical target after C creates at least one ordering inversion
on all 11 images: 112 prior-row/target pair inversions in total, ranging from 3
to 22 per image; the appended target is also an adjacent inversion on all 11.
This remains legal model behavior and is not a mechanical gate.  It does,
however, make tail continuation a poor scalable training construction for a
model with a strong `geo_sorted_xy` habit.

Prediction count rises `204 -> 205 -> 207`, but no added output matches a
selected owner and duplicate-candidate count rises `1 -> 2 -> 3`.  The observed
effect is continuation/duplicate mass rather than correct owner grounding.

## Interpretation and successor

Observation: tail-boundary CE is preservation-safe at IoU50 on this panel but
has zero selected-target uptake after both registered finite updates.

Inference: the experiment separates the two roles that isolated first-crossing
CE had conflated.  Moving the boundary avoids direct incumbent suppression,
but loses the spatially appropriate insertion state and requires a long exact
prefix.  More tail dose would not answer the registered mechanism question and
is not permitted.

The shortest justified successor is therefore **first-crossing two-row
continuation**:

\[
L = \frac{1}{11}\sum_i\left[
  \ell(u_i\mid p_i) + \ell(a_i\mid p_i,u_i)
\right],
\]

where `p_i` is the actual first-crossing prefix, `u_i` the annotated target
row, and `a_i` the displaced C row.  The first term uses the already proven
target-supporting boundary; the second trains the missing transition back to
the incumbent at a different prefix, avoiding impossible same-prefix dual
labels.

This successor still needs a truthful minimal event representation and must
handle the canonical-target versus naturally generated target-alias mismatch.
Do not build full suffix replay, a preservation regularizer, or an 8-GPU scale
path before the two-row vertical is mechanically and behaviorally tested.

## Claim boundary

This result establishes a valid same-panel negative for the frozen two-step
tail-boundary objective and supports a local boundary-competition mechanism.
It does not show that tail continuation can never work, that DoRA cannot learn
new owners, or that any method generalizes to held-out COCO.  Production and
scaling remain held.

---
title: A state-context-suffix transfer framework
description: Exact four-corner decomposition of checkpoint preference differences, with explicit limits on mechanistic interpretation.
type: investigation
role: analysis-framework
authority: non_normative_research
architecture_promotion_status: not_promoted
status: complete
updated: 2026-09-08
---

## Written before causal-transfer results

The proposed framework is a **functional decomposition**, not a trained model
of the LLM or a claim that it maintains an explicit covered set. It should
explain which tested intervention patterns contradict simple alternatives.
The [unit](unit.md) owns all populations, interventions and stopping rules.

For fixed image and prefix x, let S denote Source and O the overfit model.
At decoder depth l, the state is the whole prefix residual matrix H_l(x),
not merely the current prediction vector h_l(t). Write

`m(x) = M_suffix(H_l(x))`,

where m is the fixed endpoint-token logit margin, and the suffix includes
remaining decoder blocks, final norm and output head. The ordinary Logit Lens
instead reads h_l(t) directly through the final norm/head; these are different
functions, so late lens emergence need not imply late information formation.

## Four corners and an exact identity

Let `m[d,r]` be the margin after generating H_l with donor-prefix parameters d
and applying receiver-suffix parameters r. The full-state graft measures the
four corners SS, OS, SO, OO, with two native and two hybrid evaluations.
In this no-cache, post-DeepStack setting, the full-state graft implements a
prefix/suffix network splice without modifying parameter files. Identical
position handling and all prefix residual positions are part of the contract.

Define the native checkpoint difference `D = m[O,O] - m[S,S]`.

An equal-weight two-factor allocation is:

```
Upstream(l) = 0.5 * ((m[O,S] - m[S,S]) + (m[O,O] - m[S,O]))
Suffix(l)   = 0.5 * ((m[S,O] - m[S,S]) + (m[O,O] - m[O,S]))
Interaction(l) = m[O,O] - m[O,S] - m[S,O] + m[S,S]
```

`Upstream + Suffix = D` exactly. This is a symmetric/Shapley allocation for
these **two defined intervention factors**, not an assumption of additivity.
Interaction measures disagreement between the two possible intervention orders;
it is not a third independently additive term to sum with the other two.

For the aggregate normalized transfer fractions defined in the unit:

```
Upstream / D    = (R[O->S,full] + R[S->O,full]) / 2
Interaction / D = R[S->O,full] - R[O->S,full]
```

Compute these from sums of raw margins with a disclosed denominator, not by
averaging unstable per-site ratios. At final block 28, Upstream/D=1 is forced
by the common final norm/head and is not a scientific discovery. No clipping
or requirement of monotonicity across depth is justified.

## Where does context enter?

Let C be the same-margin current-position-only graft, with all other residual
positions retained from the receiver. Compare `R_full - R_current` at matched
depth, direction and input. This is the additional effect of giving downstream
attention donor **context states** when it already has the donor current state.
Context includes image, prompt and generated history, not just an owner ledger.
It is an operational conditional effect and need not be positive or additive.

Equal-norm random **delta** controls test whether current-vector displacement
size alone accounts for current-state effects. They do not provide a
scope-matched control for replacing an entire prefix matrix, and cannot certify
that the full-state graft remains on the natural distribution.

## Competing predictions and limitations

1. **Fixed-head alignment only:** lens preference changes do not reliably
   transfer before the final block, beyond same-size random perturbation.
   Strong nonfinal directed transfer would weaken this purely readout-only
   account, but would not rule out representational alignment as a contributor.
2. **Locally portable current decision state:** current-state graft approximates
   full-state graft in both directions across images. Large early context gaps
   or direction asymmetry contradict the simplest version.
3. **Distributed/contextual or co-adapted computation:** full/current gaps and
   four-corner interaction are appreciable and repeat across images. This
   supports a context/suffix-dependent account, not a unique symbolic mechanism.

The observable dynamics are depth-indexed functional transfer profiles, not
training-time dynamics and not recovered model-internal deliberation. Neither
successful nor failed cross-checkpoint grafts establish informational equality,
natural causal necessity, correct geometry, remaining-owner coverage or
generalization. Replication on Human13 is still replication on fitted images.

## Evidence status and a pilot-discovered limitation

The pilot and fixed replication now provide the four corners. The
independent reduction is in the output root's `pilot-analysis-v1/summary.json`.
At block 24, current overfit→Source transfer is 0.664 versus 0.080 mean random;
reverse transfer is 1.003 versus 0.610 mean random. These directional differences
require a scale caveat, not an immediate claim of superior Source robustness.

CPU inspection of saved pilot residuals finds mean Source/overfit state norms
1032/818 at block 24 and 2048/1468 at block 27. Donor-minus-receiver distance is
the same in both directions, but its size relative to the receiver is not.
Moreover, an isotropic random delta generally grows the resulting state norm,
whereas the real donor may have a smaller norm. Thus equal delta norm is not
equal resulting-state radius. Directed-vs-random effects here cannot isolate
radial scaling from angular/state-content effects. Full-state four-corner
identities remain valid; their semantic interpretation is correspondingly bounded.

Final synthesis must expose which predictions survived, which failed, and
what this framework cannot distinguish. No extra experiments are authorized
by a conceptual diagram; any scale-separation successor requires a separately
frozen contrast and must not change the active replication or its acceptance.

## Preference erasure is not donor-choice transfer

A second pilot finding makes full-vocabulary donor top-one frequency essential:
at block 24, current overfit→Source graft chooses the donor token in 8/12 sites,
whereas reverse graft does so in only 2/12 despite reverse R≈1.00. Most reverse
outcomes instead choose a third token. At block 27 these counts are 12/12 and
6/12. Therefore the two-token margin transfer metric alone cannot distinguish
restoring a donor decision from destroying the recipient preference. Both
donor top-one and third-token rates are now reduced from the already-retained
raw top-k fields in `pilot-analysis-v2`; this adds no new execution or estimand.
The exact four-corner algebra still describes this margin, not the whole
distribution. A directional result must be reported with this distinction.

## A candidate radius-direction dynamical parametrization

For a fixed text position, write the residual `h_l = r_l u_l`, where `r_l`
is its norm and `u_l` its unit direction. Let `f_l = h_(l+1) - h_l` be the
actual block update (including both attention and MLP effects). Exactly:

```
r_(l+1)^2 = r_l^2 + 2 r_l <u_l, f_l> + ||f_l||^2
u_(l+1)   = (r_l u_l + f_l) / r_(l+1)
```

When the update is small relative to r, angular change is approximately the
orthogonal component of f divided by r. Thus a late readout could arise from
progressive directional alignment, changing residual/update scale, or both.
This is a useful coordinate system, **not evidence that f is constant across
checkpoints**: here the adapters can change both update direction and magnitude.
The exact recurrence alone is a reparametrization, not an explanatory discovery.

The fitted-image data can test whether this description is useful by comparing
radius ratios, native angular updates and cross-checkpoint transfer at matched
depth. `pilot-geometry-v1` measures these from the previously saved complete
layer trajectory without new model execution. A future radius-only versus
direction-only graft would discriminate the causal roles: replace h_r by
`(||h_d||/||h_r||) h_r` or by `(||h_r||/||h_d||) h_d`. Those interventions were
not part of the present frozen causal-transfer/replication design and are not
claimed to have been tested.

## Final disposition

The [completed result](results.md) reports the image-equal 12-image replication.
Nonfinal overfit→Source current-state grafts transfer donor choices far more
often than the random controls, weakening a purely final-readout-only account.
Earlier full/current gaps and directional interaction contradict the simplest
context-free symmetric-portability model. Reverse large R frequently reflects
third-token preference erasure rather than donor choice, so margin-only
interpretation is rejected. State-radius differences repeat across the panel
but their causal contribution remains unresolved.

The framework is accepted as a bounded **functional analysis** of the executed
hybrids, not as a proven neural algorithm or explanatory architecture. The
radius-direction parametrization remains a falsifiable candidate extension.
No natural coverage/EOS or held-out-generalization claim is supported by it.

## Subsequent final discriminator — 2026-09-08

The separately authorized [radius-direction study](../2026-09-08-logit-lens-radius-direction/results.md)
has now tested the candidate extension above on the same Human13 prefixes.
Direction-only reproduces most full-current transfer; radius-only never selects
the donor endpoint at blocks 24/27, although radius has secondary effects.
Reverse third-token erasure persists with receiver radius preserved. Thus the
earlier unresolved radius caveat is narrowed to a secondary contribution, not
a radius-only explanation. The framework remains bounded functional analysis;
this final round closes the branch rather than promoting an architecture.

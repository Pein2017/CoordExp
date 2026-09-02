# Direct C-D0 mathematical notes

These statements justify the smallest direct pilot.  They do not establish
that D0 will generalize.

## 1. What partial COCO labels identify

Let `Z` be a true owner event, `R` its annotation indicator,
`pi(Z) = P(R=1 | Z)`, and `ell_theta(Z)` a model loss.  Then

\[
E[\ell_\theta(Z)\mid R=1]
=E[\ell_\theta(Z)]
+\frac{\operatorname{Cov}(\pi(Z),\ell_\theta(Z))}{E[\pi(Z)]}.
\]

This follows from
`E[ell | R=1] = E[pi ell] / E[pi]`.  Therefore ordinary observed-positive
training identifies annotation-conditional risk, but not full-owner risk when
annotation probability depends on density, size, occlusion, category, or
unobserved scene state.  Unknown unmatched rows cannot supply the missing
propensity or a valid negative class.

**Consequence.** C and D0 may be compared on the frozen observed-owner ledger.
Neither arm supports a full-scene precision, recall, hallucination, or
completion claim.

## 2. Whole-row soft minimum cannot splice coordinate tuples

For one event with `m` admitted complete rows and row losses `ell_a`, define

\[
\Phi_\tau
=-\tau\log\left(\frac1m\sum_{a=1}^m e^{-\ell_a/\tau}\right).
\]

Writing `ell_min = min_a ell_a` gives

\[
\ell_{min}\le\Phi_\tau\le\ell_{min}+\tau\log m.
\]

Moreover,

\[
\nabla\Phi_\tau
=\sum_a w_a\nabla\ell_a,
\qquad
w_a=\frac{e^{-\ell_a/\tau}}{\sum_b e^{-\ell_b/\tau}},
\qquad
\sum_a w_a=1.
\]

The gradient is a convex mixture of gradients of complete parser-valid rows.
It can favor an easier row, but it cannot create four independent
multiple-positive coordinate decisions.  With the predecessor's maximum
group size six and `tau=0.25`, the soft-min gap is at most
`0.25 log(6) = 0.448` loss units.

**Failure mode.** A multi-owner event may repeatedly favor its easiest owner.
That is measurable owner-exposure collapse, not proof that D0 is an unbiased
set likelihood.

## 3. Exact DDP global event means

For objective family `k`, let rank `r` contribute differentiable numerator
`S[r,k]` and eligible count `n[r,k]`.  Let

\[
N_k=\sum_{r=1}^R n_{r,k}.
\]

If rank `r` passes

\[
L_{r,k}=\frac{R S_{r,k}}{N_k}
\]

to an `R`-rank DDP whose gradients are averaged, then

\[
\frac1R\sum_r\nabla L_{r,k}
=\frac{\sum_r\nabla S_{r,k}}{N_k}.
\]

This equals the single-process global eligible-event mean even when ranks have
unequal counts or a rank has zero local positives.  Rank-local means generally
do not.  The result requires a shared planned-step denominator, fixed
collective order, graph-connected zero on empty local ranks, and failure when
`N_k=0`.

## 4. A frozen actual-prefix bank is a surrogate, not policy gradient

For an on-policy prefix distribution `q_theta`, define

\[
J(\theta)=E_{h\sim q_\theta}[\Phi_\theta(h)].
\]

Its formal gradient contains two terms:

\[
\nabla J
=E_{q_\theta}[\nabla\Phi_\theta(h)]
+E_{q_\theta}[\Phi_\theta(h)\nabla\log q_\theta(h)].
\]

The fixed-Source D0 phase freezes `q` at the Source and optimizes only the
first term.  It is a bounded imitation/correction surrogate on states the
Source actually visited.  It is neither an unbiased policy-gradient estimator
nor evidence that its prefixes remain reachable after the update.

**Consequence.** Fresh natural greedy decode is the decision surface.  A rise
in grouped teacher-forced margins without a D0-over-C natural gain is proxy
mismatch.  Only after the fixed-bank direct test is positive is a rollout
refresh worth testing.

## 5. What the paired pilot can discriminate

The matched exposure construction fixes Source, train images, per-image
presentation multiplicity, trainable DoRA surface, optimizer, effective batch,
and update count.  It intentionally does not equate target-token count: C
scores a complete transcript while D0 scores selected positive and
preservation events.  Therefore a positive result supports the bundled D0
recipe over C; it does not isolate actual-prefix conditioning from grouped-row
selection or preservation.

With 891 screen-dev owners, a `0.5` percentage-point effect is discrete:

\[
4/891=0.449\%,\qquad 5/891=0.561\%.
\]

Five net D0-over-C IoU50 owner matches are therefore the smallest registered
advance result.  A single seed can reject or authorize paired replication; it
cannot establish an algorithm-level theorem or population claim.

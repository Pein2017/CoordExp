# Handoff: 256-image multiple-positive owner-set supervision

## Objective

Implement and run a fast 256-image treatment screen that rewards order-independent coverage of physical object instances.

The central rule is:

> At a given rollout state, every verified uncovered physical owner is valid. Training must not designate one geometry-sorted owner, one sampled owner, or one complete row as the unique teacher target.

Here, “owner” means one physical object instance, not merely a class label.

## Correction to the previous experiment

The completed breadth experiment did **not** test this hypothesis.

Its active profile was:

```text
sampled_path_and_source_route_imitation_only
```

Each training event contained exactly one positive complete row, trained with exact-token teacher-forced cross entropy. Broadening image count did not solve owner exchange, but that result only rejects “more images will repair single-owner row imitation.”

It does **not** reject multiple-positive, order-independent set supervision.

Relevant evidence:

- [Research unit](/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/unit.md)
- [Results](/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/results.md)
- [Single-row training adapter](/data/CoordExp/.worktrees/research-probes/src/training/rollout_calibration.py)
- [Single-row loss](/data/CoordExp/.worktrees/research-probes/src/losses/rollout_calibration.py)

The existing `grouped_entity_transition_preference` function may be reusable, but it is only a local next-branch comparison. By itself, it is not a final owner-set objective.

## Training cohort

Use 256 training images for rapid iteration.

Select images that have:

- reliable physical-owner annotations;
- multiple objects, preferably dense scenes;
- variation among the existing sampled trajectories;
- at least one owner found only in some trajectories;
- enough valid trajectory comparisons to produce a nonzero learning signal.

Reuse the existing `1 greedy + 16 sampled trajectories per image` where valid. Do not recollect unless the stored artifacts are insufficient.

Do not train on the 12 manually refined validation images. Preserve them as a high-quality evaluation panel.

## Owner-set construction

For image \(I\), let:

\[
O(I)=\{\text{verified physical owners in the image}\}.
\]

For an actual rollout prefix \(P_t\), match already emitted rows to physical owners and define:

\[
C_t=\{\text{verified owners already covered by the prefix}\},
\]

\[
U_t=O(I)\setminus C_t.
\]

Every owner in \(U_t\) is a valid next owner.

The training record must therefore contain:

```text
image
actual rollout prefix
covered_owner_ids
remaining_owner_ids
multiple candidate continuations
row-to-owner matches for every continuation
final unique-owner set for every continuation
duplicate / invalid / verified-false / unknown labels
```

It must not contain a single `selected_owner_id` or `target_row` that determines the entire loss.

Unknown or potentially unlabeled predictions receive no negative gradient. Entity correctness and geometry quality must remain separate.

## Recommended first objective

Use grouped, whole-trajectory set supervision.

For each image or shared rollout state, collect candidate continuations:

\[
\mathcal T=\{\tau_1,\ldots,\tau_K\}.
\]

Map every trajectory to its unique trusted owner set:

\[
S(\tau_k)\subseteq O(I).
\]

Define strict preference using set inclusion:

\[
\tau_a \succ \tau_b
\quad\text{only if}\quad
S(\tau_b)\subset S(\tau_a),
\]

provided that \(\tau_a\) is not worse in verified hallucinations, invalid rows, or severe duplication.

This rule is important:

- A trajectory that finds \(A,B,X\) dominates one that finds only \(A,B\).
- A trajectory that finds \(A,X\) and another that finds \(A,B\) are incomparable.
- Do not reward the first by treating the second as negative; that would train owner exchange.
- Keep all nondominated valid trajectories as multiple positives. Do not select one winner.

A minimal grouped loss can raise the total model probability assigned to all positive trajectories relative to dominated trajectories:

\[
\mathcal L_{\mathrm{set}}
=
-\log
\frac{
\sum_{\tau\in\mathcal T^+}\exp s_\theta(\tau)
}{
\sum_{\tau\in\mathcal T^+}\exp s_\theta(\tau)
+
\sum_{\tau\in\mathcal T^-}\exp s_\theta(\tau)
}.
\]

Here \(s_\theta(\tau)\) should be a length-controlled or frozen-reference-relative trajectory score, so longer valid enumerations are not mechanically penalized by having more tokens.

A pairwise equivalent is acceptable:

\[
\mathcal L_{\mathrm{pair}}
=
\sum_{\tau^+\succ\tau^-}
\operatorname{softplus}
\left(
m-s_\theta(\tau^+)+s_\theta(\tau^-)
\right).
\]

Use every valid dominance pair, not one selected pair.

Token-type gating may remain as a stability mechanism, but it must not convert the objective back into exact CE on one selected owner row.

## Reward semantics

The primary quantity is unique physical-owner set expansion:

\[
R_{\mathrm{owner}}(\tau)=|S(\tau)|.
\]

Safety terms are secondary:

\[
R(\tau)
=
R_{\mathrm{owner}}(\tau)
-\lambda_d N_{\mathrm{duplicate}}
-\lambda_i N_{\mathrm{invalid}}
-\lambda_f N_{\mathrm{verified\ false}}
-\lambda_s\mathbf 1[\mathrm{STOP\ while\ trusted\ owners\ remain}].
\]

Do not penalize unreviewed unmatched predictions as hallucinations.

Prefer strict set-dominance comparisons over relying solely on this scalar reward. Set dominance prevents a newly found owner from hiding the loss of an old owner.

## Required smoke tests

Before GPU training, prove mechanically that:

1. Two trajectories covering the same owners in different orders receive equivalent supervision.
2. Finding any previously uncovered owner receives credit.
3. Repeating an already covered owner does not increase set reward.
4. A trajectory that adds an owner without losing existing owners dominates its subset.
5. Two trajectories that exchange different owners are incomparable, not positive versus negative.
6. Unknown/unreviewed objects are neutral.
7. Premature stopping is harmful only when trusted uncovered owners remain.
8. The resolved training configuration and logs show the set-level loss is active.
9. No training record silently collapses to one positive candidate row.

## Evaluation

Compare:

```text
Source checkpoint
vs
new 256-image set-level treatment
```

The completed single-owner treatment can remain a historical baseline; do not retrain it unless required for compatibility.

Evaluate free greedy rollout, not only teacher-forced likelihood.

Primary outputs:

```text
retained Source owners
gained owners
lost owners
net owner-set expansion
greedy unique-owner recall
gap between greedy coverage and 16-sample union coverage
```

Safety outputs:

```text
duplicate owners
invalid rows
verified entity hallucinations
semantic errors
geometry quality
natural termination
rollout length
```

A longer sequence alone is not improvement.

## Interpretation

- Training reward improves and greedy coverage improves without owner loss: promising; proceed to one refreshed rollout round and possibly 1,024 images.
- New owners increase but old owners decrease: owner exchange remains; inspect whether incomparable trajectories were incorrectly ranked or whether the loss lacks preservation.
- Stored trajectories contain few strict superset relations: the data does not contain enough constructive examples. Recollect or improve exploration; do not fall back to unique-row CE.
- Training objective improves but free rollout returns to Source behavior: offline state-distribution mismatch. Generate trajectories from the treated checkpoint and repeat one mixed refresh round.
- Only next-row margins improve while final unique-owner set does not: local multiple-positive supervision is insufficient; preserve the trajectory-level objective.
- No improvement even on the 256 training images: treat it as an optimization or implementation failure before blaming model capacity or dataset breadth.

## Execution request

1. Update or create the research unit with this exact scientific contract.
2. Keep the completed breadth experiment immutable as historical evidence.
3. Implement the smallest grouped trajectory dataset and set-level loss path.
4. Run an 8–16 image real smoke.
5. Run the 256-image screen on 8 GPUs.
6. Analyze owner gain, retention, and loss before deciding whether to scale.

The non-negotiable contract is:

> All verified uncovered owners are valid; no canonical next owner exists; no unique positive row is teacher-forced as the primary objective; supervision rewards the final unique physical-owner set, independent of traversal order.
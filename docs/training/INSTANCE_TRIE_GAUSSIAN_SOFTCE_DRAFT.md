---
doc_id: docs.training.instance-trie-gaussian-softce-draft
layer: docs
doc_type: implementation-draft
status: active-implementation-draft
domain: training
summary: Active implementation draft for instance-aware multi-positive Gaussian coordinate SoftCE in compact recursive detection.
updated: 2026-05-18
---

# Instance-Trie Gaussian SoftCE Draft

Status: active implementation draft. The objective is implemented on feature
branch `codex/instance-trie-gaussian-softce` in isolated worktree
`/data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/instance-trie-gaussian-softce`,
but it is not merged, stable, or current production behavior. Treat this as
feature-branch provenance until focused tests, target-shape audit, smoke
workflow, diagnosis/audit review, and branch acceptance complete.

Naming convention: this draft uses the human-facing objective name
`Instance-Trie Gaussian SoftCE`, the config `target_distribution`
`instance_trie_gaussian`, focused config suffixes such as
`instance_trie_focused_cap8_frac0p04_mix0p1`, and the numeric metric flag
`recursive_detection_ce/coord_soft_ce/is_instance_trie_gaussian`. New
code/config/docs for this objective should not introduce `*_v0`, `v_*`, or
version-prefixed names. Existing historical names such as `iou_gibbs_v0` may
still appear when referring to already-implemented negative-result paths.

## Motivation

The A5/A6 IoU-Gibbs coordinate softCE experiments showed a failure mode where
direct IoU/CIoU-shaped supervision can become too diffuse. The model can learn
oversized boxes that cover multiple nearby or same-description instances rather
than sharply separating object instances.

The replacement direction keeps the useful part of regression-style coordinate
supervision, but changes the support unit:

- description ambiguity uses token-level support/balance
- coordinate ambiguity uses active-branch remaining instance candidates
- each object instance contributes a localized bbox-aware Gaussian peak

The target should be multi-positive without making the union of several objects
implicitly valid.

## Token Group Contract

The compact recursive detection loss should explicitly separate token groups.

| Token group | Examples | Draft supervision |
|---|---|---|
| Schema/control | fixed wrapper, boundary, and protocol tokens | hard CE plus struct/eos type gate |
| Description/free text | object description tokens and semantic entry-choice branches | hard CE plus trie support/balance at description branch positions plus desc type gate |
| Coordinates | `<|coord_*|>` x1/y1/x2/y2 tokens | coord type gate plus coord-vocab Gaussian softCE or hard CE |

Important nuance: a special token such as `<|box_start|>` may occur at the same
serialized prefix depth where another object description could continue, for
example `car <|box_start|>` versus `cart ...`. The orthogonal objective keeps
that boundary token structural: the teacher path receives hard CE for
`<|box_start|>`, while description-continuation ambiguity remains scoped to
free-text description positions.

## Orthogonal Token-Type Supervision

The active feature-branch objective separates token-type validity from
coordinate smoothness. Schema/control/boundary tokens use hard CE plus the
struct/eos type gate. Free-text description tokens use hard CE and, at
description trie branch positions, support/balance trie CE plus the desc type
gate. Coordinate tokens use the coord type gate for coordinate-token
exclusivity and `instance_trie_gaussian` SoftCE over the 1000 coordinate-token
vocabulary for smooth coordinate supervision.

Coordinate SoftCE should not be interpreted as a full-vocabulary gate. Its
softmax scope is the coordinate-token vocabulary; full-vocabulary leakage is
owned by `objective.type_gate`.

## One-Forward-Pass Instance Trie

The existing recursive target builder already materializes a hard trie over the
remaining serialized object entries. That trie is the right ownership surface
for candidate discovery, but this objective should not simply reuse the current
exact trie node as the coordinate candidate set for every slot.

Critical distinction:

- candidate ownership is the active desc/object-ref branch among remaining,
  not-yet-emitted instances
- candidate posterior weighting is a soft function of previous teacher-forced
  coordinates

The candidate set must come from recursive target construction sidecars, not
from all image objects and not from loose decoded-string matching. If the
current sidecar only exposes exact-prefix trie descendants at `y1`, `x2`, or
`y2`, implementation should add the smallest explicit coordinate-block sidecar
that preserves the active semantic-branch candidates for all four coordinate
slots.

The sidecar snapshot point is after the teacher path has selected the
object-entry semantic branch and reached the coordinate block, including
`<|object_ref_start|>`, description tokens, and `<|box_start|>`, but before any
coordinate token is consumed. The sidecar should be attached unchanged to
`x1/y1/x2/y2`; the posterior update decides how strongly each candidate should
contribute at each slot.

The desired one-forward-pass behavior is:

```text
before x1:
  candidates = remaining instances compatible with the selected desc path

after teacher-forced x1:
  same semantic-branch candidates, softly reweighted by compatibility with x1

after teacher-forced x1,y1:
  same semantic-branch candidates, softly reweighted by compatibility with x1,y1

after teacher-forced x1,y1,x2:
  same semantic-branch candidates, softly reweighted by compatibility with x1,y1,x2
```

This is the one-forward-pass approximation. It does not duplicate the batch for
every possible object continuation. Instead, the forward pass follows the
teacher-forced path while the target at each coordinate position is a Gaussian
mixture over semantic-branch candidate instances, with candidate weights updated
from the teacher-forced coordinate prefix.

This gives the desired "carry the ambiguity forward" behavior:

```text
A: x1=100 -> y1=100 -> x2=200 -> y2=200
B: x1=100 -> y1=100 -> x2=350 -> y2=260
C: x1=500 -> y1=100 -> x2=620 -> y2=220
```

At `x1`, all three candidates can contribute peaks. If the teacher-forced path
emits `x1=100`, candidate C receives a low posterior weight rather than being
removed by a hard exact-token filter. Ambiguity between A and B remains through
`y1`, then moves to `x2`. The model is not rewarded for assembling a box such
as `[100,100,620,260]`, because high probability requires staying compatible
with a coherent candidate path.

## Coordinate Target Formula

For a coordinate-token position with slot `t in {x1, y1, x2, y2}`, let `C` be
the remaining candidate instances attached to the active desc/object-ref branch
for the current coordinate block.

Each candidate `j` has bbox:

```text
b_j = (x1_j, y1_j, x2_j, y2_j)
w_j = x2_j - x1_j
h_j = y2_j - y1_j
```

Use a focused, axis-aware R95 radius:

```text
R95_x,j = floor(min(gaussian_r95_cap_bins, gaussian_r95_axis_fraction * w_j))
R95_y,j = floor(min(gaussian_r95_cap_bins, gaussian_r95_axis_fraction * h_j))

sigma_x,j = R95_x,j / 1.96
sigma_y,j = R95_y,j / 1.96
```

Then:

```text
x1 and x2 use sigma_x,j
y1 and y2 use sigma_y,j
```

If `R95_axis = 0`, that candidate contributes an exact one-hot component at its
candidate coordinate for the current slot. The same exact rule applies to prefix
compatibility: a previous teacher-forced coordinate either matches the candidate
coordinate exactly or gives that candidate zero posterior weight.

The initial objective uses uniform candidate priors:

```text
pi_j = 1 / |C|
```

Sidecar probabilities may be useful for legacy behavior or diagnostics, but
`instance_trie_gaussian` should use uniform priors unless a future ablation
explicitly changes that contract.

For coordinate bin `k`, candidate `j` contributes:

```text
q_j,s(k) proportional to exp(-0.5 * (k - mu_j,s)^2 / sigma_axis,j^2)
```

where `mu_j,s` is the candidate coordinate for the current slot.

Structural legality is still enforced:

```text
x1: k < x2_j
x2: k > x1_j
y1: k < y2_j
y2: k > y1_j
```

This legality mask is not the smoothing policy. It prevents impossible boxes.
The smoothing policy is the focused R95 rule above: default
`gaussian_r95_axis_fraction=0.04`, `gaussian_r95_cap_bins=8`, and
`gaussian_mixture_weight=0.1`. This replaces the earlier wide
`sigma = sqrt(axis + 1)` draft behavior.

Normalize each current-slot candidate target distribution over the resolved
coordinate vocabulary, then compute the slot posterior from previous
teacher-forced coordinates.

The current-slot target uses a normalized distribution:

```text
q_j,t(k) =
  legal_j,t(k) * exp(-0.5 * (k - c_j,t)^2 / sigma_j,t^2) / Z_j,t
```

The prefix compatibility used in `alpha` is deliberately **unnormalized**
mismatch energy:

```text
compat_j,s =
  exp(-0.5 * (c_s^teacher - c_j,s)^2 / sigma_j,s^2)
```

There is no `1 / sigma` or discrete-normalizer term in prefix compatibility.
This preserves uniform instance priors when two candidates share the previous
coordinate exactly but have different bbox sizes.

The normalized posterior is:

```text
alpha_j^(t) =
  softmax_j(log pi_j + sum_{s<t} log compat_j,s)
```

Then mix the current-slot Gaussian peaks:

```text
q_t(k) = sum_j alpha_j^(t) * q_j,t(k)
```

The executable prefix rule is:

```text
x1: use no teacher coordinates
y1: use x1 only
x2: use x1,y1 only
y2: use x1,y1,x2 only
```

No current or future coordinate may influence the posterior for a slot.

The coordinate loss is pure soft cross-entropy:

```text
L_coord = - sum_k q_s(k) * log p_theta(coord_token_k | prefix_t)
```

The coordinate loss must not use the recursive support/balance coefficients.
Those coefficients remain for description/entry-choice token ambiguity.

## Why This Avoids The Union Basin

The unsafe marginal target would build independent mixtures:

```text
q_x1 from every object
q_y1 from every object
q_x2 from every object
q_y2 from every object
```

That can reward coordinate recombinations across different instances.

The draft objective instead uses:

```text
active semantic branch -> coherent object instances -> soft posterior from coord prefix
```

The candidate posterior sharpens as the teacher-forced coordinate prefix
advances, so ambiguous coordinates stay multi-positive only while they remain
compatible with a coherent object-entry path. Because this objective uses
focused Gaussian tails over structurally legal bins, this should be read as
avoiding a high-probability union basin, not as assigning mathematically zero
probability to every recombination coordinate when the radius is nonzero.

## Feature-Branch Config Surface

The active feature-branch successor configs are:

- main: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1.yaml`
- slope ablation: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_instance_trie_focused_cap8_frac0p06_mix0p1.yaml`
- strength ablation: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p2.yaml`

The public focused-policy surface is:

```yaml
objective:
  coord_soft_ce:
    enabled: true
    target_distribution: instance_trie_gaussian
    gaussian_mixture_weight: 0.1
    gaussian_r95_axis_fraction: 0.04
    gaussian_r95_cap_bins: 8
```

Stale knobs from older coordinate-softCE variants must be rejected, including
`tau`, `tau_source`, `weighting`, `replace_coord_hard_ce`,
`apply_to_multi_positive`, `sigma`, `truncate`, `target_sigma`, and
`target_truncate`.

Runtime should resolve the coordinate token id range from
`token_rows.groups.coord_geometry`, as the existing latest recursive detection
surface does for previous coordinate softCE experiments.

The coordinate value domain is the resolved coordinate-token vocabulary,
currently 1000 bins corresponding to `<|coord_0|>` through `<|coord_999|>`.
Do not assume a 1001-bin inclusive `[0,1000]` value domain.

The old IoU/CIoU-Gibbs configs should remain as negative-result provenance, but
should not be presented as the recommended A5 direction for the active
feature-branch implementation draft.

## Implementation Infrastructure Boundaries

The implementation should include a small infrastructure cleanup, limited to
surfaces that protect objective correctness:

- keep Gaussian target construction, posterior weighting, structural legality,
  coordinate-vocabulary mapping, fp32/log-space math, diagnostics, and
  coord-vocabulary coordinate SoftCE in a pure coordinate target module
- add an explicit candidate-only `coord_instance_candidates` sidecar for active
  semantic-branch remaining-instance candidates; the sidecar carries instance
  identity and bbox geometry, while the current coordinate slot comes from the
  `TokenTarget`/loss context
- keep legacy `coord_soft_targets` metadata separate for old IoU/CIoU-Gibbs
  provenance; `instance_trie_gaussian` must not fall back to it
- resolve coordinate value bins from token-row geometry rather than hard-coding
  a `[0,1000]` convention
- run a checked-in no-training target-shape audit command before any tiny smoke
  training
- keep coordinate metrics in candidate/vocab/target terminology and reserve
  support/balance terminology for description and entry-choice positions
- use a resolved-config diff whitelist to preserve fair comparison against
  `compact_full_support2.yaml`

Do not use this change to rewrite unrelated recursive detection, trainer,
inference, evaluation, or artifact pipelines.

## Metrics Needed

The implementation should report coordinate-specific diagnostics separately
from recursive text support/balance metrics:

```text
recursive_detection_ce/coord_soft_ce/config_enabled
recursive_detection_ce/coord_soft_ce/candidate_count
recursive_detection_ce/coord_soft_ce/gaussian_mixture_weight
recursive_detection_ce/coord_soft_ce/gaussian_r95_axis_fraction
recursive_detection_ce/coord_soft_ce/gaussian_r95_cap_bins
recursive_detection_ce/coord_soft_ce/target_entropy
recursive_detection_ce/coord_soft_ce/target_std
recursive_detection_ce/coord_soft_ce/target_r95_radius
recursive_detection_ce/coord_soft_ce/target_peak_prob
recursive_detection_ce/coord_soft_ce/effective_coord_bin_count
recursive_detection_ce/coord_soft_ce/coord_vocab_bin_count
recursive_detection_ce/coord_soft_ce/x1/effective_candidate_count
recursive_detection_ce/coord_soft_ce/y1/effective_candidate_count
recursive_detection_ce/coord_soft_ce/x2/effective_candidate_count
recursive_detection_ce/coord_soft_ce/y2/effective_candidate_count
recursive_detection_ce/coord_soft_ce/x1/posterior_entropy
recursive_detection_ce/coord_soft_ce/y1/posterior_entropy
recursive_detection_ce/coord_soft_ce/x2/posterior_entropy
recursive_detection_ce/coord_soft_ce/y2/posterior_entropy
recursive_detection_ce/coord_soft_ce/x1/posterior_top1
recursive_detection_ce/coord_soft_ce/y1/posterior_top1
recursive_detection_ce/coord_soft_ce/x2/posterior_top1
recursive_detection_ce/coord_soft_ce/y2/posterior_top1
recursive_detection_ce/coord_soft_ce/x1/target_r95_radius
recursive_detection_ce/coord_soft_ce/y1/target_r95_radius
recursive_detection_ce/coord_soft_ce/x2/target_r95_radius
recursive_detection_ce/coord_soft_ce/y2/target_r95_radius
```

`target_distribution` is string provenance and should live in resolved config,
manifests, and artifacts, not in numeric trainer metric payloads. If a flat
numeric trainer indicator is useful, use a boolean-style key such as
`recursive_detection_ce/coord_soft_ce/is_instance_trie_gaussian`.

Coordinate diagnostics should avoid overloading `support` terminology where
possible. Use candidate/vocab/target wording for coordinate mixtures; reserve
support/balance wording for description and entry-choice positions.

Healthy expectation:

- `x1` may have high candidate count and posterior entropy for repeated
  same-description objects
- later coordinates should usually sharpen as the teacher-forced prefix
  disambiguates candidates
- entropy that remains high through `y2` means the target may be too diffuse
- immediate collapse for near-shared vertices means the posterior update is too
  hard or the candidate sidecar is exact-prefix filtered

Group-level loss accounting is required before production-scale ablation:

```text
recursive_detection_ce/loss_group/schema
recursive_detection_ce/loss_group/desc
recursive_detection_ce/loss_group/coord
```

## Failure Policy

For `instance_trie_gaussian`, missing semantic-branch candidate metadata is
a hard error. Runtime must not silently fall back to selected-instance Gaussian
or hard CE while the config claims instance-aware supervision.

The runtime must also hard-error if:

- a coordinate target has only legacy exact-prefix `coord_soft_targets` but no
  semantic-branch `coord_instance_candidates`
- the teacher object is missing from `coord_instance_candidates`
- more than one candidate has the teacher `object_instance_id`
- any candidate bbox is outside the resolved coord vocabulary or violates
  `x1 < x2` / `y1 < y2`
- the teacher coordinate token id is outside the resolved coord-token id range

The implementation audit must first answer whether the current recursive target
sidecar exposes the active desc/object-ref branch candidate set for each
coordinate block. If not, add the smallest explicit sidecar during target
construction and collation. Do not infer candidates from decoded text.

## Non-Goals For The Initial Objective

- no IoU/CIoU energy target
- no Gaussian support-window truncation beyond structural legality
- no temperature, raw sigma multiplier, or per-dataset threshold knobs
- no decoded-box loss
- no W1 auxiliary objective
- no duplicated candidate-branch forward pass
- no production launch before explicit user approval
- no oracle decoding, reranking, confidence post-op headline metric, or
  GT-candidate-aware inference constraint for this training ablation

## Acceptance Invariants

The implementation plan must enforce these invariants with tests:

1. At `x1`, same-description candidates with different x1 values create
   multiple localized peaks.
2. After the teacher-forced `x1`, incompatible candidates are softly
   downweighted at `y1`, not necessarily removed by exact-token filtering.
3. If candidates share or nearly share `x1,y1`, ambiguity continues to `x2`.
4. A large bbox produces a broader per-candidate coordinate distribution than a
   tiny bbox on the same axis.
5. Coordinate positions use pure softCE even when the global recursive
   support/balance weights are `2.0/1.0`.
6. Description/free-text multi-positive positions still use support/balance.
7. Schema/control positions that are not entry-choice branches remain hard CE.
8. The target is normalized, finite, and structurally legal over the resolved
   coordinate vocabulary.
9. Candidate sets are semantic-branch conditioned and never assembled from all
   image objects or decoded-string matching.
10. Candidate sets contain remaining, not-yet-emitted instances only.
11. Prefix compatibility uses unnormalized Gaussian mismatch energy, while
    current-slot coordinate targets are normalized over legal coordinate bins.
12. Posterior computation never peeks at the current or future coordinate slot.
13. Coordinate SoftCE uses a coordinate-vocabulary softmax; non-coordinate
    token leakage is penalized by `objective.type_gate`.

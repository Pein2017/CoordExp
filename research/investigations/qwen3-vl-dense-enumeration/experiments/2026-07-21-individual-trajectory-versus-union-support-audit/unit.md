---
title: Individual-Trajectory versus Sampled-Union Object-Support Audit
description: Determine whether bagging-only object discoveries already occur in a better complete sampled trajectory or exist only across the union of complementary trajectories.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-21-individual-trajectory-versus-union-support-audit
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: route_local_conservative_certificates
updated: 2026-07-21
---

# Individual-Trajectory versus Sampled-Union Object-Support Audit

## Question

When repeated low-temperature rollouts discover physical objects that greedy
decoding misses, does at least one complete sampled trajectory already cover
more verified objects than greedy under the same row budget, or are the extra
objects available only across the union of mutually complementary
trajectories?

This unit is a diagnostic gate. It decides whether the next treatment should
distill a better complete trajectory, teach local remaining-object completion,
or return to visual-support accessibility. It does not train a model.

## Terminology

- **Physical owner**: one specific visible Common Objects in Context
  80-category (`COCO-80`) entity. Multiple boxes for the same entity share one
  physical owner even when their categories or boundaries disagree.
- **Fixed row budget**: the first `B` complete generated object rows used to
  compare every trajectory. Results are reported separately at `B = 4`, `8`,
  `16`, and `32` so a longer rollout cannot silently define improvement. A
  trajectory that naturally terminates before `B` contributes all of its
  complete rows and is not missing; `B` is a maximum evaluation horizon, not
  an enforced minimum length.
- **Greedy Unique-Owner Coverage (`C_g(B)`)**: the number of verified physical
  owners matched by greedy decoding within the first `B` complete rows.
- **Best Single Sampled-Trajectory Coverage (`C_best(B)`)**: the greatest
  verified unique-owner coverage achieved by any one sampled trajectory for
  the same image and fixed row budget.
- **Sampled-Trajectory Union Coverage (`C_union(B)`)**: the number of verified
  owners found at least once across all sampled trajectories within the same
  per-trajectory row budget. Repeated hits on one owner count once.
- **Individual-trajectory gain**: `C_best(B) - C_g(B)`. A positive value means
  at least one sampled trajectory demonstrates a better complete route than
  greedy.
- **Union-only gain**: `C_union(B) - C_best(B)`. A positive value means useful
  owners are distributed across trajectories beyond what the best single
  trajectory demonstrates.
- **Development and validation cohort**: images that may guide mechanism
  diagnosis, treatment selection, matching rules, and early stopping but never
  contribute training gradients. They are not a final blind evaluation set.
- **Sampled-only owner**: a verified physical owner present in
  `C_union(B)` but absent from `C_g(B)` at the same fixed row budget.
- **Harmful-row count (`H(B)`)**: the number of entity hallucinations,
  semantic or category errors, duplicate physical owners, and malformed row
  attempts emitted before the cutoff immediately after the `B`-th complete
  row, or before natural termination when fewer than `B` complete rows exist.
  A real owner with the correct category but imperfect geometry is not a
  harmful row; its geometry error is reported separately.

## Motivation and Upstream Boundary

Prior units establish that repeated full-image sampling exposes real object
support, that prefix order and complete rows can redirect later trajectories,
and that sampled novelty does not automatically have positive downstream set
value. They do not establish whether the current bagging advantage is mainly:

1. a route-selection failure, where greedy chooses a weaker complete route;
2. a route-composition failure, where no one sampled route contains the useful
   union; or
3. an object-support failure, where the sampled distribution also fails to
   expose the missing owners.

This distinction must precede trajectory-level training. Positive trajectory
imitation has a demonstrated target only when a better complete sampled route
exists. A union of complementary routes is not itself one autoregressive
training trajectory.

The twelve human-refined dense validation images were declared blind in an
earlier unit. The user has now explicitly reclassified them for this new
research question as a high-quality development and validation cohort. This
does not alter the historical interpretation of earlier units. From this unit
forward, these images may influence research-route selection but must never be
used for training. A new blind dense cohort will be created only after a
treatment and its thresholds are frozen.

## Competing Explanations and Predictions

| Explanation | Expected observation | Next treatment if supported |
|---|---|---|
| Greedy selects a weaker route even though a better complete sampled route exists. | `C_best(B) > C_g(B)` on multiple images without a matching rise in verified unsupported, malformed, or duplicate rows. | A separate positive-only weighted trajectory self-imitation unit with a within-image shuffled-reward control. |
| Useful owners are distributed across complementary routes, but no one sampled route dominates greedy. | `C_union(B) > C_g(B)` while `C_best(B)` remains near `C_g(B)`. | A separate local remaining-object completion unit at natural prefixes; do not imitate whole trajectories. |
| Bagging gain is mainly extra length, duplicate spray, or matching error. | Apparent gains vanish at fixed row budget, under one-to-one owner matching, or after crop-assisted review. | Reject the training premise and repair the evidence or matching rule. |
| Missing objects remain outside the sampled support of the current policy. | `C_union(B)` remains near `C_g(B)` despite valid sampling and review. | Return to visual-support accessibility, object-specific control-state synthesis, or another perception-to-decoder discriminator. |
| Entity discovery exists but physical extent estimation is weak. | Sampled trajectories find correct owners while owner-conditioned box geometry remains poor or contaminated. | Keep entity support and geometry treatment separate; do not reject discovery solely because the box is imperfect. |

## Strongest Confounds and Controls

1. **Longer-output confound**: compare trajectories only at the same fixed row
   budgets and report all four budget points.
2. **Duplicate-spray confound**: use global one-to-one owner matching; extra
   predictions for a matched owner do not increase coverage.
3. **Incomplete-annotation confound**: use generation-7 human-refined labels
   and crop-assisted review for unresolved predictions. Official mismatch is
   not hallucination evidence.
4. **Geometry-as-entity confound**: report owner discovery and geometry quality
   separately. A real owner with imperfect geometry remains a discovery but
   cannot be called a correct box.
5. **Category-only clustering confound**: assign physical owners before
   aggregating category frequencies. Same-category boxes in a dense scene are
   not automatically duplicates.
6. **Repetition-penalty confound**: use repetition penalty `1.0` for the
   primary result. The historical `1.10` heuristic is outside the primary
   comparison.
7. **Sampling-budget confound**: use the same sixteen sampled trajectories per
   image and immutable seeds. Do not extend only images with an attractive
   preliminary result.
8. **Checkpoint-family confound**: use only the geometry-sorted pure
   cross-entropy checkpoint in this unit. Random-order and Gaussian-coordinate
   checkpoints are later replications, not concurrent factors.

## Primary Observation

For every image and fixed row budget, produce:

```text
C_g(B)
C_best(B)
C_union(B)
individual-trajectory gain
union-only gain
best sampled trajectory identity
sampled-only physical-owner identities
duplicate, malformed, unsupported, and unresolved row counts
harmful-row count H(B)
owner-conditioned geometry quality
```

Also retain one owner-by-trajectory inclusion matrix per image. Its rows are
verified physical owners and its columns are greedy plus the sixteen sampled
trajectories. This matrix is the primary visual explanation of whether support
is concentrated in one route or scattered across routes.

Aggregate summaries are secondary. The verdict must show representative
image-level matrices and crop-reviewed objects so one dense scene cannot hide
an opposing mechanism in another.

## Scope

### Source model

Use the Qwen3 Vision-Language 2-billion-parameter description-first,
geometry-sorted, pure-cross-entropy, token-type-gated Weight-Decomposed
Low-Rank Adaptation checkpoint at step `4,887`:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/
qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_
accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json
```

Checkpoint JavaScript Object Notation (`JSON`) Secure Hash Algorithm 256-bit
(`SHA-256`) digest:

```text
c8ad1ab01550fc640c67457fec9ad1f8b3bd1b8cef351cb90d41666233b80da1
```

Use the existing inference configuration only as the model-loading and prompt
authority:

```text
/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/
qwen3_vl_2b_description_first_geometry_sorted_pure_cross_entropy_type_gate_
dora_step4887_same_covered_set_prefix_order.yaml
```

Configuration `SHA-256` digest:

```text
d6790bb1bd130d5c2823e0ffc68791aa97054cf6cbc7d423b86f3c9f55ac9dea
```

Do not inherit its `generation.max_new_tokens: 32` value. That configuration
was created for short candidate-row scoring, not complete trajectory rollout.
The canonical inference schema intentionally admits deterministic decoding
only, so sampled settings must be executed through the existing
request-scoped Hugging Face sampling seam rather than encoded as a canonical
inference configuration. Preserve the source configuration's model,
processor, prompt, and parser semantics, and write an experiment-local
execution receipt containing the exact selected image set, artifact root, and
actual decode settings specified below. The receipt must explicitly state
that the sampled policy bypasses the deterministic configuration policy.

### Development and validation cohort

Use exactly these twelve human-refined images:

```text
1584, 2685, 4134, 5001, 6040, 7511,
10707, 13348, 13923, 14038, 14439, 16228
```

Their current annotation authority is generation `7` of:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

Dataset `SHA-256` digest:

```text
81d674070d4b588488a2cb911c09f765b63c0e6d035b50db27ee0a41ff2a1894
```

Publication receipt:

```text
/data/CoordExp/outputs/coco_refinement/gate-a-20260717/val/
training.publish.receipt.json
```

No image in this cohort may appear in later self-imitation, counterfactual
completion, or hyperparameter-selection training data. Because this cohort now
selects the treatment family, it is not a final blind cohort.

### Decode conditions

Generate one native greedy trajectory and sixteen sampled trajectories per
image.

| Setting | Greedy | Sampled |
|---|---:|---:|
| Sampling enabled | no | yes |
| Temperature | `0` | `0.4` |
| Nucleus probability threshold (`top_p`) | not applicable | `0.95` |
| Repetition penalty | `1.0` | `1.0` |
| Maximum newly generated tokens | `512` | `512` |
| Evaluation row budgets | `4`, `8`, `16`, `32` | `4`, `8`, `16`, `32` |
| Effective per-device generation batch | `1` | `1` |
| Image resize | disabled | disabled |

Use these sixteen immutable, unique sampled seeds shared across images:

```text
21001, 21002, 21003, 21004, 21005, 21006, 21007, 21008,
21009, 21010, 21011, 21012, 21013, 21014, 21015, 21016
```

A seed is a paired request identifier, not an independent image-level case.
Do not tune temperature, nucleus threshold, seed values, or seed count per
image. The batch size of one is intentional for request-scoped seed isolation;
do not claim that the source configuration's batch size of four was executed
by the sampling seam.

Generate up to `512` new tokens, parse all complete rows, and compute each
budgeted result by truncating the parsed trajectory after its first `B`
complete rows. The `32`-row value is an evaluation horizon, not a claim that a
dense image has been exhaustively enumerated. Images in this cohort contain up
to `50` reference objects. If generation reaches the token limit before a
natural terminal action, record the trajectory as right-censored rather than
as a normal stop. At any budget `B`, if even one of the greedy or sixteen
sampled trajectories is right-censored before its `B`-th complete row, mark
`C_g(B)`, `C_best(B)`, `C_union(B)`, and all image-level branch evidence
missing at that budget. Coverage from available prefixes may be reported only
as a non-branch-bearing lower bound; never drop the censored trajectory and
silently change the sampled-union estimand. Lower complete budgets remain
usable.

### Numerical precision

Ordinary generation may use the qualified model dtype owned by the inference
configuration. Accumulate matching utilities, token scores when retained,
geometry reductions, and summary differences in 32-bit floating point. This
unit does not require full-model 32-bit floating-point execution because the
primary observation is the parsed physical-owner set, not a close logit
difference.

## Global One-to-One Matching and Human Review

For each trajectory prefix ending at budget `B`, solve one global one-to-one
assignment between predicted rows and current human-refined reference owners.
The assignment utility keeps these components separate:

1. entity existence and category compatibility;
2. physical-owner compatibility;
3. box geometry quality.

An automatic exact-category match with box intersection over union at least
`0.50` is provisionally admissible. Lower-overlap, cross-category,
multi-instance, part-only, or neighboring-instance cases enter enlarged-crop
review rather than automatic rejection. Human review labels the entity axis as
verified owner, duplicate, semantic error, unsupported hallucination, or
unresolved; it labels the geometry axis independently as acceptable, shifted,
oversized, undersized, incomplete, neighbor-contaminated, mixed-instance, or
unresolved.

Matching is recomputed at every fixed row budget. A later accurate row may own
an entity instead of an earlier poor row; an irreversible first-arrival claim
is not used.

## Execution Outline

### Stage 0: inventory and real smoke

1. Inventory compatible existing full-image trajectories before generating
   new output.
2. Reuse a trajectory only if checkpoint, prompt, image processing, decode
   settings, seed, and row budget match this unit exactly.
3. Run one image through one greedy and two sampled trajectories as the first
   real smoke.
4. Verify row parsing, maximum-row handling, coordinate conversion, owner
   matching, and the owner-by-trajectory matrix.
5. Expand to all twelve images only after the smoke can distinguish one owner
   discovered twice from two distinct owners.

### Stage 1: frozen rollout collection

Run one greedy plus sixteen sampled trajectories for every image. Retain raw
text, generated token identifiers, parsed rows, parser failures, request seeds,
checkpoint/config identities, and terminal reason. Do not run a treatment or
modify decoding based on interim results.

### Stage 2: matching and crop-assisted adjudication

Compute provisional global matches at budgets `4`, `8`, `16`, and `32`. Build
one compact review queue only for predictions whose entity or physical owner
can change `C_g`, `C_best`, or `C_union`. Review enlarged crops and freeze the
owner decision before producing the final summaries.

### Stage 3: route classification

Classify each image and row budget as:

- better individual sampled route demonstrated;
- complementary union without a better individual route;
- no material sampled support beyond greedy;
- inconclusive because ownership or execution is unresolved.

Then issue one program-level route recommendation using the branch rules
below. Do not average away contradictory image families.

## Branch and Stop Rules

Apply the following program-level rules in order. The first satisfied rule is
the only successor recommendation. This ordering makes the routes mutually
exclusive.

### Rule 1: admit a trajectory-self-imitation successor only when

- at least three distinct images contain a sampled trajectory with at least
  one more verified owner than greedy at the same fixed row budget;
- on each counted image, that trajectory has no more entity hallucinations,
  semantic or category errors, duplicate-owner rows, or malformed attempts
  than greedy at that budget; equivalently, no harmful-row component and
  `H(B)` may increase;
- on at least two of the counted images, the same sampled trajectory retains
  its verified-owner gain and its safety condition at an adjacent row budget;
  and
- the best-trajectory identities and newly covered owners are fully auditable.

The successor must be a separate research unit. Its first treatment is
positive-only, span-masked weighted self-imitation against a within-image
shuffled-reward control. No signed negative advantage or full policy-gradient
claim is admitted by this diagnostic alone.

### Rule 2: otherwise, admit a local remaining-object-completion successor when

- Rule 1 failed; and
- at least three distinct images have a verified positive
  `C_union(B) - C_g(B)` at one or more row budgets.

The successor trains verified uncovered owner continuations at natural
prefixes. It does not treat the sampled union as one trajectory. Report
`C_union(B) - C_best(B)` separately to distinguish genuinely complementary
routes from sampled support that appears inside an unsafe complete trajectory.

### Rule 3: otherwise, return to visual-support accessibility

Choose this branch only when Rules 1 and 2 failed, fewer than three distinct
images show any verified positive `C_union(B) - C_g(B)`, and the absence cannot
be explained by parsing, annotation, or sampling failure.

### Invalidate or rerun this unit when

- the checkpoint, prompt, dataset generation, image resize, repetition
  penalty, temperature, nucleus threshold, or seeds differ across compared
  trajectories;
- owner matching counts one physical entity more than once;
- a material review candidate remains unresolved; or
- the derived execution configuration does not use `512` newly generated
  tokens; or
- execution artifacts cannot reconstruct the first `B` complete rows and do
  not record whether the trajectory was naturally terminated or
  right-censored.

## Non-Goals

- Estimating publication-level population performance from twelve selected
  dense images.
- Training on the twelve development and validation images.
- Comparing random-order and geometry-sorted training.
- Suppressing the terminal action, enforcing a minimum row count, or applying
  Non-Maximum Suppression as the primary intervention.
- Treating every unmatched prediction as hallucination.
- Optimizing Average Precision, designing a final architecture, or deciding
  whether an explicit covered-set carrier is necessary.
- Starting Group Relative Policy Optimization, REINFORCE Leave-One-Out, or any
  other online policy optimization before the support regime is classified.
- Creating an OpenSpec change or stable code interface for an exploratory
  analyzer.

## Reused Surfaces and Expected Implementation Boundary

Reuse current model loading, prompt rendering, token parsing, and batched
inference. The implementation reused the one-to-one matching foundations from
the historical implementation at
`f5af926ba:src/analysis/spatial_scope_history/metrics.py` and trajectory parsing
under `src/analysis/sampled_rescue_transition/`. The commit-qualified spatial
path is a provenance handle rather than a current application programming
interface. Add only an experiment-local launcher or summarizer under
`scripts/research/` for the missing fixed-budget owner-union calculation and
review packet.

Do not promote a shared application programming interface from this first
consumer. Implementation may be direct and experiment-local as long as
checkpoint, decode, owner, and budget semantics remain inspectable.

## Artifact Handle

Logical root:

```text
outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-individual-trajectory-versus-union-support-audit/<run-id>/
```

Resolved durable root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-individual-trajectory-versus-union-support-audit/<run-id>/
```

Assign one immutable run identifier when the first real smoke starts. The
compact receipt must record source commit or dirty-diff identity, checkpoint,
configuration, dataset, image identifiers, decode settings, seeds, raw output,
parser failures, matching version, review decisions, and terminal status.

The first real smoke assigned this run identifier:

```text
support-audit-step4887-20260721a
```

Final interpretation and branch decision:

```text
research/investigations/qwen3-vl-dense-enumeration/experiments/
2026-07-21-individual-trajectory-versus-union-support-audit/results.md
```

Independent audit and accepted corrections:

```text
research/investigations/qwen3-vl-dense-enumeration/experiments/
2026-07-21-individual-trajectory-versus-union-support-audit/audit.md
```

Primary reviewed analysis:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-individual-trajectory-versus-union-support-audit/
support-audit-step4887-20260721a/analysis/reviewed-support-v3.json
```

## Expected Cost

The full primary panel contains `12 * (1 + 16) = 204` trajectories, each with a
`512`-new-token generation cap and evaluation horizons through `32` complete
rows. It is inference-led and may use all eight available graphics-processing
units through independent image or seed partitions. Human review is restricted
to cases that can change the three coverage quantities.

## Closure Contract

Close this unit with a separate `results.md` containing:

- **Observed**: executed trajectory, owner, and geometry facts;
- **Supported**: which support regime survived its controls;
- **Ruled out**: explanations contradicted by fixed-budget matching and review;
- **Unresolved**: remaining ambiguity and unexecuted replication;
- **Not claimed**: population generalization, training efficacy, final
  architecture, and blind-test performance; and
- **Next discriminator**: exactly one successor treatment or diagnostic branch.

No architecture or training method is promoted directly from this unit.

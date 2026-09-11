---
title: Physical-Owner Duplication Causality and Training Treatment
description: Tests whether generated history causally induces repeated physical instances and whether direct rejection, recovery, and duplicate-cleaned trajectory supervision improve greedy unique-object coverage.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: completed
unit_id: 2026-07-22-physical-owner-duplication-causality-and-training-treatment
topic: qwen3-vl-dense-enumeration
status: completed
evidence_status: bounded_treatment_result
updated: 2026-07-22
---

# Physical-Owner Duplication Causality and Training Treatment

## Final Decision

The bounded unit supports the duplicate-cleaned trajectory training recipe as
a promising treatment and rejects the combined local-plus-cleaned profile. The
decision is based on equal-update Source-only controls, a 240-image never-trained
cohort, a disjoint twelve-image human-refined panel, and enlarged visual
review. Cleaned imitation increases matched unique owners while reducing
strict duplicate candidates against its six-update Source-only control on both
evaluation panels. The effect is not uniformly safe: image `10707` loses a
valid laptop owner and develops a repeated remote row.

The equal-update controls repeat Source rows and therefore do not match event
composition or Source exposure. They exclude optimizer-step count alone as the
explanation, but they do not isolate cleaned-trajectory semantics as the sole
cause of the result.

This result promotes a training direction, not a final architecture or a
full-dataset recipe. Expansion requires a larger explicitly reviewed
physical-owner ledger. The current expanded queue is candidate evidence only;
it must not be automatically converted into training labels.

## Question

When a Qwen3 Vision-Language autoregressive detection rollout repeats one
physical object instance, is the repeated branch caused by the preceding
generated history, and can training directly reject that branch while teaching
the model to recover toward a later valid, not-yet-covered physical instance?

The unit is successful only if it distinguishes symptom suppression from set
improvement. Fewer repeated rows alone are insufficient: released probability
must move toward new verified instances without unacceptable losses of Source
instances, row validity, category correctness, or natural termination.

## Terminology

- **Physical owner**: one physical object instance in the image. Category and
  bounding-box overlap are evidence about ownership, not ownership itself.
- **Covered owner**: a physical owner already represented by an accepted row in
  the current rollout prefix.
- **New owner**: a verified physical owner not represented by an earlier
  accepted row in that prefix.
- **Duplication burst**: one or more consecutive valid-looking rows assigned to
  a covered physical owner.
- **Recovery row**: the first later row after a duplication burst that is
  assigned to a new verified owner.
- **Source checkpoint**: the frozen geometry-sorted pure cross-entropy adapter
  at training step 4887, evaluated with repetition penalty 1.0.
- **Local duplicate rejection and recovery treatment**: a complete-row
  preference objective that ranks a recovery row above a duplicate row at the
  exact self-rollout state where the choice is supervised.
- **Duplicate-cleaned trajectory treatment**: ordinary supervised training on
  a rollout-derived sequence in which only confirmed duplicate rows are
  deleted and the later valid suffix is retained.
- **Combined treatment**: training that uses both local duplicate
  rejection and recovery events and duplicate-cleaned trajectory examples.
- **Counterfactual rewritten trajectory**: a derived rollout sequence whose
  confirmed duplicate rows were deleted. It is set-valid evidence under the
  declared owner ledger, but it is not claimed to be a naturally sampled
  autoregressive trajectory.

## Motivation and Current Evidence Boundary

Repeated full-image sampling shows that additional physical instances are in
the model distribution, while one greedy rollout often follows a narrower
route. Earlier positive-row imitation changes which owners greedy decoding
finds but tends to exchange owners rather than expand the final set. This unit
tests whether repeated-owner branches are one concrete place where useful
generation opportunity is consumed and whether a direct local treatment can
improve that transition.

The current evidence does **not** establish that every official false positive
is a hallucination. Common Objects in Context annotations are incomplete,
especially in dense images. An unmatched prediction is neutral until visual
review establishes one of: real omitted instance, category error, physical
duplicate, localization or adjacent-instance binding error, entity
hallucination, or unresolved evidence.

## Competing Explanations

1. **Prefix-caused repetition**: one or more earlier rows create a state that
   reinforces selection of an already covered owner. Reordering or replacing
   those rows should change the next-row owner distribution.
2. **Current-image competition**: duplication is driven mainly by current
   visual evidence and decoder dynamics. Semantically valid prefix
   counterfactuals should have little specific effect.
3. **Duplicate-as-fallback**: the duplicate is emitted because no new owner has
   a coherent complete-row path. Rejecting it should redirect probability to
   termination, invalid rows, or category errors instead of new owners.
4. **Training-signal mismatch**: a useful recovery owner exists, but complete
   row preference at one exact prefix does not generalize to self rollout.
   Duplicate-cleaned trajectories may be required to teach the altered state
   sequence.
5. **Sequence-only repair**: deleting duplicate rows and retaining the later
   valid suffix may be sufficient; a new local preference loss may not be
   necessary.

## Hypotheses and Testable Predictions

### Prefix causality hypothesis

At the exact prefix immediately before a confirmed repeated-owner row,
same-coverage complete-row shuffling, length-matched covered-owner replacement,
or removal of the earlier occurrence will change the duplicate-row score or
the first continuation owner in a direction specific to the modified history.

Falsification: exact, shuffled, replaced, removed, and coordinate-corrupted
prefixes yield materially indistinguishable complete-row scores and owner
outcomes under matched inference.

### Local rejection and recovery hypothesis

For a trajectory

```text
valid prefix -> duplicate row 1 -> ... -> duplicate row k -> recovery row -> valid suffix
```

the model can learn both:

1. at the pre-burst prefix, rank the recovery row above duplicate row 1;
2. at intermediate duplicate prefixes, rank the recovery row above the next
   duplicate row.

Each duplication burst receives one total unit of training weight so long
bursts do not dominate.

The recovery row is not assumed feasible at an earlier prefix merely because
it was observed after the burst. Before an entry-boundary or intermediate
comparison receives gradient, the row must retain a finite, non-degenerate
teacher-forced score under that replay prefix and pass a short free-continuation
feasibility check. The post-burst recovery state is the only naturally observed
same-prefix positive.

Falsification: exact-prefix complete-row margins improve but greedy physical-
owner duplication and unique-owner coverage do not improve, or the model only
terminates earlier.

### Duplicate-cleaned trajectory hypothesis

Removing only confirmed duplicate rows while retaining the first occurrence of
each owner and the later valid suffix supplies a coherent counterfactual route:

```text
valid prefix -> recovery row -> valid suffix
```

Falsification: the treatment reduces sequence length without increasing unique
verified owners, or it causes category, geometry, invalid-row, or Source-owner
retention regressions.

### Combined-treatment hypothesis

Local decision supervision and cleaned sequence supervision solve different
parts of the problem. Their combined profile should outperform either
alone on greedy unique-owner coverage and duplication-burst reduction without
merely increasing or shortening rollout length.

Falsification: the combination is no better than the strongest single
treatment or produces incompatible gradient interference and worse rollout
health.

## Evidence Scope

### Model

- Base model:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- Source adapter:
  `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/adapter`
- Source special-token embedding delta:
  `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/special_token_embeddings`

### Decode invariants

- repetition penalty: 1.0;
- canonical Source prompt and row serialization;
- normalized 0-to-999 coordinate tokens;
- complete rows, not individual repeated tokens, define the training action;
- model, image, decode policy, candidate budget, and seeds are held constant
  within paired prefix analyses.

### Cohorts

1. **Mechanism panel**: 4 to 8 crop-reviewed dense images with reproducible
   physical-owner duplication, separable instances, and at least one sampled
   recovery when possible.
2. **Training event pool**: begin from the existing 256-image Source and sampled
   trajectories; expand only when the owner-confirmed event census is too small
   for a useful one-epoch screen.
3. **Development and held-out evaluation**: include images that supplied no
   training gradient and the twelve human-refined dense validation images as
   high-quality development evidence, not training data.

The initial mechanism roster is:

- primary: image `455649` bottle repeat followed by a distinct bottle;
- primary: image `55232` umbrella burst followed by a person;
- primary: image `351528` book burst followed by a cat;
- primary: image `187464` knife-owner burst followed by a bottle;
- no-recovery control: image `505243` book burst ending without a later new
  annotated owner;
- irregular-geometry stress only: image `57676` overlapping boats;
- dense binding stress only: image `424960` people and chairs.

The original 113-row summary was never materialized as a review ledger and is
not used as training evidence. A fresh deterministic join of one greedy and
sixteen sampled routes per image produced two reproducible review queues:

- a strict queue at prediction-to-earlier-prediction intersection over union
  at least 0.9: 37 candidate bursts, 57 repeated rows, 28 later recoveries,
  and 17 represented images;
- an expanded queue at prediction-to-earlier-prediction intersection over
  union at least 0.8: 90 candidate bursts, 119 repeated rows, 62 later
  recoveries, and 32 represented images.

Both queues require assignment to the same category-consistent annotated
owner at intersection over union at least 0.5. They are review aids, not
training labels. Official-unmatched and unresolved rows remain neutral.

The first training screen uses a lower-noise set of 24 one-row bursts from 14
duplicate-mechanism images. Every burst contains one candidate repeated-owner
row followed immediately by a row assigned to a previously uncovered
annotated owner at annotation intersection over union at least 0.8. The subset
retains only two trajectories from the overrepresented image `65530` and
removes one extra trajectory from image `305573`.

The materialized StateBanks contain 24 Source-preservation events, 24 recovery
events, 24 local duplicate-rejection events, and 24 duplicate-cleaned
imitation events. Consequently the five arm sizes are 24, 48, 48, 48, and 72
records. Source preservation touches 11 images, duplicate-mechanism events
touch 14 images, and their union touches 16 images. No event is replicated in
these treatment banks. Each selected event is still subject to an explicit
decision ledger; queue membership alone cannot admit training gradient.

Training sample size may grow to 1,024 or 2,048 images under the authorized
goal if the 256-image event pool is too sparse or a smaller screen is promising.
The changed sample scope must be recorded before its run begins.

## Owner Review and Eligibility

Training-negative eligibility requires a confirmed physical-owner duplicate.
Category-and-overlap heuristics may create a review queue but cannot create a
training label by themselves in crowded scenes.

Training-positive eligibility requires a verified new physical owner. A row
may have imperfect geometry and still be positive for entity discovery only if
the treatment does not supervise its untrusted coordinates. Category errors,
neighbor-contaminated rows, malformed rows, and unresolved rows are neutral.

Official false negatives and visually verified omitted Common Objects in
Context 80-category instances are valid positive evidence. Official unmatched
predictions are not negative by default.

## Causal Prefix Analysis

At each selected pre-duplication boundary compare:

1. exact self-rollout prefix;
2. same covered owners with complete rows shuffled;
3. earlier occurrence of the impending repeated owner removed;
4. length-matched replacement by a different already covered owner;
5. description preserved with a valid but incorrect coordinate address.

Record normalized complete-row scores for the impending duplicate, reviewed
new-owner candidates, and terminal continuation. Also run short matched free
continuations and record first owner, downstream new-owner set, duplication,
category errors, invalid rows, and termination.

The primary causal comparisons are exact versus same-coverage shuffle and exact
versus length-matched owner replacement. Removal and coordinate corruption are
secondary because they change length or may create less natural prefixes.

## Training Treatments

All treatments start from the same Source checkpoint, update the language-model
tower through the existing low-rank adaptation path, keep the vision tower and
visual-language projector frozen, and use the existing token-type gradient gate
as a stabilizer rather than an independent experimental arm.

All treatment arms use the same selected mechanism trajectories, aggregate
per-burst mechanism credit, random seeds, and globally sampled trusted Source
preservation rows. The inherited Source-preservation pool does not cover every
mechanism image, so Source rows are selected deterministically without
replication from the full immutable pool rather than falsely claiming
same-image pairing. Arms receive one pass over their admitted evidence; their
optimizer-step counts differ when a cleaned suffix contains several complete
rows and are reported explicitly. Source preservation is a control for generic
fine-tuning drift, not proof that all native behavior is retained.

### Source checkpoint

Evaluation only. It defines the matched rollout, physical-owner, geometry,
format, and termination baselines.

### Source-preservation-only control

Run the same optimizer dose using only trusted Source-route rows. This measures
generic low-learning-rate fine-tuning drift.

### Recovery-positive-only control

Use the same replay prefixes and recovery rows as the local pairwise treatment,
but omit the duplicate negative. This separates learning a useful positive row
from the incremental effect of explicitly penalizing the repeated owner.

### Local duplicate rejection and recovery treatment

For exact model-visited prefix `P`, verified new-owner row `U`, and confirmed
duplicate row `D`, use field-balanced complete-row scores. Wrapper and delimiter
tokens are excluded from the comparison; the score is the mean of the
description-field mean log probability and trusted coordinate-field mean log
probability:

```text
local_loss(P, U, D) = softplus(margin - score(U | P) + score(D | P))
```

Description and coordinate contributions are logged separately. The primary
mechanism stratum uses same-category positive and duplicate owners when such
pairs exist, preventing a cross-category frequency shortcut from masquerading
as owner exclusion.

For a burst of length `k`, include the pre-burst comparison and each
intermediate duplicate-prefix comparison, then divide the total burst loss by
the number of included comparisons. A later recovery row may be transplanted
as the positive action only when its physical owner is verified and its origin
is recorded explicitly; it must not be misrepresented as a row naturally
sampled from the earlier prefix.

### Duplicate-cleaned trajectory treatment

Construct a rollout-derived counterfactual sequence by deleting confirmed
duplicate rows, retaining the first accepted occurrence of every owner, and
retaining subsequent valid rows after recovery. Supervise the resulting
sequence with ordinary row-token likelihood and the token-type gate. Do not
replace the retained first owner occurrence with a later better box in the
primary arm.

Every rewritten prefix is labeled `counterfactual_rewritten`; its original
candidate-generation prefix hash is preserved separately. Each successive
cleaned-prefix transition is validated and the automatic suffix stops at the
first invalid, category-error, owner-ambiguous, binding-ambiguous, or otherwise
untrusted row.

If a burst ends in terminal output without a verified recovery, the cleaned
trajectory may teach terminal output after the preceding valid prefix, but it
does not create a new-owner positive.

### Combined treatment

Use the same admitted bursts and total optimization dose as the single
treatments. Combine the local preference terms with duplicate-cleaned
trajectory likelihood; do not add unrelated canonical supervised examples in
this unit.

The core causal contrasts are:

- local duplicate rejection and recovery minus recovery-positive-only:
  incremental value of the duplicate negative;
- duplicate-cleaned trajectory minus recovery-positive-only: value of the
  rewritten state sequence;
- combined minus local duplicate rejection and recovery: incremental value of
  cleaned-prefix supervision;
- Source-preservation-only minus frozen Source: generic update drift.

## Measurements

### Primary

- greedy unique verified physical-owner count;
- physical-owner duplication event count and burst-length distribution;
- recovered-owner rate for owners that occur after a duplicate burst;
- Source-owner retention;
- owner gain and owner loss separately, rather than only their difference.

### Safety and interpretation

- category correctness and confirmed entity hallucination;
- unresolved unmatched-prediction count;
- complete-row validity and natural closure;
- row count and terminal position;
- entity discovery separate from geometry quality;
- center error, size error, per-coordinate error, and neighbor contamination
  for trusted geometry cases;
- treatment effects on images that supplied no gradient.

### Exact-prefix learning checks

- positive-minus-duplicate complete-row margin;
- duplicate-row score change;
- recovery-row score change;
- whether improvement is localized to one token or distributed across the
  complete description-plus-coordinate row.

## Expected Outcomes and Decisions

| Observation | Interpretation | Decision |
|---|---|---|
| Duplicate count decreases and new owners increase | Duplicate branches were consuming useful traversal opportunity | Continue or scale the best treatment |
| Duplicate count decreases but termination increases | Symptom suppressed without remaining-owner evidence | Do not scale; investigate new-owner selection |
| Duplicate count decreases but category or invalid errors rise | Duplicate was replaced by unsafe fallback behavior | Do not scale; strengthen owner or row-validity evidence |
| One duplicate is replaced by another | Current treatment lacks persistent covered-owner exclusion | Retain result as evidence for stronger owner state, not as success |
| Exact-prefix margins improve but clean rollout does not | Local credit is learnable but fails under the changed state distribution | Prefer cleaned trajectories or refreshed on-policy event collection |
| Cleaned trajectories outperform local preference | Sequence-state coherence is more important than the local margin alone | Develop rollout-derived sequence treatment |
| Local preference outperforms cleaned trajectories | The main defect is the decision boundary, not the later state sequence | Refine complete-row owner preference |
| Combined treatment wins safely | Both local choice and downstream state require supervision | Promote to a broader bounded screen |

## Implementation Outline

Reuse the current exact-prefix StateBank replay, complete-row scoring,
rollout-derived trajectory assembly, training, inference, matching, and
visualization infrastructure. Add only the minimum truthful provenance and
event-construction support required for:

1. confirmed duplication-burst records;
2. downstream recovery-row origin;
3. per-burst weight normalization;
4. duplicate-cleaned rollout-derived sequences;
5. matched treatment receipts.

Do not build object slots, a persistent coverage architecture, a new detector,
or a general online-learning service. Event collection, training, and rollout
evaluation remain separate offline stages.

## Execution Order

1. Freeze the 4-to-8-image mechanism panel and visually confirm owner labels.
2. Run the prefix counterfactual panel and complete-row scoring.
3. Build the owner-confirmed burst and recovery ledger from existing
   trajectories; expand rollout collection only if the ledger is too sparse.
4. Implement and test the Source-preservation-only and recovery-positive-only
   controls plus the three training treatments.
5. Run deterministic loss checks and one representative real gradient smoke.
6. Launch matched 8-graphics-processing-unit one-epoch treatment screens.
7. Evaluate exact-prefix learning, clean greedy rollout, owner-set movement,
   duplication, geometry, category, validity, and termination.
8. Scale the most promising treatment to 1,024 or 2,048 images only when the
   smaller screen shows useful treatment behavior or when the smaller event
   pool is demonstrably underpowered.

## Representative Smoke

The first real smoke must contain one reviewed trajectory with at least one
confirmed duplicate followed by a verified new-owner recovery. It must prove:

- original and constructed prefix tokens replay exactly where claimed;
- positive and duplicate row identities remain distinct;
- burst weights sum to one per burst;
- gradients reach only declared token sites and trainable language-model
  parameters;
- the cleaned trajectory contains no confirmed duplicate row and retains the
  valid suffix;
- a saved checkpoint reloads and completes one clean rollout.

The causal analysis and bounded training arms proceed in parallel under the
authorized goal. A negative causal panel does not cancel an already valid
training smoke; it changes interpretation and prevents unsupported mechanism
claims.

## Artifact Root

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-physical-owner-duplication-causality-and-training-treatment/<immutable-run-id>/
```

Each run receipt records source commit and dirty diff identity, model and
adapter, decode settings, cohort and review ledger hashes, treatment, optimizer
dose, event counts, exclusions, checkpoint paths, raw rollout paths, and
terminal failures.

## Non-Goals

- proving that Qwen3-VL lacks or possesses an explicit covered-set memory;
- treating all official false positives as hallucinations;
- optimizing standard detection metrics without owner-level review;
- changing the object wrapper or coordinate vocabulary;
- adding a specialized detector, object-query bank, or slot architecture;
- claiming generalization from exact-prefix learning alone.

## Stop Rule

Stop a treatment family when duplicate suppression consistently redirects to
termination, alternative duplicates, category errors, or invalid rows instead
of new verified owners, or when exact-prefix learning repeatedly fails to enter
clean greedy rollout.

The unit closes when one of the following is true:

1. a treatment yields decision-grade owner-level improvement and a clear next
   scale decision;
2. all three treatments produce bounded negative evidence with the failure
   destination identified;
3. a long authorized training run has passed smoke, launched successfully, and
   has a complete continuation receipt for later analysis.

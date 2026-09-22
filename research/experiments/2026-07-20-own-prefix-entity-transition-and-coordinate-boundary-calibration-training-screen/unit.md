---
title: Own-Prefix Entity-Transition and Coordinate-Boundary Calibration Training Screen
description: A rollout-only training screen that tests whether local entity-transition and first-wrong-coordinate supervision can improve greedy dense enumeration without changing the Qwen3 Vision-Language (Qwen3-VL) inference architecture.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: smoke_b_closed_not_promoted
implementation_status: implemented_and_smoke_b_evaluated
unit_id: 2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: smoke_b_complete_bounded_negative
updated: 2026-07-20
---

# Own-Prefix Entity-Transition and Coordinate-Boundary Calibration Training Screen

## Decision Status

Smoke B is complete. The [Smoke B first-wrong-coordinate calibration
results](smoke-b-v1/results.md) close this unit without promotion to the
256-image screen. Both coordinate-loss arms improve the frozen held-out
coordinate margin relative to source and their learning-rate-matched gate-only
controls on four of five events, but neither arm beats its matched control on
ordinary rollout health or matched-box geometry. The local objective is real;
its end-to-end treatment value is not established.

This document originally froze the scientific design for a small training
screen. The independent implementation race, source-parity check, deterministic
one-event Smoke A, learning-rate dose screen, checkpoint reload, and ordinary
greedy inference have now completed. The [Smoke A and dose-screen results](smoke-a-dose-screen-results.md)
record the evidence and the required interpretation changes.

The formal 256-image screen is not authorized. The completed Smoke B bank
contains 13 current-row boundary decisions: 8 image-grouped training events
and 5 evaluation events. Seven nonempty-prefix events have an unresolved
prior-row covered set, while all 13 retain an exact prefix and an independently
reviewed current physical owner and first wrong coordinate. The bank contract
encodes that distinction and binds the canonical prompt to the source traces.

The screen deliberately tests a loss-only treatment before adding object
slots, object queries, an external detector, a persistent covered-object
ledger, a visual write-back mechanism, or any inference-time controller.

## Evidence-Driven Amendment After Smoke A

The one-event dose screen changes the original plan in four ways.

1. The term `entity-transition preference` is too strong for a same-category
   path whose owner becomes distinguishable only through coordinates. The
   operational treatment is now called **same-prefix uncovered-versus-covered
   candidate-row preference**.
2. Any scored positive or harmful candidate row whose owner-resolution
   interval contains coordinate tokens must have trusted geometry for that
   candidate's own sampled path. Entity trust alone is not enough, because
   those coordinates receive direct gradient. A reviewer-corrected box does
   not replace the sampled candidate path; it belongs to the separate
   coordinate-boundary objective unless the corrected path was independently
   produced by the checkpoint and admitted.
3. The complete-row preference has a provisional safe dose of `5e-6` on one
   event. The coordinate-boundary package keeps `1e-5` and `2e-5` for the real
   Smoke B. A joint arm is deferred because the two objectives have different
   observed dose ranges.
4. The coordinate-boundary package changed whole-rollout entity composition
   on an untrained image. This motivates the multi-state replication but does
   not yet prove a general coverage or commitment mechanism.

5. Expanded exact-prefix sampling produced no visually trusted same-prefix
   rescue: all apparent person rescues mixed adjacent instances and the
   terminal state remained terminal in 128 of 128 samples. Entity-transition
   and joint arms are therefore deferred rather than filled with weak labels.
6. The real Smoke B comparison is now coordinate preference plus the selected-
   site token-type gate versus a matched gate-only control, each at learning
   rates `1e-5` and `2e-5`. This separates the boundary objective from the
   accompanying gate instead of attributing their combined effect to one term.

These amendments narrow the next experiment. They do not retroactively change
the historical two-event bank or its completed evidence.

## Primary Question

At exact prefixes produced by the model itself, can direct supervision at the
first harmful entity-transition or coordinate-boundary decision move verified
objects from stochastic sampling into ordinary greedy rollout without merely
making the rollout longer, more repetitive, or less geometrically reliable?

The strongest alternative is that the needed covered-object or object-owner
state is absent, unreadable, or too dependent on a longer trajectory. Under
that explanation, local loss can improve an offline margin without improving
multi-row greedy enumeration. Such a result would justify a later compact
state or visual write-back experiment; it would not justify increasing this
loss indefinitely.

## Evidence That Opens This Screen

The immediately preceding matched-checkpoint study found that:

1. complete-row random-order training did not learn a useful order-invariant
   covered-object state;
2. the random-order checkpoint collapsed to one habitual successor on the
   dense image-`2299` control;
3. the geometry-sorted checkpoint retained more route diversity;
4. earlier prefix order can tip a local category or coordinate decision;
5. exact canonical box likelihood is not a reliable measure of whether an
   entity was visually discovered; and
6. terminal output was not the active failure in the selected scored states.

See the [matched random-order versus geometry-sorted results](../2026-07-20-matched-random-sorted-prefix-order-screen/results.md).

The program also has repeated evidence that entity discovery and exact box
extent must be reviewed separately. Unmatched output is not automatically a
hallucination because Common Objects in Context annotations are incomplete,
especially in dense scenes.

## Hypotheses and Distinguishing Predictions

### Hypothesis One: local entity-transition calibration is sufficient

**Mechanism.** A valid uncovered object already exists in the model's
same-prefix probability support, but a duplicate, premature terminal action,
or habitual successor has a slightly stronger local path. Directly comparing
these paths at the same exact prefix should move useful probability without a
new state carrier.

**Predictions.**

- sampled-only verified entities enter greedy rollout more often;
- the gap between bagged sampling and greedy unique-entity recall shrinks;
- unique entities found per emitted row improve;
- duplicate, malformed, and unsupported output do not rise materially; and
- the gain survives more than one subsequent committed row.

**Falsification.** The offline preference margin improves while free rollout
does not, or rollout length rises without unique-entity gain.

### Hypothesis Two: coordinate-boundary calibration is separately useful

**Mechanism.** The model discovers the correct physical entity but one early
coordinate decision enters a shifted, partial, oversized, undersized, merged,
or neighbor-contaminated box path. Correcting only the first known wrong
coordinate should repair later geometry more safely than supervising an
entire replacement row.

**Predictions.**

- owner-conditioned boundary error falls;
- box intersection over union improves for already discovered entities;
- mixed-instance boxes do not rise; and
- unique-entity recall is preserved.

**Falsification.** The selected coordinate changes but later coordinates
drift back, entity ownership changes, or entity recall declines.

### Hypothesis Three: the two treatments can coexist in shared Qwen3-VL parameters

**Mechanism.** Entity transition and coordinate extent are different
eligibility surfaces but can update the same Weight-Decomposed Low-Rank
Adaptation parameters. Separate masks and normalization should prevent the
larger event family from silently dominating the smaller one.

**Predictions.** A joint arm retains most of each single-treatment gain without
introducing a new cross-axis regression.

**Falsification.** Joint training erases either single-treatment gain or
creates a new entity or geometry failure.

### Hypothesis Four: geometry-sorted and random-order adapters expose different treatment response

**Mechanism.** Geometry-sorted training retains a useful traversal scaffold,
while random-order training has a different and sometimes collapsed
transition policy. The same local objective may therefore help both, only the
geometry-sorted checkpoint, or neither.

**Predictions.** Improvement on both checkpoints supports a native-state,
loss-only explanation. Improvement only on geometry-sorted supports reliance
on its learned traversal scaffold.

**Falsification of a general loss-only claim.** The random-order arm remains
collapsed or degrades while only the geometry-sorted arm improves.

## Fixed Scope

### Models

Use two step-`4,887` description-first, pure-cross-entropy plus token-type-gate
Weight-Decomposed Low-Rank Adaptation checkpoints:

1. geometry-sorted complete-row training, used as the primary performance
   base; and
2. random complete-row training, used as a mechanistic ablation.

Each checkpoint receives its own independently collected state bank. Prefixes,
candidates, or hidden states must never be transferred between checkpoints.

### Training data source

Training uses only reviewed records derived from the two checkpoints' own
rollouts. It does not mix in a fixed percentage of canonical supervised
fine-tuning examples. In particular, there is no `20%` or `30%` replay of the
original ground-truth sequence data.

The exact historical prefix is an input condition. Prefix tokens themselves
are not replayed as teacher-forced targets. Loss is applied only at explicitly
eligible decision sites derived from the rollout record.
Every training-eligible prefix event therefore retains direct supervision at
its selected decision sites; the design removes canonical sequence replay, not
the per-prefix corrective signal.

### Model and inference behavior

- Recompute the full image and exact token prefix on every training forward.
- Do not cache or freeze hidden states or key-value tensors as training input.
- Do not change the Qwen3-VL forward graph at inference.
- Do not add a detector, object slot, object query, persistent ledger, crop
  policy, or special decoding controller.
- Evaluate the trained adapter through ordinary greedy Qwen3-VL rollout.

### Excluded objectives

- no ordinary full-row cross-entropy replay over canonical supervised data;
- no Kullback-Leibler divergence anchor;
- no Gaussian coordinate smoothing;
- no blanket terminal-token suppression;
- no whole-row unlikelihood over tokens shared by valid and harmful paths;
- no long-horizon reinforcement learning in this first screen.

## Offline State-Bank Lifecycle

This is not streaming or online learning. The first screen has one frozen
cycle:

```text
collect checkpoint rollouts
  -> freeze exact-prefix candidate records
  -> review entity and geometry status
  -> freeze image-grouped train and evaluation splits
  -> train the declared arms
  -> evaluate all arms on the same frozen evidence
```

The state bank stores exact token identifiers. It must never reconstruct a
prefix by decoding text and tokenizing it again. A later winning method may be
tested with one refreshed bank, but refresh is outside this unit.

## Physical-Entity Ledger for Review

Every admitted image has one stable physical-entity ledger. The ledger is a
data and review device, not a model input or inference-time memory.

Each entity record must identify at least:

- image identity;
- stable per-image physical entity identifier;
- Common Objects in Context 80-category class;
- whether entity existence and category are trusted;
- whether reference geometry is trusted;
- reference geometry and any accepted boundary tolerance;
- source, reviewer, review confidence, and comment.

Trusted sources are official annotation, user-reviewed additions, and
lead-agent additions that are visually clear after original-image and enlarged
crop inspection. Ambiguous additions are held for user review. Before the
formal 256-image screen, the user spot-audits approximately 10 percent or at
least 20 of the lead-agent-only additions, whichever is larger.

Official annotation mismatch alone never makes a prediction negative.
Unknown status is axis-specific: entity-unknown disables direct entity loss,
while geometry-unknown disables direct geometry loss. One unknown axis does
not disable a separately trusted and eligible axis.

## Prefix and Candidate Admission

### Prefix admission

Entity-transition and covered-set claims are eligible only when every prior
object row maps uniquely to one physical entity identifier. All token aliases
or sampled rows that refer to the same physical entity are grouped; they must
not multiply that entity's weight. The owner score is the maximum candidate-
path score among that owner's admitted aliases. The multi-positive smooth
maximum is then applied across distinct physical-owner scores, not across raw
aliases.

A coordinate-only diagnostic may keep a nonempty exact prefix whose prior-row
owners are unresolved when, and only when, the current-row physical owner,
geometry, earlier accepted coordinates, and first wrong coordinate are
independently reviewed. Such an event records unresolved prefix coverage,
uses no entity-transition gradient, and makes no claim about novelty,
commitment, covered-object redistribution, or coverage-aware enumeration. The
exception preserves the conditional question: given this exact natural prefix
and current owner, can training improve the selected boundary decision?

An actual greedy row that is a valid new entity is committed and the rollout
continues. It is not used as a negative merely because another valid entity
was preferred by a canonical ordering policy. The first training event is the
first verified harmful decision.

### Positive entity-transition candidate

A positive must satisfy all of the following:

- generated from the same exact image and prefix;
- actually present in the checkpoint's sampled support;
- reviewed as a real Common Objects in Context 80-category entity;
- not yet represented by the physical entities in the prefix; and
- sufficiently resolved to one physical owner.

A canonical ground-truth row is not inserted as a transition positive unless
the checkpoint actually produced an equivalent owner path at the same exact
prefix. Ground truth establishes identity and coverage; it does not invent
model support.

An object reachable only after a different sampled history is not a local
positive for the current prefix.

### Harmful entity-transition candidate

The first version permits only the model's actual greedy harmful branch:

1. a physical duplicate of an entity already represented in the prefix; or
2. terminal output when a verified uncovered entity remains and a valid
   same-prefix sampled rescue exists.

Unsupported but visually ambiguous output is `unknown`, not negative. A valid
new greedy entity is also not negative.

### Mixed and ambiguous rows

A row that mixes several physical owners, falls between owners, or cannot be
assigned uniquely has unknown entity ownership and receives no direct
entity-transition gradient.

It may receive coordinate supervision only when a same-owner corrected target
is independently established and geometry is trusted. Sharing an exact prefix
does not prove same-owner correspondence because another valid entity may be
selected there. Otherwise the row remains an evaluation and audit example.

### Imperfect geometry on a real entity

A verified real uncovered entity with imperfect geometry can participate in
the candidate-row preference only when its physical owner is resolved before
the first coordinate token. If ownership requires any coordinate token, the
candidate's own scored geometry must be trusted; otherwise the row is
diagnostic-only for candidate-row preference. Independently trusted corrected
geometry may still define a first-wrong-coordinate objective for the same
physical owner. Entity and geometry eligibility remain separate.

## Sampling Policy for State Collection

Collection and primary evaluation use repetition penalty `1.0` so the old
repetition heuristic does not define the treatment. A secondary compatibility
evaluation uses repetition penalty `1.10`.

At an admitted exact prefix:

1. reuse an existing low-temperature bag of eight samples when available;
2. if no verified valid rescue exists, collect eight samples at a predeclared
   moderate temperature;
3. if the second bag still has no verified rescue, keep the state for
   diagnosis only and do not train on it.

The exact temperature, top-p threshold, repetition penalty, seed, checkpoint,
prompt fingerprint, and token-prefix hash are stored in every record. The
moderate-temperature value is frozen from an existing qualified inference
configuration before collection rather than tuned per image.

## State-Bank Balance and Splits

Use at most four training states per image. Sample across:

- duplicate transition;
- verified premature terminal output with a same-prefix rescue;
- early, middle, and late prefix depth;
- first geometry error at `x1`, `y1`, `x2`, or `y2`;
- oversized, undersized, shifted, incomplete, contaminated, between-instance,
  and mixed-instance geometry outcomes.

The four coordinates mean left boundary, top boundary, right boundary, and
bottom boundary, respectively; `x` is horizontal and `y` is vertical.

Split by image identity. All exact-prefix sibling candidates, alternate
sampling seeds, counterfactual candidates, and physical-entity aliases remain
in the same split.

The formal screen uses a frozen 256-image state-bank source. The following 12
dense images form a completed blind evaluation cohort:

```text
1584, 2685, 4134, 5001, 6040, 7511,
10707, 13348, 13923, 14038, 14439, 16228
```

They are published as generation `7` inside the complete validation files,
not as a separate 12-row dataset:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.norm.jsonl
sha256: 44ab9fd985890129128bdacb07081cc00bd4affe8c6768d5e49609ca20a72e52

/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
sha256: 81d674070d4b588488a2cb911c09f765b63c0e6d035b50db27ee0a41ff2a1894
```

The publication receipt is:

```text
/data/CoordExp/outputs/coco_refinement/gate-a-20260717/val/training.publish.receipt.json
```

The human pass added objects, resized or repositioned existing boxes, and in
some cases deleted and redrew an object. Therefore annotation-identifier sign
and old-versus-new row difference are provenance only; they must not be used
to infer whether a physical entity is newly discovered. The generation-7
current row is the evaluation authority.

These 12 images must not be used for training, candidate mining, learning-rate
selection, stopping, arm selection, implementation-race fixtures, or smoke
fixtures. Before the full screen is launched, the lead freezes their exact row
hashes and asserts that no image appears in a state-bank split.

## Objective One: Entity-Transition Preference

For an exact prefix `P` and candidate path `y`, define the causal candidate
score as the sum of log probabilities from the next-row decision through the
shortest token prefix that uniquely resolves the physical owner:

```text
S(y | P) = sum_k log p(y_k | image, P, y_<k)
```

For a cross-category choice this may end in the description. For same-category
instances it may need `x1`, `y1`, `x2`, or `y2`. The score does not extend
through unrelated row closure once the owner is resolved. Summed, mean, and
per-token scores are retained as diagnostics, but the summed causal path score
is the training score.

A premature terminal candidate has no physical owner and no owner-resolution
interval. Its candidate score is exactly the log probability of the single
terminal token at the shared boundary. Its intended token type for the
token-type gate remains object-row schema because the verified valid action is
to begin another object row.

First reduce admitted aliases for each physical owner:

```text
owner_score(o | P) = max(S(alias | P) for alias owned by o)
```

Then, when several uncovered physical owners are valid, combine their owner
scores with a smooth maximum so that at least one valid owner must beat the
harmful branch without forcing every valid next entity to have equal
probability:

```text
positive_score(P) = temperature *
  (logsumexp(owner_score(u | P) / temperature
             for u in valid_uncovered_owners)
   - log(number_of_valid_uncovered_owners))
```

For the actual harmful greedy branch `c`, use:

```text
transition_loss(P) =
  softplus(margin - positive_score(P) + S(c | P))
```

The margin and smooth-maximum temperature are explicit configuration values
shared by all arms and seeds; they are not tuned per checkpoint or image. The
one-event smoke verifies their sign and scale before the formal screen.

## Objective Two: First-Wrong-Coordinate Preference

Use only a row with a uniquely trusted physical owner and trusted corrected
geometry. Find the first coordinate whose actual token lies outside the
accepted discrete boundary set. Earlier generated tokens remain fixed as the
causal context. Later wrong coordinates receive no direct loss in the initial
screen.

Let `A` be the accepted coordinate-token set at that position and `w` be the
actual wrong coordinate token. With 32-bit floating-point objective logits:

```text
correct_mass = logsumexp(logit[a] for a in A)
wrong_score = logit[w]
geometry_loss = softplus(margin - correct_mass + wrong_score)
```

The accepted set is derived from reviewed coordinate tolerance. It is not a
Gaussian target and must not cross the coordinate axis or the `[0, 999]`
vocabulary boundary. It must be nonempty, contain unique integer coordinate
tokens, and be derived from horizontal tolerance for `x1` or `x2` and vertical
tolerance for `y1` or `y2`.

## Token-Type Gate on Rollout-Derived Sites

Every decision site selected by either research objective also receives a
small token-type gate. The gate supervises the intended structural type at the
site, not the entire historical prefix and not an unrelated canonical row.
Every positive and harmful candidate token included in a training score must
therefore carry one intended phase type. For a harmful nonterminal branch this
preserves schema, description, or coordinate legality while the preference
loss changes the exact branch ranking.

- entity text sites allow description tokens;
- coordinate sites allow coordinate tokens;
- object-row wrapper sites allow the corresponding schema token group;
- when terminal output is the harmful branch, the shared boundary's intended
  type is object-row schema, not terminal output.

The harmful branch still receives the preference gradient. Its token-type gate
must not contradict the intended valid branch at a shared causal position.
This is especially important for premature terminal output: simultaneously
gating that same boundary as both schema and terminal would cancel the
experiment's meaning.

This gate is the only generic token-level preservation term in the first
screen. There is no ordinary supervised fine-tuning replay mixture and no
full-row base cross-entropy anchor.

Within one event, gate-site identity is the causal segment plus logits
position. Every declaration for the same site must agree on one intended token
type; a conflict rejects the event. Repeated identical declarations are
deduplicated. The gate is averaged over distinct selected sites within each
eligible event and then averaged over eligible events in the complete optimizer
step. A site shared by transition and geometry metadata is counted once, not
once per research term.

## Loss Arms

The original checkpoint is an evaluation-only baseline. The current Smoke B
trains four tightly matched arms from that checkpoint:

1. first-wrong-coordinate preference plus the selected-site token-type gate
   at learning rate `1e-5`;
2. the same package at learning rate `2e-5`;
3. selected-site token-type gate only at learning rate `1e-5`; and
4. selected-site token-type gate only at learning rate `2e-5`.

The gate-only arm consumes the exact same coordinate events and selected sites
but sets both research-objective weights to zero. It is not a no-training
baseline. Its purpose is to identify whether any observed change comes from
the first-wrong-coordinate preference or from legal token-type mass alone.
Entity-transition and joint profiles remain implemented but are not launched
without a visually trusted same-prefix positive.

Use a low learning rate, gradient clipping, few optimizer steps, and frequent
evaluation. One shared implementation-race smoke fixture and smoke config must
be frozen before worktrees are forked. The fixture binds the source checkpoint,
state-bank manifest, exact event identifiers, optimizer values, loss values,
and random seeds used to compare implementations. Formal-screen optimizer
values are chosen once after implementation comparison and smoke health review,
then frozen across both checkpoints, all arms, and both training seeds. They
are not tuned per checkpoint or image.

## Experiment Matrix

Smoke B uses one frozen geometry-sorted pure-cross-entropy source checkpoint,
one frozen 13-event StateBank, one seed, and the four arms above. Every arm
sees each of the 8 training events exactly once with effective batch size 4,
for two optimizer steps. The unchanged source checkpoint is evaluated with the
same ordinary greedy inference fixture and does not consume a training job.

This is a mechanism and launch screen, not a variance estimate. Replicated
seeds and a 256-image screen are permitted only if one coordinate arm improves
held-out coordinate margins, preserves owner consistency and ordinary rollout
health, and exceeds its learning-rate-matched gate-only control.

## Smoke Ladder

### Smoke A: one transition event and one geometry event

Verify:

- exact prefix and image replay;
- physical-owner and geometry masks;
- positive and harmful candidate identities;
- first-wrong-coordinate selection;
- token-type-group selection;
- 32-bit floating-point loss math;
- finite gradients; and
- one tiny optimizer update moves every declared margin in the intended
  direction.

All competing implementations must consume the same pre-fork smoke fixture;
an implementation agent may not substitute an easier event.

### Smoke B: 8 to 16 reviewed states

Verify:

- reviewed first-wrong-coordinate events train under both learning rates;
- each coordinate arm changes its held-out coordinate margin more than its
  learning-rate-matched selected-site token-type-gate-only control;
- ordinary free-row output remains parseable; and
- no hidden canonical supervised-fine-tuning replay is present.

Entity-transition and joint training are not part of this Smoke B because the
same-prefix sampling panel produced no visually trusted complete-row rescue.

### Formal screen: frozen 256-image state bank

Run the 12-job matrix only after both smokes pass. Do not expand the dataset,
add loss terms, or change inference architecture during the formal screen.

## Measurements

### Local measurements

- positive-versus-harmful transition margin;
- valid-positive rank at the owner-resolving boundary;
- accepted coordinate-mass versus actual wrong-token margin;
- token-type legal probability mass;
- eligible, ignored, unknown, and rejected event counts; and
- results by prefix depth and error family.

### Free-rollout entity measurements

- unique verified physical entities under a fixed emitted-row budget;
- unique verified entities per emitted row;
- sampled-only verified entities that enter greedy rollout;
- duplicate physical entities;
- verified premature terminal events;
- unsupported or hallucinated entities;
- malformed rows; and
- bagging-versus-greedy unique-entity gap.

Entity existence and category correctness are measured separately from box
geometry.

### Free-rollout geometry measurements

- owner-conditioned box intersection over union;
- left, top, right, and bottom boundary error;
- center and size error;
- oversized, undersized, shifted, incomplete, contaminated, between-instance,
  and mixed-instance rates; and
- phrase-owner versus geometry-owner agreement.

### Decode settings

Primary evaluation uses repetition penalty `1.0`. Secondary compatibility
evaluation uses repetition penalty `1.10`. Improvements must not exist only
under the old penalty heuristic.

## Pilot Success Gates

These are pilot decision thresholds, not paper-level statistical claims.

### Entity-transition-only arm

- sampled-rescue-to-greedy conversion improves by at least 10 percentage
  points;
- fixed-row-budget unique-entity recall improves by at least 3 percentage
  points;
- both training seeds move in the same direction;
- unsupported plus malformed output rises by no more than 2 percentage points;
  and
- conditional geometry intersection over union drops by no more than `0.02`.

### Coordinate-boundary-only arm

- owner-conditioned boundary error falls by at least 10 percent relative, or
  owner-conditioned intersection over union rises by at least `0.03`;
- unique-entity recall drops by no more than 2 percentage points; and
- mixed-instance geometry does not increase.

### Joint arm

- retain at least 70 percent of the entity-transition-only entity gain;
- retain at least 70 percent of the coordinate-boundary-only geometry gain;
  and
- introduce no new cross-axis regression.

## Interpretation Matrix

| Outcome | Interpretation and next move |
| --- | --- |
| Both checkpoint families improve | Strongest evidence that native state is sufficient for a loss-only treatment; scale cautiously. |
| Only geometry-sorted improves | The treatment depends on its traversal scaffold; do not claim general prefix robustness. |
| Offline margin improves but free rollout does not | Move next to a one-row Group Relative Policy Optimization test rather than increasing local loss. |
| One-row behavior improves but the gain disappears after two or three commits | A compact state carrier or visual write-back gains justification. |
| Entity transition improves but geometry does not | Keep the entity treatment and investigate phrase-to-instance and coordinate binding separately. |
| Geometry improves but entity recall does not | Geometry calibration is useful but does not treat enumeration. |
| Same-prefix sampling cannot produce a valid rescue | Local preference training cannot invent the missing route; test a short-horizon or visual intervention. |
| More rows appear without more verified unique entities | Fail the arm; this is continuation inflation, not treatment. |
| Duplicate, malformed, or unsupported output explains the gain | Fail the arm. |

## Confounders and Controls

- **Incomplete annotation:** visually review unmatched candidates; never treat
  official mismatch alone as hallucination.
- **Entity versus geometry:** maintain separate eligibility and metrics.
- **Checkpoint-specific state:** collect separate banks and never cross them.
- **Prefix reconstruction:** store and replay exact token identifiers.
- **Data leakage:** group by image and keep the 12 user-relabelled dense images
  blind.
- **Sampling compute:** compare fixed sampling budgets and fixed row budgets.
- **Repetition penalty:** make `1.0` primary and `1.10` secondary.
- **Length inflation:** report unique entities per row and fixed-row-budget
  recall, not only total recall.
- **Physical aliases:** group by physical entity identifier before weighting.
- **Shared parameters:** report gradient and outcome interaction in the joint
  arm; separate masks do not make the parameter updates independent.

## Artifact Root

All execution artifacts, when later authorized, belong under:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/
  <run-id>/
```

The research unit owns hypotheses, cohorts, review status, thresholds, and
verdicts. Runtime outputs own raw state-bank records, resolved configs,
training logs, checkpoints, inference results, metrics, and compact run
receipts. The OpenSpec change owns reusable implementation behavior only.

## Cost and Stop Rule

The intended cost is one bounded state-bank collection and review pass, two
small smokes, and 12 short training jobs executed in waves on eight graphics
processing units. It is not a full retraining campaign.

Stop this unit after:

1. both smokes and all declared formal arms either complete or have a visible
   material failure;
2. every result is evaluated on the same frozen state banks and blind cohort;
3. the success gates are applied without post-hoc threshold changes;
4. a result is classified using the interpretation matrix; and
5. the program compass records whether loss-only treatment is promoted,
   narrowed, or closed.

Do not add architecture, more losses, more samples, or a refreshed online bank
inside this unit. Any such move requires a new research unit and a new user
decision.

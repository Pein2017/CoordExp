## Context

The current CoordExp-Swift training path is optimized for rendered supervised
examples, canonical `TokenSequence` supervision, no-padding segment-isolated
packing, protected token losses, Accelerate runtime, and standard
Weight-Decomposed Low-Rank Adaptation checkpoints for the Qwen3
Vision-Language model (Qwen3-VL). The inference path already owns checkpoint
composition, exact prompt-token receipts, Hugging Face and vLLM (the
high-throughput large-language-model inference project) decoding,
selected-token likelihood traces, and detection evaluation.

The research screen introduces a different data shape: one reviewed event has
an exact model-produced prefix, one or more valid candidate continuations, one
actual harmful greedy continuation, physical-entity identity, optional trusted
geometry, and selected causal positions. Candidate alternatives must be scored
jointly while remaining separate causal segments. The implementation must not
turn this bounded experiment into a second general trainer or inference stack.

Scientific admission rules, sample counts, experimental arms, thresholds, and
interpretation are owned by the linked research unit. This design owns only the
reusable execution path.

## Goals / Non-Goals

**Goals:**

- load and validate one immutable, checkpoint-bound rollout state bank;
- replay exact image and token prefixes without decoding and tokenizing them
  again;
- score grouped candidate continuations with gradients;
- separate the same-state first-divergence decision from coherent positive-row
  continuation supervision;
- compute entity-transition, first-wrong-coordinate, and rollout-site
  token-type-gate losses in 32-bit floating point;
- compute positive-only complete-row imitation with separate
  schema-and-description and trusted-coordinate group means;
- preserve separate event masks, normalization, metrics, and provenance;
- distinguish the checkpoint that generated a StateBank from the checkpoint
  used to warm-start an explicitly declared off-policy replay run;
- reuse existing model loading, packing, runtime, optimizer, artifact, and
  checkpoint owners; and
- support deterministic one-event and small-state smokes before the formal
  training screen.

**Non-Goals:**

- online rollout collection during optimization or an in-trainer refresh loop;
- automatic human-review replacement;
- canonical supervised-fine-tuning replay or full-row base cross-entropy;
- Kullback-Leibler divergence anchoring;
- new model heads, object slots, object queries, persistent memory, visual
  write-back, or modified inference forward propagation;
- a general reinforcement-learning trainer;
- exact optimizer-state resume; or
- a finalized large-scale training interface.

## Execution Flow

```text
existing inference and research tools
  -> immutable reviewed state-bank JSON Lines records plus bank manifest
  -> exact-token calibration loader
  -> checkpoint and image identity validation
  -> candidate segments grouped by event
  -> existing Qwen image preparation and forward
  -> companion calibration metadata maps event candidates to logits rows
  -> research loss terms plus rollout-site token-type gate
  -> existing Accelerate optimizer/runtime
  -> existing run logs and DoRA checkpoint writer
  -> existing inference and detection evaluation
```

The state bank stores inputs and review decisions, not hidden states. Every
training step recomputes image features and language-model states from the
current adapter parameters.

## Research Contract Alignment

| Frozen scientific decision | Implementation contract |
| --- | --- |
| Rollout-derived training data only | The calibration profile rejects canonical supervised-fine-tuning mixtures and full-row base cross-entropy. |
| Keep per-prefix corrective supervision | Every eligible exact-prefix event selects explicit transition or coordinate decision sites. |
| Keep local correction tied to set value | Entity-transition events carry equal-budget counterfactual admission evidence; trajectory quality controls admission rather than gradient weight. |
| Keep a small token-type gate | Every selected positive and harmful site has one intended phase type; a harmful terminal boundary is gated as object-row schema. |
| Prefix is conditioning, not a replay target | Exact prefix identifiers enter forward computation but receive no historical-prefix teacher-forced loss. |
| Entity and geometry trust are separate | State-bank fields, eligibility masks, normalizers, losses, and metrics are separate. |
| Premature STOP needs only target-specific absence | A target-scoped non-coverage mode checks every prior prediction against one target while keeping all unrelated prior owners unknown; duplicate negatives still require full resolved coverage. |
| Unknown ownership has zero direct gradient | Loader validation excludes unknown or ambiguous status only from its corresponding entity or geometry term. |
| Fixed offline bank for the first screen | The trainer has no online collector or refresh loop. |
| Offline mixed correction successor | Orchestration performs rollout, StateBank construction, and the next training run as separate stages; each training run still consumes exactly one immutable bank. |
| Positive sampled-path successor | One admitted sampled path provides exact context; only verified first-occurrence rows through its last added owner receive gradient, with equal total loss weight per image. |
| Truthful off-policy replay | The bank keeps the trajectory-generating checkpoint identity, while the run separately records the compatible training warm-start checkpoint identity. |
| Geometry-sorted primary and random-order ablation | Each source checkpoint requires its own bound bank and run; cross-bank reuse fails. |
| No final architecture yet | Produced adapters use unchanged ordinary greedy Qwen3-VL inference. |
| Independent implementation race before full training | Tasks end after the two smoke levels and prohibit the formal 256-image launch. |

## Decisions

### 1. Keep research admission outside the trainer

The state-bank builder and review process decide whether an event is positive,
harmful, unknown, entity-eligible, or geometry-eligible. The trainer validates
these declarations and fails on contradictions; it does not infer truth from
official annotation mismatch, intersection-over-union thresholds, or class
names.

This keeps human and scientific judgment auditable and prevents each
implementation agent from silently inventing a different admission policy.

**Alternative rejected:** infer labels inside the loss from ground truth and
generated boxes. This would conflate omitted annotations, entity discovery,
and geometry quality.

### 2. Use exact token replay, not text reconstruction

Each event stores the executed prompt and prefix token identifiers and their
hashes. Candidate token identifiers and selected causal intervals are also
stored. A narrow replay owner combines those exact identifiers with the
existing image preparation and Qwen position-building path. It verifies token,
image, tokenizer, special-token, prompt, checkpoint, and processor identity
before forward computation.

No path may decode stored tokens to text and tokenize them again. Human-readable
text is optional evidence only.

**Alternative rejected:** render the prefix through the normal training
template. Semantically identical text can produce a different physical token
trajectory and invalidate the same-prefix experiment.

### 3. Represent each candidate as an isolated packed segment

Every positive or harmful candidate is a separate causal segment containing
the same exact image and prefix plus that candidate's continuation. Existing
segment-isolated no-padding packing and Qwen forward are reused. All candidates
for one calibration event are atomic for planning: they must be present in the
same planned optimizer step so the grouped loss is complete.

Companion typed calibration metadata maps:

- event and candidate identity;
- positive, harmful, or diagnostic role;
- coverage status;
- selected target and logits positions;
- nullable physical owner and nullable owner-resolution interval, with an
  explicit single-token score interval for a premature terminal candidate;
- first wrong coordinate and accepted coordinate set; and
- intended token type at each selected site.

The metadata complements `TokenSequence`; it does not redefine ordinary
`TokenAtom` semantics or make dense labels authoritative.

**Alternative rejected:** one padded candidate batch in a new research-only
trainer. That would duplicate position, visual replacement, runtime, and
checkpoint behavior already owned by CoordExp-Swift.

### 4. Separate branch selection from coherent-row continuation

The existing normal supervised mode remains unchanged. A separately named
rollout-calibration mode may enable:

- entity-transition preference;
- first-wrong-coordinate preference; or
- the coordinate selected-site token-type gate alone as a matched control; or
- both terms jointly.

It omits ordinary full-row base cross-entropy and consumes only rollout-derived
selected sites. The loss runner keeps per-term eligibility and complete
planned-step denominators. A joint run normalizes transition and geometry
separately before applying their configured weights.

The gate-only control reuses the coordinate event stream and selected sites,
sets both research-objective weights to zero, and differs from the coordinate
package only by removing the first-wrong-coordinate preference term.

All selected logits used by the research objectives are converted to 32-bit
floating point before `logsumexp`, `softplus`, normalization, or metric math.
For entity transition, each valid positive is compared with the harmful branch
at their first divergent token, where both paths share the same causal history.
Aliases are first grouped by physical owner and reduced by the strongest
admitted alias. The event asks that at least one distinct valid owner beat the
harmful token by the configured margin. A premature terminal negative has null
owner metadata and contributes exactly its terminal token at the shared
boundary.

The winning positive path then receives a continuation loss from the first
divergence through row closure. Schema-and-description sites and trusted
coordinate sites are mean-normalized separately before their non-empty group
means are combined. Untrusted geometry receives no exact coordinate-token
continuation loss. The transition term therefore changes the greedy branch
decision while the continuation term teaches a coherent valid row. It never
compares a summed long-row likelihood with a one-token terminal likelihood.

**Alternatives rejected:**

- canonical supervised-fine-tuning replay, because the pilot tests direct
  correction at own-prefix failures;
- summed whole-row likelihood against a shorter harmful path, because horizon
  length would determine the comparison; and
- Gaussian coordinate targets, because reviewed discrete boundary tolerance
  and physical-owner trust are the intended supervision.

### 5. Gate intended token type at every selected research site

The small token-type gate remains active in rollout-calibration mode. Its
allowed group comes from typed event metadata and the intended valid phase,
not automatically from the harmful token identifier.

Positive and harmful candidate sites included in a research score receive a
declared intended type. For an ordinary harmful duplicate branch, the phase
still distinguishes schema, description, and coordinate positions. For a
premature terminal branch at the shared boundary, the intended type is
object-row schema; applying a terminal-output gate at that same boundary would
contradict the treatment.

The loader rejects a selected gradient site without one unambiguous intended
type. This is the only generic token-level preservation term in the pilot.
Gate-site identity is segment plus logits position. All declarations for that
site must agree on intended type or validation fails; repeated identical
declarations are counted once. The loss averages distinct sites within an
event, then averages eligible events over the complete optimizer step.

### 6. Freeze every state bank before its training stage

Collection, review, and split assignment finish before each optimization
stage. Every bank manifest binds all records to the checkpoint that generated
their exact prefixes and declares image-grouped train and evaluation
partitions. StateBank refresh remains external orchestration: a model first
finishes training, ordinary inference produces new trajectories, a new bank is
assembled, and only then may a later training process consume that immutable
bank.

Geometry-sorted and random-order checkpoints use different bank identities and
different runs. Unrelated cross-bank prefix or candidate reuse is rejected.

The default remains strict on-policy replay: the warm-start checkpoint must
equal the StateBank trajectory source. A separately declared off-policy mode
may relax only the adapter and selected-token embedding-payload equality. Base
configuration, tokenizer, token identity, special-token identity, and
processor identity must still match exactly. The run receipt records both
checkpoint identities and the explicit off-policy status. Neither identity may
be rewritten to make them appear equal.

**Alternative rejected:** periodic refresh inside the trainer. It obscures
which model produced which state and is unnecessary for the bounded staged
comparison.

### 7. Reuse current training and inference owners

The implementation extends current config, data/supervision, loss, and pipeline
assembly seams. It continues to use:

- current Qwen component and image loading;
- segment-isolated packing and position construction;
- existing Weight-Decomposed Low-Rank Adaptation loading from a source
  checkpoint;
- existing optimizer, gradient clipping, schedule, and Accelerate runtime;
- current rank-zero run records and checkpoint payloads; and
- current inference backend and detection evaluator after training.

No new external dependency is planned.

### 8. Keep the public configuration small

The strict configuration needs only enough information to select the
rollout-calibration mode, state-bank manifest, enabled objective terms,
objective weights and fixed margins, token-type-gate weight, and source
adapter. One default-false switch permits compatible off-policy StateBank
replay. Collection policy and scientific thresholds remain in the state-bank
manifest and research unit rather than becoming trainer knobs.

Implementation agents may choose the narrowest existing internal owner and
concrete class names. They may not add speculative plugin registries, generic
policy engines, online collectors, or alternate runtimes.

### 9. Publish compact evidence through existing run artifacts

The resolved config and run record bind the state-bank identity. Each training
log row includes weighted and raw research losses, eligible event counts,
positive/harmful margins, coordinate-mass margins, token-type legal mass,
unknown/rejected counts, and finite status. Rank zero writes one compact bank
validation receipt and smoke receipt; ranks do not create parallel evidence
trees. An off-policy run additionally records the trajectory-source identity,
training-warm-start identity, their compatible shared identities, and the
explicit replay mode.

Raw review assets and state-bank source files remain outside the run tree and
are referenced by identity and checksum.

### 10. Reuse the same pipeline for positive sampled-path imitation

The positive-path profile is a narrow additional use of StateBank replay, not a
second trainer. Each event contains one positive candidate and no harmful
candidate. Historical sampled rows remain part of the exact causal prefix but
receive no gradient. The selected complete row provides two optional site
groups: schema-and-description and trusted coordinates. Each non-empty group is
averaged in 32-bit floating point, their means are averaged, and the result is
multiplied by a precomputed image-balanced event weight.

The assembler, not the trainer, owns route admission, physical-owner trust,
last-added-owner truncation, and deterministic batch-fit exclusion. The
trainer validates the declarations and preserves old profile behavior. This
keeps the scientific policy visible in the research artifact while reusing the
existing exact replay, optimizer, checkpoint, and inference paths.

The executed 512-event screen shows why this remains a research surface rather
than a final objective. It shifts greedy output toward the owner set represented
by the selected sampled routes, but loses ordinary owners at nearly the same
rate and regresses outside admitted images. Only 118 of 238 route-added owners
are direct positive-row event targets, so the result does not isolate direct
owner-wise imitation. A matched-arm preservation-aware successor requires a
separate research contract; it is not silently added to this single-route
profile.

## Risks / Trade-offs

- **Incorrect review labels create direct harmful gradients** -> validate
  physical-entity identifiers, preserve unknown as zero gradient, require
  review provenance, and retain the user spot-audit gate from the research
  unit.
- **Positive and harmful paths have different lengths** -> compare only their
  first divergent token, then train the positive continuation with separate
  mean-normalized entity/schema and trusted-geometry groups.
- **A locally rescued row harms later coverage** -> require equal row-and-token
  budget admission evidence before the event becomes gradient-eligible.
- **Candidate groups split across optimizer steps** -> make event groups atomic
  in the packing and planned-step plan and fail on incomplete groups.
- **The gate conflicts with a harmful terminal token** -> derive intended type
  from the valid action at the shared boundary and validate one type per site.
- **Physical aliases receive excess weight** -> group candidates by stable
  per-image physical entity and take one maximum alias-path score before
  smooth aggregation across owners.
- **Image leakage inflates results** -> split by image identity and bind split
  groups in the bank manifest.
- **Repeated exact prefixes increase computation** -> accept the cost for this
  bounded screen; optimize only after scientific value is demonstrated.
- **A local margin improves without rollout benefit** -> keep free-rollout
  evaluation outside the training objective and stop according to the research
  unit rather than expanding implementation.
- **One positive route overwrites other useful routes** -> report targeted-
  owner gain and ordinary-owner retention separately; do not scale the profile
  unchanged when final owner coverage is flat or worse.

## Migration Plan

1. Add the state-bank schema and validation without changing normal training.
2. Add exact-token replay and a no-gradient parity test against a frozen
   reference event.
3. Add transition, geometry, and intended-type-gate loss math with synthetic
   unit tests.
4. Wire the explicit rollout-calibration config mode into the existing
   training pipeline.
5. Run the one-event smoke and the 8-to-16-state smoke.
6. Stop for review before any formal 256-image launch.
7. Add the default-off compatible off-policy replay seam after the original
   screen establishes the need for a trajectory-refresh test.
8. Execute refresh as separate rollout, assembly, and training stages rather
   than adding an online collector to the trainer.

Rollback is deletion or disabling of the explicit rollout-calibration mode.
Normal supervised configs, inference configs, and checkpoint loading remain
unchanged throughout.

## Open Questions

No research-direction decision remains open for implementation. Before the
implementation worktrees are forked, the lead agent must publish one immutable
smoke fixture manifest that binds the source checkpoint, exact event set,
state-bank checksum, learning rate, gradient-clipping value, optimizer-step
count, smooth-maximum temperature, preference margin, token-type-gate weight,
and random seeds. Every implementation consumes that same fixture.

Formal-screen optimizer values are a later lead decision made once after the
implementation comparison and smoke health review. Once selected, they are
shared across matched arms and must not be tuned per checkpoint, image, or
training seed.

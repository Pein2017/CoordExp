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
- compute entity-transition, first-wrong-coordinate, and rollout-site
  token-type-gate losses in 32-bit floating point;
- preserve separate event masks, normalization, metrics, and provenance;
- reuse existing model loading, packing, runtime, optimizer, artifact, and
  checkpoint owners; and
- support deterministic one-event and small-state smokes before the formal
  training screen.

**Non-Goals:**

- online rollout collection during optimization;
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
| Keep a small token-type gate | Every selected positive and harmful site has one intended phase type; a harmful terminal boundary is gated as object-row schema. |
| Prefix is conditioning, not a replay target | Exact prefix identifiers enter forward computation but receive no historical-prefix teacher-forced loss. |
| Entity and geometry trust are separate | State-bank fields, eligibility masks, normalizers, losses, and metrics are separate. |
| Unknown ownership has zero direct gradient | Loader validation excludes unknown or ambiguous status only from its corresponding entity or geometry term. |
| Fixed offline bank for the first screen | The trainer has no online collector or refresh loop. |
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

### 4. Add one explicit rollout-calibration loss mode

The existing normal supervised mode remains unchanged. A separately named
rollout-calibration mode may enable:

- entity-transition preference;
- first-wrong-coordinate preference; or
- both terms jointly.

It omits ordinary full-row base cross-entropy and consumes only rollout-derived
selected sites. The loss runner keeps per-term eligibility and complete
planned-step denominators. A joint run normalizes transition and geometry
separately before applying their configured weights.

All selected logits used by the research objectives are converted to 32-bit
floating point before `logsumexp`, `softplus`, normalization, or metric math.
For entity transition, candidate aliases are first grouped by physical owner
and reduced by maximum path score. The configured smooth maximum is applied
only across the resulting distinct valid-owner scores. A premature terminal
negative has null owner metadata and contributes exactly its one terminal-token
log probability at the shared boundary.

**Alternatives rejected:**

- canonical supervised-fine-tuning replay, because the pilot tests direct
  correction at own-prefix failures;
- one whole-row cross-entropy target, because several next entities are valid
  and only a short owner-resolving path is the decision surface; and
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

### 6. Freeze state banks before training

Collection, review, and split assignment finish before optimization. The bank
manifest binds all records to one source checkpoint and declares image-grouped
train and evaluation partitions. State-bank refresh is not a trainer feature
in this change.

Geometry-sorted and random-order checkpoints use different bank identities and
different runs. Cross-bank prefix or candidate reuse is rejected.

**Alternative deferred:** periodic on-policy refresh. It changes the scientific
question and adds worker orchestration, versioning, and new review burden before
the fixed-bank treatment is known to work.

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
adapter. Collection policy and scientific thresholds remain in the state-bank
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
trees.

Raw review assets and state-bank source files remain outside the run tree and
are referenced by identity and checksum.

## Risks / Trade-offs

- **Incorrect review labels create direct harmful gradients** -> validate
  physical-entity identifiers, preserve unknown as zero gradient, require
  review provenance, and retain the user spot-audit gate from the research
  unit.
- **Candidate-path length affects summed likelihood** -> freeze the shortest
  physical-owner-resolving interval in the bank and log summed, mean, and
  per-token scores together.
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

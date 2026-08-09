## Context

The existing rollout-calibration route replays exact integer prompt, image-pad,
prefix, and candidate tokens through the standard Qwen forward pass. It already
supports reviewed owner-transition events and complete-row imitation, but the
owner-transition objective compares only the first divergent token and then
adds positive-row imitation. It therefore cannot isolate the effect of a
complete duplicate-row negative. Existing provenance also assumes that every
candidate was generated under the same prefix used for replay; that is false
for a recovery row moved before a duplicate burst and for rows replayed after
duplicate deletion.

The implementation must preserve the ordinary model, packing, Accelerate,
checkpoint, and inference paths. The new scientific labels are created offline
from reviewed physical-owner assignments. Official ground-truth mismatch alone
does not create a negative label.

## Goals / Non-Goals

**Goals:**

- represent reviewed duplicate bursts, downstream recovery rows, and
  duplicate-deleted prefixes without falsifying provenance;
- compare verified new-owner and covered-owner duplicate rows using a pure,
  field-balanced complete-row preference loss;
- imitate validated rows under explicitly counterfactual cleaned prefixes;
- support matched local, cleaned, combined, positive-only, and Source-only
  experiments through the existing trainer;
- normalize training credit by burst and image and expose enough diagnostics to
  audit the effective treatment;
- keep all new behavior opt-in and isolated from historical profiles.

**Non-Goals:**

- automatic physical-owner truth from category and bounding-box overlap alone;
- a persistent covered-set memory, object slots, detector, or online trainer;
- negative training on unresolved official false positives;
- changing the object wrapper, coordinate vocabulary, model forward graph,
  checkpoint payload, or ordinary inference behavior;
- claiming that a counterfactual cleaned trajectory was naturally sampled.

## Decisions

### Add an experiment-specific duplication training capability

The current own-prefix calibration change remains the implementation base, but
duplication treatment receives its own OpenSpec change and opt-in profiles. This
keeps historical transition and coordinate behavior unchanged and avoids
silently changing the meaning of existing experiments.

Alternative: overload `entity_transition_eligible`. Rejected because its
first-divergence comparison and mandatory positive continuation cannot isolate
the duplicate negative.

### Preserve generation and replay provenance separately

An optional typed duplicate-trajectory evidence record belongs to each new
event. It records trajectory and burst identity, replay context, the true
candidate-generation prefix hash, the replay-prefix hash, retained first-owner
row, duplicate rows, removed rows, recovery row, and burst credit. Historical
events retain the existing equality requirement. New events may differ only
when their typed evidence explicitly declares an exact-self-prefix transplant
or a complete-row-deletion rewrite.

For cleaned events, the assembler operates on exact stored integer row slices.
It does not decode and retokenize rows. Original and rewritten trajectories are
published separately.

Alternative: overwrite candidate generation hashes with the replay hash.
Rejected because it produces false provenance and makes later scientific audit
impossible.

### Use a pure field-balanced complete-row pairwise loss

For a verified new-owner row `U` and confirmed duplicate row `D` under replay
prefix `P`, the new loss computes separate mean log probabilities for
description tokens and trusted coordinate tokens, then averages those two
field scores:

```text
row_score = 0.5 * description_mean_log_probability
          + 0.5 * coordinate_mean_log_probability

loss = softplus(margin - row_score(U | P) + row_score(D | P))
```

Schema and delimiter tokens remain subject to the token-type gate but do not
dominate the pairwise score. Both candidate rows require trusted physical-owner
and coordinate evidence in the primary profile. The implementation logs each
field score and gradient-bearing token count.

Alternative: compare only the first divergent token. Rejected because the
research question concerns a coherent physical-owner row, and because multiple
same-category instances often separate only in coordinate fields.

Alternative: sum all row log probabilities. Rejected because row length and
fixed wrapper syntax would create avoidable bias.

### Treat cleaned trajectories as counterfactual row imitation

A cleaned event contains one positive row under a prefix derived only by
deleting confirmed duplicate complete rows. The first accepted occurrence of
each owner is retained. Automatic suffix construction stops at the first
invalid, category-error, owner-ambiguous, binding-ambiguous, or otherwise
untrusted row. Every successive rewritten transition is validated.

The existing positive complete-row imitation primitive remains the loss owner.
Only provenance, family admission, scheduling, and logging change.

Alternative: train the complete remaining suffix after one burst edit without
validation. Rejected because downstream rows were generated under a different
history and can become unsupported after editing.

### Match treatment dose with explicit controls

The frozen Source checkpoint is evaluation only. The runnable controls are:

- Source-preservation-only using existing Source complete-row events;
- recovery-positive-only using the same prefixes and positive rows but no
  duplicate negative.

The treatments are local duplicate rejection and recovery,
duplicate-cleaned imitation, and their combination. Banks match image
allocation, optimizer updates, total mechanism credit, seeds, and Source
preservation dose. The combined bank allocates its mechanism credit between
local and cleaned events without increasing the total dose.

### Normalize by burst, then image

All comparisons from one correlated duplication burst share one total unit of
mechanism credit. The entry boundary receives half the credit and later burst
states share the other half; a one-row burst assigns the full unit to entry.
Image-level normalization then prevents images with many bursts from
dominating. The resulting scalar is carried through the existing event weight
path and must affect the local pairwise loss as well as complete-row imitation.

### Stratify every multi-family optimizer window

Every planned optimizer window for a multi-family profile contains every
enabled loss family across ranks. For the combined profile this means local
duplicate rejection and duplicate-cleaned imitation. For local duplicate
rejection and recovery this means Source preservation and local duplicate
rejection. The trainer emits deterministic family-stratified windows and
validates their completeness before forward. This avoids zero-denominator
failures and prevents JSONL ordering from silently changing the treatment.

### Keep the assembler local to the research unit

One script consumes exact rollout token rows and a reviewed physical-owner
ledger and emits immutable matched StateBanks plus an allocation receipt. It
reuses existing row-slicing and StateBank assembly helpers but does not create a
new general trajectory framework.

## Risks / Trade-offs

- **Recovery row was not naturally sampled at an earlier prefix** -> Preserve
  its true source prefix, label the transplant, require finite teacher-forced
  feasibility, and report exact-self and cleaned-prefix results separately.
- **A later row is a localization correction rather than a duplicate** ->
  Require a usable earlier occurrence, trusted physical owner, trusted category
  and binding, and exclude uncertain or irregular cases from training.
- **Cross-category pairs teach category frequency rather than owner exclusion**
  -> Make same-category pairs the primary mechanism stratum and report
  cross-category pairs separately.
- **Long bursts dominate optimization** -> Normalize by burst and image and
  verify effective weights in deterministic tests and receipts.
- **Duplicate suppression only causes termination or invalid output** -> Measure
  the destination of released decisions and do not promote the treatment on
  duplicate count alone.
- **Combined family scheduling changes the effective objective** -> Require
  every planned step to contain each enabled family and log global
  denominators.
- **Source replay is mistaken for native-capability preservation** -> Treat it
  only as a matched drift control and evaluate never-trained images directly.
- **Research-only schema expands stable surface area** -> Keep all fields and
  profiles opt-in; historical banks and profiles reject the new metadata and
  remain behaviorally unchanged.

## Migration Plan

No data or checkpoint migration is required. Existing configs and StateBanks
continue to validate under historical profiles. New duplicate training banks
are assembled from immutable rollout evidence and used only by new profiles.
Rollback consists of removing the new configs and using the unchanged Source
checkpoint; ordinary inference never depends on the StateBank.

## Open Questions

- Whether enough same-category recovery pairs exist for a statistically useful
  primary stratum is determined by the reviewed census, not by the config
  schema.
- Whether the first 256-image screen should expand to 1,024 or 2,048 images is a
  research decision based on event count and observed treatment behavior, not a
  stable implementation requirement.

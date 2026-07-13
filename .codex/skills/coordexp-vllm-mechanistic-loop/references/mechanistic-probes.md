# Mechanistic Probe Design

Read this after the run is bounded and before comparing checkpoints, selecting a
probe family, or escalating an experimental handle toward training or rollout.

## Comparability matrix

Record checkpoint and adapter/loading surfaces; prompt/template, tokenizer, and
special-coordinate-token handling; dataset slice, sample IDs, ordering,
geometry, and resize policy; teacher forcing versus free rollout, decode and
parser settings; and every raw, scored, traced, visualized, or probe artifact
root. Compare effective surfaces, not coarse checkpoint-family labels.

## Probe ladder

Treat this as a branch-deciding menu, not a mandatory sequence:

1. Case and onset ledger: validity, emitted count, pair/component onset,
   component growth, row position, description/class, basin, and termination.
2. Slot and boundary readouts: keep `x1`, `y1`, `x2`, `y2`, `box_end`,
   stop/continue, and schema tokens separate.
3. Prefix and guidance splits: full/no/wrong history, same-description
   competitors, GT guidance, and generated bad prefixes.
4. Coordinate-token geometry: embeddings, LM-head neighborhoods, valid-token
   mass, target-bin ranks, and local smoothness.
5. Hidden-state and residual flow: object identity, region availability,
   coordinate basin, and target-bin margins across layers.
6. Attention routing: route evidence only, not causal proof.
7. Causal intervention: patch, ablate, mask, or source-swap after pinning the
   row, window, and slot, with paired controls.
8. Sublayer decomposition: attention, MLP, residual, final norm, and LM head.

Skip a rung that cannot change the current route decision; revisit it when later
evidence makes it consequential.

## Causal escalation gate

For a bridge, pulse, cursor, coverage state, or other causal handle:

1. Reuse a checkpoint or train only the minimum arm needed to expose the handle.
2. Pin 16–32 attributable examples at one divergence or pre-row boundary.
   Compare correct source with exact-off and the smallest relevant source-swap,
   token-permutation, position-only, and magnitude-matched-random controls.
3. Score the first distinguishing token, semantic span, first coordinate, full
   geometry, validity, and semantic/geometry binding. Free-run at most one row
   when later behavior would confound the causal question.
4. Require the correct source to uniquely beat semantic controls without a
   safety regression; generic continuation gain is insufficient.
5. Only after local specificity passes, run a small own-prefix safety pilot.
   Long rollout, crossover, additional arms, and multi-seed confirmation come
   after both gates pass.

Stop at the earliest predeclared terminal result. A failed screen retires the
tested target, lifetime, and delivery route—not every intermediate
representation. Teacher-forced representation evidence and free-rollout causal
evidence are separate gates. Execution receipts are not hypothesis progress.

## Interpretation

- Separate symptom, candidate mechanism, causal evidence, and alternatives.
- A positive intervention identifies a usable handle, not automatically the
  final runtime interface; test duration, span, prefix dependence, and
  synthesizability first.
- Keep architecture updates conditional.
- Use `probe handle mismatch` or `inconclusive-needs-mechanism-panel` when a
  negative result misses the suspected surface.
- Treat attention-only findings as route hypotheses until causal intervention.
- Distinguish local slot rescue from full-span recovery, and x1-boundary basin
  selection from later coordinate-slot movement.
- Reconcile aggregate metrics with stronger sample-level internal evidence.
- Treat perturbation checkpoints as representation-shaping interventions unless
  their training/runtime contract proves a stronger claim.

## Failure modes

- Answering a mechanism question with AP/mAP deltas alone.
- Designing the final architecture first and choosing probes to advance it.
- Promoting an experimental handle into a required module before testing scope
  and replaceability.
- Studying only normal samples or flattening slot/condition evidence.
- Misidentifying duplicate-burst onset or treating attention as causal proof.
- Training every eventual arm before the primary handle passes specificity.
- Continuing after the earliest ordered terminal label without separately
  authorizing localization work.
- Calling a working-tree contract frozen before its decision logic and gate are
  committed or hash-bound.
- Spending more on overlapping agents, polling, narration, or audit-of-audit
  loops than on the scientific comparison they protect.

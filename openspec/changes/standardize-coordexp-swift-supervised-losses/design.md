## Context

See `proposal.md` for motivation. The current strict schema puts
`coord_gaussian_rps` inside `losses.protected`, permits several gate weights,
and lets loss-runner branching determine which denominators and metrics exist.
The streaming loss protocol already owns complete planned-step denominators,
fp32 term math, and Accelerate/DDP compensation; those proven semantics must
survive the config and composition cleanup.

The change crosses strict config, loss construction, streaming train/eval
reduction, logging projection, and all current supported configs. Historical
configs and artifacts are provenance, not migration targets.

This is the third change in the strict sequence
`reconcile -> decompose -> losses -> observability`. Work begins only after the
first two changes are synced and archived at exact recorded commits. Their
post-decomposition owner graph and the train/eval cache determinant payloads
and hashes form the entry baseline. Loss config and objective code are not
cache semantic owners; if an implementation edit changes either determinant
payload or hash, the change stops for contract review and MUST NOT build or
publish a second cache target.

## Goals / Non-Goals

**Goals:**

- Make one canonical enabled SFT objective and one explicit protected-gate
  ablation impossible to confuse.
- Give protected and auxiliary terms closed, typed composition semantics
  without a plugin framework.
- Separate semantic loss values from configured weighting and backend-only
  gradient compensation.
- Make zero-weight behavior testable at construction, denominator, autograd,
  reduction, and artifact boundaries.
- Migrate supported configs atomically with strict schema and artifact tests.

**Non-Goals:**

- RL, rollout-derived, hidden-state, policy, value, KL, or reward loss
  composition.
- User-defined import paths, callable config, entry points, or a public loss
  registry.
- New normalizers, distributed backends, loss schedules, or learnable loss
  weights.
- Rewriting historical configs, resolved configs, or completed artifacts.
- Claiming a speed or peak-memory improvement from graph omission without a
  separate benchmark.

## Decisions

### 1. Use a strict discriminated protected-gate mode

The protected config remains structurally small:

```yaml
losses:
  normalizer: segment_balanced
  protected:
    base_ce:
      weight: 1.0
    token_type_gate:
      mode: enabled                 # or: zero_weight_ablation
      weight: 0.1                   # or exactly 0.0
      groups: [desc_text, schema, coordinate, eos]
  auxiliary:
    coord_gaussian_rps:
      weight: 1.0
      # typed term-specific fields
```

`mode=enabled` pairs only with `0.1`; `mode=zero_weight_ablation` pairs only
with `0`. Base CE pairs only with `1.0`, and the gate group tuple must exactly
match the canonical ordered tuple. The explicit mode prevents a stray zero
from silently changing research meaning and remains visible in
`resolved_config.json`.

Alternative considered: infer ablation solely from `weight: 0`. Rejected
because it cannot distinguish an intentional contrast from an accidental
disabled baseline. Another alternative was a general per-term `enabled` flag;
rejected because it creates two owners (`enabled` and `weight`) and does not
encode the protected gate's diagnostic behavior.

### 2. Keep composition closed through `TokenLossBinding`

The loss package owns a frozen internal binding per implemented token term.
Each binding carries the canonical name, role (`protected` or `auxiliary`),
normalizer (`segment_balanced`), and zero policy (`forbid`,
`detached_diagnostic`, or `omit`). Assembly selects from an explicit closed
mapping or tuple compiled with the code. Config models remain term-specific and
typed; they do not expose factories or implementation identifiers.

The binding is metadata used by one deep loss-runner implementation, not a
second runtime abstraction. Existing term objects continue to own per-atom
math; the runner owns streaming lifecycle, aggregation, and the final bundle.

Alternative considered: a dynamic registry/factory protocol. Rejected because
the repository has three known token losses, strict research semantics, and no
need for third-party installation. A dynamic surface would make config
validation and artifact provenance weaker while adding indirection.

### 3. Give each zero policy one execution path

- `base_ce` uses `forbid`: validation rejects zero or absence.
- `token_type_gate` uses `detached_diagnostic` only in the named ablation:
  denominator and fp32 per-atom math still run inside a no-grad boundary;
  raw/count/finite diagnostics are finalized, weighted is zero, and no gate
  tensor enters `total_loss` or the autograd graph. Because this is still a
  protected diagnostic, its finite status remains part of the existing
  all-rank pre-backward safety gate.
- `coord_gaussian_rps` uses `omit` at weight zero: do not instantiate the term,
  build/gather its denominator, call its math, add it to the bundle, run a
  finite check, or emit its fields.

The gate diagnostic has its own diagnostic finite status. A finite detached
gate value cannot change the pure-CE ablation's objective or gradients. A
non-finite protected diagnostic remains unsafe: it is normalized for the
canonical row, participates in the existing all-rank scalar consensus, and
prevents backward and optimizer update. This retains the repository's
fail-closed policy without making the diagnostic differentiable.

Alternative considered: compute every zero-weight term and multiply by zero.
Rejected because `0 * NaN` is unsafe, graph construction is unnecessary, and
optional-term metrics would misleadingly imply participation. Omitting the
protected gate was also rejected because the approved ablation requires the
same legality diagnostic for comparison.

### 4. Separate semantic values from backward contributions

For each active term the streaming reducer tracks three concepts:

1. raw semantic value: global planned-step segment-balanced term value before
   configured weighting;
2. weighted semantic value: raw value multiplied by configured weight;
3. differentiable local contribution: rank-local numerator divided by the
   global denominator, weighted, and multiplied exactly once by world size to
   compensate Accelerate/DDP mean-gradient reduction.

Only (3) participates in backward. Finalization derives (1) and (2) from
all-rank sufficient statistics without the backend scale, then constructs
`loss/total` from weighted objective terms. This prevents the existing
`raw_loss` label from accidentally representing a backend-compensated local
contribution.

Alternative considered: retain the current tensor as both raw metric and
backward contribution. Rejected because its value is rank- and world-size-
dependent even when the global objective is not.

### 5. Replace ambiguous per-term logging names

Computed terms use `loss/<term>/raw` and `loss/<term>/weighted`; the optimized
sum remains `loss/total`. Counts, denominators, and finite fields follow the
same term namespace. The old ambiguous `loss/<term>` field is not dual-written:
all current consumers and tests migrate in this change, while historical JSONL
retains its commit-bound schema.

The same projection is used for train and forward eval. Gate-ablation fields
remain visible; omitted auxiliary fields are absent as a family. The artifact
writer continues to normalize non-finite computed values to JSON `null` and
list their exact keys.

Alternative considered: retain `loss/<term>` as a weighted alias. Rejected
because permanent aliases undermine the goal of an unambiguous compact row and
create two names for one value.

### 6. Migrate only current supported configs

The migration enumerates config files through the repository's canonical
catalog/current roots, rewrites enabled gates to `0.1`, marks pure-CE variants
with `zero_weight_ablation`, preserves the exact gate groups, and moves any
coordinate term into `losses.auxiliary`. Config mutation tests alter each
protected constant, mode/weight pair, group tuple, auxiliary placement, and
unknown hook to prove fail-closed behavior.

Historical and archived config roots are not edited. There is no compatibility
alias for `protected.coord_gaussian_rps`; its error should name the new field.

Alternative considered: loader-side automatic migration. Rejected because it
would hide the objective change, make fingerprints harder to interpret, and
allow stale configs to appear current.

## Risks / Trade-offs

- [Changing active gate weights changes research meaning] → Treat config edits
  as an explicit breaking migration, inspect every current diff, and retain
  resolved mode/weight/groups in run artifacts.
- [A term can be scaled twice or metrics can inherit DDP compensation] → Add
  unequal-rank denominator fixtures that compare raw/weighted values,
  parameter gradients, and optimizer updates with a world-size-one reference.
- [Gate-ablation no-grad work can accidentally affect objective math or weaken
  finite gating] → For finite diagnostics, test the same inputs against pure
  base CE and assert identical objective/gradients/updates. Separately inject a
  non-finite gate diagnostic and assert all ranks skip before backward.
- [Optional zero still constructs graph or persistent tensors] → Instrument
  term construction/calls, denominator gathering, saved-tensor/autograd edges,
  bundle fields, and retained state. Treat this as correctness evidence only;
  make no efficiency claim without a separate benchmark.
- [Logging rename breaks current consumers] → Search code, tests, docs, and
  scripts for the old field family; migrate supported consumers atomically and
  add exact row-schema tests for train/eval, gate ablation, optional omission,
  and non-finite normalization.
- [Typed binding grows into a framework] → Keep it private, immutable, and
  closed; do not add generic discovery, lifecycle callbacks, import strings,
  or pass-through factory layers.

## Migration Plan

1. Pin the exact reconcile/decompose commits, final owner graph, admitted
   train/eval cache identities, and an exact command manifest. Verify the
   untouched baseline and stop if the prerequisites disagree.
2. Add failing strict-config and composition tests for the canonical baseline,
   named gate ablation, auxiliary placement, optional omission, and forbidden
   dynamic hooks.
3. Introduce the closed binding metadata and refactor streaming planning,
   micro-step computation, and finalization around the three zero policies and
   separated semantic/backward values.
4. Add world-size-one and unequal-rank reduction/gradient/update parity tests,
   followed by train/eval artifact-schema tests.
5. Migrate every current supported config and consumer; validate the complete
   current config inventory. Leave historical roots unchanged.
6. Recompute the determinant payloads and hashes and require byte-for-byte/hash
   equality with the entry baseline. Any difference is a blocking contract
   failure; do not materialize a replacement cache.
7. Update canonical operator docs and run focused tests, a production-shaped
   distributed vertical smoke, strict OpenSpec validation, and independent
   contract plus overdesign audits.

The command manifest is frozen before implementation and records exact cwd,
environment, command, config, world size/devices, artifact root, and expected
evidence for each test or probe. A reviewed amendment must retain the original
entry and explain why it changed. Before every GPU-backed action, obtain fresh
user authorization and record quantitative bounds for planned steps, model
forwards, world size, cache/materialization passes (zero for this change), wall
time, peak GPU memory, and artifact payload. Crossing a bound stops the action;
it does not silently expand the evidence budget.

Rollback is a source/config rollback to the pre-change commit together with
its matching configs. A new-schema config MUST NOT be run against old code, and
an old protected-coordinate config MUST NOT be silently accepted by new code.

# OTA-SQP-lite implementation plan

Implementation starts only after the entropy-reclaim worker completes and the
lead verifies that its cuts preserve the certified Image2299 route.

## Reuse boundary

Reuse the current evaluator, parser, immutable bindings, model/session loader,
DoRA cloning/restoration, cold checkpoint verification, radius assignment, and
receipt helpers from the certified-46, parallel-corridor, and projected-corridor
runners.  Do not create another evaluator, matcher, checkpoint format, model
loader, or distributed launcher.

The implementation should add one runner and one focused test file only unless
a proved shared extraction reduces the total surface without changing the
historical runners.

## Work packages

### 1. Frozen alias catalog

- Materialize candidate rows for all 13 missing owners.
- Reuse the production parser and global matcher to certify
  `Parent33 + alias + EOS` as a zero-debt 34-owner static target.
- Re-score aliases at the exact current pre-tail prefix.
- Retain at most two aliases per owner for gradient work.
- Seal row tokens, boxes, owner edges, margin panels, and hashes in the receipt.

Acceptance: a CPU-only fixture rejects occupied-owner, ambiguous, malformed,
and owner-set-losing aliases while admitting known strict `gt32` aliases and
zero-debt Hungarian row permutations of the same 34-owner set.

### 2. Complete-action margins and gradients

- Generalize the existing frozen-pair helper from one divergent token to every
  token in `row + EOS` under teacher forcing.
- Record target token, top competitor, real-prefix hash, margin, and gradient
  identity for each active position.
- Include `person > tie` and `person > chair` competition explicitly.
- Fail closed if the real current prefix is not a prefix of the natural route.

Acceptance: changing any target prefix or token makes the invariant fixture
fail; the known v18 first-coordinate and description margins reproduce.

### 3. Small constrained solver

- Build the gradient Gram matrix and express `Delta` in the active gradient
  span.
- Use installed SciPy SLSQP; add no dependency.
- Solve the max-final-min-margin problem with a bounded ordinary L2 trust
  region.
- Check primal residuals, strict safety floors, finite values, and predicted
  directional signs independently of solver success status.
- Treat an infeasible or numerically invalid solve as HOLD, never as a zero
  direction or reachability result.

Acceptance: deterministic synthetic gradients cover feasible, conflicting,
and solver-invalid cases; perturbing one constraint flips the expected result.

### 4. Automatic active cuts and natural admission

- Seed from the current G0--G6 bindings and low-slack current-route decisions.
- Evaluate one radius per rank using the existing eight-way panel.
- Select proper superset first; otherwise select the largest admissible working
  candidate with exact action progress.
- On owner loss, add the earliest causal divergence and relinearize.
- On match-equivalent route changes, rebind prefixes instead of enforcing whole
  token-route identity.
- Cold-reload selected working states and every promotion/terminal checkpoint.

Acceptance: fixtures distinguish owner-preserving alias changes from owner
exchange, and a rejected candidate cannot mutate the incumbent checkpoint.

### 5. Parameter-surface screen

- Add a no-update r16 baseline panel.
- Implement and verify function-preserving r16/alpha32 to r32/alpha64 adapter
  expansion without changing the base, target modules, dropout, or DoRA
  magnitude vectors.
- Verify new B-channel gradients are live and quantify whether their constrained
  component lies outside the r16 span.
- Discover the existing eight aligner merger linears and compute aligner-DoRA
  tangent evidence only; do not checkpoint or train them in this unit.

Acceptance: r32 step-zero logits, generated tokens, owners, counters, and
frozen payloads reproduce r16; zeroing both new A and B is rejected as a dead
expansion.

### 6. Production-shaped smoke and launch packet

- Run a one-relinearization, eight-rank no-update/sensitivity vertical slice.
- Bound model forwards, backwards, candidate decodes, stored gradient payload,
  rank agreement, wall time, and receipt size before the 16-update run.
- Re-run focused tests, collection, compile, source snapshot/hash checks,
  `git diff --check`, and cold save/load parity.
- Seal exact command, output root, tmux socket, environment, expected receipt,
  promotion gate, budget, and stop rule before launch.

## Implementation acceptance

Implementation is ready for GPU execution only when:

- all immutable identities match the predecessor;
- the alias catalog and global-matcher fixtures pass;
- solver residual and sensitivity checks pass;
- exactly the declared r16 language-DoRA surface is trainable in the main arm;
- r32 and aligner diagnostics cannot mutate or save training state;
- the eight-rank vertical slice cold-reproduces its selected behavior;
- the receipt distinguishes working debt, promotion, bounded negative, and
  mechanical HOLD.

No launch, r32 training, aligner training, tied-embedding update, Notion
mutation, commit, or push is implied by this plan.

## Historical CPU preparation boundary

The frozen bindings, bounded alias catalog, production global-matcher gate,
complete row-plus-EOS margin bindings, gradient-span Gram construction,
SLSQP final-margin trust-region solve, candidate/stop gates, and matched r32
factor expansion had CPU implementations and one focused nine-test contract.
At that checkpoint, the model/session loop, exact margin-gradient capture,
eight-rank candidate execution, cold checkpoint admission, and no-update
parameter-surface measurements had not run.  The final status below supersedes
that preparation boundary.

## Final execution status

All decision-bearing packages subsequently ran.  The valid r16 smoke and
continuation, r32 no-update tangent gate, and sole r32 matched step are sealed
in [results.md](results.md).  No owner promotion occurred, no second r32 update
is allowed, and aligner/tied training remained outside the opened gate.

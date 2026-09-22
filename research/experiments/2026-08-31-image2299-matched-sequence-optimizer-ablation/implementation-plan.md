# Matched sequence-optimizer ablation implementation plan

## Execution truth

Completed against the frozen plan: preflight v6 and all four v5 cells were
finalized by the cold 41-owner gate. The only successful cell is M+QP; M+CE and
G+CE exhaust their fixed CE schedules without a warm milestone, and G+QP's
feasible minimum norm exceeds the unchanged cap before warm decoding. The
finalizer and result record are authoritative; preflight v1--v5 and cell v1--v4
are technical-invalid lineage only.

## Admission (CPU/no-update)

1. Verify every input SHA in `unit.md`, reproduce M's 41-entry ordered ledger
   from `matcher.prediction_receipts`, construct G from `target.json`, and run
   the production parser/global matcher.  Persist the static-G receipt before
   loading a model; otherwise stop `static_route_hold`.
2. In one atomic preflight run, load the frozen r32 checkpoint, teacher-force
   M/G with zero residual, derive positives and the canonical JSON S_union
   hash/count.  Persist both TF captures; refuse changed S_union thereafter.
   Before persistence, serialize and bind the live runner SHA, checkpoint/readback,
   prompt-token/image, start/frozen surface and sentinel, ordinary/M/G route,
   positive/S_union, and basis/protected identities.
3. For each sequence, derive one protected-null basis after S_union freezes and
   reuse its exact normalized `D_s = ||W_s|| X_s B^T`, `B`, selected rows, and norm coordinate convention in QP
   and CE.  Existing implementation seams are `SparseOutputResidual`,
   `_capture_route`, `_protected_hidden_matrix`, `_row_basis`,
   `_full_vocab_violation`, `_evaluate_route`, and `_greedy_route` in the
   Image2299 protected-null/dyadic runners.

## Four independent cells

Launch M+QP, M+CE, G+QP, G+CE as one-GPU, world-size-1 processes only after they receive the atomic preflight receipt and its exact SHA; they recapture every bound identity. Resource accounting is owned by that one preflight plus each cell's declared bound. QP uses
HiGHS then SLSQP on the unit program and its one resulting residual.  CE runs
exactly the four pre-registered SGD traces and milestone selector in `unit.md`.
Every candidate records the common surface identity and performs the same
ordinary-greedy production gate.  Only selected warm successes spawn their
fresh cold verifier; no result can cause another LR, solver, route, row, rank,
cap, or step.

## Receipt schema and finalizer

The unit receipt must include immutable inputs and the live preflight runner
source SHA, M/G route and
ledger hashes, static-G receipt, M/G TF positive receipts, S_union IDs/count/
hash, per-sequence basis/protected records, parameter counts, per-cell QP or
CE trace/milestone selection, full-vocabulary/null/norm checks, warm/cold
gate details, frozen before/after surface snapshots, observed resources versus
the declared limits, and one of `cold_greedy_41_owner_success`,
`frozen_recipe_negative`, `certified_qp_infeasible`, or a specific technical HOLD.

Run the existing CPU binding checks before implementation review:

```bash
conda run -n ms python scripts/research/run_image2299_dyadic_norm_release_distillation.py --check-bindings
conda run -n ms python scripts/research/run_image2299_canonical_five_tie_protected_null_sentinel.py --check-bindings
```

After the new runner exists, its deterministic CPU static-G/S_union schema
check is the acceptance command; its GPU invocation is outside this planning
package.  A full 46/46 runner or artifact must never be imported as a mutable
surface.

The sole preflight invocation is `python
scripts/research/run_image2299_matched_sequence_optimizer_ablation.py
--preflight-run-id ID`; a cell instead receives `--preflight-receipt-run-id ID
--preflight-sha256 SHA256` alongside its own `--run-id` and `--cell`.

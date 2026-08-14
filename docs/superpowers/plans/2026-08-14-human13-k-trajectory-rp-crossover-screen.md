# Human-13 K-Trajectory RP-Crossover Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` for independent implementation
> tasks.  OpenSpec remains the sole scope and completion authority.

**Goal:** Implement and, only after a separate execution authorization, run a
bounded Human-13 screen that compares shared K-trajectory credit, sparse greedy
compilation, and exact proposal preservation under training RP values 1.0 and
1.10.

**Architecture:** Add an experiment-local evidence pipeline around the existing
Human-13 live spine.  Fresh batch-four K16 acquisition produces sealed policy
traces; pure projectors derive row credit and compiler sites; one packed
backward produces an exact fresh-AdamW proposal; the preservation arm projects
that actual delta; every private proposal receives two HF clean-greedy audits
and an exact rollback.

**Tech stack:** Python 3.11, PyTorch, Qwen3-VL/Transformers, vLLM batch decode,
FlashAttention-2 varlen packing, Accelerate world-size one, AdamW, pytest,
Ruff, OpenSpec.

**Authority:**
[OpenSpec](../../../openspec/changes/add-human13-k-trajectory-rp-crossover-screen/)
and [research unit](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-14-human13-k-trajectory-rp-crossover-screen/unit.md).
The [design note](../specs/2026-08-14-human13-k-trajectory-rp-crossover-screen-design.md)
is explanatory only.  This plan supplies execution order and TDD discipline;
it does not redefine the experiment.

**Global constraints:** This file is planning-only.  Do not begin Tasks 1--7
until the user explicitly authorizes implementation, and obtain separate
model/GPU execution authority before Task 8.  Preserve unrelated dirty changes;
use the `ms` Conda environment; keep vision, aligner, embeddings, and base
language weights frozen; use language-only DoRA and world-size one; do not
commit, push, or archive without an explicit user request.

**Dose constraint:** `3e-6` is the default, not an online-tuning promise.  The
only allowed calibration is the disjoint qualification ray
`{3e-7,1e-6,3e-6,1e-5,3e-5}` under the research unit's mechanics-only rule.
It publishes one global LR receipt before matrix materialization.  Gradient
norm, delta norm, predicted KL, and owner outcomes are covariates only; the
matrix never adapts dose.

**Review routing:** Use `agent-routing` for hard questions that can change the
research objective, interpretation, scope, or launch verdict.  Treat
Fable-5-xhigh/max and GPT-5.6-sol-max as peer read-only principal researchers;
the lead owns synthesis and the user owns the large-direction decision.  Do not
spend these routes on routine implementation details, and do not let reviewer
advice mutate files or grant execution authority.

---

### Task 1: Shared contracts and exact RP transform

**Files:**

- Create: `scripts/research/human13_k_trajectory_contracts.py`
- Create: `scripts/research/human13_rp_policy.py`
- Test: `tests/research/test_human13_rp_policy.py`

1. Add RED tests for positive/negative repeated logits, complete prompt and
   generated history, RP 1.0 identity, RP 1.10 sign-aware transform,
   temperature placement, full-support normalization, chosen-token gathering,
   exact stop/cap behavior, fixed parity tolerances, non-finite inputs, and
   content-addressed round trips.
2. Define immutable records for the policy contract, generated-token evidence,
   complete trajectory evidence, acquisition group, and artifact identity.
   Every record binds Source/manifest/request/model/tokenizer/processor/history.
3. Implement `processed_policy_logprobs(raw_logits, history_token_ids,
   contract)` in FP32 and `validate_policy_replay(sampled, replayed, tolerance)`.
   Keep the processor order literal and make the history convention explicit.
4. Run:

   ```bash
   conda run -n ms python -m pytest -c /dev/null tests/research/test_human13_rp_policy.py -q
   conda run -n ms ruff check scripts/research/human13_k_trajectory_contracts.py scripts/research/human13_rp_policy.py tests/research/test_human13_rp_policy.py
   ```

### Task 2: Fresh batch-four K16 acquisition

**Files:**

- Create: `scripts/research/collect_human13_rp_crossover.py`
- Test: `tests/research/test_collect_human13_rp_crossover.py`
- Reuse: `scripts/research/collect_human13_discovery.py`

1. Add RED tests for exactly 13 images, four physical batches per image,
   sixteen distinct requests, the frozen seed groups, `temperature=0.4`,
   `top_p=1.0`, no top-k truncation, explicit RP, exact 512-token/im_end cap-
   stop semantics, complete token/logprob coverage, native request ordering,
   and zero-action dry run.
2. Implement `plan_acquisition_group(...)` and an injected live executor around
   the existing vLLM session.  Store chosen-token processed log probabilities
   from the native sampler and raw token histories; never reuse historical K16
   traces as score-function evidence.
3. Add a replay adapter that obtains packed raw logits only at generated causal
   positions, applies Task 1's policy transform, and emits one parity receipt
   per acquisition group.  Do not retain full-vocabulary rows as Python float
   tuples.
4. Fail before artifact publication on request gaps, duplicated seeds,
   truncated score evidence, processor mismatch, or parity failure.
5. Run focused tests plus the existing discovery/backend contract suites.

### Task 3: Detached trajectory-credit ledger and loss

**Files:**

- Create: `scripts/research/human13_trajectory_credit.py`
- Test: `tests/research/test_human13_trajectory_credit.py`
- Reuse: `scripts/research/analyze_human13_k_union.py`

1. Add table-driven RED cases for first-hit marginal utility, fixed uniform
   trusted-owner weights, duplicate/invalid/repeat/unmatched precedence,
   malformed cost, legacy-M direct-token masking, natural-STOP versus cap
   shortfall, terminated-path padding, tied RLOO groups, and deterministic
   artifact hashes.
2. Implement `build_trajectory_credit_ledger(manifest, acquisition)` as a pure
   projection using the canonical one-to-one matcher and frozen G/H/M
   identities.  The ledger must retain every row outcome and detached return/
   advantage; it must not import a model or optimizer.
3. Implement `trajectory_score_function_loss(policy_logprobs, ledger)` as
   `-(1/(N*K)) * sum(log_pi * detached_row_advantage)` over scored non-M
   tokens, using unnormalized microstep numerators and one logical denominator.
   Test reward/cost gradient signs, STOP clamping, zero M-row direct gradient,
   zero gradient for tied groups, and loss/gradient invariance under arbitrary
   pack and accumulation partitions.
4. Run the focused suite and adjacent Human-13 matcher/analyzer tests.

### Task 4: Sparse Source-boundary greedy compiler

**Files:**

- Create: `scripts/research/human13_greedy_compiler.py`
- Test: `tests/research/test_human13_greedy_compiler.py`
- Reuse: `scripts/research/human13_on_policy_scoring.py`

1. Add RED tests for exact 309-alias binding, owner-then-alias normalization,
   `log-mean-exp <= max(valid)`, realized bad-child binding, greedy logits
   `RP_r(raw_logits)` without temperature division, margin crossing, absent
   sites, selector detachment, and rejection of fresh-support leakage.
2. Implement `build_compiler_ledger(...)` from each RP-specific Source decode
   and the frozen alias bank.  Retain only compact causal positions and token
   IDs needed by the packed forward.
3. Implement `greedy_compiler_loss(logits, ledger)` as the declared image-mean
   hinge and `combined_loss = trajectory_loss + compiler_loss` with fixed
   coefficient 1.0.  Apply the one global image denominator across packs and
   test partition-invariant loss and gradient.
4. Verify A and B use byte-identical acquisition/credit ledgers and that B adds
   only the compiler sites.  Run focused and adjacent packed-forward tests.

### Task 5: Exact AdamW proposal and owner-wise preservation

**Files:**

- Create: `scripts/research/human13_adamw_proposal_preservation.py`
- Test: `tests/research/test_human13_adamw_proposal_preservation.py`
- Reuse: `scripts/research/human13_training_transaction.py`
- Reuse: `scripts/research/human13_live_model.py`

1. Add RED tests with a real toy AdamW optimizer showing that captured
   parameter deltas and bias-corrected denominators equal an actual step for
   dense, zero, mixed-sign, and non-finite gradients.
2. Implement `capture_exact_adamw_proposal(...)` inside the existing full-state
   transaction.  Bind pre/post parameter hashes, gradients, optimizer config,
   delta tensors, metric denominators, state counters, and trust-radius value.
3. Add RED projection tests for the exact `D`-metric equation in the research
   unit: no-active-constraint identity, one and multiple active constraints,
   trust-radius activation, infeasibility, singular dual systems, non-finite
   values, and deterministic active-set order.
4. Implement `project_adamw_proposal(...)` with streamed witness screening and
   a bounded active-set dual solve.  It must fail closed on uncertified first-
   order feasibility, non-finite measurement, trust-radius failure, or wrong
   applied delta.  Finite realized witness degradation is receipted and still
   proceeds to both behavioral audits.
5. Add a toy end-to-end rollback case proving that an unprojected step, a
   projected manual apply, and complete transaction restore have the declared
   hashes and counter behavior.
6. Run focused tests and the existing transaction/live-model suites.  Record
   peak host/device memory in the later real vertical before admitting scale.

### Task 6: One-cell runtime, matrix launcher, configs, and analyzer

**Files:**

- Create: `scripts/research/human13_rp_crossover_runtime.py`
- Create: `scripts/research/train_human13_k_trajectory_rp_crossover.py`
- Create: `scripts/research/launch_human13_k_trajectory_rp_crossover.py`
- Create: `scripts/research/analyze_human13_k_trajectory_rp_crossover.py`
- Create: `tests/research/test_human13_rp_crossover_runtime.py`
- Create: `tests/research/test_train_human13_k_trajectory_rp_crossover.py`
- Create: `tests/research/test_launch_human13_k_trajectory_rp_crossover.py`
- Create: `tests/research/test_analyze_human13_k_trajectory_rp_crossover.py`
- Create: `configs/coordexp_swift/research/human13_k_trajectory_rp_crossover/01_rp100_trajectory.yaml`
- Create: `configs/coordexp_swift/research/human13_k_trajectory_rp_crossover/02_rp100_trajectory_compiler.yaml`
- Create: `configs/coordexp_swift/research/human13_k_trajectory_rp_crossover/03_rp100_trajectory_compiler_preservation.yaml`
- Create: `configs/coordexp_swift/research/human13_k_trajectory_rp_crossover/04_rp110_trajectory.yaml`
- Create: `configs/coordexp_swift/research/human13_k_trajectory_rp_crossover/05_rp110_trajectory_compiler.yaml`
- Create: `configs/coordexp_swift/research/human13_k_trajectory_rp_crossover/06_rp110_trajectory_compiler_preservation.yaml`

1. Add RED tests for the exact six config templates, three seed groups, eighteen
   proposals, shared acquisition only within one RP/seed group, independent
   Source/fresh optimizer per proposal, one update, both RP audits, always-
   rollback lifecycle, disjoint qualification seeds, unique roots, maximum
   eight concurrent GPUs, explicit execute authority, and dry-run action
   counters of zero.
   The leaf configs retain provisional `3e-6`, but execution must resolve it
   through a content-bound qualification receipt and reject per-RP/per-arm/
   per-seed drift.
2. Implement the one-cell runtime by composing existing model assembly,
   no-padding packed forward, private proposal, HF fp32/SDPA batch-one
   evaluation, and full transaction modules.  Do not create a second trainer or
   matcher.
3. Implement the matrix launcher as a static acquisition/proposal DAG.  It may
   parallelize independent jobs across at most eight explicitly assigned GPUs,
   but must not adapt settings or retry a scientific cell.
4. Implement the analyzer with RP-specific Source baselines, trusted gain,
   named baseline loss, historical G/H/M identity accounting,
   `protected_but_undefendable_M`, duplicate/malformed/invalid/row/token burden,
   paired A/B/C differences, same-seed contract-local/RP-robust rules, and
   complete-matrix admission.
5. Add deterministic receipt and forged-lineage tests, then run all new tests
   and adjacent Human-13 runtime/eval/config tests.

### Task 7: CPU integration and independent review gate

**Files:**

- Create at execution time:
  `.superpowers/sdd/2026-08-14-human13-k-trajectory-rp-crossover-screen/progress.md`
- Modify only when evidenced:
  `openspec/changes/add-human13-k-trajectory-rp-crossover-screen/tasks.md`

1. Run the complete research test slice explicitly because repository pytest
   discovery does not automatically include every `tests/research` file:

   ```bash
   conda run -n ms python -m pytest -c /dev/null \
     tests/research/test_human13_rp_policy.py \
     tests/research/test_collect_human13_rp_crossover.py \
     tests/research/test_human13_trajectory_credit.py \
     tests/research/test_human13_greedy_compiler.py \
     tests/research/test_human13_adamw_proposal_preservation.py \
     tests/research/test_human13_rp_crossover_runtime.py \
     tests/research/test_train_human13_k_trajectory_rp_crossover.py \
     tests/research/test_launch_human13_k_trajectory_rp_crossover.py \
     tests/research/test_analyze_human13_k_trajectory_rp_crossover.py -q
   ```

2. Run Ruff check/format, `python -m compileall` on the new scripts, strict
   OpenSpec validation, explicit dry-run zero-action verification, `git
   diff --check`, and scoped residue checks.
3. Ask separate standards and research-intent reviewers to inspect only
   conclusion-changing mismatches.  When a hard scientific decision remains,
   route the same bounded read-only question to Fable-5-xhigh/max and
   GPT-5.6-sol-max as peers, then record the lead synthesis.  Resolve all P0/P1
   findings without adding unrelated framework work; leave lower-priority
   observations recorded.
4. Mark only implemented and verified OpenSpec tasks complete.  Stop here until
   the user separately authorizes model/GPU execution.

### Task 8: Real production-shaped vertical

**Artifacts:** fresh root under
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-14-human13-k-trajectory-rp-crossover-screen/vertical/`

1. Confirm explicit execution authority, inspect live process/GPU state, and
   allocate the smallest non-conflicting set of GPUs.  Prevalidate Source,
   manifest SHA, alias bank, configs, seeds, and nonexistent output roots.
2. Acquire one full-panel K16 group under each RP contract with the disjoint
   qualification seeds `30001..30016`, in physical batches of four, and require
   complete token-level replay parity before proceeding.
3. Run the predeclared qualification dose ray from fresh Source state, with one
   independent C proposal, dual-RP audit, and rollback for each attempted
   `(training RP, dose)` point.  Keep
   `3e-6` when both RP contracts pass.  Otherwise follow the declared one-
   direction floor/ceiling search; stop on mixed/non-monotone evidence or no
   common admissible LR.  Selection may inspect greedy token changes,
   malformed/cap/parser status, witness activity, finite differences, and
   decision-margin displacement, but not owner identities or gain/loss.
4. Publish the selected global LR and all rejected dose receipts as immutable
   qualification evidence.  Bind that one LR to both RPs, every arm, and every
   matrix seed; reject any arbitrary or adaptive override.
5. Materialize all three nested objectives from each shared acquisition and
   admit the selected ray proposal per training RP only after packed backward,
   exact AdamW capture, active-set projection, both clean-greedy audits, and
   exact rollback have completed.
6. Verify the A/B dry-run payload differences, Source baselines, private
   proposal lifecycle, full owner identities, and zero residual model/process
   state.
7. Publish wall time, peak GPU/host memory, decode and packed tokens,
   forward/backward and witness counts, correction norm, artifact sizes, and a
   bounded admission verdict.  Exclude vertical owner outcomes from the matrix
   and forbid outcome-driven setting changes.  Stop on any unresolved P0/P1.

### Task 9: Fixed eighteen-proposal screen and closure

**Artifacts:** fresh matrix root under the owning experiment root.

1. Materialize the exact two-RP by three-seed by three-arm matrix and validate
   every immutable dependency against the admitted vertical.
2. Execute all eighteen independent one-update proposals, using up to eight
   GPUs for throughput.  Every cell performs both RP audits and exact rollback;
   no accepted checkpoint, adaptive retry, or state carryover is permitted.
3. Run the analyzer and report individual cells, paired A-to-B-to-C changes,
   RP crossover, trusted gains, named losses, legacy identities, burdens,
   compute, and failure receipts.  Union@K is diagnostic only.
4. Create `results.md`, update the unit and research index, update project
   memory, and run strict verification plus independent standards and
   scientific-interpretation reviews.
5. Stop after the bounded result.  Wider image coverage or multi-update depth
   requires a new user-owned research decision and a new unit/change.

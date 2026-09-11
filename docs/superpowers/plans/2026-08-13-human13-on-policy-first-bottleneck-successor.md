# Human-13 On-Policy First-Bottleneck Successor Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` for independent implementation
> tasks.  The owning OpenSpec change is the sole scope and completion authority.

**Goal:** Implement and run a bounded closed-loop Human-13 overfit controller
that compares complete selected-row supervision with one first-greedy-blocker
update, while retaining updates only after direct clean-greedy owner
preservation.

**Architecture:** Extend the existing experiment-local Human-13 pipeline with a
current-frontier sidecar, packed-prefilter/HF-decision candidate selector,
first-bottleneck loss adapter, and an in-memory model-plus-optimizer transaction.
Reuse current model assembly, packing, AdamW, checkpoint, HF evaluation, parser,
matcher, and analyzer modules.

**Tech stack:** Python 3.11, PyTorch, Qwen3-VL/Transformers, FlashAttention-2
varlen packing, Accelerate world-size one, vLLM batch K refresh, pytest, Ruff,
OpenSpec.

**Authority:**
[OpenSpec](../../../openspec/changes/archive/2026-08-28-add-human13-on-policy-first-bottleneck-successor/)
and [research unit](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-13-human13-on-policy-first-bottleneck-successor/unit.md).
This file is execution discipline only and deliberately avoids a second copy of
the scientific contract.

---

### Task 1: Current-frontier ledger

**Files:**

- Create: `scripts/research/build_human13_on_policy_frontier.py`
- Test: `tests/research/test_build_human13_on_policy_frontier.py`

1. Write failing tests for Source/current checkpoint binding, exact natural
   tokens/rows, canonical and constrained owner matching, protected-set age,
   covered/uncovered H, K-miss exclusion, current duplicate events, deterministic
   JSON/digest, and wrong span/hash/parser rejection.
2. Run the focused test through `conda run -n ms ... -c /dev/null` and retain RED.
3. Implement immutable dataclasses and pure manifest/current-output projection.
   Reuse existing Human-13 analyzer/matcher helpers rather than creating a new
   owner metric.
4. Re-run focused and adjacent manifest/analyzer tests to GREEN.

### Task 2: Candidate scoring and continuation selection

**Files:**

- Create: `scripts/research/human13_frontier_selection.py`
- Test: `tests/research/test_human13_frontier_selection.py`
- Modify: `scripts/research/human13_live_eval.py`
- Test: `tests/research/test_human13_live_eval.py`

1. Add RED tests for packed prefilter only, HF-owned barrier and first
   bottleneck, global argmax/ties, aligned cross-surface drift, owner alias
   reduction, two-to-four-owner shortlist, and missing HF evidence failure.
2. Add RED tests for forced row plus natural continuation, cap harm, constrained
   protected matching, unique-owner delta, and lexicographic selection.
3. Implement pure score/selection math, then a narrow HF adapter that accepts
   caller-owned prefixes/row token IDs and returns full-vocabulary causal score
   evidence plus unconstrained continuation.  Scientific labels remain in the
   experiment module, not the backend.
4. Run focused plus inference backend/trace tests.  Keep existing public backend
   semantics unchanged.

### Task 3: Full-row and first-bottleneck losses

**Files:**

- Modify: `src/losses/human13_k_union.py`
- Test: `tests/research/test_run_human13_k_union_overfit.py`
- Test: `tests/research/test_human13_row_contrast_live.py`

1. Add tensor RED cases for one-site first blocker, already-greedy no-op,
   competitor gradient sign, ties, full-row body CE, terminal masking,
   alternate rectangle-valid coordinate, and earliest differing duplicate
   token with shared opener/description protection.
2. Implement side-effect-free fp32 helpers and typed receipts.  Selector
   identities are detached; parsing/matching never occurs in the loss module.
3. Run focused tests plus adjacent `tests/losses`.

### Task 4: Transactional optimizer state

**Files:**

- Create: `scripts/research/human13_training_transaction.py`
- Test: `tests/research/test_human13_training_transaction.py`

1. Add RED tests using a real small AdamW model for parameter, first/second
   moment, step, scheduler, update counter, CPU RNG and CUDA-RNG-when-available
   capture; test accepted commit, rejected restore, repeated restore, and state
   digest equality.
2. Implement snapshot/restore only for the bound language-DoRA trainable surface
   and optimizer/scheduler state.  Frozen base/vision/aligner tensors are not
   copied.
3. Add a deliberate forced-reject fixture proving the next toy update matches a
   control that never applied the rejected update.

### Task 5: Live materialization and loop CLI

**Files:**

- Create: `scripts/research/human13_on_policy_live.py`
- Create: `scripts/research/train_human13_on_policy_successor.py`
- Modify: `scripts/research/human13_live_segments.py`
- Modify: `scripts/research/human13_live_payload.py`
- Test: `tests/research/test_human13_on_policy_live.py`
- Test: `tests/research/test_train_human13_on_policy_successor.py`

1. Add RED contract tests for exact current-prefix segments, arm-specific loss
   sites, shared rectangle/duplicate sites, one update per ledger, at-most-eight
   attempts, transaction ordering, accepted/rejected artifact lifecycle, fresh
   roots, and execute authority.
2. Implement the minimal loop around existing live model/payload/runner/eval
   seams.  Do not duplicate model assembly or build a general trainer.
3. One post-update decode decides the proposal and becomes the next ledger only
   on acceptance.  Rejected proposals restore before any new candidate is used.
4. Add optional K refresh after two accepted steps or exhaustion by reusing the
   current `batch_size=4`, `rp=1.10` discovery adapter; default initial path uses
   the sealed K bank.
5. Run focused and the full adjacent Human-13 research suite.

### Task 6: Configs, dry-run, and CPU gate

**Files:**

- Create: `configs/coordexp_infras/research/human13_on_policy_successor/01_o_full_safe.yaml`
- Create: `configs/coordexp_infras/research/human13_on_policy_successor/02_o_first_safe.yaml`
- Modify: `research/investigations/qwen3-vl-dense-enumeration/experiments/index.md`
- Update: `.superpowers/sdd/2026-08-13-human13-on-policy-first-bottleneck-successor/progress.md`

1. Add config/CLI RED tests for exactly two arms, fresh Source/AdamW, world-size
   one, attempt cap eight, unique outputs, exact HF surface, dry-run zero actions,
   and no hidden ungated arm.
2. Run focused/adjacent pytest, Ruff check/format, py_compile, strict OpenSpec
   validation, and explicit git diff/residue checks.
3. Ask one code-review subagent for spec/implementation mismatch only.  Resolve
   P0/P1; record lower findings without expanding scope.

### Task 7: Real vertical and rollback drill

**Artifacts:** fresh root under
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-on-policy-first-bottleneck-successor/vertical/`

1. Inspect live GPU/process state and reserve one non-conflicting GPU.
2. Dry-run exact Source/manifest/K/config/output bindings and assert zero actions.
3. Execute one O-First-Safe full-panel iteration: Source reproduction, packed
   prefilter, HF shortlist of two, two forced continuations, one packed update,
   one post-update clean decode, and analyzer.
4. Repeat the transaction path with explicit forced rejection.  Verify model,
   AdamW, scheduler, counter and RNG digests restore, then verify clean-greedy
   owner-set reproduction.
5. Publish the mechanics-only vertical receipt.  Stop on any unit gate instead
   of changing scientific semantics to make the run pass.

### Task 8: Bounded two-arm pilot

**Artifacts:** fresh `o-full-safe` and `o-first-safe` roots under the owning
experiment root.

1. Start independent Source model and fresh optimizer state for each arm.  Use
   one GPU per arm when available; inference/candidate calls may use additional
   GPUs only when their checkpoint/model identity is exact.
2. Attempt at most eight iterations per arm with one mutation per ledger and
   immutable accepted/rejected receipts.  Apply declared early stops exactly.
3. If triggered, execute versioned K16 refresh in four-request batches at
   `rp=1.10`; otherwise do not spend refresh compute.
4. Evaluate the final accepted checkpoint of each arm from a fresh process under
   exact HF fp32/SDPA batch-one clean greedy `rp=1.0` and run the analyzer.

### Task 9: Evidence closure

**Files:**

- Create: `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-13-human13-on-policy-first-bottleneck-successor/results.md`
- Modify: `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-13-human13-on-policy-first-bottleneck-successor/unit.md`
- Modify: `openspec/changes/archive/2026-08-28-add-human13-on-policy-first-bottleneck-successor/tasks.md`

1. Hash all decision-owning manifests, checkpoints, iteration receipts, raw
   outputs, and analyses.  Report attempted/accepted/rejected updates, protected
   identities, G/H/M, burden, candidate calls, packed tokens, wall time, and
   peak memory.
2. State supported, falsified, unresolved, and next-decision conclusions under
   the same-panel boundary.  Do not run validation or promote a checkpoint.
3. Re-run targeted verification, strict OpenSpec validation, standards review,
   intent-contract review, and continuity check.  Mark only evidenced tasks
   complete and stop after this pilot.

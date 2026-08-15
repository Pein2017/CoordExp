# Human-13 All-HF Shared-Surface Trajectory Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute one complete private Human-13 K16 trajectory-credit + greedy-compiler + preservation update on a shared HF surface and decide it with dual-RP clean greedy.

**Architecture:** Add one experiment-local shared-surface contract module and one live HF owner around the existing Human-13 Source assembly.  Sampling is no-cache stepwise batch four; replay is no-cache vectorized batch four on the same BF16/FA2 model object.  Existing credit, compiler, AdamW preservation, audit, and rollback modules remain scientific owners.

**Tech Stack:** Python 3.11, PyTorch, Transformers/Qwen3-VL, PEFT DoRA, FlashAttention-2, Accelerate world-size one, pytest, OpenSpec.

## Global Constraints

- Scientific authority: `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/unit.md`.
- Implementation authority: `openspec/changes/add-human13-all-hf-shared-surface-trajectory-credit-vertical/`.
- Initial case: image `1584`; seeds `35001..35016`; K=16 in four groups of four.
- Sampling policy: RP `1.0`, temperature `0.4`, `top_p=1.0`, no top-k, `max_new_tokens=512`, Qwen `<|im_end|>` only.
- Shared gradient surface: one live model object, BF16, FlashAttention-2, `eval()`, `use_cache=False`, unchanged parameters through parity admission.
- Parity: exact history/chosen token; finite processed log probabilities; max error `<=0.02`; mean error `<=0.002`.
- Objective: complete trajectory credit + compiler coefficient `1.0`, `kappa=1`, margin `1e-4` + owner-wise preservation; no missing-component fallback.
- AdamW: one update, LR `3e-6`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay zero.
- Audit: original-prompt HF fp32/SDPA batch one at RP `1.0` and `1.10` on a distinct GPU.
- No vLLM score-function data, KV/prefix reuse, CE fallback, LR ray, adaptive retry, accepted checkpoint, K-miss supervision, validation, or production change.
- Every implementation task follows TDD, stages explicit paths only, and ends with one task-local commit after its independent gate.

---

### Task 1: Shared-surface value contracts and admission

**Files:**
- Create: `scripts/research/human13_hf_shared_surface.py`
- Create: `tests/research/test_human13_hf_shared_surface.py`
- Modify: `openspec/changes/add-human13-all-hf-shared-surface-trajectory-credit-vertical/tasks.md`

**Interfaces:**
- Consumes: strict JSON values and SHA helpers from `src/artifacts/json_values.py`; no model imports.
- Produces:
  - `HFSharedSurfacePolicy`
  - `HFSharedSurfaceIdentity`
  - `HFSharedSurfacePlan`
  - `SampledHFToken`, `SampledHFRequest`, `SampledHFGroup`
  - `GradientReplayGroup`, `HFSharedSurfaceParityReceipt`
  - `plan_image1584_k16() -> HFSharedSurfacePlan`
  - `admit_sampled_group(...) -> SampledHFGroup`
  - `admit_gradient_replay(...) -> GradientReplayGroup`

- [ ] **Step 1: Write the failure-mode matrix and failing constructor tests**

Add a test table whose rows bind invariant, minimal mutation, and expected
exception.  Include model-object identity, parameter digest, adapter/delta,
dtype, attention backend, model mode, tokenizer/prompt/image, seed/order,
history/token, processor order, no-cache, finite values, and parity thresholds.

```python
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"dtype": "float32"}, "shared surface identity"),
        ({"use_cache": True}, "cache is forbidden"),
        ({"seeds": tuple(range(35002, 35018))}, "seed coverage"),
    ],
)
def test_shared_surface_admission_rejects_mutation(mutation, message):
    with pytest.raises(SharedSurfaceContractError, match=message):
        admit_sampled_group(**valid_group_kwargs() | mutation)
```

- [ ] **Step 2: Run the focused tests and record RED**

Run:

```bash
conda run -n ms python -m pytest -q \
  tests/research/test_human13_hf_shared_surface.py
```

Expected: collection fails because `human13_hf_shared_surface` does not exist.

- [ ] **Step 3: Implement immutable types and one admission choke point**

Use frozen dataclasses.  Make direct scientific constructors private and route
creation/reload/publication through one `_admit(...)` function.

```python
@dataclass(frozen=True)
class HFSharedSurfaceIdentity:
    checkpoint_payload_sha256: str
    model_object_id: int
    parameter_state_sha256: str
    adapter_sha256: str
    embedding_delta_sha256: str
    dtype: Literal["bfloat16"]
    attention_backend: Literal["flash_attention_2"]
    model_mode: Literal["eval"]
    tokenizer_sha256: str
    prompt_sha256: str
    image_sha256: str
    use_cache: Literal[False]

def plan_image1584_k16() -> HFSharedSurfacePlan:
    ...
```

The policy object must encode RP-before-temperature, prompt-inclusive history,
temperature 0.4, top-p 1.0, no top-k, cap 512, and im_end-only stop.

- [ ] **Step 4: Implement parity math and canonical serialization**

`admit_gradient_replay` must compare every `(request_id, token_index,
history_sha256, chosen_token_id)` and compute max/mean absolute processed-logp
error.  It returns an admitted replay only at `max<=0.02` and `mean<=0.002`.
Add `to_dict`, `from_dict`, and `content_sha256` through the same admission
path; copied/forged values must fail.

- [ ] **Step 5: Run GREEN and static gates**

Run:

```bash
conda run -n ms python -m pytest -q \
  tests/research/test_human13_hf_shared_surface.py
conda run -n ms ruff check \
  scripts/research/human13_hf_shared_surface.py \
  tests/research/test_human13_hf_shared_surface.py
conda run -n ms python -m compileall -q \
  scripts/research/human13_hf_shared_surface.py
openspec validate add-human13-all-hf-shared-surface-trajectory-credit-vertical --strict
```

Expected: all pass; no torch/model/GPU import or action in module import tests.

- [ ] **Step 6: Request one independent review, apply one correction bundle, and commit**

Review only the frozen failure-mode matrix and Task-1 diff.  Fix all P0/P1 in
one bundle, rerun localized checks once, then commit explicit paths:

```bash
git add -- \
  scripts/research/human13_hf_shared_surface.py \
  tests/research/test_human13_hf_shared_surface.py \
  openspec/changes/add-human13-all-hf-shared-surface-trajectory-credit-vertical/tasks.md
git commit -m "research: add all-HF shared-surface contracts"
```

### Task 2: Live no-cache HF sampler and gradient replay

**Files:**
- Create: `scripts/research/human13_hf_shared_surface_live.py`
- Create: `tests/research/test_human13_hf_shared_surface_live.py`
- Modify: `scripts/research/human13_live_model.py`
- Modify: `tests/research/test_human13_live_model.py`

**Interfaces:**
- Consumes: Task-1 plan/identity/admission types; `Human13LiveAssembly` and
  `build_human13_processor_skeletons` from `human13_live_model.py`.
- Produces:
  - `HFSharedSurfaceSession.sample_group(seeds: tuple[int, ...])`
  - `HFSharedSurfaceSession.replay_group(group: SampledHFGroup)`
  - `open_hf_shared_surface(plan, assembly, skeleton) -> HFSharedSurfaceSession`
  - `SharedSurfaceResourceReceipt`

- [ ] **Step 1: Write injected-model RED tests**

Use a tiny causal fake that records `input_ids`, masks, `use_cache`, grad mode,
model mode, dtype/backend labels, and model-object identity.  Test four active
requests, one early stop, exact history growth, and vectorized causal-position
gathering.

```python
def test_sampling_and_replay_use_one_model_object_and_no_cache():
    session = open_fake_shared_surface()
    sampled = session.sample_group((35001, 35002, 35003, 35004))
    replay = session.replay_group(sampled)
    assert sampled.surface_sha256 == replay.surface_sha256
    assert {call.use_cache for call in session.calls} == {False}
    assert session.model.training is False
```

- [ ] **Step 2: Run RED**

Run the new test file; expect missing-module failure.

- [ ] **Step 3: Add successor model-plan admission**

Extend `human13_live_model.py` only enough to admit the new unit ID and fixed
BF16/FA2 Source plan.  Do not alter existing K-union, on-policy, or RP-crossover
plans.  Assert language-only DoRA trainability and frozen vision/aligner/delta.

- [ ] **Step 4: Implement stepwise no-cache sampling**

For each token step, rebuild the exact four active prompt/image/history inputs,
call the same model with `use_cache=False`, apply the sign-aware RP transform,
divide by temperature, sample with the request-local generator, and record the
pre/post RNG digests.  Remove stopped requests without reordering survivors.

```python
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(**inputs, use_cache=False).logits[:, -1, :]
    processed = apply_repetition_penalty(logits, exact_histories, rp=1.0)
    token = torch.multinomial((processed / 0.4).softmax(-1), 1, generator=rng)
```

- [ ] **Step 5: Implement vectorized grad replay**

Pad the four completed exact histories once, run the same model object in
`eval()` with gradients enabled and `use_cache=False`, gather each sampled
causal position, apply the same processor, and call Task-1 parity admission.
Keep gathered tensors attached to the graph only inside the live replay owner;
serialized receipts contain detached values/hashes.

- [ ] **Step 6: Verify lifecycle and real-shape CPU contracts**

Test model substitution, parameter mutation, train-mode drift, dtype/backend
drift, prompt/image/tokenizer drift, cached forward, wrong causal position,
non-finite logits, early-stop padding, exception cleanup, and double-close.
Run Task-1/Task-2 plus adjacent live-model tests, Ruff, compile, Serena
diagnostics, and strict OpenSpec.

- [ ] **Step 7: Independent review and commit**

Review one live-path diff against the Task-1 failure matrix.  Apply one P0/P1
bundle, localized recheck, and commit the four explicit files.

### Task 3: Complete objective and transaction adapter

**Files:**
- Create: `scripts/research/human13_all_hf_vertical.py`
- Create: `tests/research/test_human13_all_hf_vertical.py`
- Modify only if an exact missing seam is proven:
  `scripts/research/human13_rp_crossover_runtime.py`
- Modify matching tests only with that seam:
  `tests/research/test_human13_rp_crossover_runtime.py`

**Interfaces:**
- Consumes:
  - `build_trajectory_credit_ledger(...) -> TrajectoryCreditLedger`
  - `trajectory_score_function_numerator(policy_logprobs, ledger, ...)`
  - `build_compiler_ledger(...) -> CompilerLedger`
  - `greedy_compiler_numerator(compact_logits, ledger)`
  - existing AdamW proposal/preservation and `TrainingStateTransaction`
- Produces:
  - `AllHFVerticalServices.prepare(...) -> PreparedAllHFVertical`
  - `PreparedAllHFVertical.backward_and_propose() -> PrivateProposalReceipt`
  - `PreparedAllHFVertical.rollback() -> RollbackReceipt`

- [ ] **Step 1: Write composition RED tests**

Use admitted fake replay tensors with gradients.  Assert numerator addition,
one global denominator, compiler coefficient 1.0, fixed AdamW, actual-delta
projection, one update, and exact rollback.  Add explicit tests that each
missing component fails rather than falling back.

```python
def test_complete_arm_requires_all_three_components():
    with pytest.raises(AllHFVerticalError, match="compiler is required"):
        prepare_vertical(valid_inputs(), compiler=None)
```

- [ ] **Step 2: Run RED**

Run the new test file; expect missing module.

- [ ] **Step 3: Implement the narrow adapter**

The adapter must pass the admitted replay tensor mapping directly to
`trajectory_score_function_numerator`, gather compiler logits from the same
live session, and reuse existing proposal/preservation functions.  Do not copy
credit, compiler, AdamW, projection, or transaction math.

- [ ] **Step 4: Bind complete transaction state**

Begin `TrainingStateTransaction` immediately before backward.  Include the
exact trainable parameter mapping, fresh optimizer/scheduler, runtime counters,
gradients, and RNG.  On every exception call reject/restore before releasing
the shared session.  Assert one applied optimizer step and no promoted state.

- [ ] **Step 5: Add component and resource receipts**

Record trajectory/compiler numerators, denominator, total loss, grad norm,
actual AdamW delta norm, projection correction norm, active constraints,
applied parameter hash, forward/backward counts, and peak resources.  Receipt
hashes must bind Task-1/2 surface and K16 evidence.

- [ ] **Step 6: Verify and commit**

Run new tests plus trajectory-credit, compiler, preservation, runtime, and
transaction suites.  Run Ruff/compile/Serena/OpenSpec.  Request one bounded
review, apply one correction bundle, and commit only owned files.

### Task 4: Dual-GPU live entry, audit, analyzer, and continuation gate

**Files:**
- Create: `configs/coordexp_swift/research/human13_all_hf_shared_surface_vertical/01_image1584.yaml`
- Create: `scripts/research/run_human13_all_hf_shared_surface_vertical.py`
- Create: `tests/research/test_run_human13_all_hf_shared_surface_vertical.py`
- Create: `scripts/research/analyze_human13_all_hf_shared_surface_vertical.py`
- Create: `tests/research/test_analyze_human13_all_hf_shared_surface_vertical.py`
- Reuse: `scripts/research/human13_live_eval.py`

**Interfaces:**
- Consumes: Tasks 1–3 services; `evaluate_hf_checkpoint(...,
  repetition_penalty=...)`; canonical Human-13 matcher/analyzer helpers.
- Produces:
  - zero-action `VerticalDryRunReceipt`
  - immutable phase receipts and `AllHFVerticalTerminal`
  - `OneImageOwnerOutcome`
  - `one_image_continuation_passed(outcome) -> bool`

- [ ] **Step 1: Write CLI and lifecycle RED tests**

Test default dry-run, explicit `--execute --user-model-gpu-authority`, two
distinct GPU IDs, immutable output root, exact config/manifest/seed binding,
zero vLLM imports, no retries, and fail-before-model behavior.

- [ ] **Step 2: Implement the leaf config and dry-run**

The YAML must freeze every Global Constraint and distinct train/audit GPU roles.
Dry-run may inspect files/config/GPU availability but must report zero model,
forward, backward, optimizer, checkpoint, and output-root actions.

- [ ] **Step 3: Compose the live phases**

Implement one monotonic state machine:

```text
planned -> source_audited -> sampled -> replay_admitted ->
backward -> projected -> private_checkpoint -> dual_audited ->
rolled_back -> source_reproduced -> complete
```

Every failure writes one typed terminal after rollback/cleanup.  A parity
failure occurs before backward.  The proposal checkpoint stays private.

- [ ] **Step 4: Implement dual-RP owner analysis**

Run Source and proposal clean greedy at RP 1.0 and 1.10 via
`evaluate_hf_checkpoint`.  Reuse the canonical duplicate exclusion/parser/
matcher.  Emit exact gained/lost owner IDs plus burden counts; reject aggregate
numbers that cannot be recomputed from rows.

- [ ] **Step 5: Implement the continuation gate and inert full-panel entry**

Require H gain >=1 and positive net at RP1.0, zero G loss on both surfaces, no
duplicate increase, no malformed output, and no cap stop.  The full-panel entry
must require the exact passing terminal content hash and otherwise stop before
model/GPU/output actions.

- [ ] **Step 6: Run production-shaped CPU smoke and independent prelaunch review**

Run all new tests, adjacent Human-13 live/eval/runtime/analyzer suites, strict
OpenSpec, Ruff/compile/Serena, dry-run action counters, `git diff --check`, and
one public-entry review.  Apply at most one correction bundle, then commit the
five new files plus any explicitly justified narrow adapter edit.

### Task 5: One-image live vertical and bounded closure

**Files:**
- Create on execution: immutable output root under
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/`
- Modify after evidence:
  `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/unit.md`
- Create after evidence:
  `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/results.md`
- Create after evidence:
  `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/review.md`
- Modify after evidence: `research/investigations/qwen3-vl-dense-enumeration/experiments/index.md`
- Modify after evidence: `memories/current.md`

**Interfaces:**
- Consumes: the reviewed Task-4 public CLI and exact leaf config.
- Produces: one immutable one-image terminal, independent audit, bounded result,
  and optionally one content-authorized full-panel continuation.

- [ ] **Step 1: Reverify live authority and resources**

Inspect exact Git status/HEAD, two free suitable GPUs, processes, intended root
absence, Source/config/manifest hashes, and dry-run action counters.  Stop on
dirty overlap, root collision, or insufficient GPUs rather than adapting the
contract silently.

- [ ] **Step 2: Run no-update acquisition/replay admission**

Use the Task-4 command with train and audit GPU IDs.  Require exactly image
1584, seeds 35001..35016, four groups, K16, and one admitted shared-surface
parity receipt.  On failure, confirm zero backward/update/private checkpoint.

- [ ] **Step 3: Continue to exactly one complete update**

Only after parity admission, run the combined objective, projected private
apply, dual-RP clean-greedy audits, rollback, and Source reproduction.  Do not
change LR, seeds, compiler coefficient, preservation limits, or tolerances.

- [ ] **Step 4: Independently audit evidence before interpretation**

Recompute all canonical content hashes and owner arithmetic; verify one model
surface, K16 request/token coverage, one update, all component evidence,
private-byte cleanup, no vLLM evidence, both audits, exact rollback, process
cleanup, and resource release.  Return one PASS/HOLD bounded to the declared
one-image claim.

- [ ] **Step 5: Apply the continuation rule**

If and only if the one-image terminal passes every continuation condition,
request/confirm the material full-panel execution decision and run the guarded
13-image entry.  Otherwise leave it unexecuted and close the one-image unit.

- [ ] **Step 6: Write results, continuity, verify, and commit**

Write bounded `results.md` and `review.md`, update the unit/index/memory, check
OpenSpec tasks truthfully, run strict validation and residue checks, stage
explicit paths, inspect the staged diff, and commit.  Do not archive while an
authorized conditional execution task remains intentionally unexecuted.

## Plan self-review

- Every OpenSpec requirement maps to a task: shared surface (1–2), K16/parity
  (1–2), complete objective/rollback (3), dual-RP decision (4), conditional
  width expansion and closure (5).
- Public type/function names are defined once and reused consistently.
- No placeholders, generic “add tests,” hidden fallback, or adaptive tuning
  remain.
- Each task has its own RED/GREEN cycle, bounded independent review, localized
  correction, and explicit-path commit.
- The plan reaches one complete algorithm update before adding ablation or
  scale breadth.

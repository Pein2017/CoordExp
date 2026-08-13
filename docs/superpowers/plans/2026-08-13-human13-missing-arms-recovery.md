# Human-13 Missing-Arm Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the previously unavailable A4, A6, and A8-prime Human-13 arms without changing their scientific estimands, then execute and evaluate them from fresh Source/AdamW states.

**Architecture:** A6 and A8 are narrow provenance/runtime repairs in the existing experiment-local adapters. A4 remains one per-image prefix-free union objective and uses exact fixed-parameter two-pass physical streaming: a no-grad score pass produces one global fp32 weight vector per image, then the same candidate segments are replayed with detached weights and all gradients are accumulated before one optimizer step. Existing manifest, Qwen no-padding/FA2 forward, DoRA runtime, checkpoint, HF evaluation, matcher, and analyzer surfaces are reused.

**Tech Stack:** Python 3 in Conda `ms`, PyTorch, Qwen3-VL, FlashAttention-2 varlen packing, Hugging Face fp32/SDPA evaluation, DoRA, AdamW, pytest, Ruff, OpenSpec.

## Global Constraints

- OpenSpec change `add-human13-k-union-greedy-overfit-probe` is the sole authority for scope and scientific meaning.
- Use the sealed full-panel manifest SHA `a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb`; do not rebuild or alter Source/K targets.
- Every updated arm starts from byte-identical Source with fresh AdamW state and runs world-size one.
- A4 score and replay passes use unchanged parameters; chunk-local union losses are forbidden.
- Every individual physical segment remains at or below `global_max_length=12000`; no padding, prefix-KV, or image-encoder reuse claim is added.
- A8-prime runs only if a complete immutable census reports finite aligned drift and `required_margin <= 0.5`.
- Original-prompt evaluation is HF fp32/SDPA, batch size one, greedy, and repetition penalty `1.0`.
- Write every new census, run, checkpoint, evaluation, and analysis under a fresh nonexistent immutable root; preserve all prior artifacts.
- Stop after the bounded A4/A6/A8-prime successor; do not run 100 updates, supervise K-miss, promote checkpoints, sync stable specs, or archive the change.
- Keep implementation narrow: no new generic trainer, cache framework, candidate tree, or evidence journal.

---

### Task 1: Seal the revised authority and plan

**Files:**
- Modify: `openspec/changes/add-human13-k-union-greedy-overfit-probe/proposal.md`
- Modify: `openspec/changes/add-human13-k-union-greedy-overfit-probe/design.md`
- Modify: `openspec/changes/add-human13-k-union-greedy-overfit-probe/specs/coordexp-swift-human13-k-union-greedy-probe/spec.md`
- Modify: `openspec/changes/add-human13-k-union-greedy-overfit-probe/tasks.md`
- Create: `docs/superpowers/plans/2026-08-13-human13-missing-arms-recovery.md`

**Interfaces:**
- Consumes: the completed first Human-13 matrix and its frozen execution disposition.
- Produces: exact A4 two-pass, A6 clean-prefix, A8 complete-census, execution, and stop contracts.

- [ ] **Step 1: Validate the revised OpenSpec**

Run:

```bash
openspec validate add-human13-k-union-greedy-overfit-probe --strict
```

Expected: exit `0` and no schema or scenario errors.

- [ ] **Step 2: Scan the plan and authority for forbidden placeholders or stale A4 claims**

Run:

```bash
rg -n 'TB[D]|TO[D]O|implement la[t]er|A4 kee[p]s each image.s entire candidate set in one graph|multi-pass candidate en[g]ine' docs/superpowers/plans/2026-08-13-human13-missing-arms-recovery.md openspec/changes/add-human13-k-union-greedy-overfit-probe
```

Expected: no plan placeholder and no stale one-graph A4 requirement.

- [ ] **Step 3: Commit the authority revision**

```bash
git add openspec/changes/add-human13-k-union-greedy-overfit-probe docs/superpowers/plans/2026-08-13-human13-missing-arms-recovery.md
git commit -m "spec: authorize Human-13 missing-arm recovery"
```

### Task 2: Repair A6 donor-prefix materialization

**Files:**
- Modify: `scripts/research/human13_live_segments.py`
- Modify: `tests/research/test_human13_live_segments.py`

**Interfaces:**
- Consumes: manifest trajectory fields `raw_token_ids`, `rows`, `duplicate_row_ids`, and the selected row `token_start`.
- Produces: `_clean_donor_prefix(trajectory: Any, target_row: Any) -> tuple[int, ...]`, used by every `a6_donor_h1` segment.

- [ ] **Step 1: Write a failing duplicate-before-target regression**

Add a typed fixture whose donor raw prefix contains an earlier duplicate span and assert:

```python
segment = next(item for item in result.segments if item.role == "a6_donor_h1")
assert segment.encoded_example.input_ids == prompt + expected_clean_prefix + target_row
assert duplicate_tokens not in segment.encoded_example.input_ids[len(prompt):target_start]
```

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
conda run -n ms python -m pytest -c /dev/null tests/research/test_human13_live_segments.py -q
```

Expected: the new assertion fails because the materializer currently uses the raw donor prefix.

- [ ] **Step 3: Implement exact clean-prefix derivation**

Add:

```python
def _clean_donor_prefix(trajectory: Any, target_row: Any) -> tuple[int, ...]:
    removed = {
        index
        for row in trajectory.rows
        if row.row_id in set(trajectory.duplicate_row_ids)
        and row.token_end <= target_row.token_start
        for index in range(row.token_start, row.token_end)
    }
    return tuple(
        token
        for index, token in enumerate(trajectory.raw_token_ids[: target_row.token_start])
        if index not in removed
    )
```

Require the selected donor row to exist and use this helper without a fallback.

- [ ] **Step 4: Run focused and runner-binding tests**

Run:

```bash
conda run -n ms python -m pytest -c /dev/null tests/research/test_human13_live_segments.py tests/research/test_run_human13_k_union_overfit.py tests/research/test_human13_live_payload.py -q
```

Expected: all pass.

- [ ] **Step 5: Audit every real donor against the sealed binding**

Run the existing plan/materializer derivation over the canonical manifest and assert all donor tuples equal `Human13A6DonorBinding.donor_prefix_token_ids`; expected `73/73` matches, including image `16228` owner `gt:16228:19`.

- [ ] **Step 6: Commit the A6 repair**

```bash
git add scripts/research/human13_live_segments.py tests/research/test_human13_live_segments.py
git commit -m "fix(research): align Human-13 A6 clean donor prefixes"
```

### Task 3: Repair A8 census skeleton metadata and publish a fresh census

**Files:**
- Modify: `scripts/research/human13_live_census.py`
- Modify: `tests/research/test_human13_live_census.py`
- Runtime output: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-missing-arms-successor/census/v1/`

**Interfaces:**
- Consumes: an `EncodedExample` with dynamic `human13_image_id`, `prompt_token_count`, and `owner_row_tokens` metadata.
- Produces: `_clone_skeleton(...)` that preserves those values, plus a complete canonical census and exclusive execution receipt.

- [ ] **Step 1: Write a failing real-dataclass clone regression**

Construct the same processor skeleton type used by `build_human13_processor_skeletons`, attach dynamic metadata, clone it, and assert:

```python
assert clone.human13_image_id == original.human13_image_id
assert clone.prompt_token_count == original.prompt_token_count
assert clone.owner_row_tokens == original.owner_row_tokens
```

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
conda run -n ms python -m pytest -c /dev/null tests/research/test_human13_live_census.py -q
```

Expected: the cloned dataclass lacks `prompt_token_count` or `owner_row_tokens`.

- [ ] **Step 3: Copy exact experiment-local metadata after `dataclasses.replace`**

Use an allowlist and `object.__setattr__`:

```python
for name in ("human13_image_id", "prompt_token_count", "owner_row_tokens"):
    if hasattr(skeleton, name):
        object.__setattr__(encoded, name, getattr(skeleton, name))
object.__setattr__(encoded, "human13_image_id", image_id)
```

Do not copy model/runtime state or silently manufacture absent metadata.

- [ ] **Step 4: Run census and scorer regression suites**

Run:

```bash
conda run -n ms python -m pytest -c /dev/null tests/research/test_human13_live_census.py tests/research/test_human13_hf_census.py tests/research/test_run_human13_live_census.py -q
```

Expected: all pass.

- [ ] **Step 5: Run the fresh immutable live census**

Use `scripts/research/run_human13_live_census.py` with the sealed manifest, existing Source/K discovery roots, A1 config, explicit model/GPU authority, and new output/receipt paths under the runtime output root above. Expected: exactly one canonical `human13_k_union_no_update_census.v1` artifact and one receipt; optimizer steps remain `0`.

- [ ] **Step 6: Decide A8 mechanically from the complete artifact**

Verify all target/site identities, finite packed/HF logits, and compute `required_margin=max_target_margin_drift+1e-4`. Admit A8-prime only when `0 < required_margin <= 0.5`; otherwise record `applicable=false` with the exact census values and do not train A8.

- [ ] **Step 7: Commit the A8 repair**

```bash
git add scripts/research/human13_live_census.py tests/research/test_human13_live_census.py
git commit -m "fix(research): preserve Human-13 census skeleton metadata"
```

### Task 4: Implement exact A4 two-pass physical streaming

**Files:**
- Modify: `src/losses/human13_k_union.py`
- Modify: `tests/losses/test_human13_k_union.py`
- Modify: `scripts/research/run_human13_k_union_overfit.py`
- Modify: `tests/research/test_run_human13_k_union_overfit.py`
- Modify: `scripts/research/human13_live_payload.py`
- Modify: `tests/research/test_human13_live_payload.py`
- Modify: `scripts/research/train_human13_live_arm.py`
- Modify: `tests/research/test_train_human13_live_arm.py`

**Interfaces:**
- Consumes: moved A4 microsteps and exact `union_mass` sites.
- Produces: `A4GlobalWeights`, `a4_global_candidate_weights(row_scores_by_image)`, and `Human13A4TwoPassLossRunner`, whose preparation scores all candidates at fixed theta and whose gradient pass applies detached global weights.

- [ ] **Step 1: Write pure gradient-equivalence RED tests**

Compare the reference loss gradient to the streamed surrogate:

```python
reference = -torch.logsumexp(row_scores, dim=0)
weights = torch.softmax(row_scores.detach(), dim=0)
streamed = -(weights * row_scores).sum()
assert torch.allclose(torch.autograd.grad(reference, row_scores)[0], torch.autograd.grad(streamed, row_scores)[0])
```

Also prove that summing per-chunk `-logsumexp` gradients is different and that missing/duplicate candidate identities fail.

- [ ] **Step 2: Run pure tests and observe RED**

Run:

```bash
conda run -n ms python -m pytest -c /dev/null tests/losses/test_human13_k_union.py -q
```

Expected: new streaming helper symbols are absent.

- [ ] **Step 3: Implement fp32 global weights and weighted surrogate**

Add side-effect-free helpers that return content-bound per-image weights, effective owner count, candidate count, and `-(detached_weight * row_score).sum()`. Require finite fp32 scores, one unique candidate identity, and exact complete-set coverage.

- [ ] **Step 4: Write payload RED tests for physically split A4 groups**

Construct two candidates from one image whose aggregate exceeds the test pack bound while each segment fits. Assert `build_live_payload` succeeds, places them in different packs, and retains one logical A4 candidate-set identity. Assert one overlength segment still fails.

- [ ] **Step 5: Remove aggregate/one-pack rejection but retain individual bounds**

For A4, call `preflight(..., enforce_a4_aggregate=False)`, permit the existing first-fit planner to split candidates, and store a canonical per-image candidate census in the payload/execution plan. Do not alter non-A4 packing.

- [ ] **Step 6: Write runner RED tests for score-before-backward ordering**

Use a fake score forward and fake runtime to assert, for one exposure:

```text
score(pack0), score(pack1), backward(pack0), backward(pack1), optimizer_step
```

Assert the model version is identical across score/replay, all candidates appear exactly once in both passes, weights normalize globally per image, and the optimizer step count is exactly one.

- [ ] **Step 7: Implement `Human13A4TwoPassLossRunner`**

In `prepare_planned_step`, score every moved microstep under `torch.no_grad()`, gather target log-prob row scores by `(image_id, segment_id)`, compute global fp32 weights, and seal the candidate/model-state receipt. In `compute_micro_step`, replace local union dispatch with the detached-weight row-score surrogate while leaving replay and duplicate families unchanged. In `finalize_planned_step`, report one global weight vector/effective count per image and both score/replay forward counts.

- [ ] **Step 8: Select the specialized runner only for A4**

Keep the ordinary `Human13PanelLossRunner` for every other arm. In `train_prepared_arm`, inject the live model and the same Qwen forward surface into `Human13A4TwoPassLossRunner`; preserve the existing cumulative 1/2/4/8/16 checkpoint schedule and runtime optimizer.

- [ ] **Step 9: Run the focused A4 stack**

Run:

```bash
conda run -n ms python -m pytest -c /dev/null tests/losses/test_human13_k_union.py tests/research/test_human13_live_payload.py tests/research/test_run_human13_k_union_overfit.py tests/research/test_train_human13_live_arm.py -q
```

Expected: all pass, including exact gradient and call-order tests.

- [ ] **Step 10: Commit the A4 implementation**

```bash
git add src/losses/human13_k_union.py tests/losses/test_human13_k_union.py scripts/research/run_human13_k_union_overfit.py tests/research/test_run_human13_k_union_overfit.py scripts/research/human13_live_payload.py tests/research/test_human13_live_payload.py scripts/research/train_human13_live_arm.py tests/research/test_train_human13_live_arm.py
git commit -m "feat(research): stream Human-13 A4 exact union gradients"
```

### Task 5: CPU gate and production-shaped A4 vertical slice

**Files:**
- Modify only if a failing conclusion-changing test identifies a bounded defect in Task 2–4 files.
- Runtime output: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-missing-arms-successor/vertical/a4-image-14038/`

**Interfaces:**
- Consumes: revised OpenSpec, sealed manifest, repaired payloads, A4 streaming runner.
- Produces: CPU verification receipt and one real A4 score/replay/update/checkpoint readback.

- [ ] **Step 1: Run the integrated CPU gate**

Run the focused Human-13 loss/materializer/census/payload/runner/train/config/analyzer suites, Ruff on changed Python files, `py_compile`, and strict OpenSpec validation. Expected: all pass.

- [ ] **Step 2: Verify immutable roots and GPU availability**

Require the new root to be absent, inspect `nvidia-smi`, bind one free GPU, and record physical GPU UUID. Do not modify prior `matrix/v1` or census plan artifacts.

- [ ] **Step 3: Run one-image A4 vertical slice**

Run image `14038` for one update from fresh Source. Require complete score and replay candidate coverage, no model-version change between passes, one applied AdamW update, checkpoint write/read, finite losses, and bounded memory counters.

- [ ] **Step 4: Stop on any conclusion-changing seam**

Stop before the full successor if the score/replay sets differ, a candidate is overlength, the gradient is nonfinite, optimizer steps differ from one, checkpoint readback fails, or the no-update census is incomplete.

### Task 6: Execute, evaluate, and close the bounded successor

**Files:**
- Modify: `research/qwen3-vl-dense-enumeration/2026-08-12-human13-k-union-greedy-overfit-probe/results.md`
- Modify: `openspec/changes/add-human13-k-union-greedy-overfit-probe/tasks.md`
- Create: `openspec/changes/add-human13-k-union-greedy-overfit-probe/successor-verification.md`
- Runtime root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-missing-arms-successor/`

**Interfaces:**
- Consumes: passing CPU/vertical gates and complete census disposition.
- Produces: fresh A4/A6 and conditionally A8-prime checkpoints, clean-greedy outputs, owner analysis, bounded claim, and stop receipt.

- [ ] **Step 1: Materialize fresh resolved plans**

Create unique A4/A6/A8-prime plan and output roots bound to the sealed manifest, Source payload, fresh optimizer, and census/donor artifacts. Omit A8-prime if the census gate blocks it.

- [ ] **Step 2: Launch independent world-size-one arms**

Use at most one process per GPU and no retries that overwrite state. Train exposures `1,2,4,8,16`, write/read each milestone checkpoint, and retain runtime/performance receipts.

- [ ] **Step 3: Evaluate every milestone**

Run original-prompt HF fp32/SDPA batch-size-one greedy with repetition penalty `1.0` on all thirteen images. Require complete immutable JSONL and receipt coverage for each arm/milestone.

- [ ] **Step 4: Analyze against frozen Source**

Report per-image and pooled K-hit gain, Source retention/loss, K-miss incidental gain, unique owners, duplicate/unmatched/malformed/invalid/cap burden, rows, and tokens. Do not promote from training loss, A4 weights, or A8 margins.

- [ ] **Step 5: Update authority and verify**

Mark tasks 5.2/5.3 and 8.2–8.7 only when their exact evidence exists, append a bounded successor section to the owning results record, add `successor-verification.md`, run strict OpenSpec validation and changed-path tests, and commit explicit paths.

- [ ] **Step 6: Stop at the complete successor table**

Do not launch a 100-update continuation, K-miss supervision, online refresh, checkpoint promotion, stable-spec sync, or archive. Return the supported claim and exact artifact roots to the user.

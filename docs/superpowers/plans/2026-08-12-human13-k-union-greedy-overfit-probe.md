# Human-13 K-Union-to-Greedy Overfit Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:executing-plans` to execute this plan task by task only after the
> user explicitly authorizes implementation. Use
> `superpowers:test-driven-development` for behavior changes and
> `superpowers:verification-before-completion` before any completion claim.

**Goal:** Implement the shortest experiment-local path that can test whether
native K-hit owner rows can be consolidated into the exact Human-13 panel's
clean-greedy output.

**Architecture:** OpenSpec is the sole implementation authority. A frozen
experiment manifest feeds pure objective helpers and an adapter over the
existing Swift packing/training/inference spine; an experiment analyzer
projects raw clean-greedy outputs back onto frozen owner sets. No generic
StateBank exception, new trainer, candidate-tree framework, or external bridge
is introduced.

**Tech Stack:** Python 3 in Conda `ms`, PyTorch, Transformers/Qwen3-VL, PEFT
DoRA, Accelerate world-size one, FlashAttention-2 varlen packing, vLLM sampled
discovery, pytest, OpenSpec.

## Authority and stop boundary

This execution plan is subordinate to:

- [research unit](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-12-human13-k-union-to-greedy-overfit-screen/unit.md), which owns scientific meaning;
- [OpenSpec proposal](../../../openspec/changes/add-human13-k-union-greedy-overfit-probe/proposal.md);
- [OpenSpec specification](../../../openspec/changes/add-human13-k-union-greedy-overfit-probe/specs/coordexp-swift-human13-k-union-greedy-probe/spec.md);
- [OpenSpec design](../../../openspec/changes/add-human13-k-union-greedy-overfit-probe/design.md); and
- [OpenSpec tasks](../../../openspec/changes/add-human13-k-union-greedy-overfit-probe/tasks.md).

Do not reinterpret or duplicate their arm semantics here. If this plan and
OpenSpec differ, stop and update the planning artifacts before touching code.
Planning alone authorizes no implementation, model load, GPU allocation,
optimizer action, checkpoint write, commit, or launch.

Keep execution narrow. Do not add optional abstraction, generalized policy,
cache layer, duplicate receipt family, or extra audit because it might be
useful later. Retire only conclusion-changing risks at each gate.

## Task 1: Re-enter at the explicit implementation gate

**Files:**

- Read: `AGENTS.md`
- Read: `openspec/changes/add-human13-k-union-greedy-overfit-probe/{proposal.md,design.md,tasks.md}`
- Read: `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-12-human13-k-union-to-greedy-overfit-screen/{unit.md,review.md,handoff.md}`
- Inspect: only the exact paths named in the active OpenSpec wave

**Step 1: Verify authority and worktree ownership**

Run:

```bash
RTK_HOOK_DISABLE=1 git status --short
openspec status --change add-human13-k-union-greedy-overfit-probe --json
```

Expected: explicit implementation authorization is present in the handoff;
unrelated dirty OwnerBridge/research files are identified and excluded.

**Step 2: Freeze the active wave**

Start with OpenSpec tasks 1.x and 2.x only. Do not pre-create runner, launcher,
or GPU artifacts. Record the exact commands and changed paths in the current
task commentary, not a second plan document.

**Step 3: Stop conditions**

Stop if authorization is absent, the named worktree/branch differs, the Source
or panel identity differs, or any required file overlaps unresolved user work.

## Task 2: Build the frozen manifest test-first

**Files:**

- Create: `scripts/research/build_human13_k_union_manifest.py`
- Create: `tests/research/test_build_human13_k_union_manifest.py`
- Reuse: `scripts/research/analyze_individual_trajectory_union_support.py`
- Reuse: `scripts/research/compare_clean_rollout_owner_coverage.py`
- Reuse without weakening: `src/rollout_calibration/state_bank.py`

**Step 1: Write failing identity/admission tests**

Cover exact panel hash/unit purpose, wrong-hash rejection, generic blind-path
rejection, thirteen image identities, explicit Source/prompt/tokenizer/matcher
identity, canonical serialization, and dry-run zero-action output.

Run:

```bash
conda run -n ms python -m pytest \
  tests/research/test_build_human13_k_union_manifest.py -q
```

Expected: FAIL because the manifest builder does not exist.

**Step 2: Add the smallest typed manifest surface**

Implement canonical dataclasses or strict Pydantic records in the script-local
module. Keep label construction private; expose load, validate, build,
canonical-write, and dry-run summary interfaces only.

**Step 3: Write failing owner/prefix tests**

Cover chronological class-agnostic IoU `>0.95` classification before matching,
raw `G/H/M`, deterministic native target-row selection, exact token-span
concatenation, raw duplicate-state preservation, masked unmatched/invalid
spans, every duplicate event, terminal exclusion, and global denominator
counts. Include a dense two-GT fixture where a later near-identical prediction
could otherwise receive a second owner; assert it is duplicate-only and occurs
in no matched/replay/target/candidate positive set.

**Step 4: Implement only enough builder logic to pass**

Reuse the owning research matcher semantics rather than the visualization
matcher. Never decode/re-tokenize retained spans. Emit one manifest and one
digest; do not create a separate seal hierarchy.

**Step 5: Verify**

Run the focused test plus existing matching/union tests:

```bash
conda run -n ms python -m pytest \
  tests/research/test_build_human13_k_union_manifest.py \
  tests/research/test_analyze_individual_trajectory_union_support.py \
  tests/research/test_compare_clean_rollout_owner_coverage.py -q
```

Expected: PASS with no generic StateBank behavior change.

## Task 3: Add explicit 4-by-4 K collection

**Files:**

- Create: `scripts/research/collect_human13_k16_vllm.py`
- Create: `tests/research/test_collect_human13_k16_vllm.py`
- Reuse: `src/inference/backend.py`
- Reuse: `src/inference/vllm_backend.py`

**Step 1: Write failing plan-only collector tests**

Assert four successive request batches, four independent `n=1` requests per
batch, exact seed ownership/order, frozen sampling settings, incomplete and
duplicate seed failures, optional cache telemetry, and no `n>1` request.

Run:

```bash
conda run -n ms python -m pytest \
  tests/research/test_collect_human13_k16_vllm.py -q
```

Expected: FAIL before implementation.

**Step 2: Implement mechanical request planning and result binding**

Use the existing backend session. Keep engine creation behind the ordinary
explicit execution path; `--dry-run` must not import/load a model or touch a
GPU. Restore output rows to seed order before passing them to the manifest.

**Step 3: Verify CPU behavior**

Run the focused collector tests and a dry-run for the exact panel. Inspect that
the plan contains 13 images, 52 physical batches, and 208 explicit requests,
with no model action.

## Task 4: Implement pure objectives and the no-update census

**Files:**

- Create: `src/losses/human13_k_union.py`
- Create: `tests/losses/test_human13_k_union.py`
- Create: `scripts/research/census_human13_k_union_trie.py`
- Create: `tests/research/test_census_human13_k_union_trie.py`
- Reuse: `src/losses/rollout_calibration.py`
- Reuse: `src/training/rollout_calibration.py`

**Step 1: Write failing pure-math tests**

Use tiny tensors to prove fp32 math, masks, owner/image normalization,
prefix-free union weights/effective count, zero gradient at satisfied A8-prime
sites, detached competitor selection, every duplicate event, stable finite
unlikelihood/gradient when the target margin is at least thirty nats, finite
failures for truly nonfinite logits, and zero-denominator behavior.

Run:

```bash
conda run -n ms python -m pytest tests/losses/test_human13_k_union.py -q
```

Expected: FAIL because the helper module does not exist.

**Step 2: Implement the four helpers**

Keep them tensor-only and side-effect-free. Return bounded diagnostics with
explicit numerator/denominator; do not parse, match, choose owners, or decide
arm membership inside the loss module.

**Step 3: Write and implement census tests**

Test coherent full-H traversal, first non-argmax/minimum strict margin, tie
receipts, token roles, aligned surface drift, A8-prime block conditions, and
byte-identical manifest targets before/after census.

**Step 4: Verify and gate CPU Wave 1**

Run:

```bash
conda run -n ms python -m pytest \
  tests/losses/test_human13_k_union.py \
  tests/research/test_census_human13_k_union_trie.py \
  tests/research/test_build_human13_k_union_manifest.py \
  tests/research/test_collect_human13_k16_vllm.py -q
openspec validate add-human13-k-union-greedy-overfit-probe --strict
RTK_HOOK_DISABLE=1 git diff --check -- \
  src/losses/human13_k_union.py scripts/research tests/losses tests/research
```

Request one bounded standards/intent review. Resolve P0/P1 only; do not expand
into optional infrastructure review.

## Task 5: Adapt the packed planned-step runner

**Files:**

- Create: `scripts/research/run_human13_k_union_overfit.py`
- Create: `tests/research/test_run_human13_k_union_overfit.py`
- Modify only if required by a failing interface test: `src/losses/runner.py`
- Reuse: `src/packing/planner.py`
- Reuse: `src/packing/supervision.py`
- Reuse: `src/qwen/forward.py`
- Reuse: `src/training/supervised_trainer.py`
- Reuse: `src/runtime/train_runtime.py`

**Step 1: Write failing runner contract tests**

Cover logical segment roles, stable descending-length first-fit, independent
attention/MRoPE, 12,000-token failures, coherent A1/A8/full-GT segments, atomic
A4 groups, complete-panel denominators, unchanged parameters between packs,
and exactly one optimizer step per exposure.

**Step 2: Implement the experiment adapter**

Map manifest sites onto existing compact causal logits and existing planned-
step accumulation. Prefer an experiment-local loss dispatch over modifying the
generic registry. Touch `src/losses/runner.py` only if an interface-level test
proves the current seam cannot carry explicit numerators/denominators.

**Step 3: Add compact performance counters**

Record only the counters named by OpenSpec. Do not claim common-prefix/image
reuse or add a profiler/cache subsystem.

**Step 4: Verify**

Run focused runner, packing, Qwen-forward, planned-step, and loss tests. Assert
that dry-run has zero model actions and no ordinary config acquires Human-13
admission.

## Task 6: Materialize arms and outcome analysis

**Files:**

- Create: `configs/coordexp_swift/research/human13_k_union/*.yaml`
- Create: `scripts/research/materialize_human13_k_union_configs.py`
- Create: `scripts/research/analyze_human13_k_union.py`
- Create: `scripts/research/launch_human13_k_union_matrix.py`
- Create: `tests/research/test_materialize_human13_k_union_configs.py`
- Create: `tests/research/test_analyze_human13_k_union.py`
- Create: `tests/research/test_launch_human13_k_union_matrix.py`

**Step 1: Write failing config/materializer tests**

Assert the exact approved arm set and A0 shared-background name,
Source/fresh-AdamW independence, the spec-frozen trainable surface, AdamW,
sixteen-step scheduler, clipping and H/replay/duplicate coefficients,
conditional A6, unique output roots, no deferred arm, and plan-only behavior.

**Step 2: Materialize strict configs**

Use existing strict Swift config resolution. Keep research-unit selectors in
the experiment manifest/runner rather than broadening ordinary production
schema where possible.

**Step 3: Write failing analyzer tests**

Cover chronological duplicate exclusion before cardinality-first/maximum-total-
IoU assignment, no owner credit for later duplicates, gains versus losses,
K-miss incidental recovery, burden fields, and all required panel slices. Add
a regression where one K-hit gain and one Source loss must remain separate.

**Step 4: Implement the analyzer and guarded launcher**

The launcher requires an explicit execution flag and separately documented
user launch authority; its default is dry-run. It starts at most one
world-size-one arm per selected GPU and never shares run roots or state.

**Step 5: Gate CPU Wave 2**

Run all new tests plus relevant existing config, packing, training-artifact,
inference, and matching tests; run strict OpenSpec validation, diff/residue
checks, and one bounded standards/intent review. Stop on P0/P1.

## Task 7: Acquire and freeze the full model-derived ledger

**Files:**

- Write artifacts only under the unit's declared discovery run root
- Consume: the implemented collector, manifest builder, and census

**Step 1: Reconfirm discovery authority**

The user must explicitly authorize model/GPU execution. This gate performs no
optimizer update. Inspect live GPU/process state and bind the exact Source,
panel, HF clean-greedy, and vLLM K settings.

**Step 2: Execute full-panel discovery**

Acquire or identity-verify all thirteen HF batch-size-one Source outputs and
all 208 explicit K requests. Do not use a fixture, partial image set, or stale
ledger to select an update case.

**Step 3: Seal the manifest and census**

Apply chronological duplicate exclusion before owner matching, finalize all
thirteen `G/H/M` sets and selected rows, then run the no-update census to seal
A6 applicability and A8-prime margin/applicability. Verify sixteen unique
seeds per image and an empty duplicate/positive-set intersection.

**Step 4: Stop before updates**

Publish the canonical manifest/census identities and compact runtime counters.
No successful discovery implies permission for a backward pass.

## Task 8: Run one production-shaped image only after update authorization

**Files:**

- Write artifacts only under the unit's declared output root
- Update after evidence: the owning research unit files, not OpenSpec semantics

**Step 1: Reconfirm the separate authority**

The user must explicitly authorize the update slice. Inspect live GPU/process
state and choose one eligible image from the sealed full-panel manifest without
changing targets or arm semantics.

**Step 2: Execute the vertical slice**

Run materialization, one isolated pack path, forward/backward, one applied
update, checkpoint write/read, HF batch-size-one clean greedy, declared match,
and analyzer. Do not start another arm in the same command.

**Step 3: Verify the receipt**

Require Source identity, masks, denominators, applied-step count, finite loss,
checkpoint reload, raw decode, owner table, pack counters, GPU seconds, wall
time, and peak memory. Preserve an OOM/overlength/mismatch as evidence; do not
invent a fallback.

**Step 4: Stop for the matrix decision**

Return measured cost and any P0/P1 review findings. No full matrix is implied
by a successful vertical slice.

## Task 9: Run the matrix only after a second launch decision

**Step 1: Obtain explicit matrix authority**

Bind the selected applicable arms, GPU allocation (maximum eight independent
one-rank jobs), the frozen sixteen-update optimizer contract, exposure
milestones `0,1,2,4,8,16`, and output roots.

**Step 2: Execute and analyze without adaptive changes**

Start every arm from byte-identical Source/fresh AdamW, freeze each clean-
greedy milestone output, and publish the required owner and burden tables.

**Step 3: Stop at the first table**

Do not start a fresh Source/AdamW 100-update run reaching `32,64,100`, retune,
refresh targets, supervise K-miss, add new arms, promote a checkpoint, archive
OpenSpec, commit, or publish without another explicit user decision. The long
run is not an optimizer continuation of the sixteen-update run.

**Step 4: Verification before any completion claim**

Use `superpowers:verification-before-completion`, run OpenSpec verification,
and request one final bounded review. Update the research unit and concise
project memory only with executed evidence and its claim boundary.

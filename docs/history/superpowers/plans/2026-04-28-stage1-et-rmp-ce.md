# Stage-1 ET-RMP-CE Implementation Plan

> Archived / superseded on 2026-05-02.
> Historical provenance only for the pre-refactor Stage-1 set-continuation family.
> Do not use this file as an execution source.
> Active execution sources:
> - `docs/superpowers/specs/2026-05-02-training-infra-template-mode-refactor-design.md`
> - `docs/superpowers/plans/2026-05-02-training-infra-template-mode-refactor.md`

## Historical Execution Notes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an off-by-default Stage-1 set-continuation objective mode, ET-RMP-CE, that trains one recursive full suffix with entry-trie multi-positive CE at every object-entry divergence node.

**Architecture:** Keep `custom.trainer_variant: stage1_set_continuation` and add an objective selector under `custom.stage1_set_continuation.objective`. Reuse existing prefix/object serialization and smart-batch padded-row scheduling, but add a sibling full-suffix encoder, trie target builder, and full-vocab CE loss instead of mutating current candidate-balanced branch scoring.

**Tech Stack:** Python, PyTorch, CoordExp strict dataclass config, Qwen3-VL chat-template encoding, existing `stage1_set_continuation` sampler/serialization/smart-batch utilities, pytest, `rtk conda run -n ms python -m pytest`.

---

## File Structure

- Create: `src/trainers/stage1_set_continuation/entry_trie.py`
  - Own pure serialized-entry trie construction, multiplicity tracking, branch classification, and token-level target-step metadata.
- Create: `src/trainers/stage1_set_continuation/full_suffix.py`
  - Own full-suffix order sampling, full-suffix branch encoding, ET-RMP/full-suffix CE loss computation, and smart-batched row scoring.
- Modify: `src/config/schema.py`
  - Add strict `Stage1SetContinuationObjectiveConfig` under `Stage1SetContinuationConfig`.
- Modify: `src/trainers/stage1_set_continuation/trainer.py`
  - Branch `compute_loss` to the full-suffix batch path when `objective.mode` is `full_suffix_ce` or `entry_trie_rmp_ce`.
  - Keep candidate-balanced execution only for legacy compatibility with old configs/tests.
- Modify: `src/trainers/stage1_set_continuation/metrics.py`
  - Add compact emitted ET-RMP metric keys without removing candidate-balanced keys.
- Modify: `configs/stage1/set_continuation/rmp_ce.yaml`
  - Add an experimental profile for ET-RMP-CE using current smart-batched exact runtime.
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
  - Document ET-RMP-CE semantics and probability space.
- Modify: `docs/training/METRICS.md`
  - Document trainer-native ET-RMP diagnostics and eval-side rollout hygiene.
- Create/Modify OpenSpec under `openspec/changes/add-stage1-et-rmp-ce-objective/`.
- Test: `tests/test_stage1_set_continuation_entry_trie.py`
  - Pure trie and target tests.
- Test: `tests/test_stage1_set_continuation_full_suffix.py`
  - Full-suffix rendering, recursive state update, boundary/close/EOS masks, smart-batch parity.
- Modify: `tests/test_stage1_set_continuation_config.py`
  - Strict objective config tests.
- Modify: `tests/test_stage1_set_continuation_metric_keys.py`
  - Emitted ET-RMP metric key tests.

## Task 1: Governance Artifacts

- [ ] **Step 1: Write OpenSpec proposal/design/tasks/spec deltas**

Create `openspec/changes/add-stage1-et-rmp-ce-objective/proposal.md`,
`design.md`, `tasks.md`, and
`specs/stage1-set-continuation-training/spec.md`. The delta must require:
`objective.mode`, full-suffix row semantics, all-divergence entry-trie MP,
full-vocab main CE, duplicate-entry multiplicity, and smart-batched exact
full-suffix rows.

- [ ] **Step 2: Run OpenSpec validation**

Run:

```bash
openspec validate add-stage1-et-rmp-ce-objective --strict
```

Expected: validation succeeds.

## Task 2: Red Tests For Trie Targets

- [ ] **Step 1: Add pure trie tests**

Create `tests/test_stage1_set_continuation_entry_trie.py` with tests for desc
branching, coord branching, shared coordinate unique paths, later coordinate
branching, object-uniform probabilities, and duplicate serialized-entry
multiplicity.

- [ ] **Step 2: Verify red**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_entry_trie.py
```

Expected: fails because `src.trainers.stage1_set_continuation.entry_trie` does
not exist.

## Task 3: Implement Trie Targets

- [ ] **Step 1: Add `entry_trie.py`**

Implement frozen dataclasses for trie nodes and target steps. Build tries from
`(object_index, tokens, token_types)` rows. At each teacher-forced token,
return one target step with child-token counts, object-uniform probabilities,
branch/unique flag, and token-type classification.

- [ ] **Step 2: Run pure trie tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_entry_trie.py
```

Expected: all trie tests pass.

## Task 4: Red Tests For Full-Suffix Loss And Smart Batching

- [ ] **Step 1: Add full-suffix tests**

Create `tests/test_stage1_set_continuation_full_suffix.py` covering:
recursive remaining-set update, teacher-forced suffix order, boundary tokens
outside the entry trie, final close/EOS CE outside the trie, full-suffix CE mode
with no MP, ET-RMP branch CE at all divergence nodes, and smart-batched
full-suffix parity against serial scoring.

- [ ] **Step 2: Verify red**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_full_suffix.py
```

Expected: fails because `full_suffix.py` and config hooks are not implemented.

## Task 5: Implement Full-Suffix Objective

- [ ] **Step 1: Add `full_suffix.py`**

Implement full-suffix order sampling, row input dataclasses, full-vocab
token-level CE, soft CE at branch steps, hard CE at unique/boundary/close/EOS
steps, metrics accumulation, and smart-batched retained scoring using the
existing branch-batcher row scheduler.

- [ ] **Step 2: Add trainer branch**

Modify `Stage1SetContinuationTrainer.compute_loss` so `objective.mode` in
`full_suffix_ce` or `entry_trie_rmp_ce` builds all full-suffix rows for
`meta_list` and scores them through the full-suffix batch path. Leave the
current `_process_sample` candidate-balanced path available only for legacy
compatibility.

- [ ] **Step 3: Run full-suffix tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_full_suffix.py
```

Expected: all full-suffix tests pass.

## Task 6: Config, Metrics, Profile, And Docs

- [ ] **Step 1: Add config tests and implementation**

Extend `tests/test_stage1_set_continuation_config.py` to parse
`objective.mode: entry_trie_rmp_ce`, reject unknown modes, and keep omitted
objective config behavior compatible while documenting that `candidate_balanced`
is deprecated for production.

- [ ] **Step 2: Add emitted metric keys**

Update `tests/test_stage1_set_continuation_metric_keys.py` and
`src/trainers/stage1_set_continuation/metrics.py` for compact ET-RMP keys.

- [ ] **Step 3: Add YAML profile**

Create `configs/stage1/set_continuation/rmp_ce.yaml` extending the current
production shape but changing artifact metadata and setting
`objective.mode: entry_trie_rmp_ce`.

- [ ] **Step 4: Update docs**

Update `docs/training/STAGE1_OBJECTIVE.md` and `docs/training/METRICS.md` with
ET-RMP-CE semantics and diagnostics.

## Task 7: Verification

- [ ] **Step 1: Run targeted tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_stage1_set_continuation_entry_trie.py \
  tests/test_stage1_set_continuation_full_suffix.py \
  tests/test_stage1_set_continuation_config.py \
  tests/test_stage1_set_continuation_metric_keys.py \
  tests/test_stage1_set_continuation_loss.py \
  tests/test_stage1_set_continuation_branch_runtime.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Run OpenSpec validation**

Run:

```bash
openspec validate add-stage1-et-rmp-ce-objective --strict
```

Expected: validation succeeds.

- [ ] **Step 3: Inspect final diff**

Run:

```bash
rtk git diff --stat
rtk git diff --check
```

Expected: scoped changes, no whitespace errors.

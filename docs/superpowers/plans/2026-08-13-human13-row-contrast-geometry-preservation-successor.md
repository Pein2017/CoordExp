# Human-13 Row-Contrast and Geometry-Preservation Successor Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. The owning OpenSpec change is authoritative for scope and completion.

**Goal:** Implement and run the smallest two-arm Human-13 successor that tests owner-aware duplicate-row redirection, rectangle-valid greedy coordinate decisions, and first-order preservation of previously greedy-visible owner coordinates.

**Architecture:** Extend the existing experiment-local Human-13 manifest/materialization/packed-runner/live-eval spine. Add one immutable derived sidecar, pure fp32 loss/projection helpers, and R1/R2 adapters; reuse the existing A4 fixed-parameter streaming objective, DoRA model assembly, one-rank AdamW trainer, checkpoint writer/readback, HF evaluator, and analyzer.

**Tech Stack:** Python 3.11, PyTorch, Qwen3-VL/Transformers, FlashAttention-2 varlen packing, Accelerate world-size one, pytest, Ruff, OpenSpec.

**Authority:** [OpenSpec change](../../../openspec/changes/archive/2026-08-28-add-human13-row-contrast-geometry-preservation-successor/) and [research unit](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-13-human13-row-contrast-geometry-preservation-successor/unit.md). This plan is execution discipline only.

---

### Task 1: Successor sidecar and exact event derivation

**Files:**
- Create: `scripts/research/build_human13_row_contrast_successor.py`
- Test: `tests/research/test_build_human13_row_contrast_successor.py`

1. Write failing tests for manifest identity, complete duplicate-row recovery,
   exact prefix/coordinate positions, covered/uncovered `G union H`, alias
   grouping, `G` watch rows, deterministic digest, identical-event
   deduplication, and typed optional A4 exclusion.
2. Run
   `conda run -n ms python -m pytest -c /dev/null tests/research/test_build_human13_row_contrast_successor.py -q`
   and retain the expected RED result.
3. Implement immutable dataclasses, canonical serialization, manifest-derived
   event projection, and optional prior-output adapter. Reuse existing parser,
   tokenizer-span alignment, and artifact identity helpers; do not approximate
   missing spans.
4. Re-run the focused test to GREEN and run the existing manifest/analyzer
   tests to detect semantic drift.

### Task 2: Pure hierarchical contrast and rectangle losses

**Files:**
- Modify: `src/losses/human13_k_union.py`
- Modify: `tests/research/test_run_human13_k_union_overfit.py`

1. Add failing tensor tests for same-description bbox contrast, cross-
   description owner-normalized row contrast, four-coordinate fallback UL,
   alias-count invariance, image balancing, invalid shapes/IDs, set-valued
   x2/y2 validity, alternate-valid zero loss, global-invalid competitor, and
   detached selector gradients.
2. Run the focused test and retain RED.
3. Implement side-effect-free fp32 helpers and typed results. Helpers consume
   selected tensor sites and owner/event indices only; they do not parse,
   match, or choose research labels.
4. Re-run focused tests and `tests/losses`/adjacent Human-13 objective tests.

### Task 3: Accumulated-gradient preservation

**Files:**
- Create: `scripts/research/human13_gradient_preservation.py`
- Test: `tests/research/test_human13_gradient_preservation.py`

1. Add failing tests for negative-dot projection, non-conflicting no-op,
   parameter-order independence, exact post-dot tolerance, dtype/device
   handling, missing/non-finite gradients, zero watch norm, and receipt fields.
2. Implement a world-size-one helper that accepts named trainable-parameter
   gradient buffers, computes fp32 global dot/norms, writes the projected R1
   gradient before clipping, and returns an immutable receipt.
3. Re-run tests and a finite-difference toy update demonstrating the declared
   raw first-order property.

### Task 4: Materialization, packing, and runner integration

**Files:**
- Modify: `scripts/research/human13_live_segments.py`
- Modify: `scripts/research/human13_live_payload.py`
- Modify: `scripts/research/run_human13_k_union_overfit.py`
- Modify: `scripts/research/train_human13_live_arm.py`
- Test: `tests/research/test_human13_live_segments.py`
- Test: `tests/research/test_human13_live_payload.py`
- Test: `tests/research/test_run_human13_k_union_overfit.py`
- Test: `tests/research/test_train_human13_live_arm.py`

1. Add failing interface tests for R1/R2 arm admission, exact successor-sidecar
   binding, candidate/replay/duplicate/rectangle/watch roles, global
   denominators, A4 score/replay identity, one optimizer mutation per panel,
   R2 separate backward buffers, and projection receipt serialization.
2. Extend existing typed roles/sites rather than adding an untyped metadata
   channel. Keep historical arms byte-compatible.
3. Reuse A4's global score pass and detached-weight replay for the any-valid
   term. Dispatch new row/rectangle losses from compact logits and add an R2
   runner that projects only after the complete panel gradients exist.
4. Run the four focused modules plus current Human-13 runner/config suites.

### Task 5: Resolved configs and bounded launcher

**Files:**
- Create: `configs/coordexp_swift/research/human13_row_contrast_successor/01_r1.yaml`
- Create: `configs/coordexp_swift/research/human13_row_contrast_successor/02_r2.yaml`
- Modify: `scripts/research/materialize_human13_k_union_configs.py`
- Modify: `scripts/research/launch_human13_k_union_matrix.py`
- Test: `tests/research/test_materialize_human13_k_union_configs.py`
- Test: `tests/research/test_launch_human13_k_union_matrix.py`

1. Add RED tests for the exact two-arm inventory, exposures one/two only,
   fresh Source/AdamW, world-size one, unique roots, required sidecar digest,
   R2 projection constants, six-GPU maximum critical path, default dry-run,
   and no retries/hidden arms.
2. Implement the smallest config/materializer/launcher extension. Do not alter
   the predecessor's resolved plans or artifacts.
3. Run dry-run and assert zero model, forward, backward, optimizer, checkpoint,
   inference, and analyzer actions.

### Task 6: CPU gate and production-shaped vertical slice

**Files:**
- Update: `.superpowers/sdd/2026-08-13-human13-row-contrast-geometry-preservation-successor/progress.md`

1. Run focused and adjacent pytest suites, Ruff format/check, `py_compile`,
   `openspec validate add-human13-row-contrast-geometry-preservation-successor --strict`,
   and explicit diff/residue inspection.
2. Materialize the canonical sidecar under a new immutable output root and
   record hashes/counts/exclusions.
3. Dry-run the exact image-14038 R1 command, then execute one real update on one
   GPU with explicit authority.
4. Read back the checkpoint and run exact HF fp32/SDPA batch-one greedy plus
   analyzer. Stop on any unit-declared gate; do not patch around a scientific
   mismatch.

### Task 7: Two-arm low-dose execution and analysis

**Files:**
- Update: `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-13-human13-row-contrast-geometry-preservation-successor/unit.md`
- Create: `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-13-human13-row-contrast-geometry-preservation-successor/results.md`

1. Launch R1 and R2 independently on two GPUs to cumulative exposures one and
   two. Preserve checkpoint-one before continuing to checkpoint-two.
2. After both training jobs publish valid checkpoints, evaluate four
   checkpoints on up to four GPUs under the exact clean-greedy surface.
3. Run the canonical analyzer and compare against immutable A4@2 using the full
   owner/burden/runtime tuple and per-image identities.
4. Write bounded results, artifact hashes, supported/ruled-out/unresolved
   statements, and stop. Do not launch exposure four or online refresh.

### Task 8: Completion verification

1. Re-run targeted tests, strict OpenSpec validation, config/dry-run checks,
   artifact hash/readback checks, and analyzer schema validation.
2. Review the final diff for standards and separately compare behavior against
   the research unit/OpenSpec intent. Resolve all P0/P1 findings.
3. Mark OpenSpec tasks complete only for work with direct evidence, update the
   goal only after all authorized execution/results work is finished, and
   leave archive/promotion to a later explicit decision.

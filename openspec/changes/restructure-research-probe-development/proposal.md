## Why

Research producers repeatedly reconstruct model inputs, replay/scoring, parameter selection and decoding while inheriting historical scripts and sibling-worktree imports. This change makes the independent research base smaller to use and maintain, consolidates research knowledge, and preserves exact historical sources and necessary evidence before retiring old worktrees.

## What Changes

- Establish ordinary Qwen-native input/replay/generation operations beneath the existing deterministic HF-session and packed-training adapters. Reuse neutral loading; do not turn either adapter into a universal research runtime.
- Share differentiable aligned-token scoring, explicit adapter parameter selection, and category-constrained global assignment. Scientific objectives, token selection, weights, denominators, intervention formulas and stopping choices remain visible in direction code.
- Support exact prefix/action continuation with heterogeneous budgets and batches, terminal-action fast paths and optional traces. Preserve each existing sampling policy, including DORA's raw-softmax difference, rather than imposing an invented common default.
- Initially retain four direction packages: `dora_owner_learning`, `source_rweak_row_cross`, `logit_lens`, and `human13`. Keep separate protocols/configurations within a direction; do not create a package for every run, mechanism keyword, or historical worktree.
- **BREAKING**: Replace selected historical script/private import paths with those packages and public operations. Completed C/COCO trainers, N256's hash-bound solvers and unfinished row-feedback producers are initially historical-recovery routes, not newly maintained packages. Preserve their full effective source/config dependencies and required outputs before removing them from the default tree.
- **BREAKING**: Remove the unused exported `BackendScalingReceipt` and `validate_planned_step_backend_scaling` diagnostic API after its scoped reachability check. Preserve the actual accelerator accumulation guard, normalization and runtime tests.
- Keep strict JSON publication, journal recovery, admission and v1 persisted identities. Simplify ordinary producers and local workflow defaults so they do not construct admission or hash chains unnecessarily. Do not introduce a second lightweight evidence framework.
- Consolidate unique research records, rewrite current synthesis by research question, and keep original evidence and conflicting scopes traceable. Retire `research-probe-infras` and eligible direction worktrees after preservation and intake.

## Capabilities

### New Capabilities

- `research-probe-development`: independently executable direction packages, separate scientific profiles, knowledge retention and preservation-before-retirement in one research base.

### Modified Capabilities

- `coordexp-infras-research-probe-infra-base`: native research execution independent of packed/session machinery; exact replay and budgeted continuation; aligned scoring and assignment with explicit semantics; lightweight default usage with optional strict capabilities.

The existing deterministic scored-inference, packed training, admission/journal and execution-model export contracts retain their behavior. New research operations are lower-level capabilities, not an undocumented expansion of the deterministic inference schema. Internal symbol moves without observable contract changes do not create additional capability specs.

## Impact

- Research-owned `src/` is fully eligible for redesign. The detailed module disposition in `design.md` distinguishes concrete changes from retained load-bearing infrastructure and rejected speculative cuts; no untouched directory is assumed immutable.
- Initial edits affect selected Qwen/HF mechanics, losses, adapter/optimizer inputs, evaluation geometry/assignment, direction packages, their callers/tests/configs, local documentation and this change. Production main and coordexp-infras are unaffected.
- Preserve exact tokens, image/position alignment, gradients, reductions, category/matching policy, raw versus policy likelihood, RNG behavior, output collision protection, historical receipt identity and required recovery semantics.
- Exclude `permanent-owner-bridge-cache-validation` entirely, plus unrelated dirty work, shared agent/runtime configuration, credentials and remote refs. No GPU experiment, scientific rerun, universal trainer, registry, scheduler or migration database is authorized by this planning revision.
- Authority: the handoff's accepted single-base lifecycle and package direction, followed by the user's expansion to all research `src/`, substantive knowledge consolidation, stop confirmation, and explicit request for detailed subagent investigation followed by lead-owned OpenSpec revision. The revised design is lead-selected under that request; artifact existence is not implementation completion or scientific acceptance.

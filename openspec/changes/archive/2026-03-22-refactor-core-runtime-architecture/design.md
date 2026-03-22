## Context

CoordExp is a grounding/detection research stack whose primary engineering constraint is not generic software cleanliness, but the ability to keep high-velocity research changes reproducible and auditable.

That constraint changes what a "good refactor" means:

- preserving geometry semantics matters more than aesthetic normalization,
- preserving prompt/template parity matters more than local simplification,
- and preserving stable metric/artifact contracts matters more than making modules superficially smaller.

The current codebase already contains the beginnings of a healthier architecture:

- `src/config/schema.py` is acting as a strict typed contract authority rather than a loose config bucket.
- `src/trainers/stage2_two_channel/` has already split scheduling and execution concerns into separate files.
- `src/trainers/teacher_forcing/` centralizes objective math and naming in a more reusable way than older trainer-local loss code.

The problem is that the runtime-critical orchestration layer is still too braided:

- `src/trainers/stage2_rollout_aligned.py` centralizes backend lifecycle, rollout execution, target construction, evaluation, and training-time metric/reporting concerns.
- `src/trainers/stage2_two_channel.py` still contains a large amount of clean-prefix and triage logic plus metric projection in two giant methods.
- `src/sft.py` is both a repo entrypoint and a policy owner for packing, trainer injection, manifest generation, and provenance.
- `src/infer/engine.py` and `src/eval/detection.py` are each mixing outer orchestration with backend/artifact-specific logic.

The intended refactor should preserve this end-to-end flow:

`offline JSONL + geometry authority + typed config`
-> `dataset build / template encode / packing`
-> `trainer-specific runtime orchestration`
-> `teacher-forcing / inference / evaluation execution`
-> `metrics + artifacts + provenance`

The architecture should be reorganized around explicit boundaries inside that flow, not around arbitrary file sizes.

Constraints and invariants:

- Keep training YAML-first and config-first.
- Preserve coordinate order and geometry invariants; `src/datasets/geometry.py` remains authoritative.
- Maintain Qwen3-VL chat-template compatibility.
- Do not edit upstream HF internals such as `modeling_qwen3_vl.py`.
- Preserve the current default posture:
  - single-dataset training,
  - packing as the primary efficiency lever,
  - fusion-config training as legacy/experimental.
- Keep current operator-facing artifacts and stable metrics unchanged during initial extraction phases.

## Goals / Non-Goals

**Goals**

- Make runtime-critical surfaces easier to reason about by introducing narrower ownership boundaries.
- Distinguish stable contract authority from implementation/runtime orchestration.
- Reduce cross-cutting trainer coupling so Stage-2 work does not require reopening one god module.
- Preserve or improve testability by making internal interfaces explicit.
- Allow future research changes to land in smaller, more local edits.
- Preserve current behavior while extracting seams, with verification after each phase.

**Non-Goals**

- Re-designing Stage-2 training semantics as part of the first refactor slices.
- Introducing new CLI flags as a substitute for architecture work.
- Rewriting `src/config/schema.py` purely because it is large.
- Changing geometry contracts, prompt contracts, or artifact schemas without a deliberate follow-on spec change.
- Building speculative abstractions for multi-dataset or fusion workflows ahead of proven need.

## Decisions

### Decision 1: OpenSpec stays normative; GSD stays execution-oriented

OpenSpec remains the authoritative layer for scope, stable contracts, and behavior changes.
GSD planning artifacts are execution-oriented derivatives of that authority:

- `.planning/PROJECT.md` mirrors the program mission and guardrails,
- `.planning/REQUIREMENTS.md` traces execution requirements back to the refactor change,
- `.planning/ROADMAP.md` slices the validated change into executable phases,
- `*-CONTEXT.md` files clarify how one fixed phase should be executed without redefining scope.
- every `.planning/` artifact MUST cite the source OpenSpec change path and relevant workstream/task ranges,
- if the source change evolves materially, the derivative `.planning/` artifacts MUST be refreshed before new planning/execution continues.

Why:

- large brownfield refactors need both governance and execution structure,
- but they should not accumulate two competing sources of truth.

Alternative considered:

- Let GSD planning define scope first and backfill OpenSpec later.
  Rejected because it increases drift risk for contract-heavy refactor work.

### Decision 2: Use a seam-first, contract-freeze refactor strategy

The first phase of the program will explicitly freeze current public behavior before extracting internals.

Frozen early-phase surfaces include:

- `stage2_ab.pipeline` strict schema behavior,
- Stage-2 clean-prefix / triage / dead-anchor behavior,
- stable trainer metric key families,
- inference JSONL / trace / summary artifacts,
- detection-eval output artifacts,
- run metadata and resolved-config provenance files.

Why:

- The repo already has solid behavioral tests for several of these surfaces.
- The largest risk in a broad refactor is accidental semantic drift hidden behind file movement.

Alternative considered:

- Refactor each large file freely and rely on broad tests afterward.
  Rejected because the runtime-critical surfaces are too entangled for that to be safe.

### Decision 3: Continue the `stage2_two_channel/` package split before attacking the older rollout-aligned trainer

The current two-channel trainer already has a promising composition boundary:

- `scheduler.py` for deterministic channel scheduling,
- `executors.py` for step-budgeted execution,
- `__init__.py` wrapper for import and monkeypatch compatibility,
- and `stage2_two_channel.py` as the assembly layer.

The next internal seams should be:

- `channel_b_target_builder` for anchor/explorer rollout interpretation, clean-prefix reconstruction, FN tail injection, and dead-anchor suppression-target construction,
- `objective_runner` for channel-specific `TeacherForcingContext` assembly and metric projection,
- `types` for the payload passed from batch construction to loss execution.

Compatibility surfaces to inventory before extraction include:

- `src/trainers/stage2_ab_training.py`
- `src/trainers/stage2_ab/__init__.py`
- `src/trainers/stage2_ab/executors.py`
- `src/trainers/stage2_ab/scheduler.py`
- `src/trainers/stage2_two_channel/__init__.py`

Why this first:

- The package layout already supports incremental extraction.
- The active Stage-2 path benefits immediately.
- This reduces the amount of logic that still depends directly on `RolloutMatchingSFTTrainer`.

Alternative considered:

- Start with the older `stage2_rollout_aligned.py` because it is larger.
  Rejected as phase 1 because the two-channel path already contains the lower-risk extraction footholds.

### Decision 4: Extract a shared rollout runtime after the first two-channel seam freeze

Once the two-channel path has explicit interfaces, the next step is to extract a shared rollout runtime layer from `RolloutMatchingSFTTrainer`.

That runtime should own:

- decode request resolution,
- backend selection,
- vLLM local/server lifecycle,
- rollout fanout / chunking / request-capping,
- rollout-model synchronization,
- and shared rollout-response adaptation.

Trainer-facing adapter methods such as `_rollout_many` should remain in place initially as compatibility delegates.

Why:

- Both Stage-2 trainers depend on rollout execution behavior.
- The current rollout-aligned trainer is the highest-coupling hotspot in the repo.

Alternative considered:

- Keep rollout logic trainer-local and only split target construction.
  Rejected because it leaves the largest shared runtime coupling untouched.

### Decision 5: Consolidate bootstrap authority in `sft.py` around typed config, not duplicated defaults

`src/sft.py` should remain the main entrypoint, but not continue as the long-term owner of duplicated policy.

The refactor should move toward:

- typed config as the normative authored source for Stage-2 pipeline shape,
- dedicated bootstrap helpers for packing, trainer setup, and provenance,
- and `sft.py` as a composition/orchestration root rather than a policy duplication point.

For the completion pass in this change, that means centralizing runtime manifest projection behind one helper layer while acknowledging that some variant-default interpretation still lives in bootstrap code.

Why:

- `_build_pipeline_manifest` currently overlaps with `src/config/schema.py`.
- Startup-path complexity makes every trainer refactor harder than it needs to be.

Alternative considered:

- Keep `sft.py` monolithic because it is "just the entrypoint."
  Rejected because too much stable policy now lives there.

### Decision 6: Defer inference and evaluation decomposition until the training/runtime seams are stable

`src/infer/engine.py` and `src/eval/detection.py` are worth modularizing, but they are lower priority than the trainer/runtime surfaces.

Planned later seams:

- `src/infer/backends/{hf,vllm_local,vllm_server}.py`
- `src/infer/artifacts.py`
- `src/eval/artifacts.py`
- `src/eval/orchestration.py`

A deeper `src/eval/detection/{ingest,coco,f1ish,vis}.py` split remains a valid follow-on direction, but is not required for the completion bar of this change.

Protected public API boundaries during this work include:

- `src.infer.engine` imports consumed by `src/infer/pipeline.py`
- `src.eval.detection` imports consumed by:
  - `src/infer/pipeline.py`
  - `src/callbacks/detection_eval.py`
  - `scripts/evaluate_detection.py`

Compatibility-preserving exports or shims are acceptable while internals move, but the boundary must be treated as deliberate rather than accidental.

Why later:

- They are less central to the fast path of Stage-2 research changes.
- Current behavior is guarded by targeted parity tests.

Alternative considered:

- Split infer/eval first because they appear more self-contained.
  Rejected because that would leave the most harmful trainer coupling untouched.

### Decision 7: Require explicit follow-on spec deltas for stable contract changes discovered during the refactor

This change authorizes internal decomposition work. It does not authorize silent changes to:

- Stage-2 training semantics,
- stable metrics,
- artifact shapes,
- inference/evaluation contracts,
- or geometry/prompt invariants.

If implementation reveals that one of those contracts must change, that slice must add a follow-on OpenSpec delta before landing.

Why:

- It keeps architectural progress and behavioral governance separate.
- It preserves trust in the existing test and artifact surfaces during a large engineering program.

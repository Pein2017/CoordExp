---
title: Codebase Refactoring Program Kickoff
date: 2026-06-17
status: discussion-decisions
scope: docs-and-hygiene-gate
---

# Codebase Refactoring Program Kickoff

This note records the first durable decisions from the refactoring-program
`grill-me-with-docs` loop. It is not an implementation report for code motion.

## Decision: Use A Formal Refactoring Worktree

The large cleanup starts from the formal worktree
`/data/CoordExp/.worktrees/codebase-refactoring-program` on branch
`codex/codebase-refactoring-program`.

## Rationale

The main checkout is the user's active research tree. Keeping the refactoring
program isolated protects current launch prep and prevents speculative cleanup
from leaking into active training work.

## Consequence

- The old informal `upcoming-huge-refac` name is retired.
- Worktree branches remain idea containers first, not wholesale merge units by
  default.
- If performance or mechanism evidence is promising, a worktree may be merged
  or promoted into `main`, but the promoted surface should be curated rather
  than importing accidental branch scaffolding.
- Future promoted slices should include docs, lifecycle labels, verification
  evidence, and an explicit list of files intentionally not promoted.

## Decision: Protect Current Standard-SFT Launch Prep

The current standard-SFT launch prep for sorted/random ordering with object
closure and bbox closure is active research work, not stale config fan-out.

## Rationale

This work belongs to the simple labels-only CE lane: standard teacher-forcing
first, with optional geometry/soft-CE losses as explicit additions. It should
not be confused with recursive-detection / ET-RMP, Stage-2 rollout correction,
or legacy compatibility merely because it touches detection sequence code.

## Consequence

- Lifecycle labels must distinguish simple standard SFT from typed
  teacher-forcing research and preserved comparator routes.
- Cleanup should not delete, archive, or demote the object/bbox-closure configs
  while they are the active launch path.
- Sorting/randomization and object/bbox closure should be documented as
  data/template/config semantics, not as a new trainer family.

## Decision: Preserve Recursive-Detection / ET-RMP

Recursive-detection / ET-RMP is preserved as comparator and ablation lineage.
It is not a deletion target for the refactoring program.

## Rationale

This family remains useful for interpreting prior diagnostics, historical
checkpoints, and comparison baselines. Removing it would collapse research
contrast rather than merely reduce codebase bloat.

## Consequence

- It receives lifecycle status `preserved-comparator`.
- Current infer/eval comparator lineage is routed through
  `configs/infer/recursive_detection_ce`.
- Archived Stage-1 recursive-detection training roots remain historical
  evidence.
- New default Stage-1 SFT work should not be placed under this family unless it
  is explicitly a comparator experiment.

## Decision: Start With A Lifecycle Registry And Report-Only Gate

The first executable slice is a lifecycle registry plus a report-only hygiene
checker.

## Rationale

The broad audits agree that the root problem is unlabeled lifecycle state.
Starting with a registry makes active, preserved, historical, retired, and
unclassified surfaces visible before code is moved or deleted.

## Consequence

- The initial registry lives at
  `docs/architecture/proposals/2026-06-17-refactoring-program/lifecycle_registry.yaml`.
- The report-only checker lives in the independent `repo_lifecycle/` module,
  outside both `src/` and `scripts/`.
- The current command is
  `python -m repo_lifecycle.report_lifecycle_registry`.
- Hard errors are malformed registry entries and missing registered paths.
- Warnings remain owner-review debt until analysis/script/config families and
  active-doc `compact_full` references are classified.

## Decision: Keep Lifecycle Types Small

The lifecycle system should use a small fixed set of statuses and resist
creating many fine-grained lifecycle types/classes.

## Rationale

The registry is meant to reduce navigation cost. If every research nuance gets
its own status, the lifecycle system becomes another taxonomy that agents have
to decode.

## Consequence

- Prefer the existing status set: `core`, `active-research`,
  `preserved-comparator`, `compatibility`, `historical-evidence`, `retired`,
  and `needs-classification`.
- Add a new status only if an existing status would systematically misroute
  multiple real surfaces.
- `needs-classification` remains temporary debt, not a comfortable long-term
  state.

## Decision: Treat Some Analysis Scripts As Non-Durable

Some analysis scripts/configs are expected to be temporary study scaffolding,
not durable reusable tools.

## Rationale

Research diagnosis often needs one-off harnesses tied to a checkpoint, machine,
artifact root, or hypothesis. Once the evidence and commands are preserved in
`progress/` or artifacts, keeping every harness as an active-looking tool
increases navigation cost.

## Consequence

- Analysis cleanup should classify each family before deleting or archiving it.
- Reusable analysis CLIs can stay active, but one-off study scaffolding should
  be archived or retired once evidence is preserved.
- `needs-classification` should shrink over time for `src/analysis`,
  `scripts/analysis`, and `configs/analysis`.

## Decision: Split SFT Into Distinct Lanes Before Code Motion

The refactoring program should distinguish:

1. Standard SFT.
2. Research-wise teacher-forcing objective.
3. Stage-2 rollout correction.

## Rationale

The current code and docs overload "teacher forcing". Standard SFT is teacher
forcing in the generic ML sense, but the current `objective.id:
teacher_forcing` surface is more specific: token-role tracing, valid sets,
branch state, force/weight policy, and exact label/logit positions. Stage-2 is
also not merely SFT with rollout; it belongs to the self-trajectory rollout
family with its own artifact and pipeline contracts.

## Consequence

- Use **Standard SFT** for the simple labels-only CE lane, with optional
  geometry/soft-CE auxiliary losses.
- Use **Research-Wise Teacher-Forcing Objective** for the current typed target
  IR / fine-grained token supervision concept. Treat the current
  `objective.id: teacher_forcing` key as a rename/migration candidate.
- Use **Stage-2 Rollout Correction** for rollout/self-trajectory training and
  pipeline-declared loss composition.
- Treat sequence/template semantics as dataset/preprocessing-side config shared
  by lanes, not as objective ownership.
- Migrate away from generic `custom.*` for durable sequence controls.
- Keep the efficient packed-forward path broad and general. Research-wise
  teacher-forcing may reject packing only until exact atom-position remapping is
  implemented and tested; Stage-1 eventually needs packed support.
- Recursive-detection / ET-RMP remains preserved comparator lineage. If it can
  reuse the standard packed-forward path, it may do so while staying labeled as
  research/comparator behavior rather than default Standard SFT.

## Decision: Use A Docs-First Planning Sequence Before Implementation

The SFT disentanglement should proceed as:

1. `grill-me-with-docs` context and decision batches.
2. OpenSpec proposals for stable compatibility-sensitive schema/loss/packing
   contracts.
3. A superpower implementation roadmap.
4. Multiple rounds of self-audit/review.
5. Code implementation only after the above converge.

## Rationale

This change touches stable config schema, training behavior, packing alignment,
loss semantics, and research interpretation. Implementing before agreement
would risk making the refactor another source of hidden lifecycle debt.

## Consequence

- Stable behavior docs are not updated as if the refactor is already complete.
- Proposal/progress records may capture decisions now.
- OpenSpec should own contract changes when the target hierarchy is concrete.

## Decision: Use `pipeline.id` And `sample_factory.target_sequence`

The target authored config hierarchy should use `pipeline.id` as the training
family selector and `sample_factory.target_sequence` as the universal
assistant/label sequence construction owner.

## Rationale

`surface.id` exists in the current shadow architecture, but it is too abstract
as durable public config vocabulary. `pipeline.id` better names the executable
training family. The former `custom.*` sequence controls are not objective
semantics; they are preprocessing/materialization controls that combine dataset
rows, prompt policy, object ordering, field order, detection template, bbox
format, coordinate surface, target text, and sidecar/token-row metadata.

## Consequence

- Use `pipeline.id: stage1_standard_sft` for the high-throughput GT-sequence
  SFT lane.
- Use `pipeline.id: stage1_research_teacher_forcing` for fine-grained
  research-wise teacher-forcing work.
- Use `pipeline.id: stage2_rollout_correction` for rollout/self-trajectory
  training.
- Do not support `custom.trainer_variant` as a backward-compatible selector for
  `pipeline.id` in the target hierarchy; active configs that author it should
  fail fast after the migration.
- Keep `surface` as docs/lifecycle vocabulary, not the preferred authored
  config field. Rename/refactor the shadow `src/training/surfaces.py` concept
  toward pipeline vocabulary rather than preserving a long-term surface layer.
- Use `sample_factory.id: detection_sequence` for row-to-trainable-example
  materialization for the current object/bbox task family. Do not make this
  phase a broad migration away from `detection`; the term is already clear and
  deeply embedded in code, configs, tests, docs, and artifacts.
- Use `sample_factory.target_sequence` for task family, object ordering, field
  order, bbox format, coordinate surface, and strict parsing.
- Keep `detection_template.id` as the stable template identity in this phase;
  do not move template identity under `sample_factory.target_sequence`.
- Use `target_sequence.task_family: detection` for the current object/bbox
  target-sequence task until a better abstraction earns explicit migration.
- Candidate umbrella terms such as `grounding` remain future options, not a
  Phase 1.5 requirement.
- Treat durable `custom.*` authoring for these controls as deprecated. Any
  temporary sequence-control reader must be explicitly scoped and must fail
  when old and new paths are both authored.
- Use `objective.id: standard_ce` as the public standard CE objective id.
  Treat `token_ce` as implementation/metric vocabulary unless a later OpenSpec
  deliberately promotes it.

## Decision: Prefer `research_teacher_forcing`

The research objective should use `research_` naming, not `typed_` naming.

## Rationale

`typed_teacher_forcing` is technically suggestive but semantically unclear.
`research_teacher_forcing` makes the distinction from ordinary Standard SFT
visible in config and prose.

## Consequence

- Use prose name **Research-Wise Teacher-Forcing Objective**.
- Target future config identity: `objective.id: research_teacher_forcing`.
- Treat current `objective.id: teacher_forcing` as a compatibility/migration
  name, not the desired long-term public name.

## Decision: Use OpenSpec For The SFT Hierarchy Contract

The next durable step should be an OpenSpec change, suggested name
`refactor-sft-pipeline-hierarchy`.

## Rationale

The agreed hierarchy changes stable config schema, compatibility behavior,
packing invariants, objective identity, and artifact/provenance expectations.
Those are compatibility-sensitive contracts, not ordinary implementation
checklist details.

## Consequence

- Scope OpenSpec to stable contracts only: config hierarchy, compatibility
  behavior, packing invariants, loss/objective identities, and
  artifact/provenance expectations.
- New durable configs should use `pipeline.id`, `sample_factory`, and
  `sample_factory.target_sequence`.
- `custom.trainer_variant` should fail fast in the target hierarchy instead of
  acting as a compatibility reader for `pipeline.id`.
- `custom.*` sequence controls become migration-only, with fail-fast behavior
  when old and new paths are both authored.
- Rename the research objective contract toward
  `objective.id: research_teacher_forcing`.
- Define `objective.id: standard_ce` as the public standard CE objective, with
  `token_ce` kept as an implementation/metric-component name unless explicitly
  promoted later.
- Accepted `pipeline.id` values are `stage1_standard_sft`,
  `stage1_research_teacher_forcing`, and `stage2_rollout_correction`.
- Use `sample_factory.id: detection_sequence` for current object/bbox
  target-sequence materialization, with `target_sequence.task_family:
  detection`.
- Keep `detection_template.id` as the stable template identity for this phase.
- Do not require a broad `detection` terminology migration in this OpenSpec;
  broader aliases such as `grounding` need separate justification.
- Treat packing as first-class: Standard SFT owns the simplest high-speed path;
  research teacher forcing requires exact atom-position remapping before packed
  forwards; Stage-2 keeps its rollout-specific post-rollout packing contract.
- Keep Stage-2 a rollout/self-trajectory family, not an SFT-family subtype.
- Produce a superpower implementation roadmap with review/self-audit rounds
  before code edits.

## Decision: OpenSpec Shape For SFT Pipeline Hierarchy

The OpenSpec change `refactor-sft-pipeline-hierarchy` should be a deliberately
breaking change for active repo-owned configs, not a broad compatibility layer.

## Rationale

The goal is to make active training routes obvious and reduce future agent
navigation cost. Keeping `custom.trainer_variant` as a selector alias would
leave two public sources of truth. Moving Stage-2 internals and Stage-1 SFT
hierarchy at the same time would over-widen the blast radius. Keeping
`detection_template.id` stable preserves the already-active template contract
while moving sequence-materialization controls out of generic `custom.*`.

## Consequence

- Active repo-owned configs should migrate with the implementation slice;
  archive/historical configs may remain evidence, but should not be advertised
  as active runnable routes.
- Use a top-level `pipeline.id` mapping, not scalar `pipeline_id`.
- Distinguish top-level training-family `pipeline.id` from the existing
  Stage-2 internal namespace `stage2_rollout_correction.pipeline`.
- Rename/refactor `src/training/surfaces.py` toward
  `src/training/pipeline_registry.py`; keep `src/training/pipelines/` for
  concrete implementations.
- Standard SFT authors `objective.id: standard_ce`, with optional auxiliaries
  beneath it.
- Research-wise teacher forcing authors
  `objective.id: research_teacher_forcing`, with internal weighted terms rather
  than a new public pipeline per tactic.
- Keep `stage2_rollout_correction.pipeline.objective[]` unchanged in this
  OpenSpec except for documenting its relation to top-level `pipeline.id`.
- Resolved provenance and cache fingerprints should include the normalized
  training identity: `pipeline.id`, `objective.id`, `detection_template.id`,
  `sample_factory.id`, target-sequence ordering/field-order, bbox/coord
  surface, packing length, and strict-parse mode.
- Fail when old and new sequence-control paths are both authored, even if
  values match, outside explicit migration tests.
- Keep the OpenSpec focused on stable config/schema/provenance/packing
  invariants; code-motion order belongs in the later superpower roadmap.

## Evidence

- Scope: `docs-and-hygiene-gate`
- Handles:
  - `docs/architecture/proposals/2026-06-17-refactoring-program/REFACTORING_PROGRAM_CHARTER.md`
  - `docs/architecture/proposals/2026-06-17-refactoring-program/lifecycle_registry.yaml`
  - `repo_lifecycle/report_lifecycle_registry.py`
  - `tests/test_lifecycle_registry_report.py`
  - `docs/catalog.yaml`
  - `docs/AGENT_INDEX.md`

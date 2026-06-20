# CoordExp Refactoring Program Charter

Date: 2026-06-17
Status: proposal-scoped initial guidance
Repository: `/data/CoordExp`

Source audits:

- `/data/CoordExp/codebase-analysis-claude.md`
- `/data/CoordExp/codebase-analysis-codex.md`

Program artifacts:

- `docs/architecture/proposals/2026-06-17-refactoring-program/lifecycle_registry.yaml`
- `repo_lifecycle/`
- `repo_lifecycle/report_lifecycle_registry.py`
- `tests/test_lifecycle_registry_report.py`

This charter merges the Claude and Codex codebase-level audits into one practical refactoring-program guide. Claude provides the sharper quantitative diagnosis and cleanup pressure. Codex provides the safety envelope around active template work, geometry correctness, Stage-2 stability, provenance, artifacts, and worktree discipline.

The first principle is: lifecycle state before code motion.

## 1. Purpose And Non-Goals

This document is not another broad audit. It is the starting charter for a codebase-level refactoring program.

The program should:

- reduce agent navigation/token cost,
- make active surfaces obvious,
- isolate speculative research,
- preserve research reproducibility,
- preserve image/geometry alignment and bbox semantics,
- preserve training/eval parity,
- preserve artifact and provenance contracts,
- prevent temporary experiments from silently becoming permanent infrastructure.

The program should not:

- delete active research work without owner review,
- change training/eval behavior silently,
- collapse geometry code for cosmetic reasons,
- break current chat/detection template variant work,
- break current uncommitted standard-SFT launch prep for sorted/random ordering with object and bbox closure,
- delete or archive the preserved recursive-detection / ET-RMP algorithm family,
- prune or wholesale-merge worktrees without promotion evidence and owner review,
- run expensive jobs as a default validation strategy,
- turn historical evidence into current guidance,
- treat all large files as bad or all small files as good.

## 2. Executive Diagnosis

CoordExp is not bloated because it lacks architecture. It is bloated because useful research artifacts became permanent infrastructure by default.

The repository already contains strong intended guidance in:

- `docs/PROJECT_CONTEXT.md`
- `docs/SYSTEM_OVERVIEW.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/catalog.yaml`
- `docs/standards/REPO_HYGIENE.md`
- `openspec/specs/*`

The missing piece is enforcement. No cheap check currently prevents:

- one-off study code from entering importable `src`,
- temporary launch wrappers from looking like stable entrypoints,
- study configs from accumulating beside production configs,
- compatibility shims from living forever,
- stale docs/specs/tests from referencing retired names,
- or worktree ideas from being promoted as unreviewed whole branches instead of
  evidence-backed promoted setups.

The main cleanup target is unlabeled lifecycle state.

The largest bloat mass is `src/analysis`, `scripts/analysis`, and `configs/analysis`. The highest-risk active monoliths are Stage-2 rollout correction, `src/config/schema.py`, and `src/sft.py`. The most urgent correctness/navigation drift is the active `compact_full` to semantic-template migration. The long-term structural issue is the package-level dependency cycle that makes small changes non-local.

## 3. Quantitative Grounding

Snapshot evidence from the 2026-06-17 audits:

| Surface | Size / status | Program signal |
|---|---:|---|
| `src/` | about 394 Python files / 180,291 LOC | broad importable surface |
| `src/analysis/` | about 89 Python files / 62,259 LOC | largest non-core mass, about one third of `src` |
| `src/trainers/` | about 65 Python files / 35,542 LOC | active Stage-2 core, monolith-heavy |
| `src/infer/` | about 17 Python files / 14,323 LOC | active runtime, backend/provenance complexity |
| `src/detection/` | about 24 Python files / 12,963 LOC | active Stage-1/template owner |
| `src/training/` | about 58 Python files / 11,722 LOC | mixed live utilities plus shadow architecture surfaces |
| `src/config/schema.py` | about 4,893 LOC | schema/validation monolith |
| `src/sft.py` | about 4,324 LOC | launch script carrying library-scale concerns |
| `configs/` | 218 YAML files | config-first is right; lifecycle labels are weak |
| `configs/analysis/` | 91 YAML files | study-config mass |
| `scripts/` | about 95 tracked `.py`/`.sh` scripts, 97 tracked files total | stable entrypoints mixed with wrappers |
| `.worktrees/` | 7 linked worktrees | idea containers first; merge/promote only with evidence |

Interpretation:

- `src/analysis` is the biggest likely low-runtime-risk cleanup opportunity, but archive by family only after preserving research evidence.
- `configs/analysis` is a lifecycle problem, not a YAML problem.
- `scripts/analysis` is a stable-entrypoint clarity problem.
- `schema.py`, `sft.py`, and Stage-2 monoliths are active-code decomposition targets, not deletion targets.

## 4. Protected Research Contracts

These surfaces can be refactored, but their behavior and interfaces are protected unless a deliberate docs/OpenSpec decision says otherwise.

### 4.1 Governance And Routing

- `docs/PROJECT_CONTEXT.md`
- `docs/SYSTEM_OVERVIEW.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/AGENT_INDEX.md`
- `docs/catalog.yaml`
- `docs/standards/REPO_HYGIENE.md`
- stable specs under `openspec/specs`

### 4.2 Geometry, Data, And Image Alignment

- `src/datasets/geometry.py`
- `src/datasets/builders/jsonlines.py`
- `src/common/geometry`
- `src/tokens/coord/*`
- `src/tokens/row_offsets.py`
- `docs/data/CONTRACT.md`

Rules:

- preserve bbox semantics,
- preserve coordinate order,
- preserve image-size and geometry alignment,
- preserve pixel/norm1000/token transitions,
- do not route around `src/datasets/geometry.py` for dataset bbox math unless the contract changes.

### 4.3 Detection Templates And Current Template Variant Work

Protected active surfaces:

- `src/detection/template.py`
- `src/detection/template_contracts.py`
- `src/detection/runtime.py`
- `src/detection/dataset.py`
- `src/detection/teacher_forcing/compact_full_policy.py`
- `src/common/detection_sequence.py`
- `src/common/detection_compact_rows.py`
- `src/config/prompts.py`
- `src/training/stage2/rollout_codec.py`

Current strict Stage-1 semantic IDs include:

- `stage1_json_pretty`
- `compact`
- `compact_box_closed`
- `compact_object_box_closed`
- `compact_object_box_closed_lines`

Guardrail:

`compact_full` should not be reintroduced as a new strict Stage-1 public template ID. It may still be legitimate in whitelisted contexts:

- rejection tests,
- archived fixtures,
- compatibility parser paths,
- historical config names,
- Stage-2 rollout-family compatibility.

The current chat/detection template variant work is active and necessary. The program should finish convergence, not delete the churn.

Current uncommitted main-tree work for standard SFT launch prep is also active
and necessary. In particular, sorted/random ordering variants with object
closure and bbox closure should be treated as protected launch work, not as
stale per-run config fan-out. Known active handles include:

- `configs/stage1/profiles/2b/pure_ce_coco80_desc_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent.yaml`
- `configs/stage1/smoke/pure_ce_coco80_desc_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent_tiny.yaml`
- the supporting prompt/template/packing/schema/test edits currently dirty in
  the main worktree.

This work belongs to the simple standard-SFT lane: labels-only CE first, with
ordering and object/bbox closure as data/template/config semantics. Do not
classify it as recursive-detection / ET-RMP, Stage-2 rollout correction, or a
legacy compatibility path merely because it touches detection sequence code.

### 4.4 Stage-1 And Stage-2 Training

Protected active surfaces:

- `src/sft.py`
- `src/config/schema.py`
- `src/training_runtime`
- `src/detection/objective.py`
- `src/detection/loss.py`
- `src/detection/rollin.py`
- `src/training/stage2/assignment.py`
- `src/training/stage2/duplicate_filter.py`
- `src/training/stage2/planners.py`
- `src/trainers/stage2_rollout_correction.py`
- `src/trainers/stage2_rollout_correction_impl.py`
- `src/trainers/stage2_rollout_runtime.py`
- `src/trainers/rollout_correction/target_builder.py`
- `src/trainers/rollout_matching`
- `configs/stage1/detection_teacher_forcing`
- `configs/stage1/profiles`
- `configs/stage1/smoke`
- `configs/stage2/rollout_correction`
- `configs/infer/recursive_detection_ce`

Guardrail:

`rollout_matching.*` is still an active private migration/runtime namespace. Do not retire it until schema/runtime ownership has been deliberately moved.

Recursive-detection / ET-RMP is a preserved research algorithm family and comparator baseline, not a deletion target for this refactoring program. It should be made easier to identify and safer to run, but its algorithmic implementation, infer/config lineage, and historical comparator value must be preserved unless the research owner explicitly reverses this decision.

Preservation does not mean it should become the default new Stage-1 SFT route. The clean current Stage-1 path should still distinguish:

- standard labels-only SFT / CE,
- typed teacher-forcing research objectives,
- preserved recursive-detection / ET-RMP comparator and ablation routes.

### 4.5 SFT Pipeline Lane Model

The refactoring program should separate three active training lanes before code
motion:

1. **Standard SFT**
   - Meaning: ordinary assistant-label teacher forcing with pure CE by default.
   - Optional geometry/soft-CE auxiliaries remain explicit loss additions; they
     do not by themselves make the lane a research teacher-forcing objective.
   - Durable public objective identity should be `objective.id: standard_ce`.
     Treat `token_ce` as an implementation or metric-component name during
     migration, not the long-term authored objective id.
   - The efficient packed-forward path should be broad and general here.
   - Current object/bbox-closure launch prep belongs to this lane.

2. **Research-Wise Teacher-Forcing Objective**
   - Meaning: fine-grained token role/value/span tracing, valid sets, branch
     state, force/weight policy, and exact label/logit positions.
   - This is the clearer name for the current `objective.id: teacher_forcing`
     concept. Active migrated configs should reject the old key rather than
     accepting a rename alias because plain "teacher forcing" is too generic.
   - Packing is not optional long-term for Stage-1. It remains invalid only
     until exact atom-position remapping is implemented and tested.

3. **Stage-2 Rollout Correction**
   - Meaning: rollout/self-trajectory family rather than GT-sequence SFT
     family. It prepares rollout-prefix/correction sequences based on roles,
     assigns weights, and composes losses through pipeline-declared modules.
   - Do not resurrect generic `custom.coord_soft_ce_w1.*` style Stage-2 knobs
     after the pipeline surface exists.

Sequence/template semantics should be treated as dataset/preprocessing-side
configuration shared by these lanes, not as objective ownership. The current
generic `custom.*` surface is a transition state: future configs should promote
object ordering, object field order, bbox format, closure tokens, and token-row
requirements into first-class sequence/data-format configuration while keeping
template identity under the existing `detection_template.id` contract.

Target authored hierarchy:

```yaml
pipeline:
  id: stage1_standard_sft  # or stage1_research_teacher_forcing / stage2_rollout_correction

detection_template:
  id: compact_object_box_closed

sample_factory:
  id: detection_sequence
  target_sequence:
    task_family: detection
    object_ordering: sorted
    object_field_order: desc_first
    bbox_format: xyxy
    coordinate_surface: coord_token
    strict_parse: true

prompt:
  variant: coco_80

objective:
  id: standard_ce  # or research_teacher_forcing for the research-wise lane

token_embeddings_adapter:
  enabled: true
```

Use `pipeline.id` as the authored training-family selector. Keep `surface` as
documentation/lifecycle vocabulary, not the preferred public config field. Do
not support `custom.trainer_variant` as a backward-compatible selector in the
target hierarchy; active configs that author it should fail fast after the
OpenSpec migration. Rename/refactor the current shadow
`src/training/surfaces.py` concept toward pipeline vocabulary instead of
preserving a long-term "surface" implementation layer. The SFT hierarchy
OpenSpec resolves this as immediate migration/delete scope rather than a
long-lived shim. Use `sample_factory` for
universal row-to-example materialization, and `sample_factory.target_sequence`
for assistant/label sequence construction.
Use top-level `token_embeddings_adapter` for compact structural token embedding
setup; after the SFT hierarchy migration, flat `token_rows` and
`custom.token_embeddings_adapter` should not remain active config authoring
paths.
Do not make Phase 1.5 a broad rename from `detection` to a speculative
umbrella term. Existing `detection` names are clear for the current object/bbox
task family and appear throughout code, configs, tests, docs, and artifacts.
The generalization should happen first through neutral structural fields such
as `pipeline`, `sample_factory`, `target_sequence`, and `objective`. Candidate
future umbrella terms such as `grounding` can be revisited only if they reduce
real ambiguity enough to justify compatibility churn. Durable new configs
should not put these controls under generic `custom.*`; any temporary
compatibility for sequence controls must be explicitly scoped by OpenSpec and
must fail if both old and new paths are authored.

Recursive-detection / ET-RMP remains preserved comparator lineage. If parts of
that algorithm can reuse the standard high-throughput packed-forward path, they
may do so, but the resulting surface must remain labeled as research
teacher-forcing/comparator behavior rather than becoming the default standard
SFT path.

### 4.6 Infer, Eval, Artifacts, Provenance

Protected active surfaces:

- `scripts/run_infer.py`
- `scripts/postop_confidence.py`
- `scripts/evaluate_detection.py`
- `scripts/evaluate_proxy_detection_bundle.py`
- `scripts/evaluate_oracle_k.py`
- `scripts/export_coco_submission.py`
- `scripts/materialize_proxy_eval_views.py`
- `scripts/stamp_inference_provenance.py`
- `src/infer/runtime.py`
- `src/infer/pipeline.py`
- `src/infer/backend.py`
- `src/infer/artifacts.py`
- `src/eval/detection.py`
- `src/eval/detection_orchestrator.py`
- `src/eval/artifacts.py`
- `src/bootstrap`
- `src/metrics/events.py`

Guardrail:

Keep `decode_policy_fingerprint`. It is active provenance, not the removed live `decode_policy` knob. Remove only live-looking authored `decode_policy:` keys after verifying consumers.

## 5. Lifecycle Categories

Every nontrivial file, module family, script family, config family, spec, or compatibility alias should be classified.

### Core

Stable active runtime/library/contract surface. Behavior changes need tests and docs/spec review.

Examples:

- geometry,
- config schema,
- Stage-1/Stage-2 training,
- infer/eval artifacts,
- stable CLI entrypoints.

### Active Research

Currently used research implementation or diagnostic tooling. It may be temporary, but it is not stale.

Examples:

- current detection-template variant work,
- active diagnostic configs required for current reruns,
- live worktree idea slices before promotion.

### Preserved Comparator

Research algorithm, config lineage, or diagnostic route that is not the default
new-training path but remains intentionally available for comparison,
interpretation, reruns, or ablations.

Examples:

- recursive-detection CE,
- random-permutation ET-RMP-CE,
- prefix-rollin ET-RMP-CE ablations,
- checkpoint-bound infer configs under `configs/infer/recursive_detection_ce`.

Requirements:

- explicit owner/status,
- documented canonical or historical entrypoints,
- config preflight that distinguishes live checkpoints from historical evidence,
- narrow regression tests for target construction, loss semantics, and artifact compatibility,
- no silent promotion into the default SFT path.

### Compatibility

Old name/import/config/path preserved for artifacts, callers, or staged migration.

Requirements:

- allowed callers,
- forbidden new callers,
- removal condition,
- review date or OpenSpec change ID.

### Temporary

Short-lived local or experimental code. It should live outside durable current docs and should not become importable `src` without promotion.

Temporary is a lifecycle reality, not a failure. Some research scripts and
configs exist to answer one question once; when their evidence is preserved in
`progress/` or artifacts, the scratch implementation does not need to remain a
durable reusable surface.

### Status Discipline

Keep lifecycle statuses intentionally small. Prefer the existing statuses over
inventing finer-grained types/classes for every research situation. Add a new
status only if an existing one would repeatedly misroute real surfaces; too many
lifecycle labels would become another navigation problem.

### Historical Evidence

Evidence or rerun material preserved for research provenance. It should usually live in `progress/`, archive folders, or proposal/history docs, not as an active-looking production route.

### Retired

No new use. Should fail fast if invoked. Remaining references should be docs/history/archive/rejection tests only.

## 6. Hotspots And Recommended Disposition

### 6.1 `src/analysis`

Disposition: classify, then quarantine/archive by family.

Why:

- about 62k LOC,
- biggest token/navigation cost,
- audit scans found no core runtime/training/infer/eval importers outside analysis/test/script surfaces,
- likely contains both valuable research helpers and completed-study residue.
- some files are likely temporary study scaffolding, not reusable tools.

Do first:

- build a family inventory,
- mark active/reusable/historical/delete-candidate,
- explicitly mark one-off scratch scripts as non-durable when their evidence is
  already preserved elsewhere,
- preserve artifacts and progress links before moves.

Do not:

- delete the whole tree,
- delete reusable helpers before checking dependents,
- lose launch commands or metrics context.

### 6.2 `configs/analysis`

Disposition: classify as config evidence or active tool input.

Why:

- 91 YAML files,
- many likely one-off study configs,
- config-first is correct, but historical study configs should not look like production launch surfaces.

Do first:

- separate active rerunnable configs from historical artifact pointers,
- move completed-study config context into `progress/` where possible.

### 6.3 `scripts/analysis`

Disposition: retain tested reusable CLIs; archive hardcoded wrappers.

Why:

- one-off `run_*` and `launch_*_tmux.sh` wrappers encode machine/checkpoint/artifact assumptions,
- stable entrypoints become harder to identify.
- many wrappers are allowed to be temporary and should not be promoted merely
  because they once produced useful evidence.

Do first:

- preserve exact commands and artifact roots in progress notes,
- archive launch recipes that are not maintained entrypoints.

### 6.4 Stage-2 Rollout Correction Monoliths

Disposition: active core; refactor behind stable facades.

Paths:

- `src/trainers/stage2_rollout_correction_impl.py`
- `src/trainers/stage2_rollout_runtime.py`
- `src/trainers/rollout_correction/target_builder.py`

Do first:

- extract lower-risk internal concerns after lifecycle gates exist,
- keep `stage2_rollout_correction` public identity stable,
- split `target_builder.py` last with golden IR fixtures.

### 6.5 `src/config/schema.py`

Disposition: active core; split by domain without changing semantics.

Do first:

- preserve strict unknown-key behavior,
- preserve error semantics where tests depend on them,
- keep import compatibility.

### 6.6 `src/sft.py`

Disposition: active launch path; extract library-scale helpers.

Candidate extractions:

- preflight,
- fingerprinting,
- manifest writing,
- template metadata,
- launch/runtime payload assembly.

Guardrail:

Preserve byte-identical fingerprints where artifacts/caches depend on them.

### 6.7 Shadow Or Designed-But-Not-Converged Architecture

Disposition: promote or retire, but do not let it remain parallel indefinitely.

Review carefully:

- `src/training/surfaces.py`
- `src/training/pipelines`
- `src/training/templates`
- `src/training/observability`
- `src/training/sidecars`

Decision:

- either make this the production launch owner,
- or retire/shrink shadow descriptors after verifying callers.

### 6.8 Recursive-Detection / ET-RMP

Disposition: preserve as a comparator/research-baseline family; clarify lifecycle, entrypoints, and checks.

Paths:

- `src/detection/objective.py`
- `src/detection/loss.py`
- `src/detection/rollin.py`
- `configs/infer/recursive_detection_ce`
- archived Stage-1 recursive-detection config roots under `configs/archive/detection_scene_clean_break/stage1/`
- Stage-1 objective/spec/docs that define ET-RMP comparator behavior.

Why:

- ET-RMP and recursive-detection CE remain research-relevant comparator and ablation mechanisms.
- Prior diagnostics and benchmark notes still depend on this family for interpretation.
- Deleting it would remove useful research contrast, not just bloat.

Do first:

- label it as `Preserved Comparator` in the lifecycle registry,
- document canonical current infer/config handles separately from historical archived handles,
- add or keep targeted tests for recursive target construction, prefix-rollin behavior, loss normalization, and strict artifact parsing,
- add config/checkpoint preflight so dead-checkpoint leaves fail clearly or are labeled historical,
- keep new standard SFT configs out of this family unless they explicitly compare against ET-RMP.

Do not:

- delete the implementation as part of general cleanup,
- rename away ET-RMP lineage where it is needed to interpret checkpoints,
- let preserved comparator status make it the default place for new teacher-forcing experiments.

### 6.9 Stale And Misleading Names

Disposition: low-risk clarity cleanup after verification.

Candidates:

- `tests/test_stage2_ab_*`
- `src/infer/backend_sync.py`
- `scripts/tools/test_coord_tokenization.py`
- `src/eval/orchestration.py` if confirmed orphaned,
- `configs/stage1/teacher_forcing` if superseded.

### 6.10 Package-Level Import Cycles

Disposition: structural refactor after lifecycle cleanup.

Known cycles:

- `detection <-> training`,
- `eval <-> infer`,
- `metrics <-> trainers`,
- high-fan-in `src/common`.

Do first:

- measure targeted cycle,
- cut one reciprocal edge at a time,
- do not demand whole-repo acyclicity in one phase.

## 7. Active Template Migration Gate

This is the first correctness/navigation phase.

Known drift:

- stable Stage-1 specs still reference `compact_full_support2.yaml` and `detection_template.id: compact_full`,
- current configs and catalog route toward `compact_support2.yaml` and `compact_tiny.yaml`,
- live code validates semantic IDs such as `compact`,
- some smoke tests and scripts still point to `compact_full_*`,
- some active tests still call `get_detection_template("compact_full")`,
- current standard-SFT launch prep is adding sorted/random object-closure and bbox-closure semantics in the main worktree,
- prefix-rollin schema/spec/runtime support is not fully converged,
- `configs/infer/recursive_detection_ce` filenames carry historical `compact_full` lineage while the parser contract moves toward `compact`.

Completion criteria:

- stable specs, docs, configs, tests, and scripts agree on strict current template IDs,
- `compact_full` appears only in whitelisted compatibility/history contexts,
- sorted/random object-closure and bbox-closure standard-SFT launch configs remain protected and clearly labeled as current Stage-1 SFT work,
- prefix-rollin contract is decided and tested,
- recursive-detection / ET-RMP config naming is documented as preserved comparator lineage, with any semantic-template migration handled through explicit compatibility notes rather than deletion.

Suggested gate:

```bash
rg -n "compact_full_support2|compact_full_tiny|detection_template.id: compact_full|compact_detection_sequence" docs openspec/specs scripts tests configs
python -m pytest tests/test_detection_template_registry.py tests/test_detection_template_variants.py tests/test_detection_training_config_contract.py tests/test_training_architecture_tiny_smoke.py tests/test_infer_compact_full_policy_contract.py -q
```

Important caveat:

The goal is not `rg compact_full -> 0` across the whole repo. The goal is no unapproved `compact_full` in active Stage-1 strict template IDs or active config authoring, with explicit whitelists for rejection tests, archived fixtures, parser compatibility, and Stage-2 rollout-family compatibility.

## 8. Phased Roadmap

### Phase 0: Adopt The Charter

Goal: align on vocabulary and protected surfaces.

Actions:

1. Treat this proposal as the initial program guidance.
2. Decide whether to promote it later into canonical docs.
3. Choose the lifecycle registry location.
4. Do not delete code in this phase.

Exit gate:

- user accepts lifecycle categories and protected-surface list.

### Phase 1: Finish Template Convergence

Goal: remove current contradiction between stable specs/docs/tests and template code.

Actions:

1. Sync Stage-1 template specs/docs.
2. Update stale smoke tests and analysis launchers.
3. Preserve and document current simple standard-SFT sorted/random object-closure and bbox-closure launch prep.
4. Resolve prefix-rollin compact-family support.
5. Document recursive-detection / ET-RMP preservation status and config lineage.
6. Add stale-template grep/test gates.

Risk: medium.

### Phase 1.5: Specify The SFT Lane Split

Goal: turn the SFT lane model into an OpenSpec proposal before implementation.
Suggested change name: `refactor-sft-pipeline-hierarchy`.

Actions:

1. Define the target config hierarchy for sequence/data-format controls outside
   `custom.*`.
2. Adopt `pipeline.id` as the authored training-family selector and reserve
   `surface` for docs/lifecycle language.
   `custom.trainer_variant` should be rejected rather than used as a
   compatibility reader in the target hierarchy.
3. Adopt `sample_factory.id: detection_sequence` plus
   `sample_factory.target_sequence` as the universal sequence-construction
   owner for task family, ordering, field order, bbox format, coordinate
   surface, and strict parsing.
   Do not require a broad `detection` terminology migration in this phase.
4. Keep `detection_template.id` as the stable template identity for this phase;
   do not move it under `sample_factory.target_sequence`.
5. Define `objective.id: standard_ce` as the public standard CE objective
   identity, with `token_ce` retained only as an implementation/metric-component
   term unless the OpenSpec explicitly proves otherwise.
6. Define the fail-fast migration path from generic
   `objective.id: teacher_forcing` to `objective.id:
   research_teacher_forcing`.
7. Rename/refactor the current shadow `src/training/surfaces.py` vocabulary
   toward pipeline naming in the first implementation slice, rather than
   leaving `surface` as a long-term implementation concept.
8. Define the Stage-1 packing/remap contract required before research-wise
   teacher-forcing can use packed forwards.
9. Define how Stage-2 rollout-correction reuses sequence/teacher-forcing
   machinery without becoming an SFT-family surface.
10. Produce a superpower implementation roadmap with self-audit/review rounds
   before code implementation.

Risk: medium; mostly schema, naming, packing-alignment, and compatibility risk.

OpenSpec shape constraints:

- Treat `refactor-sft-pipeline-hierarchy` as a deliberately breaking change for
  active repo-owned configs. Active configs should migrate in the same
  implementation slice; archive/historical configs may remain as evidence but
  should not be advertised as active runnable routes.
- Use a top-level mapping shape, `pipeline.id`, rather than a scalar
  `pipeline_id`.
- Explicitly distinguish top-level training-family `pipeline.id` from the
  existing Stage-2 internal namespace `stage2_rollout_correction.pipeline`.
- Rename/refactor `src/training/surfaces.py` toward
  `src/training/pipeline_registry.py`; keep `src/training/pipelines/` for
  concrete pipeline implementations. Do not keep a long-lived compatibility
  shim.
- Standard SFT should author `objective.id: standard_ce`, with optional
  auxiliary terms below it rather than separate public pipeline ids.
- Research-wise teacher forcing should author
  `objective.id: research_teacher_forcing` with internal weighted terms, rather
  than creating a new public pipeline for every research tactic.
- Active migrated configs should reject `objective.id: teacher_forcing`; keep
  the old id only in historical/archive evidence or explicit rejection tests.
- Compact structural token embedding setup should use top-level
  `token_embeddings_adapter`; flat `token_rows` and
  `custom.token_embeddings_adapter` should be rejected after migration.
- Active config filenames should expose semantic axes such as pipeline,
  objective, template, ordering, prompt, and packing, rather than ambiguous
  numbers or version-like labels.
- Keep Stage-2 `stage2_rollout_correction.pipeline.objective[]` unchanged in
  this OpenSpec except for documenting its relationship to top-level
  `pipeline.id`.
- Resolved provenance and cache fingerprints should include the normalized
  training identity: `pipeline.id`, `objective.id`, `detection_template.id`,
  `sample_factory.id`, target-sequence ordering/field-order, bbox/coord
  surface, packing length, and strict-parse mode.
- Fail fast when old and new sequence-control paths are both authored, even if
  their values match, outside explicit migration tests.
- Keep this OpenSpec focused on config/schema/provenance/packing invariants.
  Code-motion order and decomposition tasks belong in the later superpower
  implementation roadmap.

### Phase 2: Add Lifecycle Registry And Hygiene Gates

Goal: prevent new bloat before moving old bloat.

Actions:

1. Register lifecycle states for analysis/script/config/spec/compat surfaces.
2. Start from the proposal-scoped lifecycle registry at `docs/architecture/proposals/2026-06-17-refactoring-program/lifecycle_registry.yaml`.
3. Add cheap checks for missing paths, retired keys, lifecycle metadata, study specs, and compat expiry.
4. Start as report-only or xfail if needed, then make blocking once initial violations are resolved.

Risk: low to medium.

### Phase 3: Classify And Quarantine Analysis Surfaces

Goal: reduce the largest token/navigation burden.

Actions:

1. Inventory `src/analysis`, `scripts/analysis`, and `configs/analysis` by family.
2. Preserve evidence for historical studies.
3. Archive one high-confidence family as a pilot.
4. Keep only maintained reusable analysis tools in active surfaces.

Risk: medium, mostly research reproducibility rather than runtime behavior.

### Phase 4: Rename Misleading Surfaces

Goal: improve agent navigation without behavior changes.

Actions:

1. Rename tests that use retired mechanism names for live behavior.
2. Rename backend/sync modules to reflect actual responsibility.
3. Rename CLI files that look like pytest files.
4. Archive confirmed orphan eval orchestration code.

Risk: low.

### Phase 5: Consolidate Config Lifecycle

Goal: reduce runnable-looking historical configs.

Actions:

1. Classify `configs/bench`.
2. Consolidate repeated infer/postop/eval leaves.
3. Label dead-checkpoint configs historical or make preflight fail clearly.
4. Convert completed-study configs into progress/artifact references.
5. Keep `configs/infer/recursive_detection_ce` as preserved comparator lineage; consolidate repeated launch knobs only when the canonical ET-RMP comparison handles remain obvious.

Risk: medium.

### Phase 6: Retire Mechanical Shims

Goal: collapse obvious shallow compatibility layers.

Actions:

1. Codemod `src/coord_tokens` callers to canonical token modules.
2. Move trainer metric shim callers to canonical or trainer-private owners.
3. Collapse local geometry duplicate helpers only after geometry tests.

Risk: low to medium.

### Phase 7: Split Active Monoliths

Goal: improve locality behind stable interfaces.

Order:

1. Split `src/config/schema.py` by domain.
2. Extract `src/sft.py` preflight/fingerprint/manifest clusters.
3. Decompose `src/trainers/stage2_rollout_correction_impl.py`.
4. Decompose `src/trainers/stage2_rollout_runtime.py`.
5. Split `src/trainers/rollout_correction/target_builder.py` last.

Risk: medium to high.

### Phase 8: Cut Package Cycles

Goal: reduce structural navigation cost.

Order:

1. `eval -> infer`: share artifact/decode-provenance structures so eval does not depend on infer runtime.
2. `detection <-> training`: move template contracts and detection IR to a neutral owner.
3. `metrics <-> trainers`: keep generic metrics in `src/metrics`; trainer-specific projections stay private.

Risk: high.

## 9. Prioritized Roadmap

| Priority | Work | Impact | Risk | Why this order |
|---:|---|---|---|---|
| 1 | Finish detection-template migration | Very high | Medium | active correctness/docs/spec risk |
| 2 | Add lifecycle registry and hygiene gates | Very high | Low/medium | prevents new bloat before cleanup |
| 3 | Classify/quarantine analysis surfaces | Very high | Medium | largest token/navigation win |
| 4 | Rename misleading surfaces | High | Low | improves navigation with low behavioral risk |
| 5 | Label and guard recursive-detection / ET-RMP comparator lineage | High | Medium | prevents accidental deletion while clarifying non-default status |
| 6 | Consolidate config bundles | High | Medium | reduces misleading runnable configs |
| 7 | Retire mechanical shims | Medium/high | Low/medium | cheap simplification after gates |
| 8 | Split `schema.py` and `sft.py` | High | Medium/high | active monoliths, needs stable tests |
| 9 | Decompose Stage-2 internals | High | High | important but research-semantics sensitive |
| 10 | Split `target_builder.py` | Medium | High | highest loss-semantics risk, do last |
| 11 | Cut package cycles | High | High | structural payoff, needs prior stabilization |

## 10. Verification And Hygiene Gates

Use narrow gates first. Broaden only when shared contracts move.

### Lifecycle Registry Gate

```bash
python -m repo_lifecycle.report_lifecycle_registry
python -m pytest tests/test_lifecycle_registry_report.py -q
```

This gate is report-only while the first classifications settle. Registry
syntax, duplicate ids, unknown statuses, and missing paths are errors.
Unclassified analysis/script/config roots and active-doc `compact_full` mentions
are warnings until owner review resolves or whitelists them. Use
`python -m repo_lifecycle.report_lifecycle_registry --strict` only after those
warnings are intentionally burned down.

### Template Gate

```bash
rg -n "compact_full_support2|compact_full_tiny|detection_template.id: compact_full|compact_detection_sequence" docs openspec/specs scripts tests configs
python -m pytest tests/test_detection_template_registry.py tests/test_detection_template_variants.py tests/test_detection_training_config_contract.py tests/test_training_architecture_tiny_smoke.py tests/test_infer_compact_full_policy_contract.py -q
```

### Geometry Gate

```bash
python -m pytest tests/test_coord_geometry_invariants.py tests/test_object_geometry_extract.py tests/test_eval_detection_records.py -q
```

### Config Gate

```bash
python -m pytest tests/test_training_config_strict_unknown_keys.py tests/test_detection_training_config_contract.py tests/test_training_surface_resolver.py tests/test_objective_profile_resolution.py -q
```

Add checks for:

- active config roots parse,
- missing checkpoints are labeled historical or fail preflight,
- retired config keys fail fast.

### Recursive-Detection / ET-RMP Comparator Gate

```bash
rg -n "recursive_detection_ce|prefix_rollin_et_rmp_ce|random_permutation_et_rmp_ce|ET-RMP" src configs docs openspec/specs tests --glob '!docs/history/**'
python -m pytest tests/test_recursive_detection_ce_target_builder.py tests/test_recursive_detection_ce_loss_adapter.py tests/test_prefix_rollin_sampler.py tests/test_prefix_rollin_dataset_alignment.py -q
```

Use this gate before moving, renaming, or consolidating recursive-detection /
ET-RMP files. The expected outcome is not zero references. The expected outcome
is that live references are classified as preserved comparator, compatibility,
or documented history, and that target/loss/alignment behavior remains intact.

### Analysis Archive Gate

```bash
rg -n "from src.analysis|import src.analysis" src --glob '!src/analysis/**'
```

Also require:

- progress/artifact pointer preserved,
- exact launch command preserved when rerunnable,
- analysis tests run when moved family had tests.

### Docs/Spec Gate

Check that path-shaped references in:

- `docs/catalog.yaml`,
- `docs/IMPLEMENTATION_MAP.md`,
- `docs/SYSTEM_OVERVIEW.md`,
- `docs/training`,
- `docs/eval`,
- relevant `openspec/specs`,

either resolve or are explicitly historical.

### Provenance Gate

Verify that these remain intact when infer/eval/training surfaces move:

- `decode_policy_fingerprint`,
- `resolved_config.json`,
- `effective_runtime.json`,
- rank-0 artifacts,
- Stage-2 policy provenance,
- copied summaries and manifests.

### Stage-2 Gate

```bash
python -m pytest tests/test_stage2_rollout_correction_contract.py tests/test_stage2_rollout_runtime.py tests/test_stage2_assignment_greedy_iou.py tests/test_stage2_duplicate_filter.py tests/test_stage2_supervision_planning_smoke.py tests/test_teacher_forcing_token_ce.py -q
```

Add golden IR fixtures before changing `target_builder.py`.

### Import-Graph Gate

Measure only the targeted cycle being cut:

- `eval -> infer`,
- `detection <-> training`,
- `metrics <-> trainers`.

Do not require the whole repo to become acyclic in one phase.

### Final Hygiene

```bash
git diff --check
```

When staging:

```bash
git diff --cached --check
```

Update docs/specs in the same change whenever behavior, schema, artifact names, metric semantics, entrypoints, or recommended workflows move.

## 11. Worktree Promotion Policy

Treat `.worktrees` as idea containers first, not merge units by default.
However, when a worktree demonstrates promising performance or a valuable
research mechanism, it is acceptable to merge/promote it into `main` and make
the idea part of the permanent setup. The promotion should be evidence-backed
and should still curate the permanent surface instead of importing accidental
branch scaffolding.

Before promoting work from any worktree:

```bash
git cherry -v main <branch>
git diff --name-only main...<branch>
git diff --shortstat main...<branch>
```

Require:

- branch owner/status,
- exact hypothesis,
- performance or mechanism evidence when promotion is research-driven,
- minimal promoted file slice,
- files intentionally not promoted,
- config lifecycle label,
- docs/spec impact decision,
- verification evidence,
- artifact/progress pointer.

Do not prune, merge, or permanently promote worktrees based on this charter
alone. Re-run `git worktree list`, `git cherry`, and owner review at the time
of cleanup or promotion.

## 12. Open Decisions For Research Owner

These decisions should be resolved before large cleanup starts:

Resolved for this charter:

- Recursive-detection / ET-RMP is preserved. Refactoring may clarify ownership,
  lifecycle state, preflight, naming, and docs, but the algorithm family is not
  a deletion/archive target.
- The target authored training-family selector is `pipeline.id`, not
  `surface.id`; `surface` remains docs/lifecycle vocabulary. The implementation
  should migrate away from a long-term `src/training/surfaces.py` "surface"
  resolver and toward pipeline vocabulary.
- `custom.trainer_variant` should not be a compatibility reader for
  `pipeline.id`; target active configs that author it should fail fast.
- The universal preprocessing/materialization owner is `sample_factory`, with
  assistant/label sequence controls under `sample_factory.target_sequence`.
- `detection_template.id` remains the stable template identity for this phase.
  Sequence controls move under `sample_factory.target_sequence`, but template
  identity does not move in the first migration.
- The durable public standard CE objective id is `standard_ce`; `token_ce`
  remains an implementation/metric-component term unless the OpenSpec finds a
  stronger reason to expose it publicly.
- No broad `detection` terminology migration is required for this phase.
  Existing `detection` names remain the clear compatibility vocabulary for the
  current object/bbox task family. The broader architecture should use neutral
  structural fields such as `pipeline`, `sample_factory`, `target_sequence`,
  and `objective`; future `grounding` aliases require explicit OpenSpec
  justification.
- The research-wise teacher-forcing objective should migrate toward
  `objective.id: research_teacher_forcing`; avoid `typed_teacher_forcing`.
  The old `teacher_forcing` id should be rejected for active migrated configs;
  keep it only in historical/archive evidence or explicit rejection tests.
- Durable new configs should avoid authoring sequence/objective-family behavior
  under generic `custom.*`.
- Compact structural token embedding setup should use top-level
  `token_embeddings_adapter`, not flat `token_rows` or
  `custom.token_embeddings_adapter`.

1. Should `src/analysis` move outside importable `src`, or is lifecycle metadata/import gating enough?
2. Which analysis studies are expected to be rerun?
3. Should `vllm.mode: colocate` remain a supported fallback?
4. How long should `compact_full` remain as a private compatibility term?
5. Should prefix-rollin support all semantic compact variants now, or only `compact` until validated?
7. Should study-style OpenSpec specs be demoted retroactively, or only blocked going forward?
8. Are pytest-based hygiene gates acceptable as enforcement?
9. Which linked worktrees are still active research?
10. How aggressively should internal import compatibility be preserved?

## 13. First Concrete Program Slice

Recommended first slice:

1. Use the proposal-scoped lifecycle registry at `docs/architecture/proposals/2026-06-17-refactoring-program/lifecycle_registry.yaml`.
2. Run the report-only lifecycle gate:
   - `python -m repo_lifecycle.report_lifecycle_registry`
   - `python -m pytest tests/test_lifecycle_registry_report.py -q`
3. Use the gate output to track:
   - missing active doc paths,
   - retired template IDs in active specs/tests,
   - study specs under stable OpenSpec,
   - `src/analysis` families without lifecycle state.
4. Finish active template migration inconsistencies.
5. Convert the hygiene check to blocking once the first violations are resolved.

This creates a ratchet: the repo can still carry research complexity, but new complexity must be explicitly labeled.

## Appendix A: Evidence Inventory

Important audit findings to preserve:

- `src/analysis`: about 62k LOC, about one third of `src`, no known core importers from audit scans.
- `configs/analysis`: 91 YAML files, about 42 percent of configs by count.
- active monoliths: `schema.py`, `sft.py`, `stage2_rollout_correction_impl.py`, `stage2_rollout_runtime.py`, `target_builder.py`.
- package cycles: `detection <-> training`, `eval <-> infer`, `metrics <-> trainers`, high-fan-in `common`.
- active template drift: `compact_full` vs semantic `compact*` IDs across specs/docs/tests/configs/code.
- recursive-detection / ET-RMP: preserved comparator lineage, not a general cleanup deletion target.
- stale/misleading names: `test_stage2_ab_*`, `backend_sync.py`, CLI named `test_*`, possible orphan `src/eval/orchestration.py`.
- decode policy distinction: keep `decode_policy_fingerprint`; verify and remove/rename live-looking authored `decode_policy:` keys.
- worktree snapshot: linked worktrees are isolated idea containers; do not merge wholesale.

## Appendix B: Adoption Path

This file is proposal-scoped. If the program is accepted:

1. Keep this full document as proposal/history.
2. Promote the stable vocabulary and gates into `docs/standards/REPO_HYGIENE.md` or a canonical architecture document.
3. Add lifecycle registry metadata in the selected location.
4. Track phase outcomes in `progress/` or a dedicated refactoring program progress note.

## Bottom Line

The cleanup program should not begin with broad deletion. It should begin by finishing the active template migration, adding lifecycle gates, and classifying the analysis/config/script mass. Once lifecycle state is explicit, deletion and refactoring decisions become evidence-backed engineering work rather than risky archaeology.

The largest safe payoff is moving historical analysis code and study configs out of the apparent production surface. The largest active-code payoff is decomposing Stage-2, schema, and launch monoliths behind stable facades. The long-term payoff is enforcement: make the repo reject unlabeled bloat before it becomes architecture.

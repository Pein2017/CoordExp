# CoordExp Refactoring Program Charter

Date: 2026-06-17
Status: proposal-scoped initial guidance
Repository: `/data/CoordExp`

Source audits:

- `/data/CoordExp/codebase-analysis-claude.md`
- `/data/CoordExp/codebase-analysis-codex.md`

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
- prune or wholesale-merge worktrees,
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
- or worktree ideas from being promoted as whole branches instead of extracted slices.

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
| `.worktrees/` | 7 linked worktrees | idea containers, not merge units |

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

### 4.4 Stage-1 And Stage-2 Training

Protected active surfaces:

- `src/sft.py`
- `src/config/schema.py`
- `src/training_runtime`
- `src/training/stage2/assignment.py`
- `src/training/stage2/duplicate_filter.py`
- `src/training/stage2/planners.py`
- `src/trainers/stage2_rollout_correction.py`
- `src/trainers/stage2_rollout_correction_impl.py`
- `src/trainers/stage2_rollout_runtime.py`
- `src/trainers/rollout_correction/target_builder.py`
- `src/trainers/rollout_matching`
- `configs/stage1/detection_teacher_forcing`
- `configs/stage2/rollout_correction`

Guardrail:

`rollout_matching.*` is still an active private migration/runtime namespace. Do not retire it until schema/runtime ownership has been deliberately moved.

### 4.5 Infer, Eval, Artifacts, Provenance

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

### Compatibility

Old name/import/config/path preserved for artifacts, callers, or staged migration.

Requirements:

- allowed callers,
- forbidden new callers,
- removal condition,
- review date or OpenSpec change ID.

### Temporary

Short-lived local or experimental code. It should live outside durable current docs and should not become importable `src` without promotion.

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

Do first:

- build a family inventory,
- mark active/reusable/historical/delete-candidate,
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

### 6.8 Stale And Misleading Names

Disposition: low-risk clarity cleanup after verification.

Candidates:

- `tests/test_stage2_ab_*`
- `src/infer/backend_sync.py`
- `scripts/tools/test_coord_tokenization.py`
- `src/eval/orchestration.py` if confirmed orphaned,
- `configs/stage1/teacher_forcing` if superseded.

### 6.9 Package-Level Import Cycles

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
- prefix-rollin schema/spec/runtime support is not fully converged,
- `configs/infer/recursive_detection_ce` filenames carry historical `compact_full` lineage while the parser contract moves toward `compact`.

Completion criteria:

- stable specs, docs, configs, tests, and scripts agree on strict current template IDs,
- `compact_full` appears only in whitelisted compatibility/history contexts,
- prefix-rollin contract is decided and tested,
- recursive-detection config naming is either updated or documented as historical checkpoint lineage.

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
3. Resolve prefix-rollin compact-family support.
4. Document or rename recursive-detection config lineage.
5. Add stale-template grep/test gates.

Risk: medium.

### Phase 2: Add Lifecycle Registry And Hygiene Gates

Goal: prevent new bloat before moving old bloat.

Actions:

1. Register lifecycle states for analysis/script/config/spec/compat surfaces.
2. Add cheap checks for missing paths, retired keys, lifecycle metadata, study specs, and compat expiry.
3. Start as report-only or xfail if needed, then make blocking once initial violations are resolved.

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
| 5 | Consolidate config bundles | High | Medium | reduces misleading runnable configs |
| 6 | Retire mechanical shims | Medium/high | Low/medium | cheap simplification after gates |
| 7 | Split `schema.py` and `sft.py` | High | Medium/high | active monoliths, needs stable tests |
| 8 | Decompose Stage-2 internals | High | High | important but research-semantics sensitive |
| 9 | Split `target_builder.py` | Medium | High | highest loss-semantics risk, do last |
| 10 | Cut package cycles | High | High | structural payoff, needs prior stabilization |

## 10. Verification And Hygiene Gates

Use narrow gates first. Broaden only when shared contracts move.

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

Treat `.worktrees` as idea containers, not merge units.

Before promoting work from any worktree:

```bash
git cherry -v main <branch>
git diff --name-only main...<branch>
git diff --shortstat main...<branch>
```

Require:

- branch owner/status,
- exact hypothesis,
- minimal promoted file slice,
- files intentionally not promoted,
- config lifecycle label,
- docs/spec impact decision,
- verification evidence,
- artifact/progress pointer.

Do not prune or merge worktrees based on this charter alone. Re-run `git worktree list`, `git cherry`, and owner review at the time of cleanup.

## 12. Open Decisions For Research Owner

These decisions should be resolved before large cleanup starts:

1. Should `src/analysis` move outside importable `src`, or is lifecycle metadata/import gating enough?
2. Which analysis studies are expected to be rerun?
3. Should `src/training/surfaces.py` become the production launch owner, or should shadow surfaces retire?
4. Should `vllm.mode: colocate` remain a supported fallback?
5. How long should `compact_full` remain as a private compatibility term?
6. Should prefix-rollin support all semantic compact variants now, or only `compact` until validated?
7. Should study-style OpenSpec specs be demoted retroactively, or only blocked going forward?
8. Are pytest-based hygiene gates acceptable as enforcement?
9. Which linked worktrees are still active research?
10. How aggressively should internal import compatibility be preserved?

## 13. First Concrete Program Slice

Recommended first slice:

1. Choose lifecycle registry location.
2. Add a report-only or xfail hygiene check for:
   - missing active doc paths,
   - retired template IDs in active specs/tests,
   - study specs under stable OpenSpec,
   - `src/analysis` families without lifecycle state.
3. Finish active template migration inconsistencies.
4. Convert the hygiene check to blocking once the first violations are resolved.

This creates a ratchet: the repo can still carry research complexity, but new complexity must be explicitly labeled.

## Appendix A: Evidence Inventory

Important audit findings to preserve:

- `src/analysis`: about 62k LOC, about one third of `src`, no known core importers from audit scans.
- `configs/analysis`: 91 YAML files, about 42 percent of configs by count.
- active monoliths: `schema.py`, `sft.py`, `stage2_rollout_correction_impl.py`, `stage2_rollout_runtime.py`, `target_builder.py`.
- package cycles: `detection <-> training`, `eval <-> infer`, `metrics <-> trainers`, high-fan-in `common`.
- active template drift: `compact_full` vs semantic `compact*` IDs across specs/docs/tests/configs/code.
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

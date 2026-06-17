# CoordExp Codebase Analysis - Codex Revised

Date: 2026-06-17
Repository: `/data/CoordExp`
Mode: read-only architectural diagnosis, revised after comparison with `/data/CoordExp/codebase-analysis-claude.md`
Output: `/data/CoordExp/codebase-analysis-codex.md`

## Revision Note

This revision intentionally incorporates the strongest parts of the Claude report:

- a quantitative opening frame,
- more decisive archive and owner-review categories,
- a tighter impact/risk/refactor table,
- and enforceable hygiene gates rather than general advice.

It also keeps the strongest parts of the original Codex report:

- caution around the active detection-template variant migration,
- explicit protection for core research-correctness surfaces,
- separation of retired decode-policy knobs from active provenance fields,
- and a conservative worktree promotion rule.

The resulting recommendation is: use Claude's diagnosis as the primary cleanup roadmap, but keep the Codex guardrails below so cleanup does not break active template, geometry, Stage-2, or artifact contracts.

## 0. Quantitative Grounding

These counts frame the repository problem. They were gathered from read-only static inspection and should be treated as approximate but decision-useful.

| Surface | Size / status | Signal |
|---|---:|---|
| `src/` | about 394 Python files / 180,291 LOC | baseline importable surface |
| `src/analysis/` | about 89 Python files / 62,259 LOC | largest non-core mass, about one third of `src` |
| `src/trainers/` | about 65 Python files / 35,542 LOC | active Stage-2 core, but monolith-heavy |
| `src/infer/` | about 17 Python files / 14,323 LOC | active runtime, naming and backend complexity |
| `src/detection/` | about 24 Python files / 12,963 LOC | active Stage-1/detection-template owner |
| `src/training/` | about 58 Python files / 11,722 LOC | mixed live utilities plus shadow architecture surfaces |
| `src/config/schema.py` | about 4,893 LOC | schema/validation monolith |
| `src/sft.py` | about 4,324 LOC | launch script carrying library-scale concerns |
| `configs/` | 218 YAML files | config-first is correct, lifecycle labels are weak |
| `configs/analysis/` | 91 YAML files | large study-config mass |
| `scripts/` | about 95 tracked `.py`/`.sh` scripts, 97 tracked files total | stable entrypoints mixed with one-off wrappers |
| `.worktrees/` | 7 linked worktrees | none should be merged wholesale |

The two biggest facts:

1. `src/analysis/`, `scripts/analysis/`, and `configs/analysis/` are the primary bloat mass and should no longer look like ordinary production infrastructure.
2. The repo already has good governance docs, but weak enforcement. The problem is not absence of rules; it is that no check fails when a rule is broken.

## 1. High-Level Diagnosis

CoordExp grew because research code became permanent infrastructure by default. That does not mean the code is useless. It means useful research artifacts were promoted into `src/`, `scripts/`, and `configs/` without an explicit lifecycle state, owner, expiry condition, or archive path.

The recurring mechanism is:

1. A research question needs real training, inference, or diagnosis code.
2. The fastest safe implementation lands as a module, script, config family, wrapper, or compatibility alias.
3. The experiment produces useful evidence, so agents preserve it.
4. New ideas layer on top of the old surface.
5. No one later asks whether the surface is core, active research, compatibility, historical evidence, or temporary scaffolding.

The repo already says the right things in:

- `docs/PROJECT_CONTEXT.md`
- `docs/SYSTEM_OVERVIEW.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/catalog.yaml`
- `docs/standards/REPO_HYGIENE.md`
- `openspec/specs/*`

But the implementation has not fully converged to those boundaries. The result is a large apparent public surface where stable runtime code, one-off studies, compatibility shims, archived configs, and launch recipes all look similarly alive to agents.

The important nuance: current chat/detection template variant work is active and necessary. Do not classify it as stale just because it is new, renamed, or partially synchronized. The right diagnosis is "finish the migration atomically," not "delete the churn."

## 2. Biggest Hotspots

### 2.1 `src/analysis/`: Prime Bloat Mass

Path: `/data/CoordExp/src/analysis`

Evidence:

- about 62,259 LOC, about 89 Python files,
- larger than `src/trainers`,
- no known core runtime/training/infer/eval importer outside analysis/test/script surfaces from the audit scans,
- closely paired with `scripts/analysis` and `configs/analysis`.

Largest examples:

- `src/analysis/qwen3_vl_instance_binding.py`
- `src/analysis/autoreg_fn_rescue_continuation.py`
- `src/analysis/unmatched_proposal_verifier.py`
- `src/analysis/duplication_collapse_analysis.py`
- `src/analysis/hard_ce_coord_logit_locality.py`

Diagnosis:

This is not all junk. It is valuable research diagnosis. But most of it should not live as ordinary importable `src` production surface. It increases token cost, hides active architecture, and makes historical studies look like maintained runtime code.

Recommendation:

Create a classification pass over every `src/analysis` family:

- maintained reusable analysis library,
- active experiment,
- historical evidence,
- one-off study harness,
- delete/archive candidate.

Then move or fence historical study code. The safest first move is lifecycle metadata and import gates; the highest-impact later move is relocating historical analysis outside ordinary `src`.

### 2.2 `configs/analysis/`: Study Config Accretion

Path: `/data/CoordExp/configs/analysis`

Evidence:

- 91 YAML files, about 42 percent of all YAML configs by count,
- many families are one-off study configs rather than maintained launch surfaces,
- several analysis config families have no durable code owner or are tied to historical reports.

Diagnosis:

The problem is not YAML. Config-first is correct for CoordExp. The problem is that analysis/study YAMLs live beside durable config families without lifecycle labels.

Recommendation:

For each family, classify as:

- active reusable analysis config,
- historical evidence config,
- artifact pointer better represented in `progress/`,
- or deletion/archive candidate.

Do not retain one YAML per debugging attempt unless the config has a live rerun purpose.

### 2.3 `scripts/analysis/`: One-Experiment Wrappers

Path: `/data/CoordExp/scripts/analysis`

Evidence:

- many `run_*` wrappers,
- multiple `launch_*_tmux.sh` launch recipes,
- hardcoded checkpoints, local roots, GPU/session assumptions, and artifact directories in several wrappers.

Examples:

- `scripts/analysis/launch_autoreg_object_rollout_lane_b_tmux.sh`
- `scripts/analysis/prefix_state_transition_tomography/launch_prefix_state_transition_tomography_tmux.sh`
- `scripts/analysis/launch_policy_objective_mechanism_comparison_tmux.sh`
- `scripts/analysis/run_ckpt_pair_confidence_eval.sh`

Diagnosis:

Stable scripts are not the issue. The issue is that one-off operator recipes live in the same broad script tree as reportable entrypoints.

Recommendation:

Keep tested, config-backed analysis CLIs if they have repeat value. Move hardcoded tmux recipes and one-off wrappers to progress notes or archive after preserving exact commands and artifact roots.

### 2.4 Stage-2 Rollout Correction Monoliths

Paths:

- `src/trainers/stage2_rollout_correction_impl.py`
- `src/trainers/stage2_rollout_runtime.py`
- `src/trainers/rollout_correction/target_builder.py`

Evidence:

- `stage2_rollout_correction_impl.py`: about 5,923 LOC,
- `stage2_rollout_runtime.py`: about 4,585 LOC,
- `rollout_correction/target_builder.py`: about 4,382 LOC,
- Stage-2 concerns span target construction, rollout runtime, duplicate/residual planning, metrics, debug dumps, artifact/provenance, and trainer execution.

Diagnosis:

This is active, high-value infrastructure. It is not a deletion target. But it is the highest-risk active monolith cluster in the repo.

Recommendation:

Keep public identity stable:

- `custom.trainer_variant: stage2_rollout_correction`
- `src/trainers/stage2_rollout_correction.py`
- `configs/stage2/rollout_correction`

Then extract internal concerns behind facades:

- rollout runtime/backend bridge,
- target construction,
- duplicate and residual planning,
- metric projection,
- artifact/provenance emission,
- monitor/debug dump plumbing.

### 2.5 `src/config/schema.py`

Path: `/data/CoordExp/src/config/schema.py`

Evidence:

- about 4,893 LOC,
- many dataclasses and strict validation helpers,
- schema, compatibility guards, removed-key checks, and cross-surface policy live together.

Diagnosis:

This is active and important, but oversized. The risk is not just readability; it is that schema drift and compatibility drift become hard to audit.

Recommendation:

Split by domain while preserving current imports and exact strictness:

- base/custom schema,
- detection schema,
- Stage-1 schema,
- Stage-2 schema,
- cache/packing schema,
- infer/eval schema,
- retired-key/compatibility guards.

### 2.6 `src/sft.py`

Path: `/data/CoordExp/src/sft.py`

Evidence:

- about 4,324 LOC,
- launch script plus preflight, fingerprint, manifest, template metadata, cache/packing, checkpoint/provenance, and runtime payload concerns.

Diagnosis:

`src/sft.py` is a script doing library-scale orchestration work. It is active, but too broad.

Recommendation:

Extract preflight, fingerprint, manifest, and launch metadata helpers into owned modules under `src/bootstrap`, `src/training_runtime`, or a similarly documented owner, preserving byte-level fingerprint behavior where caches/artifacts depend on it.

### 2.7 Shadow Or Designed-But-Not-Converged Architecture

Paths to review carefully:

- `src/training/surfaces.py`
- `src/training/pipelines`
- `src/training/templates`
- `src/training/observability`
- `src/training/sidecars`

Diagnosis:

There appears to be a designed target architecture that is not fully wired as the production launch owner. Parts of `src/training` are definitely live, so this is not a clean deletion target. But some surfaces look more like architecture research or shadow descriptors than active execution paths.

Recommendation:

Make a user-level decision:

- promote this architecture into the real launch path,
- or retire the shadow pieces after verifying no live callers.

Do not let it remain indefinitely as a parallel vocabulary.

## 3. Protected Active Surfaces

These should not be deleted or aggressively moved without narrow tests and, where needed, OpenSpec/docs sync.

### Routing And Governance

- `docs/PROJECT_CONTEXT.md`
- `docs/SYSTEM_OVERVIEW.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/AGENT_INDEX.md`
- `docs/catalog.yaml`
- `docs/standards/REPO_HYGIENE.md`
- `openspec/specs/*`

### Data And Geometry

- `src/datasets/geometry.py`
- `src/datasets/builders/jsonlines.py`
- `src/common/geometry`
- `docs/data/CONTRACT.md`

Do not collapse geometry casually. Dataset/public-data preprocessing and runtime/eval helpers currently have distinct roles. Preserve image/geometry alignment and bbox semantics over cosmetic cleanup.

### Detection Templates And Active Template Variants

- `src/detection/template.py`
- `src/detection/template_contracts.py`
- `src/detection/runtime.py`
- `src/detection/dataset.py`
- `src/detection/teacher_forcing/compact_full_policy.py`
- `src/common/detection_sequence.py`
- `src/common/detection_compact_rows.py`
- `src/config/prompts.py`
- `src/training/stage2/rollout_codec.py`

Strict Stage-1 semantic IDs appear to be:

- `stage1_json_pretty`
- `compact`
- `compact_box_closed`
- `compact_object_box_closed`
- `compact_object_box_closed_lines`

`compact_full` remains compatibility vocabulary in parser, rollout, history, and archived contexts. It should not be used as a new strict Stage-1 template ID, but it also should not be deleted until active callers and Stage-2 rollout IO are handled.

### Stage-1 And Stage-2 Training

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

`rollout_matching.*` is still an active private migration/runtime namespace, not dead code.

### Infer, Eval, Artifacts, Provenance

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

Keep `decode_policy_fingerprint`: it is active provenance, not the removed live `decode_policy` knob.

## 4. Archive, Retire, Or Owner-Review Candidates

### High-Confidence Archive / Quarantine Candidates

These are the largest likely wins, but still require owner sign-off before deletion.

- `src/analysis` study families with no live core importer.
- Completed-study modules under `src/analysis` that correspond to archived OpenSpec changes or historical reports.
- `scripts/analysis` one-off wrappers and tmux launch recipes.
- `configs/analysis` families without durable rerun purpose.
- Study-style OpenSpec specs under `openspec/specs` that are better represented as `progress/` evidence.

Recommended action:

Archive/quarantine by family, not file-by-file. Preserve:

- progress notes,
- artifact roots,
- exact launch command if rerunnable,
- metrics/claim context,
- and the minimal reusable analysis helper if future studies depend on it.

### Retire-After-Verify Candidates

- `src/coord_tokens`: compatibility re-export shims to `src/tokens`.
- trainer metric shims under `src/trainers/metrics`.
- shallow descriptor wrappers under `src/training/pipelines`.
- `src/infer/backend_vllm_infer.py` if colocate mode is confirmed non-production and no longer needed.
- old `compact_full` Stage-1 references after template migration completes.
- local geometry helpers in `src/common/prediction_parsing.py` that overlap `src/common/geometry/coord_utils.py`.
- authored live-looking `decode_policy:` keys in runtime-looking configs after proving they are not consumed by live infer code.

### Stale Or Misleading Names

- `tests/test_stage2_ab_*.py`: if they test live rollout correction, rename to match current surface.
- `src/infer/backend_sync.py`: if it is weight/token-row-offset sync rather than an inference backend, rename to a domain-specific name.
- `scripts/tools/test_coord_tokenization.py`: if it is a CLI rather than pytest, rename to `verify_*` or similar.
- `configs/stage1/teacher_forcing`: likely superseded by `configs/stage1/detection_teacher_forcing`.
- `scripts/pipelines/train_task_manager.py`: owner-review; promote-with-docs or archive.

### Configs That Look Runnable But Need Lifecycle Review

- `configs/bench/*`
- repeated `coord_components_2b` infer/postop/eval leaves,
- old per-run leaves under `configs/infer/recursive_detection_ce`,
- any YAML with dead local checkpoint paths,
- analysis configs that encode one completed study.

Do not delete configs until artifact roots and metric evidence are recorded.

## 5. Active Template Migration Risks

This is the main place where the Claude report was too compressed and the original Codex report had useful guardrails.

### 5.1 Stage-1 Template Naming Is Split

Evidence from the prior audit lanes:

- stable spec/docs still reference `compact_full_support2.yaml` and `detection_template.id: compact_full`,
- current `docs/catalog.yaml` and active config work point to `compact_support2.yaml` and `compact_tiny.yaml`,
- current code validates semantic IDs such as `compact`, `compact_box_closed`, `compact_object_box_closed`, and `compact_object_box_closed_lines`,
- some tests and scripts still reference the old `compact_full_*` paths.

Impact:

An agent following stable specs can create configs current schema rejects. An agent following current code can break older tests/scripts.

Recommendation:

Finish this rename lane atomically:

- update stable OpenSpec specs,
- update canonical training docs,
- update stale tests,
- update analysis launchers,
- decide whether compatibility copies remain,
- then add a grep/test gate to prevent reintroduction.

### 5.2 Active Tests Still Use `compact_full`

Known affected areas from prior audit:

- dataset metadata tests,
- norm1000/latest-view tests,
- detection training dataset tests,
- detection length bucketing tests,
- Stage-2 residual boundary adapter tests,
- at least one analysis helper.

Recommendation:

Use `compact` or another semantic strict ID for active Stage-1 tests. Keep `compact_full` only in:

- explicit rejection tests,
- archived fixtures,
- compatibility parser tests,
- Stage-2 rollout-family tests that intentionally need old rollout vocabulary.

### 5.3 Prefix-Rollin Contract Is Not Fully Converged

Observed mismatch:

- active change intends compact semantic IDs,
- schema accepts compact-family contracts,
- dataset runtime may still reject anything except `compact` for `prefix_rollin_et_rmp_ce`,
- stable spec may still say `compact_full`.

Decision needed:

- Either narrow schema/spec to `compact` until variants are validated,
- or implement runtime support for all compact semantic IDs and add tests.

### 5.4 `configs/infer/recursive_detection_ce` Is Active Migration Work

This family is confusing because filenames and checkpoint lineage still say `compact_full`, while current parser contract is moving to `compact`.

Recommendation:

Do not delete or revert it during template migration. After migration:

- rename active leaves,
- or add a README saying `compact_full` in filenames is historical checkpoint lineage, not a live template ID.

## 6. Duplicated Concepts And Naming Confusion

### Template Vocabulary

Current confusion:

- `compact_full`,
- `compact`,
- semantic compact variants,
- `coordjson`,
- helper formats such as `compact_no_desc`, `compact_no_bbox`, and `compact_min`.

Resolution:

Use semantic IDs for strict current templates. Keep helper/compat names private and time-boxed.

### Decode Policy

Do not conflate:

- removed live config knob: `decode_policy`,
- active provenance field: `decode_policy_fingerprint`,
- analysis-runner metadata fields such as `a3_2_launch_spec.decode_policy`.

Resolution:

Keep `decode_policy_fingerprint`. Remove or rename authored live-looking `decode_policy:` keys in runtime configs after verifying consumers.

### Runtime Names

Repeated names hide ownership:

- `src/detection/runtime.py`
- `src/infer/runtime.py`
- `src/trainers/stage2_rollout_runtime.py`
- `src/training_runtime`

Resolution:

Prefer domain-qualified names and package docs that state dependency direction.

### Metrics And Token Shims

Confusion:

- `src/metrics` vs `src/trainers/metrics`,
- `src/coord_tokens` vs `src/tokens`.

Resolution:

Make canonical imports obvious. Keep old imports only as compatibility shims with expiry.

## 7. Prioritized Cleanup Plan

| Priority | Work | Impact | Risk | Verification |
|---:|---|---|---|---|
| 1 | Finish detection-template variant convergence across specs/docs/tests/scripts/configs | Very high | Medium | `rg` gates for old names; template/config/dataset pytest slice |
| 2 | Add lifecycle registry and hygiene gates | Very high | Low/medium | new `tests/test_repo_hygiene.py` or equivalent script |
| 3 | Classify and quarantine `src/analysis`, `scripts/analysis`, `configs/analysis` | Very high | Medium | importer scan, analysis test slice, progress/artifact preservation |
| 4 | Rename misleading surfaces (`test_stage2_ab_*`, `backend_sync.py`, CLI named `test_*`) | High | Low | import/test-collection smoke, `rg` old names |
| 5 | Consolidate misleading config families and benchmark bundles | High | Medium | config parse, artifact-root checks, eval-only smoke where possible |
| 6 | Codemod obvious compatibility shims such as `src/coord_tokens` | Medium/high | Low | import scan and coord geometry/token tests |
| 7 | Split `src/config/schema.py` by domain without changing strictness | High | Medium/high | strict unknown-key tests and config-load tests |
| 8 | Extract `src/sft.py` preflight/fingerprint/manifest clusters | Medium/high | Medium | manifest/fingerprint tests, config-only launch checks |
| 9 | Decompose Stage-2 rollout correction internals behind existing facade | High | High | Stage-2 contract/runtime/assignment/duplicate-filter tests |
| 10 | Split `target_builder.py` last | Medium | High | golden IR fixtures and Stage-2 supervision tests |

## 8. Future Development Constitution

### Rule 1: Every Surface Has Lifecycle State

Every nontrivial file or family under `src/`, `scripts/`, `configs/`, `docs/`, or `openspec/` should be one of:

- core,
- active research,
- compatibility,
- temporary,
- historical,
- retired.

Unlabeled experiment code should not enter normal `src`.

### Rule 2: New Ideas Start Outside Stable Surfaces

Start speculative work in:

- `.worktrees/<idea>`,
- `temp/<date>-<topic>`,
- `progress/<topic>`,
- `configs/analysis/<topic>`,
- `scripts/analysis/<topic>` only for reusable config-backed tools.

Promote later through evidence, tests, and docs/spec decisions.

### Rule 3: Promotion Requires A Dossier

Before promotion into `src`, top-level `scripts`, or active `configs`, require:

- owner,
- hypothesis or purpose,
- minimal file slice,
- lifecycle label,
- artifact/progress pointer,
- verification command,
- docs/spec impact,
- compatibility expiry condition,
- explicit list of files not promoted from the worktree.

### Rule 4: Compatibility Expires

Every shim, alias, facade, or old config key needs:

- reason,
- allowed callers,
- forbidden new callers,
- removal condition,
- removal check,
- review date or OpenSpec change ID.

Examples:

- `compact_full`,
- `rollout_matching.*`,
- `src/common/detection_sequence.py`,
- `src/eval/detection.py`,
- `src/coord_tokens`,
- trainer metric shims.

### Rule 5: Worktrees Are Idea Containers, Not Merge Units

Before promoting from `.worktrees`:

```bash
git cherry -v main <branch>
git diff --name-only main...<branch>
git diff --shortstat main...<branch>
```

Then extract a single idea. Do not merge a whole worktree just because part of it is valuable.

### Rule 6: OpenSpec Is For Stable Contracts, Not Ordinary Studies

Use OpenSpec for:

- training behavior,
- eval behavior,
- config schemas,
- loss/metric semantics,
- artifact names/contracts,
- stable runtime compatibility.

Use `progress/` for:

- study plans,
- diagnostic notes,
- empirical evidence,
- historical analysis conclusions.

### Rule 7: Enforce With Cheap Gates

Add a lightweight repo hygiene gate, likely as `tests/test_repo_hygiene.py` or a script invoked by tests.

Candidate checks:

```bash
rg -n "compact_full_support2|compact_full_tiny|detection_template.id: compact_full|compact_detection_sequence" docs openspec/specs scripts tests configs
rg -n "decode_policy:" configs src scripts
rg -n "run_infer_eval.sh|run_rollout_stability_probe|launch_.*tmux" docs scripts configs tests
```

Programmatic gates should check:

- active docs do not reference missing config paths,
- active configs do not use retired keys,
- new `src/analysis` families have lifecycle metadata,
- compatibility expiry dates have not passed,
- study specs do not appear under stable `openspec/specs`,
- path-shaped doc references resolve,
- current config roots parse,
- dead checkpoint paths are either labeled historical or fail preflight.

## 9. Open Questions For The User

1. Should `src/analysis` be physically moved outside importable `src`, or is lifecycle metadata plus import gates enough?
2. Which analysis studies are still expected to be rerun?
3. Is `src/training/surfaces.py` intended to become the production launch owner, or is it a parked architecture experiment?
4. Should `vllm.mode: colocate` remain a documented fallback?
5. Should `compact_full` remain a Stage-2 rollout-family term for one compatibility window?
6. For prefix-rollin, should semantic compact variants be supported now, or should support narrow to `compact` until validated?
7. Should study-style OpenSpec specs be demoted retroactively, or only blocked going forward?
8. Which worktrees are still active research versus historical references?
9. Are pytest-based hygiene gates acceptable as enforcement, or should checks remain agent-discipline-only?
10. How aggressively should internal import compatibility be preserved for `src/coord_tokens` and trainer metric shims?

## 10. Recommended First Implementation Sequence

1. Finish the detection-template variant migration and fix stale `compact_full` specs/docs/tests/scripts.
2. Add lifecycle metadata and a cheap hygiene test for analysis/scripts/configs.
3. Classify `src/analysis`, `scripts/analysis`, and `configs/analysis` by family.
4. Archive one-off analysis launch wrappers with progress/artifact pointers.
5. Rename misleading tests/modules that refer to retired surfaces.
6. Consolidate config bundles and remove dead-checkpoint ambiguity.
7. Codemod obvious compatibility shims.
8. Split `schema.py` and `sft.py` only after the above gates are in place.
9. Decompose Stage-2 internals last, behind stable public facades.

## 11. Verification Status

This report revision did not rerun repository-wide tests, training, inference, eval, artifact generation, or worktree pruning.

Read-only evidence used for the original audit and comparison included:

- `wc`, `find`, `rg`, `sed`, `nl`,
- `git status`,
- `git worktree list`,
- CodeGraph and Serena file/symbol overview,
- six Codex subagent audit lanes,
- comparison against `/data/CoordExp/codebase-analysis-claude.md`.

Recommended verification for future cleanup lanes is listed in the plan above.

## Bottom Line

Claude's report was the better standalone cleanup memo because it was sharper, more quantitative, and more decisive. This revised Codex report adopts that structure, while keeping the safety details needed for CoordExp: active template migration, geometry correctness, Stage-2 stability, decode provenance, artifact contracts, and worktree promotion discipline.

The main cleanup target is not "bad code." It is unlabeled lifecycle state. Once surfaces are labeled as core, active research, compatibility, historical, temporary, or retired, the repo becomes much easier for agents to navigate and much safer to simplify.

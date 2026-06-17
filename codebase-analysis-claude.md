# CoordExp Codebase Architectural Diagnosis (claude)

**Scope:** read-only architectural diagnosis of `/data/CoordExp` (main working tree).
**Date:** 2026-06-17 · **Mode:** read-only (no edits/deletes/stages/commits/worktree-prunes/jobs).
**Method:** docs-route grounding (`docs/PROJECT_CONTEXT.md → SYSTEM_OVERVIEW.md → IMPLEMENTATION_MAP.md → catalog.yaml → domain docs → openspec/specs/`), then 7 parallel evidence-cited audits (src-core, trainers/infer/eval, src/analysis, scripts/, configs/, docs+specs+tests+governance, .worktrees), then independent verification of the highest-stakes claims.

All major claims cite exact paths. Section headings are deliberately aligned with the requested deliverables so this report can be diffed against an independent audit.

**Revision (2026-06-17, post cross-audit):** reconciled against the independent `codebase-analysis-codex.md`. Folded in three verified findings that audit surfaced and this one missed — the **package-level import cycle** (§2.7), the **docs/spec/test-vs-code drift list** (§4.5), and the **`decode_policy` provenance duplication** (§5) — each re-verified here, not imported on trust. Also self-corrected one error this report originally contained: `src/eval/orchestration.py` was wrongly listed as a protected live surface; it is an orphan (0 importers) superseded by `detection_orchestrator.py` (§3, §4.3).

---

## 0. Quantitative grounding (the numbers that frame everything)

| Surface | Size | Signal |
|---|---|---|
| `src/` total | 180,291 LOC / 394 `.py` | baseline |
| **`src/analysis/`** | **62,259 LOC / 89 files = 34.5% of all src** | **0 importers outside `src/analysis` (verified). Larger than `src/trainers/` (35.5k).** |
| `src/trainers/` | 35,542 LOC / 65 files | core Stage-2, but 2 god-files dominate |
| `src/infer/` | 14,323 / 17 | core; backend naming confusion |
| `src/detection/` | 12,963 / 24 | core Stage-1 |
| `src/training/` | 11,722 / 58 | live + a large test-only **shadow** island |
| `src/config/schema.py` | **4,893 LOC single file** | validation-boilerplate monolith |
| `src/sft.py` | **4,324 LOC** | a script (0 prod importers); `main()` alone ≈1,900 LOC |
| `configs/` | 218 YAML | config-first is a hard rule |
| **`configs/analysis/`** | **91 / 218 YAML = 42%** (verified) | only 5 of 22 families have backing code |
| `scripts/` | 93 scripts (73 py / 20 sh) | only **3** are docs-blessed entrypoints |
| `tests/` | 339 files / ~115k LOC | ~22–26% targets retired/shadow/study surfaces |
| `openspec/specs/` | ~35 spec dirs | 4 are one-off "study" specs; 4 retired-pointers |
| `.worktrees/` | 7 branches | **0 merged into main** (verified) |

Two facts dominate the whole diagnosis:

1. **`src/analysis/` (62k LOC, 35% of src) and `configs/analysis/` (91 YAML, 42% of configs) are research-study residue that no core code imports.** This is the single largest, lowest-risk bloat mass.
2. **The governance docs already say the right things; there is zero enforcement.** Bloat recurs because no check fails when a rule is broken.

---

## 1. High-level diagnosis: why the repository grew this way

This is **not** an undisciplined codebase. The intended architecture is unusually well-documented (a clean docs route, a `runtime-architecture-refactor-program` spec, an `INDEPENDENT_ARCHITECTURE_REVIEW.md`, a `REPO_HYGIENE.md` lifecycle policy, an `AGENT_ENGINEERING_CONSTITUTION.md`). The bloat comes from five specific, identifiable lifecycle failures — each one a *mechanism*, not a vibe:

1. **Research studies became permanent importable modules.** Every diagnostic experiment ("why does the model duplicate small objects", "autoregressive FN-rescue", "instance binding", "prefix-state tomography") was written as a full `src/analysis/<study>.py` module + a `scripts/analysis/run_<study>.py` runner + a `configs/analysis/<study>/*.yaml` family + (sometimes) an `openspec/specs/<study>-study/spec.md`. None of it is imported by training/infer/eval. It is **leaf code that never had a promotion gate or an expiry date**, so it accreted to 62k LOC. This is ~80% of the bloat.

2. **Contracts race ahead of code (designed-but-not-converged architecture).** The repo designs a clean target (`DetectionScene` IR, a `TrainingSurfaceResolver` that owns launch, a unified runtime) as an OpenSpec contract *before* migrating the implementation. The result is **two parallel vocabularies living simultaneously**: the live `src/detection/{objective,loss,template}` world and a test-only shadow `src/training/{surfaces.py,pipelines/,templates/}` island with **zero production importers**. Three breaking OpenSpec changes are open at once while a prior rename (`compact_full`→`compact`, still in 22 active modules) is unfinished.

3. **Retired surfaces were collapsed cleanly in *code* but their *scaffolding* persists.** Retired trainers (rollout-matching-sft, stage2-ab/two-channel, self-context) were genuinely removed via `training_runtime/plan.py` remapping + `config/schema.py` fail-fast — good. But ~12k LOC of tests still live under the dead name `test_stage2_ab_*.py` (they actually test live `rollout_correction`), retired specs linger in `openspec/specs/`, and comparator configs/tests for `recursive_detection_ce` remain. The names mislead every future reader.

4. **God-files at the launch and Stage-2 core.** `sft.py` (4,324), `config/schema.py` (4,893), `stage2_rollout_correction_impl.py` (5,923), `stage2_rollout_runtime.py` (4,585), `rollout_correction/target_builder.py` (4,382) accreted by the "add one more conditional to the big entrypoint" pattern the constitution itself warns against (`AGENT_ENGINEERING_CONSTITUTION.md` §16).

5. **Per-launch config/file fan-out.** The active `configs/infer/recursive_detection_ce/` encodes up to 7 experiment axes in a single filename (`...ckpt3664_val200_bsz4_rep1p10_max3084_chatfix_8gpu.yaml`). One YAML per debugging attempt — exactly what `configs/README.md` already prohibits.

**Crucially, the worktrees did NOT cause main's bloat.** All 7 are isolated; `git cherry`/`branch --merged` confirm **zero** of them landed in main. `src/analysis/` grew via *direct-to-main commits* (Mar–Jun 2026) that predate the worktrees. The worktree workflow is working as designed — the heaviest speculative mass (mechanistic's +74k diff) is correctly isolated. The cleanup target is main itself, not the worktrees.

---

## 2. Main bloat / complexity hotspots (exact paths)

### 2.1 `src/analysis/` — the prime mass (62,259 LOC, 35% of src, 0 core importers)

The single biggest lever. Verified: `rg "from src.analysis|import src.analysis" src --glob '!src/analysis/**'` → **none**; no core package imports it. Reachable only from `scripts/analysis/*` and `tests/*`. By the repo's own CLAUDE.md rule ("one-off research code should NOT pollute the main importable `src/` surface"), the overwhelming majority does not belong in `src/`.

Highest-confidence archive targets within it — **5 study subpackages with no OpenSpec spec at all = 17,684 LOC / 62 files**:
- `src/analysis/sorted_random_no_newline_phenotype/` (9,006 LOC — by itself larger than `src/metrics/` + `src/utils/` combined)
- `src/analysis/post_x1_instance_basin_tomography/` (3,282)
- `src/analysis/prefix_state_transition_tomography/` (2,734)
- `src/analysis/candidate_field_cardinality_tomography/` (2,064)
- `src/analysis/policy_objective_mechanism_comparison/` (598)

Three of the largest top-level modules correspond to **archived (completed) OpenSpec changes** — they are study artifacts, not living code:
- `src/analysis/rollout_fn_factor_study.py` (3,384) → `openspec/changes/archive/2026-03-31-analyze-2b-fn-rollout-factors/`
- `src/analysis/unmatched_proposal_verifier.py` (4,359) → `…/2026-03-16-unmatched-proposal-verifier-ablation/`
- `src/analysis/duplication_collapse_analysis.py` (4,013) → `…/2026-05-27-add-duplication-collapse-analysis-study/`

Concrete name-duplication waste (~7k LOC of twins): `small_object_duplication_study.py` (1,806) vs `small_object_duplication_diagnostics.py` (1,825) — same subject, same imports, only one has a runner; `duplication_collapse_analysis.py` + `duplication_followup.py`; `autoreg_fn_rescue_desc_x1_phase2/4/5` (phase3 already deleted).

**The only "infrastructure" in here is infrastructure-for-archived-code:** `hard_ce_coord_logit_locality.py` (18 sibling imports), `unmatched_proposal_verifier.py` (9), `rollout_fn_factor_study.py` (3) are heavily reused — but exclusively by *other archive-eligible studies*. If the dependents are archived, these move with them. Worth extracting at most `src/analysis/rollout_parity.py` (417 LOC, reused) and the `TeacherForcedScorer` helper if future studies are planned.

### 2.2 `config/schema.py` (4,893 LOC) — validation-boilerplate monolith

55 `@dataclass` + 31 hand-rolled key-allowlist helpers + 35 `__post_init__` guards + 112 strict-error raises — **despite a shared `strict_dataclass.parse_dataclass_strict` already existing**. An agent cannot edit one validator without risking dozens of near-identical siblings.

### 2.3 `src/sft.py` (4,324 LOC) — a script masquerading as a library

Zero production importers; it is the launch *script*. `main()` spans ~1,900 LOC as one function, plus 75 module-level `_build_*_fingerprint` / `_validate_*` / `_*_runtime_payload` one-offs. Fingerprint/manifest/preflight clusters belong in `src/bootstrap/`.

### 2.4 Stage-2 god-files

- `src/trainers/stage2_rollout_correction_impl.py` (5,923) — one class + ~80 free functions mixing ≥6 prefix-delimited concerns (UL-consensus artifacts, Channel-B assignment, pending-metrics aggregation, rollout temperature/attempt telemetry, DetectionScene target-state/provenance, train/eval monitor dumps). Sibling modules (`ul_consensus.py`, `monitoring/`, `metrics/`, `rollout_matching/telemetry.py`) already exist as extraction targets. The public facade `src/trainers/stage2_rollout_correction.py` (20 lines) makes internal extraction low-risk for external importers.
- `src/trainers/stage2_rollout_runtime.py` (4,585) — a god-base whose `__init__` initializes ~25 state fields across 4 unrelated subsystems (packing carry buffer, vLLM colocate, vLLM server client/lock, monitor-dump budgets). Should be a composition of collaborators, not one inheritance base.
- `src/trainers/rollout_correction/target_builder.py` (4,382) — most defensibly deep (one coherent domain: residual-set supervision IR), but two ~600-LOC mega-functions dominate. Split last, gated on golden-IR fixtures (highest loss-semantics sensitivity).

### 2.5 `configs/analysis/` (91 YAML, 42% of all configs)

Only **5 of 22 families** have backing `src/analysis/` code; the other ~17 (incl. `duplication_collapse`=21, `mixed_objective_sota_probe`=10, `duplication_followup`=9, all `autoreg_*`) are one-off study harnesses with no module — they should be `progress/` artifact pointers, not runnable YAML.

### 2.6 `scripts/analysis/` (~7,600 LOC of one-experiment wrappers)

~30 `run_*` shims + 10 `launch_*_tmux.sh` + 5 study subpackages, referenced **only** by archived `docs/history/superpowers/plans/*`. Thin CLI skins over the `src/analysis/*` studies above.

### 2.7 Package-level import cycle (the deeper reason navigation is expensive)

The god-files explain *vertical* complexity; this explains *horizontal* complexity. The top-level `src/` packages form one large dependency cycle, so there is no clean low→high layering an agent can follow. Verified reciprocal edges:
- `src/detection ↔ src/training` — `detection/{dataset,loss}.py` import `training`, while `training/{span_adapters/*,templates/compact_full.py}` import `detection`.
- `src/eval ↔ src/infer` — `eval/{detection_orchestrator,oracle_k,detection_coco,…}.py` import `infer`, while `infer/{pipeline,artifacts}.py` import `eval`.
- `src/common` is a gravity well: **69 files import it** (verified), so it behaves like a second config layer rather than a thin base.

**Why it matters:** any change to template/IR contracts, decode/artifact provenance, or metric projection ripples in both directions, so impact analysis can't be localized — this is the structural root of the agent token cost, on top of the file sizes in §2.2–2.4. Highest-value cycle cuts (lowest risk first): (a) `eval→infer`: share artifact/decode-provenance structures so the evaluator stops importing infer runtime; (b) `detection↔training`: move template contracts + detection IR to an owned neutral layer that `training` consumes one-directionally; (c) `metrics↔trainers`: keep generic metrics in `src/metrics`, make trainer-specific projections private to the trainer package. (Folded from the independent audit's Hotspot C; reciprocity re-verified here.)

---

## 3. Active / core surfaces to PROTECT

These are deep, widely-imported, contract-bearing, or docs-blessed. Do not refactor their *interfaces* without an OpenSpec change.

**Stage-1 detection (live):** `src/detection/{runtime,template,objective,loss,dataset,data}.py` (`detection.template` has 14 prod importers); `src/datasets/geometry.py` (the bbox-math single source of truth, mandated by CLAUDE.md); `src/common/{detection_sequence,detection_compact_rows}.py`; `src/tokens/coord/*` + `src/tokens/row_offsets.py` (the real coord-token codec).

**Stage-2 (live):** `src/trainers/stage2_rollout_correction.py` (public facade), `…_impl.py` (canonical trainer), `stage2_rollout_runtime.py` (spec-blessed base), `rollout_correction/target_builder.py` (loss IR), `rollout_matching/{contracts,parsing,matching,packing,preflight,telemetry}.py` (the Stage-2 "stdlib", 10+ importers), `src/training/{stage2,teacher_forcing,objectives,supervision}/`, `src/launchers/stage2_vllm_server.py`.

**Infer/eval (live):** `src/infer/rollout_dispatch.py` (backend selector) + the genuine backend family `backend.py` (HF), `backend_vllm_infer.py` (colocate), `backend_vllm_server.py` (server), `backend_vllm_engine.py`/`backend_vllm_config.py` (support); `src/infer/{runtime,pipeline,artifacts}.py`; `src/eval/{detection,detection_orchestrator,artifacts}.py` (single scoring kernel `evaluate_and_save`; the live orchestrator is `detection_orchestrator.py` — **not** the orphaned `orchestration.py`, see §4.3).

**Config contract (live):** `src/config/{schema,loader}.py` — must stay behaviorally stable even if split internally.

**Launch path (live):** `src/sft.py` + `src/training_runtime/` (`plan.py` + `TrainerFactory`) + `src/bootstrap/`.

**Docs-blessed scripts (only 3 per `docs/IMPLEMENTATION_MAP.md`):** `scripts/run_infer.py`, `scripts/evaluate_detection.py`, `scripts/postop_confidence.py`. Spec/WORKFLOW-referenced (keep): `scripts/evaluate_oracle_k.py`, `evaluate_proxy_detection_bundle.py`, `materialize_proxy_eval_views.py`, `export_coco_submission.py`, `train.sh`, `train_stage2.sh`, `merge_coord.sh`, `absorb_output_remote_into_outputs.py`, `pipelines/run_rollout_stability_probe.sh`. Reusable tools: `scripts/tools/{convert_to_coord_tokens,inspect_chat_template,expand_coord_vocab,materialize_rescaled_images_from_jsonl}.py`.

**Active config routes (protect content):** `configs/stage1/detection_teacher_forcing/`, `configs/stage2/rollout_correction/`, `configs/_shared/{datasets,prompts}`, `configs/{infer,eval,postop}/` pipeline anchors, and **`configs/infer/recursive_detection_ce/`** (the current chat/detection template work — placement is correct; only the naming needs discipline, see §5).

**Canonical routing docs (well-maintained — verified 21/21 IMPLEMENTATION_MAP paths and 50/50 sampled catalog.yaml paths exist):** `docs/catalog.yaml`, `docs/IMPLEMENTATION_MAP.md`, `docs/SYSTEM_OVERVIEW.md`.

---

## 4. Likely legacy / temporary / historical surfaces — delete / archive / owner-review

Confidence reflects strength of evidence. **None of these should be deleted by an agent without your sign-off**; this is an evidence-backed candidate list.

### 4.1 ARCHIVE candidates — HIGH confidence (no core importer, completed study, evidence cited)
- **`src/analysis/`'s 5 spec-less subpackages** (17,684 LOC / 62 files) — §2.1. Move to `research/` or `progress/`; keep the spec where one exists.
- **`src/analysis/` completed-study modules** mapped to archived OpenSpec changes (`rollout_fn_factor_study`, `unmatched_proposal_verifier`, `duplication_collapse_analysis`, `coco_lvis_missing_objects` [deprecated spec]) and the autoreg cluster (~18k LOC). Aggregate archive-eligible: **~50k of 62k LOC (~80%)**.
- **`scripts/analysis/` one-experiment wrappers** (~7.6k LOC): all 10 `launch_*_tmux.sh`, all `run_autoreg_*`/`run_*_study.py` shims, the 5 study subpackages, `run_ckpt_pair_confidence_eval.sh`, `build_mixed_objective_sota_probe_report.py`, `debug_rollout_collection_parity.py`. Archive alongside their `src/analysis/` counterparts.
- **`configs/analysis/` families without backing code** (~17 of 22 families): `autoreg_*`, `duplication_collapse` (21), `duplication_followup` (9), `mixed_objective_sota_probe` (10), `small_object_duplication*`, `qwen3_vl_instance_binding`, `rollout_fn_factor_study`, `unmatched_proposal_verifier`. Per `configs/README.md`'s own rule, replace with a `progress/` artifact pointer.

### 4.2 RETIRE-AFTER-VERIFY — MEDIUM/HIGH confidence
- **`src/training/{surfaces.py, pipelines/, templates/, observability/, sidecars/}` — the shadow architecture island.** `surfaces.py::TrainingSurfaceResolver` and `pipelines/{stage1_json_ce,stage1_compact_trie_ce}` have **zero production importers** (only `tests/test_training_surface_resolver.py` etc.); `sft.py` routes through `training_runtime/plan.py` instead. Docs call it "the guarded shadow resolver… private to architecture research." ⚠️ **Caution:** the *lower* utility modules (`training/encoding`, `span_adapters`, `bridge`, `ordering`) DO have live consumers — the island is not cleanly severable. Verify with `rg` before any removal. **This is the #1 "designed-but-not-wired" question for you (§7).**
- **`src/coord_tokens/*`** — 5 pure 6-LOC re-export shims over `src/tokens/coord/*`, still imported by ~16 prod modules. NEEDED today; retire only after a codemod migrates imports to `tokens.coord` + a guard test. Cheap, mechanical.
- **`src/infer/backend_vllm_infer.py` (vLLM colocate)** — live but **no prod config selects `vllm.mode: colocate`** (prod uses `server`); only `base.yaml` defaults + bench scripts exercise it. Keep as documented non-prod fallback, or retire after confirming the bench harness (`scripts/analysis/rollout_backend_bench/`) doesn't depend on it.

### 4.3 STALE / MISLEADING — needs cleanup (low risk, high clarity gain)
- **`tests/test_stage2_ab_*.py` (12 files, ~11.9k LOC)** — named after the retired `stage2_ab` surface but actually test **live `rollout_correction`**. Worst-case shadow: active tests under a dead name. Rename to `test_rollout_correction_*`.
- **`src/infer/backend_sync.py`** — **not an inference backend at all.** It's `CoordExpWeightSyncWorkerExtension` (a vLLM worker-side weight/token-row-offset sync patch) used by `src/launchers/swift_rollout_coordexp.py`. The name causes it to be miscounted as a third generation backend. Rename (e.g. `rollout_server_weight_sync.py`).
- **`configs/stage1/teacher_forcing/` (singular, 1 YAML, 2026-05-20)** — orphan superseded by canonical `detection_teacher_forcing/`. Consolidate/remove.
- **`scripts/tools/test_coord_tokenization.py` (353 LOC)** — `test_`-named but is an argparse CLI with no pytest hooks; will be falsely collected. Rename to `verify_*`.
- **`src/eval/orchestration.py` (5.1 KB, 2026-04-03)** — **orphan: zero importers anywhere** (verified). Superseded by the live `src/eval/detection_orchestrator.py` (19.6 KB, imported by `src/eval/detection.py`). A stale duplicate of the eval-orchestration concept; archive candidate. *(This report originally mis-listed it as a protected surface — corrected here.)*
- **`scripts/pipelines/train_task_manager.py` (912 LOC)** — accreted tmux/GPU-babysitting orchestrator referenced by zero docs/specs. Owner-review: promote-with-docs or archive.
- **Dead-checkpoint configs:** sample-check found **5 of 8 `model_checkpoint` paths in `configs/infer/recursive_detection_ce/` are MISSING on disk** (evicted run dirs / ckpt2200 / ckpt3600). Current-looking but would fail at launch. The "misleadingly runnable" risk here is dead-*checkpoint*, not dead-*trainer*.
- **4 study-specs in `openspec/specs/`** (`duplication-collapse-analysis-study`, `rollout-fn-factor-analysis-study`, `unmatched-proposal-verifier-study`, and arguably `training-pipeline-audit`) — violate the repo's own "Do not use OpenSpec for ordinary experiment planning." Demote to `progress/`.
- **Docs-vs-tree drift:** `docs/data/{PREPARATION,CONTRACT,VISUAL_GENOME}.md` cite ~a dozen `scripts/<name>` data-prep scripts that don't exist at that path (only `tools/` variants exist). `INDEPENDENT_ARCHITECTURE_REVIEW.md:479` lists stale template IDs (`compact_full` instead of the live `compact`/`compact_box_closed/...`).

### 4.4 Worktree owner-review (do NOT prune without confirming)
- **`row-conditioned-visual-coverage`** — cleanest redundancy: its only commit not already byte-identical inside `mechanistic-diagnosis-experiments` is the shared "Remove decode-time grammar constraints."
- **`loss-only-instance-enumeration`** — coverage/instance_enumeration/ablation trees byte-identical to mechanistic; unique work = one commit (`Tighten instance enumeration mass loss`). Confirm it's preserved in mechanistic, then redundant.
- Leave isolated (genuinely distinct, active): `geometry-aware-denoising-sft` (own `src/detection/prefix_denoising/`, newest), `segment-aware-packing-infra` (+14 unique FA2/packing commits), `mechanistic-diagnosis-experiments` (umbrella superset — its +74k analysis diff is exactly what should stay OUT of main), `coord-repel-conservative-design`, `fully-compact-2x2-ablation`. Note: `geometry-aware-denoising-sft` dir name ≠ branch name `codex/prefix-denoising-sft` (drift to confirm).

### 4.5 Docs / specs / tests that disagree with current code (cross-audit, each re-verified)

The canonical routing docs are accurate (§3), but the **`compact_full`→`compact` rename has desynchronized the stable specs, smoke tests, and a stale package README** — agents following these will write configs the live schema rejects. All re-verified during this revision:

- **Stable spec lags the rename.** `openspec/specs/stage1-detection-objectives/spec.md` still mandates `detection_template.id: compact_full` and references `compact_full_support2.yaml` (lines 52, 55, 68, 81) — but the live config tree authors `compact` (`configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`). A *stable contract* now contradicts the code.
- **Spec vs runtime contradiction (a real bug class).** The spec says `prefix_rollin_et_rmp_ce` MUST require `detection_template.id: compact_full` (spec line 81), but `src/detection/dataset.py:483` raises *"prefix_rollin_et_rmp_ce currently requires the compact template"*. A config that satisfies the spec/schema can fail at dataset construction. Pick one contract.
- **Smoke test pins a renamed file.** `tests/test_training_architecture_tiny_smoke.py:17,41,44` hardcode `…/smoke/compact_full_tiny.yaml`, but the directory now contains `compact_tiny.yaml`.
- **Active (non-skip) tests still call the rejected ID.** `get_detection_template("compact_full")` appears in `tests/test_latest_detection_view_metadata.py`, `tests/test_latest_detection_norm1000_view.py`, `tests/test_stage2_residual_boundary_adapter.py`, and `src/analysis/hard_ce_coord_logit_locality.py`. (`tests/test_detection_template_registry.py` is the legitimate *rejection* test — keep it.) Move live tests to `compact`; reserve `compact_full` for rejection tests, archived fixtures, and the Stage-2 `rollout_template_family: compact_full` path.
- **`src/README.md` is from a previous era.** It still describes a dense-captioning pipeline: *"YAML Config → ConfigLoader → SwiftSft → BaseCaptionDataset"* (lines 12, 40-41, 78-79, 102), conflicting with the detection-first architecture in `docs/IMPLEMENTATION_MAP.md`. Replace with a source-package map or delete.
- **Canonical doc points to a deleted dir.** `docs/training/README.md` (lines 46, 99, 164) references `configs/stage1/compact_detection_sequence/` which no longer exists. It *is* labeled "legacy bridge," but the path is gone — either restore the explicit "non-file/historical" wording or drop the references.

These are the concrete, high-value targets for "finish the rename atomically" — they were the strongest contribution of the independent audit and hold up under verification.

---

## 5. Refactoring / simplification opportunities (prioritized by impact × risk)

Ordered so the cheapest, highest-leverage, lowest-risk moves come first. Each lists the narrow verification that proves contracts stayed intact.

| # | Opportunity | Impact | Risk | Verification |
|---|---|---|---|---|
| 1 | **Archive `src/analysis/` + `scripts/analysis/` + `configs/analysis/` study residue** (move out of importable `src/`, keep specs/artifacts). ~50k src LOC + ~7.6k script LOC + ~17 config families. | **Very High** | Low | `rg "from src.analysis" src --glob '!src/analysis/**'` stays empty (verified empty today); run full test suite minus `tests/analysis/`. |
| 2 | **Rename misleading names:** `backend_sync.py`→weight-sync module; `test_stage2_ab_*`→`test_rollout_correction_*`; `scripts/tools/test_coord_tokenization.py`→`verify_*`. | High (agent navigation) | Very Low | `rg` old names → only intended refs; import + test-collection smoke. |
| 3 | **Codemod `coord_tokens` imports → `tokens.coord`, delete the 5 shims.** | High | Low | `rg "coord_tokens" src` → only deleted files; `tests/test_coord_geometry_invariants.py`, `test_coord_softce_w1_loss.py`. |
| 4 | **Finish the `compact_full`→`compact` rename across the 22 active modules** (it's mid-flight; the active `detection-template-variants` change bans even an alias). | High | Medium | `rg "compact_full" src --glob '!src/analysis/**'` → 0; template factory tests; infer/parse smoke. |
| 5 | **Consolidate `configs/infer/recursive_detection_ce/` to base+overlay.** Move `bsz/rep/max/gpu/chatfix/preflight` out of filenames into CLI overrides or a thin overlay; keep ~1 prod + 1 smoke + a base `detection_template` per checkpoint family. Add a config-parse checkpoint-existence preflight. | High | Low (consolidate, not delete) | Config-parse smoke; confirm the 3 live checkpoints still resolve. |
| 6 | **Split `config/schema.py` by domain** (stage1/stage2/coord/cache/experiment/detection) and route key-validation through the existing `strict_dataclass.parse_dataclass_strict` to kill 31 allowlist helpers + 112 manual raises. | High | Medium (strict-key semantics are OpenSpec-governed) | `tests/test_training_config_strict_unknown_keys.py`, `test_prefix_rollin_schema.py`, full config-parse; preserve exact error messages/paths. |
| 7 | **Extract `sft.py::main()` preflight/fingerprint/manifest clusters into `bootstrap/`.** | Medium-High | Medium (fingerprint changes invalidate caches) | `tests/test_experiment_manifest_file.py`, `test_run_manifest_files.py`; keep fingerprint byte-output identical. |
| 8 | **Decompose `stage2_rollout_correction_impl.py`** by prefix into existing siblings (`ul_consensus.py`, `monitoring/`, `metrics/`, `rollout_matching/telemetry.py`); public facade keeps external imports stable. | High | Medium | `tests/test_stage2_rollout_correction_contract.py`, `test_stage2_rollout_import_boundaries.py`, `test_stage2_pending_metrics_aggregation.py`. |
| 9 | **Decompose `Stage2RolloutRuntime` state into collaborators** (packer / vLLM-server-session / eval-artifact-writer). | High | Higher (touches checkpoint runtime-state serialization) | `tests/test_stage2_rollout_runtime.py` + a checkpoint-resume smoke. Stage after #8. |
| 10 | **Audit the trie-supervision trio** (`detection/teacher_forcing/trie.py`, `trainers/teacher_forcing/modules/stage2_trie_ce.py`, `rollout_correction/trie_supervision.py`) and the 4 coord-soft-CE implementations for true duplication before merging. | Medium | Medium | Golden-output fixtures before any merge. |
| 11 | **Split `target_builder.py`** into prefix-spans / triage / residual-render (do LAST — highest loss-semantics risk). | Medium | High | Golden-IR fixtures + `test_stage2_supervision_planning_smoke.py`, `test_stage2_duplicate_filter.py`. |
| 12 | **Reduce the package import cycle (§2.7)** — start with `eval→infer` (share artifact/decode-provenance structs so eval stops importing infer runtime), then `detection↔training` (neutral template/IR layer). | High (navigation/token cost) | High | `tests/test_detection_eval_output_parity.py`, `test_unified_infer_pipeline.py`, `test_infer_artifact_metadata.py`; import-graph check that the targeted reciprocal edge is gone. |

**Duplicated concepts under different names** (the chief agent-navigation friction): `detection/{objective,loss,template}` (live) vs `training/{objectives,bridge,templates}` (shadow); four "runtime" modules (`trainers/stage2_rollout_runtime`, `infer/runtime`, `detection/runtime`, `training_runtime/`); coord-soft-CE in 4 places; trie-supervision in 3; three greedy-IoU surfaces (`training/stage2/assignment.py`, `planners.py`, `rollout_matching/matching.py`); **two eval orchestrators** (`detection_orchestrator.py` live vs `orchestration.py` orphan, §4.3). Most are *layered, not redundant* — but the shared names hide dependency direction. Prefer domain-qualified names.

**Stale provenance vs retired-knob naming:** `decode_policy_fingerprint` is active provenance and must stay, but an authored `decode_policy:` key still appears in ~10 configs — mostly `configs/analysis/*` (e.g. `post_x1_instance_basin_tomography/`, `sorted_random_no_newline_phenotype/`) plus **one active** `configs/infer/recursive_detection_ce/fullobj_random_purece_ckpt3668_a3_2_rollout1024_greedy.yaml`. Confirm whether `decode_policy:` is still consumed by live infer; if not, drop it from the active config so it stops reading as a live knob. (Minor adjacent item: `src/common/prediction_parsing.py:273` carries a local `clamp_points` geometry helper that overlaps `src/common/geometry/` — low-priority collapse.)

---

## 6. Proposed future development constitution

The repo **already has** `docs/standards/REPO_HYGIENE.md` (canonical) and `docs/AGENT_ENGINEERING_CONSTITUTION.md` (still `status: draft`). They already prescribe the lifecycle (temp→scripts-on-2nd-use→src-with-tests; deprecation = delete + git history; config prod/smoke/ablation folders). **The content is good; the gap is enforcement.** So the constitution below is mostly *mechanism*, not new prose.

**A. Promote the constitution to binding.** Move `AGENT_ENGINEERING_CONSTITUTION.md` from `draft` → `canonical`; add a short "Lifecycle & Bloat-Control Addendum" to `REPO_HYGIENE.md` (the doc that is least cited but most operational).

**B. Single entry funnel for new ideas.** New idea → `superpowers:brainstorming`/`grill-me-with-docs` → if it touches a stable contract, an OpenSpec change; else a `progress/` note + a `temp/YYYY-MM-DD_topic/` prototype. **No new `src/` module without either an active OpenSpec change or a `progress/` decision linking it.**

**C. `src/analysis/` becomes a gated zone.** Every study module must carry a header `# study: <spec-or-progress-link>; promote-or-delete-by: <date>`. Unlinked or expired modules are deletion candidates. Better: studies live under `research/` or `progress/`, not importable `src/`, unless promoted.

**D. Promotion gates (make REPO_HYGIENE §3 testable).** To enter `src/`: (a) ≥1 importer from a live surface OR a registered entrypoint, (b) a targeted test, (c) a `catalog.yaml`/`IMPLEMENTATION_MAP` entry. To enter `scripts/`: the §3 Stage-B IO contract. To enter `configs/.../prod/`: referenced by a doc or test.

**E. Compat-code expiry (the missing piece).** Every shim/alias/facade/comparator carries `# compat-until: <YYYY-MM-DD | OpenSpec-change-id>`. When the date passes or the named change archives, it is **deleted, not extended**. Attach one to every `compact_full` alias and the `recursive_detection_ce` comparators now.

**F. One-rename-at-a-time / archive-before-open.** No new overlapping breaking OpenSpec change opens while a prior breaking rename it depends on is unarchived. Finish `compact_full`→`compact` before broadening clean-break code motion. (This directly addresses the "contracts racing ahead of code" mechanism.)

**G. OpenSpec-vs-study boundary.** OpenSpec only for training/eval behavior, config schema, loss/metric/artifact semantics. **A study contract is never an OpenSpec spec.** Demote the 4 study-specs.

**H. Worktree merge discipline.** Use `worktree-feature-loop`; merge only after the narrowest verification passes; a worktree renaming a public concept must finish the rename across all active modules *or* leave a dated `compat-until` — no half-renamed merges. Periodically reconcile redundant sibling worktrees (the diagnosis family shares one spine).

**I. Config lifecycle.** Directory = research decision; filename = surface role (`<family>/<prod|smoke>/<variant>.yaml`), NOT launch knobs. Reference checkpoints by symbolic alias (the repo already has `recursive_detection_ce_latest/`), never a hardcoded `v0-<timestamp>/checkpoint-N` path that outlives the run dir.

**J. Automate it — this is the highest-leverage gap (there is currently no CI / pre-commit at all).** Add `tests/test_repo_hygiene.py` (the repo already enforces contracts via pytest):
1. **Orphan-src check** — every `src/**.py` (minus name-whitelisted facades) must be imported outside its own package or be a registered entrypoint. *Fails today's `src/analysis/`.*
2. **Config-references-live-surface** — every `trainer_variant`/`detection_template.id`/`objective.id` must resolve in the current schema/factory. *Catches `compact_full` after the rename.*
3. **Retired-surface guard** — retired tokens (`stage2_ab`, `two_channel`, `rollout_matching_sft`, fusion-trainer, post-rename `compact_full`) appear in **no active src module** and fail-fast in config loading.
4. **`compat-until` expiry** — fail CI when a date passed or a named change archived.
5. **Test-name/surface coherence** — flag `test_stage2_ab_*` and require rename or a `# guard:` header.
6. **Doc-path existence** — assert path-shaped strings in `catalog.yaml`/`IMPLEMENTATION_MAP.md`/`SYSTEM_OVERVIEW.md` exist (catches the `compact_full_support2.yaml` and `scripts/<name>` drift).
7. **OpenSpec study-leak** — fail if a new `openspec/specs/*-study` or `*-analysis-*` dir is added.
8. **Checkpoint-path preflight** — config-parse warns when `model_checkpoint` is missing on disk.

---

## 7. Open questions needing YOUR research judgment

These are the load-bearing decisions an audit cannot make for you:

1. **`src/analysis/` (62k LOC, 35% of src): durable research infrastructure or archive-able residue?** This is the single biggest bloat decision. Are the 3 spec-backed studies (fn-factor, unmatched-proposal, duplication-collapse) expected to be **re-run**, or are their findings settled? If settled → archive code, keep only the spec/artifact.
2. **The shadow training-surface resolver** (`src/training/surfaces.py` + `pipelines/`): is it intended to *become* the production launch owner (replacing `training_runtime/plan.py` + `TrainerFactory`), or is it a parked experiment? This decides retire-vs-promote for ~2,000+ LOC and shapes the whole `harden-unified-training-runtime-boundaries` change.
3. **vLLM colocate** (`backend_vllm_infer.py`, `vllm.mode: colocate`): keep as a documented fallback or retire? No prod config selects it.
4. **Coord-soft-CE (4 impls) and the three greedy-IoU surfaces:** intentional layering (target vs shadow vs eval) or accreted duplication to collapse?
5. **Rename `test_stage2_ab_*` now, or wait for the clean-break change to land?** It misleads every reader today.
6. **Demote the 4 study-specs out of `openspec/specs/` retroactively, or only enforce the boundary going forward?**
7. **Are pytest-based hygiene gates acceptable** given there is currently no CI, or is enforcement intended to stay agent-discipline-only?
8. **Worktrees:** is `loss-only`/`row-conditioned`'s unique work fully captured in `mechanistic`, or do those leaves carry config/result deltas you still need? Is `geometry-aware-denoising-sft` dir/branch name drift intentional?

---

## Appendix A — Answers to your specific questions (quick index)

- **Biggest sources of complexity in src/scripts/configs?** `src/analysis/` (62k, §2.1); the 5 god-files `schema.py`/`sft.py`/`stage2_rollout_correction_impl.py`/`stage2_rollout_runtime.py`/`target_builder.py` (§2.2–2.4); `configs/analysis/` (91 YAML, §2.5); `scripts/analysis/` wrappers (§2.6).
- **Too large / shallow / coupled / hard for agents?** Monoliths in §2.2–2.4; the shadow `training/` island (§4.2); the parallel live-vs-shadow vocabularies and 4× "runtime" naming (§5).
- **One-off code that shouldn't be in importable src/?** All of `src/analysis/` (§2.1, §4.1) — verified 0 core importers.
- **Stable entrypoints vs wrappers vs stale vs delete?** §3 (3 docs-blessed + ~12 spec-referenced) vs §4.1/§4.3 (wrappers, `train_task_manager.py`, fake test, etc.).
- **Active vs historical vs duplicated vs misleadingly-runnable configs?** §3 (active), §4.1 (`configs/analysis/`), §4.3 (dead-checkpoint configs are the real misleading-runnable risk; **no** config references a retired trainer — that cleanup was done well).
- **Legacy/compat paths needed vs retire-able?** Needed: `coord_tokens` shims, all infer backends. Retire-after-verify: shadow `training/` island, vLLM colocate (§4.2).
- **Duplicated concepts under different names?** §5 (detection-vs-shadow vocabularies; 4 runtimes; coord-soft-CE×4; trie×3; greedy-IoU×3; analysis twins in §2.1).
- **Public vs internal surfaces?** Protect §3; collapse/hide the shadow island, the impl god-files behind their facades, and the schema monolith (§5).
- **Docs/specs/tests vs code disagreements?** Routing docs are accurate (21/21, 50/50). Drift is concentrated in the `compact_full`→`compact` rename desync — stable `stage1-detection-objectives` spec, a spec-vs-runtime prefix-rollin contradiction, a smoke test pinning the renamed file, live tests calling the rejected ID, a stale `src/README.md`, and a doc pointing at a deleted config dir — full verified list in **§4.5**; plus the mis-named `test_stage2_ab_*` block and the orphaned `eval/orchestration.py` (§4.3).
- **Good intended architecture not yet converged?** `DetectionScene` IR (designed, 6+ parallel representations exist); `TrainingSurfaceResolver` (designed, `sft.py` still owns launch); unified runtime (the `harden-…-boundaries` change targets exactly this). Contracts are ahead of code.
- **Future constitution?** §6.

## Appendix B — Verification performed by this audit (not just agent assertion)
- `rg "from src.analysis|import src.analysis" src --glob '!src/analysis/**'` → **empty** (0 core importers of `src/analysis`).
- `find configs/analysis -name '*.yaml' | wc -l` → **91**; `find configs -name '*.yaml'` → **218** (42%).
- `rg -l "compact_full" src --glob '!src/analysis/**' | wc -l` → **22** active files (rename in-flight).
- `git worktree list` / `git cherry main <branch>` / `git branch --merged main` → **0 worktrees merged into main**.
- Independent agent cross-validation: the src-core agent and the analysis agent independently concluded `src/analysis` has zero core importers; the configs and docs agents independently found the 91/218 and `compact_full`-in-22-modules figures.
- **Cross-audit reconciliation (vs `codebase-analysis-codex.md`):** re-verified before folding in — package cycle reciprocates (`detection↔training`, `eval↔infer`; `common` = 69 importers); `stage1-detection-objectives/spec.md` lines 52/55/68/81 still mandate `compact_full`; `dataset.py:483` requires `compact` (prefix-rollin contradiction); `test_training_architecture_tiny_smoke.py` pins `compact_full_tiny.yaml`; `src/README.md` still SwiftSft/BaseCaptionDataset-era; `decode_policy:` authored in ~10 configs incl. 1 active. Rejected two of that audit's classifications (kept this report's): `src/training/surfaces.py` has 0 production importers (shadow, not "active/protect") and `scripts/pipelines/train_task_manager.py` has 0 doc refs (orphan, not "stable tool"). Self-corrected one error here: `src/eval/orchestration.py` is an orphan (0 importers), not a protected surface.

---

*This report is read-only diagnosis. No files were edited, deleted, staged, committed, or moved; no worktrees were pruned; no training/inference/eval jobs were run.*

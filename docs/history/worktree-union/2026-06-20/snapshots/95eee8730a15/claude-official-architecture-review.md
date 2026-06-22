# CoordExp — Independent Architecture & Maintenance Review

Author: Claude (Opus 4.8), acting as an independent architecture reviewer
Date: 2026-06-18
Stance: read-only static inspection of git-tracked contents. No edits, no commits, no training/inference/eval runs. Evidence is source layout, line counts, import-graph greps, config/doc/spec inspection, and git history.

This review deliberately treats the existing architecture material — `docs/architecture/INDEPENDENT_ARCHITECTURE_REVIEW.md` (2026-06-15), the 2026-05-31 simplification proposal, and the 2026-06-17 `REFACTORING_PROGRAM_CHARTER.md` (which itself merged prior `codebase-analysis-claude.md` / `codebase-analysis-codex.md`) — as **hypotheses to verify against the live code, not ground truth**. Where the code disagrees with those documents, I say so. The most useful thing I can add to that lineage is fresh verification and my own ranking of what actually matters.

---

## 0. Headline judgment

CoordExp is **not** an architectureless research dump, and it is **not** primarily a "too much code" problem. It is a well-governed *spine* wrapped in three distinguishable kinds of sediment, plus **two migrations the codebase has already silently decided but not finished**.

- The spine — config → data/geometry → template/IR → training surface → decode → eval → artifacts/metrics — is genuinely contract-defended and, in places, defended *in code* rather than only in docs.
- The sediment is what raises navigation cost and audit risk:
  1. **Compatibility sediment** — re-export shims and facades (`src/coord_tokens/*`, `src/trainers/metrics/mixins.py`, `src/eval/detection.py`, `src/common/detection_sequence.py`).
  2. **Aspirational sediment** — a parallel "training surface" architecture (`src/training/surfaces.py` + `src/training/pipelines/*`) that the real entrypoint never calls.
  3. **Research sediment** — the `analysis` estate (`src/analysis` ≈62k LOC, plus `configs/analysis`, `scripts/analysis`, `tests/analysis`) living *inside* the importable / spec / test surfaces instead of beside them.

My thesis differs slightly from the 2026-06-17 charter's "useful artifacts became permanent infrastructure." I'd put it: **the spine is fine; the cheapest high-leverage wins are to finish decisions the code has already made, then to physically separate research sediment from the runtime/contract surface.** Deletion-first is wrong; *converge-first* is right, and several convergences are 90% done.

---

## 1. My mental model of the codebase

I do not find the repo's own `docs/` layering (docs / openspec / progress) to be the most useful *architectural* model — that's a governance model. For reasoning about change risk, I divide the system into a **spine**, an **estate**, and a **scaffold**.

### 1.1 The spine (must stay correct and reproducible)

A single linear dataflow, each stage with a contract:

```
configs/*.yaml
  └─ src/config/loader.py + schema.py        (strict resolve; fail-fast unknown keys)
      └─ JSONL + image root contract          (docs/data/CONTRACT.md)
          └─ src/datasets/geometry.py         (geometry; no runtime resize, enforced)
              └─ DetectionScene  (src/detection/scene.py)   ← the live semantic IR
                  └─ DetectionSequenceTemplate (src/detection/template.py)  render/parse/spans
                      └─ training surface:
                           • Stage-1 teacher forcing  (src/detection/runtime.py, objective.py, loss.py)
                           • Stage-2 rollout correction (src/trainers/stage2_*; src/training/stage2/*)
                      └─ decode  (src/infer/runtime.py, backend.py, backend_vllm_server.py)
                          └─ eval (src/eval/detection_{coco,lvis,f1ish,geometry,duplicate_guard,orchestrator}.py)
                              └─ artifacts + provenance (src/bootstrap/*, src/infer/artifacts.py,
                                                          src/eval/artifacts.py, src/metrics/events.py)
```

If you protect exactly one thing in this repo, protect the **right-hand tail**: the artifact/provenance/metric contract. Everything upstream exists to feed `gt_vs_pred(_scored)`, `resolved_config.json`, `effective_runtime.json`, the manifests, and `MetricEvent`. That tail is what makes results believable.

### 1.2 The estate (research exploration — should sit *beside* the spine, not inside it)

`src/analysis/` (89 files, ≈62k LOC — about one third of `src/`), `configs/analysis/` (93 files), `scripts/analysis/` (62 files), `tests/analysis/` (53 files), and most of `progress/` (228 files). Verified runtime-isolated: `rg "src\.analysis" src --glob '!src/analysis/**'` returns **nothing** — no core module imports the analysis package. This is a parallel research codebase that happens to live under importable `src/` and the spec/test trees.

### 1.3 The scaffold (governance + history)

`docs/` (143 files, a genuinely good routed spine), `openspec/` (621 files: **36 stable specs, but 86 archived changes / 584 files under `changes/`**), and the `progress/` history. The scaffold is high quality but oversized; `openspec/changes/archive` is a 500-file proposal graveyard that inflates every repo-wide search.

**Why this model and not the docs' own model:** the docs model answers "where is the truth written?" My model answers "what breaks research validity if I touch it?" — which is the question an architecture review must answer. The spine carries correctness; the estate carries token/navigation cost but low runtime risk; the scaffold carries governance.

---

## 2. Strengths — what is already well-designed and should be preserved

These are not generic compliments; each is something I verified and would actively defend against refactors.

| Strength | Evidence | Why it matters |
|---|---|---|
| **Routed documentation spine** | `docs/PROJECT_CONTEXT.md` → `SYSTEM_OVERVIEW.md` → `IMPLEMENTATION_MAP.md` → `catalog.yaml`; per-file `doc_id`/`status`/`updated` frontmatter | Rare in research code. You can reconstruct current behavior without reading source first. Protect the routing discipline. |
| **Artifact / provenance contract** | `src/bootstrap/{pipeline_manifest,run_metadata,experiment_manifest}.py`, `src/infer/artifacts.py`, `src/eval/artifacts.py`, `decode_policy_fingerprint`, `resolved_config.path`, score-provenance, guarded companions | This is the reproducibility backbone. It is treated as a contract with tests (`tests/test_artifact_contract_docs.py`, `test_run_manifest_files.py`). Do not let any refactor reconstruct artifact payloads ad hoc. |
| **Geometry/no-resize invariant defended in code** | `do_resize=False` enforced in `src/detection/dataset.py`, `src/trainers/rollout_aligned_evaluator.py`, and a **fail-fast preflight** in `src/trainers/rollout_matching/preflight.py` ("`do_resize=false` is required for Stage-2 server preflight") | The single most important grounding invariant is not just documented — it raises at runtime. Excellent. |
| **`DetectionScene` as a single live semantic IR** | `src/detection/scene.py` imported by `detection/{__init__,data,dataset,template}.py`, `teacher_forcing/target_builder.py`, **and** Stage-2 `trainers/rollout_correction/projections.py`, `stage2_rollout_correction_impl.py` | A genuine recent win: one image's detection semantics now feed both Stage-1 and Stage-2 projections. The 2026-06-15 review proposed this; the code shipped it. (See §3.4 for the unfinished half.) |
| **Decomposed evaluator** | `src/eval/` split into `detection_{records,geometry,coco,lvis,duplicate_guard,f1ish,orchestrator}.py` behind a thin `detection.py` facade | This is what *good* decomposition looks like here: each module is deep (real COCO/LVIS/F1 logic) behind a small interface, and the facade is honest about being a facade. Leave it alone. |
| **Typed metric identity** | `src/metrics/events.py::MetricEvent` + `flatten_metric_events` + reserved-alias collision guard; `src/training/observability/events.py::DiagnosticEvent` | Prevents flat dashboards from silently lying about denominator/reducer/semantics. Keep expanding adoption rather than redesigning. |
| **Strict config resolution** | `src/config/loader.py` + `schema.py` with fail-fast unknown keys (`tests/test_training_config_strict_unknown_keys.py`) | Strictness is the right default for a config-first stack: typos and retired keys fail loudly. |
| **Test density on the spine** | 339 test files; contract tests for IR (`test_detection_scene_contract.py`), templates, artifacts, geometry invariants, Stage-2 | The spine's interfaces are also its test surface. This is the asset that makes the rest of my recommendations *safe*. |

---

## 3. Problems — where the hierarchy is unclear, shallow, coupled, or hard to audit

Ordered by how much they actually impede research, not by size.

### 3.1 `src/training/` vs `src/trainers/` — two packages, one job, a real cycle

This is the single most confusing structural fact in the repo.

- Both are top-level packages about training. They **mutually import**: `src/training → src.trainers` (4 files) and `src/trainers → src.training` (13 files). A real bidirectional cycle.
- **Both contain a `teacher_forcing` subpackage.** `src/training/teacher_forcing/` (ir, vocab, probabilities, roles, validation, metrics) is the teacher-forcing *IR/vocabulary* layer; `src/trainers/teacher_forcing/` (module_registry, objective_atoms, modules/{token_ce, stage2_trie_ce, residual_set_correction, schema_format_ce}, forwards, geometry) is the *executable objective* layer. They are not literal duplicates, but a newcomer cannot tell which `teacher_forcing` to open, and the IDE cannot either.

The split has no stable conceptual boundary you can state in one sentence. That is the test of a bad seam. (Contrast: `src/detection` vs `src/infer` vs `src/eval` *do* have one-sentence boundaries.)

### 3.2 The launch monolith and its siblings are still monoliths

Verified line counts (the charter's numbers are essentially unchanged — I confirmed `sft.py` did **not** shrink despite recent "adapter surface" commits):

| File | LOC | Top-level symbols | Role |
|---|---:|---:|---|
| `src/trainers/stage2_rollout_correction_impl.py` | 5,923 | — | Stage-2 trainer impl (highest-risk active code) |
| `src/config/schema.py` | 4,831 | 54 classes / 108 dataclass markers | Config schema monolith |
| `src/trainers/stage2_rollout_runtime.py` | 4,585 | — | Stage-2 rollout runtime |
| `src/trainers/rollout_correction/target_builder.py` | 4,382 | — | Stage-2 target/loss construction |
| `src/sft.py` | 4,340 | **76 module-level functions** | Launch script carrying library-scale concerns |
| `src/infer/runtime.py` | 3,064 | — | Inference runtime |

`src/sft.py` with 76 top-level functions is the place where config mutation, trainer selection, dataset construction, packing/cache preflight, detection runtime handoff, Stage-2 projection, and manifest writing all meet. It is the highest-*locality*-risk file: a refactor anywhere upstream tends to surface here. The Stage-2 trio (impl + runtime + target_builder ≈ 14.9k LOC) is the highest-*semantics*-risk code, because that's where loss meaning lives.

### 3.3 Shadow training surfaces are maintained only for their own tests

`src/training/surfaces.py` (690 LOC) + `src/training/pipelines/{stage1_json_ce, stage1_compact_trie_ce, stage2_rollout_correction}.py` (26 LOC each, descriptors) present a clean top-level domain model (`run/surface/data/template/supervision/objectives/observability/artifacts/runtime`).

I verified who actually imports them: **only the pipeline package itself and three test files** (`tests/test_training_surface_resolver.py`, `tests/test_objective_profile_resolution.py`, `tests/helpers/training_architecture_fixture_builder.py`). `src/sft.py` contains **zero** references to `surfaces`/`TrainingSurfaceResolver`/`pipelines`. `rg "TrainingSurfaceResolver|training\.pipelines" src --glob '!src/training/**'` → nothing.

This is worse than "shallow": it is a **parallel architecture validated only by tests it ships with**. Apply the deletion test — deleting `surfaces.py` + `pipelines/` concentrates *no* runtime complexity, because no runtime path depends on it. It is a hypothesis frozen in code. The honest options are *promote it to be the real launch owner* or *delete it*; keeping it indefinitely is the worst outcome because it makes the architecture look more converged than it is and costs maintenance on every schema change.

### 3.4 The data-IR migration is 90% done and stalled at the cleanup step

The code has **already chosen** `DetectionScene` over the legacy `DetectionDocument` — but did not finish:

- `src/detection/scene.py`: `DetectionScene`, `DetectionObject`, `DetectionGeometry` — live (§2).
- `src/detection/ir.py`: `DetectionDocument`, `DetectionObjectEntry`, **and a second class also named `DetectionGeometry`** — imported by **tests only** (`test_detection_ir_contract.py`, `test_detection_scene_contract.py`). Zero runtime importers.

So `src/detection/ir.py` is orphaned production code kept alive by a contract test that tests itself, and it collides on the `DetectionGeometry` name with the live IR. This is exactly the "added a new IR without retiring the old one" risk the 2026-06-15 review warned about — and the 2026-06-15 review *did not know it had already happened*, because it treated `DetectionScene` as a proposal and `DetectionDocument` as the incumbent. The code moved past the doc. Finishing this is a near-zero-risk deletion.

(Related but separate: `src/detection/data.py::{RawDetectionRow, NormalizedDetectionSample}` and `src/common/schemas.py::ConversationRecord` are still distinct representations. Those are *legitimate* layered views around `DetectionScene` — don't collapse them. The orphan is specifically `ir.py`.)

### 3.5 Geometry math is the documented single-owner — but isn't

`CLAUDE.md` states a hard rule: "Route bbox math through `src/datasets/geometry.py`." The canonical `box_iou_xyxy` is indeed there. But IoU is independently reimplemented across the active spine:

- `src/trainers/rollout_matching/matching.py` (`_bbox_iou_xyxy`, `_mask_iou_norm1000`)
- `src/trainers/rollout_correction/{target_builder,residual_set,ul_consensus}.py` (`_bbox_iou_norm1000_xyxy`, `_bbox_iou`×2)
- `src/trainers/stage2_rollout_correction_impl.py` (`_bbox_smoothl1_ciou_loss`, `_stage2_bbox_iou_from_gt_objects`)
- `src/training/stage2/{assignment,duplicate_filter}.py` (`_bbox_iou`×2)
- `src/eval/detection_geometry.py` (`_bbox_iou`, `_segm_iou`)
- `src/vis/gt_vs_pred.py` (`_bbox_iou`)
- `src/detection/coord_soft_targets.py` (`_replaced_slot_iou`, `_replaced_slot_ciou`)

That is ~12 private IoU implementations, each with a **different and implicit coordinate-frame assumption** (xyxy-pixel vs norm1000 vs `ObjectBBox` vs `GTObject`). The 2026-06-17 charter listed "collapse local geometry duplicate helpers" as *low priority*. **I disagree — I rank this higher**, because geometry is the load-bearing research invariant, and the danger isn't "many functions," it's "many functions whose coordinate frame is encoded only in a prefix like `_norm1000_`." That's precisely where a silent grounding bug (a norm1000 box compared against a pixel box) would hide and *still produce a plausible number*. There is a defensible reason for some duplication (eval IoU is COCO-pixel semantics; training IoU is differentiable norm1000 CIoU) — so the fix is not "one function" but "**one owner per coordinate frame, with the frame in the type, not the function name**."

### 3.6 Compatibility sediment: `coord_tokens` shim and metric/template facades

- `src/coord_tokens/` is now **51 LOC of pure re-export** (`codec.py` is literally `from src.tokens.coord.codec import *`). The real owner is `src/tokens/` (1,020 LOC). Yet the shim is still imported by **live trainer code**: `src/trainers/losses/coord_soft_ce_w1.py`, `trainers/metrics/{batch_contract,coord_losses,aggregate_tokens}.py`, `rollout_aligned_evaluator.py`, `rollout_matching/parsing.py`. Deletion test: removing the shim concentrates nothing — it's a mechanical import rewrite in ~10 callers. Pure pass-through.
- `src/trainers/metrics/mixins.py` (re-export facade), `src/eval/detection.py` (import-compatible facade), `src/common/detection_sequence.py` (compatibility facade for salvage/helper formats). The eval and detection_sequence facades are *honest and useful* (they serve real salvage/compat needs). The `coord_tokens` shim and `mixins` re-export earn nothing.

### 3.7 The `compact_full` → semantic-template migration is mid-flight (but partly sanctioned)

`compact_full` still appears across active surfaces: heaviest in Stage-2 code (`src/training/stage2/rollout_codec.py` ×25, `trainers/rollout_correction/{rollout_views,target_builder}.py`, `stage2_rollout_correction_impl.py`), plus Stage-1 docs (`docs/training/STAGE1_OBJECTIVE.md` ×10, `README.md` ×8) and tests/configs. **Nuance the charter is right about:** most src usage is in the Stage-2 rollout family, which is an explicitly *whitelisted* compatibility context. So this is less "drift" than "an incomplete rename with a legitimate tail." The real, fixable inconsistency is in **Stage-1 docs/configs/tests** still teaching `compact_full` while live code validates semantic IDs (`compact`, `compact_box_closed`, …). Recent commits (`6b8c6a75 Add semantic detection template variants`, `56935d42 Fix compact SFT geometry-first rendering`) show this is the active front.

### 3.8 OpenSpec is being used past its own charter

`docs/PROJECT_CONTEXT.md` says `openspec/specs/` is for "stable compatibility contracts only." But among the 36 stable specs sit clear **study/ablation** specs: `duplication-collapse-analysis-study`, `rollout-fn-factor-analysis-study`, `unmatched-proposal-verifier-study`, `compact-template-field-order-ablation`; plus **retired-surface** specs (`stage2-ab-training`, `rollout-matching-sft`) kept for rejection tests. Combined with 86 archived changes / 584 files under `openspec/changes/`, OpenSpec has become the de-facto research-record system. That's the same "research sediment in a contract surface" pattern as `src/analysis`, just in the governance layer.

### 3.9 Package cycles (structural, lower urgency)

Verified bidirectional edges: `detection ↔ training` (4/6), `eval ↔ infer` (5/2), `metrics ↔ trainers` (2/14), `training ↔ trainers` (4/13). High fan-in to `src/common` (66 importers) and `src/detection` (21). These make small changes non-local, but they are a *consequence* of §3.1–§3.4, not an independent problem. Cutting them before resolving the IR/surface/teacher_forcing questions would be premature.

---

## 4. Development recommendations

### 4.1 Maintain as stable (do not refactor without docs/OpenSpec + tests)

- The artifact/provenance tail: `src/bootstrap/*`, `src/infer/artifacts.py`, `src/eval/artifacts.py`, `decode_policy_fingerprint`, `resolved_config(.path)`, `effective_runtime.json`, manifests, `gt_vs_pred(_scored)` schema.
- Geometry invariants and the no-resize fail-fast.
- `DetectionScene` as the central IR (it just won; stabilize it).
- The decomposed `src/eval/*` modules and the `MetricEvent`/`DiagnosticEvent` identity layer.
- The routed `docs/` spine and `catalog.yaml`.
- Stage-2 **loss/target semantics** in `target_builder.py` (refactor shape, never meaning).
- The recursive-detection / ET-RMP comparator family (`src/detection/{objective,loss,rollin}.py`) — preserve per the charter; it's research contrast, not bloat.

### 4.2 Refactor / simplify (finish what's already decided first)

1. **Delete `src/detection/ir.py`** after re-pointing its two tests at `scene.py`. Removes the duplicate `DetectionGeometry` name and an orphan IR. (§3.4)
2. **Inline the `src/coord_tokens` shim** into its ~10 callers and remove the package. (§3.6)
3. **Resolve the shadow surfaces** (§3.3): pick *promote* (route one Stage-1 smoke through a pipeline behind a `--cfg-only` parity check) or *delete*. Do not leave them as test-only.
4. **Consolidate IoU/coordinate-frame math** behind `src/datasets/geometry.py` (or a `src/common/geometry/` owner), with the coordinate frame expressed in the *type* (`Norm1000Box` vs `PixelXYXYBox`), not the function name. Keep eval-pixel vs train-norm1000 as two named, typed entry points. (§3.5)
5. **Split `schema.py` by domain** (data / template / objective / packing / stage2 / eval) preserving strict unknown-key behavior and import surface. (54 classes is a clean seam set.)
6. **Extract `sft.py` library concerns** (preflight, fingerprinting, manifest assembly, template metadata, runtime-payload assembly) into modules under `src/bootstrap`/`src/training`, keeping `python -m src.sft --config …` and **byte-identical fingerprints**.

### 4.3 Document better

- A one-page **lifecycle registry** (active / active-research / preserved-comparator / compatibility / retired) — I agree with the charter that this is the highest-leverage doc artifact. Make it machine-checkable.
- A **parser-policy matrix** (strict template parse vs compact-salvage vs CoordJSON-salvage vs eval ingestion) so "parser" is never ambiguous.
- A one-sentence boundary statement for `training` vs `trainers` — *or* merge them (see §6).

### 4.4 Test more directly

- A **geometry coordinate-frame parity test**: feed the same box through every retained IoU entry point and assert agreement within frame. This is the test that would catch §3.5 silently.
- A **train/infer template parity test** on `DetectionScene` round-trips (render → parse → scene) for each active semantic template ID, replacing scattered `compact_full` assertions.
- An **import-cycle guard** scoped to one edge at a time (start with `eval → infer`), report-only first.

### 4.5 Avoid changing

- Salvage parsers (`src/common/prediction_parsing.py`, `coord_standardizer.py`) — inference diagnostics need them.
- The `rollout_matching.*` runtime namespace — still load-bearing for Stage-2; renaming it now is high blast-radius for no research payoff.
- Artifact filenames (`gt_vs_pred*.jsonl`) — terse but accurate; rename only as a separate, tested artifact-contract decision.
- Stage-2 internals before golden IR fixtures exist.

---

## 5. Research-direction advice — where new ideas should live

The repo's biggest long-term risk is the one it already demonstrates: **a good experiment becomes permanent infrastructure by default.** The fix is a decision tree for *where a new idea lands first*:

| New idea is… | First home | Not here |
|---|---|---|
| A hyperparameter / data-ordering / template-variant sweep | a `configs/` leaf + a `progress/` note | not a new `sft.py` flag, not a spec |
| A new training objective | a module in `src/trainers/teacher_forcing/modules/` registered via `module_registry.py` | not inline in `stage2_*_impl.py` |
| A new analysis/probe study | `src/analysis/<study>/` **with a lifecycle label**, ideally behind an import gate so it can't leak into runtime | not a new `openspec/specs/*-study` |
| A new artifact/metric | extend `MetricEvent` / `src/*/artifacts.py` with a doc + contract test | not a flat key in a trainer |
| A stable contract change (loss/eval/schema/artifact semantics) | OpenSpec change → docs | not `progress/` as the authority |
| A speculative redesign | a worktree branch + a superpower plan | not a parallel package in `src/` (that is how shadow surfaces happened) |

The deepest, healthiest seam for *new research* is the objective module registry (`src/trainers/teacher_forcing/module_registry.py` + `objective_atoms.py`). New supervision ideas should be expressed as typed atoms there, not as new branches in the monoliths. That is the one place where "add an experiment" already means "add a small module behind a stable interface."

To keep fast experimentation compatible with long-term cleanliness: adopt the charter's **ratchet** — a cheap, initially report-only hygiene check that flags (a) `src/analysis` files without a lifecycle label, (b) retired template IDs in active specs/tests, (c) study specs under stable OpenSpec, (d) doc/catalog path references that don't resolve. New complexity is allowed; *unlabeled* new complexity is not.

---

## 6. Priority plan

**Tier 0 — finish decisions the code already made (days, near-zero risk).** Highest leverage per unit risk in the whole repo.
1. Delete orphan `src/detection/ir.py`; re-point its tests. (§3.4)
2. Inline + remove `src/coord_tokens`. (§3.6)
3. Decide shadow surfaces: promote one path or delete `surfaces.py` + `pipelines/`. (§3.3)
4. Finish Stage-1 `compact_full` → semantic-ID cleanup in `docs/training/*`, Stage-1 configs, and Stage-1 tests (leave the sanctioned Stage-2 tail). (§3.7)

**Tier 1 — separate research sediment (weeks, low runtime risk, biggest navigation win).**
5. Lifecycle-label and **physically relocate** the `analysis` estate (`src/analysis`, `configs/analysis`, `scripts/analysis`) out of the apparent production surface, preserving progress/artifact pointers. It's runtime-isolated, so the only risk is research-reproducibility bookkeeping — which is exactly what the lifecycle label protects.
6. Demote study/retired specs out of `openspec/specs/`; keep them as progress/history. (§3.8)
7. Add the ratchet hygiene check (report-only). (§5)

**Tier 2 — correctness leverage (weeks, medium risk).**
8. Consolidate geometry/IoU behind typed coordinate-frame entry points + the parity test. (§3.5) I'd promote this above generic monolith-splitting because it touches research validity.

**Tier 3 — locality (weeks–months, medium/high risk, needs the Tier-0/1 tests in place).**
9. Split `schema.py` by domain.
10. Extract `sft.py` helpers (fingerprint parity required).
11. Resolve `training` vs `trainers`: my recommendation is to **merge toward `src/training/` as the package and treat `src/trainers/` as the executable objective/loss implementations namespace** — but only after #3, because the shadow `surfaces.py` is the thing pretending to be the top of `src/training/`. Do not start here.

**Tier 4 — structural (later, high risk).**
12. Decompose Stage-2 internals (golden IR fixtures first; `target_builder.py` last).
13. Cut package cycles one edge at a time (`eval → infer` first).

**Explicitly not worth it / avoid:**
- A broad rename to the proposed `DetectionScene`-family vocabulary across configs/artifacts now — the operational blast radius (artifact files, config namespaces, dashboards) dwarfs the architectural payoff. Rename *internal symbols* opportunistically; freeze *external names*.
- Deleting salvage parsers or the `rollout_matching.*` namespace.
- Chasing `rg compact_full → 0` repo-wide (the charter itself warns against this; the Stage-2 tail is legitimate).
- Whole-repo acyclicity in one pass.

---

## 7. Self-reflection

**Assumptions I made.**
- *Static-only.* I never executed training/inference/eval, so I cannot confirm that the ~12 IoU implementations numerically agree, that configs resolve, or that Stage-2 loss semantics are correct. I treated import-graph greps as a proxy for the real dependency graph.
- *Greps under-count dynamic edges.* String-keyed registries, `getattr` dispatch, and config-driven class selection can hide dependencies my `rg`-based cycle/import counts miss. The "shadow surfaces are test-only" claim is the one I'd most want to double-check against a runtime trace — though `sft.py` having zero textual references to them is strong evidence.
- *I trusted recency signals.* I used `git log` and importer sets to decide what's "live," e.g., concluding `DetectionScene` won over `DetectionDocument`. If there's an unmerged worktree where `ir.py` is still primary, that conclusion is local to `main`.

**What I inspected first, and why.** The authority spine (`PROJECT_CONTEXT` → `SYSTEM_OVERVIEW` → `IMPLEMENTATION_MAP`), then the *existing* architecture docs — specifically to avoid duplicating them and to find places to verify or contradict. Then I went straight to cheap, high-signal measurements: LOC of named monoliths, importer sets for the contested modules (analysis, coord_tokens, surfaces, ir/scene), the `compact_full`/IoU/cycle greps, and OpenSpec counts. I prioritized *claims that, if false, would change the recommendation* (did `sft.py` shrink? is `DetectionScene` live? are shadow surfaces reachable?).

**Where a strong reviewer might disagree with me.**
- *Geometry priority.* I elevated IoU consolidation above monolith-splitting; the charter ranked it low. A reviewer who has verified the IoU functions agree numerically would call my concern theoretical.
- *Analysis relocation.* I recommend moving `src/analysis` out of `src/`; others (charter Q1) argue lifecycle metadata + import gating is enough without the churn of moving 89 files.
- *training/trainers merge.* This is a large, churny refactor over two cyclically-coupled packages; a reasonable reviewer would say the cost outweighs the navigation benefit and leave them split with a documented boundary instead.
- *Shadow surfaces.* I'd delete-or-promote now; the original authors clearly intend to promote, and may reasonably want to keep the descriptors as a design anchor.

**What I'm most confident I detected** (directly measured, not inferred): the monolith sizes; `coord_tokens` being a pure re-export shim; `ir.py`/`DetectionDocument` being a test-only orphan with a duplicate `DetectionGeometry` name; the shadow surfaces being imported only by tests; `DetectionScene` being live across Stage-1 and Stage-2; the `training ↔ trainers` cycle with dual `teacher_forcing`; OpenSpec containing study specs; the IoU proliferation.

**What I likely missed.** Anything requiring execution: semantic loss correctness, DDP/post-rollout packing behavior, cache/fingerprint stability under refactor, numerical geometry agreement, performance hotspots, and whether the 339 tests actually pass on `main`. I also did not deeply read the `analysis` estate — I classified it by isolation and size, not by which studies are still scientifically load-bearing, which is precisely the judgment the lifecycle registry exists to capture and which only the research owner can make.

---

### One-line bottom line

CoordExp's spine is sound and well-defended; spend the next cycle **finishing the migrations the code has already chosen** (retire `ir.py`, kill the `coord_tokens` shim, resolve the test-only shadow surfaces, close the Stage-1 `compact_full` gap), then **move the research estate out of the runtime/contract surface** behind a lifecycle ratchet — and treat geometry-frame consolidation as a correctness task, not cosmetic cleanup. Save the monolith and package-cycle work for after those cheap, high-leverage convergences land.

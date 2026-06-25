# Critical Review — Stage-1 Static Packing Exact Atom-Position Mapping Plan

- **Plan reviewed:** `docs/superpowers/plans/2026-06-25-stage1-static-packing-exact-atom-position-mapping.md`
- **Worktree scope:** `.worktrees/ledger-auxiliary-loss/` (branch `codex/ledger-auxiliary-loss`, HEAD `43de325b`)
- **Date:** 2026-06-25
- **Method:** 4 parallel read-only reviewer subagents, one per lane (data plumbing / forward-pass geometry / config-runtime guards / plan methodology), each required to ground findings in `path:line` evidence; two headline blockers re-verified directly by the coordinator.
- **Evidence scope:** Static review only. No tests executed; no files modified. The ms-swift `swift` package is **not importable** in this environment and is not vendored, so the cross-segment attention-mask behavior (Blocker B4) could not be empirically confirmed and remains the single largest open risk.

---

## Verdict: NOT READY FOR APPROVAL

The plan is unusually detailed and several hard parts are correct (notably the visual-token geometry math — see "Verified Correct"). But it has **5 blocking defects**, two of which mean the plan **cannot reach green in its written order** and one of which (attention leakage) is a potential **silent correctness corruption** that the plan asserts but never verifies. The plan's own Self-Review claim *"No task relies on undefined future decisions"* is false.

Recommend: **request changes**, do not implement as written.

Confidence is high on the config/sequencing blockers (multiple independent agents + direct coordinator verification with quoted source). The two deepest correctness risks (attention isolation, loss-scale equivalence) are well-argued from real code but one depends on un-inspectable upstream collator behavior.

---

## CRITICAL — blockers (must fix before approval)

### C1. The plan never touches `_detection_validate_teacher_forcing_coverage_ledger_contract`, which independently blocks the whole feature
*(Found independently by Lane C and Lane D; re-verified directly by coordinator.)*

- **Claim in plan:** Loosening `_detection_validate_packing_runtime_contract` (Task 6 Step 4) + the runtime mirror is enough to let Stage-1 `research_teacher_forcing` static-pack.
- **Evidence:** A *second, separate* validator runs for every coverage-ledger config and is unmentioned in the plan (grep of the plan for this function: **0 hits**). `src/config/schema.py:3170-3189`:
  ```
  if _detection_runtime_bool(training, "packing"):
      raise ValueError("...coverage_ledger.enabled=true requires training.packing=false")
  if _detection_runtime_bool(training, "eval_packing"):
      raise ValueError("...coverage_ledger.enabled=true requires training.eval_packing=false")
  if packing.static_packing:
      raise ValueError("...coverage_ledger.enabled=true requires packing.static_packing=false")
  ```
  Invoked unconditionally at `src/config/schema.py:4977`, **before** the guard the plan does loosen (`:4989`). Both ledger configs enable it: `configs/.../smoke/coverage_ledger_closed_hard_sft_128.yaml:48` (`enabled: true`) and `configs/.../prod/coverage_ledger_closed_hard_sft.yaml:52` (`enabled: true`).
- **Impact:** After implementing Task 6 exactly as written, `DetectionTrainingConfig.from_mapping` still raises `"...requires packing.static_packing=false"`. Task 6 Step 1's positive test, all of Task 7's config tests, and Task 8's preflight fail. The plan's central goal is unreachable.
- **Fix:** Add this function to the File Map and loosen its `packing`/`static_packing` (and, per C2, `eval_packing`) branches in lockstep with the generic guard, keeping `padding_free_packed` rejected. Note it also enforces `per_device_train_batch_size == 1` (configs satisfy this).

### C2. `eval_packing: true` is required by configs/tests but never un-rejected in any guard
*(Lane D; eval-safety also flagged by Lane C; re-verified by coordinator.)*

- **Evidence:** `eval_packing=true` is rejected independently of `packing` in the generic guard (`src/config/schema.py:3065-3070`), the ledger guard (`:3175-3178`), and the runtime mirror (`src/detection/runtime.py:270-275`). Task 6 Steps 4-5 rewrite only the `packing` and `padding_free_packed` branches. Yet Task 6 Step 1 asserts `cfg.training["eval_packing"] is True` survives `from_mapping` (plan line ~1314), and Task 7 sets and asserts `eval_packing: true` (plan lines ~1477, ~1504).
- **Impact:** Internal contradiction; configs remain unloadable even after C1 is fixed. Separately, **eval-path safety is unverified**: `eval_packing` appears only in `loader.py`, `enrichers.py`, `sft.py`, `schema.py`, `runtime.py` — **not under `src/eval/`** — so it is unclear the eval/generation path routes coverage-ledger through the same packed bridge that training does.
- **Fix:** Decide scope explicitly. If eval packing is in scope: loosen `eval_packing` in all three guards *and* add an eval-path packed test. If not: drop `eval_packing: true` from Task 7 configs and assertions, and keep eval unpacked for v1 (recommended for a first cut).

### C3. Task 2 imports a symbol that Task 3 creates → Task 2 commits broken code
*(Spotted by coordinator pre-dispatch; confirmed by Lane A and Lane D.)*

- **Evidence:** Task 2 Step 6 adds `from src.training.coverage_ledger.sidecars import shift_coverage_ledger_sidecar` to `src/data_collators/enrichers.py`, but `shift_coverage_ledger_sidecar` is first defined only in Task 3 Step 4. Today `src/training/coverage_ledger/sidecars.py:238` exports only `["CoverageLedgerObjectEntry", "CoverageLedgerSidecar"]`. The import is module-level and the collator import chain is eager (`src/data_collators/batch_extras_collator.py:18-27`).
- **Impact:** Task 2 Step 7 (`pytest tests/test_teacher_forcing_sidecar_bridge.py …`) raises `ImportError` at collection; Task 2 Step 8 commits a red tree — a direct violation of the plan's own TDD/commit contract.
- **Fix:** Reorder so every symbol used in task N is created by task ≤ N — move the sidecar/visual-region shift helpers (Task 3 Steps 4-5) ahead of Task 2, or merge Tasks 2+3, or use a local import. Audit all tasks for other forward dependencies.

### C4. "No cross-segment attention leakage" is asserted but never verified — likely UNSOUND as scoped
*(Lane B rates CRITICAL; Lane D flags as asserted-not-tested. Could not be empirically confirmed — `swift` not importable.)*

- **Claim in plan:** With `packing.padding_free_packed=false` while `template.packing=True`/`template.padding_free=True`, packed segments are isolated, so coverage-ledger reads of `final_hidden_states` at shifted positions are segment-local ("batch-equivalent").
- **Evidence:** The Qwen3-VL **text** decoder builds attention purely from the passed mask + `text_position_ids` and consumes **no** varlen/`cu_seqlens` metadata (only the vision tower does). `prepare_forward_inputs(..., packing_enabled=True)` validates **only** the 4-row `position_ids` and passes **no** block-diagonal mask (`src/trainers/teacher_forcing/forwards.py:39-48`). With `padding_free_packed=false`, an ordinary all-ones mask over the concatenated row yields **one causal mask spanning the whole pack** → segment B attends to segment A. No repo test asserts block-diagonal masking for static packing (grep of `tests/` = none).
- **Impact:** If the ms-swift static (non-padding-free) collator does not emit a segment-aware mask, every packed segment after the first is contaminated; both the coverage-ledger auxiliary loss and the main CE train on leaked cross-segment context — **silent corruption, not a crash**. This invalidates the "batch-equivalent" premise the entire feature rests on.
- **Fix:** Make block-diagonal isolation an explicit, *tested* invariant. Add a numerical forward test: perturb segment A's input tokens and assert segment B's `final_hidden_states` are unchanged (and vice versa). Do this against a real two-segment static pack before any reliance on the property. If the static collator cannot isolate without padding-free/FA2-varlen, this feature is not viable as scoped — which directly contradicts the plan's decision to defer the varlen runtime gate.

### C5. Per-segment ledger loss is SUMMED while the main CE is MEAN-reduced → not batch-equivalent
*(Lane B CRITICAL; Lane D completeness item (b).)*

- **Evidence:** Main teacher-forcing CE is mean-reduced over atoms (`src/training/objectives/teacher_forcing.py:134-135,233`) and `ObjectiveRunner` sums *per-objective*, not per-sample (`src/training/objectives/runner.py:167-173`). Each coverage-ledger segment loss is itself a fixed mean BCE (`src/training/coverage_ledger/loss.py:229-233,258-261`). The plan adds **N** segment means together (`loss = loss + Σ weighted_loss_i`) with no normalization, and the Task 5 test bakes this in (`result.loss == 0.5` for two `0.25` segments, plan lines ~1130).
- **Impact:** Packing N samples into one row inflates the auxiliary gradient ~N× relative to both the same-row CE and to an unpacked N-sample batch. The effective `coverage_weight`/`region_anchor_weight` then scales with average pack fill and interacts with `effective_batch_size`/grad-accum — silent rescaling of the loss the experiment is studying.
- **Fix:** Decide and document the intended reduction (divide by segment count / object count to match the unpacked mean). Add an equivalence test: total loss of {2 samples packed in 1 row} vs {2 samples as a 2-row unpacked batch} must match.

---

## IMPORTANT

### I1. Task 4 rewrite removes `_with_batch_index`, regressing non-packed multi-sample batches *(Lane A; current behavior confirmed by Lane D)*
The dataset authors `batch_index=0` for every sample (`src/detection/teacher_forcing/target_builder.py:314,340`). Current code calls `_with_batch_index(...)` to assign sequential rows (`src/trainers/metrics/teacher_forcing.py:124-126`). The plan's rewrite reads `atom.batch_index` directly and explicitly says "Do not call `_with_batch_index`", so a non-packed batch of N>1 maps **every** sample to row 0, corrupting `sample_id_to_batch_index` and the coordinate mapper (`src/training/bridge/coordinate_mapper.py:42`). The plan adds no non-packed multi-sample regression test. **Fix:** gate the bypass on `ir.metadata.get("packed")` (set by `shift_teacher_forcing_target_ir`); keep `_with_batch_index` for the unpacked path; add a 2-sample non-packed test asserting `{"a":0,"b":1}`.

### I2. `packed_segment_offsets` is written into `collated` as an unregistered model-input key *(Lane A)*
It is absent from `DETECTION_MODEL_INPUT_KEYS`/`REGISTERED_DETECTION_SIDECAR_KEYS`/`TRAINER_BATCH_EXTRA_KEYS` (`src/detection/dataset.py:66-123`) and from `_STATIC_KEY_DISPOSITIONS`, so `strip_non_model_detection_sidecars` (`:1375`) / `classify_backend_key` (`src/training/encoding/model_inputs.py:124-126`) fail-fast for any non-teacher-forcing trainer using the same `build_dataset_metrics_collator` (`src/sft.py:4219`). It only survives in the TF path because the mixin pops extras first. **Fix:** register the key in `TRAINER_BATCH_EXTRA_KEYS`/`SIDECAR_ONLY_KEYS` so it is uniformly stripped, and confirm `pop_batch_extras` (not just `BATCH_EXTRAS_KEYS`) pops it.

### I3. Stage-2 boundary test regex will not match the real error message *(Lane C)*
Task 6 Step 2 expects `pytest.raises(ValueError, match=r"Stage-2.*teacher-forcing.*packing")`, but the real guard message is `f"objective.id={...} with pipeline.id=stage2_rollout_correction rejects training.packing=true ..."` (`src/training_runtime/preflight.py:134-139`) — contains neither "Stage-2" nor "teacher-forcing". Also the guard only fires when `post_rollout_packing_owner == "trainer"` and `variant == "stage2_rollout_correction"` (`:128-131`), so the fixture must produce exactly that plan. **Fix:** match `r"rejects training\.packing=true"` (or change the source message) and verify the fixture's runtime plan.

### I4. Dormant landmine: `resolve_static_sft_training_mode` hard-rejects `objective_variant="teacher_forcing"` *(Lane C)*
`_validate_stage1_static_packing_policy` works today only by accident: `research_teacher_forcing` has no `variant` attribute (`src/config/schema.py:4266-4276`), so `objective_variant` resolves to `None` and falls through to an eligible branch (`src/detection/packing.py:138-142`). Task 6 Step 6's instruction to "explicitly check the pipeline/objective id" risks feeding `"research_teacher_forcing"` into `resolve_static_sft_training_mode`, which raises `"Unsupported detection training mode"` (`packing.py:145-154`). `src/detection/packing.py` is absent from the File Map. **Fix:** decide explicitly how teacher-forcing maps for eligibility; do not route the objective id into `resolve_static_sft_training_mode`; add a positive `_validate_stage1_static_packing_policy` test.

### I5. qwen_capture relaxation validates only TOTAL token count, not per-image order *(Lane B; Lane D MINOR)*
After relaxing the `image_grid_thw.shape == (1,3)` check, `CoverageLedgerForwardCaptureResult` still exposes only a single fused `image_embeds` with no per-image boundaries (`src/training/coverage_ledger/qwen_capture.py:17-25`); only the summed token count is validated (`:177-185`). The plan asserts "sidecar order must match the flattened `image_grid_thw` row order" but nothing binds them beyond `sample_id` equality. **Fix:** in the bridge loop, validate each sidecar's `image_grid_thw` row equals the model's `image_grid_thw[segment_index]` (T,H,W); keep `test_packed_coverage_ledger_visual_offsets_pool_from_the_correct_image` mandatory and add a mismatched-order case that must fail.

### I6. Task 2 Step 3 references an undefined, prose-only test — the most safety-critical one *(Lane D; Lane A MINOR)*
Step 3 specifies a test only in prose; Step 4's run command invokes `tests/test_teacher_forcing_sidecar_bridge.py::test_static_packed_collator_preserves_qwen_position_and_media_order`, a name that appears **only** in the plan. That test (4-row position_ids, varlen boundary == segment offsets, media order) is the contract that would catch C4/I5. **Fix:** replace prose with concrete test code using that exact name.

### I7. Metric "contribution == sum" is asserted with no implementation code *(Lane B confirms semantics; Lane D flags missing code)*
`WEIGHTED_LOSS_KEY` (`teacher_forcing/loss/coverage_ledger_auxiliary/contribution`) is emitted as a `last` event (`src/training/coverage_ledger/metrics.py:155-170`; reducer at `src/metrics/events.py:388-389`). Naively concatenating two segment events reports `0.25` (last wins), not the asserted `0.5`. Task 5 Step 3 gives only prose for the fix. **Fix:** specify concrete aggregation — sum `weighted_loss` across segments into one synthesized `last` event; sum count events; recombine weighted-mean numerators/denominators — and ensure the reported number equals the loss actually added (which, per C5, is currently the non-equivalent raw sum).

### I8. Task 8 preflight is asserted to do packed checks it has no code for *(Lane D)*
`run_coverage_ledger_preflight` iterates individual samples and reads one `sidecar.image_grid_thw` each (`src/training/coverage_ledger/preflight.py:156,166`); it has **zero** pack/segment/position_ids logic, and no task modifies `preflight.py` or `scripts/training/coverage_ledger_preflight.py`. The Task 8 Step 5 "two-segment pack materialization / 4-row position_ids / 16 overlays" gate is unimplementable as written. **Fix:** add an explicit task (with tests) to extend the preflight for packed materialization, or downscope Task 8 Step 5 to what the unit/collator tests actually prove and label the gap.

### I9. `sample_id` collision across packs is unhandled *(Lane D)*
`sample_id_to_batch_index` is a flat dict (`src/trainers/metrics/teacher_forcing.py:123`); two packs in one batch sharing a `sample_id` silently overwrite the mapping. All plan tests use a single pack (`[[a,b]]`), so this is never exercised — yet `effective_batch_size: 32` means 32 packed rows/step. **Fix:** assert global `sample_id` uniqueness across all flattened segments in `build_packed_segment_offsets`; add a multi-pack test.

### I10. `effective_batch_size: 32` is only valid for GPU counts that divide 32 *(Lane C)*
`src/config/loader.py:765-778` raises at load if `effective_batch_size % (per_device * world_size) != 0`. 32 works for 4 and 8 GPUs (the cited topologies) but **hard-fails at config load** on e.g. a 6-GPU host. The Task 8 Step 5 probe runs `load_training_config` and would fail there. **Fix:** document the divisibility constraint / make `effective_batch_size` topology-aware. (Note: `load_materialized_training_config`, used by Task 7 tests, does *not* run this block — so the grad-accum claim is only exercised by the Task 8 probe.)

### I11. Field-by-field dataclass reconstruction is correct today but silently drops future fields *(Lane B)*
`shift_coverage_ledger_sidecar`, the `_shifted_sidecar` test helper, and `offset_visual_token_region` rebuild frozen `slots=True` dataclasses field-by-field. All current fields are carried (verified: `src/training/coverage_ledger/sidecars.py:58-69,142-150`; `visual_regions.py:23-27`), but any future field is dropped in the packed path only. **Fix:** use `dataclasses.replace(...)` instead of full rebuilds.

---

## MINOR

- **Task 4 Step 1 snippet lacks `from src.trainers.metrics.teacher_forcing import _build_teacher_forcing_supervision`** → `NameError` as pasted *(Lane A)*.
- **Task 4 Step 2 expected-failure is hedged ("Expected: failure *if* …")** though the test is deterministically red — current code reassigns `batch_index` to the enumerate position, yielding `{"a":0,"b":1}` vs the asserted `{"a":0,"b":0}` *(Lane D)*. Weakens the TDD red contract; state the concrete current behavior.
- **Duplicate `build_packed_segment_offsets` computation** in both enrichers, both overwriting `collated["packed_segment_offsets"]` *(Lane A)*. Compute once and reuse.
- **`sample_index=offset.segment_index` resets per pack**, so multi-pack error messages are ambiguous *(Lane D)*. Pass the global flattened index.
- **`TrainerLossBridge.compute_loss` has no offsets parameter**; cumulative `visual_token_start` relies entirely on sidecar list order (`src/training/bridge/loss_bridge.py:98-108`) *(Lane D; overlaps I5)*.
- **Smoke `effective_batch_size: 32` with a 128-sample cap** (`debug.train_sample_limit: 128`) collapses the overfit-review smoke to ~4 optimizer steps/epoch on 1 GPU and will trip partial-window warnings (`src/sft.py:1426-1463`) *(Lane C)*. Confirm intended.
- **`global_max_length: 12000`** is consistent (parent `compact_support2.yaml:15` already sets `template.max_length: 12000`; loader uses `setdefault`) — **no conflict** *(Lane C, verified)*.

---

## Verified Correct (credit where due)

- **The central visual-token geometry is CORRECT** *(Lane B, verified against upstream modeling)*. `get_image_features` splits per-image post-merge chunks by `image_grid_thw.prod(-1)//merge**2` in grid-row order and concatenates them (`modeling_qwen3_vl.py:1061-1064,1138-1143`); `captured.image_embeds` is a flat `[total_visual_tokens, dim]` in that order (`src/training/coverage_ledger/qwen_capture.py:188-202`). The cumulative base offset `Σ t*(h//merge)*(w//merge)` selects exactly the right image's contiguous slice, and the within-image row-major flatten matches the encoder (`src/training/coverage_ledger/visual_regions.py:122-126`). The merge math is valid because `T==1` is enforced. This is the part most likely to be wrong, and it is right.
- **API/signature grounding is accurate** *(Lane A)*: `SupervisionAtom`/`TeacherForcingTargetIR` fields match the constructors exactly (`src/training/teacher_forcing/ir.py:15-28,45-48`); enricher `__call__(self, *, collated, raw_batch, packed)` and helper names match (`src/data_collators/enrichers.py:223,275,347,356,368`); the real packed `raw_batch` is list-of-packs of per-sample dicts with `sample_id`/`input_ids`/`length` (`src/datasets/wrappers/packed_caption.py:1124-1126`, `src/detection/dataset.py:657,87,470`); `BATCH_EXTRAS_KEYS`/`BatchExtras`/`pop_batch_extras` match the append pattern (`src/trainers/batch_extras.py:22-33,60-76`).
- **Config primitives exist as assumed** *(Lane C)*: `DetectionTrainingConfig.from_mapping` (`schema.py:4840`), `packing.static_packing`/`padding_free_packed` (`:4647-4649`), `TEACHER_FORCING_OBJECTIVE_ID="research_teacher_forcing"` (`src/objective_ids.py:4-6`), `assert_detection_runtime_supported` + its rejections (`src/detection/runtime.py:256,264-293`), `EncodedSampleCacheConfig.ineligible_policy` Literal["error","bypass"] (`schema.py:997,1014-1022`), `packing_mode` is an accepted key (`schema.py:366`). Grad-accum formula is `effective_batch_size/(per_device*world_size)` with divisibility enforcement.
- **Metric reducer hazard is correctly identified** *(Lane B)*: `last`/`sum`/`weighted_mean` semantics are real (`src/metrics/events.py:388-389`; `src/training/coverage_ledger/metrics.py:97-124`), and the plan's warning not to concat `last` events is right.
- **Task 1 is genuinely TDD-clean** *(Lane D)*: concrete red, exact ImportError, full implementation, matching green, narrow commit; the `_is_position_key` vs `*_token_ids` distinction is careful and tested.
- **Stage-2 boundary discipline is consistent** *(Lane D)*: the plan repeatedly refuses to reuse trainer-owned post-rollout packing and routes positive Stage-2 cases away from `_validate_stage1_static_packing_policy`.

---

## Completeness gaps (from Lane D)

| # | Item | Status |
|---|------|--------|
| a | `sample_id` uniqueness/collision across packs | **Silently ignored** (flat dict overwrite; all tests single-pack) — see I9 |
| b | Summed packed loss vs grad-accum / `effective_batch_size` scaling | **Under-verified** — defined as packed rows, but sum-vs-mean equivalence never tested — see C5 |
| c | Packs not filling `global_max_length` (pad / partial last pack) | **Silently ignored** (only `token_cursor <= seq_len` checked) |
| d | Determinism of which samples land in which pack | **Silently ignored** (claimed "deterministic", never pinned/tested) |
| e | Checkpoint/resume & eval-time differences | **Ignored for eval** (`eval_packing: true` shipped, semantics unreconciled) — see C2 |
| f | Rollback if a packed run misbehaves | **Silently ignored** (no documented disable/revert path) |

---

## Recommended remediation order

1. **Resolve C4 first (attention isolation).** If static, non-padding-free packing cannot guarantee block-diagonal masking, the feature is unsound and the rest is moot. Prove it with a numerical leakage test before anything else.
2. **Fix the config-guard blockers (C1, C2)** so the configs can load; add the missing guard + decide `eval_packing` scope. Without this nothing else is testable.
3. **Fix the loss-scale semantics (C5)** and add the packed-vs-unpacked equivalence test; this defines what "batch-equivalent" actually means numerically.
4. **Reorder tasks (C3)** and fix the non-packed regression (I1) and the key registration (I2).
5. **Close the verification gaps** (I6, I7, I8) so the plan's own green/preflight gates are real, not aspirational.
6. Address the remaining IMPORTANT/MINOR items.

---

## What could not be determined

- **ms-swift static (non-padding-free) collator attention mask & per-segment `position_ids`** — `swift` is not importable here and not vendored. C4 hinges on this; must be checked against the actual collator output on a real two-segment pack.
- **Eval path packing consumption** — `eval_packing` does not appear under `src/eval/`; whether eval routes coverage-ledger/teacher-forcing through the same packed bridge is unverified (C2).
- **Exact trainer-level batch normalization for the unpacked baseline** — `ObjectiveRunner` does no batch-dim division; the per-step mean (if any) happens outside the reviewed files, so the precise correct packed normalization for C5 needs that confirmed.

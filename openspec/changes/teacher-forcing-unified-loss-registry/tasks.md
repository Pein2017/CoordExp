## 0. Naming / Scope Hygiene (Optional but Recommended)

- [x] 0.1 Decide a canonical change name that matches the expanded scope (pipeline + unified loss registry + ST bridge).
- [x] 0.2 If renaming, rename: worktree dir, branch, and `openspec/changes/<name>` directory; ensure `openspec list` reflects the updated name.
- [ ] 0.3 Remove backward-compat trainer naming and sync the codebase to a single naming:
  - `custom.trainer_variant: stage2_two_channel` (two-channel Expectation/Rollout)
  - `custom.trainer_variant: stage2_rollout_aligned` (rollout-only)
  - The old strings (`stage2_ab_training`, `rollout_matching_sft`) MUST fail fast with actionable guidance.
- [ ] 0.4 Rename trainer modules + subpackages to match the canonical naming, and update all imports/references:
  - `src/trainers/stage2_ab_training.py` → `src/trainers/stage2_two_channel.py`
  - `src/trainers/rollout_matching_sft.py` → `src/trainers/stage2_rollout_aligned.py`
  - `src/trainers/stage2_ab/` → `src/trainers/stage2_two_channel/`
  - `configs/stage2_ab/` → `configs/stage2_two_channel/`
  - Update: `src/config/schema.py`, `src/config/loader.py`, `configs/**`, `docs/**`, `tests/**`, and any scripts that reference the old module paths.
- [ ] 0.5 Sync all documents to use:
  - “Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout)” for the main Stage-2 engine
  - “Stage-2 Rollout-Aligned Teacher Forcing” for rollout-only mode.

## 1. Unified Loss Registry (Code-Level Internal Contract)

- [ ] 1.1 Implement token-type partitioning utilities (`struct/desc/coord/eos`) with deterministic, packing-safe behavior.
- [ ] 1.2 Implement registry contexts (`gt/self_context/rollout`) and rollout object subsets (`matched/fp/fn`) with FP-neutral + EOS-enforced masking.
- [ ] 1.3 Implement canonical loss components (`struct_ce/desc_ce/coord_token_ce/coord_reg/geo`) and shared helpers (bbox canonicalization + SmoothL1+CIoU).
- [ ] 1.4 Make `full_idea` Channel-B semantics the canonical registry behavior:
  - matched prefix: struct-only CE,
  - FP spans: fully masked out,
  - FN injected: struct+desc CE (default),
  - closure/EOS: supervised.
- [ ] 1.5 Provide explicit ablation knobs to represent rollout-matching supervision variants (e.g., FN `desc` weight = 0,
  matched-prefix struct CE weight = 0) as a registry-backed configuration rather than an ad-hoc code path.

## 2. Straight-Through (ST) Bridge (Enable/Ablate via YAML)

- [ ] 2.1 Add typed YAML keys for ST modes in `src/config/schema.py` under `stage2_ab`:
  - `coord_ctx_embed_mode: soft|st|hard`
  - `coord_decode_mode: exp|st`
  with strict unknown-key fail-fast.
- [ ] 2.2 Implement ST-embedding for Channel-A coord-slot context embeddings (hard forward + soft grads).
- [ ] 2.3 Implement ST coord decode for geometry loss (hard forward + soft grads).
- [ ] 2.4 Add at least one Stage-2 Two-Channel smoke config that enables ST explicitly (to “see the difference” without changing defaults).
- [ ] 2.5 Add rollout-matching SFT typed config for `rollout_matching.coord_decode_mode: exp|st` (ST decode for geo when enabled).

## 3. Config Surface + Schema Contracts (Pipeline)

- [ ] 3.1 Add a typed YAML surface for `stage2_ab.pipeline` (objective + diagnostics module lists) in `src/config/schema.py` with strict unknown-key fail-fast.
- [ ] 3.2 Define module-spec validation rules (required `name`, optional `enabled/weight/channels/config`) and fail fast on duplicates and unknown module names.
- [ ] 3.3 Enforce single-mode precedence: when `stage2_ab.pipeline` is present, disallow objective-affecting flat knobs (fail fast with guidance to move them into module configs).
- [ ] 3.4 Emit a stable pipeline checksum + resolved module list in trainer init logs (include config path, run_name, seed context for reproducibility).
- [ ] 3.5 Add `stage2_ab.text_gate_weight: float` (default `0.0`) to the typed Stage-2 Two-Channel schema so bi-directional gate supervision can be enabled without requiring a pipeline declaration (pipeline mode still disallows flat knobs).

## 3B. Config Surface + Schema Contracts (Rollout-Matching SFT Pipeline)

- [ ] 3B.1 Add a typed YAML surface for rollout-matching SFT pipeline declaration at `rollout_matching.pipeline` (objective + diagnostics module lists) with strict unknown-key fail-fast.
- [ ] 3B.2 Add trainer-variant guardrails:
  - if `custom.trainer_variant=stage2_two_channel`, presence of `rollout_matching.pipeline` fails fast with guidance,
  - if `custom.trainer_variant=stage2_rollout_aligned`, presence of `stage2_ab.pipeline` fails fast with guidance.
- [ ] 3B.3 Emit a stable pipeline checksum + resolved module list in rollout-matching SFT trainer init logs (include config path, run_name, seed context).

## 4. Pipeline Runtime Scaffolding (TeacherForcingContext + Executor)

- [ ] 4.1 Introduce a `TeacherForcingContext` contract that supports multiple registry contexts (`gt/self_context/rollout`) and packing-aware segment meta, and exposes standardized derived views (token-type masks, coord/bbox groups, subset masks) so modules do not depend on trainer-specific raw meta formats.
- [ ] 4.2 Implement a pipeline executor that runs objective modules (loss) and diagnostics modules (metrics) in config order.
- [ ] 4.3 Preserve Stage-2 Two-Channel forward strategy semantics:
  - Channel-A: A1 CE anchoring (`context=gt`) + final self-context logits (`context=self_context`)
  - Channel-B: one-pass rollout-context logits (`context=rollout`)
- [ ] 4.4 Factor shared forward preamble utilities (Qwen3-VL padding-free packing `position_ids` fix, `use_cache=false`, logits slicing guard) into reusable helpers and reuse in Stage-2 Two-Channel and rollout-aligned trainers.
- [ ] 4.5 Unify rollout-context metadata into a single “rich meta” contract that can be produced by both Stage-2 Channel-B and rollout-matching SFT, including (minimum):
  - `prefix_struct_pos` (matched-prefix structure positions),
  - `tail_desc_pos` (tail-local desc value positions for CE weighting),
  - `bbox_groups_prefix` and `bbox_groups_fn` (bbox supervision groups),
  - plus existing `prompt_len/prefix_len/train_len` and coord-supervision lists.
- [ ] 4.6 Refactor Stage-2 Channel-B and rollout-matching SFT batch-prep code to use shared helpers for:
  - suffix-only prefix trimming (append-ready inside `objects[]`),
  - FN append fragment construction,
  - matched-prefix structure position export,
  - and tail desc position export,
  so objective modules do not need trainer-specific parsing logic.

## 5. Objective Modules (Full-Idea Anchored; Variants are Configured)

- [ ] 5.1 Implement `token_ce` objective module using registry masks and `full_idea` semantics:
  - matched prefix: struct-only CE,
  - FP spans: fully masked out,
  - FN injected: struct+desc CE (desc supervised by default),
  - closure/EOS: supervised (EOS-enforced).
- [ ] 5.2 Implement `bbox_geo` objective module using CoordExp decode + bbox canonicalization + SmoothL1 + CIoU:
  - Channel-A (`self_context`): identity-aligned GT objects,
  - Channel-B (`rollout`): matched + FN injected objects (FP excluded).
- [ ] 5.3 Implement `coord_reg` objective module(s) for optional coord regularizers (coord CE, soft-CE, W1, entropy, coord_gate, text_gate, EL1/EHuber), with explicit per-term weights and strict validation.
- [ ] 5.4 Make rollout-matching previous defaults expressible by module config (e.g., FN `desc` weight 0, matched-prefix struct CE off) without forking trainer code.
- [ ] 5.5 Ensure module-emitted metric keys follow the unified registry naming contract.

## 6. Diagnostics Modules + Best-Effort Isolation

- [ ] 6.1 Implement coord diagnostics (e.g., `coord_diag/*`) as a diagnostics module with best-effort exception isolation (warn once, skip/disable on failure).
- [ ] 6.2 Ensure time/packing telemetry remains intact and consistent with existing Stage-2 Two-Channel and rollout-aligned logging conventions.
- [ ] 6.3 Add a minimal diagnostics-only module example that is safe to toggle for ablations without changing the objective.

## 6B. Eval-Step Detection Metrics (COCO mAP)

- [ ] 6B.0 Activate COCO mAP eval by default for Stage-2 trainers:
  - Make `rollout_matching.eval_detection` non-optional in the typed schema (default constructed),
  - Set `rollout_matching.eval_detection.enabled=true` by default (and `metrics=\"coco\"`),
  - Allow disabling explicitly via YAML (`enabled: false`).
- [ ] 6B.1 Make Stage-2 Two-Channel `eval_step` compute the same rollout matching evaluator metrics as Stage-2 Rollout-Aligned, including COCO mAP when `rollout_matching.eval_detection.enabled=true`.
- [ ] 6B.2 Ensure `rollout_matching.eval_detection` behavior is consistent across both Stage-2 trainers (same metric keys, same masking/parse strictness, same failure surfacing).
- [ ] 6B.3 Add an eval-step regression test that asserts `rollout/mAP` is always present when COCO eval is enabled:
  - `rollout/mAP` is a float,
  - `rollout/mAP=0.0` on COCO eval failure (and training continues).
- [ ] 6B.4 Refactor eval-step COCO evaluation to reuse the offline detection evaluator implementation:
  - Prefer extracting a small, public helper in `src/eval/detection.py` that takes in-memory `gt_vs_pred` records and returns COCO metrics/counters,
  - Call that helper from both Stage-2 Two-Channel and Stage-2 Rollout-Aligned trainers (no duplicated COCO prep logic in trainer code).

## 7. Step-Level Aggregation + DDP-Safe Logging

- [ ] 7.1 Extend `_PendingStage2Log` aggregation to accept module-emitted metrics while preserving counter vs gauge semantics (avoid grad-accum dilution for sparse gauge keys).
- [ ] 7.2 Preserve the Stage-2 Two-Channel `Trainer.log()` DDP contract (no rank-conditional collectives) while merging pipeline metrics into the optimizer-step log line.

## 7C. Metrics Normalization + Naming (Make Scalars Comparable)

- [ ] 7C.1 Audit emitted metrics across Stage-1, Stage-2 Two-Channel, and Stage-2 Rollout-Aligned to classify each key as:
  - **mean-like** (scale-invariant; comparable across packing/batch), or
  - **counter-like** (totals; scale with samples/tokens/objects), or
  - **rate-like** (ratio; bounded).
- [ ] 7C.2 Enforce a single naming contract for reduction semantics:
  - `loss/*` keys MUST be mean-like (never raw sums).
  - Counter-like keys MUST use an explicit suffix: `*_total`, `*_count`, `*_sum`, `*_num`, or `*_den`.
  - Mean-like monitor keys SHOULD use `*_mean` (or `*_rate`/`*_frac` for ratios).
  - Internal reduction helpers MUST be prefixed with an underscore (e.g., `rollout/_...`) and MUST be removed from final logs.
- [ ] 7C.3 Normalize rollout monitoring payload keys that are currently counter-like but ambiguous (example targets from `src/trainers/stage2_rollout_aligned.py` payload):
  - `rollout/fn_appended` → `rollout/fn_appended_total` (and optionally add `rollout/fn_appended_per_sample_mean`).
  - `rollout/gt_objects` → `rollout/gt_objects_total` (keep `rollout/gt_per_sample` as mean-like).
  - `rollout/valid_pred_objects` → `rollout/valid_pred_objects_total` (keep `rollout/pred_per_sample`).
  - `rollout/fp` → `rollout/fp_total`, `rollout/fn` → `rollout/fn_total`.
- [ ] 7C.4 Ensure that any sum/count helpers used only for building means (e.g., semantic similarity sums) do not leak into final logs unless explicitly named as counters.
- [ ] 7C.5 Add a metrics-contract test that asserts:
  - every emitted `loss/*` key is mean-like (bounded scale vs token count), and
  - every key ending in `_total|_count|_sum|_num|_den` is reduced as a sum across micro-batches/segments,
  - and underscore-prefixed internal keys are absent from the final log payload.

## 7B. Stage-1 Integration (Static Packing Only)

- [ ] 7B.1 Refactor Stage-1 auxiliary loss mixins to reuse the unified registry + module implementations (no duplicated loss math).
- [ ] 7B.2 Ensure Stage-1 static-only packing policy remains enforced (no dynamic iterable packing) and that the shared registry/pipeline code is packing-safe under dataset-level static packing.
- [ ] 7B.3 (Optional) Introduce a Stage-1 pipeline declaration surface if it reduces MRO/mixin complexity; otherwise keep mixins as thin wrappers over shared modules.

## 8. YAML Migration + Documentation

- [ ] 8.1 Add a Stage-2 Two-Channel smoke config that uses `stage2_ab.pipeline` explicitly while matching default weights/behavior.
- [ ] 8.2 Update `docs/training/METRICS_LOSSES.md` to document:
  - pipeline identity keys (module list + checksum),
  - ST-related diagnostics,
  - and the metrics reduction/naming contract (mean-like `loss/*` vs counter-like `*_total|*_count|*_sum|*_num|*_den`).
- [ ] 8.3 Update/extend the delta specs in this change if the final YAML key names or semantics differ from the draft.
- [ ] 8.4 Update `docs/training/STAGE2_RUNBOOK.md` to:
  - document the config-declared objective pipeline and how it maps to Stage-2 loss terms,
  - make bbox geometry loss (SmoothL1 + CIoU) expectation explicit for both `stage2_rollout_aligned` and `stage2_two_channel`,
  - and document the deterministic Stage-2 Two-Channel scheduler as the canonical router.
- [ ] 8.5 Update `progress/full_idea.md` routing language to match the deterministic schedule used in code (Bresenham on `global_step`), and clarify what “realized b_ratio” means under strict fail-fast vs optional fallbacks.
- [ ] 8.6 Update `docs/training/STAGE2_RUNBOOK.md` and `docs/training/METRICS_LOSSES.md` to document eval-step detection metrics:
  - COCO `mAP` is logged under `rollout/mAP` when `rollout_matching.eval_detection.enabled=true`.

## 9. Verification (Unit Tests + Targeted Smoke)

- [ ] 9.1 Add unit tests for pipeline parsing and validation:
  - unknown module name fails fast,
  - duplicate module name fails fast,
  - channel scoping works (`channels: ["A"]` / `["B"]`),
  - module `config` unknown keys fail fast (per module schema).
- [ ] 9.1B Add a **default-manifest resolution** test:
  - “pipeline omitted” resolves to the Default Pipeline Manifest (module list + ordered names),
  - resolved effective module configs match the manifest defaults / mapped flat keys,
  - resolved pipeline checksum matches a golden string (so implementers cannot drift while remaining “spec compliant”).
- [ ] 9.1C Add **guardrail tests** for single-mode config:
  - if `custom.trainer_variant=stage2_two_channel`, presence of `rollout_matching.pipeline` fails fast,
  - if `custom.trainer_variant=stage2_rollout_aligned`, presence of `stage2_ab.pipeline` fails fast,
  - if a pipeline is provided, presence of disallowed flat objective knobs fails fast (per spec; list the knobs in the test).
- [ ] 9.2 Add parity tests asserting “current monolith” == “implicit pipeline default” for fixed teacher-forced batches (Channel-A and Channel-B; packed and unpacked modes when feasible).
  - MUST include a case with `stage2_ab.desc_ce_weight != 1.0` and assert Channel-B FN `desc` weighting matches the current behavior (tail `desc` tokens use the same weight).
- [ ] 9.3 Add focused tests for ST modes (forward differs vs `soft/exp`, gradients flow through expectation path).
- [ ] 9.3B Add tests for `full_idea` rollout-context masking semantics:
  - matched prefix: `desc` CE masked, struct CE on,
  - FN injected: `desc` CE on by default,
  - FP spans: fully masked out,
  - closure/EOS: supervised.
- [ ] 9.3C Add a minimal Stage-1 regression test for registry-backed dedup:
  - Stage-1 mask building delegates to the unified registry, or
  - Stage-1 registry masks match a frozen synthetic batch expectation (packing-safe).
- [ ] 9.3D Add a minimal metrics-contract test:
  - a short training step emits canonical `loss/<component>` keys, and
  - does not emit any trainer-specific `loss_*` alias keys for registry-defined loss components (canonical-only).
- [ ] 9.4 Run targeted validation commands and record results:
  - `conda run -n ms python -m pytest tests/test_stage2_ab_config_contract.py`
  - `conda run -n ms python -m pytest tests/test_stage2_pending_metrics_aggregation.py`
  - `conda run -n ms python -m pytest tests/test_stage2_two_channel_training.py`
  - `conda run -n ms python -m pytest tests/test_stage2_rollout_aligned.py`

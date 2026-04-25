# Tasks

## 1. OpenSpec and Governance

- [x] Convert all change-local specs to OpenSpec delta form with `ADDED` or `MODIFIED` requirements.
- [x] Add modified deltas for packing, coord-token mode, coord aux, bbox aux, trainer metrics, and encoded-cache eligibility.
- [ ] Run `openspec validate add-stage1-set-continuation-training --strict` once the OpenSpec CLI is available on `PATH`. Blocked during smoke-prep check on 2026-04-25 because `openspec` was not found on `PATH`.
- [x] Add or update canonical docs routing so `progress/directions/full_idea_v5.md` is discoverable through progress routers.

## 2. Strict Config and Setup Routing

- [x] Add strict `custom.stage1_set_continuation` config dataclasses.
- [x] Add a typed strict top-level `benchmark` config section for benchmark
      group identity, comparator, intended variable, and comparability label.
- [x] Add `custom.trainer_variant: stage1_set_continuation` routing before ordinary Stage-1 trainer factory fallback.
- [x] Update trainer class composition so ordinary one-sequence Stage-1 loss mixins are excluded for this variant.
- [x] Add a raw-sample-preserving setup path with `remove_unused_columns=false`.
- [x] Validate subset ratios, prefix order, candidate scoring mode/K, structural-close settings, PEM settings, aux adapter settings, and metric schema version at config resolution.
- [x] Require effective coord-token rendering and reject raw-text integer coordinate mode.
- [x] Fail fast when `training.packing=true` before static pack-plan train or eval datasets can be built.
- [x] Fail fast or bypass encoded-sample cache according to `training.encoded_sample_cache.ineligible_policy`.
- [x] Fail fast when an enabled auxiliary loss lacks a set-continuation branch adapter.

## 3. Metadata, Collator, and Branch Encoder

- [x] Preserve `assistant_payload`, `messages` or equivalent image/prompt identity, `metadata`, `sample_id`, `base_idx`, and dataset label through the collator.
- [x] Use `assistant_payload.objects` as the canonical object-entry source for subset/candidate sampling.
- [x] Add a regression test proving `sample["objects"]` is not assumed to contain serialized object entries.
- [x] Implement `Stage1SetContinuationBranchEncoder` or equivalent helper for branch construction.
- [x] Ensure the branch encoder wraps template state, disables packing/padding-free assumptions, preserves image inputs, and runs existing image-token/grid contract checks in smoke tests.
- [x] Ensure non-model extras are stripped before `model(**inputs)`.

## 4. Serialization and Structural Spans

- [x] Implement indexed canonical object-entry fragment extraction from the existing CoordJSON assistant serialization path.
- [x] Avoid content-based substring matching for object spans.
- [x] Include candidate entry delimiter and object-entry terminator in candidate labels.
- [x] Exclude global schema closure, assistant suffix, EOS, and chat-template stop tokens from candidate-entry scores.
- [x] Implement canonical structural-close span extraction for close-start and close-sequence supervision.
- [x] Test empty-prefix, non-empty-prefix, full-prefix, first/middle/last candidate, duplicate identical entries, same-desc different-bbox entries, and supported object field orders.
- [x] Test that `<|im_end|>`, `<|end_of_text|>`, EOS, and object-entry close tokens are excluded from structural-close targets.

## 5. Subset and Candidate Sampling

- [x] Implement empty-prefix, random-subset, leave-one-out, and small full-prefix sampling.
- [x] Implement random prefix ordering by default, plus a preserved dataset-order ablation mode.
- [x] Define sampler RNG as a pure function of resolved seed, epoch, sample identity, rank, and documented microstep salt.
- [x] Implement deterministic small-object-count fallbacks for `|O| = 0`, `|O| = 1`, and invalid mode selections.
- [x] Implement exact all-remaining candidate scoring selection.
- [x] Implement optional uniform candidate subsampling with `K`.
- [x] Log configured and resolved valid-mode mixtures.

## 6. Losses and Objective Semantics

- [x] Implement coord-aware full-entry candidate scoring with full-vocab non-coord logprobs and coord-vocab-normalized coord-token logprobs.
- [x] Implement `mp/logZ_scored_raw`, `mp/logZ_remaining_exact`, and `mp/logZ_remaining_est`.
- [x] Implement `loss/mp` from exact or raw scored-candidate logZ according to candidate mode.
- [x] Implement responsibility statistics over scored candidates.
- [x] Implement PEM threshold-loss objective in v1: `positive_evidence_margin.objective=threshold_loss`.
- [x] Require exactly one of `rho` or `log_rho` when PEM threshold loss is enabled.
- [x] Ensure PEM uses exact logZ or uniform-importance estimated logZ, not raw sampled logZ.
- [x] Log `loss/mp_diagnostic` when PEM threshold loss optimizes `loss/pem`.
- [x] Implement `loss/anti_close_start` for `R != empty`.
- [x] Implement `loss/weak_schema_close` for `R = empty`.
- [x] Exclude `R = empty` and zero-weight full-prefix metric-only samples from MP objective denominators.
- [x] Implement branch-local `coord_soft_ce_w1` auxiliary adapter.
- [x] Implement branch-local `bbox_geo` auxiliary adapter.
- [x] Implement branch-local `bbox_size_aux` auxiliary adapter.
- [x] Aggregate branch-local aux as mean-like candidate atoms uniformly over scored valid candidates.
- [x] Keep responsibility-weighted aux out of v1 unless added as a separately named future mode.

## 7. Metrics, Artifacts, and Benchmark Profiles

- [x] Add variant-specific metric documentation and metric-key parity tests.
- [x] Log all required `loss/*`, `mp/*`, `stop/*`, and `aux/*` keys from the set-continuation spec.
- [x] Log candidate entry token counts, logprob sums, per-token logprobs, coord/non-coord diagnostics, responsibility-vs-length diagnostics, and small-n validity counters.
- [x] Log repeated-forward budget metrics and scored-candidate fractions.
- [x] Preserve ordinary Stage-1 metric parity and prove ordinary SFT does not emit set-continuation MP keys.
- [x] Add artifact checks for `resolved_config.json`, `effective_runtime.json`, `experiment_manifest.json`, and existing run-manifest surfaces.
- [x] Record set-continuation metric schema version in resolved config or run manifest.
- [x] Add a single checked-in production profile with stable `benchmark.group_id` and `benchmark.control_group_id`.
- [x] Add benchmark manifest validation that records intended variable, comparator, eval scope/view, sample count, checkpoint/base identity, inference decoding controls, and comparability label.
- [x] Add benchmark manifest validation that records effective coord-slot scoring, aux objective settings, PEM threshold calibration provenance, realized prefix-mode coverage, and realized branch/token budget.
- [x] Add sparse-label FP caveat/report fields for proxy or partial-label eval views.

## 8. Documentation

- [x] Update `docs/training/STAGE1_OBJECTIVE.md` with a Stage-1 set-continuation section.
- [x] Update `docs/training/METRICS.md` with variant-specific MP/PEM/structural-close metrics.
- [x] Update `docs/data/PACKING.md` with v1 fail-fast behavior for `stage1_set_continuation`.
- [x] Update encoded-cache docs/spec references with metadata-only eligibility or bypass behavior.
- [x] Update `docs/training/README.md` and `docs/IMPLEMENTATION_MAP.md` with the new trainer variant and setup-path fork.
- [x] Document that v1 uses repeated independent forwards, not prefix sharing or branch masks.
- [x] Document that v1 is coord-token-only and rejects raw-text integer coordinate training.
- [x] Document the production entry config and the required benchmark manifest/report fields.

## 9. Verification

- [x] Add unit tests for strict config parsing and unknown/invalid nested keys.
- [x] Add fail-fast tests for raw-text coordinates, packing, encoded-cache policy, and missing auxiliary adapters.
- [x] Add tests proving ordinary one-sequence Stage-1 mixins are not composed for the variant.
- [x] Add collator/batch-extra tests proving set-continuation metadata survives collation and is stripped before model forward.
- [x] Add unit tests for fragment serialization, object-index spans, duplicate entries, and token-span boundaries.
- [x] Add unit tests for structural close-start and close-sequence token spans.
- [x] Add unit tests for subset sampling ratios, deterministic seeded behavior, small-object-count fallback, and random/preserved prefix order modes.
- [x] Add unit tests for candidate subsampling, exact mode, raw/corrected logZ, and PEM calibration.
- [x] Add unit tests for MP logsumexp, PEM threshold-loss total loss, close-start suppression, and weak schema close.
- [x] Add branch-local auxiliary adapter tests and missing-state failure tests.
- [x] Add a tiny trainer smoke test with one image and at least two objects.
- [x] Add config-resolution tests for the production entry config and its all-feature contract.
- [x] Run the focused pytest command listed in `docs/superpowers/plans/2026-04-25-stage1-set-continuation-training.md`.

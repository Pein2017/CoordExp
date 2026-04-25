# Tasks

## 1. OpenSpec and Governance

- [ ] Convert all change-local specs to OpenSpec delta form with `ADDED` or `MODIFIED` requirements.
- [ ] Add modified deltas for packing, coord-token mode, coord aux, bbox aux, trainer metrics, and encoded-cache eligibility.
- [ ] Run `openspec validate add-stage1-set-continuation-training --strict` once the OpenSpec CLI is available on `PATH`.
- [ ] Add or update canonical docs routing so `progress/directions/full_idea_v5.md` is discoverable through progress routers.

## 2. Strict Config and Setup Routing

- [ ] Add strict `custom.stage1_set_continuation` config dataclasses.
- [ ] Add a typed strict top-level `benchmark` config section for benchmark
      group identity, comparator, intended variable, and comparability label.
- [ ] Add `custom.trainer_variant: stage1_set_continuation` routing before ordinary Stage-1 trainer factory fallback.
- [ ] Update trainer class composition so ordinary one-sequence Stage-1 loss mixins are excluded for this variant.
- [ ] Add a raw-sample-preserving setup path with `remove_unused_columns=false`.
- [ ] Validate subset ratios, prefix order, candidate scoring mode/K, structural-close settings, PEM settings, aux adapter settings, and metric schema version at config resolution.
- [ ] Require effective coord-token rendering and reject raw-text integer coordinate mode.
- [ ] Fail fast when `training.packing=true` before static pack-plan train or eval datasets can be built.
- [ ] Fail fast or bypass encoded-sample cache according to `training.encoded_sample_cache.ineligible_policy`.
- [ ] Fail fast when an enabled auxiliary loss lacks a set-continuation branch adapter.

## 3. Metadata, Collator, and Branch Encoder

- [ ] Preserve `assistant_payload`, `messages` or equivalent image/prompt identity, `metadata`, `sample_id`, `base_idx`, and dataset label through the collator.
- [ ] Use `assistant_payload.objects` as the canonical object-entry source for subset/candidate sampling.
- [ ] Add a regression test proving `sample["objects"]` is not assumed to contain serialized object entries.
- [ ] Implement `Stage1SetContinuationBranchEncoder` or equivalent helper for branch construction.
- [ ] Ensure the branch encoder wraps template state, disables packing/padding-free assumptions, preserves image inputs, and runs existing image-token/grid contract checks in smoke tests.
- [ ] Ensure non-model extras are stripped before `model(**inputs)`.

## 4. Serialization and Structural Spans

- [ ] Implement indexed canonical object-entry fragment extraction from the existing CoordJSON assistant serialization path.
- [ ] Avoid content-based substring matching for object spans.
- [ ] Include candidate entry delimiter and object-entry terminator in candidate labels.
- [ ] Exclude global schema closure, assistant suffix, EOS, and chat-template stop tokens from candidate-entry scores.
- [ ] Implement canonical structural-close span extraction for close-start and close-sequence supervision.
- [ ] Test empty-prefix, non-empty-prefix, full-prefix, first/middle/last candidate, duplicate identical entries, same-desc different-bbox entries, and supported object field orders.
- [ ] Test that `<|im_end|>`, `<|end_of_text|>`, EOS, and object-entry close tokens are excluded from structural-close targets.

## 5. Subset and Candidate Sampling

- [ ] Implement empty-prefix, random-subset, leave-one-out, and small full-prefix sampling.
- [ ] Implement random prefix ordering by default, plus preserved/canonical order ablation modes.
- [ ] Define sampler RNG as a pure function of resolved seed, epoch, sample identity, rank, and documented microstep salt.
- [ ] Implement deterministic small-object-count fallbacks for `|O| = 0`, `|O| = 1`, and invalid mode selections.
- [ ] Implement exact all-remaining candidate scoring selection.
- [ ] Implement optional uniform candidate subsampling with `K`.
- [ ] Log configured and resolved valid-mode mixtures.

## 6. Losses and Objective Semantics

- [ ] Implement coord-aware full-entry candidate scoring with full-vocab non-coord logprobs and coord-vocab-normalized coord-token logprobs.
- [ ] Implement `mp/logZ_scored_raw`, `mp/logZ_remaining_exact`, and `mp/logZ_remaining_est`.
- [ ] Implement `loss/mp` from exact or raw scored-candidate logZ according to candidate mode.
- [ ] Implement responsibility statistics over scored candidates.
- [ ] Implement PEM replacement mode in v1: `positive_evidence_margin.mode=replace_mp`.
- [ ] Require exactly one of `rho` or `log_rho` when PEM replacement mode is enabled.
- [ ] Ensure PEM uses exact logZ or uniform-importance estimated logZ, not raw sampled logZ.
- [ ] Log `loss/mp_diagnostic` when PEM replacement mode optimizes `loss/pem`.
- [ ] Implement `loss/anti_close_start` for `R != empty`.
- [ ] Implement `loss/weak_schema_close` for `R = empty`.
- [ ] Exclude `R = empty` and zero-weight full-prefix metric-only samples from MP objective denominators.
- [ ] Implement branch-local `coord_soft_ce_w1` auxiliary adapter.
- [ ] Implement branch-local `bbox_geo` auxiliary adapter if required decoded bbox state is available.
- [ ] Implement branch-local `bbox_size_aux` auxiliary adapter if `bbox_geo` state is available.
- [ ] Aggregate branch-local aux as mean-like candidate atoms uniformly over scored valid candidates.
- [ ] Keep responsibility-weighted aux out of v1 unless added as a separately named future mode.

## 7. Metrics, Artifacts, and Benchmark Profiles

- [ ] Add variant-specific metric documentation and metric-key parity tests.
- [ ] Log all required `loss/*`, `mp/*`, `stop/*`, and `aux/*` keys from the set-continuation spec.
- [ ] Log candidate entry token counts, logprob sums, per-token logprobs, coord/non-coord diagnostics, responsibility-vs-length diagnostics, and small-n validity counters.
- [ ] Log repeated-forward budget metrics and scored-candidate fractions.
- [ ] Preserve ordinary Stage-1 metric parity and prove ordinary SFT does not emit set-continuation MP keys.
- [ ] Add artifact checks for `resolved_config.json`, `effective_runtime.json` or `pipeline_manifest.json`, `experiment_manifest.json`, `run_metadata.json`, and data-provenance sidecars.
- [ ] Record set-continuation metric schema version in resolved config or run manifest.
- [ ] Add checked-in or generated benchmark profiles for Groups A-F with stable `benchmark.group_id` and `benchmark.control_group_id`.
- [ ] Add benchmark manifest validation that records intended variable, comparator, eval scope/view, sample count, prediction volume, checkpoint/base/adapter identity, inference decoding controls, and comparability label.
- [ ] Add benchmark manifest validation that records effective coord-slot scoring, aux objective settings, PEM threshold calibration provenance, realized prefix-mode coverage, and realized branch/token budget.
- [ ] Add sparse-label FP caveat/report fields for proxy or partial-label eval views.

## 8. Documentation

- [ ] Update `docs/training/STAGE1_OBJECTIVE.md` with a Stage-1 set-continuation section.
- [ ] Update `docs/training/METRICS.md` with variant-specific MP/PEM/structural-close metrics.
- [ ] Update `docs/data/PACKING.md` with v1 fail-fast behavior for `stage1_set_continuation`.
- [ ] Update encoded-cache docs/spec references with metadata-only eligibility or bypass behavior.
- [ ] Update `docs/training/README.md` and `docs/IMPLEMENTATION_MAP.md` with the new trainer variant and setup-path fork.
- [ ] Document that v1 uses repeated independent forwards, not prefix sharing or branch masks.
- [ ] Document that v1 is coord-token-only and rejects raw-text integer coordinate training.
- [ ] Document benchmark Groups A-F and the required benchmark manifest/report fields.

## 9. Verification

- [ ] Add unit tests for strict config parsing and unknown/invalid nested keys.
- [ ] Add fail-fast tests for raw-text coordinates, packing, encoded-cache policy, and missing auxiliary adapters.
- [ ] Add tests proving ordinary one-sequence Stage-1 mixins are not composed for the variant.
- [ ] Add collator/batch-extra tests proving set-continuation metadata survives collation and is stripped before model forward.
- [ ] Add unit tests for fragment serialization, object-index spans, duplicate entries, and token-span boundaries.
- [ ] Add unit tests for structural close-start and close-sequence token spans.
- [ ] Add unit tests for subset sampling ratios, deterministic seeded behavior, small-object-count fallback, and random/preserved prefix order modes.
- [ ] Add unit tests for candidate subsampling, exact mode, raw/corrected logZ, and PEM calibration.
- [ ] Add unit tests for MP logsumexp, PEM replacement-mode total loss, anti-close, and weak schema close.
- [ ] Add branch-local auxiliary adapter tests and missing-state failure tests.
- [ ] Add a tiny trainer smoke test with one image and at least two objects.
- [ ] Add config-resolution tests for Groups A-F and a controlled-diff benchmark matrix check.
- [ ] Run the focused pytest command listed in `docs/superpowers/plans/2026-04-25-stage1-set-continuation-training.md`.

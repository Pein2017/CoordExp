## 1. OpenSpec Foundation

- [x] 1.1 Keep the proposal aligned with the final v1 scope:
  - opt-in arbitrary-`K` support when enabled
  - legacy `K=2` compatibility unchanged when disabled
  - default pseudo-positive profile uses `K=4`
  - pseudo-positive remains coord-positive, desc-neutral, and shares the global rollout-prefix structure CE surface
  - duplicate-like dead suppression only
  - support-rate semantics rather than fixed support-count buckets
  - minimum promotion evidence of `support_count >= 2`
- [x] 1.2 Keep the delta spec set in sync for:
  - `channel-b-lightweight-pseudopositive-v1`
  - `stage2-ab-training`
  - `teacher-forcing-unified-loss-registry`
  - `trainer-metrics-components`
- [x] 1.3 Re-run change validation before implementation starts:
  - `openspec validate study-channel-b-pseudopositive-promotion --type change --strict --json --no-interactive`

## 2. Config Surface And Contract Guardrails

- [x] 2.1 Extend `src/config/schema.py` so `stage2_ab.channel_b` accepts the typed `pseudo_positive` mapping with exactly:
  - `enabled`
  - `coord_weight`
- [x] 2.2 Enforce config invariants:
  - `enabled` defaults to `false`
  - `coord_weight` defaults to `0.5`
  - `0.0 < coord_weight < 1.0`
- [x] 2.3 When `pseudo_positive.enabled=true`, require `stage2_ab.channel_b.triage_posterior.num_rollouts >= 2`.
- [x] 2.4 Preserve legacy config behavior when `pseudo_positive.enabled=false` and keep enabled `K=2` available as an explicit no-promotion control.
- [x] 2.5 Add config-contract tests covering:
  - unknown nested keys fail fast
  - invalid pseudo-positive weight fails fast
  - enabled + `num_rollouts < 2` fails fast
  - disabled path preserves legacy `K=2`
  - versioned `v*` knob aliases are rejected

## 3. Arbitrary-K Rollout Preparation

- [x] 3.1 Extend Channel-B rollout preparation in `src/trainers/stage2_two_channel.py` to request:
  - `1` greedy anchor rollout
  - `num_rollouts - 1` explorer rollouts
- [x] 3.2 Ensure all explorers share the same authored decode profile and differ only by deterministic per-explorer stochastic identity derived from rollout seed base + explorer ordinal.
- [x] 3.3 Reuse the existing per-view accepted-clean preparation path independently for all views:
  - bounded salvage
  - strict acceptance
  - bbox-valid filtering
  - sequential dedup
  - Hungarian matching
- [x] 3.4 When `pseudo_positive.enabled=true`, drop malformed anchor-preparation samples instead of using the canonical empty-prefix fallback.
- [x] 3.5 Treat malformed/missing explorer preparation as a hard error that aborts the current optimizer step.
- [x] 3.6 Keep zero accepted-clean explorer output valid and counted as zero support.
- [x] 3.7 Add trainer/unit coverage for:
  - `1 + (K-1)` rollout scheduling
  - deterministic per-explorer identity
  - drop-on-malformed-anchor behavior
  - hard abort on failed explorer prep
  - at least two enabled `num_rollouts` values
  - mean-over-valid-explorer-view semantics for legacy `rollout/explorer/*` metrics under arbitrary `K`

## 4. Support Counting, Support Rates, Recovered GT, And Bucket Assignment

- [x] 4.1 Extend `src/trainers/stage2_two_channel/target_builder.py` to associate each explorer view to anchor objects independently using the existing deterministic one-to-one IoU rule.
- [x] 4.2 Compute per-anchor `explorer_support_count` and `explorer_support_rate` using the v1 gate:
  - anchor unmatched
  - explorer counterpart exists
  - explorer counterpart unmatched
  - no GT-backed anchor conflict
- [x] 4.3 Keep the GT-backed conflict rule geometry-first using `unlabeled_consistent_iou_threshold`.
- [x] 4.4 Detect `recovered_fn` with any-hit semantics across explorers:
  - anchor misses GT
  - at least one explorer matches GT
  - multiple explorer hits collapse to one recovered FN object
- [x] 4.5 Record per-recovered-GT support counts and support rates in per-sample triage metadata.
- [x] 4.6 Map support rates to buckets:
  - `support_count = 0` -> `dead_anchor`
  - `support_count > 0` but failing promotion thresholds -> `shielded_anchor`
  - `support_count >= 2` and `support_rate >= 2/3` -> pseudo-positive candidate
- [x] 4.7 Add unit coverage for:
  - support-rate bucket assignment across multiple `K` values
  - GT-conflict exclusion
  - recovered-FN collapse across multiple explorers
  - exact rate numerator / denominator computation
  - no pseudo-positive promotion at enabled `K=2`

## 5. Anchor Overlap Clustering And Pseudo-Positive Promotion

- [x] 5.1 Build anchor-side pseudo-positive candidate overlap clusters as connected components of the undirected graph with edges at `IoU >= duplicate_iou_threshold`.
- [x] 5.2 Select exactly one pseudo-positive winner per connected component:
  - highest support rate wins
  - ties break by earlier anchor order
- [x] 5.3 Demote all non-winning component members to `shielded_anchor`.
- [x] 5.4 Keep selected pseudo-positive objects in the final edited anchor prefix in anchor order.
- [x] 5.5 Thread `coord_weight` through the existing per-bbox-group weight carrier.
- [x] 5.6 Pin pseudo-positive bbox/coord targets to the selected anchor object's own canonical coordinates.
- [x] 5.7 Add unit coverage for:
  - connected-component clustering
  - tie-breaking by support rate then anchor order
  - non-winning pseudo-positive candidates falling back to shielded

## 6. Loss Projection Under One Forward

- [x] 6.1 Preserve one clean edited-target teacher-forced forward for the entire Channel-B sample.
- [x] 6.2 Keep bucketed loss application explicit:
  - retained prefix objects share one global rollout-prefix structure CE surface
  - `matched_clean` -> coord + global rollout-prefix structure CE
  - `fn_injection` -> coord + FN desc CE
  - `pseudo_positive` -> coord + global rollout-prefix structure CE
  - `shielded_anchor` -> global rollout-prefix structure CE only
  - `dead_anchor` -> no positive supervision
- [x] 6.3 Ensure `coord_weight` scales only:
  - `bbox_geo`
  - `coord_reg`
  - `bbox_size_aux`
- [x] 6.4 Ensure pseudo-positive objects do not create:
  - desc CE
  - pseudo-positive-only text supervision branches
- [x] 6.5 Keep ordinary matched-clean and FN-injection behavior unchanged apart from the new pseudo-positive groups.
- [x] 6.6 Add objective-path tests proving:
  - one-forward realization only
  - pseudo-positive groups use anchor-owned target geometry
  - pseudo-positive weights hit only coord-side modules
  - pseudo-positive text-side supervision uses only the shared global rollout-prefix structure CE surface

## 7. Dead-Anchor Filtering And Suppression

- [x] 7.1 Keep all dead anchors out of the final edited target.
- [x] 7.2 Define duplicate-like dead anchors operationally in `target_builder.py` using:
  - same local continuation boundary group
  - overlap to an earlier kept anchor at `IoU >= duplicate_iou_threshold`
  - same normalized desc under the existing duplicate-style normalization rule
- [x] 7.3 Only duplicate-like dead anchors may create `dead_anchor_suppression_targets`.
- [x] 7.4 Non-duplicate dead anchors must be dropped without explicit suppression.
- [x] 7.5 Preserve the existing first-divergence suppression mechanism in `loss_dead_anchor_suppression.py`.
- [x] 7.6 Add unit coverage for:
  - boundary-local duplicate-like detection
  - non-duplicate dead anchors producing no suppression targets
  - duplicate-like dead branches consuming the same clean-forward logits

## 8. Metadata, Metrics, And Monitoring

- [x] 8.1 Extend per-sample Channel-B triage metadata to include:
  - `valid_explorer_count`
  - per-anchor explorer support counts
  - per-anchor explorer support rates
  - `pseudo_positive_anchor_indices`
  - `dead_explorer_indices_by_view`
  - per-recovered-GT explorer support counts
  - per-recovered-GT explorer support rates
- [x] 8.2 Add aggregate metrics/counters for:
  - `train/triage/pseudo_positive_candidate_count`
  - `train/triage/pseudo_positive_subthreshold_count`
  - `train/triage/pseudo_positive_selected_count`
  - `train/triage/pseudo_positive_cluster_demoted_count`
  - `train/triage/anchor_preparation_dropped_count`
  - `train/triage/pseudo_positive_support_rate_num`
  - `train/triage/pseudo_positive_support_rate_den`
  - `train/triage/pseudo_positive_selected_support_rate_num`
  - `train/triage/pseudo_positive_selected_support_rate_den`
  - `train/triage/recovered_ground_truth_rate_num`
  - `train/triage/recovered_ground_truth_rate_den`
- [x] 8.3 Preserve compatibility semantics:
  - `train/triage/unlabeled_consistent_count` remains the total shielded-anchor count
  - singular explorer-local metadata is replaced with explorer-indexed carriers
- [x] 8.4 Keep monitor payloads optional, but mirror the metadata fields there when monitoring is enabled.
- [x] 8.5 Add regression coverage that these observability fields appear whenever pseudo-positive is enabled and that legacy explorer metrics retain the documented aggregate meaning.
- [x] 8.6 Ensure explorer-preparation aborts are recorded through failure telemetry or ablation reporting rather than as ordinary finalized step metrics.
- [x] 8.7 Update canonical docs after implementation:
  - `docs/training/METRICS.md`
  - `docs/training/STAGE2_RUNBOOK.md`

## 9. YAML Profiles And Smoke Validation

- [x] 9.1 Add or update an explicit Stage-2 YAML profile under `configs/stage2_two_channel/` that authors:
  - `pseudo_positive.enabled`
  - `pseudo_positive.coord_weight`
  - `triage_posterior.num_rollouts: 4`
- [x] 9.2 Ensure the default `K=4` profile can be ablated by varying `triage_posterior.num_rollouts` without renaming any knobs.
- [x] 9.3 Materialize the selected profile through the normal config loader and confirm the typed config contract is realized as expected.
- [x] 9.4 Run targeted validation after implementation:
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - any targeted objective-pipeline tests needed for pseudo-positive bbox groups and dead-anchor suppression
- [x] 9.5 Run a short Stage-2 smoke that exercises the enabled pseudo-positive Channel-B path and verifies:
  - `1 + (K-1)` rollout execution
  - non-zero support-rate accounting
  - pseudo-positive groups appear in coord supervision only
  - dead-anchor suppression targets exist only for duplicate-like dead anchors
  - no second teacher-forced forward is introduced
  - at least one two-point `best-K` ablation emits comparable rate-based observability
  - recovered-GT rate metrics are emitted alongside pseudo-positive metrics
  - explorer-preparation abort telemetry remains covered by regression and failure-path verification because successful smoke does not trigger the hard-abort path by design
  - Verified by successful launcher-managed runs captured in `.planning/phases/05-validation-and-best-k-readiness/05-02-SUMMARY.md` and `.planning/phases/05-validation-and-best-k-readiness/05-03-SUMMARY.md`.

## 10. Post-Implementation Review

- [x] 10.1 Re-validate the OpenSpec change after implementation:
  - `openspec validate study-channel-b-pseudopositive-promotion --type change --strict --json --no-interactive`
- [x] 10.2 Review inference feedback on the first enabled run for the stop-signal metrics:
  - hallucination rate
  - duplicate burst rate
  - enumeration-style overproduction
  - oversized / entangled boxes
  - dense-scene recall
  - Smoke-level qualitative review completed from the enabled `K=4` monitor dump: zero duplicate bursts, atomic `clock` / `vase` boxes instead of an oversized group box, and no obvious enumeration-style collapse. This is smoke evidence rather than a long-run eval.
- [x] 10.3 Compare at least two enabled `num_rollouts` settings using the rate-based triage metrics before broadening pseudo-positive semantics.
  - Completed with the committed `K=4` default smoke and the committed enabled `K=2` control smoke. `K=4` emitted non-zero pseudo-positive winners, while enabled `K=2` emitted zero winners and retained comparable rate-based observability.
- [x] 10.4 Verify that any enabled `K=2` run behaves as a no-promotion control under the `support_count >= 2` floor.
  - Completed at runtime in `.planning/phases/05-validation-and-best-k-readiness/05-03-SUMMARY.md`: enabled `K=2` produced `pseudo_positive_selected_count=0.0` while still exercising Channel-B.
- [x] 10.5 Do not widen pseudo-positive semantics, semantic gating, or dead-negative scope unless the first implementation slice is stable under those feedback checks.
